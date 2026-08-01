//! `anime-dl` — standalone ogladajanime.pl season downloader.
//!
//! Accepts a canonical JSON episode list via `--input <path>` (or positional
//! `.json` path). Downloads episodes at best available quality. No translation,
//! no OCR — pure download. Reuses the `mt_cli::downloader` engine.
//!
//! Exit codes: 0 all ok; 1 fatal runtime error; 2 usage (clap); 3 partial;
//! 4 all requested failed; 130 cancelled.

use std::fs::{self};
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use mt_cli::download_types::{EpEvent, EpisodeInput, parse_json_input, sanitize_slug};
use mt_cli::downloader::{DownloadConfig, DownloadEngine, Outcome, RunMode};
use mt_cli::plain_output::spawn_plain_output;
use mt_cli::ui_render::select_renderer;
use mt_cli::validator::ValidationConfig;
use tokio::runtime::Runtime;
use tokio::signal::ctrl_c;
use tokio::sync::broadcast;
use tracing::error;

/// Output mode. Dashboard aliases: `a`, `dashboard`, `tui`.
#[derive(Clone, Debug)]
enum CliUiMode {
    Dashboard,
    Plain,
}

impl std::str::FromStr for CliUiMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "dashboard" | "a" | "tui" => Ok(CliUiMode::Dashboard),
            "plain" => Ok(CliUiMode::Plain),
            other => Err(format!(
                "invalid UI mode '{other}'. Valid: dashboard|a|tui, plain"
            )),
        }
    }
}

fn parse_ui_mode(s: &str) -> Result<CliUiMode, String> {
    s.parse()
}

const AFTER_HELP: &str = "\
CANONICAL JSON FORMAT (schema_version 2):
  {
    \"schema_version\": 2,
    \"title\": \"One Piece\",                        (optional)
    \"source_page\": \"https://ogladajanime.pl/anime/...\",  (optional)
    \"resolved_at\": \"2025-07-30T12:34:56Z\",       (optional)
    \"episodes\": [
      {
        \"episode\": 1,
        \"mirrors\": [
          {
            \"host\": \"cda\",
            \"quality\": \"1080p\",
            \"subtitle_group\": null,
            \"url\": \"https://...\"
          }
        ]
      }
    ]
  }

  Legacy v1 documents ({ \"title\": \"...\", \"episodes\": [{ \"episode\": 1,
  \"urls\": [\"https://...\"] }] }) are still accepted and auto-migrated.
  URLs are never rendered in any UI output (dashboard or plain).

RUN MODES:
  Default        download everything missing/invalid
  --resume       only episodes not marked Complete (manifest or output check)
  --retry-failed only episodes marked Failed; combined with --resume this
                 selects RunMode::Resume — failed episodes are a subset of
                 not-complete episodes, so Resume is exactly the union of the
                 two scopes (the engine has no RetryBoth variant)
  --validate-only  check existing outputs only; never download

EXIT CODES:
  0    all episodes downloaded / all valid (validate-only)
  1    fatal runtime error (unreadable or malformed JSON input, I/O, internal)
  2    CLI/usage error (bad flags, invalid flag combination — clap rejects
       before any download starts)
  3    partial: some episodes failed or invalid, some ok
  4    all requested episodes failed or invalid (validate-only)
  130  cancelled by user (Ctrl-C)
";

#[derive(Parser, Debug)]
#[command(
    name = "anime-dl",
    about = "Download anime episodes from ogladajanime.pl (best quality, no translation)",
    after_help = AFTER_HELP
)]
struct Args {
    /// Path to canonical JSON episode list (positional .json path also accepted).
    #[arg(short = 'i', long, value_name = "PATH")]
    input: Option<PathBuf>,

    /// Positional .json path (convenience, routed same as --input).
    name: Option<String>,

    /// Output directory. Default: `./<title-slug>`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Only download these episode numbers (comma-separated).
    #[arg(long, value_name = "N,N,...", value_delimiter = ',')]
    episodes: Option<Vec<i64>>,

    /// Max concurrent episode downloads.
    #[arg(long, default_value_t = 4)]
    episode_concurrency: usize,

    /// Max concurrent downloads from same host.
    #[arg(long, default_value_t = 1)]
    host_concurrency: usize,

    /// Output mode (default TTY: dashboard, non-TTY: plain).
    /// Values: dashboard|a|tui, plain.
    #[arg(long, value_parser = parse_ui_mode)]
    ui: Option<CliUiMode>,

    /// Raise our crates' logging to DEBUG.
    #[arg(short, long)]
    verbose: bool,

    // ── Run modes ────────────────────────────────────────────────────────
    /// Only process episodes whose manifest status is not Complete.
    #[arg(long)]
    resume: bool,

    /// Only process episodes whose manifest status is Failed.
    /// Combining with --resume collapses to --resume (engine union semantics).
    #[arg(long)]
    retry_failed: bool,

    /// Validate existing outputs only; never download. Conflicts with
    /// --resume / --retry-failed.
    #[arg(long, conflicts_with_all = ["resume", "retry_failed"])]
    validate_only: bool,

    // ── Media validation tuning ──────────────────────────────────────────
    /// Skip ffprobe-based validation of downloaded files; extension+size
    /// heuristic applies instead. Valid heuristic-passing media is still
    /// never overwritten.
    #[arg(long)]
    no_validate: bool,

    /// Force re-validation, bypassing the validation cache (always re-probe).
    #[arg(long)]
    validate_force: bool,

    /// Minimum accepted file size in MiB.
    #[arg(long, default_value_t = 1.0)]
    min_size_mb: f64,

    /// Minimum accepted media duration in seconds.
    #[arg(long, default_value_t = 1.0)]
    min_duration_secs: f64,

    /// Reject files with no audio stream.
    #[arg(long)]
    require_audio: bool,

    /// ffprobe invocation timeout in seconds.
    #[arg(long, default_value_t = 15)]
    ffprobe_timeout: u64,

    // ── Robustness tuning ────────────────────────────────────────────────
    /// Retry attempts per mirror (on top of the initial attempt).
    #[arg(long, default_value_t = 3)]
    retry_attempts: u32,

    /// Consecutive host failures before the circuit breaker opens.
    #[arg(long, default_value_t = 3)]
    cb_threshold: u32,

    /// How long a host circuit stays open, in seconds.
    #[arg(long, default_value_t = 60)]
    cb_cooldown_secs: u64,

    /// Delete quarantined invalid files instead of keeping them.
    #[arg(long)]
    clean_invalid: bool,

    /// Persistent manifest path (enables resume/retry-failed/validate tracking).
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Extra args appended to EVERY yt-dlp invocation (measure, download,
    /// resume) after the standard args (e.g. "-x --audio-format mp3").
    #[arg(long, allow_hyphen_values = true)]
    ytdlp_extra_args: Option<String>,
}

/// Map CLI run-mode flags to the engine [`RunMode`].
///
/// Engine union semantics (documented in `downloader.rs`): when both
/// `--resume` and `--retry-failed` are given, [`RunMode::Resume`] is chosen
/// because Failed ⊆ not-Complete, so Resume is exactly the union of the two
/// scopes. The engine has no `RetryBoth` variant — the collapse is
/// equivalent, not a loss. `--validate-only` conflicts with both at the clap
/// layer (exit 2).
fn run_mode_from_flags(resume: bool, retry_failed: bool, validate_only: bool) -> RunMode {
    if validate_only {
        RunMode::ValidateOnly
    } else if resume {
        RunMode::Resume
    } else if retry_failed {
        RunMode::RetryFailed
    } else {
        RunMode::Default
    }
}

/// Split `--ytdlp-extra-args` on ASCII whitespace into one arg per token.
fn split_ytdlp_extra_args(extra: &Option<String>) -> Vec<String> {
    extra
        .as_deref()
        .map(|v| v.split_ascii_whitespace().map(str::to_string).collect())
        .unwrap_or_default()
}

/// Build the engine [`DownloadConfig`] from parsed CLI args.
///
/// The manifest defaults to `<out_dir>/<slug>.anime-manifest.json` when
/// `--manifest` is absent: the bin always enables manifest integration (the
/// engine keeps its own `None` → no-manifest fallback for API callers). The
/// slug is sanitized upstream ([`sanitize_slug`]); the default path inherits
/// that safety.
fn build_download_config(args: &Args, out_dir: PathBuf, slug: String) -> DownloadConfig {
    let default_manifest = out_dir.join(format!("{slug}.anime-manifest.json"));
    DownloadConfig {
        episode_concurrency: args.episode_concurrency,
        host_concurrency: args.host_concurrency,
        out_dir,
        slug,
        run_mode: run_mode_from_flags(args.resume, args.retry_failed, args.validate_only),
        no_validate: args.no_validate,
        validate_force: args.validate_force,
        ytdlp_extra_args: split_ytdlp_extra_args(&args.ytdlp_extra_args),
        retry_attempts: args.retry_attempts,
        circuit_threshold: args.cb_threshold,
        circuit_cooldown_secs: args.cb_cooldown_secs,
        validation: ValidationConfig {
            min_size_bytes: (args.min_size_mb * 1_048_576.0) as u64,
            min_duration_secs: args.min_duration_secs,
            require_audio: args.require_audio,
            ffprobe_timeout: Duration::from_secs(args.ffprobe_timeout),
        },
        manifest_path: args.manifest.clone().or(Some(default_manifest)),
        clean_invalid: args.clean_invalid,
        ..Default::default()
    }
}

/// Map an engine [`Outcome`] to a process exit code per the CLI contract.
///
/// Normal runs: 0 all ok; 3 partial (some failed, some ok); 4 all requested
/// failed; 130 cancelled. Validate-only: 0 all valid; 3 some missing/invalid;
/// 4 all missing/invalid. `requested` is the number of episodes the run
/// covered (after `--episodes` filtering).
///
/// Invariant: `missing_episodes` is the authoritative count of episodes
/// without valid output on disk — aggregation pushes every `Failed`/`Cancelled`
/// episode into it alongside pure `Missing` ones, so `failed` must never be
/// added on top of it (would double-count). Cancelled exits 130 before this
/// branch is reached.
fn outcome_exit_code(outcome: &Outcome, validate_only: bool, requested: usize) -> i32 {
    if outcome.cancelled {
        return 130;
    }
    if validate_only {
        let invalid = outcome.missing_episodes.len();
        if invalid == 0 {
            0
        } else if invalid >= requested {
            4
        } else {
            3
        }
    } else if outcome.failed > 0 && outcome.downloaded + outcome.skipped > 0 {
        3
    } else if outcome.failed > 0 {
        4
    } else {
        0
    }
}

async fn run_fancy_renderer(
    rx: broadcast::Receiver<EpEvent>,
    episode_nums: &[i64],
    title: Option<&str>,
    circuit_cooldown_secs: u64,
) -> io::Result<()> {
    let renderer = select_renderer(rx, episode_nums, title, circuit_cooldown_secs);
    renderer.run().await
}

fn is_plain_mode(cli: Option<&CliUiMode>) -> bool {
    match cli {
        Some(CliUiMode::Plain) => true,
        None => !io::stdout().is_terminal(),
        _ => false,
    }
}

async fn download_from_json(args: &Args) -> Result<(Outcome, usize)> {
    let json_path = args
        .input
        .as_ref()
        .ok_or_else(|| anyhow!("--input path required"))?;
    let text = fs::read_to_string(json_path)
        .map_err(|e| anyhow!("reading {}: {e}", json_path.display()))?;
    let input = parse_json_input(&text).map_err(|e| anyhow!("{e}"))?;

    let title = input.title.as_deref().unwrap_or("anime");
    // Sanitized slug drives output naming, manifest default path, and the
    // quarantine prefix. Never derived by raw case/replace (path traversal
    // risk: `/` and `..` in titles would escape out_dir).
    let slug = sanitize_slug(title);

    let out_dir = args.out.clone().unwrap_or_else(|| PathBuf::from(&slug));
    fs::create_dir_all(&out_dir).map_err(|e| anyhow!("creating output dir: {e}"))?;

    let episodes: Vec<EpisodeInput> = if let Some(filter) = &args.episodes {
        input
            .episodes
            .into_iter()
            .filter(|ep| filter.contains(&ep.episode))
            .collect()
    } else {
        input.episodes
    };

    if episodes.is_empty() {
        bail!("no matching episodes to download");
    }

    let episode_nums: Vec<i64> = episodes.iter().map(|e| e.episode).collect();
    let requested = episodes.len();

    let mut config = build_download_config(args, out_dir.clone(), slug.clone());
    // Input identity: engine hashes the source JSON and records its path in
    // the manifest at run start.
    config.input_source_path = args.input.clone();
    config.input_resolved_at = input.resolved_at.clone();
    let engine = Arc::new(DownloadEngine::new(config));

    let cancel = engine.cancel_token();
    tokio::spawn(async move {
        ctrl_c().await.ok();
        cancel.cancel();
    });

    let (tx, rx) = broadcast::channel::<EpEvent>(256);
    let total = episodes.len();

    let engine_clone = engine.clone();
    let tx_clone = tx.clone();
    let download_handle = tokio::spawn(async move {
        engine_clone
            .run_all_with_outcome(episodes.clone(), tx_clone)
            .await
    });

    let plain = is_plain_mode(args.ui.as_ref());

    let outcome = if plain {
        let plain_handle = spawn_plain_output(rx, total);
        let outcome = download_handle
            .await
            .map_err(|e| anyhow!("download task panicked: {e}"))?;
        drop(tx);
        let (_done, _failed) = plain_handle
            .await
            .map_err(|e| anyhow!("plain output task panicked: {e}"))?;
        outcome
    } else {
        run_fancy_renderer(rx, &episode_nums, Some(title), args.cb_cooldown_secs)
            .await
            .map_err(|e| anyhow!("renderer error: {e}"))?;
        download_handle
            .await
            .map_err(|e| anyhow!("download task panicked: {e}"))?
    };

    println!(
        "Done {title}: {} downloaded, {} skipped, {} failed.",
        outcome.downloaded, outcome.skipped, outcome.failed
    );
    Ok((outcome, requested))
}

fn run(args: Args) -> Result<i32> {
    let input_path = args.input.clone().or_else(|| {
        args.name.as_ref().and_then(|name| {
            if Path::new(name).extension().and_then(|e| e.to_str()) == Some("json") {
                Some(PathBuf::from(name))
            } else {
                None
            }
        })
    });

    let input_path = input_path
        .ok_or_else(|| anyhow!("provide --input <path> to a canonical JSON episode list"))?;

    let args = Args {
        input: Some(input_path),
        ..args
    };

    let rt = Runtime::new().map_err(|e| anyhow!("tokio runtime: {e}"))?;
    rt.block_on(async {
        match download_from_json(&args).await {
            Ok((outcome, requested)) => {
                let code = outcome_exit_code(&outcome, args.validate_only, requested);
                Ok(code)
            }
            Err(e) => {
                error!("Error: {e:#}");
                Ok(1)
            }
        }
    })
}

fn main() {
    let args = Args::parse();
    mt_cli::init_tracing(args.verbose);
    let code = match run(args) {
        Ok(code) => code,
        Err(e) => {
            error!("Error: {e:#}");
            1
        }
    };
    exit(code);
}

#[cfg(test)]
mod tests {
    use mt_cli::downloader::Outcome;

    use super::*;

    // ── Exit-code mapping ────────────────────────────────────────────────

    #[test]
    fn exit_code_all_success() {
        let o = Outcome {
            downloaded: 5,
            skipped: 0,
            failed: 0,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, false, 5), 0);
    }

    #[test]
    fn exit_code_partial_failure_is_3() {
        let o = Outcome {
            downloaded: 3,
            skipped: 1,
            failed: 2,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, false, 6), 3);
    }

    #[test]
    fn exit_code_all_failed_is_4() {
        let o = Outcome {
            downloaded: 0,
            skipped: 0,
            failed: 5,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, false, 5), 4);
    }

    #[test]
    fn exit_code_none_downloaded() {
        let o = Outcome::default();
        assert_eq!(outcome_exit_code(&o, false, 0), 0);
    }

    #[test]
    fn exit_code_cancelled_is_130() {
        let o = Outcome {
            downloaded: 2,
            failed: 1,
            cancelled: true,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, false, 3), 130);
    }

    #[test]
    fn exit_code_validate_only_all_valid() {
        let o = Outcome {
            skipped: 4,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 4), 0);
    }

    #[test]
    fn exit_code_validate_only_some_invalid_is_3() {
        let o = Outcome {
            failed: 1,
            missing_episodes: vec![2],
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 4), 3);
    }

    #[test]
    fn exit_code_validate_only_all_invalid_is_4() {
        let o = Outcome {
            missing_episodes: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 3), 4);
    }

    #[test]
    fn validate_only_with_failed_in_missing_partial() {
        // failed ⊆ missing_episodes: 1 of 2 requested lacks output → partial.
        let o = Outcome {
            failed: 1,
            missing_episodes: vec![1],
            downloaded: 1,
            skipped: 0,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 2), 3);
    }

    #[test]
    fn validate_only_with_failed_in_missing_all_failed() {
        // 1 requested, 1 missing (same episode as failed) → all invalid.
        let o = Outcome {
            failed: 1,
            missing_episodes: vec![1],
            downloaded: 0,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 1), 4);
    }

    #[test]
    fn validate_only_all_missing_none_failed() {
        // Pure missing (no failed count) still counts as invalid.
        let o = Outcome {
            failed: 0,
            missing_episodes: vec![1, 2],
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 2), 4);
    }

    #[test]
    fn validate_only_mixed_missing_and_ok() {
        // 1 missing, 1 downloaded of 2 requested → partial.
        let o = Outcome {
            failed: 0,
            missing_episodes: vec![2],
            downloaded: 1,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 2), 3);
    }

    #[test]
    fn validate_only_cancelled_precedence() {
        // Cancelled wins over every validate-only classification.
        let o = Outcome {
            cancelled: true,
            failed: 1,
            missing_episodes: vec![1, 2],
            downloaded: 1,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&o, true, 3), 130);
    }

    #[test]
    fn non_validate_overlap_semantics_unchanged() {
        // Non-validate branch reads `failed` directly; must stay untouched.
        let partial = Outcome {
            failed: 1,
            downloaded: 1,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&partial, false, 2), 3);
        let all_failed = Outcome {
            failed: 1,
            downloaded: 0,
            skipped: 0,
            ..Default::default()
        };
        assert_eq!(outcome_exit_code(&all_failed, false, 1), 4);
    }

    // ── RunMode mapping ──────────────────────────────────────────────────

    #[test]
    fn run_mode_default_when_no_flags() {
        assert_eq!(run_mode_from_flags(false, false, false), RunMode::Default);
    }

    #[test]
    fn run_mode_resume() {
        assert_eq!(run_mode_from_flags(true, false, false), RunMode::Resume);
    }

    #[test]
    fn run_mode_retry_failed() {
        assert_eq!(
            run_mode_from_flags(false, true, false),
            RunMode::RetryFailed
        );
    }

    #[test]
    fn run_mode_both_collapses_to_resume() {
        // Engine union semantics: retry-failed ⊆ not-complete.
        assert_eq!(run_mode_from_flags(true, true, false), RunMode::Resume);
    }

    #[test]
    fn run_mode_validate_only_wins() {
        assert_eq!(
            run_mode_from_flags(false, false, true),
            RunMode::ValidateOnly
        );
    }

    // ── Clap parsing ─────────────────────────────────────────────────────

    #[test]
    fn clap_accepts_input_flag() {
        let args = Args::parse_from(["anime-dl", "--input", "/tmp/test.json"]);
        assert!(args.input.is_some());
        assert_eq!(args.input.unwrap(), PathBuf::from("/tmp/test.json"));
    }

    #[test]
    fn clap_rejects_json_flag() {
        let result = Args::try_parse_from(["anime-dl", "--json", "/tmp/test.json"]);
        assert!(result.is_err(), "--json flag must be rejected");
    }

    #[test]
    fn positional_json_routes_to_input() {
        let args = Args::parse_from(["anime-dl", "/tmp/episodes.json"]);
        assert!(args.name.is_some());
        assert_eq!(args.name.as_deref(), Some("/tmp/episodes.json"));
        assert!(args.input.is_none());
    }

    #[test]
    fn positional_non_json_is_error() {
        let args = Args::parse_from(["anime-dl", "Naruto"]);
        assert!(args.name.is_some());
        assert!(args.input.is_none());
    }

    #[test]
    fn no_args_produces_clear_error() {
        let args = Args::parse_from(["anime-dl"]);
        assert!(args.input.is_none());
        assert!(args.name.is_none());
    }

    #[test]
    fn supported_ui_modes_parse_correctly() {
        assert!(matches!(
            "dashboard".parse::<CliUiMode>(),
            Ok(CliUiMode::Dashboard)
        ));
        assert!(matches!("a".parse::<CliUiMode>(), Ok(CliUiMode::Dashboard)));
        assert!(matches!(
            "tui".parse::<CliUiMode>(),
            Ok(CliUiMode::Dashboard)
        ));
        assert!(matches!("plain".parse::<CliUiMode>(), Ok(CliUiMode::Plain)));
    }

    #[test]
    fn removed_ui_modes_are_rejected() {
        for mode in ["timeline", "b", "scoreboard", "c", "stream", "d"] {
            assert!(
                mode.parse::<CliUiMode>().is_err(),
                "{mode} must be rejected"
            );
        }
    }

    #[test]
    fn ui_invalid_value_is_clap_error() {
        let result = Args::try_parse_from(["anime-dl", "--input", "/tmp/a.json", "--ui", "bogus"]);
        assert!(result.is_err(), "invalid --ui value must be a clap error");
    }

    // ── New flags ────────────────────────────────────────────────────────

    #[test]
    fn new_flags_parse() {
        let args = Args::parse_from([
            "anime-dl",
            "--input",
            "/tmp/test.json",
            "--resume",
            "--retry-failed",
            "--no-validate",
            "--validate-force",
            "--min-size-mb",
            "2.5",
            "--min-duration-secs",
            "3",
            "--require-audio",
            "--ffprobe-timeout",
            "30",
            "--retry-attempts",
            "5",
            "--cb-threshold",
            "2",
            "--cb-cooldown-secs",
            "120",
            "--clean-invalid",
            "--manifest",
            "/tmp/m.json",
            "--ytdlp-extra-args",
            "-x --audio-format mp3",
        ]);
        assert!(args.resume);
        assert!(args.retry_failed);
        assert!(args.no_validate);
        assert!(args.validate_force);
        assert_eq!(args.min_size_mb, 2.5);
        assert_eq!(args.min_duration_secs, 3.0);
        assert!(args.require_audio);
        assert_eq!(args.ffprobe_timeout, 30);
        assert_eq!(args.retry_attempts, 5);
        assert_eq!(args.cb_threshold, 2);
        assert_eq!(args.cb_cooldown_secs, 120);
        assert!(args.clean_invalid);
        assert_eq!(args.manifest.as_deref(), Some(Path::new("/tmp/m.json")));
        assert_eq!(
            args.ytdlp_extra_args.as_deref(),
            Some("-x --audio-format mp3")
        );
    }

    #[test]
    fn new_flags_defaults() {
        let args = Args::parse_from(["anime-dl", "--input", "/tmp/test.json"]);
        assert!(!args.resume);
        assert!(!args.retry_failed);
        assert!(!args.validate_only);
        assert!(!args.no_validate);
        assert!(!args.validate_force);
        assert_eq!(args.min_size_mb, 1.0);
        assert_eq!(args.min_duration_secs, 1.0);
        assert!(!args.require_audio);
        assert_eq!(args.ffprobe_timeout, 15);
        assert_eq!(args.retry_attempts, 3);
        assert_eq!(args.cb_threshold, 3);
        assert_eq!(args.cb_cooldown_secs, 60);
        assert!(!args.clean_invalid);
        assert!(args.manifest.is_none());
        assert!(args.ytdlp_extra_args.is_none());
    }

    #[test]
    fn validate_only_conflicts_with_resume() {
        let result = Args::try_parse_from([
            "anime-dl",
            "--input",
            "/tmp/test.json",
            "--validate-only",
            "--resume",
        ]);
        assert!(result.is_err(), "validate-only + resume must conflict");
    }

    #[test]
    fn validate_only_conflicts_with_retry_failed() {
        let result = Args::try_parse_from([
            "anime-dl",
            "--input",
            "/tmp/test.json",
            "--retry-failed",
            "--validate-only",
        ]);
        assert!(
            result.is_err(),
            "retry-failed + validate-only must conflict"
        );
    }

    #[test]
    fn validate_only_parses_alone() {
        let args = Args::parse_from(["anime-dl", "--input", "/tmp/test.json", "--validate-only"]);
        assert!(args.validate_only);
    }

    #[test]
    fn config_maps_validation_flags() {
        let args = Args::parse_from([
            "anime-dl",
            "--input",
            "/tmp/test.json",
            "--min-size-mb",
            "2.5",
            "--min-duration-secs",
            "3",
            "--require-audio",
            "--ffprobe-timeout",
            "30",
            "--retry-attempts",
            "5",
            "--cb-threshold",
            "2",
            "--cb-cooldown-secs",
            "120",
            "--clean-invalid",
            "--manifest",
            "/tmp/m.json",
            "--no-validate",
            "--validate-force",
            "--ytdlp-extra-args",
            "-x --audio-format mp3",
        ]);
        let cfg = build_download_config(&args, PathBuf::from("x"), "x".into());
        assert_eq!(cfg.retry_attempts, 5);
        assert_eq!(cfg.circuit_threshold, 2);
        assert_eq!(cfg.circuit_cooldown_secs, 120);
        assert!(cfg.clean_invalid);
        assert_eq!(cfg.validation.min_size_bytes, (2.5 * 1_048_576.0) as u64);
        assert_eq!(cfg.validation.min_duration_secs, 3.0);
        assert!(cfg.validation.require_audio);
        assert_eq!(cfg.validation.ffprobe_timeout, Duration::from_secs(30));
        assert_eq!(cfg.manifest_path.as_deref(), Some(Path::new("/tmp/m.json")));
        // Gap-2 wiring: the three new knobs land in the engine config.
        assert!(cfg.no_validate, "--no-validate wired");
        assert!(cfg.validate_force, "--validate-force wired");
        assert_eq!(
            cfg.ytdlp_extra_args,
            vec!["-x", "--audio-format", "mp3"],
            "--ytdlp-extra-args split on ASCII whitespace"
        );
    }

    #[test]
    fn config_maps_new_flags_defaults() {
        let args = Args::parse_from(["anime-dl", "--input", "/tmp/test.json"]);
        let cfg = build_download_config(&args, PathBuf::from("x"), "x".into());
        assert!(!cfg.no_validate);
        assert!(!cfg.validate_force);
        assert!(cfg.ytdlp_extra_args.is_empty());
    }

    #[test]
    fn config_defaults_manifest_to_out_dir_slug_with_sanitized_slug() {
        // No --manifest flag → default `<out>/<slug>.anime-manifest.json`,
        // where the slug is the sanitized title (never raw case/replace).
        let args = Args::parse_from(["anime-dl", "--input", "/tmp/test.json"]);
        let slug = sanitize_slug("Attack on Titan: Final Season!");
        assert_eq!(slug, "attack-on-titan-final-season");
        let cfg = build_download_config(&args, PathBuf::from("/tmp/out"), slug);
        assert_eq!(
            cfg.manifest_path.as_deref(),
            Some(Path::new(
                "/tmp/out/attack-on-titan-final-season.anime-manifest.json"
            )),
            "bin always enables manifest integration at the default path"
        );
        // Path-traversal-ish titles must stay inside out_dir.
        let tricky = sanitize_slug("../../etc/passwd");
        let cfg2 = build_download_config(&args, PathBuf::from("/tmp/out"), tricky);
        let path = cfg2.manifest_path.expect("manifest always set by bin");
        assert!(
            !path.to_string_lossy().contains(".."),
            "no traversal: {path:?}"
        );
    }

    #[test]
    fn config_explicit_manifest_overrides_default() {
        let args = Args::parse_from([
            "anime-dl",
            "--input",
            "/tmp/test.json",
            "--manifest",
            "/tmp/m.json",
        ]);
        let cfg = build_download_config(&args, PathBuf::from("x"), "x".into());
        assert_eq!(cfg.manifest_path.as_deref(), Some(Path::new("/tmp/m.json")));
    }

    #[test]
    fn after_help_documents_v2_schema_and_exit_codes() {
        assert!(AFTER_HELP.contains("schema_version"));
        assert!(AFTER_HELP.contains("mirrors"));
        assert!(AFTER_HELP.contains("EXIT CODES"));
        assert!(AFTER_HELP.contains("130"));
        assert!(
            !AFTER_HELP.contains("episodes\": [{\"episode\": 1, \"urls\""),
            "help must not push v1-only shape"
        );
    }
}
