//! `anime-dl` — standalone ogladajanime.pl season downloader.
//!
//! Accepts a canonical JSON episode list via `--input <path>` (or positional
//! `.json` path). Downloads episodes at best available quality. No translation,
//! no OCR — pure download. Reuses the `mt_cli::downloader` engine.

use std::fs::{self};
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use mt_cli::download_types::{EpEvent, EpisodeInput, parse_json_input};
use mt_cli::downloader::{DownloadConfig, DownloadEngine};
use mt_cli::plain_output::spawn_plain_output;
use mt_cli::ui_render::{UiMode, select_renderer};
use tokio::runtime::Runtime;
use tokio::signal::ctrl_c;
use tokio::sync::broadcast;
use tracing::error;

/// UI mode with descriptive names and short aliases.
///
/// | Alias | Full name   | Description                          |
/// |-------|-------------|--------------------------------------|
/// | a     | dashboard   | Pinned header/footer, row per ep     |
/// | b     | timeline    | Row-per-ep timeline with stage glyphs |
/// | c     | scoreboard  | Compact multi-column auto-paging grid |
/// | d     | stream      | Styled recent event stream            |
/// | tui   | dashboard   | Legacy compatibility alias            |
#[derive(Clone, Debug)]
enum CliUiMode {
    Dashboard,
    Timeline,
    Scoreboard,
    Stream,
    Plain,
}

impl std::str::FromStr for CliUiMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "dashboard" | "a" | "tui" => Ok(CliUiMode::Dashboard),
            "timeline" | "b" => Ok(CliUiMode::Timeline),
            "scoreboard" | "c" => Ok(CliUiMode::Scoreboard),
            "stream" | "d" => Ok(CliUiMode::Stream),
            "plain" => Ok(CliUiMode::Plain),
            other => Err(format!(
                "invalid UI mode '{other}'. Valid: dashboard|a|tui, timeline|b, scoreboard|c, stream|d, plain"
            )),
        }
    }
}

fn parse_ui_mode(s: &str) -> Result<CliUiMode, String> {
    s.parse()
}

#[derive(Parser, Debug)]
#[command(
    name = "anime-dl",
    about = "Download anime episodes from ogladajanime.pl (best quality, no translation)",
    after_help = "\
CANONICAL JSON FORMAT:
  { \"title\": \"...\", \"episodes\": [{\"episode\": 1, \"urls\": [\"https://...\"]}] }

  Each episode must have non-empty \"urls\" array. Episodes must be unique.
  Quality metadata optional: {\"quality\": {\"height\": 1080}} per episode.
"
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
    /// Aliases: dashboard|a|tui, timeline|b, scoreboard|c, stream|d, plain.
    #[arg(long, value_parser = parse_ui_mode)]
    ui: Option<CliUiMode>,

    /// Raise our crates' logging to DEBUG.
    #[arg(short, long)]
    verbose: bool,
}

fn resolve_ui_mode(cli: Option<&CliUiMode>) -> UiMode {
    match cli {
        Some(CliUiMode::Dashboard) => UiMode::Dashboard,
        Some(CliUiMode::Timeline) => UiMode::Timeline,
        Some(CliUiMode::Scoreboard) => UiMode::Scoreboard,
        Some(CliUiMode::Stream) => UiMode::Stream,
        _ => UiMode::Dashboard,
    }
}

async fn run_fancy_renderer(
    mode: UiMode,
    rx: broadcast::Receiver<EpEvent>,
    episode_nums: &[i64],
) -> io::Result<()> {
    let renderer = select_renderer(mode, rx, episode_nums);
    renderer.run().await
}

fn is_plain_mode(cli: Option<&CliUiMode>) -> bool {
    match cli {
        Some(CliUiMode::Plain) => true,
        None => !io::stdout().is_terminal(),
        _ => false,
    }
}

async fn download_from_json(args: &Args) -> Result<(u32, u32, u32)> {
    let json_path = args
        .input
        .as_ref()
        .ok_or_else(|| anyhow!("--input path required"))?;
    let text = fs::read_to_string(json_path)
        .map_err(|e| anyhow!("reading {}: {e}", json_path.display()))?;
    let input = parse_json_input(&text).map_err(|e| anyhow!("{e}"))?;

    let title = input.title.as_deref().unwrap_or("anime");
    let slug = title.to_lowercase().replace(' ', "-");

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

    let config = DownloadConfig {
        episode_concurrency: args.episode_concurrency,
        host_concurrency: args.host_concurrency,
        out_dir: out_dir.clone(),
        slug: slug.clone(),
        ..Default::default()
    };
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
    let download_handle =
        tokio::spawn(async move { engine_clone.run_all(episodes.clone(), tx_clone).await });

    let plain = is_plain_mode(args.ui.as_ref());

    if plain {
        let plain_handle = spawn_plain_output(rx, total);
        let _results = download_handle
            .await
            .map_err(|e| anyhow!("download task panicked: {e}"))?;
        drop(tx);
        let (done, failed) = plain_handle
            .await
            .map_err(|e| anyhow!("plain output task panicked: {e}"))?;
        println!("Done {title}: {done} downloaded, {failed} failed.");
        Ok((done, 0, failed))
    } else {
        let mode = resolve_ui_mode(args.ui.as_ref());
        run_fancy_renderer(mode, rx, &episode_nums)
            .await
            .map_err(|e| anyhow!("renderer error: {e}"))?;

        let results = download_handle
            .await
            .map_err(|e| anyhow!("download task panicked: {e}"))?;
        let (done, failed) = count_results(&results);
        println!("Done {title}: {done} downloaded, {failed} failed.");
        Ok((done as u32, 0, failed as u32))
    }
}

fn count_results(results: &[(i64, Option<PathBuf>)]) -> (usize, usize) {
    let done = results.iter().filter(|r| r.1.is_some()).count();
    let failed = results.len() - done;
    (done, failed)
}

fn exit_code(done: u32, failed: u32) -> i32 {
    if failed > 0 && done == 0 {
        2
    } else if failed > 0 {
        1
    } else {
        0
    }
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
            Ok((done, _skipped, failed)) => Ok(exit_code(done, failed)),
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
    use super::*;

    #[test]
    fn exit_code_all_success() {
        assert_eq!(exit_code(5, 0), 0);
    }

    #[test]
    fn exit_code_partial_failure() {
        assert_eq!(exit_code(3, 2), 1);
    }

    #[test]
    fn exit_code_all_failed() {
        assert_eq!(exit_code(0, 5), 2);
    }

    #[test]
    fn exit_code_none_downloaded() {
        assert_eq!(exit_code(0, 0), 0);
    }

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
    fn all_ui_modes_parse_correctly() {
        assert!(matches!(
            "dashboard".parse::<CliUiMode>(),
            Ok(CliUiMode::Dashboard)
        ));
        assert!(matches!("a".parse::<CliUiMode>(), Ok(CliUiMode::Dashboard)));
        assert!(matches!(
            "tui".parse::<CliUiMode>(),
            Ok(CliUiMode::Dashboard)
        ));
        assert!(matches!(
            "timeline".parse::<CliUiMode>(),
            Ok(CliUiMode::Timeline)
        ));
        assert!(matches!("b".parse::<CliUiMode>(), Ok(CliUiMode::Timeline)));
        assert!(matches!(
            "scoreboard".parse::<CliUiMode>(),
            Ok(CliUiMode::Scoreboard)
        ));
        assert!(matches!(
            "c".parse::<CliUiMode>(),
            Ok(CliUiMode::Scoreboard)
        ));
        assert!(matches!(
            "stream".parse::<CliUiMode>(),
            Ok(CliUiMode::Stream)
        ));
        assert!(matches!("d".parse::<CliUiMode>(), Ok(CliUiMode::Stream)));
        assert!(matches!("plain".parse::<CliUiMode>(), Ok(CliUiMode::Plain)));
    }
}
