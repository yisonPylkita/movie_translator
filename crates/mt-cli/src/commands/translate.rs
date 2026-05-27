//! `translate` subcommand (the default command).
//!
//! Port of `movie_translator/commands/translate_cmd.py`.

use std::path::{Path, PathBuf};

use clap::Parser;
use mt_core::PipelineConfig;
use mt_discovery::find_videos;
use mt_pipeline::{run_all, FileStatus};

use crate::common::{check_dependencies, resolve_models};
use crate::progress::Progress;

/// Default device per platform — `mps` on macOS, `cpu` elsewhere.
fn default_device() -> String {
    if cfg!(target_os = "macos") {
        "mps".to_string()
    } else {
        "cpu".to_string()
    }
}

/// Translate command arguments, mirroring `translate_cmd.parse_args`.
#[derive(Debug, Parser)]
#[command(
    name = "movie-translator",
    about = "Movie Translator - Extract English dialogue -> AI translate to Polish -> Replace original video"
)]
pub struct TranslateArgs {
    /// Video file or directory containing video files.
    pub input: String,

    /// Inference device.
    #[arg(long, value_parser = ["cpu", "mps"], default_value_t = default_device())]
    pub device: String,

    #[arg(long = "batch-size", default_value_t = 16)]
    pub batch_size: u32,

    /// Translation backend. Default on macOS runs BOTH allegro and apple.
    #[arg(long, value_parser = ["allegro", "apple"])]
    pub model: Option<String>,

    #[arg(long = "no-fetch", default_value_t = false)]
    pub no_fetch: bool,

    /// Remove burned-in subtitles from video frames via inpainting (slow).
    #[arg(long, default_value_t = false)]
    pub inpaint: bool,

    /// Disk-frugal mode: write mux output beside the original and atomically
    /// replace it. Incompatible with --inpaint.
    #[arg(long = "in-place", default_value_t = false)]
    pub in_place: bool,

    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,

    #[arg(long = "keep-artifacts", default_value_t = false)]
    pub keep_artifacts: bool,

    /// Concurrent pipeline workers (default: auto, min(files, 4)).
    #[arg(long, default_value_t = 0)]
    pub workers: u32,

    /// Directory with pre-extracted subtitles to add as additional tracks.
    #[arg(long = "external-subs")]
    pub external_subs: Option<String>,

    #[arg(long, short = 'v', default_value_t = false)]
    pub verbose: bool,

    /// Collect performance metrics.
    #[arg(long, default_value_t = false)]
    pub metrics: bool,
}

impl TranslateArgs {
    /// Build a [`PipelineConfig`] from parsed args + resolved models.
    ///
    /// Mirrors `_async_main`'s `PipelineConfig(...)` construction. `--no-fetch`
    /// maps to `enable_fetch = false`; `workers` is left as parsed (0 = auto)
    /// so the orchestrator applies the `min(files, 4)` fallback.
    pub fn to_config(&self, model: String, extra_models: Vec<String>) -> PipelineConfig {
        PipelineConfig {
            device: self.device.clone(),
            batch_size: self.batch_size,
            model,
            extra_models,
            enable_fetch: !self.no_fetch,
            enable_inpaint: self.inpaint,
            dry_run: self.dry_run,
            in_place: self.in_place,
            workers: self.workers,
            external_subs_dir: self.external_subs.as_ref().map(PathBuf::from),
        }
    }
}

/// Format the per-run summary line.
///
/// Port of `_show_summary`. Returns the summary string (so it is testable) and
/// whether the dry-run note should be appended.
pub fn format_summary(results: &[(PathBuf, FileStatus)], dry_run: bool) -> String {
    let successful = results.iter().filter(|(_, s)| *s == FileStatus::Success).count();
    let failed = results.iter().filter(|(_, s)| *s == FileStatus::Failed).count();
    let skipped = results.iter().filter(|(_, s)| *s == FileStatus::Skipped).count();

    let mut parts = Vec::new();
    if successful > 0 {
        parts.push(format!("✓ {successful} translated"));
    }
    if skipped > 0 {
        parts.push(format!("⏭ {skipped} skipped"));
    }
    if failed > 0 {
        parts.push(format!("✗ {failed} failed"));
    }
    let mut out = parts.join(" | ");
    if dry_run && successful > 0 {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("Dry run - originals not modified");
    }
    out
}

/// Delete stale `*.translating.*` temp files from a prior crashed run.
///
/// Port of `_cleanup_in_place_orphans`. Only files whose stem maps back to a
/// discovered input video are removed.
pub fn cleanup_in_place_orphans(video_files: &[PathBuf]) -> usize {
    let mut removed = 0;
    for vp in video_files {
        let orphan = mt_pipeline::stages::mux::in_place_temp_path(vp);
        if orphan.exists() {
            match std::fs::remove_file(&orphan) {
                Ok(()) => {
                    removed += 1;
                    tracing::warn!(
                        "Removed orphan temp: {}",
                        orphan.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
                    );
                }
                Err(e) => tracing::warn!("Could not remove orphan {}: {e}", orphan.display()),
            }
        }
    }
    if removed > 0 {
        eprintln!("Cleaned up {removed} orphan temp file(s) from prior run");
    }
    removed
}

/// Run the translate flow. Returns the process exit code.
///
/// Port of `translate_cmd.run`. Async because it drives `run_all` on the
/// multi-threaded runtime.
pub async fn run(args: TranslateArgs) -> i32 {
    crate::init_tracing(args.verbose);

    let (model, extra_models) = resolve_models(args.model.as_deref());

    if args.in_place && args.inpaint {
        eprintln!(
            "--in-place is incompatible with --inpaint (inpainting requires an extra full-size temp copy)."
        );
        return 2;
    }

    let input_path = PathBuf::from(&args.input);
    if !input_path.exists() {
        eprintln!("Not found: {}", input_path.display());
        return 1;
    }

    if !check_dependencies() {
        return 1;
    }

    let video_files = find_videos(&input_path);
    if video_files.is_empty() {
        eprintln!("No video files found in {}", input_path.display());
        return 1;
    }

    let root_dir: PathBuf = if input_path.is_dir() {
        input_path.clone()
    } else {
        input_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };

    if args.in_place {
        cleanup_in_place_orphans(&video_files);
        eprintln!("In-place mode: peak disk ~2x per worker; originals replaced atomically.");
    }

    if args.dry_run {
        eprintln!("Dry run mode - originals will not be modified");
    }

    let config = args.to_config(model, extra_models);

    let progress = Progress::new(video_files.len() as u64);
    for vp in &video_files {
        progress.start_file(&display_name(vp, &root_dir));
    }

    let results = match run_all(video_files, root_dir, config).await {
        Ok(r) => r,
        Err(e) => {
            progress.finish();
            eprintln!("Pipeline error: {e}");
            return 1;
        }
    };

    for (vp, status) in &results {
        progress.finish_file(&vp.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), status.as_str());
    }
    progress.finish();

    let summary = format_summary(&results, args.dry_run);
    if !summary.is_empty() {
        println!("{summary}");
    }

    let any_failed = results.iter().any(|(_, s)| *s == FileStatus::Failed);
    if any_failed {
        1
    } else {
        0
    }
}

fn display_name(video_path: &Path, root_dir: &Path) -> String {
    video_path
        .strip_prefix(root_dir)
        .unwrap_or(video_path)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> TranslateArgs {
        let mut full = vec!["movie-translator"];
        full.extend_from_slice(argv);
        TranslateArgs::try_parse_from(full).expect("parse")
    }

    #[test]
    fn defaults_match_python_argparse() {
        let args = parse(&["movie.mkv"]);
        assert_eq!(args.input, "movie.mkv");
        assert_eq!(args.batch_size, 16);
        assert_eq!(args.workers, 0); // 0 = auto -> min(files, 4)
        assert!(args.model.is_none()); // default None -> resolve_models
        assert!(!args.no_fetch); // enable_fetch true unless --no-fetch
        assert!(!args.dry_run);
        assert!(!args.keep_artifacts);
        assert!(!args.in_place);
        assert!(!args.inpaint);
        assert!(!args.metrics);
        assert!(args.external_subs.is_none());
    }

    #[test]
    fn default_device_per_platform() {
        let args = parse(&["movie.mkv"]);
        if cfg!(target_os = "macos") {
            assert_eq!(args.device, "mps");
        } else {
            assert_eq!(args.device, "cpu");
        }
    }

    #[test]
    fn no_fetch_maps_to_enable_fetch_false() {
        let args = parse(&["movie.mkv", "--no-fetch"]);
        let cfg = args.to_config("allegro".into(), vec![]);
        assert!(!cfg.enable_fetch);

        let args = parse(&["movie.mkv"]);
        let cfg = args.to_config("allegro".into(), vec![]);
        assert!(cfg.enable_fetch);
    }

    #[test]
    fn flags_map_into_config() {
        let args = parse(&[
            "/some/dir",
            "--device",
            "cpu",
            "--batch-size",
            "32",
            "--workers",
            "8",
            "--in-place",
            "--dry-run",
            "--external-subs",
            "/subs",
        ]);
        let cfg = args.to_config("allegro".into(), vec!["apple".into()]);
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.batch_size, 32);
        assert_eq!(cfg.workers, 8);
        assert!(cfg.in_place);
        assert!(cfg.dry_run);
        assert_eq!(cfg.model, "allegro");
        assert_eq!(cfg.extra_models, vec!["apple".to_string()]);
        assert_eq!(cfg.external_subs_dir, Some(PathBuf::from("/subs")));
    }

    #[test]
    fn explicit_model_choice_parses() {
        let args = parse(&["movie.mkv", "--model", "apple"]);
        assert_eq!(args.model.as_deref(), Some("apple"));
    }

    #[test]
    fn invalid_device_rejected() {
        let r = TranslateArgs::try_parse_from(["movie-translator", "m.mkv", "--device", "cuda"]);
        assert!(r.is_err());
    }

    #[test]
    fn summary_counts_and_dry_run_note() {
        let results = vec![
            (PathBuf::from("a.mkv"), FileStatus::Success),
            (PathBuf::from("b.mkv"), FileStatus::Skipped),
            (PathBuf::from("c.mkv"), FileStatus::Failed),
            (PathBuf::from("d.mkv"), FileStatus::Success),
        ];
        let s = format_summary(&results, false);
        assert_eq!(s, "✓ 2 translated | ⏭ 1 skipped | ✗ 1 failed");

        let s = format_summary(&results, true);
        assert!(s.contains("Dry run - originals not modified"));
    }

    #[test]
    fn summary_empty_when_no_results() {
        assert_eq!(format_summary(&[], false), "");
    }
}
