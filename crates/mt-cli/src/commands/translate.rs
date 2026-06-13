//! `translate` subcommand (the default command).

use std::path::{Path, PathBuf};

use clap::Parser;
use mt_core::PipelineConfig;
use mt_discovery::find_videos;
use mt_pipeline::{FileStatus, ProgressSender, run_all_with_progress};
use tracing::{error, info, warn};

use crate::common::{check_dependencies, resolve_models};
use crate::tui::{python_stderr_capture_path, spawn_tui, stdout_is_tty};

/// Default device per platform — `mps` on macOS, `cpu` elsewhere.
fn default_device() -> String {
    if cfg!(target_os = "macos") {
        "mps".to_string()
    } else {
        "cpu".to_string()
    }
}

/// Translate command arguments.
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

    #[arg(long = "batch-size", default_value_t = 4)]
    pub batch_size: u32,

    /// Translation backend. Default: MLX on Apple Silicon, allegro (PyTorch) otherwise.
    #[arg(long, value_parser = ["allegro", "apple", "mlx"])]
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

    /// Source Polish subtitles by OCRing burned-in subs from ogladajanime.pl.
    /// Opens your browser at the matched anime; you run the resolver userscript
    /// and the run picks up its JSON from ~/Downloads. macOS-only (Vision OCR).
    #[arg(long = "hardsub-ocr", default_value_t = false)]
    pub hardsub_ocr: bool,

    /// Re-process files that already have Polish subtitles (normally skipped).
    #[arg(long, default_value_t = false)]
    pub force: bool,

    /// Source English dialogue from the audio track via ASR when no subtitle
    /// text is found (embedded/fetched/burned-in all missed). Transcribes the
    /// English audio track; see benchmarks/asr/REPORT.md for the engine bake-off.
    #[arg(long = "transcribe", default_value_t = false)]
    pub transcribe: bool,

    /// ASR engine for --transcribe: "apple" (SpeechAnalyzer, macOS 26+,
    /// fastest) or "whisper" (mlx-whisper large-v3, Metal).
    #[arg(long = "transcribe-engine", default_value = "apple",
          value_parser = ["apple", "whisper"])]
    pub transcribe_engine: String,

    #[arg(long, short = 'v', default_value_t = false)]
    pub verbose: bool,

    /// Collect performance metrics. Accepted for CLI compatibility, but the
    /// metrics subsystem is not implemented (see [`warn_unimplemented_metrics`]).
    #[arg(long, default_value_t = false)]
    pub metrics: bool,
}

/// Message emitted to stderr when `--metrics` is passed.
///
/// The metrics subsystem is not implemented. The flag stays accepted so existing
/// invocations don't error, but we warn so users aren't misled into expecting an
/// output file.
pub const METRICS_NOT_IMPLEMENTED_WARNING: &str =
    "warning: --metrics is not implemented (no metrics are collected)";

/// Emit the metrics-not-implemented warning to stderr if `metrics` is set.
fn warn_unimplemented_metrics(metrics: bool) {
    if metrics {
        warn!("{METRICS_NOT_IMPLEMENTED_WARNING}");
    }
}

impl TranslateArgs {
    /// Build a [`PipelineConfig`] from parsed args + resolved models.
    ///
    /// `--no-fetch` maps to `enable_fetch = false`; `workers` is left as parsed
    /// (0 = auto) so the orchestrator applies the `min(files, 4)` fallback.
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
            keep_artifacts: self.keep_artifacts,
            enable_hardsub_ocr: self.hardsub_ocr,
            force: self.force,
            enable_transcription: self.transcribe,
            transcribe_engine: self.transcribe_engine.clone(),
        }
    }
}

/// Format the per-run summary line.
///
/// Returns the summary string (so it is testable); a dry-run note is appended
/// when `dry_run` is set and there were successes.
pub fn format_summary(results: &[(PathBuf, FileStatus)], dry_run: bool) -> String {
    let successful = results
        .iter()
        .filter(|(_, s)| *s == FileStatus::Success)
        .count();
    let failed = results
        .iter()
        .filter(|(_, s)| *s == FileStatus::Failed)
        .count();
    let skipped = results
        .iter()
        .filter(|(_, s)| *s == FileStatus::Skipped)
        .count();

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
/// Only files whose stem maps back to a discovered input video are removed.
pub fn cleanup_in_place_orphans(video_files: &[PathBuf]) -> usize {
    let mut removed = 0;
    for vp in video_files {
        let orphan = mt_pipeline::stages::mux::in_place_temp_path(vp);
        if orphan.exists() {
            match std::fs::remove_file(&orphan) {
                Ok(()) => {
                    removed += 1;
                    warn!(
                        "Removed orphan temp: {}",
                        orphan
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default()
                    );
                }
                Err(e) => warn!("Could not remove orphan {}: {e}", orphan.display()),
            }
        }
    }
    if removed > 0 {
        info!("Cleaned up {removed} orphan temp file(s) from prior run");
    }
    removed
}

/// Run the translate flow. Returns the deliberate process exit code as
/// `Ok(code)`, or an `Err` carrying a `.context()` chain (printed by `main`)
/// for a genuine pipeline failure — the structured `PipelineError` cause from
/// `run_all` propagates through anyhow.
///
/// Async because it drives `run_all` on the multi-threaded runtime.
pub async fn run(args: TranslateArgs) -> anyhow::Result<i32> {
    use anyhow::Context;

    warn_unimplemented_metrics(args.metrics);

    let (model, extra_models) = resolve_models(args.model.as_deref());

    if args.in_place && args.inpaint {
        error!(
            "--in-place is incompatible with --inpaint (inpainting requires an extra full-size temp copy)."
        );
        return Ok(2);
    }

    let input_path = PathBuf::from(&args.input);
    if !input_path.exists() {
        error!("Not found: {}", input_path.display());
        return Ok(1);
    }

    if !check_dependencies() {
        return Ok(1);
    }

    let video_files = find_videos(&input_path);
    if video_files.is_empty() {
        error!("No video files found in {}", input_path.display());
        return Ok(1);
    }

    let root_dir = if input_path.is_dir() {
        input_path.clone()
    } else {
        input_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };

    if args.in_place {
        cleanup_in_place_orphans(&video_files);
        info!("In-place mode: peak disk ~2x per worker; originals replaced atomically.");
    }

    if args.dry_run {
        info!("Dry run mode - originals will not be modified");
    }

    let config = args.to_config(model, extra_models);

    // Wire up the TUI: a bounded mpsc carries ProgressEvents from the pipeline
    // (and tracing events via the TuiTracingLayer) to a renderer thread. In
    // plain (no-TTY) mode the same channel is drained by a one-line-per-event
    // printer instead.
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let sender = ProgressSender::new(event_tx);

    // Tracing must be initialised AFTER the sender exists so the layer can
    // ship into the channel. Once a TUI sender is supplied, tracing skips the
    // fmt-stderr layer to avoid clobbering the alternate screen.
    let interactive = stdout_is_tty();
    if interactive {
        crate::init_tracing_with(args.verbose, Some(sender.clone()));
    } else {
        crate::init_tracing(args.verbose);
    }

    let python_log_path = python_stderr_capture_path(&root_dir);
    let tui = spawn_tui(event_rx, python_log_path.clone(), !interactive);

    let result = run_all_with_progress(video_files, root_dir.clone(), config, sender.clone()).await;

    // Drop the sender so the TUI's receiver hits EOF and the renderer exits.
    drop(sender);

    let results = match result {
        Ok(r) => r,
        Err(e) => {
            // Tear the TUI down before printing the error so the chain isn't
            // hidden behind the alternate-screen.
            let _ = tui.join();
            return Err(e).context("running the translation pipeline");
        }
    };

    // Wait for the TUI to drain remaining events and tear itself down.
    let _ = tui.join();

    let summary = format_summary(&results, args.dry_run);
    if !summary.is_empty() {
        println!("{summary}");
    }
    // Best-effort hint about the python stderr capture file.
    if python_log_path.exists() {
        info!("python stderr captured at: {}", python_log_path.display());
    }

    let any_failed = results.iter().any(|(_, s)| *s == FileStatus::Failed);
    Ok(if any_failed { 1 } else { 0 })
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
    fn defaults() {
        let args = parse(&["movie.mkv"]);
        assert_eq!(args.input, "movie.mkv");
        assert_eq!(args.batch_size, 4);
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
    fn metrics_flag_parses_and_has_warning() {
        let args = parse(&["movie.mkv", "--metrics"]);
        assert!(args.metrics, "--metrics must still be accepted");
        // warn helper is a no-op when the flag is unset (smoke test, no panic).
        warn_unimplemented_metrics(false);
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
    fn keep_artifacts_maps_into_config() {
        let args = parse(&["movie.mkv", "--keep-artifacts"]);
        assert!(args.keep_artifacts);
        let cfg = args.to_config("allegro".into(), vec![]);
        assert!(cfg.keep_artifacts);

        // Default off -> cleanup happens.
        let args = parse(&["movie.mkv"]);
        let cfg = args.to_config("allegro".into(), vec![]);
        assert!(!cfg.keep_artifacts);
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
