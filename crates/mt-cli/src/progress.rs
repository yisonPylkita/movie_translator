//! Functional progress display built on `indicatif`.
//!
//! Minimal port of the spirit of `movie_translator/progress.py::ProgressTracker`:
//! one overall bar tracking files completed plus per-file status lines. The Rust
//! pipeline orchestrator (`run_all`) currently exposes no per-stage progress
//! events, so we render overall file completion and emit per-file start/finish
//! lines. A richer per-stage TUI is a documented follow-up.

use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// A small progress UI: an overall bar over the file count.
pub struct Progress {
    multi: MultiProgress,
    overall: ProgressBar,
}

impl Progress {
    /// Create a progress display tracking `total` files.
    pub fn new(total: u64) -> Self {
        let multi = MultiProgress::new();
        let overall = multi.add(ProgressBar::new(total));
        overall.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} files {wide_msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
        );
        overall.enable_steady_tick(Duration::from_millis(120));
        Progress { multi, overall }
    }

    /// Announce a file starting.
    pub fn start_file(&self, name: &str) {
        self.overall.set_message(format!("→ {name}"));
    }

    /// Record a finished file with its status, advancing the overall bar.
    pub fn finish_file(&self, name: &str, status: &str) {
        let tag = match status {
            "success" => "✓",
            "skipped" => "⏭",
            "failed" => "✗",
            _ => "?",
        };
        self.multi.suspend(|| {
            eprintln!("{tag} {name} ({status})");
        });
        self.overall.inc(1);
    }

    /// Finish the overall bar.
    pub fn finish(&self) {
        self.overall.finish_and_clear();
    }
}
