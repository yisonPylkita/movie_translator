//! Structured progress events emitted by the pipeline.
//!
//! Stages and the orchestrator emit [`ProgressEvent`]s into a
//! [`ProgressSender`]; the CLI consumes them and renders the TUI. The pipeline
//! never touches the rendering layer directly, so headless callers (tests,
//! `process_video_file`) can ignore events entirely.
//!
//! The sender is an optional `tokio::sync::mpsc::UnboundedSender<ProgressEvent>`
//! wrapped in [`ProgressSender`]; if `None`, all `emit_*` helpers no-op.

use std::path::PathBuf;

use tokio::sync::mpsc;

/// Which pipeline stage a `StageEntered` / progress event came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    Identify,
    ExtractRef,
    Fetch,
    ExtractEnglish,
    Translate,
    CreateTracks,
    Mux,
}

impl Stage {
    /// Compact human label used in TUI rows.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Identify => "identify",
            Stage::ExtractRef => "extract-ref",
            Stage::Fetch => "fetch",
            Stage::ExtractEnglish => "extract-eng",
            Stage::Translate => "translate",
            Stage::CreateTracks => "tracks",
            Stage::Mux => "mux",
        }
    }
}

/// Outcome reported by `FileFinished`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishStatus {
    Success,
    Failed,
    Skipped,
    /// Skipped because no English subtitle source could be obtained (NC/OP/ED
    /// clips, music videos, etc.). Distinguished from `Skipped` so the TUI /
    /// summary can report the reason.
    SkippedNoSubs,
}

impl FinishStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishStatus::Success => "success",
            FinishStatus::Failed => "failed",
            FinishStatus::Skipped => "skipped",
            FinishStatus::SkippedNoSubs => "skipped (no subtitles)",
        }
    }
}

/// One progress event emitted by the pipeline. Extensible: new variants do not
/// break consumers that wildcard-match the enum.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Static list of files the run will attempt — emitted once at the start
    /// of `run_all`. The TUI uses this to pre-populate "queued" rows.
    Queued {
        files: Vec<PathBuf>,
    },
    FileStarted {
        path: PathBuf,
    },
    StageEntered {
        path: PathBuf,
        stage: Stage,
    },
    OcrProgress {
        path: PathBuf,
        done: u64,
        total: u64,
    },
    FetchResult {
        path: PathBuf,
        candidates_found: u32,
        downloaded: u32,
    },
    TranslateBatch {
        path: PathBuf,
        lines_done: u64,
        lines_total: u64,
        model: String,
    },
    /// A free-form log line (forwarded from tracing or Python stderr).
    Log {
        level: String,
        target: String,
        message: String,
    },
    FileFinished {
        path: PathBuf,
        status: FinishStatus,
    },
}

/// Optional sink for `ProgressEvent`s.
///
/// Cheap to clone (wraps an `Arc`-internal mpsc sender). `None` disables event
/// emission entirely — every `send` becomes a no-op so the pipeline can run
/// headless (tests, sync `process_video_file`).
#[derive(Debug, Clone, Default)]
pub struct ProgressSender(Option<mpsc::UnboundedSender<ProgressEvent>>);

impl ProgressSender {
    /// A sink that drops every event. Equivalent to `Self::default()`.
    pub fn disabled() -> Self {
        Self(None)
    }

    /// Wrap an mpsc sender.
    pub fn new(tx: mpsc::UnboundedSender<ProgressEvent>) -> Self {
        Self(Some(tx))
    }

    /// True if this sender is wired to a receiver.
    pub fn is_enabled(&self) -> bool {
        self.0.is_some()
    }

    /// Send an event. Errors (closed channel) are silently ignored — the
    /// pipeline must not fail because the UI went away.
    pub fn send(&self, event: ProgressEvent) {
        if let Some(tx) = self.0.as_ref() {
            let _ = tx.send(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_sender_is_a_noop() {
        let s = ProgressSender::disabled();
        assert!(!s.is_enabled());
        s.send(ProgressEvent::FileStarted {
            path: PathBuf::from("x"),
        });
    }

    #[test]
    fn enabled_sender_delivers_events_in_order() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let s = ProgressSender::new(tx);
        assert!(s.is_enabled());
        s.send(ProgressEvent::FileStarted {
            path: PathBuf::from("a"),
        });
        s.send(ProgressEvent::FileFinished {
            path: PathBuf::from("a"),
            status: FinishStatus::Success,
        });
        let ev1 = rx.try_recv().unwrap();
        let ev2 = rx.try_recv().unwrap();
        assert!(matches!(ev1, ProgressEvent::FileStarted { .. }));
        assert!(matches!(
            ev2,
            ProgressEvent::FileFinished {
                status: FinishStatus::Success,
                ..
            }
        ));
    }

    #[test]
    fn stage_labels_are_unique_short_strings() {
        let labels = [
            Stage::Identify.label(),
            Stage::ExtractRef.label(),
            Stage::Fetch.label(),
            Stage::ExtractEnglish.label(),
            Stage::Translate.label(),
            Stage::CreateTracks.label(),
            Stage::Mux.label(),
        ];
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "labels must be unique");
        for label in labels {
            assert!(!label.is_empty());
            assert!(label.len() <= 16);
        }
    }

    #[test]
    fn finish_status_strings() {
        assert_eq!(FinishStatus::Success.as_str(), "success");
        assert_eq!(FinishStatus::Failed.as_str(), "failed");
        assert_eq!(FinishStatus::Skipped.as_str(), "skipped");
        assert_eq!(
            FinishStatus::SkippedNoSubs.as_str(),
            "skipped (no subtitles)"
        );
    }
}
