//! Translation pipeline orchestration: stages, GPU abstraction, deferred-OCR
//! resolution.
//!
//! This crate runs the 7 sequential pipeline stages plus the deferred-OCR
//! resolution logic.
//!
//! # Module boundary
//!
//! - [`stages`] — `identify`, `extract_ref`, `fetch`, `extract_english`,
//!   `translate`, `create_tracks`, `mux`. Each is a free `run(...)` function.
//! - [`gpu`] — the [`gpu::GpuExecutor`] trait (mirrors `mt_ml`), the inline
//!   [`gpu::DirectGpuExecutor`], and [`gpu::resolve_pending_ocr`].
//! - [`worker`] — the tokio [`worker::GpuWorker`] serialising all GPU work
//!   through a single task.
//! - [`orchestrator`] — the async [`orchestrator::process_file`] /
//!   [`orchestrator::run_all`] and the synchronous
//!   [`orchestrator::process_video_file`].
//! - [`error`] — [`error::PipelineError`] / [`error::Result`].
//! - [`vision`] — Vision-OCR availability probe used by the extract stages.

pub mod error;
pub mod gpu;
pub mod orchestrator;
pub mod progress;
pub mod proper_nouns;
pub mod stages;
pub mod vision;
pub mod worker;

pub use error::{PipelineError, Result};
pub use gpu::{resolve_pending_ocr, DirectGpuExecutor, GpuExecutor, OcrStageLabel};
pub use orchestrator::{
    process_file, process_video_file, run_all, run_all_with_progress, FileStatus,
};
pub use progress::{FinishStatus, ProgressEvent, ProgressSender, Stage};
pub use proper_nouns::extract_proper_nouns_from_subtitles;
pub use vision::{default_vision_ocr_probe, VisionOcrProbe};
pub use worker::{GpuWorker, GpuWorkerHandle};
