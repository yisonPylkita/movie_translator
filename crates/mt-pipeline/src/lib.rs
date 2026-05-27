//! Translation pipeline orchestration: stages, GPU abstraction, deferred-OCR
//! resolution.
//!
//! This crate ports the 7 sequential pipeline stages from
//! `movie_translator/stages/` plus the deferred-OCR resolution logic from the
//! Python orchestrators (`pipeline.py` / `async_pipeline.py`).
//!
//! # Module boundary
//!
//! - [`stages`] — `identify`, `extract_ref`, `fetch`, `extract_english`,
//!   `translate`, `create_tracks`, `mux`. Each is a free `run(...)` function.
//! - [`gpu`] — the [`gpu::GpuExecutor`] trait (mirrors `mt_ml`), the inline
//!   [`gpu::DirectGpuExecutor`], and [`gpu::resolve_pending_ocr`] (the
//!   `_handle_pending_ocr` / `_resolve_pending_ocr` port).
//! - [`worker`] — the tokio [`worker::GpuWorker`] serialising all GPU work
//!   through a single task (port of `gpu_queue.py`).
//! - [`orchestrator`] — [`orchestrator::process_file`] / [`orchestrator::run_all`]
//!   (port of `async_pipeline.py`) and the synchronous
//!   [`orchestrator::process_video_file`] (port of `pipeline.py`).
//! - [`error`] — [`error::PipelineError`] / [`error::Result`].
//! - [`vision`] — Vision-OCR availability probe used by the extract stages.

pub mod error;
pub mod gpu;
pub mod orchestrator;
pub mod proper_nouns;
pub mod stages;
pub mod vision;
pub mod worker;

pub use error::{PipelineError, Result};
pub use proper_nouns::extract_proper_nouns_from_subtitles;
pub use gpu::{resolve_pending_ocr, DirectGpuExecutor, GpuExecutor, OcrStageLabel};
pub use orchestrator::{process_file, process_video_file, run_all, FileStatus};
pub use vision::{default_vision_ocr_probe, VisionOcrProbe};
pub use worker::{GpuWorker, GpuWorkerHandle};
