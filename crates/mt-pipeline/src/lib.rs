//! Translation pipeline orchestration: stages, GPU abstraction, deferred-OCR
//! resolution.
//!
//! This crate ports the 7 sequential pipeline stages from
//! `movie_translator/stages/` plus the deferred-OCR resolution logic from the
//! Python orchestrators (`pipeline.py` / `async_pipeline.py`).
//!
//! # Module boundary
//!
//! The stages and the GPU abstraction live here. The orchestrators themselves
//! (`process_file` / `run_all` / the tokio GPU worker) are implemented in a
//! later dispatch — this crate deliberately stops at the stage boundary.
//!
//! - [`stages`] — `identify`, `extract_ref`, `fetch`, `extract_english`,
//!   `translate`, `create_tracks`, `mux`. Each is a free `run(...)` function.
//! - [`gpu`] — the [`gpu::GpuExecutor`] trait (mirrors `mt_ml`), the inline
//!   [`gpu::DirectGpuExecutor`], and [`gpu::resolve_pending_ocr`] (the
//!   `_handle_pending_ocr` / `_resolve_pending_ocr` port).
//! - [`error`] — [`error::PipelineError`] / [`error::Result`].
//! - [`vision`] — Vision-OCR availability probe used by the extract stages.

pub mod error;
pub mod gpu;
pub mod stages;
pub mod vision;

pub use error::{PipelineError, Result};
pub use gpu::{resolve_pending_ocr, DirectGpuExecutor, GpuExecutor, OcrStageLabel};
pub use vision::{default_vision_ocr_probe, VisionOcrProbe};
