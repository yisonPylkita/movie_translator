//! Pipeline stages — one module per stage, run sequentially per file.
//!
//! Each stage exposes a `run(ctx, deps...) -> Result<PipelineContext>` function
//! and a `NAME` constant giving the stage's role name. Stages never
//! perform GPU work inline: translation goes through a [`crate::gpu::GpuExecutor`]
//! and OCR is deferred via [`mt_core::PendingOcr`] (resolved by the orchestrator
//! through [`crate::gpu::resolve_pending_ocr`]).

pub mod create_tracks;
pub mod extract_english;
pub mod extract_ref;
pub mod fetch;
pub mod hardsub_ocr;
pub mod identify;
pub mod mux;
pub mod transcribe;
pub mod translate;
