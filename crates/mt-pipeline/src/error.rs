//! Pipeline-level error type.
//!
//! Wraps the various error types from the lower `mt-*` crates plus a few
//! pipeline-specific failure modes (mirroring the Python `RuntimeError`s
//! raised by individual stages).

use thiserror::Error;

/// Errors produced while running a pipeline stage.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("media identification failed: {0}")]
    Identify(#[from] mt_core::MtError),

    #[error("subtitle extraction failed: {0}")]
    Extraction(#[from] mt_media::SubtitleExtractionError),

    #[error("video operation failed: {0}")]
    VideoOperation(#[from] mt_media::VideoOperationError),

    #[error("subtitle processing failed: {0}")]
    Subtitle(#[from] mt_subtitles::SubtitleProcessingError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A stage precondition was not met or produced no usable result. Mirrors
    /// the `RuntimeError(...)` raised by the Python stages.
    #[error("{0}")]
    Stage(String),
}

/// Convenience `Result` alias for pipeline stages.
pub type Result<T> = std::result::Result<T, PipelineError>;
