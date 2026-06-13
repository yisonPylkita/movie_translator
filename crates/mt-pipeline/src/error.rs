//! Pipeline-level error type.
//!
//! Composes the structured error types from the lower `mt-*` crates so the
//! underlying cause always propagates (and is visible in the `Error::source`
//! chain), rather than being flattened into a string. A small number of
//! genuine precondition/invariant failures are carried by
//! [`PipelineError::Stage`].

use thiserror::Error;

/// Errors produced while running a pipeline stage.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Core errors (media identification, path resolution, and — because the
    /// `mt-ml` drivers return `mt_core::Result` — translation/OCR/inpaint
    /// subprocess failures, which carry their stderr/exit context).
    #[error("core operation failed: {0}")]
    Core(#[from] mt_core::MtError),

    /// Subtitle extraction from a video container failed.
    #[error("subtitle extraction failed: {0}")]
    Extraction(#[from] mt_media::SubtitleExtractionError),

    /// A high-level video operation (mux + verify) failed.
    #[error("video operation failed: {0}")]
    VideoOperation(#[from] mt_media::VideoOperationError),

    /// A low-level ffmpeg/mkvmerge mux operation failed.
    #[error("mux failed: {0}")]
    Mux(#[from] mt_media::VideoMuxError),

    /// Subtitle processing (load / clean / re-style) failed.
    #[error("subtitle processing failed: {0}")]
    Subtitle(#[from] mt_subtitles::SubtitleProcessingError),

    /// Parsing subtitle text failed.
    #[error("subtitle parse failed: {0}")]
    Parse(#[from] mt_subtitles::ParseError),

    /// Fetching a subtitle from a remote provider failed.
    #[error("subtitle fetch failed: {0}")]
    Fetch(#[from] mt_fetch::retry::FetchError),

    /// Re-timing/aligning a fetched subtitle against the reference failed.
    #[error("subtitle alignment failed: {0}")]
    Align(#[from] mt_fetch::align::AlignError),

    /// I/O error not already attributable to a more specific operation.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A stage precondition was not met or produced no usable result (e.g. no
    /// English subtitle source, no dialogue lines). Reserved for genuine
    /// invariant failures — real underlying errors propagate with their cause
    /// via the variants above.
    #[error("{0}")]
    Stage(String),

    /// No English subtitle source could be obtained for this file (no embedded
    /// English track, no PGS, no fetched subtitle, no burned-in OCR yield).
    /// This is treated as a *skip*, not a failure, because it usually means the
    /// file is intentionally dialogue-less (NC/OP/ED clips, music videos).
    #[error("no English subtitle source available")]
    NoEnglishSource,
}

impl PipelineError {
    /// True if this error should mark the file as "skipped (no subtitles)"
    /// rather than "failed". Used by the orchestrator to translate stage
    /// failures into the appropriate per-file [`crate::FileStatus`].
    pub fn is_no_english_source(&self) -> bool {
        matches!(self, PipelineError::NoEnglishSource)
    }
}

/// Convenience `Result` alias for pipeline stages.
pub type Result<T> = std::result::Result<T, PipelineError>;

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io::{Error as IoError, ErrorKind};

    use super::*;

    /// A wrapped lower-crate error must keep its cause reachable via
    /// `Error::source`, not be stringified into the message.
    #[test]
    fn carries_source_chain_for_io() {
        let io = IoError::new(ErrorKind::PermissionDenied, "denied");
        let err: PipelineError = io.into();
        assert!(
            err.source().is_some(),
            "io cause must be in the source chain"
        );
    }

    #[test]
    fn carries_source_chain_for_parse() {
        let pe = mt_subtitles::ParseError::Malformed {
            detail: "bad".into(),
            line_no: Some(3),
        };
        let err: PipelineError = pe.into();
        assert!(
            err.source().is_some(),
            "parse cause must be in source chain"
        );
        assert!(err.to_string().contains("subtitle parse failed"));
    }

    /// `Stage` is reserved for precondition messages and has no source cause.
    #[test]
    fn stage_is_message_only() {
        let err = PipelineError::Stage("No English subtitle source".into());
        assert!(err.source().is_none());
        assert_eq!(err.to_string(), "No English subtitle source");
    }

    /// `NoEnglishSource` is a distinct variant the orchestrator routes to
    /// `Skipped (no subtitles)` rather than `Failed`.
    #[test]
    fn no_english_source_predicate() {
        let err = PipelineError::NoEnglishSource;
        assert!(err.is_no_english_source());
        assert!(!PipelineError::Stage("x".into()).is_no_english_source());
    }
}
