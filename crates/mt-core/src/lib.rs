//! Core types, errors, and utilities shared across all mt-* crates.

pub mod context;
pub mod error;
pub mod exec;
pub mod identity;
pub mod types;

// Re-export the most-used items at crate root for ergonomic imports.
pub use context::{
    FetchedSubtitle, FontInfo, OriginalTrack, PendingOcr, PipelineConfig, PipelineContext,
};
pub use error::{MtError, Result};
pub use identity::MediaIdentity;
pub use types::replace_polish_chars;
pub use types::{
    is_non_dialogue_style, BoundingBox, BurnedInResult, DialogueLine, OCRResult, SubtitleFile,
    NON_DIALOGUE_STYLES, POLISH_CHARS,
};
