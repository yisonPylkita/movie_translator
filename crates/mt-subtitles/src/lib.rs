//! Subtitle parsing, writing, and manipulation (SRT, ASS, etc.).
//!
//! # ASS Parser Decision: Hand-Rolled
//!
//! ## Decision
//! Use a **hand-rolled ASS parser**. `subparse` v0.7.0 was evaluated and rejected.
//!
//! ## Fidelity Results (corpus: 11 anime fansubbing ASS files)
//!
//! ### Spike A — Hand-Rolled Parser
//! - Files parsed:                  11/11
//! - Files with all fields correct: 11/11 (PERFECT)
//! - Total events compared:         4141
//! - Event field mismatches:        0
//! - Fields verified per-event:     start_ms, end_ms, style, text, type (Dialogue/Comment), layer
//! - Styles verified:               correct count and names on all 11 files
//!
//! ### Spike B — `subparse` v0.7.0
//! - Files parsed:  11/11 (no parse errors)
//! - Styles access: **NONE** — no API exists in `SsaFile` to enumerate or inspect `[V4+ Styles]`
//! - Per-event Style field: **NONE** — `SubtitleEntry` only exposes `(timespan, text)`
//! - Comment events: **NOT ACCESSIBLE** — treated as opaque `Filler` bytes
//! - Event count mismatch on `onepace_arlongpark_01_pl.ass`:
//!   subparse returned 540 Dialogue lines; pysubs2 returned 555 (540 Dialogue + 15 Comment).
//!   The 15 Comment events are invisible to subparse.
//!
//! ## Rationale for Rejection of subparse
//!
//! The mt-subtitles workflow requires:
//! 1. Read and write `[V4+ Styles]` (copy all styles from source to translated file).
//! 2. Per-event `Style` field (to route each event to the correct display style).
//! 3. Per-event `Layer`, `Name`, `MarginL/R/V`, `Effect` (preserved verbatim).
//! 4. `Comment:` events (chapter markers, editor notes) preserved as structured events.
//! 5. `{\override}` tags in `Text` preserved exactly.
//!
//! `subparse` provides none of (1), (2), (3), (4). Its `SsaFilePart` enum treats everything
//! except `(Start, End, Text)` as opaque `Filler`, making it impossible to read or write
//! per-event Style or any `[V4+ Styles]` data. Additionally, it pulls in deprecated crates
//! (nom v2.1.0, combine v2.5.2, failure v0.1.8) which introduce future-incompatibility warnings.
//!
//! ## Implementation Notes (for the full parser, a later task)
//! - ASS timing: `H:MM:SS.cs` (centiseconds); convert cs×10 → ms to match pysubs2.
//! - The `[Aegisub Project Garbage]` and `[Aegisub Extradata]` sections must be preserved
//!   verbatim (they appear in real-world fansub files).
//! - BOM (`\u{FEFF}`) must be handled.
//! - `Text` field is the last comma-separated field and may contain commas: use `splitn(n, ',')`.
//! - See `examples/spike_handroll.rs` for a working minimal implementation as evidence.

pub mod ass;
pub mod encoding;
pub mod error;
pub mod model;
pub mod processor;
pub mod srt;

#[cfg(test)]
mod tests;

use std::path::Path;

pub use encoding::normalize_encoding;
pub use error::ParseError;
pub use model::{AssTime, Event, EventKind, RawSection, Style, Subtitles};
pub use processor::{find_dialogue_style, SubtitleProcessingError, SubtitleProcessor};

/// Load a subtitle file by path, dispatching on extension (`.ass`/`.ssa` or `.srt`).
pub fn load(path: &Path) -> Result<Subtitles, ParseError> {
    let content = std::fs::read_to_string(path)?;
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("ass") | Some("ssa") => ass::load_ass(&content),
        Some("srt") => srt::load_srt(&content),
        other => Err(ParseError::UnsupportedExtension(other.map(String::from))),
    }
}
