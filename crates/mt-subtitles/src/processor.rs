//! Subtitle processing: dialogue extraction, file creation, font overrides,
//! and validation.

use std::path::Path;

use mt_core::types::{DialogueLine, is_non_dialogue_style, replace_polish_chars};
use thiserror::Error;
use tracing::{info, warn};

use crate::{
    ass::{load_ass, to_ass_string},
    error::ParseError,
    model::{Event, EventKind, Subtitles, strip_ass_overrides},
    srt::{load_srt, to_srt_string},
};

/// Error type for subtitle processing failures.
#[derive(Debug, Error)]
pub enum SubtitleProcessingError {
    /// The requested subtitle file does not exist on disk.
    #[error("subtitle file not found: {0}")]
    NotFound(String),

    /// Parsing the subtitle content failed; the underlying [`ParseError`] is
    /// preserved as the error source.
    #[error("failed to parse subtitle file: {0}")]
    Parse(#[from] ParseError),

    /// Writing the subtitle file failed.
    #[error("failed to save subtitle file: {0}")]
    Save(#[source] std::io::Error),

    /// A validation precondition was violated (e.g. cleaned file has no events).
    #[error("{0}")]
    Validation(String),
}

type Result<T> = std::result::Result<T, SubtitleProcessingError>;

/// Find the best dialogue style name in a `Subtitles`.
///
/// Selection logic:
/// 1. If no styles → `"Default"`
/// 2. If `"Default"` exists → `"Default"`
/// 3. Look for `"Dialogue"`, `"Dialog"`, `"Main"`, `"Dialogi"`, `"Normal"` (case-sensitive)
/// 4. Case-insensitive search for any name containing `"dialog"`, `"default"`, or `"main"`
/// 5. Last resort: first style name
pub fn find_dialogue_style(subs: &Subtitles) -> &str {
    if subs.styles.is_empty() {
        return "Default";
    }

    // Prefer "Default" if present
    if subs.styles.iter().any(|s| s.name == "Default") {
        return "Default";
    }

    // Common dialogue names (case-sensitive)
    let preferred = ["Dialogue", "Dialog", "Main", "Dialogi", "Normal"];
    for &name in &preferred {
        if let Some(style) = subs.styles.iter().find(|s| s.name == name) {
            return &style.name;
        }
    }

    // Case-insensitive fallback
    for style in &subs.styles {
        let lower = style.name.to_lowercase();
        if lower.contains("dialog") || lower.contains("default") || lower.contains("main") {
            return &style.name;
        }
    }

    // Last resort: first style
    &subs.styles[0].name
}

/// Unified subtitle processor functions for parsing, writing, and validation.
/// All functions are free-standing (no state needed).
/// Extract dialogue lines from a subtitle file.
pub fn extract_dialogue_lines(subtitle_file: &Path) -> Result<Vec<DialogueLine>> {
    if !subtitle_file.exists() {
        return Err(SubtitleProcessingError::NotFound(
            subtitle_file.display().to_string(),
        ));
    }

    let subs = load_file(subtitle_file)?;

    let unique_events = deduplicate_events(subs.events);
    let dialogue_lines = filter_dialogue(&unique_events);

    Ok(dialogue_lines)
}

/// Create a new subtitle file from dialogue lines.
///
/// Copies `[Script Info]` and `[V4+ Styles]` from `original_file`.
pub fn create_subtitle_file(
    original_file: &Path,
    dialogue_lines: &[DialogueLine],
    output_path: &Path,
    text_transform: Option<fn(&str) -> String>,
) -> Result<()> {
    if !original_file.exists() {
        return Err(SubtitleProcessingError::NotFound(
            original_file.display().to_string(),
        ));
    }

    let mut subs = load_file(original_file)?;

    let dialogue_style = find_dialogue_style(&subs).to_string();

    // Replace events with the provided dialogue lines.
    subs.events = dialogue_lines
        .iter()
        .map(|line| {
            let mut text = line.text.clone();
            if let Some(transform) = text_transform {
                text = transform(&text);
            }
            // \n in text → \N (ASS hard line break)
            text = text.replace('\n', "\\N");
            Event {
                kind: EventKind::Dialogue,
                layer: 0,
                start_ms: line.start_ms,
                end_ms: line.end_ms,
                style: dialogue_style.clone(),
                name: String::new(),
                margin_l: 0,
                margin_r: 0,
                margin_v: 0,
                effect: String::new(),
                text,
            }
        })
        .collect();

    save_file(&subs, output_path)?;
    Ok(())
}

/// Create a clean English subtitle file (no transform).
pub fn create_english_subtitles(
    original_file: &Path,
    dialogue_lines: &[DialogueLine],
    output_path: &Path,
) -> Result<()> {
    create_subtitle_file(original_file, dialogue_lines, output_path, None)
}

/// Create a Polish subtitle file with optional diacritic replacement.
pub fn create_polish_subtitles(
    original_file: &Path,
    translated_dialogue: &[DialogueLine],
    output_path: &Path,
    replace_chars: bool,
) -> Result<()> {
    let transform: Option<fn(&str) -> String> = if replace_chars {
        Some(replace_polish_chars)
    } else {
        None
    };
    create_subtitle_file(original_file, translated_dialogue, output_path, transform)
}

/// Replace all font names in ASS styles with `new_font_name`.
///
/// The style raw field has fontname at index 1 (0-indexed after name).
pub fn override_font_name(ass_file: &Path, new_font_name: &str) -> Result<()> {
    if !ass_file.exists() {
        return Err(SubtitleProcessingError::NotFound(
            ass_file.display().to_string(),
        ));
    }

    let mut subs = load_file(ass_file)?;

    for style in &mut subs.styles {
        let mut owned: Vec<String> = style.raw.split(',').map(|s| s.to_string()).collect();
        // owned[0] = Name, owned[1] = Fontname
        if owned.len() > 1 {
            owned[1] = new_font_name.to_string();
            style.raw = owned.join(",");
        }
    }

    save_file(&subs, ass_file)?;
    Ok(())
}

/// Validate that cleaned subtitles maintain proper timing coverage.
///
/// Timing mismatches are logged as warnings but not fatal.
pub fn validate_cleaned_subtitles(original_file: &Path, cleaned_file: &Path) -> Result<()> {
    let original_subs = load_file(original_file)?;
    let cleaned_subs = load_file(cleaned_file)?;

    let original_events: Vec<_> = original_subs
        .events
        .iter()
        .filter(|e| !e.text.trim().is_empty())
        .collect();
    let cleaned_events: Vec<_> = cleaned_subs
        .events
        .iter()
        .filter(|e| !e.text.trim().is_empty())
        .collect();

    let original_dialogue: Vec<_> = original_events
        .iter()
        .filter(|e| !is_non_dialogue_style(&e.style))
        .copied()
        .collect();
    let non_dialogue_count = original_events.len() - original_dialogue.len();

    if original_dialogue.is_empty() {
        warn!("No dialogue events found in original file");
        return Ok(());
    }

    if cleaned_events.is_empty() {
        return Err(SubtitleProcessingError::Validation(
            "Cleaned subtitle file has no events".to_string(),
        ));
    }

    let original_start = original_dialogue.iter().map(|e| e.start_ms).min().unwrap();
    let original_end = original_dialogue.iter().map(|e| e.end_ms).max().unwrap();
    let cleaned_start = cleaned_events.iter().map(|e| e.start_ms).min().unwrap();
    let cleaned_end = cleaned_events.iter().map(|e| e.end_ms).max().unwrap();

    const TOLERANCE_MS: i64 = 50;
    let start_diff = cleaned_start - original_start;
    let end_diff = cleaned_end - original_end;

    if start_diff.abs() > TOLERANCE_MS {
        if non_dialogue_count > 0 {
            info!(
                start_diff = start_diff,
                non_dialogue_count = non_dialogue_count,
                "start time offset (expected — original has non-dialogue events)"
            );
        } else {
            warn!("unexpected start time offset: {start_diff:+}ms");
        }
    }
    if end_diff.abs() > TOLERANCE_MS {
        if non_dialogue_count > 0 {
            info!(
                end_diff = end_diff,
                non_dialogue_count = non_dialogue_count,
                "end time offset (expected — original has non-dialogue events after last dialogue)"
            );
        } else {
            warn!("unexpected end time offset: {end_diff:+}ms");
        }
    }

    Ok(())
}

/// Remove duplicate consecutive events with the same plain text.
///
/// Consecutive events with identical plain text are merged: the kept event's
/// `end_ms` = max of group `end_ms`, and its text is normalized to the stripped
/// plain text. Events with plain text shorter than 2 characters are dropped.
pub fn deduplicate_events(events: Vec<Event>) -> Vec<Event> {
    let mut unique: Vec<Event> = Vec::new();
    let mut last_text: Option<String> = None;
    let mut group_end: i64 = 0;

    for mut event in events {
        let clean_text = strip_ass_overrides(&event.text).trim().to_string();

        if clean_text.is_empty() || clean_text.chars().count() < 2 {
            continue;
        }

        if last_text.as_deref() == Some(clean_text.as_str()) {
            group_end = group_end.max(event.end_ms);
            if let Some(last) = unique.last_mut() {
                last.end_ms = group_end;
            }
        } else {
            // Normalize the kept event's text to its stripped plain text.
            event.text = clean_text.clone();
            last_text = Some(clean_text);
            group_end = event.end_ms;
            unique.push(event);
        }
    }

    unique
}

/// Filter events to dialogue only, converting to `DialogueLine`.
///
/// Skips events with:
/// - empty text
/// - style matching any `NON_DIALOGUE_STYLES` keyword (case-insensitive)
/// - plain text (after stripping override tags) that is empty
fn filter_dialogue(events: &[Event]) -> Vec<DialogueLine> {
    let mut result = Vec::new();

    for event in events {
        if event.text.trim().is_empty() {
            continue;
        }

        if is_non_dialogue_style(&event.style) {
            continue;
        }

        // Match pysubs2 `SSAEvent.plaintext`: strip override tags, then
        // convert ASS hard breaks (`\N`) and soft breaks (`\n`) into real
        // newlines so translation models do not receive literal `\N` tokens.
        let clean = strip_ass_overrides(&event.text)
            .replace("\\N", "\n")
            .replace("\\n", "\n")
            .trim()
            .to_string();
        if clean.is_empty() {
            continue;
        }

        result.push(DialogueLine {
            start_ms: event.start_ms,
            end_ms: event.end_ms,
            text: clean,
        });
    }

    result
}

fn load_file(path: &Path) -> Result<Subtitles> {
    let content = std::fs::read_to_string(path).map_err(ParseError::Io)?;
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("ass") | Some("ssa") => Ok(load_ass(&content)?),
        Some("srt") => Ok(load_srt(&content)?),
        other => Err(ParseError::UnsupportedExtension(other.map(String::from)).into()),
    }
}

fn save_file(subs: &Subtitles, path: &Path) -> Result<()> {
    let content = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("srt") => to_srt_string(subs),
        _ => to_ass_string(subs),
    };
    std::fs::write(path, content).map_err(SubtitleProcessingError::Save)
}
