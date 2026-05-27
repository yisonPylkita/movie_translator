//! Port of Python's `SubtitleProcessor` class.

use std::path::Path;

use mt_core::types::{replace_polish_chars, DialogueLine, NON_DIALOGUE_STYLES};

use crate::{
    ass::{load_ass, to_ass_string},
    model::{strip_ass_overrides, Event, EventKind, Subtitles},
    srt::{load_srt, to_srt_string},
};

/// Error type for subtitle processing failures.
#[derive(Debug)]
pub struct SubtitleProcessingError(pub String);

impl std::fmt::Display for SubtitleProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SubtitleProcessingError {}

impl SubtitleProcessingError {
    fn new(msg: impl Into<String>) -> Self {
        SubtitleProcessingError(msg.into())
    }
}

type Result<T> = std::result::Result<T, SubtitleProcessingError>;

/// Find the best dialogue style name in a `Subtitles`.
///
/// Port of Python's `_find_dialogue_style`. Logic:
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

/// Unified subtitle processor for parsing, writing, and validation.
pub struct SubtitleProcessor;

impl SubtitleProcessor {
    /// Extract dialogue lines from a subtitle file.
    ///
    /// Port of `SubtitleProcessor.extract_dialogue_lines`.
    pub fn extract_dialogue_lines(subtitle_file: &Path) -> Result<Vec<DialogueLine>> {
        if !subtitle_file.exists() {
            return Err(SubtitleProcessingError::new(format!(
                "Subtitle file not found: {}",
                subtitle_file.display()
            )));
        }

        let subs = load_file(subtitle_file)?;

        let unique_events = Self::deduplicate_events(subs.events);
        let dialogue_lines = Self::filter_dialogue(&unique_events);

        Ok(dialogue_lines)
    }

    /// Create a new subtitle file from dialogue lines.
    ///
    /// Port of `SubtitleProcessor.create_subtitle_file`.
    /// Copies `[Script Info]` and `[V4+ Styles]` from `original_file`.
    pub fn create_subtitle_file(
        original_file: &Path,
        dialogue_lines: &[DialogueLine],
        output_path: &Path,
        text_transform: Option<fn(&str) -> String>,
    ) -> Result<()> {
        if !original_file.exists() {
            return Err(SubtitleProcessingError::new(format!(
                "Original subtitle file not found: {}",
                original_file.display()
            )));
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
    ///
    /// Port of `SubtitleProcessor.create_english_subtitles`.
    pub fn create_english_subtitles(
        original_file: &Path,
        dialogue_lines: &[DialogueLine],
        output_path: &Path,
    ) -> Result<()> {
        Self::create_subtitle_file(original_file, dialogue_lines, output_path, None)
    }

    /// Create a Polish subtitle file with optional diacritic replacement.
    ///
    /// Port of `SubtitleProcessor.create_polish_subtitles`.
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
        Self::create_subtitle_file(original_file, translated_dialogue, output_path, transform)
    }

    /// Replace all font names in ASS styles with `new_font_name`.
    ///
    /// Port of `SubtitleProcessor.override_font_name`.
    /// The style raw field has fontname at index 1 (0-indexed after name).
    pub fn override_font_name(ass_file: &Path, new_font_name: &str) -> Result<()> {
        if !ass_file.exists() {
            return Err(SubtitleProcessingError::new(format!(
                "Subtitle file not found: {}",
                ass_file.display()
            )));
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
    /// Port of `SubtitleProcessor.validate_cleaned_subtitles`.
    /// Timing mismatches are logged as warnings but not fatal.
    pub fn validate_cleaned_subtitles(original_file: &Path, cleaned_file: &Path) -> Result<()> {
        let original_subs = load_file(original_file).map_err(|e| {
            SubtitleProcessingError::new(format!("Failed to load subtitle files: {e}"))
        })?;
        let cleaned_subs = load_file(cleaned_file).map_err(|e| {
            SubtitleProcessingError::new(format!("Failed to load subtitle files: {e}"))
        })?;

        let original_events: Vec<&Event> = original_subs
            .events
            .iter()
            .filter(|e| !e.text.trim().is_empty())
            .collect();
        let cleaned_events: Vec<&Event> = cleaned_subs
            .events
            .iter()
            .filter(|e| !e.text.trim().is_empty())
            .collect();

        let original_dialogue: Vec<&Event> = original_events
            .iter()
            .filter(|e| {
                let style_lower = e.style.to_lowercase();
                !NON_DIALOGUE_STYLES.iter().any(|kw| style_lower.contains(kw))
            })
            .copied()
            .collect();
        let non_dialogue_count = original_events.len() - original_dialogue.len();

        if original_dialogue.is_empty() {
            eprintln!("WARNING: No dialogue events found in original file");
            return Ok(());
        }

        if cleaned_events.is_empty() {
            return Err(SubtitleProcessingError::new(
                "Cleaned subtitle file has no events",
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
                eprintln!(
                    "INFO: start time offset: {start_diff:+}ms (expected — original has {non_dialogue_count} non-dialogue events)"
                );
            } else {
                eprintln!("WARNING: unexpected start time offset: {start_diff:+}ms");
            }
        }
        if end_diff.abs() > TOLERANCE_MS {
            if non_dialogue_count > 0 {
                eprintln!(
                    "INFO: end time offset: {end_diff:+}ms (expected — original has non-dialogue events after last dialogue)"
                );
            } else {
                eprintln!("WARNING: unexpected end time offset: {end_diff:+}ms");
            }
        }

        Ok(())
    }

    /// Remove duplicate consecutive events with the same plain text.
    ///
    /// Port of `SubtitleProcessor._deduplicate_events`.
    /// Consecutive events with identical plain text are merged: the kept event's
    /// `end_ms` = max of group `end_ms`, and its text is normalized to the stripped
    /// plain text (matching the Python consolidation). Events with plain text shorter
    /// than 2 characters are dropped.
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
                // Normalize the kept event's text to its stripped plain text,
                // matching Python's consolidated SSAEvent(text=clean_text).
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
    /// Port of `SubtitleProcessor._filter_dialogue`.
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

            let style_lower = event.style.to_lowercase();
            if NON_DIALOGUE_STYLES.iter().any(|kw| style_lower.contains(kw)) {
                continue;
            }

            let clean = strip_ass_overrides(&event.text).trim().to_string();
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
}

fn load_file(path: &Path) -> std::result::Result<Subtitles, SubtitleProcessingError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        SubtitleProcessingError::new(format!("failed to read {}: {e}", path.display()))
    })?;
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("ass") | Some("ssa") => load_ass(&content)
            .map_err(|e| SubtitleProcessingError::new(format!("Failed to parse subtitle file: {e}"))),
        Some("srt") => load_srt(&content)
            .map_err(|e| SubtitleProcessingError::new(format!("Failed to parse subtitle file: {e}"))),
        other => Err(SubtitleProcessingError::new(format!(
            "unsupported extension: {other:?}"
        ))),
    }
}

fn save_file(subs: &Subtitles, path: &Path) -> std::result::Result<(), SubtitleProcessingError> {
    let content = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("srt") => to_srt_string(subs),
        _ => to_ass_string(subs),
    };
    std::fs::write(path, content)
        .map_err(|e| SubtitleProcessingError::new(format!("Failed to save subtitle file: {e}")))
}
