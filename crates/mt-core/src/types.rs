use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Styles that indicate non-dialogue content (signs, songs, etc.).
/// Covers common fansub naming: OP/ED/IN (insert song) layers with
/// romaji (OPRO/INRO) and English (OPEN/INEN) suffixes.
pub const NON_DIALOGUE_STYLES: &[&str] =
    &["sign", "song", "title", "op", "ed", "insert", "inro", "inen"];

/// Polish diacritical characters.
pub const POLISH_CHARS: &str = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ";

/// Replace Polish diacritical characters with ASCII equivalents.
pub fn replace_polish_chars(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'ą' => 'a',
            'ć' => 'c',
            'ę' => 'e',
            'ł' => 'l',
            'ń' => 'n',
            'ó' => 'o',
            'ś' => 's',
            'ź' => 'z',
            'ż' => 'z',
            'Ą' => 'A',
            'Ć' => 'C',
            'Ę' => 'E',
            'Ł' => 'L',
            'Ń' => 'N',
            'Ó' => 'O',
            'Ś' => 'S',
            'Ź' => 'Z',
            'Ż' => 'Z',
            other => other,
        })
        .collect()
}

/// A single line of dialogue with start/end timestamps in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DialogueLine {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// A subtitle file with associated metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubtitleFile {
    pub path: PathBuf,
    pub language: String,
    pub title: String,
    pub is_default: bool,
}

/// A normalized bounding box (values in [0, 1], top-left origin).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Normalized left edge (0–1).
    pub x: f32,
    /// Normalized top edge (0–1, top-left origin).
    pub y: f32,
    /// Normalized width (0–1).
    pub width: f32,
    /// Normalized height (0–1).
    pub height: f32,
}

/// OCR result for a single video frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OCRResult {
    pub timestamp_ms: i64,
    pub text: String,
    pub boxes: Vec<BoundingBox>,
}

/// The output of a burned-in subtitle extraction pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurnedInResult {
    pub srt_path: PathBuf,
    pub ocr_results: Vec<OCRResult>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_polish_chars_typical() {
        assert_eq!(replace_polish_chars("Cześć ŻÓŁW"), "Czesc ZOLW");
    }

    #[test]
    fn replace_polish_chars_ascii_unchanged() {
        let ascii = "Hello, world! 123";
        assert_eq!(replace_polish_chars(ascii), ascii);
    }

    #[test]
    fn replace_polish_chars_empty() {
        assert_eq!(replace_polish_chars(""), "");
    }

    #[test]
    fn dialogue_line_serde_round_trip() {
        let line = DialogueLine {
            start_ms: 1000,
            end_ms: 3500,
            text: "Cześć!".to_string(),
        };
        let json = serde_json::to_string(&line).expect("serialize");
        let back: DialogueLine = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(line, back);
    }

    #[test]
    fn non_dialogue_styles_contains_expected() {
        assert!(NON_DIALOGUE_STYLES.contains(&"sign"));
        assert!(NON_DIALOGUE_STYLES.contains(&"inen"));
        assert!(!NON_DIALOGUE_STYLES.contains(&"dialogue"));
    }
}
