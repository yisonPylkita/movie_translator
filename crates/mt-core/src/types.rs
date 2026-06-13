use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Styles that indicate non-dialogue content (signs, songs, etc.).
/// Covers common fansub naming: OP/ED/IN (insert song) layers. Because
/// matching is token-based (split on `-`/`_`/space/digit), separated forms
/// like `OP-RO` / `OP-EN` match via the `op` token, and `INRO` / `INEN`
/// match as explicit keywords. Note: glued forms like `OPRO`/`OPEN` are NOT
/// keywords and so are NOT matched (avoids matching a legit style named
/// "Open").
pub const NON_DIALOGUE_STYLES: &[&str] = &[
    "sign", "signs", "song", "songs", "title", "titles", "op", "ed", "insert", "inro", "inen",
];

/// Returns `true` if `name` is a non-dialogue style (sign, song, OP/ED, etc.).
///
/// Uses WHOLE-WORD/token matching rather than substring matching: the style
/// name is split into tokens on `-`, `_`, whitespace, and digit boundaries,
/// and a token must *equal* one of [`NON_DIALOGUE_STYLES`] (case-insensitive)
/// to count. This prevents abbreviations like `op`/`ed` from matching inside
/// unrelated words such as "Top", "Opening", "Named", or "Graded".
///
/// # Examples
/// ```
/// use mt_core::types::is_non_dialogue_style;
/// assert!(is_non_dialogue_style("OP-Romaji"));
/// assert!(is_non_dialogue_style("Sign"));
/// assert!(is_non_dialogue_style("Insert Song"));
/// assert!(!is_non_dialogue_style("Dialogue Top")); // "Top" is not "op"
/// assert!(!is_non_dialogue_style("Named"));        // "Named" is not "ed"
/// assert!(!is_non_dialogue_style("Graded"));
/// ```
pub fn is_non_dialogue_style(name: &str) -> bool {
    name.split(|c: char| c == '-' || c == '_' || c.is_whitespace() || c.is_ascii_digit())
        .filter(|tok| !tok.is_empty())
        .any(|tok| {
            let tok_lower = tok.to_ascii_lowercase();
            NON_DIALOGUE_STYLES.iter().any(|kw| *kw == tok_lower)
        })
}

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
///
/// Fields use `f64` (IEEE 754 double) for JSON/serde interop with the ML
/// helpers, which emit plain JSON numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Normalized left edge (0–1).
    pub x: f64,
    /// Normalized top edge (0–1, top-left origin).
    pub y: f64,
    /// Normalized width (0–1).
    pub width: f64,
    /// Normalized height (0–1).
    pub height: f64,
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
    use serde_json::{from_str, to_string};

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
        let json = to_string(&line).expect("serialize");
        let back = from_str::<DialogueLine>(&json).expect("deserialize");
        assert_eq!(line, back);
    }

    #[test]
    fn non_dialogue_styles_contains_expected() {
        assert!(NON_DIALOGUE_STYLES.contains(&"sign"));
        assert!(NON_DIALOGUE_STYLES.contains(&"inen"));
        assert!(!NON_DIALOGUE_STYLES.contains(&"dialogue"));
    }

    #[test]
    fn is_non_dialogue_style_matches_whole_tokens() {
        assert!(is_non_dialogue_style("Sign"));
        assert!(is_non_dialogue_style("sign"));
        assert!(is_non_dialogue_style("Signs"));
        assert!(is_non_dialogue_style("Songs"));
        assert!(is_non_dialogue_style("OP"));
        assert!(is_non_dialogue_style("ED"));
        assert!(is_non_dialogue_style("OP-Romaji"));
        assert!(is_non_dialogue_style("ED_EN"));
        assert!(is_non_dialogue_style("Insert Song"));
        assert!(is_non_dialogue_style("Title Card"));
        // Digit boundaries split tokens (OP1 → "OP" + "1").
        assert!(is_non_dialogue_style("OP1"));
        assert!(is_non_dialogue_style("Sign2"));
    }

    #[test]
    fn is_non_dialogue_style_rejects_substrings() {
        // The whole point of the fix: abbreviations must not match as substrings.
        assert!(!is_non_dialogue_style("Top")); // contains "op" but is not a token "op"
        assert!(!is_non_dialogue_style("Dialogue Top"));
        assert!(!is_non_dialogue_style("Opening")); // contains "op"
        assert!(!is_non_dialogue_style("Named")); // contains "ed"
        assert!(!is_non_dialogue_style("Graded")); // contains "ed"
        assert!(!is_non_dialogue_style("Default"));
        assert!(!is_non_dialogue_style("Dialogue"));
        assert!(!is_non_dialogue_style("Main"));
        assert!(!is_non_dialogue_style(""));
    }
}
