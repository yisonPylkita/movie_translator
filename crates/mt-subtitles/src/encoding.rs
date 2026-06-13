//! Encoding detection and normalization for subtitle files.
//!
//! Polish subtitles frequently arrive in CP1250 or ISO-8859-2 encoding.
//! This module detects the encoding and rewrites the file to UTF-8 if needed.

use std::path::Path;

use encoding_rs::{Encoding, ISO_8859_2, WINDOWS_1250};

/// Polish diacritical characters.
const POLISH_CHARS: &str = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ";

/// Count Polish diacritical characters in a string.
pub fn count_polish(text: &str) -> usize {
    text.chars().filter(|c| POLISH_CHARS.contains(*c)).count()
}

/// Detect encoding of a subtitle file and re-save as UTF-8 if needed.
///
/// Algorithm:
/// 1. Read raw bytes.
/// 2. UTF-8 BOM (`\xef\xbb\xbf`) → already fine, return.
/// 3. Try decode as UTF-8 → if clean, return.
/// 4. Try CP1250 and ISO-8859-2; score by Polish char count; pick best
///    (ties favor CP1250: it is tried first and only a strictly higher score wins).
/// 5. If best found → rewrite file as UTF-8.
/// 6. Fallback: ISO-8859-1 (closest match: never errors).
/// 7. If none work → leave unchanged.
pub fn normalize_encoding(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::fs::read(path)?;

    // UTF-8 BOM → already fine
    if raw.starts_with(b"\xef\xbb\xbf") {
        return Ok(());
    }

    // Try UTF-8
    if std::str::from_utf8(&raw).is_ok() {
        return Ok(());
    }

    // Try CP1250 and ISO-8859-2, pick by Polish char score.
    let ambiguous: &[&'static Encoding] = &[WINDOWS_1250, ISO_8859_2];

    let mut best_text: Option<String> = None;
    let mut best_score: i64 = -1;

    for enc in ambiguous {
        let (cow, _, had_errors) = enc.decode(&raw);
        if !had_errors {
            let score = count_polish(&cow) as i64;
            if score > best_score {
                best_score = score;
                best_text = Some(cow.into_owned());
            }
        }
    }

    if let Some(text) = best_text {
        std::fs::write(path, text.as_bytes())?;
        return Ok(());
    }

    // Fallback: ISO-8859-1 (1:1 byte→char mapping, never errors).
    // `encoding_rs` does not expose iso-8859-1 directly; map bytes manually.
    let text: String = raw.iter().map(|&b| b as char).collect();
    std::fs::write(path, text.as_bytes())?;

    Ok(())
}
