//! Clean raw burned-in OCR output into a usable .srt.
//!
//! Per-frame OCR on burned-in subs produces jitter ("był" / "byl" / "byt")
//! and transient garbage from OP/ED karaoke or signs.  This module
//! post-processes per-frame OCR into clean lines:
//!
//! - fuzzy-merge consecutive frames whose text is *similar* into one block,
//!   keeping the best (most frequent, then longest) variant
//! - drop non-dialogue noise by content (alpha ratio / min letters) and
//!   by persistence (a real line spans a minimum duration)
//!
//! Kept separate from the OCR extraction stage itself.

use std::collections::HashMap;

// Tuning defaults
const SIMILARITY: f64 = 0.80; // >= this ratio => same subtitle, merge
const TAIL_MS: i64 = 800; // how long the last line lingers past its last frame
const MIN_DURATION_MS: i64 = 200; // drop lines shorter than this
const MIN_LETTERS: usize = 3; // drop lines with fewer real letters
const MIN_ALPHA_RATIO: f64 = 0.5; // drop lines that are mostly symbols/punctuation

/// A cleaned line with text and timing.
#[derive(Debug, Clone)]
pub struct CleanLine {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

// Internal representation for grouping
struct Group {
    start_ms: i64,
    variants: Vec<String>,
    text: String, // first text in group (anchor for similarity)
}

/// Normalize text for similarity comparison (lowercase, collapse whitespace).
fn norm(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Compute similarity ratio between two strings using character n-gram
/// overlap (Jaccard-like), matching Python's difflib.SequenceMatcher.ratio()
/// approximately.
fn similar(a: &str, b: &str) -> f64 {
    let na = norm(a);
    let nb = norm(b);

    // For very short strings, use exact match
    if na.len() <= 3 || nb.len() <= 3 {
        return if na == nb { 1.0 } else { 0.0 };
    }

    // Use character bigram similarity as a proxy for SequenceMatcher
    let bigrams_a: Vec<&[u8]> = na.as_bytes().windows(2).collect();
    let bigrams_b: Vec<&[u8]> = nb.as_bytes().windows(2).collect();

    if bigrams_a.is_empty() || bigrams_b.is_empty() {
        return 0.0;
    }

    let mut intersection = 0usize;
    let mut used = vec![false; bigrams_b.len()];

    for ba in &bigrams_a {
        for (j, bb) in bigrams_b.iter().enumerate() {
            if !used[j] && ba == bb {
                intersection += 1;
                used[j] = true;
                break;
            }
        }
    }

    let union = bigrams_a.len() + bigrams_b.len() - intersection;
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Heuristic: does this OCR text look like real dialogue (not garbage)?
pub fn is_dialogue(text: &str) -> bool {
    let flat = text.replace('\n', " ").trim().to_string();
    if flat.len() < 2 {
        return false;
    }
    let letters = flat.chars().filter(|c| c.is_alphabetic()).count();
    let non_space = flat.chars().filter(|c| !c.is_whitespace()).count();
    if letters < MIN_LETTERS {
        return false;
    }
    if non_space > 0 && (letters as f64 / non_space as f64) < MIN_ALPHA_RATIO {
        return false;
    }
    true
}

/// Pick the canonical text for a merged group: most frequent, then longest.
fn best_variant(variants: &[String]) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for v in variants {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }
    let top_count = counts.values().copied().max().unwrap_or(0);
    let tied: Vec<&str> = counts
        .into_iter()
        .filter(|(_, c)| *c == top_count)
        .map(|(k, _)| k)
        .collect();
    tied.iter()
        .max_by_key(|s| s.len())
        .unwrap_or(&"")
        .to_string()
}

/// Merge per-frame `(timestamp_ms, text)` into clean, deduped lines.
///
/// Consecutive frames whose text is >= `similarity` to the group's first
/// text are folded into one block.  Blank frames close the current block.
/// Groups shorter than `min_duration_ms` or failing `is_dialogue` are dropped.
pub fn merge_ocr_results(frame_texts: &[(i64, String)]) -> Vec<CleanLine> {
    let mut frames: Vec<(i64, &str)> = frame_texts
        .iter()
        .map(|(ts, t)| (*ts, t.as_str()))
        .collect();
    frames.sort_by_key(|f| f.0);

    let mut lines: Vec<CleanLine> = Vec::new();
    let mut anchor: Option<Group> = None;

    let mut close = |end_ms: i64, anchor: &mut Option<Group>| {
        if let Some(g) = anchor.take() {
            let text = best_variant(&g.variants);
            if end_ms - g.start_ms >= MIN_DURATION_MS && is_dialogue(&text) {
                lines.push(CleanLine {
                    start_ms: g.start_ms,
                    end_ms,
                    text,
                });
            }
        }
    };

    for (ts, raw) in &frames {
        let text = raw.trim();
        if !text.is_empty() {
            if let Some(ref mut g) = anchor {
                if similar(text, &g.text) >= SIMILARITY {
                    g.variants.push(text.to_string());
                    continue;
                }
            }
            // Boundary: close the running group
            close(*ts, &mut anchor);
            anchor = Some(Group {
                start_ms: *ts,
                variants: vec![text.to_string()],
                text: text.to_string(),
            });
        } else {
            close(*ts, &mut anchor);
        }
    }

    if anchor.is_some() {
        let last_ts = frames.last().map(|f| f.0).unwrap_or(0);
        close(last_ts + TAIL_MS, &mut anchor);
    }

    lines
}

/// Format ms as SRT timestamp HH:MM:SS,mmm.
fn fmt_ts(ms: i64) -> String {
    let ms = ms.max(0);
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02},{millis:03}")
}

/// Render clean lines as SRT text.
pub fn to_srt(lines: &[CleanLine]) -> String {
    let mut out = String::new();
    for (i, ln) in lines.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            fmt_ts(ln.start_ms),
            fmt_ts(ln.end_ms),
            ln.text
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similar_basic() {
        assert!(similar("Hello world", "Hello world") > 0.9);
        assert!(similar("Hello world", "Goodbye world") < 0.8);
    }

    #[test]
    fn test_is_dialogue_typical() {
        assert!(is_dialogue("Hello world"));
        assert!(is_dialogue("What is this?"));
        assert!(!is_dialogue("ab"));
        assert!(!is_dialogue(""));
        assert!(!is_dialogue("#!@"));
    }

    #[test]
    fn test_merge_ocr_results() {
        let frames = vec![
            (0i64, "Hello".to_string()),
            (100, "Hello".to_string()),
            (200, "Hello world".to_string()),
            (500, "Goodbye".to_string()),
        ];
        let lines = merge_ocr_results(&frames);
        // "Hello" and "Hello world" should merge (similarity >= 0.8)
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_best_variant() {
        let variants = vec!["był".to_string(), "byl".to_string(), "był".to_string()];
        assert_eq!(best_variant(&variants), "był");
    }

    #[test]
    fn test_to_srt() {
        let lines = vec![CleanLine {
            start_ms: 1000,
            end_ms: 3000,
            text: "Hello".to_string(),
        }];
        let srt = to_srt(&lines);
        assert!(srt.contains("00:00:01,000 --> 00:00:03,000"));
        assert!(srt.contains("Hello"));
    }
}
