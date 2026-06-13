//! Split coarse ASR utterances into subtitle-sized lines.
//!
//! Apple SpeechAnalyzer returns long multi-sentence utterances (the PoC
//! measured 16 segments where the reference had ~58 dialogue lines).
//! Subtitles want one sentence-ish line at a time, so we split on sentence
//! punctuation and allocate the utterance's time span proportionally to each
//! piece's character length.

use mt_core::DialogueLine;
use regex::Regex;

lazy_static::lazy_static! {
    /// A sentence piece: text up to (and including) sentence punctuation plus
    /// any trailing quote, or a final unpunctuated remainder.
    static ref SENTENCE_RE: Regex = Regex::new(
        r#"[^.!?。！？]*[.!?。！？]+["」』]?\s*|[^.!?。！？]+$"#
    ).unwrap();
}

/// A piece must contain at least one word character (any script) to stand on
/// its own; bare punctuation runs ('...') glue onto the following sentence.
fn has_word(text: &str) -> bool {
    text.chars().any(|c| c.is_alphanumeric())
}

fn pieces(text: &str) -> Vec<String> {
    let raw: Vec<_> = SENTENCE_RE
        .find_iter(text)
        .map(|m| m.as_str().trim())
        .filter(|s| !s.is_empty())
        .collect();

    let mut out: Vec<String> = Vec::new();
    let mut carry = String::new();

    for p in raw {
        if !has_word(p) {
            carry.push_str(p); // e.g. a leading '...' — prefix to next sentence
            continue;
        }
        let mut full = carry.clone();
        full.push_str(p);
        out.push(full);
        carry.clear();
    }
    if !carry.is_empty() {
        if let Some(last) = out.last_mut() {
            last.push_str(&carry);
        } else {
            out.push(carry);
        }
    }

    out
}

/// Split one segment into sentence pieces with optional VAD boundaries.
///
/// When `boundaries` (VAD-detected pause timestamps in ms within this
/// segment's time span) are provided, each piece's end time is snapped to
/// the nearest boundary.  Falls back to proportional timing when no
/// boundaries are given or none fall inside the segment's range.
pub fn split_segment(seg: &DialogueLine, boundaries: Option<&[i64]>) -> Vec<DialogueLine> {
    let text_pieces = pieces(&seg.text);
    if text_pieces.len() <= 1 {
        return vec![seg.clone()];
    }

    // Narrow to boundaries that actually fall inside this segment.
    let inner: Vec<_> = boundaries
        .map(|b| {
            b.iter()
                .filter(|&&b| seg.start_ms < b && b < seg.end_ms)
                .copied()
                .collect()
        })
        .unwrap_or_default();

    // Cap to number of sentence pieces
    let inner = if inner.len() >= text_pieces.len() {
        inner[..text_pieces.len() - 1].to_vec()
    } else {
        inner
    };

    if inner.is_empty() {
        proportional_split(seg, &text_pieces)
    } else {
        boundary_split(seg, &text_pieces, &inner)
    }
}

fn proportional_split(seg: &DialogueLine, pieces: &[String]) -> Vec<DialogueLine> {
    let total: usize = pieces.iter().map(|p| p.len()).sum();
    let span = seg.end_ms - seg.start_ms;
    let mut out: Vec<DialogueLine> = Vec::new();
    let mut cursor = seg.start_ms;

    for (i, piece) in pieces.iter().enumerate() {
        let end = if i == pieces.len() - 1 {
            seg.end_ms
        } else {
            let offset = (span as f64 * piece.len() as f64 / total as f64).round() as i64;
            (seg.end_ms).min(cursor.max(cursor + offset))
        };
        out.push(DialogueLine {
            start_ms: cursor,
            end_ms: end,
            text: piece.clone(),
        });
        cursor = end;
    }

    out
}

fn boundary_split(seg: &DialogueLine, pieces: &[String], boundaries: &[i64]) -> Vec<DialogueLine> {
    let num_boundaries = boundaries.len();
    let num_pieces = pieces.len();

    if num_pieces <= num_boundaries {
        return one_per_piece(seg, pieces, boundaries);
    }

    // More pieces than boundaries: distribute across boundaries+1 groups.
    let base = num_pieces / (num_boundaries + 1);
    let rem = num_pieces % (num_boundaries + 1);

    let mut groups: Vec<Vec<&str>> = Vec::new();
    let mut idx = 0;
    for b in 0..=num_boundaries {
        let count = base + if b < rem { 1 } else { 0 };
        let group = pieces[idx..idx + count]
            .iter()
            .map(|s| s.as_str())
            .collect();
        groups.push(group);
        idx += count;
    }

    let mut out: Vec<DialogueLine> = Vec::new();
    let mut cursor = seg.start_ms;
    for (i, group) in groups.iter().enumerate() {
        let text = group.join(" ");
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let end = if i < boundaries.len() {
            boundaries[i]
        } else {
            seg.end_ms
        };
        out.push(DialogueLine {
            start_ms: cursor,
            end_ms: end,
            text: text.to_string(),
        });
        cursor = end;
    }

    out
}

fn one_per_piece(seg: &DialogueLine, pieces: &[String], boundaries: &[i64]) -> Vec<DialogueLine> {
    let mut out: Vec<DialogueLine> = Vec::new();
    let mut cursor = seg.start_ms;
    for (i, piece) in pieces.iter().enumerate() {
        let end = if i < boundaries.len() {
            boundaries[i]
        } else {
            seg.end_ms
        };
        out.push(DialogueLine {
            start_ms: cursor,
            end_ms: end,
            text: piece.clone(),
        });
        cursor = end;
    }
    out
}

/// Split every segment; order preserved.
pub fn split_segments(segs: &[DialogueLine], boundaries: Option<&[i64]>) -> Vec<DialogueLine> {
    let mut out: Vec<DialogueLine> = Vec::new();
    for seg in segs {
        out.extend(split_segment(seg, boundaries));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_single_segment() {
        let seg = DialogueLine {
            start_ms: 0,
            end_ms: 1000,
            text: "Hello world! How are you?".to_string(),
        };
        let result = split_segment(&seg, None);
        assert!(result.len() >= 2);
        assert_eq!(result[0].text, "Hello world!");
        assert_eq!(result[1].text, "How are you?");
    }

    #[test]
    fn test_split_short_no_split() {
        let seg = DialogueLine {
            start_ms: 0,
            end_ms: 1000,
            text: "Hello".to_string(),
        };
        let result = split_segment(&seg, None);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_split_with_boundaries() {
        let seg = DialogueLine {
            start_ms: 0,
            end_ms: 2000,
            text: "Hello. How are you? I am fine.".to_string(),
        };
        let boundaries = vec![500, 1200];
        let result = split_segment(&seg, Some(&boundaries));
        // Should use boundaries for splitting
        assert_eq!(result.len(), 3);
    }
}
