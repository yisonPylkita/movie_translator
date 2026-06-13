//! Clean raw Whisper output for subtitle use.
//!
//! Whisper hallucinates on trailing music/silence (the bake-off caught it
//! looping "ご視聴ありがとうございました" over the ED with timestamps past
//! the end of the audio).  This filter drops empties, drops segments that
//! start at/past the audio end, clamps end times, and collapses consecutive
//! duplicate texts.

use mt_core::DialogueLine;

/// Drop/clamp/de-loop raw ASR segments against the real audio length.
pub fn clean_segments(segs: &[DialogueLine], audio_ms: i64) -> Vec<DialogueLine> {
    let mut out: Vec<DialogueLine> = Vec::with_capacity(segs.len());
    let mut prev_text: Option<String> = None;

    for seg in segs {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        if seg.start_ms >= audio_ms {
            continue;
        }
        if let Some(ref prev) = prev_text {
            if text == prev.as_str() {
                continue;
            }
        }
        out.push(DialogueLine {
            start_ms: seg.start_ms,
            end_ms: seg.end_ms.min(audio_ms),
            text: text.to_string(),
        });
        prev_text = Some(text.to_string());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_segments_past_audio() {
        let segs = vec![
            DialogueLine {
                start_ms: 0,
                end_ms: 1000,
                text: "Hello".to_string(),
            },
            DialogueLine {
                start_ms: 5000,
                end_ms: 6000,
                text: "World".to_string(),
            },
        ];
        let cleaned = clean_segments(&segs, 3000);
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].text, "Hello");
    }

    #[test]
    fn test_clean_segments_deduplicate() {
        let segs = vec![
            DialogueLine {
                start_ms: 0,
                end_ms: 1000,
                text: "Hello".to_string(),
            },
            DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "Hello".to_string(),
            },
            DialogueLine {
                start_ms: 2000,
                end_ms: 3000,
                text: "World".to_string(),
            },
        ];
        let cleaned = clean_segments(&segs, 5000);
        assert_eq!(cleaned.len(), 2);
        assert_eq!(cleaned[0].text, "Hello");
        assert_eq!(cleaned[1].text, "World");
    }

    #[test]
    fn test_clean_segments_empty_is_skipped() {
        let segs = vec![
            DialogueLine {
                start_ms: 0,
                end_ms: 1000,
                text: "Hello".to_string(),
            },
            DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "".to_string(),
            },
        ];
        let cleaned = clean_segments(&segs, 5000);
        assert_eq!(cleaned.len(), 1);
    }

    #[test]
    fn test_clean_segments_clamp_end() {
        let segs = vec![DialogueLine {
            start_ms: 0,
            end_ms: 10000,
            text: "Hello".to_string(),
        }];
        let cleaned = clean_segments(&segs, 3000);
        assert_eq!(cleaned[0].end_ms, 3000);
    }
}
