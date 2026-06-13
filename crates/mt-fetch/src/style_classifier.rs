//! Structural classification of ASS subtitle styles.
//!
//! Classifies styles as dialogue or non-dialogue based on aggregate event
//! properties rather than style name keywords. This is robust to arbitrary
//! naming conventions across different fansub groups.
//!
//! The classifier uses three signals:
//!   1. Positioning ratio — signs and karaoke use explicit `\pos()`/`\move()`
//!   2. Average text length — karaoke syllables are 1–3 characters
//!   3. Event count and duration — karaoke has many rapid-fire events

use std::collections::{HashMap, HashSet};

use mt_subtitles::model::Event;

/// Per-style aggregate metrics collected during classification.
#[derive(Debug, Default)]
struct StyleMetrics {
    count: usize,
    total_duration: i64,
    total_text_len: usize,
    positioned_count: usize,
}

/// Classify ASS styles and return the set of dialogue style names.
///
/// Analyses aggregate properties of events per style to determine which
/// styles contain dialogue.  Works with a slice of `mt_subtitles::model::Event`.
pub fn classify_styles(events: &[Event]) -> HashSet<String> {
    let mut style_metrics: HashMap<String, StyleMetrics> = HashMap::new();

    for event in events {
        let text = &event.text;
        if text.trim().is_empty() {
            continue;
        }

        let style = event.style.as_str();
        let m = style_metrics.entry(style.to_string()).or_default();

        m.count += 1;
        m.total_duration += event.end_ms - event.start_ms;

        // plaintext: strip ASS override tags, then trim.
        let plaintext = event.plaintext();
        let plain_trimmed = plaintext.trim().to_string();
        m.total_text_len += plain_trimmed.chars().count();

        if text.contains("\\pos(") || text.contains("\\move(") {
            m.positioned_count += 1;
        }
    }

    let mut dialogue_styles = HashSet::new();

    for (style, m) in &style_metrics {
        let n = m.count;
        if n == 0 {
            continue;
        }

        let avg_text = m.total_text_len as f64 / n as f64;
        let avg_dur = m.total_duration as f64 / n as f64;
        let pos_ratio = m.positioned_count as f64 / n as f64;

        if is_dialogue(pos_ratio, avg_text, avg_dur, n) {
            dialogue_styles.insert(style.clone());
        }
    }

    dialogue_styles
}

/// Determine if a style is dialogue based on its aggregate metrics.
pub fn is_dialogue(pos_ratio: f64, avg_text: f64, avg_dur: f64, count: usize) -> bool {
    // Rule 1: High positioning = non-dialogue (signs/typesetting)
    if pos_ratio >= 0.5 {
        // Rescue: long text + long duration = positioned dialogue (e.g. \an8 top lines)
        if avg_text > 20.0 && avg_dur > 1500.0 {
            return true;
        }
        return false;
    }

    // Rule 2: Very short text + many events = karaoke syllables
    if avg_text < 5.0 && count > 50 {
        return false;
    }

    // Rule 3: Rapid-fire short events = karaoke
    if count > 500 && avg_dur < 500.0 {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use mt_subtitles::model::{Event, EventKind};

    use super::*;

    // Helper: build an Event with the given style, start, end, text.
    fn make_event(style: &str, start_ms: i64, end_ms: i64, text: &str) -> Event {
        Event {
            kind: EventKind::Dialogue,
            layer: 0,
            start_ms,
            end_ms,
            style: style.to_string(),
            name: String::new(),
            margin_l: 0,
            margin_r: 0,
            margin_v: 0,
            effect: String::new(),
            text: text.to_string(),
        }
    }

    #[test]
    fn normal_dialogue_classified_as_dialogue() {
        let events = vec![
            make_event("Dialogue", 1000, 3000, "Hello, world!"),
            make_event("Dialogue", 4000, 6000, "How are you doing today?"),
            make_event("Dialogue", 7000, 9000, "This is a normal subtitle line."),
        ];
        let result = classify_styles(&events);
        assert!(result.contains("Dialogue"), "expected 'Dialogue' in result");
    }

    #[test]
    fn positioned_signs_classified_as_non_dialogue() {
        let events = vec![
            make_event("Signs", 1000, 3000, "{\\pos(960,100)}Location Name"),
            make_event("Signs", 5000, 7000, "{\\pos(960,100)}Another Sign"),
            make_event("Signs", 10000, 12000, "{\\pos(100,500)}Shop Name"),
        ];
        let result = classify_styles(&events);
        assert!(
            !result.contains("Signs"),
            "expected 'Signs' NOT in result, got: {result:?}"
        );
    }

    #[test]
    fn karaoke_short_text_classified_as_non_dialogue() {
        // Per-character karaoke: many events, very short text
        let chars = "abcdefghijklmnop".repeat(5);
        let events: Vec<_> = chars
            .chars()
            .enumerate()
            .map(|(i, c)| {
                let start = i as i64 * 200;
                make_event("OP-Romaji", start, start + 150, &c.to_string())
            })
            .collect();
        let result = classify_styles(&events);
        assert!(
            !result.contains("OP-Romaji"),
            "expected 'OP-Romaji' NOT in result, got: {result:?}"
        );
    }

    #[test]
    fn rapid_fire_events_classified_as_non_dialogue() {
        // 600 events, 300ms each — karaoke pattern
        let events: Vec<_> = (0..600)
            .map(|i| {
                let start = i as i64 * 300;
                make_event("EDRO", start, start + 250, &format!("syllable {i}"))
            })
            .collect();
        let result = classify_styles(&events);
        assert!(
            !result.contains("EDRO"),
            "expected 'EDRO' NOT in result, got: {result:?}"
        );
    }

    #[test]
    fn positioned_dialogue_rescued() {
        // Dialogue with \an8 top positioning should NOT be filtered.
        // avg_text > 20, avg_dur > 1500 → rescued as dialogue
        let events = vec![
            make_event(
                "Dialogue Top",
                1000,
                3500,
                "{\\pos(960,50)}This is a top-positioned dialogue line",
            ),
            make_event(
                "Dialogue Top",
                4000,
                6500,
                "{\\pos(960,50)}Another line spoken by a character",
            ),
            make_event(
                "Dialogue Top",
                7000,
                9500,
                "{\\pos(960,50)}Third dialogue line with positioning",
            ),
        ];
        let result = classify_styles(&events);
        assert!(
            result.contains("Dialogue Top"),
            "expected 'Dialogue Top' in result (rescued), got: {result:?}"
        );
    }

    #[test]
    fn mixed_styles_classified_correctly() {
        let mut events: Vec<Event> = vec![
            make_event("Default", 1000, 3000, "Normal dialogue here"),
            make_event("Default", 4000, 6000, "More dialogue text for testing"),
            make_event("Default", 7000, 9000, "Third line of regular dialogue"),
            make_event("Signs", 1000, 3000, "{\\pos(960,100)}Location"),
            make_event("Signs", 5000, 7000, "{\\pos(960,100)}Shop"),
        ];
        // OP: 80 single-char events
        let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            .chars()
            .cycle()
            .take(80)
            .collect::<Vec<_>>();
        for (i, c) in chars.iter().enumerate() {
            let start = i as i64 * 200;
            events.push(make_event("OP", start, start + 150, &c.to_string()));
        }

        let result = classify_styles(&events);
        assert!(result.contains("Default"), "expected 'Default' in result");
        assert!(!result.contains("Signs"), "expected 'Signs' NOT in result");
        assert!(!result.contains("OP"), "expected 'OP' NOT in result");
    }

    #[test]
    fn empty_events_returns_empty() {
        let result = classify_styles(&[]);
        assert!(result.is_empty(), "expected empty result");
    }

    #[test]
    fn srt_default_style_is_dialogue() {
        // SRT files produce a single 'Default' style — always dialogue.
        let events = vec![
            make_event("Default", 1000, 3000, "First line of subtitles"),
            make_event("Default", 4000, 6000, "Second line of subtitles"),
            make_event("Default", 7000, 9000, "Third line of subtitles"),
        ];
        let result = classify_styles(&events);
        assert!(result.contains("Default"), "expected 'Default' in result");
    }

    #[test]
    fn positioned_short_text_is_non_dialogue() {
        // Positioned events with short text = signs, not dialogue.
        let events = vec![
            make_event("TypeSetting", 1000, 5000, "{\\pos(100,200)}EP 01"),
            make_event("TypeSetting", 6000, 10000, "{\\pos(100,200)}Title"),
            make_event("TypeSetting", 11000, 15000, "{\\pos(100,200)}Day 1"),
        ];
        let result = classify_styles(&events);
        assert!(
            !result.contains("TypeSetting"),
            "expected 'TypeSetting' NOT in result, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Unit tests for is_dialogue()
    // -----------------------------------------------------------------------

    #[test]
    fn is_dialogue_normal_returns_true() {
        assert!(is_dialogue(0.0, 30.0, 2000.0, 10));
    }

    #[test]
    fn is_dialogue_high_pos_ratio_returns_false() {
        assert!(!is_dialogue(0.5, 5.0, 1000.0, 10));
    }

    #[test]
    fn is_dialogue_high_pos_ratio_rescued_by_long_text_and_dur() {
        assert!(is_dialogue(0.6, 25.0, 2000.0, 10));
    }

    #[test]
    fn is_dialogue_karaoke_short_text_many_events() {
        assert!(!is_dialogue(0.0, 2.0, 200.0, 100));
    }

    #[test]
    fn is_dialogue_rapid_fire_large_count() {
        assert!(!is_dialogue(0.0, 10.0, 300.0, 600));
    }
}
