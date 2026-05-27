use crate::model::EventKind;
use crate::srt::{load_srt, to_srt_string};

const SAMPLE_SRT: &str = "1\n00:00:01,000 --> 00:00:03,000\nHello, how are you?\n\n2\n00:00:04,000 --> 00:00:06,000\nI am fine, thank you.\n\n3\n00:00:10,000 --> 00:00:12,000\nWhat a beautiful day!\n";

#[test]
fn parse_event_count() {
    let subs = load_srt(SAMPLE_SRT).unwrap();
    assert_eq!(subs.events.len(), 3);
}

#[test]
fn parse_timing_ms() {
    let subs = load_srt(SAMPLE_SRT).unwrap();
    assert_eq!(subs.events[0].start_ms, 1000);
    assert_eq!(subs.events[0].end_ms, 3000);
    assert_eq!(subs.events[1].start_ms, 4000);
    assert_eq!(subs.events[1].end_ms, 6000);
}

#[test]
fn parse_text() {
    let subs = load_srt(SAMPLE_SRT).unwrap();
    assert_eq!(subs.events[0].text, "Hello, how are you?");
    assert_eq!(subs.events[1].text, "I am fine, thank you.");
    assert_eq!(subs.events[2].text, "What a beautiful day!");
}

#[test]
fn parse_multiline_text() {
    let input = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two\n";
    let subs = load_srt(input).unwrap();
    assert_eq!(subs.events[0].text, "Line one\nLine two");
}

#[test]
fn events_are_dialogue() {
    let subs = load_srt(SAMPLE_SRT).unwrap();
    for ev in &subs.events {
        assert_eq!(ev.kind, EventKind::Dialogue);
        assert_eq!(ev.style, "Default");
        assert_eq!(ev.layer, 0);
    }
}

#[test]
fn round_trip() {
    let subs = load_srt(SAMPLE_SRT).unwrap();
    let out = to_srt_string(&subs);
    let subs2 = load_srt(&out).unwrap();
    assert_eq!(subs.events.len(), subs2.events.len());
    for (e1, e2) in subs.events.iter().zip(subs2.events.iter()) {
        assert_eq!(e1.start_ms, e2.start_ms);
        assert_eq!(e1.end_ms, e2.end_ms);
        assert_eq!(e1.text, e2.text);
    }
}

#[test]
fn empty_input() {
    let subs = load_srt("").unwrap();
    assert_eq!(subs.events.len(), 0);
}

#[test]
fn timing_precision() {
    let input = "1\n01:23:45,678 --> 02:34:56,789\nTest\n";
    let subs = load_srt(input).unwrap();
    let expected_start = 3_600_000 + 23 * 60_000 + 45 * 1_000 + 678;
    let expected_end = 2 * 3_600_000 + 34 * 60_000 + 56 * 1_000 + 789;
    assert_eq!(subs.events[0].start_ms, expected_start);
    assert_eq!(subs.events[0].end_ms, expected_end);
}
