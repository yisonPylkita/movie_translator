use crate::ass::{load_ass, to_ass_string};
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
fn srt_loaded_has_canonical_ass_defaults() {
    let subs = load_srt(SAMPLE_SRT).unwrap();

    // Events format must be the standard 10-field ASS v4+ order.
    assert_eq!(
        subs.events_format,
        vec![
            "Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect",
            "Text"
        ]
    );

    // Styles format must be the standard V4+ Styles order (pysubs2 parity).
    assert_eq!(
        subs.styles_format,
        vec![
            "Name",
            "Fontname",
            "Fontsize",
            "PrimaryColour",
            "SecondaryColour",
            "OutlineColour",
            "BackColour",
            "Bold",
            "Italic",
            "Underline",
            "StrikeOut",
            "ScaleX",
            "ScaleY",
            "Spacing",
            "Angle",
            "BorderStyle",
            "Outline",
            "Shadow",
            "Alignment",
            "MarginL",
            "MarginR",
            "MarginV",
            "Encoding"
        ]
    );

    // A single Default style with pysubs2's exact values.
    assert_eq!(subs.styles.len(), 1);
    assert_eq!(subs.styles[0].name, "Default");
    assert_eq!(
        subs.styles[0].raw,
        "Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1"
    );

    // Script info should contain ScriptType so the file is valid.
    assert!(
        subs.script_info_lines
            .iter()
            .any(|l| l.trim() == "ScriptType: v4.00+"),
        "missing ScriptType, got: {:?}",
        subs.script_info_lines
    );
}

#[test]
fn srt_to_ass_produces_valid_complete_ass() {
    let subs = load_srt(SAMPLE_SRT).unwrap();
    let ass = to_ass_string(&subs);

    // Events Format: line is the standard 10-field line (not empty).
    assert!(
        ass.contains(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ),
        "missing standard events format line:\n{ass}"
    );

    // Dialogue lines carry real timing + text, not empty.
    assert!(
        ass.contains("Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello, how are you?"),
        "missing populated dialogue line:\n{ass}"
    );

    // No empty Dialogue lines.
    assert!(
        !ass.contains("Dialogue: \n") && !ass.lines().any(|l| l.trim() == "Dialogue:"),
        "found an empty Dialogue line:\n{ass}"
    );

    // Re-parsing the produced ASS must succeed (no "missing field Layer").
    let reparsed = load_ass(&ass).unwrap_or_else(|e| panic!("re-parse of SRT-sourced ASS failed: {e}"));
    assert_eq!(reparsed.events.len(), subs.events.len());
    for (a, b) in reparsed.events.iter().zip(subs.events.iter()) {
        assert_eq!(a.start_ms, b.start_ms);
        assert_eq!(a.end_ms, b.end_ms);
        assert_eq!(a.text, b.text);
        assert_eq!(a.layer, b.layer);
        assert_eq!(a.style, b.style);
    }
}

#[test]
fn srt_to_ass_matches_pysubs2_structure() {
    // Parity values captured from:
    //   pysubs2.load("x.srt"); s.save("y.ass")  (pysubs2 1.8.1)
    let subs = load_srt(SAMPLE_SRT).unwrap();
    let ass = to_ass_string(&subs);

    // The exact V4+ Styles format + Default row pysubs2 emits.
    assert!(ass.contains(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    ));
    assert!(ass.contains(
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1"
    ));
    assert!(ass.contains("ScriptType: v4.00+"));

    // Per-event parity: start/end/text round-trips through ASS.
    assert_eq!(subs.events.len(), 3);
    assert_eq!(subs.events[0].text, "Hello, how are you?");
    assert_eq!(subs.events[2].start_ms, 10000);
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
