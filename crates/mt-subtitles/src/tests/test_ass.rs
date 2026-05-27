use std::path::Path;

use crate::ass::{load_ass, to_ass_string};

fn ground_truth_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benchmarks/onepiece/ground_truth/ground_truth.json")
}

fn ground_truth() -> serde_json::Value {
    let path = ground_truth_path();
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("ground_truth.json not found at {}", path.display()));
    serde_json::from_str(&content).expect("parse ground_truth.json")
}

fn corpus_path(filename: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benchmarks/onepiece")
        .join(filename)
}

const CORPUS_FILES: &[&str] = &[
    "op_0031.ass",
    "op_0040.ass",
    "op_0050.ass",
    "op_0055.ass",
    "op_0061.ass",
    "op_0031.gold.pl.ass",
    "op_0040.gold.pl.ass",
    "op_0050.gold.pl.ass",
    "op_0055.gold.pl.ass",
    "op_0061.gold.pl.ass",
    "onepace_arlongpark_01_pl.ass",
];

#[test]
fn corpus_round_trip_all_11_files() {
    // The One Piece corpus lives in the untracked `benchmarks/onepiece/`
    // directory (local benchmark data). When it's absent — e.g. on CI — skip:
    // the inline-fixture tests in this module provide committed coverage.
    if !ground_truth_path().exists() {
        eprintln!("skipping corpus_round_trip: benchmarks/onepiece corpus not present");
        return;
    }
    let gt = ground_truth();
    let mut failures: Vec<String> = Vec::new();

    for filename in CORPUS_FILES {
        let path = corpus_path(filename);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("cannot read {}", path.display()));

        let subs = load_ass(&content).unwrap_or_else(|e| panic!("parse error for {filename}: {e}"));

        let gt_file = &gt[filename];
        let gt_event_count = gt_file["event_count"].as_u64().unwrap() as usize;
        let gt_style_count = gt_file["style_count"].as_u64().unwrap() as usize;
        let gt_events = gt_file["events"].as_array().unwrap();

        if subs.events.len() != gt_event_count {
            failures.push(format!(
                "{filename}: event count {}/{gt_event_count}",
                subs.events.len()
            ));
            continue;
        }
        if subs.styles.len() != gt_style_count {
            failures.push(format!(
                "{filename}: style count {}/{gt_style_count}",
                subs.styles.len()
            ));
        }

        for (i, (ev, gt_ev)) in subs.events.iter().zip(gt_events.iter()).enumerate() {
            let gt_start = gt_ev["start_ms"].as_i64().unwrap();
            let gt_end = gt_ev["end_ms"].as_i64().unwrap();
            let gt_style = gt_ev["style"].as_str().unwrap();
            let gt_text = gt_ev["text"].as_str().unwrap();
            let gt_type = gt_ev["type"].as_str().unwrap();
            let gt_layer = gt_ev["layer"].as_i64().unwrap() as i32;
            let gt_name = gt_ev["name"].as_str().unwrap_or("");
            let gt_marginl = gt_ev["marginl"].as_i64().unwrap_or(0) as i32;
            let gt_marginr = gt_ev["marginr"].as_i64().unwrap_or(0) as i32;
            let gt_marginv = gt_ev["marginv"].as_i64().unwrap_or(0) as i32;
            let gt_effect = gt_ev["effect"].as_str().unwrap_or("");

            if ev.start_ms != gt_start {
                failures.push(format!(
                    "{filename}[{i}].start_ms: got {} expected {gt_start}",
                    ev.start_ms
                ));
            }
            if ev.end_ms != gt_end {
                failures.push(format!(
                    "{filename}[{i}].end_ms: got {} expected {gt_end}",
                    ev.end_ms
                ));
            }
            if ev.style != gt_style {
                failures.push(format!(
                    "{filename}[{i}].style: got {:?} expected {:?}",
                    ev.style, gt_style
                ));
            }
            if ev.text != gt_text {
                let got: String = ev.text.chars().take(60).collect();
                let exp: String = gt_text.chars().take(60).collect();
                failures.push(format!(
                    "{filename}[{i}].text: got {got:?} expected {exp:?}"
                ));
            }
            if ev.kind.as_str() != gt_type {
                failures.push(format!(
                    "{filename}[{i}].kind: got {:?} expected {:?}",
                    ev.kind.as_str(),
                    gt_type
                ));
            }
            if ev.layer != gt_layer {
                failures.push(format!(
                    "{filename}[{i}].layer: got {} expected {gt_layer}",
                    ev.layer
                ));
            }
            if ev.name != gt_name {
                failures.push(format!(
                    "{filename}[{i}].name: got {:?} expected {:?}",
                    ev.name, gt_name
                ));
            }
            if ev.margin_l != gt_marginl {
                failures.push(format!(
                    "{filename}[{i}].margin_l: got {} expected {gt_marginl}",
                    ev.margin_l
                ));
            }
            if ev.margin_r != gt_marginr {
                failures.push(format!(
                    "{filename}[{i}].margin_r: got {} expected {gt_marginr}",
                    ev.margin_r
                ));
            }
            if ev.margin_v != gt_marginv {
                failures.push(format!(
                    "{filename}[{i}].margin_v: got {} expected {gt_marginv}",
                    ev.margin_v
                ));
            }
            if ev.effect != gt_effect {
                failures.push(format!(
                    "{filename}[{i}].effect: got {:?} expected {:?}",
                    ev.effect, gt_effect
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Corpus fidelity failures ({}):\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn round_trip_stability() {
    let files = ["op_0031.ass", "op_0040.ass", "onepace_arlongpark_01_pl.ass"];

    // Untracked local corpus; skip on CI (see corpus_round_trip_all_11_files).
    if !corpus_path(files[0]).exists() {
        eprintln!("skipping round_trip_stability: benchmarks/onepiece corpus not present");
        return;
    }

    for filename in &files {
        let path = corpus_path(filename);
        let content = std::fs::read_to_string(&path).unwrap();

        let subs1 = load_ass(&content).unwrap();
        let serialized = to_ass_string(&subs1);
        let subs2 =
            load_ass(&serialized).unwrap_or_else(|e| panic!("re-parse failed for {filename}: {e}"));

        assert_eq!(
            subs1.events.len(),
            subs2.events.len(),
            "{filename}: event count changed after round-trip"
        );
        assert_eq!(
            subs1.styles.len(),
            subs2.styles.len(),
            "{filename}: style count changed after round-trip"
        );

        for (i, (e1, e2)) in subs1.events.iter().zip(subs2.events.iter()).enumerate() {
            assert_eq!(e1.start_ms, e2.start_ms, "{filename}[{i}].start_ms");
            assert_eq!(e1.end_ms, e2.end_ms, "{filename}[{i}].end_ms");
            assert_eq!(e1.text, e2.text, "{filename}[{i}].text");
            assert_eq!(e1.style, e2.style, "{filename}[{i}].style");
        }
    }
}

#[test]
fn inline_round_trip_stability() {
    // Self-contained round-trip (no external corpus) so CI exercises
    // load -> serialize -> reload fidelity across styles, comments, override
    // tags, multi-line text, and varied timing/layers.
    let input = "[Script Info]\nTitle: RT Test\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\nStyle: Sign,Arial,36,&H00FFFF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nComment: 0,0:00:00.00,0:00:00.00,Default,,0,0,0,,marker\nDialogue: 0,0:00:01.00,0:00:03.50,Default,Speaker,0,0,0,,{\\i1}Line one{\\i0}\\NLine two\nDialogue: 5,0:00:04.00,0:00:06.00,Sign,,20,20,40,fade,Sign text, with comma\n";

    let subs1 = load_ass(input).unwrap();
    let serialized = to_ass_string(&subs1);
    let subs2 = load_ass(&serialized).expect("re-parse of serialized output failed");

    assert_eq!(subs1.events.len(), subs2.events.len());
    assert_eq!(subs1.styles.len(), subs2.styles.len());
    for (i, (e1, e2)) in subs1.events.iter().zip(subs2.events.iter()).enumerate() {
        assert_eq!(e1.start_ms, e2.start_ms, "event {i} start_ms");
        assert_eq!(e1.end_ms, e2.end_ms, "event {i} end_ms");
        assert_eq!(e1.text, e2.text, "event {i} text");
        assert_eq!(e1.style, e2.style, "event {i} style");
        assert_eq!(e1.layer, e2.layer, "event {i} layer");
        assert_eq!(e1.kind, e2.kind, "event {i} kind");
    }
}

#[test]
fn ssa_v4_marked_field_maps_to_layer() {
    // Legacy SSA v4 uses `Marked` (value `Marked=N`) instead of `Layer`.
    // pysubs2 treats it as the layer; we must not fail with "missing field Layer".
    let input = "[Script Info]\nScriptType: v4.00\n\n[Events]\nFormat: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: Marked=0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world\n";
    let subs = load_ass(input).unwrap_or_else(|e| panic!("SSA v4 parse failed: {e}"));
    assert_eq!(subs.events.len(), 1);
    assert_eq!(subs.events[0].layer, 0);
    assert_eq!(subs.events[0].text, "Hello world");
}

#[test]
fn bom_preserved() {
    let with_bom = "\u{FEFF}[Script Info]\nTitle: Test\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello\n";
    let subs = load_ass(with_bom).unwrap();
    assert!(subs.bom);
    let out = to_ass_string(&subs);
    assert!(out.starts_with('\u{FEFF}'));
}

#[test]
fn comment_events_preserved() {
    use crate::model::EventKind;

    let input = "[Script Info]\nTitle: Test\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nComment: 0,0:00:00.00,0:00:00.00,Default,,0,0,0,,{=71}Chapter marker\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello\n";
    let subs = load_ass(input).unwrap();
    assert_eq!(subs.events.len(), 2);
    assert_eq!(subs.events[0].kind, EventKind::Comment);
    assert_eq!(subs.events[1].kind, EventKind::Dialogue);

    let out = to_ass_string(&subs);
    assert!(out.contains("Comment: 0,0:00:00.00,0:00:00.00,Default,,0,0,0,,{=71}Chapter marker"));
}

#[test]
fn text_with_commas_preserved() {
    let input = "[Script Info]\nTitle: Test\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello, world, goodbye\n";
    let subs = load_ass(input).unwrap();
    assert_eq!(subs.events[0].text, "Hello, world, goodbye");
}
