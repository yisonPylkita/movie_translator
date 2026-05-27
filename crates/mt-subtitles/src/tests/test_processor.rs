use std::path::Path;

use crate::model::{Event, EventKind};
use crate::processor::{find_dialogue_style, SubtitleProcessor};

fn make_ass_content() -> &'static str {
    "[Script Info]\nTitle: Test\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\nStyle: Signs,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello, how are you?\nDialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,I am fine, thank you.\nDialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,What a beautiful day!\nDialogue: 0,0:00:15.00,0:00:17.00,Signs,,0,0,0,,EPISODE 1\n"
}

fn make_ass_file(tmp_path: &Path) -> std::path::PathBuf {
    let p = tmp_path.join("test.ass");
    std::fs::write(&p, make_ass_content()).unwrap();
    p
}

fn make_srt_content() -> &'static str {
    "1\n00:00:01,000 --> 00:00:03,000\nHello, how are you?\n\n2\n00:00:04,000 --> 00:00:06,000\nI am fine, thank you.\n\n3\n00:00:10,000 --> 00:00:12,000\nWhat a beautiful day!\n"
}

fn make_srt_file(tmp_path: &Path) -> std::path::PathBuf {
    let p = tmp_path.join("test.srt");
    std::fs::write(&p, make_srt_content()).unwrap();
    p
}

#[test]
fn extract_dialogue_lines_from_ass() {
    let tmp = tempfile_path();
    let ass_file = make_ass_file(&tmp);
    let lines = SubtitleProcessor::extract_dialogue_lines(&ass_file).unwrap();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0].text, "Hello, how are you?");
    assert_eq!(lines[1].text, "I am fine, thank you.");
    assert_eq!(lines[2].text, "What a beautiful day!");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn ass_line_breaks_become_real_newlines() {
    use mt_core::types::DialogueLine;
    let tmp = tempfile_path();
    // Event with an ASS hard break (\N) and a soft break (\n) plus an override tag.
    let content = "[Script Info]\nTitle: Test\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\i1}First line\\NSecond line\\nThird line\n";
    let ass_file = tmp.join("breaks.ass");
    std::fs::write(&ass_file, content).unwrap();

    let lines = SubtitleProcessor::extract_dialogue_lines(&ass_file).unwrap();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].text, "First line\nSecond line\nThird line");
    assert!(
        !lines[0].text.contains("\\N") && !lines[0].text.contains("\\n"),
        "literal ASS break tokens must not survive: {:?}",
        lines[0].text
    );

    // Round-trip: create_subtitle_file should turn real newlines back into \N on output.
    let output = tmp.join("breaks_out.ass");
    let out_lines = vec![DialogueLine {
        start_ms: 1000,
        end_ms: 3000,
        text: "First line\nSecond line".to_string(),
    }];
    SubtitleProcessor::create_english_subtitles(&ass_file, &out_lines, &output).unwrap();
    let out_content = std::fs::read_to_string(&output).unwrap();
    assert!(
        out_content.contains("First line\\NSecond line"),
        "newline should serialize back to \\N: {out_content}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn extract_dialogue_lines_from_srt() {
    let tmp = tempfile_path();
    let srt_file = make_srt_file(&tmp);
    let lines = SubtitleProcessor::extract_dialogue_lines(&srt_file).unwrap();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0].text, "Hello, how are you?");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn filters_signs_style() {
    let tmp = tempfile_path();
    let ass_file = make_ass_file(&tmp);
    let lines = SubtitleProcessor::extract_dialogue_lines(&ass_file).unwrap();
    let texts: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
    assert!(!texts.contains(&"EPISODE 1"), "Signs events should be filtered");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn timing_types_are_correct() {
    let tmp = tempfile_path();
    let ass_file = make_ass_file(&tmp);
    let lines = SubtitleProcessor::extract_dialogue_lines(&ass_file).unwrap();
    for line in &lines {
        assert!(
            line.end_ms > line.start_ms,
            "end_ms must be greater than start_ms"
        );
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn raises_for_nonexistent_file() {
    let result = SubtitleProcessor::extract_dialogue_lines(Path::new("/nonexistent/file.ass"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("not found") || msg.contains("No such"),
        "error: {msg}"
    );
}

#[test]
fn returns_empty_for_no_dialogue() {
    let tmp = tempfile_path();
    let no_dialogue = tmp.join("no_dialogue.ass");
    let content = "[Script Info]\nTitle: Test\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Signs,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Signs,,0,0,0,,EPISODE 1\n";
    std::fs::write(&no_dialogue, content).unwrap();
    let lines = SubtitleProcessor::extract_dialogue_lines(&no_dialogue).unwrap();
    assert_eq!(lines.len(), 0);
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn create_english_subtitles() {
    use mt_core::types::DialogueLine;
    let tmp = tempfile_path();
    let original = make_ass_file(&tmp);
    let output = tmp.join("english.ass");
    let lines = vec![
        DialogueLine {
            start_ms: 1000,
            end_ms: 3000,
            text: "Hello, how are you?".to_string(),
        },
        DialogueLine {
            start_ms: 4000,
            end_ms: 6000,
            text: "I am fine, thank you.".to_string(),
        },
    ];
    SubtitleProcessor::create_english_subtitles(&original, &lines, &output).unwrap();
    assert!(output.exists());
    let content = std::fs::read_to_string(&output).unwrap();
    assert!(content.contains("Hello, how are you?"));
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn create_polish_subtitles_with_replacement() {
    use mt_core::types::DialogueLine;
    let tmp = tempfile_path();
    let original = make_ass_file(&tmp);
    let output = tmp.join("polish.ass");
    let lines = vec![DialogueLine {
        start_ms: 1000,
        end_ms: 3000,
        text: "Cześć".to_string(),
    }];
    SubtitleProcessor::create_polish_subtitles(&original, &lines, &output, true).unwrap();
    let content = std::fs::read_to_string(&output).unwrap();
    assert!(
        content.contains("Czesc") || content.contains("Cześć"),
        "content: {content}"
    );
    assert!(!content.contains('ś'), "Polish char 'ś' should be replaced");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn create_polish_subtitles_without_replacement() {
    use mt_core::types::DialogueLine;
    let tmp = tempfile_path();
    let original = make_ass_file(&tmp);
    let output = tmp.join("polish.ass");
    let lines = vec![DialogueLine {
        start_ms: 1000,
        end_ms: 3000,
        text: "Cześć".to_string(),
    }];
    SubtitleProcessor::create_polish_subtitles(&original, &lines, &output, false).unwrap();
    let content = std::fs::read_to_string(&output).unwrap();
    assert!(
        content.contains("Cześć"),
        "Polish chars should be preserved when replace_chars=false"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn override_font_name() {
    let tmp = tempfile_path();
    let ass_file = make_ass_file(&tmp);
    SubtitleProcessor::override_font_name(&ass_file, "DejaVu Sans").unwrap();
    let content = std::fs::read_to_string(&ass_file).unwrap();
    for line in content.lines().filter(|l| l.starts_with("Style:")) {
        let fields: Vec<&str> = line["Style:".len()..].trim().split(',').collect();
        assert_eq!(fields[1], "DejaVu Sans", "fontname field: {line}");
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn find_dialogue_style_prefers_default() {
    use crate::ass::load_ass;
    let subs = load_ass(make_ass_content()).unwrap();
    let style = find_dialogue_style(&subs);
    assert_eq!(style, "Default");
}

#[test]
fn find_dialogue_style_dialogue_name() {
    use crate::ass::load_ass;
    let content = "[Script Info]\nTitle: Test\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Dialogue,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\nStyle: Signs,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Dialogue,,0,0,0,,Hello\n";
    let subs = load_ass(content).unwrap();
    let style = find_dialogue_style(&subs);
    assert_eq!(style, "Dialogue");
}

#[test]
fn find_dialogue_style_empty_styles() {
    use crate::srt::load_srt;
    let subs = load_srt("1\n00:00:01,000 --> 00:00:02,000\nHello\n").unwrap();
    let style = find_dialogue_style(&subs);
    assert_eq!(style, "Default");
}

#[test]
fn deduplicate_consecutive_same_text() {
    let events_in = vec![
        mk_event("Hello", 0, 1000, 2000),
        mk_event("Hello", 0, 2100, 3000),
        mk_event("World", 0, 3500, 4500),
    ];
    let result = SubtitleProcessor::deduplicate_events(events_in);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].text, "Hello");
    assert_eq!(result[0].end_ms, 3000);
    assert_eq!(result[1].text, "World");
}

#[test]
fn deduplicate_non_consecutive_same_text_kept_separate() {
    let events_in = vec![
        mk_event("Hello", 0, 1000, 2000),
        mk_event("World", 0, 2100, 3000),
        mk_event("Hello", 0, 3500, 4500),
    ];
    let result = SubtitleProcessor::deduplicate_events(events_in);
    assert_eq!(result.len(), 3);
}

fn mk_event(text: &str, layer: i32, start_ms: i64, end_ms: i64) -> Event {
    Event {
        kind: EventKind::Dialogue,
        layer,
        start_ms,
        end_ms,
        style: "Default".to_string(),
        name: String::new(),
        margin_l: 0,
        margin_r: 0,
        margin_v: 0,
        effect: String::new(),
        text: text.to_string(),
    }
}

fn tempfile_path() -> std::path::PathBuf {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("mt_test_{id}_{:?}", std::thread::current().id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}
