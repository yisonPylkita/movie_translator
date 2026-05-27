//! SRT (SubRip Text) parser and serializer.

use crate::model::{Event, EventKind, Subtitles};

/// Parse an SRT file from a string.
pub fn load_srt(input: &str) -> Result<Subtitles, String> {
    // Strip a leading UTF-8 BOM if present.
    let input = input.strip_prefix('\u{FEFF}').unwrap_or(input);

    let mut events: Vec<Event> = Vec::new();

    // SRT blocks are separated by blank lines.
    // Each block: index line, timing line, text lines.
    let mut block: Vec<&str> = Vec::new();

    for line in input.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !block.is_empty() {
                if let Some(event) = parse_srt_block(&block)? {
                    events.push(event);
                }
                block.clear();
            }
        } else {
            block.push(line);
        }
    }

    Ok(Subtitles {
        bom: false,
        script_info_lines: Vec::new(),
        pre_styles_sections: Vec::new(),
        styles_format: Vec::new(),
        styles: Vec::new(),
        pre_events_sections: Vec::new(),
        events_format: Vec::new(),
        events,
        post_events_sections: Vec::new(),
    })
}

fn parse_srt_block(block: &[&str]) -> Result<Option<Event>, String> {
    // First line is the sequence number (ignore). Need at least number + timing.
    if block.len() < 2 {
        return Ok(None);
    }

    // Second line: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
    let timing_line = block[1].trim();
    let arrow = "-->";
    let pos = timing_line
        .find(arrow)
        .ok_or_else(|| format!("no '-->' in timing line: {timing_line}"))?;
    let start_str = timing_line[..pos].trim();
    let end_part = timing_line[pos + arrow.len()..].trim();

    // Handle optional position info after end time
    // (e.g. "00:00:01,000 --> 00:00:02,000  X1:0 X2:0 Y1:0 Y2:0")
    let end_str = end_part.split_whitespace().next().unwrap_or(end_part);

    let start_ms = parse_srt_time(start_str)?;
    let end_ms = parse_srt_time(end_str)?;

    // Remaining lines are text
    let text = if block.len() > 2 {
        block[2..].join("\n")
    } else {
        String::new()
    };

    if text.trim().is_empty() {
        return Ok(None);
    }

    Ok(Some(Event {
        kind: EventKind::Dialogue,
        layer: 0,
        start_ms,
        end_ms,
        style: "Default".to_string(),
        name: String::new(),
        margin_l: 0,
        margin_r: 0,
        margin_v: 0,
        effect: String::new(),
        text,
    }))
}

fn parse_srt_time(s: &str) -> Result<i64, String> {
    // Format: HH:MM:SS,mmm (also accept '.' as the ms separator)
    let s = s.trim();
    let sep_pos = s
        .rfind([',', '.'])
        .ok_or_else(|| format!("no ms separator in SRT time: {s}"))?;
    let hms = &s[..sep_pos];
    let ms_str = &s[sep_pos + 1..];

    let ms: i64 = ms_str.parse().map_err(|e| format!("ms parse: {e}"))?;

    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return Err(format!("expected HH:MM:SS, got: {hms}"));
    }
    let h: i64 = parts[0].trim().parse().map_err(|e| format!("hours: {e}"))?;
    let m: i64 = parts[1]
        .trim()
        .parse()
        .map_err(|e| format!("minutes: {e}"))?;
    let sec: i64 = parts[2]
        .trim()
        .parse()
        .map_err(|e| format!("seconds: {e}"))?;

    Ok(h * 3_600_000 + m * 60_000 + sec * 1_000 + ms)
}

fn format_srt_time(ms: i64) -> String {
    let ms = ms.max(0);
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02},{millis:03}")
}

/// Serialize a `Subtitles` to SRT format.
///
/// Only `Dialogue` events are emitted; `Comment` events are skipped.
pub fn to_srt_string(subs: &Subtitles) -> String {
    let mut out = String::new();
    let mut idx = 1usize;

    for event in &subs.events {
        if event.kind != EventKind::Dialogue {
            continue;
        }
        out.push_str(&idx.to_string());
        out.push('\n');
        out.push_str(&format_srt_time(event.start_ms));
        out.push_str(" --> ");
        out.push_str(&format_srt_time(event.end_ms));
        out.push('\n');
        out.push_str(&event.text);
        out.push('\n');
        out.push('\n');
        idx += 1;
    }

    out
}
