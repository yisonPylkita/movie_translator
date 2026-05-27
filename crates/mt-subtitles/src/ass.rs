//! ASS (Advanced SubStation Alpha) parser and serializer.

use crate::model::{AssTime, Event, EventKind, RawSection, Style, Subtitles};

/// Parse an ASS file from a string.
///
/// Handles:
/// - BOM (`\u{FEFF}`)
/// - `[Script Info]`, `[V4+ Styles]`, `[Events]` sections
/// - Unknown sections (preserved as `RawSection`)
/// - Timing: `H:MM:SS.cs` → ms via `AssTime::parse`
/// - `Text` field as last (may contain commas): `splitn(n, ',')`
pub fn load_ass(input: &str) -> Result<Subtitles, String> {
    // Handle BOM
    let (bom, s) = if let Some(stripped) = input.strip_prefix('\u{FEFF}') {
        (true, stripped)
    } else {
        (false, input)
    };

    let mut script_info_lines: Vec<String> = Vec::new();
    let mut pre_styles_sections: Vec<RawSection> = Vec::new();
    let mut styles_format: Vec<String> = Vec::new();
    let mut styles: Vec<Style> = Vec::new();
    let mut pre_events_sections: Vec<RawSection> = Vec::new();
    let mut events_format: Vec<String> = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    let mut post_events_sections: Vec<RawSection> = Vec::new();

    #[derive(Debug, PartialEq)]
    enum Section {
        None,
        ScriptInfo,
        Styles,
        Events,
        Other,
    }

    let mut current_section = Section::None;
    let mut current_other: Option<RawSection> = None;
    let mut found_styles = false;
    let mut found_events = false;

    fn flush_other(
        current_other: Option<RawSection>,
        found_styles: bool,
        found_events: bool,
        pre_styles_sections: &mut Vec<RawSection>,
        pre_events_sections: &mut Vec<RawSection>,
        post_events_sections: &mut Vec<RawSection>,
    ) {
        if let Some(sec) = current_other {
            if !found_styles {
                pre_styles_sections.push(sec);
            } else if !found_events {
                pre_events_sections.push(sec);
            } else {
                post_events_sections.push(sec);
            }
        }
    }

    for line in s.lines() {
        let trimmed = line.trim();

        // Section headers
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            flush_other(
                current_other.take(),
                found_styles,
                found_events,
                &mut pre_styles_sections,
                &mut pre_events_sections,
                &mut post_events_sections,
            );

            let header = &trimmed[1..trimmed.len() - 1];
            match header {
                "Script Info" => current_section = Section::ScriptInfo,
                "V4+ Styles" | "V4 Styles" => {
                    found_styles = true;
                    current_section = Section::Styles;
                }
                "Events" => {
                    found_events = true;
                    current_section = Section::Events;
                }
                _ => {
                    current_other = Some(RawSection {
                        header: trimmed.to_string(),
                        lines: Vec::new(),
                    });
                    current_section = Section::Other;
                }
            }
            continue;
        }

        match current_section {
            Section::None => {}
            Section::ScriptInfo => {
                script_info_lines.push(line.to_string());
            }
            Section::Styles => {
                if let Some(fields_str) = trimmed.strip_prefix("Format:") {
                    let fields_str = fields_str.trim_start();
                    styles_format = fields_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect();
                } else if let Some(raw) = trimmed.strip_prefix("Style:") {
                    let raw = raw.trim_start().to_string();
                    let name = raw.split(',').next().unwrap_or("").to_string();
                    styles.push(Style { name, raw });
                }
                // Blank lines inside sections are ignored
            }
            Section::Events => {
                if let Some(fields_str) = trimmed.strip_prefix("Format:") {
                    let fields_str = fields_str.trim_start();
                    events_format = fields_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect();
                } else if trimmed.starts_with("Dialogue:") || trimmed.starts_with("Comment:") {
                    if events_format.is_empty() {
                        return Err("encountered Dialogue/Comment before Format line".to_string());
                    }
                    let event = parse_event(trimmed, &events_format)?;
                    events.push(event);
                }
            }
            Section::Other => {
                if let Some(ref mut sec) = current_other {
                    sec.lines.push(line.to_string());
                }
            }
        }
    }

    // Flush final raw section
    flush_other(
        current_other,
        found_styles,
        found_events,
        &mut pre_styles_sections,
        &mut pre_events_sections,
        &mut post_events_sections,
    );

    Ok(Subtitles {
        bom,
        script_info_lines,
        pre_styles_sections,
        styles_format,
        styles,
        pre_events_sections,
        events_format,
        events,
        post_events_sections,
    })
}

fn parse_event(line: &str, field_order: &[String]) -> Result<Event, String> {
    let (kind, rest) = if let Some(r) = line.strip_prefix("Dialogue:") {
        (EventKind::Dialogue, r)
    } else if let Some(r) = line.strip_prefix("Comment:") {
        (EventKind::Comment, r)
    } else {
        return Err(format!("not a Dialogue/Comment line: {line}"));
    };

    let rest = rest.trim_start_matches(' ');
    let n = field_order.len();
    let fields: Vec<&str> = rest.splitn(n, ',').collect();
    if fields.len() < n {
        return Err(format!(
            "expected {n} fields, got {} in: {line}",
            fields.len()
        ));
    }

    // Build map from field name to raw value
    let field_map: std::collections::HashMap<&str, &str> = field_order
        .iter()
        .map(|s| s.as_str())
        .zip(fields.iter().copied())
        .collect();

    let get = |name: &str| -> Result<&str, String> {
        field_map
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing field {name}"))
    };

    // ASS v4+ uses `Layer`; legacy SSA v4 uses `Marked` (value `Marked=N`).
    // pysubs2 treats SSA `Marked` as the layer source, so do the same.
    let layer_field = match field_map.get("Layer") {
        Some(v) => *v,
        None => field_map
            .get("Marked")
            .copied()
            .ok_or("missing field Layer")?,
    };
    let layer_value = layer_field
        .trim()
        .strip_prefix("Marked=")
        .unwrap_or_else(|| layer_field.trim());
    let layer: i32 = layer_value.parse().map_err(|e| format!("Layer: {e}"))?;
    let start_ms = AssTime::parse(get("Start")?)?.0;
    let end_ms = AssTime::parse(get("End")?)?.0;

    let margin_l: i32 = get("MarginL")?
        .trim()
        .parse()
        .map_err(|e| format!("MarginL: {e}"))?;
    let margin_r: i32 = get("MarginR")?
        .trim()
        .parse()
        .map_err(|e| format!("MarginR: {e}"))?;
    let margin_v: i32 = get("MarginV")?
        .trim()
        .parse()
        .map_err(|e| format!("MarginV: {e}"))?;

    Ok(Event {
        kind,
        layer,
        start_ms,
        end_ms,
        style: get("Style")?.to_string(),
        name: get("Name")?.to_string(),
        margin_l,
        margin_r,
        margin_v,
        effect: get("Effect")?.to_string(),
        text: get("Text")?.to_string(),
    })
}

/// Serialize a `Subtitles` back to an ASS string.
///
/// Output format matches pysubs2's default output:
/// - BOM re-emitted if original had one
/// - Blank line before each section header (except `[Script Info]`)
/// - `Format:` fields joined with `", "` (comma-space)
/// - `Style:` raw field line prepended with `"Style: "`
/// - Event line: `"Dialogue: "` or `"Comment: "` + fields joined with `","` (no spaces)
pub fn to_ass_string(subs: &Subtitles) -> String {
    let mut out = String::new();

    if subs.bom {
        out.push('\u{FEFF}');
    }

    // [Script Info]
    out.push_str("[Script Info]\n");
    for line in &subs.script_info_lines {
        out.push_str(line);
        out.push('\n');
    }

    // Pre-styles sections
    for sec in &subs.pre_styles_sections {
        out.push('\n');
        out.push_str(&sec.header);
        out.push('\n');
        for line in &sec.lines {
            out.push_str(line);
            out.push('\n');
        }
    }

    // [V4+ Styles]
    out.push('\n');
    out.push_str("[V4+ Styles]\n");
    out.push_str("Format: ");
    out.push_str(&subs.styles_format.join(", "));
    out.push('\n');
    for style in &subs.styles {
        out.push_str("Style: ");
        out.push_str(&style.raw);
        out.push('\n');
    }

    // Pre-events sections
    for sec in &subs.pre_events_sections {
        out.push('\n');
        out.push_str(&sec.header);
        out.push('\n');
        for line in &sec.lines {
            out.push_str(line);
            out.push('\n');
        }
    }

    // [Events]
    out.push('\n');
    out.push_str("[Events]\n");
    out.push_str("Format: ");
    out.push_str(&subs.events_format.join(", "));
    out.push('\n');
    for event in &subs.events {
        out.push_str(event.kind.as_str());
        out.push_str(": ");
        let mut parts = Vec::with_capacity(subs.events_format.len());
        for field in &subs.events_format {
            let v = match field.as_str() {
                "Layer" => event.layer.to_string(),
                "Start" => AssTime(event.start_ms).format(),
                "End" => AssTime(event.end_ms).format(),
                "Style" => event.style.clone(),
                "Name" => event.name.clone(),
                "MarginL" => event.margin_l.to_string(),
                "MarginR" => event.margin_r.to_string(),
                "MarginV" => event.margin_v.to_string(),
                "Effect" => event.effect.clone(),
                "Text" => event.text.clone(),
                other => format!("<unknown:{other}>"),
            };
            parts.push(v);
        }
        out.push_str(&parts.join(","));
        out.push('\n');
    }

    // Post-events sections
    for sec in &subs.post_events_sections {
        out.push('\n');
        out.push_str(&sec.header);
        out.push('\n');
        for line in &sec.lines {
            if !line.is_empty() {
                out.push_str(line);
                out.push('\n');
            }
        }
    }

    out
}
