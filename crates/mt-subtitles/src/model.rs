//! In-memory model for subtitle files (ASS and SRT).

/// Whether an event is dialogue or a comment (chapter marker, editor note).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Dialogue,
    Comment,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Dialogue => "Dialogue",
            EventKind::Comment => "Comment",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Dialogue" => Some(EventKind::Dialogue),
            "Comment" => Some(EventKind::Comment),
            _ => None,
        }
    }
}

/// An ASS timestamp stored as milliseconds (matching pysubs2 convention).
///
/// ASS timing format: `H:MM:SS.cs` (centiseconds, 2 digits).
/// Conversion: `cs * 10 = ms`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AssTime(pub i64);

impl AssTime {
    /// Parse `"H:MM:SS.cs"` → milliseconds.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let mut parts = s.splitn(2, ':');
        let h: i64 = parts
            .next()
            .ok_or("missing hours")?
            .parse()
            .map_err(|e| format!("{e}"))?;
        let rest = parts.next().ok_or("missing minutes")?;
        let mut parts2 = rest.splitn(2, ':');
        let m: i64 = parts2
            .next()
            .ok_or("missing minutes")?
            .parse()
            .map_err(|e| format!("{e}"))?;
        let rest2 = parts2.next().ok_or("missing seconds")?;
        let mut parts3 = rest2.splitn(2, '.');
        let sec: i64 = parts3
            .next()
            .ok_or("missing seconds")?
            .parse()
            .map_err(|e| format!("{e}"))?;
        let cs: i64 = parts3
            .next()
            .ok_or("missing centiseconds")?
            .parse()
            .map_err(|e| format!("{e}"))?;
        let ms = h * 3_600_000 + m * 60_000 + sec * 1_000 + cs * 10;
        Ok(AssTime(ms))
    }

    /// Format back to `"H:MM:SS.cs"`.
    pub fn format(self) -> String {
        let ms = self.0;
        let sign = if ms < 0 { "-" } else { "" };
        let ms_abs = ms.abs();
        let h = ms_abs / 3_600_000;
        let m = (ms_abs % 3_600_000) / 60_000;
        let s = (ms_abs % 60_000) / 1_000;
        let cs = (ms_abs % 1_000) / 10;
        format!("{sign}{h}:{m:02}:{s:02}.{cs:02}")
    }
}

/// A single `[V4+ Styles]` Style.
///
/// All fields are stored as the raw comma-separated string from the file so that
/// unknown/extra fields are preserved verbatim. The `name` field is also parsed out
/// for quick lookup.
#[derive(Debug, Clone)]
pub struct Style {
    /// Parsed name (first field of the raw Style line).
    pub name: String,
    /// Raw comma-separated fields after `"Style: "` (e.g. `"Default,Arial,48,..."`).
    pub raw: String,
}

/// A single `[Events]` Dialogue or Comment line — all 10 standard fields.
#[derive(Debug, Clone)]
pub struct Event {
    pub kind: EventKind,
    pub layer: i32,
    pub start_ms: i64,
    pub end_ms: i64,
    pub style: String,
    pub name: String,
    pub margin_l: i32,
    pub margin_r: i32,
    pub margin_v: i32,
    pub effect: String,
    /// Raw text including ASS override tags like `{\i1}`.
    pub text: String,
}

impl Event {
    /// Return the plain text with ASS `{...}` override blocks stripped.
    pub fn plaintext(&self) -> String {
        strip_ass_overrides(&self.text)
    }
}

/// Strip ASS override blocks `{...}` from text.
pub fn strip_ass_overrides(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for ch in text.chars() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
            }
            _ => {
                if depth == 0 {
                    out.push(ch);
                }
            }
        }
    }
    out
}

/// An unknown/extra section preserved verbatim (e.g. `[Aegisub Project Garbage]`).
#[derive(Debug, Clone)]
pub struct RawSection {
    /// The full `[Header]` line.
    pub header: String,
    /// Body lines (not including the header).
    pub lines: Vec<String>,
}

/// Top-level subtitle file model.
///
/// Preserves all sections in original order to allow round-trip serialization.
#[derive(Debug, Clone)]
pub struct Subtitles {
    /// Whether a BOM (`\u{FEFF}`) was present at the start of the file.
    pub bom: bool,
    /// `[Script Info]` body lines (raw, including blank lines).
    pub script_info_lines: Vec<String>,
    /// Unknown sections before `[V4+ Styles]` (e.g. `[Aegisub Project Garbage]`).
    pub pre_styles_sections: Vec<RawSection>,
    /// Field names from the `[V4+ Styles]` `Format:` line.
    pub styles_format: Vec<String>,
    /// Parsed styles.
    pub styles: Vec<Style>,
    /// Unknown sections between `[V4+ Styles]` and `[Events]`.
    pub pre_events_sections: Vec<RawSection>,
    /// Field names from the `[Events]` `Format:` line.
    pub events_format: Vec<String>,
    /// Parsed events.
    pub events: Vec<Event>,
    /// Unknown sections after `[Events]` (e.g. `[Aegisub Extradata]`).
    pub post_events_sections: Vec<RawSection>,
}

impl Subtitles {
    /// Find a style by name (case-sensitive).
    pub fn find_style(&self, name: &str) -> Option<&Style> {
        self.styles.iter().find(|s| s.name == name)
    }

    /// Return a list of style names.
    pub fn style_names(&self) -> Vec<&str> {
        self.styles.iter().map(|s| s.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ass_time_roundtrip_zero() {
        let t = AssTime::parse("0:00:00.00").unwrap();
        assert_eq!(t.0, 0);
        assert_eq!(t.format(), "0:00:00.00");
    }

    #[test]
    fn ass_time_roundtrip_known() {
        // From spike: "0:00:16.71" → 16710 ms
        let t = AssTime::parse("0:00:16.71").unwrap();
        assert_eq!(t.0, 16710);
        assert_eq!(t.format(), "0:00:16.71");
    }

    #[test]
    fn ass_time_hours() {
        let t = AssTime::parse("1:23:45.67").unwrap();
        let expected = 3_600_000 + 23 * 60_000 + 45 * 1_000 + 67 * 10;
        assert_eq!(t.0, expected);
        assert_eq!(t.format(), "1:23:45.67");
    }

    #[test]
    fn strip_ass_overrides_no_tags() {
        assert_eq!(strip_ass_overrides("Hello world"), "Hello world");
    }

    #[test]
    fn strip_ass_overrides_italic() {
        assert_eq!(strip_ass_overrides("{\\i1}Hello{\\i0}"), "Hello");
    }

    #[test]
    fn strip_ass_overrides_nested_braces() {
        assert_eq!(strip_ass_overrides("{=71}text"), "text");
    }

    #[test]
    fn event_kind_roundtrip() {
        assert_eq!(EventKind::from_str("Dialogue"), Some(EventKind::Dialogue));
        assert_eq!(EventKind::from_str("Comment"), Some(EventKind::Comment));
        assert_eq!(EventKind::from_str("Unknown"), None);
        assert_eq!(EventKind::Dialogue.as_str(), "Dialogue");
        assert_eq!(EventKind::Comment.as_str(), "Comment");
    }
}
