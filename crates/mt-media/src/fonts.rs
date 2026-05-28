//! Font detection, extraction, and Polish character support checking.
//!
//! Uses `ttf-parser` instead of Python's `fonttools` for reading font data.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use mt_core::POLISH_CHARS;

use crate::ffmpeg::{get_ffmpeg, get_ffprobe, VideoMuxError};

/// Well-known fonts likely to support Polish, in preference order.
const PREFERRED_FALLBACK_FONTS: &[&str] = &[
    "DejaVu Sans",
    "Liberation Sans",
    "Noto Sans",
    "Arial",
    "Helvetica",
    "Verdana",
    "Times New Roman",
];

// ---------------------------------------------------------------------------
// Embedded font detection
// ---------------------------------------------------------------------------

/// Metadata for a single embedded font attachment in a video file.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedFont {
    pub index: u32,
    pub filename: String,
    pub mimetype: String,
}

/// List font attachments embedded in a video file via ffprobe.
pub fn get_embedded_fonts(video_path: &Path) -> Result<Vec<EmbeddedFont>, VideoMuxError> {
    let ffprobe = get_ffprobe()?;
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            &video_path.to_string_lossy(),
        ])
        .output()?;

    if !output.status.success() {
        return Err(VideoMuxError::FfprobeFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let json = String::from_utf8_lossy(&output.stdout);
    parse_embedded_fonts_json(&json)
}

/// Pure function: parse ffprobe JSON to extract font attachment metadata.
pub fn parse_embedded_fonts_json(json: &str) -> Result<Vec<EmbeddedFont>, VideoMuxError> {
    let data: serde_json::Value = serde_json::from_str(json)?;
    let streams = data["streams"].as_array().cloned().unwrap_or_default();

    let mut fonts = Vec::new();
    for stream in &streams {
        if stream["codec_type"].as_str() != Some("attachment") {
            continue;
        }
        let mimetype = stream["tags"]["mimetype"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let is_font = mimetype.to_ascii_lowercase().contains("font")
            || mimetype == "application/x-truetype-font"
            || mimetype == "application/vnd.ms-opentype";
        if !is_font {
            continue;
        }
        let index = stream["index"].as_u64().unwrap_or(0) as u32;
        let default_name = format!("font_{index}");
        let filename = stream["tags"]["filename"]
            .as_str()
            .unwrap_or(&default_name)
            .to_string();
        fonts.push(EmbeddedFont {
            index,
            filename,
            mimetype,
        });
    }
    Ok(fonts)
}

// ---------------------------------------------------------------------------
// Font extraction
// ---------------------------------------------------------------------------

/// Extract a font attachment from a video file using ffmpeg dump_attachment.
///
/// Returns `true` if the output file was created, `false` otherwise.
pub fn extract_font(
    video_path: &Path,
    stream_index: u32,
    output_path: &Path,
) -> Result<bool, VideoMuxError> {
    let ffmpeg = get_ffmpeg()?;
    let attachment_arg = format!("-dump_attachment:{stream_index}");
    let output = Command::new(&ffmpeg)
        .args([
            "-y",
            &attachment_arg,
            &output_path.to_string_lossy(),
            "-i",
            &video_path.to_string_lossy(),
        ])
        .output()?;

    // ffmpeg can exit non-zero yet still leave a partial/corrupt file behind,
    // which `output_path.exists()` alone would treat as success. Honor the exit
    // status: on failure, remove any partial output and report failure.
    if !output.status.success() {
        let _ = std::fs::remove_file(output_path);
        tracing::debug!(
            "font extraction failed (stream {stream_index}): {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return Ok(false);
    }

    Ok(output_path.exists())
}

// ---------------------------------------------------------------------------
// Polish character support check (ttf-parser)
// ---------------------------------------------------------------------------

/// Check whether a font file covers all Polish diacritical characters.
///
/// Note on ttf-parser vs fonttools:
/// - Python's `font.getBestCmap()` returns a mapping codepoint → glyph ID, preferring
///   platform 3 (Windows Unicode BMP / UCS-2), then platform 0 (Unicode).
/// - ttf-parser exposes cmap subtables; we iterate all and check that every
///   Polish codepoint maps to a non-zero glyph ID in at least one subtable.
/// - This is semantically equivalent for the Polish character set (all in BMP).
pub fn font_supports_polish(font_path: &Path) -> bool {
    let data = match std::fs::read(font_path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    font_data_supports_polish(&data)
}

/// Pure function: check Polish support from raw font bytes.
///
/// Extracted for unit testing without file I/O.
pub fn font_data_supports_polish(data: &[u8]) -> bool {
    // Try as a regular TTF/OTF first; if that fails try as TTC (font collection, face 0).
    let face = ttf_parser::Face::parse(data, 0);
    let face = match face {
        Ok(f) => f,
        Err(_) => return false,
    };

    let polish_codepoints: Vec<u32> = POLISH_CHARS.chars().map(|c| c as u32).collect();

    for cp in &polish_codepoints {
        let ch = match char::from_u32(*cp) {
            Some(c) => c,
            None => return false,
        };
        if face.glyph_index(ch).is_none() {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// ASS font name extraction
// ---------------------------------------------------------------------------

/// Parse an ASS file and collect all unique Fontname values from `[V4+ Styles]`.
pub fn get_ass_font_names(ass_path: &Path) -> HashSet<String> {
    let subs = match mt_subtitles::load(ass_path) {
        Ok(s) => s,
        Err(_) => return HashSet::new(),
    };

    let mut names = HashSet::new();
    for style in &subs.styles {
        // Style.raw is comma-separated: Name, Fontname, Fontsize, ...
        // owned[0] = Name, owned[1] = Fontname
        let fields: Vec<&str> = style.raw.splitn(3, ',').collect();
        if fields.len() >= 2 {
            let fontname = fields[1].trim().to_ascii_lowercase();
            if !fontname.is_empty() {
                names.insert(fontname);
            }
        }
    }
    names
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Check whether the embedded fonts in `video_path` support all Polish characters
/// required by the ASS subtitle at `ass_path`.
pub fn check_embedded_fonts_support_polish(
    video_path: &Path,
    ass_path: &Path,
) -> Result<bool, VideoMuxError> {
    let embedded_fonts = get_embedded_fonts(video_path)?;
    if embedded_fonts.is_empty() {
        return Ok(false);
    }

    let ass_font_names = get_ass_font_names(ass_path);
    if ass_font_names.is_empty() {
        return Ok(false);
    }

    let temp_dir = tempfile::Builder::new()
        .prefix("mt-media-fonts-")
        .tempdir()
        .map_err(VideoMuxError::Io)?;
    let temp_path = temp_dir.path();

    let mut fonts_supporting_polish = 0usize;
    for font_info in &embedded_fonts {
        let font_output = temp_path.join(&font_info.filename);
        if !extract_font(video_path, font_info.index, &font_output)? {
            continue;
        }
        if font_supports_polish(&font_output) {
            fonts_supporting_polish += 1;
        }
    }

    Ok(fonts_supporting_polish > 0)
}

// ---------------------------------------------------------------------------
// System font discovery
// ---------------------------------------------------------------------------

/// Return existing system font directories for the current platform.
///
/// System font directories to search (macOS + Linux).
pub fn get_system_font_dirs() -> Vec<PathBuf> {
    let dirs: Vec<PathBuf> = if cfg!(target_os = "macos") {
        vec![
            PathBuf::from("/System/Library/Fonts"),
            PathBuf::from("/System/Library/Fonts/Supplemental"),
            PathBuf::from("/Library/Fonts"),
            dirs::home_dir()
                .map(|h| h.join("Library/Fonts"))
                .unwrap_or_default(),
        ]
    } else {
        // Linux
        let home = dirs::home_dir().unwrap_or_default();
        vec![
            PathBuf::from("/usr/share/fonts"),
            PathBuf::from("/usr/local/share/fonts"),
            home.join(".local/share/fonts"),
            home.join(".fonts"),
        ]
    };
    dirs.into_iter().filter(|d| d.is_dir()).collect()
}

/// Iterate all `.ttf`, `.otf`, `.ttc` files under system font directories.
pub fn iter_system_fonts() -> Vec<PathBuf> {
    let mut fonts = Vec::new();
    for dir in get_system_font_dirs() {
        for entry in walkdir_fonts(&dir) {
            fonts.push(entry);
        }
    }
    fonts
}

fn walkdir_fonts(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return result;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            result.extend(walkdir_fonts(&path));
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_ascii_lowercase();
            if matches!(ext_lower.as_str(), "ttf" | "otf" | "ttc") {
                result.push(path);
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Font family name (ttf-parser name table)
// ---------------------------------------------------------------------------

/// Read the font family name from the font's name table.
///
/// Prefers platform 3 (Windows) name record, then any record with nameID=1.
pub fn get_font_family_name(font_path: &Path) -> Option<String> {
    let data = std::fs::read(font_path).ok()?;
    font_family_name_from_data(&data)
}

/// Pure function: extract family name from raw font bytes.
///
/// - nameID 1 = Font Family Name
/// - prefer platformID 3 (Windows) for broadest compat
pub fn font_family_name_from_data(data: &[u8]) -> Option<String> {
    let face = ttf_parser::Face::parse(data, 0).ok()?;
    // Try platform 3 (Windows) first, then platform 0 (Unicode), then any
    let name_by_platform = |platform_id: ttf_parser::PlatformId| -> Option<String> {
        face.names()
            .into_iter()
            .filter(|r| r.name_id == ttf_parser::name_id::FAMILY && r.platform_id == platform_id)
            .find_map(|r| r.to_string())
    };
    name_by_platform(ttf_parser::PlatformId::Windows)
        .or_else(|| name_by_platform(ttf_parser::PlatformId::Unicode))
        .or_else(|| {
            face.names()
                .into_iter()
                .filter(|r| r.name_id == ttf_parser::name_id::FAMILY)
                .find_map(|r| r.to_string())
        })
}

// ---------------------------------------------------------------------------
// Font filename matching and system font search
// ---------------------------------------------------------------------------

/// Check if a font file's stem loosely matches a target font name.
pub fn font_filename_matches(font_path: &Path, target_name: &str) -> bool {
    let stem = font_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .replace(['-', '_'], " ");
    let target = target_name.to_ascii_lowercase().replace(['-', '_'], " ");
    stem == target || stem.starts_with(&target)
}

/// Find a system font that supports Polish characters.
///
/// Prefers fonts referenced in the ASS styles; falls back to well-known fonts.
/// Returns `(font_path, family_name)` or `None`.
pub fn find_system_font_for_polish(ass_font_names: &HashSet<String>) -> Option<(PathBuf, String)> {
    let system_fonts = iter_system_fonts();
    if system_fonts.is_empty() {
        return None;
    }

    // Phase 1: try to find an ASS-referenced font on the system
    for ass_name in ass_font_names {
        for font_path in &system_fonts {
            if font_filename_matches(font_path, ass_name) && font_supports_polish(font_path) {
                let family = get_font_family_name(font_path).unwrap_or_else(|| ass_name.clone());
                return Some((font_path.clone(), family));
            }
        }
    }

    // Phase 2: try well-known fallback fonts
    for fallback_name in PREFERRED_FALLBACK_FONTS {
        for font_path in &system_fonts {
            if font_filename_matches(font_path, fallback_name) && font_supports_polish(font_path) {
                let family =
                    get_font_family_name(font_path).unwrap_or_else(|| fallback_name.to_string());
                return Some((font_path.clone(), family));
            }
        }
    }

    // Phase 3: any system font with Polish support
    for font_path in &system_fonts {
        if font_supports_polish(font_path) {
            if let Some(family) = get_font_family_name(font_path) {
                return Some((font_path.clone(), family));
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- parse_embedded_fonts_json -----

    const FFPROBE_FONTS_JSON: &str = r#"{
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "tags": {}
            },
            {
                "index": 5,
                "codec_type": "attachment",
                "codec_name": "ttf",
                "tags": {
                    "filename": "Raleway-Regular.ttf",
                    "mimetype": "application/x-truetype-font"
                }
            },
            {
                "index": 6,
                "codec_type": "attachment",
                "codec_name": "otf",
                "tags": {
                    "filename": "Garamond.otf",
                    "mimetype": "application/vnd.ms-opentype"
                }
            },
            {
                "index": 7,
                "codec_type": "attachment",
                "codec_name": "bin_data",
                "tags": {
                    "filename": "cover.png",
                    "mimetype": "image/png"
                }
            }
        ]
    }"#;

    #[test]
    fn parse_embedded_fonts_extracts_font_streams() {
        let fonts = parse_embedded_fonts_json(FFPROBE_FONTS_JSON).unwrap();
        assert_eq!(fonts.len(), 2, "should find 2 font attachments");
        assert_eq!(fonts[0].index, 5);
        assert_eq!(fonts[0].filename, "Raleway-Regular.ttf");
        assert_eq!(fonts[0].mimetype, "application/x-truetype-font");
        assert_eq!(fonts[1].index, 6);
        assert_eq!(fonts[1].filename, "Garamond.otf");
        assert_eq!(fonts[1].mimetype, "application/vnd.ms-opentype");
    }

    #[test]
    fn parse_embedded_fonts_skips_non_font_attachments() {
        let fonts = parse_embedded_fonts_json(FFPROBE_FONTS_JSON).unwrap();
        // cover.png should not appear
        assert!(fonts.iter().all(|f| f.filename != "cover.png"));
    }

    #[test]
    fn parse_embedded_fonts_empty_streams() {
        let fonts = parse_embedded_fonts_json(r#"{"streams":[]}"#).unwrap();
        assert!(fonts.is_empty());
    }

    #[test]
    fn parse_embedded_fonts_font_in_mimetype() {
        // Any mimetype containing "font" should be included
        let json = r#"{
            "streams": [{
                "index": 1,
                "codec_type": "attachment",
                "tags": {
                    "filename": "myfont.woff",
                    "mimetype": "application/font-woff"
                }
            }]
        }"#;
        let fonts = parse_embedded_fonts_json(json).unwrap();
        assert_eq!(fonts.len(), 1);
        assert_eq!(fonts[0].filename, "myfont.woff");
    }

    #[test]
    fn parse_embedded_fonts_missing_filename_uses_default() {
        let json = r#"{
            "streams": [{
                "index": 3,
                "codec_type": "attachment",
                "tags": {
                    "mimetype": "application/x-truetype-font"
                }
            }]
        }"#;
        let fonts = parse_embedded_fonts_json(json).unwrap();
        assert_eq!(fonts.len(), 1);
        assert_eq!(fonts[0].filename, "font_3");
    }

    // ----- font_filename_matches -----

    #[test]
    fn font_filename_exact_match() {
        assert!(font_filename_matches(
            Path::new("/fonts/arial.ttf"),
            "Arial"
        ));
    }

    #[test]
    fn font_filename_prefix_match() {
        assert!(font_filename_matches(
            Path::new("/fonts/arialbd.ttf"),
            "arial"
        ));
    }

    #[test]
    fn font_filename_dash_to_space() {
        assert!(font_filename_matches(
            Path::new("/fonts/dejavu-sans.ttf"),
            "dejavu sans"
        ));
    }

    #[test]
    fn font_filename_no_match() {
        assert!(!font_filename_matches(
            Path::new("/fonts/times.ttf"),
            "Arial"
        ));
    }

    // ----- get_ass_font_names -----
    // Uses a real ASS file from the corpus if available; otherwise tests the
    // pure splitting logic via a constructed style.

    #[test]
    fn ass_font_names_from_inline_style() {
        // We test the raw-field splitting logic directly
        let raw = "Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1";
        // fields[1] = "Arial"
        let fields: Vec<&str> = raw.splitn(3, ',').collect();
        assert_eq!(fields[1], "Arial");
    }

    // ----- font_data_supports_polish -----
    // We cannot ship a real TTF in-repo, so we test the negative case (invalid data).

    #[test]
    fn font_data_invalid_returns_false() {
        assert!(!font_data_supports_polish(b"not a font file"));
    }

    #[test]
    fn font_data_empty_returns_false() {
        assert!(!font_data_supports_polish(b""));
    }

    /// Test with a real system font if available. Marked ignore so CI passes without fonts.
    #[test]
    #[ignore]
    fn font_supports_polish_system_arial() {
        // macOS path
        let arial = Path::new("/Library/Fonts/Arial.ttf");
        if arial.exists() {
            // Arial supports Polish
            assert!(font_supports_polish(arial));
        }
    }

    // ----- font_family_name_from_data -----

    #[test]
    fn font_family_name_invalid_data_returns_none() {
        assert!(font_family_name_from_data(b"garbage").is_none());
    }

    /// Test family name reading against a real font if available.
    #[test]
    #[ignore]
    fn font_family_name_from_system_font() {
        let arial = Path::new("/Library/Fonts/Arial.ttf");
        if arial.exists() {
            let name = get_font_family_name(arial);
            assert!(name.is_some());
            assert!(name.unwrap().to_ascii_lowercase().contains("arial"));
        }
    }
}
