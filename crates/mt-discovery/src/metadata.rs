//! Container metadata extraction via ffprobe.
//!
//! The ffprobe binary is resolved via `mt_core::exec::get_ffprobe()` so
//! discovery and mt-media share one binary-discovery path.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use mt_core::exec::get_ffprobe;
use mt_core::{MtError, Result};
use serde::Deserialize;
use serde_json::from_str;

/// Metadata extracted from a video container's format tags.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContainerMetadata {
    /// Title tag from the container (e.g. `format.tags.title`).
    pub title: Option<String>,
    /// Episode identifier from the container tags.
    pub episode: Option<String>,
}

// ── Internal ffprobe JSON shapes ──────────────────────────────────────────────

#[derive(Deserialize)]
struct FfprobeOutput {
    #[serde(default)]
    format: FfprobeFormat,
}

#[derive(Deserialize, Default)]
struct FfprobeFormat {
    #[serde(default)]
    tags: HashMap<String, String>,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Parse ffprobe JSON output into [`ContainerMetadata`].
///
/// Factored out from the subprocess call to allow unit testing.
pub(crate) fn parse_ffprobe_output(json: &str) -> ContainerMetadata {
    let probe = match from_str::<FfprobeOutput>(json) {
        Ok(p) => p,
        Err(_) => return ContainerMetadata::default(),
    };

    let tags = &probe.format.tags;

    // Common tag names across containers.
    let title = tags.get("title").or_else(|| tags.get("TITLE")).cloned();

    let episode = tags
        .get("episode_id")
        .or_else(|| tags.get("episode_sort"))
        .or_else(|| tags.get("track"))
        .cloned();

    ContainerMetadata { title, episode }
}

/// Extract title and episode metadata from a video container's tags.
///
/// Shells out to `ffprobe` (resolved via `mt_core::exec::get_ffprobe()`).
/// On any error returns a default [`ContainerMetadata`] with all fields `None`.
pub fn extract_container_metadata(video_path: &Path) -> ContainerMetadata {
    match run_ffprobe(video_path) {
        Ok(json) => parse_ffprobe_output(&json),
        Err(_) => ContainerMetadata::default(),
    }
}

fn run_ffprobe(video_path: &Path) -> Result<String> {
    // Resolve ffprobe through the shared resolver so discovery and mt-media
    // agree on the same binary (instead of shelling a bare `ffprobe` from PATH).
    let ffprobe = get_ffprobe()?;
    let output = Command::new(&ffprobe)
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            video_path.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(MtError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(MtError::Subprocess {
            cmd: "ffprobe".to_string(),
            code: output.status.code(),
            stderr,
        });
    }

    String::from_utf8(output.stdout)
        .map_err(|e| MtError::Parse(format!("ffprobe output not utf-8: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_FFPROBE_JSON: &str = r#"{
        "streams": [],
        "format": {
            "filename": "/fake/path.mkv",
            "tags": {
                "title": "One Piece Episode 101",
                "encoder": "libebml v1.4.2"
            }
        }
    }"#;

    const SAMPLE_WITH_EPISODE: &str = r#"{
        "streams": [],
        "format": {
            "tags": {
                "title": "One Piece",
                "episode_id": "101"
            }
        }
    }"#;

    const SAMPLE_NO_TAGS: &str = r#"{
        "streams": [],
        "format": {}
    }"#;

    const SAMPLE_EMPTY_TAGS: &str = r#"{
        "streams": [],
        "format": {
            "tags": {}
        }
    }"#;

    const SAMPLE_UPPERCASE_TITLE: &str = r#"{
        "streams": [],
        "format": {
            "tags": {
                "TITLE": "Uppercase Title"
            }
        }
    }"#;

    #[test]
    fn parse_title_from_format_tags() {
        let meta = parse_ffprobe_output(SAMPLE_FFPROBE_JSON);
        assert_eq!(meta.title.as_deref(), Some("One Piece Episode 101"));
        assert!(meta.episode.is_none());
    }

    #[test]
    fn parse_episode_id_tag() {
        let meta = parse_ffprobe_output(SAMPLE_WITH_EPISODE);
        assert_eq!(meta.title.as_deref(), Some("One Piece"));
        assert_eq!(meta.episode.as_deref(), Some("101"));
    }

    #[test]
    fn missing_tags_returns_none() {
        let meta = parse_ffprobe_output(SAMPLE_NO_TAGS);
        assert!(meta.title.is_none());
        assert!(meta.episode.is_none());
    }

    #[test]
    fn empty_tags_returns_none() {
        let meta = parse_ffprobe_output(SAMPLE_EMPTY_TAGS);
        assert!(meta.title.is_none());
        assert!(meta.episode.is_none());
    }

    #[test]
    fn uppercase_title_tag_is_found() {
        let meta = parse_ffprobe_output(SAMPLE_UPPERCASE_TITLE);
        assert_eq!(meta.title.as_deref(), Some("Uppercase Title"));
    }

    #[test]
    fn invalid_json_returns_default() {
        let meta = parse_ffprobe_output("not json at all");
        assert!(meta.title.is_none());
        assert!(meta.episode.is_none());
    }

    /// Test episode_sort fallback tag.
    #[test]
    fn episode_sort_fallback() {
        let json = r#"{"format": {"tags": {"episode_sort": "5"}}}"#;
        let meta = parse_ffprobe_output(json);
        assert_eq!(meta.episode.as_deref(), Some("5"));
    }

    /// Test track fallback tag.
    #[test]
    fn track_fallback() {
        let json = r#"{"format": {"tags": {"track": "7"}}}"#;
        let meta = parse_ffprobe_output(json);
        assert_eq!(meta.episode.as_deref(), Some("7"));
    }
}
