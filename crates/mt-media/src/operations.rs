//! High-level video operations (mux, verify).
//!
//! Port of `movie_translator/video/operations.py`.

use std::path::{Path, PathBuf};

use mt_core::SubtitleFile;

use crate::ffmpeg::{get_video_info, mux_video_with_subtitles, VideoMuxError};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum VideoOperationError {
    #[error("Output video not found: {0}")]
    OutputNotFound(String),
    #[error("Expected {expected} subtitle tracks, found {actual}")]
    TrackCountMismatch { expected: usize, actual: usize },
    #[error("Track {index}: expected language \"{expected}\", found \"{actual}\"")]
    TrackLanguageMismatch {
        index: usize,
        expected: String,
        actual: String,
    },
    #[error("Mux error: {0}")]
    Mux(#[from] VideoMuxError),
}

// ---------------------------------------------------------------------------
// Subtitle track descriptor (returned by _get_subtitle_tracks)
// ---------------------------------------------------------------------------

/// Minimal subtitle track info used for verification.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleTrackInfo {
    pub index: u32,
    pub language: String,
    pub title: String,
}

// ---------------------------------------------------------------------------
// VideoOperations
// ---------------------------------------------------------------------------

/// High-level video operations — mux and verify.
///
/// Port of `VideoOperations` class.
pub struct VideoOperations;

impl VideoOperations {
    pub fn new() -> Self {
        VideoOperations
    }

    /// Create a clean video by muxing `original_video` with `subtitle_files`.
    ///
    /// Port of `create_clean_video`.
    pub fn create_clean_video(
        &self,
        original_video: &Path,
        subtitle_files: &[SubtitleFile],
        output_video: &Path,
        font_attachments: Option<&[PathBuf]>,
        original_sub_index: Option<usize>,
        original_sub_title: Option<&str>,
    ) -> Result<(), VideoOperationError> {
        mux_video_with_subtitles(
            original_video,
            subtitle_files,
            output_video,
            font_attachments,
            original_sub_index,
            original_sub_title,
        )?;
        Ok(())
    }

    /// Verify that `output_video` exists and optionally validate subtitle tracks.
    ///
    /// Port of `verify_result`.
    pub fn verify_result(
        &self,
        output_video: &Path,
        expected_tracks: Option<&[SubtitleFile]>,
    ) -> Result<Vec<SubtitleTrackInfo>, VideoOperationError> {
        if !output_video.exists() {
            return Err(VideoOperationError::OutputNotFound(
                output_video.to_string_lossy().to_string(),
            ));
        }

        let info = get_video_info(output_video)?;
        let subtitle_tracks = get_subtitle_tracks_from_info(&info);

        if let Some(expected) = expected_tracks {
            validate_tracks(&subtitle_tracks, expected)?;
        }

        Ok(subtitle_tracks)
    }
}

impl Default for VideoOperations {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions
// ---------------------------------------------------------------------------

/// Extract subtitle track descriptors from parsed video info.
///
/// Port of `_get_subtitle_tracks`.
pub fn get_subtitle_tracks_from_info(info: &crate::ffmpeg::VideoInfo) -> Vec<SubtitleTrackInfo> {
    info.streams
        .iter()
        .filter(|s| s.codec_type.as_deref() == Some("subtitle"))
        .map(|s| SubtitleTrackInfo {
            index: s.index,
            language: s
                .tags
                .get("language")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
            title: s
                .tags
                .get("title")
                .cloned()
                .unwrap_or_else(|| "unnamed".to_string()),
        })
        .collect()
}

/// Validate that `actual` tracks match `expected` subtitle files.
///
/// Port of `_validate_tracks`.
/// Returns `Ok(())` on success or a `VideoOperationError` describing the first mismatch.
pub fn validate_tracks(
    actual: &[SubtitleTrackInfo],
    expected: &[SubtitleFile],
) -> Result<(), VideoOperationError> {
    if actual.len() != expected.len() {
        return Err(VideoOperationError::TrackCountMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }

    for (i, (track, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        if track.language != exp.language {
            return Err(VideoOperationError::TrackLanguageMismatch {
                index: i + 1,
                expected: exp.language.clone(),
                actual: track.language.clone(),
            });
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffmpeg::parse_video_info;

    fn make_sub(lang: &str, title: &str, is_default: bool) -> SubtitleFile {
        SubtitleFile {
            path: PathBuf::from(format!("/tmp/{title}.ass")),
            language: lang.to_string(),
            title: title.to_string(),
            is_default,
        }
    }

    fn make_track_info(lang: &str, title: &str) -> SubtitleTrackInfo {
        SubtitleTrackInfo {
            index: 0,
            language: lang.to_string(),
            title: title.to_string(),
        }
    }

    // ----- get_subtitle_tracks_from_info -----

    const VIDEO_INFO_JSON: &str = r#"{
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "tags": {}, "disposition": {}},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "tags": {"language": "jpn"}, "disposition": {}},
            {"index": 2, "codec_type": "subtitle", "codec_name": "ass",
             "tags": {"language": "eng", "title": "English"},
             "disposition": {}},
            {"index": 3, "codec_type": "subtitle", "codec_name": "subrip",
             "tags": {"language": "pol", "title": "Polish"},
             "disposition": {}}
        ]
    }"#;

    #[test]
    fn get_subtitle_tracks_extracts_correct_streams() {
        let info = parse_video_info(VIDEO_INFO_JSON).unwrap();
        let tracks = get_subtitle_tracks_from_info(&info);
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].language, "eng");
        assert_eq!(tracks[0].title, "English");
        assert_eq!(tracks[1].language, "pol");
        assert_eq!(tracks[1].title, "Polish");
    }

    #[test]
    fn get_subtitle_tracks_missing_tags_use_defaults() {
        let json = r#"{"streams": [
            {"index": 0, "codec_type": "subtitle", "codec_name": "ass", "tags": {}, "disposition": {}}
        ]}"#;
        let info = parse_video_info(json).unwrap();
        let tracks = get_subtitle_tracks_from_info(&info);
        assert_eq!(tracks[0].language, "unknown");
        assert_eq!(tracks[0].title, "unnamed");
    }

    // ----- validate_tracks -----

    #[test]
    fn validate_tracks_success() {
        let actual = vec![
            make_track_info("eng", "English"),
            make_track_info("pol", "Polish"),
        ];
        let expected = vec![
            make_sub("eng", "English", false),
            make_sub("pol", "Polish", true),
        ];
        assert!(validate_tracks(&actual, &expected).is_ok());
    }

    #[test]
    fn validate_tracks_count_mismatch() {
        let actual = vec![make_track_info("eng", "English")];
        let expected = vec![
            make_sub("eng", "English", false),
            make_sub("pol", "Polish", true),
        ];
        let result = validate_tracks(&actual, &expected);
        assert!(matches!(
            result,
            Err(VideoOperationError::TrackCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn validate_tracks_language_mismatch() {
        let actual = vec![make_track_info("jpn", "Japanese")];
        let expected = vec![make_sub("pol", "Polish", true)];
        let result = validate_tracks(&actual, &expected);
        assert!(matches!(
            result,
            Err(VideoOperationError::TrackLanguageMismatch {
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn validate_tracks_empty_both() {
        let result = validate_tracks(&[], &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_tracks_language_only_checked_not_title() {
        // title can differ — only language is validated (matches Python)
        let actual = vec![make_track_info("pol", "Different Title")];
        let expected = vec![make_sub("pol", "Polish", true)];
        assert!(validate_tracks(&actual, &expected).is_ok());
    }
}
