//! Subtitle track selection and extraction logic.
//!
//! Port of `movie_translator/subtitles/extractor.py` — mostly pure track-selection logic.

use std::path::Path;
use std::process::Command;

use mt_core::NON_DIALOGUE_STYLES;

use crate::ffmpeg::{get_ffmpeg, get_video_info, VideoMuxError};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum SubtitleExtractionError {
    #[error("Video file not found: {0}")]
    VideoNotFound(String),
    #[error("Subtitle extraction failed: {0}")]
    ExtractionFailed(String),
    #[error("FFmpeg error: {0}")]
    Ffmpeg(#[from] VideoMuxError),
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// Properties of a single subtitle track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackProperties {
    pub language: String,
    pub track_name: String,
    pub codec_id: String,
    pub forced_track: bool,
}

/// A subtitle track as returned by `get_track_info` / `convert_ffprobe_info`.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleTrack {
    /// ffprobe stream index (absolute, 0-based across all streams).
    pub id: u32,
    /// Codec name (e.g. `"ass"`, `"subrip"`, `"hdmv_pgs_subtitle"`).
    pub codec: String,
    /// Subtitle-relative index (0-based within subtitle streams only).
    pub subtitle_index: u32,
    pub properties: TrackProperties,
}

/// Collection of subtitle tracks from a video file.
#[derive(Debug, Clone, Default)]
pub struct TrackInfo {
    pub tracks: Vec<SubtitleTrack>,
}

// ---------------------------------------------------------------------------
// Codec sets
// ---------------------------------------------------------------------------

const TEXT_CODECS: &[&str] = &["ass", "ssa", "subrip", "srt", "webvtt", "mov_text"];
const IMAGE_CODECS: &[&str] = &["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"];

// ---------------------------------------------------------------------------
// SubtitleExtractor
// ---------------------------------------------------------------------------

/// Extracts and selects subtitle tracks from video files.
///
/// Port of `SubtitleExtractor` class.
pub struct SubtitleExtractor;

impl SubtitleExtractor {
    pub fn new() -> Self {
        SubtitleExtractor
    }

    /// Return subtitle track info for a video file.
    ///
    /// Port of `get_track_info`.
    pub fn get_track_info(
        &self,
        video_path: &Path,
    ) -> Result<TrackInfo, SubtitleExtractionError> {
        if !video_path.exists() {
            return Err(SubtitleExtractionError::VideoNotFound(
                video_path.to_string_lossy().to_string(),
            ));
        }
        let info = get_video_info(video_path)?;
        Ok(convert_ffprobe_info(&info))
    }

    /// Return `true` if the video contains a Polish (`pol`/`pl`) subtitle track.
    ///
    /// Port of `has_polish_subtitles`.
    pub fn has_polish_subtitles(
        &self,
        video_path: &Path,
    ) -> Result<bool, SubtitleExtractionError> {
        let track_info = self.get_track_info(video_path)?;
        Ok(track_info
            .tracks
            .iter()
            .any(|t| matches!(t.properties.language.as_str(), "pol" | "pl")))
    }

    /// Find the best English subtitle track in `track_info`.
    ///
    /// Port of `find_english_track`.
    pub fn find_english_track(&self, track_info: &TrackInfo) -> Option<SubtitleTrack> {
        let english_tracks = get_english_tracks(track_info);
        if english_tracks.is_empty() {
            return None;
        }
        select_best_track(&english_tracks)
    }

    /// Return the file extension appropriate for the track's codec.
    ///
    /// Port of `get_subtitle_extension`.
    pub fn get_subtitle_extension(&self, track: &SubtitleTrack) -> &'static str {
        get_subtitle_extension_for_codec(&track.codec)
    }

    /// Extract a subtitle track from `video_path` to `output_path`.
    ///
    /// Port of `extract_subtitle`.
    ///
    /// For PGS/image-based tracks the extraction still works (copies the
    /// binary stream); OCR is a separate concern — left as a TODO/hook.
    pub fn extract_subtitle(
        &self,
        video_path: &Path,
        _track_id: u32,
        output_path: &Path,
        subtitle_index: Option<u32>,
    ) -> Result<(), SubtitleExtractionError> {
        if !video_path.exists() {
            return Err(SubtitleExtractionError::VideoNotFound(
                video_path.to_string_lossy().to_string(),
            ));
        }

        let ffmpeg = get_ffmpeg()?;
        let sub_idx = subtitle_index.unwrap_or(0);
        let output = Command::new(ffmpeg)
            .args([
                "-y",
                "-i",
                &video_path.to_string_lossy(),
                "-map",
                &format!("0:s:{sub_idx}"),
                "-c:s",
                "copy",
                &output_path.to_string_lossy(),
            ])
            .output()
            .map_err(VideoMuxError::Io)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let error_lines: Vec<&str> = stderr
                .lines()
                .filter(|l| {
                    let ll = l.to_ascii_lowercase();
                    ll.contains("error") || ll.contains("invalid")
                })
                .collect();
            let msg = if !error_lines.is_empty() {
                error_lines.join("; ")
            } else {
                "Unknown ffmpeg error".to_string()
            };
            return Err(SubtitleExtractionError::ExtractionFailed(format!(
                "Failed to extract subtitle track {sub_idx}: {msg}"
            )));
        }
        Ok(())
    }
}

impl Default for SubtitleExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pure functions — all public for direct testing
// ---------------------------------------------------------------------------

/// Convert parsed `VideoInfo` to `TrackInfo`.
///
/// Port of `_convert_ffprobe_info` — pure function.
pub fn convert_ffprobe_info(info: &crate::ffmpeg::VideoInfo) -> TrackInfo {
    let mut tracks = Vec::new();
    let mut subtitle_index: u32 = 0;

    for stream in &info.streams {
        if stream.codec_type.as_deref() != Some("subtitle") {
            continue;
        }
        let language = stream.tags.get("language").cloned().unwrap_or_else(|| "und".to_string());
        let track_name = stream.tags.get("title").cloned().unwrap_or_default();
        let codec = stream.codec_name.clone().unwrap_or_default();
        let forced = stream.disposition.get("forced").copied().unwrap_or(0) == 1;

        tracks.push(SubtitleTrack {
            id: stream.index,
            codec: codec.clone(),
            subtitle_index,
            properties: TrackProperties {
                language,
                track_name,
                codec_id: codec,
                forced_track: forced,
            },
        });
        subtitle_index += 1;
    }

    TrackInfo { tracks }
}

/// Return English (or language-unset) tracks from `track_info`.
///
/// Port of `_get_english_tracks`.
pub fn get_english_tracks(track_info: &TrackInfo) -> Vec<SubtitleTrack> {
    track_info
        .tracks
        .iter()
        .filter(|t| {
            let lang = t.properties.language.to_ascii_lowercase();
            lang.is_empty() || matches!(lang.as_str(), "eng" | "en" | "und")
        })
        .cloned()
        .collect()
}

/// Select the best track from a list of English tracks.
///
/// Port of `_select_best_track`.
pub fn select_best_track(english_tracks: &[SubtitleTrack]) -> Option<SubtitleTrack> {
    let (dialogue_tracks, signs_tracks) = categorize_tracks(english_tracks);

    if !dialogue_tracks.is_empty() {
        if let Some(result) = select_from_dialogue_tracks(&dialogue_tracks) {
            return Some(result);
        }
    }

    if !signs_tracks.is_empty() {
        // Only signs tracks available — reject (no full dialogue track)
        return None;
    }

    None
}

/// Categorize tracks into dialogue vs signs/songs.
///
/// Port of `_categorize_tracks`.
pub fn categorize_tracks(tracks: &[SubtitleTrack]) -> (Vec<SubtitleTrack>, Vec<SubtitleTrack>) {
    let mut dialogue_tracks = Vec::new();
    let mut signs_tracks = Vec::new();

    for track in tracks {
        let track_name = track.properties.track_name.to_ascii_lowercase();

        // Empty or very generic names → dialogue
        if track_name.is_empty()
            || matches!(
                track_name.as_str(),
                "default" | "ass" | "subtitle" | "subtitles"
            )
        {
            dialogue_tracks.push(track.clone());
            continue;
        }

        // Only mark as signs if the name explicitly indicates it
        let is_signs = NON_DIALOGUE_STYLES.iter().any(|kw| name_has_keyword(&track_name, kw));

        if is_signs {
            signs_tracks.push(track.clone());
        } else {
            dialogue_tracks.push(track.clone());
        }
    }

    (dialogue_tracks, signs_tracks)
}

/// Check if `track_name` contains a whole-word match for `keyword`
/// (or keyword + optional 's').
///
/// Ports the `re.search(rf'\b{re.escape(keyword)}s?\b', track_name)` check.
fn name_has_keyword(track_name: &str, keyword: &str) -> bool {
    // Find all occurrences of keyword (or keyword+'s') with word boundaries.
    let mut search = track_name;
    let kw_lower = keyword.to_ascii_lowercase();
    while !search.is_empty() {
        if let Some(pos) = search.find(&kw_lower[..]) {
            let end_simple = pos + kw_lower.len();
            // Check word-boundary before: pos == 0 or prev char is non-alphanumeric
            let before_ok = pos == 0
                || search[..pos]
                    .chars()
                    .last()
                    .map(|c| !c.is_alphanumeric())
                    .unwrap_or(true);
            // Allow optional 's' after keyword
            let end_with_s = end_simple + 1;
            let after_ok = {
                let end = if search[end_simple..].starts_with('s') {
                    end_with_s
                } else {
                    end_simple
                };
                end >= search.len()
                    || search[end..].chars().next().map(|c| !c.is_alphanumeric()).unwrap_or(true)
            };
            if before_ok && after_ok {
                return true;
            }
            search = &search[pos + 1..];
        } else {
            break;
        }
    }
    false
}

/// Select the best track from already-categorized dialogue tracks.
///
/// Port of `_select_from_dialogue_tracks`.
pub fn select_from_dialogue_tracks(dialogue_tracks: &[SubtitleTrack]) -> Option<SubtitleTrack> {
    let (text_tracks, image_tracks) = separate_by_codec(dialogue_tracks);

    if !text_tracks.is_empty() {
        return Some(text_tracks[0].clone());
    }
    if !image_tracks.is_empty() {
        // TODO: hook for OCR processing of image-based PGS/DVD tracks
        return handle_image_tracks(&image_tracks);
    }
    // Fallback: return first dialogue track regardless of codec
    dialogue_tracks.first().cloned()
}

/// Select from signs/songs-only tracks (used when no dialogue tracks exist).
///
/// Port of `_select_from_signs_tracks`.
pub fn select_from_signs_tracks(
    signs_tracks: &[SubtitleTrack],
    english_tracks: &[SubtitleTrack],
) -> Option<SubtitleTrack> {
    let (text_signs, image_signs) = separate_by_codec(signs_tracks);

    if !text_signs.is_empty() {
        return Some(text_signs[0].clone());
    }
    if !image_signs.is_empty() {
        return None;
    }

    let non_forced: Vec<_> = english_tracks
        .iter()
        .filter(|t| !t.properties.forced_track)
        .collect();
    if !non_forced.is_empty() {
        return Some(non_forced[0].clone());
    }

    english_tracks.first().cloned()
}

/// Partition tracks into text-based and image-based by codec.
///
/// Port of `_separate_by_codec`.
pub fn separate_by_codec(tracks: &[SubtitleTrack]) -> (Vec<SubtitleTrack>, Vec<SubtitleTrack>) {
    let mut text_tracks = Vec::new();
    let mut image_tracks = Vec::new();

    for track in tracks {
        let codec = track.codec.to_ascii_lowercase();
        if TEXT_CODECS
            .iter()
            .any(|&c| codec == c || codec.starts_with(c))
        {
            text_tracks.push(track.clone());
        } else if IMAGE_CODECS
            .iter()
            .any(|&c| codec == c || codec.starts_with(c))
        {
            image_tracks.push(track.clone());
        } else {
            // Unknown codec: treat as text
            text_tracks.push(track.clone());
        }
    }

    (text_tracks, image_tracks)
}

/// Handle image-based (PGS/DVD) dialogue tracks.
///
/// Port of `_handle_image_tracks`.
/// TODO: Wire in OCR pipeline when implemented.
pub fn handle_image_tracks(image_tracks: &[SubtitleTrack]) -> Option<SubtitleTrack> {
    image_tracks.first().cloned()
}

/// Return the file extension for a subtitle codec name.
///
/// Port of `get_subtitle_extension`.
pub fn get_subtitle_extension_for_codec(codec: &str) -> &'static str {
    match codec.to_ascii_lowercase().as_str() {
        "ass" => ".ass",
        "ssa" => ".ssa",
        "subrip" | "srt" => ".srt",
        "webvtt" => ".vtt",
        "mov_text" => ".srt",
        _ => ".srt",
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffmpeg::parse_video_info;

    fn make_track(id: u32, sub_idx: u32, codec: &str, lang: &str, name: &str) -> SubtitleTrack {
        SubtitleTrack {
            id,
            codec: codec.to_string(),
            subtitle_index: sub_idx,
            properties: TrackProperties {
                language: lang.to_string(),
                track_name: name.to_string(),
                codec_id: codec.to_string(),
                forced_track: false,
            },
        }
    }

    // ----- convert_ffprobe_info -----

    const FFPROBE_WITH_SUBS: &str = r#"{
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "tags": {}, "disposition": {}},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "tags": {"language": "jpn"}, "disposition": {}},
            {"index": 2, "codec_type": "subtitle", "codec_name": "ass",
             "tags": {"language": "eng", "title": "English Full Dialogue"},
             "disposition": {"default": 1, "forced": 0}},
            {"index": 3, "codec_type": "subtitle", "codec_name": "ass",
             "tags": {"language": "eng", "title": "English Signs/Songs"},
             "disposition": {"default": 0, "forced": 0}},
            {"index": 4, "codec_type": "subtitle", "codec_name": "subrip",
             "tags": {"language": "pol", "title": "Polish"},
             "disposition": {"default": 0, "forced": 0}}
        ]
    }"#;

    #[test]
    fn convert_ffprobe_info_extracts_subtitle_streams() {
        let info = parse_video_info(FFPROBE_WITH_SUBS).unwrap();
        let track_info = convert_ffprobe_info(&info);
        assert_eq!(track_info.tracks.len(), 3);
    }

    #[test]
    fn convert_ffprobe_info_assigns_subtitle_index() {
        let info = parse_video_info(FFPROBE_WITH_SUBS).unwrap();
        let track_info = convert_ffprobe_info(&info);
        assert_eq!(track_info.tracks[0].subtitle_index, 0);
        assert_eq!(track_info.tracks[1].subtitle_index, 1);
        assert_eq!(track_info.tracks[2].subtitle_index, 2);
    }

    #[test]
    fn convert_ffprobe_info_language_and_name() {
        let info = parse_video_info(FFPROBE_WITH_SUBS).unwrap();
        let track_info = convert_ffprobe_info(&info);
        assert_eq!(track_info.tracks[0].properties.language, "eng");
        assert_eq!(track_info.tracks[0].properties.track_name, "English Full Dialogue");
        assert_eq!(track_info.tracks[2].properties.language, "pol");
    }

    #[test]
    fn convert_ffprobe_info_forced_flag() {
        let json = r#"{
            "streams": [{
                "index": 0,
                "codec_type": "subtitle",
                "codec_name": "ass",
                "tags": {"language": "eng", "title": ""},
                "disposition": {"forced": 1}
            }]
        }"#;
        let info = parse_video_info(json).unwrap();
        let track_info = convert_ffprobe_info(&info);
        assert!(track_info.tracks[0].properties.forced_track);
    }

    // ----- get_english_tracks -----

    #[test]
    fn get_english_tracks_includes_eng() {
        let info = parse_video_info(FFPROBE_WITH_SUBS).unwrap();
        let track_info = convert_ffprobe_info(&info);
        let eng = get_english_tracks(&track_info);
        assert_eq!(eng.len(), 2); // two eng tracks; pol excluded
    }

    #[test]
    fn get_english_tracks_includes_und() {
        let track_info = TrackInfo {
            tracks: vec![
                make_track(0, 0, "ass", "und", ""),
                make_track(1, 1, "ass", "jpn", ""),
            ],
        };
        let eng = get_english_tracks(&track_info);
        assert_eq!(eng.len(), 1);
        assert_eq!(eng[0].properties.language, "und");
    }

    // ----- categorize_tracks -----

    #[test]
    fn categorize_empty_name_is_dialogue() {
        let tracks = vec![make_track(0, 0, "ass", "eng", "")];
        let (dialogue, signs) = categorize_tracks(&tracks);
        assert_eq!(dialogue.len(), 1);
        assert_eq!(signs.len(), 0);
    }

    #[test]
    fn categorize_signs_songs_name() {
        let tracks = vec![make_track(0, 0, "ass", "eng", "English Signs/Songs")];
        let (dialogue, signs) = categorize_tracks(&tracks);
        assert_eq!(dialogue.len(), 0);
        assert_eq!(signs.len(), 1);
    }

    #[test]
    fn categorize_dialogue_name_even_if_not_empty() {
        let tracks = vec![make_track(0, 0, "ass", "eng", "English Full Dialogue")];
        let (dialogue, signs) = categorize_tracks(&tracks);
        assert_eq!(dialogue.len(), 1);
        assert_eq!(signs.len(), 0);
    }

    #[test]
    fn categorize_op_ed_are_signs() {
        let tracks = vec![
            make_track(0, 0, "ass", "eng", "OP/ED"),
            make_track(1, 1, "ass", "eng", "Songs"),
        ];
        let (dialogue, signs) = categorize_tracks(&tracks);
        assert_eq!(dialogue.len(), 0);
        assert_eq!(signs.len(), 2);
    }

    #[test]
    fn categorize_generic_names_are_dialogue() {
        let tracks = vec![
            make_track(0, 0, "ass", "eng", "default"),
            make_track(1, 1, "ass", "eng", "subtitles"),
        ];
        let (dialogue, signs) = categorize_tracks(&tracks);
        assert_eq!(dialogue.len(), 2);
        assert_eq!(signs.len(), 0);
    }

    // ----- separate_by_codec -----

    #[test]
    fn separate_text_and_image_codecs() {
        let tracks = vec![
            make_track(0, 0, "ass", "eng", ""),
            make_track(1, 1, "hdmv_pgs_subtitle", "eng", ""),
            make_track(2, 2, "subrip", "eng", ""),
        ];
        let (text, image) = separate_by_codec(&tracks);
        assert_eq!(text.len(), 2);
        assert_eq!(image.len(), 1);
    }

    #[test]
    fn separate_unknown_codec_goes_to_text() {
        let tracks = vec![make_track(0, 0, "webm_vtt", "eng", "")];
        let (text, _image) = separate_by_codec(&tracks);
        assert_eq!(text.len(), 1);
    }

    // ----- get_subtitle_extension_for_codec -----

    #[test]
    fn extension_ass() {
        assert_eq!(get_subtitle_extension_for_codec("ass"), ".ass");
    }

    #[test]
    fn extension_ssa() {
        assert_eq!(get_subtitle_extension_for_codec("ssa"), ".ssa");
    }

    #[test]
    fn extension_subrip() {
        assert_eq!(get_subtitle_extension_for_codec("subrip"), ".srt");
    }

    #[test]
    fn extension_webvtt() {
        assert_eq!(get_subtitle_extension_for_codec("webvtt"), ".vtt");
    }

    #[test]
    fn extension_mov_text() {
        assert_eq!(get_subtitle_extension_for_codec("mov_text"), ".srt");
    }

    #[test]
    fn extension_unknown() {
        assert_eq!(get_subtitle_extension_for_codec("unknown_codec"), ".srt");
    }

    // ----- Track selection (port of test_track_selection.py) -----

    fn make_track_info(tracks: Vec<SubtitleTrack>) -> TrackInfo {
        TrackInfo { tracks }
    }

    #[test]
    fn single_signs_track_rejected() {
        // Port of test_single_signs_track_rejected
        let extractor = SubtitleExtractor::new();
        let track_info = make_track_info(vec![
            make_track(0, 0, "ass", "eng", "English Signs/Songs"),
        ]);
        let result = extractor.find_english_track(&track_info);
        assert!(result.is_none());
    }

    #[test]
    fn prefers_dialogue_over_signs_when_both_present() {
        // Port of test_prefers_dialogue_over_signs_when_both_present
        let extractor = SubtitleExtractor::new();
        let track_info = make_track_info(vec![
            make_track(0, 0, "ass", "eng", "English Signs/Songs"),
            make_track(1, 1, "ass", "eng", "English Full Dialogue"),
        ]);
        let result = extractor.find_english_track(&track_info);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.id, 1);
        assert!(r.properties.track_name.contains("Dialogue"));
    }

    #[test]
    fn prefers_dialogue_even_when_signs_listed_first() {
        // Port of test_prefers_dialogue_even_when_signs_listed_first
        let extractor = SubtitleExtractor::new();
        let track_info = make_track_info(vec![
            make_track(0, 0, "ass", "eng", "Signs and Songs"),
            make_track(1, 1, "ass", "eng", "English"),
        ]);
        let result = extractor.find_english_track(&track_info);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.id, 1);
        assert_eq!(r.properties.track_name, "English");
    }

    #[test]
    fn track_without_name_treated_as_dialogue() {
        // Port of test_track_without_name_treated_as_dialogue
        let extractor = SubtitleExtractor::new();
        let track_info = make_track_info(vec![
            make_track(0, 0, "ass", "eng", "Signs"),
            make_track(1, 1, "ass", "eng", ""),
        ]);
        let result = extractor.find_english_track(&track_info);
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, 1);
    }

    #[test]
    fn multiple_signs_tracks_all_rejected() {
        // Port of test_multiple_signs_tracks_all_rejected
        let extractor = SubtitleExtractor::new();
        let track_info = make_track_info(vec![
            make_track(0, 0, "ass", "eng", "English Signs"),
            make_track(1, 1, "ass", "eng", "English Songs"),
            make_track(2, 2, "ass", "eng", "OP/ED"),
        ]);
        let result = extractor.find_english_track(&track_info);
        assert!(result.is_none());
    }

    // ----- Additional categorization tests -----

    #[test]
    fn categorize_insert_song_is_signs() {
        let tracks = vec![make_track(0, 0, "ass", "eng", "Insert Song")];
        let (dialogue, signs) = categorize_tracks(&tracks);
        assert_eq!(dialogue.len(), 0, "insert song should be signs");
        assert_eq!(signs.len(), 1);
    }

    #[test]
    fn categorize_title_is_signs() {
        let tracks = vec![make_track(0, 0, "ass", "eng", "title card")];
        let (dialogue, signs) = categorize_tracks(&tracks);
        assert_eq!(dialogue.len(), 0, "title should be signs");
        assert_eq!(signs.len(), 1);
    }

    #[test]
    fn has_polish_subtitles_checks_language_code() {
        // Test via convert_ffprobe_info on fixture with pol track
        let info = parse_video_info(FFPROBE_WITH_SUBS).unwrap();
        let track_info = convert_ffprobe_info(&info);
        let has_pol = track_info
            .tracks
            .iter()
            .any(|t| matches!(t.properties.language.as_str(), "pol" | "pl"));
        assert!(has_pol);
    }

    // ----- name_has_keyword edge cases -----

    #[test]
    fn keyword_with_plural_s_matches() {
        assert!(name_has_keyword("english signs", "sign"));
        assert!(name_has_keyword("english songs", "song"));
    }

    #[test]
    fn keyword_within_word_does_not_match() {
        // "signing" has "sign" but it's not a whole word boundary followed by 's?' end
        assert!(!name_has_keyword("signing", "sign"));
    }

    #[test]
    fn keyword_standalone_matches() {
        assert!(name_has_keyword("sign", "sign"));
        assert!(name_has_keyword("signs", "sign"));
    }
}
