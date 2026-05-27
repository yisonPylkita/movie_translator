//! Media container inspection and muxing (FFmpeg wrappers, MKV/MP4 handling).

pub mod extractor;
pub mod ffmpeg;
pub mod fonts;
pub mod operations;

// Re-export key public API at crate root.
pub use extractor::{
    categorize_tracks, convert_ffprobe_info, get_english_tracks, get_subtitle_extension_for_codec,
    handle_image_tracks, select_best_track, select_from_dialogue_tracks, select_from_signs_tracks,
    separate_by_codec, SubtitleExtractionError, SubtitleExtractor, SubtitleTrack, TrackInfo,
    TrackProperties,
};
pub use ffmpeg::{
    build_ffmpeg_mux_args, build_mkvmerge_args, get_ffmpeg, get_ffmpeg_paths, get_ffmpeg_version,
    get_ffprobe, get_mkvmerge, get_video_info, mimetype_for_font, mux_video_with_subtitles,
    parse_ffmpeg_version_string, parse_frame_rate, parse_video_encoding_from_info,
    parse_video_info, probe_video_encoding, resolve_mkvmerge_sub_track_id, FfprobeStream,
    VideoEncoding, VideoInfo, VideoMuxError,
};
pub use fonts::{
    check_embedded_fonts_support_polish, extract_font, find_system_font_for_polish,
    font_data_supports_polish, font_family_name_from_data, font_filename_matches,
    font_supports_polish, get_ass_font_names, get_embedded_fonts, get_font_family_name,
    get_system_font_dirs, iter_system_fonts, parse_embedded_fonts_json, EmbeddedFont,
};
pub use operations::{
    get_subtitle_tracks_from_info, validate_tracks, SubtitleTrackInfo, VideoOperationError,
    VideoOperations,
};
