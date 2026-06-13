//! FFmpeg/ffprobe/mkvmerge binary resolution and media operations.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde::Deserialize;
use thiserror::Error;

use mt_core::SubtitleFile;

mod binaries {
    //! Binary resolution, delegating to the shared `mt_core::exec` resolver so
    //! `mt-media` and `mt-discovery` find the same `ffmpeg`/`ffprobe`.
    use super::*;

    /// Resolve `ffmpeg` and `ffprobe`, returning a [`VideoMuxError`] on failure.
    pub(super) fn ffmpeg() -> Result<PathBuf, VideoMuxError> {
        mt_core::exec::get_ffmpeg().map_err(|e| VideoMuxError::FfmpegNotFound(e.to_string()))
    }

    pub(super) fn ffprobe() -> Result<PathBuf, VideoMuxError> {
        mt_core::exec::get_ffprobe().map_err(|e| VideoMuxError::FfmpegNotFound(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum VideoMuxError {
    #[error("FFmpeg not found: {0}")]
    FfmpegNotFound(String),
    #[error("ffprobe failed: {0}")]
    FfprobeFailed(String),
    #[error("No video stream found in {0}")]
    NoVideoStream(String),
    #[error("mkvmerge identify failed: {0}")]
    MkvmergeIdentifyFailed(String),
    #[error("Subtitle index {index} out of range (video has {count} subtitle tracks)")]
    SubtitleIndexOutOfRange { index: usize, count: usize },
    #[error("Video file not found: {0}")]
    VideoNotFound(String),
    #[error("Subtitle file not found: {0}")]
    SubtitleNotFound(String),
    #[error("Mux failed: {0}")]
    MuxFailed(String),
    #[error("JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Cached binary resolution
// ---------------------------------------------------------------------------

static MKVMERGE_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Resolve `ffmpeg` and `ffprobe`.
///
/// Delegates to the shared [`mt_core::exec`] resolver (which caches successes,
/// retries failures, and validates the resolved path is a real file).
pub fn get_ffmpeg_paths() -> Result<(PathBuf, PathBuf), VideoMuxError> {
    Ok((binaries::ffmpeg()?, binaries::ffprobe()?))
}

/// Return the path to the `ffmpeg` binary.
pub fn get_ffmpeg() -> Result<PathBuf, VideoMuxError> {
    binaries::ffmpeg()
}

/// Return the path to the `ffprobe` binary.
pub fn get_ffprobe() -> Result<PathBuf, VideoMuxError> {
    binaries::ffprobe()
}

/// Find `mkvmerge` binary.  Returns `None` if unavailable.
///
/// Checks `PATH` first (via the shared resolver), then the Homebrew location
/// `/opt/homebrew/bin/mkvmerge`. A successful resolution is cached; mkvmerge is
/// optional, so a miss is represented as `None` (cached) rather than an error.
pub fn get_mkvmerge() -> Option<&'static PathBuf> {
    MKVMERGE_PATH
        .get_or_init(|| {
            if let Ok(p) = mt_core::exec::find_binary("mkvmerge") {
                return Some(p);
            }
            let homebrew = PathBuf::from("/opt/homebrew/bin/mkvmerge");
            if homebrew.is_file() {
                return Some(homebrew);
            }
            None
        })
        .as_ref()
}

// ---------------------------------------------------------------------------
// Video info / probing
// ---------------------------------------------------------------------------

/// Raw ffprobe JSON output structure.
#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    #[serde(default)]
    streams: Vec<FfprobeStream>,
    #[serde(default)]
    format: Option<FfprobeFormat>,
}

#[derive(Debug, Deserialize, Default)]
struct FfprobeFormat {
    #[serde(default)]
    bit_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FfprobeStream {
    pub index: u32,
    pub codec_type: Option<String>,
    pub codec_name: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub r_frame_rate: Option<String>,
    pub pix_fmt: Option<String>,
    pub profile: Option<String>,
    pub bit_rate: Option<String>,
    #[serde(default)]
    pub duration: Option<String>,
    #[serde(default)]
    pub tags: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub disposition: std::collections::HashMap<String, i64>,
}

/// High-level info returned by `get_video_info`.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub streams: Vec<FfprobeStream>,
    pub format_bit_rate: Option<String>,
    /// `format.duration` from ffprobe, in seconds (string form).
    pub format_duration: Option<String>,
}

/// Video encoding parameters extracted by `probe_video_encoding`.
#[derive(Debug, Clone)]
pub struct VideoEncoding {
    pub codec_name: String,
    pub profile: String,
    pub width: u32,
    pub height: u32,
    pub bit_rate: String,
    pub pix_fmt: String,
    pub fps: f64,
}

/// Run ffprobe on `video_path` and return parsed `VideoInfo`.
///
/// Note: this does not cache; callers may cache at a higher level.
pub fn get_video_info(video_path: &Path) -> Result<VideoInfo, VideoMuxError> {
    let ffprobe = get_ffprobe()?;
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            &video_path.to_string_lossy(),
        ])
        .output()?;

    if !output.status.success() {
        return Err(VideoMuxError::FfprobeFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    let json = String::from_utf8_lossy(&output.stdout);
    parse_video_info(&json)
}

/// Pure function: parse ffprobe JSON into `VideoInfo`.
///
/// Factored out for unit testing against captured fixtures.
pub fn parse_video_info(json: &str) -> Result<VideoInfo, VideoMuxError> {
    let raw: FfprobeOutput = serde_json::from_str(json)?;
    let format = raw.format;
    Ok(VideoInfo {
        streams: raw.streams,
        format_bit_rate: format.as_ref().and_then(|f| f.bit_rate.clone()),
        format_duration: format.and_then(|f| f.duration),
    })
}

/// Extract video encoding parameters for re-encoding decisions.
pub fn probe_video_encoding(video_path: &Path) -> Result<VideoEncoding, VideoMuxError> {
    let info = get_video_info(video_path)?;
    parse_video_encoding_from_info(&info, video_path)
}

/// Pure function: extract encoding params from already-parsed `VideoInfo`.
pub fn parse_video_encoding_from_info(
    info: &VideoInfo,
    video_path: &Path,
) -> Result<VideoEncoding, VideoMuxError> {
    let video_stream = info
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| VideoMuxError::NoVideoStream(video_path.to_string_lossy().to_string()))?;

    // Parse r_frame_rate: "24/1" or "24000/1001"
    let fps =
        parse_frame_rate(video_stream.r_frame_rate.as_deref().unwrap_or("24/1")).unwrap_or(24.0);

    let bit_rate = video_stream
        .bit_rate
        .clone()
        .or_else(|| info.format_bit_rate.clone())
        .unwrap_or_else(|| "5000000".to_string());

    Ok(VideoEncoding {
        codec_name: video_stream
            .codec_name
            .clone()
            .unwrap_or_else(|| "h264".to_string()),
        profile: video_stream.profile.clone().unwrap_or_default(),
        width: video_stream.width.unwrap_or(1920),
        height: video_stream.height.unwrap_or(1080),
        bit_rate,
        pix_fmt: video_stream
            .pix_fmt
            .clone()
            .unwrap_or_else(|| "yuv420p".to_string()),
        fps,
    })
}

/// Parse an ffprobe `r_frame_rate` string like `"24/1"` or `"24000/1001"`.
pub fn parse_frame_rate(r_frame_rate: &str) -> Option<f64> {
    let mut parts = r_frame_rate.splitn(2, '/');
    let num: f64 = parts.next()?.parse().ok()?;
    let den: f64 = parts.next()?.parse().ok()?;
    if den == 0.0 {
        return None;
    }
    Some(num / den)
}

/// Return the first line of `ffmpeg -version`.
pub fn get_ffmpeg_version() -> Result<String, VideoMuxError> {
    let ffmpeg = get_ffmpeg()?;
    let output = Command::new(ffmpeg).arg("-version").output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next().unwrap_or("").to_string();
    Ok(first_line)
}

/// Parse the ffmpeg version line into just the version token.
///
/// Input:  `"ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers"`
/// Output: `"7.1"`
pub fn parse_ffmpeg_version_string(first_line: &str) -> &str {
    // Format: "ffmpeg version <ver> Copyright..."
    let mut tokens = first_line.split_whitespace();
    // skip "ffmpeg", "version"
    tokens.next();
    tokens.next();
    tokens.next().unwrap_or("")
}

// ---------------------------------------------------------------------------
// Font MIME type
// ---------------------------------------------------------------------------

/// Map a font file extension to its MIME type.
pub fn mimetype_for_font(font_path: &Path) -> &'static str {
    match font_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("otf") => "application/vnd.ms-opentype",
        _ => "application/x-truetype-font",
    }
}

// ---------------------------------------------------------------------------
// mux_video_with_subtitles — top-level entry point
// ---------------------------------------------------------------------------

/// Mux `video_path` with `subtitle_files` into `output_path`.
///
/// Uses mkvmerge when available and output is MKV; falls back to ffmpeg.
pub fn mux_video_with_subtitles(
    video_path: &Path,
    subtitle_files: &[SubtitleFile],
    output_path: &Path,
    font_attachments: Option<&[PathBuf]>,
    original_sub_index: Option<usize>,
    original_sub_title: Option<&str>,
) -> Result<(), VideoMuxError> {
    if !video_path.exists() {
        return Err(VideoMuxError::VideoNotFound(
            video_path.to_string_lossy().to_string(),
        ));
    }
    for sub in subtitle_files {
        if !sub.path.exists() {
            return Err(VideoMuxError::SubtitleNotFound(
                sub.path.to_string_lossy().to_string(),
            ));
        }
    }

    let is_mkv = matches!(
        output_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("mkv") | Some("mka") | Some("mks")
    );

    let mkvmerge = if is_mkv { get_mkvmerge() } else { None };

    if let Some(mkv) = mkvmerge {
        let args = build_mkvmerge_args(
            mkv,
            video_path,
            subtitle_files,
            output_path,
            font_attachments,
            original_sub_index,
            original_sub_title,
        )?;
        run_mkvmerge(mkv, &args)
    } else {
        let ffmpeg = get_ffmpeg()?;
        let args = build_ffmpeg_mux_args(
            video_path,
            subtitle_files,
            output_path,
            font_attachments,
            original_sub_index,
            original_sub_title,
        );
        run_ffmpeg_mux(&ffmpeg, &args)
    }
}

// ---------------------------------------------------------------------------
// mkvmerge argv builder (pure, testable)
// ---------------------------------------------------------------------------

/// Resolve the absolute mkvmerge track ID for a subtitle-relative index.
pub fn resolve_mkvmerge_sub_track_id(
    mkvmerge: &Path,
    video_path: &Path,
    subtitle_index: usize,
) -> Result<u64, VideoMuxError> {
    let output = Command::new(mkvmerge)
        .args(["-J", &video_path.to_string_lossy()])
        .output()?;
    if !output.status.success() {
        return Err(VideoMuxError::MkvmergeIdentifyFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    let json = String::from_utf8_lossy(&output.stdout);
    parse_mkvmerge_sub_track_id(&json, subtitle_index)
}

/// Pure helper: select the subtitle track id from mkvmerge `-J` JSON.
///
/// Factored out for unit testing. A missing/non-integer `id` returns an error
/// rather than silently defaulting to track 0 (which would mux the wrong track).
pub fn parse_mkvmerge_sub_track_id(
    json: &str,
    subtitle_index: usize,
) -> Result<u64, VideoMuxError> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let tracks = value["tracks"].as_array().cloned().unwrap_or_default();
    let sub_tracks: Vec<&serde_json::Value> = tracks
        .iter()
        .filter(|t| t["type"].as_str() == Some("subtitles"))
        .collect();
    if subtitle_index >= sub_tracks.len() {
        return Err(VideoMuxError::SubtitleIndexOutOfRange {
            index: subtitle_index,
            count: sub_tracks.len(),
        });
    }
    sub_tracks[subtitle_index]["id"].as_u64().ok_or_else(|| {
        VideoMuxError::MkvmergeIdentifyFailed(format!(
            "mkvmerge track {subtitle_index} has no integer `id` field"
        ))
    })
}

/// Build the mkvmerge command-line argument list (excluding the binary itself).
///
/// Pure function — factored for unit testing.
pub fn build_mkvmerge_args(
    mkvmerge: &Path,
    video_path: &Path,
    subtitle_files: &[SubtitleFile],
    output_path: &Path,
    font_attachments: Option<&[PathBuf]>,
    original_sub_index: Option<usize>,
    original_sub_title: Option<&str>,
) -> Result<Vec<String>, VideoMuxError> {
    let mut args: Vec<String> = Vec::new();
    args.push("-o".to_string());
    args.push(output_path.to_string_lossy().to_string());

    if let Some(sub_idx) = original_sub_index {
        let track_id = resolve_mkvmerge_sub_track_id(mkvmerge, video_path, sub_idx)?;
        args.push("--subtitle-tracks".to_string());
        args.push(track_id.to_string());
        if let Some(title) = original_sub_title {
            args.push("--track-name".to_string());
            args.push(format!("{track_id}:{title}"));
        }
        args.push("--default-track-flag".to_string());
        args.push(format!("{track_id}:0"));
    } else {
        args.push("--no-subtitles".to_string());
    }

    args.push(video_path.to_string_lossy().to_string());

    for sub in subtitle_files {
        args.push("--language".to_string());
        args.push(format!("0:{}", sub.language));
        args.push("--track-name".to_string());
        args.push(format!("0:{}", sub.title));
        args.push("--default-track-flag".to_string());
        args.push(format!("0:{}", if sub.is_default { "1" } else { "0" }));
        args.push(sub.path.to_string_lossy().to_string());
    }

    if let Some(fonts) = font_attachments {
        for font_path in fonts {
            args.push("--attach-file".to_string());
            args.push(font_path.to_string_lossy().to_string());
        }
    }

    Ok(args)
}

fn run_mkvmerge(mkvmerge: &Path, args: &[String]) -> Result<(), VideoMuxError> {
    let output = Command::new(mkvmerge).args(args).output()?;
    // mkvmerge exit codes: 0 = success, 1 = warnings (still OK), 2+ = error
    if output.status.code().unwrap_or(2) >= 2 {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let msg = if !stdout.trim().is_empty() {
            stdout.trim().to_string()
        } else if !stderr.trim().is_empty() {
            stderr.trim().to_string()
        } else {
            "Unknown mkvmerge error".to_string()
        };
        return Err(VideoMuxError::MuxFailed(format!(
            "Failed to mux video: {msg}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ffmpeg mux argv builder (pure, testable)
// ---------------------------------------------------------------------------

/// Build the ffmpeg mux command-line argument list (excluding the binary itself).
///
/// Pure function — factored for unit testing.
pub fn build_ffmpeg_mux_args(
    video_path: &Path,
    subtitle_files: &[SubtitleFile],
    output_path: &Path,
    font_attachments: Option<&[PathBuf]>,
    original_sub_index: Option<usize>,
    original_sub_title: Option<&str>,
) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();

    args.push("-y".to_string());
    args.push("-i".to_string());
    args.push(video_path.to_string_lossy().to_string());

    for sub in subtitle_files {
        args.push("-i".to_string());
        args.push(sub.path.to_string_lossy().to_string());
    }

    args.push("-map".to_string());
    args.push("0:v".to_string());
    args.push("-map".to_string());
    args.push("0:a".to_string());
    // Preserve existing font/attachment streams from the original video
    args.push("-map".to_string());
    args.push("0:t?".to_string());

    // Preserve original subtitle track if specified
    let orig_sub_offset: usize = if let Some(idx) = original_sub_index {
        args.push("-map".to_string());
        args.push(format!("0:s:{idx}"));
        1
    } else {
        0
    };

    for i in 1..=subtitle_files.len() {
        args.push("-map".to_string());
        args.push(format!("{i}:0"));
    }

    args.push("-c:v".to_string());
    args.push("copy".to_string());
    args.push("-c:a".to_string());
    args.push("copy".to_string());

    // Select subtitle codec based on output container
    let is_mp4 = output_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
        == Some("mp4");
    let subtitle_codec = if is_mp4 { "mov_text" } else { "ass" };
    args.push("-c:s".to_string());
    args.push(subtitle_codec.to_string());

    // Attach new fonts (MKV only — not for MP4).
    //
    // Each attachment needs its mimetype/filename metadata bound to its own
    // attachment-stream index (`-metadata:s:t:0`, `:1`, …). Using the
    // index-less `-metadata:s:t` for every font would land all values on the
    // same (wrong) attachment when there is more than one font.
    if let Some(fonts) = font_attachments
        && !is_mp4
    {
        for (attach_idx, font_path) in fonts.iter().enumerate() {
            args.push("-attach".to_string());
            args.push(font_path.to_string_lossy().to_string());
            args.push(format!("-metadata:s:t:{attach_idx}"));
            args.push(format!("mimetype={}", mimetype_for_font(font_path)));
            args.push(format!("-metadata:s:t:{attach_idx}"));
            args.push(format!(
                "filename={}",
                font_path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
    }

    // Metadata for preserved original track
    if let Some(_idx) = original_sub_index {
        if let Some(title) = original_sub_title {
            args.push("-metadata:s:s:0".to_string());
            args.push(format!("title={title}"));
        }
        args.push("-disposition:s:0".to_string());
        args.push("0".to_string());
    }

    // Metadata for our added subtitle tracks (offset by orig_sub_offset)
    for (i, sub) in subtitle_files.iter().enumerate() {
        let idx = i + orig_sub_offset;
        args.push(format!("-metadata:s:s:{idx}"));
        args.push(format!("language={}", sub.language));
        args.push(format!("-metadata:s:s:{idx}"));
        args.push(format!("title={}", sub.title));
        let disposition = if sub.is_default { "default" } else { "0" };
        args.push(format!("-disposition:s:{idx}"));
        args.push(disposition.to_string());
    }

    args.push(output_path.to_string_lossy().to_string());
    args
}

fn run_ffmpeg_mux(ffmpeg: &Path, args: &[String]) -> Result<(), VideoMuxError> {
    let output = Command::new(ffmpeg).args(args).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let error_lines: Vec<&str> = stderr
            .lines()
            .filter(|l| l.to_ascii_lowercase().contains("error"))
            .collect();
        let msg = if !error_lines.is_empty() {
            error_lines.join("; ")
        } else {
            "Unknown ffmpeg error".to_string()
        };
        return Err(VideoMuxError::MuxFailed(format!(
            "Failed to mux video: {msg}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- parse_video_info -----

    const FFPROBE_VIDEO_JSON: &str = r#"{
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "High",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "24000/1001",
                "pix_fmt": "yuv420p",
                "bit_rate": "4200000",
                "tags": {},
                "disposition": {}
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "tags": {},
                "disposition": {}
            }
        ],
        "format": {
            "bit_rate": "5000000"
        }
    }"#;

    #[test]
    fn parse_video_info_streams() {
        let info = parse_video_info(FFPROBE_VIDEO_JSON).unwrap();
        assert_eq!(info.streams.len(), 2);
        assert_eq!(info.streams[0].codec_type.as_deref(), Some("video"));
        assert_eq!(info.streams[0].codec_name.as_deref(), Some("h264"));
        assert_eq!(info.streams[0].width, Some(1920));
        assert_eq!(info.streams[0].height, Some(1080));
        assert_eq!(info.format_bit_rate.as_deref(), Some("5000000"));
    }

    #[test]
    fn parse_video_info_empty() {
        let info = parse_video_info(r#"{"streams":[]}"#).unwrap();
        assert!(info.streams.is_empty());
        assert!(info.format_bit_rate.is_none());
    }

    // ----- probe_video_encoding (pure) -----

    #[test]
    fn parse_video_encoding_h264() {
        let info = parse_video_info(FFPROBE_VIDEO_JSON).unwrap();
        let enc = parse_video_encoding_from_info(&info, Path::new("test.mkv")).unwrap();
        assert_eq!(enc.codec_name, "h264");
        assert_eq!(enc.profile, "High");
        assert_eq!(enc.width, 1920);
        assert_eq!(enc.height, 1080);
        assert_eq!(enc.pix_fmt, "yuv420p");
        // 24000/1001 ≈ 23.976
        assert!((enc.fps - 23.976).abs() < 0.01, "fps={}", enc.fps);
        assert_eq!(enc.bit_rate, "4200000");
    }

    #[test]
    fn parse_video_encoding_fallback_bitrate_from_format() {
        let json = r#"{
            "streams": [{
                "index": 0,
                "codec_type": "video",
                "codec_name": "hevc",
                "tags": {},
                "disposition": {}
            }],
            "format": {"bit_rate": "8000000"}
        }"#;
        let info = parse_video_info(json).unwrap();
        let enc = parse_video_encoding_from_info(&info, Path::new("test.mkv")).unwrap();
        assert_eq!(enc.bit_rate, "8000000");
        assert_eq!(enc.fps, 24.0); // default
    }

    #[test]
    fn parse_video_encoding_no_video_stream_errors() {
        let json = r#"{"streams":[{"index":0,"codec_type":"audio","tags":{},"disposition":{}}]}"#;
        let info = parse_video_info(json).unwrap();
        let result = parse_video_encoding_from_info(&info, Path::new("test.mkv"));
        assert!(matches!(result, Err(VideoMuxError::NoVideoStream(_))));
    }

    // ----- parse_frame_rate -----

    #[test]
    fn frame_rate_integer() {
        assert_eq!(parse_frame_rate("24/1"), Some(24.0));
        assert_eq!(parse_frame_rate("30/1"), Some(30.0));
    }

    #[test]
    fn frame_rate_ntsc() {
        let fps = parse_frame_rate("24000/1001").unwrap();
        assert!((fps - 23.976023976).abs() < 1e-6);
    }

    #[test]
    fn frame_rate_invalid() {
        assert_eq!(parse_frame_rate("notanumber"), None);
        assert_eq!(parse_frame_rate("24/0"), None);
    }

    // ----- mimetype_for_font -----

    #[test]
    fn mimetype_otf() {
        assert_eq!(
            mimetype_for_font(Path::new("MyFont.otf")),
            "application/vnd.ms-opentype"
        );
        assert_eq!(
            mimetype_for_font(Path::new("MyFont.OTF")),
            "application/vnd.ms-opentype"
        );
    }

    #[test]
    fn mimetype_ttf() {
        assert_eq!(
            mimetype_for_font(Path::new("MyFont.ttf")),
            "application/x-truetype-font"
        );
        assert_eq!(
            mimetype_for_font(Path::new("MyFont.TTF")),
            "application/x-truetype-font"
        );
    }

    #[test]
    fn mimetype_unknown_extension() {
        assert_eq!(
            mimetype_for_font(Path::new("font.woff")),
            "application/x-truetype-font"
        );
    }

    // ----- parse_ffmpeg_version_string -----

    #[test]
    fn version_parse_typical() {
        let line = "ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers";
        assert_eq!(parse_ffmpeg_version_string(line), "7.1");
    }

    #[test]
    fn version_parse_git_build() {
        let line =
            "ffmpeg version N-113757-g12345abcd Copyright (c) 2000-2024 the FFmpeg developers";
        assert_eq!(parse_ffmpeg_version_string(line), "N-113757-g12345abcd");
    }

    // ----- parse_mkvmerge_sub_track_id -----

    const MKVMERGE_TRACKS_JSON: &str = r#"{
        "tracks": [
            {"id": 0, "type": "video"},
            {"id": 1, "type": "audio"},
            {"id": 2, "type": "subtitles"},
            {"id": 3, "type": "subtitles"}
        ]
    }"#;

    #[test]
    fn mkvmerge_track_id_selects_nth_subtitle() {
        assert_eq!(
            parse_mkvmerge_sub_track_id(MKVMERGE_TRACKS_JSON, 0).unwrap(),
            2
        );
        assert_eq!(
            parse_mkvmerge_sub_track_id(MKVMERGE_TRACKS_JSON, 1).unwrap(),
            3
        );
    }

    #[test]
    fn mkvmerge_track_id_out_of_range_errors() {
        let r = parse_mkvmerge_sub_track_id(MKVMERGE_TRACKS_JSON, 5);
        assert!(matches!(
            r,
            Err(VideoMuxError::SubtitleIndexOutOfRange { index: 5, count: 2 })
        ));
    }

    #[test]
    fn mkvmerge_track_id_missing_id_errors_not_zero() {
        // A subtitle track with no `id` must NOT silently mux track 0.
        let json = r#"{"tracks":[{"type":"subtitles"}]}"#;
        let r = parse_mkvmerge_sub_track_id(json, 0);
        assert!(
            matches!(r, Err(VideoMuxError::MkvmergeIdentifyFailed(_))),
            "expected MkvmergeIdentifyFailed, got {r:?}"
        );
    }

    #[test]
    fn mkvmerge_track_id_noninteger_id_errors() {
        let json = r#"{"tracks":[{"type":"subtitles","id":"two"}]}"#;
        let r = parse_mkvmerge_sub_track_id(json, 0);
        assert!(matches!(r, Err(VideoMuxError::MkvmergeIdentifyFailed(_))));
    }

    // ----- build_ffmpeg_mux_args -----

    fn make_sub(name: &str, lang: &str, is_default: bool) -> SubtitleFile {
        SubtitleFile {
            path: PathBuf::from(format!("/tmp/{name}.ass")),
            language: lang.to_string(),
            title: name.to_string(),
            is_default,
        }
    }

    #[test]
    fn ffmpeg_args_basic_mkv_no_orig() {
        let subs = vec![make_sub("Polish", "pol", true)];
        let args = build_ffmpeg_mux_args(
            Path::new("/tmp/input.mkv"),
            &subs,
            Path::new("/tmp/output.mkv"),
            None,
            None,
            None,
        );

        // Input file
        assert!(args.contains(&"-i".to_string()));
        assert!(args.contains(&"/tmp/input.mkv".to_string()));
        // Subtitle input
        assert!(args.contains(&"/tmp/Polish.ass".to_string()));
        // Maps
        assert!(args.contains(&"0:v".to_string()));
        assert!(args.contains(&"0:a".to_string()));
        assert!(args.contains(&"1:0".to_string()));
        // No original sub map
        assert!(!args.contains(&"0:s:0".to_string()));
        // codec
        assert!(args.contains(&"ass".to_string()));
        // metadata
        assert!(args.contains(&"-metadata:s:s:0".to_string()));
        assert!(args.contains(&"language=pol".to_string()));
        assert!(args.contains(&"title=Polish".to_string()));
        assert!(args.contains(&"-disposition:s:0".to_string()));
        assert!(args.contains(&"default".to_string()));
        // output last
        assert_eq!(args.last().unwrap(), "/tmp/output.mkv");
    }

    #[test]
    fn ffmpeg_args_mp4_uses_mov_text() {
        let subs = vec![make_sub("English", "eng", false)];
        let args = build_ffmpeg_mux_args(
            Path::new("/tmp/input.mp4"),
            &subs,
            Path::new("/tmp/output.mp4"),
            None,
            None,
            None,
        );
        assert!(args.contains(&"mov_text".to_string()));
        // No font attachments for MP4
        assert!(!args.contains(&"-attach".to_string()));
    }

    #[test]
    fn ffmpeg_args_with_original_sub_offset() {
        let subs = vec![make_sub("Polish", "pol", true)];
        let args = build_ffmpeg_mux_args(
            Path::new("/tmp/input.mkv"),
            &subs,
            Path::new("/tmp/output.mkv"),
            None,
            Some(0), // original_sub_index
            Some("English"),
        );

        // Original sub preserved
        assert!(args.contains(&"0:s:0".to_string()));
        // Our sub mapped as input 1
        assert!(args.contains(&"1:0".to_string()));
        // Metadata for original sub at s:0
        assert!(args.contains(&"-metadata:s:s:0".to_string()));
        assert!(args.contains(&"title=English".to_string()));
        assert!(args.contains(&"-disposition:s:0".to_string()));
        // Our sub is at offset 1
        assert!(args.contains(&"-metadata:s:s:1".to_string()));
        assert!(args.contains(&"language=pol".to_string()));
    }

    #[test]
    fn ffmpeg_args_font_attachment_mkv() {
        let subs = vec![make_sub("Polish", "pol", true)];
        let fonts = vec![PathBuf::from("/tmp/MyFont.ttf")];
        let args = build_ffmpeg_mux_args(
            Path::new("/tmp/input.mkv"),
            &subs,
            Path::new("/tmp/output.mkv"),
            Some(&fonts),
            None,
            None,
        );
        assert!(args.contains(&"-attach".to_string()));
        assert!(args.contains(&"/tmp/MyFont.ttf".to_string()));
        // Single font → attachment index 0.
        assert!(args.contains(&"-metadata:s:t:0".to_string()));
        assert!(args.contains(&"mimetype=application/x-truetype-font".to_string()));
        assert!(args.contains(&"filename=MyFont.ttf".to_string()));
    }

    #[test]
    fn ffmpeg_args_multiple_fonts_use_per_attachment_metadata_index() {
        let subs = vec![make_sub("Polish", "pol", true)];
        let fonts = vec![
            PathBuf::from("/tmp/FontOne.ttf"),
            PathBuf::from("/tmp/FontTwo.otf"),
        ];
        let args = build_ffmpeg_mux_args(
            Path::new("/tmp/input.mkv"),
            &subs,
            Path::new("/tmp/output.mkv"),
            Some(&fonts),
            None,
            None,
        );

        // Two attachments → two distinct per-attachment metadata indices.
        assert!(args.contains(&"-metadata:s:t:0".to_string()));
        assert!(args.contains(&"-metadata:s:t:1".to_string()));
        // The index-less form must NOT be used (that was the bug).
        assert!(!args.contains(&"-metadata:s:t".to_string()));

        // Verify the metadata for the SECOND font lands right after its
        // `-metadata:s:t:1` flag (not on attachment 0).
        let pos = args
            .iter()
            .position(|a| a == "-metadata:s:t:1")
            .expect("second attachment index present");
        assert_eq!(args[pos + 1], "mimetype=application/vnd.ms-opentype");
        assert_eq!(args[pos + 2], "-metadata:s:t:1");
        assert_eq!(args[pos + 3], "filename=FontTwo.otf");

        // And the first font's mimetype is truetype on index 0.
        let pos0 = args
            .iter()
            .position(|a| a == "-metadata:s:t:0")
            .expect("first attachment index present");
        assert_eq!(args[pos0 + 1], "mimetype=application/x-truetype-font");
        assert_eq!(args[pos0 + 3], "filename=FontOne.ttf");
    }

    #[test]
    fn ffmpeg_args_font_attachment_skipped_for_mp4() {
        let subs = vec![make_sub("Polish", "pol", true)];
        let fonts = vec![PathBuf::from("/tmp/MyFont.ttf")];
        let args = build_ffmpeg_mux_args(
            Path::new("/tmp/input.mp4"),
            &subs,
            Path::new("/tmp/output.mp4"),
            Some(&fonts),
            None,
            None,
        );
        assert!(!args.contains(&"-attach".to_string()));
    }

    #[test]
    fn ffmpeg_args_multiple_subs_no_orig() {
        let subs = vec![
            make_sub("English", "eng", false),
            make_sub("Polish", "pol", true),
        ];
        let args = build_ffmpeg_mux_args(
            Path::new("/tmp/input.mkv"),
            &subs,
            Path::new("/tmp/output.mkv"),
            None,
            None,
            None,
        );
        // Two sub inputs
        assert!(args.contains(&"1:0".to_string()));
        assert!(args.contains(&"2:0".to_string()));
        // Two metadata blocks
        assert!(args.contains(&"-metadata:s:s:0".to_string()));
        assert!(args.contains(&"-metadata:s:s:1".to_string()));
        assert!(args.contains(&"language=eng".to_string()));
        assert!(args.contains(&"language=pol".to_string()));
    }

    // ----- parse_video_info with subtitle streams (for extractor) -----

    const FFPROBE_WITH_SUBS_JSON: &str = r#"{
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "tags": {}, "disposition": {}},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "tags": {"language": "jpn"}, "disposition": {}},
            {"index": 2, "codec_type": "subtitle", "codec_name": "ass",
             "tags": {"language": "eng", "title": "English Full Dialogue"},
             "disposition": {"default": 1, "forced": 0}},
            {"index": 3, "codec_type": "subtitle", "codec_name": "ass",
             "tags": {"language": "eng", "title": "English Signs/Songs"},
             "disposition": {"default": 0, "forced": 0}}
        ]
    }"#;

    #[test]
    fn parse_video_info_with_subtitle_streams() {
        let info = parse_video_info(FFPROBE_WITH_SUBS_JSON).unwrap();
        let sub_streams: Vec<_> = info
            .streams
            .iter()
            .filter(|s| s.codec_type.as_deref() == Some("subtitle"))
            .collect();
        assert_eq!(sub_streams.len(), 2);
        assert_eq!(
            sub_streams[0].tags.get("title").map(|s| s.as_str()),
            Some("English Full Dialogue")
        );
        assert_eq!(
            sub_streams[1].tags.get("title").map(|s| s.as_str()),
            Some("English Signs/Songs")
        );
    }
}
