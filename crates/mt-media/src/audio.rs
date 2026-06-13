//! Audio track extraction for ASR (16 kHz mono s16le WAV).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::info;

use mt_core::{MtError, Result};

use crate::ffmpeg::{get_ffmpeg, get_video_info};

/// ISO 639-1 language code plus common aliases.
fn language_matches(requested: &str, tag: &str) -> bool {
    let tag = tag.to_lowercase();
    match requested {
        "en" => tag == "en" || tag == "eng",
        "ja" => tag == "ja" || tag == "jpn" || tag == "jp",
        _ => tag == requested,
    }
}

/// Find the stream index of the first audio track tagged with `language`.
///
/// Falls back to the first audio stream when the file has exactly one
/// audio track with no language tag (common for web rips).
pub fn find_audio_stream(video: &Path, language: &str) -> Result<Option<i64>> {
    let info = get_video_info(video).map_err(|e| MtError::Subprocess {
        cmd: "ffprobe".to_string(),
        code: None,
        stderr: e.to_string(),
    })?;
    let audio_streams: Vec<_> = info
        .streams
        .iter()
        .filter(|s| s.codec_type.as_deref() == Some("audio"))
        .collect();

    for s in &audio_streams {
        let lang = s.tags.get("language").map(|s| s.as_str()).unwrap_or("");
        if language_matches(language, lang) {
            return Ok(Some(s.index as i64));
        }
    }

    // Fallback: single untagged audio track
    if audio_streams.len() == 1 && !audio_streams[0].tags.contains_key("language") {
        return Ok(Some(audio_streams[0].index as i64));
    }

    Ok(None)
}

/// Extract `stream_index` to a 16 kHz mono pcm_s16le WAV at `out_wav`.
pub fn extract_wav(video: &Path, stream_index: i64, out_wav: &Path) -> Result<PathBuf> {
    if let Some(parent) = out_wav.parent() {
        fs::create_dir_all(parent).map_err(MtError::Io)?;
    }

    let ffmpeg = get_ffmpeg().map_err(|e| MtError::Subprocess {
        cmd: "ffmpeg".to_string(),
        code: None,
        stderr: e.to_string(),
    })?;
    let result = Command::new(&ffmpeg)
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-map")
        .arg(format!("0:{}", stream_index))
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(out_wav)
        .output()
        .map_err(MtError::Io)?;

    if !result.status.success() || !out_wav.exists() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(MtError::Subprocess {
            cmd: "ffmpeg audio extract".to_string(),
            code: result.status.code(),
            stderr: truncate(&stderr, 500),
        });
    }

    info!(
        "Extracted audio stream {} -> {}",
        stream_index,
        out_wav.display()
    );
    Ok(out_wav.to_path_buf())
}

/// Get WAV duration in milliseconds.
pub fn wav_duration_ms(wav: &Path) -> Result<i64> {
    let data = fs::read(wav).map_err(MtError::Io)?;
    if data.len() < 44 {
        return Err(MtError::Parse("invalid WAV file".into()));
    }
    // Parse WAV header
    let channels = u16::from_le_bytes([data[22], data[23]]);
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);
    let data_size = u32::from_le_bytes([data[40], data[41], data[42], data[43]]);

    let bytes_per_second = sample_rate as u64 * channels as u64 * (bits_per_sample / 8) as u64;
    if bytes_per_second == 0 {
        return Err(MtError::Parse("invalid WAV parameters".into()));
    }
    Ok((data_size as u64 * 1000 / bytes_per_second) as i64)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_matches() {
        assert!(language_matches("en", "en"));
        assert!(language_matches("en", "eng"));
        assert!(language_matches("ja", "jpn"));
        assert!(!language_matches("en", "pl"));
    }

    #[test]
    fn test_wav_duration_invalid() {
        let result = wav_duration_ms(Path::new("/nonexistent.wav"));
        assert!(result.is_err());
    }
}
