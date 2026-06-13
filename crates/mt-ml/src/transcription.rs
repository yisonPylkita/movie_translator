//! Audio→subtitle transcription — pure Rust, no Python.
//!
//! Two engines (picked by the `benchmarks/asr` bake-off):
//! - `"apple"`  — Apple SpeechAnalyzer (macOS 26+, ANE). Calls the compiled
//!   Swift `transcribe_bridge` binary via subprocess.
//! - `"whisper"` — whisper.cpp subprocess via `whisper-cli`.
//!
//! Audio extraction (16 kHz mono WAV), pause-boundary detection (energy-based
//! VAD), sentence splitting, and post-processing (hallucination filtering) are
//! handled in Rust with zero Python dependencies.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use mt_core::swift_bridge::{ensure_compiled, macos_at_least};
use mt_core::{DialogueLine, MtError, Result};
use mt_media::audio::{extract_wav, find_audio_stream, wav_duration_ms};
use mt_subtitles::postfilter::clean_segments;
use mt_subtitles::splitter::split_segments;
use mt_subtitles::srt::to_srt_string;
use serde::Deserialize;
use serde_json::from_slice;
use tracing::{info, warn};

// ── Paths for the transcription Swift bridge ─────────────────────────────────

fn transcribe_source() -> PathBuf {
    let candidates = [
        "crates/mt-ml/swift/transcribe_bridge.swift",
        "../crates/mt-ml/swift/transcribe_bridge.swift",
        "movie_translator/transcription/swift/transcribe_bridge.swift",
        "../movie_translator/transcription/swift/transcribe_bridge.swift",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    if let Ok(root) = env::var("MT_REPO_ROOT") {
        let p = PathBuf::from(root)
            .join("movie_translator/transcription/swift/transcribe_bridge.swift");
        if p.exists() {
            return p;
        }
    }
    if let Ok(cwd) = env::current_dir() {
        for ancestor in cwd.ancestors() {
            let p = ancestor.join("movie_translator/transcription/swift/transcribe_bridge.swift");
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from("movie_translator/transcription/swift/transcribe_bridge.swift")
}

fn transcribe_binary() -> PathBuf {
    let src = transcribe_source();
    let mut bin = src;
    bin.set_file_name("transcribe_bridge");
    bin.with_extension("")
}

fn ensure_transcribe_bridge() -> Result<PathBuf> {
    let source = transcribe_source();
    let binary = transcribe_binary();
    ensure_compiled(&source, &binary, &[], Duration::from_secs(120))
}

// ── Apple SpeechAnalyzer backend ────────────────────────────────────────────

#[derive(Deserialize)]
struct TranscribeSegment {
    #[serde(default)]
    text: String,
    #[serde(default)]
    start_ms: i64,
    #[serde(default)]
    end_ms: i64,
}

#[derive(Deserialize)]
struct TranscribeResponse {
    segments: Vec<TranscribeSegment>,
}

fn language_to_locale(language: &str) -> Option<&'static str> {
    match language {
        "en" => Some("en-US"),
        "ja" => Some("ja-JP"),
        _ => None,
    }
}

/// Transcribe using Apple SpeechAnalyzer via Swift bridge.
fn transcribe_apple(wav: &Path, language: &str) -> Result<Vec<DialogueLine>> {
    let locale = language_to_locale(language).ok_or_else(|| {
        MtError::Parse(format!("Apple backend: unsupported language {language:?}"))
    })?;

    let binary = ensure_transcribe_bridge()?;
    info!("SpeechAnalyzer transcribing {} ({locale})", wav.display());

    let output = Command::new(&binary)
        .arg(wav)
        .arg(locale)
        .output()
        .map_err(MtError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MtError::Subprocess {
            cmd: "transcribe_bridge".to_string(),
            code: output.status.code(),
            stderr: truncate(&stderr, 500),
        });
    }

    let response: TranscribeResponse = from_slice(&output.stdout).map_err(|e| {
        let out = String::from_utf8_lossy(&output.stdout);
        MtError::Parse(format!(
            "Invalid JSON from transcribe_bridge: {e}\nOutput: {}",
            truncate(&out, 200),
        ))
    })?;

    let mut lines: Vec<DialogueLine> = Vec::new();
    for s in response.segments {
        let text = s.text.trim().to_string();
        if !text.is_empty() && s.end_ms > s.start_ms {
            lines.push(DialogueLine {
                start_ms: s.start_ms,
                end_ms: s.end_ms,
                text,
            });
        }
    }

    Ok(lines)
}

// ── Whisper backend (whisper.cpp subprocess) ─────────────────────────────────

fn transcribe_whisper(wav: &Path, language: &str) -> Result<Vec<DialogueLine>> {
    // Check for whisper-cli on PATH
    let has_whisper = Command::new("whisper-cli")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !has_whisper {
        return Err(MtError::Parse(
            "whisper-cli not found on PATH; install whisper.cpp (brew install whisper-cpp)".into(),
        ));
    }

    // Call whisper-cli -- language must be the ISO code whisper.cpp understands
    let lang = match language {
        "en" => Some("en"),
        "ja" => Some("ja"),
        "pl" => Some("pl"),
        _ => None,
    }
    .unwrap_or(language);

    let output_srt = wav.with_extension("srt");

    let output = Command::new("whisper-cli")
        .arg("-m")
        .arg(find_whisper_model()?)
        .arg("-f")
        .arg(wav)
        .arg("-l")
        .arg(lang)
        .arg("-osrt")
        .arg("-of")
        .arg(output_srt.with_extension(""))
        .output()
        .map_err(MtError::Io)?;

    if !output.status.success() || !output_srt.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MtError::Subprocess {
            cmd: "whisper-cli".to_string(),
            code: output.status.code(),
            stderr: truncate(&stderr, 500),
        });
    }

    // Parse the SRT output back into DialogueLines
    let srt_content = fs::read_to_string(&output_srt).map_err(MtError::Io)?;
    let _ = fs::remove_file(&output_srt);

    // Simple SRT parser
    let mut lines = Vec::new();
    let srt_lines: Vec<_> = srt_content.lines().collect();
    let mut i = 0;

    while i + 2 < srt_lines.len() {
        // Skip empty lines and index lines
        if srt_lines[i].trim().is_empty() || srt_lines[i].trim().chars().all(|c| c.is_ascii_digit())
        {
            i += 1;
            continue;
        }
        // Check for timestamp line: 00:01:23,456 --> 00:01:25,789
        if srt_lines[i].contains("-->") {
            let time_parts: Vec<_> = srt_lines[i].split("-->").collect();
            if time_parts.len() == 2 {
                let start_ms = parse_srt_time(time_parts[0]);
                let end_ms = parse_srt_time(time_parts[1]);
                // Collect text lines until next blank line
                let mut text_parts = Vec::new();
                i += 1;
                while i < srt_lines.len()
                    && !srt_lines[i].trim().is_empty()
                    && !srt_lines[i].contains("-->")
                {
                    text_parts.push(srt_lines[i].trim());
                    i += 1;
                }
                let text = text_parts.join(" ").trim().to_string();
                if !text.is_empty() && end_ms > start_ms {
                    lines.push(DialogueLine {
                        start_ms,
                        end_ms,
                        text,
                    });
                }
                continue;
            }
        }
        i += 1;
    }

    Ok(lines)
}

fn parse_srt_time(s: &str) -> i64 {
    let s = s.trim();
    let parts: Vec<_> = s.split(':').collect();
    if parts.len() == 3 {
        let h: i64 = parts[0].parse().unwrap_or(0);
        let m: i64 = parts[1].parse().unwrap_or(0);
        let sec_parts: Vec<_> = parts[2].split(',').collect();
        let sec: i64 = sec_parts[0].parse().unwrap_or(0);
        let ms: i64 = sec_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        h * 3_600_000 + m * 60_000 + sec * 1000 + ms
    } else {
        0
    }
}

fn find_whisper_model() -> Result<PathBuf> {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let candidates = [
        format!("{home}/.cache/whisper/ggml-large-v3.bin"),
        format!("{home}/.cache/whisper/ggml-medium.bin"),
        format!("{home}/.cache/whisper/ggml-small.bin"),
        format!("{home}/.cache/whisper/ggml-base.bin"),
        format!("{home}/.cache/whisper/ggml-tiny.bin"),
        "models/ggml-large-v3.bin".to_string(),
        "models/ggml-medium.bin".to_string(),
        "models/ggml-tiny.bin".to_string(),
        "/usr/local/share/whisper/models/ggml-large-v3.bin".to_string(),
        "/opt/homebrew/share/whisper/models/ggml-large-v3.bin".to_string(),
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(MtError::Parse(
        "No whisper model found. Download one: \
         curl -L -o ~/.cache/whisper/ggml-large-v3.bin \
         https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
            .into(),
    ))
}

// ── Energy-based VAD (silence detection) ────────────────────────────────────

/// Find pause boundaries (silence gaps > `min_pause_ms`) in a WAV file.
///
/// Uses RMS energy-based voice activity detection — no Python/webrtcvad needed.
fn find_pause_boundaries(wav: &Path, min_pause_ms: i64) -> Result<Vec<i64>> {
    let data = fs::read(wav).map_err(MtError::Io)?;

    // Parse WAV header
    if data.len() < 44 {
        return Err(MtError::Parse("invalid WAV file".into()));
    }
    let channels = u16::from_le_bytes([data[22], data[23]]) as usize;
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]) as usize;
    let data_size = u32::from_le_bytes([data[40], data[41], data[42], data[43]]) as usize;

    // Extract PCM samples (handle 16-bit signed mono)
    let bytes_per_sample = channels * (bits_per_sample / 8);
    let num_samples = data_size / bytes_per_sample;

    let mut samples = Vec::with_capacity(num_samples / channels);
    let base = 44; // start of PCM data

    if bits_per_sample == 16 {
        for i in (base..base + data_size)
            .step_by(bytes_per_sample)
            .take(num_samples)
        {
            if i + 1 < data.len() {
                let sample = i16::from_le_bytes([data[i], data[i + 1]]);
                let normalized = sample as f64 / 32768.0f64;
                samples.push(normalized);
            }
        }
    } else {
        // 8-bit unsigned
        for i in (base..base + data_size)
            .step_by(bytes_per_sample)
            .take(num_samples)
        {
            if i < data.len() {
                let sample = (data[i] as f64 - 128.0) / 128.0;
                samples.push(sample);
            }
        }
    }

    // Frame-level energy computation (30ms frames)
    let frame_size = (sample_rate as usize * 30 / 1000).max(1);
    let min_silence_frames =
        (min_pause_ms as usize * sample_rate as usize / 1000 / frame_size).max(1);

    let mut boundaries = Vec::new();
    let mut silence_start: Option<usize> = None;
    let energy_threshold = 0.01; // RMS threshold for silence

    for frame in 0..samples.len() / frame_size {
        let start = frame * frame_size;
        let end = (start + frame_size).min(samples.len());
        if start >= end {
            break;
        }

        let frame_samples = &samples[start..end];
        let rms: f64 =
            (frame_samples.iter().map(|s| s * s).sum::<f64>() / frame_samples.len() as f64).sqrt();

        let is_speech = rms > energy_threshold;

        if is_speech {
            if let Some(silence_frame_start) = silence_start {
                let silence_frames = frame - silence_frame_start;
                if silence_frames >= min_silence_frames {
                    // Convert frame index to milliseconds
                    let silence_ms =
                        (frame as f64 * frame_size as f64 / sample_rate as f64 * 1000.0) as i64;
                    let start_ms = (silence_frame_start as f64 * frame_size as f64
                        / sample_rate as f64
                        * 1000.0) as i64;
                    let mid_ms = (start_ms + silence_ms) / 2;
                    boundaries.push(mid_ms);
                }
                silence_start = None;
            }
        } else {
            if silence_start.is_none() {
                silence_start = Some(frame);
            }
        }
    }

    Ok(boundaries)
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Check if a transcription engine is available.
pub fn is_available(engine: &str) -> bool {
    match engine {
        "apple" => macos_at_least(26) && transcribe_source().exists(),
        "whisper" => Command::new("whisper-cli")
            .arg("--help")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false),
        _ => false,
    }
}

/// Transcribe `video`'s `language` audio track to an SRT, or `None` when the
/// track / engine / usable lines are missing.
pub fn transcribe_to_srt(
    video: &Path,
    output_dir: &Path,
    language: &str,
    engine: &str,
) -> Result<Option<PathBuf>> {
    // Check engine availability
    match engine {
        "apple" => {
            if !is_available(engine) {
                warn!("Apple SpeechAnalyzer unavailable (need macOS 26+ + bridge source)");
                return Ok(None);
            }
        }
        "whisper" => {
            if !is_available(engine) {
                warn!("Whisper unavailable (install whisper-cli)");
                return Ok(None);
            }
        }
        other => {
            return Err(MtError::Parse(format!(
                "unknown transcription engine {other:?} (use \"apple\" or \"whisper\")"
            )));
        }
    }

    // Find audio stream via ffprobe (pure Rust, mt-media)
    let stream = match find_audio_stream(video, language)? {
        Some(s) => s,
        None => {
            info!(
                "no {language:?} audio track in {}; skipping transcription",
                video.display()
            );
            return Ok(None);
        }
    };

    // Extract WAV via ffmpeg (pure Rust, mt-media)
    fs::create_dir_all(output_dir).map_err(MtError::Io)?;
    let wav = output_dir.join(format!("transcribe_{language}.wav"));
    extract_wav(video, stream, &wav)?;

    // Get WAV duration before cleaning up
    let wav_dur = wav_duration_ms(&wav).unwrap_or(0);

    // Transcribe
    let raw_lines = match engine {
        "apple" => {
            let lines = transcribe_apple(&wav, language)?;

            // Use VAD pause boundaries to improve segmentation
            let boundaries = find_pause_boundaries(&wav, 300).ok();
            let split_lines = split_segments(&lines, boundaries.as_deref());

            // Clean up WAV
            let _ = fs::remove_file(&wav);

            split_lines
        }
        "whisper" => {
            let lines = transcribe_whisper(&wav, language)?;

            // Clean up WAV
            let _ = fs::remove_file(&wav);

            // Post-filter
            clean_segments(&lines, wav_dur)
        }
        _ => unreachable!(),
    };

    if raw_lines.is_empty() {
        warn!("{engine} transcription produced no usable lines");
        return Ok(None);
    }

    // Write SRT
    let srt_path = output_dir.join(format!("transcribed_{language}.srt"));
    let subs = mt_subtitles::model::Subtitles {
        bom: false,
        script_info_lines: vec![],
        pre_styles_sections: vec![],
        styles_format: vec![],
        styles: vec![],
        pre_events_sections: vec![],
        events_format: vec![],
        events: raw_lines
            .iter()
            .map(|l| mt_subtitles::model::Event {
                kind: mt_subtitles::model::EventKind::Dialogue,
                layer: 0,
                start_ms: l.start_ms,
                end_ms: l.end_ms,
                style: "Default".to_string(),
                name: String::new(),
                margin_l: 0,
                margin_r: 0,
                margin_v: 0,
                effect: String::new(),
                text: l.text.clone(),
            })
            .collect(),
        post_events_sections: vec![],
    };
    let srt_content = to_srt_string(&subs);
    fs::write(&srt_path, srt_content).map_err(MtError::Io)?;
    info!(
        "{engine} transcription: {} lines -> {}",
        raw_lines.len(),
        srt_path.display()
    );

    Ok(Some(srt_path))
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max..])
    }
}
