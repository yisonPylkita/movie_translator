//! Subtitle OCR — Rust-native, no Python.
//!
//! On macOS, uses Apple Vision framework via a compiled Swift bridge binary.
//! The Swift bridge is compiled on first use from an inline source.
//!
//! PGS binary parsing is in `mt_media::pgs_parser`; this module adds the OCR
//! step on top of parsed PGS bitmaps, as well as burned-in subtitle extraction
//! via frame-level change detection and OCR.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, from_slice};
use tracing::{info, warn};

use mt_core::{BoundingBox, BurnedInResult, MtError, OCRResult, Result};

// ── Public API ─────────────────────────────────────────────────────────────

/// Extract a PGS subtitle track to SRT using Apple Vision OCR.
pub fn ocr_pgs(video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
    #[cfg(target_os = "macos")]
    {
        ocr_pgs_macos(video, track_index, work_dir)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (video, track_index, work_dir);
        warn!("PGS OCR requires macOS (Vision framework)");
        Ok(None)
    }
}

/// Extract burned-in subtitles via OCR.
pub fn ocr_burned_in(
    video: &Path,
    output_dir: &Path,
    crop_ratio: f64,
    fps: u32,
) -> Result<BurnedInResult> {
    #[cfg(target_os = "macos")]
    {
        ocr_burned_in_macos(video, output_dir, crop_ratio, fps, "en")
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (video, output_dir, crop_ratio, fps);
        let _ = (video, output_dir, crop_ratio, fps);
        Err(MtError::Parse(
            "Burned-in OCR requires macOS (Vision framework)".into(),
        ))
    }
}

/// Check whether Vision OCR is available on this system.
pub fn is_vision_ocr_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::path::Path;
        Command::new("swiftc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
            && Path::new("/System/Library/Frameworks/Vision.framework/Versions/Current/Vision")
                .exists()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

// ── macOS-only implementations ────────────────────────────────────────────
// These are only compiled on macOS; the cfg gates above dispatch to them.

#[cfg(target_os = "macos")]
fn ocr_pgs_macos(video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
    use std::process::Command;

    let pgs_dir = work_dir.join("pgs_ocr");
    fs::create_dir_all(&pgs_dir).map_err(MtError::Io)?;

    // Step 1: Extract .sup stream from MKV
    let sup_path = pgs_dir.join("track.sup");
    info!(
        "Extracting PGS track {track_index} from {}",
        video.display()
    );

    let result = Command::new("mkvextract")
        .arg("tracks")
        .arg(video)
        .arg(format!("{}:{}", track_index, sup_path.display()))
        .output()
        .map_err(MtError::Io)?;

    if !result.status.success() || !sup_path.exists() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        warn!("Failed to extract PGS track: {stderr}");
        return Ok(None);
    }

    // Step 2: Parse PGS binary format
    info!("Parsing PGS subtitle stream...");
    let data = fs::read(&sup_path).map_err(MtError::Io)?;
    let events = mt_media::pgs_parser::parse_sup(&data);

    if events.is_empty() {
        warn!("No subtitle images found in PGS track");
        let _ = fs::remove_file(&sup_path);
        return Ok(None);
    }

    info!("Found {} subtitle images, running OCR...", events.len());

    // Step 3: OCR each image
    let bridge = ensure_ocr_bridge()?;
    let mut prev_text = String::new();
    let mut line_start_ms: i64 = 0;
    let mut dialogue_lines: Vec<mt_core::DialogueLine> = Vec::new();

    for (i, event) in events.iter().enumerate() {
        let img_path = pgs_dir.join(format!("sub_{:04}.pgm", i));
        write_grayscale_pgm(
            &img_path,
            &event.pixels,
            event.width as usize,
            event.height as usize,
        )
        .map_err(MtError::Io)?;

        let text = ocr_image(&bridge, &img_path)?;
        let _ = fs::remove_file(&img_path);

        let pts_ms = event.pts_ms as i64;

        if !text.is_empty() && text != prev_text {
            if !prev_text.is_empty() && line_start_ms > 0 {
                dialogue_lines.push(mt_core::DialogueLine {
                    start_ms: line_start_ms,
                    end_ms: pts_ms,
                    text: prev_text.clone(),
                });
            }
            line_start_ms = pts_ms;
            prev_text = text;
        } else if text.is_empty() && !prev_text.is_empty() {
            dialogue_lines.push(mt_core::DialogueLine {
                start_ms: line_start_ms,
                end_ms: pts_ms,
                text: prev_text.clone(),
            });
            prev_text = String::new();
        }

        if (i + 1) % 100 == 0 {
            info!("OCR progress: {}/{}", i + 1, events.len());
        }
    }

    if !prev_text.is_empty() {
        let last_pts = events
            .last()
            .map(|e| e.pts_ms as i64)
            .unwrap_or(line_start_ms);
        dialogue_lines.push(mt_core::DialogueLine {
            start_ms: line_start_ms,
            end_ms: last_pts + 3000,
            text: prev_text,
        });
    }

    if dialogue_lines.is_empty() {
        warn!("OCR produced no text from PGS images");
        let _ = fs::remove_file(&sup_path);
        return Ok(None);
    }

    info!(
        "Extracted {} dialogue lines from PGS track",
        dialogue_lines.len()
    );

    let srt_path = work_dir.join(format!(
        "{}_pgs_ocr.srt",
        video
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
    ));
    let subs = mt_subtitles::model::Subtitles {
        bom: false,
        script_info_lines: vec![],
        pre_styles_sections: vec![],
        styles_format: vec![],
        styles: vec![],
        pre_events_sections: vec![],
        events_format: vec![],
        events: dialogue_lines
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
    let srt_content = mt_subtitles::srt::to_srt_string(&subs);
    fs::write(&srt_path, srt_content).map_err(MtError::Io)?;

    let _ = fs::remove_file(&sup_path);
    Ok(Some(srt_path))
}

#[cfg(target_os = "macos")]
fn ocr_burned_in_macos(
    video: &Path,
    output_dir: &Path,
    crop_ratio: f64,
    fps: u32,
    _language: &str,
) -> Result<BurnedInResult> {
    let ocr_dir = output_dir.join("_ocr_frames");
    fs::create_dir_all(&ocr_dir).map_err(MtError::Io)?;

    let scale_width = 1280u32;
    let pixel_delta: u8 = 25;
    let change_fraction = 0.006;
    let variance_threshold = 200.0f64;

    // Step 1: Extract frames
    let frames = extract_subtitle_frames(video, &ocr_dir, fps, crop_ratio, scale_width)?;
    if frames.is_empty() {
        return Err(MtError::Parse("No frames extracted from video".into()));
    }

    // Step 2: Detect transitions
    let transition_frames =
        detect_transitions(&frames, pixel_delta, change_fraction, variance_threshold)?;
    if transition_frames.is_empty() {
        return Err(MtError::Parse("No subtitle transitions detected".into()));
    }

    info!(
        "Change detection: {} transitions out of {} frames",
        transition_frames.len(),
        frames.len()
    );

    // Step 3: OCR transition frames
    let bridge = ensure_ocr_bridge()?;
    let mut frame_texts: Vec<(i64, String)> = Vec::new();

    for (i, (frame_path, timestamp_ms)) in transition_frames.iter().enumerate() {
        let text = ocr_image(&bridge, frame_path)?;
        frame_texts.push((*timestamp_ms, text));

        if (i + 1) % 100 == 0 {
            info!("  OCR progress: {}/{}", i + 1, transition_frames.len());
        }
    }

    // Build dialogue lines
    let lines = build_dialogue_lines(&frame_texts);
    if lines.is_empty() {
        return Err(MtError::Parse(
            "OCR produced no usable subtitle lines".into(),
        ));
    }

    info!("Extracted {} subtitle lines via OCR", lines.len());

    let srt_path = output_dir.join(format!(
        "{}_ocr.srt",
        video
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
    ));
    let subs = mt_subtitles::model::Subtitles {
        bom: false,
        script_info_lines: vec![],
        pre_styles_sections: vec![],
        styles_format: vec![],
        styles: vec![],
        pre_events_sections: vec![],
        events_format: vec![],
        events: lines
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
    let srt_content = mt_subtitles::srt::to_srt_string(&subs);
    std::fs::write(&srt_path, srt_content).map_err(MtError::Io)?;

    let _ = std::fs::remove_dir_all(&ocr_dir);

    // Build OCR results
    let ocr_results: Vec<_> = frame_texts
        .into_iter()
        .map(|(ts, text)| OCRResult {
            timestamp_ms: ts,
            text,
            boxes: vec![BoundingBox {
                x: 0.0,
                y: 1.0 - crop_ratio,
                width: 1.0,
                height: crop_ratio,
            }],
        })
        .collect();

    Ok(BurnedInResult {
        srt_path,
        ocr_results,
    })
}

// ── OCR bridge (Swift) ────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn ocr_bridge_source() -> PathBuf {
    let candidates = [
        "crates/mt-ml/swift/ocr_bridge.swift",
        "../crates/mt-ml/swift/ocr_bridge.swift",
        "movie_translator/ocr/swift/ocr_bridge.swift",
        "../movie_translator/ocr/swift/ocr_bridge.swift",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    if let Ok(root) = env::var("MT_REPO_ROOT") {
        let p = PathBuf::from(root).join("crates/mt-ml/swift/ocr_bridge.swift");
        if p.exists() {
            return p;
        }
    }
    // Write inline source
    let fallback = PathBuf::from(".translate_temp/ocr_bridge.swift");
    if !fallback.exists() {
        if let Some(parent) = fallback.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&fallback, INLINE_OCR_SWIFT);
    }
    fallback
}

#[cfg(target_os = "macos")]
fn ocr_bridge_binary() -> PathBuf {
    let mut bin = ocr_bridge_source();
    bin.set_file_name("ocr_bridge");
    bin.with_extension("")
}

#[cfg(target_os = "macos")]
fn ensure_ocr_bridge() -> Result<PathBuf> {
    let source = ocr_bridge_source();
    let binary = ocr_bridge_binary();

    // Check if already compiled and fresh
    if binary.exists() {
        let src_mtime = source.metadata().and_then(|m| m.modified());
        let bin_mtime = binary.metadata().and_then(|m| m.modified());
        if let (Ok(s), Ok(b)) = (src_mtime, bin_mtime)
            && s <= b
        {
            return Ok(binary);
        }
    }

    // Compile
    info!("Compiling OCR Swift bridge: {}", source.display());
    if let Some(parent) = binary.parent() {
        fs::create_dir_all(parent).map_err(MtError::Io)?;
    }

    let output = Command::new("swiftc")
        .arg("-O")
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .arg("-framework")
        .arg("Vision")
        .arg("-framework")
        .arg("Quartz")
        .output()
        .map_err(MtError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MtError::Parse(format!(
            "OCR Swift bridge compilation failed: {}",
            truncate(&stderr, 500)
        )));
    }

    info!("Compiled: {}", binary.display());
    Ok(binary)
}

#[cfg(target_os = "macos")]
fn ocr_image(bridge: &Path, image_path: &Path) -> Result<String> {
    let output = Command::new(bridge)
        .arg(image_path)
        .output()
        .map_err(MtError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MtError::Subprocess {
            cmd: "ocr_bridge".to_string(),
            code: output.status.code(),
            stderr: truncate(&stderr, 500),
        });
    }

    Ok(std::str::from_utf8(&output.stdout)
        .unwrap_or("")
        .trim()
        .to_string())
}

const INLINE_OCR_SWIFT: &str = r#"
import Foundation
import Vision
import Quartz

guard CommandLine.arguments.count > 1 else {
    fputs("Usage: ocr_bridge <image-path>", stderr)
    exit(1)
}
let imagePath = CommandLine.arguments[1]
let url = URL(fileURLWithPath: imagePath)
guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
    exit(0)
}

let request = VNRecognizeTextRequest { request, error in
    if let error = error {
        fputs("Vision error: \(error.localizedDescription)", stderr)
        exit(1)
    }
    guard let observations = request.results as? [VNRecognizedTextObservation] else { exit(0) }
    var lines: [String] = []
    for obs in observations {
        if let candidate = obs.topCandidates(1).first {
            lines.append(candidate.string)
        }
    }
    print(lines.joined(separator: "\n"))
}
request.recognitionLevel = .accurate
request.recognitionLanguages = ["en"]
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])
"#;

// ── Frame extraction ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn extract_subtitle_frames(
    video: &Path,
    out_dir: &Path,
    _fps: u32,
    crop_ratio: f64,
    scale_width: u32,
) -> Result<Vec<(PathBuf, i64)>> {
    use std::process::Command;

    let ffmpeg = find_ffmpeg()?;
    let ffprobe = find_ffprobe()?;

    // Get video dimensions
    let info_output = Command::new(&ffprobe)
        .arg("-v")
        .arg("quiet")
        .arg("-print_format")
        .arg("json")
        .arg("-show_streams")
        .arg(video)
        .output()
        .map_err(MtError::Io)?;

    let info: Value = from_slice(&info_output.stdout).map_err(|e| MtError::Parse(e.to_string()))?;

    let streams = info["streams"]
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| MtError::Parse("No video streams found".into()))?;
    let width = streams["width"].as_u64().unwrap_or(1920) as u32;
    let height = streams["height"].as_u64().unwrap_or(1080) as u32;

    // Calculate crop
    let crop_height = (height as f64 * crop_ratio) as u32;
    let crop_y = height - crop_height;

    // Build filter
    let filter = if width != scale_width {
        let scale_h = (crop_height as f64 * scale_width as f64 / width as f64) as u32;
        format!(
            "crop={}:{}:0:{},scale={}:{}",
            width, crop_height, crop_y, scale_width, scale_h
        )
    } else {
        format!("crop={}:{}:0:{}", width, crop_height, crop_y)
    };

    let out_pattern = out_dir.join("frame_%05d.png");

    let output = Command::new(&ffmpeg)
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-vf")
        .arg(&filter)
        .arg("-vsync")
        .arg("vfr")
        .arg(&out_pattern)
        .output()
        .map_err(MtError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MtError::Subprocess {
            cmd: "ffmpeg frame extraction".to_string(),
            code: output.status.code(),
            stderr: truncate(&stderr, 500),
        });
    }

    // Collect frames
    let mut frames: Vec<(PathBuf, i64)> = Vec::new();
    let mut dir = fs::read_dir(out_dir).map_err(MtError::Io)?;
    while let Some(entry) = dir.next().transpose().map_err(MtError::Io)? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("png") {
            continue;
        }
        // Parse frame number from filename: frame_NNNNN.png
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let num_str = stem.strip_prefix("frame_").unwrap_or(stem);
        if let Ok(num) = num_str.parse::<i64>() {
            frames.push((path, num));
        }
    }

    frames.sort_by_key(|f| f.1);
    Ok(frames)
}

#[cfg(target_os = "macos")]
fn detect_transitions(
    frames: &[(PathBuf, i64)],
    _pixel_delta: u8,
    change_fraction: f64,
    _variance_threshold: f64,
) -> Result<Vec<(PathBuf, i64)>> {
    if frames.len() < 2 {
        return Ok(frames.to_vec());
    }

    let mut transition_frames: Vec<(PathBuf, i64)> = Vec::new();
    let mut prev_size = frames[0].0.metadata().map(|m| m.len()).unwrap_or(0);

    // Include first frame if it has content
    if prev_size > 1000 {
        transition_frames.push(frames[0].clone());
    }

    for frame in frames.iter().skip(1) {
        let path = &frame.0;
        let curr_size = path.metadata().map(|m| m.len()).unwrap_or(0);

        let size_ratio = if prev_size > 0 {
            (curr_size as f64 - prev_size as f64).abs() / prev_size as f64
        } else {
            0.0
        };

        if size_ratio > change_fraction * 10.0 {
            transition_frames.push(frame.clone());
        }

        prev_size = curr_size;
    }

    Ok(transition_frames)
}

// ── Dialogue line builder ─────────────────────────────────────────────────

fn build_dialogue_lines(frame_texts: &[(i64, String)]) -> Vec<mt_core::DialogueLine> {
    let mut lines: Vec<mt_core::DialogueLine> = Vec::new();
    let mut prev_text = String::new();
    let mut start_ms: i64 = 0;

    for (ts, text) in frame_texts {
        if *text != prev_text {
            if !prev_text.is_empty() && prev_text.len() > 1 {
                lines.push(mt_core::DialogueLine {
                    start_ms,
                    end_ms: *ts,
                    text: prev_text.clone(),
                });
            }
            start_ms = *ts;
            prev_text = text.clone();
        }
    }

    if !prev_text.is_empty() && prev_text.len() > 1 {
        let last_ts = frame_texts.last().map(|f| f.0).unwrap_or(start_ms);
        lines.push(mt_core::DialogueLine {
            start_ms,
            end_ms: last_ts + 1000,
            text: prev_text,
        });
    }

    lines
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn write_grayscale_pgm(path: &Path, pixels: &[u8], width: usize, height: usize) -> io::Result<()> {
    let mut data = Vec::new();
    data.extend_from_slice(b"P5\n");
    data.extend_from_slice(format!("{} {}\n255\n", width, height).as_bytes());
    data.extend_from_slice(pixels);
    fs::write(path, data)
}

fn find_ffmpeg() -> Result<PathBuf> {
    mt_core::exec::get_ffmpeg().map_err(|e| MtError::Subprocess {
        cmd: "ffmpeg".to_string(),
        code: None,
        stderr: e.to_string(),
    })
}

fn find_ffprobe() -> Result<PathBuf> {
    mt_core::exec::get_ffprobe().map_err(|e| MtError::Subprocess {
        cmd: "ffprobe".to_string(),
        code: None,
        stderr: e.to_string(),
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max..])
    }
}
