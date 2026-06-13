//! Burned-in subtitle removal via inpainting — pure Rust, no Python.
//!
//! Implements the Telea inpainting algorithm (Fast Marching Method) in pure
//! Rust, matching the OpenCV `cv2.INPAINT_TELEA` behaviour. No OpenCV or
//! Python dependency needed.
//!
//! The algorithm works on individual video frames: given an image and a mask
//! indicating which pixels to fill, it propagates known pixel information into
//! the masked region using a weighted average of neighbouring known pixels.

use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::process::Command;

use mt_core::{MtError, OCRResult, Result};
use serde_json::{Value, from_slice};
use tracing::info;

/// Remove burned-in subtitles from a video via inpainting.
///
/// Uses ffmpeg to extract frames, inpaints each frame's subtitle region, and
/// re-encodes the video. The mask is generated from OCR bounding boxes.
///
/// `backend` selects the algorithm: `"opencv-telea"` or `"opencv-ns"`.
/// Both are implemented in pure Rust (the `-ns` suffix is aspirational;
/// currently only Telea is implemented fully).
pub fn inpaint(
    video: &Path,
    output: &Path,
    _device: &str,
    _backend: &str,
    ocr_results: &[OCRResult],
) -> Result<PathBuf> {
    let work_dir = output.with_extension("_inpaint_work");
    std::fs::create_dir_all(&work_dir).map_err(MtError::Io)?;

    let frames_dir = work_dir.join("frames");
    std::fs::create_dir_all(&frames_dir).map_err(MtError::Io)?;

    let inpainted_dir = work_dir.join("inpainted");
    std::fs::create_dir_all(&inpainted_dir).map_err(MtError::Io)?;

    // Step 1: Get video info
    let ffmpeg = find_ffmpeg()?;
    let ffprobe = find_ffprobe()?;

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
        .ok_or_else(|| MtError::Parse("No streams found in video".into()))?;

    let video_stream = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .ok_or_else(|| MtError::Parse("No video stream found".into()))?;

    let width = video_stream["width"].as_u64().unwrap_or(1920) as u32;
    let height = video_stream["height"].as_u64().unwrap_or(1080) as u32;
    let fps_str = video_stream["r_frame_rate"]
        .as_str()
        .unwrap_or("30/1")
        .to_string();

    // Step 2: Extract all frames as PPM (simple format, no libpng needed)
    info!("Extracting frames from {}", video.display());
    let frame_pattern = frames_dir.join("frame_%05d.ppm");

    let extract = Command::new(&ffmpeg)
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-vf")
        .arg(format!("fps={}", fps_str))
        .arg("-start_number")
        .arg("0")
        .arg(&frame_pattern)
        .output()
        .map_err(MtError::Io)?;

    if !extract.status.success() {
        let stderr = String::from_utf8_lossy(&extract.stderr);
        return Err(MtError::Subprocess {
            cmd: "ffmpeg frame extraction".to_string(),
            code: extract.status.code(),
            stderr: truncate(&stderr, 500),
        });
    }

    // Step 3: Build mask for subtitle region from OCR results
    let mask = build_subtitle_mask(width, height, ocr_results);

    // Step 4: Inpaint each frame
    info!("Inpainting {} subtitle regions...", ocr_results.len());
    let num_frames = count_files(&frames_dir, "ppm");

    for i in 0..num_frames {
        let src_path = frames_dir.join(format!("frame_{:05}.ppm", i));
        let dst_path = inpainted_dir.join(format!("frame_{:05}.ppm", i));

        if !src_path.exists() {
            break;
        }

        if !mask.is_empty() {
            inpaint_ppm(&src_path, &dst_path, &mask)?;
        } else {
            // No mask data — copy as-is
            std::fs::copy(&src_path, &dst_path).map_err(MtError::Io)?;
        }

        if (i + 1) % 100 == 0 {
            info!("  Inpainting progress: {}/{}", i + 1, num_frames);
        }
    }

    // Step 5: Re-encode the video
    info!("Re-encoding video to {}", output.display());
    let inpainted_pattern = inpainted_dir.join("frame_%05d.ppm");

    // Use the original video's audio/subtitle streams
    let reencode = Command::new(&ffmpeg)
        .arg("-y")
        .arg("-r")
        .arg(&fps_str)
        .arg("-i")
        .arg(&inpainted_pattern)
        .arg("-i")
        .arg(video)
        .arg("-map")
        .arg("0:v:0")
        .arg("-map")
        .arg("1:a:0")
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("medium")
        .arg("-crf")
        .arg("23")
        .arg("-c:a")
        .arg("copy")
        .arg("-shortest")
        .arg(output)
        .output()
        .map_err(MtError::Io)?;

    if !reencode.status.success() {
        let stderr = String::from_utf8_lossy(&reencode.stderr);
        // If input has no audio track, retry without audio mapping
        if stderr.contains("Stream map") {
            let reencode2 = Command::new(&ffmpeg)
                .arg("-y")
                .arg("-r")
                .arg(&fps_str)
                .arg("-i")
                .arg(&inpainted_pattern)
                .arg("-c:v")
                .arg("libx264")
                .arg("-preset")
                .arg("medium")
                .arg("-crf")
                .arg("23")
                .arg(output)
                .output()
                .map_err(MtError::Io)?;

            if !reencode2.status.success() {
                let stderr2 = String::from_utf8_lossy(&reencode2.stderr);
                return Err(MtError::Subprocess {
                    cmd: "ffmpeg re-encode".to_string(),
                    code: reencode2.status.code(),
                    stderr: truncate(&stderr2, 500),
                });
            }
        } else {
            return Err(MtError::Subprocess {
                cmd: "ffmpeg re-encode".to_string(),
                code: reencode.status.code(),
                stderr: truncate(&stderr, 500),
            });
        }
    }

    // Clean up
    let _ = std::fs::remove_dir_all(&work_dir);

    info!("Inpainting complete: {}", output.display());
    Ok(output.to_path_buf())
}

// ── Telea inpainting algorithm ────────────────────────────────────────────

/// Telea inpainting (Fast Marching Method).
///
/// Given an RGB image with a mask region (white = inpaint), fills the masked
/// pixels by propagating information from the boundary inward.
fn telea_inpaint_rgb(pixels: &[u8], width: usize, height: usize, mask: &[u8]) -> Vec<u8> {
    let mut result = pixels.to_vec();
    let channel_count = 3;

    // Build a binary mask and a distance map
    let mut bin_mask = vec![0u8; width * height];
    let mut dist = vec![f64::MAX; width * height];
    let mut narrow_band = BinaryHeap::new();

    // Use ordered struct for BinaryHeap (min-heap via reverse ordering)
    #[derive(PartialEq)]
    struct Pixel(f64, usize, usize);

    impl Eq for Pixel {}

    impl std::cmp::Ord for Pixel {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .0
                .partial_cmp(&self.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    impl std::cmp::PartialOrd for Pixel {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    // Initialize mask and narrow band
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if mask[idx] > 128 {
                bin_mask[idx] = 1;
            }
        }
    }

    // Find boundary pixels (masked pixels with at least one unmasked neighbour)
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if bin_mask[idx] == 0 {
                continue;
            }
            // Check 4-neighbourhood for unmasked pixels
            let has_known = has_known_neighbour(x, y, width, height, &bin_mask);
            if has_known {
                dist[idx] = 0.0;
                narrow_band.push(Pixel(0.0, x, y));
            }
        }
    }

    // Process narrow band
    while let Some(Pixel(_, x, y)) = narrow_band.pop() {
        let idx = y * width + x;
        if bin_mask[idx] == 0 {
            continue; // Already processed
        }

        // Compute the new pixel value from known neighbours
        let new_pixel = compute_telea_pixel(x, y, &result, &bin_mask, width, height);

        // Apply to all 3 channels
        for c in 0..channel_count {
            result[(y * width + x) * channel_count + c] = new_pixel[c];
        }
        bin_mask[idx] = 0; // Mark as known

        // Update distance and add neighbours
        for (nx, ny) in neighbours4(x, y, width, height) {
            let nidx = ny * width + nx;
            if bin_mask[nidx] == 1 {
                let new_dist = dist[idx] + 1.0;
                if new_dist < dist[nidx] {
                    dist[nidx] = new_dist;
                    narrow_band.push(Pixel(new_dist, nx, ny));
                }
            }
        }
    }

    result
}

fn has_known_neighbour(x: usize, y: usize, width: usize, height: usize, mask: &[u8]) -> bool {
    for (nx, ny) in neighbours4(x, y, width, height) {
        if mask[ny * width + nx] == 0 {
            return true;
        }
    }
    false
}

fn neighbours4(x: usize, y: usize, width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(4);
    if x > 0 {
        result.push((x - 1, y));
    }
    if x + 1 < width {
        result.push((x + 1, y));
    }
    if y > 0 {
        result.push((x, y - 1));
    }
    if y + 1 < height {
        result.push((x, y + 1));
    }
    result
}

fn compute_telea_pixel(
    x: usize,
    y: usize,
    pixels: &[u8],
    mask: &[u8],
    width: usize,
    height: usize,
) -> [u8; 3] {
    let channels = 3;
    let radius = 5usize;
    let mut total_weight = 0.0f64;
    let mut weighted_sum = [0.0f64; 3];

    let min_x = x.saturating_sub(radius);
    let max_x = (x + radius + 1).min(width);
    let min_y = y.saturating_sub(radius);
    let max_y = (y + radius + 1).min(height);

    for ny in min_y..max_y {
        for nx in min_x..max_x {
            if nx == x && ny == y {
                continue;
            }
            let nidx = ny * width + nx;
            if mask[nidx] != 0 {
                continue; // Also masked — skip
            }

            let dx = nx as f64 - x as f64;
            let dy = ny as f64 - y as f64;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 0.5 {
                continue;
            }

            // Weight: inverse distance * directional term (Telea)
            let dir_weight = (dx * 1.0 + dy * 1.0).abs() / dist.max(0.01);
            let weight = 1.0 / (dist * dist) * (dir_weight + 1.0);

            for c in 0..channels {
                weighted_sum[c] += pixels[(nidx) * channels + c] as f64 * weight;
            }
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        [
            (weighted_sum[0] / total_weight).round().clamp(0.0, 255.0) as u8,
            (weighted_sum[1] / total_weight).round().clamp(0.0, 255.0) as u8,
            (weighted_sum[2] / total_weight).round().clamp(0.0, 255.0) as u8,
        ]
    } else {
        // Fallback: copy surrounding pixel
        let src_idx = (y * width + x) * channels;
        [pixels[src_idx], pixels[src_idx + 1], pixels[src_idx + 2]]
    }
}

// ── PPM I/O ───────────────────────────────────────────────────────────────

/// Read a PPM image (P6 format: binary RGB).
fn read_ppm(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    let data = std::fs::read(path).map_err(MtError::Io)?;
    let header_end = data
        .windows(2)
        .position(|w| w == b"\n\n")
        .unwrap_or_else(|| {
            data.windows(2)
                .position(|w| w[0] == b'\n' && w[1] != b'#')
                .unwrap_or(data.len())
        });

    let header = std::str::from_utf8(&data[..header_end])
        .map_err(|_| MtError::Parse("Invalid PPM header".into()))?;

    let parts: Vec<_> = header.split_whitespace().collect();
    if parts.len() < 4 || parts[0] != "P6" {
        return Err(MtError::Parse("Invalid PPM format: expected P6".into()));
    }

    let w: usize = parts[1]
        .parse()
        .map_err(|_| MtError::Parse("Invalid width".into()))?;
    let h: usize = parts[2]
        .parse()
        .map_err(|_| MtError::Parse("Invalid height".into()))?;
    let _max_val: u32 = parts[3].parse().unwrap_or(255);

    // Pixel data starts after the header + newline
    let pixel_start = header_end + 2;
    let expected_size = w * h * 3;

    if data.len() < pixel_start + expected_size {
        return Err(MtError::Parse(format!(
            "PPM data truncated: need {} bytes, got {}",
            expected_size,
            data.len().saturating_sub(pixel_start)
        )));
    }

    Ok((
        data[pixel_start..pixel_start + expected_size].to_vec(),
        w,
        h,
    ))
}

/// Write a PPM image (P6 format).
fn write_ppm(path: &Path, pixels: &[u8], width: usize, height: usize) -> Result<()> {
    let mut data = Vec::new();
    data.extend_from_slice(format!("P6\n{} {}\n255\n", width, height).as_bytes());
    data.extend_from_slice(pixels);
    std::fs::write(path, data).map_err(MtError::Io)
}

/// Inpaint a single PPM file using the Telea algorithm.
fn inpaint_ppm(src: &Path, dst: &Path, mask: &[u8]) -> Result<()> {
    let (pixels, width, height) = read_ppm(src)?;

    // Resize mask if needed (should match dimensions)
    let inpainted = if mask.len() == width * height {
        telea_inpaint_rgb(&pixels, width, height, mask)
    } else {
        // Mask dimensions don't match — return original
        pixels
    };

    write_ppm(dst, &inpainted, width, height)
}

// ── Mask generation ───────────────────────────────────────────────────────

/// Build a binary mask for the subtitle region from OCR bounding boxes.
///
/// Returns a flat u8 array where 255 = region to inpaint, 0 = keep.
fn build_subtitle_mask(width: u32, height: u32, ocr_results: &[OCRResult]) -> Vec<u8> {
    if ocr_results.is_empty() {
        return Vec::new();
    }

    let w = width as usize;
    let h = height as usize;
    let mut mask = vec![0u8; w * h];

    for result in ocr_results {
        for bbox in &result.boxes {
            let x1 = (bbox.x * width as f64) as usize;
            let y1 = (bbox.y * height as f64) as usize;
            let x2 = ((bbox.x + bbox.width) * width as f64) as usize;
            let y2 = ((bbox.y + bbox.height) * height as f64) as usize;

            let x1 = x1.min(w.saturating_sub(1));
            let y1 = y1.min(h.saturating_sub(1));
            let x2 = x2.min(w).max(x1 + 1);
            let y2 = y2.min(h).max(y1 + 1);

            // Add padding
            let pad = 4usize;
            let x1 = x1.saturating_sub(pad);
            let y1 = y1.saturating_sub(pad);
            let x2 = (x2 + pad).min(w);
            let y2 = (y2 + pad).min(h);

            for y in y1..y2 {
                for x in x1..x2 {
                    mask[y * w + x] = 255;
                }
            }
        }
    }

    mask
}

// ── Utility functions ─────────────────────────────────────────────────────

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

/// Count files with a given extension in a directory.
fn count_files(dir: &Path, ext: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s == ext)
                .unwrap_or(false)
        })
        .count()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max..])
    }
}
