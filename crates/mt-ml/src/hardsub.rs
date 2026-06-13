//! Hardsub download via yt-dlp subprocess (no Python).
//!
//! `hardsub_download` calls yt-dlp as an external subprocess (like FFmpeg),
//! handling format selection for OCR-legible or best-quality copies.
//!
//! `hardsub_ocr_clean` is a Vision/GPU job — it OCRs burned-in subs from a
//! downloaded video using the Rust-native OCR pipeline.

use std::path::{Path, PathBuf};
use std::process::Command;

use mt_core::{MtError, Result};
#[cfg_attr(target_os = "macos", allow(unused_imports))]
use tracing::{info, warn};

/// Default minimum height for OCR-legible download (480p).
const DEFAULT_MIN_HEIGHT: u32 = 480;

/// Build yt-dlp format selector for OCR mode.
fn build_format_selector(min_height: u32) -> String {
    format!(
        "bv*[height>={h}]+ba/b[height>={h}]/bv*+ba/b",
        h = min_height
    )
}

/// Download a player embed URL via yt-dlp.
///
/// `best = false` (OCR mode): grabs the smallest copy whose height is still >=
/// `min_height`. `best = true`: grabs the highest-quality video+audio and lets
/// yt-dlp choose the container extension.
///
/// Returns the path to the downloaded file.
pub fn hardsub_download(
    embed_url: &str,
    out_path: &Path,
    min_height: u32,
    best: bool,
    referer: Option<&str>,
) -> Result<PathBuf> {
    let min_h = if min_height == 0 {
        DEFAULT_MIN_HEIGHT
    } else {
        min_height
    };

    // Create output directory
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(MtError::Io)?;
    }

    let out_str = out_path.to_string_lossy().to_string();

    let mut cmd = Command::new("yt-dlp");
    cmd.arg("--quiet")
        .arg("--no-warnings")
        .arg("--no-progress")
        .arg("--overwrites");

    if best {
        // Best quality mode (for anime-dl / watch-it)
        let stem = out_path.with_extension("");
        let stem_str = stem.to_string_lossy().to_string();
        cmd.arg("-f").arg("bv*+ba/b");
        cmd.arg("-o").arg(format!("{stem_str}.%(ext)s"));
        cmd.arg("--merge-output-format").arg("mkv");
    } else {
        // OCR mode: lowest resolution >= min_height
        cmd.arg("-f").arg(build_format_selector(min_h));
        cmd.arg("-o").arg(&out_str);
        cmd.arg("--format-sort").arg("+size,+res");
    }

    if let Some(r) = referer {
        cmd.arg("--add-header").arg(format!("Referer: {}", r));
        cmd.arg("--user-agent")
            .arg("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36");
    }

    cmd.arg(embed_url);

    info!(
        "yt-dlp download: {} -> {} (best={best})",
        embed_url,
        out_path.display()
    );

    let output = cmd.output().map_err(MtError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let truncated = truncate(&stderr, 500);
        return Err(MtError::Subprocess {
            cmd: "yt-dlp".to_string(),
            code: output.status.code(),
            stderr: truncated,
        });
    }

    // Find the actual written file
    let written = if best {
        // Best mode uses %(ext)s, find the real file
        let stem = out_path.with_extension("");
        let candidates: Vec<_> = std::fs::read_dir(stem.parent().unwrap_or(Path::new(".")))
            .map_err(MtError::Io)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_stem() == stem.file_stem()
                    && p.is_file()
                    && p.metadata().is_ok_and(|m| m.len() > 0)
            })
            .collect();
        candidates
            .into_iter()
            .max_by_key(|p| p.metadata().map(|m| m.modified().ok()).ok().flatten())
            .unwrap_or_else(|| out_path.to_path_buf())
    } else {
        if out_path.exists() && out_path.metadata().is_ok_and(|m| m.len() > 0) {
            out_path.to_path_buf()
        } else {
            // yt-dlp may have added an extension
            let candidates: Vec<_> = std::fs::read_dir(out_path.parent().unwrap_or(Path::new(".")))
                .map_err(MtError::Io)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    let expected = out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    stem == expected && p.is_file() && p.metadata().is_ok_and(|m| m.len() > 0)
                })
                .collect();
            candidates
                .into_iter()
                .max_by_key(|p| p.metadata().map(|m| m.modified().ok()).ok().flatten())
                .ok_or_else(|| MtError::Subprocess {
                    cmd: "yt-dlp".to_string(),
                    code: None,
                    stderr: "yt-dlp produced no output file".to_string(),
                })?
        }
    };

    info!(
        "yt-dlp download complete: {} ({} bytes)",
        written.display(),
        written.metadata().map(|m| m.len()).unwrap_or(0)
    );

    Ok(written)
}

/// OCR burned-in subs from a downloaded video and clean them into a `.srt`.
///
/// Calls the Rust-native OCR pipeline (on macOS via Vision framework) or
/// returns `None` on unsupported platforms. No Python involved.
pub fn hardsub_ocr_clean(video: &Path, out_dir: &Path, language: &str) -> Result<Option<PathBuf>> {
    #[cfg(target_os = "macos")]
    {
        use ocr_postprocess::{postprocess_ocr_results, render_to_srt};

        let _ = language; // Vision OCR uses en by default; language support TBD in Rust port
        let result = crate::ocr::ocr_burned_in(video, out_dir, 0.25, 6)?;

        if result.ocr_results.is_empty() {
            info!("No burned-in OCR results for {}", video.display());
            return Ok(None);
        }

        let frame_texts: Vec<_> = result
            .ocr_results
            .iter()
            .map(|r| (r.timestamp_ms, r.text.clone()))
            .collect();

        let clean = postprocess_ocr_results(&frame_texts);
        if clean.is_empty() {
            info!("No usable dialogue after cleanup for {}", video.display());
            return Ok(None);
        }

        std::fs::create_dir_all(out_dir).map_err(MtError::Io)?;
        let srt_path = out_dir.join(format!(
            "{}.pl.cleaned.srt",
            video
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output")
        ));

        let srt_content = render_to_srt(&clean);
        std::fs::write(&srt_path, &srt_content).map_err(MtError::Io)?;

        info!(
            "Wrote cleaned hardsub srt: {} ({} lines)",
            srt_path.display(),
            clean.len()
        );
        Ok(Some(srt_path))
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (language, video, out_dir);
        warn!("hardsub_ocr_clean requires macOS (Vision framework)");
        Ok(None)
    }
}

#[cfg(target_os = "macos")]
mod ocr_postprocess {
    use std::collections::HashMap;

    /// A single cleaned dialogue line.
    #[derive(Debug, Clone)]
    pub(super) struct CleanLine {
        pub(super) start_ms: i64,
        pub(super) end_ms: i64,
        pub(super) text: String,
    }

    const SIMILARITY: f64 = 0.80;
    const TAIL_MS: i64 = 800;
    const MIN_DURATION_MS: i64 = 200;
    const MIN_LETTERS: usize = 3;
    const MIN_ALPHA_RATIO: f64 = 0.5;

    fn norm(text: &str) -> String {
        text.to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn similar(a: &str, b: &str) -> f64 {
        let na = norm(a);
        let nb = norm(b);
        if na == nb {
            return 1.0;
        }
        let max_len = na.len().max(nb.len());
        if max_len == 0 {
            return 1.0;
        }
        // Simple character-level similarity (edit-distance based approximation)
        let common: usize = na.chars().filter(|c| nb.contains(*c)).count();
        common as f64 / max_len as f64
    }

    fn is_dialogue(text: &str) -> bool {
        let flat = text.replace('\n', " ").trim().to_string();
        if flat.len() < 2 {
            return false;
        }
        let letters = flat.chars().filter(|c| c.is_alphabetic()).count();
        let non_space = flat.chars().filter(|c| !c.is_whitespace()).count();
        if letters < MIN_LETTERS {
            return false;
        }
        if non_space > 0 && (letters as f64 / non_space as f64) < MIN_ALPHA_RATIO {
            return false;
        }
        true
    }

    fn best_variant(variants: &[String]) -> String {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for v in variants {
            *counts.entry(v.as_str()).or_insert(0) += 1;
        }
        let top_count = counts.values().max().copied().unwrap_or(0);
        let tied: Vec<_> = variants
            .iter()
            .filter(|v| counts.get(v.as_str()).copied().unwrap_or(0) == top_count)
            .collect();
        tied.iter()
            .max_by_key(|v| v.len())
            .map(|s| (*s).clone())
            .unwrap_or_default()
    }

    pub(super) fn postprocess_ocr_results(frame_texts: &[(i64, String)]) -> Vec<CleanLine> {
        let mut frames: Vec<&(i64, String)> = frame_texts.iter().collect();
        frames.sort_by_key(|f| f.0);

        let mut lines: Vec<CleanLine> = Vec::new();
        let mut anchor: Option<String> = None;
        let mut variants: Vec<String> = Vec::new();
        let mut start_ms: i64 = 0;

        for (ts, raw) in &frames {
            let text = raw.trim().to_string();
            if let Some(ref a) = anchor
                && !text.is_empty()
                && similar(&text, a) >= SIMILARITY
            {
                variants.push(text);
                continue;
            }
            // Close the running group
            if anchor.is_some() {
                let t = best_variant(&variants);
                if *ts - start_ms >= MIN_DURATION_MS && is_dialogue(&t) {
                    lines.push(CleanLine {
                        start_ms,
                        end_ms: *ts,
                        text: t,
                    });
                }
            }
            if text.is_empty() {
                anchor = None;
                variants.clear();
            } else {
                anchor = Some(text.clone());
                variants = vec![text];
                start_ms = *ts;
            }
        }

        if anchor.is_some() {
            let last_ts = frames.last().map(|f| f.0).unwrap_or(start_ms);
            let t = best_variant(&variants);
            if last_ts + TAIL_MS - start_ms >= MIN_DURATION_MS && is_dialogue(&t) {
                lines.push(CleanLine {
                    start_ms,
                    end_ms: last_ts + TAIL_MS,
                    text: t,
                });
            }
        }

        lines
    }

    fn fmt_ts(ms: i64) -> String {
        let h = ms / 3_600_000;
        let rem = ms % 3_600_000;
        let m = rem / 60_000;
        let rem = rem % 60_000;
        let s = rem / 1000;
        let ms_rem = rem % 1000;
        format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms_rem)
    }

    pub(super) fn render_to_srt(lines: &[CleanLine]) -> String {
        let mut parts = Vec::new();
        for (i, ln) in lines.iter().enumerate() {
            parts.push(format!(
                "{}\n{} --> {}\n{}\n",
                i + 1,
                fmt_ts(ln.start_ms),
                fmt_ts(ln.end_ms),
                ln.text
            ));
        }
        parts.join("\n")
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max..])
    }
}
