//! Subtitle OCR via the Python `ml/ocr.py` helper script.
//!
//! Mirrors the `OcrTask` GPU task in `movie_translator/gpu_queue.py`, which
//! has two modes: PGS bitmap tracks and burned-in subtitles.

use crate::runner::run_script_json;
use mt_core::{BurnedInResult, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Response shape for `--type pgs`: `extract_pgs_track` returns a path or None.
#[derive(Debug, Deserialize)]
struct PgsResponse {
    srt_path: Option<PathBuf>,
}

/// Extract a PGS subtitle track to SRT (PGS bitmap OCR).
///
/// Returns the path to the generated SRT, or `None` when extraction failed
/// (mirrors `extract_pgs_track` returning `Path | None`).
///
/// # Errors
/// Propagates [`mt_core::MtError::Subprocess`] (with stderr) on script failure.
pub fn ocr_pgs(video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
    ocr_pgs_with_args(video, track_index, work_dir, &[])
}

pub(crate) fn ocr_pgs_with_args(
    video: &Path,
    track_index: u32,
    work_dir: &Path,
    extra: &[&str],
) -> Result<Option<PathBuf>> {
    let track = track_index.to_string();
    let mut args: Vec<&str> = vec![
        "--type",
        "pgs",
        "--video",
        video.to_str().unwrap_or_default(),
        "--track-index",
        &track,
        "--work-dir",
        work_dir.to_str().unwrap_or_default(),
    ];
    args.extend_from_slice(extra);

    let resp: PgsResponse = run_script_json::<(), _>("ocr.py", &args, None)?;
    Ok(resp.srt_path)
}

/// Extract burned-in subtitles via OCR.
///
/// Returns a [`BurnedInResult`] (SRT path + per-frame OCR boxes), mirroring
/// `extract_burned_in_subtitles`. Note: the Python function can return `None`,
/// in which case the script exits non-zero and this surfaces as an error.
///
/// # Errors
/// Propagates [`mt_core::MtError::Subprocess`] (with stderr) on script failure
/// or when extraction produced no result.
pub fn ocr_burned_in(
    video: &Path,
    output_dir: &Path,
    crop_ratio: f64,
    fps: u32,
) -> Result<BurnedInResult> {
    ocr_burned_in_with_args(video, output_dir, crop_ratio, fps, &[])
}

pub(crate) fn ocr_burned_in_with_args(
    video: &Path,
    output_dir: &Path,
    crop_ratio: f64,
    fps: u32,
    extra: &[&str],
) -> Result<BurnedInResult> {
    let crop = crop_ratio.to_string();
    let fps_s = fps.to_string();
    let mut args: Vec<&str> = vec![
        "--type",
        "burned_in",
        "--video",
        video.to_str().unwrap_or_default(),
        "--output-dir",
        output_dir.to_str().unwrap_or_default(),
        "--crop-ratio",
        &crop,
        "--fps",
        &fps_s,
    ];
    args.extend_from_slice(extra);

    run_script_json::<(), _>("ocr.py", &args, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::BurnedInResult;

    /// Unit test: burned-in response deserialises from the script's shape.
    #[test]
    fn burned_in_response_deserialize_shape() {
        let json = r#"{
            "srt_path": "/tmp/foo.srt",
            "ocr_results": [
                {"timestamp_ms": 1000, "text": "hi",
                 "boxes": [{"x": 0.1, "y": 0.8, "width": 0.5, "height": 0.1}]}
            ]
        }"#;
        let resp: BurnedInResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.srt_path.to_str(), Some("/tmp/foo.srt"));
        assert_eq!(resp.ocr_results.len(), 1);
        assert_eq!(resp.ocr_results[0].text, "hi");
        assert_eq!(resp.ocr_results[0].boxes[0].width, 0.5);
    }

    /// Integration test: `ocr_pgs` via `ml/ocr.py --self-test` (no models).
    /// Not `#[ignore]` — runs in CI.
    #[test]
    fn ocr_pgs_self_test_via_public_path() {
        let srt = ocr_pgs_with_args(
            Path::new("/tmp/foo.mkv"),
            2,
            Path::new("/tmp/wd"),
            &["--self-test"],
        )
        .expect("self-test pgs");
        assert_eq!(srt.as_deref(), Some(Path::new("/tmp/foo.mkv.srt")));
    }

    /// Integration test: `ocr_burned_in` via `ml/ocr.py --self-test`.
    /// Not `#[ignore]` — runs in CI.
    #[test]
    fn ocr_burned_in_self_test_via_public_path() {
        let result = ocr_burned_in_with_args(
            Path::new("/tmp/foo.mkv"),
            Path::new("/tmp/out"),
            0.3,
            2,
            &["--self-test"],
        )
        .expect("self-test burned_in");
        assert_eq!(result.srt_path.to_str(), Some("/tmp/foo.mkv.srt"));
        assert_eq!(result.ocr_results.len(), 1);
        assert_eq!(result.ocr_results[0].text, "self-test subtitle");
    }
}
