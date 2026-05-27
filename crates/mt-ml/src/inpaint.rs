//! Burned-in subtitle removal via the Python `ml/inpaint.py` helper script.
//!
//! Mirrors the `InpaintTask` GPU task in `movie_translator/gpu_queue.py`.

use crate::runner::run_script_json;
use mt_core::{MtError, OCRResult, Result};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Response shape: the script echoes the output path on success.
#[derive(Debug, Deserialize)]
struct InpaintResponse {
    output_path: PathBuf,
}

/// Remove burned-in subtitles from a video via inpainting.
///
/// The OCR results are written to a temporary JSON file whose path is passed
/// to the script (mirroring how the Python side reads boxes per frame).
/// Returns the output path on success.
///
/// # Errors
/// Propagates [`mt_core::MtError::Subprocess`] (with stderr) on script failure,
/// or [`mt_core::MtError::Io`] / [`mt_core::MtError::Parse`] if the temp file
/// can't be written.
pub fn inpaint(
    video: &Path,
    output: &Path,
    device: &str,
    backend: &str,
    ocr_results: &[OCRResult],
) -> Result<PathBuf> {
    inpaint_with_args(video, output, device, backend, ocr_results, &[])
}

pub(crate) fn inpaint_with_args(
    video: &Path,
    output: &Path,
    device: &str,
    backend: &str,
    ocr_results: &[OCRResult],
    extra: &[&str],
) -> Result<PathBuf> {
    // Write OCR results to a temp JSON file and pass its path to the script.
    let mut temp = tempfile::Builder::new()
        .prefix("mt-ocr-results-")
        .suffix(".json")
        .tempfile()
        .map_err(MtError::Io)?;
    let payload = serde_json::to_vec(ocr_results)
        .map_err(|e| MtError::Parse(format!("failed to serialize ocr_results: {e}")))?;
    temp.write_all(&payload).map_err(MtError::Io)?;
    temp.flush().map_err(MtError::Io)?;

    let ocr_path = temp.path().to_str().unwrap_or_default().to_string();
    let mut args: Vec<&str> = vec![
        "--video",
        video.to_str().unwrap_or_default(),
        "--output",
        output.to_str().unwrap_or_default(),
        "--device",
        device,
        "--backend",
        backend,
        "--ocr-results",
        &ocr_path,
    ];
    args.extend_from_slice(extra);

    let resp: InpaintResponse = run_script_json::<(), _>("inpaint.py", &args, None)?;
    Ok(resp.output_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit test: response deserialises from the script's stdout shape.
    #[test]
    fn inpaint_response_deserialize_shape() {
        let json = r#"{"output_path": "/tmp/out.mp4"}"#;
        let resp: InpaintResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.output_path.to_str(), Some("/tmp/out.mp4"));
    }

    /// Integration test: `inpaint` via `ml/inpaint.py --self-test`, which
    /// copies input -> output without torch. Exercises temp-file write +
    /// spawn + JSON. Not `#[ignore]` — runs in CI.
    #[test]
    fn inpaint_self_test_via_public_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let video = dir.path().join("in.mp4");
        let output = dir.path().join("out.mp4");
        std::fs::write(&video, b"dummy video bytes").expect("write video");

        let ocr_results = vec![OCRResult {
            timestamp_ms: 1000,
            text: "x".to_string(),
            boxes: vec![],
        }];

        let result = inpaint_with_args(
            &video,
            &output,
            "cpu",
            "lama",
            &ocr_results,
            &["--self-test"],
        )
        .expect("self-test inpaint");

        assert_eq!(result, output);
        assert_eq!(std::fs::read(&output).expect("read output"), b"dummy video bytes");
    }
}
