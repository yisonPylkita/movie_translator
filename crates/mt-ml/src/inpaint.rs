//! Burned-in subtitle removal via the embedded Python
//! `movie_translator.inpainting`.

use mt_core::{OCRResult, Result};
use std::path::{Path, PathBuf};

/// Remove burned-in subtitles from a video via inpainting.
///
/// # Errors
/// Returns [`mt_core::MtError::Parse`] (with Python traceback) on failure.
pub fn inpaint(
    video: &Path,
    output: &Path,
    device: &str,
    backend: &str,
    ocr_results: &[OCRResult],
) -> Result<PathBuf> {
    crate::backend::inpaint(video, output, device, backend, ocr_results)
}
