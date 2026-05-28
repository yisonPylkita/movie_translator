//! Subtitle OCR via the embedded Python `movie_translator.ocr`.

use mt_core::{BurnedInResult, Result};
use std::path::{Path, PathBuf};

/// Extract a PGS subtitle track to SRT (PGS bitmap OCR).
///
/// Returns the path to the generated SRT, or `None` when extraction failed.
///
/// # Errors
/// Returns [`mt_core::MtError::Parse`] (with Python traceback) on failure.
pub fn ocr_pgs(video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
    crate::backend::ocr_pgs(video, track_index, work_dir)
}

/// Extract burned-in subtitles via OCR.
///
/// # Errors
/// Returns [`mt_core::MtError::Parse`] (with Python traceback) on failure
/// or when extraction produced no result.
pub fn ocr_burned_in(
    video: &Path,
    output_dir: &Path,
    crop_ratio: f64,
    fps: u32,
) -> Result<BurnedInResult> {
    crate::backend::ocr_burned_in(video, output_dir, crop_ratio, fps)
}
