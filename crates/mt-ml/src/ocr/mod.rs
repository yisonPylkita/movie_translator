//! Subtitle OCR — dispatcher module.
//!
//! On macOS, delegates to the Apple Vision backend (`vision`).
//! On other platforms, returns errors indicating OCR is unavailable.

use std::path::{Path, PathBuf};

use mt_core::{BurnedInResult, Result};

/// Apple Vision backend (macOS only).
#[cfg(target_os = "macos")]
pub mod vision;

/// Extract a PGS subtitle track to SRT using Apple Vision OCR.
pub fn ocr_pgs(video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
    #[cfg(target_os = "macos")]
    {
        vision::ocr_pgs_macos(video, track_index, work_dir)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (video, track_index, work_dir);
        tracing::warn!("PGS OCR requires macOS (Vision framework)");
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
        vision::ocr_burned_in_macos(video, output_dir, crop_ratio, fps, "en")
    }
    #[cfg(not(target_os = "macos"))]
    {
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
        vision::is_vision_ocr_available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
