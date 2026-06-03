//! Hardsub-OCR helpers via the embedded `movie_translator.hardsub` package.
//!
//! `hardsub_download` is network/CPU only — call it off the GPU worker. The
//! OCR step (`hardsub_ocr_clean`) is a Vision/GPU job and must go through the
//! serialised GPU worker, like the other ML stages.

use std::path::{Path, PathBuf};

use mt_core::Result;

/// Download a player embed URL via yt-dlp.
///
/// `best = false` grabs the smallest copy whose height is still >= `min_height`
/// (OCR-legible, small); `best = true` grabs the highest-quality video+audio
/// and lets yt-dlp choose the container extension (the watch-it download path).
pub fn hardsub_download(
    embed_url: &str,
    out_path: &Path,
    min_height: u32,
    best: bool,
    referer: Option<&str>,
) -> Result<PathBuf> {
    crate::backend::hardsub_download(embed_url, out_path, min_height, best, referer)
}

/// OCR burned-in subs from a downloaded video and clean them into a `.srt`.
/// Returns `None` when OCR yields no usable lines.
pub fn hardsub_ocr_clean(video: &Path, out_dir: &Path, language: &str) -> Result<Option<PathBuf>> {
    crate::backend::hardsub_ocr_clean(video, out_dir, language)
}
