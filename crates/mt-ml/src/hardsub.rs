//! Hardsub-OCR helpers via the embedded `movie_translator.hardsub` package.
//!
//! `hardsub_download` is network/CPU only — call it off the GPU worker. The
//! OCR step (`hardsub_ocr_clean`) is a Vision/GPU job and must go through the
//! serialised GPU worker, like the other ML stages.

use std::path::{Path, PathBuf};

use mt_core::Result;

/// Download the lowest OCR-legible copy of a player embed URL via yt-dlp.
pub fn hardsub_download(
    embed_url: &str,
    out_path: &Path,
    min_height: u32,
    referer: Option<&str>,
) -> Result<PathBuf> {
    crate::backend::hardsub_download(embed_url, out_path, min_height, referer)
}

/// OCR burned-in subs from a downloaded video and clean them into a `.srt`.
/// Returns `None` when OCR yields no usable lines.
pub fn hardsub_ocr_clean(video: &Path, out_dir: &Path, language: &str) -> Result<Option<PathBuf>> {
    crate::backend::hardsub_ocr_clean(video, out_dir, language)
}
