//! Vision OCR availability probe.
//!
//! Mirrors `movie_translator/ocr/vision_ocr.py::is_available`, which returns
//! `True` only on macOS where the `Quartz`/`Vision` Python bindings import
//! cleanly. The actual OCR runs in Python (via the `mt_ml` helper scripts), so
//! here we only need the *availability* signal used by the extract stages to
//! decide whether to defer a burned-in OCR pass.
//!
//! The check is injected into the extract stages as a closure so tests can
//! force it on/off without touching the host platform (matching how the Python
//! tests `patch('...is_vision_ocr_available')`).

/// A predicate that reports whether Vision-based OCR is available.
pub type VisionOcrProbe = fn() -> bool;

/// Default probe: best-effort port of the Python check.
///
/// We cannot import the macOS `Vision` framework from Rust, so we approximate:
/// the binding is *potentially* available only on macOS. The orchestrator may
/// substitute a more precise probe that shells out to the Python helper.
pub fn default_vision_ocr_probe() -> bool {
    cfg!(target_os = "macos")
}
