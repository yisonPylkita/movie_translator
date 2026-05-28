//! Vision OCR availability probe.
//!
//! Reports `true` only on macOS where the `Quartz`/`Vision` Python bindings
//! import cleanly inside the embedded interpreter. The actual OCR runs via
//! `mt_ml`; here we only need the *availability* signal used by the extract
//! stages to decide whether to defer a burned-in OCR pass.
//!
//! The check is injected into the extract stages as a closure so tests can
//! force it on/off without touching the host platform.

/// A predicate that reports whether Vision-based OCR is available.
pub type VisionOcrProbe = fn() -> bool;

/// Default probe: a precise, cached availability check via the embedded
/// CPython interpreter (no subprocess). On non-macOS this is trivially
/// `false` without booting the interpreter.
pub fn default_vision_ocr_probe() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    mt_ml::vision_ocr_available()
}
