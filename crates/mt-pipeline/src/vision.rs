//! Vision OCR availability probe.
//!
//! Reports `true` only on macOS where the `Quartz`/`Vision` Python bindings
//! import cleanly. The actual OCR runs via the `mt_ml` helper scripts, so here
//! we only need the *availability* signal used by the extract stages to decide
//! whether to defer a burned-in OCR pass.
//!
//! The check is injected into the extract stages as a closure so tests can
//! force it on/off without touching the host platform.

use std::sync::OnceLock;

/// A predicate that reports whether Vision-based OCR is available.
pub type VisionOcrProbe = fn() -> bool;

/// Default probe: a precise, cached availability check.
///
/// On non-macOS this is trivially `false`. On macOS we verify the `Vision` and
/// `Quartz` Python bindings actually import, shelling out to
/// `python3 -c "import Vision, Quartz"` exactly **once** and caching the boolean
/// for the rest of the process. If the probe cannot be run at all we fall back
/// to the conservative platform check.
pub fn default_vision_ocr_probe() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(probe_macos_vision)
}

/// Run the one-shot macOS Vision import check. Cached by the caller.
fn probe_macos_vision() -> bool {
    match std::process::Command::new("python3")
        .args(["-c", "import Vision, Quartz"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
    {
        Ok(status) => status.success(),
        // python3 missing / not runnable: fall back to the platform signal.
        Err(_) => cfg!(target_os = "macos"),
    }
}
