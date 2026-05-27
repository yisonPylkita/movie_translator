//! Shared CLI utilities: dependency checks and model resolution.
//!
//! Port of `movie_translator/commands/common.py`
//! (`check_dependencies`, `resolve_model`, `resolve_models`).

use std::path::PathBuf;

/// Verify required external tools are present.
///
/// Port of `check_dependencies`. The Python version checks the Python version
/// and `pysubs2`/`torch`/`transformers` import availability — those are runtime
/// concerns of the ML helper scripts, not the Rust binary. Here we verify the
/// media toolchain the pipeline shells out to: `ffmpeg`/`ffprobe` must be
/// discoverable (via `mt_media`). Returns `true` if all satisfied.
pub fn check_dependencies() -> bool {
    if mt_media::get_ffmpeg_version().is_err() {
        eprintln!("FFmpeg not available. Run ./setup.sh first.");
        return false;
    }
    if mt_media::get_ffprobe().is_err() {
        eprintln!("ffprobe not available. Run ./setup.sh first.");
        return false;
    }
    true
}

/// Back-compat: returns just the primary model.
///
/// Port of `resolve_model`.
#[allow(dead_code)]
pub fn resolve_model(explicit_choice: Option<&str>) -> String {
    resolve_models(explicit_choice).0
}

/// Pick the primary translation backend + any extra backends to also run.
///
/// Port of `resolve_models`. Returns `(primary_model, extra_models)`. On macOS
/// where Apple Translation is available we default to running Allegro AND Apple
/// (two PL tracks). An explicit `--model X` is honoured with no extras.
pub fn resolve_models(explicit_choice: Option<&str>) -> (String, Vec<String>) {
    resolve_models_with(explicit_choice, apple_translation_available)
}

/// Like [`resolve_models`] with an injectable apple-availability probe (tests).
pub fn resolve_models_with(
    explicit_choice: Option<&str>,
    apple_available: impl Fn() -> bool,
) -> (String, Vec<String>) {
    if let Some(choice) = explicit_choice {
        return (choice.to_string(), Vec::new());
    }

    if apple_available() {
        tracing::info!("Apple Translation available -- running Allegro + Apple (two PL tracks)");
        return ("allegro".to_string(), vec!["apple".to_string()]);
    }

    ("allegro".to_string(), Vec::new())
}

/// Directory holding the Apple Translation Swift bridge, relative to the repo.
fn apple_swift_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../movie_translator/translation/swift")
}

/// Port of `apple_backend.is_available()` + `check_languages_installed()`.
///
/// `is_available`: macOS 26.0+ (Tahoe) and the Swift source exists.
/// `check_languages_installed`: run the compiled `translate_bridge test`
/// command and treat success as "languages installed". If the binary is not
/// yet compiled we conservatively return `false` (the Python path compiles it
/// on demand; doing a `swiftc` build here is out of scope, see the crate
/// concern note in the report).
fn apple_translation_available() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    let swift_dir = apple_swift_dir();
    if !swift_dir.join("translate_bridge.swift").exists() {
        return false;
    }
    // macOS major version >= 26.
    if !macos_major_at_least(26) {
        return false;
    }
    // check_languages_installed: the compiled bridge must respond to `test`.
    let binary = swift_dir.join("translate_bridge");
    if !binary.exists() {
        return false;
    }
    std::process::Command::new(&binary)
        .arg("test")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Best-effort `platform.mac_ver()` major-version check via `sw_vers`.
#[cfg(target_os = "macos")]
fn macos_major_at_least(min_major: u32) -> bool {
    std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .and_then(|o| {
            if !o.status.success() {
                return None;
            }
            let v = String::from_utf8_lossy(&o.stdout);
            v.trim().split('.').next()?.parse::<u32>().ok()
        })
        .map(|major| major >= min_major)
        .unwrap_or(false)
}

#[cfg(not(target_os = "macos"))]
fn macos_major_at_least(_min_major: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_choice_runs_no_extras() {
        let (primary, extra) = resolve_models_with(Some("apple"), || true);
        assert_eq!(primary, "apple");
        assert!(extra.is_empty(), "explicit choice must not add extras");

        let (primary, extra) = resolve_models_with(Some("allegro"), || true);
        assert_eq!(primary, "allegro");
        assert!(extra.is_empty());
    }

    #[test]
    fn default_with_apple_available_adds_apple_extra() {
        let (primary, extra) = resolve_models_with(None, || true);
        assert_eq!(primary, "allegro");
        assert_eq!(extra, vec!["apple".to_string()]);
    }

    #[test]
    fn default_without_apple_is_allegro_only() {
        let (primary, extra) = resolve_models_with(None, || false);
        assert_eq!(primary, "allegro");
        assert!(extra.is_empty());
    }

    #[test]
    fn resolve_model_back_compat_returns_primary() {
        // explicit path doesn't touch the platform probe.
        assert_eq!(resolve_model(Some("apple")), "apple");
    }
}
