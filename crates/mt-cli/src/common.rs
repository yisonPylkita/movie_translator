//! Shared CLI utilities: dependency checks and model resolution.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, from_slice};
use tracing::{error, info, warn};

/// Verify required external tools are present.
///
/// Checks the media toolchain the pipeline shells out to: `ffmpeg`/`ffprobe`
/// must be discoverable (via `mt_media`). Returns `true` if all satisfied.
pub fn check_dependencies() -> bool {
    if mt_media::get_ffmpeg_version().is_err() {
        error!("FFmpeg not available. Run ./setup.sh first.");
        return false;
    }
    if mt_media::get_ffprobe().is_err() {
        error!("ffprobe not available. Run ./setup.sh first.");
        return false;
    }
    true
}

/// Pick the primary translation backend + any extra backends to also run.
///
/// Returns `(primary_model, extra_models)`. On macOS where Apple Translation is
/// available we default to running Allegro AND Apple (two PL tracks). An
/// explicit `--model X` is honoured with no extras.
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
        info!("Apple Translation available — using Apple (fastest, zero memory)");
        return ("apple".to_string(), Vec::new());
    }

    // Fallback to MLX on Apple Silicon when Apple Translation is unavailable
    // (e.g. macOS < 26). MLX is Metal-native, INT8 quantised, and stable.
    ("mlx".to_string(), Vec::new())
}

/// Directory holding the Apple Translation Swift bridge, relative to the repo.
fn apple_swift_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../movie_translator/translation/swift")
}

/// Port of `apple_backend.is_available()` + `check_languages_installed()`.
///
/// `is_available`: macOS 26.0+ (Tahoe) and the Swift source exists.
/// `check_languages_installed`: ensure the `translate_bridge` binary is built
/// (compiling on demand via `swiftc`, like Python's `_ensure_binary`), then run
/// the `test` command and treat success as "languages installed".
///
/// The result is cached for the process lifetime so we don't recompile / re-probe
/// on every `resolve_models` call.
fn apple_translation_available() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(apple_translation_available_uncached)
}

fn apple_translation_available_uncached() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    let swift_dir = apple_swift_dir();
    let source = swift_dir.join("translate_bridge.swift");
    if !source.exists() {
        return false;
    }
    // macOS major version >= 26.
    if !macos_major_at_least(26) {
        return false;
    }
    // _ensure_binary: compile the bridge if missing or stale, then probe.
    let binary = match ensure_apple_bridge(&source, &swift_dir.join("translate_bridge")) {
        Some(b) => b,
        None => return false,
    };
    // check_languages_installed: do a test EN->PL translation, exactly like
    // Python's `_call_swift_binary(['test'])`. The bridge ignores argv and reads
    // a JSON request from stdin (`{"texts":[...],"source":"en","target":"pl"}`),
    // writing `{"translations":[...]}` (or `{"error":...}`) to stdout. We must
    // feed stdin and inspect the response — passing "test" as an argv with empty
    // stdin makes the bridge fail to decode and exit non-zero.
    bridge_test_translation_ok(&binary)
}

/// Run the Swift bridge's probe translation (port of `check_languages_installed`
/// → `_call_swift_binary(['test'])`): returns true iff the bridge produces a
/// translation with no error, which confirms the EN→PL language pack is
/// installed and on-device translation works.
fn bridge_test_translation_ok(binary: &Path) -> bool {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let request = br#"{"texts":["test"],"source":"en","target":"pl"}"#;
    let mut child = match Command::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    if let Some(stdin) = child.stdin.as_mut()
        && stdin.write_all(request).is_err()
    {
        return false;
    }
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(_) => return false,
    };
    // Success = valid JSON response carrying a non-empty `translations` array and
    // no `error` field (mirrors Python's AppleTranslationError checks).
    match from_slice::<Value>(&output.stdout) {
        Ok(v) => {
            let no_error = v.get("error").map(|e| e.is_null()).unwrap_or(true);
            let has_translation = v
                .get("translations")
                .and_then(|t| t.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            no_error && has_translation
        }
        Err(_) => false,
    }
}

/// Compile the Apple Translation Swift bridge on demand if needed.
///
/// Port of `apple_backend._ensure_binary`: recompile when the binary is missing
/// or older than the source, using the same `swiftc` invocation/output path.
/// Returns `Some(binary_path)` if the binary exists (already or after a
/// successful build), `None` if it cannot be built (no source, no `swiftc`, or
/// the compile failed).
fn ensure_apple_bridge(source: &Path, binary: &Path) -> Option<PathBuf> {
    let needs_compile = match (binary.metadata(), source.metadata()) {
        (Ok(bin_meta), Ok(src_meta)) => match (bin_meta.modified(), src_meta.modified()) {
            (Ok(bin_mtime), Ok(src_mtime)) => src_mtime > bin_mtime,
            // If we can't read mtimes, only compile when the binary is absent
            // (it exists here), so don't recompile.
            _ => false,
        },
        // Binary missing -> must compile.
        (Err(_), Ok(_)) => true,
        // Source missing -> caller already checked, but be safe.
        _ => return None,
    };

    if needs_compile {
        let swiftc = which_swiftc()?;
        info!("Compiling Apple Translation bridge...");
        let status = Command::new(swiftc)
            .args(["-parse-as-library", "-O", "-framework", "Translation"])
            .arg(source)
            .arg("-o")
            .arg(binary)
            .output();
        match status {
            Ok(out) if out.status.success() => {
                info!("Compiled Apple Translation bridge: {}", binary.display());
            }
            Ok(out) => {
                warn!(
                    "Failed to compile Apple Translation bridge: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
                return None;
            }
            Err(e) => {
                warn!("Failed to invoke swiftc for Apple Translation bridge: {e}");
                return None;
            }
        }
    }

    if binary.exists() {
        Some(binary.to_path_buf())
    } else {
        None
    }
}

/// Locate `swiftc` on PATH (equivalent to Python's `shutil.which('swiftc')`).
fn which_swiftc() -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let candidate = dir.join("swiftc");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Best-effort `platform.mac_ver()` major-version check via `sw_vers`.
#[cfg(target_os = "macos")]
fn macos_major_at_least(min_major: u32) -> bool {
    Command::new("sw_vers")
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
    fn apple_is_default_when_available() {
        let (primary, extra) = resolve_models_with(None, || true);
        assert_eq!(primary, "apple");
        assert!(extra.is_empty(), "Apple default must not add extras");

        let (primary, extra) = resolve_models_with(None, || false);
        assert_eq!(primary, "mlx");
        assert!(extra.is_empty());
    }
}
