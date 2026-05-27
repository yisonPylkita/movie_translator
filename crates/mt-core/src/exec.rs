//! Shared executable resolution for the `mt-*` crates.
//!
//! Both `mt-media` and `mt-discovery` need to locate the same `ffmpeg` /
//! `ffprobe` binaries. Centralizing the lookup here ensures they agree, and
//! gives us one place to get the resolution logic right:
//!
//! - Use the [`which`] crate (cross-platform; no spawning the Unix-only
//!   `which` binary, no missing timeout).
//! - Validate that the resolved path is non-empty **and** [`Path::is_file`]
//!   (a bare `which` output line could otherwise become an empty `PathBuf`).
//! - Only memoize **successful** lookups; a failed lookup is retried on the
//!   next call (so installing the binary mid-process is picked up, and a
//!   transient failure isn't cached forever).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::error::{MtError, Result};

/// Cache of successfully resolved binaries, keyed by the requested name.
fn cache() -> &'static Mutex<HashMap<String, PathBuf>> {
    static CACHE: OnceLock<Mutex<HashMap<String, PathBuf>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolve an executable `name` to an absolute path.
///
/// Returns [`MtError::PathResolution`] if the binary cannot be found or the
/// resolved entry is not a regular file. Successful results are cached;
/// failures are **not** cached and will be retried on the next call.
pub fn find_binary(name: &str) -> Result<PathBuf> {
    if let Some(hit) = cache().lock().unwrap().get(name).cloned() {
        return Ok(hit);
    }

    let resolved = which::which(name).map_err(|e| {
        MtError::PathResolution(format!(
            "`{name}` not found on PATH ({e}). Install FFmpeg (e.g. `brew install ffmpeg`) \
             or run ./setup.sh"
        ))
    })?;

    // Guard against an empty / non-file result: a valid binary must be a real
    // regular file, not an empty path or a directory.
    if resolved.as_os_str().is_empty() || !resolved.is_file() {
        return Err(MtError::PathResolution(format!(
            "resolved path for `{name}` is not a regular file: {}",
            resolved.display()
        )));
    }

    cache()
        .lock()
        .unwrap()
        .insert(name.to_string(), resolved.clone());
    Ok(resolved)
}

/// Resolve the `ffmpeg` binary.
pub fn get_ffmpeg() -> Result<PathBuf> {
    find_binary("ffmpeg")
}

/// Resolve the `ffprobe` binary.
pub fn get_ffprobe() -> Result<PathBuf> {
    find_binary("ffprobe")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_binary_is_not_cached_and_errors() {
        let name = "mt_core_definitely_missing_binary_xyz";
        let first = find_binary(name);
        assert!(first.is_err());
        // A failed lookup must not be memoized (cache stays empty for it).
        assert!(!cache().lock().unwrap().contains_key(name));
        // Retrying still errors (and doesn't panic).
        assert!(find_binary(name).is_err());
    }

    #[test]
    fn error_message_is_clear() {
        let e = find_binary("mt_core_missing_binary_abc").unwrap_err();
        assert!(e.to_string().contains("not found on PATH"));
    }

    #[test]
    fn resolves_a_real_binary_and_caches_it() {
        // `sh` exists on every supported platform's PATH in CI.
        if let Ok(p) = find_binary("sh") {
            assert!(p.is_file());
            // Second call hits the cache and returns the same path.
            assert_eq!(find_binary("sh").unwrap(), p);
        }
    }
}
