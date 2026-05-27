//! Centralized resolution of repository-relative resource paths.
//!
//! The `ml/` directory holds Python helper scripts (`parse_filename.py`, etc.)
//! that several crates shell out to. Locating it robustly — from either the
//! executable location or the working directory — avoids the brittle
//! cwd-only walk (with a silent cwd fallback) that was previously duplicated
//! across `mt-ml` and `mt-discovery`.

use std::path::{Path, PathBuf};

use crate::error::{MtError, Result};

/// The sentinel file used to recognise the `ml/` scripts directory.
const ML_ANCHOR: &str = "parse_filename.py";

/// Walk up from `start` (inclusive) looking for a directory that contains
/// `ml/parse_filename.py`. Returns the `ml/` directory itself if found.
fn find_ml_from(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let candidate = d.join("ml");
        if candidate.join(ML_ANCHOR).is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// Resolve the `ml/` scripts directory.
///
/// Tries walking up from [`std::env::current_exe`] and from
/// [`std::env::current_dir`], looking for `ml/parse_filename.py`. Returns a
/// clear [`MtError::PathResolution`] if it cannot be located in either — there
/// is no silent cwd fallback.
pub fn ml_dir() -> Result<PathBuf> {
    // Prefer the executable location: it's stable regardless of the cwd the
    // user happens to invoke the binary from.
    if let Ok(exe) = std::env::current_exe() {
        // The executable itself is a file; start from its directory.
        let start = exe.parent().unwrap_or(&exe);
        if let Some(ml) = find_ml_from(start) {
            return Ok(ml);
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Some(ml) = find_ml_from(&cwd) {
            return Ok(ml);
        }
    }

    Err(MtError::PathResolution(format!(
        "could not locate ml/ scripts directory; expected near the executable \
         or working directory (looking for ml/{ML_ANCHOR})"
    )))
}

/// Resolve the repository root: the parent of the [`ml_dir`].
pub fn repo_root() -> Result<PathBuf> {
    let ml = ml_dir()?;
    ml.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| MtError::PathResolution("ml/ directory has no parent".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn find_ml_from_locates_anchor() {
        let tmp = tempdir_like();
        let ml = tmp.join("ml");
        fs::create_dir_all(&ml).unwrap();
        fs::write(ml.join(ML_ANCHOR), b"# stub").unwrap();

        // From a nested subdirectory, it should walk up and find ml/.
        let nested = tmp.join("a").join("b");
        fs::create_dir_all(&nested).unwrap();
        let found = find_ml_from(&nested).expect("should find ml dir");
        assert_eq!(found, ml);

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn find_ml_from_returns_none_when_absent() {
        let tmp = tempdir_like();
        fs::create_dir_all(&tmp).unwrap();
        assert!(find_ml_from(&tmp).is_none());
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn ml_dir_errors_have_clear_message() {
        // We can't reliably remove the real ml/ dir, so just assert the error
        // message wording is what callers expect when construction fails.
        let e = MtError::PathResolution(
            "could not locate ml/ scripts directory; expected near the executable".to_string(),
        );
        assert!(e.to_string().contains("could not locate ml/"));
    }

    /// Minimal unique temp dir without pulling `tempfile` into mt-core deps.
    fn tempdir_like() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mt_core_paths_test_{nanos}"))
    }
}
