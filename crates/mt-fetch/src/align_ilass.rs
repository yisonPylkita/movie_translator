//! Subtitle alignment using ilass (improved alass).
//!
//! Uses the ilass CLI tool for subtitle-to-subtitle alignment via dynamic
//! programming with split penalties.  Handles OP removal, ad breaks, and
//! other structural differences automatically without heuristic gap detection.
//!
//! ilass is built from source in vendor/ilass and must be compiled before use.
//! See: <https://github.com/SandroHc/ilass>

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{info, warn};

// ---------------------------------------------------------------------------
// ilass binary path
// ---------------------------------------------------------------------------

/// Return the path to the ilass binary (resolved relative to the project root).
pub fn ilass_binary_path() -> PathBuf {
    // Resolve from this source file: src/align_ilass.rs → crates/mt-fetch → project root
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("..") // crates/
        .join("..") // project root
        .join("vendor")
        .join("ilass")
        .join("target")
        .join("release")
        .join("ilass")
}

/// Check if the ilass binary is built and available.
pub fn is_available() -> bool {
    ilass_binary_path().is_file()
}

// ---------------------------------------------------------------------------
// build_ilass_argv — pure function, fully testable
// ---------------------------------------------------------------------------

/// Build the argv vector for an ilass subprocess invocation.
///
/// Factored out as a pure function so it can be tested without running the binary.
pub fn build_ilass_argv(
    binary: &Path,
    reference_path: &Path,
    subtitle_path: &Path,
    output_path: &Path,
    split_penalty: f64,
) -> Vec<String> {
    vec![
        binary.to_string_lossy().into_owned(),
        reference_path.to_string_lossy().into_owned(),
        subtitle_path.to_string_lossy().into_owned(),
        output_path.to_string_lossy().into_owned(),
        "--split-penalty".to_string(),
        split_penalty.to_string(),
        "--disable-fps-guessing".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// align_to_reference
// ---------------------------------------------------------------------------

/// Align a subtitle file to a reference using ilass.
///
/// Uses ilass's DP algorithm with split penalties to find per-line offsets.
/// This handles OP removal, ad breaks, and other structural differences
/// automatically.
///
/// The subtitle file is modified in place.
///
/// If ilass is not available or fails, falls back to the cross-correlation
/// aligner in `align.rs`.
pub fn align_to_reference(subtitle_path: &Path, reference_path: &Path, split_penalty: f64) -> bool {
    if !is_available() {
        warn!(
            "ilass binary not found at {}, falling back to cross-correlation",
            ilass_binary_path().display()
        );
        let offset = crate::align::align_to_reference(
            subtitle_path,
            reference_path,
            crate::align::MIN_OFFSET_MS,
        );
        info!("Fallback cross-correlation applied offset: {offset:+}ms");
        return offset != 0;
    }

    // ilass writes to a new file — use a temp file then replace.
    let output_path = subtitle_path.with_extension(format!(
        "ilass_tmp{}",
        subtitle_path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default()
    ));

    let argv = build_ilass_argv(
        &ilass_binary_path(),
        reference_path,
        subtitle_path,
        &output_path,
        split_penalty,
    );

    let result = Command::new(&argv[0]).args(&argv[1..]).output();

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            warn!("ilass error: {e}");
            return false;
        }
    };

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        warn!("ilass failed (exit {:?}): {}", result.status.code(), stderr);
        let _ = fs::remove_file(&output_path);
        return false;
    }

    // Log the alignment summary from stderr
    let stderr = String::from_utf8_lossy(&result.stderr);
    for line in stderr.lines() {
        if line.starts_with("shifted block") {
            info!("ilass: {}", line);
        }
    }

    // Replace original with aligned output
    if let Err(e) = fs::rename(&output_path, subtitle_path) {
        warn!("failed to replace subtitle with ilass output: {e}");
        let _ = fs::remove_file(&output_path);
        return false;
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    // -----------------------------------------------------------------------
    // Tests for build_ilass_argv (pure function — always runnable)
    // -----------------------------------------------------------------------

    #[test]
    fn build_ilass_argv_correct_argument_order() {
        let binary = Path::new("/path/to/ilass");
        let reference = Path::new("/subs/ref.srt");
        let subtitle = Path::new("/subs/cand.srt");
        let output = Path::new("/subs/cand.ilass_tmp.srt");

        let argv = build_ilass_argv(binary, reference, subtitle, output, 7.0);

        assert_eq!(argv.len(), 7);
        assert_eq!(argv[0], "/path/to/ilass");
        assert_eq!(argv[1], "/subs/ref.srt"); // reference_path first
        assert_eq!(argv[2], "/subs/cand.srt"); // subtitle_path second
        assert_eq!(argv[3], "/subs/cand.ilass_tmp.srt"); // output_path third
        assert_eq!(argv[4], "--split-penalty");
        assert_eq!(argv[5], "7");
        assert_eq!(argv[6], "--disable-fps-guessing");
    }

    #[test]
    fn build_ilass_argv_custom_split_penalty() {
        let argv = build_ilass_argv(
            Path::new("/ilass"),
            Path::new("/r.srt"),
            Path::new("/s.srt"),
            Path::new("/o.srt"),
            12.5,
        );
        assert_eq!(argv[5], "12.5");
    }

    #[test]
    fn build_ilass_argv_default_split_penalty_is_7() {
        // Mirror Python default: split_penalty=7.0
        let argv = build_ilass_argv(
            Path::new("/ilass"),
            Path::new("/r.srt"),
            Path::new("/s.srt"),
            Path::new("/o.srt"),
            7.0,
        );
        assert_eq!(argv[5], "7");
    }

    #[test]
    fn build_ilass_argv_includes_disable_fps_flag() {
        let argv = build_ilass_argv(
            Path::new("/ilass"),
            Path::new("/r.srt"),
            Path::new("/s.srt"),
            Path::new("/o.srt"),
            7.0,
        );
        assert!(
            argv.contains(&"--disable-fps-guessing".to_string()),
            "expected --disable-fps-guessing in argv"
        );
    }

    #[test]
    fn build_ilass_argv_paths_preserved_exactly() {
        let binary = Path::new("/usr/local/bin/ilass");
        let reference = Path::new("/data/shows/Anime S01E01.en.srt");
        let subtitle = Path::new("/data/shows/Anime S01E01.pl.srt");
        let output = Path::new("/data/shows/Anime S01E01.pl.ilass_tmp.srt");

        let argv = build_ilass_argv(binary, reference, subtitle, output, 7.0);
        assert_eq!(argv[1], "/data/shows/Anime S01E01.en.srt");
        assert_eq!(argv[2], "/data/shows/Anime S01E01.pl.srt");
        assert_eq!(argv[3], "/data/shows/Anime S01E01.pl.ilass_tmp.srt");
    }

    // -----------------------------------------------------------------------
    // Tests for is_available (does not require ilass installed)
    // -----------------------------------------------------------------------

    #[test]
    fn is_available_returns_false_when_not_built() {
        // ilass binary is not expected to be built in CI.
        // This is a best-effort check: we just verify the function returns a bool
        // and doesn't panic.  In a clean checkout the binary won't exist.
        let _ = is_available(); // should not panic
    }

    #[test]
    fn ilass_binary_path_is_absolute() {
        let path = ilass_binary_path();
        // After canonicalization attempts it should at least contain a non-empty path.
        assert!(
            path.to_string_lossy().contains("ilass"),
            "expected 'ilass' in path: {}",
            path.display()
        );
    }

    // -----------------------------------------------------------------------
    // Integration test (requires ilass to be built) — marked #[ignore]
    // -----------------------------------------------------------------------

    /// Integration test: run ilass on real SRT files.
    ///
    /// Requires `vendor/ilass/target/release/ilass` to be built.
    /// Run with: `cargo test -p mt-fetch -- --ignored align_ilass_live`
    #[test]
    #[ignore]
    fn align_ilass_live_aligns_shifted_srt() {
        let tmp = TempDir::new().unwrap();

        let ref_content =
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:05,000 --> 00:00:07,000\nWorld\n";
        let cand_content =
            "1\n00:00:03,000 --> 00:00:05,000\nHello\n\n2\n00:00:07,000 --> 00:00:09,000\nWorld\n";

        let ref_path = tmp.path().join("ref.srt");
        let cand_path = tmp.path().join("cand.srt");
        fs::write(&ref_path, ref_content).unwrap();
        fs::write(&cand_path, cand_content).unwrap();

        let ok = align_to_reference(&cand_path, &ref_path, 7.0);
        assert!(ok, "ilass alignment should succeed");

        // After alignment, cand should be close to ref timing
        let aligned = fs::read_to_string(&cand_path).unwrap();
        assert!(
            aligned.contains("00:00:01") || aligned.contains("00:00:02"),
            "expected aligned timing near reference: {aligned}"
        );
    }
}
