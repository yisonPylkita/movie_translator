//! Compile-on-first-use helper for Swift bridge binaries.
//!
//! Both Apple backends (Translation, SpeechAnalyzer transcription) ship a Swift
//! source file and compile it lazily.  This is the single home for that
//! mechanism so toolchain fixes land once.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::error::{MtError, Result};

/// Check whether the current system is macOS with at least the given major version.
pub fn macos_at_least(major: u32) -> bool {
    let os = std::env::consts::OS;
    if os != "macos" {
        return false;
    }
    // Parse version from `sw_vers -productVersion`
    let output = Command::new("sw_vers").arg("-productVersion").output();
    let output = match output {
        Ok(o) => o,
        Err(_) => return false,
    };
    let version_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let major_str = version_str.split('.').next().unwrap_or("0");
    major_str.parse::<u32>().unwrap_or(0) >= major
}

/// Compile `source` to `binary` with swiftc if missing or stale (by mtime).
///
/// Returns the path to the compiled binary.
pub fn ensure_compiled(
    source: &Path,
    binary: &Path,
    extra_args: &[&str],
    _timeout: Duration,
) -> Result<PathBuf> {
    if !source.exists() {
        return Err(MtError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Swift bridge source not found: {}", source.display()),
        )));
    }

    // Check if binary is already fresh (exists and source is not newer)
    if let (Ok(src_mtime), Ok(bin_mtime)) = (
        source.metadata().and_then(|m| m.modified()),
        binary.metadata().and_then(|m| m.modified()),
    ) && src_mtime <= bin_mtime
    {
        return Ok(binary.to_path_buf());
    }

    // Find swiftc
    let swiftc = which::which("swiftc").map_err(|_| {
        MtError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Swift compiler (swiftc) not found. Install Command Line Tools: xcode-select --install",
        ))
    })?;

    tracing::info!("Compiling Swift bridge: {}", source.display());

    // Create parent dir if needed
    if let Some(parent) = binary.parent() {
        std::fs::create_dir_all(parent).map_err(MtError::Io)?;
    }

    let mut cmd = Command::new(&swiftc);
    cmd.arg("-O");
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg(source);
    cmd.arg("-o");
    cmd.arg(binary);

    let output = cmd.output().map_err(MtError::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Truncate to last 1000 chars
        let truncated = if stderr.len() > 1000 {
            format!("...{}", &stderr[stderr.len() - 1000..])
        } else {
            stderr.to_string()
        };
        return Err(MtError::Parse(format!(
            "Swift bridge compilation failed ({}):\n{}",
            source.display(),
            truncated
        )));
    }

    tracing::info!("Compiled: {}", binary.display());
    Ok(binary.to_path_buf())
}
