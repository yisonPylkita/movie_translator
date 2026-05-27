//! Filename parsing via the Python `ml/parse_filename.py` helper script.
//!
//! The underlying logic relies on `guessit` and `aniparse` Python libraries,
//! which have no Rust equivalent, so this module shells out to the script.
//! The script reads JSON from stdin and writes JSON to stdout.

use mt_core::{MtError, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};

/// Parsed filename metadata returned by `ml/parse_filename.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedName {
    /// Best-guess title from filename/aniparse/guessit.
    pub title: Option<String>,
    /// Release year, if detected.
    pub year: Option<i32>,
    /// Season number, if applicable.
    pub season: Option<i32>,
    /// Episode number, if applicable.
    pub episode: Option<i32>,
    /// `"movie"` or `"episode"`.
    pub media_type: String,
    /// `true` when anime signals (e.g. fansub group) were detected.
    pub is_anime: bool,
    /// Fansub / release group, e.g. `"HorribleSubs"`.
    pub release_group: Option<String>,
}

/// JSON request sent to the Python script via stdin.
#[derive(Serialize)]
struct ParseRequest<'a> {
    filename: &'a str,
    folder_name: Option<&'a str>,
}

/// Locate the repository root by walking up from the current directory
/// until we find the `ml/` subdirectory (which contains `parse_filename.py`).
/// Falls back to the current directory if not found.
fn find_repo_root() -> std::path::PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    loop {
        if dir.join("ml").join("parse_filename.py").exists() {
            return dir;
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
}

/// Invoke `ml/parse_filename.py` to parse a video filename.
///
/// Spawns `uv run python ml/parse_filename.py` (from the repo root),
/// passes JSON to stdin, and deserialises the JSON response from stdout.
///
/// # Errors
/// Returns [`MtError::Subprocess`] on non-zero exit or if stderr contains
/// an error message.
pub fn parse_filename(filename: &str, folder: Option<&str>) -> Result<ParsedName> {
    let request = ParseRequest {
        filename,
        folder_name: folder,
    };
    let input = serde_json::to_string(&request)
        .map_err(|e| MtError::Parse(format!("failed to serialize request: {e}")))?;

    let repo_root = find_repo_root();

    // Try `uv run python` first; if uv isn't available fall back to `python3`.
    let output =
        try_run_script(&input, &["uv", "run", "python", "ml/parse_filename.py"], &repo_root)
            .or_else(|_| {
                try_run_script(&input, &["python3", "ml/parse_filename.py"], &repo_root)
            })?;

    serde_json::from_slice(&output)
        .map_err(|e| MtError::Parse(format!("failed to parse script output: {e}")))
}

fn try_run_script(
    input: &str,
    argv: &[&str],
    cwd: &std::path::Path,
) -> Result<Vec<u8>> {
    let mut child = Command::new(argv[0])
        .args(&argv[1..])
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(MtError::Io)?;

    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(input.as_bytes())
        .map_err(MtError::Io)?;

    let out = child.wait_with_output().map_err(MtError::Io)?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        return Err(MtError::Subprocess {
            cmd: argv.join(" "),
            code: out.status.code(),
            stderr,
        });
    }

    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit test: ParsedName JSON round-trip (no subprocess required).
    #[test]
    fn parsed_name_deserialize_sample() {
        let json = r#"{
            "title": "One Piece",
            "year": null,
            "season": null,
            "episode": 1000,
            "media_type": "episode",
            "is_anime": true,
            "release_group": "HorribleSubs"
        }"#;
        let parsed: ParsedName = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.title.as_deref(), Some("One Piece"));
        assert_eq!(parsed.episode, Some(1000));
        assert!(parsed.is_anime);
        assert_eq!(parsed.release_group.as_deref(), Some("HorribleSubs"));
        assert_eq!(parsed.media_type, "episode");
        assert!(parsed.year.is_none());
        assert!(parsed.season.is_none());
    }

    /// Unit test: ParsedName for a movie (all nullable fields absent).
    #[test]
    fn parsed_name_deserialize_movie() {
        let json = r#"{
            "title": "Spirited Away",
            "year": 2001,
            "season": null,
            "episode": null,
            "media_type": "movie",
            "is_anime": false,
            "release_group": null
        }"#;
        let parsed: ParsedName = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.title.as_deref(), Some("Spirited Away"));
        assert_eq!(parsed.year, Some(2001));
        assert!(!parsed.is_anime);
        assert!(parsed.release_group.is_none());
        assert_eq!(parsed.media_type, "movie");
    }

    /// Integration test: actually runs `uv run python ml/parse_filename.py`.
    ///
    /// Marked `#[ignore]` because it requires the Python environment.
    /// Run with `cargo test -p mt-discovery -- --ignored` to execute.
    #[test]
    #[ignore]
    fn integration_parse_anime_filename() {
        let result =
            parse_filename("[HorribleSubs] One Piece - 1000 [1080p].mkv", None).unwrap();
        assert!(
            result.title.as_deref().is_some_and(|t| !t.is_empty()),
            "expected non-empty parsed_title, got {:?}",
            result.title
        );
        assert!(result.is_anime, "expected is_anime=true for fansub filename");
        assert_eq!(result.release_group.as_deref(), Some("HorribleSubs"));
        assert_eq!(result.episode, Some(1000));
    }
}
