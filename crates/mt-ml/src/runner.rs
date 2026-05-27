//! Generic subprocess runner for the Python `ml/*.py` helper scripts.
//!
//! ML inference stays in Python; this crate spawns a single-purpose script
//! per stage-per-file (the model loads once per call). Each script reads
//! JSON from stdin (optional) and writes JSON to stdout.

use mt_core::{MtError, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Locate the repository root by walking up from the current directory
/// until we find the `ml/` directory (which contains the helper scripts).
/// Falls back to the current directory if not found.
///
// TODO: centralize this helper — duplicated from `mt-discovery::parser`.
fn find_repo_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if dir.join("ml").join("parse_filename.py").exists() {
            return dir;
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Run an `ml/<script>` Python helper, optionally piping a JSON request to
/// stdin, and deserialise the JSON response from stdout.
///
/// Spawns `uv run python ml/<script> [args...]` from the repo root, falling
/// back to `python3 ml/<script>` if `uv` is unavailable.
///
/// # Errors
/// Returns [`MtError::Subprocess`] on non-zero exit (carrying the captured
/// stderr text), or [`MtError::Parse`] if the request can't be serialised or
/// the response can't be deserialised.
pub fn run_script_json<Req, Resp>(
    script: &str,
    args: &[&str],
    stdin_json: Option<&Req>,
) -> Result<Resp>
where
    Req: Serialize,
    Resp: DeserializeOwned,
{
    let stdin_payload = match stdin_json {
        Some(req) => Some(
            serde_json::to_string(req)
                .map_err(|e| MtError::Parse(format!("failed to serialize request: {e}")))?,
        ),
        None => None,
    };

    let repo_root = find_repo_root();
    let script_rel = format!("ml/{script}");

    // Build argv prefixes for the two launchers we try.
    let mut uv_argv: Vec<&str> = vec!["uv", "run", "python", &script_rel];
    uv_argv.extend_from_slice(args);
    let mut py_argv: Vec<&str> = vec!["python3", &script_rel];
    py_argv.extend_from_slice(args);

    let output = try_run(&uv_argv, stdin_payload.as_deref(), &repo_root)
        .or_else(|_| try_run(&py_argv, stdin_payload.as_deref(), &repo_root))?;

    serde_json::from_slice(&output)
        .map_err(|e| MtError::Parse(format!("failed to parse script output: {e}")))
}

/// Spawn a single command, optionally write `stdin` to it, and return its
/// captured stdout bytes. Non-zero exit becomes [`MtError::Subprocess`].
fn try_run(argv: &[&str], stdin: Option<&str>, cwd: &Path) -> Result<Vec<u8>> {
    let mut command = Command::new(argv[0]);
    command
        .args(&argv[1..])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if stdin.is_some() {
        command.stdin(Stdio::piped());
    } else {
        command.stdin(Stdio::null());
    }

    let mut child = command.spawn().map_err(MtError::Io)?;

    if let Some(payload) = stdin {
        child
            .stdin
            .take()
            .expect("stdin piped")
            .write_all(payload.as_bytes())
            .map_err(MtError::Io)?;
    }

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
