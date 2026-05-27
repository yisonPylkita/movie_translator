//! Generic subprocess runner for the Python `ml/*.py` helper scripts.
//!
//! ML inference stays in Python; this crate spawns a single-purpose script
//! per stage-per-file (the model loads once per call). Each script reads
//! JSON from stdin (optional) and writes JSON to stdout.

use mt_core::{paths, MtError, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Maximum wall-clock time a single `ml/*.py` invocation may run before it is
/// killed and reported as a timeout.
///
/// OCR / inpaint of a full episode is legitimately several minutes, so this is
/// deliberately generous. Tune here (or override per-call via
/// [`run_script_json_with_timeout`]) if a stage needs longer.
pub const DEFAULT_SCRIPT_TIMEOUT: Duration = Duration::from_secs(600);

/// How often the watchdog re-checks whether the child has exited.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Run an `ml/<script>` Python helper, optionally piping a JSON request to
/// stdin, and deserialise the JSON response from stdout.
///
/// Spawns `uv run python ml/<script> [args...]` from the repo root, falling
/// back to `python3 ml/<script>` if `uv` is unavailable. Uses
/// [`DEFAULT_SCRIPT_TIMEOUT`]; see [`run_script_json_with_timeout`] to override.
///
/// # Errors
/// Returns [`MtError::Subprocess`] on non-zero exit (carrying the captured
/// stderr text) or on timeout (the child is killed), or [`MtError::Parse`] if
/// the request can't be serialised or the response can't be deserialised.
pub fn run_script_json<Req, Resp>(
    script: &str,
    args: &[&str],
    stdin_json: Option<&Req>,
) -> Result<Resp>
where
    Req: Serialize,
    Resp: DeserializeOwned,
{
    run_script_json_with_timeout(script, args, stdin_json, DEFAULT_SCRIPT_TIMEOUT)
}

/// Like [`run_script_json`], with an explicit per-call timeout (used by tests
/// and any stage that legitimately needs a different deadline).
pub fn run_script_json_with_timeout<Req, Resp>(
    script: &str,
    args: &[&str],
    stdin_json: Option<&Req>,
    timeout: Duration,
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

    // Resolve the ml/ scripts directory via the shared helper (no cwd-only
    // walk, no silent fallback): its parent is the repo root we run from.
    let ml_dir = paths::ml_dir()?;
    let repo_root = ml_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| MtError::PathResolution("ml/ directory has no parent".to_string()))?;
    let script_rel = format!("ml/{script}");

    // Build argv prefixes for the two launchers we try.
    let mut uv_argv: Vec<&str> = vec!["uv", "run", "python", &script_rel];
    uv_argv.extend_from_slice(args);
    let mut py_argv: Vec<&str> = vec!["python3", &script_rel];
    py_argv.extend_from_slice(args);

    let output = match try_run(&uv_argv, stdin_payload.as_deref(), &repo_root, timeout) {
        Ok(out) => out,
        Err(uv_err) => {
            // Don't lose the uv failure when we fall back to python3 (the root
            // cause is often "uv not installed" or "venv not synced").
            tracing::debug!("`uv run` failed, falling back to python3: {uv_err}");
            try_run(&py_argv, stdin_payload.as_deref(), &repo_root, timeout)?
        }
    };

    serde_json::from_slice(&output)
        .map_err(|e| MtError::Parse(format!("failed to parse script output: {e}")))
}

/// Spawn a single command, optionally write `stdin` to it, and return its
/// captured stdout bytes. Non-zero exit becomes [`MtError::Subprocess`].
///
/// A watchdog enforces `timeout`: if the child does not exit in time it is
/// killed and an [`MtError::Subprocess`] ("timed out after Ns") is returned.
/// stdout/stderr are drained on dedicated threads so a child that fills its
/// pipe buffers can't deadlock the wait.
fn try_run(argv: &[&str], stdin: Option<&str>, cwd: &Path, timeout: Duration) -> Result<Vec<u8>> {
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

    // Write the stdin request on its OWN thread, concurrently with the
    // stdout/stderr drain below. A blocking `write_all` on the main thread
    // would deadlock if the child emits to stdout/stderr (filling its ~64KB
    // pipe buffer) BEFORE consuming all of stdin — which torch-importing
    // scripts do (deprecation warnings to stderr) — while we block filling its
    // stdin pipe with a >64KB request. Dropping the stdin handle at the end
    // signals EOF so the child's `sys.stdin.read()` returns.
    let stdin_handle = stdin.map(|s| {
        let payload = s.as_bytes().to_vec();
        let mut child_stdin = child.stdin.take().expect("stdin piped");
        std::thread::spawn(move || {
            // A child that died early closes its stdin → BrokenPipe. That's not
            // a runner bug (the real error surfaces via exit code/stderr), so
            // swallow it rather than panic. The handle drops here → EOF.
            let _ = child_stdin.write_all(&payload);
        })
    });

    // Drain stdout/stderr on their own threads to avoid pipe-buffer deadlock:
    // a child that fills its pipe while we block on it (or on the deadline)
    // would otherwise hang. `read_to_end` returns once the child closes the
    // pipe — which happens when it exits or when we kill it.
    let mut stdout_pipe = child.stdout.take().expect("stdout piped");
    let mut stderr_pipe = child.stderr.take().expect("stderr piped");
    let stdout_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        buf
    });
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });

    // Join the stdin-writer; never propagate its error (covered above) but
    // always reap the thread so it can't leak.
    let join_stdin = |handle: Option<std::thread::JoinHandle<()>>| {
        if let Some(h) = handle {
            let _ = h.join();
        }
    };

    // Poll for exit with a non-blocking `try_wait`, enforcing the deadline.
    // Polling (rather than a blocking waiter thread holding the child) keeps
    // `kill()` reachable the instant the deadline passes.
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    break None;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                // Couldn't query the child; kill it and surface the error.
                let _ = child.kill();
                let _ = child.wait();
                join_stdin(stdin_handle);
                let _ = stdout_handle.join();
                let _ = stderr_handle.join();
                return Err(MtError::Io(e));
            }
        }
    };

    let status = match status {
        Some(status) => status,
        None => {
            // Deadline hit: kill the hung child and report a timeout. `kill`
            // closes its pipes, releasing the drain threads (and the stdin
            // writer, which gets BrokenPipe).
            let _ = child.kill();
            let _ = child.wait();
            join_stdin(stdin_handle);
            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
            return Err(MtError::Subprocess {
                cmd: argv.join(" "),
                code: None,
                stderr: format!("timed out after {}s", timeout.as_secs()),
            });
        }
    };

    join_stdin(stdin_handle);
    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr_bytes = stderr_handle.join().unwrap_or_default();

    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
        return Err(MtError::Subprocess {
            cmd: argv.join(" "),
            code: status.code(),
            stderr,
        });
    }

    Ok(stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deliberately-hung child must be killed once the timeout fires, and the
    /// error must say it timed out. Without the watchdog this test would hang.
    #[test]
    fn timeout_kills_hung_child() {
        // Resolve the repo root the same way the runner does; skip if the ml/
        // scripts dir isn't locatable in this environment.
        let repo_root = match paths::repo_root() {
            Ok(r) => r,
            Err(_) => return,
        };

        // Build a child that sleeps far longer than the timeout, bypassing the
        // ml/ script layer (we test the runner's killing behaviour directly).
        let start = Instant::now();
        let argv = ["python3", "-c", "import time; time.sleep(60)"];
        let timeout = Duration::from_millis(300);
        let result = try_run(&argv, None, &repo_root, timeout);

        let err = result.expect_err("hung child must error, not hang");
        match err {
            MtError::Subprocess { stderr, .. } => {
                assert!(
                    stderr.contains("timed out"),
                    "expected timeout message, got: {stderr}"
                );
            }
            other => panic!("expected Subprocess timeout error, got {other:?}"),
        }
        // The watchdog must have returned promptly — well under the 60s sleep.
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "timeout took too long to fire: {:?}",
            start.elapsed()
        );
    }

    /// A fast-exiting successful child returns its stdout before the deadline.
    #[test]
    fn fast_child_succeeds_within_timeout() {
        let repo_root = std::env::current_dir().unwrap();
        let argv = ["python3", "-c", "print('{\"ok\": true}')"];
        let out = try_run(&argv, None, &repo_root, Duration::from_secs(30));
        // python3 may be absent in some CI images; only assert when present.
        if let Ok(bytes) = out {
            let s = String::from_utf8_lossy(&bytes);
            assert!(s.contains("ok"), "unexpected stdout: {s}");
        }
    }

    /// Regression: a child that writes a large volume to stdout AND stderr
    /// WHILE the parent pushes a large (>128KB) stdin payload must not deadlock.
    /// The previous main-thread `write_all(stdin)` blocked filling the child's
    /// stdin pipe while the child blocked filling its stdout/stderr pipes →
    /// classic pipe deadlock that the watchdog couldn't even break. Concurrent
    /// stdin-write + stdout/stderr-drain must let this complete.
    #[test]
    fn large_bidirectional_io_does_not_deadlock() {
        let repo_root = std::env::current_dir().unwrap();
        // Child: read ALL of stdin (only after dumping output), and emit
        // ~256KB to each of stdout/stderr — well past the ~64KB pipe buffer.
        // It echoes a fixed JSON to stdout LAST so the parse path is exercised.
        let script = "\
import sys
sys.stderr.write('E' * 262144)
sys.stderr.flush()
sys.stdout.write('O' * 262144)
sys.stdout.flush()
data = sys.stdin.read()
sys.stdout.write('{\"len\": %d}' % len(data))
";
        // >128KB stdin payload.
        let payload = "x".repeat(200_000);
        let argv = ["python3", "-c", script];
        let start = Instant::now();
        let out = try_run(&argv, Some(&payload), &repo_root, Duration::from_secs(30));
        // python3 may be absent in some CI images; only assert when present.
        if let Ok(bytes) = out {
            let s = String::from_utf8_lossy(&bytes);
            assert!(
                s.contains("\"len\": 200000"),
                "unexpected stdout tail: {}",
                &s[s.len().saturating_sub(40)..]
            );
        }
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "bidirectional IO deadlocked or was too slow: {:?}",
            start.elapsed()
        );
    }

    /// A non-zero exit is reported as a Subprocess error carrying stderr.
    #[test]
    fn nonzero_exit_reports_stderr() {
        let repo_root = std::env::current_dir().unwrap();
        let argv = [
            "python3",
            "-c",
            "import sys; sys.stderr.write('boom'); sys.exit(3)",
        ];
        let out = try_run(&argv, None, &repo_root, Duration::from_secs(30));
        match out {
            Err(MtError::Subprocess { code, stderr, .. }) => {
                assert_eq!(code, Some(3));
                assert!(stderr.contains("boom"), "stderr: {stderr}");
            }
            // python3 absent → spawn error is acceptable in that environment.
            Err(MtError::Io(_)) => {}
            other => panic!("expected Subprocess error, got {other:?}"),
        }
    }
}
