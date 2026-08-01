//! Concurrent anime episode download engine.
//!
//! Uses tokio for async process management. Quality-first mirror selection,
//! concurrent mirror race with measurement interval, per-host semaphore,
//! cancellation via `CancellationToken`. Winner process continues without
//! restart after measurement phase.
//!
//! Robustness layer (task D): per-mirror retry with exponential backoff +
//! jitter, per-host circuit breaker, media validation of downloaded output
//! (with quarantine of invalid files), persistent manifest integration
//! (resume / retry-failed / validate-only run modes), bounded stderr tail,
//! URL/token redaction, RAII child/artifact ownership ([`ChildGuard`],
//! [`TempRegistry`]).

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use tokio::fs::create_dir_all;
#[cfg(test)]
use tokio::fs::read_dir;
use tokio::fs::{remove_file, rename};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::runtime::Handle;
use tokio::sync::{Mutex, Semaphore, broadcast};
use tokio::task::{JoinSet, spawn_blocking};
use tokio::time::{Instant, sleep, timeout};
use tokio_util::sync::CancellationToken;

use crate::download_types::{
    EpEvent, EpisodeInput, Phase, Quality, host_preference_rank, parse_speed_bps,
    quality_height_from_str, try_canonicalize_vk_url,
};
use crate::hosts::{
    ErrorClass, HostAdapter, PermanentKind, RetryableKind, TimeoutProfile, classify,
};
use crate::manifest::{
    AttemptRecord, AttemptStatus, CacheEntry, FinalStatus, Manifest, OutputMeta, sha256_file,
};
use crate::plain_output::iso_timestamp;
// Public re-exports: the CLI bin (anime-dl) constructs `DownloadConfig` with a
// `validation:` field and imports these from `mt_cli::downloader`.
pub use crate::validator::{
    FfprobeValidator, MediaValidator, ValidationConfig, ValidationOutcome,
    cache_key as validation_cache_key,
};

// ── Constants ──────────────────────────────────────────────────────────────

const MEASUREMENT_SECS: u64 = 3;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const STALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Timeout profile for a download URL: known hosts use their per-host profile
/// ([`HostAdapter::timeout_profile`]); generic http(s) hosts and non-http(s)
/// URLs (recognize → `None`) keep the fixed default constants, preserving
/// legacy behavior byte-for-byte.
fn profile_for_url(url: &str) -> TimeoutProfile {
    match HostAdapter::recognize(url) {
        Some(host) => HostAdapter::timeout_profile(host),
        None => TimeoutProfile {
            startup_secs: STARTUP_TIMEOUT.as_secs(),
            stall_secs: STALL_TIMEOUT.as_secs(),
        },
    }
}
const OVERALL_TIMEOUT: Duration = Duration::from_secs(7200);
const DEFAULT_HOST_CONCURRENCY: usize = 1;
const DEFAULT_EP_CONCURRENCY: usize = 4;
const DEFAULT_RETRY_ATTEMPTS: u32 = 3;
const DEFAULT_BACKOFF_BASE_SECS: f64 = 2.0;
const DEFAULT_BACKOFF_CAP_SECS: f64 = 60.0;
const DEFAULT_JITTER_SECS: f64 = 1.0;
const DEFAULT_CIRCUIT_THRESHOLD: u32 = 3;
const DEFAULT_CIRCUIT_COOLDOWN_SECS: u64 = 60;
/// Host-lock wait after which a `MirrorBusy` event is emitted (repeat window).
const MIRROR_BUSY_REPORT_SECS: u64 = 5;
/// Bounded stderr tail kept for classification / diagnostics.
const STDERR_TAIL_LINES: usize = 5;

// ── Run mode & outcome ─────────────────────────────────────────────────────

/// What subset of episodes to process and how.
///
/// `Resume|RetryFailed` (the CLI union) collapses to [`RunMode::Resume`]:
/// retry-failed episodes are a subset of not-complete episodes, so Resume
/// already covers both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    /// Reconcile: validate existing outputs, download episodes lacking a
    /// valid output, retry failed ones per retry policy.
    Default,
    /// Process only episodes whose manifest status is not `Complete`.
    Resume,
    /// Process only episodes whose manifest status is `Failed`.
    RetryFailed,
    /// Validate existing outputs only; never download.
    ValidateOnly,
}

/// Aggregate result of a full engine run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    pub downloaded: u32,
    pub skipped: u32,
    pub failed: u32,
    pub cancelled: bool,
    /// Episodes that ended the run without a valid output file on disk.
    pub missing_episodes: Vec<u32>,
    /// Failure/cancellation reasons per episode (episode, reason).
    pub per_episode_reasons: Vec<(u32, String)>,
}

impl Outcome {
    /// Map to process exit code per contract:
    /// 0 all ok; 3 partial (some failed, some ok); 4 all failed; 130 cancelled.
    pub fn exit_code(&self) -> i32 {
        if self.cancelled {
            130
        } else if self.failed > 0 && self.downloaded + self.skipped > 0 {
            3
        } else if self.failed > 0 {
            4
        } else {
            0
        }
    }
}

// ── Download configuration ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub episode_concurrency: usize,
    pub host_concurrency: usize,
    pub measurement_secs: u64,
    pub out_dir: PathBuf,
    pub slug: String,
    // ── robustness (task D) ───────────────────────────────────────────────
    pub run_mode: RunMode,
    /// Retries per mirror (on top of the initial attempt). Default 3.
    pub retry_attempts: u32,
    /// Backoff base in seconds; actual = min(base * 2^attempt, cap) + jitter.
    pub backoff_base_secs: f64,
    pub backoff_cap_secs: f64,
    /// Random jitter in seconds added to each backoff (0..jitter).
    pub jitter_secs: f64,
    /// Consecutive systemic failures that open a host circuit. Default 3.
    pub circuit_threshold: u32,
    /// How long a host circuit stays open. Default 60s.
    pub circuit_cooldown_secs: u64,
    /// Media validation tuning (size/duration/ffprobe timeout).
    pub validation: ValidationConfig,
    /// Skip ffprobe-based validation entirely; downloaded outputs are accepted
    /// on the legacy extension+size heuristic ([`is_valid_output`]) instead.
    /// Valid media that passes the heuristic is still never overwritten, and
    /// the ffprobe validation cache is never consulted nor written in this
    /// mode (no cache poisoning for later default runs).
    pub no_validate: bool,
    /// Bypass the validation cache: always re-probe via [`MediaValidator`] even
    /// when a size+mtime cache entry would otherwise hit.
    pub validate_force: bool,
    /// Extra arguments appended to EVERY yt-dlp invocation (measure, download,
    /// resume) after the engine's standard arguments and before the URL.
    /// Evidence-based knob for hosts that need per-host yt-dlp tuning.
    pub ytdlp_extra_args: Vec<String>,
    /// Optional persistent manifest path; `None` disables manifest integration.
    pub manifest_path: Option<PathBuf>,
    /// Path to the source JSON episode list the run was started from. When
    /// set, the manifest's [`InputIdentity`] is populated at run start:
    /// sha256 streamed from this file plus the path itself.
    pub input_source_path: Option<PathBuf>,
    /// `resolved_at` from the source input (ISO8601), when present. Carried
    /// into the manifest's [`InputIdentity`]; `None` when absent.
    pub input_resolved_at: Option<String>,
    /// Where invalid downloads are moved. Relative paths resolve under out_dir.
    pub quarantine_dir: PathBuf,
    /// Also delete quarantined files instead of keeping them.
    pub clean_invalid: bool,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            episode_concurrency: DEFAULT_EP_CONCURRENCY,
            host_concurrency: DEFAULT_HOST_CONCURRENCY,
            measurement_secs: MEASUREMENT_SECS,
            out_dir: PathBuf::from("."),
            slug: "anime".to_string(),
            run_mode: RunMode::Default,
            retry_attempts: DEFAULT_RETRY_ATTEMPTS,
            backoff_base_secs: DEFAULT_BACKOFF_BASE_SECS,
            backoff_cap_secs: DEFAULT_BACKOFF_CAP_SECS,
            jitter_secs: DEFAULT_JITTER_SECS,
            circuit_threshold: DEFAULT_CIRCUIT_THRESHOLD,
            circuit_cooldown_secs: DEFAULT_CIRCUIT_COOLDOWN_SECS,
            validation: ValidationConfig::default(),
            no_validate: false,
            validate_force: false,
            ytdlp_extra_args: Vec::new(),
            manifest_path: None,
            input_source_path: None,
            input_resolved_at: None,
            quarantine_dir: PathBuf::from(".quarantine"),
            clean_invalid: false,
        }
    }
}

/// Resolve the effective quarantine directory (relative → under out_dir).
pub(crate) fn effective_quarantine_dir(config: &DownloadConfig) -> PathBuf {
    if config.quarantine_dir.is_relative() {
        config.out_dir.join(&config.quarantine_dir)
    } else {
        config.quarantine_dir.clone()
    }
}

// ── RAII ownership ─────────────────────────────────────────────────────────

/// Owns a running subprocess; kills the process group and reaps on drop.
///
/// Drop order matters on the engine error/cancel path: child guards are
/// dropped (kill + reap) before the temp registry removes artifacts, so a
/// still-running child can never repopulate a file we just cleaned.
pub struct ChildGuard {
    child: Option<RunningSubprocess>,
}

impl ChildGuard {
    pub fn new(proc: RunningSubprocess) -> Self {
        Self { child: Some(proc) }
    }

    pub fn child_mut(&mut self) -> &mut RunningSubprocess {
        self.child.as_mut().expect("ChildGuard child missing")
    }

    /// Take the subprocess out without killing it (e.g. after a clean exit).
    pub fn take(&mut self) -> Option<RunningSubprocess> {
        self.child.take()
    }

    /// Consume the guard, yielding the live subprocess (winner continuation).
    pub fn into_inner(mut self) -> RunningSubprocess {
        self.child.take().expect("ChildGuard child missing")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut proc) = self.child.take() {
            proc.kill_group();
            // Reap asynchronously when inside a runtime (no zombies); otherwise
            // poll `try_wait` briefly.
            if Handle::try_current().is_ok() {
                let mut child = proc.child;
                tokio::spawn(async move {
                    let _ = child.wait().await;
                });
            } else {
                let _ = proc.child.start_kill();
                for _ in 0..100 {
                    if proc.child.try_wait().ok().flatten().is_some() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        }
    }
}

/// Registry of tool-owned temp artifacts; removes them all on drop.
///
/// Files are registered per created artifact (`{stem}.meas*`, `{stem}.*.part`,
/// `{stem}.part`, yt-dlp fragment files). `promote` marks a path as final
/// (never deleted). Drop removes every still-registered path. Unregistered
/// paths are never touched.
#[derive(Debug, Default)]
pub struct TempRegistry {
    paths: HashSet<PathBuf>,
}

impl TempRegistry {
    pub fn new() -> Self {
        Self {
            paths: HashSet::new(),
        }
    }

    pub fn register(&mut self, path: PathBuf) {
        self.paths.insert(path);
    }

    /// Promote a path out of the registry (successful final artifact).
    pub fn promote(&mut self, path: &Path) {
        self.paths.remove(path);
    }

    pub fn is_registered(&self, path: &Path) -> bool {
        self.paths.contains(path)
    }

    /// Scan the parent directory for tool-owned artifacts sharing `stem`'s
    /// file-name prefix and register them. Skips files that look like valid
    /// final media (valid media is never deleted). Call before drop on
    /// failure/cancel paths to catch late-appearing artifacts.
    pub fn register_prefix_artifacts(&mut self, stem: &Path) {
        let Some(parent) = stem.parent() else {
            return;
        };
        let Some(name) = stem.file_name() else {
            return;
        };
        let prefix = format!("{}.", name.to_string_lossy());
        let Ok(entries) = fs::read_dir(parent) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(fname) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !fname.starts_with(&prefix) {
                continue;
            }
            // Final-looking files (no .part/.meas/fragment markers) that pass
            // the size heuristic are treated as media and never registered.
            let rest = &fname[prefix.len()..];
            let is_fragment = rest.starts_with('f')
                && rest.chars().take_while(|c| c.is_ascii_digit()).count() > 0
                && rest[rest.chars().take_while(|c| c.is_ascii_digit()).count()..].starts_with('.');
            let final_like = !fname.contains(".part") && !fname.contains(".meas") && !is_fragment;
            if final_like && is_valid_output(&path) {
                continue;
            }
            self.paths.insert(path);
        }
    }
}

impl Drop for TempRegistry {
    fn drop(&mut self) {
        for p in &self.paths {
            let _ = fs::remove_file(p);
        }
    }
}

// ── Process group helpers ──────────────────────────────────────────────────

#[cfg(unix)]
fn kill_process_group(pid: u32) {
    // SAFETY: pid is the process group ID obtained from child.id() after process_group(0).
    // Negative pid sends signal to the entire process group. libc::kill is async-signal-safe
    // and the only way to send signals to child process groups on Unix.
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) {}

// ── Injectable subprocess trait ────────────────────────────────────────────

/// Abstract subprocess spawning for testability.
pub trait SubprocessFactory: Send + Sync + 'static {
    fn spawn_measure(
        &self,
        url: &str,
        out_path: &Path,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>>;
    fn spawn_download(
        &self,
        url: &str,
        out_path: &Path,
        continue_part: bool,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>>;
    /// Inspect available formats for a URL and return best quality.
    fn inspect_formats(
        &self,
        url: &str,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<Quality>> + Send>>;
}

/// A running subprocess with process group support.
pub struct RunningSubprocess {
    pub child: Child,
    pub pgid: u32,
}

impl RunningSubprocess {
    /// Kill child process group.
    pub fn kill_group(&mut self) {
        if self.pgid > 0 {
            kill_process_group(self.pgid);
        }
        let _ = self.child.start_kill();
    }
}

/// Real yt-dlp subprocess factory.
///
/// `extra_args` are appended to every yt-dlp invocation after the engine's
/// standard arguments and before the URL. They come from
/// [`DownloadConfig::ytdlp_extra_args`] via [`DownloadEngine::new`]; the
/// [`SubprocessFactory`] trait signature stays stable (no per-call args).
#[derive(Default)]
pub struct RealYtDlpFactory {
    pub extra_args: Vec<String>,
}

impl RealYtDlpFactory {
    /// Standard measure args + `extra_args` + url (pure; unit-tested).
    fn measure_args(&self, stem_str: &str, url: &str) -> Vec<String> {
        let mut args = vec![
            "--progress".into(),
            "--newline".into(),
            "-f".into(),
            "bv*+ba/b".into(),
            "-o".into(),
            format!("{stem_str}.%(ext)s"),
            "--merge-output-format".into(),
            "mkv".into(),
        ];
        args.extend(self.extra_args.iter().cloned());
        args.push(url.to_string());
        args
    }

    /// Standard download args (+ `-c` on part-continue) + `extra_args` + url.
    fn download_args(&self, stem_str: &str, continue_part: bool, url: &str) -> Vec<String> {
        let mut args = vec![
            "--progress".into(),
            "--newline".into(),
            "--force-overwrites".into(),
            "-f".into(),
            "bv*+ba/b".into(),
            "-o".into(),
            format!("{stem_str}.%(ext)s"),
            "--merge-output-format".into(),
            "mkv".into(),
        ];
        if continue_part {
            args.push("-c".into());
        }
        args.extend(self.extra_args.iter().cloned());
        args.push(url.to_string());
        args
    }

    /// Standard inspect args + `extra_args` + url.
    fn inspect_args(&self, url: &str) -> Vec<String> {
        let mut args = vec![
            "--dump-json".into(),
            "-f".into(),
            "bv*+ba/b".into(),
            "--no-download".into(),
        ];
        args.extend(self.extra_args.iter().cloned());
        args.push(url.to_string());
        args
    }
}

impl SubprocessFactory for RealYtDlpFactory {
    fn spawn_measure(
        &self,
        url: &str,
        out_path: &Path,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>> {
        let url = url.to_string();
        let stem_str = out_path.to_string_lossy().to_string();
        let args = self.measure_args(&stem_str, &url);
        Box::pin(async move {
            let mut cmd = Command::new("yt-dlp");
            #[cfg(unix)]
            {
                cmd.process_group(0);
            }
            cmd.args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            let child = cmd.spawn().ok()?;
            let pgid = child.id().unwrap_or(0);
            Some(RunningSubprocess { child, pgid })
        })
    }

    fn spawn_download(
        &self,
        url: &str,
        out_path: &Path,
        continue_part: bool,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>> {
        let url = url.to_string();
        let stem = out_path.with_extension("");
        let stem_str = stem.to_string_lossy().to_string();
        let args = self.download_args(&stem_str, continue_part, &url);
        Box::pin(async move {
            let mut cmd = Command::new("yt-dlp");
            #[cfg(unix)]
            {
                cmd.process_group(0);
            }
            cmd.args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            let child = cmd.spawn().ok()?;
            let pgid = child.id().unwrap_or(0);
            Some(RunningSubprocess { child, pgid })
        })
    }

    fn inspect_formats(
        &self,
        url: &str,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<Quality>> + Send>> {
        let url = url.to_string();
        let args = self.inspect_args(&url);
        Box::pin(async move {
            let mut cmd = Command::new("yt-dlp");
            #[cfg(unix)]
            {
                cmd.process_group(0);
            }
            cmd.args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let child = cmd.spawn().ok()?;
            let pgid = child.id().unwrap_or(0);

            let profile = profile_for_url(&url);
            let result = timeout(
                Duration::from_secs(profile.startup_secs),
                child.wait_with_output(),
            )
            .await;

            match result {
                Ok(Ok(output)) if output.status.success() => {
                    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
                    let height = json.get("height")?.as_u64()? as u32;
                    Some(Quality::new(height))
                }
                _ => {
                    if pgid > 0 {
                        kill_process_group(pgid);
                    }
                    None
                }
            }
        })
    }
}

// ── Fake factory for tests ─────────────────────────────────────────────────

#[cfg(test)]
pub mod test_factory {
    use std::collections::{HashMap, VecDeque};
    use std::path::Path;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    /// Programmable per-URL outcome for a fake yt-dlp download spawn.
    #[derive(Debug, Clone)]
    pub enum FakeOutcome {
        /// Write a completed `{out}.mkv` (≥1 MiB) and exit 0. `bad` content
        /// marker (`BAD!`) makes the injected fake validator reject the file.
        Success { bad: bool },
        /// Fail with retryable stderr (`Timed out`) `n` times, then `Success`.
        Transient(u32),
        /// Fail with permanent stderr (`HTTP Error 403`). Never retried.
        Permanent,
        /// Fail with permanent stderr (`HTTP Error 404`). Never retried.
        Permanent404,
        /// Write `{out}.mkv.part` progressively over `total_secs`, then
        /// complete. Used for cancellation-during-download tests.
        SlowPart { total_secs: f64 },
        /// Single retryable failure (internal, produced by `Transient` countdown).
        FailRetryable,
    }

    /// Fake subprocess factory with per-URL programmable outcome queues.
    #[derive(Default)]
    pub struct FakeFactory {
        pub fail_measure: AtomicBool,
        pub fail_download: AtomicBool,
        pub fake_quality: AtomicBool,
        /// If > 0, every spawned download child writes this many stderr lines
        /// before its normal behavior (flood test).
        pub stderr_flood_lines: AtomicUsize,
        /// If `Some(line)`, the spawned child prints this line to stderr and
        /// exits 1, overriding all other outcomes (redaction tests).
        pub error_line_override: std::sync::Mutex<Option<String>>,
        outcomes: std::sync::Mutex<HashMap<String, VecDeque<FakeOutcome>>>,
        /// Records whether the last spawned download was asked to continue a
        /// partial file (`-c`).
        pub continue_flag_seen: Arc<AtomicBool>,
        spawn_counts: std::sync::Mutex<HashMap<String, usize>>,
    }

    impl FakeFactory {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn set_outcomes(&self, url: &str, outcomes: Vec<FakeOutcome>) {
            self.outcomes
                .lock()
                .expect("outcomes lock")
                .insert(url.to_string(), outcomes.into_iter().collect());
        }

        /// Pop the next outcome for `url`; `Transient(n)` decrements in place.
        fn next_outcome(&self, url: &str) -> FakeOutcome {
            let mut map = self.outcomes.lock().expect("outcomes lock");
            let q = map.entry(url.to_string()).or_default();
            match q.front().cloned() {
                Some(FakeOutcome::Transient(n)) if n > 0 => {
                    *q.front_mut().expect("front") = FakeOutcome::Transient(n - 1);
                    FakeOutcome::FailRetryable
                }
                Some(FakeOutcome::Transient(0)) => {
                    q.pop_front();
                    FakeOutcome::Success { bad: false }
                }
                Some(other) => {
                    q.pop_front();
                    other
                }
                // Default: write a completed file (fast path for engine tests
                // that don't program failures).
                None => FakeOutcome::Success { bad: false },
            }
        }

        pub fn spawn_count(&self, url: &str) -> usize {
            self.spawn_counts
                .lock()
                .expect("spawn counts lock")
                .get(url)
                .copied()
                .unwrap_or(0)
        }

        pub fn total_spawns(&self) -> usize {
            self.spawn_counts
                .lock()
                .expect("spawn counts lock")
                .values()
                .sum()
        }

        fn spawn_sh(script: String) -> Option<RunningSubprocess> {
            let mut cmd = Command::new("sh");
            cmd.arg("-c")
                .arg(&script)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let child = cmd.spawn().ok()?;
            let pgid = child.id().unwrap_or(0);
            Some(RunningSubprocess { child, pgid })
        }
    }

    impl SubprocessFactory for FakeFactory {
        fn spawn_measure(
            &self,
            _url: &str,
            _out_path: &Path,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>> {
            let fail = self.fail_measure.load(Ordering::SeqCst);
            Box::pin(async move {
                if fail {
                    return None;
                }
                Self::spawn_sh("echo measure".to_string())
            })
        }

        fn spawn_download(
            &self,
            url: &str,
            out_path: &Path,
            continue_part: bool,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>> {
            if continue_part {
                self.continue_flag_seen.store(true, Ordering::SeqCst);
            }
            *self
                .spawn_counts
                .lock()
                .expect("spawn counts lock")
                .entry(url.to_string())
                .or_insert(0) += 1;
            if self.fail_download.load(Ordering::SeqCst) {
                return Box::pin(async move { None });
            }
            let outcome = self.next_outcome(url);
            let flood = self.stderr_flood_lines.load(Ordering::SeqCst);
            let override_line = self
                .error_line_override
                .lock()
                .expect("override lock")
                .clone();
            let stem_str = out_path.to_string_lossy().to_string();
            Box::pin(async move {
                let flood_prefix = if flood > 0 {
                    format!("for i in $(seq 1 {flood}); do echo \"flood line $i\" >&2; done; ")
                } else {
                    String::new()
                };
                let script = if let Some(line) = override_line {
                    format!("{flood_prefix}echo '{line}' >&2; exit 1")
                } else {
                    match outcome {
                        FakeOutcome::Success { bad } => {
                            let marker = if bad { "BAD!" } else { "GOOD" };
                            format!(
                                "printf '{marker}' > '{stem_str}.mkv'; head -c 1048576 /dev/zero >> '{stem_str}.mkv'"
                            )
                        }
                        FakeOutcome::FailRetryable => {
                            format!("{flood_prefix}echo 'ERROR: [test] Timed out' >&2; exit 1")
                        }
                        FakeOutcome::Permanent => {
                            format!(
                                "{flood_prefix}echo 'ERROR: HTTP Error 403: Forbidden' >&2; exit 1"
                            )
                        }
                        FakeOutcome::Permanent404 => {
                            format!(
                                "{flood_prefix}echo 'ERROR: HTTP Error 404: Not Found' >&2; exit 1"
                            )
                        }
                        FakeOutcome::SlowPart { total_secs } => {
                            let n = ((total_secs * 10.0).max(1.0)) as usize;
                            let dt = total_secs / n as f64;
                            format!(
                                "out='{stem_str}.mkv.part'; rm -f \"$out\"; \
                                 for i in $(seq 1 {n}); do head -c 1048576 /dev/zero >> \"$out\"; sleep {dt}; done; \
                                 mv \"$out\" '{stem_str}.mkv'"
                            )
                        }
                        // Transient is consumed by next_outcome's countdown;
                        // a raw occurrence here degenerates to a retryable fail.
                        FakeOutcome::Transient(_) => {
                            format!("{flood_prefix}echo 'ERROR: [test] Timed out' >&2; exit 1")
                        }
                    }
                };
                Self::spawn_sh(script)
            })
        }

        fn inspect_formats(
            &self,
            _url: &str,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<Quality>> + Send>> {
            let has_quality = self.fake_quality.load(Ordering::SeqCst);
            Box::pin(async move {
                if has_quality {
                    Some(Quality::new(1080))
                } else {
                    None
                }
            })
        }
    }
}

// ── Per-run state ──────────────────────────────────────────────────────────

/// Circuit breaker state for one host.
#[derive(Debug, Default)]
struct CircuitState {
    consecutive_failures: u32,
    open_until: Option<Instant>,
}

/// State shared by all episode tasks of one engine run.
struct RunContext {
    manifest: Mutex<Option<Manifest>>,
    circuits: Mutex<HashMap<String, CircuitState>>,
}

// ── Download engine ────────────────────────────────────────────────────────

/// The anime download engine.
#[derive(Clone)]
pub struct DownloadEngine {
    pub config: DownloadConfig,
    pub cancel: CancellationToken,
    factory: Arc<dyn SubprocessFactory>,
    validator: Arc<dyn MediaValidator + Send + Sync>,
    host_semaphores: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
}

impl DownloadEngine {
    pub fn new(config: DownloadConfig) -> Self {
        let extra_args = config.ytdlp_extra_args.clone();
        Self {
            cancel: CancellationToken::new(),
            host_semaphores: Arc::new(Mutex::new(HashMap::new())),
            factory: Arc::new(RealYtDlpFactory { extra_args }),
            validator: Arc::new(FfprobeValidator::new()),
            config,
        }
    }

    pub fn with_factory(config: DownloadConfig, factory: Arc<dyn SubprocessFactory>) -> Self {
        Self {
            cancel: CancellationToken::new(),
            host_semaphores: Arc::new(Mutex::new(HashMap::new())),
            factory,
            validator: Arc::new(FfprobeValidator::new()),
            config,
        }
    }

    pub fn with_factory_and_validator(
        config: DownloadConfig,
        factory: Arc<dyn SubprocessFactory>,
        validator: Arc<dyn MediaValidator + Send + Sync>,
    ) -> Self {
        Self {
            cancel: CancellationToken::new(),
            host_semaphores: Arc::new(Mutex::new(HashMap::new())),
            factory,
            validator,
            config,
        }
    }

    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    async fn host_semaphore(&self, host: &str) -> Arc<Semaphore> {
        let mut map = self.host_semaphores.lock().await;
        // Clamp to 1: host_concurrency=0 would create a permanently-blocked
        // semaphore, causing every mirror to emit MirrorBusy.
        let permits = self.config.host_concurrency.max(1);
        map.entry(host.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(permits)))
            .clone()
    }

    fn host_from_url(url: &str) -> String {
        url.trim_start_matches("https://")
            .trim_start_matches("http://")
            .split('/')
            .next()
            .unwrap_or("unknown")
            .to_string()
    }

    /// Acquire a host semaphore, emitting `MirrorBusy` every 5 s while blocked.
    /// Returns `None` when cancelled or the semaphore is closed.
    async fn acquire_host_semaphore(
        &self,
        host: &str,
        episode: i64,
        tx: &broadcast::Sender<EpEvent>,
    ) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let sem = self.host_semaphore(host).await;
        let mut waited = 0u64;
        loop {
            if self.cancel.is_cancelled() {
                return None;
            }
            match timeout(
                Duration::from_secs(MIRROR_BUSY_REPORT_SECS),
                sem.clone().acquire_owned(),
            )
            .await
            {
                Ok(Ok(permit)) => return Some(permit),
                Ok(Err(_)) => return None,
                Err(_) => {
                    waited += MIRROR_BUSY_REPORT_SECS;
                    let _ = tx.send(EpEvent::MirrorBusy {
                        ep: episode,
                        host: host.to_string(),
                        wait_secs: waited,
                    });
                }
            }
        }
    }

    // ── Circuit breaker ───────────────────────────────────────────────────

    /// Remaining cooldown if the host circuit is open (closing it lazily once
    /// expired). `None` = circuit closed, host usable.
    async fn circuit_cooldown_remaining(
        &self,
        host: &str,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
    ) -> Option<u64> {
        let mut circuits = ctx.circuits.lock().await;
        let st = circuits.entry(host.to_string()).or_default();
        if let Some(until) = st.open_until {
            if Instant::now() >= until {
                st.open_until = None;
                st.consecutive_failures = 0;
                let _ = tx.send(EpEvent::CircuitClosed {
                    host: host.to_string(),
                });
                None
            } else {
                Some(until.saturating_duration_since(Instant::now()).as_secs())
            }
        } else {
            None
        }
    }

    async fn note_systemic_failure(
        &self,
        host: &str,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
    ) {
        let mut circuits = ctx.circuits.lock().await;
        let st = circuits.entry(host.to_string()).or_default();
        if st.open_until.is_some() {
            return;
        }
        st.consecutive_failures += 1;
        if st.consecutive_failures >= self.config.circuit_threshold {
            st.open_until =
                Some(Instant::now() + Duration::from_secs(self.config.circuit_cooldown_secs));
            let _ = tx.send(EpEvent::CircuitOpened {
                host: host.to_string(),
            });
        }
    }

    async fn note_success(&self, host: &str, ctx: &Arc<RunContext>) {
        let mut circuits = ctx.circuits.lock().await;
        if let Some(st) = circuits.get_mut(host) {
            st.consecutive_failures = 0;
        }
    }

    // ── Retry / classification ────────────────────────────────────────────

    /// Systemic failures count toward the host circuit breaker; URL-specific
    /// failures never do.
    fn is_systemic(class: &ErrorClass) -> bool {
        matches!(
            class,
            ErrorClass::Retryable(
                RetryableKind::Timeout
                    | RetryableKind::Dns
                    | RetryableKind::Connect
                    | RetryableKind::Http5xx
                    | RetryableKind::Http429
                    | RetryableKind::Stall
            )
        )
    }

    /// Short, sanitized reason for a failure class. Unknown failures fall back
    /// to the first stderr error line with URLs redacted.
    fn failure_reason(class: &ErrorClass, stderr_tail: &[String]) -> String {
        match class {
            ErrorClass::Retryable(RetryableKind::Dns) => "dns resolution failure".into(),
            ErrorClass::Retryable(RetryableKind::Connect) => "connection refused/reset".into(),
            ErrorClass::Retryable(RetryableKind::Timeout) => "timeout".into(),
            ErrorClass::Retryable(RetryableKind::Http429) => "http 429 rate limited".into(),
            ErrorClass::Retryable(RetryableKind::Http5xx) => "http 5xx server error".into(),
            ErrorClass::Retryable(RetryableKind::Stall) => "stalled download".into(),
            ErrorClass::Retryable(RetryableKind::ExtractNotReady) => "extractor not ready".into(),
            ErrorClass::Permanent(PermanentKind::Http403) => "http 403 forbidden".into(),
            ErrorClass::Permanent(PermanentKind::Http404) => "http 404 not found".into(),
            ErrorClass::Permanent(PermanentKind::UnsupportedUrl) => "unsupported url".into(),
            ErrorClass::Permanent(PermanentKind::AuthChallenge) => "auth challenge".into(),
            ErrorClass::Permanent(PermanentKind::FormatNotAvailable) => {
                "format not available".into()
            }
            ErrorClass::Permanent(PermanentKind::InvalidInput) => "invalid input".into(),
            ErrorClass::Unknown => stderr_tail
                .iter()
                .find(|l| l.to_ascii_lowercase().contains("error:"))
                .or_else(|| stderr_tail.first())
                .map(|l| redact_urls(l))
                .unwrap_or_else(|| "unknown failure".into()),
        }
    }

    fn backoff_for(&self, retry: u32, url: &str, episode: i64) -> f64 {
        let exp = self.config.backoff_base_secs * 2f64.powi(retry as i32);
        let capped = exp.min(self.config.backoff_cap_secs);
        let seed = (episode as u64)
            .wrapping_mul(31)
            .wrapping_add(
                url.bytes()
                    .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64)),
            )
            .wrapping_add(retry as u64);
        capped + self.jitter(seed)
    }

    /// Deterministic pseudo-jitter in `0..jitter_secs` (no external RNG dep).
    fn jitter(&self, seed: u64) -> f64 {
        if self.config.jitter_secs <= 0.0 {
            return 0.0;
        }
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let frac = (x >> 33) as f64 / (1u64 << 32) as f64;
        frac * self.config.jitter_secs
    }

    /// Resume continuation decision: same URL as the last manifest attempt AND
    /// a `.part` file present on disk. Part-continue via yt-dlp `-c` exists
    /// ONLY for [`RunMode::Resume`] from an interrupted manifest; default runs
    /// always start fresh (the winner-continuation path handles live
    /// downloads without touching `.part` files).
    async fn should_continue_part(
        &self,
        episode: i64,
        url: &str,
        stem: &Path,
        ctx: &Arc<RunContext>,
    ) -> bool {
        if !matches!(self.config.run_mode, RunMode::Resume) {
            return false;
        }
        if find_part_file(stem).is_none() {
            return false;
        }
        let guard = ctx.manifest.lock().await;
        let Some(m) = guard.as_ref() else {
            return false;
        };
        let Some(rec) = m.episodes.iter().find(|r| r.episode == episode as u32) else {
            return false;
        };
        rec.attempts.last().map(|a| a.url == url).unwrap_or(false)
    }

    // ── Manifest helpers ──────────────────────────────────────────────────

    async fn save_manifest(&self, ctx: &Arc<RunContext>) {
        let guard = ctx.manifest.lock().await;
        if let (Some(m), Some(path)) = (guard.as_ref(), self.config.manifest_path.as_ref()) {
            let _ = m.save_atomic(path);
        }
    }

    async fn manifest_begin_episode(&self, episode: i64, ctx: &Arc<RunContext>) {
        let mut guard = ctx.manifest.lock().await;
        if let Some(m) = guard.as_mut() {
            m.ensure_episode(episode as u32);
            m.set_final_status(episode as u32, FinalStatus::InProgress);
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn record_attempt(
        &self,
        episode: i64,
        mirror_idx: usize,
        host: &str,
        url: &str,
        ok: bool,
        reason: Option<String>,
        bytes: u64,
        secs: f64,
        ctx: &Arc<RunContext>,
    ) {
        let mut guard = ctx.manifest.lock().await;
        if let Some(m) = guard.as_mut() {
            m.ensure_episode(episode as u32);
            m.record_attempt(
                episode as u32,
                AttemptRecord {
                    mirror_idx,
                    host: Some(host.to_string()),
                    url: url.to_string(),
                    status: if ok {
                        AttemptStatus::Ok
                    } else {
                        AttemptStatus::Failed
                    },
                    reason,
                    bytes_downloaded: bytes,
                    secs,
                    started_at: Some(iso_timestamp()),
                },
            );
        }
    }

    async fn manifest_set_output(
        &self,
        episode: i64,
        path: &Path,
        size: u64,
        sha256: Option<String>,
        outcome: &ValidationOutcome,
        ctx: &Arc<RunContext>,
    ) {
        let mut guard = ctx.manifest.lock().await;
        if let Some(m) = guard.as_mut() {
            m.ensure_episode(episode as u32);
            m.set_output(
                episode as u32,
                OutputMeta {
                    path: path.to_path_buf(),
                    size,
                    sha256,
                    validated: true,
                    ffprobe_version: outcome.ffprobe_version.clone(),
                    checked_at: Some(iso_timestamp()),
                },
            );
            m.set_final_status(episode as u32, FinalStatus::Complete);
        }
    }

    async fn manifest_set_failed(&self, episode: i64, ctx: &Arc<RunContext>) {
        let mut guard = ctx.manifest.lock().await;
        if let Some(m) = guard.as_mut() {
            m.ensure_episode(episode as u32);
            m.set_final_status(episode as u32, FinalStatus::Failed);
        }
    }

    /// Startup reconcile: validate every requested episode's existing output
    /// (cache-aware — unchanged files skip the ffprobe probe) and record
    /// `Complete` + validated output metadata in the manifest. This is what
    /// makes the manifest an accurate per-episode record: a `--resume` run
    /// fast-paths episodes whose output validated OK (via the manifest status
    /// AND the validation cache) and re-probes when size/mtime changed.
    ///
    /// Invalid present artifacts are never recorded: the episode stays
    /// `Pending`/`Failed` (eligible for download, quarantined by the
    /// per-episode path). A previously-`Complete` episode whose file is now
    /// invalid is demoted to `Pending` so resume re-downloads it. No mirror
    /// attempts are fabricated — the attempts array only records real mirror
    /// downloads. Quiet: no validation events (the per-episode path emits
    /// those when it actually processes an episode).
    async fn reconcile_existing_outputs(&self, episodes: &[EpisodeInput], ctx: &Arc<RunContext>) {
        // Validate first, outside the manifest lock (validator runs on
        // spawn_blocking; holding the lock across an await is a deadlock
        // risk for other manifest writers).
        let mut validated: Vec<(u32, PathBuf, ValidationOutcome)> = Vec::new();
        for ep in episodes {
            let stem = self
                .config
                .out_dir
                .join(format!("{}-E{:02}", self.config.slug, ep.episode));
            if let Some(existing) = existing_download(&stem, self.heuristic_min_bytes()) {
                let v = self.validate_media_quiet(&existing, ctx).await;
                validated.push((ep.episode as u32, existing, v));
            }
        }
        if validated.is_empty() {
            return;
        }
        let mut guard = ctx.manifest.lock().await;
        let Some(m) = guard.as_mut() else {
            return;
        };
        for (ep, path, v) in validated {
            let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if v.valid {
                // Reuse the previously-recorded sha256 when the output is
                // unchanged (validation-cache hit): avoid re-hashing every
                // resume run. Fresh/unknown outputs are hashed streamed.
                let prev_sha = m
                    .episodes
                    .iter()
                    .find(|r| r.episode == ep)
                    .and_then(|r| r.output.as_ref())
                    .filter(|o| o.validated && o.path == path && o.size == size)
                    .and_then(|o| o.sha256.clone());
                let sha256 = match prev_sha {
                    Some(sha) => Some(sha),
                    None => sha256_file(&path).ok(),
                };
                m.ensure_episode(ep);
                m.set_output(
                    ep,
                    OutputMeta {
                        path,
                        size,
                        sha256,
                        validated: true,
                        ffprobe_version: v.ffprobe_version.clone(),
                        checked_at: Some(iso_timestamp()),
                    },
                );
                m.set_final_status(ep, FinalStatus::Complete);
            } else if m
                .episodes
                .iter()
                .any(|r| r.episode == ep && r.final_status == FinalStatus::Complete)
            {
                // Corrupted/truncated file: episode is eligible again.
                m.set_final_status(ep, FinalStatus::Pending);
            }
        }
    }

    // ── Validation ────────────────────────────────────────────────────────

    async fn run_validator(&self, path: &Path) -> ValidationOutcome {
        let validator = self.validator.clone();
        let cfg = self.config.validation.clone();
        let path = path.to_path_buf();
        match spawn_blocking(move || validator.validate(&path, &cfg)).await {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => ValidationOutcome {
                valid: false,
                reason: Some(format!("validation error: {e}")),
                video_stream: false,
                audio_stream: false,
                duration_secs: None,
                width: None,
                height: None,
                ffprobe_version: None,
            },
            Err(e) => ValidationOutcome {
                valid: false,
                reason: Some(format!("validator task failed: {e}")),
                video_stream: false,
                audio_stream: false,
                duration_secs: None,
                width: None,
                height: None,
                ffprobe_version: None,
            },
        }
    }

    async fn validation_cache_get(&self, path: &Path, ctx: &Arc<RunContext>) -> Option<CacheEntry> {
        let key = validation_cache_key(path)?;
        let guard = ctx.manifest.lock().await;
        let m = guard.as_ref()?;
        let pstr = path.to_string_lossy().to_string();
        m.cache_get(&pstr, key.0, key.1).cloned()
    }

    async fn validation_cache_put(
        &self,
        path: &Path,
        out: &ValidationOutcome,
        ctx: &Arc<RunContext>,
    ) {
        let Some(key) = validation_cache_key(path) else {
            return;
        };
        let mut guard = ctx.manifest.lock().await;
        if let Some(m) = guard.as_mut() {
            m.cache_put(
                path.to_string_lossy().to_string(),
                CacheEntry {
                    size: key.0,
                    mtime_ns: key.1,
                    ok: out.valid,
                    reason: out.reason.clone(),
                    ffprobe_version: out.ffprobe_version.clone(),
                    checked_at: Some(iso_timestamp()),
                },
            );
        }
    }

    /// Effective minimum download size for the acceptance heuristics: the
    /// configured ffprobe floor (`--min-size-mb`), never below the legacy
    /// 1 MiB floor. Default config → exactly [`MIN_VALID_DOWNLOAD_BYTES`], so
    /// default runs behave identically to the pre-config heuristic.
    fn heuristic_min_bytes(&self) -> u64 {
        self.config
            .validation
            .min_size_bytes
            .max(MIN_VALID_DOWNLOAD_BYTES)
    }

    /// Legacy extension+size heuristic outcome, used when `no_validate` is set.
    /// The ffprobe cache is deliberately not consulted nor written: heuristic
    /// results must never satisfy a later default-mode cache hit. Applies the
    /// effective size floor (`--min-size-mb` lifted, never below 1 MiB) so the
    /// heuristic accepts exactly what the ffprobe validator would.
    pub(crate) fn heuristic_outcome(&self, path: &Path) -> ValidationOutcome {
        let ok = is_valid_output_with_min(path, self.heuristic_min_bytes());
        ValidationOutcome {
            valid: ok,
            reason: Some(if ok {
                "validation skipped (--no-validate); extension+size heuristic pass".into()
            } else {
                "invalid media: extension+size heuristic failed".into()
            }),
            video_stream: false,
            audio_stream: false,
            duration_secs: None,
            width: None,
            height: None,
            ffprobe_version: None,
        }
    }

    /// Cache-aware validation without events. Shared by the loud per-episode
    /// [`Self::validate_media`] and the quiet startup reconcile pass
    /// ([`Self::reconcile_existing_outputs`]).
    async fn validate_media_inner(&self, path: &Path, ctx: &Arc<RunContext>) -> ValidationOutcome {
        if self.config.no_validate {
            return self.heuristic_outcome(path);
        }
        if !self.config.validate_force
            && let Some(cached) = self.validation_cache_get(path, ctx).await
        {
            return cached_to_outcome(&cached);
        }
        let out = self.run_validator(path).await;
        self.validation_cache_put(path, &out, ctx).await;
        out
    }

    /// Validate a media file (cache-aware), emitting validation events.
    /// `no_validate` short-circuits to the legacy heuristic (never the cache);
    /// `validate_force` bypasses the cache read but still writes the fresh
    /// result back.
    async fn validate_media(
        &self,
        path: &Path,
        episode: i64,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
    ) -> ValidationOutcome {
        let _ = tx.send(EpEvent::ValidationStarted { ep: episode });
        let out = self.validate_media_inner(path, ctx).await;
        let _ = tx.send(EpEvent::ValidationResult {
            ep: episode,
            ok: out.valid,
            reason: out.reason.clone(),
        });
        out
    }

    /// Cache-aware validation without events, for the startup reconcile pass.
    async fn validate_media_quiet(&self, path: &Path, ctx: &Arc<RunContext>) -> ValidationOutcome {
        self.validate_media_inner(path, ctx).await
    }

    /// Move (or, with `clean_invalid`, delete) a tool-owned invalid artifact.
    ///
    /// Gate is the download namespace only (`{slug}-E*`); marker files
    /// (`.part`/`.meas`/fragments) are handled by the stray-artifact sweep and
    /// registry-owned files by [`TempRegistry`] drop, so quarantine never
    /// touches unregistered user files.
    async fn quarantine_file(&self, path: &Path) {
        let Some(fname) = path.file_name().map(|n| n.to_string_lossy().to_string()) else {
            return;
        };
        if !fname.starts_with(&format!("{}-E", self.config.slug)) {
            return;
        }
        if self.config.clean_invalid {
            let _ = remove_file(path).await;
            return;
        }
        let quarantine = effective_quarantine_dir(&self.config);
        let _ = create_dir_all(&quarantine).await;
        let dest = quarantine.join(format!("{}-{fname}", iso_timestamp()));
        if rename(path, &dest).await.is_err() {
            let _ = remove_file(path).await;
        }
    }

    /// Sweep the out dir for stray tool-owned temp artifacts (`.part`/`.meas`/
    /// fragments under `{slug}-E*`) that are NOT registered in the temp
    /// registry — leftovers of killed loser mirrors, crashed runs, or a yt-dlp
    /// that skipped fragment cleanup. Registered artifacts are owned by the
    /// registry (removed at drop); everything else is quarantined, never
    /// deleted outright.
    async fn sweep_stray_artifacts(&self, registry: &TempRegistry) {
        let Ok(entries) = fs::read_dir(&self.config.out_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(fname) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !is_tool_owned_artifact(&fname, &self.config.slug) {
                continue;
            }
            if registry.is_registered(&path) {
                continue;
            }
            self.quarantine_file(&path).await;
        }
    }

    // ── Run-mode filtering ────────────────────────────────────────────────

    /// Decide whether this episode is excluded by the run mode, returning the
    /// pre-built result when skipped.
    async fn mode_skip(
        &self,
        episode: i64,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
    ) -> Option<EpisodeResult> {
        self.config.manifest_path.as_ref()?;
        let status = {
            let guard = ctx.manifest.lock().await;
            guard.as_ref().and_then(|m| {
                m.episodes
                    .iter()
                    .find(|r| r.episode == episode as u32)
                    .map(|r| r.final_status.clone())
            })
        };
        let status = status?;
        let skip = match self.config.run_mode {
            RunMode::Default | RunMode::ValidateOnly => false,
            // Resume processes everything except episodes already Complete.
            RunMode::Resume => status == FinalStatus::Complete,
            RunMode::RetryFailed => status != FinalStatus::Failed,
        };
        if !skip {
            return None;
        }
        // Out of scope: report the manifest output path when the file exists.
        let path = {
            let guard = ctx.manifest.lock().await;
            guard.as_ref().and_then(|m| {
                m.episodes
                    .iter()
                    .find(|r| r.episode == episode as u32)
                    .and_then(|r| r.output.as_ref())
                    .map(|o| o.path.clone())
                    .filter(|p| p.is_file())
            })
        };
        let size_mb = path
            .as_ref()
            .and_then(|p| fs::metadata(p).ok())
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);
        if let Some(p) = &path {
            let _ = tx.send(EpEvent::Done {
                ep: episode,
                host: "cached".into(),
                size_mb,
            });
            Some(EpisodeResult {
                path: Some(p.clone()),
                kind: EpisodeEndKind::Skipped,
                reason: Some("excluded by run mode; output present".into()),
            })
        } else {
            Some(EpisodeResult {
                path: None,
                kind: EpisodeEndKind::Skipped,
                reason: Some("excluded by run mode".into()),
            })
        }
    }

    // ── Episode lifecycle ─────────────────────────────────────────────────

    /// Run an episode download end to end.
    async fn download_episode(
        &self,
        ep: &EpisodeInput,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
    ) -> EpisodeResult {
        let episode = ep.episode;
        let stem = self
            .config
            .out_dir
            .join(format!("{}-E{:02}", self.config.slug, episode));

        let _ = tx.send(EpEvent::Measuring {
            ep: episode,
            host: "queued".into(),
        });
        let mut tracker = PhaseTracker::new(episode, tx);

        // Run-mode filtering (resume / retry-failed scopes).
        if let Some(result) = self.mode_skip(episode, tx, ctx).await {
            return result;
        }

        if self.cancel.is_cancelled() {
            tracker.set(episode, tx, Phase::Cancelled);
            return EpisodeResult {
                path: None,
                kind: EpisodeEndKind::Cancelled,
                reason: Some("cancelled".into()),
            };
        }

        // Validate existing output first (Default + ValidateOnly).
        if let Some(existing) = existing_download(&stem, self.heuristic_min_bytes()) {
            // Startup reconcile already validated + recorded this output this
            // run (manifest enabled): skip the duplicate probe but keep the
            // same event shape from the recorded result.
            let recorded = {
                let guard = ctx.manifest.lock().await;
                guard
                    .as_ref()
                    .and_then(|m| m.episodes.iter().find(|r| r.episode == episode as u32))
                    .and_then(|r| r.output.as_ref())
                    .map(|o| o.validated && o.path == existing)
                    .unwrap_or(false)
            };
            let v = if recorded {
                let _ = tx.send(EpEvent::ValidationStarted { ep: episode });
                let out = ValidationOutcome {
                    valid: true,
                    reason: None,
                    video_stream: false,
                    audio_stream: false,
                    duration_secs: None,
                    width: None,
                    height: None,
                    ffprobe_version: None,
                };
                let _ = tx.send(EpEvent::ValidationResult {
                    ep: episode,
                    ok: true,
                    reason: None,
                });
                out
            } else {
                self.validate_media(&existing, episode, tx, ctx).await
            };
            if v.valid {
                let size_mb = fs::metadata(&existing)
                    .map(|m| m.len() as f64 / 1_048_576.0)
                    .unwrap_or(0.0);
                if self.config.run_mode == RunMode::Default && !recorded {
                    let size = fs::metadata(&existing).map(|m| m.len()).unwrap_or(0);
                    // Reconcile (manifest enabled) already recorded the output
                    // with a streamed sha256; record here only when the
                    // manifest lacks the meta, so the sha256 is never
                    // overwritten with None.
                    self.manifest_set_output(
                        episode,
                        &existing,
                        size,
                        sha256_file(&existing).ok(),
                        &v,
                        ctx,
                    )
                    .await;
                }
                let _ = tx.send(EpEvent::Done {
                    ep: episode,
                    host: "cached".into(),
                    size_mb,
                });
                tracker.set(
                    episode,
                    tx,
                    Phase::Done {
                        host: "cached".into(),
                        size_mb,
                    },
                );
                return EpisodeResult {
                    path: Some(existing),
                    kind: EpisodeEndKind::Skipped,
                    reason: None,
                };
            }
            // Existing output is invalid.
            if self.config.run_mode == RunMode::ValidateOnly {
                let reason = format!(
                    "invalid media: {}",
                    v.reason
                        .clone()
                        .unwrap_or_else(|| "validation failed".into())
                );
                let _ = tx.send(EpEvent::Failed { ep: episode });
                tracker.set(episode, tx, Phase::Failed);
                return EpisodeResult {
                    path: None,
                    kind: EpisodeEndKind::Failed,
                    reason: Some(reason),
                };
            }
            self.quarantine_file(&existing).await;
        } else if self.config.run_mode == RunMode::ValidateOnly {
            let _ = tx.send(EpEvent::ValidationStarted { ep: episode });
            let _ = tx.send(EpEvent::ValidationResult {
                ep: episode,
                ok: false,
                reason: Some("no output file".into()),
            });
            let _ = tx.send(EpEvent::Failed { ep: episode });
            tracker.set(episode, tx, Phase::Failed);
            return EpisodeResult {
                path: None,
                kind: EpisodeEndKind::Missing,
                reason: Some("no output file".into()),
            };
        }

        // Canonicalize VK URLs before any processing.
        let canonical_urls: Vec<String> = ep
            .urls
            .iter()
            .map(|url| try_canonicalize_vk_url(url).unwrap_or_else(|| url.clone()))
            .collect();

        let mut mirrors: Vec<(String, Quality)> = Vec::new();
        for (i, url) in canonical_urls.iter().enumerate() {
            // v2: prefer per-mirror quality metadata when present (mirrors
            // with an explicit quality skip format inspection entirely); fall
            // back to episode-level metadata, then unknown (inspected later).
            let quality = ep
                .mirrors
                .get(i)
                .and_then(|m| m.quality.as_deref())
                .map(quality_height_from_str)
                .filter(|h| *h > 0)
                .or_else(|| ep.quality.as_ref().map(|q| q.height))
                .map(Quality::new)
                .unwrap_or_else(|| Quality::new(0));
            mirrors.push((url.clone(), quality));
        }

        if mirrors.is_empty() {
            let _ = tx.send(EpEvent::Failed { ep: episode });
            tracker.set(episode, tx, Phase::Failed);
            return EpisodeResult {
                path: None,
                kind: EpisodeEndKind::Failed,
                reason: Some("no mirrors".into()),
            };
        }

        // Cancellation check before inspection phase.
        if self.cancel.is_cancelled() {
            tracker.set(episode, tx, Phase::Cancelled);
            return EpisodeResult {
                path: None,
                kind: EpisodeEndKind::Cancelled,
                reason: Some("cancelled".into()),
            };
        }

        // Inspect unknown-quality mirrors concurrently.
        let unknown_mirrors: Vec<usize> = mirrors
            .iter()
            .enumerate()
            .filter(|(_, (_, q))| q.is_unknown())
            .map(|(i, _)| i)
            .collect();

        if !unknown_mirrors.is_empty() {
            tracker.set(episode, tx, Phase::Inspecting);
            let mut inspect_set = JoinSet::new();
            for &idx in &unknown_mirrors {
                let url = mirrors[idx].0.clone();
                let factory = self.factory.clone();
                inspect_set.spawn(async move {
                    let quality = factory.inspect_formats(&url).await;
                    (idx, quality)
                });
            }

            while let Some(result) = inspect_set.join_next().await {
                if let Ok((idx, Some(q))) = result
                    && idx < mirrors.len()
                {
                    mirrors[idx].1 = q;
                }
            }
        }

        // Quality-first filter.
        let max_height = mirrors.iter().map(|(_, q)| q.height).max().unwrap_or(0);
        let eligible: Vec<(String, Quality)> = mirrors
            .into_iter()
            .filter(|(_, q)| q.height == max_height)
            .collect();

        let _ = tx.send(EpEvent::Measuring {
            ep: episode,
            host: format!("{} eligible mirrors", eligible.len()),
        });
        tracker.set(episode, tx, Phase::Measuring);

        self.manifest_begin_episode(episode, ctx).await;
        let mut registry = TempRegistry::new();

        // Single mirror: skip measurement race.
        if eligible.len() == 1 {
            let (url, _quality) = &eligible[0];
            let host = Self::host_from_url(url);
            if let Some(cooldown) = self.circuit_cooldown_remaining(&host, tx, ctx).await {
                let reason = format!("host circuit open; cooldown {cooldown}s remaining");
                self.record_attempt(
                    episode,
                    0,
                    &host,
                    url,
                    false,
                    Some(reason.clone()),
                    0,
                    0.0,
                    ctx,
                )
                .await;
                return self
                    .finish_failed_episode(episode, &stem, reason, tx, ctx, registry)
                    .await;
            }
            let continue_part = self.should_continue_part(episode, url, &stem, ctx).await;
            let result = self
                .attempt_mirror(
                    episode,
                    url,
                    None,
                    None,
                    &stem,
                    continue_part,
                    &mut registry,
                    tx,
                    ctx,
                    0,
                )
                .await;
            return match result {
                MirrorAttempt::Success(path) => {
                    match self
                        .handle_mirror_success(episode, url, &host, path, 0, tx, ctx, &mut registry)
                        .await
                    {
                        Ok(r) => r,
                        Err(reason) => {
                            self.finish_failed_episode(episode, &stem, reason, tx, ctx, registry)
                                .await
                        }
                    }
                }
                MirrorAttempt::Failed { reason, .. } => {
                    self.finish_failed_episode(episode, &stem, reason, tx, ctx, registry)
                        .await
                }
                MirrorAttempt::Cancelled => {
                    self.cancel_episode(episode, &stem, tx, ctx, registry).await
                }
            };
        }

        // Multi-mirror: measure race, then try mirrors in order (winner first).
        let mut meas_results = self
            .measure_mirrors(episode, &eligible, &stem, &mut registry, tx, ctx)
            .await;

        // Winner selection. The winning measure child keeps running to
        // COMPLETION as the real download — yt-dlp still owns its live
        // `.part` file, so it is NEVER renamed in place (renaming a live file
        // breaks yt-dlp's own finalize, worse for merged formats). After the
        // winner exits 0 the engine renames `{stem}.meas{idx}.{ext}` → final.
        // Every loser measure child is killed + reaped here and its partial
        // artifacts removed, so no live child can repopulate files after
        // cleanup, and no stale loser files survive into the sweep.
        for meas in meas_results.iter_mut().skip(1) {
            if let Some(guard) = meas.child.take() {
                let mut proc = guard.into_inner();
                proc.kill_group();
                let _ = proc.child.wait().await;
            }
        }
        for meas in &meas_results[1..] {
            cleanup_meas_stem(&meas.meas_stem);
        }

        let mut last_reason: Option<String> = None;
        let n_mirrors = meas_results.len();
        let mut mirror_idx = 0usize;
        while mirror_idx < n_mirrors {
            if self.cancel.is_cancelled() {
                // Drop-order guarantee: kill + reap every remaining measure
                // child BEFORE the temp registry is dropped in cancel_episode.
                for meas in meas_results.iter_mut() {
                    if let Some(guard) = meas.child.take() {
                        let mut proc = guard.into_inner();
                        proc.kill_group();
                        let _ = proc.child.wait().await;
                    }
                }
                return self.cancel_episode(episode, &stem, tx, ctx, registry).await;
            }

            let host = Self::host_from_url(&meas_results[mirror_idx].url);

            // Circuit breaker: skip mirrors of an open host; others continue.
            if let Some(cooldown) = self.circuit_cooldown_remaining(&host, tx, ctx).await {
                let reason = format!("host circuit open; cooldown {cooldown}s remaining");
                self.record_attempt(
                    episode,
                    mirror_idx,
                    &host,
                    &meas_results[mirror_idx].url,
                    false,
                    Some(reason.clone()),
                    0,
                    0.0,
                    ctx,
                )
                .await;
                last_reason = Some(reason);
                mirror_idx += 1;
                continue;
            }

            if meas_results[mirror_idx].child.is_none() {
                // Measurement produced no usable child (spawn failure / early
                // exit without artifact) — fall back to a fresh download.
                let _ = tx.send(EpEvent::MirrorMeasFailed {
                    ep: episode,
                    host: host.clone(),
                });
            }

            // Winner continuation: the measured child IS the real download.
            // `meas_stem` lets the exit-0 handler rename exactly the winner's
            // finalized meas output (never a stale file from an earlier run).
            let meas_stem = meas_results[mirror_idx].meas_stem.clone();
            let mut continuation = meas_results[mirror_idx]
                .child
                .take()
                .map(|g| g.into_inner());
            // A measure child that already exited WITHOUT producing any
            // artifact is a silent measurement failure — continuing it would
            // waste a retry before seeing the real (often permanent) failure.
            // Drop it and start a fresh download instead.
            if let Some(child) = continuation.as_mut() {
                let exited = child.child.try_wait().ok().flatten().is_some();
                let has_artifact = find_meas_output(&meas_stem).is_some()
                    || find_stem_part_file(&meas_stem).is_some();
                if exited && !has_artifact {
                    let _ = tx.send(EpEvent::MirrorMeasFailed {
                        ep: episode,
                        host: host.clone(),
                    });
                    continuation = None;
                }
            }

            let continue_part = self
                .should_continue_part(episode, &meas_results[mirror_idx].url, &stem, ctx)
                .await;
            let result = self
                .attempt_mirror(
                    episode,
                    &meas_results[mirror_idx].url,
                    continuation,
                    Some(meas_stem),
                    &stem,
                    continue_part,
                    &mut registry,
                    tx,
                    ctx,
                    mirror_idx,
                )
                .await;

            match result {
                MirrorAttempt::Success(path) => {
                    match self
                        .handle_mirror_success(
                            episode,
                            &meas_results[mirror_idx].url,
                            &host,
                            path,
                            mirror_idx,
                            tx,
                            ctx,
                            &mut registry,
                        )
                        .await
                    {
                        Ok(r) => return r,
                        Err(reason) => {
                            last_reason = Some(reason);
                            mirror_idx += 1;
                            continue;
                        }
                    }
                }
                MirrorAttempt::Failed { reason, .. } => {
                    last_reason = Some(reason);
                    mirror_idx += 1;
                    continue;
                }
                MirrorAttempt::Cancelled => {
                    return self.cancel_episode(episode, &stem, tx, ctx, registry).await;
                }
            }
        }

        // All mirrors exhausted.
        let reason = last_reason.unwrap_or_else(|| "all mirrors failed".into());
        self.finish_failed_episode(episode, &stem, reason, tx, ctx, registry)
            .await
    }

    /// Terminal handling for a mirror attempt whose download succeeded: media
    /// validation, manifest output, quarantine on invalid. `Err(reason)` means
    /// "invalid media — try the next mirror".
    #[allow(clippy::too_many_arguments)]
    async fn handle_mirror_success(
        &self,
        episode: i64,
        url: &str,
        host: &str,
        path: PathBuf,
        mirror_idx: usize,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
        registry: &mut TempRegistry,
    ) -> Result<EpisodeResult, String> {
        let v = self.validate_media(&path, episode, tx, ctx).await;
        if v.valid {
            let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            // Stream the file through SHA-256 (bounded 64 KiB buffer) — the
            // whole file must never be buffered in RAM.
            let sha = sha256_file(&path).ok();
            self.record_attempt(episode, mirror_idx, host, url, true, None, size, 0.0, ctx)
                .await;
            self.manifest_set_output(episode, &path, size, sha, &v, ctx)
                .await;
            let size_mb = size as f64 / 1_048_576.0;
            let _ = tx.send(EpEvent::Done {
                ep: episode,
                host: host.to_string(),
                size_mb,
            });
            registry.promote(&path);
            // Final artifact sweep: quarantine stray unregistered marker files
            // (killed-loser leftovers, crashed-run temps) without touching
            // user files or registry-owned artifacts.
            self.sweep_stray_artifacts(registry).await;
            Ok(EpisodeResult {
                path: Some(path),
                kind: EpisodeEndKind::Downloaded,
                reason: None,
            })
        } else {
            let reason = format!(
                "invalid media: {}",
                v.reason
                    .clone()
                    .unwrap_or_else(|| "validation failed".into())
            );
            let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            self.quarantine_file(&path).await;
            self.record_attempt(
                episode,
                mirror_idx,
                host,
                url,
                false,
                Some(reason.clone()),
                size,
                0.0,
                ctx,
            )
            .await;
            Err(reason)
        }
    }

    async fn finish_failed_episode(
        &self,
        episode: i64,
        stem: &Path,
        reason: String,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
        mut registry: TempRegistry,
    ) -> EpisodeResult {
        // Register late-appearing artifacts (registry owns them), then sweep
        // anything unregistered to quarantine.
        registry.register_prefix_artifacts(stem);
        self.sweep_stray_artifacts(&registry).await;
        self.manifest_set_failed(episode, ctx).await;
        let _ = tx.send(EpEvent::Failed { ep: episode });
        EpisodeResult {
            path: None,
            kind: EpisodeEndKind::Failed,
            reason: Some(reason),
        }
    }

    async fn cancel_episode(
        &self,
        episode: i64,
        stem: &Path,
        tx: &broadcast::Sender<EpEvent>,
        _ctx: &Arc<RunContext>,
        mut registry: TempRegistry,
    ) -> EpisodeResult {
        registry.register_prefix_artifacts(stem);
        self.sweep_stray_artifacts(&registry).await;
        // Leave the manifest status InProgress (set at episode start) so a
        // later resume sees it as unfinished; the manifest is flushed by the
        // caller with attempts so far.
        let _ = tx.send(EpEvent::Cancelled { ep: episode });
        EpisodeResult {
            path: None,
            kind: EpisodeEndKind::Cancelled,
            reason: Some("cancelled".into()),
        }
    }

    /// Try one mirror with retry/backoff/circuit policy.
    #[allow(clippy::too_many_arguments)]
    async fn attempt_mirror(
        &self,
        episode: i64,
        url: &str,
        continuation: Option<RunningSubprocess>,
        meas_stem: Option<PathBuf>,
        stem: &Path,
        continue_part: bool,
        registry: &mut TempRegistry,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
        mirror_idx: usize,
    ) -> MirrorAttempt {
        let host = Self::host_from_url(url);

        // Unsupported URL → permanent immediately, no yt-dlp spawn.
        if HostAdapter::recognize(url).is_none() {
            let reason = "unsupported url".to_string();
            self.record_attempt(
                episode,
                mirror_idx,
                &host,
                url,
                false,
                Some(reason.clone()),
                0,
                0.0,
                ctx,
            )
            .await;
            return MirrorAttempt::Failed { reason };
        }

        let mut retry = 0u32;
        // Wrap the continuation child in a guard immediately: any early return
        // (cancel, circuit, semaphore) drops the guard → kill + reap, so a
        // live child can never outlive the episode's registry cleanup.
        let mut cont = continuation.map(ChildGuard::new);
        loop {
            if self.cancel.is_cancelled() {
                self.record_attempt(
                    episode,
                    mirror_idx,
                    &host,
                    url,
                    false,
                    Some("cancelled".into()),
                    0,
                    0.0,
                    ctx,
                )
                .await;
                return MirrorAttempt::Cancelled;
            }

            // Circuit check before EVERY mirror attempt, including retries of
            // the same episode: once a failure opens the host circuit, the
            // remaining attempts of this episode skip too (not just other
            // episodes).
            if let Some(cooldown) = self.circuit_cooldown_remaining(&host, tx, ctx).await {
                let reason = format!("host circuit open; cooldown {cooldown}s remaining");
                self.record_attempt(
                    episode,
                    mirror_idx,
                    &host,
                    url,
                    false,
                    Some(reason.clone()),
                    0,
                    0.0,
                    ctx,
                )
                .await;
                return MirrorAttempt::Failed { reason };
            }

            // Host lock with MirrorBusy reporting (cancellation-aware).
            let Some(permit) = self.acquire_host_semaphore(&host, episode, tx).await else {
                if self.cancel.is_cancelled() {
                    return MirrorAttempt::Cancelled;
                }
                let reason = "host semaphore unavailable".to_string();
                return MirrorAttempt::Failed { reason };
            };
            let _ = tx.send(EpEvent::Winner {
                ep: episode,
                host: host.clone(),
            });

            let attempt_result = self
                .perform_download_with_child(
                    episode,
                    url,
                    stem,
                    host.clone(),
                    cont.take().map(|g| g.into_inner()),
                    meas_stem.as_deref(),
                    continue_part,
                    tx,
                    registry,
                )
                .await;
            drop(permit);

            match attempt_result {
                Ok(path) => {
                    self.note_success(&host, ctx).await;
                    return MirrorAttempt::Success(path);
                }
                Err(fail) => {
                    let stderr_joined = fail.stderr_tail.join("\n");
                    let class = classify(&stderr_joined, fail.exit);
                    let reason = Self::failure_reason(&class, &fail.stderr_tail);
                    self.record_attempt(
                        episode,
                        mirror_idx,
                        &host,
                        url,
                        false,
                        Some(reason.clone()),
                        fail.bytes,
                        fail.secs,
                        ctx,
                    )
                    .await;
                    if Self::is_systemic(&class) {
                        self.note_systemic_failure(&host, tx, ctx).await;
                    }
                    if matches!(class, ErrorClass::Permanent(_)) {
                        return MirrorAttempt::Failed { reason };
                    }
                    retry += 1;
                    if retry > self.config.retry_attempts {
                        return MirrorAttempt::Failed { reason };
                    }
                    let backoff = self.backoff_for(retry, url, episode);
                    let _ = tx.send(EpEvent::RetryWait {
                        ep: episode,
                        mirror: host.clone(),
                        attempt: retry,
                        backoff_secs: backoff.round().max(0.0) as u64,
                    });
                    // Cancellation-aware backoff sleep.
                    tokio::select! {
                        _ = self.cancel.cancelled() => {
                            self.record_attempt(
                                episode,
                                mirror_idx,
                                &host,
                                url,
                                false,
                                Some("cancelled".into()),
                                0,
                                0.0,
                                ctx,
                            ).await;
                            return MirrorAttempt::Cancelled;
                        }
                        _ = sleep(Duration::from_secs_f64(backoff)) => {}
                    }
                }
            }
        }
    }

    /// Measure all eligible mirrors concurrently, keeping children alive.
    async fn measure_mirrors(
        &self,
        episode: i64,
        eligible: &[(String, Quality)],
        stem: &Path,
        registry: &mut TempRegistry,
        tx: &broadcast::Sender<EpEvent>,
        ctx: &Arc<RunContext>,
    ) -> Vec<MeasResult> {
        let measure_dur = Duration::from_secs(self.config.measurement_secs);
        let mut meas_set = JoinSet::new();

        for (idx, (url, _quality)) in eligible.iter().enumerate() {
            let url = url.clone();
            let factory = self.factory.clone();
            let cancel = self.cancel.clone();
            let duration = measure_dur;
            let meas_stem = stem.with_extension(format!("meas{}", idx));

            meas_set.spawn(async move {
                let proc = factory.spawn_measure(&url, &meas_stem).await;
                let Some(mut proc) = proc else {
                    return MeasResult {
                        url,
                        child: None,
                        avg_bps: None,
                        meas_stem,
                    };
                };
                let deadline = Instant::now() + duration;
                let stdout = proc.child.stdout.take();
                let mut speeds = Vec::new();

                if let Some(stdout) = stdout {
                    let mut reader = BufReader::new(stdout).lines();

                    loop {
                        tokio::select! {
                            _ = cancel.cancelled() => { break; }
                            result = timeout(Duration::from_millis(500), reader.next_line()) => {
                                match result {
                                    Ok(Ok(Some(line))) => {
                                        if let Some(bps) = parse_speed_from_line(&line) {
                                            speeds.push(bps);
                                        }
                                        if Instant::now() >= deadline || speeds.len() >= 4 {
                                            break;
                                        }
                                    }
                                    _ => break,
                                }
                            }
                        }
                    }
                }

                let avg_bps = if speeds.is_empty() {
                    None
                } else {
                    Some(speeds.iter().sum::<f64>() / speeds.len() as f64)
                };

                // Wait bounded duration for measurement artifact to appear.
                // Slow extractors (HQQ, etc.) may need extra time after samples.
                let artifact_deadline = Instant::now() + Duration::from_millis(2000);
                while Instant::now() < artifact_deadline && !cancel.is_cancelled() {
                    if find_stem_part_file(&meas_stem).is_some()
                        || find_stem_output(&meas_stem).is_some()
                    {
                        break;
                    }
                    sleep(Duration::from_millis(200)).await;
                }

                // Keep child alive — don't kill it.
                MeasResult {
                    url,
                    child: Some(ChildGuard::new(proc)),
                    avg_bps,
                    meas_stem,
                }
            });
        }

        let mut meas_results: Vec<MeasResult> = Vec::new();
        while let Some(result) = meas_set.join_next().await {
            if let Ok(res) = result {
                registry.register_prefix_artifacts(&res.meas_stem);
                let host_str = Self::host_from_url(&res.url);
                if let Some(bps) = res.avg_bps {
                    let _ = tx.send(EpEvent::Measured {
                        ep: episode,
                        host: host_str,
                        bps,
                    });
                }
                meas_results.push(res);
            }
        }

        let _ = tx.send(EpEvent::MeasurementComplete { ep: episode });
        let _ = ctx; // per-run context is owned by the caller's loop

        // Sort by speed desc, then host pref, then URL.
        meas_results.sort_by(|a, b| {
            let a_bps = a.avg_bps.unwrap_or(0.0);
            let b_bps = b.avg_bps.unwrap_or(0.0);
            b_bps
                .partial_cmp(&a_bps)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    let a_rank = host_preference_rank(&Self::host_from_url(&a.url));
                    let b_rank = host_preference_rank(&Self::host_from_url(&b.url));
                    a_rank.cmp(&b_rank)
                })
                .then_with(|| a.url.cmp(&b.url))
        });

        meas_results
    }

    /// Perform download from a given URL, optionally continuing an existing
    /// child (`meas_stem` names the winner's measurement output, renamed to
    /// the final file only after the child exits 0).
    #[allow(clippy::too_many_arguments)]
    async fn perform_download_with_child(
        &self,
        episode: i64,
        url: &str,
        stem: &Path,
        host: String,
        existing_child: Option<RunningSubprocess>,
        meas_stem: Option<&Path>,
        continue_part: bool,
        tx: &broadcast::Sender<EpEvent>,
        registry: &mut TempRegistry,
    ) -> Result<PathBuf, DownloadFailure> {
        if let Some(parent) = stem.parent() {
            let _ = create_dir_all(parent).await;
        }

        let (mut guard, started) = if let Some(existing) = existing_child {
            (ChildGuard::new(existing), true)
        } else {
            let proc = self.factory.spawn_download(url, stem, continue_part).await;
            let Some(proc) = proc else {
                let _ = tx.send(EpEvent::MirrorDone {
                    ep: episode,
                    host: host.clone(),
                    success: false,
                });
                return Err(DownloadFailure {
                    exit: None,
                    stderr_tail: Vec::new(),
                    bytes: 0,
                    secs: 0.0,
                });
            };
            (ChildGuard::new(proc), false)
        };

        // Register whatever artifacts exist right after spawn.
        registry.register_prefix_artifacts(stem);

        let profile = profile_for_url(url);
        let start = Instant::now();
        let mut started_flag = started;
        let mut stall_start: Option<Instant> = None;
        let mut prev_bytes = 0u64;
        let mut last_progress = Instant::now();

        let stdout = guard.child_mut().child.stdout.take();
        let mut reader = stdout.map(|o| BufReader::new(o).lines());
        let stderr = guard.child_mut().child.stderr.take();
        let stderr_lines: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let stderr_lines_clone = stderr_lines.clone();
        let mut stderr_task = None;
        if let Some(stderr) = stderr {
            // Bounded tail: drain to EOF but keep only the last 5 lines.
            stderr_task = Some(tokio::spawn(async move {
                let mut stderr_reader = BufReader::new(stderr).lines();
                let mut buf: Vec<String> = Vec::new();
                while let Ok(Some(line)) = stderr_reader.next_line().await {
                    buf.push(line);
                    if buf.len() > STDERR_TAIL_LINES {
                        buf.remove(0);
                    }
                }
                if let Ok(mut target) = stderr_lines_clone.lock() {
                    *target = buf;
                }
            }));
        }

        let part_bytes = |part: &Option<PathBuf>| -> u64 {
            part.as_ref()
                .and_then(|p| fs::metadata(p).ok())
                .map(|m| m.len())
                .unwrap_or(0)
        };
        let snapshot = |lines: &Arc<std::sync::Mutex<Vec<String>>>| -> Vec<String> {
            lines.lock().map(|g| g.clone()).unwrap_or_default()
        };
        let drain_stderr = async {
            if let Some(task) = stderr_task.take() {
                let _ = timeout(Duration::from_millis(250), task).await;
            }
            snapshot(&stderr_lines)
        };

        let mut part_path: Option<PathBuf> = None;

        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    // ChildGuard drop kills + reaps; registry cleans temps.
                    let tail = snapshot(&stderr_lines);
                    let _ = tx.send(EpEvent::Cancelled { ep: episode });
                    return Err(DownloadFailure {
                        exit: None,
                        stderr_tail: tail,
                        bytes: part_bytes(&part_path),
                        secs: start.elapsed().as_secs_f64(),
                    });
                }
                _ = sleep(Duration::from_millis(200)) => {
                    if part_path.is_none() {
                        part_path = find_part_file(stem);
                    }

                    if !started_flag {
                        if start.elapsed() > Duration::from_secs(profile.startup_secs) {
                            let tail = drain_stderr.await;
                            let _ = tx.send(EpEvent::MirrorDone { ep: episode, host: host.clone(), success: false });
                            return Err(DownloadFailure {
                                exit: None,
                                stderr_tail: tail,
                                bytes: part_bytes(&part_path),
                                secs: start.elapsed().as_secs_f64(),
                            });
                        }
                        if part_path.as_ref().and_then(|p| fs::metadata(p).ok()).map(|m| m.len() > 0).unwrap_or(false) {
                            started_flag = true;
                        }
                    } else {
                        let current_bytes = part_bytes(&part_path);
                        if current_bytes == prev_bytes {
                            match stall_start {
                                None => stall_start = Some(Instant::now()),
                                Some(s) if s.elapsed() > Duration::from_secs(profile.stall_secs) => {
                                    let tail = drain_stderr.await;
                                    let _ = tx.send(EpEvent::MirrorDone { ep: episode, host: host.clone(), success: false });
                                    return Err(DownloadFailure {
                                        exit: None,
                                        stderr_tail: tail,
                                        bytes: part_bytes(&part_path),
                                        secs: start.elapsed().as_secs_f64(),
                                    });
                                }
                                _ => {}
                            }
                        } else {
                            stall_start = None;
                        }
                        prev_bytes = current_bytes;
                    }

                    if start.elapsed() > OVERALL_TIMEOUT {
                        let tail = drain_stderr.await;
                        let _ = tx.send(EpEvent::MirrorDone { ep: episode, host: host.clone(), success: false });
                        return Err(DownloadFailure {
                            exit: None,
                            stderr_tail: tail,
                            bytes: part_bytes(&part_path),
                            secs: start.elapsed().as_secs_f64(),
                        });
                    }
                }
                line_result = async {
                    if let Some(ref mut reader) = reader {
                        tokio::time::timeout(Duration::from_millis(100), reader.next_line()).await
                    } else {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        Ok(Ok(None))
                    }
                } => {
                    match line_result {
                        Ok(Ok(Some(line)))
                            if parse_speed_from_line(&line).is_some() => {
                                let pct = line.split('%').next()
                                    .and_then(|s| s.rsplit(' ').next())
                                    .and_then(|s| s.trim().parse::<f64>().ok())
                                    .unwrap_or(0.0);
                                let eta = line.split("ETA ")
                                    .nth(1)
                                    .map(|s| s.trim().to_string())
                                    .unwrap_or_default();
                                let speed_str = line.split(" at ")
                                    .nth(1)
                                    .and_then(|s| s.split_whitespace().next())
                                    .unwrap_or("")
                                    .to_string();
                                let total_str = line.split("of ~")
                                    .nth(1)
                                    .and_then(|s| s.split(' ').next())
                                    .unwrap_or("");
                                let total_bytes = parse_speed_bps(total_str).unwrap_or(0.0) as u64;
                                let downloaded = (total_bytes as f64 * pct / 100.0) as u64;

                                // Rate-limit progress sends to ~250ms.
                                if last_progress.elapsed() >= Duration::from_millis(250) {
                                    let _ = tx.send(EpEvent::Progress {
                                        ep: episode,
                                        host: host.clone(),
                                        pct,
                                        speed: speed_str,
                                        eta,
                                        downloaded,
                                        total: total_bytes,
                                    });
                                    last_progress = Instant::now();
                                }
                            }
                        _ => {}
                    }
                }
                status = guard.child_mut().child.wait() => {
                    match status {
                        Ok(s) if s.success() => {
                            // Path 1: completed output via find_output_file.
                            if let Some(path) = find_output_file(stem, self.heuristic_min_bytes()) {
                                let _ = guard.take();
                                return Ok(path);
                            }

                            // Path 2: continuation child — the winner's yt-dlp
                            // finalized its OWN live .part inside the meas
                            // namespace on exit 0; now (and only now) the engine
                            // renames `{stem}.meas{idx}.{ext}` → final. The
                            // live file is never renamed while the child owns it.
                            if started
                                && let Some(meas_stem) = meas_stem
                                && let Some(meas_path) = find_meas_output(meas_stem)
                            {
                                let final_name = stem.with_extension(
                                    meas_path.extension().unwrap_or_default(),
                                );
                                registry.promote(&meas_path);
                                let _ = rename(&meas_path, &final_name).await;
                                if is_valid_output_with_min(&final_name, self.heuristic_min_bytes()) {
                                    let _ = guard.take();
                                    return Ok(final_name);
                                }
                                let _ = remove_file(&final_name).await;
                            }

                            // Path 3: part file → final.
                            if part_path.is_none() {
                                part_path = find_part_file(stem);
                            }
                            if let Some(ref pp) = part_path
                                && fs::metadata(pp).map(|m| m.len() > 0).unwrap_or(false) {
                                    let fname = pp.file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_default();
                                    let final_path = if let Some(base) = fname.strip_suffix(".part") {
                                        stem.with_extension(
                                            Path::new(base).extension().unwrap_or_default()
                                        )
                                    } else {
                                        stem.with_extension("mkv")
                                    };
                                    let _ = rename(pp, &final_path).await;
                                    if is_valid_output_with_min(&final_path, self.heuristic_min_bytes()) {
                                        let _ = guard.take();
                                        return Ok(final_path);
                                    }
                                    let _ = remove_file(&final_path).await;
                                }

                            // Child exited 0 but produced no valid output.
                            let tail = drain_stderr.await;
                            let _ = guard.take();
                            let _ = tx.send(EpEvent::MirrorDone {
                                ep: episode,
                                host: host.clone(),
                                success: false,
                            });
                            return Err(DownloadFailure {
                                exit: Some(0),
                                stderr_tail: tail,
                                bytes: part_bytes(&part_path),
                                secs: start.elapsed().as_secs_f64(),
                            });
                        }
                        _ => {
                            let tail = drain_stderr.await;
                            let exit = status.ok().and_then(|s| s.code());
                            let _ = guard.take();
                            let _ = tx.send(EpEvent::MirrorDone {
                                ep: episode,
                                host: host.clone(),
                                success: false,
                            });
                            // Log sanitized stderr tail for diagnostics.
                            if !tail.is_empty() {
                                let sanitized: Vec<String> = tail.iter().map(|l| redact_urls(l)).collect();
                                tracing::warn!(
                                    ep = episode,
                                    host = %host,
                                    stderr_tail = %sanitized.join(" | "),
                                    "yt-dlp download failed"
                                );
                            }
                            return Err(DownloadFailure {
                                exit,
                                stderr_tail: tail,
                                bytes: part_bytes(&part_path),
                                secs: start.elapsed().as_secs_f64(),
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Run orchestration ─────────────────────────────────────────────────

    /// Run multiple episodes with concurrent spawn (no retry at this level;
    /// retries are per-mirror inside [`DownloadEngine::download_episode`]).
    /// Returns per-episode output paths in ascending episode order.
    pub async fn run_all(
        &self,
        episodes: Vec<EpisodeInput>,
        tx: broadcast::Sender<EpEvent>,
    ) -> Vec<(i64, Option<PathBuf>)> {
        let (outcome, paths) = self.run_engine(episodes, tx).await;
        let _ = outcome;
        let mut v: Vec<(i64, Option<PathBuf>)> = paths.into_iter().collect();
        v.sort_by_key(|(ep, _)| *ep);
        v
    }

    /// Full engine run returning the aggregate [`Outcome`].
    pub async fn run_all_with_outcome(
        &self,
        episodes: Vec<EpisodeInput>,
        tx: broadcast::Sender<EpEvent>,
    ) -> Outcome {
        let (outcome, _paths) = self.run_engine(episodes, tx).await;
        outcome
    }

    async fn run_engine(
        &self,
        episodes: Vec<EpisodeInput>,
        tx: broadcast::Sender<EpEvent>,
    ) -> (Outcome, HashMap<i64, Option<PathBuf>>) {
        let config = self.config.clone();

        // Manifest bootstrap + reconcile (deleted outputs reset to Pending).
        let manifest = if config.manifest_path.is_some() {
            let mut m = Manifest::load(config.manifest_path.as_deref().expect("checked"))
                .unwrap_or_else(|| {
                    let mut fresh = Manifest::new();
                    fresh.input.title = Some(config.slug.clone());
                    fresh
                });
            m.input.episode_count = episodes.len();
            // Input identity computed at run start: sha256 streamed from the
            // source JSON plus its path; `resolved_at` from the input when
            // present, else `None`. A failed hash read leaves any prior value
            // in place (fresh manifests default to `None`).
            if let Some(src) = config.input_source_path.as_ref() {
                m.input.source_json_path = Some(src.clone());
                if let Ok(sha) = sha256_file(src) {
                    m.input.sha256 = Some(sha);
                }
            }
            m.input.resolved_at = config.input_resolved_at.clone();
            for e in &episodes {
                m.ensure_episode(e.episode as u32);
            }
            let present = collect_files(&config.out_dir);
            m.reconcile_episodes(&present);
            Some(m)
        } else {
            None
        };

        let ctx = Arc::new(RunContext {
            manifest: Mutex::new(manifest),
            circuits: Mutex::new(HashMap::new()),
        });
        if config.manifest_path.is_some() {
            // Reconcile + record: validate existing outputs (cache-aware) and
            // mark validated ones Complete with output metadata, so the
            // manifest is accurate (resume fast-path, accurate summaries)
            // before any episode processing starts.
            self.reconcile_existing_outputs(&episodes, &ctx).await;
            self.save_manifest(&ctx).await;
        }

        let ep_map: HashMap<i64, EpisodeInput> =
            episodes.into_iter().map(|e| (e.episode, e)).collect();
        let total_eps: BTreeSet<i64> = ep_map.keys().copied().collect();
        let sem = Arc::new(Semaphore::new(config.episode_concurrency));
        let mut active = JoinSet::new();
        let self_arc = Arc::new(self.clone());

        // Spawn initial batch: acquire permits and spawn into JoinSet. The
        // permit acquire is cancellation-aware: when episode_concurrency <
        // episode count, later acquires block until an earlier episode
        // finishes — that wait must abort promptly on cancel.
        for ep_num in &total_eps {
            let Some(ep) = ep_map.get(ep_num) else {
                continue;
            };
            let ep = ep.clone();
            let permit = tokio::select! {
                p = sem.clone().acquire_owned() => p,
                _ = self.cancel.cancelled() => break,
            };
            let engine = self_arc.clone();
            let tx = tx.clone();
            let ctx = ctx.clone();

            active.spawn(async move {
                let _permit = permit;
                let result = engine.download_episode(&ep, &tx, &ctx).await;
                (ep.episode, result)
            });
        }

        // Collect results.
        let mut results: HashMap<i64, EpisodeResult> = HashMap::new();
        let mut terminal: HashSet<i64> = HashSet::new();
        let cancel = self.cancel.clone();
        loop {
            if terminal.len() == total_eps.len() && active.is_empty() {
                break;
            }

            tokio::select! {
                Some(result) = active.join_next() => {
                    if let Ok((ep, out)) = result {
                        results.insert(ep, out);
                        terminal.insert(ep);
                    }
                }
                _ = sleep(Duration::from_millis(200)) => {
                    if cancel.is_cancelled() {
                        while let Some(result) = active.join_next().await {
                            if let Ok((ep, out)) = result {
                                results.insert(ep, out);
                                terminal.insert(ep);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Aggregate outcome.
        let mut downloaded = 0u32;
        let mut skipped = 0u32;
        let mut failed = 0u32;
        let mut cancelled = 0u32;
        let mut missing_episodes: Vec<u32> = Vec::new();
        let mut per_episode_reasons: Vec<(u32, String)> = Vec::new();
        let mut paths: HashMap<i64, Option<PathBuf>> = HashMap::new();
        let mut any_cancelled = self.cancel.is_cancelled();

        for ep in &total_eps {
            let out = results.get(ep).cloned().unwrap_or(EpisodeResult {
                path: None,
                kind: EpisodeEndKind::Cancelled,
                reason: Some("cancelled".into()),
            });
            match out.kind {
                EpisodeEndKind::Downloaded => downloaded += 1,
                EpisodeEndKind::Skipped => skipped += 1,
                EpisodeEndKind::Failed => {
                    failed += 1;
                    per_episode_reasons.push((
                        *ep as u32,
                        out.reason.clone().unwrap_or_else(|| "failed".into()),
                    ));
                    // Invariant: failed episodes are also pushed to
                    // missing_episodes (failed ⊆ missing). `missing_episodes`
                    // is the authoritative count of episodes without valid
                    // output — exit-code logic must never add `failed` on top.
                    missing_episodes.push(*ep as u32);
                }
                EpisodeEndKind::Cancelled => {
                    cancelled += 1;
                    any_cancelled = true;
                    per_episode_reasons.push((*ep as u32, "cancelled".into()));
                    missing_episodes.push(*ep as u32);
                }
                EpisodeEndKind::Missing => {
                    missing_episodes.push(*ep as u32);
                    per_episode_reasons.push((
                        *ep as u32,
                        out.reason.clone().unwrap_or_else(|| "no output".into()),
                    ));
                }
            }
            paths.insert(*ep, out.path);
        }

        let _ = tx.send(EpEvent::FinalSummary {
            downloaded: downloaded as usize,
            skipped: skipped as usize,
            failed: failed as usize,
            cancelled: cancelled as usize,
            per_episode_reasons: per_episode_reasons.clone(),
        });

        if config.manifest_path.is_some() {
            // Persist the derived summary (complete/failed/pending -> counts)
            // so the manifest carries accurate totals, not zeros.
            {
                let mut guard = ctx.manifest.lock().await;
                if let Some(m) = guard.as_mut() {
                    m.summary = m.to_summary();
                }
            }
            self.save_manifest(&ctx).await;
        }

        let outcome = Outcome {
            downloaded,
            skipped,
            failed,
            cancelled: any_cancelled,
            missing_episodes,
            per_episode_reasons,
        };
        (outcome, paths)
    }
}

// ── Internal result types ──────────────────────────────────────────────────

/// How one episode ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpisodeEndKind {
    Downloaded,
    Skipped,
    Failed,
    Cancelled,
    Missing,
}

/// Per-episode outcome returned by [`DownloadEngine::download_episode`].
#[derive(Debug, Clone)]
pub(crate) struct EpisodeResult {
    path: Option<PathBuf>,
    kind: EpisodeEndKind,
    reason: Option<String>,
}

/// Result of trying one mirror (with all its retries).
enum MirrorAttempt {
    Success(PathBuf),
    Failed { reason: String },
    Cancelled,
}

/// Result of one measured mirror (child kept alive for winner continuation).
struct MeasResult {
    url: String,
    child: Option<ChildGuard>,
    avg_bps: Option<f64>,
    meas_stem: PathBuf,
}

/// A failed download attempt (pre-classification).
struct DownloadFailure {
    exit: Option<i32>,
    stderr_tail: Vec<String>,
    bytes: u64,
    secs: f64,
}

/// Tracks the per-episode runtime [`Phase`], emitting the matching events on
/// every transition.
struct PhaseTracker {
    phase: Phase,
}

impl PhaseTracker {
    fn new(episode: i64, tx: &broadcast::Sender<EpEvent>) -> Self {
        let _ = tx.send(EpEvent::Measuring {
            ep: episode,
            host: "queued".into(),
        });
        Self {
            phase: Phase::Queued,
        }
    }

    fn set(&mut self, episode: i64, tx: &broadcast::Sender<EpEvent>, new: Phase) {
        if self.phase == new {
            return;
        }
        self.phase = new.clone();
        match &new {
            Phase::Queued => {
                let _ = tx.send(EpEvent::Measuring {
                    ep: episode,
                    host: "queued".into(),
                });
            }
            Phase::Inspecting => {
                let _ = tx.send(EpEvent::Measuring {
                    ep: episode,
                    host: "inspecting".into(),
                });
            }
            Phase::Measuring => {
                let _ = tx.send(EpEvent::Measuring {
                    ep: episode,
                    host: "measuring".into(),
                });
            }
            Phase::WaitingHost => {
                // MirrorBusy is emitted by the semaphore acquire loop.
            }
            Phase::Downloading { .. } => {
                // Winner/Progress emitted by the download flow.
            }
            Phase::Done { host, size_mb } => {
                let _ = tx.send(EpEvent::Done {
                    ep: episode,
                    host: host.clone(),
                    size_mb: *size_mb,
                });
            }
            Phase::Failed => {
                let _ = tx.send(EpEvent::Failed { ep: episode });
            }
            Phase::Cancelled => {
                let _ = tx.send(EpEvent::Cancelled { ep: episode });
            }
        }
    }
}

/// Convert a cached manifest entry back into a validation outcome.
fn cached_to_outcome(c: &CacheEntry) -> ValidationOutcome {
    ValidationOutcome {
        valid: c.ok,
        reason: c.reason.clone(),
        video_stream: false,
        audio_stream: false,
        duration_secs: None,
        width: None,
        height: None,
        ffprobe_version: c.ffprobe_version.clone(),
    }
}

/// Does `fname` carry a yt-dlp temp marker (`.part`, `.meas`, or a
/// `.<n>.` fragment segment like `.f001.mp4`)?
fn has_temp_marker(fname: &str) -> bool {
    fname.contains(".part")
        || fname.contains(".meas")
        || (fname
            .rsplit_once('.')
            .map(|(stem, _ext)| {
                stem.rsplit_once('.')
                    .map(|(_, seg)| {
                        seg.len() > 1
                            && seg.starts_with('f')
                            && seg[1..].chars().all(|c| c.is_ascii_digit())
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false))
}

/// Is this file name a tool-owned temp artifact of this download?
/// Narrow check: must live in the `{slug}-E*` namespace AND carry a temp
/// marker. A bare `.part`/`.meas` file outside the namespace is a user file
/// and is never touched. (Registry-owned files are handled by registry drop;
/// the broad marker-only check is intentionally not applied to quarantine.)
pub(crate) fn is_tool_owned_artifact(fname: &str, slug: &str) -> bool {
    fname.starts_with(&format!("{slug}-E")) && has_temp_marker(fname)
}

/// Replace URL-like substrings (with query tokens) by `[URL]`, and redact
/// common token parameter values.
pub(crate) fn redact_urls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find("http://").or_else(|| rest.find("https://")) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];
        let end = after
            .find(|c: char| c.is_whitespace() || c == '\'' || c == '"' || c == '`')
            .unwrap_or(after.len());
        out.push_str("[URL]");
        rest = &after[end..];
    }
    out.push_str(rest);

    let mut result = out;
    for key in [
        "token",
        "hash",
        "sig",
        "signature",
        "apikey",
        "api_key",
        "secret",
        "password",
    ] {
        let mut pos = 0;
        while let Some(rel) = result[pos..].find(key) {
            let start = pos + rel;
            // Only redact `key=value` / `key = value` assignments.
            let after_key = &result[start + key.len()..];
            let Some(eq) = after_key.find('=').map(|i| i + key.len()) else {
                pos = start + key.len();
                continue;
            };
            let value_start = start + eq + 1;
            let value_end = result[value_start..]
                .find(['&', ' ', '\'', '"', '`'])
                .map(|i| value_start + i)
                .unwrap_or(result.len());
            if value_end > value_start {
                result.replace_range(value_start..value_end, "[REDACTED]");
                pos = value_start + "[REDACTED]".len();
            } else {
                pos = value_start;
            }
        }
    }
    result
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Video extensions accepted as completed downloads.
const VIDEO_EXTS: &[&str] = &["mkv", "mp4", "webm", "flv", "mov", "avi"];

/// Minimum valid download size in bytes (1 MB). Rejects tiny stubs/error pages
/// while allowing legitimate short episodes (legitimate minimum ~50MB for 1min 1080p).
/// Used as the floor for the extension+size heuristic; a raised
/// [`ValidationConfig::min_size_bytes`] (CLI `--min-size-mb`) lifts the
/// effective floor via [`DownloadEngine::heuristic_min_bytes`].
pub(crate) const MIN_VALID_DOWNLOAD_BYTES: u64 = 1_048_576;

/// Check if a path is a valid completed download: supported extension + size
/// at least `min_bytes`.
pub(crate) fn is_valid_output_with_min(path: &Path, min_bytes: u64) -> bool {
    if !path.is_file() {
        return false;
    }
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e,
        None => return false,
    };
    if !VIDEO_EXTS.contains(&ext) {
        return false;
    }
    match fs::metadata(path).map(|m| m.len()) {
        Ok(size) => size >= min_bytes,
        Err(_) => false,
    }
}

/// Check if a path is a valid completed download: supported extension + minimum
/// size (legacy 1 MB heuristic floor).
pub(crate) fn is_valid_output(path: &Path) -> bool {
    is_valid_output_with_min(path, MIN_VALID_DOWNLOAD_BYTES)
}

fn parse_speed_from_line(line: &str) -> Option<f64> {
    let line = line.trim();
    if !line.contains("[download]") {
        return None;
    }
    let idx = line.find(" at ")?;
    let after = &line[idx + 4..];
    let speed_str = after.split_whitespace().next()?;
    parse_speed_bps(speed_str)
}

/// Recursively collect all files under `dir` (for manifest reconciliation).
fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

/// Find output file matching stem. Note: cannot match .mkv.part files
/// (double-extension issue — file_stem is "ep1.mkv" not "ep1").
/// Only files at least `min_bytes` qualify (effective size floor).
pub(crate) fn find_output_file(stem: &Path, min_bytes: u64) -> Option<PathBuf> {
    let parent = stem.parent()?;
    let name = stem.file_name()?;
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if path.file_stem() == Some(name) && is_valid_output_with_min(&path, min_bytes) {
            return Some(path);
        }
    }
    None
}

/// Find the completed measurement output for exactly `meas_stem`
/// (e.g. `stem.meas0.mkv` for meas_stem=`.../stem.meas0`). Scanning is
/// constrained to this stem's prefix so a stale meas file from a different
/// mirror/run can never be picked up by mistake.
fn find_meas_output(meas_stem: &Path) -> Option<PathBuf> {
    let parent = meas_stem.parent()?;
    let name = meas_stem.file_name()?;
    let prefix = format!("{}.", name.to_string_lossy());
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(fname) = path.file_name() else {
            continue;
        };
        let fname = fname.to_string_lossy();
        if !fname.starts_with(&prefix) {
            continue;
        }
        if !path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| VIDEO_EXTS.contains(&e))
            .unwrap_or(false)
        {
            continue;
        }
        if fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false) {
            return Some(path);
        }
    }
    None
}

/// Remove every file under the `{stem}.` prefix — the partial artifacts of a
/// killed loser mirror (`.part`, fragments, or a meas file it finalized just
/// before being killed). Called only for losers after they are killed+reaped,
/// so no live process can recreate them.
fn cleanup_meas_stem(stem: &Path) {
    let Some(parent) = stem.parent() else {
        return;
    };
    let Some(name) = stem.file_name() else {
        return;
    };
    let prefix = format!("{}.", name.to_string_lossy());
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(fname) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if fname.starts_with(&prefix) {
            let _ = fs::remove_file(path);
        }
    }
}

/// Find any `.part` file matching stem (e.g. `stem.mp4.part`, `stem.mkv.part`).
/// Scans parent dir for `{stem}.*.part` patterns.
pub(crate) fn find_part_file(stem: &Path) -> Option<PathBuf> {
    let parent = stem.parent()?;
    let name = stem.file_name()?;
    let prefix = format!("{}.", name.to_string_lossy());
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(fname) = path.file_name() else {
            continue;
        };
        let fname = fname.to_string_lossy();
        if fname.starts_with(&prefix)
            && fname.ends_with(".part")
            && fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false)
        {
            return Some(path);
        }
    }
    None
}

/// Find `.part` file matching stem (e.g. `stem.mp4.part`).
/// Scans parent dir for `{stem}.*.part` patterns.
/// Unlike `find_stem_output`, does not require completed output.
pub(crate) fn find_stem_part_file(stem: &Path) -> Option<PathBuf> {
    let parent = stem.parent()?;
    let name = stem.file_name()?;
    let prefix = format!("{}.", name.to_string_lossy());
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = path.file_name()?.to_string_lossy();
        if fname.starts_with(&prefix) && fname.ends_with(".part") {
            return Some(path);
        }
    }
    None
}

/// Find output file matching stem (any extension, excluding .part files).
/// Returns e.g. `stem.mp4` or `stem.mkv` for stem=`out_dir/name`.
pub(crate) fn find_stem_output(stem: &Path) -> Option<PathBuf> {
    let parent = stem.parent()?;
    let name = stem.file_name()?;
    let prefix = format!("{}.", name.to_string_lossy());
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(fname) = path.file_name() else {
            continue;
        };
        let fname = fname.to_string_lossy();
        if fname.starts_with(&prefix)
            && !fname.ends_with(".part")
            && fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false)
            && path
                .extension()
                .map(|e| VIDEO_EXTS.contains(&e.to_str().unwrap_or("")))
                .unwrap_or(false)
        {
            return Some(path);
        }
    }
    None
}

/// Legacy prefix-based cleanup helper, kept for tests (engine uses
/// [`TempRegistry`] instead).
#[cfg(test)]
pub(crate) async fn cleanup_stale_part(path: &Path) {
    let _ = remove_file(path).await;
    // Scan parent dir for files prefixed with `{stem}.` (bare file name)
    // — catches `{stem}.meas0.mp4.part` and other yt-dlp output.
    // Uses file_name() to match bare filename, not full path.
    if let Some(parent) = path.parent()
        && let Some(fname) = path.file_name()
    {
        let stem_name = fname.to_string_lossy();
        let prefix = format!("{stem_name}.");
        if let Ok(mut entries) = read_dir(parent).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let efname = entry.file_name().to_string_lossy().to_string();
                if efname.starts_with(&prefix) {
                    let _ = remove_file(entry.path()).await;
                }
            }
        }
    }
}

pub(crate) fn existing_download(stem: &Path, min_bytes: u64) -> Option<PathBuf> {
    find_output_file(stem, min_bytes)
}

#[cfg(test)]
mod real_factory_tests {
    use super::*;

    /// `ytdlp_extra_args` must be appended to EVERY yt-dlp invocation after
    /// the standard engine args and before the URL. The [`SubprocessFactory`]
    /// trait signature is deliberately stable (no per-call args), so the
    /// passthrough point is [`RealYtDlpFactory`]'s arg builder — tested here
    /// without spawning real yt-dlp.
    #[test]
    fn ytdlp_extra_args_appended_to_commands() {
        let factory = RealYtDlpFactory {
            extra_args: vec!["--http-chunk-size".into(), "10M".into()],
        };
        let url = "https://example.com/v.mp4";
        for (kind, args) in [
            ("measure", factory.measure_args("/tmp/stem", url)),
            ("download", factory.download_args("/tmp/stem", false, url)),
            ("resume", factory.download_args("/tmp/stem", true, url)),
            ("inspect", factory.inspect_args(url)),
        ] {
            let url_pos = args
                .iter()
                .position(|a| a == url)
                .unwrap_or_else(|| panic!("{kind}: url must be last"));
            assert_eq!(
                &args[url_pos - 2..url_pos],
                ["--http-chunk-size", "10M"],
                "{kind}: extra args immediately before url, after standard args"
            );
            assert!(
                args.iter().any(|a| a == "-f") && args.iter().any(|a| a == "bv*+ba/b"),
                "{kind}: standard format args retained"
            );
        }
        // The `-c` continue flag precedes extra args on the resume path.
        let resume = factory.download_args("/tmp/stem", true, url);
        let url_pos = resume.iter().position(|a| a == url).unwrap();
        assert_eq!(resume[url_pos - 3], "-c", "resume keeps -c before extras");
    }
}

#[cfg(test)]
mod timeout_profile_tests {
    use super::*;

    #[test]
    fn profile_for_url_cda_uses_slow_startup() {
        let p = profile_for_url("https://cda.pl/video/123");
        assert_eq!(p.startup_secs, 45);
        assert_eq!(p.stall_secs, 120);
    }

    #[test]
    fn profile_for_url_hqq_uses_slow_startup_and_long_stall() {
        let p = profile_for_url("https://hqq.tv/watch/abc");
        assert_eq!(p.startup_secs, 60);
        assert_eq!(p.stall_secs, 180);
    }

    #[test]
    fn profile_for_url_rumble_uses_long_stall() {
        let p = profile_for_url("https://rumble.com/embed/abc/");
        assert_eq!(p.startup_secs, 30);
        assert_eq!(p.stall_secs, 180);
    }

    #[test]
    fn profile_for_url_generic_uses_defaults() {
        let p = profile_for_url("https://example.com/v.mp4");
        assert_eq!(p.startup_secs, 30);
        assert_eq!(p.stall_secs, 120);
    }

    #[test]
    fn profile_for_url_unknown_scheme_falls_back_to_defaults() {
        let p = profile_for_url("ftp://example.com/v.mp4");
        assert_eq!(
            p,
            TimeoutProfile {
                startup_secs: 30,
                stall_secs: 120,
            }
        );
    }
}

#[cfg(test)]
mod backoff_tests {
    use super::*;

    fn test_engine(jitter: f64) -> DownloadEngine {
        let config = DownloadConfig {
            backoff_base_secs: 2.0,
            backoff_cap_secs: 60.0,
            jitter_secs: jitter,
            ..DownloadConfig::default()
        };
        DownloadEngine::new(config)
    }

    /// Pure computation, no sleeps: capped at `backoff_cap_secs` plus
    /// jitter below 1s.
    #[test]
    fn backoff_capped_at_cap_secs_plus_jitter() {
        let e = test_engine(1.0);
        let v = e.backoff_for(5, "https://cdn.example.com/v.mp4", 3);
        assert!((60.0..61.0).contains(&v), "retry=5 must cap at 60s: {v}");
        let v10 = e.backoff_for(10, "https://cdn.example.com/v.mp4", 3);
        assert!((60.0..61.0).contains(&v10), "retry=10 also capped: {v10}");
    }

    #[test]
    fn backoff_retry_zero_is_base_plus_jitter() {
        let e = test_engine(1.0);
        let v = e.backoff_for(0, "https://cdn.example.com/v.mp4", 3);
        assert!((2.0..3.0).contains(&v), "retry=0 → base 2 + jitter: {v}");
    }

    #[test]
    fn backoff_jitter_within_bounds() {
        let e = test_engine(0.5);
        for retry in 0..6 {
            let base = (2.0 * 2f64.powi(retry as i32)).min(60.0);
            for ep in 1..=3 {
                let v = e.backoff_for(retry, "https://cdn.example.com/v.mp4", ep);
                let jitter = v - base;
                assert!(
                    (0.0..0.5).contains(&jitter),
                    "retry {retry} ep {ep}: jitter {jitter} must be in [0, 0.5)"
                );
            }
        }
    }

    #[test]
    fn backoff_deterministic_same_inputs() {
        let e = test_engine(1.0);
        let a = e.backoff_for(2, "https://cdn.example.com/v.mp4", 7);
        let b = e.backoff_for(2, "https://cdn.example.com/v.mp4", 7);
        assert_eq!(a, b, "identical (retry,url,episode) → identical backoff");
        // A second engine with the same config is equally deterministic.
        let e2 = test_engine(1.0);
        assert_eq!(
            e2.backoff_for(2, "https://cdn.example.com/v.mp4", 7),
            a,
            "same seed across engines"
        );
        let c = e.backoff_for(2, "https://cdn.example.com/v.mp4", 8);
        assert_ne!(a, c, "different episode → different seed");
    }

    #[test]
    fn backoff_jitter_disabled_is_exact_capped_base() {
        let e = test_engine(0.0);
        assert_eq!(e.backoff_for(0, "https://cdn.example.com/v.mp4", 1), 2.0);
        assert_eq!(e.backoff_for(3, "https://cdn.example.com/v.mp4", 1), 16.0);
        assert_eq!(e.backoff_for(5, "https://cdn.example.com/v.mp4", 1), 60.0);
        assert_eq!(e.backoff_for(8, "https://cdn.example.com/v.mp4", 1), 60.0);
    }
}
