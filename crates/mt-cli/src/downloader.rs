//! Concurrent anime episode download engine.
//!
//! Uses tokio for async process management. Quality-first mirror selection,
//! concurrent mirror race with measurement interval, per-host semaphore,
//! cancellation via `CancellationToken`. Winner process continues without
//! restart after measurement phase.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use tokio::fs::read_dir;
use tokio::fs::{create_dir_all, remove_file, rename};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Semaphore, broadcast};
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep, timeout};
use tokio_util::sync::CancellationToken;

use crate::download_types::{
    EpEvent, EpisodeInput, Quality, host_preference_rank, parse_speed_bps,
};

// ── Constants ──────────────────────────────────────────────────────────────

const MEASUREMENT_SECS: u64 = 3;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const STALL_TIMEOUT: Duration = Duration::from_secs(120);
const OVERALL_TIMEOUT: Duration = Duration::from_secs(7200);
const DEFAULT_HOST_CONCURRENCY: usize = 1;
const DEFAULT_EP_CONCURRENCY: usize = 4;

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
#[derive(Default)]
pub struct RealYtDlpFactory;

impl SubprocessFactory for RealYtDlpFactory {
    fn spawn_measure(
        &self,
        url: &str,
        out_path: &Path,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>> {
        let url = url.to_string();
        let stem_str = out_path.to_string_lossy().to_string();
        Box::pin(async move {
            let mut cmd = Command::new("yt-dlp");
            #[cfg(unix)]
            {
                cmd.process_group(0);
            }
            cmd.arg("--progress")
                .arg("--newline")
                .arg("-f")
                .arg("bv*+ba/b")
                .arg("-o")
                .arg(format!("{stem_str}.%(ext)s"))
                .arg("--merge-output-format")
                .arg("mkv")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .arg(&url);

            let child = cmd.spawn().ok()?;
            let pgid = child.id().unwrap_or(0);
            Some(RunningSubprocess { child, pgid })
        })
    }

    fn spawn_download(
        &self,
        url: &str,
        out_path: &Path,
    ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>> {
        let url = url.to_string();
        let stem = out_path.with_extension("");
        let stem_str = stem.to_string_lossy().to_string();
        Box::pin(async move {
            let mut cmd = Command::new("yt-dlp");
            #[cfg(unix)]
            {
                cmd.process_group(0);
            }
            cmd.arg("--progress")
                .arg("--newline")
                .arg("--force-overwrites")
                .arg("-f")
                .arg("bv*+ba/b")
                .arg("-o")
                .arg(format!("{stem_str}.%(ext)s"))
                .arg("--merge-output-format")
                .arg("mkv")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .arg(&url);

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
        Box::pin(async move {
            let mut cmd = Command::new("yt-dlp");
            #[cfg(unix)]
            {
                cmd.process_group(0);
            }
            cmd.arg("--dump-json")
                .arg("-f")
                .arg("bv*+ba/b")
                .arg("--no-download")
                .arg(&url)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true);

            let child = cmd.spawn().ok()?;
            let pgid = child.id().unwrap_or(0);

            let result = timeout(STARTUP_TIMEOUT, child.wait_with_output()).await;

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

/// Fake factory for tests.
#[cfg(test)]
pub mod test_factory {
    use std::path::Path;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[derive(Default)]
    pub struct FakeFactory {
        pub fail_measure: AtomicBool,
        pub fail_download: AtomicBool,
        pub fake_quality: AtomicBool,
    }

    impl FakeFactory {
        pub fn new() -> Self {
            Self {
                fail_measure: AtomicBool::new(false),
                fail_download: AtomicBool::new(false),
                fake_quality: AtomicBool::new(false),
            }
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
                let mut cmd = Command::new("echo");
                cmd.arg("test");
                let child = cmd.spawn().ok()?;
                let pgid = child.id().unwrap_or(0);
                Some(RunningSubprocess { child, pgid })
            })
        }

        fn spawn_download(
            &self,
            _url: &str,
            _out_path: &Path,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>> {
            let fail = self.fail_download.load(Ordering::SeqCst);
            Box::pin(async move {
                if fail {
                    return None;
                }
                let mut cmd = Command::new("echo");
                cmd.arg("test");
                let child = cmd.spawn().ok()?;
                let pgid = child.id().unwrap_or(0);
                Some(RunningSubprocess { child, pgid })
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

// ── Download engine ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub episode_concurrency: usize,
    pub host_concurrency: usize,
    pub measurement_secs: u64,
    pub out_dir: PathBuf,
    pub slug: String,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            episode_concurrency: DEFAULT_EP_CONCURRENCY,
            host_concurrency: DEFAULT_HOST_CONCURRENCY,
            measurement_secs: MEASUREMENT_SECS,
            out_dir: PathBuf::from("."),
            slug: "anime".to_string(),
        }
    }
}

/// The anime download engine.
#[derive(Clone)]
pub struct DownloadEngine {
    pub config: DownloadConfig,
    pub cancel: CancellationToken,
    factory: Arc<dyn SubprocessFactory>,
    host_semaphores: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
}

impl DownloadEngine {
    pub fn new(config: DownloadConfig) -> Self {
        Self {
            cancel: CancellationToken::new(),
            host_semaphores: Arc::new(Mutex::new(HashMap::new())),
            factory: Arc::new(RealYtDlpFactory),
            config,
        }
    }

    pub fn with_factory(config: DownloadConfig, factory: Arc<dyn SubprocessFactory>) -> Self {
        Self {
            cancel: CancellationToken::new(),
            host_semaphores: Arc::new(Mutex::new(HashMap::new())),
            factory,
            config,
        }
    }

    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    async fn host_semaphore(&self, host: &str) -> Arc<Semaphore> {
        let mut map = self.host_semaphores.lock().await;
        let permits = self.config.host_concurrency;
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

    /// Clean up temp files for a stem: .part files and .meas*.part files.
    async fn cleanup_temp_files(stem: &Path) {
        if let Some(parent) = stem.parent()
            && let Ok(mut entries) = read_dir(parent).await
        {
            let name = stem.file_name().map(|n| n.to_string_lossy().to_string());
            if let Some(name) = name {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let fname = entry.file_name().to_string_lossy().to_string();
                    if fname == format!("{}.mkv.part", name)
                        || fname.starts_with(&format!("{}.meas", name))
                    {
                        let _ = remove_file(entry.path()).await;
                    }
                }
            }
        }
    }

    /// Run an episode download.
    pub async fn download_episode(
        &self,
        ep: &EpisodeInput,
        tx: broadcast::Sender<EpEvent>,
    ) -> Option<PathBuf> {
        let episode = ep.episode;
        let _ = tx.send(EpEvent::Measuring {
            ep: episode,
            host: "queued".into(),
        });

        let stem = self
            .config
            .out_dir
            .join(format!("{}-E{:02}", self.config.slug, episode));

        // Check if already downloaded
        if let Some(existing) = existing_download(&stem) {
            let size_mb = fs::metadata(&existing)
                .map(|m| m.len() as f64 / 1_048_576.0)
                .unwrap_or(0.0);
            let _ = tx.send(EpEvent::Done {
                ep: episode,
                host: "cached".into(),
                size_mb,
            });
            return Some(existing);
        }

        let mut mirrors: Vec<(String, Quality)> = Vec::new();
        for url in &ep.urls {
            let quality = ep
                .quality
                .as_ref()
                .map(|q| Quality::new(q.height))
                .unwrap_or(Quality::new(0));
            mirrors.push((url.clone(), quality));
        }

        if mirrors.is_empty() {
            let _ = tx.send(EpEvent::Failed { ep: episode });
            return None;
        }

        // B6: Check cancellation before inspection phase
        if self.cancel.is_cancelled() {
            let _ = tx.send(EpEvent::Cancelled { ep: episode });
            return None;
        }

        // Inspect unknown-quality mirrors concurrently
        let unknown_mirrors: Vec<usize> = mirrors
            .iter()
            .enumerate()
            .filter(|(_, (_, q))| q.is_unknown())
            .map(|(i, _)| i)
            .collect();

        if !unknown_mirrors.is_empty() {
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

        // Quality-first filter
        let max_height = mirrors.iter().map(|(_, q)| q.height).max().unwrap_or(0);
        let eligible: Vec<(String, Quality)> = mirrors
            .into_iter()
            .filter(|(_, q)| q.height == max_height)
            .collect();

        let _ = tx.send(EpEvent::Measuring {
            ep: episode,
            host: format!("{} eligible mirrors", eligible.len()),
        });

        // Single mirror: skip measurement race
        if eligible.len() == 1 {
            let (url, _quality) = &eligible[0];
            let host = Self::host_from_url(url);
            let sem = self.host_semaphore(&host).await;
            let _permit = sem.acquire().await;
            let _ = tx.send(EpEvent::Winner {
                ep: episode,
                host: host.clone(),
            });
            return self
                .perform_download_with_child(episode, url, &stem, host.clone(), tx, None)
                .await;
        }

        // BLOCKER 1: Concurrent measurement race — write to real .part files
        let measure_dur = Duration::from_secs(self.config.measurement_secs);

        // Each measurement returns: (url, child, avg_bps, meas_stem)
        struct MeasResult {
            url: String,
            child: Option<RunningSubprocess>,
            avg_bps: Option<f64>,
            meas_stem: PathBuf,
        }

        let mut meas_set = JoinSet::new();

        for (idx, (url, _quality)) in eligible.iter().enumerate() {
            let url = url.clone();
            let factory = self.factory.clone();
            let cancel = self.cancel.clone();
            let duration = measure_dur;
            // Unique measurement output stem
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

                // Keep child alive — don't kill it
                MeasResult {
                    url,
                    child: Some(proc),
                    avg_bps,
                    meas_stem,
                }
            });
        }

        // Collect results
        let mut meas_results: Vec<MeasResult> = Vec::new();
        while let Some(result) = meas_set.join_next().await {
            if let Ok(res) = result {
                let host_str = Self::host_from_url(&res.url);
                meas_results.push(res);
                if let Some(bps) = meas_results.last().and_then(|r| r.avg_bps) {
                    let _ = tx.send(EpEvent::Measured {
                        ep: episode,
                        host: host_str,
                        bps,
                    });
                }
            }
        }

        let _ = tx.send(EpEvent::MeasurementComplete { ep: episode });

        // Sort by speed desc, then host pref, then URL
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

        // Kill losers, keep winner alive
        let losers = meas_results.iter_mut().skip(1);

        for loser in losers {
            if let Some(mut child) = loser.child.take() {
                child.kill_group();
                let _ = child.child.wait().await;
            }
            // Clean up loser's measurement temp files
            cleanup_stale_part(&loser.meas_stem).await;
        }

        // Try mirrors in order (winner first)
        for meas in meas_results.iter_mut() {
            if self.cancel.is_cancelled() {
                if let Some(mut child) = meas.child.take() {
                    child.kill_group();
                    let _ = child.child.wait().await;
                    cleanup_stale_part(&meas.meas_stem).await;
                }
                Self::cleanup_temp_files(&stem).await;
                let _ = tx.send(EpEvent::Cancelled { ep: episode });
                return None;
            }

            let host = Self::host_from_url(&meas.url);
            let sem = self.host_semaphore(&host).await;
            let permit = match timeout(Duration::from_secs(5), sem.acquire()).await {
                Ok(Ok(p)) => p,
                _ => {
                    // Kill this mirror's child if it's still alive
                    if let Some(mut child) = meas.child.take() {
                        child.kill_group();
                        let _ = child.child.wait().await;
                        cleanup_stale_part(&meas.meas_stem).await;
                    }
                    let _ = tx.send(EpEvent::MirrorBusy {
                        ep: episode,
                        host: host.clone(),
                    });
                    continue;
                }
            };

            let _ = tx.send(EpEvent::Winner {
                ep: episode,
                host: host.clone(),
            });

            // Rename winner's measurement output file to real part path
            let part_path = stem.with_extension("mkv.part");
            let meas_output = meas.meas_stem.with_extension("mkv");
            let _ = rename(&meas_output, &part_path).await;

            let result = self
                .perform_download_with_child(
                    episode,
                    &meas.url,
                    &stem,
                    host.clone(),
                    tx.clone(),
                    meas.child.take(),
                )
                .await;

            drop(permit);

            // If this was the first (winner) and it failed, clean up and try next
            if result.is_some() {
                return result;
            }

            // Clean up after failed attempt
            let _ = remove_file(&part_path).await;
        }

        Self::cleanup_temp_files(&stem).await;
        let _ = tx.send(EpEvent::Failed { ep: episode });
        None
    }

    /// Perform download from a given URL, optionally continuing an existing child.
    #[allow(clippy::too_many_arguments)]
    async fn perform_download_with_child(
        &self,
        episode: i64,
        url: &str,
        stem: &Path,
        host: String,
        tx: broadcast::Sender<EpEvent>,
        existing_child: Option<RunningSubprocess>,
    ) -> Option<PathBuf> {
        if let Some(parent) = stem.parent() {
            let _ = create_dir_all(parent).await;
        }

        let part_path = stem.with_extension("mkv.part");

        let (mut child, started) = if let Some(existing) = existing_child {
            // BLOCKER 1: Continue with existing child — already downloading to .part
            (existing, true)
        } else {
            // Spawn new download
            let proc = self.factory.spawn_download(url, stem).await;

            let Some(proc) = proc else {
                let _ = tx.send(EpEvent::MirrorDone {
                    ep: episode,
                    host: host.clone(),
                    success: false,
                });
                return None;
            };
            (proc, false)
        };

        let start = Instant::now();
        let mut started_flag = started;
        let mut stall_start: Option<Instant> = None;
        let mut prev_bytes = 0u64;
        let mut last_progress = Instant::now();

        // Take stdout for progress reading
        let stdout = child.child.stdout.take();
        let mut reader = stdout.map(|o| BufReader::new(o).lines());

        // Monitor download with progress
        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    child.kill_group();
                    let _ = child.child.wait().await;
                    Self::cleanup_temp_files(stem).await;
                    let _ = tx.send(EpEvent::Cancelled { ep: episode });
                    return None;
                }
                _ = sleep(Duration::from_millis(200)) => {
                    // Startup timeout: if child hasn't produced output yet
                    if !started_flag {
                        if start.elapsed() > STARTUP_TIMEOUT {
                            child.kill_group();
                            let _ = child.child.wait().await;
                            cleanup_stale_part(&part_path).await;
                            let _ = tx.send(EpEvent::Failed { ep: episode });
                            return None;
                        }
                        if fs::metadata(&part_path).map(|m| m.len() > 0).unwrap_or(false) {
                            started_flag = true;
                        }
                    } else {
                        // Stall check
                        let current_bytes = fs::metadata(&part_path)
                            .map(|m| m.len())
                            .unwrap_or(0);
                        if current_bytes == prev_bytes {
                            match stall_start {
                                None => stall_start = Some(Instant::now()),
                                Some(s) if s.elapsed() > STALL_TIMEOUT => {
                                    child.kill_group();
                                    let _ = child.child.wait().await;
                                    cleanup_stale_part(&part_path).await;
                                    let _ = tx.send(EpEvent::Failed { ep: episode });
                                    return None;
                                }
                                _ => {}
                            }
                        } else {
                            stall_start = None;
                        }
                        prev_bytes = current_bytes;
                    }

                    if start.elapsed() > OVERALL_TIMEOUT {
                        child.kill_group();
                        let _ = child.child.wait().await;
                        let _ = tx.send(EpEvent::Failed { ep: episode });
                        return None;
                    }
                }
                line_result = async {
                    if let Some(ref mut reader) = reader {
                        tokio::time::timeout(Duration::from_millis(100), reader.next_line()).await
                    } else {
                        // No reader available — sleep briefly
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        Ok(Ok(None))
                    }
                } => {
                    match line_result {
                        Ok(Ok(Some(line))) => {
                            if let Some(_bps) = parse_speed_from_line(&line) {
                                // Parse pct, speed str, eta from line
                                // yt-dlp format: [download]  45.2% of ~417.22MiB at  7.50MiB/s ETA 02:15
                                let pct = line.split('%').next()
                                    .and_then(|s| s.rsplit(' ').next())
                                    .and_then(|s| s.trim().parse::<f64>().ok())
                                    .unwrap_or(0.0);
                                let eta = line.split("ETA ")
                                    .nth(1)
                                    .map(|s| s.trim().to_string())
                                    .unwrap_or_default();
                                let speed_str = line.rsplit(" at ")
                                    .nth(1)
                                    .and_then(|s| s.split(' ').next())
                                    .unwrap_or("")
                                    .to_string();
                                let total_str = line.split("of ~")
                                    .nth(1)
                                    .and_then(|s| s.split(' ').next())
                                    .unwrap_or("");
                                let total_bytes = parse_speed_bps(total_str).unwrap_or(0.0) as u64;
                                let downloaded = (total_bytes as f64 * pct / 100.0) as u64;

                                // Rate-limit progress sends to ~250ms
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
                        }
                        _ => {
                            // No more lines or timeout — continue
                        }
                    }
                }
                status = child.child.wait() => {
                    match status {
                        Ok(s) if s.success() => {
                            if let Some(path) = find_output_file(stem) {
                                let size_mb = fs::metadata(&path)
                                    .map(|m| m.len() as f64 / 1_048_576.0)
                                    .unwrap_or(0.0);
                                let _ = tx.send(EpEvent::Done {
                                    ep: episode,
                                    host: host.clone(),
                                    size_mb,
                                });
                                return Some(path);
                            }
                            // If continuing child: try to rename meas file
                            if started
                                && let Some(meas_path) = find_meas_output(stem) {
                                    let final_name = stem.with_extension(
                                        meas_path.extension().unwrap_or_default()
                                    );
                                    let _ = rename(&meas_path, &final_name).await;
                                    let size_mb = fs::metadata(&final_name)
                                        .map(|m| m.len() as f64 / 1_048_576.0)
                                        .unwrap_or(0.0);
                                    let _ = tx.send(EpEvent::Done {
                                        ep: episode,
                                        host: host.clone(),
                                        size_mb,
                                    });
                                    return Some(final_name);
                                }
                            // Fallback: check part_path directly (winner continuation writes here)
                            if fs::metadata(&part_path).map(|m| m.len() > 0).unwrap_or(false) {
                                let final_path = stem.with_extension("mkv");
                                let _ = rename(&part_path, &final_path).await;
                                let size_mb = fs::metadata(&final_path)
                                    .map(|m| m.len() as f64 / 1_048_576.0)
                                    .unwrap_or(0.0);
                                let _ = tx.send(EpEvent::Done {
                                    ep: episode,
                                    host: host.clone(),
                                    size_mb,
                                });
                                return Some(final_path);
                            }
                        }
                        _ => {
                            let _ = tx.send(EpEvent::MirrorDone {
                                ep: episode,
                                host: host.clone(),
                                success: false,
                            });
                        }
                    }
                    return None;
                }
            }
        }
    }

    /// BLOCKER 3: Run multiple episodes with concurrent retry support.
    pub async fn run_all(
        &self,
        episodes: Vec<EpisodeInput>,
        tx: broadcast::Sender<EpEvent>,
    ) -> Vec<(i64, Option<PathBuf>)> {
        let self_arc = Arc::new(self.clone());
        let sem = Arc::new(Semaphore::new(self.config.episode_concurrency));
        let mut results: HashMap<i64, Option<PathBuf>> = HashMap::new();
        let mut terminal: HashSet<i64> = HashSet::new();
        let ep_map: HashMap<i64, EpisodeInput> =
            episodes.into_iter().map(|e| (e.episode, e)).collect();
        let total_eps: HashSet<i64> = ep_map.keys().copied().collect();
        let mut active = JoinSet::new();
        let cancel = self.cancel.clone();

        // Spawn initial batch: acquire permits and spawn into JoinSet
        for ep_num in &total_eps {
            let Some(ep) = ep_map.get(ep_num) else {
                continue;
            };
            let ep = ep.clone();
            let permit = sem.clone().acquire_owned().await;
            let engine = self_arc.clone();
            let tx = tx.clone();

            active.spawn(async move {
                let _permit = permit;
                let result = engine.download_episode(&ep, tx).await;
                (ep.episode, result)
            });
        }

        // Collect results
        loop {
            if terminal.len() == total_eps.len() && active.is_empty() {
                break;
            }

            tokio::select! {
                Some(result) = active.join_next() => {
                    if let Ok((ep, path)) = result {
                        results.insert(ep, path);
                        terminal.insert(ep);
                    }
                }
                _ = sleep(Duration::from_millis(200)) => {
                    if cancel.is_cancelled() {
                        // Drain remaining tasks
                        while let Some(result) = active.join_next().await {
                            if let Ok((ep, path)) = result {
                                results.insert(ep, path);
                                terminal.insert(ep);
                            }
                        }
                        break;
                    }
                }
            }
        }

        let mut final_results: Vec<(i64, Option<PathBuf>)> = total_eps
            .iter()
            .map(|&ep| (ep, results.remove(&ep).unwrap_or(None)))
            .collect();
        final_results.sort_by_key(|(ep, _)| *ep);
        final_results
    }

    /// Legacy wrapper for backward compat (no child reuse).
    #[allow(dead_code)]
    async fn perform_download(
        &self,
        episode: i64,
        url: &str,
        stem: &Path,
        host: String,
        tx: broadcast::Sender<EpEvent>,
    ) -> Option<PathBuf> {
        self.perform_download_with_child(episode, url, stem, host, tx, None)
            .await
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

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

/// Find output file matching stem. Note: cannot match .mkv.part files
/// (double-extension issue — file_stem is "ep1.mkv" not "ep1").
fn find_output_file(stem: &Path) -> Option<PathBuf> {
    let parent = stem.parent()?;
    let name = stem.file_name()?;
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if path.file_stem() == Some(name)
            && path.is_file()
            && fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false)
        {
            return Some(path);
        }
    }
    None
}

/// Find a measurement output file (e.g. stem.meas0.mkv).
fn find_meas_output(stem: &Path) -> Option<PathBuf> {
    let parent = stem.parent()?;
    let name = stem.file_name()?;
    let prefix = format!("{}.meas", name.to_string_lossy());
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if let Some(fname) = path.file_stem() {
            let fname = fname.to_string_lossy();
            if fname.starts_with(&prefix)
                && path.is_file()
                && fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false)
            {
                return Some(path);
            }
        }
    }
    None
}

async fn cleanup_stale_part(path: &Path) {
    let _ = remove_file(path).await;
    // Also try with .part extension
    let part = path.with_extension("mkv.part");
    let _ = remove_file(&part).await;
    // For meas stems (e.g. slug-E01.meas0), also remove actual output (slug-E01.meas0.mkv)
    let mkv = path.with_extension("mkv");
    let _ = remove_file(&mkv).await;
}

fn existing_download(stem: &Path) -> Option<PathBuf> {
    find_output_file(stem)
}
