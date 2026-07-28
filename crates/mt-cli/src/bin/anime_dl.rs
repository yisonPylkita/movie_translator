//! `anime-dl` — standalone ogladajanime.pl season downloader.
//!
//! Pass an anime name, or a plain-text list via `--file`. The tool finds each
//! title on ogladajanime.pl, opens the browser, and waits for the resolver
//! userscript to drop its players JSON in Downloads. It then downloads the
//! requested episodes at best available quality. No translation, no OCR —
//! pure download. Reuses the `mt_fetch::ogladajanime` discovery/pickup
//! plumbing and `mt_ml::hardsub_download` (with `best = true`).
//!
//! Downloads are network/CPU only (no GPU worker, no TUI). Episodes from the
//! same host are serialised; different hosts run concurrently.

use std::collections::HashMap;
use std::fs::read_to_string;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use mt_cli::tui_download::{DownloadUi, EpEvent};
use mt_fetch::ogladajanime::{self, Discovery, HardsubPlan};
use tracing::{error, warn};

/// How often to poll Downloads for the resolver JSON.
const POLL: Duration = Duration::from_secs(2);

#[derive(Parser, Debug)]
#[command(
    name = "anime-dl",
    about = "Download anime episodes from ogladajanime.pl (best quality, no translation)",
    after_help = "\
LIST FILE FORMAT (--file):
  One entry per line. Blank lines and lines starting with # are ignored.

    One Piece                 # whole series / season (all episodes)
    Naruto 1                  # episode 1 only
    Naruto E02                # episode 2 only
    Bleach S01E03             # episode 3 only (season tag ignored for matching)

  Same-title lines are grouped so the browser/userscript runs once per anime.
"
)]
struct Args {
    /// Anime name to search for on ogladajanime.pl. Optional when `--json` or
    /// `--file` is given.
    name: Option<String>,

    /// Path to a plain-text list of anime / episode names (see --help).
    /// Processes each entry sequentially; same title grouped into one pickup.
    #[arg(short = 'f', long, value_name = "PATH")]
    file: Option<PathBuf>,

    /// Output directory. Default: `./<slug>` (per anime when using `--file`).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Directory to watch for the resolver userscript JSON.
    /// Default: the system Downloads directory.
    #[arg(long)]
    downloads_dir: Option<PathBuf>,

    /// Seconds to wait for the resolver JSON to appear (per anime).
    #[arg(long, default_value_t = 600)]
    timeout: u64,

    /// Parse an existing resolver JSON instead of opening the browser.
    /// Handy for re-runs without re-running the userscript. Single-anime only
    /// (ignored with `--file` multi-title lists).
    #[arg(long)]
    json: Option<PathBuf>,

    /// Only download these episode numbers (comma-separated, e.g. `20,21`).
    /// Applies to single-name / `--json` runs. For multi-title lists, put the
    /// episode on each `--file` line instead.
    #[arg(long, value_name = "N,N,...", value_delimiter = ',')]
    episodes: Option<Vec<i64>>,

    /// Raise our crates' logging to DEBUG.
    #[arg(short, long)]
    verbose: bool,
}

/// One request line after parsing: anime title + optional episode filter.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DownloadRequest {
    /// Original title text used for ogladajanime discovery.
    title: String,
    /// `None` = every episode in the resolver plan.
    episodes: Option<Vec<i64>>,
}

/// Build the (extension-less) output stem for an episode:
/// `<out_dir>/<slug>-E{NN}` with the episode number zero-padded to two digits.
/// yt-dlp appends the real container extension during a best-quality download.
fn episode_stem(out_dir: &Path, slug: &str, episode: i64) -> PathBuf {
    out_dir.join(format!("{slug}-E{episode:02}"))
}

/// If episode `stem` was already downloaded, return its path. Matches any
/// non-empty file sharing the stem's file name (the extension is unknown until
/// yt-dlp picks it), so a resumed run skips episodes already on disk.
fn existing_download(stem: &Path) -> Option<PathBuf> {
    let parent = stem.parent()?;
    let name = stem.file_name()?;
    for entry in std::fs::read_dir(parent).ok()?.flatten() {
        let path = entry.path();
        if path.file_stem() == Some(name)
            && path.is_file()
            && std::fs::metadata(&path)
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            return Some(path);
        }
    }
    None
}

/// Parse one list-file line into a title + optional episode number.
///
/// Accepts trailing selectors: bare `12`, `E12`/`e12`, `EP12`, or `S01E12`
/// (season number ignored — ogladajanime plans are flat episode lists).
fn parse_list_line(line: &str) -> Result<Option<(String, Option<i64>)>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(None);
    }

    // Split off a trailing episode token if present.
    let mut parts = trimmed.rsplitn(2, char::is_whitespace);
    let last = parts.next().unwrap_or("");
    let head = parts.next();

    if let Some(title) = head {
        let title = title.trim();
        if !title.is_empty()
            && let Some(ep) = parse_episode_token(last)
        {
            return Ok(Some((title.to_string(), Some(ep))));
        }
    }

    // Whole line is the title (no episode filter).
    Ok(Some((trimmed.to_string(), None)))
}

/// Parse `12`, `E12`, `EP12`, `S01E12` → episode number. `None` if not a token.
fn parse_episode_token(token: &str) -> Option<i64> {
    let t = token.trim();
    if t.is_empty() {
        return None;
    }
    // Bare integer.
    if let Ok(n) = t.parse::<i64>() {
        return (n > 0).then_some(n);
    }
    let upper = t.to_ascii_uppercase();
    // Strip optional leading season tag: S01E12 → E12, S2EP3 → EP3.
    let body = if let Some(rest) = upper.strip_prefix('S') {
        let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
        if digits == 0 {
            return None;
        }
        &rest[digits..]
    } else {
        upper.as_str()
    };
    // E12 / EP12
    let ep_part = if let Some(rest) = body.strip_prefix("EP") {
        rest
    } else {
        body.strip_prefix('E')?
    };
    if ep_part.is_empty() || !ep_part.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: i64 = ep_part.parse().ok()?;
    (n > 0).then_some(n)
}

/// Read `--file` into grouped download requests (title order = first appearance).
fn load_requests_from_file(path: &Path) -> Result<Vec<DownloadRequest>> {
    let text =
        read_to_string(path).map_err(|e| anyhow!("reading list file {}: {e}", path.display()))?;

    // Preserve first-seen title order; merge episode filters per title.
    let mut order = Vec::new();
    // title -> (all_episodes?, specific set)
    let mut acc: HashMap<String, (bool, Vec<i64>)> = HashMap::new();

    for (lineno, raw) in text.lines().enumerate() {
        let parsed =
            parse_list_line(raw).map_err(|e| anyhow!("{}:{}: {e}", path.display(), lineno + 1))?;
        let Some((title, ep)) = parsed else {
            continue;
        };
        let entry = acc.entry(title.clone()).or_insert_with(|| {
            order.push(title.clone());
            (false, Vec::new())
        });
        match ep {
            None => entry.0 = true, // whole series
            Some(n) => {
                if !entry.1.contains(&n) {
                    entry.1.push(n);
                }
            }
        }
    }

    if order.is_empty() {
        bail!("list file {} has no anime entries", path.display());
    }

    let mut out = Vec::with_capacity(order.len());
    for title in order {
        let (all, mut eps) = acc.remove(&title).unwrap_or((false, Vec::new()));
        eps.sort_unstable();
        let episodes = if all || eps.is_empty() {
            None
        } else {
            Some(eps)
        };
        out.push(DownloadRequest { title, episodes });
    }
    Ok(out)
}

/// Build request list from CLI args (single name and/or `--file`).
fn collect_requests(args: &Args) -> Result<Vec<DownloadRequest>> {
    if let Some(file) = &args.file {
        let mut reqs = load_requests_from_file(file)?;
        // Optional positional name prepended as whole-series job.
        if let Some(name) = &args.name {
            let name = name.trim();
            if !name.is_empty() {
                reqs.insert(
                    0,
                    DownloadRequest {
                        title: name.to_string(),
                        episodes: None,
                    },
                );
            }
        }
        return Ok(reqs);
    }

    let only = args.episodes.as_ref().map(|eps| {
        let mut v = eps.clone();
        v.sort_unstable();
        v.dedup();
        v
    });
    if only
        .as_ref()
        .is_some_and(|v| v.is_empty() || v.iter().any(|&n| n <= 0))
    {
        bail!("--episodes expects positive episode numbers, e.g. --episodes 20,21");
    }

    if args.json.is_some() {
        // Name optional with --json; slug comes from JSON. Title is informational.
        return Ok(vec![DownloadRequest {
            title: args.name.clone().unwrap_or_else(|| "from-json".to_string()),
            episodes: only,
        }]);
    }

    let name = args
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("provide an anime name, `--file <path>`, or `--json <path>`"))?;
    Ok(vec![DownloadRequest {
        title: name.to_string(),
        episodes: only,
    }])
}

/// Resolve the season plan: either parse an existing `--json`, or open the
/// browser at the discovered anime page and wait for the userscript's JSON.
fn resolve_plan(args: &Args, title: &str, allow_json: bool) -> Result<HardsubPlan> {
    if allow_json && let Some(json) = &args.json {
        return ogladajanime::parse_plan(json, "")
            .map_err(|e| anyhow!("parsing resolver JSON {}: {e}", json.display()));
    }

    let downloads_dir = args
        .downloads_dir
        .clone()
        .unwrap_or_else(ogladajanime::default_downloads_dir);

    // Stamp the cutoff BEFORE opening the browser so only a fresh JSON counts.
    let since = SystemTime::now();
    let slug = match ogladajanime::discover(title) {
        Discovery::Found { slug, url } => {
            println!("Found: {url}");
            ogladajanime::open_in_browser(&url).map_err(|e| anyhow!("opening browser: {e}"))?;
            println!(
                "Browser opened. Run the resolver userscript on the anime page — it downloads the players JSON."
            );
            Some(slug)
        }
        Discovery::Search { url } => {
            println!("Couldn't match a slug directly; opening search: {url}");
            ogladajanime::open_in_browser(&url).map_err(|e| anyhow!("opening browser: {e}"))?;
            println!(
                "Pick the anime, then run the resolver userscript to download the players JSON."
            );
            None
        }
    };

    println!(
        "Waiting up to {}s for the resolver JSON in {} ...",
        args.timeout,
        downloads_dir.display()
    );
    let json = ogladajanime::wait_for_resolver_json(
        slug.as_deref(),
        since,
        &downloads_dir,
        Duration::from_secs(args.timeout),
        POLL,
    )
    .map_err(|e| anyhow!("{e}"))?;
    println!("Got resolver JSON: {}", json.display());

    ogladajanime::parse_plan(&json, slug.as_deref().unwrap_or("")).map_err(|e| anyhow!("{e}"))
}

/// Tracks per-host download throughput (bytes/sec) so faster hosts float to
/// the top of the mirror list automatically. Shared across episode threads.
struct HostRanker {
    stats: Mutex<HashMap<String, f64>>, // host → bytes/sec (exponential moving avg)
}

impl HostRanker {
    fn new() -> Self {
        Self {
            stats: Mutex::new(HashMap::new()),
        }
    }

    /// Record a completed download: `bytes` in `elapsed` seconds.
    fn record(&self, host: &str, bytes: u64, elapsed_secs: f64) {
        if elapsed_secs <= 0.0 || bytes == 0 {
            return;
        }
        let bps = bytes as f64 / elapsed_secs;
        let mut map = self.stats.lock().unwrap();
        let entry = map.entry(host.to_string()).or_insert(bps);
        // Exponential moving average — recent runs weigh 30%
        *entry = *entry * 0.7 + bps * 0.3;
    }

    /// Estimated bytes/sec for `host`, or `None` if never measured.
    fn speed(&self, host: &str) -> Option<f64> {
        self.stats.lock().unwrap().get(host).copied()
    }

    /// Reorder `players` so fastest-known hosts come first.
    /// Ties (no data yet) preserve the original `pl_players` order from the
    /// JSON — host preference ranking.
    fn sort(&self, players: &mut [(String, String)]) {
        let mut indexed: Vec<_> = players.iter().enumerate().collect();
        indexed.sort_by(|(ia, a), (ib, b)| {
            let sa = self.speed(&a.0).unwrap_or(-1.0);
            let sb = self.speed(&b.0).unwrap_or(-1.0);
            // Descending speed, then original position for ties
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ia.cmp(ib))
        });
        let sorted: Vec<_> = indexed.into_iter().map(|(_, v)| v.clone()).collect();
        players.clone_from_slice(&sorted);
    }
}

/// Per-host mutex: only one download per host at a time.
struct HostLocks {
    map: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl HostLocks {
    fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }

    /// Run `f` while holding the per-host mutex for `host` (blocking).
    fn with_host<F, T>(&self, host: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let m = {
            let mut map = self.map.lock().unwrap();
            map.entry(host.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _guard = m.lock().unwrap();
        f()
    }

    /// Try to run `f` under the per-host mutex. Returns `None` if another
    /// thread already holds it — caller should try the next mirror.
    fn try_with_host<F, T>(&self, host: &str, f: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        let m = {
            let mut map = self.map.lock().unwrap();
            map.entry(host.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let guard = m.try_lock().ok()?;
        let result = f();
        drop(guard);
        Some(result)
    }
}

/// Spawn a short-lived yt-dlp, measure its speed for up to 6 seconds,
/// then kill it. Returns average bytes/sec, or `None` if it never starts.
fn measure_mirror_bps(embed_url: &str) -> Option<f64> {
    let mut cmd = Command::new("yt-dlp");
    cmd.arg("--progress")
        .arg("--newline")
        .arg("-f")
        .arg("bv*+ba/b")
        .arg("-o")
        .arg("/dev/null")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .arg(embed_url);

    let mut child = cmd.spawn().ok()?;
    let stdout = child.stdout.take()?;
    let reader = BufReader::new(stdout);

    let deadline = Instant::now() + Duration::from_secs(6);
    let mut speeds: Vec<f64> = Vec::new();

    for line in reader.lines() {
        if Instant::now() >= deadline || speeds.len() >= 4 {
            break;
        }
        let line = line.ok()?;
        if let Some(idx) = line.find(" at ") {
            let after = &line[idx + 4..];
            let speed_str = after.split_whitespace().next()?;
            if let Some(bps) = parse_speed_bps(speed_str) {
                speeds.push(bps);
            }
        }
    }

    child.kill().ok();
    let _ = child.wait();

    if speeds.is_empty() {
        return None;
    }
    Some(speeds.iter().sum::<f64>() / speeds.len() as f64)
}

/// Parse a yt-dlp speed string like "7.50MiB/s" to bytes/sec.
fn parse_speed_bps(s: &str) -> Option<f64> {
    let s = s.trim();
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());
    let num: f64 = s[..num_end].parse().ok()?;
    let unit = &s[num_end..];
    Some(match unit {
        "KiB/s" => num * 1024.0,
        "MiB/s" => num * 1_048_576.0,
        "GiB/s" => num * 1_073_741_824.0,
        _ => num,
    })
}

/// Download one episode via two-phase race:
/// 1. Measure every available mirror for ~6 s, pick fastest bytes/sec.
/// 2. Full download from the fastest, with progress bar. Falls back through
///    slower mirrors if the fastest fails.
fn download_episode(
    plan: &HardsubPlan,
    episode: i64,
    stem: &Path,
    locks: &HostLocks,
    ranker: &HostRanker,
    tx: &Sender<EpEvent>,
) -> Option<PathBuf> {
    use std::collections::HashSet;

    let mut players: Vec<_> = plan
        .pl_players(episode)
        .into_iter()
        .map(|p| {
            (
                p.host.as_deref().unwrap_or("?").to_string(),
                p.embed_url.clone(),
            )
        })
        .collect();

    ranker.sort(&mut players);

    // ── Phase 1: measure speed of every available mirror ──
    let _ = tx.send(EpEvent::Measuring {
        ep: episode,
        host: "".into(),
    }); // signal start

    let mut measured: Vec<(String, String, f64)> = Vec::new();

    for (host, embed_url) in &players {
        let _ = tx.send(EpEvent::Measuring {
            ep: episode,
            host: host.clone(),
        });
        match locks.try_with_host(host, || ()) {
            Some(()) => {}
            None => {
                let _ = tx.send(EpEvent::MirrorBusy {
                    ep: episode,
                    host: host.clone(),
                });
                continue;
            }
        }
        if let Some(bps) = measure_mirror_bps(embed_url) {
            let _ = tx.send(EpEvent::Measured {
                ep: episode,
                host: host.clone(),
                bps,
            });
            measured.push((host.clone(), embed_url.clone(), bps));
        } else {
            let _ = tx.send(EpEvent::MirrorDone {
                ep: episode,
                host: host.clone(),
                success: false,
            });
        }
    }

    measured.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // ── Phase 2: download from fastest, fall back down the list ──
    for (host, embed_url, _bps) in &measured {
        let _ = tx.send(EpEvent::Winner {
            ep: episode,
            host: host.clone(),
        });

        let host = host.clone();
        let embed_url = embed_url.clone();
        let started = Instant::now();

        let result = locks.with_host(&host, || {
            mt_ml::hardsub_download_with_progress(&embed_url, stem, true, None, |p| {
                let _ = tx.send(EpEvent::Progress {
                    ep: episode,
                    host: host.clone(),
                    pct: p.percent,
                    speed: p.speed.clone().unwrap_or_default(),
                    eta: p.eta.clone().unwrap_or_default(),
                });
            })
        });

        match result {
            Ok(path) => {
                let elapsed = started.elapsed().as_secs_f64();
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                ranker.record(&host, bytes, elapsed);
                let size_mb = bytes as f64 / 1_048_576.0;
                let _ = tx.send(EpEvent::Done {
                    ep: episode,
                    host: host.clone(),
                    size_mb,
                });
                return Some(path);
            }
            Err(_e) => {
                let _ = tx.send(EpEvent::MirrorDone {
                    ep: episode,
                    host: host.clone(),
                    success: false,
                });
            }
        }
    }

    // ── Fallback: try remaining unmeasured mirrors ──
    let measured_hosts: HashSet<&str> = measured.iter().map(|(h, _, _)| h.as_str()).collect();
    for (host, embed_url) in &players {
        if measured_hosts.contains(host.as_str()) {
            continue;
        }
        let _ = tx.send(EpEvent::Winner {
            ep: episode,
            host: host.clone(),
        });
        let host = host.clone();
        let embed_url = embed_url.clone();
        let started = Instant::now();
        let result = locks.with_host(&host, || {
            mt_ml::hardsub_download_with_progress(&embed_url, stem, true, None, |p| {
                let _ = tx.send(EpEvent::Progress {
                    ep: episode,
                    host: host.clone(),
                    pct: p.percent,
                    speed: p.speed.clone().unwrap_or_default(),
                    eta: p.eta.clone().unwrap_or_default(),
                });
            })
        });
        match result {
            Ok(path) => {
                let elapsed = started.elapsed().as_secs_f64();
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                ranker.record(&host, bytes, elapsed);
                let size_mb = bytes as f64 / 1_048_576.0;
                let _ = tx.send(EpEvent::Done {
                    ep: episode,
                    host: host.clone(),
                    size_mb,
                });
                return Some(path);
            }
            Err(_e) => {}
        }
    }

    let _ = tx.send(EpEvent::Failed { ep: episode });
    None
}

/// Download selected episodes from an already-resolved plan.
/// Episodes run in parallel; per-host mutex keeps same-source downloads serial.
/// Uses ratatui TUI for multi-panel progress display.
fn download_from_plan(
    plan: &HardsubPlan,
    out_dir: &Path,
    only: Option<&[i64]>,
) -> Result<(u32, u32, u32)> {
    use std::sync::mpsc;

    std::fs::create_dir_all(out_dir)
        .map_err(|e| anyhow!("creating output dir {}: {e}", out_dir.display()))?;

    let mut episodes: Vec<i64> = match only {
        Some(filter) => {
            let mut missing = Vec::new();
            let mut present = Vec::new();
            for &ep in filter {
                if plan.episodes.contains_key(&ep) {
                    present.push(ep);
                } else {
                    missing.push(ep);
                }
            }
            if !missing.is_empty() {
                warn!(
                    "{}: episode(s) not in resolver JSON (skipped): {:?}",
                    plan.slug, missing
                );
            }
            present
        }
        None => {
            let mut eps: Vec<i64> = plan.episodes.keys().copied().collect();
            eps.sort_unstable();
            eps
        }
    };

    if episodes.is_empty() {
        bail!(
            "resolver JSON for '{}' listed no matching episodes with PL players",
            plan.slug
        );
    }
    episodes.sort_unstable();

    let (tx, rx) = mpsc::channel::<EpEvent>();
    let locks = Arc::new(HostLocks::new());
    let ranker = Arc::new(HostRanker::new());
    let plan = Arc::new(plan.clone());

    let mut handles = Vec::new();
    let mut skipped = 0u32;

    for &episode in &episodes {
        let stem = episode_stem(out_dir, &plan.slug, episode);
        if let Some(path) = existing_download(&stem) {
            let size_mb = std::fs::metadata(&path)
                .map(|m| m.len() as f64 / 1_048_576.0)
                .unwrap_or(0.0);
            let _ = tx.send(EpEvent::Done {
                ep: episode,
                host: format!("cached — {}", path.display()),
                size_mb,
            });
            skipped += 1;
            continue;
        }
        let tx = tx.clone();
        let plan = Arc::clone(&plan);
        let locks = Arc::clone(&locks);
        let ranker = Arc::clone(&ranker);
        handles.push(std::thread::spawn(move || {
            let result = download_episode(&plan, episode, &stem, &locks, &ranker, &tx);
            (episode, result)
        }));
    }

    // Run TUI — blocks until all episodes signal Done/Failed
    let episodes_list: Vec<i64> = episodes.clone();
    let ui = DownloadUi::new(rx, &episodes_list).map_err(|e| anyhow!("starting TUI: {e}"))?;
    ui.run().map_err(|e| anyhow!("TUI error: {e}"))?;

    // Collect results
    let (mut downloaded, mut failed) = (0u32, 0u32);
    for h in handles {
        let (episode, result) = h.join().map_err(|_| anyhow!("thread panicked"))?;
        match result {
            Some(_path) => {
                downloaded += 1;
            }
            None => {
                error!("ep {episode}: all mirrors failed");
                failed += 1;
            }
        }
    }

    println!(
        "Done {}: {downloaded} downloaded, {skipped} skipped, {failed} failed.",
        plan.slug
    );
    Ok((downloaded, skipped, failed))
}

fn run(args: Args) -> Result<i32> {
    let requests = collect_requests(&args)?;
    let multi = requests.len() > 1 || args.file.is_some();
    if multi && args.json.is_some() {
        warn!("--json is only applied to a single anime run; ignoring with multi/list mode");
    }

    let (mut downloaded, mut skipped, mut failed, mut errored) = (0u32, 0u32, 0u32, 0u32);

    for (idx, req) in requests.iter().enumerate() {
        if multi {
            println!(
                "\n=== [{}/{}] {}{} ===",
                idx + 1,
                requests.len(),
                req.title,
                match &req.episodes {
                    None => " (all episodes)".to_string(),
                    Some(eps) => format!(" (episodes {eps:?})"),
                }
            );
        }

        let allow_json = !multi && args.json.is_some();
        let plan = match resolve_plan(&args, &req.title, allow_json) {
            Ok(p) => p,
            Err(e) => {
                error!("failed to resolve '{}': {e:#}", req.title);
                errored += 1;
                continue;
            }
        };

        // With --file and no --out: each anime gets `./<slug>`.
        // With --out: single shared root; still nest by slug when multi.
        let out_dir = match &args.out {
            Some(root) if multi => root.join(&plan.slug),
            Some(root) => root.clone(),
            None => PathBuf::from(&plan.slug),
        };

        match download_from_plan(&plan, &out_dir, req.episodes.as_deref()) {
            Ok((d, s, f)) => {
                downloaded += d;
                skipped += s;
                failed += f;
            }
            Err(e) => {
                error!("'{}': {e:#}", req.title);
                errored += 1;
            }
        }
    }

    if multi {
        println!(
            "\nAll done: {downloaded} downloaded, {skipped} skipped, {failed} failed, {errored} resolve errors."
        );
    }

    Ok(if failed > 0 || errored > 0 { 1 } else { 0 })
}

fn main() {
    let args = Args::parse();
    mt_cli::init_tracing(args.verbose);

    let code = match run(args) {
        Ok(code) => code,
        Err(e) => {
            error!("Error: {e:#}");
            1
        }
    };
    exit(code);
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn episode_stem_zero_pads_to_two_digits() {
        let stem = episode_stem(Path::new("/tmp/out"), "isekai-ojisan", 3);
        assert_eq!(stem, PathBuf::from("/tmp/out/isekai-ojisan-E03"));
    }

    #[test]
    fn episode_stem_keeps_wide_numbers() {
        let stem = episode_stem(Path::new("/tmp/out"), "show", 124);
        assert_eq!(stem, PathBuf::from("/tmp/out/show-E124"));
    }

    #[test]
    fn existing_download_finds_any_extension() {
        let dir = tempdir().unwrap();
        let stem = dir.path().join("show-E01");
        let mkv = dir.path().join("show-E01.mkv");
        std::fs::write(&mkv, b"data").unwrap();
        assert_eq!(existing_download(&stem), Some(mkv));
    }

    #[test]
    fn existing_download_none_when_missing() {
        let dir = tempdir().unwrap();
        let stem = dir.path().join("show-E99");
        assert_eq!(existing_download(&stem), None);
    }

    #[test]
    fn existing_download_ignores_empty_file() {
        let dir = tempdir().unwrap();
        let stem = dir.path().join("show-E02");
        std::fs::write(dir.path().join("show-E02.part"), b"").unwrap();
        assert_eq!(existing_download(&stem), None);
    }

    #[test]
    fn parse_episode_token_variants() {
        assert_eq!(parse_episode_token("12"), Some(12));
        assert_eq!(parse_episode_token("E03"), Some(3));
        assert_eq!(parse_episode_token("e7"), Some(7));
        assert_eq!(parse_episode_token("EP10"), Some(10));
        assert_eq!(parse_episode_token("S01E05"), Some(5));
        assert_eq!(parse_episode_token("s2e11"), Some(11));
        assert_eq!(parse_episode_token("0"), None);
        assert_eq!(parse_episode_token("THE"), None);
        assert_eq!(parse_episode_token("Piece"), None);
    }

    #[test]
    fn parse_list_line_title_only_and_with_ep() {
        assert_eq!(
            parse_list_line("One Piece").unwrap(),
            Some(("One Piece".into(), None))
        );
        assert_eq!(
            parse_list_line("  Naruto E02  ").unwrap(),
            Some(("Naruto".into(), Some(2)))
        );
        assert_eq!(
            parse_list_line("Bleach S01E03").unwrap(),
            Some(("Bleach".into(), Some(3)))
        );
        assert_eq!(
            parse_list_line("Attack on Titan 12").unwrap(),
            Some(("Attack on Titan".into(), Some(12)))
        );
        assert_eq!(parse_list_line("# comment").unwrap(), None);
        assert_eq!(parse_list_line("   ").unwrap(), None);
    }

    #[test]
    fn load_requests_groups_and_merges() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("list.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            "\
# my watch list
One Piece
Naruto 1
Naruto E02
Naruto S01E01
Bleach
One Piece 5
"
        )
        .unwrap();

        let reqs = load_requests_from_file(&path).unwrap();
        assert_eq!(reqs.len(), 3);

        // One Piece: whole series requested → episodes None (5 ignored as subset)
        assert_eq!(reqs[0].title, "One Piece");
        assert_eq!(reqs[0].episodes, None);

        // Naruto: eps 1,2 only (duplicate 1 collapsed)
        assert_eq!(reqs[1].title, "Naruto");
        assert_eq!(reqs[1].episodes, Some(vec![1, 2]));

        assert_eq!(reqs[2].title, "Bleach");
        assert_eq!(reqs[2].episodes, None);
    }

    #[test]
    fn load_requests_empty_file_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, "# only comments\n\n").unwrap();
        assert!(load_requests_from_file(&path).is_err());
    }
}
