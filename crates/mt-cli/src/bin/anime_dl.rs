//! `anime-dl` — standalone ogladajanime.pl season downloader.
//!
//! Pass an anime name; the tool finds it on ogladajanime.pl, opens the browser,
//! and waits for the resolver userscript to drop its players JSON in Downloads.
//! It then downloads every episode at best available quality. No translation,
//! no OCR — pure download. Reuses the `mt_fetch::ogladajanime` discovery/pickup
//! plumbing and `mt_ml::hardsub_download` (with `best = true`).
//!
//! Downloads are network/CPU only (no GPU worker, no TUI) and run sequentially.
//! The embedded Python interpreter initialises lazily on the first
//! `hardsub_download` call, so there's nothing to set up here.

use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::{Duration, SystemTime};

use anyhow::{Result, anyhow};
use clap::Parser;
use mt_fetch::ogladajanime::{self, Discovery, HardsubPlan};
use tracing::{error, info, warn};

/// How often to poll Downloads for the resolver JSON.
const POLL: Duration = Duration::from_secs(2);

#[derive(Parser, Debug)]
#[command(
    name = "anime-dl",
    about = "Download a whole anime season from ogladajanime.pl (best quality, no translation)"
)]
struct Args {
    /// Anime name to search for on ogladajanime.pl. Optional when `--json` is given.
    name: Option<String>,

    /// Output directory. Default: `./<slug>`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Directory to watch for the resolver userscript JSON.
    /// Default: the system Downloads directory.
    #[arg(long)]
    downloads_dir: Option<PathBuf>,

    /// Seconds to wait for the resolver JSON to appear.
    #[arg(long, default_value_t = 600)]
    timeout: u64,

    /// Parse an existing resolver JSON instead of opening the browser.
    /// Handy for re-runs without re-running the userscript.
    #[arg(long)]
    json: Option<PathBuf>,

    /// Raise our crates' logging to DEBUG.
    #[arg(short, long)]
    verbose: bool,
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

/// Resolve the season plan: either parse an existing `--json`, or open the
/// browser at the discovered anime page and wait for the userscript's JSON.
fn resolve_plan(args: &Args) -> Result<HardsubPlan> {
    if let Some(json) = &args.json {
        return ogladajanime::parse_plan(json, "")
            .map_err(|e| anyhow!("parsing resolver JSON {}: {e}", json.display()));
    }

    let name = args.name.as_deref().ok_or_else(|| {
        anyhow!("provide an anime name, or `--json <path>` to use an existing JSON")
    })?;
    let downloads_dir = args
        .downloads_dir
        .clone()
        .unwrap_or_else(ogladajanime::default_downloads_dir);

    // Stamp the cutoff BEFORE opening the browser so only a fresh JSON counts.
    let since = SystemTime::now();
    let slug = match ogladajanime::discover(name) {
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

/// Download one episode, walking its PL mirrors best-first until one succeeds.
/// Returns the written path, or `None` if every mirror failed.
fn download_episode(plan: &HardsubPlan, episode: i64, stem: &Path) -> Option<PathBuf> {
    for player in plan.pl_players(episode) {
        let host = player.host.as_deref().unwrap_or("?");
        info!(
            "ep {episode}: trying {host} {} ({})",
            player.quality.as_deref().unwrap_or("?"),
            player.embed_url
        );
        // best = true: highest-quality video+audio; min_height ignored.
        match mt_ml::hardsub_download(&player.embed_url, stem, 0, true, None) {
            Ok(path) => return Some(path),
            Err(e) => {
                warn!("ep {episode}: {host} mirror failed ({e}); trying next");
            }
        }
    }
    None
}

fn run(args: Args) -> Result<i32> {
    let plan = resolve_plan(&args)?;

    let out_dir = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(&plan.slug));
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| anyhow!("creating output dir {}: {e}", out_dir.display()))?;

    let mut episodes: Vec<i64> = plan.episodes.keys().copied().collect();
    episodes.sort_unstable();
    if episodes.is_empty() {
        return Err(anyhow!("resolver JSON listed no episodes with PL players"));
    }

    let total = episodes.len();
    println!("{}: {total} episode(s) → {}", plan.slug, out_dir.display());

    let (mut downloaded, mut skipped, mut failed) = (0u32, 0u32, 0u32);
    for (i, episode) in episodes.iter().enumerate() {
        let n = i + 1;
        let stem = episode_stem(&out_dir, &plan.slug, *episode);
        if let Some(path) = existing_download(&stem) {
            println!(
                "[{n}/{total}] ep {episode}: already have {} — skipping",
                path.display()
            );
            skipped += 1;
            continue;
        }
        match download_episode(&plan, *episode, &stem) {
            Some(path) => {
                println!("[{n}/{total}] ep {episode} → {}", path.display());
                downloaded += 1;
            }
            None => {
                error!("[{n}/{total}] ep {episode}: all mirrors failed");
                failed += 1;
            }
        }
    }

    println!("Done: {downloaded} downloaded, {skipped} skipped, {failed} failed.");
    Ok(if failed > 0 { 1 } else { 0 })
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
}
