//! ogladajanime.pl hardsub discovery + assisted pickup.
//!
//! The site can't be driven headlessly (Cloudflare Turnstile + anti-debug), so
//! the actual player-URL resolution happens in the user's real browser via a
//! Tampermonkey userscript that downloads a canonical JSON. This module does
//! the parts the app *can* automate: discover the anime page, open the browser
//! there, and watch `~/Downloads` for the userscript's JSON — guarded so a
//! download still in flight is never read.
//!
//! Pure helpers (`slugify`, `parse_plan`) are unit-tested without network or
//! filesystem.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime};

use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::from_str;

use crate::retry::FetchError;

const BASE_URL: &str = "https://ogladajanime.pl";
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36";

/// Parsed canonical resolver JSON: episode number -> its resolved URLs.
#[derive(Debug, Clone)]
pub struct HardsubPlan {
    pub slug: String,
    pub episodes: HashMap<i64, Vec<String>>,
}

impl HardsubPlan {
    /// All URLs for `episode`, ordered best-first (userscript already curates).
    pub fn pl_players(&self, episode: i64) -> Vec<&str> {
        self.episodes
            .get(&episode)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }
}

/// Result of trying to locate the anime on ogladajanime.
#[derive(Debug, Clone, PartialEq)]
pub enum Discovery {
    /// Slug verified as a real anime page; open this URL.
    Found { slug: String, url: String },
    /// Slug guess missed; open the search page for the user to pick.
    Search { url: String },
}

// --- Canonical JSON shapes (userscript output) ---------------------------

#[derive(Deserialize)]
struct CanonicalJson {
    #[serde(default)]
    title: Option<String>,
    episodes: Vec<CanonicalEpisode>,
}

#[derive(Deserialize)]
struct CanonicalEpisode {
    episode: i64,
    #[serde(default)]
    urls: Vec<String>,
}

/// Lowercase, ASCII-fold-ish, non-alphanumeric → single dash, trimmed.
///
/// `"Isekai Ojisan"` → `"isekai-ojisan"`. Non-ASCII letters are dropped (rare
/// in the romaji titles ogladajanime uses); the search fallback covers misses.
pub fn slugify(title: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in title.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_dash = false;
        } else if !out.is_empty() && !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Discover the anime page for `title`: verify the slug guess over HTTP, else
/// fall back to the search page URL.
pub fn discover(title: &str) -> Discovery {
    let slug = slugify(title);
    let url = format!("{BASE_URL}/anime/{slug}");
    if slug_is_anime_page(&url) {
        return Discovery::Found { slug, url };
    }
    let query = title.replace(' ', "%20");
    Discovery::Search {
        url: format!("{BASE_URL}/search/name/{query}"),
    }
}

fn slug_is_anime_page(url: &str) -> bool {
    let client = match Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let Ok(resp) = client.get(url).send() else {
        return false;
    };
    if !resp.status().is_success() {
        return false;
    }
    match resp.text() {
        // Markers present on a real episode/anime page.
        Ok(body) => {
            body.contains("Odcinki") || body.contains("ep_list") || body.contains("changePlayer")
        }
        Err(_) => false,
    }
}

/// Open `url` in the user's default browser (`open` on macOS, `xdg-open` else).
pub fn open_in_browser(url: &str) -> Result<(), FetchError> {
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(not(target_os = "macos"))]
    let program = "xdg-open";
    Command::new(program)
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(FetchError::Io)
}

/// The default Downloads directory (`$XDG_DOWNLOAD_DIR`, then `$HOME/Downloads`).
pub fn default_downloads_dir() -> PathBuf {
    if let Ok(dir) = env::var("XDG_DOWNLOAD_DIR")
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(home) = env::var("HOME") {
        return PathBuf::from(home).join("Downloads");
    }
    PathBuf::from("Downloads")
}

fn is_resolver_json(name: &str, slug: Option<&str>) -> bool {
    if !(name.starts_with("anime-") && name.ends_with(".json")) {
        return false;
    }
    // Exclude in-flight partials
    if name.ends_with(".crdownload") || name.ends_with(".part") || name.contains(".download") {
        return false;
    }
    match slug {
        Some(s) if !s.is_empty() => name.contains(s),
        _ => true,
    }
}

/// Watch `downloads_dir` for a finished resolver JSON newer than `since`.
///
/// Guards against reading a download in flight: only matches the final
/// `anime-<slug>.json` name (browsers rename atomically on completion),
/// requires the file's mtime to be at/after `since`, and accepts it only once
/// its size is stable across two polls and it parses as a valid canonical JSON.
pub fn wait_for_resolver_json(
    slug: Option<&str>,
    since: SystemTime,
    downloads_dir: &Path,
    timeout: Duration,
    poll: Duration,
) -> Result<PathBuf, FetchError> {
    let start = Instant::now();
    // Small skew tolerance: accept files modified just before `since`.
    let cutoff = since
        .checked_sub(Duration::from_secs(2))
        .unwrap_or(SystemTime::UNIX_EPOCH);

    loop {
        if let Some(path) = newest_match(downloads_dir, slug, cutoff) {
            let size1 = file_size(&path);
            sleep(poll);
            let size2 = file_size(&path);
            if size1 > 0 && size1 == size2 && parse_plan(&path, "").is_ok() {
                return Ok(path);
            }
        } else {
            sleep(poll);
        }
        if start.elapsed() >= timeout {
            return Err(FetchError::NotFound(format!(
                "no resolver JSON (anime-*.json) appeared in {} within {}s",
                downloads_dir.display(),
                timeout.as_secs()
            )));
        }
    }
}

fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn newest_match(dir: &Path, slug: Option<&str>, cutoff: SystemTime) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut best: Option<(SystemTime, PathBuf)> = None;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !is_resolver_json(&name, slug) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(mtime) = meta.modified() else { continue };
        if mtime < cutoff {
            continue;
        }
        if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
            best = Some((mtime, entry.path()));
        }
    }
    best.map(|(_, p)| p)
}

/// Parse a canonical resolver JSON file into a [`HardsubPlan`]. `fallback_slug`
/// is used when the JSON omits `title`.
pub fn parse_plan(path: &Path, fallback_slug: &str) -> Result<HardsubPlan, FetchError> {
    let text = fs::read_to_string(path)?;
    let json: CanonicalJson = from_str(&text).map_err(|e| FetchError::Parse(e.to_string()))?;
    let slug = json
        .title
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(slugify)
        .unwrap_or_else(|| fallback_slug.to_string());
    let mut episodes = HashMap::new();
    for ep in json.episodes {
        let urls: Vec<String> = ep.urls.into_iter().filter(|u| !u.is_empty()).collect();
        if !urls.is_empty() {
            episodes.insert(ep.episode, urls);
        }
    }
    Ok(HardsubPlan { slug, episodes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("Isekai Ojisan"), "isekai-ojisan");
        assert_eq!(slugify("  Strike Witches  "), "strike-witches");
        assert_eq!(slugify("Re:ZERO -Starting-"), "re-zero-starting");
    }

    #[test]
    fn is_resolver_json_matches_final_only() {
        assert!(is_resolver_json("anime-isekai-ojisan.json", None));
        assert!(is_resolver_json(
            "anime-isekai-ojisan.json",
            Some("isekai-ojisan")
        ));
        // browser duplicate-name suffix: ` (N)` is inserted before `.json`
        assert!(is_resolver_json(
            "anime-isekai-ojisan (4).json",
            Some("isekai-ojisan")
        ));
        // in-flight partials never match the final suffix
        assert!(!is_resolver_json("anime-x.json.crdownload", None));
        assert!(!is_resolver_json("anime-x.json.part", None));
        // slug filter
        assert!(!is_resolver_json("anime-other.json", Some("isekai-ojisan")));
        // unrelated files
        assert!(!is_resolver_json("something.json", None));
    }

    #[test]
    fn pl_players_returns_urls_from_canonical() {
        let mut episodes = HashMap::new();
        episodes.insert(
            4,
            vec![
                "https://cda.pl/4".to_string(),
                "https://vk.com/4".to_string(),
            ],
        );
        let plan = HardsubPlan {
            slug: "x".into(),
            episodes,
        };
        let urls = plan.pl_players(4);
        assert_eq!(urls, vec!["https://cda.pl/4", "https://vk.com/4"]);
        assert!(plan.pl_players(99).is_empty());
    }

    #[test]
    fn parse_plan_from_canonical_json() {
        let json = r#"{
            "title": "Isekai Ojisan",
            "episodes": [
                {"episode": 1, "urls": ["https://cda.pl/v/1", "https://sibnet.ru/2"]},
                {"episode": 2, "urls": []}
            ]
        }"#;
        let dir = env::temp_dir();
        let path = dir.join("anime-test-canonical.json");
        fs::write(&path, json).unwrap();
        let plan = parse_plan(&path, "fallback").unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(plan.slug, "isekai-ojisan"); // slugified from title
        assert_eq!(plan.episode_count(), 1); // ep 2 had empty urls -> dropped
        let urls = plan.pl_players(1);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://cda.pl/v/1");
        assert_eq!(urls[1], "https://sibnet.ru/2");
    }

    #[test]
    fn parse_plan_fallback_slug_when_no_title() {
        let json = r#"{
            "episodes": [
                {"episode": 1, "urls": ["https://cda.pl/v/1"]}
            ]
        }"#;
        let dir = env::temp_dir();
        let path = dir.join("anime-test-no-title.json");
        fs::write(&path, json).unwrap();
        let plan = parse_plan(&path, "fallback-slug").unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(plan.slug, "fallback-slug");
        assert_eq!(plan.episode_count(), 1);
    }

    #[test]
    fn parse_plan_skips_empty_urls() {
        let json = r#"{
            "episodes": [
                {"episode": 1, "urls": ["https://cda.pl/v/1", ""]},
                {"episode": 2, "urls": ["", ""]},
                {"episode": 3, "urls": ["https://vk.com/3"]}
            ]
        }"#;
        let dir = env::temp_dir();
        let path = dir.join("anime-test-empty.json");
        fs::write(&path, json).unwrap();
        let plan = parse_plan(&path, "fallback").unwrap();
        fs::remove_file(&path).ok();
        // ep1: one url kept (non-empty), ep2: all empty -> dropped, ep3: kept
        assert_eq!(plan.episode_count(), 2);
        let urls1 = plan.pl_players(1);
        assert_eq!(urls1.len(), 1);
        assert_eq!(urls1[0], "https://cda.pl/v/1");
        let urls3 = plan.pl_players(3);
        assert_eq!(urls3.len(), 1);
        assert_eq!(urls3[0], "https://vk.com/3");
    }

    /// Verify round-trip: canonical JSON -> parse_plan -> each episode accessible.
    #[test]
    fn canonical_round_trip() {
        let json = r#"{
            "title": "Test Series",
            "episodes": [
                {"episode": 1, "urls": ["https://cda.pl/a", "https://sibnet.ru/b"]},
                {"episode": 5, "urls": ["https://vk.com/c"]},
                {"episode": 10, "urls": []}
            ]
        }"#;
        let dir = env::temp_dir();
        let path = dir.join("anime-roundtrip.json");
        fs::write(&path, json).unwrap();
        let plan = parse_plan(&path, "fallback").unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(plan.slug, "test-series");
        assert_eq!(plan.episode_count(), 2);
        assert_eq!(
            plan.pl_players(1),
            vec!["https://cda.pl/a", "https://sibnet.ru/b"]
        );
        assert_eq!(plan.pl_players(5), vec!["https://vk.com/c"]);
        assert!(plan.pl_players(10).is_empty());
    }
}
