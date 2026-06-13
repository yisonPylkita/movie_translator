//! ogladajanime.pl hardsub discovery + assisted pickup.
//!
//! The site can't be driven headlessly (Cloudflare Turnstile + anti-debug), so
//! the actual player-URL resolution happens in the user's real browser via a
//! Tampermonkey userscript that downloads a JSON. This module does the parts
//! the app *can* automate: discover the anime page, open the browser there, and
//! watch `~/Downloads` for the userscript's JSON — guarded so a download still
//! in flight is never read.
//!
//! Pure helpers (`slugify`, `parse_plan`, `best_pl_player`) are unit-tested
//! without network or filesystem.

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

/// Host preference for picking the best player (CDA resolves cleanest in
/// yt-dlp). Lower index = more preferred; unknown hosts sort last.
const HOST_PREFERENCE: &[&str] = &[
    "cda",
    "sibnet",
    "vk",
    "mega",
    "ok",
    "dood",
    "myvi",
    "google",
    "hqq",
    "voe",
    "mp4upload",
];

/// One resolved player from the userscript JSON (`resolved[]` entry).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ResolvedPlayer {
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub sub: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    pub embed_url: String,
    #[serde(default)]
    pub sub_group: Option<String>,
}

/// Parsed resolver JSON: episode number -> its resolved players.
#[derive(Debug, Clone)]
pub struct HardsubPlan {
    pub slug: String,
    pub episodes: HashMap<i64, Vec<ResolvedPlayer>>,
}

impl HardsubPlan {
    /// The best PL-sub player for `episode` (CDA preferred, then resolution).
    pub fn best_player(&self, episode: i64) -> Option<&ResolvedPlayer> {
        best_pl_player(self.episodes.get(&episode)?)
    }

    /// All PL-sub players for `episode`, ordered best-first (host preference,
    /// then resolution). Used to fall back to the next mirror when a download
    /// fails — a dead/410 cda link then drops to vk/mega/etc. automatically.
    pub fn pl_players(&self, episode: i64) -> Vec<&ResolvedPlayer> {
        let Some(players) = self.episodes.get(&episode) else {
            return Vec::new();
        };
        let mut v: Vec<&ResolvedPlayer> = players
            .iter()
            .filter(|p| p.sub.as_deref() == Some("pl") && !p.embed_url.is_empty())
            .collect();
        v.sort_by_key(|p| (host_rank(p), -quality_height(p)));
        v
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

// --- JSON shapes (userscript output) -------------------------------------

#[derive(Deserialize)]
struct ResolverJson {
    #[serde(default)]
    anime_slug: Option<String>,
    #[serde(default)]
    episodes: Vec<EpisodeEntry>,
}

#[derive(Deserialize)]
struct EpisodeEntry {
    #[serde(default)]
    episode: Option<i64>,
    #[serde(default)]
    resolved: Vec<ResolvedPlayer>,
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
    // The userscript downloads `oga-<slug>-*.players.json`. When the browser
    // hits a duplicate name it inserts ` (N)` BEFORE `.json` (e.g.
    // `oga-...-all.players (4).json`), so we match on `.players` + `.json`
    // rather than a literal `.players.json` suffix. In-flight partials end in
    // `.crdownload`/`.part`/`.download` (not `.json`), so requiring a `.json`
    // tail already excludes them.
    if !(name.starts_with("oga-") && name.contains(".players") && name.ends_with(".json")) {
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
/// `oga-<slug>-*.players.json` name (browsers rename atomically on completion),
/// requires the file's mtime to be at/after `since`, and accepts it only once
/// its size is stable across two polls and it parses as a valid resolver JSON.
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
                "no resolver JSON (oga-*.players.json) appeared in {} within {}s",
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

/// Parse a resolver JSON file into a [`HardsubPlan`]. `fallback_slug` is used
/// when the JSON omits `anime_slug`.
pub fn parse_plan(path: &Path, fallback_slug: &str) -> Result<HardsubPlan, FetchError> {
    let text = fs::read_to_string(path)?;
    let json: ResolverJson = from_str(&text).map_err(|e| FetchError::Parse(e.to_string()))?;
    let slug = json
        .anime_slug
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback_slug.to_string());
    let mut episodes = HashMap::new();
    for ep in json.episodes {
        if let Some(n) = ep.episode
            && !ep.resolved.is_empty()
        {
            episodes.insert(n, ep.resolved);
        }
    }
    Ok(HardsubPlan { slug, episodes })
}

fn host_rank(player: &ResolvedPlayer) -> usize {
    player
        .host
        .as_deref()
        .and_then(|h| HOST_PREFERENCE.iter().position(|x| *x == h))
        .unwrap_or(HOST_PREFERENCE.len())
}

fn quality_height(player: &ResolvedPlayer) -> i32 {
    player
        .quality
        .as_deref()
        .map(|q| q.chars().filter(|c| c.is_ascii_digit()).collect::<String>())
        .and_then(|d| d.parse::<i32>().ok())
        .unwrap_or(0)
}

/// Pick the best PL-sub player: host preference first, then highest resolution.
pub fn best_pl_player(players: &[ResolvedPlayer]) -> Option<&ResolvedPlayer> {
    players
        .iter()
        .filter(|p| p.sub.as_deref() == Some("pl") && !p.embed_url.is_empty())
        .min_by_key(|p| (host_rank(p), -quality_height(p)))
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
        assert!(is_resolver_json("oga-isekai-ojisan-all.players.json", None));
        assert!(is_resolver_json(
            "oga-isekai-ojisan-all.players.json",
            Some("isekai-ojisan")
        ));
        // browser duplicate-name suffix: ` (N)` is inserted before `.json`
        assert!(is_resolver_json(
            "oga-isekai-ojisan-all.players (4).json",
            Some("isekai-ojisan")
        ));
        // in-flight partials never match the final suffix
        assert!(!is_resolver_json(
            "oga-isekai-ojisan-all.players.json.crdownload",
            None
        ));
        assert!(!is_resolver_json("oga-x.players.json.part", None));
        // slug filter
        assert!(!is_resolver_json(
            "oga-other-anime-all.players.json",
            Some("isekai-ojisan")
        ));
        // unrelated files
        assert!(!is_resolver_json("something.json", None));
    }

    #[test]
    fn best_player_prefers_cda_pl() {
        let players = vec![
            ResolvedPlayer {
                host: Some("sibnet".into()),
                sub: Some("pl".into()),
                quality: Some("720p".into()),
                embed_url: "https://sibnet/x".into(),
                sub_group: None,
            },
            ResolvedPlayer {
                host: Some("cda".into()),
                sub: Some("pl".into()),
                quality: Some("1080p".into()),
                embed_url: "https://cda/x".into(),
                sub_group: None,
            },
            ResolvedPlayer {
                host: Some("cda".into()),
                sub: Some("en".into()),
                quality: Some("1080p".into()),
                embed_url: "https://cda/en".into(),
                sub_group: None,
            },
        ];
        let best = best_pl_player(&players).unwrap();
        assert_eq!(best.host.as_deref(), Some("cda"));
        assert_eq!(best.sub.as_deref(), Some("pl"));
    }

    #[test]
    fn pl_players_ordered_best_first_for_fallback() {
        let mut episodes = HashMap::new();
        episodes.insert(
            4,
            vec![
                ResolvedPlayer {
                    host: Some("vk".into()),
                    sub: Some("pl".into()),
                    quality: Some("1080p".into()),
                    embed_url: "https://vk/4".into(),
                    sub_group: Some("Mioro-Subs".into()),
                },
                ResolvedPlayer {
                    host: Some("cda".into()),
                    sub: Some("pl".into()),
                    quality: Some("1080p".into()),
                    embed_url: "https://cda/4".into(),
                    sub_group: None,
                },
                ResolvedPlayer {
                    host: Some("mp4upload".into()),
                    sub: Some("en".into()),
                    quality: Some("720p".into()),
                    embed_url: "https://mp4/4".into(),
                    sub_group: None,
                },
            ],
        );
        let plan = HardsubPlan {
            slug: "x".into(),
            episodes,
        };
        let order: Vec<_> = plan
            .pl_players(4)
            .iter()
            .map(|p| p.host.clone().unwrap())
            .collect();
        // cda first (fallback target = vk next); en player excluded.
        assert_eq!(order, vec!["cda", "vk"]);
        assert!(plan.pl_players(99).is_empty());
    }

    #[test]
    fn best_player_none_when_no_pl() {
        let players = vec![ResolvedPlayer {
            host: Some("cda".into()),
            sub: Some("en".into()),
            quality: Some("1080p".into()),
            embed_url: "https://cda/en".into(),
            sub_group: None,
        }];
        assert!(best_pl_player(&players).is_none());
    }

    #[test]
    fn parse_plan_from_userscript_json() {
        let json = r#"{
            "anime_slug": "isekai-ojisan",
            "episodes": [
                {"episode": 1, "episode_url": "u1", "resolved": [
                    {"host": "cda", "sub": "pl", "quality": "1080p", "embed_url": "https://cda/1"}
                ]},
                {"episode": 2, "episode_url": "u2", "resolved": []}
            ]
        }"#;
        let dir = env::temp_dir();
        let path = dir.join("oga-test-parse.players.json");
        fs::write(&path, json).unwrap();
        let plan = parse_plan(&path, "fallback").unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(plan.slug, "isekai-ojisan");
        // episode 2 had no resolved players → dropped
        assert_eq!(plan.episode_count(), 1);
        assert_eq!(plan.best_player(1).unwrap().host.as_deref(), Some("cda"));
        assert!(plan.best_player(2).is_none());
    }
}
