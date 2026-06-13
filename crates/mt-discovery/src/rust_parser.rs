//! Pure-Rust filename parser — replaces the Python `aniparse`/`guessit` backend.
//!
//! Strategy:
//!  1. Try `anitomy-pure` first.  If it finds anime signals (release_group,
//!     anime title, anime season), mark as anime and use its results.
//!  2. If not anime, use regex-based fallback for conventional TV/movie
//!     show patterns: `S01E01`, `1x01`, year, etc.
//!  3. If the title is still missing and a `folder_name` is provided, try
//!     extracting from the folder name.

use regex::Regex;
use std::sync::LazyLock;

use crate::parser::ParsedName;

// ── Statically compiled regexes ─────────────────────────────────────────────

/// Standard TV episode: S01E01, s01e01, S01E01E02 (multi-episode), S01-EP03
static RE_SEASON_EP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:^|[^a-z0-9])S(\d+)[.\s_-]*E(\d+)").unwrap());

/// Alternative: 1x01 format (common for some Western shows)
static RE_X_SEP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:^|[^a-z0-9])(\d+)x(\d+)").unwrap());

/// Bare episode number: "Episode 01", "EP01", ".01." — but NOT a 4-digit year
static RE_EPISODE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:^|[^a-z0-9])(?:EP|Episode|E)[.\s_-]*(\d+)").unwrap());

/// Year: 4 digits in 1900-2099 range (no lookaround — verify boundaries in code)
static RE_YEAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(19\d{2}|20[0-2]\d)").unwrap());

/// Multi-word title from dotted/underscored names: `Breaking.Bad.S01E03` → `Breaking Bad`
static RE_DOT_TITLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(.+?)[._\s-]+(?:S\d{2}E\d+|Season\s*\d+|E\d+|Episode\s*\d+|\d{4}|\d+x\d+|1080p|720p|BluRay|WEB-DL|HDTV|x264|x265|h\.?264|h\.?265|mkv|mp4|avi|m4v)").unwrap()
});

/// Anime bracket pattern: `[Group] Title` but captured cleanly
/// Not used for extraction (anitomy handles this) — used for fallback detection.
static RE_ANIME_BRACKET: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[([^\]]+)\]\s*(.+)").unwrap());

// ── Public entry point ─────────────────────────────────────────────────────

/// Parse a video filename into structured metadata, using pure Rust logic.
///
/// First tries `anitomy-pure` for anime-style filenames.  Falls back to
/// regex-based extraction for conventional TV/movie filenames.
pub fn parse_filename(filename: &str, folder_name: Option<&str>) -> ParsedName {
    // 1. Try anitomy-pure
    let anitomy_result = parse_with_anitomy(filename);

    // 2. Try regex fallback
    let regex_result = parse_with_regex(filename);

    // 3. Merge results — prefer anitomy for anime, regex for non-anime
    let is_anime = anitomy_result.is_anime || RE_ANIME_BRACKET.is_match(filename);

    let (title, year, season, episode, release_group) = if is_anime {
        // Prefer anitomy results
        (
            anitomy_result.title.or(regex_result.title),
            anitomy_result.year.or(regex_result.year),
            anitomy_result.season.or(regex_result.season),
            anitomy_result.episode.or(regex_result.episode),
            anitomy_result.release_group.or(regex_result.release_group),
        )
    } else {
        // Prefer regex results.  Filter out anitomy episode numbers that
        // look like years (e.g. anitomy extracts "2001" as an episode number
        // from "Spirited.Away.2001.1080p.BluRay.mkv").
        let ani_episode = anitomy_result.episode.filter(|&e| !is_year_like(e));
        (
            regex_result.title.or(anitomy_result.title),
            regex_result.year.or(anitomy_result.year),
            regex_result.season.or(anitomy_result.season),
            regex_result.episode.or(ani_episode),
            regex_result.release_group.or(anitomy_result.release_group),
        )
    };

    let title = title.or_else(|| {
        // If still no title, try folder
        folder_name.and_then(extract_title_from_folder_simple)
    });

    // Determine media type
    let media_type = if episode.is_some() || season.is_some() {
        "episode".to_string()
    } else {
        "movie".to_string()
    };

    ParsedName {
        title,
        year,
        season,
        episode,
        media_type,
        is_anime,
        release_group,
    }
}

// ── anitomy-pure parsing ───────────────────────────────────────────────────

fn parse_with_anitomy(filename: &str) -> ParsedName {
    use anitomy_pure::Parser;
    use anitomy_pure::elements::Category;

    let result = match Parser::new(filename).parse() {
        Ok(r) => r,
        Err(_) => {
            return ParsedName {
                title: None,
                year: None,
                season: None,
                episode: None,
                media_type: "movie".to_string(),
                is_anime: false,
                release_group: None,
            };
        }
    };

    let title = result
        .find(Category::AnimeTitle)
        .map(|e| e.value.to_string());

    let year = result
        .find(Category::AnimeYear)
        .and_then(|e| e.value.parse::<i32>().ok());

    let season = result
        .find(Category::AnimeSeason)
        .and_then(|e| e.value.parse::<i32>().ok());

    let episode = result
        .find(Category::EpisodeNumber)
        .and_then(|e| e.value.parse::<i32>().ok());

    let release_group = result
        .find(Category::ReleaseGroup)
        .map(|e| e.value.to_string());

    // Anime detection: release_group or AnimeType is the definitive signal.
    // Just having an AnimeTitle + AnimeSeason (e.g. "Breaking.Bad.S01E03")
    // is NOT sufficient — those are standard TV patterns too.
    let is_anime = release_group.is_some()
        || result.find(Category::AnimeType).is_some()
        || RE_ANIME_BRACKET.is_match(filename);

    ParsedName {
        title,
        year,
        season,
        episode,
        media_type: "movie".to_string(), // adjusted later
        is_anime,
        release_group,
    }
}

// ── Regex-based fallback ───────────────────────────────────────────────────

struct RegexResult {
    title: Option<String>,
    year: Option<i32>,
    season: Option<i32>,
    episode: Option<i32>,
    release_group: Option<String>,
}

fn parse_with_regex(filename: &str) -> RegexResult {
    let mut result = RegexResult {
        title: None,
        year: None,
        season: None,
        episode: None,
        release_group: None,
    };

    // Extract season/episode
    if let Some(caps) = RE_SEASON_EP.captures(filename) {
        result.season = caps[1].parse::<i32>().ok();
        result.episode = caps[2].parse::<i32>().ok();
    } else if let Some(caps) = RE_X_SEP.captures(filename) {
        result.season = caps[1].parse::<i32>().ok();
        result.episode = caps[2].parse::<i32>().ok();
    } else if let Some(caps) = RE_EPISODE.captures(filename) {
        result.episode = caps[1].parse::<i32>().ok();
        // Don't set season here — ambiguous
    }

    // Extract year — find all 4-digit years in range, pick first that
    // is NOT part of a longer digit sequence (word-boundary check).
    result.year = RE_YEAR.find_iter(filename).find_map(|m| {
        let start = m.start();
        let end = m.end();
        // Check character before: must not be a digit
        if start > 0 {
            let prev = filename.as_bytes()[start - 1];
            if prev.is_ascii_digit() {
                return None;
            }
        }
        // Check character after: must not be a digit
        if end < filename.len() {
            let next = filename.as_bytes()[end];
            if next.is_ascii_digit() {
                return None;
            }
        }
        m.as_str()
            .parse::<i32>()
            .ok()
            .filter(|&v| (1900..=2099).contains(&v))
    });

    // Extract title from dotted name
    if let Some(caps) = RE_DOT_TITLE.captures(filename) {
        let raw = caps[1].to_string();
        // Replace separators with spaces
        let clean = raw
            .replace(['.', '_', '-'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !clean.is_empty() && clean.len() > 1 {
            result.title = Some(title_case(&clean));
        }
    }

    // Fallback: strip extension and use first meaningful segment
    if result.title.is_none() {
        let stem = filename.rsplit('.').nth(1).unwrap_or(filename);
        // Strip brackets (anime groups) from the start
        let cleaned = RE_ANIME_BRACKET
            .replace(stem, |caps: &regex::Captures| {
                // Release group matched
                result.release_group = Some(caps[1].to_string());
                caps[2].to_string()
            })
            .to_string();

        // If no dot-title pattern matched but we have meaningful content, try first token
        if result.title.is_none() && !cleaned.is_empty() {
            let spaced = cleaned.replace(['.', '_', '-'], " ");
            let tokens: Vec<&str> = spaced
                .split_whitespace()
                .filter(|t| {
                    t.len() > 1
                        && !t.starts_with('[')
                        && !t.ends_with(']')
                        && !t.contains("1080")
                        && !t.contains("720")
                        && !t.contains("x264")
                        && !t.contains("x265")
                        && !t.contains("h264")
                        && !t.contains("h265")
                })
                .collect();
            if !tokens.is_empty() {
                // Take leading title tokens (stop before a season/ep tag)
                let mut title_tokens: Vec<&str> = Vec::new();
                for t in tokens {
                    if t.len() <= 1 || t.starts_with(|c: char| c.is_ascii_digit()) {
                        break;
                    }
                    title_tokens.push(t);
                }
                if !title_tokens.is_empty() {
                    result.title = Some(title_case(&title_tokens.join(" ")));
                }
            }
        }
    }

    // Extract release group from anime brackets if not already set
    if result.release_group.is_none()
        && let Some(caps) = RE_ANIME_BRACKET.captures(filename)
    {
        result.release_group = Some(caps[1].to_string());
    }

    result
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Is this number likely a year (1900-2099) rather than an episode number?
fn is_year_like(n: i32) -> bool {
    (1900..=2099).contains(&n)
}

/// Minimal title-casing: uppercase first letter of each word.
fn title_case(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut at_start = true;
    for c in s.chars() {
        if c.is_whitespace() {
            at_start = true;
            result.push(c);
        } else if at_start {
            result.extend(c.to_uppercase());
            at_start = false;
        } else {
            // Keep the rest as-is (don't lowercase — proper nouns matter)
            result.push(c);
        }
    }
    result
}

/// Extract a title guess from a simple folder name.
fn extract_title_from_folder_simple(folder: &str) -> Option<String> {
    let cleaned = folder
        .replace(['.', '_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if cleaned.len() > 1 {
        Some(title_case(&cleaned))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Anime bracket format ────────────────────────────────────────────────

    #[test]
    fn test_anime_bracket_format() {
        let result = parse_filename(
            "[One Pace][101-102] Reverse Mountain 01 [1080p][En Sub][583096D8].mp4",
            None,
        );
        assert!(
            result.title.is_some(),
            "expected title, got {:?}",
            result.title
        );
        assert!(result.is_anime, "expected is_anime for [One Pace]");
    }

    #[test]
    fn test_fansub_bracket() {
        let result = parse_filename("[HorribleSubs] Attack on Titan - 25 [1080p].mkv", None);
        assert!(result.is_anime);
        assert_eq!(result.release_group.as_deref(), Some("HorribleSubs"));
        assert_eq!(result.title.as_deref(), Some("Attack on Titan"));
        assert_eq!(result.episode, Some(25));
    }

    #[test]
    fn test_subsplease() {
        let result = parse_filename(
            "[SubsPlease] Chainsaw Man - 07 (1080p) [ABC12345].mkv",
            None,
        );
        assert!(result.is_anime);
        assert_eq!(result.release_group.as_deref(), Some("SubsPlease"));
        assert_eq!(result.title.as_deref(), Some("Chainsaw Man"));
        assert_eq!(result.episode, Some(7));
    }

    #[test]
    fn test_anime_with_year() {
        let result = parse_filename("[TaigaSubs] Toradora! (2008) - 01 [720p].mkv", None);
        assert!(result.is_anime);
        assert_eq!(result.year, Some(2008));
        assert_eq!(result.episode, Some(1));
    }

    #[test]
    fn test_anime_title_extraction() {
        let result = parse_filename("[Erai-raws] Jujutsu Kaisen - 01 [1080p].mkv", None);
        assert!(result.is_anime);
        assert!(result.title.is_some(), "expected title");
    }

    // ── Standard TV ─────────────────────────────────────────────────────────

    #[test]
    fn test_standard_tv_episode() {
        let result = parse_filename("Breaking.Bad.S01E03.720p.BluRay.x264.mkv", None);
        assert_eq!(result.title.as_deref(), Some("Breaking Bad"));
        assert_eq!(result.season, Some(1));
        assert_eq!(result.episode, Some(3));
        assert!(!result.is_anime);
    }

    #[test]
    fn test_standard_tv_not_anime() {
        let result = parse_filename("Breaking.Bad.S01E03.720p.BluRay.mkv", None);
        assert!(!result.is_anime);
        assert!(result.release_group.is_none());
    }

    #[test]
    fn test_episode_detected_as_episode_type() {
        let result = parse_filename("Naruto.S02E15.720p.mkv", None);
        assert_eq!(result.media_type, "episode");
    }

    #[test]
    fn test_x_sep_format() {
        let result = parse_filename("Show.Name.1x05.720p.mkv", None);
        assert_eq!(result.title.as_deref(), Some("Show Name"));
        assert_eq!(result.season, Some(1));
        assert_eq!(result.episode, Some(5));
    }

    // ── Movies ──────────────────────────────────────────────────────────────

    #[test]
    fn test_movie_with_year() {
        let result = parse_filename("Spirited.Away.2001.1080p.BluRay.mkv", None);
        assert_eq!(result.title.as_deref(), Some("Spirited Away"));
        assert_eq!(result.year, Some(2001));
        assert_eq!(result.episode, None);
        assert_eq!(result.season, None);
        assert_eq!(result.media_type, "movie");
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_returns_none_for_missing_fields() {
        let result = parse_filename("random_video.mp4", None);
        assert!(result.season.is_none());
        assert!(result.year.is_none());
        assert!(result.episode.is_none());
    }

    #[test]
    fn test_folder_provides_series_context() {
        let result = parse_filename("Episode 01 [1080p].mkv", Some("One Piece"));
        assert!(result.title.is_some(), "folder should provide title");
    }

    #[test]
    fn test_multi_episode_anime() {
        let result = parse_filename(
            "[One Pace][101-102] Reverse Mountain 01 [1080p][En Sub][583096D8].mp4",
            None,
        );
        assert!(result.title.is_some());
        assert!(result.is_anime);
    }
}
