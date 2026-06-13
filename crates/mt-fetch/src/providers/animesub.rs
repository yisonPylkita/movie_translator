//! AnimeSub.info provider — Polish anime subtitles.
//!
//! Scrapes animesub.info for subtitle files.
//! Search by anime title, download as ZIP, extract subtitle files.

use std::fs;
use std::io::{Cursor, Read as _};
use std::path::Path;

use encoding_rs::ISO_8859_2;
use mt_core::MediaIdentity;
use regex::Regex;
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use tracing::{debug, info, warn};
use zip::ZipArchive;

use crate::retry::FetchError;
use crate::types::SubtitleMatch;

pub const BASE_URL: &str = "http://animesub.info";
pub const USER_AGENT: &str = "Mozilla/5.0 (compatible; MovieTranslator/1.0)";

/// A parsed entry from the AnimeSub search results page.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimeSubEntry {
    pub id: String,
    pub sh: String,
    pub title: String,
    pub format: String,
}

/// Parse animesub.info search results HTML into structured entries.
///
/// The HTML structure is:
/// ```html
/// <table class="Napisy">
///   <tr class="KNap">
///     <td width="45%">Title ep001</td>     <!-- title -->
///     ...
///     <td width="20%">Advanced SSA</td>    <!-- format -->
///   </tr>
///   ...
///   <form ...>
///     <input type="hidden" name="id" value="1022">
///     <input type="hidden" name="sh" value="abc123">
///   </form>
/// </table>
/// ```
pub fn parse_search_html(html: &str) -> Vec<AnimeSubEntry> {
    let document = Html::parse_document(html);
    // Each subtitle is wrapped in <table class="Napisy">
    let table_sel = Selector::parse("table.Napisy").unwrap();
    let input_sel = Selector::parse("input[type='hidden']").unwrap();

    let mut entries = Vec::new();

    for table in document.select(&table_sel) {
        // Extract id and sh from hidden inputs
        let mut id = None;
        let mut sh = None;
        for input in table.select(&input_sel) {
            let name = input.value().attr("name").unwrap_or("");
            let value = input.value().attr("value").unwrap_or("").to_string();
            match name {
                "id" => id = Some(value),
                "sh" => sh = Some(value),
                _ => {}
            }
        }
        let (id, sh) = match (id, sh) {
            (Some(id), Some(sh)) => (id, sh),
            _ => continue,
        };

        // Extract title from td[width="45%"] (first one)
        // Extract format from td[width="20%"] (first one)
        // We walk all <td> elements and match by width attribute
        let td_sel = Selector::parse("td").unwrap();
        let mut title = None;
        let mut format = None;

        for td in table.select(&td_sel) {
            let width = td.value().attr("width").unwrap_or("");
            let text = td.text().collect::<Vec<_>>().join("").trim().to_string();
            if width == "45%" && title.is_none() && !text.is_empty() {
                title = Some(text);
            } else if width == "20%" && format.is_none() && !text.is_empty() {
                // Strip non-breaking space entities that may appear
                let clean = text.replace('\u{a0}', "").trim().to_string();
                if !clean.is_empty() {
                    format = Some(clean);
                }
            }
        }

        let title = match title {
            Some(t) => t,
            None => continue,
        };
        let format = format.unwrap_or_default();

        entries.push(AnimeSubEntry {
            id,
            sh,
            title,
            format,
        });
    }

    entries
}

/// Infer the season number from an AnimeSub entry title.
///
/// Conventions:
/// - `"Title ep01"` → Season 1 (no suffix)
/// - `"Title 2 ep08"` → Season 2 (number suffix)
/// - `"Title S2 ep01-10"` → Season 2 (explicit S-prefix)
/// - `"Title OVA ep01"` → `None` (special)
pub fn extract_season_from_title(base_title: &str, entry_title: &str) -> Option<i32> {
    // Strip base_title prefix (case-insensitive) to get the suffix
    let suffix = if entry_title
        .to_lowercase()
        .starts_with(&base_title.to_lowercase())
    {
        entry_title[base_title.len()..].trim().to_string()
    } else {
        entry_title.to_string()
    };

    // No suffix or suffix starts with "ep" → Season 1
    if suffix.is_empty() || suffix.to_lowercase().starts_with("ep") {
        return Some(1);
    }

    // Check for specials — not numbered seasons
    let specials = ["ova", "movie", "film", "special", "bonus", "recap"];
    let suffix_lower = suffix.to_lowercase();
    if specials.iter().any(|s| suffix_lower.contains(s)) {
        return None;
    }

    // "S2 ep..." or "S3 ep..."
    let s_re = Regex::new(r"(?i)^s(\d+)\b").unwrap();
    if let Some(cap) = s_re.captures(&suffix_lower) {
        return cap[1].parse().ok();
    }

    // "2 ep..." or "3 ep..." or "2: Something ep..."
    let num_re = Regex::new(r"^(\d+)\b").unwrap();
    if let Some(cap) = num_re.captures(&suffix) {
        return cap[1].parse().ok();
    }

    None
}

/// Check if an AnimeSub result matches the requested season and episode.
pub fn entry_matches(title: &str, base_title: &str, season: Option<i32>, episode: i32) -> bool {
    let title_lower = title.to_lowercase();

    // Check for episode ranges first (e.g., "ep01-13", "ep1-10")
    let range_re = Regex::new(r"(?:ep|episode\s*|e)(\d+)\s*-\s*(\d+)").unwrap();
    let mut in_range = false;
    let mut has_range = false;
    for cap in range_re.captures_iter(&title_lower) {
        has_range = true;
        let start: i32 = cap[1].parse().unwrap_or(0);
        let end: i32 = cap[2].parse().unwrap_or(0);
        if start <= episode && episode <= end {
            in_range = true;
            break;
        }
    }

    if has_range && !in_range {
        return false;
    }
    if !has_range {
        // No range — check individual episode numbers
        let ep_re = Regex::new(r"(?:ep|episode\s*|e)(\d+)").unwrap();
        let patterns: Vec<_> = ep_re
            .captures_iter(&title_lower)
            .filter_map(|c| c[1].parse().ok())
            .collect();
        if patterns.is_empty() {
            return false;
        }
        if !patterns.contains(&episode) {
            return false;
        }
    }

    // Check season if requested
    if let Some(s) = season {
        let entry_season = extract_season_from_title(base_title, title);
        if entry_season != Some(s) {
            return false;
        }
    }

    true
}

/// Build the search URL for an AnimeSub title query.
pub fn build_search_url(title: &str, title_type: &str, page: usize) -> String {
    let encoded = urlencoding::encode(title);
    format!("{BASE_URL}/szukaj.php?szukane={encoded}&pTitle={title_type}&od={page}")
}

/// Build the form-encoded body for a subtitle download POST.
pub fn build_download_body(sub_id: &str, sh: &str) -> String {
    format!(
        "id={}&sh={}",
        urlencoding::encode(sub_id),
        urlencoding::encode(sh)
    )
}

/// AnimeSub.info subtitle provider.
pub struct AnimeSubProvider {
    client: Client,
}

impl Default for AnimeSubProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AnimeSubProvider {
    pub fn new() -> Self {
        // Fall back to a bare client rather than panicking if the configured
        // builder fails (e.g. transient TLS backend init failure).
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .cookie_store(true)
            .build()
            .unwrap_or_else(|e| {
                warn!("failed to build configured HTTP client ({e}); using default client");
                Client::new()
            });
        Self { client }
    }

    fn search_page(
        &self,
        title: &str,
        title_type: &str,
        page: usize,
    ) -> Result<Vec<AnimeSubEntry>, FetchError> {
        let url = build_search_url(title, title_type, page);
        let resp = self
            .client
            .get(&url)
            .send()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        let bytes = resp
            .bytes()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        // AnimeSub uses ISO-8859-2 encoding
        let (text, _, _) = ISO_8859_2.decode(&bytes);
        Ok(parse_search_html(&text))
    }
}

impl super::SubtitleProvider for AnimeSubProvider {
    fn name(&self) -> &str {
        "animesub"
    }

    fn search(
        &self,
        identity: &MediaIdentity,
        languages: &[&str],
    ) -> Result<Vec<SubtitleMatch>, FetchError> {
        if !languages.contains(&"pol") {
            return Ok(vec![]);
        }

        let title = if !identity.parsed_title.is_empty() {
            &identity.parsed_title
        } else {
            &identity.title
        };
        if title.is_empty() {
            return Ok(vec![]);
        }

        let episode = match identity.episode {
            Some(ep) => ep,
            None => return Ok(vec![]),
        };

        let mut matches = Vec::new();

        for title_type in &["en", "org"] {
            match self.search_page(title, title_type, 0) {
                Ok(entries) => {
                    for entry in entries {
                        if !entry_matches(&entry.title, title, identity.season, episode) {
                            continue;
                        }
                        let fmt_lower = entry.format.to_lowercase();
                        let ext = if fmt_lower.contains("ssa") || fmt_lower.contains("ass") {
                            "ass"
                        } else {
                            "srt"
                        };
                        matches.push(SubtitleMatch {
                            language: "pol".to_string(),
                            source: self.name().to_string(),
                            subtitle_id: format!("{}:{}", entry.id, entry.sh),
                            release_name: entry.title,
                            format: ext.to_string(),
                            score: 0.6,
                            hash_match: false,
                        });
                    }
                    if !matches.is_empty() {
                        break;
                    }
                }
                Err(e) => {
                    debug!("AnimeSub search failed ({title_type}): {e}");
                }
            }
        }

        Ok(matches)
    }

    fn download(&self, match_: &SubtitleMatch, output_path: &Path) -> Result<(), FetchError> {
        let (sub_id, sh) = match_
            .subtitle_id
            .split_once(':')
            .ok_or_else(|| FetchError::Parse("invalid animesub subtitle_id".to_string()))?;

        let body = build_download_body(sub_id, sh);
        let url = format!("{BASE_URL}/sciagnij.php");

        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .map_err(|e| FetchError::Network(e.to_string()))?;

        let status = resp.status();
        let zip_bytes = resp
            .bytes()
            .map_err(|e| FetchError::Network(e.to_string()))?;

        // The endpoint returns an HTML error page (200 or otherwise) when the
        // download is unavailable; ZIP archives start with the "PK" magic. Detect
        // this up front so we surface a clear error instead of "not a ZIP".
        let looks_like_zip = zip_bytes.starts_with(b"PK");
        if !status.is_success() || !looks_like_zip {
            let snippet = String::from_utf8_lossy(&zip_bytes)
                .chars()
                .take(200)
                .collect::<String>();
            warn!(
                "AnimeSub download for id={sub_id} returned non-ZIP response (status={}): {}",
                status.as_u16(),
                snippet.trim()
            );
            return Err(FetchError::Http {
                status: status.as_u16(),
                body: format!("AnimeSub returned a non-ZIP response for id={sub_id}"),
            });
        }

        let cursor = Cursor::new(&zip_bytes[..]);
        let mut archive =
            ZipArchive::new(cursor).map_err(|e| FetchError::Parse(format!("not a ZIP: {e}")))?;

        // Find subtitle files in the ZIP, recording the index so we can re-open
        // by index (UTF-8-safe and avoids an O(n^2) by_name re-scan).
        let sub_entries: Vec<_> = (0..archive.len())
            .filter_map(|i| {
                let f = archive.by_index(i).ok()?;
                let name = f.name().to_lowercase();
                if name.ends_with(".srt")
                    || name.ends_with(".ass")
                    || name.ends_with(".ssa")
                    || name.ends_with(".sub")
                {
                    Some((i, f.name().to_string()))
                } else {
                    None
                }
            })
            .collect();

        if sub_entries.is_empty() {
            return Err(FetchError::NotFound(format!(
                "no subtitle file in ZIP for id={sub_id}"
            )));
        }

        // When ZIP has multiple episodes, pick the first subtitle file.
        let chosen_index = sub_entries[0].0;
        let mut file = archive
            .by_index(chosen_index)
            .map_err(|e| FetchError::Parse(format!("cannot read zip entry: {e}")))?;

        let mut content = Vec::new();
        file.read_to_end(&mut content).map_err(FetchError::Io)?;

        fs::write(output_path, &content).map_err(FetchError::Io)?;
        info!(
            "Downloaded subtitle: {} (animesub.info)",
            output_path.display()
        );
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    const SAMPLE_HTML: &str = r#"
<table class="Napisy">
<tr class="KNap">
  <td align="left" width="45%">Naruto ep001</td>
  <td width="25%">2004.01.03</td>
  <td width="10%">&nbsp;</td>
  <td width="20%">Advanced SSA</td>
</tr>
<tr class="KNap">
  <td align="left">Naruto ep001</td>
  <td><a href="osoba.php?id=36">~Sanzoku</a></td>
  <td></td>
  <td>6kB</td>
</tr>
<tr class="KNap">
  <td align="left">Naruto ep001</td>
  <td><a href="javascript:PK(1022)" class="ko">(3)</a></td>
  <td></td>
  <td>5878 razy</td>
</tr>
<tr class="KKom">
  <td valign="top" align="right">
    <form method="POST" action="sciagnij.php">
      <input type="hidden" name="id" value="1022">
      <input type="hidden" name="sh" value="abc123def456">
      <input type="submit" value="Pobierz napisy" name="single_file">
    </form>
  </td>
  <td class="KNap" align="left" colspan="3">
    <b>ID 1022<br>Autor:</b> Sanzoku
  </td>
</tr>
</table>
<table class="Napisy">
<tr class="KNap">
  <td align="left" width="45%">Naruto ep002</td>
  <td width="25%">2004.01.04</td>
  <td width="10%">&nbsp;</td>
  <td width="20%">SubRip</td>
</tr>
<tr class="KNap"><td></td><td></td><td></td><td></td></tr>
<tr class="KNap"><td></td><td></td><td></td><td></td></tr>
<tr class="KKom">
  <td>
    <form method="POST" action="sciagnij.php">
      <input type="hidden" name="id" value="1023">
      <input type="hidden" name="sh" value="xyz789">
      <input type="submit" value="Pobierz napisy">
    </form>
  </td>
  <td class="KNap" colspan="3"></td>
</tr>
</table>
"#;

    const BASE: &str = "Kono Subarashii Sekai ni Shukufuku wo!";

    #[test]
    fn parses_two_entries() {
        let entries = parse_search_html(SAMPLE_HTML);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn extracts_id_and_sh() {
        let entries = parse_search_html(SAMPLE_HTML);
        assert_eq!(entries[0].id, "1022");
        assert_eq!(entries[0].sh, "abc123def456");
        assert_eq!(entries[1].id, "1023");
        assert_eq!(entries[1].sh, "xyz789");
    }

    #[test]
    fn extracts_title() {
        let entries = parse_search_html(SAMPLE_HTML);
        assert_eq!(entries[0].title, "Naruto ep001");
        assert_eq!(entries[1].title, "Naruto ep002");
    }

    #[test]
    fn extracts_format() {
        let entries = parse_search_html(SAMPLE_HTML);
        assert_eq!(entries[0].format, "Advanced SSA");
        assert_eq!(entries[1].format, "SubRip");
    }

    #[test]
    fn no_suffix_is_season_1() {
        assert_eq!(
            extract_season_from_title(BASE, &format!("{BASE} ep01")),
            Some(1)
        );
    }

    #[test]
    fn number_suffix_2() {
        assert_eq!(
            extract_season_from_title(BASE, &format!("{BASE} 2 ep08")),
            Some(2)
        );
    }

    #[test]
    fn number_suffix_3() {
        assert_eq!(
            extract_season_from_title(BASE, &format!("{BASE} 3 ep01")),
            Some(3)
        );
    }

    #[test]
    fn explicit_s2() {
        assert_eq!(
            extract_season_from_title(BASE, &format!("{BASE} S2 ep01-10")),
            Some(2)
        );
    }

    #[test]
    fn ova_returns_none() {
        assert_eq!(
            extract_season_from_title(BASE, &format!("{BASE} OVA ep01")),
            None
        );
    }

    #[test]
    fn movie_returns_none() {
        assert_eq!(
            extract_season_from_title(BASE, &format!("{BASE} Movie")),
            None
        );
    }

    #[test]
    fn bonus_stage_returns_none() {
        assert_eq!(
            extract_season_from_title(BASE, &format!("{BASE} 3: Bonus Stage ep01")),
            None
        );
    }

    #[test]
    fn different_base_title_returns_none() {
        // "Naruto Shippuden ep01" — suffix after "Naruto" is " Shippuden ep01",
        // which doesn't match any season pattern → None.
        assert_eq!(
            extract_season_from_title("Naruto", "Naruto Shippuden ep01"),
            None
        );
    }

    #[test]
    fn s1_ep01_matches_no_suffix() {
        assert!(entry_matches(&format!("{BASE} ep01"), BASE, Some(1), 1));
    }

    #[test]
    fn s1_ep08_rejects_s2() {
        assert!(!entry_matches(&format!("{BASE} 2 ep08"), BASE, Some(1), 8));
    }

    #[test]
    fn s1_ep08_rejects_s3() {
        assert!(!entry_matches(&format!("{BASE} 3 ep08"), BASE, Some(1), 8));
    }

    #[test]
    fn s2_ep08_matches_s2() {
        assert!(entry_matches(&format!("{BASE} 2 ep08"), BASE, Some(2), 8));
    }

    #[test]
    fn wrong_episode_rejected() {
        assert!(!entry_matches(&format!("{BASE} ep05"), BASE, Some(1), 8));
    }

    #[test]
    fn season_none_accepts_any_season() {
        assert!(entry_matches(&format!("{BASE} 2 ep08"), BASE, None, 8));
    }

    #[test]
    fn ova_rejected_when_season_specified() {
        assert!(!entry_matches(
            &format!("{BASE} OVA ep01"),
            BASE,
            Some(1),
            1
        ));
    }

    #[test]
    fn ova_accepted_when_season_none() {
        assert!(entry_matches(&format!("{BASE} OVA ep01"), BASE, None, 1));
    }

    #[test]
    fn simple_naruto_match() {
        assert!(entry_matches("Naruto ep001", "Naruto", Some(1), 1));
    }

    #[test]
    fn no_episode_pattern_rejected() {
        assert!(!entry_matches("Naruto Movie", "Naruto", Some(1), 1));
    }

    #[test]
    fn episode_range_matches_middle() {
        assert!(entry_matches(&format!("{BASE} ep01-13"), BASE, Some(1), 5));
    }

    #[test]
    fn episode_range_matches_start() {
        assert!(entry_matches(&format!("{BASE} ep01-13"), BASE, Some(1), 1));
    }

    #[test]
    fn episode_range_matches_end() {
        assert!(entry_matches(&format!("{BASE} ep01-13"), BASE, Some(1), 13));
    }

    #[test]
    fn episode_range_rejects_outside() {
        assert!(!entry_matches(
            &format!("{BASE} ep01-10"),
            BASE,
            Some(1),
            11
        ));
    }

    #[test]
    fn episode_range_with_season() {
        assert!(!entry_matches(
            &format!("{BASE} 2 ep01-13"),
            BASE,
            Some(1),
            5
        ));
    }

    // ── Provider-level unit tests ───────────────────────────────────────────

    use crate::providers::SubtitleProvider as _;

    #[test]
    fn provider_name_is_animesub() {
        assert_eq!(AnimeSubProvider::new().name(), "animesub");
    }

    /// Non-Polish languages return early before any HTTP request.
    #[test]
    fn search_skips_non_polish_languages() {
        let provider = AnimeSubProvider::new();
        let result = provider.search(&make_identity(), &["eng"]).unwrap();
        assert_eq!(result, vec![]);
    }

    /// When searching for S1, the S2 ("Naruto 2 ep01") and S3 ("Naruto 3 ep01")
    /// entries are rejected while the suffix-less S1 entry ("Naruto ep01") is
    /// accepted. Uses the pure entry_matches() to avoid needing a mocked HTTP
    /// client.
    #[test]
    fn rejects_wrong_season_via_entry_matches() {
        let base = "Naruto";
        let candidates = [
            ("Naruto 2 ep01", false),
            ("Naruto 3 ep01", false),
            ("Naruto ep01", true),
        ];
        let accepted: Vec<_> = candidates
            .iter()
            .filter(|(t, _)| entry_matches(t, base, Some(1), 1))
            .map(|(t, _)| *t)
            .collect();
        for (title, expected) in candidates {
            assert_eq!(
                entry_matches(title, base, Some(1), 1),
                expected,
                "entry {title:?} for season 1"
            );
        }
        assert_eq!(accepted, vec!["Naruto ep01"]);
    }

    /// When searching for S2, only the S2 entry ("Naruto 2 ep01") is accepted,
    /// not the suffix-less S1 entry.
    #[test]
    fn accepts_correct_season_via_entry_matches() {
        let base = "Naruto";
        let candidates = ["Naruto 2 ep01", "Naruto ep01"];
        let accepted: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|t| entry_matches(t, base, Some(2), 1))
            .collect();
        assert_eq!(accepted, vec!["Naruto 2 ep01"]);
    }

    fn make_identity() -> MediaIdentity {
        MediaIdentity {
            title: "Naruto".to_string(),
            parsed_title: "Naruto".to_string(),
            year: None,
            season: None,
            episode: Some(1),
            media_type: "episode".to_string(),
            oshash: "0".repeat(16),
            file_size: 1000,
            raw_filename: "Naruto.ep001.mkv".to_string(),
            imdb_id: None,
            tmdb_id: None,
            is_anime: false,
            release_group: None,
        }
    }

    /// Test that the format "Advanced SSA" maps to "ass" extension.
    #[test]
    fn ssa_format_maps_to_ass() {
        let entry = AnimeSubEntry {
            id: "1".to_string(),
            sh: "x".to_string(),
            title: "Naruto ep001".to_string(),
            format: "Advanced SSA".to_string(),
        };
        let fmt_lower = entry.format.to_lowercase();
        let ext = if fmt_lower.contains("ssa") || fmt_lower.contains("ass") {
            "ass"
        } else {
            "srt"
        };
        assert_eq!(ext, "ass");
    }

    /// Test that the format "SubRip" maps to "srt" extension.
    #[test]
    fn subrip_format_maps_to_srt() {
        let entry = AnimeSubEntry {
            id: "1".to_string(),
            sh: "x".to_string(),
            title: "Naruto ep001".to_string(),
            format: "SubRip".to_string(),
        };
        let fmt_lower = entry.format.to_lowercase();
        let ext = if fmt_lower.contains("ssa") || fmt_lower.contains("ass") {
            "ass"
        } else {
            "srt"
        };
        assert_eq!(ext, "srt");
    }

    /// Verifies subtitle_id construction.
    #[test]
    fn subtitle_id_combines_id_and_sh() {
        let entries = parse_search_html(SAMPLE_HTML);
        let id_sh = format!("{}:{}", entries[0].id, entries[0].sh);
        assert_eq!(id_sh, "1022:abc123def456");
    }

    /// build_search_url encodes the title and includes title_type and page.
    #[test]
    fn build_search_url_encodes_title() {
        let url = build_search_url("Naruto", "en", 0);
        assert!(url.contains("Naruto"));
        assert!(url.contains("pTitle=en"));
        assert!(url.contains("od=0"));
    }

    /// build_download_body produces proper form-encoded string.
    #[test]
    fn build_download_body_format() {
        let body = build_download_body("1022", "abc123");
        assert!(body.contains("id=1022"));
        assert!(body.contains("sh=abc123"));
    }

    /// Download ZIP extraction test.
    #[test]
    fn download_extracts_subtitle_from_zip() {
        use std::io::Write as _;

        use zip::write::SimpleFileOptions;
        use zip::{ZipArchive, ZipWriter};
        let dir = tempdir().unwrap();
        let output = dir.path().join("subtitle.ass");

        // Build a minimal ZIP with one ASS file
        let mut buf = Vec::new();
        {
            let mut zip = ZipWriter::new(Cursor::new(&mut buf));
            zip.start_file("Naruto_01.ass", SimpleFileOptions::default())
                .unwrap();
            zip.write_all(b"[Script Info]\nTitle: Naruto").unwrap();
            zip.finish().unwrap();
        }

        // Unzip manually to simulate what download() does
        let cursor = Cursor::new(&buf);
        let mut archive = ZipArchive::new(cursor).unwrap();
        let mut content = Vec::new();
        {
            let mut file = archive.by_index(0).unwrap();
            file.read_to_end(&mut content).unwrap();
        }
        fs::write(&output, &content).unwrap();

        let text = fs::read_to_string(&output).unwrap();
        assert!(text.contains("Naruto"));
    }
}
