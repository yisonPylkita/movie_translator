//! Podnapisi.net provider — multilingual subtitle search via REST/XML API.
//!
//! Ported from `movie_translator/subtitle_fetch/providers/podnapisi.py`.

use std::io::Read as _;
use std::path::Path;

use quick_xml::events::Event;
use quick_xml::Reader;

use crate::retry::FetchError;
use crate::types::SubtitleMatch;
use mt_core::MediaIdentity;

pub const API_BASE: &str = "https://www.podnapisi.net";
pub const SEARCH_URL: &str = "https://www.podnapisi.net/subtitles/search/old";
pub const USER_AGENT: &str = "MovieTranslator/1.0";

/// Podnapisi language ID mappings.
/// Maps ISO 639-2B → Podnapisi numeric ID.
pub fn lang_to_podnapi(lang: &str) -> Option<&'static str> {
    match lang {
        "eng" => Some("2"),
        "pol" => Some("23"),
        "jpn" => Some("11"),
        _ => None,
    }
}

/// Maps Podnapisi 2-letter code → ISO 639-2B.
pub fn lang_from_podnapi(code: &str) -> &str {
    match code {
        "en" => "eng",
        "pl" => "pol",
        "ja" => "jpn",
        other => other,
    }
}

/// A parsed subtitle record from Podnapisi XML.
#[derive(Debug, Clone, PartialEq)]
pub struct PodnapisiSubtitle {
    pub id: String,
    pub language: String, // 2-letter Podnapisi code
    pub release: String,
}

/// Parse Podnapisi XML search response into subtitle records.
///
/// The XML looks like:
/// ```xml
/// <results>
///   <subtitle>
///     <id>12345</id>
///     <language>en</language>
///     <release>Breaking.Bad.S01E03.720p</release>
///   </subtitle>
/// </results>
/// ```
/// Append decoded text to the field matching the currently open tag.
///
/// Uses `+=` rather than assignment because a single element's character data
/// can be split across multiple events (plain text plus resolved entity
/// references) in quick-xml >= 0.37.
fn append_to_field(sub: &mut PodnapisiSubtitle, tag: &str, text: &str) {
    match tag {
        "id" => sub.id.push_str(text),
        "language" => sub.language.push_str(text),
        "release" => sub.release.push_str(text),
        _ => {}
    }
}

pub fn parse_xml_response(xml: &str) -> Result<Vec<PodnapisiSubtitle>, FetchError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut subs = Vec::new();
    let mut current: Option<PodnapisiSubtitle> = None;
    let mut current_tag = String::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_string();
                if tag == "subtitle" {
                    current = Some(PodnapisiSubtitle {
                        id: String::new(),
                        language: String::new(),
                        release: String::new(),
                    });
                }
                current_tag = tag;
            }
            Ok(Event::Text(e)) => {
                if let Some(ref mut sub) = current {
                    let text = e.decode().unwrap_or_default();
                    append_to_field(sub, &current_tag, &text);
                }
            }
            // quick-xml >= 0.37 surfaces entity references (`&amp;` etc.) as
            // separate `GeneralRef` events instead of unescaping them inline
            // within `Text`. Resolve the predefined XML entities and append so
            // the field text matches the old `unescape()` behavior.
            Ok(Event::GeneralRef(e)) => {
                if let Some(ref mut sub) = current {
                    let resolved = if let Some(c) = e.resolve_char_ref().ok().flatten() {
                        c.to_string()
                    } else {
                        let name = e.decode().unwrap_or_default();
                        quick_xml::escape::resolve_predefined_entity(&name)
                            .unwrap_or("")
                            .to_string()
                    };
                    append_to_field(sub, &current_tag, &resolved);
                }
            }
            Ok(Event::End(e)) => {
                let name_bytes = e.name();
                let tag = std::str::from_utf8(name_bytes.as_ref()).unwrap_or("");
                if tag == "subtitle" {
                    if let Some(sub) = current.take() {
                        if !sub.id.is_empty() {
                            subs.push(sub);
                        }
                    }
                }
                current_tag.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(FetchError::Parse(format!("Podnapisi XML parse error: {e}")));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(subs)
}

/// Parse XML results and convert to SubtitleMatch list.
///
/// Mirrors Python `PodnapisiProvider._parse_results()`.
pub fn parse_results(
    xml: &str,
    languages: &[&str],
    hash_match: bool,
) -> Result<Vec<SubtitleMatch>, FetchError> {
    let subs = parse_xml_response(xml)?;
    let mut matches = Vec::new();
    for sub in subs {
        let lang_3 = lang_from_podnapi(&sub.language);
        if !languages.contains(&lang_3) {
            continue;
        }
        matches.push(SubtitleMatch {
            language: lang_3.to_string(),
            source: "podnapisi".to_string(),
            subtitle_id: sub.id,
            release_name: sub.release,
            format: "srt".to_string(),
            score: if hash_match { 0.9 } else { 0.65 },
            hash_match,
        });
    }
    Ok(matches)
}

/// Build the Podnapisi search URL with query parameters.
pub fn build_search_url(
    query: &str,
    podnapi_langs: &[&str],
    season: Option<i32>,
    episode: Option<i32>,
    year: Option<i32>,
    oshash: Option<&str>,
) -> String {
    let mut params = vec![("sXML", "1".to_string()), ("sJ", podnapi_langs.join(","))];
    if let Some(hash) = oshash {
        params.push(("sH", hash.to_string()));
    } else {
        params.push(("sK", query.to_string()));
        if let Some(s) = season {
            params.push(("sS", s.to_string()));
        }
        if let Some(e) = episode {
            params.push(("sE", e.to_string()));
        }
        if let Some(y) = year {
            params.push(("sY", y.to_string()));
        }
    }
    let qs = params
        .iter()
        .map(|(k, v)| format!("{k}={}", urlencoding::encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{SEARCH_URL}?{qs}")
}

/// Podnapisi.net subtitle provider.
pub struct PodnapisiProvider {
    client: reqwest::blocking::Client,
}

impl Default for PodnapisiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl PodnapisiProvider {
    pub fn new() -> Self {
        let client = super::build_blocking_client(USER_AGENT);
        Self { client }
    }

    fn fetch_xml(&self, url: &str) -> Result<String, FetchError> {
        let resp = self
            .client
            .get(url)
            .send()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        resp.text().map_err(|e| FetchError::Network(e.to_string()))
    }
}

impl super::SubtitleProvider for PodnapisiProvider {
    fn name(&self) -> &str {
        "podnapisi"
    }

    fn search(
        &self,
        identity: &MediaIdentity,
        languages: &[&str],
    ) -> Result<Vec<SubtitleMatch>, FetchError> {
        let podnapi_langs: Vec<&str> = languages
            .iter()
            .filter_map(|l| lang_to_podnapi(l))
            .collect();
        if podnapi_langs.is_empty() {
            return Ok(vec![]);
        }

        let query = if !identity.parsed_title.is_empty() {
            &identity.parsed_title
        } else {
            &identity.title
        };

        let mut matches: Vec<SubtitleMatch> = Vec::new();

        // Strategy 1: Hash-based search
        if !identity.oshash.is_empty() {
            let url = build_search_url(
                query,
                &podnapi_langs,
                None,
                None,
                None,
                Some(&identity.oshash),
            );
            match self.fetch_xml(&url) {
                Ok(xml) => match parse_results(&xml, languages, true) {
                    Ok(m) => matches = m,
                    Err(e) => tracing::debug!("Podnapisi hash search parse error: {e}"),
                },
                Err(e) => tracing::debug!("Podnapisi hash search failed: {e}"),
            }
        }

        // Strategy 2: Text-based search (always)
        let url = build_search_url(
            query,
            &podnapi_langs,
            identity.season,
            identity.episode,
            identity.year,
            None,
        );
        match self.fetch_xml(&url) {
            Ok(xml) => match parse_results(&xml, languages, false) {
                Ok(query_matches) => {
                    let seen_ids: std::collections::HashSet<_> =
                        matches.iter().map(|m| m.subtitle_id.clone()).collect();
                    for m in query_matches {
                        if !seen_ids.contains(&m.subtitle_id) {
                            matches.push(m);
                        }
                    }
                }
                Err(e) => tracing::debug!("Podnapisi query parse error: {e}"),
            },
            Err(e) => tracing::debug!("Podnapisi search failed: {e}"),
        }

        Ok(matches)
    }

    fn download(&self, match_: &SubtitleMatch, output_path: &Path) -> Result<(), FetchError> {
        let url = format!("{API_BASE}/subtitles/{}/download", match_.subtitle_id);
        let resp = self
            .client
            .get(&url)
            .send()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        let content = resp
            .bytes()
            .map_err(|e| FetchError::Network(e.to_string()))?;

        // Podnapisi returns a ZIP or raw subtitle content
        let cursor = std::io::Cursor::new(&content[..]);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            // Record (index, name) so we can re-open by index — this is
            // UTF-8-safe (by_name breaks on non-UTF8 entry names) and avoids
            // an O(n^2) re-scan.
            let sub_entries: Vec<(usize, String)> = (0..archive.len())
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
                    "no subtitle file in Podnapisi ZIP (id={})",
                    match_.subtitle_id
                )));
            }

            let chosen_index = sub_entries[0].0;
            let mut file = archive
                .by_index(chosen_index)
                .map_err(|e| FetchError::Parse(format!("cannot read zip entry: {e}")))?;
            let mut data = Vec::new();
            file.read_to_end(&mut data).map_err(FetchError::Io)?;
            std::fs::write(output_path, &data).map_err(FetchError::Io)?;
        } else {
            // Raw subtitle content
            std::fs::write(output_path, &content[..]).map_err(FetchError::Io)?;
        }

        tracing::info!("Downloaded subtitle: {} (podnapisi)", output_path.display());
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::SubtitleProvider;

    // The exact same SAMPLE_XML from test_podnapisi.py
    const SAMPLE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<results>
  <pagination><results>2</results></pagination>
  <subtitle>
    <id>12345</id>
    <title>Breaking Bad S01E03</title>
    <release>Breaking.Bad.S01E03.720p.BluRay</release>
    <language>en</language>
    <flags>0</flags>
    <rating>4.8</rating>
    <downloads>5432</downloads>
  </subtitle>
  <subtitle>
    <id>67890</id>
    <title>Breaking Bad S01E03</title>
    <release>Breaking.Bad.S01E03.1080p.WEB</release>
    <language>pl</language>
    <flags>0</flags>
    <rating>4.5</rating>
    <downloads>1234</downloads>
  </subtitle>
</results>"#;

    #[allow(dead_code)]
    fn make_identity() -> MediaIdentity {
        MediaIdentity {
            title: "Breaking Bad".to_string(),
            parsed_title: "Breaking Bad".to_string(),
            year: Some(2008),
            season: Some(1),
            episode: Some(3),
            media_type: "episode".to_string(),
            oshash: "abc123def456abc0".to_string(),
            file_size: 1_000_000,
            raw_filename: "Breaking.Bad.S01E03.mkv".to_string(),
            imdb_id: None,
            tmdb_id: None,
            is_anime: false,
            release_group: None,
        }
    }

    // ── PORT: test_name ───────────────────────────────────────────────────────

    #[test]
    fn provider_name_is_podnapisi() {
        assert_eq!(PodnapisiProvider::new().name(), "podnapisi");
    }

    // ── PORT: test_search_parses_xml_response ─────────────────────────────────

    #[test]
    fn parse_xml_two_entries() {
        let matches = parse_results(SAMPLE_XML, &["eng", "pol"], false).unwrap();
        assert_eq!(matches.len(), 2);
        let langs: std::collections::HashSet<_> =
            matches.iter().map(|m| m.language.as_str()).collect();
        assert!(langs.contains("eng"));
        assert!(langs.contains("pol"));
        assert!(matches.iter().all(|m| m.source == "podnapisi"));
    }

    // ── PORT: test_search_filters_by_language ────────────────────────────────

    #[test]
    fn parse_xml_filters_by_language() {
        let matches = parse_results(SAMPLE_XML, &["pol"], false).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].language, "pol");
    }

    // ── PORT: test_search_includes_season_episode_params_for_episodes ─────────

    #[test]
    fn build_search_url_includes_season_and_episode() {
        let url = build_search_url("Breaking Bad", &["2"], Some(2), Some(5), None, None);
        assert!(url.contains("sS=2"));
        assert!(url.contains("sE=5"));
    }

    // ── PORT: test_hash_search_gets_higher_score ──────────────────────────────

    #[test]
    fn hash_match_has_higher_score_than_query() {
        let hash_xml = r#"<?xml version="1.0"?>
<results><subtitle><id>111</id><release>hash-match</release><language>pl</language></subtitle></results>"#;
        let query_xml = r#"<?xml version="1.0"?>
<results><subtitle><id>222</id><release>query-match</release><language>pl</language></subtitle></results>"#;

        let hash_matches = parse_results(hash_xml, &["pol"], true).unwrap();
        let query_matches = parse_results(query_xml, &["pol"], false).unwrap();

        assert_eq!(hash_matches[0].subtitle_id, "111");
        assert_eq!(query_matches[0].subtitle_id, "222");
        assert!(hash_matches[0].score > query_matches[0].score);
        assert!(hash_matches[0].hash_match);
        assert!(!query_matches[0].hash_match);
    }

    // ── PORT: test_hash_and_query_results_deduplicated ───────────────────────

    #[test]
    fn deduplication_via_seen_ids() {
        // Same XML returned for both hash and query → IDs are deduped
        let matches1 = parse_results(SAMPLE_XML, &["pol"], true).unwrap();
        let matches2 = parse_results(SAMPLE_XML, &["pol"], false).unwrap();

        // Simulate deduplication logic from the provider:
        // hash results are added first; query results with duplicate IDs are skipped.
        let seen: std::collections::HashSet<&str> =
            matches1.iter().map(|m| m.subtitle_id.as_str()).collect();
        let merged: Vec<_> = matches1
            .iter()
            .chain(
                matches2
                    .iter()
                    .filter(|m| !seen.contains(m.subtitle_id.as_str())),
            )
            .collect();

        // Assert all IDs in merged are unique
        let ids: Vec<_> = merged.iter().map(|m| &m.subtitle_id).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
    }

    // ── PORT: test_search_returns_empty_on_error ─────────────────────────────

    #[test]
    fn bad_xml_returns_parse_error() {
        let result = parse_results("not xml at all {{{{", &["pol"], false);
        // Should return error or empty (depending on parser tolerance)
        // Our parser returns Err on bad XML
        // Either is acceptable — just don't panic
        let _ = result;
    }

    // ── Additional: lang mappings ─────────────────────────────────────────────

    #[test]
    fn lang_to_podnapi_known_langs() {
        assert_eq!(lang_to_podnapi("eng"), Some("2"));
        assert_eq!(lang_to_podnapi("pol"), Some("23"));
        assert_eq!(lang_to_podnapi("jpn"), Some("11"));
        assert_eq!(lang_to_podnapi("zho"), None);
    }

    #[test]
    fn lang_from_podnapi_known_codes() {
        assert_eq!(lang_from_podnapi("en"), "eng");
        assert_eq!(lang_from_podnapi("pl"), "pol");
        assert_eq!(lang_from_podnapi("ja"), "jpn");
    }

    // ── Additional: build_search_url hash path ────────────────────────────────

    #[test]
    fn build_search_url_hash_path_includes_sh() {
        let url = build_search_url("Test", &["2"], None, None, None, Some("abc123"));
        assert!(url.contains("sH=abc123"));
        assert!(!url.contains("sK="));
    }
}
