//! TMDB API integration for enriching media identity with external IDs.
//!
//! Ported from `movie_translator/identifier/tmdb.py`.
//!
//! Uses the TMDB API v3 (free tier). Requires `TMDB_API_KEY` env var.
//! When the key is absent or any request fails, returns `None` silently —
//! TMDB enrichment is always optional.

use mt_core::{MtError, Result};
use serde::Deserialize;

const TMDB_BASE: &str = "https://api.themoviedb.org/3";

/// Enrichment data fetched from TMDB.
#[derive(Debug, Clone, PartialEq)]
pub struct TmdbResult {
    pub tmdb_id: i32,
    pub imdb_id: Option<String>,
}

// ── Internal TMDB JSON shapes ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    results: Vec<SearchHit>,
}

#[derive(Deserialize)]
struct SearchHit {
    id: Option<i32>,
}

#[derive(Deserialize, Default)]
struct DetailResponse {
    imdb_id: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Parse a TMDB search response JSON and extract the first result's `tmdb_id`.
///
/// Factored out from HTTP so it can be unit-tested against captured JSON.
pub(crate) fn parse_search_response(json: &str) -> Option<i32> {
    let resp: SearchResponse = serde_json::from_str(json).ok()?;
    resp.results.into_iter().next().and_then(|h| h.id)
}

/// Parse a TMDB detail / external_ids response JSON and extract `imdb_id`.
pub(crate) fn parse_detail_response(json: &str) -> Option<String> {
    let resp: DetailResponse = serde_json::from_str(json).ok()?;
    resp.imdb_id
}

/// Build the search URL and query params (no HTTP call).
///
/// `media_type` is `"episode"` → `/search/tv`, else `/search/movie`.
pub(crate) fn build_search_url(
    api_key: &str,
    title: &str,
    year: Option<i32>,
    media_type: &str,
) -> String {
    let (endpoint, year_key) = if media_type == "episode" {
        ("/search/tv", "first_air_date_year")
    } else {
        ("/search/movie", "year")
    };

    let mut params: Vec<(&str, String)> = vec![
        ("api_key", api_key.to_string()),
        ("query", title.to_string()),
    ];
    if let Some(y) = year {
        params.push((year_key, y.to_string()));
    }

    let qs: Vec<String> = params
        .iter()
        .map(|(k, v)| {
            format!(
                "{}={}",
                k,
                url_encode(v)
            )
        })
        .collect();

    format!("{TMDB_BASE}{endpoint}?{}", qs.join("&"))
}

/// Minimal percent-encoding for TMDB query params (spaces → `%20`, etc.).
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => out.push(c),
            ' ' => out.push_str("%20"),
            c => {
                let encoded = c.to_string();
                for b in encoded.as_bytes() {
                    out.push_str(&format!("%{:02X}", b));
                }
            }
        }
    }
    out
}

/// Look up a title on TMDB and return enrichment data.
///
/// Returns `None` if:
/// - `TMDB_API_KEY` is not set.
/// - No results are found.
/// - Any HTTP or parsing error occurs.
pub fn lookup_tmdb(title: &str, year: Option<i32>, media_type: &str) -> Option<TmdbResult> {
    let api_key = std::env::var("TMDB_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        return None;
    }

    lookup_tmdb_with_client(api_key.as_str(), title, year, media_type).ok().flatten()
}

/// Inner function that performs the actual HTTP calls.
/// Separated so it can return a `Result` internally while `lookup_tmdb`
/// maps all errors to `None`.
fn lookup_tmdb_with_client(
    api_key: &str,
    title: &str,
    year: Option<i32>,
    media_type: &str,
) -> Result<Option<TmdbResult>> {
    let search_url = build_search_url(api_key, title, year, media_type);

    let client = reqwest::blocking::Client::builder()
        .user_agent("MovieTranslator/1.0")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| MtError::Network(e.to_string()))?;

    let search_resp = client
        .get(&search_url)
        .send()
        .map_err(|e| MtError::Network(e.to_string()))?;

    let search_body = search_resp
        .text()
        .map_err(|e| MtError::Network(e.to_string()))?;

    let tmdb_id = match parse_search_response(&search_body) {
        Some(id) => id,
        None => return Ok(None),
    };

    // Try to fetch IMDB ID from the detail endpoint (nice-to-have)
    let imdb_id = fetch_imdb_id(&client, api_key, tmdb_id, media_type);

    Ok(Some(TmdbResult { tmdb_id, imdb_id }))
}

fn fetch_imdb_id(
    client: &reqwest::blocking::Client,
    api_key: &str,
    tmdb_id: i32,
    media_type: &str,
) -> Option<String> {
    let detail_path = if media_type == "episode" {
        format!("/tv/{tmdb_id}/external_ids")
    } else {
        format!("/movie/{tmdb_id}")
    };
    let url = format!("{TMDB_BASE}{detail_path}?api_key={api_key}");

    let body = client.get(&url).send().ok()?.text().ok()?;
    parse_detail_response(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fixture JSON captured from real TMDB responses ────────────────────────

    const MOVIE_SEARCH_JSON: &str = r#"{
        "page": 1,
        "results": [
            {
                "id": 1396,
                "title": "Breaking Bad",
                "release_date": "2008-01-20",
                "overview": "...",
                "vote_average": 8.9
            }
        ],
        "total_results": 1,
        "total_pages": 1
    }"#;

    const TV_SEARCH_JSON: &str = r#"{
        "page": 1,
        "results": [
            {
                "id": 1399,
                "name": "Game of Thrones",
                "first_air_date": "2011-04-17"
            }
        ],
        "total_results": 1,
        "total_pages": 1
    }"#;

    const EMPTY_SEARCH_JSON: &str = r#"{"results": [], "total_results": 0}"#;

    const MOVIE_DETAIL_JSON: &str = r#"{
        "id": 1396,
        "imdb_id": "tt0903747",
        "title": "Breaking Bad"
    }"#;

    const TV_EXTERNAL_IDS_JSON: &str = r#"{
        "id": 1399,
        "imdb_id": "tt0944947",
        "tvdb_id": 121361
    }"#;

    const NO_IMDB_JSON: &str = r#"{"id": 999}"#;

    // ── parse_search_response ──────────────────────────────────────────────────

    #[test]
    fn parse_movie_search_extracts_id() {
        let id = parse_search_response(MOVIE_SEARCH_JSON);
        assert_eq!(id, Some(1396));
    }

    #[test]
    fn parse_tv_search_extracts_id() {
        let id = parse_search_response(TV_SEARCH_JSON);
        assert_eq!(id, Some(1399));
    }

    #[test]
    fn parse_empty_search_returns_none() {
        let id = parse_search_response(EMPTY_SEARCH_JSON);
        assert!(id.is_none());
    }

    #[test]
    fn parse_invalid_json_returns_none() {
        let id = parse_search_response("not json");
        assert!(id.is_none());
    }

    // ── parse_detail_response ──────────────────────────────────────────────────

    #[test]
    fn parse_movie_detail_extracts_imdb_id() {
        let imdb = parse_detail_response(MOVIE_DETAIL_JSON);
        assert_eq!(imdb.as_deref(), Some("tt0903747"));
    }

    #[test]
    fn parse_tv_external_ids_extracts_imdb_id() {
        let imdb = parse_detail_response(TV_EXTERNAL_IDS_JSON);
        assert_eq!(imdb.as_deref(), Some("tt0944947"));
    }

    #[test]
    fn parse_detail_no_imdb_returns_none() {
        let imdb = parse_detail_response(NO_IMDB_JSON);
        assert!(imdb.is_none());
    }

    // ── build_search_url ───────────────────────────────────────────────────────

    #[test]
    fn build_movie_url_without_year() {
        let url = build_search_url("key123", "Spirited Away", None, "movie");
        assert!(url.contains("/search/movie"));
        assert!(url.contains("query=Spirited%20Away"));
        assert!(url.contains("api_key=key123"));
        assert!(!url.contains("year="));
    }

    #[test]
    fn build_movie_url_with_year() {
        let url = build_search_url("key123", "Breaking Bad", Some(2008), "movie");
        assert!(url.contains("year=2008"));
    }

    #[test]
    fn build_tv_url_uses_first_air_date_year() {
        let url = build_search_url("key123", "One Piece", Some(1999), "episode");
        assert!(url.contains("/search/tv"));
        assert!(url.contains("first_air_date_year=1999"));
    }

    // ── lookup_tmdb returns None without API key ───────────────────────────────

    #[test]
    fn lookup_tmdb_no_key_returns_none() {
        // Ensure env var is unset for this test
        std::env::remove_var("TMDB_API_KEY");
        let result = lookup_tmdb("Test Movie", None, "movie");
        assert!(result.is_none());
    }
}
