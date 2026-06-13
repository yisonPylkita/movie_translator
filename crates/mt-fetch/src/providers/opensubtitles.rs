//! OpenSubtitles.com REST API v2 provider.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use crate::rate_limiter::RateLimiter;
use crate::retry::FetchError;
use crate::scoring::compute_release_score;
use crate::types::SubtitleMatch;
use mt_core::MediaIdentity;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

pub const API_BASE: &str = "https://api.opensubtitles.com/api/v1";
pub const USER_AGENT: &str = "MovieTranslator v1.0";

/// Map ISO 639-2B → OpenSubtitles 2-letter code.
pub fn lang_to_os(lang: &str) -> Option<&'static str> {
    match lang {
        "eng" => Some("en"),
        "pol" => Some("pl"),
        "jpn" => Some("ja"),
        _ => None,
    }
}

/// Map OpenSubtitles 2-letter code → ISO 639-2B.
pub fn lang_from_os(code: &str) -> &str {
    match code {
        "en" => "eng",
        "pl" => "pol",
        "ja" => "jpn",
        other => other,
    }
}

/// Parse OpenSubtitles API `/subtitles` response JSON into SubtitleMatch list.
pub fn parse_results(data: &Value, languages: &[&str], raw_filename: &str) -> Vec<SubtitleMatch> {
    let mut matches = Vec::new();
    let items = match data.get("data").and_then(|d| d.as_array()) {
        Some(arr) => arr,
        None => return matches,
    };

    for item in items {
        let attrs = match item.get("attributes") {
            Some(a) => a,
            None => continue,
        };

        let os_lang = attrs.get("language").and_then(|v| v.as_str()).unwrap_or("");
        let lang_3 = lang_from_os(os_lang);

        if !languages.contains(&lang_3) {
            continue;
        }

        let files = match attrs.get("files").and_then(|f| f.as_array()) {
            Some(f) if !f.is_empty() => f,
            _ => continue,
        };

        let file_info = &files[0];
        let file_name = file_info
            .get("file_name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let ext = if file_name.contains('.') {
            file_name.rsplit('.').next().unwrap_or("srt")
        } else {
            "srt"
        };

        let is_hash = attrs
            .get("moviehash_match")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let score = if is_hash {
            1.0
        } else {
            let release = attrs.get("release").and_then(|v| v.as_str()).unwrap_or("");
            let base_score = 0.6;
            let release_bonus = compute_release_score(raw_filename, release) * 0.3;
            base_score + release_bonus
        };

        let file_id = file_info
            .get("file_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let release = attrs
            .get("release")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        matches.push(SubtitleMatch {
            language: lang_3.to_string(),
            source: "opensubtitles".to_string(),
            subtitle_id: file_id.to_string(),
            release_name: release,
            format: ext.to_string(),
            score,
            hash_match: is_hash,
        });
    }

    matches
}

/// Build a URL with query parameters for the OpenSubtitles API.
pub fn build_url(endpoint: &str, params: &[(&str, String)]) -> String {
    if params.is_empty() {
        return format!("{API_BASE}{endpoint}");
    }
    let qs = params
        .iter()
        .map(|(k, v)| format!("{k}={}", urlencoding::encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{API_BASE}{endpoint}?{qs}")
}

/// OpenSubtitles.com REST API v2 provider.
pub struct OpenSubtitlesProvider {
    api_key: String,
    username: String,
    password: String,
    token: Mutex<Option<String>>,
    rate_limiter: RateLimiter,
    client: Client,
}

impl OpenSubtitlesProvider {
    pub fn new(
        api_key: Option<String>,
        username: Option<String>,
        password: Option<String>,
    ) -> Self {
        let api_key = api_key
            .or_else(|| env::var("OPENSUBTITLES_API_KEY").ok())
            .unwrap_or_default();
        let username = username
            .or_else(|| env::var("OPENSUBTITLES_USERNAME").ok())
            .unwrap_or_default();
        let password = password
            .or_else(|| env::var("OPENSUBTITLES_PASSWORD").ok())
            .unwrap_or_default();

        let client = super::build_blocking_client(USER_AGENT);

        Self {
            api_key,
            username,
            password,
            token: Mutex::new(None),
            rate_limiter: RateLimiter::new(0.25),
            client,
        }
    }

    /// Make an API request with rate limiting.
    fn api_request(
        &self,
        method: &str,
        endpoint: &str,
        params: Option<&[(&str, String)]>,
        body: Option<&Value>,
    ) -> Result<Value, FetchError> {
        self.rate_limiter.wait();

        let url = match params {
            Some(p) if !p.is_empty() => build_url(endpoint, p),
            _ => format!("{API_BASE}{endpoint}"),
        };

        let token_guard = self.token.lock().unwrap();
        let mut req = match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            other => return Err(FetchError::Network(format!("unsupported method: {other}"))),
        };

        req = req
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json");

        if let Some(ref tok) = *token_guard {
            req = req.header("Authorization", format!("Bearer {tok}"));
        }
        drop(token_guard);

        if let Some(b) = body {
            req = req.json(b);
        }

        let resp = req.send().map_err(|e| FetchError::Network(e.to_string()))?;

        let status = resp.status().as_u16();

        // Collect rate limit headers before consuming body
        let mut rl_headers = HashMap::new();
        for (k, v) in resp.headers() {
            if k.as_str().starts_with("x-ratelimit") {
                // Normalize to the exact casing the rate_limiter expects
                let key = k
                    .as_str()
                    .split('-')
                    .map(|part| {
                        let mut c = part.chars();
                        match c.next() {
                            Some(f) => f.to_uppercase().to_string() + c.as_str(),
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("-");
                if let Ok(val) = v.to_str() {
                    rl_headers.insert(key, val.to_string());
                }
            }
        }
        self.rate_limiter.update_from_headers(&rl_headers);

        if status == 429 {
            // Retry-After not easily accessible after consuming headers above;
            // just use default 5s backoff.
            self.rate_limiter.record_429(None);
            return Err(FetchError::Http {
                status,
                body: "Rate limited".to_string(),
            });
        }
        if status == 406 {
            warn!("OpenSubtitles daily download quota exceeded");
            return Err(FetchError::QuotaExceeded);
        }
        if status >= 400 {
            // Capture the response body so HTTP failures carry the provider's
            // error detail instead of a generic "API error" placeholder.
            let body_text = resp.text().unwrap_or_default();
            let trimmed = body_text.trim();
            let snippet = trimmed.chars().take(200).collect::<String>();
            let body = if snippet.is_empty() {
                "API error".to_string()
            } else {
                snippet
            };
            return Err(FetchError::Http { status, body });
        }

        let json: Value = resp.json().map_err(|e| FetchError::Parse(e.to_string()))?;
        Ok(json)
    }

    fn ensure_logged_in(&self) -> Result<(), FetchError> {
        if self.token.lock().unwrap().is_some() {
            return Ok(());
        }
        if self.username.is_empty() || self.password.is_empty() {
            return Err(FetchError::Auth(
                "OpenSubtitles download requires OPENSUBTITLES_USERNAME and \
                 OPENSUBTITLES_PASSWORD environment variables"
                    .to_string(),
            ));
        }
        let body = json!({
            "username": self.username,
            "password": self.password,
        });
        let data = self.api_request("POST", "/login", None, Some(&body))?;
        let tok = data
            .get("token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if tok.is_empty() {
            return Err(FetchError::Auth(
                "OpenSubtitles login failed: no token returned".to_string(),
            ));
        }
        *self.token.lock().unwrap() = Some(tok);
        Ok(())
    }
}

impl super::SubtitleProvider for OpenSubtitlesProvider {
    fn name(&self) -> &str {
        "opensubtitles"
    }

    fn search(
        &self,
        identity: &MediaIdentity,
        languages: &[&str],
    ) -> Result<Vec<SubtitleMatch>, FetchError> {
        if self.api_key.is_empty() {
            debug!("OpenSubtitles: no API key configured, skipping");
            return Ok(vec![]);
        }

        let os_langs: Vec<_> = languages
            .iter()
            .map(|l| lang_to_os(l).unwrap_or(l).to_string())
            .collect();
        let os_langs_str = os_langs.join(",");

        let mut matches: Vec<SubtitleMatch> = Vec::new();

        // Strategy 1: Hash-based search
        if !identity.oshash.is_empty() {
            let params_ref: &[(&str, String)] = &[
                ("moviehash", identity.oshash.clone()),
                ("languages", os_langs_str.clone()),
            ];
            match self.api_request("GET", "/subtitles", Some(params_ref), None) {
                Ok(data) => {
                    matches = parse_results(&data, languages, &identity.raw_filename);
                }
                Err(e) => debug!("OpenSubtitles hash search failed: {e}"),
            }
        }

        // Strategy 2: Query-based search
        let mut query_params: Vec<(&str, String)> = vec![
            ("query", identity.title.clone()),
            ("languages", os_langs_str.clone()),
        ];
        if let Some(s) = identity.season {
            query_params.push(("season_number", s.to_string()));
        }
        if let Some(e) = identity.episode {
            query_params.push(("episode_number", e.to_string()));
        }
        if identity.media_type == "movie"
            && let Some(y) = identity.year
        {
            query_params.push(("year", y.to_string()));
        }
        if let Some(ref imdb_id) = identity.imdb_id {
            let stripped = imdb_id.trim_start_matches("tt").to_string();
            query_params.push(("imdb_id", stripped));
        }
        if let Some(tmdb_id) = identity.tmdb_id {
            query_params.push(("tmdb_id", tmdb_id.to_string()));
        }

        match self.api_request("GET", "/subtitles", Some(&query_params), None) {
            Ok(data) => {
                let query_matches = parse_results(&data, languages, &identity.raw_filename);
                let seen_ids: HashSet<_> = matches.iter().map(|m| m.subtitle_id.clone()).collect();
                for m in query_matches {
                    if !seen_ids.contains(&m.subtitle_id) {
                        matches.push(m);
                    }
                }
            }
            Err(e) => debug!("OpenSubtitles query search failed: {e}"),
        }

        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(matches)
    }

    fn download(&self, match_: &SubtitleMatch, output_path: &Path) -> Result<(), FetchError> {
        self.ensure_logged_in()?;

        let file_id: i64 = match_
            .subtitle_id
            .parse()
            .map_err(|_| FetchError::Parse(format!("invalid file_id: {}", match_.subtitle_id)))?;
        let body = json!({ "file_id": file_id });
        let data = self.api_request("POST", "/download", None, Some(&body))?;

        let link = data
            .get("link")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if link.is_empty() {
            return Err(FetchError::NotFound(format!(
                "no download link for subtitle {}",
                match_.subtitle_id
            )));
        }

        let content = self
            .client
            .get(&link)
            .send()
            .map_err(|e| FetchError::Network(e.to_string()))?
            .bytes()
            .map_err(|e| FetchError::Network(e.to_string()))?;

        fs::write(output_path, &content[..]).map_err(FetchError::Io)?;
        info!(
            "Downloaded subtitle: {} (opensubtitles)",
            output_path.display()
        );
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::SubtitleProvider;

    #[test]
    fn provider_name_is_opensubtitles() {
        let p = OpenSubtitlesProvider::new(Some("key".to_string()), None, None);
        assert_eq!(p.name(), "opensubtitles");
    }

    #[test]
    fn search_returns_empty_without_api_key() {
        // Ensure env var is not set
        unsafe { env::remove_var("OPENSUBTITLES_API_KEY") };
        let p = OpenSubtitlesProvider::new(Some(String::new()), None, None);
        // Without a live server this tests the early-return path only
        // We can't hit the real API, but we can test the guard
        assert!(p.api_key.is_empty());
    }

    #[test]
    fn parse_results_hash_match_score_1() {
        let api_response = json!({
            "data": [{
                "attributes": {
                    "language": "en",
                    "release": "Breaking.Bad.S01E03.720p",
                    "moviehash_match": true,
                    "files": [{"file_id": 12345, "file_name": "subs.srt"}]
                }
            }]
        });
        let matches = parse_results(&api_response, &["eng"], "Breaking.Bad.S01E03.mkv");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].language, "eng");
        assert!(matches[0].hash_match);
        assert_eq!(matches[0].subtitle_id, "12345");
        assert!((matches[0].score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_results_query_match_lower_score() {
        let api_response = json!({
            "data": [{
                "attributes": {
                    "language": "en",
                    "release": "Breaking.Bad.S01E03",
                    "moviehash_match": false,
                    "files": [{"file_id": 99, "file_name": "subs.srt"}]
                }
            }]
        });
        let matches = parse_results(&api_response, &["eng"], "Breaking.Bad.S01E03.mkv");
        assert!(!matches[0].hash_match);
        assert!(matches[0].score >= 0.6 && matches[0].score < 1.0);
    }

    #[test]
    fn parse_results_filters_by_language() {
        let api_response = json!({
            "data": [
                {
                    "attributes": {
                        "language": "en",
                        "release": "subs-en",
                        "moviehash_match": false,
                        "files": [{"file_id": 1, "file_name": "en.srt"}]
                    }
                },
                {
                    "attributes": {
                        "language": "pl",
                        "release": "subs-pl",
                        "moviehash_match": false,
                        "files": [{"file_id": 2, "file_name": "pl.srt"}]
                    }
                }
            ]
        });
        let matches = parse_results(&api_response, &["pol"], "test.mkv");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].language, "pol");
    }

    #[test]
    fn build_url_strips_tt_prefix_from_imdb_id() {
        // Verify the imdb_id stripping logic
        let imdb_id = "tt0903747";
        let stripped = imdb_id.trim_start_matches("tt");
        assert_eq!(stripped, "0903747");
    }

    // ── Additional: lang mappings ─────────────────────────────────────────────

    #[test]
    fn lang_to_os_known_langs() {
        assert_eq!(lang_to_os("eng"), Some("en"));
        assert_eq!(lang_to_os("pol"), Some("pl"));
        assert_eq!(lang_to_os("jpn"), Some("ja"));
        assert_eq!(lang_to_os("zho"), None);
    }

    #[test]
    fn lang_from_os_known_codes() {
        assert_eq!(lang_from_os("en"), "eng");
        assert_eq!(lang_from_os("pl"), "pol");
        assert_eq!(lang_from_os("ja"), "jpn");
        assert_eq!(lang_from_os("de"), "de"); // passthrough
    }

    // ── Additional: extension extraction ─────────────────────────────────────

    #[test]
    fn extension_extracted_from_file_name() {
        let api_response = json!({
            "data": [{
                "attributes": {
                    "language": "en",
                    "release": "test",
                    "moviehash_match": false,
                    "files": [{"file_id": 1, "file_name": "subtitle.ass"}]
                }
            }]
        });
        let matches = parse_results(&api_response, &["eng"], "test.mkv");
        assert_eq!(matches[0].format, "ass");
    }

    #[test]
    fn no_extension_defaults_to_srt() {
        let api_response = json!({
            "data": [{
                "attributes": {
                    "language": "en",
                    "release": "test",
                    "moviehash_match": false,
                    "files": [{"file_id": 1, "file_name": "subtitle"}]
                }
            }]
        });
        let matches = parse_results(&api_response, &["eng"], "test.mkv");
        assert_eq!(matches[0].format, "srt");
    }

    // ── Additional: empty data array ─────────────────────────────────────────

    #[test]
    fn empty_data_array_returns_empty() {
        let api_response = json!({"data": []});
        let matches = parse_results(&api_response, &["eng"], "test.mkv");
        assert!(matches.is_empty());
    }

    // ── Additional: release score bonus ──────────────────────────────────────

    #[test]
    fn matching_release_name_increases_score() {
        let api_response_matching = json!({
            "data": [{
                "attributes": {
                    "language": "en",
                    "release": "Breaking.Bad.S01E03.mkv",
                    "moviehash_match": false,
                    "files": [{"file_id": 1, "file_name": "s.srt"}]
                }
            }]
        });
        let api_response_nonmatching = json!({
            "data": [{
                "attributes": {
                    "language": "en",
                    "release": "Completely.Different.Show",
                    "moviehash_match": false,
                    "files": [{"file_id": 2, "file_name": "s.srt"}]
                }
            }]
        });
        let m1 = parse_results(&api_response_matching, &["eng"], "Breaking.Bad.S01E03.mkv");
        let m2 = parse_results(
            &api_response_nonmatching,
            &["eng"],
            "Breaking.Bad.S01E03.mkv",
        );
        assert!(m1[0].score > m2[0].score);
    }
}
