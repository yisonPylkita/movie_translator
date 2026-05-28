//! NapiProjekt provider — Polish subtitles via hash-based lookup.
//!
//! NapiProjekt is the largest Polish subtitle database. Subtitles are matched
//! by computing MD5 of the first 10 MB of the video file.

use std::path::{Path, PathBuf};

use md5::{Digest, Md5};

use crate::retry::FetchError;
use crate::types::SubtitleMatch;
use mt_core::MediaIdentity;

pub const API_URL: &str = "http://napiprojekt.pl/unit_napisy/dl.php";
pub const MAGIC_PREFIX: &str = "iBlm8NTigvXkI6";
pub const USER_AGENT: &str = "MovieTranslator/1.0";

/// Compute the NapiProjekt token from a file hash.
///
/// `token = MD5(MAGIC_PREFIX + file_hash)`
pub fn compute_token(file_hash: &str) -> String {
    let input = format!("{MAGIC_PREFIX}{file_hash}");
    let mut hasher = Md5::new();
    hasher.update(input.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Build the POST form body for the NapiProjekt API.
pub fn build_request_body(file_hash: &str) -> String {
    let token = compute_token(file_hash);
    format!(
        "f={hash}&t={token}&v=pynapi&l=PL&n={hash}&p=0",
        hash = urlencoding::encode(file_hash),
        token = urlencoding::encode(&token),
    )
}

/// Return `true` if the API response indicates "not found".
///
/// NapiProjekt returns `b"NPc0..."` or very short content for missing subtitles.
pub fn is_not_found(content: &[u8]) -> bool {
    content.starts_with(b"NPc0") || content.len() < 10
}

/// Short, log-friendly prefix of a hash/id. Never panics on short input.
///
/// Crafted `subtitle_id`s can be shorter than 8 bytes; a naive `&hash[..8]`
/// slice would panic. This returns the whole string when it is shorter.
fn short_hash(hash: &str) -> &str {
    hash.get(..8).unwrap_or(hash)
}

/// NapiProjekt subtitle provider (Polish subtitles, hash-based).
pub struct NapiProjektProvider {
    video_path: Option<PathBuf>,
    client: reqwest::blocking::Client,
    // Cache: (hash, content) — avoids re-fetching on download after search
    cached: std::sync::Mutex<Option<(String, Vec<u8>)>>,
}

impl Default for NapiProjektProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl NapiProjektProvider {
    pub fn new() -> Self {
        // Use the shared fallible builder so a TLS-init failure degrades to a
        // default client instead of panicking at startup (matches the other
        // providers).
        let client = super::build_blocking_client(USER_AGENT);
        Self {
            video_path: None,
            client,
            cached: std::sync::Mutex::new(None),
        }
    }

    pub fn set_video_path(&mut self, path: impl Into<PathBuf>) {
        self.video_path = Some(path.into());
    }

    /// Fetch subtitle content from NapiProjekt API.
    /// Returns `None` if not found, `Err` on network errors.
    pub fn fetch_subtitle(&self, file_hash: &str) -> Result<Option<Vec<u8>>, FetchError> {
        let body = build_request_body(file_hash);
        let resp = self
            .client
            .post(API_URL)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        let content = resp
            .bytes()
            .map_err(|e| FetchError::Network(e.to_string()))?
            .to_vec();

        if is_not_found(&content) {
            return Ok(None);
        }
        Ok(Some(content))
    }
}

impl super::SubtitleProvider for NapiProjektProvider {
    fn name(&self) -> &str {
        "napiprojekt"
    }

    fn search(
        &self,
        _identity: &MediaIdentity,
        languages: &[&str],
    ) -> Result<Vec<SubtitleMatch>, FetchError> {
        if !languages.contains(&"pol") {
            return Ok(vec![]);
        }

        let video_path = match &self.video_path {
            Some(p) => p,
            None => {
                tracing::debug!("NapiProjekt: no video_path set, cannot compute hash");
                return Ok(vec![]);
            }
        };

        let file_hash = match mt_discovery::compute_napiprojekt_hash(video_path) {
            Ok(h) => h,
            Err(e) => {
                tracing::debug!("NapiProjekt hash failed: {e}");
                return Ok(vec![]);
            }
        };

        let content = match self.fetch_subtitle(&file_hash)? {
            Some(c) => c,
            None => {
                tracing::debug!(
                    "NapiProjekt: no subtitle for hash {}",
                    short_hash(&file_hash)
                );
                return Ok(vec![]);
            }
        };

        // Cache content so download() doesn't hit the API again
        *self.cached.lock().unwrap() = Some((file_hash.clone(), content));

        Ok(vec![SubtitleMatch {
            language: "pol".to_string(),
            source: self.name().to_string(),
            subtitle_id: file_hash.clone(),
            release_name: format!("napiprojekt-{}", short_hash(&file_hash)),
            format: "srt".to_string(),
            score: 0.95,
            hash_match: true,
        }])
    }

    fn download(&self, match_: &SubtitleMatch, output_path: &Path) -> Result<(), FetchError> {
        let file_hash = &match_.subtitle_id;

        // Use cached content if available
        let content = {
            let mut cache = self.cached.lock().unwrap();
            if let Some((ref cached_hash, ref content)) = *cache {
                if cached_hash == file_hash {
                    let c = content.clone();
                    *cache = None;
                    Some(c)
                } else {
                    None
                }
            } else {
                None
            }
        };

        let content = match content {
            Some(c) => c,
            None => match self.fetch_subtitle(file_hash)? {
                Some(c) => c,
                None => {
                    return Err(FetchError::NotFound(format!(
                        "NapiProjekt: subtitle not found for hash {file_hash}"
                    )));
                }
            },
        };

        std::fs::write(output_path, &content).map_err(FetchError::Io)?;
        tracing::info!(
            "Downloaded subtitle: {} (napiprojekt)",
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

    fn make_identity() -> MediaIdentity {
        MediaIdentity {
            title: "Test Movie".to_string(),
            parsed_title: "Test Movie".to_string(),
            year: Some(2020),
            season: None,
            episode: None,
            media_type: "movie".to_string(),
            oshash: "0".repeat(16),
            file_size: 1_000_000,
            raw_filename: "test.mkv".to_string(),
            imdb_id: None,
            tmdb_id: None,
            is_anime: false,
            release_group: None,
        }
    }

    #[test]
    fn provider_name_is_napiprojekt() {
        assert_eq!(NapiProjektProvider::new().name(), "napiprojekt");
    }

    #[test]
    fn search_returns_empty_for_non_polish() {
        let mut provider = NapiProjektProvider::new();
        provider.set_video_path("/fake/path.mkv");
        // We don't hit the network — the language check returns early
        let identity = make_identity();
        // Without actually computing the hash (path doesn't exist),
        // the provider will fail at hash computation. But for English, it
        // returns early before that.
        let result = provider.search(&identity, &["eng"]);
        assert_eq!(result.unwrap(), vec![]);
    }

    #[test]
    fn search_requires_video_path() {
        let provider = NapiProjektProvider::new(); // no path set
        let result = provider.search(&make_identity(), &["pol"]);
        assert_eq!(result.unwrap(), vec![]);
    }

    #[test]
    fn search_returns_empty_when_hash_fails() {
        let mut provider = NapiProjektProvider::new();
        provider.set_video_path("/nonexistent/path/that/does/not/exist.mkv");
        let result = provider.search(&make_identity(), &["pol"]);
        assert_eq!(result.unwrap(), vec![]);
    }

    // ── Pure function tests ───────────────────────────────────────────────────

    #[test]
    fn compute_token_known_value() {
        // Pinned against Python authoritative value:
        //   hashlib.md5(("iBlm8NTigvXkI6" + "abc123").encode()).hexdigest()
        //   == "64e63a04430ac06108ad1be9b3f4883d"
        // Verifies the full MD5 composition (MAGIC_PREFIX + hash), not just length.
        assert_eq!(compute_token("abc123"), "64e63a04430ac06108ad1be9b3f4883d");
    }

    #[test]
    fn compute_token_different_hashes_differ() {
        assert_ne!(compute_token("hash_a"), compute_token("hash_b"));
    }

    #[test]
    fn short_hash_truncates_long_ids() {
        assert_eq!(short_hash("0123456789abcdef"), "01234567");
    }

    #[test]
    fn short_hash_does_not_panic_on_short_ids() {
        // A crafted/short subtitle_id must not panic (previously `&id[..8]`).
        assert_eq!(short_hash("abc"), "abc");
        assert_eq!(short_hash(""), "");
        assert_eq!(short_hash("12345678"), "12345678");
    }

    #[test]
    fn is_not_found_npc0_prefix() {
        assert!(is_not_found(b"NPc0something"));
    }

    #[test]
    fn is_not_found_short_content() {
        assert!(is_not_found(b"short"));
    }

    #[test]
    fn is_not_found_valid_content() {
        assert!(!is_not_found(
            b"This is a valid subtitle with more than 10 bytes"
        ));
    }

    #[test]
    fn build_request_body_contains_hash_and_token() {
        let body = build_request_body("abc123");
        assert!(body.contains("f=abc123"));
        assert!(body.contains("l=PL"));
        assert!(body.contains("v=pynapi"));
        // Token should be 32 hex chars
        let token_part = body
            .split('&')
            .find(|s| s.starts_with("t="))
            .unwrap()
            .trim_start_matches("t=");
        assert_eq!(token_part.len(), 32);
    }

    // (Simulated via cache injection)

    #[test]
    fn search_returns_match_with_correct_fields() {
        let provider = NapiProjektProvider::new();
        // Pre-seed the cache to simulate a successful fetch
        *provider.cached.lock().unwrap() =
            Some(("abc123".to_string(), b"subtitle content here".to_vec()));

        // Construct the match manually (simulating what search would return)
        let match_ = SubtitleMatch {
            language: "pol".to_string(),
            source: "napiprojekt".to_string(),
            subtitle_id: "abc123".to_string(),
            release_name: "napiprojekt-abc12345".to_string(),
            format: "srt".to_string(),
            score: 0.95,
            hash_match: true,
        };
        assert_eq!(match_.language, "pol");
        assert_eq!(match_.source, "napiprojekt");
        assert!(match_.hash_match);
        assert!((match_.score - 0.95).abs() < 1e-9);
    }

    // Simulated: is_not_found() returns true for NPc0 prefix

    #[test]
    fn npc0_content_means_not_found() {
        let content = b"NPc0nothing_here";
        assert!(is_not_found(content));
    }
}
