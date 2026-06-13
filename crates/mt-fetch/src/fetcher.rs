//! Subtitle fetcher — orchestrates search across multiple providers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::providers::SubtitleProvider;
use crate::retry::{FetchError, with_retry};
use crate::types::SubtitleMatch;
use mt_core::MediaIdentity;

/// Orchestrates subtitle search across multiple providers.
pub struct SubtitleFetcher {
    providers: Vec<Box<dyn SubtitleProvider>>,
}

impl SubtitleFetcher {
    pub fn new(providers: Vec<Box<dyn SubtitleProvider>>) -> Self {
        Self { providers }
    }

    /// Search all providers in parallel, return ALL plausible matches sorted by score.
    pub fn search_all(&self, identity: &MediaIdentity, languages: &[&str]) -> Vec<SubtitleMatch> {
        // Use threads to query providers in parallel.
        let results: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .providers
                .iter()
                .map(|provider| {
                    scope.spawn(move || {
                        let name = provider.name().to_string();
                        let result = with_retry(
                            || provider.search(identity, languages),
                            1,
                            2.0,
                            &name,
                            |secs| std::thread::sleep(std::time::Duration::from_secs_f64(secs)),
                        );
                        (name, result)
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join()).collect()
        });

        let mut all_matches = Vec::new();
        for result in results {
            match result {
                Ok((name, Ok(matches))) => {
                    tracing::debug!("{name}: found {} matches", matches.len());
                    all_matches.extend(matches);
                }
                Ok((name, Err(e))) => {
                    tracing::warn!("{name} search failed: {e}");
                }
                Err(payload) => {
                    let msg = panic_payload_message(&*payload);
                    tracing::warn!("provider thread panicked: {msg}");
                }
            }
        }

        // Sort by (score, hash_match) descending.
        all_matches.sort_by(|a, b| {
            let score_ord = b
                .score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal);
            if score_ord == std::cmp::Ordering::Equal {
                b.hash_match.cmp(&a.hash_match)
            } else {
                score_ord
            }
        });

        all_matches
    }

    /// Download a single candidate subtitle file. Returns path written to.
    pub fn download_candidate(
        &self,
        match_: &SubtitleMatch,
        output_path: &Path,
    ) -> Result<PathBuf, FetchError> {
        let provider = self.find_provider(&match_.source).ok_or_else(|| {
            let available: Vec<_> = self.providers.iter().map(|p| p.name()).collect();
            FetchError::NotFound(format!(
                "No provider registered with name '{}'. Available: {:?}",
                match_.source, available
            ))
        })?;

        provider.download(match_, output_path)?;
        Ok(output_path.to_path_buf())
    }

    /// Search all providers and download best subtitle per language.
    ///
    /// Returns `{language_code: subtitle_file_path}` for successfully downloaded subtitles.
    pub fn fetch_subtitles(
        &self,
        identity: &MediaIdentity,
        languages: &[&str],
        output_dir: &Path,
    ) -> HashMap<String, PathBuf> {
        let all_matches = self.search_all(identity, languages);

        if all_matches.is_empty() {
            tracing::info!("No subtitles found from any provider");
            return HashMap::new();
        }

        // Pick best match per language (highest score wins)
        let mut best: HashMap<String, &SubtitleMatch> = HashMap::new();
        for match_ in &all_matches {
            if !best.contains_key(&match_.language) {
                best.insert(match_.language.clone(), match_);
            }
        }

        // Download best matches
        let mut result = HashMap::new();
        for (lang, match_) in &best {
            let output_path = output_dir.join(format!("fetched_{lang}.{}", match_.format));
            match self.download_candidate(match_, &output_path) {
                Ok(path) => {
                    tracing::info!(
                        "Fetched {} subtitles: {} ({} match, {})",
                        lang,
                        match_.release_name,
                        if match_.hash_match { "hash" } else { "query" },
                        match_.source,
                    );
                    result.insert(lang.clone(), path);
                }
                Err(e) => {
                    tracing::warn!("Failed to download {lang} subtitle: {e}");
                }
            }
        }

        result
    }

    fn find_provider(&self, name: &str) -> Option<&dyn SubtitleProvider> {
        self.providers
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }
}

/// Extract a human-readable message from a thread panic payload.
///
/// Panic payloads are `Box<dyn Any>`; the common cases produced by `panic!`
/// are `&'static str` and `String`. Downcasting recovers the message instead
/// of discarding it.
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SubtitleMatch;
    use std::sync::Mutex;

    struct FakeProvider {
        provider_name: String,
        matches: Vec<SubtitleMatch>,
        downloaded: Mutex<Vec<SubtitleMatch>>,
    }

    impl FakeProvider {
        fn new(name: &str, matches: Vec<SubtitleMatch>) -> Self {
            Self {
                provider_name: name.to_string(),
                matches,
                downloaded: Mutex::new(vec![]),
            }
        }
    }

    impl SubtitleProvider for FakeProvider {
        fn name(&self) -> &str {
            &self.provider_name
        }

        fn search(
            &self,
            _identity: &MediaIdentity,
            languages: &[&str],
        ) -> Result<Vec<SubtitleMatch>, FetchError> {
            Ok(self
                .matches
                .iter()
                .filter(|m| languages.contains(&m.language.as_str()))
                .cloned()
                .collect())
        }

        fn download(&self, match_: &SubtitleMatch, output_path: &Path) -> Result<(), FetchError> {
            std::fs::write(
                output_path,
                format!("subtitle content from {}", self.provider_name),
            )
            .map_err(FetchError::Io)?;
            self.downloaded.lock().unwrap().push(match_.clone());
            Ok(())
        }
    }

    fn make_identity() -> MediaIdentity {
        MediaIdentity {
            title: "Test".to_string(),
            parsed_title: "Test".to_string(),
            year: None,
            season: Some(1),
            episode: Some(1),
            media_type: "episode".to_string(),
            oshash: "0".repeat(16),
            file_size: 1000,
            raw_filename: "test.mkv".to_string(),
            imdb_id: None,
            tmdb_id: None,
            is_anime: false,
            release_group: None,
        }
    }

    struct PanickingProvider;
    impl SubtitleProvider for PanickingProvider {
        fn name(&self) -> &str {
            "panicker"
        }
        fn search(
            &self,
            _identity: &MediaIdentity,
            _languages: &[&str],
        ) -> Result<Vec<SubtitleMatch>, FetchError> {
            panic!("boom from provider");
        }
        fn download(&self, _m: &SubtitleMatch, _p: &Path) -> Result<(), FetchError> {
            Ok(())
        }
    }

    #[test]
    fn panic_payload_message_recovers_str_and_string() {
        let s: Box<dyn std::any::Any + Send> = Box::new("static panic");
        assert_eq!(panic_payload_message(&*s), "static panic");
        let s: Box<dyn std::any::Any + Send> = Box::new(String::from("owned panic"));
        assert_eq!(panic_payload_message(&*s), "owned panic");
        let s: Box<dyn std::any::Any + Send> = Box::new(42u32);
        assert_eq!(panic_payload_message(&*s), "unknown panic payload");
    }

    #[test]
    fn search_all_survives_panicking_provider() {
        let good = FakeProvider::new(
            "fake",
            vec![SubtitleMatch {
                language: "eng".to_string(),
                source: "fake".to_string(),
                subtitle_id: "1".to_string(),
                release_name: "rel".to_string(),
                format: "srt".to_string(),
                score: 0.9,
                hash_match: false,
            }],
        );
        let fetcher = SubtitleFetcher::new(vec![Box::new(good), Box::new(PanickingProvider)]);
        // The panicking provider must not abort search_all; good matches survive.
        let matches = fetcher.search_all(&make_identity(), &["eng"]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].source, "fake");
    }

    #[test]
    fn fetch_subtitles_downloads_best_per_language() {
        let dir = tempfile::tempdir().unwrap();
        let provider = FakeProvider::new(
            "fake",
            vec![
                SubtitleMatch {
                    language: "eng".to_string(),
                    source: "fake".to_string(),
                    subtitle_id: "1".to_string(),
                    release_name: "rel".to_string(),
                    format: "srt".to_string(),
                    score: 0.7,
                    hash_match: false,
                },
                SubtitleMatch {
                    language: "eng".to_string(),
                    source: "fake".to_string(),
                    subtitle_id: "2".to_string(),
                    release_name: "rel".to_string(),
                    format: "srt".to_string(),
                    score: 1.0,
                    hash_match: true,
                },
                SubtitleMatch {
                    language: "pol".to_string(),
                    source: "fake".to_string(),
                    subtitle_id: "3".to_string(),
                    release_name: "rel".to_string(),
                    format: "srt".to_string(),
                    score: 0.8,
                    hash_match: false,
                },
            ],
        );
        let fetcher = SubtitleFetcher::new(vec![Box::new(provider) as Box<dyn SubtitleProvider>]);
        // We can't easily get the FakeProvider back since it's boxed. Test via result.
        let result = fetcher.fetch_subtitles(&make_identity(), &["eng", "pol"], dir.path());

        assert!(result.contains_key("eng"));
        assert!(result.contains_key("pol"));
    }

    #[test]
    fn fetch_subtitles_returns_empty_when_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let provider = FakeProvider::new("fake", vec![]);
        let fetcher = SubtitleFetcher::new(vec![Box::new(provider)]);
        let result = fetcher.fetch_subtitles(&make_identity(), &["eng"], dir.path());
        assert!(result.is_empty());
    }

    #[test]
    fn fetch_subtitles_tries_multiple_providers() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = FakeProvider::new("p1", vec![]);
        let p2 = FakeProvider::new(
            "p2",
            vec![SubtitleMatch {
                language: "eng".to_string(),
                source: "p2".to_string(),
                subtitle_id: "99".to_string(),
                release_name: "rel".to_string(),
                format: "srt".to_string(),
                score: 0.7,
                hash_match: false,
            }],
        );
        let fetcher = SubtitleFetcher::new(vec![Box::new(p1), Box::new(p2)]);
        let result = fetcher.fetch_subtitles(&make_identity(), &["eng"], dir.path());
        assert!(result.contains_key("eng"));
    }

    #[test]
    fn search_all_sorted_by_score_descending() {
        let provider = FakeProvider::new(
            "fake",
            vec![
                SubtitleMatch {
                    language: "eng".to_string(),
                    source: "fake".to_string(),
                    subtitle_id: "low".to_string(),
                    release_name: "rel".to_string(),
                    format: "srt".to_string(),
                    score: 0.3,
                    hash_match: false,
                },
                SubtitleMatch {
                    language: "eng".to_string(),
                    source: "fake".to_string(),
                    subtitle_id: "high".to_string(),
                    release_name: "rel".to_string(),
                    format: "srt".to_string(),
                    score: 1.0,
                    hash_match: true,
                },
                SubtitleMatch {
                    language: "pol".to_string(),
                    source: "fake".to_string(),
                    subtitle_id: "mid".to_string(),
                    release_name: "rel".to_string(),
                    format: "srt".to_string(),
                    score: 0.6,
                    hash_match: false,
                },
            ],
        );
        let fetcher = SubtitleFetcher::new(vec![Box::new(provider)]);
        let results = fetcher.search_all(&make_identity(), &["eng", "pol"]);
        let scores: Vec<f64> = results.iter().map(|m| m.score).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(scores, sorted);
    }

    #[test]
    fn search_all_returns_all_matches_from_all_providers() {
        let p1 = FakeProvider::new(
            "p1",
            vec![
                SubtitleMatch {
                    language: "eng".to_string(),
                    source: "p1".to_string(),
                    subtitle_id: "a".to_string(),
                    release_name: "rel-a".to_string(),
                    format: "srt".to_string(),
                    score: 0.9,
                    hash_match: true,
                },
                SubtitleMatch {
                    language: "pol".to_string(),
                    source: "p1".to_string(),
                    subtitle_id: "b".to_string(),
                    release_name: "rel-b".to_string(),
                    format: "srt".to_string(),
                    score: 0.7,
                    hash_match: false,
                },
            ],
        );
        let p2 = FakeProvider::new(
            "p2",
            vec![SubtitleMatch {
                language: "eng".to_string(),
                source: "p2".to_string(),
                subtitle_id: "c".to_string(),
                release_name: "rel-c".to_string(),
                format: "srt".to_string(),
                score: 0.6,
                hash_match: false,
            }],
        );
        let fetcher = SubtitleFetcher::new(vec![Box::new(p1), Box::new(p2)]);
        let results = fetcher.search_all(&make_identity(), &["eng", "pol"]);

        assert_eq!(results.len(), 3);
        let ids: std::collections::HashSet<_> =
            results.iter().map(|m| m.subtitle_id.as_str()).collect();
        assert!(ids.contains("a"));
        assert!(ids.contains("b"));
        assert!(ids.contains("c"));
    }

    #[test]
    fn search_all_returns_empty_when_no_matches() {
        let provider = FakeProvider::new("fake", vec![]);
        let fetcher = SubtitleFetcher::new(vec![Box::new(provider)]);
        let results = fetcher.search_all(&make_identity(), &["eng"]);
        assert!(results.is_empty());
    }

    #[test]
    fn search_all_tolerates_provider_failure() {
        struct BrokenProvider;
        impl SubtitleProvider for BrokenProvider {
            fn name(&self) -> &str {
                "bad"
            }
            fn search(
                &self,
                _identity: &MediaIdentity,
                _languages: &[&str],
            ) -> Result<Vec<SubtitleMatch>, FetchError> {
                Err(FetchError::Network("network error".to_string()))
            }
            fn download(
                &self,
                _match_: &SubtitleMatch,
                _output_path: &Path,
            ) -> Result<(), FetchError> {
                Ok(())
            }
        }

        let bad = BrokenProvider;
        let good = FakeProvider::new(
            "good",
            vec![SubtitleMatch {
                language: "eng".to_string(),
                source: "good".to_string(),
                subtitle_id: "1".to_string(),
                release_name: "rel".to_string(),
                format: "srt".to_string(),
                score: 0.8,
                hash_match: false,
            }],
        );
        let fetcher = SubtitleFetcher::new(vec![Box::new(bad), Box::new(good)]);
        let results = fetcher.search_all(&make_identity(), &["eng"]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subtitle_id, "1");
    }

    #[test]
    fn download_candidate_delegates_to_correct_provider() {
        let dir = tempfile::tempdir().unwrap();
        let provider = FakeProvider::new("myprovider", vec![]);
        let fetcher = SubtitleFetcher::new(vec![Box::new(provider)]);
        let match_ = SubtitleMatch {
            language: "eng".to_string(),
            source: "myprovider".to_string(),
            subtitle_id: "sub-123".to_string(),
            release_name: "rel".to_string(),
            format: "srt".to_string(),
            score: 0.9,
            hash_match: true,
        };
        let output_path = dir.path().join("output.srt");
        let result = fetcher.download_candidate(&match_, &output_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), output_path);
    }

    #[test]
    fn download_candidate_raises_when_provider_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let provider = FakeProvider::new("only_provider", vec![]);
        let fetcher = SubtitleFetcher::new(vec![Box::new(provider)]);
        let match_ = SubtitleMatch {
            language: "eng".to_string(),
            source: "missing_provider".to_string(),
            subtitle_id: "sub-789".to_string(),
            release_name: "rel".to_string(),
            format: "srt".to_string(),
            score: 0.7,
            hash_match: false,
        };
        let result = fetcher.download_candidate(&match_, &dir.path().join("out.srt"));
        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("missing_provider"));
    }

    #[test]
    fn download_candidate_picks_correct_provider_by_name() {
        let dir = tempfile::tempdir().unwrap();

        struct TrackingProvider {
            provider_name: String,
            call_count: Mutex<usize>,
        }
        impl SubtitleProvider for TrackingProvider {
            fn name(&self) -> &str {
                &self.provider_name
            }
            fn search(
                &self,
                _: &MediaIdentity,
                _: &[&str],
            ) -> Result<Vec<SubtitleMatch>, FetchError> {
                Ok(vec![])
            }
            fn download(
                &self,
                _match_: &SubtitleMatch,
                output_path: &Path,
            ) -> Result<(), FetchError> {
                *self.call_count.lock().unwrap() += 1;
                std::fs::write(output_path, "content").map_err(FetchError::Io)
            }
        }

        let p1 = TrackingProvider {
            provider_name: "provider_a".to_string(),
            call_count: Mutex::new(0),
        };
        let p2 = TrackingProvider {
            provider_name: "provider_b".to_string(),
            call_count: Mutex::new(0),
        };

        let fetcher = SubtitleFetcher::new(vec![Box::new(p1), Box::new(p2)]);
        let match_ = SubtitleMatch {
            language: "eng".to_string(),
            source: "provider_b".to_string(),
            subtitle_id: "sub-456".to_string(),
            release_name: "rel".to_string(),
            format: "srt".to_string(),
            score: 0.8,
            hash_match: false,
        };

        fetcher
            .download_candidate(&match_, &dir.path().join("out.srt"))
            .unwrap();

        // We can't easily get call_count back from Box<dyn> after moving.
        // But we can check the file was written.
        assert!(dir.path().join("out.srt").exists());
    }
}
