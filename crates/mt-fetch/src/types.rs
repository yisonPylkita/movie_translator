//! Subtitle fetch types.
//!
//! Ported from `movie_translator/subtitle_fetch/types.py`.

/// A candidate subtitle match returned by a provider's `search` method.
///
/// Maps directly to the Python `SubtitleMatch` NamedTuple.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleMatch {
    /// ISO 639-2B language code (e.g., `"eng"`, `"pol"`).
    pub language: String,
    /// Provider name (e.g., `"opensubtitles"`, `"podnapisi"`).
    pub source: String,
    /// Provider-specific subtitle identifier.
    pub subtitle_id: String,
    /// Human-readable release name as reported by the provider.
    pub release_name: String,
    /// File format: `"srt"`, `"ass"`, `"sub"`, etc.
    pub format: String,
    /// Match confidence score in `[0.0, 1.0]`.
    pub score: f64,
    /// `true` if the match was found by file hash (more accurate).
    pub hash_match: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtitle_match_fields() {
        let m = SubtitleMatch {
            language: "pol".to_string(),
            source: "opensubtitles".to_string(),
            subtitle_id: "12345".to_string(),
            release_name: "Test.S01E01.mkv".to_string(),
            format: "srt".to_string(),
            score: 0.95,
            hash_match: true,
        };
        assert_eq!(m.language, "pol");
        assert_eq!(m.source, "opensubtitles");
        assert_eq!(m.subtitle_id, "12345");
        assert_eq!(m.format, "srt");
        assert!((m.score - 0.95).abs() < 1e-9);
        assert!(m.hash_match);
    }

    #[test]
    fn subtitle_match_clone_eq() {
        let m = SubtitleMatch {
            language: "eng".to_string(),
            source: "podnapisi".to_string(),
            subtitle_id: "abc".to_string(),
            release_name: "rel".to_string(),
            format: "ass".to_string(),
            score: 0.7,
            hash_match: false,
        };
        assert_eq!(m.clone(), m);
    }
}
