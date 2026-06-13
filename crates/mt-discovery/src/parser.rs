//! Filename parsing — pure-Rust replacement for the Python `guessit`/`aniparse`
//! based parser.  Uses `anitomy-pure` for anime filenames and a regex-based
//! fallback for conventional TV/movie filenames.  No Python PyO3 dependency.

use serde::{Deserialize, Serialize};

/// Parsed filename metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedName {
    /// Best-guess title from filename/anitomy/regex fallback.
    pub title: Option<String>,
    /// Release year, if detected.
    pub year: Option<i32>,
    /// Season number, if applicable.
    pub season: Option<i32>,
    /// Episode number, if applicable.
    pub episode: Option<i32>,
    /// `"movie"` or `"episode"`.
    pub media_type: String,
    /// `true` when anime signals (e.g. fansub group) were detected.
    pub is_anime: bool,
    /// Fansub / release group, e.g. `"HorribleSubs"`.
    pub release_group: Option<String>,
}

/// Parse a video filename using pure Rust logic.
///
/// Delegates to the `rust_parser` module which uses `anitomy-pure` for anime
/// bracket patterns and regex fallback for conventional TV/movie filenames.
///
/// Returns a `ParsedName` with all available fields filled in.  Missing fields
/// are `None`.  Never fails (returns a best-effort result for any input).
pub fn parse_filename(filename: &str, folder: Option<&str>) -> mt_core::Result<ParsedName> {
    Ok(crate::rust_parser::parse_filename(filename, folder))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit test: ParsedName JSON round-trip (no interpreter required).
    #[test]
    fn parsed_name_deserialize_sample() {
        let json = r#"{
            "title": "One Piece",
            "year": null,
            "season": null,
            "episode": 1000,
            "media_type": "episode",
            "is_anime": true,
            "release_group": "HorribleSubs"
        }"#;
        let parsed: ParsedName = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.title.as_deref(), Some("One Piece"));
        assert_eq!(parsed.episode, Some(1000));
        assert!(parsed.is_anime);
        assert_eq!(parsed.release_group.as_deref(), Some("HorribleSubs"));
        assert_eq!(parsed.media_type, "episode");
        assert!(parsed.year.is_none());
        assert!(parsed.season.is_none());
    }

    /// Unit test: ParsedName for a movie (all nullable fields absent).
    #[test]
    fn parsed_name_deserialize_movie() {
        let json = r#"{
            "title": "Spirited Away",
            "year": 2001,
            "season": null,
            "episode": null,
            "media_type": "movie",
            "is_anime": false,
            "release_group": null
        }"#;
        let parsed: ParsedName = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.title.as_deref(), Some("Spirited Away"));
        assert_eq!(parsed.year, Some(2001));
        assert!(!parsed.is_anime);
        assert!(parsed.release_group.is_none());
        assert_eq!(parsed.media_type, "movie");
    }

    /// Integration test: pure-Rust parser handles anime bracket filenames.
    /// No Python needed.
    #[test]
    fn integration_parse_anime_filename() {
        let result = parse_filename("[HorribleSubs] One Piece - 1000 [1080p].mkv", None).unwrap();
        assert!(
            result.title.as_deref().is_some_and(|t| !t.is_empty()),
            "expected non-empty parsed_title, got {:?}",
            result.title
        );
        assert!(
            result.is_anime,
            "expected is_anime=true for fansub filename"
        );
        assert_eq!(result.release_group.as_deref(), Some("HorribleSubs"));
        assert_eq!(result.episode, Some(1000));
    }
}
