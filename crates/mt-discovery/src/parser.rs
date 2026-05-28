//! Filename parsing via the embedded Python `movie_translator.identifier.parser`.
//!
//! The underlying logic relies on `guessit` and `aniparse` Python libraries
//! (no Rust equivalent), so we call them in-process through PyO3 via the
//! `mt_ml` crate. The first call initialises the embedded interpreter; every
//! call after that is a direct function call (no subprocess).

use mt_core::Result;
use serde::{Deserialize, Serialize};

/// Parsed filename metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedName {
    /// Best-guess title from filename/aniparse/guessit.
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

/// Parse a video filename through the embedded Python interpreter.
///
/// # Errors
/// Returns [`mt_core::MtError::Parse`] (with traceback) when the Python call
/// raises, or [`mt_core::MtError::PathResolution`] if the `movie_translator/`
/// package can't be located on `sys.path`.
pub fn parse_filename(filename: &str, folder: Option<&str>) -> Result<ParsedName> {
    let parsed = mt_ml::backend::parse_filename(filename, folder)?;
    Ok(ParsedName {
        title: parsed.title,
        year: parsed.year,
        season: parsed.season,
        episode: parsed.episode,
        media_type: parsed.media_type,
        is_anime: parsed.is_anime,
        release_group: parsed.release_group,
    })
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

    /// Integration test: actually calls the embedded Python `parse_filename`.
    ///
    /// Marked `#[ignore]` because it requires the Python venv to be in place
    /// (and `PYO3_PYTHON` set at compile time). Run with
    /// `cargo test -p mt-discovery -- --ignored` after `just deps`.
    #[test]
    #[ignore]
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
