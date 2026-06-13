use serde::{Deserialize, Serialize};

/// Identified media metadata, produced by the identification stage.
///
/// Optional fields carry `#[serde(default)]` so JSON that omits them still
/// deserializes correctly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaIdentity {
    /// Best-guess title (container metadata preferred).
    pub title: String,
    /// Title derived from filename parsing (cleaner, better for text search).
    pub parsed_title: String,
    /// Release year, if known.
    pub year: Option<i32>,
    /// Season number, if applicable.
    pub season: Option<i32>,
    /// Episode number, if applicable.
    pub episode: Option<i32>,
    /// `"movie"` or `"episode"`.
    pub media_type: String,
    /// OpenSubtitles file hash — exactly 16 hex characters.
    pub oshash: String,
    /// File size in bytes (required by the OpenSubtitles API).
    pub file_size: i64,
    /// Original filename, used as a fallback search term.
    pub raw_filename: String,
    /// IMDB identifier, e.g. `"tt0903747"`. Defaults to `None`.
    #[serde(default)]
    pub imdb_id: Option<String>,
    /// TMDB numeric identifier. Defaults to `None`.
    #[serde(default)]
    pub tmdb_id: Option<i32>,
    /// `true` if the title was detected as anime (via aniparse, release group, etc.).
    /// Defaults to `false`.
    #[serde(default)]
    pub is_anime: bool,
    /// Fansub / release group name, e.g. `"HorribleSubs"`. Defaults to `None`.
    #[serde(default)]
    pub release_group: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::{from_str, to_string};

    use super::*;

    fn minimal_identity() -> MediaIdentity {
        MediaIdentity {
            title: "One Piece".to_string(),
            parsed_title: "One Piece".to_string(),
            year: Some(1999),
            season: Some(1),
            episode: Some(1),
            media_type: "episode".to_string(),
            oshash: "abcdef1234567890".to_string(),
            file_size: 1_234_567_890,
            raw_filename: "One.Piece.S01E01.mkv".to_string(),
            imdb_id: None,
            tmdb_id: None,
            is_anime: false,
            release_group: None,
        }
    }

    #[test]
    fn serde_round_trip_full() {
        let id = MediaIdentity {
            imdb_id: Some("tt0388629".to_string()),
            tmdb_id: Some(37854),
            is_anime: true,
            release_group: Some("HorribleSubs".to_string()),
            ..minimal_identity()
        };
        let json = to_string(&id).expect("serialize");
        let back = from_str::<MediaIdentity>(&json).expect("deserialize");
        assert_eq!(id, back);
    }

    #[test]
    fn serde_default_fields_omitted() {
        // JSON with no imdb_id / tmdb_id / is_anime / release_group should
        // deserialize using the defaults (None / false / None).
        let json = r#"{
            "title": "Spirited Away",
            "parsed_title": "Spirited Away",
            "year": 2001,
            "season": null,
            "episode": null,
            "media_type": "movie",
            "oshash": "0123456789abcdef",
            "file_size": 2000000000,
            "raw_filename": "Spirited.Away.mkv"
        }"#;
        let id = from_str::<MediaIdentity>(json).expect("deserialize");
        assert!(id.imdb_id.is_none());
        assert!(id.tmdb_id.is_none());
        assert!(!id.is_anime);
        assert!(id.release_group.is_none());
        assert_eq!(id.media_type, "movie");
    }

    #[test]
    fn movie_has_no_season_episode() {
        let id = MediaIdentity {
            media_type: "movie".to_string(),
            season: None,
            episode: None,
            ..minimal_identity()
        };
        assert!(id.season.is_none());
        assert!(id.episode.is_none());
    }
}
