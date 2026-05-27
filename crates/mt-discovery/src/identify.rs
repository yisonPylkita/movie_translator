//! Media identification orchestration.
//!
//! Ported from `movie_translator/identifier/identify.py`.
//!
//! Combines filename parsing, container metadata, file hash, and optional
//! TMDB enrichment into a [`mt_core::MediaIdentity`].
//!
//! TODO: metrics spans (Python `MetricsCollector`) have no Rust equivalent
//! yet — the span calls are omitted and left as a future concern.

use crate::hasher::compute_oshash;
use crate::metadata::{extract_container_metadata, ContainerMetadata};
use crate::parser::{parse_filename, ParsedName};
use crate::tmdb::{lookup_tmdb, TmdbResult};
use mt_core::{MediaIdentity, Result};
use std::path::Path;

/// Assemble a [`MediaIdentity`] from already-computed components.
///
/// Factored out from the subprocess/HTTP orchestration so it can be
/// unit-tested with stub inputs.
///
/// Priority: container metadata title > parsed filename title > raw filename.
pub(crate) fn assemble_identity(
    video_path: &Path,
    parsed: &ParsedName,
    container: &ContainerMetadata,
    oshash: String,
    file_size: i64,
    tmdb: Option<&TmdbResult>,
) -> MediaIdentity {
    let raw_filename = video_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    // parsed_title is the clean filename-derived title (better for text search)
    let parsed_title = parsed.title.clone().unwrap_or_else(|| raw_filename.clone());

    // container metadata overrides when available
    let title = container
        .title
        .clone()
        .unwrap_or_else(|| parsed_title.clone());

    let season = parsed.season;
    let mut episode = parsed.episode;
    let year = parsed.year;
    // Python does NOT re-derive media_type after merging a container-sourced
    // episode: whatever the parser decided (e.g. "movie") is preserved even
    // when a container episode is found. Match that — media_type is never
    // changed here.
    let media_type = parsed.media_type.clone();
    let is_anime = parsed.is_anime;
    let release_group = parsed.release_group.clone();

    // If container has episode info and parsed didn't get one, try to use it
    if let Some(ref container_ep_str) = container.episode {
        if episode.is_none() {
            if let Ok(ep) = container_ep_str.parse::<i32>() {
                episode = Some(ep);
            }
        }
    }

    let (imdb_id, tmdb_id) = match tmdb {
        Some(t) => (t.imdb_id.clone(), Some(t.tmdb_id)),
        None => (None, None),
    };

    MediaIdentity {
        title,
        parsed_title,
        year,
        season,
        episode,
        media_type,
        oshash,
        file_size,
        raw_filename,
        imdb_id,
        tmdb_id,
        is_anime,
        release_group,
    }
}

/// Identify a video file using filename, container metadata, and file hash.
///
/// Combines multiple signals with priority:
/// container metadata > filename > folder name.
///
/// TMDB enrichment is attempted if `TMDB_API_KEY` is set in the environment.
pub fn identify_media(video_path: &Path) -> Result<MediaIdentity> {
    let filename = video_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let folder_name = video_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str());

    // Signal 1: Parse filename (and folder as fallback context)
    // TODO: metrics span 'parse_filename'
    let parsed = parse_filename(filename, folder_name)?;

    // Signal 2: Container metadata (overrides filename when present)
    // TODO: metrics span 'extract_container_metadata'
    let container = extract_container_metadata(video_path);

    // Signal 3: File hash
    // TODO: metrics span 'compute_oshash'
    let oshash = compute_oshash(video_path).unwrap_or_default();

    let file_size = video_path.metadata()?.len() as i64;

    // Signal 4: TMDB enrichment (optional, requires TMDB_API_KEY)
    // TODO: metrics span 'lookup_tmdb'
    let parsed_title = parsed.title.clone().unwrap_or_else(|| filename.to_string());
    let tmdb_result = lookup_tmdb(&parsed_title, parsed.year, &parsed.media_type);

    Ok(assemble_identity(
        video_path,
        &parsed,
        &container,
        oshash,
        file_size,
        tmdb_result.as_ref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ContainerMetadata;
    use crate::parser::ParsedName;
    use crate::tmdb::TmdbResult;
    use std::path::Path;

    fn anime_parsed() -> ParsedName {
        ParsedName {
            title: Some("One Piece".to_string()),
            year: None,
            season: None,
            episode: Some(1000),
            media_type: "episode".to_string(),
            is_anime: true,
            release_group: Some("HorribleSubs".to_string()),
        }
    }

    fn movie_parsed() -> ParsedName {
        ParsedName {
            title: Some("Spirited Away".to_string()),
            year: Some(2001),
            season: None,
            episode: None,
            media_type: "movie".to_string(),
            is_anime: false,
            release_group: None,
        }
    }

    fn no_container() -> ContainerMetadata {
        ContainerMetadata::default()
    }

    fn container_with_title(title: &str) -> ContainerMetadata {
        ContainerMetadata {
            title: Some(title.to_string()),
            episode: None,
        }
    }

    fn container_with_episode(ep: &str) -> ContainerMetadata {
        ContainerMetadata {
            title: None,
            episode: Some(ep.to_string()),
        }
    }

    // ── title priority ────────────────────────────────────────────────────────

    #[test]
    fn container_title_overrides_parsed() {
        let video = Path::new("/fake/[HorribleSubs] One Piece - 1000 [1080p].mkv");
        let id = assemble_identity(
            video,
            &anime_parsed(),
            &container_with_title("One Piece (Container)"),
            "abcdef1234567890".to_string(),
            123456,
            None,
        );
        assert_eq!(id.title, "One Piece (Container)");
        assert_eq!(id.parsed_title, "One Piece");
    }

    #[test]
    fn parsed_title_used_when_container_empty() {
        let video = Path::new("/fake/[HorribleSubs] One Piece - 1000 [1080p].mkv");
        let id = assemble_identity(
            video,
            &anime_parsed(),
            &no_container(),
            "abcdef1234567890".to_string(),
            123456,
            None,
        );
        assert_eq!(id.title, "One Piece");
        assert_eq!(id.parsed_title, "One Piece");
    }

    #[test]
    fn raw_filename_used_when_no_title_at_all() {
        let video = Path::new("/fake/unknown_video.mkv");
        let parsed = ParsedName {
            title: None,
            year: None,
            season: None,
            episode: None,
            media_type: "movie".to_string(),
            is_anime: false,
            release_group: None,
        };
        let id = assemble_identity(video, &parsed, &no_container(), "".to_string(), 0, None);
        assert_eq!(id.title, "unknown_video.mkv");
        assert_eq!(id.parsed_title, "unknown_video.mkv");
    }

    // ── episode from container ─────────────────────────────────────────────────

    #[test]
    fn container_episode_used_when_parsed_missing() {
        let video = Path::new("/fake/show.mkv");
        let parsed = ParsedName {
            title: Some("Show".to_string()),
            year: None,
            season: None,
            episode: None,
            media_type: "movie".to_string(),
            is_anime: false,
            release_group: None,
        };
        let id = assemble_identity(
            video,
            &parsed,
            &container_with_episode("5"),
            "".to_string(),
            0,
            None,
        );
        assert_eq!(id.episode, Some(5));
        // media_type stays "movie" (the parsed value): Python does NOT
        // re-derive media_type when a container-sourced episode is merged in.
        // This matches Python intentionally.
        assert_eq!(id.media_type, "movie");
    }

    #[test]
    fn parsed_episode_preferred_over_container() {
        let video = Path::new("/fake/[HorribleSubs] One Piece - 1000 [1080p].mkv");
        let container = ContainerMetadata {
            title: None,
            episode: Some("999".to_string()),
        };
        let id = assemble_identity(video, &anime_parsed(), &container, "".to_string(), 0, None);
        // parsed episode (1000) wins because episode was already Some
        assert_eq!(id.episode, Some(1000));
    }

    // ── TMDB enrichment ────────────────────────────────────────────────────────

    #[test]
    fn tmdb_ids_populated_when_present() {
        let video = Path::new("/fake/Spirited.Away.mkv");
        let tmdb = TmdbResult {
            tmdb_id: 129,
            imdb_id: Some("tt0245429".to_string()),
        };
        let id = assemble_identity(
            video,
            &movie_parsed(),
            &no_container(),
            "0000000000000000".to_string(),
            1_000_000,
            Some(&tmdb),
        );
        assert_eq!(id.tmdb_id, Some(129));
        assert_eq!(id.imdb_id.as_deref(), Some("tt0245429"));
    }

    #[test]
    fn tmdb_none_leaves_ids_empty() {
        let video = Path::new("/fake/Spirited.Away.mkv");
        let id = assemble_identity(
            video,
            &movie_parsed(),
            &no_container(),
            "".to_string(),
            0,
            None,
        );
        assert!(id.tmdb_id.is_none());
        assert!(id.imdb_id.is_none());
    }

    // ── anime fields ───────────────────────────────────────────────────────────

    #[test]
    fn is_anime_and_release_group_preserved() {
        let video = Path::new("/fake/[HorribleSubs] One Piece - 1000 [1080p].mkv");
        let id = assemble_identity(
            video,
            &anime_parsed(),
            &no_container(),
            "".to_string(),
            0,
            None,
        );
        assert!(id.is_anime);
        assert_eq!(id.release_group.as_deref(), Some("HorribleSubs"));
    }

    // ── basic fields ───────────────────────────────────────────────────────────

    #[test]
    fn year_and_media_type_from_parsed() {
        let video = Path::new("/fake/Spirited.Away.2001.mkv");
        let id = assemble_identity(
            video,
            &movie_parsed(),
            &no_container(),
            "".to_string(),
            2_000_000_000,
            None,
        );
        assert_eq!(id.year, Some(2001));
        assert_eq!(id.media_type, "movie");
        assert_eq!(id.file_size, 2_000_000_000);
    }
}
