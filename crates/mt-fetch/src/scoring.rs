//! Subtitle match scoring utilities.

use std::collections::HashSet;

use regex::Regex;

/// Split a release name into lowercase tokens.
fn tokenize(name: &str) -> HashSet<String> {
    if name.is_empty() {
        return HashSet::new();
    }
    // Split on `.`, `-`, `_`, whitespace, `[`, `]`, `(`, `)`
    let re = Regex::new(r"[.\-_\s\[\]()]+").unwrap();
    re.split(name)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Score how well a subtitle release name matches a video filename.
///
/// Returns `0.0` to `1.0` based on token overlap (Jaccard similarity).
pub fn compute_release_score(video_name: &str, release_name: &str) -> f64 {
    let video_tokens = tokenize(video_name);
    let release_tokens = tokenize(release_name);

    if video_tokens.is_empty() || release_tokens.is_empty() {
        return 0.0;
    }

    let intersection: HashSet<_> = video_tokens.intersection(&release_tokens).collect();
    let union: HashSet<_> = video_tokens.union(&release_tokens).collect();

    intersection.len() as f64 / union.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_video_name_returns_zero() {
        assert_eq!(compute_release_score("", "Breaking.Bad.S01E03"), 0.0);
    }

    #[test]
    fn empty_release_name_returns_zero() {
        assert_eq!(compute_release_score("Breaking.Bad.S01E03.mkv", ""), 0.0);
    }

    #[test]
    fn identical_names_return_one() {
        let s = compute_release_score("Breaking.Bad.S01E03", "Breaking.Bad.S01E03");
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn completely_different_returns_zero() {
        let s = compute_release_score("Naruto.ep001", "One.Piece.ep100");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn partial_overlap_between_zero_and_one() {
        let s = compute_release_score("Breaking.Bad.S01E03.720p", "Breaking.Bad.S01E03.1080p");
        assert!(s > 0.0 && s < 1.0);
    }

    #[test]
    fn case_insensitive_matching() {
        let s = compute_release_score("breaking.bad.S01E03", "BREAKING.BAD.S01E03");
        assert!((s - 1.0).abs() < 1e-9);
    }
}
