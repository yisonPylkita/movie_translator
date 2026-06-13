//! Fetch subtitles from online providers.
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use mt_core::{FetchedSubtitle, PipelineContext};
use mt_fetch::align_ilass::is_available;
use mt_fetch::providers::{
    SubtitleProvider, animesub::AnimeSubProvider, napiprojekt::NapiProjektProvider,
    opensubtitles::OpenSubtitlesProvider, podnapisi::PodnapisiProvider,
};
use mt_fetch::retry::with_retry;
use mt_fetch::{SubtitleFetcher, SubtitleMatch, SubtitleValidator};
use mt_subtitles::normalize_encoding;
use tracing::{debug, info, warn};

use crate::error::Result;

/// Stage role name.
pub const NAME: &str = "fetch";

/// Keep all Polish subs scoring at or above this threshold.
const QUALITY_THRESHOLD: f64 = 0.8;
/// Validator window: the default line-match tolerance for `SubtitleValidator`.
const VALIDATOR_WINDOW_MS: i64 = 2000;
/// Minimum validation score for a candidate to be considered.
const MIN_VALIDATION_THRESHOLD: f64 = 0.5;
/// ilass split penalty: the default passed to `align_to_reference`.
const ILASS_SPLIT_PENALTY: f64 = 7.0;

/// Run the fetch stage.
///
/// Searches all providers, downloads every
/// candidate, validates/selects per language against the reference, then
/// realigns fetched Polish subs to the English reference.
pub fn run(mut ctx: PipelineContext) -> Result<PipelineContext> {
    if !ctx.config.enable_fetch {
        return Ok(ctx);
    }

    let identity = match ctx.identity.as_ref() {
        Some(id) => id,
        None => {
            ctx.fetched_subtitles = Some(HashMap::new());
            return Ok(ctx);
        }
    };

    let fetcher = build_fetcher(&ctx.video_path);

    let all_matches = fetcher.search_all(identity, &["eng", "pol"]);
    if all_matches.is_empty() {
        info!("No subtitles found from any provider");
        ctx.fetched_subtitles = Some(HashMap::new());
        return Ok(ctx);
    }
    info!("Found {} subtitle candidate(s)", all_matches.len());

    // Download all candidates.
    let candidates_dir = ctx.work_dir.join("candidates");
    fs::create_dir_all(&candidates_dir)?;
    let downloaded = download_all(&fetcher, &all_matches, &candidates_dir);

    if downloaded.is_empty() {
        warn!("All candidate downloads failed");
        ctx.fetched_subtitles = Some(HashMap::new());
        return Ok(ctx);
    }

    // Validate and select per language.
    let (selected, _best_score) = validate_and_select(&downloaded, ctx.reference_path.as_deref());
    ctx.fetched_subtitles = Some(selected);

    // Realign fetched Polish subtitles against the English reference.
    if let (Some(reference), Some(map)) =
        (ctx.reference_path.clone(), ctx.fetched_subtitles.clone())
        && let Some(pol_subs) = map.get("pol")
    {
        for sub in pol_subs {
            let (method, offset) = align_subtitle(&sub.path, &reference);
            debug!(
                "aligned {} via {method} (offset={offset:?})",
                sub.path.display()
            );
        }
    }

    Ok(ctx)
}

/// Build the provider stack.
fn build_fetcher(video_path: &Path) -> SubtitleFetcher {
    let mut providers: Vec<Box<dyn SubtitleProvider>> = vec![
        Box::new(AnimeSubProvider::new()),
        Box::new(PodnapisiProvider::new()),
    ];
    let mut napi = NapiProjektProvider::new();
    napi.set_video_path(video_path);
    providers.push(Box::new(napi));
    if let Ok(api_key) = env::var("OPENSUBTITLES_API_KEY")
        && !api_key.is_empty()
    {
        providers.push(Box::new(OpenSubtitlesProvider::new(
            Some(api_key),
            None,
            None,
        )));
    }
    SubtitleFetcher::new(providers)
}

/// Download every candidate, normalising encoding.
fn download_all(
    fetcher: &SubtitleFetcher,
    matches: &[SubtitleMatch],
    candidates_dir: &Path,
) -> Vec<(SubtitleMatch, PathBuf)> {
    let mut downloaded = Vec::new();
    for (i, m) in matches.iter().enumerate() {
        let filename = format!("{}_{}_{}.{}", m.source, m.language, i, m.format);
        let output_path = candidates_dir.join(filename);
        let label = format!("download_{}", m.source);
        let result = with_retry(
            || fetcher.download_candidate(m, &output_path),
            1,
            2.0,
            &label,
            |secs| sleep(Duration::from_secs_f64(secs)),
        );
        match result {
            Ok(_) => {
                // Normalize encoding to UTF-8 so the parser can read Polish chars.
                if let Err(e) = normalize_encoding(&output_path) {
                    debug!(
                        "Encoding normalization failed for {}: {e}",
                        output_path.display()
                    );
                }
                downloaded.push((m.clone(), output_path));
            }
            Err(e) => {
                warn!("Failed to download candidate {}: {e}", m.subtitle_id);
            }
        }
    }
    info!("Downloaded {} candidate(s)", downloaded.len());
    downloaded
}

/// Align a subtitle file to a reference, trying ilass first.
///
/// Returns `(method, offset_ms)`
/// where `offset_ms` is `None` for the ilass path.
pub fn align_subtitle(subtitle_path: &Path, reference_path: &Path) -> (&'static str, Option<i64>) {
    if is_available() {
        if mt_fetch::align_ilass(subtitle_path, reference_path, ILASS_SPLIT_PENALTY) {
            return ("ilass", None);
        }
        info!("ilass alignment failed, falling back to built-in");
    }
    let offset = mt_fetch::align_cross_correlation(
        subtitle_path,
        reference_path,
        mt_fetch::align::MIN_OFFSET_MS,
    );
    ("builtin", Some(offset))
}

/// Validate downloaded candidates and select per language.
///
/// Returns
/// `(result_map, best_score)` where `best_score` is `None` when validation was
/// skipped or no candidate passed.
pub fn validate_and_select(
    downloaded: &[(SubtitleMatch, PathBuf)],
    reference_path: Option<&Path>,
) -> (HashMap<String, Vec<FetchedSubtitle>>, Option<f64>) {
    let mut result: HashMap<String, Vec<FetchedSubtitle>> = HashMap::new();
    let mut best_score = None;

    // When a reference exists, score+filter. Otherwise fall back to provider order.
    let validated: Option<Vec<(SubtitleMatch, PathBuf, f64)>> = reference_path.map(|reference| {
        let validator = SubtitleValidator::new(reference, VALIDATOR_WINDOW_MS);
        validator.validate_candidates(downloaded, MIN_VALIDATION_THRESHOLD)
    });

    match validated {
        Some(validated) if !validated.is_empty() => {
            best_score = Some(validated[0].2);
            info!("{} candidate(s) passed validation", validated.len());
            for (m, path, score) in &validated {
                let sub = FetchedSubtitle {
                    path: path.clone(),
                    source: m.source.clone(),
                };
                let entry = result.entry(m.language.clone());
                match entry {
                    Entry::Vacant(v) => {
                        // First (best) candidate for this language — always keep.
                        v.insert(vec![sub]);
                        info!(
                            "Selected {}: {} (score: {score:.3}, source: {})",
                            m.language, m.release_name, m.source
                        );
                    }
                    Entry::Occupied(mut o) => {
                        if *score >= QUALITY_THRESHOLD {
                            o.get_mut().push(sub);
                            info!(
                                "Also keeping {}: {} (score: {score:.3}, source: {})",
                                m.language, m.release_name, m.source
                            );
                        }
                    }
                }
            }
        }
        Some(_) => {
            warn!("No candidates passed validation threshold");
        }
        None => {
            for (m, path) in downloaded {
                result.entry(m.language.clone()).or_insert_with(|| {
                    info!(
                        "Best {} (unvalidated): {} (source: {})",
                        m.language, m.release_name, m.source
                    );
                    vec![FetchedSubtitle {
                        path: path.clone(),
                        source: m.source.clone(),
                    }]
                });
            }
        }
    }

    (result, best_score)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    fn make_match(language: &str, source: &str, id: &str) -> SubtitleMatch {
        SubtitleMatch {
            language: language.into(),
            source: source.into(),
            subtitle_id: id.into(),
            release_name: format!("release-{id}"),
            format: "srt".into(),
            score: 1.0,
            hash_match: false,
        }
    }

    #[test]
    fn no_reference_keeps_first_per_language() {
        let p1 = PathBuf::from("/tmp/a.srt");
        let p2 = PathBuf::from("/tmp/b.srt");
        let downloaded = vec![
            (make_match("pol", "animesub", "1"), p1),
            (make_match("pol", "podnapisi", "2"), p2),
        ];
        let (result, best) = validate_and_select(&downloaded, None);
        assert!(best.is_none());
        assert_eq!(result["pol"].len(), 1);
        assert_eq!(result["pol"][0].source, "animesub");
    }

    #[test]
    fn align_subtitle_builtin_path_returns_offset() {
        // ilass binary is typically absent in CI; this exercises the builtin path.
        let dir = tempdir().unwrap();
        let sub = dir.path().join("pol.srt");
        let reference = dir.path().join("ref.srt");
        fs::write(&sub, "1\n00:00:01,000 --> 00:00:02,000\nA\n").unwrap();
        fs::write(&reference, "1\n00:00:01,000 --> 00:00:02,000\nA\n").unwrap();
        let (method, offset) = align_subtitle(&sub, &reference);
        // Either ilass (if present) or builtin; if builtin, offset is Some.
        assert!(method == "ilass" || (method == "builtin" && offset.is_some()));
    }
}
