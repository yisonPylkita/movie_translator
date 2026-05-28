//! `extract` subcommand: extract subtitles (embedded text + burned-in OCR).
//!
//! Embedded text-track extraction and identification run inline; burned-in OCR
//! routes through the pipeline GPU worker (`mt_ml::ocr_burned_in` via the
//! serialised worker).

use std::path::{Path, PathBuf};

use clap::Parser;
use mt_core::MediaIdentity;
use mt_discovery::{find_videos, identify_media};
use mt_media::SubtitleExtractor;
use mt_pipeline::GpuWorker;
use serde_json::json;

/// Image-based subtitle codecs that need OCR.
const IMAGE_CODECS: &[&str] = &["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"];
/// Burned-in OCR defaults: crop the bottom 25% of the frame, sample 3 fps.
const OCR_CROP_RATIO: f64 = 0.25;
const OCR_EXTRACT_FPS: u32 = 3;

/// Extract command arguments.
#[derive(Debug, Parser)]
#[command(
    name = "movie-translator extract",
    about = "Extract subtitles from video files (text tracks + OCR for burned-in)"
)]
pub struct ExtractArgs {
    /// Video file or directory containing video files.
    pub input: String,

    /// Output directory for SRTs and manifest (default: <input_dir>/extracted_subs/).
    #[arg(long)]
    pub output: Option<String>,

    /// Language hint for burned-in subtitle OCR.
    #[arg(long = "ocr-language", default_value = "pl")]
    pub ocr_language: String,

    #[arg(long, short = 'v', default_value_t = false)]
    pub verbose: bool,
}

/// Build a normalized filename stem from media identity.
fn build_output_stem(identity: &MediaIdentity) -> String {
    let raw = if !identity.parsed_title.is_empty() {
        identity.parsed_title.as_str()
    } else if !identity.title.is_empty() {
        identity.title.as_str()
    } else {
        "Unknown"
    };
    let title: String = raw
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
        .collect::<String>()
        .trim()
        .to_string();

    match (
        identity.media_type.as_str(),
        identity.season,
        identity.episode,
    ) {
        ("episode", Some(s), Some(e)) => format!("{title} - S{s:02}E{e:02}"),
        (_, _, Some(e)) => format!("{title} - E{e:02}"),
        _ => title,
    }
}

/// Count dialogue lines in a subtitle file.
fn count_subtitle_lines(path: &Path) -> usize {
    match mt_subtitles::load(path) {
        Ok(subs) => subs
            .events
            .iter()
            .filter(|e| e.kind == mt_subtitles::EventKind::Dialogue)
            .count(),
        Err(_) => std::fs::read_to_string(path)
            .map(|t| t.matches(" --> ").count())
            .unwrap_or(0),
    }
}

/// Extract embedded English/Polish text tracks.
fn extract_text_tracks(
    video_path: &Path,
    output_dir: &Path,
    extractor: &SubtitleExtractor,
    output_stem: &str,
) -> Vec<serde_json::Value> {
    let track_info = match extractor.get_track_info(video_path) {
        Ok(ti) => ti,
        Err(e) => {
            tracing::warn!("Failed to read track info: {e}");
            return Vec::new();
        }
    };
    let mut results = Vec::new();

    for track in &track_info.tracks {
        let lang = track.properties.language.to_ascii_lowercase();
        let out_lang = match lang.as_str() {
            "eng" | "en" | "und" => "en",
            "pol" | "pl" => "pl",
            _ => continue,
        };

        let codec = track.codec.to_ascii_lowercase();
        if IMAGE_CODECS
            .iter()
            .any(|c| codec == *c || codec.starts_with(c))
        {
            continue;
        }

        let track_name = track.properties.track_name.to_ascii_lowercase();
        if !track_name.is_empty()
            && ["sign", "song", "op", "ed"]
                .iter()
                .any(|kw| track_name.contains(kw))
        {
            continue;
        }

        let ext = extractor.get_subtitle_extension(track);
        let out_file = format!("{output_stem}.{out_lang}{ext}");
        let out_path = output_dir.join(&out_file);

        match extractor.extract_subtitle(
            video_path,
            track.id,
            &out_path,
            Some(track.subtitle_index),
        ) {
            Ok(()) => {
                let line_count = count_subtitle_lines(&out_path);
                tracing::info!("Extracted {out_lang} text track: {out_file} ({line_count} lines)");
                results.push(json!({
                    "file": out_file,
                    "language": out_lang,
                    "method": "embedded_text",
                    "line_count": line_count,
                }));
            }
            Err(e) => tracing::warn!("Failed to extract track {}: {e}", track.id),
        }
    }
    results
}

/// Extract burned-in subtitles via OCR through the GPU worker.
async fn extract_ocr(
    worker: &GpuWorker,
    video_path: &Path,
    output_dir: &Path,
    output_stem: &str,
    language: &str,
) -> Vec<serde_json::Value> {
    let work_dir = output_dir.join("_ocr_work");
    if std::fs::create_dir_all(&work_dir).is_err() {
        return Vec::new();
    }

    let result = worker
        .handle()
        .ocr_burned_in_async(
            video_path.to_path_buf(),
            work_dir.clone(),
            OCR_CROP_RATIO,
            OCR_EXTRACT_FPS,
        )
        .await;

    let out = match result {
        Ok(r) => {
            let out_file = format!("{output_stem}.{language}.ocr.srt");
            let out_path = output_dir.join(&out_file);
            if std::fs::copy(&r.srt_path, &out_path).is_ok() {
                let line_count = count_subtitle_lines(&out_path);
                tracing::info!(
                    "Extracted {language} OCR subtitles: {out_file} ({line_count} lines)"
                );
                vec![json!({
                    "file": out_file,
                    "language": language,
                    "method": "ocr_burned_in",
                    "line_count": line_count,
                })]
            } else {
                Vec::new()
            }
        }
        Err(_) => {
            tracing::warn!(
                "No burned-in subtitles found in {}",
                video_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            );
            Vec::new()
        }
    };

    let _ = std::fs::remove_dir_all(&work_dir);
    out
}

fn identity_to_json(identity: &MediaIdentity) -> serde_json::Value {
    json!({
        "title": identity.title,
        "parsed_title": identity.parsed_title,
        "season": identity.season,
        "episode": identity.episode,
        "media_type": identity.media_type,
        "is_anime": identity.is_anime,
    })
}

/// Run the extract flow. Returns the deliberate exit code as `Ok(code)`, or an
/// `Err` with a `.context()` chain (printed by `main`) for a genuine IO failure.
pub async fn run(args: ExtractArgs) -> anyhow::Result<i32> {
    use anyhow::Context;

    crate::init_tracing(args.verbose);

    let input_path = PathBuf::from(&args.input);
    if !input_path.exists() {
        eprintln!("Not found: {}", input_path.display());
        return Ok(1);
    }

    let output_dir = match &args.output {
        Some(o) => PathBuf::from(o),
        None => {
            let root = if input_path.is_dir() {
                input_path.clone()
            } else {
                input_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            root.join("extracted_subs")
        }
    };

    let video_files = find_videos(&input_path);
    if video_files.is_empty() {
        eprintln!("No video files found in {}", input_path.display());
        return Ok(1);
    }

    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;

    let extractor = SubtitleExtractor::new();
    let worker = GpuWorker::spawn();

    let mut entries: Vec<serde_json::Value> = Vec::new();
    eprintln!("Extracting subtitles from {} file(s)...", video_files.len());

    for video_path in &video_files {
        eprintln!(
            "\n{}",
            video_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default()
        );

        let identity = match identify_media(video_path) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!("Identification failed: {e}");
                continue;
            }
        };
        let output_stem = build_output_stem(&identity);
        eprintln!(
            "  Identified: {} (S{:?}E{:?})",
            identity.title, identity.season, identity.episode
        );

        let mut subtitles = extract_text_tracks(video_path, &output_dir, &extractor, &output_stem);
        let ocr = extract_ocr(
            &worker,
            video_path,
            &output_dir,
            &output_stem,
            &args.ocr_language,
        )
        .await;
        subtitles.extend(ocr);

        if subtitles.is_empty() {
            eprintln!("  No subtitles extracted");
        } else {
            for sub in &subtitles {
                eprintln!(
                    "  {} ({}, {}, {} lines)",
                    sub["file"].as_str().unwrap_or(""),
                    sub["language"].as_str().unwrap_or(""),
                    sub["method"].as_str().unwrap_or(""),
                    sub["line_count"].as_u64().unwrap_or(0)
                );
            }
        }

        entries.push(json!({
            "source_file": video_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            "identity": identity_to_json(&identity),
            "subtitles": subtitles,
        }));
    }

    worker.shutdown().await;

    let manifest = json!({
        "version": 1,
        "source_dir": input_path.canonicalize().unwrap_or(input_path.clone()).to_string_lossy(),
        "entries": entries,
    });
    let manifest_path = output_dir.join("manifest.json");
    let manifest_json =
        serde_json::to_string_pretty(&manifest).context("serializing extract manifest")?;
    std::fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("writing manifest {}", manifest_path.display()))?;
    eprintln!("\nManifest written to {}", manifest_path.display());
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> ExtractArgs {
        let mut full = vec!["extract"];
        full.extend_from_slice(argv);
        ExtractArgs::try_parse_from(full).expect("parse")
    }

    #[test]
    fn defaults_match_python() {
        let args = parse(&["/movies"]);
        assert_eq!(args.input, "/movies");
        assert!(args.output.is_none());
        assert_eq!(args.ocr_language, "pl");
        assert!(!args.verbose);
    }

    #[test]
    fn output_and_ocr_language_flags() {
        let args = parse(&["/movies", "--output", "/out", "--ocr-language", "en"]);
        assert_eq!(args.output.as_deref(), Some("/out"));
        assert_eq!(args.ocr_language, "en");
    }

    fn identity(
        media_type: &str,
        season: Option<i32>,
        episode: Option<i32>,
        title: &str,
    ) -> MediaIdentity {
        MediaIdentity {
            title: title.to_string(),
            parsed_title: title.to_string(),
            year: None,
            season,
            episode,
            media_type: media_type.to_string(),
            oshash: String::new(),
            file_size: 0,
            raw_filename: String::new(),
            imdb_id: None,
            tmdb_id: None,
            is_anime: false,
            release_group: None,
        }
    }

    #[test]
    fn output_stem_episode_format() {
        let id = identity("episode", Some(1), Some(2), "My Show!!");
        // illegal chars stripped, S/E zero-padded
        assert_eq!(build_output_stem(&id), "My Show - S01E02");
    }

    #[test]
    fn output_stem_episode_only() {
        let id = identity("movie", None, Some(5), "Title");
        assert_eq!(build_output_stem(&id), "Title - E05");
    }

    #[test]
    fn output_stem_plain_title() {
        let id = identity("movie", None, None, "Just A Movie");
        assert_eq!(build_output_stem(&id), "Just A Movie");
    }
}
