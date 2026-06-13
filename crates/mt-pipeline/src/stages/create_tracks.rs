//! Create subtitle track files and build the track list.
use std::fs;
use std::path::{Path, PathBuf};

use mt_core::{MediaIdentity, PipelineContext, SubtitleFile};
use mt_subtitles::{create_polish_subtitles, override_font_name};
use serde_json::from_str;
#[cfg(test)]
use tempfile::tempdir;
use tracing::{info, warn};

use crate::error::Result;

/// Stage role name.
pub const NAME: &str = "create_tracks";

/// Human-readable label for a translation backend (used as a track title).
pub fn model_label(model_name: &str) -> String {
    match model_name {
        "allegro" => "Allegro".to_string(),
        "apple" => "Apple".to_string(),
        "mlx" => "MLX".to_string(),
        other => other.to_string(),
    }
}

/// Title for an AI-translated Polish track. Notes ASR provenance when the
/// English source was transcribed from the audio track (`--transcribe`).
pub fn polish_ai_title(model_name: &str, from_asr: bool) -> String {
    if from_asr {
        format!(
            "Polish ({}, from AI transcription)",
            model_label(model_name)
        )
    } else {
        format!("Polish ({})", model_label(model_name))
    }
}

/// Map external manifest language codes to track language codes.
fn lang_to_track(lang: &str) -> String {
    match lang {
        "pl" => "pol".to_string(),
        "en" => "eng".to_string(),
        other => other.to_string(),
    }
}

/// Run the create-tracks stage.
///
/// Builds one Polish `.ass` per translation
/// model (primary first, then extras), applies the font fallback rename, and
/// assembles the ordered track list (fetched Polish, AI Polish, external).
pub fn run(mut ctx: PipelineContext) -> Result<PipelineContext> {
    let is_mkv = ctx
        .video_path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("mkv"))
        .unwrap_or(false);
    let mut replace_chars = false;

    let font_info = ctx
        .font_info
        .clone()
        .expect("create_tracks requires font_info");
    let english_source = ctx
        .english_source
        .clone()
        .expect("create_tracks requires english_source");
    let translated_lines = ctx
        .translated_lines
        .clone()
        .expect("create_tracks requires translated_lines");

    if !font_info.supports_polish {
        if is_mkv && !font_info.font_attachments.is_empty() {
            info!(
                "Will embed font \"{}\"",
                font_info.font_attachments[0]
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            );
        } else if is_mkv {
            warn!("No system font with Polish support, replacing characters");
            replace_chars = true;
        } else {
            replace_chars = true;
        }
    }

    let stem = ctx
        .video_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    // One Polish subtitle file per translation model. Primary first.
    let primary_model = ctx.config.model.clone();
    let mut polish_files: Vec<(String, PathBuf)> = Vec::new();

    let primary_ass = ctx
        .work_dir
        .join(format!("{stem}_polish_{primary_model}.ass"));
    create_polish_subtitles(
        &english_source,
        &translated_lines,
        &primary_ass,
        replace_chars,
    )?;
    polish_files.push((primary_model, primary_ass));

    // Iterate extra models in config order (HashMap iteration order is not
    // stable, so drive from config.extra_models which preserves order).
    for extra_model in &ctx.config.extra_models {
        if let Some(extra_lines) = ctx.extra_translations.get(extra_model) {
            let extra_ass = ctx
                .work_dir
                .join(format!("{stem}_polish_{extra_model}.ass"));
            create_polish_subtitles(&english_source, extra_lines, &extra_ass, replace_chars)?;
            polish_files.push((extra_model.clone(), extra_ass));
        }
    }
    // Any extra translations not listed in extra_models (defensive — iterate
    // ctx.extra_translations directly).
    for (extra_model, extra_lines) in &ctx.extra_translations {
        if ctx.config.extra_models.contains(extra_model) {
            continue;
        }
        let extra_ass = ctx
            .work_dir
            .join(format!("{stem}_polish_{extra_model}.ass"));
        create_polish_subtitles(&english_source, extra_lines, &extra_ass, replace_chars)?;
        polish_files.push((extra_model.clone(), extra_ass));
    }

    if let Some(fallback) = &font_info.fallback_font_family {
        for (_, ass_path) in &polish_files {
            override_font_name(ass_path, fallback)?;
        }
    }

    // Build the track list.
    let fetched_pol_list: Vec<_> = ctx
        .fetched_subtitles
        .as_ref()
        .and_then(|m| m.get("pol"))
        .cloned()
        .unwrap_or_default();

    let mut tracks: Vec<SubtitleFile> = Vec::new();

    for (i, fetched_pol) in fetched_pol_list.iter().enumerate() {
        tracks.push(SubtitleFile {
            path: fetched_pol.path.clone(),
            language: "pol".to_string(),
            title: format!("Polish ({})", fetched_pol.source),
            is_default: i == 0,
        });
        if let Some(fallback) = &font_info.fallback_font_family {
            override_font_name(&fetched_pol.path, fallback)?;
        }
    }

    for (i, (model_name, ass_path)) in polish_files.iter().enumerate() {
        let is_default = i == 0 && fetched_pol_list.is_empty();
        tracks.push(SubtitleFile {
            path: ass_path.clone(),
            language: "pol".to_string(),
            title: polish_ai_title(model_name, ctx.english_from_asr),
            is_default,
        });
    }

    // The ASR transcript itself ships as an English track so the original
    // (pre-translation) text stays inspectable alongside the Polish.
    if ctx.english_from_asr {
        tracks.push(SubtitleFile {
            path: english_source.clone(),
            language: "eng".to_string(),
            title: "English (AI Transcribed)".to_string(),
            is_default: false,
        });
    }

    // External subtitles, if configured.
    if let (Some(external_dir), Some(identity)) =
        (ctx.config.external_subs_dir.clone(), ctx.identity.clone())
    {
        let external_tracks = load_external_subs(&external_dir, &identity);
        tracks.extend(external_tracks);
    }

    ctx.subtitle_tracks = Some(tracks);
    Ok(ctx)
}

// ── External subtitle manifest ────────────────────────────────────────────

#[derive(serde::Deserialize, Default)]
struct Manifest {
    #[serde(default)]
    entries: Vec<ManifestEntry>,
}

#[derive(serde::Deserialize, Default)]
struct ManifestEntry {
    #[serde(default)]
    identity: ManifestIdentity,
    #[serde(default)]
    subtitles: Vec<ManifestSub>,
}

#[derive(serde::Deserialize, Default)]
struct ManifestIdentity {
    #[serde(default)]
    parsed_title: Option<String>,
    #[serde(default)]
    season: Option<i32>,
    #[serde(default)]
    episode: Option<i32>,
}

#[derive(serde::Deserialize, Default)]
struct ManifestSub {
    file: String,
    language: String,
    #[serde(default)]
    method: Option<String>,
}

/// Load matching external subtitles from a manifest directory.
///
/// Matches by parsed title + season + episode,
/// falling back to episode-number matching.
pub fn load_external_subs(external_dir: &Path, identity: &MediaIdentity) -> Vec<SubtitleFile> {
    let manifest_path = external_dir.join("manifest.json");
    if !manifest_path.exists() {
        warn!("No manifest.json found in {}", external_dir.display());
        return Vec::new();
    }

    let manifest: Manifest = match fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|s| from_str(&s).ok())
    {
        Some(m) => m,
        None => return Vec::new(),
    };

    let cur_title = identity.parsed_title.to_lowercase();
    let cur_season = identity.season;
    let cur_episode = identity.episode;

    let mut tracks = Vec::new();

    for entry in &manifest.entries {
        let entry_title = entry
            .identity
            .parsed_title
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        let entry_season = entry.identity.season;
        let entry_episode = entry.identity.episode;

        let title_match = !cur_title.is_empty()
            && !entry_title.is_empty()
            && (entry_title.contains(&cur_title) || cur_title.contains(&entry_title));
        let episode_match = cur_season.is_some()
            && cur_episode.is_some()
            && cur_season == entry_season
            && cur_episode == entry_episode;

        if !(title_match && episode_match) {
            // Fallback: match on episode number alone.
            if !(cur_episode.is_some() && cur_episode == entry_episode) {
                continue;
            }
        }

        for sub in &entry.subtitles {
            let sub_path = external_dir.join(&sub.file);
            if !sub_path.exists() {
                warn!("External subtitle not found: {}", sub_path.display());
                continue;
            }
            let lang_code = lang_to_track(&sub.language);
            let method = sub.method.as_deref().unwrap_or("unknown");
            let title = format!("{} (external, {method})", sub.language.to_uppercase());
            tracks.push(SubtitleFile {
                path: sub_path,
                language: lang_code,
                title,
                is_default: false,
            });
            info!("Adding external subtitle: {}", sub.file);
        }
    }

    tracks
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;

    use mt_core::{DialogueLine, FetchedSubtitle, FontInfo, PipelineConfig};

    use super::*;

    fn base_ctx(tmp: &Path) -> PipelineContext {
        let video = tmp.join("ep01.mkv");
        fs::write(&video, b"fake").unwrap();
        let eng = tmp.join("eng.srt");
        fs::write(&eng, "1\n00:00:01,000 --> 00:00:02,000\nHello\n").unwrap();
        let work = tmp.join("work");
        fs::create_dir_all(&work).unwrap();
        let mut c = PipelineContext::new(video, work, PipelineConfig::default());
        c.english_source = Some(eng);
        c.dialogue_lines = Some(vec![DialogueLine {
            start_ms: 1000,
            end_ms: 2000,
            text: "Hello".into(),
        }]);
        c.translated_lines = Some(vec![DialogueLine {
            start_ms: 1000,
            end_ms: 2000,
            text: "Cześć".into(),
        }]);
        c.font_info = Some(FontInfo {
            supports_polish: true,
            font_attachments: Vec::new(),
            fallback_font_family: None,
        });
        c.fetched_subtitles = Some(HashMap::new());
        c
    }

    fn titles(ctx: &PipelineContext) -> Vec<String> {
        ctx.subtitle_tracks
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| t.title.clone())
            .collect()
    }

    #[test]
    fn polish_ai_title_notes_asr_provenance() {
        assert_eq!(polish_ai_title("allegro", false), "Polish (Allegro)");
        assert_eq!(
            polish_ai_title("allegro", true),
            "Polish (Allegro, from AI transcription)"
        );
        assert_eq!(
            polish_ai_title("apple", true),
            "Polish (Apple, from AI transcription)"
        );
    }

    #[test]
    fn model_label_known_and_unknown() {
        assert_eq!(model_label("allegro"), "Allegro");
        assert_eq!(model_label("apple"), "Apple");
        assert_eq!(model_label("mlx"), "MLX");
        assert_eq!(model_label("nllb"), "nllb");
    }

    #[test]
    fn creates_ai_polish_track() {
        let dir = tempdir().unwrap();
        let result = run(base_ctx(dir.path())).unwrap();
        assert!(titles(&result).contains(&"Polish (MLX)".to_string()));
    }

    #[test]
    fn fetched_polish_includes_source_and_is_default() {
        let dir = tempdir().unwrap();
        let mut ctx = base_ctx(dir.path());
        let pol = dir.path().join("pol.ass");
        fs::write(&pol, "[Script Info]\n\n[Events]\n").unwrap();
        let mut m = HashMap::new();
        m.insert(
            "pol".to_string(),
            vec![FetchedSubtitle {
                path: pol,
                source: "podnapisi".into(),
            }],
        );
        ctx.fetched_subtitles = Some(m);

        let result = run(ctx).unwrap();
        let ts = titles(&result);
        assert!(ts.contains(&"Polish (podnapisi)".to_string()));
        assert!(ts.contains(&"Polish (MLX)".to_string()));
        let defaults: Vec<_> = result
            .subtitle_tracks
            .as_ref()
            .unwrap()
            .iter()
            .filter(|t| t.is_default)
            .collect();
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].title, "Polish (podnapisi)");
    }

    #[test]
    fn ai_is_default_when_no_fetched() {
        let dir = tempdir().unwrap();
        let result = run(base_ctx(dir.path())).unwrap();
        let defaults: Vec<_> = result
            .subtitle_tracks
            .as_ref()
            .unwrap()
            .iter()
            .filter(|t| t.is_default)
            .collect();
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].title, "Polish (MLX)");
    }

    #[test]
    fn extra_translations_emit_extra_track_primary_first() {
        let dir = tempdir().unwrap();
        let mut ctx = base_ctx(dir.path());
        ctx.config.extra_models = vec!["apple".into()];
        ctx.extra_translations.insert(
            "apple".to_string(),
            vec![DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "Cześć (apple)".into(),
            }],
        );
        let result = run(ctx).unwrap();
        assert_eq!(
            titles(&result),
            vec!["Polish (MLX)".to_string(), "Polish (Apple)".to_string()]
        );
        let defaults: Vec<_> = result
            .subtitle_tracks
            .as_ref()
            .unwrap()
            .iter()
            .filter(|t| t.is_default)
            .collect();
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].title, "Polish (MLX)");
    }
}
