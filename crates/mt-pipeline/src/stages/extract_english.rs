//! Determine the English subtitle source and extract dialogue lines.

use std::path::PathBuf;

use mt_core::{PendingOcr, PipelineContext};
use mt_media::{SubtitleExtractor, SubtitleTrack};
use mt_subtitles::SubtitleProcessor;

use crate::error::{PipelineError, Result};
use crate::vision::{default_vision_ocr_probe, VisionOcrProbe};

/// Stage role name.
pub const NAME: &str = "extract";

const IMAGE_CODECS: &[&str] = &["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"];

fn is_image_codec(track: &SubtitleTrack) -> bool {
    let codec = track.codec.to_ascii_lowercase();
    IMAGE_CODECS
        .iter()
        .any(|c| codec == *c || codec.starts_with(c))
}

/// Select the best English subtitle source and extract dialogue lines.
///
/// Priority: fetched English > reference >
/// embedded text track > defer burned-in OCR.
pub fn run(ctx: PipelineContext) -> Result<PipelineContext> {
    run_with_probe(ctx, default_vision_ocr_probe)
}

/// Like [`run`], with an injectable Vision-OCR availability probe (for tests).
pub fn run_with_probe(
    mut ctx: PipelineContext,
    vision_ocr_available: VisionOcrProbe,
) -> Result<PipelineContext> {
    // 1. fetched English
    let fetched_eng = ctx
        .fetched_subtitles
        .as_ref()
        .and_then(|m| m.get("eng"))
        .and_then(|subs| subs.first())
        .map(|s| s.path.clone());

    if let Some(path) = fetched_eng {
        ctx.english_source = Some(path);
    } else if let Some(reference) = ctx.reference_path.clone() {
        ctx.english_source = Some(reference);
    } else {
        // Try embedded text track, defer burned-in OCR if needed (see
        // `PipelineConfig::burned_in_fallback_allowed` for why --transcribe
        // suppresses the fallback and lets the transcribe stage fill
        // english_source instead).
        ctx.english_source = extract_text_only(&ctx)?;
        if ctx.english_source.is_none()
            && ctx.config.burned_in_fallback_allowed()
            && !ctx.burned_in_probed
            && vision_ocr_available()
        {
            ctx.pending_ocr = Some(PendingOcr {
                r#type: "burned_in".to_string(),
                track_id: None,
                output_dir: ctx.work_dir.clone(),
            });
            return Ok(ctx); // Can't extract lines yet.
        }
    }

    if ctx.english_source.is_none() && ctx.config.enable_transcription {
        // ASR transcription runs as a later stage; leave the source empty for
        // it (the orchestrator re-checks and skips the file if ASR also fails).
        return Ok(ctx);
    }
    if ctx.english_source.is_none() && ctx.pending_ocr.is_none() {
        tracing::info!(
            "{}: no English subtitle source — will skip",
            ctx.video_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        );
        return Err(PipelineError::NoEnglishSource);
    }

    if let Some(source) = ctx.english_source.clone() {
        let lines = SubtitleProcessor::extract_dialogue_lines(&source)?;
        if lines.is_empty() {
            return Err(PipelineError::Stage(format!(
                "No dialogue lines found in {}",
                source.file_name().unwrap_or_default().to_string_lossy()
            )));
        }
        tracing::info!(
            "English source: {} ({} lines)",
            source.file_name().unwrap_or_default().to_string_lossy(),
            lines.len()
        );
        ctx.dialogue_lines = Some(lines);
    }

    Ok(ctx)
}

/// Find and extract a text-based (non-image) English track.
///
/// Returns the output path
/// or `None` if no text track is found.
fn extract_text_only(ctx: &PipelineContext) -> Result<Option<PathBuf>> {
    let extractor = SubtitleExtractor::new();
    let track_info = match extractor.get_track_info(&ctx.video_path) {
        Ok(ti) => ti,
        Err(_) => return Ok(None),
    };

    let eng_track = match extractor.find_english_track(&track_info) {
        Some(t) => t,
        None => return Ok(None),
    };

    if is_image_codec(&eng_track) {
        return Ok(None);
    }

    let subtitle_ext = extractor.get_subtitle_extension(&eng_track);
    let stem = ctx
        .video_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let output = ctx.work_dir.join(format!("{stem}_extracted{subtitle_ext}"));
    extractor.extract_subtitle(
        &ctx.video_path,
        eng_track.id,
        &output,
        Some(eng_track.subtitle_index),
    )?;
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::{DialogueLine, FetchedSubtitle, PipelineConfig};
    use std::collections::HashMap;

    fn base_ctx(tmp: &std::path::Path) -> PipelineContext {
        let video = tmp.join("ep01.mkv");
        std::fs::write(&video, b"fake").unwrap();
        let work = tmp.join("work");
        std::fs::create_dir_all(&work).unwrap();
        PipelineContext::new(video, work, PipelineConfig::default())
    }

    fn write_srt(path: &std::path::Path) {
        std::fs::write(path, "1\n00:00:01,000 --> 00:00:02,000\nHello there\n").unwrap();
    }

    #[test]
    fn prefers_fetched_english_and_extracts_lines() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = base_ctx(dir.path());
        let eng = dir.path().join("fetched_eng.srt");
        write_srt(&eng);
        let mut m = HashMap::new();
        m.insert(
            "eng".to_string(),
            vec![FetchedSubtitle {
                path: eng.clone(),
                source: "opensubtitles".into(),
            }],
        );
        ctx.fetched_subtitles = Some(m);

        let result = run_with_probe(ctx, || false).unwrap();
        assert_eq!(result.english_source.as_deref(), Some(eng.as_path()));
        assert_eq!(
            result.dialogue_lines,
            Some(vec![DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "Hello there".into(),
            }])
        );
    }

    #[test]
    fn falls_back_to_reference_path() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = base_ctx(dir.path());
        let reference = dir.path().join("ref.srt");
        write_srt(&reference);
        ctx.reference_path = Some(reference.clone());

        let result = run_with_probe(ctx, || false).unwrap();
        assert_eq!(result.english_source.as_deref(), Some(reference.as_path()));
        assert!(result.dialogue_lines.is_some());
    }

    #[test]
    fn no_source_no_vision_errors() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = base_ctx(dir.path());
        let err = run_with_probe(ctx, || false).unwrap_err();
        // Fix #5: this MUST be the dedicated NoEnglishSource variant so the
        // orchestrator can route it to `Skipped (no subtitles)` rather than
        // `Failed`. Catching it as a generic `Stage(_)` would still mark the
        // file failed.
        assert!(matches!(err, PipelineError::NoEnglishSource));
        assert!(err.is_no_english_source());
    }

    #[test]
    fn no_source_with_vision_defers_burned_in() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = base_ctx(dir.path());
        let result = run_with_probe(ctx, || true).unwrap();
        let pending = result.pending_ocr.expect("pending");
        assert_eq!(pending.r#type, "burned_in");
        assert!(result.dialogue_lines.is_none());
    }

    #[test]
    fn transcription_enabled_skips_burned_in_and_defers_to_asr() {
        // --transcribe says "source English from the audio": don't OCR video
        // frames (clean BDs yield credit-text junk that would preempt ASR),
        // and don't bail NoEnglishSource — the transcribe stage runs later.
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = base_ctx(dir.path());
        ctx.config.enable_transcription = true;
        let result = run_with_probe(ctx, || true).unwrap();
        assert!(result.pending_ocr.is_none());
        assert!(result.english_source.is_none());
        assert!(result.dialogue_lines.is_none());
    }

    #[test]
    fn already_probed_does_not_defer() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = base_ctx(dir.path());
        ctx.burned_in_probed = true;
        let err = run_with_probe(ctx, || true).unwrap_err();
        // No source, no pending → the no-subs sentinel (skip-not-fail).
        assert!(matches!(err, PipelineError::NoEnglishSource));
    }
}
