//! Extract reference subtitle and record the original English track info.

use mt_core::{OriginalTrack, PendingOcr, PipelineContext};
use mt_media::{SubtitleExtractor, SubtitleTrack};

use crate::error::Result;
use crate::vision::{default_vision_ocr_probe, VisionOcrProbe};

/// Stage role name.
pub const NAME: &str = "extract_reference";

/// PGS/DVD/DVB image-based codecs that need OCR extraction.
const IMAGE_CODECS: &[&str] = &["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"];

fn is_image_codec(track: &SubtitleTrack) -> bool {
    let codec = track.codec.to_ascii_lowercase();
    IMAGE_CODECS
        .iter()
        .any(|c| codec == *c || codec.starts_with(c))
}

/// Extract the English reference subtitle track.
///
/// Text-based tracks are extracted
/// directly; image-based (PGS/DVD) tracks set `ctx.pending_ocr` to defer OCR.
/// If no track is found and Vision OCR is available, a burned-in OCR pass is
/// deferred too. OCR is never run inline here.
pub fn run(ctx: PipelineContext) -> Result<PipelineContext> {
    run_with_probe(ctx, default_vision_ocr_probe)
}

/// Like [`run`], but with an injectable Vision-OCR availability probe (for tests).
pub fn run_with_probe(
    mut ctx: PipelineContext,
    vision_ocr_available: VisionOcrProbe,
) -> Result<PipelineContext> {
    let extractor = SubtitleExtractor::new();
    let ref_dir = ctx.work_dir.join("reference");
    std::fs::create_dir_all(&ref_dir)?;

    let track_info = extractor.get_track_info(&ctx.video_path).ok();
    let eng_track = track_info
        .as_ref()
        .and_then(|ti| extractor.find_english_track(ti));

    if let Some(eng_track) = eng_track {
        ctx.original_english_track = Some(OriginalTrack {
            stream_index: eng_track.id as i32,
            subtitle_index: eng_track.subtitle_index as i32,
            codec: if eng_track.codec.is_empty() {
                "unknown".to_string()
            } else {
                eng_track.codec.clone()
            },
            language: if eng_track.properties.language.is_empty() {
                "eng".to_string()
            } else {
                eng_track.properties.language.clone()
            },
        });

        if is_image_codec(&eng_track) {
            // Defer PGS/DVD OCR.
            ctx.pending_ocr = Some(PendingOcr {
                r#type: "pgs".to_string(),
                track_id: Some(eng_track.id as i32),
                output_dir: ref_dir.clone(),
            });
        } else {
            let subtitle_ext = extractor.get_subtitle_extension(&eng_track);
            let stem = ctx
                .video_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let ref_path = ref_dir.join(format!("{stem}_reference{subtitle_ext}"));
            match extractor.extract_subtitle(
                &ctx.video_path,
                eng_track.id,
                &ref_path,
                Some(eng_track.subtitle_index),
            ) {
                Ok(()) => {
                    tracing::info!(
                        "Extracted reference: {}",
                        ref_path.file_name().unwrap_or_default().to_string_lossy()
                    );
                    ctx.reference_path = Some(ref_path);
                }
                Err(e) => {
                    tracing::warn!("Failed to extract reference: {e}");
                }
            }
        }
    }

    // If no track found and Vision is available, defer burned-in OCR (see
    // `PipelineConfig::burned_in_fallback_allowed` for why --transcribe
    // suppresses this fallback).
    if ctx.reference_path.is_none()
        && ctx.pending_ocr.is_none()
        && ctx.config.burned_in_fallback_allowed()
        && vision_ocr_available()
    {
        ctx.pending_ocr = Some(PendingOcr {
            r#type: "burned_in".to_string(),
            track_id: None,
            output_dir: ref_dir,
        });
    }

    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::PipelineConfig;
    use std::path::PathBuf;

    fn ctx(tmp: &std::path::Path) -> PipelineContext {
        let video = tmp.join("ep01.mkv");
        std::fs::write(&video, b"fake").unwrap();
        PipelineContext::new(video, tmp.join("work"), PipelineConfig::default())
    }

    #[test]
    fn is_image_codec_matches() {
        let mk = |codec: &str| SubtitleTrack {
            id: 0,
            codec: codec.into(),
            subtitle_index: 0,
            properties: mt_media::TrackProperties {
                language: "eng".into(),
                track_name: String::new(),
                codec_id: codec.into(),
                forced_track: false,
            },
        };
        assert!(is_image_codec(&mk("hdmv_pgs_subtitle")));
        assert!(is_image_codec(&mk("dvd_subtitle")));
        assert!(!is_image_codec(&mk("subrip")));
        assert!(!is_image_codec(&mk("ass")));
    }

    #[test]
    fn no_track_and_no_vision_sets_none() {
        // get_track_info will fail to probe the fake video → no tracks; vision off.
        let dir = tempfile::tempdir().unwrap();
        let c = ctx(dir.path());
        let result = run_with_probe(c, || false).unwrap();
        assert!(result.reference_path.is_none());
        assert!(result.original_english_track.is_none());
        assert!(result.pending_ocr.is_none());
    }

    #[test]
    fn transcription_enabled_skips_burned_in_reference() {
        // --transcribe sources English from the audio; OCRing a clean video's
        // frames for a reference yields credit-text junk that would later be
        // adopted as the English source. Skip it.
        let dir = tempfile::tempdir().unwrap();
        let mut c = ctx(dir.path());
        c.config.enable_transcription = true;
        let result = run_with_probe(c, || true).unwrap();
        assert!(result.pending_ocr.is_none());
        assert!(result.reference_path.is_none());
    }

    #[test]
    fn no_track_with_vision_defers_burned_in() {
        let dir = tempfile::tempdir().unwrap();
        let c = ctx(dir.path());
        let result = run_with_probe(c, || true).unwrap();
        let pending = result.pending_ocr.expect("pending ocr");
        assert_eq!(pending.r#type, "burned_in");
        assert!(pending.track_id.is_none());
        assert_eq!(
            pending.output_dir,
            PathBuf::from(dir.path()).join("work").join("reference")
        );
    }
}
