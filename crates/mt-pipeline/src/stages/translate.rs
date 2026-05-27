//! AI translation and font-checking stage.
//!
//! Port of `movie_translator/stages/translate.py`.
//!
//! The Python stage runs font checking and translation concurrently. Here the
//! GPU-bound translation is routed through a [`GpuExecutor`] (the synchronous
//! pipeline uses `DirectGpuExecutor`; the async pipeline will serialise through
//! the GPU queue), while font checking runs inline. The primary model must
//! produce output; extra models are best-effort.

use mt_core::{FontInfo, PipelineContext};
use mt_media::{
    check_embedded_fonts_support_polish, find_system_font_for_polish, get_ass_font_names,
};
use mt_ml::TranslateRequest;

use crate::error::{PipelineError, Result};
use crate::gpu::GpuExecutor;
use crate::proper_nouns::extract_proper_nouns_from_subtitles;

/// Stage role name (matches the Python `TranslateStage.name`).
pub const NAME: &str = "translate";

/// Check font support for Polish characters.
///
/// Port of `TranslateStage.check_fonts`. IO-bound; safe to run inline.
pub fn check_fonts(ctx: &PipelineContext) -> FontInfo {
    let english_source = ctx
        .english_source
        .as_ref()
        .expect("check_fonts requires english_source");

    // Treat a probe failure as "no embedded Polish support" (conservative).
    if check_embedded_fonts_support_polish(&ctx.video_path, english_source).unwrap_or(false) {
        return FontInfo {
            supports_polish: true,
            font_attachments: Vec::new(),
            fallback_font_family: None,
        };
    }

    let is_mkv = ctx
        .video_path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("mkv"))
        .unwrap_or(false);

    if is_mkv {
        let names = get_ass_font_names(english_source);
        if let Some((font_path, family)) = find_system_font_for_polish(&names) {
            // No fallback rename needed if the system font already matches one
            // of the ASS font names (case-insensitive).
            let fallback = if names.iter().any(|n| n.eq_ignore_ascii_case(&family)) {
                None
            } else {
                Some(family)
            };
            return FontInfo {
                supports_polish: false,
                font_attachments: vec![font_path],
                fallback_font_family: fallback,
            };
        }
    }

    FontInfo {
        supports_polish: false,
        font_attachments: Vec::new(),
        fallback_font_family: None,
    }
}

/// Run the translate stage.
///
/// Port of `TranslateStage.run`. Performs the font check inline and routes
/// translation (primary + extras) through `executor`.
///
/// `proper_nouns` carries the character-name protection list. When `None`, the
/// stage derives it from the dialogue itself via
/// [`extract_proper_nouns_from_subtitles`] — mirroring
/// `movie_translator/stages/translate.py` (lines 70-75). A caller-supplied
/// override is used as-is when present.
pub fn run(
    ctx: PipelineContext,
    executor: &dyn GpuExecutor,
    proper_nouns: Option<Vec<String>>,
) -> Result<PipelineContext> {
    let mut ctx = ctx;
    let dialogue_lines = ctx
        .dialogue_lines
        .clone()
        .ok_or_else(|| PipelineError::Stage("translate: dialogue_lines is None".into()))?;
    if ctx.english_source.is_none() {
        return Err(PipelineError::Stage(
            "translate: english_source is None".into(),
        ));
    }

    let total = dialogue_lines.len();
    tracing::info!("Translating {total} lines...");

    // Detect character names from dialogue for translation protection, unless
    // the caller supplied an explicit override.
    let proper_nouns: Option<Vec<String>> = proper_nouns.or_else(|| {
        let texts: Vec<String> = dialogue_lines.iter().map(|l| l.text.clone()).collect();
        let names = extract_proper_nouns_from_subtitles(&texts);
        if names.is_empty() {
            None
        } else {
            let mut v: Vec<String> = names.into_iter().collect();
            v.sort();
            Some(v)
        }
    });

    // Font check (inline; concurrent with translation in the async pipeline).
    ctx.font_info = Some(check_fonts(&ctx));

    // Primary translation — must produce output.
    let primary_req = TranslateRequest {
        lines: dialogue_lines.clone(),
        device: ctx.config.device.clone(),
        batch_size: ctx.config.batch_size,
        model: ctx.config.model.clone(),
        proper_nouns: proper_nouns.clone(),
    };
    let translated = executor.translate(&primary_req)?;
    if translated.is_empty() {
        return Err(PipelineError::Stage(
            "Translation failed — empty result".into(),
        ));
    }
    ctx.translated_lines = Some(translated);

    // Extra backends — best-effort; each emits an additional Polish track.
    for extra in ctx.config.extra_models.clone() {
        let req = TranslateRequest {
            lines: dialogue_lines.clone(),
            device: ctx.config.device.clone(),
            batch_size: ctx.config.batch_size,
            model: extra.clone(),
            proper_nouns: proper_nouns.clone(),
        };
        match executor.translate(&req) {
            Ok(lines) if !lines.is_empty() => {
                ctx.extra_translations.insert(extra, lines);
            }
            Ok(_) => {
                tracing::warn!("Extra model {extra:?} produced no output, dropping track");
            }
            Err(e) => {
                tracing::warn!("Extra model {extra:?} failed: {e}; skipping track");
            }
        }
    }

    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::DirectGpuExecutor;
    use mt_core::{BurnedInResult, DialogueLine, OCRResult, PipelineConfig};
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};

    /// Fake executor returning canned translations per model.
    struct FakeGpu {
        primary: Vec<DialogueLine>,
        extra: std::collections::HashMap<String, Vec<DialogueLine>>,
        seen_models: RefCell<Vec<String>>,
        seen_proper_nouns: RefCell<Option<Vec<String>>>,
    }

    impl GpuExecutor for FakeGpu {
        fn translate(&self, req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
            self.seen_models.borrow_mut().push(req.model.clone());
            *self.seen_proper_nouns.borrow_mut() = req.proper_nouns.clone();
            if req.model == "allegro" {
                Ok(self.primary.clone())
            } else {
                Ok(self.extra.get(&req.model).cloned().unwrap_or_default())
            }
        }
        fn ocr_pgs(&self, _v: &Path, _t: u32, _w: &Path) -> Result<Option<PathBuf>> {
            unreachable!()
        }
        fn ocr_burned_in(&self, _v: &Path, _o: &Path, _c: f64, _f: u32) -> Result<BurnedInResult> {
            unreachable!()
        }
        fn inpaint(
            &self,
            _v: &Path,
            out: &Path,
            _d: &str,
            _b: &str,
            _o: &[OCRResult],
        ) -> Result<PathBuf> {
            Ok(out.to_path_buf())
        }
    }

    fn ctx(tmp: &Path, extra_models: Vec<String>) -> PipelineContext {
        let video = tmp.join("ep01.mp4"); // .mp4 so font check skips the mkv branch
        std::fs::write(&video, b"fake").unwrap();
        let eng = tmp.join("eng.srt");
        std::fs::write(&eng, "1\n00:00:01,000 --> 00:00:02,000\nHello\n").unwrap();
        let config = PipelineConfig {
            extra_models,
            ..Default::default()
        };
        let mut c = PipelineContext::new(video, tmp.join("work"), config);
        c.english_source = Some(eng);
        c.dialogue_lines = Some(vec![DialogueLine {
            start_ms: 1000,
            end_ms: 2000,
            text: "Hello".into(),
        }]);
        c
    }

    #[test]
    fn primary_translation_sets_translated_lines() {
        let dir = tempfile::tempdir().unwrap();
        let gpu = FakeGpu {
            primary: vec![DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "Cześć".into(),
            }],
            extra: std::collections::HashMap::new(),
            seen_models: RefCell::new(vec![]),
            seen_proper_nouns: RefCell::new(None),
        };
        let result = run(ctx(dir.path(), vec![]), &gpu, Some(vec!["Luffy".into()])).unwrap();
        assert_eq!(result.translated_lines.unwrap()[0].text, "Cześć");
        assert!(result.font_info.is_some());
        assert_eq!(
            gpu.seen_proper_nouns.borrow().as_deref(),
            Some(&["Luffy".to_string()][..])
        );
    }

    #[test]
    fn empty_primary_result_errors() {
        let dir = tempfile::tempdir().unwrap();
        let gpu = FakeGpu {
            primary: vec![],
            extra: std::collections::HashMap::new(),
            seen_models: RefCell::new(vec![]),
            seen_proper_nouns: RefCell::new(None),
        };
        let err = run(ctx(dir.path(), vec![]), &gpu, None).unwrap_err();
        assert!(matches!(err, PipelineError::Stage(_)));
    }

    #[test]
    fn extra_model_emits_extra_translation() {
        let dir = tempfile::tempdir().unwrap();
        let mut extra = std::collections::HashMap::new();
        extra.insert(
            "apple".to_string(),
            vec![DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "Cześć (apple)".into(),
            }],
        );
        let gpu = FakeGpu {
            primary: vec![DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "Cześć".into(),
            }],
            extra,
            seen_models: RefCell::new(vec![]),
            seen_proper_nouns: RefCell::new(None),
        };
        let result = run(ctx(dir.path(), vec!["apple".into()]), &gpu, None).unwrap();
        assert_eq!(result.extra_translations["apple"][0].text, "Cześć (apple)");
        assert_eq!(gpu.seen_models.borrow().as_slice(), &["allegro", "apple"]);
    }

    #[test]
    fn extra_model_failure_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        // "apple" returns empty (no entry) → dropped, not fatal.
        let gpu = FakeGpu {
            primary: vec![DialogueLine {
                start_ms: 1000,
                end_ms: 2000,
                text: "Cześć".into(),
            }],
            extra: std::collections::HashMap::new(),
            seen_models: RefCell::new(vec![]),
            seen_proper_nouns: RefCell::new(None),
        };
        let result = run(ctx(dir.path(), vec!["apple".into()]), &gpu, None).unwrap();
        assert!(result.extra_translations.is_empty());
        assert!(result.translated_lines.is_some());
    }

    #[test]
    fn proper_nouns_derived_from_dialogue_when_none() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = ctx(dir.path(), vec![]);
        // Repeated capitalized name in direct address → derived as a proper noun.
        c.dialogue_lines = Some(vec![
            DialogueLine {
                start_ms: 0,
                end_ms: 1,
                text: "Guts! Get up.".into(),
            },
            DialogueLine {
                start_ms: 1,
                end_ms: 2,
                text: "We follow Guts into battle.".into(),
            },
            DialogueLine {
                start_ms: 2,
                end_ms: 3,
                text: "I trust Guts with my life.".into(),
            },
        ]);
        let gpu = FakeGpu {
            primary: vec![DialogueLine {
                start_ms: 0,
                end_ms: 1,
                text: "x".into(),
            }],
            extra: std::collections::HashMap::new(),
            seen_models: RefCell::new(vec![]),
            seen_proper_nouns: RefCell::new(None),
        };
        // None → stage derives proper nouns itself.
        run(c, &gpu, None).unwrap();
        assert_eq!(
            gpu.seen_proper_nouns.borrow().as_deref(),
            Some(&["Guts".to_string()][..])
        );
    }

    #[test]
    fn caller_override_takes_precedence() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = ctx(dir.path(), vec![]);
        c.dialogue_lines = Some(vec![DialogueLine {
            start_ms: 0,
            end_ms: 1,
            text: "Guts! Guts! Guts!".into(),
        }]);
        let gpu = FakeGpu {
            primary: vec![DialogueLine {
                start_ms: 0,
                end_ms: 1,
                text: "x".into(),
            }],
            extra: std::collections::HashMap::new(),
            seen_models: RefCell::new(vec![]),
            seen_proper_nouns: RefCell::new(None),
        };
        run(c, &gpu, Some(vec!["Override".into()])).unwrap();
        assert_eq!(
            gpu.seen_proper_nouns.borrow().as_deref(),
            Some(&["Override".to_string()][..])
        );
    }

    /// Integration test: translate via the real `DirectGpuExecutor`, which
    /// drives `ml/translate.py`. Live model load → `#[ignore]`.
    #[test]
    #[ignore = "requires translation model"]
    fn run_via_direct_executor() {
        let dir = tempfile::tempdir().unwrap();
        let executor = DirectGpuExecutor::new();
        let _ = run(ctx(dir.path(), vec![]), &executor, None);
    }
}
