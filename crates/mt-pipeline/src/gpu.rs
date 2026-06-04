//! GPU work abstraction and deferred-OCR resolution.
//!
//! GPU-bound work (translation, OCR, inpainting) is performed by the embedded
//! CPython interpreter via the [`mt_ml`] crate. To keep stages testable and to serialise
//! GPU access, stages never call these functions directly: they either go
//! through a [`GpuExecutor`] or record a [`mt_core::PendingOcr`] for the
//! orchestrator to resolve later.
//!
//! The trait methods map one-to-one onto the `mt_ml` free functions.

use std::path::{Path, PathBuf};

use mt_core::{BurnedInResult, DialogueLine, OCRResult, PipelineContext};
use mt_ml::TranslateRequest;
use mt_subtitles::SubtitleProcessor;

use crate::error::{PipelineError, Result};

/// Abstraction over GPU-bound ML work.
///
/// `DirectGpuExecutor` calls [`mt_ml`] directly (used by the synchronous
/// pipeline and integration tests). The async serialised executor that routes
/// work through a single worker is implemented in a later dispatch.
pub trait GpuExecutor {
    /// Translate dialogue lines (mirrors [`mt_ml::translate`]).
    fn translate(&self, req: &TranslateRequest) -> Result<Vec<DialogueLine>>;

    /// OCR a PGS bitmap subtitle track (mirrors [`mt_ml::ocr_pgs`]).
    fn ocr_pgs(&self, video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>>;

    /// OCR burned-in subtitles (mirrors [`mt_ml::ocr_burned_in`]).
    fn ocr_burned_in(
        &self,
        video: &Path,
        output_dir: &Path,
        crop_ratio: f64,
        fps: u32,
    ) -> Result<BurnedInResult>;

    /// Remove burned-in subtitles via inpainting (mirrors [`mt_ml::inpaint`]).
    fn inpaint(
        &self,
        video: &Path,
        output: &Path,
        device: &str,
        backend: &str,
        ocr_results: &[OCRResult],
    ) -> Result<PathBuf>;

    /// OCR burned-in subs from a downloaded hardsub video and clean them into a
    /// `.srt` (mirrors [`mt_ml::hardsub_ocr_clean`]). `None` => no usable lines.
    fn hardsub_ocr_clean(
        &self,
        video: &Path,
        out_dir: &Path,
        language: &str,
    ) -> Result<Option<PathBuf>>;

    /// Transcribe the `language` audio track to an SRT via ASR (mirrors
    /// [`mt_ml::transcribe_to_srt`]). `None` => no track / engine / lines.
    fn transcribe(
        &self,
        video: &Path,
        output_dir: &Path,
        language: &str,
        engine: &str,
    ) -> Result<Option<PathBuf>>;
}

/// A [`GpuExecutor`] that calls [`mt_ml`] inline (synchronous).
#[derive(Debug, Default, Clone, Copy)]
pub struct DirectGpuExecutor;

impl DirectGpuExecutor {
    pub fn new() -> Self {
        DirectGpuExecutor
    }
}

impl GpuExecutor for DirectGpuExecutor {
    fn translate(&self, req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
        mt_ml::translate(req).map_err(PipelineError::from)
    }

    fn ocr_pgs(&self, video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
        mt_ml::ocr_pgs(video, track_index, work_dir).map_err(PipelineError::from)
    }

    fn ocr_burned_in(
        &self,
        video: &Path,
        output_dir: &Path,
        crop_ratio: f64,
        fps: u32,
    ) -> Result<BurnedInResult> {
        mt_ml::ocr_burned_in(video, output_dir, crop_ratio, fps).map_err(PipelineError::from)
    }

    fn inpaint(
        &self,
        video: &Path,
        output: &Path,
        device: &str,
        backend: &str,
        ocr_results: &[OCRResult],
    ) -> Result<PathBuf> {
        mt_ml::inpaint(video, output, device, backend, ocr_results).map_err(PipelineError::from)
    }

    fn hardsub_ocr_clean(
        &self,
        video: &Path,
        out_dir: &Path,
        language: &str,
    ) -> Result<Option<PathBuf>> {
        mt_ml::hardsub_ocr_clean(video, out_dir, language).map_err(PipelineError::from)
    }

    fn transcribe(
        &self,
        video: &Path,
        output_dir: &Path,
        language: &str,
        engine: &str,
    ) -> Result<Option<PathBuf>> {
        mt_ml::transcribe_to_srt(video, output_dir, language, engine).map_err(PipelineError::from)
    }
}

// Burned-in OCR defaults: crop the bottom 25% of the frame, sample 1 fps.
const BURNED_IN_CROP_RATIO: f64 = 0.25;
const BURNED_IN_FPS: u32 = 1;

/// Stage label identifying which extraction stage deferred the OCR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrStageLabel {
    /// Reference extraction stage (`extract_ref`).
    ExtractRef,
    /// English source extraction stage (`extract_english`).
    ExtractEnglish,
}

/// Resolve a pending OCR task and apply its results to the context.
///
/// The OCR work runs through `executor` (so the synchronous pipeline calls
/// [`mt_ml`] inline and the async pipeline serialises through the GPU queue).
///
/// Branching:
/// - `pgs`  → `ocr_pgs`; on `Some(path)` set `reference_path` /
///   `english_source` depending on `stage_label`.
/// - `burned_in` → `ocr_burned_in`; on success set the source path AND
///   `ocr_results`. Also sets `burned_in_probed = true`.
/// - For `ExtractEnglish`, once a source is known and `dialogue_lines` is
///   still `None`, extract dialogue lines (raising on empty).
///
/// On completion `ctx.pending_ocr` is cleared. A no-op when there is no
/// pending OCR.
pub fn resolve_pending_ocr(
    ctx: &mut PipelineContext,
    executor: &dyn GpuExecutor,
    stage_label: OcrStageLabel,
) -> Result<()> {
    let pending = match ctx.pending_ocr.clone() {
        Some(p) => p,
        None => return Ok(()),
    };

    match pending.r#type.as_str() {
        "pgs" => {
            let track = pending.track_id.unwrap_or(0).max(0) as u32;
            let result = executor.ocr_pgs(&ctx.video_path, track, &pending.output_dir)?;
            if let Some(path) = result {
                match stage_label {
                    OcrStageLabel::ExtractRef => ctx.reference_path = Some(path),
                    OcrStageLabel::ExtractEnglish => ctx.english_source = Some(path),
                }
            }
        }
        "burned_in" => {
            // Mark that we have probed for burned-in subtitles.
            ctx.burned_in_probed = true;
            let result = executor.ocr_burned_in(
                &ctx.video_path,
                &pending.output_dir,
                BURNED_IN_CROP_RATIO,
                BURNED_IN_FPS,
            );
            // An empty/failed burned-in pass is non-fatal: we treat a failed
            // OCR as "nothing detected" rather than aborting the file.
            if let Ok(result) = result {
                match stage_label {
                    OcrStageLabel::ExtractRef => {
                        ctx.reference_path = Some(result.srt_path);
                        ctx.ocr_results = Some(result.ocr_results);
                    }
                    OcrStageLabel::ExtractEnglish => {
                        ctx.english_source = Some(result.srt_path);
                        ctx.ocr_results = Some(result.ocr_results);
                    }
                }
            }
        }
        other => {
            tracing::warn!("unknown pending_ocr type {other:?}; ignoring");
        }
    }

    // If extract_english deferred OCR and we now have a source, extract lines.
    if stage_label == OcrStageLabel::ExtractEnglish {
        if let Some(source) = ctx.english_source.clone() {
            if ctx.dialogue_lines.is_none() {
                let lines = SubtitleProcessor::extract_dialogue_lines(&source)?;
                if lines.is_empty() {
                    ctx.pending_ocr = None;
                    return Err(PipelineError::Stage(format!(
                        "No dialogue lines found in {}",
                        source.display()
                    )));
                }
                ctx.dialogue_lines = Some(lines);
            }
        }
    }

    ctx.pending_ocr = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::{OCRResult, PendingOcr, PipelineConfig};
    use std::cell::RefCell;

    /// A fake executor that returns canned OCR results and records calls.
    #[derive(Default)]
    struct FakeGpu {
        pgs_result: Option<PathBuf>,
        burned_in: Option<BurnedInResult>,
        calls: RefCell<Vec<String>>,
    }

    impl GpuExecutor for FakeGpu {
        fn transcribe(&self, _v: &Path, _o: &Path, _l: &str, _e: &str) -> Result<Option<PathBuf>> {
            Ok(None)
        }
        fn translate(&self, _req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
            self.calls.borrow_mut().push("translate".into());
            Ok(vec![])
        }
        fn ocr_pgs(&self, _v: &Path, _t: u32, _w: &Path) -> Result<Option<PathBuf>> {
            self.calls.borrow_mut().push("ocr_pgs".into());
            Ok(self.pgs_result.clone())
        }
        fn ocr_burned_in(&self, _v: &Path, _o: &Path, _c: f64, _f: u32) -> Result<BurnedInResult> {
            self.calls.borrow_mut().push("ocr_burned_in".into());
            self.burned_in
                .clone()
                .ok_or_else(|| PipelineError::Stage("no burned-in result".into()))
        }
        fn inpaint(
            &self,
            _v: &Path,
            out: &Path,
            _d: &str,
            _b: &str,
            _o: &[OCRResult],
        ) -> Result<PathBuf> {
            self.calls.borrow_mut().push("inpaint".into());
            Ok(out.to_path_buf())
        }
        fn hardsub_ocr_clean(&self, _v: &Path, _o: &Path, _l: &str) -> Result<Option<PathBuf>> {
            self.calls.borrow_mut().push("hardsub_ocr_clean".into());
            Ok(None)
        }
    }

    fn ctx_with_pending(pending: PendingOcr) -> PipelineContext {
        let mut ctx = PipelineContext::new(
            PathBuf::from("/tmp/ep.mkv"),
            PathBuf::from("/tmp/work"),
            PipelineConfig::default(),
        );
        ctx.pending_ocr = Some(pending);
        ctx
    }

    #[test]
    fn no_pending_is_noop() {
        let mut ctx = PipelineContext::new(
            PathBuf::from("/tmp/ep.mkv"),
            PathBuf::from("/tmp/work"),
            PipelineConfig::default(),
        );
        let gpu = FakeGpu::default();
        resolve_pending_ocr(&mut ctx, &gpu, OcrStageLabel::ExtractRef).unwrap();
        assert!(gpu.calls.borrow().is_empty());
    }

    #[test]
    fn pgs_sets_reference_path_for_extract_ref() {
        let gpu = FakeGpu {
            pgs_result: Some(PathBuf::from("/tmp/ref.srt")),
            ..Default::default()
        };
        let mut ctx = ctx_with_pending(PendingOcr {
            r#type: "pgs".into(),
            track_id: Some(3),
            output_dir: PathBuf::from("/tmp/ref"),
        });
        resolve_pending_ocr(&mut ctx, &gpu, OcrStageLabel::ExtractRef).unwrap();
        assert_eq!(
            ctx.reference_path.as_deref(),
            Some(Path::new("/tmp/ref.srt"))
        );
        assert!(ctx.pending_ocr.is_none());
    }

    #[test]
    fn pgs_sets_english_source_for_extract_english() {
        // No dialogue extraction attempted because the file doesn't exist;
        // here we use a None pgs result so english_source stays None.
        let gpu = FakeGpu {
            pgs_result: None,
            ..Default::default()
        };
        let mut ctx = ctx_with_pending(PendingOcr {
            r#type: "pgs".into(),
            track_id: None,
            output_dir: PathBuf::from("/tmp/wd"),
        });
        resolve_pending_ocr(&mut ctx, &gpu, OcrStageLabel::ExtractEnglish).unwrap();
        assert!(ctx.english_source.is_none());
        assert_eq!(gpu.calls.borrow().as_slice(), &["ocr_pgs"]);
    }

    #[test]
    fn burned_in_sets_source_and_ocr_results() {
        let gpu = FakeGpu {
            burned_in: Some(BurnedInResult {
                srt_path: PathBuf::from("/tmp/burn.srt"),
                ocr_results: vec![OCRResult {
                    timestamp_ms: 1000,
                    text: "hi".into(),
                    boxes: vec![],
                }],
            }),
            ..Default::default()
        };
        let mut ctx = ctx_with_pending(PendingOcr {
            r#type: "burned_in".into(),
            track_id: None,
            output_dir: PathBuf::from("/tmp/wd"),
        });
        resolve_pending_ocr(&mut ctx, &gpu, OcrStageLabel::ExtractRef).unwrap();
        assert_eq!(
            ctx.reference_path.as_deref(),
            Some(Path::new("/tmp/burn.srt"))
        );
        assert_eq!(ctx.ocr_results.as_ref().unwrap().len(), 1);
        assert!(ctx.burned_in_probed);
    }

    #[test]
    fn burned_in_failure_is_nonfatal() {
        let gpu = FakeGpu::default(); // ocr_burned_in returns Err
        let mut ctx = ctx_with_pending(PendingOcr {
            r#type: "burned_in".into(),
            track_id: None,
            output_dir: PathBuf::from("/tmp/wd"),
        });
        resolve_pending_ocr(&mut ctx, &gpu, OcrStageLabel::ExtractEnglish).unwrap();
        assert!(ctx.english_source.is_none());
        assert!(ctx.burned_in_probed);
        assert!(ctx.pending_ocr.is_none());
    }
}
