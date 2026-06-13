//! Stage 4.6 — ASR transcription fallback (gated by `--transcribe`).
//!
//! Runs only when extract_english (plus burned-in OCR) produced NO English
//! source: transcribes the English **audio** track via the configured engine
//! ("apple" SpeechAnalyzer or "whisper" mlx large-v3 — see
//! `benchmarks/asr/REPORT.md`) and adopts the resulting SRT as the English
//! source. A missing audio track / unavailable engine yields no source and the
//! pipeline falls through to its normal `NoEnglishSource` skip.

#[cfg(test)]
use std::fs;

use mt_core::PipelineContext;
use mt_subtitles::extract_dialogue_lines;
#[cfg(test)]
use tempfile::tempdir;
use tracing::{info, warn};

use crate::error::Result;
use crate::gpu::GpuExecutor;

pub const NAME: &str = "transcribe";

pub fn run(ctx: PipelineContext, executor: &dyn GpuExecutor) -> Result<PipelineContext> {
    let mut ctx = ctx;
    if ctx.english_source.is_some() || !ctx.config.enable_transcription {
        return Ok(ctx);
    }

    let srt = executor.transcribe(
        &ctx.video_path,
        &ctx.work_dir,
        "en",
        &ctx.config.transcribe_engine,
    )?;
    let Some(srt) = srt else {
        info!("transcribe: no English audio transcription available");
        return Ok(ctx);
    };

    let lines = extract_dialogue_lines(&srt)?;
    if lines.is_empty() {
        warn!("transcribe: SRT parsed to zero dialogue lines");
        return Ok(ctx);
    }
    info!(
        "transcribe: {} lines from the English audio track ({})",
        lines.len(),
        ctx.config.transcribe_engine
    );
    ctx.english_source = Some(srt);
    ctx.dialogue_lines = Some(lines);
    ctx.english_from_asr = true;
    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use mt_core::{BurnedInResult, DialogueLine, OCRResult, PipelineConfig, PipelineContext};
    use mt_ml::TranslateRequest;

    use super::*;

    /// Executor that counts transcribe calls and returns a canned response.
    #[derive(Default)]
    struct SpyExec {
        calls: AtomicUsize,
        result: Option<PathBuf>,
    }

    impl GpuExecutor for SpyExec {
        fn translate(&self, _req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
            unimplemented!()
        }
        fn ocr_pgs(&self, _v: &Path, _t: u32, _w: &Path) -> Result<Option<PathBuf>> {
            unimplemented!()
        }
        fn ocr_burned_in(&self, _v: &Path, _o: &Path, _c: f64, _f: u32) -> Result<BurnedInResult> {
            unimplemented!()
        }
        fn inpaint(
            &self,
            _v: &Path,
            _o: &Path,
            _d: &str,
            _b: &str,
            _r: &[OCRResult],
        ) -> Result<PathBuf> {
            unimplemented!()
        }
        fn hardsub_ocr_clean(&self, _v: &Path, _o: &Path, _l: &str) -> Result<Option<PathBuf>> {
            unimplemented!()
        }
        fn transcribe(&self, _v: &Path, _o: &Path, _l: &str, _e: &str) -> Result<Option<PathBuf>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.result.clone())
        }
    }

    fn ctx_with(enable: bool, english: Option<PathBuf>) -> PipelineContext {
        let config = PipelineConfig {
            enable_transcription: enable,
            ..PipelineConfig::default()
        };
        let mut ctx = PipelineContext::new(
            PathBuf::from("/tmp/x.mkv"),
            PathBuf::from("/tmp/wd"),
            config,
        );
        ctx.english_source = english;
        ctx
    }

    #[test]
    fn skips_when_disabled() {
        let exec = SpyExec::default();
        let ctx = run(ctx_with(false, None), &exec).unwrap();
        assert_eq!(exec.calls.load(Ordering::SeqCst), 0);
        assert!(ctx.english_source.is_none());
    }

    #[test]
    fn skips_when_english_source_already_exists() {
        let exec = SpyExec::default();
        let src = PathBuf::from("/tmp/eng.srt");
        let ctx = run(ctx_with(true, Some(src.clone())), &exec).unwrap();
        assert_eq!(exec.calls.load(Ordering::SeqCst), 0);
        assert_eq!(ctx.english_source, Some(src));
    }

    #[test]
    fn adopts_transcribed_srt_and_marks_asr_provenance() {
        let dir = tempdir().unwrap();
        let srt = dir.path().join("transcribed_en.srt");
        fs::write(&srt, "1\n00:00:01,000 --> 00:00:02,000\nHello there\n").unwrap();
        let exec = SpyExec {
            result: Some(srt.clone()),
            ..SpyExec::default()
        };
        let ctx = run(ctx_with(true, None), &exec).unwrap();
        assert_eq!(ctx.english_source, Some(srt));
        assert!(ctx.english_from_asr, "ASR provenance must be marked");
        assert!(!ctx.dialogue_lines.unwrap().is_empty());
    }

    #[test]
    fn none_from_engine_leaves_ctx_unchanged() {
        let exec = SpyExec::default();
        let ctx = run(ctx_with(true, None), &exec).unwrap();
        assert_eq!(exec.calls.load(Ordering::SeqCst), 1);
        assert!(ctx.english_source.is_none());
        assert!(ctx.dialogue_lines.is_none());
    }
}
