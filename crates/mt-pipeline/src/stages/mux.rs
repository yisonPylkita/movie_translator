//! Final video muxing stage — combines video with subtitle tracks.

use std::fs;
use std::path::{Path, PathBuf};

use mt_core::{PipelineContext, SubtitleFile};
use mt_media::VideoOperations;
use tracing::{error, info};

use crate::error::{PipelineError, Result};
use crate::gpu::GpuExecutor;

/// Stage role name.
pub const NAME: &str = "mux";

/// Suffix marker for the in-place mux output sitting next to the original.
/// Format: `<stem>.translating<suffix>` e.g. `Episode01.translating.mkv`.
pub const IN_PLACE_TEMP_MARKER: &str = ".translating";

/// Return the sibling path used for the in-place mux temp file.
pub fn in_place_temp_path(video_path: &Path) -> PathBuf {
    let stem = video_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = video_path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    video_path.with_file_name(format!("{stem}{IN_PLACE_TEMP_MARKER}{ext}"))
}

/// Abstraction over the muxing back-end so the decision logic is testable
/// without invoking ffmpeg. `FfmpegMuxOps` is the production impl backed by
/// [`mt_media::VideoOperations`]; tests substitute a fake.
pub trait MuxOps {
    /// Mux `source` with the given subtitle files (and optional fonts) into `output`.
    fn create_clean_video(
        &self,
        source: &Path,
        subtitle_files: &[SubtitleFile],
        output: &Path,
        font_attachments: Option<&[PathBuf]>,
        original_sub_index: Option<usize>,
        original_sub_title: Option<&str>,
    ) -> Result<()>;

    /// Verify the muxed `output` contains the expected tracks.
    fn verify_result(&self, output: &Path, expected_tracks: Option<&[SubtitleFile]>) -> Result<()>;
}

/// Production [`MuxOps`] backed by ffmpeg via [`mt_media::VideoOperations`].
#[derive(Debug, Default, Clone, Copy)]
pub struct FfmpegMuxOps;

impl MuxOps for FfmpegMuxOps {
    fn create_clean_video(
        &self,
        source: &Path,
        subtitle_files: &[SubtitleFile],
        output: &Path,
        font_attachments: Option<&[PathBuf]>,
        original_sub_index: Option<usize>,
        original_sub_title: Option<&str>,
    ) -> Result<()> {
        VideoOperations::new()
            .create_clean_video(
                source,
                subtitle_files,
                output,
                font_attachments,
                original_sub_index,
                original_sub_title,
            )
            .map_err(PipelineError::from)
    }

    fn verify_result(&self, output: &Path, expected_tracks: Option<&[SubtitleFile]>) -> Result<()> {
        VideoOperations::new()
            .verify_result(output, expected_tracks)
            .map(|_| ())
            .map_err(PipelineError::from)
    }
}

/// Run the mux stage with the production ffmpeg back-end.
///
/// Optionally inpaints burned-in subtitles (via
/// `executor`), muxes subtitle tracks onto the (possibly inpainted) video,
/// verifies the output, and replaces the original (atomically for in-place).
pub fn run(ctx: PipelineContext, executor: &dyn GpuExecutor) -> Result<PipelineContext> {
    run_with_ops(ctx, executor, &FfmpegMuxOps)
}

/// Like [`run`], with an injectable [`MuxOps`] back-end (for tests).
pub fn run_with_ops(
    mut ctx: PipelineContext,
    executor: &dyn GpuExecutor,
    ops: &dyn MuxOps,
) -> Result<PipelineContext> {
    // Inpaint burned-in subtitles if OCR was used and inpainting is enabled.
    let mut source_video = ctx.video_path.clone();
    let has_ocr = ctx.ocr_results.as_ref().is_some_and(|r| !r.is_empty());
    let stem = ctx
        .video_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = ctx
        .video_path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();

    if has_ocr && ctx.config.enable_inpaint && ctx.inpainted_video.is_none() {
        info!("Removing burned-in subtitles via inpainting...");
        let inpainted = ctx.work_dir.join(format!("{stem}_inpainted{ext}"));
        let ocr_results = ctx.ocr_results.clone().unwrap_or_default();
        executor.inpaint(
            &ctx.video_path,
            &inpainted,
            &ctx.config.device,
            "lama",
            &ocr_results,
        )?;
        ctx.inpainted_video = Some(inpainted.clone());
        source_video = inpainted;
    } else if let Some(inpainted) = &ctx.inpainted_video {
        source_video = inpainted.clone();
    }

    // Determine original track preservation.
    let original_sub_index = ctx
        .original_english_track
        .as_ref()
        .map(|t| t.subtitle_index.max(0) as usize);
    let original_sub_title = original_sub_index.map(|_| "English (Original)");

    let subtitle_tracks = ctx
        .subtitle_tracks
        .clone()
        .expect("mux requires subtitle_tracks");
    let font_info = ctx.font_info.clone().expect("mux requires font_info");

    // Choose temp location.
    let temp_video = if ctx.config.in_place {
        in_place_temp_path(&ctx.video_path)
    } else {
        ctx.work_dir.join(format!("{stem}_temp{ext}"))
    };

    let font_attachments: Option<&[PathBuf]> = if font_info.font_attachments.is_empty() {
        None
    } else {
        Some(&font_info.font_attachments)
    };

    let result = (|| -> Result<()> {
        ops.create_clean_video(
            &source_video,
            &subtitle_tracks,
            &temp_video,
            font_attachments,
            original_sub_index,
            original_sub_title,
        )?;

        // Build full expected track list including preserved original.
        let mut expected_tracks = subtitle_tracks.clone();
        if original_sub_index.is_some() {
            let lang = ctx
                .original_english_track
                .as_ref()
                .map(|t| t.language.clone())
                .unwrap_or_else(|| "eng".to_string());
            expected_tracks.insert(
                0,
                SubtitleFile {
                    path: PathBuf::new(),
                    language: lang,
                    title: original_sub_title
                        .unwrap_or("English (Original)")
                        .to_string(),
                    is_default: false,
                },
            );
        }
        ops.verify_result(&temp_video, Some(&expected_tracks))?;

        if !ctx.config.dry_run {
            if ctx.config.in_place {
                replace_in_place(&ctx.video_path, &temp_video, ops)?;
                if let Some(inpainted) = &ctx.inpainted_video
                    && inpainted.exists()
                {
                    let _ = fs::remove_file(inpainted);
                }
            } else {
                replace_original(&ctx.video_path, &temp_video, ops)?;
            }
        }
        Ok(())
    })();

    if result.is_err() && temp_video.exists() {
        let _ = fs::remove_file(&temp_video);
    }
    result?;

    Ok(ctx)
}

/// Replace the original with a backup-and-move strategy.
///
/// Ordered for data integrity:
///
/// 1. **Verify the muxed temp file first** — so the original is never replaced
///    by output that fails verification.
/// 2. Back up the original, rename the verified temp over it, then re-verify
///    in place as defence-in-depth.
/// 3. If the in-place verification fails, **restore the original from the
///    backup** before returning the error.
///
/// Net guarantees: (a) the original is never left replaced by unverified
/// output; (b) the backup is always either restored or removed — never
/// orphaned.
fn replace_original(video_path: &Path, temp_video: &Path, ops: &dyn MuxOps) -> Result<()> {
    // (1) Verify the muxed output *before* touching the original.
    ops.verify_result(temp_video, None)?;

    let backup_path = {
        let mut s = video_path.as_os_str().to_os_string();
        s.push(".backup");
        PathBuf::from(s)
    };
    fs::copy(video_path, &backup_path)?;

    // (2) Swap in the verified output, then re-verify in place.
    let outcome = (|| -> Result<()> {
        fs::rename(temp_video, video_path)?;
        ops.verify_result(video_path, None)?;
        Ok(())
    })();

    match outcome {
        Ok(()) => {
            // Success: drop the backup (never orphan it).
            fs::remove_file(&backup_path)?;
            Ok(())
        }
        Err(e) => {
            // (3) Restore the original from backup, then clean the backup up so
            // it's never left orphaned. Restore is best-effort but reported.
            if video_path.exists() {
                let _ = fs::remove_file(video_path);
            }
            if backup_path.exists()
                && let Err(rename_err) = fs::rename(&backup_path, video_path)
            {
                // The muxed file was already removed but we couldn't move the
                // backup back into place. No data is lost — the original is
                // preserved at the `.backup` path — but it isn't where the
                // user expects it. Tell them so manual recovery is possible.
                error!(
                    "failed to restore original from backup ({rename_err}); \
                         your original is preserved at {} — rename it back to {} \
                         to recover",
                    backup_path.display(),
                    video_path.display(),
                );
                return Err(PipelineError::Io(rename_err));
            }
            Err(e)
        }
    }
}

/// In-place replace: verify the muxed temp first, then atomically rename it
/// over the original.
///
/// `fs::rename` is atomic on the same
/// filesystem (POSIX `os.replace`). There is no separate backup file (peak
/// disk use stays <=2x original); safety comes from verifying the temp output
/// *before* the rename, so a verification failure leaves the original intact
/// and the temp untouched (the caller unlinks it).
fn replace_in_place(video_path: &Path, temp_video: &Path, ops: &dyn MuxOps) -> Result<()> {
    ops.verify_result(temp_video, None)?;
    fs::rename(temp_video, video_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;

    use mt_core::{
        BurnedInResult, DialogueLine, FontInfo, OCRResult, OriginalTrack, PipelineConfig,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::gpu::DirectGpuExecutor;

    // ── Fake GpuExecutor recording inpaint calls ──────────────────────────
    #[derive(Default)]
    struct RecordGpu {
        inpaint_calls: RefCell<Vec<PathBuf>>,
    }
    impl GpuExecutor for RecordGpu {
        fn transcribe(&self, _v: &Path, _o: &Path, _l: &str, _e: &str) -> Result<Option<PathBuf>> {
            Ok(None)
        }
        fn translate(&self, _r: &mt_ml::TranslateRequest) -> Result<Vec<DialogueLine>> {
            unreachable!()
        }
        fn ocr_pgs(&self, _v: &Path, _t: u32, _w: &Path) -> Result<Option<PathBuf>> {
            unreachable!()
        }
        fn ocr_burned_in(&self, _v: &Path, _o: &Path, _c: f64, _f: u32) -> Result<BurnedInResult> {
            unreachable!()
        }
        fn inpaint(
            &self,
            video: &Path,
            out: &Path,
            _d: &str,
            _b: &str,
            _o: &[OCRResult],
        ) -> Result<PathBuf> {
            self.inpaint_calls.borrow_mut().push(video.to_path_buf());
            // Simulate inpaint output creation.
            fs::write(out, b"inpainted").unwrap();
            Ok(out.to_path_buf())
        }
        fn hardsub_ocr_clean(&self, _v: &Path, _o: &Path, _l: &str) -> Result<Option<PathBuf>> {
            unreachable!()
        }
    }

    // ── Fake MuxOps recording calls and simulating output ─────────────────
    struct FakeOps {
        /// If set, every `verify_result` call fails with this message.
        verify_err: Option<String>,
        /// If set, `verify_result` succeeds for the first N calls and then fails
        /// with this message on call N+1 — used to simulate "the muxed temp
        /// verifies fine but the in-place re-verify fails after the rename".
        verify_fail_after: Option<(usize, String)>,
        verify_calls: RefCell<usize>,
        observed_outputs: RefCell<Vec<PathBuf>>,
        observed_sources: RefCell<Vec<PathBuf>>,
        observed_orig_index: RefCell<Option<usize>>,
        observed_orig_title: RefCell<Option<String>>,
        observed_fonts: RefCell<Option<Vec<PathBuf>>>,
    }
    impl FakeOps {
        fn new() -> Self {
            Self {
                verify_err: None,
                verify_fail_after: None,
                verify_calls: RefCell::new(0),
                observed_outputs: RefCell::new(vec![]),
                observed_sources: RefCell::new(vec![]),
                observed_orig_index: RefCell::new(None),
                observed_orig_title: RefCell::new(None),
                observed_fonts: RefCell::new(None),
            }
        }
    }
    impl MuxOps for FakeOps {
        fn create_clean_video(
            &self,
            source: &Path,
            _subs: &[SubtitleFile],
            output: &Path,
            font_attachments: Option<&[PathBuf]>,
            original_sub_index: Option<usize>,
            original_sub_title: Option<&str>,
        ) -> Result<()> {
            self.observed_sources
                .borrow_mut()
                .push(source.to_path_buf());
            self.observed_outputs
                .borrow_mut()
                .push(output.to_path_buf());
            *self.observed_orig_index.borrow_mut() = original_sub_index;
            *self.observed_orig_title.borrow_mut() = original_sub_title.map(|s| s.to_string());
            *self.observed_fonts.borrow_mut() = font_attachments.map(|f| f.to_vec());
            fs::write(output, b"muxed content").unwrap();
            Ok(())
        }
        fn verify_result(&self, _o: &Path, _e: Option<&[SubtitleFile]>) -> Result<()> {
            let n = {
                let mut c = self.verify_calls.borrow_mut();
                *c += 1;
                *c
            };
            if let Some(msg) = &self.verify_err {
                return Err(PipelineError::Stage(msg.clone()));
            }
            if let Some((after, msg)) = &self.verify_fail_after
                && n > *after
            {
                return Err(PipelineError::Stage(msg.clone()));
            }
            Ok(())
        }
    }

    fn make_ctx(tmp: &Path, dry_run: bool) -> PipelineContext {
        let video = tmp.join("ep01.mkv");
        fs::write(&video, b"fake video").unwrap();
        let work = tmp.join("work");
        fs::create_dir_all(&work).unwrap();
        let pol = tmp.join("pol.ass");
        fs::write(&pol, b"").unwrap();
        let config = PipelineConfig {
            dry_run,
            ..Default::default()
        };
        let mut ctx = PipelineContext::new(video, work, config);
        ctx.subtitle_tracks = Some(vec![SubtitleFile {
            path: pol,
            language: "pol".into(),
            title: "Polish (AI)".into(),
            is_default: true,
        }]);
        ctx.font_info = Some(FontInfo {
            supports_polish: true,
            font_attachments: Vec::new(),
            fallback_font_family: None,
        });
        ctx.original_english_track = Some(OriginalTrack {
            stream_index: 2,
            subtitle_index: 0,
            codec: "subrip".into(),
            language: "eng".into(),
        });
        ctx
    }

    #[test]
    fn in_place_temp_path_format() {
        assert_eq!(
            in_place_temp_path(Path::new("/x/Episode01.mkv")),
            PathBuf::from("/x/Episode01.translating.mkv")
        );
        assert_eq!(
            in_place_temp_path(Path::new("/x/show.s01e02.mp4")),
            PathBuf::from("/x/show.s01e02.translating.mp4")
        );
    }

    #[test]
    fn passes_original_track_to_mux() {
        let dir = tempdir().unwrap();
        let ctx = make_ctx(dir.path(), false);
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert_eq!(*ops.observed_orig_index.borrow(), Some(0));
        assert_eq!(
            ops.observed_orig_title.borrow().as_deref(),
            Some("English (Original)")
        );
    }

    #[test]
    fn no_original_track_passes_none() {
        let dir = tempdir().unwrap();
        let mut ctx = make_ctx(dir.path(), false);
        ctx.original_english_track = None;
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert_eq!(*ops.observed_orig_index.borrow(), None);
    }

    #[test]
    fn dry_run_does_not_replace_original() {
        let dir = tempdir().unwrap();
        let ctx = make_ctx(dir.path(), true);
        let video = ctx.video_path.clone();
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert_eq!(fs::read_to_string(&video).unwrap(), "fake video");
    }

    #[test]
    fn verify_result_failure_raises() {
        let dir = tempdir().unwrap();
        let ctx = make_ctx(dir.path(), false);
        let gpu = RecordGpu::default();
        let mut ops = FakeOps::new();
        ops.verify_err = Some("wrong track count".into());
        let err = run_with_ops(ctx, &gpu, &ops).unwrap_err();
        assert!(err.to_string().contains("wrong track count"));
    }

    #[test]
    fn font_attachments_passed_to_mux() {
        let dir = tempdir().unwrap();
        let mut ctx = make_ctx(dir.path(), false);
        let fa = dir.path().join("FontA.ttf");
        let fb = dir.path().join("FontB.otf");
        fs::write(&fa, b"").unwrap();
        fs::write(&fb, b"").unwrap();
        ctx.font_info = Some(FontInfo {
            supports_polish: true,
            font_attachments: vec![fa.clone(), fb.clone()],
            fallback_font_family: None,
        });
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert_eq!(ops.observed_fonts.borrow().as_deref(), Some(&[fa, fb][..]));
    }

    #[test]
    fn empty_font_attachments_passed_as_none() {
        let dir = tempdir().unwrap();
        let ctx = make_ctx(dir.path(), false);
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert!(ops.observed_fonts.borrow().is_none());
    }

    // ── Inpainting code path ──────────────────────────────────────────────

    #[test]
    fn inpainting_called_when_ocr_results_and_inpaint_enabled() {
        let dir = tempdir().unwrap();
        let mut ctx = make_ctx(dir.path(), false);
        ctx.config.enable_inpaint = true;
        ctx.ocr_results = Some(vec![OCRResult {
            timestamp_ms: 1000,
            text: "Hello".into(),
            boxes: vec![],
        }]);
        let video = ctx.video_path.clone();
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        let result = run_with_ops(ctx, &gpu, &ops).unwrap();
        assert_eq!(gpu.inpaint_calls.borrow().as_slice(), &[video]);
        assert!(result.inpainted_video.is_some());
    }

    #[test]
    fn inpainting_skipped_when_already_inpainted() {
        let dir = tempdir().unwrap();
        let mut ctx = make_ctx(dir.path(), false);
        ctx.config.enable_inpaint = true;
        ctx.ocr_results = Some(vec![OCRResult {
            timestamp_ms: 1000,
            text: "Hi".into(),
            boxes: vec![],
        }]);
        let pre = dir.path().join("already_inpainted.mkv");
        fs::write(&pre, b"inpainted video").unwrap();
        ctx.inpainted_video = Some(pre.clone());
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert!(gpu.inpaint_calls.borrow().is_empty());
        assert_eq!(ops.observed_sources.borrow()[0], pre);
    }

    #[test]
    fn inpainting_not_called_when_disabled() {
        let dir = tempdir().unwrap();
        let mut ctx = make_ctx(dir.path(), false);
        ctx.config.enable_inpaint = false;
        ctx.ocr_results = Some(vec![OCRResult {
            timestamp_ms: 1000,
            text: "Hi".into(),
            boxes: vec![],
        }]);
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert!(gpu.inpaint_calls.borrow().is_empty());
    }

    // ── In-place mode ─────────────────────────────────────────────────────

    fn make_in_place_ctx(tmp: &Path) -> PipelineContext {
        let mut ctx = make_ctx(tmp, false);
        ctx.config.in_place = true;
        ctx
    }

    #[test]
    fn in_place_writes_temp_beside_original_then_replaces() {
        let dir = tempdir().unwrap();
        let ctx = make_in_place_ctx(dir.path());
        let video = ctx.video_path.clone();
        let expected_temp = in_place_temp_path(&video);
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert_eq!(
            ops.observed_outputs.borrow().as_slice(),
            std::slice::from_ref(&expected_temp)
        );
        assert_eq!(fs::read_to_string(&video).unwrap(), "muxed content");
        assert!(!expected_temp.exists());
    }

    #[test]
    fn in_place_makes_no_backup() {
        let dir = tempdir().unwrap();
        let ctx = make_in_place_ctx(dir.path());
        let video = ctx.video_path.clone();
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        let mut backup = video.into_os_string();
        backup.push(".backup");
        assert!(!PathBuf::from(backup).exists());
    }

    #[test]
    fn in_place_failure_unlinks_temp() {
        let dir = tempdir().unwrap();
        let ctx = make_in_place_ctx(dir.path());
        let video = ctx.video_path.clone();
        let expected_temp = in_place_temp_path(&video);
        let gpu = RecordGpu::default();
        let mut ops = FakeOps::new();
        ops.verify_err = Some("bad track count".into());
        let err = run_with_ops(ctx, &gpu, &ops).unwrap_err();
        assert!(err.to_string().contains("bad track count"));
        assert_eq!(fs::read_to_string(&video).unwrap(), "fake video");
        assert!(!expected_temp.exists());
    }

    #[test]
    fn in_place_dry_run_leaves_temp() {
        let dir = tempdir().unwrap();
        let mut ctx = make_in_place_ctx(dir.path());
        ctx.config.dry_run = true;
        let video = ctx.video_path.clone();
        let expected_temp = in_place_temp_path(&video);
        let gpu = RecordGpu::default();
        let ops = FakeOps::new();
        run_with_ops(ctx, &gpu, &ops).unwrap();
        assert_eq!(fs::read_to_string(&video).unwrap(), "fake video");
        assert!(expected_temp.exists());
    }

    // ── replace_original ──────────────────────────────────────────────────

    #[test]
    fn replace_original_creates_backup_and_replaces() {
        let dir = tempdir().unwrap();
        let video = dir.path().join("ep01.mkv");
        fs::write(&video, b"original content").unwrap();
        let temp = dir.path().join("ep01_temp.mkv");
        fs::write(&temp, b"muxed content").unwrap();
        let ops = FakeOps::new();
        replace_original(&video, &temp, &ops).unwrap();
        assert_eq!(fs::read_to_string(&video).unwrap(), "muxed content");
        let mut backup = video.clone().into_os_string();
        backup.push(".backup");
        assert!(!PathBuf::from(backup).exists());
        assert!(!temp.exists());
    }

    fn backup_of(video: &Path) -> PathBuf {
        let mut s = video.as_os_str().to_os_string();
        s.push(".backup");
        PathBuf::from(s)
    }

    /// Temp verification fails *before* any replacement: the original is left
    /// untouched and no backup is created (nothing to orphan).
    #[test]
    fn replace_original_temp_verify_failure_leaves_original_intact() {
        let dir = tempdir().unwrap();
        let video = dir.path().join("ep01.mkv");
        fs::write(&video, b"original content").unwrap();
        let temp = dir.path().join("ep01_temp.mkv");
        fs::write(&temp, b"muxed content").unwrap();
        let mut ops = FakeOps::new();
        ops.verify_err = Some("temp verification failed".into());
        let err = replace_original(&video, &temp, &ops).unwrap_err();
        assert!(err.to_string().contains("temp verification failed"));
        // Original unchanged; no backup orphaned; temp still present (the
        // run_with_ops caller unlinks it, not replace_original).
        assert_eq!(fs::read_to_string(&video).unwrap(), "original content");
        assert!(!backup_of(&video).exists(), "no backup should be orphaned");
    }

    /// Verification fails *after* the rename (temp verified OK, in-place verify
    /// failed): the original must be restored from backup, the backup removed,
    /// and the error returned. Guards against silently replacing the original
    /// with unverified output.
    #[test]
    fn replace_original_restores_on_post_rename_verify_failure() {
        let dir = tempdir().unwrap();
        let video = dir.path().join("ep01.mkv");
        fs::write(&video, b"original content").unwrap();
        let temp = dir.path().join("ep01_temp.mkv");
        fs::write(&temp, b"muxed content").unwrap();
        let mut ops = FakeOps::new();
        // First verify (the temp) passes; second verify (in place) fails.
        ops.verify_fail_after = Some((1, "in-place verification failed".into()));
        let err = replace_original(&video, &temp, &ops).unwrap_err();
        assert!(err.to_string().contains("in-place verification failed"));
        // Original restored from backup; backup removed (not orphaned).
        assert_eq!(
            fs::read_to_string(&video).unwrap(),
            "original content",
            "original must be restored, never left as unverified output"
        );
        assert!(
            !backup_of(&video).exists(),
            "backup must be cleaned up after restore"
        );
    }

    /// Integration test: real ffmpeg mux via `FfmpegMuxOps` + `DirectGpuExecutor`.
    #[test]
    #[ignore = "requires ffmpeg and real media"]
    fn run_via_ffmpeg() {
        let dir = tempdir().unwrap();
        let ctx = make_ctx(dir.path(), true);
        let executor = DirectGpuExecutor::new();
        let _ = run(ctx, &executor);
    }
}
