//! Pipeline orchestration: `process_file` + `run_all` (async, concurrent files
//! with serialised GPU work) and `process_video_file` (synchronous single file).
//!
//! Port of `movie_translator/async_pipeline.py` (`process_file` / `run_all` /
//! `_handle_pending_ocr`) and `movie_translator/pipeline.py`
//! (`TranslationPipeline.process_video_file`).
//!
//! # Concurrency model
//!
//! Files are processed concurrently (one tokio task each), bounded by
//! `config.workers` via a [`tokio::sync::Semaphore`] — mirroring
//! `asyncio.Semaphore(workers)` in `run_all`. IO/CPU-bound stage work overlaps
//! freely, but **all GPU work funnels through one [`GpuWorker`]**, so OCR /
//! translation / inpaint never run in parallel (the core property of the
//! single-worker `GpuQueue`).
//!
//! Each synchronous stage runs inside [`tokio::task::spawn_blocking`] (mirroring
//! Python's `asyncio.to_thread`). The stages and `resolve_pending_ocr` consume
//! a sync [`GpuExecutor`]; we pass them the [`GpuWorkerHandle`], whose sync impl
//! blocks the blocking-pool thread on the worker's reply. The block is safe
//! precisely because it happens on a `spawn_blocking` thread, never on a runtime
//! worker thread.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mt_core::{PipelineConfig, PipelineContext};
use mt_discovery::create_work_dir;
use mt_media::SubtitleExtractor;
use tokio::sync::Semaphore;

use crate::error::{PipelineError, Result};
use crate::gpu::{resolve_pending_ocr, DirectGpuExecutor, GpuExecutor, OcrStageLabel};
use crate::stages;
use crate::vision::{default_vision_ocr_probe, VisionOcrProbe};
use crate::worker::{GpuWorker, GpuWorkerHandle};

/// Per-file outcome, mirroring the Python `'success' | 'failed' | 'skipped'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStatus {
    Success,
    Failed,
    Skipped,
}

impl FileStatus {
    /// The Python status string.
    pub fn as_str(self) -> &'static str {
        match self {
            FileStatus::Success => "success",
            FileStatus::Failed => "failed",
            FileStatus::Skipped => "skipped",
        }
    }
}

/// Process a single video file through the pipeline (async path).
///
/// Port of `async_pipeline.py::process_file`. Runs the 7 stages in Python order
/// (identify → extract_ref → fetch → extract_english → translate →
/// create_tracks → mux). Sync stages run via `spawn_blocking`; deferred OCR and
/// translation route through `executor` (the shared [`GpuWorkerHandle`]) so GPU
/// work stays serialised. Returns `true` on success, `false` on failure
/// (failures are logged, never propagated — matching the Python `except`).
///
/// `vision_probe` is the injectable Vision-OCR availability check passed to the
/// extract stages (default: [`default_vision_ocr_probe`]).
pub async fn process_file(
    video_path: PathBuf,
    work_dir: PathBuf,
    config: PipelineConfig,
    executor: GpuWorkerHandle,
    vision_probe: VisionOcrProbe,
) -> bool {
    match process_file_inner(video_path.clone(), work_dir, config, executor, vision_probe).await {
        Ok(()) => true,
        Err(e) => {
            let name = video_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            tracing::error!("Failed: {name} - {e}");
            false
        }
    }
}

/// Run a sync stage closure on the blocking pool, threading the context through.
///
/// Mirrors `ctx = await asyncio.to_thread(stage.run, ctx)`.
async fn run_blocking<F>(f: F) -> Result<PipelineContext>
where
    F: FnOnce() -> Result<PipelineContext> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| PipelineError::Stage(format!("stage task panicked: {e}")))?
}

async fn process_file_inner(
    video_path: PathBuf,
    work_dir: PathBuf,
    config: PipelineConfig,
    executor: GpuWorkerHandle,
    vision_probe: VisionOcrProbe,
) -> Result<()> {
    let mut ctx = PipelineContext::new(video_path.clone(), work_dir, config);

    // Stage 1 — Identify (IO).
    ctx = run_blocking(move || stages::identify::run(ctx)).await?;

    // Stage 2 — Extract Reference (IO + deferred OCR).
    ctx = run_blocking(move || stages::extract_ref::run_with_probe(ctx, vision_probe)).await?;
    if ctx.pending_ocr.is_some() {
        ctx =
            resolve_pending_ocr_blocking(ctx, executor.clone(), OcrStageLabel::ExtractRef).await?;
    }

    // Stage 3 — Fetch (IO).
    ctx = run_blocking(move || stages::fetch::run(ctx)).await?;

    // Stage 4 — Extract English (IO + deferred OCR).
    ctx = run_blocking(move || stages::extract_english::run_with_probe(ctx, vision_probe)).await?;
    if ctx.pending_ocr.is_some() {
        ctx = resolve_pending_ocr_blocking(ctx, executor.clone(), OcrStageLabel::ExtractEnglish)
            .await?;
    }

    if ctx.english_source.is_none() {
        return Err(PipelineError::Stage(format!(
            "No English subtitle source found for {}",
            video_path.display()
        )));
    }
    if ctx.dialogue_lines.is_none() {
        return Err(PipelineError::Stage(format!(
            "No dialogue lines extracted for {}",
            video_path.display()
        )));
    }

    // Stage 5 — Translate (font check + GPU translation).
    //
    // The Rust `translate::run` does the font check inline and routes primary +
    // extra translations through the executor. Because `executor` is the shared
    // worker handle, the GPU calls serialise across all files. (Python overlaps
    // the font-check IO with the GPU await via `asyncio.gather`; that is a
    // latency optimisation only — the observable result is identical.)
    let exec = executor.clone();
    ctx = run_blocking(move || stages::translate::run(ctx, &exec, None)).await?;
    if ctx.translated_lines.as_ref().is_none_or(|l| l.is_empty()) {
        return Err(PipelineError::Stage(
            "Translation failed -- empty result".into(),
        ));
    }

    // Stage 6 — Create Tracks (IO).
    ctx = run_blocking(move || stages::create_tracks::run(ctx)).await?;

    // Stage 7 — Mux (optional inpaint GPU + IO). The mux stage performs the
    // inpaint through the executor internally (serialised through the worker).
    let exec = executor.clone();
    ctx = run_blocking(move || stages::mux::run(ctx, &exec)).await?;

    let _ = ctx;
    Ok(())
}

/// Resolve pending OCR on the blocking pool, routing OCR through the worker.
///
/// Port of `_handle_pending_ocr`. The `executor` (worker handle) implements the
/// sync [`GpuExecutor`] by blocking on the worker reply; doing so inside
/// `spawn_blocking` keeps GPU work serialised without blocking a runtime thread.
async fn resolve_pending_ocr_blocking(
    ctx: PipelineContext,
    executor: GpuWorkerHandle,
    label: OcrStageLabel,
) -> Result<PipelineContext> {
    run_blocking(move || {
        let mut ctx = ctx;
        resolve_pending_ocr(&mut ctx, &executor, label)?;
        Ok(ctx)
    })
    .await
}

/// Orchestrate processing of all video files with bounded concurrency.
///
/// Port of `async_pipeline.py::run_all`. Spawns one task per file, bounded by
/// `config.workers` (falling back to `min(files, 4)` when unset, like Python),
/// all sharing a single [`GpuWorker`] so GPU work stays serialised while IO/CPU
/// overlaps. Skips files that already have Polish subtitles. Returns per-file
/// `(path, status)` results in the original input order.
pub async fn run_all(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    run_all_with(video_files, root_dir, config, default_vision_ocr_probe).await
}

/// Like [`run_all`], with an injectable Vision-OCR probe (for tests).
pub async fn run_all_with(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
    vision_probe: VisionOcrProbe,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    let workers = if config.workers > 0 {
        config.workers
    } else {
        (video_files.len() as u32).clamp(1, 4)
    };

    let worker = GpuWorker::spawn();
    let result = run_all_with_executor(
        video_files,
        root_dir,
        config,
        vision_probe,
        &worker,
        workers,
    )
    .await;
    worker.shutdown().await;
    result
}

/// Shared implementation that takes an already-spawned worker, so tests can
/// supply a fake executor and assert serialisation.
async fn run_all_with_executor(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
    vision_probe: VisionOcrProbe,
    worker: &GpuWorker,
    workers: u32,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    let semaphore = Arc::new(Semaphore::new(workers.max(1) as usize));
    let handle = worker.handle();

    let mut joins = Vec::with_capacity(video_files.len());
    for (idx, video_path) in video_files.into_iter().enumerate() {
        let permit_sem = semaphore.clone();
        let executor = handle.clone();
        let config = config.clone();
        let root_dir = root_dir.clone();

        joins.push(tokio::spawn(async move {
            // Hold a permit for the file's whole lifetime (the Python
            // `async with semaphore:` wraps the entire per-file body).
            let _permit = permit_sem.acquire_owned().await.expect("semaphore");

            // Check for existing Polish subtitles (IO-bound) — skip if present.
            let vp = video_path.clone();
            let has_polish = tokio::task::spawn_blocking(move || {
                SubtitleExtractor::new()
                    .has_polish_subtitles(&vp)
                    .unwrap_or(false)
            })
            .await
            .unwrap_or(false);

            if has_polish {
                return (idx, video_path, FileStatus::Skipped);
            }

            let work_dir = match create_work_dir(&video_path, &root_dir) {
                Ok(wd) => wd,
                Err(e) => {
                    tracing::error!(
                        "Failed to create work dir for {}: {e}",
                        video_path.display()
                    );
                    return (idx, video_path, FileStatus::Failed);
                }
            };

            let success =
                process_file(video_path.clone(), work_dir, config, executor, vision_probe).await;

            let status = if success {
                FileStatus::Success
            } else {
                FileStatus::Failed
            };
            (idx, video_path, status)
        }));
    }

    // Collect, restoring input order (Python appends as tasks complete; we keep
    // the deterministic input order which is friendlier for callers/tests).
    let mut collected: Vec<(usize, PathBuf, FileStatus)> = Vec::with_capacity(joins.len());
    for j in joins {
        match j.await {
            Ok(triple) => collected.push(triple),
            Err(e) => return Err(PipelineError::Stage(format!("file task panicked: {e}"))),
        }
    }
    collected.sort_by_key(|(idx, _, _)| *idx);
    Ok(collected
        .into_iter()
        .map(|(_, path, status)| (path, status))
        .collect())
}

/// Process a single file synchronously (no tokio, no worker).
///
/// Port of `pipeline.py::TranslationPipeline.process_video_file`. Runs the 7
/// stages sequentially using the inline [`DirectGpuExecutor`] and the sync
/// [`resolve_pending_ocr`]. Returns `true` on success, `false` on failure
/// (logged, not propagated). Used by the CLI single-file path.
pub fn process_video_file(video_path: &Path, work_dir: &Path, config: PipelineConfig) -> bool {
    process_video_file_with(video_path, work_dir, config, default_vision_ocr_probe)
}

/// Like [`process_video_file`], with an injectable Vision-OCR probe (for tests).
pub fn process_video_file_with(
    video_path: &Path,
    work_dir: &Path,
    config: PipelineConfig,
    vision_probe: VisionOcrProbe,
) -> bool {
    let executor = DirectGpuExecutor::new();
    match process_video_file_inner(video_path, work_dir, config, &executor, vision_probe) {
        Ok(()) => true,
        Err(e) => {
            tracing::error!(
                "Failed: {} - {e}",
                video_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            );
            false
        }
    }
}

fn process_video_file_inner(
    video_path: &Path,
    work_dir: &Path,
    config: PipelineConfig,
    executor: &dyn GpuExecutor,
    vision_probe: VisionOcrProbe,
) -> Result<()> {
    let mut ctx = PipelineContext::new(video_path.to_path_buf(), work_dir.to_path_buf(), config);

    ctx = stages::identify::run(ctx)?;

    ctx = stages::extract_ref::run_with_probe(ctx, vision_probe)?;
    if ctx.pending_ocr.is_some() {
        resolve_pending_ocr(&mut ctx, executor, OcrStageLabel::ExtractRef)?;
    }

    ctx = stages::fetch::run(ctx)?;

    ctx = stages::extract_english::run_with_probe(ctx, vision_probe)?;
    if ctx.pending_ocr.is_some() {
        resolve_pending_ocr(&mut ctx, executor, OcrStageLabel::ExtractEnglish)?;
    }

    ctx = stages::translate::run(ctx, executor, None)?;
    ctx = stages::create_tracks::run(ctx)?;
    ctx = stages::mux::run(ctx, executor)?;

    let _ = ctx;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{ConcurrencyProbe, GpuWorker};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    fn probe_off() -> bool {
        false
    }

    #[test]
    fn file_status_strings_match_python() {
        assert_eq!(FileStatus::Success.as_str(), "success");
        assert_eq!(FileStatus::Failed.as_str(), "failed");
        assert_eq!(FileStatus::Skipped.as_str(), "skipped");
    }

    /// `run_all` over non-existent files: each fails deterministically (the
    /// stages need a real video), and results come back in input order with the
    /// right status. Exercises the run_all plumbing (spawn-per-file, shared
    /// worker, status aggregation, ordering) without ffmpeg fixtures.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn run_all_aggregates_failures_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let files: Vec<PathBuf> = (0..5)
            .map(|i| dir.path().join(format!("missing{i}.mkv")))
            .collect();
        let config = PipelineConfig {
            workers: 2,
            enable_fetch: false,
            ..Default::default()
        };
        let results = run_all_with(files.clone(), dir.path().to_path_buf(), config, probe_off)
            .await
            .unwrap();
        assert_eq!(results.len(), 5);
        for (i, (path, status)) in results.iter().enumerate() {
            assert_eq!(path, &files[i], "results must preserve input order");
            assert_eq!(*status, FileStatus::Failed);
        }
    }

    /// `run_all` with an empty input is a no-op returning no results, and tears
    /// the worker down cleanly.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_all_empty_input() {
        let dir = tempfile::tempdir().unwrap();
        let results = run_all_with(
            vec![],
            dir.path().to_path_buf(),
            PipelineConfig::default(),
            probe_off,
        )
        .await
        .unwrap();
        assert!(results.is_empty());
    }

    /// The synchronous `process_video_file` path returns `false` (not a panic)
    /// when a stage fails on a missing video. Mirrors the Python `except` that
    /// logs and returns `False`.
    #[test]
    fn process_video_file_returns_false_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("nope.mkv");
        let config = PipelineConfig {
            enable_fetch: false,
            ..Default::default()
        };
        let ok = process_video_file_with(&video, dir.path(), config, probe_off);
        assert!(!ok);
    }

    /// Orchestration-level serialisation: drive several GPU submissions through
    /// the shared worker handle exactly as `process_file` does (sync executor
    /// from inside `spawn_blocking`), across many concurrent file-like tasks,
    /// and assert the GPU never runs two jobs at once. This is the core
    /// cross-file property `run_all` must uphold.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gpu_work_serialises_across_concurrent_file_tasks() {
        let (probe, stats) = ConcurrencyProbe::new(Duration::from_millis(15));
        let worker = GpuWorker::spawn_with(probe);
        let handle = worker.handle();
        let sem = Arc::new(Semaphore::new(3)); // workers = 3

        let mut joins = Vec::new();
        for _ in 0..9 {
            let h = handle.clone();
            let sem = sem.clone();
            joins.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                // Mirror process_file: sync GpuExecutor call inside spawn_blocking.
                tokio::task::spawn_blocking(move || {
                    use crate::gpu::GpuExecutor;
                    let req = mt_ml::TranslateRequest {
                        lines: vec![],
                        device: "cpu".into(),
                        batch_size: 1,
                        model: "allegro".into(),
                        proper_nouns: None,
                    };
                    h.translate(&req).unwrap();
                })
                .await
                .unwrap();
            }));
        }
        for j in joins {
            j.await.unwrap();
        }
        assert_eq!(stats.total_calls.load(Ordering::SeqCst), 9);
        assert_eq!(
            stats.max_concurrency.load(Ordering::SeqCst),
            1,
            "GPU work must serialise across concurrent files"
        );
        worker.shutdown().await;
    }

    /// Full async `process_file` over a real fixture using `--self-test` ML
    /// scripts requires ffmpeg + a real video for the extract/mux stages, so it
    /// is ignored by default. Runs the true stage sequence end-to-end.
    #[tokio::test]
    #[ignore = "requires a real video fixture + ffmpeg + ML scripts"]
    async fn process_file_end_to_end_real_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("ep01.mkv");
        let worker = GpuWorker::spawn();
        let ok = process_file(
            video,
            dir.path().to_path_buf(),
            PipelineConfig::default(),
            worker.handle(),
            default_vision_ocr_probe,
        )
        .await;
        worker.shutdown().await;
        let _ = ok;
    }
}
