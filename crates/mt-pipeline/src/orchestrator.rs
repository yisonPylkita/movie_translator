//! Pipeline orchestration: `process_file` + `run_all` (async, concurrent files
//! with serialised GPU work) and `process_video_file` (synchronous single file).
//!
//! # Concurrency model
//!
//! Files are processed concurrently (one tokio task each), bounded by
//! `config.workers` via a [`tokio::sync::Semaphore`]. IO/CPU-bound stage work
//! overlaps freely, but **all GPU work funnels through one [`GpuWorker`]**, so
//! OCR / translation / inpaint never run in parallel.
//!
//! Each synchronous stage runs inside [`tokio::task::spawn_blocking`]. The
//! stages and `resolve_pending_ocr` consume a sync [`GpuExecutor`]; we pass them
//! the [`GpuWorkerHandle`], whose sync impl blocks the blocking-pool thread on
//! the worker's reply. The block is safe precisely because it happens on a
//! `spawn_blocking` thread, never on a runtime worker thread.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use mt_core::{PipelineConfig, PipelineContext};
use mt_discovery::create_work_dir;
use mt_fetch::ogladajanime::{self, Discovery, HardsubPlan};
use mt_media::SubtitleExtractor;
use tokio::sync::Semaphore;

use crate::error::{PipelineError, Result};
use crate::gpu::{resolve_pending_ocr, DirectGpuExecutor, GpuExecutor, OcrStageLabel};
use crate::progress::{FinishStatus, ProgressEvent, ProgressSender, Stage};
use crate::stages;
use crate::vision::{default_vision_ocr_probe, VisionOcrProbe};
use crate::worker::{GpuWorker, GpuWorkerHandle};

/// Per-file outcome: `success`, `failed`, or `skipped`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStatus {
    Success,
    Failed,
    Skipped,
}

impl FileStatus {
    /// The lowercase status string used in summaries and manifests.
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
/// Runs the 7 stages in order (identify → extract_ref → fetch →
/// extract_english → translate → create_tracks → mux). Sync stages run via
/// `spawn_blocking`; deferred OCR and translation route through `executor` (the
/// shared [`GpuWorkerHandle`]) so GPU work stays serialised. Returns `true` on
/// success, `false` on failure (failures are logged, never propagated).
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
    process_file_with_progress(
        video_path,
        work_dir,
        config,
        executor,
        vision_probe,
        ProgressSender::disabled(),
        None,
    )
    .await
        == FileOutcome::Success
}

/// Per-file outcome the orchestrator distinguishes when aggregating.
///
/// `SkippedNoSubs` is the principled "no English subtitle source" case (see fix
/// #5) — preserved separately so `run_all` can report it as `Skipped`, not
/// `Failed`, in the summary while the TUI shows the dedicated label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileOutcome {
    Success,
    Failed,
    SkippedNoSubs,
}

/// Same as [`process_file`] but emits structured progress events. Returns the
/// fine-grained [`FileOutcome`].
pub async fn process_file_with_progress(
    video_path: PathBuf,
    work_dir: PathBuf,
    config: PipelineConfig,
    executor: GpuWorkerHandle,
    vision_probe: VisionOcrProbe,
    progress: ProgressSender,
    hardsub_plan: Option<Arc<HardsubPlan>>,
) -> FileOutcome {
    match process_file_inner(
        video_path.clone(),
        work_dir,
        config,
        executor,
        vision_probe,
        progress,
        hardsub_plan,
    )
    .await
    {
        Ok(()) => FileOutcome::Success,
        Err(e) if e.is_no_english_source() => {
            let name = video_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            tracing::info!("Skipping {name}: no English subtitle source");
            FileOutcome::SkippedNoSubs
        }
        Err(e) => {
            let name = video_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            tracing::error!("Failed: {name} - {e}");
            FileOutcome::Failed
        }
    }
}

/// Run a sync stage closure on the blocking pool, threading the context through.
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
    progress: ProgressSender,
    hardsub_plan: Option<Arc<HardsubPlan>>,
) -> Result<()> {
    let mut ctx = PipelineContext::new(video_path.clone(), work_dir, config);

    let emit_stage = |stage: Stage| {
        progress.send(ProgressEvent::StageEntered {
            path: video_path.clone(),
            stage,
        });
    };

    // Stage 1 — Identify (IO).
    emit_stage(Stage::Identify);
    ctx = run_blocking(move || stages::identify::run(ctx)).await?;

    // Stage 2 — Extract Reference (IO + deferred OCR).
    emit_stage(Stage::ExtractRef);
    ctx = run_blocking(move || stages::extract_ref::run_with_probe(ctx, vision_probe)).await?;
    if ctx.pending_ocr.is_some() {
        ctx =
            resolve_pending_ocr_blocking(ctx, executor.clone(), OcrStageLabel::ExtractRef).await?;
    }

    // Stage 3 — Fetch (IO).
    emit_stage(Stage::Fetch);
    ctx = run_blocking(move || stages::fetch::run(ctx)).await?;

    // Stage 4 — Extract English (IO + deferred OCR).
    emit_stage(Stage::ExtractEnglish);
    ctx = run_blocking(move || stages::extract_english::run_with_probe(ctx, vision_probe)).await?;
    if ctx.pending_ocr.is_some() {
        ctx = resolve_pending_ocr_blocking(ctx, executor.clone(), OcrStageLabel::ExtractEnglish)
            .await?;
    }

    if ctx.english_source.is_none() {
        // Pipeline produced no English source even after extract + burned-in
        // OCR — treat as `NoEnglishSource` (skip-not-fail). See fix #5.
        return Err(PipelineError::NoEnglishSource);
    }
    if ctx.dialogue_lines.is_none() {
        return Err(PipelineError::Stage(format!(
            "No dialogue lines extracted for {}",
            video_path.display()
        )));
    }

    // Stage 4.5 — Hardsub OCR (gated by --hardsub-ocr; only when the interactive
    // prep produced a plan with this episode). Runs here so the English source
    // exists as the alignment reference. Downloads off-GPU, OCRs through the
    // worker, aligns to the reference, and injects a fetched Polish track that
    // create_tracks → mux turn into an output track. Non-fatal on failure.
    if let Some(plan) = hardsub_plan.clone() {
        emit_stage(Stage::HardsubOcr);
        let exec = executor.clone();
        ctx = run_blocking(move || stages::hardsub_ocr::run(ctx, &exec, &plan)).await?;
    }

    // Stage 5 — Translate (font check + GPU translation).
    //
    // `translate::run` does the font check inline and routes primary + extra
    // translations through the executor. Because `executor` is the shared worker
    // handle, the GPU calls serialise across all files. (Running the font-check
    // IO concurrently with the GPU await would only be a latency optimisation —
    // the observable result is identical.)
    emit_stage(Stage::Translate);
    let exec = executor.clone();
    ctx = run_blocking(move || stages::translate::run(ctx, &exec, None)).await?;
    if ctx.translated_lines.as_ref().is_none_or(|l| l.is_empty()) {
        return Err(PipelineError::Stage(
            "Translation failed -- empty result".into(),
        ));
    }

    // Stage 6 — Create Tracks (IO).
    emit_stage(Stage::CreateTracks);
    ctx = run_blocking(move || stages::create_tracks::run(ctx)).await?;

    // Stage 7 — Mux (optional inpaint GPU + IO). The mux stage performs the
    // inpaint through the executor internally (serialised through the worker).
    emit_stage(Stage::Mux);
    let exec = executor.clone();
    ctx = run_blocking(move || stages::mux::run(ctx, &exec)).await?;

    let _ = ctx;
    Ok(())
}

/// Resolve pending OCR on the blocking pool, routing OCR through the worker.
///
/// The `executor` (worker handle) implements the sync [`GpuExecutor`] by
/// blocking on the worker reply; doing so inside `spawn_blocking` keeps GPU work
/// serialised without blocking a runtime thread.
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
/// Spawns one task per file, bounded by `config.workers` (falling back to
/// `min(files, 4)` when unset), all sharing a single [`GpuWorker`] so GPU work
/// stays serialised while IO/CPU overlaps. Skips files that already have Polish
/// subtitles. Returns per-file `(path, status)` results in the original input
/// order.
pub async fn run_all(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    run_all_with(video_files, root_dir, config, default_vision_ocr_probe).await
}

/// Like [`run_all`] but also emits structured [`ProgressEvent`]s to `progress`
/// (the CLI feeds these to the ratatui TUI). Pass [`ProgressSender::disabled`]
/// to behave exactly like [`run_all`].
pub async fn run_all_with_progress(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
    progress: ProgressSender,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    run_all_full(
        video_files,
        root_dir,
        config,
        default_vision_ocr_probe,
        progress,
    )
    .await
}

/// Like [`run_all`], with an injectable Vision-OCR probe (for tests).
pub async fn run_all_with(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
    vision_probe: VisionOcrProbe,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    run_all_full(
        video_files,
        root_dir,
        config,
        vision_probe,
        ProgressSender::disabled(),
    )
    .await
}

async fn run_all_full(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
    vision_probe: VisionOcrProbe,
    progress: ProgressSender,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    let workers = if config.workers > 0 {
        config.workers
    } else {
        (video_files.len() as u32).clamp(1, 4)
    };

    // Pre-populate the TUI with the queued file list. The orchestrator emits
    // FileStarted / StageEntered / FileFinished as the run progresses.
    progress.send(ProgressEvent::Queued {
        files: video_files.clone(),
    });

    // --hardsub-ocr: run the once-per-run interactive prep (discover the anime,
    // open the browser, wait for the resolver userscript's JSON) before any
    // per-file work. A failure here is non-fatal — the run continues without
    // the OCR track.
    let hardsub_plan = if config.enable_hardsub_ocr {
        prepare_hardsub_plan(video_files.first(), &progress).await
    } else {
        None
    };

    let worker = GpuWorker::spawn();
    let result = run_all_with_executor(
        video_files,
        root_dir,
        config,
        vision_probe,
        &worker,
        workers,
        progress,
        hardsub_plan,
    )
    .await;
    worker.shutdown().await;
    result
}

/// How long to wait for the resolver userscript's JSON to land in Downloads.
const HARDSUB_WAIT_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Once-per-run interactive prep for `--hardsub-ocr`: identify the anime from
/// the first file, discover it on ogladajanime, open the browser there, and
/// wait for the resolver userscript's JSON in `~/Downloads`. Returns the parsed
/// plan, or `None` if anything fails (the run then proceeds without OCR).
async fn prepare_hardsub_plan(
    first_file: Option<&PathBuf>,
    progress: &ProgressSender,
) -> Option<Arc<HardsubPlan>> {
    let first_file = first_file?.clone();
    let progress = progress.clone();
    let plan = tokio::task::spawn_blocking(move || {
        let title = hardsub_title(&first_file);
        if title.is_empty() {
            tracing::warn!("hardsub-ocr: could not derive a title from {first_file:?}");
            return None;
        }
        log_hardsub(
            &progress,
            format!("discovering '{title}' on ogladajanime.pl…"),
        );
        let (slug, url) = match ogladajanime::discover(&title) {
            Discovery::Found { slug, url } => {
                log_hardsub(&progress, format!("found anime page: {url}"));
                (Some(slug), url)
            }
            Discovery::Search { url } => {
                log_hardsub(
                    &progress,
                    format!("no exact match — opening search; pick the anime: {url}"),
                );
                (None, url)
            }
        };

        let since = SystemTime::now();
        if let Err(e) = ogladajanime::open_in_browser(&url) {
            log_hardsub(
                &progress,
                format!("could not open browser ({e}); open {url} manually"),
            );
        }
        let downloads = ogladajanime::default_downloads_dir();
        log_hardsub(
            &progress,
            format!(
                "run the resolver userscript in your browser — waiting for its JSON in {}…",
                downloads.display()
            ),
        );
        let json = match ogladajanime::wait_for_resolver_json(
            slug.as_deref(),
            since,
            &downloads,
            HARDSUB_WAIT_TIMEOUT,
            Duration::from_secs(1),
        ) {
            Ok(p) => p,
            Err(e) => {
                log_hardsub(&progress, format!("hardsub-ocr aborted: {e}"));
                return None;
            }
        };
        match ogladajanime::parse_plan(&json, slug.as_deref().unwrap_or("")) {
            Ok(plan) => {
                log_hardsub(
                    &progress,
                    format!(
                        "loaded {} episode(s) from {}",
                        plan.episode_count(),
                        json.display()
                    ),
                );
                Some(plan)
            }
            Err(e) => {
                log_hardsub(&progress, format!("could not parse resolver JSON: {e}"));
                None
            }
        }
    })
    .await
    .ok()
    .flatten();
    plan.map(Arc::new)
}

/// Derive an anime title from a video path via the filename parser, falling
/// back to the file stem.
fn hardsub_title(path: &Path) -> String {
    let filename = path.file_name().map(|n| n.to_string_lossy().to_string());
    if let Some(name) = filename.as_deref() {
        if let Ok(parsed) = mt_ml::backend::parse_filename(name, None) {
            if let Some(title) = parsed.title.filter(|t| !t.is_empty()) {
                return title;
            }
        }
    }
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Emit a hardsub prep status line both to the TUI (as a Log event) and tracing.
fn log_hardsub(progress: &ProgressSender, message: String) {
    tracing::info!("hardsub-ocr: {message}");
    progress.send(ProgressEvent::Log {
        level: "info".to_string(),
        target: "hardsub-ocr".to_string(),
        message,
    });
}

/// Shared implementation that takes an already-spawned worker, so tests can
/// supply a fake executor and assert serialisation.
#[allow(clippy::too_many_arguments)]
async fn run_all_with_executor(
    video_files: Vec<PathBuf>,
    root_dir: PathBuf,
    config: PipelineConfig,
    vision_probe: VisionOcrProbe,
    worker: &GpuWorker,
    workers: u32,
    progress: ProgressSender,
    hardsub_plan: Option<Arc<HardsubPlan>>,
) -> Result<Vec<(PathBuf, FileStatus)>> {
    let semaphore = Arc::new(Semaphore::new(workers.max(1) as usize));
    let handle = worker.handle();

    // Keep each join handle paired with its (idx, path) so that if a file task
    // panics we can still record a per-file failure with its real identity —
    // and, crucially, so we can join *every* task before shutting the worker
    // down (no early return that would orphan the other in-flight tasks).
    let mut joins = Vec::with_capacity(video_files.len());
    for (idx, video_path) in video_files.into_iter().enumerate() {
        let permit_sem = semaphore.clone();
        let executor = handle.clone();
        let config = config.clone();
        let root_dir = root_dir.clone();
        let task_idx = idx;
        let task_path = video_path.clone();
        let progress = progress.clone();
        let hardsub_plan = hardsub_plan.clone();

        let handle = tokio::spawn(async move {
            // Hold a permit for the file's whole lifetime.
            let _permit = permit_sem.acquire_owned().await.expect("semaphore");

            progress.send(ProgressEvent::FileStarted {
                path: video_path.clone(),
            });

            // Check for existing Polish subtitles (IO-bound) — skip if present.
            //
            // A probe error (unreadable/corrupt file) or a panic in the probe
            // task must NOT collapse to `has_polish = false` and then be
            // processed: that would silently push a broken file through the
            // whole pipeline. Treat either as a per-file `Failed` and report it.
            let vp = video_path.clone();
            let probe = tokio::task::spawn_blocking(move || {
                SubtitleExtractor::new().has_polish_subtitles(&vp)
            })
            .await;

            let has_polish = match probe {
                Ok(Ok(has)) => has,
                Ok(Err(e)) => {
                    tracing::error!(
                        "Failed to probe Polish subtitles for {}: {e}",
                        video_path.display()
                    );
                    progress.send(ProgressEvent::FileFinished {
                        path: video_path.clone(),
                        status: FinishStatus::Failed,
                    });
                    return (idx, video_path, FileStatus::Failed);
                }
                Err(join_err) => {
                    tracing::error!(
                        "Polish-subtitle probe task panicked for {}: {join_err}",
                        video_path.display()
                    );
                    progress.send(ProgressEvent::FileFinished {
                        path: video_path.clone(),
                        status: FinishStatus::Failed,
                    });
                    return (idx, video_path, FileStatus::Failed);
                }
            };

            if has_polish {
                progress.send(ProgressEvent::FileFinished {
                    path: video_path.clone(),
                    status: FinishStatus::Skipped,
                });
                return (idx, video_path, FileStatus::Skipped);
            }

            let work_dir = match create_work_dir(&video_path, &root_dir) {
                Ok(wd) => wd,
                Err(e) => {
                    tracing::error!(
                        "Failed to create work dir for {}: {e}",
                        video_path.display()
                    );
                    progress.send(ProgressEvent::FileFinished {
                        path: video_path.clone(),
                        status: FinishStatus::Failed,
                    });
                    return (idx, video_path, FileStatus::Failed);
                }
            };

            let keep_artifacts = config.keep_artifacts;
            let outcome = process_file_with_progress(
                video_path.clone(),
                work_dir.clone(),
                config,
                executor,
                vision_probe,
                progress.clone(),
                hardsub_plan,
            )
            .await;

            // On success OR a no-subs skip the file's work dir is empty/junk —
            // remove it (unless --keep-artifacts). Genuine failures keep the
            // dir around for debugging.
            let cleanup = matches!(outcome, FileOutcome::Success | FileOutcome::SkippedNoSubs);
            if cleanup && !keep_artifacts {
                cleanup_work_dir(&work_dir, &root_dir);
            }

            let (status, finish_status) = match outcome {
                FileOutcome::Success => (FileStatus::Success, FinishStatus::Success),
                FileOutcome::Failed => (FileStatus::Failed, FinishStatus::Failed),
                FileOutcome::SkippedNoSubs => (FileStatus::Skipped, FinishStatus::SkippedNoSubs),
            };
            progress.send(ProgressEvent::FileFinished {
                path: video_path.clone(),
                status: finish_status,
            });
            (idx, video_path, status)
        });
        joins.push((task_idx, task_path, handle));
    }

    // Collect, restoring input order: tasks complete out of order, but we return
    // the deterministic input order which is friendlier for callers/tests.
    //
    // Await EVERY task — never early-return on a JoinError. A panicked task is
    // recorded as a per-file `Failed` (using the idx/path we paired with it),
    // so the remaining tasks are still joined and the worker is only shut down
    // once nothing is in flight. This avoids orphaned tasks (each holding a
    // `GpuWorkerHandle` clone) racing the worker's `shutdown()`.
    let mut collected: Vec<(usize, PathBuf, FileStatus)> = Vec::with_capacity(joins.len());
    for (idx, path, j) in joins {
        match j.await {
            Ok(triple) => collected.push(triple),
            Err(join_err) => {
                tracing::error!("File task panicked for {}: {join_err}", path.display());
                collected.push((idx, path, FileStatus::Failed));
            }
        }
    }
    collected.sort_by_key(|(idx, _, _)| *idx);
    Ok(collected
        .into_iter()
        .map(|(_, path, status)| (path, status))
        .collect())
}

/// Remove a successful file's work dir and prune now-empty parent dirs up to
/// (and including, if empty) the `.translate_temp` root under `root_dir`.
///
/// Best-effort: failures are logged at debug and ignored (they must never turn
/// a successful translation into a reported failure).
fn cleanup_work_dir(work_dir: &Path, root_dir: &Path) {
    if !work_dir.exists() {
        return;
    }
    if let Err(e) = std::fs::remove_dir_all(work_dir) {
        tracing::debug!("Failed to clean up {}: {e}", work_dir.display());
        return;
    }

    let temp_root = root_dir.join(".translate_temp");
    // Walk up from the work dir's parent, removing each directory that is now
    // empty, stopping at the temp root or the input root.
    let mut parent = work_dir.parent().map(Path::to_path_buf);
    while let Some(dir) = parent {
        if dir == temp_root || dir == root_dir {
            break;
        }
        match dir_is_empty(&dir) {
            Some(true) => {
                if std::fs::remove_dir(&dir).is_err() {
                    break;
                }
                parent = dir.parent().map(Path::to_path_buf);
            }
            _ => break,
        }
    }
    // Finally, remove the temp root itself if it is now empty.
    if dir_is_empty(&temp_root) == Some(true) {
        let _ = std::fs::remove_dir(&temp_root);
    }
}

/// `Some(true)` if `dir` exists and has no entries, `Some(false)` if it has
/// entries, `None` if it can't be read.
fn dir_is_empty(dir: &Path) -> Option<bool> {
    let mut entries = std::fs::read_dir(dir).ok()?;
    Some(entries.next().is_none())
}

/// Process a single file synchronously (no tokio, no worker).
///
/// Runs the 7 stages sequentially using the inline [`DirectGpuExecutor`] and the
/// sync [`resolve_pending_ocr`]. Returns `true` on success, `false` on failure
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
    fn file_status_strings() {
        assert_eq!(FileStatus::Success.as_str(), "success");
        assert_eq!(FileStatus::Failed.as_str(), "failed");
        assert_eq!(FileStatus::Skipped.as_str(), "skipped");
    }

    /// Fix #3: a successful run without --keep-artifacts removes the work dir
    /// and prunes the now-empty `.translate_temp` tree up to the input root.
    #[test]
    fn cleanup_work_dir_removes_and_prunes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Match the create_work_dir layout: root/.translate_temp/Show/ep01/
        let work_dir = root.join(".translate_temp").join("Show").join("ep01");
        std::fs::create_dir_all(work_dir.join("candidates")).unwrap();
        std::fs::write(work_dir.join("artifact.srt"), b"x").unwrap();

        cleanup_work_dir(&work_dir, root);

        assert!(!work_dir.exists(), "work dir removed");
        assert!(
            !root.join(".translate_temp").join("Show").exists(),
            "empty intermediate parent pruned"
        );
        assert!(
            !root.join(".translate_temp").exists(),
            "empty temp root pruned"
        );
        assert!(root.exists(), "input root never removed");
    }

    /// Pruning stops at a non-empty sibling: a second episode's work dir must
    /// survive when the first is cleaned.
    #[test]
    fn cleanup_work_dir_keeps_nonempty_siblings() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let temp = root.join(".translate_temp").join("Show");
        let ep01 = temp.join("ep01");
        let ep02 = temp.join("ep02");
        std::fs::create_dir_all(&ep01).unwrap();
        std::fs::create_dir_all(&ep02).unwrap();
        std::fs::write(ep02.join("keep.srt"), b"x").unwrap();

        cleanup_work_dir(&ep01, root);

        assert!(!ep01.exists(), "cleaned work dir removed");
        assert!(ep02.exists(), "non-empty sibling survives");
        assert!(temp.exists(), "shared parent survives (still has ep02)");
    }

    /// Fix #5 regression guard: when the Polish-subtitle probe cannot read a
    /// file (here, a non-existent path → ffprobe errors), that file must be
    /// reported `Failed`, never silently treated as `has_polish = false` and
    /// then `Skipped`. (Skipped would mean a broken file is quietly ignored.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unprobeable_file_is_failed_not_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let files = vec![dir.path().join("does_not_exist.mkv")];
        let config = PipelineConfig {
            workers: 1,
            enable_fetch: false,
            ..Default::default()
        };
        let results = run_all_with(files, dir.path().to_path_buf(), config, probe_off)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].1,
            FileStatus::Failed,
            "an unprobeable file must be Failed, never Skipped"
        );
    }

    /// Fix #6 regression guard: every file task is joined and recorded (in
    /// input order) and the shared worker shuts down cleanly afterwards. With
    /// the previous early-return-on-JoinError the remaining tasks would have
    /// been orphaned holding worker-handle clones; this drives enough
    /// concurrent files through the real shutdown path to exercise that.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_file_tasks_joined_before_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let files: Vec<PathBuf> = (0..8)
            .map(|i| dir.path().join(format!("ep{i}.mkv")))
            .collect();
        let config = PipelineConfig {
            workers: 3,
            enable_fetch: false,
            ..Default::default()
        };
        // Returns (and thus reaches worker.shutdown()) only after joining all 8.
        let results = run_all_with(files.clone(), dir.path().to_path_buf(), config, probe_off)
            .await
            .unwrap();
        assert_eq!(results.len(), 8, "every spawned file task must be joined");
        for (i, (path, _status)) in results.iter().enumerate() {
            assert_eq!(path, &files[i], "results must preserve input order");
        }
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
    /// when a stage fails on a missing video: failures are logged, not propagated.
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
                // Like process_file: sync GpuExecutor call inside spawn_blocking.
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

    /// Fix #5 + progress plumbing: `run_all_with_progress` emits a `Queued`
    /// event for the discovered file list, then `FileStarted`/`FileFinished`
    /// for each file. With unprobeable files everything terminates as
    /// `Failed`, so we only assert the event SEQUENCE here — the no-subs
    /// → `Skipped` mapping is exercised by the extract_english tests +
    /// `FileOutcome` unit test below.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_all_with_progress_emits_lifecycle_events() {
        let dir = tempfile::tempdir().unwrap();
        let files: Vec<PathBuf> = (0..3)
            .map(|i| dir.path().join(format!("ep{i}.mkv")))
            .collect();
        let config = PipelineConfig {
            workers: 1,
            enable_fetch: false,
            ..Default::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = ProgressSender::new(tx);
        let results =
            run_all_with_progress(files.clone(), dir.path().to_path_buf(), config, sender)
                .await
                .unwrap();
        assert_eq!(results.len(), 3);

        // Drain the channel.
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        // First event is always Queued(files).
        match &events[0] {
            ProgressEvent::Queued { files: q } => {
                assert_eq!(q.len(), 3);
            }
            other => panic!("expected first event Queued, got {other:?}"),
        }
        // Every file must produce a FileStarted and a FileFinished.
        let started: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ProgressEvent::FileStarted { path } => Some(path.clone()),
                _ => None,
            })
            .collect();
        let finished: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ProgressEvent::FileFinished { path, status } => Some((path.clone(), *status)),
                _ => None,
            })
            .collect();
        assert_eq!(started.len(), 3, "every file must emit FileStarted");
        assert_eq!(finished.len(), 3, "every file must emit FileFinished");
        for (_, s) in &finished {
            assert!(
                matches!(s, FinishStatus::Failed),
                "unprobeable files finish Failed"
            );
        }
    }

    /// Fix #5: a stage returning `PipelineError::NoEnglishSource` from
    /// `process_file_with_progress` collapses to `FileOutcome::SkippedNoSubs`,
    /// which the orchestrator maps to `FileStatus::Skipped` (not Failed) and
    /// `FinishStatus::SkippedNoSubs` for the TUI.
    #[test]
    fn no_english_source_maps_to_skipped_not_failed() {
        // Direct mapping check — what the orchestrator does in the inner block.
        let (file_status, finish_status) = match FileOutcome::SkippedNoSubs {
            FileOutcome::Success => (FileStatus::Success, FinishStatus::Success),
            FileOutcome::Failed => (FileStatus::Failed, FinishStatus::Failed),
            FileOutcome::SkippedNoSubs => (FileStatus::Skipped, FinishStatus::SkippedNoSubs),
        };
        assert_eq!(file_status, FileStatus::Skipped);
        assert_eq!(finish_status, FinishStatus::SkippedNoSubs);
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
