//! Tokio GPU worker — serialises all GPU-bound work through a single task.
//!
//! Port of `movie_translator/gpu_queue.py` (`GpuQueue`). The GPU is a single
//! shared resource: every translate / OCR / inpaint call must run **one at a
//! time** across all concurrently-processed files. The Python implementation
//! achieves this with a single-worker `asyncio.Queue`; we mirror it with a
//! tokio mpsc channel feeding one worker task.
//!
//! # Design: sync `GpuExecutor` vs async worker
//!
//! [`crate::gpu::GpuExecutor`] is a **synchronous** trait — it is what
//! [`crate::gpu::resolve_pending_ocr`] and the stage `run` functions consume,
//! and those run inside `tokio::task::spawn_blocking`. The worker, however, is
//! inherently asynchronous (it owns a channel + background task).
//!
//! We reconcile the two without changing the existing trait:
//!
//! * [`GpuWorkerHandle`] is the `Clone` handle shared across file tasks. It
//!   exposes async `submit_*` methods (used directly when an async caller wants
//!   to await GPU work) **and** implements the sync [`GpuExecutor`] trait by
//!   *blocking* on the job's `oneshot` reply.
//! * Blocking is safe because the sync trait methods are only ever invoked from
//!   `spawn_blocking` threads (never from a runtime worker thread). The
//!   orchestrator always wraps the synchronous stage / `resolve_pending_ocr`
//!   calls in `spawn_blocking`, so blocking there parks a blocking-pool thread,
//!   not an async executor thread.
//!
//! The single worker task pulls jobs FIFO and `await`s each `spawn_blocking`
//! call to completion before pulling the next — that `await`-before-next is the
//! serialisation guarantee.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mt_core::{BurnedInResult, DialogueLine, OCRResult};
use mt_ml::TranslateRequest;
use tokio::sync::{mpsc, oneshot};

use crate::error::{PipelineError, Result};
use crate::gpu::{DirectGpuExecutor, GpuExecutor};

/// A unit of GPU work plus the channel to return its result.
///
/// Mirrors the `GpuTask` variants in `gpu_queue.py`. Each variant carries the
/// arguments of the corresponding [`GpuExecutor`] method.
enum Job {
    Translate {
        req: TranslateRequest,
        reply: oneshot::Sender<Result<Vec<DialogueLine>>>,
    },
    OcrPgs {
        video: PathBuf,
        track_index: u32,
        work_dir: PathBuf,
        reply: oneshot::Sender<Result<Option<PathBuf>>>,
    },
    OcrBurnedIn {
        video: PathBuf,
        output_dir: PathBuf,
        crop_ratio: f64,
        fps: u32,
        reply: oneshot::Sender<Result<BurnedInResult>>,
    },
    Inpaint {
        video: PathBuf,
        output: PathBuf,
        device: String,
        backend: String,
        ocr_results: Vec<OCRResult>,
        reply: oneshot::Sender<Result<PathBuf>>,
    },
}

/// A serialised GPU worker backed by a single tokio task.
///
/// Spawn one of these per run; clone its [`handle`](GpuWorker::handle) and share
/// it across all file tasks. Dropping the worker (and all handles) closes the
/// channel and lets the worker task finish (FIFO drain of any queued jobs).
pub struct GpuWorker {
    handle: GpuWorkerHandle,
    task: tokio::task::JoinHandle<()>,
    stop: oneshot::Sender<()>,
}

impl GpuWorker {
    /// Spawn a worker that executes jobs via `DirectGpuExecutor`
    /// (the real `mt_ml` helper scripts).
    pub fn spawn() -> Self {
        Self::spawn_with(DirectGpuExecutor::new())
    }

    /// Spawn a worker executing jobs via the supplied (sync) [`GpuExecutor`].
    ///
    /// The executor is moved into the worker task and used to run each job on a
    /// `spawn_blocking` thread, one at a time. Tests pass a fake executor here.
    pub fn spawn_with<E>(executor: E) -> Self
    where
        E: GpuExecutor + Send + Sync + 'static,
    {
        // Unbounded so `submit` never blocks the producer (matches the Python
        // `asyncio.Queue()` default of unbounded capacity).
        let (tx, mut rx) = mpsc::unbounded_channel::<Job>();
        let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
        let executor = Arc::new(executor);

        let task = tokio::spawn(async move {
            // FIFO: pull one job, run it to completion, then pull the next.
            // Exit when the channel closes (all senders dropped) OR an explicit
            // stop signal arrives — after which any already-queued jobs are
            // drained so in-flight submissions still get a reply.
            loop {
                let job = tokio::select! {
                    biased;
                    maybe = rx.recv() => match maybe {
                        Some(job) => job,
                        None => break, // all handles dropped
                    },
                    _ = &mut stop_rx => {
                        // Drain whatever is already queued, then stop.
                        while let Ok(job) = rx.try_recv() {
                            let exec = executor.clone();
                            let _ = tokio::task::spawn_blocking(move || {
                                run_job(exec.as_ref(), job)
                            })
                            .await;
                        }
                        break;
                    }
                };
                let exec = executor.clone();
                // spawn_blocking because the underlying `mt_ml` calls are
                // blocking subprocess spawns (mirrors Python's
                // `asyncio.to_thread(task.execute)`).
                let _ = tokio::task::spawn_blocking(move || run_job(exec.as_ref(), job)).await;
            }
        });

        GpuWorker {
            handle: GpuWorkerHandle { tx },
            task,
            stop: stop_tx,
        }
    }

    /// A cloneable handle for submitting work. Share across file tasks.
    pub fn handle(&self) -> GpuWorkerHandle {
        self.handle.clone()
    }

    /// Stop the worker after draining queued jobs, then wait for it to exit.
    ///
    /// Port of `GpuQueue.shutdown`. Sends an explicit stop signal so shutdown is
    /// deterministic even if external [`GpuWorkerHandle`] clones are still alive
    /// (they simply become inert — further submissions error with a
    /// "worker stopped" error). Already-queued jobs are drained first so in-flight
    /// submissions still receive their reply.
    pub async fn shutdown(self) {
        drop(self.handle);
        let _ = self.stop.send(());
        let _ = self.task.await;
    }
}

/// Run a single job synchronously and send the result back.
fn run_job(executor: &dyn GpuExecutor, job: Job) {
    match job {
        Job::Translate { req, reply } => {
            let _ = reply.send(executor.translate(&req));
        }
        Job::OcrPgs {
            video,
            track_index,
            work_dir,
            reply,
        } => {
            let _ = reply.send(executor.ocr_pgs(&video, track_index, &work_dir));
        }
        Job::OcrBurnedIn {
            video,
            output_dir,
            crop_ratio,
            fps,
            reply,
        } => {
            let _ = reply.send(executor.ocr_burned_in(&video, &output_dir, crop_ratio, fps));
        }
        Job::Inpaint {
            video,
            output,
            device,
            backend,
            ocr_results,
            reply,
        } => {
            let _ = reply.send(executor.inpaint(&video, &output, &device, &backend, &ocr_results));
        }
    }
}

/// Error message used when the worker task has gone away before replying.
const WORKER_GONE: &str = "GPU worker stopped before completing the task";

/// A cloneable handle to submit work to the [`GpuWorker`].
///
/// Implements the sync [`GpuExecutor`] trait (blocking; safe only from
/// `spawn_blocking` threads) and offers async `submit_*` methods.
#[derive(Clone)]
pub struct GpuWorkerHandle {
    tx: mpsc::UnboundedSender<Job>,
}

impl GpuWorkerHandle {
    fn send(&self, job: Job) -> std::result::Result<(), PipelineError> {
        self.tx
            .send(job)
            .map_err(|_| PipelineError::Stage(WORKER_GONE.to_string()))
    }

    /// Submit a translation job and await its result (async path).
    pub async fn translate_async(&self, req: TranslateRequest) -> Result<Vec<DialogueLine>> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::Translate { req, reply })?;
        rx.await
            .map_err(|_| PipelineError::Stage(WORKER_GONE.to_string()))?
    }

    /// Submit a PGS OCR job and await its result (async path).
    pub async fn ocr_pgs_async(
        &self,
        video: PathBuf,
        track_index: u32,
        work_dir: PathBuf,
    ) -> Result<Option<PathBuf>> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::OcrPgs {
            video,
            track_index,
            work_dir,
            reply,
        })?;
        rx.await
            .map_err(|_| PipelineError::Stage(WORKER_GONE.to_string()))?
    }

    /// Submit a burned-in OCR job and await its result (async path).
    pub async fn ocr_burned_in_async(
        &self,
        video: PathBuf,
        output_dir: PathBuf,
        crop_ratio: f64,
        fps: u32,
    ) -> Result<BurnedInResult> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::OcrBurnedIn {
            video,
            output_dir,
            crop_ratio,
            fps,
            reply,
        })?;
        rx.await
            .map_err(|_| PipelineError::Stage(WORKER_GONE.to_string()))?
    }

    /// Submit an inpaint job and await its result (async path).
    pub async fn inpaint_async(
        &self,
        video: PathBuf,
        output: PathBuf,
        device: String,
        backend: String,
        ocr_results: Vec<OCRResult>,
    ) -> Result<PathBuf> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::Inpaint {
            video,
            output,
            device,
            backend,
            ocr_results,
            reply,
        })?;
        rx.await
            .map_err(|_| PipelineError::Stage(WORKER_GONE.to_string()))?
    }
}

/// Block the current (blocking-pool) thread until the worker replies.
///
/// Only safe to call from a `spawn_blocking` thread. The `GpuExecutor` impl
/// below relies on exactly that invariant.
fn block_on_reply<T>(rx: oneshot::Receiver<Result<T>>) -> Result<T> {
    match rx.blocking_recv() {
        Ok(result) => result,
        Err(_) => Err(PipelineError::Stage(WORKER_GONE.to_string())),
    }
}

impl GpuExecutor for GpuWorkerHandle {
    fn translate(&self, req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::Translate {
            req: req.clone(),
            reply,
        })?;
        block_on_reply(rx)
    }

    fn ocr_pgs(&self, video: &Path, track_index: u32, work_dir: &Path) -> Result<Option<PathBuf>> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::OcrPgs {
            video: video.to_path_buf(),
            track_index,
            work_dir: work_dir.to_path_buf(),
            reply,
        })?;
        block_on_reply(rx)
    }

    fn ocr_burned_in(
        &self,
        video: &Path,
        output_dir: &Path,
        crop_ratio: f64,
        fps: u32,
    ) -> Result<BurnedInResult> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::OcrBurnedIn {
            video: video.to_path_buf(),
            output_dir: output_dir.to_path_buf(),
            crop_ratio,
            fps,
            reply,
        })?;
        block_on_reply(rx)
    }

    fn inpaint(
        &self,
        video: &Path,
        output: &Path,
        device: &str,
        backend: &str,
        ocr_results: &[OCRResult],
    ) -> Result<PathBuf> {
        let (reply, rx) = oneshot::channel();
        self.send(Job::Inpaint {
            video: video.to_path_buf(),
            output: output.to_path_buf(),
            device: device.to_string(),
            backend: backend.to_string(),
            ocr_results: ocr_results.to_vec(),
            reply,
        })?;
        block_on_reply(rx)
    }
}

/// A [`GpuExecutor`] that records the maximum observed concurrency.
///
/// Exposed for orchestrator tests too (so they can assert serialisation across
/// concurrent files). Each call increments a live counter, sleeps briefly, and
/// decrements it; `max_concurrency` is the peak seen.
#[doc(hidden)]
pub struct ConcurrencyProbe {
    live: std::sync::atomic::AtomicUsize,
    max: std::sync::atomic::AtomicUsize,
    calls: std::sync::atomic::AtomicUsize,
    delay: std::time::Duration,
    shared: Arc<ConcurrencyStats>,
}

/// Snapshot counters shared with a [`ConcurrencyProbe`].
#[doc(hidden)]
#[derive(Default)]
pub struct ConcurrencyStats {
    pub max_concurrency: std::sync::atomic::AtomicUsize,
    pub total_calls: std::sync::atomic::AtomicUsize,
}

impl ConcurrencyProbe {
    /// Create a probe and a shared stats handle the test can read afterwards.
    pub fn new(delay: std::time::Duration) -> (Self, Arc<ConcurrencyStats>) {
        let shared = Arc::new(ConcurrencyStats::default());
        let probe = ConcurrencyProbe {
            live: std::sync::atomic::AtomicUsize::new(0),
            max: std::sync::atomic::AtomicUsize::new(0),
            calls: std::sync::atomic::AtomicUsize::new(0),
            delay,
            shared: shared.clone(),
        };
        (probe, shared)
    }

    fn enter(&self) {
        use std::sync::atomic::Ordering;
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(now, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.delay);
        self.live.fetch_sub(1, Ordering::SeqCst);
        // Publish to the shared snapshot.
        self.shared
            .max_concurrency
            .fetch_max(self.max.load(Ordering::SeqCst), Ordering::SeqCst);
        self.shared.total_calls.fetch_add(1, Ordering::SeqCst);
    }
}

impl GpuExecutor for ConcurrencyProbe {
    fn translate(&self, _req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
        self.enter();
        Ok(vec![DialogueLine {
            start_ms: 0,
            end_ms: 1,
            text: "x".into(),
        }])
    }
    fn ocr_pgs(&self, _v: &Path, _t: u32, _w: &Path) -> Result<Option<PathBuf>> {
        self.enter();
        Ok(None)
    }
    fn ocr_burned_in(&self, _v: &Path, _o: &Path, _c: f64, _f: u32) -> Result<BurnedInResult> {
        self.enter();
        Err(PipelineError::Stage("none".into()))
    }
    fn inpaint(
        &self,
        _v: &Path,
        out: &Path,
        _d: &str,
        _b: &str,
        _o: &[OCRResult],
    ) -> Result<PathBuf> {
        self.enter();
        Ok(out.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    /// Submitting several GPU jobs concurrently from many tasks must execute
    /// them ONE AT A TIME (the core serialisation property of `GpuQueue`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn jobs_serialise_across_concurrent_submitters() {
        let (probe, stats) = ConcurrencyProbe::new(Duration::from_millis(20));
        let worker = GpuWorker::spawn_with(probe);
        let handle = worker.handle();

        // Fire 8 async submissions concurrently.
        let mut joins = Vec::new();
        for _ in 0..8 {
            let h = handle.clone();
            joins.push(tokio::spawn(async move {
                h.translate_async(TranslateRequest {
                    lines: vec![],
                    device: "cpu".into(),
                    batch_size: 1,
                    model: "allegro".into(),
                    proper_nouns: None,
                })
                .await
            }));
        }
        for j in joins {
            j.await.unwrap().unwrap();
        }

        assert_eq!(stats.total_calls.load(Ordering::SeqCst), 8);
        assert_eq!(
            stats.max_concurrency.load(Ordering::SeqCst),
            1,
            "GPU jobs must never overlap"
        );

        worker.shutdown().await;
    }

    /// The sync `GpuExecutor` impl (used by `resolve_pending_ocr`) must work
    /// from inside `spawn_blocking` and also serialise.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sync_executor_path_from_spawn_blocking_serialises() {
        let (probe, stats) = ConcurrencyProbe::new(Duration::from_millis(20));
        let worker = GpuWorker::spawn_with(probe);
        let handle = worker.handle();

        let mut joins = Vec::new();
        for _ in 0..6 {
            let h = handle.clone();
            joins.push(tokio::task::spawn_blocking(move || {
                // Sync trait call — blocks this blocking-pool thread.
                h.ocr_pgs(Path::new("/tmp/x.mkv"), 0, Path::new("/tmp/wd"))
            }));
        }
        for j in joins {
            j.await.unwrap().unwrap();
        }

        assert_eq!(stats.total_calls.load(Ordering::SeqCst), 6);
        assert_eq!(stats.max_concurrency.load(Ordering::SeqCst), 1);
        worker.shutdown().await;
    }

    /// FIFO order is preserved: jobs submitted in sequence run in sequence.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fifo_order_preserved() {
        use std::sync::Mutex;

        struct OrderProbe {
            order: Arc<Mutex<Vec<u32>>>,
        }
        impl GpuExecutor for OrderProbe {
            fn translate(&self, req: &TranslateRequest) -> Result<Vec<DialogueLine>> {
                // Encode the submission index in batch_size.
                self.order.lock().unwrap().push(req.batch_size);
                Ok(vec![])
            }
            fn ocr_pgs(&self, _v: &Path, _t: u32, _w: &Path) -> Result<Option<PathBuf>> {
                unreachable!()
            }
            fn ocr_burned_in(
                &self,
                _v: &Path,
                _o: &Path,
                _c: f64,
                _f: u32,
            ) -> Result<BurnedInResult> {
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

        let order = Arc::new(Mutex::new(Vec::new()));
        let worker = GpuWorker::spawn_with(OrderProbe {
            order: order.clone(),
        });
        let handle = worker.handle();

        for i in 0..10u32 {
            handle
                .translate_async(TranslateRequest {
                    lines: vec![],
                    device: "cpu".into(),
                    batch_size: i,
                    model: "allegro".into(),
                    proper_nouns: None,
                })
                .await
                .unwrap();
        }
        worker.shutdown().await;
        assert_eq!(*order.lock().unwrap(), (0..10).collect::<Vec<_>>());
    }
}
