//! ML inference drivers.
//!
//! ML inference itself stays in Python (PyTorch / Apple Vision / etc.). This
//! crate spawns single-purpose Python helper scripts under `ml/` — one per
//! stage-per-file, so each model loads once per call. The contracts mirror the
//! `GpuTask` classes in `movie_translator/gpu_queue.py`
//! (`TranslateTask` / `OcrTask` / `InpaintTask`).

pub mod inpaint;
pub mod ocr;
pub mod runner;
pub mod translate;

pub use inpaint::inpaint;
pub use ocr::{ocr_burned_in, ocr_pgs};
pub use runner::{run_script_json, run_script_json_with_timeout, DEFAULT_SCRIPT_TIMEOUT};
pub use translate::{translate, TranslateRequest, TranslateResponse};
