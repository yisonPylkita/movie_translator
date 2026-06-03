//! ML inference drivers, embedded.
//!
//! ML inference itself stays in Python (PyTorch / Apple Vision / etc.). This
//! crate embeds CPython via PyO3 and calls into the `movie_translator`
//! Python package directly — no subprocesses, no JSON, no script files.
//! Model objects (e.g. `SubtitleTranslator`) are loaded ONCE per binary run
//! and reused across every file in a `run_all`.
//!
//! Build requirement: set `PYO3_PYTHON=$(repo)/.venv/bin/python` when
//! invoking cargo so PyO3 links against the venv interpreter (which has the
//! `movie_translator` dependencies installed). The justfile + CI do this.

pub mod backend;
pub mod hardsub;
pub mod inpaint;
pub mod ocr;
pub mod translate;

pub use backend::{vision_ocr_available, ParsedFilename};
pub use hardsub::{hardsub_download, hardsub_ocr_clean};
pub use inpaint::inpaint;
pub use ocr::{ocr_burned_in, ocr_pgs};
pub use translate::{translate, TranslateRequest, TranslateResponse};
