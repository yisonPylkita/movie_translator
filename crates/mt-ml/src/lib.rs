//! ML inference drivers, embedded.
//!
//! ML inference that still runs in Python (Apple Vision, MLX model) is called
//! through PyO3 embedding.  The Apple Translation backend has been rewritten
//! in Rust-native code (see [`apple_translate`]) and calls the Swift bridge
//! binary directly.
//! Model objects (e.g. `SubtitleTranslator`) are loaded ONCE per binary run
//! and reused across every file in a `run_all`.
//!
//! Build requirement: set `PYO3_PYTHON=$(repo)/.venv/bin/python` when
//! invoking cargo so PyO3 links against the venv interpreter (which has the
//! `movie_translator` dependencies installed). The justfile + CI do this.

pub mod apple_translate;
pub mod backend;
pub mod hardsub;
pub mod inpaint;
pub mod ocr;
pub mod transcription;
pub mod translate;

pub use backend::{vision_ocr_available, ParsedFilename};
pub use hardsub::{hardsub_download, hardsub_ocr_clean};
pub use inpaint::inpaint;
pub use ocr::{ocr_burned_in, ocr_pgs};
pub use transcription::transcribe_to_srt;
pub use translate::{translate, TranslateRequest, TranslateResponse};
