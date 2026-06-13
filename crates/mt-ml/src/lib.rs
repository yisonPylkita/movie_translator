//! ML inference drivers — pure Rust, no Python.
//!
//! ML inference that previously ran in Python (Apple Vision, MLX model) now
//! runs natively: Apple Vision OCR calls a compiled Swift bridge, the Apple
//! Translation backend calls the Swift bridge directly, and the inpainting
//! algorithm is a pure Rust Telea implementation.
//!
//! The embedded CPython (PyO3) dependency has been removed entirely.
//! No Python or venv is needed at build or runtime.

pub mod hardsub;
pub mod inpaint;
pub mod ocr;
pub mod transcription;
pub mod translate;

pub use hardsub::{hardsub_download, hardsub_ocr_clean};
pub use inpaint::inpaint;
pub use ocr::{is_vision_ocr_available, ocr_burned_in, ocr_pgs};
pub use transcription::transcribe_to_srt;
pub use translate::{TranslateRequest, TranslateResponse, translate};
