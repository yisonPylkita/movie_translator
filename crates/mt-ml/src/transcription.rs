//! Audio→subtitle transcription via the embedded `movie_translator.transcription`
//! package. Two engines (picked by the `benchmarks/asr` bake-off): Apple
//! SpeechAnalyzer (`"apple"`, macOS 26+, ANE) and mlx-whisper large-v3
//! (`"whisper"`, Metal). Accelerated ML — route through the serialised GPU
//! worker like the other stages.

use std::path::{Path, PathBuf};

use mt_core::Result;

/// Transcribe the video's `language` audio track to an SRT, or `None` when the
/// track / engine / usable lines are missing (the pipeline then falls through
/// to its normal no-English-source handling).
pub fn transcribe_to_srt(
    video: &Path,
    output_dir: &Path,
    language: &str,
    engine: &str,
) -> Result<Option<PathBuf>> {
    crate::backend::transcribe_to_srt(video, output_dir, language, engine)
}
