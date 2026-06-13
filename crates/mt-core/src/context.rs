use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::identity::MediaIdentity;
use crate::types::{DialogueLine, OCRResult, SubtitleFile};

/// Pipeline configuration.
///
/// There is no in-memory model cache here: ML inference runs out-of-process via
/// the `ml/*.py` helpers, so there is no live model object to hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub device: String,
    pub batch_size: u32,
    pub model: String,
    /// Extra translation backends to run alongside `model`. Each adds a
    /// separate subtitle track (e.g. on macOS: Allegro + Apple Translation).
    pub extra_models: Vec<String>,
    pub enable_fetch: bool,
    pub enable_inpaint: bool,
    pub dry_run: bool,
    pub in_place: bool,
    pub workers: u32,
    pub external_subs_dir: Option<PathBuf>,
    /// Keep per-file work/`.translate_temp` directories after a successful run
    /// instead of deleting them. Failures always keep artifacts for debugging.
    pub keep_artifacts: bool,
    /// Source Polish subtitles by OCRing burned-in subs from ogladajanime.pl.
    /// Triggers an interactive discovery step (open browser, watch ~/Downloads
    /// for the resolver userscript's JSON) once per run. macOS-only.
    pub enable_hardsub_ocr: bool,
    /// Re-process files that already have Polish subtitles (the run normally
    /// skips them). Useful to re-translate or add a new track to prior outputs.
    pub force: bool,
    /// Source English dialogue from the audio track via ASR when no subtitle
    /// text is found anywhere (`--transcribe`).
    pub enable_transcription: bool,
    /// ASR engine: "apple" (SpeechAnalyzer, macOS 26+) or "whisper"
    /// (mlx-whisper large-v3). See `benchmarks/asr/REPORT.md`.
    pub transcribe_engine: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            device: "mps".to_string(),
            batch_size: 4,
            model: "mlx".to_string(),
            extra_models: Vec::new(),
            enable_fetch: true,
            enable_inpaint: false,
            dry_run: false,
            in_place: false,
            workers: 4,
            external_subs_dir: None,
            keep_artifacts: false,
            enable_hardsub_ocr: false,
            force: false,
            enable_transcription: false,
            transcribe_engine: "apple".to_string(),
        }
    }
}

/// A subtitle file fetched from an external provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FetchedSubtitle {
    pub path: PathBuf,
    /// Provider name, e.g. `"animesub"`.
    pub source: String,
}

/// Font information for subtitle rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FontInfo {
    pub supports_polish: bool,
    pub font_attachments: Vec<PathBuf>,
    pub fallback_font_family: Option<String>,
}

/// An original subtitle track found in the source media.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OriginalTrack {
    pub stream_index: i32,
    pub subtitle_index: i32,
    /// Codec name, e.g. `"subrip"`, `"ass"`.
    pub codec: String,
    pub language: String,
}

/// Typed description of an OCR task deferred to the GPU queue.
///
/// Set by `run()` on `ExtractReferenceStage` or `ExtractEnglishStage`,
/// consumed by `_handle_pending_ocr()` in the async pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingOcr {
    /// `"pgs"` or `"burned_in"`.
    pub r#type: String,
    pub track_id: Option<i32>,
    pub output_dir: PathBuf,
}

/// Full pipeline context, populated progressively as each stage runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineContext {
    // ── Inputs (set before the pipeline runs) ─────────────────────────────
    pub video_path: PathBuf,
    pub work_dir: PathBuf,
    pub config: PipelineConfig,

    // ── Stage outputs (set progressively) ─────────────────────────────────
    /// Identified media metadata. `None` until the identification stage runs.
    pub identity: Option<MediaIdentity>,
    pub reference_path: Option<PathBuf>,
    pub original_english_track: Option<OriginalTrack>,
    pub fetched_subtitles: Option<HashMap<String, Vec<FetchedSubtitle>>>,
    pub english_source: Option<PathBuf>,
    pub dialogue_lines: Option<Vec<DialogueLine>>,
    pub translated_lines: Option<Vec<DialogueLine>>,
    /// Translations from `extra_models`, keyed by model name.
    pub extra_translations: HashMap<String, Vec<DialogueLine>>,
    pub font_info: Option<FontInfo>,
    pub subtitle_tracks: Option<Vec<SubtitleFile>>,
    pub ocr_results: Option<Vec<OCRResult>>,
    pub inpainted_video: Option<PathBuf>,
    /// `true` after any stage has probed for burned-in subtitles.
    pub burned_in_probed: bool,
    /// `true` when `english_source` was produced by ASR transcription of the
    /// audio track (`--transcribe`). Output tracks then carry AI-transcribed
    /// provenance in their titles and the transcript is muxed as an English track.
    #[serde(default)]
    pub english_from_asr: bool,
    pub pending_ocr: Option<PendingOcr>,
    // No metrics/observability collector field: the metrics subsystem is not
    // part of this implementation.
}

impl PipelineConfig {
    /// Whether burned-in (video-frame) OCR may run as the no-subtitle fallback.
    ///
    /// `--transcribe` supersedes it: the user chose the audio track as the
    /// English source, and running frame-OCR on clean video would yield
    /// credit/typesetting junk lines that would preempt ASR (the burned-in
    /// extractor would find something — credits — and set it as
    /// ``english_source`` before ASR runs).
    ///
    /// If both burned-in frame OCR AND ASR are desired, use ``--hardsub-ocr``
    /// instead (it downloads hardsubbed Polish video and runs OCR on that,
    /// independent of English source extraction). PGS/image-track OCR is
    /// unaffected — those are real subtitle tracks, not a fallback.
    pub fn burned_in_fallback_allowed(&self) -> bool {
        !self.enable_transcription
    }
}

impl PipelineContext {
    /// Create a new context with the minimum required inputs.
    pub fn new(video_path: PathBuf, work_dir: PathBuf, config: PipelineConfig) -> Self {
        Self {
            video_path,
            work_dir,
            config,
            identity: None,
            reference_path: None,
            original_english_track: None,
            fetched_subtitles: None,
            english_source: None,
            dialogue_lines: None,
            translated_lines: None,
            extra_translations: HashMap::new(),
            font_info: None,
            subtitle_tracks: None,
            ocr_results: None,
            inpainted_video: None,
            burned_in_probed: false,
            english_from_asr: false,
            pending_ocr: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{from_str, to_string};

    use super::*;

    #[test]
    fn config_defaults() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.device, "mps");
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.model, "mlx");
        assert!(cfg.extra_models.is_empty());
        assert!(cfg.enable_fetch);
        assert!(!cfg.enable_inpaint);
        assert!(!cfg.dry_run);
        assert!(!cfg.in_place);
        assert_eq!(cfg.workers, 4);
        assert!(cfg.external_subs_dir.is_none());
        assert!(!cfg.keep_artifacts);
        assert!(!cfg.enable_hardsub_ocr);
        assert!(!cfg.force);
    }

    #[test]
    fn pipeline_config_serde_round_trip() {
        let cfg = PipelineConfig::default();
        let json = to_string(&cfg).expect("serialize");
        let back = from_str::<PipelineConfig>(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn pipeline_context_new_sets_defaults() {
        let ctx = PipelineContext::new(
            PathBuf::from("/tmp/video.mkv"),
            PathBuf::from("/tmp/work"),
            PipelineConfig::default(),
        );
        assert!(ctx.dialogue_lines.is_none());
        assert!(ctx.translated_lines.is_none());
        assert!(!ctx.burned_in_probed);
        assert!(ctx.extra_translations.is_empty());
    }
}
