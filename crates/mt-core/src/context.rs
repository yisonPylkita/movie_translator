use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::identity::MediaIdentity;
use crate::types::{DialogueLine, OCRResult, SubtitleFile};

/// Pipeline configuration.
///
/// `model_cache` from the Python version is omitted — it holds a PyTorch model
/// object that has no Rust equivalent. It will be handled separately when the
/// ML backend bridge is implemented.
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
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            device: "mps".to_string(),
            batch_size: 16,
            model: "allegro".to_string(),
            extra_models: Vec::new(),
            enable_fetch: true,
            enable_inpaint: false,
            dry_run: false,
            in_place: false,
            workers: 4,
            external_subs_dir: None,
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
    pub fetched_subtitles: Option<std::collections::HashMap<String, Vec<FetchedSubtitle>>>,
    pub english_source: Option<PathBuf>,
    pub dialogue_lines: Option<Vec<DialogueLine>>,
    pub translated_lines: Option<Vec<DialogueLine>>,
    /// Translations from `extra_models`, keyed by model name.
    pub extra_translations: std::collections::HashMap<String, Vec<DialogueLine>>,
    pub font_info: Option<FontInfo>,
    pub subtitle_tracks: Option<Vec<SubtitleFile>>,
    pub ocr_results: Option<Vec<OCRResult>>,
    pub inpainted_video: Option<PathBuf>,
    /// `true` after any stage has probed for burned-in subtitles.
    pub burned_in_probed: bool,
    pub pending_ocr: Option<PendingOcr>,
    // `metrics` (MetricsCollector / NullCollector) is a Python-only concern
    // and is omitted here; observability will be wired separately in Rust.
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
            extra_translations: std::collections::HashMap::new(),
            font_info: None,
            subtitle_tracks: None,
            ocr_results: None,
            inpainted_video: None,
            burned_in_probed: false,
            pending_ocr: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_config_default_matches_python() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.device, "mps");
        assert_eq!(cfg.batch_size, 16);
        assert_eq!(cfg.model, "allegro");
        assert!(cfg.extra_models.is_empty());
        assert!(cfg.enable_fetch);
        assert!(!cfg.enable_inpaint);
        assert!(!cfg.dry_run);
        assert!(!cfg.in_place);
        assert_eq!(cfg.workers, 4);
        assert!(cfg.external_subs_dir.is_none());
    }

    #[test]
    fn pipeline_config_serde_round_trip() {
        let cfg = PipelineConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: PipelineConfig = serde_json::from_str(&json).expect("deserialize");
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
