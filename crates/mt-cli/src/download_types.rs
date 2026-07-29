//! Shared types for the anime downloader overhaul.
//!
//! Defines [`Quality`], [`EpisodeInput`], [`JsonInput`], [`DownloadState`],
//! and the event types used between the download engine and the TUI.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

// ── Quality type ───────────────────────────────────────────────────────────

/// Resolution quality. Higher height = better.
/// Unknown height treated as 0 (lowest) but flagged in TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Quality {
    pub height: u32,
    pub codec: Option<Codec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Codec {
    H264,
    H265,
    AV1,
    VP9,
}

impl Quality {
    pub fn new(height: u32) -> Self {
        Self {
            height,
            codec: None,
        }
    }

    pub fn with_codec(height: u32, codec: Codec) -> Self {
        Self {
            height,
            codec: Some(codec),
        }
    }

    /// Rank: primary sort by height (desc), secondary by codec preference.
    pub fn rank(&self) -> i64 {
        let codec_bonus = match self.codec {
            Some(Codec::H265) | Some(Codec::AV1) => 10,
            Some(Codec::H264) | Some(Codec::VP9) => 5,
            None => 0,
        };
        (self.height as i64) * 100 + codec_bonus
    }

    pub fn is_unknown(&self) -> bool {
        self.height == 0
    }
}

impl std::fmt::Display for Quality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_unknown() {
            write!(f, "unknown quality")
        } else {
            write!(f, "{}p", self.height)
        }
    }
}

// ── JSON schema types ──────────────────────────────────────────────────────

/// Root of the canonical JSON episode-list schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonInput {
    /// Optional display name for the anime.
    #[serde(default)]
    pub title: Option<String>,
    /// Episodes in the season (ordered by episode number).
    pub episodes: Vec<EpisodeInput>,
}

/// One episode entry in the JSON input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeInput {
    /// Episode number (1-based). Must be present and > 0.
    pub episode: i64,
    /// One or more mirror URLs for this episode.
    /// Must have at least 1 entry. Must not be empty strings.
    pub urls: Vec<String>,
    /// Optional quality metadata for quality-first mirror selection.
    #[serde(default)]
    pub quality: Option<QualityMeta>,
}

/// Optional quality metadata for a single episode's mirrors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMeta {
    /// Resolution height in pixels (e.g. 1080, 720, 480).
    pub height: u32,
    /// Optional codec hint.
    #[serde(default)]
    pub codec: Option<String>,
}

// ── Download state machine ─────────────────────────────────────────────────

/// Phase of a single episode download.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Waiting for host locks
    Queued,
    /// Measuring mirrors to find fastest
    Measuring,
    /// Selected winner, actively downloading
    Downloading {
        pct: f64,
        speed: String,
        eta: String,
        downloaded: u64,
        total: u64,
    },
    /// Successfully completed
    Done { host: String, size_mb: f64 },
    /// All mirrors exhausted
    Failed,
    /// Cancelled by user
    Cancelled,
}

impl Phase {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Phase::Done { .. } | Phase::Failed | Phase::Cancelled)
    }
}

/// State for one episode in the download engine.
#[derive(Debug, Clone)]
pub struct EpisodeState {
    pub number: i64,
    pub mirrors: Vec<MirrorState>,
    pub winner: Option<String>,
    pub phase: Phase,
}

/// State for one mirror of an episode.
#[derive(Debug, Clone)]
pub struct MirrorState {
    pub host: String,
    pub url: String,
    pub quality: Option<Quality>,
    pub bps: Option<f64>,
    pub active: bool,
}

// ── Events from engine to TUI ──────────────────────────────────────────────

/// Events sent from the download engine to the TUI.
#[derive(Debug, Clone)]
pub enum EpEvent {
    /// Episode started queueing or measuring.
    Measuring { ep: i64, host: String },
    /// Measurement result: average bytes/sec.
    Measured { ep: i64, host: String, bps: f64 },
    /// Mirror skipped because another episode holds the host lock.
    MirrorBusy { ep: i64, host: String },
    /// A mirror was selected as the winner; full download starting.
    Winner { ep: i64, host: String },
    /// Download progress update from the winning mirror.
    Progress {
        ep: i64,
        host: String,
        pct: f64,
        speed: String,
        eta: String,
        downloaded: u64,
        total: u64,
    },
    /// Episode downloaded successfully.
    Done { ep: i64, host: String, size_mb: f64 },
    /// All mirrors for this episode failed.
    Failed { ep: i64 },
    /// Mirror measurement/attempt ended (killed or failed), remove from list.
    MirrorDone {
        ep: i64,
        host: String,
        success: bool,
    },
    /// Episode measurement phase complete, starting download.
    MeasurementComplete { ep: i64 },
    /// Episode cancelled.
    Cancelled { ep: i64 },
}

// ── JSON validation ────────────────────────────────────────────────────────

/// Error types for JSON input validation.
#[derive(Debug)]
pub enum JsonValidationError {
    NoEpisodes,
    EpisodeMissingNumber,
    EpisodeZeroUrls(usize),
    EmptyUrl(usize),
    DuplicateEpisode(i64),
    ParseError(String),
    LegacyFormat,
}

use std::fmt;

impl fmt::Display for JsonValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonValidationError::NoEpisodes => {
                write!(f, "JSON input has no episodes array")
            }
            JsonValidationError::EpisodeMissingNumber => {
                write!(f, "Episode entry missing required 'episode' field")
            }
            JsonValidationError::EpisodeZeroUrls(idx) => {
                write!(f, "Episode {} has zero URLs (must have at least 1)", idx)
            }
            JsonValidationError::EmptyUrl(idx) => {
                write!(f, "Episode {} has an empty URL string", idx)
            }
            JsonValidationError::DuplicateEpisode(ep) => {
                write!(f, "Duplicate episode number: {}", ep)
            }
            JsonValidationError::ParseError(msg) => {
                write!(f, "Failed to parse JSON: {}", msg)
            }
            JsonValidationError::LegacyFormat => {
                write!(
                    f,
                    "Legacy resolver JSON detected. This format is no longer supported. \
                    Use canonical format: {{\"title\": \"...\", \"episodes\": [{{\"episode\": 1, \"urls\": [\"https://...\"]}}]}}"
                )
            }
        }
    }
}

impl std::error::Error for JsonValidationError {}

/// Validate a parsed [`JsonInput`]. Returns `Ok(())` or the first error.
pub fn validate_json_input(input: &JsonInput) -> Result<(), JsonValidationError> {
    if input.episodes.is_empty() {
        return Err(JsonValidationError::NoEpisodes);
    }

    let mut seen = HashSet::new();

    for (i, ep) in input.episodes.iter().enumerate() {
        if ep.episode <= 0 {
            return Err(JsonValidationError::EpisodeMissingNumber);
        }
        if !seen.insert(ep.episode) {
            return Err(JsonValidationError::DuplicateEpisode(ep.episode));
        }
        if ep.urls.is_empty() {
            return Err(JsonValidationError::EpisodeZeroUrls(i));
        }
        for url in &ep.urls {
            if url.trim().is_empty() {
                return Err(JsonValidationError::EmptyUrl(i));
            }
        }
    }

    Ok(())
}

/// Detect legacy resolver JSON and provide actionable migration message.
pub fn check_legacy_format(json: &str) -> Result<(), JsonValidationError> {
    let lower = json.to_lowercase();
    if lower.contains("\"resolved\"") || lower.contains("\"embed_url\"") {
        return Err(JsonValidationError::LegacyFormat);
    }
    Ok(())
}

/// Parse a JSON string into a [`JsonInput`], applying validation.
pub fn parse_json_input(json: &str) -> Result<JsonInput, JsonValidationError> {
    check_legacy_format(json)?;
    let input: JsonInput =
        serde_json::from_str(json).map_err(|e| JsonValidationError::ParseError(e.to_string()))?;
    validate_json_input(&input)?;
    Ok(input)
}

// ── Quality extraction from ogladajanime players ───────────────────────────

/// Extract quality height from a player's quality string (e.g. "1080p" → 1080).
/// Parse a yt-dlp speed string to bytes/sec.
pub fn parse_speed_bps(s: &str) -> Option<f64> {
    let s = s.trim();
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());
    let num: f64 = s[..num_end].parse().ok()?;
    let unit = &s[num_end..];
    Some(match unit {
        "KiB/s" => num * 1024.0,
        "MiB/s" => num * 1_048_576.0,
        "GiB/s" => num * 1_073_741_824.0,
        _ => num,
    })
}

/// Extract quality height from a player's quality string (e.g. "1080p" → 1080).
pub fn quality_height_from_str(q: &str) -> u32 {
    let digits: String = q.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse().unwrap_or(0)
}

// ── Host preference ranking ────────────────────────────────────────────────

/// Host preference rank for tie-breaking. Lower = more preferred.
pub fn host_preference_rank(host: &str) -> usize {
    const PREFERRED: &[&str] = &[
        "cda",
        "sibnet",
        "vk",
        "mega",
        "ok",
        "dood",
        "myvi",
        "google",
        "hqq",
        "voe",
        "mp4upload",
    ];
    PREFERRED
        .iter()
        .position(|&h| host.contains(h))
        .unwrap_or(usize::MAX)
}
