//! Shared types for the anime downloader overhaul.
//!
//! Defines [`Quality`], [`EpisodeInput`], [`JsonInput`], [`DownloadState`],
//! and the event types used between the download engine and the TUI.
//!
//! Canonical JSON schema v2 is parsed version-aware: `schema_version: 2`
//! documents (mirrors with per-mirror host/quality/subtitle_group) are
//! normalized directly; v1 documents (`{title?, episodes:[{episode, urls}]}`)
//! migrate to the same normalized form with a warning. Both produce
//! [`JsonInput`] with episodes carrying a uniform `mirrors` vec.

use std::collections::HashSet;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::hosts;
/// Re-export for engine compatibility: VK canonicalization lives in
/// [`crate::hosts`] as the single source of truth.
pub use crate::hosts::try_canonicalize_vk_url;

// ── Schema constants ───────────────────────────────────────────────────────

/// Mirror-count warning threshold per episode (warn, don't reject).
pub const MIRROR_WARN_THRESHOLD: usize = 16;
/// `resolved_at` staleness warning threshold.
pub const STALE_RESOLVED_AT_SECS: u64 = 24 * 3600;

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

impl fmt::Display for Quality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unknown() {
            write!(f, "unknown quality")
        } else {
            write!(f, "{}p", self.height)
        }
    }
}

// ── JSON schema types ──────────────────────────────────────────────────────

/// One normalized mirror of an episode (canonical v2 shape).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Mirror {
    /// Host short name (e.g. `"cda"`) or `None` when unknown.
    pub host: Option<String>,
    /// Quality label (e.g. `"1080p"`) or `None`.
    pub quality: Option<String>,
    /// Fan-sub group that produced this mirror, or `None`.
    pub subtitle_group: Option<String>,
    /// Canonical download URL (VK `video_ext.php` already rewritten).
    pub url: String,
}

/// Root of the normalized JSON episode-list input.
///
/// Both schema v2 and legacy v1 parse into this uniform shape:
/// every episode carries a `mirrors` vec plus a flat `urls` vec (kept for
/// engine compatibility — `mirrors[].url` order matches `urls`).
#[derive(Debug, Clone)]
pub struct JsonInput {
    /// Optional display name for the anime.
    pub title: Option<String>,
    /// Source page the resolver scraped (v2 only).
    pub source_page: Option<String>,
    /// ISO8601 timestamp of resolution (v2 only; optional).
    pub resolved_at: Option<String>,
    /// Episodes in the season (ordered by episode number).
    pub episodes: Vec<EpisodeInput>,
}

/// One episode entry in the normalized JSON input.
#[derive(Debug, Clone)]
pub struct EpisodeInput {
    /// Episode number (1-based). Must be present and > 0.
    pub episode: i64,
    /// Canonical mirrors (v2-normalized: host/quality/subtitle_group/url).
    pub mirrors: Vec<Mirror>,
    /// Flat URL list, same order as `mirrors` (engine compatibility).
    pub urls: Vec<String>,
    /// Optional quality metadata for quality-first mirror selection.
    pub quality: Option<QualityMeta>,
}

impl EpisodeInput {
    /// Build an episode from flat URLs; mirrors are derived (host extracted
    /// from URL, quality/subtitle_group null).
    pub fn new(episode: i64, urls: Vec<String>) -> Self {
        let mirrors = urls
            .iter()
            .map(|u| Mirror {
                host: host_of_url(u),
                quality: None,
                subtitle_group: None,
                url: u.clone(),
            })
            .collect();
        Self {
            episode,
            mirrors,
            urls,
            quality: None,
        }
    }

    /// Attach episode-level quality metadata.
    pub fn with_quality(mut self, quality: QualityMeta) -> Self {
        self.quality = Some(quality);
        self
    }
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

/// Host name from a URL, e.g. `https://video.sibnet.ru/v.mp4` → `video.sibnet.ru`.
pub fn host_of_url(url: &str) -> Option<String> {
    let u = url.trim();
    let rest = u
        .strip_prefix("https://")
        .or_else(|| u.strip_prefix("http://"))?;
    let host = rest.split(['/', '?', '#']).next()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

// ── Raw schema shapes (private) ────────────────────────────────────────────

/// Canonical v2 raw shape.
#[derive(Debug, Clone, Deserialize)]
struct V2Input {
    #[serde(default)]
    source_page: Option<String>,
    #[serde(default)]
    resolved_at: Option<String>,
    #[serde(default)]
    title: Option<String>,
    episodes: Vec<V2Episode>,
}

#[derive(Debug, Clone, Deserialize)]
struct V2Episode {
    episode: i64,
    #[serde(default)]
    mirrors: Vec<V2Mirror>,
}

#[derive(Debug, Clone, Deserialize)]
struct V2Mirror {
    #[serde(default)]
    host: Option<String>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    subtitle_group: Option<String>,
    url: String,
}

/// Legacy v1 raw shape.
#[derive(Debug, Clone, Deserialize)]
struct V1Input {
    #[serde(default)]
    title: Option<String>,
    episodes: Vec<V1Episode>,
}

#[derive(Debug, Clone, Deserialize)]
struct V1Episode {
    episode: i64,
    urls: Vec<String>,
    #[serde(default)]
    quality: Option<QualityMeta>,
}

impl V2Input {
    fn into_normalized(self, warnings: &mut Vec<String>) -> JsonInput {
        let mut episodes = Vec::with_capacity(self.episodes.len());
        for e in self.episodes {
            let mut seen = HashSet::new();
            let mut mirrors = Vec::new();
            for m in e.mirrors {
                // Dedupe AFTER canonicalization
                let url = hosts::canonicalize(&m.url);
                if !seen.insert(url.clone()) {
                    continue;
                }
                mirrors.push(Mirror {
                    host: m.host,
                    quality: m.quality,
                    subtitle_group: m.subtitle_group,
                    url,
                });
            }
            if mirrors.len() > MIRROR_WARN_THRESHOLD {
                warnings.push(format!(
                    "episode {} has {} mirrors (> {}); downloads may be slow",
                    e.episode,
                    mirrors.len(),
                    MIRROR_WARN_THRESHOLD
                ));
            }
            let urls: Vec<String> = mirrors.iter().map(|m| m.url.clone()).collect();
            let quality = mirrors
                .iter()
                .filter_map(|m| m.quality.as_deref())
                .map(quality_height_from_str)
                .filter(|h| *h > 0)
                .max()
                .map(|height| QualityMeta {
                    height,
                    codec: None,
                });
            episodes.push(EpisodeInput {
                episode: e.episode,
                mirrors,
                urls,
                quality,
            });
        }
        JsonInput {
            title: self.title,
            source_page: self.source_page,
            resolved_at: self.resolved_at,
            episodes,
        }
    }
}

impl V1Input {
    fn into_normalized(self, warnings: &mut Vec<String>) -> JsonInput {
        warnings.push("v1 schema detected; migrated to v2 normalized form".to_string());
        let mut episodes = Vec::with_capacity(self.episodes.len());
        for e in self.episodes {
            let mut seen = HashSet::new();
            let mut mirrors = Vec::new();
            for u in e.urls {
                // Dedupe AFTER canonicalization
                let url = hosts::canonicalize(&u);
                if !seen.insert(url.clone()) {
                    continue;
                }
                mirrors.push(Mirror {
                    host: host_of_url(&url),
                    quality: None,
                    subtitle_group: None,
                    url,
                });
            }
            let urls: Vec<String> = mirrors.iter().map(|m| m.url.clone()).collect();
            episodes.push(EpisodeInput {
                episode: e.episode,
                mirrors,
                urls,
                quality: e.quality,
            });
        }
        JsonInput {
            title: self.title,
            source_page: None,
            resolved_at: None,
            episodes,
        }
    }
}

// ── Download state machine ─────────────────────────────────────────────────

/// Phase of a single episode download.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Waiting for host locks
    Queued,
    /// Inspecting mirror formats (quality discovery)
    Inspecting,
    /// Measuring mirrors to find fastest
    Measuring,
    /// Waiting for another episode to release a host lock
    WaitingHost,
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
    MirrorBusy {
        ep: i64,
        host: String,
        wait_secs: u64,
    },
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
    /// Mirror measurement produced no usable output file (measurement artifact missing).
    MirrorMeasFailed { ep: i64, host: String },
    /// A failed mirror is scheduled for retry after backoff.
    RetryWait {
        ep: i64,
        mirror: String,
        attempt: u32,
        backoff_secs: u64,
    },
    /// Media validation of a downloaded file started.
    ValidationStarted { ep: i64 },
    /// Media validation finished.
    ValidationResult {
        ep: i64,
        ok: bool,
        reason: Option<String>,
    },
    /// Host circuit breaker opened (consecutive failures).
    CircuitOpened { host: String },
    /// Host circuit breaker closed again.
    CircuitClosed { host: String },
    /// Final run summary (plain output / dashboard).
    FinalSummary {
        downloaded: usize,
        skipped: usize,
        failed: usize,
        cancelled: usize,
        /// Authoritative per-episode failure/missing/cancellation reasons,
        /// mirroring [`Outcome::per_episode_reasons`]. Renderers prefer these
        /// over best-effort event-derived reasons.
        per_episode_reasons: Vec<(u32, String)>,
    },
}

// ── JSON validation ────────────────────────────────────────────────────────

/// Error types for JSON input validation.
#[derive(Debug)]
pub enum JsonValidationError {
    NoEpisodes,
    EpisodeMissingNumber,
    /// `episode` present but <= 0 (payload is the offending episode number).
    InvalidEpisodeNumber(i64),
    /// Episode present but carries zero URLs (payload is the episode number,
    /// not the array index — indexes are meaningless once episodes are sorted).
    EpisodeZeroUrls(i64),
    /// Episode has an empty URL string (payload is the episode number).
    EmptyUrl(i64),
    DuplicateEpisode(i64),
    EpisodeOutOfRange(i64),
    UnsupportedUrlScheme(String),
    UnsupportedSchemaVersion(u64),
    ParseError(String),
    LegacyFormat,
}

impl fmt::Display for JsonValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonValidationError::NoEpisodes => {
                write!(f, "JSON input has no episodes array")
            }
            JsonValidationError::EpisodeMissingNumber => {
                write!(f, "Episode entry missing required 'episode' field")
            }
            JsonValidationError::InvalidEpisodeNumber(ep) => {
                write!(f, "invalid episode number {ep}")
            }
            JsonValidationError::EpisodeZeroUrls(ep) => {
                write!(f, "Episode {ep} has zero URLs (must have at least 1)")
            }
            JsonValidationError::EmptyUrl(ep) => {
                write!(f, "Episode {ep} has an empty URL string")
            }
            JsonValidationError::DuplicateEpisode(ep) => {
                write!(f, "Duplicate episode number: {}", ep)
            }
            JsonValidationError::EpisodeOutOfRange(ep) => {
                write!(f, "Episode number {} out of range 1..=99999", ep)
            }
            JsonValidationError::UnsupportedUrlScheme(url) => {
                write!(
                    f,
                    "Unsupported URL scheme in {url:?}: only http/https allowed"
                )
            }
            JsonValidationError::UnsupportedSchemaVersion(n) => {
                write!(
                    f,
                    "Unsupported schema_version {n}; supported: 2 or v1 {{title?, episodes:[{{episode, urls}}]}}"
                )
            }
            JsonValidationError::ParseError(msg) => {
                write!(f, "Failed to parse JSON: {}", msg)
            }
            JsonValidationError::LegacyFormat => {
                write!(
                    f,
                    "unsupported legacy schema; reinstall userscript v4+ (scripts/ogladajanime_resolver.user.js), \
                    click \"Download all\" and re-export; supported: schema_version 2 or v1 {{title?, episodes:[{{episode, urls}}]}}"
                )
            }
        }
    }
}

impl std::error::Error for JsonValidationError {}

/// Validate a normalized [`JsonInput`]. Returns `Ok(())` or the first error.
pub fn validate_json_input(input: &JsonInput) -> Result<(), JsonValidationError> {
    if input.episodes.is_empty() {
        return Err(JsonValidationError::NoEpisodes);
    }

    let mut seen = HashSet::new();

    for ep in input.episodes.iter() {
        if ep.episode <= 0 {
            return Err(JsonValidationError::InvalidEpisodeNumber(ep.episode));
        }
        if ep.episode > 99_999 {
            return Err(JsonValidationError::EpisodeOutOfRange(ep.episode));
        }
        if !seen.insert(ep.episode) {
            return Err(JsonValidationError::DuplicateEpisode(ep.episode));
        }
        if ep.urls.is_empty() {
            return Err(JsonValidationError::EpisodeZeroUrls(ep.episode));
        }
        for url in &ep.urls {
            if url.trim().is_empty() {
                return Err(JsonValidationError::EmptyUrl(ep.episode));
            }
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err(JsonValidationError::UnsupportedUrlScheme(url.clone()));
            }
        }
    }

    Ok(())
}

/// Detect incompatible legacy resolver JSON (pre-v1 keys).
pub fn check_legacy_format(json: &str) -> Result<(), JsonValidationError> {
    let lower = json.to_lowercase();
    if lower.contains("\"resolved\"") || lower.contains("\"embed_url\"") {
        return Err(JsonValidationError::LegacyFormat);
    }
    Ok(())
}

/// Parse a JSON string into a normalized [`JsonInput`], applying validation.
/// Warnings (v1 migration, stale `resolved_at`, mirror flood) are returned
/// alongside; use [`parse_json_input`] to discard them.
pub fn parse_json_input_with_warnings(
    json: &str,
) -> Result<(JsonInput, Vec<String>), JsonValidationError> {
    check_legacy_format(json)?;
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| JsonValidationError::ParseError(e.to_string()))?;

    let mut warnings = Vec::new();
    let input = match value.get("schema_version").and_then(|v| v.as_u64()) {
        None => {
            // Default v1; tolerate v2-shaped docs missing the version field.
            match serde_json::from_value::<V1Input>(value.clone()) {
                Ok(v1) => v1.into_normalized(&mut warnings),
                Err(_) => match serde_json::from_value::<V2Input>(value.clone()) {
                    Ok(v2) => v2.into_normalized(&mut warnings),
                    Err(e) => return Err(JsonValidationError::ParseError(e.to_string())),
                },
            }
        }
        Some(1) => match serde_json::from_value::<V1Input>(value) {
            Ok(v1) => v1.into_normalized(&mut warnings),
            Err(e) => return Err(JsonValidationError::ParseError(e.to_string())),
        },
        Some(2) => match serde_json::from_value::<V2Input>(value) {
            Ok(v2) => v2.into_normalized(&mut warnings),
            Err(e) => return Err(JsonValidationError::ParseError(e.to_string())),
        },
        Some(n) => return Err(JsonValidationError::UnsupportedSchemaVersion(n)),
    };

    validate_json_input(&input)?;

    // Stale resolved_at warning (optional field; warning only)
    if let Some(ts) = input.resolved_at.as_deref() {
        match parse_iso8601_epoch(ts) {
            Some(epoch) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let age = now.saturating_sub(epoch);
                if age > STALE_RESOLVED_AT_SECS {
                    warnings.push(format!(
                        "resolved_at {ts} is stale ({}h old); mirrors may have moved",
                        age / 3600
                    ));
                }
            }
            None => warnings.push(format!("resolved_at {ts} could not be parsed as ISO8601")),
        }
    }

    Ok((input, warnings))
}

/// Parse a JSON string into a normalized [`JsonInput`], applying validation.
pub fn parse_json_input(json: &str) -> Result<JsonInput, JsonValidationError> {
    parse_json_input_with_warnings(json).map(|(input, _warnings)| input)
}

// ── Quality extraction from ogladajanime players ───────────────────────────

/// Parse a yt-dlp speed string to bytes/sec.
pub fn parse_speed_bps(s: &str) -> Option<f64> {
    let s = s.trim();
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());
    let num: f64 = s[..num_end].parse().ok()?;
    let unit = s[num_end..].trim();
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

// ── Slug sanitization ──────────────────────────────────────────────────────

/// Sanitize a title/slug: lowercase, alphanumeric + dash only, max 80 chars,
/// no `..` or `/` (path traversal safe). Falls back to `"untitled"`.
pub fn sanitize_slug(s: &str) -> String {
    const MAX_SLUG: usize = 80;
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    // Collapse dash runs
    let mut collapsed = String::with_capacity(out.len());
    let mut prev_dash = false;
    for c in out.chars() {
        if c == '-' {
            if prev_dash {
                continue;
            }
            prev_dash = true;
        } else {
            prev_dash = false;
        }
        collapsed.push(c);
    }
    let trimmed = collapsed.trim_matches('-');
    let truncated: String = trimmed.chars().take(MAX_SLUG).collect();
    let slug = if truncated.is_empty() {
        "untitled".to_string()
    } else {
        truncated
    };
    debug_assert!(!slug.contains("..") && !slug.contains('/'));
    slug
}

// ── ISO8601 helper (no chrono; subset needed for resolved_at) ─────────────

/// Days in `month` (1-12) of `year`, honouring Gregorian leap years.
/// Returns 0 for an out-of-range month (caller rejects via month check).
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap { 29 } else { 28 }
        }
        _ => 0,
    }
}

/// Parse a simplified ISO8601 UTC timestamp (`2025-07-30T12:34:56Z`, with
/// optional `±HH:MM` offset and fractional seconds) into unix seconds.
///
/// Strict range validation: month 1-12, day within the month (leap years
/// honoured), hour 0-23, minute 0-59, second 0-60. Second 60 is accepted as
/// an RFC 3339 leap second (the pre-existing parser already tolerated it;
/// it is treated as +60s — a 1s skew that only affects the staleness
/// heuristic). Out-of-range values return `None` (rejected). Timezone
/// handling is unchanged.
pub(crate) fn parse_iso8601_epoch(s: &str) -> Option<u64> {
    let s = s.trim();
    let (date_part, rest) = s.split_once('T')?;
    let mut date_nums = date_part.split('-');
    let year: i64 = date_nums.next()?.parse().ok()?;
    let month: i64 = date_nums.next()?.parse().ok()?;
    let day: i64 = date_nums.next()?.parse().ok()?;
    if date_nums.next().is_some() {
        return None;
    }
    if !(1..=12).contains(&month) {
        return None;
    }
    if !(1..=days_in_month(year, month)).contains(&day) {
        return None;
    }

    let (time_part, tz) = match rest.rfind(['+', '-']) {
        Some(idx) if idx > 0 => (&rest[..idx], Some(&rest[idx..])),
        _ => (rest.trim_end_matches('Z'), None),
    };

    let mut time_nums = time_part.split(':');
    let hour: i64 = time_nums.next()?.parse().ok()?;
    let minute: i64 = time_nums.next()?.parse().ok()?;
    let sec_str = time_nums.next().unwrap_or("0");
    if time_nums.next().is_some() {
        return None;
    }
    let second: i64 = sec_str.split('.').next()?.parse().ok()?;
    if !(0..=23).contains(&hour) || !(0..=59).contains(&minute) || !(0..=60).contains(&second) {
        return None;
    }

    let days = days_from_civil(year, month, day);
    let mut epoch = days * 86400 + hour * 3600 + minute * 60 + second;

    if let Some(tz) = tz
        && tz != "Z"
    {
        let sign = if tz.starts_with('-') { -1 } else { 1 };
        let tz_clean = tz.trim_start_matches(['+', '-']);
        let (tz_h, tz_m) = tz_clean.split_once(':')?;
        let offset: i64 = tz_h.parse::<i64>().ok()? * 3600 + tz_m.parse::<i64>().ok()? * 60;
        epoch -= sign * offset;
    }

    if epoch < 0 {
        return None;
    }
    Some(epoch as u64)
}

/// Days since 1970-01-01 for a proleptic Gregorian civil date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

// ── Host preference ranking ────────────────────────────────────────────────

/// Host preference rank for tie-breaking. Lower = more preferred.
/// Resolves the host string (URL or hostname) via [`crate::hosts::identify_host`];
/// ranking order lives in [`crate::hosts`] as the single source of truth.
pub fn host_preference_rank(host: &str) -> usize {
    let id = hosts::identify_host(host);
    hosts::host_preference_rank()
        .iter()
        .position(|h| *h == id)
        .unwrap_or(usize::MAX)
}
