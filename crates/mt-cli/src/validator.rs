//! ffprobe-based media validation for downloaded episodes.
//!
//! [`FfprobeValidator`] spawns `ffprobe` (located via PATH, falling back to
//! `/opt/homebrew/bin` and `/usr/bin`) with a timeout and parses its JSON
//! output by hand with `serde_json` — no extra dependencies.

use std::env;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

use serde_json::Value;

/// Accepted container extensions (kept from the pre-existing heuristic set).
pub const VALID_EXTENSIONS: &[&str] = &["mkv", "mp4", "webm", "flv", "mov", "avi"];

/// Degraded-mode reason used when ffprobe itself is unavailable.
pub const DEGRADED_REASON: &str = "ffprobe unavailable; degraded heuristic validation";

// ── Configuration ──────────────────────────────────────────────────────────

/// Tuning knobs for media validation.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Minimum file size in bytes (default 1 MiB).
    pub min_size_bytes: u64,
    /// Minimum reported duration in seconds (default 1.0).
    pub min_duration_secs: f64,
    /// Reject files with no audio stream (default false: warn only).
    pub require_audio: bool,
    /// Timeout for a single ffprobe invocation (default 15s).
    pub ffprobe_timeout: Duration,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_size_bytes: 1_048_576,
            min_duration_secs: 1.0,
            require_audio: false,
            ffprobe_timeout: Duration::from_secs(15),
        }
    }
}

// ── Outcome ────────────────────────────────────────────────────────────────

/// Result of validating one media file.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationOutcome {
    pub valid: bool,
    /// Rejection reason, or warning text when `valid` with caveats.
    pub reason: Option<String>,
    pub video_stream: bool,
    pub audio_stream: bool,
    pub duration_secs: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub ffprobe_version: Option<String>,
}

impl ValidationOutcome {
    /// Convenience: an invalid outcome with a reason.
    fn invalid(reason: impl Into<String>) -> Self {
        Self {
            valid: false,
            reason: Some(reason.into()),
            video_stream: false,
            audio_stream: false,
            duration_secs: None,
            width: None,
            height: None,
            ffprobe_version: None,
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────────────────

/// Errors surfaced by validation (I/O on the file itself).
#[derive(Debug)]
pub enum ValidatorError {
    Io(std::io::Error),
}

impl fmt::Display for ValidatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidatorError::Io(e) => write!(f, "validation I/O error: {e}"),
        }
    }
}

impl std::error::Error for ValidatorError {}

impl From<std::io::Error> for ValidatorError {
    fn from(e: std::io::Error) -> Self {
        ValidatorError::Io(e)
    }
}

// ── Trait ──────────────────────────────────────────────────────────────────

/// Strategy for validating a downloaded media file.
pub trait MediaValidator {
    fn validate(
        &self,
        path: &Path,
        cfg: &ValidationConfig,
    ) -> Result<ValidationOutcome, ValidatorError>;
}

// ── ffprobe data model ─────────────────────────────────────────────────────

/// Parsed subset of `ffprobe -show_format -show_streams -of json`.
#[derive(Debug, Clone, Default)]
pub struct FfprobeInfo {
    pub format_name: Option<String>,
    pub duration_secs: Option<f64>,
    pub has_video: bool,
    pub video_width: Option<u32>,
    pub video_height: Option<u32>,
    pub has_audio: bool,
}

/// Either usable ffprobe info or a degraded-mode reason string.
#[derive(Debug, Clone)]
pub enum ProbeOutcome {
    Info(FfprobeInfo),
    Degraded(String),
}

/// Parse ffprobe JSON output. Pure; unit-tested against serde fixtures.
pub fn parse_ffprobe_json(json: &str) -> Result<FfprobeInfo, String> {
    let value: Value = serde_json::from_str(json).map_err(|e| format!("invalid JSON: {e}"))?;
    let format = value.get("format").cloned().unwrap_or(Value::Null);
    let format_name = format
        .get("format_name")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let duration_secs = format
        .get("duration")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| format.get("duration").and_then(|v| v.as_f64()));
    let mut has_video = false;
    let mut video_width = None;
    let mut video_height = None;
    let mut has_audio = false;
    if let Some(streams) = value.get("streams").and_then(|v| v.as_array()) {
        for stream in streams {
            let codec_type = stream.get("codec_type").and_then(|v| v.as_str());
            match codec_type {
                Some("video") => {
                    has_video = true;
                    video_width = stream
                        .get("width")
                        .and_then(|v| v.as_u64())
                        .map(|w| w as u32);
                    video_height = stream
                        .get("height")
                        .and_then(|v| v.as_u64())
                        .map(|h| h as u32);
                }
                Some("audio") => {
                    has_audio = true;
                }
                _ => {}
            }
        }
    }
    Ok(FfprobeInfo {
        format_name,
        duration_secs,
        has_video,
        video_width,
        video_height,
        has_audio,
    })
}

/// Apply every validation check except extension/size (those need disk
/// metadata). Pure; unit-tested against all check scenarios.
pub fn validate_checks(
    cfg: &ValidationConfig,
    probe: ProbeOutcome,
    ffprobe_version: Option<String>,
) -> ValidationOutcome {
    let probe = match probe {
        ProbeOutcome::Degraded(reason) => {
            return ValidationOutcome {
                valid: true,
                reason: Some(reason),
                video_stream: false,
                audio_stream: false,
                duration_secs: None,
                width: None,
                height: None,
                ffprobe_version: None,
            };
        }
        ProbeOutcome::Info(info) => info,
    };

    let mut reject: Option<&'static str> = None;
    let mut warnings: Vec<String> = Vec::new();

    // format_name must be non-empty (ffprobe could not identify the container)
    if probe.format_name.as_deref().unwrap_or("").is_empty() {
        reject = Some("ffprobe reported no format");
    }

    // At least one video stream with real dimensions; placeholder dims rejected
    match (probe.has_video, probe.video_width, probe.video_height) {
        (false, _, _) => {
            if reject.is_none() {
                reject = Some("no video stream");
            }
        }
        (true, Some(w), Some(h)) if w == 0 || h == 0 || (w == 1 && h == 1) => {
            if reject.is_none() {
                reject = Some("placeholder dimensions");
            }
        }
        (true, Some(_), Some(_)) => {}
        (true, _, _) => {
            if reject.is_none() {
                reject = Some("video stream missing dimensions");
            }
        }
    }

    // Duration checks
    if let Some(d) = probe.duration_secs {
        if d <= 0.0 {
            if reject.is_none() {
                reject = Some("placeholder duration");
            }
        } else if d < cfg.min_duration_secs && reject.is_none() {
            reject = Some("too short");
        }
    }

    // Audio: warn unless require_audio
    if !probe.has_audio && reject.is_none() {
        if cfg.require_audio {
            reject = Some("missing audio stream");
        } else {
            warnings.push("missing audio stream".to_string());
        }
    }

    let valid = reject.is_none();
    let reason = if !valid {
        reject.map(str::to_string)
    } else if warnings.is_empty() {
        None
    } else {
        Some(warnings.join("; "))
    };

    ValidationOutcome {
        valid,
        reason,
        video_stream: probe.has_video,
        audio_stream: probe.has_audio,
        duration_secs: probe.duration_secs,
        width: probe.video_width,
        height: probe.video_height,
        ffprobe_version,
    }
}

// ── ffprobe discovery ──────────────────────────────────────────────────────

/// Locate the `ffprobe` binary: PATH first, then /opt/homebrew/bin, /usr/bin.
pub fn find_ffprobe() -> Option<PathBuf> {
    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            let cand = dir.join("ffprobe");
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    for p in ["/opt/homebrew/bin/ffprobe", "/usr/bin/ffprobe"] {
        let cand = PathBuf::from(p);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// First line of `ffprobe -version`, cached per process.
fn ffprobe_version() -> Option<String> {
    let bin = find_ffprobe()?;
    let out = Command::new(bin).arg("-version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let first = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .to_string();
    if first.is_empty() { None } else { Some(first) }
}

// ── Concrete validator ─────────────────────────────────────────────────────

/// Validates media by shelling out to ffprobe with a timeout.
///
/// If ffprobe is missing, fails to spawn, times out, or its output cannot be
/// parsed, validation degrades to size + extension heuristics with a
/// `Some(reason)` on the outcome (caller warns once).
#[derive(Debug, Default)]
pub struct FfprobeValidator;

impl FfprobeValidator {
    pub fn new() -> Self {
        Self
    }

    /// Run ffprobe on `path` with the configured timeout.
    fn run_probe(
        &self,
        path: &Path,
        cfg: &ValidationConfig,
    ) -> Result<ProbeOutcome, ValidatorError> {
        let bin = match find_ffprobe() {
            Some(b) => b,
            None => return Ok(ProbeOutcome::Degraded(DEGRADED_REASON.to_string())),
        };

        let mut child = match Command::new(bin)
            .args([
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
            ])
            .arg(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(ProbeOutcome::Degraded(format!(
                    "ffprobe exec failed ({e}); degraded heuristic validation"
                )));
            }
        };

        let deadline = Instant::now() + cfg.ffprobe_timeout;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        let _ = child.kill();
                        return Ok(ProbeOutcome::Degraded(format!(
                            "ffprobe exited with {status}; degraded heuristic validation"
                        )));
                    }
                    let mut out = String::new();
                    if let Some(mut stdout) = child.stdout.take() {
                        let _ = stdout.read_to_string(&mut out);
                    }
                    return match parse_ffprobe_json(&out) {
                        Ok(info) => Ok(ProbeOutcome::Info(info)),
                        Err(e) => Ok(ProbeOutcome::Degraded(format!(
                            "ffprobe output unreadable ({e}); degraded heuristic validation"
                        ))),
                    };
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        return Ok(ProbeOutcome::Degraded(
                            "ffprobe timed out; degraded heuristic validation".to_string(),
                        ));
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Err(e) => {
                    return Ok(ProbeOutcome::Degraded(format!(
                        "ffprobe wait failed ({e}); degraded heuristic validation"
                    )));
                }
            }
        }
    }
}

impl MediaValidator for FfprobeValidator {
    fn validate(
        &self,
        path: &Path,
        cfg: &ValidationConfig,
    ) -> Result<ValidationOutcome, ValidatorError> {
        // Extension check (kept from pre-existing heuristic set)
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| VALID_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false);
        if !ext_ok {
            return Ok(ValidationOutcome::invalid("unsupported extension"));
        }

        // Size check
        let meta = fs::metadata(path)?;
        if !meta.is_file() {
            return Ok(ValidationOutcome::invalid("not a regular file"));
        }
        let size = meta.len();
        if size < cfg.min_size_bytes {
            return Ok(ValidationOutcome::invalid("file too small"));
        }

        // ffprobe content checks
        static VERSION: OnceLock<Option<String>> = OnceLock::new();
        let version = VERSION.get_or_init(ffprobe_version).clone();
        let probe = self.run_probe(path, cfg)?;
        Ok(validate_checks(cfg, probe, version))
    }
}

// ── Cache key helper ───────────────────────────────────────────────────────

/// Cache key material for a file: `(size, mtime_ns)`.
pub fn cache_key(path: &Path) -> Option<(u64, u64)> {
    let meta = fs::metadata(path).ok()?;
    let size = meta.len();
    let mtime_ns = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos() as u64;
    Some((size, mtime_ns))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fake validator via trait ──────────────────────────────────────────

    struct FakeValidator {
        outcome: ValidationOutcome,
    }

    impl MediaValidator for FakeValidator {
        fn validate(
            &self,
            _path: &Path,
            _cfg: &ValidationConfig,
        ) -> Result<ValidationOutcome, ValidatorError> {
            Ok(self.outcome.clone())
        }
    }

    #[test]
    fn trait_plumbing_returns_canned_outcome() {
        let fake = FakeValidator {
            outcome: ValidationOutcome {
                valid: true,
                reason: None,
                video_stream: true,
                audio_stream: true,
                duration_secs: Some(24.0),
                width: Some(1920),
                height: Some(1080),
                ffprobe_version: Some("ffprobe version 7.1".into()),
            },
        };
        let cfg = ValidationConfig::default();
        let out = fake.validate(Path::new("/tmp/x.mkv"), &cfg).unwrap();
        assert!(out.valid);
        assert_eq!(out.width, Some(1920));
        assert_eq!(out.ffprobe_version.as_deref(), Some("ffprobe version 7.1"));
    }

    // ── ffprobe JSON fixtures ─────────────────────────────────────────────

    fn valid_probe_json() -> &'static str {
        r#"{
            "format": {"format_name": "matroska,webm", "duration": "24.500000"},
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio"}
            ]
        }"#
    }

    #[test]
    fn parse_ffprobe_json_valid() {
        let info = parse_ffprobe_json(valid_probe_json()).unwrap();
        assert_eq!(info.format_name.as_deref(), Some("matroska,webm"));
        assert!((info.duration_secs.unwrap() - 24.5).abs() < 1e-9);
        assert!(info.has_video);
        assert_eq!(info.video_width, Some(1920));
        assert_eq!(info.video_height, Some(1080));
        assert!(info.has_audio);
    }

    #[test]
    fn parse_ffprobe_json_placeholder_dims() {
        let json = r#"{
            "format": {"format_name": "mp4", "duration": "10.0"},
            "streams": [{"codec_type": "video", "width": 1, "height": 1}]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        assert!(info.has_video);
        assert_eq!(info.video_width, Some(1));
        let cfg = ValidationConfig::default();
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), None);
        assert!(!out.valid);
        assert_eq!(out.reason.as_deref(), Some("placeholder dimensions"));
    }

    #[test]
    fn parse_ffprobe_json_zero_duration_rejected() {
        let json = r#"{
            "format": {"format_name": "mp4", "duration": "0.000000"},
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        let cfg = ValidationConfig::default();
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), None);
        assert!(!out.valid);
        assert_eq!(out.reason.as_deref(), Some("placeholder duration"));
    }

    #[test]
    fn too_short_duration_rejected() {
        let json = r#"{
            "format": {"format_name": "mp4", "duration": "0.500000"},
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        let cfg = ValidationConfig::default(); // min 1.0s
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), None);
        assert!(!out.valid);
        assert_eq!(out.reason.as_deref(), Some("too short"));
    }

    #[test]
    fn no_video_stream_rejected() {
        let json = r#"{
            "format": {"format_name": "mp3"},
            "streams": [{"codec_type": "audio"}]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        assert!(!info.has_video);
        let cfg = ValidationConfig::default();
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), None);
        assert!(!out.valid);
        assert_eq!(out.reason.as_deref(), Some("no video stream"));
    }

    #[test]
    fn missing_audio_warns_but_valid() {
        let json = r#"{
            "format": {"format_name": "mp4", "duration": "10.0"},
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        assert!(!info.has_audio);
        let cfg = ValidationConfig::default();
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), None);
        assert!(out.valid);
        let reason = out.reason.as_deref().unwrap_or("");
        assert!(reason.contains("missing audio stream"));
    }

    #[test]
    fn require_audio_rejects_missing_audio() {
        let json = r#"{
            "format": {"format_name": "mp4", "duration": "10.0"},
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        let cfg = ValidationConfig {
            require_audio: true,
            ..Default::default()
        };
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), None);
        assert!(!out.valid);
        assert_eq!(out.reason.as_deref(), Some("missing audio stream"));
    }

    #[test]
    fn valid_file_ok() {
        let info = parse_ffprobe_json(valid_probe_json()).unwrap();
        let cfg = ValidationConfig::default();
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), Some("ffprobe 7.1".into()));
        assert!(out.valid);
        assert!(out.reason.is_none());
        assert_eq!(out.width, Some(1920));
        assert_eq!(out.height, Some(1080));
        assert!(out.audio_stream);
        assert_eq!(out.ffprobe_version.as_deref(), Some("ffprobe 7.1"));
    }

    #[test]
    fn empty_format_name_rejected() {
        let json = r#"{
            "format": {"format_name": ""},
            "streams": [{"codec_type": "video", "width": 640, "height": 360}]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        let cfg = ValidationConfig::default();
        let out = validate_checks(&cfg, ProbeOutcome::Info(info), None);
        assert!(!out.valid);
        assert_eq!(out.reason.as_deref(), Some("ffprobe reported no format"));
    }

    #[test]
    fn degrade_when_ffprobe_unavailable() {
        let cfg = ValidationConfig::default();
        let out = validate_checks(
            &cfg,
            ProbeOutcome::Degraded(DEGRADED_REASON.to_string()),
            None,
        );
        assert!(out.valid, "degraded validation stays permissive");
        assert_eq!(out.reason.as_deref(), Some(DEGRADED_REASON));
        assert!(out.ffprobe_version.is_none());
        assert!(!out.video_stream && !out.audio_stream);
    }

    #[test]
    fn unsupported_extension_rejected() {
        let _cfg = ValidationConfig::default();
        // validate_checks is content-only; extension rejection lives in
        // FfprobeValidator::validate — covered here via extension helper check.
        let ext = "txt";
        assert!(!VALID_EXTENSIONS.contains(&ext));
    }

    // ── cache_key helper ──────────────────────────────────────────────────

    #[test]
    fn cache_key_stable_for_same_file() {
        let dir = std::env::temp_dir().join("mt-cli-test-validator-cache-key");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("v.mp4");
        fs::write(&path, vec![0u8; 2048]).expect("write");
        let k1 = cache_key(&path).expect("cache key");
        let k2 = cache_key(&path).expect("cache key");
        assert_eq!(k1, k2);
        assert_eq!(k1.0, 2048);
        let _ = fs::remove_dir_all(&dir);
    }

    // ── find_ffprobe sanity ───────────────────────────────────────────────

    #[test]
    fn find_ffprobe_returns_path_or_none_without_panicking() {
        // Either a real ffprobe is on PATH (returns Some) or we are on a
        // machine without it (returns None) — both are valid.
        let _ = find_ffprobe();
    }
}
