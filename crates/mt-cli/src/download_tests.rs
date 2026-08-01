//! Tests for the anime downloader overhaul.
//!
//! Tests JSON schema validation, quality ranking, mirror race logic,
//! cancellation, retry/backoff, circuit breaker, manifest run modes, media
//! validation + quarantine, and RAII cleanup. No real network, no real
//! yt-dlp, no GPU.

use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tempfile::tempdir;
use tokio::process::Command;
use tokio::sync::broadcast;
use tokio::time::{sleep, timeout};

use crate::download_types::*;
use crate::downloader::test_factory::{FakeFactory, FakeOutcome};
use crate::downloader::{
    DownloadConfig, DownloadEngine, MIN_VALID_DOWNLOAD_BYTES, Outcome, RunMode, RunningSubprocess,
    SubprocessFactory, cleanup_stale_part, existing_download, find_output_file, find_part_file,
    find_stem_output, is_valid_output, is_valid_output_with_min, redact_urls,
};
use crate::manifest::{AttemptStatus, FinalStatus, Manifest, OutputMeta, sha256_file};
use crate::plain_output::{iso_timestamp, spawn_plain_output};
use crate::validator::{MediaValidator, ValidationConfig, ValidationOutcome, ValidatorError};

// ── Fake validators ────────────────────────────────────────────────────────

/// Canned validator: rejects files whose first 4 bytes are `BAD!`, accepts
/// everything else. Counts invocations (validation-cache tests).
struct ContentValidator {
    calls: Arc<AtomicUsize>,
}

impl MediaValidator for ContentValidator {
    fn validate(
        &self,
        path: &Path,
        _cfg: &ValidationConfig,
    ) -> Result<ValidationOutcome, ValidatorError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut buf = [0u8; 4];
        let head = File::open(path)
            .and_then(|mut f| f.read(&mut buf))
            .unwrap_or(0);
        let bad = head >= 4 && &buf[..4] == b"BAD!";
        Ok(ValidationOutcome {
            valid: !bad,
            reason: if bad {
                Some("bad content".into())
            } else {
                None
            },
            video_stream: true,
            audio_stream: true,
            duration_secs: Some(24.0),
            width: Some(1920),
            height: Some(1080),
            ffprobe_version: Some("ffprobe 7.1".into()),
        })
    }
}

fn validators() -> (Arc<ContentValidator>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let v = Arc::new(ContentValidator {
        calls: calls.clone(),
    });
    (v, calls)
}

/// Run the engine over `episodes`, collecting all events up to FinalSummary.
async fn run(
    config: DownloadConfig,
    factory: Arc<FakeFactory>,
    episodes: Vec<EpisodeInput>,
    validator: Arc<dyn MediaValidator + Send + Sync>,
) -> (Outcome, Vec<EpEvent>) {
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));
    let (tx, mut rx) = broadcast::channel(1024);
    let collector = tokio::spawn(async move {
        let mut evs = Vec::new();
        while let Ok(ev) = rx.recv().await {
            let terminal = matches!(ev, EpEvent::FinalSummary { .. });
            evs.push(ev);
            if terminal {
                break;
            }
        }
        evs
    });
    let outcome = engine.run_all_with_outcome(episodes, tx).await;
    let evs = timeout(Duration::from_secs(10), collector)
        .await
        .expect("event collector timed out")
        .expect("event collector panicked");
    (outcome, evs)
}

fn fast_retry(config: &mut DownloadConfig) {
    config.retry_attempts = 3;
    config.backoff_base_secs = 0.05;
    config.backoff_cap_secs = 2.0;
    config.jitter_secs = 0.0;
}

/// Count `.part` / `.meas` files under a directory (recursively).
fn count_temp_artifacts(dir: &Path) -> usize {
    let mut n = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let name = e.file_name().to_string_lossy().to_string();
                if name.contains(".part") || name.contains(".meas") {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Build a manifest on disk with the given (episode, status, output) entries.
fn write_manifest(dir: &Path, entries: &[(u32, FinalStatus, Option<PathBuf>)]) -> PathBuf {
    let mut m = Manifest::new();
    m.input.title = Some("Test".into());
    for (ep, status, output) in entries {
        m.ensure_episode(*ep);
        m.set_final_status(*ep, status.clone());
        if let Some(p) = output {
            m.set_output(
                *ep,
                OutputMeta {
                    path: p.clone(),
                    size: 1_048_576,
                    sha256: None,
                    validated: true,
                    ffprobe_version: None,
                    checked_at: None,
                },
            );
        }
    }
    let path = dir.join("manifest.json");
    m.save_atomic(&path).expect("save manifest");
    path
}

// ── JSON schema validation tests ───────────────────────────────────────────

#[test]
fn valid_json_input_parses_ok() {
    let json = r#"
    {
        "title": "Test Anime",
        "episodes": [
            {"episode": 1, "urls": ["https://cdn1.example.com/video.mp4"]},
            {"episode": 2, "urls": ["https://cdn2.example.com/video2.mp4", "https://cdn3.example.com/video2.mp4"]}
        ]
    }"#;
    let result = parse_json_input(json);
    assert!(
        result.is_ok(),
        "valid JSON should parse: {:?}",
        result.err()
    );
    let input = result.expect("valid JSON should parse");
    assert_eq!(input.title.as_deref(), Some("Test Anime"));
    assert_eq!(input.episodes.len(), 2);
    assert_eq!(input.episodes[0].episode, 1);
    assert_eq!(input.episodes[0].urls.len(), 1);
    assert_eq!(input.episodes[1].episode, 2);
    assert_eq!(input.episodes[1].urls.len(), 2);
}

#[test]
fn valid_json_no_title_ok() {
    let json = r#"{"episodes": [{"episode": 1, "urls": ["https://ex.com/v.mp4"]}]}"#;
    let result = parse_json_input(json);
    assert!(result.is_ok());
    let input = result.expect("valid JSON should parse");
    assert!(input.title.is_none());
}

#[test]
fn reject_empty_episodes() {
    let json = r#"{"episodes": []}"#;
    let result = parse_json_input(json);
    assert!(matches!(result, Err(JsonValidationError::NoEpisodes)));
}

#[test]
fn reject_zero_url_episode() {
    let json = r#"{"episodes": [{"episode": 1, "urls": []}]}"#;
    let result = parse_json_input(json);
    // Zero-urls errors must name the EPISODE NUMBER, not the array index.
    assert!(matches!(
        result,
        Err(JsonValidationError::EpisodeZeroUrls(1))
    ));
    let msg = format!("{:?}", result);
    assert!(msg.contains("1"), "payload is the episode number: {msg}");
}

#[test]
fn reject_empty_url_string() {
    let json = r#"{"episodes": [{"episode": 1, "urls": ["https://good.com", ""]}]}"#;
    let result = parse_json_input(json);
    assert!(matches!(result, Err(JsonValidationError::EmptyUrl(1))));
}

#[test]
fn reject_missing_episode_number() {
    let json = r#"{"episodes": [{"urls": ["https://ex.com/v.mp4"]}]}"#;
    let result = parse_json_input(json);
    assert!(
        matches!(result, Err(JsonValidationError::ParseError(_))),
        "expected ParseError for missing required field, got: {:?}",
        result
    );
}

#[test]
fn reject_zero_episode_number() {
    let json = r#"{"episodes": [{"episode": 0, "urls": ["https://ex.com/v.mp4"]}]}"#;
    let err = parse_json_input(json).expect_err("episode 0 must be rejected");
    assert!(
        matches!(&err, JsonValidationError::InvalidEpisodeNumber(0)),
        "expected InvalidEpisodeNumber(0), got: {err:?}"
    );
    let msg = format!("{err}");
    assert!(msg.contains("invalid episode number 0"), "got: {msg}");
    // Negative episode numbers get the same treatment.
    let json_neg = r#"{"episodes": [{"episode": -3, "urls": ["https://ex.com/v.mp4"]}]}"#;
    assert!(matches!(
        parse_json_input(json_neg),
        Err(JsonValidationError::InvalidEpisodeNumber(-3))
    ));
}

#[test]
fn reject_duplicate_episode() {
    let json = r#"{"episodes": [
        {"episode": 1, "urls": ["https://a.com/v.mp4"]},
        {"episode": 1, "urls": ["https://b.com/v.mp4"]}
    ]}"#;
    let result = parse_json_input(json);
    assert!(matches!(
        result,
        Err(JsonValidationError::DuplicateEpisode(1))
    ));
}

#[test]
fn reject_malformed_json() {
    let result = parse_json_input("not valid json");
    assert!(matches!(result, Err(JsonValidationError::ParseError(_))));
}

// ── Quality ranking tests ─────────────────────────────────────────────────

#[test]
fn quality_rank_higher_height_wins() {
    let q1080 = Quality::new(1080);
    let q720 = Quality::new(720);
    let q480 = Quality::new(480);
    assert!(q1080.rank() > q720.rank());
    assert!(q720.rank() > q480.rank());
    assert!(q1080 > q720);
}

#[test]
fn quality_unknown_treated_as_zero() {
    let unknown = Quality::new(0);
    let known = Quality::new(480);
    assert!(unknown.is_unknown());
    assert!(!known.is_unknown());
    assert!(known > unknown);
    assert_eq!(unknown.rank(), 0);
}

#[test]
fn quality_codec_bonus() {
    let q_h264 = Quality::with_codec(1080, Codec::H264);
    let q_h265 = Quality::with_codec(1080, Codec::H265);
    assert!(q_h265 > q_h264); // H265 > H264 for same height
}

#[test]
fn quality_height_from_str_works() {
    assert_eq!(quality_height_from_str("1080p"), 1080);
    assert_eq!(quality_height_from_str("720"), 720);
    assert_eq!(quality_height_from_str(""), 0);
    assert_eq!(quality_height_from_str("abc"), 0);
}

// ── Host preference ranking ───────────────────────────────────────────────

#[test]
fn host_preference_cda_is_top() {
    let cda_rank = host_preference_rank("cda.pl");
    let vk_rank = host_preference_rank("vk.com");
    let unknown_rank = host_preference_rank("unknown-host.example.com");
    assert!(cda_rank < vk_rank, "cda should rank higher than vk");
    assert!(vk_rank < unknown_rank, "vk should rank higher than unknown");
}

#[test]
fn host_preference_rumble_second() {
    let cda_rank = host_preference_rank("cda.pl");
    let rumble_rank = host_preference_rank("rumble.com/embed/123");
    let vk_rank = host_preference_rank("vk.com");
    assert!(cda_rank < rumble_rank, "cda ranks first");
    assert!(rumble_rank < vk_rank, "rumble ranks second (before vk)");
}

#[test]
fn host_preference_unknown_is_max() {
    let rank = host_preference_rank("some-rando-host.example.com");
    assert_eq!(rank, usize::MAX);
}

// ── parse_speed_bps tests ─────────────────────────────────────────────────

#[test]
fn parse_speed_bps_mib() {
    let bps = parse_speed_bps("7.50MiB/s");
    assert!(bps.is_some());
    let expected = 7.50 * 1_048_576.0;
    assert!((bps.unwrap() - expected).abs() < 1.0);
}

#[test]
fn parse_speed_bps_accepts_space_before_unit() {
    let bps = parse_speed_bps("7.50 MiB/s");
    assert!(bps.is_some());
    assert!((bps.unwrap() - 7.50 * 1_048_576.0).abs() < 1.0);
}

#[test]
fn parse_speed_bps_kib() {
    let bps = parse_speed_bps("500.0KiB/s");
    assert!(bps.is_some());
    assert!((bps.unwrap() - 500.0 * 1024.0).abs() < 1.0);
}

#[test]
fn parse_speed_bps_gib() {
    let bps = parse_speed_bps("1.20GiB/s");
    assert!(bps.is_some());
    assert!((bps.unwrap() - 1.20 * 1_073_741_824.0).abs() < 1.0);
}

#[test]
fn parse_speed_bps_raw_bytes() {
    let bps = parse_speed_bps("5000000");
    assert!(bps.is_some());
    assert!((bps.unwrap() - 5_000_000.0).abs() < 0.1);
}

// ── Phase state machine tests ─────────────────────────────────────────────

#[test]
fn phase_is_terminal() {
    assert!(
        Phase::Done {
            host: "a".into(),
            size_mb: 1.0
        }
        .is_terminal()
    );
    assert!(Phase::Failed.is_terminal());
    assert!(Phase::Cancelled.is_terminal());
    assert!(!Phase::Queued.is_terminal());
    assert!(!Phase::Measuring.is_terminal());
    assert!(
        !Phase::Downloading {
            pct: 0.0,
            speed: "".into(),
            eta: "".into(),
            downloaded: 0,
            total: 0
        }
        .is_terminal()
    );
}

// ── Quality display tests ─────────────────────────────────────────────────

#[test]
fn quality_display_known() {
    assert_eq!(format!("{}", Quality::new(1080)), "1080p");
    assert_eq!(format!("{}", Quality::new(720)), "720p");
}

#[test]
fn quality_display_unknown() {
    assert_eq!(format!("{}", Quality::new(0)), "unknown quality");
}

// ── Legacy format detection tests ────────────────────────────────────────

#[test]
fn reject_legacy_resolver_json() {
    let json = r#"{"resolved": true, "episodes": []}"#;
    let result = parse_json_input(json);
    assert!(matches!(result, Err(JsonValidationError::LegacyFormat)));
}

#[test]
fn legacy_rejected_actionable_msg() {
    let json = r#"{"embed_url": "https://x", "resolved": {"ep1": "u"}}"#;
    let err = parse_json_input(json).expect_err("legacy must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("reinstall userscript v4+"),
        "actionable message, got: {msg}"
    );
    assert!(
        msg.contains("Download all"),
        "actionable message, got: {msg}"
    );
}

// ── Canonical v2 schema tests ─────────────────────────────────────────────

#[test]
fn v2_schema_parses_ok() {
    let json = r#"{
        "schema_version": 2,
        "source_page": "https://ogladajanime.pl/anime/test",
        "resolved_at": "2025-07-30T12:00:00Z",
        "title": "Test Anime",
        "episodes": [
            {"episode": 1, "mirrors": [
                {"host": "cda", "quality": "1080p", "subtitle_group": "Fansub", "url": "https://cda.pl/video/123/"},
                {"host": "vk", "quality": "720p", "subtitle_group": null, "url": "https://vk.com/video_ext.php?oid=-1&id=2"}
            ]},
            {"episode": 2, "mirrors": [
                {"host": null, "quality": null, "subtitle_group": null, "url": "https://example.com/ep2.mp4"}
            ]}
        ]
    }"#;
    let (input, warnings) = parse_json_input_with_warnings(json).expect("v2 parses");
    assert_eq!(input.title.as_deref(), Some("Test Anime"));
    assert_eq!(
        input.source_page.as_deref(),
        Some("https://ogladajanime.pl/anime/test")
    );
    assert_eq!(input.resolved_at.as_deref(), Some("2025-07-30T12:00:00Z"));
    assert_eq!(input.episodes.len(), 2);
    let ep1 = &input.episodes[0];
    assert_eq!(ep1.mirrors.len(), 2);
    assert_eq!(ep1.mirrors[0].host.as_deref(), Some("cda"));
    assert_eq!(ep1.mirrors[0].quality.as_deref(), Some("1080p"));
    assert_eq!(ep1.mirrors[0].subtitle_group.as_deref(), Some("Fansub"));
    assert_eq!(ep1.mirrors[1].url, "https://vkvideo.ru/video-1_2");
    assert_eq!(ep1.urls.len(), 2);
    assert_eq!(ep1.urls[0], ep1.mirrors[0].url);
    assert_eq!(ep1.quality.as_ref().map(|q| q.height), Some(1080));
    assert!(!warnings.iter().any(|w| w.contains("v1 schema")));
}

#[test]
fn v2_duplicate_urls_deduped_after_canonicalization() {
    let json = r#"{
        "schema_version": 2,
        "episodes": [{"episode": 1, "mirrors": [
            {"host": "vk", "url": "https://vk.com/video_ext.php?oid=1&id=2"},
            {"host": "vk", "url": "https://vk.com/video_ext.php?oid=1&id=2&hash=x"},
            {"host": null, "url": "https://example.com/a.mp4"}
        ]}]
    }"#;
    let (input, _w) = parse_json_input_with_warnings(json).expect("parses");
    assert_eq!(input.episodes[0].mirrors.len(), 2);
    assert_eq!(input.episodes[0].urls.len(), 2);
}

#[test]
fn v1_schema_migrates_to_v2() {
    let json = r#"{
        "title": "Old Anime",
        "episodes": [
            {"episode": 1, "urls": ["https://video.sibnet.ru/v1.mp4", "https://example.com/v2.mp4"]}
        ]
    }"#;
    let (input, warnings) = parse_json_input_with_warnings(json).expect("v1 parses");
    assert_eq!(input.title.as_deref(), Some("Old Anime"));
    assert_eq!(input.episodes.len(), 1);
    let ep1 = &input.episodes[0];
    assert_eq!(ep1.mirrors.len(), 2, "mirrors built from urls");
    assert_eq!(ep1.mirrors[0].host.as_deref(), Some("video.sibnet.ru"));
    assert_eq!(ep1.mirrors[1].host.as_deref(), Some("example.com"));
    assert!(ep1.mirrors[0].quality.is_none());
    assert!(ep1.mirrors[0].subtitle_group.is_none());
    assert_eq!(ep1.urls.len(), 2, "flat urls preserved");
    assert!(
        warnings.iter().any(|w| w.contains("v1 schema")),
        "migration warning expected, got: {warnings:?}"
    );
    let json1 = r#"{"schema_version": 1, "episodes": [{"episode": 1, "urls": ["https://example.com/v.mp4"]}]}"#;
    assert!(parse_json_input(json1).is_ok());
}

#[test]
fn unsupported_schema_version_rejected() {
    let json = r#"{"schema_version": 3, "episodes": [{"episode": 1, "mirrors": []}]}"#;
    let err = parse_json_input(json).expect_err("schema_version 3 unsupported");
    assert!(matches!(
        err,
        JsonValidationError::UnsupportedSchemaVersion(3)
    ));
}

#[test]
fn http_only_rejected() {
    let json = r#"{"episodes": [{"episode": 1, "urls": ["ftp://example.com/v.mp4"]}]}"#;
    let err = parse_json_input(json).expect_err("ftp must be rejected");
    assert!(
        matches!(err, JsonValidationError::UnsupportedUrlScheme(_)),
        "got: {err:?}"
    );
    let json2 = r#"{"episodes": [{"episode": 1, "urls": ["file:///tmp/v.mp4"]}]}"#;
    assert!(matches!(
        parse_json_input(json2),
        Err(JsonValidationError::UnsupportedUrlScheme(_))
    ));
}

#[test]
fn episode_out_of_range_rejected() {
    let json = r#"{"episodes": [{"episode": 100000, "urls": ["https://example.com/v.mp4"]}]}"#;
    let err = parse_json_input(json).expect_err("100000 out of range");
    assert!(matches!(
        err,
        JsonValidationError::EpisodeOutOfRange(100000)
    ));
}

#[test]
fn stale_resolved_at_warning() {
    let json = r#"{
        "schema_version": 2,
        "resolved_at": "2000-01-01T00:00:00Z",
        "episodes": [{"episode": 1, "mirrors": [{"url": "https://example.com/v.mp4"}]}]
    }"#;
    let (input, warnings) = parse_json_input_with_warnings(json).expect("parses");
    assert!(input.resolved_at.is_some());
    assert!(
        warnings.iter().any(|w| w.contains("stale")),
        "stale warning expected, got: {warnings:?}"
    );
}

#[test]
fn fresh_resolved_at_no_warning() {
    let json = r#"{
        "schema_version": 2,
        "resolved_at": "2099-01-01T00:00:00Z",
        "episodes": [{"episode": 1, "mirrors": [{"url": "https://example.com/v.mp4"}]}]
    }"#;
    let (_input, warnings) = parse_json_input_with_warnings(json).expect("parses");
    assert!(
        !warnings.iter().any(|w| w.contains("stale")),
        "no stale warning for fresh timestamp, got: {warnings:?}"
    );
}

// ── Strict ISO8601 validation ─────────────────────────────────────────────

#[test]
fn iso8601_accepts_valid_full_timestamp() {
    // Roundtrip unchanged: 2025-07-30T12:00:00Z = 1753876800 unix seconds.
    assert_eq!(
        parse_iso8601_epoch("2025-07-30T12:00:00Z"),
        Some(1_753_876_800)
    );
}

#[test]
fn iso8601_rejects_month_13() {
    assert_eq!(parse_iso8601_epoch("2025-13-01T12:00:00Z"), None);
}

#[test]
fn iso8601_rejects_day_45() {
    assert_eq!(parse_iso8601_epoch("2025-01-45T12:00:00Z"), None);
    // April has 30 days; 31 must be rejected per-month.
    assert_eq!(parse_iso8601_epoch("2025-04-31T12:00:00Z"), None);
}

#[test]
fn iso8601_rejects_hour_25() {
    assert_eq!(parse_iso8601_epoch("2025-01-01T25:00:00Z"), None);
    // Negative hours were accepted by the lenient parser; now rejected.
    assert_eq!(parse_iso8601_epoch("2025-01-01T-5:00:00Z"), None);
}

#[test]
fn iso8601_rejects_minute_60() {
    assert_eq!(parse_iso8601_epoch("2025-01-01T12:60:00Z"), None);
}

#[test]
fn iso8601_rejects_non_leap_feb_29() {
    assert_eq!(parse_iso8601_epoch("2023-02-29T12:00:00Z"), None);
}

#[test]
fn iso8601_accepts_leap_feb_29() {
    // 2024 is a leap year; 2024-02-29 must be accepted.
    assert!(parse_iso8601_epoch("2024-02-29T12:00:00Z").is_some());
}

#[test]
fn iso8601_accepts_leap_second_60() {
    // RFC 3339 leap second: second 60 tolerated (pre-existing parser accepted
    // it; treated as +60s, only feeds the staleness heuristic).
    assert!(parse_iso8601_epoch("2025-07-30T12:00:60Z").is_some());
    // Second 61 is not a leap second and must be rejected.
    assert_eq!(parse_iso8601_epoch("2025-07-30T12:00:61Z"), None);
}

#[test]
fn iso8601_accepts_all_current_formats() {
    // Every format the lenient parser accepted must still parse.
    let base = 1_753_876_800; // 2025-07-30T12:00:00Z
    assert_eq!(
        parse_iso8601_epoch("2025-07-30T12:00:00+02:00"),
        Some(base - 7200)
    );
    assert_eq!(
        parse_iso8601_epoch("2025-07-30T12:00:00-05:00"),
        Some(base + 18_000)
    );
    assert_eq!(parse_iso8601_epoch("2025-07-30T12:00:00.123Z"), Some(base));
    assert_eq!(parse_iso8601_epoch("2025-07-30T12:00Z"), Some(base));
    assert_eq!(parse_iso8601_epoch("2025-07-30T12:00:00"), Some(base));
    assert_eq!(parse_iso8601_epoch("2025-7-3T5:6:7Z"), Some(1_751_519_167));
    assert!(parse_iso8601_epoch("2025-12-31T23:59:59Z").is_some());
    assert!(parse_iso8601_epoch("2025-02-28T00:00:00Z").is_some());
}

#[test]
fn iso8601_invalid_resolved_at_warns_parse_failed() {
    // Rejection stays a warning (never a hard error): the input parses, the
    // timestamp is treated as invalid (no staleness check), warning emitted.
    let json = r#"{
        "schema_version": 2,
        "resolved_at": "2025-13-01T12:00:00Z",
        "episodes": [{"episode": 1, "mirrors": [{"url": "https://example.com/v.mp4"}]}]
    }"#;
    let (input, warnings) = parse_json_input_with_warnings(json).expect("parses");
    assert_eq!(input.resolved_at.as_deref(), Some("2025-13-01T12:00:00Z"));
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("could not be parsed as ISO8601")),
        "parse-failed warning expected, got: {warnings:?}"
    );
    assert!(
        !warnings.iter().any(|w| w.contains("stale")),
        "invalid timestamp must not be treated as stale, got: {warnings:?}"
    );
}

#[test]
fn sanitize_slug_cleans_title() {
    assert_eq!(
        sanitize_slug("My Anime: Attack on Titan!"),
        "my-anime-attack-on-titan"
    );
    assert_eq!(sanitize_slug("Na Zawsze. E01 (BD)"), "na-zawsze-e01-bd");
    assert_eq!(sanitize_slug(""), "untitled");
    let long = "a".repeat(200);
    assert_eq!(sanitize_slug(&long).len(), 80);
    let tricky = sanitize_slug("../../etc/passwd");
    assert!(!tricky.contains("..") && !tricky.contains('/'));
}

#[test]
fn vk_url_canonicalized_in_schema() {
    let url = "https://vk.com/video_ext.php?oid=-229809086&id=456239061&hash=abc&hd=2";
    assert_eq!(
        try_canonicalize_vk_url(url).as_deref(),
        Some("https://vkvideo.ru/video-229809086_456239061")
    );
}

// ── Redaction unit tests ──────────────────────────────────────────────────

#[test]
fn redact_urls_strips_query_tokens() {
    assert_eq!(
        redact_urls("ERROR: https://cdn.example.com/v?token=SECRET123 denied"),
        "ERROR: [URL] denied"
    );
    assert_eq!(
        redact_urls("login https://a.b/c?hash=xyz&x=1 here"),
        "login [URL] here"
    );
    assert_eq!(
        redact_urls("token=SECRET inline"),
        "token=[REDACTED] inline"
    );
    assert_eq!(redact_urls("no secrets here"), "no secrets here");
}

#[test]
fn redact_urls_token_value_and_quote_delimiters() {
    // Bare token=value assignment → [REDACTED] (query-token coverage).
    assert_eq!(redact_urls("token=abc123&x=1"), "token=[REDACTED]&x=1");
    // URL-only case unchanged ([URL] marker, no token touch).
    assert_eq!(
        redact_urls("src https://cdn.example.com/v?plain=1 end"),
        "src [URL] end"
    );
    // Quoted / backtick delimited URLs end the URL run at the delimiter.
    assert_eq!(
        redact_urls("url='https://a.b/c?v=1' done"),
        "url='[URL]' done"
    );
    assert_eq!(
        redact_urls("url=`https://a.b/c?v=1` done"),
        "url=`[URL]` done"
    );
    assert_eq!(
        redact_urls("url=\"https://a.b/c?v=1\" done"),
        "url=\"[URL]\" done"
    );
    // token=value inside a quoted region still redacts.
    assert_eq!(redact_urls("q='token=hunter2' x"), "q='token=[REDACTED]' x");
}

// ── Integration tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn engine_run_all_with_fake_factory() {
    let dir = tempdir().expect("tempdir");
    let mut config = DownloadConfig {
        episode_concurrency: 2,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-anime".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    let (validator, _calls) = validators();

    let episodes = vec![
        EpisodeInput::new(1, vec!["https://example.com/ep1.mp4".into()]),
        EpisodeInput::new(2, vec!["https://example.com/ep2.mp4".into()]),
    ];

    let (outcome, _events) = run(config, factory, episodes, validator).await;

    assert_eq!(outcome.downloaded + outcome.skipped, 2, "both episodes ok");
}

#[tokio::test]
async fn engine_cancellation_stops_downloads() {
    let dir = tempdir().expect("tempdir");
    let config = DownloadConfig {
        episode_concurrency: 2,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-cancel".into(),
        ..Default::default()
    };
    let factory = Arc::new(FakeFactory::new());
    let (validator, _calls) = validators();
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));
    let cancel = engine.cancel_token();

    let episodes = vec![EpisodeInput::new(
        1,
        vec!["https://example.com/ep1.mp4".into()],
    )];

    let (tx, mut rx) = broadcast::channel::<EpEvent>(256);

    cancel.cancel();

    let outcome = engine.run_all_with_outcome(episodes, tx).await;

    assert!(outcome.cancelled, "cancelled run must report cancelled");
    assert_eq!(outcome.missing_episodes, vec![1]);

    while let Ok(ev) = rx.try_recv() {
        if let EpEvent::Cancelled { ep } = ev {
            assert_eq!(ep, 1);
        }
    }
}

#[tokio::test]
async fn quality_filter_selects_highest_mirror() {
    let dir = tempdir().expect("tempdir");
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-quality".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(
        1,
        vec![
            "https://720p.example.com/vid.mp4".into(),
            "https://1080p.example.com/vid.mp4".into(),
            "https://480p.example.com/vid.mp4".into(),
        ],
    )];

    let (outcome, _events) = run(config, factory, episodes, validator).await;
    assert_eq!(outcome.downloaded + outcome.skipped, 1);
}

#[tokio::test]
async fn events_flow_through_broadcast() {
    let dir = tempdir().expect("tempdir");
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-events".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(
        42,
        vec!["https://example.com/ep42.mp4".into()],
    )];

    let (_outcome, events) = run(config, factory, episodes, validator).await;

    assert!(!events.is_empty(), "should have collected events");
    let has_measuring = events
        .iter()
        .any(|e| matches!(e, EpEvent::Measuring { .. }));
    assert!(has_measuring, "should have received Measuring event");
}

/// Factory that spawns shell pipelines simulating yt-dlp with configurable
/// speed and output behavior. Used to test concurrent mirror race with
/// real process lifecycle.
///
/// Emulates the real yt-dlp contract: the partial file (`{out}.mkv.part`)
/// grows during the measurement window, then — after `completion_delay_secs`
/// — the file appears at its final size (`{out}.mkv`) exactly when the child
/// exits 0. No live-file rename ever happens inside the script.
struct SimYtDlpFactory {
    speed_mibs: f64,
    progress_lines: usize,
    work_secs: f64,
    /// Extra sleep after the partial file is written, before finalize —
    /// simulates a download still in flight past the measurement window.
    completion_delay_secs: f64,
    fail_measurement: bool,
}

impl Default for SimYtDlpFactory {
    fn default() -> Self {
        Self {
            speed_mibs: 5.0,
            progress_lines: 3,
            work_secs: 0.3,
            completion_delay_secs: 0.05,
            fail_measurement: false,
        }
    }
}

impl SubprocessFactory for SimYtDlpFactory {
    fn spawn_measure(
        &self,
        _url: &str,
        out_path: &Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>>
    {
        if self.fail_measurement {
            return Box::pin(async move { None });
        }
        let out_part = out_path.with_extension("mkv.part");
        let out_final = out_path.with_extension("mkv");
        let out_part_str = out_part.to_string_lossy().to_string();
        let out_final_str = out_final.to_string_lossy().to_string();
        let speed = self.speed_mibs;
        let _lines = self.progress_lines;
        let delay = self.work_secs;
        let done = self.completion_delay_secs;

        Box::pin(async move {
            // Shell: progress lines, grow the partial file, keep running past
            // the measurement window, then finalize (file appears at final
            // size on exit 0).
            let script = format!(
                "echo '[download]  0.0% at {speed:.1}MiB/s ETA 00:00' && \
                 sleep {delay} && \
                 echo '[download] 50.0% at {speed:.1}MiB/s ETA 00:00' && \
                 sleep {delay} && \
                 dd if=/dev/zero bs=1048576 count=1 of='{out_part_str}' 2>/dev/null && \
                 sleep {done} && \
                 echo '[download] 100% at {speed:.1}MiB/s ETA 00:00' && \
                 mv '{out_part_str}' '{out_final_str}'"
            );

            let mut cmd = Command::new("sh");
            cmd.arg("-c")
                .arg(&script)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());

            let child = cmd.spawn().ok()?;
            let pgid = child.id().unwrap_or(0);
            Some(RunningSubprocess { child, pgid })
        })
    }

    fn spawn_download(
        &self,
        url: &str,
        out_path: &Path,
        _continue_part: bool,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<RunningSubprocess>> + Send>>
    {
        // For download, same behavior as measure (but in practice would be longer)
        self.spawn_measure(url, out_path)
    }

    fn inspect_formats(
        &self,
        _url: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Quality>> + Send>> {
        Box::pin(async move { Some(Quality::new(1080)) })
    }
}

fn sim_factory() -> Arc<SimYtDlpFactory> {
    Arc::new(SimYtDlpFactory::default())
}

#[tokio::test]
async fn simulated_ytdlp_concurrent_mirror_race() {
    let dir = tempdir().expect("tempdir");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 2,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-sim".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();

    let factory = Arc::new(SimYtDlpFactory {
        speed_mibs: 50.0,
        progress_lines: 3,
        work_secs: 0.1,
        completion_delay_secs: 0.05,
        fail_measurement: false,
    });

    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));

    let episodes = vec![EpisodeInput::new(
        1,
        vec!["https://fast.example.com/vid.mp4".into()],
    )];

    let (tx, mut rx) = broadcast::channel::<EpEvent>(256);

    let engine_clone = engine.clone();
    let result = timeout(Duration::from_secs(30), async move {
        engine_clone.run_all(episodes, tx).await
    })
    .await;

    let results = match result {
        Ok(r) => r,
        Err(_) => panic!("timeout waiting for download"),
    };

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);

    let mut has_measuring = false;
    while let Ok(ev) = rx.try_recv() {
        if matches!(ev, EpEvent::Measuring { .. }) {
            has_measuring = true;
        }
    }
    assert!(has_measuring, "should have received Measuring event");
}

#[tokio::test]
async fn simulated_multi_mirror_race_with_winner_selection() {
    let dir = tempdir().expect("tempdir");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 4,
        measurement_secs: 5,
        out_dir: dir.path().to_path_buf(),
        slug: "test-race".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();

    let fast = Arc::new(SimYtDlpFactory {
        speed_mibs: 100.0,
        progress_lines: 2,
        work_secs: 0.1,
        completion_delay_secs: 0.05,
        fail_measurement: false,
    });

    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, fast, validator,
    ));

    let episodes = vec![EpisodeInput::new(
        1,
        vec![
            "https://cdn1.example.com/vid.mp4".into(),
            "https://cdn2.example.com/vid.mp4".into(),
        ],
    )];

    let (tx, _rx) = broadcast::channel::<EpEvent>(256);

    let engine_clone = engine.clone();
    let result = timeout(Duration::from_secs(30), async move {
        engine_clone.run_all(episodes, tx).await
    })
    .await;

    let results = match result {
        Ok(r) => r,
        Err(_) => panic!("timeout waiting for multi-mirror download"),
    };

    assert_eq!(results.len(), 1, "one episode result expected");
    assert_eq!(results[0].0, 1);
}

#[tokio::test]
async fn unknown_quality_triggers_format_inspection() {
    let dir = tempdir().expect("tempdir");
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-inspect".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.fake_quality.store(true, Ordering::SeqCst);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(
        1,
        vec!["https://example.com/ep1.mp4".into()],
    )];

    let (outcome, _events) = run(config, factory, episodes, validator).await;
    assert_eq!(outcome.downloaded + outcome.skipped, 1, "episode 1 ok");
}

// ── Plain output tests ────────────────────────────────────────────────────

#[tokio::test]
async fn plain_output_counts_done_and_failed() {
    let (tx, rx) = broadcast::channel::<EpEvent>(256);

    let handle = spawn_plain_output(rx, 5);

    tx.send(EpEvent::Done {
        ep: 1,
        host: "cda".into(),
        size_mb: 100.0,
    })
    .ok();
    tx.send(EpEvent::Done {
        ep: 2,
        host: "vk".into(),
        size_mb: 200.0,
    })
    .ok();
    tx.send(EpEvent::Failed { ep: 3 }).ok();
    drop(tx);

    let (done, failed) = timeout(Duration::from_secs(5), handle)
        .await
        .expect("plain output should finish")
        .expect("join handle");

    assert_eq!(done, 2, "two episodes done");
    assert_eq!(failed, 1, "one episode failed");
}

#[test]
fn plain_timestamp_is_deterministic() {
    let ts1 = iso_timestamp();
    assert!(!ts1.is_empty());
    assert!(
        ts1.chars().all(|c| c.is_ascii_digit()),
        "timestamp must be all digits (Unix ms)"
    );
}

#[tokio::test]
async fn continuation_path_renames_part_to_final() {
    let dir = tempdir().expect("tempdir");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 4,
        measurement_secs: 2,
        out_dir: dir.path().to_path_buf(),
        slug: "test-cont".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();

    let factory = Arc::new(SimYtDlpFactory {
        speed_mibs: 100.0,
        progress_lines: 2,
        work_secs: 0.05,
        completion_delay_secs: 0.05,
        fail_measurement: false,
    });
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));

    let episodes = vec![EpisodeInput::new(
        1,
        vec!["https://cdn1.example.com/vid.mp4".into()],
    )];

    let (tx, _rx) = broadcast::channel::<EpEvent>(256);
    let results = engine.run_all(episodes, tx).await;

    let mut found_meas = false;
    if let Ok(entries) = fs::read_dir(dir.path()) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.contains(".meas") {
                found_meas = true;
            }
        }
    }
    assert!(!found_meas, "no .meas files should remain");
    assert_eq!(results.len(), 1);
}

// ── Dynamic extension helper tests ───────────────────────────────────────

#[test]
fn find_part_file_discovers_all_extensions() {
    let dir = tempdir().expect("tempdir");

    let stem = dir.path().join("ep1-E01");

    for ext in &["mkv", "mp4", "webm"] {
        let p = dir.path().join(format!("ep1-E01.{ext}.part"));
        fs::write(&p, b"partial data").expect("write part file");
    }
    fs::write(dir.path().join("other.part"), b"x").expect("write other");
    fs::write(dir.path().join("ep1-E01.empty.part"), b"").expect("write empty");

    let found = find_part_file(&stem);
    assert!(found.is_some(), "should find a .part file");
    let found = found.unwrap();
    let fname = found.file_name().unwrap().to_string_lossy().to_string();
    assert!(
        fname.starts_with("ep1-E01."),
        "matches stem prefix: {fname}"
    );
    assert!(fname.ends_with(".part"), "ends with .part: {fname}");
    assert!(fname != "ep1-E01.empty.part", "ignored empty file");
}

#[test]
fn find_stem_output_discovers_completed_files_excludes_part() {
    let dir = tempdir().expect("tempdir");

    let stem = dir.path().join("ep1-E01");

    fs::write(dir.path().join("ep1-E01.mkv"), b"complete mkv").expect("write mkv");
    fs::write(dir.path().join("ep1-E01.mp4"), b"complete mp4").expect("write mp4");
    fs::write(dir.path().join("ep1-E01.mkv.part"), b"partial").expect("write part");
    fs::write(dir.path().join("ep1-E01.txt"), b"text").expect("write txt");

    let found = find_stem_output(&stem);
    assert!(found.is_some(), "should find completed output");
    let found = found.unwrap();
    let fname = found.file_name().unwrap().to_string_lossy().to_string();
    assert!(!fname.ends_with(".part"), "excludes .part files: {fname}");
    assert!(
        fname == "ep1-E01.mkv" || fname == "ep1-E01.mp4",
        "found video file: {fname}"
    );
}

#[tokio::test]
async fn cleanup_stale_part_removes_prefix_matching_files() {
    let dir = tempdir().expect("tempdir");

    let stem = dir.path().join("ep1-E01");
    let meas_stem = stem.with_extension("meas0");

    fs::write(dir.path().join("ep1-E01.meas0.mp4.part"), b"partial").expect("write");
    fs::write(dir.path().join("ep1-E01.meas0.mkv"), b"data").expect("write");
    fs::write(dir.path().join("ep1-E01.meas1.mp4.part"), b"partial").expect("write");
    fs::write(dir.path().join("ep1-E01.mp4.part"), b"winner").expect("write");

    cleanup_stale_part(&meas_stem).await;

    assert!(
        !dir.path().join("ep1-E01.meas0.mp4.part").exists(),
        "meas0 .part removed"
    );
    assert!(
        !dir.path().join("ep1-E01.meas0.mkv").exists(),
        "meas0 .mkv removed"
    );
    assert!(
        dir.path().join("ep1-E01.meas1.mp4.part").exists(),
        "meas1 .part survives"
    );
    assert!(
        dir.path().join("ep1-E01.mp4.part").exists(),
        "winner .part survives"
    );
}

#[test]
fn existing_download_finds_completed_file() {
    let dir = tempdir().expect("tempdir");

    let stem = dir.path().join("ep1-E01");

    assert!(
        existing_download(&stem, MIN_VALID_DOWNLOAD_BYTES).is_none(),
        "no file yet"
    );

    fs::write(dir.path().join("ep1-E01.mkv"), vec![0u8; 1_048_576]).expect("write mkv");

    let found = existing_download(&stem, MIN_VALID_DOWNLOAD_BYTES);
    assert!(found.is_some(), "should find completed download");
    let found = found.unwrap();
    assert_eq!(found.file_name().unwrap().to_string_lossy(), "ep1-E01.mkv");

    fs::write(dir.path().join("ep1-E01.mkv.part"), vec![0u8; 1_048_576]).expect("write part");
    let still_mkv = existing_download(&stem, MIN_VALID_DOWNLOAD_BYTES);
    assert!(
        still_mkv.is_some(),
        "existing_download still finds completed file"
    );
    assert_eq!(
        still_mkv.unwrap().extension().unwrap().to_string_lossy(),
        "mkv",
        "prefers completed file, not .part"
    );
}

#[test]
fn find_output_file_works_for_any_video_ext() {
    let dir = tempdir().expect("tempdir");

    let stem = dir.path().join("ep1-E01");

    for ext in &["mkv", "mp4", "webm", "flv", "mov", "avi"] {
        let _ = fs::remove_dir_all(dir.path());
        fs::create_dir_all(dir.path()).expect("re-create dir");
        let p = dir.path().join(format!("ep1-E01.{ext}"));
        fs::write(&p, vec![0u8; 1_048_576]).expect("write");

        let found = find_output_file(&stem, MIN_VALID_DOWNLOAD_BYTES);
        assert!(
            found.is_some(),
            "find_output_file should find .{ext}: {:?}",
            found
        );
        if let Some(f) = found {
            assert_eq!(
                f.extension().unwrap().to_string_lossy(),
                *ext,
                "correct extension"
            );
        }
    }
}

#[test]
fn find_stem_output_excludes_non_video_extensions() {
    let dir = tempdir().expect("tempdir");

    let stem = dir.path().join("ep1-E01");

    fs::write(dir.path().join("ep1-E01.txt"), b"text").expect("write txt");
    fs::write(dir.path().join("ep1-E01.json"), b"{}").expect("write json");

    assert!(
        find_stem_output(&stem).is_none(),
        "no video file should be found"
    );
}

// ── Regression tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn semaphore_contention_does_not_exhaust_mirror() {
    let dir = tempdir().expect("tempdir");

    let mut config = DownloadConfig {
        episode_concurrency: 4,
        host_concurrency: 1,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-sem".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();

    let fast = sim_factory();
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, fast, validator,
    ));

    let episodes = vec![
        EpisodeInput::new(1, vec!["https://video.sibnet.ru/v1.mp4".into()]),
        EpisodeInput::new(2, vec!["https://video.sibnet.ru/v2.mp4".into()]),
    ];

    let (tx, rx) = broadcast::channel::<EpEvent>(256);
    let _collector = tokio::spawn(async move {
        let _rx = rx;
    });

    let result = timeout(Duration::from_secs(30), async {
        engine.run_all(episodes, tx).await
    })
    .await;

    let results = match result {
        Ok(r) => r,
        Err(_) => panic!("timeout waiting for semaphore contention test"),
    };

    assert_eq!(results.len(), 2, "both episodes should complete");
    assert!(
        results[0].1.is_some() || results[1].1.is_some(),
        "at least one episode should succeed"
    );
}

#[tokio::test]
async fn missing_measurement_output_does_not_false_mirrorbusy() {
    let dir = tempdir().expect("tempdir");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 4,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-meas".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();

    let factory = Arc::new(SimYtDlpFactory {
        speed_mibs: 0.0,
        progress_lines: 0,
        work_secs: 0.01,
        completion_delay_secs: 0.05,
        fail_measurement: true,
    });

    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));

    let episodes = vec![EpisodeInput::new(
        1,
        vec![
            "https://cdn1.example.com/vid.mp4".into(),
            "https://cdn2.example.com/vid.mp4".into(),
        ],
    )];

    let (tx, mut rx) = broadcast::channel::<EpEvent>(256);

    let results_fut = engine.run_all(episodes, tx);

    let (_results, events) = tokio::join!(results_fut, async {
        let mut events = Vec::new();
        while let Ok(ev) = rx.recv().await {
            events.push(ev.clone());
            if matches!(ev, EpEvent::Failed { .. } | EpEvent::Done { .. }) {
                break;
            }
        }
        events
    });

    assert!(!events.is_empty(), "should have events");

    let has_mirror_busy = events
        .iter()
        .any(|e| matches!(e, EpEvent::MirrorBusy { .. }));
    assert!(
        !has_mirror_busy,
        "MirrorBusy should not fire for measurement failure"
    );

    let has_meas_failed = events
        .iter()
        .any(|e| matches!(e, EpEvent::MirrorMeasFailed { .. }));
    let has_failed = events.iter().any(|e| matches!(e, EpEvent::Failed { .. }));
    assert!(
        has_meas_failed || has_failed,
        "should have MirrorMeasFailed or Failed event"
    );
}

#[tokio::test]
async fn deterministic_episode_order() {
    let dir = tempdir().expect("tempdir");

    let mut config = DownloadConfig {
        episode_concurrency: 4,
        host_concurrency: 4,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-ord".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();

    let factory = sim_factory();
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));

    let episodes = vec![
        EpisodeInput::new(3, vec!["https://hqq.tv/ep3.mp4".into()]),
        EpisodeInput::new(1, vec!["https://hqq.tv/ep1.mp4".into()]),
        EpisodeInput::new(2, vec!["https://hqq.tv/ep2.mp4".into()]),
    ];

    let (tx, rx) = broadcast::channel::<EpEvent>(256);
    let _collector = tokio::spawn(async move {
        let _rx = rx;
    });

    let results = timeout(Duration::from_secs(30), async {
        engine.run_all(episodes, tx).await
    })
    .await;

    let results = match results {
        Ok(r) => r,
        Err(_) => panic!("timeout"),
    };

    for i in 0..results.len().saturating_sub(1) {
        assert!(
            results[i].0 < results[i + 1].0,
            "results must be in ascending episode order: {:?}",
            results
        );
    }
}

// ── Output validation regression tests ────────────────────────────────────

#[test]
fn is_valid_output_rejects_unknown_extension() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test.unknown_video");
    fs::write(&path, vec![0u8; 2_000_000]).expect("write");
    assert!(!is_valid_output(&path), "unknown_video must be rejected");
}

#[test]
fn is_valid_output_rejects_tiny_file() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test.mp4");
    fs::write(&path, vec![0u8; 3078]).expect("write");
    assert!(
        !is_valid_output(&path),
        "tiny mp4 must be rejected (below 1MB)"
    );
}

#[test]
fn is_valid_output_accepts_valid_file() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test.mp4");
    fs::write(&path, vec![0u8; 1_048_576]).expect("write");
    assert!(is_valid_output(&path), "1MB mp4 must be accepted");
}

#[test]
fn is_valid_output_with_min_uses_raised_floor() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test.mp4");
    let floor = 5 * 1_048_576;

    fs::write(&path, vec![0u8; 2 * 1_048_576]).expect("write 2MB");
    assert!(
        !is_valid_output_with_min(&path, floor),
        "2MB must be rejected below 5MB floor"
    );
    assert!(
        is_valid_output(&path),
        "2MB still passes the legacy 1MB floor"
    );

    fs::write(&path, vec![0u8; 6 * 1_048_576]).expect("write 6MB");
    assert!(
        is_valid_output_with_min(&path, floor),
        "6MB must be accepted"
    );
}

#[test]
fn find_output_file_respects_effective_floor() {
    let dir = tempdir().expect("tempdir");
    let stem = dir.path().join("ep1-E01");
    let floor = 5 * 1_048_576;

    fs::write(dir.path().join("ep1-E01.mkv"), vec![0u8; 2 * 1_048_576]).expect("write 2MB");
    assert!(
        find_output_file(&stem, floor).is_none(),
        "2MB must not match a 5MB acceptance floor"
    );
    assert!(
        find_output_file(&stem, MIN_VALID_DOWNLOAD_BYTES).is_some(),
        "2MB still matches the default 1MB floor"
    );
    assert!(
        existing_download(&stem, floor).is_none(),
        "existing_download must respect the effective floor"
    );

    fs::write(dir.path().join("ep1-E01.mkv"), vec![0u8; 6 * 1_048_576]).expect("write 6MB");
    assert!(
        find_output_file(&stem, floor).is_some(),
        "6MB must match the 5MB floor"
    );
}

#[test]
fn heuristic_outcome_respects_raised_floor() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test.mp4");

    // Default config: exactly the legacy 1 MiB floor.
    let default = DownloadEngine::new(DownloadConfig::default());
    fs::write(&path, vec![0u8; 512 * 1024]).expect("write 512KiB");
    assert!(
        !default.heuristic_outcome(&path).valid,
        "512KiB must fail the default heuristic"
    );
    fs::write(&path, vec![0u8; 1_048_576]).expect("write 1MiB");
    assert!(
        default.heuristic_outcome(&path).valid,
        "1MiB must pass the default heuristic"
    );

    // Raised floor via config: heuristic must reject what ffprobe would.
    let mut config = DownloadConfig::default();
    config.validation.min_size_bytes = 5 * 1_048_576;
    let raised = DownloadEngine::new(config);
    fs::write(&path, vec![0u8; 2 * 1_048_576]).expect("write 2MiB");
    assert!(
        !raised.heuristic_outcome(&path).valid,
        "2MiB must fail the 5MiB-floor heuristic"
    );
    fs::write(&path, vec![0u8; 6 * 1_048_576]).expect("write 6MiB");
    assert!(
        raised.heuristic_outcome(&path).valid,
        "6MiB must pass the 5MiB-floor heuristic"
    );
}

#[test]
fn existing_download_ignores_invalid_artifact() {
    let dir = tempdir().expect("tempdir");
    let stem = dir.path().join("ep1-E01");

    fs::write(dir.path().join("ep1-E01.unknown_video"), vec![0u8; 3078]).expect("write");
    assert!(
        existing_download(&stem, MIN_VALID_DOWNLOAD_BYTES).is_none(),
        "existing_download must not match .unknown_video file"
    );

    fs::write(dir.path().join("ep1-E01.mkv"), vec![0u8; 1_048_576]).expect("write");
    assert!(
        existing_download(&stem, MIN_VALID_DOWNLOAD_BYTES).is_some(),
        "existing_download must find valid .mkv"
    );
}

#[tokio::test]
async fn invalid_first_mirror_falls_back() {
    let dir = tempdir().expect("tempdir");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 4,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-fb".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();

    let factory = sim_factory();
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));

    let episodes = vec![EpisodeInput::new(
        1,
        vec![
            "https://cdn1.example.com/vid.mp4".into(),
            "https://cdn2.example.com/vid.mp4".into(),
        ],
    )];

    let (tx, mut rx) = broadcast::channel::<EpEvent>(256);

    let results_fut = engine.run_all(episodes, tx);

    let (results, events) = tokio::join!(results_fut, async {
        let mut events = Vec::new();
        while let Ok(ev) = rx.recv().await {
            events.push(ev.clone());
            if matches!(ev, EpEvent::Failed { .. } | EpEvent::Done { .. }) {
                break;
            }
        }
        events
    });

    assert_eq!(results.len(), 1, "one episode result expected");
    let has_done = events
        .iter()
        .any(|e| matches!(e, EpEvent::Done { ep: 1, .. }));
    let has_failed = events
        .iter()
        .any(|e| matches!(e, EpEvent::Failed { ep: 1 }));
    assert!(
        has_done || has_failed,
        "should have terminal event for ep 1"
    );
}

#[tokio::test]
async fn cleanup_invalid_output_on_failure() {
    let dir = tempdir().expect("tempdir");

    let stem = dir.path().join("test-E01");
    fs::write(dir.path().join("test-E01.unknown_video"), vec![0u8; 3078]).expect("write");

    assert!(
        find_output_file(&stem, MIN_VALID_DOWNLOAD_BYTES).is_none(),
        "find_output_file must reject .unknown_video"
    );
    assert!(
        !is_valid_output(&dir.path().join("test-E01.unknown_video")),
        "is_valid_output must reject tiny file"
    );
}

// ── Retry / backoff / classification ──────────────────────────────────────

#[tokio::test]
async fn engine_retry_transient_then_success() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-retry".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Transient(1)]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 1, "transient then success");
    assert_eq!(factory.spawn_count(url), 2, "one retry happened");
    assert!(events.iter().any(|e| matches!(
        e,
        EpEvent::RetryWait {
            ep: 1,
            attempt: 1,
            ..
        }
    )));
}

#[tokio::test]
async fn engine_no_retry_on_permanent_403() {
    let dir = tempdir().expect("tempdir");
    let url_a = "https://cda.pl/a.mp4";
    let url_b = "https://vk.com/b.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 2,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-403".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::Permanent]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url_a.into(), url_b.into()])];
    let start = Instant::now();
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 1, "falls through to second mirror");
    assert_eq!(factory.spawn_count(url_a), 1, "403 is never retried");
    assert_eq!(factory.spawn_count(url_b), 1);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, EpEvent::RetryWait { .. })),
        "no retry sleep for permanent failure"
    );
    assert!(start.elapsed() < Duration::from_secs(5), "no long sleeps");
}

#[tokio::test]
async fn engine_retry_exhausted_marks_failed_with_reasons() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-exhaust".into(),
        ..Default::default()
    };
    config.retry_attempts = 2;
    config.backoff_base_secs = 0.05;
    config.jitter_secs = 0.0;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Transient(5)]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.failed, 1, "retries exhausted");
    assert_eq!(factory.spawn_count(url), 3, "initial + 2 retries");
    assert_eq!(outcome.per_episode_reasons.len(), 1);
    assert_eq!(outcome.per_episode_reasons[0].0, 1);
    assert_eq!(outcome.per_episode_reasons[0].1, "timeout");
}

#[tokio::test]
async fn engine_failed_episode_lands_in_failed_and_missing() {
    // Invariant (failed ⊆ missing_episodes): a failed episode must appear in
    // BOTH the failed count and missing_episodes, so exit-code logic treats
    // missing_episodes as the authoritative no-output set without double
    // counting. Permanent failure skips retry → fast test.
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-invariant".into(),
        ..Default::default()
    };
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Permanent]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.failed, 1, "permanent failure counts as failed");
    assert_eq!(
        outcome.missing_episodes,
        vec![1],
        "failed episode is also in missing_episodes (failed ⊆ missing)"
    );
}

#[tokio::test]
async fn engine_backoff_exponential_observed() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-backoff".into(),
        ..Default::default()
    };
    config.backoff_base_secs = 0.5;
    config.backoff_cap_secs = 10.0;
    config.jitter_secs = 0.0;
    config.retry_attempts = 3;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Transient(2)]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let start = Instant::now();
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 1);
    // First retry backoff = base * 2^1 = 1.0s; second = 2.0s.
    assert!(
        start.elapsed() >= Duration::from_millis(1000),
        "elapsed {}ms must cover exponential backoff (>= base)",
        start.elapsed().as_millis()
    );
    let backoffs: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            EpEvent::RetryWait { backoff_secs, .. } => Some(*backoff_secs),
            _ => None,
        })
        .collect();
    assert_eq!(backoffs, vec![1, 2], "doubling backoff sequence");
}

#[tokio::test]
async fn engine_circuit_breaker_opens_after_3_systemic() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-circuit".into(),
        ..Default::default()
    };
    config.retry_attempts = 2;
    config.backoff_base_secs = 0.05;
    config.jitter_secs = 0.0;
    config.circuit_threshold = 3;
    config.circuit_cooldown_secs = 60;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Transient(3)]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.failed, 1);
    assert_eq!(factory.spawn_count(url), 3, "3 systemic failures");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EpEvent::CircuitOpened { host } if host == "h1.example.com")),
        "circuit must open after 3 systemic failures"
    );
}

#[tokio::test]
async fn engine_circuit_breaker_skips_host_while_open() {
    let dir = tempdir().expect("tempdir");
    let url_a = "https://h1.example.com/e1.mp4";
    let url_b = "https://h1.example.com/e2.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-skip".into(),
        ..Default::default()
    };
    config.retry_attempts = 2;
    config.backoff_base_secs = 0.05;
    config.jitter_secs = 0.0;
    config.circuit_threshold = 3;
    config.circuit_cooldown_secs = 60;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::Transient(3)]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![
        EpisodeInput::new(1, vec![url_a.into()]),
        EpisodeInput::new(2, vec![url_b.into()]),
    ];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.failed, 2, "both episodes fail (second skipped)");
    assert_eq!(factory.spawn_count(url_b), 0, "host skipped while open");
    assert!(
        outcome
            .per_episode_reasons
            .iter()
            .any(|(ep, r)| *ep == 2 && r.contains("circuit")),
        "skipped episode reason must mention circuit: {:?}",
        outcome.per_episode_reasons
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, EpEvent::CircuitClosed { .. })),
        "circuit stays open during cooldown"
    );
}

#[tokio::test]
async fn engine_circuit_breaker_not_opened_by_404() {
    let dir = tempdir().expect("tempdir");
    let url_a = "https://h1.example.com/e1.mp4";
    let url_b = "https://h1.example.com/e2.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 2,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-404".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::Permanent404]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![
        EpisodeInput::new(1, vec![url_a.into()]),
        EpisodeInput::new(2, vec![url_b.into()]),
    ];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 1, "episode 2 still downloads");
    assert_eq!(factory.spawn_count(url_b), 1, "host not skipped");
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, EpEvent::CircuitOpened { .. })),
        "URL-specific 404 never opens the circuit"
    );
}

#[tokio::test]
async fn engine_semaphore_wait_emits_busy_after_5s() {
    let dir = tempdir().expect("tempdir");
    let url_a = "https://h1.example.com/e1.mp4";
    let url_b = "https://h1.example.com/e2.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 2,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-busy".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::SlowPart { total_secs: 6.0 }]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![
        EpisodeInput::new(1, vec![url_a.into()]),
        EpisodeInput::new(2, vec![url_b.into()]),
    ];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 2, "both finish after lock released");
    let busy = events
        .iter()
        .filter_map(|e| match e {
            EpEvent::MirrorBusy { wait_secs, .. } => Some(*wait_secs),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        busy.iter().any(|w| *w >= 5),
        "MirrorBusy must report >=5s wait, got {busy:?}"
    );
}

#[tokio::test]
async fn engine_winner_failure_falls_to_next_mirror() {
    let dir = tempdir().expect("tempdir");
    let url_a = "https://cda.pl/a.mp4";
    let url_b = "https://vk.com/b.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 2,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-winner".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::Permanent]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url_a.into(), url_b.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 1, "fallback mirror wins");
    assert_eq!(factory.spawn_count(url_a), 1);
    assert_eq!(factory.spawn_count(url_b), 1);
}

// ── Cancellation / RAII ───────────────────────────────────────────────────

#[tokio::test]
async fn engine_cancellation_during_download_kills_and_cleans_temps() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-cancel-clean".into(),
        ..Default::default()
    };
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::SlowPart { total_secs: 2.0 }]);
    let (validator, _calls) = validators();
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));
    let cancel = engine.cancel_token();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (tx, mut rx) = broadcast::channel::<EpEvent>(256);
    let handle = tokio::spawn({
        let engine = engine.clone();
        let tx = tx.clone();
        async move { engine.run_all_with_outcome(episodes, tx).await }
    });

    sleep(Duration::from_millis(300)).await;
    cancel.cancel();

    let outcome = timeout(Duration::from_secs(10), handle)
        .await
        .expect("engine should return promptly")
        .expect("engine task");

    assert!(outcome.cancelled, "run must report cancelled");
    assert_eq!(count_temp_artifacts(dir.path()), 0, "no .part/.meas left");

    let mut saw_cancelled = false;
    while let Ok(ev) = rx.try_recv() {
        if matches!(ev, EpEvent::Cancelled { ep: 1 }) {
            saw_cancelled = true;
        }
    }
    assert!(saw_cancelled, "Cancelled event emitted");
}

#[tokio::test]
async fn engine_cancellation_flushes_manifest() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let manifest_path = dir.path().join("manifest.json");
    let config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-cancel-m".into(),
        manifest_path: Some(manifest_path.clone()),
        ..Default::default()
    };
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::SlowPart { total_secs: 2.0 }]);
    let (validator, _calls) = validators();
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));
    let cancel = engine.cancel_token();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (tx, _rx) = broadcast::channel::<EpEvent>(256);
    let handle = tokio::spawn({
        let engine = engine.clone();
        let tx = tx.clone();
        async move { engine.run_all_with_outcome(episodes, tx).await }
    });

    sleep(Duration::from_millis(300)).await;
    cancel.cancel();
    let outcome = timeout(Duration::from_secs(10), handle)
        .await
        .expect("engine should return promptly")
        .expect("engine task");
    assert!(outcome.cancelled);

    let m = Manifest::load(&manifest_path).expect("manifest flushed");
    let rec = m.episodes.iter().find(|r| r.episode == 1).expect("ep 1");
    assert_eq!(rec.final_status, FinalStatus::InProgress);
    assert!(
        !rec.attempts.is_empty(),
        "in-flight attempt recorded before flush"
    );
    assert!(
        rec.attempts
            .iter()
            .any(|a| a.reason.as_deref() == Some("cancelled"))
    );
}

#[tokio::test]
async fn engine_manifest_records_input_identity() {
    // Input identity: sha256 streamed from the source JSON, its path, and
    // `resolved_at` from the input (when present) land in the manifest.
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("source.json");
    let source_json = r#"{
        "schema_version": 2,
        "title": "Input Identity Test",
        "resolved_at": "2026-07-30T12:00:00Z",
        "episodes": [{"episode": 1, "mirrors": [{"url": "https://h1.example.com/e1.mp4"}]}]
    }"#;
    fs::write(&source, source_json).expect("write source json");
    let expected_sha = sha256_file(&source).expect("hash source json");
    let manifest_path = dir.path().join("manifest.json");

    let config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "input-identity-test".into(),
        manifest_path: Some(manifest_path.clone()),
        input_source_path: Some(source.clone()),
        input_resolved_at: Some("2026-07-30T12:00:00Z".into()),
        ..Default::default()
    };
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(
        "https://h1.example.com/e1.mp4",
        vec![FakeOutcome::Success { bad: false }],
    );
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(
        1,
        vec!["https://h1.example.com/e1.mp4".into()],
    )];
    let outcome = run(config, factory, episodes, validator).await.0;
    assert_eq!(outcome.downloaded, 1);

    let m = Manifest::load(&manifest_path).expect("manifest flushed");
    assert_eq!(
        m.input.sha256.as_deref(),
        Some(expected_sha.as_str()),
        "sha256 matches source file"
    );
    assert_eq!(
        m.input.source_json_path.as_deref(),
        Some(source.as_path()),
        "source path recorded"
    );
    assert_eq!(
        m.input.resolved_at.as_deref(),
        Some("2026-07-30T12:00:00Z"),
        "resolved_at from input"
    );
    assert_eq!(m.input.episode_count, 1);
    assert_eq!(m.input.title.as_deref(), Some("input-identity-test"));
}

// ── Validation integration ────────────────────────────────────────────────

#[tokio::test]
async fn engine_validate_newly_downloaded_ok() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-val".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let (validator, calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, events) = run(config, factory, episodes, validator).await;

    assert_eq!(outcome.downloaded, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "one validation call");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EpEvent::ValidationStarted { ep: 1 }))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        EpEvent::ValidationResult {
            ep: 1,
            ok: true,
            ..
        }
    )));
}

#[tokio::test]
async fn engine_invalid_download_quarantined_then_next_mirror() {
    let dir = tempdir().expect("tempdir");
    let url_a = "https://cda.pl/a.mp4";
    let url_b = "https://vk.com/b.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 2,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-quar".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::Success { bad: true }]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url_a.into(), url_b.into()])];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 1, "second mirror valid");
    let quarantine = dir.path().join(".quarantine");
    let quarantined: Vec<PathBuf> = fs::read_dir(&quarantine)
        .expect("quarantine dir exists")
        .flatten()
        .map(|e| e.path())
        .collect();
    assert_eq!(quarantined.len(), 1, "bad file quarantined");
    let name = quarantined[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert!(
        name.starts_with(|c: char| c.is_ascii_digit()) && name.contains("test-quar-E01.mkv"),
        "quarantine name has timestamp prefix: {name}"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            EpEvent::ValidationResult {
                ep: 1,
                ok: false,
                ..
            }
        )),
        "invalid result reported"
    );
}

#[tokio::test]
async fn engine_validation_cache_hit_no_reprobe() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let manifest_path = dir.path().join("manifest.json");
    let base_config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-cache".into(),
        manifest_path: Some(manifest_path.clone()),
        ..Default::default()
    };
    let (validator, calls) = validators();
    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let (outcome, _events) = run(
        base_config.clone(),
        factory.clone(),
        episodes.clone(),
        validator.clone(),
    )
    .await;
    assert_eq!(outcome.downloaded, 1);

    // Second run: file exists, manifest cache has it → no re-probe.
    let (outcome2, _events2) = run(base_config, factory, episodes, validator).await;
    assert_eq!(outcome2.skipped, 1, "valid cached output skipped");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "cache hit must not re-probe"
    );
}

#[tokio::test]
async fn engine_validation_cache_mtime_change_reprobes() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let manifest_path = dir.path().join("manifest.json");
    let config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-cache2".into(),
        manifest_path: Some(manifest_path.clone()),
        ..Default::default()
    };
    let (validator, calls) = validators();
    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let (outcome, _events) = run(
        config.clone(),
        factory.clone(),
        episodes.clone(),
        validator.clone(),
    )
    .await;
    assert_eq!(outcome.downloaded, 1);

    // Rewrite the file with a different size → cache miss → re-probe.
    let out = dir.path().join("test-cache2-E01.mkv");
    fs::write(&out, vec![b'G'; 2_048_576]).expect("rewrite file");

    let (outcome2, _events2) = run(config, factory, episodes, validator).await;
    assert_eq!(outcome2.skipped, 1);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "mtime/size change reprobes"
    );
}

#[tokio::test]
async fn engine_never_overwrites_valid_media() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-keep".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    // Pre-create a valid output file.
    let out = dir.path().join("test-keep-E01.mkv");
    fs::write(&out, vec![b'G'; 1_048_576]).expect("pre-create valid media");

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(
        outcome.skipped, 1,
        "valid media skipped, never redownloaded"
    );
    assert_eq!(factory.total_spawns(), 0, "no download attempt");
    let head = fs::read(&out).expect("file still present");
    assert_eq!(&head[..4], b"GGGG", "original content untouched");
}

// ── Run modes + manifest ──────────────────────────────────────────────────

#[tokio::test]
async fn engine_manifest_resume_skips_complete() {
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    let url2 = "https://h2.example.com/e2.mp4";
    let out1 = dir.path().join("test-resume-E01.mkv");
    fs::write(&out1, vec![b'G'; 1_048_576]).expect("ep1 output");
    let manifest_path = write_manifest(
        dir.path(),
        &[
            (1, FinalStatus::Complete, Some(out1.clone())),
            (2, FinalStatus::Pending, None),
        ],
    );

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 2,
        out_dir: dir.path().to_path_buf(),
        slug: "test-resume".into(),
        manifest_path: Some(manifest_path),
        ..Default::default()
    };
    config.run_mode = RunMode::Resume;
    fast_retry(&mut config);

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url2, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![
        EpisodeInput::new(1, vec![url1.into()]),
        EpisodeInput::new(2, vec![url2.into()]),
    ];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(factory.spawn_count(url1), 0, "complete episode not touched");
    assert_eq!(factory.spawn_count(url2), 1, "pending episode downloaded");
    assert_eq!(outcome.skipped, 1);
    assert_eq!(outcome.downloaded, 1);
}

#[tokio::test]
async fn engine_manifest_retry_failed_only() {
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    let url2 = "https://h2.example.com/e2.mp4";
    let url3 = "https://h3.example.com/e3.mp4";
    let out1 = dir.path().join("test-rf-E01.mkv");
    fs::write(&out1, vec![b'G'; 1_048_576]).expect("ep1 output");
    let manifest_path = write_manifest(
        dir.path(),
        &[
            (1, FinalStatus::Complete, Some(out1.clone())),
            (2, FinalStatus::Failed, None),
            (3, FinalStatus::Pending, None),
        ],
    );

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 3,
        out_dir: dir.path().to_path_buf(),
        slug: "test-rf".into(),
        manifest_path: Some(manifest_path),
        ..Default::default()
    };
    config.run_mode = RunMode::RetryFailed;
    fast_retry(&mut config);

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url2, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![
        EpisodeInput::new(1, vec![url1.into()]),
        EpisodeInput::new(2, vec![url2.into()]),
        EpisodeInput::new(3, vec![url3.into()]),
    ];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(factory.spawn_count(url1), 0, "complete excluded");
    assert_eq!(factory.spawn_count(url3), 0, "pending excluded");
    assert_eq!(factory.spawn_count(url2), 1, "failed episode retried");
    assert_eq!(outcome.downloaded, 1);
}

#[tokio::test]
async fn engine_manifest_reconcile_deleted_redownloads() {
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    // Output path registered in manifest but file deleted from disk.
    let missing_out = dir.path().join("test-rec-E01.mkv");
    let manifest_path = write_manifest(
        dir.path(),
        &[(1, FinalStatus::Complete, Some(missing_out.clone()))],
    );

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-rec".into(),
        manifest_path: Some(manifest_path),
        ..Default::default()
    };
    fast_retry(&mut config);

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url1, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url1.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(factory.spawn_count(url1), 1, "deleted output redownloaded");
    assert_eq!(outcome.downloaded, 1);
}

#[tokio::test]
async fn reconcile_marks_valid_existing_output_complete() {
    // Startup reconcile: a pre-existing valid output is recorded in the
    // manifest as Complete with full output metadata (streamed sha256) and
    // the derived summary — no download, no fabricated mirror attempt.
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    let out1 = dir.path().join("test-rec-mark-E01.mkv");
    fs::write(&out1, vec![b'G'; 1_048_576]).expect("pre-existing valid output");
    let manifest_path = dir.path().join("manifest.json");

    let config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-rec-mark".into(),
        manifest_path: Some(manifest_path.clone()),
        ..Default::default()
    };
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url1, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url1.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(
        outcome.skipped, 1,
        "existing valid output counted as skipped, not downloaded"
    );
    assert_eq!(outcome.downloaded, 0);
    assert_eq!(factory.total_spawns(), 0, "no download attempted");

    let m = Manifest::load(&manifest_path).expect("manifest flushed");
    let rec = m.episodes.iter().find(|r| r.episode == 1).expect("ep 1");
    assert_eq!(rec.final_status, FinalStatus::Complete);
    assert!(rec.attempts.is_empty(), "no fabricated mirror attempts");
    let out = rec.output.as_ref().expect("output meta recorded");
    assert_eq!(out.path, out1);
    assert_eq!(out.size, 1_048_576);
    assert!(out.validated);
    assert_eq!(out.ffprobe_version.as_deref(), Some("ffprobe 7.1"));
    assert!(out.checked_at.is_some(), "checked_at recorded");
    assert_eq!(
        out.sha256.as_deref(),
        Some(sha256_file(&out1).expect("hash output").as_str()),
        "streamed sha256 recorded"
    );
    assert_eq!(
        m.summary.downloaded, 1,
        "Complete counts as downloaded in the manifest summary"
    );
    assert_eq!(m.summary.skipped, 0);
    assert_eq!(m.summary.failed, 0);
}

#[tokio::test]
async fn reconcile_with_invalid_artifact_keeps_pending() {
    // An invalid present artifact is NOT recorded by reconcile: the episode
    // stays eligible (Pending — no prior status), the artifact is quarantined
    // by the per-episode path, and only a real mirror attempt marks status.
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    // Invalid artifact: passes the extension+size heuristic but fails the
    // fake validator (BAD! prefix).
    let out1 = dir.path().join("test-rec-bad-E01.mkv");
    let mut bad = b"BAD!".to_vec();
    bad.extend(std::iter::repeat_n(b'X', 1_048_576));
    fs::write(&out1, &bad).expect("invalid artifact");
    let manifest_path = dir.path().join("manifest.json");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-rec-bad".into(),
        manifest_path: Some(manifest_path.clone()),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url1, vec![FakeOutcome::Permanent]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url1.into()])];
    let (outcome, _events) = run(config, factory, episodes, validator).await;

    assert_eq!(
        outcome.failed, 1,
        "invalid artifact + failing mirror -> failed"
    );
    assert!(!out1.exists(), "invalid artifact quarantined");

    let m = Manifest::load(&manifest_path).expect("manifest flushed");
    let rec = m.episodes.iter().find(|r| r.episode == 1).expect("ep 1");
    assert_ne!(
        rec.final_status,
        FinalStatus::Complete,
        "invalid output never marked Complete"
    );
    assert!(
        rec.output.is_none(),
        "invalid artifact never recorded as output"
    );
    assert!(!rec.attempts.is_empty(), "real mirror attempt recorded");
}

#[tokio::test]
async fn resume_after_reconcile_skips_complete_via_manifest() {
    // Resume fast-path: manifest Complete + unchanged file -> skip WITHOUT an
    // ffprobe probe (validation-cache hit). A changed file is re-probed and
    // re-validated before the manifest status is trusted.
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    let out1 = dir.path().join("test-resume-rec-E01.mkv");
    let manifest_path = dir.path().join("manifest.json");
    let base_config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-resume-rec".into(),
        manifest_path: Some(manifest_path.clone()),
        ..Default::default()
    };
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url1, vec![FakeOutcome::Success { bad: false }]);
    let (validator, calls) = validators();

    // Run 1 (default): download -> Complete recorded in the manifest.
    let episodes = vec![EpisodeInput::new(1, vec![url1.into()])];
    let (outcome, _e) = run(
        base_config.clone(),
        factory.clone(),
        episodes.clone(),
        validator.clone(),
    )
    .await;
    assert_eq!(outcome.downloaded, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "download validated once");
    assert!(out1.exists());

    // Run 2 (resume, unchanged file): skip via manifest Complete + cache hit
    // — no re-probe, no re-download.
    let mut resume = base_config.clone();
    resume.run_mode = RunMode::Resume;
    let (outcome2, _e2) = run(
        resume.clone(),
        factory.clone(),
        episodes.clone(),
        validator.clone(),
    )
    .await;
    assert_eq!(outcome2.skipped, 1);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "unchanged file: no probe on resume"
    );
    assert_eq!(factory.spawn_count(url1), 1, "no re-download");

    // Run 3 (resume, changed file): size/mtime differ -> re-probed; the new
    // content is still valid so the episode stays Complete and is skipped.
    fs::write(&out1, vec![b'G'; 2_048_576]).expect("rewrite changed file");
    let (outcome3, _e3) = run(resume, factory, episodes, validator).await;
    assert_eq!(outcome3.skipped, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2, "changed file re-probed");
}

#[tokio::test]
async fn engine_run_mode_validate_only_no_download() {
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    let url2 = "https://h2.example.com/e2.mp4";
    let out1 = dir.path().join("test-vo-E01.mkv");
    fs::write(&out1, vec![b'G'; 1_048_576]).expect("ep1 output");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 2,
        out_dir: dir.path().to_path_buf(),
        slug: "test-vo".into(),
        ..Default::default()
    };
    config.run_mode = RunMode::ValidateOnly;

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url1, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![
        EpisodeInput::new(1, vec![url1.into()]),
        EpisodeInput::new(2, vec![url2.into()]),
    ];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(
        factory.total_spawns(),
        0,
        "no yt-dlp spawn in validate-only"
    );
    assert_eq!(outcome.skipped, 1, "ep 1 validated ok");
    assert_eq!(outcome.missing_episodes, vec![2], "ep 2 missing");
    assert!(events.iter().any(|e| matches!(
        e,
        EpEvent::ValidationResult {
            ep: 1,
            ok: true,
            ..
        }
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        EpEvent::ValidationResult {
            ep: 2,
            ok: false,
            ..
        }
    )));
}

#[tokio::test]
async fn engine_part_resume_uses_continue_flag() {
    let dir = tempdir().expect("tempdir");
    let url1 = "https://h1.example.com/e1.mp4";
    let manifest_path = dir.path().join("manifest.json");
    let mut m = Manifest::new();
    m.ensure_episode(1);
    m.set_final_status(1, FinalStatus::Failed);
    m.record_attempt(
        1,
        crate::manifest::AttemptRecord {
            mirror_idx: 0,
            host: Some("h1.example.com".into()),
            url: url1.into(),
            status: AttemptStatus::Failed,
            reason: Some("timeout".into()),
            bytes_downloaded: 0,
            secs: 0.0,
            started_at: None,
        },
    );
    m.save_atomic(&manifest_path).expect("save manifest");

    // Leftover partial file from the previous run.
    let part = dir.path().join("test-part-E01.mkv.part");
    fs::write(&part, b"partial").expect("write part");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-part".into(),
        manifest_path: Some(manifest_path),
        ..Default::default()
    };
    config.run_mode = RunMode::Resume;
    fast_retry(&mut config);

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url1, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url1.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.downloaded, 1);
    assert!(
        factory.continue_flag_seen.load(Ordering::SeqCst),
        "resume must pass the yt-dlp -c continue flag"
    );
}

// ── Leftover / hygiene ────────────────────────────────────────────────────

#[tokio::test]
async fn engine_no_leftovers_after_failed_episode() {
    let dir = tempdir().expect("tempdir");
    let url_a = "https://cda.pl/a.mp4";
    let url_b = "https://vk.com/b.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 2,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-left".into(),
        ..Default::default()
    };
    config.retry_attempts = 1;
    config.backoff_base_secs = 0.05;
    config.jitter_secs = 0.0;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::Permanent]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Permanent]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url_a.into(), url_b.into()])];
    let (outcome, _events) = run(config, factory, episodes, validator).await;

    assert_eq!(outcome.failed, 1, "all mirrors failed");
    assert_eq!(count_temp_artifacts(dir.path()), 0, "no leftovers");
}

#[tokio::test]
async fn engine_stderr_flood_bounded() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-flood".into(),
        ..Default::default()
    };
    config.retry_attempts = 1;
    config.backoff_base_secs = 0.05;
    config.jitter_secs = 0.0;
    let factory = Arc::new(FakeFactory::new());
    factory.stderr_flood_lines.store(20_000, Ordering::SeqCst);
    factory.set_outcomes(url, vec![FakeOutcome::Transient(1)]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    // Completion within a generous bound proves the flood was drained
    // without unbounded buffering (engine reads stderr to EOF, keeps 5 lines).
    let (outcome, _events) = timeout(
        Duration::from_secs(30),
        run(config, factory, episodes, validator),
    )
    .await
    .expect("flood test must complete");
    assert_eq!(outcome.downloaded, 1, "second attempt succeeds");
}

#[tokio::test]
async fn engine_stderr_urls_redacted_in_events() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-red".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    *factory.error_line_override.lock().expect("lock") =
        Some("ERROR: https://cdn.example.com/v?token=SECRET123 denied".to_string());
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, _events) = run(config, factory, episodes, validator).await;

    assert_eq!(outcome.failed, 1);
    assert_eq!(outcome.per_episode_reasons.len(), 1);
    let reason = &outcome.per_episode_reasons[0].1;
    assert!(
        !reason.contains("SECRET123"),
        "query token must be redacted, got: {reason}"
    );
    assert!(reason.contains("[URL]"), "URL redacted, got: {reason}");
}

// ── Exit code helper ──────────────────────────────────────────────────────

#[test]
fn engine_exit_code_helper() {
    let ok = Outcome {
        downloaded: 2,
        skipped: 1,
        failed: 0,
        cancelled: false,
        ..Default::default()
    };
    assert_eq!(ok.exit_code(), 0, "all ok → 0");

    let partial = Outcome {
        downloaded: 2,
        skipped: 0,
        failed: 1,
        cancelled: false,
        ..Default::default()
    };
    assert_eq!(partial.exit_code(), 3, "some failed, some ok → 3");

    let all_failed = Outcome {
        downloaded: 0,
        skipped: 0,
        failed: 3,
        cancelled: false,
        ..Default::default()
    };
    assert_eq!(all_failed.exit_code(), 4, "all failed → 4");

    let cancelled = Outcome {
        downloaded: 1,
        skipped: 0,
        failed: 0,
        cancelled: true,
        ..Default::default()
    };
    assert_eq!(cancelled.exit_code(), 130, "cancelled → 130");
}

// ── Shared-path regression: single-mirror used by many tests ──────────────

#[tokio::test]
async fn engine_part_cleanup_uses_registry_not_prefix_blast() {
    // A user file with an unrelated name in the out dir must never be deleted
    // by engine cleanup (registry only touches slug-prefixed artifacts).
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-safe".into(),
        ..Default::default()
    };
    config.retry_attempts = 1;
    config.backoff_base_secs = 0.05;
    config.jitter_secs = 0.0;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Permanent]);
    let (validator, _calls) = validators();
    let user_file = dir.path().join("user-notes.txt");
    fs::write(&user_file, b"precious user data").expect("user file");

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, _events) = run(config, factory, episodes, validator).await;

    assert_eq!(outcome.failed, 1);
    assert!(
        user_file.exists(),
        "unregistered user file must survive engine cleanup"
    );
}

// ── Review-finding regression tests (lane 2.5) ─────────────────────────────

/// Validator that reports a degraded-but-valid outcome (heuristic pass with a
/// warn reason), as real ffprobe validation does when the binary is missing.
struct DegradedValidator;

impl MediaValidator for DegradedValidator {
    fn validate(
        &self,
        _path: &Path,
        _cfg: &ValidationConfig,
    ) -> Result<ValidationOutcome, ValidatorError> {
        Ok(ValidationOutcome {
            valid: true,
            reason: Some(crate::validator::DEGRADED_REASON.to_string()),
            video_stream: false,
            audio_stream: false,
            duration_secs: None,
            width: None,
            height: None,
            ffprobe_version: None,
        })
    }
}

#[tokio::test]
async fn quality_filter_selects_highest_mirror_v2_explicit() {
    // v2 path: mirrors carry explicit per-mirror quality strings → the
    // quality-first filter picks the highest, and mirrors with explicit
    // quality skip format inspection entirely (no fake_quality needed).
    let dir = tempdir().expect("tempdir");
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 4,
        out_dir: dir.path().to_path_buf(),
        slug: "test-v2q".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    let (validator, _calls) = validators();

    let u480 = "https://h1.example.com/v.mp4";
    let u1080 = "https://h2.example.com/v.mp4";
    let u720 = "https://h3.example.com/v.mp4";
    let ep = EpisodeInput {
        episode: 1,
        mirrors: vec![
            Mirror {
                host: Some("h1.example.com".into()),
                quality: Some("480p".into()),
                subtitle_group: None,
                url: u480.into(),
            },
            Mirror {
                host: Some("h2.example.com".into()),
                quality: Some("1080p".into()),
                subtitle_group: None,
                url: u1080.into(),
            },
            Mirror {
                host: Some("h3.example.com".into()),
                quality: Some("720p".into()),
                subtitle_group: None,
                url: u720.into(),
            },
        ],
        urls: vec![u480.into(), u1080.into(), u720.into()],
        quality: None,
    };

    factory.set_outcomes(u1080, vec![FakeOutcome::Success { bad: false }]);
    let (outcome, _events) = run(config, factory.clone(), vec![ep], validator).await;

    assert_eq!(outcome.downloaded, 1);
    assert_eq!(factory.spawn_count(u1080), 1, "1080p mirror chosen");
    assert_eq!(factory.spawn_count(u480), 0, "480p mirror not attempted");
    assert_eq!(factory.spawn_count(u720), 0, "720p mirror not attempted");
}

#[tokio::test]
async fn engine_continuation_completes_no_live_rename() {
    // The winner keeps running to completion; its live `.part` is NEVER
    // renamed in place. yt-dlp finalizes inside the meas namespace on exit 0,
    // then the engine renames `{stem}.meas{idx}.mkv` → final.
    let dir = tempdir().expect("tempdir");
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 4,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-cont-new".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();
    let factory = Arc::new(SimYtDlpFactory {
        speed_mibs: 100.0,
        progress_lines: 2,
        work_secs: 0.05,
        // Still downloading when winner selection happens (~1s), completes
        // shortly after: exercises the real continuation path, not the
        // already-finished shortcut.
        completion_delay_secs: 3.0,
        fail_measurement: false,
    });
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));

    let episodes = vec![EpisodeInput::new(
        1,
        vec![
            "https://cdn1.example.com/vid.mp4".into(),
            "https://cdn2.example.com/vid.mp4".into(),
        ],
    )];

    let (tx, _rx) = broadcast::channel::<EpEvent>(256);
    let results = engine.run_all(episodes, tx).await;

    assert_eq!(results.len(), 1);
    let final_path = results[0].1.clone().expect("final output must exist");
    let fname = final_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert_eq!(fname, "test-cont-new-E01.mkv");
    // No in-place live-file rename: the winner's `.part` was finalized inside
    // the meas namespace by the fake yt-dlp and renamed by the engine only
    // after exit 0. A live rename would leave `{stem}.mkv.part` behind.
    assert!(!dir.path().join("test-cont-new-E01.mkv.part").exists());
    assert!(!dir.path().join("test-cont-new-E01.mp4.part").exists());
    assert_eq!(
        fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0),
        1_048_576,
        "final file has the completed size"
    );
}

#[tokio::test]
async fn engine_loser_killed_at_winner_selection() {
    let dir = tempdir().expect("tempdir");
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 4,
        measurement_secs: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-loser".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let (validator, _calls) = validators();
    let factory = Arc::new(SimYtDlpFactory {
        speed_mibs: 100.0,
        progress_lines: 2,
        work_secs: 0.05,
        completion_delay_secs: 3.0,
        fail_measurement: false,
    });
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));

    let episodes = vec![EpisodeInput::new(
        1,
        vec![
            "https://cdn1.example.com/vid.mp4".into(),
            "https://cdn2.example.com/vid.mp4".into(),
        ],
    )];

    let (tx, _rx) = broadcast::channel::<EpEvent>(256);
    let results = engine.run_all(episodes, tx).await;
    assert_eq!(results.len(), 1);
    assert!(results[0].1.is_some(), "winner produced output");

    let names: Vec<String> = fs::read_dir(dir.path())
        .expect("out dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(
        names.iter().all(|n| !n.contains(".meas")),
        "loser meas artifacts removed: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "test-loser-E01.mkv"),
        "final file present: {names:?}"
    );
}

#[tokio::test]
async fn engine_multi_format_fragment_cleanup() {
    // Stale artifacts from an interrupted earlier run (a loser mirror that
    // finalized its meas output + yt-dlp fragment temps), unregistered in the
    // current run: swept to quarantine, never deleted, nothing left in out_dir.
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("test-frag-E02.meas1.mkv"),
        vec![b'X'; 1_048_576],
    )
    .expect("stale meas");
    fs::write(
        dir.path().join("test-frag-E02.meas1.f001.mp4"),
        b"fragment1",
    )
    .expect("stale frag");
    fs::write(
        dir.path().join("test-frag-E02.meas1.f002.mp4"),
        b"fragment2",
    )
    .expect("stale frag2");

    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-frag".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    let url = "https://h1.example.com/e1.mp4";
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, _events) = run(config, factory, episodes, validator).await;
    assert_eq!(outcome.downloaded, 1);

    let names: Vec<String> = fs::read_dir(dir.path())
        .expect("out dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(
        names
            .iter()
            .all(|n| !n.contains(".meas") && !n.contains(".f00")),
        "stale artifacts swept from out dir: {names:?}"
    );
    let quarantine = dir.path().join(".quarantine");
    let qnames: Vec<String> = fs::read_dir(&quarantine)
        .expect("quarantine dir")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(
        qnames.iter().any(|n| n.contains("E02.meas1.mkv")),
        "meas mkv quarantined: {qnames:?}"
    );
    assert!(
        qnames.iter().any(|n| n.contains("E02.meas1.f001")),
        "fragment quarantined: {qnames:?}"
    );
}

#[tokio::test]
async fn engine_cancel_during_backoff() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-backoff-cancel".into(),
        ..Default::default()
    };
    config.retry_attempts = 3;
    config.backoff_base_secs = 60.0; // long sleep; cancel must abort it
    config.backoff_cap_secs = 60.0;
    config.jitter_secs = 0.0;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Transient(3)]);
    let (validator, _calls) = validators();
    let engine = Arc::new(DownloadEngine::with_factory_and_validator(
        config, factory, validator,
    ));
    let cancel = engine.cancel_token();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (tx, _rx) = broadcast::channel::<EpEvent>(256);
    let handle = tokio::spawn({
        let engine = engine.clone();
        async move { engine.run_all_with_outcome(episodes, tx).await }
    });

    // Let the first attempt fail and the retry backoff sleep begin.
    sleep(Duration::from_millis(600)).await;
    let start = Instant::now();
    cancel.cancel();
    let outcome = timeout(Duration::from_secs(5), handle)
        .await
        .expect("engine returns promptly after cancel")
        .expect("engine task");
    assert!(outcome.cancelled, "run reports cancelled");
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "backoff sleep aborted promptly ({}ms)",
        start.elapsed().as_millis()
    );
    assert_eq!(count_temp_artifacts(dir.path()), 0, "no temp left");
}

#[tokio::test]
async fn engine_degraded_ffprobe() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-degraded".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let validator: Arc<dyn MediaValidator + Send + Sync> = Arc::new(DegradedValidator);

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, events) = run(config, factory, episodes, validator).await;
    assert_eq!(
        outcome.downloaded, 1,
        "degraded-but-valid outcome accepted (heuristic pass)"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            EpEvent::ValidationResult {
                ep: 1,
                ok: true,
                reason: Some(r),
            } if r.contains("degraded")
        )),
        "degraded warn reason surfaced in ValidationResult"
    );
}

#[tokio::test]
async fn engine_circuit_blocks_same_episode_retries() {
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-circuit2".into(),
        ..Default::default()
    };
    config.retry_attempts = 4;
    config.backoff_base_secs = 0.05;
    config.backoff_cap_secs = 2.0;
    config.jitter_secs = 0.0;
    config.circuit_threshold = 3;
    config.circuit_cooldown_secs = 60;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Transient(5)]);
    let (validator, _calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, events) = run(config, factory.clone(), episodes, validator).await;

    assert_eq!(outcome.failed, 1);
    assert_eq!(
        factory.spawn_count(url),
        3,
        "circuit open blocks remaining same-episode retries"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EpEvent::CircuitOpened { .. })),
        "circuit opened after 3 systemic failures"
    );
    assert!(
        outcome.per_episode_reasons[0].1.contains("circuit"),
        "reason mentions circuit: {}",
        outcome.per_episode_reasons[0].1
    );
}

#[tokio::test]
async fn resume_part_continue_same_url() {
    // Part-continue (-c) applies ONLY when the resumed URL equals the last
    // manifest attempt's URL; a changed URL must start fresh.

    // Scenario A: manifest attempt URL differs from the episode URL → no -c.
    let dir_a = tempdir().expect("tempdir");
    let manifest_a = dir_a.path().join("manifest.json");
    let mut m_a = Manifest::new();
    m_a.ensure_episode(1);
    m_a.set_final_status(1, FinalStatus::Failed);
    m_a.record_attempt(
        1,
        crate::manifest::AttemptRecord {
            mirror_idx: 0,
            host: Some("h1.example.com".into()),
            url: "https://h1.example.com/old.mp4".into(),
            status: AttemptStatus::Failed,
            reason: Some("timeout".into()),
            bytes_downloaded: 0,
            secs: 0.0,
            started_at: None,
        },
    );
    m_a.save_atomic(&manifest_a).expect("save manifest a");
    fs::write(dir_a.path().join("test-resume2-E01.mkv.part"), b"partial").expect("write part");

    let mut config_a = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir_a.path().to_path_buf(),
        slug: "test-resume2".into(),
        manifest_path: Some(manifest_a),
        ..Default::default()
    };
    config_a.run_mode = RunMode::Resume;
    fast_retry(&mut config_a);
    let factory_a = Arc::new(FakeFactory::new());
    let url_new = "https://h1.example.com/new.mp4";
    factory_a.set_outcomes(url_new, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();
    let episodes_a = vec![EpisodeInput::new(1, vec![url_new.into()])];
    let (outcome_a, _e) = run(config_a, factory_a.clone(), episodes_a, validator.clone()).await;
    assert_eq!(outcome_a.downloaded, 1);
    assert!(
        !factory_a.continue_flag_seen.load(Ordering::SeqCst),
        "different URL must NOT continue the old part"
    );

    // Scenario B: same URL as the manifest attempt → -c passed.
    let dir_b = tempdir().expect("tempdir");
    let manifest_b = dir_b.path().join("manifest.json");
    let mut m_b = Manifest::new();
    m_b.ensure_episode(1);
    m_b.set_final_status(1, FinalStatus::Failed);
    m_b.record_attempt(
        1,
        crate::manifest::AttemptRecord {
            mirror_idx: 0,
            host: Some("h1.example.com".into()),
            url: "https://h1.example.com/old.mp4".into(),
            status: AttemptStatus::Failed,
            reason: Some("timeout".into()),
            bytes_downloaded: 0,
            secs: 0.0,
            started_at: None,
        },
    );
    m_b.save_atomic(&manifest_b).expect("save manifest b");
    fs::write(dir_b.path().join("test-resume2-E01.mkv.part"), b"partial").expect("write part");

    let mut config_b = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir_b.path().to_path_buf(),
        slug: "test-resume2".into(),
        manifest_path: Some(manifest_b),
        ..Default::default()
    };
    config_b.run_mode = RunMode::Resume;
    fast_retry(&mut config_b);
    let factory_b = Arc::new(FakeFactory::new());
    let url_same = "https://h1.example.com/old.mp4";
    factory_b.set_outcomes(url_same, vec![FakeOutcome::Success { bad: false }]);
    let episodes_b = vec![EpisodeInput::new(1, vec![url_same.into()])];
    let (outcome_b, _e2) = run(config_b, factory_b.clone(), episodes_b, validator).await;
    assert_eq!(outcome_b.downloaded, 1);
    assert!(
        factory_b.continue_flag_seen.load(Ordering::SeqCst),
        "same URL resumes with -c"
    );
}

// ── Final-fix-lane gap tests (validation knobs + final summary) ────────────

#[tokio::test]
async fn no_validate_skips_ffprobe() {
    // --no-validate: the fake validator must NEVER be invoked; the legacy
    // extension+size heuristic decides (fake download writes a 1MiB .mkv,
    // which passes). The ffprobe cache is neither read nor written.
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-noval".into(),
        ..Default::default()
    };
    fast_retry(&mut config);
    config.no_validate = true;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let (validator, calls) = validators();

    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, events) = run(config, factory, episodes, validator).await;

    assert_eq!(outcome.downloaded, 1, "heuristic pass accepted");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no ffprobe validation when --no-validate"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            EpEvent::ValidationResult {
                ep: 1,
                ok: true,
                reason: Some(r),
            } if r.contains("heuristic")
        )),
        "heuristic outcome surfaced in events"
    );
}

#[tokio::test]
async fn no_validate_never_overwrites_heuristic_passing_media() {
    // Pre-existing valid media must be skipped (never overwritten) even in
    // no-validate mode — the heuristic treats it as valid.
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-noval2".into(),
        no_validate: true,
        ..Default::default()
    };
    let out = dir.path().join("test-noval2-E01.mkv");
    fs::write(&out, vec![b'G'; 1_048_576]).expect("pre-create valid media");
    let factory = Arc::new(FakeFactory::new());
    let (validator, _calls) = validators();
    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];
    let (outcome, _events) = run(config, factory.clone(), episodes, validator).await;
    assert_eq!(outcome.skipped, 1, "heuristic-passing media skipped");
    assert_eq!(factory.total_spawns(), 0, "no download attempt");
    let head = fs::read(&out).expect("file still present");
    assert_eq!(&head[..4], b"GGGG", "original content untouched");
}

#[tokio::test]
async fn validate_force_reprobes_despite_cache_hit() {
    // --validate-force bypasses the size+mtime validation cache: run 1 probes
    // and caches; run 2 with validate_force probes again even though the file
    // is unchanged (cache key identical).
    let dir = tempdir().expect("tempdir");
    let url = "https://h1.example.com/e1.mp4";
    let manifest_path = dir.path().join("manifest.json");
    let base_config = DownloadConfig {
        episode_concurrency: 1,
        host_concurrency: 1,
        out_dir: dir.path().to_path_buf(),
        slug: "test-force".into(),
        manifest_path: Some(manifest_path.clone()),
        ..Default::default()
    };
    let (validator, calls) = validators();
    let episodes = vec![EpisodeInput::new(1, vec![url.into()])];

    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url, vec![FakeOutcome::Success { bad: false }]);
    let (outcome, _events) = run(
        base_config.clone(),
        factory.clone(),
        episodes.clone(),
        validator.clone(),
    )
    .await;
    assert_eq!(outcome.downloaded, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "first run probes");

    // Second run, file untouched, validate_force set.
    let mut forced = base_config;
    forced.validate_force = true;
    let (outcome2, _events2) = run(forced, factory, episodes, validator).await;
    assert_eq!(outcome2.skipped, 1, "still valid after forced reprobe");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "validate_force reprobes despite cache hit"
    );
}

#[tokio::test]
async fn final_summary_event_carries_per_episode_reasons() {
    // The engine's authoritative per-episode reasons must ride the
    // FinalSummary event so plain output and dashboard render them.
    let dir = tempdir().expect("tempdir");
    let url_a = "https://h1.example.com/e1.mp4";
    let url_b = "https://h2.example.com/e2.mp4";
    let mut config = DownloadConfig {
        episode_concurrency: 2,
        host_concurrency: 2,
        out_dir: dir.path().to_path_buf(),
        slug: "test-fs".into(),
        ..Default::default()
    };
    config.retry_attempts = 1;
    config.backoff_base_secs = 0.05;
    config.jitter_secs = 0.0;
    let factory = Arc::new(FakeFactory::new());
    factory.set_outcomes(url_a, vec![FakeOutcome::Permanent]);
    factory.set_outcomes(url_b, vec![FakeOutcome::Success { bad: false }]);
    let (validator, _calls) = validators();
    let episodes = vec![
        EpisodeInput::new(1, vec![url_a.into()]),
        EpisodeInput::new(2, vec![url_b.into()]),
    ];
    let (outcome, events) = run(config, factory, episodes, validator).await;

    assert_eq!(outcome.failed, 1);
    assert_eq!(outcome.per_episode_reasons.len(), 1);
    let summary = events
        .iter()
        .find_map(|e| match e {
            EpEvent::FinalSummary {
                per_episode_reasons,
                ..
            } => Some(per_episode_reasons.clone()),
            _ => None,
        })
        .expect("FinalSummary event present");
    assert_eq!(summary, outcome.per_episode_reasons);
    assert_eq!(summary[0].0, 1);
    assert_eq!(summary[0].1, "http 403 forbidden");
}
