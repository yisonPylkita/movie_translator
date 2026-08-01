//! Host policy layer for the anime downloader.
//!
//! Single source of truth for host identification, URL canonicalization,
//! per-host timeout profiles, and yt-dlp stderr error classification.

use std::fmt;

// ── Host identity ──────────────────────────────────────────────────────────

/// Identified streaming host. `Generic` = any unknown http(s) URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostId {
    Cda,
    Rumble,
    Vk,
    Sibnet,
    Dood,
    Hqq,
    Mega,
    Ok,
    Myvi,
    Google,
    Voe,
    Mp4upload,
    Generic,
}

impl fmt::Display for HostId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            HostId::Cda => "cda",
            HostId::Rumble => "rumble",
            HostId::Vk => "vk",
            HostId::Sibnet => "sibnet",
            HostId::Dood => "dood",
            HostId::Hqq => "hqq",
            HostId::Mega => "mega",
            HostId::Ok => "ok",
            HostId::Myvi => "myvi",
            HostId::Google => "google",
            HostId::Voe => "voe",
            HostId::Mp4upload => "mp4upload",
            HostId::Generic => "generic",
        };
        f.write_str(s)
    }
}

/// Host preference rank for tie-breaking. Lower index = more preferred.
pub const HOST_PREFERENCE_RANK: [HostId; 12] = [
    HostId::Cda,
    HostId::Rumble,
    HostId::Vk,
    HostId::Sibnet,
    HostId::Mega,
    HostId::Ok,
    HostId::Dood,
    HostId::Myvi,
    HostId::Google,
    HostId::Hqq,
    HostId::Voe,
    HostId::Mp4upload,
];

/// Canonical preference order, most preferred first.
pub fn host_preference_rank() -> &'static [HostId] {
    &HOST_PREFERENCE_RANK
}

/// All known (non-generic) hosts.
pub fn all_hosts() -> Vec<HostId> {
    HOST_PREFERENCE_RANK.to_vec()
}

/// Identify the host from a URL or bare hostname string.
///
/// Marker substrings cover both full URL patterns (`vk.com/video_ext.php`,
/// `rumble.com/embed/`) and bare hostnames (`vk.com`, `video.sibnet.ru`),
/// so callers may pass either a raw URL or the output of a hostname extractor.
pub fn identify_host(s: &str) -> HostId {
    let s_lower = s.to_lowercase();
    let has = |pat: &str| s_lower.contains(pat);
    if has("vk.com/video_ext.php") || has("vkvideo.ru") || has("vk.com") {
        HostId::Vk
    } else if has("rumble.com/embed/") || has("rumble.com") {
        HostId::Rumble
    } else if has("cda.pl/video/") || has("ebd.cda.pl") || has("cda.pl") {
        HostId::Cda
    } else if has("video.sibnet.ru") || has("sibnet.ru") {
        HostId::Sibnet
    } else if has("dood.yt") || has("dood.re") || has("dood.ws") || has("dood.la") {
        HostId::Dood
    } else if has("hqq.tv") || has("hqq.to") || has("hqq.ac") {
        HostId::Hqq
    } else if has("mega.nz") {
        HostId::Mega
    } else if has("ok.ru/video") || has("ok.ru") {
        HostId::Ok
    } else if has("myvi.tv") || has("myvi.id") {
        HostId::Myvi
    } else if has("drive.google.com") {
        HostId::Google
    } else if has("voe.sx") || has("voe-") {
        HostId::Voe
    } else if has("mp4upload.com") {
        HostId::Mp4upload
    } else {
        HostId::Generic
    }
}

// ── URL canonicalization ───────────────────────────────────────────────────

/// Canonicalize VK `video_ext.php` URL to `vkvideo.ru` direct format.
/// `https://vk.com/video_ext.php?oid=-229809086&id=456239061&hash=...&hd=2`
/// becomes `https://vkvideo.ru/video-229809086_456239061`.
/// Returns `None` if URL is not a VK video_ext.php URL or params missing.
pub fn try_canonicalize_vk_url(url: &str) -> Option<String> {
    // Must be from vk.com with video_ext.php path
    let url_lower = url.to_lowercase();
    if !url_lower.starts_with("https://vk.com/video_ext.php")
        && !url_lower.starts_with("http://vk.com/video_ext.php")
        && !url_lower.starts_with("https://www.vk.com/video_ext.php")
        && !url_lower.starts_with("http://www.vk.com/video_ext.php")
    {
        return None;
    }

    // Extract query string
    let query = url.split('?').nth(1)?;

    // Parse oid and id params manually (preserve negative sign in oid)
    let mut oid: Option<String> = None;
    let mut id: Option<String> = None;
    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        let key = parts.next()?;
        let val = parts.next().unwrap_or("");
        match key {
            "oid" => oid = Some(val.to_string()),
            "id" => id = Some(val.to_string()),
            _ => {}
        }
    }

    let oid = oid?;
    let id = id?;

    // Validate: both must be non-empty and numeric (possibly with leading -)
    if oid.is_empty() || id.is_empty() {
        return None;
    }
    let valid_oid = oid
        .as_bytes()
        .iter()
        .enumerate()
        .all(|(i, &b)| b.is_ascii_digit() || (i == 0 && b == b'-'));
    let valid_id = id.as_bytes().iter().all(|&b| b.is_ascii_digit());
    if !valid_oid || !valid_id {
        return None;
    }

    Some(format!("https://vkvideo.ru/video{}_{}", oid, id))
}

/// Canonicalize a URL: VK `video_ext.php` → `vkvideo.ru` direct form;
/// everything else passes through (trailing slash trimmed, URLs kept intact —
/// token stripping is a display concern, not a manifest concern).
pub fn canonicalize(url: &str) -> String {
    match try_canonicalize_vk_url(url) {
        Some(vk) => vk,
        None => {
            let trimmed = url.trim_end_matches('/');
            // Never strip below the "https://" / "http://" scheme prefix.
            if trimmed.len() >= url.len().saturating_sub(1) && trimmed.contains("://") {
                trimmed.to_string()
            } else {
                url.to_string()
            }
        }
    }
}

// ── Timeout profiles ───────────────────────────────────────────────────────

/// Timeout profile for a host: startup grace and per-download stall budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutProfile {
    pub startup_secs: u64,
    pub stall_secs: u64,
}

/// Default profile for generic/unknown hosts.
pub fn default_timeout_profile() -> TimeoutProfile {
    TimeoutProfile {
        startup_secs: 30,
        stall_secs: 120,
    }
}

/// Per-host timeout profile.
///
/// cda: slower startup (45s). hqq/dood: slow startup + long stall (60/180).
/// rumble/vk: normal startup, long stall (180s). Everything else: 30/120.
pub fn timeout_profile_for(host: HostId) -> TimeoutProfile {
    match host {
        HostId::Cda => TimeoutProfile {
            startup_secs: 45,
            stall_secs: 120,
        },
        HostId::Hqq | HostId::Dood => TimeoutProfile {
            startup_secs: 60,
            stall_secs: 180,
        },
        HostId::Rumble | HostId::Vk => TimeoutProfile {
            startup_secs: 30,
            stall_secs: 180,
        },
        _ => default_timeout_profile(),
    }
}

// ── Error classification ───────────────────────────────────────────────────

/// Class of a download failure, driving retry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Retryable(RetryableKind),
    Permanent(PermanentKind),
    Unknown,
}

/// Failures that deserve a retry (possibly with backoff).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryableKind {
    Dns,
    Connect,
    Timeout,
    Http429,
    Http5xx,
    Stall,
    ExtractNotReady,
}

/// Failures that will not succeed on retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermanentKind {
    Http403,
    Http404,
    UnsupportedUrl,
    AuthChallenge,
    FormatNotAvailable,
    InvalidInput,
}

/// Pick the line containing `ERROR:` from yt-dlp stderr, if any.
fn error_line(stderr: &str) -> Option<&str> {
    stderr
        .lines()
        .find(|l| l.to_ascii_lowercase().contains("error:"))
}

/// Classify a yt-dlp failure from stderr + optional exit code.
///
/// The `ERROR:` line (if present) is the primary classifier; otherwise the
/// whole stderr is scanned. Marker matching is case-insensitive. Exit code is
/// kept as a hint for callers (yt-dlp exits 1 on most errors) but the
/// classification itself is purely stderr-driven and deterministic.
///
/// HTTP status markers are anchored to the full error phrase ("HTTP Error 403",
/// "403 Forbidden", ...) rather than bare digit substrings, so format ids or
/// counts like `404 fragments` never classify as an HTTP failure.
pub fn classify(stderr: &str, exit: Option<i32>) -> ErrorClass {
    let _ = exit;
    let haystack = error_line(stderr).unwrap_or(stderr).to_lowercase();
    let has = |pat: &str| haystack.contains(pat);

    if has("http error 403") || has("403 forbidden") {
        ErrorClass::Permanent(PermanentKind::Http403)
    } else if has("http error 404") || has("404 not found") {
        ErrorClass::Permanent(PermanentKind::Http404)
    } else if has("http error 429") || has("too many requests") {
        ErrorClass::Retryable(RetryableKind::Http429)
    } else if has("5xx") || has("http error 5") {
        ErrorClass::Retryable(RetryableKind::Http5xx)
    } else if has("timed out") || has("timeout") {
        ErrorClass::Retryable(RetryableKind::Timeout)
    } else if has("could not resolve") || has("dns") {
        ErrorClass::Retryable(RetryableKind::Dns)
    } else if (has("connect") || has("connection")) && (has("refused") || has("reset")) {
        ErrorClass::Retryable(RetryableKind::Connect)
    } else if has("sign in") || has("login required") || has("cloudflare") || has("challenge") {
        ErrorClass::Permanent(PermanentKind::AuthChallenge)
    } else if has("unsupported url") {
        ErrorClass::Permanent(PermanentKind::UnsupportedUrl)
    } else if has("no video formats") || has("format not available") || has("requested format") {
        ErrorClass::Permanent(PermanentKind::FormatNotAvailable)
    } else {
        ErrorClass::Unknown
    }
}

// ── Host adapter ───────────────────────────────────────────────────────────

/// Per-host policy: recognition, canonicalization, timeouts, error classes.
#[derive(Debug, Default)]
pub struct HostAdapter;

impl HostAdapter {
    /// Recognize the host behind a URL. `None` for non-http(s) URLs,
    /// `Some(HostId::Generic)` for unknown http(s).
    pub fn recognize(url: &str) -> Option<HostId> {
        let u = url.trim();
        if !(u.starts_with("http://") || u.starts_with("https://")) {
            return None;
        }
        Some(identify_host(u))
    }

    /// Canonicalize a URL (VK conversion; otherwise passthrough).
    pub fn canonicalize(url: &str) -> String {
        canonicalize(url)
    }

    /// Timeout profile for a host.
    pub fn timeout_profile(host: HostId) -> TimeoutProfile {
        timeout_profile_for(host)
    }

    /// Classify a failure from stderr + optional exit code.
    pub fn classify(stderr: &str, exit: Option<i32>) -> ErrorClass {
        classify(stderr, exit)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognize_known_hosts() {
        assert_eq!(
            HostAdapter::recognize("https://cda.pl/video/123"),
            Some(HostId::Cda)
        );
        assert_eq!(
            HostAdapter::recognize("https://ebd.cda.pl/abc"),
            Some(HostId::Cda)
        );
        assert_eq!(
            HostAdapter::recognize("https://rumble.com/embed/abc/"),
            Some(HostId::Rumble)
        );
        assert_eq!(
            HostAdapter::recognize("https://vk.com/video_ext.php?oid=1&id=2"),
            Some(HostId::Vk)
        );
        assert_eq!(
            HostAdapter::recognize("https://vkvideo.ru/video-1_2"),
            Some(HostId::Vk)
        );
        assert_eq!(
            HostAdapter::recognize("https://video.sibnet.ru/v.mp4"),
            Some(HostId::Sibnet)
        );
        assert_eq!(
            HostAdapter::recognize("https://dood.yt/abc"),
            Some(HostId::Dood)
        );
        assert_eq!(
            HostAdapter::recognize("https://hqq.tv/watch/abc"),
            Some(HostId::Hqq)
        );
        assert_eq!(
            HostAdapter::recognize("https://mega.nz/file/abc"),
            Some(HostId::Mega)
        );
        assert_eq!(
            HostAdapter::recognize("https://ok.ru/video/123"),
            Some(HostId::Ok)
        );
        assert_eq!(
            HostAdapter::recognize("https://myvi.tv/abc"),
            Some(HostId::Myvi)
        );
        assert_eq!(
            HostAdapter::recognize("https://drive.google.com/file/d/1"),
            Some(HostId::Google)
        );
        assert_eq!(
            HostAdapter::recognize("https://voe.sx/abc"),
            Some(HostId::Voe)
        );
        assert_eq!(
            HostAdapter::recognize("https://mp4upload.com/abc"),
            Some(HostId::Mp4upload)
        );
    }

    #[test]
    fn recognize_generic_and_rejects_non_http() {
        assert_eq!(
            HostAdapter::recognize("https://cdn1.example.com/v.mp4"),
            Some(HostId::Generic)
        );
        assert_eq!(HostAdapter::recognize("ftp://example.com/v.mp4"), None);
        assert_eq!(HostAdapter::recognize("file:///tmp/v.mp4"), None);
        assert_eq!(HostAdapter::recognize("not a url"), None);
    }

    #[test]
    fn host_rank_cda_first() {
        assert_eq!(host_preference_rank()[0], HostId::Cda);
    }

    #[test]
    fn host_rank_rumble_second() {
        assert_eq!(host_preference_rank()[1], HostId::Rumble);
    }

    #[test]
    fn host_rank_contains_all_known() {
        assert_eq!(all_hosts().len(), 12);
        for host in all_hosts() {
            assert!(host != HostId::Generic);
        }
    }

    #[test]
    fn vk_url_canonicalized() {
        let url = "https://vk.com/video_ext.php?oid=-229809086&id=456239061&hash=abc&hd=2";
        assert_eq!(
            try_canonicalize_vk_url(url).as_deref(),
            Some("https://vkvideo.ru/video-229809086_456239061")
        );
        assert_eq!(try_canonicalize_vk_url("https://vk.com/other"), None);
        assert_eq!(
            try_canonicalize_vk_url("https://example.com/video_ext.php"),
            None
        );
    }

    #[test]
    fn canonicalize_passthrough_trims_slash() {
        assert_eq!(
            canonicalize("https://cda.pl/video/123/"),
            "https://cda.pl/video/123"
        );
        assert_eq!(
            canonicalize("https://example.com/v.mp4"),
            "https://example.com/v.mp4"
        );
        // VK canonicalization is idempotent
        let once = canonicalize("https://vk.com/video_ext.php?oid=1&id=2");
        assert_eq!(canonicalize(&once), once);
    }

    #[test]
    fn classify_network_retryable() {
        assert_eq!(
            classify("ERROR: [generic] Timed out", Some(1)),
            ErrorClass::Retryable(RetryableKind::Timeout)
        );
        assert_eq!(
            classify("Could not resolve: hostname", Some(1)),
            ErrorClass::Retryable(RetryableKind::Dns)
        );
        assert_eq!(
            classify("Connection reset by peer", Some(1)),
            ErrorClass::Retryable(RetryableKind::Connect)
        );
        assert_eq!(
            classify("HTTP Error 429: Too Many Requests", Some(1)),
            ErrorClass::Retryable(RetryableKind::Http429)
        );
        assert_eq!(
            classify("ERROR: HTTP Error 5xx: Server Error", Some(1)),
            ErrorClass::Retryable(RetryableKind::Http5xx)
        );
    }

    #[test]
    fn classify_404_permanent() {
        assert_eq!(
            classify("ERROR: HTTP Error 404: Not Found", Some(1)),
            ErrorClass::Permanent(PermanentKind::Http404)
        );
    }

    #[test]
    fn classify_403_permanent() {
        assert_eq!(
            classify("ERROR: HTTP Error 403: Forbidden", Some(1)),
            ErrorClass::Permanent(PermanentKind::Http403)
        );
    }

    #[test]
    fn classify_429_retryable() {
        assert_eq!(
            classify("ERROR: HTTP Error 429: Too Many Requests", Some(1)),
            ErrorClass::Retryable(RetryableKind::Http429)
        );
    }

    #[test]
    fn classify_auth_and_format_permanent() {
        assert_eq!(
            classify("ERROR: Sign in to view this video", Some(1)),
            ErrorClass::Permanent(PermanentKind::AuthChallenge)
        );
        assert_eq!(
            classify("ERROR: No video formats found", Some(1)),
            ErrorClass::Permanent(PermanentKind::FormatNotAvailable)
        );
    }

    #[test]
    fn classify_default_unknown() {
        assert_eq!(
            classify("ERROR: some weird thing", Some(1)),
            ErrorClass::Unknown
        );
        assert_eq!(classify("", None), ErrorClass::Unknown);
    }

    #[test]
    fn classify_no_false_positive_on_digits() {
        // Bare status-code digits (format ids, counters) must NOT classify as
        // HTTP failures.
        assert_eq!(
            classify(
                "ERROR: [download] 404 fragments of 403 candidates processed",
                Some(1)
            ),
            ErrorClass::Unknown
        );
        assert_eq!(
            classify("ERROR: format id 403 not available, trying 429", Some(1)),
            ErrorClass::Unknown
        );
        assert_eq!(
            classify("ERROR: episode 404 of 1000 skipped", Some(1)),
            ErrorClass::Unknown
        );
        // Anchored phrases still classify.
        assert_eq!(
            classify("ERROR: HTTP Error 403: Forbidden", Some(1)),
            ErrorClass::Permanent(PermanentKind::Http403)
        );
        assert_eq!(
            classify("ERROR: 404 Not Found for /v403.mp4", Some(1)),
            ErrorClass::Permanent(PermanentKind::Http404)
        );
        assert_eq!(
            classify("ERROR: Unsupported URL scheme in 403", Some(1)),
            ErrorClass::Permanent(PermanentKind::UnsupportedUrl)
        );
    }

    #[test]
    fn timeout_profiles() {
        let generic = timeout_profile_for(HostId::Generic);
        assert_eq!(
            generic,
            TimeoutProfile {
                startup_secs: 30,
                stall_secs: 120
            }
        );
        let cda = timeout_profile_for(HostId::Cda);
        assert_eq!(cda.startup_secs, 45);
        let hqq = timeout_profile_for(HostId::Hqq);
        assert_eq!(hqq.stall_secs, 180);
        assert_eq!(timeout_profile_for(HostId::Dood).startup_secs, 60);
        let vk = timeout_profile_for(HostId::Vk);
        assert_eq!(vk.stall_secs, 180);
        assert_eq!(timeout_profile_for(HostId::Rumble).stall_secs, 180);
    }
}
