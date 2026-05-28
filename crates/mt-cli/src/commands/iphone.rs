//! `iphone` subcommand: remux MKV -> iPhone-compatible MP4 in place.
//!
//! Port of `movie_translator/commands/iphone_cmd.py` plus the
//! `movie_translator/iphone/converter.py` module it depends on (ported inline
//! here, behavior-preserving).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::Parser;
use mt_discovery::find_videos;
use mt_media::{get_ffmpeg, get_video_info};
use tokio::sync::Semaphore;

const SUPPORTED_VIDEO_CODECS: &[&str] = &["h264", "hevc"];
const CONVERTING_MARKER: &str = ".converting";

/// iPhone command arguments, mirroring `iphone_cmd.parse_args`.
#[derive(Debug, Parser)]
#[command(
    name = "movie-translator iphone",
    about = "Remux MKV -> iPhone-compatible MP4 in place. Stream-copy only (no transcode)."
)]
pub struct IphoneArgs {
    /// Video file or directory of MKVs.
    pub input: String,

    /// Concurrent conversions (default: auto, min(files, 4)).
    #[arg(long, default_value_t = 0)]
    pub workers: u32,

    /// Probe and skip-check only.
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub verbose: bool,
}

// ── converter port ─────────────────────────────────────────────────────────

fn converting_temp_path(mkv_path: &Path) -> PathBuf {
    let stem = mkv_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    mkv_path.with_file_name(format!("{stem}{CONVERTING_MARKER}.mp4"))
}

fn target_mp4_path(mkv_path: &Path) -> PathBuf {
    mkv_path.with_extension("mp4")
}

struct Probe {
    video_codec: String,
    has_jpn_audio: bool,
    has_pol_subs: bool,
}

fn probe(mkv_path: &Path) -> Result<Probe> {
    let info =
        get_video_info(mkv_path).with_context(|| format!("probing {}", mkv_path.display()))?;
    let mut video_codec = String::new();
    let mut has_jpn = false;
    let mut has_pol = false;
    for s in &info.streams {
        match s.codec_type.as_deref() {
            Some("video") if video_codec.is_empty() => {
                video_codec = s.codec_name.clone().unwrap_or_default();
            }
            Some("audio") if s.tags.get("language").map(String::as_str) == Some("jpn") => {
                has_jpn = true;
            }
            Some("subtitle") if s.tags.get("language").map(String::as_str) == Some("pol") => {
                has_pol = true;
            }
            _ => {}
        }
    }
    Ok(Probe {
        video_codec,
        has_jpn_audio: has_jpn,
        has_pol_subs: has_pol,
    })
}

/// Returns `Some(detail)` skip reason or `None` to proceed. Port of `should_skip`.
fn should_skip(mkv_path: &Path, p: &Probe) -> Option<String> {
    let target = target_mp4_path(mkv_path);
    if target.exists() {
        return Some(format!(
            "{} already exists",
            target
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default()
        ));
    }
    if !SUPPORTED_VIDEO_CODECS.contains(&p.video_codec.as_str()) {
        return Some(format!(
            "video codec {:?} not stream-copyable into iPhone-compatible MP4",
            p.video_codec
        ));
    }
    if !p.has_jpn_audio {
        return Some("no audio track with language=jpn".to_string());
    }
    None
}

fn build_ffmpeg_cmd(mkv_path: &Path, temp_path: &Path, include_subs: bool) -> Result<Vec<String>> {
    let ffmpeg = get_ffmpeg().context("locating ffmpeg")?;
    let mut cmd: Vec<String> = vec![
        ffmpeg.to_string_lossy().to_string(),
        "-y".into(),
        "-hide_banner".into(),
        "-nostdin".into(),
        "-i".into(),
        mkv_path.to_string_lossy().to_string(),
        "-map".into(),
        "0:v:0".into(),
        "-map".into(),
        "0:a:m:language:jpn".into(),
    ];
    if include_subs {
        cmd.push("-map".into());
        cmd.push("0:s:m:language:pol".into());
    }
    cmd.extend(["-c:v".into(), "copy".into(), "-c:a".into(), "copy".into()]);
    if include_subs {
        cmd.extend(["-c:s".into(), "mov_text".into()]);
    }
    cmd.extend([
        "-movflags".into(),
        "+faststart".into(),
        temp_path.to_string_lossy().to_string(),
    ]);
    Ok(cmd)
}

fn parse_hms(value: &str) -> Option<f64> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let s: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

fn stream_duration(stream: &mt_media::FfprobeStream) -> Option<f64> {
    if let Some(raw) = &stream.duration {
        if let Ok(d) = raw.parse::<f64>() {
            return Some(d);
        }
    }
    stream.tags.get("DURATION").and_then(|t| parse_hms(t))
}

/// Duration of the video content, in seconds. Port of `video_duration`.
fn video_duration(info: &mt_media::VideoInfo) -> f64 {
    for s in &info.streams {
        if s.codec_type.as_deref() == Some("video") {
            if let Some(d) = stream_duration(s) {
                return d;
            }
        }
    }
    info.format_duration
        .as_ref()
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn verify_output(src: &Path, dst: &Path) -> Result<()> {
    let tolerance = 1.0_f64;
    let src_info =
        get_video_info(src).with_context(|| format!("probing source {}", src.display()))?;
    let dst_info =
        get_video_info(dst).with_context(|| format!("probing output {}", dst.display()))?;

    let src_dur = video_duration(&src_info);
    let dst_dur = video_duration(&dst_info);
    if src_dur > 0.0 && (src_dur - dst_dur).abs() > tolerance {
        bail!("duration mismatch: src={src_dur:.2}s dst={dst_dur:.2}s");
    }
    if !dst_info
        .streams
        .iter()
        .any(|s| s.codec_type.as_deref() == Some("video"))
    {
        bail!("output has no video stream");
    }
    if !dst_info
        .streams
        .iter()
        .any(|s| s.codec_type.as_deref() == Some("audio"))
    {
        bail!("output has no audio stream");
    }
    Ok(())
}

/// Convert one MKV -> MP4 in place. Port of `convert_file`.
/// Returns `(status, detail)`.
fn convert_file(mkv_path: &Path, dry_run: bool) -> (String, String) {
    let p = match probe(mkv_path) {
        Ok(p) => p,
        Err(e) => return ("failed".into(), format!("{e:#}")),
    };
    if let Some(detail) = should_skip(mkv_path, &p) {
        return ("skipped".into(), detail);
    }
    if dry_run {
        return (
            "dry_run".into(),
            format!("would convert (subs={})", p.has_pol_subs),
        );
    }

    let temp = converting_temp_path(mkv_path);
    let target = target_mp4_path(mkv_path);

    let cmd = match build_ffmpeg_cmd(mkv_path, &temp, p.has_pol_subs) {
        Ok(c) => c,
        Err(e) => return ("failed".into(), format!("{e:#}")),
    };

    let result = (|| -> Result<()> {
        let out = Command::new(&cmd[0])
            .args(&cmd[1..])
            .output()
            .context("spawning ffmpeg")?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail: Vec<&str> = stderr
                .trim()
                .lines()
                .rev()
                .take(5)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            bail!("ffmpeg failed: {}", tail.join(" | "));
        }
        verify_output(mkv_path, &temp).context("verifying remuxed output")?;
        std::fs::rename(&temp, &target)
            .with_context(|| format!("renaming {} -> {}", temp.display(), target.display()))?;
        std::fs::remove_file(mkv_path)
            .with_context(|| format!("removing source {}", mkv_path.display()))?;
        Ok(())
    })();

    match result {
        Ok(()) => ("success".into(), String::new()),
        Err(e) => {
            if temp.exists() {
                let _ = std::fs::remove_file(&temp);
            }
            ("failed".into(), format!("{e:#}"))
        }
    }
}

/// All `.mkv` files under `input_path`. Port of `find_mkvs`.
fn find_mkvs(input_path: &Path) -> Vec<PathBuf> {
    find_videos(input_path)
        .into_iter()
        .filter(|v| {
            v.extension()
                .map(|e| e.eq_ignore_ascii_case("mkv"))
                .unwrap_or(false)
        })
        .collect()
}

/// Delete stale `<stem>.converting.mp4` files. Port of `cleanup_orphans`.
fn cleanup_orphans(mkv_files: &[PathBuf]) -> usize {
    let mut removed = 0;
    for mkv in mkv_files {
        let orphan = converting_temp_path(mkv);
        if orphan.exists() && std::fs::remove_file(&orphan).is_ok() {
            removed += 1;
        }
    }
    removed
}

// ── command driver ───────────────────────────────────────────────────────────

async fn run_conversions(
    mkv_files: Vec<PathBuf>,
    workers: u32,
    dry_run: bool,
) -> Vec<(PathBuf, String, String)> {
    let sem = std::sync::Arc::new(Semaphore::new(workers.max(1) as usize));
    let mut joins = Vec::new();
    for mkv in mkv_files {
        let sem = sem.clone();
        joins.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.expect("semaphore");
            eprintln!(
                "-> {}",
                mkv.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            );
            let m = mkv.clone();
            let (status, detail) = tokio::task::spawn_blocking(move || convert_file(&m, dry_run))
                .await
                .unwrap_or_else(|e| ("failed".into(), format!("task panicked: {e}")));
            let tag = match status.as_str() {
                "success" => "✓",
                "dry_run" => "·",
                "skipped" => "⏭",
                "failed" => "✗",
                _ => "?",
            };
            let suffix = if detail.is_empty() {
                String::new()
            } else {
                format!(" — {detail}")
            };
            eprintln!(
                "{tag} {}{suffix}",
                mkv.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            );
            (mkv, status, detail)
        }));
    }
    let mut results = Vec::new();
    for j in joins {
        if let Ok(r) = j.await {
            results.push(r);
        }
    }
    results
}

/// Print summary; return `(success, skipped, failed)`. Port of `_print_summary`.
fn print_summary(results: &[(PathBuf, String, String)]) -> (usize, usize, usize) {
    let count = |k: &str| results.iter().filter(|(_, s, _)| s == k).count();
    let succ = count("success");
    let skip = count("skipped");
    let fail = count("failed");
    let dryr = count("dry_run");

    let mut parts = Vec::new();
    if succ > 0 {
        parts.push(format!("✓ {succ} converted"));
    }
    if dryr > 0 {
        parts.push(format!("· {dryr} would convert"));
    }
    if skip > 0 {
        parts.push(format!("⏭ {skip} skipped"));
    }
    if fail > 0 {
        parts.push(format!("✗ {fail} failed"));
    }
    let line = if parts.is_empty() {
        "(no files)".to_string()
    } else {
        parts.join(" | ")
    };
    println!("{line}");
    (succ, skip, fail)
}

/// Run the iphone flow. Returns the deliberate process exit code as `Ok(code)`,
/// or an `Err` carrying a `.context()` chain for a genuine failure (printed by
/// `main`). Port of `iphone_cmd.run`.
pub async fn run(args: IphoneArgs) -> Result<i32> {
    crate::init_tracing(args.verbose);

    let input_path = PathBuf::from(&args.input);
    if !input_path.exists() {
        eprintln!("Not found: {}", input_path.display());
        return Ok(1);
    }

    let mkv_files = find_mkvs(&input_path);
    if mkv_files.is_empty() {
        eprintln!("No .mkv files found in {}", input_path.display());
        return Ok(if input_path.is_dir() { 0 } else { 1 });
    }

    let removed = cleanup_orphans(&mkv_files);
    if removed > 0 {
        eprintln!("Cleaned up {removed} orphan .converting.mp4 file(s) from a prior run");
    }

    let workers = if args.workers > 0 {
        args.workers
    } else {
        (mkv_files.len().max(1) as u32).min(4)
    };

    eprintln!(
        "Converting {} file(s) -> MP4 in place (workers={workers})",
        mkv_files.len()
    );
    let results = run_conversions(mkv_files, workers, args.dry_run).await;
    let (_, _, fail) = print_summary(&results);

    Ok(if fail > 0 { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> IphoneArgs {
        let mut full = vec!["iphone"];
        full.extend_from_slice(argv);
        IphoneArgs::try_parse_from(full).expect("parse")
    }

    #[test]
    fn defaults_match_python() {
        let args = parse(&["/movies"]);
        assert_eq!(args.input, "/movies");
        assert_eq!(args.workers, 0);
        assert!(!args.dry_run);
        assert!(!args.verbose);
    }

    #[test]
    fn dry_run_and_workers_flags() {
        let args = parse(&["/movies", "--dry-run", "--workers", "2"]);
        assert!(args.dry_run);
        assert_eq!(args.workers, 2);
    }

    #[test]
    fn temp_and_target_paths() {
        let mkv = Path::new("/movies/Show.S01E01.mkv");
        assert_eq!(
            converting_temp_path(mkv),
            Path::new("/movies/Show.S01E01.converting.mp4")
        );
        assert_eq!(target_mp4_path(mkv), Path::new("/movies/Show.S01E01.mp4"));
    }

    #[test]
    fn parse_hms_round_trip() {
        assert_eq!(parse_hms("01:02:03.5"), Some(3723.5));
        assert_eq!(parse_hms("garbage"), None);
    }

    #[test]
    fn summary_counts() {
        let results = vec![
            (PathBuf::from("a.mkv"), "success".to_string(), String::new()),
            (
                PathBuf::from("b.mkv"),
                "skipped".to_string(),
                "x".to_string(),
            ),
            (
                PathBuf::from("c.mkv"),
                "failed".to_string(),
                "boom".to_string(),
            ),
        ];
        let (s, sk, f) = print_summary(&results);
        assert_eq!((s, sk, f), (1, 1, 1));
    }
}
