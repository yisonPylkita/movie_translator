//! Plain text output renderer for anime downloader.
//!
//! Writes pipe-safe, rate-limited progress lines to stdout when stdout is
//! not a TTY or when `--ui plain` is specified. No ANSI escape sequences,
//! no cursor control. Produces machine-readable timestamps and log levels.
//! URLs are redacted everywhere (`<url>`); output stays automation-safe.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::download_types::EpEvent;

const PROGRESS_COOLDOWN: Duration = Duration::from_secs(1);

/// Format one per-episode final-summary line: `E<NN> <host-or-mirror>: <reason>`.
/// Host falls back to `?` when no host was observed; reason is URL-redacted
/// (the emit choke point redacts again — defense in depth).
fn final_summary_reason_lines(
    per_episode_reasons: &[(u32, String)],
    last_host: &HashMap<i64, String>,
) -> Vec<String> {
    per_episode_reasons
        .iter()
        .map(|(ep, reason)| {
            let host = last_host
                .get(&(*ep as i64))
                .map(String::as_str)
                .unwrap_or("?");
            format!("E{ep:02} {host}: {}", redact_urls(reason))
        })
        .collect()
}

/// Spawn a plain output consumer task that reads from the event broadcast
/// and writes structured lines to stdout.
///
/// Returns a `JoinHandle` that can be awaited for completion, plus a
/// shutdown signal sender to stop the consumer early.
pub fn spawn_plain_output(
    mut rx: broadcast::Receiver<EpEvent>,
    _total_episodes: usize,
) -> JoinHandle<(u32, u32)> {
    tokio::spawn(async move {
        let mut done = 0u32;
        let mut failed = 0u32;
        let mut last_progress: HashMap<i64, Instant> = HashMap::new();
        // Last observed host/mirror per episode (for final E<NN> lines).
        let mut last_host: HashMap<i64, String> = HashMap::new();

        loop {
            match rx.recv().await {
                Ok(event) => {
                    let now = Instant::now();
                    match event {
                        EpEvent::Measuring { ep, host } => {
                            emit(format!(
                                "[{} INFO ep={}] Measuring mirrors ({})",
                                iso_timestamp(),
                                ep,
                                host
                            ));
                        }
                        EpEvent::Measured { ep, host, bps } => {
                            let mbps = bps / 1_048_576.0;
                            emit(format!(
                                "[{} INFO ep={}] {} measured at {:.1} MiB/s",
                                iso_timestamp(),
                                ep,
                                host,
                                mbps
                            ));
                        }
                        EpEvent::MirrorBusy {
                            ep,
                            host,
                            wait_secs,
                        } => {
                            last_host.insert(ep, host.clone());
                            emit(format!(
                                "[{} INFO ep={}] waiting host {host} {wait_secs}s",
                                iso_timestamp(),
                                ep,
                            ));
                        }
                        EpEvent::MirrorMeasFailed { ep, host } => {
                            emit(format!(
                                "[{} WARN ep={}] {} measurement failed (no output file)",
                                iso_timestamp(),
                                ep,
                                host
                            ));
                        }
                        EpEvent::Winner { ep, host } => {
                            last_host.insert(ep, host.clone());
                            emit(format!(
                                "[{} INFO ep={}] Winner: {}",
                                iso_timestamp(),
                                ep,
                                host
                            ));
                        }
                        EpEvent::Progress {
                            ep,
                            host,
                            pct,
                            speed,
                            eta,
                            downloaded,
                            total,
                        } => {
                            let last = last_progress.get(&ep).copied().unwrap_or(now);
                            if now.duration_since(last) < PROGRESS_COOLDOWN && pct < 99.9 {
                                continue; // Rate-limit: one progress per ep per second
                            }
                            last_progress.insert(ep, now);

                            let downloaded_mb = downloaded as f64 / 1_048_576.0;
                            let total_mb = total as f64 / 1_048_576.0;
                            emit(format!(
                                "[{} INFO ep={}] Downloading: {:.1}% {} ETA {} {:.1}/{:.1} MB ({})",
                                iso_timestamp(),
                                ep,
                                pct,
                                speed,
                                eta,
                                downloaded_mb,
                                total_mb,
                                host
                            ));
                        }
                        EpEvent::MeasurementComplete { ep } => {
                            emit(format!(
                                "[{} INFO ep={}] Measurement complete, selecting winner",
                                iso_timestamp(),
                                ep
                            ));
                        }
                        EpEvent::Done { ep, host, size_mb } => {
                            done += 1;
                            emit(format!(
                                "[{} INFO ep={}] Done: {} {:.1} MB",
                                iso_timestamp(),
                                ep,
                                host,
                                size_mb
                            ));
                        }
                        EpEvent::Failed { ep } => {
                            failed += 1;
                            emit(format!(
                                "[{} WARN ep={}] Failed: all mirrors exhausted",
                                iso_timestamp(),
                                ep
                            ));
                        }
                        EpEvent::MirrorDone { ep, host, success } => {
                            let status = if success { "done" } else { "failed" };
                            emit(format!(
                                "[{} INFO ep={}] Mirror {}: {}",
                                iso_timestamp(),
                                ep,
                                host,
                                status
                            ));
                        }
                        EpEvent::Cancelled { ep } => {
                            emit(format!("[{} WARN ep={}] Cancelled", iso_timestamp(), ep));
                        }
                        EpEvent::RetryWait {
                            ep,
                            mirror,
                            attempt,
                            backoff_secs,
                        } => {
                            last_host.insert(ep, mirror.clone());
                            emit(format!(
                                "[{} WARN ep={}] retry {mirror} attempt {attempt} in {backoff_secs}s",
                                iso_timestamp(),
                                ep
                            ));
                        }
                        EpEvent::ValidationStarted { ep } => {
                            emit(format!(
                                "[{} INFO ep={}] validating output with ffprobe",
                                iso_timestamp(),
                                ep
                            ));
                        }
                        EpEvent::ValidationResult { ep, ok, reason } => match reason {
                            Some(r) => emit(format!(
                                "[{} {} ep={}] validation {}",
                                iso_timestamp(),
                                if ok { "INFO" } else { "WARN" },
                                ep,
                                r
                            )),
                            None => emit(format!(
                                "[{} INFO ep={}] validation ok",
                                iso_timestamp(),
                                ep
                            )),
                        },
                        EpEvent::CircuitOpened { host } => {
                            emit(format!(
                                "[{} WARN] circuit opened for {host}",
                                iso_timestamp()
                            ));
                        }
                        EpEvent::CircuitClosed { host } => {
                            emit(format!(
                                "[{} INFO] circuit closed for {host}",
                                iso_timestamp()
                            ));
                        }
                        EpEvent::FinalSummary {
                            downloaded,
                            skipped,
                            failed,
                            cancelled,
                            per_episode_reasons,
                        } => {
                            emit(format!(
                                "[{} DONE] downloaded={downloaded} skipped={skipped} failed={failed} cancelled={cancelled}",
                                iso_timestamp()
                            ));
                            for line in final_summary_reason_lines(&per_episode_reasons, &last_host)
                            {
                                emit(line);
                            }
                        }
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    eprintln!("[{} WARN] Lagged by {} events", iso_timestamp(), n);
                    continue;
                }
            }
        }

        let total = done + failed;
        emit(format!(
            "[{} INFO] Done: {} downloaded, {} failed, {} total",
            iso_timestamp(),
            done,
            failed,
            total
        ));

        (done, failed)
    })
}

/// Print one output line with URLs redacted (contract: never render URLs).
fn emit(line: String) {
    println!("{}", redact_urls(&line));
}

pub(crate) fn iso_timestamp() -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{ms}")
}

/// Redact URLs from a string (contract: URLs never rendered in any output).
/// Replaces `http(s)://...` runs with `<url>`.
pub(crate) fn redact_urls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find("http://").or_else(|| rest.find("https://")) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];
        let scheme_len = if after.starts_with("https://") { 8 } else { 7 };
        let url_rest = &after[scheme_len..];
        let end = url_rest
            .find(|c: char| c.is_whitespace())
            .unwrap_or(url_rest.len());
        out.push_str("<url>");
        rest = &url_rest[end..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download_types::EpEvent;

    #[tokio::test]
    async fn plain_output_counts_correctly() {
        let (tx, rx) = broadcast::channel(16);
        let handle = spawn_plain_output(rx, 2);

        tx.send(EpEvent::Done {
            ep: 1,
            host: "x".into(),
            size_mb: 1.0,
        })
        .expect("send Done event to plain output");
        tx.send(EpEvent::Failed { ep: 2 })
            .expect("send Failed event to plain output");
        drop(tx);

        let (done, failed) = handle.await.expect("plain output task completed");
        assert_eq!(done, 1);
        assert_eq!(failed, 1);
    }

    #[test]
    fn redacts_http_and_https_urls() {
        assert_eq!(
            redact_urls("retry https://cda.pl/video/abc attempt 1 in 2s"),
            "retry <url> attempt 1 in 2s"
        );
        assert_eq!(
            redact_urls("host http://x.y/z trailing"),
            "host <url> trailing"
        );
        assert_eq!(redact_urls("no url here"), "no url here");
    }

    #[test]
    fn final_summary_reason_lines_formatted() {
        let mut last_host = HashMap::new();
        last_host.insert(1i64, "cda.pl".to_string());
        let lines = final_summary_reason_lines(
            &[
                (1, "timeout".into()),
                (2, "http 403 https://cdn.example.com/v?token=SECRET".into()),
            ],
            &last_host,
        );
        assert_eq!(lines[0], "E01 cda.pl: timeout");
        assert!(
            lines[1].starts_with("E02 ?: http 403 ") && !lines[1].contains("https://"),
            "reason redacted, host fallback: {}",
            lines[1]
        );
        assert!(!lines[1].contains("SECRET"), "token redacted: {}", lines[1]);
    }

    #[tokio::test]
    async fn final_summary_includes_per_episode_reasons() {
        let (tx, rx) = broadcast::channel::<EpEvent>(16);
        let handle = spawn_plain_output(rx, 1);

        tx.send(EpEvent::FinalSummary {
            downloaded: 0,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(1, "timeout".into())],
        })
        .expect("send FinalSummary");
        drop(tx);

        let (done, failed) = handle.await.expect("plain output task completed");
        assert_eq!(done, 0);
        assert_eq!(failed, 0, "FinalSummary does not alter done/failed counts");
    }

    #[tokio::test]
    async fn plain_output_lines_have_no_urls() {
        let (tx, rx) = broadcast::channel(16);
        let handle = spawn_plain_output(rx, 1);

        tx.send(EpEvent::RetryWait {
            ep: 1,
            mirror: "https://cda.pl/video/abc".into(),
            attempt: 1,
            backoff_secs: 2,
        })
        .expect("send RetryWait");
        tx.send(EpEvent::ValidationResult {
            ep: 1,
            ok: false,
            reason: Some("bad https://x.y/z".into()),
        })
        .expect("send ValidationResult");
        drop(tx);

        handle.await.expect("plain output task completed");
    }
}
