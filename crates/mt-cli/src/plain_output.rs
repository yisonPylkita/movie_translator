//! Plain text output renderer for anime downloader.
//!
//! Writes pipe-safe, rate-limited progress lines to stdout when stdout is
//! not a TTY or when `--ui plain` is specified. No ANSI escape sequences,
//! no cursor control. Produces machine-readable timestamps and log levels.

use std::time::{Duration, Instant};

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::download_types::EpEvent;

const PROGRESS_COOLDOWN: Duration = Duration::from_secs(1);

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
        let mut last_progress: std::collections::HashMap<i64, Instant> =
            std::collections::HashMap::new();

        loop {
            match rx.recv().await {
                Ok(event) => {
                    let now = Instant::now();
                    match event {
                        EpEvent::Measuring { ep, host } => {
                            println!(
                                "[{} INFO ep={}] Measuring mirrors ({})",
                                iso_timestamp(),
                                ep,
                                host
                            );
                        }
                        EpEvent::Measured { ep, host, bps } => {
                            let mbps = bps / 1_048_576.0;
                            println!(
                                "[{} INFO ep={}] {} measured at {:.1} MiB/s",
                                iso_timestamp(),
                                ep,
                                host,
                                mbps
                            );
                        }
                        EpEvent::MirrorBusy { ep, host } => {
                            println!(
                                "[{} INFO ep={}] {} busy (locked by another episode)",
                                iso_timestamp(),
                                ep,
                                host
                            );
                        }
                        EpEvent::Winner { ep, host } => {
                            println!("[{} INFO ep={}] Winner: {}", iso_timestamp(), ep, host);
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
                            println!(
                                "[{} INFO ep={}] Downloading: {:.1}% {} ETA {} {:.1}/{:.1} MB ({})",
                                iso_timestamp(),
                                ep,
                                pct,
                                speed,
                                eta,
                                downloaded_mb,
                                total_mb,
                                host
                            );
                        }
                        EpEvent::MeasurementComplete { ep } => {
                            println!(
                                "[{} INFO ep={}] Measurement complete, selecting winner",
                                iso_timestamp(),
                                ep
                            );
                        }
                        EpEvent::Done { ep, host, size_mb } => {
                            done += 1;
                            println!(
                                "[{} INFO ep={}] Done: {} {:.1} MB",
                                iso_timestamp(),
                                ep,
                                host,
                                size_mb
                            );
                        }
                        EpEvent::Failed { ep } => {
                            failed += 1;
                            println!(
                                "[{} WARN ep={}] Failed: all mirrors exhausted",
                                iso_timestamp(),
                                ep
                            );
                        }
                        EpEvent::MirrorDone { ep, host, success } => {
                            let status = if success { "done" } else { "failed" };
                            println!(
                                "[{} INFO ep={}] Mirror {}: {}",
                                iso_timestamp(),
                                ep,
                                host,
                                status
                            );
                        }
                        EpEvent::Cancelled { ep } => {
                            println!("[{} WARN ep={}] Cancelled", iso_timestamp(), ep);
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
        println!(
            "[{} INFO] Done: {} downloaded, {} failed, {} total",
            iso_timestamp(),
            done,
            failed,
            total
        );

        (done, failed)
    })
}

pub(crate) fn iso_timestamp() -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{ms}")
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
}
