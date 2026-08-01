//! Shared neutral UI model consumed by all fancy renderers.
//!
//! [`UiModel`] is a single reducer over [`EpEvent`] broadcast events.
//! No event-specific rendering logic — purely state aggregation.

use std::time::Instant;

use crate::download_types::{EpEvent, Phase, Quality, parse_speed_bps};

/// Per-episode state for the UI layer.
#[derive(Debug, Clone)]
pub struct EpState {
    pub number: i64,
    pub mirrors: Vec<MirrorInfo>,
    pub winner: Option<String>,
    pub phase: Phase,
    pub quality: Option<Quality>,
    /// Last known host for this episode (winner, waiting mirror, retry mirror).
    pub host: Option<String>,
    /// Transient status line: `waiting host <host> <secs>s`, retry, validation.
    pub status_line: Option<String>,
    /// Failure reason when the episode ended failed (best effort from events).
    pub reason: Option<String>,
}

/// Circuit-breaker state for one host.
#[derive(Debug, Clone)]
pub struct CircuitState {
    pub host: String,
    pub open: bool,
    pub opened_at: Instant,
    pub cooldown_secs: u64,
}

impl CircuitState {
    /// Remaining cooldown seconds while the circuit is open.
    pub fn remaining_secs(&self) -> u64 {
        self.cooldown_secs
            .saturating_sub(self.opened_at.elapsed().as_secs())
    }
}

/// Final run summary: counts + authoritative per-episode failure reasons
/// (engine aggregates them in `Outcome.per_episode_reasons` and sends them
/// with the [`EpEvent::FinalSummary`] event; renderers prefer these over
/// best-effort event-derived reasons).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalSummary {
    pub downloaded: usize,
    pub skipped: usize,
    pub failed: usize,
    pub cancelled: usize,
    /// `(episode, reason)` for every episode that ended without a valid
    /// output (failed / missing / cancelled). Reasons are engine-sanitized;
    /// renderers additionally redact URLs.
    pub per_episode_reasons: Vec<(u32, String)>,
}

/// Lightweight mirror info for display.
#[derive(Debug, Clone)]
pub struct MirrorInfo {
    pub host: String,
    pub bps: Option<f64>,
    pub active: bool,
    pub success: Option<bool>,
}

/// Aggregate model consumed by all renderers.
///
/// Construction: [`UiModel::new`] pre-populates episodes in Queued state.
/// Update: [`UiModel::apply`] handles one [`EpEvent`] at a time.
#[derive(Debug)]
pub struct UiModel {
    pub episodes: Vec<EpState>,
    pub total_done: u64,
    pub total_failed: u64,
    pub started_at: Instant,
    pub events_since_start: u64,
    /// Open/closed circuit states per host.
    pub circuits: Vec<CircuitState>,
    /// Final run summary once the engine sends it.
    pub final_summary: Option<FinalSummary>,
    /// Anime title for the header row.
    pub title: Option<String>,
    /// Circuit cooldown used for countdown rendering.
    pub circuit_cooldown_secs: u64,
}

impl UiModel {
    /// Build model from an ordered list of episode numbers.
    /// All start as [`Phase::Queued`].
    pub fn new(episode_numbers: &[i64]) -> Self {
        Self::new_with_options(episode_numbers, None, 60)
    }

    /// Build model with title and circuit cooldown (used by the dashboard).
    pub fn new_with_options(
        episode_numbers: &[i64],
        title: Option<String>,
        circuit_cooldown_secs: u64,
    ) -> Self {
        Self {
            episodes: episode_numbers
                .iter()
                .map(|&n| EpState {
                    number: n,
                    mirrors: Vec::new(),
                    winner: None,
                    phase: Phase::Queued,
                    quality: None,
                    host: None,
                    status_line: None,
                    reason: None,
                })
                .collect(),
            total_done: 0,
            total_failed: 0,
            started_at: Instant::now(),
            events_since_start: 0,
            circuits: Vec::new(),
            final_summary: None,
            title,
            circuit_cooldown_secs,
        }
    }

    /// Apply one event to the model (reducer pattern).
    pub fn apply(&mut self, ev: &EpEvent) {
        self.events_since_start += 1;
        match *ev {
            EpEvent::Measuring { ep, ref host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && !ep_state.mirrors.iter().any(|m| m.host == *host)
                {
                    ep_state.mirrors.push(MirrorInfo {
                        host: host.clone(),
                        bps: None,
                        active: true,
                        success: None,
                    });
                    if !matches!(ep_state.phase, Phase::Downloading { .. }) {
                        ep_state.phase = Phase::Measuring;
                    }
                }
            }
            EpEvent::Measured { ep, ref host, bps } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && let Some(m) = ep_state.mirrors.iter_mut().find(|m| m.host == *host)
                {
                    m.bps = Some(bps);
                }
            }
            EpEvent::MirrorBusy {
                ep,
                ref host,
                wait_secs,
            } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    if !ep_state.mirrors.iter().any(|m| m.host == *host) {
                        ep_state.mirrors.push(MirrorInfo {
                            host: host.clone(),
                            bps: None,
                            active: false,
                            success: None,
                        });
                    }
                    if !matches!(ep_state.phase, Phase::Downloading { .. }) {
                        ep_state.phase = Phase::WaitingHost;
                    }
                    ep_state.host = Some(host.clone());
                    ep_state.status_line = Some(format!("waiting host {host} {wait_secs}s"));
                }
            }
            EpEvent::Winner { ep, ref host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.winner = Some(host.clone());
                    ep_state.host = Some(host.clone());
                    for m in &mut ep_state.mirrors {
                        if m.host != *host {
                            m.active = false;
                        }
                    }
                    ep_state.phase = Phase::Downloading {
                        pct: 0.0,
                        speed: String::new(),
                        eta: String::new(),
                        downloaded: 0,
                        total: 0,
                    };
                    ep_state.status_line = None;
                }
            }
            EpEvent::Progress {
                ep,
                host: ref _host,
                pct,
                ref speed,
                ref eta,
                downloaded,
                total,
            } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Downloading {
                        pct,
                        speed: speed.clone(),
                        eta: eta.clone(),
                        downloaded,
                        total,
                    };
                }
            }
            EpEvent::Done {
                ep,
                ref host,
                size_mb,
            } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Done {
                        host: host.clone(),
                        size_mb,
                    };
                    ep_state.host = Some(host.clone());
                    ep_state.mirrors.clear();
                    ep_state.status_line = None;
                    ep_state.reason = None;
                    self.total_done += 1;
                }
            }
            EpEvent::Failed { ep } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Failed;
                    ep_state.reason = Some(
                        ep_state
                            .reason
                            .take()
                            .unwrap_or_else(|| "all mirrors failed".into()),
                    );
                    self.total_failed += 1;
                }
            }
            EpEvent::MirrorDone {
                ep,
                ref host,
                success,
            } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && let Some(m) = ep_state.mirrors.iter_mut().find(|m| m.host == *host)
                {
                    m.active = false;
                    m.success = Some(success);
                }
            }
            EpEvent::MeasurementComplete { .. } => {
                // No UI state change needed; winner event follows.
            }
            EpEvent::Cancelled { ep } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Cancelled;
                    ep_state.reason = Some("cancelled".into());
                }
            }
            EpEvent::MirrorMeasFailed { ep, ref host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && !ep_state.mirrors.iter().any(|m| m.host == *host)
                {
                    ep_state.mirrors.push(MirrorInfo {
                        host: host.clone(),
                        bps: None,
                        active: false,
                        success: Some(false),
                    });
                }
            }
            // ── Robustness events (retry/validation/circuit) ──
            EpEvent::RetryWait {
                ep,
                ref mirror,
                attempt,
                backoff_secs,
            } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.host = Some(mirror.clone());
                    ep_state.status_line = Some(format!(
                        "retry {mirror} attempt {attempt} in {backoff_secs}s"
                    ));
                }
            }
            EpEvent::ValidationStarted { ep } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.status_line = Some("validating...".into());
                }
            }
            EpEvent::ValidationResult { ep, ok, ref reason } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    if ok {
                        ep_state.status_line = Some("validation ok".into());
                        ep_state.reason = None;
                    } else {
                        let r = reason.clone().unwrap_or_else(|| "validation failed".into());
                        ep_state.status_line = Some(format!("validation failed: {r}"));
                        ep_state.reason = Some(r);
                    }
                }
            }
            EpEvent::CircuitOpened { ref host } => {
                let cooldown = self.circuit_cooldown_secs;
                self.circuits.retain(|c| c.host != *host);
                self.circuits.push(CircuitState {
                    host: host.clone(),
                    open: true,
                    opened_at: Instant::now(),
                    cooldown_secs: cooldown,
                });
            }
            EpEvent::CircuitClosed { ref host } => {
                if let Some(c) = self.circuits.iter_mut().find(|c| c.host == *host) {
                    c.open = false;
                }
            }
            EpEvent::FinalSummary {
                downloaded,
                skipped,
                failed,
                cancelled,
                ref per_episode_reasons,
            } => {
                self.final_summary = Some(FinalSummary {
                    downloaded,
                    skipped,
                    failed,
                    cancelled,
                    per_episode_reasons: per_episode_reasons.clone(),
                });
            }
        }
    }

    /// True when every episode is in a terminal state.
    pub fn all_terminal(&self) -> bool {
        self.episodes.iter().all(|e| e.phase.is_terminal())
    }

    /// Number of episodes in non-terminal state.
    pub fn active_count(&self) -> usize {
        self.episodes
            .iter()
            .filter(|e| !e.phase.is_terminal())
            .count()
    }

    /// Combined current throughput across downloading episodes, in bytes/s.
    pub fn active_bps(&self) -> f64 {
        self.episodes
            .iter()
            .filter_map(|episode| match &episode.phase {
                Phase::Downloading { speed, .. } => parse_speed_bps(speed),
                _ => None,
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download_types::EpEvent;

    fn make_model(n: &[i64]) -> UiModel {
        UiModel::new(n)
    }

    #[test]
    fn new_model_all_queued() {
        let m = make_model(&[1, 2, 3]);
        assert_eq!(m.episodes.len(), 3);
        assert!(m.episodes.iter().all(|e| e.phase == Phase::Queued));
        assert_eq!(m.total_done, 0);
        assert_eq!(m.total_failed, 0);
    }

    #[test]
    fn apply_measuring_adds_mirror() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        });
        assert_eq!(m.episodes[0].mirrors.len(), 1);
        assert_eq!(m.episodes[0].mirrors[0].host, "cda.pl");
        assert!(m.episodes[0].mirrors[0].active);
        assert_eq!(m.episodes[0].phase, Phase::Measuring);
    }

    #[test]
    fn apply_measured_updates_bps() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        });
        m.apply(&EpEvent::Measured {
            ep: 1,
            host: "cda.pl".into(),
            bps: 5_000_000.0,
        });
        assert_eq!(m.episodes[0].mirrors[0].bps, Some(5_000_000.0));
    }

    #[test]
    fn apply_mirror_busy_adds_inactive_mirror() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::MirrorBusy {
            ep: 1,
            host: "sibnet.ru".into(),
            wait_secs: 0,
        });
        assert_eq!(m.episodes[0].mirrors.len(), 1);
        assert!(!m.episodes[0].mirrors[0].active);
    }

    #[test]
    fn apply_winner_sets_phase_and_deactivates_others() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "a.pl".into(),
        });
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "b.pl".into(),
        });
        m.apply(&EpEvent::Winner {
            ep: 1,
            host: "a.pl".into(),
        });
        assert_eq!(m.episodes[0].winner, Some("a.pl".into()));
        assert!(
            m.episodes[0]
                .mirrors
                .iter()
                .find(|m| m.host == "a.pl")
                .unwrap()
                .active
        );
        assert!(
            !m.episodes[0]
                .mirrors
                .iter()
                .find(|m| m.host == "b.pl")
                .unwrap()
                .active
        );
        assert!(matches!(m.episodes[0].phase, Phase::Downloading { .. }));
    }

    #[test]
    fn apply_progress_updates_download_phase() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Winner {
            ep: 1,
            host: "a.pl".into(),
        });
        m.apply(&EpEvent::Progress {
            ep: 1,
            host: "a.pl".into(),
            pct: 42.5,
            speed: "5.2 MiB/s".into(),
            eta: "30s".into(),
            downloaded: 4_000_000,
            total: 10_000_000,
        });
        if let Phase::Downloading {
            pct, speed, eta, ..
        } = &m.episodes[0].phase
        {
            assert!((*pct - 42.5).abs() < 0.01);
            assert_eq!(speed, "5.2 MiB/s");
            assert_eq!(eta, "30s");
        } else {
            panic!("expected Downloading phase");
        }
    }

    #[test]
    fn active_bps_sums_downloading_episode_speeds() {
        let mut m = make_model(&[1, 2]);
        for (ep, speed) in [(1, "5.2 MiB/s"), (2, "800 KiB/s")] {
            m.apply(&EpEvent::Winner {
                ep,
                host: format!("host-{ep}"),
            });
            m.apply(&EpEvent::Progress {
                ep,
                host: format!("host-{ep}"),
                pct: 10.0,
                speed: speed.into(),
                eta: "1m".into(),
                downloaded: 1,
                total: 10,
            });
        }

        let expected = 5.2 * 1_048_576.0 + 800.0 * 1024.0;
        assert!((m.active_bps() - expected).abs() < 0.1);
    }

    #[test]
    fn apply_done_clears_mirrors_and_counts() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        });
        m.apply(&EpEvent::Winner {
            ep: 1,
            host: "cda.pl".into(),
        });
        m.apply(&EpEvent::Done {
            ep: 1,
            host: "cda.pl".into(),
            size_mb: 250.0,
        });
        assert!(m.episodes[0].mirrors.is_empty());
        assert_eq!(m.total_done, 1);
        assert!(matches!(m.episodes[0].phase, Phase::Done { .. }));
    }

    #[test]
    fn apply_failed_sets_phase() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Failed { ep: 1 });
        assert_eq!(m.episodes[0].phase, Phase::Failed);
        assert_eq!(m.total_failed, 1);
    }

    #[test]
    fn apply_cancelled_sets_phase() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Cancelled { ep: 1 });
        assert_eq!(m.episodes[0].phase, Phase::Cancelled);
    }

    #[test]
    fn apply_mirror_done_deactivates() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        });
        m.apply(&EpEvent::MirrorDone {
            ep: 1,
            host: "cda.pl".into(),
            success: false,
        });
        let mir = &m.episodes[0].mirrors[0];
        assert!(!mir.active);
        assert_eq!(mir.success, Some(false));
    }

    #[test]
    fn all_terminal_true_when_done() {
        let mut m = make_model(&[1, 2]);
        m.apply(&EpEvent::Done {
            ep: 1,
            host: "a.pl".into(),
            size_mb: 100.0,
        });
        m.apply(&EpEvent::Failed { ep: 2 });
        assert!(m.all_terminal());
    }

    #[test]
    fn all_terminal_false_when_active() {
        let mut m = make_model(&[1, 2]);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "a.pl".into(),
        });
        assert!(!m.all_terminal());
    }

    #[test]
    fn active_count_only_non_terminal() {
        let mut m = make_model(&[1, 2, 3]);
        m.apply(&EpEvent::Done {
            ep: 1,
            host: "a.pl".into(),
            size_mb: 100.0,
        });
        m.apply(&EpEvent::Measuring {
            ep: 2,
            host: "a.pl".into(),
        });
        assert_eq!(m.active_count(), 2); // ep2 measuring, ep3 queued
    }

    #[test]
    fn events_since_start_increments() {
        let mut m = make_model(&[1]);
        assert_eq!(m.events_since_start, 0);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "a".into(),
        });
        assert_eq!(m.events_since_start, 1);
        m.apply(&EpEvent::Measured {
            ep: 1,
            host: "a".into(),
            bps: 1.0,
        });
        assert_eq!(m.events_since_start, 2);
    }

    #[test]
    fn duplicate_mirror_not_added() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        });
        m.apply(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        }); // duplicate
        assert_eq!(m.episodes[0].mirrors.len(), 1);
    }

    #[test]
    fn mirror_busy_sets_waiting_host_status() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::MirrorBusy {
            ep: 1,
            host: "sibnet.ru".into(),
            wait_secs: 5,
        });
        assert_eq!(m.episodes[0].phase, Phase::WaitingHost);
        assert_eq!(
            m.episodes[0].status_line.as_deref(),
            Some("waiting host sibnet.ru 5s")
        );
        assert_eq!(m.episodes[0].host.as_deref(), Some("sibnet.ru"));
    }

    #[test]
    fn mirror_busy_does_not_clobber_downloading_phase() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Winner {
            ep: 1,
            host: "a.pl".into(),
        });
        m.apply(&EpEvent::MirrorBusy {
            ep: 1,
            host: "b.pl".into(),
            wait_secs: 10,
        });
        assert!(matches!(m.episodes[0].phase, Phase::Downloading { .. }));
    }

    #[test]
    fn retry_wait_sets_status_line() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::RetryWait {
            ep: 1,
            mirror: "cda.pl".into(),
            attempt: 2,
            backoff_secs: 4,
        });
        assert_eq!(
            m.episodes[0].status_line.as_deref(),
            Some("retry cda.pl attempt 2 in 4s")
        );
    }

    #[test]
    fn validation_result_fail_records_reason() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::ValidationResult {
            ep: 1,
            ok: false,
            reason: Some("too short".into()),
        });
        assert!(
            m.episodes[0]
                .status_line
                .as_deref()
                .unwrap()
                .starts_with("validation failed:")
        );
        assert_eq!(m.episodes[0].reason.as_deref(), Some("too short"));
    }

    #[test]
    fn validation_result_ok_clears_reason() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::ValidationResult {
            ep: 1,
            ok: false,
            reason: Some("too short".into()),
        });
        m.apply(&EpEvent::ValidationResult {
            ep: 1,
            ok: true,
            reason: None,
        });
        assert_eq!(m.episodes[0].reason, None);
        assert_eq!(m.episodes[0].status_line.as_deref(), Some("validation ok"));
    }

    #[test]
    fn circuit_open_and_close_tracked() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::CircuitOpened {
            host: "cda.pl".into(),
        });
        assert_eq!(m.circuits.len(), 1);
        assert!(m.circuits[0].open);
        assert_eq!(m.circuits[0].remaining_secs(), 60);
        m.apply(&EpEvent::CircuitClosed {
            host: "cda.pl".into(),
        });
        assert!(!m.circuits[0].open);
    }

    #[test]
    fn circuit_opened_twice_replaces_state() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::CircuitOpened {
            host: "a.pl".into(),
        });
        m.apply(&EpEvent::CircuitOpened {
            host: "a.pl".into(),
        });
        assert_eq!(m.circuits.len(), 1);
    }

    #[test]
    fn final_summary_stored() {
        let mut m = make_model(&[1, 2]);
        m.apply(&EpEvent::FinalSummary {
            downloaded: 1,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(2, "timeout".into())],
        });
        let s = m.final_summary.clone().expect("summary stored");
        assert_eq!(s.downloaded, 1);
        assert_eq!(s.failed, 1);
        assert_eq!(s.per_episode_reasons, vec![(2, "timeout".into())]);
    }

    #[test]
    fn failed_keeps_validation_reason() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::ValidationResult {
            ep: 1,
            ok: false,
            reason: Some("invalid media: corrupt".into()),
        });
        m.apply(&EpEvent::Failed { ep: 1 });
        assert_eq!(
            m.episodes[0].reason.as_deref(),
            Some("invalid media: corrupt")
        );
    }

    #[test]
    fn failed_defaults_reason() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Failed { ep: 1 });
        assert_eq!(m.episodes[0].reason.as_deref(), Some("all mirrors failed"));
    }

    #[test]
    fn done_clears_status_and_reason() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::ValidationResult {
            ep: 1,
            ok: false,
            reason: Some("bad".into()),
        });
        m.apply(&EpEvent::Done {
            ep: 1,
            host: "a.pl".into(),
            size_mb: 10.0,
        });
        assert_eq!(m.episodes[0].status_line, None);
        assert_eq!(m.episodes[0].reason, None);
        assert_eq!(m.episodes[0].host.as_deref(), Some("a.pl"));
    }
}
