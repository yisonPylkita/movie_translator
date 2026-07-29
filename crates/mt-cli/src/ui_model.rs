//! Shared neutral UI model consumed by all fancy renderers.
//!
//! [`UiModel`] is a single reducer over [`EpEvent`] broadcast events.
//! No event-specific rendering logic — purely state aggregation.

use std::collections::HashMap;
use std::time::Instant;

use crate::download_types::{EpEvent, Phase, Quality};

/// Per-episode state for the UI layer.
#[derive(Debug, Clone)]
pub struct EpState {
    pub number: i64,
    pub mirrors: Vec<MirrorInfo>,
    pub winner: Option<String>,
    pub phase: Phase,
    pub quality: Option<Quality>,
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
    pub total_bytes: u64,
    pub sum_bytes: u64,
    pub started_at: Instant,
    pub events_since_start: u64,
    /// Per-episode latest downloaded byte count.
    ep_bytes: HashMap<i64, u64>,
}

impl UiModel {
    /// Build model from an ordered list of episode numbers.
    /// All start as [`Phase::Queued`].
    pub fn new(episode_numbers: &[i64]) -> Self {
        Self {
            episodes: episode_numbers
                .iter()
                .map(|&n| EpState {
                    number: n,
                    mirrors: Vec::new(),
                    winner: None,
                    phase: Phase::Queued,
                    quality: None,
                })
                .collect(),
            total_done: 0,
            total_failed: 0,
            total_bytes: 0,
            sum_bytes: 0,
            started_at: Instant::now(),
            events_since_start: 0,
            ep_bytes: HashMap::new(),
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
            EpEvent::MirrorBusy { ep, ref host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && !ep_state.mirrors.iter().any(|m| m.host == *host)
                {
                    ep_state.mirrors.push(MirrorInfo {
                        host: host.clone(),
                        bps: None,
                        active: false,
                        success: None,
                    });
                }
            }
            EpEvent::Winner { ep, ref host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.winner = Some(host.clone());
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
                    self.ep_bytes.insert(ep, downloaded);
                    self.sum_bytes = self.ep_bytes.values().sum();
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
                    ep_state.mirrors.clear();
                    self.total_done += 1;
                    self.total_bytes += (size_mb * 1_048_576.0) as u64;
                    let ep_bytes = (size_mb * 1_048_576.0) as u64;
                    self.ep_bytes.insert(ep, ep_bytes);
                    self.sum_bytes = self.ep_bytes.values().sum();
                }
            }
            EpEvent::Failed { ep } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Failed;
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
                }
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
        assert!(m.total_bytes > 0);
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
}
