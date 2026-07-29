//! Read-only auto-display TUI for anime downloader.
//!
//! No keyboard/mouse input. Displays all episodes with progress bars, mirrors,
//! and gauge indicators. Automatically pages through episodes if they exceed
//! viewport height. Exits automatically when all episodes reach terminal state.
//! No user interaction — Ctrl+C handled by SIGINT cancellation in engine.

use std::io::{self, Stdout, stdout};
use std::time::Duration;

use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::{
    Frame, Terminal,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
};
use tokio::time::sleep;

pub use crate::download_types::EpEvent;
use crate::download_types::Phase;

// ── Internal UI state ─────────────────────────────────────────────────────

struct Mirror {
    host: String,
    bps: Option<f64>,
    active: bool,
}

struct Episode {
    number: i64,
    mirrors: Vec<Mirror>,
    winner: Option<String>,
    phase: Phase,
}

// ── Public API ────────────────────────────────────────────────────────────

pub struct DownloadUi {
    rx: tokio::sync::broadcast::Receiver<EpEvent>,
    episodes: Vec<Episode>,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    page_offset: usize,
    total_done: u64,
    total_failed: u64,
    total_bytes: u64,
    sum_bytes: u64,
}

impl DownloadUi {
    pub fn new(
        rx: tokio::sync::broadcast::Receiver<EpEvent>,
        episodes: &[i64],
    ) -> io::Result<Self> {
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        let episodes: Vec<Episode> = episodes
            .iter()
            .map(|&e| Episode {
                number: e,
                mirrors: Vec::new(),
                winner: None,
                phase: Phase::Queued,
            })
            .collect();

        Ok(Self {
            rx,
            episodes,
            terminal,
            page_offset: 0,
            total_done: 0,
            total_failed: 0,
            total_bytes: 0,
            sum_bytes: 0,
        })
    }

    pub async fn run(mut self) -> io::Result<()> {
        let tick = Duration::from_millis(100);
        let page_cycle = Duration::from_secs(3);
        let mut last_page_cycle = std::time::Instant::now();

        loop {
            // Drain events
            while let Ok(ev) = self.rx.try_recv() {
                self.handle(ev);
            }

            // Auto-page: cycle page_offset every 3s if total episodes > visible
            let visible = self.visible_count();
            if self.episodes.len() > visible {
                let now = std::time::Instant::now();
                if now.duration_since(last_page_cycle) >= page_cycle {
                    let new_offset = self.page_offset + visible;
                    if new_offset >= self.episodes.len() {
                        self.page_offset = 0;
                    } else {
                        self.page_offset = new_offset;
                    }
                    last_page_cycle = now;
                }
            }

            // Capture state for render (avoids borrow conflict)
            let episodes: Vec<_> = self
                .episodes
                .iter()
                .map(|e| Episode {
                    number: e.number,
                    mirrors: e
                        .mirrors
                        .iter()
                        .map(|m| Mirror {
                            host: m.host.clone(),
                            bps: m.bps,
                            active: m.active,
                        })
                        .collect(),
                    winner: e.winner.clone(),
                    phase: e.phase.clone(),
                })
                .collect();
            let total_done = self.total_done;
            let total_failed = self.total_failed;
            let total_bytes = self.total_bytes;
            let sum_bytes = self.sum_bytes;
            let page_offset = self.page_offset;

            // Redraw
            self.terminal.draw(|f| {
                Self::draw(
                    f,
                    &episodes,
                    page_offset,
                    total_done,
                    total_failed,
                    sum_bytes,
                    total_bytes,
                )
            })?;

            if self.all_terminal() {
                sleep(Duration::from_millis(500)).await;
                break;
            }

            sleep(tick).await;
        }

        // Final redraw — capture state again
        let episodes: Vec<_> = self
            .episodes
            .iter()
            .map(|e| Episode {
                number: e.number,
                mirrors: e
                    .mirrors
                    .iter()
                    .map(|m| Mirror {
                        host: m.host.clone(),
                        bps: m.bps,
                        active: m.active,
                    })
                    .collect(),
                winner: e.winner.clone(),
                phase: e.phase.clone(),
            })
            .collect();
        let total_done = self.total_done;
        let total_failed = self.total_failed;
        let total_bytes = self.total_bytes;
        let sum_bytes = self.sum_bytes;
        let page_offset = self.page_offset;

        self.terminal.draw(|f| {
            Self::draw(
                f,
                &episodes,
                page_offset,
                total_done,
                total_failed,
                sum_bytes,
                total_bytes,
            )
        })?;

        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        Ok(())
    }

    fn all_terminal(&self) -> bool {
        self.episodes.iter().all(|e| e.phase.is_terminal())
    }

    fn visible_count(&self) -> usize {
        let area = self.terminal.size().unwrap_or_default();
        (area.height.saturating_sub(3) as usize).max(1) / 4
    }

    fn handle(&mut self, ev: EpEvent) {
        match ev {
            EpEvent::Measuring { ep, host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && !ep_state.mirrors.iter().any(|m| m.host == host)
                {
                    ep_state.mirrors.push(Mirror {
                        host,
                        bps: None,
                        active: true,
                    });
                    ep_state.phase = Phase::Measuring;
                }
            }
            EpEvent::Measured { ep, host, bps } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && let Some(m) = ep_state.mirrors.iter_mut().find(|m| m.host == host)
                {
                    m.bps = Some(bps);
                }
            }
            EpEvent::MirrorBusy { ep, host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.mirrors.push(Mirror {
                        host,
                        bps: None,
                        active: false,
                    });
                }
            }
            EpEvent::Winner { ep, host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.winner = Some(host);
                    for m in &mut ep_state.mirrors {
                        if m.host != ep_state.winner.as_deref().unwrap_or("") {
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
                host: _,
                pct,
                speed,
                eta,
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
                    self.sum_bytes = downloaded;
                }
            }
            EpEvent::Done { ep, host, size_mb } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Done {
                        host: host.clone(),
                        size_mb,
                    };
                    ep_state.mirrors.clear();
                    self.total_done += 1;
                    self.total_bytes += (size_mb * 1_048_576.0) as u64;
                }
            }
            EpEvent::Failed { ep } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Failed;
                    self.total_failed += 1;
                }
            }
            EpEvent::MirrorDone { ep, host, .. } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep)
                    && let Some(m) = ep_state.mirrors.iter_mut().find(|m| m.host == host)
                {
                    m.active = false;
                }
            }
            EpEvent::MeasurementComplete { .. } => {}
            EpEvent::Cancelled { ep } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == ep) {
                    ep_state.phase = Phase::Cancelled;
                }
            }
        }
    }

    // ── Rendering (static, avoids borrow conflicts) ────────────────

    fn draw(
        f: &mut Frame,
        episodes: &[Episode],
        page_offset: usize,
        total_done: u64,
        total_failed: u64,
        sum_bytes: u64,
        total_bytes: u64,
    ) {
        let area = f.area();
        let (title_bar, body, summary_bar) = Self::layout_rects(area);

        let title = Paragraph::new(" anime-dl — download progress").style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(title, title_bar);

        if episodes.is_empty() {
            let empty = Paragraph::new(" No episodes to download")
                .style(Style::default().fg(Color::DarkGray));
            f.render_widget(empty, body);
        } else {
            Self::render_episodes(f, episodes, page_offset, body);
        }

        Self::render_summary(
            f,
            summary_bar,
            episodes.len(),
            total_done,
            total_failed,
            sum_bytes,
            total_bytes,
        );
    }

    fn layout_rects(area: Rect) -> (Rect, Rect, Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(area);
        (chunks[0], chunks[1], chunks[2])
    }

    fn render_episodes(f: &mut Frame, episodes: &[Episode], page_offset: usize, area: Rect) {
        let max_visible = area.height as usize;
        let mut used = 0usize;
        let mut idx = page_offset;

        while used < max_visible && idx < episodes.len() {
            let ep = &episodes[idx];
            let h = Self::episode_height(ep);
            if h == 0 {
                idx += 1;
                continue;
            }
            let remaining = max_visible - used;
            let actual_h = h.min(remaining);
            if actual_h < 2 {
                break;
            }
            let item_area = Rect::new(area.x, area.y + used as u16, area.width, actual_h as u16);
            Self::render_episode(f, item_area, ep);
            used += actual_h;
            idx += 1;
        }
    }

    fn episode_height(ep: &Episode) -> usize {
        match ep.phase {
            Phase::Queued => 3,
            Phase::Measuring => 3 + ep.mirrors.len().min(3),
            Phase::Downloading { .. } => 4,
            Phase::Done { .. } | Phase::Failed | Phase::Cancelled => 3,
        }
    }

    fn render_episode(f: &mut Frame, area: Rect, ep: &Episode) {
        let (border_color, _title_str) = Self::episode_style(ep);

        let mut title = format!(" Ep {} ", ep.number);
        if let Some(ref w) = ep.winner {
            title.push_str(&format!("— {w} "));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(border_color));
        let inner = block.inner(area);
        f.render_widget(block, area);

        match ep.phase {
            Phase::Queued => {
                let text = Paragraph::new("  waiting for slot...")
                    .style(Style::default().fg(Color::DarkGray));
                f.render_widget(text, inner);
            }
            Phase::Measuring => {
                if ep.mirrors.is_empty() {
                    let text = Paragraph::new("  waiting for host locks...")
                        .style(Style::default().fg(Color::DarkGray));
                    f.render_widget(text, inner);
                } else {
                    let items: Vec<ListItem> = ep
                        .mirrors
                        .iter()
                        .take(3)
                        .map(|m| {
                            let icon = if m.active { "⠿" } else { " " };
                            let speed = match m.bps {
                                Some(bps) => {
                                    let mbps = bps / 1_048_576.0;
                                    format!("{mbps:.1} MiB/s")
                                }
                                None if m.active => "measuring...".into(),
                                None => "skipped".into(),
                            };
                            let color = if m.active {
                                Color::Cyan
                            } else {
                                Color::DarkGray
                            };
                            ListItem::new(format!("  {icon} {:<10} {}", m.host, speed))
                                .style(Style::default().fg(color))
                        })
                        .collect();
                    let list = List::new(items);
                    f.render_widget(list, inner);
                }
            }
            Phase::Downloading {
                pct,
                ref speed,
                ref eta,
                ..
            } => Self::render_gauge(f, inner, pct, speed, eta, border_color),
            Phase::Done { ref host, size_mb } => {
                let msg = format!("  ✓  {size_mb:.1} MB  ({host})");
                let text = Paragraph::new(msg).style(
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                );
                f.render_widget(text, inner);
            }
            Phase::Failed => {
                let text = Paragraph::new("  ✗  all mirrors failed")
                    .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD));
                f.render_widget(text, inner);
            }
            Phase::Cancelled => {
                let text =
                    Paragraph::new("  ✕  cancelled").style(Style::default().fg(Color::DarkGray));
                f.render_widget(text, inner);
            }
        }
    }

    fn episode_style(ep: &Episode) -> (Color, String) {
        match ep.phase {
            Phase::Queued => (Color::DarkGray, "queued".into()),
            Phase::Measuring => (Color::Yellow, "measuring".into()),
            Phase::Downloading { .. } => (Color::Cyan, "downloading".into()),
            Phase::Done { .. } => (Color::Green, "done".into()),
            Phase::Failed => (Color::Red, "failed".into()),
            Phase::Cancelled => (Color::DarkGray, "cancelled".into()),
        }
    }

    fn render_gauge(f: &mut Frame, area: Rect, pct: f64, speed: &str, eta: &str, color: Color) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(area);

        let gauge_color = if pct > 90.0 {
            Color::Green
        } else if pct > 60.0 {
            Color::LightGreen
        } else if pct > 30.0 {
            Color::Yellow
        } else {
            color
        };

        let gauge = Gauge::default()
            .gauge_style(
                Style::default()
                    .fg(gauge_color)
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .percent((pct as u16).min(100))
            .label(format!("{pct:.1}%"));
        f.render_widget(gauge, chunks[0]);

        let info = format!("  {speed:>12}  ETA {eta}");
        let text = Paragraph::new(info).style(Style::default().fg(Color::White));
        f.render_widget(text, chunks[1]);
    }

    fn render_summary(
        f: &mut Frame,
        area: Rect,
        ep_count: usize,
        done: u64,
        failed: u64,
        sum_bytes: u64,
        total_bytes: u64,
    ) {
        let active = ep_count as u64 - done - failed;
        let total_mb = if total_bytes > 0 {
            total_bytes as f64 / 1_048_576.0
        } else {
            0.0
        };
        let sum_mb = sum_bytes as f64 / 1_048_576.0;

        let summary = format!(
            " {done} done · {active} active · {failed} failed · Total: {sum_mb:.1} MB / {total_mb:.1} MB"
        );
        let text =
            Paragraph::new(summary).style(Style::default().fg(Color::White).bg(Color::Black));
        f.render_widget(text, area);
    }
}
