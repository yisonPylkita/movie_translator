//! Ratatui-based multi-episode download progress UI.
//!
//! One panel per episode. During measurement, each mirror's speed is shown.
//! When the fastest mirror is selected, the panel collapses to a single
//! download bar. Completed panels show a green checkmark and file size.
//!
//! Episode threads send [`EpEvent`]s through a [`std::sync::mpsc::Sender`].
//! The TUI thread receives them and redraws.

use std::io::{self, Stdout, stdout};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
};

// ── Event types ───────────────────────────────────────────────────────────

/// Events sent from episode threads to the TUI.
#[derive(Debug, Clone)]
pub enum EpEvent {
    /// Started measuring this mirror.
    Measuring { ep: i64, host: String },
    /// Measurement result: average bytes/sec.
    Measured { ep: i64, host: String, bps: f64 },
    /// Mirror skipped because another episode holds the host lock.
    MirrorBusy { ep: i64, host: String },
    /// A mirror was selected as the winner; full download starting.
    Winner { ep: i64, host: String },
    /// Download progress update from the winning mirror.
    Progress {
        ep: i64,
        host: String,
        pct: f64,
        speed: String,
        eta: String,
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
}

// ── Internal UI state ─────────────────────────────────────────────────────

struct Mirror {
    host: String,
    bps: Option<f64>, // measured speed, if any
    active: bool,     // still running (measuring or downloading)
}

struct Episode {
    number: i64,
    mirrors: Vec<Mirror>,
    winner: Option<String>, // host name of selected mirror
    phase: Phase,
}

enum Phase {
    Measuring,
    Downloading {
        pct: f64,
        speed: String,
        eta: String,
    },
    Done {
        host: String,
        size_mb: f64,
    },
    Failed,
}

// ── Public API ────────────────────────────────────────────────────────────

/// Build a [`DownloadUi`], spawn episode threads, then call [`DownloadUi::run`].
pub struct DownloadUi {
    rx: Receiver<EpEvent>,
    episodes: Vec<Episode>,
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

type CrosstermBackend<W> = ratatui::backend::CrosstermBackend<W>;

impl DownloadUi {
    /// Create the UI. `rx` receives events from all episode threads.
    pub fn new(rx: Receiver<EpEvent>, episodes: &[i64]) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        let episodes: Vec<Episode> = episodes
            .iter()
            .map(|&e| Episode {
                number: e,
                mirrors: Vec::new(),
                winner: None,
                phase: Phase::Measuring,
            })
            .collect();

        Ok(Self {
            rx,
            episodes,
            terminal,
        })
    }

    /// Run the event loop. Blocks until all episodes finish or the user
    /// presses `q` / `Esc`.
    pub fn run(mut self) -> io::Result<()> {
        let tick = Duration::from_millis(100);
        let deadline = Instant::now() + Duration::from_secs(7200); // 2h max
        let mut all_done = false;

        while !all_done && Instant::now() < deadline {
            // Drain pending events
            while let Ok(ev) = self.rx.try_recv() {
                self.handle(ev);
            }

            // Check for keyboard input
            if event::poll(tick)?
                && let Ok(Event::Key(key)) = event::read()
                && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
            {
                break;
            }

            self.terminal.draw(|f| Self::draw(f, &self.episodes))?;

            all_done = self
                .episodes
                .iter()
                .all(|e| matches!(e.phase, Phase::Done { .. } | Phase::Failed));
        }

        // Final draw
        self.terminal.draw(|f| Self::draw(f, &self.episodes))?;

        // Cleanup
        disable_raw_mode()?;
        execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;

        Ok(())
    }

    // ── Event handler ──────────────────────────────────────────────────

    fn handle(&mut self, ev: EpEvent) {
        match &ev {
            EpEvent::Measuring { ep, host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep) {
                    // Don't duplicate
                    if !ep_state.mirrors.iter().any(|m| m.host == *host) {
                        ep_state.mirrors.push(Mirror {
                            host: host.clone(),
                            bps: None,
                            active: true,
                        });
                    }
                }
            }
            EpEvent::Measured { ep, host, bps } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep)
                    && let Some(m) = ep_state.mirrors.iter_mut().find(|m| m.host == *host)
                {
                    m.bps = Some(*bps);
                }
            }
            EpEvent::MirrorBusy { ep, host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep) {
                    ep_state.mirrors.push(Mirror {
                        host: host.clone(),
                        bps: None,
                        active: false,
                    });
                }
            }
            EpEvent::Winner { ep, host } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep) {
                    ep_state.winner = Some(host.clone());
                    // Mark all non-winner mirrors as inactive
                    for m in &mut ep_state.mirrors {
                        if m.host != *host {
                            m.active = false;
                        }
                    }
                    ep_state.phase = Phase::Downloading {
                        pct: 0.0,
                        speed: String::new(),
                        eta: String::new(),
                    };
                }
            }
            EpEvent::Progress {
                ep,
                host: _,
                pct,
                speed,
                eta,
            } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep) {
                    ep_state.phase = Phase::Downloading {
                        pct: *pct,
                        speed: speed.clone(),
                        eta: eta.clone(),
                    };
                }
            }
            EpEvent::Done { ep, host, size_mb } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep) {
                    ep_state.phase = Phase::Done {
                        host: host.clone(),
                        size_mb: *size_mb,
                    };
                    ep_state.mirrors.clear();
                }
            }
            EpEvent::Failed { ep } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep) {
                    ep_state.phase = Phase::Failed;
                }
            }
            EpEvent::MirrorDone { ep, host, .. } => {
                if let Some(ep_state) = self.episodes.iter_mut().find(|e| e.number == *ep)
                    && let Some(m) = ep_state.mirrors.iter_mut().find(|m| m.host == *host)
                {
                    m.active = false;
                }
            }
        }
    }

    // ── Rendering ──────────────────────────────────────────────────────

    fn draw(f: &mut Frame, episodes: &[Episode]) {
        let area = f.area();

        // Title bar
        let title = Paragraph::new("anime-dl — download progress").style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(title, Rect::new(area.x, area.y, area.width, 1));

        // Episodes stacked vertically
        let body = Rect::new(
            area.x,
            area.y + 1,
            area.width,
            area.height.saturating_sub(1),
        );

        let constraints: Vec<Constraint> = episodes
            .iter()
            .map(|ep| {
                let lines = match &ep.phase {
                    Phase::Measuring => 2u16 + ep.mirrors.len().max(1) as u16,
                    Phase::Downloading { .. } => 3,
                    Phase::Done { .. } | Phase::Failed => 3,
                };
                Constraint::Length(lines)
            })
            .collect();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(body);

        for (i, ep) in episodes.iter().enumerate() {
            if i >= chunks.len() {
                break;
            }
            Self::draw_episode(f, chunks[i], ep);
        }
    }

    fn draw_episode(f: &mut Frame, area: Rect, ep: &Episode) {
        match &ep.phase {
            Phase::Measuring => Self::draw_measuring(f, area, ep),
            Phase::Downloading { pct, speed, eta } => {
                Self::draw_downloading(f, area, ep, *pct, speed, eta)
            }
            Phase::Done { host, size_mb } => Self::draw_done(f, area, ep, host, *size_mb),
            Phase::Failed => Self::draw_failed(f, area, ep),
        }
    }

    fn draw_measuring(f: &mut Frame, area: Rect, ep: &Episode) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" Episode {} — measuring ", ep.number))
            .border_style(Style::default().fg(Color::Yellow));
        let inner = block.inner(area);
        f.render_widget(block, area);

        if ep.mirrors.is_empty() {
            let p = Paragraph::new("  waiting for host locks...")
                .style(Style::default().fg(Color::DarkGray));
            f.render_widget(p, inner);
            return;
        }

        let items: Vec<ListItem> = ep
            .mirrors
            .iter()
            .map(|m| {
                let icon = if m.active { "⠿" } else { " " };
                let speed_str = match m.bps {
                    Some(bps) => {
                        let mbps = bps / 1_048_576.0;
                        format!("{mbps:.1} MiB/s")
                    }
                    None if m.active => "measuring...".to_string(),
                    None => "skipped".to_string(),
                };
                let color = if m.active {
                    Color::Cyan
                } else {
                    Color::DarkGray
                };
                let content = format!("  {icon} {:<10} {}", m.host, speed_str);
                ListItem::new(content).style(Style::default().fg(color))
            })
            .collect();

        let list = List::new(items);
        f.render_widget(list, inner);
    }

    fn draw_downloading(f: &mut Frame, area: Rect, ep: &Episode, pct: f64, speed: &str, eta: &str) {
        let host = ep.winner.as_deref().unwrap_or("?");
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" Episode {} — {}", ep.number, host))
            .border_style(Style::default().fg(Color::Cyan));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(inner);

        // Progress bar
        let gauge = Gauge::default()
            .block(Block::default())
            .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
            .percent(pct as u16)
            .label(format!("{pct:.1}%"));
        f.render_widget(gauge, chunks[0]);

        // Speed + ETA
        let info = format!("  {speed:>12}  ETA {eta}");
        let text = Paragraph::new(info).style(Style::default().fg(Color::White));
        f.render_widget(text, chunks[1]);
    }

    fn draw_done(f: &mut Frame, area: Rect, ep: &Episode, host: &str, size_mb: f64) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" Episode {} ", ep.number))
            .border_style(Style::default().fg(Color::Green));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let msg = format!("  ✓  {size_mb:.1} MB  ({host})");
        let text = Paragraph::new(msg).style(
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(text, inner);
    }

    fn draw_failed(f: &mut Frame, area: Rect, ep: &Episode) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" Episode {} ", ep.number))
            .border_style(Style::default().fg(Color::Red));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let text = Paragraph::new("  ✗  all mirrors failed")
            .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD));
        f.render_widget(text, inner);
    }
}
