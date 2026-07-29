//! Dashboard renderer — pinned header/footer with auto-paging episode rows.
//!
//! Columns: episode, state glyph, provider, quality, speed, ETA, progress bar.
//! Responsive: drops columns on narrow terminals.

use std::io;
use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};
use tokio::sync::broadcast;
use tokio::time::sleep;

use crate::download_types::{EpEvent, Phase};
use crate::ui_model::UiModel;
use crate::ui_render::{Renderer, TerminalGuard};

const TICK: std::time::Duration = std::time::Duration::from_millis(100);
const PAGE_CYCLE: std::time::Duration = std::time::Duration::from_secs(3);

pub struct DashboardRenderer {
    rx: broadcast::Receiver<EpEvent>,
    model: UiModel,
}

impl DashboardRenderer {
    pub fn new(rx: broadcast::Receiver<EpEvent>, model: UiModel) -> Self {
        Self { rx, model }
    }

    fn visible_count(area_height: u16) -> usize {
        (area_height.saturating_sub(3) as usize).max(1)
    }

    fn render(model: &UiModel, page_offset: usize, area: Rect, f: &mut Frame) {
        let (title_bar, body, summary_bar) = layout_rects(area);

        let title = Paragraph::new(" anime-dl — download progress ").style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(title, title_bar);

        if model.episodes.is_empty() {
            let empty = Paragraph::new(" No episodes to download")
                .style(Style::default().fg(Color::DarkGray));
            f.render_widget(empty, body);
        } else {
            let (use_compact, use_minimal) = column_mode(area.width);
            render_episodes(f, model, page_offset, body, use_compact, use_minimal);
        }

        render_summary(f, summary_bar, model);
    }
}

fn column_mode(width: u16) -> (bool, bool) {
    if width < 40 {
        (true, true)
    } else if width < 70 {
        (true, false)
    } else {
        (false, false)
    }
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

fn render_episodes(
    f: &mut Frame,
    model: &UiModel,
    page_offset: usize,
    area: Rect,
    compact: bool,
    minimal: bool,
) {
    let max_visible = area.height as usize;
    let mut used = 0usize;
    let mut idx = page_offset;

    while used < max_visible && idx < model.episodes.len() {
        let ep = &model.episodes[idx];
        let h = episode_height(ep, compact);
        if h == 0 {
            idx += 1;
            continue;
        }
        let remaining = max_visible - used;
        let actual_h = h.min(remaining);
        if !compact && actual_h < 2 {
            break;
        }
        if actual_h == 0 {
            break;
        }
        let item_area = Rect::new(area.x, area.y + used as u16, area.width, actual_h as u16);
        render_episode(f, item_area, ep, compact, minimal);
        used += actual_h;
        idx += 1;
    }
}

fn episode_height(ep: &crate::ui_model::EpState, compact: bool) -> usize {
    if compact {
        match ep.phase {
            Phase::Queued => 1,
            Phase::Measuring => 1,
            Phase::Downloading { .. } => 2,
            Phase::Done { .. } | Phase::Failed | Phase::Cancelled => 1,
        }
    } else {
        match ep.phase {
            Phase::Queued => 3,
            Phase::Measuring => 3 + ep.mirrors.len().min(3),
            Phase::Downloading { .. } => 4,
            Phase::Done { .. } | Phase::Failed | Phase::Cancelled => 3,
        }
    }
}

fn phase_glyph(phase: &Phase) -> (&str, Color) {
    match phase {
        Phase::Queued => ("·", Color::DarkGray),
        Phase::Measuring => ("?", Color::Yellow),
        Phase::Downloading { .. } => ("↓", Color::Cyan),
        Phase::Done { .. } => ("✓", Color::Green),
        Phase::Failed => ("✗", Color::Red),
        Phase::Cancelled => ("⨯", Color::DarkGray),
    }
}

fn phase_color(phase: &Phase) -> Color {
    match phase {
        Phase::Queued => Color::DarkGray,
        Phase::Measuring => Color::Yellow,
        Phase::Downloading { .. } => Color::Cyan,
        Phase::Done { .. } => Color::Green,
        Phase::Failed => Color::Red,
        Phase::Cancelled => Color::DarkGray,
    }
}

fn render_episode(
    f: &mut Frame,
    area: Rect,
    ep: &crate::ui_model::EpState,
    compact: bool,
    minimal: bool,
) {
    let color = phase_color(&ep.phase);
    let (glyph, _) = phase_glyph(&ep.phase);

    // In compact or minimal mode, use a one-liner without borders.
    // In normal mode, use full bordered blocks.
    if compact || minimal {
        let line = if let Phase::Downloading { pct, speed, .. } = &ep.phase {
            format!(" {glyph} Ep {:>3} {:>5.1}% {speed}", ep.number, pct)
        } else {
            let winner_str = ep.winner.as_deref().unwrap_or("");
            let quality_str = ep
                .quality
                .as_ref()
                .map(|q| q.to_string())
                .unwrap_or_default();
            format!(" {glyph} Ep {:>3} {winner_str} {quality_str}", ep.number)
        };
        let text = Paragraph::new(line).style(Style::default().fg(color));
        f.render_widget(text, area);
        return;
    }

    let mut title = format!(" {glyph} Ep {} ", ep.number);
    if let Some(ref w) = ep.winner {
        title.push_str(&format!("— {w} "));
    }
    if let Some(ref q) = ep.quality {
        title.push_str(&format!("[{q}] "));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(color));
    let inner = block.inner(area);
    f.render_widget(block, area);

    match &ep.phase {
        Phase::Queued => {
            let text =
                Paragraph::new("  waiting for slot...").style(Style::default().fg(Color::DarkGray));
            f.render_widget(text, inner);
        }
        Phase::Measuring => {
            if ep.mirrors.is_empty() {
                let text = Paragraph::new("  waiting for host locks...")
                    .style(Style::default().fg(Color::DarkGray));
                f.render_widget(text, inner);
            } else {
                use ratatui::widgets::{List, ListItem};
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
            pct, speed, eta, ..
        } => {
            render_gauge(f, inner, *pct, speed, eta, color);
        }
        Phase::Done { host, size_mb } => {
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
            let text = Paragraph::new("  ⨯  cancelled").style(Style::default().fg(Color::DarkGray));
            f.render_widget(text, inner);
        }
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

fn render_summary(f: &mut Frame, area: Rect, model: &UiModel) {
    let total_mb = if model.total_bytes > 0 {
        model.total_bytes as f64 / 1_048_576.0
    } else {
        0.0
    };
    let sum_mb = model.sum_bytes as f64 / 1_048_576.0;

    let active = model.active_count();
    let summary = format!(
        " {} done · {} active · {} failed · Total: {sum_mb:.1} MB / {total_mb:.1} MB",
        model.total_done, active, model.total_failed,
    );
    let text = Paragraph::new(summary).style(Style::default().fg(Color::White).bg(Color::Black));
    f.render_widget(text, area);
}

#[async_trait::async_trait]
impl Renderer for DashboardRenderer {
    async fn run(self: Box<Self>) -> io::Result<()> {
        let mut guard = TerminalGuard::enter()?;

        let mut rx = self.rx;
        let mut model = self.model;
        let mut page_offset = 0usize;
        let mut last_page_cycle = Instant::now();

        loop {
            // Drain events
            while let Ok(ev) = rx.try_recv() {
                model.apply(&ev);
            }

            // Auto-page
            let area = guard
                .terminal()
                .map(|t| t.size().unwrap_or_default())
                .unwrap_or_default();
            let visible = Self::visible_count(area.height);
            if model.episodes.len() > visible {
                let now = Instant::now();
                if now.duration_since(last_page_cycle) >= PAGE_CYCLE {
                    let new_offset = page_offset + visible;
                    if new_offset >= model.episodes.len() {
                        page_offset = 0;
                    } else {
                        page_offset = new_offset;
                    }
                    last_page_cycle = now;
                }
            }

            // Draw
            if let Some(term) = guard.terminal() {
                let offset = page_offset;
                term.draw(|f| Self::render(&model, offset, f.area(), f))?;
            }

            if model.all_terminal() {
                sleep(std::time::Duration::from_millis(500)).await;
                // Final draw
                if let Some(term) = guard.terminal() {
                    let offset = page_offset;
                    term.draw(|f| Self::render(&model, offset, f.area(), f))?;
                }
                break;
            }

            sleep(TICK).await;
        }

        drop(guard); // explicit drop before Ok — restore happens in Drop
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::download_types::EpEvent;

    fn make_model(eps: &[i64]) -> UiModel {
        UiModel::new(eps)
    }

    /// Helper: create a TestBackend terminal and render into it, returning the buffer.
    fn render_to_buffer(model: &UiModel, page_offset: usize, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal =
            Terminal::new(backend).expect("create TestBackend terminal for dashboard test");
        let _ = terminal.draw(|f| {
            DashboardRenderer::render(model, page_offset, f.area(), f);
        });
        let buf = terminal.backend().buffer().clone();
        let mut lines = Vec::new();
        for y in 0..height {
            let mut line = String::new();
            for x in 0..width {
                line.push(buf[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    #[test]
    fn no_input_apis() {
        let src = include_str!("dashboard.rs");
        // Strip test module to avoid self-match on assertion strings
        let src = match src.find("#[cfg(test)]") {
            Some(pos) => &src[..pos],
            None => src,
        };
        assert!(
            !src.contains("enable_raw_mode"),
            "renderer uses enable_raw_mode"
        );
        assert!(!src.contains("event::poll"), "renderer polls for events");
        assert!(!src.contains("event::read"), "renderer reads events");
        assert!(!src.contains("KeyCode"), "renderer references KeyCode");
        assert!(
            !src.contains("crossterm::event"),
            "renderer imports crossterm::event"
        );
    }

    #[test]
    fn startup_empty_model() {
        let model = make_model(&[]);
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("No episodes"), "should show empty message");
        // No ANSI escape sequences in buffer
        assert!(!buf.contains('\x1b'), "no escape sequences in buffer");
    }

    #[test]
    fn startup_queued_episodes() {
        let model = make_model(&[1, 2, 3]);
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("Ep 1"), "episode 1 visible");
        assert!(buf.contains("Ep 2"), "episode 2 visible");
        assert!(buf.contains("Ep 3"), "episode 3 visible");
        assert!(buf.contains("waiting"), "queued indicator");
    }

    #[test]
    fn progress_phase_render() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Winner {
            ep: 1,
            host: "cda.pl".into(),
        });
        model.apply(&EpEvent::Progress {
            ep: 1,
            host: "cda.pl".into(),
            pct: 42.5,
            speed: "5.2 MiB/s".into(),
            eta: "30s".into(),
            downloaded: 4_000_000,
            total: 10_000_000,
        });
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("42.5%"), "percentage shown");
        assert!(
            buf.contains("5.2 MiB/s") || buf.contains("5.2"),
            "speed shown"
        );
    }

    #[test]
    fn done_phase_render() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Done {
            ep: 1,
            host: "cda.pl".into(),
            size_mb: 250.0,
        });
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("✓"), "done glyph");
        assert!(buf.contains("250"), "size shown");
    }

    #[test]
    fn failed_phase_render() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Failed { ep: 1 });
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("✗"), "failed glyph");
    }

    #[test]
    fn narrow_terminal_minimal_mode() {
        let model = make_model(&[1, 2]);
        let buf = render_to_buffer(&model, 0, 39, 24);
        // At < 40 cols, uses minimal mode (one-liner per episode)
        assert!(buf.contains("Ep"), "episode labels visible");
    }

    #[test]
    fn narrow_terminal_compact_mode() {
        let model = make_model(&[1]);
        // At 40-69 cols, compact mode
        let buf = render_to_buffer(&model, 0, 60, 24);
        assert!(buf.contains("Ep"), "episode labels visible");
    }

    #[test]
    fn summary_bar_shows_counts() {
        let mut model = make_model(&[1, 2, 3]);
        model.apply(&EpEvent::Done {
            ep: 1,
            host: "a.pl".into(),
            size_mb: 100.0,
        });
        model.apply(&EpEvent::Failed { ep: 2 });
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("1 done"), "done count");
        assert!(buf.contains("1 failed"), "failed count");
    }

    #[test]
    fn page_offset_shows_different_episodes() {
        let model = make_model(&[1, 2, 3, 4, 5]);
        // With taller terminal, page_offset=2 should show ep3, ep4, ep5
        let buf = render_to_buffer(&model, 2, 80, 20);
        assert!(buf.contains("Ep 3"), "ep3 visible after page");
        assert!(buf.contains("Ep 4"), "ep4 visible after page");
        assert!(buf.contains("Ep 5"), "ep5 visible after page");
    }

    #[test]
    fn no_ansi_escape_codes() {
        let model = make_model(&[1]);
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(!buf.contains('\x1b'), "no ANSI escapes: {buf:?}");
    }
}
