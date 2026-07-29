//! Scoreboard renderer — compact multi-column grid (3-4 cols depending on width).
//!
//! Auto-pages ~4s. Active/failing episodes float to top. Each cell: ep number,
//! 1-char state icon, mini progress bar. Tiny terminal fallback (single-column
//! mode if width < 40).

use std::io;
use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::Paragraph;
use tokio::sync::broadcast;
use tokio::time::sleep;

use crate::download_types::{EpEvent, Phase};
use crate::ui_model::UiModel;
use crate::ui_render::{Renderer, TerminalGuard};

const TICK: std::time::Duration = std::time::Duration::from_millis(100);
const PAGE_CYCLE: std::time::Duration = std::time::Duration::from_secs(4);

pub struct ScoreboardRenderer {
    rx: broadcast::Receiver<EpEvent>,
    model: UiModel,
}

impl ScoreboardRenderer {
    pub fn new(rx: broadcast::Receiver<EpEvent>, model: UiModel) -> Self {
        Self { rx, model }
    }

    fn columns(width: u16) -> usize {
        if width < 40 {
            1
        } else if width < 70 {
            2
        } else if width < 100 {
            3
        } else {
            4
        }
    }

    fn sorted_episodes(model: &UiModel) -> Vec<(usize, &crate::ui_model::EpState)> {
        let mut eps: Vec<_> = model.episodes.iter().enumerate().collect();
        // Active/failing float to top; then by ep number
        eps.sort_by(|a, b| {
            let a_active = !a.1.phase.is_terminal() || matches!(a.1.phase, Phase::Failed);
            let b_active = !b.1.phase.is_terminal() || matches!(b.1.phase, Phase::Failed);
            b_active
                .cmp(&a_active)
                .then_with(|| a.1.number.cmp(&b.1.number))
        });
        eps
    }

    fn render(model: &UiModel, area: Rect, page_offset: usize, f: &mut Frame) {
        let cols = Self::columns(area.width);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(area);

        // Header
        let header = Paragraph::new(" anime-dl — scoreboard ").style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(header, chunks[0]);

        // Grid body
        let body = chunks[1];
        let sorted = Self::sorted_episodes(model);

        let cell_w = body.width / cols as u16;
        let max_rows = body.height as usize;
        let cells_per_page = max_rows * cols;
        let start = page_offset * cells_per_page;
        let visible: Vec<_> = sorted.iter().skip(start).take(cells_per_page).collect();

        for (i, (_idx, ep)) in visible.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let x = body.x + col as u16 * cell_w;
            let y = body.y + row as u16;
            if y >= body.y + body.height {
                break;
            }
            let cell_area = Rect::new(x, y, cell_w, 1);
            render_cell(f, cell_area, ep);
        }

        // Footer with page indicator
        let total_pages = sorted.len().div_ceil(cells_per_page);
        let footer = Paragraph::new(format!(
            " {} done · {} failed · page {}/{}",
            model.total_done,
            model.total_failed,
            page_offset + 1,
            total_pages.max(1)
        ))
        .style(Style::default().fg(Color::White).bg(Color::Black));
        f.render_widget(footer, chunks[2]);
    }
}

fn phase_icon(phase: &Phase) -> &'static str {
    match phase {
        Phase::Queued => "·",
        Phase::Measuring => "?",
        Phase::Downloading { .. } => "↓",
        Phase::Done { .. } => "✓",
        Phase::Failed => "✗",
        Phase::Cancelled => "⨯",
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

fn render_cell(f: &mut Frame, area: Rect, ep: &crate::ui_model::EpState) {
    let color = phase_color(&ep.phase);
    let icon = phase_icon(&ep.phase);

    let mini_bar = match &ep.phase {
        Phase::Downloading { pct, .. } => {
            let bar_w = (area.width.saturating_sub(8) as usize).clamp(3, 10);
            let filled = ((*pct / 100.0) * bar_w as f64) as usize;
            let bar: String = (0..bar_w)
                .map(|i| if i < filled { '█' } else { '░' })
                .collect();
            bar.to_string()
        }
        _ => String::new(),
    };

    let text = if !mini_bar.is_empty() {
        format!("{icon}Ep{:>2}{mini_bar}", ep.number)
    } else {
        format!("{icon}Ep{:>2}", ep.number)
    };

    let widget = Paragraph::new(text).style(Style::default().fg(color));
    f.render_widget(widget, area);
}

#[async_trait::async_trait]
impl Renderer for ScoreboardRenderer {
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
            let cols = Self::columns(area.width);
            let cell_h = if area.height > 3 {
                (area.height - 3) as usize
            } else {
                1
            };
            let cells_per_page = cell_h * cols;
            let total_sorted = Self::sorted_episodes(&model).len();

            if total_sorted > 0 && cells_per_page > 0 {
                let now = Instant::now();
                if now.duration_since(last_page_cycle) >= PAGE_CYCLE {
                    let total_pages = total_sorted.div_ceil(cells_per_page);
                    page_offset = (page_offset + 1) % total_pages.max(1);
                    last_page_cycle = now;
                }
            }

            // Draw
            if let Some(term) = guard.terminal() {
                let offset = page_offset;
                term.draw(|f| Self::render(&model, f.area(), offset, f))?;
            }

            if model.all_terminal() {
                sleep(std::time::Duration::from_millis(500)).await;
                if let Some(term) = guard.terminal() {
                    let offset = page_offset;
                    term.draw(|f| Self::render(&model, f.area(), offset, f))?;
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

    fn render_to_buffer(model: &UiModel, page_offset: usize, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal =
            Terminal::new(backend).expect("create TestBackend terminal for scoreboard test");
        let _ = terminal.draw(|f| {
            ScoreboardRenderer::render(model, f.area(), page_offset, f);
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
        let src = include_str!("scoreboard.rs");
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
    fn startup_queued_episodes() {
        let model = make_model(&[1, 2, 3, 4, 5]);
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("scoreboard"), "title visible");
        assert!(buf.contains("Ep 1") || buf.contains("·Ep"), "episode cells");
    }

    #[test]
    fn column_count_depends_on_width() {
        assert_eq!(ScoreboardRenderer::columns(30), 1);
        assert_eq!(ScoreboardRenderer::columns(50), 2);
        assert_eq!(ScoreboardRenderer::columns(80), 3);
        assert_eq!(ScoreboardRenderer::columns(110), 4);
    }

    #[test]
    fn active_episodes_float_to_top() {
        let mut model = make_model(&[1, 2, 3]);
        model.apply(&EpEvent::Done {
            ep: 1,
            host: "a.pl".into(),
            size_mb: 100.0,
        });
        // Ep 2 should be active (queued), ep 3 active (queued)
        let sorted = ScoreboardRenderer::sorted_episodes(&model);
        // Done ep is terminal — goes last
        assert_eq!(sorted[0].1.number, 2, "ep2 should float to top");
        assert_eq!(sorted[1].1.number, 3, "ep3 should be second");
        assert_eq!(sorted[2].1.number, 1, "ep1 (done) should be last");
    }

    #[test]
    fn done_episode_shows_icon() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Done {
            ep: 1,
            host: "a.pl".into(),
            size_mb: 100.0,
        });
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(buf.contains("✓"), "done icon visible");
    }

    #[test]
    fn narrow_terminal_uses_single_column() {
        let model = make_model(&[1, 2]);
        let buf = render_to_buffer(&model, 0, 39, 10);
        // Single column: each ep on its own line
        assert!(buf.contains("Ep 1"), "ep1 visible");
        assert!(buf.contains("Ep 2"), "ep2 visible");
    }

    #[test]
    fn no_ansi_escape_codes() {
        let model = make_model(&[1]);
        let buf = render_to_buffer(&model, 0, 80, 24);
        assert!(!buf.contains('\x1b'), "no ANSI escapes");
    }

    #[test]
    fn mini_bar_shows_for_downloading() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Winner {
            ep: 1,
            host: "cda.pl".into(),
        });
        model.apply(&EpEvent::Progress {
            ep: 1,
            host: "cda.pl".into(),
            pct: 50.0,
            speed: "1 MiB/s".into(),
            eta: "10s".into(),
            downloaded: 5_000_000,
            total: 10_000_000,
        });
        let buf = render_to_buffer(&model, 0, 80, 24);
        // Should have a progress bar character or block
        assert!(
            buf.contains('█') || buf.contains('░'),
            "progress bar visible"
        );
    }
}
