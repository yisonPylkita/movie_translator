//! Timeline renderer — row-per-episode timeline with stage/history segments.
//!
//! Glyph vocabulary: ⏳ queued, 🔍 quality inspection, 🏁 racing, ⬇ downloading,
//! 🔄 fallback, ✅ done, ❌ failed. Auto-scroll active group. Vertical layout
//! with time axis.

use std::io;

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

pub struct TimelineRenderer {
    rx: broadcast::Receiver<EpEvent>,
    model: UiModel,
}

impl TimelineRenderer {
    pub fn new(rx: broadcast::Receiver<EpEvent>, model: UiModel) -> Self {
        Self { rx, model }
    }

    fn render(model: &UiModel, area: Rect, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(area);

        // Header
        let header = Paragraph::new(" anime-dl — episode timeline ").style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(header, chunks[0]);

        // Body: table-like rows
        let body_area = chunks[1];
        let max_rows = body_area.height as usize;
        let mut offset = if model.active_count() > 0 {
            // Find first active episode and try to center it
            model
                .episodes
                .iter()
                .position(|e| !e.phase.is_terminal())
                .unwrap_or(0)
                .saturating_sub(max_rows / 4)
        } else {
            0
        };
        // Clamp
        if offset + max_rows > model.episodes.len() {
            offset = model.episodes.len().saturating_sub(max_rows);
        }

        let visible = &model.episodes[offset..(offset + max_rows).min(model.episodes.len())];

        for (i, ep) in visible.iter().enumerate() {
            let y = body_area.y + i as u16;
            if y >= body_area.y + body_area.height {
                break;
            }
            let row_area = Rect::new(body_area.x, y, body_area.width, 1);
            render_timeline_row(f, row_area, ep, model.started_at.elapsed());
        }

        // Summary
        let summary = format!(
            " {} done · {} failed · {} active · {} total",
            model.total_done,
            model.total_failed,
            model.active_count(),
            model.episodes.len()
        );
        let footer =
            Paragraph::new(summary).style(Style::default().fg(Color::White).bg(Color::Black));
        f.render_widget(footer, chunks[2]);
    }
}

fn phase_timeline_glyph(phase: &Phase) -> &'static str {
    match phase {
        Phase::Queued => "⏳",
        Phase::Measuring => "🔍",
        Phase::Downloading { .. } => "⬇",
        Phase::Done { .. } => "✅",
        Phase::Failed => "❌",
        Phase::Cancelled => "⏹",
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

fn render_timeline_row(
    f: &mut Frame,
    area: Rect,
    ep: &crate::ui_model::EpState,
    elapsed: std::time::Duration,
) {
    let color = phase_color(&ep.phase);
    let glyph = phase_timeline_glyph(&ep.phase);

    // Build stage history string
    let stage_info = match &ep.phase {
        Phase::Measuring => {
            let active = ep.mirrors.iter().filter(|m| m.active).count();
            let measured = ep.mirrors.iter().filter(|m| m.bps.is_some()).count();
            format!("{glyph} meas {measured}/{active}")
        }
        Phase::Downloading {
            pct, speed, eta, ..
        } => {
            let gauge_width = (area.width.saturating_sub(40) as usize).clamp(10, 30);
            let filled = ((*pct / 100.0) * gauge_width as f64) as usize;
            let bar: String = (0..gauge_width)
                .map(|i| if i < filled { '█' } else { '░' })
                .collect();
            format!("{glyph} {:>3.0}% {bar} {speed} ETA {eta}", pct)
        }
        Phase::Done { host, size_mb } => {
            format!("{glyph} {size_mb:.1} MB ({host})")
        }
        Phase::Failed => format!("{glyph} all mirrors failed"),
        Phase::Cancelled => format!("{glyph} cancelled"),
        Phase::Queued => format!("{glyph} waiting..."),
    };

    // Time since start
    let secs = elapsed.as_secs();
    let time_str = format!("{:>4}s", secs);

    let line = format!(
        " Ep{:>3} {stage_info:<width$} {time_str}",
        ep.number,
        width = (area.width.saturating_sub(20) as usize).min(40)
    );
    let text = Paragraph::new(line).style(Style::default().fg(color));
    f.render_widget(text, area);
}

#[async_trait::async_trait]
impl Renderer for TimelineRenderer {
    async fn run(self: Box<Self>) -> io::Result<()> {
        let mut guard = TerminalGuard::enter()?;

        let mut rx = self.rx;
        let mut model = self.model;

        loop {
            // Drain events
            while let Ok(ev) = rx.try_recv() {
                model.apply(&ev);
            }

            // Draw
            if let Some(term) = guard.terminal() {
                term.draw(|f| Self::render(&model, f.area(), f))?;
            }

            if model.all_terminal() {
                sleep(std::time::Duration::from_millis(500)).await;
                if let Some(term) = guard.terminal() {
                    term.draw(|f| Self::render(&model, f.area(), f))?;
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

    fn render_to_buffer(model: &UiModel, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal =
            Terminal::new(backend).expect("create TestBackend terminal for timeline test");
        let _ = terminal.draw(|f| {
            TimelineRenderer::render(model, f.area(), f);
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
        let src = include_str!("timeline.rs");
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
        let model = make_model(&[1, 2, 3]);
        let buf = render_to_buffer(&model, 80, 24);
        assert!(buf.contains("⏳"), "queued glyph");
        assert!(buf.contains("Ep  1"), "episode 1");
        assert!(buf.contains("Ep  2"), "episode 2");
        assert!(buf.contains("Ep  3"), "episode 3");
    }

    #[test]
    fn done_episode_shows_glyph() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Done {
            ep: 1,
            host: "cda.pl".into(),
            size_mb: 200.0,
        });
        let buf = render_to_buffer(&model, 80, 24);
        assert!(buf.contains("✅"), "done glyph");
        assert!(buf.contains("200"), "size");
    }

    #[test]
    fn failed_episode_shows_glyph() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Failed { ep: 1 });
        let buf = render_to_buffer(&model, 80, 24);
        assert!(buf.contains("❌"), "failed glyph");
    }

    #[test]
    fn measuring_phase_shows_glyph() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        });
        let buf = render_to_buffer(&model, 80, 24);
        assert!(buf.contains("🔍"), "measuring glyph");
    }

    #[test]
    fn downloading_phase_shows_progress() {
        let mut model = make_model(&[1]);
        model.apply(&EpEvent::Winner {
            ep: 1,
            host: "cda.pl".into(),
        });
        model.apply(&EpEvent::Progress {
            ep: 1,
            host: "cda.pl".into(),
            pct: 50.0,
            speed: "3 MiB/s".into(),
            eta: "10s".into(),
            downloaded: 5_000_000,
            total: 10_000_000,
        });
        let buf = render_to_buffer(&model, 80, 24);
        assert!(buf.contains("⬇"), "download glyph");
        assert!(buf.contains("50%"), "percentage");
    }

    #[test]
    fn summary_shows_counts() {
        let mut model = make_model(&[1, 2, 3]);
        model.apply(&EpEvent::Done {
            ep: 1,
            host: "a.pl".into(),
            size_mb: 100.0,
        });
        let buf = render_to_buffer(&model, 80, 24);
        assert!(buf.contains("1 done"), "done count");
        assert!(buf.contains("3 total"), "total count");
    }

    #[test]
    fn narrow_terminal_no_panic() {
        let model = make_model(&[1, 2, 3]);
        let _buf = render_to_buffer(&model, 30, 12);
        // Should not panic
    }

    #[test]
    fn no_ansi_escape_codes() {
        let model = make_model(&[1]);
        let buf = render_to_buffer(&model, 80, 24);
        assert!(!buf.contains('\x1b'), "no ANSI escapes");
    }
}
