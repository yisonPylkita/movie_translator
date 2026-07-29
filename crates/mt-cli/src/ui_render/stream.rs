//! Stream renderer — alternate screen, styled colored event stream.
//!
//! Bounded ring buffer (last 200 events). Auto-scroll. Final summary on exit.
//! Colors by event type: measuring=cyan, progress=white, done=green, failed=red.
//! Distinct from plain mode: this is alternate-screen, styled, with summary footer.

use std::collections::VecDeque;
use std::io;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use tokio::sync::broadcast;
use tokio::time::sleep;

use crate::download_types::EpEvent;
use crate::ui_model::UiModel;
use crate::ui_render::{Renderer, TerminalGuard};

const TICK: std::time::Duration = std::time::Duration::from_millis(100);
const MAX_EVENTS: usize = 200;

/// A styled log line for the stream display.
struct StreamEvent {
    line: String,
    color: Color,
}

pub struct StreamRenderer {
    rx: broadcast::Receiver<EpEvent>,
    model: UiModel,
}

impl StreamRenderer {
    pub fn new(rx: broadcast::Receiver<EpEvent>, model: UiModel) -> Self {
        Self { rx, model }
    }

    fn render(events: &VecDeque<StreamEvent>, model: &UiModel, area: Rect, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(2),
            ])
            .split(area);

        // Header
        let header = Paragraph::new(" anime-dl — event stream ").style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(header, chunks[0]);

        // Event stream body
        let body = chunks[1];
        let max_rows = body.height as usize;

        // Show most recent events
        let visible: Vec<_> = events.iter().rev().take(max_rows).collect();
        let lines: Vec<_> = visible
            .iter()
            .rev()
            .map(|e| Line::from(Span::styled(&e.line, Style::default().fg(e.color))))
            .collect();

        let event_block = Block::default().borders(Borders::NONE);
        let p = Paragraph::new(lines)
            .block(event_block)
            .wrap(Wrap { trim: false });
        f.render_widget(p, body);

        // Summary footer
        let summary = format!(
            " {} done · {} failed · {} events | {} active | elapsed {}s",
            model.total_done,
            model.total_failed,
            model.events_since_start,
            model.active_count(),
            model.started_at.elapsed().as_secs(),
        );
        let footer = Paragraph::new(summary)
            .style(Style::default().fg(Color::White).bg(Color::Black))
            .block(Block::default().borders(Borders::TOP));
        f.render_widget(footer, chunks[2]);
    }

    fn event_to_stream(ev: &EpEvent) -> StreamEvent {
        match ev {
            EpEvent::Measuring { ep, host } => StreamEvent {
                line: format!("[{:>3}] measuring on {host}", ep),
                color: Color::Cyan,
            },
            EpEvent::Measured { ep, host, bps } => {
                let mbps = bps / 1_048_576.0;
                StreamEvent {
                    line: format!("[{:>3}] {host} measured at {mbps:.1} MiB/s", ep),
                    color: Color::Cyan,
                }
            }
            EpEvent::MirrorBusy { ep, host } => StreamEvent {
                line: format!("[{:>3}] {host} busy (locked)", ep),
                color: Color::DarkGray,
            },
            EpEvent::Winner { ep, host } => StreamEvent {
                line: format!("[{:>3}] winner: {host}", ep),
                color: Color::LightGreen,
            },
            EpEvent::Progress {
                ep,
                host,
                pct,
                speed,
                eta,
                downloaded,
                total,
            } => {
                let dl_mb = *downloaded as f64 / 1_048_576.0;
                let total_mb = *total as f64 / 1_048_576.0;
                StreamEvent {
                    line: format!(
                        "[{:>3}] {:.1}% {speed} ETA {eta} {dl_mb:.0}/{total_mb:.0} MB ({host})",
                        ep, pct
                    ),
                    color: Color::White,
                }
            }
            EpEvent::MeasurementComplete { ep } => StreamEvent {
                line: format!("[{:>3}] measurement complete", ep),
                color: Color::LightCyan,
            },
            EpEvent::Done { ep, host, size_mb } => StreamEvent {
                line: format!("[{:>3}] DONE {size_mb:.1} MB ({host})", ep),
                color: Color::Green,
            },
            EpEvent::Failed { ep } => StreamEvent {
                line: format!("[{:>3}] FAILED — all mirrors exhausted", ep),
                color: Color::Red,
            },
            EpEvent::MirrorDone { ep, host, success } => {
                let status = if *success { "ok" } else { "fail" };
                StreamEvent {
                    line: format!("[{:>3}] mirror {host} {status}", ep),
                    color: if *success {
                        Color::DarkGray
                    } else {
                        Color::Red
                    },
                }
            }
            EpEvent::Cancelled { ep } => StreamEvent {
                line: format!("[{:>3}] CANCELLED", ep),
                color: Color::Yellow,
            },
        }
    }
}

#[async_trait::async_trait]
impl Renderer for StreamRenderer {
    async fn run(self: Box<Self>) -> io::Result<()> {
        let mut guard = TerminalGuard::enter()?;

        let mut rx = self.rx;
        let mut model = self.model;
        let mut events: VecDeque<StreamEvent> = VecDeque::with_capacity(MAX_EVENTS + 1);

        loop {
            // Drain events
            while let Ok(ev) = rx.try_recv() {
                model.apply(&ev);
                let se = StreamRenderer::event_to_stream(&ev);
                if events.len() >= MAX_EVENTS {
                    events.pop_front();
                }
                events.push_back(se);
            }

            // Draw
            if let Some(term) = guard.terminal() {
                term.draw(|f| Self::render(&events, &model, f.area(), f))?;
            }

            if model.all_terminal() {
                sleep(std::time::Duration::from_millis(500)).await;
                if let Some(term) = guard.terminal() {
                    term.draw(|f| Self::render(&events, &model, f.area(), f))?;
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

    fn make_model(eps: &[i64]) -> UiModel {
        UiModel::new(eps)
    }

    fn render_to_buffer(
        events: &VecDeque<StreamEvent>,
        model: &UiModel,
        width: u16,
        height: u16,
    ) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal =
            Terminal::new(backend).expect("create TestBackend terminal for stream test");
        let _ = terminal.draw(|f| {
            StreamRenderer::render(events, model, f.area(), f);
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
        let src = include_str!("stream.rs");
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
    fn startup_empty_events() {
        let model = make_model(&[1, 2, 3]);
        let events = VecDeque::new();
        let buf = render_to_buffer(&events, &model, 80, 24);
        assert!(buf.contains("event stream"), "title visible");
        assert!(buf.contains("3 active"), "active count");
    }

    #[test]
    fn measuring_events_colored_cyan() {
        let se = StreamRenderer::event_to_stream(&EpEvent::Measuring {
            ep: 1,
            host: "cda.pl".into(),
        });
        assert_eq!(se.color, Color::Cyan);
        assert!(se.line.contains("measuring"), "measuring text");
    }

    #[test]
    fn done_events_colored_green() {
        let se = StreamRenderer::event_to_stream(&EpEvent::Done {
            ep: 1,
            host: "cda.pl".into(),
            size_mb: 250.0,
        });
        assert_eq!(se.color, Color::Green);
        assert!(se.line.contains("DONE"), "done text");
    }

    #[test]
    fn failed_events_colored_red() {
        let se = StreamRenderer::event_to_stream(&EpEvent::Failed { ep: 3 });
        assert_eq!(se.color, Color::Red);
        assert!(se.line.contains("FAILED"), "failed text");
    }

    #[test]
    fn progress_events_colored_white() {
        let se = StreamRenderer::event_to_stream(&EpEvent::Progress {
            ep: 1,
            host: "cda.pl".into(),
            pct: 50.0,
            speed: "1 MiB/s".into(),
            eta: "10s".into(),
            downloaded: 5_000_000,
            total: 10_000_000,
        });
        assert_eq!(se.color, Color::White);
        assert!(se.line.contains("50"), "percentage in text");
    }

    #[test]
    fn winner_events_colored() {
        let se = StreamRenderer::event_to_stream(&EpEvent::Winner {
            ep: 1,
            host: "cda.pl".into(),
        });
        assert_eq!(se.color, Color::LightGreen);
        assert!(se.line.contains("winner"), "winner text");
    }

    #[test]
    fn ring_buffer_bounded_to_max() {
        let mut events: VecDeque<StreamEvent> = VecDeque::with_capacity(MAX_EVENTS + 1);
        for i in 0..(MAX_EVENTS + 50) {
            let ev = StreamEvent {
                line: format!("event {i}"),
                color: Color::White,
            };
            if events.len() >= MAX_EVENTS {
                events.pop_front();
            }
            events.push_back(ev);
        }
        assert_eq!(events.len(), MAX_EVENTS);
    }

    #[test]
    fn summary_shows_counts() {
        let model = make_model(&[1, 2]);
        let events = VecDeque::new();
        let buf = render_to_buffer(&events, &model, 80, 24);
        assert!(buf.contains("2 active"), "active count in summary");
    }

    #[test]
    fn no_ansi_escape_codes() {
        let model = make_model(&[1]);
        let events = VecDeque::new();
        let buf = render_to_buffer(&events, &model, 80, 24);
        assert!(!buf.contains('\x1b'), "no ANSI escapes");
    }
}
