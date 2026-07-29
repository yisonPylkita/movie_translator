//! UI renderer dispatch — selects a renderer based on the requested mode.
//!
//! All four fancy renderers (Dashboard, Timeline, Scoreboard, Stream) use the
//! alternate screen via [`TerminalGuard`] and are read-only: no keyboard/mouse
//! input, no raw mode, no prompt.

mod dashboard;
mod scoreboard;
mod stream;
mod timeline;

use std::io::{self, Stdout};

use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
pub use dashboard::DashboardRenderer;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
pub use scoreboard::ScoreboardRenderer;
pub use stream::StreamRenderer;
pub use timeline::TimelineRenderer;
use tokio::sync::broadcast;

use crate::download_types::EpEvent;
use crate::ui_model::UiModel;

/// Fancy renderer modes (no keyboard/mouse input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Dashboard,
    Timeline,
    Scoreboard,
    Stream,
}

/// Shared trait for all fancy renderers.
#[async_trait::async_trait]
pub trait Renderer: Send {
    /// Run the renderer to completion. Consumes self.
    async fn run(self: Box<Self>) -> io::Result<()>;
}

/// Factory: build the correct renderer from a mode and event stream.
///
/// The caller provides the initial episode numbers; the factory creates a
/// shared [`UiModel`] and passes a receiver + model to the chosen renderer.
pub fn select_renderer(
    mode: UiMode,
    rx: broadcast::Receiver<EpEvent>,
    episode_numbers: &[i64],
) -> Box<dyn Renderer> {
    let model = UiModel::new(episode_numbers);
    match mode {
        UiMode::Dashboard => Box::new(DashboardRenderer::new(rx, model)),
        UiMode::Timeline => Box::new(TimelineRenderer::new(rx, model)),
        UiMode::Scoreboard => Box::new(ScoreboardRenderer::new(rx, model)),
        UiMode::Stream => Box::new(StreamRenderer::new(rx, model)),
    }
}

// ── TerminalGuard (no raw mode) ──────────────────────────────────

/// Guards alternate-screen entry/restore without enabling raw mode.
///
/// All four fancy renderers use this. No keyboard input required, so
/// raw mode is omitted. Restores on drop, normal return, or panic.
pub struct TerminalGuard {
    inner: Option<Terminal<CrosstermBackend<Stdout>>>,
}

impl TerminalGuard {
    /// Enter alternate screen and create a terminal.
    pub fn enter() -> io::Result<Self> {
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(Self {
            inner: Some(terminal),
        })
    }

    /// Access the terminal for drawing.
    pub fn terminal(&mut self) -> Option<&mut Terminal<CrosstermBackend<Stdout>>> {
        self.inner.as_mut()
    }

    /// Restore the terminal — leave alternate screen.
    pub fn restore(&mut self) {
        if let Some(mut terminal) = self.inner.take() {
            let _ = terminal.show_cursor();
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Install a panic hook that restores the terminal guard first.
pub fn install_panic_hook(guard: std::sync::Arc<std::sync::Mutex<TerminalGuard>>) {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut g) = guard.lock() {
            g.restore();
        }
        default(info);
    }));
}
