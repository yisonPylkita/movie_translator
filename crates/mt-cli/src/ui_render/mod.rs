//! Dashboard renderer setup.
//!
//! Dashboard uses the alternate screen via [`TerminalGuard`] and is read-only:
//! no keyboard/mouse input, raw mode, or prompt.

mod dashboard;

use std::io::{self, Stdout};

use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
pub use dashboard::DashboardRenderer;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::broadcast;

use crate::download_types::EpEvent;
use crate::ui_model::UiModel;

/// Shared trait for all fancy renderers.
#[async_trait::async_trait]
pub trait Renderer: Send {
    /// Run the renderer to completion. Consumes self.
    async fn run(self: Box<Self>) -> io::Result<()>;
}

/// Build dashboard renderer from event stream and initial episode numbers.
pub fn select_renderer(
    rx: broadcast::Receiver<EpEvent>,
    episode_numbers: &[i64],
    title: Option<&str>,
    circuit_cooldown_secs: u64,
) -> Box<dyn Renderer> {
    let model = UiModel::new_with_options(
        episode_numbers,
        title.map(|t| t.to_string()),
        circuit_cooldown_secs,
    );
    Box::new(DashboardRenderer::new(rx, model))
}

// ── TerminalGuard (no raw mode) ──────────────────────────────────

/// Guards alternate-screen entry/restore without enabling raw mode.
///
/// Dashboard uses this. No keyboard input required, so
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
