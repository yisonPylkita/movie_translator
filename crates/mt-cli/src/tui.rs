//! Ratatui-driven progress UI consuming `mt_pipeline::ProgressEvent` events.
//!
//! Two operating modes:
//!
//! * **Interactive TUI** (when stdout is a TTY): enters the alternate screen,
//!   draws a per-file table with status icons, current stage, OCR/translate
//!   sub-progress, plus a bottom log pane fed by tracing + Python stderr. Tears
//!   down cleanly on drop, on `q`/`Ctrl-C`, or on panic via [`set_hook`].
//!
//! * **Plain mode** (CI, pipes, redirected stdout): no alternate screen, no raw
//!   mode — just prints one line per file lifecycle event. Same event stream,
//!   different sink.
//!
//! The renderer runs on a dedicated OS thread, NOT a tokio task: ratatui's
//! crossterm backend uses blocking poll() and we want the rendering to survive
//! tokio runtime stalls (the pipeline's `spawn_blocking` GPU work can park
//! every worker thread). The pipeline's `mpsc::UnboundedSender<ProgressEvent>`
//! crosses the thread boundary cheaply.
//!
//! # Tracing layer
//!
//! [`TuiTracingLayer`] is a [`tracing_subscriber::Layer`] that drains
//! tracing events into the same progress channel as `ProgressEvent::Log`. The
//! TUI shows the most recent N entries in a scrollable log pane. The layer is
//! always installed; in plain mode the log events still print to stderr via the
//! sibling `fmt` layer, so behaviour for unsuspecting users is unchanged.

use std::io::{self, IsTerminal, Stdout, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use mt_pipeline::{FinishStatus, ProgressEvent, ProgressSender, Stage};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Row, Table, Wrap};
use ratatui::Terminal;
use tokio::sync::mpsc;
use tracing::field::Visit;
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// Status icon + colour for the file table.
fn status_glyph(s: FileViewStatus) -> (&'static str, Color) {
    match s {
        FileViewStatus::Queued => ("·", Color::DarkGray),
        FileViewStatus::Active => ("▶", Color::Cyan),
        FileViewStatus::Success => ("✓", Color::Green),
        FileViewStatus::Skipped => ("⏭", Color::Yellow),
        FileViewStatus::SkippedNoSubs => ("⏭", Color::Yellow),
        FileViewStatus::Failed => ("✗", Color::Red),
    }
}

/// One row's worth of UI state.
#[derive(Debug, Clone)]
struct FileView {
    path: PathBuf,
    status: FileViewStatus,
    stage: Option<Stage>,
    /// Per-stage progress (e.g. OCR n/N, translate lines/total).
    sub_progress: Option<(u64, u64)>,
    sub_label: Option<String>,
    /// True once we've seen FileFinished for this row.
    finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileViewStatus {
    Queued,
    Active,
    Success,
    Skipped,
    SkippedNoSubs,
    Failed,
}

impl From<FinishStatus> for FileViewStatus {
    fn from(s: FinishStatus) -> Self {
        match s {
            FinishStatus::Success => FileViewStatus::Success,
            FinishStatus::Failed => FileViewStatus::Failed,
            FinishStatus::Skipped => FileViewStatus::Skipped,
            FinishStatus::SkippedNoSubs => FileViewStatus::SkippedNoSubs,
        }
    }
}

/// Aggregate UI state shared between the TUI thread and the (single) event
/// consumer that owns it. Owned only by the renderer thread — events are
/// pushed in via the mpsc channel.
#[derive(Debug, Default)]
struct UiState {
    files: Vec<FileView>,
    log_lines: Vec<String>,
    /// Newest at the end; trimmed to MAX_LOG_LINES.
    started_at: Option<Instant>,
    /// Path of the python stderr capture file, shown in the log header.
    python_log_path: Option<PathBuf>,
}

const MAX_LOG_LINES: usize = 500;
const LOG_TAIL_RENDER: usize = 100;

impl UiState {
    fn upsert_file(&mut self, path: &PathBuf) -> &mut FileView {
        if let Some(idx) = self.files.iter().position(|f| &f.path == path) {
            return &mut self.files[idx];
        }
        self.files.push(FileView {
            path: path.clone(),
            status: FileViewStatus::Queued,
            stage: None,
            sub_progress: None,
            sub_label: None,
            finished: false,
        });
        self.files.last_mut().unwrap()
    }

    fn push_log(&mut self, line: String) {
        self.log_lines.push(line);
        if self.log_lines.len() > MAX_LOG_LINES {
            let drop = self.log_lines.len() - MAX_LOG_LINES;
            self.log_lines.drain(..drop);
        }
    }

    fn apply(&mut self, ev: ProgressEvent) {
        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
        match ev {
            ProgressEvent::Queued { files } => {
                for path in files {
                    self.upsert_file(&path);
                }
            }
            ProgressEvent::FileStarted { path } => {
                let f = self.upsert_file(&path);
                f.status = FileViewStatus::Active;
                f.stage = None;
                f.sub_progress = None;
                f.sub_label = None;
            }
            ProgressEvent::StageEntered { path, stage } => {
                let f = self.upsert_file(&path);
                f.status = FileViewStatus::Active;
                f.stage = Some(stage);
                // Clear sub-progress between stages.
                f.sub_progress = None;
                f.sub_label = None;
            }
            ProgressEvent::OcrProgress { path, done, total } => {
                let f = self.upsert_file(&path);
                f.sub_progress = Some((done, total));
                f.sub_label = Some(format!("OCR {done}/{total}"));
            }
            ProgressEvent::FetchResult {
                path,
                candidates_found,
                downloaded,
            } => {
                let f = self.upsert_file(&path);
                f.sub_label = Some(format!("fetch: {downloaded}/{candidates_found} dl'd"));
            }
            ProgressEvent::TranslateBatch {
                path,
                lines_done,
                lines_total,
                model,
            } => {
                let f = self.upsert_file(&path);
                f.sub_progress = Some((lines_done, lines_total));
                f.sub_label = Some(format!("translate ({model}) {lines_done}/{lines_total}"));
            }
            ProgressEvent::Log {
                level,
                target,
                message,
            } => {
                let line = format!("[{level} {target}] {message}");
                self.push_log(line);
            }
            ProgressEvent::FileFinished { path, status } => {
                let f = self.upsert_file(&path);
                f.status = status.into();
                f.stage = None;
                f.sub_progress = None;
                f.sub_label = None;
                f.finished = true;
            }
        }
    }
}

/// Tracing layer that forwards events into the TUI's progress channel.
///
/// Newly logged events become `ProgressEvent::Log { level, target, message }`,
/// which the TUI accumulates into its log pane. When the channel is closed
/// (TUI torn down), events are dropped silently.
pub struct TuiTracingLayer {
    sender: ProgressSender,
}

impl TuiTracingLayer {
    pub fn new(sender: ProgressSender) -> Self {
        Self { sender }
    }
}

/// Visitor pulling the most useful fields off a tracing event into a string.
/// We surface the `message` field; other fields are appended as `k=v`.
struct MessageVisitor {
    message: String,
    extras: String,
}

impl MessageVisitor {
    fn new() -> Self {
        Self {
            message: String::new(),
            extras: String::new(),
        }
    }

    fn into_line(self) -> String {
        if self.extras.is_empty() {
            self.message
        } else if self.message.is_empty() {
            self.extras
        } else {
            format!("{} {}", self.message, self.extras)
        }
    }
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            use std::fmt::Write;
            let _ = write!(self.message, "{value:?}");
        } else {
            use std::fmt::Write;
            if !self.extras.is_empty() {
                self.extras.push(' ');
            }
            let _ = write!(self.extras, "{}={:?}", field.name(), value);
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            if !self.extras.is_empty() {
                self.extras.push(' ');
            }
            self.extras.push_str(field.name());
            self.extras.push('=');
            self.extras.push_str(value);
        }
    }
}

impl<S: Subscriber> Layer<S> for TuiTracingLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut v = MessageVisitor::new();
        event.record(&mut v);
        let message = v.into_line();
        let meta = event.metadata();
        self.sender.send(ProgressEvent::Log {
            level: meta.level().to_string(),
            target: meta.target().to_string(),
            message,
        });
    }
}

/// Capture handle for Python stderr. The init script in `mt_ml::backend`
/// redirects Python's `sys.stderr` to a file under `.translate_temp/`; this
/// type holds the path and tails the file into the log pane.
///
/// The capture file is created on demand at the path returned by
/// [`python_stderr_capture_path`]. We do NOT manage Python's interpreter
/// directly here — `mt_ml::backend::init_python_runtime` does that on first
/// use of the embedded interpreter.
pub fn python_stderr_capture_path(root: &std::path::Path) -> PathBuf {
    root.join(".translate_temp").join("python.stderr.log")
}

/// A guard ensuring the terminal is restored regardless of how the program exits.
pub struct TerminalGuard {
    /// `None` after `restore`, which is also called from Drop.
    inner: Option<Terminal<CrosstermBackend<Stdout>>>,
    raw_enabled: bool,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.execute(EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(Self {
            inner: Some(terminal),
            raw_enabled: true,
        })
    }

    pub fn terminal(&mut self) -> Option<&mut Terminal<CrosstermBackend<Stdout>>> {
        self.inner.as_mut()
    }

    pub fn restore(&mut self) {
        // Idempotent — also safe to call from Drop.
        if let Some(mut terminal) = self.inner.take() {
            let _ = terminal.show_cursor();
        }
        if self.raw_enabled {
            let _ = disable_raw_mode();
            let _ = io::stdout().execute(LeaveAlternateScreen);
            self.raw_enabled = false;
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Public entry: spawn a TUI consumer thread on the given event receiver.
///
/// Returns a handle that the caller joins after `run_all_with_progress` returns
/// (and the pipeline's sender is dropped). The TUI also exits early on `q` or
/// Ctrl-C; in that case `quit_requested()` returns `true`.
pub struct TuiHandle {
    join: Option<JoinHandle<UiState>>,
    quit_flag: Arc<Mutex<bool>>,
    python_log_path: PathBuf,
}

impl TuiHandle {
    /// Wait for the renderer thread to finish (the sender must be dropped or
    /// the user must press `q`). Returns the final `UiState` for summary use.
    pub fn join(mut self) -> Option<JoinedTuiState> {
        let st = self.join.take()?.join().ok()?;
        Some(JoinedTuiState {
            files: st.files.into_iter().map(|f| (f.path, f.status)).collect(),
            python_log_path: self.python_log_path,
        })
    }

    pub fn quit_requested(&self) -> bool {
        *self.quit_flag.lock().unwrap()
    }

    pub fn python_log_path(&self) -> &std::path::Path {
        &self.python_log_path
    }
}

/// Final state harvested from the TUI on join, used to print the summary
/// after teardown.
pub struct JoinedTuiState {
    pub files: Vec<(PathBuf, FileViewStatus)>,
    pub python_log_path: PathBuf,
}

impl JoinedTuiState {
    pub fn was_failure(&self, status: FileViewStatus) -> bool {
        matches!(status, FileViewStatus::Failed)
    }
}

/// Public glyph for printing the JoinedTuiState (used by the summary printer).
pub fn finish_glyph(s: FileViewStatus) -> &'static str {
    status_glyph(s).0
}

/// Public-readable name for `FileViewStatus`.
pub fn status_name(s: FileViewStatus) -> &'static str {
    match s {
        FileViewStatus::Queued => "queued",
        FileViewStatus::Active => "active",
        FileViewStatus::Success => "success",
        FileViewStatus::Skipped => "skipped",
        FileViewStatus::SkippedNoSubs => "skipped (no subtitles)",
        FileViewStatus::Failed => "failed",
    }
}

/// True if stdout looks like a real terminal (TUI is appropriate).
pub fn stdout_is_tty() -> bool {
    io::stdout().is_terminal()
}

/// Spawn the renderer thread. The renderer pulls events from `rx`; when `rx`
/// is closed (sender dropped) the thread exits cleanly.
pub fn spawn_tui(
    mut rx: mpsc::UnboundedReceiver<ProgressEvent>,
    python_log_path: PathBuf,
    force_plain: bool,
) -> TuiHandle {
    let quit_flag = Arc::new(Mutex::new(false));
    let quit_flag_for_thread = quit_flag.clone();
    let python_log_path_clone = python_log_path.clone();

    let interactive = !force_plain && stdout_is_tty();

    let join = std::thread::spawn(move || {
        if interactive {
            run_interactive(&mut rx, quit_flag_for_thread, python_log_path_clone)
        } else {
            run_plain(&mut rx)
        }
    });

    TuiHandle {
        join: Some(join),
        quit_flag,
        python_log_path,
    }
}

/// Interactive renderer loop.
fn run_interactive(
    rx: &mut mpsc::UnboundedReceiver<ProgressEvent>,
    quit_flag: Arc<Mutex<bool>>,
    python_log_path: PathBuf,
) -> UiState {
    let mut state = UiState {
        python_log_path: Some(python_log_path),
        ..UiState::default()
    };
    state.started_at = Some(Instant::now());

    // Best-effort terminal setup. If we can't enter raw mode (no TTY in
    // practice), degrade to the plain renderer.
    let guard = match TerminalGuard::enter() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("TUI init failed ({e}); falling back to plain mode");
            return run_plain(rx);
        }
    };
    // Move into a Mutex so the panic hook can restore the terminal too.
    let guard = Arc::new(Mutex::new(guard));
    install_panic_hook(guard.clone());

    let tick = Duration::from_millis(100);
    let mut last_render = Instant::now() - tick;

    loop {
        // Drain whatever events are available (non-blocking).
        let mut received_any = false;
        while let Ok(ev) = rx.try_recv() {
            state.apply(ev);
            received_any = true;
        }

        // Check for keypresses (non-blocking poll, 50ms).
        if event::poll(Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.kind == KeyEventKind::Press
                    && (k.code == KeyCode::Char('q')
                        || (k.code == KeyCode::Char('c')
                            && k.modifiers.contains(KeyModifiers::CONTROL)))
                {
                    *quit_flag.lock().unwrap() = true;
                    break;
                }
            }
        }

        // Render at ~10 Hz, plus immediately after an event burst.
        if received_any || last_render.elapsed() >= tick {
            let mut g = guard.lock().unwrap();
            if let Some(term) = g.terminal() {
                let _ = term.draw(|f| render(f, &state));
            }
            drop(g);
            last_render = Instant::now();
        }

        // Channel closed AND no events queued → pipeline is done, exit.
        if rx.is_closed() && rx.is_empty() {
            // Final render so the closing screen reflects FileFinished states.
            let mut g = guard.lock().unwrap();
            if let Some(term) = g.terminal() {
                let _ = term.draw(|f| render(f, &state));
            }
            drop(g);
            break;
        }
    }

    // Tear down — explicit so it happens before we return state (in case any
    // caller wants to print right after).
    if let Ok(mut g) = guard.lock() {
        g.restore();
    }
    state
}

fn install_panic_hook(guard: Arc<Mutex<TerminalGuard>>) {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut g) = guard.lock() {
            g.restore();
        }
        default(info);
    }));
}

/// Plain-mode renderer: one line per FileStarted / FileFinished, no TUI.
fn run_plain(rx: &mut mpsc::UnboundedReceiver<ProgressEvent>) -> UiState {
    let mut state = UiState::default();
    while let Some(ev) = rx.blocking_recv() {
        match &ev {
            ProgressEvent::FileStarted { path } => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let mut out = io::stderr();
                let _ = writeln!(out, "▶ {name}");
            }
            ProgressEvent::FileFinished { path, status } => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let glyph = status_glyph(FileViewStatus::from(*status)).0;
                let mut out = io::stderr();
                let _ = writeln!(out, "{glyph} {name} ({})", status.as_str());
            }
            _ => {}
        }
        state.apply(ev);
    }
    state
}

fn render(frame: &mut ratatui::Frame<'_>, state: &UiState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Percentage(35),
            Constraint::Length(1),
        ])
        .split(area);

    render_header(frame, chunks[0], state);
    render_files(frame, chunks[1], state);
    render_logs(frame, chunks[2], state);
    render_footer(frame, chunks[3]);
}

fn render_header(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let total = state.files.len();
    let done = state
        .files
        .iter()
        .filter(|f| {
            matches!(
                f.status,
                FileViewStatus::Success | FileViewStatus::Skipped | FileViewStatus::SkippedNoSubs
            )
        })
        .count();
    let active = state
        .files
        .iter()
        .filter(|f| matches!(f.status, FileViewStatus::Active))
        .count();
    let failed = state
        .files
        .iter()
        .filter(|f| matches!(f.status, FileViewStatus::Failed))
        .count();
    let queued = state
        .files
        .iter()
        .filter(|f| matches!(f.status, FileViewStatus::Queued))
        .count();

    let elapsed = state
        .started_at
        .map(|t| t.elapsed())
        .unwrap_or_else(|| Duration::from_secs(0));
    let secs = elapsed.as_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;

    let mut spans = vec![
        Span::styled(
            "movie-translator ",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "  {done}/{total} done · {active} active · {queued} queued"
        )),
    ];
    if failed > 0 {
        spans.push(Span::styled(
            format!(" · {failed} failed"),
            Style::default().fg(Color::Red),
        ));
    }
    spans.push(Span::raw(format!("  ·  elapsed {h:02}:{m:02}:{s:02}")));

    let text = Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL).title("Status"));
    frame.render_widget(text, area);
}

fn render_files(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let inner = area;
    let max_rows = inner.height.saturating_sub(2) as usize; // borders
    let total = state.files.len();
    // Show the most recently-updated window: prioritise active/finished files
    // already at top, but ensure we always see the bottom of the queue. Simple
    // heuristic: take the FIRST `max_rows` files (so initially queued shows the
    // queue; as they finish they remain in order).
    let visible_count = max_rows.min(total);
    let start = total.saturating_sub(visible_count);
    let rows = state.files[start..]
        .iter()
        .map(|f| {
            let (glyph, color) = status_glyph(f.status);
            let stage_label = f
                .stage
                .map(|s| s.label().to_string())
                .unwrap_or_else(|| status_name(f.status).to_string());
            let sub_text = f.sub_label.clone().unwrap_or_default();
            let pct = match (f.sub_progress, f.finished) {
                (_, true) => 100,
                (Some((d, t)), false) if t > 0 => ((d as f64 / t as f64) * 100.0) as u16,
                _ => 0,
            };
            let name = f
                .path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            Row::new(vec![
                Span::styled(glyph.to_string(), Style::default().fg(color)),
                Span::raw(name),
                Span::raw(stage_label),
                Span::raw(format!("{pct:>3}%")),
                Span::raw(sub_text),
            ])
        })
        .collect::<Vec<_>>();

    let widths = [
        Constraint::Length(2),
        Constraint::Percentage(45),
        Constraint::Length(12),
        Constraint::Length(5),
        Constraint::Min(20),
    ];
    let table = Table::new(rows, widths)
        .header(
            Row::new(vec!["", "file", "stage", "pct", "detail"])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("Files ({total})")),
        );
    frame.render_widget(table, inner);

    // Render a wide progress gauge underneath for the most-recent active file
    // — useful for OCR / translate batches.
    if let Some(active) = state
        .files
        .iter()
        .find(|f| matches!(f.status, FileViewStatus::Active) && f.sub_progress.is_some())
    {
        if let Some((d, t)) = active.sub_progress {
            if t > 0 {
                let ratio = (d as f64 / t as f64).clamp(0.0, 1.0);
                // Place at bottom inside the files frame's area; tight stack.
                let gauge_area = Rect {
                    x: inner.x + 1,
                    y: inner.y + inner.height.saturating_sub(2),
                    width: inner.width.saturating_sub(2),
                    height: 1,
                };
                let label = active
                    .sub_label
                    .clone()
                    .unwrap_or_else(|| format!("{d}/{t}"));
                let g = Gauge::default()
                    .gauge_style(Style::default().fg(Color::Cyan))
                    .ratio(ratio)
                    .label(label);
                frame.render_widget(g, gauge_area);
            }
        }
    }
}

fn render_logs(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let tail_start = state.log_lines.len().saturating_sub(LOG_TAIL_RENDER);
    let lines: Vec<Line<'_>> = state.log_lines[tail_start..]
        .iter()
        .map(|l| {
            let style = if l.contains("[ERROR") || l.contains("error") || l.contains("Failed") {
                Style::default().fg(Color::Red)
            } else if l.contains("[WARN") || l.contains("warning") {
                Style::default().fg(Color::Yellow)
            } else if l.contains("[DEBUG") || l.contains("[TRACE") {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default()
            };
            Line::from(Span::styled(l.clone(), style))
        })
        .collect();
    let title = match state.python_log_path.as_ref() {
        Some(p) => format!("Logs (python stderr → {})", p.display()),
        None => "Logs".to_string(),
    };
    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(p, area);
}

fn render_footer(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let p = Paragraph::new(Line::from(vec![
        Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" or "),
        Span::styled("Ctrl-C", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(": quit  "),
        Span::raw("(pipeline runs until current file completes)"),
    ]))
    .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(p, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Event-channel plumbing: a fake orchestrator sends a realistic event
    /// sequence into the UiState aggregator and we assert the per-file state
    /// transitions land where we expect. No terminal IO.
    #[test]
    fn ui_state_aggregates_event_sequence() {
        let mut s = UiState::default();
        let a = PathBuf::from("a.mkv");
        let b = PathBuf::from("b.mkv");
        s.apply(ProgressEvent::Queued {
            files: vec![a.clone(), b.clone()],
        });
        s.apply(ProgressEvent::FileStarted { path: a.clone() });
        s.apply(ProgressEvent::StageEntered {
            path: a.clone(),
            stage: Stage::Translate,
        });
        s.apply(ProgressEvent::TranslateBatch {
            path: a.clone(),
            lines_done: 50,
            lines_total: 100,
            model: "allegro".into(),
        });
        s.apply(ProgressEvent::FileFinished {
            path: a.clone(),
            status: FinishStatus::Success,
        });
        s.apply(ProgressEvent::FileStarted { path: b.clone() });
        s.apply(ProgressEvent::FileFinished {
            path: b.clone(),
            status: FinishStatus::SkippedNoSubs,
        });

        assert_eq!(s.files.len(), 2);
        let row_a = s.files.iter().find(|f| f.path == a).unwrap();
        assert_eq!(row_a.status, FileViewStatus::Success);
        assert!(row_a.finished);
        let row_b = s.files.iter().find(|f| f.path == b).unwrap();
        assert_eq!(row_b.status, FileViewStatus::SkippedNoSubs);
        assert!(row_b.finished);
    }

    /// The log buffer caps at MAX_LOG_LINES and drains FIFO.
    #[test]
    fn log_buffer_trims_to_max() {
        let mut s = UiState::default();
        for i in 0..(MAX_LOG_LINES + 50) {
            s.apply(ProgressEvent::Log {
                level: "INFO".into(),
                target: "t".into(),
                message: format!("msg{i}"),
            });
        }
        assert_eq!(s.log_lines.len(), MAX_LOG_LINES);
        // Oldest 50 dropped → first surviving line is msg50.
        assert!(s.log_lines.first().unwrap().contains("msg50"));
        assert!(s
            .log_lines
            .last()
            .unwrap()
            .contains(&format!("msg{}", MAX_LOG_LINES + 49)));
    }

    /// FinishStatus → FileViewStatus mapping is total.
    #[test]
    fn finish_status_maps_to_view_status() {
        assert_eq!(
            FileViewStatus::from(FinishStatus::Success),
            FileViewStatus::Success
        );
        assert_eq!(
            FileViewStatus::from(FinishStatus::Failed),
            FileViewStatus::Failed
        );
        assert_eq!(
            FileViewStatus::from(FinishStatus::Skipped),
            FileViewStatus::Skipped
        );
        assert_eq!(
            FileViewStatus::from(FinishStatus::SkippedNoSubs),
            FileViewStatus::SkippedNoSubs
        );
    }

    /// The TuiTracingLayer forwards record_str events into the progress channel.
    #[test]
    fn tracing_layer_forwards_to_channel() {
        use tracing_subscriber::prelude::*;

        let (tx, mut rx) = mpsc::unbounded_channel::<ProgressEvent>();
        let layer = TuiTracingLayer::new(ProgressSender::new(tx));
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "test_target", "hello from {}", "test");
        });

        // Drain.
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        assert!(!events.is_empty(), "tracing emit must reach the channel");
        let log_ev = events
            .iter()
            .find_map(|e| match e {
                ProgressEvent::Log {
                    level,
                    target,
                    message,
                } => Some((level.clone(), target.clone(), message.clone())),
                _ => None,
            })
            .expect("expected a Log event");
        assert_eq!(log_ev.0, "INFO");
        assert_eq!(log_ev.1, "test_target");
        assert!(log_ev.2.contains("hello from test"));
    }

    /// The plain-mode renderer drains the channel without doing terminal IO,
    /// updating UiState exactly like the interactive path. We don't observe
    /// stderr output here — just ensure it terminates and the state captures
    /// the events.
    #[test]
    fn plain_renderer_drains_and_exits_cleanly() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ProgressEvent>();
        tx.send(ProgressEvent::Queued {
            files: vec![PathBuf::from("x.mkv")],
        })
        .unwrap();
        tx.send(ProgressEvent::FileStarted {
            path: PathBuf::from("x.mkv"),
        })
        .unwrap();
        tx.send(ProgressEvent::FileFinished {
            path: PathBuf::from("x.mkv"),
            status: FinishStatus::Success,
        })
        .unwrap();
        drop(tx); // close so blocking_recv exits

        let state = run_plain(&mut rx);
        assert_eq!(state.files.len(), 1);
        assert_eq!(state.files[0].status, FileViewStatus::Success);
    }
}
