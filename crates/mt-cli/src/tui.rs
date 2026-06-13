//! Ratatui-driven progress dashboard consuming `mt_pipeline::ProgressEvent` events.
//!
//! # Dashboard layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │ Status Bar (title · files · GPU · elapsed)      │
//! ├────────────────────────┬────────────────────────┤
//! │                        │ GPU Worker             │
//! │  File Progress Table   │ ▸ translate            │
//! │  (animated gauges,     │ ████████░░ 80%         │
//! │   per-stage colors)    │ Queue: 2  Last: 1.2s   │
//! │                        │                        │
//! ├────────────────────────┴────────────────────────┤
//! │ Log Pane (scrollable, color-coded by level)     │
//! ├─────────────────────────────────────────────────┤
//! │ q/Ctrl-C: quit  │ j/k: scroll logs  │ Tab: pane │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! Two operating modes:
//!
//! * **Interactive TUI** (when stdout is a TTY): enters the alternate screen,
//!   draws the dashboard layout above. Tears down cleanly on drop, on `q`/`Ctrl-C`,
//!   or on panic via [`set_hook`].
//!
//! * **Plain mode** (CI, pipes, redirected stdout): no alternate screen, no raw
//!   mode — just prints one line per file lifecycle event.
//!
//! The renderer runs on a dedicated OS thread, NOT a tokio task: ratatui's
//! crossterm backend uses blocking poll() and we want the rendering to survive
//! tokio runtime stalls.

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Stdout, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use mt_pipeline::{FinishStatus, ProgressEvent, ProgressSender, Stage};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Gauge, Paragraph, Row, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Table, Wrap,
};
use tokio::sync::mpsc;
use tracing::field::Visit;
use tracing::{Subscriber, warn};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

// ── Colour palette ────────────────────────────────────────────────

/// Each pipeline stage gets a distinct foreground colour so users can
/// visually scan the file table.
fn stage_color(stage: Stage) -> Color {
    match stage {
        Stage::Identify => Color::Cyan,
        Stage::ExtractRef => Color::Blue,
        Stage::Fetch => Color::Magenta,
        Stage::HardsubOcr => Color::LightMagenta,
        Stage::ExtractEnglish => Color::LightBlue,
        Stage::Transcribe => Color::LightYellow,
        Stage::Translate => Color::Green,
        Stage::CreateTracks => Color::Yellow,
        Stage::Mux => Color::Red,
    }
}

/// A subset of the 216‑colour cube and terminal‑safe colours used for
/// GPU job types in the worker panel.
fn gpu_job_color(job_type: &str) -> Color {
    match job_type {
        "translate" => Color::Green,
        "ocr_pgs" => Color::Blue,
        "ocr_burned_in" => Color::LightBlue,
        "inpaint" => Color::Red,
        "hardsub_ocr" => Color::LightMagenta,
        "transcribe" => Color::LightYellow,
        _ => Color::DarkGray,
    }
}

fn log_level_color(level: &str) -> Color {
    match level {
        "ERROR" => Color::Red,
        "WARN" => Color::Yellow,
        "INFO" => Color::White,
        "DEBUG" => Color::DarkGray,
        "TRACE" => Color::DarkGray,
        _ => Color::White,
    }
}

// ── Animation (smooth progress interpolation) ─────────────────────

/// Smoothly interpolates a progress value towards its target each frame,
/// giving animated gauge movement instead of instantaneous jumps.
#[derive(Debug, Clone)]
struct SmoothProgress {
    current: f64,
    target: f64,
    /// How much of the gap we close per tick (0.1 = 10%).
    lerp_factor: f64,
}

impl SmoothProgress {
    fn new() -> Self {
        Self {
            current: 0.0,
            target: 0.0,
            lerp_factor: 0.25,
        }
    }

    fn set_target(&mut self, target: f64) {
        self.target = target;
    }

    /// Advance the animation by one frame, returning the current ratio.
    fn tick(&mut self) -> f64 {
        let gap = self.target - self.current;
        if gap.abs() < 0.005 {
            self.current = self.target;
        } else {
            self.current += gap * self.lerp_factor;
        }
        self.current
    }

    fn ratio(&self) -> f64 {
        self.current
    }
}

// ── Data types ────────────────────────────────────────────────────

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

#[derive(Debug, Clone)]
struct FileView {
    path: PathBuf,
    status: FileViewStatus,
    stage: Option<Stage>,
    sub_progress: Option<(u64, u64)>,
    sub_label: Option<String>,
    finished: bool,
    /// Animated gauge driven by this file's sub_progress.
    progress_bar: SmoothProgress,
}

#[derive(Debug, Clone)]
struct LogLine {
    level: String,
    target: String,
    message: String,
}

#[derive(Debug, Clone)]
struct GpuJobInfo {
    job_type: String,
    path: PathBuf,
    elapsed_ms: Option<u64>,
    success: Option<bool>,
}

#[derive(Debug, Clone)]
struct GpuStatus {
    current_job: Option<GpuJobInfo>,
    /// Last few completed jobs (newest first).
    last_jobs: VecDeque<GpuJobInfo>,
    /// Track whether a job animation should show a "working" spinner.
    job_spinner_frame: u8,
    /// Animated fill for the GPU job gauge (represents "elapsed time" visually).
    progress: SmoothProgress,
}

impl GpuStatus {
    fn new() -> Self {
        Self {
            current_job: None,
            last_jobs: VecDeque::with_capacity(8),
            job_spinner_frame: 0,
            progress: SmoothProgress::new(),
        }
    }

    fn push_finished(&mut self, info: GpuJobInfo) {
        if self.last_jobs.len() >= 8 {
            self.last_jobs.pop_back();
        }
        self.last_jobs.push_front(info);
    }

    fn tick_animation(&mut self) {
        self.job_spinner_frame = self.job_spinner_frame.wrapping_add(1);
        if self.current_job.is_some() {
            // Oscillate the gauge to show "in progress" visually.
            let phase = (self.job_spinner_frame as f64 * 0.1).sin().abs();
            self.progress.set_target(phase);
        } else {
            self.progress.set_target(0.0);
        }
    }
}

/// Which panel has keyboard focus for scrollable content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FocusPane {
    Files,
    Logs,
}

// ── UI state ──────────────────────────────────────────────────────

#[derive(Debug)]
struct UiState {
    files: Vec<FileView>,
    log_lines: Vec<LogLine>,
    started_at: Option<Instant>,
    python_log_path: Option<PathBuf>,
    gpu: GpuStatus,
    /// Scroll offset for the log pane (0 = newest, grows upward).
    log_scroll: usize,
    focus: FocusPane,
}

const MAX_LOG_LINES: usize = 1000;
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
            progress_bar: SmoothProgress::new(),
        });
        self.files.last_mut().unwrap()
    }

    fn push_log(&mut self, level: String, target: String, message: String) {
        self.log_lines.push(LogLine {
            level,
            target,
            message,
        });
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
                f.sub_progress = None;
                f.sub_label = None;
            }
            ProgressEvent::OcrProgress { path, done, total } => {
                let f = self.upsert_file(&path);
                f.sub_progress = Some((done, total));
                f.sub_label = Some(format!("OCR {done}/{total}"));
                if total > 0 {
                    f.progress_bar.set_target(done as f64 / total as f64);
                }
            }
            ProgressEvent::FetchResult {
                path,
                candidates_found,
                downloaded,
            } => {
                let f = self.upsert_file(&path);
                f.sub_label = Some(format!("fetch: {downloaded}/{candidates_found} dl'd"));
                if candidates_found > 0 {
                    f.progress_bar
                        .set_target(downloaded as f64 / candidates_found as f64);
                }
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
                if lines_total > 0 {
                    f.progress_bar
                        .set_target(lines_done as f64 / lines_total as f64);
                }
            }
            ProgressEvent::TranscribeProgress { path, percent } => {
                let f = self.upsert_file(&path);
                f.sub_label = Some(format!("transcribing… {percent}%"));
                f.progress_bar.set_target(percent as f64 / 100.0);
            }
            ProgressEvent::Log {
                level,
                target,
                message,
            } => {
                self.push_log(level, target, message);
            }
            ProgressEvent::GpuJobStarted { job_type, path } => {
                self.gpu.current_job = Some(GpuJobInfo {
                    job_type,
                    path,
                    elapsed_ms: None,
                    success: None,
                });
                self.gpu.progress = SmoothProgress::new();
            }
            ProgressEvent::GpuJobFinished {
                job_type,
                path,
                elapsed_ms,
                success,
            } => {
                let info = GpuJobInfo {
                    job_type,
                    path,
                    elapsed_ms: Some(elapsed_ms),
                    success: Some(success),
                };
                self.gpu.current_job = None;
                self.gpu.push_finished(info);
            }
            ProgressEvent::FileFinished { path, status } => {
                let f = self.upsert_file(&path);
                f.status = status.into();
                f.stage = None;
                f.sub_progress = None;
                f.sub_label = None;
                f.finished = true;
                f.progress_bar.set_target(1.0);
            }
        }
    }

    /// Advance frame‑based animations (smooth progress, spinner).
    fn tick_animations(&mut self) {
        for f in &mut self.files {
            f.progress_bar.tick();
        }
        self.gpu.tick_animation();
    }
}

// ── Tracing layer ─────────────────────────────────────────────────

/// Tracing layer that forwards events into the TUI's progress channel.
pub struct TuiTracingLayer {
    sender: ProgressSender,
}

impl TuiTracingLayer {
    pub fn new(sender: ProgressSender) -> Self {
        Self { sender }
    }
}

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

// ── Python stderr capture path ────────────────────────────────────

pub fn python_stderr_capture_path(root: &Path) -> PathBuf {
    root.join(".translate_temp").join("python.stderr.log")
}

// ── Terminal guard ────────────────────────────────────────────────

pub struct TerminalGuard {
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

// ── Public API ────────────────────────────────────────────────────

pub struct TuiHandle {
    join: Option<JoinHandle<UiState>>,
    quit_flag: Arc<Mutex<bool>>,
    python_log_path: PathBuf,
}

impl TuiHandle {
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

    pub fn python_log_path(&self) -> &Path {
        &self.python_log_path
    }
}

pub struct JoinedTuiState {
    pub files: Vec<(PathBuf, FileViewStatus)>,
    pub python_log_path: PathBuf,
}

impl JoinedTuiState {}

pub fn finish_glyph(s: FileViewStatus) -> &'static str {
    status_glyph(s).0
}

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

pub fn stdout_is_tty() -> bool {
    io::stdout().is_terminal()
}

/// Spawn the renderer thread.
pub fn spawn_tui(
    mut rx: mpsc::UnboundedReceiver<ProgressEvent>,
    python_log_path: PathBuf,
    force_plain: bool,
) -> TuiHandle {
    let quit_flag = Arc::new(Mutex::new(false));
    let quit_flag_for_thread = quit_flag.clone();
    let python_log_path_clone = python_log_path.clone();

    let interactive = !force_plain && stdout_is_tty();

    let join = thread::spawn(move || {
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

// ── Interactive renderer ─────────────────────────────────────────

fn run_interactive(
    rx: &mut mpsc::UnboundedReceiver<ProgressEvent>,
    quit_flag: Arc<Mutex<bool>>,
    python_log_path: PathBuf,
) -> UiState {
    let mut state = UiState {
        python_log_path: Some(python_log_path),
        files: Vec::new(),
        log_lines: Vec::new(),
        started_at: Some(Instant::now()),
        gpu: GpuStatus::new(),
        log_scroll: 0,
        focus: FocusPane::Logs,
    };

    let guard = match TerminalGuard::enter() {
        Ok(g) => g,
        Err(e) => {
            warn!("TUI init failed ({e}); falling back to plain mode");
            return run_plain(rx);
        }
    };
    let guard = Arc::new(Mutex::new(guard));
    install_panic_hook(guard.clone());

    let tick = Duration::from_millis(50); // 20 FPS animation
    let mut last_render = Instant::now() - tick;

    loop {
        // Drain events (non-blocking).
        let mut received_any = false;
        while let Ok(ev) = rx.try_recv() {
            state.apply(ev);
            received_any = true;
        }

        // Advance frame animations regardless of events (smooth bars/spinner).
        state.tick_animations();

        // Key input (non-blocking poll, 30ms).
        let poll_timeout = if received_any || last_render.elapsed() >= tick {
            Duration::ZERO
        } else {
            tick.saturating_sub(last_render.elapsed())
        };
        if event::poll(poll_timeout).unwrap_or(false)
            && let Ok(Event::Key(k)) = event::read()
            && k.kind == KeyEventKind::Press
        {
            let quit = match k.code {
                KeyCode::Char('q') => true,
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => true,
                KeyCode::Down | KeyCode::Char('j') => {
                    state.log_scroll = state.log_scroll.saturating_add(1);
                    state.focus = FocusPane::Logs;
                    false
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    state.log_scroll = state.log_scroll.saturating_sub(1);
                    state.focus = FocusPane::Logs;
                    false
                }
                KeyCode::PageDown => {
                    state.log_scroll = state.log_scroll.saturating_add(20);
                    state.focus = FocusPane::Logs;
                    false
                }
                KeyCode::PageUp => {
                    state.log_scroll = state.log_scroll.saturating_sub(20);
                    state.focus = FocusPane::Logs;
                    false
                }
                KeyCode::End => {
                    state.log_scroll = usize::MAX;
                    state.focus = FocusPane::Logs;
                    false
                }
                KeyCode::Home => {
                    state.log_scroll = 0;
                    state.focus = FocusPane::Logs;
                    false
                }
                KeyCode::Tab | KeyCode::Char('l') => {
                    state.focus = match state.focus {
                        FocusPane::Files => FocusPane::Logs,
                        FocusPane::Logs => FocusPane::Files,
                    };
                    false
                }
                _ => false,
            };
            if quit {
                *quit_flag.lock().unwrap() = true;
                break;
            }
        }

        // Render at ~20 FPS, plus immediately after event bursts.
        if received_any || last_render.elapsed() >= tick {
            let mut g = guard.lock().unwrap();
            if let Some(term) = g.terminal() {
                let _ = term.draw(|f| render(f, &state));
            }
            drop(g);
            last_render = Instant::now();
        }

        // Channel closed + no events → pipeline done.
        if rx.is_closed() && rx.is_empty() {
            // Final render so we see the last state.
            let mut g = guard.lock().unwrap();
            if let Some(term) = g.terminal() {
                let _ = term.draw(|f| render(f, &state));
            }
            drop(g);
            break;
        }
    }

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

// ── Plain mode ────────────────────────────────────────────────────

fn run_plain(rx: &mut mpsc::UnboundedReceiver<ProgressEvent>) -> UiState {
    let mut state = UiState {
        files: Vec::new(),
        log_lines: Vec::new(),
        started_at: Some(Instant::now()),
        gpu: GpuStatus::new(),
        log_scroll: 0,
        focus: FocusPane::Logs,
        python_log_path: None,
    };
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

// ── Rendering ─────────────────────────────────────────────────────

fn render(frame: &mut ratatui::Frame<'_>, state: &UiState) {
    let area = frame.area();

    // ── Layout ────────────────────────────────────────────────────
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),      // Status bar
            Constraint::Min(6),         // Files + GPU (horizontal split)
            Constraint::Percentage(35), // Logs
            Constraint::Length(1),      // Footer
        ])
        .split(area);

    render_header(frame, chunks[0], state);

    // Middle: horizontal split of Files | GPU
    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[1]);

    render_files(frame, mid_chunks[0], state);
    render_gpu_panel(frame, mid_chunks[1], state);

    render_logs(frame, chunks[2], state);
    render_footer(frame, chunks[3]);
}

// ── Header ────────────────────────────────────────────────────────

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

    // Build summary spans
    let mut spans = vec![
        Span::styled(
            " movie-translator ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("{done}/{total}"),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" done "),
    ];

    if active > 0 {
        spans.push(Span::styled(
            format!("{active} active "),
            Style::default().fg(Color::Cyan),
        ));
    }
    if queued > 0 {
        spans.push(Span::styled(
            format!("{queued} queued "),
            Style::default().fg(Color::DarkGray),
        ));
    }
    if failed > 0 {
        spans.push(Span::styled(
            format!("{failed} failed "),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ));
    }

    // GPU status indicator
    if let Some(job) = &state.gpu.current_job {
        let jc = gpu_job_color(&job.job_type);
        spans.push(Span::styled(
            format!(" ◇ {} ", job.job_type),
            Style::default().fg(jc).add_modifier(Modifier::BOLD),
        ));
    }

    spans.push(Span::raw(format!("  ·  {h:02}:{m:02}:{s:02}")));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let text = Paragraph::new(Line::from(spans)).block(block);
    frame.render_widget(text, area);
}

// ── File table ────────────────────────────────────────────────────

fn render_files(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" Files ({}) ", state.files.len()))
        .border_style(Style::default().fg(Color::DarkGray))
        .title_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::White),
        );

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.files.is_empty() {
        let empty = Paragraph::new("Waiting for files…")
            .style(Style::default().fg(Color::DarkGray))
            .centered();
        frame.render_widget(empty, inner);
        return;
    }

    let max_rows = inner.height.saturating_sub(2) as usize; // header + bottom gauge margin
    if max_rows == 0 {
        return;
    }

    let total = state.files.len();
    let start = total.saturating_sub(max_rows);
    let visible = &state.files[start..];
    let rows: Vec<Row<'_>> = visible
        .iter()
        .map(|f| {
            let (glyph, color) = status_glyph(f.status);

            // Stage label with color
            let stage_style = match f.stage {
                Some(s) => Style::default().fg(stage_color(s)),
                None => Style::default().fg(Color::DarkGray),
            };
            let stage_label = f
                .stage
                .map(|s| s.label().to_string())
                .unwrap_or_else(|| status_name(f.status).to_string());

            let name = f
                .path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            // Animated gauge bar
            let pct = (f.progress_bar.ratio() * 100.0) as u16;

            let detail = f.sub_label.clone().unwrap_or_default();

            Row::new(vec![
                Cell::from(Span::styled(glyph.to_string(), Style::default().fg(color))),
                Cell::from(Span::raw(name)),
                Cell::from(Span::styled(stage_label, stage_style)),
                gauge_cell(pct, f.stage),
                Cell::from(Span::raw(detail)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(2),
        Constraint::Percentage(40),
        Constraint::Length(13),
        Constraint::Length(8),
        Constraint::Min(15),
    ];
    let header = Row::new(vec![
        Cell::from(Span::styled("", Style::default())),
        Cell::from(Span::styled(
            "file",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Cell::from(Span::styled(
            "stage",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Cell::from(Span::styled(
            "progress",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Cell::from(Span::styled(
            "detail",
            Style::default().add_modifier(Modifier::BOLD),
        )),
    ]);

    // Use high-water gauge: area below the last visible row uses the
    // remaining space, but we render the gauge on the file that is
    // currently ACTIVE with sub_progress.
    let _available_rows = max_rows.min(visible.len());
    let gauge_row_idx = visible
        .iter()
        .rposition(|f| matches!(f.status, FileViewStatus::Active) && f.sub_progress.is_some());

    // Table rendering
    let table_rows: Vec<Row<'_>> = if let Some(g_idx) = gauge_row_idx {
        // Build rows but replace the gauge row with one that also has a Gauge widget
        rows.into_iter()
            .enumerate()
            .map(|(i, row)| {
                if i == g_idx {
                    // Already done above
                    row
                } else {
                    row
                }
            })
            .collect()
    } else {
        rows
    };

    let table = Table::new(table_rows, widths)
        .header(header)
        .style(Style::default())
        .column_spacing(1);

    // Render table inside a scrollable area
    let table_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: inner.height.saturating_sub(1), // leave room for gauge
    };
    frame.render_widget(table, table_area);

    // Render a thin animated gauge for the active file at the bottom
    if let Some(active) = state
        .files
        .iter()
        .find(|f| matches!(f.status, FileViewStatus::Active) && f.sub_progress.is_some())
        && let Some((d, t)) = active.sub_progress
        && t > 0
    {
        let ratio = active.progress_bar.ratio().clamp(0.0, 1.0);
        let gauge_color = active.stage.map(stage_color).unwrap_or(Color::Cyan);
        let label = active
            .sub_label
            .clone()
            .unwrap_or_else(|| format!("{d}/{t}"));
        let gauge_area = Rect {
            x: inner.x,
            y: inner.y + inner.height.saturating_sub(1),
            width: inner.width,
            height: 1,
        };
        let g = Gauge::default()
            .gauge_style(
                Style::default()
                    .fg(gauge_color)
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .ratio(ratio)
            .label(label)
            .use_unicode(true);
        frame.render_widget(g, gauge_area);
    }
}

/// Build a small unicode gauge cell for the table row.
fn gauge_cell(pct: u16, stage: Option<Stage>) -> Cell<'static> {
    let color = stage.map(stage_color).unwrap_or(Color::Cyan);
    let bar_char = if pct >= 90 {
        '█'
    } else if pct >= 60 {
        '▓'
    } else if pct >= 30 {
        '▒'
    } else {
        '░'
    };

    // 5 characters of bar
    let full = pct.min(100) as usize / 20; // 0..5
    let bar: String = (0..5)
        .map(|i| if i < full { bar_char } else { '░' })
        .collect();

    Cell::from(Span::styled(
        format!("{} {:>3}%", bar, pct),
        Style::default().fg(color),
    ))
}

// ── GPU Worker Panel ──────────────────────────────────────────────

fn render_gpu_panel(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" GPU ")
        .border_style(Style::default().fg(Color::DarkGray))
        .title_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::LightCyan),
        );

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Current job section
    let (top_area, bottom_area) = {
        let h = inner.height;
        if h < 5 {
            (inner, Rect::default())
        } else {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(1)])
                .split(inner);
            (chunks[0], chunks[1])
        }
    };

    if let Some(job) = &state.gpu.current_job {
        let jc = gpu_job_color(&job.job_type);
        let elapsed = state
            .started_at
            .map(|t| format!("{:.1}s", t.elapsed().as_secs_f64()))
            .unwrap_or_default();

        let top_text = vec![
            Line::from(vec![
                Span::styled("▶ ", Style::default().fg(Color::Green)),
                Span::styled(
                    &job.job_type,
                    Style::default().fg(jc).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(elapsed, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(Span::styled(
                job.path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                Style::default().fg(Color::White),
            )),
        ];
        let p = Paragraph::new(top_text);
        frame.render_widget(p, top_area);

        // Animated progress gauge for active job
        let ratio = state.gpu.progress.ratio();
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(jc).bg(Color::DarkGray))
            .ratio(ratio)
            .label("working…")
            .use_unicode(true);
        frame.render_widget(
            gauge,
            Rect {
                x: top_area.x,
                y: top_area.y + top_area.height.saturating_sub(1),
                width: top_area.width,
                height: 1,
            },
        );
    } else {
        let idle = Paragraph::new(Line::from(vec![Span::styled(
            "● idle",
            Style::default().fg(Color::DarkGray),
        )]));
        frame.render_widget(idle, top_area);
    }

    // Last few jobs history
    if !state.gpu.last_jobs.is_empty() && bottom_area.height > 0 {
        let lines: Vec<Line<'_>> = state
            .gpu
            .last_jobs
            .iter()
            .map(|j| {
                let jc = gpu_job_color(&j.job_type);
                let glyph = match j.success {
                    Some(true) => "✓",
                    Some(false) => "✗",
                    None => "·",
                };
                let glyph_color = match j.success {
                    Some(true) => Color::Green,
                    Some(false) => Color::Red,
                    None => Color::DarkGray,
                };
                let elapsed_str = j
                    .elapsed_ms
                    .map(|ms| {
                        if ms >= 1000 {
                            format!("{:.1}s", ms as f64 / 1000.0)
                        } else {
                            format!("{ms}ms")
                        }
                    })
                    .unwrap_or_default();
                let name = j
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();

                Line::from(vec![
                    Span::styled(glyph, Style::default().fg(glyph_color)),
                    Span::raw(" "),
                    Span::styled(&j.job_type, Style::default().fg(jc)),
                    Span::raw(format!(" {elapsed_str}")),
                    if name.is_empty() {
                        Span::raw("")
                    } else {
                        Span::styled(format!(" ({name})"), Style::default().fg(Color::DarkGray))
                    },
                ])
            })
            .collect();

        let p = Paragraph::new(lines)
            .block(Block::default().title(format!(" Last {} ", state.gpu.last_jobs.len())));
        frame.render_widget(p, bottom_area);
    }
}

// ── Logs ──────────────────────────────────────────────────────────

fn render_logs(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let title = match state.python_log_path.as_ref() {
        Some(p) => format!(" Logs (python → {}) ", p.display()),
        None => " Logs ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title.as_str())
        .border_style(Style::default().fg(if state.focus == FocusPane::Logs {
            Color::Cyan
        } else {
            Color::DarkGray
        }))
        .title_style(Style::default().add_modifier(Modifier::BOLD));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.log_lines.is_empty() {
        let empty = Paragraph::new("Waiting for log output…")
            .style(Style::default().fg(Color::DarkGray))
            .centered();
        frame.render_widget(empty, inner);
        return;
    }

    // Figure out visible range based on scroll offset.
    let max_visible = inner.height as usize;
    // Clamp scroll to valid range.
    let max_scroll = state.log_lines.len().saturating_sub(max_visible);
    let scroll = state.log_scroll.min(max_scroll);

    // We show from (log_lines.len() - max_visible - scroll) to end, scrolling upward.
    let end = state.log_lines.len().saturating_sub(scroll);
    let start = end.saturating_sub(max_visible);

    let lines: Vec<Line<'_>> = state.log_lines[start..end]
        .iter()
        .map(|l| {
            let lc = log_level_color(&l.level);
            Line::from(vec![
                Span::styled(
                    format!("{:5} ", l.level),
                    Style::default().fg(lc).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("[{}] ", l.target),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(&l.message, Style::default().fg(lc)),
            ])
        })
        .collect();

    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    frame.render_widget(p, inner);

    // Scrollbar
    if state.log_lines.len() > max_visible {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"))
            .track_symbol(Some("│"))
            .thumb_symbol("█")
            .style(Style::default().fg(Color::DarkGray));
        let mut scrollbar_state =
            ScrollbarState::new(state.log_lines.len().saturating_sub(max_visible)).position(scroll);
        frame.render_stateful_widget(scrollbar, inner, &mut scrollbar_state);
    }
}

// ── Footer ────────────────────────────────────────────────────────

fn render_footer(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let p = Paragraph::new(Line::from(vec![
        Span::styled(" q", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" quit  "),
        Span::styled("↑↓/j k", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" scroll  "),
        Span::styled("Tab", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" focus pane  "),
        Span::styled("Home/End", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" jump  "),
    ]))
    .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(p, area);
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn setup_state() -> UiState {
        UiState {
            files: Vec::new(),
            log_lines: Vec::new(),
            started_at: Some(Instant::now()),
            gpu: GpuStatus::new(),
            log_scroll: 0,
            focus: FocusPane::Logs,
            python_log_path: None,
        }
    }

    #[test]
    fn ui_state_aggregates_event_sequence() {
        let mut s = setup_state();
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
            model: "mlx".into(),
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

    #[test]
    fn gpu_events_update_status() {
        let mut s = setup_state();
        let p = PathBuf::from("f.mkv");

        s.apply(ProgressEvent::GpuJobStarted {
            job_type: "translate".into(),
            path: p.clone(),
        });
        assert!(s.gpu.current_job.is_some());
        assert_eq!(s.gpu.current_job.as_ref().unwrap().job_type, "translate");

        s.apply(ProgressEvent::GpuJobFinished {
            job_type: "translate".into(),
            path: p.clone(),
            elapsed_ms: 1234,
            success: true,
        });
        assert!(s.gpu.current_job.is_none());
        assert_eq!(s.gpu.last_jobs.len(), 1);
        assert_eq!(s.gpu.last_jobs[0].elapsed_ms, Some(1234));
        assert_eq!(s.gpu.last_jobs[0].success, Some(true));
    }

    #[test]
    fn log_buffer_trims_to_max() {
        let mut s = setup_state();
        for i in 0..(MAX_LOG_LINES + 50) {
            s.apply(ProgressEvent::Log {
                level: "INFO".into(),
                target: "t".into(),
                message: format!("msg{i}"),
            });
        }
        assert_eq!(s.log_lines.len(), MAX_LOG_LINES);
        assert!(s.log_lines.first().unwrap().message.contains("msg50"));
        assert!(
            s.log_lines
                .last()
                .unwrap()
                .message
                .contains(&format!("msg{}", MAX_LOG_LINES + 49))
        );
    }

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

    #[test]
    fn gpu_colors_are_defined_for_all_job_types() {
        for t in &[
            "translate",
            "ocr_pgs",
            "ocr_burned_in",
            "inpaint",
            "hardsub_ocr",
            "transcribe",
        ] {
            let c = gpu_job_color(t);
            assert_ne!(
                c,
                Color::DarkGray,
                "job type {t} should have a specific color"
            );
        }
    }

    #[test]
    fn stage_colors_are_all_distinct() {
        use std::collections::HashSet;
        let stages = [
            Stage::Identify,
            Stage::ExtractRef,
            Stage::Fetch,
            Stage::HardsubOcr,
            Stage::ExtractEnglish,
            Stage::Transcribe,
            Stage::Translate,
            Stage::CreateTracks,
            Stage::Mux,
        ];
        let colors: Vec<Color> = stages.iter().map(|s| stage_color(*s)).collect();
        let unique: HashSet<&Color> = colors.iter().collect();
        assert_eq!(
            unique.len(),
            stages.len(),
            "each stage must have a distinct color"
        );
    }

    #[test]
    fn tracing_layer_forwards_to_channel() {
        use tracing_subscriber::prelude::*;

        let (tx, mut rx) = mpsc::unbounded_channel::<ProgressEvent>();
        let layer = TuiTracingLayer::new(ProgressSender::new(tx));
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "test_target", "hello from {}", "test");
        });

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
        drop(tx);

        let state = run_plain(&mut rx);
        assert_eq!(state.files.len(), 1);
        assert_eq!(state.files[0].status, FileViewStatus::Success);
    }
}
