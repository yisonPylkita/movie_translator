//! Dashboard renderer — width-aware pure render + thin ratatui frame.
//!
//! Layouts by terminal width:
//! - `>= 100`: full — per-episode rows with MiB progress bars.
//! - `60..=99`: compact — percent only.
//! - `< 60`: minimal — single summary line + rotating active episode.
//!
//! The render core is a pure function [`render`] over [`UiModel`], shared
//! verbatim by the TUI frame and unit tests. URLs are never rendered.

use std::io;

use ratatui::Frame;
use ratatui::style::{Color, Style};
use ratatui::widgets::Paragraph;
use tokio::sync::broadcast;
use tokio::time::sleep;

use crate::download_types::{EpEvent, Phase};
use crate::plain_output::redact_urls;
use crate::ui_model::FinalSummary;
use crate::ui_model::UiModel;
use crate::ui_render::{Renderer, TerminalGuard};

const TICK: std::time::Duration = std::time::Duration::from_millis(100);

/// Width-based dashboard layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardLayout {
    /// `width >= 100`: per-episode rows with MiB progress bars.
    Full,
    /// `60..=99`: percent only.
    Compact,
    /// `< 60`: single summary line + rotating active episode.
    Minimal,
}

impl DashboardLayout {
    pub fn from_width(width: usize) -> Self {
        if width >= 100 {
            Self::Full
        } else if width >= 60 {
            Self::Compact
        } else {
            Self::Minimal
        }
    }
}

/// Pure, width-aware render of the dashboard as plain text lines.
///
/// No ANSI escapes, no terminal state — safe for tests. The TUI frame draws
/// these lines; a paragraph clips overflow rows and columns.
pub fn render(model: &UiModel, width: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    match DashboardLayout::from_width(width) {
        DashboardLayout::Full => render_full(model, width, &mut lines),
        DashboardLayout::Compact => render_compact(model, width, &mut lines),
        DashboardLayout::Minimal => render_minimal(model, width, &mut lines),
    }
    if let Some(summary) = &model.final_summary {
        lines.push(String::new());
        lines.push(format!(
            "Downloaded {}, Skipped {}, Failed {}, Cancelled {}",
            summary.downloaded, summary.skipped, summary.failed, summary.cancelled
        ));
        // Authoritative per-episode reasons (engine `Outcome`), replacing the
        // best-effort event-derived ones. Host is best-effort from episode
        // state; reasons are sanitized by the engine and URL-redacted here.
        for (ep, reason) in &summary.per_episode_reasons {
            let host = model
                .episodes
                .iter()
                .find(|e| e.number == *ep as i64)
                .and_then(|e| e.host.as_deref())
                .unwrap_or("?");
            let host = truncate(host, 12);
            let reason = truncate(&redact_urls(reason), 40);
            lines.push(format!("E{ep:02} {host}: {reason}"));
        }
    }
    lines.join("\n")
}

// ── Layouts ───────────────────────────────────────────────────────────────

fn render_full(model: &UiModel, width: usize, lines: &mut Vec<String>) {
    let title = model.title.as_deref().unwrap_or("anime-dl");
    let max_title = (width / 4).max(8);
    // Whole title bar must stay within width/4 (contract: title <= width/4).
    let bar = format!(" {} ", truncate(title, max_title.saturating_sub(2)));
    lines.push(truncate(&bar, max_title));
    for ep in &model.episodes {
        lines.push(truncate(
            &episode_row_full(ep, width, model.final_summary.as_ref()),
            width,
        ));
    }
    lines.push(summary_line(model));
    for c in &model.circuits {
        if c.open {
            lines.push(truncate(
                &format!(
                    "circuit open {} {}s",
                    truncate(&c.host, 12),
                    c.remaining_secs()
                ),
                width,
            ));
        }
    }
}

fn render_compact(model: &UiModel, width: usize, lines: &mut Vec<String>) {
    for ep in &model.episodes {
        lines.push(truncate(
            &episode_row_compact(ep, width, model.final_summary.as_ref()),
            width,
        ));
    }
    lines.push(summary_line(model));
    for c in &model.circuits {
        if c.open {
            lines.push(truncate(
                &format!(
                    "circuit open {} {}s",
                    truncate(&c.host, 12),
                    c.remaining_secs()
                ),
                width,
            ));
        }
    }
}

fn render_minimal(model: &UiModel, width: usize, lines: &mut Vec<String>) {
    lines.push(truncate(&summary_line(model), width));
    // Rotating active episode: deterministic rotation by event counter.
    let active: Vec<&crate::ui_model::EpState> = model
        .episodes
        .iter()
        .filter(|e| !e.phase.is_terminal())
        .collect();
    if !active.is_empty() {
        let idx = (model.events_since_start as usize) % active.len();
        lines.push(truncate(
            &episode_row_compact(active[idx], width, model.final_summary.as_ref()),
            width,
        ));
    } else if let Some(last) = model.episodes.last() {
        // All terminal: show the last episode's final state.
        lines.push(truncate(
            &episode_row_compact(last, width, model.final_summary.as_ref()),
            width,
        ));
    }
}

// ── Row rendering ─────────────────────────────────────────────────────────

/// Authoritative engine reason for an episode, if the final summary has one.
fn authoritative_reason(summary: Option<&FinalSummary>, ep_number: i64) -> Option<&str> {
    summary.and_then(|s| {
        s.per_episode_reasons
            .iter()
            .find(|(ep, _)| *ep as i64 == ep_number)
            .map(|(_, reason)| reason.as_str())
    })
}

fn episode_row_full(
    ep: &crate::ui_model::EpState,
    width: usize,
    summary: Option<&FinalSummary>,
) -> String {
    let (glyph, _) = phase_glyph(&ep.phase);
    let num = format!("Ep {:>3}", ep.number);
    match &ep.phase {
        Phase::Downloading {
            pct,
            speed,
            eta,
            downloaded,
            total,
        } => {
            let mb = if *total > 0 {
                let dl = *downloaded as f64 / 1_048_576.0;
                let tot = *total as f64 / 1_048_576.0;
                format!("{dl:.1}/{tot:.1} MiB ({pct:.0}%)")
            } else {
                format!("{pct:.1}%")
            };
            let host = truncate(ep.winner.as_deref().unwrap_or(""), 12);
            format!(" {num} {glyph} {host:<12} {mb:<20} {speed:<10} ETA {eta}")
        }
        Phase::Done { host, size_mb } => {
            let host = truncate(host, 12);
            format!(" {num} {glyph} {host:<12} {size_mb:.1} MiB")
        }
        Phase::Failed => {
            // Authoritative engine reason wins; fall back to best-effort
            // event reason, then the generic default. URL-redacted before
            // truncation (URLs never rendered, contract).
            let reason = truncate(
                &redact_urls(
                    authoritative_reason(summary, ep.number)
                        .or(ep.reason.as_deref())
                        .unwrap_or("all mirrors failed"),
                ),
                40,
            );
            format!(" {num} {glyph} {reason}")
        }
        Phase::Queued => {
            if let Some(status) = &ep.status_line {
                format!(" {num} {glyph} {}", truncate_status_line(status, 60))
            } else {
                format!(" {num} {glyph} queued")
            }
        }
        _ => {
            // Measuring / Inspecting / WaitingHost / Cancelled: status line.
            let status = ep
                .status_line
                .as_deref()
                .unwrap_or_else(|| phase_default_text(&ep.phase));
            let _ = width;
            format!(" {num} {glyph} {}", truncate_status_line(status, 60))
        }
    }
}

fn episode_row_compact(
    ep: &crate::ui_model::EpState,
    width: usize,
    summary: Option<&FinalSummary>,
) -> String {
    let (glyph, _) = phase_glyph(&ep.phase);
    let num = format!("Ep {:>3}", ep.number);
    match &ep.phase {
        Phase::Downloading { pct, .. } => {
            format!(" {num} {glyph} {pct:>5.1}%")
        }
        Phase::Done { .. } => {
            let host = truncate(ep.host.as_deref().unwrap_or("?"), 12);
            format!(" {num} {glyph} {host}")
        }
        Phase::Failed => format!(
            " {num} {glyph} {}",
            truncate(
                &redact_urls(
                    authoritative_reason(summary, ep.number)
                        .or(ep.reason.as_deref())
                        .unwrap_or("all mirrors failed"),
                ),
                40,
            )
        ),
        Phase::Queued => {
            if let Some(status) = &ep.status_line {
                format!(
                    " {num} {glyph} {}",
                    truncate_status_line(status, width.saturating_sub(12))
                )
            } else {
                format!(" {num} {glyph} queued")
            }
        }
        _ => {
            let status = ep
                .status_line
                .as_deref()
                .unwrap_or_else(|| phase_default_text(&ep.phase));
            format!(
                " {num} {glyph} {}",
                truncate_status_line(status, width.saturating_sub(12))
            )
        }
    }
}

/// Truncate a status line's variable segments per contract:
/// host/mirror <= 12 chars, reason <= 40 chars; URLs always redacted.
fn truncate_status_line(status: &str, width: usize) -> String {
    let redacted = redact_urls(status);
    if let Some(rest) = redacted.strip_prefix("waiting host ")
        && let Some((host, tail)) = rest.rsplit_once(' ')
    {
        return format!("waiting host {} {}", truncate(host, 12), tail);
    }
    if let Some(rest) = redacted.strip_prefix("retry ")
        && let Some(idx) = rest.find(" attempt ")
    {
        let (mirror, tail) = rest.split_at(idx);
        return format!("retry {} {}", truncate(mirror, 12), tail.trim_start());
    }
    if let Some(rest) = redacted.strip_prefix("validation failed: ") {
        return format!("validation failed: {}", truncate(rest, 40));
    }
    truncate(&redacted, width)
}

fn phase_default_text(phase: &Phase) -> &'static str {
    match phase {
        Phase::Measuring => "measuring mirrors",
        Phase::Inspecting => "inspecting formats",
        Phase::WaitingHost => "waiting host",
        Phase::Cancelled => "cancelled",
        _ => "",
    }
}

fn summary_line(model: &UiModel) -> String {
    let active = model.active_count();
    format!(
        " {} done · {} active · {} failed · Speed: {:.1} MiB/s",
        model.total_done,
        active,
        model.total_failed,
        model.active_bps() / 1_048_576.0,
    )
}

fn phase_glyph(phase: &Phase) -> (&str, Color) {
    match phase {
        Phase::Queued | Phase::WaitingHost => ("·", Color::DarkGray),
        Phase::Inspecting | Phase::Measuring => ("?", Color::Yellow),
        Phase::Downloading { .. } => ("↓", Color::Cyan),
        Phase::Done { .. } => ("✓", Color::Green),
        Phase::Failed => ("✗", Color::Red),
        Phase::Cancelled => ("⨯", Color::DarkGray),
    }
}

/// Truncate to `max` chars, appending `…` when cut.
fn truncate(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        s.to_string()
    } else {
        let keep = max.saturating_sub(1);
        let cut: String = s.chars().take(keep).collect();
        format!("{cut}…")
    }
}

// ── TUI frame wrapper ─────────────────────────────────────────────────────

pub struct DashboardRenderer {
    rx: broadcast::Receiver<EpEvent>,
    model: UiModel,
}

impl DashboardRenderer {
    pub fn new(rx: broadcast::Receiver<EpEvent>, model: UiModel) -> Self {
        Self { rx, model }
    }

    fn draw_frame(f: &mut Frame, model: &UiModel) {
        let width = f.area().width as usize;
        let text = render(model, width);
        let para = Paragraph::new(text).style(Style::default().fg(Color::White));
        f.render_widget(para, f.area());
    }
}

#[async_trait::async_trait]
impl Renderer for DashboardRenderer {
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
                term.draw(|f| Self::draw_frame(f, &model))?;
            }

            if model.all_terminal() || model.final_summary.is_some() {
                sleep(std::time::Duration::from_millis(500)).await;
                // Final draw
                if let Some(term) = guard.terminal() {
                    term.draw(|f| Self::draw_frame(f, &model))?;
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
    use super::*;
    use crate::download_types::EpEvent;

    fn make_model(eps: &[i64]) -> UiModel {
        UiModel::new(eps)
    }

    fn downloading_model(eps: &[i64]) -> UiModel {
        let mut m = make_model(eps);
        m.apply(&EpEvent::Winner {
            ep: 1,
            host: "cda.pl".into(),
        });
        m.apply(&EpEvent::Progress {
            ep: 1,
            host: "cda.pl".into(),
            pct: 27.3,
            speed: "5.2 MiB/s".into(),
            eta: "30s".into(),
            downloaded: 12_900_000,
            total: 47_800_000,
        });
        m
    }

    // ── Golden width layouts ─────────────────────────────────────────────

    #[test]
    fn layout_from_width_boundaries() {
        assert_eq!(DashboardLayout::from_width(100), DashboardLayout::Full);
        assert_eq!(DashboardLayout::from_width(99), DashboardLayout::Compact);
        assert_eq!(DashboardLayout::from_width(60), DashboardLayout::Compact);
        assert_eq!(DashboardLayout::from_width(59), DashboardLayout::Minimal);
        assert_eq!(DashboardLayout::from_width(120), DashboardLayout::Full);
    }

    #[test]
    fn render_full_120() {
        let m = downloading_model(&[1]);
        let out = render(&m, 120);
        assert!(out.contains("Ep   1"), "episode row present");
        assert!(
            out.contains("12.3/45.6 MiB (27%)"),
            "MB progress format: {out}"
        );
        assert!(out.contains("MiB/s"), "speed present");
        assert!(out.contains("done"), "summary present");
    }

    #[test]
    fn render_compact_80() {
        let m = downloading_model(&[1]);
        let out = render(&m, 80);
        assert!(out.contains("Ep   1"), "episode row present");
        assert!(out.contains("27.3%"), "percent-only in compact");
        assert!(
            !out.lines().any(|l| l.contains("MiB (")),
            "no MB progress bar in compact: {out}"
        );
        assert!(
            !out.lines()
                .any(|l| l.starts_with("Ep") && l.contains("MiB/s")),
            "no speed in compact episode rows: {out}"
        );
    }

    #[test]
    fn render_minimal_40() {
        let m = downloading_model(&[1]);
        let out = render(&m, 40);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2, "minimal = summary + active episode");
        assert!(lines[0].contains("done"), "summary line");
        assert!(lines[1].contains("27.3%"), "active episode line");
    }

    #[test]
    fn minimal_rotates_active_episode() {
        let mut m = make_model(&[1, 2, 3]);
        m.apply(&EpEvent::Winner {
            ep: 1,
            host: "a.pl".into(),
        });
        m.apply(&EpEvent::Progress {
            ep: 1,
            host: "a.pl".into(),
            pct: 10.0,
            speed: "1 MiB/s".into(),
            eta: "1m".into(),
            downloaded: 1,
            total: 10,
        });
        m.apply(&EpEvent::Measuring {
            ep: 2,
            host: "b.pl".into(),
        });
        // events: Winner(1), Progress(1), Measuring(2) → events_since_start=3;
        // active = ep1..=ep3 → idx 3 % 3 = 0 → ep1
        let out = render(&m, 40);
        assert!(out.contains("10.0%"), "rotated to ep1: {out}");
    }

    // ── Truncation ───────────────────────────────────────────────────────

    #[test]
    fn truncates_long_host() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::MirrorBusy {
            ep: 1,
            host: "very-long-hostname.example.com".into(),
            wait_secs: 5,
        });
        let out = render(&m, 120);
        assert!(
            out.contains("very-long-h…"),
            "host truncated to 12 chars + ellipsis: {out}"
        );
        assert!(!out.contains("very-long-hostname.example.com"));
    }

    #[test]
    fn truncates_long_title_to_quarter_width() {
        let long = "A".repeat(200);
        let m = UiModel::new_with_options(&[1], Some(long), 60);
        let out = render(&m, 120);
        // max_title = 120/4 = 30 → 29 chars + ellipsis
        assert!(out.lines().next().unwrap().chars().count() <= 30, "{out}");
    }

    #[test]
    fn truncates_long_reason() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Failed { ep: 1 });
        m.apply(&EpEvent::FinalSummary {
            downloaded: 0,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(1, "r".repeat(80))],
        });
        let out = render(&m, 120);
        let failed_line = out.lines().find(|l| l.starts_with("E01")).unwrap();
        assert!(
            failed_line.chars().count() <= 50,
            "reason <= 40: {failed_line}"
        );
    }

    #[test]
    fn failed_row_reason_url_redacted_before_truncation() {
        // Failed rows (full + compact layouts) must redact URLs in the
        // rendered reason — same contract as the summary block.
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Failed { ep: 1 });
        m.apply(&EpEvent::FinalSummary {
            downloaded: 0,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(
                1,
                "http 403 from https://cdn.example.com/v1?token=SECRET".into(),
            )],
        });
        let full = render(&m, 120);
        assert!(!full.contains("https://"), "URL never rendered: {full}");
        assert!(!full.contains("SECRET"), "token never rendered: {full}");
        let compact = render(&m, 80);
        assert!(
            !compact.contains("https://"),
            "URL never rendered: {compact}"
        );
        assert!(
            !compact.contains("SECRET"),
            "token never rendered: {compact}"
        );
    }

    #[test]
    fn failed_row_without_summary_redacts_event_reason() {
        // No final summary: best-effort event reason path must redact too.
        // ValidationResult seeds the reason; Failed then retains it.
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::ValidationResult {
            ep: 1,
            ok: false,
            reason: Some("host https://cda.pl/video/x reject".into()),
        });
        m.apply(&EpEvent::Failed { ep: 1 });
        let out = render(&m, 120);
        assert!(!out.contains("https://"), "URL never rendered: {out}");
        assert!(!out.contains("cda.pl"), "URL host never rendered: {out}");
    }

    #[test]
    fn urls_never_rendered() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::RetryWait {
            ep: 1,
            mirror: "https://cda.pl/video/abc".into(),
            attempt: 1,
            backoff_secs: 2,
        });
        let out = render(&m, 120);
        assert!(!out.contains("https://"), "URLs never rendered: {out}");
        assert!(!out.contains("http://"), "URLs never rendered: {out}");
    }

    // ── State rendering ──────────────────────────────────────────────────

    #[test]
    fn renders_waiting_host_state() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::MirrorBusy {
            ep: 1,
            host: "sibnet.ru".into(),
            wait_secs: 5,
        });
        let out = render(&m, 120);
        assert!(out.contains("waiting host sibnet.ru 5s"), "{out}");
    }

    #[test]
    fn renders_retry_wait_state() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::RetryWait {
            ep: 1,
            mirror: "cda.pl".into(),
            attempt: 2,
            backoff_secs: 4,
        });
        let out = render(&m, 120);
        assert!(out.contains("retry cda.pl attempt 2 in 4s"), "{out}");
    }

    #[test]
    fn renders_validation_result() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::ValidationResult {
            ep: 1,
            ok: false,
            reason: Some("too short".into()),
        });
        let out = render(&m, 120);
        assert!(out.contains("validation failed: too short"), "{out}");
    }

    #[test]
    fn renders_circuit_open() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::CircuitOpened {
            host: "cda.pl".into(),
        });
        let out = render(&m, 120);
        assert!(out.contains("circuit open cda.pl 60s"), "{out}");
    }

    #[test]
    fn renders_final_summary_block() {
        let mut m = make_model(&[1, 2]);
        m.apply(&EpEvent::Winner {
            ep: 1,
            host: "cda.pl".into(),
        });
        m.apply(&EpEvent::Failed { ep: 1 });
        m.apply(&EpEvent::Done {
            ep: 2,
            host: "cda.pl".into(),
            size_mb: 250.0,
        });
        m.apply(&EpEvent::FinalSummary {
            downloaded: 1,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(1, "invalid media: corrupt".into())],
        });
        let out = render(&m, 120);
        assert!(
            out.contains("Downloaded 1, Skipped 0, Failed 1, Cancelled 0"),
            "{out}"
        );
        assert!(
            out.contains("E01 cda.pl: invalid media: corrupt"),
            "per-failed line: {out}"
        );
    }

    #[test]
    fn final_summary_prefers_authoritative_reasons() {
        // Event-derived best-effort reason differs from the authoritative
        // engine reason → the final block must render the authoritative one.
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Failed { ep: 1 });
        m.apply(&EpEvent::FinalSummary {
            downloaded: 0,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(1, "timeout".into())],
        });
        let out = render(&m, 120);
        assert!(
            out.contains("E01 ?: timeout"),
            "authoritative reason: {out}"
        );
        assert!(
            !out.contains("all mirrors failed"),
            "best-effort reason must not leak into final block: {out}"
        );
    }

    #[test]
    fn final_summary_reasons_redacted() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::FinalSummary {
            downloaded: 0,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(
                1,
                "http 403 on https://cdn.example.com/v?token=SECRET".into(),
            )],
        });
        let out = render(&m, 120);
        assert!(!out.contains("https://"), "URLs never rendered: {out}");
        assert!(!out.contains("SECRET"), "token redacted: {out}");
    }

    #[test]
    fn failed_without_host_uses_placeholder() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Failed { ep: 1 });
        m.apply(&EpEvent::FinalSummary {
            downloaded: 0,
            skipped: 0,
            failed: 1,
            cancelled: 0,
            per_episode_reasons: vec![(1, "no output file".into())],
        });
        let out = render(&m, 120);
        assert!(out.contains("E01 ?: no output file"), "{out}");
    }

    #[test]
    fn done_row_shows_size() {
        let mut m = make_model(&[1]);
        m.apply(&EpEvent::Done {
            ep: 1,
            host: "cda.pl".into(),
            size_mb: 250.0,
        });
        let out = render(&m, 120);
        assert!(out.contains("250.0 MiB"), "{out}");
    }

    // ── No terminal APIs in render path ──────────────────────────────────

    #[test]
    fn no_input_apis() {
        let src = include_str!("dashboard.rs");
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
}
