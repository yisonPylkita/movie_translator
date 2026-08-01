# Anime Downloader — Design Document

> **ARCHIVED — 2026-07-31**
> This document is a **historical record** and is no longer authoritative.
> Superseded by two specs:
> [`docs/superpowers/specs/2026-06-03-anime-downloader-design.md`](../superpowers/specs/2026-06-03-anime-downloader-design.md)
> (original approved design) and
> [`docs/superpowers/specs/2026-07-31-anime-dl-robustness-design.md`](../superpowers/specs/2026-07-31-anime-dl-robustness-design.md)
> (current approved design). Content below may be stale — do not implement from
> this file. For current usage see `anime-dl --help` and the README.

> **OBSOLETE — 2026-07-29**
> This document describes the legacy anime-dl architecture (browser-based resolver,
> userscript, `--json`/`--file`/name-search flows). The current CLI accepts only
> canonical JSON via `--input <path>` (or positional `.json`). The resolver
> userscript lives at `scripts/ogladajanime_resolver.user.js` — install in
> Tampermonkey to generate `anime-<slug>.json` files. For current usage see
> `anime-dl --help` and the README.

**Last updated:** 2026-07-28
**Status:** Historical architecture reference · Dashboard UI retained; alternate B/C/D renderers removed

---

## 1. Overview

`anime-dl` is a high-throughput anime episode downloader. It takes a list of
anime titles (from CLI args or a plain-text file), resolves them on
ogladajanime.pl via the browser + Tampermonkey userscript flow, and downloads
episodes at best available quality. No translation, no OCR — pure download.

The downloader is designed for users with **fast, stable connections** where the
bottleneck is almost always **server-side throttling**. The core challenge is
identifying the fastest mirror per episode as quickly as possible and keeping
the pipeline saturated without hammering any single source.

---

## 2. Input Formats

### 2.1 Single title

```bash
just anime-dl "Boku no Hero Academia 7th Season"
```

### 2.2 Batch list (`--file`)

```text
# watchlist.txt — comments and blank lines ignored
One Piece                    # all episodes
Naruto 1                     # episode 1 only
Naruto E02                   # episode 2 only
Bleach S01E03                # episode 3 (season tag ignored)
Attack on Titan              # all episodes
```

Run:

```bash
just anime-dl -- --file watchlist.txt
```

Same-title lines are grouped — browser opens once per anime. Resolver JSON
pickup happens per anime. Downloads for ALL animes happen across the combined
episode pool.

### 2.3 Pre-resolved JSON (`--json`)

Skips browser + userscript. Useful for re-runs.

```bash
just anime-dl -- --json ~/Downloads/anime.json --episodes 20,21
```

### 2.4 Output

Default: `./<slug>/<slug>-E{NN}.mkv`. `--out DIR` overrides root; multi-title
runs nest each anime under `DIR/<slug>/`.

---

## 3. Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Input list  │───▶│  ogladajanime.pl  │───▶│  HardsubPlan        │
│  (file/CLI)  │    │  browser + JSON   │    │  (episodes→mirrors) │
└──────────────┘    └──────────────────┘    └─────────┬───────────┘
                                                      │
                    ┌─────────────────────────────────┘
                    ▼
     ┌──────────────────────────────┐
     │  Episode thread pool         │
     │  (one OS thread per episode) │
     │                              │
     │  ┌─ HostLocks ────────────┐  │
     │  │ per-host Mutex: at most │  │
     │  │ 1 download per source  │  │
     │  └────────────────────────┘  │
     │                              │
     │  ┌─ HostRanker ──────────┐   │
     │  │ EMA bytes/sec per host│   │
     │  │ faster floats to top  │   │
     │  └────────────────────────┘  │
     │                              │
     │  ┌─ Per-episode logic ───┐   │
     │  │ Phase 1: measure all  │   │
     │  │   mirrors 6s → pick   │   │
     │  │   fastest             │   │
     │  │ Phase 2: full download│   │
     │  │   from winner         │   │
     │  └────────────────────────┘  │
     │                              │
     │  Events sent via mpsc::Sender│
     └──────────────┬───────────────┘
                    │ EpEvent channel
                    ▼
     ┌──────────────────────────────┐
     │  ratatui TUI (main thread)   │
     │  — receives EpEvents         │
     │  — renders multi-panel view  │
     │  — 100ms tick, q/Esc to quit │
     └──────────────────────────────┘
```

---

## 4. Two-Phase Download (per episode)

### Phase 1 — Speed measurement (~6 seconds)

All PL-sub mirrors for the episode are tested simultaneously. yt-dlp is spawned
per mirror with `-o /dev/null` — the file is not saved. Progress is read from
stdout, speed parsed, and the yt-dlp process is killed after 4 samples or 6
seconds (whichever comes first). Average bytes/sec is recorded.

**Per-host try-lock:** before measuring, the episode thread tries to acquire the
per-host mutex (`try_lock`). If another episode already holds it (e.g. ep 18 is
already measuring vk), that host is skipped for this episode — it falls through
to the next mirror. This naturally spreads episodes across different sources
during measurement.

### Phase 2 — Full download (fastest mirror)

Mirrors are sorted descending by measured bytes/sec. The fastest mirror is
selected as the **winner**. A full download with progress tracking begins. If
the winner fails mid-download, the code falls back to the next-fastest measured
mirror, then to unmeasured mirrors as a last resort.

**▶ Future optimization:** The measurement download on the winning mirror
already downloaded a partial file. The engine now KEEPS the winner's measurement
file (renames it to `.part` and continues the existing yt-dlp process). yt-dlp
supports HTTP Range resumes; the current approach renames the measurement output
and keeps the child alive. A future improvement could use `--continue` for
cleaner resumption.

---

## 5. Per-Host Concurrency Control

**Rule:** at most one download per host at any time, across all episodes.

Implemented via `HostLocks` — a `HashMap<String, Arc<Mutex<()>>>`. Two operations:

- `with_host(host, f)` — blocking: waits for the host lock, runs `f`.
- `try_with_host(host, f) -> Option<T>` — non-blocking: if the host is free,
  locks it and runs `f`; otherwise returns `None` immediately.

During measurement (Phase 1), `try_with_host` is used — busy hosts are skipped
and the next mirror is tried. During full download (Phase 2), `with_host` is
used — the winner blocks until the host is free, guaranteeing exclusive access.

**Rationale:** Server-side throttling is usually per-connection. Two concurrent
downloads from the same host would split the available bandwidth, making both
slower. Serializing per-host maximizes throughput per source.

---

## 6. Speed Ranking (HostRanker)

`HostRanker` maintains an exponential moving average of bytes/sec per host:

```
new_speed = old_speed × 0.7 + measured_speed × 0.3
```

- Updated after every successful full download.
- Used to pre-sort mirrors before measurement — known-fast hosts get a head
  start in the measurement race.
- Unknown hosts default to −1.0 (sort after known hosts), with original
  `pl_players` order as tiebreaker.
- Persists across episodes within a single run.

---

## 7. Terminal UI

The TUI is built with `ratatui` (already a workspace dependency). It runs on
the main thread while episode threads run in the background.

### 7.1 Measurement phase

```
┌─ Episode 18 — measuring ──────────────────────────────────────────┐
│  ⠿ vk       7.5 MiB/s                                             │
│  ⠿ ok       3.1 MiB/s                                             │
│  ⠿ sibnet   0.8 MiB/s                                             │
└────────────────────────────────────────────────────────────────────┘

┌─ Episode 19 — measuring ──────────────────────────────────────────┐
│  ⠿ vk       6.8 MiB/s                                             │
│  ⠿ ok       2.1 MiB/s                                             │
│    sibnet   busy (ep 18 using it)                                  │
└────────────────────────────────────────────────────────────────────┘
```

- One bordered panel per episode.
- Each mirror shows current measured speed.
- Busy hosts are greyed out with a note.
- The `⠿` spinner indicates an active measurement.

### 7.2 Winner selected — download phase

```
┌─ Episode 18 — vk ─────────────────────────────────────────────────┐
│  ██████████████████░░░░░░░░░░░░  58%                               │
│     7.2 MiB/s    ETA 02:15                                         │
│     248.3 / 417.2 MB                                               │
└────────────────────────────────────────────────────────────────────┘

┌─ Episode 19 — ok ─────────────────────────────────────────────────┐
│  ██████████████████████████████  94%                               │
│     6.4 MiB/s    ETA 00:12                                         │
│     360.1 / 383.0 MB                                               │
└────────────────────────────────────────────────────────────────────┘
```

- Panel title changes to show the selected host.
- Progress bar (ratatui `Gauge`) with percentage.
- **▶ DONE (this session):** Downloaded / total megabytes shown as
  `248.3 / 417.2 MB` on third line in full mode, appended in compact mode.
  Requires `downloaded`/`total` fields in `EpEvent::Progress`.
- Speed and ETA on a separate line.

### 7.3 Completed / Failed

```
┌─ Episode 18 ───────────────────────────────────────────────────────────┐
│  ✓  417.2 MB  (vk)                                                     │
└────────────────────────────────────────────────────────────────────────┘

┌─ Episode 19 ───────────────────────────────────────────────────────────┐
│  ✗  all mirrors failed                                                 │
└────────────────────────────────────────────────────────────────────────┘
```

- Green border + checkmark for success.
- Red border + cross for failure.
- File size and winning host shown.

### 7.4 Interaction

- **No keyboard/mouse input.** The TUI is read-only auto-display.
- Ctrl+C cancels all downloads (handled by the engine via `CancellationToken`).
- TUI auto-exits when all episodes reach Done or Failed state (with 500ms final
  display).
- 100ms refresh tick.
- If episodes exceed viewport height, the display auto-pages every 3 seconds.

---

## 8. Event Flow

Episode threads communicate with the TUI via `broadcast::Sender<EpEvent>`:

```rust
enum EpEvent {
    Measuring  { ep, host },           // started measuring this host
    Measured   { ep, host, bps },      // measurement result
    MirrorBusy { ep, host },           // host locked by another episode
    Winner     { ep, host },           // this host selected for full dl
    Progress   { ep, host, pct, speed, eta, downloaded, total },  // dl progress
    Done       { ep, host, size_mb },  // download completed
    Failed     { ep },                 // all mirrors exhausted
    MirrorDone { ep, host, success },  // measurement ended (killed/failed)
    MeasurementComplete { ep },        // all mirrors measured
    Cancelled  { ep },                 // cancelled via CancellationToken
}
```

Events are broadcast to all consumers (TUI, plain output, etc.) and consumed
by the renderer on its own task.

---

## 9. File Structure

```
crates/mt-cli/src/
├── bin/anime_dl.rs        # CLI entry, args, download orchestration
├── ui_model.rs            # dashboard state reducer
├── ui_render/dashboard.rs # ratatui progress dashboard
└── lib.rs                 # module declaration

crates/mt-ml/src/
└── hardsub.rs             # hardsub_download, hardsub_download_with_progress
                           #   DownloadProgress, parse_progress_line

crates/mt-fetch/src/
└── ogladajanime.rs        # discover, wait_for_resolver_json, HardsubPlan, pl_players
```

---

## 10. Next-Session Requirements

### 10.1 Continue measurement download (no restart) — PARTIALLY IMPLEMENTED

**Current behavior:** Winner's measurement file is kept and renamed to `.part`,
and the existing yt-dlp process continues without restarting. Loser measurement
files are cleaned up.

**Remaining:** Explicit `--continue` / HTTP Range resume for cleaner
resumption when the measurement process exits before download completion. The
current approach keeps the child alive through measurement into download, which
works but could be more robust.

### 10.2 Progress bar — downloaded / total MB (IMPLEMENTED)

Third line in download panel showing `248.3 / 417.2 MB`. Surfaced via
`EpEvent::Progress.downloaded`/`total` fields. Dashboard `render_gauge`
shows gauge, speed/ETA, and MB line when total > 0. Compact mode appends
`dl/total MB` to the one-liner.

### 10.3 Speed in KB/s / MB/s

Parsed by `parse_speed_bps`. Dashboard displays each episode's current speed
and combined active throughput in footer:

```text
     Speed: 7.2 MiB/s
```

### 10.4 Per-host concurrency — formal guarantee

Documented in §5. Already implemented. No changes needed, but verify with a
stress test (many episodes, few unique hosts).

### 10.5 Fancy terminal UI

The ratatui TUI in §7 is implemented. Next session should polish:

- Color gradients on progress bars (cyan → green near completion).
- Smooth spinner animation during measurement.
- Auto-resize handling (episode panels fill terminal width).
- Summary line at the bottom: "3 downloaded · 1 failed · 2 skipped".
- Option to auto-exit on completion vs. wait for keypress.

---

## 11. Canonical JSON Schema (--input flow)

New `--input PATH` CLI arg accepts a JSON file with the following schema:

```json
{
  "title": "string (optional display name)",
  "episodes": [
    {
      "episode": 1,
      "urls": ["https://cdn1.example.com/video.mp4"],
      "quality": {"height": 1080}
    }
  ]
}
```

- `title`: optional display name shown in TUI.
- `episodes`: non-empty array. Each entry must have `episode` (positive int, unique) and `urls` (non-empty array of non-empty strings).
- `quality`: optional object with `height` (u32) for quality-first mirror selection.

Validation rejects: zero-url episodes, empty URL strings, missing episode numbers, duplicate episodes, empty episode array, malformed JSON.

### Quality-first semantics

1. Inspect quality metadata per mirror.
2. Rank: higher height > lower height. Unknown height (0) = lowest.
3. Only mirrors in global maximum quality tier race. All equal → all race.
4. Single mirror in max tier → skip race, go direct.
5. Tie-break: speed → height → host preference rank → URL alphabetically.

### Concurrency defaults

- `--episode-concurrency`: default 4, max simultaneous episode downloads.
- `--host-concurrency`: default 1, max simultaneous downloads from one host.

### Temp / cancellation behavior

- Downloads go to `.part` files, renamed on completion.
- Cancellation kills subprocess, deletes `.part`/loser files.
- CancellationToken wired through all workers.
- SIGINT / Ctrl+C triggers cancel-all.

### `--ui` flag and plain output mode

`--ui <MODE>` supports dashboard (`dashboard`, `a`, or `tui`) and pipe-safe
`plain` output. Alternate B/C/D renderers were removed.

**Default selection:** When stdout is a TTY, dashboard is selected. When stdout
is piped or redirected, `plain` is selected. Explicit `--ui` always wins.

Dashboard footer shows combined current throughput in MiB/s. Plain output writes
timestamped `[INFO/WARN]` lines to stdout without ANSI codes; lag warnings remain
on stderr.

### TUI is read-only

The TUI has no keyboard or mouse input handling. It is a passive progress
display only. All control flows through the download engine's
`CancellationToken` (Ctrl+C triggers cancel-all). The Paused/Resumed event
variants were removed entirely from the type system; the engine no longer
supports per-episode pause/resume.

### Example workflows

```bash
# New --input flow (canonical JSON)
just anime-dl --input docs/anime-dl-example.json --episode-concurrency 6

# Smart routing: positional .json detected as --input
just anime-dl path/to/episodes.json

# Smart routing with dashboard UI
just anime-dl episodes.json --ui a

# Legacy ogladajanime flow (unchanged)
just anime-dl "Boku no Hero Academia"
just anime-dl -- --file watchlist.txt --out ~/Videos/anime

# Pre-resolved JSON (skip browser)
just anime-dl -- --json ~/Downloads/anime.json
```

## 12. Testing

```bash
# Build
just setup

# Single anime with pre-resolved JSON
cargo run --release --bin anime-dl -- --json ~/Downloads/anime.json

# Batch from list file
cargo run --release --bin anime-dl -- --file watchlist.txt --out ~/Videos/anime

# With episode filter
cargo run --release --bin anime-dl -- --json ~/Downloads/anime.json --episodes 20,21
```
