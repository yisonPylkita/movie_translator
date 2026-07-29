# Anime Downloader — Design Document

> **OBSOLETE — 2025-07-29**
> This document describes the legacy anime-dl architecture (browser-based resolver,
> userscript, `--json`/`--file`/name-search flows). The current CLI accepts only
> canonical JSON via `--input <path>` (or positional `.json`). The resolver
> userscript lives at `scripts/ogladajanime_resolver.user.js` — install in
> Tampermonkey to generate `anime-<slug>.json` files. For current usage see
> `anime-dl --help` and the README.

**Last updated:** 2026-07-28
**Status:** Phase 1 implemented + stabilized · Phase 2–3 specified below · Visual renderers A/B/C/D deferred

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

**▶ TODO (next session):** The measurement download on the winning mirror
already downloaded ~6 seconds of data. Instead of restarting the download from
scratch, continue from the partial file. yt-dlp supports resuming partial
downloads via HTTP Range requests — the winner's temp file from Phase 1 should
be kept and resumed, not discarded.

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
- **▶ TODO (next session):** Add line showing downloaded / total megabytes
  (e.g. `248.3 / 417.2 MB`). This requires plumbing the total file size
  through the progress events — yt-dlp reports it in the first progress line
  (`of ~498.00MiB`).
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
├── tui_download.rs        # ratatui-based multi-episode progress TUI
└── lib.rs                 # module declaration

crates/mt-ml/src/
└── hardsub.rs             # hardsub_download, hardsub_download_with_progress
                           #   DownloadProgress, parse_progress_line

crates/mt-fetch/src/
└── ogladajanime.rs        # discover, wait_for_resolver_json, HardsubPlan, pl_players
```

---

## 10. Next-Session Requirements

### 10.1 Continue measurement download (no restart)

**Current behavior:** Phase 1 downloads to `/dev/null`, discarding all data.
Phase 2 starts a fresh download from scratch.

**Desired behavior:** During Phase 1, the winning mirror's partial download file
is kept. In Phase 2, yt-dlp resumes from where Phase 1 left off (using HTTP
Range requests / `--continue` flag). Non-winning mirrors' temp files are
deleted as before.

This eliminates redundant data transfer on the winning mirror — the ~6 seconds
of measurement counts toward the full download.

### 10.2 Progress bar — downloaded / total MB

Add a third line to the download panel showing:

```
     248.3 / 417.2 MB
```

Requires surfacing `total_bytes` through the `EpEvent::Progress` variant.
yt-dlp reports the total file size in the first progress line
(`of ~498.00MiB`). The `DownloadProgress` struct in `mt-ml` already parses
this; it just needs to be plumbed through to the TUI.

### 10.3 Speed in KB/s / MB/s

Already parsed (`parse_speed_bps` in `anime_dl.rs`). The TUI currently shows
the raw speed string from yt-dlp. Next step: display it on the progress panel
with appropriate unit scaling:

```
     7.2 MiB/s     (or 7380 KiB/s for slow connections)
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

A `--ui <MODE>` flag selects the output renderer. All five modes with their aliases:

| Alias | Full name   | Description                          |
|-------|-------------|--------------------------------------|
| a     | dashboard   | Pinned header/footer, row per ep     |
| b     | timeline    | Row-per-ep timeline with stage glyphs |
| c     | scoreboard  | Compact multi-column auto-paging grid |
| d     | stream      | Styled recent event stream            |
| tui   | dashboard   | Legacy compatibility alias            |
| plain | plain       | Pipe-safe structured log lines        |

**Default selection:** When stdout is a TTY, the default is `dashboard` (alias `a`
or `tui`). When stdout is not a TTY (piped, redirected), `plain` is auto-selected.
Explicit `--ui` always wins over auto-detection.

The plain renderer writes timestamped `[INFO/WARN]` lines to stdout with no
ANSI codes, suitable for logging or CI. Warnings about lagged events remain on
stderr.

### Extension seam

Renderers implement an `EpEvent` consumer trait. New renderers can be plugged
in via the `--ui <name>` flag without modifying engine code.

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
