# anime-dl — Robustness Overhaul Design

**Date:** 2026-07-31
**Status:** Approved
**Supersedes:** [`2026-06-03-anime-downloader-design.md`](./2026-06-03-anime-downloader-design.md)
(original approved design; kept as historical record). The earlier
implementation doc at `docs/anime-downloader-design.md` is archived at
`docs/archive/anime-downloader-design.md` and is stale by comparison.

---

## 1. Overview

`anime-dl` downloads anime episodes from ogladajanime.pl at best available
quality, with Polish hardsubs, from a canonical JSON episode list produced by
the resolver userscript. This overhaul hardens the downloader for real-world
conditions: flaky mirrors, dead hosts, partial/corrupt files, interrupted
runs, and missing system tools.

Goals:

- **Robustness first:** every failure mode has an explicit policy (retry,
  circuit breaker, quarantine, manifest-driven resume). No silent data loss.
- **Deterministic validation:** downloaded files are verified with `ffprobe`
  against explicit minimums; invalid files are quarantined or deleted
  (`--clean-invalid`).
- **Resumable runs:** the manifest records per-episode state; `--resume` and
  `--retry-failed` restart interrupted runs without re-downloading.
- **Host discipline:** per-host concurrency, per-host timeout profiles, and a
  per-host circuit breaker protect slow/failing sources and the user's network.

Non-goals (unchanged from the original spec):

- Translation / OCR / subtitle muxing — that is the `movie-translator`
  pipeline.
- A new resolver mechanism — the userscript (v4) remains the discovery path.
- New host integrations — hosts are curated in the userscript and the host
  policy layer only orders/limits them.

---

## 2. CLI + Exit Codes

```
anime-dl [OPTIONS] [NAME]
```

Flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--input, -i PATH` | — | Canonical JSON episode list. A positional `.json` path is routed identically. |
| `NAME` (positional) | — | Positional `.json` path (convenience, same as `--input`). |
| `--out DIR` | `./<slug>` | Output directory. |
| `--episodes N,N,...` | all | Only download these episode numbers. |
| `--episode-concurrency N` | 4 | Max concurrent episode downloads. |
| `--host-concurrency N` | 1 | Max concurrent downloads from one host. |
| `--ui MODE` | auto | `dashboard` (TTY) or `plain` (piped). Explicit flag wins. |
| `-v` | off | Debug logging for our crates. |
| `--resume` | off | Resume from existing manifest: skip Done, retry Failed. |
| `--retry-failed` | off | Re-run episodes marked Failed in the manifest. |
| `--validate-only` | off | Validate inputs/manifest only; download nothing. |
| `--no-validate` | off | Skip ffprobe validation of downloaded files. |
| `--validate-force` | off | Revalidate even when a valid cached result exists. |
| `--min-size-mb F` | 1.0 | Minimum accepted file size. |
| `--min-duration-secs F` | 1.0 | Minimum accepted media duration. |
| `--require-audio` | off | Audio stream required (default: audio missing is a warning). |
| `--ffprobe-timeout SECS` | 15 | Per-file ffprobe timeout. |
| `--retry-attempts N` | 3 | Transient-failure retry attempts per episode download. |
| `--cb-threshold N` | 3 | Systemic failures before the host circuit breaker opens. |
| `--cb-cooldown-secs SECS` | 60 | Circuit breaker cooldown before half-open retry. |
| `--clean-invalid` | off | Delete quarantined invalid files instead of keeping them. |
| `--manifest PATH` | `<out>/<slug>.anime-manifest.json` | Manifest path override. |
| `--ytdlp-extra-args ARGS` | — | Extra raw arguments passed to each yt-dlp invocation. |

Removed in earlier overhauls (do not reintroduce): `--json`, `--file`,
name-search positional titles, and the alternate dashboard renderers B/C/D.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | All episodes downloaded and validated. |
| 1 | Fatal error (bad input file, I/O, internal). |
| 2 | Usage error (unknown flag, bad value). |
| 3 | Partial success (some episodes done, some failed). |
| 4 | All episodes failed. |
| 130 | Cancelled by user (SIGINT/Ctrl+C). |

---

## 3. Input Schema v2 + Migration

### Canonical v2 (emitted by userscript v4)

```json
{
  "schema_version": 2,
  "source_page": "https://ogladajanime.pl/anime/<slug>",
  "resolved_at": "2026-07-30T12:00:00Z",
  "title": "Boku no Hero Academia 7th Season",
  "episodes": [
    {
      "episode": 1,
      "mirrors": [
        {
          "host": "cda",
          "quality": "1080p",
          "subtitle_group": "MioroSubs",
          "url": "https://..."
        }
      ]
    }
  ]
}
```

- `schema_version`: integer, required. `2` is canonical.
- `source_page` / `resolved_at`: informational; `resolved_at` staleness is
  surfaced as a warning beyond a threshold, never fatal.
- `episodes[].mirrors[]`: canonical per-mirror records (`host`, `quality`,
  `subtitle_group` optional, `url` required). Parsed into a uniform internal
  shape where every episode carries `mirrors` plus a flat `urls` vec kept for
  engine compatibility (`mirrors[].url` order matches `urls`).

### v1 migration

`schema_version: 1` documents (episodes with flat `urls` and optional
`quality` metadata) are accepted and normalized to the v2 shape — flat URLs
become mirrors with the host derived from the URL and null
quality/subtitle_group — with a printed warning that the file should be
re-exported with userscript v4.

### Rejected legacy shapes

Legacy `resolved` / `embed_url` fields are rejected with an actionable error:
state that the file is an old format, that userscript v4 must be re-run to
produce canonical v2 JSON, and (when possible) which flag/file to use instead.
Malformed JSON, empty episode lists, duplicate episode numbers, and empty
mirror/URL lists are rejected at parse time.

Validation (structural, shared by `--validate-only`):

- `schema_version` present and `1` or `2`.
- `episodes` non-empty; each `episode` a unique positive integer.
- each episode has non-empty `mirrors` (or `urls` for v1) with non-empty URL
  strings.

---

## 4. Host Policy Layer

Curated host order (userscript v4 `HOST_PREFERENCE`, mirrored by the Rust
policy layer): **cda first** (best for yt-dlp, PL primary), **rumble second**
(reliable host), then sibnet, vk, mega, ok, dood, myvi, google, hqq, voe,
mp4upload. Unknown hosts sort last.

Per-host behavior:

- **Canonicalization:** `vk` URLs are canonicalized `video_ext.php` →
  `vkvideo.ru` before use.
- **Concurrency:** at most `--host-concurrency` (default 1) active downloads
  per host across all episodes; additional attempts wait in a per-host queue.
- **Timeout profiles:** per-host timeout values tuned for that source's typical
  response/connect behavior; the profile follows the canonical host label.
- **Circuit breaker:** see §7. Breaker state is per-host and **run-scoped only** — it is never persisted in the manifest. A fresh run starts with all circuits closed; `--resume` does not carry breaker state across runs.

---

## 5. Validation + ffprobe

After a download completes, the file is validated before it is accepted.

### Rules

- **Extension allowlist:** `mkv`, `mp4`, `webm`, `flv`, `mov`, `avi`. Anything
  else fails validation (suspect container or HTML error page).
- **Size:** ≥ `--min-size-mb` (default 1.0 MiB). Files below the minimum are
  rejected as truncated/failed.
- **Duration:** ≥ `--min-duration-secs` (default 1.0 s). Placeholder files
  (1-frame, sub-second) fail.
- **Video stream required.** A file with no video stream is rejected.
- **Audio:** warn-only when missing, unless `--require-audio` makes it
  fatal.
- **Placeholder detection:** bogus dimensions (e.g. 0×0 or identical tiny
  dims) and sub-second duration are treated as placeholder artifacts and
  rejected.
- **ffprobe missing:** if ffprobe is not available, validation degrades —
  only extension + size checks apply — with a prominent warning. Never a hard
  failure by default.
- **Timeout:** each ffprobe invocation is capped at `--ffprobe-timeout`
  (default 15 s). A probe that hangs or times out is **not** a hard
  failure: validation degrades to the permissive size + extension heuristic
  with a prominent warning (deliberate policy — never fail open on a single
  probe hiccup).

### Cache

Validation results are cached in the manifest, keyed by file `size + mtime`.
A file whose size/mtime is unchanged reuses the cached verdict
(`--validate-force` bypasses). This makes `--resume` and `--retry-failed`
cheap: already-validated files are not re-probed.

### Flags

- `--no-validate` — skip ffprobe entirely (size/extension still checked by
  default; use with care).
- `--validate-only` — run structural input validation plus ffprobe checks on
  any existing outputs listed in the manifest; download nothing.
- `--validate-force` — ignore cached verdicts.

---

## 6. State Machine + Events

Per-episode state machine:

```
Queued → Inspecting → Measuring → WaitingHost → Downloading → Done
   │          │            │            │               │
   │          │            │            │               └──▶ Failed
   │          │            └── mirror exhausted ────────────▶ Failed
   │          └── all mirrors invalid ──────────────────────▶ Failed
   └── cancelled ───────────────────────────────────────────▶ Cancelled
```

Transitions:

- **Queued** — waiting for a concurrency slot.
- **Inspecting** — mirror list resolved, output path decided
  (`<out>/<slug>-E{NN}`, zero-padded 2).
- **Measuring** — optional probe of mirror viability (kept from earlier
  design; partial measurement files are retained and continued).
- **WaitingHost** — blocked on the per-host concurrency slot.
- **Downloading** — yt-dlp running (via `mt-ml` bridge), progress emitted.
- **Done** — file validated and accepted (or skipped via `--resume`).
- **Failed** — all mirrors exhausted or validation failed after
  `--retry-attempts`.
- **Cancelled** — user interrupt; partial `.part` files are removed.

### Events

Episode progress is broadcast as typed events (progress pct/speed/ETA/
downloaded/total, mirror changes, state transitions). Consumers: dashboard UI,
plain output, and the manifest writer. The manifest is updated on every
terminal transition (Done/Failed/Cancelled) — atomic write, see §8.

---

## 7. Retry + Circuit Breaker

### Retry policy (per download attempt)

- Transient failures only (network errors, yt-dlp exit, host timeout,
  validation-timeout). Permanent errors (bad input, unsupported URL) do not
  retry.
- Backoff: `2s × 2^n + jitter`, capped at 60 s, for `--retry-attempts`
  (default 3) attempts per episode.
- Attempts cycle through remaining mirrors; a mirror that fails transiently is
  skipped for the current episode and tried again on the next retry round.

### Per-host circuit breaker

- Counts systemic failures per host (downloads failing before producing data,
  host-level timeouts, repeated 4xx/5xx).
- Opens at `--cb-threshold` (default 3) consecutive systemic failures: the
  host is excluded from all mirror ordering for `--cb-cooldown-secs`
  (default 60 s), then half-opens (one probe) before closing again.
- **URL-specific failures are excluded** from the systemic counter — a single
  dead video on a healthy host must not trip the breaker.
- While open, episodes skip that host's mirrors and try the next host in
  preference order.

---

## 8. Manifest + Resume / Retry-failed / Validate-only

### Manifest file

Default path: `<out>/<slug>.anime-manifest.json` (`--manifest` overrides).

Contents:

- Input identity (title, sha256 of the source JSON, source JSON path,
  `resolved_at` when present, episode count).
- Per-episode records: state, chosen mirror, output path, size/mtime,
  ffprobe verdict (cached, keyed size+mtime), attempt history (capped).
- Schema version of the manifest itself (migration-safe).

Write semantics:

- **Atomic:** write temp file in the same directory, fsync, rename over the
  target. A partial/corrupt manifest on crash is discarded.
- **Attempt history cap:** each episode keeps at most 8 attempt records
  (`MAX_ATTEMPTS`); the oldest is dropped when exceeded. This caps manifest
  growth on pathological retry storms — it is a history cap, not a
  write-attempt limit.
- Written on every terminal state transition and on clean exit.

Circuit-breaker state is intentionally **not** part of the manifest: it is
run-scoped (see §4/§7), so `--resume` never inherits stale breaker verdicts.

### Semantics

- `--resume`: load manifest; episodes already `Done` with a valid cached
  verdict are skipped (no re-download, no re-probe); `Failed` episodes are
  re-queued; `Cancelled` episodes are re-queued. Output path collisions with
  existing files are validated rather than blindly re-downloaded.
- `--retry-failed`: like `--resume` but only re-queues `Failed` episodes.
- `--validate-only`: load input (+ manifest if present), run structural
  validation and ffprobe on existing outputs; print a per-episode verdict
  table; exit code reflects the aggregate (0 all valid, 3 partial, 4 all
  invalid). Downloads nothing.
- No manifest and no `--resume`/`--retry-failed`: fresh run; existing
  `<slug>-E{NN}` files that match the planned stem are validated and skipped
  if valid (same size+mtime cache semantics), keeping interrupted-run resume
  implicit and safe.

---

## 9. Quarantine

Files that fail validation are handled explicitly:

- Default: moved to `<out>/.quarantine/` (dotdir, matches the engine's
  `quarantine_dir` default) with a reason suffix
  (e.g. `-E01.min-size`, `-E01.no-video`, `-E01.bad-ext`), and the manifest
  records the quarantine path. Nothing is silently deleted.
- `--clean-invalid`: quarantine is skipped and the invalid file is deleted
  immediately (explicit user choice).
- Quarantined files are never re-validated as candidates for `--resume`
  unless re-downloaded (a fresh download replaces the quarantined path).

---

## 10. UI

- **dashboard** (default on TTY): ratatui progress panels — per-episode
  state, active mirror, gauge (pct, speed, ETA, downloaded/total MB), host
  breaker status; footer shows aggregate throughput and counts. Read-only
  display; Ctrl+C cancels.
- **plain** (default when stdout is piped/redirected): timestamped
  `[INFO/WARN/ERROR]` lines, no ANSI codes. Safe for logs and CI.
- `--ui` wins over auto-detection. The alternate B/C/D renderers were removed
  earlier; do not reintroduce.
- Summary line at exit: downloaded / skipped / failed / cancelled counts.

---

## 11. Testing Strategy

Unit + integration tests are **hermetic** — no real network, no real hosts,
no yt-dlp/ffprobe dependency:

- **Hermetic fakes:** a fake yt-dlp subprocess (script that writes a
  configurable file, exits with a configurable code), a fake ffprobe (emits
  canned probe JSON or fails), and fake host responses. This covers the
  retry/backoff ladder, circuit-breaker open/close, quarantine, manifest
  atomicity, and exit-code mapping deterministically and fast.
- **State machine tests:** scripted event sequences drive every transition;
  invalid transitions are rejected.
- **Input schema tests:** v2 canonical, v1 migration (warning + normalization),
  legacy `resolved`/`embed_url` rejection (actionable error), malformed/empty/
  duplicate inputs.
- **Validation tests:** allowlist, min-size, min-duration, placeholder
  rejection, missing-audio warn vs `--require-audio` fatal, ffprobe-missing
  degradation, timeout-as-failure, cache hit/miss/`--validate-force`.
- **Manifest tests:** atomic write, corrupted-manifest recovery, resume/
  retry-failed/validate-only semantics, write-attempts cap.
- **Real-host verification** remains a documented follow-up (see §13):
  `ANIME_DL_LIVE=1` is the planned opt-in for a scripted live pass. No
  code-gated live test exists in this initiative; the default gate stays
  fully hermetic.
- **Gate:** `just check && just test`.

---

## 12. Docs Pointers

- README — "Anime downloader" section: quickstart, flags, exit codes,
  validation, host priority, v2 example.
- `docs/superpowers/specs/2026-06-03-anime-downloader-design.md` — original
  spec (superseded, historical).
- `docs/archive/anime-downloader-design.md` — archived implementation-era
  design (historical, may be stale).
- `scripts/ogladajanime_resolver.user.js` — userscript v4, emits canonical
  v2 JSON (`anime-<slug>.json`, single episodes `anime-<slug>-ep<N>.json`).
- `docs/anime-dl-example.json` — canonical v2 example input.
- Runtime truth: `anime-dl --help`.

---

## 13. Follow-ups (genuine, scoped)

- **Real-host verification run** (`ANIME_DL_LIVE=1`): documented follow-up
  (not code-gated in this initiative) — a scripted live pass
  against real hosts to validate timeout profiles and breaker thresholds on
  production traffic. Not part of the default gate.
- **Future `--file` batch flag:** multi-title batch from a watchlist file was
  removed in the CLI overhaul; a future design may reintroduce it on top of
  the canonical-JSON flow (one JSON per title) without changing the engine.
- **Per-host timeout profile tuning** after live data collection (first item).
