# anime-dl — first-class ogladajanime season downloader

**Date:** 2026-06-03
**Status:** Approved (brainstorming) → implementation

## Goal

Pass an anime name, get its episodes on disk. The tool finds the anime on
ogladajanime.pl, opens the browser, lets the existing resolver userscript
enumerate the whole season, then downloads every episode locally at best
available quality. No translation, no OCR, no muxing — pure download.

This promotes plumbing already built for `--hardsub-ocr` (which downloads a
low-res copy to OCR burned-in subs) into a standalone download tool.

## Non-goals

- Translation / OCR / subtitle muxing (that's the existing `run` pipeline).
- Episode selection / ranges — always download the full season.
- A new resolver mechanism — the existing userscript already does whole-season
  discovery and is reused unchanged.

## Interface

A second binary target in the `mt-cli` crate (NOT a new crate, NOT a subcommand
of `movie-translator`):

```
crates/mt-cli/src/bin/anime_dl.rs   → binary `anime-dl`
```

`mt-cli` already depends on `mt-fetch` + `mt-ml`, so the new binary inherits the
PyO3/libpython build contract (`PYO3_PYTHON` = `.venv/bin/python`) with zero new
Cargo wiring.

### CLI

```
anime-dl "<anime name>" [--out DIR] [--downloads-dir DIR] [--timeout SECS] [--json PATH]
```

- `name` (positional, required) — anime title; slugified for discovery.
- `--out DIR` — output directory. Default: `./<slug>`.
- `--downloads-dir DIR` — where to watch for the userscript JSON.
  Default: `ogladajanime::default_downloads_dir()`.
- `--timeout SECS` — how long to wait for the resolver JSON. Default: 600.
- `--json PATH` — bypass discover + browser; parse an existing resolver JSON.
  For cheap re-runs and testing.

## Architecture

Standalone binary. No tokio GPU worker, no ratatui TUI — downloads are
network/CPU only (`hardsub_download` is explicitly safe to call off the GPU
worker). The binary initializes the embedded Python interpreter the same way
`mt-cli`'s `main` does (so `mt_ml::hardsub_download` can reach yt-dlp), then runs
a simple sequential orchestration loop.

### Reused unchanged (`mt_fetch::ogladajanime`)

| Function | Role |
| --- | --- |
| `discover(title)` | slug guess → `Found{slug,url}` or `Search{url}` |
| `open_in_browser(url)` | launch the user's browser |
| `default_downloads_dir()` | where the userscript JSON lands |
| `wait_for_resolver_json(slug, since, dir, timeout, poll)` | poll for finished JSON |
| `parse_plan(path, fallback_slug)` | JSON → `HardsubPlan` |
| `HardsubPlan::pl_players(ep)` | PL players, best-first (mirror fallback) |

The resolver userscript (`scripts/ogladajanime_resolver.user.js`) is reused
as-is: it already enumerates the full season and keeps the best PL player per
translation group, which is exactly what a downloader wants.

### Data flow

1. Parse args; init embedded interpreter.
2. Resolve the plan:
   - `--json PATH` given → `parse_plan(PATH, "")`.
   - else `discover(name)`:
     - `Found{slug,url}` → `open_in_browser(url)`.
     - `Search{url}` → `open_in_browser(url)`; print instructions to navigate to
       the anime page and run the userscript.
   - stamp `since = SystemTime::now()` **before** opening the browser, then
     `wait_for_resolver_json(...)` → `parse_plan(...)`.
3. For each episode in `sorted(plan.episodes.keys())`:
   - stem = `<out>/<slug>-E{NN}` (zero-padded to 2).
   - if a non-empty file matching the stem already exists → skip (resume), log it.
   - else walk `plan.pl_players(ep)` best-first: call
     `mt_ml::hardsub_download(embed_url, stem, /*min_height*/ 0, /*best*/ true, None)`;
     first success wins; a failed mirror falls through to the next.
   - if every mirror fails → record the failure, continue (never abort the
     season on one bad episode).
4. Print a summary: downloaded / skipped / failed counts. Exit nonzero if any
   episode failed all mirrors.

### Output / naming

`<out>/<slug>-E{NN}.<ext>`. Best-quality selection merges separate video+audio
tracks, so the container ext is decided by yt-dlp (`%(ext)s`, typically `.mkv`).
The Python layer returns the actual written path; Rust propagates it.

## Best-quality download (B1)

`movie_translator/hardsub/download.py::download_episode` gains a `best: bool`
parameter (default `False`, preserving the OCR path exactly):

- `best=False` (OCR, unchanged): selector `bv*[height>=H]+ba/b[height>=H]/bv*+ba/b`
  with ascending `format_sort=['+size','+res']` → smallest legible copy.
- `best=True` (download): selector `bv*+ba/b`, no ascending sort (yt-dlp's
  default already prefers best). outtmpl becomes `{stem}.%(ext)s` (caller passes a
  stem; the suffix is stripped). Returns the real written path.

Threaded through the bridge:

- `mt_ml::hardsub_download(embed_url, out_path, min_height, best, referer)` —
  add the `best: bool` param.
- `mt-ml` backend PyO3 wrapper passes `best` to Python.
- The existing OCR caller (`crates/mt-pipeline/src/stages/hardsub_ocr.rs`) passes
  `best = false` — no behavior change.

## Error handling

- Per-episode mirror fallback via `pl_players` ordering.
- An episode that exhausts all mirrors is logged + counted, never aborts the run.
- yt-dlp failures surface as `HardsubError` from Python.
- No resolver JSON within `--timeout` → `FetchError::NotFound`, clear message.

## Testing

- **Rust unit:** episode-stem formatting helper; clap arg parsing (defaults,
  `--json` bypass). The real-network end-to-end download is `#[ignore]`'d, matching
  the repo convention for network/model-dependent tests (keeps Linux CI green).
- **Python unit:** `best=True` builds the `bv*+ba/b` selector and the `%(ext)s`
  outtmpl, and omits the ascending `format_sort` — assert on the opts dict /
  selector string without invoking yt-dlp. (Refactor opts-building into a small
  testable helper if needed.) Existing OCR-path tests must still pass unchanged.
- **Userscript:** unchanged, no new tests.
- **Gate:** `just check && just test && just py-test`.

## Tooling

Add a `just anime-dl <name> [args]` recipe wrapping
`cargo run --release --bin anime-dl --`.
