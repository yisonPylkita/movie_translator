# Hardsub-OCR integration plan

**Status: IMPLEMENTED** (behind `--hardsub-ocr`). Builds on the proof-of-concept
in `scripts/hardsub_poc/` (design: `2026-06-03-hardsub-ocr-poc-design.md`).

What shipped vs. this plan:
- Python ML graduated to `movie_translator/hardsub/` (`download_episode`,
  `ocr_and_clean`), exposed via `crates/mt-ml` (`hardsub_download`,
  `hardsub_ocr_clean`).
- Discovery + open-browser + watch-`~/Downloads` + JSON parse + best-player
  selection live in `crates/mt-fetch/src/ogladajanime.rs`.
- A `GpuExecutor::hardsub_ocr_clean` worker job (OCR stays serialised on the GPU
  worker); download runs off-GPU.
- Orchestration: a once-per-run interactive prep in `run_all_full`
  (`prepare_hardsub_plan`) and a per-file `stages::hardsub_ocr` (Stage 4.5,
  after `extract_english` so the English alignment reference exists), which
  ilass-aligns the OCR'd Polish and injects it as a fetched `pol` track that
  `create_tracks` → `mux` already handle.
- **Known limitation (v1):** the OCR track is produced only for files that have
  an English source (used as the alignment reference) and thus go through the
  normal flow; files with no English at all still skip. Lifting that needs the
  translate/font-check path to run without an English source.

## Goal

Add an **optional** path to source Polish subtitles by OCRing the burned-in
Polish subs from `ogladajanime.pl`, for anime where the normal subtitle
providers find nothing. It is gated behind a CLI flag and is **semi-interactive
by necessity**.

## Why human-in-the-loop (the load-bearing constraint)

The PoC proved the site cannot be driven headlessly:

- Player embed URLs are gated by **Cloudflare Turnstile** (single-use, domain-
  bound token per call) — pure HTTP replay is impossible.
- The site's **anti-debug** detects the DevTools/CDP protocol and bounces any
  Playwright/CDP-driven browser to `/error/20/NN`.

The only reliable resolver is a **Tampermonkey userscript running in the user's
real browser** (`ogladajanime_resolver.user.js`), which the user triggers. So
integration is "assisted": the app does discovery + browser-open + pickup +
the heavy download/OCR/align; the human does one click to mint the Turnstile
tokens. Everything downstream of the userscript's JSON is fully automated.

## User-facing flow

```
movie-translator run <video-or-dir> --hardsub-ocr
```

1. **Identify** the anime from the filename (existing `mt-discovery` /
   `movie_translator.identifier`).
2. **Discover on ogladajanime**: slugify the title → check `/anime/<slug>`
   exists (GET; look for the episode list / "Odcinki: N"); fall back to the
   site search + fuzzy slug match (mirrors the userscript's logic). Resolve the
   episode count.
3. **Open the browser** at `https://ogladajanime.pl/anime/<slug>` (macOS
   `open`, Linux `xdg-open`). Print a one-line instruction: "Run the resolver
   userscript (⏬ All / ▶ This episode), then press Enter."
4. **Wait for the signal — auto-watch `~/Downloads`** for a new
   `oga-<slug>-*.players.json` (no Enter needed). Must guard against picking up
   a download still in flight — see "Detecting a fully-downloaded file" below.
5. **Per episode** in the JSON: pick the best PL player (CDA preferred) →
   yt-dlp download the lowest legible track → Vision OCR → clean (merge jitter
   + drop garbage) → **align to the local episode's English reference** →
   produce a Polish `.srt`.
6. **Mux** the aligned Polish track into the output, like any other found
   subtitle track.

## Where the code lives (respecting the Rust-orchestration / Python-ML split)

| Concern | Home | Notes |
|---|---|---|
| `--hardsub-ocr` flag, discovery, browser-open, Downloads-watch, interactive wait, orchestration | **Rust** | new `crates/mt-fetch/src/ogladajanime.rs` + a `mt-pipeline` stage + `mt-cli` flag |
| Download (yt-dlp) + OCR + clean (merge/garbage-filter) | **Python** | graduate `scripts/hardsub_poc/{download,postprocess}.py` into `movie_translator/hardsub/`, reuse `movie_translator/ocr`, expose via `crates/mt-ml` |
| Alignment to the local timeline | **reuse Rust `mt-fetch`** | drop the PoC's `align.py` duplicate — call the existing ilass (`align_ilass.rs`) + xcorr fallback against the local English reference |
| The userscript | tracked asset | `scripts/hardsub_poc/ogladajanime_resolver.user.js`; CLI prints its path / install hint |

The English reference already exists in the pipeline: the normal flow extracts
the local video's English track (soft-sub or English-hardsub OCR). The
hardsub-OCR path aligns the OCR'd Polish to that reference — fixing timing
drift between the ogladajanime encode and the local file, and (bonus) letting
us drop residual OP/ED/sign garbage that falls outside English dialogue windows.

## Pipeline placement & invariants

- **GPU is serialized.** OCR is a GPU/Vision stage — it must go through the
  single tokio GPU worker (`crates/mt-pipeline/src/worker.rs`), like the other
  ML stages. Don't add a parallel OCR path.
- **Interactivity is once-per-anime, up front.** The browser-open + wait can't
  run silently inside the parallel per-file lanes. Design: when `--hardsub-ocr`
  is set, do discovery → browser → wait → JSON pickup **once** for the anime,
  then the per-episode download/OCR/align/mux flows through the normal
  (serialized-GPU) pipeline.
- **macOS-only** (Apple Vision OCR + Apple-Silicon). The flag errors clearly on
  Linux, consistent with the existing burned-in-OCR `#[ignore]` tests.
- **Outward/rate-limited**: only the user's manual clicks hit the site (the app
  doesn't loop provider calls). The download step hits the video hosts (cda
  etc.) via yt-dlp — already rate-limited per file.

## Detecting a fully-downloaded file (the watch signal)

Watching `~/Downloads` is the trigger, but we must not read a half-written
file. Layered guards, all required:

1. **Ignore in-flight partials.** Browsers download to a temp name and rename
   to the final name only on completion — Chrome `*.crdownload`, Firefox
   `*.part`, Safari `*.download`. So we only ever consider files matching
   `oga-<slug>-*.players.json` (the final name); a partial simply isn't
   matched yet. This is the primary, near-sufficient signal (rename is atomic).
2. **Freshness.** Only accept files whose `mtime` is **after** the browser was
   opened, so a stale JSON from a previous run is never picked up.
3. **Size stability.** Require the file size to be unchanged across two polls
   (~500 ms apart) before reading — belt-and-suspenders against odd writers.
4. **Valid + complete JSON.** Final check: `json.loads` succeeds **and** the
   parsed object has the expected shape (`episodes: [...]` with `resolved`
   entries). A truncated write fails to parse, so this rejects partials the
   other guards somehow missed; retry on failure.

Poll ~1 s. Overall timeout (e.g. 10 min) with a clear "still waiting…" line and
Ctrl-C to bail. Pick the **newest** matching file if several appear.

## Episode ↔ local-file mapping

The userscript JSON keys episodes by number (`episode`). Match each to the
local file via the existing parsed `(season, episode)`. A `run` on a directory
processes all matched episodes; on a single file, just that episode.

## Open decisions

1. **Signal mechanism**: ~~decided~~ — **auto-watch `~/Downloads`** for the
   new matching JSON (no manual Enter), with the layered fully-downloaded
   detection above (ignore `*.crdownload`/`*.part`/`*.download`, fresh mtime,
   stable size, valid+complete JSON).
2. **When to trigger**: always when the flag is set, or only as a fallback
   after providers find no Polish subs? Leaning: explicit (flag set ⇒ run it),
   independent of provider results, since it's opt-in.
3. **`yt-dlp` dependency**: promote from the `poc` group into the runtime deps
   (it kept getting pruned by stray `uv sync` in the PoC).
4. **Downloads dir**: honor `$XDG_DOWNLOAD_DIR` / a `--downloads-dir` override
   rather than hardcoding `~/Downloads`.

## Phasing

- **Phase 1 — graduate the ML.** Move `download.py` + `postprocess.py` into
  `movie_translator/hardsub/`, reuse `movie_translator/ocr`, expose a
  `resolve_json → [aligned PL srt]` entry through `crates/mt-ml`. Reuse the
  Rust ilass align (delete the PoC `align.py`). Unit-test the pure bits.
- **Phase 2 — discovery + assisted pickup.** Rust: slug discovery on
  ogladajanime, `open`-the-browser, interactive wait, newest-Downloads-JSON
  pickup, behind `--hardsub-ocr`.
- **Phase 3 — wire to mux + mapping.** Episode↔file mapping, mux the aligned
  Polish tracks, end-to-end on a directory; ship the userscript as a tracked
  asset with a CLI install hint.

## What the PoC already proved (carried in)

Resolve (userscript, 28/28 players, all sub_groups) → download lowest (cda
~480p) → Vision OCR → clean (1355 raw → ~162 legible PL lines) → ilass align
(recovers offsets, verified). The integration is wiring these proven pieces
into the Rust pipeline behind a flag; no new unsolved problems remain.
