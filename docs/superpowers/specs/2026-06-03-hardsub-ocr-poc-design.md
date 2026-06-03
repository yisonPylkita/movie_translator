# Hardsub-OCR Polish-subtitle PoC — design

**Date:** 2026-06-03
**Status:** Design (pre-implementation)
**Scope:** Standalone proof-of-concept. **Not** wired into the Rust pipeline.

## Motivation

For some anime, no soft-sub Polish track exists on the usual providers
(animesub.info, napiprojekt, …), but Polish-translated versions *are*
streamed on community sites such as **ogladajanime.pl** with the Polish
subtitles **baked into the video** (hardsubbed — cda.pl and similar hosts
carry no soft-sub track). The user wants those community Polish subs, but
keeps watching their own **high-quality** local copy. So: acquire the
low-quality hardsubbed stream only to **OCR the Polish text**, then (later,
out of scope here) attach that text to the high-quality local video.

This document specifies a **proof-of-concept** that proves the acquisition
+ OCR chain end-to-end on one real episode. Pipeline integration
(validate / score / align, a `mt-fetch` provider, CLI flags) is explicitly
**deferred** until the PoC has shaken the quirks out.

## What the PoC does

Input: a **local anime directory** (the user's real download). For the
**first episode only**, end-to-end:

```
local dir (e.g. ".../[Judas] Isekai Ojisan ... [Eng-Subs]")
  → identify        movie_translator.identifier.parse_filename → {title, season, episode}
  → match           search ogladajanime.pl → anime slug → episode URL  (auto-pick, log choice)
  → cookies         read ogladajanime.pl cookies from local Chrome
  → scrape          headless Chrome + cookies → open episode → click player
                    → select cda host → read cross-origin iframe video.src → (url, headers)
  → download        lowest *legible*-res stream (~480p floor) → temp .mp4
  → ocr             movie_translator.ocr.extract_burned_in_subtitles(..., language='pl')
  → output          write Polish .srt + keep OCR frames/text dump for eyeballing
```

The PoC's job is to surface the quirks (cookie reuse, JS click-to-load
player, cross-origin iframe extraction, host choice, download throttling,
Polish-OCR accuracy / crop region / timing), not to be production-clean.

### Reference target

- Directory: `/Users/w/Downloads/Torrents/completed/[Judas] Isekai Ojisan (Uncle from Another World) (Season 01) [BD 1080p][HEVC x265 10bit][Dual-Audio][Eng-Subs]`
- First episode file: `[Judas] Isekai Ojisan - S01E01.mkv`
- Site entry: `https://ogladajanime.pl/anime/isekai-ojisan`

## Architecture

A single standalone script orchestrating small Python helpers. **Mostly
Python; zero Rust for the PoC** — the two functions we'd otherwise reach
into Rust for (`parse_filename`, burned-in OCR) are *already* Python in the
`movie_translator` package, so the script imports them directly. (Rust
enters only at the later integration phase, calling these via `mt-ml` as it
already does.)

```
scripts/hardsub_poc/                     (PoC lives here, isolated from the package)
  __main__.py        orchestrator + argparse: dir → first episode → run all steps
  cookies.py         ogladajanime.pl cookies from Chrome (browser_cookie3)
  scraper.py         OgladajanimeScraper: search slug, resolve episode → cda video.src
  download.py        pick lowest-legible-res, download stream → temp .mp4
  (OCR + identify reused from movie_translator.* — no new code)
```

### Components & boundaries

- **`cookies.py`** — `load_ogladajanime_cookies() -> list[Cookie]`.
  Reads the local Chrome cookie store for the `ogladajanime.pl` domain via
  `browser_cookie3` (handles macOS Keychain decryption). No password
  stored by us; reuses the user's existing logged-in session. Returns
  cookies in a shape Playwright's `context.add_cookies()` accepts.
  Depends on: local Chrome, `browser_cookie3`.

- **`scraper.py`** — `OgladajanimeScraper`.
  - `find_episode_url(title, episode) -> str` — search the site for the
    anime, fuzzy-match the best slug, build `/anime/<slug>/<N>` (1-indexed,
    `season→absolute` heuristic for S01 = pass episode through). Auto-picks
    the top candidate and **logs** the chosen slug + URL.
  - `resolve_video(episode_url) -> VideoSource` — launch headless Chrome
    (Playwright `channel="chrome"`, `--disable-web-security`
    `--disable-features=site-per-process` to allow the cross-origin iframe
    read), seed it with the loaded cookies, open the episode, click the
    poster (`#playerStartImg img`) to inject the player, select the **cda**
    entry from the host dropdown (`#changePlayerData`), then read
    `iframe.contentWindow.document.querySelector('video').src`. Returns
    `VideoSource{ url, headers (Referer/Origin/UA) }`.
  - Defined as a small `SiteScraper` protocol so other sites (wbijam, desu)
    can slot in later, but **only ogladajanime ships in the PoC**.
  - Blueprint: `Zlvsky/ogladajanime-scrapper` (Node+Puppeteer) proves this
    exact chain works; we port the selector logic to Playwright/Python.
  - Depends on: `playwright`, cookies.

- **`download.py`** — `download_lowest_legible(source, out_path, min_height=480) -> Path`.
  Direct `https` GET of the cda `video.src` with the referer/UA headers
  (cda serves a plain stream once you have the in-iframe URL). Picks the
  lowest available resolution **at or above ~480p** — text must stay
  OCR-legible, so we deliberately avoid absolute-worst (144p → OCR garbage).
  `yt-dlp` is an optional fallback for hosts that need format negotiation.
  Depends on: `requests`/`yt-dlp`.

- **OCR (reused)** — `movie_translator.ocr.extract_burned_in_subtitles(`
  `video_path, output_dir, crop_ratio, fps, language='pl')`. Already does
  transition-detection + Apple-Vision OCR + SRT writing; we just pass
  `language='pl'`. **macOS / Apple-Vision only** (matches repo
  constraints).

- **Identify (reused)** — `movie_translator.identifier.parse_filename(`
  `filename, folder_name)` → `{title, season, episode, is_anime, …}`.

### `__main__.py` flow

`python -m scripts.hardsub_poc <anime-dir> [--episode N] [--min-height 480]
[--keep-temp] [--out <dir>]`:
1. List video files in `<anime-dir>`, `parse_filename` each, select the
   episode whose `(season, episode)` is the first (default E01).
2. `cookies.load_ogladajanime_cookies()`.
3. `scraper.find_episode_url(title, episode)` → `resolve_video(...)`.
4. `download.download_lowest_legible(...)` → temp `.mp4`.
5. `extract_burned_in_subtitles(mp4, out_dir, language='pl')` → `.srt`.
6. Write `<out>/<title>-S0xE0y.pl.srt`; on `--keep-temp`, leave the mp4 and
   `_ocr_frames/` for inspection. Print the chosen slug/URL, the resolved
   video URL, format picked, frame/OCR counts, and the SRT path.

## Error handling (PoC posture)

Fail **loudly with a clear message** at the failing step (this is a
quirk-hunting tool, not a degrade-gracefully pipeline). Each step prints
what it tried so failures localize:
- No/expired cookies or login wall → message: "log into ogladajanime.pl in
  Chrome first"; exit.
- No slug match / multiple ambiguous → log candidates, pick top, continue
  (auto-pick by design); `--episode`/explicit-URL override available.
- Poster/iframe/host-dropdown selector not found → dump page HTML to temp
  for inspection; exit.
- cda `video.src` empty (cross-origin still blocked) → note the
  web-security flags, dump iframe HTML; exit.
- Download stalls → timeout + clear error.
- OCR empty / no transitions → keep frames, report; likely wrong crop
  region or res too low — tune `crop_ratio` / `min_height`.
- Non-macOS → OCR unavailable (Apple Vision); refuse early.

## Testing

- **Unit (fixtures, no network/browser):** slug fuzzy-match against saved
  search-result HTML; iframe/`#changePlayerData` extraction against a saved
  episode-page DOM fixture; download format-selection logic; cookie
  shape-conversion (mocked store).
- **Manual end-to-end:** run against the real Isekai Ojisan dir on macOS
  with a logged-in Chrome; eyeball the `.srt` + OCR frame dump for Polish
  accuracy and timing. This is the PoC's actual acceptance test.
- No Rust tests (no Rust touched). No CI wiring (browser + login + macOS
  Vision can't run in CI — consistent with the repo's `#[ignore]`'d
  real-model convention).

## Dependencies added

Python only, via `uv` (the `poc` group, kept out of the lean runtime deps):
`playwright`, `browser_cookie3`, and `yt-dlp`. `playwright` uses the system
Chrome (`channel="chrome"`) — no bundled Chromium download required.

**Gotcha:** because `poc` is an *optional* group, plain `uv run` (including
the PostToolUse `uv run ruff` hook) re-syncs the venv to the default groups
and **uninstalls** these deps. Run PoC code with `uv run --group poc python
-m scripts.hardsub_poc ...`, or `uv sync --group poc` then `.venv/bin/python
-m scripts.hardsub_poc ...`.

## Explicitly out of scope (deferred to integration)

- `mt-fetch` provider / `SubtitleProvider` impl, `validate → score → align`
  against the English reference, multi-candidate track output.
- `mt-ml` bridge functions, `mt-cli` flags, pipeline stage ordering.
- Multi-episode looping, multi-*site* scrapers (wbijam/desu),
  auto-attaching the OCR'd subs to the high-quality local file.
  (Multi-*host* fallback within ogladajanime — cda→sibnet→dood→… — IS in:
  every player for a title is PL-hardsubbed, so the orchestrator tries them
  in preference order until one resolves + downloads.)

These come **after** the PoC validates the acquisition + OCR quirks.

## Addendum — ogladajanime player mechanism (reverse-engineered 2026-06-03)

Live investigation against `/anime/isekai-ojisan/1` (logged-in Chrome
cookies) revealed the real player flow. This supersedes the original
"click poster → dropdown → cross-origin iframe `<video>.src`" guess.

- **Cold deep-links are bounced.** Navigating straight to
  `/anime/<slug>/<N>` redirects to `/error/20/NN` (anti-bot). A human-like
  warm-up (homepage → anime page → episode, with referer + small delays)
  avoids it. The site **rate-limits** rapid repeated automated hits — space
  runs out; don't loop.
- **The player list is a client-side API call.** On the episode page the JS
  POSTs `command('get_player_list')` → `/manager.php?action=get_player_list`
  (credentialed) → JSON `data.players[]`, each
  `{id, audio, sub, url:<hostname>, quality}`. Every entry for this title is
  `sub:"pl"` — i.e. **all hosts are Polish hardsubs** (these hosts carry no
  soft-sub track), so cda is *preferred* but not the only viable hardsub.
- **`#changePlayerData` mirrors the list**; each `<a>` onclick is
  `changePlayerUrl(<internal_id>)`. For ep 1 the cda entry was
  `changePlayerUrl(1011633)` (`[1080p] cda`). The links are hidden until the
  dropdown opens → query them as `attached`, not `visible`.
- **`changePlayerUrl(<id>)`** GETs `https://ogladajanime.pl:8443/Player/<id>`
  (port 8443, `withCredentials`) and injects the resolved URL into the
  `#playerFrame` iframe. A stale page-default id (`205242`) returned **403**;
  the real per-host id from the list is what works. A direct server-to-server
  GET of `:8443/Player/<id>` (plain `requests` + cookies) also returned 403 —
  it needs the in-browser request context — so the PoC drives the browser to
  call `changePlayerUrl` and reads the URL off `#playerFrame.src` rather than
  hitting `:8443` directly.
- **yt-dlp does the cda hop.** The resolved cda URL is an embed/player page,
  not a direct file; yt-dlp has a cda.pl extractor, so `download.py` routes
  **all `*.cda.pl` URLs to yt-dlp** (and we never read the cross-origin
  `<video>.src` — that whole step is deleted).

Net scraper flow now: warm up → episode page → detect `/error` bounce →
click poster (best-effort) → open dropdown → `get_player_list`-backed
`#changePlayerData` → parse cda `changePlayerUrl(id)` → call it → poll
`#playerFrame.src` for the cda URL → hand to yt-dlp.
