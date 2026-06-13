//! Hardsub-OCR stage (gated by `--hardsub-ocr`).
//!
//! Runs after `fetch`, only when the once-per-run interactive prep produced a
//! [`HardsubPlan`] containing this file's episode. For that episode it:
//!   1. downloads the lowest OCR-legible copy of the best PL player (yt-dlp;
//!      network/CPU — NOT a GPU job),
//!   2. OCRs + cleans the burned-in Polish subs through the serialised GPU
//!      worker,
//!   3. aligns the result to the English reference (ilass — sync to the local
//!      timeline; also drops OP/ED junk outside dialogue windows),
//!   4. injects it as a fetched Polish track, which `create_tracks` → `mux`
//!      already turn into an output subtitle track.
//!
//! Every failure here is non-fatal (logged, skipped): a missing player, a dead
//! host, or empty OCR must not fail the whole file — the normal AI-translation
//! track still ships.

use std::collections::HashMap;

use mt_core::{FetchedSubtitle, PipelineContext};
use mt_fetch::ogladajanime::HardsubPlan;

use crate::error::Result;
use crate::gpu::GpuExecutor;
use tracing::{info, warn};

/// Lowest stream height to accept for OCR legibility (matches the PoC default).
const MIN_HEIGHT: u32 = 480;
/// ilass split penalty — same default as the fetch-align path.
const SPLIT_PENALTY: f64 = 7.0;

/// Run the hardsub-OCR stage against `plan`. Always returns `Ok` (failures are
/// logged and skipped); on success appends a Polish [`FetchedSubtitle`].
pub fn run(
    mut ctx: PipelineContext,
    executor: &dyn GpuExecutor,
    plan: &HardsubPlan,
) -> Result<PipelineContext> {
    let Some(episode) = ctx.identity.as_ref().and_then(|i| i.episode) else {
        info!("hardsub-ocr: no episode number identified; skipping");
        return Ok(ctx);
    };
    let players = plan.pl_players(episode as i64);
    if players.is_empty() {
        info!("hardsub-ocr: no PL player resolved for episode {episode}; skipping");
        return Ok(ctx);
    }

    let work = ctx.work_dir.join("hardsub");
    std::fs::create_dir_all(&work)?;
    let video = work.join(format!("ep{episode}.mp4"));

    // 1. Download the lowest legible copy, trying each mirror best-first until
    // one resolves (network/CPU — not a GPU job). A dead/410 mirror (e.g. a cda
    // upload that was removed) fails fast at extraction with no wasted bytes, so
    // we drop to the next mirror (vk/mega/…). Pass NO Referer: these are yt-dlp
    // extractor hosts that set their own headers; an ogladajanime Referer leaks
    // onto cda's internal API calls and breaks extraction.
    let mut downloaded = false;
    for player in &players {
        info!(
            "hardsub-ocr: episode {episode} -> trying {} {} ({})",
            player.host.as_deref().unwrap_or("?"),
            player.quality.as_deref().unwrap_or("?"),
            player.embed_url
        );
        match mt_ml::hardsub_download(&player.embed_url, &video, MIN_HEIGHT, false, None) {
            Ok(_) => {
                downloaded = true;
                break;
            }
            Err(e) => {
                warn!(
                    "hardsub-ocr: {} mirror failed for episode {episode} ({e}); trying next",
                    player.host.as_deref().unwrap_or("?")
                );
            }
        }
    }
    if !downloaded {
        warn!(
            "hardsub-ocr: all {} mirror(s) failed for episode {episode}; skipping",
            players.len()
        );
        return Ok(ctx);
    }

    // 2. OCR + clean through the serialised GPU worker.
    let cleaned = match executor.hardsub_ocr_clean(&video, &work, "pl") {
        Ok(Some(path)) => path,
        Ok(None) => {
            warn!("hardsub-ocr: OCR produced no usable lines for episode {episode}");
            return Ok(ctx);
        }
        Err(e) => {
            warn!("hardsub-ocr: OCR failed for episode {episode}: {e}");
            return Ok(ctx);
        }
    };

    // 3. Align to the English reference (sync to the local timeline). Best-effort.
    if let Some(reference) = ctx
        .reference_path
        .clone()
        .or_else(|| ctx.english_source.clone())
    {
        let changed = mt_fetch::align_ilass(&cleaned, &reference, SPLIT_PENALTY);
        info!("hardsub-ocr: aligned episode {episode} to reference (changed={changed})");
    } else {
        info!("hardsub-ocr: no English reference for episode {episode}; leaving OCR timing as-is");
    }

    // 4. Inject as a fetched Polish track (create_tracks turns it into a track).
    let map = ctx.fetched_subtitles.get_or_insert_with(HashMap::new);
    map.entry("pol".to_string())
        .or_default()
        .push(FetchedSubtitle {
            path: cleaned,
            source: "ogladajanime-ocr".to_string(),
        });
    info!("hardsub-ocr: added Polish OCR track for episode {episode}");
    Ok(ctx)
}
