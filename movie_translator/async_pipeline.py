"""Async pipeline orchestration — runs multiple files concurrently with GPU serialisation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from movie_translator.context import PipelineConfig, PipelineContext
from movie_translator.gpu_queue import GpuQueue, InpaintTask, OcrTask, TranslateTask
from movie_translator.logging import current_file_tag, logger
from movie_translator.metrics.collector import MetricsCollector, NullCollector
from movie_translator.progress import ProgressTracker
from movie_translator.stages import (
    CreateTracksStage,
    ExtractEnglishStage,
    ExtractReferenceStage,
    FetchSubtitlesStage,
    IdentifyStage,
    MuxStage,
    TranslateStage,
)
from movie_translator.subtitles import SubtitleExtractor, SubtitleProcessor


def _make_file_tag(video_path: Path) -> str:
    """Short display name from path stem, max 20 chars."""
    stem = video_path.stem
    if len(stem) <= 20:
        return stem
    return stem[:17] + '...'


def _make_stages() -> dict[str, Any]:
    """Create a dict of stage instances keyed by role name."""
    return {
        'identify': IdentifyStage(),
        'extract_ref': ExtractReferenceStage(),
        'fetch': FetchSubtitlesStage(),
        'extract_english': ExtractEnglishStage(),
        'translate': TranslateStage(),
        'create_tracks': CreateTracksStage(),
        'mux': MuxStage(),
    }


async def _handle_pending_ocr(
    ctx: PipelineContext,
    gpu_queue: GpuQueue,
    file_tag: str,
    tracker_key: str,
    tracker: ProgressTracker,
    stage_label: str,
) -> None:
    """Submit a deferred OCR task to the GPU queue and apply results to ctx."""
    if ctx.pending_ocr is None:
        return

    pending = ctx.pending_ocr
    tracker.set_gpu_status(tracker_key, 'queued')

    if pending.type == 'pgs':
        task = OcrTask(
            ocr_type='pgs',
            video_path=ctx.video_path,
            track_index=pending.track_id or 0,
            work_dir=pending.output_dir,
            file_tag=file_tag,
        )
    else:
        task = OcrTask(
            ocr_type='burned_in',
            video_path=ctx.video_path,
            output_dir=pending.output_dir,
            file_tag=file_tag,
        )

    result = await gpu_queue.submit(task)
    tracker.set_gpu_status(tracker_key, 'none')

    # Apply results based on which stage deferred the OCR
    if stage_label == 'extract_ref':
        if pending.type == 'pgs' and result is not None:
            ctx.reference_path = result
        elif pending.type == 'burned_in' and result is not None:
            ctx.reference_path = result.srt_path
            ctx.ocr_results = result.ocr_results
    elif stage_label == 'extract_english':
        if pending.type == 'pgs' and result is not None:
            ctx.english_source = result
        elif pending.type == 'burned_in' and result is not None:
            ctx.english_source = result.srt_path
            ctx.ocr_results = result.ocr_results

        # Now extract dialogue lines if we got a source
        if ctx.english_source is not None and ctx.dialogue_lines is None:
            ctx.dialogue_lines = await asyncio.to_thread(
                SubtitleProcessor.extract_dialogue_lines, ctx.english_source
            )
            if not ctx.dialogue_lines:
                raise RuntimeError(f'No dialogue lines found in {ctx.english_source.name}')

    ctx.pending_ocr = None


async def process_file(
    video_path: Path,
    work_dir: Path,
    config: PipelineConfig,
    stages: dict[str, Any],
    gpu_queue: GpuQueue,
    tracker: ProgressTracker,
    metrics: MetricsCollector | NullCollector | None = None,
    display_name: str = '',
) -> bool:
    """Process a single video file through the async pipeline.

    Returns True on success, False on failure.
    """
    # display_name is the tracker key (set by run_all via start_file).
    # file_tag is a short name for log message tagging via ContextVar.
    file_tag = _make_file_tag(video_path)
    tracker_key = display_name or file_tag
    token = current_file_tag.set(file_tag)

    ctx = PipelineContext(
        video_path=video_path,
        work_dir=work_dir,
        config=config,
        metrics=metrics or NullCollector(),
    )

    try:
        # Stage 1 - Identify (IO)
        tracker.set_stage(tracker_key, 'identify')
        with ctx.metrics.span('identify'):
            await asyncio.to_thread(stages['identify'].run, ctx)

        # Stage 2 - Extract Reference (IO + deferred OCR)
        tracker.set_stage(tracker_key, 'extract')
        with ctx.metrics.span('extract_reference'):
            await asyncio.to_thread(stages['extract_ref'].run, ctx)
        if ctx.pending_ocr:
            await _handle_pending_ocr(ctx, gpu_queue, file_tag, tracker_key, tracker, 'extract_ref')

        # Stage 3 - Fetch (IO)
        tracker.set_stage(tracker_key, 'fetch')
        with ctx.metrics.span('fetch'):
            await asyncio.to_thread(stages['fetch'].run, ctx)

        # Stage 4 - Extract English (IO + deferred OCR)
        tracker.set_stage(tracker_key, 'extract')
        with ctx.metrics.span('extract'):
            await asyncio.to_thread(stages['extract_english'].run, ctx)
        if ctx.pending_ocr:
            await _handle_pending_ocr(
                ctx, gpu_queue, file_tag, tracker_key, tracker, 'extract_english'
            )

        if ctx.english_source is None:
            raise RuntimeError(f'No English subtitle source found for {video_path.name}')
        if ctx.dialogue_lines is None:
            raise RuntimeError(f'No dialogue lines extracted for {video_path.name}')

        # Stage 5 - Translate (font check IO + GPU translation concurrently)
        tracker.set_stage(tracker_key, 'translate')

        translate_stage = stages['translate']
        translate_stage.set_tracker(tracker)

        # Detect character names for translation protection
        from movie_translator.translation.proper_nouns import extract_proper_nouns_from_subtitles

        assert ctx.dialogue_lines is not None
        proper_nouns = extract_proper_nouns_from_subtitles(
            [line.text for line in ctx.dialogue_lines]
        )

        async def _check_fonts():
            with ctx.metrics.span('translate.check_fonts'):
                font_info = await asyncio.to_thread(translate_stage.check_fonts, ctx)
                ctx.font_info = font_info

        async def _translate_gpu():
            tracker.set_gpu_status(tracker_key, 'queued')

            def _on_progress(lines_done: int, total_lines: int, rate: float) -> None:
                tracker.set_stage_progress(tracker_key, lines_done, total_lines, rate=rate)
                tracker.gpu_task_progress(lines_done, total_lines, rate)

            assert ctx.dialogue_lines is not None
            task = TranslateTask(
                dialogue_lines=ctx.dialogue_lines,
                device=config.device,
                batch_size=config.batch_size,
                model=config.model,
                progress_callback=_on_progress,
                file_tag=file_tag,
                translation_cache=config.model_cache,
                proper_nouns=proper_nouns,
            )
            with ctx.metrics.span('translate.batch'):
                result = await gpu_queue.submit(task)
            tracker.set_gpu_status(tracker_key, 'none')
            ctx.translated_lines = result

        await asyncio.gather(_check_fonts(), _translate_gpu())

        if not ctx.translated_lines:
            raise RuntimeError('Translation failed -- empty result')

        # Stage 6 - Create Tracks (IO)
        tracker.set_stage(tracker_key, 'create')
        with ctx.metrics.span('create_tracks'):
            await asyncio.to_thread(stages['create_tracks'].run, ctx)

        # Stage 7 - Inpaint (optional GPU) then Mux (IO)
        tracker.set_stage(tracker_key, 'mux')
        if ctx.ocr_results and config.enable_inpaint and ctx.inpainted_video is None:
            tracker.set_gpu_status(tracker_key, 'queued')
            inpainted = work_dir / f'{video_path.stem}_inpainted{video_path.suffix}'
            task = InpaintTask(
                video_path=video_path,
                output_path=inpainted,
                ocr_results=ctx.ocr_results,
                device=config.device,
                file_tag=file_tag,
            )
            with ctx.metrics.span('inpaint'):
                await gpu_queue.submit(task)
            tracker.set_gpu_status(tracker_key, 'none')
            ctx.inpainted_video = inpainted

        with ctx.metrics.span('mux'):
            await asyncio.to_thread(stages['mux'].run, ctx)

        return True

    except Exception as e:
        logger.error(f'Failed: {video_path.name} - {e}')
        return False
    finally:
        current_file_tag.reset(token)


async def run_all(
    video_files: list[Path],
    root_dir: Path,
    config: PipelineConfig,
    metrics: MetricsCollector | NullCollector,
    gpu_queue: GpuQueue,
    tracker: ProgressTracker,
) -> list[tuple[Path, str]]:
    """Orchestrate processing of all video files with concurrency control.

    Returns a list of (path, status) where status is 'success', 'failed', or 'skipped'.
    """
    from movie_translator.discovery import create_work_dir

    workers = config.workers or min(len(video_files), 4)
    semaphore = asyncio.Semaphore(workers)
    stages = _make_stages()
    extractor = SubtitleExtractor()

    results: list[tuple[Path, str]] = []
    results_lock = asyncio.Lock()

    async def _process_one(video_path: Path) -> None:
        relative_name = (
            str(video_path.relative_to(root_dir))
            if root_dir != video_path.parent
            else video_path.name
        )

        async with semaphore:
            tracker.start_file(relative_name)

            # Check for existing Polish subtitles (IO-bound)
            has_polish = await asyncio.to_thread(extractor.has_polish_subtitles, video_path)
            if has_polish:
                tracker.complete_file(relative_name, 'skipped')
                async with results_lock:
                    results.append((video_path, 'skipped'))
                return

            work_dir = create_work_dir(video_path, root_dir)

            success = await process_file(
                video_path=video_path,
                work_dir=work_dir,
                config=config,
                stages=stages,
                gpu_queue=gpu_queue,
                tracker=tracker,
                metrics=metrics,
                display_name=relative_name,
            )

            status = 'success' if success else 'failed'
            tracker.complete_file(relative_name, status)
            async with results_lock:
                results.append((video_path, status))

    await asyncio.gather(*[_process_one(vp) for vp in video_files])
    return results
