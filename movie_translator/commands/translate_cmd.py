"""CLI handler for the translate command (default)."""

import asyncio
import logging
import shutil
import sys
from pathlib import Path

from ..context import PipelineConfig
from ..discovery import create_work_dir, find_videos
from ..ffmpeg import get_video_info
from ..logging import console, logger, set_verbose
from ..metrics.collector import MetricsCollector, NullCollector
from ..metrics.listeners import ReportBuilder
from ..metrics.report import build_report, save_report
from ..pipeline import TranslationPipeline
from ..progress import ProgressTracker
from ..subtitles import SubtitleExtractor
from ..translation import ModelCache
from .common import check_dependencies, resolve_model


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Movie Translator - Extract English dialogue → AI translate to Polish → Replace original video'
    )
    parser.add_argument('input', help='Video file or directory containing video files')
    parser.add_argument(
        '--device',
        choices=['cpu', 'mps'],
        default='mps' if sys.platform == 'darwin' else 'cpu',
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument(
        '--model',
        choices=['allegro', 'apple'],
        default=None,
        help='Translation backend (default: auto-detect, prefers apple on macOS 26+)',
    )
    parser.add_argument('--no-fetch', action='store_true')
    parser.add_argument(
        '--inpaint',
        action='store_true',
        help='Remove burned-in subtitles from video frames via inpainting (slow)',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--keep-artifacts', action='store_true')
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Concurrent pipeline workers (default: auto, min(files, 4))',
    )
    parser.add_argument(
        '--external-subs',
        default=None,
        help='Directory with pre-extracted subtitles (from extract command) to add as additional tracks',
    )
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--metrics', action='store_true', help='Collect performance metrics')
    return parser.parse_args(argv)


def _show_summary(results: list[tuple[str, str]], dry_run: bool = False) -> None:
    successful = sum(1 for _, status in results if status == 'success')
    failed = sum(1 for _, status in results if status == 'failed')
    skipped = sum(1 for _, status in results if status == 'skipped')

    parts = []
    if successful > 0:
        parts.append(f'[green]✓ {successful} translated[/green]')
    if skipped > 0:
        parts.append(f'[blue]⏭ {skipped} skipped[/blue]')
    if failed > 0:
        parts.append(f'[red]✗ {failed} failed[/red]')

    console.print(' | '.join(parts))

    if dry_run and successful > 0:
        console.print('[yellow]Dry run - originals not modified[/yellow]')


def _sync_main(video_files, root_dir, args, collector, report_builder):
    """Run the pipeline sequentially for each video file (workers == 1)."""
    extractor = SubtitleExtractor()

    with ProgressTracker(len(video_files), console=console) as tracker:
        pipeline = TranslationPipeline(
            device=args.device,
            batch_size=args.batch_size,
            model=args.model,
            enable_fetch=not args.no_fetch,
            enable_inpaint=args.inpaint,
            tracker=tracker,
            metrics=collector,
            external_subs_dir=Path(args.external_subs) if args.external_subs else None,
        )

        for video_path in video_files:
            relative_name = (
                str(video_path.relative_to(root_dir))
                if root_dir != video_path.parent
                else video_path.name
            )
            tracker.start_file(relative_name)
            work_dir = create_work_dir(video_path, root_dir)
            success = False

            if report_builder is not None:
                report_builder.start_video(
                    path=str(video_path),
                    hash='',
                    file_size_bytes=video_path.stat().st_size if video_path.exists() else 0,
                    duration_ms=0,
                    identity={},
                )

            try:
                if extractor.has_polish_subtitles(video_path):
                    tracker.complete_file('skipped')
                    success = True
                elif pipeline.process_video_file(video_path, work_dir, dry_run=args.dry_run):
                    tracker.complete_file('success')
                    success = True
                else:
                    tracker.complete_file('failed')
            except Exception as e:
                logger.error(f'Unexpected error: {e}')
                tracker.complete_file('failed')

            if report_builder is not None:
                identity = getattr(pipeline, 'last_identity', None)
                if identity is not None:
                    identity_dict = {
                        'title': identity.title,
                        'parsed_title': identity.parsed_title,
                        'media_type': identity.media_type,
                        'season': identity.season,
                        'episode': identity.episode,
                        'year': identity.year,
                        'is_anime': identity.is_anime,
                        'release_group': identity.release_group,
                        'imdb_id': identity.imdb_id,
                        'tmdb_id': identity.tmdb_id,
                    }
                    video_duration_ms = 0
                    try:
                        info = get_video_info(video_path)
                        duration_s = float(info.get('format', {}).get('duration', 0))
                        video_duration_ms = int(duration_s * 1000)
                    except Exception:
                        pass
                    report_builder.update_current_video(
                        identity=identity_dict,
                        hash=identity.oshash,
                        duration_ms=video_duration_ms,
                    )
                report_builder.end_video()

            if success and not args.keep_artifacts and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    parent = work_dir.parent
                    temp_root = root_dir / '.translate_temp'
                    while parent != temp_root and parent != root_dir:
                        if parent.exists() and not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                    if temp_root.exists() and not any(temp_root.iterdir()):
                        temp_root.rmdir()
                except OSError as e:
                    logger.debug(f'Failed to clean up {work_dir}: {e}')


async def _async_main(video_files, root_dir, args, collector, report_builder, workers):
    """Run the pipeline concurrently via async orchestration."""
    from ..async_pipeline import run_all
    from ..gpu_queue import GpuQueue

    config = PipelineConfig(
        device=args.device,
        batch_size=args.batch_size,
        model=args.model,
        enable_fetch=not args.no_fetch,
        enable_inpaint=args.inpaint,
        dry_run=args.dry_run,
        workers=workers,
        external_subs_dir=Path(args.external_subs) if args.external_subs else None,
        model_cache=ModelCache(),
    )

    with ProgressTracker(len(video_files), console=console) as tracker:
        gpu_queue = GpuQueue(tracker=tracker)
        gpu_worker = asyncio.create_task(gpu_queue.run_worker())
        results = await run_all(video_files, root_dir, config, collector, gpu_queue, tracker)
        await gpu_queue.shutdown()
        await gpu_worker

    if report_builder is not None:
        for video_path, _status in results:
            report_builder.start_video(
                path=str(video_path),
                hash='',
                file_size_bytes=video_path.stat().st_size if video_path.exists() else 0,
                duration_ms=0,
                identity={},
            )
            report_builder.end_video()


def run(argv: list[str] | None = None) -> None:
    """Entry point for the translate flow (default command)."""
    args = parse_args(argv)
    set_verbose(args.verbose)
    args.model = resolve_model(args.model)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f'[red]❌ Not found: {input_path}[/red]')
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    video_files = find_videos(input_path)
    if not video_files:
        console.print(f'[red]❌ No video files found in {input_path}[/red]')
        sys.exit(1)

    root_dir = input_path if input_path.is_dir() else input_path.parent

    if args.dry_run:
        console.print('[yellow]Dry run mode - originals will not be modified[/yellow]')

    logging.getLogger('transformers').setLevel(logging.ERROR)

    if args.metrics:
        collector = MetricsCollector()
        report_builder = ReportBuilder()
        collector.add_listener(report_builder.on_event)
    else:
        collector = NullCollector()
        report_builder = None

    workers = args.workers if args.workers > 0 else min(len(video_files), 4)

    if workers > 1:
        asyncio.run(_async_main(video_files, root_dir, args, collector, report_builder, workers))
    else:
        _sync_main(video_files, root_dir, args, collector, report_builder)

    if report_builder is not None:
        report = build_report(
            videos=report_builder.videos,
            config={
                'device': args.device,
                'batch_size': args.batch_size,
                'model': args.model,
                'enable_fetch': not args.no_fetch,
                'enable_inpaint': args.inpaint,
            },
        )
        report_path = root_dir / '.translate_temp' / 'metrics.json'
        save_report(report, report_path)
        console.print(f'[dim]Metrics saved to {report_path}[/dim]')
