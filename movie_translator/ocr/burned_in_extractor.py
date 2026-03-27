import shutil
from pathlib import Path

from ..logging import logger
from ..types import DialogueLine
from .change_detector import SubtitleTransition, detect_transitions
from .frame_extractor import extract_subtitle_region_frames, get_video_duration_ms
from .vision_ocr import recognize_text


def _build_dialogue_lines(
    transitions: list[SubtitleTransition],
    ocr_results: dict[Path, str],
) -> list[DialogueLine]:
    lines: list[DialogueLine] = []

    appeared_events = [t for t in transitions if t.event_type == 'appeared']

    for event in appeared_events:
        text = ocr_results.get(event.frame_path, '').strip()

        # Filter garbage
        if len(text) <= 1:
            continue

        # Determine end time: next transition (of any type) after this appeared event
        event_idx = transitions.index(event)
        if event_idx + 1 < len(transitions):
            end_ms = transitions[event_idx + 1].timestamp_ms
        else:
            # Last event — use a reasonable default (3 seconds after start)
            end_ms = event.timestamp_ms + 3000

        start_ms = event.timestamp_ms

        # Deduplicate: if previous line has same text, extend its end time
        if lines and lines[-1].text == text:
            lines[-1] = DialogueLine(lines[-1].start_ms, end_ms, text)
        else:
            lines.append(DialogueLine(start_ms, end_ms, text))

    return lines


def _format_srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    seconds = (ms % 60_000) // 1000
    millis = ms % 1000
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}'


def _write_srt(lines: list[DialogueLine], output_path: Path) -> None:
    parts: list[str] = []
    for i, line in enumerate(lines, 1):
        start = _format_srt_time(line.start_ms)
        end = _format_srt_time(line.end_ms)
        parts.append(f'{i}\n{start} --> {end}\n{line.text}\n')

    output_path.write_text('\n'.join(parts), encoding='utf-8')


def extract_burned_in_subtitles(
    video_path: Path,
    output_dir: Path,
    crop_ratio: float = 0.25,
    fps: int = 10,
) -> Path | None:
    frames_dir = output_dir / '_ocr_frames'

    try:
        # Step 1: Extract cropped frames
        frames = extract_subtitle_region_frames(
            video_path, frames_dir, fps=fps, crop_ratio=crop_ratio
        )
        if not frames:
            logger.error('No frames extracted from video')
            return None

        # Step 2: Detect transitions
        transitions = detect_transitions(frames)
        if not transitions:
            logger.warning(
                'No subtitle transitions detected — video may not have burned-in subtitles'
            )
            return None

        appeared_count = sum(1 for t in transitions if t.event_type == 'appeared')
        video_duration_ms = get_video_duration_ms(video_path)
        video_duration_min = video_duration_ms / 60_000

        if appeared_count < 5 and video_duration_min > 10:
            logger.warning(
                f'Only {appeared_count} subtitle appearances in {video_duration_min:.0f}min video — '
                'this may not have burned-in subtitles'
            )

        # Step 3: OCR on transition frames
        appeared_frames = [t for t in transitions if t.event_type == 'appeared']
        logger.info(f'Running OCR on {len(appeared_frames)} transition frames...')

        ocr_results: dict[Path, str] = {}
        for t in appeared_frames:
            text = recognize_text(t.frame_path)
            ocr_results[t.frame_path] = text

        # Step 4: Build dialogue lines
        lines = _build_dialogue_lines(transitions, ocr_results)
        if not lines:
            logger.warning('OCR produced no usable subtitle lines')
            return None

        logger.info(f'Extracted {len(lines)} subtitle lines via OCR')

        # Step 5: Write SRT
        srt_path = output_dir / f'{video_path.stem}_ocr.srt'
        _write_srt(lines, srt_path)

        return srt_path

    finally:
        # Clean up frame images
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
