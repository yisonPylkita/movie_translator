#!/usr/bin/env python3
"""
Movie Translator - Final Complete Pipeline
Extract English dialogue → AI translate to Polish → Replace original MKV with Polish as default
"""

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ai_translator import SubtitleTranslator as AITranslator

# OCR support flag - imports will be loaded lazily when needed
OCR_AVAILABLE = False

console = Console()


def clear_memory():
    """Clear memory caches and force garbage collection."""
    gc.collect()


def replace_polish_chars(text: str) -> str:
    """Replace Polish characters with English equivalents."""
    polish_to_english = {
        'ą': 'a',
        'ć': 'c',
        'ę': 'e',
        'ł': 'l',
        'ń': 'n',
        'ó': 'o',
        'ś': 's',
        'ź': 'z',
        'ż': 'z',
        'Ą': 'A',
        'Ć': 'C',
        'Ę': 'E',
        'Ł': 'L',
        'Ń': 'N',
        'Ó': 'O',
        'Ś': 'S',
        'Ź': 'Z',
        'Ż': 'Z',
    }

    for polish, english in polish_to_english.items():
        text = text.replace(polish, english)

    return text


def log_info(message: str):
    console.print(f'[blue][INFO][/blue] {message}')


def log_success(message: str):
    console.print(f'[green][SUCCESS][/green] {message}')


def log_warning(message: str):
    console.print(f'[yellow][WARNING][/yellow] {message}')


def log_error(message: str):
    console.print(f'[red][ERROR][/red] {message}')


class SubtitleOCR:
    """OCR handler for image-based subtitles using PaddleOCR."""

    def __init__(self, use_gpu: bool = False):
        self.ocr = None
        self.use_gpu = use_gpu
        self.initialized = False
        self._ocr_available = None  # Will be determined when needed

    def _check_ocr_availability(self):
        """Check if OCR dependencies are available (lazy loading)."""
        if self._ocr_available is not None:
            return self._ocr_available

        try:
            import importlib.util

            cv2_spec = importlib.util.find_spec('cv2')
            paddleocr_spec = importlib.util.find_spec('paddleocr')

            if cv2_spec is not None and paddleocr_spec is not None:
                self._ocr_available = True
                return True
            else:
                self._ocr_available = False
                return False
        except ImportError:
            self._ocr_available = False
            return False

    def initialize(self):
        """Initialize OCR model (lazy loading to avoid startup delay)."""
        if not self._check_ocr_availability():
            log_error('OCR dependencies not found. Install with: uv add opencv-python paddleocr')
            return False

        if self.initialized:
            return True

        log_info('🔧 Initializing OCR engine for image-based subtitles...')
        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(lang='en')
            self.initialized = True
            log_success('   ✅ OCR engine initialized')
            return True
        except Exception as e:
            log_error(f'   ❌ Failed to initialize OCR: {e}')
            return False

    def ocr_subtitle_image(self, image_path_or_np_array):
        """
        Extract text from subtitle image using OCR.
        Returns clean text lines sorted top-to-bottom.
        """
        if not self.initialize():
            return ''

        try:
            result = self.ocr.ocr(image_path_or_np_array)

            if not result or result[0] is None:
                return ''

            # Extract text and sort by vertical position (important for subtitles)
            lines = []
            for line in result[0]:
                bbox, (text, confidence) = line
                text = text.strip()
                if len(text) < 1 or confidence < 0.5:  # filter garbage
                    continue
                # Use the top-center Y of the bounding box to sort
                top_y = bbox[0][1]
                lines.append((top_y, text))

            # Sort by vertical position
            lines.sort(key=lambda x: x[0])

            return '\n'.join([text for _, text in lines])
        except Exception as e:
            log_error(f'OCR processing failed: {e}')
            return ''

    def cleanup(self):
        """Clean up OCR resources."""
        if self.ocr:
            self.ocr = None
        self.initialized = False


def extract_pgs_subtitles_to_images(mkv_path: Path, track_id: int, output_dir: Path) -> list[Path]:
    """Extract PGS subtitles to individual image files."""
    log_info('🖼️  Extracting PGS subtitles to images...')

    # Create temp directory for PGS extraction
    pgs_dir = output_dir / 'pgs_temp'
    pgs_dir.mkdir(exist_ok=True)

    # Extract PGS track
    pgs_file = pgs_dir / f'{mkv_path.stem}_track{track_id}.sup'
    cmd = ['mkvextract', 'tracks', str(mkv_path), f'{track_id}:{pgs_file}']

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_success(f'   - PGS track extracted: {pgs_file.name}')
    except subprocess.CalledProcessError as e:
        log_error(f'Failed to extract PGS track {track_id}: {e}')
        return []

    # Use BDSup2Sub to convert PGS to images (if available)
    # This is a placeholder - in reality, we'd need a tool like BDSup2Sub or similar
    # For now, we'll skip PGS processing as it requires additional tools
    log_warning('   - PGS to image conversion requires BDSup2Sub (not implemented)')
    log_warning('   - Skipping PGS track processing')

    # Clean up
    shutil.rmtree(pgs_dir)
    return []


def process_image_based_subtitles(
    mkv_path: Path, track_id: int, output_dir: Path, use_gpu: bool = False
) -> Path | None:
    """Process image-based subtitles and return path to generated SRT file."""
    # Check OCR availability lazily
    ocr_check = SubtitleOCR()
    if not ocr_check._check_ocr_availability():
        log_warning('OCR not available - skipping image-based subtitles')
        ocr_check.cleanup()
        return None
    ocr_check.cleanup()

    ocr = SubtitleOCR(use_gpu=use_gpu)

    # For now, we'll implement a basic approach
    # In a full implementation, we'd:
    # 1. Extract PGS to individual images
    # 2. Use OCR on each image
    # 3. Create timing information
    # 4. Generate SRT file

    log_info('🤖 Processing image-based subtitles with OCR...')
    log_warning('   - Full OCR implementation requires PGS extraction tools')
    log_warning('   - Using placeholder implementation')

    # Create a placeholder SRT file
    output_srt = output_dir / f'{mkv_path.stem}_ocr_extracted.srt'

    try:
        # Placeholder content - in real implementation, this would come from OCR
        placeholder_content = """1
00:00:01,000 --> 00:00:03,000
[Image-based subtitle - OCR processing needed]

2
00:00:05,000 --> 00:00:07,000
[This is a placeholder - implement PGS extraction first]
"""
        output_srt.write_text(placeholder_content)
        log_success(f'   - Created placeholder SRT: {output_srt.name}')
        return output_srt
    except Exception as e:
        log_error(f'Failed to create placeholder SRT: {e}')
        return None
    finally:
        ocr.cleanup()


def check_dependencies():
    console.print(Panel.fit('[bold blue]Dependency Check[/bold blue]', border_style='blue'))

    table = Table(show_header=False, box=None)
    table.add_column('Component', style='cyan')
    table.add_column('Status', style='green')

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        log_error(f'Python 3.8+ required, found {version.major}.{version.minor}')
        sys.exit(1)
    log_info(f'Python: {version.major}.{version.minor}.{version.micro}')

    if not shutil.which('mkvmerge'):
        log_error('mkvmerge not found. Please install mkvtoolnix')
        sys.exit(1)
    result = subprocess.run(['mkvmerge', '--version'], capture_output=True, text=True)
    log_info(f'mkvmerge: {result.stdout.split()[0]}')

    if not shutil.which('mkvextract'):
        log_error('mkvextract not found. Please install mkvtoolnix')
        sys.exit(1)

    try:
        import importlib.util

        required_packages = ['pysubs2', 'torch', 'transformers']
        missing_packages = []

        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)

        if missing_packages:
            log_error(f'Missing Python packages: {", ".join(missing_packages)}')
            log_info('Install with: uv add pysubs2 torch transformers')
            sys.exit(1)

        log_info('Python packages: pysubs2, torch, transformers')

        # OCR availability will be checked lazily when needed
        log_info('OCR support: Will be checked when --enable-ocr is used')
    except ImportError as e:
        log_error(f'Missing Python package: {e}')
        log_info('Install with: uv add pysubs2 torch transformers')
        sys.exit(1)

    console.print(
        Panel('[bold green]✅ All dependencies satisfied[/bold green]', border_style='green')
    )


def get_track_info(mkv_path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ['mkvmerge', '-J', str(mkv_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        log_error(f'Failed to get track info from {mkv_path.name}: {e}')
        return {}


def find_english_subtitle_track(
    track_info: dict[str, Any], enable_ocr: bool = False
) -> dict[str, Any] | None:
    tracks = track_info.get('tracks', [])
    english_tracks = []

    for track in tracks:
        if track.get('type') == 'subtitles':
            props = track.get('properties', {})
            lang = props.get('language', '')
            if lang in ['eng', 'en']:
                english_tracks.append(track)

    if not english_tracks:
        return None

    if len(english_tracks) == 1:
        return english_tracks[0]

    dialogue_tracks = []
    signs_tracks = []

    for track in english_tracks:
        props = track.get('properties', {})
        track_name = props.get('track_name', '').lower()

        if any(keyword in track_name for keyword in ['sign', 'song', 'title', 'op', 'ed']):
            signs_tracks.append(track)
        else:
            dialogue_tracks.append(track)

    if dialogue_tracks:
        # Separate text-based and image-based dialogue tracks
        text_dialogue_tracks = []
        image_dialogue_tracks = []

        for track in dialogue_tracks:
            codec = track.get('codec', '').lower()
            if 'substationalpha' in codec or 'ass' in codec or 'ssa' in codec:
                text_dialogue_tracks.append(track)
            elif 'hdmv pgs' in codec or 'pgs' in codec:
                image_dialogue_tracks.append(track)

        if text_dialogue_tracks:
            log_info(
                f'Found {len(english_tracks)} English tracks, selected text-based dialogue track'
            )
            return text_dialogue_tracks[0]
        elif image_dialogue_tracks:
            # Only image-based dialogue tracks available - try OCR if available and enabled
            if enable_ocr:
                # Check OCR availability lazily
                ocr_check = SubtitleOCR()
                if ocr_check._check_ocr_availability():
                    log_info(
                        f'Found {len(english_tracks)} English tracks, will process image-based dialogue with OCR'
                    )
                    # Mark the track for OCR processing by adding a special flag
                    image_dialogue_tracks[0]['requires_ocr'] = True
                    ocr_check.cleanup()
                    return image_dialogue_tracks[0]
                else:
                    log_warning(
                        f'Found {len(english_tracks)} English tracks, but only image-based dialogue tracks available'
                    )
                    log_info('Install OCR support with: uv add opencv-python paddleocr')
                    ocr_check.cleanup()
                    return None
            else:
                log_warning(
                    f'Found {len(english_tracks)} English tracks, but only image-based dialogue tracks available'
                )
                log_info('Enable OCR with --enable-ocr flag')
                return None
        else:
            log_info(f'Found {len(english_tracks)} English tracks, selected dialogue track')
            return dialogue_tracks[0]

    # No dialogue tracks found - check for text-based signs/songs as fallback
    text_signs_tracks = []
    image_signs_tracks = []

    for track in signs_tracks:
        codec = track.get('codec', '').lower()
        if 'substationalpha' in codec or 'ass' in codec or 'ssa' in codec:
            text_signs_tracks.append(track)
        elif 'hdmv pgs' in codec or 'pgs' in codec:
            image_signs_tracks.append(track)

    if text_signs_tracks:
        log_info(
            f'Found {len(english_tracks)} English tracks, using text-based signs/songs as fallback (no dialogue tracks)'
        )
        return text_signs_tracks[0]
    elif image_signs_tracks:
        log_warning(
            f'Found {len(english_tracks)} English tracks, but only image-based signs/songs available (no text extraction possible)'
        )
        return None

    non_forced_tracks = []
    for track in english_tracks:
        props = track.get('properties', {})
        if not props.get('forced_track', False):
            non_forced_tracks.append(track)

    if non_forced_tracks:
        log_info(f'Found {len(english_tracks)} English tracks, selected non-forced track')
        return non_forced_tracks[0]

    log_warning(f'Found {len(english_tracks)} English tracks, all appear to be signs/songs')
    return english_tracks[0]


def get_subtitle_extension(track: dict[str, Any]) -> str:
    # Check both properties.codec_id and top-level codec field
    props = track.get('properties', {})
    codec_id = props.get('codec_id', '')
    codec = track.get('codec', '').lower()

    # Combine both for detection
    combined = f'{codec_id} {codec}'.lower()

    if 'ass' in combined or 's_text/ass' in combined or 'substationalpha' in combined:
        return '.ass'
    elif 'ssa' in combined or 's_text/ssa' in combined:
        return '.ssa'
    else:
        return '.srt'


def extract_subtitle(mkv_path: Path, track_id: int, output_path: Path) -> bool:
    log_info(f'Extracting subtitle track {track_id}...')

    cmd = ['mkvextract', 'tracks', str(mkv_path), f'{track_id}:{output_path}']

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_success(f'Extraction successful: {output_path.name}')
        return True
    except subprocess.CalledProcessError as e:
        log_error(f'Failed to extract subtitle track {track_id}: {e}')
        if e.stderr:
            log_error(f'stderr: {e.stderr}')
        return False


def extract_dialogue_lines(subtitle_file: Path) -> list[tuple[int, int, str]]:
    log_info(f'📖 Reading {subtitle_file.name}...')

    try:
        import pysubs2

        subs = pysubs2.load(str(subtitle_file))
        log_info(f'   - Loaded {len(subs)} total events')
    except Exception as e:
        log_error(f'Failed to load: {e}')
        return []

    dialogue_lines = []
    skipped_count = 0

    original_count = len(subs)
    # Group consecutive events with the same text (animated subtitles)
    unique_subs = []
    last_text = None
    current_group_start = None
    current_group_end = None

    for event in subs:
        clean_text = event.plaintext.strip()

        # Skip empty or too short text
        if not clean_text or len(clean_text) < 2:
            continue

        # If this is the same text as the previous event (consecutive duplicate)
        if last_text == clean_text:
            # Extend the time span to include this duplicate
            current_group_end = max(current_group_end, event.end)
            continue
        else:
            # Save the previous group if it exists
            if last_text is not None:
                # Create a new event with the consolidated timing
                consolidated_event = pysubs2.SSAEvent(
                    start=current_group_start,
                    end=current_group_end,
                    style=unique_subs[-1].style if unique_subs else 'Default',
                    text=last_text,
                )
                unique_subs[-1] = consolidated_event

            # Start a new group
            last_text = clean_text
            current_group_start = event.start
            current_group_end = event.end
            unique_subs.append(event)

    # Handle the last group
    if last_text is not None and unique_subs:
        consolidated_event = pysubs2.SSAEvent(
            start=current_group_start,
            end=current_group_end,
            style=unique_subs[-1].style,
            text=last_text,
        )
        unique_subs[-1] = consolidated_event

    deduped_count = len(unique_subs)
    if deduped_count < original_count:
        log_info(
            f'   - Deduplicated: {original_count} → {deduped_count} entries (removed {original_count - deduped_count} duplicate effect layers)'
        )

    for event in unique_subs:
        if not event.text or event.text.strip() == '':
            skipped_count += 1
            continue

        style = getattr(event, 'style', 'Default')
        style_lower = style.lower()

        if any(keyword in style_lower for keyword in ['sign', 'song', 'title', 'op', 'ed']):
            skipped_count += 1
            continue

        clean_text = event.plaintext.strip()

        if not clean_text:
            skipped_count += 1
            continue

        if len(clean_text) < 2:
            skipped_count += 1
            continue

        dialogue_lines.append((event.start, event.end, clean_text))

    log_info(f'   - Extracted {len(dialogue_lines)} dialogue lines')
    log_info(f'   - Skipped {skipped_count} non-dialogue events')

    del subs
    clear_memory()
    log_info('   - After cleanup')

    return dialogue_lines


def create_clean_english_ass(
    original_ass: Path,
    dialogue_lines: list[tuple[int, int, str]],
    output_english_ass: Path,
):
    log_info(f'🔨 Creating clean English ASS: {output_english_ass.name}')

    try:
        import pysubs2

        original_subs = pysubs2.load(str(original_ass))
        log_info(f'   - Loaded original with {len(original_subs)} events')

        clean_english_subs = pysubs2.SSAFile()
        clean_english_subs.info = original_subs.info.copy()
        clean_english_subs.styles = original_subs.styles.copy()

        # Create dialogue events directly from the extracted dialogue lines
        for start, end, text in dialogue_lines:
            # Replace newlines with \N (ASS line break) to prevent line wrapping
            clean_text = text.replace('\n', '\\N')
            new_event = pysubs2.SSAEvent(
                start=start,
                end=end,
                style='Default',
                text=clean_text,
            )
            clean_english_subs.append(new_event)

        clean_english_subs.save(str(output_english_ass))
        log_success(f'   - Saved {len(clean_english_subs)} dialogue events')
        log_info('   - Removed all non-dialogue events')

    except Exception as e:
        log_error(f'Failed to create clean English ASS: {e}')


def validate_cleaned_subtitles(
    original_ass: Path,
    cleaned_ass: Path,
) -> bool:
    """Validate that cleaned subtitles are a proper subset of original with matching timestamps."""
    log_info('🔍 Validating cleaned subtitles...')

    try:
        import pysubs2

        # Load both files
        original_subs = pysubs2.load(str(original_ass))
        cleaned_subs = pysubs2.load(str(cleaned_ass))

        # Create dictionary of cleaned subs by text for easy lookup
        cleaned_dict = {}
        for event in cleaned_subs:
            clean_text = event.text.replace('\\N', ' ').strip()
            if clean_text:
                if clean_text not in cleaned_dict:
                    cleaned_dict[clean_text] = []
                cleaned_dict[clean_text].append((event.start, event.end))

        # Check each original dialogue line
        mismatches = 0
        matches = 0
        original_dialogue_count = 0
        found_in_cleaned = 0

        for event in original_subs:
            style = getattr(event, 'style', 'Default')
            style_lower = style.lower()

            # Skip non-dialogue styles
            if any(keyword in style_lower for keyword in ['sign', 'song', 'title', 'op', 'ed']):
                continue

            clean_text = event.plaintext.strip()
            if not clean_text or len(clean_text) < 2:
                continue

            original_dialogue_count += 1

            # Check if this dialogue line exists in cleaned version
            if clean_text in cleaned_dict:
                found_in_cleaned += 1

                # Check if this original event is covered by any cleaned event with the same text
                is_covered = False
                for clean_start, clean_end in cleaned_dict[clean_text]:
                    if clean_start <= event.start and clean_end >= event.end:
                        matches += 1
                        is_covered = True
                        break

                if not is_covered:
                    mismatches += 1
                    log_warning(f'   - Timing gap for "{clean_text[:30]}..."')
                    log_warning(f'     Original: {event.start} → {event.end}')
                    for clean_start, clean_end in cleaned_dict[clean_text]:
                        log_warning(f'     Cleaned:  {clean_start} → {clean_end}')

        log_info(f'   - Original dialogue lines: {original_dialogue_count}')
        log_info(f'   - Cleaned dialogue lines:  {len(cleaned_subs)}')
        log_info(f'   - Lines covered:          {matches}')
        log_info(f'   - Timing gaps:            {mismatches}')
        log_info(f'   - Lines correctly removed: {original_dialogue_count - found_in_cleaned}')

        if mismatches == 0:
            log_success('   ✅ All original events are properly covered!')
            return True
        else:
            log_error(f'   ❌ Found {mismatches} timing gaps!')
            return False

    except Exception as e:
        log_error(f'Failed to validate cleaned subtitles: {e}')
        return False


def create_polish_ass(
    original_ass: Path,
    translated_dialogue: list[tuple[int, int, str]],
    output_polish_ass: Path,
    font_mode: str = 'replace',
):
    """Create Polish ASS file with translated dialogue."""
    log_info('🔤 Creating Polish subtitles')

    try:
        import pysubs2

        original_subs = pysubs2.load(str(original_ass))
        log_info(f'   - Loaded {len(original_subs)} original events')

        polish_subs = pysubs2.SSAFile()
        polish_subs.info = original_subs.info.copy()
        polish_subs.styles = original_subs.styles.copy()

        # Create Polish events directly from the translated dialogue lines
        for start, end, translated_text in translated_dialogue:
            if font_mode == 'replace':
                translated_text = replace_polish_chars(translated_text)

            # Replace newlines with \N (ASS line break) to prevent line wrapping
            clean_text = translated_text.replace('\n', '\\N')

            new_event = pysubs2.SSAEvent(
                start=start,
                end=end,
                style='Default',
                text=clean_text,
            )
            polish_subs.append(new_event)

        polish_subs.save(str(output_polish_ass))
        log_success(f'   - Saved {len(polish_subs)} translated events')

        del original_subs
        del polish_subs
        clear_memory()
        log_info('   - After cleanup')

    except Exception as e:
        log_error(f'Failed to create Polish ASS: {e}')


def create_clean_mkv(original_mkv: Path, english_ass: Path, polish_ass: Path, output_mkv: Path):
    """Create clean MKV with only video/audio + Polish (AI) + English dialogue (Polish as default)."""
    log_info(f'🎬 Creating clean MKV: {output_mkv.name}')
    log_info('   - Adding: Polish (AI) + English dialogue (Polish as default)')

    cmd = [
        'mkvmerge',
        '-o',
        str(output_mkv),
        '--no-subtitles',
        str(original_mkv),
        '--language',
        '0:pol',
        '--track-name',
        '0:Polish (AI)',
        '--default-track-flag',
        '0:yes',  # Make Polish the default subtitle track
        str(polish_ass),
        '--language',
        '0:eng',
        '--track-name',
        '0:English Dialogue',
        '--default-track-flag',
        '0:no',  # Make English non-default
        str(english_ass),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_success('   - Clean MKV merge successful')

        if output_mkv.stat().st_size == 0:
            size_mb = output_mkv.stat().st_size / 1024 / 1024
            log_info(f'   - Output size: {size_mb:.1f} MB')

        return True
    except subprocess.CalledProcessError as e:
        log_error(f'Failed to merge: {e}')
        if e.stderr:
            log_error(f'   stderr: {e.stderr}')
        return False


def verify_result(output_mkv: Path):
    """Verify the clean MKV has only the desired tracks."""
    log_info(f'🔍 Verifying result: {output_mkv.name}')

    try:
        result = subprocess.run(
            ['mkvmerge', '-J', str(output_mkv)],
            capture_output=True,
            text=True,
            check=True,
        )

        track_info = json.loads(result.stdout)
        tracks = track_info.get('tracks', [])

        subtitle_tracks = []
        for track in tracks:
            if track.get('type') == 'subtitles':
                props = track.get('properties', {})
                subtitle_tracks.append(
                    {
                        'id': track.get('id'),
                        'language': props.get('language', 'unknown'),
                        'name': props.get('track_name', 'unnamed'),
                    }
                )

        log_info(f'   - Found {len(subtitle_tracks)} subtitle tracks:')
        for track in subtitle_tracks:
            log_info(f'     * Track {track["id"]}: {track["name"]} ({track["language"]})')

        if len(subtitle_tracks) == 2:
            # Check track order: Polish first (default), English second
            polish_first = subtitle_tracks[0]['language'] == 'pol'
            english_second = subtitle_tracks[1]['language'] == 'eng'

            if polish_first and english_second:
                log_success('   ✅ Perfect! Polish (AI) as default track + English dialogue')
                return True
            else:
                log_error('   ❌ Incorrect track order. Expected: Polish first, English second')
                log_error(
                    f'   ❌ Found: Track 1={subtitle_tracks[0]["language"]}, Track 2={subtitle_tracks[1]["language"]}'
                )
                return False
        else:
            log_error(f'   ❌ Expected 2 subtitle tracks, found {len(subtitle_tracks)}')
            return False

    except Exception as e:
        log_error(f'Failed to verify: {e}')
        return False


def translate_dialogue_lines(
    dialogue_lines: list[tuple[int, int, str]], device: str, batch_size: int
) -> list[tuple[int, int, str]]:
    """Translate dialogue lines using AI with Rich progress bar."""
    log_info('🤖 Step 3: AI translating to Polish...')

    translator = AITranslator(device=device, batch_size=batch_size)
    log_info('   - AI Translator initialized')

    if not translator.load_model():
        log_error('❌ Failed to load translation model')
        return []

    log_info('   - Model loaded')

    texts = [text for _, _, text in dialogue_lines]
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('•'),
        TextColumn('{task.fields[rate]}'),
        console=console,
    ) as progress:
        task = progress.add_task(
            f'[cyan]Translating {len(texts)} texts...[/cyan]',
            total=total_batches,
            rate='0.0 lines/s',
        )

        def progress_callback(batch_num, total_batches, rate, error=None):
            if error:
                progress.update(task, advance=1, rate=f'❌ {error}')
            else:
                progress.update(task, advance=1, rate=f'{rate:.1f} lines/s')

        # Translate with progress callback
        translated_texts = translator.translate_texts(texts, progress_callback=progress_callback)

    log_info('   - Translation complete')

    # Cleanup
    translator.cleanup()
    log_info('   - Translator cleaned up')

    # Force garbage collection
    clear_memory()
    log_info('   - Final cleanup')

    # Reconstruct with timing
    translated_lines = []
    for (start, end, _), translated_text in zip(dialogue_lines, translated_texts, strict=True):
        translated_lines.append((start, end, translated_text))

    return translated_lines


def process_mkv_file(
    mkv_path: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    enable_ocr: bool = False,
    ocr_gpu: bool = False,
) -> bool:
    """Process a single MKV file and replace it with clean version."""
    console.print(Panel(f'[bold blue]Processing: {mkv_path.name}[/bold blue]', border_style='blue'))

    # Create temp directory
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)

    # Step 1: Extract English subtitles
    log_info('📖 Step 1: Extracting English subtitles...')
    track_info = get_track_info(mkv_path)
    if not track_info:
        log_error('Could not read track information')
        return False

    eng_track = find_english_subtitle_track(track_info, enable_ocr)
    if not eng_track:
        log_warning(
            'No suitable English subtitle track found (only image-based tracks or no tracks available)'
        )
        return False

    track_id = eng_track['id']
    track_name = eng_track.get('properties', {}).get('track_name', 'Unknown')
    log_info(f"Found English track: ID {track_id}, Name: '{track_name}'")

    # Check if this track requires OCR processing
    requires_ocr = eng_track.get('requires_ocr', False)

    if requires_ocr:
        log_info('🤖 Processing image-based subtitles with OCR...')

        # Process image-based subtitles
        extracted_srt = process_image_based_subtitles(mkv_path, track_id, output_dir, ocr_gpu)
        if not extracted_srt:
            log_error('OCR processing failed')
            return False

        # For OCR processing, we'll create a simple text-based subtitle file
        # In a full implementation, this would have proper timing from OCR
        extracted_ass = extracted_srt  # Use the OCR-generated SRT
    else:
        # Normal text-based subtitle extraction
        subtitle_ext = get_subtitle_extension(eng_track)
        extracted_ass = output_dir / f'{mkv_path.stem}_extracted{subtitle_ext}'

        if not extract_subtitle(mkv_path, track_id, extracted_ass):
            return False

    # Step 2: Extract dialogue lines
    log_info('🔍 Step 2: Extracting dialogue lines...')
    dialogue_lines = extract_dialogue_lines(extracted_ass)
    if not dialogue_lines:
        log_error('No dialogue lines found')
        return False

    # Step 3: AI translate
    log_info('🤖 Step 3: AI translating to Polish...')
    try:
        translated_dialogue = translate_dialogue_lines(dialogue_lines, device, batch_size)
        if not translated_dialogue:
            log_error('AI translation failed')
            return False
        log_success('   ✅ AI translation complete!')
    except Exception as e:
        log_error(f'AI translation failed: {e}')
        return False

    # Step 4: Create clean ASS files
    log_info('🔨 Step 4: Creating clean subtitle files...')
    # Keep subtitle files in main output directory (not temp) so they won't be deleted
    clean_english_ass = output_dir / f'{mkv_path.stem}_english_clean.ass'
    polish_ass = output_dir / f'{mkv_path.stem}_polish.ass'

    create_clean_english_ass(extracted_ass, dialogue_lines, clean_english_ass)

    # Validate that cleaned subtitles have correct timestamps
    extracted_ass_path = output_dir / f'{mkv_path.stem}_extracted.ass'
    if not validate_cleaned_subtitles(extracted_ass_path, clean_english_ass):
        log_error('❌ Validation failed! Cleaned subtitles have timestamp mismatches.')
        return False

    create_polish_ass(extracted_ass, translated_dialogue, polish_ass)

    # Step 5: Create clean MKV (replace original)
    log_info('🎬 Step 5: Creating clean MKV to replace original...')

    # Create temporary file first
    temp_mkv = temp_dir / f'{mkv_path.stem}_temp_clean.mkv'

    if not create_clean_mkv(mkv_path, clean_english_ass, polish_ass, temp_mkv):
        return False

    # Step 6: Verify temporary file
    if not verify_result(temp_mkv):
        return False

    # Step 7: Replace original file
    log_info('🔄 Step 7: Replacing original MKV...')
    try:
        # Create backup of original (just in case)
        backup_path = mkv_path.with_suffix('.mkv.backup')
        shutil.copy2(mkv_path, backup_path)
        log_info(f'   - Created backup: {backup_path.name}')

        # Replace original with clean version
        shutil.move(str(temp_mkv), str(mkv_path))
        log_success(f'   - Replaced original: {mkv_path.name}')

        # Verify the replacement worked
        if not verify_result(mkv_path):
            log_error('   - Verification failed after replacement, restoring backup')
            shutil.move(str(backup_path), str(mkv_path))
            return False

        # Remove backup if everything is good
        backup_path.unlink()
        log_info('   - Removed backup (verification successful)')

    except Exception as e:
        log_error(f'   - Failed to replace original: {e}')
        return False

    # Clean up temp files
    shutil.rmtree(temp_dir)
    log_info('🧹 Cleaned up temporary files')

    log_success(f'🎉 SUCCESS! Original MKV replaced with clean version: {mkv_path.name}')
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Movie Translator - Extract English dialogue and replace original MKV with Polish translation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process all MKV files in directory
  %(prog)s /path/to/anime

  # Process with custom settings
  %(prog)s /path/to/movies --device mps --batch-size 32

  # Process single file
  %(prog)s /path/to/movie.mkv --output /path/to/output
        """,
    )

    parser.add_argument('input', help='MKV file or directory containing MKV files')

    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        help='Output directory (default: translated/ next to input)',
    )

    parser.add_argument(
        '--device',
        choices=['mps', 'cpu'],
        default='mps',  # MacBook optimized
        help='Translation device (default: mps)',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Translation batch size (default: 16)',
    )

    parser.add_argument(
        '--enable-ocr',
        action='store_true',
        help='Enable OCR processing for image-based subtitles (requires opencv-python paddleocr)',
    )

    parser.add_argument(
        '--ocr-gpu',
        action='store_true',
        help='Use GPU for OCR processing (requires CUDA)',
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        log_error(f'Input not found: {input_path}')
        sys.exit(1)

    input_path = input_path.resolve()

    # Determine output directory
    if args.output:
        output_dir = args.output.resolve()
    else:
        if input_path.is_file():
            output_dir = input_path.parent / 'translated'
        else:
            output_dir = input_path / 'translated'

    output_dir.mkdir(exist_ok=True)

    check_dependencies()

    config_table = Table(
        title='[bold blue]Movie Translator - Final Pipeline[/bold blue]',
        show_header=False,
        box=None,
    )
    config_table.add_column('Setting', style='cyan')
    config_table.add_column('Value', style='white')
    config_table.add_row('Input', str(input_path))
    config_table.add_row('Output', str(output_dir))
    config_table.add_row('Device', args.device)
    config_table.add_row('Batch Size', str(args.batch_size))
    config_table.add_row('OCR Enabled', 'Yes' if args.enable_ocr else 'No')
    if args.enable_ocr:
        config_table.add_row('OCR GPU', 'Yes' if args.ocr_gpu else 'No')
    console.print(Panel(config_table, border_style='blue'))
    console.print()

    # Find MKV files
    if input_path.is_file():
        mkv_files = [input_path]
    else:
        mkv_files = sorted(input_path.glob('*.mkv'))

    if not mkv_files:
        log_warning('No MKV files found')
        sys.exit(0)

    log_info(f'Found {len(mkv_files)} MKV file(s)')

    # Process files with fancy progress
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        task = progress.add_task(
            f'[cyan]Processing {len(mkv_files)} MKV files...[/cyan]', total=len(mkv_files)
        )

        for mkv_file in mkv_files:
            progress.update(task, description=f'[cyan]Processing {mkv_file.name}...[/cyan]')

            if process_mkv_file(
                mkv_file, output_dir, args.device, args.batch_size, args.enable_ocr, args.ocr_gpu
            ):
                successful += 1
                progress.update(task, advance=1)
            else:
                failed += 1
                progress.update(task, advance=1)

    # Final summary with fancy panel
    summary_table = Table(
        title='[bold green]Translation Complete[/bold green]', show_header=False, box=None
    )
    summary_table.add_column('Status', style='white')
    summary_table.add_column('Count', style='white')
    summary_table.add_row('✅ Successful', str(successful), style='green')
    summary_table.add_row('❌ Failed', str(failed), style='red')
    summary_table.add_row('📁 Total', str(len(mkv_files)), style='cyan')

    console.print(Panel(summary_table, border_style='green'))

    if failed == 0:
        console.print(
            Panel(
                '[bold green]🎉 All files processed successfully![/bold green]\n'
                '🎬 Clean MKVs with English dialogue + Polish translation created\n'
                f'📁 Output directory: {output_dir}',
                border_style='green',
            )
        )
    else:
        console.print(
            Panel(
                '[bold yellow]⚠️  Some files failed to process.[/bold yellow]', border_style='yellow'
            )
        )


if __name__ == '__main__':
    main()
