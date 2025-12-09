#!/usr/bin/env python3
"""
Subtitle processing functions for the Movie Translator pipeline.
Handles extraction, dialogue line processing, and subtitle file creation.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .utils import log_error, log_info, log_success, log_warning


class SubtitleOCR:
    """OCR support for image-based subtitles (lazy loading)."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.ocr = None
        self.initialized = False

    def _check_ocr_availability(self) -> bool:
        """Check if OCR dependencies are available."""
        try:
            import importlib.util

            if (
                importlib.util.find_spec('cv2') is None
                or importlib.util.find_spec('paddleocr') is None
            ):
                return False
            return True
        except ImportError:
            return False

    def initialize(self):
        """Initialize OCR engine."""
        if not self.initialized and self._check_ocr_availability():
            try:
                from paddleocr import PaddleOCR

                # Initialize PaddleOCR with minimal configuration
                self.ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang='en',
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
                self.initialized = True
                log_info('   - OCR initialized successfully')
            except Exception as e:
                log_warning(f'Failed to initialize OCR: {e}')
                return False
        return self.initialized

    def extract_text_from_image(self, image_path: Path) -> str:
        """Extract text from subtitle image."""
        if not self.initialize():
            return ''

        try:
            result = self.ocr.ocr(str(image_path), cls=False)
            if result and result[0]:
                text = ' '.join([line[1][0] for line in result[0] if line[1][0].strip()])
                return text
        except Exception as e:
            log_warning(f'OCR failed for {image_path.name}: {e}')
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
    """Check if all required dependencies are available."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
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
    """Get track information from MKV file."""
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
    """Find the best English subtitle track from available tracks."""
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
    """Get the appropriate subtitle file extension for a track."""
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
    """Extract subtitle track from MKV file."""
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
    """Extract dialogue lines from subtitle file with duplicate handling."""
    log_info(f'📖 Reading {subtitle_file.name}...')

    try:
        import pysubs2
    except ImportError:
        log_error('pysubs2 package not found. Install with: uv add pysubs2')
        return []

    try:
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

        # Add to dialogue lines list
        dialogue_lines.append((event.start, event.end, clean_text))

    log_info(f'   - Extracted {len(dialogue_lines)} dialogue lines')
    log_info(f'   - Skipped {skipped_count} non-dialogue events')
    log_info('   - After cleanup')

    return dialogue_lines


def create_clean_english_ass(
    original_ass: Path, dialogue_lines: list[tuple[int, int, str]], output_english_ass: Path
):
    """Create clean English ASS file with only dialogue lines."""
    log_info(f'🔨 Creating clean English ASS: {output_english_ass.name}')

    try:
        import pysubs2
    except ImportError:
        log_error('pysubs2 package not found. Install with: uv add pysubs2')
        return

    try:
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
    except ImportError:
        log_error('pysubs2 package not found. Install with: uv add pysubs2')
        return

    try:
        original_subs = pysubs2.load(str(original_ass))
        log_info(f'   - Loaded {len(original_subs)} original events')

        polish_subs = pysubs2.SSAFile()
        polish_subs.info = original_subs.info.copy()
        polish_subs.styles = original_subs.styles.copy()

        # Create Polish events directly from the translated dialogue lines
        for start, end, translated_text in translated_dialogue:
            if font_mode == 'replace':
                from .utils import replace_polish_chars

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
        from .utils import clear_memory

        clear_memory()
        log_info('   - After cleanup')

    except Exception as e:
        log_error(f'Failed to create Polish ASS: {e}')
