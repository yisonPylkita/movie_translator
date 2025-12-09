import json
import subprocess
import tempfile
from pathlib import Path

from .ffmpeg import get_ffmpeg, get_ffprobe
from .logging import logger

POLISH_CHARS = 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'


def get_embedded_fonts(video_path: Path) -> list[dict]:
    ffprobe = get_ffprobe()
    cmd = [
        ffprobe,
        '-v',
        'quiet',
        '-print_format',
        'json',
        '-show_streams',
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    fonts = []
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'attachment':
            mimetype = stream.get('tags', {}).get('mimetype', '')
            if 'font' in mimetype.lower() or mimetype in (
                'application/x-truetype-font',
                'application/vnd.ms-opentype',
            ):
                fonts.append(
                    {
                        'index': stream.get('index'),
                        'filename': stream.get('tags', {}).get(
                            'filename', f'font_{stream.get("index")}'
                        ),
                        'mimetype': mimetype,
                    }
                )
    return fonts


def extract_font(video_path: Path, stream_index: int, output_path: Path) -> bool:
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg,
        '-y',
        '-dump_attachment:' + str(stream_index),
        str(output_path),
        '-i',
        str(video_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_path.exists()


def font_supports_polish(font_path: Path) -> bool:
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(str(font_path))
        cmap = font.getBestCmap()
        if cmap is None:
            return False
        return all(ord(c) in cmap for c in POLISH_CHARS)
    except Exception:
        return False


def get_ass_font_names(ass_path: Path) -> set[str]:
    from .subtitles._pysubs2 import get_pysubs2

    pysubs2 = get_pysubs2()
    if pysubs2 is None:
        return set()

    try:
        subs = pysubs2.load(str(ass_path))
        font_names = set()
        for style in subs.styles.values():
            if hasattr(style, 'fontname') and style.fontname:
                font_names.add(style.fontname.lower())
        return font_names
    except Exception:
        return set()


def check_embedded_fonts_support_polish(video_path: Path, ass_path: Path) -> bool:
    embedded_fonts = get_embedded_fonts(video_path)
    if not embedded_fonts:
        logger.info('   - No embedded fonts found, will replace Polish characters')
        return False

    ass_font_names = get_ass_font_names(ass_path)
    if not ass_font_names:
        logger.info('   - No font names in ASS styles, will replace Polish characters')
        return False

    logger.info(
        f'   - Found {len(embedded_fonts)} embedded font(s), checking Polish character support...'
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fonts_supporting_polish = 0

        for font_info in embedded_fonts:
            font_filename = font_info['filename']
            font_output = temp_path / font_filename

            if not extract_font(video_path, font_info['index'], font_output):
                continue

            if font_supports_polish(font_output):
                fonts_supporting_polish += 1
                font_name_lower = Path(font_filename).stem.lower()
                if any(
                    font_name_lower in ass_font or ass_font in font_name_lower
                    for ass_font in ass_font_names
                ):
                    logger.info(f'   - Font "{font_filename}" supports Polish characters')

        if fonts_supporting_polish > 0:
            logger.info(
                f'   - {fonts_supporting_polish}/{len(embedded_fonts)} embedded font(s) support Polish characters'
            )
            return True

    logger.warning('   - Embedded fonts do not support Polish characters, will replace them')
    return False
