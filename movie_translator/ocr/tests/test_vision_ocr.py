import platform
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Darwin',
    reason='Apple Vision framework only available on macOS',
)


@pytest.fixture
def image_with_text(tmp_path):
    """Create a simple image with known text using FFmpeg."""
    from movie_translator.ffmpeg import get_ffmpeg

    ffmpeg = get_ffmpeg()
    output = tmp_path / 'test_text.png'

    cmd = [
        ffmpeg,
        '-y',
        '-f',
        'lavfi',
        '-i',
        'color=black:s=640x120:d=1',
        '-vf',
        "drawtext=text='Hello world':fontsize=48:fontcolor=white:x=(w-tw)/2:y=(h-th)/2",
        '-frames:v',
        '1',
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'Could not create test image: {result.stderr}')

    return output


@pytest.fixture
def blank_image(tmp_path):
    """Create a blank black image with no text."""
    from movie_translator.ffmpeg import get_ffmpeg

    ffmpeg = get_ffmpeg()
    output = tmp_path / 'blank.png'

    cmd = [
        ffmpeg,
        '-y',
        '-f',
        'lavfi',
        '-i',
        'color=black:s=640x120:d=1',
        '-frames:v',
        '1',
        str(output),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return output


def test_is_available():
    from movie_translator.ocr.vision_ocr import is_available

    assert is_available() is True


def test_recognize_text_finds_known_text(image_with_text):
    from movie_translator.ocr.vision_ocr import recognize_text

    result = recognize_text(image_with_text)

    assert 'hello' in result.lower() or 'Hello' in result


def test_recognize_text_returns_empty_for_blank(blank_image):
    from movie_translator.ocr.vision_ocr import recognize_text

    result = recognize_text(blank_image)

    assert result == ''


def test_recognize_text_returns_empty_for_nonexistent():
    from movie_translator.ocr.vision_ocr import recognize_text

    result = recognize_text(Path('/nonexistent/image.png'))

    assert result == ''
