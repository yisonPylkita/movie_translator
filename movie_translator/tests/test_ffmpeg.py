import subprocess
from pathlib import Path

import pytest

from movie_translator.ffmpeg import (
    get_ffmpeg,
    get_ffmpeg_version,
    get_ffprobe,
    get_video_info,
    probe_video_encoding,
)


def test_get_ffmpeg_returns_path():
    path = get_ffmpeg()
    assert path is not None
    assert len(path) > 0


def test_get_ffprobe_returns_path():
    path = get_ffprobe()
    assert path is not None
    assert len(path) > 0


def test_get_ffmpeg_version_returns_string():
    version = get_ffmpeg_version()
    assert 'ffmpeg' in version.lower()


def test_get_video_info_returns_dict(create_test_mkv):
    mkv_file = create_test_mkv()
    info = get_video_info(mkv_file)

    assert isinstance(info, dict)
    assert 'streams' in info
    assert 'format' in info


def test_get_video_info_raises_on_nonexistent_file():
    with pytest.raises(subprocess.CalledProcessError):
        get_video_info(Path('/nonexistent/file.mkv'))


@pytest.fixture
def sample_video(tmp_path):
    """Create a short test video with known properties."""
    ffmpeg = get_ffmpeg()
    output = tmp_path / 'test.mp4'
    cmd = [
        ffmpeg,
        '-y',
        '-f',
        'lavfi',
        '-i',
        'color=black:s=320x240:d=1:r=24',
        '-f',
        'lavfi',
        '-i',
        'anullsrc=r=44100:cl=stereo',
        '-t',
        '1',
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-c:a',
        'aac',
        '-b:a',
        '128k',
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'Could not create test video: {result.stderr}')
    return output


class TestProbeVideoEncoding:
    def test_returns_codec_info(self, sample_video):
        info = probe_video_encoding(sample_video)

        assert info['codec_name'] == 'h264'
        assert info['width'] == 320
        assert info['height'] == 240
        assert abs(info['fps'] - 24.0) < 0.1

    def test_returns_pix_fmt(self, sample_video):
        info = probe_video_encoding(sample_video)

        assert info['pix_fmt'] == 'yuv420p'

    def test_raises_for_nonexistent(self, tmp_path):
        with pytest.raises(subprocess.CalledProcessError):
            probe_video_encoding(tmp_path / 'nonexistent.mp4')
