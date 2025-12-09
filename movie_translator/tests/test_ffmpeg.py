import subprocess
from pathlib import Path

import pytest

from movie_translator.ffmpeg import get_ffmpeg, get_ffmpeg_version, get_ffprobe, get_video_info


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
