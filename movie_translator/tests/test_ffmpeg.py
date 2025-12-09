from movie_translator.ffmpeg import (
    get_ffmpeg,
    get_ffmpeg_version,
    get_ffprobe,
    get_subtitle_streams,
    get_video_info,
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


def test_get_subtitle_streams_returns_only_subtitles(create_test_mkv):
    mkv_file = create_test_mkv()
    streams = get_subtitle_streams(mkv_file)

    assert len(streams) == 1
    assert streams[0]['codec_type'] == 'subtitle'
