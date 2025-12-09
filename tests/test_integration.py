from pathlib import Path
from unittest.mock import patch

from movie_translator.main import find_mkv_files_with_temp_dirs
from movie_translator.pipeline import TranslationPipeline
from movie_translator.types import DialogueLine


def test_full_pipeline_with_ass_subtitles(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(
        subtitle_content="""[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,How are you
""",
        language='eng',
        track_name='English',
    )

    mock_translated = [
        DialogueLine(1000, 3000, 'Witaj swiecie'),
        DialogueLine(4000, 6000, 'Jak sie masz'),
    ]

    pipeline = TranslationPipeline(device='cpu', batch_size=1, model='allegro')

    with patch.object(pipeline, '_translate', return_value=mock_translated):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is True
    assert mkv_file.exists()


def test_full_pipeline_with_srt_subtitles(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(
        subtitle_content="""1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
How are you
""",
        subtitle_format='srt',
        language='eng',
        track_name='English',
    )

    mock_translated = [
        DialogueLine(1000, 3000, 'Witaj swiecie'),
        DialogueLine(4000, 6000, 'Jak sie masz'),
    ]

    pipeline = TranslationPipeline(device='cpu', batch_size=1, model='allegro')

    with patch.object(pipeline, '_translate', return_value=mock_translated):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is True
    assert mkv_file.exists()


def test_full_pipeline_detects_no_english_subtitles(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(language='jpn', track_name='Japanese')

    pipeline = TranslationPipeline(device='cpu', batch_size=1)

    with patch.object(pipeline, '_translate', return_value=[]):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is False


def test_full_pipeline_fails_when_translation_returns_empty(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(language='eng', track_name='English')

    pipeline = TranslationPipeline(device='cpu', batch_size=1)

    with patch.object(pipeline, '_translate', return_value=[]):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is False


def test_full_pipeline_fails_when_translation_returns_none(create_test_mkv, tmp_output_dir):
    mkv_file = create_test_mkv(language='eng', track_name='English')

    pipeline = TranslationPipeline(device='cpu', batch_size=1)

    with patch.object(pipeline, '_translate', return_value=None):
        result = pipeline.process_video_file(mkv_file, tmp_output_dir)

    assert result is False


def test_full_pipeline_fails_with_nonexistent_file(tmp_output_dir):
    pipeline = TranslationPipeline(device='cpu', batch_size=1)

    result = pipeline.process_video_file(Path('/nonexistent/video.mkv'), tmp_output_dir)

    assert result is False


class TestFindMkvFiles:
    def test_finds_mkv_in_root_directory(self, tmp_path):
        mkv1 = tmp_path / 'video1.mkv'
        mkv2 = tmp_path / 'video2.mkv'
        mkv1.touch()
        mkv2.touch()

        results = find_mkv_files_with_temp_dirs(tmp_path)

        assert len(results) == 2
        assert results[0][0] == mkv1
        assert results[1][0] == mkv2
        assert results[0][1] == tmp_path / '.translate_temp'
        assert (tmp_path / '.translate_temp').exists()

    def test_finds_mkv_in_subdirectories(self, tmp_path):
        season1 = tmp_path / 'Season 1'
        season2 = tmp_path / 'Season 2'
        season1.mkdir()
        season2.mkdir()

        (season1 / 'ep01.mkv').touch()
        (season1 / 'ep02.mkv').touch()
        (season2 / 'ep01.mkv').touch()

        results = find_mkv_files_with_temp_dirs(tmp_path)

        assert len(results) == 3
        assert results[0][1] == season1 / '.translate_temp'
        assert results[2][1] == season2 / '.translate_temp'
        assert (season1 / '.translate_temp').exists()
        assert (season2 / '.translate_temp').exists()

    def test_returns_empty_for_no_mkv_files(self, tmp_path):
        (tmp_path / 'video.mp4').touch()

        results = find_mkv_files_with_temp_dirs(tmp_path)

        assert results == []

    def test_ignores_hidden_directories(self, tmp_path):
        hidden = tmp_path / '.hidden'
        hidden.mkdir()
        (hidden / 'video.mkv').touch()

        results = find_mkv_files_with_temp_dirs(tmp_path)

        assert results == []

    def test_prefers_root_over_subdirectories(self, tmp_path):
        (tmp_path / 'root.mkv').touch()
        subdir = tmp_path / 'subdir'
        subdir.mkdir()
        (subdir / 'sub.mkv').touch()

        results = find_mkv_files_with_temp_dirs(tmp_path)

        assert len(results) == 1
        assert results[0][0].name == 'root.mkv'
