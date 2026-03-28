from pathlib import Path
from unittest.mock import patch

import pytest

from movie_translator.ffmpeg import VideoMuxError, get_mkvmerge
from movie_translator.types import SubtitleFile
from movie_translator.video import VideoOperationError, VideoOperations


def _make_subtitle_files(polish_ass: Path, english_ass: Path) -> list[SubtitleFile]:
    return [
        SubtitleFile(polish_ass, 'pol', 'Polish (AI)', is_default=True),
        SubtitleFile(english_ass, 'eng', 'English Dialogue', is_default=False),
    ]


class TestVideoOperations:
    def test_create_clean_video(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        subs = _make_subtitle_files(polish_ass, english_ass)
        ops.create_clean_video(mkv_file, subs, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_verify_result_correct_order(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        subs = _make_subtitle_files(polish_ass, english_ass)
        ops.create_clean_video(mkv_file, subs, output_path)

        ops.verify_result(output_path, expected_tracks=subs)

    def test_verify_result_with_three_tracks(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        downloaded_pol = create_ass_file('downloaded_pol.ass')
        ai_pol = create_ass_file('ai_pol.ass')
        english_ass = create_ass_file('english.ass')
        output_path = tmp_path / 'output.mkv'

        subs = [
            SubtitleFile(downloaded_pol, 'pol', 'Polish', is_default=True),
            SubtitleFile(ai_pol, 'pol', 'Polish (AI)', is_default=False),
            SubtitleFile(english_ass, 'eng', 'English Dialogue', is_default=False),
        ]

        ops = VideoOperations()
        ops.create_clean_video(mkv_file, subs, output_path)
        ops.verify_result(output_path, expected_tracks=subs)

    def test_verify_result_wrong_track_count(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        polish_ass = create_ass_file('polish.ass')
        english_ass = create_ass_file('english.ass')
        output_path = tmp_path / 'output.mkv'

        subs = _make_subtitle_files(polish_ass, english_ass)
        ops = VideoOperations()
        ops.create_clean_video(mkv_file, subs, output_path)

        # Expect 3 but only 2 exist
        wrong_expected = [
            SubtitleFile(polish_ass, 'pol', 'Polish', is_default=True),
            SubtitleFile(polish_ass, 'pol', 'Polish (AI)', is_default=False),
            SubtitleFile(english_ass, 'eng', 'English Dialogue', is_default=False),
        ]
        with pytest.raises(VideoOperationError, match='Expected 3 subtitle tracks, found 2'):
            ops.verify_result(output_path, expected_tracks=wrong_expected)

    def test_create_clean_video_raises_for_nonexistent_input(self, create_ass_file, tmp_path):
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        subs = _make_subtitle_files(polish_ass, english_ass)
        ops = VideoOperations()
        with pytest.raises(VideoMuxError, match='not found'):
            ops.create_clean_video(Path('/nonexistent/video.mkv'), subs, output_path)

    def test_verify_result_raises_for_nonexistent_file(self):
        ops = VideoOperations()
        with pytest.raises(VideoOperationError, match='not found'):
            ops.verify_result(Path('/nonexistent/video.mkv'))

    def test_create_clean_video_with_font_attachment(
        self, create_test_mkv, create_ass_file, tmp_path
    ):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        from movie_translator.fonts import find_system_font_for_polish

        system_font = find_system_font_for_polish({'arial'})
        if system_font is None:
            pytest.skip('No system font with Polish support available')

        font_path, _ = system_font
        subs = _make_subtitle_files(polish_ass, english_ass)
        ops = VideoOperations()
        ops.create_clean_video(mkv_file, subs, output_path, font_attachments=[font_path])

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        from movie_translator.fonts import get_embedded_fonts

        fonts = get_embedded_fonts(output_path)
        assert len(fonts) >= 1

    def test_create_clean_video_mp4_output(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mp4'

        subs = _make_subtitle_files(polish_ass, english_ass)
        ops = VideoOperations()
        ops.create_clean_video(mkv_file, subs, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestMuxBackendSelection:
    """Tests for mkvmerge vs ffmpeg backend selection and fallback."""

    def test_mkv_uses_mkvmerge_when_available(self, create_test_mkv, create_ass_file, tmp_path):
        if get_mkvmerge() is None:
            pytest.skip('mkvmerge not installed')

        mkv_file = create_test_mkv()
        subs = _make_subtitle_files(create_ass_file('pol.ass'), create_ass_file('eng.ass'))
        output_path = tmp_path / 'output.mkv'

        with patch('movie_translator.ffmpeg._mux_with_mkvmerge', wraps=_mux_with_mkvmerge_ref()) as mock_mkv, \
             patch('movie_translator.ffmpeg._mux_with_ffmpeg') as mock_ff:
            from movie_translator.ffmpeg import mux_video_with_subtitles
            mux_video_with_subtitles(mkv_file, subs, output_path)
            mock_mkv.assert_called_once()
            mock_ff.assert_not_called()

    def test_mkv_falls_back_to_ffmpeg_without_mkvmerge(
        self, create_test_mkv, create_ass_file, tmp_path
    ):
        mkv_file = create_test_mkv()
        subs = _make_subtitle_files(create_ass_file('pol.ass'), create_ass_file('eng.ass'))
        output_path = tmp_path / 'output.mkv'

        with patch('movie_translator.ffmpeg.get_mkvmerge', return_value=None):
            from movie_translator.ffmpeg import mux_video_with_subtitles
            mux_video_with_subtitles(mkv_file, subs, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_mp4_always_uses_ffmpeg(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        subs = _make_subtitle_files(create_ass_file('pol.ass'), create_ass_file('eng.ass'))
        output_path = tmp_path / 'output.mp4'

        with patch('movie_translator.ffmpeg._mux_with_mkvmerge') as mock_mkv:
            from movie_translator.ffmpeg import mux_video_with_subtitles
            mux_video_with_subtitles(mkv_file, subs, output_path)
            mock_mkv.assert_not_called()

        assert output_path.exists()

    def test_mkvmerge_output_has_correct_tracks(self, create_test_mkv, create_ass_file, tmp_path):
        if get_mkvmerge() is None:
            pytest.skip('mkvmerge not installed')

        mkv_file = create_test_mkv()
        subs = [
            SubtitleFile(create_ass_file('pol.ass'), 'pol', 'Polish', is_default=True),
            SubtitleFile(create_ass_file('ai.ass'), 'pol', 'Polish (AI)', is_default=False),
            SubtitleFile(create_ass_file('eng.ass'), 'eng', 'English Dialogue', is_default=False),
        ]
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        ops.create_clean_video(mkv_file, subs, output_path)
        ops.verify_result(output_path, expected_tracks=subs)

    def test_mkvmerge_strips_original_subtitles(self, create_test_mkv, create_ass_file, tmp_path):
        if get_mkvmerge() is None:
            pytest.skip('mkvmerge not installed')

        mkv_file = create_test_mkv()
        subs = _make_subtitle_files(create_ass_file('pol.ass'), create_ass_file('eng.ass'))
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        ops.create_clean_video(mkv_file, subs, output_path)

        from movie_translator.ffmpeg import get_video_info
        info = get_video_info(output_path)
        sub_streams = [s for s in info['streams'] if s['codec_type'] == 'subtitle']
        # Should have exactly our 2 tracks, not the original + our 2
        assert len(sub_streams) == 2


def _mux_with_mkvmerge_ref():
    """Return the real _mux_with_mkvmerge for wraps= usage."""
    from movie_translator.ffmpeg import _mux_with_mkvmerge
    return _mux_with_mkvmerge
