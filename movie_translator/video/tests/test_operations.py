from pathlib import Path

import pytest

from movie_translator.ffmpeg import VideoMuxError
from movie_translator.video import VideoOperationError, VideoOperations


class TestVideoOperations:
    def test_create_clean_video(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        ops.create_clean_video(mkv_file, english_ass, polish_ass, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_verify_result_correct_order(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        ops.create_clean_video(mkv_file, english_ass, polish_ass, output_path)

        ops.verify_result(output_path)

    def test_verify_result_raises_for_missing_tracks(self, create_test_mkv):
        mkv_file = create_test_mkv()

        ops = VideoOperations()
        with pytest.raises(VideoOperationError, match='Expected 2 subtitle tracks'):
            ops.verify_result(mkv_file)

    def test_create_clean_video_raises_for_nonexistent_input(self, create_ass_file, tmp_path):
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        with pytest.raises(VideoMuxError, match='not found'):
            ops.create_clean_video(
                Path('/nonexistent/video.mkv'), english_ass, polish_ass, output_path
            )

    def test_verify_result_raises_for_nonexistent_file(self):
        ops = VideoOperations()
        with pytest.raises(VideoOperationError, match='not found'):
            ops.verify_result(Path('/nonexistent/video.mkv'))
