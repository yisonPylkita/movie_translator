from movie_translator.video import VideoOperations


class TestVideoOperations:
    def test_create_clean_video(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        result = ops.create_clean_video(mkv_file, english_ass, polish_ass, output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_verify_result_correct_order(self, create_test_mkv, create_ass_file, tmp_path):
        mkv_file = create_test_mkv()
        english_ass = create_ass_file('english.ass')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        ops = VideoOperations()
        ops.create_clean_video(mkv_file, english_ass, polish_ass, output_path)

        result = ops.verify_result(output_path)

        assert result is True

    def test_verify_result_detects_missing_tracks(self, create_test_mkv):
        mkv_file = create_test_mkv()

        ops = VideoOperations()
        result = ops.verify_result(mkv_file)

        assert result is False
