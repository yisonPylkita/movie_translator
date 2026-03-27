from unittest.mock import patch

from movie_translator.ocr.probe import probe_for_burned_in_subtitles


class TestProbeForBurnedInSubtitles:
    def test_returns_true_when_text_found(self, tmp_path):
        video = tmp_path / 'test.mkv'
        video.write_bytes(b'\x00' * 100)

        def fake_ffmpeg_run(cmd, **kwargs):
            """Create the frame file that FFmpeg would produce."""
            # Find the output path (last argument)
            output = cmd[-1]
            from pathlib import Path

            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_bytes(b'\xff\xd8\xff\xe0')  # Minimal JPEG header
            from unittest.mock import MagicMock

            return MagicMock(returncode=0)

        with (
            patch(
                'movie_translator.ocr.probe.get_video_info',
                return_value={'format': {'duration': '300'}},
            ),
            patch('movie_translator.ocr.probe.subprocess.run', side_effect=fake_ffmpeg_run),
            patch('movie_translator.ocr.probe.recognize_text', return_value='Hello world'),
        ):
            result = probe_for_burned_in_subtitles(video, num_samples=5)

        assert result is True

    def test_returns_false_when_no_text_found(self, tmp_path):
        video = tmp_path / 'test.mkv'
        video.write_bytes(b'\x00' * 100)

        with (
            patch(
                'movie_translator.ocr.probe.get_video_info',
                return_value={'format': {'duration': '300'}},
            ),
            patch('movie_translator.ocr.probe.subprocess.run') as mock_run,
            patch('movie_translator.ocr.probe.recognize_text', return_value=''),
        ):
            mock_run.return_value.returncode = 0
            result = probe_for_burned_in_subtitles(video, num_samples=5)

        assert result is False

    def test_returns_true_when_duration_unknown(self, tmp_path):
        """When we can't determine duration, assume subtitles might exist."""
        video = tmp_path / 'test.mkv'
        video.write_bytes(b'\x00' * 100)

        with patch(
            'movie_translator.ocr.probe.get_video_info',
            return_value={'format': {}},
        ):
            result = probe_for_burned_in_subtitles(video)

        assert result is True
