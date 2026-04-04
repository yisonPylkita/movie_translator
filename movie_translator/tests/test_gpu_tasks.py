"""Tests for concrete GPU task implementations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from movie_translator.gpu_queue import InpaintTask, OcrTask, TranslateTask
from movie_translator.types import DialogueLine, OCRResult


class TestTranslateTask:
    def test_model_type(self):
        t = TranslateTask()
        assert t.model_type == 'translate'

    @patch('movie_translator.translation.translate_dialogue_lines')
    def test_execute_calls_translate(self, mock_translate: MagicMock):
        lines = [DialogueLine(0, 1000, 'Hello')]
        expected = [DialogueLine(0, 1000, 'Cześć')]
        mock_translate.return_value = expected

        task = TranslateTask(
            dialogue_lines=lines,
            device='cpu',
            batch_size=16,
            model='test-model',
        )
        result = task.execute({}, None)

        mock_translate.assert_called_once_with(
            dialogue_lines=lines,
            device='cpu',
            batch_size=16,
            model='test-model',
            progress_callback=None,
        )
        assert result == expected


class TestOcrTask:
    def test_model_type(self):
        t = OcrTask()
        assert t.model_type == 'ocr'

    @patch('movie_translator.ocr.pgs_extractor.extract_pgs_track')
    def test_pgs_execute(self, mock_pgs: MagicMock, tmp_path: Path):
        mock_pgs.return_value = tmp_path / 'out.srt'
        task = OcrTask(
            ocr_type='pgs',
            video_path=tmp_path / 'video.mkv',
            track_index=2,
            work_dir=tmp_path,
        )
        result = task.execute({}, None)
        mock_pgs.assert_called_once_with(
            video_path=tmp_path / 'video.mkv',
            track_index=2,
            work_dir=tmp_path,
        )
        assert result == tmp_path / 'out.srt'

    @patch('movie_translator.ocr.extract_burned_in_subtitles')
    def test_burned_in_execute(self, mock_burned: MagicMock, tmp_path: Path):
        mock_burned.return_value = None
        task = OcrTask(
            ocr_type='burned_in',
            video_path=tmp_path / 'video.mkv',
            output_dir=tmp_path,
            crop_ratio=0.3,
            fps=2,
        )
        result = task.execute({}, None)
        mock_burned.assert_called_once_with(
            video_path=tmp_path / 'video.mkv',
            output_dir=tmp_path,
            crop_ratio=0.3,
            fps=2,
        )
        assert result is None


class TestInpaintTask:
    def test_model_type(self):
        t = InpaintTask()
        assert t.model_type == 'inpaint'

    @patch('movie_translator.inpainting.remove_burned_in_subtitles')
    def test_execute_calls_remove(self, mock_remove: MagicMock, tmp_path: Path):
        mock_remove.return_value = None
        ocr_results: list[OCRResult] = []
        task = InpaintTask(
            video_path=tmp_path / 'in.mkv',
            output_path=tmp_path / 'out.mkv',
            ocr_results=ocr_results,
            device='mps',
            backend='opencv-ns',
        )
        result = task.execute({}, None)
        mock_remove.assert_called_once_with(
            video_path=tmp_path / 'in.mkv',
            output_path=tmp_path / 'out.mkv',
            ocr_results=ocr_results,
            device='mps',
            backend='opencv-ns',
        )
        assert result is None
