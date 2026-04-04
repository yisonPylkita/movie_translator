from unittest.mock import MagicMock, patch

from movie_translator.context import PipelineConfig, PipelineContext
from movie_translator.stages.extract_ref import ExtractReferenceStage
from movie_translator.subtitles import SubtitleExtractor


class TestExtractReferenceStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'reference').mkdir()
        return PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )

    def test_extracts_embedded_english_track(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        mock_extractor = MagicMock()
        mock_extractor.get_track_info.return_value = {'tracks': [{'type': 'subtitles'}]}
        mock_extractor.find_english_track.return_value = {
            'id': 2,
            'codec': 'subrip',
            'properties': {'language': 'eng'},
            'subtitle_index': 0,
        }
        mock_extractor.get_subtitle_extension.return_value = '.srt'
        mock_extractor.extract_subtitle.return_value = None

        with patch(
            'movie_translator.stages.extract_ref.SubtitleExtractor', return_value=mock_extractor
        ):
            result = ExtractReferenceStage().run(ctx)

        assert result.reference_path is not None
        assert result.original_english_track is not None
        assert result.original_english_track.stream_index == 2
        assert result.original_english_track.codec == 'subrip'

    def test_no_english_track_sets_none(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        mock_extractor = MagicMock()
        mock_extractor.get_track_info.return_value = {'tracks': []}
        mock_extractor.find_english_track.return_value = None

        with (
            patch(
                'movie_translator.stages.extract_ref.SubtitleExtractor', return_value=mock_extractor
            ),
            patch(
                'movie_translator.stages.extract_ref.is_vision_ocr_available', return_value=False
            ),
        ):
            result = ExtractReferenceStage().run(ctx)

        assert result.reference_path is None
        assert result.original_english_track is None


class TestExtractRefDeferredOcr:
    def test_text_track_extracts_directly(self, tmp_path):
        """run() with a text-based track should extract without setting pending_ocr."""
        video = tmp_path / 'ep01.mkv'
        video.touch()
        ctx = PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )

        text_track = {
            'id': 2,
            'codec': 'subrip',
            'subtitle_index': 0,
            'properties': {'language': 'eng'},
        }

        with (
            patch.object(SubtitleExtractor, 'get_track_info', return_value=[text_track]),
            patch.object(SubtitleExtractor, 'find_english_track', return_value=text_track),
            patch.object(SubtitleExtractor, 'get_subtitle_extension', return_value='.srt'),
            patch.object(SubtitleExtractor, 'extract_subtitle'),
        ):
            result = ExtractReferenceStage().run(ctx)

        assert result.pending_ocr is None
        assert result.reference_path is not None

    def test_pgs_track_defers_ocr(self, tmp_path):
        """run() with a PGS track should set pending_ocr instead of doing OCR."""
        video = tmp_path / 'ep01.mkv'
        video.touch()
        ctx = PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )

        pgs_track = {
            'id': 3,
            'codec': 'hdmv_pgs_subtitle',
            'subtitle_index': 0,
            'properties': {'language': 'eng'},
        }

        with (
            patch.object(SubtitleExtractor, 'get_track_info', return_value=[pgs_track]),
            patch.object(SubtitleExtractor, 'find_english_track', return_value=pgs_track),
        ):
            result = ExtractReferenceStage().run(ctx)

        assert result.pending_ocr is not None
        assert result.pending_ocr.type == 'pgs'
        assert result.pending_ocr.track_id == 3
