from unittest.mock import MagicMock, patch

from movie_translator.context import PipelineConfig, PipelineContext
from movie_translator.stages.fetch import FetchSubtitlesStage


class TestFetchSubtitlesStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        (work / 'candidates').mkdir(parents=True)
        return PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
            identity=MagicMock(),
        )

    def test_sets_fetched_subtitles(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        pol_path = tmp_path / 'pol.ass'
        pol_path.touch()

        mock_fetcher = MagicMock()
        mock_match = MagicMock(language='pol', source='animesub', subtitle_id='123', format='ass')
        mock_fetcher.search_all.return_value = [mock_match]
        mock_fetcher.download_candidate.return_value = pol_path

        with (
            patch('movie_translator.stages.fetch.SubtitleFetcher', return_value=mock_fetcher),
            patch('movie_translator.stages.fetch.SubtitleValidator') as MockValidator,
        ):
            mock_validator = MockValidator.return_value
            mock_validator.validate_candidates.return_value = [(mock_match, pol_path, 0.95)]
            result = FetchSubtitlesStage().run(ctx)

        assert result.fetched_subtitles is not None
        assert 'pol' in result.fetched_subtitles
        assert result.fetched_subtitles['pol'][0].source == 'animesub'

    def test_fetch_disabled_skips(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.config.enable_fetch = False
        result = FetchSubtitlesStage().run(ctx)
        assert result.fetched_subtitles is None

    def test_no_matches_sets_empty_dict(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        mock_fetcher = MagicMock()
        mock_fetcher.search_all.return_value = []

        with patch('movie_translator.stages.fetch.SubtitleFetcher', return_value=mock_fetcher):
            result = FetchSubtitlesStage().run(ctx)

        assert result.fetched_subtitles == {}

    def test_aligns_polish_subtitles_when_reference_exists(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.reference_path = tmp_path / 'ref.srt'

        pol_path = tmp_path / 'pol.srt'
        pol_path.touch()

        mock_fetcher = MagicMock()
        mock_match = MagicMock(language='pol', source='animesub', subtitle_id='1', format='srt')
        mock_fetcher.search_all.return_value = [mock_match]
        mock_fetcher.download_candidate.return_value = pol_path

        with (
            patch('movie_translator.stages.fetch.SubtitleFetcher', return_value=mock_fetcher),
            patch('movie_translator.stages.fetch.SubtitleValidator') as MockValidator,
            patch.object(FetchSubtitlesStage, '_align_subtitle') as mock_align,
        ):
            mock_validator = MockValidator.return_value
            mock_validator.validate_candidates.return_value = [(mock_match, pol_path, 0.9)]
            FetchSubtitlesStage().run(ctx)

        mock_align.assert_called_once_with(pol_path, ctx.reference_path)

    def test_keeps_multiple_high_scoring_polish_subs(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.reference_path = tmp_path / 'ref.srt'
        ctx.reference_path.touch()

        pol1 = tmp_path / 'pol1.srt'
        pol1.touch()
        pol2 = tmp_path / 'pol2.srt'
        pol2.touch()

        mock_fetcher = MagicMock()
        match1 = MagicMock(language='pol', source='opensubtitles', subtitle_id='1', format='srt')
        match2 = MagicMock(language='pol', source='podnapisi', subtitle_id='2', format='srt')
        mock_fetcher.search_all.return_value = [match1, match2]

        with (
            patch('movie_translator.stages.fetch.SubtitleFetcher', return_value=mock_fetcher),
            patch('movie_translator.stages.fetch.SubtitleValidator') as MockValidator,
            patch.object(FetchSubtitlesStage, '_align_subtitle'),
        ):
            mock_validator = MockValidator.return_value
            # Both score above 0.8 quality threshold
            mock_validator.validate_candidates.return_value = [
                (match1, pol1, 0.95),
                (match2, pol2, 0.85),
            ]
            result = FetchSubtitlesStage().run(ctx)

        assert len(result.fetched_subtitles['pol']) == 2
        assert result.fetched_subtitles['pol'][0].source == 'opensubtitles'
        assert result.fetched_subtitles['pol'][1].source == 'podnapisi'

    def test_keeps_only_best_when_others_below_threshold(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.reference_path = tmp_path / 'ref.srt'
        ctx.reference_path.touch()

        pol1 = tmp_path / 'pol1.srt'
        pol1.touch()
        pol2 = tmp_path / 'pol2.srt'
        pol2.touch()

        mock_fetcher = MagicMock()
        match1 = MagicMock(language='pol', source='opensubtitles', subtitle_id='1', format='srt')
        match2 = MagicMock(language='pol', source='podnapisi', subtitle_id='2', format='srt')
        mock_fetcher.search_all.return_value = [match1, match2]

        with (
            patch('movie_translator.stages.fetch.SubtitleFetcher', return_value=mock_fetcher),
            patch('movie_translator.stages.fetch.SubtitleValidator') as MockValidator,
            patch.object(FetchSubtitlesStage, '_align_subtitle'),
        ):
            mock_validator = MockValidator.return_value
            # First is best (always kept), second below 0.8 threshold
            mock_validator.validate_candidates.return_value = [
                (match1, pol1, 0.75),
                (match2, pol2, 0.60),
            ]
            result = FetchSubtitlesStage().run(ctx)

        assert len(result.fetched_subtitles['pol']) == 1
