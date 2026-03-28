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
        assert result.fetched_subtitles['pol'].source == 'animesub'

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
