from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.fetcher import SubtitleFetcher
from movie_translator.subtitle_fetch.types import SubtitleMatch


def _make_identity():
    return MediaIdentity(
        title='Test',
        parsed_title='Test',
        year=None,
        season=1,
        episode=1,
        media_type='episode',
        oshash='0' * 16,
        file_size=1000,
        raw_filename='test.mkv',
    )


class FakeProvider:
    def __init__(self, name, matches):
        self._name = name
        self._matches = matches
        self.downloaded = []

    @property
    def name(self):
        return self._name

    def search(self, identity, languages):
        return [m for m in self._matches if m.language in languages]

    def download(self, match, output_path):
        output_path.write_text(f'subtitle content from {self._name}')
        self.downloaded.append(match)
        return output_path


class TestSubtitleFetcher:
    def test_returns_best_match_per_language(self, tmp_path):
        provider = FakeProvider(
            'fake',
            [
                SubtitleMatch('eng', 'fake', '1', 'rel', 'srt', 0.7, False),
                SubtitleMatch('eng', 'fake', '2', 'rel', 'srt', 1.0, True),
                SubtitleMatch('pol', 'fake', '3', 'rel', 'srt', 0.8, False),
            ],
        )
        fetcher = SubtitleFetcher([provider])
        result = fetcher.fetch_subtitles(_make_identity(), ['eng', 'pol'], tmp_path)

        assert 'eng' in result
        assert 'pol' in result
        # Should have downloaded the hash-matched English (id=2) and Polish (id=3)
        ids = [m.subtitle_id for m in provider.downloaded]
        assert '2' in ids
        assert '3' in ids

    def test_returns_empty_when_no_matches(self, tmp_path):
        provider = FakeProvider('fake', [])
        fetcher = SubtitleFetcher([provider])
        result = fetcher.fetch_subtitles(_make_identity(), ['eng'], tmp_path)
        assert result == {}

    def test_tries_multiple_providers(self, tmp_path):
        p1 = FakeProvider('p1', [])
        p2 = FakeProvider(
            'p2',
            [
                SubtitleMatch('eng', 'p2', '99', 'rel', 'srt', 0.7, False),
            ],
        )
        fetcher = SubtitleFetcher([p1, p2])
        result = fetcher.fetch_subtitles(_make_identity(), ['eng'], tmp_path)

        assert 'eng' in result
        assert len(p2.downloaded) == 1

    def test_prefers_hash_match_over_query_match(self, tmp_path):
        provider = FakeProvider(
            'fake',
            [
                SubtitleMatch('eng', 'fake', 'query', 'rel', 'srt', 0.7, False),
                SubtitleMatch('eng', 'fake', 'hash', 'rel', 'srt', 1.0, True),
            ],
        )
        fetcher = SubtitleFetcher([provider])
        fetcher.fetch_subtitles(_make_identity(), ['eng'], tmp_path)

        assert provider.downloaded[0].subtitle_id == 'hash'


class TestSearchAll:
    def test_search_all_returns_all_matches(self):
        """search_all should return all matches from all providers, not just the best per language."""
        p1 = FakeProvider(
            'p1',
            [
                SubtitleMatch('eng', 'p1', 'a', 'rel-a', 'srt', 0.9, True),
                SubtitleMatch('pol', 'p1', 'b', 'rel-b', 'srt', 0.7, False),
            ],
        )
        p2 = FakeProvider(
            'p2',
            [
                SubtitleMatch('eng', 'p2', 'c', 'rel-c', 'srt', 0.6, False),
            ],
        )
        fetcher = SubtitleFetcher([p1, p2])
        results = fetcher.search_all(_make_identity(), ['eng', 'pol'])

        assert len(results) == 3
        ids = {m.subtitle_id for m in results}
        assert ids == {'a', 'b', 'c'}

    def test_search_all_sorted_by_score_descending(self):
        """search_all results should be sorted by score descending."""
        provider = FakeProvider(
            'fake',
            [
                SubtitleMatch('eng', 'fake', 'low', 'rel', 'srt', 0.3, False),
                SubtitleMatch('eng', 'fake', 'high', 'rel', 'srt', 0.9, True),
                SubtitleMatch('pol', 'fake', 'mid', 'rel', 'srt', 0.6, False),
            ],
        )
        fetcher = SubtitleFetcher([provider])
        results = fetcher.search_all(_make_identity(), ['eng', 'pol'])

        scores = [m.score for m in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_all_returns_empty_when_no_matches(self):
        provider = FakeProvider('fake', [])
        fetcher = SubtitleFetcher([provider])
        results = fetcher.search_all(_make_identity(), ['eng'])
        assert results == []

    def test_search_all_tolerates_provider_failure(self):
        """Providers that raise should be skipped, not crash search_all."""
        from unittest.mock import MagicMock

        bad = MagicMock()
        bad.name = 'bad'
        bad.search.side_effect = RuntimeError('network error')

        good = FakeProvider(
            'good',
            [SubtitleMatch('eng', 'good', '1', 'rel', 'srt', 0.8, False)],
        )
        fetcher = SubtitleFetcher([bad, good])
        results = fetcher.search_all(_make_identity(), ['eng'])

        assert len(results) == 1
        assert results[0].subtitle_id == '1'


class TestDownloadCandidate:
    def test_download_candidate_delegates_to_provider(self, tmp_path):
        """download_candidate should delegate to the correct provider."""
        from unittest.mock import MagicMock

        provider = MagicMock()
        provider.name = 'myprovider'

        fetcher = SubtitleFetcher([provider])
        match = SubtitleMatch('eng', 'myprovider', 'sub-123', 'rel', 'srt', 0.9, True)
        output_path = tmp_path / 'output.srt'

        result = fetcher.download_candidate(match, output_path)

        provider.download.assert_called_once_with(match, output_path)
        assert result == output_path

    def test_download_candidate_picks_correct_provider_by_name(self, tmp_path):
        """download_candidate should use source name to select the right provider."""
        from unittest.mock import MagicMock

        p1 = MagicMock()
        p1.name = 'provider_a'
        p2 = MagicMock()
        p2.name = 'provider_b'

        fetcher = SubtitleFetcher([p1, p2])
        match = SubtitleMatch('eng', 'provider_b', 'sub-456', 'rel', 'srt', 0.8, False)
        output_path = tmp_path / 'out.srt'

        fetcher.download_candidate(match, output_path)

        p1.download.assert_not_called()
        p2.download.assert_called_once_with(match, output_path)

    def test_download_candidate_raises_when_provider_not_found(self, tmp_path):
        """download_candidate should raise ValueError if provider is not registered."""
        provider = FakeProvider('only_provider', [])
        fetcher = SubtitleFetcher([provider])
        match = SubtitleMatch('eng', 'missing_provider', 'sub-789', 'rel', 'srt', 0.7, False)

        import pytest
        with pytest.raises(ValueError, match='missing_provider'):
            fetcher.download_candidate(match, tmp_path / 'out.srt')
