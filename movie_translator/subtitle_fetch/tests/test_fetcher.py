from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.fetcher import SubtitleFetcher
from movie_translator.subtitle_fetch.types import SubtitleMatch


def _make_identity():
    return MediaIdentity(
        title='Test',
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
