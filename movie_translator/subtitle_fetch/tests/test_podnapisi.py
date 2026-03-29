from unittest.mock import patch

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.providers.podnapisi import PodnapisiProvider


def _make_identity(**overrides):
    defaults = {
        'title': 'Breaking Bad',
        'parsed_title': 'Breaking Bad',
        'year': 2008,
        'season': 1,
        'episode': 3,
        'media_type': 'episode',
        'oshash': 'abc123def456abc0',
        'file_size': 1_000_000,
        'raw_filename': 'Breaking.Bad.S01E03.mkv',
    }
    defaults.update(overrides)
    return MediaIdentity(**defaults)


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<results>
  <pagination><results>2</results></pagination>
  <subtitle>
    <id>12345</id>
    <title>Breaking Bad S01E03</title>
    <release>Breaking.Bad.S01E03.720p.BluRay</release>
    <language>en</language>
    <flags>0</flags>
    <rating>4.8</rating>
    <downloads>5432</downloads>
  </subtitle>
  <subtitle>
    <id>67890</id>
    <title>Breaking Bad S01E03</title>
    <release>Breaking.Bad.S01E03.1080p.WEB</release>
    <language>pl</language>
    <flags>0</flags>
    <rating>4.5</rating>
    <downloads>1234</downloads>
  </subtitle>
</results>"""


class TestPodnapisiProvider:
    def test_name(self):
        assert PodnapisiProvider().name == 'podnapisi'

    def test_search_parses_xml_response(self):
        provider = PodnapisiProvider()
        with patch.object(provider, '_fetch_xml', return_value=SAMPLE_XML):
            matches = provider.search(_make_identity(), ['eng', 'pol'])

        assert len(matches) == 2
        langs = {m.language for m in matches}
        assert langs == {'eng', 'pol'}
        assert matches[0].source == 'podnapisi'

    def test_search_filters_by_language(self):
        provider = PodnapisiProvider()
        with patch.object(provider, '_fetch_xml', return_value=SAMPLE_XML):
            matches = provider.search(_make_identity(), ['pol'])

        assert len(matches) == 1
        assert matches[0].language == 'pol'

    def test_search_includes_season_episode_params_for_episodes(self):
        provider = PodnapisiProvider()
        called_params = {}

        def capture_fetch(url):
            called_params['url'] = url
            return SAMPLE_XML

        with patch.object(provider, '_fetch_xml', side_effect=capture_fetch):
            provider.search(_make_identity(season=2, episode=5), ['eng'])

        assert 'sS=2' in called_params['url']
        assert 'sE=5' in called_params['url']

    def test_search_returns_empty_on_error(self):
        provider = PodnapisiProvider()
        with patch.object(provider, '_fetch_xml', side_effect=Exception('network error')):
            matches = provider.search(_make_identity(), ['eng'])
        assert matches == []
