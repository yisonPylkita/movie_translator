import io
import zipfile
from unittest.mock import patch

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.providers.animesub import (
    AnimeSubProvider,
    _entry_matches,
    _extract_season_from_title,
    _ResultParser,
)

SAMPLE_HTML = """
<table class="Napisy">
<tr class="KNap">
  <td align="left" width="45%">Naruto ep001</td>
  <td width="25%">2004.01.03</td>
  <td width="10%">&nbsp;</td>
  <td width="20%">Advanced SSA</td>
</tr>
<tr class="KNap">
  <td align="left">Naruto ep001</td>
  <td><a href="osoba.php?id=36">~Sanzoku</a></td>
  <td></td>
  <td>6kB</td>
</tr>
<tr class="KNap">
  <td align="left">Naruto ep001</td>
  <td><a href="javascript:PK(1022)" class="ko">(3)</a></td>
  <td></td>
  <td>5878 razy</td>
</tr>
<tr class="KKom">
  <td valign="top" align="right">
    <form method="POST" action="sciagnij.php">
      <input type="hidden" name="id" value="1022">
      <input type="hidden" name="sh" value="abc123def456">
      <input type="submit" value="Pobierz napisy" name="single_file">
    </form>
  </td>
  <td class="KNap" align="left" colspan="3">
    <b>ID 1022<br>Autor:</b> Sanzoku
  </td>
</tr>
</table>
<table class="Napisy">
<tr class="KNap">
  <td align="left" width="45%">Naruto ep002</td>
  <td width="25%">2004.01.04</td>
  <td width="10%">&nbsp;</td>
  <td width="20%">SubRip</td>
</tr>
<tr class="KNap"><td></td><td></td><td></td><td></td></tr>
<tr class="KNap"><td></td><td></td><td></td><td></td></tr>
<tr class="KKom">
  <td>
    <form method="POST" action="sciagnij.php">
      <input type="hidden" name="id" value="1023">
      <input type="hidden" name="sh" value="xyz789">
      <input type="submit" value="Pobierz napisy">
    </form>
  </td>
  <td class="KNap" colspan="3"></td>
</tr>
</table>
"""


def _make_identity(**overrides):
    defaults = {
        'title': 'Naruto',
        'parsed_title': 'Naruto',
        'year': None,
        'season': None,
        'episode': 1,
        'media_type': 'episode',
        'oshash': '0' * 16,
        'file_size': 1000,
        'raw_filename': 'Naruto.ep001.mkv',
    }
    defaults.update(overrides)
    return MediaIdentity(**defaults)


class TestResultParser:
    def test_parses_two_entries(self):
        parser = _ResultParser()
        parser.feed(SAMPLE_HTML)
        assert len(parser.entries) == 2

    def test_extracts_id_and_sh(self):
        parser = _ResultParser()
        parser.feed(SAMPLE_HTML)
        assert parser.entries[0]['id'] == '1022'
        assert parser.entries[0]['sh'] == 'abc123def456'
        assert parser.entries[1]['id'] == '1023'
        assert parser.entries[1]['sh'] == 'xyz789'

    def test_extracts_title(self):
        parser = _ResultParser()
        parser.feed(SAMPLE_HTML)
        assert parser.entries[0]['title'] == 'Naruto ep001'
        assert parser.entries[1]['title'] == 'Naruto ep002'

    def test_extracts_format(self):
        parser = _ResultParser()
        parser.feed(SAMPLE_HTML)
        assert parser.entries[0]['format'] == 'Advanced SSA'
        assert parser.entries[1]['format'] == 'SubRip'


class TestExtractSeasonFromTitle:
    BASE = 'Kono Subarashii Sekai ni Shukufuku wo!'

    def test_no_suffix_is_season_1(self):
        assert _extract_season_from_title(self.BASE, f'{self.BASE} ep01') == 1

    def test_number_suffix_2(self):
        assert _extract_season_from_title(self.BASE, f'{self.BASE} 2 ep08') == 2

    def test_number_suffix_3(self):
        assert _extract_season_from_title(self.BASE, f'{self.BASE} 3 ep01') == 3

    def test_explicit_s2(self):
        assert _extract_season_from_title(self.BASE, f'{self.BASE} S2 ep01-10') == 2

    def test_ova_returns_none(self):
        assert _extract_season_from_title(self.BASE, f'{self.BASE} OVA ep01') is None

    def test_movie_returns_none(self):
        assert _extract_season_from_title(self.BASE, f'{self.BASE} Movie') is None

    def test_bonus_stage_returns_none(self):
        assert _extract_season_from_title(self.BASE, f'{self.BASE} 3: Bonus Stage ep01') is None

    def test_different_base_title(self):
        assert _extract_season_from_title('Naruto', 'Naruto Shippuden ep01') is None


class TestEntryMatches:
    BASE = 'Kono Subarashii Sekai ni Shukufuku wo!'

    def test_s1_ep01_matches_no_suffix(self):
        assert _entry_matches(f'{self.BASE} ep01', self.BASE, season=1, episode=1)

    def test_s1_ep08_rejects_s2(self):
        assert not _entry_matches(f'{self.BASE} 2 ep08', self.BASE, season=1, episode=8)

    def test_s1_ep08_rejects_s3(self):
        assert not _entry_matches(f'{self.BASE} 3 ep08', self.BASE, season=1, episode=8)

    def test_s2_ep08_matches_s2(self):
        assert _entry_matches(f'{self.BASE} 2 ep08', self.BASE, season=2, episode=8)

    def test_wrong_episode_rejected(self):
        assert not _entry_matches(f'{self.BASE} ep05', self.BASE, season=1, episode=8)

    def test_season_none_accepts_any_season(self):
        # When season is unknown, accept any
        assert _entry_matches(f'{self.BASE} 2 ep08', self.BASE, season=None, episode=8)

    def test_ova_rejected_when_season_specified(self):
        # OVA is not season 1 — reject it
        assert not _entry_matches(f'{self.BASE} OVA ep01', self.BASE, season=1, episode=1)

    def test_ova_accepted_when_season_none(self):
        # When season is unknown, accept OVAs
        assert _entry_matches(f'{self.BASE} OVA ep01', self.BASE, season=None, episode=1)

    def test_simple_naruto_match(self):
        assert _entry_matches('Naruto ep001', 'Naruto', season=1, episode=1)

    def test_no_episode_pattern_rejected(self):
        assert not _entry_matches('Naruto Movie', 'Naruto', season=1, episode=1)


class TestAnimeSubProvider:
    def test_name(self):
        assert AnimeSubProvider().name == 'animesub'

    def test_skips_non_polish_languages(self):
        provider = AnimeSubProvider()
        result = provider.search(_make_identity(), ['eng'])
        assert result == []

    def test_search_returns_matches(self):
        provider = AnimeSubProvider()
        with patch.object(
            provider,
            '_search_page',
            return_value=[
                {'id': '1022', 'sh': 'abc', 'title': 'Naruto ep001', 'format': 'Advanced SSA'},
            ],
        ):
            matches = provider.search(_make_identity(), ['pol'])

        assert len(matches) == 1
        assert matches[0].language == 'pol'
        assert matches[0].source == 'animesub'
        assert matches[0].subtitle_id == '1022:abc'
        assert matches[0].format == 'ass'
        assert matches[0].score == 0.6

    def test_rejects_wrong_season(self):
        """S2 and S3 results must not match when searching for S1."""
        provider = AnimeSubProvider()
        with patch.object(
            provider,
            '_search_page',
            return_value=[
                {'id': '1', 'sh': 'a', 'title': 'Naruto 2 ep01', 'format': 'SubRip'},
                {'id': '2', 'sh': 'b', 'title': 'Naruto 3 ep01', 'format': 'SubRip'},
                {'id': '3', 'sh': 'c', 'title': 'Naruto ep01', 'format': 'SubRip'},
            ],
        ):
            matches = provider.search(_make_identity(season=1), ['pol'])

        # Only the S1 result (no suffix) should match
        assert len(matches) == 1
        assert matches[0].release_name == 'Naruto ep01'

    def test_accepts_correct_season(self):
        provider = AnimeSubProvider()
        with patch.object(
            provider,
            '_search_page',
            return_value=[
                {'id': '1', 'sh': 'a', 'title': 'Naruto 2 ep01', 'format': 'SubRip'},
                {'id': '2', 'sh': 'b', 'title': 'Naruto ep01', 'format': 'SubRip'},
            ],
        ):
            matches = provider.search(_make_identity(season=2), ['pol'])

        assert len(matches) == 1
        assert matches[0].release_name == 'Naruto 2 ep01'

    def test_srt_format_detected(self):
        provider = AnimeSubProvider()
        with patch.object(
            provider,
            '_search_page',
            return_value=[
                {'id': '1', 'sh': 'x', 'title': 'Test ep01', 'format': 'SubRip'},
            ],
        ):
            matches = provider.search(_make_identity(), ['pol'])
        assert matches[0].format == 'srt'

    def test_download_extracts_from_zip(self, tmp_path):
        # Create a mock ZIP with a subtitle file
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zf:
            zf.writestr('Naruto_01.ass', '[Script Info]\nTitle: Naruto')
        zip_bytes = zip_buf.getvalue()

        from movie_translator.subtitle_fetch.types import SubtitleMatch

        match = SubtitleMatch(
            language='pol',
            source='animesub',
            subtitle_id='1022:abc123',
            release_name='Naruto ep001',
            format='ass',
            score=0.6,
            hash_match=False,
        )

        output = tmp_path / 'subtitle.ass'

        # Mock the opener to return our ZIP
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.read.return_value = zip_bytes
        mock_resp.headers = {'Content-Type': 'application/zip'}
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        provider = AnimeSubProvider()
        provider._opener = MagicMock()
        provider._opener.open.return_value = mock_resp
        result = provider.download(match, output)

        assert result == output
        assert output.exists()
        content = output.read_text()
        assert 'Naruto' in content
