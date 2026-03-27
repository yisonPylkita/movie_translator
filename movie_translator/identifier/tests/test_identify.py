from unittest.mock import patch

from movie_translator.identifier import identify_media
from movie_translator.identifier.types import MediaIdentity


class TestIdentifyMedia:
    def test_combines_filename_and_hash(self, tmp_path):
        video = tmp_path / 'Breaking.Bad.S01E03.720p.mkv'
        video.write_bytes(b'\x00' * 1024)

        with patch(
            'movie_translator.identifier.identify.extract_container_metadata',
            return_value={'title': None, 'episode': None},
        ):
            result = identify_media(video)

        assert isinstance(result, MediaIdentity)
        assert result.title == 'Breaking Bad'
        assert result.season == 1
        assert result.episode == 3
        assert result.media_type == 'episode'
        assert len(result.oshash) == 16
        assert result.file_size == 1024

    def test_container_metadata_overrides_filename(self, tmp_path):
        video = tmp_path / 'random_name.mkv'
        video.write_bytes(b'\x00' * 1024)

        with patch(
            'movie_translator.identifier.identify.extract_container_metadata',
            return_value={'title': 'The Real Title', 'episode': None},
        ):
            result = identify_media(video)

        assert result.title == 'The Real Title'

    def test_folder_name_fills_missing_title(self, tmp_path):
        folder = tmp_path / 'One Piece'
        folder.mkdir()
        video = folder / 'Episode 05 [1080p].mkv'
        video.write_bytes(b'\x00' * 1024)

        with patch(
            'movie_translator.identifier.identify.extract_container_metadata',
            return_value={'title': None, 'episode': None},
        ):
            result = identify_media(video)

        assert result.title is not None
        assert result.raw_filename == 'Episode 05 [1080p].mkv'
