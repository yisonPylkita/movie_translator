from unittest.mock import patch

from movie_translator.identifier.metadata import extract_container_metadata


class TestExtractContainerMetadata:
    def test_extracts_title_from_format_tags(self):
        mock_info = {
            'format': {
                'tags': {'title': 'One Piece Episode 101'},
            },
            'streams': [],
        }
        with patch('movie_translator.identifier.metadata.get_video_info', return_value=mock_info):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] == 'One Piece Episode 101'

    def test_returns_none_for_missing_tags(self):
        mock_info = {'format': {}, 'streams': []}
        with patch('movie_translator.identifier.metadata.get_video_info', return_value=mock_info):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] is None
        assert result['episode'] is None

    def test_extracts_episode_tag(self):
        mock_info = {
            'format': {
                'tags': {'title': 'One Piece', 'episode_id': '101'},
            },
            'streams': [],
        }
        with patch('movie_translator.identifier.metadata.get_video_info', return_value=mock_info):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] == 'One Piece'

    def test_handles_ffprobe_failure_gracefully(self):
        with patch(
            'movie_translator.identifier.metadata.get_video_info',
            side_effect=Exception('ffprobe failed'),
        ):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] is None
