# movie_translator/tests/test_discovery.py
"""Tests for recursive video discovery and working directory creation."""

from movie_translator.discovery import create_work_dir, find_videos


class TestFindVideos:
    def test_single_file(self, tmp_path):
        mkv = tmp_path / 'video.mkv'
        mkv.touch()
        assert find_videos(mkv) == [mkv]

    def test_single_file_non_video_returns_empty(self, tmp_path):
        txt = tmp_path / 'readme.txt'
        txt.touch()
        assert find_videos(txt) == []

    def test_flat_directory(self, tmp_path):
        a = tmp_path / 'a.mkv'
        b = tmp_path / 'b.mp4'
        a.touch()
        b.touch()
        result = find_videos(tmp_path)
        assert result == [a, b]

    def test_nested_anime_structure(self, tmp_path):
        # Anime/Show/Season 1/ep01.mkv
        s1 = tmp_path / 'Show' / 'Season 1'
        s1.mkdir(parents=True)
        ep1 = s1 / 'ep01.mkv'
        ep2 = s1 / 'ep02.mkv'
        ep1.touch()
        ep2.touch()

        s2 = tmp_path / 'Show' / 'Season 2'
        s2.mkdir(parents=True)
        ep3 = s2 / 'ep01.mkv'
        ep3.touch()

        result = find_videos(tmp_path)
        assert len(result) == 3
        assert ep1 in result
        assert ep2 in result
        assert ep3 in result

    def test_skips_hidden_directories(self, tmp_path):
        hidden = tmp_path / '.translate_temp'
        hidden.mkdir()
        (hidden / 'temp.mkv').touch()
        visible = tmp_path / 'ep01.mkv'
        visible.touch()
        assert find_videos(tmp_path) == [visible]

    def test_sorted_output(self, tmp_path):
        c = tmp_path / 'c.mkv'
        a = tmp_path / 'a.mkv'
        b = tmp_path / 'b.mkv'
        c.touch()
        a.touch()
        b.touch()
        assert find_videos(tmp_path) == [a, b, c]

    def test_empty_directory(self, tmp_path):
        assert find_videos(tmp_path) == []

    def test_nonexistent_path(self, tmp_path):
        assert find_videos(tmp_path / 'nope') == []


class TestCreateWorkDir:
    def test_creates_work_dir_with_subdirs(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = create_work_dir(video, tmp_path)
        assert work.exists()
        assert (work / 'candidates').exists()
        assert (work / 'reference').exists()

    def test_preserves_relative_structure(self, tmp_path):
        video = tmp_path / 'Show' / 'S1' / 'ep01.mkv'
        video.parent.mkdir(parents=True)
        video.touch()
        work = create_work_dir(video, tmp_path)
        assert '.translate_temp' in str(work)
        assert 'Show' in str(work)
        assert 'S1' in str(work)
        assert 'ep01' in str(work)
