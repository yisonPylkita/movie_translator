from movie_translator.main import create_working_dirs


def test_creates_per_anime_per_episode_dirs(tmp_path):
    """Working dirs should nest as .translate_temp/<anime>/<episode>/."""
    anime_dir = tmp_path / 'Aho-Girl'
    anime_dir.mkdir()
    video = anime_dir / 'Aho-Girl - 01.mkv'
    video.touch()

    work_dir = create_working_dirs(video, tmp_path)

    assert work_dir.parent.name == 'Aho-Girl'  # per-anime parent
    assert work_dir.name == 'Aho-Girl - 01'  # per-episode dir
    assert work_dir.parent.parent.name == '.translate_temp'
    assert (work_dir / 'candidates').is_dir()
    assert (work_dir / 'reference').is_dir()


def test_creates_subdirs(tmp_path):
    """Working dir should have candidates/ and reference/ subdirs."""
    anime_dir = tmp_path / 'Show'
    anime_dir.mkdir()
    video = anime_dir / 'Show - 01.mkv'
    video.touch()

    work_dir = create_working_dirs(video, tmp_path)

    assert (work_dir / 'candidates').is_dir()
    assert (work_dir / 'reference').is_dir()
