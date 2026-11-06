"""Media glob expansion must return files, not directories with media suffixes."""
from earshot import audio, cli


def test_media_glob_excludes_directories(tmp_path, wav, sine):
    path = wav(sine, 'clip.wav')
    (tmp_path / 'folder.wav').mkdir()
    assert audio.list_media(str(tmp_path / '*.wav')) == [path]


def test_rank_rejects_media_glob_with_only_directories(tmp_path, capsys):
    (tmp_path / 'folder.wav').mkdir()
    pattern = str(tmp_path / '*.wav')
    assert cli.main(['rank', 'a click', pattern]) == 1
    assert 'nothing playable' in capsys.readouterr().err


def test_media_directory_inputs_still_expand(tmp_path, wav, sine):
    (tmp_path / 'folder.wav').mkdir()
    path = wav(sine, 'folder.wav/clip.wav')
    assert audio.list_media(str(tmp_path / 'folder.wav')) == [path]
