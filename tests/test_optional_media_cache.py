"""Cache filesystem failures must not turn successful media work into failed calls."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import wave

import numpy as np
import pytest

from earshot import audio, config, video
from earshot import timeline as tl


@pytest.fixture
def media(tmp_path, monkeypatch):
    home = tmp_path / 'cache'
    for key, value in [('HOME', home), ('HF_CACHE', home/'hf'), ('EMB_CACHE', home/'emb'),
                       ('DEC_CACHE', home/'decoded')]:
        monkeypatch.setattr(config, key, str(value))
    sound = tmp_path / 'silence.wav'
    with wave.open(str(sound), 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(audio.SR)
        f.writeframes(bytes(2 * audio.SR * 4))
    film = tmp_path / 'film.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=96x54:r=25:d=1',
                    '-f', 'lavfi', '-i', 'color=c=white:s=96x54:r=25:d=1', '-filter_complex',
                    '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]', '-c:v', 'mpeg4', str(film)],
                   check=True, capture_output=True)
    return home, str(sound), str(film)


@contextmanager
def readonly(home):
    folders = [home] + [p for p in home.rglob('*') if p.is_dir()]
    modes = [(p, p.stat().st_mode & 0o777) for p in folders]
    try:
        for p, _ in modes:
            p.chmod(0o555)
        if os.access(home, os.W_OK):
            pytest.skip('host does not enforce directory write permissions for this user')
        yield
    finally:
        for p, mode in modes:
            p.chmod(mode)


def run(action, sound, film):
    if action == 'audio':
        assert audio.load(sound).shape == (2 * audio.SR, 2)
    elif action in ('diff', 'scene'):
        assert video.cuts(film, method=action) == [1.0]
    else:
        r = tl.timeline(sound if action == 'timeline_audio' else film)
        assert r['duration'] == 2 and r['loudness']['digital_silence']
        assert all(v == 0 for v in r['model_calls'].values())


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('action', ['audio', 'diff', 'scene', 'timeline_audio', 'timeline_video'])
def test_readonly_cache_does_not_fail_media(media, existing, action):
    home, sound, film = media
    home.mkdir()
    if existing:
        config.ensure_dirs()
        (home/'emb'/'video').mkdir()
        (home/'emb'/'timeline').mkdir()
    with readonly(home):
        run(action, sound, film)
    assert not [p for p in home.rglob('*') if p.is_file()]


@pytest.mark.parametrize('action', ['audio', 'diff', 'scene', 'timeline_audio', 'timeline_video'])
def test_readonly_populated_cache_is_still_reused(media, monkeypatch, action):
    home, sound, film = media
    run(action, sound, film)
    assert any(p.is_file() for p in home.rglob('*'))
    def unexpected(*args, **kwargs):
        raise AssertionError('cache hit must not recompute')
    if action == 'audio':
        monkeypatch.setattr(audio, '_decode', unexpected)
    elif action in ('diff', 'scene'):
        monkeypatch.setattr(subprocess, 'run', unexpected)
    else:
        monkeypatch.setattr(tl, '_analyse', unexpected)
    with readonly(home):
        run(action, sound, film)


def test_span_cache_write_is_optional(tmp_path):
    model = SimpleNamespace(dir=str(tmp_path))
    with readonly(tmp_path):
        result = tl._cached(model, 'test', [(0., 1.)], 'test', lambda spans: [np.array([1., 2.])])
    np.testing.assert_array_equal(result, [[1., 2.]])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('error', [OSError('disk full'), ValueError('invalid value')])
def test_partial_cache_write_preserves_old_cache_and_cleans_up(tmp_path, error):
    path = tmp_path/'cached.npy'
    path.write_bytes(b'previous cache')
    def fail(tmp):
        Path(tmp).write_bytes(b'partial')
        raise error
    if isinstance(error, OSError):
        assert audio.write_cache(str(path), fail) is False
    else:
        with pytest.raises(ValueError, match='invalid value'):
            audio.write_cache(str(path), fail)
    assert path.read_bytes() == b'previous cache'
    assert list(tmp_path.iterdir()) == [path]


def test_cache_writer_keeps_numpy_extension_and_replaces_atomically(tmp_path):
    path = tmp_path/'new'/'cached.npz'
    assert audio.write_cache(str(path), lambda tmp: np.savez(tmp, values=[1, 2])) is True
    with np.load(path) as data:
        np.testing.assert_array_equal(data['values'], [1, 2])
    assert list(path.parent.iterdir()) == [path]


def test_decode_failure_is_not_hidden_by_readonly_cache(media):
    home, _, film = media
    home.mkdir()
    Path(film).write_bytes(b'not media')
    with readonly(home):
        with pytest.raises(audio.AudioError):
            audio.load(film)
        with pytest.raises(audio.AudioError):
            video.cuts(film)
