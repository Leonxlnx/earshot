"""An interrupted or unavailable internal cut cache does not break a valid scan."""
import builtins
import json
import subprocess
from pathlib import Path

import pytest

from earshot import audio, config, video


@pytest.fixture
def film(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    path = str(tmp_path / 'film.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=96x54:r=30:d=1.5',
                    '-f', 'lavfi', '-i', 'color=c=white:s=96x54:r=30:d=1.5', '-filter_complex',
                    '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]', '-c:v', 'mpeg4', path],
                   check=True, capture_output=True)
    assert video.cuts(path, method='scene') == [1.5]
    return path, next((tmp_path / 'emb' / 'video').glob('cuts_*.json'))


@pytest.mark.parametrize('broken', [b'[', b'\xff'])
def test_damaged_scene_cache_is_recomputed(film, broken):
    path, cached = film
    cached.write_bytes(broken)
    assert video.cuts(path, method='scene') == [1.5]
    assert json.loads(cached.read_text()) == [1.5]


def test_valid_scene_cache_avoids_another_scan(film, monkeypatch):
    path, _ = film

    def unexpected_scan(*args, **kwargs):
        pytest.fail('a valid cached scene scan must not call ffmpeg')

    monkeypatch.setattr(video.subprocess, 'run', unexpected_scan)
    assert video.cuts(path, method='scene') == [1.5]


def test_unreadable_scene_cache_is_recomputed(film, monkeypatch):
    path, cached = film

    def unreadable(name, *args, **kwargs):
        if str(name) == str(cached):
            raise PermissionError('cache unavailable')
        return builtins.open(name, *args, **kwargs)

    monkeypatch.setattr(video, 'open', unreadable, raising=False)
    assert video.cuts(path, method='scene') == [1.5]


def test_scene_cache_is_complete_before_replacing_old_result(film, monkeypatch):
    path, cached = film
    cached.write_text('[]')
    replace = audio.replace_quietly
    seen = []

    def check_then_replace(tmp, dst):
        assert Path(dst) == cached
        assert json.loads(cached.read_text()) == []
        assert json.loads(Path(tmp).read_text()) == [1.5]
        seen.append(tmp)
        return replace(tmp, dst)

    monkeypatch.setattr(audio, 'replace_quietly', check_then_replace)
    assert video.cuts(path, method='scene', cache=False) == [1.5]
    assert len(seen) == 1
    assert json.loads(cached.read_text()) == [1.5]
    assert not list(cached.parent.glob('*.tmp'))


def test_failed_scene_cache_write_keeps_successful_scan(film, monkeypatch):
    path, cached = film
    cached.write_text('[]')

    def unavailable(**kwargs):
        raise PermissionError('cache unavailable')

    monkeypatch.setattr(video.tempfile, 'NamedTemporaryFile', unavailable)
    assert video.cuts(path, method='scene', cache=False) == [1.5]
    assert json.loads(cached.read_text()) == []
    assert not list(cached.parent.glob('*.tmp'))


@pytest.mark.parametrize('error', [OSError('disk full'), TypeError('unexpected serialization error')])
def test_scene_cache_write_cleanup_does_not_hide_programming_errors(film, monkeypatch, error):
    path, cached = film
    cached.write_text('[]')

    def failed_write(value, stream):
        stream.write('[')
        raise error

    monkeypatch.setattr(video.json, 'dump', failed_write)
    if isinstance(error, OSError):
        assert video.cuts(path, method='scene', cache=False) == [1.5]
    else:
        with pytest.raises(TypeError, match='unexpected serialization error'):
            video.cuts(path, method='scene', cache=False)
    assert json.loads(cached.read_text()) == []
    assert not list(cached.parent.glob('*.tmp'))
