"""A failed scene scan is an error, not a film with no cuts. No models needed."""
import hashlib
import subprocess

import pytest

from earshot import audio, config, video


def test_failed_scene_scan_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    path = tmp_path / 'broken.mp4'
    path.write_bytes(b'not a video')
    with pytest.raises(audio.AudioError, match='broken.mp4'):
        video.cuts(str(path), method='scene')
    assert not list((tmp_path / 'emb' / 'video').glob('cuts_*.json'))


def test_scene_scan_can_succeed_with_no_cuts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    path = str(tmp_path / 'still.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=96x54:r=30:d=1',
                    '-c:v', 'mpeg4', path], check=True, capture_output=True)
    assert video.cuts(path, method='scene') == []
    assert len(list((tmp_path / 'emb' / 'video').glob('cuts_*.json'))) == 1


def test_scene_scan_ignores_a_legacy_cached_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    path = tmp_path / 'broken.mp4'
    path.write_bytes(b'not a video')
    key = hashlib.sha1(f'{audio.file_id(str(path))}|0.3|0.2|v1'.encode()).hexdigest()[:20]
    folder = tmp_path / 'emb' / 'video'
    folder.mkdir(parents=True)
    legacy = folder / f'cuts_{key}.json'
    legacy.write_text('[]')
    with pytest.raises(audio.AudioError, match='broken.mp4'):
        video.cuts(str(path), method='scene')
    assert list(folder.glob('cuts_*.json')) == [legacy]
