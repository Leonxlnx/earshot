"""Fresh timelines bypass decoded input caches too, without loading any models."""
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from earshot import audio, config
from earshot.timeline import timeline


@pytest.fixture
def caches(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    monkeypatch.setattr(config, 'DEC_CACHE', str(tmp_path / 'decoded'))


@pytest.mark.parametrize('options', [{'fresh': True}, {'cache': False}])
def test_timeline_bypasses_saved_frame_differences(tmp_path, caches, options):
    path = str(tmp_path / 'film.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=96x54:r=30:d=1.5',
                    '-f', 'lavfi', '-i', 'color=c=white:s=96x54:r=30:d=1.5', '-filter_complex',
                    '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]', '-c:v', 'mpeg4', path],
                   check=True, capture_output=True)
    assert timeline(path)['cuts'] == [1.5]
    cached = next(Path(config.EMB_CACHE).glob('video/diff_*.npz'))
    with np.load(cached) as saved:
        fps, d = float(saved['fps']), np.zeros_like(saved['d'])
    np.savez(cached, fps=fps, d=d)
    result = timeline(path, **options)
    assert result['cached'] is False
    assert result['cuts'] == [1.5]


@pytest.mark.parametrize('options', [{'fresh': True}, {'cache': False}])
@pytest.mark.parametrize('bounds, duration', [({}, 3.0), ({'start': 1.5, 'end': 2.5}, 1.0)])
def test_timeline_bypasses_saved_audio_decode(tmp_path, caches, options, bounds, duration):
    path = str(tmp_path / 'silence.wav')
    with wave.open(path, 'wb') as sound:
        sound.setnchannels(2)
        sound.setsampwidth(2)
        sound.setframerate(audio.SR)
        sound.writeframes(b'\0' * (audio.SR * 3 * 4))
    assert len(audio.load(path)) == audio.SR * 3
    cached = Path(config.DEC_CACHE) / (audio._key(path, audio.SR, 2) + '.npy')
    np.save(cached, np.zeros((audio.SR, 2), np.float32))
    result = timeline(path, **options, **bounds)
    assert result['cached'] is False
    assert result['duration'] == duration
    assert result['loudness']['digital_silence'] is True
