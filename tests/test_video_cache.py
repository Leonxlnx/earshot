"""Video cut cache bypass, using a tiny synthetic film and no models."""
import subprocess

import numpy as np

from earshot import config, video


def test_cut_cache_can_be_bypassed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    path = str(tmp_path / 'film.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=96x54:r=30:d=1.5',
                    '-f', 'lavfi', '-i', 'color=c=white:s=96x54:r=30:d=1.5', '-filter_complex',
                    '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]', '-c:v', 'mpeg4', path],
                   check=True, capture_output=True)
    assert video.cuts(path) == [1.5]

    cached = next((tmp_path / 'emb' / 'video').glob('diff_*.npz'))
    with np.load(cached) as saved:
        fps, d = float(saved['fps']), np.zeros_like(saved['d'])
    np.savez(cached, fps=fps, d=d)
    assert video.cuts(path) == []
    assert video.cuts(path, cache=False) == [1.5]
