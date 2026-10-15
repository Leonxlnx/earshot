"""Video helpers (ffmpeg, CPU): picture cuts, frames for contact strips, basic stream info.

    from earshot import video
    video.cuts('film.mp4')                 # [0.0, 1.83, 2.5, ...] hard cuts / big visual changes (seconds)
    video.frames('film.mp4', [0.5, 1.0])   # [PIL.Image, ...]

Cut detection (method='diff', default) reads 96x54 grey frames at the native frame rate and marks frame-difference
peaks that stand >= 4x above the median of the surrounding second (and above 1.2 % grey change), or any change of
>= 15 %, at most one per 0.12 s. Unlike ffmpeg's scene score (method='scene', threshold 0.3), it also catches cuts and
whip transitions on white-background motion graphics, where the scene score finds almost nothing. Slow crossfades are
missed or found late; treat the result as "big visual change" markers, not an edit decision list.
"""
import hashlib
import json
import os
import re
import subprocess

import numpy as np

from . import audio, config

FRAME_W, FRAME_H = 96, 54


def has_video(path):
    return bool(audio.probe(path).get('has_video'))


def info(path):
    p = audio.probe(path)
    v = p.get('video') or {}
    return {'duration': p.get('duration'), 'fps': v.get('fps'), 'width': v.get('width'), 'height': v.get('height'),
            'has_audio': bool(p.get('codec')), 'has_video': p.get('has_video', False)}


def _cache(name):
    config.ensure_dirs()
    d = os.path.join(config.EMB_CACHE, 'video')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def frame_diff(path):
    """(fps, d): mean absolute difference (0..1) between consecutive 96x54 grey frames (d[0] = 0), cached"""
    f = _cache('diff_' + hashlib.sha1(f'{audio.file_id(path)}|{FRAME_W}x{FRAME_H}|v1'.encode()).hexdigest()[:20] + '.npz')
    if os.path.exists(f):
        try:
            with np.load(f, allow_pickle=False) as z:
                return float(z['fps']), np.array(z['d'])
        except Exception:
            pass
    fps = (audio.probe(path).get('video') or {}).get('fps') or 30.0
    p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-an', '-vf',
                        f'scale={FRAME_W}:{FRAME_H}:flags=area,format=gray', '-f', 'rawvideo', '-'], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise audio.AudioError(f'cannot decode the picture of {path}: {p.stderr.decode(errors="replace")[:200]}')
    fr = np.frombuffer(p.stdout, np.uint8)
    fr = fr[: len(fr) // (FRAME_W * FRAME_H) * FRAME_W * FRAME_H].reshape(-1, FRAME_H, FRAME_W).astype(np.float32)
    d = np.concatenate([[0.0], np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2)) / 255]).astype(np.float32)
    tmp = f + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, fps=fps, d=d)
    audio.replace_quietly(tmp, f)
    return float(fps), d


def diff_events(d, fps, ratio=4.0, floor=0.012, big=0.15, gap=0.12):
    """frame indices of cuts / big picture changes in a frame-difference signal (see module doc)"""
    from scipy.ndimage import maximum_filter1d, median_filter
    if len(d) < 3:
        return np.zeros(0, int)
    med = median_filter(d, size=int(round(fps)) | 1, mode='nearest')
    loc = maximum_filter1d(d, size=3, mode='nearest')
    out = []
    for i in np.where((d == loc) & (((d > floor) & (d > ratio * med + 0.004)) | (d >= big)))[0]:
        if i == 0:
            continue
        if out and i - out[-1] < gap * fps:
            if d[i] > d[out[-1]]:
                out[-1] = int(i)
            continue
        out.append(int(i))
    return np.array(out, int)


def cuts(path, method='diff', threshold=0.3, min_gap=0.2, cache=True):
    """times (s) of picture cuts / big visual changes; method 'diff' (default) or 'scene' (ffmpeg scene score)"""
    if method == 'diff':
        fps, d = frame_diff(path)
        return [round(float(i) / fps, 3) for i in diff_events(d, fps)]
    k = hashlib.sha1(f'{audio.file_id(path)}|{threshold}|{min_gap}|v1'.encode()).hexdigest()[:20]
    f = _cache(f'cuts_{k}.json')
    if cache and os.path.exists(f):
        return json.load(open(f))
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-i', path, '-an', '-vf',
                        f"scale=320:-2,select='gt(scene,{threshold})',showinfo", '-f', 'null', '-'],
                       capture_output=True, text=True, errors='replace')
    ts = [float(m) for m in re.findall(r'pts_time:([0-9.]+)', p.stderr)]
    out = []
    for t in ts:
        if not out or t - out[-1] >= min_gap:
            out.append(round(t, 3))
    json.dump(out, open(f, 'w'))
    return out


def frame(path, t, width=320):
    """one frame at time t as a PIL image"""
    from PIL import Image
    import io
    p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', f'{max(0.0, t):.3f}', '-i', path, '-frames:v', '1',
                        '-vf', f'scale={width}:-2', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return Image.open(io.BytesIO(p.stdout)).convert('RGB')


def frames(path, times, width=320):
    return [frame(path, t, width) for t in times]
