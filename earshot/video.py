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
from fractions import Fraction
import json
import os
import re
import subprocess
import tempfile
import threading

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


def frame_diff(path, cache=True):
    """(decoded fps, d): consecutive 96x54 grey-frame differences (d[0] = 0), cached.

    The rate belongs to FFmpeg's selected decode, not necessarily probe()'s first stream.
    """
    f = _cache('diff_' + hashlib.sha1(f'{audio.file_id(path)}|{FRAME_W}x{FRAME_H}|v2'.encode()).hexdigest()[:20] + '.npz')
    if cache and os.path.exists(f):
        try:
            with np.load(f, allow_pickle=False) as z:
                return float(z['fps']), np.array(z['d'])
        except Exception:
            pass
    raw, log, returncode = _decode_picture(path)
    if returncode != 0 or not raw:
        reason = audio._ffmpeg_reason(log.tail.decode(errors='replace'))
        raise audio.AudioError(f'cannot decode the picture of {path}: {reason}')
    fps = log.rate(path)
    d = _frame_differences(raw)
    tmp = f + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, fps=fps, d=d)
    audio.replace_quietly(tmp, f)
    return float(fps), d


class _PictureLog:
    """Drain verbose FFmpeg output without retaining per-frame logs or long metadata."""
    LIMIT = 4096

    def __init__(self):
        self.tail = self.pending = b''
        self.dropping = self.invalid = False
        self.fps = None

    @staticmethod
    def _is_config(line):
        return re.match(rb'^\[showinfo@earshot_rate @ [^\]]+\] config in ', line) is not None

    def _line(self, line):
        if not self._is_config(line):
            return
        try:
            fps = _decoded_rate(line, 'FFmpeg output')
            if self.fps is not None and self.fps != fps:
                self.invalid = True
            self.fps = fps
        except audio.AudioError:
            self.invalid = True

    def feed(self, chunk):
        self.tail = (self.tail + chunk)[-self.LIMIT:]
        if self.dropping:
            end = chunk.find(b'\n')
            if end < 0:
                return
            chunk = chunk[end + 1:]
            self.dropping = False
        data = self.pending + chunk
        lines = data.split(b'\n')
        for line in lines[:-1]:
            if len(line) <= self.LIMIT:
                self._line(line)
            elif self._is_config(line):
                self.invalid = True
        self.pending = lines[-1]
        if len(self.pending) > self.LIMIT:
            if self._is_config(self.pending):
                self.invalid = True
            self.pending = b''
            self.dropping = True

    def finish(self):
        if self.pending:
            self._line(self.pending)
        self.pending = b''

    def rate(self, path):
        if self.invalid or self.fps is None:
            raise audio.AudioError(f'cannot determine decoded picture frame rate of {path}')
        return float(self.fps)


def _decode_picture(path):
    """The original raw-frame decode, with a bounded concurrent stderr consumer."""
    args = ['ffmpeg', '-hide_banner', '-v', 'info', '-nostats', '-nostdin', '-i', path, '-an', '-vf',
            f'scale={FRAME_W}:{FRAME_H}:flags=area,format=gray,showinfo@earshot_rate=checksum=0',
            '-f', 'rawvideo', '-']
    log = _PictureLog()
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        def drain():
            try:
                while True:
                    chunk = proc.stderr.read(8192)
                    if not chunk:
                        break
                    log.feed(chunk)
                log.finish()
            except Exception:
                log.invalid = True
                proc.stderr.close()

        reader = threading.Thread(target=drain)
        reader.start()
        try:
            raw = proc.stdout.read()
            returncode = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            reader.join()
    return raw, log, returncode


def _decoded_fps(stderr, path):
    return float(_decoded_rate(stderr, path))


def _decoded_rate(stderr, path):
    """Read the rate of the actual selected decode, not ffprobe's first picture stream.

    showinfo observes the existing filter graph without changing its frames or
    FFmpeg's automatic stream selection / output synchronization.
    """
    entries = re.findall(rb'^\[showinfo@earshot_rate @ [^\]]+\] config in (.*)$',
                         stderr, re.MULTILINE)
    try:
        rates = set()
        for entry in entries:
            match = re.fullmatch(rb'time_base: (\d+)/(\d+), frame_rate: (\d+)/(\d+)', entry.strip())
            if match is None:
                raise ValueError('malformed rate configuration')
            tn, td, n, d = map(int, match.groups())
            if Fraction(tn, td) <= 0 or Fraction(n, d) <= 0:
                raise ValueError('nonpositive timing')
            rates.add(Fraction(n, d))
        if len(rates) != 1:
            raise ValueError('missing or changing rate')
        rate = rates.pop()
        if not np.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError('invalid rate')
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise audio.AudioError(f'cannot determine decoded picture frame rate of {path}') from exc
    return rate


def _frame_differences(raw):
    """Mean differences of complete raw grey frames, with bounded float32 intermediates."""
    fr = np.frombuffer(raw, np.uint8)
    fr = fr[: len(fr) // (FRAME_W * FRAME_H) * FRAME_W * FRAME_H].reshape(-1, FRAME_H, FRAME_W)
    # Retain the original single zero even when a nonempty decode has no complete frame.
    d = np.zeros(max(1, len(fr)), np.float32)
    for start in range(1, len(fr), 256):
        stop = min(start + 256, len(fr))
        block = fr[start - 1:stop].astype(np.float32)
        d[start:stop] = np.abs(np.diff(block, axis=0)).mean(axis=(1, 2)) / 255
    return d


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
    """times (s) of picture cuts / big visual changes; method 'diff' (default) or 'scene' (ffmpeg scene score).
    cache=False recomputes rather than reusing a saved decode or scene scan."""
    if method == 'diff':
        fps, d = frame_diff(path, cache=cache)
        return [round(float(i) / fps, 3) for i in diff_events(d, fps)]
    k = hashlib.sha1(f'{audio.file_id(path)}|{threshold}|{min_gap}|v2'.encode()).hexdigest()[:20]
    f = _cache(f'cuts_{k}.json')
    if cache and os.path.exists(f):
        try:
            with open(f) as saved:
                return json.load(saved)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass  # a missing, unreadable or interrupted cache is not a failed scan
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-i', path, '-an', '-vf',
                        f"scale=320:-2,select='gt(scene,{threshold})',showinfo", '-f', 'null', '-'],
                       capture_output=True, text=True, errors='replace')
    if p.returncode != 0:
        raise audio.AudioError(f'cannot decode the picture of {path}: {audio._ffmpeg_reason(p.stderr)}')
    ts = [float(m) for m in re.findall(r'pts_time:([0-9.]+)', p.stderr)]
    out = []
    for t in ts:
        if not out or t - out[-1] >= min_gap:
            out.append(round(t, 3))
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(f),
                                         prefix=os.path.basename(f) + '.', suffix='.tmp', delete=False) as saved:
            tmp = saved.name
            json.dump(out, saved)
        audio.replace_quietly(tmp, f)
    except OSError:
        pass  # caching is optional; the scan itself succeeded
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
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
