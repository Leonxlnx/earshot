"""The last stage of every made sound: a clean start and end, no rumble, a level that fits where it plays, and the
file formats products ship.

    from earshot.synth import master
    y = master.finish(y)                                  # edges, low cut, tail, -1 dBFS true peak
    y = master.finish(y, lufs=-18.0)                      # at a loudness instead (momentary max), peaks limited
    master.write('tap.ogg', y)                            # wav / flac / ogg / mp3 / m4a / webm by extension

Short sounds are levelled by their momentary-max loudness (the loudest 400 ms, K-weighted): integrated loudness is
unreliable under 3 s. Peaks are true peaks (4x oversampled), so a file does not clip after lossy encoding.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

from ..forge import dsp, fx

SR = dsp.SR
FORMATS = ('wav', 'flac', 'ogg', 'mp3', 'm4a', 'webm')
BITRATE = {'ogg': '96k', 'mp3': '128k', 'm4a': '96k', 'webm': '64k'}
CODEC = {'ogg': 'libvorbis', 'mp3': 'libmp3lame', 'm4a': 'aac', 'webm': 'libopus'}


def stereo(y):
    y = np.asarray(y, np.float64)
    return y if y.ndim == 2 else np.stack([y, y], 1)


def trim(y, floor_db=-66.0, keep=0.012, fade=0.008):
    """cut the silent end (everything after the last sample within floor_db of the peak, plus `keep` s) and fade the
    new end out, so a tail is never cut while it is audible"""
    y = np.array(dsp.trim_tail(np.asarray(y, np.float64), floor_db, keep))
    k = min(len(y), dsp.n_of(fade))
    if k > 1:
        w = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
        y[-k:] *= w if y.ndim == 1 else w[:, None]
    return y


def lead(y, floor_db=-50.0, keep=0.002):
    """drop the silence before a sound (keeping `keep` s), so every file starts when its sound does"""
    y = np.asarray(y, np.float64)
    m = np.abs(y).reshape(len(y), -1).max(1)
    if not m.size or m.max() <= 0:
        return y
    i = int(np.argmax(m >= m.max() * 10 ** (floor_db / 20)))
    return y[max(0, i - dsp.n_of(keep)):]


def true_peak_db(y):
    """the 4x oversampled peak in dBFS"""
    from scipy import signal
    y = np.asarray(y, np.float64)
    up = signal.resample_poly(y, 4, 1, axis=0) if len(y) > 8 else y
    return float(20 * np.log10(max(np.abs(y).max(), np.abs(up).max()) + 1e-12))


def loudness(y):
    """momentary-max loudness in LUFS (the loudest 400 ms)"""
    from ..forge import polish
    return polish.lufs_max(np.asarray(y, np.float64))


def finish(y, peak_db=-1.0, lufs=None, low_cut=25.0, fade_in=0.0004, tail_db=-66.0, loop=False, cap=None):
    """a made sound as it ships: non-finite or silent renders refused, the low end cut under `low_cut` Hz (no DC, no
    rumble that small speakers turn into distortion), leading silence dropped, a `fade_in` s start from zero, the
    silent end trimmed and faded; then the level: the true peak at peak_db, but no louder than `cap` LUFS
    (momentary max: a dense sound at full peak is far louder than a tap at full peak); or with `lufs` the
    momentary-max loudness at that value with the peaks limited to peak_db. A loop keeps its length and both ends
    (no trim, no fades)."""
    y = np.array(y, np.float64)
    if not np.isfinite(y).all():
        raise ValueError('the render has non-finite samples')
    if y.size == 0 or np.abs(y).max() <= 0:
        raise ValueError('the render is silent')
    if low_cut:
        y = dsp.butter(y, 'high', low_cut, order=2)
    if not loop:
        y = np.array(lead(y))
        a = min(len(y), dsp.n_of(fade_in))
        if a > 1:
            w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
            y[:a] *= w if y.ndim == 1 else w[:, None]
        y = trim(y, tail_db)
    if lufs is None:
        y = y * 10 ** ((peak_db - true_peak_db(y)) / 20)
        over = loudness(y) - cap if cap is not None else 0.0
        return y * 10 ** (-over / 20) if over > 0 else y
    y = y * 10 ** ((lufs - loudness(y)) / 20)
    return fx.limit(y, ceiling_db=peak_db) if true_peak_db(y) > peak_db else y


def write(path, y, sr=SR, mono=False, bits=24, bitrate=None):
    """write y to `path` in the format its extension names (FORMATS). wav and flac are written directly (24-bit by
    default; 16 for games and the web); ogg (Vorbis), mp3, m4a (AAC) and webm (Opus) go through ffmpeg, which must
    be on PATH for them. mono=True sums the channels and restores the peak. Returns the path."""
    import soundfile as sf
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext not in FORMATS:
        raise ValueError(f'unknown audio format .{ext}: one of {", ".join(FORMATS)}')
    y = np.asarray(y, np.float64)
    if mono and y.ndim == 2:
        pk = np.abs(y).max()
        y = y.mean(1)
        y = y * (pk / (np.abs(y).max() + 1e-12))
    if sr != SR:
        import soxr
        y = soxr.resample(y, SR, sr)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    y32 = np.clip(y, -1.0, 1.0).astype(np.float32)
    if ext in ('wav', 'flac'):
        sf.write(path, y32, sr, subtype={16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}[min(bits, 24) if ext == 'flac' else bits])
        return path
    ff = shutil.which('ffmpeg')
    if not ff:
        raise RuntimeError(f'writing .{ext} needs ffmpeg on PATH (wav and flac do not)')
    fd, tmp = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        sf.write(tmp, y32, sr, subtype='PCM_24')
        cmd = [ff, '-v', 'error', '-y', '-i', tmp, '-c:a', CODEC[ext], '-b:a', bitrate or BITRATE[ext]]
        if ext == 'webm' and sr not in (8000, 12000, 16000, 24000, 48000):
            cmd += ['-ar', '48000']                        # Opus has no 44.1 kHz
        p = subprocess.run(cmd + [path], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f'ffmpeg could not write {path}: {p.stderr.strip()[:300]}')
    finally:
        os.remove(tmp)
    return path
