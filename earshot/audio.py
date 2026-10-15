"""Audio I/O shared by every earshot module.

Any file ffmpeg can read works (wav/flac/ogg/mp3/m4a and the audio track of mp4/mov/webm). Audio is decoded once to
float32 at 48 kHz and cached under EARSHOT_HOME/decoded (keyed on path, size, mtime and 16 sampled 4 KB blocks of the
content, so a same-size replacement that kept its timestamp is still noticed), so repeated calls are fast.

    x = load('kick.ogg')                 # (n, 2) float32 at 48 kHz
    m = load('kick.ogg', mono=True)      # (n,)
    seg = load('long.wav', start=1.2, end=1.8, mono=True)

Levels: the stereo decode keeps true sample levels (a mono file is duplicated to both channels at unity gain), so
measure peaks and loudness on it (features.measure does). mono=True is ffmpeg's downmix, which sums a stereo source
at -3 dB per channel: correlated stereo reads up to +3 dB hotter than mean(L, R). It is the models' input convention
(their calibration was measured with it), so keep it for model input and timing, not for level readings.
"""
import hashlib
import json
import os
import subprocess

import numpy as np

from . import config

SR = 48000
AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.oga', '.opus', '.mp3', '.m4a', '.aac', '.aif', '.aiff', '.wma'}
VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi'}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS
LOSSY = {'aac', 'mp3', 'mp2', 'opus', 'vorbis', 'wmav2', 'ac3', 'eac3'}


class AudioError(RuntimeError):
    pass


DECODE_VERSION = 2  # bump when decoded samples change (derived caches may key on it)


def _sample_hash(path, size, blocks=16, block=4096):
    """sha1 of `blocks` evenly spaced 4 KB blocks (first and last included): cheap, and catches a same-size file
    swapped in with its old timestamp kept (cp -p, robocopy, unzip)"""
    h = hashlib.sha1()
    try:
        with open(path, 'rb') as fh:
            for off in sorted({int(i * max(0, size - block) / max(1, blocks - 1)) for i in range(blocks)}):
                fh.seek(off)
                h.update(fh.read(block))
    except OSError:
        return 'unreadable'
    return h.hexdigest()[:12]


def _state(path):
    st = os.stat(path)
    return f'{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{_sample_hash(path, st.st_size)}'


def _key(path, sr, ch):
    return hashlib.sha1(f'{_state(path)}|{sr}|{ch}|v{DECODE_VERSION}'.encode()).hexdigest()[:20]


def replace_quietly(tmp, dst, tries=5):
    """os.replace(tmp, dst) for a cache write: retried briefly (Windows: another process may hold dst for a moment),
    then given up (tmp removed) - a cache write must never fail the call"""
    import time
    for attempt in range(tries):
        try:
            os.replace(tmp, dst)
            return True
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def _src_channels(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels',
                              '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out.splitlines()[0]) if out else 0
    except Exception:  # noqa: BLE001
        return 0


def file_id(path):
    """stable id of a file's content state (path + size + mtime + sampled content); used as a cache key by the models"""
    return hashlib.sha1(_state(path).encode()).hexdigest()[:20]


def array_id(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return hashlib.sha1(x.tobytes() + str(x.shape).encode()).hexdigest()[:20]


def _ffmpeg_reason(err):
    """the last meaningful line of ffmpeg's stderr, without its [component @ 0x...] prefix, on one line"""
    import re
    lines = [re.sub(r'^\[[^\]]*@ [0-9a-fA-Fx]+\]\s*', '', ln).strip() for ln in err.splitlines()]
    lines = [ln for ln in lines if ln and not ln.lower().startswith(('error opening output', 'conversion failed'))]
    return (lines[-1] if lines else 'ffmpeg failed')[:200]


def _decode(path, sr, ch):
    up = ch == 2 and _src_channels(path) == 1  # ffmpeg would upmix mono at -3 dB; duplicate at unity instead
    p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-vn', '-f', 'f32le', '-acodec', 'pcm_f32le',
                        '-ac', '1' if up else str(ch), '-ar', str(sr), '-'], capture_output=True)
    err = p.stderr.decode(errors='replace')
    if p.returncode != 0 or not p.stdout:
        pr = probe(path)
        if not pr.get('probe_error') and pr.get('has_video') and not pr.get('codec'):
            raise AudioError(f'{path} has no audio track (a silent render? mux the soundtrack in first)')
        if p.returncode == 0 or 'does not contain any stream' in err:
            raise AudioError(f'{path}: no audio stream')
        raise AudioError(f'cannot decode {path}: {_ffmpeg_reason(err)}')
    x = np.frombuffer(p.stdout, np.float32).copy()
    if up:
        return np.repeat(x[:, None], 2, axis=1)
    return x.reshape(-1, ch) if ch > 1 else x


def load(path, sr=SR, mono=False, start=None, end=None, cache=True):
    """decode `path` to float32 (n, 2) or (n,) when mono; optional [start, end) in seconds"""
    if not os.path.exists(path):
        raise AudioError(f'no such file: {path}')
    ch = 1 if mono else 2
    x = None
    npy = None
    if cache:
        config.ensure_dirs()
        npy = os.path.join(config.DEC_CACHE, _key(path, sr, ch) + '.npy')
        if os.path.exists(npy):
            try:
                x = np.load(npy)
            except Exception:
                x = None
    if x is None:
        x = _decode(path, sr, ch)
        if npy:
            tmp = f'{npy}.{os.getpid()}.tmp.npy'
            np.save(tmp, x)
            replace_quietly(tmp, npy)
    if start is not None or end is not None:
        a = max(0, int(round((start or 0.0) * sr)))
        b = len(x) if end is None else min(len(x), int(round(end * sr)))
        x = x[a:b]
    return x


def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    import soxr
    return soxr.resample(x, sr_from, sr_to, quality='HQ').astype(np.float32)


def duration(path):
    return len(load(path, mono=True)) / SR


def probe(path):
    """ffprobe summary: codec, sample rate, channels, kbps, duration, has_video"""
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                              'stream=codec_type,codec_name,sample_rate,channels,bit_rate,width,height,r_frame_rate:'
                              'stream_disposition=attached_pic:'
                              'format=duration,bit_rate', '-of', 'json', path],
                             capture_output=True, text=True, timeout=60).stdout
        j = json.loads(out or '{}')
    except Exception as e:  # noqa: BLE001
        return {'probe_error': str(e)[:200]}
    streams = j.get('streams') or []
    a = next((s for s in streams if s.get('codec_type') == 'audio'), {})
    v = next((s for s in streams if s.get('codec_type') == 'video'
              and not (s.get('disposition') or {}).get('attached_pic')), None)  # cover art is not a picture track
    fmt = j.get('format') or {}
    br = a.get('bit_rate')
    if not (br and str(br).isdigit() and int(br) < 10_000_000):  # Vorbis can report 2^32-1: use the container's rate
        br = fmt.get('bit_rate') if not v else None
    res = {'codec': a.get('codec_name'), 'src_rate': int(a.get('sample_rate') or 0), 'channels': int(a.get('channels') or 0),
           'kbps': round(int(br) / 1000) if br and str(br).isdigit() else None, 'lossy': a.get('codec_name') in LOSSY,
           'duration': round(float(fmt.get('duration') or 0), 3), 'has_video': v is not None}
    if v is not None:
        num, _, den = (v.get('r_frame_rate') or '0/1').partition('/')
        res['video'] = {'width': v.get('width'), 'height': v.get('height'),
                        'fps': round(float(num) / float(den or 1), 3) if float(den or 1) else None}
    return res


def windows(x, win, hop, sr=SR, pad_last=False):
    """[(t0_seconds, segment)] over the first axis"""
    n, w, h = len(x), int(round(win * sr)), int(round(hop * sr))
    out = []
    if n <= w:
        return [(0.0, x)]
    for s in range(0, n - w + 1, h):
        out.append((s / sr, x[s:s + w]))
    if pad_last and (n - w) % h:
        out.append(((n - w) / sr, x[n - w:]))
    return out


def list_media(paths, recursive=True, exts=MEDIA_EXTS):
    """expand files / folders / glob results into a sorted list of media files"""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs.sort()
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in exts:
                        out.append(os.path.join(root, f))
                if not recursive:
                    break
        elif os.path.isfile(p):
            out.append(p)
        else:
            import glob
            out += sorted(g for g in glob.glob(p, recursive=True) if os.path.splitext(g)[1].lower() in exts)
    seen, res = set(), []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            res.append(p)
    return res


def check_paths(paths, what='paths'):
    """raise AudioError naming every given path that yields no media file (a typo must not vanish from a batch)"""
    paths = [paths] if isinstance(paths, str) else list(paths or [])
    bad = [f"{p} ({'no such file or folder' if not os.path.exists(p) else 'no audio or video files in it'})"
           for p in paths if not list_media([p])]
    if bad:
        raise AudioError(f'{what}: nothing playable in ' + ', '.join(bad[:5]) + (' ...' if len(bad) > 5 else ''))


def db(v, floor=1e-12):
    return float(20 * np.log10(max(float(v), floor)))
