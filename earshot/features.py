"""Objective measurements of a sound (no models): what a meter, a spectrum analyser and an oscilloscope would tell you.

    from earshot.features import measure
    m = measure('hit.wav')            # or an array (n,) / (n, 2) at 48 kHz
    m['lufs'], m['attack_ms'], m['centroid_hz'], m['noise_floor_dbfs'], m['digital_silence_frac']

Fields (all optional: short sounds skip what needs length):
  duration, peak_dbfs, true_peak_dbtp (4x oversampled), rms_dbfs, lufs (BS.1770 integrated), lra_lu, short_term_max_lufs,
  clipped_samples, dc, noise_floor_dbfs (10th pct of 20 ms RMS), dynamic_range_db, digital_silence_frac (samples at
  exactly zero, |x| < 1e-6: code-generated audio often has long stretches of it, but so do trimmed, gated or
  noise-reduced real recordings (59% of one licensed real-recording library) - weak evidence on its own),
  side_to_mid_db + correlation (stereo width; None for mono-identical channels), attack_ms (10% -> 90% rise of the
  first event on a 5 ms RMS envelope: ~4 ms floor for an instant onset), decay_ms (overall peak -> 30 dB down on a
  10 ms RMS envelope, staying down for 20 ms; see envelope_times), brightness_hz (median over active frames of the magnitude-weighted spectral
  centroid: tracks perceived brightness), centroid_hz (power-weighted over the whole spectrum: dominated by the low end,
  kept for compatibility), rolloff95_hz, bandwidth_hz (highest 1/3-oct band within 60 dB of the loudest), bands_db
  (energy share per band: sub 20-60, low 60-250, lowmid 250-1k, mid 1-4k, presence 4-8k, air 8-16k, top 16k+), flatness
  (0 = pure tone .. 1 = white noise), tonal_peak_ratio_db (strongest spectral peak vs median: sine-like when huge),
  harmonic_ratio (HPSS harmonic share, first 30 s), f0_hz (median pitch, only when harmonic_ratio >= 0.8), onsets
  (times) + onsets_per_s, and with music=True: tempo, beats, key.
"""
import math
import os

import numpy as np
from scipy import ndimage, signal

from . import audio

SR = audio.SR
BANDS = {'sub': (20, 60), 'low': (60, 250), 'lowmid': (250, 1000), 'mid': (1000, 4000), 'presence': (4000, 8000),
         'air': (8000, 16000), 'top': (16000, 24000)}
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def db(v):
    return audio.db(v)


def _stereo(x):
    x = np.asarray(x, np.float32)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def frame_rms(m, win):
    n = len(m) // win
    if n == 0:
        return np.array([math.sqrt(float(np.mean(m.astype(np.float64) ** 2)) + 1e-20)])
    f = m[: n * win].reshape(n, win).astype(np.float64)
    return np.sqrt(np.mean(f ** 2, axis=1) + 1e-20)


def smooth_env(m, win_ms):
    """RMS envelope with one value per millisecond, each over a centred `win_ms` window (truncated at the file edges).
    1 ms frames are useless for envelopes: they dip at every zero crossing below ~500 Hz and flicker on noise"""
    m = np.asarray(m, np.float64)
    n, h, w2 = len(m), SR // 1000, max(1, int(win_ms * SR / 2000))
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    mid = np.arange(0, n, h) + h // 2
    a, b = np.clip(mid - w2, 0, n), np.clip(mid + w2, 0, n)
    return np.sqrt((c[b] - c[a]) / np.maximum(b - a, 1) + 1e-20)


def envelope_times(m):
    """attack_at / peak_at (s), attack_ms and decay_ms of a mono 48 kHz array.
    attack: the first event (first ms within 20 dB of the loudest), its 10% -> 90% rise to its own maximum within 300 ms,
    on a 5 ms RMS envelope (an instant onset reads ~4 ms, a sound that starts at full level ~0 ms);
    decay: from the overall peak until the 10 ms RMS envelope is 30 dB down and stays there for 20 ms (None when it
    never gets there: a sustained sound, or one cut off / buried in its own noise floor)"""
    e5, e10 = smooth_env(m, 5), smooth_env(m, 10)
    i0 = int(np.argmax(e5 >= e5.max() * 0.1))
    loc = e5[i0: i0 + 300]
    ip = int(np.argmax(e10))
    after = e10[ip:]
    ahead = ndimage.maximum_filter1d(after, size=21, origin=-10, mode='nearest')  # max over [i, i + 20 ms]
    below = np.nonzero(ahead < e10[ip] * 10 ** (-30 / 20))[0]
    return {'attack_at': round(i0 / 1000, 3), 'peak_at': round(ip / 1000, 3),
            'attack_ms': float(int(np.argmax(loc >= 0.9 * loc.max()))) if len(loc) else None,
            'decay_ms': float(below[0]) if len(below) else None}


def true_peak_dbtp(x):
    y = signal.resample_poly(_stereo(x), 4, 1, axis=0)
    return db(np.abs(y).max())


def _kw(x):
    """BS.1770 K-weighting at 48 kHz"""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)


def loudness_curve(x, win=0.4, hop=0.1):
    """(t_mid, LUFS) of sliding blocks: win 0.4 = momentary, 3.0 = short-term"""
    y = _kw(_stereo(x).astype(np.float64))
    p = np.cumsum(np.concatenate([np.zeros((1, 2)), y ** 2]), axis=0)
    w, h = int(win * SR), int(hop * SR)
    if len(y) < w:
        ms = (p[-1] - p[0]) / max(1, len(y))
        return np.array([len(y) / SR / 2]), np.array([-0.691 + 10 * np.log10(ms.sum() + 1e-20)])
    st = np.arange(0, len(y) - w + 1, h)
    ms = (p[st + w] - p[st]) / w
    return (st + w / 2) / SR, -0.691 + 10 * np.log10(ms.sum(axis=1) + 1e-20)


def integrated_lufs(x):
    t, L = loudness_curve(x, 0.4, 0.1)
    L = L[L > -70]
    if not len(L):
        return -70.0
    rel = 10 * np.log10(np.mean(10 ** (L / 10))) - 10
    L2 = L[L > rel]
    return float(10 * np.log10(np.mean(10 ** (L2 / 10)))) if len(L2) else float(rel + 10)


def lra(x):
    t, S = loudness_curve(x, 3.0, 0.1)
    S = S[S > -70]
    if len(S) < 5:
        return None
    rel = 10 * np.log10(np.mean(10 ** (S / 10))) - 20
    S = S[S > rel]
    return float(np.percentile(S, 95) - np.percentile(S, 10)) if len(S) > 4 else None


def onsets(m, sr=SR, delta=0.07):
    import librosa
    y = audio.resample(m, sr, 22050)
    o = librosa.onset.onset_detect(y=y, sr=22050, hop_length=256, units='time', backtrack=True, delta=delta)
    return [round(float(v), 3) for v in o]


def key_estimate(m, sr=SR):
    import librosa
    y = audio.resample(m, sr, 22050)
    c = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=512).mean(axis=1)
    best = []
    for mode, prof in (('major', KK_MAJ), ('minor', KK_MIN)):
        for k in range(12):
            best.append((float(np.corrcoef(c, np.roll(prof, k))[0, 1]), f'{NOTE[k]} {mode}'))
    best.sort(reverse=True)
    return [{'key': b[1], 'r': round(b[0], 3)} for b in best[:3]]


def brightness_hz(m, sr=SR):
    """median magnitude-weighted spectral centroid over frames within 30 dB of the loudest (perceived brightness)"""
    m = np.asarray(m, np.float32)
    if m.ndim == 2:
        m = m.mean(axis=1)
    if len(m) < 1024:
        return None
    f, t, Z = signal.stft(m, sr, nperseg=1024, noverlap=768, boundary=None)
    A = np.abs(Z)
    e = A.sum(axis=0)
    act = e >= e.max() * 10 ** (-30 / 20)
    if not act.any():
        return None
    c = (f[:, None] * A[:, act]).sum(axis=0) / (A[:, act].sum(axis=0) + 1e-20)
    return int(np.median(c))


def measure(src, music=False, start=None, end=None, light=False):
    """objective measurements of a file path or a 48 kHz array; light=True skips the slow parts (onsets, HPSS, f0);
    a negative, reversed or out-of-file range raises ValueError"""
    if start is not None and start < 0:
        raise ValueError(f'start must be >= 0 s (got {start:g})')
    if end is not None and end <= (start or 0.0):
        raise ValueError(f'end ({end:g} s) must be greater than start ({start or 0.0:g} s)')
    if isinstance(src, str):
        if start:
            total = audio.probe(src).get('duration') or 0
            if total and start >= total:
                raise ValueError(f'start {start:g} s is at or beyond the end of {os.path.basename(src)} ({total:g} s long)')
        x = audio.load(src, start=start, end=end)
    else:
        x = _stereo(src)
    m = x.mean(axis=1)
    n = len(m)
    r = {'duration': round(n / SR, 3)}
    if n < int(0.005 * SR):
        r['error'] = 'too short to measure'
        return r
    pk = float(np.abs(x).max())
    r['peak_dbfs'] = round(db(pk), 1)
    r['true_peak_dbtp'] = round(true_peak_dbtp(x), 1)
    r['rms_dbfs'] = round(db(math.sqrt(float(np.mean(x.astype(np.float64) ** 2)))), 1)
    if n >= int(0.4 * SR):
        r['lufs'] = round(integrated_lufs(x), 1)
        t, S = loudness_curve(x, 3.0, 0.5) if n >= 3 * SR else (None, None)
        if S is not None:
            r['short_term_max_lufs'] = round(float(S.max()), 1)
        v = lra(x) if n >= 6 * SR else None
        if v is not None:
            r['lra_lu'] = round(v, 1)
    r['clipped_samples'] = int(np.sum(np.abs(x) >= 0.999))
    r['dc'] = round(float(np.mean(m)), 5)
    r20 = frame_rms(m, int(0.02 * SR))
    r['noise_floor_dbfs'] = round(db(np.percentile(r20, 10)), 1)
    r['dynamic_range_db'] = round(db(r20.max()) - r['noise_floor_dbfs'], 1)
    r['digital_silence_frac'] = round(float(np.mean(np.abs(m) < 1e-6)), 4)
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    sm = float(np.sqrt(np.mean(side ** 2)))
    r['side_to_mid_db'] = round(db(sm) - db(np.sqrt(np.mean(mid ** 2)) + 1e-12), 1) if sm > 1e-7 else None
    if np.std(x[:, 0]) > 1e-9 and np.std(x[:, 1]) > 1e-9:
        r['correlation'] = round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 3)
    r.update(envelope_times(m))
    # spectrum (whole file)
    seg = m.astype(np.float64)
    nfft = 1 << int(min(16, max(10, math.ceil(math.log2(max(1024, min(len(seg), 65536)))))))
    f, P = signal.welch(seg, SR, nperseg=min(nfft, len(seg)), noverlap=None)
    P = P + 1e-30
    tot = float(P.sum())
    r['centroid_hz'] = int(float((f * P).sum() / tot))
    c = np.cumsum(P) / tot
    r['rolloff95_hz'] = int(f[min(len(f) - 1, int(np.searchsorted(c, 0.95)))])
    r['bands_db'] = {k: round(10 * math.log10(float(P[(f >= a) & (f < b)].sum()) / tot + 1e-12), 1) for k, (a, b) in BANDS.items()}
    # bandwidth: highest third-octave band within 60 dB of the loudest band
    edges = 20 * 2 ** (np.arange(0, 31) / 3)
    lv = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = P[(f >= a) & (f < b)]
        lv.append(10 * math.log10(float(s.mean())) if len(s) else -300)
    lv = np.array(lv)
    ok = np.where(lv >= lv.max() - 60)[0]
    r['bandwidth_hz'] = int(edges[ok[-1] + 1]) if len(ok) else None
    r['flatness'] = round(float(np.exp(np.mean(np.log(P))) / np.mean(P)), 4)
    r['brightness_hz'] = brightness_hz(m)
    r['tonal_peak_ratio_db'] = round(10 * math.log10(float(P.max() / np.median(P))), 1)
    if light:
        return r
    import librosa
    y22 = audio.resample(m, SR, 22050)
    yt = y22[: 30 * 22050]  # tonality: the first 30 s are plenty and keep long files fast
    if len(yt) >= 2048:
        H, Pp = librosa.effects.hpss(yt)
        eh, ep = float(np.sum(H ** 2)), float(np.sum(Pp ** 2))
        r['harmonic_ratio'] = round(eh / (eh + ep + 1e-20), 3)
        try:
            f0 = librosa.yin(yt, fmin=50, fmax=2000, sr=22050, frame_length=2048)
            rms = librosa.feature.rms(y=yt, frame_length=2048, hop_length=512)[0][: len(f0)]
            voiced = f0[: len(rms)][rms > rms.max() * 0.1]
            if len(voiced) and r.get('harmonic_ratio', 0) >= 0.8:
                r['f0_hz'] = round(float(np.median(voiced)), 1)
        except Exception:
            pass
    on = onsets(m)
    r['onsets'] = on[:200]
    r['onset_count'] = len(on)
    r['onsets_per_s'] = round(len(on) / (n / SR), 2)
    if music and n >= 5 * SR:
        oenv = librosa.onset.onset_strength(y=y22, sr=22050, hop_length=256)
        tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=22050, hop_length=256, units='time')
        r['tempo'] = round(float(np.atleast_1d(tempo)[0]), 1)
        r['beats'] = [round(float(b), 3) for b in beats]
        if len(beats) > 4:
            r['beat_ibi_std_ms'] = round(float(np.std(np.diff(beats))) * 1000, 1)
        r['key'] = key_estimate(m)
    return r


def _text(r, path):
    if r.get('error'):
        return f'{path}: {r["error"]}'
    if r.get('peak_dbfs', 0) < -70:
        what = 'digitally silent' if r.get('digital_silence_frac', 0) >= 0.999 else 'near-silent'
        return (f'{path}: {r["duration"]:.3f} s, {what} (peak {r["peak_dbfs"]:.0f} dBFS): nothing to measure; the loudness, '
                'spectrum and envelope values of silence are meaningless.')
    b = r.get('bands_db') or {}
    parts = [f'{path}: {r["duration"]:.3f} s, peak {r["peak_dbfs"]} dBFS (true peak {r["true_peak_dbtp"]} dBTP)']
    if r.get('lufs') is not None:
        parts.append(f'{r["lufs"]} LUFS' + (f', LRA {r["lra_lu"]} LU' if r.get('lra_lu') is not None else ''))
    env = f'attack {r["attack_ms"]:g} ms' if r.get('attack_ms') is not None else ''
    env += (', ' if env else '') + (f'decay {r["decay_ms"]:g} ms' if r.get('decay_ms') is not None else
                                    'no -30 dB decay (sustained or cut off)')
    parts.append(f'noise floor {r["noise_floor_dbfs"]} dBFS, {env}, '
                 f'brightness {r.get("brightness_hz")} Hz, bandwidth {r.get("bandwidth_hz")} Hz')
    if r.get('clipped_samples'):
        parts.append(f'{r["clipped_samples"]} clipped samples')
    if r.get('tempo'):
        parts.append(f'tempo {r["tempo"]} BPM, key {r["key"][0]["key"] if r.get("key") else "?"}')
    if b:
        parts.append('bands ' + ' '.join(f'{k} {v}' for k, v in b.items()) + ' dB')
    return '; '.join(parts) + '.'


def add_cli(sub):
    from .cli import common
    from .report import emit
    p = common(sub.add_parser('measure', help='objective measurements only (loudness, peaks, envelope, spectrum; no models)'))
    p.add_argument('files', nargs='+')
    p.add_argument('--music', action='store_true', help='also tempo, beats and key')
    p.add_argument('--light', action='store_true', help='skip the slow parts (onsets, HPSS, pitch)')
    p.add_argument('--start', type=float, help='measure only from this time (s)')
    p.add_argument('--end', type=float, help='measure only up to this time (s)')

    def run(a):
        res = []
        audio.check_paths(a.files, 'files')
        files = audio.list_media(a.files)
        if not files:
            raise audio.AudioError('no audio/video files found in: ' + ', '.join(a.files))
        for f in files:
            r = measure(f, music=a.music, start=a.start, end=a.end, light=a.light)
            r['file'] = f
            r['summary'] = _text(r, os.path.basename(f))
            res.append(r)
        out = res[0] if len(res) == 1 else {'summary': f'{len(res)} files measured.', 'files': res}
        emit(out, a, text='\n'.join(r['summary'] for r in res))
        return 0 if res else 1
    p.set_defaults(func=run)
