"""Targeted extra measurements for the study doc: tonal sweep ridges in a band (zips, the riser's tonal layer, the
sting's chirp) and pulse rates of gated / stuttered events (peaks of the 2 ms high-band envelope).

    python eval/forge/pro/extra.py
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from videos import wav  # noqa: E402

SR = 48000


def mono(ref, a, b):
    x, _ = sf.read(wav(ref), dtype='float64', always_2d=True, start=int(a * SR), stop=int(b * SR))
    return x.mean(1)


def ridge(ref, a, b, fmin, fmax, hop_ms=5, n=1024, rel_db=18):
    m = mono(ref, a, b)
    f, t, Z = signal.stft(m, SR, nperseg=n, noverlap=n - int(SR * hop_ms / 1000), nfft=8192)
    P = np.abs(Z) ** 2
    k = (f >= fmin) & (f <= fmax)
    pk = f[k][np.argmax(P[k], 0)]
    lv = 10 * np.log10(P[k].max(0) + 1e-14)
    ok = lv >= lv.max() - rel_db
    tt = a + t[ok]
    ff = pk[ok]
    return tt, ff


def pulses(ref, a, b, lo=2000, hi=16000, min_gap_ms=15):
    m = mono(ref, a, b)
    sos = signal.butter(4, [lo, hi], 'bandpass', fs=SR, output='sos')
    y = signal.sosfiltfilt(sos, m)
    n = int(0.002 * SR)
    k = len(y) // n
    e = 10 * np.log10((y[:k * n].reshape(k, n) ** 2).mean(1) + 1e-14)
    pk, _ = signal.find_peaks(e, prominence=6, distance=int(min_gap_ms / 2))
    ts = a + pk * 0.002
    return ts


def show_ridge(name, ref, a, b, fmin, fmax):
    tt, ff = ridge(ref, a, b, fmin, fmax)
    if len(tt) < 2:
        print(name, 'no ridge')
        return
    q = max(1, len(ff) // 6)
    f0, f1 = np.median(ff[:q]), np.median(ff[-q:])
    span = tt[-1] - tt[0]
    oc = np.log2(f1 / f0)
    print(f'{name:28s} {ref} {a:.2f}-{b:.2f}: ridge {f0:.0f} Hz -> {f1:.0f} Hz ({oc:+.2f} oct over {span * 1000:.0f} ms, '
          f'{oc / max(span, 1e-3):+.1f} oct/s), min {ff.min():.0f} max {ff.max():.0f}; '
          f'samples {[(round(t, 3), int(f)) for t, f in list(zip(tt, ff))[::max(1, len(tt) // 8)]]}')


def show_pulses(name, ref, a, b, **kw):
    ts = pulses(ref, a, b, **kw)
    if len(ts) < 3:
        print(name, 'few pulses', ts)
        return
    d = np.diff(ts) * 1000
    print(f'{name:28s} {ref} {a:.2f}-{b:.2f}: {len(ts)} pulses, interval median {np.median(d):.0f} ms '
          f'(rate {1000 / np.median(d):.1f}/s), first half {np.median(d[:len(d) // 2]):.0f} ms, second half '
          f'{np.median(d[len(d) // 2:]):.0f} ms')


if __name__ == '__main__':
    show_ridge('zip up (ref_6 a)', 'ref_6', 1.60, 1.86, 250, 4000)
    show_ridge('zip up (ref_6 b)', 'ref_6', 3.78, 4.06, 250, 14000)
    show_ridge('zip down (ref_5 a)', 'ref_5', 22.80, 22.98, 250, 12000)
    show_ridge('zip arc (ref_5 b)', 'ref_5', 28.22, 28.48, 250, 6000)
    show_ridge('riser tonal layer (ref_2)', 'ref_2', 16.85, 17.55, 1500, 12000)
    show_ridge('pop chirp (ref_2)', 'ref_2', 13.10, 13.22, 200, 14000)
    show_ridge('sting chirp (ref_2)', 'ref_2', 28.60, 28.75, 80, 6000)
    show_ridge('sting hum (ref_2)', 'ref_2', 28.75, 29.40, 60, 600)
    show_ridge('sub drop (ref_8)', 'ref_8', 85.62, 86.40, 25, 120)
    show_ridge('notify fundamental (ref_2)', 'ref_2', 15.28, 15.45, 600, 9000)
    show_pulses('riser stutter (ref_2)', 'ref_2', 16.78, 17.55, lo=1000)
    show_pulses('glitch stutter (ref_2)', 'ref_2', 28.10, 28.60)
    show_pulses('typing run (ref_2)', 'ref_2', 10.40, 11.95)
    show_pulses('pop_in run (ref_2)', 'ref_2', 20.05, 23.60, lo=40, hi=400, min_gap_ms=60)
    show_pulses('impact rumble grains (ref_2)', 'ref_2', 4.80, 5.90, lo=40, hi=600, min_gap_ms=20)
    show_pulses('shimmer grains (ref_2)', 'ref_2', 6.15, 8.10, lo=1500, hi=16000, min_gap_ms=10)
