#!/usr/bin/env python3
"""
master.py: cinetic's mix bus and mastering chain, plus loudness measurement.

Chain (finish): 28 Hz high-pass -> mono below 120 Hz -> 2:1 bus compressor above -14 dBFS
(5 ms RMS detector, 15 ms attack, 120 ms release) -> soft saturation + air -> loudness passes to
the target LUFS, each followed by a true-peak lookahead limiter (4x oversampled detection, 4 ms
lookahead, 80 ms release, ceiling 0.77 linear = -2.3 dBFS, which leaves room for AAC overshoot)
-> edge fades, always: a 3 ms raised-cosine fade-in from exact zero at sample 0 (a sound already at
level on sample 0 clicks and cuts in mid-action) and a raised-cosine fade-out of at least 10 ms
(60 ms by default) onto the last sample -> last 480 samples zeroed.

score.py imports sidechain_times(), finish() and measure(). Standalone use:

    python3 master.py mix.wav public/audio/soundtrack.wav --lufs -14 --ceiling 0.77
    python3 master.py music.wav ducked.wav --sidechain kick.wav --sc-depth 0.55 --lufs -14
    python3 master.py --measure public/audio/soundtrack.wav --json out/qa/loudness.json

Exit code 0 when the output meets the gate (|LUFS - target| <= --tol and true peak <= --max-tp),
1 when it does not or on error. The JSON report holds integrated LUFS, true peak (dBTP, 4x
oversampled), LRA, the loudest momentary block and its time, limiter gain reduction and tail level.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

SR = 48000
HEAD_FADE = 0.003     # s, raised-cosine fade-in at sample 0, applied after the limiter on every master
END_FADE_MIN = 0.010  # s, the shortest fade-out onto the last sample, whatever end_fade asks for


# ------------------------------------------------------------------------------------------
# measurement
# ------------------------------------------------------------------------------------------
def _stereo(x):
    x = np.asarray(x, dtype=np.float64)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def _kw(x):
    """BS.1770 K-weighting (48 kHz coefficients): high shelf then RLB high-pass."""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)


def blocks(x, win=0.4, hop=0.1):
    """Loudness (LUFS) of sliding blocks; 0.4 s = momentary, 3 s = short-term. Returns (t_mid, L)."""
    y = _kw(_stereo(x))
    p = np.cumsum(np.concatenate([np.zeros((1, 2)), y ** 2]), axis=0)
    w, h = int(win * SR), int(hop * SR)
    if len(y) < w:
        return np.array([len(y) / SR / 2]), np.array([-70.0])
    starts = np.arange(0, len(y) - w + 1, h)
    ms = (p[starts + w] - p[starts]) / w
    L = -0.691 + 10 * np.log10(np.maximum(ms.sum(axis=1), 1e-12))
    return (starts + w / 2) / SR, L


def integrated(x):
    import pyloudnorm as pyln
    return float(pyln.Meter(SR).integrated_loudness(_stereo(x)))


def lra(x):
    """EBU loudness range: 10th-95th percentile of gated 3 s short-term loudness."""
    _, st = blocks(x, 3.0, 0.1)
    st = st[st > -70]
    if len(st) < 2:
        return 0.0
    rel = 10 * np.log10(np.mean(10 ** (st / 10))) - 20
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) > 1 else 0.0


def true_peak_db(x):
    up = signal.resample_poly(_stereo(x), 4, 1, axis=0)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-12))


def measure(x):
    x = _stereo(x)
    tm, m = blocks(x, 0.4, 0.1)
    k = int(np.argmax(m))
    tail = x[-int(0.05 * SR):]
    return {
        'lufs': round(integrated(x), 2),
        'true_peak_db': round(true_peak_db(x), 2),
        'sample_peak_db': round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 2),
        'lra': round(lra(x), 2),
        'momentary_max': round(float(m[k]), 2),
        'momentary_max_t': round(float(tm[k]), 3),
        'tail_50ms_db': round(float(20 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-12)), 1),
        'last_480_zero': bool(np.all(x[-480:] == 0)),
        'seconds': round(len(x) / SR, 4),
    }


# ------------------------------------------------------------------------------------------
# dynamics
# ------------------------------------------------------------------------------------------
def sidechain_times(hits, n, depth=0.55, tau=0.14, attack=0.003, length=0.6):
    """Duck gain (n, 1) from (t_seconds, weight) pairs: each hit pulls the music down by
    depth*weight with a 3 ms attack and recovers with time constant tau."""
    g = np.ones(n)
    a = int(attack * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(a, 1)))
    seg_t = np.arange(int(length * SR)) / SR
    shape = np.exp(-seg_t / tau)
    shape[:a] *= ramp
    for t, w in hits:
        i = int(round(t * SR))
        if i >= n:
            continue
        j = min(n, i + len(shape))
        s0 = max(0, -i)
        g[max(i, 0):j] = np.minimum(g[max(i, 0):j], 1 - depth * w * shape[s0:j - i])
    return g[:, None]


def sidechain_key(key, depth=0.55, tau=0.14, hop=48):
    """Duck gain (n, 1) that follows a key signal (a kick stem): instant attack, release tau."""
    k = np.abs(_stereo(key)).max(axis=1)
    nb = int(np.ceil(len(k) / hop))
    v = np.pad(k, (0, nb * hop - len(k))).reshape(nb, hop).max(axis=1)
    v = v / (np.percentile(v[v > 0], 99.9) if np.any(v > 0) else 1.0)
    rel = np.exp(-hop / (tau * SR))
    e = np.empty(nb)
    cur = 0.0
    for i, x in enumerate(v.tolist()):
        cur = x if x > cur else cur * rel
        e[i] = cur
    g = 1 - depth * np.clip(e, 0, 1)
    return np.interp(np.arange(len(k)), np.arange(nb) * hop + hop / 2, g)[:, None]


def mono_below(x, fc=120):
    """Sum the stereo image to mono below fc so the low end is centred and translates to phones."""
    x = _stereo(x)
    mid = x.mean(axis=1, keepdims=True)
    side = signal.sosfilt(signal.butter(2, fc, 'high', fs=SR, output='sos'), (x[:, :1] - x[:, 1:]) / 2, axis=0)
    return np.concatenate([mid + side, mid - side], axis=1)


def bus_comp(x, thr=-14.0, ratio=2.0, attack=0.015, release=0.12, rms=0.005, hop=48):
    """Glue compressor: true RMS detector (rms seconds), gain computed at a 1 ms control rate.
    Returns (y, gain_reduction_db per sample)."""
    x = _stereo(x)
    det = signal.sosfilt(signal.butter(1, 1 / (2 * np.pi * rms), 'low', fs=SR, output='sos'), np.mean(x ** 2, axis=1))
    nb = int(np.ceil(len(x) / hop))
    d = np.pad(det, (0, nb * hop - len(det)), mode='edge').reshape(nb, hop).max(axis=1)
    over = np.maximum(0, 10 * np.log10(np.maximum(d, 1e-12)) - thr)
    target = -over * (1 - 1 / ratio)
    att, rel = np.exp(-hop / (attack * SR)), np.exp(-hop / (release * SR))
    g = np.empty(nb)
    cur = 0.0
    for i, v in enumerate(target.tolist()):
        cur = v + (cur - v) * (att if v < cur else rel)
        g[i] = cur
    gr = np.interp(np.arange(len(x)), np.arange(nb) * hop + hop / 2, g)
    return x * (10 ** (gr / 20))[:, None], gr


def limiter(x, ceiling=0.77, look=0.004, release=0.08, hop=16):
    """True-peak lookahead limiter. Detection on a 4x oversampled copy catches inter-sample
    overs; the gain is computed per 16-sample block (min over neighbouring blocks, so the
    interpolated gain never exceeds what any sample needs), attacks over the lookahead and
    releases with time constant `release`. Returns (y, gain per sample)."""
    x = _stereo(x)
    n = len(x)
    up = np.abs(signal.resample_poly(x, 4, 1, axis=0)).max(axis=1)
    a = np.pad(up, (0, max(0, 4 * n - len(up))))[:4 * n].reshape(n, 4).max(axis=1)
    la = max(hop, int(look * SR))
    peak = maximum_filter1d(a, size=la, origin=-(la // 2), mode='nearest')  # max over [i, i+la)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    nb = int(np.ceil(n / hop))
    gb = np.pad(need, (0, nb * hop - n), mode='edge').reshape(nb, hop).min(axis=1)
    k = max(3, la // hop)
    win = np.hanning(k + 2)[1:-1]
    win /= win.sum()
    gb = np.minimum(gb, np.convolve(np.pad(gb, (len(win) // 2, len(win) - 1 - len(win) // 2), mode='edge'), win, mode='valid'))
    rel = np.exp(-hop / (release * SR))
    out = np.empty(nb)
    cur = 1.0
    for i, v in enumerate(gb.tolist()):
        cur = v if v < cur else v + (cur - v) * rel
        out[i] = cur
    out = minimum_filter1d(out, 3, mode='nearest')
    g = np.minimum(np.interp(np.arange(n), np.arange(nb) * hop + hop / 2, out), need)
    return x * g[:, None], g


def loudness_passes(x, target=-14.0, ceiling=0.77, passes=3, tol=0.05, max_passes=8):
    """Normalize to target LUFS then limit; repeat (limiting lowers loudness) until within tol."""
    g = np.ones(len(x))
    for p in range(max_passes):
        x = x * 10 ** ((target - integrated(x)) / 20)
        x, g = limiter(x, ceiling)
        if p + 1 >= passes and abs(integrated(x) - target) <= tol:
            break
    return x, g


def glue(x, drive=1.15, air=0.25):
    """Soft saturation and a gentle high shelf (air)."""
    x = np.tanh(x * 0.9 * drive) / np.tanh(drive)
    if air:
        x = x + signal.sosfilt(signal.butter(1, 2500, 'high', fs=SR, output='sos'), x, axis=0) * air
    return x


def edge_fades(x, head=HEAD_FADE, tail=END_FADE_MIN):
    """Raised-cosine fade-in over `head` s from exact zero at sample 0, and fade-out over `tail` s
    to zero on the last sample. Run after the limiter, so nothing raises the edges again."""
    x = _stereo(x).copy()
    n = len(x)
    for d, rising in ((head, True), (tail, False)):
        k = min(n, int(round(d * SR)))
        if k > 1:
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / (k - 1))  # 0 -> 1, raised cosine
            if rising:
                x[:k] *= w[:, None]
            else:
                x[-k:] *= w[::-1, None]
    return x


def finish(mix, lufs=-14.0, ceiling=0.77, mono_hz=120, comp=True, drive=1.15, air=0.25,
           hp_hz=28, fade=0.0, end_fade=0.06, tail_zero=480):
    """Full chain for a summed mix. Returns (master, info) where info has the limiter trace."""
    x = _stereo(mix)
    if hp_hz:
        x = signal.sosfilt(signal.butter(2, hp_hz, 'high', fs=SR, output='sos'), x, axis=0)
    if mono_hz:
        x = mono_below(x, mono_hz)
    gr = np.zeros(len(x))
    if comp:
        x, gr = bus_comp(x)
    if drive:
        x = glue(x, drive, air)
    if fade and fade > 0:
        t = np.arange(len(x)) / SR
        x = x * (np.clip((len(x) / SR - 0.03 - t) / fade, 0, 1) ** 2)[:, None]
    x, g = loudness_passes(x, lufs, ceiling)
    x = edge_fades(x, HEAD_FADE, max(end_fade or 0.0, END_FADE_MIN))
    if tail_zero:
        x[-tail_zero:] = 0
    lim_db = -20 * np.log10(np.maximum(g, 1e-9))
    return x, {'limiter_gr_db': lim_db, 'comp_gr_db': -gr}


def gate(report, target, tol, max_tp):
    fails = []
    if abs(report['lufs'] - target) > tol:
        fails.append(f"integrated {report['lufs']} LUFS not within {target}+-{tol}")
    if report['true_peak_db'] > max_tp:
        fails.append(f"true peak {report['true_peak_db']} dBTP above {max_tp}")
    if not report['last_480_zero']:
        fails.append('last 480 samples are not digital zero')
    return fails


# ------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------
def _read(path):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=True, dtype='float64')
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g, axis=0)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description='Master a WAV to target loudness, or measure one (--measure).')
    ap.add_argument('input', help='input WAV (any rate; resampled to 48 kHz)')
    ap.add_argument('output', nargs='?', help='output WAV (48 kHz, 24-bit)')
    ap.add_argument('--measure', action='store_true', help='only measure the input')
    ap.add_argument('--lufs', type=float, default=-14.0, help='target integrated loudness (default -14)')
    ap.add_argument('--ceiling', type=float, default=0.77, help='limiter ceiling, linear (default 0.77)')
    ap.add_argument('--tol', type=float, default=0.5, help='gate tolerance in LU (default 0.5)')
    ap.add_argument('--max-tp', type=float, default=-1.5, help='gate: max true peak dBTP (default -1.5)')
    ap.add_argument('--sidechain', metavar='KEY.wav', help='duck the input from this key signal first')
    ap.add_argument('--sc-depth', type=float, default=0.55)
    ap.add_argument('--sc-tau', type=float, default=0.14)
    ap.add_argument('--mono-hz', type=float, default=120, help='mono below this frequency; 0 disables')
    ap.add_argument('--no-comp', action='store_true', help='skip the bus compressor')
    ap.add_argument('--no-glue', action='store_true', help='skip saturation and air')
    ap.add_argument('--fade', type=float, default=0.0, help='quadratic fade-out over the last N seconds')
    ap.add_argument('--tail-zero', type=int, default=480, help='samples zeroed at the end (default 480)')
    ap.add_argument('--json', metavar='PATH', help="write the report JSON here ('-' for stdout)")
    a = ap.parse_args(argv)
    try:
        x = _read(a.input)
        if a.measure:
            rep = measure(x)
            fails = gate(rep, a.lufs, a.tol, a.max_tp)
        else:
            if not a.output:
                ap.error('output path required unless --measure')
            if a.sidechain:
                key = _read(a.sidechain)
                key = np.pad(key, ((0, max(0, len(x) - len(key))), (0, 0)))[:len(x)]
                x = x * sidechain_key(key, a.sc_depth, a.sc_tau)
            y, info = finish(x, a.lufs, a.ceiling, a.mono_hz, not a.no_comp,
                             0 if a.no_glue else 1.15, 0 if a.no_glue else 0.25,
                             fade=a.fade, tail_zero=a.tail_zero)
            import soundfile as sf
            os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
            sf.write(a.output, y.astype(np.float32), SR, subtype='PCM_24')
            rep = measure(y)
            rep['limiter_max_gr_db'] = round(float(info['limiter_gr_db'].max()), 2)
            rep['output'] = a.output
            fails = gate(rep, a.lufs, a.tol, a.max_tp)
        rep['pass'] = not fails
        rep['fails'] = fails
    except Exception as e:  # noqa: BLE001
        print('error:', e, file=sys.stderr)
        return 1
    if a.json == '-':
        print(json.dumps(rep, indent=1))
    elif a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(rep, open(a.json, 'w'), indent=1)
    print(f"{'PASS' if not fails else 'FAIL'}  {rep['lufs']} LUFS  TP {rep['true_peak_db']} dBTP  "
          f"LRA {rep['lra']} LU  loudest {rep['momentary_max']} LUFS-M at {rep['momentary_max_t']}s"
          + ''.join(f'\n  - {f}' for f in fails))
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
