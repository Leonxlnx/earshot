"""A burst glitch: a digital glitch that HITS (its loudest burst first, within ~20 ms), then bursts that speed up into
the cut; every sample is computed.

What the reference clips do (a private licensed set, 'glitch', 10 events in 10 films;
the clips are local study material and are not shipped): onset 18 ms (16-22) from the first moment within 20 dB of
the loudest point to it, 0.23 s (0.1-0.45) long, falling 20 dB in 76 ms (71-118); in the music films a mid glitch
(centroid 902 Hz, 802-1281) laid near mono (correlation 0.98), in the effects-led film crackle, then 11 bright bursts
(~10 kHz) speeding up into the final cut, fully decorrelated. earshot 0.3's glitches swell for 300-450 ms into their
loudest burst (pro_glitch, glitch: onset 12-18x too slow on the profile).
Layers:
  crackle  the first `crackle` share of the length: sparse 1.5-6 ms grains high-passed at crackle_hp, density rising
           to crackle_rate / s, crackle_db under the bursts (kept under -20 dB so the first burst is the onset)
  bursts   from crackle x dur to dur on a grid accelerating grid0 -> grid1 (power accel), each gated to `gate` of its
           slot with 0.4 ms edges; the FIRST burst is the loudest (accent_db over the others), the rest fall
           `fall_db` across the train with lvl_jit dB of spread; a burst is a crushed buffer (bits, srate) of noise
           mixed with a square chirp stepping down around tone_f (`tonal`), replayed from the previous burst with
           probability `repeat` (the buffer-repeat identity of a stutter)
  colour   a band-pass lo_f .. hi_f on the bursts (10 kHz glitches: lo_f 4-6 kHz; mid glitches: ~0.4-2.5 kHz)
  low      an optional low pulse under each burst (low_db, around low_f): the effects-led film's bursts thud at
           100-400 Hz under their 10 kHz crack
  stereo   `width` = share of bursts whose right channel reads its own buffer with a 0-2 ms offset; crackle grains are
           independent per channel
  end      the last burst ends on `dur`, then only its own decay end_tail
"""
import types

import numpy as np

from .. import dsp
from .. import fx

SR = dsp.SR
TARGET = 'a short digital glitch burst'
ALTS = ['radio static', 'a woodpecker', 'rain on a tin roof', 'a camera shutter']
REFSET = 'glitch'

SPACE = {
    'dur': (0.1, 0.8),
    'crackle': (0.0, 0.6),             # share of the length before the first burst
    'crackle_db': (-45.0, -20.0),
    'crackle_rate': (40.0, 600.0, 'log'),
    'crackle_hp': (1500.0, 9000.0, 'log'),
    'grid0': (0.018, 0.07, 'log'),     # first burst interval (s)
    'grid1': (0.008, 0.04, 'log'),     # last
    'accel': (0.5, 2.5, 'log'),
    'gate': (0.25, 0.9),
    'decay': (0.15, 2.0, 'log'),       # each burst's own decay, in slots
    'accent_db': (0.0, 12.0),          # the first burst over the others
    'fall_db': (-12.0, 6.0),           # level change across the rest of the train
    'lvl_jit': (0.0, 8.0),
    'bits': (3.0, 10.0),
    'srate': (1500.0, 20000.0, 'log'),
    'tonal': (0.0, 1.0),
    'tone_f': (300.0, 6000.0, 'log'),
    'repeat': (0.0, 0.9),
    'lo_f': (150.0, 7000.0, 'log'),
    'hi_f': (1500.0, 18000.0, 'log'),
    'width': (0.0, 1.0),
    'low_db': (-50.0, 0.0),            # a low pulse under each burst, re the bursts' peak (-50 = none)
    'low_f': (60.0, 500.0, 'log'),
    'end_tail': (0.004, 0.06, 'log'),
    'pattern': (0, 999, 'int'),
}
FAMILIES = {
    # earshot 0.4 phase B: hand-designed corners, then tuned toward each style on the measured features (the study's
    # dev-split clip of the effects-led film for burst, CLAP only up to the grade bar; the music films' glitch profile
    # for mid and chirp), every take audited (no rising chirp survives: the audit's rising-glide check is in the loss)
    'glitch': [
        # the effects-led film's: quiet crackle, then a bright 5-17 kHz burst train that speeds up into the cut, wide,
        # a faint low thud under each burst
        ('burst', {'dur': 0.726, 'crackle': 0.4906, 'crackle_db': -35.95, 'crackle_rate': 579.1,
                   'crackle_hp': 7237.0, 'grid0': 0.0229, 'grid1': 0.02146, 'accel': 1.588, 'gate': 0.6991,
                   'decay': 0.4964, 'accent_db': 3.442, 'fall_db': -12.0, 'lvl_jit': 7.164, 'bits': 7.138,
                   'srate': 5507.0, 'tonal': 0.02252, 'tone_f': 3311.0, 'repeat': 0.5548, 'lo_f': 5000.0,
                   'hi_f': 16990.0, 'width': 1.0, 'low_db': -40.98, 'low_f': 156.6, 'end_tail': 0.02849,
                   'pattern': 17}),
        # the music films' mid glitch: a 0.22 s crushed stutter around 1 kHz that hits first, near mono
        ('mid', {'dur': 0.2159, 'crackle': 0.1444, 'crackle_db': -32.74, 'crackle_rate': 40.0, 'crackle_hp': 3000.0,
                 'grid0': 0.03, 'grid1': 0.01798, 'accel': 0.8936, 'gate': 0.6, 'decay': 0.7133, 'accent_db': 3.068,
                 'fall_db': -4.756, 'lvl_jit': 2.992, 'bits': 6.225, 'srate': 8000.0, 'tonal': 0.2935,
                 'tone_f': 636.8, 'repeat': 0.371, 'lo_f': 150.0, 'hi_f': 2764.0, 'width': 0.1, 'low_db': -50.0,
                 'low_f': 150.0, 'end_tail': 0.005826, 'pattern': 5}),
        # data chirps: a pitched, crushed stutter (square tones stepping down) that hits first and rolls faster
        ('chirp', {'dur': 0.2788, 'crackle': 0.2608, 'crackle_db': -45.0, 'crackle_rate': 94.3, 'crackle_hp': 1734.0,
                   'grid0': 0.03896, 'grid1': 0.009084, 'accel': 0.7294, 'gate': 0.7683, 'decay': 1.285,
                   'accent_db': 6.973, 'fall_db': -9.256, 'lvl_jit': 2.413, 'bits': 4.329, 'srate': 6357.0,
                   'tonal': 0.85, 'tone_f': 446.1, 'repeat': 0.2995, 'lo_f': 459.1, 'hi_f': 6342.0,
                   'width': 0.7961, 'low_db': -50.0, 'low_f': 150.0, 'end_tail': 0.02571, 'pattern': 29}),
    ],
}
DEFAULTS = dict(FAMILIES['glitch'][0][1])
TARGETS = {'glitch': TARGET}


def _box(n, edge=0.0004):
    e = np.ones(n)
    k = min(n // 2, max(2, dsp.n_of(edge)))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    e[:k] = ramp
    e[-k:] = ramp[::-1]
    return e


def _buffer(p, pr, n, r):
    """one burst buffer: crushed noise + a stepped square chirp (pr = the pattern rng: pitches; r = the noise)"""
    out = r.standard_normal(n)
    if p['tonal'] > 0.02:
        f = p['tone_f'] * 2 ** pr.uniform(-0.5, 0.5)
        steps = int(pr.integers(1, 4))                  # the chirp steps DOWN inside a burst (never a rising whine)
        fr = np.repeat(f * 2 ** (np.sort(pr.integers(-4, 5, steps))[::-1] / 6), int(np.ceil(n / steps)))[:n]
        out = (1 - 0.7 * p['tonal']) * out / 3.0 + p['tonal'] * dsp.osc(fr, 'square')
    return fx.crush(out / (np.abs(out).max() + 1e-9), bits=int(round(p['bits'])), rate=p['srate'])


def _grid(p, t0, t1):
    """burst start times from t0, intervals accelerating grid0 -> grid1, the last slot ending at t1"""
    out, t = [], t0
    span = max(1e-3, t1 - t0)
    while True:
        u = min(1.0, (t - t0) / span) ** p['accel']
        d = p['grid0'] + (p['grid1'] - p['grid0']) * u
        if t + d > t1 + 0.25 * d:
            break
        out.append((t, d))
        t += d
    return out or [(t0, span)]


def render(p, r):
    p = {**DEFAULTS, **p}
    pr = dsp.rng(int(p['pattern']) * 7919 + 101)
    dur = p['dur']
    n_end = dsp.n_of(dur)
    tail = dsp.n_of(p['end_tail'] * 5)
    n = n_end + tail
    y = np.zeros((n, 2))
    t_st = dur * p['crackle']
    bursts = _grid(p, t_st, dur)
    L = dsp.n_of(max(d for _, d in bursts) + 0.004)
    bufs = [[_buffer(p, pr, L, r) for _ in range(3)] for _ in range(2)]    # a buffer set per channel
    nb = len(bursts)
    prev = [0, 0]
    pick_l = (0, 0)
    for k, (tb, d) in enumerate(bursts):
        last = k == nb - 1
        g = max(16, dsp.n_of(d * p['gate']))
        if last:
            g = max(16, n_end - dsp.n_of(tb))
        db = 0.0 if k == 0 else -p['accent_db'] + p['fall_db'] * (k - 1) / max(1, nb - 2) \
            - abs(p['lvl_jit'] * pr.standard_normal())
        for c in range(2):
            own = c == 0 or pr.uniform() <= p['width']    # the right channel reads its own set, or the left's take
            if not own:
                j, off = pick_l
            else:
                j = prev[c] if pr.uniform() < p['repeat'] else int(pr.integers(0, 3))
                off = dsp.n_of(pr.uniform(0, 0.002)) if c == 1 and not last and k else 0
            if c == 0:
                pick_l = (j, off)
            prev[c] = j
            b = bufs[c if own else 0][j]
            m = min(g, L) if not last else g
            seg = np.resize(b, m) if m > L else b[:m].copy()
            env = _box(m) * np.exp(-np.arange(m) / SR / (p['decay'] * d))
            if last:                                # the final burst holds to the cut, then its own decay
                seg = np.concatenate([seg, np.resize(b, tail)])
                env = np.concatenate([env, env[-1] * np.exp(-np.arange(tail) / SR / (p['end_tail'] / 1.3))])
            i = dsp.n_of(tb) + off
            mm = min(len(seg), n - i)
            if mm > 0:
                y[i:i + mm, c] += (seg * env)[:mm] * 10 ** (db / 20) * float(np.exp(0.1 * r.standard_normal()))
    y = dsp.butter(y, 'band', (p['lo_f'], max(p['hi_f'], p['lo_f'] * 1.5)), 2)
    pk = np.abs(y).max() + 1e-12
    if p.get('low_db', -50.0) > -49.5:                 # drawn only when used: members without it render as before
        low = np.zeros(n)
        for k, (tb, d) in enumerate(bursts):
            i = dsp.n_of(tb)
            m = min(dsp.n_of(d * p['gate']) + dsp.n_of(0.01), n - i)
            if m < 16:
                continue
            tt = np.arange(m) / SR
            f0 = p['low_f'] * 2 ** pr.uniform(-0.4, 0.4)
            lo = np.sin(2 * np.pi * f0 * tt + r.uniform(0, 2 * np.pi)) * np.exp(-tt / max(0.004, 0.5 * d))
            low[i:i + m] += lo * _box(m, 0.001) * (1.0 if k == 0 else 10 ** (-p['accent_db'] / 20))
        y += (pk * 10 ** (p['low_db'] / 20) * low / (np.abs(low).max() + 1e-12))[:, None]
    if p['crackle'] > 0.02 and t_st > 0.01:
        cr = np.zeros((n, 2))
        for c in range(2):
            t = 0.0
            while True:
                u = t / t_st
                t += float(r.exponential(1.0 / (p['crackle_rate'] * (0.25 + 0.75 * u ** 1.5))))
                if t >= t_st:
                    break
                m = dsp.n_of(r.uniform(0.0015, 0.006))
                gr = r.standard_normal(m) * np.hanning(m)
                i = dsp.n_of(t)
                mm = min(m, n - i)
                cr[i:i + mm, c] += gr[:mm] * 10 ** ((18 * np.log10(r.uniform(0.05, 1.0)) + 6 * u) / 20)
        cr = dsp.butter(cr, 'high', p['crackle_hp'], 2)
        y += cr / (np.abs(cr).max() + 1e-12) * pk * 10 ** (p['crackle_db'] / 20)
    y = dsp.butter(y, 'low', 18500.0, 6)
    y[:2] = 0.0
    return dsp.fade(y, 0.0005, 0.003) / (np.abs(y).max() + 1e-12) * 0.89


def recipe_for(cat='glitch', preset=0):
    fam = FAMILIES[cat]
    i = preset if isinstance(preset, int) else [nm for nm, _ in fam].index(preset)
    return types.SimpleNamespace(__name__=__name__, SPACE=dict(SPACE), DEFAULTS=dict(fam[i][1]), render=render,
                                 TARGET=TARGETS[cat], ALTS=ALTS, REFSET=REFSET, PRESET=fam[i][0])
