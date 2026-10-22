"""A reverse cymbal: the bright, wide swell that rises into a cut and stops just after it, and the darker reverse swell
of the music films; every sample is computed.

What the reference clips do (a private licensed set, 'reverse_swell', 16 events in 10
films; the clips are local study material and are not shipped): about one per film, at a section change, ~1 s long
(0.57-1.29), the loudest point 42 ms after the cut (+23..+62). The effects-led film's are reverse CYMBALS: bright
(centroid 2.8-3.2 kHz, the energy between 1 and 16 kHz), fully decorrelated (L/R correlation 0.08-0.16), a granular
shimmer running through them, one stopping ~250 ms after it peaks, one ringing out (-20 dB 0.5 s later). The music
films' are darker and near mono: centroid 556 Hz (310-884), 19 % under 150 Hz, rise 189 ms (76-413) to the loudest
point, -20 dB 130 ms (68-243) after it. earshot 0.3 had only reverse whooshes (52-463 Hz, near mono).

How it is built. A real reverse cymbal is a struck cymbal's ring played backwards: the partials that ring longest
(the lower ones) are there first, the fast-dying highs arrive last, and it ends on the strike. Here the swell is built
directly in that order (no recorded or rendered sound is reversed, so the stop is designed, not a reversed transient):
  texture   a stationary metal: `partials` inharmonic sines spread log-uniformly over `bw` octaves around cym_f
            (Gaussian weights in log-frequency, each shimmering a little), mixed (`metal`) with noise in the same band;
            a separate texture per channel, blended by `width` (correlation about sqrt(1 - width^2))
  level     in dB: -swell_db * (1 - t / length)^curve up to the peak (curve > 1: slow, then accelerating into it),
            then `hold` s easing hold_db down (the plateau after the cut), then straight in dB with time constant
            tail_tau (a stop or a ring-out); grains: a log-normal flutter at about flut_f (the striations heard in the
            designed cymbals)
  colour    a low-pass that opens `open` octaves with the level (the highs arrive late), a high-pass at hp, an optional
            low body (`low_db`: noise 100-450 Hz following the level, the music films' darker swells)
  space     a short smooth reverb (room, rt60), the edit ends at -60 dB with a fade.
"""
import types

import numpy as np

from .. import dsp
from .. import fx

SR = dsp.SR
TARGET = 'a reverse cymbal swell'
ALTS = ['a jet engine', 'a wind gust', 'rain', 'a sword being sharpened']
REFSET = 'reverse_swell'

SPACE = {
    'length': (0.3, 2.0),              # s from the first sample to the peak
    'swell_db': (12.0, 60.0),          # how far below the peak it starts
    'curve': (0.5, 3.5, 'log'),        # shape of the rise in dB (1 = a straight dB ramp, a reversed exponential ring)
    'hold': (0.0, 0.45),               # s of plateau after the peak
    'hold_db': (-12.0, 0.0),           # level change across the plateau
    'tail_tau': (0.008, 0.6, 'log'),   # s, the fall after the plateau (time constant)
    'cym_f': (300.0, 7000.0, 'log'),   # centre of the metal band
    'bw': (1.0, 6.0),                  # its width (octaves)
    'metal': (0.0, 1.0),               # partials vs noise
    'partials': (12, 160, 'int'),
    'shimmer': (0.0, 1.0),             # slow random AM of each partial
    'open': (0.0, 4.0),                # octaves the low-pass opens with the level
    'hp': (80.0, 2500.0, 'log'),
    'low_db': (-50.0, 0.0),            # a low noise body following the level (re the metal)
    'flut_depth': (0.0, 1.0),
    'flut_f': (6.0, 60.0, 'log'),
    'width': (0.0, 1.0),
    'room': (0.0, 0.5),
    'rt60': (0.2, 2.0, 'log'),
}
FAMILIES = {
    # earshot 0.4 phase B: hand-designed corners, then tuned toward each style on the measured features (the study's
    # dev-split clips of the effects-led film for the cymbals, the music films' category profile for the dark swell)
    # and, for the cymbals, on CLAP only up to the grade bar (cosine 0.70: no pull past it)
    'reverse_swell': [
        # the effects-led film's opening cymbal: bright, wide, grainy, a plateau after the peak, then a stop
        ('cymbal', {'length': 0.5037, 'swell_db': 19.55, 'curve': 1.831, 'hold': 0.22, 'hold_db': -8.039,
                    'tail_tau': 0.03096, 'cym_f': 2600.0, 'bw': 3.469, 'metal': 0.6317, 'partials': 78,
                    'shimmer': 0.5093, 'open': 0.0, 'hp': 118.7, 'low_db': -50.0, 'flut_depth': 0.6266,
                    'flut_f': 18.8, 'width': 0.98, 'room': 0.0, 'rt60': 1.386}),
        # a cymbal that swells harder and rings out after the peak (-20 dB a quarter second later)
        ('ring', {'length': 0.808, 'swell_db': 40.86, 'curve': 2.308, 'hold': 0.0, 'hold_db': -2.062,
                  'tail_tau': 0.1542, 'cym_f': 2070.0, 'bw': 5.575, 'metal': 0.7482, 'partials': 79,
                  'shimmer': 0.8912, 'open': 3.093, 'hp': 658.9, 'low_db': -50.0, 'flut_depth': 0.4417,
                  'flut_f': 21.97, 'width': 1.0, 'room': 0.129, 'rt60': 1.209}),
        # the music films' reverse swell: darker, near mono, a low body under it, a quick fall after the peak
        ('dark', {'length': 0.5391, 'swell_db': 30.0, 'curve': 0.5, 'hold': 0.03684, 'hold_db': -0.4249,
                  'tail_tau': 0.03842, 'cym_f': 346.0, 'bw': 3.456, 'metal': 0.259, 'partials': 117,
                  'shimmer': 0.3166, 'open': 2.554, 'hp': 80.0, 'low_db': 0.0, 'flut_depth': 0.004984,
                  'flut_f': 28.16, 'width': 0.35, 'room': 0.08302, 'rt60': 0.5783}),
        # added for the 0.4 bank (a model-free search from a hand-set corner toward a designed aim, takes audited):
        # a short bright cymbal for a quick cut (0.33 s into the peak, a short plateau, a stop) ...
        ('short', {'length': 0.33, 'swell_db': 24.0, 'curve': 0.8282, 'hold': 0.08, 'hold_db': -6.0,
                   'tail_tau': 0.025, 'cym_f': 3200.0, 'bw': 3.2, 'metal': 0.6, 'partials': 70, 'shimmer': 0.04221,
                   'open': 0.5, 'hp': 300.0, 'low_db': -50.0, 'flut_depth': 0.8351, 'flut_f': 24.0, 'width': 0.95,
                   'room': 0.05, 'rt60': 0.8}),
        # ... and a mid-band reverse breath (more noise than metal, half wide) between the dark swell and the cymbals
        ('breath', {'length': 0.7, 'swell_db': 30.0, 'curve': 1.3, 'hold': 0.094, 'hold_db': 0.0, 'tail_tau': 0.06,
                    'cym_f': 1200.0, 'bw': 1.499, 'metal': 0.12, 'partials': 60, 'shimmer': 0.3, 'open': 1.5,
                    'hp': 150.0, 'low_db': -24.0, 'flut_depth': 0.366, 'flut_f': 14.0, 'width': 0.5818,
                    'room': 0.2272, 'rt60': 0.5846}),
    ],
}
DEFAULTS = dict(FAMILIES['reverse_swell'][0][1])
TARGETS = {'reverse_swell': TARGET}


def _lfnoise(n, fc, r):
    w = dsp.butter(r.standard_normal(n), 'low', fc, order=2)
    return w / (w.std() + 1e-12)


def level_db(p, t):
    """the level curve in dB re the peak (module doc) and the peak time"""
    T, hd = p['length'], p['hold']
    u = np.clip(t / T, 0.0, 1.0)
    L = -p['swell_db'] * (1.0 - u) ** p['curve']
    if hd > 0:
        s = (t > T) & (t <= T + hd)
        L[s] = p['hold_db'] * (t[s] - T) / hd
    late = t > T + hd
    L[late] = p['hold_db'] - 8.686 * (t[late] - T - hd) / p['tail_tau']
    return L, T


def _texture(p, n, r):
    """one channel of stationary metal: inharmonic partials + band noise (module doc), unit RMS"""
    t = np.arange(n) / SR
    lo = max(60.0, p['cym_f'] / 2 ** (p['bw'] / 2))
    hi = min(18000.0, p['cym_f'] * 2 ** (p['bw'] / 2))
    k = int(round(p['partials']))
    f = np.exp(r.uniform(np.log(lo), np.log(hi), k))
    sig = max(0.3, p['bw'] / 3.0)
    a = np.exp(-0.5 * (np.log2(f / p['cym_f']) / sig) ** 2) * r.uniform(0.4, 1.0, k)
    met = np.zeros(n)
    sh = _lfnoise(n, 3.0, r)
    for fi, ai in zip(f, a):
        am = 1.0 + 0.35 * p['shimmer'] * np.roll(sh, int(r.integers(0, n)))
        met += ai * am * np.sin(2 * np.pi * fi * t + r.uniform(0, 2 * np.pi))
    met /= np.sqrt(np.mean(met ** 2)) + 1e-12
    nz = dsp.butter(r.standard_normal(n), 'band', (lo, hi), 2)
    nz = dsp.peq(nz, p['cym_f'], 6.0, 1.0)
    nz /= np.sqrt(np.mean(nz ** 2)) + 1e-12
    x = p['metal'] * met + (1 - p['metal']) * nz
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12), hi


def render(p, r):
    p = {**DEFAULTS, **p}
    tail = p['length'] + p['hold'] + p['tail_tau'] * 60.0 / 8.686 + 0.02
    n = dsp.n_of(min(4.0, tail + 0.3 * p['rt60'] * min(1.0, 4 * p['room'])))
    t = np.arange(n) / SR
    L, _ = level_db(p, t)
    a = 10 ** (np.maximum(L, -120.0) / 20)
    xL, hi = _texture(p, n, r)
    xR, _ = _texture(p, n, r)
    lowL = dsp.butter(r.standard_normal(n), 'band', (100.0, 450.0), 2)
    lowR = dsp.butter(r.standard_normal(n), 'band', (100.0, 450.0), 2)
    flut = [np.exp(p['flut_depth'] * _lfnoise(n, p['flut_f'], r) - 0.5 * p['flut_depth'] ** 2) for _ in range(2)]
    w = p['width']
    k = np.sqrt(max(0.0, 1 - w * w))
    chans = []
    for x, lo_, fl in ((xL, lowL, flut[0]), (k * xL + w * xR, k * lowL + w * lowR, k * flut[0] + w * flut[1])):
        cut = np.clip(hi * 1.3 * 2.0 ** (-p['open'] * (1.0 - a ** 0.5)), 300.0, 20000.0)
        y = dsp.svf(x, cut, 0.7, 'low')
        if p['low_db'] > -49.5:
            y = y + 10 ** (p['low_db'] / 20) * lo_ / (np.sqrt(np.mean(lo_ ** 2)) + 1e-12)
        chans.append(y * a * fl)
    y = np.stack(chans, axis=1)
    y = dsp.butter(y, 'high', p['hp'], order=2)
    if p['room'] > 0.01:
        yr = fx.reverb(y, r, rt60=p['rt60'], size=0.5, predelay=0.008, damp=0.4, wet=p['room'], width=1.0,
                       lines=16, stereo=True, tail=max(0.2, p['rt60']))
        y = yr[:n] if np.isfinite(yr).all() else y
    y = dsp.butter(y, 'low', 19000, order=4)
    y = _edit(y, -60.0)
    return y / (np.abs(y).max() + 1e-12) * 0.89


def _edit(y, end_db, fin=0.004, fout=0.02):
    """a fade in from the first sample, the end where the 5 ms envelope has fallen end_db under its peak, a fade out"""
    h = dsp.n_of(0.005)
    m = y.mean(axis=1)
    k = len(m) // h
    e = 20 * np.log10(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12)
    i = int(np.argmax(e))
    below = np.nonzero(e[i:] < e[i] + end_db)[0]
    end = min(len(y), (i + (below[0] if len(below) else k - i)) * h + dsp.n_of(fout))
    return dsp.fade(y[:end], fin, fout)


def recipe_for(cat='reverse_swell', preset=0):
    fam = FAMILIES[cat]
    i = preset if isinstance(preset, int) else [nm for nm, _ in fam].index(preset)
    return types.SimpleNamespace(__name__=__name__, SPACE=dict(SPACE), DEFAULTS=dict(fam[i][1]), render=render,
                                 TARGET=TARGETS[cat], ALTS=ALTS, REFSET=REFSET, PRESET=fam[i][0])
