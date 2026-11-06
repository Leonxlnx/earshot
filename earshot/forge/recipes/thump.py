"""A thump-whoosh: the big low whoosh that hits fast and rings out, the music films' scene-change whoosh (forge
category 'whoosh_big', members on the pass engine of recipes/whoosh.py).

What the reference clips do (a private licensed set, 'whoosh_big', 23 events in 11
films, medians with the interquartile range; the clips are local study material and are not shipped): 0.69 s
(0.42-0.84) long, its loudest point 82 ms (57-190) after the onset and within a frame of the cut, then a long low
tail: -20 dB 475 ms (180-766) after the peak; centroid 336 Hz (176-415) with 80 % of the energy under 150 Hz, mono
(L/R correlation 0.95). earshot 0.3's big whooshes approach for 0.25-0.6 s and die 3-6x faster after the peak (the
profile distance stayed 0.99-1.67 whatever the knobs did).

How it is built: the pass engine (`whoosh.render`: one level curve, moving air bands, low weight, sub swell, hiss,
room, edit) with a short accelerating approach (rise ~0.1 s at rise_k < 1) and a held, convex fall (fall_k > 1:
the tail stays up, then drops), most of the level in the sub and low bands; plus the THUMP: a low body struck at the
pass peak (a sine sliding thump_f0 -> thump_f1, decaying in thump_tau, thump_lead s before the peak), which is what
makes the whoosh land on the cut instead of drifting through it. The thump follows the engine's peak time, so the
whoosh knobs (dur 'shape', apex: rise / hold / fall) move it with the pass.
Every sample is computed; no recorded audio is used.
"""
import types

import numpy as np

from .. import dsp
from . import whoosh as _w

SR = dsp.SR
TARGET = 'a deep low whoosh that lands with a thump'
ALTS = ['a strong wind', 'a bass drum', 'a car passing', 'thunder']
REFSET = 'whoosh_big'

THUMP_SPACE = {
    'thump_db': (-40.0, 6.0),          # the thump's peak re the pass's (-40 or less = off)
    'thump_f0': (60.0, 220.0, 'log'),
    'thump_f1': (32.0, 90.0, 'log'),
    'thump_tau': (0.03, 0.4, 'log'),   # its amplitude time constant (s)
    'thump_lead': (-0.04, 0.03),       # s before (+) or after (-) the pass peak that it strikes
}
SPACE = {**_w.SPACE, **THUMP_SPACE}
THUMP_KEYS = tuple(THUMP_SPACE)

# the base: a short accelerating approach, a held convex fall, sub + low weight, a low air band, a dull hiss
BASE = {**_w.BIG, **_w.OPTIONAL,
        'rise': 0.13, 'rise_k': 0.8, 'hold': 0.01, 'fall': 0.55, 'fall_k': 1.6, 'start_db': -34.0, 'track': 0.7,
        'f_peak': 200.0, 'up_oct': 1.0, 'down_oct': 1.6, 'dop_st': 2.0, 'dop_w': 0.06, 'q_air': 1.0,
        'lp_ratio': 4.0, 'flut_depth': 0.2, 'flut_f': 10.0, 'low_db': -2.0, 'low_f': 110.0, 'low_smooth': 0.03,
        'sub_db': 0.0, 'sub_f': 52.0, 'sub_glide': 0.4, 'sub_drive': 1.0, 'hiss_db': -26.0, 'hiss_hp': 1500.0,
        'hiss_open': 0.8, 'hiss_pow': 1.3, 'tone_db': -50.0, 'grain_db': -50.0, 'room_wet': 0.15, 'rt60': 0.7,
        'room_damp': 0.45, 'pan': 0.05, 'width': 0.08, 'drive_db': 2.5, 'end_db': -64.0, 'end_fade': 0.08,
        'thump_db': -6.0, 'thump_f0': 120.0, 'thump_f1': 50.0, 'thump_tau': 0.12, 'thump_lead': 0.01}
FAMILIES = {
    # earshot 0.4 phase B: three hand-designed corners of BASE, each then tuned toward the music films' whoosh_big
    # profile on the measured features (length, approach, tail, centroid, low share, tonal share, width); the deep and
    # airy members keep their character keys (f_peak, sub_f / hiss, width) and a fixed thump level while tuning
    'whoosh_big': [
        ('thump', {'sub_db': -14.85, 'lp_ratio': 4.211, 'up_oct': 2.118, 'flut_f': 10.37, 'hiss_hp': 3308.0,
                   'thump_f1': 70.85, 'sub_drive': 1.462, 'rise_k': 1.266, 'flut_depth': 0.26, 'sub_f': 59.26,
                   'low_db': -22.24, 'rt60': 0.3, 'sub_glide': 0.8055, 'low_f': 283.6, 'width': 0.4326,
                   'hiss_db': -15.21, 'track': 0.871, 'down_oct': 1.68}),
        # deeper and longer: the sub sinks further, the tail rings for ~0.8 s
        ('thump_deep', {'rise': 0.1401, 'fall': 0.75, 'fall_k': 1.761, 'f_peak': 150.0, 'sub_f': 44.0,
                        'sub_glide': 0.8703, 'low_f': 241.1, 'hiss_db': -15.95, 'thump_f0': 92.21, 'thump_f1': 41.61,
                        'thump_tau': 0.1913, 'rt60': 0.8878, 'q_air': 1.343, 'lp_ratio': 1.959, 'sub_db': -20.64,
                        'sub_drive': 0.9437, 'low_db': -17.48, 'flut_depth': 0.06493, 'width': 0.2907,
                        'rise_k': 0.8922, 'flut_f': 22.93, 'drive_db': 5.561, 'thump_db': -9.0, 'up_oct': 2.965,
                        'hiss_hp': 4168.0, 'thump_lead': 1e-05, 'track': 0.9656, 'down_oct': 1.4,
                        'room_wet': 0.2214}),
        # airier: more of the moving air band and a brighter hiss over a lighter sub, a little wider
        ('thump_airy', {'rise': 0.11, 'fall': 0.45, 'f_peak': 320.0, 'q_air': 1.52, 'sub_db': -4.72,
                        'low_db': -4.037, 'hiss_db': -20.0, 'hiss_hp': 5687.0, 'width': 0.2, 'flut_depth': 0.4371,
                        'flut_f': 9.589, 'thump_db': -12.0, 'thump_tau': 0.06806, 'drive_db': 6.444, 'low_f': 50.94,
                        'lp_ratio': 7.464, 'up_oct': 2.596, 'thump_f1': 45.69, 'thump_f0': 115.4, 'track': 0.7214,
                        'down_oct': 2.257, 'sub_f': 57.99}),
    ],
}
DEFAULTS = dict(BASE)
TARGETS = {'whoosh_big': TARGET}


def preset(name, cat='whoosh_big'):
    """the full parameter dict of one family member"""
    return {**BASE, **dict(FAMILIES[cat])[name]}


def preset_full(cat, params):
    return {**BASE, **params}


def _thump(p, n, i0, r):
    """the struck low body from sample i0: a sine sliding thump_f0 -> thump_f1, 0.6 ms rise, decay thump_tau"""
    y = np.zeros(n)
    m = n - i0
    if m <= 16:
        return y
    t = np.arange(m) / SR
    f = p['thump_f1'] + (p['thump_f0'] - p['thump_f1']) * np.exp(-t / 0.025)
    ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(-0.3, 0.3)
    y[i0:] = np.sin(ph) * np.exp(-t / p['thump_tau']) * (1 - np.exp(-t / 0.0006))
    return y


def render(p, r):
    """the pass engine on BASE + p (whoosh.OPTIONAL keys absent from p are off), plus the thump at the pass peak"""
    q = {**BASE, **p}
    x = _w.render({k: v for k, v in q.items() if k not in THUMP_KEYS}, r)
    _, _, tp, _ = _w.envelope(q)
    n = len(x)
    i0 = max(0, min(n - 1, dsp.n_of(max(0.0, tp - q['thump_lead']))))
    th = _thump(q, n, i0, r)                       # drawn after the engine: the engine's noise is the same take
    if q['thump_db'] > -39.5:
        th = dsp.fade(th, 0.0, min(0.05, 0.5 * n / SR))
        lvl = np.abs(x).max() + 1e-12
        x = x + (10 ** (q['thump_db'] / 20) * lvl * th)[:, None]
    return x / (np.abs(x).max() + 1e-12) * 0.89


def recipe(cat='whoosh_big', preset_name='thump'):
    """a recipe object for forge() around one member: the full pass-engine SPACE plus the thump keys"""
    ns = types.SimpleNamespace(TARGET=TARGETS[cat], ALTS=ALTS, REFSET=REFSET)
    ns.__name__ = __name__
    ns.SPACE = SPACE
    ns.DEFAULTS = preset(preset_name, cat)
    ns.render = render
    ns.PRESET = preset_name
    return ns
