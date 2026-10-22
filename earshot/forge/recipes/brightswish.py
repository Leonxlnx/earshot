"""Bright swishes and the designed bright whip: high, wide air flicks for text and pane moves (forge categories
'swish' and 'whoosh_fast', members on the pass engine of recipes/whoosh.py).

What the reference clips do (a private licensed set and its category notes; the clips
are local study material and are not shipped):
  * swish (6 events in 5 films): short and BRIGHT, 0.25 s (0.21-0.29), centroid 2.1 kHz (0.8-3.6), its loudest point
    134 ms (115-150) after the first moment within 20 dB of it (it rises, then stops), fairly wide (L/R correlation
    0.76, 0.43-0.94; one of the clean ones -0.04 at 4.4 kHz). earshot 0.3's swishes sat at ~600 Hz, 2-5x darker;
  * the effects-led film's whip: 0.74 s, 2.7 kHz, decorrelated (correlation 0.07), hits in ~15 ms and falls ~45
    semitones while it rings out (-20 dB 0.45 s after the peak), a comb-like fan of falling partials and a granular
    flutter through it. It FALLS: the rising synth whip is the cheap version this recipe never makes (the audit's
    rising-glide check stays silent on every member).
How it is built: the pass engine (`whoosh.render`) with the air band high (f_peak 2-5 kHz), no low weight or sub, the
hiss as loud as the air body, a decorrelated hiss side (hiss_width) and air side (width); the whip adds a fast rise,
a band that sinks down_oct after the peak (track > 1: it falls with the level) and a deep fast flutter. One optional
layer of its own: `comb_mix` runs the pass through a feedback comb whose fundamental falls comb_f0 -> comb_f1 over the
file (a fan of falling partials, after impact._vcomb_core), per channel. The whoosh knobs (dur 'shape', apex) apply.
Every sample is computed; no recorded audio is used.
"""
import types

import numpy as np

from .. import dsp
from . import whoosh as _w
from .impact import _vcomb_core

SR = dsp.SR
TARGET = 'a bright quick swish'
ALTS = ['a short burst of steam', 'a page being flipped', 'a sword being sharpened', 'rain']
REFSET = 'swish'

COMB_SPACE = {
    'comb_mix': (0.0, 1.0),
    'comb_f0': (150.0, 4000.0, 'log'),  # the comb's fundamental at the first sample (Hz)
    'comb_f1': (40.0, 1500.0, 'log'),   # and at the last
    'comb_fb': (0.0, 0.92),
}
COMB_KEYS = tuple(COMB_SPACE)
COMB_OFF = {'comb_mix': 0.0, 'comb_f0': 1000.0, 'comb_f1': 200.0, 'comb_fb': 0.5}
SPACE = {**_w.SPACE, **COMB_SPACE}

SWISH = {**_w.FAST, **_w.OPTIONAL, **COMB_OFF,
         'rise': 0.13, 'rise_k': 0.8, 'hold': 0.0, 'fall': 0.07, 'fall_k': 1.1, 'start_db': -32.0, 'track': 0.7,
         'f_peak': 2300.0, 'up_oct': 1.0, 'down_oct': 1.3, 'dop_st': 3.0, 'dop_w': 0.03, 'q_air': 1.4,
         'lp_ratio': 2.6, 'flut_depth': 0.2, 'flut_f': 30.0, 'low_db': -40.0, 'sub_db': -50.0, 'hiss_db': -6.0,
         'hiss_hp': 3000.0, 'hiss_open': 1.2, 'hiss_pow': 1.3, 'hiss_tail': None, 'hiss_pre': None,
         'hiss_mfloor': 0.0, 'tail_sweep': 0.0, 'hiss_q': 0.8, 'tone_db': -50.0, 'grain_db': -50.0,
         'room_wet': 0.08, 'rt60': 0.35, 'room_damp': 0.3, 'pan': 0.15, 'width': 0.35, 'hiss_width': 0.5,
         'drive_db': 1.0, 'end_db': -62.0, 'end_fade': 0.03, 'eq_lo_db': -12.0, 'eq_lo_f': 300.0,
         'eq_hi_db': 0.0, 'eq_hi_f': 6000.0}
WHIP = {**SWISH,
        'rise': 0.02, 'rise_k': 1.0, 'hold': 0.0, 'fall': 0.5, 'fall_k': 0.75, 'start_db': -30.0, 'track': 1.2,
        'f_peak': 4200.0, 'up_oct': 0.5, 'down_oct': 3.5, 'dop_st': 6.0, 'dop_w': 0.05, 'q_air': 2.5,
        'lp_ratio': 2.2, 'flut_depth': 0.6, 'flut_f': 32.0, 'hiss_db': -12.0, 'hiss_hp': 2500.0, 'hiss_tail': 1.0,
        'room_wet': 0.15, 'rt60': 0.6, 'width': 0.6, 'hiss_width': 1.0, 'pan': 0.1, 'end_db': -60.0,
        'end_fade': 0.06, 'comb_mix': 0.5, 'comb_f0': 2200.0, 'comb_f1': 160.0, 'comb_fb': 0.7}
BASES = {'swish': SWISH, 'whoosh_fast': WHIP}
FAMILIES = {
    # earshot 0.4 phase B: hand-designed corners of SWISH / WHIP, then tuned toward each style on the measured
    # features (the music films' swish profile for bright; the study's dev-split clips of the flick and the designed
    # whip) and, for flick and whip, on CLAP only up to the grade bar
    'swish': [
        # the music films' bright swish: ~2.4 kHz, a 100 ms rise, a quick fall, fairly wide
        ('bright', {'rise_k': 0.9287, 'hold': 0.05657, 'fall_k': 1.127, 'track': 0.6361, 'up_oct': 1.004,
                    'down_oct': 2.161, 'q_air': 2.813, 'flut_depth': 0.3016, 'flut_f': 33.74, 'hiss_db': -3.588,
                    'hiss_hp': 2054.0, 'hiss_open': 1.037, 'hiss_pow': 1.567, 'grain_rate': 2789.0,
                    'grain_f': 2753.0, 'room_wet': 0.0, 'rt60': 0.4884, 'pan': 0.1457, 'width': 0.4008,
                    'eq_lo_db': -10.06, 'eq_hi_db': -12.75, 'hiss_width': 0.7457, 'comb_f0': 327.0,
                    'comb_f1': 66.57, 'comb_fb': 0.5578}),
        # a flick that rises and stops dead: higher (~4 kHz), fluttering, decorrelated, a touch of falling comb
        ('flick', {'rise': 0.09435, 'rise_k': 0.45, 'fall': 0.03, 'fall_k': 1.335, 'track': 0.7737, 'f_peak': 800.3,
                   'up_oct': 1.338, 'down_oct': 3.046, 'dop_st': 2.201, 'q_air': 0.6, 'lp_ratio': 2.779,
                   'flut_depth': 0.524, 'flut_f': 44.69, 'hiss_db': -2.058, 'hiss_hp': 3720.0, 'hiss_open': 0.7442,
                   'hiss_pow': 1.669, 'grain_rate': 381.1, 'grain_f': 3588.0, 'room_wet': 0.06281, 'rt60': 0.307,
                   'pan': 0.05213, 'width': 0.5124, 'end_fade': 0.02, 'eq_lo_db': -12.61, 'eq_hi_db': 0.8779,
                   'hiss_width': 1.0, 'comb_mix': 0.3188, 'comb_f0': 4000.0, 'comb_f1': 99.3, 'comb_fb': 0.4513}),
    ],
    'whoosh_fast': [
        # the designed bright whip: a 20 ms hit, a band and a ringing comb falling ~45 semitones, wide
        ('whip', {'rise': 0.02, 'rise_k': 0.7391, 'hold': 0.05657, 'fall_k': 1.127, 'track': 1.071, 'f_peak': 4000.0,
                  'up_oct': 0.2647, 'dop_st': 10.95, 'q_air': 3.505, 'lp_ratio': 1.2, 'flut_f': 25.21, 'hiss_db': 0.0,
                  'hiss_hp': 1807.0, 'hiss_open': 0.0, 'hiss_pow': 1.51, 'grain_db': -46.8, 'grain_rate': 50.0,
                  'grain_f': 5843.0, 'room_wet': 0.03593, 'rt60': 0.6411, 'pan': 0.2634, 'eq_lo_db': -15.23,
                  'eq_hi_db': 1.169, 'comb_mix': 1.0, 'comb_f0': 1464.0, 'comb_f1': 40.0, 'comb_fb': 0.7323}),
    ],
}
DEFAULTS = dict(SWISH)
TARGETS = {'swish': TARGET, 'whoosh_fast': 'a bright whip whoosh'}


def preset(cat, name):
    return {**BASES[cat], **dict(FAMILIES[cat])[name]}


def preset_full(cat, params):
    return {**BASES[cat], **params}


def _comb(y, p):
    """a feedback comb whose fundamental falls comb_f0 -> comb_f1 over the file, mixed in parallel, per channel"""
    if p['comb_mix'] < 0.01:
        return y
    n = len(y)
    u = np.arange(n) / max(1, n - 1)
    d = SR / (p['comb_f0'] * (p['comb_f1'] / p['comb_f0']) ** u)
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        x = np.ascontiguousarray(y[:, c])
        z = _vcomb_core(x, np.ascontiguousarray(d), p['comb_fb']) * (1 - p['comb_fb'])
        out[:, c] = x + p['comb_mix'] * (z - x)
    return out


def render(p, r):
    """the pass engine on SWISH + p (whoosh.OPTIONAL keys absent from p are off), then the falling comb"""
    q = {**SWISH, **p}
    y = _w.render({k: v for k, v in q.items() if k not in COMB_KEYS}, r)
    y = _comb(y, q)
    y = dsp.fade(y, 0.0, min(q['end_fade'], 0.3 * len(y) / SR))
    return y / (np.abs(y).max() + 1e-12) * 0.89


def recipe(cat='swish', name=None):
    ns = types.SimpleNamespace(TARGET=TARGETS[cat], ALTS=ALTS, REFSET=cat)
    ns.__name__ = __name__
    ns.SPACE = SPACE
    ns.DEFAULTS = preset(cat, name or FAMILIES[cat][0][0])
    ns.render = render
    return ns
