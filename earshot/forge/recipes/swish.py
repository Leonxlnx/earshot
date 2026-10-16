"""A thin fast swish: a cane, rod, blade or thrown object cutting the air (forge v2, category 'swish').

Every sample is computed; no recorded audio is loaded (references only judge). The pass engine is recipes/whoosh.py
(`whoosh.render`: shared level curve, moving air bands, hiss top, Doppler, tail, edit). What makes a swish a swish
rather than a whoosh (docs/motion-sfx-techniques.md section 1):
  * Short and steep: 80-300 ms around the pass, a fast rise (rise 15-45 ms) and a quick fall.
  * Thin: a small object moves little air, so the low weight layer is low or gone and the air band sits higher
    (f_peak 500-2500 Hz) with a narrower Q.
  * The aeolian whistle: a thin rod sheds a near-pure tone at f = 0.2 U / d (Strouhal; 30 m/s past a 1 cm rod is about
    600 Hz), heard as a narrow band (Q 15-40) that glides with the speed and the Doppler shift (tone layer).
The family (PRESETS) spans the character: a cane (whistle), a blade (bright, steep), a thrown stick (lower, rounder,
longer), a zing (a strong Doppler-gliding whistle) and a soft swish (dark, airy).
"""
import types

from . import whoosh as _w

TARGET = 'a thin quick swish of a cane through the air'
ALTS = ['a sword being sharpened', 'a short burst of steam', 'a page being flipped', 'a bird wing flap']
REFSET = 'swish'
REFS = []

SPACE = dict(_w.SPACE)
DEFAULTS = {**_w.DEFAULTS,
            'rise': 0.045, 'rise_k': 0.8, 'hold': 0.0, 'fall': 0.08, 'fall_k': 0.7, 'start_db': -34.0,
            'track': 0.6, 'f_peak': 520.0, 'up_oct': 1.4, 'down_oct': 1.8, 'dop_st': 4.0, 'dop_w': 0.02,
            'q_air': 1.8, 'lp_ratio': 2.2, 'flut_depth': 0.15, 'flut_f': 30.0,
            'low_db': -16.0, 'low_f': 180.0, 'low_smooth': 0.02, 'sub_db': -50.0,
            'hiss_db': -14.0, 'hiss_hp': 3800.0, 'hiss_open': 1.2, 'hiss_pow': 1.8,
            'tone_db': -16.0, 'tone_q': 28.0, 'tone_ratio': 1.0, 'tone_pure': 0.25,
            'grain_db': -50.0, 'room_wet': 0.15, 'rt60': 0.35, 'pan': 0.2, 'width': 0.1, 'drive_db': 1.5,
            'end_db': -66.0, 'end_fade': 0.03}

# round 2 (06:30): rounds 1's short rises (16-30 ms) read as hits to CLAP ('a wooden block hit', 'a punchy impact');
# the refs rise in 25-87 ms (median 34 ms -20 dB -> peak) and keep a long low tail (t40 109-286 ms)
PRESETS = {
    'swish': {
        'cane':   {},
        'blade':  {'rise': 0.035, 'fall': 0.06, 'f_peak': 1100.0, 'q_air': 2.6, 'tone_db': -20.0, 'tone_ratio': 1.3,
                   'hiss_db': -10.0, 'hiss_hp': 4500.0, 'low_db': -26.0, 'dop_st': 5.0},
        'stick':  {'rise': 0.05, 'fall': 0.1, 'f_peak': 430.0, 'q_air': 1.5, 'low_db': -10.0, 'low_f': 150.0,
                   'tone_db': -26.0, 'hiss_db': -18.0, 'room_wet': 0.16, 'rt60': 0.4},
        'zing':   {'rise': 0.035, 'rise_k': 0.6, 'fall': 0.07, 'f_peak': 800.0, 'tone_db': -11.0, 'tone_q': 40.0,
                   'tone_ratio': 1.2, 'tone_pure': 0.5, 'dop_st': 7.0, 'dop_w': 0.015, 'hiss_db': -14.0},
        'soft':   {'rise': 0.055, 'rise_k': 1.0, 'fall': 0.09, 'f_peak': 600.0, 'q_air': 1.1, 'tone_db': -36.0,
                   'hiss_db': -15.0, 'hiss_hp': 2600.0, 'width': 0.3, 'room_wet': 0.2, 'rt60': 0.5, 'pan': 0.3},
    },
}
BASES = {'swish': DEFAULTS}
KINDS = {'swish': dict(TARGET=TARGET, ALTS=ALTS, REFSET=REFSET)}
CHARACTER = _w.CHARACTER


def preset(kind, name):
    return {**BASES[kind], **PRESETS[kind][name]}


def local_space(params, space=None, keys=None, char_w=0.12, free_w=0.3):
    return _w.local_space(params, space or SPACE, keys, char_w, free_w, character=CHARACTER)


def recipe(kind='swish', defaults=None):
    ns = types.SimpleNamespace(**KINDS[kind])
    ns.__name__ = __name__
    ns.SPACE = SPACE
    ns.DEFAULTS = dict(defaults if defaults is not None else BASES[kind])
    ns.render = render
    ns.KIND = kind
    return ns


def render(p, r):
    return _w.render({**DEFAULTS, **p}, r)
