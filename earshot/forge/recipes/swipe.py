"""A soft swipe: cloth or paper slid or swept past, for UI slides and panel moves (forge v2, category 'swipe').

Every sample is computed; no recorded audio is loaded (references only judge). The pass engine is recipes/whoosh.py
(`whoosh.render`); a swipe differs from a whoosh in what moves and how:
  * A soft, wide surface moves a lot of air slowly: the low weight layer (buffet, the air shoved at the listener)
    carries most of the energy (the cloth refs put nearly all of it under 150 Hz) with a soft mid band on top.
  * Texture, not tone: fabric fibres and paper edges rub, so the air band is roughened by fast amplitude modulation
    (rustle, 50-200 Hz) and sprinkled with tiny crackles (grain layer, density rising with the speed). No whistle.
  * Slower, smoother envelopes than a whoosh: a 40-150 ms rise, an optional hold (a slide that keeps moving), a soft
    fall; wider in stereo than an object pass (a surface, not a point: the cloth refs sit at L/R correlation 0.3-0.6).
The family (PRESETS): a sleeve / cloth swoop, a paper slide (brighter, crackly, little low end), a heavy velvet
swipe (dark, slow), a quick sheet flick (flutter) and a smooth UI glide (a held slide with a gentle filter move).
"""
import types

from . import whoosh as _w

TARGET = 'a soft cloth swipe'
ALTS = ['a gust of wind', 'a broom on a floor', 'paper tearing', 'a zipper pulled']
REFSET = 'swipe'
REFS = []

SPACE = dict(_w.SPACE)
DEFAULTS = {**_w.DEFAULTS,
            'rise': 0.12, 'rise_k': 0.9, 'hold': 0.03, 'fall': 0.14, 'fall_k': 0.8, 'start_db': -34.0,
            'track': 0.7, 'f_peak': 350.0, 'up_oct': 1.2, 'down_oct': 2.0, 'dop_st': 1.0, 'dop_w': 0.05,
            'q_air': 0.9, 'lp_ratio': 3.5, 'flut_depth': 0.3, 'flut_f': 14.0, 'rustle': 0.55, 'rustle_f': 90.0,
            'low_db': 4.0, 'low_f': 110.0, 'low_smooth': 0.05, 'sub_db': -50.0,
            'hiss_db': -24.0, 'hiss_hp': 2500.0, 'hiss_open': 0.8, 'hiss_pow': 1.5,
            'tone_db': -50.0, 'grain_db': -24.0, 'grain_rate': 600.0, 'grain_f': 2600.0,
            'room_wet': 0.12, 'rt60': 0.35, 'pan': 0.15, 'width': 0.35, 'drive_db': 1.5,
            'end_db': -66.0, 'end_fade': 0.05}

# round 2 (06:40): round 1's buffet (low layer +8 dB, smoothed over 20 ms, 90 ms rise) read as a thump to CLAP
# ('a punchy impact hit', 'a kick drum') and AudioSet (Door, Slam, Knock): the low layer now swells over 50 ms under
# a slower rise (attack ~80 ms, the cloth refs take 49-134 ms), 4 dB lower, with more fabric texture on the air band
PRESETS = {
    'swipe': {
        'cloth':  {},
        'paper':  {'f_peak': 650.0, 'q_air': 1.2, 'low_db': 0.0, 'low_f': 140.0, 'rustle': 0.6, 'rustle_f': 140.0,
                   'grain_db': -16.0, 'grain_rate': 1200.0, 'grain_f': 3500.0, 'hiss_db': -20.0, 'hiss_hp': 3500.0,
                   'rise': 0.1, 'fall': 0.11, 'width': 0.25},
        'velvet': {'rise': 0.16, 'fall': 0.2, 'hold': 0.05, 'f_peak': 240.0, 'low_db': 5.0, 'low_f': 90.0,
                   'low_smooth': 0.07, 'rustle': 0.35, 'rustle_f': 60.0, 'grain_db': -32.0, 'hiss_db': -30.0,
                   'flut_depth': 0.4},
        'flick':  {'rise': 0.08, 'rise_k': 0.7, 'fall': 0.1, 'f_peak': 450.0, 'flut_depth': 0.55, 'flut_f': 32.0,
                   'low_db': 1.0, 'low_smooth': 0.04, 'grain_db': -24.0, 'grain_rate': 900.0, 'hiss_db': -22.0,
                   'pan': 0.3},
        'glide':  {'rise': 0.14, 'rise_k': 1.1, 'hold': 0.1, 'fall': 0.14, 'fall_k': 1.0, 'f_peak': 450.0,
                   'up_oct': 1.6, 'down_oct': 1.6, 'q_air': 1.0, 'rustle': 0.2, 'low_db': 2.0, 'grain_db': -36.0,
                   'hiss_db': -24.0, 'dop_st': 2.0, 'dop_w': 0.1, 'width': 0.45, 'room_wet': 0.18, 'rt60': 0.5},
    },
}
BASES = {'swipe': DEFAULTS}
KINDS = {'swipe': dict(TARGET=TARGET, ALTS=ALTS, REFSET=REFSET)}
CHARACTER = _w.CHARACTER + ('rustle', 'grain_rate')


def preset(kind, name):
    return {**BASES[kind], **PRESETS[kind][name]}


def local_space(params, space=None, keys=None, char_w=0.12, free_w=0.3):
    return _w.local_space(params, space or SPACE, keys, char_w, free_w, character=CHARACTER)


def recipe(kind='swipe', defaults=None):
    ns = types.SimpleNamespace(**KINDS[kind])
    ns.__name__ = __name__
    ns.SPACE = SPACE
    ns.DEFAULTS = dict(defaults if defaults is not None else BASES[kind])
    ns.render = render
    ns.KIND = kind
    return ns


def render(p, r):
    return _w.render({**DEFAULTS, **p}, r)
