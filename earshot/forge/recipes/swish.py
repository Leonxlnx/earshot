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
DEFAULTS_R2 = {**_w.DEFAULTS_R2,
               'rise': 0.045, 'rise_k': 0.8, 'hold': 0.0, 'fall': 0.08, 'fall_k': 0.7, 'start_db': -34.0,
               'track': 0.6, 'f_peak': 520.0, 'up_oct': 1.4, 'down_oct': 1.8, 'dop_st': 4.0, 'dop_w': 0.02,
               'q_air': 1.8, 'lp_ratio': 2.2, 'flut_depth': 0.15, 'flut_f': 30.0,
               'low_db': -16.0, 'low_f': 180.0, 'low_smooth': 0.02, 'sub_db': -50.0,
               'hiss_db': -14.0, 'hiss_hp': 3800.0, 'hiss_open': 1.2, 'hiss_pow': 1.8,
               'tone_db': -16.0, 'tone_q': 28.0, 'tone_ratio': 1.0, 'tone_pure': 0.25,
               'grain_db': -50.0, 'room_wet': 0.15, 'rt60': 0.35, 'pan': 0.2, 'width': 0.1, 'drive_db': 1.5,
               'end_db': -66.0, 'end_fade': 0.03}
# round 3 (judge, eval/forge/results_v2r2.md): the round-2 swishes rang out in half the held-out tail (t40 ~150 ms),
# carried 15 dB too much 4-12 kHz hiss and ~8 dB too much under 150 Hz, and AudioSet heard 'Whack, thwack' / 'Knock'
# (a hit-like onset). Now: a rounder peak (rise 60 ms at rise_k 1 with a 4 ms hold), a fast drop into a long concave
# tail (fall 0.14 s at fall_k 0.58) and a longer room, the low band thinner and higher (-24 dB at 220 Hz, a -10 dB shelf
# under 150 Hz), the hiss 10 dB lower, from 2.4 kHz and following the level curve (hiss_pow 1.2: no 5-15 kHz streak
# at the peak, no 2-4 kHz hole under it), a -4 dB shelf over 6 kHz
DEFAULTS = {**DEFAULTS_R2,
            'rise': 0.07, 'rise_k': 1.0, 'hold': 0.004, 'fall': 0.14, 'fall_k': 0.58, 'lp_ratio': 3.0,
            'low_db': -24.0, 'low_f': 220.0, 'hiss_db': -24.0, 'hiss_hp': 2400.0, 'hiss_pow': 1.2,
            'room_wet': 0.22, 'rt60': 0.5, 'end_fade': 0.05,
            'eq_lo_db': -10.0, 'eq_lo_f': 150.0, 'eq_hi_db': -4.0, 'eq_hi_f': 6000.0}
DEFAULTS_R3 = DEFAULTS
# round 4 (judge, eval/forge/results_v2r3.md: AudioSet Door / Slam / Whack first on 4 of 5, tail darken -0.73 vs the
# forge refs' -0.16): the same cause and fix as whoosh_fast (recipes/whoosh.py FAST, out/forge_v2/swish/LOG.md round 4):
# the tail was a dark room ring of the peak's mid band (tail 4-12 kHz -35.1 dB vs forge refs -24.1, 0.5-1 kHz -5.8 vs
# -16.7). Now the wake carries the tail: source fall 0.21 s, hiss decaying slower than the body after the peak with its
# top kept open, a smaller brighter room, and a hiss that leads a little on the approach (onset > 6 kHz -38 -> ~-28 dB,
# forge refs -24.4)
DEFAULTS = {**DEFAULTS_R3, 'fall': 0.21, 'room_wet': 0.1, 'room_damp': 0.3, 'end_db': -62.0,
            'hiss_tail': 0.85, 'hiss_mfloor': 0.3, 'hiss_pre': 0.8}

# round 2 (06:30): rounds 1's short rises (16-30 ms) read as hits to CLAP ('a wooden block hit', 'a punchy impact');
# the refs rise in 25-87 ms (median 34 ms -20 dB -> peak) and keep a long low tail (t40 109-286 ms). Round 3: the
# same five characters on the round-3 base (the round-2 members' full parameters are in final_r2/report.json)
PRESETS = {
    'swish': {
        # round 3 final (final/ = the first round-3 run, r3a): five characters on the round-3 base. Round 3.2 tried six
        # texture-owning members (an aeolian cane, a gritty blade with 6 kHz grains, a fluttering rod at 16 Hz, a pure
        # zing, a fabric swish, a 60 ms flick; probe at the presets 0.853), but after the search the finals sat at CLAP
        # cos 0.918 (r3a 0.908) with a shorter tail (t40 198 vs 230 ms) and more hiss (4-12k -25.8 vs -29.1 dB), so
        # r3a stays final (final_r3b/ holds 3.2; out/forge_v2/swish/LOG.md round 3)
        'cane':   {},
        'blade':  {'rise': 0.055, 'fall': 0.085, 'fall_k': 0.8, 'f_peak': 900.0, 'q_air': 2.8, 'tone_db': -22.0,
                   'tone_ratio': 1.3, 'hiss_db': -19.0, 'hiss_hp': 3200.0, 'low_db': -32.0, 'dop_st': 5.0,
                   'room_wet': 0.1, 'rt60': 0.3},
        'stick':  {'rise': 0.08, 'fall': 0.17, 'fall_k': 0.55, 'end_db': -60.0, 'f_peak': 380.0, 'q_air': 1.3,
                   'low_db': -16.0, 'low_f': 180.0, 'flut_depth': 0.5, 'flut_f': 9.0, 'tone_db': -30.0,
                   'hiss_db': -27.0, 'room_wet': 0.14, 'rt60': 0.5},
        'zing':   {'rise': 0.06, 'rise_k': 0.8, 'fall': 0.13, 'f_peak': 700.0, 'tone_db': -3.0, 'tone_q': 50.0,
                   'tone_ratio': 1.2, 'tone_pure': 0.9, 'dop_st': 9.0, 'dop_w': 0.015, 'hiss_db': -25.0},
        'soft':   {'rise': 0.08, 'f_peak': 560.0, 'q_air': 1.0, 'tone_db': -40.0, 'rustle': 0.6, 'rustle_f': 120.0,
                   'hiss_rustle': 0.6, 'hiss_lo': 1500.0, 'hiss_db': -21.0, 'grain_db': -26.0, 'grain_rate': 900.0,
                   'grain_f': 3500.0, 'width': 0.35, 'room_wet': 0.14, 'rt60': 0.5, 'pan': 0.3},
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
    """the pass engine on DEFAULTS + p, except that whoosh.OPTIONAL keys absent from p are OFF (round 4 fix: merging
    today's DEFAULTS let each round's new optional keys leak into older parameter sets, so the round-2 / round-3 finals
    and seed-bank entries no longer re-rendered; full parameter sets, as forge() and preset() pass, are unchanged)"""
    return _w.render({**DEFAULTS, **_w.OPTIONAL, **p}, r)
