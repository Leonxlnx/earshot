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
DEFAULTS_R2 = {**_w.DEFAULTS_R2,
               'rise': 0.12, 'rise_k': 0.9, 'hold': 0.03, 'fall': 0.14, 'fall_k': 0.8, 'start_db': -34.0,
               'track': 0.7, 'f_peak': 350.0, 'up_oct': 1.2, 'down_oct': 2.0, 'dop_st': 1.0, 'dop_w': 0.05,
               'q_air': 0.9, 'lp_ratio': 3.5, 'flut_depth': 0.3, 'flut_f': 14.0, 'rustle': 0.55, 'rustle_f': 90.0,
               'low_db': 4.0, 'low_f': 110.0, 'low_smooth': 0.05, 'sub_db': -50.0,
               'hiss_db': -24.0, 'hiss_hp': 2500.0, 'hiss_open': 0.8, 'hiss_pow': 1.5,
               'tone_db': -50.0, 'grain_db': -24.0, 'grain_rate': 600.0, 'grain_f': 2600.0,
               'room_wet': 0.12, 'rt60': 0.35, 'pan': 0.15, 'width': 0.35, 'drive_db': 1.5,
               'end_db': -66.0, 'end_fade': 0.05}
# round 3 (judge, eval/forge/results_v2r2.md): the round-2 swipes were muffled (4-12 kHz share -35.9 dB of total where
# the held-out cloth swipes sit at -19.2; onset > 6 kHz -27 vs -11.5) and their low buffet onset read as a thump
# (AudioSet Knock / Door). The forge cloth refs (one ORTF t-shirt session) are just as dark, and the licensed library
# has no brighter cloth or paper outside the bench family, so this is designed from the study and the judge's numbers:
# a fabric friction layer (the hiss top from 1.5 kHz, roughened by fast irregular AM: fibres catching and releasing)
# and denser crisp fibre crackle carry the motion, and the air shove (low band) swells 50 ms after the brush instead
# of thumping on the onset. Round 3.2: the first round-3 family still read 'a punchy impact hit' to the critic's CLAP
# (3 of 5) with the delayed swell at +2 dB over the air: the swell is now 5 dB lower and smoother (80 ms), and a -4 dB
# shelf over 9 kHz (the SPACE edge; the search box was clipped there) tilts the flat fabric top down like a rustle
DEFAULTS = {**DEFAULTS_R2,
            'rise': 0.16, 'rise_k': 1.0, 'hold': 0.03, 'fall': 0.17, 'fall_k': 0.65, 'track': 0.7,
            'f_peak': 380.0, 'q_air': 0.8, 'lp_ratio': 4.0, 'rustle': 0.5, 'rustle_f': 110.0,
            'low_db': -3.0, 'low_f': 120.0, 'low_smooth': 0.08, 'low_lag': 0.05, 'eq_lo_db': -2.0, 'eq_lo_f': 120.0,
            'hiss_db': -15.0, 'hiss_lo': 1800.0, 'hiss_hp': 3000.0, 'hiss_open': 1.2, 'hiss_pow': 1.0,
            'hiss_rustle': 0.7, 'grain_db': -18.0, 'grain_rate': 1500.0, 'grain_f': 5500.0, 'width': 0.4,
            'eq_hi_db': -4.0, 'eq_hi_f': 9000.0}
DEFAULTS_R3 = DEFAULTS
# round 4 (judge, eval/forge/results_v2r3.md): the onset was still 11 dB dull (> 6 kHz share of the first 15 ms -22.4
# vs the held-out cloth -11.6), the image narrow (L/R corr 0.90 vs 0.68) and the tail darkened hard (darken -1.96 vs
# +0.61: the fabric top died while the delayed low swell rang on). Now the brush leads and outlives the body (hiss
# envelope exponent 0.4 before / 0.5 after the peak, its low-pass kept open: hiss_mfloor 0.6), the swell follows
# closer (low_lag 0.02 s, smoothed 50 ms, 2 dB lower), a brighter room, and the surface is wide: an independent fabric
# hiss and low band as side signals (hiss_width 0.7, low_width 0.75; the mono sum is unchanged); the hiss 4-5 dB
# lower (the longer hiss envelope adds 4-12 kHz energy; members' own hiss_db moved by the same amount). Design check
# (r4/design_r4.py, 2 seeds x 5 members): onset > 6 kHz -29 -> -15 dB, corr 0.90 -> 0.6, darken -1.3 -> +0.2
DEFAULTS = {**DEFAULTS_R3, 'hiss_pre': 0.4, 'hiss_tail': 0.5, 'hiss_mfloor': 0.6, 'hiss_width': 0.7,
            'room_damp': 0.3, 'low_lag': 0.02, 'low_smooth': 0.05, 'low_db': -5.0, 'low_width': 0.75, 'hiss_db': -20.0}

# round 2 (06:40): round 1's buffet (low layer +8 dB, smoothed over 20 ms, 90 ms rise) read as a thump to CLAP
# ('a punchy impact hit', 'a kick drum') and AudioSet (Door, Slam, Knock). Round 3: the five characters rebuilt on the
# bright fabric base (round-2 members' full parameters: final_r2/report.json)
PRESETS = {
    'swipe': {
        'cloth':  {},
        'paper':  {'rise': 0.12, 'fall': 0.13, 'f_peak': 520.0, 'q_air': 1.1, 'low_db': -6.0, 'low_f': 150.0,
                   'grain_db': -16.0, 'grain_rate': 2500.0, 'grain_f': 6000.0, 'hiss_db': -19.0, 'hiss_lo': 2000.0,
                   'hiss_rustle': 0.5, 'rustle_f': 150.0, 'width': 0.3},
        # round 4: velvet stays the weakest member (CLAP 'a punchy impact hit, a heavy thud'; AudioSet 'Breaking, Door'):
        # a retry with the swell 4 dB lower, closer and smoother and more fabric (final_r4b/) read worse ('an explosion',
        # critic -6.38 vs -4.76), so this preset (the delivered one) stays
        'velvet': {'rise': 0.2, 'fall': 0.22, 'fall_k': 0.6, 'hold': 0.05, 'f_peak': 260.0, 'low_db': -1.0,
                   'low_f': 95.0, 'low_smooth': 0.07, 'low_lag': 0.06, 'rustle': 0.35, 'rustle_f': 70.0,
                   'grain_db': -24.0, 'hiss_db': -19.0, 'flut_depth': 0.4},
        'flick':  {'rise': 0.1, 'rise_k': 0.8, 'fall': 0.13, 'f_peak': 450.0, 'flut_depth': 0.55, 'flut_f': 32.0,
                   'low_db': -7.0, 'low_smooth': 0.06, 'low_lag': 0.03, 'grain_db': -18.0, 'grain_rate': 900.0,
                   'hiss_db': -17.0, 'pan': 0.3},
        'glide':  {'rise': 0.18, 'hold': 0.1, 'fall': 0.16, 'fall_k': 0.9, 'f_peak': 450.0, 'up_oct': 1.6,
                   'down_oct': 1.6, 'q_air': 1.0, 'rustle': 0.2, 'hiss_rustle': 0.35, 'low_db': -5.0, 'grain_db': -32.0,
                   'hiss_db': -19.0, 'dop_st': 2.0, 'dop_w': 0.1, 'width': 0.45, 'room_wet': 0.18, 'rt60': 0.5},
    },
}
BASES = {'swipe': DEFAULTS}
KINDS = {'swipe': dict(TARGET=TARGET, ALTS=ALTS, REFSET=REFSET)}
CHARACTER = _w.CHARACTER + ('rustle', 'grain_rate', 'hiss_rustle', 'low_lag')


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
    """the pass engine on DEFAULTS + p, except that whoosh.OPTIONAL keys absent from p are OFF (round 4 fix: merging
    today's DEFAULTS let each round's new optional keys leak into older parameter sets, so the round-2 / round-3 finals
    and seed-bank entries no longer re-rendered; full parameter sets, as forge() and preset() pass, are unchanged)"""
    return _w.render({**DEFAULTS, **_w.OPTIONAL, **p}, r)
