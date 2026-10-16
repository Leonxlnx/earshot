"""A designed element pop-in / soft tap for motion-design launch videos: a low thump with a thin tick on top.

The pop-in is the most used effect in the SFX-only reference film (docs/motion-sfx-study.md, 'Element pop-in'): one
per element appearing, fired in runs every ~200 ms. Measured: a thump body at 80-150 Hz (ridge ~105 Hz; under 150 Hz
holds all but 0.5 dB of the energy; tilt -4 to -6 dB/oct) plus a short noise tick (1-10 kHz, ~30 ms, 30-40 dB under
the body), often a quick double hit (two bodies 60-70 ms apart); 0.14-0.27 s, attack 6-19 ms (median 16), t20 16-35 ms,
t40 30-80 ms, convex (holds, then stops) in 7 of 10; moderately wide (side 3.5-9 dB under mid, correlation 0.38-0.78).
The pro tap is the same idiom, shorter: a 150 Hz thump with a faint broadband click, 0.1 s, attack 11 ms, t20 18 ms,
t40 27 ms, tilt -6.5 dB/oct, side 5 dB under mid. Real finger taps on glass / a hard surface (the 'tap' refset) are
the same low thump (centroid 66-190 Hz) with a sharper attack (1-6 ms) and a longer fall (t40 50-120 ms).

Layers (each owns a band, all peak together):
  1. body: a sine at body_f whose pitch falls from body_f * (1 + drop) (the 'boop' of a soft object landing), a
     raised-cosine attack (att_ms), a hold (hold_ms) and an exponential release (rel_ms): the convex 'holds, then
     stops' shape; a little 2nd harmonic (harm) so it survives laptop / phone speakers
  2. tick: a short band-passed noise burst (tick_f, 1-10 kHz) tick_db under the body; optionally a glassy partial
     (glass, glass_f) for a fingertip on glass
  3. air: a faint broadband texture (1-14 kHz noise shaped by the body's own envelope, air_db under it): the pro tap
     and pop-in spectrograms show it over the whole body, not only on the onset
  4. double: a second, smaller body dbl_dt later (the pro double hit)
  5. width: the body's channels offset by a phase-based Haas delay (width / (6 body_f): <= 72 deg at the body, L/R
     correlation >= ~0.3, so the mono fold-down keeps the thump; a fixed 2 ms offset gave taps a NEGATIVE correlation in
     round 1) and the tick decorrelated, then an end gate at `length`; real taps (the 'tap' refs) are mono-ish and add
     a hollow surface mode (knock, knock_f: a desk, a phone case)
CHARACTER parameters: body_f (pitch), att_ms (softness), hold_ms / rel_ms (length), drop (boop), tick_db / tick_f
(how much click), dbl (double hit), width, knock. FAMILIES below are the palettes for pro_pop_in, pro_tap and tap.
Every sample is computed; no recorded audio is used.
"""
import numpy as np

from .. import dsp, fx
from .click import burst, centre, gate, narrow, partials, thump

TARGET = 'a soft round low pop as an icon appears on screen'
ALTS = ['a door slam', 'a balloon popping', 'a wooden knock', 'a cork popping']   # not 'a kick drum': CLAP hears the
# real pro pop-ins as kick drums too (forge round 1), so it was penalising the right sound
TARGETS = {'pro_pop_in': 'a soft round low pop as an icon appears on screen',
           'pro_tap': 'a soft low thump of a finger touching a screen',
           'tap': 'a soft low thump of a finger touching a screen'}
REFSET = 'pro_pop_in'

SPACE = {
    'body_f': (70.0, 220.0, 'log'),
    'drop': (0.0, 1.0),                 # start pitch = body_f * (1 + drop)
    'att_ms': (1.0, 22.0, 'log'),
    'hold_ms': (0.5, 45.0, 'log'),
    'sag_db': (0.0, 12.0),              # level lost during the hold (the pro gate shape: holds, sags, then stops)
    'rel_ms': (3.0, 40.0, 'log'),
    'harm': (0.0, 0.35),
    'sub': (0.0, 0.7),                  # an octave-down partner sine (63 Hz weight under a 125 Hz body)
    'sat': (0.0, 8.0),                  # tanh drive dB on the body (harmonics)
    'tick_db': (-34.0, -2.0),           # tick peak level re the body peak (a short burst: its energy is ~20 dB lower)
    'tick_f': (1200.0, 8000.0, 'log'),  # tick band centre
    'tick_bw': (1.0, 3.0),              # tick band half-width in octaves
    'tick_ms': (2.0, 40.0, 'log'),
    'glass': (0.0, 1.0),                # glassy partial share of the tick
    'air_db': (-60.0, -24.0),           # a faint 1-14 kHz texture riding the body envelope, dB re the body peak
    'air_tilt': (-1.0, 1.0),            # its tilt (-1 dark .. +1 bright)
    'glass_f': (1800.0, 7000.0, 'log'),
    'dbl': (0.0, 0.9),                  # second body level (0 = single hit)
    'dbl_dt': (0.05, 0.08),
    'dbl_ratio': (0.85, 1.25),          # second body pitch ratio
    'width': (0.0, 1.2),                # body Haas offset as a phase at body_f: delay = width / (6 body_f), i.e. up to
                                        # 72 deg (L/R correlation >= ~0.3 at the body; mono fold-down keeps the body)
    'knock': (0.0, 0.5),                # hollow surface mode (a desk, a phone case: real taps), level re the body
    'knock_f': (280.0, 900.0, 'log'),
    'tick_width': (0.0, 1.0),
    'hp': (35.0, 70.0, 'log'),
    'lp': (2500.0, 16000.0, 'log'),     # top roll-off of the whole sound
    'length': (0.08, 0.3),
    'release_ms': (6.0, 40.0, 'log'),
    'contact_db': (-60.0, 6.0),         # round 3: the fingertip / nail contact, a short bright burst ON the onset, dB re
    'contact_ms': (0.2, 3.0, 'log'),    # the body peak (-60 = none: the round-2 sounds): the real taps fall 20 dB in
    'contact_f': (800.0, 6000.0, 'log'),  # ~7 ms from a 1 ms attack (the contact), then ring on quietly (the body)
    'ring_db': (-60.0, -10.0),          # round 3: a second, quiet, long body under the short one (the two-stage decay
    'ring_ms': (10.0, 90.0, 'log'),     # of real taps: -20..-25 dB in ~10 ms, then a ring); -60 = none
    'contact_lp_x': (0.0, 2.5),         # round 4: 0 = off; else a 4th-order low-pass on the contact at contact_f x this
                                        # (the round-3 contact's 2nd-order skirt and 0.2 ms onset WERE the taps' 4-12 kHz:
                                        # muting it moved 4-12 kHz -18.4 -> -36.9 dB; so keep the contact, darken its skirt)
}
DEFAULTS = {'body_f': 105.0, 'drop': 0.35, 'att_ms': 12.0, 'hold_ms': 6.0, 'sag_db': 0.0, 'rel_ms': 8.0, 'harm': 0.12,
            'sub': 0.2, 'sat': 2.0, 'tick_db': -12.0, 'tick_f': 3500.0, 'tick_bw': 2.5, 'tick_ms': 12.0, 'glass': 0.0,
            'air_db': -60.0, 'air_tilt': 0.0, 'glass_f': 3200.0, 'dbl': 0.0, 'dbl_dt': 0.065, 'dbl_ratio': 1.0,
            'width': 0.6, 'knock': 0.0, 'knock_f': 500.0, 'tick_width': 0.6, 'hp': 45.0, 'lp': 9000.0, 'length': 0.18,
            'release_ms': 15.0, 'contact_db': -60.0, 'contact_ms': 0.8, 'contact_f': 2500.0, 'ring_db': -60.0,
            'ring_ms': 40.0, 'contact_lp_x': 0.0}

FAMILIES = {      # hold + 2.3 rel ~ t20, hold + 4.6 rel ~ t40 (ms); measured attack ~ 0.8 att_ms
    'pro_pop_in': {   # the pro shape: attack 6-19 ms, a sagging hold, a fast stop (t20 10-18, t40 25-105 ms in the forge
        # refs). Round 3: the round-2 palette were near clones to CLAP (variant cos 0.895 vs the refs' 0.696; it hears
        # every low thump as one kick drum), so the families now differ where the refs differ: 150 Hz-1 kHz -5..-23 dB
        # (harmonics: 'knock'), 4-12 kHz -17..-35 (texture / tick: 'airy'), a darker 'deep', a RISING double hit
        'round': {'width': (0.6, 1.1), 'knock': 0.0, 'body_f': (95, 120, 'log'), 'att_ms': (12, 20, 'log'),
                  'hold_ms': (5, 14, 'log'), 'sag_db': (4, 10), 'rel_ms': (3, 7, 'log'), 'glass': 0.0,
                  'air_db': (-50, -38), 'dbl': 0.0, 'length': (0.12, 0.18)},
        'double': {'width': (0.6, 1.1), 'knock': 0.0, 'body_f': (95, 125, 'log'), 'att_ms': (6, 14, 'log'),
                   'hold_ms': (4, 12, 'log'), 'sag_db': (3, 9), 'rel_ms': (3, 7, 'log'), 'glass': 0.0,
                   'air_db': (-50, -38), 'dbl': (0.5, 0.85), 'dbl_dt': (0.06, 0.07), 'dbl_ratio': (1.1, 1.25),
                   'length': (0.16, 0.22)},
        'knock': {'width': (0.5, 0.9), 'knock': 0.0, 'body_f': (115, 135, 'log'), 'harm': (0.25, 0.35),
                  'sat': (4, 8), 'att_ms': (5, 10, 'log'), 'hold_ms': (4, 10, 'log'), 'sag_db': (3, 8),
                  'rel_ms': (3, 6, 'log'), 'glass': 0.0, 'tick_db': (-10, -4), 'tick_f': (1500, 3000, 'log'),
                  'contact_db': (-16, -8), 'contact_ms': (0.5, 1.5, 'log'), 'contact_f': (1500, 3000, 'log'),
                  'air_db': (-48, -38), 'dbl': 0.0, 'length': (0.1, 0.15)},
        'deep': {'width': (0.7, 1.1), 'knock': 0.0, 'body_f': (75, 92, 'log'), 'att_ms': (14, 22, 'log'),
                 'hold_ms': (8, 20, 'log'), 'sag_db': (5, 12), 'rel_ms': (4, 9, 'log'), 'sub': (0.3, 0.7),
                 'harm': (0.0, 0.08), 'sat': (0, 2), 'glass': 0.0, 'air_db': (-58, -46), 'dbl': 0.0,
                 'tick_db': (-30, -20), 'lp': (4000, 7000, 'log'), 'length': (0.14, 0.22)},
        'airy': {'width': (0.6, 1.1), 'knock': 0.0, 'body_f': (100, 130, 'log'), 'att_ms': (8, 16, 'log'),
                 'hold_ms': (5, 14, 'log'), 'sag_db': (3, 9), 'rel_ms': (3, 7, 'log'), 'glass': 0.0,
                 'air_db': (-32, -24), 'air_tilt': (0.3, 1.0), 'tick_db': (-12, -6), 'tick_f': (4000, 8000, 'log'),
                 'tick_ms': (15, 35, 'log'), 'tick_width': (0.8, 1.0), 'lp': (12000, 16000, 'log'), 'dbl': 0.0,
                 'length': (0.12, 0.18)},
    },
    'pro_tap': {   # the pro tap (ref_2): 150 Hz thump, attack ~11 ms, a faint broadband texture over the body, gated.
        # Round 3: brighter contact layers so it reads as a tap (the forge ref has > 12 kHz at -37 dB of total where the
        # round-2 taps had -58): a contact burst on the onset in every family, more air texture; holds ~30 % shorter
        # (r3 first run: t20 18 / t40 35 ms vs the ref's 11 / 25).
        # Round 4 (the round-3 judge: more recognisable as a fingertip tap; CLAP heard 'a punchy kick drum pattern' first
        # on all five): the kick-drum cue is the body's pitch drop and the octave-down sub, so drop <= 0.15 and sub <= 0.2
        # (the ref's ridge holds at 130-160 Hz), and the contact a few dB up (the fingertip)
        'finger': {'drop': (0.0, 0.15), 'sub': (0.0, 0.2), 'width': (0.7, 1.1), 'knock': 0.0, 'glass': 0.0, 'body_f': (135, 165, 'log'),
                   'att_ms': (8, 13, 'log'), 'hold_ms': (5, 12, 'log'), 'sag_db': (3, 9), 'rel_ms': (2.5, 5, 'log'),
                   'air_db': (-40, -30), 'air_tilt': (0.0, 0.8), 'dbl': 0.0, 'contact_db': (-11, -3),
                   'contact_ms': (0.4, 1.2, 'log'), 'contact_f': (1500, 3500, 'log'), 'lp': (9000, 16000, 'log'),
                   'length': (0.07, 0.1)},
        'glass': {'drop': (0.0, 0.15), 'sub': (0.0, 0.2), 'width': (0.7, 1.1), 'knock': 0.0, 'body_f': (140, 170, 'log'), 'att_ms': (6, 11, 'log'),
                  'glass': (0.4, 0.8), 'glass_f': (2500, 5000, 'log'), 'tick_db': (-14, -6),
                  'hold_ms': (4, 11, 'log'), 'sag_db': (3, 9), 'rel_ms': (2.5, 5, 'log'), 'air_db': (-44, -32),
                  'air_tilt': (0.0, 1.0), 'dbl': 0.0, 'contact_db': (-9, -1), 'contact_ms': (0.3, 0.8, 'log'),
                  'contact_f': (3000, 6000, 'log'), 'lp': (11000, 16000, 'log'), 'length': (0.07, 0.1)},
        'soft': {'drop': (0.0, 0.15), 'sub': (0.0, 0.2), 'width': (0.7, 1.1), 'knock': 0.0, 'glass': 0.0, 'body_f': (110, 140, 'log'),
                 'att_ms': (10, 18, 'log'), 'tick_db': (-24, -14), 'hold_ms': (7, 15, 'log'), 'sag_db': (4, 10),
                 'rel_ms': (3, 6, 'log'), 'air_db': (-46, -36), 'dbl': 0.0, 'contact_db': (-17, -9),
                 'contact_ms': (0.8, 2.0, 'log'), 'contact_f': (1200, 2500, 'log'), 'length': (0.08, 0.11)},
        'knock': {'drop': (0.0, 0.15), 'sub': (0.0, 0.2), 'width': (0.7, 1.1), 'knock': 0.0, 'glass': 0.0, 'body_f': (150, 210, 'log'),
                  'att_ms': (3, 7, 'log'), 'tick_db': (-10, -2), 'hold_ms': (3, 9, 'log'), 'sag_db': (3, 9),
                  'rel_ms': (2.5, 5, 'log'), 'air_db': (-38, -28), 'air_tilt': (0.0, 1.0), 'dbl': 0.0,
                  'contact_db': (-5, 3), 'contact_ms': (0.4, 1.0, 'log'), 'contact_f': (2000, 4500, 'log'),
                  'lp': (11000, 16000, 'log'), 'length': (0.06, 0.09)},
        'double': {'drop': (0.0, 0.15), 'sub': (0.0, 0.2), 'width': (0.7, 1.1), 'knock': 0.0, 'glass': 0.0, 'body_f': (130, 165, 'log'),
                   'att_ms': (6, 12, 'log'), 'dbl': (0.3, 0.6), 'dbl_dt': (0.05, 0.065), 'hold_ms': (4, 10, 'log'),
                   'sag_db': (3, 9), 'rel_ms': (2.5, 5, 'log'), 'air_db': (-42, -32), 'contact_db': (-11, -3),
                   'contact_ms': (0.4, 1.2, 'log'), 'contact_f': (1500, 3500, 'log'), 'length': (0.11, 0.14)},
    },
    'tap': {   # real finger / knuckle taps: mono, sharp attack, a hollow surface mode, a longer tail. Round 3: the round-2
        # taps did not read as taps (phrase z 0.52 vs real 1.6-1.9): the real ones fall ~20 dB in ~7 ms from a 1 ms
        # attack and then ring on quietly. So: a short loud body (att 1-2 ms, rel 2.5-5 ms) over a quiet long ring
        # (ring_db, ring_ms: the two-stage decay) and a contact burst on the onset (the fingertip / nail).
        # Round 4 (the round-3 judge: 4-12 kHz 8 dB over, t40 1.8x long; forge refs 4-12 kHz -29.6, > 12 kHz -38 dB of
        # total, t40 50-110 ms): the contact keeps round 3's level and band (r4a: 4 dB quieter contacts read less as taps:
        # CLAP match z 1.9 -> 1.1-1.5) but loses its 4-12 kHz skirt (contact_lp_x), the tick and air are pinned quiet, a
        # top roll-off at 6-12 kHz, shorter quieter rings (12-40 ms),
        # shorter files, and a high-pass at 50-80 Hz (no 30-40 Hz sub under the body)
        # Round 5 (polish-ui; the round-4 bench: the darker taps lost the model checks, AudioSet names 0.6 -> 0.2,
        # typ z -1.6 -> -2.3). The forge-split critic (out/forge_v2/_polish_ui/tap_probe.py) put the loss in round 4's
        # 6-8.5 kHz top roll-off and missing air (> 12 kHz -48 dB vs the forge refs' -38; typ z -7.0, p_real 0.42) rather
        # than in the contact. The delivery (out/forge_v2/tap/final/, tap_deliver.py) is the round-3 picks with the
        # contact skirt low-passed on finger / glass / table (contact_lp_x 1.5-2.2; knuckle / soft keep it, for variety),
        # quieter ticks on glass / knuckle, glass contact 4.6 -> 3.2 kHz, shorter rings, the octave-down sub capped (0.12,
        # table 0.3) over a 55 Hz high-pass: critic medians r4 -> r5 total -2.39 -> -1.6, match z 1.29 -> 1.6, p_real 0.42
        # -> 0.8, typ z -7.0 -> -3.5; 4-12 kHz -23.4 -> -25.3 dB of total (forge refs -29.6; r3 -19.4), t40 70 -> 75-95 ms
        # (forge refs 90). The boxes below are round 4's
        'finger': {'width': (0.0, 0.3), 'knock': (0.05, 0.3), 'glass': 0.0, 'body_f': (110, 160, 'log'),
                   'att_ms': (1.0, 2.0, 'log'), 'hold_ms': (0.5, 2, 'log'), 'rel_ms': (2.5, 5, 'log'),
                   'ring_db': (-30, -21), 'ring_ms': (17, 32, 'log'), 'dbl': 0.0, 'contact_db': (-6, 2), 'contact_lp_x': (1.1, 1.8),
                   'contact_ms': (0.4, 1.2, 'log'), 'contact_f': (1200, 2800, 'log'), 'tick_db': (-24, -14),
                   'air_db': (-60, -42), 'lp': (7000, 12000, 'log'), 'hp': (50, 75, 'log'), 'sub': (0.0, 0.15),
                   'length': (0.08, 0.12)},
        'glass': {'width': (0.0, 0.3), 'knock': (0.05, 0.25), 'body_f': (120, 180, 'log'), 'att_ms': (1.0, 2.0, 'log'),
                  'glass': (0.3, 0.6), 'glass_f': (1800, 3200, 'log'), 'tick_db': (-18, -10), 'tick_f': (1500, 3500, 'log'),
                  'dbl': 0.0, 'hold_ms': (0.5, 2, 'log'), 'tick_ms': (2, 8, 'log'),
                  'rel_ms': (2.5, 5, 'log'), 'ring_db': (-30, -22), 'ring_ms': (18, 35, 'log'), 'contact_db': (-2, 4), 'contact_lp_x': (1.1, 1.8),
                  'contact_ms': (0.3, 0.8, 'log'), 'contact_f': (2000, 4000, 'log'), 'air_db': (-60, -40),
                  'lp': (8000, 12000, 'log'), 'hp': (50, 75, 'log'), 'sub': (0.0, 0.15), 'length': (0.08, 0.12)},
        'table': {'width': (0.0, 0.3), 'knock': (0.05, 0.4), 'glass': 0.0, 'body_f': (80, 110, 'log'),
                  'att_ms': (1.2, 3, 'log'), 'hold_ms': (0.5, 3, 'log'), 'dbl': 0.0, 'sub': 0.0,
                  'rel_ms': (3, 6, 'log'), 'ring_db': (-29, -20), 'ring_ms': (15, 30, 'log'), 'contact_db': (-8, 0), 'contact_lp_x': (1.1, 1.8),
                  'contact_ms': (0.8, 2.0, 'log'), 'contact_f': (900, 2000, 'log'), 'tick_db': (-26, -16),
                  'air_db': (-60, -46), 'lp': (6000, 10000, 'log'), 'hp': (55, 80, 'log'), 'length': (0.09, 0.14)},
        'knuckle': {'width': (0.0, 0.3), 'knock': (0.1, 0.5), 'glass': 0.0, 'body_f': (150, 220, 'log'),
                    'att_ms': (0.8, 1.6, 'log'), 'tick_db': (-16, -8), 'tick_f': (1500, 3500, 'log'), 'dbl': 0.0,
                    'hold_ms': (0.5, 1.5, 'log'), 'tick_ms': (2, 8, 'log'),
                    'rel_ms': (2.5, 4.5, 'log'), 'ring_db': (-27, -19), 'ring_ms': (18, 35, 'log'),
                    'contact_db': (-2, 4), 'contact_ms': (0.4, 1.0, 'log'), 'contact_f': (1500, 3200, 'log'),
                    'contact_lp_x': (1.1, 1.8),
                    'air_db': (-60, -46), 'lp': (7000, 11000, 'log'), 'hp': (55, 80, 'log'), 'sub': (0.0, 0.1),
                    'length': (0.08, 0.12)},
        'soft': {'width': (0.0, 0.3), 'knock': (0.05, 0.25), 'glass': 0.0, 'body_f': (95, 130, 'log'),
                 'att_ms': (1.5, 4, 'log'), 'tick_db': (-28, -18), 'dbl': 0.0, 'hold_ms': (1, 4, 'log'),
                 'rel_ms': (3.5, 7, 'log'), 'ring_db': (-28, -19), 'ring_ms': (15, 30, 'log'), 'contact_db': (-12, -4), 'contact_lp_x': (1.1, 1.8),
                 'contact_ms': (0.8, 2.0, 'log'), 'contact_f': (900, 2200, 'log'), 'air_db': (-60, -44),
                 'lp': (6000, 10000, 'log'), 'hp': (50, 75, 'log'), 'sub': (0.0, 0.15), 'length': (0.09, 0.14)},
    },
}


def _body(p, r, f, level=1.0, seconds=0.32):
    y = thump(f, seconds, p['att_ms'] * 1e-3, p['hold_ms'] * 1e-3, p['rel_ms'] * 1e-3, drop=p['drop'],
              harm=p['harm'], r=r)
    if p['sag_db'] > 0.05:                                       # a straight-in-dB sag from the peak to the hold's end
        t = dsp.t_of(len(y))
        a, h = p['att_ms'] * 1e-3, max(1e-3, p['hold_ms'] * 1e-3)
        y = y * 10 ** (-p['sag_db'] * np.clip((t - a) / h, 0, 1) / 20)
    if p['sub'] > 0.01:
        y = y + p['sub'] * thump(f * 0.5, seconds, p['att_ms'] * 1.3e-3, p['hold_ms'] * 1e-3, p['rel_ms'] * 1.2e-3,
                                 drop=p['drop'] * 0.5, harm=0.0, r=r)
    if p.get('ring_db', -60.0) > -59.5:                          # the quiet long second stage (no rng draw unless used)
        rg = thump(f, seconds, p['att_ms'] * 1e-3, 0.0, p['ring_ms'] * 1e-3, drop=p['drop'] * 0.5, harm=p['harm'], r=r)
        nz = dsp.butter(r.standard_normal(len(rg)), 'band', (0.5 * f, 3.0 * f), order=2)   # half of it a noisy ring:
        nz = nz / (np.abs(nz).max() + 1e-12) * np.abs(rg) / (np.abs(rg).max() + 1e-12)     # a surface, not a sine line
        rg = 0.6 * rg / (np.abs(rg).max() + 1e-12) + 0.8 * nz
        y = y / (np.abs(y).max() + 1e-12) + 10 ** (p['ring_db'] / 20) * rg / (np.abs(rg).max() + 1e-12)
    if p['sat'] > 0.05:
        y = fx.saturate(y, drive_db=p['sat'], mode='tanh', level='peak')
    return level * y / (np.abs(y).max() + 1e-12)


def render(p, r):
    sec = 0.34
    n = dsp.n_of(sec)
    pre = 0.002
    f = p['body_f'] * np.exp(0.012 * r.standard_normal())       # round robin: +-1.2 % pitch, +-0.4 dB
    lvl = 10 ** (0.02 * r.standard_normal())
    body = np.zeros(n)
    dsp.place(body, _body(p, r, f, lvl), pre)
    if p['dbl'] > 0.02:
        dsp.place(body, _body(p, r, f * p['dbl_ratio'], p['dbl'] * lvl), pre + p['dbl_dt'])
    if p['knock'] > 0.005:                                        # the surface's own hollow mode, struck with the body
        kn = partials(p['knock_f'] * np.exp(0.02 * r.standard_normal()), [1.0, 2.7], [1.0, 0.25],
                      0.004 + 0.006 * (p['knock_f'] < 500), 0.08, r, att_s=max(0.0005, p['att_ms'] * 0.3e-3))
        dsp.place(body, kn * p['knock'] * lvl, pre + 0.3 * p['att_ms'] * 1e-3)
    ct = np.zeros(n)
    if p.get('contact_db', -60.0) > -59.5:                       # the contact: mono (L = R, after the body's Haas
        f_c = p['contact_f']                                      # offset), on the onset, its own band
        c = burst(p['contact_ms'] * 1e-3, r, f_c / 2.5, min(18000.0, f_c * 2.5), att_s=0.0002)
        if p.get('contact_lp_x', 0.0) > 0.05:                     # round 4: no 4-12 kHz skirt (no rng draw)
            c = dsp.butter(c, 'low', min(18000.0, f_c * max(1.0, p['contact_lp_x'])), order=4)
            c = c / (np.abs(c).max() + 1e-12)
        dsp.place(ct, c * np.abs(body).max() * 10 ** (p['contact_db'] / 20), pre)
    body = dsp.butter(body, 'high', p['hp'], order=2)
    # tick: noise burst (+ optional glassy partial) on the onset, tick_db under the body peak
    tk = np.zeros(n)
    lo = p['tick_f'] / 2 ** (p['tick_bw'] / 2)
    hi = min(16000, p['tick_f'] * 2 ** (p['tick_bw'] / 2))
    nz = burst(p['tick_ms'] * 1e-3, r, lo, hi, att_s=0.0004)
    gl = partials(p['glass_f'], [1.0, 2.32], [1.0, 0.3], min(0.012, max(0.002, p['tick_ms'] * 0.6e-3)), 0.08, r,
                  att_s=0.0003)
    t = (1 - p['glass']) * nz[:len(gl)] if len(nz) >= len(gl) else np.pad((1 - p['glass']) * nz, (0, len(gl) - len(nz)))
    t = t + p['glass'] * gl
    t = t / (np.abs(t).max() + 1e-12)
    pk = np.abs(body).max()
    t_on = pre + 0.6 * p['att_ms'] * 1e-3                         # the tick lands as the body reaches its peak
    dsp.place(tk, t * pk * 10 ** (p['tick_db'] / 20), t_on)
    if p['dbl'] > 0.02:
        dsp.place(tk, t * pk * p['dbl'] * 10 ** ((p['tick_db'] - 3) / 20), t_on + p['dbl_dt'])
    if p['air_db'] > -59.5:                                       # the texture: noise shaped by the body's own envelope
        env = np.sqrt(np.convolve(body ** 2, np.ones(96) / 96, 'same'))
        nz = r.standard_normal(n)
        nz = dsp.butter(nz, 'band', (1000.0, 14000.0), order=2)
        tilt = abs(p['air_tilt'])
        nz = dsp.butter(nz, 'high' if p['air_tilt'] > 0 else 'low', 4000.0, order=1) * tilt + nz * (1 - tilt)
        nz = nz / (np.sqrt(np.mean(nz ** 2)) + 1e-12)
        dsp.place(tk, nz * env / (env.max() + 1e-12) * pk * 10 ** (p['air_db'] / 20), 0.0)
    # width: Haas offset of the body (low end stays phase-coherent enough for mono), decorrelated tick
    d = dsp.n_of(p['width'] / (6 * f)) if p['width'] > 0.02 else 0
    L = body.copy()
    R = np.concatenate([np.zeros(d), body[:n - d]]) if d else body.copy()
    st = np.stack([L + ct, R + ct], 1)
    tst = fx.decorrelate(tk, amount=p['tick_width'], cut=800.0)[:n]
    st = st + tst
    st = dsp.butter(st, 'low', p['lp'], order=2)
    st = gate(st, p['length'], p['release_ms'] * 1e-3, start=pre)
    if p.get('contact_db', -60.0) > -59.5:              # round 3: peak-normalise first, so the limiter's 80 ms release
        st = st * (10 ** (-1.5 / 20) / (np.abs(st).max() + 1e-12))   # does not squash the contact into the body
    st = fx.limit(st, ceiling_db=-1.0)
    return dsp.fade(st, 0.0005, 0.002)


def family_space(cat, fam):
    return narrow(SPACE, FAMILIES[cat][fam])


def family_params(cat, fam):
    return centre(DEFAULTS, FAMILIES[cat][fam])
