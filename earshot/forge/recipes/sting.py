"""A logo sting / reveal, in two idioms; every sample is computed (other recipes' layers are imported, not recorded).

1. The designed motion sting (`pro_logo_sting`, docs/motion-sfx-study.md "Logo / end sting"): ONE gesture - a falling
   chirp 1.5 kHz -> 150 Hz in 85 ms (about -38 oct/s, a sine with a few harmonics) landing on a low tonal hum
   (fundamental 64-80 Hz, buzzy harmonics through a formant near 400-500 Hz, sinking -0.6 oct) that rings about 0.8 s
   with a concave tail (t20 ~60 ms, t40 ~260 ms: a fast drop, then a long quiet ring). Mono. Balance peaks at
   250-500 Hz, tilt about -7 dB/oct.
2. The reveal sting (`logo_sting`, docs/motion-sfx-techniques.md section 12): an optional build (a reverse swell of the
   bloom's own hall), a hit on the logo frame (impact.render with a short punchy preset, at a low level) and a sub
   (subdrop.render), and the TONAL BLOOM it ends on: a chord (root, fifth, octave, optionally a third or ninth) of
   struck bells - modal partial sets from dsp.BELLS (glockenspiel bar, glass, tubular chime) or 2-op FM bells - whose
   low-pass opens from ~1 kHz to full over open_ms (the bloom), a granular shimmer of the chord's top partials, wide,
   in a computed hall. The library refs for this category are the struck metal / glass rings a sting ends on, so the
   bloom carries the tail (1-4 s, bright) and the hit / sub sit under it.
3. The launch reveal (`logo_sting` from round 3, mode 'launch', `_render_launch`): the judge found the reveal family a
   bell chord ringing 2 s (t40 2.1 s vs ~0.9 s held out). Now one composed gesture: a lead-in (rising air whoosh, rising
   zip chirp, the bloom's reverse swell, or none) -> a bright strike on the logo frame (a > 3-9 kHz noise tick + an
   optional light thump) -> a SHORT bloom (one note or a dyad, never a chord; concave decay) + a granular shimmer ->
   a short room that hears only > 500 Hz and a clean fade; t40 ~0.8-1.0 s. The round-2 reveal family is kept in
   FAMILIES_R2. Round 3 also adds weight to the chirp idiom (`_landing`: a landing blob + a low thud; a short sub with
   `sub_db`).
4. Round 4 (out/forge_v2/logo_sting/LOG.md, pro_logo_sting/LOG.md): the launch reveal gets a low strike under the hit
   (`low`, `low_f`, `low_tau`: a sine falling onto 46-82 Hz that blooms in over 12 ms, after the low cut, outside the
   room); the chirp idiom gets a top end that belongs to the gesture (`zap`: noise riding the chirp at zap_k x its
   pitch; `ctick`: a > 5 kHz landing tick) and a second gesture (`double`: a shorter, higher pre-chirp). New keys only:
   older parameter sets render exactly; the r3 families are kept in FAMILIES_R3.
Character parameters: pitch (root / chord / chirp range), material (bar, glass, tubular, fm), length (bloom_decay,
hum tails), brightness (bright, open_ms), weight (hit, sub), texture (shimmer, swell).
"""
import types

import numpy as np
from scipy import signal

from .. import dsp
from . import impact as _impact
from . import subdrop as _subdrop
from .impact import hall_ir, measure, study_extra, study_penalty  # noqa: F401

SR = dsp.SR
TARGET = 'a bright logo sting: a chime chord ringing out'
ALTS = ['a doorbell', 'a church bell', 'a notification sound', 'a gong']
REFSET = 'logo_sting'

CHORDS = {'fifth': [0, 7, 12], 'major': [0, 4, 7, 12], 'sus2': [0, 2, 7, 12], 'add9': [0, 4, 7, 14],
          'octave': [0, 12, 24], 'minor9': [0, 3, 7, 14]}
MATERIALS = ['bar', 'glass', 'tubular', 'fm', 'triangle']
# a struck triangle / small metal bar cluster: dense inharmonic partials (free-free beam with bending), highs ringing
TRIANGLE = ([1.0, 2.76, 3.9, 5.4, 6.7, 8.93, 10.3, 13.34], [0.6, 1.0, 0.5, 0.8, 0.45, 0.5, 0.3, 0.25])

SPACE = {
    'mode': ['reveal', 'chirp', 'launch'],  # the idioms (launch: round 3)
    # reveal: bloom chord
    'root': (300, 2200, 'log'),
    'chord': list(CHORDS),
    'material': MATERIALS,
    'bloom_decay': (0.6, 4.0, 'log'),
    'bright': (0.0, 1.0),
    'open_ms': (10, 400, 'log'),
    'strike': (0.0, 1.0),
    'shimmer': (0.0, 0.6),
    'hit': (0.0, 0.8),
    'sub': (0.0, 0.8),
    'swell': (0.0, 1.2),
    'swell_level': (0.05, 0.6, 'log'),
    'verb': (0.0, 0.8),
    'verb_rt': (0.8, 5.0, 'log'),
    'width': (0.0, 1.0),
    'low_cut': (25, 800, 'log'),            # final high-pass of the reveal (round 2: the ring refs have nothing < 300 Hz)
    # chirp idiom
    'chirp_f0': (700, 3500, 'log'),
    'chirp_f1': (90, 320, 'log'),
    'chirp_ms': (40, 170, 'log'),
    'chirp_harm': (0.0, 1.0),
    'hum_f': (52, 110, 'log'),
    'hum_sink': (0.0, 1.0),                 # octaves the hum sinks over its ring
    'hum_formant': (200, 900, 'log'),
    'hum_buzz': (0.0, 1.0),                 # harmonic richness
    'hum_level': (0.2, 1.5, 'log'),
    'hum_fast': (0.02, 0.15, 'log'),        # the concave tail: a fast drop (s) ...
    'hum_slow': (0.15, 0.9, 'log'),         # ... then a slow ring (s)
    'hum_mix': (0.03, 0.5, 'log'),          # level of the slow ring
    'land': (0.0, 1.0),                     # round 3: the landing blob (land_f falling half an octave, ~25 ms)
    'land_f': (400, 1300, 'log'),
    'thud': (0.0, 1.0),                     # round 3: a low thud onto 2 x hum_f at the landing
    'sub_db': (30.0, 130.0),
    'bell_decay': (0.08, 0.6, 'log'),       # round 3: the chirp idiom's bell partial (0.35 = r2)                # round 3: decay (dB/s) of the chirp idiom's sub (45 = r2): a short sub
                                            # under the hum is the weight at 63 Hz without a long tail
    'length': (0.5, 1.8),
    # launch idiom (round 3): lead-in -> hit -> short bloom + shimmer -> clean tail (see _render_launch)
    'lead': ['whoosh', 'chirp', 'swell', 'none'],
    'lead_s': (0.1, 0.7),                   # s of lead-in before the hit
    'lead_level': (0.0, 0.8),               # its peak against the hit + bloom
    'lead_f0': (200, 2000, 'log'),          # whoosh band centre / chirp start
    'lead_f1': (1500, 12000, 'log'),        # ... at the hit
    'tick': (0.0, 1.5),                     # the bright strike (> tick_f noise burst)
    'tick_f': (2500, 9000, 'log'),
    'tick_ms': (2.0, 15.0, 'log'),
    'thump': (0.0, 0.8),                    # a light body under the hit (above low_cut)
    'thump_f': (70, 260, 'log'),
    'notes': ['one', 'fifth', 'octave', 'fourth', 'ninth', 'third'],   # one note or a dyad, never a chord
    'ring': (0.3, 1.2),                     # bloom level against the tick
    'strike_tau': (0.04, 0.3, 'log'),       # the concave decay: the struck level falls fast ...
    'ring_mix': (0.05, 0.5),                # ... to this share ...
    'ring_tau': (0.15, 0.6, 'log'),         # ... which rings on with this time constant
    'bloom_att': (0.0, 0.15),               # a swell-in of the bloom (with a low-pass opening): the tonal bloom
    'shimmer_tau': (0.15, 0.8, 'log'),      # decay of the granular shimmer
    'bloom_delay': (0.0, 0.03),             # the bloom enters this long after the tick (techniques doc: 0-30 ms), so
                                            # the strike owns the onset (r3 first run: 'dull onset' on 3 of 5)
    # round 4 (new keys only: every older parameter set renders exactly; the layers draw only when used)
    'low': (0.0, 1.0),                      # launch: a low strike under the hit (a sine falling onto low_f), bypassing
    'low_f': (38.0, 95.0, 'log'),           # low_cut and the room (r3: < 150 Hz -52.7 dB of total, no body at all)
    'low_tau': (0.06, 0.4, 'log'),
    'zap': (0.0, 1.0),                      # chirp: band-passed noise riding the chirp at zap_k x its pitch (the top
    'zap_k': (3.0, 10.0),                   # end: r3 had nothing above ~5 kHz, 4-12 kHz -41.8 vs the film's -21)
    'ctick': (0.0, 1.0),                    # chirp: a bright > 5 kHz tick on the landing
    'double': (0.0, 1.0),                   # chirp: a shorter, higher pre-chirp before the main one (a 'ba-dum')
    'swell_trim': (0.0, 0.08),              # launch, swell lead: s of the hall's build-up left out (0 = r3: a dip before the hit)
    'double_gap': (0.0, 0.12),              # gap between the pre-chirp and the main chirp (s; >= 20 ms of zeros reads
                                            # as 'digital silence inside': palette run 1)
}

_REVEAL = {'mode': 'reveal', 'root': 880.0, 'chord': 'add9', 'material': 'bar', 'bloom_decay': 2.2, 'bright': 0.6,
           'open_ms': 120.0, 'strike': 0.5, 'shimmer': 0.25, 'hit': 0.3, 'sub': 0.25, 'swell': 0.0,
           'swell_level': 0.2, 'verb': 0.35, 'verb_rt': 2.6, 'width': 0.6, 'low_cut': 25.0}
_CHIRP = {'mode': 'chirp', 'chirp_f0': 1500.0, 'chirp_f1': 150.0, 'chirp_ms': 85.0, 'chirp_harm': 0.7,
          'hum_f': 76.0, 'hum_sink': 0.6, 'hum_formant': 500.0, 'hum_buzz': 0.9, 'hum_level': 0.8, 'hum_fast': 0.045,
          'hum_slow': 0.2, 'hum_mix': 0.06, 'length': 0.85, 'sub': 0.0, 'shimmer': 0.0, 'verb': 0.05,
          'verb_rt': 0.8, 'width': 0.0, 'root': 880.0, 'material': 'bar', 'bright': 0.5, 'land': 0.0, 'land_f': 800.0,
          'thud': 0.0, 'sub_db': 45.0}
_CHIRP3 = {**_CHIRP, 'land': 0.5, 'land_f': 1400.0, 'thud': 0.5, 'hum_level': 0.95, 'hum_fast': 0.05, 'chirp_harm': 1.0,
           'sub': 0.25, 'sub_db': 100.0}
_LAUNCH = {'mode': 'launch', 'lead': 'whoosh', 'lead_s': 0.45, 'lead_level': 0.45, 'lead_f0': 600.0,
           'lead_f1': 9000.0, 'tick': 1.0, 'tick_f': 5000.0, 'tick_ms': 6.0, 'thump': 0.08, 'thump_f': 150.0,
           'material': 'glass', 'root': 1318.5, 'notes': 'fifth', 'bright': 0.7, 'ring': 0.9, 'strike_tau': 0.12,
           'ring_mix': 0.25, 'ring_tau': 0.3, 'bloom_att': 0.0, 'shimmer': 0.4, 'shimmer_tau': 0.35, 'verb': 0.2,
           'verb_rt': 1.0, 'width': 0.6, 'low_cut': 140.0, 'bloom_delay': 0.012}
FAMILIES = {
    'logo_sting': [   # round 3: composed launch reveals (lead-in -> hit -> short bloom / shimmer -> clean tail)
        ('whoosh_glass', dict(_LAUNCH)),
        ('zip_ping', {**_LAUNCH, 'lead': 'chirp', 'lead_s': 0.16, 'lead_level': 0.4, 'lead_f0': 420.0,
                      'lead_f1': 2600.0, 'tick': 0.9, 'tick_f': 6000.0, 'tick_ms': 4.0, 'thump': 0.0,
                      'material': 'bar', 'root': 1760.0, 'notes': 'octave', 'strike_tau': 0.1, 'ring_mix': 0.2, 'bright': 0.45,
                      'ring_tau': 0.28, 'shimmer': 0.15, 'verb': 0.15, 'width': 0.3, 'low_cut': 300.0}),
        ('reverse_bloom', {**_LAUNCH, 'lead': 'swell', 'lead_s': 0.5, 'lead_level': 0.35, 'tick': 0.8,
                           'tick_f': 4500.0, 'tick_ms': 8.0, 'thump': 0.15, 'thump_f': 110.0, 'material': 'bar',
                           'root': 1480.0, 'notes': 'ninth', 'strike_tau': 0.15, 'ring_mix': 0.3, 'ring_tau': 0.32,
                           'shimmer': 0.3, 'verb': 0.3, 'verb_rt': 1.2, 'width': 0.7, 'low_cut': 90.0}),
        ('spark', {**_LAUNCH, 'lead': 'none', 'lead_level': 0.3, 'tick': 1.1, 'tick_f': 7000.0, 'tick_ms': 3.0,
                   'thump': 0.0, 'material': 'triangle', 'root': 1318.5, 'notes': 'one', 'strike_tau': 0.09, 'bright': 0.35,
                   'ring_mix': 0.3, 'ring_tau': 0.28, 'shimmer': 0.6, 'shimmer_tau': 0.4, 'verb': 0.15,
                   'width': 0.9, 'low_cut': 300.0}),
        ('soft_bloom', {**_LAUNCH, 'lead': 'whoosh', 'lead_s': 0.3, 'lead_f0': 300.0, 'lead_f1': 4000.0,
                        'lead_level': 0.3, 'tick': 0.7, 'tick_f': 4000.0, 'tick_ms': 10.0, 'thump': 0.18,
                        'thump_f': 130.0, 'material': 'tubular', 'root': 987.8, 'notes': 'fourth', 'bloom_att': 0.06,
                        'strike_tau': 0.16, 'ring_mix': 0.25, 'ring_tau': 0.32, 'shimmer': 0.25, 'verb': 0.3,
                        'verb_rt': 1.1, 'width': 0.5, 'low_cut': 100.0}),
    ],
    'logo_sting_r2': [   # round 2 (the r2 finals): bright bell-chord rings over a light, high-passed hit (the category's
                         # refs: nothing < 300 Hz, bright rings over a light, high-passed hit (the category's refs: nothing < 300 Hz,
                      # the onset almost all above 6 kHz, t40 0.9-2 s)
        ('glass_bloom', {**_REVEAL, 'material': 'glass', 'root': 1760.0, 'chord': 'add9', 'bloom_decay': 1.2,
                         'bright': 0.8, 'strike': 0.8, 'shimmer': 0.35, 'hit': 0.15, 'sub': 0.0, 'low_cut': 450.0,
                         'open_ms': 30.0}),
        ('triangle', {**_REVEAL, 'material': 'triangle', 'root': 1480.0, 'chord': 'fifth', 'bloom_decay': 1.9,
                      'bright': 0.85, 'open_ms': 10.0, 'strike': 0.9, 'shimmer': 0.15, 'hit': 0.1, 'sub': 0.0,
                      'verb': 0.25, 'low_cut': 500.0}),
        ('fm_reveal', {**_REVEAL, 'material': 'fm', 'root': 1320.0, 'chord': 'major', 'bloom_decay': 1.3,
                       'bright': 0.8, 'swell': 0.6, 'swell_level': 0.25, 'hit': 0.2, 'sub': 0.0, 'open_ms': 60.0,
                       'low_cut': 350.0}),
        ('chime_hit', {**_REVEAL, 'material': 'tubular', 'root': 1175.0, 'chord': 'sus2', 'bloom_decay': 1.5,
                       'bright': 0.75, 'strike': 0.8, 'hit': 0.3, 'sub': 0.0, 'shimmer': 0.1, 'verb': 0.4,
                       'verb_rt': 2.4, 'low_cut': 300.0, 'open_ms': 25.0}),
        ('bar_sparkle', {**_REVEAL, 'material': 'bar', 'root': 1980.0, 'chord': 'octave', 'bloom_decay': 1.1,
                         'bright': 0.8, 'strike': 0.9, 'shimmer': 0.55, 'hit': 0.0, 'sub': 0.0, 'width': 0.9,
                         'open_ms': 10.0, 'low_cut': 500.0}),
    ],
    'pro_logo_sting': [   # round 3: the same gestures with weight on the landing (land / thud), fuller hums, and
                          # members further apart (purer / buzzier hums, a sub on 'deep'); 'bell' re-made (r2: rejected)
        ('ref', dict(_CHIRP3)),
        ('high', {**_CHIRP3, 'chirp_f0': 2600.0, 'chirp_f1': 240.0, 'chirp_ms': 70.0, 'hum_f': 98.0,
                  'hum_formant': 600.0, 'length': 0.7, 'hum_slow': 0.16, 'land_f': 1100.0, 'thud': 0.3,
                  'hum_buzz': 0.55, 'sub': 0.15, 'sub_db': 110.0}),
        ('deep', {**_CHIRP3, 'chirp_f0': 1200.0, 'chirp_f1': 120.0, 'chirp_ms': 105.0, 'hum_f': 62.0,
                  'hum_formant': 380.0, 'sub': 0.35, 'length': 1.0, 'hum_slow': 0.26, 'thud': 0.7, 'land': 0.4,
                  'land_f': 650.0, 'hum_level': 1.05, 'hum_buzz': 1.0, 'sub_db': 100.0}),
        ('bell', {**_CHIRP3, 'shimmer': 0.1, 'root': 1100.0, 'hum_buzz': 0.4, 'hum_formant': 520.0, 'land': 0.35,
                  'thud': 0.4, 'bell_decay': 0.16, 'chirp_f0': 1900.0, 'chirp_f1': 190.0, 'chirp_ms': 75.0,
                  'hum_f': 84.0}),   # r3 run 1: dropped as a near duplicate of 'ref', its 950 Hz partial rang past the hum
        ('long', {**_CHIRP3, 'hum_slow': 0.45, 'hum_mix': 0.1, 'length': 1.3, 'hum_sink': 0.8, 'chirp_harm': 0.7,
                  'verb': 0.12, 'verb_rt': 1.2, 'land': 0.3, 'thud': 0.55, 'sub': 0.2, 'sub_db': 80.0}),
    ],
    'pro_logo_sting_r2': [
        ('ref', dict(_CHIRP)),
        ('high', {**_CHIRP, 'chirp_f0': 2600.0, 'chirp_f1': 240.0, 'chirp_ms': 70.0, 'hum_f': 98.0,
                  'hum_formant': 600.0, 'length': 0.7, 'hum_slow': 0.16}),
        ('deep', {**_CHIRP, 'chirp_f0': 1200.0, 'chirp_f1': 120.0, 'chirp_ms': 105.0, 'hum_f': 62.0,
                  'hum_formant': 380.0, 'sub': 0.25, 'length': 1.0, 'hum_slow': 0.26}),
        ('bell', {**_CHIRP, 'shimmer': 0.1, 'root': 950.0, 'hum_buzz': 0.4, 'hum_formant': 520.0}),   # round 3:
                     # the bell partial at 3.3 kHz read as a narrow +31 dB resonance at shimmer 0.25 / root 1.2 kHz
        ('long', {**_CHIRP, 'hum_slow': 0.45, 'hum_mix': 0.1, 'length': 1.3, 'hum_sink': 0.8, 'chirp_harm': 0.7,
                  'verb': 0.12, 'verb_rt': 1.2}),
    ],
}
FAMILIES_R2 = {'logo_sting': FAMILIES.pop('logo_sting_r2'), 'pro_logo_sting': FAMILIES.pop('pro_logo_sting_r2')}   # round 2's reveal family (bell chords, r2 finals)
FAMILIES_R3 = {k: list(v) for k, v in FAMILIES.items()}   # the round-3 families (the r3 finals, in final_r3/)
# round 4. logo_sting (the judge: no body, < 150 Hz -52.7 dB of total vs the held-out -26.8): a low strike under every
# hit, pitched / timed per member. pro_logo_sting (the judge: near clones at CLAP cos 0.943, nothing above ~5 kHz, 4-12 kHz
# -41.8 dB vs the film's -21.0): a top end that belongs to the gesture (noise riding the chirp, a landing tick, a high
# bell ping) and members that are different gestures, not one gesture at five pitches (a zap, a double chirp, a pure
# chirp onto a bell, a heavy dark drop, the reference chirp-to-hum)
FAMILIES['logo_sting'] = [
    ('whoosh_glass', {**FAMILIES_R3['logo_sting'][0][1], 'low': 0.04, 'low_f': 55.0, 'low_tau': 0.16}),
    ('zip_ping', {**FAMILIES_R3['logo_sting'][1][1], 'low': 0.044, 'low_f': 73.0, 'low_tau': 0.1}),
    ('reverse_bloom', {**FAMILIES_R3['logo_sting'][2][1], 'low': 0.045, 'low_f': 46.0, 'low_tau': 0.24}),
    # (swell_trim 0.045 removes a -20 dB dip just before the hit, also in r3; found after the bench, not delivered yet)
    ('spark', {**FAMILIES_R3['logo_sting'][3][1], 'low': 0.03, 'low_f': 82.0, 'low_tau': 0.09}),
    ('soft_bloom', {**FAMILIES_R3['logo_sting'][4][1], 'low': 0.041, 'low_f': 62.0, 'low_tau': 0.2}),
]
_CHIRP4 = {**_CHIRP3, 'zap': 0.0, 'zap_k': 6.0, 'ctick': 0.0, 'double': 0.0, 'double_gap': 0.05}
FAMILIES['pro_logo_sting'] = [
    ('ref', {**_CHIRP4, 'ctick': 0.4}),
    ('zap', {**_CHIRP4, 'chirp_f0': 2600.0, 'chirp_f1': 240.0, 'chirp_ms': 70.0, 'hum_f': 98.0, 'hum_formant': 600.0,
             'length': 0.7, 'hum_slow': 0.16, 'land_f': 1100.0, 'thud': 0.3, 'hum_buzz': 0.55, 'sub': 0.15,
             'sub_db': 110.0, 'zap': 0.6, 'zap_k': 4.0, 'chirp_harm': 0.5, 'ctick': 0.2}),
    ('deep', {**_CHIRP4, 'chirp_f0': 950.0, 'chirp_f1': 100.0, 'chirp_ms': 130.0, 'hum_f': 56.0, 'hum_formant': 380.0,
              'sub': 0.35, 'length': 1.0, 'hum_slow': 0.26, 'thud': 0.7, 'land': 0.4, 'land_f': 650.0,
              'hum_level': 1.05, 'hum_buzz': 1.0, 'sub_db': 100.0, 'ctick': 0.3, 'hum_sink': 1.0}),
              # (palette run 2: at 1.2 kHz -> 120 Hz / 105 ms onto 62 Hz it was 'ref' an octave down, CLAP cos 0.964)
    ('bell', {**_CHIRP4, 'shimmer': 0.1, 'root': 1100.0, 'hum_buzz': 0.3, 'hum_formant': 520.0, 'land': 0.35, 'zap': 0.2,
              'zap_k': 8.0, 'thud': 0.4, 'bell_decay': 0.16,   # (palette run 2: a 3.15 kHz ping read 'narrow resonance +31 dB') 'chirp_f0': 1900.0, 'chirp_f1': 190.0, 'chirp_ms': 75.0, 'hum_f': 84.0,
              'chirp_harm': 0.4, 'ctick': 0.25, 'hum_level': 0.75}),
    ('double', {**_CHIRP4, 'hum_slow': 0.3, 'hum_mix': 0.08, 'length': 1.0, 'hum_sink': 0.8, 'chirp_harm': 0.8,
                'land': 0.3, 'thud': 0.55, 'sub': 0.2, 'sub_db': 90.0, 'double': 0.7, 'double_gap': 0.012,
                'zap': 0.4, 'zap_k': 7.0, 'chirp_f0': 1700.0, 'hum_f': 70.0}),
]
DEFAULTS = dict(FAMILIES['logo_sting'][0][1])
TARGETS = {'logo_sting': 'a short bright logo reveal sting: a whoosh into a hit with a shimmering chime', 'pro_logo_sting': 'a designed logo sting: a falling chirp landing on a low hum'}
STUDY_STING = {'logo_sting': {'t40': (650, 1150), 'tilt': (1.5, 10.0), 'sh150': (0.0, 0.12)},  # r3: the forge ref
               # mk-3108_1 (t40 0.86 s, tilt +8.9) and kit-sting_tail_01 (tilt +5.6): bright, short, no sub
               'pro_logo_sting': {'len50': (0.5, 1.5), 't20': (30, 110), 't40': (150, 450), 'tilt': (-9.0, -5.0),
                                  'corr': (0.9, 1.0), 'ridge0': (900, 2400), 'ridge1': (55, 110)}}
_SEARCH = {
    'logo_sting': ['lead_s', 'lead_level', 'tick', 'tick_f', 'thump', 'ring', 'strike_tau', 'ring_mix', 'ring_tau',
                   'shimmer', 'verb', 'width', 'low_cut', 'bright', 'bloom_delay'],
    'pro_logo_sting': ['chirp_f0', 'chirp_f1', 'chirp_ms', 'chirp_harm', 'hum_f', 'hum_sink', 'hum_formant',
                       'hum_buzz', 'hum_level', 'hum_fast', 'hum_slow', 'hum_mix', 'length', 'land', 'thud', 'sub',
                       'sub_db'],
}
_LIMITS = {'logo_sting': {'lead_s': (0.15, 0.6), 'ring_tau': (0.15, 0.45, 'log'), 'verb': (0.0, 0.45),
                          'low_cut': (60, 600, 'log'), 'lead_level': (0.25, 0.7), 'tick': (0.6, 1.5),
                          'thump': (0.0, 0.4), 'bloom_delay': (0.004, 0.025)},
           'pro_logo_sting': {'chirp_f0': (900, 3000, 'log'), 'chirp_f1': (100, 280, 'log'),
                              'chirp_ms': (50, 130, 'log'), 'hum_f': (55, 105, 'log'), 'length': (0.6, 1.6),
                              'land': (0.2, 0.9), 'thud': (0.2, 0.9), 'sub': (0.1, 0.5), 'sub_db': (60.0, 130.0)}}


# ------------------------------------------------------------------------------------------------ the chirp idiom
def _chirp(p, n, r):
    """an exponential falling sine (constant octaves per second) with harmonics 2 and 3, 3 ms attack, gone by its end"""
    L = min(n, dsp.n_of(p['chirp_ms'] / 1000 * 1.25))
    t = np.arange(L) / SR
    T = p['chirp_ms'] / 1000
    f = p['chirp_f0'] * (p['chirp_f1'] / p['chirp_f0']) ** np.clip(t / T, 0, 1)
    ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 0.3)
    h = p['chirp_harm']
    y = np.sin(ph) + h * (0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph))
    y *= np.where(2 * f < SR * 0.45, 1, 0)                 # (harmonic 3 of a 3.5 kHz start stays under Nyquist)
    # a crescendo into the landing (the loudest point is the bottom of the chirp, as in the pro sting: attack 72 ms),
    # then gone within a quarter of its length
    e = (1 - np.exp(-t / 0.003)) * (0.3 + 0.7 * np.clip(t / T, 0, 1)) * np.clip((1.25 - t / T) / 0.25, 0, 1) ** 1.5
    out = np.zeros(n)
    out[:L] = y * e
    return out / (np.abs(out).max() + 1e-12)


def _hum(p, n, r, t0):
    """the landing: a buzzy harmonic tone (fundamental hum_f sinking hum_sink octaves) through a formant, with a
    concave two-stage decay (fast drop hum_fast, then a quiet ring hum_slow at hum_mix)"""
    t = np.arange(n) / SR
    tt = np.maximum(t - t0, 0)
    dur = max(t[-1] - t0, 0.1)
    f = p['hum_f'] * 2 ** (-p['hum_sink'] * np.clip(tt / dur, 0, 1) ** 0.7)
    ph = 2 * np.pi * np.cumsum(f) / SR
    K = 12
    y = np.zeros(n)
    for k in range(1, K + 1):
        a = (1.0 / k) ** (1.6 - 1.1 * p['hum_buzz'])
        fk = k * f
        form = 1.0 / (1 + ((np.log2(fk / p['hum_formant'])) / 0.7) ** 2)       # a soft formant bump
        y += a * (0.35 + form) * np.sin(k * ph + r.uniform(0, 6.28)) * (fk < SR * 0.45)
    e = (np.exp(-tt / p['hum_fast']) + p['hum_mix'] * np.exp(-tt / p['hum_slow'])) * (1 - np.exp(-tt / 0.004))
    e *= t >= t0
    y *= e
    return y / (np.abs(y).max() + 1e-12)


def _landing(p, n, t0):
    """round 3 (the judge: 'the chirp-to-hum gesture with more weight'): what the pro sting lands on, made heavier: a
    bright landing blob (a sine at land_f falling half an octave, ~25 ms; the ref's blob near 0.6-1 kHz) and a low
    thud (a sine falling an octave onto 2 x hum_f in 40 ms, ~50 ms decay) at the bottom of the chirp; mono"""
    t = np.maximum(np.arange(n) / SR - t0, 0)
    on = (np.arange(n) / SR >= t0) * (1 - np.exp(-t / 0.002))
    y = np.zeros(n)
    if p.get('land', 0) > 0.01:
        f = p['land_f'] * 2 ** (-0.5 * (1 - np.exp(-t / 0.03)))
        y += p['land'] * np.sin(2 * np.pi * np.cumsum(f * (t > 0)) / SR) * np.exp(-t / 0.025) * on
    if p.get('thud', 0) > 0.01:
        f = 2 * p['hum_f'] * (1 + np.exp(-t / 0.04))
        y += p['thud'] * np.sin(2 * np.pi * np.cumsum(f * (t > 0)) / SR) * np.exp(-t / 0.05) * on
    return y


def _zap(p, n, r):
    """round 4: the top of the chirp (the judge: nothing above ~5 kHz, 4-12 kHz -41.8 dB of total vs the film's -21):
    white noise through a resonant band-pass whose centre rides the chirp at zap_k x its pitch (1.5 kHz -> 150 Hz at
    zap_k 6: 9 kHz -> 900 Hz), with the chirp's crescendo, gone with it; mono, peak 1"""
    L = min(n, dsp.n_of(p['chirp_ms'] / 1000 * 1.25))
    t = np.arange(L) / SR
    T = p['chirp_ms'] / 1000
    f = p['chirp_f0'] * (p['chirp_f1'] / p['chirp_f0']) ** np.clip(t / T, 0, 1)
    fc = np.minimum(f * p['zap_k'], 15000.0)
    y = dsp.svf(r.standard_normal(L), fc, 3.0, 'bandn')
    e = (1 - np.exp(-t / 0.002)) * (0.4 + 0.6 * np.clip(t / T, 0, 1)) * np.clip((1.25 - t / T) / 0.25, 0, 1) ** 1.5
    out = np.zeros(n)
    out[:L] = dsp.butter(y * e, 'low', 13000.0, 2)
    return out / (np.abs(out).max() + 1e-12)


def _ctick(n, r, t0):
    """round 4: a bright tick on the landing: > 5 kHz noise (rolled off above 13 kHz), (1 - u)^3 over 5 ms; mono"""
    L = dsp.n_of(0.005)
    u = np.arange(L) / L
    w = dsp.butter(r.standard_normal(L) * (1 - u) ** 3 * (1 - np.exp(-np.arange(L) / (SR * 0.0002))), 'band',
                   (5000.0, 13000.0), 2)
    out = np.zeros(n)
    i = dsp.n_of(t0)
    m = min(L, n - i)
    out[i:i + m] = w[:m] / (np.abs(w).max() + 1e-12)
    return out


def _render_chirp(p, r):
    n = dsp.n_of(p['length'])
    ch = _chirp(p, n, r)
    t0 = p['chirp_ms'] / 1000 * 0.92                       # lands as the chirp reaches its bottom
    hum = _hum(p, n, r, t0)
    y = ch + p['hum_level'] * hum
    if p.get('land', 0) > 0.01 or p.get('thud', 0) > 0.01:     # round 3: weight on the landing (no rng draws, so
        y = y + _landing(p, n, t0)                             # older parameter sets render exactly)
    if p.get('zap', 0) > 0.005:                            # round 4 (drawn only when used: older sets render exactly)
        y = y + p['zap'] * _zap(p, n, r)
    if p.get('ctick', 0) > 0.005:
        y = y + p['ctick'] * _ctick(n, r, t0)
    if p.get('sub', 0) > 0.01:
        sd = {**_subdrop.FAMILIES['pro_sub_drop'][0][1], 'f0': p['hum_f'], 'f1': p['hum_f'] * 0.66, 'bloom': 0.06,
              'click': 0.0, 'length': p['length'], 'decay_db': p.get('sub_db', 45.0)}
        s = _subdrop.render(sd, r).mean(axis=1)
        s = np.concatenate([np.zeros(dsp.n_of(t0)), s])[:n]
        y[:len(s)] += p['sub'] * s
    if p.get('shimmer', 0) > 0.01:                         # a soft bell partial on the landing
        b = dsp.bell(p['root'], max(0.3, p['length'] - t0), r, 'bar', decay=p.get('bell_decay', 0.35), bright=0.3,
                     strike=0.0)                           # r3: bell_decay (0.35 = r2; the partial outlasted the hum)
        b = np.concatenate([np.zeros(dsp.n_of(t0)), b])[:n]
        y[:len(b)] += 0.6 * p['shimmer'] * b
    if p.get('verb', 0) > 0.01:
        ir = hall_ir(r, p['verb_rt'], 0.6, 0.5, 0.008)
        y = y + p['verb'] * signal.fftconvolve(y, ir)[:n]
    if p.get('double', 0) > 0.005:                         # round 4: a shorter, higher pre-chirp ('ba-dum')
        q = {**p, 'chirp_f0': min(3500.0, p['chirp_f0'] * 1.35), 'chirp_f1': p['chirp_f1'] * 2.5,
             'chirp_ms': p['chirp_ms'] * 0.55}
        k = dsp.n_of(q['chirp_ms'] / 1000 * 1.25)
        pre = _chirp(q, k, r)
        if q.get('zap', 0) > 0.005:
            pre = pre + q['zap'] * _zap(q, k, r)
        pre = np.concatenate([pre / (np.abs(pre).max() + 1e-12), np.zeros(dsp.n_of(p.get('double_gap', 0.05)))])
        y = np.concatenate([p['double'] * np.abs(y).max() * pre, y])
    y = dsp.butter(y, 'high', 30.0, 2)
    y = dsp.fade(y, 0.0008, min(0.12, 0.2 * p['length']))
    return np.stack([y, y], axis=1)


# ------------------------------------------------------------------------------------------------ the reveal idiom
def _note(p, f, seconds, r):
    m = p['material']
    if m == 'fm':
        x = dsp.fm(f, 1.4, 3 + 5 * p['bright'], seconds, decay=p['bloom_decay'] * 0.45, index_decay=0.12,
                   attack=0.002)
        x += 0.35 * dsp.fm(f * 2.0, 3.5, 2.0, seconds, decay=p['bloom_decay'] * 0.2, index_decay=0.05)
        return x / (np.abs(x).max() + 1e-12)
    parts = TRIANGLE if m == 'triangle' else m
    return dsp.bell(f, seconds, r, parts, decay=p['bloom_decay'] * (0.35 if m == 'triangle' else 0.5),
                    slope=0.5 if m in ('triangle', 'glass') else 0.8, bright=p['bright'], beat=1.2,
                    strike=0.2 + 0.6 * p['strike'])


def _bloom(p, r):
    """the chord: each note a struck bell, lightly staggered (a flam of 4-12 ms), summed through an opening low-pass"""
    seconds = min(5.5, p['bloom_decay'] * 1.6 + 0.4)
    n = dsp.n_of(seconds)
    y = np.zeros(n)
    for j, st in enumerate(CHORDS[p['chord']]):
        f = p['root'] * 2 ** (st / 12)
        if f > 9000:
            continue
        x = _note(p, f, seconds, r) * (0.85 ** j)
        d = dsp.n_of(j * r.uniform(0.004, 0.012))
        y[d:] += x[:n - d]
    t = np.arange(n) / SR
    fc = 900 + 17000 * (1 - np.exp(-t / max(p['open_ms'] / 1000, 0.005)))
    lp = dsp.svf(y, fc, 0.6, 'low')
    return lp / (np.abs(lp).max() + 1e-12)


def _render_reveal(p, r):
    bloom = _bloom(p, r)
    n = len(bloom)
    out = np.zeros((n, 2))
    out[:, 0] += bloom
    out[:, 1] += bloom
    if p['shimmer'] > 0.01:                                # granular shimmer of the chord's top, wide
        src = dsp.butter(bloom[: dsp.n_of(1.0)], 'high', 2500.0, 2)
        sh = fx_granular(src, n / SR, r)
        sh *= np.exp(-np.arange(n) / SR / max(0.4, p['bloom_decay'] * 0.6))[:, None]
        out += p['shimmer'] * sh / (np.abs(sh).max() + 1e-12)
    if p['width'] > 0.01:                                  # decorrelated bloom above 600 Hz (all-pass side signal;
        from ..fx import decorrelate                       # round 2: a 2.6 ms Haas delay printed a comb ring, z 16)
        hi = dsp.butter(bloom, 'high', 600.0, 2)
        out += decorrelate(hi, 0.8 * p['width']) - hi[:, None]
    if p['verb'] > 0.01:
        for c in range(2):
            ir = hall_ir(r, p['verb_rt'], 0.35, 0.8, 0.02 + 0.004 * c)
            wet = signal.fftconvolve(out[:, c], ir)[:n]
            out[:, c] += p['verb'] * wet
    pk = np.abs(out).max() + 1e-12
    if p['hit'] > 0.01:                                    # the hit on the logo frame (impact v3, short + punchy)
        hp = {**_impact.FAMILIES['impact'][2][1], 'roll_hold': 0.5, 'roll_fall': 0.25, 'hall_rt': 1.4,
              'hall_wet': 0.35, 'floor_db': -110.0, 'crack_f': 3500.0}
        h = _impact.render(hp, r)[:n]
        out[:len(h)] += p['hit'] * pk * h / (np.abs(h).max() + 1e-12)
    if p['sub'] > 0.01:
        s = _subdrop.render({**_subdrop.FAMILIES['sub_drop'][0][1], 'length': 1.6, 'click': 0.0}, r)[:n]
        out[:len(s)] += p['sub'] * pk * s / (np.abs(s).max() + 1e-12)
    if p['swell'] > 0.02:                                  # the build: the bloom's own hall, reversed into the hit
        k = dsp.n_of(p['swell'])
        ir = hall_ir(r, max(0.6, p['swell'] * 1.8), 0.3, 0.7, 0.0, seconds=p['swell'] * 1.3 + 0.2)
        sw = signal.fftconvolve(bloom[: dsp.n_of(0.25)], ir)[:k][::-1]
        sw *= np.linspace(0, 1, len(sw)) ** 2
        sw = dsp.fade(sw, 0.01, 0.002)
        sw = p['swell_level'] * pk * sw / (np.abs(sw).max() + 1e-12)
        out = np.concatenate([np.stack([sw, sw], axis=1), out])
    out = dsp.butter(out, 'high', p['low_cut'], 2 if p['low_cut'] < 60 else 4)
    out = dsp.trim_tail(out, -66, 0.05)
    return dsp.fade(out, 0.001, min(0.4, 0.15 * len(out) / SR))


# ------------------------------------------------------------------------------------------------ the launch idiom (r3)
# Round 3 (the judge: logo_sting was the worst category, t40 2.14 s against ~0.9 s, a bell chord where a launch film
# wants a composed reveal). One composed gesture in four short parts, each with its own band:
#   lead-in (a rising air whoosh / a rising zip chirp / the bloom's own reverse swell)  ->  the HIT on the logo frame
#   (a bright > 4 kHz tick of struck metal or glass + an optional light thump above the low cut)  ->  a SHORT bloom
#   (one note or a dyad of a modal / FM material, entering with the hit, a concave decay: a fast drop then a quiet ring)
#   + a granular shimmer of its top  ->  a short damped room and a clean fade (t40 ~0.7-1.1 s: the forge ref mk-3108_1
#   rings 0.86 s; kit-sting_tail_01's 2.06 s is a struck-metal tail, the end of a longer cue).
DYADS = {'one': [0], 'fifth': [0, 7], 'octave': [0, 12], 'fourth': [0, 5], 'ninth': [0, 14], 'third': [0, 4]}
LEADS = ['whoosh', 'chirp', 'swell', 'none']


def _lead_whoosh(p, n, r):
    """air pulled in: band-passed air noise (partly independent per channel) whose centre rises lead_f0 -> lead_f1,
    steepest at the end, while the level grows exponentially into the hit (a reversed whoosh), + an octave-up layer"""
    t = np.arange(n) / SR
    u = t / max(t[-1], 1e-6)
    fc = p['lead_f0'] * (p['lead_f1'] / p['lead_f0']) ** (u ** 1.7)
    lvl = 0.03 + 0.97 * (np.exp(4.5 * u) - 1) / (np.exp(4.5) - 1)
    out = np.zeros((n, 2))
    base = dsp.air(n / SR, r, center=2000.0, flutter=0.35, rate=7.0)[:n]
    for c in range(2):
        w = 0.5 * base + 0.5 * dsp.air(n / SR, r, center=2000.0, flutter=0.35, rate=7.0)[:n]
        y = dsp.svf(w, fc, 1.1, 'bandn') + 0.35 * dsp.svf(w, np.minimum(fc * 2.2, 16000), 0.8, 'bandn')
        out[:, c] = y * lvl
    out = dsp.fade(out, 0.03, 0.002)
    return out / (np.abs(out).max() + 1e-12)


def _lead_chirp(p, n, r):
    """a rising zip into the hit: an exponential sine sweep lead_f0 -> lead_f1 (study: zips at 20-60 oct/s), harmonics
    2 / 3, a crescendo (the loudest point is the arrival), mono"""
    t = np.arange(n) / SR
    u = t / max(t[-1], 1e-6)
    f = p['lead_f0'] * (p['lead_f1'] / p['lead_f0']) ** (u ** 1.3)
    ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 0.3)
    y = np.sin(ph) + 0.35 * np.sin(2 * ph) * (2 * f < SR * 0.4) + 0.15 * np.sin(3 * ph) * (3 * f < SR * 0.4)
    y *= 0.05 + 0.95 * u ** 2.2
    y = dsp.fade(y, 0.004, 0.0)
    y /= np.abs(y).max() + 1e-12
    return np.stack([y, y], axis=1)


def _lead_swell(p, n, r, bloom):
    """the bloom's own hall, reversed into the hit (the reverse-cymbal move), decorrelated"""
    from ..fx import decorrelate
    ir = hall_ir(r, max(0.6, p['lead_s'] * 1.8), 0.3, 0.7, 0.0, seconds=p['lead_s'] * 1.3 + 0.2)
    d = int(round(p.get('swell_trim', 0.0) * SR))          # r4: skip the hall's first ms (its build-up), which reversed
    sw = signal.fftconvolve(bloom[: dsp.n_of(0.25)], ir)[d:n + d][::-1]   # is a -20 dB dip just before the hit
    sw = sw * (0.03 + 0.97 * np.linspace(0, 1, len(sw)) ** 2)
    ld = dsp.fade(decorrelate(sw, 0.6), 0.03, 0.0)
    return ld / (np.abs(ld).max() + 1e-12)


def _tick(p, n, r):
    """the strike: a > tick_f noise burst with a (1 - u)^3 envelope over tick_ms (struck metal / glass: the refs' onset
    is almost all above 6 kHz) + a 0.4 ms contact pulse; peak 1"""
    L = min(n, max(16, dsp.n_of(p['tick_ms'] / 1000)))
    u = np.arange(L) / L
    w = r.standard_normal(L) * (1 - u) ** 3 * (1 - np.exp(-np.arange(L) / (SR * 0.0002)))
    w = dsp.butter(w, 'high', p['tick_f'], 2)
    y = np.zeros(n)
    y[:L] = w / (np.abs(w).max() + 1e-12)
    k = dsp.n_of(0.0004)
    y[:k] += 0.35 * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, k)))
    return y / (np.abs(y).max() + 1e-12)


def _thump3(p, n, r):
    """a light body under the hit: a sine falling an octave onto thump_f in ~30 ms, ~70 ms decay"""
    t = np.arange(n) / SR
    f = p['thump_f'] * (1 + np.exp(-t / 0.03))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07) * (1 - np.exp(-t / 0.001))
    return y / (np.abs(y).max() + 1e-12)


def _low_strike(p, n):
    """round 4 (the judge: the composed reveal has no body at all, < 150 Hz -52.7 dB of total; the techniques doc's hit
    on the logo frame is an impact with a sub): a sine falling from 2.5 x low_f onto low_f in ~35 ms with a 2nd
    harmonic at -12 dB (it survives small speakers), decaying with low_tau; mono, peak 1; it bypasses low_cut and the
    room (a low tone in a room rang through the tail in r3's first runs). No rng draw."""
    t = np.arange(n) / SR
    f = p['low_f'] * (1 + 1.5 * np.exp(-t / 0.035))
    ph = 2 * np.pi * np.cumsum(f) / SR
    rise = np.clip(t / 0.012, 0, 1) ** 2                 # blooms in over 12 ms under the tick (palette run 1: a 2 ms
    y = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * rise * np.exp(-t / p['low_tau'])   # rise read 'dull onset' -5..-12 dB)
    y = dsp.butter(y, 'high', 25.0, 2)
    return y / (np.abs(y).max() + 1e-12)


def _bloom_note(p, f, seconds, r):
    m = p['material']
    if m == 'fm':
        x = dsp.fm(f, 1.4, 2 + 4 * p['bright'], seconds, decay=10.0, index_decay=0.06, attack=0.002)
        x += 0.25 * dsp.fm(f * 2.0, 3.5, 1.5, seconds, decay=10.0, index_decay=0.03)
        return x / (np.abs(x).max() + 1e-12)
    parts = TRIANGLE if m == 'triangle' else m
    # partials decay with ratio^-slope around a long base decay; the shared concave envelope (below) sets the length
    return dsp.bell(f, seconds, r, parts, decay=1.2, slope=0.5 if m in ('triangle', 'glass') else 0.7,
                    bright=p['bright'], beat=1.5, strike=0.0)


def _short_bloom(p, n, r):
    """one note or a dyad (DYADS) of the material, flammed 6-14 ms, with a concave envelope: the struck level falls
    with strike_tau to ring_mix, then rings on quietly (ring_tau); an optional swell-in (bloom_att) while a low-pass
    opens from 1.5 kHz (the tonal bloom). Mono, peak 1."""
    seconds = n / SR
    y = np.zeros(n)
    for j, st in enumerate(DYADS[p['notes']]):
        f = p['root'] * 2 ** (st / 12)
        if f > 9500:
            continue
        x = _bloom_note(p, f, seconds, r)[:n] * (0.7 ** j)
        d = dsp.n_of(j * r.uniform(0.006, 0.014))
        y[d:] += x[:n - d]
    t = np.arange(n) / SR
    e = (1 - p['ring_mix']) * np.exp(-t / p['strike_tau']) + p['ring_mix'] * np.exp(-t / p['ring_tau'])
    if p['bloom_att'] > 0.003:
        a = t < p['bloom_att']
        e[a] *= 1 - (1 - t[a] / p['bloom_att']) ** 2         # (fast start: a raised cosine left a 25 dB gap after the tick)
        fc = 1500 + 16000 * (1 - np.exp(-t / max(p['bloom_att'] * 1.5, 0.005)))
        y = dsp.svf(y * e, fc, 0.6, 'low')
    else:
        y = y * e
    return y / (np.abs(y).max() + 1e-12)


def _render_launch(p, r):
    lead = p['lead'] if p['lead_level'] > 0.005 else 'none'     # (a 'none' lead ignores lead_level)
    n_lead = dsp.n_of(p['lead_s']) if lead != 'none' else 0
    n = dsp.n_of(min(4.0, max(0.6, p['ring_tau'] * 5.5, p['strike_tau'] * 6) + 0.3))
    bloom = _short_bloom(p, n, r)
    d = dsp.n_of(p.get('bloom_delay', 0.0))
    if d > 1:                                              # the bloom fades in over bloom_delay under the tick (a plain
        bloom = bloom.copy()                               # delay left a 10-15 ms dropout after a short tick: a
        bloom[:d] *= 1 - (1 - np.linspace(0, 1, d)) ** 3   # double hit; a raised cosine still dipped 20 dB after a
                                                           # 4 ms tick: this curve starts fast, -6 dB at 0.2 x delay)
    hit = p['tick'] * _tick(p, n, r)
    if p['thump'] > 0.01:
        hit = hit + p['thump'] * _thump3(p, n, r)
    body = np.zeros((n, 2))
    body += (p['ring'] * bloom + hit)[:, None]
    if p['width'] > 0.01:                                  # decorrelated bloom above 600 Hz (all-pass side signal)
        from ..fx import decorrelate
        hi = dsp.butter(bloom, 'high', 600.0, 2) * p['ring']
        body += decorrelate(hi, 0.8 * p['width']) - hi[:, None]
    if p['shimmer'] > 0.01:                                # granular sparkle of the bloom's top, an octave up, wide
        src = dsp.butter(bloom[: dsp.n_of(0.5)], 'band', (2500.0, 9000.0), 4)   # r3 run E: an unlimited top pitched
        sh = fx_granular(src, n / SR, r)                   # up an octave aliased (spark: > 20 kHz at -9 dB)
        t = np.arange(n) / SR
        sh *= ((1 - np.exp(-t / 0.02)) * np.exp(-t / p['shimmer_tau']))[:, None]
        body += p['shimmer'] * p['ring'] * sh / (np.abs(sh).max() + 1e-12)
    if p['verb'] > 0.01:                                   # a short damped room (the clean tail); it hears only what is
        send = dsp.butter(body, 'high', 500.0, 4)          # above 500 Hz (a thump in the room rang a 100-250 Hz band
        for c in range(2):                                 # through the whole tail, where the refs are empty)
            ir = hall_ir(r, p['verb_rt'], 0.6, 0.5, 0.008 + 0.003 * c)
            body[:, c] += p['verb'] * signal.fftconvolve(send[:, c], ir)[:n]
    pk = np.abs(body).max() + 1e-12
    if n_lead > 64:
        k = dsp.n_of(0.012)                                # the lead runs 12 ms into the hit, fading under it (a
        m = n_lead + k                                     # 2 ms fade before the hit printed a -60 dB dip)
        ld = (_lead_whoosh(p, m, r) if lead == 'whoosh' else _lead_chirp(p, m, r) if lead == 'chirp'
              else _lead_swell(p, m, r, bloom))
        ld[n_lead:] *= np.linspace(1, 0, k)[:, None] ** 2
        body = np.concatenate([np.zeros((n_lead, 2)), body])
        body[:n_lead + k] += p['lead_level'] * pk * ld
    out = dsp.butter(body, 'high', p['low_cut'], 2 if p['low_cut'] < 60 else 4)
    if p.get('low', 0) > 0.005:                            # round 4: the low strike, under the hit, after the low cut
        i = n_lead if n_lead > 64 else 0
        out[i:i + n] += (p['low'] * pk * _low_strike(p, n))[:, None]
    out = dsp.butter(out, 'low', 16500.0, 4)               # r3 run E: the noise tick / whoosh ran flat to 24 kHz (energy
                                                           # > 20 kHz at -17..-29 dB; recordings and the refs roll off)
    out = dsp.trim_tail(out, -62, 0.05)
    return dsp.fade(out, 0.0015, min(0.3, 0.2 * len(out) / SR))


def fx_granular(src, seconds, r):
    from ..fx import granular
    return granular(src, seconds, r, grain=0.05, density=50.0, pos=(0.0, 0.6), pos_jitter=0.05, pitch=12.0,
                    pitch_jitter=0.15, spread=0.9)


def render(p, r):
    mode = p.get('mode', DEFAULTS['mode'])
    p = {**{'chirp': _CHIRP, 'launch': _LAUNCH}.get(mode, _REVEAL), **p}
    y = _render_chirp(p, r) if mode == 'chirp' else _render_launch(p, r) if mode == 'launch' else _render_reveal(p, r)
    return y / (np.abs(y).max() + 1e-12) * 0.89


def space_for(cat):
    lim = _LIMITS.get(cat, {})
    return {k: lim.get(k, SPACE[k]) for k in _SEARCH[cat]}


def recipe_for(cat, preset=0):
    """a recipe object for one category and family preset (index or name); see impact.recipe_for"""
    fam = FAMILIES[cat]
    i = preset if isinstance(preset, int) else [n for n, _ in fam].index(preset)
    return types.SimpleNamespace(__name__=__name__, SPACE=space_for(cat), DEFAULTS=dict(fam[i][1]), render=render,
                                 TARGET=TARGETS[cat], ALTS=ALTS, REFSET=cat, PRESET=fam[i][0])
