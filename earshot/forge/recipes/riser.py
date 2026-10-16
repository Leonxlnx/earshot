"""A designed riser (uplifter / build) for motion-design edits: 0.8-4 s of rising tension that ends exactly on the hit.

Every sample is computed; no recorded audio is loaded, sampled or convolved. This is a DESIGNED sound, not a physical
model: it follows how pros build risers (docs/motion-sfx-techniques.md section 2) and what the pro films measurably do
(docs/motion-sfx-study.md, riser):
  * One motion leads, the others support it. Up to five layers, each owning a band:
      noise   white / pink noise under a resonant low-pass whose cutoff opens nf0 -> nf1 (the filter riser), high-passed
              so nothing piles up under 100-200 Hz (the cheap-riser mistake); decorrelated, its width grows to the peak
      tonal   a unison of 5 detuned voices (sine / tri / saw) gliding up `toct` octaves from tf0 on an exponential
              (equal octaves per second) curve, with an octave layer that fades in late; tmode 'blips' instead plays
              short percussive tone blips on the gate, each a small falling chirp, their pitch stepping up pulse by
              pulse: the stepped arpeggio of the pro stutter riser (ref_2: blips ~2 kHz -> 8-13 kHz)
      shepard octave-spaced sines gliding up under a fixed loudness bell (dsp.shepard): an endless rise
      drone   a steady low saw + sine (35-110 Hz, low-passed at ~450 Hz) under it all, swelling only a little
      verb    a stereo FDN reverb of the build, so the space grows with it (cut at the end, never after the hit)
  * Curves accelerate (concave in dB). Level, filter and pitch move on w = u^curve (u = t / length): with curve 2-3
    most of the change lands in the last third, as in all three pro risers. The level rises rise_db over the build.
  * The stutter gate. A gate whose period shrinks gate_p0 -> gate_p1 (pro: 24 -> 20 ms, 40-50 pulses/s; slower
    tremolo 1 -> 12 Hz for long builds), depth fading in from gate_on; gate_sharp blends a smooth tremolo into a hard
    gate with 1 ms ramps (no clicks). Its phase is laid out backwards from the end so the last pulse is open on the
    final sample: the riser stops on its peak.
  * The end. Hard: the loudest moment is the last ~50 ms, then an end_ms fade (2-20 ms). The file ends on the cut and
    hands over to the hit (study: risers end on the cut, 0-20 ms).
  * Glue: light tanh saturation on the sum, a 30 Hz high-pass, peak normalised by the forge.

CHARACTER parameters (what makes the variants of a family audibly different but one palette): length, curve, rise
depth, brightness (nf1, tbright), pitch range (tf0, toct), weight (drone, hp), texture (noise vs tonal vs shepard,
blips, gate). FAMILIES holds the hand-designed character presets the final runs start from; DEFAULTS is the 'air'
build.

Round 3 (after the round-2 judge: steady single lines for seconds, a chiptune pro set): `drone_oct` / `drone_noise`
(the drone glides up with the build, or is a low noise rumble), `metal_n` / `metal_glide` (the cymbal's metal moves),
tmode 'nblips' (stepped pulses of band-passed noise instead of tone blips); FAMILIES_BY_CAT['riser'] = round 2's picks
with those fixes, FAMILIES_BY_CAT['pro_riser'] = PRO_R3 (noise-led stutter builds; round 2's PRO kept for the record).

REFSET 'riser' is thin (one designed trailer riser in the forge split; the bench is two real cymbal swells); the
'pro_riser' set (one stutter riser from the pro films) is used for the short stutter family. The search runs in
designed mode, and `target_check()` adds the study's model-free targets to the critic (concave rise, ends on its
peak, high band rising, little energy under 100 Hz).
"""
import numpy as np
from scipy import signal

from .. import dsp, fx

TARGET = 'a cinematic riser building tension into a hit'
ALTS = ['a jet engine', 'an air raid siren', 'heavy rain on a roof', 'a vacuum cleaner']
REFSET = 'riser'

SPACE = {
    'length': (0.8, 4.0),                 # s, the hit lands on the last sample
    'curve': (1.0, 4.0),                  # acceleration of level / filter / pitch (1 = straight in dB)
    'rise_db': (12.0, 45.0),              # level rise over the build
    'noise_db': (-36.0, 0.0),             # layer levels (each layer normalised at its loudest)
    'nf0': (150.0, 1500.0, 'log'),        # noise low-pass: start ...
    'nf1': (2500.0, 16000.0, 'log'),      # ... and end cutoff (brightness at the peak)
    'nq': (0.7, 4.0, 'log'),              # resonance of the noise filter
    'pink': (0.0, 1.0),                   # 0 white (hissy) .. 1 pink (rounder)
    'tone_db': (-36.0, 0.0),
    'tmode': ['glide', 'blips', 'nblips'],
    'wave': ['sine', 'tri', 'saw'],
    'tf0': (60.0, 4000.0, 'log'),         # tonal start pitch (Hz)
    'toct': (0.5, 3.0),                   # octaves of rise
    'tbright': (1.5, 12.0, 'log'),        # saw low-pass = pitch x tbright
    'detune': (0.0, 30.0),                # unison spread in cents (widens as it rises)
    'oct_db': (-36.0, -3.0),              # +12 layer, fades in late
    'shep_db': (-48.0, 0.0),
    'shep_rate': (0.3, 1.6),              # octaves per second of the shepard glide
    'drone_db': (-48.0, -3.0),
    'drone_f': (35.0, 110.0, 'log'),
    'metal_db': (-48.0, 0.0),             # inharmonic shimmer partials (a reverse-cymbal wash)
    'metal_f': (1500.0, 5000.0, 'log'),   # lowest metal partial
    'gate_depth': (0.0, 1.0),
    'gate_on': (0.0, 0.7),                # fraction of the length where the gate fades in
    'gate_p0': (0.02, 0.5, 'log'),        # gate period at the start (s)
    'gate_p1': (0.012, 0.08, 'log'),      # gate period at the end (s)
    'gate_duty': (0.25, 0.8),
    'gate_sharp': (0.0, 1.0),             # 0 smooth tremolo .. 1 hard gate (1 ms ramps)
    'gate_drone': (0.0, 1.0),             # how much the gate also chops the drone
    'blip_dec': (0.1, 0.5),               # blips: decay time constant as a fraction of the pulse period
    'verb': (0.0, 0.6),
    'rt60': (0.6, 3.5, 'log'),
    'width': (0.0, 1.0),
    'sat': (0.0, 12.0),                   # tanh drive on the sum (dB)
    'hp': (60.0, 500.0, 'log'),           # high-pass of everything but the drone
    'end_ms': (2.0, 20.0, 'log'),         # final fade (ms): the end is a cut, not a ring-out
    # round 3 (the judge: no steady single line for seconds: riser_05's 2.5 kHz metal partial, riser_02's sub drone;
    # pro_riser reads as a chiptune arpeggio, AudioSet 'Music' 4/5)
    'drone_oct': (0.0, 1.5),              # the drone glides up this many octaves with the build (round 2: 0.08)
    'drone_noise': (0.0, 1.0),            # 0 saw + sine drone .. 1 a low noise rumble (50-400 Hz band, no line)
    'metal_n': (9, 40, 'int'),            # metal partials: 9 = round 2's fixed ratios; more = random dense ratios
    'metal_glide': (0.0, 1.0),            # the metal partials glide up this many octaves with the build
    'nb_q': (2.0, 16.0, 'log'),           # tmode 'nblips': Q of the noise band each blip is (filtered noise steps)
}

# character presets (round 1 hand design, then refined by the forge; see out/forge_v2/riser/LOG.md)
FAMILIES = {
    # a bright noise swell: the filter leads, a faint glide and growing space under it
    'air': {'length': 2.6, 'curve': 2.4, 'rise_db': 36.0, 'noise_db': 0.0, 'nf0': 300.0, 'nf1': 12000.0, 'nq': 1.6,
            'pink': 0.55, 'tone_db': -20.0, 'tmode': 'glide', 'wave': 'tri', 'tf0': 400.0, 'toct': 1.5, 'tbright': 4.0,
            'detune': 12.0, 'oct_db': -30.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -30.0, 'drone_f': 55.0,
            'gate_depth': 0.0, 'gate_on': 0.5, 'gate_p0': 0.2, 'gate_p1': 0.04, 'gate_duty': 0.5, 'gate_sharp': 0.0,
            'gate_drone': 0.0, 'verb': 0.3, 'rt60': 2.0, 'width': 0.7, 'sat': 2.0, 'hp': 180.0, 'end_ms': 6.0},
    # a supersaw tension build: pitch leads (+2 oct), tremolo speeding up 4 -> 16 Hz, noise and drone support
    'tension': {'length': 3.2, 'curve': 2.6, 'rise_db': 32.0, 'noise_db': -10.0, 'nf0': 250.0, 'nf1': 9000.0,
                'nq': 1.2, 'pink': 0.7, 'tone_db': 0.0, 'tmode': 'glide', 'wave': 'saw', 'tf0': 110.0, 'toct': 2.0,
                'tbright': 5.0, 'detune': 18.0, 'oct_db': -10.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -14.0,
                'drone_f': 55.0, 'gate_depth': 0.55, 'gate_on': 0.25, 'gate_p0': 0.25, 'gate_p1': 0.06,
                'gate_duty': 0.5, 'gate_sharp': 0.2, 'gate_drone': 0.0, 'verb': 0.3, 'rt60': 2.4, 'width': 0.5,
                'sat': 5.0, 'hp': 120.0, 'end_ms': 5.0},
    # the pro stutter riser (ref_2): stepped tone blips 2 -> 8 kHz on an accelerating 45/s gate over a gated drone
    'stutter': {'length': 1.1, 'curve': 2.0, 'rise_db': 26.0, 'noise_db': -14.0, 'nf0': 600.0, 'nf1': 12000.0,
                'nq': 1.0, 'pink': 0.3, 'tone_db': 0.0, 'tmode': 'blips', 'wave': 'sine', 'tf0': 1800.0, 'toct': 2.1,
                'tbright': 4.0, 'detune': 6.0, 'oct_db': -24.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -16.0,
                'drone_f': 95.0, 'gate_depth': 1.0, 'gate_on': 0.0, 'gate_p0': 0.05, 'gate_p1': 0.022,
                'gate_duty': 0.55, 'gate_sharp': 1.0, 'gate_drone': 0.8, 'verb': 0.12, 'rt60': 1.0, 'width': 0.3,
                'sat': 4.0, 'hp': 250.0, 'end_ms': 4.0},
    # an endless shepard rise in a big space, dark to bright, slow and wide
    'shepard': {'length': 3.8, 'curve': 1.8, 'rise_db': 30.0, 'noise_db': -12.0, 'nf0': 200.0, 'nf1': 7000.0,
                'nq': 2.2, 'pink': 0.9, 'tone_db': -36.0, 'tmode': 'glide', 'wave': 'sine', 'tf0': 300.0,
                'toct': 1.0, 'tbright': 3.0, 'detune': 8.0, 'oct_db': -36.0, 'shep_db': 0.0, 'shep_rate': 0.9,
                'drone_db': -22.0, 'drone_f': 45.0, 'gate_depth': 0.0, 'gate_on': 0.5, 'gate_p0': 0.2,
                'gate_p1': 0.04, 'gate_duty': 0.5, 'gate_sharp': 0.0, 'gate_drone': 0.0, 'verb': 0.45, 'rt60': 3.0,
                'width': 0.9, 'sat': 1.5, 'hp': 140.0, 'end_ms': 8.0},
    # a short punchy pitch riser: a sine-tri glide +3 oct with a hard gate rolling 12 -> 50 Hz, little noise
    'rush': {'length': 1.6, 'curve': 3.0, 'rise_db': 30.0, 'noise_db': -8.0, 'nf0': 400.0, 'nf1': 14000.0,
             'nq': 2.8, 'pink': 0.2, 'tone_db': -2.0, 'tmode': 'glide', 'wave': 'tri', 'tf0': 220.0, 'toct': 3.0,
             'tbright': 6.0, 'detune': 10.0, 'oct_db': -14.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -20.0,
             'drone_f': 82.0, 'gate_depth': 0.8, 'gate_on': 0.15, 'gate_p0': 0.085, 'gate_p1': 0.02,
             'gate_duty': 0.5, 'gate_sharp': 0.85, 'gate_drone': 0.3, 'verb': 0.18, 'rt60': 1.4, 'width': 0.4,
             'sat': 6.0, 'hp': 200.0, 'end_ms': 4.0},
}
for _f in FAMILIES.values():
    _f.setdefault('metal_db', -48.0)
    _f.setdefault('metal_f', 2600.0)
    _f.setdefault('blip_dec', 0.35)
    _f.update({'drone_oct': 0.08, 'drone_noise': 0.0, 'metal_n': 9, 'metal_glide': 0.0, 'nb_q': 6.0})
# round 3: 'rush' was a gated tri / saw glide that CLAP heard as 'a square wave chiptune blip' (z 1.7-2.6, 0/3 pass);
# now the noise leads (a resonant whistle opening 500 Hz -> 15 kHz) with a sine glide under it and a smooth flutter
# speeding up 10 -> 35 Hz
FAMILIES['rush'].update({'length': 1.5, 'curve': 3.2, 'noise_db': 0.0, 'nf0': 500.0, 'nf1': 15000.0, 'nq': 2.4,
                         'pink': 0.4, 'tone_db': -10.0, 'wave': 'sine', 'tf0': 260.0, 'toct': 2.8, 'oct_db': -20.0,
                         'gate_depth': 0.5, 'gate_on': 0.1, 'gate_p0': 0.1, 'gate_p1': 0.028, 'gate_sharp': 0.1,
                         'gate_drone': 0.0, 'drone_db': -22.0, 'sat': 2.0, 'verb': 0.22, 'width': 0.5})
# a reverse-cymbal swell: metallic inharmonic wash + bright noise, exponential, no tone (the bench riser kind)
FAMILIES['cymbal'] = {**FAMILIES['air'], 'length': 2.2, 'curve': 2.8, 'rise_db': 40.0, 'noise_db': -4.0, 'nf0': 900.0,
                      'nf1': 16000.0, 'nq': 0.8, 'pink': 0.25, 'tone_db': -36.0, 'oct_db': -36.0, 'drone_db': -48.0,
                      'metal_db': 0.0, 'metal_f': 2400.0, 'verb': 0.35, 'rt60': 1.6, 'width': 0.85, 'sat': 1.0,
                      'hp': 400.0, 'end_ms': 4.0}
# the pro stutter set (short builds, docs/motion-sfx-study.md riser: 0.8-1.5 s, gate 24 -> 20 ms, +2 oct tone, drone)
_ST = FAMILIES['stutter']
_ST.update({'length': 0.9, 'curve': 1.6, 'rise_db': 16.0, 'noise_db': -20.0, 'tf0': 1700.0, 'toct': 2.6,
            'gate_p0': 0.055, 'gate_p1': 0.036, 'gate_duty': 0.5, 'drone_db': -12.0, 'gate_drone': 1.0, 'verb': 0.08,
            'blip_dec': 0.2})
PRO = {
    'blips': dict(_ST),
    # gated noise: a filter sweep chopped by the same accelerating 45/s gate, a thin tone on top
    'chop': {**_ST, 'length': 1.0, 'tmode': 'glide', 'wave': 'tri', 'tone_db': -14.0, 'noise_db': 0.0, 'nf0': 700.0,
             'nf1': 14000.0, 'nq': 2.0, 'gate_p0': 0.04, 'gate_p1': 0.02, 'gate_duty': 0.45, 'drone_db': -20.0,
             'hp': 500.0, 'rise_db': 26.0, 'curve': 2.2, 'verb': 0.05},
    # a saw glide under a hard accelerating gate, darker, longer
    'gated_saw': {**_ST, 'length': 1.4, 'tmode': 'glide', 'wave': 'saw', 'tf0': 300.0, 'toct': 2.0, 'tbright': 3.0,
                  'detune': 14.0, 'noise_db': -12.0, 'gate_p0': 0.06, 'gate_p1': 0.022, 'gate_duty': 0.6,
                  'drone_db': -14.0, 'drone_f': 60.0, 'hp': 150.0, 'sat': 3.0, 'rise_db': 26.0, 'curve': 2.2,
                  'verb': 0.06},
    # a roll: noise bursts speeding up 12 -> 45 per second (a snare-roll build), no tone
    'roll': {**_ST, 'length': 1.3, 'curve': 2.4, 'tmode': 'glide', 'tone_db': -36.0, 'oct_db': -36.0, 'noise_db': 0.0,
             'nf0': 1200.0, 'nf1': 12000.0, 'nq': 0.9, 'pink': 0.6, 'gate_p0': 0.085, 'gate_p1': 0.022,
             'gate_duty': 0.4, 'gate_sharp': 1.0, 'drone_db': -24.0, 'gate_drone': 1.0, 'hp': 300.0, 'verb': 0.1,
             'rise_db': 28.0},
    # the shortest: 0.8 s, higher blips 3 -> 12 kHz, faster gate
    'blips_hi': {**_ST, 'length': 0.8, 'tf0': 2600.0, 'toct': 2.2, 'gate_p0': 0.045, 'gate_p1': 0.028,
                 'drone_db': -18.0, 'drone_f': 110.0, 'noise_db': -14.0, 'rise_db': 20.0},
}
FAMILIES_BY_CAT = {'riser': {k: FAMILIES[k] for k in ('air', 'tension', 'shepard', 'rush', 'cymbal')},
                   'pro_riser': PRO}
TARGETS = {'pro_riser': 'a short stuttering digital riser into a cut'}
DEFAULTS = dict(FAMILIES['air'])

# ------------------------------------------------------------------------------------------------ round 3 presets
# riser: the round-2 picks (out/forge_v2/riser/final_r2/report.json) with every steady line made to move: the drones
# glide up with the build (or become a low noise rumble), the cymbal's metal is 28 dense partials gliding up
_R2 = {
    'air': {'length': 2.47, 'curve': 2.043, 'rise_db': 33.76, 'noise_db': -2.34, 'nf0': 294.7, 'nf1': 10379.0,
            'nq': 1.856, 'pink': 0.646, 'tone_db': -22.34, 'tmode': 'glide', 'wave': 'tri', 'tf0': 440.8, 'toct': 1.225,
            'tbright': 3.63, 'detune': 13.64, 'oct_db': -29.46, 'shep_db': -43.4, 'shep_rate': 0.778,
            'drone_db': -32.06, 'drone_f': 50.92, 'metal_db': -46.76, 'metal_f': 2615.8, 'gate_depth': 0.039,
            'gate_on': 0.584, 'gate_p0': 0.173, 'gate_p1': 0.041, 'gate_duty': 0.506, 'gate_sharp': 0.058,
            'gate_drone': 0.103, 'blip_dec': 0.35, 'verb': 0.259, 'rt60': 1.765, 'width': 0.586, 'sat': 0.861,
            'hp': 193.6, 'end_ms': 5.46},
    'tension': {'length': 2.816, 'curve': 2.681, 'rise_db': 35.24, 'noise_db': -9.59, 'nf0': 214.0, 'nf1': 7710.6,
                'nq': 1.253, 'pink': 0.814, 'tone_db': -3.34, 'tmode': 'glide', 'wave': 'saw', 'tf0': 76.65,
                'toct': 1.767, 'tbright': 6.06, 'detune': 20.45, 'oct_db': -10.69, 'shep_db': -45.8,
                'shep_rate': 0.765, 'drone_db': -16.86, 'drone_f': 53.29, 'metal_db': -45.95, 'metal_f': 2641.2,
                'gate_depth': 0.558, 'gate_on': 0.264, 'gate_p0': 0.254, 'gate_p1': 0.058, 'gate_duty': 0.493,
                'gate_sharp': 0.213, 'gate_drone': 0.052, 'blip_dec': 0.35, 'verb': 0.286, 'rt60': 2.495,
                'width': 0.522, 'sat': 5.94, 'hp': 109.6, 'end_ms': 5.56},
    'shepard': {'length': 3.708, 'curve': 2.026, 'rise_db': 30.53, 'noise_db': -8.98, 'nf0': 206.5, 'nf1': 8656.4,
                'nq': 2.572, 'pink': 0.974, 'tone_db': -35.18, 'tmode': 'glide', 'wave': 'sine', 'tf0': 260.0,
                'toct': 1.046, 'tbright': 2.34, 'detune': 9.42, 'oct_db': -32.85, 'shep_db': -4.73,
                'shep_rate': 0.887, 'drone_db': -22.49, 'drone_f': 51.44, 'metal_db': -44.93, 'metal_f': 2843.6,
                'gate_depth': 0.048, 'gate_on': 0.578, 'gate_p0': 0.173, 'gate_p1': 0.049, 'gate_duty': 0.483,
                'gate_sharp': 0.018, 'gate_drone': 0.098, 'blip_dec': 0.35, 'verb': 0.415, 'rt60': 2.889,
                'width': 0.909, 'sat': 0.291, 'hp': 108.6, 'end_ms': 10.47},
    'rush': {'length': 1.756, 'curve': 3.117, 'rise_db': 32.74, 'noise_db': -4.32, 'nf0': 468.2, 'nf1': 12035.0,
             'nq': 2.54, 'pink': 0.344, 'tone_db': -8.74, 'tmode': 'glide', 'wave': 'sine', 'tf0': 397.2,
             'toct': 2.516, 'tbright': 5.63, 'detune': 9.84, 'oct_db': -19.99, 'shep_db': -42.3, 'shep_rate': 0.889,
             'drone_db': -24.66, 'drone_f': 76.65, 'metal_db': -43.25, 'metal_f': 2909.5, 'gate_depth': 0.497,
             'gate_on': 0.074, 'gate_p0': 0.146, 'gate_p1': 0.026, 'gate_duty': 0.458, 'gate_sharp': 0.188,
             'gate_drone': 0.092, 'blip_dec': 0.364, 'verb': 0.292, 'rt60': 1.687, 'width': 0.565, 'sat': 3.2,
             'hp': 171.7, 'end_ms': 3.32},
    'cymbal': {'length': 2.531, 'curve': 2.469, 'rise_db': 41.5, 'noise_db': -3.21, 'nf0': 730.3, 'nf1': 15030.0,
               'nq': 0.704, 'pink': 0.312, 'tone_db': -33.79, 'tmode': 'glide', 'wave': 'tri', 'tf0': 542.9,
               'toct': 1.24, 'tbright': 3.7, 'detune': 11.5, 'oct_db': -32.17, 'shep_db': -44.76, 'shep_rate': 0.725,
               'drone_db': -46.7, 'drone_f': 61.19, 'metal_db': -4.46, 'metal_f': 2153.3, 'gate_depth': 0.035,
               'gate_on': 0.514, 'gate_p0': 0.209, 'gate_p1': 0.046, 'gate_duty': 0.508, 'gate_sharp': 0.035,
               'gate_drone': 0.05, 'blip_dec': 0.35, 'verb': 0.396, 'rt60': 1.688, 'width': 0.96, 'sat': 0.901,
               'hp': 403.8, 'end_ms': 4.21},
}
_R3_FIX = {
    'air': dict(drone_oct=0.9, drone_noise=0.7),
    'tension': dict(drone_oct=1.0, drone_noise=0.5, drone_db=-20.0),
    'shepard': dict(drone_oct=0.7, drone_noise=0.8),
    'rush': dict(drone_oct=0.6, drone_noise=0.3, drone_db=-28.0),
    # the cymbal: round 2's 9 metal ratios, now gliding up 0.6 oct (28 dense random partials were tried first: the
    # description z fell to 2.9 against 3.3-3.45 for the 9 gliding ratios)
    'cymbal': dict(metal_n=9, metal_glide=0.6, metal_db=-6.0, drone_db=-48.0, width=0.75),
}
FAMILIES_BY_CAT['riser'] = {k: {**FAMILIES[k], **_R2[k], **_R3_FIX[k]} for k in _R2}
# Round 4 (ui-tonal polish; the round-3 judge: AudioSet hears engines, 'Car' / 'Jet engine', and the bench's closest
# phrase was 'a jet engine starting' for all five): what reads as an engine is a low rumble that revs up (the round-3
# drones glide up 0.6-1.0 oct as 30-80 % noise), a saw starting at 77 Hz under 6 dB of drive (tension), a resonant
# low-pass sweep (jet spool-up) and metal partials gliding together (turbine whine). Edits on the round-3 delivered
# members (applied by out/forge_v2/_uitonal_r4/fam4.py; order air, tension, shepard, rush, cymbal): no drone, the
# high-pass at 180-250 Hz, gentler noise resonance, tension's supersaw from 150 Hz (not 77) with less drive, a wider
# unison and its noise layer up (a triangle from 220 Hz, tried first, read as a siren: critic 3.65 -> 1.86), the
# cymbal's metal glide 0.6 -> 0.2 oct (0.35 brought the turbine back: AudioSet 'Jet engine' first again, and its
# longest steady line 0.58 s vs 0.52 s at 0.2), and every curve concave (2.2-3.1; tension's
# 2.8 held its saw near one pitch for 0.85 s, a steady line, so 2.2 with a 1.8 oct rise).
R4_EDITS = {'riser': [
    dict(drone_db=-48.0, hp=220.0, nq=1.2, curve=2.6),
    dict(drone_db=-48.0, tf0=150.0, toct=1.8, hp=200.0, sat=3.0, detune=24.0, noise_db=-5.0, curve=2.2),
    dict(drone_db=-48.0, hp=180.0, curve=2.4),
    dict(drone_db=-48.0, hp=220.0, nq=1.6),
    dict(drone_db=-48.0, metal_glide=0.2),
]}
# pro_riser: noise-led stutter builds (round 2's tone blips and gated saw read as a chiptune arpeggio): the stepped
# rising gesture is now carried by filtered noise (band-passed noise blips stepping up), the drones are gated noise
# rumbles, the gate still accelerates into the cut
PRO_R3 = {
    'nblips': {**_ST, 'length': 1.0, 'tmode': 'nblips', 'nb_q': 7.0, 'tf0': 1700.0, 'toct': 2.4, 'tone_db': 0.0,
               'noise_db': -10.0, 'nf0': 900.0, 'nf1': 13000.0, 'nq': 1.2, 'gate_p0': 0.05, 'gate_p1': 0.024,
               'gate_duty': 0.5, 'blip_dec': 0.25, 'drone_db': -16.0, 'drone_noise': 1.0, 'drone_oct': 0.3,
               'gate_drone': 1.0, 'oct_db': -20.0, 'rise_db': 20.0, 'curve': 1.8, 'verb': 0.03, 'sat': 2.0,
               'metal_n': 9, 'metal_glide': 0.0},
    'chop': {**PRO['chop'], 'tone_db': -24.0, 'drone_noise': 1.0, 'drone_oct': 0.3, 'nq': 2.4, 'gate_p0': 0.045,
             'gate_p1': 0.02, 'sat': 2.0, 'drone_db': -18.0, 'metal_n': 9, 'metal_glide': 0.0, 'nb_q': 6.0},
    'roll': {**PRO['roll'], 'drone_noise': 1.0, 'drone_oct': 0.3, 'metal_n': 9, 'metal_glide': 0.0, 'nb_q': 6.0},
    'air_gate': {**_ST, 'length': 1.2, 'tmode': 'glide', 'wave': 'saw', 'tf0': 400.0, 'toct': 2.0, 'tbright': 2.0,
                 'tone_db': -16.0, 'noise_db': 0.0, 'nf0': 500.0, 'nf1': 12000.0, 'nq': 3.0, 'pink': 0.5,
                 'gate_depth': 0.85, 'gate_on': 0.1, 'gate_p0': 0.07, 'gate_p1': 0.022, 'gate_duty': 0.55,
                 'gate_sharp': 0.6, 'drone_db': -18.0, 'drone_noise': 1.0, 'drone_oct': 0.5, 'gate_drone': 0.8,
                 'hp': 200.0, 'sat': 2.0, 'rise_db': 26.0, 'curve': 2.2, 'verb': 0.12, 'width': 0.4,
                 'metal_n': 9, 'metal_glide': 0.0, 'nb_q': 6.0},     # a resonant noise whistle under a gate
    'nblips_hi': {**_ST, 'length': 0.85, 'tmode': 'nblips', 'nb_q': 10.0, 'tf0': 2600.0, 'toct': 2.1,
                  'tone_db': 0.0, 'noise_db': -12.0, 'nf0': 1500.0, 'nf1': 14000.0, 'gate_p0': 0.045,
                  'gate_p1': 0.026, 'blip_dec': 0.22, 'drone_db': -20.0, 'drone_f': 110.0, 'drone_noise': 1.0,
                  'drone_oct': 0.3, 'gate_drone': 1.0, 'rise_db': 20.0, 'verb': 0.03, 'sat': 2.0, 'metal_n': 9,
                  'metal_glide': 0.0},
}
FAMILIES_BY_CAT['pro_riser'] = PRO_R3
# (round 2's pro set stays in PRO for the record; its picks re-render from out/forge_v2/pro_riser/final_r2/)

# ------------------------------------------------------------------------------------------------ round 6 'build'
# The round-5 judge: CLAP's nearest confusion is still 'a jet engine starting' for all five round-4 risers. What a jet
# spool-up is, acoustically: broadband noise under a resonant low-pass that opens from a low rumble, steady (no pulse),
# with a turbine whine gliding with it, and it does not stop. The round-4 members still start as low-passed noise at
# 200-470 Hz (a rumble), keep a continuous noise bed with no rhythm (gate depth 0.04 in air / shepard / cymbal) and
# glide metal / saw partials together. The 'build' render (p['r6'] = 1; render() is unchanged without it) follows the
# study instead (docs/motion-sfx-study.md riser; techniques section 2):
#   noise   a BAND that moves up (low-pass nf0 -> nf1 and a high-pass nbw octaves under it, both 24 dB/oct and
#           unresonant: no whistle, no rumble at the start), high-passed at hp >= 250 Hz
#   tonal   a unison chord (root + fifth, + a late octave) of detuned voices in the mid / high register (tf0 300-1500
#           Hz), rising toct (about 2) octaves on a concave curve: the 'riser' pitch cue an engine does not have
#   gate    an accelerating gate on everything (period gate_p0 -> gate_p1, e.g. 8 Hz -> 45 Hz, pro 24 -> 20 ms at the
#           end), depth fading in, laid out from the end so the last pulse is open: rhythm an engine does not have
#   swell   rswell_db: a reversed computed reverb tail of a bright noise burst (the reverse-cymbal build) under it
#   end     the loudest moment is the last ~45 ms, then an end_ms cut: it stops on the hit (an engine keeps going)
# No drone, no sub, no metal partials, no saturation of a low layer.
SPACE_R6 = {   # extra keys of the build render (not in SPACE: a pro_riser search must not mid-range-fill them)
    'r6': (0.0, 1.0),                    # 1 = the build render
    'nbw': (0.8, 4.0),                   # noise band width in octaves (moving high-pass at fc / 2^nbw)
    'fifth_db': (-48.0, 0.0),            # the fifth (x1.5) of the tonal chord, re the root
    'tone_gate': (0.0, 1.0),             # how much the gate chops the tonal layer (noise: always)
    'gate_pow': (0.6, 3.0),              # how late the gate accelerates (period on u^gate_pow)
    'rswell_db': (-48.0, 0.0),           # reverse reverb swell layer
    'rswell_rt': (1.0, 4.0),             # its decay (s)
    'pre_dip_ms': (0.0, 60.0),           # a short duck just before the last pulse (the 'suck' into the hit); 0 = none
}
R6_OFF = {'r6': 0.0, 'nbw': 2.0, 'fifth_db': -48.0, 'tone_gate': 1.0, 'gate_pow': 1.4, 'rswell_db': -48.0,
          'rswell_rt': 2.0, 'pre_dip_ms': 0.0}
_B6 = {**FAMILIES['air'], 'metal_db': -48.0, 'metal_f': 2600.0, 'blip_dec': 0.35, 'drone_oct': 0.0,
       'drone_noise': 0.0, 'metal_n': 9, 'metal_glide': 0.0, 'nb_q': 6.0, 'drone_db': -48.0, 'drone_f': 55.0,
       'shep_db': -48.0, 'gate_drone': 0.0, 'tmode': 'glide', 'sat': 1.0, 'nq': 0.7, **R6_OFF, 'r6': 1.0}
R6 = {
    # a noise band opening up under a fast-accelerating tremolo (4 -> 30 Hz), a faint tri chord rising 2 oct
    'air': {**_B6, 'length': 2.4, 'curve': 2.6, 'rise_db': 34.0, 'noise_db': 0.0, 'nf0': 900.0, 'nf1': 13000.0,
            'nbw': 2.2, 'pink': 0.5, 'tone_db': -16.0, 'fifth_db': -8.0, 'wave': 'tri', 'tf0': 700.0, 'toct': 2.0,
            'detune': 14.0, 'oct_db': -26.0, 'gate_depth': 0.6, 'gate_on': 0.15, 'gate_p0': 0.25, 'gate_p1': 0.034,
            'gate_duty': 0.55, 'gate_sharp': 0.3, 'gate_pow': 1.6, 'tone_gate': 0.6, 'verb': 0.3, 'rt60': 2.0,
            'width': 0.6, 'hp': 300.0, 'end_ms': 5.0, 'rswell_db': -14.0, 'rswell_rt': 2.2},
    # pitch leads: a detuned saw chord 300 Hz -> 1.2 kHz, chopped by a gate rolling 6 -> 40 Hz, noise supports
    'tension': {**_B6, 'length': 2.8, 'curve': 2.4, 'rise_db': 30.0, 'noise_db': -9.0, 'nf0': 1200.0,
                'nf1': 11000.0, 'nbw': 1.8, 'pink': 0.6, 'tone_db': 0.0, 'fifth_db': -4.0, 'wave': 'saw',
                'tf0': 300.0, 'toct': 2.0, 'tbright': 4.0, 'detune': 22.0, 'oct_db': -12.0, 'gate_depth': 0.75,
                'gate_on': 0.1, 'gate_p0': 0.17, 'gate_p1': 0.025, 'gate_duty': 0.5, 'gate_sharp': 0.55,
                'gate_pow': 1.5, 'tone_gate': 1.0, 'verb': 0.25, 'rt60': 2.0, 'width': 0.5, 'hp': 260.0,
                'end_ms': 4.0, 'rswell_db': -48.0},
    # an endless shepard rise over a soft noise band, a slow tremolo speeding up 2 -> 14 Hz, big space
    'shepard': {**_B6, 'length': 3.4, 'curve': 2.2, 'rise_db': 30.0, 'noise_db': -8.0, 'nf0': 700.0, 'nf1': 9000.0,
                'nbw': 2.6, 'pink': 0.9, 'tone_db': -36.0, 'fifth_db': -48.0, 'oct_db': -36.0, 'shep_db': 0.0,
                'shep_rate': 0.9, 'gate_depth': 0.45, 'gate_on': 0.2, 'gate_p0': 0.5, 'gate_p1': 0.07,
                'gate_duty': 0.5, 'gate_sharp': 0.0, 'gate_pow': 1.3, 'tone_gate': 0.8, 'verb': 0.45, 'rt60': 3.0,
                'width': 0.85, 'hp': 260.0, 'end_ms': 8.0, 'rswell_db': -20.0, 'rswell_rt': 3.0},
    # the pro build, short: a sine chord 1 -> 4.5 kHz and a bright band under the pro stutter (24 -> 20 ms at the end)
    'rush': {**_B6, 'length': 1.3, 'curve': 2.2, 'rise_db': 26.0, 'noise_db': -3.0, 'nf0': 1500.0, 'nf1': 15000.0,
             'nbw': 1.6, 'pink': 0.35, 'tone_db': -4.0, 'fifth_db': -6.0, 'wave': 'sine', 'tf0': 1000.0,
             'toct': 2.2, 'detune': 10.0, 'oct_db': -18.0, 'gate_depth': 0.85, 'gate_on': 0.0, 'gate_p0': 0.06,
             'gate_p1': 0.021, 'gate_duty': 0.5, 'gate_sharp': 0.8, 'gate_pow': 1.2, 'tone_gate': 1.0,
             'verb': 0.15, 'rt60': 1.2, 'width': 0.4, 'hp': 400.0, 'end_ms': 3.0, 'rswell_db': -48.0},
    # the reverse-cymbal build: a reversed computed reverb tail of a bright burst, a bright band, a thin high chord
    'swell': {**_B6, 'length': 2.2, 'curve': 2.8, 'rise_db': 38.0, 'noise_db': -8.0, 'nf0': 2000.0,
              'nf1': 16000.0, 'nbw': 1.4, 'pink': 0.3, 'tone_db': -22.0, 'fifth_db': -6.0, 'wave': 'sine',
              'tf0': 1500.0, 'toct': 1.6, 'detune': 8.0, 'oct_db': -30.0, 'gate_depth': 0.35, 'gate_on': 0.5,
              'gate_p0': 0.12, 'gate_p1': 0.03, 'gate_duty': 0.5, 'gate_sharp': 0.2, 'gate_pow': 2.0,
              'tone_gate': 0.5, 'verb': 0.2, 'rt60': 1.6, 'width': 0.8, 'hp': 500.0, 'end_ms': 4.0,
              'rswell_db': 0.0, 'rswell_rt': 2.6},
}


def _lp24(x, fc):
    return dsp.svf(dsp.svf(x, fc, 0.54, 'low'), fc, 1.31, 'low')      # Butterworth-4 pole pair: no peak


def _hp24(x, fc):
    return dsp.svf(dsp.svf(x, fc, 0.54, 'high'), fc, 1.31, 'high')


def render_build(p, r):
    """round 6: the 'build' riser (see the R6 note above). Stereo, ends on its peak."""
    p = {**DEFAULTS, **R6_OFF, **p}
    L = float(p['length'])
    n = dsp.n_of(L)
    t = dsp.t_of(n)
    u = t / L
    w = u ** p['curve']
    lv = 10 ** (-p['rise_db'] * (1 - w) / 20)
    # the accelerating gate, laid out from the end (the last pulse open on the final sample)
    per = p['gate_p0'] * (p['gate_p1'] / p['gate_p0']) ** (u ** p['gate_pow'])
    ph = phase_from_end(1.0 / per, end_phase=0.5 * p['gate_duty'])
    g = gate_shape(ph, per, p['gate_duty'], p['gate_sharp'])
    on = np.clip((u - p['gate_on']) / 0.2, 0, 1)
    fin = np.clip((L - t) / FINAL_OPEN, 0, 1)
    fin = fin * fin * (3 - 2 * fin)
    depth = p['gate_depth'] * (on * on * (3 - 2 * on)) * fin
    gate_n = 1 - depth * (1 - g)
    gate_t = 1 - p['tone_gate'] * depth * (1 - g)
    # noise: an unresonant band moving up
    wn = (1 - p['pink']) * r.standard_normal(n) + p['pink'] * dsp.noise(L, r, 'pink')[:n] * 1.6
    fc = p['nf0'] * (p['nf1'] / p['nf0']) ** w
    nz = _hp24(_lp24(wn, np.minimum(fc, 18000.0)), np.maximum(fc / 2 ** p['nbw'], 20.0))
    nz = dsp.butter(nz, 'high', p['hp'], order=4)
    nz_st = fx.width(fx.decorrelate(nz, 1.0), p['width'] * (0.35 + 0.65 * w))
    # tonal chord: root + fifth (+ a late octave), concave glide
    wt = u ** max(1.0, 0.8 * p['curve'])
    f = p['tf0'] * 2 ** (p['toct'] * wt)
    ph0 = r.uniform(0, 1, 15)
    det = p['detune'] * (0.6 + 0.8 * w)
    tone = unison(f, p['wave'], det, 0.5 * p['width'], ph0[:5])
    if p['fifth_db'] > -47:
        tone = tone + 10 ** (p['fifth_db'] / 20) * unison(1.5 * f, p['wave'], det, 0.7 * p['width'], ph0[5:10])
    octv = unison(2 * f, p['wave'], det * 0.7, 0.6 * p['width'], ph0[10:])
    octv *= np.clip((u - 0.4) / 0.6, 0, 1)[:, None] ** 2
    if p['wave'] == 'saw':
        cut = np.minimum(f * p['tbright'], 18000)
        tone = np.stack([dsp.svf(tone[:, c], cut, 0.7, 'low') for c in range(2)], axis=1)
        octv = np.stack([dsp.svf(octv[:, c], np.minimum(2 * cut, 18000), 0.7, 'low') for c in range(2)], axis=1)
    tone = dsp.butter(tone, 'high', p['hp'] * 0.8, order=2)
    octv = dsp.butter(octv, 'high', p['hp'], order=2)
    # shepard (optional)
    if p['shep_db'] > -47:
        sh = dsp.shepard(L, r, up=True, rate=p['shep_rate'], f_lo=150.0, f_hi=9000.0, kind='tone')[:n]
        sh = dsp.butter(sh, 'high', p['hp'], order=2)
        sh_st = fx.width(fx.decorrelate(sh, 0.6), 0.6 * p['width'])
    else:
        sh_st = None
    # reverse swell: the reversed reverb tail of a bright burst (computed), concave by nature
    if p['rswell_db'] > -47:
        hb = dsp.butter(r.standard_normal(dsp.n_of(0.03)), 'high', 1500.0, order=2) * np.exp(
            -dsp.t_of(dsp.n_of(0.03)) / 0.008)
        wet = wet_of(hb, p['rswell_rt'], dsp.n_of(p['rswell_rt'] + 0.1), predelay=0.0, damp=0.3, lowcut=400.0)
        k = min(n, len(wet))
        sw = np.zeros((n, 2))
        sw[n - k:] = wet[:k][::-1]
        sw = fx.width(dsp.butter(sw, 'high', p['hp'], order=2), 0.25 + 0.5 * p['width'])   # mostly centred
    else:
        sw = None

    def at(db, x):
        return 10 ** (db / 20) * x / lvl(x, 0.25)

    noise_l = at(p['noise_db'], nz_st) * gate_n[:, None]
    tone_l = (at(p['tone_db'], tone) + at(p['oct_db'], octv)) * gate_t[:, None]
    send = noise_l + tone_l
    if sh_st is not None:
        send = send + at(p['shep_db'], sh_st) * gate_t[:, None]
    send = send * lv[:, None]
    if sw is not None:                                   # the swell carries its own (reversed-decay) curve
        send = send + at(p['rswell_db'], sw) * (0.25 + 0.75 * gate_n[:, None])
    mix = send
    if p['verb'] > 0.005:
        wet = wet_of(send, p['rt60'], n)
        mix = mix + p['verb'] * wet / lvl(wet, 0.25) * lvl(send, 0.25)
    if p['pre_dip_ms'] > 0.5:                            # a short duck just before the last open pulse
        d0 = L - FINAL_OPEN - p['pre_dip_ms'] * 1e-3
        dip = 1 - 0.5 * np.exp(-0.5 * ((t - d0) / (0.35 * p['pre_dip_ms'] * 1e-3)) ** 2)
        mix = mix * dip[:, None]
    if p['sat'] > 0.1:
        mix = fx.saturate(mix, drive_db=p['sat'], mode='tanh')
    mix = dsp.butter(mix, 'high', 120.0, order=2)
    mix = dsp.butter(mix, 'low', 18000, order=4)
    return dsp.fade(mix, 0.004, p['end_ms'] / 1000.0)

UNISON = (-1.0, -0.45, 0.0, 0.45, 1.0)
FINAL_OPEN = 0.045    # s: the gate opens (and the blips sustain) over the last 45 ms, so the riser peaks on the cut     # voice offsets (x detune cents) and pans (x width)


# ------------------------------------------------------------------------------------------------ shared helpers
# (downlifter.py and zip.py import these; kept here because recipes may not change the core modules)
def lvl(x, frac=None):
    """peak of a 10 ms RMS envelope (frac: only the last `frac` of the sound): a layer's level for mixing"""
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    if frac:
        m = m[int(len(m) * (1 - frac)):]
    h = dsp.n_of(0.01)
    k = max(1, len(m) // h)
    return float(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)).max()) + 1e-12


def phase_from_end(rate, end_phase=0.0):
    """cycles (float) of a pulse train with a per-sample rate (Hz), laid out backwards so phase(end) = end_phase"""
    c = np.cumsum(rate[::-1])[::-1] / dsp.SR             # cycles from each sample to the end
    return end_phase - c


def gate_shape(phase, period, duty=0.5, sharp=1.0, ramp=0.0012, perc=0.0):
    """one value per sample in 0..1: a smooth tremolo (sharp 0) .. a hard gate with `ramp` s edges (sharp 1);
    perc > 0 makes each pulse a percussive blip decaying with time constant perc x period"""
    fr = np.mod(phase, 1.0)
    trem = 0.5 + 0.5 * np.cos(2 * np.pi * (fr - 0.5 * duty))
    on = np.clip(fr * period / ramp, 0, 1)
    off = np.clip((duty - fr) * period / (ramp * 1.5), 0, 1)
    sq = np.minimum(on, off)
    if perc > 0:
        sq = on * np.exp(-fr / max(perc, 1e-3)) * np.clip((0.98 - fr) * period / ramp, 0, 1)
    return (1 - sharp) * trem + sharp * sq


def unison(freq, wave, detune, width, phases):
    """5 detuned voices (cents offsets UNISON x detune, an array over time allowed), panned UNISON x width: stereo"""
    out = np.zeros((len(freq), 2))
    for k, off in enumerate(UNISON):
        f = freq * 2 ** (off * detune / 1200.0)
        v = dsp.osc(f, wave, phase=phases[k])
        th = (np.clip(off * width, -1, 1) + 1) * np.pi / 4
        out[:, 0] += v * np.cos(th)
        out[:, 1] += v * np.sin(th)
    return out * np.sqrt(2) / len(UNISON)


_IR = {}


def verb_ir(rt60, size=0.75, predelay=0.02, damp=0.5, lowcut=200.0):
    """a cached stereo FDN reverb IR (fx.reverb_ir on a fixed rng, rt60 rounded to 0.1 s): building the IR costs ~1 s,
    convolving it ~0.1 s, so a search re-uses one IR per decay time (the noise seeds still vary every render)"""
    key = (round(float(rt60), 1), round(size, 2), round(predelay, 3), round(damp, 2), round(lowcut))
    if key not in _IR:
        _IR[key] = fx.reverb_ir(dsp.rng(1234), key[0], size, predelay, damp, inputs=2, lowcut=lowcut)
    return _IR[key]


def wet_of(x, rt60, n_out=None, **kw):
    """the reverb return alone (stereo, unit-energy IR) of x, cut to n_out samples (default len(x) + the tail)"""
    ir = verb_ir(rt60, **kw)
    x2 = stereo_of(x)
    n_out = n_out or len(x2) + len(ir) - 1
    out = np.zeros((len(x2) + len(ir) - 1, 2))
    for c in range(2):
        out += signal.fftconvolve(x2[:, c:c + 1], ir[:, c, :], axes=0)
    return out[:n_out]


def stereo_of(x):
    x = np.asarray(x, np.float64)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def env_db(x, hop_ms=10.0):
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    h = max(1, int(hop_ms * dsp.SR / 1000))
    k = max(1, len(m) // h)
    return 20 * np.log10(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-9)


def band_env_db(x, lo, hi, hop_ms=10.0):
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    y = dsp.butter(m, 'band', (lo, min(hi, 23000)), order=4) if lo > 20 else dsp.butter(m, 'low', hi, order=4)
    return env_db(y, hop_ms)


def side_db(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        return -99.0
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return float(10 * np.log10((side ** 2).sum() / ((mid ** 2).sum() + 1e-20) + 1e-12))


def low_share_db(x, f=100.0):
    """energy under f Hz re the total (dB)"""
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    lo = dsp.butter(m, 'low', f, order=4)
    return float(10 * np.log10((lo ** 2).sum() / ((m ** 2).sum() + 1e-20) + 1e-12))


def centroid_track(x, hop_ms=20.0):
    """spectral centroid (Hz) per frame of a mono fold"""
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    h = int(hop_ms * dsp.SR / 1000)
    k = max(1, len(m) // h)
    fr = m[:k * h].reshape(k, h) * np.hanning(h)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(h, 1 / dsp.SR)
    return (S * f).sum(1) / (S.sum(1) + 1e-20), S.sum(1)


# ------------------------------------------------------------------------------------------------ measurement
def measure(x):
    """the study's riser targets, model-free: length_s, last_third (share of the dB rise in the last third; pro risers
    are concave: >= 0.5), end_db (the last 50 ms re the peak: ~0 = it stops on its peak), hf_rise_db (> 4 kHz band,
    last 10 % vs first 25 %: pro +30-35 dB), cent_oct (centroid rise in octaves), side_db (pro: about -14..-18),
    low_db (energy < 100 Hz re total)"""
    e = env_db(x)
    k = len(e)
    pk = e.max()
    a = np.median(e[:max(1, k // 10)])
    b = e[-max(1, k // 10):].max()
    sm = np.convolve(e, np.ones(5) / 5, mode='same')
    two = sm[min(k - 1, int(k * 2 / 3))]
    rise = max(b - a, 1e-3)
    hf = band_env_db(x, 4000, 20000)
    c, pw = centroid_track(x)
    loud = pw > pw.max() * 1e-4
    ci = np.nonzero(loud)[0]
    cent_oct = float(np.log2(np.median(c[ci[-3:]]) / max(np.median(c[ci[:3]]), 1.0))) if len(ci) >= 6 else 0.0
    return {'length_s': round(len(x) / dsp.SR, 3), 'last_third': round(float((b - two) / rise), 2),
            'end_db': round(float(e[-5:].max() - pk), 1),
            'hf_rise_db': round(float(hf[-max(1, k // 10):].max() - np.median(hf[:max(1, k // 4)])), 1),
            'cent_oct': round(cent_oct, 2), 'side_db': round(side_db(x), 1), 'low_db': round(low_share_db(x), 1)}


def target_check(x, row=None):
    """critic `extra`: minus the distance (about 0..3) from the study's riser targets"""
    m = measure(x)
    pen = max(0.0, 0.45 - m['last_third']) * 4                  # concave: most of the rise late
    pen += max(0.0, -3.0 - m['end_db']) / 4                      # stops on its peak
    pen += max(0.0, 20.0 - m['hf_rise_db']) / 10                 # the top opens up
    pen += max(0.0, m['low_db'] + 10.0) / 6                      # no pile-up under 100 Hz
    pen += max(0.0, m['side_db'] + 6.0) / 6                      # mostly centred
    return -min(3.0, pen)


# ------------------------------------------------------------------------------------------------ render
def render(p, r):
    if p.get('r6', 0.0) > 0.5:                                   # round 6: the build render (default off)
        return render_build(p, r)
    p = {**DEFAULTS, **p}
    L = float(p['length'])
    n = dsp.n_of(L)
    t = dsp.t_of(n)
    u = t / L
    w = u ** p['curve']                                          # accelerating progress
    lv = 10 ** (-p['rise_db'] * (1 - w) / 20)

    # gate (laid out from the end so the last pulse is open on the final sample)
    per = p['gate_p0'] * (p['gate_p1'] / p['gate_p0']) ** (u ** 1.4)
    ph = phase_from_end(1.0 / per, end_phase=0.5 * p['gate_duty'])
    blips = p['tmode'] == 'blips'
    g = gate_shape(ph, per, p['gate_duty'], p['gate_sharp'])
    on = np.clip((u - p['gate_on']) / 0.2, 0, 1)
    fin = np.clip((L - t) / FINAL_OPEN, 0, 1)                     # the last FINAL_OPEN s: gate open, the peak
    fin = fin * fin * (3 - 2 * fin)
    depth = p['gate_depth'] * (on * on * (3 - 2 * on)) * fin
    gate = 1 - depth * (1 - g)

    # noise: resonant low-pass opening on the curve, high-passed, decorrelated and widening
    wn = (1 - p['pink']) * r.standard_normal(n) + p['pink'] * dsp.noise(L, r, 'pink')[:n] * 1.6
    fc = p['nf0'] * (p['nf1'] / p['nf0']) ** w
    nz = dsp.svf(wn, fc, p['nq'], 'low')
    nz = dsp.butter(nz, 'high', p['hp'], order=4)
    nz_st = fx.width(fx.decorrelate(nz, 1.0), p['width'] * (0.35 + 0.65 * w))

    # tonal: unison glide (or stepped blips on the gate) + a late octave layer
    wt = u ** max(1.0, 0.8 * p['curve'])
    f = p['tf0'] * 2 ** (p['toct'] * wt)
    ph0 = r.uniform(0, 1, 2 * len(UNISON))
    nblips = p['tmode'] == 'nblips'
    blips = blips or nblips
    if blips:
        idx = np.floor(ph).astype(np.int64)
        starts = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
        held = np.repeat(f[starts], np.diff(np.r_[starts, n]))   # pitch held per pulse: a stepped arpeggio
        fr = np.mod(ph, 1.0)
        fb = held * 2 ** (-0.35 * fr)                             # each blip a small falling chirp
        if nblips:                                                # round 3: each blip a band of noise at the step
            bl = np.stack([dsp.svf(r.standard_normal(n), np.minimum(fb, 16000.0), p['nb_q'], 'band')
                           for _ in range(2)], axis=1)
            mid = bl.mean(1, keepdims=True)
            tone = mid + (bl - mid) * min(1.0, 0.5 * p['width'] + 0.2)
        else:
            tone = unison(fb, p['wave'], p['detune'], 0.25 * p['width'], ph0[:5])
        tone *= (gate_shape(ph, per, p['gate_duty'], 1.0, perc=p['blip_dec']) * fin + 1 - fin)[:, None]
    else:
        tone = unison(f, p['wave'], p['detune'] * (0.6 + 0.8 * w), 0.5 * p['width'], ph0[:5])
    if nblips:
        octv = stereo_of(dsp.svf(r.standard_normal(n), np.minimum(2 * fb, 16000.0), p['nb_q'] * 1.5, 'band'))
    else:
        octv = unison(2 * (held * 2 ** (-0.35 * np.mod(ph, 1.0)) if blips else f), p['wave'], p['detune'] * 0.7,
                      0.6 * p['width'], ph0[5:])
    octv *= np.clip((u - 0.4) / 0.6, 0, 1)[:, None] ** 2
    if blips:
        octv *= (gate_shape(ph, per, p['gate_duty'], 1.0, perc=0.7 * p['blip_dec']) * fin + 1 - fin)[:, None]
    if p['wave'] == 'saw':
        cut = np.minimum(f * p['tbright'], 18000)
        tone = np.stack([dsp.svf(tone[:, c], cut, 0.8, 'low') for c in range(2)], axis=1)
        octv = np.stack([dsp.svf(octv[:, c], np.minimum(2 * cut, 18000), 0.8, 'low') for c in range(2)], axis=1)
    tone = dsp.butter(tone, 'high', p['hp'] * 0.8, order=2)
    octv = dsp.butter(octv, 'high', p['hp'], order=2)

    # shepard: an endless glide up
    if p['shep_db'] > -47:
        sh = dsp.shepard(L, r, up=True, rate=p['shep_rate'], f_lo=60.0, f_hi=9000.0, kind='tone')[:n]
        sh = dsp.butter(sh, 'high', p['hp'], order=2)
        sh_st = fx.width(fx.decorrelate(sh, 0.6), 0.6 * p['width'])
    else:
        sh_st = np.zeros((n, 2))

    # drone: steady low saw + sine, gently swelling
    df = p['drone_f'] * 2 ** (p['drone_oct'] * w)
    dr = sum(dsp.osc(df * 2 ** (c / 1200), 'saw', phase=r.uniform()) for c in (-7, 0, 7)) / 3
    dr = dsp.butter(dr, 'low', 500, order=2) * 0.8 + 0.35 * dsp.osc(df, 'sine', phase=r.uniform())
    if p['drone_noise'] > 0.01:                                  # round 3: a low noise rumble instead of a line
        rn = dsp.svf(r.standard_normal(n), np.clip(2.2 * df, 60, 2000), 1.2, 'band')
        rn = dsp.butter(rn, 'low', 600, order=2)
        dr = (1 - p['drone_noise']) * dr / (np.std(dr) + 1e-12) + p['drone_noise'] * rn / (np.std(rn) + 1e-12)
    dr = dsp.butter(dr, 'high', 30, order=2)
    dr *= 10 ** (-0.3 * p['rise_db'] * (1 - w) / 20)
    g2 = gate_shape(2 * ph, per / 2, p['gate_duty'], p['gate_sharp']) if blips else g   # blips: drone at 2x rate
    dr *= 1 - p['gate_drone'] * depth * (1 - g2)

    # metal: inharmonic partials, each shimmering under its own slow random amplitude, brightening with the build
    if p['metal_db'] > -47:
        ratios = np.array([1.0, 1.37, 1.71, 2.13, 2.58, 3.11, 3.66, 4.29, 5.02])
        if int(p['metal_n']) > 9:                                # round 3: dense random partials, no single line
            ratios = np.sort(np.exp(r.uniform(0, np.log(6.0), int(p['metal_n']))))
        glide = 2 ** (p['metal_glide'] * u ** 1.2)                 # moves from the start (w barely moves early)
        dense = int(p['metal_n']) > 9
        if dense:                                                # each partial opens at its own random point
            u_on = np.where(np.arange(len(ratios)) % 5 == 0, -1.0, r.uniform(0.0, 0.75, len(ratios)))
        met = np.zeros(n)
        for k, q in enumerate(ratios):
            fk = p['metal_f'] * q * (1 + 0.004 * r.standard_normal())
            if fk * 2 ** p['metal_glide'] > 19000:
                continue
            am = dsp.butter(r.standard_normal(n), 'low', 220, order=2)
            am = np.abs(am) / (np.abs(am).mean() + 1e-12)
            if dense:
                open_k = np.clip((w - u_on[k]) / 0.25, 0, 1)
            else:
                open_k = np.clip(w * len(ratios) * 0.9 - k * 0.6 + 1.0, 0, 1)
            if p['metal_glide'] > 0.001:
                met += np.sin(2 * np.pi * np.cumsum(fk * glide) / dsp.SR + r.uniform(0, 2 * np.pi)) * am * open_k \
                    / (1 + 0.15 * k * 9 / len(ratios))
            else:
                met += np.sin(2 * np.pi * fk * t + r.uniform(0, 2 * np.pi)) * am * open_k / (1 + 0.15 * k)
        met = dsp.butter(met, 'high', 1200, order=2)
        met_st = fx.width(fx.decorrelate(met, 1.0), p['width'])
    else:
        met_st = None

    def at(db, x):
        return 10 ** (db / 20) * x / lvl(x, 0.25)

    lay_n, lay_t, lay_o = at(p['noise_db'], nz_st), at(p['tone_db'], tone), at(p['oct_db'], octv)
    lay_s = at(p['shep_db'], sh_st) if p['shep_db'] > -47 else 0.0
    if met_st is not None:
        lay_s = lay_s + at(p['metal_db'], met_st)
    if blips:                                                    # the blips carry their own percussive gate
        send = ((lay_n + lay_s) * gate[:, None] + lay_t + lay_o) * lv[:, None]
    else:
        send = (lay_n + lay_t + lay_o + lay_s) * (lv * gate)[:, None]
    mix = send + at(p['drone_db'], stereo_of(dr)) * (lv ** 0.3)[:, None]
    if p['verb'] > 0.005:
        wet = wet_of(send, p['rt60'], n)
        mix = mix + p['verb'] * wet / lvl(wet, 0.25) * lvl(send, 0.25)
    if p['sat'] > 0.1:
        mix = fx.saturate(mix, drive_db=p['sat'], mode='tanh')
    mix = dsp.butter(mix, 'high', 30, order=2)
    mix = dsp.butter(mix, 'low', 18000, order=4)                 # round 3: polish found -30..-34 dB above 20 kHz
    return dsp.fade(mix, 0.004, p['end_ms'] / 1000.0)
