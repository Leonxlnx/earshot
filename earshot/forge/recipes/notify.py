"""Notification / success dings for motion design, written in code (every sample computed, no recorded audio).

    from earshot.forge import dsp
    from earshot.forge.recipes import notify
    x = notify.render(notify.FAMILY['pro_notify'][0], dsp.rng(1))      # 48 kHz stereo, the pro two-stage ding
    xs = [notify.render(p, dsp.rng(i)) for i, p in enumerate(notify.FAMILY['pro_notify'])]   # a palette of 5

What a pro launch-film ding is (docs/motion-sfx-study.md, measured on the 4 dings of ref_2, looked at in spectrograms):
  * A struck tone at 1.8-2.2 kHz with ONE inharmonic partial near 3.4x (6.3-6.9 kHz); nothing under 500 Hz.
  * Two stages, not a natural ring-out: the bright HEAD holds almost flat for 30-50 ms, then a gate drops it about
    18-22 dB within a few ms; a quiet BODY (-20 dB) carries on to 0.22-0.36 s with a slow tremolo (~20-30 Hz) and
    lines an octave under the head (0.5x) and at 1.5x / 3.5x, then fades (convex: it holds, then stops).
  * Attack 1-15 ms (a -30 dB step, then a short rise); t20 27-42 ms, t40 30-180 ms; near mono (side 5-13 dB under
    mid: the body is a little wide, the head centred).
  * A sequence steps up one semitone per ding (1822, 1934, 2045, 2168 Hz).
Real notification bells (the 'notify' refset: struck bells, glass, kalimba; held-out: celesta, glass, triangle) ring
out naturally for 0.8-2 s; style 'bell' makes that kind (modal partials whose highs die first, a mallet, a room).

Styles (`style`):
  'ding'     the pro two-stage ding above (head + gated drop + tremolo body), 0.2-0.4 s
  'success'  two dings of the same patch, the second a third / fourth / fifth up (`interval` semitones) `gap` s later
  'bell'     a struck modal bell / glass / celesta / glock bar ringing out (`partials`), 0.6-1.8 s
CHARACTER parameters (what a family varies on purpose): f0 (the note), hold (head length), body_db / body_ratio /
trem_hz (the body), bright (the 3.4x partial), length, width; for 'bell': partials, decay, hardness.
FAMILY[refset] = the hand-designed palette members (pro_notify: the semitone ladder + a success + a glassy variant;
notify: celesta / glass / glock bar / small bell / kalimba tine / high celesta). The forge tunes the fine parameters of
each member with the character ones fixed (family_space()). Round 3 (forge v2, after the round-2 judge): the pro ding's
head is clean (`head_drive` 0: round 2's drive made a harmonic comb that CLAP heard as 'a cheap synthesized beep') and
gets a struck bar's fast upper modes (`p3`), a breath of band noise (`air`) and a louder body that rings down; the
real-style bells are an octave or two lower (440-1319 Hz), with strike modes (`strike`) and an early fast decay
(`early_db`) over a ring that decays on its own (t40 1.3-2.0 s). The delivered members (tuned where a run tuned them)
are in out/forge_v2/<refset>/final/report.json (picks[i]['family_params']) and in the seed bank; round 2's in
final_r2/.

Helpers shared with the other tonal recipes (pop, shimmer): measure(x) = the pro-study measurements (attack, t20 / t40,
band shares, tilt, stereo correlation, head pitch, ridge sweep) and pro_penalty(x, targets) = how far a render falls
outside a category's pro targets (used as the critic's `extra=` check).
"""
import numpy as np

from .. import dsp, fx

SR = dsp.SR
TARGET = 'a short bright notification ding'
ALTS = ['a phone ringtone', 'an alarm clock beep', 'a wine glass clink', 'a xylophone note', 'a typewriter bell']
# (deliberately not the benchmark's notify confusions: eval/forge/bench.py CONFUSIONS stay independent)
REFSET = 'pro_notify'

SEMI = 2 ** (1 / 12)

SPACE = {
    'style': ['ding', 'success', 'bell', 'chime'],   # 'chime': round 6 (appended: older choice indices unchanged)
    'f0': (1100.0, 2600.0, 'log'),    # the note (head fundamental); pro dings 1822-2168 Hz
    # head
    'att': (0.0008, 0.012, 'log'),    # rise after the -30 dB step
    'step_db': (-36.0, -18.0),        # the onset step before the rise
    'hold': (0.018, 0.07),            # head holds this long ...
    'hold_tau': (0.04, 0.4, 'log'),   # ... sagging with this time constant
    'drop_db': (-26.0, -12.0),        # then a gate drops the head this far
    'drop_ms': (2.0, 14.0),           # within this time
    'ring_tau': (0.02, 0.15, 'log'),  # what is left of the head rings out with this tau
    'bright': (0.05, 0.9),            # level of the inharmonic partial
    'p2': (3.2, 3.6),                 # its ratio (3.4x in the films)
    'oct': (0.0, 0.3),                # a faint octave (2x) in the head
    'detune': (0.0, 3.0),             # head doublet beating (Hz)
    'tick_db': (-40.0, -18.0),        # mallet tick (band-passed noise, 1-2 ms)
    'tick_f': (4000.0, 11000.0, 'log'),
    # body
    'body_db': (-26.0, -12.0),        # body level re head
    'body_ratio': [0.5, 1.0, 1.5],    # body fundamental / f0 (0.5 = the films' octave-under line)
    'body_in': (0.004, 0.03),         # body fade-in time
    'body_tau': (0.08, 0.6, 'log'),   # body decay
    'body_p3': (0.0, 0.8),            # body partial at 3x its fundamental
    'body_p7': (0.0, 0.6),            # and near 7x (the films' ~3.5x-of-head line)
    'trem_hz': (12.0, 34.0),          # body tremolo rate
    'trem': (0.0, 0.85),              # and depth
    # glue / space / edit
    'drive': (0.0, 9.0),              # soft saturation dB (presence on small speakers)
    'verb': (0.0, 0.35),              # small plate send
    'rt60': (0.15, 0.7, 'log'),
    'width': (0.0, 1.0),              # body / reverb decorrelation
    'head_width': (0.0, 0.35),        # a little decorrelation on the head too (the films: corr 0.67-0.92)
    'length': (0.2, 0.42),            # total length
    'fade': (0.02, 0.09),             # convex end fade
    'hp': (450.0, 900.0, 'log'),      # nothing under 500 Hz
    # success
    'interval': [3, 4, 5, 7],
    'gap': (0.05, 0.13),
    'second_db': (-6.0, 1.0),
    # bell
    'partials': ['celesta', 'glass', 'bar', 'bowl', 'tine'],
    'decay': (0.2, 1.6, 'log'),
    'slope': (0.4, 1.2),
    'hard': (0.0, 1.0),               # mallet hardness (strike brightness)
    'beat': (0.3, 4.0, 'log'),
    'room': (0.0, 0.3),
    # round 3 (the judge: no strike transient, the ring decays too evenly, members 1-2 octaves too high)
    'early_db': (0.0, 18.0),          # bell: an early fast decay (dB lost in the first ~0.1 s on top of the ring)
    'early_tau': (0.02, 0.12, 'log'),
    'head_drive': (-1.0, 9.0),        # ding: drive on the head alone (< 0: the same as `drive`, as in round 2); the
                                      # round-2 drive made a comb of harmonics up to 20 kHz: CLAP's 'cheap beep'
    'air': (-60.0, -10.0),            # ding: a breath of band noise (3-14 kHz) under the head, dB re head: the
                                      # films' upper bands are full, but with noise, not a harmonic comb
    'air_tau': (0.02, 0.2, 'log'),
    'air_lo': (0.0, 12000.0),         # round 4: the breath's lower band edge in Hz (0 = min(3 kHz, 1.4 f0), as round 3):
                                      # 8-10 kHz puts it where the film dings' noise sits (> 12 kHz -17 dB) without the
                                      # 3-9 kHz hiss CLAP heard as 'crisp hi-hats'
    'p3': (0.0, 0.5),                 # ding: extra inharmonic head partials (2.76x, 5.4x: a struck bar's modes)
    'p3_tau': (0.008, 0.06, 'log'),   # ... that die fast (a natural strike, not a static beep)
    'strike': (-60.0, -4.0),          # bell: strike modes (high inharmonic partials dying in 10-40 ms), dB re ring
    'strike_tau': (0.006, 0.04, 'log'),
    'strike_max': (6.0, 20.0),        # round 4: bell strike modes above this ratio are left out (20 = all five, as round 3;
                                      # 12 = 6.1 / 8.7 / 11.9 only: fewer inharmonic partials up high, less 'clang')
}

PARTIALS = {   # ratios, amplitudes
    'celesta': ([1.0, 2.0, 3.98, 6.9, 10.1], [1.0, 0.06, 0.22, 0.05, 0.03]),   # tuned steel bar over a resonator
    'glass': ([1.0, 2.32, 4.25, 6.63, 9.38], [1.0, 0.45, 0.25, 0.12, 0.06]),
    'bar': ([1.0, 2.756, 5.404, 8.933], [1.0, 0.35, 0.14, 0.06]),               # glockenspiel (free-free beam)
    'bowl': ([1.0, 2.71, 5.13, 8.27], [1.0, 0.6, 0.3, 0.12]),
    'tine': ([1.0, 2.0, 5.8, 6.27], [1.0, 0.12, 0.12, 0.25]),                    # clamped tine (kalimba-like)
}

DEFAULTS = {'style': 'ding', 'f0': 1822.0, 'att': 0.003, 'step_db': -30.0, 'hold': 0.032, 'hold_tau': 0.15,
            'drop_db': -24.0, 'drop_ms': 5.0, 'ring_tau': 0.05, 'bright': 0.35, 'p2': 3.42, 'oct': 0.06,
            'detune': 1.2, 'tick_db': -30.0, 'tick_f': 7000.0, 'body_db': -23.0, 'body_ratio': 0.5, 'body_in': 0.012,
            'body_tau': 0.3, 'body_p3': 0.45, 'body_p7': 0.6, 'trem_hz': 24.0, 'trem': 0.55, 'drive': 3.0,
            'verb': 0.12, 'rt60': 0.3, 'width': 0.8, 'head_width': 0.15, 'length': 0.26, 'fade': 0.07, 'hp': 550.0, 'interval': 7,
            'gap': 0.085, 'second_db': -2.0, 'partials': 'celesta', 'decay': 0.8, 'slope': 0.8, 'hard': 0.4,
            'beat': 1.0, 'room': 0.12, 'early_db': 0.0, 'early_tau': 0.05, 'strike': -60.0, 'strike_tau': 0.015, 'head_drive': -1.0, 'air': -60.0,
            'air_tau': 0.06, 'p3': 0.0, 'p3_tau': 0.02, 'strike_max': 20.0, 'air_lo': 0.0}


def _fam(**kw):
    return {**DEFAULTS, **kw}


_LADDER = [1822.0 * SEMI ** k for k in range(5)]      # 1822, 1930, 2045, 2167, 2296 Hz
FAMILY = {
    # the pro palette: one patch, five members a listener can tell apart (pitch step, head length, body, motif)
    'pro_notify': [   # round 3: from the round-2 tuned members, minus the 'cheap beep': a clean head (no drive
                      # comb) with a struck bar's fast upper modes, a breath of noise, a tiny strike, a louder body
                      # that rings down naturally (exponential, not flat-then-gated)
        _fam(style='success', f0=_LADDER[0], interval=7, att=0.002, step_db=-28.0, hold=0.026, hold_tau=0.05,
             drop_db=-12.0, drop_ms=12.0, ring_tau=0.07, bright=0.3, p2=3.4, oct=0.12, detune=0.6, tick_db=-26.0,
             tick_f=9000.0, body_db=-20.0, body_ratio=0.5, body_in=0.02, body_tau=0.14, body_p3=0.5, body_p7=0.3,
             trem=0.45, trem_hz=24.0, drive=2.0, head_drive=0.0, air=-19.0, air_tau=0.05, p3=0.3, p3_tau=0.02,
             verb=0.22, rt60=0.3, width=0.8, head_width=0.25, length=0.4, fade=0.05, hp=500.0, gap=0.06,
             second_db=-1.5),                                                               # success: root + fifth
        _fam(f0=_LADDER[2], att=0.002, step_db=-30.0, hold=0.036, hold_tau=0.1, drop_db=-14.0, drop_ms=8.0,
             ring_tau=0.06, bright=0.2, p2=3.55, oct=0.15, detune=2.0, tick_db=-24.0, tick_f=9000.0, body_db=-18.0,
             body_ratio=0.5, body_in=0.015, body_tau=0.16, body_p3=0.7, body_p7=0.45, trem=0.5, trem_hz=18.0,
             drive=3.0, head_drive=0.0, air=-18.0, air_tau=0.06, p3=0.25, p3_tau=0.025, verb=0.22, rt60=0.3,
             width=1.0, head_width=0.3, length=0.34, fade=0.06, hp=450.0),                  # 2045 Hz, long body
        _fam(f0=_LADDER[0], att=0.0015, step_db=-22.0, hold=0.03, hold_tau=0.1, drop_db=-13.0, drop_ms=6.0,
             ring_tau=0.05, bright=0.35, p2=3.44, oct=0.05, detune=1.0, tick_db=-22.0, tick_f=7000.0,
             body_db=-19.0, body_ratio=0.5, body_in=0.006, body_tau=0.13, body_p3=0.45, body_p7=0.3, trem=0.55,
             trem_hz=24.0, drive=3.0, head_drive=0.0, air=-17.0, air_tau=0.04, p3=0.35, p3_tau=0.018, verb=0.12,
             rt60=0.35, width=0.8, head_width=0.2, length=0.28, fade=0.05, hp=600.0),        # the film ding 1822 Hz
        _fam(f0=_LADDER[1], att=0.002, step_db=-34.0, hold=0.022, hold_tau=0.2, drop_db=-12.0, drop_ms=8.0,
             ring_tau=0.045, bright=0.5, p2=3.46, oct=0.05, detune=0.5, tick_db=-25.0, tick_f=9500.0,
             body_db=-20.0, body_ratio=1.0, body_in=0.02, body_tau=0.12, body_p3=0.7, body_p7=0.2, trem=0.25,
             trem_hz=24.0, drive=3.0, head_drive=0.0, air=-19.0, air_tau=0.05, p3=0.2, p3_tau=0.015, verb=0.3,
             rt60=0.3, width=0.8, head_width=0.2, length=0.26, fade=0.06, hp=520.0),        # 1930 Hz, same-pitch body
        _fam(f0=_LADDER[3], att=0.003, step_db=-20.0, hold=0.018, hold_tau=0.05, drop_db=-12.0, drop_ms=5.0,
             ring_tau=0.04, bright=0.7, p2=3.3, oct=0.08, detune=0.8, tick_db=-26.0, tick_f=8000.0, body_db=-20.0,
             body_ratio=1.5, body_in=0.011, body_tau=0.12, body_p3=0.6, body_p7=0.4, trem=0.6, trem_hz=30.0,
             drive=3.0, head_drive=0.0, air=-18.0, air_tau=0.035, p3=0.25, p3_tau=0.012, verb=0.16, rt60=0.5,
             width=0.8, head_width=0.25, length=0.24, fade=0.05, hp=450.0),                 # 2167 Hz, glassy
    ],
    # real-style notification bells: struck, ringing out
    'notify': [   # round 3: lower notes (the bench bells sit 150 Hz-1 kHz; round 2 rang 1-2 octaves above), a
                  # strike transient, an early fast decay over a longer ring (t40 x 0.63 in round 2)
        _fam(style='bell', f0=523.25, partials='celesta', decay=0.5, slope=1.0, hard=0.7, early_db=8.0,
             early_tau=0.05, strike=-14.0, strike_tau=0.015, beat=0.8, length=2.3, room=0.12, hp=180.0,
             tick_db=-26.0, drive=1.0, width=0.2),                                          # celesta C5
        _fam(style='bell', f0=880.0, partials='glass', decay=0.42, slope=1.1, hard=0.75, early_db=7.0,
             early_tau=0.04, strike=-12.0, strike_tau=0.012, beat=1.6, length=2.0, room=0.15, hp=300.0,
             tick_db=-24.0, drive=1.0, width=0.2),                                          # glass clink A5
        _fam(style='bell', f0=783.99, partials='bar', decay=0.38, slope=1.0, hard=0.85, early_db=8.0,
             early_tau=0.035, strike=-10.0, strike_tau=0.01, beat=1.0, length=1.8, room=0.1, hp=300.0,
             tick_db=-22.0, drive=1.5, width=0.2),                                          # glock bar G5
        _fam(style='bell', f0=440.0, partials='bowl', decay=0.6, slope=0.9, hard=0.6, early_db=10.0,
             early_tau=0.07, strike=-16.0, strike_tau=0.02, beat=1.2, length=2.5, room=0.15, hp=150.0,
             tick_db=-30.0, drive=1.0, width=0.25),                                         # small bell A4
        _fam(style='bell', f0=587.33, partials='tine', decay=0.36, slope=1.1, hard=0.65, early_db=5.0,
             early_tau=0.04, strike=-18.0, strike_tau=0.012, beat=0.6, length=1.8, room=0.1, hp=150.0,
             tick_db=-26.0, drive=1.0, width=0.15),                                         # kalimba tine D5
        _fam(style='bell', f0=1318.5, partials='celesta', decay=0.34, slope=1.0, hard=0.8, early_db=7.0,
             early_tau=0.03, strike=-12.0, strike_tau=0.01, beat=0.9, length=1.7, room=0.12, hp=300.0,
             tick_db=-22.0, drive=1.0, width=0.2),                                          # a high celesta E6
    ],
}
# what each style lets the forge tune (the character keys stay fixed per family member)
FINE = {
    'ding': ['att', 'step_db', 'head_width', 'hold_tau', 'drop_db', 'drop_ms', 'ring_tau', 'p2', 'oct', 'detune', 'tick_db',
             'tick_f', 'body_in', 'body_tau', 'body_p3', 'body_p7', 'drive', 'verb', 'rt60', 'fade', 'hp', 'air',
             'air_tau', 'p3', 'p3_tau'],
    'bell': ['slope', 'hard', 'beat', 'room', 'tick_db', 'tick_f', 'drive', 'fade', 'att', 'early_db', 'early_tau',
             'strike', 'strike_tau'],
}
FINE['success'] = FINE['ding'] + ['gap', 'second_db']
FINE['chime'] = ['att', 'c_fast', 'c_tf', 'c_ts', 'c_mallet', 'c_hard', 'c_bend', 'detune', 'verb', 'rt60', 'fade', 'hp']

# Round 4 (ui-tonal polish): edits on top of the round-3 DELIVERED members (out/forge_v2/<cat>/final_r3/report.json,
# 'family' members), applied by out/forge_v2/_uitonal_r4/fam4.py.
# notify (the round-3 judge: AudioSet hears 'Clang' first on 4 of 6, round 2 'Ding'; keep the strike and the early decay,
# keep the lower notes): a softer mallet (hard 0.15 lower: q^1.5(hard-0.5) lifts the upper partials less), upper partials
# die a little faster (slope 1.1), strike modes only up to 12x f0 and 6 dB softer (r4a = delivered). Softer again on the
# three low members (C5, A4, D5; hard 0.4-0.45, strike modes to 9x: r4b) gave dull onsets (polish 1.3-1.4, onset > 6 kHz
# -42..-49 dB) and changed nothing else, so r4a stays. (A first try at hard 0.38-0.5 and
# strike -22..-26 went too dark: 4-12 kHz -30 dB of total vs the forge refs' -15, onset > 6 kHz -29 vs -14.)
# pro_notify (the judge: closer to the film dings, 2 kHz + ~3.4x, clean; CLAP heard 'crisp hi-hats' / 'triangle hit'):
# the head's noise breath moved above 9 kHz (air_lo; the films' > 12 kHz sits at -17 dB of total, their 4-12 kHz at -14
# where round 3 had -9: the 3-9 kHz hiss and a loud 3.4x partial), the struck-bar modes nearly gone (p3 0.06), a quieter
# tick (the films' onset > 6 kHz -18.7 dB vs round 3's -9.8), the 3.4x partial kept at ~60 % of round 3, a faint octave
# (the 4 kHz band stood 9 dB over the films) and a louder body (the films' 1 kHz octave is -6 dB re total next to 2 kHz
# -5: round 3's body sat 10-13 dB under them at 315 Hz-1.26 kHz). NOT delivered: the critic scored every member lower
# than round 3 (-5.9..-10.4 vs -5.1..-8.5, CLAP 'a cheap synthesized beep' on all five), so pro_notify/final = round 3.
# That attempt (r4a) was: [{'air': -22.0, 'air_lo': 9000.0, 'p3': 0.06, 'tick_db': -32.0, 'bright': b, 'oct': 0.04,
# 'body_db': -12.0} for b in (0.2, 0.15, 0.22, 0.3, 0.4)]. r4b keeps round 3's head and breath and only raises the body
# (-18..-20 -> -13 dB re head: the measured gap to the films is at 315 Hz-1.26 kHz). Also NOT delivered: the critic was
# mixed (median -5.97 -> -6.14: two members up, three down). pro_notify/final stays round 3 (fam3 members, no R4 edit).
R4_EDITS = {
    'notify': [{'hard': h, 'strike': s, 'strike_max': 12.0, 'slope': 1.1}
               for h, s in ((0.55, -20.0), (0.6, -18.0), (0.65, -17.0), (0.48, -22.0), (0.5, -24.0), (0.62, -18.0))],
    'pro_notify': [{'body_db': -13.0}] * 5,      # r4b (NOT delivered, see below): round 3 + a louder body

}


# Round 5 (polish-pro; the brief: match the film dings - a 2 kHz fundamental + a ~3.4x partial, 1-15 ms attack, t20
# 27-42 ms - clean and pleasant, a semitone family). What earshot showed first: CLAP calls the four film dings themselves
# 'a cheap synthesized beep' / 'a computer-generated electronic tone' (out/forge_v2/pro_notify/LOG.md, round 5), so the
# 'cheap beep' words that sank the round-4 attempts are the films' own reading, not a fault; the round-3 members carry a
# 3-16 kHz noise breath ('crisp hi-hats'), struck-bar modes and a driven body (a harmonic comb) the films do not have.
# The round-5 members are the film's two-stage ding made clean: a pure head (sine + the 3.4x partial dying faster, a slow
# doublet beat, no drive, no breath, no bar modes), held 30-40 ms and gated ~20 dB, then the body an octave under (with
# its 3x / 7x lines: the films' 1.5x / 3.5x-of-head lines) swelling in under the gate with a gentle tremolo, a small
# plate, a little width. One per film ding on the semitone ladder, each with that ding's measured attack (1 / 15 / 3 /
# 14 ms), and a fifth member a semitone up. The round-3 family (the delivered finals until round 5) is FAMILY_R3.
FAMILY_R3 = {'pro_notify': [dict(p) for p in FAMILY['pro_notify']]}
_CLEAN = dict(style='ding', step_db=-30.0, hold=0.034, hold_tau=0.25, drop_db=-20.0, drop_ms=4.0, ring_tau=0.05,
              bright=0.25, p2=3.42, oct=0.0, detune=0.8, tick_db=-30.0, tick_f=9000.0, body_db=-22.0, body_ratio=0.5,
              body_in=0.03, body_tau=0.13, body_p3=0.25, body_p7=0.2, trem=0.35, trem_hz=24.0, drive=0.0,
              head_drive=0.0, air=-60.0, p3=0.0, verb=0.15, rt60=0.3, width=0.7, head_width=0.15, length=0.3,
              fade=0.05, hp=500.0)
FAMILY['pro_notify'] = [
    _fam(**{**_CLEAN, 'f0': _LADDER[0], 'att': 0.0015, 'length': 0.26}),                                    # film ding 1
    _fam(**{**_CLEAN, 'f0': _LADDER[1], 'att': 0.012, 'step_db': -26.0, 'hold': 0.04, 'body_db': -21.0,
            'width': 0.9, 'head_width': 0.25, 'length': 0.27}),                                             # film ding 2
    _fam(**{**_CLEAN, 'f0': _LADDER[2], 'att': 0.003, 'hold': 0.036, 'body_tau': 0.2, 'body_db': -20.0,
            'length': 0.36, 'trem': 0.45}),                                                                 # film ding 3
    _fam(**{**_CLEAN, 'f0': _LADDER[3], 'att': 0.011, 'step_db': -26.0, 'hold': 0.04, 'bright': 0.32,
            'width': 0.9, 'head_width': 0.25, 'length': 0.3}),                                              # film ding 4
    _fam(**{**_CLEAN, 'f0': _LADDER[4], 'att': 0.004, 'hold': 0.03, 'bright': 0.2, 'body_ratio': 1.0,
            'body_p3': 0.15, 'body_p7': 0.0, 'body_db': -19.0, 'body_tau': 0.16, 'trem': 0.25,
            'length': 0.26}),                                                                               # a semitone up
]
FAMILY_NAMES = {'pro_notify': ['ding_1822', 'ding_1930', 'ding_2045', 'ding_2167', 'ding_2296']}


def family_space(params, style=None):
    """the search space for one family member: its FINE keys free (full SPACE ranges), every other key fixed to its
    value (a one-element choice list: the forge's way of pinning a parameter)"""
    st = style or params.get('style', 'ding')
    free = set(FINE.get(st, []))
    sp = {}
    for k, spec in SPACE.items():
        if k in free:
            sp[k] = spec
        else:
            sp[k] = [params.get(k, DEFAULTS.get(k))]
    return sp


# ------------------------------------------------------------------------------------------------ synthesis
def _partial(t, f, amp, r, detune=0.0):
    ph = r.uniform(0, 2 * np.pi)
    y = np.sin(2 * np.pi * f * t + ph)
    if detune > 0:                       # a doublet that starts in phase at the strike, then beats
        y = (y + 0.6 * np.sin(2 * np.pi * (f + detune * r.uniform(0.6, 1.4)) * t + ph)) / 1.6
    return amp * y


def _head_env(t, p):
    """the -30 dB step, a short rise, a sagging hold, a fast gated drop, then what is left rings out"""
    att, hold = p['att'], p['hold']
    step = 10 ** (p['step_db'] / 20)
    rise = np.clip(t / att, 0, 1)
    rise = step + (1 - step) * (0.5 - 0.5 * np.cos(np.pi * rise))
    sag = np.exp(-t / p['hold_tau'])
    drop_t = max(p['drop_ms'] / 1000, 0.001)
    floor = 10 ** (p['drop_db'] / 20)
    u = np.clip((t - hold) / drop_t, 0, 1)
    gate = 1 - (1 - floor) * (0.5 - 0.5 * np.cos(np.pi * u))        # 1 -> floor along a half cosine
    after = np.where(t > hold + drop_t, np.exp(-(t - hold - drop_t) / p['ring_tau']), 1.0)
    e = rise * sag * gate * after
    e[0] = 0.0
    return e


def _tick(n, p, r):
    k = min(n, dsp.n_of(0.004))
    w = r.standard_normal(k)
    lo = p['tick_f'] / 1.6
    w = dsp.butter(w, 'band', (lo, min(p['tick_f'] * 1.6, 20000)), order=2)
    w *= np.exp(-dsp.t_of(k) / 0.0007)
    w /= np.abs(w).max() + 1e-12
    y = np.zeros(n)
    y[:k] = w * 10 ** (p['tick_db'] / 20)
    return y


def _ding(p, r, n):
    """one two-stage ding: (head mono, body mono); both start at t = 0"""
    t = dsp.t_of(n)
    f0 = p['f0']
    head = _partial(t, f0, 1.0, r, p['detune'])
    hp2 = p['bright'] * np.exp(-t / 0.09)                                  # the high partial dies a little faster
    if f0 * p['p2'] < 0.45 * SR:
        head += hp2 * _partial(t, f0 * p['p2'], 1.0, r)
    if p['oct'] > 0:
        head += p['oct'] * _partial(t, 2 * f0, 1.0, r)
    if p['p3'] > 0:                                                        # round 3: a struck bar's upper modes
        for q, a in ((2.756, 1.0), (5.404, 0.6)):
            if f0 * q < 0.4 * SR:
                head += p['p3'] * a * np.exp(-t / (p['p3_tau'] * (2.756 / q))) * _partial(t, f0 * q, 1.0, r, 1.5)
    head *= _head_env(t, p)
    if p['air'] > -59:                                                     # round 3: noise, not a harmonic comb
        lo = p.get('air_lo', 0.0) or min(3000.0, 1.4 * f0)
        w = dsp.butter(r.standard_normal(n), 'band', (lo, 16000.0), order=2 if lo < 6000 else 4)
        w /= np.sqrt(np.mean(w ** 2)) + 1e-12
        ea = np.exp(-t / p['air_tau'])
        k1 = dsp.n_of(0.001)
        ea[:k1] *= np.linspace(0, 1, k1)
        head += 0.7 * 10 ** (p['air'] / 20) * w * ea
    head += _tick(n, p, r)
    # body: a quieter tone under / beside the head, swelling in during the hold, tremolo, slow decay
    F = f0 * p['body_ratio']
    body = _partial(t, F, 1.0, r, 0.8)
    norm = 1.0
    for mult, a in ((3.0, p['body_p3']), (7.0 * 0.995, p['body_p7'])):
        if a > 0 and F * mult < 0.45 * SR:
            body += a * _partial(t, F * mult, 1.0, r)
            norm += a
    body /= norm                                                           # body_db = the body's peak level
    t0 = p['hold'] * 0.6
    ein = np.clip((t - t0) / max(p['body_in'], 1e-3), 0, 1)
    env = (0.5 - 0.5 * np.cos(np.pi * ein)) * np.exp(-np.maximum(t - t0, 0) / p['body_tau'])
    trem = 1 - p['trem'] * (0.5 - 0.5 * np.cos(2 * np.pi * p['trem_hz'] * t + r.uniform(0, 1)))
    body *= env * trem * 10 ** (p['body_db'] / 20)
    return head, body


def _bell(p, r, n):
    """a struck modal bell: partials whose highs die first, doublet beating, a mallet whose hardness sets how much
    of the upper partials it excites; round 3: an early fast decay (early_db over early_tau: the strike energy that
    real bells shed in the first ~0.1 s) and strike modes (inharmonic high partials dying in 10-40 ms)"""
    t = dsp.t_of(n)
    ratios, amps = PARTIALS[p['partials']]
    y = np.zeros(n)
    g_e = 10 ** (p['early_db'] / 20) - 1.0
    for q, a in zip(ratios, amps):
        f = p['f0'] * q
        if f > 0.45 * SR:
            continue
        a = a * q ** (1.5 * (p['hard'] - 0.5))
        tau = max(p['decay'] * q ** -p['slope'], 0.004)
        b = p['beat'] * r.uniform(0.5, 1.5)                      # doublet beating as a slow AM, loudest at the strike
        am = 1 - 0.35 * (0.5 - 0.5 * np.cos(2 * np.pi * b * t))
        env = np.exp(-t / tau)
        if g_e > 0:                                              # higher partials shed more of their strike energy
            env = env * (1 + g_e * min(1.0, 0.5 + 0.25 * q) * np.exp(-t / p['early_tau']))
        y += a * env * am * np.sin(2 * np.pi * f * t + r.uniform(0, 6.3))
    if g_e > 0:
        y /= 1 + g_e * 0.75
    if p['strike'] > -59:
        ys = np.zeros(n)
        for q in (6.1, 8.7, 11.9, 15.3, 19.6):
            f = p['f0'] * q * r.uniform(0.98, 1.02)
            if f > 17000 or q > p.get('strike_max', 20.0):
                continue
            ys += r.uniform(0.5, 1.0) * np.exp(-t / (p['strike_tau'] * (6.1 / q) ** 0.5)) *                 np.sin(2 * np.pi * f * t + r.uniform(0, 6.3))
        y += 10 ** (p['strike'] / 20) * ys
    a = dsp.n_of(max(p['att'] * 0.5, 0.0005))
    y[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    # mallet: low-passed noise pulse, brighter when hard
    k = min(n, dsp.n_of(0.006))
    m = dsp.butter(r.standard_normal(k), 'low', 2000 + 14000 * p['hard'], order=2) * np.exp(-dsp.t_of(k) / 0.001)
    y[:k] += (0.05 + 0.25 * p['hard']) * m / (np.abs(m).max() + 1e-12) * np.abs(y).max()
    y += _tick(n, p, r) * np.abs(y).max()
    return y


# round 6 (pop-notify; the round-5 judge: clean but near clones, one patch on a semitone ladder): style 'chime' = a
# struck ding of a chosen material around the film targets (a ~2 kHz fundamental that rings longest, a partial near
# 3.4x, 1-15 ms attack, t20 ~30-40 ms from a natural two-rate decay: most of the strike dies in ~10-15 ms, a quiet
# aftersound rings on), varied by what is struck and how: `mat` (the partial set), `c_fast` / `c_tf` / `c_ts` (the
# two decay rates), `c_mallet` / `c_hard` (felt .. metal strike), `c_bend` (the strike's pitch drop, cents), `c_sub`
# (an octave-under body: the films' 1 kHz line), and a motif (`c_motif`: one strike, 'up' = a second note `interval`
# semitones up `gap` s later, 'flam' = a quieter second strike of the same note). 'fm' = a 1 : 2.4 FM bell (its first
# sidebands sit at 1.4x and 3.4x), the index decaying fast.
MATS = {   # (ratio, amplitude, decay multiplier on both rates): every set keeps the fundamental strongest
    'film': [(1.0, 1.0, 1.0), (3.42, 0.3, 0.45)],
    'glass': [(1.0, 1.0, 1.0), (2.32, 0.3, 0.5), (3.4, 0.28, 0.35), (4.25, 0.12, 0.25), (6.63, 0.05, 0.15)],
    'bowl': [(1.0, 1.0, 1.0), (2.71, 0.4, 0.6), (3.38, 0.22, 0.4), (5.13, 0.14, 0.3)],
    'tine': [(1.0, 1.0, 1.0), (3.44, 0.12, 0.3), (5.8, 0.08, 0.25), (6.27, 0.1, 0.2)],
    'wood': [(1.0, 1.0, 1.0), (3.4, 0.18, 0.3), (3.98, 0.12, 0.22), (9.2, 0.03, 0.1)],
    'fm': [(1.0, 1.0, 1.0)],
}
SPACE.update({'mat': list(MATS), 'c_fast': (0.5, 0.97), 'c_tf': (0.004, 0.03, 'log'), 'c_ts': (0.06, 0.4, 'log'),
              'c_mallet': (-60.0, -6.0), 'c_hard': (0.0, 1.0), 'c_bend': (0.0, 60.0), 'c_sub': (0.0, 0.6),
              'c_motif': ['single', 'up', 'flam'], 'c_index': (0.5, 6.0), 'c_itau': (0.006, 0.06, 'log')})
DEFAULTS.update({'mat': 'film', 'c_fast': 0.9, 'c_tf': 0.012, 'c_ts': 0.15, 'c_mallet': -30.0, 'c_hard': 0.5,
                 'c_bend': 0.0, 'c_sub': 0.0, 'c_motif': 'single', 'c_index': 2.5, 'c_itau': 0.02})


def _strike(p, r, n, f0):
    """one struck note of material p['mat']: mono, starts at t = 0"""
    t = dsp.t_of(n)
    bend = 2 ** (p['c_bend'] / 1200 * np.exp(-t / 0.01))                 # struck hard, the pitch settles
    att = max(p['att'], 0.0006)
    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / att, 0, 1))
    tm = np.maximum(t - att, 0)
    fast, tf, ts = p['c_fast'], p['c_tf'], p['c_ts']
    y = np.zeros(n)
    if p['mat'] == 'fm':                                                   # 1 : 2.4 FM bell, the index decays fast
        idx = p['c_index'] * np.exp(-tm / p['c_itau']) + 0.25
        mph = 2 * np.pi * np.cumsum(2.4 * f0 * bend) / SR + r.uniform(0, 6.3)
        cph = 2 * np.pi * np.cumsum(f0 * bend) / SR + r.uniform(0, 6.3)
        env = fast * np.exp(-tm / tf) + (1 - fast) * np.exp(-tm / ts)
        y = np.sin(cph + idx * np.sin(mph)) * env
    else:
        for q, a, dm in MATS[p['mat']]:
            f = f0 * q
            if f > 0.42 * SR:
                continue
            ph = 2 * np.pi * np.cumsum(f * bend) / SR + r.uniform(0, 6.3)
            tone = np.sin(ph)
            if p['detune'] > 0:                                            # a slow doublet beat
                tone = (tone + 0.5 * np.sin(ph + 2 * np.pi * p['detune'] * r.uniform(0.6, 1.4) * t)) / 1.5
            env = fast * np.exp(-tm / (tf * dm)) + (1 - fast) * np.exp(-tm / (ts * dm))
            y += a * env * tone
    if p['c_sub'] > 0:                                                     # the octave-under body (slow, rounded)
        ein = np.clip(t / 0.02, 0, 1)
        y += p['c_sub'] * (0.5 - 0.5 * np.cos(np.pi * ein)) * np.exp(-tm / ts) * \
            np.sin(2 * np.pi * np.cumsum(0.5 * f0 * bend) / SR + r.uniform(0, 6.3))
    y *= rise
    if p['c_mallet'] > -59:                                                # felt (soft, dark, longer) .. metal (bright tick)
        k = min(n, dsp.n_of(0.008))
        tk = dsp.t_of(k)
        h = p['c_hard']
        w = r.standard_normal(k)
        w = dsp.butter(w, 'band', (1500.0 + 4000.0 * h, min(4000.0 + 12000.0 * h, 18000.0)), order=2)
        w *= (1 - np.exp(-tk / 0.0003)) * np.exp(-tk / (0.0035 - 0.0027 * h))
        y[:k] += 10 ** (p['c_mallet'] / 20) * w / (np.abs(w).max() + 1e-12)
    return y


def _chime(p, r, n):
    y = _strike(p, r, n, p['f0'])
    if p['c_motif'] in ('up', 'flam'):
        off = dsp.n_of(p['gap'])
        f2 = p['f0'] * (SEMI ** p['interval'] if p['c_motif'] == 'up' else 1.0)
        y2 = _strike(p, r, n - off, f2)
        if p['c_motif'] == 'up':                                           # the first note ducks under the second
            y *= np.clip(1 - dsp.t_of(n) / (p['gap'] + 0.04), 0.3, 1)
        y[off:] += 10 ** (p['second_db'] / 20) * y2
    return y


def _finish(x, p, r, n_len, drive=None):
    """glue, space, stereo, band limit, edit (drive: the head's own drive when >= 0, else p['drive'])"""
    drive = p['drive'] if drive is None or drive < 0 else drive
    if drive > 0.2:
        x = fx.saturate(x, drive, mix=0.6)
    x = dsp.butter(x, 'high', p['hp'], order=4)
    return x


def _end(y, p, n_len, fade=None):
    y = y[:n_len]
    if len(y) < n_len:
        pad = np.zeros((n_len - len(y),) + y.shape[1:])
        y = np.concatenate([y, pad])
    k = min(len(y), dsp.n_of(fade or p['fade']))
    w = np.cos(np.linspace(0, np.pi / 2, k)) ** 2            # convex: holds, then stops
    y[-k:] *= w[:, None] if y.ndim == 2 else w
    a = dsp.n_of(0.0003)                                     # round 2: the mallet / tick started off zero
    wi = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    y[:a] *= wi[:, None] if y.ndim == 2 else wi
    return y


def render(p, r):
    p = {**DEFAULTS, **p}
    if p.get('ring', 1.0) != 1.0:                         # earshot 0.4: the decay knob (off at 1: no change at all)
        p = ring_scaled(p)
    st = p['style']
    n_len = dsp.n_of(p['length'] if st != 'bell' else (p['bell_len'] or max(p['length'], 0.5)))
    n = n_len + dsp.n_of(0.05)
    if st == 'bell':
        y = _bell(p, r, n)
        if p.get('soft', 0.0) > 0:
            y = _soft(y, p['soft'])
        y = _finish(y, p, r, n_len)
        y = fx.decorrelate(y, p['width'] * 0.5, cut=600.0)
        if p['room'] > 0:
            y = fx.reverb(y, r, rt60=0.35 + 1.5 * p['room'], size=0.35, predelay=0.008, damp=0.55, wet=p['room'],
                          stereo=True, width=0.8)
        if p['early_db'] > 0 or p['strike'] > -59:          # round 3: the ring decays on its own, a short end fade
            return _end(y, p, n_len, fade=max(p['fade'], 0.25))
        return _end(y, p, n_len, fade=max(p['fade'], 0.4 * n_len / SR))
    if st == 'chime':                                                      # round 6: a struck material + motif
        y = _chime(p, r, n)
        if p.get('soft', 0.0) > 0:
            y = _soft(y, p['soft'])
        y = _finish(y, p, r, n_len)
        y = fx.decorrelate(y, p['width'] * 0.5, cut=1200.0)
        if p['verb'] > 0.005:
            wet = fx.reverb(y, r, rt60=p['rt60'], size=0.25, predelay=0.006, damp=0.45, wet=1.0, dry=0.0,
                            stereo=True, width=0.4 + p['width'])
            y = np.concatenate([y, np.zeros((len(wet) - len(y), 2))]) + p['verb'] * wet
        return _end(y, p, n_len)
    head, body = _ding(p, r, n)
    if st == 'success':
        q = {**p, 'f0': p['f0'] * SEMI ** p['interval']}
        h2, b2 = _ding(q, r, n)
        g = 10 ** (p['second_db'] / 20)
        off = dsp.n_of(p['gap'])
        h2 = np.concatenate([np.zeros(off), h2])[:n] * g
        b2 = np.concatenate([np.zeros(off), b2])[:n] * g
        body = body * np.clip(1 - dsp.t_of(n) / (p['gap'] + 0.03), 0.25, 1) + b2   # the first body ducks under
        head = head + h2
    head = _finish(head, p, r, n_len, p['head_drive'])    # round 3 tried a clean head (no drive): every critic number got worse
    body = _finish(body, p, r, n_len)    # (totals -5.7..-7.2 -> -10.0..-12.9, tone 8-11 -> 16-18 dB), so it stays
    st_body = fx.decorrelate(body, p['width'], cut=600.0)
    y = fx.decorrelate(head, p['head_width'], cut=1500.0) + st_body
    if p['verb'] > 0.005:                    # the plate is fed mostly by the body: the head's gate stays a gate
        send = 0.3 * np.stack([head, head], 1) + st_body
        wet = fx.reverb(send, r, rt60=p['rt60'], size=0.25, predelay=0.006, damp=0.45, wet=1.0, dry=0.0, stereo=True,
                        width=0.4 + p['width'])
        y = np.concatenate([y, np.zeros((len(wet) - len(y), 2))]) + p['verb'] * wet
    if p.get('soft', 0.0) > 0:
        y = _soft(y, p['soft'])
    return _end(y, p, n_len)


# ------------------------------------------------------------------------------------------------ measurements
REGIONS = ((0, 150), (150, 500), (500, 1000), (1000, 4000), (4000, 12000), (12000, 24000))


def _env_db(m, ms):
    h = max(1, int(SR * ms / 1000))
    k = len(m) // h
    if k < 1:
        return np.array([-120.0])
    e = np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12
    return 20 * np.log10(e)


def measure(x):
    """the pro-study measurements of one render (bed = silence): length_s, attack_ms, t20_ms, t40_ms, shares (dB of
    total in REGIONS), tilt (dB / octave, 100 Hz - 10 kHz), corr / side_mid_db (stereo), head_hz (strongest peak in
    the 40 ms after the onset), ridge (strongest peak per 5 ms frame while within 30 dB of the peak: start / end Hz,
    sweep octaves), peak_t (s)"""
    x = np.asarray(x, np.float64)
    st = x if x.ndim == 2 else np.stack([x, x], 1)
    m = st.mean(1)
    out = {'length_s': round(len(m) / SR, 3)}
    e1 = _env_db(m, 1.0)
    pk = int(np.argmax(e1))
    first = int(np.argmax(e1 >= e1[pk] - 20))
    out['attack_ms'] = float(pk - first)
    e10 = _env_db(m, 2.0)
    e10 = np.convolve(e10, np.ones(5) / 5, 'same')
    p10 = int(np.argmax(e10))
    for d, key in ((20, 't20_ms'), (40, 't40_ms')):
        below = np.nonzero(e10[p10:] < e10[p10] - d)[0]
        out[key] = float(below[0] * 2.0) if len(below) else None
    out['peak_t'] = round(pk / 1000, 4)
    F = np.fft.rfft(m * np.hanning(len(m)))
    P = np.abs(F) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    tot = P.sum() + 1e-20
    out['shares'] = [round(float(10 * np.log10(P[(f >= a) & (f < b)].sum() / tot + 1e-12)), 1) for a, b in REGIONS]
    cs = [1000 * 2 ** (k / 3) for k in range(-10, 11)]
    bl = []
    for c in cs:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        bl.append(max(-60.0, 10 * np.log10(P[sel].sum() / tot + 1e-12)))   # empty bands floored (as the study)
    out['tilt'] = round(float(np.polyfit(np.log2(cs), bl, 1)[0]), 2)
    L, R = st[:, 0], st[:, 1]
    out['corr'] = round(float(np.corrcoef(L, R)[0, 1]) if L.std() > 0 and R.std() > 0 else 1.0, 3)
    M, S = (L + R) / 2, (L - R) / 2
    out['side_mid_db'] = round(float(10 * np.log10((S ** 2).mean() / ((M ** 2).mean() + 1e-20) + 1e-12)), 1)
    a = int(first * SR / 1000)
    seg = m[a:a + int(0.04 * SR)]
    if len(seg) > 256:
        S1 = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8192))
        out['head_hz'] = round(float(np.fft.rfftfreq(8192, 1 / SR)[int(np.argmax(S1))]), 0)
    hop = int(0.005 * SR)
    win = np.hanning(512)
    tr = []
    for i in range(0, len(m) - 512, hop):
        fr = m[i:i + 512] * win
        lev = 10 * np.log10((fr ** 2).mean() + 1e-20)
        tr.append((lev, float(np.fft.rfftfreq(4096, 1 / SR)[int(np.argmax(np.abs(np.fft.rfft(fr, 4096))))])))
    if tr:
        top = max(l for l, _ in tr)
        act = [hz for l, hz in tr if l >= top - 30]
        out['ridge'] = {'start_hz': round(act[0]), 'end_hz': round(act[-1]), 'max_hz': round(max(act)),
                        'min_hz': round(min(act)),
                        'sweep_oct': round(float(np.log2(max(act[-1], 1.0) / max(act[0], 1.0))), 2)}
    return out


def _outside(v, lo, hi, scale):
    if v is None:
        return 1.0
    if v < lo:
        return (lo - v) / scale
    if v > hi:
        return (v - hi) / scale
    return 0.0


# per category: (measurement key, lo, hi, scale) -> penalty = sum of (distance outside / scale), capped per key
TARGETS = {
    'pro_notify': [('length_s', 0.2, 0.42, 0.1), ('attack_ms', 0.0, 15.0, 10.0), ('t20_ms', 20.0, 45.0, 20.0),
                   ('t40_ms', 150.0, 340.0, 80.0), ('head_hz', 1700.0, 2400.0, 400.0), ('low_db', -200.0, -38.0, 8.0),
                   ('corr', 0.6, 0.97, 0.2)],
    # round 3: low_db (< 500 Hz) contradicted the forge refs themselves (kalimba notes at 392-659 Hz put -0.2..-4 dB
    # there); the real-bell target only keeps rumble out (< 150 Hz) and allows rings to 2.6 s
    'notify': [('length_s', 0.5, 2.6, 0.4), ('attack_ms', 0.0, 10.0, 10.0), ('sub_db', -200.0, -20.0, 8.0)],
}


def pro_penalty(x, cat):
    """sum over the category's TARGETS of how far the render falls outside each range (in the range's own scale,
    each capped at 2); 0 = every pro target met"""
    m = measure(x)
    m['low_db'] = 10 * np.log10(sum(10 ** (s / 10) for s in m['shares'][:2]) + 1e-12)
    m['sub_db'] = m['shares'][0]
    if 'ridge' in m:
        m['sweep_oct'] = m['ridge']['sweep_oct']
    pen = 0.0
    for key, lo, hi, sc in TARGETS.get(cat, []):
        pen += min(2.0, _outside(m.get(key), lo, hi, sc))
    return pen


def extra_for(cat, weight=1.0):
    """an `extra=` check for the Critic: -weight * pro_penalty"""
    return lambda x, row: -weight * pro_penalty(x, cat)


# Round 6 (pop-notify): the round-5 family (the clean film ding on a semitone ladder; variety 0.936 on the bench, the
# five read as one patch) stays in FAMILY_R5; FAMILY['pro_notify'] = five distinct dings around the film targets: the
# round-5 film ding itself, a 1 : 2.4 FM bell (sidebands at 1.4x / 3.4x), a felt-struck wooden bar, a two-note glass
# 'ding-dong' (down a major third) and the film partials with an octave-under body. All clean (no drive, no breath),
# nothing under 500 Hz, heads on the film's semitone ladder (1822-2296 Hz).
FAMILY_R5 = {'pro_notify': [dict(p) for p in FAMILY['pro_notify']]}
FAMILY_NAMES_R5 = {'pro_notify': list(FAMILY_NAMES['pro_notify'])}
_CH = dict(style='chime', drive=0.0, head_drive=0.0, hp=500.0, verb=0.14, rt60=0.3, width=0.7, fade=0.05, detune=0.8)
FAMILY_R6 = {'pro_notify': [
    dict(FAMILY_R5['pro_notify'][0]),                                                      # the film ding (round 5)
    _fam(**{**_CH, 'f0': _LADDER[1], 'mat': 'fm', 'att': 0.005, 'c_fast': 0.9, 'c_tf': 0.011, 'c_ts': 0.16,
            'c_index': 1.8, 'c_itau': 0.015, 'c_mallet': -40.0, 'c_hard': 0.4, 'length': 0.3, 'detune': 0.0}),   # FM bell
    _fam(**{**_CH, 'f0': _LADDER[4], 'mat': 'wood', 'att': 0.002, 'c_fast': 0.95, 'c_tf': 0.007, 'c_ts': 0.09,
            'c_mallet': -14.0, 'c_hard': 0.1, 'length': 0.22, 'c_bend': 40.0, 'detune': 0.0, 'verb': 0.1}),     # felt on wood
    _fam(**{**_CH, 'f0': _LADDER[3], 'mat': 'glass', 'att': 0.0015, 'c_fast': 0.9, 'c_tf': 0.012, 'c_ts': 0.12,
            'c_mallet': -26.0, 'c_hard': 0.7, 'length': 0.4, 'c_motif': 'up', 'interval': -4, 'gap': 0.11,
            'second_db': -2.0}),                                                           # glass ding-dong
    _fam(**{**_CH, 'f0': _LADDER[2], 'mat': 'film', 'att': 0.0015, 'c_fast': 0.9, 'c_tf': 0.013, 'c_ts': 0.1,
            'c_mallet': -30.0, 'c_hard': 0.7, 'length': 0.3, 'c_sub': 0.1}),               # film partials + octave body
]}
FAMILY_NAMES_R6 = {'pro_notify': ['film_ding', 'fm_bell', 'wood', 'ding_dong', 'octave_body']}
# NOT delivered (held-out bench, one run at 19:10): variety 0.936 -> 0.814, phrase z 2.16 -> 1.45, margin 0.38 -> -0.03;
# by blind_v2.verdict (no real variety row for pro_notify) that is models 0 up / 2 down, so FAMILY['pro_notify'] stays
# the round-5 family (out/forge_v2/pro_notify/final/); the round-6 palette is in final_r6/ for a swap by ear.


# ------------------------------------------------------------------------------------------------ earshot 0.4: decay
# The decay knob (earshot 0.4). The music films' rounded mid ding rings 0.36 s
# (0.24-0.5), 20 dB down 116 ms (25-229) after a 36 ms (16-121) onset, at ~690 Hz with a tonal share of 0.6; the 0.3
# knob pass got notify/m1 / m3 to 600 Hz and 0.36 s only by fading the end of a 2 s ring (dur), so the ring stayed 2.6x
# too slow, the onset 3.5x too hard and the audit called every such take truncated. Two keys fix that inside the
# recipe: `ring` scales every decay constant of the style (bell: decay, early_tau, strike_tau; ding / success:
# hold_tau, ring_tau, body_tau, air_tau, p3_tau; chime: c_tf, c_ts, c_itau), so a note rings DOWN to silence within its
# file; `soft` (s) is a felt mallet's onset, a raised-cosine rise over the whole note (mallet and tick included).
# Both are off by default (ring 1, soft 0): every earlier member renders sample for sample. knobs.py aims `ring` at a
# t20 target (decay=) and `soft` at an attack target (onset=). bell_len (s, 0 = off) replaces the bell style's 0.5 s
# minimum file length for a short note (the ring must still reach -45 dB inside it: knobs set it from the decay).
SPACE.update({'ring': (0.1, 4.0, 'log'), 'soft': (0.0, 0.08), 'bell_len': (0.0, 2.6)})
DEFAULTS.update({'ring': 1.0, 'soft': 0.0, 'bell_len': 0.0})
RING_KEYS = {'bell': ('decay', 'early_tau', 'strike_tau'),
             'ding': ('hold_tau', 'ring_tau', 'body_tau', 'air_tau', 'p3_tau'),
             'chime': ('c_tf', 'c_ts', 'c_itau')}
RING_KEYS['success'] = RING_KEYS['ding']


def ring_scaled(p, ring=None):
    """p with every decay constant of its style multiplied by `ring` (p['ring'] when None): the decay knob"""
    ring = p.get('ring', 1.0) if ring is None else ring
    if ring == 1.0:
        return dict(p)
    keys = RING_KEYS.get(p.get('style', 'ding'), ())
    return {**p, **{k: p[k] * ring for k in keys if k in p}}


def _soft(y, soft):
    """a felt mallet: a raised-cosine rise over `soft` s on the whole note"""
    k = min(len(y), dsp.n_of(soft))
    if k > 1:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
        y = np.array(y, np.float64)
        y[:k] *= w[:, None] if y.ndim == 2 else w
    return y


# The earshot 0.4 mid ding: the music films' rounded counter / success ding. A struck bell (style 'bell')
# cannot get there: its tonal share stays >= 0.8 and its centroid 1.6x high at every hardness (90 candidates: best
# profile distance 0.64). A struck tine with its octave-under body (style 'chime', c_sub ~0.3-0.4: two lines an octave
# apart give the films' tonal share of ~0.6 and pull the centroid to ~640-830 Hz) does, with no drive: the decay knob
# (knobs.apply(decay=116)) and the onset knob (onset=36) set `ring` and `soft`. Profile distance 0.20 / 0.29, all 7 /
# 6 of 7 features inside the films' IQRs (the earshot 0.4 study).
_MID = dict(style='chime', drive=0.0, c_motif='single')
FAMILY_MID = {'mid_ding': [
    _fam(**{**_MID, 'f0': 830.61, 'mat': 'tine', 'att': 0.00316, 'detune': 0.9, 'verb': 0.255, 'rt60': 0.358,
            'width': 0.313, 'length': 0.425, 'fade': 0.0435, 'hp': 135.6, 'c_fast': 0.623, 'c_tf': 0.0187,
            'c_ts': 0.0623, 'c_mallet': -59.0, 'c_hard': 1.0, 'c_bend': 18.8, 'c_sub': 0.45, 'c_index': 1.655,
            'c_itau': 0.0097, 'ring': 0.82, 'soft': 0.049}),                    # a felt-struck tine over its octave
    _fam(**{**_MID, 'f0': 880.0, 'mat': 'wood', 'att': 0.00814, 'detune': 1.68, 'verb': 0.35, 'rt60': 0.363,
            'width': 0.229, 'length': 0.317, 'fade': 0.0613, 'hp': 254.7, 'c_fast': 0.596, 'c_tf': 0.0291,
            'c_ts': 0.077, 'c_mallet': -25.3, 'c_hard': 0.058, 'c_bend': 22.9, 'c_sub': 0.45, 'c_index': 4.38,
            'c_itau': 0.0064, 'ring': 0.65, 'soft': 0.047}),                     # a soft wooden bar, a step higher
# (the two lines an octave apart are near level, so which is the strongest bin flips from take to take: the measured
# tonal share of one member spans 0.55-0.7 over seeds, its centroid 600-760 Hz; c_sub 0.45 keeps both inside the IQRs)
]}
FAMILY_NAMES_MID = {'mid_ding': ['tine', 'wood']}
