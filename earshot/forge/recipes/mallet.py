"""Gated mallet notes (glockenspiel, celesta, metallophone, marimba), a short soft pluck, and a chime-run builder,
written in code; every sample is computed, no recorded audio anywhere.

    from earshot.forge import dsp
    from earshot.forge.recipes import mallet
    x = mallet.render(mallet.FAMILY['mallet'][0], dsp.rng(1))                 # one gated glockenspiel note, stereo
    run = mallet.render_run(mallet.RUNS['chime_run'][1], dsp.rng(1))          # gated notes, panned note by note

What the films' mallet sounds are (earshot 0.4 catalog, music films, high + medium events >= 8 dB clear of the music):
  * glockenspiel / celesta notes (28 events, 14 films): 0.33 s (0.28-0.44), 25 ms (16-45) to the peak, 20 dB down 129 ms
    (38-256) after it, centroid 1079 Hz (550-1505), nothing under 150 Hz, tonal 0.62 (0.55-0.68), WIDE: L/R correlation
    0.67 (0.33-0.99). The notes are gated: they stop within ~0.35 s, where a free bar rings 1.5-3 s.
  * short plucks (21 events, 8 films): 0.3 s (0.25-0.35), 49 ms (15-131) to the peak, t20 112 ms (104-150), centroid
    359 Hz (335-809), 12 % under 150 Hz, tonal 0.58, mono: a soft low pluck (a muted string, a felt-struck bar).
  * chime runs (13 events, 11 films): 0.35-1 s of discrete gated notes ~110 ms apart, centroid ~980 Hz.
The 0.3 kalimba rings 5x too long (1.7-2.7 s), stays mono, and faded to 0.33 s the audit calls it truncated.

The note (style 'note'): a bar's partial set (`bar`, BARS: ratios, amplitudes) on `f0`, each partial decaying with
tau = `decay` x (ratio) ** -`slope`, a mallet whose hardness (`hard`) sets how strongly it excites the upper partials
(and how bright its contact noise is: `strike` dB), a slow doublet beat (`beat` Hz: the two bending planes), a soft
rise (`onset`), and a DAMPER: from `gate` s on, every partial dies with `gate_tau` (a muted bar, a gated sample: the
note stops in a few tens of ms instead of being faded), and an optional octave double (`oct` dB at `oct_ratio` x the
pitch: a second bar an octave off, as film scores double a glockenspiel); then the finish: band limits, `width` (a
decorrelated side) and `pan` (a per-note position), a small room.
The pluck (style 'pluck'): a string's harmonics (k = 1 .. 12, amplitude k ** -`tilt` shaped by where it is plucked:
`pos`), the upper ones dying faster (`slope`), a soft finger (`onset`, `contact` low-pass), muted by the same damper,
plus a body resonance (`body_f`, `body`).
A run (render_run): `notes` notes `step` s apart on a scale (`scale`, `pattern` up / down / arc, `span` scale steps),
each a gated note of these params, panned note by note (`spread`), the last one accented (`accent` dB) and allowed to
ring (`last_gate` x the gate).
CHARACTER keys: style, bar, f0, gate, pan / spread; FAMILY['mallet'] (glock, pluck, celesta, marimba) /
RUNS['chime_run'] (flourish, up4) = the palettes; RUN = a recipe object over whole runs (render blocks name
render_run).
"""
import types

import numpy as np

from .. import dsp
from . import _tonal as T

SR = dsp.SR
TARGET = 'a short glockenspiel note'
ALTS = ['a wine glass clink', 'a phone notification', 'a xylophone note', 'a triangle', 'a music box']
STUDY = {'dur_s': (0.28, 0.44, 0.1), 'attack_ms': (16.0, 45.0, 12.0), 't20_ms': (38.0, 256.0, 60.0),
         'centroid_hz': (550.0, 1505.0, 300.0), 'lo150': (0.0, 0.04, 0.05), 'tonal': (0.55, 0.68, 0.1),
         'corr': (0.33, 0.99, 0.2)}
STUDY_PLUCK = {'dur_s': (0.25, 0.35, 0.08), 'attack_ms': (15.0, 131.0, 20.0), 't20_ms': (104.0, 150.0, 30.0),
               'centroid_hz': (335.0, 809.0, 100.0), 'lo150': (0.02, 0.17, 0.08), 'tonal': (0.4, 0.62, 0.1),
               'corr': (0.95, 0.99, 0.1)}

BARS = {   # (ratio, amplitude): free / tuned bars
    'glock': [(1.0, 1.0), (2.756, 0.35), (5.404, 0.12), (8.933, 0.05)],     # glockenspiel: a free-free steel bar
    'celesta': [(1.0, 1.0), (2.0, 0.05), (3.98, 0.22), (6.9, 0.05)],        # a steel bar over a resonator
    'metal': [(1.0, 1.0), (3.9, 0.25), (9.9, 0.08)],                        # a tuned metallophone bar (1 : 4 : 10)
    'marimba': [(1.0, 1.0), (3.92, 0.3), (9.8, 0.08)],                      # wood, the same tuning, dies faster
    'tine': [(1.0, 1.0), (5.8, 0.12), (6.27, 0.2)],                         # a clamped tine (music box / kalimba)
}

SPACE = {
    'style': ['note', 'pluck'],
    'bar': list(BARS),
    'f0': (150.0, 3200.0, 'log'),
    'decay': (0.05, 1.6, 'log'),       # the fundamental's time constant (before the damper)
    'slope': (0.3, 1.6),               # upper partials die faster: tau ~ ratio ** -slope
    'hard': (0.0, 1.0),                # mallet hardness: excites the upper partials, brightens the contact
    'strike': (-60.0, -6.0),           # the mallet's contact noise, dB re the note
    'beat': (0.0, 6.0),                # doublet beat (Hz)
    'onset': (0.0005, 0.06, 'log'),    # the soft rise to the peak
    'gate': (0.05, 1.0, 'log'),        # the damper engages here (s) ...
    'gate_tau': (0.005, 0.08, 'log'),  # ... and every partial then dies with this time constant
    'tilt': (0.6, 2.5),                # pluck: harmonic amplitudes k ** -tilt
    'pos': (0.05, 0.5),                # pluck: where the string is plucked (fraction of its length)
    'contact': (0.0003, 0.01, 'log'),  # pluck: the finger's contact (low-passes the pluck)
    'body': (-60.0, 0.0),              # pluck: a body resonance, dB re the string
    'body_f': (90.0, 400.0, 'log'),
    'hp': (60.0, 800.0, 'log'),
    'lp': (3000.0, 16000.0, 'log'),
    'width': (0.0, 1.0),
    'pan': (-1.0, 1.0),
    'room': (0.0, 0.35),
    'rt': (0.1, 0.8, 'log'),
    'oct': (-60.0, 0.0),               # an octave double (dB re the note; -60 = none) ...
    'oct_ratio': [0.5, 2.0],           # ... an octave under or over
    'jit': (0.0, 1.0),                 # take-to-take variation (decay +-8 %, gate +-5 %, hardness +-0.04 at 1; the
}                                      # note itself never moves)
DEFAULTS = {'style': 'note', 'bar': 'glock', 'f0': 1046.5, 'decay': 0.6, 'slope': 0.9, 'hard': 0.5, 'strike': -26.0,
            'beat': 1.2, 'onset': 0.004, 'gate': 0.2, 'gate_tau': 0.03, 'tilt': 1.4, 'pos': 0.18, 'contact': 0.003,
            'body': -14.0, 'body_f': 180.0, 'hp': 300.0, 'lp': 12000.0, 'width': 0.5, 'pan': 0.0, 'room': 0.1,
            'rt': 0.3, 'oct': -60.0, 'oct_ratio': 0.5, 'jit': 0.5}
FINE = ['decay', 'slope', 'hard', 'strike', 'beat', 'onset', 'gate', 'gate_tau', 'tilt', 'pos', 'contact', 'body',
        'body_f', 'hp', 'lp', 'width', 'room', 'rt', 'oct']


def _fam(**kw):
    return {**DEFAULTS, **kw}


FAMILY = {'mallet': [
    _fam(f0=1274.0, decay=0.5027, slope=0.8703, hard=0.6343, strike=-14.89, beat=4.807, onset=0.00162, gate=0.08478,
         gate_tau=0.0422, tilt=0.6, pos=0.4314, contact=0.00436, body=-21.52, body_f=360.3, hp=194.3, lp=6499.0,
         width=0.3488, room=0.2474, rt=0.4467, oct=-2.483, jit=0.4495),          # a gated glockenspiel note, doubled
    _fam(style='pluck', f0=284.2, decay=0.1234, slope=0.3789, hard=0.1259, strike=-49.8, beat=2.786, onset=0.06,
         gate=0.1256, gate_tau=0.01994, tilt=0.7913, pos=0.07761, contact=0.00563, body=-40.86, body_f=125.1, hp=372.5,
         lp=4270.0, width=0.08427, room=0.07512, rt=0.6068, jit=0.02373),       # a soft low pluck
    _fam(bar='celesta', f0=784.0, decay=0.7148, slope=0.4249, hard=0.5432, strike=-49.98, beat=1.934, onset=0.00548,
         gate=0.08034, gate_tau=0.04521, hp=678.6, lp=8593.0, width=0.8211, room=0.1831, rt=0.4,
         jit=0.4),                                                              # a wide celesta note (G5)
    _fam(bar='marimba', f0=392.0, decay=1.6, slope=0.3, hard=0.45, strike=-16.89, beat=1.747, onset=0.004282,
         gate=0.06156, hp=120.0, lp=16000.0, width=0.3335, room=0.1626, rt=0.6887,
         jit=0.4),                                                              # a short wooden marimba note (G4)
]}
FAMILY_NAMES = {'mallet': ['glock', 'pluck', 'celesta', 'marimba']}
# How they were found (the earshot 0.4 study; local, not shipped): random + local searches on the music films' profiles.
# A struck bar alone stays too pure (tonal share 0.8-0.95 where the films' bell notes sit at 0.62) and too bright; the
# same note doubled an octave under, 2.5 dB down (`oct`), lands all 7 features inside the bell-note IQRs (profile d
# 0.17-0.32 over takes). 'pluck' is a felt-soft string pluck (profile d 0.34-0.38, 5-6 of 7 inside). Added for the
# 0.4 bank (a model-free search from a hand-set corner, every candidate's takes audited): 'celesta', a wide G5 note
# (L/R correlation ~0.5) gated at 80 ms, and 'marimba', a dry wooden G4 note; both purer than the typical bell note.


def family_space(params):
    return T.family_space(SPACE, {**DEFAULTS, **params}, FINE)


def _damper(t, gate, gate_tau):
    """1 until the damper engages at `gate` s, then an exponential stop with gate_tau (multiplied onto each partial's
    own decay: the note keeps its timbre while it is muted)"""
    return np.exp(-np.maximum(t - gate, 0) / max(gate_tau, 1e-4))


def _vary(p, r):
    """one take's small differences (decay, damper time, hardness): a played note, never a copy"""
    j = p['jit'] * r.standard_normal(3)
    return {**p, 'decay': p['decay'] * np.exp(0.08 * j[0]), 'gate': p['gate'] * np.exp(0.05 * j[1]),
            'hard': float(np.clip(p['hard'] + 0.04 * j[2], 0, 1))}


def _note(p, r, n, f0):
    """one struck bar note (mono, the strike at t = 0)"""
    t = dsp.t_of(n)
    y = np.zeros(n)
    damp = _damper(t, p['gate'], p['gate_tau'])
    for q, a in BARS[p['bar']]:
        f = f0 * q
        if f >= 0.42 * SR:
            continue
        a = a * q ** (1.5 * (p['hard'] - 0.5))                 # a hard mallet lifts the upper partials
        tau = max(p['decay'] * q ** -p['slope'], 0.003)
        b = p['beat'] * r.uniform(0.6, 1.4)
        am = 1 - 0.3 * (0.5 - 0.5 * np.cos(2 * np.pi * b * t)) if b > 0.01 else 1.0
        y += a * np.exp(-t / tau) * am * np.sin(2 * np.pi * f * t + r.uniform(0, 2 * np.pi))
    y *= damp
    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / max(p['onset'], 1e-4), 0, 1))
    y *= rise
    pk = np.abs(y).max() + 1e-12
    if p['strike'] > -59:                                      # the mallet's contact: brighter when hard
        lo = 800.0 + 3000.0 * p['hard']
        y += pk * 10 ** (p['strike'] / 20) * T.burst(n, r, lo, min(lo * 4.0, 16000.0), 0.0006 + 0.002 * (1 - p['hard']),
                                                    rise=max(0.0002, 0.3 * p['onset']))
    return y


def _pluck(p, r, n, f0):
    """one soft pluck (mono): a string's harmonics shaped by the pluck point, upper ones dying faster, a finger's soft
    contact, the damper, a body resonance"""
    t = dsp.t_of(n)
    y = np.zeros(n)
    for k in range(1, 13):
        f = f0 * k * (1 + 0.0004 * k * k)                      # a little stiffness
        if f >= 0.4 * SR:
            break
        a = k ** -p['tilt'] * abs(np.sin(np.pi * k * p['pos'])) / (1 + (2 * np.pi * f * p['contact'] / 6) ** 2)
        tau = max(p['decay'] * k ** -p['slope'], 0.004)
        y += a * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + r.uniform(0, 0.3))
    y *= _damper(t, p['gate'], p['gate_tau'])
    y *= 0.5 - 0.5 * np.cos(np.pi * np.clip(t / max(p['onset'], 1e-4), 0, 1))
    if p['body'] > -59:                                        # the instrument's body, driven by the string
        bd = dsp.svf(y, p['body_f'], 3.0, 'bandn')
        y = y + 10 ** (p['body'] / 20) * bd / (np.abs(bd).max() + 1e-12) * np.abs(y).max()
    return y


def _voice(p, r, n, f0):
    """one note of the style, doubled `oct` dB at `oct_ratio` x its pitch when oct > -59 (a second bar an octave off:
    the celesta / glockenspiel doubling of film scores; two lines an octave apart bring the tonal share down to the
    films' ~0.6 without noise)"""
    fn = _pluck if p['style'] == 'pluck' else _note
    y = fn(p, r, n, f0)
    if p['oct'] > -59:
        y = y + 10 ** (p['oct'] / 20) * fn(p, r, n, f0 * p['oct_ratio'])
    return y


def _len(p):
    return min(2.0, p['gate'] + 7.0 * p['gate_tau'] + 0.02, max(p['onset'] + 7.0 * p['decay'], 0.06) + 0.02)


def render(p, r):
    p = _vary({**DEFAULTS, **p}, r)
    n = dsp.n_of(_len(p))
    y = _voice(p, r, n, p['f0'])
    return T.finish(y, r, hp=p['hp'], lp=p['lp'], room_wet=p['room'], rt60=p['rt'], width=p['width'], cut=500.0,
                    pan=p['pan'], size=0.3)


# ------------------------------------------------------------------------------------------------ the run builder
SCALES = {'penta': [0, 2, 4, 7, 9], 'major': [0, 2, 4, 5, 7, 9, 11], 'lydian': [0, 2, 4, 6, 7, 9, 11],
          'sus': [0, 2, 5, 7, 10]}
RUN_SPACE = {**SPACE, 'notes': (2, 8, 'int'), 'step': (0.05, 0.18), 'swing': (0.0, 0.3), 'scale': list(SCALES),
             'pattern': ['up', 'down', 'arc'], 'span': (1, 3, 'int'), 'spread': (0.0, 1.0), 'accent': (-6.0, 6.0),
             'last_gate': (1.0, 4.0), 'ramp': (-6.0, 6.0)}
RUN_DEFAULTS = {**DEFAULTS, 'notes': 4, 'step': 0.11, 'swing': 0.0, 'scale': 'penta', 'pattern': 'up', 'span': 1,
                'spread': 0.6, 'accent': 2.0, 'last_gate': 2.0, 'ramp': 0.0}


def run_freqs(p):
    """the run's note frequencies: `notes` steps of `span` scale degrees from f0 along `pattern`"""
    sc = SCALES[p['scale']]
    k = int(p['notes'])
    idx = [i * int(p['span']) for i in range(k)]
    if p['pattern'] == 'down':
        idx = idx[::-1]
    elif p['pattern'] == 'arc':
        h = (k + 1) // 2
        idx = idx[:h] + idx[:k - h][::-1]
    st = [12 * (i // len(sc)) + sc[i % len(sc)] for i in idx]
    return [p['f0'] * 2 ** (s / 12) for s in st]


def render_run(p, r):
    """a chime run (module doc): discrete gated notes, panned note by note, the last accented and ringing longer"""
    p = {**RUN_DEFAULTS, **p}
    fs = run_freqs(p)
    k = len(fs)
    last = {**p, 'gate': p['gate'] * p['last_gate']}
    times = [i * p['step'] * (1 + (p['swing'] if i % 2 else 0.0)) for i in range(k)]
    n = dsp.n_of(times[-1] + _len(last) + 0.05)
    out = np.zeros((n, 2))
    for i, (f, t0) in enumerate(zip(fs, times)):
        q = _vary(last if i == k - 1 else p, r)
        g = 10 ** ((p['ramp'] * (i / max(1, k - 1)) + (p['accent'] if i == k - 1 else 0.0)) / 20)
        pan = p['pan'] + p['spread'] * (((i % 2) * 2 - 1) * (0.4 + 0.6 * i / max(1, k - 1)))
        m = dsp.n_of(_len(q))
        y = _voice(q, r, m, f)
        st = T.finish(y, r, hp=p['hp'], lp=p['lp'], width=p['width'], cut=500.0, pan=float(np.clip(pan, -1, 1)),
                      end_db=-70.0, fade=0.004)
        i0 = dsp.n_of(t0)
        j = min(len(st), n - i0)
        out[i0:i0 + j] += g * st[:j]
    out = T.room(out, r, p['room'], p['rt'], size=0.3, width=0.3 + p['width'])
    out = dsp.trim_tail(out, -60.0, 0.01)
    kf = min(len(out), dsp.n_of(0.006))
    out[-kf:] *= (np.cos(np.linspace(0, np.pi / 2, kf)) ** 2)[:, None]
    return out


RUNS = {'chime_run': [
    dict(RUN_DEFAULTS, f0=802.3, decay=0.06476, slope=0.4859, hard=0.3246, strike=-51.63, beat=2.417, onset=0.00098,
         gate=0.2162, gate_tau=0.04794, tilt=0.7552, pos=0.3692, contact=0.00257, body=-19.32, body_f=400.0, hp=142.8,
         lp=3000.0, width=0.01723, room=0.09026, rt=0.2059, jit=0.5908, notes=3, step=0.04412, swing=0.06969,
         scale='lydian', pattern='arc', span=2, spread=0.1809, accent=-4.474, last_gate=3.365),   # a quick flourish
    dict(RUN_DEFAULTS, f0=1046.5, notes=4, step=0.11, span=2, gate=0.12, gate_tau=0.03),   # up in thirds, 110 ms
]}
RUN_NAMES = {'chime_run': ['flourish', 'up4']}
# 'flourish' is the music films' chime run (profile search: 3 gated notes in an arc, 44 ms apart, near mono; profile d
# 0.50 with all 7 features inside the IQRs); 'up4' a plain run, notes ~110 ms apart (the reference runs' spacing),
# a skip of two scale degrees per note: steps of 2-3 semitones read as one rising glide to the audit.
# a recipe object for forge() / palette() over whole runs (as pop.V2): its render blocks name render_run
RUN = types.SimpleNamespace(__name__='earshot.forge.recipes.mallet', SPACE=RUN_SPACE, DEFAULTS=RUN_DEFAULTS,
                            render=render_run, TARGET='a quick run of chime notes',
                            ALTS=['a music box melody', 'a harp glissando', 'a phone ringtone', 'a xylophone scale'])
