"""A kalimba (thumb piano) tine pluck, built from physics; every sample is computed, no recorded audio anywhere.

    from earshot.forge import dsp
    from earshot.forge.recipes import pluck
    x = pluck.render({**pluck.DEFAULTS, 'f0': pluck.note('C5')}, dsp.rng(1))     # 48 kHz stereo, one C5 pluck

What a kalimba note is, physically, and how each part is modelled:
  * The tine is a steel cantilever (clamped at the bridge, free at the tip). A clamped-free bar's bending modes are
    strongly inharmonic: Euler-Bernoulli gives 1 : 6.27 : 17.55 : 34.39 (real tines, with the tuning mass of their
    length and a curved tip, sit at ~5-7x for the second mode). The first mode is the note; the second is the bright
    metallic "ping" in the attack, and it dies far faster than the fundamental (air and bridge losses grow with
    frequency): taus fall roughly as f^-0.9.
  * Two polarisations: the tine bends up / down and, slightly stiffer, sideways. A pluck is never perfectly vertical,
    so the note is two close modes (beat_hz apart): the slow shimmer of a real kalimba note.
  * Geometric nonlinearity: a large deflection stretches the tine, giving a weak 2nd harmonic that follows the
    SQUARED envelope (it decays twice as fast as the fundamental).
  * The thumb: the flesh pushes the tine down for a few ms, then the tine slips off. The release is soft (the contact
    has a width thumb_ms), so high modes are excited less (a low-pass on the excitation, ~1 / (1 + (f T)^2)), plus a
    little skin / nail noise on the slip.
  * The box: the tine drives the bridge, the bridge drives a wooden box: a Helmholtz resonance (the sound hole) and a
    cloud of short wooden plate modes (150 Hz - 2.5 kHz, 3-30 ms) that colour the attack (the "wooden" knock in it).
  * Sympathetic tines: every other tine on the bridge is driven a little (strongest where its note coincides with a
    partial of the played note: octaves, fifths); they ring up slowly and keep a faint chord under the note (clearly
    visible in real kalimba recordings as the other tines' lines).
  * A small room, a close microphone with a noise floor, a little stereo from the room.
Pitch: `f0` (Hz) renders the note at an exact pitch (note('C5') = 523.25 Hz, A4 = 440). Without f0 a forge run picks
one of the reference recordings' pitches per render (G4 A4 B4 D5 E5) so the critic compares like with like.
"""
import numpy as np
from scipy import signal

from .. import dsp
from . import REF_ROOT

SR = dsp.SR
TARGET = 'a kalimba note, a thumb piano pluck'
ALTS = ['a music box', 'a xylophone note', 'a glockenspiel', 'an electric piano note', 'a synthesizer pluck']
_K = REF_ROOT + '/kalimba/'
REFS = [_K + f for f in ('fs-681787_kalimba-G.ogg', 'fs-681788_kalimba-A.ogg', 'fs-681789_kalimba-B.ogg',
                         'fs-681791_kalimba-D-high.ogg', 'fs-681792_kalimba-E-high.ogg')]
REF_F0 = [392.0, 440.0, 493.9, 587.3, 659.3]             # the references' notes (measured: G4 A4 B4 D5 E5)
# a 17-tine kalimba in C (C4 .. E6): the tines that can ring sympathetically
_SCALE = [0, 2, 4, 5, 7, 9, 11]
TINES = [261.63 * 2 ** ((12 * o + s) / 12) for o in range(3) for s in _SCALE][:17]

SPACE = {
    'p2_ratio': (5.0, 7.0),            # second bending mode / fundamental (ideal bar 6.27)
    'p2_lvl': (0.02, 0.6, 'log'),      # its level at the pluck (before the thumb's low-pass)
    'p3_lvl': (0.0, 0.25),             # third bending mode (2.8 x the second)
    'tau1': (0.35, 2.4, 'log'),        # fundamental decay at C5 (higher notes shorter, ~ f^-0.5)
    'tau_exp': (0.5, 1.4),             # upper modes: tau ~ tau1 * (f / f0)^-tau_exp
    'beat_hz': (0.2, 4.0, 'log'),      # polarisation split
    'beat_lvl': (0.0, 0.6),
    'h2_lvl': (0.0, 0.25),             # 2nd harmonic from the large-deflection nonlinearity
    'thumb_ms': (0.1, 4.0, 'log'),     # contact width of the release (soft flesh = long = darker)
    'zing_lvl': (0.0, 0.4),            # round 2: the release slap + bridge rattle: a short metallic buzz at f0's period
    'zing_f': (2500, 12000, 'log'),
    'zing_tau': (0.004, 0.06, 'log'),
    'skin_lvl': (0.0, 0.25),           # skin / nail slip noise
    'skin_f': (1500, 9000, 'log'),
    'helm_f': (160, 420, 'log'),       # box Helmholtz resonance
    'helm_tau': (0.008, 0.06, 'log'),
    'box_lvl': (0.0, 0.8),             # the wooden box's share (drives through the bridge)
    'box_lo': (150, 600, 'log'),
    'box_tau': (0.003, 0.03, 'log'),
    'knock_lvl': (0.0, 0.8),           # the thumb's push on the instrument: a low thud of the box / table
    'knock_f': (40, 160, 'log'),
    'knock_tau': (0.008, 0.08, 'log'),
    'symp_lvl': (0.0, 0.25),           # sympathetic tines
    'symp_rise': (0.005, 0.08, 'log'),
    'room_wet': (0.0, 0.35),
    'rt60': (0.15, 0.9, 'log'),
    'width': (0.0, 0.5),
    'air_db': (-9.0, 0.0),
    'floor_db': (-88.0, -58.0),
    'rumble': (0.0, 0.8),              # low-frequency share of the room tone (real recordings are never silent)
}
# round 2 pick 1 (launch film forge: the launch-film forge run (pluck), tone=1.5)
DEFAULTS = {'p2_ratio': 5.0, 'p2_lvl': 0.386, 'p3_lvl': 0.2038, 'tau1': 0.3529, 'tau_exp': 1.309, 'beat_hz': 2.759,
            'beat_lvl': 0.01985, 'h2_lvl': 0.1751, 'thumb_ms': 0.3572, 'zing_lvl': 0.09041, 'zing_f': 6908.0,
            'zing_tau': 0.006356, 'skin_lvl': 0.2027, 'skin_f': 1500.0, 'helm_f': 197.2, 'helm_tau': 0.01108,
            'box_lvl': 0.1924, 'box_lo': 545.6, 'box_tau': 0.01542, 'knock_lvl': 0.5643, 'knock_f': 77.94,
            'knock_tau': 0.03731, 'symp_lvl': 0.02239, 'symp_rise': 0.07959, 'room_wet': 0.2094, 'rt60': 0.4881,
            'width': 0.0, 'air_db': -7.11, 'floor_db': -64.71, 'rumble': 0.3376}
LEN = 3.0
_NOTES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note(name, a4=440.0):
    """'C5' -> 523.25 Hz ('F#4', 'Bb3' work too)"""
    s = _NOTES[name[0].upper()]
    i = 1
    while name[i] in '#b':
        s += 1 if name[i] == '#' else -1
        i += 1
    octave = int(name[i:])
    return a4 * 2 ** ((s + 12 * (octave + 1) - 69) / 12)


def _t(n):
    return np.arange(n) / SR


def _mode(t, f, tau, a, r, attack):
    """one damped mode; its onset follows the thumb's release (raised-cosine over `attack` s)"""
    ph = r.uniform(0, 2 * np.pi)
    y = a * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + ph * 0.15)       # released from rest: near-sine phase
    k = max(2, dsp.n_of(attack))
    if k < len(y):
        y[:k] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    return y


def _tine(p, f0, n, r):
    t = _t(n)
    T = p['thumb_ms'] / 1000
    lp = lambda f: 1.0 / (1.0 + f * T * 0.5)                            # the soft release excites high modes less
    tau1 = p['tau1'] * (f0 / 523.25) ** -0.5
    att = max(T * 0.35, 0.0002)
    y = _mode(t, f0, tau1, 1.0, r, att)
    if p['beat_lvl'] > 0.005:                                           # the second polarisation
        y += _mode(t, f0 + p['beat_hz'], tau1 * 0.85, p['beat_lvl'], r, att)
    f2 = f0 * p['p2_ratio']
    y += _mode(t, f2, tau1 * (f2 / f0) ** -p['tau_exp'], p['p2_lvl'] * lp(f2) / lp(f0), r, att)
    f3 = f2 * 2.8
    if p['p3_lvl'] > 0.005 and f3 < SR * 0.45:
        y += _mode(t, f3, tau1 * (f3 / f0) ** -p['tau_exp'], p['p3_lvl'] * lp(f3) / lp(f0), r, att)
    f4 = f2 * 5.5                                                       # fourth bending mode (34.4 / 6.27)
    if p['p3_lvl'] > 0.005 and f4 < SR * 0.45:
        y += _mode(t, f4, tau1 * (f4 / f0) ** -p['tau_exp'], 0.5 * p['p3_lvl'] * lp(f4) / lp(f0), r, att)
    if p['h2_lvl'] > 0.005:                                             # nonlinearity: squared envelope
        y += _mode(t, 2 * f0, tau1 / 2, p['h2_lvl'], r, att)
    return y


def _symp(p, f0, n, r):
    """the other tines, driven through the bridge: strongest where a tine coincides with a partial of the note"""
    out = np.zeros(n)
    if p['symp_lvl'] < 0.003:
        return out
    t = _t(n)
    rise = 1 - np.exp(-t / p['symp_rise'])
    for ft in TINES:
        if abs(ft / f0 - 1) < 0.03:
            continue
        cents = min(abs(1200 * np.log2(ft / (f0 * h))) for h in (0.5, 1, 1.5, 2, 3, 4))
        a = p['symp_lvl'] * (0.15 + np.exp(-(cents / 30.0) ** 2)) * r.uniform(0.5, 1.0)
        tau = p['tau1'] * (ft / 523.25) ** -0.5 * 1.2
        out += a * np.sin(2 * np.pi * ft * (1 + 0.0015 * r.standard_normal()) * t + r.uniform(0, 6.3)) \
            * np.exp(-t / tau) * rise
    return out


def _box_ir(p, r):
    """the wooden box seen from the bridge: Helmholtz mode + short plate modes"""
    n = dsp.n_of(0.25)
    t = _t(n)
    ir = np.zeros(n)
    ir[0] = 1.0
    ir += 0.9 * np.sin(2 * np.pi * p['helm_f'] * t) * np.exp(-t / p['helm_tau']) * 2 * np.pi * p['helm_f'] / SR * 6
    cnt = 18
    f = np.exp(r.uniform(np.log(p['box_lo']), np.log(2500.0), cnt))
    for fi in f:
        tau = p['box_tau'] * (fi / p['box_lo']) ** -0.5
        ir += r.uniform(0.2, 1.0) * np.sin(2 * np.pi * fi * t + r.uniform(0, 6.3)) * np.exp(-t / tau) \
            * 2 * np.pi * fi / SR * 2
    return ir


def _zing(p, f0, n, r):
    """the tine slapping free of the thumb and rattling on the bridge saddle: band-passed noise, gated at the period
    of the note (a buzz, not a hiss), dying in tens of ms"""
    m = min(n, dsp.n_of(0.4))
    t = _t(m)
    w = dsp.butter(r.standard_normal(m), 'band', (p['zing_f'] * 0.5, min(p['zing_f'] * 1.8, SR * 0.45)), 2)
    gate = np.maximum(0.0, np.sin(2 * np.pi * f0 * t)) ** 4
    y = w * (0.35 + gate) * np.exp(-t / p['zing_tau'])
    out = np.zeros(n)
    out[:m] = y / (np.abs(y).max() + 1e-12)
    return out


def _skin(p, n, r):
    m = dsp.n_of(0.02)
    w = r.standard_normal(m) * np.exp(-_t(m) / 0.003)
    w = dsp.butter(w, 'band', (p['skin_f'] * 0.5, min(p['skin_f'] * 2.0, SR * 0.45)), 2)
    out = np.zeros(n)
    out[:m] = w / (np.abs(w).max() + 1e-12)
    return out


def _knock(p, n, r):
    """the thumb pushes the whole instrument: a low, heavily damped thud of the box on its support"""
    m = min(n, dsp.n_of(0.4))
    t = _t(m)
    f = p['knock_f'] * (1 + 0.05 * r.standard_normal())
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / p['knock_tau']) * (1 - np.exp(-t / 0.004))
    y += 0.5 * np.sin(2 * np.pi * f * 2.3 * t) * np.exp(-t / (p['knock_tau'] * 0.5)) * (1 - np.exp(-t / 0.002))
    out = np.zeros(n)
    out[:m] = y / (np.abs(y).max() + 1e-12)
    return out


def _room_tone(n, r, rumble):
    w = dsp.noise(n / SR, r, 'pink')[:n]
    lo = dsp.butter(dsp.noise(n / SR, r, 'brown')[:n], 'low', 120, 2)
    lo /= np.sqrt(np.mean(lo ** 2)) + 1e-12
    return (1 - rumble) * w + rumble * 2.0 * lo


def render(p, r):
    f0 = p.get('f0') or float(r.choice(REF_F0))
    n = dsp.n_of(LEN)
    pre = dsp.n_of(0.004)
    tine = _tine(p, f0, n - pre, r)
    pk = np.abs(tine).max() + 1e-12
    # the bridge drives the box; the tine's own radiation is weak at low notes, the box carries the body
    body = signal.fftconvolve(tine, _box_ir(p, r))[: len(tine)]
    body *= pk / (np.abs(body).max() + 1e-12)
    x = (1 - 0.5 * p['box_lvl']) * tine + p['box_lvl'] * body
    x += _symp(p, f0, len(x), r) * pk
    x += p['skin_lvl'] * pk * _skin(p, len(x), r)
    x += p.get('knock_lvl', 0.0) * pk * _knock(p, len(x), r)
    if p.get('zing_lvl', 0.0) > 0.003:
        x += p['zing_lvl'] * pk * _zing(p, f0, len(x), r)
    x = np.concatenate([np.zeros(pre), x])
    x = dsp.stereo(x, r, width=p['width'], pan=0.0)
    x = dsp.room(x, r, rt60=p['rt60'], size=0.3, damp=0.6, wet=p['room_wet'])
    x = dsp.mic(x, r, low_cut=20.0, air_db=p['air_db'], floor_db=-140.0)
    x = x[:n]
    pk = np.abs(x).max()
    for c in range(x.shape[1]):
        x[:, c] += _room_tone(len(x), r, p.get('rumble', 0.0)) * pk * 10 ** (p['floor_db'] / 20)
    x = dsp.trim_tail(x, p['floor_db'] - 2, 0.05)
    return dsp.fade(x, 0.0005, min(0.4, len(x) / SR * 0.3))
