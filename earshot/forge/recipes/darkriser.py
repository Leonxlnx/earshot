"""A short dark riser: a build of noise and sub that speeds up through a stutter gate into a cut, with no rising
synth tone; every sample is computed.

What the reference clips do (a private licensed set, 'riser', 14 events in 8 films;
the clips are local study material and are not shipped): risers are SHORT and dark: 0.74 s (0.42-1.77), the loudest
point 248 ms (112-433) after the first moment within 20 dB of it, centroid 458 Hz (374-590), 17 % (10-57 %) of the
energy under 150 Hz, near mono (L/R correlation 0.94); they end 23 ms before the cut (median) and hand over to a hit,
falling 20 dB in 91 ms (79-128). The effects-led film's designed build is 1.3 s and sub-heavy (86 % under 150 Hz) with
crackle and a speeding stutter gate on top. Every earshot 0.3 riser was 6-12x brighter (2.7-5.8 kHz), and the cheap
riser is a synth tone sweeping up, which this recipe never draws: the build comes from level, a noise band that
opens, a sub that swells at one pitch, and a gate that accelerates.
Layers:
  body     pink noise through a low-Q band-pass whose centre opens body_f0 -> body_f1 with the build (Q body_q <= 2: a
           band of noise, never a tone)
  sub      a sine at sub_f (constant pitch) with a little tanh drive, swelling with the level (sub_db re the body)
  crackle  sparse 3-8 ms grains high-passed at crackle_hp, their density rising to crackle_rate / s, independent per
           channel (crackle_db re the body)
  gate     from gate_at (share of the length) a raised-cosine gate whose rate accelerates gate_f0 -> gate_f1 into the
           peak, gate_depth deep, duty `duty`; it opens fully over the last 30 ms so the riser peaks on its end
  level    in dB: -start_db * (1 - t / length)^curve up to the peak at `length`, then a fall of time constant tail_tau
           (the hand-over), an end fade
  space    width decorrelates the body and crackle above 200 Hz, a short room
"""
import types

import numpy as np

from .. import dsp

SR = dsp.SR
TARGET = 'a short dark riser building into a hit'
ALTS = ['a jet engine', 'a vacuum cleaner', 'heavy rain on a roof', 'a helicopter']
REFSET = 'riser'

SPACE = {
    'length': (0.3, 1.8),              # s to the peak
    'start_db': (12.0, 48.0),          # the first sample's level below the peak
    'curve': (0.5, 3.5, 'log'),        # > 1: slow, then accelerating into the peak
    'tail_tau': (0.01, 0.12, 'log'),   # fall after the peak (time constant, s)
    'body_f0': (60.0, 600.0, 'log'),   # body band centre at the start
    'body_f1': (150.0, 3000.0, 'log'),  # and at the peak
    'body_q': (0.5, 2.0),
    'sub_f': (32.0, 90.0, 'log'),
    'sub_db': (-40.0, 12.0),
    'sub_drive': (0.0, 2.0),
    'crackle_db': (-50.0, -6.0),
    'crackle_rate': (20.0, 600.0, 'log'),
    'crackle_hp': (1500.0, 9000.0, 'log'),
    'gate_at': (0.2, 0.95),            # where the stutter starts (share of the length)
    'gate_f0': (3.0, 16.0, 'log'),     # its rate there (Hz)
    'gate_f1': (10.0, 45.0, 'log'),    # and at the peak
    'gate_depth': (0.0, 1.0),
    'duty': (0.25, 0.85),
    'width': (0.0, 1.0),
    'room': (0.0, 0.4),
}
FAMILIES = {
    # earshot 0.4 phase B: hand-designed corners, then tuned toward each style on the measured features (the music
    # films' riser profile for dark and snatch; the study's dev-split designed build for build, with its length and
    # stutter depth held, and its catalog glide left out: that +97 st is a ridge jump from the sub to the crackle, not
    # a sweep), every take audited
    'riser': [
        # the music films' short dark riser: 0.74 s, ~430 Hz, a fifth of the energy in the sub, a deep late gate
        ('dark', {'length': 0.74, 'start_db': 41.86, 'curve': 1.378, 'tail_tau': 0.02767, 'body_f0': 148.4,
                  'body_f1': 502.3, 'body_q': 1.136, 'sub_f': 47.85, 'sub_db': -10.93, 'sub_drive': 0.7114,
                  'crackle_db': -50.0, 'crackle_rate': 134.9, 'crackle_hp': 3039.0, 'gate_at': 0.7039,
                  'gate_f0': 7.817, 'gate_f1': 19.91, 'gate_depth': 0.8088, 'duty': 0.3958, 'width': 0.2022,
                  'room': 0.2256}),
        # the effects-led film's designed build: 1.2 s, a driven 34 Hz sub under a low body band, loud high crackle
        # and a stutter gate speeding up from a fifth of the way in
        ('build', {'length': 1.2, 'start_db': 44.04, 'curve': 0.8445, 'tail_tau': 0.07787, 'body_f0': 94.27,
                   'body_f1': 150.0, 'body_q': 2.0, 'sub_f': 33.58, 'sub_db': 3.405, 'sub_drive': 1.116,
                   'crackle_db': -6.0, 'crackle_rate': 112.1, 'crackle_hp': 9000.0, 'gate_at': 0.2,
                   'gate_f0': 6.657, 'gate_f1': 12.49, 'gate_depth': 0.75, 'duty': 0.5742, 'width': 0.0,
                   'room': 0.1399}),
        # a half-second snatch: a steep late build (24 dB, mostly in its last 0.15 s) under a light fast stutter
        ('snatch', {'length': 0.45, 'start_db': 24.0, 'curve': 3.108, 'tail_tau': 0.02393, 'body_f0': 279.2,
                    'body_f1': 373.3, 'body_q': 2.0, 'sub_f': 37.65, 'sub_db': -13.57, 'sub_drive': 1.628,
                    'crackle_db': -22.68, 'crackle_rate': 20.44, 'crackle_hp': 4463.0, 'gate_at': 0.337,
                    'gate_f0': 7.732, 'gate_f1': 45.0, 'gate_depth': 0.2926, 'duty': 0.3132, 'width': 0.0431,
                    'room': 0.2851}),
    ],
}
DEFAULTS = dict(FAMILIES['riser'][0][1])
TARGETS = {'riser': TARGET}


def _peak1(x):
    return x / (np.abs(x).max() + 1e-12)


def _pink(n, r):
    F = np.fft.rfft(r.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 15.0)
    w = np.fft.irfft(F / np.sqrt(f), n)
    return w / (w.std() + 1e-12)


def level_db(p, t):
    """the build in dB re the peak (module doc)"""
    T = p['length']
    u = np.clip(t / T, 0.0, 1.0)
    L = -p['start_db'] * (1.0 - u) ** p['curve']
    late = t > T
    L[late] = -8.686 * (t[late] - T) / p['tail_tau']
    return L


def gate(p, t):
    """the accelerating stutter gate (1 = open): rate gate_f0 -> gate_f1 from gate_at to the peak, open over the last
    30 ms and after the peak"""
    T = p['length']
    t0 = p['gate_at'] * T
    v = np.clip((t - t0) / max(T - t0, 1e-3), 0.0, 1.0)
    rate = p['gate_f0'] * (p['gate_f1'] / p['gate_f0']) ** v
    ph = np.cumsum(np.where(t >= t0, rate, 0.0)) / SR
    frac = (ph + 0.5 * p['duty']) % 1.0                    # starts on an open peak: no step where it begins
    d = p['duty']
    on = np.where(frac < d, 0.5 - 0.5 * np.cos(2 * np.pi * np.minimum(frac / d, 1.0)), 0.0)
    g = 1.0 - p['gate_depth'] * (1.0 - on)
    g = np.where(t < t0, 1.0, g)
    k = np.clip((t - (T - 0.03)) / 0.03, 0.0, 1.0)            # opens fully into the peak
    g = g * (1 - k) + k
    return np.where(t > T, 1.0, g)


def render(p, r):
    p = {**DEFAULTS, **p}
    T = p['length']
    n = dsp.n_of(T + p['tail_tau'] * 60.0 / 8.686 + 0.02)
    t = np.arange(n) / SR
    a = 10 ** (np.maximum(level_db(p, t), -120.0) / 20)
    g = gate(p, t)
    m = np.clip(t / T, 0.0, 1.0)
    fc = p['body_f0'] * (p['body_f1'] / p['body_f0']) ** (m ** 1.5)
    bodies = []
    for _ in range(2):
        b = dsp.svf(_pink(n, r), fc, p['body_q'], 'bandn')
        bodies.append(b / (np.sqrt(np.mean(b ** 2)) + 1e-12))
    ph = 2 * np.pi * p['sub_f'] * t + r.uniform(0, 2 * np.pi)
    gd = 1.0 + 2.0 * p['sub_drive']
    sub = np.tanh(gd * np.sin(ph)) / np.tanh(gd) / 0.7
    crk = []
    for _ in range(2):
        c = np.zeros(n)
        tg = 0.0
        while True:
            rate = p['crackle_rate'] * (0.15 + 0.85 * min(1.0, tg / T) ** 1.5)
            tg += float(r.exponential(1.0 / rate))
            if tg >= T:
                break
            k = dsp.n_of(r.uniform(0.003, 0.008))          # 3-8 ms Hann grains: crackle, not sample clicks
            i = dsp.n_of(tg)
            kk = min(k, n - i)
            grain = r.standard_normal(k) * np.hanning(k)          # (noise, then level: the draw order is the take)
            c[i:i + kk] += grain[:kk] * 10 ** (12 * np.log10(r.uniform(0.05, 1.0)) / 20)
        c = dsp.butter(c, 'high', p['crackle_hp'], 2)
        crk.append(c / (np.sqrt(np.mean(c ** 2)) + 1e-12) * 0.25)
    w = p['width']
    kw = np.sqrt(max(0.0, 1 - w * w))
    hiR = dsp.butter(bodies[1], 'high', 200.0, 2)
    bR = (bodies[0] - dsp.butter(bodies[0], 'high', 200.0, 2)) + kw * dsp.butter(bodies[0], 'high', 200.0, 2) + w * hiR
    out = []
    for b, c in ((bodies[0], crk[0]), (bR, kw * crk[0] + w * crk[1])):
        x = b + 10 ** (p['sub_db'] / 20) * sub + 10 ** (p['crackle_db'] / 20) * c / 0.25
        out.append(x * a * g)
    y = np.stack(out, axis=1)
    if p['room'] > 0.01:
        y = dsp.room(y, r, rt60=0.35, size=0.4, damp=0.6, wet=p['room'])[:n]
    else:
        dsp.room_ir(r, 0.35, 0.4, 0.6)                    # keep the draw order
    y = dsp.butter(y, 'high', 25.0, 2)
    y = dsp.butter(y, 'low', 18000.0, 4)
    y = dsp.fade(y, 0.002, min(0.02, 0.5 * p['tail_tau'] * 60.0 / 8.686))
    return y / (np.abs(y).max() + 1e-12) * 0.89


def recipe_for(cat='riser', preset=0):
    fam = FAMILIES[cat]
    i = preset if isinstance(preset, int) else [nm for nm, _ in fam].index(preset)
    return types.SimpleNamespace(__name__=__name__, SPACE=dict(SPACE), DEFAULTS=dict(fam[i][1]), render=render,
                                 TARGET=TARGETS[cat], ALTS=ALTS, REFSET=REFSET, PRESET=fam[i][0])
