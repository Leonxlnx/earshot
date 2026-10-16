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
    'mode': ['reveal', 'chirp'],            # the two idioms
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
    'length': (0.5, 1.8),
}

_REVEAL = {'mode': 'reveal', 'root': 880.0, 'chord': 'add9', 'material': 'bar', 'bloom_decay': 2.2, 'bright': 0.6,
           'open_ms': 120.0, 'strike': 0.5, 'shimmer': 0.25, 'hit': 0.3, 'sub': 0.25, 'swell': 0.0,
           'swell_level': 0.2, 'verb': 0.35, 'verb_rt': 2.6, 'width': 0.6, 'low_cut': 25.0}
_CHIRP = {'mode': 'chirp', 'chirp_f0': 1500.0, 'chirp_f1': 150.0, 'chirp_ms': 85.0, 'chirp_harm': 0.7,
          'hum_f': 76.0, 'hum_sink': 0.6, 'hum_formant': 500.0, 'hum_buzz': 0.9, 'hum_level': 0.8, 'hum_fast': 0.045,
          'hum_slow': 0.2, 'hum_mix': 0.06, 'length': 0.85, 'sub': 0.0, 'shimmer': 0.0, 'verb': 0.05,
          'verb_rt': 0.8, 'width': 0.0, 'root': 880.0, 'material': 'bar', 'bright': 0.5}
FAMILIES = {
    'logo_sting': [   # round 3: bright rings over a light, high-passed hit (the category's refs: nothing < 300 Hz,
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
    'pro_logo_sting': [
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
DEFAULTS = dict(FAMILIES['logo_sting'][0][1])
TARGETS = {'logo_sting': TARGET, 'pro_logo_sting': 'a designed logo sting: a falling chirp landing on a low hum'}
STUDY_STING = {'pro_logo_sting': {'len50': (0.5, 1.5), 't20': (30, 110), 't40': (150, 450), 'tilt': (-9.0, -5.0),
                                  'corr': (0.9, 1.0), 'ridge0': (900, 2400), 'ridge1': (55, 110)}}
_SEARCH = {
    'logo_sting': ['root', 'bloom_decay', 'bright', 'open_ms', 'strike', 'shimmer', 'hit', 'verb', 'verb_rt',
                   'width', 'low_cut'],
    'pro_logo_sting': ['chirp_f0', 'chirp_f1', 'chirp_ms', 'chirp_harm', 'hum_f', 'hum_sink', 'hum_formant',
                       'hum_buzz', 'hum_level', 'hum_fast', 'hum_slow', 'hum_mix', 'length'],
}
_LIMITS = {'pro_logo_sting': {'chirp_f0': (900, 3000, 'log'), 'chirp_f1': (100, 280, 'log'),
                              'chirp_ms': (50, 130, 'log'), 'hum_f': (55, 105, 'log'), 'length': (0.6, 1.6)}}


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


def _render_chirp(p, r):
    n = dsp.n_of(p['length'])
    ch = _chirp(p, n, r)
    t0 = p['chirp_ms'] / 1000 * 0.92                       # lands as the chirp reaches its bottom
    hum = _hum(p, n, r, t0)
    y = ch + p['hum_level'] * hum
    if p.get('sub', 0) > 0.01:
        sd = {**_subdrop.FAMILIES['pro_sub_drop'][0][1], 'f0': p['hum_f'], 'f1': p['hum_f'] * 0.66, 'bloom': 0.06,
              'click': 0.0, 'length': p['length'], 'decay_db': 45.0}
        s = _subdrop.render(sd, r).mean(axis=1)
        s = np.concatenate([np.zeros(dsp.n_of(t0)), s])[:n]
        y[:len(s)] += p['sub'] * s
    if p.get('shimmer', 0) > 0.01:                         # a soft bell partial on the landing
        b = dsp.bell(p['root'], max(0.3, p['length'] - t0), r, 'bar', decay=0.35, bright=0.3, strike=0.0)
        b = np.concatenate([np.zeros(dsp.n_of(t0)), b])[:n]
        y[:len(b)] += 0.6 * p['shimmer'] * b
    if p.get('verb', 0) > 0.01:
        ir = hall_ir(r, p['verb_rt'], 0.6, 0.5, 0.008)
        y = y + p['verb'] * signal.fftconvolve(y, ir)[:n]
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


def fx_granular(src, seconds, r):
    from ..fx import granular
    return granular(src, seconds, r, grain=0.05, density=50.0, pos=(0.0, 0.6), pos_jitter=0.05, pitch=12.0,
                    pitch_jitter=0.15, spread=0.9)


def render(p, r):
    chirp = p.get('mode', DEFAULTS['mode']) == 'chirp'
    p = {**(_CHIRP if chirp else _REVEAL), **p}
    y = _render_chirp(p, r) if chirp else _render_reveal(p, r)
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
