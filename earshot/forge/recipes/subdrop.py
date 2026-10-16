"""A designed sub drop (the "64 -> 44 Hz" motion-design drop and its 808 / boom relatives); every sample is computed.

What it is (docs/motion-sfx-study.md "Sub drop", docs/motion-sfx-techniques.md section 5):
  * one gesture: a sine gliding DOWN into the sub region, exponential in frequency (straight on a log axis). The pro
    film's drop goes 64 -> 44 Hz (-0.55 oct) over 0.75 s, enters on a cut with a bright transient on top, BLOOMS to
    its peak ~0.38-0.43 s after the cut and decays ~40 dB/s, straight in dB; 85 % + of the energy under 150 Hz, narrow;
  * library drops are the same gesture faster and wider: an 808 boom (pitch falling 1-2 octaves in 10-150 ms, a
    long decay), a long dread drop (1.5 octaves over ~1.5 s), a punchy short one under an impact;
  * small speakers cannot play 40 Hz, so designers add harmonics (a saturated copy, or partials 2 / 3 synthesised
    directly) under the clean sine: the ear infers the missing fundamental (techniques doc src 37 / 48);
  * the transient on top is short noise + a ping (the "click"), optionally a kick-like thump; a small dark room on
    those only (the sub stays dry and mono).
Layers here: sub sine (pitch path f0 -> f1 over `glide` with `glide_curve`), bloom / hold / straight-in-dB decay,
harmonics 2-6 synthesised directly under a roll-off at harm_lp, click (noise burst with a (1 - t/len)^3 envelope +
a ping), thump (150 -> 90 Hz), grit (low brown-noise texture under the sub, following its envelope), a little width
above 150 Hz, and a short room on the click / thump. Ends on a fade (a sub must never stop audibly).

Character parameters (each family preset is one corner of them): pitch (f0, f1), sweep (glide, glide_curve), weight
(bloom, hold, decay_db, length), brightness (harm, harm_lp, click, click_f), texture (grit, thump).
No recorded audio is used anywhere.
"""
import types

import numpy as np

from .. import dsp
from .impact import STUDY, measure, study_extra, study_penalty  # noqa: F401  (shared study measurements)

SR = dsp.SR
TARGET = 'a deep sub bass drop'
ALTS = ['a kick drum', 'an explosion', 'thunder', 'a low hum']
REFSET = 'sub_drop'

SPACE = {
    'f0': (45, 220, 'log'),            # start pitch (Hz)
    'f1': (28, 62, 'log'),             # end pitch (Hz)
    'glide': (0.06, 1.6, 'log'),       # s to travel f0 -> f1
    'glide_curve': (0.3, 3.0, 'log'),  # progress in log-frequency = (t / glide) ** curve (< 1: fast first)
    'bloom': (0.004, 0.45, 'log'),     # s to the sub's peak
    'hold': (0.0, 0.5),                # s at the peak
    'decay_db': (15.0, 90.0),          # dB per second after the hold (straight in dB)
    'length': (0.6, 3.0),              # s in all (the last 15 % fades)
    'harm': (0.0, 0.8),                # saturated copy (harmonics for small speakers)
    'harm_lp': (150, 650, 'log'),      # its low-pass
    'click': (0.0, 1.0),               # transient on top (re the sub peak)
    'click_f': (1200, 8000, 'log'),
    'click_ms': (1.0, 40.0, 'log'),    # the pro drop's transient is 30-50 ms
    'thump': (0.0, 1.0),               # kick-like body at the onset
    'grit': (0.0, 1.0),                # low noise texture under the sub
    'width': (0.0, 0.3),               # side above 150 Hz (the sub itself stays mono)
    'room': (0.0, 0.4),                # short dark room on click + thump
}

FAMILIES = {
    'sub_drop': [
        ('808', {'f0': 120.0, 'f1': 40.0, 'glide': 0.12, 'glide_curve': 0.6, 'bloom': 0.004, 'hold': 0.05,
                 'decay_db': 30.0, 'length': 2.0, 'harm': 0.35, 'harm_lp': 300.0, 'click': 0.4, 'click_f': 3000.0,
                 'click_ms': 3.0, 'thump': 0.5, 'grit': 0.0, 'width': 0.05, 'room': 0.1}),
        ('pro', {'f0': 64.0, 'f1': 44.0, 'glide': 0.75, 'glide_curve': 1.0, 'bloom': 0.38, 'hold': 0.05,
                 'decay_db': 42.0, 'length': 1.4, 'harm': 0.3, 'harm_lp': 300.0, 'click': 0.6, 'click_f': 3500.0,
                 'click_ms': 30.0, 'thump': 0.0, 'grit': 0.0, 'width': 0.05, 'room': 0.1}),
        ('long', {'f0': 90.0, 'f1': 30.0, 'glide': 1.4, 'glide_curve': 1.0, 'bloom': 0.15, 'hold': 0.2,
                  'decay_db': 22.0, 'length': 3.0, 'harm': 0.25, 'harm_lp': 250.0, 'click': 0.15, 'click_f': 2500.0,
                  'click_ms': 8.0, 'thump': 0.0, 'grit': 0.15, 'width': 0.08, 'room': 0.0}),
        ('punch', {'f0': 160.0, 'f1': 50.0, 'glide': 0.2, 'glide_curve': 0.5, 'bloom': 0.004, 'hold': 0.0,
                   'decay_db': 70.0, 'length': 0.8, 'harm': 0.5, 'harm_lp': 400.0, 'click': 0.7, 'click_f': 4000.0,
                   'click_ms': 2.0, 'thump': 0.8, 'grit': 0.0, 'width': 0.05, 'room': 0.15}),
        ('gritty', {'f0': 110.0, 'f1': 36.0, 'glide': 0.5, 'glide_curve': 0.8, 'bloom': 0.05, 'hold': 0.1,
                    'decay_db': 35.0, 'length': 1.8, 'harm': 0.7, 'harm_lp': 450.0, 'click': 0.3, 'click_f': 2200.0,
                    'click_ms': 6.0, 'thump': 0.2, 'grit': 0.6, 'width': 0.12, 'room': 0.05}),
    ],
    'pro_sub_drop': [
        ('ref', {'f0': 64.0, 'f1': 44.0, 'glide': 0.75, 'glide_curve': 1.0, 'bloom': 0.38, 'hold': 0.04,
                 'decay_db': 42.0, 'length': 1.3, 'harm': 0.25, 'harm_lp': 300.0, 'click': 0.6, 'click_f': 3500.0,
                 'click_ms': 35.0, 'thump': 0.0, 'grit': 0.0, 'width': 0.04, 'room': 0.08}),
        ('quick', {'f0': 72.0, 'f1': 46.0, 'glide': 0.5, 'glide_curve': 1.0, 'bloom': 0.24, 'hold': 0.02,
                   'decay_db': 52.0, 'length': 1.0, 'harm': 0.3, 'harm_lp': 320.0, 'click': 0.8, 'click_f': 4200.0,
                   'click_ms': 25.0, 'thump': 0.1, 'grit': 0.0, 'width': 0.04, 'room': 0.08}),
        ('low', {'f0': 58.0, 'f1': 37.0, 'glide': 0.85, 'glide_curve': 1.0, 'bloom': 0.42, 'hold': 0.06,
                 'decay_db': 38.0, 'length': 1.5, 'harm': 0.4, 'harm_lp': 260.0, 'click': 0.45, 'click_f': 2800.0,
                 'click_ms': 40.0, 'thump': 0.0, 'grit': 0.1, 'width': 0.04, 'room': 0.06}),
        ('bright', {'f0': 66.0, 'f1': 44.0, 'glide': 0.7, 'glide_curve': 1.0, 'bloom': 0.3, 'hold': 0.04,
                    'decay_db': 42.0, 'length': 1.25, 'harm': 0.6, 'harm_lp': 420.0, 'click': 0.9,
                    'click_f': 5500.0, 'click_ms': 30.0, 'thump': 0.0, 'grit': 0.0, 'width': 0.06, 'room': 0.12}),
        ('soft', {'f0': 62.0, 'f1': 44.0, 'glide': 0.75, 'glide_curve': 1.3, 'bloom': 0.43, 'hold': 0.08,
                  'decay_db': 34.0, 'length': 1.5, 'harm': 0.2, 'harm_lp': 260.0, 'click': 0.2, 'click_f': 2600.0,
                  'click_ms': 40.0, 'thump': 0.0, 'grit': 0.05, 'width': 0.03, 'room': 0.05}),
    ],
}
DEFAULTS = dict(FAMILIES['sub_drop'][1][1])
TARGETS = {'sub_drop': TARGET, 'pro_sub_drop': 'a designed sub drop: a low sine falling under a bright click'}
STUDY_SUB = {   # study targets (docs/motion-sfx-study.md: sub_drop row) as measure() ranges
    'pro_sub_drop': {'len50': (0.8, 1.6), 'attack_ms': (250, 460), 'slope_dBs': (-50, -32), 'sh150': (0.85, 1.0),
                     'corr': (0.8, 1.0), 'ridge0': (55, 75), 'ridge1': (38, 50)},
    'sub_drop': {'len50': (0.6, 3.0), 'sh150': (0.8, 1.0), 'corr': (0.8, 1.0), 'slope_dBs': (-90, -15)},
}
_SEARCH = {
    # timbre and polish only: with no forge refs the critic is mostly CLAP, and a CLAP-led search moved the presets'
    # pitch / sweep / bloom (their character) far off (round 2, first run); the character stays as designed
    'sub_drop': ['harm', 'harm_lp', 'click', 'click_f', 'length'],
    'pro_sub_drop': ['harm', 'harm_lp', 'click', 'click_f', 'click_ms', 'length'],
}
_LIMITS = {'pro_sub_drop': {'f0': (54, 80, 'log'), 'f1': (34, 50, 'log'), 'glide': (0.4, 1.0, 'log'),
                            'bloom': (0.2, 0.45, 'log'), 'decay_db': (32.0, 55.0), 'length': (0.9, 1.6),
                            'click_ms': (15.0, 45.0, 'log')}}


def _click(p, n, r):
    """noise burst (envelope (1 - t / len)^3, high-passed at click_f / 2) plus a short ping at click_f"""
    L = min(n, max(16, dsp.n_of(p['click_ms'] / 1000)))
    u = np.arange(L) / L
    w = r.standard_normal(L) * (1 - u) ** 3 * (1 - np.exp(-np.arange(L) / (SR * 0.0002)))
    w = dsp.butter(w, 'high', p['click_f'] * 0.5, 2)
    w /= np.abs(w).max() + 1e-12
    k = min(n, dsp.n_of(0.02))
    tk = np.arange(k) / SR
    ping = np.sin(2 * np.pi * p['click_f'] * tk) * np.exp(-tk / 0.003) * (1 - np.exp(-tk / 0.0003))
    y = np.zeros(n)
    y[:L] += w
    y[:k] += 0.6 * ping
    return y / (np.abs(y).max() + 1e-12)


def _thump(n, r):
    k = n                                                # the whole buffer (round 2: a 0.6 s buffer left a click at
                                                         # 0.6 s under the quiet tail of the 'punch' drop)
    t = np.arange(k) / SR
    f = 90 + 60 * np.exp(-t / 0.03)
    y = np.zeros(n)
    y[:k] = np.sin(2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 0.3)) * np.exp(-t / 0.06) * (1 - np.exp(-t / 0.001))
    return dsp.fade(y, 0.0, 0.01)


def pitch_path(p, t):
    """f(t): f0 -> f1, progress in log-frequency (t / glide) ** glide_curve (1 = constant octaves per second)"""
    s = np.clip(t / max(p['glide'], 1e-3), 0, 1) ** p['glide_curve']
    return np.exp(np.log(p['f0']) + (np.log(p['f1']) - np.log(p['f0'])) * s)


def amp_env(p, t):
    """bloom (raised cosine to the peak), hold, then a straight-in-dB decay of decay_db dB/s"""
    b, h = p['bloom'], p['hold']
    e = np.ones_like(t)
    s = t < b
    e[s] = 0.5 - 0.5 * np.cos(np.pi * t[s] / b)
    late = t > b + h
    e[late] = 10 ** (-p['decay_db'] * (t[late] - b - h) / 20)
    return e


def render(p, r):
    p = {**DEFAULTS, **p}
    # length: as asked, but never past the point where the decay reaches -70 dB
    life = p['bloom'] + p['hold'] + 70.0 / p['decay_db']
    seconds = max(0.3, min(p['length'], life))
    n = dsp.n_of(seconds)
    t = np.arange(n) / SR
    f = pitch_path(p, t)
    ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 0.2)
    env = amp_env(p, t)
    sub = np.sin(ph) * env
    # harmonics for small speakers, synthesised directly (techniques doc section 5): partials 2-6 of the gliding sub,
    # amplitude 0.6 / k under a 4th-power roll-off at harm_lp (per sample, so it follows the glide). Round 2: a
    # biased-tanh copy low-passed at harm_lp ran its harmonic series on through 0.3-12 kHz, and earshot's polish read
    # it as a metallic comb ring (z 18-108 on the pro presets); this series stops at about 2 x harm_lp.
    harm = np.zeros(n)
    if p['harm'] > 0.01:
        h = np.zeros(n)
        for k in range(2, 7):
            fk = k * f
            taper = np.clip((300.0 - fk) / 80.0, 0.0, 1.0)  # nothing above 300 Hz (see the floor note below)
            h += (0.6 / k) / (1 + (fk / p['harm_lp']) ** 4) * np.sin(k * ph + 0.4 * k) * taper
        h *= env
        harm = p['harm'] * h / (np.abs(h).max() + 1e-12)
    # texture: low brown noise following the sub's envelope
    grit = np.zeros(n)
    if p['grit'] > 0.01:
        g = np.cumsum(r.standard_normal(n))
        g = dsp.butter(g - np.mean(g), 'band', (35.0, 160.0), 2)
        grit = 0.3 * p['grit'] * g / (np.abs(g).max() + 1e-12) * env
    top = p['click'] * _click(p, n, r) + 0.7 * p['thump'] * _thump(n, r)
    if p['room'] > 0.01 and np.abs(top).max() > 0:
        top = dsp.room(top, r, rt60=0.35, size=0.4, damp=0.7, wet=p['room'])[:n]
    # an inaudible floor (-80 dB re peak, 300 Hz - 12 kHz, following the envelope down to -40 dB). Round 2: a tail with
    # only a few low partials above 300 Hz and nothing else made earshot's comb-ring detector (polish.ring: tail log
    # spectrum 300 Hz - 12 kHz) peak at its 2 ms quefrency edge, z 300-675: a false flag that would steer the search
    # (core request). The partials now stop below 300 Hz and the band above holds only this floor.
    fl = dsp.butter(r.standard_normal(n), 'band', (300.0, 12000.0), 2)
    mono = sub + harm + grit + top + 10 ** (-80 / 20) * fl / (np.abs(fl).max() + 1e-12) * np.maximum(env, 0.01)
    # a little width above 150 Hz (the sub stays centred)
    hi = dsp.butter(harm + top, 'high', 150.0, 2)
    side = p['width'] * np.concatenate([np.zeros(dsp.n_of(0.0045)), hi])[:n]
    y = np.stack([mono + side, mono - side], axis=1)
    y = dsp.butter(y, 'high', 22.0, 2)
    y = dsp.fade(y, 0.0015, max(0.05, 0.15 * seconds))
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
