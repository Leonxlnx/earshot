"""A water drop / drip (the 'bloop' of a drop into water), written in code from the physics of a bubble; every sample is
computed, no recorded audio anywhere.

    from earshot.forge import dsp
    from earshot.forge.recipes import drip
    x = drip.render(drip.FAMILY['drip'][0], dsp.rng(1))          # 48 kHz stereo, one drop

What the films' drip is (earshot 0.4 catalog, 29 drops of 9 music-led films standing >= 8 dB clear of the music; 51
events in 14 films, the very first sound of three of them): 0.19 s (0.11-0.33), 39 ms (16-66) from the first sound to
the peak, 20 dB down 129 ms (90-188) after it, centroid 596 Hz (379-1045), 10 % under 150 Hz, tonal share 0.55
(0.48-0.61), near mono (0.95); the catalog's glide (the ridge's last frames vs its first, 43 ms frames) reads 0, yet the
analysts hear the pitch RISE: it rises within the first tens of ms. No 0.3 bank member is a drip: the pops are 3-5x
shorter and harder, pro_pop/bloop rises slowly (+6 st over 55-75 ms: the audit's synth-whip tell).

The physics (Minnaert / Longuet-Higgins): a drop hits the surface (a short splash `click`, `click_hz`) and pinches off
a bubble whose volume breathes at f = 3.26 Hz m / R; as the cavity under the drop closes, the bubble's pitch RISES
(`rise` semitones, most of it within ~3 `rtau`) and its ring builds (`onset`) and decays (`tau`, damped by radiation
and heat: Q ~ 10-30). Modelled: a sine on that rising track (`f0` = the pitch at the pinch-off) with a weak 2nd harmonic
(`h2`, the breathing's nonlinearity, dying `hdec` x faster), the cavity's hiss following the pitch (`breath`, `bq`), a
smaller daughter bubble (`daughter` dB, `d_ratio` x higher, `d_delay` later), a soft low `thump` of the drop's impact,
then the finish (hp, lp, a small room `room` / `rt`, a little `width`).
The rise is the design: by default it completes within ~40 ms (rtau <= 12 ms), where the model-free audit's
rising-glide check (a climb of more than 4 st over at least 50 ms: earshot.forge.sfx) stays quiet. A slower bubble
(a big drop, a 'bloop') climbs 6-12 st over 50-100 ms and trips it, although nothing about it is a synth whip: the
audit allowance (allows(), check()) accepts a climb a bubble can make (at most RISE_MAX_ST semitones, starting within
the first RISE_START_S and done within RISE_MAX_S) and keeps flagging anything longer or larger. sfx.check applies it
to the drip category (earshot 0.4): 'bloop' is the one member whose climb (~5 st over 0.1-0.15 s) needs it, the others
rise within the audit's 50 ms window.
CHARACTER keys: f0, rise, daughter; FAMILY['drip'] = the palette (FAMILY_NAMES): drop, bubble, plink, bloop, tiny.
"""
import numpy as np

from .. import dsp
from . import _tonal as T

SR = dsp.SR
TARGET = 'a water drop falling into water, a drip'
ALTS = ['a bubble pop', 'a mouse click', 'a wooden knock', 'a synth beep', 'a finger snap']
STUDY = {'dur_s': (0.11, 0.33, 0.08), 'attack_ms': (16.0, 66.0, 15.0), 't20_ms': (90.0, 188.0, 30.0),
         'centroid_hz': (379.0, 1045.0, 200.0), 'lo150': (0.02, 0.25, 0.1), 'tonal': (0.48, 0.61, 0.1),
         'corr': (0.87, 0.99, 0.1)}
RISES = True       # the pitch rises by design (a drip): the audit's rising-glide check needs an allowance for it
RISE_MAX_ST = 14.0  # the audit allowance (module doc): the largest climb a bubble makes (SPACE rise + its jitter) ...
RISE_MAX_S = 0.25   # ... done within this long (rising_glide's from_s -> to_s: its exponential approach creeps) ...
RISE_START_S = 0.06  # ... starting this early (the climb is the pinch-off, at the onset, never a late sweep)

SPACE = {
    'f0': (250.0, 1600.0, 'log'),      # the bubble's pitch at pinch-off
    'rise': (0.0, 14.0),               # semitones it rises as the cavity closes
    'rtau': (0.003, 0.03, 'log'),      # how fast (most of the rise within ~3 rtau)
    'onset': (0.001, 0.06, 'log'),     # the ring builds up
    'curve': (0.6, 2.5),
    'tau': (0.015, 0.12, 'log'),       # its decay (20 dB in 2.3 tau)
    'h2': (0.0, 0.6),                  # the breathing's 2nd harmonic
    'hdec': (1.0, 3.0),
    'breath': (-60.0, -2.0),           # cavity hiss following the pitch, dB re the ring
    'bq': (1.0, 8.0, 'log'),
    'daughter': (-60.0, -3.0),         # a smaller bubble (-60 = none)
    'd_ratio': (1.3, 2.6),
    'd_delay': (0.004, 0.04),
    'click': (-60.0, -6.0),            # the drop hitting the surface
    'click_hz': (800.0, 8000.0, 'log'),
    'click_ms': (0.2, 3.0, 'log'),
    'thump': (-60.0, -6.0),            # the impact's low thump
    'thump_f': (70.0, 220.0, 'log'),
    'hp': (60.0, 300.0, 'log'),
    'lp': (2500.0, 14000.0, 'log'),
    'room': (0.0, 0.35),
    'rt': (0.06, 0.5, 'log'),
    'width': (0.0, 0.8),
    'jit': (0.0, 1.0),                 # take-to-take variation (pitch +-3 %, rise +-10 %, decay +-8 % at 1)
}
DEFAULTS = {'f0': 520.0, 'rise': 6.0, 'rtau': 0.008, 'onset': 0.02, 'curve': 1.0, 'tau': 0.05, 'h2': 0.25,
            'hdec': 1.6, 'breath': -10.0, 'bq': 3.0, 'daughter': -60.0, 'd_ratio': 1.8, 'd_delay': 0.012,
            'click': -24.0, 'click_hz': 2500.0, 'click_ms': 0.6, 'thump': -20.0, 'thump_f': 120.0, 'hp': 80.0,
            'lp': 6000.0, 'room': 0.12, 'rt': 0.18, 'width': 0.15, 'jit': 0.5}
FINE = ['rtau', 'onset', 'curve', 'tau', 'h2', 'hdec', 'breath', 'bq', 'd_ratio', 'd_delay', 'click', 'click_hz',
        'click_ms', 'thump', 'thump_f', 'hp', 'lp', 'room', 'rt', 'width']


def _fam(**kw):
    return {**DEFAULTS, **kw}


FAMILY = {'drip': [
    _fam(f0=306.4, rise=8.033, rtau=0.00719, onset=0.02135, curve=2.192, tau=0.0411, h2=0.09568, hdec=1.765,
         breath=-22.71, bq=6.109, daughter=-3.0, d_ratio=2.271, d_delay=0.03833, click=-40.83, click_hz=2300.0,
         click_ms=2.804, thump=-23.28, thump_f=164.8, hp=183.0, lp=10010.0, room=0.2097, rt=0.08963, width=0.164,
         jit=0.2181),                                                         # the films' drop (profile)
    _fam(f0=254.5, rise=9.767, rtau=0.00977, onset=0.01237, curve=1.123, tau=0.03743, h2=0.1299, hdec=1.755,
         breath=-26.28, bq=1.0, daughter=-3.0, d_ratio=1.976, d_delay=0.02129, click=-51.96, click_hz=7393.0,
         click_ms=0.4349, thump=-8.339, thump_f=155.3, hp=148.0, lp=5147.0, room=0.1926, rt=0.1608, width=0.297,
         jit=0.7479),                                                         # a lower bubble with its daughter
    _fam(f0=694.6, rise=3.332, rtau=0.01, onset=0.03436, curve=0.7727, tau=0.03, h2=0.1039, hdec=2.57, breath=-24.02,
         bq=1.697, daughter=-8.628, d_ratio=1.607, d_delay=0.04, click=-44.91, click_hz=1378.0, click_ms=0.5007,
         thump=-39.42, thump_f=121.8, hp=97.33, lp=4560.0, room=0.07537, rt=0.1075, width=0.1842,
         jit=0.225),                                                          # a small high plink
    _fam(f0=420.0, rise=10.0, rtau=0.025, daughter=-6.0, d_ratio=2.2, d_delay=0.03),   # a big drop's slow bloop
    _fam(f0=1150.0, rise=0.9682, rtau=0.004, onset=0.006, curve=1.296, tau=0.01727, h2=0.3661, breath=-2.0,
         click=-21.63, click_hz=4898.0, thump=-40.0, hp=200.0, lp=12230.0, room=0.0, rt=0.12, width=0.2436,
         jit=0.3),                                                            # a tiny fast droplet
]}
FAMILY_NAMES = {'drip': ['drop', 'bubble', 'plink', 'bloop', 'tiny']}
# How they were found (the earshot 0.4 study; local, not shipped): random + local searches on the reference drip
# profile (no dev clean clip exists: the one clean drip is held out), the audit run on the best. 'drop' d 0.36 (5 of 7
# inside the IQRs), 'bubble' 0.34 (6 of 7), 'plink' 0.58 (4 of 7: higher and purer than the typical drop, on purpose).
# The cavity's hiss (breath -22..-26 dB) keeps the tail from reading as a comb ring. Added for the 0.4 bank (a
# model-free search from a hand-set corner toward a designed aim, every candidate's takes audited): 'bloop', the slow
# bubble whose climb (~5 st over 0.1-0.15 s) only the drip allowance accepts, and 'tiny', a 1.15 kHz droplet of 0.1 s.


def family_space(params):
    return T.family_space(SPACE, {**DEFAULTS, **params}, FINE)


def _bubble(p, r, n, f0, tau, rise):
    t = dsp.t_of(n)
    e = T.env(t, p['onset'], tau, 0.0, p['curve'])
    f = T.rise_track(t, f0, rise, p['rtau'])
    y = T.tone(f, r, [(2.0, p['h2'], p['hdec'])], e)
    if p['breath'] > -59:
        y = y + 10 ** (p['breath'] / 20) * T.follow_noise(n, r, f, p['bq'], e)
    return y


def render(p, r):
    p = {**DEFAULTS, **p}
    j = p['jit'] * r.standard_normal(3)                     # every drop is a slightly different bubble
    p = {**p, 'f0': p['f0'] * np.exp(0.03 * j[0]), 'rise': p['rise'] * np.exp(0.1 * j[1]),
         'tau': p['tau'] * np.exp(0.08 * j[2])}
    dau = p['daughter'] > -59
    n = dsp.n_of(min(1.0, p['onset'] + 7.0 * p['tau'] + (p['d_delay'] if dau else 0.0) + 0.02))
    y = _bubble(p, r, n, p['f0'], p['tau'], p['rise'])
    if dau:                                                 # a smaller bubble pinched off a moment later
        k = dsp.n_of(p['d_delay'])
        y[k:] += 10 ** (p['daughter'] / 20) * _bubble(p, r, n - k, p['f0'] * p['d_ratio'], 0.6 * p['tau'],
                                                      0.6 * p['rise'])
    pk = np.abs(y).max() + 1e-12
    if p['click'] > -59:
        hz = p['click_hz']
        y += pk * 10 ** (p['click'] / 20) * T.burst(n, r, hz / 2.0, hz * 2.0, p['click_ms'] / 1000, rise=0.0002)
    if p['thump'] > -59:
        y += pk * 10 ** (p['thump'] / 20) * T.thump(n, p['thump_f'], 0.012, rise=0.002)
    return T.finish(y, r, hp=p['hp'], lp=p['lp'], room_wet=p['room'], rt60=p['rt'], width=p['width'], cut=400.0)


# ------------------------------------------------------------------------------------------- the audit allowance
def allows(glide):
    """True when a climb measured by earshot.forge.sfx.rising_glide is one a bubble makes (module doc): at most
    RISE_MAX_ST semitones, from within RISE_START_S of the start, done within RISE_MAX_S"""
    if not glide or glide.get('from_s') is None:
        return True
    return bool(glide['rise_st'] <= RISE_MAX_ST and glide['from_s'] <= RISE_START_S
                and glide['to_s'] - glide['from_s'] <= RISE_MAX_S)


def check(x, role=None, sound=None):
    """sfx.check for a drip: the audit with the category's allowance (sfx.check applies allows() to the drip
    category since earshot 0.4: a bubble's climb is a note, a climb no bubble makes stays a warning)"""
    from .. import sfx
    return sfx.check(x, category='drip', role=role, sound=sound)
