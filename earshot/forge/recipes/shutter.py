"""A camera shutter: a two-stage mechanical click (release, then the curtain / mirror landing), written in code; every
sample is computed, no recorded audio anywhere.

    from earshot.forge import dsp
    from earshot.forge.recipes import shutter
    x = shutter.render(shutter.FAMILY['shutter'][0], dsp.rng(1))     # 48 kHz stereo, one shutter click

What the films' shutter is (earshot 0.4 catalog, 6 shutters of 5 films standing >= 8 dB clear of the music; 21 events
in 9 films, on snap zooms and whip-cuts, often stacked with a click): 0.19 s (0.16-0.38), 39 ms (16-170) from the
first sound to the peak (the two stages: the louder one is often the second), 20 dB down 87 ms (20-123) after it,
centroid 1032 Hz (624-2187), 7 % under 150 Hz, 3 % over 4 kHz, tonal share 0.4 (0.3-0.52): a darker, longer click
than any 0.3 bank toggle (toggle/off, fit/snap/switch_flip decay 4-7x faster and sit 2-10x brighter).

The mechanism, part by part (a focal-plane shutter, a mirror box):
  * stage A, the release (`a_db` re stage B): a small latch letting go: a bright contact (band-passed noise around
    `a_hz`, `a_ms`) over the latch's few modes (`a_modes`: 3 stiff steel modes from a_hz)
  * the travel (`whirr` dB, `whirr_hz`): the curtain / mirror moving for the `gap` s between the two stages
  * stage B, the landing (at `gap`): the curtain or mirror slapping its stop: modes of the box from `b_hz` (a plastic /
    metal body, `b_decay`, darker than A), its contact noise (`b_noise`), the camera body's low thump (`body`,
    `body_f`), and a short spring rattle after it (`rattle` dB)
  * the finish (recipes/_tonal.finish): hp, lp, a small room (`room`, `rt`), a little `width`
CHARACTER keys: gap, a_db, b_hz; FAMILY['shutter'] = the palette (FAMILY_NAMES): dslr, snap, mirror, film, quick.
"""
import numpy as np

from .. import dsp
from . import _tonal as T

SR = dsp.SR
TARGET = 'a camera shutter click'
ALTS = ['a door latch', 'a stapler', 'a mouse click', 'a light switch', 'a gun being cocked']
STUDY = {'dur_s': (0.16, 0.38, 0.08), 'attack_ms': (16.0, 170.0, 20.0), 't20_ms': (20.0, 123.0, 30.0),
         'centroid_hz': (624.0, 2187.0, 300.0), 'lo150': (0.01, 0.24, 0.1), 'tonal': (0.3, 0.52, 0.1),
         'corr': (0.96, 0.99, 0.1)}

SPACE = {
    'gap': (0.015, 0.14, 'log'),       # release -> landing
    'a_db': (-24.0, 6.0),              # stage A re stage B
    'a_hz': (1500.0, 9000.0, 'log'),
    'a_ms': (0.2, 3.0, 'log'),
    'a_modes': (0.0, 1.0),             # the latch's modes, re its contact noise
    'whirr': (-60.0, -6.0),            # the travel between the stages (-60 = none)
    'whirr_hz': (600.0, 5000.0, 'log'),
    'b_hz': (300.0, 3000.0, 'log'),    # stage B: the box's lowest mode
    'b_decay': (0.004, 0.08, 'log'),
    'b_noise': (-40.0, 0.0),           # stage B's contact noise, dB re its modes
    'b_noise_hz': (600.0, 6000.0, 'log'),
    'b_contact': (0.0002, 0.003, 'log'),
    'body': (-60.0, 0.0),              # the camera body's low thump
    'body_f': (70.0, 300.0, 'log'),
    'body_ms': (4.0, 50.0, 'log'),
    'rattle': (-60.0, -6.0),           # a short spring rattle after B
    'rattle_hz': (2000.0, 7000.0, 'log'),
    'hp': (60.0, 500.0, 'log'),
    'lp': (3000.0, 16000.0, 'log'),
    'room': (0.0, 0.3),
    'rt': (0.05, 0.4, 'log'),
    'width': (0.0, 0.5),
}
DEFAULTS = {'gap': 0.045, 'a_db': -6.0, 'a_hz': 3500.0, 'a_ms': 0.8, 'a_modes': 0.5, 'whirr': -30.0,
            'whirr_hz': 1800.0, 'b_hz': 900.0, 'b_decay': 0.025, 'b_noise': -10.0, 'b_noise_hz': 2000.0,
            'b_contact': 0.0008, 'body': -16.0, 'body_f': 150.0, 'body_ms': 14.0, 'rattle': -30.0,
            'rattle_hz': 3500.0, 'hp': 90.0, 'lp': 9000.0, 'room': 0.1, 'rt': 0.14, 'width': 0.06}
FINE = ['a_db', 'a_hz', 'a_ms', 'a_modes', 'whirr', 'whirr_hz', 'b_decay', 'b_noise', 'b_noise_hz', 'b_contact',
        'body', 'body_f', 'body_ms', 'rattle', 'rattle_hz', 'hp', 'lp', 'room', 'rt', 'width']

B_MODES = [(1.0, 1.0, 1.0), (1.62, 0.7, 0.7), (2.31, 0.5, 0.5), (3.17, 0.35, 0.35), (4.4, 0.2, 0.25)]
A_MODES = [(1.0, 1.0), (1.71, 0.6), (2.63, 0.35)]


def _fam(**kw):
    return {**DEFAULTS, **kw}


FAMILY = {'shutter': [
    _fam(gap=0.0391, a_db=-23.5, a_hz=5537.0, a_ms=0.5005, a_modes=0.3966, whirr=-9.218, whirr_hz=1085.0, b_hz=1433.0,
         b_decay=0.006, b_noise=-1.243, b_noise_hz=1355.0, b_contact=0.00221, body=-20.61, body_f=238.2, body_ms=19.41,
         rattle=-26.17, rattle_hz=5575.0, hp=92.91, lp=5091.0, room=0.2557, rt=0.2552, width=0.1069),   # a DSLR
    _fam(gap=0.05821, a_db=-22.88, a_hz=3300.0, a_ms=2.964, a_modes=0.1957, whirr=-15.59, whirr_hz=5000.0, b_hz=2037.0,
         b_decay=0.004, b_noise=-2.783, b_noise_hz=2530.0, b_contact=0.00031, body=-10.33, body_f=156.8, body_ms=5.495,
         rattle=-48.36, rattle_hz=2581.0, hp=173.8, lp=9437.0, room=0.01662, rt=0.0526, width=0.234),  # a dry snap
    _fam(gap=0.06462, a_db=-18.51, a_hz=8113.0, a_ms=0.5578, a_modes=0.1115, whirr=-6.0, whirr_hz=688.7, b_hz=1511.0,
         b_decay=0.00622, b_noise=0.0, b_noise_hz=1468.0, b_contact=0.003, body=-33.33, body_f=300.0, body_ms=22.0,
         rattle=-26.59, rattle_hz=7000.0, hp=128.8, lp=6009.0, room=0.3, rt=0.374, width=0.09233),  # a mirror box
    _fam(gap=0.095, a_db=-12.29, a_hz=4200.0, a_ms=0.9, whirr=-14.0, whirr_hz=900.0, b_hz=720.0, b_decay=0.01103,
         b_noise=-7.79, b_noise_hz=600.0, b_contact=0.001728, body=-12.0, body_f=112.7, body_ms=26.32, rattle=-24.0,
         rattle_hz=4200.0, hp=84.56, lp=6500.0, room=0.3, rt=0.2, width=0.08),          # a heavier film camera
    _fam(gap=0.021, a_db=-4.0, a_hz=3800.0, a_ms=0.2281, a_modes=0.4098, whirr=-40.0, b_hz=1250.0, b_decay=0.008,
         b_noise=-6.0, b_noise_hz=2200.0, b_contact=0.001477, body=-17.09, body_f=199.6, body_ms=8.0, rattle=-50.0,
         hp=150.0, room=0.09189, rt=0.2138, width=0.1),                                # a quick double click
]}
FAMILY_NAMES = {'shutter': ['dslr', 'snap', 'mirror', 'film', 'quick']}
# How they were found (the earshot 0.4 study; local, not shipped): random + local searches on the reference shutter
# profile and the one dev clean clip, the audit run on the best. 'dslr' sits inside all 7 profile IQRs (d 0.35): a faint
# latch, 39 ms of travel, the curtain landing on a 1.4 kHz box with its contact noise, in a small room; 'snap' a dry
# 2 kHz snap-zoom shutter (fair on its dev clip); 'mirror' a slower mirror box (65 ms travel). Added for the 0.4 bank (a
# model-free search from a hand-set corner, every candidate's takes audited): 'film', a 720 Hz box after 95 ms of
# travel with the body's thump and a spring rattle; 'quick', release and curtain 21 ms apart, mid-bright and dry.


def family_space(params):
    return T.family_space(SPACE, {**DEFAULTS, **params}, FINE)


def render(p, r):
    p = {**DEFAULTS, **p}
    gap = p['gap'] * np.exp(0.03 * r.standard_normal())    # each take: the mechanism a little off its last run
    p = {**p, 'b_hz': p['b_hz'] * np.exp(0.03 * r.standard_normal()),
         'a_hz': p['a_hz'] * np.exp(0.03 * r.standard_normal())}
    tail = 7.0 * max(p['b_decay'], p['body_ms'] / 1000 if p['body'] > -59 else 0.0)
    n = dsp.n_of(min(1.0, gap + tail + 0.03))
    y = np.zeros(n)
    # stage A: the release (a bright latch)
    ka = dsp.n_of(0.03)
    a = T.burst(ka, r, p['a_hz'] / 1.8, min(p['a_hz'] * 1.8, 18000.0), p['a_ms'] / 1000, rise=0.0002)
    if p['a_modes'] > 0.005:
        am = T.modes(ka, r, [p['a_hz'] * q for q, _ in A_MODES], [g for _, g in A_MODES],
                     [0.004 / q for q, _ in A_MODES], jitter=0.01)
        a += p['a_modes'] * am / (np.abs(am).max() + 1e-12)
    a /= np.abs(a).max() + 1e-12
    # stage B: the landing (the box's modes, struck)
    kb = n - dsp.n_of(gap)
    bm = T.modes(kb, r, [p['b_hz'] * q for q, _, _ in B_MODES], [g for _, g, _ in B_MODES],
                 [max(0.0008, p['b_decay'] * d) for _, _, d in B_MODES], jitter=0.01)
    b = T.strike(bm, r, p['b_contact'], 0.6)[:kb]
    b /= np.abs(b).max() + 1e-12
    if p['b_noise'] > -39:
        hz = p['b_noise_hz']
        b += 10 ** (p['b_noise'] / 20) * T.burst(kb, r, hz / 2.0, hz * 2.0, 0.001, rise=0.0002)
    if p['body'] > -59:
        b += 10 ** (p['body'] / 20) * T.thump(kb, p['body_f'], p['body_ms'] / 1000, rise=0.0008)
    if p['rattle'] > -59:
        rr = T.burst(kb, r, p['rattle_hz'] / 1.6, p['rattle_hz'] * 1.6, 0.006, rise=0.002)
        s = min(kb, dsp.n_of(0.004))                        # 4 ms after the slap
        b += 10 ** (p['rattle'] / 20) * np.concatenate([np.zeros(s), rr[:kb - s]])
    y[:ka] += 10 ** (p['a_db'] / 20) * a
    y[n - kb:] += b
    if p['whirr'] > -59:                                    # the curtain travelling between the stages
        kw = dsp.n_of(gap)
        w = dsp.butter(r.standard_normal(kw), 'band', (p['whirr_hz'] / 1.7, p['whirr_hz'] * 1.7), order=2)
        w *= np.sin(np.pi * np.linspace(0, 1, kw)) ** 0.7
        y[:kw] += 10 ** (p['whirr'] / 20) * w / (np.abs(w).max() + 1e-12)
    return T.finish(y, r, hp=p['hp'], lp=p['lp'], room_wet=p['room'], rt60=p['rt'], width=p['width'], cut=600.0)
