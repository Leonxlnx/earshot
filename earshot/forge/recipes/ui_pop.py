"""A soft, rounded mid-register UI pop (the films' most used effect), written in code; every sample is computed.

    from earshot.forge import dsp
    from earshot.forge.recipes import ui_pop
    x = ui_pop.render(ui_pop.FAMILY['ui_pop'][0], dsp.rng(1))      # 48 kHz stereo, the films' typical pop
    xs = [ui_pop.render(p, dsp.rng(i)) for i, p in enumerate(ui_pop.FAMILY['ui_pop'])]   # the palette

What the films' pop is (earshot 0.4 catalog of a private set of licensed reference clips, 147 pops of 13 music-led
films standing >= 8 dB clear of the music): 0.16 s long (IQR 0.10-0.24), a SOFT onset of 33 ms (18-62) to the peak,
20 dB down 81 ms (57-129) after it, centroid 615 Hz (475-830) with almost nothing under 150 Hz (5 %) or over 4 kHz,
a tonal share of 0.55 (0.48-0.63: a pitched body with air in it, not a pure beep), pitch flat or falling (0 .. -8
st), mono (corr 0.97); they come in runs ~114 ms apart (4.7 per 10 s, 253 events in 20 films). Every earshot 0.3
bank pop is 3-5x harder at the onset, 2x faster to decay and purer (tonal 0.82-0.93): the pop / pro_pop recipes model
a bubble or a designed chirp, not this.

The model, part by part:
  * body: a sine on a pitch track that starts `bend` semitones above `f0` and settles on it in `ptau` (a membrane or
    a mouth's cavity relaxing; bend 0 = flat; or with `sweep` > 0 a log-linear fall over `sweep` s, eased by `ease`:
    the effects-led film's designed chirps), soft 2nd / 3rd partials (`h2`, `h3`) that die `hdec` times faster
    (rounded, never square), an optional inharmonic second mode (`m2` at `m2r` x f0, dying `m2dec` x faster: a hollow
    tube or a wooden body under the pop)
  * envelope: a soft rise over `onset` (a raised cosine, `curve` > 1 starts slower), `hold`, then a real exponential
    decay (`tau`: 20 dB in 2.3 tau; the file ends where it is 60 dB down, never a fade over a ring)
  * air: noise through a band-pass that follows the pitch (`breath` dB re the body, `bq`, its envelope ** `bdec`:
    < 1 lasts longer than the tone), which is what keeps the tonal share at the films' ~0.55; a short broadband
    breath at the release (`puff`, `puff_hz`, `puff_ms`)
  * a faint low `thump` (`thump_f`: the films' ~5 % under 150 Hz), an optional lip `click` (`click_hz`)
  * an optional second, smaller pop (`dbl` dB, `dgap` s later, `dratio` x the pitch)
  * the finish (recipes/_tonal.finish): high-pass `hp`, low-pass `lp`, a tiny room (`room`, `rt`), a little `width`
  * `jit`: each take (seed) moves the pitch, decay and onset a little, so a run of pops never clones one sound
CHARACTER keys (what a palette member is): f0, bend, sweep, m2 / m2r, dbl; the forge tunes the FINE keys per member
(family_space()). FAMILY['ui_pop'] = the palette (FAMILY_NAMES): 'soft' = the music films' typical pop (DEFAULTS; all
7 profile features inside the films' IQRs), 'step' the same a step higher falling in, 'boop' a rounder boop falling
9 st, 'blip' the effects-led film's bright ~890 Hz blip with fully decorrelated channels (width 1.34: L / R corr ~0.1),
'fall' an eased 7 st fall, 'wide' a wide 620 Hz pop with a lip click. How they were found
(random + local search on the films' numbers and the dev clips, then earshot forge) is in the earshot 0.4 study
notes (local, not shipped).
"""
import numpy as np

from .. import dsp
from . import _tonal as T

SR = dsp.SR
TARGET = 'a soft rounded pop, a small bubble popping in a user interface'
ALTS = ['a mouse click', 'a water drop', 'a wooden knock', 'a synth beep', 'a finger snap']
STUDY = {'dur_s': (0.10, 0.24, 0.08), 'attack_ms': (18.0, 62.0, 15.0), 't20_ms': (57.0, 129.0, 30.0),
         'centroid_hz': (475.0, 830.0, 200.0), 'lo150': (0.0, 0.14, 0.1), 'tonal': (0.48, 0.63, 0.1),
         'corr': (0.93, 1.0, 0.1)}      # the reference IQRs: (lo, hi, scale) for _tonal.outside

SPACE = {
    'f0': (250.0, 1200.0, 'log'),      # the body's landing pitch
    'bend': (-3.0, 72.0),              # semitones the pitch starts above f0 (> 0 falls into it; 40-72: designed chirp)
    'ptau': (0.004, 0.08, 'log'),      # how fast it settles (sweep 0) ...
    'sweep': (0.0, 0.15),              # ... or a log-linear fall over this long (s; the designed chirp of the effects
    'ease': (0.4, 2.5),                # film: 19 kHz -> 274 Hz in ~0.1 s), eased (> 1 falls fast first)
    'onset': (0.004, 0.07, 'log'),     # the soft rise to the peak
    'curve': (0.6, 2.5),               # rise shape (> 1 starts slower)
    'hold': (0.0, 0.03),
    'tau': (0.004, 0.09, 'log'),       # decay time constant (20 dB in 2.3 tau)
    'dcurve': (1.0, 2.5),              # decay shape: > 1 = damped as it rings (falls faster late: shorter t40)
    'h2': (0.0, 0.8),                  # 2nd / 3rd partials and how much faster they die
    'h3': (0.0, 0.5),
    'hdec': (1.0, 3.0),
    'inh': (0.0, 0.08),                # how sharp the 2nd / 3rd sit (x (1 + inh), x (1 + 2 inh)): a body, no oscillator
    'm2': (0.0, 0.8),                  # an inharmonic second mode (tube / wood)
    'm2r': (1.4, 4.5),
    'm2dec': (1.0, 4.0),
    'breath': (-60.0, -2.0),           # pitch-following air, dB re the body (-60 = none)
    'bq': (1.0, 8.0, 'log'),
    'bdec': (0.4, 2.0),
    'puff': (-60.0, -6.0),             # a broadband breath at the release
    'puff_hz': (600.0, 5000.0, 'log'),
    'puff_ms': (0.5, 8.0, 'log'),
    'thump': (-60.0, -6.0),            # a faint low thump (the < 150 Hz share)
    'thump_f': (70.0, 200.0, 'log'),
    'click': (-60.0, -10.0),           # a lip click on the onset
    'click_hz': (1500.0, 8000.0, 'log'),
    'dbl': (-60.0, 0.0),               # a second pop (-60 = none)
    'dgap': (0.03, 0.12),
    'dratio': (0.6, 1.6),
    'hp': (60.0, 300.0, 'log'),
    'lp': (2500.0, 12000.0, 'log'),
    'room': (0.0, 0.35),
    'rt': (0.06, 0.35, 'log'),
    'width': (0.0, 1.5),               # a decorrelated side above 400 Hz (1.43: L / R correlation ~0)
    'jit': (0.0, 1.0),                 # take-to-take variation (pitch +-2 %, decay +-8 %, onset +-10 % at 1)
}
# the films' typical pop (random + local search on the catalog profile, then the forge): a round
# 370 Hz body with rich soft partials (2nd, 3rd and a near-octave mode at 1.72x: the tonal share lands at ~0.5), a
# 24 ms soft rise held 15 ms, a damped decay (dcurve 2.1), low-passed at 3.5 kHz, a faint 130 Hz thump, and air that
# outlasts the tone a little (breath -14 dB, bdec 0.6: without it the tail is three bare partials, which the audit's
# comb check reads as a metallic ring at the pitch period, z 30-80; with it z 4-7)
DEFAULTS = {'f0': 370.0, 'bend': 0.7, 'ptau': 0.0075, 'sweep': 0.0, 'ease': 1.0, 'onset': 0.0236, 'curve': 2.45,
            'hold': 0.0147, 'tau': 0.041, 'dcurve': 2.12, 'h2': 0.57, 'h3': 0.35, 'hdec': 1.0, 'inh': 0.0, 'm2': 0.66,
            'm2r': 1.72,
            'm2dec': 1.43, 'breath': -14.0, 'bq': 3.6, 'bdec': 0.6, 'puff': -40.0, 'puff_hz': 4270.0, 'puff_ms': 2.7,
            'thump': -15.7, 'thump_f': 129.0, 'click': -60.0, 'click_hz': 3500.0, 'dbl': -60.0, 'dgap': 0.07,
            'dratio': 0.8, 'hp': 112.0, 'lp': 3530.0, 'room': 0.15, 'rt': 0.1, 'width': 0.1, 'jit': 0.5}
FINE = ['ptau', 'ease', 'onset', 'curve', 'hold', 'tau', 'dcurve', 'h2', 'h3', 'hdec', 'inh', 'm2dec', 'breath', 'bq',
        'bdec', 'puff', 'puff_hz', 'puff_ms', 'thump', 'thump_f', 'click', 'click_hz', 'hp', 'lp', 'room', 'rt',
        'width']


def _fam(**kw):
    return {**DEFAULTS, **kw}


FAMILY = {'ui_pop': [
    _fam(),                                                                                   # the films' typical pop
    _fam(f0=466.16, bend=3.0, ptau=0.03),                                                     # a step up, falling in
    _fam(f0=357.1, bend=9.0, ptau=0.03, onset=0.025, curve=2.0, hold=0.0062, tau=0.026, dcurve=1.4, h2=0.61,
         h3=0.17, hdec=1.63, m2=0.79, m2r=3.69, m2dec=2.68, breath=-13.8, bq=7.0, bdec=0.4, puff=-51.0, puff_hz=4600.0,
         puff_ms=1.9, thump=-8.0, thump_f=76.0, click=-54.0, hp=95.0, lp=12000.0, room=0.115, rt=0.093,
         width=0.2),                                                                          # a boop falling 9 st
    _fam(f0=889.1, bend=-3.0, ptau=0.08, ease=1.576, onset=0.01359, curve=2.5, hold=0.00311, tau=0.01052, dcurve=2.007,
         h2=0.3, h3=0.09244, hdec=1.107, inh=0.01188, m2=0.2913, m2r=4.115, m2dec=1.032, breath=-21.12, bq=2.26,
         bdec=1.076, puff=-46.31, puff_hz=4264.0, puff_ms=0.5636, thump=-36.45, thump_f=108.6, click=-45.55,
         click_hz=4013.0, dbl=-55.33, dgap=0.05156, dratio=0.7141, hp=60.0, lp=12000.0, room=0.0, rt=0.2241,
         width=1.341, jit=0.6717),                                                # a bright, fully wide blip
    _fam(f0=345.76, bend=7.36, ptau=0.0058, sweep=0.0894, ease=1.123, onset=0.0177, curve=1.658, hold=0.0006,
         tau=0.0225, dcurve=2.138, h2=0.671, h3=0.49, hdec=1.424, m2=0.53, m2r=4.5, m2dec=1.0, breath=-8.81, bq=2.61,
         bdec=1.89, puff=-47.5, puff_hz=5000.0, puff_ms=6.13, thump=-26.1, thump_f=81.0, click=-33.5, click_hz=4083.0,
         dbl=-29.6, dgap=0.113, dratio=1.155, hp=79.7, lp=11076.0, room=0.0, rt=0.076, width=0.64,
         jit=0.83),                                    # earshot forge's pick: an eased fall (dry: its room rang)
    _fam(f0=622.5, bend=-0.9783, ptau=0.08, sweep=0.00291, ease=1.822, onset=0.04115, curve=2.5, hold=0.00244,
         tau=0.03267, dcurve=1.844, h2=0.5426, h3=0.1129, hdec=1.054, inh=0.02561, m2=0.2683, m2r=4.5, m2dec=1.0,
         breath=-9.013, bq=3.179, bdec=0.5391, puff=-13.68, puff_hz=5000.0, puff_ms=1.019, thump=-46.85, thump_f=107.4,
         click=-10.1, click_hz=6070.0, dbl=-47.17, dgap=0.07519, dratio=0.6551, hp=67.89, lp=11620.0, room=0.00401,
         rt=0.1363, width=0.8105, jit=0.3896),                                    # a wide mid pop with a lip click
]}
FAMILY_NAMES = {'ui_pop': ['soft', 'step', 'boop', 'blip', 'fall', 'wide']}


def family_space(params):
    """the search space of one palette member: FINE keys free, the character keys pinned"""
    return T.family_space(SPACE, {**DEFAULTS, **params}, FINE)


def _one(p, r, n, f0, tau, gain=1.0):
    """one pop of length n (mono, starts at 0): body + air, before the finish"""
    t = dsp.t_of(n)
    e = T.env(t, p['onset'], tau, p['hold'], p['curve'], p['dcurve'])
    if p['sweep'] > 0.001:                                  # a designed chirp: a log-linear fall, eased, then settled
        u = np.clip(t / p['sweep'], 0, 1)
        f = f0 * 2 ** (p['bend'] / 12 * (1 - u) ** p['ease'])
    else:
        f = T.glide(t, f0, p['bend'], p['ptau'])
    s = p['inh']                                            # a cavity's overtones sit a little sharp of harmonic
    parts = [(2.0 * (1 + s), p['h2'], p['hdec']), (3.0 * (1 + 2 * s), p['h3'], p['hdec'] * 1.5)]
    if p['m2'] > 0:
        parts.append((p['m2r'], p['m2'], p['m2dec']))
    y = T.tone(f, r, parts, e)
    if p['breath'] > -59:
        y = y + 10 ** (p['breath'] / 20) * T.follow_noise(n, r, f, p['bq'], e ** p['bdec'])
    return gain * y


def render(p, r):
    p = {**DEFAULTS, **p}
    j = p['jit'] * r.standard_normal(3)                     # each take: a slightly different pop (a run never clones)
    p = {**p, 'f0': p['f0'] * np.exp(0.02 * j[0]), 'tau': p['tau'] * np.exp(0.08 * j[1]),
         'onset': p['onset'] * np.exp(0.1 * j[2])}
    dbl = p['dbl'] > -59
    gap = p['dgap'] if dbl else 0.0
    dur = p['onset'] + p['hold'] + 7.0 * p['tau'] + gap + 0.01
    n = dsp.n_of(min(0.8, dur))
    y = _one(p, r, n, p['f0'], p['tau'])
    if dbl:
        k = dsp.n_of(gap)
        y[k:] += _one(p, r, n - k, p['f0'] * p['dratio'], p['tau'] * 0.9, 10 ** (p['dbl'] / 20))
    pk = np.abs(y).max() + 1e-12
    if p['puff'] > -59:                                     # a breath at the release, riding the soft rise
        y += pk * 10 ** (p['puff'] / 20) * T.burst(n, r, p['puff_hz'] / 2.2, p['puff_hz'] * 2.2, p['puff_ms'] / 1000,
                                                   rise=max(0.0006, 0.3 * p['onset']))
    if p['thump'] > -59:
        y += pk * 10 ** (p['thump'] / 20) * T.thump(n, p['thump_f'], max(0.012, 0.6 * p['tau']),
                                                    rise=max(0.002, 0.5 * p['onset']))
    if p['click'] > -59:
        y += pk * 10 ** (p['click'] / 20) * T.burst(n, r, p['click_hz'] / 1.6, p['click_hz'] * 1.6, 0.0008)
    return T.finish(y, r, hp=p['hp'], lp=p['lp'], room_wet=p['room'], rt60=p['rt'], width=p['width'], cut=400.0)
