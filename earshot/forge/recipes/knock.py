"""A mid-band wood / plastic knock (a knuckle or a stick on a block, a shell, a desk), written in code; every sample is
computed, no recorded audio anywhere.

    from earshot.forge import dsp
    from earshot.forge.recipes import knock
    x = knock.render(knock.FAMILY['knock'][0], dsp.rng(1))       # 48 kHz stereo, the films' wooden knock

What the films' knock is (earshot 0.4 catalog, 15 knocks of 7 films standing >= 8 dB clear of the music, 28 events in
15 films: the widest spread of any foley touch): 0.18 s (0.11-0.26), 19 ms (10-28) from the first sound to the peak,
20 dB down 34 ms (12-131) after it, centroid 661 Hz (422-1327), 10 % under 150 Hz (2-57 %), almost nothing over 4 kHz,
tonal share 0.47 (0.42-0.55), near mono (0.95); kinds: a wooden block (~660 Hz), a plastic click (2.2-2.5 kHz, the
collage hops), a tonal wood block (1.2 kHz, 30 ms ring); they lead the cut by ~40 ms. The 0.3 bank has only low knocks
(tap / pro_tap 130-180 Hz, 4-5x too dark) and a 5.9 kHz bamboo tok (9x too bright).

The model (a struck body, modal synthesis):
  * the body: a material's mode set (`mat`, MATS: ratios, amplitudes x ratio ** `tilt`, decay multipliers) on the
    lowest mode `f1`, each mode ringing with tau = `decay` x its multiplier x (f / f1) ** -`slope` (higher modes die
    first; a negative slope: a hollow shell whose upper modes carry on), tiny per-take detuning (`jitter`), and a
    strike that stiffens the body for a few ms (`bend`: the pitch starts that fraction high and settles in `bend_ms`;
    a falling 'tock')
  * the grain: a dense cloud of 28 weaker modes (`cloud` dB re the main modes, spread log-uniformly from 1.1 x f1 to
    `cloud_hi` x f1, dying `cloud_dec` x as fast as the body): real wood and plastic have many modes, and without them
    a few clean sines read as a struck bar (the films' knocks are noisy: tonal share 0.47, the 0.3 candidates 0.97)
  * the strike: a contact pulse of `contact` s (a knuckle 1-3 ms, a stick 0.3 ms: a longer contact excites the high
    modes less) with grit, plus the contact noise itself (`noise` dB, centred on `noise_hz`, dying in `noise_ms`)
  * a second, softer contact `flam` dB before it, `flam_ms` earlier (two knuckles, a 'knock-knock': the films'
    10-28 ms to the peak), and / or one after it (`bounce` dB, `bounce_ms` later: a stick or a plastic part bouncing
    back onto the body)
  * the low body under it (`body` dB, `body_f`, `body_ms`: a desk or a table, the < 150 Hz share)
  * the finish (recipes/_tonal.finish): hp, lp, a small room (`room`, `rt`), a little `width`
CHARACTER keys: mat, f1, bend, body; FAMILY['knock'] = the palette (FAMILY_NAMES): wood, plastic, snap, desk,
double (each described where it is defined).
"""
import numpy as np

from .. import dsp
from . import _tonal as T

SR = dsp.SR
TARGET = 'a knock on a wooden block'
ALTS = ['a door knock', 'a mouse click', 'a hand clap', 'a bubble pop', 'a metal clank']
STUDY = {'dur_s': (0.11, 0.26, 0.08), 'attack_ms': (10.0, 28.0, 10.0), 't20_ms': (12.0, 131.0, 30.0),
         'centroid_hz': (422.0, 1327.0, 300.0), 'lo150': (0.02, 0.57, 0.1), 'tonal': (0.42, 0.55, 0.1),
         'corr': (0.94, 0.98, 0.1)}

MATS = {   # (ratio, amplitude, decay multiplier): measured-shape sets, the lowest mode = f1
    'block': [(1.0, 1.0, 1.0), (1.48, 0.55, 0.6), (2.15, 0.45, 0.45), (3.4, 0.25, 0.3), (4.6, 0.12, 0.2)],  # slit block
    'shell': [(1.0, 0.7, 1.0), (1.31, 1.0, 0.8), (1.77, 0.8, 0.6), (2.43, 0.6, 0.5), (3.1, 0.4, 0.35),
              (4.2, 0.25, 0.25), (5.6, 0.15, 0.2)],                                                       # plastic case
    'tok': [(1.0, 1.0, 1.0), (2.7, 0.15, 0.3), (5.1, 0.05, 0.15)],                                       # temple block
    'desk': [(1.0, 1.0, 1.0), (1.9, 0.6, 0.5), (2.8, 0.5, 0.4), (4.4, 0.3, 0.25), (6.3, 0.15, 0.15)],     # a table top
}

SPACE = {
    'mat': list(MATS),
    'f1': (250.0, 2600.0, 'log'),      # the lowest mode
    'decay': (0.004, 0.12, 'log'),     # its time constant
    'slope': (-0.4, 1.4),              # higher modes die faster: tau ~ (f / f1) ** -slope
    'tilt': (-1.0, 1.0),               # mode amplitudes x ratio ** tilt (> 0: the upper modes lead)
    'cloud': (-60.0, 6.0),             # the dense mode cloud (grain), dB re the main modes (-60 = none)
    'cloud_hi': (2.0, 10.0, 'log'),    # its top, x f1
    'cloud_dec': (0.3, 2.0, 'log'),    # its decay re the body's
    'jitter': (0.0, 0.02),             # per-take mode detuning (fraction)
    'bend': (0.0, 4.0),                # pitch starts this fraction high (a stiff strike; 1-4: a falling 'tock')
    'bend_ms': (1.0, 80.0, 'log'),
    'contact': (0.0002, 0.004, 'log'), # contact pulse length (s): longer = softer, darker
    'grit': (0.0, 1.0),
    'noise': (-60.0, 6.0),             # the contact noise, dB re the body's peak
    'noise_hz': (300.0, 6000.0, 'log'),
    'noise_ms': (0.3, 20.0, 'log'),
    'flam': (-60.0, -2.0),             # a softer first contact before the main one (-60 = none)
    'flam_ms': (4.0, 70.0),
    'bounce': (-60.0, -2.0),           # a softer contact after the main one (-60 = none)
    'bounce_ms': (6.0, 60.0),
    'body': (-60.0, 6.0),              # the low body under it (desk / table)
    'body_f': (50.0, 260.0, 'log'),
    'body_ms': (4.0, 60.0, 'log'),
    'hp': (40.0, 400.0, 'log'),
    'lp': (2500.0, 16000.0, 'log'),
    'room': (0.0, 0.3),
    'rt': (0.05, 0.4, 'log'),
    'width': (0.0, 0.6),
    'jit': (0.0, 1.0),                 # take-to-take variation (pitch +-2 %, decay +-10 %, contact +-15 % at 1)
}
DEFAULTS = {'mat': 'block', 'f1': 620.0, 'decay': 0.03, 'slope': 0.6, 'tilt': 0.0, 'cloud': -60.0, 'cloud_hi': 4.0,
            'cloud_dec': 0.8, 'jitter': 0.006, 'bend': 0.05, 'bend_ms': 4.0,
            'contact': 0.0012, 'grit': 0.4, 'noise': -18.0, 'noise_hz': 1800.0, 'noise_ms': 1.5, 'flam': -60.0,
            'flam_ms': 14.0, 'bounce': -60.0, 'bounce_ms': 20.0, 'body': -20.0, 'body_f': 140.0, 'body_ms': 15.0,
            'hp': 70.0, 'lp': 8000.0, 'room': 0.08, 'rt': 0.12, 'width': 0.08, 'jit': 0.5}
FINE = ['decay', 'slope', 'tilt', 'cloud', 'cloud_hi', 'cloud_dec', 'jitter', 'bend_ms', 'contact', 'grit', 'noise',
        'noise_hz', 'noise_ms', 'flam', 'flam_ms', 'bounce', 'bounce_ms', 'body_f', 'body_ms', 'hp', 'lp', 'room', 'rt',
        'width']


def _fam(**kw):
    return {**DEFAULTS, **kw}


FAMILY = {'knock': [
    _fam(f1=400.0, decay=0.00898, slope=1.4, tilt=0.4048, cloud=-9.5, cloud_hi=6.712, cloud_dec=0.8552, jitter=0.01331,
         bend=2.919, bend_ms=3.488, contact=0.00076, grit=0.6466, noise=0.01577, noise_hz=300.0, noise_ms=0.3,
         flam=-42.5, flam_ms=12.92, bounce=-19.75, bounce_ms=44.74, body=-11.36, body_f=102.0, body_ms=4.521, hp=107.2,
         lp=11210.0, room=0.1648, rt=0.1519, width=0.2037, jit=0.1328),         # the films' wooden knock (profile)
    _fam(f1=589.7, decay=0.0042, slope=1.4, tilt=0.2389, cloud_hi=6.513, cloud_dec=0.5565, jitter=0.00101, bend=3.485,
         bend_ms=5.989, contact=0.004, grit=0.9598, noise=-29.01, noise_hz=304.7, noise_ms=9.585, flam=-26.35,
         flam_ms=48.96, bounce=-10.06, bounce_ms=18.4, body=-48.31, body_f=177.7, body_ms=38.28, hp=40.0, lp=3815.0,
         room=0.03, rt=0.08539, width=0.2528, jit=0.2616),                       # a plastic click that bounces back
    _fam(f1=1055.0, decay=0.00446, slope=0.9923, tilt=0.4321, cloud=-53.72, cloud_hi=10.0, cloud_dec=0.4418,
         jitter=0.01216, bend=4.0, bend_ms=1.017, contact=0.004, grit=1.0, noise=-21.06, noise_hz=1316.0,
         noise_ms=11.51, flam=-15.46, flam_ms=38.32, bounce=-11.12, bounce_ms=21.61, body=-51.53, body_f=180.6,
         body_ms=23.32, hp=94.95, lp=5813.0, room=0.03, rt=0.08, width=0.2403, jit=0.2051),   # a bright snap
    _fam(f1=699.2, decay=0.02025, slope=0.552, tilt=0.08284, cloud=-0.8436, cloud_hi=5.867, cloud_dec=0.9966,
         jitter=0.01775, bend=2.157, bend_ms=71.36, contact=0.00066, grit=0.894, noise=3.033, noise_hz=1279.0,
         noise_ms=14.94, flam=-20.16, flam_ms=12.09, bounce=-47.54, bounce_ms=16.56, body=4.088, body_f=95.0,
         body_ms=28.8, lp=4889.0, room=0.09192, rt=0.276, width=0.5584, jit=0.8611),     # a knuckle on a desk
    _fam(mat='tok', f1=537.0, decay=0.005, slope=-0.00403, tilt=0.5245, cloud=-50.47, cloud_hi=3.995, cloud_dec=1.394,
         jitter=0.01402, bend=1.762, bend_ms=9.208, contact=0.00244, grit=0.6739, noise=-27.15, noise_hz=2665.0,
         noise_ms=3.251, flam=-2.088, flam_ms=55.14, bounce=-14.68, bounce_ms=28.8, body=-15.67, body_f=179.5,
         body_ms=4.703, hp=169.2, lp=7249.0, room=0.00579, rt=0.2958, width=0.3936, jit=0.8699),  # knock-knock
]}
FAMILY_NAMES = {'knock': ['wood', 'plastic', 'snap', 'desk', 'double']}
# How they were found (the earshot 0.4 study; local, not shipped): random + local searches over
# SPACE scored on the films' knock profile and the dev clean clips (CLAP in fixed 1.0 s windows + the model-free feature
# distance). 'wood' sits inside 6-7 of the films' 7 IQRs on every take (profile d 0.34-0.42 over takes: scored over 4
# takes, since one take's contact pulse can null a mode and fake a rounder knock); 'plastic' and 'snap' are bright
# plastic hops (2.4-3 kHz, a contact 20 ms after the strike; one fair on each of two dev clips); 'desk' a low
# knuckle knock with the table under it, 'double' a knock-knock 55 ms apart (its decay cut from 7.9 to 5 ms: a
# longer pure tok tail read as a comb ring to the audit on 1-4 takes in 60).


def family_space(params):
    return T.family_space(SPACE, {**DEFAULTS, **params}, FINE)


def _hit(p, r, n):
    """one strike of the body (mono, the contact at t = 0)"""
    f1 = p['f1']
    fr, am, ta = [], [], []
    for q, a, dm in MATS[p['mat']]:
        f = f1 * q
        if f >= 0.42 * SR:
            continue
        fr.append(f)
        am.append(a * q ** p['tilt'])
        ta.append(max(0.0008, p['decay'] * dm * q ** -p['slope']))
    y = T.modes(n, r, fr, am, ta, jitter=p['jitter'], bend=p['bend'], bend_tau=p['bend_ms'] / 1000)
    if p['cloud'] > -59:                                    # the grain: many weak modes over the main ones
        rc = np.random.default_rng(r.integers(1 << 31))
        hi = min(f1 * p['cloud_hi'], 0.4 * SR)
        fc = np.exp(rc.uniform(np.log(1.1 * f1), np.log(max(hi, 1.2 * f1)), 28))
        tc = np.maximum(0.0008, p['decay'] * p['cloud_dec'] * (fc / f1) ** -p['slope'])
        ac = rc.uniform(0.3, 1.0, 28) * (fc / f1) ** (p['tilt'] - 0.5)
        c = T.modes(n, rc, fc, ac, tc, bend=p['bend'], bend_tau=p['bend_ms'] / 1000)
        y = y / (np.abs(y).max() + 1e-12) + 10 ** (p['cloud'] / 20) * c / (np.abs(c).max() + 1e-12)
    y = T.strike(y, r, p['contact'], p['grit'])[:n]
    y = y / (np.abs(y).max() + 1e-12)
    if p['noise'] > -59:
        hz = p['noise_hz']
        y += 10 ** (p['noise'] / 20) * T.burst(n, r, hz / 2.0, hz * 2.0, p['noise_ms'] / 1000, rise=0.0002)
    return y


def render(p, r):
    p = {**DEFAULTS, **p}
    j = p['jit'] * r.standard_normal(3)                     # every knock lands a little differently
    p = {**p, 'f1': p['f1'] * np.exp(0.02 * j[0]), 'decay': p['decay'] * np.exp(0.1 * j[1]),
         'contact': p['contact'] * np.exp(0.15 * j[2])}
    flam = p['flam'] > -59
    lead = p['flam_ms'] / 1000 if flam else 0.0
    ring = p['decay'] * max(dm * q ** -p['slope'] for q, _, dm in MATS[p['mat']])   # the longest mode (finish trims)
    if p['cloud'] > -59:
        ring = max(ring, p['decay'] * p['cloud_dec'] * max(1.1 ** -p['slope'], p['cloud_hi'] ** -p['slope']))
    bounce = p['bounce'] > -59
    dur = lead + (p['bounce_ms'] / 1000 if bounce else 0.0) + 7.0 * max(ring, p['body_ms'] / 1000 if p['body'] > -59
                                                                       else 0.0) + 0.02
    n = dsp.n_of(min(1.0, dur))
    y = np.zeros(n)
    k = dsp.n_of(lead) if flam else 0
    y[k:] += _hit(p, r, n - k)
    if flam:                                                # a softer first contact (another knuckle)
        y += 10 ** (p['flam'] / 20) * _hit({**p, 'contact': p['contact'] * 1.5}, r, n)
    if bounce:                                              # a softer contact after it (a bounce back)
        kb = k + dsp.n_of(p['bounce_ms'] / 1000)
        if kb < n - 8:
            y[kb:] += 10 ** (p['bounce'] / 20) * _hit({**p, 'contact': p['contact'] * 1.3}, r, n - kb)
    if p['body'] > -59:                                     # the desk / table under the knock
        tb = T.thump(n - k, p['body_f'] * (1 + 0.03 * r.standard_normal()), p['body_ms'] / 1000, rise=0.001)
        y[k:] += 10 ** (p['body'] / 20) * tb
    return T.finish(y, r, hp=p['hp'], lp=p['lp'], room_wet=p['room'], rt60=p['rt'], width=p['width'], cut=500.0)
