"""Shimmer / sparkle / chime textures for motion design, written in code (every sample computed, no recorded audio).

    from earshot.forge import dsp
    from earshot.forge.recipes import shimmer
    x = shimmer.render(shimmer.FAMILY['pro_shimmer'][0], dsp.rng(1))    # 48 kHz stereo, the film's granular sparkle

What the pro shimmers are (docs/motion-sfx-study.md, the two ref_2 trims looked at in spectrograms):
  * Granular sparkle (1.95 s): short partial grains every ~20 ms (50/s) scattered over 1-10 kHz, over ONE steady tone
    near 500 Hz; the level swells for ~0.5-1 s (straight in dB, from ~-45 dB) and fades ~0.8 s; tilt +6.5 dB/oct;
    WIDE (correlation 0.08, side as loud as mid), the grains move across the image.
  * Melodic chime run (1.95 s): glockenspiel-like notes (270 Hz - 10 kHz partials) panned note by note (hard-panned),
    cut off by a gate at the end.
Real shimmer recordings (the 'shimmer' refset: bar chimes, wind chimes, glock glissandi, dense metallic textures;
held out: a bar-chime sweep, a bell-tree stroke, a chromatic glock run) ring out naturally for 2-5 s.
Rules from docs/motion-sfx-techniques.md section 9: pitches on a scale (pentatonic / major) so the texture never
clashes with the music, a density envelope (never constant density), no fatiguing 8-12 kHz glare, width.

Styles (`style`):
  'sparkle'  a cloud of short grains (sine or small-bar partials) on a scale between g_lo and g_hi, density and
             level following a swell -> hold -> fade envelope, random pans, over a steady tone bed, plus an air wash
             and a wide plate
  'run'      a chime run: `notes` glock-bar notes on a scale going up / down / arc, accelerating or not, panned note
             by note, ringing (n_decay), gated (pro) or left to ring out (real)
  'chimes'   wind chimes / a bell tree: a set of tuned tubes struck at random, thinning out, each ringing for seconds
CHARACTER keys: style, length, swell, fade, rate, g_lo / g_hi, scale / key, tone_hz / tone_db, notes / direction,
n_decay, gate_end, ring. FAMILY[refset] = the palette members; FINE[style] = what a forge run tunes with the character
fixed.
Round 3 (forge v2, after the round-2 judge: real shimmer tails half the bench's, narrow steady lines, mono chimes):
`ring` = 1 lets runs / chimes ring out freely (no drawn envelope; only the file end fades), every partial is a beating
doublet (`beat`), strikes are detuned (`detune`) and swing (`swing`), each note has a bright strike burst (`sizzle`),
higher notes die faster (`f_decay`), the bus is decorrelated, the air bed follows the ringing notes, and a tone bed at
its 'off' level (-40 dB) is really off. FAMILY['shimmer'] holds the six round-3 ring-outs (t40 1.9-3.0 s).
Round 6 (re-render path only, no new sound): `r2_path` = 1 renders runs / chimes on the round-2 code (no detune draw
per note / strike, the 4 s note cap, a tone bed at -40 dB left on), so the pro_shimmer run members delivered in round 2
(pro_shimmer_04 / 05) re-render exactly from their report.json. Not in SPACE or DEFAULTS; absent = today's code.
"""
import numpy as np

from .. import dsp, fx
from .notify import _outside, measure

SR = dsp.SR
TARGET = 'a sparkling magical shimmer'
ALTS = ['a xylophone run', 'a piano trill', 'rain on a window', 'jingling keys', 'a synthesizer pad']
# (deliberately not the benchmark's shimmer confusions: eval/forge/bench.py CONFUSIONS stay independent)
REFSET = 'pro_shimmer'

SCALES = {'penta': [0, 2, 4, 7, 9], 'major': [0, 2, 4, 5, 7, 9, 11], 'lydian': [0, 2, 4, 6, 7, 9, 11]}
BAR = ([1.0, 2.756, 5.404], [1.0, 0.32, 0.1])
TUBE = ([1.0, 2.76, 5.40, 8.93], [0.5, 1.0, 0.6, 0.25])

SPACE = {
    'style': ['sparkle', 'run', 'chimes'],
    'length': (1.0, 3.0),
    'start_db': (-50.0, -20.0),       # envelope level at t = 0
    'swell': (0.2, 1.1),              # time to the peak
    'rise_curve': (0.6, 2.5),         # > 1: the swell accelerates into the peak
    'hold': (0.0, 0.3),
    'fade': (0.4, 1.4),               # peak -> -45 dB
    'gate_end': [0, 1],               # 1: a gate cuts the end (pro); 0: it rings / fades out
    # grains
    'rate': (20.0, 75.0),             # grains per second at the peak
    'dens_floor': (0.1, 0.7),         # density at the quiet ends, re the peak
    'g_lo': (350.0, 3500.0, 'log'),
    'g_hi': (4500.0, 11000.0, 'log'),
    'scale': ['penta', 'major', 'lydian'],
    'key': (0, 11, 'int'),            # scale root (0 = C)
    'g_tau': (0.0015, 0.06, 'log'),   # grain decay
    'g_noise': (0.0, 1.0),            # share of each grain that is a band of noise around its note (the film's
                                      # grains are short broadband-ish bursts: vertical streaks, not pings)
    'cluster': (1, 4, 'int'),         # notes per grain (struck together: a wider, brighter streak)
    'g_bell': (0.0, 1.0),             # 0 sine grains .. 1 small-bar partials
    'g_jit_db': (0.0, 9.0),           # grain level spread
    'tilt': (-3.0, 6.0),              # dB / octave on the grains (+: brighter)
    'spread': (0.3, 1.0),             # pan spread
    'decor': (0.0, 1.5),              # decorrelation of the grain bus (all-pass side signal)
    'octave': (0.0, 0.5),             # an octave-up echo of each grain (a cheap shimmer-reverb climb)
    # tone bed
    'tone_hz': (250.0, 800.0, 'log'), # snapped to the scale's root octave
    'tone_db': (-40.0, -6.0),
    'tone_h': (0.0, 0.5),
    'vib': (0.0, 0.012),
    'tone_w': (0.2, 1.2),             # tone decorrelation (the film's sparkle: side as loud as mid)
    # air wash
    'air_db': (-50.0, -16.0),
    'air_hp': (4500.0, 9000.0, 'log'),
    # space / finish
    'verb': (0.0, 0.7),
    'rt60': (0.5, 2.8, 'log'),
    'hp': (150.0, 1200.0, 'log'),
    'deess': (0.0, 9.0),              # cut at 9 kHz (no fatiguing glare)
    # run
    'notes': (6, 24, 'int'),
    'run_t': (0.4, 1.6),
    'direction': ['up', 'down', 'arc'],
    'accel': (0.6, 1.6),              # note times ~ (k / N)^accel: < 1 slows down, > 1 speeds up
    'n_decay': (0.1, 2.5, 'log'),
    'hard': (0.0, 1.0),               # mallet hardness
    # chimes
    'tubes': (4, 9, 'int'),
    'strikes': (4, 30, 'int'),
    # round 3 (real ring-outs: the judge found tails half the bench's, narrow steady lines, mono chimes)
    'ring': [0, 1],                   # 1: notes ring out freely (no shaped envelope), only the file's end fades
    'beat': (0.0, 3.0),               # doublet beating per partial (Hz): a struck bar / tube is two close modes
    'detune': (0.0, 12.0),            # random per-strike detune (cents): no two strikes make the same exact line
    'swing': (0.0, 8.0),              # slow pitch wobble (cents, 0.7-2 Hz): a swinging tube / a hand-held tree
    'sizzle': (-60.0, -8.0),          # bright strike noise 5-18 kHz per note (dB re the note): the metallic onset
    'f_decay': (0.0, 0.8),            # higher notes decay faster: tau ~ (f / 1 kHz)^-f_decay
    'tail_fade': (0.2, 1.2),          # ring mode: the end fade
}

DEFAULTS = {'style': 'sparkle', 'length': 1.95, 'start_db': -42.0, 'swell': 0.85, 'rise_curve': 1.2, 'hold': 0.1,
            'fade': 0.85, 'gate_end': 0, 'rate': 55.0, 'dens_floor': 0.5, 'g_lo': 450.0, 'g_hi': 9000.0,
            'scale': 'penta', 'key': 2, 'g_tau': 0.004, 'g_noise': 0.6, 'cluster': 2, 'g_bell': 0.3, 'g_jit_db': 6.0, 'tilt': 1.5,
            'spread': 0.95, 'decor': 1.2, 'octave': 0.15, 'tone_hz': 500.0, 'tone_db': -10.0, 'tone_h': 0.2, 'vib': 0.004, 'tone_w': 1.0,
            'air_db': -30.0, 'air_hp': 6500.0, 'verb': 0.3, 'rt60': 1.2, 'hp': 250.0, 'deess': 3.0, 'notes': 12,
            'run_t': 1.2, 'direction': 'up', 'accel': 1.0, 'n_decay': 0.5, 'hard': 0.5, 'tubes': 6, 'strikes': 14,
            'ring': 0, 'beat': 0.0, 'detune': 0.0, 'swing': 0.0, 'sizzle': -60.0, 'f_decay': 0.0, 'tail_fade': 0.3}


def _fam(**kw):
    return {**DEFAULTS, **kw}


FAMILY = {
    'pro_shimmer': [
        _fam(),                                                                         # the film's sparkle
        _fam(length=1.4, swell=0.35, fade=0.7, rate=65.0, g_lo=1200.0, tone_db=-18.0, g_tau=0.003, cluster=3,
             hold=0.05, air_db=-26.0),                                                  # a quick bright glint
        _fam(style='run', notes=16, run_t=1.4, direction='arc', gate_end=1, n_decay=0.09, hard=0.95, length=1.65,
             start_db=-18.0, swell=0.9, hold=0.3, fade=2.0, tone_db=-22.0, tone_hz=270.0, verb=0.2, rt60=0.6,
             g_lo=400.0, g_hi=6000.0, air_db=-34.0),                                                              # the film's chime run
        _fam(style='run', notes=18, run_t=0.75, direction='down', accel=0.8, gate_end=1, n_decay=0.08, hard=0.8,
             length=1.15, start_db=-4.0, swell=0.1, hold=0.2, fade=1.4, tone_db=-40.0, scale='major', key=7,
             g_lo=800.0, g_hi=7000.0, verb=0.25, rt60=0.9),                                       # a falling glitter run
        _fam(length=2.4, swell=1.0, fade=1.0, rate=34.0, g_hi=6000.0, g_tau=0.012, g_noise=0.15, cluster=1,
             tone_hz=400.0, tone_db=-12.0, verb=0.5, rt60=2.0, g_bell=0.7, scale='lydian', key=5),   # a warm wash
    ],
    'shimmer': [   # round 3: real ring-outs (the judge: tails half the bench's, narrow steady lines, mono chimes)
        _fam(style='run', ring=1, notes=22, run_t=0.9, direction='up', accel=0.9, n_decay=0.8, f_decay=0.3,
             length=4.2, tail_fade=0.7, tone_db=-40.0, scale='major', g_lo=1800.0, g_hi=9000.0, hard=0.65,
             g_bell=0.5, beat=1.2, detune=3.0, swing=1.0, sizzle=-3.0, verb=0.45, rt60=3.2, spread=0.95,
             air_db=-30.0, air_hp=3500.0, deess=2.0, decor=0.8),                                                                # bar-chime sweep
        _fam(style='run', ring=1, notes=14, run_t=0.8, direction='down', accel=0.8, n_decay=0.6, f_decay=0.25,
             length=4.0, tail_fade=0.7, tone_db=-40.0, g_lo=1500.0, g_hi=8000.0, g_bell=0.15, hard=0.55,
             beat=1.8, detune=5.0, swing=1.5, sizzle=-5.0, verb=0.5, rt60=3.2, spread=0.9, air_db=-32.0,
             air_hp=4500.0, decor=0.8),   # bell-tree stroke
        _fam(style='chimes', ring=1, tubes=6, strikes=12, n_decay=0.7, f_decay=0.2, length=4.2, tail_fade=0.9,
             tone_db=-40.0, g_lo=900.0, g_hi=3200.0, hard=0.75, beat=0.9, detune=4.0, swing=4.0, sizzle=-6.0,
             verb=0.42, rt60=3.0, spread=1.0, air_db=-30.0, air_hp=4000.0, tilt=3.0, decor=1.0),                   # wind chimes
        _fam(style='run', ring=1, notes=8, run_t=0.7, direction='up', accel=1.0, n_decay=0.55, f_decay=0.3,
             length=3.6, tail_fade=0.6, tone_db=-40.0, scale='major', key=0, g_lo=1000.0, g_hi=4200.0, hard=0.8,
             g_bell=0.6, beat=0.8, detune=2.0, swing=1.0, sizzle=-9.0, verb=0.42, rt60=2.9, spread=0.9,
             air_db=-32.0, air_hp=3000.0, decor=0.8),   # glock run
        _fam(length=4.2, swell=0.6, fade=3.4, rate=40.0, g_tau=0.06, g_noise=0.0, cluster=1, g_bell=1.0,
             verb=0.5, rt60=3.0, tone_db=-30.0, octave=0.15),                           # a ringing sparkle
        _fam(style='run', ring=1, notes=16, run_t=1.1, direction='arc', accel=1.0, n_decay=0.75, f_decay=0.35,
             length=4.4, tail_fade=0.8, tone_db=-40.0, scale='penta', key=9, g_lo=1300.0, g_hi=7000.0, hard=0.5,
             g_bell=0.8, beat=1.4, detune=6.0, swing=2.0, sizzle=-5.0, verb=0.48, rt60=3.2, spread=1.0,
             air_db=-30.0, air_hp=4000.0, decor=0.8),                                                             # a slow arc on a bell tree
    ],
}
FINE = {
    'sparkle': ['start_db', 'rise_curve', 'dens_floor', 'g_tau', 'g_noise', 'g_bell', 'g_jit_db', 'tilt', 'spread', 'decor', 'octave',
                'tone_db', 'tone_h', 'vib', 'tone_w', 'air_db', 'air_hp', 'verb', 'rt60', 'hp', 'deess'],
    'run': ['accel', 'hard', 'g_bell', 'g_jit_db', 'tilt', 'spread', 'octave', 'verb', 'rt60', 'hp', 'deess',
            'air_db', 'beat', 'detune', 'sizzle'],
    'chimes': ['hard', 'g_jit_db', 'tilt', 'verb', 'rt60', 'hp', 'deess', 'air_db', 'beat', 'detune', 'swing',
               'sizzle'],     # round 3: spread pinned (round 2's search narrowed the wind chimes to corr 0.97)
}


def family_space(params, style=None):
    """one family member's search space: its style's FINE keys free, every other key pinned (one-element list)"""
    free = set(FINE.get(style or params.get('style', 'sparkle'), []))
    return {k: (spec if k in free else [params.get(k, DEFAULTS.get(k))]) for k, spec in SPACE.items()}


# ------------------------------------------------------------------------------------------------ helpers
def scale_freqs(lo, hi, scale, key):
    """every note of the scale (rooted on `key` semitones above C) between lo and hi Hz"""
    out = []
    c0 = 16.3516
    for octv in range(0, 11):
        for s in SCALES[scale]:
            f = c0 * 2 ** (octv + (s + key) / 12)
            if lo <= f <= hi:
                out.append(f)
    return sorted(out) or [float(np.sqrt(lo * hi))]


def envelope(p, n):
    """level (linear) over the sound: start_db -> 0 dB at `swell` (rise_curve shapes it), hold, then a straight-in-dB
    fade to -45 dB over `fade`; with gate_end the level holds its fade until a 30 ms gate at the end"""
    t = dsp.t_of(n)
    up = np.clip(t / p['swell'], 0, 1)
    db = p['start_db'] * (1 - up ** (1 / max(p['rise_curve'], 1e-3)))
    down = np.clip((t - p['swell'] - p['hold']) / p['fade'], 0, None)
    db = np.where(t > p['swell'] + p['hold'], -45.0 * down, db)
    return 10 ** (np.maximum(db, -90) / 20)


def _pan_gains(pos):
    th = (np.clip(pos, -1, 1) + 1) * np.pi / 4
    return np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)


def _note(f, n, tau, r, partials, hard, bell, beat=0.0, swing=0.0, sizzle=None):
    """one struck note (mono): partials with highs dying first, a short mallet; round 3: each partial a doublet that
    beats (`beat` Hz), a slow swing wobble (`swing` cents) and an optional bright strike noise (`sizzle` dB)"""
    t = dsp.t_of(n)
    ratios, amps = partials
    y = np.zeros(n)
    wob = None
    if swing > 0.05:
        wob = 2 ** (swing / 1200 * np.sin(2 * np.pi * r.uniform(0.7, 2.0) * t + r.uniform(0, 6.3)))
    for q, a in zip(ratios, amps):
        if q > 1 and bell <= 0:
            continue
        fq = f * q
        if fq > 18000.0:                     # round 2: partials near Nyquist read as aliasing ('fold')
            continue
        aa = a * (bell if q > 1 else 1.0) * q ** (1.2 * (hard - 0.5))
        env = np.exp(-t / max(tau * q ** -0.9, 0.002))
        ph0 = r.uniform(0, 6.3)
        if wob is None:
            z = np.sin(2 * np.pi * fq * t + ph0)
        else:
            z = np.sin(2 * np.pi * fq * np.cumsum(wob) / SR + ph0)
        if beat > 0.02:                      # the doublet starts in phase (loudest at the strike), then beats
            fb = fq + beat * r.uniform(0.5, 1.5) * np.sqrt(q)
            z = (z + 0.7 * np.sin(2 * np.pi * fb * t + ph0)) / 1.7
        y += aa * env * z
    k = min(n, dsp.n_of(0.0006))
    y[:k] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    if hard > 0.05:
        m = min(n, dsp.n_of(0.003))
        w = dsp.butter(r.standard_normal(m), 'high', min(2 * f, 12000), order=2) * np.exp(-dsp.t_of(m) / 0.0006)
        y[:m] += 0.25 * hard * w / (np.abs(w).max() + 1e-12)
    if sizzle is not None and sizzle > -59:
        m = min(n, dsp.n_of(0.08))
        w = dsp.butter(r.standard_normal(m), 'band', (max(5000.0, 1.5 * f), 18000.0), order=2)
        w *= np.exp(-dsp.t_of(m) / r.uniform(0.005, 0.016))
        w[:k] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))      # (the strike starts on zero: no edge click)
        y[:m] += 10 ** (sizzle / 20) * w / (np.sqrt(np.mean(w[:dsp.n_of(0.01)] ** 2)) + 1e-12) * 0.35
    return y


def _tilt_gain(f, tilt):
    return 10 ** (tilt * np.log2(f / 2000.0) / 20)


def _sparkle(p, r, n):
    env = envelope(p, n)
    out = np.zeros((n, 2))
    notes = scale_freqs(p['g_lo'], p['g_hi'], p['scale'], p['key'])
    # grain times: Poisson at the peak rate, thinned by the density envelope (density follows the swell)
    t, times = 0.0, []
    while t < n / SR:
        t += r.exponential(1.0 / p['rate'])
        e = env[min(n - 1, int(t * SR))]
        d = p['dens_floor'] + (1 - p['dens_floor']) * min(1.0, e)
        if r.uniform() < d and t < n / SR - 0.01:
            times.append(t)
    gl = dsp.n_of(min(0.35, 7 * p['g_tau']))
    for tk in times:
        i = int(tk * SR)
        m = min(gl, n - i)
        if m < 16:
            continue
        y = np.zeros(m)
        for c in range(int(p['cluster'])):
            f = notes[int(r.integers(0, len(notes)))]
            yc = (1 - p['g_noise']) * _note(f, m, p['g_tau'], r, BAR, 0.0, p['g_bell'])
            if p['g_noise'] > 0.01:
                w = dsp.butter(r.standard_normal(m), 'band', (f / 1.25, min(f * 1.25, 20000)), order=2)
                w *= np.exp(-dsp.t_of(m) / p['g_tau'])
                yc += p['g_noise'] * w / (np.abs(w).max() + 1e-12)
            y += yc * _tilt_gain(f, p['tilt'])
        g = env[i] * 10 ** (r.uniform(-1, 1) * p['g_jit_db'] / 20) / np.sqrt(p['cluster'])
        if p['octave'] > 0.01 and 2 * f < 0.42 * SR:      # (f = the cluster's last note)
            y2 = _note(2 * f, m, p['g_tau'] * 0.7, r, BAR, 0.0, 0.0)
            y = y + p['octave'] * np.concatenate([np.zeros(dsp.n_of(0.006)), y2])[:m]
        gL, gR = _pan_gains((1 if r.uniform() < 0.5 else -1) * r.uniform(0.35, 1.0) * p['spread'])   # wide
        out[i:i + m, 0] += g * gL * y
        out[i:i + m, 1] += g * gR * y
    out *= 0.25
    if p['decor'] > 0.01:                    # decorrelate the grain bus's mid (the film's sparkle: corr ~0.1)
        mid, side = (out[:, 0] + out[:, 1]) / 2, (out[:, 0] - out[:, 1]) / 2
        out = fx.decorrelate(mid, p['decor'], cut=400.0) + np.stack([side, -side], 1)
    return out, env


def _run(p, r, n):
    env = envelope(p, n)
    out = np.zeros((n, 2))
    notes = scale_freqs(p['g_lo'], p['g_hi'], p['scale'], p['key'])
    N = int(p['notes'])
    if p['direction'] == 'up':
        idx = np.linspace(0, len(notes) - 1, N)
    elif p['direction'] == 'down':
        idx = np.linspace(len(notes) - 1, 0, N)
    else:
        idx = (len(notes) - 1) * np.sin(np.linspace(0, np.pi, N)) ** 0.8
    idx = np.round(idx).astype(int)
    r2 = p.get('r2_path', 0)                 # round-2 code path (see render): no detune draw, 4 s note cap
    for k in range(N):
        tk = p['run_t'] * (k / max(N - 1, 1)) ** p['accel'] + r.uniform(-0.004, 0.004) * (k > 0)
        i = int(max(tk, 0) * SR)
        m = min(dsp.n_of(min(6 * p['n_decay'], 4.0 if r2 else 6.0)), n - i)
        if m < 16:
            continue
        f = notes[idx[k]] if r2 else notes[idx[k]] * 2 ** (r.uniform(-1, 1) * p['detune'] / 1200)
        tau = p['n_decay'] * (f / 1000.0) ** -p['f_decay']
        y = _note(f, m, tau, r, BAR, p['hard'], p['g_bell'] * 0.8 + 0.2, p['beat'], p['swing'], p['sizzle'])
        g = _tilt_gain(f, p['tilt']) * 10 ** (r.uniform(-1, 1) * p['g_jit_db'] / 40)
        pos = (1 if k % 2 else -1) * p['spread'] * r.uniform(0.6, 1.0)        # note by note, hard-panned
        gL, gR = _pan_gains(pos)
        out[i:i + m, 0] += g * gL * y
        out[i:i + m, 1] += g * gR * y
    return out * 0.3, env


def _chimes(p, r, n):
    env = envelope(p, n)
    out = np.zeros((n, 2))
    notes = scale_freqs(p['g_lo'], p['g_hi'], p['scale'], p['key'])
    T = int(p['tubes'])
    pick = np.linspace(0, len(notes) - 1, min(T, len(notes))).round().astype(int)
    tubes = [(notes[j], r.uniform(-1, 1) * p['spread']) for j in pick]
    t = 0.0
    r2 = p.get('r2_path', 0)                 # round-2 code path (see render): no detune draw, 4 s note cap
    for k in range(int(p['strikes'])):
        i = int(t * SR)
        if i >= n - 16:
            break
        f, pos = tubes[int(r.integers(0, len(tubes)))]
        if not r2:
            f = f * 2 ** (r.uniform(-1, 1) * p['detune'] / 1200)
        m = min(dsp.n_of(min(5 * p['n_decay'], 4.0 if r2 else 6.0)), n - i)
        tau = p['n_decay'] * (f / 1000.0) ** -p['f_decay']
        y = _note(f, m, tau, r, TUBE, p['hard'], 0.8, p['beat'], p['swing'], p['sizzle'])
        g = 10 ** (r.uniform(-1, 1) * p['g_jit_db'] / 20) * (0.92 ** k)
        gL, gR = _pan_gains(pos)
        out[i:i + m, 0] += g * gL * y
        out[i:i + m, 1] += g * gR * y
        t += r.exponential(0.06 + 0.012 * k)                  # a burst of strikes, thinning out
    return out * 0.3, np.ones(n)


def _bed(p, r, n, env):
    """the steady tone (snapped to the scale root) + an air wash, both under the envelope"""
    t = dsp.t_of(n)
    root = 16.3516 * 2 ** (p['key'] / 12)
    f = root * 2 ** np.round(np.log2(p['tone_hz'] / root))
    vib = 1 + p['vib'] * np.sin(2 * np.pi * 5.1 * t + r.uniform(0, 6.3))
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    tone = np.sin(ph) + p['tone_h'] * (0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph))
    tone = tone / (np.sqrt(np.mean(tone ** 2)) + 1e-12) * 0.05 * 10 ** (p['tone_db'] / 20)
    if p['tone_db'] <= -39.0 and not p.get('r2_path', 0):   # round 3: 'off' means off (an air-only bed left a
        tone = np.zeros(n)                                  # -40 dB tone line; r2_path keeps that line)
    tone = fx.decorrelate(tone * env, p['tone_w'], cut=200.0)
    air = dsp.butter(r.standard_normal(n), 'high', p['air_hp'], order=2)
    air = dsp.butter(air, 'low', 14000, order=2)
    air = air / (np.sqrt(np.mean(air ** 2)) + 1e-12) * 0.05 * 10 ** (p['air_db'] / 20)
    air2 = dsp.butter(r.standard_normal(n), 'high', p['air_hp'], order=2)
    air2 = dsp.butter(air2, 'low', 14000, order=2)
    air2 = air2 / (np.sqrt(np.mean(air2 ** 2)) + 1e-12) * 0.05 * 10 ** (p['air_db'] / 20)
    return tone + np.stack([air * env, air2 * env], 1)


def render(p, r):
    p = {**DEFAULTS, **p}
    n = dsp.n_of(p['length'])
    st = p['style']
    grains, env = (_sparkle if st == 'sparkle' else _run if st == 'run' else _chimes)(p, r, n)
    y = grains
    if p['tone_db'] > -39.0 or p['air_db'] > -49.0:      # bed levels are relative to the grains' loudest 200 ms
        h = dsp.n_of(0.2)
        k = max(1, n // h)
        ref = np.sqrt(((grains[:k * h] ** 2).mean(1)).reshape(k, h).mean(1).max()) + 1e-9
        if st == 'sparkle':
            benv = env
        elif p['ring']:                                      # round 3: the bed follows the ringing notes
            hh = dsp.n_of(0.02)
            kk = max(1, n // hh)
            e = np.sqrt(((grains[:kk * hh] ** 2).mean(1)).reshape(kk, hh).mean(1))
            benv = np.interp(np.arange(n), np.arange(kk) * hh + hh / 2, e / (e.max() + 1e-12))
        else:
            benv = envelope(p, n)
        bed = _bed(p, r, n, benv)
        y = y + bed * ref / 0.05
    if p['ring'] and st != 'sparkle' and p['decor'] > 0.01:     # round 3: wider ring-outs (the chimes read corr 0.97)
        mid, side = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
        y = fx.decorrelate(mid, p['decor'], cut=500.0) + np.stack([side, -side], 1)
    y = dsp.butter(y, 'high', p['hp'], order=4)
    if p['deess'] > 0.2:
        y = dsp.peq(y, 9000.0, -p['deess'], 0.8)
    if p['verb'] > 0.01:
        y = fx.reverb(y, r, rt60=p['rt60'], size=0.7, predelay=0.015, damp=0.45, wet=p['verb'], stereo=True,
                      width=1.0)
    y = y[:n]
    if st != 'sparkle' and not p['ring']:                    # runs / chimes: the envelope shapes the whole mix
        y = y * envelope(p, n)[:, None]
    if p['gate_end']:
        k = dsp.n_of(0.03)
    elif p['ring']:                                          # round 3: free ring-out, only the end fades
        k = dsp.n_of(min(p['tail_fade'], 0.4 * p['length']))
    else:
        k = dsp.n_of(min(0.3, 0.15 * p['length']))
    w = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    y = np.array(y)
    y[-k:] *= w[:, None]
    if p['ring']:                                            # round 3: start on zero (a mallet / strike at t = 0)
        a = dsp.n_of(0.0005)
        y[:a] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a)))[:, None]
    return y


# ------------------------------------------------------------------------------------------------ pro targets
TARGETS = {   # (key, lo, hi, scale): the two ref_2 trims measured with measure() + docs/motion-sfx-study.md
    'pro_shimmer': [('length_s', 1.4, 2.6, 0.4), ('corr', -0.2, 0.45, 0.2), ('tilt', 2.0, 7.0, 3.0),
                    ('low_db', -200.0, -30.0, 6.0)],
    'shimmer': [('length_s', 1.5, 5.0, 0.5), ('low_db', -200.0, -25.0, 6.0)],
}


def pro_penalty(x, cat):
    m = measure(x)
    m['low_db'] = 10 * np.log10(10 ** (m['shares'][0] / 10) + 1e-12)
    return sum(min(2.0, _outside(m.get(k), lo, hi, sc)) for k, lo, hi, sc in TARGETS.get(cat, []))


def extra_for(cat, weight=1.0):
    return lambda x, row: -weight * pro_penalty(x, cat)
