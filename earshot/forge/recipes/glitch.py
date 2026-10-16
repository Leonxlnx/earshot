"""A designed digital glitch / stutter for motion design: crackle, then a gated stutter that speeds up into the cut and
stops dead on it.

What the pro films do (docs/motion-sfx-study.md, 'Glitch / stutter'; the one exposed example, ref_2 27.90-28.60):
crackle first (sparse high grains, 4-12 kHz), then a dense broadband stutter: 11 bursts at 28 ms median spacing
(36/s), speeding up from 32 ms to 22 ms into the cut; bright (centroid 9.2 kHz, tilt +3.8 dB/oct, 8 kHz band
strongest; shares <150 Hz -22, 150 Hz-1 kHz -25, 1-4 kHz -17, 4-12 kHz -1, > 12 kHz -7 dB re total); fully
decorrelated (L/R correlation 0.04, side as loud as mid); it ends exactly on the cut and the next sound takes over.
The techniques notes (docs/motion-sfx-techniques.md, section 10): buffer repeats of 5-50 ms slices, bit-crush 4-10
bits, sample-rate reduction (2-8 kHz = harsh aliasing on purpose), data chirps (square / pulse bursts at 1-4 kHz, a
new pitch per burst), sharp transients with clear pitch survive chopping; a steady low layer keeps it from thinning;
lock the slices to the picture's frames (33 / 42 ms). Mistakes: chops drifting off the grid, stacked harshness, a thin
low end.

How it is built (every sample computed, no recorded audio):
  source   per burst, a short 'buffer' made of crushed noise (bits `bits`, sample-and-hold at `srate`), a data chirp
           (band-limited square at a new random-stepped pitch around `tone_f`, `tonal` share) and an FM 'modem' layer
           (`fm` share); `slices` distinct buffers are made, and a burst replays the previous one (`repeat`: the
           buffer-repeat identity of a stutter), takes another, or plays reversed (`rev`); `pitch_step` semitones per
           burst (a stutter that climbs or sinks)
  stutter  bursts on a grid that accelerates from `grid0` to `grid1` s (power `accel`), each gated to `gate` of its
           slot (box with 0.4 ms edges: hard, but no clicks), a crescendo of `swell` dB across the stutter
  crackle  the first `crackle` share of the length: sparse grains (1.5-6 ms) whose density rises to `crackle_rate`/s,
           high-passed at `crackle_hp`, at `crackle_db` under the stutter, independent per channel
  low      an optional low pulse under each burst (`low_db`, 55-110 Hz) so the low end does not vanish
  stereo   `width` = the share of bursts whose right channel gets its own buffer and a 0-3 ms offset (1 = the films'
           fully decorrelated texture); crackle grains are always independent per channel
  finish   high-pass `hp`, a high-shelf tilt `bright` dB at 6 kHz, light tanh glue (`drive` dB); the last burst ends on
           the cut and only its own decay `end_tail` s follows (a hard end, not a chopped one)

Character (families): length + crackle share (a quick hit vs a crackling build), grid0 / grid1 (frame-locked 42 ->
33 ms or a fast 30 -> 18 ms roll), gate (choppy vs dense), bits / srate (grit), tonal / fm (noise glitch vs data
bleeps), pitch_step (rising / falling), width. `pattern` fixes the slice layout and the burst choices of one glitch;
each render varies the noise only, so a variant is reproducible.
"""
import numpy as np

from .. import dsp
from .. import fx

SR = dsp.SR
TARGET = 'a short digital glitch stutter with bit-crushed noise bursts that speed up and cut off'
ALTS = ['radio static', 'a helicopter', 'a woodpecker', 'rain on a tin roof']
REFSET = 'pro_glitch'

SPACE = {
    'dur': (0.35, 0.85),               # length up to the hard end (ref 0.7 s)
    'crackle': (0.12, 0.5),            # share of the length that is crackle before the stutter (ref ~0.38)
    'crackle_rate': (40.0, 300.0, 'log'),
    'crackle_hp': (1500.0, 6000.0, 'log'),
    'crackle_db': (-30.0, -6.0),
    'grid0': (0.026, 0.046),           # first stutter interval (ref 32 ms)
    'grid1': (0.015, 0.028),           # last (ref 22 ms)
    'accel': (0.6, 2.5, 'log'),        # how the grid speeds up (> 1: late)
    'gate': (0.35, 0.95),              # burst length / slot
    'swell': (0.0, 10.0),              # crescendo across the stutter, dB
    'bits': (3.0, 10.0),
    'srate': (1500.0, 16000.0, 'log'),
    'tonal': (0.0, 1.0),               # data-chirp share of the buffer
    'tone_f': (500.0, 5000.0, 'log'),
    'fm': (0.0, 1.0),
    'slices': (1, 5, 'int'),
    'repeat': (0.0, 0.9),
    'rev': (0.0, 0.5),
    'pitch_step': (-2.0, 2.0),
    'low_db': (-50.0, -14.0),          # low pulse under the bursts (-40 or less = none; ref: < 150 Hz -22 dB)
    'decay': (0.15, 3.0, 'log'),       # each burst's own decay, in slots (small = a hit that dies inside its slot)
    'lvl_jit': (0.0, 8.0),             # burst-to-burst level spread, dB (ref: bursts at -15..0 dB)
    'floor_db': (-72.0, -36.0),        # the bed between bursts (crackle residue; the ref's gaps sit at -40..-60 dB)
    'width': (0.0, 1.0),
    'hp': (300.0, 7000.0, 'log'),      # the stutter's high-pass (ref: 1-4 kHz -17 dB re total)
    'bright': (-6.0, 9.0),             # high shelf at 6 kHz
    'drive': (0.0, 12.0),
    'end_tail': (0.004, 0.025, 'log'), # the last burst's own decay after the cut (ref t40 42 ms)
    'pattern': (0, 999, 'int'),
}
DEFAULTS = {'dur': 0.65, 'crackle': 0.35, 'crackle_rate': 200.0, 'crackle_hp': 3500.0, 'crackle_db': -14.0,
            'grid0': 0.032, 'grid1': 0.022, 'accel': 1.2, 'gate': 0.7, 'swell': 4.0, 'bits': 5.0, 'srate': 6000.0,
            'tonal': 0.35, 'tone_f': 1800.0, 'fm': 0.3, 'slices': 3, 'repeat': 0.5, 'rev': 0.15, 'pitch_step': 0.0,
            'low_db': -34.0, 'decay': 0.6, 'lvl_jit': 4.0, 'floor_db': -52.0, 'width': 0.9, 'hp': 4500.0, 'bright': 3.0, 'drive': 4.0, 'end_tail': 0.012,
            'pattern': 11}


def _db(v):
    return 10 ** (v / 20)


def _t(n):
    return np.arange(n) / SR


def _buffer(p, pr, n, r):
    """one stutter buffer (n samples): crushed noise + a stepped data chirp + an FM 'modem' layer. pr = the pattern
    rng (layout: pitches, ratios), r = the render rng (noise)"""
    t = _t(n)
    w = r.standard_normal(n)
    noise = fx.crush(w, bits=int(round(p['bits'])) + 2, rate=p['srate'])
    out = (1 - 0.6 * max(p['tonal'], p['fm'])) * noise / (np.std(noise) + 1e-9)
    if p['tonal'] > 0.02:
        f = p['tone_f'] * 2 ** pr.uniform(-1, 1)
        steps = int(pr.integers(1, 4))
        fr = np.repeat(f * 2 ** (pr.integers(-5, 6, steps) / 6), int(np.ceil(n / steps)))[:n]
        sq = dsp.osc(fr, 'square' if pr.uniform() < 0.7 else 'saw')
        out = out + p['tonal'] * 1.4 * sq
    if p['fm'] > 0.02:
        fc = pr.uniform(800, 6000)
        ratio = pr.choice([0.5, 1.0, 1.41, 2.0, 3.0, 7.13])
        idx = pr.uniform(1.5, 8)
        y = np.sin(2 * np.pi * fc * t + idx * np.sin(2 * np.pi * fc * ratio * t))
        out = out + p['fm'] * 1.2 * y
    out = fx.crush(out / (np.abs(out).max() + 1e-9), bits=int(round(p['bits'])), rate=p['srate'])
    return out


def _box(n, edge=0.0004):
    e = np.ones(n)
    k = min(n // 2, max(2, dsp.n_of(edge)))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    e[:k] = ramp
    e[-k:] = ramp[::-1]
    return e


def _grid(p, t0, t1):
    """burst start times from t0, intervals accelerating grid0 -> grid1, the last slot ending at t1"""
    times = []
    t = t0
    span = max(1e-3, t1 - t0)
    while True:
        u = min(1.0, (t - t0) / span) ** p['accel']
        d = p['grid0'] + (p['grid1'] - p['grid0']) * u
        if t + d > t1 + 0.25 * d:
            break
        times.append((t, d))
        t += d
    if not times:
        times = [(t0, span)]
    return times


def render(p, r):
    p = {**DEFAULTS, **p}
    pr = dsp.rng(int(p['pattern']) * 7919 + 13)
    dur = p['dur']
    n_end = dsp.n_of(dur)
    tail_n = dsp.n_of(p['end_tail'] * 4)
    n = n_end + tail_n
    y = np.zeros((n, 2))
    t_st = dur * p['crackle']
    bursts = _grid(p, t_st, dur)
    # stutter buffers (the slice layout is the pattern's; noise per render)
    L = dsp.n_of(max(d for _, d in bursts) + 0.004)
    bufs = [_buffer(p, pr, L, r) for _ in range(int(p['slices']))]
    nb = len(bursts)
    low = np.zeros(n)
    prev = [0, 0]
    for k, (tb, d) in enumerate(bursts):
        last = k == nb - 1
        g = max(16, dsp.n_of(d * p['gate']))
        if last:
            g = max(16, n_end - dsp.n_of(tb))           # the last burst runs to the cut
        lvl = _db(p['swell'] * (k / max(1, nb - 1) - 1.0) - abs(p['lvl_jit'] * pr.standard_normal()) * (not last))
        rate = 2 ** (k * p['pitch_step'] / 12)
        for c in range(2):
            if c == 1 and pr.uniform() > p['width']:
                j, rv, off = choice_l
            else:
                j = prev[c] if pr.uniform() < p['repeat'] else int(pr.integers(0, len(bufs)))
                rv = pr.uniform() < p['rev']
                off = dsp.n_of(pr.uniform(0, 0.003)) if c == 1 and not last else 0   # both end on the cut
            if c == 0:
                choice_l = (j, rv, off)
            prev[c] = j
            b = bufs[j][::-1] if rv else bufs[j]
            if rate != 1.0:
                idx = np.minimum((np.arange(L) * rate).astype(int), L - 1)
                b = b[idx]
            m = min(g, L) if not last else g
            seg = np.resize(b, m) if m > L else b[:m].copy()
            env = _box(m) * np.exp(-_t(m) / (p['decay'] * d))
            if last:                                   # the final burst: holds to the cut, then its own decay
                seg = np.concatenate([seg, np.resize(b, tail_n)])
                hold = np.exp(-_t(m - 1) / max(p['decay'] * d, 0.5 * (m - 1) / SR))
                env = np.concatenate([hold, hold[-1] * np.exp(-_t(tail_n + 1) / (p['end_tail'] / 1.3))])[:len(seg)]
                env[:dsp.n_of(0.0004)] *= np.linspace(0, 1, dsp.n_of(0.0004))
            i = dsp.n_of(tb) + off
            seg = seg * env * lvl * float(np.exp(0.12 * r.standard_normal()))
            mm = min(len(seg), n - i)
            if mm > 0:
                y[i:i + mm, c] += seg[:mm]
        if p['low_db'] > -39.5:
            m = min(dsp.n_of(d * p['gate']) + dsp.n_of(0.006), n - dsp.n_of(tb))
            f0 = pr.uniform(55, 110)
            lo = np.sin(2 * np.pi * f0 * _t(m)) * np.exp(-_t(m) / 0.012) * _box(m, 0.001)
            i = dsp.n_of(tb)
            low[i:i + m] += _db(p['low_db']) * lvl * lo
    y = dsp.butter(y, 'high', p['hp'], order=4)          # the stutter is bright (ref: 4-12 kHz holds the energy)
    y += low[:, None] * np.abs(y).max()
    # crackle: sparse grains, density rising, independent per channel
    cr = np.zeros((n, 2))
    if p['crackle'] > 0.02 and t_st > 0.02:
        for c in range(2):
            t = 0.0
            while True:
                u = t / t_st
                rate = p['crackle_rate'] * (0.25 + 0.75 * u ** 1.5)
                t += float(r.exponential(1.0 / rate))
                if t >= t_st:
                    break
                m = dsp.n_of(r.uniform(0.0015, 0.006))      # grains, not clicks: a few ms, smooth edges
                gr = r.standard_normal(m) * np.hanning(m) ** 0.5 * np.exp(-_t(m) / (m / SR / 2))
                if r.uniform() < 0.2:                 # a few low blips among the high grains (smooth, not crushed:
                    gr = np.sin(2 * np.pi * r.uniform(150, 900) * _t(m)) * np.hanning(m) * 1.5   # steps would click)
                i = dsp.n_of(t)
                a = _db(p['crackle_db'] + 18 * np.log10(r.uniform(0.05, 1.0)) + 6 * u)
                mm = min(m, n - i)
                cr[i:i + mm, c] += a * gr[:mm]
        # the crackle grains go through their own high-pass (the low pops keep a little body)
        cr = 0.85 * dsp.butter(cr, 'high', p['crackle_hp'], order=2) + 0.15 * dsp.butter(cr, 'high', 120, order=2)
        y += cr * np.abs(y).max()
    if p['floor_db'] > -71.5:          # the bed under the stutter: faint crushed hiss, so the gaps are not digital silence
        bed = fx.crush(dsp.butter(r.standard_normal((n, 2)), 'high', 3000, order=2), 6, p['srate'])
        u = np.minimum(1.0, _t(n) / max(0.02, t_st + 0.02))[:, None]
        y += _db(p['floor_db']) * np.abs(y).max() * bed / (np.std(bed) * 4) * u * (_t(n) < dur)[:, None]
    y = dsp.peq(y, 6000, p['bright'], 0.5)
    if p['drive'] > 0.2:
        y = fx.saturate(y / (np.abs(y).max() + 1e-9), p['drive'], 'tanh', os=2)
    y = dsp.butter(y, 'low', 18000, order=8)            # nothing above ~19 kHz (crush images; polish flags > 20 kHz)
    y[:2] = 0.0
    return dsp.fade(y, 0.0005, 0.003)


# ------------------------------------------------------------------------------------------------ study checks
def glitch_stats(x):
    """the study's glitch numbers, model-free: L/R correlation, spectral centroid, 4-12 kHz share (dB re total),
    the end (last 10 ms dB re the loudest 5 ms: a hard end that is not a chopped one sits at -30..-60), the burst
    intervals of the stutter (1 ms envelope onsets) early vs late (accelerating = late < early)"""
    x = np.asarray(x, np.float64)
    st = x if x.ndim == 2 else np.stack([x, x], 1)
    m = st.mean(1)
    corr = float(np.corrcoef(st[:, 0], st[:, 1])[0, 1]) if np.std(st[:, 0] - st[:, 1]) > 1e-9 else 1.0
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    cen = float((f * P).sum() / (P.sum() + 1e-14))
    s412 = 10 * np.log10(P[(f >= 4000) & (f < 12000)].sum() / (P.sum() + 1e-14) + 1e-14)
    k5 = int(SR * 0.005)
    n5 = len(m) // k5
    e5 = 10 * np.log10((m[:n5 * k5].reshape(n5, k5) ** 2).mean(1) + 1e-14)
    tail = 10 * np.log10(np.mean(m[-int(SR * 0.01):] ** 2) + 1e-14) - e5.max()
    k1 = int(SR * 0.001)
    n1 = len(m) // k1
    e1 = 10 * np.log10((m[:n1 * k1].reshape(n1, k1) ** 2).mean(1) + 1e-14)
    e1 -= e1.max()
    on = [i for i in range(1, len(e1)) if e1[i] > -14 >= e1[i - 1]]
    keep = []
    for i in on:
        if not keep or i - keep[-1] >= 12:
            keep.append(i)
    d = np.diff(keep)
    h = len(d) // 2
    early, late = (float(np.median(d[:h])), float(np.median(d[h:]))) if len(d) >= 4 else (None, None)
    edges = [0, 150, 1000, 4000, 12000, 24001]
    sh = [round(float(10 * np.log10(P[(f >= a) & (f < b)].sum() / (P.sum() + 1e-14) + 1e-14)), 1)
          for a, b in zip(edges[:-1], edges[1:])]
    return {'corr': round(corr, 2), 'centroid_hz': round(cen), 's412_db': round(float(s412), 1), 'shares': sh,
            'end_db': round(float(tail), 1), 'bursts': len(keep), 'early_ms': early, 'late_ms': late}


def study_extra(x, row=None):
    """Critic(extra=...) for the pro glitch: minus the distance to the study's box (correlation < 0.3, centroid
    6.5-11 kHz, 4-12 kHz share >= -4 dB, end -30..-70 dB in the last 10 ms, bursts accelerating), capped at -3"""
    s = glitch_stats(x)
    pen = (min(1.0, max(0.0, s['corr'] - 0.3) / 0.4) + min(1.0, max(0.0, 6500 - s['centroid_hz'], s['centroid_hz']
                                                                    - 11000) / 2500)
           + min(1.0, max(0.0, -4 - s['s412_db']) / 6) + min(1.0, max(0.0, s['end_db'] + 30) / 10)
           + (0.5 if s['bursts'] < 6 else 0.0)
           + (min(1.0, max(0.0, s['late_ms'] - s['early_ms'] + 2) / 8) if s['early_ms'] is not None else 0.5)
           + min(1.5, sum(max(0.0, abs(a - b) - 6) for a, b in zip(s['shares'], PRO_SHARES)) / 12))
    return -min(3.0, 0.6 * pen)


PRO_SHARES = (-21.7, -24.8, -17.0, -1.2, -6.6)   # the ref's energy in < 150 Hz, 150-1k, 1-4k, 4-12k, > 12k (dB re total)


def generic_extra(x, row=None):
    """Critic(extra=...) for the generic glitch category (no references): only the gesture rules of the study (wide,
    a stutter of >= 6 bursts that does not slow down, a hard end that is not chopped), not the pro film's exact balance"""
    s = glitch_stats(x)
    pen = (min(1.0, max(0.0, s['corr'] - 0.5) / 0.4) + min(1.0, max(0.0, s['end_db'] + 30) / 10)
           + (0.5 if s['bursts'] < 6 else 0.0)
           + (min(1.0, max(0.0, s['late_ms'] - s['early_ms'] + 2) / 8) if s['early_ms'] is not None else 0.5))
    return -min(3.0, 0.6 * pen)
