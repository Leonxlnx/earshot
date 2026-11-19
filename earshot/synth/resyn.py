"""Analysis and resynthesis at the resolution of hearing: a recording becomes a few kilobytes of numbers (never
audio), every render of those numbers is a new take, and several analysed recordings of one kind become a small model
that makes new sounds of that kind.

Why. A sound built from a handful of modes and one noise envelope is too clean: what makes a recording a recording
is its detail, the grain of every band's envelope, the exact shape of every onset, partials that glide and beat. The
ear resolves about one ERB in frequency and one or two milliseconds in time (a quarter of a millisecond at an
onset), so for a listener a sound is what happens on that grid. This module keeps that and nothing finer.

The representation (the `model`, a plain dict of small integer and float arrays):

  bands      56 bands from 40 Hz to 20 kHz, one every 0.73 ERB, half-cosine windows on the ERB scale whose squares
             sum to one. Each band is held as a complex baseband signal at 4 kHz (its analytic signal moved down to
             zero frequency: one small inverse FFT per band), so analysis and synthesis invert each other exactly
             and a render costs a few small FFTs per band and one large one.
  envelope   every band's level in dB on one time grid: a frame every 2 ms, and every 0.25 ms from 3 ms before to
             9 ms after each detected onset (onsets are what a vocoder smears). Stored as a smooth part (six
             components, their time rows every 8 ms and at every onset frame, 8 bits) plus the grain: what the
             smooth part misses, cell by cell, in steps of 2 dB (4 dB under -30 dB). The grain is where a
             recording's detail lives; deflate packs it into about one bit per cell.
  carrier    what a band's envelope is put on. A band is as coherent as it was measured to be: at anchors every
             8 ms and at every onset the model keeps how much of the band is one carrier (two statistics of how
             predictable the phase is over the time in which band noise forgets itself; the second also passes a
             gliding partial), that carrier's frequency (in hundredths of the band's width, the rounding error
             carried on so the phase does not drift) and, where a coherent stretch starts, its phase. The coherent
             share is rendered by a phase-continuous oscillator: a partial stays a partial within a cent, a low
             thump keeps its pitch and its glide, neighbouring bands that share a partial stay in step, and at an
             onset the bands start together as the one pulse they were. The rest is fresh noise in the band with
             the noise's own envelope divided out (a band of noise has a Rayleigh envelope of its own; an envelope
             imposed on top of it is the watery sound of a noise vocoder). Because neighbouring bands overlap, the
             take's band envelopes are then measured once and corrected towards the model's.
  stereo     per band the level difference, the signed coherence and the phase between the two channels (a
             ringing band's carrier is turned by it in the right channel), plus one level-difference track over time
             (a pass moves across).

A take is not a copy: every render draws its noise, and the model can be transformed in ways that stay physical:
`length` (time-warp between onsets, onsets stay as sharp), `pitch` and `size` (a bigger thing is lower and slower:
the take is resampled, the warp compensates for pitch alone), `bright` (tilt, dB per octave), `hard` (level and
brightness of the onsets), `ring` (decay time of the ringing bands), `weight` (level under 150 Hz), `vary` (small
random amounts of all of them, and that share of the grain drawn fresh from its statistics). `full=True` adds the
finish of a library sound (low-end weight, presence, air, a little tape saturation, a limiter to a loudness target):
see finish().

From replication to generation: _canon() puts a model on a fixed grid around its main event (a relative part before,
12 ms at full resolution around it, a relative part after, the two lengths kept as numbers), so models of one kind
line up. morph(a, b, t) is the weighted mean of two; fit_class(models) is their mean and principal components;
member(class_model, r) draws a point of that space, gives it back the fine detail averaging removes and the onset
phases of the nearest member, and sample() renders it: a sound of the kind that no one recorded. Members must be
single events of one kind: a second event (the release of a click) is not aligned and blurs.

Fidelity, measured the way hearing would notice (fidelity()): env_db, the distance between the band envelopes of
recording and take in dB RMS at 2 ms over the cells that are heard (within 40 dB of the loudest and 25 dB of their
band's loudest moment within 6 ms); shape_db, the same after smoothing over one band and 5 ms (the structure
without the grain: two recordings of one noise differ by about 3 dB in env_db and by under 1 dB in shape_db);
rise_ms, the 10-90 % rise of the main onset; mod_db, the modulation spectrum of each band's envelope in octaves from
4 to 128 Hz; for tonal sounds the strongest partials' frequencies and decay times (partials()). A model is right
when its take differs from the recording by about as much as two of its takes differ from each other.

Measured on 41 public-domain recordings of 0.04 to 1.8 s (whooshes, hits, pops, clicks, swells): models of 4 to 35 kB
(9 kB median: about 5 kB for a quarter second, 15 kB per second of dense noise), analysis 0.2 s and a render 0.3 to
0.4 s per second of sound on one laptop core.

    from earshot.forge import dsp
    from earshot.synth import resyn
    m = resyn.analyze(x, 48000)                         # x: a recording, mono (n,) or stereo (n, 2)
    y = resyn.render(m, dsp.rng(1))                     # a take, (n, 2) float64
    z = resyn.render(m, dsp.rng(2), size=1.3, hard=0.5, vary=0.4, full=True)
    resyn.save('knock.npz', m); resyn.size(m)           # numbers only; bytes
    c = resyn.fit_class([m1, m2, m3]); new = resyn.sample(c, dsp.rng(5))
    resyn.fidelity(x, y, 48000)                         # {'env_db', 'shape_db', 'rise_ms', 'mod_db', 'level_db'}

Models made from recordings are derivatives of them: fit only recordings in the public domain and keep the credit.
"""
import io
import json
from functools import lru_cache

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, signal

from ..forge import dsp, fx

SR = dsp.SR
NB = 56                          # bands
DEC = 12                         # a band lives at SR / DEC
FS = SR / DEC                    # 4 kHz: one baseband sample is 0.25 ms
COARSE = 8                       # baseband samples per envelope frame away from onsets (2 ms)
PRE, POST = 12, 36               # baseband samples at full resolution before and after an onset (3 ms, 9 ms)
ANCHOR = 32                      # baseband samples between carrier anchors (8 ms)
PANHOP = 64                      # baseband samples between points of the stereo track (16 ms)
PADB = 320                       # silence around a signal, baseband samples (80 ms: the low bands ring that long)
FLOOR_DB = -80.0                 # under the loudest cell: silence
CA, CD = 32, 120                 # canonical frames before and after the main event (see _canon)
LATE = 0.4                       # s after the start from which analyze(late=) stores coarser frames and anchors


def _erb(f):
    """ERB number of a frequency (Glasberg & Moore 1990)"""
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f, np.float64))


def _hz(e):
    return (10 ** (np.asarray(e, np.float64) / 21.4) - 1.0) / 0.00437


E0 = float(_erb(40.0))
STEP = (float(_erb(20000.0)) - E0) / (NB - 1)
CENTRES = _hz(E0 + STEP * np.arange(NB))
WIDTH = STEP * 24.7 * (0.00437 * CENTRES + 1.0)                # Hz from one centre to the next
LAG = np.clip(np.round(FS / WIDTH), 2, 64).astype(int)         # baseband samples after which band noise forgets itself


# ------------------------------------------------------------------------------------------------ filter bank
@lru_cache(maxsize=32)
def _bank(m):
    """the bands for an FFT of DEC * m points: per band the first and last bin, the window over those bins, the
    reference bin (the band's zero frequency) and where each bin lands in the band's own m-point FFT"""
    n = DEC * m
    e = (_erb(np.arange(n // 2 + 1) * (SR / n)) - E0) / STEP
    out = []
    for b in range(NB):
        d = e - b
        h = np.cos(0.5 * np.pi * np.clip(d, -1.0, 1.0))
        if b == 0:
            h[d < 0] = 1.0
        k = np.nonzero(h > 1e-6)[0]
        k0, k1 = max(int(k[0]), 1), int(k[-1]) + 1
        kc = int(round(CENTRES[b] * n / SR))
        out.append((k0, k1, h[k0:k1].copy(), kc, (np.arange(k0, k1) - kc) % m))
    return tuple(out)


def _fast(count):
    return sfft.next_fast_len(int(np.ceil(count)), real=False)


def _split(x, m):
    """the complex baseband signal of every band, (NB, m) at FS, of x padded to DEC * m samples"""
    X = sfft.rfft(x, DEC * m)
    w = np.empty((NB, m), np.complex128)
    S = np.zeros(m, np.complex128)
    for b, (k0, k1, h, kc, idx) in enumerate(_bank(m)):
        S[idx] = X[k0:k1] * h
        w[b] = sfft.ifft(S)
        S[idx] = 0.0
    return w * (2.0 / DEC)


def _prep(x, sr):
    x = np.asarray(x, np.float64)
    x = x[:, None] if x.ndim == 1 else x[:, :2]
    if int(sr) != SR:
        g = int(np.gcd(int(sr), SR))
        x = signal.resample_poly(x, SR // g, int(sr) // g, axis=0)
    x = x - x.mean(axis=0)
    if x.shape[1] == 2 and np.abs(x[:, 0] - x[:, 1]).max() < 1e-4 * (np.abs(x).max() + 1e-12):
        x = x[:, :1]
    return x


def _bands(x):
    """band signals of a prepared signal (n, channels): a list of (NB, m) arrays, PADB baseband samples of silence
    before the first sample"""
    m = _fast(len(x) / DEC + 2 * PADB)
    out = []
    for c in range(x.shape[1]):
        p = np.zeros(DEC * m)
        p[PADB * DEC:PADB * DEC + len(x)] = x[:, c]
        out.append(_split(p, m))
    return out


def _tri(p, step=COARSE):
    """power smoothed over one frame of `step` baseband samples each side (the envelope such a frame stores)"""
    k = 1.0 - np.abs(np.arange(1 - step, step)) / step
    return ndimage.convolve1d(p, k / k.sum(), axis=-1, mode='constant')


# ------------------------------------------------------------------------------------------------ analysis
def _onsets(P, limit=24):
    """onsets: baseband positions where the bands above 300 Hz rise together by more than 4 dB within 2 ms (mean
    over the audible bands) and something audible follows; each placed on the peak of the pulse it starts"""
    e = np.sqrt(P)
    ref = e.max()
    act = (e.max(axis=1) > ref * 0.01) & (CENTRES > 300.0)
    if act.sum() < 4:
        act = e.max(axis=1) > ref * 0.01
    c = np.log10(e[act] + ref * 1e-3)
    nov = np.zeros(e.shape[1])
    nov[8:] = np.maximum(c[:, 8:] - c[:, :-8], 0.0).mean(axis=0)
    broad = np.sqrt(P[act].sum(axis=0))
    peak = (nov == ndimage.maximum_filter1d(nov, 33)) & (nov > 0.2)
    peak &= ndimage.maximum_filter1d(broad, 17) > 0.02 * broad.max()
    idx = np.nonzero(peak)[0]
    idx = idx[np.argsort(nov[idx])[::-1][:limit]]
    out = []
    for i in idx:
        a = max(int(i) - 6, 0)
        out.append(a + int(np.argmax(broad[a:int(i) + 7])))
    return np.unique(np.array(out, np.int64))


def _step(smooth, top):
    """the step in dB in which the grain of an envelope cell is kept, by the level of the smooth part there: 2 dB
    down to 30 dB under the loudest cell, 4 dB below (the smooth part can miss a short quiet event by tens of dB, so
    no cell goes without)"""
    return np.where(smooth - top > -30.0, 2.0, 4.0)


def _calm(grid):
    """which frames of a grid lie outside onset windows (a coarse step from the frame before)"""
    return np.diff(grid, prepend=grid[0] - COARSE) > 0.5 * COARSE


def _kept(grid):
    """the frames at which the smooth part's time rows are stored: every frame of an onset window, every fourth
    calm one, the last"""
    keep = ~_calm(grid) | (np.arange(len(grid)) % 4 == 0)
    keep[-1] = True
    return np.nonzero(keep)[0]


def _smooth(model, grid):
    """the smooth part of a model's envelope in dB, (NB, frames)"""
    at, j = _kept(grid), np.arange(len(grid))
    rows = model['h'].astype(np.float64) * (model['hs'].astype(np.float64)[:, None] / 127.0)
    return model['mean'].astype(np.float64)[:, None] + model['u'].astype(np.float64) @ np.stack(
        [np.interp(j, at, v) for v in rows])


def _pack(G, top, grid, rank=6):
    """an envelope G (bands, frames) in dB as the model keeps it: a smooth part (mean per band plus `rank`
    components whose time rows are kept in 8 bits at the frames of _kept) and the grain, what the smooth part
    misses, in whole steps (_step): `q` (deflate packs these small integers into about one bit per cell). `rough`
    is the grain's standard deviation per band over the audible calm frames: what a take that draws its own grain
    uses. Returns the dict of arrays."""
    calm = _calm(grid)
    wt = np.where(calm, 1.0, 1.5)
    mean = G.mean(axis=1)
    U, S, Vt = np.linalg.svd((G - mean[:, None]) * wt[None, :], full_matrices=False)
    r = max(1, min(int(rank), len(S)))
    h = (Vt[:r] / wt[None, :])[:, _kept(grid)]
    hs = np.abs(h).max(axis=1) + 1e-12
    out = {'top': np.float32(top), 'mean': mean.astype(np.float16), 'u': (U[:, :r] * S[:r]).astype(np.float16),
           'h': np.round(h / hs[:, None] * 127.0).astype(np.int8), 'hs': hs.astype(np.float32)}
    L = _smooth(out, grid)
    st = _step(L, float(out['top']))
    q = np.round((G - L) / st)
    q[(G < top - 50.0) & (L < G + 4.0)] = 0.0                # too quiet to hear, and the smooth part no louder
    out['q'] = q.clip(-127, 127).astype(np.int8)
    aud = (G > top - 50.0) & calm[None, :]
    out['rough'] = np.sqrt(np.sum(np.where(aud, out['q'] * st, 0.0) ** 2, axis=1)
                           / np.maximum(aud.sum(axis=1), 1)).astype(np.float16)
    return out


def _steps(f, dt, entry, used, width):
    """frequencies f (Hz from the band's centre, one per interval between anchors) in whole hundredths of the band's
    width, as int8. The rounding error is carried into the next interval (in cycles), so the phase an oscillator
    reaches never drifts from the measured one by more than half a step times an interval; a stretch that starts
    with a stored phase starts with no debt."""
    step = width / 100.0
    out = np.zeros(len(f), np.int8)
    debt = 0.0
    for a in np.nonzero(used)[0]:
        d = dt[a - 1] if a > 0 else 0.0
        v = int(np.clip(np.round((f[a] + (debt / d if d > 0 else 0.0)) / step), -127, 127))
        out[a] = v
        debt = 0.0 if entry[a] else debt + (f[a] - v * step) * d
    return out


def _sums(v):
    """running sums of v with a leading zero, so that a window's sum is a difference"""
    return np.concatenate([np.zeros(1, v.dtype), np.cumsum(v)])


def _local(w, pos, half, lag, on):
    """a band's carrier at positions pos, each from up to 2 * half samples around it that no onset (on) divides:
    frequency (Hz from the band's zero), two coherences and the phase. The first coherence asks how far the phase
    `lag` samples on is predictable (1 for a partial or a lone pulse, small for noise: lag is where band noise has
    forgotten itself); the second asks the same of the phase's advance (half a lag on, twice), which a gliding
    partial also passes (a kick, a chirp)."""
    n = len(w)
    mag = np.abs(w)
    a1 = _sums(w[1:] * np.conj(w[:-1]))
    al, nl = _sums(w[lag:] * np.conj(w[:-lag])), _sums(mag[lag:] * mag[:-lag])
    g = max(lag // 2, 1)
    ag = _sums(w[2 * g:] * np.conj(w[g:-g]) ** 2 * w[:-2 * g])
    ng = _sums(mag[2 * g:] * mag[g:-g] ** 2 * mag[:-2 * g])
    lo, hi = np.clip(pos - half, 0, n), np.clip(pos + half, 0, n)
    if len(on):
        i = np.searchsorted(on, pos, 'right')
        lo = np.maximum(lo, np.where(i > 0, on[np.maximum(i - 1, 0)], -1))
        hi = np.minimum(hi, np.where(i < len(on), on[np.minimum(i, len(on) - 1)], n))
    f = np.angle(a1[np.clip(hi - 1, lo, n - 1)] - a1[np.minimum(lo, n - 1)]) * (FS / (2 * np.pi))
    out = []
    for acc, nrm, span in ((al, nl, lag), (ag, ng, 2 * g)):
        h2 = np.clip(hi - span, lo, len(acc) - 1)
        l2 = np.minimum(lo, len(acc) - 1)
        out.append(np.where(h2 - l2 > span // 2 + 1, np.abs(acc[h2] - acc[l2]) / (nrm[h2] - nrm[l2] + 1e-30), 0.0))
    d = np.exp(2j * np.pi * f / FS)
    p0, p1, p2 = np.clip(pos - 1, 0, n - 1), np.clip(pos, 0, n - 1), np.clip(pos + 1, 0, n - 1)
    return f, out[0], out[1], np.angle(w[p0] * d + w[p1] + w[p2] / d)


def _pulse(w, p, half):
    """a band's carrier at an onset p, from the `half` samples before it and the `half` after it apart (what rings
    in before an event need not have the frequency of what follows): the frequency of each side, how much of the
    band is one carrier per side (1: the band's share of a click, a partial starting), and the phase at p of the
    side with more energy"""
    side = []
    for lo, hi in ((max(p - half, 0), p + 1), (p, min(p + half + 1, len(w)))):
        seg = w[lo:hi]
        f = np.angle(np.sum(seg[1:] * np.conj(seg[:-1]))) * (FS / (2 * np.pi))
        s = np.sum(seg * np.exp(-2j * np.pi * f * (np.arange(lo, hi) - p) / FS))
        side.append((f, np.abs(s), np.sum(np.abs(seg)), np.angle(s)))
    (fl, sl, el, pl), (fr, sr, er, pr) = side
    return fl, fr, (sl + sr) / (el + er + 1e-30), pr if er >= el else pl


def analyze(x, sr=SR, late=1):
    """the model of a recording x (mono (n,) or stereo (n, 2)) at sample rate sr: see the module text. late: the
    step of the envelope frames and of the carrier anchors from LATE s after the sound starts, as a multiple of the
    steps before (1: 2 ms and 8 ms throughout; 4: 8 ms and 32 ms there, at most 8). Onset windows keep full
    resolution wherever they are. A long tail or a slow swell costs about 20 kB per second at 1; at 4 a quarter of
    that, its grain smoothed over 8 ms (tests: test_late_frames_cost_little)"""
    late = int(np.clip(late, 1, 8))
    x = _prep(x, sr)
    n = len(x)
    ws = _bands(x)
    m = ws[0].shape[1]
    stereo = len(ws) == 2
    if stereo:
        pl, pr = np.sum(np.abs(ws[0]) ** 2, axis=1), np.sum(np.abs(ws[1]) ** 2, axis=1)
        cross = np.sum(ws[0] * np.conj(ws[1]), axis=1)
        icc = np.sign(cross.real) * np.abs(cross) / np.sqrt(pl * pr + 1e-30)
        w = 0.5 * (ws[0] + np.where(icc < 0, -1.0, 1.0)[:, None] * ws[1])
        P = 0.5 * (np.abs(ws[0]) ** 2 + np.abs(ws[1]) ** 2)
    else:
        w = ws[0]
        P = np.abs(w) ** 2
    ref = P.max() + 1e-30
    on = _onsets(P)
    loud = np.nonzero(P.max(axis=0) > ref * 10 ** ((FLOOR_DB + 12.0) / 10))[0]
    first, last = max(int(loud[0]) - COARSE, 0), min(int(loud[-1]) + COARSE, m - 1)
    on = on[(on >= first) & (on <= last)]
    fine = np.zeros(m, bool)
    for p in on:
        fine[max(p - PRE, 0):p + POST] = True
    j = np.arange(m)
    after = j > first + int(round(LATE * FS))
    grid = np.nonzero((fine | (j % (COARSE * np.where(after, late, 1)) == 0)) & (j >= first) & (j <= last))[0]
    smooth = _tri(P)
    if late > 1:
        smooth[:, after] = _tri(P, COARSE * late)[:, after]
    Pg = np.where(fine[grid][None, :], P[:, grid], smooth[:, grid])
    top = 10 * np.log10(Pg.max() + 1e-30)
    G = np.maximum(10 * np.log10(Pg + 1e-30), top + FLOOR_DB)
    ap = np.arange(first - first % ANCHOR, last + ANCHOR, ANCHOR).clip(0, m - 1)
    ap = ap[(ap <= first + LATE * FS) | (ap % (ANCHOR * late) == 0)]
    ap = np.unique(np.concatenate([ap, on]))
    at_on = np.isin(ap, on)
    quiet = _tri(P)[:, ap] < ref * 10 ** ((FLOOR_DB + 14.0) / 10)
    ton = np.zeros((NB, len(ap)))
    frq = np.zeros((NB, len(ap)))
    pha = np.zeros((NB, len(ap)))
    entry = np.zeros((NB, len(ap)), bool)
    dt = np.diff(ap) / FS
    bank = _bank(m)
    for b in range(NB):
        lag = int(LAG[b])
        f, rho, glide, ph = _local(w[b], ap, int(np.clip(3 * lag, 16, 128)), lag, on)
        # the coherent share of the band's power. Above 500 Hz band noise reads 0.4 on the first statistic (0.65 one
        # time in ten, 0.85 one in a hundred) and 0.55 (0.8, 0.9) on the second, a partial 0.95 and more on both: the
        # ramps start where noise mostly ends. (Below 500 Hz the windows are shorter than the bands need and noise
        # reads higher; there a carrier with a frequency and a phase every 8 ms describes the band completely,
        # whatever it holds.)
        k = np.maximum(np.clip((rho - 0.6) / 0.3, 0.0, 1.0), np.clip((glide - 0.9) / 0.07, 0.0, 1.0))
        fl, fr = f.copy(), f.copy()
        for i in np.nonzero(at_on)[0]:
            fl[i], fr[i], ro, ph[i] = _pulse(w[b], int(ap[i]), int(np.clip(lag, 3, 40)))
            k[i] = max(k[i], float(np.clip((ro - 0.6) / 0.3, 0.0, 1.0)))
        k[quiet[b]] = 0.0
        k = np.round(np.sqrt(np.where(k < 0.2, 0.0, k)) * 7.0) / 7.0             # the amplitude share, in sevenths
        pred = np.pi * (fr[:-1] + fl[1:]) * dt
        d = np.angle(np.exp(1j * (ph[1:] - ph[:-1] - pred)))
        sure = (k[1:] > 0) & (k[:-1] > 0) & (np.abs(d) < 1.6)
        lone = np.where(k[1:] > 0, fl[1:], np.where(k[:-1] > 0, fr[:-1], 0.5 * (fr[:-1] + fl[1:])))
        fb = np.concatenate([fl[:1], np.where(sure, (pred + d) / (2 * np.pi * dt), lone)])
        kc = bank[b][3]
        used = (k > 0) | np.concatenate([[False], k[:-1] > 0])
        ton[b] = k
        entry[b] = (k > 0) & ~np.concatenate([[False], sure])
        frq[b] = _steps(fb + kc * FS / m - CENTRES[b], dt, entry[b], used, WIDTH[b])
        pha[b] = np.where(entry[b], np.angle(np.exp(1j * (ph + 2 * np.pi * ((kc * ap) % m) / m))), 0.0)
    model = {'n': np.int32(n), 'g0': np.int32(grid[0] - PADB), 'dt': np.diff(grid).astype(np.uint8),
             'on': (on - PADB).astype(np.int32), 'ap': (ap - PADB).astype(np.int32), 'ton': _coded(ton, entry),
             'frq': frq.astype(np.int8), 'pha': np.round(pha * (128 / np.pi)).clip(-128, 127).astype(np.int8)}
    model.update(_pack(G, top, grid.astype(np.float64) - PADB))
    if stereo:
        tl, tr = [ndimage.uniform_filter1d(np.sum(np.abs(v) ** 2, axis=0), PANHOP)[::PANHOP] for v in ws]
        pan = 10 * np.log10((tl + 1e-30) / (tr + 1e-30)) - 10 * np.log10((pl.sum() + 1e-30) / (pr.sum() + 1e-30))
        pan[tl + tr < 1e-3 * (tl + tr).max()] = 0.0
        model.update(ild=np.clip(10 * np.log10((pl + 1e-30) / (pr + 1e-30)), -24, 24).astype(np.float16),
                     icc=icc.astype(np.float16), pan=np.clip(pan, -18, 18).astype(np.float16), ipd=_ipd(cross))
    return model


def _ipd(cross):
    """the phase of each band's cross spectrum (left against right) in 128ths of half a turn, int8: where the
    channels of a band are coherent but shifted (a ringing mode, a low band picked up by spaced microphones), how
    far; the coherence alone (icc) does not say"""
    return np.round(np.angle(cross) * (128 / np.pi)).clip(-128, 127).astype(np.int8)


def channel_phase(x, sr):
    """the stored `ipd` of a stereo recording x (n, 2) at sample rate sr, as analyze() measures it: for a model
    made before analyze() kept it"""
    ws = _bands(_prep(x, sr))
    return _ipd(np.sum(ws[0] * np.conj(ws[1]), axis=1))


# ------------------------------------------------------------------------------------------------ the model as arrays
def _grid(model):
    """positions of the envelope frames in baseband samples from the start of the sound"""
    if 'tg' in model:
        return np.asarray(model['tg'], np.float64)
    return float(model['g0']) + np.concatenate([[0.0], np.cumsum(model['dt'], dtype=np.float64)])


def _coded(ton, entry):
    """coherence (0 .. 1 in sevenths) and the mark of an anchor that starts with its stored phase, in one byte"""
    return (np.round(np.clip(ton, 0.0, 1.0) * 7).astype(np.uint8) << 1) | entry.astype(np.uint8)


def _grain(model, grid):
    """the grain of a model's envelope in dB, (NB, frames): the stored steps times their size"""
    return model['q'].astype(np.float64) * _step(_smooth(model, grid), float(model['top']))


def _env(model):
    """the band envelopes of a model in dB on its own grid: (NB, frames)"""
    grid = _grid(model)
    return _smooth(model, grid) + _grain(model, grid)


def _offsets(model):
    """the carriers' frequencies in Hz from the band centres, (NB, anchors)"""
    return model['frq'].astype(np.float64) * (WIDTH[:, None] / 100.0)


def _blob(model):
    """a model as one byte array: a JSON list of (name, dtype, shape), then the arrays' bytes (one array in the npz:
    a zip entry per array would cost more than the small arrays hold)"""
    names = sorted(model)
    head = json.dumps([[k, np.asarray(model[k]).dtype.str, list(np.shape(model[k]))] for k in names]).encode()
    body = b''.join(np.ascontiguousarray(model[k]).tobytes() for k in names)
    return np.frombuffer(np.uint32(len(head)).tobytes() + head + body, np.uint8)


def save(path, model):
    """write a model (or a class model) as a compressed npz: numbers only"""
    np.savez_compressed(path, model=_blob(model))
    return path


def load(path):
    with np.load(path) as z:
        raw = z['model'].tobytes()
    k = int(np.frombuffer(raw[:4], np.uint32)[0])
    out, i = {}, 4 + k
    for name, dt, shape in json.loads(raw[4:4 + k].decode()):
        n = int(np.prod(shape, dtype=np.int64)) * np.dtype(dt).itemsize
        out[name] = np.frombuffer(raw[i:i + n], dt).reshape(shape).copy()
        i += n
    return out


def size(model):
    """bytes of a model as save() writes it"""
    buf = io.BytesIO()
    np.savez_compressed(buf, model=_blob(model))
    return buf.getbuffer().nbytes


# ------------------------------------------------------------------------------------------------ synthesis
def _phase(pos, pa, F, live, entry, pha, kc, m):
    """the baseband phase (bands, len(pos)) of the bands' oscillators at positions pos. Rows of F, live, entry, pha:
    the mean frequency of the interval ending at each anchor, whether the band is coherent there, whether a
    coherent stretch starts there (or the recording's phase jumped) and the phase it starts with; pa = the anchors'
    positions in this take. An oscillator starts every stretch with the stored phase and runs at the stored
    frequencies (moving linearly inside an interval): it meets the recording's phase at every anchor of an unwarped
    take, so neighbouring bands that share a partial stay in step, and it keeps its frequencies under a time-warp."""
    fb = F - (kc * FS / m)[:, None]
    T = np.maximum(np.diff(pa), 1e-3) / FS
    cum = np.concatenate([np.zeros((len(F), 1)), np.cumsum(2 * np.pi * fb[:, 1:] * T[None, :], axis=1)], axis=1)
    rad = pha * (np.pi / 128.0)
    A = len(pa)
    idx = np.arange(A)[None, :]
    last = np.maximum.accumulate(np.where(entry, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(entry, idx, A)[:, ::-1], axis=1)[:, ::-1]
    own = np.where(live | (nxt >= A), last, nxt)             # a silent anchor belongs to the stretch that follows
    own = np.where(own < 0, np.argmax(entry, axis=1)[:, None], own)
    off = np.angle(np.exp(1j * (rad - 2 * np.pi * kc[:, None] * (pa[None, :] + PADB) / m - cum)))
    beta = cum + np.take_along_axis(off, own, axis=1)
    g = np.concatenate([0.5 * (fb[:, :-1] + fb[:, 1:]), fb[:, -1:]], axis=1)
    i = np.clip(np.searchsorted(pa, pos, 'right') - 1, 0, A - 2)
    tau, Ti = (pos - pa[i]) / FS, T[i]
    b0, g0, g1 = beta[:, i], g[:, i], g[:, i + 1]
    c = np.angle(np.exp(1j * (beta[:, i + 1] - b0 - np.pi * (g0 + g1) * Ti)))
    return b0 + 2 * np.pi * (g0 * tau + (g1 - g0) * (tau * tau / (2 * Ti))) + c * (tau / Ti)


def _weight(pos, pa, ton, at_on, reach):
    """the coherent share (bands, len(pos)) of each band's carrier: the stored coherence between anchors, and
    around every onset the onset's own coherence for as long as the band's share of a pulse lasts (reach, baseband
    samples per band; half as long again to fade)"""
    k = np.stack([np.interp(pos, pa, t, left=0.0, right=0.0) for t in ton])
    span = int(1.5 * reach.max()) + 2
    for i in np.nonzero(at_on)[0]:
        a = int(np.searchsorted(pos, pa[i] - span))
        z = int(np.searchsorted(pos, pa[i] + span))
        u = np.abs(pos[None, a:z] - pa[i]) / reach[:, None]
        win = np.cos(np.clip(u - 1.0, 0.0, 0.5) * np.pi) ** 2
        k[:, a:z] = np.maximum(k[:, a:z], ton[:, i:i + 1] * win)
    return k


def _carriers(r, bank, m, fine, rows):
    """fresh complex noise in the bands `rows`, (NB, m) at unit power, each band's own envelope divided out (a band
    of noise has a Rayleigh envelope of its own; the model's envelope already holds the recording's): sample by
    sample inside the onset windows (`fine`), over a coarse frame elsewhere"""
    out = np.zeros((NB, m), np.complex64)
    sel = np.nonzero(rows)[0]
    if not len(sel):
        return out
    N = np.zeros((len(sel), m), np.complex64)
    for j, b in enumerate(sel):
        h, idx = bank[b][2], bank[b][4]
        N[j, idx] = h * (r.standard_normal(len(h)) + 1j * r.standard_normal(len(h)))
    z = sfft.ifft(N, axis=1)
    p = z.real ** 2 + z.imag ** 2
    p /= p.mean(axis=1, keepdims=True) + 1e-30
    slow = _tri(p)
    slow[:, fine] = p[:, fine]
    z = z / np.sqrt(slow + 1e-4)
    out[sel] = z / np.sqrt(np.mean(z.real ** 2 + z.imag ** 2, axis=1, keepdims=True) + 1e-30)
    return out


def _warp(model, grid, stretch, ring):
    """the frame positions of a take: (plain, ringing). Frames inside onset windows keep their spacing, the rest is
    stretched; `ring` stretches further what follows the last onset (or the loudest frame of a sound without one)"""
    on = np.asarray(model['on'], np.float64)
    d = np.diff(grid)
    mid = grid[:-1] + 0.5 * d
    fine = np.zeros(len(d), bool)
    for p in on:
        fine |= (mid >= p - PRE) & (mid <= p + POST)
    if len(on):
        t0 = on.max() + POST
    else:
        t0 = grid[int(np.argmax(np.sum(10 ** (_env(model) / 10), axis=0)))]
    step = d * np.where(fine | (mid < 0), 1.0, stretch)        # what rings in before the first sample stays put
    gu = grid[0] + np.concatenate([[0.0], np.cumsum(step)])
    gs = grid[0] + np.concatenate([[0.0], np.cumsum(step * np.where(mid > t0, ring, 1.0))])
    return gu, gs


def _avg(p, k):
    """the moving mean of p over k samples, centred, same length (shorter windows at the ends)"""
    c = np.concatenate([[0.0], np.cumsum(p)])
    i = np.arange(len(p))
    a, b = np.clip(i - k // 2, 0, len(p)), np.clip(i - k // 2 + k, 0, len(p))
    return (c[b] - c[a]) / np.maximum(b - a, 1)


TAIL_FROM = 30.0                 # dB under the loudest 10 ms where a tail's decay line starts (tidy())
# dB under the take's peak under which tidy() leaves the tail (50 ms level) and the lead-in (2 ms level) as they are:
# a finished file is trimmed where it falls under -66 dB and starts where it first passes -50 dB (master.finish),
# so the take keeps its length and its start; and the most tidy() takes away anywhere
TAIL_LOW, LEAD_LOW, DEPTH = -90.0, -48.0, 40.0


def tidy(y, take_s, model_s, tail=None, lead=None):
    """a take with what the stored envelope cannot hold taken out, by gain only (never raised, never brightened).
    The envelope is kept to about 40 to 60 dB under its loudest cell, so a take's tail stays on a faint floor of
    noise where the recording kept falling, and what leads into a strike is smeared a few milliseconds early.
    tail = (s, dB per s): from s (the recording's time, where it last stood TAIL_FROM dB under its loudest 10 ms) a
    line falling at that rate that the recording never rises above, starting at the take's own level there (or
    TAIL_FROM dB under its loudest 10 ms, if higher); the take's level over 50 ms is held under it, so the floor
    fades with the sound's own decay and a later sound under the line is left as it is. lead = (onset s, dB, s):
    for that long before the main onset (to 2 ms before it) the take's level (2 ms) is held at most that far under
    its strike (its loudest 10 ms in the 50 ms after the onset). take_s / model_s map the recording's seconds to the
    take's and back (a time-warp, size and pitch move them). The gain is smoothed over 1 ms."""
    p = np.mean(np.asarray(y, np.float64) ** 2, axis=1)
    n = len(p)
    if n < dsp.n_of(0.02):
        return y
    g = np.zeros(n)
    top = 10 * np.log10(_avg(p, dsp.n_of(0.01)).max() + 1e-20)
    peak = 20 * np.log10(np.abs(y).max() + 1e-20)
    if tail is not None:
        a = int(round(take_s(tail[0]) * SR))
        if 0 <= a < n:
            lev = 10 * np.log10(_avg(p, dsp.n_of(0.05))[a:] + 1e-20)
            line = max(top - TAIL_FROM, lev[0]) + tail[1] * (model_s(np.arange(a, n) / SR) - tail[0])
            g[a:] = np.minimum(np.maximum(line, peak + TAIL_LOW) - lev, 0.0)
    if lead is not None:
        o = take_s(lead[0])
        i0, i1 = int(o * SR), int((o + 0.05) * SR)
        a, b = max(int(round((o - lead[2]) * SR)), 0), int(round((o - 0.002) * SR))
        if b > a and i0 < n:
            hit = 10 * np.log10(_avg(p, dsp.n_of(0.01))[i0:i1].max() + 1e-20)
            lev = 10 * np.log10(_avg(p, dsp.n_of(0.002))[a:b] + 1e-20)
            g[a:b] = np.minimum(g[a:b], np.minimum(max(hit + lead[1], peak + LEAD_LOW) - lev, 0.0))
    if not g.any():
        return y
    g = _avg(np.maximum(g, -DEPTH), dsp.n_of(0.001))
    return y * (10 ** (g / 20.0))[:, None]


def render(model, r, length=1.0, pitch=0.0, size=1.0, bright=0.0, hard=0.0, ring=1.0, weight=0.0, vary=0.0,
           full=False, exact=None, tail=None, lead=None):
    """a take of a model: stereo (n, 2) float64 at the level of the recording (turned down if it would pass full
    scale). r = dsp.rng(seed): the same model, seed and arguments give the same samples, another seed another take.
    length: duration factor (the time between onsets is warped, onsets stay as sharp); pitch: semitones; size: a
    thing `size` times as big (frequencies / size, durations * size); bright: tilt in dB per octave around 1 kHz;
    hard: -1 .. 1, the onsets 6 dB softer and duller .. louder and brighter; ring: decay time factor of the ringing
    (coherent) bands; weight: dB on the bands under 150 Hz; vary: 0 .. 1, small random amounts of all of these for
    this take, and that share of the envelope's grain drawn fresh from its statistics instead of kept (a footstep
    played 500 times); full: the finish() of a library sound; exact: how many times the take's own band envelopes
    are measured and corrected towards the model's (bands overlap, and what leaks from random neighbours moves an
    envelope by several dB; one pass removes most of that): 1, or 2 for a model made by morph() or member(), whose
    bands were not measured together. tail, lead: what the stored envelope cannot hold, taken out (tidy())."""
    if exact is None:
        exact = 2 if 'tg' in model else 1
    ripple = np.zeros(NB)
    if vary > 0:
        length = length * float(np.exp(0.05 * vary * r.standard_normal()))
        pitch = pitch + 0.3 * vary * float(r.standard_normal())
        bright = bright + 0.4 * vary * float(r.standard_normal())
        hard = hard + 0.3 * vary * float(r.standard_normal())
        ripple = 1.5 * vary * ndimage.gaussian_filter1d(r.standard_normal(NB + 16), 2.5)[8:-8] * 3.0
    q = 2.0 ** (pitch / 12.0) / size
    grid = _grid(model)
    gu, gs = _warp(model, grid, length * 2.0 ** (pitch / 12.0), ring)
    end = max(gu[-1], gs[-1]) if ring != 1.0 else gu[-1]
    m = _fast(end + 2 * PADB + 1)
    pos = np.arange(m, dtype=np.float64) - PADB
    top = float(model['top'])
    ton = (model['ton'] >> 1).astype(np.float64) / 7.0
    rings = ton.max(axis=1) > 0.5 if ring != 1.0 else np.zeros(NB, bool)
    Gg = _smooth(model, grid)
    grain = _grain(model, grid)
    if vary > 0:
        keep = 1.0 - min(float(vary), 1.0)
        calm = _calm(grid)
        fresh = model['rough'].astype(np.float64)[:, None] * r.standard_normal(grain.shape)
        grain = np.where(calm[None, :], keep * grain + np.sqrt(1.0 - keep * keep) * fresh, grain)
    Gg = Gg + grain
    G = np.stack([np.interp(pos, gu, v) for v in Gg])
    if rings.any():
        G[rings] = np.stack([np.interp(pos, gs, v) for v in Gg[rings]])
    silent = G < top + FLOOR_DB + 4.0
    octs = np.log2(CENTRES / 1000.0)
    G += (bright * octs + weight / (1.0 + (CENTRES / 150.0) ** 4) + ripple)[:, None]
    on = np.interp(np.asarray(model['on'], np.float64), grid, gu)
    fine = np.zeros(m, bool)
    bump = np.zeros(m)
    for p in on:
        i = int(round(p)) + PADB
        fine[max(i - PRE, 0):i + POST] = True
        e = np.exp(-np.maximum(pos - p - 0.25 * POST, 0.0) / (0.004 * FS)) * (pos >= p - PRE)
        bump = np.maximum(bump, e * (1.0 + (0.15 * vary * r.standard_normal() if vary > 0 else 0.0)))
    if hard != 0.0 and len(on):
        G += 6.0 * hard * (0.6 + 0.4 * np.clip(octs / 3.0, -1.0, 1.0))[:, None] * bump[None, :]
    A = np.exp(G * (np.log(10.0) / 20.0))
    A[silent] = 0.0
    A[:, (pos < gu[0]) | (pos > end)] = 0.0

    stereo = 'ild' in model
    bank = _bank(m)
    kc = np.array([v[3] for v in bank], np.float64)
    ap = np.asarray(model['ap'], np.float64)
    frq, pha = _offsets(model), model['pha'].astype(np.float64)
    at_on = np.isin(ap, np.asarray(model['on'], np.float64))
    reach = np.clip(1.5 * LAG, 4.0, 60.0)
    kap = np.zeros((NB, m))
    coh = np.zeros((NB, m), np.complex128)
    live = ton.any(axis=1) & A.any(axis=1) & (len(ap) > 1)
    for sel, gg in ((live & ~rings, gu), (live & rings, gs)):
        if sel.any():
            pa = np.interp(ap, grid, gg)
            kap[sel] = _weight(pos, pa, ton[sel], at_on, reach[sel])
            beta = _phase(pos, pa, CENTRES[sel, None] + frq[sel], ton[sel] > 0, (model['ton'][sel] & 1) > 0,
                          pha[sel], kc[sel], m)
            coh[sel] = A[sel] * kap[sel] * np.exp(1j * beta)
    coh = coh[live].astype(np.complex64)
    full_a = A.astype(np.float32)
    A = (A * np.sqrt(1.0 - kap * kap)).astype(np.float32)
    heard = A.any(axis=1)
    nz = _carriers(r, bank, m, fine, heard)
    if stereo:
        icc = model['icc'].astype(np.float64)
        if 'ipd' in model:
            # the channels of a band correlate (at no lag) by the real part of their coherence: a mode shifted a
            # quarter turn between the channels is fully coherent and not correlated at all (a model made before
            # analyze() kept the phase has the coherence with the sign of its real part, a coarser stand-in)
            icc = np.abs(icc) * np.cos(model['ipd'].astype(np.float64) * (np.pi / 128))
        c = np.abs(icc)[:, None]
        ppos = np.interp(np.arange(len(model['pan'])) * float(PANHOP) - PADB, grid, gu)
        ratio = np.exp((model['ild'].astype(np.float64)[:, None]
                        + np.interp(pos, ppos, model['pan'].astype(np.float64))[None, :]) * (np.log(10.0) / 10.0))
        gl = np.sqrt(2.0 * ratio / (1.0 + ratio)).astype(np.float32)
        gr = np.sqrt(2.0 / (1.0 + ratio)).astype(np.float32)
        sg = np.where(icc < 0, -1.0, 1.0).astype(np.float32)[:, None]
        n2 = _carriers(r, bank, m, fine, heard & (c[:, 0] < 0.97)) * np.sqrt(0.5 * (1.0 - c)).astype(np.float32)
        nz = nz * np.sqrt(0.5 * (1.0 + c)).astype(np.float32)
        # a coherent carrier is one narrow-band signal: the right channel gets it turned by the phase whose cosine is
        # that correlation, so a ringing band correlates as the recording's did (not fully in or out of phase)
        turn = np.exp(-1j * np.arccos(np.clip(icc, -1.0, 1.0)))[:, None]
        parts = [(gl * full_a, gl * A * (nz + n2), gl[live] * coh),
                 (gr * full_a, sg * gr * A * (nz - n2), (gr * turn)[live].astype(np.complex64) * coh)]
    else:
        parts = [(full_a, A * nz, coh)]
    Y = np.zeros((len(parts), DEC * m // 2 + 1), np.complex128)
    row = np.cumsum(live) - 1
    S = np.zeros((NB, m), np.complex64)
    floor = (10 ** ((top + FLOOR_DB) / 20.0)) ** 2
    for ch, (want, sn, sc) in enumerate(parts):
        N, C = sfft.fft(sn, axis=1), sfft.fft(sc, axis=1)
        B = [N[b, v[4]] + C[row[b], v[4]] * v[2] if live[b] else N[b, v[4]] for b, v in enumerate(bank)]
        for _ in range(int(exact)):
            mix = np.zeros(Y.shape[1], np.complex64)
            for b, v in enumerate(bank):
                mix[v[0]:v[1]] += B[b]
            for b, v in enumerate(bank):
                S[b, v[4]] = mix[v[0]:v[1]] * v[2]
            w = sfft.ifft(S, axis=1)
            got = w.real ** 2 + w.imag ** 2
            slow = _tri(got)
            slow[:, fine] = got[:, fine]
            slow = ((want * want + floor) / (slow + floor)).clip(0.1, 10.0) ** 0.3
            S[:] = 0.0
            for b, v in enumerate(bank):
                S[b, v[4]] = B[b]
            F = sfft.fft(sfft.ifft(S, axis=1) * slow.astype(np.float32), axis=1)
            B = [F[b, v[4]] for b, v in enumerate(bank)]
            S[:] = 0.0
        for b, v in enumerate(bank):
            Y[ch, v[0]:v[1]] += B[b]
    y = sfft.irfft(Y * (DEC / 2.0), DEC * m, axis=1).T
    last = int(model['n']) / DEC
    n = int(round((np.interp(last, grid, gs if rings.any() else gu) + max(last - grid[-1], 0.0)) * DEC))
    y = y[PADB * DEC:PADB * DEC + max(n, DEC)]
    if not stereo:
        y = np.repeat(y, 2, axis=1)
    if abs(q - 1.0) > 1e-6:
        y = signal.resample(y, max(16, int(round(len(y) / q))), axis=0)
    if tail is not None or lead is not None:
        gx = gs if rings.any() else gu
        y = tidy(y, lambda s: np.interp(s * FS, grid, gx) / (FS * q),
                 lambda s: np.interp(s * FS * q, gx, grid) / FS, tail=tail, lead=lead)
    y = dsp.fade(y, 0.0001, 0.004)          # five samples: a recording cut at its onset keeps its onset
    peak = np.abs(y).max()
    if peak > 0.98:                         # a transform or a generated model may ask for more than full scale
        y *= 0.98 / peak
    return finish(y) if full else y


def finish(y, lufs=-11.0, weight=3.0, presence=2.5, air=3.5, drive=7.0):
    """the finish of a library sound: weight (dB around 85 Hz), presence (dB around 3.5 kHz), air (dB shelf above
    9 kHz), tape saturation in parallel (drive dB into the curve, half wet: density and a little harmonic body),
    then the loudest 400 ms brought to `lufs` with the peaks limited to -1 dBFS. Louder and fuller than the plain
    take by design; the plain take is what was measured."""
    from . import master
    y = np.asarray(y, np.float64)
    y = y * (0.5 / (np.abs(y).max() + 1e-12))
    y = dsp.peq(y, 85.0, weight, 0.7)
    y = dsp.peq(y, 3500.0, presence, 0.6)
    y = fx._shelf_filter(y, 9000.0, air)
    y = fx.saturate(y, drive_db=drive, mode='tape', mix=0.5, level='peak')
    y = y * 10 ** ((lufs - master.loudness(y)) / 20.0)
    return dsp.fade(fx.limit(y, ceiling_db=-1.0), 0.0004, 0.004)


# ------------------------------------------------------------------------------------------------ morph and classes
def _canon(model):
    """a model on the fixed grid on which models of one kind line up: CA frames from its start to 3 ms before its
    main event (the first onset within 10 dB of the strongest, or the loudest moment of a sound without onsets),
    PRE + POST frames at full resolution around the event, CD frames from 9 ms after it to the end (denser early).
    The two lengths are kept (log baseband samples). Carrier data every fourth frame, and `pha`: the phase (in
    turns) each band's oscillator had at the event: how a real onset is dispersed over the bands."""
    G, grid = _env(model), _grid(model)
    lv = np.sum(10 ** (G / 10), axis=0)
    on = np.asarray(model['on'], np.float64)
    if len(on):
        after = np.array([lv[(grid >= p) & (grid < p + POST)].sum() for p in on])
        p0 = float(on[int(np.argmax(after >= 0.1 * after.max()))])
    else:
        p0 = float(grid[int(np.argmax(ndimage.uniform_filter1d(lv, 5)))])
    start = max(grid[0], 0.0)
    pre, post = max(p0 - PRE - start, 1.0), max(grid[-1] - p0 - POST, 1.0)
    pos = np.concatenate([start + pre * (np.arange(CA) + 0.5) / CA, p0 - PRE + np.arange(PRE), p0 + np.arange(POST),
                          p0 + POST + post * ((np.arange(CD) + 1.0) / CD) ** 1.5])
    ap, at = np.asarray(model['ap'], np.float64), pos[::4]
    ton = np.stack([np.interp(at, ap, v) for v in (model['ton'] >> 1).astype(np.float64) / 7.0])
    wf = np.stack([np.interp(at, ap, v) for v in _offsets(model) * (model['ton'] > 1)])
    live = np.stack([np.interp(at, ap, v) for v in (model['ton'] > 1).astype(np.float64)])
    c = {'env': np.stack([np.interp(pos, grid, g) for g in G]) - float(model['top']), 'ton': ton,
         'frq': wf / np.maximum(live, 1e-3) * (live > 0.05), 'len': np.log([pre, post]),
         'top': float(model['top']), 'onset': float(len(on) > 0)}
    turns = np.concatenate([np.zeros((NB, 1)), np.cumsum((CENTRES[:, None] + _offsets(model))[:, 1:]
                                                         * (np.diff(ap) / FS)[None, :], axis=1)], axis=1)
    held = np.maximum.accumulate(np.where((model['ton'] & 1) > 0, np.arange(len(ap))[None, :], 0), axis=1)
    turns = turns + np.take_along_axis(model['pha'] / 256.0 - turns, held, axis=1)
    c['pha'] = turns[:, int(np.argmin(np.abs(ap - p0)))] % 1.0
    if 'ild' in model:
        c.update(ild=model['ild'].astype(np.float64), icc=model['icc'].astype(np.float64))
    return c


def _uncanon(c):
    """a model from its canonical form. Neighbouring bands that hold one partial are given one frequency, and every
    band's oscillator passes the main event with the phase in c['pha'] (a member's: an onset with all phases zero is
    the perfect click no object makes) and runs from there."""
    pre, post = np.exp(np.asarray(c['len'], np.float64))
    pos = np.concatenate([pre * (np.arange(CA) + 0.5) / CA, pre + np.arange(PRE), pre + PRE + np.arange(POST),
                          pre + PRE + POST + post * ((np.arange(CD) + 1.0) / CD) ** 1.5])
    pos = pos.astype(np.float32).astype(np.float64)          # as the model will hold them
    top = float(c['top'])
    G = np.clip(np.asarray(c['env'], np.float64), FLOOR_DB, 0.0) + top
    ton, frq = np.clip(np.asarray(c['ton'], np.float64), 0.0, 1.0), np.array(c['frq'], np.float64)
    ton = np.round(np.where(ton < 0.25, 0.0, ton) * 7.0) / 7.0
    ap = pos[::4]
    F = CENTRES[:, None] + frq
    for b in range(NB - 1):
        same = (ton[b] > 0.2) & (ton[b + 1] > 0.2) & (np.abs(F[b] - F[b + 1]) < 0.3 * WIDTH[b])
        both = (ton[b] * F[b] + ton[b + 1] * F[b + 1]) / np.maximum(ton[b] + ton[b + 1], 1e-9)
        F[b], F[b + 1] = np.where(same, both, F[b]), np.where(same, both, F[b + 1])
    F = CENTRES[:, None] + np.round((F - CENTRES[:, None]) / (WIDTH[:, None] / 100.0)).clip(-127, 127) * (
        WIDTH[:, None] / 100.0)
    a0 = (CA + PRE) // 4
    cyc = np.concatenate([np.zeros((NB, 1)), np.cumsum(F[:, 1:] * np.diff(ap)[None, :] / FS, axis=1)], axis=1)
    cyc = cyc - cyc[:, a0:a0 + 1] + np.asarray(c.get('pha', np.zeros(NB)), np.float64)[:, None]
    live = ton > 0
    entry = live & ~np.concatenate([np.zeros((NB, 1), bool), live[:, :-1]], axis=1)
    used = live | np.concatenate([np.zeros((NB, 1), bool), live[:, :-1]], axis=1)
    model = {'n': np.int32((pos[-1] + COARSE) * DEC), 'tg': pos.astype(np.float32),
             'on': np.array([pre + PRE] if c.get('onset', 0.0) >= 0.5 else [], np.float32),
             'ap': ap.astype(np.float32), 'ton': _coded(ton, entry),
             'frq': (np.round((F - CENTRES[:, None]) / (WIDTH[:, None] / 100.0)) * used).astype(np.int8),
             'pha': (np.round(((cyc + 0.5) % 1.0 - 0.5) * 256.0).clip(-128, 127) * entry).astype(np.int8)}
    model.update(_pack(G, top, pos))
    if 'ild' in c:
        model.update(ild=np.asarray(c['ild'], np.float16), icc=np.asarray(c['icc'], np.float16),
                     pan=np.zeros(2, np.float16))
    return model


def _blend(cs, wts):
    """the weighted mean of canonical forms (weights sum to one): levels and lengths in the log domain, carrier
    frequencies weighted by how coherent each member is there"""
    wts = np.asarray(wts, np.float64)
    out = {k: sum(w * np.asarray(c[k], np.float64) for w, c in zip(wts, cs))
           for k in ('env', 'ton', 'len', 'top', 'onset')}
    out['pha'] = cs[int(np.argmax(np.abs(wts)))]['pha']
    tw = sum(np.abs(w) * c['ton'] for w, c in zip(wts, cs))
    out['frq'] = sum(np.abs(w) * c['ton'] * c['frq'] for w, c in zip(wts, cs)) / np.maximum(tw, 1e-6)
    if all('ild' in c for c in cs):
        out['ild'] = sum(w * c['ild'] for w, c in zip(wts, cs))
        out['icc'] = np.clip(sum(w * c['icc'] for w, c in zip(wts, cs)), -1.0, 1.0)
    return out


def _detail(env):
    """the fine detail of a canonical envelope: what a blur over 2 bands and 3 frames removes"""
    return env - ndimage.gaussian_filter(env, (1.0, 1.5), mode='nearest')


def morph(a, b, t):
    """a model between two models of one kind: t = 0 gives a, 1 gives b. Both are put on the canonical grid (start,
    main event and end aligned), envelopes are interpolated in dB, lengths in the log domain, carriers by coherence;
    the fine detail that averaging flattens is scaled back to the mean of the two, the onset phases are those of the
    nearer one."""
    ca, cb = _canon(a), _canon(b)
    c = _blend([ca, cb], [1.0 - t, t])
    ga, gb, g = _detail(ca['env']), _detail(cb['env']), _detail(c['env'])
    want = (1.0 - t) * ga.std(axis=1) + t * gb.std(axis=1)
    c['env'] = c['env'] + g * (np.clip(want / (g.std(axis=1) + 1e-6), 1.0, 3.0) - 1.0)[:, None]
    return _uncanon(c)


def fit_class(models, keep=0.98):
    """the model of a kind of sound from analysed members: the mean of their canonical forms and the principal
    components of the differences (as many as hold `keep` of the variance, at most members - 1), with the standard
    deviation of the members along each, the members' places in that space and their onset phases, and the mean
    detail per band. A plain dict of arrays."""
    cs = [_canon(m) for m in models]
    mean = _blend(cs, np.full(len(cs), 1.0 / len(cs)))
    X = np.stack([_vec(c) for c in cs])
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S ** 2 / max(len(cs) - 1, 1)
    k = max(1, min(len(cs) - 1, int(np.searchsorted(np.cumsum(var) / (var.sum() + 1e-30), keep) + 1)))
    out = {'mu': mu.astype(np.float32), 'pc': Vt[:k].astype(np.float16), 'sd': np.sqrt(var[:k]).astype(np.float32),
           'z': (U[:, :k] * S[:k] / np.sqrt(var[:k])).astype(np.float32),
           'grain': np.mean([_detail(c['env']).std(axis=1) for c in cs], axis=0).astype(np.float16),
           'pha': np.stack([c['pha'] for c in cs]).astype(np.float16), 'onset': np.float32(mean['onset'])}
    if 'ild' in mean:
        out.update(ild=mean['ild'].astype(np.float16), icc=mean['icc'].astype(np.float16))
    return out


TC = CA + PRE + POST + CD                     # frames of a canonical form
TA = (TC + 3) // 4                            # its carrier anchors
SPLIT = np.cumsum([NB * TC, NB * TA, NB * TA, 2])


def _vec(c):
    """a canonical form as one vector: envelope (dB), coherence and carrier offset (in band steps, both scaled to
    count like 20 dB), the two log lengths (scaled likewise) and the level"""
    return np.concatenate([c['env'].ravel(), 20.0 * c['ton'].ravel(),
                           20.0 * (c['ton'] * c['frq'] / WIDTH[:, None]).ravel(), 20.0 * np.asarray(c['len']),
                           [c['top']]])


def member(cm, r, spread=0.7):
    """a new member of a class: a model at a random point of the class's space (spread = standard deviations of the
    members along each component), its fine detail scaled back to the class's, its onset phases those of the
    nearest member"""
    z = spread * r.standard_normal(len(cm['sd'])) * cm['sd'].astype(np.float64)
    v = cm['mu'].astype(np.float64) + z @ cm['pc'].astype(np.float64)
    i, j, k, l = SPLIT
    env, ton = v[:i].reshape(NB, TC), np.clip(v[i:j].reshape(NB, TA) / 20.0, 0.0, 1.0)
    g = _detail(env)
    env = env + g * (np.clip(cm['grain'].astype(np.float64) / (g.std(axis=1) + 1e-6), 1.0, 3.0) - 1.0)[:, None]
    c = {'env': env, 'ton': ton, 'frq': v[j:k].reshape(NB, TA) / 20.0 / np.maximum(ton, 0.05) * WIDTH[:, None],
         'len': v[k:l] / 20.0, 'top': float(v[l]), 'onset': float(cm['onset']),
         'pha': cm['pha'][int(np.argmin(np.sum((cm['z'] - z / cm['sd'].astype(np.float64)) ** 2, axis=1)))]}
    if 'ild' in cm:
        c.update(ild=cm['ild'].astype(np.float64), icc=cm['icc'].astype(np.float64))
    return _uncanon(c)


def sample(cm, r, spread=0.7, **transform):
    """a take of a new member of a class (see member and render)"""
    return render(member(cm, r, spread), r, **transform)


# ------------------------------------------------------------------------------------------------ fidelity
def envelopes(x, sr=SR):
    """what the fidelity measures look at: the band powers of a signal at FS, (NB, m), PADB samples of silence first
    (stereo: the mean power of the channels)"""
    ws = _bands(_prep(x, sr))
    return sum(np.abs(w) ** 2 for w in ws) / len(ws)


def _rise(P):
    """10-90 % rise time in ms of the loudest onset of the broadband amplitude envelope"""
    e = np.sqrt(P[CENTRES > 300.0].sum(axis=0))
    i = int(np.argmax(e))
    under = np.nonzero(e[:i + 1] < 0.1 * e[i])[0]
    a = int(under[-1]) if len(under) else 0
    over = a + int(np.argmax(e[a:i + 1] >= 0.9 * e[i]))
    return (over - a) * 1000.0 / FS


def _modulation(P):
    """per band the power of the envelope's modulation in octaves around 4 .. 128 Hz, dB re the mean level: (NB, 6)"""
    e = np.sqrt(_tri(P)[:, ::COARSE])
    e = e / (e.mean(axis=1, keepdims=True) + 1e-30)
    F = np.abs(np.fft.rfft((e - 1.0) * np.hanning(e.shape[1])[None, :], axis=1)) ** 2 / e.shape[1]
    f = np.fft.rfftfreq(e.shape[1], COARSE / FS)
    return np.stack([10 * np.log10(F[:, (f >= c / 1.414) & (f < c * 1.414)].sum(axis=1) + 1e-6)
                     for c in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)], axis=1)


def fidelity(x, y, sr=SR, align=False):
    """how far a take y is from a recording x where hearing would notice (both at sr, mono or stereo), after the
    levels are matched: env_db = dB RMS between the band envelopes at 2 ms over the cells that are heard (within
    40 dB of the loudest and within 25 dB of their band's loudest moment in the 6 ms around them); shape_db = the
    same after smoothing over one band and 5 ms (the structure without the grain); rise_ms = the 10-90 % rise times
    of the main onset (x, y); mod_db = dB RMS between the modulation spectra of the audible bands; level_db = y's
    level against x's before matching. align: slide y by up to 60 ms to the best match of the broadband envelopes
    first (for a take whose timing is not the recording's)."""
    Px, Py = envelopes(x, sr), envelopes(y, sr)
    m = max(Px.shape[1], Py.shape[1])
    Px, Py = [np.pad(p, ((0, 0), (0, m - p.shape[1]))) for p in (Px, Py)]
    if align:
        ex, ey = np.sqrt(Px.sum(axis=0)), np.sqrt(Py.sum(axis=0))
        lags = np.arange(-240, 241)
        best = lags[int(np.argmax([np.sum(ex * np.roll(ey, k)) for k in lags]))]
        Py = np.roll(Py, best, axis=1)
    level = 10 * np.log10((Py.sum() + 1e-30) / (Px.sum() + 1e-30))
    Py = Py * (Px.sum() / (Py.sum() + 1e-30))
    Gx, Gy = [10 * np.log10(_tri(p)[:, ::COARSE] + 1e-30) for p in (Px, Py)]
    top = Gx.max()
    both = np.maximum(Gx, Gy)
    cells = (both > top - 40.0) & (both > ndimage.maximum_filter1d(both, 7, axis=1) - 25.0)
    d = np.maximum(Gx, top - 50.0) - np.maximum(Gy, top - 50.0)
    heard = 10 * np.log10(Px.sum(axis=1) + 1e-30) > 10 * np.log10(Px.sum(axis=1).max() + 1e-30) - 30.0
    ds = ndimage.gaussian_filter(d, (1.0, 2.5), mode='nearest')
    return {'env_db': float(np.sqrt(np.mean(d[cells] ** 2))), 'shape_db': float(np.sqrt(np.mean(ds[cells] ** 2))),
            'rise_ms': (_rise(Px), _rise(Py)),
            'mod_db': float(np.sqrt(np.mean((_modulation(Px) - _modulation(Py))[heard] ** 2))),
            'level_db': float(level)}


def partials(x, sr=SR, count=6, floor_db=-30.0):
    """the strongest partials of a sound: [(Hz, dB re the strongest, seconds to fall 60 dB), ...] by frequency.
    Peaks of the whole sound's spectrum (interpolated), each one's decay from the slope of its narrow-band envelope
    after its maximum."""
    x = _prep(x, sr).mean(axis=1)
    n = sfft.next_fast_len(4 * len(x), real=True)
    X = sfft.rfft(x * np.hanning(len(x)), n)
    mag = 20 * np.log10(np.abs(X) + 1e-12)
    width = max(3, int(30.0 * n / SR))
    pk = np.nonzero((mag == ndimage.maximum_filter1d(mag, 2 * width + 1)) & (mag > mag.max() + floor_db))[0]
    pk = pk[(pk > 1) & (pk < len(mag) - 2)]
    pk = np.sort(pk[np.argsort(mag[pk])[::-1][:count]])
    out = []
    full = sfft.rfft(x, n)
    for k in pk:
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        f = (k + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)) * SR / n
        half = max(25.0, 0.02 * f)
        band = np.zeros(len(full), complex)
        sel = np.abs(np.arange(len(full)) * SR / n - f) < half
        band[sel] = full[sel] * np.cos(0.5 * np.pi * (np.arange(len(full))[sel] * SR / n - f) / half) ** 2
        z = np.zeros(n, complex)
        z[:len(band)] = 2 * band
        e = 20 * np.log10(np.abs(sfft.ifft(z))[:len(x)] + 1e-12)
        i = int(np.argmax(e))
        tail = np.nonzero(e[i:] > e[i] - 25.0)[0]
        j = i + (int(tail[-1]) if len(tail) else 1)
        slope = np.polyfit(np.arange(i, j + 1) / SR, e[i:j + 1], 1)[0] if j > i + 32 else -1e6
        out.append((float(f), float(b - mag.max()), float(-60.0 / min(slope, -1e-6))))
    return out
