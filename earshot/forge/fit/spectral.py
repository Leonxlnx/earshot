"""Spectral fitter: a real noisy / airy / textured recording -> a small JSON model -> fresh re-renders of it.

    from earshot.forge.fit import spectral
    m = spectral.fit('swish.wav')                  # analysis-by-synthesis; m is a plain JSON-serialisable dict
    y = spectral.render(m, seed=3)                 # (n, 2) float32 at 48 kHz: same sound, fresh noise
    y = spectral.render(m, seed=4, length=1.4, speed=0.8, sweep=1.3, tilt=-1.5, roughness=0.5, width=1.2)
    print(spectral.to_code(m))                     # the model as readable Python that re-renders it
    print(spectral.summary(m))                     # 'swish.wav: 0.42 s noise gesture, peak at 0.11 s, ...'
    q = spectral.compare(x, y)                     # mel dB (256/1024/4096), envelope corr, attack, tone, (CLAP)
    r = spectral.fit_report('swish.wav', clap=True)   # fit + two renders + compare(orig, r1) and the
                                                      # render-vs-render floor compare(r1, r2)

For whooshes, swishes, cloth / paper swipes, risers, downlifters, wind, shimmer beds and glitch textures: sounds that
are noise (or dense grains) shaped by a time-varying spectral envelope. The model (kind 'spectral_noise'):

  env      the GESTURE: keyframes in time (non-uniform, chosen greedily where linear interpolation in dB misses the
           gesture by more than the tolerance, as a weighted mean over the cells that are loud in each frame), each
           a 48-band spectrum (20 Hz - 20 kHz, spaced on ln(1 + f / 80 Hz): ~1/3 octave in the lows, ~1/6 octave
           in the highs) of power density in dB re a full-scale white noise. Analysis: multi-resolution STFT (512 -
           4096 per band, ~2 / bandwidth of window) on a 5 ms grid; 25 ms smoothing that never crosses the onset
           frame (the attack stays sharp), widened while the level still wanders > 2 dB around a running quadratic
           (crackle, flutter: those are texture, the tolerance rises to that wander too); a weighted low-rank
           (SVD) denoise whose rank stops where the residual reaches what a noise band's estimate wanders by
           (measured on white noise through the same analysis; quiet cells keep their measured level); then the
           gains are corrected by analysing renders of the model with the same analysis (analysis-by-synthesis, 3
           iterations x 2 seeds: band overlap, window smear and the low bands' slow filters are paid back).
  stereo   per band group (0-300-1000-2500-5000-10000-24000 Hz) and keyframe: inter-channel level difference (dB)
           and inter-channel correlation; rendered from two independent noise sources mixed per band (L = cA + sB,
           R = cA - sB, c^2 - s^2 = correlation) with ILD gains, so width and pan trajectories come back. None for a
           mono source (both channels then carry the same render).
  texture  what the smooth gesture does not hold, as statistics fitted by renders (mono, 3 seeds):
           grains  crackle (cloth, paper, glitch dust): the excitation becomes a Poisson click train (log-normal
                   amplitudes, spread 1 neper) of `density` clicks per second instead of Gaussian noise. Detected by
                   excess fast (>= 128 Hz) log-amplitude variance (clicks put variance at every rate, flutter does
                   not); density fitted to the std of the log analytic envelope around the gesture (Gaussian noise
                   0.64 nepers; it is monotone in density and stable across seeds, where kurtosis is ruled by the
                   largest click)
           am      flutter: the 6-64 Hz log-amplitude variance beyond what Gaussian noise (and the grains) already
                   make, kept only where both halves of the active span have it; a half-octave rate spectrum, a depth
                   per band group (std of log amplitude, nepers, capped at 0.6) and the coherence across groups
           stats   the measured statistics behind those decisions (for reports)
  ridges   tonal ridges (whistle / aeolian tones inside a whoosh): spectral peaks > 8 dB over the local floor on
           both sides, tracked for >= 50 ms, as keyframes of frequency (Hz) and amplitude (dB re full scale) with a
           small frequency jitter; their energy is taken out of the noise envelope before it is fitted.
  fine     (opt-in, fit(fine=True)) a static fine spectral curve every 1/24 octave above ~300 Hz: resonances the
           bands cannot hold. Helps long steady sounds (the riser: CLAP 0.82 -> 0.88), costs a little on short
           ones (estimation noise), hence off by default.
  fit      what the fit chose: gesture_ms (smoothing), rank, tol_db, fluct_db.

Rendering (`render`): each call draws NEW noise (seeded, deterministic per seed), so every render is a fresh but
statistically identical sound; knobs: length (time-stretch of the gesture, texture rates kept), speed (movement
speed: shorter, higher by SPEED_OCT octaves per doubling, faster flutter), sweep (scales the spectral centroid's
excursion around its mean), shift (octaves), tilt (dB / octave about 1 kHz), roughness (texture depth; 0 = smooth
Gaussian), width (side / pan amount), ridges (tonal ridge gain), gain_db. Level: the gain that puts the EXPECTED
loudest 50 ms RMS (computed from the gesture: band envelopes, their overlaps, the ridges) on the original's
(+ gain_db), times the take's own correction to its expected K-weighted energy (BS.1770 weighting, the loudness a
meter reads). A noise band holds few independent bins over a short gesture, so a take's loudness wanders with the
seed; levelled by its own loudest 50 ms, as before 0.3, takes of one model came out 2-7 LU apart (a cloth whoosh),
now within about 0.5 LU. A scalar per take: the fit-quality numbers, which level both sounds, do not move
(steady=False keeps the old levelling: a hybrid's air layer, whose post stage was measured on it). Sample peak capped
at -0.5 dBFS (limit=False: not capped; the fit engine renders so, as the cap would make a high-crest take quieter
than the others).

Judging a fit: a fresh-noise render can never match the samples, and even two renders of one model differ by a few
dB cell by cell (a noise band's level estimate fluctuates). So `fit_report` also gives the floor: compare(render,
render') with another seed. A fit is as good as this representation allows when compare(orig, render) is near that
floor; the gap is what the model misses.
Pure numpy / scipy. Deterministic: everything is a function of (model, seed, knobs).
"""
import json
import os

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import convolve1d, median_filter, uniform_filter

KIND = 'spectral_noise'
VERSION = 1
SR = 48000
HOP = 240                                   # 5 ms analysis / control grid
N_BANDS = 48
F_LO, F_HI = 20.0, 20000.0
WARP_HZ = 80.0
GROUPS = (0.0, 300.0, 1000.0, 2500.0, 5000.0, 10000.0, 24000.0)
RATES = 2.0 ** np.arange(0.0, 9.5, 0.5)     # modulation rate band centres, 1 .. 512 Hz (half-octave)
AM_MAX_RATE = 64.0                          # AM statistics up to here; faster structure is the grains' business
AM_MIN_RATE = 6.0                           # slower than this belongs to the gesture (key frames)
AM_MAX_DEPTH = 0.6                          # nepers (~5 dB std of log amplitude): a safety cap
SPEED_OCT = 0.5                             # octaves up per doubling of `speed`
NFFTS = (512, 1024, 2048, 4096)
NFFT_CYCLES = 2.0                           # analysis window ~ this many / bandwidth per band
FLOOR_DB = 90.0                             # envelope floor below its loudest cell
SPREAD = 1.0                                # grain amplitude spread (log-normal sigma, nepers)


def _workers():
    try:
        return max(1, int(os.environ.get('EARSHOT_THREADS', '2')))
    except ValueError:
        return 2


# ----------------------------------------------------------------------------------------------- bands


def warp(f):
    """the band axis: ln(1 + f / 80 Hz). About 1/3 octave per band below 300 Hz (where the ERB scale is coarser
    than the 1/3-octave balance the forge measures), ~1/6 octave per band above 2 kHz (finer than 1/3 octave, near
    the ERB)."""
    return np.log1p(np.asarray(f, np.float64) / WARP_HZ)


def unwarp(e):
    return np.expm1(np.asarray(e, np.float64)) * WARP_HZ


def centers(n=N_BANDS, f_lo=F_LO, f_hi=F_HI):
    """band centre frequencies (Hz), uniformly spaced on the band axis (warp)"""
    return unwarp(np.linspace(warp(f_lo), warp(f_hi), int(n)))


def _wband(e, ec, b):
    """raised-cosine weight of band b on the band axis; the bands sum to exactly 1 above the first centre"""
    w = np.zeros(len(e))
    B = len(ec)
    if b == 0:
        lo = ec[0] - (ec[1] - ec[0])                  # roll off below the first centre (no flat shelf to DC)
        m = (e >= lo) & (e < ec[0])
        w[m] = np.sin(0.5 * np.pi * (e[m] - lo) / (ec[0] - lo)) ** 2
    else:
        lo = ec[b - 1]
        m = (e >= lo) & (e < ec[b])
        w[m] = np.sin(0.5 * np.pi * (e[m] - lo) / (ec[b] - lo)) ** 2
    if b == B - 1:
        w[e >= ec[-1]] = 1.0
    else:
        hi = ec[b + 1]
        m = (e >= ec[b]) & (e < hi)
        w[m] = np.cos(0.5 * np.pi * (e[m] - ec[b]) / (hi - ec[b])) ** 2
    return w


def weights(freqs, cents):
    """(B, F) partition-of-unity band weights at the given frequencies"""
    e = warp(freqs)
    ec = warp(cents)
    return np.stack([_wband(e, ec, b) for b in range(len(ec))])


_BW = {}


def bandwidths(cents):
    """effective bandwidth (Hz) of each band: the integral of its weight"""
    key = tuple(np.round(cents, 6))
    if key not in _BW:
        f = np.linspace(0, SR / 2, 96001)
        _BW[key] = weights(f, cents).sum(1) * (f[1] - f[0])
    return _BW[key].copy()


def _band_nfft(cents):
    bw = bandwidths(cents)
    n = 2.0 ** np.round(np.log2(NFFT_CYCLES * SR / bw))
    return np.clip(n, NFFTS[0], NFFTS[-1]).astype(int)


def _group_of(cents):
    return np.clip(np.searchsorted(GROUPS, cents, side='right') - 1, 0, len(GROUPS) - 2)


def _group_centres():
    g = np.array(GROUPS, np.float64)
    c = np.sqrt(np.maximum(g[:-1], 50.0) * g[1:])
    c[0] = 150.0
    return c


def _group_blend(cents):
    """per band: (g0, g1, a) so a band's group parameter = (1-a) p[g0] + a p[g1] (log-frequency interpolation)"""
    gc = np.log2(_group_centres())
    lf = np.log2(np.maximum(cents, 1.0))
    out = []
    for v in lf:
        if v <= gc[0]:
            out.append((0, 0, 0.0))
        elif v >= gc[-1]:
            out.append((len(gc) - 1, len(gc) - 1, 0.0))
        else:
            i = int(np.searchsorted(gc, v) - 1)
            out.append((i, i + 1, float((v - gc[i]) / (gc[i + 1] - gc[i]))))
    return out


# ------------------------------------------------------------------------------------------- analysis


def _load(x):
    if isinstance(x, str):
        from ... import audio
        x = audio.load(x)
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    return x[:, :2]


def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _stft(x, n_fft, T):
    """(T, n_fft/2+1) complex, centred frames every HOP, scaled so |X|^2 is power density (white var s2 -> s2)"""
    win = _hann(n_fft)
    pad = n_fft // 2
    need = (T - 1) * HOP + n_fft
    xp = np.zeros(need)
    seg = x[:max(0, min(len(x), need - pad))]
    xp[pad:pad + len(seg)] = seg
    fr = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::HOP][:T]
    X = sfft.rfft(fr * win, axis=1, workers=_workers())
    return X / np.sqrt((win ** 2).sum())


def n_frames(n):
    return int(n) // HOP + 1


def analyse(x, cents=None, stereo=True):
    """band power densities on the 5 ms grid: {'D': (B, T) mid density, 'P_lr': ..., 'stereo': group L/R/cross}"""
    cents = centers() if cents is None else cents
    x = np.asarray(x, np.float64)
    mono = x.ndim == 1
    T = n_frames(len(x))
    sizes = _band_nfft(cents)
    D = np.zeros((len(cents), T))
    st = None
    todo = sorted(set(sizes.tolist()) | ({1024} if (stereo and not mono) else set()))
    for N in todo:
        if mono:
            XL = _stft(x, N, T)
            P = np.abs(XL) ** 2
        else:
            XL = _stft(x[:, 0], N, T)
            XR = _stft(x[:, 1], N, T)
            PL, PR = np.abs(XL) ** 2, np.abs(XR) ** 2
            P = 0.5 * (PL + PR)
        f = np.fft.rfftfreq(N, 1 / SR)
        sel = np.nonzero(sizes == N)[0]
        if len(sel):
            Wf = weights(f, cents)[sel]
            Wn = Wf / np.maximum(Wf.sum(1, keepdims=True), 1e-12)
            D[sel] = (P @ Wn.T).T
        if N == 1024 and stereo and not mono:
            gi = np.clip(np.searchsorted(GROUPS, f, side='right') - 1, 0, len(GROUPS) - 2)
            G = len(GROUPS) - 1
            M = np.zeros((len(f), G))
            M[np.arange(len(f)), gi] = 1.0
            C = np.real(XL * np.conj(XR))
            st = {'PL': (PL @ M).T, 'PR': (PR @ M).T, 'C': (C @ M).T}
    return {'D': D, 'stereo': st, 'T': T}


def _smooth_t(D, k):
    k = int(k)
    if k <= 1:
        return D
    w = _hann(k + 1)[1:]
    w = w / w.sum()
    return convolve1d(D, w, axis=-1, mode='nearest')


def _db(p, floor=1e-20):
    return 10.0 * np.log10(np.maximum(p, floor))


# -------------------------------------------------------------------------------------------- keyframes


def _interp_rows(Y, keys):
    T = len(Y)
    k = np.asarray(keys)
    t = np.arange(T)
    i = np.clip(np.searchsorted(k, t, side='right') - 1, 0, len(k) - 2)
    a = (t - k[i]) / np.maximum(k[i + 1] - k[i], 1)
    return Y[k[i]] * (1 - a)[:, None] + Y[k[i + 1]] * a[:, None]


def _keyframes(Y, Wc, tol, must=(), max_keys=200):
    """greedy keyframe insertion: frames where linear interpolation between keys misses Y by > tol (weighted RMS)"""
    T = len(Y)
    if T <= 2:
        return list(range(T))
    keys = sorted(set([0, T - 1] + [int(m) for m in must if 0 <= m < T]))
    ws = Wc.sum(1)
    den = np.maximum(ws, 0.25 * ws.max()) + 1e-12      # weighted mean over the cells that matter in each frame
    while len(keys) < max_keys:
        Yh = _interp_rows(Y, keys)
        err = np.sqrt((Wc * (Yh - Y) ** 2).sum(1) / den)
        err[keys] = 0.0
        j = int(np.argmax(err))
        if err[j] <= tol:
            break
        keys = sorted(keys + [j])
    return keys


# ------------------------------------------------------------------------------------------ synthesis


def _gauss_process(m, fs, rates, spec, rng):
    """unit-variance Gaussian process of m samples at fs with the given (rate band -> variance) spectrum"""
    if m < 4 or spec is None or np.sum(spec) <= 0:
        return np.zeros(m)
    N = sfft.next_fast_len(max(m, 16))
    f = np.fft.rfftfreq(N, 1 / fs)
    lr = np.log2(np.maximum(f, 1e-3))
    s = np.asarray(spec, np.float64) / (np.asarray(rates) * (2 ** 0.25 - 2 ** -0.25))   # variance -> density
    dens = np.interp(lr, np.log2(rates), s, left=0.0, right=0.0)
    dens[f < rates[0] / 2 ** 0.25] = 0.0
    if dens.sum() <= 0:
        return np.zeros(m)
    Z = sfft.rfft(rng.standard_normal(N)) * np.sqrt(dens)
    z = sfft.irfft(Z, N)[:m]
    sd = z.std()
    return z / sd if sd > 0 else z


def _clicks(N, rate, rng, spread=SPREAD):
    """unit-variance Poisson click train: `rate` clicks / s, log-normal amplitudes with random signs"""
    k = int(rng.poisson(max(rate, 1e-3) * N / SR))
    c = np.zeros(N)
    if k == 0:
        return c
    pos = rng.integers(0, N, k)
    amp = np.exp(spread * rng.standard_normal(k)) * rng.choice((-1.0, 1.0), k)
    np.add.at(c, pos, amp)
    p = np.sqrt((c ** 2).mean())
    return c / p if p > 0 else c


def _k_gain2(f):
    """|K-weighting|^2 (ITU-R BS.1770, 48 kHz) at frequencies f (Hz)"""
    from scipy import signal
    from ..polish import K_SOS
    return np.abs(signal.sosfreqz(K_SOS, worN=np.asarray(f, np.float64), fs=SR)[1]) ** 2


def _synth(n, tk, env, cents, seed, stereo=None, texture=None, ridges=(), mono=False, rate_scale=1.0,
           rough=1.0, fine=None, fine_shift=0.0, power=False):
    """the renderer core. tk (K,) key times in s, env (K, B) density dB. Returns (n,) if mono else (n, 2).
    power=True returns (y, p, pk): the expected channel-mean power of the render, plain and K-weighted ((n,) each:
    the bands' envelopes with their neighbours' overlap, the ridges; flutter has E[mod^2] = 1), the same for every
    seed (render levels by them)"""
    n = int(n)
    ss = np.random.SeedSequence(abs(int(seed)) % (2 ** 32))
    rA, rB, rC, rM, rR = [np.random.default_rng(s) for s in ss.spawn(5)]
    N = sfft.next_fast_len(max(n, 64))
    ts = np.arange(n) / SR
    tk = np.asarray(tk, np.float64)
    env = np.asarray(env, np.float64)
    G = len(GROUPS) - 1
    wk = _workers()

    grains = (texture or {}).get('grains')
    eA = rA.standard_normal(N)
    if grains and rough > 0:
        dens = float(grains['density']) / max(rough, 1e-3) ** 2   # rougher = sparser clicks
        eA = _clicks(N, dens * rate_scale, rC)
    FA = sfft.rfft(eA, workers=wk)
    FB = None if (mono or stereo is None) else sfft.rfft(rB.standard_normal(N), workers=wk)
    f = np.fft.rfftfreq(N, 1 / SR)
    if fine:
        gf = _fine_gain(fine, f / 2 ** fine_shift)
        FA = FA * gf
        if FB is not None:
            FB = FB * gf
    ef = warp(f)
    ec = warp(cents)
    blend = _group_blend(cents)

    # per group control signals at the sample rate
    am = (texture or {}).get('am')
    if am and rough > 0 and max(am['depth']) > 0:
        n1 = int(np.ceil(n / SR * 1000)) + 2
        t1 = np.arange(n1) / 1000.0
        rates = np.asarray(am['rates'], np.float64) * rate_scale
        zc = _gauss_process(n1, 1000.0, rates, am['spec'], rM)
        coh = float(np.clip(am.get('coh', 1.0), 0, 1))
        mods = []
        for g in range(G):
            zi = _gauss_process(n1, 1000.0, rates, am['spec'], rM)
            z = np.sqrt(coh) * zc + np.sqrt(1 - coh) * zi
            sg = float(am['depth'][g]) * rough
            mods.append(np.interp(ts, t1, np.exp(sg * z - sg * sg)))
        mods = np.stack(mods)
    else:
        mods = None
    if stereo is not None and not mono:
        ild = np.asarray(stereo['ild'], np.float64)       # (K, G)
        icc = np.asarray(stereo['icc'], np.float64)
        th = 0.5 * np.arccos(np.clip(icc, -1, 1))
        r = 10 ** (ild / 10)
        par = np.stack([np.cos(th), np.sin(th), np.sqrt(2 * r / (1 + r)), np.sqrt(2 / (1 + r))])   # (4, K, G)
        ctl = np.stack([[np.interp(ts, tk, par[p, :, g]) for g in range(G)] for p in range(4)])   # (4, G, n)
    else:
        ctl = None

    L = np.zeros(n)
    R = None if mono else np.zeros(n)
    P = np.zeros(n) if power else None
    PK = np.zeros(n) if power else None
    K2 = _k_gain2(f) if power else None
    prev = None                              # (w, amp) of the band below: neighbours overlap (their cross power)
    for b in range(len(cents)):
        w = _wband(ef, ec, b)
        nz = np.nonzero(w)[0]
        if not len(nz):
            continue
        lo, hi = nz[0], nz[-1] + 1
        Xb = np.zeros(len(f), complex)
        Xb[lo:hi] = FA[lo:hi] * w[lo:hi]
        nA = sfft.irfft(Xb, N, workers=wk)[:n]
        amp = 10 ** (np.interp(ts, tk, env[:, b]) / 20)
        g0, g1, a = blend[b]
        if P is not None:                    # expected power: sum over bins of (sum_b amp_b w_b)^2 x 2 / N
            ww = w[lo:hi] ** 2
            P += amp * amp * (2.0 * float(np.sum(ww)) / N)
            PK += amp * amp * (2.0 * float(np.sum(ww * K2[lo:hi])) / N)
            if prev is not None:
                wx = prev[0][lo:hi] * w[lo:hi]
                P += 2.0 * (2.0 * float(np.sum(wx)) / N) * prev[1] * amp
                PK += 2.0 * (2.0 * float(np.sum(wx * K2[lo:hi])) / N) * prev[1] * amp
            prev = (w, amp)
        if mods is not None:
            amp = amp * ((1 - a) * mods[g0] + a * mods[g1])
        if mono:
            L += amp * nA
            continue
        if ctl is None:                      # a mono model renders the same signal on both channels
            L += amp * nA
            continue
        Xb[lo:hi] = FB[lo:hi] * w[lo:hi]
        nB = sfft.irfft(Xb, N, workers=wk)[:n]
        c, s, gl, gr = [(1 - a) * ctl[p, g0] + a * ctl[p, g1] for p in range(4)]
        L += amp * gl * (c * nA + s * nB)
        R += amp * gr * (c * nA - s * nB)

    for rd in ridges or ():
        rt = np.asarray(rd['t'], np.float64)
        fr = np.interp(ts, rt, np.asarray(rd['f'], np.float64))
        A = 10 ** (np.interp(ts, rt, np.asarray(rd['db'], np.float64)) / 20)
        A[(ts < rt[0]) | (ts > rt[-1])] = 0.0
        n1 = int(np.ceil(n / SR * 1000)) + 2
        z = _gauss_process(n1, 1000.0, RATES, np.where(RATES <= 16, 1.0, 0.0), rR)
        jit = float(rd.get('jitter', 0.002))
        fr = fr * (1 + jit * np.interp(ts, np.arange(n1) / 1000.0, z))
        ph = 2 * np.pi * np.cumsum(fr) / SR + rR.uniform(0, 2 * np.pi)
        y = A * np.sin(ph)
        # fade the ridge in / out over 5 ms at its ends
        e = np.clip(np.minimum(ts - rt[0], rt[-1] - ts) / 0.005, 0, 1)
        y = y * e
        if P is not None:
            P += 0.5 * (A * e) ** 2
            PK += 0.5 * (A * e) ** 2 * float(_k_gain2([max(float(np.median(rd['f'])), 1.0)])[0])
        if mono or ctl is None:
            L += y
        else:
            g = int(_group_of(np.array([np.median(rd['f'])]))[0])
            L += y * ctl[2, g]
            R += y * ctl[3, g]
    if mono:
        return (L, P, PK) if power else L
    if ctl is None:
        R = L.copy()
    y = np.stack([L, R], 1)
    return (y, P, PK) if power else y


def _take_gain(y, pk):
    """the gain that puts a take's K-weighted energy (channel mean) on its expected one"""
    from scipy import signal
    from ..polish import K_SOS
    y2 = np.asarray(y, np.float64)
    y2 = y2 if y2.ndim == 2 else y2[:, None]
    act = float(np.mean([np.sum(signal.sosfilt(K_SOS, y2[:, c]) ** 2) for c in range(y2.shape[1])]))
    want = float(np.sum(pk))
    return float(np.sqrt(want / act)) if act > 1e-300 and want > 0 else 1.0


def _peak_mean_db(p, win_s=0.05):
    """the loudest win_s mean of a power curve (dB)"""
    k = max(1, int(win_s * SR))
    if len(p) < k:
        return float(_db(np.mean(p)))
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(_db(((c[k:] - c[:-k]) / k).max()))


def _peak_rms_db(x, win_s=0.05):
    x = np.asarray(x, np.float64)
    p = (x ** 2).mean(1) if x.ndim == 2 else x ** 2
    k = max(1, int(win_s * SR))
    if len(p) < k:
        return float(_db(p.mean()))
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(_db(((c[k:] - c[:-k]) / k).max()))


# ------------------------------------------------------------------------------------------------- fit


def _stereo_traj(st, T, mono):
    G = len(GROUPS) - 1
    if mono or st is None:
        return np.zeros((T, G)), np.ones((T, G)), np.ones((T, G))
    PL, PR, C = [_smooth_t(st[k], 7) for k in ('PL', 'PR', 'C')]
    ild = np.clip(_db(PL + 1e-30) - _db(PR + 1e-30), -20, 20)
    icc = np.clip(C / np.sqrt(PL * PR + 1e-30), -1, 1)
    pw = PL + PR
    act = _db(pw) > _db(pw.max()) - 45
    # quiet cells take the group's power-weighted mean, so noise does not wander the image
    for g in range(G):
        wgt = pw[g] * act[g]
        if wgt.sum() <= 0:
            ild[g], icc[g] = 0.0, 1.0
            continue
        mi = (ild[g] * wgt).sum() / wgt.sum()
        mc = (icc[g] * wgt).sum() / wgt.sum()
        ild[g] = np.where(act[g], ild[g], mi)
        icc[g] = np.where(act[g], icc[g], mc)
    gw = pw.sum(1)
    gw = gw / max(gw.sum(), 1e-30)
    return ild.T, icc.T, np.broadcast_to(gw, (T, G))


def _find_ridges(xm, D, cents, max_ridges=4, min_prom=8.0, min_frames=10, f_lo=400.0, f_hi=16000.0):
    """tonal ridge tracks: list of {'k': frame idx, 'f': Hz, 'p': peak density, 'fl': floor density}"""
    N = 2048
    T = D.shape[1]
    X = _stft(xm, N, T)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(N, 1 / SR)
    Pd = _db(P + 1e-30)
    # local floor on BOTH sides (median of 8 bins starting 2 bins away; the higher side wins), so the edge of a
    # band-limited noise is not a peak; +1.6 dB: the median of an exponential is ln 2 under its mean
    med = median_filter(Pd, size=(1, 8), mode='nearest')
    fl = np.maximum(np.roll(med, 6, axis=1), np.roll(med, -6, axis=1)) + 1.6
    prom = Pd - fl
    lvl = _db(P.sum(1))
    act = lvl > lvl.max() - 50
    band = (f >= f_lo) & (f <= f_hi)
    tracks, live = [], []
    for k in range(T):
        if not act[k]:
            live = [tr for tr in live if k - tr['k'][-1] <= 2]
            continue
        row = prom[k]
        cand = np.nonzero(band[2:-2] & (row[2:-2] > min_prom) & (Pd[k, 2:-2] >= Pd[k, 1:-3])
                          & (Pd[k, 2:-2] >= Pd[k, 3:-1]) & (Pd[k, 2:-2] >= Pd[k, :-4]) & (Pd[k, 2:-2] >= Pd[k, 4:]))[0] + 2
        cand = sorted(cand, key=lambda i: -row[i])[:12]
        used = set()
        for i in cand:
            a, b0, c = Pd[k, i - 1], Pd[k, i], Pd[k, i + 1]
            den = a - 2 * b0 + c
            d = 0.5 * (a - c) / den if den < 0 else 0.0
            fi = (i + d) * SR / N
            pk = b0 - 0.25 * (a - c) * d
            best, bd = None, 0.04
            for j, tr in enumerate(live):
                if j in used:
                    continue
                dd = abs(np.log2(fi / tr['f'][-1]))
                if dd < bd:
                    best, bd = j, dd
            if best is None:
                tr = {'k': [], 'f': [], 'p': [], 'fl': []}
                live.append(tr)
                tracks.append(tr)
                best = len(live) - 1
            used.add(best)
            tr = live[best]
            tr['k'].append(k)
            tr['f'].append(fi)
            tr['p'].append(10 ** (pk / 10))
            tr['fl'].append(10 ** (fl[k, i] / 10))
        live = [tr for tr in live if k - tr['k'][-1] <= 2]
    good = []
    for tr in tracks:
        if len(tr['k']) < min_frames:
            continue
        pr = _db(np.array(tr['p'])) - _db(np.array(tr['fl']))
        if np.median(pr) < min_prom + 2:
            continue
        good.append(tr)
    good.sort(key=lambda tr: -float(np.sum(np.array(tr['p']) - np.array(tr['fl']))))
    return good[:max_ridges]


def _ridge_model(tr, n_fft=2048):
    k = np.array(tr['k'])
    fz = np.array(tr['f'])
    p = np.maximum(np.array(tr['p']) - np.array(tr['fl']), 1e-30)
    A = np.sqrt(6 * p / n_fft)          # Hann peak bin of a sinusoid of amplitude A: density A^2 N / 6
    adb = 20 * np.log10(A)
    t = k * HOP / SR
    Y = np.stack([np.log2(fz) / 0.006, adb / 1.5], 1)
    keys = _keyframes(Y, np.ones_like(Y), 1.0, max_keys=60)
    fs = _smooth_t(np.log2(fz)[None], 5)[0]
    jit = float(np.clip(np.std(np.log2(fz) - fs) * np.log(2), 0.0005, 0.02))
    return {'t': [round(float(t[i]), 4) for i in keys], 'f': [round(float(fz[i]), 1) for i in keys],
            'db': [round(float(adb[i]), 1) for i in keys], 'jitter': round(jit, 4)}, (k, fz, A)


def _ridge_remove(D, cents, k, fz, A):
    """subtract a sinusoid's contribution from the band densities (A^2 SR / (4 bw) spread by the band weights)"""
    bw = bandwidths(cents)
    W = weights(fz, cents)            # (B, len)
    inc = W * (A ** 2)[None] * SR / (4 * bw[:, None])
    D = D.copy()
    D[:, k] = np.maximum(D[:, k] - inc, 0.05 * D[:, k])
    return D


def _seg_smooth(D, k, on):
    """time smoothing that does not cross the onset frame (the attack stays sharp)"""
    if on <= 1 or on >= D.shape[1] - 1:
        return _smooth_t(D, k)
    return np.concatenate([_smooth_t(D[:, :on], k), _smooth_t(D[:, on:], k)], 1)


def _lowrank(Y, W, K, iters=10):
    """weighted rank-K approximation of Y (T, B): per-band weighted mean + K components (EM over the weights)"""
    Z = Y.copy()
    Yh = Y.copy()
    for _ in range(iters):
        Z = W * Y + (1 - W) * Yh
        mu = Z.mean(0)
        U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
        Yh = mu + (U[:, :K] * S[:K]) @ Vt[:K]
    return Yh


def _noise_sigma(n, cents, k, on, seed=12345):
    """per band std (dB) of the analysis of white noise of this length: what a noise band's estimate wanders"""
    wn = np.random.default_rng(seed).standard_normal(n)
    Dn = _db(_seg_smooth(analyse(wn, cents, stereo=False)['D'], k, on))
    T = Dn.shape[1]
    sig = np.zeros(len(cents))
    sizes = _band_nfft(cents)
    for b in range(len(cents)):
        e = int(sizes[b] // 2 // HOP)
        row = Dn[b, e:T - e] if T - 2 * e > 8 else Dn[b]
        sig[b] = row.std()
    return sig


def _wander(L, k, on):
    """RMS (dB) of a level curve around a running quadratic over ~4k frames, on its loud frames, onset excluded"""
    from scipy.signal import savgol_filter
    T = len(L)
    w = min(T - (1 - T % 2), 4 * k + 1)
    if w < 5:
        return 0.0
    Q = savgol_filter(L, w, 2, mode='interp')
    fa = (L > L.max() - 30) & (np.abs(np.arange(T) - on) > k)
    return float(np.sqrt(np.mean((L - Q)[fa] ** 2))) if fa.sum() > 4 else 0.0


FINE_F0 = 40.0
FINE_STEP = 1.0 / 24                        # octaves per point of the fine spectral curve


def _fine_gain(fine, f):
    """amplitude gain of the static fine spectral curve at frequencies f"""
    db = np.asarray(fine['db'], np.float64)
    lg = np.log2(np.maximum(f, 1.0) / fine['f0'])
    return 10 ** (np.interp(lg, np.arange(len(db)) * fine['step'], db, left=0.0, right=0.0) / 20)


def _fine_psd(x, n_pts):
    """long-term PSD on the 1/24-octave grid (Welch 4096, each point the mean over +-1/48 octave, nearest bin
    when that holds none)"""
    from scipy.signal import welch
    nper = int(min(4096, 2 ** int(np.log2(max(len(x), 256)))))
    f, P = welch(x, SR, nperseg=nper, noverlap=nper * 3 // 4, window='hann')
    g = FINE_F0 * 2 ** (np.arange(n_pts) * FINE_STEP)
    c = np.concatenate([[0.0], np.cumsum(P)])
    lo = np.searchsorted(f, g * 2 ** (-FINE_STEP / 2))
    hi = np.searchsorted(f, g * 2 ** (FINE_STEP / 2))
    near = np.clip(np.searchsorted(f, g), 0, len(f) - 1)
    out = np.where(hi > lo, (c[np.maximum(hi, lo + 1)] - c[lo]) / np.maximum(hi - lo, 1), P[near])
    return out


def _fit_fine(xm, model, seeds=(41, 42), verbose=False):
    """static fine spectral structure the bands cannot hold (resonances, comb, a whistle that does not move): the
    long-term 1/24-octave PSD of the original over that of renders, minus its own band-scale trend (so the band
    levels stay calibrated), clipped to +-12 dB; None when it is under 0.4 dB RMS"""
    n_pts = int(np.floor(np.log2(F_HI / FINE_F0) / FINE_STEP)) + 1
    Po = _fine_psd(xm, n_pts)
    Pr = np.mean([_fine_psd(_mono_render(model, s, None), n_pts) for s in seeds], 0)
    r = _db(Po + 1e-30) - _db(Pr + 1e-30)
    trend = convolve1d(r, np.ones(9) / 9, mode='nearest')            # +-1/6 octave
    fd = np.clip(r - trend, -12, 12)
    lv = _db(Po + 1e-30)
    fd[lv < lv.max() - 60] = 0.0
    g = FINE_F0 * 2 ** (np.arange(n_pts) * FINE_STEP)
    # below ~300 Hz the 4096-point PSD has too few bins per point for a fine curve to mean anything
    fd *= np.clip((g - 150.0) / 150.0, 0, 1)
    rms = float(np.sqrt(np.mean(fd ** 2)))
    if verbose:
        print(f'  fine spectral curve: {rms:.2f} dB RMS')
    if rms < 0.4:
        return None
    return {'f0': FINE_F0, 'step': FINE_STEP, 'db': [round(float(v), 1) for v in fd]}


def fit(x, name=None, n_bands=N_BANDS, tol_db=1.0, max_keys=160, gesture_ms=25.0, max_rank=16, iters=3, diff_frames=3,
        ridges=True, fine=False, texture=True, stereo=True, seeds=(1, 2), verbose=False):
    """fit a spectral-noise model to a recording (path or (n,)/(n, 2) 48 kHz array); returns the model dict"""
    if name is None and isinstance(x, str):
        name = os.path.basename(x)
    x2 = _load(x)
    n = len(x2)
    mono = bool(np.allclose(x2[:, 0], x2[:, 1]))
    cents = centers(n_bands)
    bw = bandwidths(cents)
    A = analyse(x2, cents, stereo=stereo and not mono)
    T = A['T']
    D = A['D']
    xm = x2.mean(1)

    rmods = []
    if ridges:
        for tr in _find_ridges(xm, D, cents):
            rm, (k, fz, amp) = _ridge_model(tr)
            rmods.append(rm)
            D = _ridge_remove(D, cents, k, fz, amp)

    k = max(1, int(round(gesture_ms / (1000.0 * HOP / SR))))
    lv0 = _db((D * bw[:, None]).sum(0))
    on = int(np.argmax(lv0 > lv0.max() - 30))
    k_max = int(max(k, min(32, T // 6)))
    while True:
        # texture-aware gesture scale: how much the smoothed level wanders around a local quadratic (crackle,
        # flutter; a quadratic follows the gesture's own curvature); widen the smoothing while that wander is well
        # above tolerance, the rest is left to the texture statistics
        L1 = _db((_seg_smooth(D, k, on) * bw[:, None]).sum(0))
        fluct = _wander(L1, k, on)
        if fluct <= 2.0 * tol_db or k >= k_max:
            break
        k = min(k_max, int(np.ceil(k * 1.5)))
    tgt = _db(_seg_smooth(D, k, on))
    top = tgt.max()
    tgt = np.maximum(tgt, top - FLOOR_DB)
    act = tgt > top - 70
    Gp0 = tgt + 10 * np.log10(bw)[:, None]
    W = np.clip((Gp0 - (Gp0.max() - 50)) / 25, 0, 1).T                 # (T, B) cell weights
    # low-rank denoising: keep only the structure that stands above what a noise band's estimate wanders
    sig = _noise_sigma(n, cents, k, on)
    sw = float(np.sqrt((W * sig[None] ** 2).sum() / max(W.sum(), 1e-9)))
    rank = max_rank
    Y = tgt.T
    for K in range(1, max_rank + 1):
        Yh = _lowrank(Y, W, K)
        r = float(np.sqrt((W * (Y - Yh) ** 2).sum() / max(W.sum(), 1e-9)))
        if r <= sw:
            rank = K
            break
    # quiet cells (weight < 1) keep their measured level: the low-rank fill there is a guess (a dark corner where
    # the highs have died must stay dark)
    tgtK = np.maximum((W * Yh + (1 - W) * Y).T, top - FLOOR_DB)
    if verbose:
        print(f'  rank {rank}: residual {r:.2f} dB, noise {sw:.2f} dB, onset frame {on}')
    Gm = tgtK.copy()
    tk = np.arange(T) * HOP / SR
    for it in range(iters):
        acc = np.zeros_like(Gm)
        for s in seeds:
            y = _synth(n, tk, Gm.T, cents, 7919 * (it + 1) + s, mono=True)
            acc += _db(_seg_smooth(analyse(y, cents, stereo=False)['D'], k, on))
        diff = tgtK - acc / len(seeds)
        diff = uniform_filter(np.where(act, diff, 0.0), size=(3, diff_frames), mode="nearest")
        Gm = np.maximum(Gm + 0.85 * np.clip(diff, -12, 12), top - FLOOR_DB)
        if verbose:
            print(f'  iter {it}: mean |diff| on active cells {np.abs(diff[act]).mean():.2f} dB')

    ild, icc, gw = _stereo_traj(A['stereo'], T, mono or not stereo)
    Gp = Gm + 10 * np.log10(bw)[:, None]
    wc = np.clip((Gp - (Gp.max() - 50)) / 25, 0, 1).T          # (T, B)
    lvl = _db((10 ** (Gp / 10)).sum(0))
    peak = int(np.argmax(lvl))
    onset = int(np.argmax(lvl > lvl.max() - 30))
    gact = np.clip((lvl - (lvl.max() - 40)) / 20, 0, 1)[:, None] * gw * len(GROUPS)   # (T, G)
    Y = np.concatenate([Gm.T, ild / 2.0, icc * 10.0], 1)
    Wc = np.concatenate([wc, 0.5 * gact, 0.5 * gact], 1)
    tol = max(tol_db, fluct)
    keys = _keyframes(Y, Wc, tol, must=(peak, onset), max_keys=max_keys)

    model = {
        'kind': KIND, 'version': VERSION, 'source': name, 'sr': SR, 'dur': round(n / SR, 4),
        'level_db': round(_peak_rms_db(x2), 2),
        'bands': {'n': int(n_bands), 'f_lo': F_LO, 'f_hi': F_HI, 'scale': 'ln(1+f/80)'},
        'fit': {'gesture_ms': round(k * 1000.0 * HOP / SR, 1), 'rank': rank, 'tol_db': round(tol, 2), 'fluct_db': round(fluct, 2)},
        'env': {'t': [round(float(tk[k]), 4) for k in keys],
                'db': [[round(float(v), 1) for v in Gm[:, k]] for k in keys]},
        'stereo': None if (mono or not stereo) else {
            'groups': list(GROUPS),
            'ild': [[round(float(v), 2) for v in ild[k]] for k in keys],
            'icc': [[round(float(v), 3) for v in icc[k]] for k in keys]},
        'fine': None,
        'texture': None,
        'ridges': rmods,
    }
    if fine:
        model['fine'] = _fit_fine(xm, model, verbose=verbose)
    if texture:
        model['texture'] = _fit_texture(xm, model, verbose=verbose)
    return model


# --------------------------------------------------------------------------------------------- texture


def _env_dense(model, times):
    """interpolated band density dB (B, len(times)) of a model's env keyframes"""
    tk = np.asarray(model['env']['t'], np.float64)
    E = np.asarray(model['env']['db'], np.float64)
    return np.stack([np.interp(times, tk, E[:, b]) for b in range(E.shape[1])])


def _group_env(model, n):
    """per group gesture amplitude at 1 kHz frames: sqrt(sum of band density x bandwidth / (SR/2))"""
    cents = centers(model['bands']['n'])
    bw = bandwidths(cents)
    m1 = n // 48
    t1 = (np.arange(m1) + 0.5) * 48 / SR
    Dl = 10 ** (_env_dense(model, t1) / 10) * (bw / (SR / 2))[:, None]
    gi = _group_of(cents)
    G = len(GROUPS) - 1
    Eg = np.stack([np.sqrt(Dl[gi == g].sum(0) + 1e-30) for g in range(G)])
    return Eg


def _texture_stats(xm, Eg, onset_s=None):
    """per group: modulation spectrum (G, len(RATES)) of log amplitude around the gesture, the std of the log
    analytic envelope ('kurt' key: the impulsiveness statistic) and the low-rate log-amplitude series"""
    n = len(xm)
    m1 = Eg.shape[1]
    N = n
    X = sfft.rfft(xm, workers=_workers())
    f = np.fft.rfftfreq(N, 1 / SR)
    G = Eg.shape[0]
    spec = np.full((G, len(RATES)), np.nan)
    kurt = np.full(G, np.nan)
    halves = np.full((2, G, len(RATES)), np.nan)
    lows = [None] * G
    Edb = 20 * np.log10(Eg + 1e-30)
    gpow = (Eg ** 2).sum(1)
    share = gpow / max(gpow.sum(), 1e-30)
    t1 = np.arange(m1) / 1000.0
    fm = np.fft.rfftfreq(m1, 1e-3)
    ed = RATES * 2 ** -0.25, RATES * 2 ** 0.25
    for g in range(G):
        if share[g] < 0.004:
            continue
        mask = (f >= GROUPS[g]) & (f < GROUPS[g + 1])
        xg = sfft.irfft(X * mask, N, workers=_workers())
        a = np.sqrt((xg[:m1 * 48].reshape(m1, 48) ** 2).mean(1))
        act = Edb[g] > Edb[g].max() - 30
        if onset_s is not None:
            act &= np.abs(t1 - onset_s) > 0.015
        if act.sum() < 24:
            continue
        r = np.log(a + 1e-12) - np.log(Eg[g] + 1e-12)

        def modspec(sel):
            w = convolve1d(sel.astype(np.float64), np.ones(5) / 5, mode='constant')   # soft edges
            mu = (r * w).sum() / w.sum()
            rc = (r - mu) * w
            S = np.abs(np.fft.rfft(rc)) ** 2
            var = (rc ** 2).sum() / w.sum()
            Sb = np.array([S[(fm >= lo) & (fm < hi)].sum() for lo, hi in zip(*ed)])
            return Sb / max(S[1:].sum(), 1e-30) * var, rc
        spec[g], rc = modspec(act)
        # the two halves of the active span apart: flutter that is there throughout, not a gesture misfit in one spot
        idx = np.nonzero(act)[0]
        mid = idx[len(idx) // 2]
        for h, sel in enumerate((act & (np.arange(m1) < mid), act & (np.arange(m1) >= mid))):
            halves[h, g] = modspec(sel)[0] if sel.sum() >= 12 else np.nan
        # impulsiveness: std of the log analytic envelope around the gesture (Gaussian noise: Rayleigh, 0.64 nepers;
        # sparse crackle: deep gaps between clicks, up to ~1.5). Robust where kurtosis is ruled by the largest click.
        Xa = np.zeros(N, complex)
        Xa[:len(f)] = X * mask
        Xa[1:len(f)] *= 2.0
        ae = np.abs(sfft.ifft(Xa, workers=_workers()))[:m1 * 48]
        sel = np.repeat(act, 48)
        lr_ = np.log(ae[sel] + 1e-12) - np.log(np.repeat(Eg[g], 48)[sel] + 1e-12)
        kurt[g] = float(lr_.std())
        lp = np.fft.irfft(np.fft.rfft(rc) * (fm <= AM_MAX_RATE), m1)
        lows[g] = (lp, act)
    return {'spec': spec, 'kurt': kurt, 'share': share, 'lows': lows, 'halves': halves}


def _coherence(st):
    """mean correlation of the low-rate log amplitude between the two loudest active groups"""
    ok = [g for g in range(len(st['share'])) if st['lows'][g] is not None]
    if len(ok) < 2:
        return 0.0
    ok.sort(key=lambda g: -st['share'][g])
    (a, ma), (b, mb) = st['lows'][ok[0]], st['lows'][ok[1]]
    m = ma & mb
    if m.sum() < 24:
        return 0.0
    a, b = a[m], b[m]
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _mono_render(model, seed, texture):
    cents = centers(model['bands']['n'])
    n = int(round(model['dur'] * SR))
    return _synth(n, model['env']['t'], model['env']['db'], cents, seed, texture=texture,
                  ridges=model.get('ridges') or (), mono=True, fine=model.get('fine'))


def _avg_stats(model, n, Eg, texture, seeds, onset_s):
    sts = [_texture_stats(_mono_render(model, s, texture)[:n], Eg, onset_s) for s in seeds]
    spec = np.nanmean(np.stack([s['spec'] for s in sts]), 0)
    kurt = np.nanmean(np.stack([s['kurt'] for s in sts]), 0)
    halves = np.nanmean(np.stack([s['halves'] for s in sts]), 0)
    return {'spec': spec, 'kurt': kurt, 'share': sts[0]['share'], 'lows': sts[0]['lows'], 'all': sts, 'halves': halves}


def _fit_am(Sr, base, wg, ok, low):
    """AM from the low-rate log-amplitude variance the real has beyond `base` (a render's stats)"""
    spr, sp0 = np.nan_to_num(Sr['spec']), np.nan_to_num(base['spec'])
    sd0 = np.nan_to_num(np.std(np.stack([s['spec'] for s in base['all']]), 0))
    sig = (spr > 1.25 * sp0 + 2.0 * sd0) & low[None]
    ex = np.where(sig, spr - sp0, 0.0)
    # only what both halves of the active span have in excess (stationary flutter, not one gesture misfit)
    hx = np.nan_to_num(Sr['halves'] - 1.25 * base['halves'], nan=0.0)
    ex = np.minimum(ex, np.maximum(np.minimum(hx[0], hx[1]), 0.0))
    ex[~ok] = 0.0
    depth = np.minimum(np.sqrt(ex.sum(1)), AM_MAX_DEPTH)
    shape = (wg[:, None] * ex).sum(0)
    if (depth * (wg > 0.03)).max() <= 0.08 or shape.sum() <= 0:
        return None
    coh = float(np.clip(_coherence(Sr) - _coherence(base['all'][0]), 0.0, 1.0))
    return {'rates': [float(r) for r in RATES], 'spec': [round(float(v), 5) for v in shape / shape.sum()],
            'depth': [round(float(v), 4) for v in depth], 'coh': round(coh, 3)}


def _fit_texture(xm, model, seeds=(31, 32, 33), verbose=False):
    """texture statistics by analysis-by-synthesis on mono renders of the gesture:
    grains are detected by excess fast (>= 128 Hz) log-amplitude variance (Poisson clicks put variance at every rate,
    flutter does not) and their density is fitted to the std of the log analytic envelope (monotone, stable across
    seeds; kurtosis is not: it is ruled by the largest click); AM is the low-rate (<= 64 Hz) log-amplitude variance
    beyond what Gaussian noise / the grains already make. Grains and AM are refitted once against each other."""
    import warnings
    n = len(xm)
    Eg = _group_env(model, n)
    tk = np.asarray(model['env']['t'])
    lv = np.asarray(model['env']['db']).max(1)
    onset_s = float(tk[int(np.argmax(lv > lv.max() - 30))])
    low = (RATES <= AM_MAX_RATE) & (RATES >= AM_MIN_RATE)
    high = RATES >= 128
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        Sr = _texture_stats(xm, Eg, onset_s)
        S0 = _avg_stats(model, n, Eg, None, seeds, onset_s)
        ok = ~np.isnan(Sr['kurt']) & ~np.isnan(S0['kurt'])
        if not ok.any():
            return None
        share = np.where(ok, Sr['share'], 0.0)
        wg = share / share.sum()
        rnd = lambda v, k=3: [None if not np.isfinite(e) else round(float(e), k) for e in v]
        hr = np.nansum(np.nan_to_num(Sr['spec'])[:, high], 1)
        h0 = np.nansum(np.nan_to_num(S0['spec'])[:, high], 1)
        hsd = np.std([np.nansum(np.nan_to_num(a['spec'])[:, high], 1) for a in S0['all']], 0)
        hx = float(np.sum(wg * (hr - h0)))
        hratio = float(np.sum(wg * hr) / max(np.sum(wg * h0), 1e-9))
        tex = {'am': None, 'grains': None, 'stats': {
            'lenv_real': rnd(Sr['kurt']), 'lenv_gauss': rnd(S0['kurt']),
            'fast_var_real': rnd(hr), 'fast_var_gauss': rnd(h0), 'fast_ratio': round(hratio, 2)}}
        grainy = hx > max(0.03, 3 * float(np.sum(wg * hsd))) and hratio > 1.5
        if verbose:
            print('  texture: lenv real', np.round(Sr['kurt'], 2), 'gauss', np.round(S0['kurt'], 2),
                  f'fast var ratio {hratio:.2f}', 'grainy' if grainy else '')

        def fit_density(am):
            cache = {}

            def loss(d):
                st = _avg_stats(model, n, Eg, {'am': am, 'grains': {'density': d}}, seeds, onset_s)
                cache[d] = st
                return float(np.nansum(wg * (st['kurt'] - Sr['kurt']) ** 2))
            grid = [25, 60, 150, 400, 1000, 2500, 6000, 15000]
            ls = [loss(d) for d in grid]
            i = int(np.argmin(ls))
            best, bl = grid[i], ls[i]
            for j in (i - 1, i + 1):
                if 0 <= j < len(grid):
                    d = float(np.sqrt(grid[i] * grid[j]))
                    v = loss(d)
                    if v < bl:
                        best, bl = d, v
            return best, bl, cache[best]

        base = S0
        if grainy:
            dens, bl, base = fit_density(None)
            tex['grains'] = {'density': round(float(dens), 1), 'spread': SPREAD}
            if verbose:
                print(f'  grains: density {dens:.0f}/s (lenv loss {bl:.4f})')
        am = _fit_am(Sr, base, wg, ok, low)
        if am is not None and grainy:
            # flutter raises the log-envelope spread too: refit the density with the AM in, then the AM again
            dens, bl, _ = fit_density(am)
            tex['grains'] = {'density': round(float(dens), 1), 'spread': SPREAD}
            base = _avg_stats(model, n, Eg, {'grains': tex['grains']}, seeds, onset_s)
            am = _fit_am(Sr, base, wg, ok, low)
            if verbose:
                print(f'  grains refit with AM: density {dens:.0f}/s')
        if am is not None:
            S1 = _avg_stats(model, n, Eg, {'am': am, 'grains': tex['grains']}, seeds, onset_s)
            spr, sp0, sp1 = [np.nan_to_num(S['spec']) for S in (Sr, base, S1)]
            want = np.maximum(spr[:, low].sum(1) - sp0[:, low].sum(1), 0)
            got = np.maximum(sp1[:, low].sum(1) - sp0[:, low].sum(1), 1e-6)
            depth = np.asarray(am['depth'])
            depth = np.where(depth > 0, np.minimum(depth * np.sqrt(np.clip(want / got, 0.25, 4.0)), AM_MAX_DEPTH), 0.0)
            am['depth'] = [round(float(v), 4) for v in depth]
            if verbose:
                print('  am depth', np.round(depth, 3), 'peak rate', am['rates'][int(np.argmax(am['spec']))])
        tex['am'] = am
        tex['stats']['am_var_real'] = rnd(np.nan_to_num(Sr['spec'])[:, low].sum(1), 4)
        tex['stats']['am_var_base'] = rnd(np.nan_to_num(base['spec'])[:, low].sum(1), 4)
    return tex


# ---------------------------------------------------------------------------------------------- render


def check(model):
    if not isinstance(model, dict) or model.get('kind') != KIND:
        raise ValueError(f"not a {KIND} model (kind={model.get('kind') if isinstance(model, dict) else None!r})")
    if int(model.get('version', 0)) > VERSION:
        raise ValueError(f"model version {model.get('version')} is newer than this renderer ({VERSION})")


def _centroid_oct(E, cents, bw):
    """per keyframe: centroid in log2(Hz) weighted by band loudness (power ** 0.3, so a sub rumble does not rule
    it), and the keyframe's total power"""
    p = 10 ** (E / 10) * bw[None]
    lw = (p / max(p.max(), 1e-30)) ** 0.3
    return (lw * np.log2(cents)[None]).sum(1) / np.maximum(lw.sum(1), 1e-30), p.sum(1)


def render(model, seed=0, length=1.0, speed=1.0, sweep=1.0, shift=0.0, tilt=0.0, roughness=1.0, width=1.0,
           ridges=1.0, gain_db=0.0, level=True, limit=True, steady=True):
    """render a model: (n, 2) float32 at 48 kHz. New `seed` = new noise, same statistics. See the module doc.
    limit=False skips the -0.5 dBFS sample-peak cap (a per-take gain: the fit engine levels takes itself).
    steady=False levels the take by its own loudest 50 ms (the 0.2 renderer; a hybrid's air layer was fitted so)"""
    check(model)
    cents = centers(model['bands']['n'])
    bw = bandwidths(cents)
    speed = max(float(speed), 1e-3)
    s = max(float(length), 1e-3) / speed
    n = max(64, int(round(model['dur'] * s * SR)))
    tk = np.asarray(model['env']['t'], np.float64) * s
    E = np.asarray(model['env']['db'], np.float64)
    oct_ = float(shift) + SPEED_OCT * np.log2(speed)
    co, cp = _centroid_oct(E, cents, bw)
    cm = (co * cp).sum() / max(cp.sum(), 1e-30)
    octs = oct_ + (float(sweep) - 1.0) * (co - cm)
    if np.any(np.abs(octs) > 1e-9):
        ec = warp(cents)
        E = np.stack([np.interp(warp(cents / 2 ** o), ec, row) for o, row in zip(octs, E)])
    if tilt:
        E = E + float(tilt) * np.log2(cents / 1000.0)[None]
    st = model.get('stereo')
    if st is not None:
        icc = 1 - float(width) * (1 - np.asarray(st['icc'], np.float64))
        st = {'ild': np.asarray(st['ild'], np.float64) * float(width), 'icc': np.clip(icc, -1, 1)}
    rds = []
    if ridges:
        for rd in model.get('ridges') or ():
            rt = np.asarray(rd['t'], np.float64)
            o = oct_ + (float(sweep) - 1.0) * (np.interp(rt, np.asarray(model['env']['t']), co) - cm)
            rds.append({'t': rt * s, 'f': np.minimum(np.asarray(rd['f']) * 2 ** o, SR * 0.45),
                        'db': np.asarray(rd['db']) + 20 * np.log10(float(ridges)), 'jitter': rd.get('jitter', 0.002)})
    y, p, pk = _synth(n, tk, E, cents, seed, stereo=st, texture=model.get('texture'), ridges=rds,
                      rate_scale=speed, fine=model.get('fine'), fine_shift=oct_, rough=float(roughness), power=True)
    if level and steady:   # the expected loudest 50 ms on the original's, this take's K-weighted energy on its own
        y *= 10 ** ((model['level_db'] + float(gain_db) - _peak_mean_db(p)) / 20) * _take_gain(y, pk)   # expectation
    elif level:
        y *= 10 ** ((model['level_db'] + float(gain_db) - _peak_rms_db(y)) / 20)
    pk = np.abs(y).max()
    if limit and pk > 10 ** (-0.5 / 20):
        y *= 10 ** (-0.5 / 20) / pk
    return y.astype(np.float32)


# ------------------------------------------------------------------------------------- serialisation


def save(model, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(model, fh, separators=(',', ':'))


def load(path):
    with open(path, encoding='utf-8') as fh:
        m = json.load(fh)
    check(m)
    return m


def summary(model):
    """one paragraph in words: length, peak, centroid path, width, texture, ridges, model size"""
    check(model)
    cents = centers(model['bands']['n'])
    bw = bandwidths(cents)
    tk = np.asarray(model['env']['t'])
    E = np.asarray(model['env']['db'])
    co, cp = _centroid_oct(E, cents, bw)
    lv = _db(cp)
    pk = int(np.argmax(lv))
    on = int(np.argmax(lv > lv.max() - 20))
    act = lv > lv.max() - 20
    last = len(lv) - 1 - int(np.argmax(act[::-1]))
    hz = lambda o: f'{2 ** o / 1000:.2g} kHz' if 2 ** o >= 1000 else f'{2 ** o:.0f} Hz'
    w = [f"{model.get('source') or 'model'}: {model['dur']:.2f} s noise gesture, {len(tk)} keyframes",
         f'up 20 dB in {1000 * (tk[pk] - tk[on]):.0f} ms to the peak at {tk[pk]:.2f} s, '
         f'within 20 dB until {tk[last]:.2f} s',
         f'centroid {hz(co[on])} -> {hz(co[pk])} (peak) -> {hz(co[last])}']
    st = model.get('stereo')
    if st:
        icc = np.asarray(st['icc'])
        ild = np.asarray(st['ild'])
        w.append(f'inter-channel corr {np.median(icc):.2f} (min {icc.min():.2f}), pan {ild.min():+.1f}..{ild.max():+.1f} dB '
                 '(per band group)')
    else:
        w.append('mono')
    tx = model.get('texture') or {}
    if tx.get('am'):
        am = tx['am']
        sp = np.asarray(am['spec'])
        w.append(f"flutter around {am['rates'][int(np.argmax(sp))]:.0f} Hz, depth up to "
                 f"{8.686 * max(am['depth']):.1f} dB, coherence {am['coh']:.2f}")
    if tx.get('grains'):
        w.append(f"crackle: {tx['grains']['density']:.0f} grains / s")
    for rd in model.get('ridges') or ():
        w.append(f"tonal ridge {rd['f'][0]:.0f} -> {rd['f'][-1]:.0f} Hz over {rd['t'][-1] - rd['t'][0]:.2f} s")
    w.append(f"{len(json.dumps(model)) / 1024:.1f} kB JSON")
    return '; '.join(w)


def to_code(model, var='MODEL'):
    """the model as readable Python source that re-renders it (fresh noise per seed)"""
    check(model)
    m = {k: v for k, v in model.items() if k not in ('env', 'stereo', 'texture', 'ridges', 'fine')}
    lines = [f'"""{summary(model)}"""',
             'from earshot.forge.fit.spectral import render', '',
             f'{var} = {{']
    for k, v in m.items():
        lines.append(f'    {k!r}: {v!r},')
    cents = centers(model['bands']['n'])
    lines.append('    # band centres (Hz): ' + ' '.join(f'{c:.0f}' for c in cents))
    lines.append("    # the gesture: keyframe time (s) -> band power density (dB re full-scale white noise),")
    lines.append("    # linear in dB between keyframes")
    lines.append("    'env': {'t': " + repr(model['env']['t']) + ",")
    lines.append("            'db': [")
    for t, row in zip(model['env']['t'], model['env']['db']):
        lines.append(f"                {row!r},  # {t:.3f} s")
    lines.append('            ]},')
    st = model.get('stereo')
    if st:
        lines.append("    # per band group " + repr(st['groups']) + ' Hz at each keyframe: L/R level difference (dB) and correlation')
        lines.append("    'stereo': {'groups': " + repr(st['groups']) + ",")
        lines.append("               'ild': " + repr(st['ild']) + ",")
        lines.append("               'icc': " + repr(st['icc']) + '},')
    else:
        lines.append("    'stereo': None,")
    lines.append("    # optional static fine spectral curve (dB every 1/24 octave from f0), multiplied into the noise")
    lines.append(f"    'fine': {model.get('fine')!r},")
    lines.append("    # statistics of what the gesture does not hold: flutter (am) and crackle (grains)")
    lines.append(f"    'texture': {model.get('texture')!r},")
    lines.append("    # tonal ridges: keyframes of frequency (Hz) and amplitude (dB re full scale)")
    lines.append(f"    'ridges': {model.get('ridges')!r},")
    lines.append('}')
    lines += ['', '', 'def make(seed=0, **knobs):',
              '    """knobs: length, speed, sweep, shift, tilt, roughness, width, ridges, gain_db"""',
              f'    return render({var}, seed=seed, **knobs)', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------- quality


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(1) if x.ndim == 2 else x


def _onset_trim(m, db=-30.0, pre_ms=5.0):
    k = 48
    nb = len(m) // k
    if nb < 2:
        return m
    e = _db((m[:nb * k].reshape(nb, k) ** 2).mean(1))
    on = int(np.argmax(e > e.max() + db)) * k
    return m[max(0, on - int(SR * pre_ms / 1000)):]


def _mel_fb(n_fft, n_mel=64, f_lo=30.0, f_hi=20000.0):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(f_lo), mel(f_hi), n_mel + 2))
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n_mel, len(f)))
    for i in range(n_mel):
        a, c, b = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - a) / (c - a), (b - f) / (b - c)), 0, None)
        if fb[i].sum() == 0:
            fb[i, int(np.argmin(np.abs(f - c)))] = 1.0
    return fb


def _melspec(m, n_fft):
    hop = n_fft // 4
    T = max(1, (len(m) + hop - 1) // hop)
    win = _hann(n_fft)
    xp = np.zeros(T * hop + n_fft)
    xp[n_fft // 2:n_fft // 2 + len(m)] = m
    fr = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:T]
    P = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
    return _db(P @ _mel_fb(n_fft).T + 1e-20)


def _env_db(m, ms=5.0):
    k = int(SR * ms / 1000)
    nb = max(1, len(m) // k)
    return _db((m[:nb * k].reshape(nb, k) ** 2).mean(1) + 1e-20)


def _attack_ms(m):
    e = _env_db(m, 2.0)
    e = np.convolve(e, np.ones(3) / 3, mode='same')
    pk = int(np.argmax(e))
    on = int(np.argmax(e > e[pk] - 20))
    return 2.0 * (pk - on)


def compare(a, b, clap=False):
    """distances between two sounds (each levelled to its loudest 50 ms RMS, onset-aligned):
    mel_db (mean |dB| over active log-mel cells, mean of FFT 256 / 1024 / 4096), mel_256/1024/4096, env_corr
    (5 ms dB envelopes), attack_ms (a, b), tone_db (1/3-octave balance RMS), len_s (a, b), optional clap cosine"""
    from ..tone import profile
    A = _onset_trim(_mono(a))
    B = _onset_trim(_mono(b))
    A = A * 10 ** (-_peak_rms_db(A) / 20)
    B = B * 10 ** (-_peak_rms_db(B) / 20)
    n = max(len(A), len(B))
    Ap, Bp = np.zeros(n), np.zeros(n)
    Ap[:len(A)] = A
    Bp[:len(B)] = B
    out = {}
    ds = []
    for nf in (256, 1024, 4096):
        MA, MB = _melspec(Ap, nf), _melspec(Bp, nf)
        top = max(MA.max(), MB.max())
        MA, MB = np.maximum(MA, top - 80), np.maximum(MB, top - 80)
        actv = np.maximum(MA, MB) > top - 60
        d = float(np.abs(MA - MB)[actv].mean()) if actv.any() else 0.0
        out[f'mel_{nf}'] = round(d, 2)
        ds.append(d)
    out['mel_db'] = round(float(np.mean(ds)), 2)
    ea, eb = _env_db(Ap), _env_db(Bp)
    top = max(ea.max(), eb.max())
    ea, eb = np.maximum(ea, top - 60), np.maximum(eb, top - 60)
    actv = np.maximum(ea, eb) > top - 60
    out['env_corr'] = round(float(np.corrcoef(ea[actv], eb[actv])[0, 1]), 3) if actv.sum() > 3 else 0.0
    out['attack_ms'] = (_attack_ms(A), _attack_ms(B))
    out['tone_db'] = round(float(np.sqrt(np.mean((profile(A) - profile(B)) ** 2))), 2)
    out['len_s'] = (round(len(_mono(a)) / SR, 3), round(len(_mono(b)) / SR, 3))
    if clap:
        from ...models import clap as _clap
        E = _clap.get().embed_audio([_mono(a).astype(np.float32), _mono(b).astype(np.float32)])
        out['clap_cos'] = round(float(E[0] @ E[1]), 4)
    return out


def fit_report(x, clap=False, seeds=(11, 12), **fit_kw):
    """fit + two renders; {'model', 'renders', 'fit': compare(orig, r1), 'floor': compare(r1, r2), 'seconds'}"""
    import time
    t0 = time.time()
    x2 = _load(x)
    m = fit(x if isinstance(x, str) else x2, **fit_kw)
    t1 = time.time()
    r1 = render(m, seed=seeds[0])
    r2 = render(m, seed=seeds[1])
    q = compare(x2, r1, clap=clap)
    fl = compare(r1, r2, clap=clap)
    return {'model': m, 'renders': (r1, r2), 'orig': x2, 'fit': q, 'floor': fl,
            'seconds': {'fit': round(t1 - t0, 2), 'render': round((time.time() - t1) / 2, 2)}}


def pair_png(path, rows, span_s=None, title=None):
    """spectrogram pairs: rows of (label, original, resynthesis[, resynthesis 2]); log-frequency, same dB range"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncol = max(len(r) - 1 for r in rows) + 1
    fig, axs = plt.subplots(len(rows), ncol, figsize=(4.2 * ncol, 2.6 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        label, sigs = row[0], [_onset_trim(_mono(s)) for s in row[1:]]
        sp = span_s or max(len(s) for s in sigs) / SR
        nsp = int(sp * SR)
        Ss = []
        for s in sigs:
            s = s * 10 ** (-_peak_rms_db(s) / 20)
            z = np.zeros(nsp)
            z[:min(nsp, len(s))] = s[:nsp]
            Ss.append(_melspec(z, 1024))
        top = max(S.max() for S in Ss)
        for j, S in enumerate(Ss):
            ax = axs[i, j]
            ax.imshow(np.maximum(S, top - 80).T, origin='lower', aspect='auto', cmap='magma', vmin=top - 80, vmax=top,
                      extent=(0, sp, 0, 64))
            ax.set_yticks([0, 16, 32, 48, 64])
            ax.set_yticklabels(['30', '600', '2k', '6k', '20k'], fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{label} | {'orig' if j == 0 else f'render {j}'}", fontsize=7)
        ax = axs[i, -1]
        for j, s in enumerate(sigs):
            s = s * 10 ** (-_peak_rms_db(s) / 20)
            e = _env_db(s[:nsp])
            ax.plot(np.arange(len(e)) * 0.005, np.maximum(e, -70), lw=0.8, label='orig' if j == 0 else f'r{j}')
        ax.set_ylim(-70, 5)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_title('5 ms envelope dB', fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
