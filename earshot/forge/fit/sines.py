"""Sines fitter: a recording of a tonal / pitched sound -> a small JSON model -> a close re-render, or Python code.

    from earshot.forge.fit import sines
    m = sines.fit('ding.wav')                     # or a 48 kHz array (mono or (n, 2))
    y = sines.render(m)                           # the re-render (mono, 48 kHz, same level and timing)
    y = sines.render(m, seed=3, knobs={'transpose': 2, 'decay': 1.4, 'vary': 0.3})   # a new variation
    src = sines.to_code(m, name='ding')           # standalone numpy code that renders the same sound
    print(sines.describe(m))                      # 'strike at 12 ms, 23 partials 523-9870 Hz, ...'

Analysis by synthesis in the McAulay-Quatieri / SMS tradition, with no harmonic series forced on the sound:
  1. STFT peaks: zero-phase Blackman frames, x2 zero padding, local maxima that stand out from the local spectrum,
     frequency and amplitude by parabolic interpolation on the dB magnitude, phase at the peak.
  2. Partial tracking: peaks joined frame to frame to the nearest predicted frequency (last frequency + slope, so
     chirps and glides track), births for unmatched peaks, deaths after a gap; short or quiet tracks are dropped and
     the strongest `max_partials` kept. Inharmonic partials (bells, bars, glass, kalimba tines) keep their own
     frequencies; doublet beating lives in the amplitude trajectory.
  3. Onsets: a long window smears a strike backwards by half a window (a soft pre-echo attack). Strikes are found on
     a 1 ms band envelope; a partial born near one starts exactly at the strike with the measured rise, and its
     smeared first frames are replaced by the decay line extrapolated back from the first clean frames.
     A track is kept only if it stands well above the local median spectrum (a chain of noise peaks does not), and a
     second pass tracks the residual for partials hidden under stronger ones (doublet partners, weak highs).
  4. Residual: the partials are re-synthesised phase-accurately (cubic phase interpolation between frames) and
     subtracted. At each strike the residual's first ms become damped sinusoids (matrix pencil: a 6-8 ms window above
     400 Hz for the click, a 64 ms window below for the body thump), kept only if they explain the strike. What is
     left (breath, hiss, scrape, the recording's own noise floor, which CLAP and ears both notice when it is missing)
     is noise in 1/3-octave bands with dB envelopes on one shared adaptive time grid (dense around the attack, sparse
     in the tail); a band's level counts only its noise-like part (median of its STFT bins), so leftover partial
     energy never turns into hiss.
  5. Compression: every partial's frequency (cents) and amplitude (dB) trajectories become breakpoints (Ramer-Douglas-
     Peucker with a vertical tolerance), the residual grid likewise.
  6. Window choice by analysis-by-synthesis: the sound is fitted at several window sizes (short for chirps and pops,
     long for close bell partials) and the model whose re-render is spectrally closest to the recording is kept.

Model (plain JSON-serialisable dict):
    {'kind': 'sines', 'version': 1, 'sr': 48000, 'duration': s, 'onset': s (first strike), 'onsets': [s, ...],
     'partials': [{'t0': birth s, 'rise': s, 't1': last breakpoint s, 'rel': release s, 'ph': phase at t0 (rad),
                   'freq': [[t, Hz], ...], 'amp': [[t, dBFS], ...]}, ...]          (strongest first)
     'transients': [{'t': start s, 'atoms': [[f Hz, tau s, amplitude, phase], ...]}, ...]   (one per strike)
     'residual': {'bands': [Hz centres, 1/3 octave], 'times': [s, ...], 'db': [[dBFS per time] per band],
                  'hf': [[Hz, dB], ...]},
     'width': side/mid energy of the source in dB, 'side': {'bands': [Hz, octaves], 'times': [s, ...],
              'pan': [[gain per time] per band], 'diffuse': [[gain per time] per band]} (a stereo source; else None),
     'coherent': True (0.4; see vary below), 'analysis': {...}, 'summary': {...}}
The residual bands are triangles on log frequency that sum to one; a band's dB is the RMS of the band-passed residual.
The top band (16 kHz) is flat to Nyquist and the lowest (50 Hz) flat to DC. Above 16 kHz the noise follows the
recording's own roll-off `hf` (its residual's spectrum re 16 kHz, made monotone); models fitted before 0.3 have no
`hf` and keep the flat top band, which leaves noise above 20 kHz that the recording did not have (hf_rolloff(x)
measures it for them). The lowest band, modulated by its envelope, spreads energy down to DC: a new seed can carry an
offset of 1-2 % of its peak; the fit engine's takes (auto.render(take=True)) are DC-blocked for that.
The side (0.4; a stereo source): per octave band and keyframe (dense after the first strike, sparse in the tail) the
source's side split into its part coherent with the mid ('pan', a signed gain: L = M (1 + g), R = M (1 - g)) and the
rest ('diffuse', a gain re the mid), measured by space.side_powers. A stereo render makes the side FROM the render's
mid: each band's mid times its pan gain plus the band's quadrature (Hilbert transform, uncorrelated with it) times its
diffuse gain (space.shaped_side). So the side follows the mid (a ringing partial's tail stays as wide as the source's)
and partials are panned and decorrelated like the source's; the mono sum is the mono render. On the 42 wide or
decorrelated stereo dev targets of the replication benchmark (sines fits alone): side / mid error 13.4 dB median ->
0.15 dB (33 -> 0 off by more than 3 dB), tail L/R correlation error 0.39 -> 0.04. Models fitted before 0.4
have only 'width' and keep their old side (a fresh draw of the residual noise times 10^(width / 20), which sits far
under the mid when the sound is tonal).
Renders start at zero (a FADE_IN_S half-cosine fade-in, not mid-way into a bed; shorter, down to 0.25 ms, when the
first strike comes earlier: a clip cut into its own attack keeps it) and are peak-limited to PEAK_DB dBFS by a gain
(finish(); a hybrid's deterministic layer is not limited: the hybrid's sum is what counts). On the 78 dev targets of
the replication benchmark: CLAP -0.0002 median, 7 renders over -1 dBFS (6 over 0 dBFS) -> 0, 44 starting off zero -> 0.
A model fitted before 0.4 (no 'coherent' flag: the 0.3 bank's fit entries) renders as it did, with neither.

render(model, seed, knobs) knobs (all default to the identity, which re-renders the fit):
    transpose (semitones; every partial and the residual's spectrum move together, relative partials kept),
    stretch (time factor about the first strike; attacks keep their rise), decay (factor on every decay after its
    peak, frequency trajectories keep their timing), brightness (dB per octave about the partials' centroid),
    vibrato (cents) + vibrato_rate (Hz), drift (cents of slow random wander), residual (gain on the noise),
    transient (gain on the noise in the first 15 ms after each strike), vary (0-1: seeded per-partial detune,
    level and decay jitter, band jitter: new takes of the same object; a model with 'coherent' (0.4: every fit) draws
    the partials' jitter from smooth random functions of log frequency, vary_field(), so a harmonic series' upper
    partials move together and keep their ratios: independently detuned, they went off the harmonic grid and the
    takes rang or read as aliasing (on the films' dev fits 4 of 21 failing takes passed at vary 0)), partials (keep
    the N strongest),
    inharm (stretch of partial ratios about the lowest strong partial: f/f_low -> (f/f_low)^(1+inharm)),
    phases ('fit' or 'random'), gain_db, stereo (False: mono; True: (n, 2) with the fitted side), width (factor on
    the side), ring (None: True when vary > 0; True renders until the last partial's release ends, so a decay that vary
    lengthens is not cut at the recording's length, and fades the residual noise out over RING_FADE_S past the
    recording's length instead of stopping on it).

Deterministic: render(model, seed, knobs) is a pure function (the noise and every jitter come from `seed`).
Pure numpy / scipy; no recorded audio is stored in a model, only parameters.
"""
import inspect
import json

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

SR = 48000
KIND = 'sines'
VERSION = 1
WINDOWS = (512, 1024, 2048, 4096)

DEFAULT_KNOBS = {'transpose': 0.0, 'stretch': 1.0, 'decay': 1.0, 'brightness': 0.0, 'vibrato': 0.0,
                 'vibrato_rate': 5.5, 'drift': 0.0, 'residual': 1.0, 'transient': 1.0, 'vary': 0.0, 'partials': None,
                 'inharm': 0.0, 'phases': 'fit', 'gain_db': 0.0, 'stereo': False, 'width': 1.0, 'ring': None}
HF_POINTS = (16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0, 23000.0, 24000.0)   # the roll-off grid
RING_FADE_S = 0.06          # ring: the residual noise fades out over this past the recording's length
SIDE_BANDS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)   # the side's octave bands
SIDE_STEPS = ((0.1, 0.01), (0.3, 0.02), (0.7, 0.04), (1e9, 0.08))   # side keyframes after the onset: (until s, step s)
SIDE_TOL = 0.04             # keyframes are dropped where the gains stay this close to the line through their neighbours
SIDE_QUIET_DB = 50.0        # a keyframe this far under its band's loudest takes the band's overall side
FADE_IN_S = 0.002           # every render fades in over this (it starts at zero, not mid-way into a bed)
PEAK_DB = -1.0              # and is peak-limited to this (dBFS) by a gain


# ================================================================================================ small helpers
def _load(x, sr=SR):
    """(mono float64 signal, side/mid dB, the (n, 2) source or None) from a path or an array (mono or (n, 2))"""
    if isinstance(x, str):
        from ... import audio
        x = audio.load(x, sr=SR)
        sr = SR
    x = np.asarray(x, np.float64)
    width = -120.0
    st = None
    if x.ndim == 2:
        if x.shape[1] == 2:
            mid, side = x.mean(1), (x[:, 0] - x[:, 1]) / 2
            width = float(10 * np.log10((side ** 2).sum() / ((mid ** 2).sum() + 1e-20) + 1e-12))
            st = x
            x = mid
        else:
            x = x.mean(1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, int(sr) // g)
        st = None if st is None else signal.resample_poly(st, SR // g, int(sr) // g, axis=0)
    return x, max(width, -120.0), st


def _db(v, floor=-200.0):
    return np.maximum(20 * np.log10(np.abs(v) + 1e-20), floor)


def _simplify(t, v, tol, max_pts=None):
    """indices of breakpoints whose linear interpolation stays within `tol` (vertical; a scalar or one per point) of v
    (Ramer-Douglas-Peucker); the tolerance grows until there are at most `max_pts` points"""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    n = len(t)
    if n <= 2:
        return np.arange(n)
    while True:
        keep = np.zeros(n, bool)
        keep[[0, n - 1]] = True
        stack = [(0, n - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            seg = v[i] + (v[j] - v[i]) * (t[i + 1:j] - t[i]) / max(t[j] - t[i], 1e-12)
            err = np.abs(v[i + 1:j] - seg) / (tol[i + 1:j] if np.ndim(tol) else tol)
            k = int(np.argmax(err))
            if err[k] > 1.0:
                m = i + 1 + k
                keep[m] = True
                stack += [(i, m), (m, j)]
        idx = np.nonzero(keep)[0]
        if max_pts is None or len(idx) <= max_pts:
            return idx
        tol = tol * 1.4


def _simplify_multi(t, V, tol, max_pts=None):
    """like _simplify for several rows sharing one time grid: a point is added where any row's error exceeds tol"""
    t = np.asarray(t, float)
    V = np.asarray(V, float)
    n = len(t)
    if n <= 2:
        return np.arange(n)
    while True:
        keep = np.zeros(n, bool)
        keep[[0, n - 1]] = True
        stack = [(0, n - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            fr = (t[i + 1:j] - t[i]) / max(t[j] - t[i], 1e-12)
            seg = V[:, i:i + 1] + (V[:, j:j + 1] - V[:, i:i + 1]) * fr[None, :]
            err = np.abs(V[:, i + 1:j] - seg).max(axis=0)
            k = int(np.argmax(err))
            if err[k] > tol:
                m = i + 1 + k
                keep[m] = True
                stack += [(i, m), (m, j)]
        idx = np.nonzero(keep)[0]
        if max_pts is None or len(idx) <= max_pts:
            return idx
        tol *= 1.3


def _band_centers(lo=50.0, hi=16000.0, per_oct=3):
    k = int(round(np.log2(hi / lo) * per_oct))
    return lo * 2.0 ** (np.arange(k + 1) / per_oct)


# ================================================================================================ synthesis kernels
# These functions are copied verbatim into to_code() output: keep them self-contained (numpy only, SR global).
def band_masks(centers, nfft, sr=48000):
    """triangles on log frequency, one per band centre, that sum to one at every frequency (flat beyond the ends)"""
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    lf = np.log2(np.maximum(f, 1.0))
    lc = np.log2(np.asarray(centers, float))
    eye = np.eye(len(lc))
    return np.array([np.interp(lf, lc, eye[b]) for b in range(len(lc))])


def synth_partials(partials, n, sr=48000, cents=None):
    """sum of sinusoids: each partial follows its frequency breakpoints [[t, Hz]] (linear in log frequency) and
    amplitude breakpoints [[t, dB]] (linear in dB), starts at t0 with a linear rise and ends at t1 with a linear
    release; `cents` (optional array of n) bends every partial's pitch together (vibrato, drift)"""
    y = np.zeros(n)
    for p in partials:
        i0 = max(0, int(np.floor(p['t0'] * sr)))
        i1 = min(n, int(np.ceil((p['t1'] + p['rel']) * sr)) + 1)
        if i1 <= i0 + 1:
            continue
        t = np.arange(i0, i1) / sr
        fq = np.asarray(p['freq'], float)
        am = np.asarray(p['amp'], float)
        f = 2.0 ** np.interp(t, fq[:, 0], np.log2(fq[:, 1]))
        if cents is not None:
            f = f * 2.0 ** (cents[i0:i1] / 1200.0)
        a = 10.0 ** (np.interp(t, am[:, 0], am[:, 1]) / 20.0)
        a *= np.clip((t - p['t0']) / max(p['rise'], 1e-5), 0.0, 1.0)
        a *= np.clip((p['t1'] + p['rel'] - t) / max(p['rel'], 1e-5), 0.0, 1.0)
        a[f >= 0.5 * sr * 0.98] = 0.0
        ph = p['ph'] + 2 * np.pi * (np.cumsum(f) - f[0]) / sr + 2 * np.pi * f[0] * (i0 / sr - p['t0'])
        y[i0:i1] += a * np.cos(ph)
    return y


def synth_noise(bands, times, levels, n, seed=0, sr=48000, pad=4096, hf=None):
    """noise shaped by band envelopes: white noise split by band_masks(), each band at unit RMS times its dB envelope
    (levels[b] at `times`, linear in dB between them). hf ([[Hz, dB], ...]): above its first frequency the masks
    follow it, a roll-off that takes the energy out (the band's level below it stays as fitted)"""
    r = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    nfft = 1 << int(np.ceil(np.log2(n + 2 * pad)))
    spec = np.fft.rfft(r.standard_normal(nfft))
    masks = band_masks(bands, nfft, sr)
    keep = np.ones(len(masks))
    if hf:
        f = np.fft.rfftfreq(nfft, 1.0 / sr)
        h = np.asarray(hf, float)
        shaped = masks * np.where(f > h[0, 0], 10.0 ** (np.interp(f, h[:, 0], h[:, 1]) / 20.0), 1.0)
        keep = np.sqrt((shaped ** 2).sum(1) / np.maximum((masks ** 2).sum(1), 1e-30))
        masks = shaped
    t = np.arange(n) / sr
    y = np.zeros(n)
    for b in range(len(bands)):
        lv = np.asarray(levels[b], float)
        if lv.max() < -140:
            continue
        u = np.fft.irfft(spec * masks[b], nfft)[pad:pad + n]
        u *= keep[b] / (np.sqrt(np.mean(u ** 2)) + 1e-20)
        y += u * 10.0 ** (np.interp(t, times, lv) / 20.0)
    return y


def hf_rolloff(x, sr=48000, points=(16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0, 23000.0,
                                    24000.0)):
    """a recording's (or a residual's) roll-off above points[0]: [[Hz, dB re points[0]], ...] from its long-term
    spectrum (1/6-octave means), made monotone (a partial up there does not lift the noise), floored at -80 dB"""
    m = np.asarray(x, float)
    m = m.mean(1) if m.ndim == 2 else m
    nper = int(min(4096, 1 << int(np.floor(np.log2(max(len(m), 256))))))
    f, P = signal.welch(m, sr, nperseg=nper)
    out, run, ref = [], 0.0, None
    for p in points:
        k = (f >= p * 2 ** (-1 / 12)) & (f <= min(p * 2 ** (1 / 12), sr / 2))
        v = 10 * np.log10(float(np.mean(P[k])) + 1e-30) if k.any() else -300.0
        ref = v if ref is None else ref
        run = min(run, v - ref)
        out.append([float(p), round(max(run, -80.0), 1)])
    return out


def finish(y, sr=48000, fade_s=0.002, peak_db=-1.0, limit=True):
    """a render's edges: a half-cosine fade-in over fade_s (it starts at zero) and, when limit and its peak is over
    peak_db dBFS, a gain down to it (no clipping, no limiter colour)"""
    y = np.array(y, dtype=float)
    k = min(len(y), int(round(fade_s * sr)))
    if k > 1:
        ramp = np.sin(0.5 * np.pi * np.arange(k) / k) ** 2
        y[:k] *= ramp[:, None] if y.ndim == 2 else ramp
    pk = float(np.abs(y).max()) if y.size else 0.0
    if limit and pk > 10 ** (peak_db / 20):
        y *= 10 ** (peak_db / 20) / pk
    return y


def synth_atoms(transients, n, sr=48000):
    """strike transients: each a start time and damped sinusoids [f Hz, tau s, amp, phase] that start at full
    amplitude (a click is a burst of very short modes)"""
    y = np.zeros(n)
    for tr in transients:
        i0 = int(round(tr['t'] * sr))
        for f, tau, a, ph in tr['atoms']:
            m = min(n - max(i0, 0), int(7 * tau * sr) + 1)
            if m <= 0:
                continue
            k = np.arange(m) + max(0, -i0)
            y[max(i0, 0):max(i0, 0) + m] += a * np.exp(-k / (tau * sr)) * np.cos(2 * np.pi * f * k / sr + ph)
    return y


# ================================================================================================ analysis
def _pencil(y, sr=SR, max_atoms=24, energy=0.998, max_t60=0.02, clamp=True, ridge=1e-4):
    """damped sinusoids [f, tau, amp, phase] that rebuild the segment y (matrix pencil poles, real least squares)"""
    L = len(y)
    if L < 16 or not np.any(y):
        return []
    P = L // 3
    Y = np.lib.stride_tricks.sliding_window_view(y, P + 1)
    _, sv, Vh = np.linalg.svd(Y, full_matrices=False)
    cum = np.cumsum(sv ** 2) / (np.sum(sv ** 2) + 1e-30)
    M = int(min(2 * max_atoms, np.searchsorted(cum, energy) + 1, len(sv) - 1))
    V = Vh[:M].T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    z = z[np.angle(z) >= -1e-9]                       # one of each conjugate pair
    zmax = 10 ** (-3.0 / (max_t60 * sr))
    if clamp:                                           # long modes are shortened to max_t60
        z = np.where(np.abs(z) > zmax, z / np.abs(z) * zmax, z)
    else:                                               # or dropped: only modes that die inside the window
        z = z[np.abs(z) <= zmax]
    z = z[np.abs(z) > 1e-3]
    mag = np.abs(z)
    w = np.abs(np.angle(z))
    # dedupe near-identical poles
    order = np.argsort(w)
    w, mag = w[order], mag[order]
    keep = np.ones(len(w), bool)
    for i in range(1, len(w)):
        if abs(w[i] - w[i - 1]) < 1e-4 and abs(mag[i] - mag[i - 1]) < 1e-3:
            keep[i] = False
    w, mag = w[keep], mag[keep]
    k = np.arange(L)
    cols, meta = [], []
    for wi, mi in zip(w, mag):
        d = mi ** k
        if wi < 1e-6 or wi > np.pi - 1e-6:
            cols.append(d * np.cos(wi * k))
            meta.append((wi, mi, 'r'))
        else:
            cols += [d * np.cos(wi * k), d * np.sin(wi * k)]
            meta += [(wi, mi, 'c'), None]
    if not cols:
        return []
    B = np.array(cols).T
    lam = ridge * np.trace(B.T @ B) / B.shape[1]          # a little ridge: no huge cancelling pairs
    coef = np.linalg.solve(B.T @ B + lam * np.eye(B.shape[1]), B.T @ y)
    atoms, j = [], 0
    for m in meta:
        if m is None:
            j += 1
            continue
        wi, mi, kind = m
        if kind == 'r':
            a, ph = coef[j], 0.0
            if a < 0:
                a, ph = -a, np.pi
            j += 1
        else:
            c, s_ = coef[j], coef[j + 1]
            a, ph = np.hypot(c, s_), np.arctan2(-s_, c)
            j += 1
        tau = -1.0 / (sr * np.log(mi))
        atoms.append([wi * sr / (2 * np.pi), tau, a, ph])
    atoms.sort(key=lambda q: -q[2] ** 2 * q[1])
    return atoms[:max_atoms]


def _transients(res, onsets, sr, peak, pre=0.0005, rel_db=-40.0, max_strikes=16, split=400.0):
    """matrix-pencil atoms for the residual at each strike whose residual is not negligible, in two bands: above
    `split` Hz a 6 ms window at the full rate (the click), below it a 60 ms window at 1/8 rate (the body thump)"""
    sos = signal.butter(4, split, 'low', fs=sr, output='sos')
    lo = signal.sosfiltfilt(sos, res)
    hi = res - lo
    D = 8
    out = []
    for o in sorted(onsets, key=lambda o: -o['strength'])[:max_strikes]:
        i0 = max(0, int(round((o['t'] - pre) * sr)))
        atoms = []
        for band, win, rate, na, kw, ok in (
                (hi, 0.008, 1, 24, {'max_t60': 0.02, 'clamp': True}, lambda f: f >= 0.6 * split),
                (lo, 0.064, D, 6, {'max_t60': 0.096, 'clamp': False, 'ridge': 1e-3}, lambda f: 40.0 <= f <= 1.5 * split)):
            seg = band[i0:i0 + int(win * sr)]
            if len(seg) < 32 or np.abs(seg).max() < peak * 10 ** (rel_db / 20):
                continue
            got = [q for q in _pencil(seg[::rate], sr / rate, max_atoms=na, **kw) if ok(q[0])]
            if not got:
                continue
            # keep them only if they explain the strike (analysis by synthesis): the residual energy must drop over
            # the window and the ring-out after it (atoms must not add energy where the recording has none)
            ext = band[i0:i0 + int((win * 2 + 0.02) * sr)]
            fit_ = synth_atoms([{'t': 0.0, 'atoms': got}], len(ext), sr)
            if ((ext - fit_) ** 2).sum() < 0.7 * (ext ** 2).sum():
                atoms += got
        if atoms:
            out.append({'t': round(i0 / sr, 6),
                        'atoms': [[round(f, 2), round(tau, 7), round(a, 7), round(ph, 4)] for f, tau, a, ph in atoms]})
    out.sort(key=lambda t: t['t'])
    return out


def _frames(x, W, hop):
    """zero-phase windowed frames centred at k * hop (k = 0..K-1), x2 zero padded rfft: (spectrum, K, N)"""
    n = len(x)
    K = n // hop + 2
    xp = np.concatenate([np.zeros(W // 2), x, np.zeros(W // 2 + 2 * hop)])
    fr = np.lib.stride_tricks.sliding_window_view(xp, W)[::hop][:K]
    w = signal.get_window('blackman', W)
    N = 2 * W
    buf = np.zeros((len(fr), N))
    h = W // 2
    buf[:, :W - h] = fr[:, h:] * w[h:]
    buf[:, N - h:] = fr[:, :h] * w[:h]
    return np.fft.rfft(buf, axis=1), w.sum(), N


def _peaks(X, wsum, N, sr, dyn_db=80.0, prom_db=6.0, max_peaks=60, fmin=30.0, ref_db=None):
    """per frame: arrays (freq Hz, amp linear, phase) of the spectral peaks that pass the thresholds (dyn_db under
    the loudest bin, or under `ref_db` when given: a residual pass keeps the original's reference)"""
    mag = np.abs(X)
    db = _db(mag)
    gmax = db.max() if ref_db is None else ref_db
    pw = mag ** 2
    local = 10 * np.log10(np.maximum(uniform_filter1d(pw, size=33, axis=1, mode="nearest"), 0) + 1e-40)
    is_pk = np.zeros_like(db, bool)
    is_pk[:, 1:-1] = (db[:, 1:-1] > db[:, :-2]) & (db[:, 1:-1] >= db[:, 2:])
    fmax_row = db.max(axis=1, keepdims=True)
    ok = is_pk & (db > gmax - dyn_db) & (db > fmax_row - dyn_db) & (db > local + prom_db)
    kmin = int(np.ceil(fmin * N / sr))
    ok[:, :kmin] = False
    ok[:, -2:] = False
    phase = np.angle(X)
    # the local noise floor: the median of blocks of 32 bins, interpolated (narrow peaks do not move a median)
    B = 32
    nb_ = db.shape[1] // B
    med = np.median(db[:, :nb_ * B].reshape(len(db), nb_, B), axis=2)
    cen_ = np.arange(nb_) * B + B / 2
    floor = np.array([np.interp(np.arange(db.shape[1]), cen_, row) for row in med])
    out = []
    for k in range(len(db)):
        b = np.nonzero(ok[k])[0]
        if len(b) > max_peaks:
            b = b[np.argsort(db[k, b])[::-1][:max_peaks]]
            b.sort()
        if not len(b):
            out.append((np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)))
            continue
        a, c, e = db[k, b - 1], db[k, b], db[k, b + 1]
        den = a - 2 * c + e
        p = np.where(np.abs(den) > 1e-12, 0.5 * (a - e) / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
        p = np.clip(p, -0.5, 0.5)
        f = (b + p) * sr / N
        amp = 10 ** ((c - 0.25 * (a - e) * p) / 20) * 2.0 / wsum
        nb = np.where(p >= 0, b + 1, b - 1)
        d = np.angle(np.exp(1j * (phase[k, nb] - phase[k, b])))
        ph = phase[k, b] + np.abs(p) * d
        out.append((f, amp, ph, c - floor[k, b]))
    return out


def _sinusoidal(tr, W, sr, snr_db=10.0):
    """is this track a partial (not a chain of noise peaks)? A noise peak stands only ~5-9 dB above the median of
    its neighbourhood (an exponential power distribution almost never reaches 16x its median); a partial, even a
    short chirpy one in a pop, stands 15-50 dB above it. Energy-weighted over the track."""
    a2 = tr['a'] ** 2
    return float((a2 * tr['c']).sum() / (a2.sum() + 1e-30)) >= snr_db


def _track(peaks, hop, sr, W, max_gap=2, rel_tol=0.025):
    """join peaks into partial tracks; returns a list of dicts with contiguous frame arrays k (frame index), f, a, ph"""
    df_min = 1.5 * sr / W
    active, done = [], []
    for k, (fs, amps, phs, cvs) in enumerate(peaks):
        used = np.zeros(len(fs), bool)
        if active and len(fs):
            pred = np.array([t['f'][-1] + t['slope'] * (k - t['k'][-1]) for t in active])
            tol = np.maximum(df_min, rel_tol * pred)
            D = np.abs(fs[None, :] - pred[:, None])
            lastdb = np.array([20 * np.log10(t['a'][-1] + 1e-20) for t in active])
            cost = D / tol[:, None] + 0.01 * np.abs(_db(amps)[None, :] - lastdb[:, None])
            cost[D > tol[:, None]] = np.inf
            ti, pi = np.nonzero(np.isfinite(cost))
            order = np.argsort(cost[ti, pi])
            took = np.zeros(len(active), bool)
            for o in order:
                a_, p_ = ti[o], pi[o]
                if took[a_] or used[p_]:
                    continue
                took[a_] = used[p_] = True
                t = active[a_]
                gap = k - t['k'][-1]
                if len(t['f']) >= 2 or gap > 1:
                    s = (fs[p_] - t['f'][-1]) / gap
                    t['slope'] = 0.5 * t['slope'] + 0.5 * s if len(t['f']) >= 2 else s
                t['k'].append(k)
                t['f'].append(fs[p_])
                t['a'].append(amps[p_])
                t['ph'].append(phs[p_])
                t['c'].append(cvs[p_])
            keep = []
            for i, t in enumerate(active):
                if not took[i] and k - t['k'][-1] > max_gap:
                    done.append(t)
                else:
                    keep.append(t)
            active = keep
        elif active:
            keep = []
            for t in active:
                (done if k - t['k'][-1] > max_gap else keep).append(t)
            active = keep
        for p_ in np.nonzero(~used)[0]:
            active.append({'k': [k], 'f': [fs[p_]], 'a': [amps[p_]], 'ph': [phs[p_]], 'c': [cvs[p_]], 'slope': 0.0})
    done += active
    out = []
    for t in done:
        k = np.array(t['k'])
        kk = np.arange(k[0], k[-1] + 1)
        f = np.interp(kk, k, t['f'])
        a = np.interp(kk, k, t['a'])
        cv = np.interp(kk, k, t['c'])
        ph = np.zeros(len(kk))
        ph[k - k[0]] = t['ph']
        have = np.zeros(len(kk), bool)
        have[k - k[0]] = True
        for j in np.nonzero(~have)[0]:      # across a gap: integrate the interpolated frequency (gaps are short)
            ph[j] = ph[j - 1] + 2 * np.pi * 0.5 * (f[j - 1] + f[j]) * hop / sr
        out.append({'k': kk, 'f': f, 'a': a, 'ph': ph, 'c': cv})
    return out


def _band_env(x, sr, centers, hop_s=0.001, max_smooth=0.030):
    """per band: dB envelope (RMS of the band-passed signal, smoothed for its bandwidth) at `hop_s` steps"""
    n = len(x)
    pad = 4096
    nfft = 1 << int(np.ceil(np.log2(n + 2 * pad)))
    spec = np.fft.rfft(np.concatenate([np.zeros(pad), x, np.zeros(nfft - n - pad)]))
    masks = band_masks(centers, nfft, sr)
    step = max(1, int(round(hop_s * sr)))
    idx = np.arange(0, n, step)
    env = np.zeros((len(centers), len(idx)))
    bw = centers * (2 ** (1 / 3) - 2 ** (-1 / 3))
    for b in range(len(centers)):
        v = np.fft.irfft(spec * masks[b], nfft)[pad:pad + n]
        L = max(2, int(np.clip(4.0 / bw[b], 0.0005, max_smooth) * sr))
        c = np.concatenate([[0.0], np.cumsum(v ** 2)])
        j = np.arange(n)
        trail = (c[j + 1] - c[np.maximum(j + 1 - L, 0)]) / L          # mean over (t - L, t]
        lead = (c[np.minimum(j + L, n)] - c[j]) / L                   # mean over [t, t + L)
        pw = np.minimum(trail, lead)                                   # edges stay sharp: no smear before a strike
        env[b] = 10 * np.log10(pw[idx] + 1e-24)
    return idx / sr, env


def _peak_env(x, sr, hold=0.0025):
    """causal peak envelope in dB: max |x| over the trailing `hold` seconds (rises exactly at a strike)"""
    from scipy.ndimage import maximum_filter1d
    L = max(1, int(hold * sr))
    e = maximum_filter1d(np.abs(x), size=L, origin=(L - 1) // 2, mode='constant')
    return _db(e, -240.0)


def _onsets(x, sr, thresh_db=6.0, min_gap=0.025):
    """strikes as dicts {'t' (s), 'rise' (s), 'strength'}: detected where the mean positive rise of 1 ms octave-band
    envelopes (each against its own minimum of the previous 8 ms) peaks above `thresh_db`, then timed on the causal
    peak envelope (the strike starts where it leaves its pre-strike floor; the rise ends 1 dB under the peak)"""
    cen = _band_centers(150.0, 12000.0, per_oct=1)
    t, env = _band_env(x, sr, cen, 0.001, max_smooth=0.008)
    top = env.max()
    env = np.maximum(env, top - 70)
    tot = 10 * np.log10((10 ** (env / 10)).sum(0) + 1e-24)
    w = 5
    pad = np.concatenate([np.repeat(env[:, :1], w, axis=1), env], axis=1)
    mins = np.min(np.stack([pad[:, j:j + env.shape[1]] for j in range(w)]), axis=0)
    rise = np.clip(env - mins, 0, None)
    loud = env > top - 60
    O = (rise * loud).sum(0) / np.maximum(loud.sum(0), 1)
    O[tot < top - 45] = 0
    O[((rise > thresh_db) & loud).sum(0) < 3] = 0          # a strike is broadband; a beating partial is not
    peaks, _ = signal.find_peaks(O, height=thresh_db, distance=max(1, int(min_gap / 0.001)))
    on_first = int(np.argmax(tot > tot.max() - 25))
    cands = sorted(set(peaks.tolist()) | {on_first})
    pe = _peak_env(x, sr)
    ms = sr // 1000
    out = []
    for i in cands:
        c = int(t[i] * sr)
        lo = max(0, c - 25 * ms)
        hi = min(len(x), c + 30 * ms)
        if hi - lo < 2:
            continue
        seg = pe[lo:hi]
        ipk = int(np.argmax(seg[max(0, c - 8 * ms - lo):])) + max(0, c - 8 * ms - lo)
        pk = float(seg[ipk])
        base = max(float(seg[:ipk + 1].min()), pk - 70.0)
        if pk - base < 3.0 and i != on_first:
            continue
        ib = int(np.argmin(seg[:ipk + 1]))
        thr = max(base + 0.25 * (pk - base), pk - 30.0)
        st = ib + int(np.argmax(seg[ib:ipk + 1] >= thr))
        ir = st + int(np.argmax(seg[st:ipk + 1] >= pk - 1.0))
        out.append({'t': float((lo + st) / sr), 'rise': float(max(3e-4, (ir - st) / sr)),
                    'strength': float(O[i]) if i in set(peaks.tolist()) else float(pk - base)})
    merged = []
    for o in sorted(out, key=lambda o: o['t']):
        if merged and o['t'] - merged[-1]['t'] < min_gap:
            if o['strength'] > merged[-1]['strength']:
                merged[-1] = o
            continue
        merged.append(o)
    return merged


def _split_at_strikes(tracks, onsets, W, hop, sr, jump_db=10.0, min_frames=3):
    """a track that jumps up by `jump_db` across a strike is two sounds (pre-strike junk or the previous strike's
    ring, then the new strike): cut it there, so the new part can start exactly at the strike"""
    half = W / 2 / sr
    out = []
    for tr in tracks:
        parts = [tr]
        for o in sorted(onsets, key=lambda o: o['t']):
            cur = parts[-1]
            times = cur['k'] * hop / sr
            if not (times[0] < o['t'] - 0.5 * half and times[-1] > o['t'] + 0.55 * half):
                continue
            adb = 20 * np.log10(cur['a'] + 1e-20)
            before = adb[(times >= o['t'] - 2 * half) & (times < o['t'] - 0.5 * half)]
            after = adb[(times >= o['t'] + 0.55 * half) & (times < o['t'] + 0.55 * half + 2 * half)]
            if len(before) and len(after) and after.max() - before.max() > jump_db:
                cut_a = times < o['t'] - 0.5 * half
                cut_b = times >= o['t'] - 0.5 * half
                a = {key: cur[key][cut_a] for key in ('k', 'f', 'a', 'ph', 'c')}
                b = {key: cur[key][cut_b] for key in ('k', 'f', 'a', 'ph', 'c')}
                parts = parts[:-1] + ([a] if len(a['k']) >= min_frames else []) + [b]
        out += parts
    # a track that lives only in the smear zone around a strike is the strike's pre-echo, not a partial
    zones = [(o['t'] - 1.1 * half, o['t'] + 0.55 * half) for o in onsets]

    def smear(t):
        tt, e = t['k'] * hop / sr, t['a'] ** 2
        return any(e[(tt >= a) & (tt <= b)].sum() >= 0.9 * e.sum() and tt[-1] <= b for a, b in zones)
    return [t for t in out if not smear(t)]


def _fix_onsets(tracks, onsets, W, hop, sr):
    """start every track born in the smear zone of a strike exactly at the strike; replace its smeared first frames
    by the line through its first clean frames"""
    half = W / 2 / sr
    tracks = _split_at_strikes(tracks, onsets, W, hop, sr)
    for tr in tracks:
        times = tr['k'] * hop / sr
        tb = times[0]
        cands = [o for o in onsets if tb - 0.25 * half <= o['t'] <= tb + 1.05 * half]
        tr['t0'], tr['rise'] = None, None
        if not cands:
            continue
        o = max(cands, key=lambda o: o['strength'])
        to = o['t']
        adb = 20 * np.log10(tr['a'] + 1e-20)
        clean = times >= to + 0.55 * half
        sel = clean & (times <= to + 0.55 * half + max(4 * hop / sr, half))
        mx = adb.max()
        if sel.sum() >= 2:
            s, c = np.polyfit(times[sel] - to, adb[sel], 1)
            a0 = float(np.clip(c, mx - 40, mx + 6))
            line = a0 + s * (times - to)
            smear = (times < to + 0.55 * half)
            adb = np.where(smear, np.minimum(line, mx + 6), adb)
        elif clean.any():
            a0 = float(adb[clean][0])
            adb = np.where(times < to + 0.55 * half, a0, adb)
        else:
            a0 = float(mx)
            adb = np.full_like(adb, a0)
        keep = times > to
        if not keep.any():
            tr['k'] = tr['k'][-1:]
            keep = np.ones(1, bool)
            adb = adb[-1:]
            for key in ('f', 'a', 'ph', 'c'):
                tr[key] = tr[key][-1:]
        else:
            for key in ('k', 'f', 'ph', 'c'):
                tr[key] = tr[key][keep]
            adb = adb[keep]
        tr['a'] = 10 ** (adb / 20)
        tr['t0'], tr['rise'], tr['a0'] = to, o['rise'], 10 ** (a0 / 20)
    return tracks


def _mq_synth(tracks, n, hop, sr):
    """phase-accurate re-synthesis of the tracks (cubic phase between frames, linear amplitude) for the residual"""
    y = np.zeros(n + 4 * hop)
    H = hop
    tt = np.arange(H, dtype=float)
    for tr in tracks:
        c = tr['k'] * hop
        w = 2 * np.pi * tr['f'] / sr
        th = tr['ph']
        a = tr['a']
        if len(c) >= 2:
            th0, th1, w0, w1 = th[:-1], th[1:], w[:-1], w[1:]
            M = np.round(((th0 + w0 * H - th1) + (w1 - w0) * H / 2) / (2 * np.pi))
            e = th1 - th0 - w0 * H + 2 * np.pi * M
            al = 3 / H ** 2 * e - (w1 - w0) / H
            be = -2 / H ** 3 * e + (w1 - w0) / H ** 2
            theta = th0[:, None] + w0[:, None] * tt + al[:, None] * tt ** 2 + be[:, None] * tt ** 3
            amp = a[:-1, None] + (a[1:] - a[:-1])[:, None] * tt / H
            seg = (amp * np.cos(theta)).ravel()
            s0 = c[0]
            lo = max(0, s0)
            y[lo:s0 + len(seg)] += seg[lo - s0:]
        # head: from the birth (strike) or one hop before the first frame
        if tr.get('t0') is not None:
            sb = int(round(tr['t0'] * sr))
            rise = max(1, int(round(tr['rise'] * sr)))
            pts_t = [sb, min(sb + rise, c[0]), c[0]]
            pts_a = [0.0, tr['a0'], a[0]]
        else:
            sb = c[0] - H
            pts_t, pts_a = [sb, c[0]], [0.0, a[0]]
        sb = max(0, sb)
        if c[0] > sb:
            s = np.arange(sb, c[0])
            amp = np.interp(s, pts_t, pts_a)
            y[s] += amp * np.cos(th[0] - w[0] * (c[0] - s))
        # tail: one hop of release
        s = np.arange(c[-1], c[-1] + H)
        s = s[(s >= 0) & (s < len(y))]
        y[s] += a[-1] * (1 - (s - c[-1]) / H) * np.cos(th[-1] + w[-1] * (s - c[-1]))
    return y[:n]


def _noise_ratio(r, sr, centers, t, onsets, W=2048, hop=240, attack=0.03):
    """per band and time (dB <= 0): how much of the residual's band energy is noise-like. Leftover partial energy
    (narrow peaks) must not become band noise (a tonal leftover rendered as hiss is a loud tell): the noise power is
    estimated from the median of the band's STFT bins (/ ln 2), which narrow peaks do not move. Around strikes
    (-2 ms .. `attack`) the full band energy is kept (the attack burst is noise)."""
    f, tt, S = signal.stft(r, sr, nperseg=W, noverlap=W - hop, boundary='zeros')
    P = np.abs(S) ** 2
    # whiten by a running median across frequency (follows a sloping noise spectrum, ignores narrow peaks), so the
    # median / mean test inside a band measures peakiness, not the slope of the band's noise
    P = P / (median_filter(P, size=(31, 1), mode='nearest') + 1e-30)
    lc = np.log2(np.asarray(centers, float))
    lf = np.log2(np.maximum(f, 1.0))
    rho = np.zeros((len(centers), P.shape[1]))
    for b in range(len(centers)):
        lo = lc[b - 1] if b else lc[0] - (lc[1] - lc[0])
        hi = lc[b + 1] if b + 1 < len(lc) else lc[b] + (lc[b] - lc[b - 1])
        sel = (lf >= 0.5 * (lo + lc[b])) & (lf < 0.5 * (lc[b] + hi))
        if sel.sum() < 5:
            rho[b] = 0.0
            continue
        mean = P[sel].mean(0) + 1e-30
        med = np.median(P[sel], axis=0) / np.log(2)
        rho[b] = 10 * np.log10(np.clip(med / mean, 1e-4, 1.0))
    rho = median_filter(rho, size=(1, 5), mode='nearest')
    out = np.array([np.interp(t, tt, rho[b]) for b in range(len(centers))])
    for o in onsets:
        out[:, (t >= o['t'] - 0.002) & (t <= o['t'] + attack)] = 0.0
    return out


_BIAS = {}


def _env_bias(centers, sr=SR):
    """per band, how far the smoothed dB envelope of steady noise sits under its true level (the min of two short
    means and the dB-domain median are biased low, more so in wide high bands with short windows): calibrated once
    on seeded white noise and added back"""
    key = tuple(np.round(centers, 3))
    if key not in _BIAS:
        w = np.random.default_rng(12345).standard_normal(2 * sr)
        t, env = _band_env(w, sr, centers, 0.001)
        env = median_filter(env, size=(1, 5), mode='nearest')
        nfft = 1 << int(np.ceil(np.log2(len(w) + 2 * 4096)))
        spec = np.fft.rfft(np.concatenate([np.zeros(4096), w, np.zeros(nfft - len(w) - 4096)]))
        masks = band_masks(centers, nfft, sr)
        true = np.array([10 * np.log10(np.mean(np.fft.irfft(spec * masks[b], nfft)[4096:4096 + len(w)] ** 2))
                         for b in range(len(centers))])
        mid = slice(int(0.1 * len(t)), int(0.9 * len(t)))
        _BIAS[key] = true - np.mean(env[:, mid], axis=1)
    return _BIAS[key]


def _residual_model(r, sr, centers, floor_db, onsets=(), tol_db=2.0, max_times=96):
    """band dB envelopes of the residual's noise on one adaptive time grid"""
    t, env = _band_env(r, sr, centers, 0.001)
    env = env + _noise_ratio(r, sr, centers, t, onsets)
    env = median_filter(env, size=(1, 5), mode='nearest') + _env_bias(centers, sr)[:, None]
    env = np.maximum(env, floor_db)
    live = env.max(axis=1) > floor_db + 1
    g = _simplify_multi(t, env[live], tol_db, max_times) if live.any() else np.array([0, env.shape[1] - 1])
    return {'bands': [round(float(c), 2) for c in centers], 'times': [round(float(v), 5) for v in t[g]],
            'db': [[round(float(v), 1) if v > floor_db + 0.05 else -150.0 for v in env[b, g]]
                   for b in range(len(centers))]}


def _partials_model(tracks, hop, sr, amp_tol_db, freq_tol_cents, max_amp_pts=64, max_freq_pts=48):
    """compress tracks to breakpoint partials (strongest first)"""
    out = []
    for tr in tracks:
        times = tr['k'] * hop / sr
        adb = 20 * np.log10(tr['a'] + 1e-20)
        f = tr['f']
        if len(f) >= 5:
            f = median_filter(f, size=5, mode='nearest')
        if tr.get('t0') is not None:
            t0, rise = tr['t0'], tr['rise']
            ta = np.concatenate([[t0], times])
            va = np.concatenate([[20 * np.log10(tr['a0'] + 1e-20)], adb])
            tf = np.concatenate([[t0], times])
            vf = np.concatenate([[f[0]], f])
            ph0 = tr['ph'][0] - 2 * np.pi * f[0] * (times[0] - t0)
        else:
            t0 = times[0] - hop / sr
            rise = hop / sr
            ta, va = np.concatenate([[t0], times]), np.concatenate([[adb[0]], adb])
            tf, vf = ta, np.concatenate([[f[0]], f])
            ph0 = tr['ph'][0] - 2 * np.pi * f[0] * hop / sr
        ia = _simplify(ta, va, amp_tol_db, max_amp_pts)
        cents = 1200 * np.log2(np.maximum(vf, 1.0))
        # frequency estimates get noisy as a partial sinks toward the floor: the tolerance widens 2x per 20 dB
        rel = np.clip((va.max() - np.interp(tf, ta, va)) / 20.0, 0.0, 4.0)
        jf = _simplify(tf, cents, freq_tol_cents * 2.0 ** rel, max_freq_pts)
        out.append({'t0': round(float(t0), 5), 'rise': round(float(max(rise, 2e-4)), 5),
                    't1': round(float(times[-1]), 5), 'rel': round(float(hop / sr), 5),
                    'ph': round(float(np.angle(np.exp(1j * ph0))), 3),
                    'freq': [[round(float(tf[j]), 5), round(float(vf[j]), 2)] for j in jf],
                    'amp': [[round(float(ta[i]), 5), round(float(va[i]), 1)] for i in ia]})
    out.sort(key=lambda p: -max(v for _, v in p['amp']))
    return out


def _lsd(x, y, sizes=(256, 1024, 4096), range_db=60.0):
    """multi-resolution log-spectral distance (dB): mean |dB difference| of STFT magnitudes over the cells where
    either is within `range_db` of the louder signal's peak (differences capped at 40 dB), averaged over sizes"""
    n = max(len(x), len(y))
    x = np.pad(x, (0, n - len(x)))
    y = np.pad(y, (0, n - len(y)))
    ds = []
    for s in sizes:
        if n < s:
            continue
        _, _, A = signal.stft(x, nperseg=s, noverlap=s - s // 4, boundary='zeros')
        _, _, B = signal.stft(y, nperseg=s, noverlap=s - s // 4, boundary='zeros')
        a, b = _db(A), _db(B)
        fl = max(a.max(), b.max()) - range_db
        a, b = np.maximum(a, fl), np.maximum(b, fl)
        act = (a > fl) | (b > fl)
        if act.any():
            ds.append(float(np.minimum(np.abs(a - b)[act], 40).mean()))
    return float(np.mean(ds)) if ds else 0.0


def _fit_window(x, sr, W, onsets, width, opts):
    return _analyze(x, sr, W, onsets, width, opts)['model']


def _analyze(x, sr, W, onsets, width, opts):
    """the model at one window size plus the internals: {'model', 'det' (phase-exact partials), 'res' (x - det),
    'atoms' (the strike transients' signal), 'tracks'}"""
    hop = max(32, W // 8)
    X, wsum, N = _frames(x, W, hop)
    pk = _peaks(X, wsum, N, sr, dyn_db=opts['dyn_db'], prom_db=opts['prom_db'], max_peaks=opts['max_peaks'])
    tracks = _track(pk, hop, sr, W)
    if not tracks:
        tracks = []
    # overlapping frames make a noise peak persist for about a window: a partial must outlive 1.25 windows
    min_frames = max(3, int(np.ceil(max(opts['min_ms'] / 1000, 1.25 * W / sr) * sr / hop)))
    gpk = max([t['a'].max() for t in tracks], default=1e-9)
    tracks = [t for t in tracks if len(t['k']) >= min_frames and t['a'].max() > gpk * 10 ** (-opts['keep_db'] / 20)
              and _sinusoidal(t, W, sr, opts['snr_db'])]
    tracks.sort(key=lambda t: -float((t['a'] ** 2).sum()))
    tracks = tracks[:opts['max_partials']]
    tracks = _fix_onsets(tracks, onsets, W, hop, sr)
    det = _mq_synth(tracks, len(x), hop, sr)
    res = x - det
    ref_db = float(_db(np.abs(X).max()))
    for _ in range(opts['passes'] - 1):
        # analysis by synthesis: partials hidden under stronger ones (doublet partners, beating sidebands, weak
        # high partials) show up in the residual; track them there and add them
        room = opts['max_partials'] - len(tracks)
        if room <= 0:
            break
        R, _, _ = _frames(res, W, hop)
        more = _track(_peaks(R, wsum, N, sr, dyn_db=opts['dyn_db'], prom_db=opts['prom_db'],
                             max_peaks=opts['max_peaks'], ref_db=ref_db), hop, sr, W)
        more = [t for t in more if len(t['k']) >= min_frames and t['a'].max() > gpk * 10 ** (-opts['keep_db'] / 20)
                and _sinusoidal(t, W, sr, opts['snr_db'])]
        more.sort(key=lambda t: -float((t['a'] ** 2).sum()))
        more = _fix_onsets(more[:room], onsets, W, hop, sr)
        if not more:
            break
        d2 = _mq_synth(more, len(x), hop, sr)
        if ((res - d2) ** 2).sum() > 0.98 * (res ** 2).sum():
            break
        tracks += more
        det = det + d2
        res = x - det
    peak_db = float(_db(np.abs(x).max()))
    trans = _transients(res, onsets, sr, np.abs(x).max()) if opts['transients'] else []
    res2 = res - synth_atoms(trans, len(x), sr)
    centers = _band_centers()
    resid = _residual_model(res2, sr, centers, floor_db=peak_db - opts['res_range_db'], onsets=onsets)
    resid['hf'] = hf_rolloff(res2, sr, HF_POINTS)
    e_x = float((x ** 2).sum()) + 1e-20
    model = {'kind': KIND, 'version': VERSION, 'sr': SR, 'duration': round(len(x) / sr, 5),
             'onset': round(onsets[0]['t'], 5) if onsets else 0.0,
             'onsets': [round(o['t'], 5) for o in onsets], 'peak_db': round(peak_db, 2), 'width': round(width, 1),
             'partials': _partials_model(tracks, hop, sr, opts['amp_tol_db'], opts['freq_tol_cents']),
             'transients': trans, 'residual': resid,
             'analysis': {'window': W, 'hop': hop, 'sines_share_db': round(float(10 * np.log10(
                 max(1e-12, 1 - (res ** 2).sum() / e_x))), 2),
                 'residual_db': round(float(10 * np.log10((res ** 2).sum() / e_x + 1e-12)), 2)}}
    return {'model': model, 'det': det, 'res': res, 'atoms': res - res2, 'tracks': tracks}


def _side_times(onset, duration, steps=SIDE_STEPS):
    """keyframe times: 0, 10 ms before the onset, then SIDE_STEPS after it, and the end"""
    tk = [0.0, onset - 0.01, onset]
    t = onset
    for until, step in steps:
        while t + step <= onset + until and t + step < duration:
            t += step
            tk.append(t)
    tk.append(duration)
    return np.unique(np.round(np.clip(tk, 0.0, duration), 5))


def side_model(st, duration=None, bands=SIDE_BANDS, tol=SIDE_TOL):
    """the side of a stereo source ((n, 2) at 48 kHz) per octave band at keyframes (_side_times after the source's
    onset, space.active_span, then dropped where a line through the neighbours stays within `tol`): 'pan' =
    Re(S M*) / |M|^2 (the side's part coherent with the mid, signed) and 'diffuse' = sqrt((|S|^2 - Re(S M*)^2 / |M|^2)
    / |M|^2) (the rest re the mid), from the STFT frames under a tent from the previous keyframe to the next (what
    linear interpolation between keyframes spreads it over); a keyframe SIDE_QUIET_DB under its band's loudest (a
    pre-roll, a faded tail) takes the band's whole-sound values, a band with hardly any energy the whole sound's"""
    from .space import active_span, side_powers
    st = np.asarray(st, np.float64)
    duration = len(st) / SR if duration is None else float(duration)
    P = side_powers(st, bands)
    t, PM, PS, C = P['t'], P['PM'], P['PS'], P['C']
    tk = _side_times(active_span(st)[0] / SR, duration)
    J = len(tk)
    edges = np.concatenate([[tk[0] - 0.01], tk, [tk[-1] + 0.01]])
    sPM, sPS, sC = np.zeros((len(bands), J)), np.zeros((len(bands), J)), np.zeros((len(bands), J))
    for j in range(J):
        a, c_, b = edges[j], edges[j + 1], edges[j + 2]
        w = np.clip(np.minimum((t - a) / max(c_ - a, 1e-6), (b - t) / max(b - c_, 1e-6)), 0.0, 1.0)
        if not w.any():                                                    # no frame under its tent: the nearest
            w[int(np.argmin(np.abs(t - c_)))] = 1.0
        sPM[:, j], sPS[:, j], sC[:, j] = PM @ w, PS @ w, C @ w

    def gains(pm, ps, c):
        pm = np.maximum(pm, 1e-30)
        return np.clip(c / pm, -1.5, 1.5), np.clip(np.sqrt(np.maximum(ps - c ** 2 / pm, 0.0) / pm), 0.0, 2.0)
    pan, dif = gains(sPM, sPS, sC)
    tot = PM.sum(1)
    gp, gd = gains(PM.sum(), PS.sum(), C.sum())                            # the whole sound, all bands
    for b in range(len(bands)):
        if tot[b] < tot.max() * 1e-6:
            bp, bd = gp, gd
        else:
            bp, bd = gains(tot[b], PS[b].sum(), C[b].sum())
        quiet = sPM[b] < sPM[b].max() * 10 ** (-SIDE_QUIET_DB / 10) if tot[b] >= tot.max() * 1e-6 else np.ones(J, bool)
        pan[b, quiet], dif[b, quiet] = bp, bd
    keep = _simplify_multi(tk, np.vstack([pan, dif]), tol)
    return {'bands': [float(b) for b in bands], 'times': [round(float(v), 5) for v in tk[keep]],
            'pan': [[round(float(v), 3) for v in row[keep]] for row in pan],
            'diffuse': [[round(float(v), 3) for v in row[keep]] for row in dif]}


def morph_side(sa, sb, w, map_a=None, map_b=None):
    """the side of a morph between two sines models' sides (fit.auto morph): each side's keyframe times through its
    time map (map_a / map_b: arrays -> arrays, the morph's anchors; None = unchanged), the gains interpolated on the
    union of the two grids; a model without a side counts as a centred one (all gains 0). None when neither has one"""
    if not sa and not sb:
        return None
    bands = (sa or sb)['bands']
    if sa and sb and list(sa['bands']) != list(sb['bands']):
        raise ValueError('sines sides with different bands cannot be morphed')
    grids = []
    for s, f in ((sa, map_a), (sb, map_b)):
        if s:
            t = np.asarray(s['times'], float)
            grids.append((np.asarray(f(t) if f else t, float), np.asarray(s['pan'], float),
                          np.asarray(s['diffuse'], float)))
        else:
            grids.append(None)
    tt = np.unique(np.round(np.concatenate([g[0] for g in grids if g is not None]), 5))

    def at(g, i):
        return np.zeros((len(bands), len(tt))) if g is None else np.array([np.interp(tt, g[0], r) for r in g[i]])
    pan = (1 - w) * at(grids[0], 1) + w * at(grids[1], 1)
    dif = (1 - w) * at(grids[0], 2) + w * at(grids[1], 2)
    return {'bands': [float(b) for b in bands], 'times': [float(v) for v in tt],
            'pan': [[round(float(v), 3) for v in row] for row in pan],
            'diffuse': [[round(float(v), 3) for v in row] for row in dif]}


def fit(x, sr=SR, windows=WINDOWS, max_partials=160, dyn_db=80.0, keep_db=60.0, prom_db=4.0, max_peaks=100, min_ms=15.0,
        amp_tol_db=1.0, freq_tol_cents=1.0, res_range_db=140.0, transients=True, passes=2, snr_db=10.0, source=None):
    """a sines model of `x` (a path or an array at `sr`); every size in `windows` is tried and the model whose
    re-render (seed 0) is closest to `x` by multi-resolution log-spectral distance is kept"""
    if isinstance(x, str) and source is None:
        import os
        source = os.path.basename(x)
    x, width, st = _load(x, sr)
    opts = {'dyn_db': dyn_db, 'keep_db': keep_db, 'prom_db': prom_db, 'max_peaks': max_peaks, 'min_ms': min_ms,
            'max_partials': max_partials, 'amp_tol_db': amp_tol_db, 'freq_tol_cents': freq_tol_cents,
            'res_range_db': res_range_db, 'transients': transients, 'passes': passes, 'snr_db': snr_db}
    onsets = _onsets(x, SR)
    best, tried = None, {}
    for W in windows:
        if W > 2 * len(x) and best is not None:
            continue
        m = _fit_window(x, SR, W, onsets, width, opts)
        d = _lsd(x, render(m))
        tried[str(W)] = round(d, 2)
        if best is None or d < best[0]:
            best = (d, m)
    model = best[1]
    if st is not None and width > -60.0:
        model['side'] = side_model(st, model['duration'])
    model['coherent'] = True
    model['analysis']['lsd_db'] = round(best[0], 2)
    model['analysis']['tried'] = tried
    model['analysis']['onset_rise_ms'] = [round(o['rise'] * 1000, 2) for o in onsets]
    if source:
        model['source'] = source
    model['summary'] = summary(model)
    return model


# ================================================================================================ reading a model
def _t60(p):
    """seconds for the partial to fall 60 dB, from the line through its amplitude breakpoints after the peak"""
    am = np.asarray(p['amp'], float)
    i = int(np.argmax(am[:, 1]))
    tail = am[i:]
    if len(tail) < 2 or tail[-1, 0] - tail[0, 0] < 0.01:
        return None
    s = np.polyfit(tail[:, 0] - tail[0, 0], tail[:, 1], 1)[0]
    return None if s >= -0.5 else float(-60.0 / s)


def partial_freq(p):
    """a partial's energy-weighted mean frequency (Hz) from its breakpoints"""
    fq = np.asarray(p['freq'], float)
    am = np.asarray(p['amp'], float)
    t = np.linspace(am[0, 0], max(am[-1, 0], am[0, 0] + 1e-4), 200)
    w = 10 ** (np.interp(t, am[:, 0], am[:, 1]) / 10)
    f = 2 ** np.interp(t, fq[:, 0], np.log2(fq[:, 1]))
    return float((w * f).sum() / (w.sum() + 1e-30))


def summary(model):
    """numbers a person (or a recipe) can read: partial frequencies, ratios, decays, inharmonicity, levels"""
    ps = model['partials']
    if not ps:
        return {'partials': 0}
    pk = np.array([max(v for _, v in p['amp']) for p in ps])
    fm = np.array([partial_freq(p) for p in ps])
    life = np.array([p['t1'] - p['t0'] for p in ps])
    strong = (pk > pk.max() - 30) & (life >= min(0.1, 0.5 * life.max()))   # a short strike thump is not the pitch
    if not strong.any():
        strong = pk > pk.max() - 30
    f_low = float(fm[strong].min())
    w = 10 ** (pk / 10)
    centroid = float(np.exp((w * np.log(fm)).sum() / w.sum()))
    top = np.argsort(-pk)[:8]
    ratios = fm[strong] / f_low
    harm_dev = float(np.median(np.abs(ratios - np.round(ratios)) / np.maximum(np.round(ratios), 1))) if strong.sum() > 1 else 0.0
    glide = []
    for p in ps[:8]:
        fq = np.asarray(p['freq'], float)
        glide.append(round(float(1200 * np.log2(fq[-1, 1] / fq[0, 1])), 1))
    return {'partials': len(ps), 'f_low': round(f_low, 2), 'centroid': round(centroid, 2),
            'f_range': [round(float(fm.min()), 1), round(float(fm.max()), 1)],
            'top': [{'f': round(float(fm[i]), 2), 'db': round(float(pk[i]), 1),
                     't60': None if _t60(ps[i]) is None else round(_t60(ps[i]), 3)} for i in top],
            'ratios': [round(float(r), 3) for r in sorted(ratios)[:12]],
            'inharmonicity': round(harm_dev, 4), 'glide_cents': glide}


def describe(model):
    """one line of words about a model"""
    s = model.get('summary') or summary(model)
    if not s.get('partials'):
        return 'no partials (noise only)'
    a = model['analysis']
    t60 = [t['t60'] for t in s['top'] if t['t60']]
    kind = 'harmonic' if s['inharmonicity'] < 0.01 else ('slightly inharmonic' if s['inharmonicity'] < 0.04 else 'inharmonic')
    gl = max(s['glide_cents'], key=abs) if s['glide_cents'] else 0
    words = (f"strike at {model['onset'] * 1000:.0f} ms ({len(model['onsets'])} strike(s)), {s['partials']} partials "
             f"{s['f_range'][0]:.0f}-{s['f_range'][1]:.0f} Hz, lowest strong {s['f_low']:.0f} Hz, {kind} "
             f"(ratios {', '.join(f'{r:.2f}' for r in s['ratios'][:5])}), centroid {s['centroid']:.0f} Hz")
    if t60:
        words += f", T60 {min(t60):.2f}-{max(t60):.2f} s"
    if abs(gl) > 50:
        words += f", glide {gl:+.0f} cents"
    words += (f"; residual {a['residual_db']:.1f} dB of the energy; window {a['window']}, "
              f"fit {a.get('lsd_db', float('nan')):.1f} dB LSD")
    return words


# ================================================================================================ rendering
def _smooth_wander(n, r, rate=1.5, sr=SR):
    """unit-RMS slow random curve (sum of a few low sinusoids with seeded frequencies and phases)"""
    t = np.arange(n) / sr
    y = np.zeros(n)
    for _ in range(4):
        y += np.sin(2 * np.pi * rate * r.uniform(0.2, 1.0) * t + r.uniform(0, 2 * np.pi))
    return y / 2.0


def render(model, seed=0, knobs=None, **kw):
    """the sound of a model (float64, 48 kHz, mono unless knobs['stereo']); knobs as in the module docstring"""
    if model.get('kind') != KIND:
        raise ValueError(f"not a sines model: kind={model.get('kind')!r}")
    k = dict(DEFAULT_KNOBS)
    k.update(knobs or {})
    k.update(kw)
    unknown = set(k) - set(DEFAULT_KNOBS)
    if unknown:
        raise ValueError(f'unknown knobs: {sorted(unknown)}')
    sr = model.get('sr', SR)
    r = np.random.default_rng((int(seed) * 7919 + 17) & 0xFFFFFFFF)
    on = float(model.get('onset', 0.0))
    s_, d_ = float(k['stretch']), float(k['decay'])
    ratio = 2.0 ** (float(k['transpose']) / 12.0)
    vary = float(k['vary'])
    summ = model.get('summary') or summary(model)
    f_ref = summ.get('centroid', 1000.0) or 1000.0
    f_low = summ.get('f_low', 100.0) or 100.0

    def T(t):
        t = np.asarray(t, float)
        return on + (t - on) * s_

    parts = list(model['partials'])
    if k['partials'] is not None:
        parts = parts[:int(k['partials'])]
    identity = (s_ == 1.0 and d_ == 1.0 and ratio == 1.0 and vary == 0.0 and k['brightness'] == 0.0
                and k['inharm'] == 0.0 and k['phases'] == 'fit')
    coherent = bool(model.get('coherent')) and vary > 0
    if coherent:                     # the jitter as smooth functions of log frequency (module doc)
        jf, jl, jd = vary_field(r), vary_field(r), vary_field(r)
    new = []
    for p in parts:
        if identity:
            new.append(p)
            continue
        fq = np.asarray(p['freq'], float).copy()
        am = np.asarray(p['amp'], float).copy()
        fmed = partial_freq(p)
        fac = ratio * (fmed / f_low) ** float(k['inharm']) if fmed > 0 else ratio
        if coherent:
            detune, jit_db = 8.0 * vary * jf(fmed), 2.0 * vary * jl(fmed)
            jit_dec = float(np.exp(0.15 * vary * jd(fmed)))
        else:
            detune = r.normal(0, 8.0 * vary)
            jit_db = r.normal(0, 2.0 * vary)
            jit_dec = float(np.exp(r.normal(0, 0.15 * vary)))
        ph = float(p['ph']) if k['phases'] == 'fit' else float(r.uniform(-np.pi, np.pi))
        fq[:, 1] = fq[:, 1] * fac * 2 ** (detune / 1200)
        am[:, 1] = am[:, 1] + jit_db + float(k['brightness']) * np.log2(max(fmed * ratio, 1.0) / f_ref)
        # decay: stretch the amplitude breakpoints after the peak about the peak
        ip = int(np.argmax(am[:, 1]))
        tp = am[ip, 0]
        dd = d_ * jit_dec
        am[ip + 1:, 0] = tp + (am[ip + 1:, 0] - tp) * dd
        t1 = tp + (float(p['t1']) - tp) * dd if p['t1'] > tp else float(p['t1'])
        new.append({'t0': float(T(p['t0'])), 'rise': float(p['rise']), 't1': float(T(t1)),
                    'rel': float(p['rel']) * max(1.0, s_),
                    'ph': ph, 'freq': np.column_stack([T(fq[:, 0]), fq[:, 1]]).tolist(),
                    'amp': np.column_stack([T(am[:, 0]), am[:, 1]]).tolist()})
    res = model['residual']
    bands = np.asarray(res['bands'], float)
    times = np.asarray(res['times'], float)
    lev = np.asarray(res['db'], float)
    if not identity or k['residual'] != 1.0:
        lev = lev.copy()
        if ratio != 1.0:
            lb = np.log2(bands)
            lev = np.array([np.interp(lb - np.log2(ratio), lb, lev[:, j]) for j in range(lev.shape[1])]).T
        lev += float(k['brightness']) * np.log2(bands * ratio / f_ref)[:, None]
        if vary:
            lev += r.normal(0, 1.5 * vary, size=(len(bands), 1))
        piv = on + 0.010
        times = np.where(times > piv, piv + (times - piv) * d_, times)
        times = T(times)
        if k['residual'] != 1.0:
            lev = lev + 20 * np.log10(max(float(k['residual']), 1e-6))
        lev = np.where(np.asarray(res['db'], float) <= -149, -150.0, lev)
    ends = [p['t1'] + p['rel'] for p in new] + [float(T(model['duration']))]
    ring = vary > 0 if k['ring'] is None else bool(k['ring'])
    n_id = int(round(model['duration'] * sr))
    n = int(np.ceil(max(ends) * sr)) + 1 if (d_ > 1.0 or s_ != 1.0 or ring) else n_id
    if ring and len(times):          # the noise fades out over RING_FADE_S past the recording's length (linear in dB)
        t_end = max(float(times[-1]), float(T(model['duration'])))
        n = max(n, int(np.ceil((t_end + RING_FADE_S) * sr)) + 1)
        times = np.append(times, t_end + RING_FADE_S)
        lev = np.column_stack([lev, np.full(len(lev), -150.0)])
    cents = None
    if k['vibrato'] or k['drift']:
        tt = np.arange(n) / sr
        cents = float(k['vibrato']) * np.sin(2 * np.pi * float(k['vibrato_rate']) * tt + r.uniform(0, 2 * np.pi))
        cents = cents * np.clip((tt - on) / 0.15, 0, 1)
        cents = cents + float(k['drift']) * _smooth_wander(n, r)
    y = synth_partials(new, n, sr, cents)
    trans = model.get('transients', [])
    if trans and (not identity or k['transient'] != 1.0):
        tg = float(k['transient'])
        out = []
        for tr in trans:             # a coherent model jitters a strike's atoms together: they were fitted jointly
            if coherent:             # and cancel each other's step at the onset (6 atoms of a sub drop's strike: amp
                d1, g1 = r.normal(0, 30.0 * vary), r.normal(0, 2.0 * vary)     # up to 1.5, their sum there 0.015;
                jit = [(d1, g1)] * len(tr['atoms'])                             # jittered apart, the takes clicked)
            else:
                jit = [(r.normal(0, 30.0 * vary), r.normal(0, 2.0 * vary)) for _ in tr['atoms']]
            out.append({'t': float(T(tr['t'])),
                        'atoms': [[f * ratio * 2 ** (dc / 1200), tau / ratio,
                                   a * tg * 10 ** ((dg + float(k['brightness'])
                                                    * np.log2(max(f * ratio, 20.0) / f_ref)) / 20), ph]
                                  for (f, tau, a, ph), (dc, dg) in zip(tr['atoms'], jit)]})
        trans = out
    y = y + synth_atoms(trans, n, sr)
    hf = res.get('hf')
    noise = synth_noise(bands, times, lev, n, seed, sr, hf=hf) if len(bands) else np.zeros(n)
    if k['transient'] != 1.0:
        tt = np.arange(n) / sr
        w = np.zeros(n)
        for o in model.get('onsets', [on]):
            to = float(T(o))
            w = np.maximum(w, np.clip(1 - np.abs(tt - to - 0.0075) / 0.0125, 0, 1) * (tt >= to - 0.002))
        noise = noise * (1 + (float(k['transient']) - 1) * w)
    y = y + noise
    if k['gain_db']:
        y = y * 10 ** (float(k['gain_db']) / 20)
    limit = _limited(model)
    fade = fade_in_s(model)
    if k['stereo']:
        sd = model.get('side')
        if sd:                       # the side made from the mid, per band (module doc); keyframes move like the noise
            from .space import shaped_side
            st = np.asarray(sd['times'], float)
            piv = on + 0.010
            st = T(np.where(st > piv, piv + (st - piv) * d_, st))
            sign = -1.0 if np.random.default_rng((int(seed) * 31 + 7) & 0xFFFFFFFF).uniform() < 0.5 else 1.0
            side = shaped_side(y, np.asarray(sd['bands'], float) * ratio, st, sd['pan'], sd['diffuse'], sign, sr)
        else:                        # a model fitted before 0.4: its old side (module doc)
            w_old = 10 ** (model.get('width', -120.0) / 20)
            side = synth_noise(bands, times, lev, n, seed + 1000003, sr, hf=hf) * w_old
            side *= 10 ** (float(k['gain_db']) / 20)     # 0.3's two steps exactly (one differs in the last bits)
        side = side * float(k['width'])
        return finish(np.column_stack([y + side, y - side]), sr, fade, limit=limit)
    return finish(y, sr, fade, limit=limit)


def vary_field(r, parts=6):
    """a smooth random function of frequency (Hz -> unit-variance value): `parts` cosines over log2 frequency of
    0.25-1.25 cycles per octave with random phases. A coherent model's vary draws its jitter from it: neighbouring
    partials (a harmonic series' upper partials) move together and keep their ratios"""
    c, p = r.uniform(0.25, 1.25, parts), r.uniform(0.0, 2 * np.pi, parts)
    return lambda f: float(np.sqrt(2.0 / parts) * np.cos(2 * np.pi * np.log2(max(float(f), 20.0)) * c + p).sum())


def fade_in_s(model):
    """the render's fade-in: FADE_IN_S, shortened (to 0.25 ms at least) when the first strike comes earlier, so a clip
    cut into its own attack keeps the attack; 0 for a model fitted before 0.4 (no 'coherent'): it renders as it did"""
    if not model.get('coherent'):
        return 0.0
    return float(np.clip(float(model.get('onset', 1.0)) - 0.0005, 0.00025, FADE_IN_S))


def _limited(model):
    """is the render peak-limited to PEAK_DB? Not a hybrid's deterministic layer (the sum is what counts), not a model
    fitted before 0.4 (it renders as it did)"""
    return model.get('layer') != 'deterministic' and bool(model.get('coherent'))


# ================================================================================================ model -> code
def _lit(v):
    return json.dumps(v, separators=(',', ':'))


def to_code(model, name='sound'):
    """standalone Python (numpy only) that renders the model at its fit (render(model, seed) with default knobs)"""
    ps = model['partials']
    res = model['residual']
    s = model.get('summary') or summary(model)
    lines = [f'"""{name}: a sines model as code (earshot.forge.fit.sines v{model["version"]}).',
             '',
             f'{describe(model)}',
             '',
             'render(seed) -> 48 kHz mono float array. Partials: (start s, rise s, end s, release s, phase rad,',
             'frequency breakpoints [(t s, Hz)], amplitude breakpoints [(t s, dBFS)]). Residual: noise in 1/3-octave',
             'bands (Hz centres) with dB envelopes at TIMES."""',
             'import numpy as np',
             '',
             'SR = 48000',
             f'DURATION = {model["duration"]}',
             f'# lowest strong partial {s.get("f_low")} Hz; ratios to it: {s.get("ratios", [])[:8]}',
             'PARTIALS = [']
    for p in ps:
        lines.append(f"    ({p['t0']}, {p['rise']}, {p['t1']}, {p['rel']}, {p['ph']},")
        lines.append(f"     {_lit(p['freq'])},")
        lines.append(f"     {_lit(p['amp'])}),")
    lines.append(']')
    lines.append('# strike transients: (start s, [(f Hz, decay tau s, amplitude, phase rad), ...])')
    lines.append('TRANSIENTS = [')
    for tr in model.get('transients', []):
        lines.append(f"    ({tr['t']}, {_lit(tr['atoms'])}),")
    lines.append(']')
    lines.append(f"BANDS = {_lit(res['bands'])}")
    lines.append(f"HF = {_lit(res['hf']) if res.get('hf') else None}   # the noise above 16 kHz: the recording's "
                 "roll-off [Hz, dB] (None: flat, a model fitted before 0.3)")
    lines.append(f"TIMES = {_lit(res['times'])}")
    lines.append('LEVELS = [')
    for row in res['db']:
        lines.append(f'    {_lit(row)},')
    lines.append(']')
    lines.append(f"LIMIT = {_limited(model)}   # peak-limit the render to -1 dBFS")
    lines.append(f"FADE_S = {fade_in_s(model)!r}   # the fade-in (s): the render starts at zero")
    lines.append('')
    for fn in (band_masks, synth_partials, synth_atoms, synth_noise, finish):
        lines.append('')
        lines.append(inspect.getsource(fn).rstrip())
        lines.append('')
    lines += ['',
              'def render(seed=0):',
              '    n = int(round(DURATION * SR))',
              "    parts = [{'t0': p[0], 'rise': p[1], 't1': p[2], 'rel': p[3], 'ph': p[4], 'freq': p[5], 'amp': p[6]}",
              '             for p in PARTIALS]',
              "    trans = [{'t': t, 'atoms': a} for t, a in TRANSIENTS]",
              '    return finish(synth_partials(parts, n, SR) + synth_atoms(trans, n, SR)',
              '                  + synth_noise(BANDS, TIMES, LEVELS, n, seed, SR, hf=HF), SR, FADE_S, limit=LIMIT)',
              '',
              '',
              "if __name__ == '__main__':",
              '    import sys, wave',
              '    y = render(int(sys.argv[2]) if len(sys.argv) > 2 else 0)',
              '    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)',
              f"    with wave.open(sys.argv[1] if len(sys.argv) > 1 else '{name}.wav', 'wb') as w:",
              '        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())',
              '']
    return '\n'.join(lines)


def save(model, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(model, fh, separators=(',', ':'))
    return path


def load(path):
    with open(path, encoding='utf-8') as fh:
        m = json.load(fh)
    if m.get('kind') != KIND:
        raise ValueError(f'{path} is not a sines model')
    return m
