"""Modal fitter: a struck / ringing recording -> a small physical model that re-renders it (analysis by synthesis).

    from earshot.forge.fit import modal
    model = modal.fit('click.wav')           # or a 48 kHz array (n,) / (n, 2)
    y = modal.render(model, seed=0)          # the resynthesis, same length and channels
    y2 = modal.render(model, seed=3, size=1.3, damping=0.8, brightness=-2, hardness=1.5, vary=0.5)
    src = modal.to_code(model)               # a standalone numpy script that renders it (same samples as render)
    modal.mel_distance(x, y)                 # multi-resolution log-mel distance (dB), the fit's own check

What the model holds (a plain JSON-serialisable dict, kind 'modal', version 1):
- strikes: one per detected onset (a click, a toggle's two contacts, a key's press + bottom-out + release): its
  time `t` (s), an attack ramp `ramp_ms` (the modes rise over it), its modes, rows [freq Hz, decay time constant s,
  amplitude, phase rad] referenced at the onset (amp * exp(-t / tau) * cos(2 pi f t + phase) * ramp(t)), and its
  `attack`: the contact as fast modes (tau <= 2 ms, same row format, no ramp) that keep the first ms coherent.
  A later strike also holds exact copies of earlier poles that still ring at its onset, so it can re-excite or stop
  them (a finger that lifts or holds).
- sub: the residual under 20 Hz as a 100 Hz waveform (a pressure pulse / bump), or None.
- noise: what the deterministic part does not carry, as band envelopes: 28 power-complementary bands (cosine
  crossovers on the ERB-rate axis, `edges` Hz) x breakpoints [t s, dB] (power, linear in dB between breakpoints).
  Each band's power is the original's minus the deterministic part's, so modes + noise keep the band energy.
- fade_out_s: the recording's linear fade-out (estimated; the modes are fitted un-faded and faded at render).
- side: per noise band, the side / mid amplitude ratio of a stereo original (None for mono): one ratio for the whole
  sound (models from before 0.4 render with it).
- side_t (0.4, stereo only): the side over time, sines.side_model of the original: per octave band at keyframes
  after the onset, 'pan' (the side's part coherent with the mid, signed) and 'diffuse' (the rest, re the mid). A
  static ratio set by a wide attack left the tails too wide (a pop's tail L/R correlation 0.30 against 0.72).
- coherent (0.4: True on every fit): vary and the knobs keep the structure the fit gave the modes (render below).
  Models without it, the 0.3 bank's fit entries among them, render exactly as before.
- fit: diagnostics (mode count, the modes' energy share, the residual's level).

How it fits (`fit`):
1. onsets: peaks of a capped log-spectral flux (broadband rises count, a beating pair of modes or a glide's few bins
   barely do), refined to the first sample over 10 % of the local peak, or to the first step of the signal's
   difference when an earlier strike still rings; a fade-out estimate, and the signal un-faded for the modes;
2. per strike, damped-sinusoid candidates by ESPRIT on complex-demodulated, decimated sub-bands (15 bands, 20 Hz to
   21.5 kHz; long windows for the low modes, short ones for the high), two overlapping windows per band merged,
   the order by MDL on the Hankel singular values (capped by a dynamic-range floor); poles outside the band's core,
   growing, or with Q < 2 are dropped;
3. amplitudes + phases by least squares on the full-band signal from `skip_ms` after each onset (closed-form Gram
   matrix of damped exponentials: many modes over a long file stay cheap; ridge raised while near-duplicates of
   one strike, or a pile of strikes, cancel each other), modes under -50 dB of their strike pruned, clusters of
   cancelling near-duplicates cut to the one that alone explains the most (matching pursuit), a second pass on the
   residual adds what the first missed, then one joint fit of every strike's modes;
4. per strike the attack ramp that leaves the least residual in the first ms; the sub wave; the contact modes
   (full-rate ESPRIT on the first 2.5 ms of the residual);
5. band envelopes (analytic-signal power; 0.5 ms smoothing from 8 ms before to 15 ms after onsets, 6 ms after,
   30 ms in long tails; over the PRE_CAP_S before an onset no band rises above its level just before it),
   simplified to breakpoints (Ramer-Douglas-Peucker, 1 dB near strikes, 2 dB in tails).

render(model, seed, knobs): size (x: mode frequencies / size, decays * size, the noise spectrum shifted with them),
damping (decays / damping, noise tails shortened), brightness (dB / octave around 1 kHz on modes and noise),
hardness (x: a shorter contact and +4 dB / octave per doubling on the modes and the attack noise), noise (dB on
the noise bands), width (x side), room (0..1 wet of a small stochastic room), vary (0..1: seeded jitter of mode
frequencies / decays / amplitudes and noise bands: new takes of the same object), spacing (x the time between
strikes), gain (dB). seed changes only the noise (and, with vary > 0, the jitter). A `coherent` model (0.4) keeps the
structure the least squares fit gave its modes (close modes, and a later strike's modes, shape and cancel the earlier
ones' ring): its jitter is a smooth random function of log frequency (6 cosines of 0.25-1.25 cycles per octave), the
same for every strike, its strikes keep their times, and when a knob moves a strike or a mode's frequency (vary, size,
spacing) every mode keeps its phase on the first strike's timeline. Without it (models fitted before 0.4) every mode
is detuned on its own (0.6 % x vary) and the strike times jitter (2 ms x vary): even 0.1 % undid the cancellations, so
a key's takes rang on 2.5x the recording's tail (on the films' modal and hybrid dev fits the takes' CLAP to the
recording was 0.75-0.87 against 0.89-0.94 for vary 0; coherent, it is as for vary 0).
The side of a stereo render is made from the mid: with `side_t` per octave band the band's mid times its pan gain plus
the band's Hilbert transform times its diffuse gain, both linear between keyframes (which move with the strikes and
stretch with the decays, the bands with the modes); else the mid's Hilbert transform shaped per band by `side`.
ring: a take that rings out (default None: on when vary > 0; off, the render keeps the recording's length and
fade-out, what the fit is measured against). A trimmed recording stops while its modes still ring (a key's release
70 ms before the cut), and vary lengthens some decays further; with ring the render goes on until every mode is
RING_DB (60 dB) under the loudest strike's mode sum, at least RING_MIN_S (RING_LOUD_S when the modes are still
above RING_LOUD_DB at the cut) and at most RING_EXT x the sound's span longer, and the added stretch fades 60 dB
(linear in dB) instead of the recording's fade-out (the noise keeps falling at its last slope). A fitted model often
holds a few faint poles with decays of seconds (noise the fitter took for modes); the cap keeps a take short, the
fade keeps it from stopping on them. The fit engine's takes (auto.render(take=True), the bank) ring out. A coherent
model's ring-out continues its noise un-faded too, falling at its slope from before the recording's fade-out (from the
fade, the modes rang on alone in the added stretch: a comb in the tail).
What it is not for: glides / chirps (a rising pop, a zip) split into many false strikes; designed impacts with long
reverb tails fit, but the tail is mostly noise bands.
Pure numpy / scipy.
"""
import inspect
import json

import numpy as np
from scipy import fft as sfft
from scipy import signal

SR = 48000
KIND = 'modal'
VERSION = 1

MODE_BANDS = (20, 60, 120, 200, 320, 500, 750, 1100, 1600, 2300, 3300, 4700, 6600, 9300, 13000, 21500)
NOISE_EDGES = (50, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
               4400, 5300, 6400, 7700, 9500, 12000, 14000, 16000, 18500)   # critical bands + low / top splits: 28
NOISE_H = 0.4              # crossover half-width, ERB-rate units
GRID = 0.0005              # dense envelope grid, s
SUB_HZ = 20.0              # below this the residual is kept as a decimated waveform (a pressure pulse)
SUB_DEC = 480              # its decimation (100 Hz)
RING_DB = 60.0             # ring=True: the render goes on until every mode is this far under the loudest strike's
RING_EXT = 0.5             # mode sum, at most this x the sound's span (first strike to the nominal end) longer, at
RING_MIN_S = 0.06          # least this much longer; the added stretch fades RING_DB (linear in dB)
RING_LOUD_DB = -50.0       # modes still above this at the recording's end (re the loudest strike's mode sum) ...
RING_LOUD_S = 0.2          # ... get at least this long a fade (a fast fade from a loud ring reads as a cut)
STRIKE_RAMP_S = 0.00015    # a later strike's modes and contact start over this raised-cosine ramp (7 samples): from
                           # full amplitude they put a step into the ring of the strikes before it, which the audit
                           # hears as a click (an impact's 4 takes and its fit clicked at two later strikes). The first
                           # strike (the onset) keeps its step; a model without 'coherent' (fitted before 0.4: the
                           # bank's fit entries) renders as before
RAMPS_MS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)   # the attack ramps a strike's modes may rise over (ms)
PRE_CAP_S = 0.002          # the noise bands may not rise over this long before an onset (0.4; 0 = as before): the
                           # analysis smeared a strike's noise up to 1.5 ms ahead of it, 10-20 dB over the recording,
                           # the main cause of blunt modal onsets (on the films' 27 modal dev clips the onset error
                           # fell 0.39 -> 0.25 octave, a glockenspiel note's 3.6 -> 0.1; CLAP and mel unchanged)


# ------------------------------------------------------------------------------------------------ helpers
def _load(x):
    if isinstance(x, str):
        from ... import audio
        x = audio.load(x)
    x = np.asarray(x, np.float64)
    return x[:, None] if x.ndim == 1 else x


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _sig(v, n=5):
    """round to n significant digits (JSON-stable)"""
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return 0.0
    return float(f'{v:.{n}g}')


def _env_db(m, hop=12, win=24):
    """energy envelope in dB at `hop` samples (window `win`), and the frame centres"""
    p = np.convolve(m * m, np.ones(win) / win, mode='same')
    e = p[hop // 2::hop]
    return 10 * np.log10(e + 1e-20), np.arange(len(e)) * hop + hop // 2


# ------------------------------------------------------------------------------------------------ onsets
def onset_strength(m, nper=256, hop=32, cap=1.5):
    """log-spectral flux (half-wave rectified rise of log(1 + |X| / ref) summed over bins, 2-frame lag) and the
    frame energy in dB, at `hop` samples; broadband rises count, a beating pair of modes barely does"""
    f, t, X = signal.stft(m, SR, nperseg=nper, noverlap=nper - hop, boundary='zeros', padded=True)
    A = np.abs(X)
    ref = A.max() * 1e-3 + 1e-12
    Sl = np.log1p(A / ref)
    flux = np.zeros(Sl.shape[1])
    flux[2:] = np.minimum(np.maximum(Sl[:, 2:] - Sl[:, :-2], 0.0), cap).sum(axis=0)   # capped: a glide's few bins
                                                                                     # cannot outvote a strike
    e = 10 * np.log10((A ** 2).sum(axis=0) + 1e-20)
    centre = (np.arange(Sl.shape[1]) * hop).astype(int)
    return flux, e, centre


def onsets(x, rel=0.2, floor_db=-36.0, min_gap_ms=12.0, max_n=10):
    """strike onsets (samples): peaks of the capped log-spectral flux >= rel x its maximum whose frame energy lies
    within floor_db of the loudest frame and rises >= 3 dB (1 dB for a flux peak over half the maximum: a
    second contact over a loud ring), each refined to the first sample above 10 % of the
    local peak amplitude (from near silence) or to the first step of the signal's difference (over a ringing
    strike)"""
    m = _mono(x)
    if not len(m) or np.abs(m).max() <= 0:
        return [0]
    flux, e, c = onset_strength(m)
    top = e.max()
    w = 6                                                   # +-4 ms local maximum
    fp = np.pad(flux, w, mode='constant')
    locmax = flux >= np.lib.stride_tricks.sliding_window_view(fp, 2 * w + 1).max(axis=1)
    emax = np.lib.stride_tricks.sliding_window_view(np.pad(e, (0, 8), mode='edge'), 9).max(axis=1)[:len(e)]
    emin = np.lib.stride_tricks.sliding_window_view(np.pad(e, (8, 0), mode='edge'), 9).min(axis=1)[:len(e)]
    cand = np.nonzero(locmax & (flux >= rel * flux.max()) & (emax >= top + floor_db) & ((emax - emin >= 3.0) | ((flux >= 0.5 * flux.max()) & (emax - emin >= 1.0))))[0]
    out = []
    early = np.abs(m[:int(0.005 * SR)]).max() >= 0.25 * np.abs(m).max()
    for i in cand:
        if (c[i] < 128 and not early) or c[i] > len(m) - 192:   # the STFT's zero padding, not a strike
            continue
        s = int(c[i])
        a, b = max(0, s - 240), min(len(m), s + 240)
        loc = np.abs(m[a:b])
        if len(loc) < 100 or loc.max() <= 0:
            continue
        pre = np.sqrt(np.mean(m[a:a + 96] ** 2))
        if pre < 0.05 * loc.max():                          # from near silence: 10 % of the local peak
            o = a + int(np.nonzero(loc >= 0.1 * loc.max())[0][0])
        else:                                               # over a ringing strike: the new contact's step
            d = np.abs(np.diff(m[a:b]))
            thr = min(max(0.3 * d.max(), 4.0 * np.median(d[:96])), d.max())
            o = a + int(np.nonzero(d >= thr)[0][0])
        if out and o - out[-1][0] < min_gap_ms * SR / 1000:
            if flux[i] > out[-1][1] * 2.5:                     # a much stronger rise right after: that is the strike
                out[-1] = (out[-1][0], flux[i])
            continue
        out.append((o, float(flux[i])))
    if not out:
        return [int(np.argmax(np.abs(m) > np.abs(m).max() * 0.1))]
    if len(out) > max_n:
        out = sorted(sorted(out, key=lambda q: -q[1])[:max_n])
    return [o for o, _ in out]


def fade_out(x, cands=(0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25), gain_db=0.5):
    """length (s) of a linear fade-out at the end of x (a trimmed recording), 0.0 when none fits: on one common end
    window, the 1 ms dB envelope is fitted by a line + 20 log10 of each candidate ramp (and by a line alone); the
    best ramp must beat the line alone by gain_db (mean absolute dB)"""
    m = _mono(x)
    h = int(0.001 * SR)
    k = len(m) // h
    if k < 20:
        return 0.0
    w = np.hanning(8 * h + 2)[1:-1]                  # 8 ms: low modes' ripple must not look like a fade
    e = 10 * np.log10(np.convolve(m[:k * h] ** 2, w / w.sum(), mode='same')[h // 2::h][:k] + 1e-20)
    t = (np.arange(k) + 0.5) * h / SR
    T = k * h / SR
    W = min(0.45 * T, 0.6)
    win = (t >= T - W) & (e > e.max() - 80)
    if win.sum() < 10:
        return 0.0
    tw, ew = t[win], e[win]
    def err(L):
        r = 20 * np.log10(np.clip((T - tw) / L, 1e-3, 1.0)) if L > 0 else np.zeros(len(tw))
        a, b = np.polyfit(tw, ew - r, 1)
        return float(np.mean(np.abs(ew - (a * tw + b + r))))
    e0 = err(0.0)
    best, bl = e0 - gain_db, 0.0
    for L in cands:
        if L > W / 2:
            break
        v = err(L)
        if v < best:
            best, bl = v, L
    return bl


# ------------------------------------------------------------------------------------------------ ESPRIT
def _order(sv, M, kmax, dyn_db):
    lam = sv.astype(np.float64) ** 2 + 1e-300
    p = len(lam)
    kcap = int(min(kmax, p - 2, np.count_nonzero(sv > sv[0] * 10 ** (-dyn_db / 20))))
    if kcap <= 0:
        return 0
    best, kb = np.inf, 0
    for k in range(kcap + 1):
        tail = lam[k:]
        g = np.exp(np.mean(np.log(tail)))
        a = np.mean(tail)
        mdl = -M * (p - k) * np.log(g / a) + 0.5 * k * (2 * p - k) * np.log(M)
        if mdl < best:
            best, kb = mdl, k
    return kb


def esprit(s, kmax=16, dyn_db=60.0):
    """poles z of the damped exponentials in the (complex) sequence s, model order by MDL; (z, order)"""
    s = np.asarray(s)
    N = len(s)
    if N < 12:
        return np.zeros(0, complex), 0
    L = max(4, N // 3)
    M = N - L + 1
    H = np.lib.stride_tricks.sliding_window_view(s, M)[:L]
    try:
        U, sv, _ = np.linalg.svd(H, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(0, complex), 0
    k = _order(sv, M, kmax, dyn_db)
    if k == 0:
        return np.zeros(0, complex), 0
    Us = U[:, :k]
    Phi = np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0]
    return np.linalg.eigvals(Phi), k


def _band_candidates(seg, kmax=16, nmax=384, min_n=16):
    """(freq, tau) candidates in each MODE_BANDS core from a real segment that starts just after an onset"""
    Ns = len(seg)
    if Ns < 32:
        return []
    n_fft = sfft.next_fast_len(2 * Ns)
    X = np.fft.fft(seg, n_fft)
    f = np.fft.fftfreq(n_fft, 1 / SR)
    n = np.arange(Ns)
    out = []
    for lo, hi in zip(MODE_BANDS[:-1], MODE_BANDS[1:]):
        B = hi - lo
        tw = max(0.15 * B, 8.0)
        a0, a1, b1, b0 = lo - 2 * tw, lo - tw, hi + tw, hi + 2 * tw
        mask = np.zeros(n_fft)
        up = (f > a0) & (f < a1)
        mask[up] = 0.5 - 0.5 * np.cos(np.pi * (f[up] - a0) / tw)
        mask[(f >= a1) & (f <= b1)] = 1.0
        dn = (f > b1) & (f < b0)
        mask[dn] = 0.5 + 0.5 * np.cos(np.pi * (f[dn] - b1) / tw)
        span = B + 4 * tw
        D = max(1, min(int(SR / (1.15 * span)), Ns // 48))
        if Ns // D < min_n:
            continue
        fc = 0.5 * (lo + hi)
        y = np.fft.ifft(X * mask)[:Ns] * np.exp(-2j * np.pi * fc * n / SR)
        yd = y[::D]
        fs = SR / D
        wins = [yd[:nmax]]
        o2 = (nmax * 3) // 4
        if len(yd) > o2 + nmax // 2:
            w1 = yd[o2:o2 + nmax]
            if np.mean(np.abs(w1) ** 2) > 1e-4 * np.mean(np.abs(wins[0]) ** 2):
                wins.append(w1)
        for w in wins:
            z, _ = esprit(w, kmax)
            if not len(z):
                continue
            r = np.abs(z)
            ok = (r < 0.99999) & (r > 1e-6)
            z = z[ok]
            fk = fc + np.angle(z) * fs / (2 * np.pi)
            tau = -1.0 / (np.log(np.abs(z)) * fs)
            good = (fk >= lo) & (fk < hi) & (tau >= 3e-4) & (np.pi * fk * tau >= 2.0) & (tau < 30.0)
            out += [(float(a), float(b)) for a, b in zip(fk[good], tau[good])]
    return out


def _dedupe(cands, existing=()):
    """drop candidates that duplicate an existing mode or an earlier candidate (same freq within its bandwidth / 4)"""
    keep = list(existing)
    new = []
    for fk, tk in cands:
        dup = False
        for fe, te in keep:
            tol = max(0.3, 0.0015 * fk, 0.25 / (np.pi * np.sqrt(tk * te)))
            if abs(fk - fe) < tol and abs(np.log(tk / te)) < 1.2:
                dup = True
                break
        if not dup:
            keep.append((fk, tk))
            new.append((fk, tk))
    return new


# ------------------------------------------------------------------------------------------------ least squares
def _poles(freqs, taus):
    freqs, taus = np.asarray(freqs, float), np.asarray(taus, float)
    return np.exp((-1.0 / (taus * SR)) + 2j * np.pi * freqs / SR)


def _geo(w, L):
    """sum_{n<L} w^n (complex, elementwise)"""
    d = 1.0 - w
    small = np.abs(d) < 1e-12
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        v = (1.0 - np.exp(L * np.log(w + 0j))) / np.where(small, 1.0, d)
    return np.where(small, L, v)


def _gram(starts, Z, end):
    A = starts[:, None].astype(np.float64)
    B = starts[None, :].astype(np.float64)
    mx = np.maximum(A, B)
    L = np.maximum(end - mx, 0.0)
    lz = np.log(Z)
    with np.errstate(under='ignore'):
        pi = np.exp((mx - A) * lz[:, None])
        pj = np.exp((mx - B) * lz[None, :])
        S1 = pi * pj * _geo(Z[:, None] * Z[None, :], L)
        S2 = pi * np.conj(pj) * _geo(Z[:, None] * np.conj(Z)[None, :], L)
    RR = 0.5 * (S1 + S2).real
    II = 0.5 * (S2 - S1).real
    RI = 0.5 * (S1 - S2).imag
    IR = 0.5 * (S1 + S2).imag
    return np.block([[RR, RI], [IR, II]])


def _project(m, starts, Z, end, chunk=8192):
    K = len(Z)
    bR, bI = np.zeros(K), np.zeros(K)
    lz = np.log(Z)
    for a in np.unique(starts):
        idx = np.nonzero(starts == a)[0]
        taus = -1.0 / (np.log(np.abs(Z[idx])) * SR)
        T = int(min(end - a, np.ceil(16 * taus.max() * SR) + 1))
        if T <= 0:
            continue
        B = min(chunk, T)
        with np.errstate(under='ignore'):
            base = np.exp(np.arange(B)[:, None] * lz[idx][None, :])      # z^0 .. z^(B-1), scaled per chunk
            for c0 in range(0, T, B):
                k = min(B, T - c0)
                v = (m[a + c0:a + c0 + k] @ base[:k]) * np.exp(c0 * lz[idx])
                bR[idx] += v.real
                bI[idx] += v.imag
    return np.concatenate([bR, bI])


def ls_modes(m, starts, freqs, taus, end=None, ridge=1e-5, cross=10.0):
    """complex amplitudes c (mode_k[n] = Re(c_k z_k^(n - start_k)) for start_k <= n < end) by least squares on m"""
    starts = np.asarray(starts, np.int64)
    K = len(starts)
    if K == 0:
        return np.zeros(0, complex)
    end = len(m) if end is None else int(end)
    Z = _poles(freqs, taus)
    G = _gram(starts, Z, end)
    b = _project(m, starts, Z, end)
    dg = np.maximum(np.diag(G), 1e-30)
    lam = ridge
    for _ in range(5):
        Gr = G + lam * np.diag(dg)
        try:
            x = np.linalg.solve(Gr, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(Gr, b, rcond=None)[0]
        bad = False                                 # heavy cancellation between near-duplicates of one strike
        strike_e = 0.0
        for a in np.unique(starts):
            g = np.nonzero(starts == a)[0]
            gg = np.concatenate([g, g + K])
            tot = float(x[gg] @ G[np.ix_(gg, gg)] @ x[gg])
            strike_e += max(tot, 0.0)
            parts = float(sum(x[[k, k + K]] @ G[np.ix_([k, k + K], [k, k + K])] @ x[[k, k + K]] for k in g))
            if tot > 0 and parts > 2.0 * tot:
                bad = True
                break
        # ... or between strikes: a stopped ring cancels its own tail (ratio < ~6), a pile of close strikes
        # building the sound from huge opposite rings does not
        if not bad and strike_e > cross * max(float(x @ G @ x), 1e-30):
            bad = True
        if not bad:
            break
        lam *= 10.0
    return x[:K] - 1j * x[K:]


def _cancelling(m, starts, freqs, taus, c, ratio=2.0):
    """indices of modes to drop: in each cluster of one strike's modes whose bandwidths overlap, when the modes'
    separate energies add up to more than `ratio` x the energy of their sum (they cancel), all but the one that
    alone explains the most of the signal m (a matching-pursuit choice: not the longest, loudest ring)"""
    n = len(m)
    starts, freqs, taus = np.asarray(starts), np.asarray(freqs), np.asarray(taus)
    E = _mode_energy(c, taus)
    drop = []
    for a in np.unique(starts):
        g = np.nonzero(starts == a)[0]
        g = g[np.argsort(freqs[g])]
        bw = 1.0 / (np.pi * taus[g])
        clusters, cur = [], [g[0]]
        for k in range(1, len(g)):
            if freqs[g[k]] - freqs[g[k - 1]] < max(2.0, 0.5 * (bw[k] + bw[k - 1])):
                cur.append(g[k])
            else:
                clusters.append(cur)
                cur = [g[k]]
        clusters.append(cur)
        for cl in clusters:
            if len(cl) < 2:
                continue
            cl = np.array(cl)
            Zc = _poles(freqs[cl], taus[cl])
            G = _gram(starts[cl], Zc, n)
            x = np.concatenate([c[cl].real, -c[cl].imag])
            tot = float(x @ G @ x)
            if E[cl].sum() > ratio * max(tot, 1e-30):
                b = _project(m, starts[cl], Zc, n)
                L = len(cl)
                expl = []
                for k in range(L):
                    ix = [k, k + L]
                    Gk = G[np.ix_(ix, ix)] + 1e-12 * np.eye(2) * max(G[k, k], 1e-30)
                    expl.append(float(b[ix] @ np.linalg.solve(Gk, b[ix])))
                best = cl[int(np.argmax(expl))]
                drop += [int(k) for k in cl if k != best]
    return drop


def _mode_energy(c, taus):
    r2 = np.exp(-2.0 / (np.asarray(taus) * SR))
    return 0.5 * np.abs(c) ** 2 / (1.0 - r2)


def _synth_strike(n, o, freqs, taus, amps, phases, ramp_s=0.0, start=None):
    """the summed modes of one strike over n samples: onset o (samples); start (samples, default o) = first sample"""
    y = np.zeros(n)
    s0 = o if start is None else start
    if s0 >= n:
        return y
    for f, tau, a, ph in zip(freqs, taus, amps, phases):
        L = int(min(n - s0, np.ceil(14 * tau * SR) + 1))
        if L <= 0 or a == 0:
            continue
        t = (np.arange(L) + (s0 - o)) / SR
        y[s0:s0 + L] += a * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + ph)
    if ramp_s > 0 and start is None:
        R = int(round(ramp_s * SR))
        if R > 1:
            k = min(R, n - o)
            y[o:o + k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / R)
    return y


def _attack_modes(r, o, end, kmax=24, tau_max=0.002):
    """the contact transient as fast damped sinusoids: full-rate ESPRIT on r[o:end] (real: conjugate pairs), poles
    with tau <= tau_max, amplitudes by least squares over the window; rows [f, tau, amp, phase] at the onset"""
    seg = r[o:end]
    if len(seg) < 24 or not np.any(seg):
        return []
    z, k = esprit(seg, kmax=kmax, dyn_db=50.0)
    if not len(z):
        return []
    z = z[(np.abs(z) < 0.9999) & (np.abs(z) > 1e-4) & (np.angle(z) >= 0)]
    f = np.angle(z) * SR / (2 * np.pi)
    tau = -1.0 / (np.log(np.abs(z)) * SR)
    ok = (tau >= 1e-5) & (tau <= tau_max) & (f < 0.47 * SR)
    f, tau = f[ok], tau[ok]
    if not len(f):
        return []
    c = ls_modes(r, np.full(len(f), o), f, tau, end=min(len(r), o + int(3 * tau_max * SR)))
    rows = [[round(float(a), 2), _sig(b, 5), _sig(abs(cc), 5), round(float(np.angle(cc)), 4)]
            for a, b, cc in zip(f, tau, c) if abs(cc) > 0]
    return sorted(rows, key=lambda q: q[0])


def _sub_wave(xs, n):
    y = signal.resample_poly(np.asarray(xs, float), SUB_DEC, 1)[:n]
    return np.pad(y, (0, max(0, n - len(y))))


# ------------------------------------------------------------------------------------------------ noise bands
def band_masks(n_fft, edges=NOISE_EDGES, h=NOISE_H):
    """power-complementary zero-phase masks on the rfft bins (sum of squares = 1): (bands, bins)"""
    return _masks(n_fft, edges, h)


def band_centres(edges=NOISE_EDGES, top=22000.0):
    e = np.asarray(edges, float)
    return np.concatenate([[e[0] / np.sqrt(2)], np.sqrt(e[:-1] * e[1:]), [np.sqrt(e[-1] * top)]])


def _rdp(t, v, tol):
    """indices kept by Ramer-Douglas-Peucker on the polyline (t, v) with tolerance tol (v units; scalar or per point)"""
    n = len(t)
    if n <= 2:
        return np.arange(n)
    tol = np.broadcast_to(np.asarray(tol, float), (n,))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        tt = t[a + 1:b]
        line = v[a] + (v[b] - v[a]) * (tt - t[a]) / (t[b] - t[a])
        d = np.abs(v[a + 1:b] - line) / tol[a + 1:b]
        i = int(np.argmax(d))
        if d[i] > 1.0:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k))
            stack.append((k, b))
    return np.nonzero(keep)[0]


def _fit_noise(m, det, ons, tol_db=1.0, edges=NOISE_EDGES, h=NOISE_H, pre_cap=0.0):
    """band envelopes of the stochastic part: per band, the original's power minus the deterministic part's power
    (analytic-signal power, 0.5 ms smoothing from 8 ms before to 15 ms after onsets, 6 ms elsewhere), so the
    rendered modes + noise keep the original's band energy even where the modes misfit; a list per band of [t, dB]
    breakpoints. pre_cap (s): over this long before each onset a band stays at its level just before (no rise ahead
    of the strike)"""
    n = len(m)
    n_fft = sfft.next_fast_len(n + 16384)
    W = _masks(n_fft, edges, h)
    R = np.fft.rfft(m, n_fft)
    Rd = np.fft.rfft(det, n_fft)
    grid = np.arange(0, n, int(GRID * SR))
    tg = grid / SR
    near = np.zeros(len(grid))
    since = np.full(len(grid), 10.0)                 # s since the latest onset (tails get longer smoothing)
    for o in ons:
        d = (grid - o) / SR
        w = np.where((d >= -0.008) & (d <= 0.015), 1.0,
                     np.where((d > 0.015) & (d < 0.03), 1.0 - (d - 0.015) / 0.015, 0.0))
        near = np.maximum(near, w)
        since = np.where(d >= 0, np.minimum(since, d), since)
    far = np.clip((since - 0.15) / 0.15, 0.0, 1.0)
    hs, hl = np.hanning(int(0.0005 * SR) + 2)[1:-1], np.hanning(int(0.006 * SR) + 2)[1:-1]
    hf = np.hanning(int(0.03 * SR) + 2)[1:-1]
    hs, hl, hf = hs / hs.sum(), hl / hl.sum(), hf / hf.sum()
    def bandpow(Rx, b):
        Xa = np.zeros(n_fft, complex)
        Xa[:len(Rx)] = Rx * W[b]
        Xa[1:len(Rx) - 1] *= 2.0
        p2 = 0.5 * np.abs(np.fft.ifft(Xa)[:n]) ** 2
        ps = np.maximum(signal.oaconvolve(p2, hs, mode='same')[grid], 0.0)
        pl = np.maximum(signal.oaconvolve(p2, hl, mode='same')[grid], 0.0)
        pf = np.maximum(signal.oaconvolve(p2, hf, mode='same')[grid], 0.0)
        return near * ps + (1 - near) * ((1 - far) * pl + far * pf)
    P_all = []
    for b in range(len(W)):
        po, pd = bandpow(R, b), bandpow(Rd, b)
        P_all.append(np.maximum(po - pd, 1e-6 * po))
    P_all = np.array(P_all)
    if pre_cap > 0:                                  # the analysis smears a strike's energy ahead of it (the 0.5 ms
        step = int(GRID * SR)                        # smoothing, the band filters' pre-ring): over the pre_cap s
        k_cap = int(round(pre_cap / GRID))           # before an onset no band may rise above its level just before
        for o in ons:
            io = int(o // step)
            a = io - k_cap
            if a >= 1:
                ref = P_all[:, max(0, a - 2):a + 1].min(axis=1)
                P_all[:, a:io + 1] = np.minimum(P_all[:, a:io + 1], ref[:, None])
    fl = 10 * np.log10(P_all.max() + 1e-30) - 100.0
    env = []
    for b in range(len(W)):
        dB = np.maximum(10 * np.log10(P_all[b] + 1e-30), fl)
        k = _rdp(tg, dB, tol_db * (1.0 + far))      # 1 dB near the strikes, 2 dB in long tails
        env.append([[round(float(tg[i]), 5), round(float(dB[i]), 2)] for i in k])
    return env


# ------------------------------------------------------------------------------------------------ fit
def fit(x, *args, **kw):
    """fit a modal model to a struck / ringing recording (path or 48 kHz array); returns the model dict. Options:
    max_strikes, kmax (ESPRIT order cap per band), nmax (window, decimated samples), passes, skip_ms (the LS starts
    this long after each onset), prune_db, max_modes (per strike), onset_rel, noise_tol_db, analysis_s, verbose,
    ramps (the attack ramps tried, ms), pre_cap_s (no noise rise this long before an onset), side_t (a stereo
    source: the side over time, model['side_t'])"""
    try:                                   # small matrices: one BLAS thread is ~5x faster than a contended pool
        from threadpoolctl import threadpool_limits
    except ImportError:
        return _fit(x, *args, **kw)
    with threadpool_limits(1):
        return _fit(x, *args, **kw)


def _fit(x, max_strikes=10, kmax=16, nmax=384, passes=2, skip_ms=0.5, prune_db=-50.0, max_modes=80, onset_rel=0.2,
        noise_tol_db=1.0, analysis_s=3.0, attack_ms=2.5, verbose=False, ramps=RAMPS_MS, pre_cap_s=PRE_CAP_S,
        side_t=True):
    X = _load(x)
    m_orig = X.mean(axis=1)
    n = len(m_orig)
    ons = onsets(m_orig, rel=onset_rel, max_n=max_strikes)
    fo = fade_out(m_orig)
    m = m_orig.copy()                                # the modes are fitted on the un-faded signal
    n_fit = n
    if fo > 0:
        nf = int(round(fo * SR))
        ramp = np.clip((n - np.arange(n - nf, n)) / nf, 0.0, 1.0)
        m[n - nf:] /= np.maximum(ramp, 0.25)
        n_fit = n - int(0.25 * nf)
    skip = int(round(skip_ms * SR / 1000))
    skip_b = skip + int(round(0.0005 * SR))
    e, c = _env_db(m_orig)
    act = np.nonzero(e >= e.max() - 75)[0]
    act_end = int(min(n_fit, c[act[-1]] + 12)) if len(act) else n_fit
    seg_end = [min(ons[i + 1], act_end) if i + 1 < len(ons) else act_end for i in range(len(ons))]
    cap = int(analysis_s * SR)
    S = len(ons)
    modes = [[] for _ in range(S)]           # per strike: list of (f, tau)
    work = m.copy()
    for p in range(passes):
        for i, o in enumerate(ons):
            a = o + skip_b
            e_ = min(seg_end[i], a + cap)
            if e_ - a < 48:
                continue
            cands = _band_candidates(work[a:e_], kmax=kmax, nmax=nmax)
            new = _dedupe(cands, modes[i])
            if not new:
                continue
            fr = np.array([q[0] for q in new])
            ta = np.array([q[1] for q in new])
            st = np.full(len(new), o + skip)
            cc = ls_modes(work, st, fr, ta, end=min(seg_end[i], o + skip + cap) if i + 1 < S else n_fit)
            E = _mode_energy(cc, ta)
            ref = max(E.sum(), 1e-30)
            order = np.argsort(-E)
            room_left = max_modes - len(modes[i])
            keep = [j for j in order[:max(room_left, 0)] if E[j] >= ref * 10 ** (prune_db / 10)]
            for j in keep:
                modes[i].append((float(fr[j]), float(ta[j])))
            if keep:
                amps, phs = np.abs(cc[keep]), np.angle(cc[keep])
                work -= _synth_strike(n, o + skip, fr[keep], ta[keep], amps, phs)
        # a later strike may re-excite or stop (a finger lifts, a second contact damps) an earlier one's modes: it
        # gets copies of the earlier poles that still ring at its onset, the joint fit picks their amplitudes
        for i in range(1, S):
            prev = [(fk, tk) for j in range(i) for fk, tk in modes[j]
                    if np.exp(-(ons[i] - ons[j]) / (tk * SR)) > 1e-2]
            prev = _dedupe(sorted(prev, key=lambda q: -q[1])[:40])
            own = [q for q in modes[i] if q not in prev]
            modes[i] = prev + _dedupe(own, prev)
        # one joint fit of every strike's modes on the original
        starts, fr, ta, owner = [], [], [], []
        for i, o in enumerate(ons):
            for fk, tk in modes[i]:
                starts.append(o + skip)
                fr.append(fk)
                ta.append(tk)
                owner.append(i)
        if not starts:
            break
        starts, fr, ta, owner = map(np.asarray, (starts, fr, ta, owner))
        cc = ls_modes(m, starts, fr, ta, end=n_fit)
        E = _mode_energy(cc, ta)
        keep = np.zeros(len(E), bool)
        for i in range(S):
            sel = owner == i
            if sel.any():
                keep[sel] = E[sel] >= E[sel].sum() * 10 ** (prune_db / 10)
        if not keep.all():
            starts, fr, ta, owner = starts[keep], fr[keep], ta[keep], owner[keep]
            cc = ls_modes(m, starts, fr, ta, end=n_fit)
        for _ in range(3):                           # clusters of near-duplicates that cancel: keep their strongest
            drop = _cancelling(m[:n_fit], starts, fr, ta, cc)
            if not drop:
                break
            keep = np.ones(len(fr), bool)
            keep[drop] = False
            starts, fr, ta, owner = starts[keep], fr[keep], ta[keep], owner[keep]
            cc = ls_modes(m, starts, fr, ta, end=n_fit)
        modes = [[(float(a), float(b)) for a, b in zip(fr[owner == i], ta[owner == i])] for i in range(S)]
        work = m - sum((_synth_strike(n, ons[i] + skip, fr[owner == i], ta[owner == i], np.abs(cc[owner == i]),
                                      np.angle(cc[owner == i])) for i in range(S) if (owner == i).any()),
                       np.zeros(n))
        if verbose:
            print(f'pass {p}: {len(fr)} modes, residual {10 * np.log10(np.sum(work ** 2) / np.sum(m ** 2) + 1e-30):.1f} dB')
    # per strike: modes referenced at the onset, and the attack ramp that leaves the least residual
    strikes = []
    mode_sig = np.zeros(n)
    have = len(starts) if isinstance(starts, np.ndarray) else 0
    for i, o in enumerate(ons):
        if have and (owner == i).any():
            sel = owner == i
            f_i, t_i, c_i = fr[sel], ta[sel], cc[sel]
        else:
            f_i, t_i, c_i = np.zeros(0), np.zeros(0), np.zeros(0, complex)
        amp0 = np.abs(c_i) * np.exp(skip / (t_i * SR)) if len(t_i) else np.zeros(0)
        ph0 = np.angle(c_i) - 2 * np.pi * f_i * skip / SR if len(t_i) else np.zeros(0)
        rows = sorted([[round(float(a), 2), _sig(b, 5), _sig(c_, 5), round(float(np.angle(np.exp(1j * d))), 4)]
                       for a, b, c_, d in zip(f_i, t_i, amp0, ph0)], key=lambda q: q[0])
        rows = [q for q in rows if q[2] > 0 and q[1] > 0]
        fq, tq, aq, pq = (np.array([q[j] for q in rows]) for j in range(4)) if rows else (np.zeros(0),) * 4
        base = m - mode_sig
        w0, w1 = o, min(n, o + skip + int(0.0015 * SR))
        best, rb = np.inf, 0.0
        for rms in ramps:
            y = _synth_strike(w1 + int(0.002 * SR), o, fq, tq, aq, pq, rms / 1000) if rows else None
            err = np.sum((base[w0:w1] - (y[w0:w1] if y is not None else 0)) ** 2)
            if err < best * (1 - 1e-9):
                best, rb = err, rms
        mode_sig += _synth_strike(n, o, fq, tq, aq, pq, rb / 1000) if rows else 0
        strikes.append({'t': round(o / SR, 6), 'ramp_ms': rb, 'modes': rows})
    resid = m - mode_sig
    sub = None
    if n > 4 * SUB_DEC:                              # a slow pressure pulse under 20 Hz (thump, bump): its waveform
        lo = signal.sosfiltfilt(signal.butter(4, SUB_HZ, 'low', fs=SR, output='sos'), resid)
        if np.sum(lo ** 2) > 1e-3 * np.sum(m ** 2):
            xs = signal.resample_poly(lo, 1, SUB_DEC)
            sub = {'rate': SR // SUB_DEC, 'x': [_sig(v, 4) for v in xs]}
            resid = resid - _sub_wave(sub['x'], n)
    for i, o in enumerate(ons):                      # the contact: fast coherent modes in the first ms
        end_a = min(n, o + int(attack_ms * SR / 1000), ons[i + 1] if i + 1 < S else n)
        rows = _attack_modes(resid, o, end_a)
        strikes[i]['attack'] = rows
        if rows:
            q = np.array(rows)
            resid -= _synth_strike(n, o, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    det = m - resid                                  # modes + sub + contact, un-faded
    if fo > 0:
        nf = int(round(fo * SR))
        det[n - nf:] *= np.clip((n - np.arange(n - nf, n)) / nf, 0.0, 1.0)
        resid = m_orig - det
    model = {'kind': KIND, 'version': VERSION, 'sr': SR, 'length_s': round(n / SR, 6), 'channels': int(X.shape[1]),
             'fade_out_s': fo, 'strikes': strikes, 'sub': sub,
             'noise': {'edges': list(NOISE_EDGES), 'h_erb': NOISE_H,
                       'env': _fit_noise(m_orig, det, ons, noise_tol_db, pre_cap=pre_cap_s)},
             'side': None, 'coherent': True}
    if X.shape[1] == 2:
        side = 0.5 * (X[:, 0] - X[:, 1])
        n_fft = sfft.next_fast_len(n)
        W = band_masks(n_fft)
        Ms, Ss = np.abs(np.fft.rfft(m_orig, n_fft)) ** 2, np.abs(np.fft.rfft(side, n_fft)) ** 2
        rho = np.sqrt(((W ** 2) @ Ss + 1e-30) / ((W ** 2) @ Ms + 1e-30))
        model['side'] = [round(float(min(v, 1.5)), 4) for v in rho]
        if side_t:
            from .sines import side_model
            model['side_t'] = side_model(X)
    pm = float(np.sum(m_orig ** 2)) + 1e-30
    model['fit'] = {'skip_ms': skip_ms, 'modes': int(sum(len(s['modes']) for s in strikes)),
                    'mode_share_db': round(float(10 * np.log10(np.sum(mode_sig ** 2) / pm + 1e-30)), 2),
                    'resid_db': round(float(10 * np.log10(np.sum(resid ** 2) / pm + 1e-30)), 2)}
    return model


# ------------------------------------------------------------------------------------------------ render
# Everything below `render_model` uses only numpy / scipy and the constants SR, GRID: to_code() copies these
# functions verbatim into a standalone script, so the script renders the same samples as render().
def _plan(model, seed, size, damping, brightness, hardness, noise, vary, spacing, ring=False):
    """the model with the knobs applied: strikes (onset s, ramp s, f, tau, amp, phase arrays), the dense noise
    envelope (bands x grid, dB) on a GRID-second grid, the band centres and the output length (samples). ring: when
    a mode outlives the recording's length, render on until every mode is RING_DB down (see render)"""
    rv = np.random.default_rng([int(seed) & 0xFFFFFFFF, 1])
    coherent = bool(model.get('coherent'))       # a 0.4 fit (module doc); older models keep the per-mode jitter
    st = model['strikes']
    t0 = [s['t'] for s in st]
    tn = []
    for i, t in enumerate(t0):
        if i == 0:
            tn.append(t)
        else:
            gap = (t - t0[i - 1]) * spacing + (0.0 if coherent else vary * 0.002 * rv.standard_normal())
            tn.append(tn[-1] + max(0.003, gap))

    def field():                                  # a smooth random function of log frequency (unit variance)
        c, p = rv.uniform(0.25, 1.25, 6), rv.uniform(0.0, 2 * np.pi, 6)
        return lambda fr: np.sqrt(1 / 3) * np.cos(2 * np.pi * np.log2(np.maximum(fr, 20.0))[:, None] * c + p).sum(1)
    if coherent:
        jf, jt, ja = field(), field(), field()
    shifted = coherent and (vary > 0 or size != 1.0 or spacing != 1.0)
    stretch = max(size / damping, 1e-3)
    tilt = brightness + 4.0 * np.log2(max(hardness, 1e-3))
    strikes = []
    for i, s in enumerate(st):
        md = np.array(s['modes'], float).reshape(-1, 4)
        f0 = md[:, 0]
        k = len(md)
        if coherent:
            zf, zt, za = jf(f0), jt(f0), ja(f0)
        else:
            zf, zt, za = rv.standard_normal(k), rv.standard_normal(k), rv.standard_normal(k)
        f = f0 / size * (1 + 0.006 * vary * zf)
        tau = md[:, 1] * stretch * np.exp(0.12 * vary * zt)
        a = md[:, 2] * 10 ** ((tilt * np.log2(np.maximum(f, 20.0) / 1000.0) + 1.5 * vary * za) / 20)
        ph = md[:, 3]
        if shifted and i > 0:                     # every mode keeps its phase on the first strike's timeline
            ph = ph + 2 * np.pi * (f * (tn[i] - tn[0]) - f0 * (t0[i] - t0[0]))
        keep = f < 0.47 * SR
        at = np.array(s.get('attack', []), float).reshape(-1, 4)
        fa = at[:, 0] / size
        ta = at[:, 1] * size / np.sqrt(max(hardness, 1e-3))
        aa = at[:, 2] * 10 ** ((tilt * np.log2(np.maximum(fa, 20.0) / 1000.0)) / 20)
        ka = fa < 0.47 * SR
        strikes.append((tn[i], s['ramp_ms'] / 1000.0 / max(hardness, 1e-3), f[keep], tau[keep], a[keep],
                        ph[keep], fa[ka], ta[ka], aa[ka], at[ka, 3]))
    L0 = model['length_s']
    t_last0, t_last = t0[-1], tn[-1]
    length = t_last + (L0 - t_last0) * max(1.0, stretch)
    n_nom = int(round(length * SR))
    if ring:
        t_ring, lv_end = _ring_end(strikes, length)
        ring = None
        if t_ring > length + 0.001:
            lo = RING_LOUD_S if lv_end > RING_LOUD_DB else RING_MIN_S
            ext = min(max(t_ring - length, lo), max(lo, RING_EXT * (length - tn[0])))
            length += ext
            ring = (n_nom, int(round(length * SR)))
    n_out = int(round(length * SR))
    # dense noise envelope on the original's time axis
    nz = model['noise']
    edges = np.asarray(nz['edges'], float)
    cen = np.concatenate([[edges[0] / np.sqrt(2)], np.sqrt(edges[:-1] * edges[1:]), [np.sqrt(edges[-1] * 22000.0)]])
    g_old = np.arange(0.0, L0 + GRID, GRID)
    E0 = np.array([np.interp(g_old, [p[0] for p in env], [p[1] for p in env]) for env in nz['env']])
    # time map: new grid -> old time (per strike segment: shift, tails stretched by size / damping)
    g_new = np.arange(0.0, n_out / SR + GRID, GRID)
    tn_a, t0_a = np.asarray(tn), np.asarray(t0)
    seg = np.clip(np.searchsorted(tn_a, g_new, side='right') - 1, 0, len(tn_a) - 1)
    hold = 0.002
    rel = g_new - tn_a[seg]
    rel_old = np.where(rel < hold, rel, hold + (rel - hold) / stretch)
    told = t0_a[seg] + rel_old
    # a coherent model's ring-out goes on un-faded like its modes (module doc)
    end = np.append(t0_a[1:], L0 - (float(model.get('fade_out_s', 0.0)) if ring and coherent else 0.0))[seg]
    new_len = np.append(tn_a[1:] - tn_a[:-1], 1e9)[seg]
    old_len = np.append(t0_a[1:] - t0_a[:-1], 1e9)[seg]
    moved = (stretch != 1.0) | (np.abs(new_len - old_len) > 1e-9)
    cut = np.where(moved, end - GRID, end)
    extra = np.maximum(told - cut, 0.0)
    told = np.clip(np.minimum(told, cut), 0.0, L0)
    E = np.array([np.interp(told, g_old, row) for row in E0])
    if np.any(extra > 0):                         # past the old segment: keep falling at its last slope
        back = np.array([np.interp(np.maximum(told - 0.01, 0.0), g_old, row) for row in E0])
        E = E + np.clip((E - back) / 0.01, -3000.0, 0.0) * extra[None, :]
    if size != 1.0:                               # shift the noise spectrum with the modes
        lc = np.log2(cen)
        q = np.interp(lc + np.log2(size), lc, np.arange(len(cen)))
        i0 = np.clip(np.floor(q).astype(int), 0, len(cen) - 2)
        fr = (q - i0)[:, None]
        E = E[i0] * (1 - fr) + E[i0 + 1] * fr
    E = E + (brightness * np.log2(cen / 1000.0))[:, None] + noise
    if hardness != 1.0:                           # the contact's noise follows the hardness tilt
        hw = np.zeros(len(g_new))
        for t in tn:
            d = g_new - t
            hw = np.maximum(hw, np.where((d >= -0.001) & (d <= 0.004), 1.0,
                                         np.where((d > 0.004) & (d < 0.008), 1.0 - (d - 0.004) / 0.004, 0.0)))
        E = E + (4.0 * np.log2(hardness) * np.log2(cen / 1000.0))[:, None] * hw[None, :]
    if vary > 0:
        E = E + (1.5 * vary * rv.standard_normal(len(cen)))[:, None]
    sub = model.get('sub')
    sub = (int(round((tn[0] - t0[0]) * SR)), sub['x']) if sub else None
    return {'strikes': strikes, 'E': E, 'cen': cen, 'edges': edges, 'h': nz['h_erb'], 'n': n_out, 'sub': sub,
            'fade': 0.0 if ring else float(model.get('fade_out_s', 0.0)), 'ring': ring or None, 'told': told,
            'coherent': coherent}


def _ring_end(strikes, t_end):
    """(when every mode of the planned strikes has decayed RING_DB under the loudest strike's mode sum (s), the
    modes' summed level at t_end in dB re that sum); the contact modes are a few ms long and do not count"""
    ref = max([float(np.sqrt(np.sum(np.asarray(s[4]) ** 2))) for s in strikes] + [0.0])
    if ref <= 0:
        return 0.0, -300.0
    thr = ref * 10 ** (-RING_DB / 20)
    end, p_end = 0.0, 0.0
    for s in strikes:
        a = np.abs(np.asarray(s[4]))
        tau = np.asarray(s[3], float)
        k = a > thr
        if k.any():
            end = max(end, s[0] + float(np.max(tau[k] * np.log(a[k] / thr))))
        if t_end >= s[0]:
            p_end += float(np.sum((a * np.exp(-(t_end - s[0]) / np.maximum(tau, 1e-9))) ** 2))
    return end, float(10 * np.log10(p_end / ref ** 2 + 1e-30))


def _masks(n_fft, edges, h):
    """power-complementary zero-phase band masks on the rfft bins: cosine crossovers of half-width h on the ERB-rate
    axis (21.4 log10(1 + 0.00437 f))"""
    erb = lambda v: 21.4 * np.log10(1.0 + 0.00437 * v)
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    u = erb(f)
    le = erb(np.asarray(edges, float))
    s = np.clip((u[None, :] - le[:, None] + h) / (2 * h), 0.0, 1.0)
    nb = len(edges) + 1
    W = np.ones((nb, len(f)))
    for b in range(nb):
        if b > 0:
            W[b] *= np.sin(0.5 * np.pi * s[b - 1])
        if b < nb - 1:
            W[b] *= np.cos(0.5 * np.pi * s[b])
    return W


def _ring(L, f, tau, a, ph, block=4096):
    """sum of a * exp(-t / tau) * cos(2 pi f t + ph) over L samples: blocks of z^n times each block's start power"""
    y = np.zeros(L)
    if not len(f) or L <= 0:
        return y
    lz = -1.0 / (np.asarray(tau) * SR) + 2j * np.pi * np.asarray(f) / SR
    C = np.asarray(a) * np.exp(1j * np.asarray(ph))
    T = int(min(L, np.ceil(14 * np.max(tau) * SR) + 1))
    B = min(block, T)
    with np.errstate(under='ignore'):
        base = np.exp(np.arange(B)[:, None] * lz[None, :])
        for c0 in range(0, T, B):
            k = min(B, T - c0)
            y[c0:c0 + k] = (base[:k] @ (C * np.exp(c0 * lz))).real
    return y


def _modes(plan, n):
    """the strikes' modes (with their contact ramps and contact modes) and the sub wave, summed, faded"""
    y = np.zeros(n)
    for i, (t, ramp, f, tau, a, ph, fa, ta, aa, pa) in enumerate(plan['strikes']):
        o = int(round(t * SR))
        if o >= n:
            continue
        ys = _ring(n - o, f, tau, a, ph)
        R = int(round(ramp * SR))
        if R > 1:
            k = min(R, len(ys))
            ys[:k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / R)
        ys += _ring(n - o, fa, ta, aa, pa)                   # the contact: fast modes from the onset, no ramp
        R = int(round(STRIKE_RAMP_S * SR)) if i and plan.get('coherent') else 0     # ... a later strike's step is a
        if R > 1:                                            # click (STRIKE_RAMP_S; a 0.4 fit's, older ones as
            k = min(R, len(ys))                              # they were)
            ys[:k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / R)
        y[o:] += ys
    if plan.get('sub'):                                      # the slow pressure pulse (decimated waveform)
        shift, xs = plan['sub']
        w = signal.resample_poly(np.asarray(xs, float), SUB_DEC, 1)
        a0 = max(0, shift)
        w = w[max(0, -shift):][:max(0, n - a0)]
        y[a0:a0 + len(w)] += w
    nf = min(n, int(round(plan.get('fade', 0.0) * SR)))
    if nf > 0:                                               # the recording's fade-out (the noise bands carry it)
        y[n - nf:] *= np.clip((n - np.arange(n - nf, n)) / nf, 0.0, 1.0)
    return y


def _noise(plan, n, rng):
    """band noise: per band, white noise x the band's envelope, then the band's zero-phase mask"""
    n_fft = sfft.next_fast_len(n + 16384)
    W = _masks(n_fft, plan['edges'], plan['h'])
    g = np.mean(W ** 2, axis=1)
    ts = np.arange(n) / SR
    grid = np.arange(plan['E'].shape[1]) * GRID
    Y = np.zeros(W.shape[1], complex)
    for b in range(len(W)):
        amp = 10 ** (np.interp(ts, grid, plan['E'][b]) / 20) / np.sqrt(g[b])
        Y += np.fft.rfft(rng.standard_normal(n) * amp, n_fft) * W[b]
    return np.fft.irfft(Y, n_fft)[:n]


def _room(y, rng, wet, rt60=0.35):
    """a small stochastic room: exponentially decaying, slowly darkening noise per channel, unit energy"""
    L = int(rt60 * SR)
    t = np.arange(L) / SR
    out = y.copy()
    for c in range(y.shape[1]):
        ir = rng.standard_normal(L) * np.exp(-6.9 * t / rt60)
        ir = signal.lfilter([0.35], [1, -0.65], ir)
        ir[:int(0.004 * SR)] = 0.0
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
        out[:, c] += wet * signal.fftconvolve(y[:, c], ir)[:len(y)]
    return out


def _side_t(mid, sd, told, size=1.0, sign=1.0):
    """the side over time (model['side_t']) made from the mid: per octave band (log-frequency triangles that sum to
    one, the centres / size like the modes) the band's mid times its pan gain plus the band's Hilbert transform times
    its diffuse gain, the gains read at the original's time of each GRID frame (told: the plan's time map, so the
    keyframes move with the strikes and stretch with the decays) and linear between frames"""
    n = len(mid)
    n_fft = sfft.next_fast_len(n + 8192)
    F = np.fft.rfft(mid, n_fft)
    F[0] = 0.0
    lf = np.log2(np.maximum(np.fft.rfftfreq(n_fft, 1.0 / SR), 1.0))
    lc = np.log2(np.asarray(sd['bands'], float) / size)
    eye = np.eye(len(lc))
    tk = np.asarray(sd['times'], float)
    grid = np.arange(len(told)) * GRID
    ts = np.arange(n) / SR
    side = np.zeros(n)
    for b in range(len(lc)):
        W = np.interp(lf, lc, eye[b])
        pb, db = np.asarray(sd['pan'][b], float), np.asarray(sd['diffuse'][b], float)
        if np.abs(pb).max() > 1e-4:
            side += np.interp(ts, grid, np.interp(told, tk, pb)) * np.fft.irfft(F * W, n_fft)[:n]
        if db.max() > 1e-4:
            side += np.interp(ts, grid, np.interp(told, tk, db)) * np.fft.irfft(F * W * (-1j * sign), n_fft)[:n]
    return side


def render_model(model, seed=0, size=1.0, damping=1.0, brightness=0.0, hardness=1.0, noise=0.0, width=1.0,
                 room=0.0, vary=0.0, spacing=1.0, gain=0.0, stereo=None, ring=None):
    ring = vary > 0 if ring is None else bool(ring)
    plan = _plan(model, seed, size, damping, brightness, hardness, noise, vary, spacing, ring)
    n = plan['n']
    s = int(seed) & 0xFFFFFFFF
    mid = _modes(plan, n) + _noise(plan, n, np.random.default_rng([s, 2]))
    if plan.get('ring'):                      # the ring-out past the recording's length fades RING_DB (linear in dB)
        a, b = plan['ring']
        mid[a:b] *= 10 ** (-RING_DB / 20 * np.arange(b - a) / max(b - a, 1))
    two = (model.get('channels', 1) == 2) if stereo is None else bool(stereo)
    if two:
        side = np.zeros(n)
        rho = model.get('side')
        sign = -1.0 if np.random.default_rng([s, 3]).uniform() < 0.5 else 1.0
        if model.get('side_t') and width > 0:          # the side over time (0.4), made from the mid
            side = _side_t(mid, model['side_t'], plan['told'], size, sign) * width
        elif rho and width > 0:   # the side: the mid's Hilbert transform (uncorrelated with it at zero lag, same
            n_fft = sfft.next_fast_len(n + 16384)            # envelope), shaped per band by the original's
            W = _masks(n_fft, plan['edges'], plan['h'])      # side / mid ratio
            gain_f = np.sqrt(((np.asarray(rho) ** 2)[:, None] * W ** 2).sum(0)) * width
            H = np.fft.rfft(mid, n_fft) * gain_f * (-1j * sign)
            H[0] = 0.0
            side = np.fft.irfft(H, n_fft)[:n]
        y = np.stack([mid + side, mid - side], axis=1)
    else:
        y = mid[:, None]
    if room > 0:
        y = _room(y, np.random.default_rng([s, 5]), room, 0.25 + 0.15 * size)
    k = min(n, int(0.001 * SR))
    y[n - k:] *= np.linspace(1.0, 0.0, k)[:, None]
    y *= 10 ** (gain / 20)
    return y if two else y[:, 0]


def render(model, seed=0, **knobs):
    """re-synthesise a modal model: (n, 2) for a stereo original (or stereo=True), else (n,); knobs: size, damping,
    brightness, hardness, noise, width, room, vary, spacing, gain, stereo, ring (see the module doc)"""
    check(model)
    return render_model(model, seed, **knobs)


def check(model):
    if not isinstance(model, dict) or model.get('kind') != KIND:
        raise ValueError(f"not a modal model (kind {model.get('kind') if isinstance(model, dict) else type(model)})")
    if int(model.get('version', 0)) > VERSION:
        raise ValueError(f"modal model version {model.get('version')} is newer than this code ({VERSION})")


# ------------------------------------------------------------------------------------------------ io, code, checks
def save(model, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(model, fh, separators=(',', ':'))
    return path


def load(path):
    with open(path, encoding='utf-8') as fh:
        model = json.load(fh)
    check(model)
    return model


def summary(model):
    """a few lines on what the fit found"""
    lines = [f"modal v{model['version']}: {model['length_s'] * 1000:.0f} ms, {len(model['strikes'])} strike(s), "
             f"{model.get('fit', {}).get('modes', '?')} modes, modes carry "
             f"{model.get('fit', {}).get('mode_share_db', '?')} dB of the energy, residual "
             f"{model.get('fit', {}).get('resid_db', '?')} dB"]
    for s in model['strikes']:
        md = sorted(s['modes'], key=lambda q: -q[2] ** 2 * q[1])[:5]
        top = ', '.join(f'{q[0]:.0f} Hz / {q[1] * 1000:.1f} ms' for q in md)
        lines.append(f"  strike at {s['t'] * 1000:.1f} ms, contact {s['ramp_ms']} ms, {len(s['modes'])} modes; "
                     f"strongest: {top}")
    return '\n'.join(lines)


_CODE_FUNCS = (_plan, _ring_end, _masks, _ring, _modes, _noise, _room, _side_t, render_model)


def _literal(model):
    """the model as a readable Python literal (one mode per line)"""
    out = ['MODEL = {', f"    'kind': {model['kind']!r}, 'version': {model['version']}, 'sr': {model['sr']},",
           f"    'length_s': {model['length_s']!r}, 'channels': {model['channels']}, "
           f"'fade_out_s': {model.get('fade_out_s', 0.0)!r},", "    'strikes': ["]
    for s in model['strikes']:
        out.append(f"        {{'t': {s['t']!r}, 'ramp_ms': {s['ramp_ms']!r}, 'modes': [   # [Hz, decay s, amp, phase]")
        for q in s['modes']:
            out.append(f'            [{q[0]!r}, {q[1]!r}, {q[2]!r}, {q[3]!r}],')
        out.append("        ], 'attack': [   # the contact: fast modes from the onset")
        for q in s.get('attack', []):
            out.append(f'            [{q[0]!r}, {q[1]!r}, {q[2]!r}, {q[3]!r}],')
        out.append('        ]},')
    out.append('    ],')
    nz = model['noise']
    out.append(f"    'noise': {{'edges': {nz['edges']!r}, 'h_erb': {nz['h_erb']!r}, 'env': [   # per band: [t s, dB]")
    for env in nz['env']:
        out.append(f'        {json.dumps(env, separators=(",", ":"))},')
    out.append('    ]},')
    out.append(f"    'side': {model['side']!r},")
    if model.get('coherent'):
        out.append("    'coherent': True,   # vary and the knobs keep the modes' structure (0.4)")
    if model.get('side_t'):
        out.append(f"    'side_t': {json.dumps(model['side_t'], separators=(',', ':'))},   # the side over time")
    sub = model.get('sub')
    out.append(f"    'sub': {json.dumps(sub, separators=(',', ':')) if sub else None},   # waveform under 20 Hz")
    out.append('}')
    return '\n'.join(out)


def to_code(model, name='sound'):
    """a standalone Python script (numpy + scipy) that renders the model: `python <file> [seed] [out.wav]`; its
    render(seed, **knobs) returns the same samples as modal.render(model, seed, **knobs)"""
    check(model)
    head = (f'"""{name}: a modal model fitted by earshot.forge.fit.modal (kind {model["kind"]!r}, version '
            f'{model["version"]}).\n\n{summary(model)}\n\nrender(seed, size=, damping=, brightness=, hardness=, '
            f'noise=, width=, room=, vary=, spacing=, gain=) -> float array at 48 kHz.\n"""\n')
    body = [head, 'import sys', '', 'import numpy as np', 'from scipy import fft as sfft', 'from scipy import signal', '',
            f'SR = {SR}', f'GRID = {GRID!r}', f'SUB_DEC = {SUB_DEC}', f'RING_DB = {RING_DB!r}', f'RING_EXT = {RING_EXT!r}',
            f'RING_MIN_S = {RING_MIN_S!r}', f'RING_LOUD_DB = {RING_LOUD_DB!r}', f'RING_LOUD_S = {RING_LOUD_S!r}',
            f'STRIKE_RAMP_S = {STRIKE_RAMP_S!r}', '',
            _literal(model), '', '']
    for fn in _CODE_FUNCS:
        body.append(inspect.getsource(fn))
        body.append('')
    body.append('def render(seed=0, **knobs):\n    return render_model(MODEL, seed, **knobs)\n\n')
    body.append("if __name__ == '__main__':\n    from scipy.io import wavfile\n"
                "    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0\n"
                f"    out = sys.argv[2] if len(sys.argv) > 2 else '{name}.wav'\n"
                "    wavfile.write(out, SR, render(seed).astype(np.float32))\n")
    return '\n'.join(body)


def _melbank(n_fft, n_mels, fmin=30.0, fmax=20000.0):
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    mel = lambda v: 2595 * np.log10(1 + v / 700.0)
    imel = lambda v: 700 * (10 ** (v / 2595) - 1)
    pts = imel(np.linspace(mel(fmin), mel(min(fmax, SR / 2)), n_mels + 2))
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        a, c, b = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((f - a) / (c - a), (b - f) / (b - c)))
    return fb[fb.sum(1) > 0]


def mel_distance(x, y, frames=(256, 1024, 4096), floor_db=80.0):
    """multi-resolution log-mel distance (dB): per frame size (hop = frame / 4; 32 / 64 / 96 mel bands), the mean
    |dB difference| over the cells where either sound is within floor_db of x's loudest cell; returns
    (mean over sizes, [per size])"""
    a, b = _mono(x), _mono(y)
    n = max(len(a), len(b))
    a, b = np.pad(a, (0, n - len(a))), np.pad(b, (0, n - len(b)))
    out = []
    for nf, nm in zip(frames, (32, 64, 96)):
        hop = nf // 4
        _, _, A = signal.stft(a, SR, nperseg=nf, noverlap=nf - hop, boundary='zeros', padded=True)
        _, _, B = signal.stft(b, SR, nperseg=nf, noverlap=nf - hop, boundary='zeros', padded=True)
        fb = _melbank(nf, nm)
        dA = 10 * np.log10(fb @ np.abs(A) ** 2 + 1e-20)
        dB = 10 * np.log10(fb @ np.abs(B) ** 2 + 1e-20)
        fl = dA.max() - floor_db
        dA, dB = np.maximum(dA, fl), np.maximum(dB, fl)
        act = (dA > fl) | (dB > fl)
        out.append(float(np.mean(np.abs(dA - dB)[act])) if act.any() else 0.0)
    return float(np.mean(out)), out


def env_corr(x, y, hop_ms=2.0, floor_db=60.0):
    """Pearson correlation of the two dB envelopes (hop_ms RMS, floored floor_db under x's peak)"""
    a, b = _mono(x), _mono(y)
    n = max(len(a), len(b))
    a, b = np.pad(a, (0, n - len(a))), np.pad(b, (0, n - len(b)))
    h = int(SR * hop_ms / 1000)
    k = n // h
    ea = 10 * np.log10((a[:k * h].reshape(k, h) ** 2).mean(1) + 1e-20)
    eb = 10 * np.log10((b[:k * h].reshape(k, h) ** 2).mean(1) + 1e-20)
    fl = ea.max() - floor_db
    ea, eb = np.maximum(ea, fl), np.maximum(eb, fl)
    if ea.std() < 1e-9 or eb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def rise_ms(x):
    """onset sharpness: ms from 10 % to 90 % of the loudest peak of a 1 ms RMS envelope (0.125 ms hop), searched
    back over 30 ms"""
    m = _mono(x)
    w, h = 48, 6
    p = np.sqrt(np.convolve(m * m, np.ones(w) / w, mode='same'))[::h]
    i = int(np.argmax(p))
    lo = int(max(0, i - int(0.03 * SR / h)))
    seg = p[lo:i + 1]
    a = np.nonzero(seg >= 0.1 * p[i])[0]
    b = np.nonzero(seg >= 0.9 * p[i])[0]
    return float((b[0] - a[0]) * h / SR * 1000) if len(a) and len(b) else 0.0
