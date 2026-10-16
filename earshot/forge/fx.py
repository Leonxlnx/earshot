"""Effects for sounds written in code (48 kHz float, mono (n,) or stereo (n, 2)): space, dynamics, colour, motion,
glitch. Everything is deterministic (randomness only through an explicit rng) and computed: no recorded audio.

    from earshot.forge import dsp, fx
    r = dsp.rng(7)
    hit = dsp.sub_drop(1.2, r)                                                      # a computed source
    y = fx.saturate(hit, drive_db=9, mode="tape")                                   # colour, oversampled
    y = fx.transient(y, attack_db=4, sustain_db=3)                                  # punchier onset, longer tail
    y = fx.reverb(y, r, rt60=1.8, size=0.7, predelay=0.012, wet=0.3, stereo=True)   # smooth FDN hall
    y = fx.limit(y, ceiling_db=-1.0)                                                # never above -1 dBFS

Space      reverb (16-line modulated FDN, frequency-dependent decay, early reflections), early (reflections only);
           reverb IRs are cached (see reverb_ir: exact replays of one rng state, or ir_seed= fixed rooms)
Dynamics   saturate (tanh / tape / hard / fold, oversampled), transient (attack / sustain), limit (look-ahead)
Colour     split (Linkwitz-Riley bands that sum flat), match_curve / eq_bands / eq_match (fit a smooth 1/3-octave EQ
           so the tonal balance moves toward a target: tone.profile bands, minimum- or linear-phase FIR)
Motion     doppler (pass-by with distance, air absorption, pan), pan / width / decorrelate (static or over time),
           sweep_filter (resonant exponential sweeps for risers and downlifters), varispeed / tape_stop
Glitch     crush (bits + sample-rate), stutter, glitch (sliced repeat / reverse / drop / octave / crush), granular
           (grain cloud over a computed source), reverse_swell (a computed tail reversed into a hit)
"""
import collections
import threading

import numpy as np
from scipy import ndimage, signal

from .dsp import SR, butter, curve, n_of, njit, normalize, place, svf
from .dsp import rng as _rng

IR_CACHE_MB = 64                    # reverb IR cache budget (per process, LRU)
_IR_CACHE = collections.OrderedDict()
_IR_LOCK = threading.Lock()
_IR_STATS = {'hits': 0, 'misses': 0}


def ir_cache_stats():
    """{'hits', 'misses', 'entries', 'mb'} of the reverb IR cache"""
    with _IR_LOCK:
        return {**_IR_STATS, 'entries': len(_IR_CACHE),
                'mb': round(sum(v[0].nbytes for v in _IR_CACHE.values()) / 2 ** 20, 1)}


def ir_cache_clear():
    with _IR_LOCK:
        _IR_CACHE.clear()
        _IR_STATS.update(hits=0, misses=0)


def _ir_get(key):
    with _IR_LOCK:
        v = _IR_CACHE.get(key)
        if v is not None:
            _IR_CACHE.move_to_end(key)
            _IR_STATS['hits'] += 1
        else:
            _IR_STATS['misses'] += 1
        return v


def _ir_put(key, ir, post):
    with _IR_LOCK:
        _IR_CACHE[key] = (ir, post)
        _IR_CACHE.move_to_end(key)
        total = sum(v[0].nbytes for v in _IR_CACHE.values())
        while _IR_CACHE and total > IR_CACHE_MB * 2 ** 20:
            _, (old, _) = _IR_CACHE.popitem(last=False)
            total -= old.nbytes


def ir_grid(rt60=1.6, size=0.6, predelay=0.01, damp=0.5, bass=1.2, mod=1.0, diffuse=0.7, width=1.0, tail=None,
            lowcut=0.0):
    """reverb parameters snapped to a grid finer than anyone hears (times to 1/24 octave = 2.9 %, below the ~5 %
    reverb-time JND; size / damp / bass / mod / diffuse / width to 0.05; predelay to 0.5 ms; lowcut to 1/12 octave),
    so a search over continuous room settings keeps hitting the IR cache (reverb(..., ir_seed=k, grid=True))"""
    def lg(v, step):
        return float(2 ** (round(np.log2(v) / step) * step)) if v and v > 0 else v
    q = lambda v: round(round(float(v) / 0.05) * 0.05, 4)
    return dict(rt60=lg(rt60, 1 / 24), size=q(size), predelay=round(round(predelay / 0.0005) * 0.0005, 5), damp=q(damp),
                bass=q(bass), mod=q(mod), diffuse=q(diffuse), width=q(width),
                tail=None if tail is None else lg(tail, 1 / 24), lowcut=lg(lowcut, 1 / 12))


def _2d(x):
    x = np.asarray(x, np.float64)
    return (x[:, None], True) if x.ndim == 1 else (x, False)


def _back(y, mono):
    return y[:, 0] if mono else y


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12)


def _primes(hi):
    s = np.ones(hi + 1, bool)
    s[:2] = False
    for i in range(2, int(hi ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


# ------------------------------------------------------------------------------------------------ space
@njit(cache=True)
def _allpass(x, d, g):
    y = np.empty_like(x)
    buf = np.zeros(d)
    j = 0
    for i in range(len(x)):
        v = buf[j]
        u = x[i] + g * v
        y[i] = v - g * u
        buf[j] = u
        j += 1
        if j >= d:
            j = 0
    return y


@njit(cache=True)
def _fdn(xin, B, C, dl, depth, rate, phase, lb0, lb1, la1, hb0, hb1, ha1, n_out):
    N = len(dl)
    L = 1
    while L < int(dl.max() + depth.max() + 8):
        L *= 2
    msk = L - 1
    buf = np.zeros((N, L))
    zl = np.zeros(N)
    zh = np.zeros(N)
    ya = np.zeros(N)
    v = np.zeros(N)
    out = np.zeros((n_out, C.shape[1]))
    nin = xin.shape[0]
    nc = xin.shape[1]
    sc = 1.0 / np.sqrt(N)
    w = 0
    for i in range(n_out):
        for k in range(N):
            d = dl[k] + depth[k] * np.sin(2 * np.pi * rate[k] * i / 48000.0 + phase[k])
            M = int(np.floor(d - 0.5))                          # all-pass (Thiran) fractional read: unity gain at
            fd = d - M                                         # every frequency, so modulation does not darken
            eta = (1.0 - fd) / (1.0 + fd)
            y = eta * buf[k, (w - M) & msk] + buf[k, (w - M - 1) & msk] - eta * ya[k]
            ya[k] = y
            t = lb0[k] * y + zl[k]                             # low shelf (bass decay) then high shelf (damping)
            zl[k] = lb1[k] * y - la1[k] * t
            u = hb0[k] * t + zh[k]
            zh[k] = hb1[k] * t - ha1[k] * u
            v[k] = u
        for c in range(C.shape[1]):
            s = 0.0
            for k in range(N):
                s += C[k, c] * v[k]
            out[i, c] = s
        h = 1                                                  # fast Walsh-Hadamard mix (orthogonal, dense)
        while h < N:
            for a in range(0, N, 2 * h):
                for b in range(a, a + h):
                    p = v[b]
                    q = v[b + h]
                    v[b] = p + q
                    v[b + h] = p - q
            h *= 2
        for k in range(N):
            s = v[k] * sc
            if i < nin:
                for c in range(nc):
                    s += B[c, k] * xin[i, c]
            buf[k, w & msk] = s
        w += 1
    return out


def _shelf(G, fc, high):
    """first-order shelf coefficients (b0, b1, a1): DC gain G (low) or Nyquist gain G (high), the other end 1"""
    K = np.tan(np.pi * np.clip(fc, 20, SR * 0.45) / SR)
    s = np.sqrt(G)
    if high:
        b0, b1, a0, a1 = s + K, K - s, 1 / s + K, K - 1 / s
    else:
        b0, b1, a0, a1 = 1 + K * s, K * s - 1, 1 + K / s, K / s - 1
    return b0 / a0, b1 / a0, a1 / a0


def early(x, r, size=0.5, taps=18, damp=0.5, width=1.0):
    """early reflections only: a sparse tapped delay per channel (3 ms .. 6+50*size ms, 1/distance amplitudes, later
    taps darker), energy-normalised; stereo out (n + spread, 2).
        er = fx.early(click, dsp.rng(1), size=0.4)"""
    xm = np.asarray(x, np.float64)
    xm = xm.mean(axis=1) if xm.ndim == 2 else xm
    t_hi = 0.006 + 0.05 * size
    m = n_of(t_hi) + 2
    ir = np.zeros((m, 2))
    for c in range(2):
        tt = np.sort(np.exp(r.uniform(np.log(0.003), np.log(t_hi), taps)))
        if c == 1:
            tt = tt * (1 + width * r.uniform(-0.12, 0.12, taps))
        for k, ti in enumerate(tt):
            ir[min(m - 1, n_of(ti)), c] += r.choice([-1.0, 1.0]) * 0.003 / ti
    for c in range(2):                                         # distance darkening: split in thirds, darker later
        thirds = np.array_split(np.arange(m), 3)
        parts = []
        for j, idx in enumerate(thirds):
            seg = np.zeros(m)
            seg[idx] = ir[idx, c]
            parts.append(butter(seg, 'low', 14000 - damp * 4000 * (j + 1), order=1))
        ir[:, c] = np.sum(parts, axis=0)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True)) + 1e-12
    return np.stack([signal.fftconvolve(xm, ir[:, c]) for c in range(2)], axis=1)


def _late_noise(n, r, rt_f, f_axis, width):
    """Polack's late field: Gaussian noise whose every frequency decays at its own rt60 (STFT-shaped), unit energy at
    the median rt60; mid / side with the side high-passed at 200 Hz (the low end stays coherent, as in real rooms)"""
    out = []
    for _ in range(2):
        f, t, Z = signal.stft(r.standard_normal(n + 1024), SR, nperseg=1024, noverlap=768)
        rt = np.interp(f, f_axis, rt_f)
        Z *= np.exp(-6.91 * t[None, :] / rt[:, None])
        out.append(signal.istft(Z, SR, nperseg=1024, noverlap=768)[1][:n])
    m, sd = out[0], butter(out[1], 'high', 200)
    late = np.stack([m + width * sd, m - width * sd], axis=1) / np.sqrt(1 + width ** 2 * 0.9)
    return late * np.sqrt(13.8 / (np.median(rt_f) * SR))


def reverb_ir(r, rt60=1.6, size=0.6, predelay=0.01, damp=0.5, bass=1.2, mod=1.0, diffuse=0.7, early_db=-6.0,
              width=1.0, late='noise', lines=16, tail=None, inputs=1, hf_hz=4000.0, lf_hz=250.0, air=1.0, lowcut=0.0,
              seed=None, cache=True):
    """the impulse response reverb() convolves with, shape (n, inputs, 2) (_reverb_ir has the design notes).
    Cached (cache=True, IR_CACHE_MB, LRU, thread-safe), two ways:
      seed=None   the IR draws its randomness from r, as always; the cache key is r's exact bit-generator state plus
                  every parameter, and a hit sets r to the state the build would have left, so a render replayed
                  with the same (params, seed) is sample-identical and costs no IR build (forge's final re-render,
                  palette re-renders, render_params). Different seeds never hit.
      seed=int    a fixed room: the IR comes from dsp.rng(seed) and r is not touched (so the draws after it shift
                  against the seed=None path: opt in per recipe, then keep it). Cached by parameters: every candidate
                  of a search with the same room settings reuses one IR (with ir_grid() snapping, a search over rt60
                  hits too). A real reverb plugin is deterministic per setting; per-seed IR noise is not variation
                  anyone hears. Measured on the whoosh recipe: see reverb()."""
    args = (float(rt60), float(size), float(predelay), float(damp), float(bass), float(mod), float(diffuse),
            None if early_db is None else float(early_db), float(width), str(late), int(lines),
            None if tail is None else float(tail), int(inputs), float(hf_hz), float(lf_hz), float(air), float(lowcut))
    if seed is not None:
        key, src = ('seed', int(seed)) + args, None
    elif cache and hasattr(r, 'bit_generator'):
        src = r.bit_generator
        key = ('state', repr(src.state)) + args
    else:
        key, src = None, None
    if cache and key is not None:
        hit = _ir_get(key)
        if hit is not None:
            if src is not None:
                src.state = hit[1]
            return hit[0].copy()
    ir = _reverb_ir(_rng(seed) if seed is not None else r, *args)
    if cache and key is not None:
        _ir_put(key, ir.copy(), src.state if src is not None else None)
    return ir


def _reverb_ir(r, rt60=1.6, size=0.6, predelay=0.01, damp=0.5, bass=1.2, mod=1.0, diffuse=0.7, early_db=-6.0,
               width=1.0, late='noise', lines=16, tail=None, inputs=1, hf_hz=4000.0, lf_hz=250.0, air=1.0, lowcut=0.0):
    """the impulse response reverb() convolves with, shape (n, inputs, 2); unit energy at mid frequencies (a darker
    tail has less). late='noise': the FDN builds the onset and the first 0.1-0.2 s, then hands over (equal power, one
    shared decay design) to a stochastic late field whose every frequency decays at its own rt60. A real room's late
    reverb is Gaussian noise, so this tail never rings; a pure FDN (late='fdn') keeps resolvable modes in long tails
    (long-FFT peaks 13-16 dB over the local mean at rt60 2.5-5 s, against 8-10 dB for noise). air scales the
    absorption of air itself (about 0.17 dB per metre at 10 kHz, growing with f^2: a long tail darkens as it decays,
    10 kHz cannot ring longer than about 1 s); lowcut (Hz) high-passes the reverb (keeps a sub out of the room).
        ir = fx.reverb_ir(dsp.rng(1), rt60=2.0)"""
    nl = int(lines)
    assert nl in (4, 8, 16, 32), 'lines must be 4, 8, 16 or 32 (Hadamard mixing)'
    rt_lo, rt_hi = rt60 * bass, rt60 * max(0.05, 1 - 0.85 * damp)
    tail = 1.2 * max(rt60, rt_lo) if tail is None else tail
    pd = n_of(predelay) if predelay > 0 else 0
    n = max(n_of(tail), 64)
    lo = (0.006 + 0.024 * size) * SR                          # delays log-spread over [lo, 2.7 lo], distinct primes
    tgt = np.exp(np.linspace(np.log(lo), np.log(2.7 * lo), nl)) * np.exp(r.uniform(-0.06, 0.06, nl))
    pr = _primes(int(3.2 * lo) + 50)
    dl, used = [], set()
    for d in tgt:
        p = next(int(c) for c in pr[np.argsort(np.abs(pr - d))] if int(c) not in used)
        used.add(p)
        dl.append(p)
    dl = r.permutation(np.array(dl, np.float64))
    g = lambda rt: 10 ** (-3 * dl / (max(rt, 0.02) * SR))     # per-pass gain for a decay rt (s)
    gm, gl, gh = g(rt60), g(rt_lo), g(rt_hi)
    lb0, lb1, la1 = _shelf(gl / gm, lf_hz, False)
    hb0, hb1, ha1 = _shelf(gh / gm, hf_hz, True)
    lb0, lb1 = lb0 * gm, lb1 * gm
    ir = np.zeros((n, inputs, 2))
    for c in range(inputs):
        x = np.zeros(n)
        x[0] = 1.0
        if diffuse > 0:                                        # input diffusion: 4 series all-passes
            for j, d in enumerate((np.array([1.1, 1.7, 2.9, 4.3]) * (0.6 + 0.8 * size) * 1e-3 * SR).astype(int)):
                x = _allpass(np.ascontiguousarray(x), int(d + 3 * c + j), min(0.8, diffuse))
        B = r.choice([-1.0, 1.0], (1, nl)) / np.sqrt(nl)
        Cm = r.choice([-1.0, 1.0], (nl, 2))
        if abs(Cm[:, 0] @ Cm[:, 1]) >= nl:                     # equal / opposite columns (p = 2 / 2^nl): Gram-Schmidt
            Cm[nl // 2:, 1] *= -1.0                            # would zero column 1 (NaN IR); no extra rng draw
        Cm[:, 1] -= Cm[:, 0] * (Cm[:, 0] @ Cm[:, 1]) / nl      # Gram-Schmidt: decorrelated L / R taps
        Cm /= np.linalg.norm(Cm, axis=0, keepdims=True)
        depth = mod * (3 + 9 * r.uniform(0, 1, nl)) * (0.5 + 0.5 * size)
        y = _fdn(np.ascontiguousarray(x[:, None]), B, np.ascontiguousarray(Cm), dl, depth, r.uniform(0.15, 1.1, nl),
                 r.uniform(0, 2 * np.pi, nl), lb0, lb1, la1, hb0, hb1, ha1, n)
        y /= np.sqrt(np.mean(gm ** 2 / (1 - gm ** 2)) / nl)     # unit energy: 1/N reaches the taps per pass
        if late == 'noise':
            fa = np.geomspace(20, SR / 2 * 0.999, 64)          # the FDN's own decay per frequency, for the noise
            k = int(np.argmin(np.abs(dl - np.median(dl))))
            H = np.abs(signal.freqz([lb0[k], lb1[k]], [1, la1[k]], fa, fs=SR)[1] *
                       signal.freqz([hb0[k], hb1[k]], [1, ha1[k]], fa, fs=SR)[1])
            rt_f = -3 * dl[k] / (SR * np.log10(np.minimum(H, 0.999999)))
            tx = 0.06 + 0.1 * size
            u = np.clip((np.arange(n) / SR - 0.5 * tx) / tx, 0, 1)
            y = y * np.cos(u * np.pi / 2)[:, None] + _late_noise(n, r, rt_f, fa, 1.0) * np.sin(u * np.pi / 2)[:, None]
        if early_db is not None and early_db > -90:
            e = 10 ** (early_db / 10)
            y /= np.sqrt(1 + e)
            place(y, early(np.eye(1, n, 0)[0], r, size, damp=damp, width=width)[:n] * np.sqrt(e / (1 + e)), 0.0)
        m, sd = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
        ir[:, c] = np.stack([m + width * sd, m - width * sd], axis=1)
    if air > 0:                                                # air absorption: every bin decays a little more
        a_db = air * 0.17 * 343.0 * (np.fft.rfftfreq(1024, 1 / SR) / 10000) ** 2      # dB per second
        for c in range(inputs):
            for ch in range(2):
                f, t, Z = signal.stft(ir[:, c, ch], SR, nperseg=1024, noverlap=768, boundary='even')
                Z *= 10 ** (-a_db[:, None] * np.maximum(t[None, :] - 512 / SR, 0) / 20)
                ir[:, c, ch] = signal.istft(Z, SR, nperseg=1024, noverlap=768, boundary=True)[1][:n]
    if lowcut:
        ir = butter(ir, 'high', lowcut)
    k = min(n, n_of(min(0.1, tail * 0.2)))
    ir[-k:] *= (np.linspace(1, 0, k) ** 2)[:, None, None]
    return np.concatenate([np.zeros((pd, inputs, 2)), ir])


def reverb(x, r, rt60=1.6, size=0.6, predelay=0.01, damp=0.5, bass=1.2, mod=1.0, diffuse=0.7, early_db=-6.0,
           width=1.0, wet=0.3, dry=1.0, late='noise', lines=16, tail=None, stereo=None, air=1.0, lowcut=0.0,
           ir_seed=None, grid=False):
    """a smooth algorithmic reverb: a 16-line feedback delay network (Hadamard mix, mutually prime delays scaled by
    size 0..1, slowly modulated all-pass-interpolated taps) builds a dense onset, then a stochastic late field takes
    over (see reverb_ir: it never rings metallic); decay rt60 s at mid frequencies, damp 0..1 shortens it above 4 kHz
    (rt60 * (1 - 0.85 damp): soft walls) and air absorption (air, growing with f^2) darkens long tails further, bass
    multiplies it below 250 Hz, lowcut (Hz) keeps the lows dry; an all-pass input diffuser (diffuse 0..0.8: smooth
    onset, no flutter), early reflections at early_db under the tail, predelay s, stereo width 0..1 (the low end
    stays coherent). The wet IR has unit energy, so wet is the reverb level relative to the direct
    sound (0.3 = -10 dB). Returns len(x) + predelay + tail (default 1.2 * the longest rt60); mono in gives mono out
    (dry + the wet's mid) unless stereo=True. ir_seed=k: a fixed room (reverb_ir seed=; r untouched), cached by
    parameters; grid=True snaps the room parameters first (ir_grid) so a search keeps hitting the cache. Measured
    on the whoosh recipe's DEFAULTS (cProfile, laptop shared with other jobs): the IR build is 43 % of a render
    (3.0 of 7.0 s for 4 renders), so a recipe that opts in saves up to that share whenever its room settings repeat
    (a family with fixed rooms: every candidate after the first; a search over rt60 with grid=True: one build per
    2.9 % step). The default path stays sample-identical and only saves replays of the same (params, seed).
        y = fx.reverb(hit, dsp.rng(1), rt60=2.4, size=0.8, damp=0.6, wet=0.35, stereo=True)"""
    x2, mono = _2d(x)
    stereo = (not mono) if stereo is None else stereo
    if grid:
        g = ir_grid(rt60, size, predelay, damp, bass, mod, diffuse, width, tail, lowcut)
        rt60, size, predelay, damp, bass, mod, diffuse, width, tail, lowcut = (
            g['rt60'], g['size'], g['predelay'], g['damp'], g['bass'], g['mod'], g['diffuse'], g['width'], g['tail'],
            g['lowcut'])
    ir = reverb_ir(r, rt60, size, predelay, damp, bass, mod, diffuse, early_db, width, late, lines, tail, x2.shape[1],
                   air=air, lowcut=lowcut, seed=ir_seed)
    n_out = len(x2) + len(ir) - 1
    wetv = np.zeros((n_out, 2))
    for c in range(x2.shape[1]):
        wetv += signal.fftconvolve(x2[:, c:c + 1], ir[:, c], axes=0)
    if not stereo:                                             # mono: the mid, sqrt(2) keeps the wet energy
        y = wet * wetv.mean(axis=1) * np.sqrt(2)
        y[:len(x2)] += dry * x2.mean(axis=1)
        return y
    out = wet * wetv
    out[:len(x2)] += dry * (np.repeat(x2, 2, axis=1) if x2.shape[1] == 1 else x2)
    return out


# ------------------------------------------------------------------------------------------------ dynamics + colour
def saturate(x, drive_db=6.0, mode='tanh', mix=1.0, bias=0.0, os=None, level='rms'):
    """saturation, oversampled `os` times (Kaiser polyphase; default 4, 8 for 'hard' and 'fold') so harmonics do
    not alias (a 7 kHz sine driven 18 dB through tanh: aliases -53 dB under the harmonics, -14 dB without). Modes:
    'tanh' (soft clip), 'tape' (pre-emphasis, asymmetric soft clip with a little even-harmonic bias, de-emphasis,
    gentle top roll-off), 'hard' (clip), 'fold' (sine wavefolder, aggressive). drive_db into the curve, mix dry/wet,
    bias adds even harmonics; level 'rms' matches the input RMS, 'peak' the input peak, None leaves the curve's.
        y = fx.saturate(sub, drive_db=12, mode='tape')"""
    x2, mono = _2d(x)
    g = 10 ** (drive_db / 20)
    os = int(os or (8 if mode in ('hard', 'fold') else 4))
    win = ('kaiser', 9.0)
    u = signal.resample_poly(x2, os, 1, axis=0, window=win) if os > 1 else x2.copy()
    if mode == 'tape':
        u = _shelf_filter(u, 3000, 4.0, os)
        b = 0.12 + bias
        u = np.tanh(g * u + b) - np.tanh(b)
        u = _shelf_filter(u, 3000, -4.0, os)
    elif mode == 'hard':
        u = np.clip(g * u + bias, -1, 1) - np.clip(bias, -1, 1)
    elif mode == 'fold':
        u = np.sin(np.pi / 2 * (g * u + bias)) - np.sin(np.pi / 2 * bias)
    else:
        u = np.tanh(g * u + bias) - np.tanh(bias)
    y = signal.resample_poly(u, 1, os, axis=0, window=win)[:len(x2)] if os > 1 else u
    if mode == 'tape':
        y = butter(y, 'low', 17000, order=1)
    if bias or mode == 'tape':
        y = butter(y, 'high', 8, order=1)
    if level == 'rms':
        y *= _rms(x2) / _rms(y)
    elif level == 'peak':
        y *= np.abs(x2).max() / (np.abs(y).max() + 1e-12)
    return _back(mix * y + (1 - mix) * x2, mono)


def _shelf_filter(x, fc, gain_db, os=1):
    b0, b1, a1 = _shelf(10 ** (gain_db / 20), fc / os, True)
    return signal.lfilter([b0, b1], [1, a1], x, axis=0)


@njit(cache=True)
def _follow(x, att, rel):
    y = np.empty_like(x)
    e = 0.0
    for i in range(len(x)):
        a = att if x[i] > e else rel
        e = a * e + (1 - a) * x[i]
        y[i] = e
    return y


def _coef(ms):
    return float(np.exp(-1.0 / max(1.0, ms * 1e-3 * SR)))


def transient(x, attack_db=3.0, sustain_db=0.0, attack_ms=8.0, speed=1.0, floor_db=-60.0):
    """transient shaper (level independent): attack_db boosts (or, negative, softens) the first ~attack_ms of every
    onset, sustain_db lifts (or cuts) the decaying part progressively, e.g. a tail that dies too fast. Detection: the
    envelope against a slow-rising copy of itself finds onsets, a slow- against a fast-release envelope finds decays
    (a steady tone is left alone); speed scales the time constants; nothing below floor_db under the peak is lifted
    (no pumped noise floor). Stereo is processed linked. Boosts can push peaks over 0 dBFS: limit afterwards.
        y = fx.transient(impact, attack_db=4, sustain_db=5)"""
    x2, mono = _2d(x)
    m = np.ascontiguousarray(np.abs(x2).max(axis=1))
    s = speed
    ef = _follow(m, _coef(0.1 * s), _coef(15 * s))
    es = _rise(ef, _coef(attack_ms * s))
    a = np.clip((20 * np.log10((ef + 1e-9) / (es + 1e-9))) / 6.0, 0, 1)
    el = _follow(m, _coef(0.3 * s), _coef(350 * s))
    eq = _follow(m, _coef(0.3 * s), _coef(25 * s))
    d = np.clip((20 * np.log10((el + 1e-9) / (eq + 1e-9))) / 9.0, 0, 1) * (1 - a)
    gate = np.clip((20 * np.log10(el / (m.max() + 1e-12) + 1e-12) - floor_db) / 12.0, 0, 1)
    gdb = attack_db * a + sustain_db * d * gate
    gdb = _follow(gdb - gdb.min(), _coef(0.2), _coef(5)) + gdb.min()        # de-zipper
    return _back(x2 * 10 ** (gdb / 20)[:, None], mono)


@njit(cache=True)
def _rise(e, att):
    """a slow-rising copy of envelope e that may never exceed it (follows every fall at once)"""
    y = np.empty_like(e)
    s = 0.0
    for i in range(len(e)):
        s = att * s + (1 - att) * e[i]
        if s > e[i]:
            s = e[i]
        y[i] = s
    return y


@njit(cache=True)
def _release(g, rel):
    y = np.empty_like(g)
    s = 1.0
    for i in range(len(g)):
        s = 1.0 - (1.0 - s) * rel
        if g[i] < s:
            s = g[i]
        y[i] = s
    return y


def limit(x, ceiling_db=-1.0, lookahead=0.0025, release=0.08, true_peak=True):
    """one-shot look-ahead peak limiter (offline, zero latency): the gain reaches the needed reduction smoothly
    `lookahead` s before each peak and recovers with time constant `release`; true_peak measures 4x oversampled
    peaks; a final safety clip keeps every sample at or under the ceiling. Stereo is linked.
        y = fx.limit(mix, ceiling_db=-1.0)"""
    x2, mono = _2d(x)
    c = 10 ** (ceiling_db / 20)
    pk = np.abs(x2).max(axis=1)
    if true_peak:
        up = np.abs(signal.resample_poly(x2, 4, 1, axis=0)).max(axis=1)
        pk = np.maximum(pk, up[:len(up) // 4 * 4].reshape(-1, 4).max(axis=1)[:len(pk)])
    need = np.minimum(1.0, c / np.maximum(pk, 1e-12))
    L = max(1, n_of(lookahead))
    g = ndimage.minimum_filter1d(need, 2 * L + 1, mode='nearest')
    g = _release(np.ascontiguousarray(g), _coef(release * 1000))
    g = ndimage.uniform_filter1d(g, 2 * L + 1, mode='nearest')  # each average spans only gains <= the sample's need
    y = x2 * g[:, None]
    return _back(np.clip(y, -c, c), mono)


def split(x, freqs):
    """Linkwitz-Riley (4th order) band split at freqs (Hz, ascending): returns len(freqs) + 1 bands, low to high, that
    sum back to an all-pass of x (flat magnitude), so bands can be processed and re-summed without a notch.
        lo, mid, hi = fx.split(x, [200, 3000])"""
    def lr4(v, f, kind):
        sos = signal.butter(2, f, kind, fs=SR, output='sos')
        return signal.sosfilt(sos, signal.sosfilt(sos, v, axis=0), axis=0)
    bands, rest = [], np.asarray(x, np.float64)
    for f in sorted(freqs):
        bands = [lr4(b, f, 'low') + lr4(b, f, 'high') for b in bands]   # same all-pass on the bands already split
        bands.append(lr4(rest, f, 'low'))
        rest = lr4(rest, f, 'high')
    return bands + [rest]


def _target_profile(target):
    from .tone import ToneRef, profile
    if isinstance(target, ToneRef):
        return target.med
    if isinstance(target, (list, tuple)) and len(target) and not np.isscalar(target[0]):
        return ToneRef(list(target)).med
    a = np.asarray(target, np.float64)
    from .tone import CEN
    return a if a.shape == (len(CEN),) else profile(a)


def match_curve(x, target, amount=1.0, max_db=12.0, smooth=1):
    """the per-band EQ gains (dB, tone.CEN 1/3-octave bands, 31.5 Hz - 16 kHz) that move x's tonal balance toward
    `target`: a ToneRef, a list of reference files / arrays (their median profile), a 28-band profile (dB) or a
    signal. Centred on the loud bands (level is not the point), smoothed over neighbours `smooth` times, clipped to
    +-max_db; bands where x is near silent are not boosted more than 6 dB (EQ cannot create what is not there).
        g = fx.match_curve(x, ['real_01.wav', 'real_02.wav'])"""
    from .tone import profile
    p, t = profile(x), _target_profile(target)
    d = t - p
    w = 10 ** (p / 10)
    d -= np.sum(w * d) / np.sum(w)
    d = np.where(p < -50, np.minimum(d, 6.0), d)
    d = np.clip(d, -max_db, max_db)
    for _ in range(int(smooth)):
        d = np.convolve(np.pad(d, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
    return [round(float(v), 2) for v in amount * d]


def eq_bands(x, gains_db, phase='min', taps=4096):
    """apply per-band gains (dB at the tone.CEN centres, interpolated smoothly over log frequency, flat beyond the
    ends) as one FIR: phase 'min' (minimum phase: no pre-ringing on transients, default) or 'linear'. Same length as x.
        y = fx.eq_bands(x, fx.match_curve(x, refs))"""
    from .tone import CEN
    x = np.asarray(x, np.float64)
    nf = taps * 4
    f = np.fft.rfftfreq(nf, 1 / SR)
    gdb = np.interp(np.log2(np.maximum(f, 1.0)), np.log2(CEN), np.asarray(gains_db, np.float64))
    if phase == 'linear':
        H = 10 ** (gdb / 20)
        h = np.roll(np.fft.irfft(H, nf), taps // 2)[:taps] * np.hanning(taps)
        delay = taps // 2
    else:
        c = np.fft.irfft(np.log(10 ** (gdb / 20)), nf)        # real cepstrum -> fold -> minimum phase
        fold = np.zeros(nf)
        fold[0] = c[0]
        fold[1:nf // 2] = 2 * c[1:nf // 2]
        fold[nf // 2] = c[nf // 2]
        h = np.fft.irfft(np.exp(np.fft.rfft(fold)), nf)[:taps]
        h *= np.r_[np.ones(taps // 2), np.hanning(taps)[taps // 2:]]
        delay = 0
    y = signal.fftconvolve(x, h if x.ndim == 1 else h[:, None], axes=0)
    return y[delay:delay + len(x)]


def eq_match(x, target, amount=1.0, max_db=12.0, smooth=1, phase='min', iters=2):
    """move x's 1/3-octave tonal balance toward `target` (see match_curve) with a smooth EQ; iterated (re-measured)
    `iters` times; the output keeps the input's RMS.
        y = fx.eq_match(made, ['kit/impact/real_01.wav', ...], amount=0.8)"""
    from .tone import profile
    x = np.asarray(x, np.float64)
    tot = np.zeros(len(profile(x)))
    y = x
    for _ in range(max(1, int(iters))):                        # iterate the full correction, then scale it
        tot = np.clip(tot + np.asarray(match_curve(y, target, 1.0, max_db, smooth)), -max_db, max_db)
        y = eq_bands(x, tot, phase)
    if amount != 1.0:
        y = eq_bands(x, amount * tot, phase)
    return y * (_rms(x) / _rms(y))


# ------------------------------------------------------------------------------------------------ motion
def _hermite(x, pos):
    """cubic Hermite read of x (n,) or (n, c) at fractional positions (zeros outside)"""
    x2, mono = _2d(x)
    n = len(x2)
    i = np.floor(pos).astype(np.int64)
    fr = (pos - i)[:, None]
    def at(k):
        ok = (k >= 0) & (k < n)
        v = np.zeros((len(k), x2.shape[1]))
        v[ok] = x2[k[ok]]
        return v
    xm1, x0, x1, x2_ = at(i - 1), at(i), at(i + 1), at(i + 2)
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2_
    c3 = 0.5 * (x2_ - xm1) + 1.5 * (x0 - x1)
    return _back(((c3 * fr + c2) * fr + c1) * fr + x0, mono)


def doppler(x, speed=40.0, dist=4.0, t_close=None, air=1.0, width=0.8, stereo=True, c=343.0, floor_db=-30.0):
    """a source passing the listener on a straight line: x is what the source emits (mono); speed m/s, closest
    distance dist m, reached at source time t_close s (default the middle of x). Exact propagation delay (the pitch
    falls as it passes), 1/distance level (0 dB at the closest point, never below floor_db), air absorption growing
    with distance (high frequencies fade far away; air > 1 exaggerates it) and the pan following the angle (width).
    Output starts when the first sound arrives.
        car = fx.doppler(dsp.air(3, r, center=500), speed=30, dist=3)"""
    xm = np.asarray(x, np.float64)
    xm = xm.mean(axis=1) if xm.ndim == 2 else xm
    T = len(xm) / SR
    t0 = T / 2 if t_close is None else t_close
    v = max(1e-3, min(speed, 0.95 * c))
    rr = lambda te: np.hypot(v * (te - t0), dist)
    t_first, t_last = rr(0.0) / c, T + rr(T) / c
    t = t_first + np.arange(int((t_last - t_first) * SR)) / SR
    b = t - t0
    a = (c * c * b - np.sqrt(v * v * c * c * b * b + dist * dist * (c * c - v * v))) / (c * c - v * v)
    te = a + t0
    y = _hermite(xm, te * SR)
    rad = rr(te)
    y *= np.maximum(dist / rad, 10 ** (floor_db / 20))
    att = air * 0.17 * np.maximum(rad - dist, 0)               # dB lost at 10 kHz (air: about 0.17 dB / m)
    fc = 10000 / np.maximum(10 ** (att / 10) - 1, 1e-9) ** 0.25
    y = svf(y, np.clip(fc, 200, SR * 0.45), 0.707, 'low')
    if not stereo:
        return y
    s = np.clip(v * (te - t0) / rad, -1, 1) * width
    th = (s + 1) * np.pi / 4
    return np.stack([y * np.cos(th), y * np.sin(th)], axis=1) * np.sqrt(2)


def decorrelate(x, amount=1.0, cut=250.0):
    """mono -> stereo: side = the signal through three short prime all-passes (high-passed at `cut`, the low end stays
    mono), L = x + amount*side, R = x - amount*side. Deterministic.
        st = fx.decorrelate(whoosh, 0.6)"""
    xm = np.asarray(x, np.float64)
    xm = xm.mean(axis=1) if xm.ndim == 2 else xm
    s = np.ascontiguousarray(xm)
    for d, g in ((149, 0.6), (263, -0.55), (397, 0.5)):
        s = _allpass(s, d, g)
    s = butter(s, 'high', cut)
    s *= _rms(xm) / _rms(s) * 0.7
    return np.stack([xm + amount * s, xm - amount * s], axis=1) / np.sqrt(1 + 0.49 * amount ** 2)


def pan(x, pos):
    """constant-power pan, static or moving: pos is -1..1 as a scalar, a per-sample array or breakpoints
    [(t, pos), ...]; mono in -> stereo out; a stereo input is balanced.
        st = fx.pan(whoosh, [(0, -0.9), (0.6, 0.9)])"""
    x = np.asarray(x, np.float64)
    p = np.clip(curve(pos, len(x)), -1, 1)
    th = (p + 1) * np.pi / 4
    g = np.stack([np.cos(th), np.sin(th)], axis=1) * np.sqrt(2)
    return x[:, None] * g if x.ndim == 1 else x * g


def width(x, w):
    """stereo width over time (mid/side): w 0 = mono, 1 = as is, up to 2 = wider; scalar, array or breakpoints.
    A mono input is decorrelated first.
        st = fx.width(x, [(0, 0.2), (1.0, 1.4)])"""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = decorrelate(x)
    k = np.clip(curve(w, len(x)), 0, 2)
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2 * k
    return np.stack([m + s, m - s], axis=1)


def sweep_filter(x, f0, f1, q=4.0, curve_='exp', power=1.0, mode='bandn'):
    """a resonant filter sweeping exponentially from f0 to f1 Hz over the length of x (riser: up, downlifter: down);
    curve_ 'exp' | 'lin' | 'ease' with power (see dsp.sweep), mode as dsp.svf. Mono or stereo.
        rise = fx.sweep_filter(dsp.noise(3, r, 'pink'), 300, 9000, q=6, curve_='ease', power=2)"""
    from .dsp import sweep
    x2, mono = _2d(x)
    f = sweep(f0, f1, len(x2) / SR, curve_, power)[:len(x2)]
    f = np.pad(f, (0, len(x2) - len(f)), mode='edge')
    return _back(np.stack([svf(x2[:, c], f, q, mode) for c in range(x2.shape[1])], axis=1), mono)


def varispeed(x, speed):
    """play x at a speed that changes over time (1 = normal, 2 = an octave up, 0 = stopped): scalar, array or
    breakpoints over the OUTPUT time; ends when x runs out or the speed array ends.
        y = fx.varispeed(x, [(0, 1.0), (0.5, 0.25)])"""
    x = np.asarray(x, np.float64)
    if np.isscalar(speed):
        n = int(len(x) / max(float(speed), 1e-3))
        sp = np.full(n, float(speed))
    else:
        n = n_of(max(p[0] for p in speed)) if isinstance(speed, (list, tuple)) and np.ndim(speed[0]) else len(speed)
        sp = np.maximum(curve(speed, n), 0)
    pos = np.concatenate([[0.0], np.cumsum(sp[:-1])])
    pos = pos[pos < len(x) - 1]
    return _hermite(x, pos)


def tape_stop(x, start=None, seconds=0.6, curve_=1.6):
    """tape stop: normal speed until `start` s (default: so the stop ends with x), then the speed falls to 0 over
    `seconds` (curve_ > 1 holds speed longer, then drops), the pitch sliding down; a tiny fade at the very end.
        y = fx.tape_stop(chord, seconds=0.8)"""
    x = np.asarray(x, np.float64)
    T = len(x) / SR
    start = max(0.0, T - seconds * 0.6) if start is None else start
    n1, n2 = n_of(start), n_of(seconds)
    u = np.linspace(0, 1, n2)
    sp = np.concatenate([np.ones(n1), (1 - u) ** curve_])
    pos = np.concatenate([[0.0], np.cumsum(sp[:-1])])
    pos = pos[pos < len(x) - 1]
    y = _hermite(x, pos)
    k = min(len(y), n_of(0.005))
    y[-k:] *= np.linspace(1, 0, k) if y.ndim == 1 else np.linspace(1, 0, k)[:, None]
    return y


# ------------------------------------------------------------------------------------------------ glitch + grains
def crush(x, bits=8, rate=None, mix=1.0):
    """bit-crush (quantise to `bits`, no dither: the grit is the point) and sample-rate reduction (sample-and-hold at
    `rate` Hz, aliasing included on purpose).
        y = fx.crush(x, bits=6, rate=8000)"""
    x = np.asarray(x, np.float64)
    y = x
    if rate:
        idx = np.floor(np.floor(np.arange(len(x)) * rate / SR) * SR / rate).astype(np.int64)
        y = y[np.minimum(idx, len(x) - 1)]
    if bits:
        q = 2 ** (bits - 1)
        y = np.round(y * q) / q
    return mix * y + (1 - mix) * x


def _fade_edges(seg, k):
    k = min(k, len(seg) // 2)
    if k > 1:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
        w = w if seg.ndim == 1 else w[:, None]
        seg[:k] *= w
        seg[-k:] *= w[::-1]
    return seg


def stutter(x, start, length, repeats=4, gate=0.85, decay_db=0.0, pitch_step=0.0, fade=0.002):
    """stutter: the slice [start, start + length) s plays `repeats` times (each gated to `gate` of its slot, decay_db
    quieter and pitch_step semitones higher than the one before; micro-fades, no clicks), then x continues.
        y = fx.stutter(x, 0.0, 0.06, repeats=6, decay_db=-2, pitch_step=1)"""
    x = np.asarray(x, np.float64)
    i0, L = n_of(start) if start > 0 else 0, n_of(length)
    sl = x[i0:i0 + L]
    parts = [x[:i0]]
    for k in range(int(repeats)):
        rate = 2 ** (k * pitch_step / 12)
        seg = _hermite(sl, np.arange(L) * rate) if pitch_step else sl.copy()
        g = max(2, int(L * gate))
        seg[g:] = 0
        seg[:g] = _fade_edges(seg[:g].copy(), n_of(fade))
        parts.append(seg * 10 ** (k * decay_db / 20))
    parts.append(x[i0 + L:])
    return np.concatenate(parts)


def glitch(x, r, grid=0.03, repeat=0.25, reverse=0.1, drop=0.08, octave=0.08, crush_=0.08, fade=0.0015):
    """slice x on a `grid` (s) and re-arrange it: each slice is, with these probabilities, a repeat of the previous
    output slice, reversed, dropped (silence), an octave up (played twice at double speed) or crushed (6 bit, 6 kHz);
    otherwise kept. Micro-fades on every edge. Same length as x; deterministic for a given rng.
        g = fx.glitch(chord, dsp.rng(3), grid=0.045, repeat=0.3)"""
    x = np.asarray(x, np.float64)
    L = max(16, n_of(grid))
    out = np.zeros_like(x)
    prev = None
    k = n_of(fade)
    for i in range(0, len(x), L):
        seg = x[i:i + L].copy()
        u = r.uniform()
        acts = np.cumsum([repeat, reverse, drop, octave, crush_])
        if u < acts[0] and prev is not None and len(prev) == len(seg):
            seg = prev.copy()
        elif u < acts[1]:
            seg = seg[::-1].copy()
        elif u < acts[2]:
            seg[:] = 0
        elif u < acts[3]:
            half = seg[::2]
            seg = np.concatenate([_fade_edges(half.copy(), k), _fade_edges(half.copy(), k)])[:len(seg)]
            seg = np.pad(seg, [(0, len(x[i:i + L]) - len(seg))] + [(0, 0)] * (seg.ndim - 1))
        elif u < acts[4]:
            seg = crush(seg, 6, 6000)
        seg = _fade_edges(seg, k)
        out[i:i + len(seg)] = seg
        prev = seg
    return out


def granular(src, seconds, r, grain=0.06, density=40.0, pos=(0.0, 1.0), pos_jitter=0.02, pitch=0.0, pitch_jitter=0.0,
             time_jitter=0.5, spread=0.6, reverse=0.0):
    """granular cloud over a computed source: `density` grains per second of `grain` s (Hann windows), read from
    position `pos` (fraction of src: scalar, (start, end) scanned over the output, or breakpoints) +- pos_jitter,
    transposed pitch +- pitch_jitter semitones, onsets jittered by time_jitter (0 = a regular train, 1 = random),
    panned randomly within +-spread, a `reverse` share played backwards. Stereo, RMS about the source's.
        cloud = fx.granular(dsp.bell(880, 1, r), 3.0, r, grain=0.08, density=60, pos=(0.05, 0.6), pitch_jitter=0.3)"""
    s = np.asarray(src, np.float64)
    s = s.mean(axis=1) if s.ndim == 2 else s
    n = n_of(seconds)
    out = np.zeros((n, 2))
    if isinstance(pos, tuple) and len(pos) == 2 and np.isscalar(pos[0]):
        pc = np.linspace(pos[0], pos[1], n)
    else:
        pc = curve(pos, n)
    period = 1.0 / max(density, 1e-3)
    t = 0.0
    gl = n_of(grain)
    win = np.hanning(gl)
    count = 0
    while t < seconds:
        tt = t + time_jitter * period * r.uniform(-0.5, 0.5)
        i = n_of(max(tt, 0.0)) if tt > 0 else 0
        if i < n:
            rate = 2 ** ((pitch + pitch_jitter * r.standard_normal()) / 12)
            p0 = (pc[min(i, n - 1)] + pos_jitter * r.standard_normal()) * (len(s) - gl * rate)
            idx = np.clip(p0, 0, max(0, len(s) - gl * rate - 2)) + np.arange(gl) * rate
            gr = _hermite(s, idx) * win
            if r.uniform() < reverse:
                gr = gr[::-1]
            th = (np.clip(spread * r.uniform(-1, 1), -1, 1) + 1) * np.pi / 4
            m = min(gl, n - i)
            out[i:i + m, 0] += gr[:m] * np.cos(th) * np.sqrt(2)
            out[i:i + m, 1] += gr[:m] * np.sin(th) * np.sqrt(2)
            count += 1
        t += period
    overlap = max(1.0, density * grain * 0.375)                   # Hann power is 3/8 of the rectangle's
    return out / np.sqrt(overlap)


def reverse_swell(hit, r, seconds=1.0, rt60=2.5, size=0.8, damp=0.4, swell_db=-6.0, gap=0.0, hit_room=0.15):
    """a reverse swell into a hit: the hit's own computed reverb tail, reversed, rises for `seconds` and lands on the
    hit, whose onset sits exactly at `seconds` + gap in the output; swell_db sets the swell's peak against the hit's,
    hit_room adds a little of the same space after the hit. Stereo.
        y = fx.reverse_swell(dsp.sub_drop(1.2, r), r, seconds=1.2)"""
    h = np.asarray(hit, np.float64)
    hm = h.mean(axis=1) if h.ndim == 2 else h
    wet = reverb(hm, r, rt60=rt60, size=size, damp=damp, predelay=0.0, wet=1.0, dry=0.0, stereo=True,
                 tail=max(seconds, rt60) + 0.1)
    k = n_of(seconds)
    sw = wet[:k][::-1].copy()
    f = min(k, n_of(seconds * 0.35))
    sw[:f] *= (np.linspace(0, 1, f) ** 2)[:, None]
    hs = np.stack([h, h], axis=1) if h.ndim == 1 else h
    after = reverb(hm, r, rt60=rt60 * 0.6, size=size, damp=damp, predelay=0.0, wet=hit_room, dry=1.0,
                   stereo=True) if hit_room else hs
    sw *= np.abs(after).max() * 10 ** (swell_db / 20) / (np.abs(sw).max() + 1e-12)
    out = np.zeros((k + n_of(gap) * (gap > 0) + len(after), 2))
    place(out, sw, 0.0)
    place(out, after, seconds + gap)
    return out
