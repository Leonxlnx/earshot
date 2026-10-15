"""Building blocks for sounds written in code (48 kHz float). Physical first: real objects ring in modes, air moves
as filtered turbulence, every recording has a room and a microphone. Recipes in earshot/forge/recipes/ combine these.

    from earshot.forge import dsp
    r = dsp.rng(7)
    hit = dsp.modal([620, 1490, 2710], [0.035, 0.02, 0.012], [1, .5, .3], 0.3, r)   # a small plastic body ringing
    x = dsp.excite(hit, dsp.contact(0.0015, r))                                       # struck for 1.5 ms
    x = dsp.mic(dsp.room(x, r, rt60=0.18, wet=0.08), r)                               # in a room, through a mic

No recorded audio is used anywhere: every sample is computed.
"""
import numpy as np
from scipy import signal

SR = 48000


def rng(seed):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def n_of(seconds):
    return max(1, int(round(seconds * SR)))


def t_of(n):
    return np.arange(n) / SR


# ------------------------------------------------------------------------------------------------ noise
def noise(seconds, r, color='white'):
    """white / pink (1/f) / brown (1/f^2) noise, unit RMS"""
    n = n_of(seconds)
    w = r.standard_normal(n)
    if color != 'white':
        F = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1 / SR)
        f[0] = f[1]
        F /= f ** (0.5 if color == 'pink' else 1.0)
        w = np.fft.irfft(F, n)
    return (w / (np.sqrt(np.mean(w ** 2)) + 1e-12)).astype(np.float64)


# ------------------------------------------------------------------------------------------------ envelopes
def env_exp(seconds, tau, attack=0.0005):
    """fast linear attack then exponential decay with time constant tau (s)"""
    n = n_of(seconds)
    t = t_of(n)
    e = np.exp(-t / max(tau, 1e-5))
    a = n_of(attack)
    if a > 1:
        e[:a] *= np.linspace(0, 1, a)
    return e


def env_curve(points, seconds, shape='lin'):
    """piecewise envelope through [(t, value), ...]; shape 'lin' or 'exp' (interpolated in dB, values > 0)"""
    n = n_of(seconds)
    t = t_of(n)
    pts = sorted(points)
    tx, vx = [p[0] for p in pts], [p[1] for p in pts]
    if shape == 'exp':
        return 10 ** (np.interp(t, tx, 20 * np.log10(np.maximum(vx, 1e-6))) / 20)
    return np.interp(t, tx, vx)


def contact(seconds, r, bright=0.7):
    """the excitation of a strike: a short raised-cosine force pulse plus a little contact noise (bright 0..1)"""
    n = max(2, n_of(seconds))
    pulse = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, n))
    grit = r.standard_normal(n) * pulse * bright
    return pulse + grit


# ------------------------------------------------------------------------------------------------ resonators
def modal(freqs, decays, amps, seconds, r, jitter=0.003, glide=0.0):
    """sum of exponentially damped sinusoids (a body's modes): freqs Hz, decays = time constants s, amps linear.
    jitter: random detune per mode (fraction); glide: pitch drop per mode over the decay (fraction, e.g. 0.01)"""
    n = n_of(seconds)
    t = t_of(n)
    out = np.zeros(n)
    for f, d, a in zip(freqs, decays, amps):
        if f >= SR / 2 * 0.95 or a == 0:
            continue
        f = f * (1 + jitter * r.standard_normal())
        inst = f * (1 - glide * (1 - np.exp(-t / max(d, 1e-4))))
        ph = 2 * np.pi * np.cumsum(inst) / SR + r.uniform(0, 2 * np.pi)
        out += a * np.exp(-t / max(d, 1e-5)) * np.sin(ph)
    return out


def modal_cloud(f_lo, f_hi, count, decay_lo, decay_hi, seconds, r, tilt=-3.0):
    """many modes spread log-uniformly between f_lo and f_hi (dense bodies: plastic shells, wood, cases); higher modes
    decay faster (decay interpolated log from decay_lo at f_lo to decay_hi at f_hi), tilt dB/octave"""
    f = np.sort(np.exp(r.uniform(np.log(f_lo), np.log(f_hi), int(count))))
    u = np.log(f / f_lo) / max(1e-9, np.log(f_hi / f_lo))
    d = np.exp(np.log(decay_lo) + u * (np.log(decay_hi) - np.log(decay_lo)))
    a = 10 ** (tilt * np.log2(f / f_lo) / 20) * r.uniform(0.3, 1.0, len(f))
    return modal(f, d, a, seconds, r, jitter=0.0)


def excite(resonance, excitation):
    """drive a resonator response (its impulse response) with an excitation signal"""
    y = signal.fftconvolve(excitation, resonance)[: len(resonance)]
    return y


# ------------------------------------------------------------------------------------------------ filters
def butter(x, kind, f, order=2):
    """kind 'low' | 'high' | 'band' (f = (lo, hi)) | 'bandstop'"""
    f = np.clip(f, 10, SR / 2 * 0.95)
    sos = signal.butter(order, f, kind if kind != 'band' else 'bandpass', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def peq(x, f0, gain_db, q=0.9):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * min(f0, SR / 2 * 0.95) / SR
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    b = np.array([1 + al * A, -2 * c, 1 - al * A])
    a = np.array([1 + al / A, -2 * c, 1 - al / A])
    return signal.lfilter(b / a[0], a / a[0], x, axis=0)


def svf(x, cutoff, q=0.7, mode='band'):
    """time-varying state-variable filter (Cytomic TPT): cutoff and q may be arrays (one value per sample); mode
    'low' | 'band' | 'high' | 'bandn'. Use it for moving resonances: whooshes, swells, sweeps.
    Gains: 'low' and 'high' are 1 in their pass bands (q above 0.707 adds a peak of about q at the cutoff); 'band' peaks
    at q (not 1) at the cutoff, so a q of 8 is +18 dB; 'bandn' is the same band-pass normalised to a peak gain of 1"""
    n = len(x)
    fc = np.broadcast_to(np.clip(np.asarray(cutoff, float), 20, SR * 0.45), (n,))
    qq = np.broadcast_to(np.asarray(q, float), (n,))
    return _svf(np.asarray(x, np.float64), np.ascontiguousarray(fc), np.ascontiguousarray(qq),
                {'low': 0, 'band': 1, 'high': 2, 'bandn': 3}[mode])


try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _svf(x, fc, q, mode):
    y = np.empty_like(x)
    ic1 = 0.0
    ic2 = 0.0
    for i in range(len(x)):
        g = np.tan(np.pi * fc[i] / 48000.0)
        k = 1.0 / max(q[i], 0.05)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1
        elif mode == 3:
            y[i] = k * v1
        else:
            y[i] = x[i] - k * v1 - v2
    return y


# ------------------------------------------------------------------------------------------------ space + mic
def room_ir(r, rt60=0.25, size=0.5, damp=0.6, seconds=None):
    """a small procedural room impulse response: a few early reflections (size 0..1 sets their spread) and an
    exponentially decaying noise tail whose highs die faster (damp 0..1)"""
    seconds = seconds or min(2.0, rt60 * 1.2 + 0.05)
    n = n_of(seconds)
    t = t_of(n)
    tail = r.standard_normal(n) * np.exp(-6.91 * t / max(rt60, 0.02))
    lo = butter(tail, 'low', 1500 + (1 - damp) * 9000, order=1)
    mix = np.interp(t, [0, seconds], [0.0, 1.0]) ** 0.7
    tail = (1 - mix * damp) * tail + mix * damp * lo
    ir = tail * 0.3
    ir[0] += 1.0
    for _ in range(6):
        d = int(r.uniform(0.002, 0.004 + 0.02 * size) * SR)
        if d < n:
            ir[d] += r.uniform(0.15, 0.45) * r.choice([-1, 1])
    return ir


def room(x, r, rt60=0.25, size=0.5, damp=0.6, wet=0.12):
    """place x in a small room (wet 0..1 = reverb share); output is as long as x plus the tail. x may be mono (n,)
    or stereo (n, 2); a stereo x gets the same room on both channels (for a wide room, build one IR per channel)"""
    ir = room_ir(r, rt60, size, damp)
    ir[0] = 0.0
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        y = signal.fftconvolve(x, ir[:, None], axes=0)
        out = np.zeros(y.shape)
        out[: len(x)] += x
        return out + wet * y / (np.sqrt(np.sum(ir ** 2)) + 1e-9)
    y = signal.fftconvolve(x, ir)
    out = np.zeros(len(y))
    out[: len(x)] += x
    return out + wet * y / (np.sqrt(np.sum(ir ** 2)) + 1e-9)


def mic(x, r, low_cut=35.0, air_db=-2.0, floor_db=-66.0, color=None):
    """a microphone and preamp: gentle low cut, a slightly soft top (air_db at 10 kHz), an optional presence colour
    (color=(f0, gain_db)), and a pink noise floor at floor_db relative to the peak (real recordings are never silent).
    x may be mono (n,) or stereo (n, 2); a stereo x gets an independent noise floor per channel"""
    y = butter(x, 'high', low_cut, order=1)
    y = peq(y, 10000, air_db, 0.6)
    if color:
        y = peq(y, color[0], color[1], 0.8)
    pk = np.abs(y).max() + 1e-12
    if y.ndim == 2:
        fl = np.stack([noise(len(y) / SR, r, 'pink')[:len(y)] for _ in range(y.shape[1])], axis=1)
        return y + fl * pk * 10 ** (floor_db / 20)
    y = y + noise(len(y) / SR, r, 'pink') * pk * 10 ** (floor_db / 20)
    return y


def stereo(x, r, width=0.3, pan=0.0):
    """mono -> stereo with a decorrelated side (two short random all-pass-like delays) and constant-power pan"""
    d1, d2 = int(r.uniform(0.2, 0.9) * 1e-3 * SR), int(r.uniform(0.2, 0.9) * 1e-3 * SR)
    side = np.zeros_like(x)
    side[d1:] += x[: len(x) - d1]
    side[d2:] -= x[: len(x) - d2]
    side = butter(side, 'high', 300)
    L, R = x + width * side, x - width * side
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.stack([L * np.cos(th) * np.sqrt(2), R * np.sin(th) * np.sqrt(2)], axis=1)


# ------------------------------------------------------------------------------------------------ finishing
def fade(x, fin=0.001, fout=0.02):
    x = np.array(x, dtype=np.float64)
    a, b = min(len(x), n_of(fin)), min(len(x), n_of(fout))
    if a > 1:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
        x[:a] *= w if x.ndim == 1 else w[:, None]
    if b > 1:
        w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b))
        x[-b:] *= w if x.ndim == 1 else w[:, None]
    return x


def place(dst, x, t, gain=1.0):
    """add x into dst starting at t seconds (both mono or both stereo)"""
    i = n_of(t) if t > 0 else 0
    if i >= len(dst):
        return dst
    m = min(len(x), len(dst) - i)
    dst[i:i + m] += gain * x[:m]
    return dst


def trim_tail(x, floor_db=-70.0, keep=0.01):
    """cut trailing samples quieter than floor_db (relative to the peak), keeping `keep` s after the last loud one"""
    m = np.abs(x if x.ndim == 1 else x.max(axis=1))
    loud = np.nonzero(m >= m.max() * 10 ** (floor_db / 20))[0]
    end = min(len(x), (loud[-1] if len(loud) else len(x)) + n_of(keep))
    return x[:end]


def normalize(x, peak_db=-1.0):
    return x * (10 ** (peak_db / 20) / (np.abs(x).max() + 1e-12))
