"""Building blocks for sounds written in code (48 kHz float). Physical first: real objects ring in modes, air moves
as filtered turbulence, every recording has a room and a microphone. Recipes in earshot/forge/recipes/ combine these.

    from earshot.forge import dsp
    r = dsp.rng(7)
    hit = dsp.modal([620, 1490, 2710], [0.035, 0.02, 0.012], [1, .5, .3], 0.3, r)   # a small plastic body ringing
    x = dsp.excite(hit, dsp.contact(0.0015, r))                                       # struck for 1.5 ms
    x = dsp.mic(dsp.room(x, r, rt60=0.18, wet=0.08), r)                               # in a room, through a mic

Designed sources for motion design (v2), all deterministic for a given rng:
    boom = dsp.sub_drop(1.6, r, f0=130, f1=32)              # pitch-dropping sine + harmonics + click
    ding = dsp.bell(1320, 1.2, r, 'bar', decay=0.5)          # inharmonic partials, doublet beating; or dsp.fm(...)
    rise = dsp.shepard(3.0, r, up=True, kind='noise')        # endless riser; dsp.sweep + dsp.osc for pitch sweeps
    gust = dsp.air(1.0, r, center=900, flutter=0.6)          # 1/f-ish turbulence with gusts and an optional whistle
    hit = dsp.stack([(boom, 0, 0), (ding, 0.0, -9, 0.3)], onset_db=-24, peak_db=-1)   # layers: offsets, gains, pans
Effects (reverb, saturation, transient shaping, limiting, EQ match, Doppler, glitch, grains) live in fx.py.

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


# --------------------------------------------------------------------------------------------- v2: curves + oscillators
def curve(spec, n, shape='lin'):
    """a per-sample control signal of length n from a scalar, an array (resampled to n if its length differs) or
    breakpoints [(t_seconds, value), ...] ('lin' or 'exp' interpolation, see env_curve).
        cut = dsp.curve([(0, 200), (1.5, 8000)], dsp.n_of(1.5), 'exp')"""
    if np.isscalar(spec):
        return np.full(n, float(spec))
    if isinstance(spec, (list, tuple)) and len(spec) and isinstance(spec[0], (list, tuple)):
        return env_curve([tuple(p) for p in spec], n / SR, shape)[:n]
    a = np.asarray(spec, np.float64).ravel()
    if len(a) == n:
        return a
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


def sweep(f0, f1, seconds, curve='exp', power=1.0):
    """a frequency trajectory (Hz per sample) from f0 to f1: 'exp' (equal octaves per second: the musical riser),
    'lin', or 'ease' (exponential with a power-law time warp: power > 1 starts slow and accelerates, the classic
    tension riser; power < 1 moves fast first and then settles, a downlifter).
        f = dsp.sweep(200, 6000, 3.0, 'ease', power=2.2); y = dsp.osc(f, 'saw')"""
    n = n_of(seconds)
    u = np.linspace(0, 1, n)
    if curve == 'lin':
        return f0 + (f1 - f0) * u
    if curve == 'ease':
        u = u ** power
    return f0 * (f1 / f0) ** u


@njit(cache=True)
def _blep(t, dt):
    if t < dt:
        t = t / dt
        return t + t - t * t - 1.0
    if t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


@njit(cache=True)
def _osc(freq, shape, ph):
    n = len(freq)
    y = np.empty(n)
    for i in range(n):
        dt = min(max(freq[i] / 48000.0, 1e-9), 0.49)
        if shape == 0:
            y[i] = np.sin(2 * np.pi * ph)
        elif shape == 1:
            y[i] = 2.0 * ph - 1.0 - _blep(ph, dt)
        else:
            v = 1.0 if ph < 0.5 else -1.0
            y[i] = v + _blep(ph, dt) - _blep((ph + 0.5) % 1.0, dt)
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
    return y


def osc(freq, shape='sine', seconds=None, phase=0.0):
    """band-limited oscillator (PolyBLEP saw / square, so sweeps up to 10 kHz do not alias): freq = Hz scalar (with
    seconds) or a per-sample trajectory (dsp.sweep); shape 'sine' | 'saw' | 'square' | 'tri'; peak about 1.
        y = dsp.osc(dsp.sweep(110, 880, 2.0), 'saw')"""
    f = np.asarray(freq, np.float64)
    if f.ndim == 0:
        f = np.full(n_of(seconds or 1.0), float(f))
    k = {'sine': 0, 'saw': 1, 'square': 2, 'tri': 3}[shape]
    y = _osc(np.ascontiguousarray(f), 2 if k == 3 else k, float(phase) % 1.0)
    if k == 3:                    # integrate the band-limited square (leaky, from -1): a triangle, harmonics -12 dB/oct
        y, _ = signal.lfilter([1.0], [1.0, -0.9995], y * 4 * f / SR, zi=[-0.9995 * (1 - 2 * (float(phase) % 1.0))])
    return y


def shepard(seconds, r, up=True, rate=0.5, f_lo=40.0, f_hi=10000.0, kind='tone', q=6.0, tilt=-1.5):
    """Shepard / Risset glissando: voices an octave apart glide forever (rate octaves per second, up or down) under a
    raised-cosine loudness bell over log-frequency, so the pitch seems to rise (or fall) without end: the endless
    riser / downlifter. kind 'tone' (sines) or 'noise' (resonant noise bands, q), tilt dB per octave. Unit RMS.
        rise = dsp.shepard(4.0, dsp.rng(1), up=True, rate=0.6, kind='noise')"""
    n = n_of(seconds)
    t = t_of(n)
    octs = np.log2(f_hi / f_lo)
    v = max(2, int(round(octs)))
    out = np.zeros(n)
    sgn = 1.0 if up else -1.0
    for k in range(v):
        u = (k / v + sgn * rate * t / octs) % 1.0
        f = f_lo * 2 ** (u * octs)
        a = (0.5 - 0.5 * np.cos(2 * np.pi * u)) * 10 ** (tilt * u * octs / 20)
        if kind == 'tone':
            out += a * np.sin(2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 2 * np.pi))
        else:
            out += a * svf(r.standard_normal(n), f, q, 'bandn') * np.sqrt(q)
    return out / (np.sqrt(np.mean(out ** 2)) + 1e-12)


# ------------------------------------------------------------------------------------------------ v2: air
def wander(seconds, r, rate=4.0, octaves=3):
    """a smooth 1/f-ish random control signal, unit std, mean 0: low-passed noises at rate, rate/3, rate/9 Hz with
    rising weights (slow drift dominates, fast flutter on top). For gusts, drifting pitch, moving pans.
        c = dsp.wander(2.0, dsp.rng(5), rate=6)"""
    n = n_of(seconds)
    m = max(16, int(n / SR * 1000) + 4)                        # computed at 1 kHz, then interpolated
    c = np.zeros(m)
    for i in range(octaves):
        fr = float(np.clip(rate / 3 ** i, 0.05, 400))
        pad = int(2000 / fr) + 8
        w = r.standard_normal(m + 2 * pad)
        sos = signal.butter(2, fr / 500, output='sos')
        c += signal.sosfiltfilt(sos, w)[pad:pad + m] / np.sqrt(fr) * 0.8 ** i
    c = np.interp(np.arange(n) / SR * 1000, np.arange(m), c)
    return (c - c.mean()) / (c.std() + 1e-12)


def air(seconds, r, center=1200.0, lo_db=6.0, hi_db=4.5, flutter=0.5, rate=6.0, bright=0.6, whistle=0.0):
    """moving-air turbulence: noise with a broad aerodynamic spectrum (its density rises lo_db per octave below
    `center` and falls hi_db per octave above: 3-5 is the 1/f-ish slope of real airflow; 1/3-octave band levels fall
    3 dB per octave less), gusting `flutter` (0..1) at about `rate` Hz
    with a 1/f wander, louder gusts also brighter (bright 0..1: faster air moves the spectrum up), and an optional
    `whistle` (0..1): a narrow resonance that wanders with the flow. Unit RMS; shape it with an envelope.
        w = dsp.air(1.2, dsp.rng(3), center=900, flutter=0.6) * dsp.env_curve([(0, 0), (.5, 1), (1.2, 0)], 1.2)"""
    n = n_of(seconds)
    F = np.fft.rfft(r.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    u = np.log2(np.maximum(f, 5.0) / center)
    F *= 10 ** (-10 * np.log10(2 ** (-lo_db / 3.0103 * u) + 2 ** (hi_db / 3.0103 * u)) / 20)
    x = np.fft.irfft(F, n)
    x /= np.sqrt(np.mean(x ** 2)) + 1e-12
    c = wander(seconds, r, rate)[:n]
    if bright > 0:
        x = svf(x, np.clip(center * 6 * 2 ** (1.2 * bright * flutter * c), 200, 20000), 0.6, 'low')
    if whistle > 0:
        c2 = 0.6 * c + 0.8 * wander(seconds, r, rate * 0.4)[:n]
        wh = svf(r.standard_normal(n), center * 2 ** (0.5 * c2), 9.0, 'bandn')
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12) + whistle * wh / (np.sqrt(np.mean(wh ** 2)) + 1e-12)
    x = x * np.exp(1.1 * flutter * c)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


# ------------------------------------------------------------------------------------------------ v2: designed sources
def sub_drop(seconds=1.6, r=None, f0=120.0, f1=34.0, drop=0.25, decay=0.6, harm=0.3, click=0.25, attack=0.004):
    """sub drop / boom: a sine whose pitch falls exponentially from f0 to f1 (time constant `drop` s), harmonics 2 and
    3 at `harm` (so it still reads on laptop speakers), a short bright `click` on the onset (0 = none), amplitude
    decaying with time constant `decay` s and a clean fade. Peak -1 dBFS.
        boom = dsp.sub_drop(1.8, dsp.rng(2), f0=140, f1=30, drop=0.3)"""
    r = r if r is not None else rng(0)
    n = n_of(seconds)
    t = t_of(n)
    f = f1 + (f0 - f1) * np.exp(-t / max(drop, 1e-3))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + harm * (0.55 * np.sin(2 * ph + 0.3) + 0.3 * np.sin(3 * ph + 0.7))
    e = np.exp(-t / max(decay, 1e-3))
    a = min(n, n_of(attack))
    e[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    y *= e
    if click > 0:
        k = min(n, n_of(0.012))
        tk = t_of(k)
        ck = butter(r.standard_normal(k), 'high', 1800) * np.exp(-tk / 0.0012)
        ck += 0.6 * np.sin(2 * np.pi * 2400 * tk) * np.exp(-tk / 0.003)
        y[:k] += click * ck / (np.abs(ck).max() + 1e-12)
    return normalize(fade(y, 0.0, seconds * 0.3), -1.0)            # a long fade: a sub must not stop audibly


BELLS = {   # partial ratios and amplitudes: church bell (hum, prime, tierce, quint, nominal, ...), tubular chime,
            # glockenspiel bar (free-free beam), wine glass, singing bowl
    'church': ([0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 2.662, 3.011, 4.166, 5.433, 6.796, 8.215],
               [0.55, 0.8, 0.7, 0.35, 1.0, 0.45, 0.4, 0.3, 0.25, 0.18, 0.12, 0.08]),
    'tubular': ([1.0, 2.756, 5.404, 8.933, 13.34], [0.35, 1.0, 0.7, 0.4, 0.2]),
    'bar': ([1.0, 2.756, 5.404, 8.933], [1.0, 0.3, 0.12, 0.06]),
    'glass': ([1.0, 2.32, 4.25, 6.63, 9.38], [1.0, 0.45, 0.25, 0.12, 0.06]),
    'bowl': ([1.0, 2.71, 5.13, 8.27], [1.0, 0.6, 0.3, 0.12]),
}


def bell(f0, seconds, r, partials='bar', decay=1.2, slope=0.8, bright=0.5, beat=1.0, stretch=0.0, strike=0.3):
    """additive bell / chime / ding with inharmonic partials (a BELLS preset name or (ratios, amps)): each partial
    decays with time constant decay * ratio^-slope (highs die first, as in real metal and glass) and is a doublet
    beating at about `beat` Hz (the slow shimmer of a real bell); bright 0..1 lifts the upper partials, stretch bends
    the ratios (ratio^(1+stretch)), strike adds a short mallet transient. Peak -1 dBFS.
        ding = dsp.bell(1320, 1.5, dsp.rng(4), 'bar', decay=0.5, bright=0.3)"""
    ratios, amps = BELLS[partials] if isinstance(partials, str) else partials
    n = n_of(seconds)
    t = t_of(n)
    y = np.zeros(n)
    for q, a in zip(ratios, amps):
        q = q ** (1 + stretch)
        f = f0 * q
        if f > SR * 0.45:
            continue
        a = a * q ** (1.2 * (bright - 0.5))
        b = beat * r.uniform(0.4, 1.6) * (1 if q >= 1 else 0.5)
        env = a * np.exp(-t / max(decay * q ** -slope, 1e-4))
        y += env * (np.sin(2 * np.pi * f * t + r.uniform(0, 6.3)) +
                    0.7 * np.sin(2 * np.pi * (f + b) * t + r.uniform(0, 6.3))) / 1.7
    a = min(n, n_of(0.0015))
    y[:a] *= np.linspace(0, 1, a)
    if strike > 0:
        k = min(n, n_of(0.006))
        s = butter(r.standard_normal(k), 'band', (f0 * 2, min(f0 * 12, 18000))) * np.exp(-t_of(k) / 0.0008)
        y[:k] += strike * s * np.abs(y).max() / (np.abs(s).max() + 1e-12)
    return normalize(fade(y, 0.0, seconds * 0.15), -1.0)


def fm(fc, ratio, index, seconds, decay=1.0, index_decay=0.3, attack=0.002):
    """two-operator FM (Chowning): carrier fc, modulator fc*ratio (non-integer ratios ring inharmonic: 1.4 is the
    classic bell, 3.5 a metallic ding), modulation index decaying from `index` with time constant index_decay (bright
    strike, pure tail), amplitude time constant `decay`. Peak about 1.
        ding = dsp.fm(880, 1.4, 6, 1.2, decay=0.4, index_decay=0.08)"""
    n = n_of(seconds)
    t = t_of(n)
    I = index * np.exp(-t / max(index_decay, 1e-4))
    y = np.sin(2 * np.pi * fc * t + I * np.sin(2 * np.pi * fc * ratio * t)) * np.exp(-t / max(decay, 1e-4))
    a = min(n, n_of(attack))
    y[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    return fade(y, 0.0, seconds * 0.15)


# ------------------------------------------------------------------------------------------------ v2: layering
def onset(x, db=-24.0):
    """seconds until |x| (mono or stereo) first reaches `db` under its peak: where a layer 'hits'"""
    m = np.abs(np.asarray(x, np.float64))
    m = m.max(axis=1) if m.ndim == 2 else m
    if not m.size or m.max() <= 0:
        return 0.0
    return float(np.argmax(m >= m.max() * 10 ** (db / 20))) / SR


def align(x, t=0.0, db=-24.0):
    """shift x (mono or stereo) so its onset (see onset) lands at t seconds: leading samples are trimmed or padded"""
    x = np.asarray(x, np.float64)
    d = n_of(t) * (t > 0) - int(round(onset(x, db) * SR))
    if d >= 0:
        return np.concatenate([np.zeros((d,) + x.shape[1:]), x])
    return x[-d:]


def pan_to_stereo(x, pan=0.0):
    """mono -> stereo with constant-power pan (-1 left .. 1 right; centre keeps the level of each channel = x);
    a stereo x is balanced the same way"""
    x = np.asarray(x, np.float64)
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    g = np.array([np.cos(th), np.sin(th)]) * np.sqrt(2)
    return x[:, None] * g if x.ndim == 1 else x * g


def stack(layers, length=None, onset_db=None, peak_db=None):
    """mix layers: each a dict {'x', 't' (s, default 0), 'gain' (dB, default 0), 'pan' (-1..1, optional)} or a tuple
    (x, t, gain_db[, pan]). onset_db (e.g. -24) aligns each layer's onset (not its first sample) to its t. Returns
    stereo (n, 2) when any layer is stereo or panned, else mono; length in s (default: until the last layer ends);
    peak_db normalises the mix (layers add up: two -1 dBFS layers can peak above 0 dBFS without it).
        hit = dsp.stack([(sub, 0, 0), (crack, 0.0, -6, -0.2), {'x': tail, 't': 0.01, 'gain': -12, 'pan': 0.3}])"""
    ls = []
    for L in layers:
        if not isinstance(L, dict):
            L = dict(zip(('x', 't', 'gain', 'pan'), L))
        x = np.asarray(L['x'], np.float64)
        t = float(L.get('t') or 0.0)
        if onset_db is not None:
            t -= onset(x, onset_db)
        if t < 0:
            x, t = x[n_of(-t):], 0.0
        ls.append((x, t, 10 ** (float(L.get('gain') or 0.0) / 20), L.get('pan')))
    st = any(x.ndim == 2 or p is not None for x, _, _, p in ls)
    n = n_of(length) if length else max(int(round(t * SR)) + len(x) for x, t, _, _ in ls)
    out = np.zeros((n, 2) if st else n)
    for x, t, g, p in ls:
        place(out, pan_to_stereo(x, p or 0.0) if st and (x.ndim == 1 or p) else x, t, g)
    return out if peak_db is None else normalize(out, peak_db)


def xfade(a, b, seconds=0.02, power=True):
    """a then b, overlapped by `seconds` with an equal-power (power=True: for unrelated material) or linear crossfade
    (power=False: for correlated material, e.g. two takes of the same tone); both mono or both stereo"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    k = min(n_of(seconds), len(a), len(b))
    u = np.linspace(0, 1, k)
    fo, fi = (np.cos(u * np.pi / 2), np.sin(u * np.pi / 2)) if power else (1 - u, u)
    if a.ndim == 2:
        fo, fi = fo[:, None], fi[:, None]
    return np.concatenate([a[:len(a) - k], a[len(a) - k:] * fo + b[:k] * fi, b[k:]])
