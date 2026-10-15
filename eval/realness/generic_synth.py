"""Generic procedural sounds written from scratch (independent of the cinetic engine) as negatives for the realness
detector: the kind of audio a coding agent writes when it "synthesizes a sound effect" in numpy.

    from generic_synth import render, TYPES
    x = render('beep', seed=3)        # float32 mono or (n, 2) at 48 kHz, peak 1

Every type draws its parameters from `seed`, so a (type, seed) pair always renders the same samples. Types cover
tones (beeps, chords, chiptune arps, FM bells, additive pads, Karplus-Strong plucks, sirens, DTMF), noise designs
(whooshes, risers, impacts, wind, rain, clicks, LFSR noise), a drum-machine loop and Schroeder reverb applied to some.
"""
import numpy as np
from scipy import signal

SR = 48000


def _n(d):
    return int(round(d * SR))


def _t(d):
    return np.arange(_n(d)) / SR


def _osc(kind, f, d, rng):
    """naive (aliasing) oscillators as code usually writes them; f may be an array (glide)"""
    t = _t(d)
    f = np.resize(np.asarray(f, float), t.shape)
    ph = np.cumsum(f) / SR + rng.random()
    if kind == 'sine':
        return np.sin(2 * np.pi * ph)
    if kind == 'square':
        return np.sign(np.sin(2 * np.pi * ph))
    if kind == 'saw':
        return 2 * (ph % 1.0) - 1
    return 2 * np.abs(2 * (ph % 1.0) - 1) - 1


def _env(d, rng, kind=None):
    t = _t(d)
    kind = kind or rng.choice(['exp', 'lin', 'adsr', 'gate'])
    if kind == 'exp':
        return np.exp(-t / (d * rng.uniform(0.1, 0.4)))
    if kind == 'lin':
        return np.clip(1 - t / d, 0, 1)
    if kind == 'gate':
        e = np.ones_like(t)
        k = int(0.002 * SR)
        e[:k] = np.linspace(0, 1, k)
        e[-k:] = np.linspace(1, 0, k)
        return e
    a, dc, s, r = rng.uniform(0.002, 0.05), rng.uniform(0.02, 0.2), rng.uniform(0.3, 0.8), rng.uniform(0.05, 0.3)
    return np.interp(t, [0, a, a + dc, max(a + dc, d - r), d], [0, 1, s, s, 0])


def _bandpass_sweep(x, f_lo, f_hi, q=2.0, block=512):
    """block-wise band-pass whose centre follows f_lo -> f_hi (arrays allowed)"""
    n = len(x)
    fc = np.geomspace(f_lo, f_hi, n) if np.isscalar(f_lo) else np.asarray(f_lo)
    y = np.zeros(n)
    zi = None
    for s in range(0, n, block):
        c = float(np.clip(fc[min(n - 1, s + block // 2)], 60, SR * 0.4))
        sos = signal.butter(2, [c / (1 + 0.5 / q), min(c * (1 + 0.5 / q), SR * 0.45)], 'band', fs=SR, output='sos')
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y[s:s + block], zi = signal.sosfilt(sos, x[s:s + block], zi=zi)
    return y


def _pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X[1:] /= np.sqrt(f[1:])
    X[0] = 0
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def schroeder(x, rng, rt=None, mix=None):
    """classic Schroeder reverb: 4 parallel combs + 2 allpasses (the textbook algorithmic reverb)"""
    rt = rt or rng.uniform(0.6, 2.5)
    mix = mix if mix is not None else rng.uniform(0.2, 0.5)
    x = np.concatenate([x, np.zeros(int(rt * SR))])
    wet = np.zeros_like(x)
    for dl in (1557, 1617, 1491, 1422):
        g = 10 ** (-3 * dl / SR / rt)
        wet += signal.lfilter([1.0], np.r_[1.0, np.zeros(dl - 1), -g], x)
    for dl, g in ((225, 0.5), (556, 0.5)):
        wet = signal.lfilter(np.r_[-g, np.zeros(dl - 1), 1.0], np.r_[1.0, np.zeros(dl - 1), -g], wet)
    return (1 - mix) * x + mix * wet / (np.abs(wet).max() + 1e-12) * np.abs(x).max()


def beep(rng):
    d = rng.uniform(0.05, 0.6)
    f = rng.choice([rng.uniform(300, 3000), 440.0, 880.0, 1000.0, 1200.0])
    if rng.random() < 0.3:
        f = np.geomspace(f, f * rng.choice([0.5, 1.5, 2.0]), _n(d))
    return _osc(rng.choice(['sine', 'sine', 'square', 'saw', 'tri']), f, d, rng) * _env(d, rng)


def chord(rng):
    d = rng.uniform(0.5, 3.0)
    root = 110 * 2 ** (rng.integers(0, 24) / 12)
    kind = rng.choice(['sine', 'saw', 'square', 'tri'])
    ivs = [[0, 4, 7], [0, 3, 7], [0, 4, 7, 11], [0, 7, 12, 16]][rng.integers(0, 4)]
    x = sum(_osc(kind, root * 2 ** (iv / 12), d, rng) for iv in ivs)
    if kind != 'sine':
        x = signal.sosfilt(signal.butter(2, rng.uniform(800, 5000), 'low', fs=SR, output='sos'), x)
    return x * _env(d, rng, 'adsr')


def chiptune(rng):
    bpm = rng.uniform(100, 180)
    step = 60 / bpm / rng.choice([2, 4])
    root = 220 * 2 ** (rng.integers(0, 12) / 12)
    scale = [0, 2, 4, 5, 7, 9, 11, 12]
    kind = rng.choice(['square', 'square', 'tri', 'saw', 'sine'])
    out = []
    for _ in range(int(rng.uniform(1.5, 4.0) / step)):
        n = _osc(kind, root * 2 ** (scale[rng.integers(0, 8)] / 12), step, rng) * _env(step, rng, rng.choice(['gate', 'exp']))
        out.append(n * 0.6)
    return np.concatenate(out)


def fm(rng):
    d = rng.uniform(0.8, 3.5)
    t = _t(d)
    fc = 200 * 2 ** (rng.uniform(0, 4))
    ratio, idx, tau = rng.choice([1.0, 1.4, 2.0, 3.5, 7.0]), rng.uniform(1, 8), rng.uniform(0.2, 1.5)
    x = np.sin(2 * np.pi * fc * t + idx * np.exp(-t / (tau * 0.5)) * np.sin(2 * np.pi * fc * ratio * t))
    return x * np.exp(-t / tau) * np.minimum(1, t / 0.002)


def pad(rng):
    d = rng.uniform(2.0, 4.0)
    t = _t(d)
    f0 = 80 * 2 ** rng.uniform(0, 3)
    x = sum((1 / k ** rng.uniform(0.8, 2)) * np.sin(2 * np.pi * f0 * k * t + rng.random() * 6) *
            (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 1.5) * t)) for k in range(1, 12))
    return x * _env(d, rng, 'adsr')


def pluck(rng):
    """Karplus-Strong string"""
    f = 110 * 2 ** rng.uniform(0, 3)
    d = rng.uniform(0.6, 2.0)
    n, p = _n(d), int(SR / f)
    buf = rng.uniform(-1, 1, p)
    out = np.zeros(n)
    k = rng.uniform(0.990, 0.999)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = k * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out


def siren(rng):
    d = rng.uniform(1.0, 3.0)
    t = _t(d)
    f = rng.uniform(500, 1200) + rng.uniform(100, 500) * np.sin(2 * np.pi * rng.uniform(0.5, 6) * t)
    return _osc(rng.choice(['sine', 'square', 'saw']), f, d, rng) * _env(d, rng, 'gate')


def dtmf(rng):
    rows, cols = [697, 770, 852, 941], [1209, 1336, 1477]
    out = []
    for _ in range(rng.integers(3, 9)):
        d = rng.uniform(0.08, 0.2)
        t = _t(d)
        out += [np.sin(2 * np.pi * rows[rng.integers(0, 4)] * t) + np.sin(2 * np.pi * cols[rng.integers(0, 3)] * t),
                np.zeros(int(rng.uniform(0.05, 0.12) * SR))]
    return np.concatenate(out)


def whoosh(rng):
    d = rng.uniform(0.4, 2.0)
    n = _n(d)
    src = _pink(n, rng) if rng.random() < 0.5 else rng.standard_normal(n)
    f = np.geomspace(rng.uniform(150, 600), rng.uniform(1500, 6000), n)
    if rng.random() < 0.5:
        f = np.concatenate([f[: n // 2], f[: n - n // 2][::-1]])
    return _bandpass_sweep(src, f, None, q=rng.uniform(1, 4)) * np.hanning(n) ** rng.uniform(0.7, 2)


def riser(rng):
    d = rng.uniform(1.0, 4.0)
    n = _n(d)
    u = np.arange(n) / n
    x = _bandpass_sweep(rng.standard_normal(n), np.geomspace(200, rng.uniform(4000, 10000), n), None) * u ** 2
    if rng.random() < 0.6:
        x += 0.5 * _osc(rng.choice(['saw', 'sine', 'square']), np.geomspace(rng.uniform(80, 200), rng.uniform(800, 2000), n), d, rng) * u ** 2
    return x


def impact(rng):
    d = rng.uniform(0.5, 2.5)
    t = _t(d)
    f = rng.uniform(35, 60) + rng.uniform(60, 200) * np.exp(-t / rng.uniform(0.03, 0.2))
    x = _osc('sine', f, d, rng) * np.exp(-t / rng.uniform(0.2, 0.8))
    x += rng.uniform(0.2, 0.8) * rng.standard_normal(len(t)) * np.exp(-t / rng.uniform(0.01, 0.08))
    return np.tanh(x * rng.uniform(1, 3))


def wind(rng):
    d = rng.uniform(2.0, 4.0)
    n = _n(d)
    t = _t(d)
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.random() * 6)
    fc = rng.uniform(300, 900) * (0.7 + 0.6 * gust)
    return _bandpass_sweep(_pink(n, rng), fc, None, q=rng.uniform(0.7, 2)) * gust


def rain(rng):
    d = rng.uniform(2.0, 4.0)
    n = _n(d)
    x = _pink(n, rng) * 0.05
    for _ in range(int(d * rng.uniform(30, 300))):
        i = rng.integers(0, n - 2000)
        k = np.arange(int(rng.uniform(100, 1500)))
        x[i:i + len(k)] += rng.uniform(0.1, 1) * np.sin(2 * np.pi * rng.uniform(1500, 6000) * k / SR) * np.exp(-k / rng.uniform(50, 400))
    return signal.sosfilt(signal.butter(2, rng.uniform(300, 1000), 'high', fs=SR, output='sos'), x)


def click(rng):
    d = rng.uniform(0.02, 0.15)
    t = _t(d)
    x = rng.standard_normal(len(t)) * np.exp(-t / rng.uniform(0.001, 0.01))
    if rng.random() < 0.5:
        x = signal.sosfilt(signal.butter(2, sorted(rng.uniform(500, 8000, 2)), 'band', fs=SR, output='sos'), x)
    return x


def lfsr(rng):
    """8-bit console noise channel: a 15-bit LFSR clocked at a low rate, gated in bursts"""
    d = rng.uniform(0.2, 1.5)
    n = _n(d)
    hold = int(SR / rng.uniform(2000, 16000))
    reg, bits = 1, []
    for _ in range(n // hold + 1):
        b = (reg ^ (reg >> 1)) & 1
        reg = (reg >> 1) | (b << 14)
        bits.append(1.0 if reg & 1 else -1.0)
    return np.repeat(bits, hold)[:n] * _env(d, rng, rng.choice(['exp', 'lin']))


def drums(rng):
    bpm = rng.uniform(90, 140)
    step = 60 / bpm / 4
    n = int(16 * step * SR)
    out = np.zeros(n + SR)
    t = _t(0.5)
    kick = np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.15)
    snare = rng.standard_normal(len(t)) * np.exp(-t / 0.06) * 0.6 + np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05) * 0.4
    hat = signal.sosfilt(signal.butter(2, 7000, 'high', fs=SR, output='sos'), rng.standard_normal(len(t))) * np.exp(-t / 0.02) * 0.4
    for s in range(16):
        i = int(s * step * SR)
        for hit, on in ((kick, s % 4 == 0), (snare, s % 8 == 4), (hat, s % 2 == 0 or rng.random() < 0.3)):
            if on:
                out[i:i + len(hit)] += hit
    return out[:n]


TYPES = {f.__name__: f for f in (beep, chord, chiptune, fm, pad, pluck, siren, dtmf, whoosh, riser, impact, wind, rain,
                                  click, lfsr, drums)}


def render(kind, seed=0):
    """one sound of `kind` from `seed`: float32, peak 1, mono (stereo for a random third, as code often widens)"""
    rng = np.random.default_rng(1000 * (list(TYPES).index(kind) + 1) + seed)
    x = TYPES[kind](rng).astype(np.float64)
    if kind not in ('wind', 'rain', 'drums') and rng.random() < 0.3:
        x = schroeder(x, rng)
    x = x / (np.abs(x).max() + 1e-12)
    if rng.random() < 0.33:
        dl = int(rng.uniform(0.0003, 0.02) * SR)
        x = np.stack([x, np.r_[np.zeros(dl), x[:-dl]]], axis=1)
    return x.astype(np.float32)
