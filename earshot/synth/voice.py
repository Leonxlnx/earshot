"""A synthesizer voice with the craft of a hardware polysynth built in, rendered offline (48 kHz float, stereo (n, 2)).

A raw oscillator under a linear envelope is thin, static and harsh. What makes a synth patch sound finished is
specific, and each piece is here:

  oscillators  saw / pulse (PWM) / triangle / sine without aliasing: every step or corner of the waveform is replaced
               by a band-limited one (a windowed-sinc step 64 samples long, placed at its sub-sample time), which also
               makes hard sync and audio-rate FM clean; unison stacks with the uneven detune of a supersaw, a louder
               centre voice, free-running phases, per-voice pan, slow pitch drift and a few ms of onset smear
  filters      a zero-delay-feedback 4-pole ladder (resonance in tune up to self-oscillation, soft-limited feedback),
               next to the 2-pole state-variable filter of forge.dsp; cutoff moves per sample
  envelopes    ADSR with capacitor curves, never a step: the output is slewed like a real VCA, so it starts at zero
  FM           phase modulation between up to six operators, each with its own envelope, ratio or fixed frequency,
               feedback, velocity and key scaling; oversampled, with the index held under the alias limit
  motion       LFOs (sine, triangle, smooth random), vibrato that arrives late, drift
  effects      a BBD-style chorus / string ensemble, a ping-pong tape echo with damped feedback, a phaser

    from earshot.forge import dsp
    from earshot.synth import voice
    r = dsp.rng(3)
    y = voice.analog(220.0, 0.8, 1.5, r, osc=(dict(shape='saw', voices=5, detune=14, spread=0.8),), cutoff=900,
                     env=1.5, fenv=(0.4, 1.2, 0.5, 1.0), amp=(0.3, 1.0, 0.8, 1.2))      # a slow unison pad note
    y = voice.chorus(y, rate=0.513, depth=1.85)                                         # through the ensemble
    bell = voice.fm(880.0, [dict(out=1.0, mod=(1,), env=(0.001, 3.0, 0.0, 1.5)),
                            dict(ratio=3.5, index=2.0, env=(0.001, 0.8, 0.0, 0.5))], 0.8, 0.3)

earshot.synth.patches builds the patch library on these.
"""
import numpy as np
from scipy import ndimage, signal

from ..forge import dsp, fx
from ..forge.dsp import SR, n_of, njit

C4 = 261.6256
SHAPES = {'saw': 0, 'pulse': 1, 'square': 1, 'tri': 2, 'sine': 3}
_W = 32                      # half-width of a band-limited step, samples
_L = 128                     # table points per sample
_HOP = 256                   # samples between the control points of a voice's pitch drift
_FM_LIMIT = 1.5 * SR         # at 96 kHz, anything under 72 kHz folds above 24 kHz, where the decimator removes it
_HALF = signal.firwin(127, 22000.0, fs=2 * SR, window=('kaiser', 8.0))    # 96 -> 48 kHz: flat to 20, -80 dB from 24


def _tables():
    """the band-limited unit step (the integral of a Kaiser-windowed sinc: flat to 20 kHz, 80 dB down from 24 kHz)
    and the residual of its own integral (a band-limited corner), _L points per sample over +-_W samples"""
    m = np.arange(-_W * _L, _W * _L + 1) / _L
    h = np.sinc(0.92 * m) * np.kaiser(len(m), 8.0)
    step = np.cumsum(h) - 0.5 * (h + h[0])
    step /= step[-1]
    ramp = (np.cumsum(step) - 0.5 * (step + step[0])) / _L - np.maximum(m, 0.0)
    return np.append(step, 1.0), np.append(ramp, 0.0)


_STEP, _RAMP = _tables()


# ------------------------------------------------------------------------------------------------ oscillators
@njit(cache=True)
def _add(out, i, a, gl, gr, tab, jump):
    """add the band-limited version of a discontinuity at time i + a (0 < a <= 1) to out, whose row k is sample
    k - _W: a unit step (jump, tab = the step table) or a unit change of slope (tab = the ramp residual)"""
    a = min(max(a, 1e-9), 1.0)
    p = (1.0 - a) * _L
    k = int(p)
    f = p - k
    for m in range(2 * _W):
        v = tab[k] + f * (tab[k + 1] - tab[k])
        if jump and m >= _W:
            v -= 1.0
        out[i + 1 + m, 0] += gl * v
        out[i + 1 + m, 1] += gr * v
        k += _L


@njit(cache=True)
def _advance(out, i, a0, a1, ph, dt, shape, w, gl, gr, step, ramp):
    """move an oscillator from phase ph across the part a0..a1 of sample i and correct every step (saw and pulse
    edges) or corner (triangle peaks) it passes; returns the new phase"""
    if dt <= 0.0:
        return ph
    nxt = ph + (a1 - a0) * dt
    if shape == 1 and ph < w <= nxt:
        _add(out, i, a0 + (w - ph) / dt, -2.0 * gl, -2.0 * gr, step, True)
    elif shape == 2 and ph < 0.5 <= nxt:
        _add(out, i, a0 + (0.5 - ph) / dt, -8.0 * dt * gl, -8.0 * dt * gr, ramp, False)
    if nxt >= 1.0:
        a = a0 + (1.0 - ph) / dt
        nxt -= 1.0
        if shape == 0:
            _add(out, i, a, -2.0 * gl, -2.0 * gr, step, True)
        elif shape == 1:
            _add(out, i, a, 2.0 * gl, 2.0 * gr, step, True)
        elif shape == 2:
            _add(out, i, a, 8.0 * dt * gl, 8.0 * dt * gr, ramp, False)
    return nxt


@njit(cache=True)
def _voice(out, freq, rat, shape, pw, srat, ph, gl, gr, start, fade, step, ramp):
    """one oscillator voice added into out (rows offset by _W): freq Hz per sample times the ratio interpolated from
    rat (one value per _HOP samples: detune and drift); pw per sample for a pulse; srat (per sample) makes it a slave
    at srat times the pitch that restarts with every cycle of the pitch (hard sync); it starts at sample `start`
    with a fade-in over `fade` samples"""
    n = len(freq)
    sync = len(srat) == n
    wide = len(pw) == n
    mph = ph
    tau = 2.0 * np.pi
    w = pw[0]
    for i in range(start, n):
        k = i // _HOP
        dtm = freq[i] * (rat[k] + (i - k * _HOP) * (rat[k + 1] - rat[k]) * (1.0 / _HOP)) * (1.0 / SR)
        dtm = min(max(dtm, 0.0), 0.49)
        dt = min(dtm * srat[i], 0.49) if sync else dtm
        g = 1.0
        if i - start < fade:
            u = (i - start) / fade
            g = u * u * (3.0 - 2.0 * u)
        al = gl * g
        ar = gr * g
        if wide:
            w = pw[i]
        if shape == 0:
            v = 2.0 * ph - 1.0
        elif shape == 1:
            v = (1.0 if ph < w else -1.0) - (2.0 * w - 1.0)
        elif shape == 2:
            v = 1.0 - 4.0 * abs(ph - 0.5)
        else:
            v = np.sin(tau * ph)
        out[i + _W, 0] += al * v
        out[i + _W, 1] += ar * v
        if sync and mph + dtm >= 1.0:
            a = (1.0 - mph) / dtm
            ph = _advance(out, i, 0.0, a, ph, dt, shape, w, al, ar, step, ramp)
            if shape == 0:                                 # the restart: a step from the value at ph to the value at 0
                _add(out, i, a, -2.0 * ph * al, -2.0 * ph * ar, step, True)
            elif shape == 1:
                if ph >= w:
                    _add(out, i, a, 2.0 * al, 2.0 * ar, step, True)
            elif shape == 2:
                d = 4.0 * abs(ph - 0.5) - 2.0
                _add(out, i, a, d * al, d * ar, step, True)
                if ph >= 0.5:
                    _add(out, i, a, 8.0 * dt * al, 8.0 * dt * ar, ramp, False)
            else:
                d = -np.sin(tau * ph)
                _add(out, i, a, d * al, d * ar, step, True)
                d = tau * dt * (1.0 - np.cos(tau * ph))
                _add(out, i, a, d * al, d * ar, ramp, False)
            ph = _advance(out, i, a, 1.0, 0.0, dt, shape, w, al, ar, step, ramp)
            mph += dtm - 1.0
        else:
            nxt = ph + dt
            if nxt >= 1.0 or (shape == 1 and ph < w <= nxt) or (shape == 2 and ph < 0.5 <= nxt):
                ph = _advance(out, i, 0.0, 1.0, ph, dt, shape, w, al, ar, step, ramp)
            else:
                ph = nxt
            mph += dtm


def unison(freq, shape='saw', r=None, voices=1, detune=0.0, spread=0.0, blend=0.5, drift=0.0, smear=0.0, pw=0.5,
           sync=None, phase=None):
    """a stack of 1-9 band-limited oscillators as stereo (n, 2), unit power. freq: Hz per sample. The stack follows
    the supersaw (Szabo, "How to Emulate the Super Saw", KTH 2010): the voices are not evenly detuned (the measured
    offsets of 7 voices are 0.18 : 0.57 : 1 of the outermost, here the law |k|^1.5 with the sharp side 2 % narrower),
    the centre voice is louder (the paper's mix curves at `blend` 0..1) and every voice runs free (random start
    phase), so no two notes begin alike. detune: cents of the outermost voice; spread 0..1: pan of the outermost
    voices, sharp and flat alternating sides; drift: cents of slow random pitch wander per voice (an analogue
    oscillator never holds still, so stacked voices do not lock into a comb); smear: s, the side voices start up to
    this late (and fade in over 2 ms). pw: pulse width 0..1, scalar or per sample; sync: ratio (scalar or per sample)
    of a hard-synced slave to the pitch; phase: one start phase 0..1 for every voice instead of random ones (a bass
    that must hit the same way every time). r is needed whenever something is random.
        pad = voice.unison(np.full(96000, 220.0), 'saw', dsp.rng(1), voices=7, detune=18, spread=0.8, drift=2)"""
    freq = np.ascontiguousarray(freq, np.float64)
    n = len(freq)
    u = int(np.clip(voices, 1, 9))
    k = np.arange(u) - (u - 1) / 2
    far = np.abs(k) / max((u - 1) / 2, 1e-9)
    cents = detune * np.sign(k) * far ** 1.5 * np.where(k > 0, 0.977, 1.0)
    g = np.where(np.abs(k) < 0.25, 0.99785 - 0.55366 * blend, 0.044372 + 1.2841 * blend - 0.73764 * blend ** 2)
    g = g / np.sqrt(np.sum(g ** 2)) if u > 1 else np.ones(1)
    th = (np.clip(spread * far * np.sign(k) * (-1.0) ** np.ceil(np.abs(k)), -1, 1) + 1) * np.pi / 4
    ph = r.uniform(0, 1, u) if phase is None else np.full(u, float(phase) % 1.0)
    st = (r.uniform(0, smear, u) * SR).astype(np.int64) * (np.abs(k) >= 0.25) if smear > 0 else np.zeros(u, np.int64)
    pts = n // _HOP + 2
    pwa = np.ascontiguousarray(np.clip(np.atleast_1d(np.asarray(pw, np.float64)), 0.03, 0.97))
    srat = np.zeros(1) if sync is None else np.ascontiguousarray(np.broadcast_to(np.asarray(sync, np.float64), (n,)))
    out = np.zeros((n + 2 * _W, 2))
    for v in range(u):
        d = drift * wobble(pts, 0.8, r, SR / _HOP) if drift > 0 else np.zeros(pts)
        _voice(out, freq, 2.0 ** ((cents[v] + d) / 1200), SHAPES[shape], pwa, srat, ph[v],
               g[v] * np.cos(th[v]) * np.sqrt(2), g[v] * np.sin(th[v]) * np.sqrt(2), min(int(st[v]), n),
               n_of(0.002) if st[v] else 0, _STEP, _RAMP)
    y = out[_W:_W + n]
    return dsp.butter(y, 'high', 18.0, order=1) if sync is not None else y      # a synced wave is not DC-free


def osc(freq, shape='saw', pw=0.5, sync=None, phase=0.0, seconds=None):
    """one band-limited oscillator, mono (n,), peak about 1: freq Hz as a scalar (with seconds) or per sample, so
    another oscillator can modulate it (linear FM: freq * (1 + depth * modulator), depth up to 1); shape 'saw' |
    'pulse' | 'tri' | 'sine'; pw and sync as in unison. Aliases of a static tone stay 80 dB under its fundamental.
        y = voice.osc(dsp.sweep(110, 3520, 2.0), 'saw', sync=2.5)"""
    f = np.asarray(freq, np.float64)
    if f.ndim == 0:
        f = np.full(n_of(seconds or 1.0), float(f))
    return unison(f, shape, pw=pw, sync=sync, phase=phase)[:, 0]


def sine(freq, phase=0.0):
    """a sine at freq Hz per sample, starting at `phase` cycles (0 = a rising zero crossing: no thump at note-on)"""
    freq = np.asarray(freq, np.float64)
    return np.sin(2 * np.pi * ((np.cumsum(freq) - freq[0]) / SR + phase))


# ------------------------------------------------------------------------------------------------ control signals
def wobble(n, rate, r, sr=SR):
    """a smooth random control signal, about unit size, mean 0: a spline through random points `rate` per second
    (evaluated 32 times per point, straight lines between). For drift, filters that never sit still and anything a
    sine LFO would make mechanical"""
    rate = max(rate, 1e-3)
    step = max(1, int(sr / (32.0 * rate)))
    x = np.arange(0, n + step, step) * (rate / sr)
    i = x.astype(np.int64)
    u = x - i
    v = r.standard_normal(int(i[-1]) + 4)
    a, b, c, d = v[i], v[i + 1], v[i + 2], v[i + 3]
    y = b + 0.5 * u * (c - a + u * (2 * a - 5 * b + 4 * c - d + u * (3 * (b - c) + d - a)))
    return y[:n] if step == 1 else np.interp(np.arange(n), np.arange(len(y)) * step, y)


def lfo(n, rate, shape='sine', phase=0.0, delay=0.0, fade=0.0, r=None):
    """a low-frequency control signal in about -1..1: 'sine', 'tri' (the linear sweep of an analogue LFO: PWM, the
    clock of a BBD chorus) or 'random' (wobble; needs r). phase in cycles. delay s of nothing, then a fade-in over
    fade s: vibrato that arrives late in a held note, as a player adds it"""
    t = np.arange(n) / SR
    if shape == 'random':
        y = 0.6 * wobble(n, rate, r)
    elif shape == 'tri':
        y = 1.0 - 2.0 * np.abs(2.0 * ((t * rate + phase + 0.25) % 1.0) - 1.0)
    else:
        y = np.sin(2 * np.pi * (t * rate + phase))
    if delay > 0 or fade > 0:
        u = np.clip((t - delay) / max(fade, 1e-4), 0.0, 1.0)
        y = y * u * u * (3.0 - 2.0 * u)
    return y


def _bend(u, c):
    """0..1 -> 0..1 along the curve of a charging capacitor: c > 0 moves fast first, c < 0 slow first, 0 is a line"""
    return u if abs(c) < 1e-3 else (1.0 - np.exp(-c * u)) / (1.0 - np.exp(-c))


def adsr(dur, a=0.002, d=0.3, s=0.0, rel=0.2, hold=0.0, curve=(2.0, 5.0, 5.0), slew=0.0008):
    """an envelope for a note held dur s: attack a s to 1, hold s at 1, decay d s to the sustain level s, then from
    wherever it is at the release, rel s to exactly 0. The segments are exponential like the RC stages of an analogue
    envelope: curve = (attack, decay, release) bends them (0 = linear, 5 = the segment covers 43 dB of its
    exponential, so d is about the audible length of a pluck; a negative attack curve starts slow: a swell). The
    result is slewed over `slew` s (two running means, as a VCA's control input is band-limited), which is also the
    shortest possible attack: it starts at exactly 0 with no step and no corner clicks at the release.
    Length n_of(dur) + rel + slew.
        e = voice.adsr(0.3, 0.003, 0.8, 0.0, 0.3)             # a pluck: 3 ms up, 0.8 s down, released at 0.3 s"""
    ng = n_of(dur)
    na, nd, nr = (max(1, int(round(v * SR))) for v in (a, d, rel))
    up = _bend(np.arange(1, na + 1) / na, curve[0])
    down = s + (1.0 - s) * (1.0 - _bend(np.arange(1, nd + 1) / nd, curve[1]))
    held = np.concatenate([up, np.ones(int(round(hold * SR))), down, np.full(max(0, ng - na - nd), float(s))])[:ng]
    tail = held[-1] * (1.0 - _bend(np.arange(1, nr + 1) / nr, curve[2]))
    k = max(2, int(slew * SR / 2))
    e = np.concatenate([np.zeros(k), held, tail, np.zeros(k)])
    e = ndimage.uniform_filter1d(ndimage.uniform_filter1d(e, k, mode='constant'), k, mode='constant')
    return np.where(e < 1e-12, 0.0, e)                     # the running means leave rounding dust around zero


def fit(x, n):
    """x cut or extended (with its last value) to n samples"""
    x = np.asarray(x, np.float64)
    return x[:n] if len(x) >= n else np.concatenate([x, np.full((n - len(x),) + x.shape[1:], x[-1])])


# ------------------------------------------------------------------------------------------------ filter
@njit(cache=True)
def _soft(x):
    """tanh as a rational curve (its Pade form), flat beyond +-3"""
    if x > 3.0:
        return 1.0
    if x < -3.0:
        return -1.0
    return x * (27.0 + x * x) / (27.0 + 9.0 * x * x)


@njit(cache=True)
def _ladder(x, G, k, tap):
    y = np.empty_like(x)
    s = np.zeros((x.shape[1], 4))
    for i in range(x.shape[0]):
        g = G[i]
        kk = k[i]
        g4 = g * g * g * g
        for c in range(x.shape[1]):
            u = x[i, c] * (1.0 + 0.5 * kk)
            mem = (g * g * g * s[c, 0] + g * g * s[c, 1] + g * s[c, 2] + s[c, 3]) * (1.0 - g)
            v = u - kk * 2.0 * _soft(0.5 * (g4 * u + mem) / (1.0 + kk * g4))
            for j in range(4):
                d = (v - s[c, j]) * g
                v = d + s[c, j]
                s[c, j] = v + d
                if j + 1 == tap:
                    y[i, c] = v
    return y


def ladder(x, cutoff, res=0.0, poles=4):
    """the 4-pole transistor-ladder low-pass as a zero-delay-feedback filter (Zavalishin's TPT form): four one-pole
    stages in a feedback loop that is solved within the sample, so the resonance sits exactly at the cutoff at any
    frequency and the cutoff can move every sample without instability. cutoff Hz, res 0..1.1 (scalars or per
    sample): 1 is the edge of self-oscillation, where it sings a sine at the cutoff; the feedback runs through a soft
    limiter, which keeps that sine at a moderate level and lets a hot input press the resonance down, as in the
    circuit. The pass band keeps half of the level the feedback would take away (res thins a ladder's bass). poles 4
    = 24 dB / octave, 2 or 1 = the gentler taps of the same ladder. x mono or stereo; drive it with fx.saturate.
        y = voice.ladder(saws, 300 * 2 ** (3 * env), res=0.3)"""
    x2 = np.asarray(x, np.float64)
    mono = x2.ndim == 1
    x2 = np.ascontiguousarray(x2[:, None] if mono else x2)
    n = len(x2)
    g = np.tan(np.pi * np.clip(np.broadcast_to(np.asarray(cutoff, np.float64), (n,)), 16.0, 0.45 * SR) / SR)
    k = 4.0 * np.clip(np.broadcast_to(np.asarray(res, np.float64), (n,)), 0.0, 1.1)
    y = _ladder(x2, np.ascontiguousarray(g / (1.0 + g)), np.ascontiguousarray(k), int(poles))
    return y[:, 0] if mono else y


# ------------------------------------------------------------------------------------------------ FM
@njit(cache=True)
def _feedback(ph, e, fb):
    """an operator that modulates itself: sin(ph + fb * e * the mean of its last two samples)"""
    y = np.empty(len(ph))
    y1 = 0.0
    y2 = 0.0
    for i in range(len(ph)):
        v = np.sin(ph[i] + fb * e[i] * 0.5 * (y1 + y2))
        y2 = y1
        y1 = v
        y[i] = v
    return y


def _reach(index):
    """how many sidebands of a phase modulation of this index stay above -70 dB (Bessel J_n(index) is under 3e-4
    past it)"""
    return index + 2.4 * index ** (1 / 3) + 2.0 if index > 0 else 0.0


def _cap(limit, ok):
    """the largest value in 0..limit that ok() accepts (ok is true toward 0)"""
    if ok(limit):
        return limit
    lo, hi = 0.0, limit
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _fb_decay(fb):
    """nepers per harmonic by which the spectrum of a self-modulating operator falls (its n-th harmonic is
    2 J_n(n fb) / (n fb): exponential in n for fb < 1, a sawtooth at 1)"""
    a = np.arccosh(1.0 / min(max(fb, 1e-6), 0.9999))
    return a - np.tanh(a)


OP = {'ratio': 1.0, 'hz': None, 'cents': 0.0, 'index': 0.0, 'out': 0.0, 'pan': 0.0, 'mod': (), 'fb': 0.0,
      'env': (0.002, 0.5, 0.0, 0.2), 'hold': 0.0, 'curve': (2.0, 5.0, 5.0), 'vel': 0.0, 'key': 0.0, 'rate': 0.0,
      'wobble': None}


def fm(freq, ops, vel=0.8, dur=0.3, r=None):
    """phase modulation between operators (sines with their own envelopes), stereo (n, 2) with the release. freq: Hz,
    a scalar or per sample. ops: up to six dicts over the defaults in OP:
      ratio / cents   frequency = freq * ratio, detuned by cents; or hz = a fixed frequency
      mod             the operators that phase-modulate this one (higher-numbered ones only: an algorithm is a list
                      that reads from carriers down to modulators), each by its own `index` in radians at full level
      out, pan        a carrier: its amplitude in the mix and its place -1..1
      fb              self-modulation in radians (toward a sawtooth at 1)
      env, hold, curve   its adsr: an index that decays faster than the carrier gives the bright attack and pure
                      tail of struck metal, tines and plucked strings
      vel             0..1: how far the level follows velocity (level * vel^(2 vel)): harder notes are brighter
      key, rate       per octave above C4: level * 2^key (negative: less index up high, as real instruments have
                      fewer partials there) and envelope times * 2^-rate (high notes die sooner)
      wobble          (depth 0..1, Hz): a slow random movement of its level, for pads that shimmer (needs r)
    Where the spectrum reaches past 21.6 kHz it is computed at 96 kHz and decimated; where a modulation would still
    put sidebands past the alias limit (high notes), its index is reduced until it does not.
        ep = voice.fm(220.0, [dict(out=1, mod=(1,), env=(0.002, 2.5, 0, 0.2)),
                              dict(ratio=14, index=0.8, vel=0.8, env=(0.001, 0.1, 0, 0.05))], vel=0.9, dur=0.5)"""
    ops = [{**OP, **o} for o in ops]
    for o in ops:
        if len(o) != len(OP):
            raise ValueError(f'unknown operator setting {sorted(set(o) - set(OP))}: one of {", ".join(OP)}')
    f = np.atleast_1d(np.asarray(freq, np.float64))
    octs = np.log2(np.median(f) / C4)
    envs = []
    for o in ops:
        ts = 2.0 ** (-o['rate'] * octs)
        a, d, s, rel = o['env']
        envs.append(adsr(dur, a * ts, d * ts, s, rel * ts, o['hold'], o['curve']) * vel ** (2.0 * o['vel'])
                    * 2.0 ** (o['key'] * octs))
    n = max(len(e) for e in envs)
    f = fit(f, n)
    fmax = float(f.max())
    k = [o['ratio'] * 2.0 ** (o['cents'] / 1200) for o in ops]
    fi = [o['hz'] or fmax * q for o, q in zip(ops, k)]
    bw, sc, fb = list(fi), [1.0] * len(ops), [0.0] * len(ops)
    for i in reversed(range(len(ops))):                    # how far each operator's spectrum reaches; cap what aliases
        o = ops[i]
        if any(j <= i or j >= len(ops) for j in o['mod']):
            raise ValueError(f'operator {i} can only be modulated by operators {i + 1}..{len(ops) - 1}')
        if o['mod']:
            idx = sum(ops[j]['index'] * envs[j].max() for j in o['mod'])
            wide = max(bw[j] for j in o['mod'])
            sc[i] = _cap(1.0, lambda q: fi[i] + _reach(q * idx) * wide <= _FM_LIMIT)
            bw[i] = fi[i] + _reach(sc[i] * idx) * wide
        if o['fb'] > 0:
            fb[i] = _cap(float(o['fb']), lambda q: q <= 0 or (_FM_LIMIT / fi[i]) * 8.69 * _fb_decay(q) >= 70.0)
            bw[i] = max(bw[i], fi[i] * 8.06 / max(_fb_decay(fb[i]), 1e-3))
    up = 1 if max(bw) < 0.45 * SR else 2                   # a spectrum that fits under 21.6 kHz needs no oversampling
    ph = 2 * np.pi * np.cumsum(np.repeat(f, up)) / (up * SR)
    t = np.arange(up * n) / (up * SR)
    sig = [None] * len(ops)
    mix = np.zeros((up * n, 2))
    for i in reversed(range(len(ops))):
        o = ops[i]
        arg = 2 * np.pi * o['hz'] * t if o['hz'] else ph * k[i]
        e = np.concatenate([envs[i], np.zeros(n - len(envs[i]))])
        if up == 2:
            e = np.interp(np.arange(2 * n) * 0.5, np.arange(n), e)
        if o['wobble']:
            e = e * (1.0 + o['wobble'][0] * np.tanh(wobble(up * n, o['wobble'][1], r, up * SR)))
        if o['mod']:
            arg = arg + sc[i] * sum(ops[j]['index'] * sig[j] for j in o['mod'])
        sig[i] = e * (_feedback(arg, e / (e.max() + 1e-12), fb[i]) if fb[i] > 0 else np.sin(arg))
        if o['out']:
            th = (np.clip(o['pan'], -1, 1) + 1) * np.pi / 4
            mix += o['out'] * np.sqrt(2) * sig[i][:, None] * np.array([np.cos(th), np.sin(th)])
    y = signal.resample_poly(mix, 1, 2, axis=0, window=_HALF)[:n] if up == 2 else mix
    return dsp.butter(y, 'high', 16.0, order=1)            # equal ratios put a sideband at 0 Hz: block it, as a DX does


# ------------------------------------------------------------------------------------------------ the analogue voice
OSC = {'shape': 'saw', 'level': 1.0, 'semi': 0.0, 'cents': 0.0, 'voices': 1, 'detune': 0.0, 'spread': 0.0,
       'blend': 0.5, 'pw': 0.5, 'pwm': 0.0, 'pwm_rate': 0.4, 'sync': None, 'phase': None}
ANALOG = {
    'osc': ({},),                 # oscillators: dicts over OSC (unison settings per oscillator; pwm = width swing)
    'sub': 0.0, 'sub_shape': 'sine',      # level of a centred oscillator one octave down
    'noise': 0.0, 'noise_width': 0.0,     # white noise into the filter; 0 = mono .. 1 = independent left and right
    'drift': 2.0, 'smear': 0.0,           # cents of pitch wander per voice; s of onset smear across unison voices
    'bend': None,                 # (semitones, s): a pitch offset at note-on that dies away with this time constant
    'pitch': None,                # a function of time (s, array) giving semitone offsets: designed pitch contours
    'vib': None,                  # (cents, Hz, delay s, fade s): vibrato that arrives late
    'filter': 'ladder',           # 'ladder' (24 dB), 'ladder2' (its 12 dB tap), 'low' | 'band' | 'high' (svf), None
    'cutoff': 2000.0, 'res': 0.1,         # Hz at C4 with the envelope at 0; resonance 0..1
    'keytrack': 0.5,              # octaves the cutoff moves per octave of pitch
    'env': 0.0, 'fenv': (0.002, 0.3, 0.0, 0.2), 'fcurve': (2.0, 5.0, 5.0),    # octaves the filter envelope opens
    'vel_cut': 1.0,               # octaves the cutoff opens from velocity 0 to 1 (referred to 1)
    'cut_lfo': None,              # (octaves, Hz, shape): 'sine' | 'tri' | 'random' on the cutoff
    'hp': 0.0, 'drive': 0.0,              # a 12 dB high-pass after the filter, Hz; dB of oversampled tanh before it
    'amp': (0.003, 0.3, 0.7, 0.2), 'hold': 0.0, 'curve': (2.0, 5.0, 5.0), 'slew': 0.0008,
    'vel_amp': 0.5,               # level = vel^(2 vel_amp): 0.5 = linear in velocity
}


def noise2(n, r, width=0.0):
    """white noise as stereo (n, 2), unit RMS per channel: width 0 = the same left and right .. 1 = independent"""
    a = r.standard_normal(n)
    if width <= 0:
        return np.stack([a, a], axis=1)
    b = r.standard_normal(n)
    th = np.clip(width, 0, 1) * np.pi / 4
    return np.stack([a * np.cos(th) + b * np.sin(th), a * np.cos(th) - b * np.sin(th)], axis=1)


def analog(f0, vel, dur, r, **p):
    """one note of a subtractive voice, stereo (n, 2) with its release: oscillators (unison, sub, noise) -> drive ->
    filter with its own envelope, key tracking, velocity and LFO -> VCA. Settings are the keys of ANALOG (with OSC
    for each oscillator); the defaults are a plain saw. The signal path is the one of the classic polysynths, and
    each stage keeps what made them sound alive: free-running drifting oscillators, exponential envelopes, a filter
    that follows the keyboard and the velocity, a VCA that never steps.
        y = voice.analog(110.0, 0.9, 0.4, dsp.rng(1), osc=(dict(shape='saw'), dict(shape='pulse', pw=0.4, cents=7)),
                         sub=0.4, cutoff=300, env=3.0, fenv=(0.001, 0.15, 0.0, 0.1), res=0.25,
                         amp=(0.002, 0.5, 0.0, 0.1))"""
    bad = set(p) - set(ANALOG)
    if bad:
        raise ValueError(f'unknown voice setting {sorted(bad)}: one of {", ".join(ANALOG)}')
    q = {**ANALOG, **p}
    amp = adsr(dur, *q['amp'], hold=q['hold'], curve=q['curve'], slew=q['slew'])
    n = len(amp)
    t = np.arange(n) / SR
    semi = np.zeros(n)
    if q['bend']:
        semi += q['bend'][0] * np.exp(-t / q['bend'][1])
    if q['pitch']:
        semi += q['pitch'](t)
    if q['vib']:
        cents, rate, delay, fade = q['vib']
        semi += cents / 100 * lfo(n, rate, 'sine', r.uniform(), delay, fade)
    freq = f0 * 2.0 ** (semi / 12)
    mix = np.zeros((n, 2))
    for o in q['osc']:
        bad = set(o) - set(OSC)
        if bad:
            raise ValueError(f'unknown oscillator setting {sorted(bad)}: one of {", ".join(OSC)}')
        o = {**OSC, **o}
        pw = o['pw'] + o['pwm'] * lfo(n, o['pwm_rate'], 'tri', r.uniform()) if o['pwm'] else o['pw']
        mix += o['level'] * unison(freq * 2.0 ** ((o['semi'] + o['cents'] / 100) / 12), o['shape'], r, o['voices'],
                                   o['detune'], o['spread'], o['blend'], q['drift'], q['smear'], pw, o['sync'],
                                   o['phase'])
    if q['sub']:
        s = sine(freq / 2) if q['sub_shape'] == 'sine' else osc(freq / 2, q['sub_shape'])
        mix += q['sub'] * s[:, None]
    if q['noise']:
        mix += q['noise'] * noise2(n, r, q['noise_width'])
    if q['drive'] > 0:                                     # a clipped narrow pulse gains DC: block it
        mix = dsp.butter(fx.saturate(mix, q['drive'], 'tanh'), 'high', 12.0, order=1)
    if q['filter']:
        octs = q['keytrack'] * np.log2(f0 / C4) + q['vel_cut'] * (vel - 1.0)
        if q['env']:
            octs = octs + q['env'] * fit(adsr(dur, *q['fenv'], curve=q['fcurve'], slew=q['slew']), n)
        if q['cut_lfo']:
            depth, rate, shape = q['cut_lfo']
            octs = octs + depth * lfo(n, rate, shape, r.uniform(), r=r)
        cut = np.clip(q['cutoff'] * 2.0 ** octs, 20.0, 0.45 * SR) * np.ones(n)
        if q['filter'] in ('ladder', 'ladder2'):
            mix = ladder(mix, cut, q['res'], 4 if q['filter'] == 'ladder' else 2)
        else:
            qq = 0.707 * 17.0 ** q['res']
            mix = np.stack([dsp.svf(mix[:, c], cut, qq, q['filter']) for c in (0, 1)], axis=1)
    if q['hp']:
        mix = dsp.butter(mix, 'high', q['hp'], order=2)
    return mix * (amp * vel ** (2.0 * q['vel_amp']))[:, None]


# ------------------------------------------------------------------------------------------------ effects
@njit(cache=True)
def _tap(x, d, pad):
    """x read d[i] samples back (cubic Hermite between samples); x carries `pad` zeros in front and 3 behind"""
    n = len(d)
    y = np.empty(n)
    for i in range(n):
        p = i + pad - d[i]
        j = int(p)
        f = p - j
        a, b, c, e = x[j - 1], x[j], x[j + 1], x[j + 2]
        y[i] = b + 0.5 * f * (c - a + f * (2 * a - 5 * b + 4 * c - e + f * (3 * (b - c) + e - a)))
    return y


def _stereo(x):
    x = np.asarray(x, np.float64)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def chorus(x, rate=0.513, depth=1.85, delay=3.5, mix=0.5, voices=2, tone=9000.0, lowcut=120.0):
    """a bucket-brigade chorus: delay lines whose length a triangle LFO sweeps (delay +- depth ms at rate Hz), read
    between samples, mixed with the direct sound. The triangle is the character: the pitch of each line sits a
    constant few cents flat, then sharp, never gliding as a sine-swept chorus does. voices 2: the left line and the
    right line move in opposite directions (the defaults are the measured mode I of the Juno-60 chorus: 0.513 Hz,
    1.66-5.35 ms; mode II is 0.863 Hz). voices 3: a string ensemble, three lines 120 degrees apart, each swept by a
    slow and a fast LFO at once (about 0.6 Hz and 6 Hz), left / centre / right. The lines are band-limited to `tone`
    Hz as a BBD's filters do and carry nothing under `lowcut`, so the bass stays solid; the lines are added, never
    subtracted, so the mono sum loses nothing. mix 0..1; mono or stereo in, stereo out.
        wide = voice.chorus(pad, rate=0.863, mix=0.6)"""
    x2 = _stereo(x)
    n = len(x2)
    t = np.arange(n) / SR
    pad = int((delay + 1.3 * depth) * 1e-3 * SR) + 4
    w = dsp.butter(dsp.butter(x2, 'low', tone), 'high', lowcut) if lowcut else dsp.butter(x2, 'low', tone)
    src = [np.concatenate([np.zeros(pad), s, np.zeros(3)]) for s in (w[:, 0], w[:, 1], w.mean(axis=1))]
    if voices == 3:
        sw = [lfo(n, 0.6 * rate / 0.513, 'tri', k / 3) + 0.12 * np.sin(2 * np.pi * (6.0 * t + k / 3)) for k in range(3)]
        a, b, c = (_tap(src[i], (delay + depth * s) * 1e-3 * SR, pad) for i, s in zip((0, 2, 1), sw))
        wet = np.stack([a + 0.7 * b, c + 0.7 * b], axis=1) / 1.22
    else:
        s = lfo(n, rate, 'tri')
        wet = np.stack([_tap(src[0], (delay + depth * s) * 1e-3 * SR, pad),
                        _tap(src[1], (delay - depth * s) * 1e-3 * SR, pad)], axis=1)
    return (x2 + mix * wet) / np.sqrt(1.0 + mix * mix)


@njit(cache=True)
def _echo(x, n_out, dl, dr, g, lp, hp, cross, wow):
    wet = np.empty((n_out, 2))
    buf = np.zeros((n_out, 2))
    n = len(x)
    low = np.zeros(2)
    dc = np.zeros(2)
    d = np.array([dl, dr])
    for i in range(n_out):
        for c in range(2):
            p = i - d[c] * wow[i]
            v = 0.0
            if p >= 0.0:
                j = int(p)
                v = buf[j, c] + (p - j) * (buf[j + 1, c] - buf[j, c])
            low[c] += lp * (v - low[c])                    # each repeat loses highs (tape, a BBD's filters) ...
            dc[c] += hp * (low[c] - dc[c])                 # ... and lows, so repeats thin out instead of piling up
            wet[i, c] = 2.0 * _soft(0.5 * (low[c] - dc[c]))
        xl = x[i, 0] if i < n else 0.0
        xr = x[i, 1] if i < n else 0.0
        if cross:
            buf[i, 0] = 0.5 * (xl + xr) + g * wet[i, 1]
            buf[i, 1] = g * wet[i, 0]
        else:
            buf[i, 0] = xl + g * wet[i, 0]
            buf[i, 1] = xr + g * wet[i, 1]
    return wet


def echo(x, time=0.3, feedback=0.4, mix=0.25, damp=4500.0, lowcut=200.0, pingpong=True, skew=1.0, wow=0.0):
    """a tape-style stereo echo, returned with its tail: the repeats run through a low-pass (`damp` Hz) and a
    high-pass (`lowcut` Hz) inside the feedback loop and a soft limiter, so each one is darker and thinner than the
    last and they sink behind the sound instead of cluttering it. pingpong: the sound (its mono sum) goes left first
    and bounces right, left, ... every `time` s, each bounce `feedback` times the one before; otherwise each channel
    repeats on its own side every `time` s (the right every time * skew s). wow 0..1 moves the delay time slowly
    (0.6 Hz plus a little 5 Hz flutter, up to 0.4 %), which detunes the repeats like a tape loop.
        y = voice.echo(pluck, time=0.25, feedback=0.45, mix=0.2)"""
    x2 = np.ascontiguousarray(_stereo(x))
    g = float(np.clip(feedback, 0.0, 0.9))
    hops = int(np.ceil(np.log(1e-3) / np.log(max(g, 1e-3)))) + 1
    n_out = len(x2) + n_of(min(time * max(skew, 1.0) * hops, 6.0))
    t = np.arange(n_out) / SR
    w = 1.0 + 0.004 * wow * (np.sin(2 * np.pi * 0.6 * t) + 0.25 * np.sin(2 * np.pi * 5.1 * t))
    y = mix * _echo(x2, n_out, time * SR, time * skew * SR, g, 1.0 - np.exp(-2 * np.pi * damp / SR),
                    1.0 - np.exp(-2 * np.pi * lowcut / SR), bool(pingpong), w)
    y[:len(x2)] += x2
    return y


@njit(cache=True)
def _allpass_chain(x, a, fb, stages):
    y = np.empty(len(x))
    z = np.zeros(stages)
    last = 0.0
    for i in range(len(x)):
        v = x[i] + fb * last
        for s in range(stages):
            o = z[s] - a[i] * v
            z[s] = v + a[i] * o
            v = o
        last = v
        y[i] = v
    return y


def phaser(x, rate=0.3, lo=300.0, hi=2400.0, feedback=0.3, mix=0.5, stages=4, spread=0.25):
    """a phaser: `stages` first-order all-passes whose corner a sine LFO sweeps between lo and hi Hz (in octaves),
    mixed with the direct sound, so stages / 2 notches travel through the spectrum; feedback sharpens them; the right
    channel's LFO runs `spread` cycles ahead. Mono or stereo in, stereo out.
        y = voice.phaser(strings, rate=0.2, mix=0.4)"""
    x2 = _stereo(x)
    out = np.empty_like(x2)
    for c in (0, 1):
        f = lo * (hi / lo) ** (0.5 + 0.5 * lfo(len(x2), rate, 'sine', c * spread))
        g = np.tan(np.pi * f / SR)
        out[:, c] = _allpass_chain(np.ascontiguousarray(x2[:, c]), (1.0 - g) / (1.0 + g), float(feedback), int(stages))
    return (x2 + mix * out) / (1.0 + mix)


def mono_below(x, f=120.0):
    """a stereo signal with everything under f Hz summed to the centre (a bass that plays on one speaker, on a club
    system and through a vinyl cutter is the same bass); the bands sum back flat"""
    lo, hi = fx.split(_stereo(x), [f])
    return hi + lo.mean(axis=1, keepdims=True)


def tick(r, seconds, lo, hi, tau):
    """a contact transient, mono, peak 1: noise between lo and hi Hz dying with time constant tau s (the click of a
    kick, the thock of a mallet), starting from zero"""
    n = n_of(seconds)
    y = dsp.butter(r.standard_normal(n), 'band', (lo, hi)) * np.exp(-np.arange(n) / SR / tau)
    return dsp.fade(y / (np.abs(y).max() + 1e-12), 0.0003, seconds * 0.3)
