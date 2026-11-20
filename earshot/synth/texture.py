"""Nature and ambience textures computed from physics and statistics: rain, wind, fire, a stream, a forest, a
night, hums (48 kHz stereo beds, no recorded audio). The sea, the underwater rush, room tone and the city are
learned from real recordings instead (earshot.synth.quilt, earshot.sounds.nature), as are the campfire, the
crickets, the brook, wind in trees and rain on a street of the catalogue; the scenes here still make those for
code that asks for them (texture.render('night', ...): the invented crickets, kept as the fallback).

Procedural ambience usually sounds fake for two reasons: the spectrum is static (noise through one filter and an LFO)
and the events have no statistics (rain as hiss, fire without crackle clusters). Every scene here is built from two
engines instead:

  events   thousands of small synthesized events per second (drops, crackles, bubbles, leaf ticks, insect pulses):
           Poisson or self-exciting arrival times, power-law / log-normal / Marshall-Palmer sizes that set level, pitch
           and duration together, a stereo position and a distance per event (arrivals, sizes, rings, bursts)
  noise    band-wise noise whose envelopes have a chosen marginal, modulation spectrum and cross-band correlation
           (McDermott & Simoncelli 2011: these statistics, not the power spectrum, make a texture recognisable), with
           the spectra taken from the physics: every wind band follows one gusting wind speed, water bands move on
           their own, howls sit at the Aeolian frequency 0.2 v / d (drift, envelopes, band_noise)

    from earshot.forge import dsp
    from earshot.synth import texture
    bed = texture.render('rain', 12.0, dsp.rng(3), intensity=0.4, surface='window')   # (n, 2) float64, quiet
    fire = texture.loop('fire', 10.0, dsp.rng(1), size=0.2)                           # repeats without a seam
    texture.seam_db(fire)                                                             # {'level', 'spectral', 'click'}
    texture.SCENES['stream']                                                          # says, loop, min_loop, controls

A bed is rendered around a circle: controls are periodic over its length, events wrap past the end onto the start and
filters and rooms are applied circularly, so loop() returns a buffer with no seam to hide and no event cut, and
render() is the same buffer with a fade at both ends. Beds are quiet by design (peak at a quarter of -1 dBFS).
"""
import numpy as np
from scipy import signal

from earshot.forge import dsp, fx
from earshot.forge.dsp import SR, njit

CR = 1000                                   # control rate (Hz) of drifts, event rates and band envelopes
BED_PEAK = 0.25 * 10 ** (-1 / 20)           # a bed's peak: a quarter of -1 dBFS, about -13 dBFS
EVENT_PEAK = 10 ** (-3 / 20)                # the peak of an event (thunder)


# ------------------------------------------------------------------------------------------------ controls
def drift(dur, r, f_lo=0.05, f_hi=2.0, slope=1.0, knee=0.0):
    """a random control signal at CR Hz, unit std, mean 0, periodic over dur s (a bed built from it loops): Gaussian,
    its power falling 3 * slope dB per octave between f_lo and f_hi Hz (slope 1 = 1/f: flow, rain coming in waves;
    5/3 = the inertial range of turbulence; 0 = white) and flat below `knee` Hz (the von Karman shape of wind speed).
        gust = texture.drift(20.0, dsp.rng(1), 0.02, 5.0, 5 / 3, knee=0.08)"""
    m = max(16, int(round(dur * CR)))
    f = np.fft.rfftfreq(m, 1 / CR)
    k = (f >= f_lo) & (f <= f_hi)
    k[0] = False
    if not k.any():
        k[1] = True
    F = np.zeros(len(f), complex)
    F[k] = (knee ** 2 + f[k] ** 2) ** (-slope / 4) * (r.standard_normal(k.sum()) + 1j * r.standard_normal(k.sum()))
    c = np.fft.irfft(F, m)
    return c / (c.std() + 1e-12)


def envelopes(dur, r, bands, sigma=0.3, rho=0.5, f_lo=0.2, f_hi=20.0, slope=1.0, common=None):
    """band envelopes with chosen statistics, shape (bands, dur * CR), mean power 1. Each is exp(sigma * g): log-normal,
    so sigma sets the marginal (0.1-0.3 smooth as wind or a far wash, 0.8 and more sparse as the hiss of a fire); g is
    Gaussian with a 1/f^slope modulation spectrum between f_lo and f_hi Hz; a share rho of its power is common to all
    bands (1 = a gust: every band moves together, 0 = water: every band on its own). `common` supplies that shared
    part (a drift) when other layers must follow it too."""
    c = drift(dur, r, f_lo, f_hi, slope) if common is None else common
    g = np.stack([np.sqrt(rho) * c + np.sqrt(1 - rho) * drift(dur, r, f_lo, f_hi, slope) for _ in range(bands)])
    return np.exp(sigma * g - sigma ** 2)


def _at(c, n):
    """a control (periodic over the buffer) at audio rate: n samples, linear interpolation"""
    m = len(c)
    if n % m == 0:
        k = n // m
        return (c[:, None] + (np.roll(c, -1) - c)[:, None] * (np.arange(k) / k)[None, :]).ravel()
    return np.interp(np.arange(n) * (m / n), np.arange(m + 1), np.append(c, c[0]))


def _val(c, t, dur):
    """a control's value at times t (s)"""
    return c[(np.asarray(t) / dur * len(c)).astype(np.int64) % len(c)]


def _delayed(c, seconds, dur):
    """a control read `seconds` later (the same gust reaching another object), still periodic"""
    return np.roll(c, int(round(seconds / dur * len(c))))


# ------------------------------------------------------------------------------------------------ event statistics
def arrivals(r, rate, dur, cluster=0.0, spread=0.015):
    """event times in [0, dur) s: a Poisson process with `rate` events per second (a number, or a control array over
    dur: a rate that follows a gust or a flame is a doubly stochastic process), optionally self-exciting: every event
    triggers on average `cluster` (0..0.9) more within about `spread` s, which trigger more (a Hawkes process: the
    bursts of 2-5 crackles of a fire, sparks, drops shaken off one leaf). The mean rate stays `rate`. Times wrap
    around dur. Returns (times, generation), sorted by time; generation 0 = not triggered by another event."""
    if np.ndim(rate) == 0:
        t = r.uniform(0, dur, r.poisson(max(float(rate), 0.0) * (1 - cluster) * dur))
    else:
        cum = np.concatenate([[0.0], np.cumsum(np.maximum(rate, 0.0))]) * (dur / len(rate)) * (1 - cluster)
        t = np.interp(r.uniform(0, cum[-1], r.poisson(cum[-1])), cum, np.linspace(0, dur, len(cum)))
    ts, gs, gen = [t], [np.zeros(len(t), np.int64)], 0
    while cluster > 0 and len(t) and gen < 16:
        gen += 1
        t = np.repeat(t, r.poisson(cluster, len(t)))
        t = t + r.exponential(spread, len(t))
        ts.append(t)
        gs.append(np.full(len(t), gen))
    t, g = np.concatenate(ts) % dur, np.concatenate(gs)
    o = np.argsort(t, kind='stable')
    return t[o], g[o]


def sizes(r, count, lo, hi, law='power', shape=2.0):
    """event sizes in [lo, hi]. law 'power': density ~ size^-shape (bubbles, crackles, shards: many small, few large);
    'exp': density ~ exp(-shape * size) (Marshall-Palmer raindrop diameters, shape = 4.1 R^-0.21 per mm at R mm/h);
    'lognormal': median sqrt(lo * hi), sigma `shape` in log units, clipped to the range."""
    if law == 'lognormal':
        return np.clip(np.sqrt(lo * hi) * np.exp(shape * r.standard_normal(count)), lo, hi)
    u = r.uniform(size=count)
    if law == 'exp':
        return lo - np.log(1 - u * (1 - np.exp(-shape * (hi - lo)))) / shape
    if law != 'power':
        raise ValueError(f'unknown law {law!r}; valid: power, exp, lognormal')
    if abs(shape - 1) < 1e-9:
        return lo * (hi / lo) ** u
    return (lo ** (1 - shape) + u * (hi ** (1 - shape) - lo ** (1 - shape))) ** (1 / (1 - shape))


def _pans(r, count, width=1.0, centre=0.0):
    """stereo positions pushed towards the sides (pan ~ sign * sqrt(uniform)): with constant-power panning the two
    channels of a dense event layer then correlate about 0.45 (uniform positions give 0.64, a narrow image)"""
    u = r.uniform(-1, 1, count)
    return np.clip(centre + width * np.sign(u) * np.sqrt(np.abs(u)), -1, 1)


# ------------------------------------------------------------------------------------------------ event kernels
@njit(cache=True)
def _rings(out, i0, gl, gr, wr, wi, dr, di, ga, m, wrap):
    n = out.shape[0]
    for e in range(i0.shape[0]):
        zr, zi = 1.0, 0.0
        ar, ai = wr[e], wi[e]
        g = 1.0
        me = m[e]
        tail = me - me // 4
        i = i0[e]
        for k in range(me):
            y = zi * (1.0 - g)
            if k >= tail:
                y *= (me - k) / (me - tail)
            if i >= n:
                if not wrap:
                    break
                i -= n
            out[i, 0] += gl[e] * y
            out[i, 1] += gr[e] * y
            i += 1
            t = zr * ar - zi * ai
            zi = zr * ai + zi * ar
            zr = t
            t = ar * dr[e] - ai * di[e]
            ai = ar * di[e] + ai * dr[e]
            ar = t
            g *= ga[e]


@njit(cache=True)
def _bursts(out, i0, gl, gr, b1, b2, mx, m, off, sharp, noise, wrap):
    n = out.shape[0]
    mask = noise.shape[0] - 1
    for e in range(i0.shape[0]):
        y1, y2 = 0.0, 0.0
        me, mxe = m[e], mx[e]
        tail = me - me // 4
        i = i0[e]
        for k in range(me):
            x = 0.0
            if k < mxe:
                u = (k + 0.5) / mxe
                x = noise[(off[e] + k) & mask] * (1.0 - u) * (sharp * (1.0 - u) + (1.0 - sharp) * 4.0 * u)
            y = x + b1[e] * y1 - b2[e] * y2
            v = y - y2
            y2 = y1
            y1 = y
            if k >= tail:
                v *= (me - k) / (me - tail)
            if i >= n:
                if not wrap:
                    break
                i -= n
            out[i, 0] += gl[e] * v
            out[i, 1] += gr[e] * v
            i += 1


def _starts(out, t, wrap):
    i0 = np.round(np.asarray(t, np.float64) * SR).astype(np.int64)
    return i0 % len(out) if wrap else i0


def rings(out, t, gain, pan, f, decay, attack=0.0002, glide=0.0, wrap=True):
    """add damped sinusoids to the stereo buffer `out`, one per event (arrays or scalars, broadcast): a bubble, a
    ping of glass or sheet metal, the thud of a membrane, an insect's wing stroke, a grain. t = onset s, gain = peak
    amplitude, pan -1..1 (constant power), f = Hz at the onset, decay = time constant s, attack = rise time constant s
    (0.2 ms: struck; a few ms: bowed, scraped), glide = relative frequency change per second (a rising bubble:
    +0.1 * its damping rate; a drum skin relaxing: negative). A ring runs 6.2 time constants and its last quarter is
    ramped to zero. wrap: events past the end continue at the start (a loop); False cuts them at the end.
        texture.rings(out, t, 0.3, pans, 3.26 / radius_m, tau, glide=0.1 / tau)       # Minnaert bubbles"""
    t, gain, pan, f, decay, attack, glide = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, np.float64)) for a in (t, gain, pan, f, decay, attack, glide)])
    if not t.size:
        return out
    f = np.minimum(f, 0.42 * SR)
    m = np.ceil((6.2 * decay + 3.0 * attack) * SR).astype(np.int64) + 8
    glide = np.clip(glide, -0.9 * SR / m, (0.45 * SR / f - 1) * SR / m)          # stays in (0.1 f, 0.45 SR)
    w = np.exp(-1 / (decay * SR) + 2j * np.pi * f / SR)
    d = np.exp(2j * np.pi * f * glide / SR ** 2)
    k = np.maximum(attack, 1e-7) / decay                                          # peak of (1 - e^-t/a) e^-t/d
    g = gain / ((k / (1 + k)) ** k / (1 + k))
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    ga = np.where(attack > 1e-6, np.exp(-1 / (np.maximum(attack, 1e-6) * SR)), 0.0)
    _rings(out, _starts(out, t, wrap), g * np.cos(th), g * np.sin(th), w.real.copy(), w.imag.copy(), d.real.copy(),
           d.imag.copy(), ga, m, wrap)
    return out


def bursts(out, t, gain, pan, f, q, length, r, sharp=1.0, wrap=True):
    """add band-passed noise bursts to the stereo buffer `out`, one per event (arrays or scalars, broadcast): the tick
    of a drop on a hard surface, the patter on a leaf, a crackle of burning wood, a leaf brushing a leaf, a pebble.
    t = onset s, pan -1..1, f = centre Hz, q = f / bandwidth (1: a broadband click; 20 and more: the burst rings),
    length = duration of the noise s, sharp 0..1 = its shape (1: full at the onset, then falling: an impact; 0: a
    symmetric swell: a rustle). gain is in energy units: gain 1 is the energy of 1 ms of unit-RMS noise whatever the
    length and band, so a size law can set the level of an event apart from its duration and pitch.
        texture.bursts(out, t, drop ** 1.5, pans, 5000 * drop ** -0.45, 1.0, 0.0008 * drop, r)"""
    t, gain, pan, f, q, length = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, np.float64)) for a in (t, gain, pan, f, q, length)])
    if not t.size:
        return out
    f = np.clip(f, 30.0, 0.42 * SR)
    rho = np.exp(-np.pi * f / q / SR)
    mx = np.maximum(2, np.round(length * SR)).astype(np.int64)
    m = mx + np.ceil(6.2 * SR * q / (np.pi * f)).astype(np.int64) + 4
    ew = mx * (0.2 * sharp ** 2 + 0.5333 * (1 - sharp) ** 2 + 0.4 * sharp * (1 - sharp))      # energy of the window
    g = gain * np.sqrt(0.001 * SR * (1 - rho) / ew)                 # the resonator's own energy gain is 1 / (1 - rho)
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    noise = r.standard_normal(1 << 15)
    _bursts(out, _starts(out, t, wrap), g * np.cos(th), g * np.sin(th), 2 * rho * np.cos(2 * np.pi * f / SR), rho ** 2,
            mx, m, r.integers(0, 1 << 15, len(t)), float(sharp), noise, wrap)
    return out


def stick_slip(r, rate, jitter=0.05, hold=0.0, hold_s=(0.03, 0.1), rough=0.25, edge=0.00015):
    """the friction force of stick-slip (a hinge, a peeling tape, a bowed string, a wet finger on glass): while the
    surfaces stick the force builds up linearly, at the slip it drops at once, so the force is a sawtooth whose
    edges are the bright part. One slip per period of `rate` (Hz per sample, as long as the output; under 5 Hz
    nothing moves), each period moved by `jitter` of itself, now and then (`hold`: the chance per slip) the
    surfaces holding for hold_s s, after which the next slip is bigger (more force was stored); the drops vary
    log-normally by `rough`. The ramp of a slip starts at most one period before it, so a held contact is silent.
    edge = the fall time of a slip in s. Returns (force (n,) mono, slip times s).
        f, ts = texture.stick_slip(dsp.rng(1), np.full(48000, 120.0), jitter=0.06, hold=0.02)"""
    rate = np.asarray(rate, np.float64)
    n = len(rate)
    ts, amps, ps = [], [], []
    i, stored = 0, 0.0
    while i < n:
        f = rate[i]
        if f < 5.0:
            i += SR // 500
            continue
        if hold and r.uniform() < hold:
            gap = r.uniform(*hold_s)
            i += int(gap * SR)
            stored = min(gap * f, 3.0)
            continue
        p = SR / f * max(0.3, 1 + jitter * r.standard_normal())
        i += max(3, int(p))
        if i >= n:
            break
        ts.append(i)
        ps.append(p)
        amps.append(np.exp(rough * r.standard_normal()) * (1 + 0.3 * stored))
        stored = 0.0
    out = np.zeros(n)
    if not ts:
        return out, np.zeros(0)
    s, a, p = np.array(ts, np.float64), np.array(amps), np.array(ps)
    e = max(1.0, edge * SR)
    start = np.maximum(np.concatenate([[0.0], s[:-1]]), s - p) + 1.0
    peak = np.maximum(s - e, start + 1.0)
    knots = np.stack([start, peak, s], 1).ravel()
    vals = np.stack([np.zeros_like(a), a, np.zeros_like(a)], 1).ravel()
    knots = np.maximum.accumulate(knots)
    j = np.arange(n)
    out = np.interp(j, knots, vals, left=0.0, right=0.0)
    return out, s / SR


# ------------------------------------------------------------------------------------------------ noise textures
def band_noise(n, r, centres, env, side=0.7):
    """stereo Gaussian noise as a sum of bands, each multiplied by its own envelope: the engine of every wash, roar and
    hiss. centres = band centres Hz, log-spaced (raised-cosine bands over log frequency, powers summing flat); env =
    amplitude envelopes (bands, m) at any control rate, periodic over the n samples, or (bands, m, 2) for one per
    channel (a gust or a breaking wave crossing the stereo field); side = share of the uncorrelated side signal
    (scalar or per band): 0 mono, 0.7 gives an L/R correlation near 0.35, 1 none. Every band is a mid and a side noise
    (L = M + side * S, R = M - side * S), so the mono sum loses nothing to comb filtering. Circular: the buffer loops.
        wash = texture.band_noise(n, r, 125 * 2.0 ** np.arange(7), texture.envelopes(n / SR, r, 7, 0.15, 0.3))"""
    centres, env = np.asarray(centres, np.float64), np.asarray(env, np.float64)
    side = np.broadcast_to(np.asarray(side, np.float64), centres.shape)
    f = np.fft.rfftfreq(n, 1 / SR)
    step = np.log(centres[1] / centres[0])
    out = np.zeros((n, 2))
    F = np.zeros(len(f), complex)
    for b, fc in enumerate(centres):
        i0, i1 = np.searchsorted(f, [fc * np.exp(-step), fc * np.exp(step)])
        i0, i1 = max(i0, 1), min(i1, len(f) - 1)
        if i1 <= i0 or not env[b].any():
            continue
        h = np.cos(0.5 * np.pi * np.log(f[i0:i1] / fc) / step)
        h *= n / (2 * np.sqrt(np.sum(h ** 2)))                                    # unit expected RMS
        car = []
        for _ in range(2):
            F[i0:i1] = h * (r.standard_normal(i1 - i0) + 1j * r.standard_normal(i1 - i0))
            car.append(np.fft.irfft(F, n))
        F[i0:i1] = 0
        k = 1 / np.sqrt(1 + side[b] ** 2)
        mid, sd = car[0] * k, car[1] * (k * side[b])
        if env.ndim == 3:
            out[:, 0] += (mid + sd) * _at(env[b, :, 0], n)
            out[:, 1] += (mid - sd) * _at(env[b, :, 1], n)
        else:
            e = _at(env[b], n)
            out[:, 0] += (mid + sd) * e
            out[:, 1] += (mid - sd) * e
    return out


def _bands(lo, hi, count):
    return lo * (hi / lo) ** (np.arange(count) / (count - 1))


def _hump(f, centre, lo_db, hi_db):
    """amplitude of a broad spectral hump: power density rising lo_db per octave below `centre`, falling hi_db above
    (the shape of dsp.air); f and centre broadcast, so a centre that moves in time gives a moving spectrum"""
    u = np.log2(np.maximum(f, 1.0) / centre)
    return (2 ** (-lo_db / 3.0103 * u) + 2 ** (hi_db / 3.0103 * u)) ** -0.5


def _levels(centres, centre, lo_db, hi_db):
    """the amplitudes of constant-relative-width bands under a _hump spectrum (band power = density * bandwidth)"""
    return _hump(centres, centre, lo_db, hi_db) * np.sqrt(centres / centre)


# ------------------------------------------------------------------------------------------------ circular processing
def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def _top(x, share=1e-4):
    """the level of the loudest moments of an event layer: the amplitude that only `share` of its samples exceed
    (a steadier reference than the single highest peak when a bed is balanced against its events)"""
    a = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    k = max(1, int(len(a) * share))
    return float(np.partition(a, len(a) - k)[len(a) - k]) + 1e-12


def _filt(x, gain):
    """zero-phase filter around the circle: the spectrum of x (n,) or (n, 2) times gain(f Hz)"""
    g = gain(np.fft.rfftfreq(len(x), 1 / SR))
    return np.fft.irfft(np.fft.rfft(x, axis=0) * (g if x.ndim == 1 else g[:, None]), len(x), axis=0)


def _lp(fc, order=2):
    return lambda f: 1 / np.sqrt(1 + (f / fc) ** (2 * order))


def _hp(fc, order=2):
    return lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-6)) ** (2 * order))


def _space(x, rt60, wet, seed, size=0.6, damp=0.5):
    """the stereo bus x in a diffuse space (a fixed fx.reverb_ir room per seed, each channel through its own side of
    it), convolved around the circle: the tail of the end wraps onto the start, so a loop stays a loop. wet = reverb
    level against the direct sound, as in fx.reverb. Distance is this, level and a low-pass: far layers are wetter."""
    ir = fx.reverb_ir(dsp.rng(seed), rt60=rt60, size=size, damp=damp, lowcut=150.0, seed=seed)[:, 0, :]
    n = len(x)
    k = -(-len(ir) // n)
    if k > 1:
        ir = np.pad(ir, ((0, k * n - len(ir)), (0, 0))).reshape(k, n, 2).sum(0)
    H = np.fft.rfft(ir, n, axis=0)
    return x + wet * np.fft.irfft(np.fft.rfft(x, axis=0) * H, n, axis=0)


def _svf(x, fc, q, mode):
    """dsp.svf around the circle: the filter runs a quarter second before the start, so its state at the loop point
    is the state it will have there on the next pass"""
    pre = min(len(x), SR // 4)
    fc = np.broadcast_to(np.asarray(fc, np.float64), x.shape)
    return dsp.svf(np.concatenate([x[-pre:], x]), np.concatenate([fc[-pre:], fc]), q, mode)[pre:]


def _put(out, x, t, gain=1.0, pan=0.0):
    """add the mono snippet x at t s, panned, wrapping past the end onto the start"""
    n = len(out)
    idx = (int(round(t * SR)) + np.arange(len(x))) % n
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    if len(x) > n:
        np.add.at(out[:, 0], idx, gain * np.cos(th) * x)
        np.add.at(out[:, 1], idx, gain * np.sin(th) * x)
    else:
        out[idx, 0] += gain * np.cos(th) * x
        out[idx, 1] += gain * np.sin(th) * x
    return out


def _bubbles(out, t, gain, pan, radius_mm, rise=0.1):
    """air bubbles in water, the sound of every drip, stream and breaker: each rings at the Minnaert frequency
    3.26 / radius Hz (radius in m), damped at 0.043 f + 0.0014 f^1.5 per second (radiation, viscous and thermal
    losses) and rising in pitch as it nears the surface, f (1 + rise * damping * t) (van den Doel 2005)"""
    f = 3260.0 / np.asarray(radius_mm, np.float64)
    d = 0.043 * f + 0.0014 * f ** 1.5
    return rings(out, t, gain, pan, f, 1 / d, 0.25 / f, rise * d)


# ------------------------------------------------------------------------------------------------ rain
SURFACES = {
    # what one drop does to each surface. tick: the impact noise (centre Hz for a 2 mm drop, q, ms for a 2 mm drop,
    # sharpness, gain); body: the low thud of the surface under larger drops (Hz from, to, ms of noise, gain, q:
    # a leaf or a pane is damped, a taut canvas rings a few cycles); modes: the ringing modes of a hard sheet (count,
    # Hz from, to, decay ms at 1 kHz, gain); bubble: share of drops that entrain an air bubble; wash: the spectrum of
    # the far rain (centre Hz, dB/oct below, above); near: metres to the nearest and the farthest drop heard one by
    # one; drips: gutter or canopy drip sources
    'leaves': dict(tick=(4300, 1.0, 2.2, 0.3, 1.0), body=(280, 800, 3.0, 0.8, 1.5), modes=None, bubble=0.0,
                   wash=(3800, 4.0, 5.0), near=(0.8, 4.0), drips=2, indoors=False),
    # a roof is a sheet heard from under it: every drop drums on it (a short low thud of the panel and a short
    # ping of its many modes, damped by the battens), and the roof hides most of the far rain (wash_db)
    'roof': dict(tick=(4200, 1.2, 0.5, 1.0, 0.4), body=(110, 330, 2.5, 1.4, 5.0), modes=(36, 500, 4800, 20.0, 1.0),
                 bubble=0.0, wash=(1900, 3.0, 5.0), near=(1.2, 5.0), drips=2, indoors=True, wash_db=-9.0),
    # behind a pane the drip of a gutter is its tick alone: its bubble, a pure 1-2 kHz ring every second or two,
    # stood out of the rain as the one tonal and regular sound in it (puddle=False)
    'window': dict(tick=(6500, 2.0, 0.35, 1.0, 1.0), body=(170, 520, 2.0, 0.6, 2.5), modes=(10, 1700, 6200, 16.0, 0.35),
                   bubble=0.0, wash=(1100, 3.0, 6.5), near=(0.8, 2.5), drips=1, indoors=True, puddle=False),
    'street': dict(tick=(5000, 0.9, 0.9, 0.8, 1.0), body=(450, 1100, 2.0, 0.4, 1.2), modes=None, bubble=0.3,
                   wash=(3000, 3.5, 5.0), near=(0.8, 5.0), drips=1, indoors=False),
    'water': dict(tick=(6000, 1.0, 0.5, 0.8, 0.45), body=None, modes=None, bubble=0.55,
                  wash=(4500, 4.0, 5.0), near=(0.8, 5.0), drips=0, indoors=False),
    'tent': dict(tick=(2800, 1.0, 1.0, 0.7, 0.55), body=(140, 380, 3.0, 1.1, 4.0), modes=None, bubble=0.0,
                 wash=(1600, 3.0, 4.5), near=(0.7, 2.5), drips=1, indoors=True),
}


def _fall_speed(d_mm):
    """terminal velocity of a raindrop in m/s (Atlas 1973)"""
    return 9.65 - 10.3 * np.exp(-0.6 * d_mm)


def _drop_sizes(r, count, mm_h):
    """raindrop diameters (mm) as they arrive on a surface: the Marshall-Palmer number density exp(-L D) with
    L = 4.1 R^-0.21 per mm at R mm/h, times the fall speed (the flux). Drops under 0.7 mm belong to the wash."""
    d = np.linspace(0.7, 6.0, 256)
    cum = np.cumsum(np.exp(-4.1 * mm_h ** -0.21 * d) * _fall_speed(d))
    return np.interp(r.uniform(0, cum[-1], count), cum, d)


def _drop_level(d_mm):
    """amplitude of a drop's impact against a 2 mm drop: the square root of its kinetic energy (~ D^3 v^2),
    compressed with exponent 0.7 (large drops splash: less of their energy becomes sound)"""
    return ((d_mm / 2) ** 1.5 * _fall_speed(d_mm) / 6.55) ** 0.7


def _drops(out, r, surf, modes, t, a, d_mm, pan, detail):
    """raindrops (onsets t, amplitudes a, diameters d_mm) landing on a surface. A bigger drop is louder, lower and
    longer in every part. detail 1 = every part of a near drop; less = the cheaper, plainer drops of the middle
    distance (fewer modes, fewer bubbles)."""
    k = len(t)
    if not k:
        return
    s = d_mm / 2.0
    f2, q, ms, sharp, g = surf['tick']
    bursts(out, t, g * a, pan, f2 * s ** -0.45 * r.uniform(0.75, 1.3, k), q * r.uniform(0.8, 1.5, k),
           0.001 * ms * s ** 0.6 * r.uniform(0.8, 1.4, k), r, sharp)
    if surf['body']:
        lo, hi, ms, g, q = surf['body']
        b = d_mm > 1.0
        f = lo * (hi / lo) ** r.uniform(size=b.sum()) * s[b] ** -0.25
        bursts(out, t[b], g * a[b], pan[b], f, q, 0.001 * ms * r.uniform(0.6, 1.5, b.sum()), r, 0.7)
    if modes is not None:
        ms, g = surf['modes'][3:]
        b = r.uniform(size=k) < (1.0 if detail >= 1 else 0.2)          # far drops: one in five rings, twice as loud
        for _ in range(2 if detail >= 1 else 1):
            f = modes[r.integers(0, len(modes), b.sum())]
            rings(out, t[b], g * a[b] * r.uniform(0.15, 1.0, b.sum()) * (1.0 if detail >= 1 else 2.2), pan[b], f,
                  0.001 * ms * (f / 1000.0) ** -0.6, 0.0001)
    if surf['bubble']:
        b = r.uniform(size=k) < surf['bubble'] * detail
        _bubbles(out, t[b] + r.uniform(0.001, 0.005, b.sum()), 0.8 * a[b], pan[b],
                 (0.3 + 0.35 * d_mm[b]) * r.uniform(0.6, 1.4, b.sum()))


def _drip_times(r, dur, interval, jitter=0.22):
    """a dripping source: nearly periodic (every `interval` s, jittered), stretched so that it fits the circle"""
    k = max(1, int(round(dur / interval)))
    gaps = np.maximum(0.3, 1 + jitter * r.standard_normal(k))
    return (np.cumsum(gaps) / gaps.sum() + r.uniform()) % 1.0 * dur


def _rain(n, r, c):
    """Rain: drops arrive as a Poisson process whose rate comes and goes in slow waves, with Marshall-Palmer sizes;
    each drop is rendered on the chosen surface. Drops within a few metres are single events at their own position;
    a ring eight times that area is a denser, duller, more diffuse layer; everything beyond is a wash whose level
    follows the same waves. Light rain is mostly drops, a downpour mostly wash."""
    dur, surf, heavy = n / SR, SURFACES[c['surface']], c['intensity']
    mm_h = 5.0 * 10 ** heavy                                  # 0.5 drizzle .. 5 moderate .. 50 downpour
    wave = drift(dur, r, 0.03, 0.5, 1.0)
    modes = None
    if surf['modes']:
        cnt, lo, hi = surf['modes'][:3]
        modes = np.sort(lo * (hi / lo) ** r.uniform(size=cnt))
    near, mid = np.zeros((n, 2)), np.zeros((n, 2))
    lo, hi = surf['near']
    rate = 130.0 * 10 ** (0.65 * heavy)                       # single drops per second: 30 .. 130 .. 580
    for bus, lam, d0, d1, detail in ((near, rate, lo, hi, 1.0), (mid, 8 * rate, hi, 3 * hi, 0.34)):
        cap = min(lam, 5000.0)
        t, _ = arrivals(r, cap * np.exp(0.35 * wave - 0.06), dur)
        d_mm = _drop_sizes(r, len(t), mm_h)
        a = _drop_level(d_mm) / np.sqrt(r.uniform(d0 ** 2, d1 ** 2, len(t)))
        a = a / np.sqrt(1 + (a / 1.6) ** 2)                   # the rare giant right at the ear is held back
        _drops(bus, r, surf, modes, t, a * np.sqrt(lam / cap), d_mm, _pans(r, len(t)), detail)
    for _ in range(surf['drips']):
        t = _drip_times(r, dur, r.uniform(0.5, 1.8) * 10 ** (-0.25 * heavy))
        g, pan, size = 0.9 * np.exp(0.25 * r.standard_normal(len(t))), r.uniform(-0.8, 0.8), r.uniform(1.6, 3.2)
        if c['surface'] == 'leaves':                          # big drops off the canopy onto the litter
            bursts(near, t, g, pan, r.uniform(260, 520) * np.exp(0.08 * r.standard_normal(len(t))), 2.0, 0.004, r, 0.6)
            bursts(near, t, 0.7 * g, pan, 2600.0, 1.0, 0.0025, r, 0.4)
        else:                                                 # a gutter dripping into a puddle
            bub = size * np.exp(0.06 * r.standard_normal(len(t)))
            if surf.get('puddle', True):
                _bubbles(near, t + 0.003, g, pan, bub, 0.12)
            bursts(near, t, 0.6 * g, pan, 4500.0, 1.0, 0.0006, r)
    centres = _bands(90.0, 14000.0, 8)
    env = _levels(centres, *surf['wash'])[:, None] * envelopes(dur, r, 8, 0.12, 0.3, 0.1, 4.0)
    wash = band_noise(n, r, centres, env * np.exp(0.25 * wave)[None, :], 0.85)
    mid = _space(_filt(mid, _lp(6500.0, 1)), 0.25, 0.3, 21)
    # Real rain lands thousands of drops per second and square metre, a thousand times what is rendered one by one
    # here, and with levels falling as 1 / distance every doubling of distance adds the same energy: the wash is
    # the loud part. It sits 16 dB under the peaks of the nearest drops (22 in a drizzle, 10 in a downpour), with
    # the middle ring of drops just under it as its grain.
    wash *= _top(near) / _rms(wash) * 10 ** ((-16.0 + 6.0 * heavy + 6.0 * c['distance'] - 3.0 * surf['indoors']
                                              + surf.get('wash_db', 0.0)) / 20)
    y = near * 10 ** (-0.6 * c['distance']) + mid * (0.5 * _rms(wash) / _rms(mid)) + wash
    if surf['indoors']:
        y = _space(y, 0.3, 0.12, 22, size=0.3)
    return _filt(y, _lp(9000.0 * 10 ** (-0.5 * max(c['distance'], 0.0)), 1))        # air and distance take the top


# ------------------------------------------------------------------------------------------------ wind
PLACES = {
    # bed: the turbulence spectrum at 7 m/s (centre Hz, dB/oct below, above); top: a second, weaker hump of small
    # scale hiss (Hz, dB under the bed); turb: turbulence intensity (std of speed / mean); howl: default strength;
    # tones: Aeolian frequencies 0.2 v / d at 7 m/s of the gaps and wires around (d of 3-5 mm); whoosh: the broad
    # resonances (Hz at 7 m/s) of the flow past edges and hollows near the listener, which slide with the speed;
    # room: reverb of an interior
    'open': dict(bed=(190.0, 3.0, 5.5), top=(4500.0, -30.0), turb=0.20, howl=0.15, tones=(300.0, 470.0),
                 whoosh=(230.0, 520.0), room=0.0),
    'trees': dict(bed=(500.0, 3.0, 5.0), top=(5000.0, -36.0), turb=0.36, howl=0.2, tones=(380.0, 610.0),
                  whoosh=(300.0, 720.0), room=0.0),
    'window': dict(bed=(140.0, 2.5, 7.5), top=(3000.0, -44.0), turb=0.26, howl=0.5, tones=(265.0, 400.0, 655.0),
                   whoosh=(170.0, 340.0), room=0.14),
    'mountain': dict(bed=(320.0, 2.5, 5.0), top=(6000.0, -26.0), turb=0.26, howl=0.45,
                     tones=(420.0, 640.0, 980.0), whoosh=(260.0, 520.0, 950.0), room=0.0),
}


def _gusts(dur, r):
    """the fluctuation of wind speed, unit std: a von Karman spectrum (flat below 0.12 Hz, then the -5/3 slope of
    the inertial range), its slow part steepened like a real gust front (fast rise, slow fall: the control is read
    ahead in proportion to its own height), plus the fast flutter of the buffeting near the listener (a fifth of
    the variance: more than the slope alone gives, and what tells wind from surf)"""
    g = drift(dur, r, 0.02, 1.0, 5 / 3, knee=0.12)
    m = len(g)
    lead = 0.6 / (np.abs(np.diff(g, append=g[0])).max() + 1e-9) * np.maximum(g, 0.0)
    g = np.interp((np.arange(m) + lead) % m, np.arange(m + 1), np.append(g, g[0]))
    return (g + 0.5 * drift(dur, r, 1.0, 8.0, 5 / 3)) / 1.118


def _wind(n, r, c):
    """Wind: one wind speed v(t) drives everything. The bed is turbulence noise whose spectrum scales with v (its
    centre moves up with speed, as every shedding frequency does) and whose level goes as v^2.5; the gust reaches the
    left and right side a moment apart; edges near the listener add broad resonances that slide with it; leaves are
    thousands of small contacts whose rate goes as v^3, tree by tree as the gust arrives; howls are narrow noise
    bands at the Aeolian frequency 0.2 v / d of gaps and wires, heard only above an onset speed and gliding with
    the gusts."""
    dur, place = n / SR, PLACES[c['place']]
    speed = 7.0 * 2.5 ** c['intensity']                       # mean wind m/s: 2.8 breeze .. 7 .. 17.5 gale
    far = max(c['distance'], 0.0)
    rel = np.exp(place['turb'] * c.get('gust', 1.0) * _gusts(dur, r))             # v(t) / mean, log-normal
    lag = r.uniform(0.15, 0.45) * r.choice([-1.0, 1.0])
    centres = _bands(35.0, 14000.0, 10)
    fc, lo_db, hi_db = place['bed']
    fc = fc * (speed / 7.0) * 10 ** (-0.3 * far)
    top_f, top_db = place['top']
    own = envelopes(dur, r, 10, 0.18, 0.0, 0.3, 8.0)
    env = np.empty((10, len(rel), 2))
    for ch, shift in enumerate((-lag / 2, lag / 2)):
        v = _delayed(rel, shift, dur)[None, :]
        bed = _hump(centres[:, None], fc * v, lo_db, hi_db + 2.0 * far) * np.sqrt(centres[:, None] / fc)
        top = _hump(centres[:, None], top_f * np.sqrt(v), 6.0, 6.0) * np.sqrt(centres[:, None] / top_f)
        env[:, :, ch] = np.hypot(bed, 10 ** ((top_db - 12.0 * far) / 20) * top * v) * v ** 2.5 * own
    y = band_noise(n, r, centres, env, np.linspace(0.3, 1.0, 10))
    level = _rms(y)
    whoosh = np.zeros((n, 2))
    for f0 in place['whoosh']:                                # broad resonances that slide with the gusts
        v = _delayed(rel, r.uniform(-0.6, 0.6), dur) * np.exp(0.12 * drift(dur, r, 0.2, 4.0))
        f = _at(f0 * (speed / 7.0) * v * 10 ** (-0.2 * far), n)
        x = _svf(r.standard_normal(n), f, r.uniform(2.5, 5.0), 'bandn')
        _put(whoosh, x * _at(v ** 3, n), 0.0, 1.0, r.uniform(-0.8, 0.8))
    y += whoosh * (0.8 * level / _rms(whoosh))
    if c['place'] == 'trees':
        leaves = np.zeros((n, 2))
        # leaves are papery flaps (1.5-8 kHz, about a millisecond each) whose rate goes as v^3: the rustle swells
        # and falls with every gust, tree by tree, and stands well out of the bed (it is what says "trees")
        lam = 900.0 * (speed / 7.0) ** 3                      # leaf contacts per second and tree at the mean speed
        for pan in (-0.85, -0.35, 0.3, 0.8):
            v = _delayed(rel, r.uniform(0.1, 1.2) + 0.5 * pan * lag, dur) * np.exp(0.15 * drift(dur, r, 0.2, 3.0))
            cap = min(lam, 4000.0)
            t, _ = arrivals(r, cap * v ** 3, dur)
            a = sizes(r, len(t), 0.15, 6.0, 'lognormal', 0.7)
            bursts(leaves, t, a * _val(v, t, dur), _pans(r, len(t), 0.3, pan + r.uniform(-0.1, 0.1)),
                   4500.0 * a ** -0.3 * r.uniform(0.6, 1.6, len(t)), r.uniform(1.0, 3.0, len(t)),
                   0.001 * a ** 0.5 * r.uniform(0.6, 1.6, len(t)), r, 0.3)
        y += leaves * (0.8 * (speed / 7.0) ** 0.5 * 10 ** (-0.5 * far) * level / _rms(leaves))
    howl = (place['howl'] if c['howl'] is None else c['howl']) * (1 - far)
    if howl > 0:
        tones = np.zeros((n, 2))
        for j, f0 in enumerate(place['tones']):
            v = _delayed(rel, r.uniform(-0.8, 0.8), dur) * np.exp(0.1 * drift(dur, r, 0.1, 2.0))
            f = _at(f0 * (speed / 7.0) ** 0.7 * v ** 0.7, n)              # the pitch follows the gust
            blow = _at(np.clip((v - r.uniform(0.75, 0.95)) / 0.35, 0.0, 1.0) ** 2 * v ** 3, n)
            x = _svf(r.standard_normal(n), f, 40.0, 'bandn')
            if j == 0:
                x += 0.2 * _svf(r.standard_normal(n), 2 * f, 40.0, 'bandn')
            _put(tones, x * blow, 0.0, r.uniform(0.6, 1.0), r.uniform(-0.7, 0.7))
        y += _space(tones, 0.5, 0.35, 23) * (1.1 * howl * level / _rms(tones))
    return _space(y, 0.35, place['room'], 24, size=0.3) if place['room'] else y


# ------------------------------------------------------------------------------------------------ fire
def _fire(n, r, c):
    """Fire: what a campfire is heard as is its cracks: wood splitting as sap and gas burst, sharp broadband
    clicks of very different sizes (power law over 25 dB: many small ticks, now and then a loud snap whose body
    rings in the wood a few ms) that come in irregular bursts (a self-exciting process: a crack sets off a few
    more within 20 ms) separated by quiet gaps of a second or more (the rate itself comes and goes in episodes,
    the way rain comes in waves). Under them a soft low flutter of the flames that breathes at the puffing
    frequency 1.5 / sqrt(diameter) Hz (Cetegen & Ahmed 1993: a candle flickers near 10 Hz, a campfire breathes at
    2 Hz, a large fire below 1 Hz), almost no hiss (the hiss of escaping gas and the roar belong to a big fire),
    and rare low pops that throw a spray of sparks."""
    dur, size = n / SR, c['size']
    diameter = 0.5 * 10 ** (1.2 * size)                       # m: 0.03 candle .. 0.5 campfire .. 8 large fire
    puff = 1.5 / np.sqrt(diameter)
    breath = 0.8 * drift(dur, r, 0.4 * puff, 2.5 * puff, 1.0) + 0.6 * drift(dur, r, 0.03, 0.4 * puff, 1.0)
    flame = np.exp(0.45 * breath)
    centres = _bands(40.0, 14000.0, 10)
    roar = _levels(centres, 160.0 * 10 ** (-0.35 * size), 3.0, 8.0)
    hiss = _levels(centres, 5200.0, 6.0, 3.0)[:, None] * envelopes(dur, r, 10, 0.8, 0.5, 2.0, 25.0, 0.5)
    env = np.hypot(roar[:, None] * flame[None, :] * 10 ** (0.6 * size),
                   0.06 * 10 ** (1.2 * size) * hiss * np.sqrt(flame)[None, :])
    y = band_noise(n, r, centres, env, np.linspace(0.4, 1.0, 10))
    spread = 0.2 + 0.4 * (size + 1)                           # stereo width of the fire: 0.2 candle .. 1
    busy = 7.0 * 10 ** ((0.6 if size > 0 else 0.8) * size + 0.5 * c['intensity'])         # cracks per second
    # episodes: log-normal with a wide spread, so the fire has busy spells and lulls of a second or two
    episodes = np.exp(1.1 * drift(dur, r, 0.08, 1.5, 1.0) - 0.6)
    t, gen = arrivals(r, busy * flame ** 1.5 * episodes, dur, cluster=0.5, spread=0.02)
    k = len(t)
    a = sizes(r, k, 1.0, 18.0, 'power', 2.0) * 0.75 ** gen    # amplitude: Pareto over 25 dB, echoes smaller
    a = a / np.sqrt(1 + (a / 9.0) ** 2)                       # the rare giant snap is held back (as the rain's drops)
    wood = np.zeros((n, 2))
    pan = np.clip(spread * 0.8 * r.standard_normal(k), -1, 1)
    bursts(wood, t, a, pan, 3800.0 * a ** -0.3 * np.exp(0.35 * r.standard_normal(k)), r.uniform(0.6, 1.6, k),
           0.0004 * a ** 0.5 * r.uniform(0.7, 1.4, k), r)
    big = a > 4.0                                             # a loud snap rings in the stick for a few ms
    rings(wood, t[big], 0.35 * a[big], pan[big], r.uniform(500.0, 1800.0, big.sum()),
          r.uniform(0.002, 0.006, big.sum()))
    if size > -0.6:
        tp, _ = arrivals(r, 0.15 * 10 ** (0.4 * size + 0.3 * c['intensity']) * flame, dur)
        kp = len(tp)
        ap, pp = r.uniform(7.0, 12.0, kp), np.clip(spread * 0.6 * r.standard_normal(kp), -1, 1)
        rings(wood, tp, 0.4 * ap, pp, r.uniform(250.0, 600.0, kp), r.uniform(0.006, 0.014, kp), 0.0003, -10.0)
        bursts(wood, tp, ap, pp, r.uniform(1800.0, 3200.0, kp), 0.7, r.uniform(0.0012, 0.0025, kp), r)
        sparks = np.repeat(np.arange(kp), r.integers(3, 9, kp))
        ts = (tp[sparks] + r.exponential(0.04, len(sparks))) % dur
        bursts(wood, ts, r.uniform(0.5, 3.0, len(ts)), np.clip(pp[sparks] + 0.2 * r.standard_normal(len(ts)), -1, 1),
               r.uniform(3500.0, 9000.0, len(ts)), 1.5, 0.0004, r)
    if k:                    # the flutter: 26 dB under the cracks' peaks at a campfire, a roar under a big fire
        y *= _top(wood) / _rms(y) * 10 ** ((-26.0 + 14.0 * max(size, 0.0) + 7.0 * min(size, 0.0)) / 20)
        #                                                                   and a whisper under a small one
    y = _space(y + wood, 0.4, 0.05, 25)
    far = max(c['distance'], 0.0)
    return _filt(y, _lp(16000.0 * 10 ** (-0.6 * far), 1)) if far else y


# ------------------------------------------------------------------------------------------------ water
def _stream(n, r, c):
    """Running water: a few riffles (small falls over stones), each a source of bubbles with its own preferred bubble
    size, rate and flow fluctuation, so the stream has a place-by-place tone and its bands move independently; under
    them the general population of bubbles (power-law radii, many tiny) and a broadband splash wash. Trickle to
    river: 70 to 7000 bubbles per second, the wash and a low roar taking over."""
    dur, size = n / SR, c['size']
    lam = 700.0 * 10 ** (1.0 * size + 0.3 * c['intensity'])   # bubbles per second: 70 trickle .. 700 brook .. 7000
    riffles = int(round(3 + 2 * size))
    share = r.dirichlet(np.full(riffles, 2.0))
    water = np.zeros((n, 2))
    where = r.uniform(0.35, 0.95, riffles) * (-1.0) ** (np.arange(riffles) + r.integers(2))       # left, right, ..
    for j in range(riffles):
        flow = np.exp(0.45 * drift(dur, r, 0.2, 2.0, 1.0) - 0.1)
        t, _ = arrivals(r, 0.7 * lam * share[j] * flow, dur, cluster=0.3, spread=0.02)
        radius = sizes(r, len(t), 0.25, 5.0, 'lognormal', r.uniform(0.3, 0.55)) * r.uniform(0.55, 1.6) * 1.3 ** size
        depth = r.uniform(0.4, 1.0, len(t)) ** 1.5
        _bubbles(water, t, radius ** 0.8 * depth, _pans(r, len(t), 0.3, where[j]), radius,
                 r.choice([0.0, 0.1, 0.2], len(t)))
    t, _ = arrivals(r, 0.3 * lam * np.exp(0.3 * drift(dur, r, 0.2, 2.0, 1.0)), dur)
    radius = sizes(r, len(t), 0.3, 8.0, 'power', 2.5)
    _bubbles(water, t, radius ** 0.8 * r.uniform(0.4, 1.0, len(t)) ** 1.5, _pans(r, len(t)), radius,
             r.choice([0.0, 0.1, 0.2], len(t)))
    centres = _bands(60.0, 14000.0, 9)
    splash = _levels(centres, 3200.0 * 10 ** (-0.15 * size), 3.0, 4.5)
    roar = 10 ** (0.9 * size - 0.6) * _levels(centres, 160.0, 3.0, 6.0)
    env = np.hypot(splash, roar)[:, None] * envelopes(dur, r, 9, 0.3, 0.15, 0.3, 12.0, 0.8)
    wash = band_noise(n, r, centres, env, 0.9)
    far = c['distance']
    wash *= _rms(water) / _rms(wash) * 10 ** ((-5.0 + (5.0 if size > 0 else 11.0) * size + 5.0 * far) / 20)
    y = water * 10 ** (-0.5 * far) + wash
    return _filt(y, _lp(16000.0 * 10 ** (-0.9 * far), 1)) if far > 0 else y


# ------------------------------------------------------------------------------------------------ birds
def _note(r, dur, f0, f1, bow=0.0, trill=0.0, trill_hz=40.0, harm=0.08):
    """one note of a syrinx: a tone gliding from f0 to f1 Hz over dur s (bow arches the glide up and back, or down
    when negative) after a short downward chip at the onset, with the small pitch jitter and loudness flutter of a
    living source, an optional frequency trill
    (trill Hz of deviation at trill_hz), a second and third harmonic (harm) and a soft attack and release"""
    n = dsp.n_of(dur)
    u = np.linspace(0.0, 1.0, n)
    knots = np.linspace(0.0, 1.0, max(3, int(dur * 60)))
    f = f0 * (f1 / f0) ** u * (1 + bow * np.sin(np.pi * u)) * (1 + 0.06 * np.exp(-u * dur / 0.006))    # onset chip
    f = f * (1 + 0.006 * np.interp(u, knots, r.standard_normal(len(knots))))
    if trill:
        f = f + trill * np.sin(2 * np.pi * trill_hz * dur * u + r.uniform(0, 6.28))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph)
    for h, a in ((2, harm), (3, 0.4 * harm)):
        if h * f.max() < 0.45 * SR:
            y += a * np.sin(h * ph + r.uniform(0, 6.28))
    a = max(2, min(n // 3, dsp.n_of(0.006 + 0.08 * dur)))
    env = np.ones(n)
    env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    env[n - 2 * a:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, 2 * a))
    return y * env * (1 + 0.15 * np.interp(u, knots, r.standard_normal(len(knots))))


def _tit(r, v):
    """two pure notes, high and low, alternating three to six times (the seesaw song of a tit)"""
    hi = 4300.0 + 900.0 * v[0]
    lo, d, gap = hi * (0.72 + 0.08 * v[1]), 0.085 + 0.03 * v[2], 0.05 + 0.03 * v[3]
    out, t = [], 0.0
    for _ in range(int(r.integers(3, 7))):
        for f, k in ((hi, 1.0), (lo, 1.25)):
            out.append((t, _note(r, d * k, f * 1.02, f * 0.985)))
            t += d * k + gap
    return out


def _chiff(r, v):
    """an irregular series of sharp downslurred chips on two or three pitches (a chiffchaff-like warbler)"""
    pitches = (5300.0 + 500.0 * v[0]) * np.array([1.0, 0.78 + 0.05 * v[1], 0.88])
    out, t = [], 0.0
    for _ in range(int(r.integers(6, 15))):
        f = pitches[r.choice(3, p=[0.45, 0.4, 0.15])]
        out.append((t, _note(r, 0.09, f * 1.35, f * 0.97, -0.08)))
        t += (0.27 + 0.04 * v[2]) * r.uniform(0.92, 1.1)
    return out


def _finch(r, v):
    """a rattle of chips that falls in three steps and speeds up, ended by a flourish (a chaffinch-like finch)"""
    top, count = 5600.0 + 600.0 * v[0], int(r.integers(10, 15))
    out, t = [], 0.0
    for i in range(count):
        f = top * (1.0, 0.8, 0.62)[3 * i // count]
        out.append((t, _note(r, 0.06, f * 1.15, f * 0.85, 0.05)))
        t += 0.15 - 0.065 * i / count
    for d, f0, f1, bow in ((0.09, 2600.0, 5200.0, 0.0), (0.12, 5000.0, 3000.0, 0.0), (0.15, 3600.0, 2400.0, 0.1)):
        out.append((t, _note(r, d, f0 * (0.95 + 0.1 * v[1]), f1 * (0.95 + 0.1 * v[1]), bow)))
        t += d + 0.015
    return out


def _thrush(r, v):
    """slow fluting whistles that slur between pitches, then a short high twitter (a blackbird-like thrush)"""
    centre = 2000.0 + 500.0 * v[0]
    out, t = [], 0.0
    for _ in range(int(r.integers(4, 8))):
        f, d = centre * 2 ** r.uniform(-0.3, 0.6), r.uniform(0.08, 0.24)
        out.append((t, _note(r, d, f, f * 2 ** r.uniform(-0.4, 0.4), r.uniform(-0.05, 0.08), harm=0.05)))
        t += d + r.uniform(0.03, 0.14)
    for _ in range(int(r.integers(2, 6))):
        f = 6500.0 * r.uniform(0.85, 1.15)
        out.append((t, 0.4 * _note(r, 0.05, f, 0.86 * f * r.uniform(0.9, 1.1), 0.0, 500.0, 45.0)))
        t += 0.075
    return out


def _wren(r, v):
    """two or three fast trills, each on its own pitch (a wren-like song)"""
    out, t = [], 0.0
    for _ in range(int(r.integers(2, 4))):
        f, rate = (4200.0 + 2800.0 * r.uniform()) * (0.9 + 0.2 * v[0]), r.uniform(14.0, 22.0)
        for _ in range(int(r.uniform(0.4, 0.8) * rate)):
            out.append((t, _note(r, 0.62 / rate, f * 1.22, f * 0.86)))
            t += 1 / rate
        t += r.uniform(0.02, 0.08)
    return out


def _dove(r, v):
    """a soft, low five-note coo in a fixed rhythm, two or three times over (a wood-pigeon-like dove)"""
    f = 430.0 + 70.0 * v[0]
    out, t = [], 0.0
    for _ in range(int(r.integers(2, 4))):
        for d, k in ((0.32, 0.96), (0.52, 1.04), (0.50, 1.0), (0.24, 0.94), (0.24, 0.94)):
            out.append((t, 0.5 * _note(r, d, f * k * 0.96, f * k * 0.97, 0.06, harm=0.22)))
            t += d + 0.09
        t += 0.35
    return out


def _pecker(r, v):
    """a woodpecker's drum roll on a dead branch: 10-16 knocks at about 20 per second, fading"""
    f, rate, count = 950.0 + 500.0 * v[0], 17.0 + 6.0 * v[1], int(r.integers(10, 17))
    tk = dsp.t_of(dsp.n_of(0.05))
    knock = (np.sin(2 * np.pi * f * tk) * np.exp(-tk / 0.004) + 0.6 * np.sin(2 * np.pi * 0.43 * f * tk)
             * np.exp(-tk / 0.009)) * np.minimum(1.0, tk / 0.0003)
    return [(i / rate, 0.7 * (1 - 0.5 * i / count) * r.uniform(0.8, 1.0) * dsp.fade(knock, 0.0, 0.01))
            for i in range(count)]


BIRDS = {   # song (r, voice) -> [(onset s, note)], and the median pause in s between two songs of one bird
    'tit': (_tit, 4.0), 'chiff': (_chiff, 5.0), 'finch': (_finch, 7.0), 'thrush': (_thrush, 5.0),
    'wren': (_wren, 8.0), 'dove': (_dove, 12.0), 'pecker': (_pecker, 14.0),
}


def _forest(n, r, c):
    """A forest by day: a light wind in the leaves (the wind model at a breeze) under birds of several species. Every
    bird keeps its own voice, perch and distance, sings its species' phrase with fresh variations and waits a
    log-normal pause; most are far off (quieter, duller, with the long diffuse echo of a wood). The low coo of a
    dove is heard by listeners as an owl and is left out unless `dove` is above 0."""
    dur = n / SR
    y = _wind(n, r, dict(intensity=-0.75 + 0.2 * c['intensity'], place='trees', howl=0.0, distance=0.0,
                         gust=0.5))                           # under the canopy the gusts are half as deep
    y *= 0.4 / _rms(y)                                        # a quiet wood: the birds stand 25 dB over the air
    near, far = np.zeros((n, 2)), np.zeros((n, 2))
    names = [list(BIRDS)[i] for i in r.permutation(len(BIRDS))]
    if c.get('dove', 0.0) <= 0:
        names.remove('dove')
    for i in range(int(round(6 + 4 * c['intensity']))):                           # 2 .. 6 .. 10 singers
        song, pause = BIRDS[names[i % len(names)]]
        voice, pan = r.uniform(size=4), r.uniform(0.3, 0.9) * r.choice([-1.0, 1.0])       # perched to one side
        away = r.uniform() < 0.55 + 0.3 * c['distance']
        gain = (0.25 if away else 1.3) * r.uniform(0.6, 1.3)
        start = t = r.uniform(0, min(dur, pause))
        while True:
            notes = song(r, voice)
            length = notes[-1][0] + len(notes[-1][1]) / SR
            if t + length > start + dur:                      # a bird never sings over its own first song
                break
            for tn, x in notes:
                _put(far if away else near, x, t + tn, gain, pan)
            t += length + pause * np.exp(0.6 * r.standard_normal())
    birds = _space(near, 1.2, 0.22, 26, size=0.8) + _space(_filt(far, _lp(5500.0, 1)), 1.6, 0.7, 27, size=0.9)
    return y + 6.0 * birds


# ------------------------------------------------------------------------------------------------ night
def _chirps(r, dur, period, jitter=0.04, sway=0.05):
    """the chirp onsets of one insect: nearly periodic, every interval jittered, the tempo swaying slowly, the whole
    stretched to fit the circle"""
    k = max(1, int(round(dur / period)))
    tempo = _val(drift(dur, r, 0.03, 0.3, 1.0), np.arange(k) * period, dur)
    gaps = np.exp(jitter * r.standard_normal(k) + sway * tempo)
    return (np.cumsum(gaps) / gaps.sum() + r.uniform()) % 1.0 * dur


def _night(n, r, c):
    """A summer night: insects are pulse trains. Each wing stroke is a short, nearly pure ring (the file and scraper
    excite a resonance of the wing, gliding down a little); strokes form chirps and trills whose rates rise with the
    temperature: a snowy tree cricket chirps (T_F - 40) / 15 times per second (Dolbear's law), each at its own
    slightly different tempo; field crickets chirp faster and in bouts; ground crickets trill without a break. A
    few sing close by, the crowd is far and soft; now and then a tree frog (an owl only when `owl` is above 0); a
    soft air under it all. (Lower, rougher strokes are heard as random trills rather than crickets.)"""
    dur = n / SR
    temp = 20.0 + 8.0 * c['warmth']                           # deg C: 12 .. 20 .. 28
    beat = (temp * 1.8 - 8.0) / 15.0                          # chirps per second, Dolbear
    quick = 1.0 + 0.04 * (temp - 20.0)                        # stroke rates rise about 4 % per degree
    tune = 1.0 + 0.012 * (temp - 20.0)                        # and the carrier a little
    near, far = np.zeros((n, 2)), np.zeros((n, 2))
    seats = r.permutation([-0.85, -0.3, 0.35, 0.9])
    for i in range(int(round(14 * 10 ** (0.35 * c['intensity'])))):                # 6 .. 14 .. 31 singers
        away = i >= 4
        gain = 1 / (r.uniform(4.0, 30.0) if away else r.uniform(1.0, 3.0))
        pan = _pans(r, 1)[0] if away else seats[i] + r.uniform(-0.1, 0.1)
        awake = drift(dur, r, 0.04, 0.3, 0.0)                 # a singer rests when this dips
        if i % 3 == 0:                                        # tree cricket: slow, even chirps at Dolbear's rate
            f, strokes, rate = r.uniform(2500.0, 3200.0) * tune, int(r.integers(5, 9)), 48.0 * quick
            t = _chirps(r, dur, 1 / (beat * r.uniform(0.97, 1.03)), 0.05, 0.08)
            rise, decay = np.minimum(1.0, 0.45 + 0.3 * np.arange(strokes)), 0.004
        elif i % 3 == 1:                                      # field cricket: quick chirps of 3-4 strokes
            f, strokes, rate = r.uniform(4300.0, 5100.0) * tune, int(r.integers(3, 5)), 28.0 * quick
            t = _chirps(r, dur, 1 / (2.6 * quick * r.uniform(0.8, 1.25)), 0.07, 0.08)
            t = t[_val(awake, t, dur) > -0.7]
            rise, decay = np.minimum(1.0, 0.6 + 0.25 * np.arange(strokes)), 0.006
        else:                                                 # ground cricket: an unbroken trill
            f, strokes, rate = r.uniform(5600.0, 7600.0) * tune, 1, 42.0 * quick * r.uniform(0.85, 1.15)
            t = _chirps(r, dur, 1 / rate, 0.02, 0.02)
            t = t[_val(awake, t, dur) > -1.0]
            rise, decay, gain = np.ones(1), 0.003, 0.35 * gain
        tt = (t[:, None] + np.arange(strokes)[None, :] / rate).ravel()
        swell = np.exp(0.3 * _val(drift(dur, r, 0.05, 0.5, 1.0), tt, dur))
        rings(far if away else near, tt, gain * swell * np.tile(rise, len(t)), pan,
              f * np.exp(0.004 * r.standard_normal(len(tt))), decay, decay, -2.0)
    if c['intensity'] > -0.5:
        for tb in arrivals(r, 0.11, dur)[0]:                  # a tree frog: a few short rattling calls
            f, pan = r.uniform(950.0, 1300.0), r.uniform(-0.8, 0.8)
            for j in range(int(r.integers(2, 5))):
                tt = tb + 0.55 * j + np.arange(int(r.integers(9, 14))) / 38.0
                rings(far, tt, 0.07, pan, f, 0.005, 0.002)
                rings(far, tt, 0.04, pan, 2.05 * f, 0.004, 0.002)
        for tb in arrivals(r, 0.035 * c.get('owl', 0.0), dur)[0] if c.get('owl', 0.0) > 0 else ():   # an owl
            f, pan = r.uniform(330.0, 400.0), r.uniform(-0.8, 0.8)
            for t0, d in ((0.0, 0.32), (0.85, 0.14), (1.07, 0.14), (1.5, 0.65)):
                _put(far, _note(r, d, f * 0.97, f * 0.985, 0.04, harm=0.15), tb + t0, 0.12, pan)
    far = _space(_filt(far, _lp(9000.0, 1)), 1.0, 0.5, 28, size=0.9)
    y = near * 10 ** (-0.5 * c['distance']) + 2.5 * far
    centres = _bands(60.0, 9000.0, 8)
    env = _levels(centres, 250.0, 3.0, 4.5)[:, None] * envelopes(dur, r, 8, 0.15, 0.6, 0.03, 1.0, 1.2)
    air = band_noise(n, r, centres, env, 0.9)
    y = y + air * (10 ** (-20 / 20) * _rms(y) / _rms(air))
    return _filt(y, _lp(12000.0 * 10 ** (-0.5 * c['distance']), 1)) if c['distance'] > 0 else y


# ------------------------------------------------------------------------------------------------ interiors
def _space_hum(n, r, c):
    """The hum of a ship or a machine room: a stack of low harmonics, every one a pair of tones a fraction of a hertz
    apart, one towards each side, so the stack beats slowly and never settles; a second, weaker machine a fifth above;
    the broad noise of moving air; a faint high whine. All tones complete whole cycles in the buffer (it loops)."""
    dur, grid = n / SR, SR / n
    t = np.arange(n) / SR
    f0 = 55.0 * 2 ** (-0.5 * c['size'])                       # 78 Hz small craft .. 55 .. 39 Hz large ship
    tones = np.zeros((n, 2))
    stack = [(k, k ** -(1.3 - 0.4 * c['bright'])) for k in (1, 2, 3, 4, 5, 6, 8)] + [(1.5, 0.3), (3.0 * 1.5, 0.12)]
    stack += [(110.0, 10 ** ((-44 + 8 * c['bright']) / 20)), (163.0, 10 ** ((-50 + 8 * c['bright']) / 20))]   # whine
    for mult, amp in stack:
        f = max(1, round(f0 * mult / grid)) * grid
        beat = max(1, round(r.uniform(0.07, 0.45) * (1 + 0.02 * mult) / grid)) * grid
        for fk, ch in ((f, 0), (f + beat, 1)):                # a pair: each tone mostly on its own side, its phase
            wander = r.uniform(0, 6.28) + 2.5 * ch * _at(drift(dur, r, 0.02, 0.25), n)     # wandering: uneven beats
            x = np.sin(2 * np.pi * fk * t + wander) * amp * np.exp(0.15 * _at(drift(dur, r, 0.02, 0.3), n))
            tones[:, ch] += 0.97 * x
            tones[:, 1 - ch] += 0.22 * x
    centres = _bands(40.0, 9000.0, 8)
    env = _levels(centres, 180.0 * 2 ** (0.5 * c['bright']), 2.0, 5.0)[:, None] * envelopes(
        dur, r, 8, 0.12, 0.6, 0.03, 0.6, 1.2)
    air = band_noise(n, r, centres, env, np.linspace(0.3, 0.9, 8))
    return tones + air * (10 ** (-20 / 20) * _rms(tones) / _rms(air))


# ------------------------------------------------------------------------------------------------ events and grains
def _thunder(n, r, c):
    """Thunder: the lightning channel is a tortuous path a few km long, straight up from the strike and then along
    the cloud; every few metres of it radiate one N-shaped pressure wave, and what arrives is their sum in order of
    distance (343 m/s): where the path runs across the line of sight the arrivals pile up into claps, elsewhere they
    spread into the roll (Few 1969; Ribner & Roy 1982). Farther parts arrive weaker (as distance^-0.6 here: slower
    than one source falls, for the branches that are not traced), longer (2-14 ms), duller (air has taken
    the highs: about 2 kHz left at 1 km, 200 Hz at 10 km) and from where they are; terrain echoes and a long low
    reverberation follow. distance: 240 m (a crack, then the roll) .. 1.5 km .. 9.5 km (a soft rumble only)."""
    dur = n / SR
    away, step = 1500.0 * 10 ** (0.8 * c['distance']), 3.0
    k = 3000
    rise = r.uniform(2500.0, 4500.0) / step                   # sources in the vertical part of the channel
    heading = r.uniform(0, 2 * np.pi)
    wander = r.standard_normal((k, 3))
    for a, w in ((0.997, 1.3), (0.975, 1.0), (0.85, 0.7)):    # tortuosity on three scales: ~1 km, 120 m, 20 m
        wander += w * signal.lfilter([np.sqrt(1 - a * a)], [1.0, -a], r.standard_normal((k, 3)), axis=0)
    idx = np.arange(k)[:, None]
    bias = np.where(idx < rise, [[0.0, 0.0, 1.0]], [[np.cos(heading), np.sin(heading), 0.05]])
    d = 1.4 * bias + 0.55 * wander
    az = r.uniform(-0.6, 0.6)
    pos = np.array([away * np.cos(az), away * np.sin(az), 0.0]) + step * np.cumsum(
        d / np.linalg.norm(d, axis=1, keepdims=True), axis=0)
    dist = np.linalg.norm(pos, axis=1)
    arrive = 0.25 + (dist - dist.min()) / 343.0
    amp = np.exp(0.5 * r.standard_normal(k)) / dist ** 0.6
    th = (np.clip(pos[:, 1] / dist / 0.6, -1, 1) + 1) * np.pi / 4
    keep = arrive < dur
    cls = np.minimum((np.log(dist / dist.min()) / (np.log(dist.max() / dist.min()) + 1e-9) * 6).astype(int), 5)
    y = np.zeros((n, 2))
    for j in range(6):
        sel = keep & (cls == j)
        if not sel.any():
            continue
        metres = np.exp(np.mean(np.log(dist[sel])))
        length = int(np.clip(0.004 * np.sqrt(metres / 1000.0), 0.0015, 0.014) * SR)
        kernel = dsp.butter(np.concatenate([np.linspace(1.0, -1.0, length), np.zeros(8 * length + 512)]), 'low',
                            np.clip(2.0e6 / metres, 90.0, 7000.0))
        at = np.round(arrive[sel] * SR).astype(np.int64)
        train = np.stack([np.bincount(at, amp[sel] * g, n)[:n] for g in (np.cos(th[sel]), np.sin(th[sel]))], axis=1)
        y += signal.fftconvolve(train, kernel[:, None], axes=0)[:n]
    for j in range(3):                                        # echoes off the terrain: late, low, from the other side
        lag = dsp.n_of(r.uniform(0.35, 2.4))
        if lag < n:
            y[lag:] += 0.35 * 0.7 ** j * dsp.butter(y[:n - lag, ::-1], 'low', 600.0)
    far = 0.5 * (c['distance'] + 1)
    y = fx.reverb(y, r, rt60=3.2, size=0.95, damp=0.75, wet=0.25 + 0.35 * far, ir_seed=30)[:n]
    return dsp.butter(y, 'high', 18.0)


def _crackle(n, r, c):
    """Crackle of a worn record or a loose contact: very short broadband ticks in bursts (a self-exciting process),
    power-law sizes, the largest with a low thump of the stylus, over a faint steady hiss and rumble."""
    dur = n / SR
    busy = 22.0 * 10 ** (0.7 * c['intensity'])                # ticks per second: 4 .. 22 .. 110
    t, _ = arrivals(r, busy * np.exp(0.4 * drift(dur, r, 0.05, 1.0, 1.0) - 0.08), dur, cluster=0.35, spread=0.02)
    k = len(t)
    a, pan = sizes(r, k, 1.0, 15.0, 'power', 2.2), _pans(r, k, 0.8)
    y = np.zeros((n, 2))
    bursts(y, t, a, pan, 7000.0 * a ** -0.3 * r.uniform(0.7, 1.4, k), r.uniform(0.7, 1.5, k), 0.00012 * a ** 0.35, r)
    big = a > 6.0
    rings(y, t[big], 0.25 * a[big], pan[big], r.uniform(180.0, 420.0, big.sum()), r.uniform(0.002, 0.004, big.sum()))
    centres = _bands(40.0, 14000.0, 8)
    lev = np.hypot(_levels(centres, 3500.0, 2.5, 2.5), 0.5 * _levels(centres, 45.0, 3.0, 8.0))
    floor = band_noise(n, r, centres, lev[:, None] * envelopes(dur, r, 8, 0.06, 0.3, 0.05, 1.0), 0.9)
    return y + floor * (10 ** (-23 / 20) * _top(y) / _rms(floor)) if k else floor


NOTES = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
         'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def _sparkle(n, r, c):
    """A shimmer cloud: bright grains at random times (in loose runs, the density coming and going), each a struck
    partial on a note of the key's pentatonic scale between 2 and 10 kHz with the second mode of a small bar above
    it (2.76 f), a level drawn log-normally (sigma 6 dB), its own position and a slight upward glide, in a long
    bright room."""
    dur = n / SR
    busy = 80.0 * 10 ** (0.75 * c['density'])                 # grains per second: 14 .. 80 .. 450
    root, minor = c['key']
    scale = np.array([261.626 * 2 ** ((root + s) / 12 + o) for o in range(2, 6)
                      for s in ((0, 3, 5, 7, 10) if minor else (0, 2, 4, 7, 9))])
    scale = scale[(scale > 1900.0) & (scale < 10500.0)]
    t, _ = arrivals(r, busy * np.exp(0.5 * drift(dur, r, 0.1, 2.0, 1.0) - 0.12), dur, cluster=0.3, spread=0.06)
    k = len(t)
    f = scale[np.minimum((r.uniform(size=k) ** 1.4 * len(scale)).astype(np.int64), len(scale) - 1)]
    f = f * 2 ** (4.0 * r.standard_normal(k) / 1200)
    a, pan = np.exp(0.69 * r.standard_normal(k)), _pans(r, k)
    decay = r.uniform(0.05, 0.28, k) * np.sqrt(2500.0 / f) * min(1.0, (120.0 / busy) ** 0.4)
    y = np.zeros((n, 2))
    rings(y, t, a, pan, f, decay, r.uniform(0.001, 0.005, k), r.uniform(0.01, 0.03, k) / (3 * decay))
    up = 2.756 * f < 0.42 * SR
    rings(y, t[up], 10 ** ((-14 + 6 * c['bright']) / 20) * a[up], pan[up], 2.756 * f[up], 0.35 * decay[up], 0.001)
    return _space(y, 2.2, 0.35, 29, size=0.8, damp=0.3)


# ------------------------------------------------------------------------------------------------ the scenes
SCENES = {
    'rain': {'says': ['steady rain falling', 'the sound of rainfall', 'rain pattering on a surface'],
             'loop': True, 'min_loop': 8, 'controls': ['intensity', 'surface', 'distance', 'bright', 'width'],
             'choices': {'surface': list(SURFACES)}},
    'wind': {'says': ['wind blowing', 'gusts of wind', 'wind howling outside'],
             'loop': True, 'min_loop': 12, 'controls': ['intensity', 'howl', 'place', 'distance', 'bright', 'width'],
             'choices': {'place': list(PLACES)}},
    'fire': {'says': ['a crackling campfire', 'a wood fire burning and crackling', 'fire crackle'],
             'loop': True, 'min_loop': 8, 'controls': ['size', 'intensity', 'distance', 'bright', 'width']},
    'stream': {'says': ['a babbling brook', 'water flowing in a stream', 'running water'],
               'loop': True, 'min_loop': 6, 'controls': ['size', 'intensity', 'distance', 'bright', 'width']},
    'forest': {'says': ['birds singing in a forest', 'a quiet forest in the morning with birdsong',
                        'birdsong and a light breeze in the trees'],
               'loop': True, 'min_loop': 20, 'controls': ['intensity', 'distance', 'bright', 'width', 'dove']},
    'night': {'says': ['crickets chirping at night', 'night insects in a field',
                       'a summer night ambience with crickets'],
              'loop': True, 'min_loop': 10, 'controls': ['intensity', 'warmth', 'distance', 'bright', 'width', 'owl']},
    'space_hum': {'says': ['the low hum of a spaceship engine room', 'a deep machine drone',
                           'a science fiction ship interior hum'],
                  'loop': True, 'min_loop': 8, 'controls': ['size', 'bright', 'width']},
    'thunder': {'says': ['a clap of thunder and a rolling rumble', 'thunder rumbling', 'a thunderstorm'],
                'loop': False, 'min_loop': None, 'controls': ['distance', 'bright', 'width']},
    'crackle': {'says': ['vinyl record crackle', 'crackling static from an old record',
                         'electrical crackle and pops'],
                'loop': True, 'min_loop': 6, 'controls': ['intensity', 'bright', 'width']},
    'sparkle': {'says': ['a magical shimmering sparkle', 'twinkling chimes', 'fairy dust glitter sound'],
                'loop': True, 'min_loop': 8, 'controls': ['key', 'density', 'bright', 'width']},
}
_BUILD = {'rain': _rain, 'wind': _wind, 'fire': _fire, 'stream': _stream, 'forest': _forest, 'night': _night,
          'space_hum': _space_hum, 'thunder': _thunder, 'crackle': _crackle, 'sparkle': _sparkle}


def _key(key):
    """'F#' or 'F#m' or a pitch class 0-11 -> (pitch class, minor)"""
    if isinstance(key, str) and key.rstrip('m') in NOTES:
        return NOTES[key.rstrip('m')], key.endswith('m')
    if not isinstance(key, str) and float(key) == int(key) and 0 <= int(key) < 12:
        return int(key), False
    raise ValueError(f'unknown key {key!r}; valid: {", ".join(NOTES)} (add m for minor) or 0-11')


def _settings(scene, controls):
    if scene not in SCENES:
        raise ValueError(f'unknown scene {scene!r}; valid: {", ".join(SCENES)}')
    spec = SCENES[scene]
    choices = spec.get('choices', {})
    c = {k: choices[k][0] if k in choices else None if k == 'howl' else (0, False) if k == 'key' else 0.0
         for k in spec['controls']}
    for k, v in controls.items():
        if k not in c:
            raise ValueError(f'unknown control {k!r} for {scene}; valid: {", ".join(spec["controls"])}')
        if k in choices:
            if v not in choices[k]:
                raise ValueError(f'unknown {k} {v!r} for {scene}; valid: {", ".join(choices[k])}')
            c[k] = v
        elif k == 'key':
            c[k] = _key(v)
        else:
            c[k] = float(np.clip(v, 0.0 if k == 'howl' else -1.0, 1.0))
    return c


def _cut(y):
    """where to open a circular bed: the 50 ms step at which the level over the 0.4 s before and after differs least,
    preferring quiet places (a wave bed opens in a lull, a fire between pops)"""
    hop = SR // 20
    k = len(y) // hop
    if k < 24:
        return 0
    p = np.tile(np.mean(np.sum(y[:k * hop] ** 2, axis=1).reshape(k, hop), axis=1), 3) + 1e-20
    cs = np.concatenate([[0.0], np.cumsum(p)])
    j = k + np.arange(k)
    before, after = cs[j] - cs[j - 8], cs[j + 8] - cs[j]
    cost = np.abs(10 * np.log10(before / after)) + 1.5 * np.log10((before + after) / (before + after).min())
    return int(np.argmin(cost)) * hop


def _render(scene, dur, r, controls, as_loop):
    c = _settings(scene, controls)
    r = dsp.rng(0) if r is None else r
    n = dsp.n_of(dur)
    if n < SR // 2:
        raise ValueError('a texture needs at least 0.5 s')
    y = _BUILD[scene](n, r, c)
    f = np.fft.rfftfreq(n, 1 / SR)
    tilt = 2 ** (c['bright'] * 1.5 / 6.0206 * np.log2(np.clip(f, 40.0, 16000.0) / 1000.0))    # +-1.5 dB per octave
    F = np.fft.rfft(y, axis=0) * (tilt * _hp(22.0)(f))[:, None]
    F[0] = 0.0
    w = 1 + c['width'] if c['width'] <= 0 else 1 + 0.6 * c['width']
    mid, side = 0.5 * (F[:, 0] + F[:, 1]), 0.5 * w * (F[:, 0] - F[:, 1])
    y = np.fft.irfft(np.stack([mid + side, mid - side], axis=1), n, axis=0)
    if not SCENES[scene]['loop']:
        return dsp.fade(y * (EVENT_PEAK / (np.abs(y).max() + 1e-12)), 0.005, 0.2 * dur)
    y = np.roll(y, -_cut(y), axis=0)
    y *= BED_PEAK / (np.abs(y).max() + 1e-12)
    return y if as_loop else dsp.fade(y, min(0.1, dur / 10), min(0.3, dur / 6))


def render(scene, dur, r=None, **controls):
    """dur seconds of a scene as stereo float64 (n, 2). scene: a key of SCENES; r: numpy Generator (dsp.rng(seed);
    the same arguments give the same samples); controls: those listed in SCENES[scene]['controls'], numbers in
    [-1, 1] with 0 = the default (intensity, size, distance: -1 near .. 1 far, bright: -1.5 .. +1.5 dB per octave,
    width: -1 mono .. 1 wide, warmth, density), howl 0..1, owl (night) and dove (forest): off at 0 or less, more
    calls up to 1, and the named choices in SCENES[scene]['choices']
    (surface, place) or key ('C', 'F#m', ...). A bed peaks at a quarter of -1 dBFS and fades in and out; thunder is
    an event (peak -3 dBFS, its tail faded). Unknown scenes, controls or choices raise ValueError with the valid ones.
        y = texture.render('wind', 20.0, dsp.rng(2), place='window', howl=0.8, intensity=0.5)"""
    return _render(scene, dur, r, controls, False)


def loop(scene, dur, r=None, **controls):
    """the same bed as render(), without the fades: exactly dur seconds that repeat without a seam (every event,
    gust, wave and bird phrase is scheduled around the circle; nothing is cut and nothing is crossfaded) and are
    opened at their steadiest, quietest point. Loops shorter than SCENES[scene]['min_loop'] s work but repeat audibly.
        bed = texture.loop('rain', 16.0, dsp.rng(5), surface='leaves'); texture.seam_db(bed)"""
    if scene in SCENES and not SCENES[scene]['loop']:
        raise ValueError(f'{scene} is an event, not a bed: use render')
    return _render(scene, dur, r, controls, True)


def seam_db(buf, window=0.4):
    """how audible the loop point of a buffer is when it repeats, in dB: 'level' = the step in RMS level from the
    last `window` s to the first, 'spectral' = the mean absolute difference of their octave-band levels (bands within
    30 dB of the loudest), 'click' = the sample step across the seam against the 99.9th percentile of the sample steps
    inside (0 or less: the seam is no rougher than the signal itself). Two neighbouring stretches of any live texture
    differ a little, so compare with a cut at another place: seam_db(np.roll(buf, k, axis=0))."""
    x = np.asarray(buf, np.float64)
    x = x[:, None] if x.ndim == 1 else x
    w = min(dsp.n_of(window), len(x) // 2)
    a, b = x[-w:], x[:w]
    f = np.fft.rfftfreq(w, 1 / SR)
    band = np.clip(np.floor(np.log2(np.maximum(f, 1.0) / 44.2)).astype(np.int64), -1, 9) + 1
    pa, pb = [np.bincount(band, np.sum(np.abs(np.fft.rfft(s * np.hanning(w)[:, None], axis=0)) ** 2, axis=1), 11)[1:]
              + 1e-30 for s in (a, b)]
    keep = np.maximum(pa, pb) > 1e-3 * max(pa.max(), pb.max())
    steps = np.abs(np.diff(x, axis=0)).max(axis=1)
    return {'level': float(20 * np.log10(_rms(b) / _rms(a))),
            'spectral': float(np.mean(np.abs(10 * np.log10(pb[keep] / pa[keep])))),
            'click': float(20 * np.log10((np.abs(x[0] - x[-1]).max() + 1e-12) / (np.percentile(steps, 99.9) + 1e-12)))}
