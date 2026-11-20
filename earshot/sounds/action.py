"""Physical action sounds for games and animation: jumps, landings, footsteps, swings, hits, explosions, shots,
doors, breaking things, water, magic and the small signals of a race start. The modern versions (the chip ones are
in retro.py): what a polished indie or mobile game or an animated film plays.

Each sound is layered the way a sound designer builds it, every layer from the physics of its part:
    transient   the contact itself: a force pulse a fraction of a millisecond to a few ms long (Hertz), a blast wave
                (Friedlander), the click of a latch
    body        what rings or moves: modes of a plate, a board, a box (modal synthesis, materials by their damping),
                a pitch-dropping low thump for weight, Aeolian tones at St v / d for anything swung through air
    events      clouds of small events with power-law sizes and decaying arrival rates: grit, gravel, snow, shards,
                splinters, droplets, debris, sparkle (earshot.synth.texture)
    tail        one shared room (earshot.synth.space), terrain echoes for blasts, Doppler for things flying past

    from earshot.sounds import registry, action      # action registers its designs on import
    y, meta = registry.render('hit/punch', seed=3, energy=0.5, size=0.2)
    y, meta = registry.render('step/gravel', seed=11)        # every seed is another footstep

The shared axes mean the same everywhere: energy = how hard (louder, brighter, faster), size = how big (lower,
longer, more low layer), bright = more top, hard = a faster and louder onset, ring = a longer tail, space = more room,
width = wider. Tonal ones (countdown/beep, countdown/go, ui/error_buzz) read pitch; the loops (laser/beam,
rumble/quake) read dur. Designs that scatter events need earshot.synth.texture; heal/glow and magic/cast also use a
pad of earshot.synth.patches; both are imported inside the functions that use them.
"""
from types import SimpleNamespace

import numpy as np
from scipy import fft as sfft
from scipy import signal

from ..forge import dsp, fx
from ..synth import space
from .registry import design

SR = dsp.SR
AXES = ('energy', 'size', 'bright', 'hard', 'ring', 'space', 'width')
DOM = ('game', 'motion')
PENTA = (0, 2, 4, 7, 9)


# ------------------------------------------------------------------------------------------------ controls
def _k(c):
    """the shared axes as factors: f = frequency factor of size (bigger is lower), L = length factor of size, lo = dB
    of the low layer (size), h = dB of the onset layer (hard), tc = contact-time factor (hard, energy: a harder or
    faster strike is shorter), ring = tail length factor, e = energy in -1..1"""
    e = c.energy
    return SimpleNamespace(e=e, f=c.lerp('size', 1.45, 0.65, 1.0, log=True),
                           L=c.lerp('size', 0.75, 1.45, 1.0, log=True), lo=6.0 * c.size, h=6.0 * c.hard + 2.0 * e,
                           tc=c.lerp('hard', 1.9, 0.5, 1.0, log=True) * 0.85 ** e,
                           ring=c.lerp('ring', 0.5, 2.0, 1.0, log=True))


def _out(c, y, ev=None, room='room', spread=0.3, tilt=0.0):
    """mono layers y (and stereo event layers ev) to the finished stereo take: energy as level, width, a high shelf
    for bright (and energy: harder is brighter), the shared room"""
    w = c.lerp('width', 0.0, 2.0, 1.0)
    st = np.zeros((0, 2))
    if y is not None:
        st = fx.decorrelate(y, min(1.0, spread * w)) if np.ndim(y) == 1 else fx.width(y, w)
    if ev is not None:
        st = _sum(st, fx.width(ev, w))
    st = _shelf(st, 6.0 * c.bright + 2.0 * c.energy + tilt) * c.db('energy', 6.0)
    return space.room(st, c.room(room), amount=c.space)


def _shelf(y, db, fc=2500.0):
    """first-order high shelf: low pass + G * high pass of the same corner sum to a flat line when G = 1"""
    return y + (10 ** (db / 20) - 1) * dsp.butter(y, 'high', fc, order=1) if db else y


def _sum(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.ndim != b.ndim:
        a, b = (dsp.pan_to_stereo(a) if a.ndim == 1 else a), (dsp.pan_to_stereo(b) if b.ndim == 1 else b)
    n = max(len(a), len(b))
    out = np.zeros((n,) + a.shape[1:])
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def _mix(layers, seconds=None):
    """layers (x, t s, gain dB) of mono signals into one; every layer arrives with its own peak at 1. With a length,
    whatever runs past it is faded out over the last 20 ms, never cut"""
    y = dsp.stack([(x, t, g) for x, t, g in layers], length=seconds)
    return dsp.fade(y, 0.0, 0.02) if seconds else y


def _vary(r, x, s):
    """x scaled by a log-normal take-to-take variation of s (a fraction)"""
    return x * float(np.exp(s * r.standard_normal()))


# ------------------------------------------------------------------------------------------------ sources
def _t(n):
    return np.arange(n) / SR


def _env(n, attack, tau, hold=0.0):
    """a raised-cosine attack, an optional hold, an exponential decay with time constant tau s"""
    t = _t(n)
    e = np.exp(-np.maximum(t - attack - hold, 0.0) / max(tau, 1e-4))
    a = min(n, dsp.n_of(attack))
    if a > 1:
        e[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    return e


def _fast(n):
    return sfft.next_fast_len(n, real=True)


def _noise(r, seconds, color='pink'):
    """pink (1/f) or brown (1/f^2) noise of unit RMS, as dsp.noise, drawn at a fast FFT length and cut"""
    n = dsp.n_of(seconds)
    m = _fast(n)
    F = sfft.rfft(r.standard_normal(m))
    f = sfft.rfftfreq(m, 1 / SR)
    f[0] = f[1]
    w = sfft.irfft(F / f ** (0.5 if color == 'pink' else 1.0), m)[:n]
    return w / (np.sqrt(np.mean(w ** 2)) + 1e-12)


def _air(seconds, r, **kw):
    """dsp.air (moving-air turbulence) drawn at a fast FFT length and cut to `seconds`"""
    n = dsp.n_of(seconds)
    return dsp.air(_fast(n) / SR, r, **kw)[:n]


def _norm(x):
    return x / (np.abs(x).max() + 1e-12)


def _burst(r, seconds, lo, hi, tau, attack=0.0003, order=2):
    """band-passed noise with a fast attack and an exponential decay, peak 1: a crack, a scuff, a splash. The decay
    moves +-15 % from take to take, as no two blows are alike"""
    n = dsp.n_of(seconds)
    tau = _vary(r, tau, 0.15)
    x = dsp.butter(r.standard_normal(n), 'band', (lo, min(hi, 0.45 * SR)), order) * _env(n, attack, tau)
    return dsp.fade(_norm(x), 0.0, min(0.01, 0.3 * seconds))


def _thump(r, seconds, f0, f1, drop, tau, attack=0.002, harm=0.25):
    """the weight of an impact: a sine whose pitch falls from f0 to f1 Hz (time constant drop s: a membrane or a
    floor relaxing after the blow) with its second harmonic at `harm`, so it still reads on small speakers. Peak 1"""
    n = dsp.n_of(seconds)
    t = _t(n)
    tau, f0 = _vary(r, tau, 0.15), _vary(r, f0, 0.05)              # every blow its own
    f = f1 + (f0 - f1) * np.exp(-t / max(drop, 1e-4))
    ph = 2 * np.pi * (np.cumsum(f) - f[0]) / SR
    y = (np.sin(ph) + harm * np.sin(2 * ph)) * _env(n, attack, tau)
    return dsp.fade(_norm(y), 0.0, min(0.03, 0.3 * seconds))


def _dull(r, seconds, f, tau, attack=0.002, q=1.1):
    """a dull thud: noise through a broad resonance at f Hz (soft ground, a padded body answering a blow); with no
    steady pitch it is never heard as a tone the way a sine thump can be. Peak 1"""
    n = dsp.n_of(seconds)
    tau = _vary(r, tau, 0.15)
    x = dsp.svf(r.standard_normal(n), f, q, 'bandn') * _env(n, attack, tau)
    x = dsp.butter(x, 'low', 3 * f, order=2)
    return dsp.fade(_norm(x), 0.0, min(0.03, 0.3 * seconds))


def _heel(r, k, attack=0.002, tau=0.02, top=1100.0):
    """a heel or a boot landing: 30-60 ms of broadband low noise (the shoe, the foot and the floor under it), weighted
    toward 80-250 Hz but with no pitch of its own. Peak 1"""
    n = dsp.n_of(0.12)
    x = r.standard_normal(n) * _env(n, attack, _vary(r, tau, 0.2))
    x = dsp.butter(x, 'low', top * k.f ** 0.5, order=2)
    x = dsp.peq(x, 130 * k.f, 6.0, 0.7)
    return dsp.fade(_norm(x), 0.0, 0.03)


def _knock(r, lo, hi, tau):
    """a small piece landing: a noise knock in lo..hi Hz with a broad low resonance, a few ms long. Peak 1"""
    x = _burst(r, 8 * tau, lo, hi, tau, attack=0.001) + 0.6 * _dull(r, 8 * tau, 1.5 * lo, 1.5 * tau, 0.001)
    return _norm(x)


def _strike(r, freqs, taus, amps, seconds, contact=0.001, grit=0.4):
    """modes (Hz, time constants s, amplitudes) driven by a contact force pulse `contact` s long: shorter is
    brighter, as a harder mallet or a faster blow is (the pulse passes modes below about 1 / contact). Peak 1"""
    y = dsp.modal(freqs, taus, amps, seconds, r, jitter=0.004)
    p = dsp.contact(max(contact, 4.0 / SR), r, grit)
    y = dsp.excite(y, p / np.abs(p).sum())
    return dsp.fade(_norm(y), 0.0, min(0.03, 0.3 * seconds))


def _cloud(r, lo, hi, count, d_lo, d_hi, seconds, contact, tilt=-3.0, grit=0.3, rough=0.4):
    """a dense body (a board, a box, a door, a lid): count modes log-uniform in lo..hi Hz, decay time constants
    d_lo..d_hi s (wood: t60 20-150 ms, so tau 3-20 ms), struck for `contact` s; `rough` mixes in the band's noise
    decaying with the body (the many modes no listener resolves, the rattle of joints), so it never rings as notes"""
    f = np.sort(_logu(r, lo, hi, int(count)))
    u = np.log(f / lo) / np.log(hi / lo)
    d = np.exp(np.log(d_lo) + u * np.log(d_hi / d_lo))
    a = 10 ** (tilt * np.log2(f / lo) / 20) * r.uniform(0.3, 1.0, len(f))
    y = _strike(r, f, d, a, seconds, contact, grit)
    if rough:
        y = y + rough * _burst(r, seconds, lo, hi, d_lo * 0.6, attack=contact * 0.5)
    return _norm(y)


def _blast(r, tp, seconds, b=1.6):
    """a blast wave (Friedlander): p(t) = (1 - t / t+) exp(-b t / t+), positive phase t+ s (1-10 ms; it grows with
    the cube root of the charge), with turbulent grain on the negative phase. Peak 1"""
    n = dsp.n_of(seconds)
    t = _t(n)
    p = (1 - t / tp) * np.exp(-b * t / tp)
    p *= 1 + 0.35 * dsp.butter(r.standard_normal(n), 'low', 3000.0 / max(tp * 1000, 1.0)) * (t > tp)
    a = dsp.n_of(0.00015)
    p[:a] *= np.linspace(0, 1, a)
    return dsp.fade(_norm(dsp.butter(p, 'high', 25.0)), 0.0, 0.2 * seconds)


def _aeolian(r, seconds, peak, fall, v_peak, d, r_in=0.3, r_out=1.0, sections=8, wake=0.35):
    """the swish of a swung object (Selfridge 2018): its length split into compact sources at radii r_i, each
    shedding vortices at f = St v / d (St 0.2) with amplitude v^3 (intensity v^6, Curle), the tone narrow while
    the flow is laminar and wide when it is turbulent (Q from the Reynolds number v d / nu), plus its drag tone at
    2 f and a broad wake. The angular speed rises to v_peak at the tip in `peak` s and falls over about `fall` s.
    Mono, peak 1"""
    n = dsp.n_of(seconds)
    t = _t(n)
    w = np.where(t < peak, np.sin(0.5 * np.pi * np.minimum(t / peak, 1.0)) ** 2, np.exp(-((t - peak) / fall) ** 2))
    y = np.zeros(n)
    for i in range(sections):
        ri = r_in + (r_out - r_in) * (i + 0.5) / sections
        v = w * v_peak * ri / r_out * (1 + 0.03 * r.standard_normal())
        f = np.maximum(0.2 * v / d, 25.0)
        q = np.clip(40.0 * (np.maximum(v, 0.5) * d / 1.5e-5 / 1000.0) ** -0.5, 2.5, 40.0)
        x = r.standard_normal(n)
        y += (v / v_peak) ** 3 * (dsp.svf(x, f, q, 'bandn') * np.sqrt(q) + 0.18 * dsp.svf(x, 2 * f, q, 'bandn'))
    tip = np.maximum(0.2 * w * v_peak / d, 25.0)
    y += wake * w ** 3 * dsp.svf(r.standard_normal(n), tip, 0.8, 'bandn') * 2.0
    return dsp.fade(_norm(y), 0.002, 0.2 * seconds)


def _creak(r, seconds, rate, formants, jitter=0.25, skip=0.1, stuck=0.04, grain=0.0015):
    """stick-slip friction (a hinge, a lid, cold snow): slips at a rate that follows the drive (`rate`: Hz per
    sample; 5-20 ms periods are a creak, 1-3 ms a squeak) but never steadily: each period jitters, a slip is now
    and then missed (skip), the surfaces now and then hold for 20-60 ms (stuck), and every slip is a short burst of
    noise of its own size, not an impulse. The bursts excite the body's broad resonances (formants: (Hz, q, gain)).
    A periodic train of identical clicks through sharp modes would be a buzzing tone; this is a rough, wandering
    creak (Farnell, creaking door). Mono, peak 1"""
    n = dsp.n_of(seconds)
    exc = np.zeros(n)
    top = rate.max() + 1e-9
    i = 0
    while i < n:
        f = rate[i]
        if f <= 1.0:
            i += dsp.n_of(0.004)
            continue
        if r.uniform() < stuck:
            i += dsp.n_of(r.uniform(0.02, 0.06))
            continue
        i += max(2, int(SR / f * (1 + jitter * r.standard_normal())))
        if i >= n or r.uniform() < skip:
            continue
        L = max(4, dsp.n_of(grain * r.uniform(0.6, 1.6)))
        size = np.exp(np.clip(0.4 * r.standard_normal(), -0.8, 0.5))
        seg = r.standard_normal(L) * np.hanning(L) * (f / top) ** 0.5 * size
        exc[i:i + L] += seg[:n - i]
    y = sum(g * dsp.svf(exc, fc, q, 'bandn') for fc, q, g in formants)
    return dsp.fade(_norm(dsp.butter(y, 'low', 5000.0)), 0.002, 0.04)     # a lone slip, unfiltered, is a click


def _disperse(x, a=-0.88, stages=200):
    """x through a chain of first-order all-passes: low frequencies are delayed most (a < 0), so an impulse comes
    out as a falling chirp: a stretched spring, a struck guy wire, a blaster (Valimaki, Parker and Abel 2010)"""
    y = np.asarray(x, np.float64)
    for _ in range(stages):
        y = signal.lfilter([a, 1.0], [1.0, a], y)
    return y


def _tex():
    from ..synth import texture        # lazy: only the designs that scatter events need it
    return texture


def _grains(r, n, t, gain, f, q, length, sharp=1.0, spread=0.6):
    """band-passed noise events into a stereo buffer: grit, gravel, snow, splinters, crumbs. Each event is its own
    fresh noise, `length` s long and at least 1 ms (sharp 1: full after a 0.5 ms ramp, then falling: an impact; 0: a
    swell, a rustle), through a band-pass resonance at f Hz (at most 12 kHz) with quality q; gain in energy units
    (1 = the energy of 1 ms of unit noise), pans random within +-spread; the bus is low-passed at 8 kHz. Give
    events spread pitches: two loud ones ringing at one pitch a few ms apart make a comb filter"""
    out = np.zeros((n, 2))
    t, gain, f, q, length = np.broadcast_arrays(*[np.atleast_1d(np.asarray(a, np.float64)) for a in
                                                  (t, gain, f, q, length)])
    for t0, g, fc, qq, ln in zip(t, gain, f, q, length):
        i = dsp.n_of(t0) if t0 > 0 else 0
        L = max(48, int(ln * SR))                       # at least 1 ms: shorter is a bare click
        if i >= n:
            continue
        fc = min(fc, 12000.0)
        rho = np.exp(-np.pi * fc / qq / SR)
        m = min(n - i, L + int(6.2 * SR * qq / (np.pi * fc)) + 4)
        u = np.linspace(0, 1, L)
        w = sharp * (1 - u) ** 2 + (1 - sharp) * np.sin(np.pi * u)
        a = min(L // 2, 24)
        w[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
        x = np.zeros(m)
        k = min(L, m)
        x[:k] = (r.standard_normal(L) * w)[:k]
        y = signal.lfilter([1, 0, -1], [1, -2 * rho * np.cos(2 * np.pi * fc / SR), rho * rho], x)    # a band-pass
        e = m - m // 4
        y[e:] *= np.linspace(1, 0, m - e)
        y *= np.sqrt(g * 0.001 * SR / (np.sum(y * y) + 1e-12))
        th = (np.clip(r.uniform(-spread, spread), -1, 1) + 1) * np.pi / 4
        out[i:i + m, 0] += y * np.cos(th)
        out[i:i + m, 1] += y * np.sin(th)
    return dsp.butter(out, 'low', 8000.0)        # a lone event wider than this reads as a click, not a grain


def _pings(r, n, t, gain, f, decay, attack=0.0002, glide=0.0, spread=0.6, pan=None, wrap=False):
    """damped sinusoids into a stereo buffer (texture.rings): shards, droplets, sparkle, small pieces"""
    out = np.zeros((n, 2))
    if np.size(t):
        pan = r.uniform(-spread, spread, np.size(t)) if pan is None else pan
        _tex().rings(out, t, gain, pan, f, decay, attack, glide, wrap=wrap)
    return out


def _logu(r, lo, hi, k):
    """k values log-uniform in lo..hi (frequencies of events with no preferred pitch)"""
    return np.exp(r.uniform(np.log(lo), np.log(hi), k))


def _times(r, lam, t0, t1, peak):
    """Poisson arrivals in [t0, t1) at rate lam(t) events/s (at most `peak`), by thinning"""
    k = r.poisson(max(peak * (t1 - t0), 0.0))
    t = np.sort(r.uniform(t0, t1, k))
    return t[r.uniform(0, peak, k) < lam(t)]


def _power(r, k, lo, hi, shape=2.0):
    """sizes with density ~ size^-shape in [lo, hi]: many small, few large (shards, gravel, debris)"""
    u = r.uniform(size=k)
    a = 1 - shape
    return (lo ** a + u * (hi ** a - lo ** a)) ** (1 / a)


def _sparkle(r, n, t, root, lo, hi, level=1.0, decay=(0.06, 0.25), glide=0.02, spread=0.8, rise=None):
    """sine grains on a pentatonic scale between lo and hi Hz, each a small struck bar (its 2.76 f mode 14 dB
    down), log-normal levels (sigma 6 dB), a 1-5 ms attack, a slight upward glide. rise: 0..1 per grain picks the
    note by its position in the range (a rising run), else at random"""
    scale = np.array([root * 2 ** ((s + 12 * o) / 12) for o in range(-3, 6) for s in PENTA])
    scale = scale[(scale >= lo) & (scale <= hi)]
    k = len(t)
    u = r.uniform(size=k) if rise is None else np.clip(rise + 0.12 * r.standard_normal(k), 0, 0.999)
    f = scale[(u * len(scale)).astype(int)] * 2 ** (4 * r.standard_normal(k) / 1200)
    a = level * np.exp(0.69 * r.standard_normal(k)) * (2500.0 / f) ** 0.3
    dec = r.uniform(*decay, k) * np.sqrt(2500.0 / f)
    att = r.uniform(0.001, 0.005, k)
    pan = r.uniform(-spread, spread, k)
    y = _pings(r, n, t, a, f, dec, att, glide / dec, pan=pan)
    up = 2.756 * f < 0.42 * SR
    return y + _pings(r, n, t[up], 0.2 * a[up], 2.756 * f[up], 0.35 * dec[up], 0.001, pan=pan[up])


# ------------------------------------------------------------------------------------------------ jumps and landings
@design('jump/soft', kind='whoosh', says=('a short rising breathy whistle with a soft thump',
                                         'a light airy upward puff', 'a quick soft rising swish of air'),
        uses=('jump', 'hop', 'double jump', 'bounce'), domains=DOM, dur=0.3, axes=AXES,
        allows=('rising_glide',))     # the breath band climbs as the body leaves the ground: the gesture itself
def jump_soft(c, r):
    """a cartoon hop: the push-off (a soft low thump and a cloth flick) and the body leaving: breath through a
    narrow band whose centre climbs fast and settles (ease-out), so it reads as up without a siren's sweep"""
    k = _k(c)
    T = 0.3 * k.L
    n = dsp.n_of(T)
    t = _t(n)
    f0, f1 = 480 * k.f, _vary(r, 1450 * k.f, 0.1)
    fc = f0 * (f1 / f0) ** (1 - np.exp(-t / (0.055 * k.L)))
    breath = _air(T, r, center=1100 * k.f, flutter=0.3, rate=18, bright=0.3)
    y = dsp.svf(breath, fc, 4.5, 'bandn') + 0.5 * dsp.svf(breath, 0.55 * fc, 1.2, 'bandn')
    y = dsp.fade(_norm(y * _env(n, 0.03, 0.075 * k.ring)), 0.0, 0.05)
    thump = _thump(r, 0.12 * k.L, 170 * k.f, 95 * k.f, 0.02, 0.028, harm=0.15)
    cloth = _burst(r, 0.05, 2500, 9000, 0.01)
    return _out(c, _mix([(y, 0.01, 0), (thump, 0, -5 + k.lo), (cloth, 0, -17 + k.h)]), room='room', spread=0.35)


@design('jump/spring', kind='zap', says=('a springy boing with a wobbling twang', 'a cartoon spring bounce',
                                        'a coiled metal spring twanging'),
        uses=('jump', 'bounce', 'spring', 'trampoline', 'boing'), domains=DOM, dur=0.6, axes=AXES,
        allows=('ring',))      # the coil's own wobbling harmonic tone: the comb detector reads any such tone as one
def jump_spring(c, r):
    """a coil spring released: its transverse modes are a stiff string's (f_n = n f0 sqrt(1 + B n^2), stretched
    upward) and the coil sways at a few Hz, wobbling pitch and level as the sway dies; the wave travelling along
    the coil is dispersive, so the twang starts as a falling chirp. Under it the push-off thump"""
    k = _k(c)
    T = 0.62 * k.L * k.ring ** 0.5
    n = dsp.n_of(T)
    t = _t(n)
    f0 = _vary(r, 205 * k.f, 0.05)
    sway = _vary(r, 8.0, 0.1) / k.L ** 0.5
    depth = 0.075 * np.exp(-t / (0.22 * k.ring))
    wob = np.sin(2 * np.pi * sway * t + r.uniform(-0.3, 0.3))
    y = np.zeros(n)
    B = 0.012
    for m in range(1, 7):
        fm = f0 * m * np.sqrt(1 + B * m * m) * (1 + depth * wob) * (1 + 0.06 * np.exp(-t / 0.02))
        if fm.max() > 0.4 * SR:
            break
        ph = 2 * np.pi * (np.cumsum(fm) - fm[0]) / SR
        y += m ** -0.9 * np.sin(ph) * np.exp(-t / (0.32 * k.ring * m ** -0.7))
    y *= (1 + 0.45 * np.exp(-t / (0.22 * k.ring)) * np.sin(2 * np.pi * sway * t + 1.1)) * _env(n, 0.002, 10.0)
    y = dsp.fade(_norm(y), 0.0, 0.08)
    imp = np.zeros(dsp.n_of(0.12))
    imp[8] = 1.0
    chirp = dsp.butter(_disperse(imp, -0.86, 120), 'band', (250 * k.f, 6000))
    chirp = dsp.fade(_norm(chirp * _env(len(chirp), 0.0, 0.04)), 0.0, 0.02)
    thump = _thump(r, 0.12 * k.L, 150 * k.f, 80 * k.f, 0.02, 0.03, harm=0.2)
    clank = _strike(r, f0 * np.array([11.3, 17.2, 23.9]), [0.01, 0.007, 0.005], [1, 0.6, 0.4], 0.05, 0.0002)
    return _out(c, _mix([(y, 0.004, 0), (chirp, 0.0, -9), (thump, 0, -6 + k.lo), (clank, 0.002, -22 + k.h)]),
                room='room', spread=0.3)


@design('jump/heavy', kind='whoosh',
        says=('a heavy whoosh with a low thump underneath', 'a big body launching into the air',
              'a deep effortful swoosh'),
        uses=('jump', 'leap', 'heavy jump', 'launch'), domains=DOM, dur=0.45, axes=AXES)
def jump_heavy(c, r):
    """a big character pushes off: a low push (a sub thump and the scrape of feet on the ground), then the effort
    of the body moving, an Aeolian whoosh of a thick body (d 0.25 m, low and broad)"""
    k = _k(c)
    T = 0.5 * k.L
    swish = _aeolian(r, T, 0.11 * k.L, 0.16 * k.L * k.ring, _vary(r, 9.0, 0.1) * (1 + 0.15 * k.e), 0.25 / k.f,
                     r_in=0.4, sections=5, wake=0.6)
    push = _thump(r, 0.3 * k.L, 105 * k.f, 48 * k.f, 0.05, 0.08 * k.L, attack=0.004, harm=0.35)
    scrape = _burst(r, 0.12, 300 * k.f, 2500, 0.03, attack=0.006)
    cloth = _burst(r, 0.16, 1800, 8000, 0.05, attack=0.02)
    return _out(c, _mix([(push, 0, -2 + k.lo), (scrape, 0.002, -10 + k.h), (swish, 0.02, -1), (cloth, 0.03, -16)]),
                room='room', spread=0.35)


@design('land/soft', kind='thud',
        says=('a soft low thump of feet landing on the ground', 'a light dull thud with a little grit',
              'a muffled footfall'),
        uses=('land', 'landing', 'touch down', 'drop down'), domains=DOM, dur=0.25, axes=AXES)
def land_soft(c, r):
    """two feet landing on packed earth 10-35 ms apart: each a broadband low thud of the sole, the leg and the
    ground (no pitch: a sine here is a kick drum), the earth crunching under it (PhISEM grains at a rate that
    follows the rise of the force, Cook 2002), the soles sliding a little as the weight settles (a rough scuff),
    and the clothes catching up with the body"""
    k = _k(c)
    T = 0.3 * k.L
    n = dsp.n_of(T)
    t = _t(n)
    gap = r.uniform(0.01, 0.035)
    layers, F = [], np.zeros(n)
    for t0, g in ((0.0, 0.0), (gap, r.uniform(-5.0, -2.0))):
        rise = _vary(r, 0.014, 0.2) * k.tc
        u = t - t0
        F += 10 ** (g / 20) * np.where(u < 0, 0.0, np.where(u < rise, np.sin(0.5 * np.pi * u / rise) ** 2,
                                                             np.exp(-((u - rise) / (0.05 * k.L)) ** 2)))
        layers += [(_heel(r, k, 0.003 * k.tc, 0.03 * k.L, 1300), t0, g + k.lo / 2),
                   (_dull(r, 0.15, 95 * k.f, 0.03 * k.L), t0, g - 7 + k.lo)]
    dF = np.maximum(np.gradient(F) * SR, 0.0)
    tg = np.concatenate([_rate_times(r, n, dF, 3000.0, 900.0 / (dF.max() + 1e-9)),
                         _rate_times(r, n, F, 600.0, 250.0 / F.max())])
    m = len(tg)
    crunch = _grains(r, n, tg, _power(r, m, 0.03, 1.0, 1.6), _logu(r, 600, 4500, m) * k.f ** 0.5,
                     r.uniform(1.0, 3.0, m), r.uniform(0.001, 0.003, m), sharp=0.9)
    ts = gap + r.uniform(0.01, 0.03)
    scuff = _burst(r, 0.12, 500 * k.f, 5000, 0.035, attack=0.012) * (1 + 0.6 * dsp.wander(0.12, r, 60.0))
    cloth = _burst(r, 0.1, 1500, 7000, 0.025, attack=0.006)
    y = _mix(layers + [(scuff, ts, -15 + k.h / 2), (cloth, 0.012, -20)], T)
    y = fx.saturate(y, 3.0, 'tanh', level='rms')                                # the rare noise peak tamed
    return _out(c, y, _norm(crunch) * 10 ** ((-9 + k.h) / 20) * np.abs(y).max(), room='room', spread=0.25)


@design('land/heavy', kind='thud', says=('a heavy body thudding onto the ground with a deep boom',
                                        'a deep thud with dust and a short rattle', 'a big heavy landing impact'),
        uses=('land', 'heavy landing', 'stomp', 'ground pound'), domains=DOM, dur=0.7, axes=AXES)
def land_heavy(c, r):
    """a heavy body lands: a sub thud (the ground and the body, pitch falling as the floor relaxes), the knock of
    the contact through the floor's modes, dust (low-passed noise swelling a little after), then a short rattle of
    loose bits bouncing (each bounce e = 0.5 of the last, shorter and softer)"""
    k = _k(c)
    T = 0.8 * k.L * k.ring ** 0.5
    n = dsp.n_of(T)
    sub = _thump(r, T, 85 * k.f, 38 * k.f, 0.06, 0.16 * k.L * k.ring, attack=0.004, harm=0.35)
    knock = _cloud(r, 110 * k.f, 1400 * k.f, 24, 0.025, 0.006, 0.25, 0.003 * k.tc)
    crack = _burst(r, 0.04, 600, 5000, 0.006 * k.tc, attack=0.0005)
    dust = dsp.fade(_norm(dsp.butter(_noise(r, T, 'pink'), 'band', (180, 2400 * k.f)) *
                          _env(n, 0.03, 0.14 * k.ring)), 0.0, 0.1)
    y = _mix([(sub, 0, 0 + k.lo), (knock, 0, -5), (crack, 0, -12 + k.h), (dust, 0.012, -15)], T)
    # the rattle: bits bouncing, e = 0.5 .. 0.65
    tb, v, gap = [], [], r.uniform(0.07, 0.11)
    tt, vv = 0.035, 1.0
    while gap > 0.006 and tt < T - 0.05:
        tb.append(tt)
        v.append(vv)
        e = r.uniform(0.5, 0.65)
        tt, vv, gap = tt + gap, vv * e, gap * e
    tb = np.repeat(tb, 3) + r.uniform(0, 0.012, 3 * len(tb))
    v = np.repeat(v, 3) * r.uniform(0.3, 1.0, len(tb))
    ev = _grains(r, n, tb, v ** 2 * 0.6, _logu(r, 700, 4000, len(tb)) * k.f ** 0.5,
                 r.uniform(2, 6, len(tb)), r.uniform(0.001, 0.003, len(tb)))
    return _out(c, y, _norm(ev) * 10 ** (-17 / 20), room='room', spread=0.3)


# ------------------------------------------------------------------------------------------------ footsteps
def _grf(r, n, heel=0.004, toe=None):
    """the ground reaction force of one step (biomechanics: the heel peak, then the ball of the foot 50-150 ms later,
    a little weaker) and its rise rate, which drives granular surfaces; each step its own"""
    t = _t(n)
    toe = r.uniform(0.05, 0.11) if toe is None else toe
    hv, tv = _vary(r, 1.0, 0.08), r.uniform(0.6, 0.85)

    def bump(t0, rise, fall):
        u = t - t0
        up = np.sin(0.5 * np.pi * np.clip(u / rise, 0, 1)) ** 2
        return np.where(u < 0, 0.0, np.where(u < rise, up, np.exp(-((u - rise) / fall) ** 2)))
    F = hv * bump(heel, 0.012, 0.04) + tv * bump(heel + toe, 0.02, 0.05)
    return F, np.maximum(np.gradient(F) * SR, 0.0), heel, heel + toe, hv, tv


def _rate_times(r, n, drive, peak, scale):
    """event times at a rate drive * scale (events/s), by thinning: granular surfaces crunch as the force rises"""
    lam = lambda tt: np.interp(tt, _t(n), drive) * scale
    return _times(r, lam, 0.0, n / SR, peak)


def _step(c, r, surface):
    """one footstep: heel then the ball of the foot (_grf), each through the surface's own sound"""
    k = _k(c)
    T = 0.3 * k.L ** 0.5
    n = dsp.n_of(T)
    F, dF, th, tt, hv, tv = _grf(r, n)
    ev = None
    room = 'room'
    if surface == 'wood':
        layers = []
        for t0, v in ((th, hv), (tt, tv)):
            board = _cloud(r, 85 * k.f, 520 * k.f, 16, 0.022 * k.ring, 0.01, 0.15, 0.0022 * k.tc)
            knock = _cloud(r, 600 * k.f, 3000 * k.f, 10, 0.01, 0.004, 0.06, 0.0012 * k.tc, tilt=0.0, rough=0.8)
            thump = _dull(r, 0.12, 110 * k.f, 0.02)
            g = 20 * np.log10(v)
            layers += [(board, t0, g), (knock, t0, g - 8 + k.h), (thump, t0, g - 6 + k.lo)]
        layers.append((_burst(r, 0.03, 1500, 6000, 0.008, attack=0.003), tt + 0.01, -22))
        y = _mix(layers, T)
    elif surface == 'stone':
        # a hard sole on stone: the heel thud with a sharp broadband tick on top, a scuff of grit as the foot rolls
        layers = []
        for t0, v in ((th, hv), (tt, tv)):
            tick = _burst(r, 0.02, 1500, 9000 * k.f ** 0.5, 0.0015 * k.tc, attack=0.0002)
            g = 20 * np.log10(v)
            layers += [(_heel(r, k, 0.0015, 0.015), t0, g - 2 + k.lo), (tick, t0, g - 3 + k.h)]
        layers.append((_burst(r, 0.06, 1200, 7000, 0.015, attack=0.006), tt + 0.008, -16))
        m = r.integers(10, 20)
        ev = _grains(r, n, tt + r.uniform(0.0, 0.04, m), _power(r, m, 0.05, 1.0) * 0.3,
                     _logu(r, 2000, 7000, m), r.uniform(0.7, 1.5, m), r.uniform(0.001, 0.002, m))
        y = _mix(layers, T)
        ev = _norm(ev) * np.abs(y).max() * 10 ** (-16 / 20)
        room = 'studio'
    elif surface == 'metal':
        # a boot on a steel deck: the boot's thud, the plate ringing dull (dense low modes damped by the weight on
        # it, 0.1-0.4 s) under a rough band of noise, the grating rattling; a 2 ms rubber contact keeps the top soft
        layers = []
        for t0, v in ((th, hv), (tt, tv)):
            plate = _cloud(r, 160 * k.f, 2600 * k.f, 40, 0.07 * k.ring, 0.015, 0.4, 0.002 * k.tc, tilt=-4.0,
                           rough=0.7)
            g = 20 * np.log10(v)
            rattle = [(_knock(r, 900 * k.f, 5000, 0.004), t0 + d, g - 12 - 3 * i)        # the grating rattling
                      for i, d in enumerate(np.cumsum(r.uniform(0.01, 0.025, 3)))]
            layers += [(_heel(r, k, 0.0015, 0.02), t0, g + k.lo), (plate, t0, g - 3)] + rattle
        y = _mix(layers, T + 0.1)
    elif surface == 'gravel':
        # PhISEM (Cook): stones colliding at a Poisson rate that follows the rise of the force, power-law energies
        tt_ = _rate_times(r, n, dF, 4000.0, 1500.0 / (dF.max() + 1e-9))
        tt_ = np.concatenate([tt_, _rate_times(r, n, F, 900.0, 600.0 / F.max())])
        m = len(tt_)
        ev = _grains(r, n, tt_, _power(r, m, 0.03, 1.0, 1.6), _logu(r, 900, 6000, m) * k.f ** 0.5,
                     r.uniform(1.5, 5.0, m), r.uniform(0.0008, 0.003, m))
        grind = dsp.svf(r.standard_normal(n), 900 * k.f, 0.8, 'bandn') * F
        thump = _dull(r, 0.1, 100 * k.f, 0.02)
        ev = _norm(ev)
        y = _mix([(grind, 0, -10), (thump, th, -9 + k.lo)], T)
        ev = ev * 10 ** (k.h / 3 / 20)
    elif surface == 'snow':
        # compaction (Fontana and Bresin 2003): crumpling events whose rate grows through the stance, power-law
        # energy; cold snow also squeaks: short stick-slip episodes of the grains rubbing (periods near 1 ms)
        comp = np.cumsum(F) / SR
        tt_ = _rate_times(r, n, F * (0.3 + comp / comp.max()), 8000.0, 5000.0 / F.max())
        m = len(tt_)
        ev = _grains(r, n, tt_, _power(r, m, 0.02, 1.0, 1.5), _logu(r, 500, 4500, m) * k.f ** 0.5,
                     r.uniform(0.7, 1.5, m), r.uniform(0.001, 0.003, m), sharp=0.8)
        ev = _norm(ev)
        squeak = np.zeros(n)
        for _ in range(r.integers(1, 3)):
            a, d = r.uniform(th + 0.01, tt + 0.03), r.uniform(0.015, 0.04)
            rate = np.where((_t(n) > a) & (_t(n) < a + d), _vary(r, 900, 0.2) * k.f ** 0.5, 0.0)
            squeak += _creak(r, T, rate, [(_vary(r, 1800, 0.2), 3.0, 1.0), (3200, 3.0, 0.5)], jitter=0.2, stuck=0.0)
        y = _mix([(_heel(r, k, 0.008, 0.03, 600), th, -8 + k.lo), (squeak, 0, -9 + k.h)], T)
        ev = ev * np.abs(y).max() * 1.2
    else:      # grass
        # a soft crunch: many small dry events (blades and stalks snapping, 2-6 kHz, broad, no pitch) at a rate
        # following the force, over a soft heel; almost no tone at all
        tt_ = _rate_times(r, n, F, 6000.0, 3500.0 / F.max())
        m = len(tt_)
        ev = _grains(r, n, tt_, _power(r, m, 0.05, 1.0, 1.5), _logu(r, 2000, 6500, m),
                     r.uniform(0.6, 1.2, m), r.uniform(0.001, 0.002, m), sharp=0.9)
        swish = dsp.butter(r.standard_normal(n), 'high', 2500) * np.sqrt(F)
        y = _mix([(_heel(r, k, 0.006, 0.025, 700), th, -9 + k.lo), (swish, 0, -20 + k.h / 2)], T)
        ev = _norm(ev) * np.abs(y).max() * 1.6
    return _out(c, y, ev, room=room, spread=0.2)


STEPS = {
    'grass': ('a soft crunchy footstep in dry grass', 'a foot stepping on grass and leaves'),
    'wood': ('a footstep on a wooden floor, heel then toe', 'a hollow knock of a shoe on floorboards'),
    'stone': ('a hard footstep on stone, a sharp heel click', 'a shoe tapping on a concrete floor'),
    'gravel': ('a crunchy footstep on gravel', 'feet crunching on loose stones'),
    'snow': ('a crunchy squeaky footstep in snow', 'a boot crunching into packed snow'),
    'metal': ('a heavy boot step on a metal floor', 'a footstep on a steel plate, a dull metallic clunk'),
}


def _step_fn(surface):
    def fn(c, r):
        return _step(c, r, surface)
    fn.__doc__ = f'one footstep on {surface}: heel then the ball of the foot through the surface (_step)'
    return fn


for _s, _says in STEPS.items():
    design(f'step/{_s}', kind='tick', says=_says, uses=('footstep', 'step', 'walk', 'run'), tags=(_s,), domains=DOM,
           dur=0.25, axes=AXES, material=_s)(_step_fn(_s))


# ------------------------------------------------------------------------------------------------ swings and dashes
@design('dash/air', kind='whoosh', says=('a fast short whoosh of air that drops in pitch as it passes',
                                        'a quick swoosh rushing by', 'a fast air swish'),
        uses=('dash', 'dodge', 'sprint', 'lunge', 'speed burst'), domains=DOM, dur=0.4, axes=AXES)
def dash_air(c, r):
    """a fast body rushing past the ear: air turbulence with a faint wandering whistle carried past at 55 m/s and
    under a metre (fx.doppler: the pitch falls fastest at the closest point, the level peaks sharply with 1 / r),
    a cloth flick on the start"""
    k = _k(c)
    T = 0.5 * k.L
    n = dsp.n_of(T)
    src = _air(T, r, center=1500 * k.f, flutter=0.3, rate=25, bright=0.5, whistle=0.25)
    src *= dsp.fade(np.ones(n), 0.08, 0.15)
    y = fx.doppler(src, speed=_vary(r, 55.0, 0.1) * (1 + 0.2 * k.e) / k.L ** 0.5, dist=0.7 * k.L, t_close=0.2 * k.L,
                   width=0.9)
    y = dsp.fade(y / (np.abs(y).max() + 1e-12), 0.002, 0.04)
    flick = _burst(r, 0.05, 3000, 10000, 0.01, attack=0.003)
    return _out(c, _sum(y, dsp.pan_to_stereo(flick, -0.3) * 10 ** ((-20 + k.h) / 20)), room='room', spread=0.3)


@design('swing/blade', kind='whoosh',
        says=('a thin sharp swish of a sword cutting the air', 'a fast blade whoosh with a whistle',
              'a quick high swoosh'),
        uses=('swing', 'sword', 'slash', 'attack', 'melee', 'swish'), domains=DOM, dur=0.3, axes=AXES)
def swing_blade(c, r):
    """a sword swung edge-on (d about 6 mm, tip 30 m/s): the Aeolian tones of its sections whistle together near
    1 kHz, narrow because the flow off a thin edge is nearly laminar, with v^6 making the peak short and sharp"""
    k = _k(c)
    T = 0.32 * k.L
    peak, fall = _vary(r, 0.09, 0.1) * k.L, _vary(r, 0.07, 0.1) * k.L * k.ring
    v = _vary(r, 32.0, 0.08) * (1 + 0.15 * k.e)
    y = _aeolian(r, T, peak, fall, v, 0.0065 / k.f, r_in=0.25, sections=9, wake=0.4)
    air = _burst(r, T, 3000, 12000, 0.04, attack=0.06 * k.L)
    y = fx.saturate(_mix([(y, 0, 0), (air, 0, -18 + k.h)]), 6.0, 'tanh', level='rms')   # the rare noise peak tamed
    return _out(c, y, room='room', spread=0.4)


@design('swing/heavy', kind='whoosh', says=('a low heavy whoosh of a club swung through the air', 'a slow deep swoosh',
                                           'a hammer swinging past'),
        uses=('swing', 'club', 'hammer', 'heavy attack', 'whoosh'), domains=DOM, dur=0.5, axes=AXES)
def swing_heavy(c, r):
    """a club or hammer (d about 5 cm, tip 25 m/s): its tones sit near 100 Hz and are broad (turbulent flow), and
    it moves slower, so the swish is low, long and dark; the head adds a thicker low wake"""
    k = _k(c)
    T = 0.55 * k.L
    y = _aeolian(r, T, 0.17 * k.L, 0.13 * k.L * k.ring, _vary(r, 25.0, 0.08) * (1 + 0.15 * k.e), 0.05 / k.f,
                 r_in=0.35, sections=6, wake=0.8)
    head = _aeolian(r, T, 0.17 * k.L, 0.12 * k.L * k.ring, 26.0, 0.012 / k.f, r_in=0.85, sections=2, wake=0.2)
    return _out(c, _mix([(y, 0, 0 + k.lo / 2), (head, 0, -9 + k.h)]), room='room', spread=0.4)


@design('whoosh/pass', kind='whoosh',
        says=('an object flying past with a falling pitch', 'a whoosh passing by from one side to the other',
              'something rushing past the listener'),
        uses=('whoosh', 'fly by', 'pass by', 'projectile', 'transition'), domains=DOM, dur=1.4, axes=AXES)
def whoosh_pass(c, r):
    """a source of turbulence with a whistle on a straight line past the listener (fx.doppler: exact delay, so the
    pitch falls through the pass, 1 / r level, air absorption far away, the pan following the angle); bigger
    objects are lower and slower"""
    k = _k(c)
    T = 1.5 * k.L
    n = dsp.n_of(T)
    src = _air(T, r, center=800 * k.f, flutter=0.35, rate=10, bright=0.5, whistle=0.3)
    src = src + 0.5 * dsp.svf(r.standard_normal(n), 220 * k.f, 1.0, 'bandn') * 3
    src *= dsp.fade(np.ones(n), 0.2, 0.3)
    speed = _vary(r, 32.0, 0.1) * (1 + 0.25 * k.e) / k.L ** 0.5
    y = fx.doppler(src, speed=speed, dist=_vary(r, 2.2, 0.15) * k.L ** 0.5, t_close=0.5 * T, width=0.9)
    y = dsp.fade(y, 0.01, 0.1)
    return _out(c, _shelf(y, k.h / 2), room='room')


# ------------------------------------------------------------------------------------------------ hits
@design('hit/punch', kind='hit',
        says=('a punch landing on a body, a fast heavy thwack', 'a fist hitting flesh with a thud',
              'a meaty smack'),
        uses=('punch', 'hit', 'melee', 'karate kick', 'smack', 'impact'), domains=DOM, dur=0.3, axes=AXES)
def hit_punch(c, r):
    """a fist on a body: the slap of skin (a short mid crack), the flesh and the chest below it (a dense, heavily
    damped body of modes at 150-700 Hz struck by a soft 3 ms contact), a pitch-dropping thump for weight, cloth"""
    k = _k(c)
    T = 0.32 * k.L
    crack = _burst(r, 0.03, 1500, 7000, 0.0025 * k.tc, attack=0.0002)
    slap = _burst(r, 0.05, 600 * k.f, 2600, 0.01 * k.tc, attack=0.0004)
    flesh = _cloud(r, 140 * k.f, 750 * k.f, 12, 0.035 * k.ring, 0.012, 0.2, 0.003 * k.tc)
    thump = _thump(r, T, 160 * k.f, 65 * k.f, _vary(r, 0.025, 0.3), _vary(r, 0.045, 0.2) * k.L * k.ring,
                   attack=0.0015, harm=0.35)
    cloth = _burst(r, 0.08, 2500, 9000, 0.02, attack=0.002)
    y = _mix([(thump, 0, -2 + k.lo), (flesh, 0, -2), (slap, 0, -4 + k.h / 2), (crack, 0, -7 + k.h),
              (cloth, 0.003, -20)], T)
    y = fx.saturate(y, 5.0 + 2 * k.e, 'tanh', level='peak')
    return _out(c, y, room='room', spread=0.2)


@design('hit/slap', kind='hit', says=('a sharp slap of a hand on skin', 'a crisp smack', 'a face slap'),
        uses=('slap', 'hit', 'smack', 'clap'), domains=DOM, dur=0.2, axes=AXES)
def hit_slap(c, r):
    """an open hand on skin: the air trapped between them is pushed out in a few ms, a broad crack whose colour is
    the cupped hand's air pocket (a Helmholtz-like resonance at 0.7-2 kHz, q 1.5-3: flatter is higher and drier); the
    palm and the fingers do not land at once, so the crack is two to four contacts spread over 1-20 ms, a
    different spread every time; the flesh under it gives a short meaty smack at 150-400 Hz and the skin a short
    bright splat (15 ms of 1.5-9 kHz noise); a short room"""
    k = _k(c)
    T = 0.22 * k.L
    pocket = _logu(r, 700, 2000, 1)[0] * k.f ** 0.5
    layers = []
    for i, dt in enumerate(np.concatenate([[0.0], np.sort(r.uniform(0.001, 0.02, r.integers(1, 4)))])):
        crack = _burst(r, 0.06, 500 * k.f ** 0.5, 10000, _vary(r, 0.005, 0.35) * k.tc, attack=0.0002)
        crack = 0.6 * dsp.svf(crack, pocket * r.uniform(0.85, 1.2), r.uniform(1.5, 3.0), 'bandn') + crack
        layers.append((crack, dt, (0.0 if i == 0 else r.uniform(-9, -2)) + k.h / 2))
    meat = _dull(r, 0.08, _vary(r, 260, 0.3) * k.f, _vary(r, 0.012, 0.3), attack=0.0008, q=1.6)
    skin = _cloud(r, 250 * k.f, 1400 * k.f, 12, r.uniform(0.006, 0.015), 0.006, 0.12, 0.001 * k.tc)   # cheek or arm
    splat = _burst(r, 0.08, 1500, 9000, _vary(r, 0.015, 0.3), attack=0.001)     # the skin and the air after it
    y = _mix(layers + [(meat, 0.0005, r.uniform(-12, -8) + k.lo), (skin, 0, r.uniform(-18, -12)),
                       (splat, 0.001, r.uniform(-14, -9))], T)
    return _out(c, y, room='room', spread=0.25)


@design('hit/metal', kind='hit',
        says=('a sword striking metal armour, a sharp clang', 'a metallic clank with a short ring',
              'a weapon hitting a steel plate'),
        uses=('hit', 'armour', 'sword hit', 'clang', 'metal impact'), domains=DOM, dur=0.8, axes=AXES)
def hit_metal(c, r):
    """steel on a steel shell: a curved, riveted plate is no free plate: its modes lie dense and irregular (here 40,
    log-spread over 0.4-9 kHz on top of the plate's lowest few, each a beating doublet), every mode decays at the
    steel law (higher die faster) cut short by the padding and straps behind it; a 0.15 ms steel-on-steel contact
    makes it bright, the shell's rattle is a band of noise dying with it; the body behind gives a low thunk; a
    scrape of the edge sliding off"""
    k = _k(c)
    T = 0.75 * k.L * k.ring
    f0 = _vary(r, 470 * k.f, 0.06)
    rat = np.concatenate([[1.0, 1.73, 2.33], _logu(r, 0.85, 19.0, 37)])
    fr = np.concatenate([f0 * rat, f0 * rat * (1 + r.uniform(0.001, 0.004, len(rat)))])
    amp = np.tile(r.uniform(0.2, 1.0, len(rat)) * rat ** -0.35, 2)
    tau = 0.16 * k.ring * (fr / 1000) ** -0.6
    clang = _strike(r, fr, tau, amp, T, 0.00015 * k.tc, grit=0.6)
    clang = _norm(clang + 0.5 * _burst(r, T, 1500, 12000, 0.05 * k.ring, attack=0.0002))
    tick = _burst(r, 0.02, 3000, 14000, 0.0012 * k.tc, attack=0.0001)
    thunk = _thump(r, 0.15, 160 * k.f, 110 * k.f, 0.015, 0.035, harm=0.3)
    scrape = _burst(r, 0.12, 2500, 9000, 0.03, attack=0.008)
    y = _mix([(clang, 0, 0), (tick, 0, -8 + k.h), (thunk, 0, -9 + k.lo), (scrape, 0.008, -24)], T)
    return _out(c, y, room='room', spread=0.35)


@design('hit/wood', kind='hit',
        says=('a hard knock on a wooden crate', 'a hollow wooden thunk', 'a blow on a wooden shield'),
        uses=('hit', 'wood hit', 'knock', 'shield hit', 'crate'), domains=DOM, dur=0.35, axes=AXES)
def hit_wood(c, r):
    """a blow on a board box: the planks' modes (100 Hz-3 kHz, wood damping: 20-150 ms, higher die first), the air
    inside the box ringing low and short, a 1 ms contact knock"""
    k = _k(c)
    T = 0.4 * k.L * k.ring ** 0.5
    planks = _cloud(r, 130 * k.f, 2800 * k.f, 30, 0.035 * k.ring, 0.006, T, 0.0011 * k.tc, tilt=-2.0)
    box = _strike(r, [_vary(r, 165 * k.f, 0.08), 0], [0.022 * k.ring, 0.01], [1, 0], 0.25, 0.003)
    knock = _burst(r, 0.02, 1500, 7000, 0.0015 * k.tc, attack=0.0002)
    y = _mix([(planks, 0, 0), (box, 0, -4 + k.lo), (knock, 0, -10 + k.h)], T)
    return _out(c, y, room='room', spread=0.25)


@design('hit/crit', kind='hit',
        says=('a heavy powerful punch with a sharp crack and a deep boom', 'a crushing bright impact',
              'a hard smash with a bass thump'),
        uses=('critical hit', 'hit', 'heavy hit', 'smash', 'impact'), domains=DOM, dur=0.5, axes=AXES)
def hit_crit(c, r):
    """a critical: the punch's body with a brighter, harder crack in front (shorter contact, more top), a short
    sub drop under it and more drive, all inside 10 ms so it stays one hit"""
    k = _k(c)
    T = 0.55 * k.L * k.ring ** 0.5
    crack = _burst(r, 0.04, 2000, 12000, 0.003 * k.tc, attack=0.0001)
    snap = _burst(r, 0.06, 900, 4000, 0.008 * k.tc, attack=0.0002)
    flesh = _cloud(r, 140 * k.f, 900 * k.f, 14, 0.04 * k.ring, 0.012, 0.25, 0.0022 * k.tc)
    thump = _thump(r, 0.3, 170 * k.f, 65 * k.f, 0.03, 0.08 * k.L, attack=0.001, harm=0.35)
    sub = _thump(r, T, 70 * k.f, 38 * k.f, 0.08, 0.15 * k.L * k.ring, attack=0.003, harm=0.3)
    y = _mix([(thump, 0, -2), (sub, 0, -2 + k.lo), (flesh, 0, -4), (snap, 0, -5 + k.h / 2), (crack, 0, -6 + k.h)], T)
    y = fx.saturate(y, 8.0 + 2 * k.e, 'tanh', level='peak')
    return _out(c, y, room='studio', spread=0.3)


@design('hurt/thud', kind='thud', says=('a dull muffled thud', 'a soft heavy body blow', 'a low dull impact'),
        uses=('hurt', 'damage', 'player hit', 'take damage'), domains=DOM, dur=0.35, axes=AXES)
def hurt_thud(c, r):
    """a neutral damage cue with no voice: a blow on a padded body. The torso answers as a heavily damped box (a
    cloud of low modes at 70-450 Hz, t60 40-120 ms, struck by a soft 4-6 ms contact), the blow pushes the air out
    of the clothes and the chest (a short low whump of noise), the cloth and skin give a dull slap at the contact
    and no crack at all; a sine under it only for weight (a falling sine on its own is a kick drum)"""
    k = _k(c)
    T = 0.38 * k.L * k.ring ** 0.5
    torso = _cloud(r, 70 * k.f, 450 * k.f, 24, 0.03 * k.ring, 0.01, T, 0.005 * k.tc, tilt=-2.0, rough=0.7)
    whump = _dull(r, T, _vary(r, 110, 0.15) * k.f, 0.05 * k.L * k.ring, attack=0.003, q=0.9)
    slap = _burst(r, 0.06, 250 * k.f, 2200, _vary(r, 0.008, 0.3) * k.tc, attack=0.0012 * k.tc)
    body = _thump(r, T, _vary(r, 150, 0.08) * k.f, 80 * k.f, 0.05, 0.07 * k.L * k.ring, attack=0.004, harm=0.2)
    cloth = _burst(r, 0.12, 900, 5000, 0.03, attack=0.005)
    y = _mix([(torso, 0, 0), (whump, 0, -2 + k.lo / 2), (slap, 0, -7 + k.h / 2), (body, 0, -11 + k.lo),
              (cloth, 0.004, -20)], T)
    return _out(c, dsp.butter(y, 'low', 3500 * k.f ** 0.5), room='room', spread=0.15)


@design('shield/block', kind='hit',
        says=('a dull metallic thunk with a short shimmer', 'a blow deflected off a metal shield',
              'a heavy clank that rings briefly'),
        uses=('shield', 'block', 'parry', 'deflect', 'guard'), domains=DOM, dur=0.6, axes=AXES)
def shield_block(c, r):
    """a blow taken on a thick padded shield: its low plate modes heavily damped (a thunk, 50-120 ms), a low body,
    and the rim's high modes left free to ring a little as beating doublets (the shimmer)"""
    k = _k(c)
    T = 0.65 * k.L * k.ring ** 0.5
    f0 = _vary(r, 290 * k.f, 0.06)
    rat = np.array([1.0, 1.61, 2.27, 2.93, 3.88, 4.6]) * (1 + 0.02 * r.standard_normal(6))
    thunk = _strike(r, f0 * rat, 0.035 * k.ring * rat ** -0.5, rat ** -0.3, 0.3, 0.0006 * k.tc)
    thunk = _norm(thunk + 0.6 * _burst(r, 0.3, 200 * k.f, 3000, 0.02, attack=0.0004))
    hi = _logu(r, 2500, 9000, 16) * k.f ** 0.3
    hi = np.concatenate([hi, hi * (1 + r.uniform(0.0015, 0.004, 16))])
    shimmer = _strike(r, hi, 0.15 * k.ring * (hi / 4000) ** -0.5, r.uniform(0.3, 1.0, 32), T, 0.0002)
    body = _dull(r, 0.18, 130 * k.f, 0.04)
    tick = _burst(r, 0.02, 2000, 9000, 0.0015 * k.tc)
    y = _mix([(thunk, 0, 0), (body, 0, -5 + k.lo), (tick, 0, -12 + k.h), (shimmer, 0.001, -17)], T)
    return _out(c, y, room='room', spread=0.4)


# ------------------------------------------------------------------------------------------------ explosions
def _debris(r, n, t0, t1, rate, tau, lo, hi, fk=1.0):
    """debris after a blast: Poisson impacts at a rate decaying with time constant tau s, power-law sizes (big
    pieces lower and louder), pebble ticks and short stone knocks"""
    t = _times(r, lambda tt: rate * np.exp(-(tt - t0) / tau), t0, t1, rate)
    m = len(t)
    s = _power(r, m, 0.05, 1.0, 1.8)
    f = _logu(r, lo, hi, m) * (0.5 / (s + 0.5)) ** 0.5     # every piece its own pitch, bigger ones lower
    ev = _grains(r, n, t, s ** 1.2, f * fk, r.uniform(0.8, 2.0, m),
                 r.uniform(0.002, 0.006, m), sharp=0.6, spread=0.8)            # rounded: distant, in dust
    big = s > 0.4
    ev += _grains(r, n, t[big], s[big] ** 1.2, r.uniform(300, 900, big.sum()) * fk, 1.2, 0.006, sharp=0.6,
                  spread=0.8)
    return ev


@design('explosion/small', kind='boom', says=('a small sharp explosion, a pop and a short burst', 'a firecracker bang',
                                             'a small grenade blast'),
        uses=('explosion', 'firecracker', 'grenade', 'pop', 'blast'), domains=DOM, dur=0.8, axes=AXES)
def explosion_small(c, r):
    """a small charge: a 1.2 ms Friedlander blast (the crack), a short saturated body of low-passed noise, a small
    sub, a few bits of debris and a room"""
    k = _k(c)
    T = 0.8 * k.L * k.ring ** 0.5
    n = dsp.n_of(T)
    blast = _blast(r, 0.0012 * k.L ** 0.33 * k.tc ** 0.5, 0.06)
    body = dsp.svf(_noise(r, T, 'brown'), np.geomspace(2500 * k.f, 300 * k.f, n), 0.7, 'low')
    body *= _env(n, 0.002, 0.09 * k.L * k.ring)
    body = fx.saturate(_norm(body), 9.0, 'tanh', level='peak')
    sub = _dull(r, 0.4 * k.L, 85 * k.f, 0.08 * k.L, attack=0.002)
    crack = _burst(r, 0.1, 1200, 12000, 0.012 * k.tc, attack=0.0002)
    y = _mix([(blast, 0, 0 + k.h), (crack, 0, -4 + k.h), (body, 0.001, -2), (sub, 0, -5 + k.lo)], T)
    y = dsp.fade(y, 0.0, 0.4 * T)
    ev = _debris(r, n, 0.03, 0.45 * T, 120.0, 0.12, 800, 6000, k.f)
    ev = ev * np.minimum(1.0, 1.25 - 1.25 * _t(n) / T)[:, None] ** 2                 # all settled by T
    return _out(c, y, _norm(ev) * np.abs(y).max() * 10 ** (-12 / 20), room='studio', spread=0.35)


def _rumble(r, n, lo, hi, attack, tau, roll=4.0):
    """a rolling low rumble: band-limited noise with a slow 1/f wander in level (terrain echoes arriving)"""
    x = dsp.butter(_noise(r, n / SR, 'brown'), 'band', (lo, hi), 2)
    x *= np.exp(0.5 * dsp.wander(n / SR, r, roll)[:n]) * _env(n, attack, tau)
    return dsp.fade(_norm(x), 0.0, 0.3 * n / SR)


@design('explosion/big', kind='boom',
        says=('a big explosion with a deep boom and a rumbling tail', 'a huge blast with debris falling',
              'a powerful detonation'),
        uses=('explosion', 'blast', 'bomb', 'detonation', 'boom'), domains=DOM, dur=2.2, axes=AXES)
def explosion_big(c, r):
    """a large blast (Farnell; Hopkinson scaling: t+ grows with the cube root of the charge): a 6 ms Friedlander
    shock, a saturated body of low-passed noise whose cutoff falls as the fireball cools, a sub drop, debris
    landing at a decaying rate, a rumble that rolls on for two seconds, terrain echoes; highs die first"""
    k = _k(c)
    T = 2.3 * k.L * k.ring ** 0.5
    n = dsp.n_of(T)
    blast = _blast(r, 0.006 * k.L ** 0.33 * k.tc ** 0.5, 0.25)
    crack = _burst(r, 0.2, 900, 9000, 0.03 * k.tc, attack=0.0003)
    nb = dsp.n_of(1.4 * k.L)
    body = dsp.svf(_noise(r, nb / SR, 'brown'), np.geomspace(1400 * k.f, 160 * k.f, nb), 0.8, 'low')
    body = fx.saturate(_norm(body * _env(nb, 0.004, 0.32 * k.L * k.ring)), 10.0, 'tanh', level='peak')
    sub = _thump(r, 1.4 * k.L, 75 * k.f, 30 * k.f, 0.18, 0.45 * k.L * k.ring, attack=0.004, harm=0.35)
    rumble = _rumble(r, n, 25, 160 * k.f, 0.12, 0.7 * k.L * k.ring)
    y = _mix([(blast, 0, -1 + k.h), (crack, 0, -8 + k.h), (body, 0.002, -1), (sub, 0, -1 + k.lo),
              (rumble, 0.03, -8)], T)
    echo = dsp.butter(body, 'low', 600 * k.f)
    y = _mix([(y, 0, 0), (_norm(echo), _vary(r, 0.38, 0.15), -15), (_norm(echo), _vary(r, 0.85, 0.15), -21)], T)
    ev = _debris(r, n, 0.12, 0.65 * T, 120.0, 0.45 * k.L, 400, 5000, k.f)
    return _out(c, y, _norm(ev) * 10 ** (-17 / 20), room='hall', spread=0.4)


@design('explosion/distant', kind='boom',
        says=('a distant explosion, a muffled low rumble', 'a far away boom rolling in',
              'a deep dull explosion heard from far off'),
        uses=('explosion', 'distant explosion', 'artillery', 'thunder', 'rumble'), domains=DOM, dur=2.0, axes=AXES,
        allows=('stacked',))      # the terrain echoes 0.3-1.5 s after the blast are what makes it distant
def explosion_distant(c, r):
    """a blast a kilometre off: air has taken everything above a few hundred Hz, the shock has smoothed into a soft
    low thump, and what arrives is mostly the rumble and the echoes off the terrain, rolling in over seconds"""
    k = _k(c)
    T = 2.1 * k.L * k.ring ** 0.5
    n = dsp.n_of(T)
    thump = _thump(r, 1.0 * k.L, 70 * k.f, 34 * k.f, 0.15, 0.35 * k.L, attack=0.015 / k.tc ** 0.5, harm=0.4)
    shock = dsp.butter(_blast(r, 0.008, 0.3), 'low', 350 * k.f)
    rumble = _rumble(r, n, 25, 220 * k.f, 0.18, 0.75 * k.L * k.ring, roll=2.5)
    y = _mix([(thump, 0, 0 + k.lo), (_norm(shock), 0, -4 + k.h), (rumble, 0.05, -4)], T)
    for d, g in ((r.uniform(0.25, 0.45), -12), (r.uniform(0.6, 1.0), -15), (r.uniform(1.1, 1.5), -19)):
        y = _mix([(y, 0, 0), (dsp.butter(_norm(thump), 'low', 200 * k.f), d, g)], T)
    return _out(c, dsp.butter(y, 'low', 600 * k.f), room='hall', spread=0.5)


# ------------------------------------------------------------------------------------------------ shots
def _gun(c, r, k, tp, body_f, sub_f, body_tau, tail, mech, room):
    """a designed gunshot (Maher 2006; film practice): a muzzle-blast crack (Friedlander, HP), a mid snap at 1-3 kHz,
    a body (a fast pitch drop and driven noise), a short sub, the action's clicks, a slapback and the room"""
    T = tail * k.L * k.ring ** 0.5
    body_tau = _vary(r, body_tau, 0.2)
    crack = dsp.butter(_blast(r, tp * k.tc ** 0.5, 0.05), 'high', 1500)
    snap = _burst(r, 0.04, 1000, 3500, 0.006 * k.tc, attack=0.0002)
    body = _thump(r, 0.35, body_f * k.f, 0.35 * body_f * k.f, 0.025, body_tau * k.L, attack=0.0008, harm=0.5)
    grit = _burst(r, 0.3, 150, 2200 * k.f, body_tau * 0.6, attack=0.0005)
    sub = _thump(r, 0.6 * k.L, sub_f * k.f, 0.6 * sub_f * k.f, 0.05, 2.2 * body_tau * k.L, attack=0.002, harm=0.2)
    core = _mix([(crack, 0, 0 + k.h), (snap, 0, -6), (body, 0, -3), (grit, 0, -6), (sub, 0, -6 + k.lo)])
    core = fx.saturate(core, 9.0 + 2 * k.e, 'tanh', level='peak')
    slap = dsp.butter(core, 'low', 2500)
    layers = [(core, 0, 0), (slap, _vary(r, 0.055, 0.15), -14), (slap, _vary(r, 0.13, 0.15), -21)]
    for t0, g in mech:
        click = _strike(r, _logu(r, 1800, 5500, 4), r.uniform(0.004, 0.012, 4),
                        r.uniform(0.4, 1, 4), 0.05, 0.0002)
        layers.append((click, t0 + 0.01 * r.standard_normal(), g))
    y = _mix(layers, T)
    return _out(c, fx.limit(y / np.abs(y).max(), -1.0), room=room, spread=0.3)


@design('shot/pistol', kind='boom',
        says=('a punchy pistol gunshot with a sharp crack', 'a single gun shot with a short echo',
              'a handgun firing'),
        uses=('shoot', 'gunshot', 'pistol', 'fire weapon', 'shot'), domains=DOM, dur=0.9, axes=AXES)
def shot_pistol(c, r):
    """a pistol: a 0.8 ms blast, a body at 170 Hz falling, a slide cycling 70-110 ms later, slapback and a room"""
    k = _k(c)
    return _gun(c, r, k, 0.0008, 170, 85, 0.06, 0.9, [(0.08, -24)], 'studio')


@design('shot/heavy', kind='boom',
        says=('a heavy powerful gunshot with a deep boom', 'a shotgun blast with a long echo',
              'a big cannon-like shot'),
        uses=('shoot', 'shotgun', 'rifle', 'cannon', 'heavy weapon'), domains=DOM, dur=1.4, axes=AXES)
def shot_heavy(c, r):
    """a shotgun or heavy rifle: a 2.5 ms blast, a lower and longer body and sub, a pump racked after the shot"""
    k = _k(c)
    return _gun(c, r, k, 0.0025, 120, 60, 0.1, 1.5, [(0.5, -20), (0.62, -18)], 'hall')


@design('laser/pulse', kind='zap', says=('a sci-fi laser blaster shot', 'a punchy electronic pew with a zap',
                                        'a ray gun firing'),
        uses=('laser', 'blaster', 'pew', 'zap', 'shoot', 'ray gun'), domains=DOM, dur=0.45, axes=AXES)
def laser_pulse(c, r):
    """a designed blaster: a dispersive chirp (an impulse through an all-pass chain falls in pitch like a struck
    guy wire), an FM tone diving 2.4 kHz -> 260 Hz whose index dies as it falls (bright to pure), two takes
    detuned for width, a click and a low thump for body, saturated and placed in a plate"""
    k = _k(c)
    T = 0.42 * k.L * k.ring ** 0.5
    n = dsp.n_of(T)
    t = _t(n)
    imp = np.zeros(dsp.n_of(0.3))
    imp[8] = 1.0
    chirp = dsp.butter(_disperse(imp, -0.9, 180), 'band', (300 * k.f, 7000))
    chirp = dsp.fade(_norm(chirp * _env(len(chirp), 0.0, _vary(r, 0.09, 0.25) * k.ring)), 0.0, 0.05)
    f0, f1 = _vary(r, 2400, 0.08) * k.f, 260 * k.f
    dive, decay = _vary(r, 0.045, 0.2) * k.L, _vary(r, 0.08, 0.2) * k.ring
    st = []
    for det in (-1, 1):
        fc = f1 + (f0 - f1) * np.exp(-t / dive) * 2 ** (det * 6 / 1200)
        idx = 3.5 * np.exp(-t / 0.05) + 0.4
        ph = 2 * np.pi * (np.cumsum(fc) - fc[0]) / SR
        st.append(np.sin(ph + idx * np.sin(1.41 * ph)) * _env(n, 0.0015, decay))
    tone = np.stack(st, 1) * (1 + 0.25 * dsp.wander(T, r, 30.0)[:n, None])          # the bolt flickers
    click = _burst(r, 0.01, 2500, 12000, 0.0008 * k.tc)
    thump = _thump(r, 0.12, 180 * k.f, 70 * k.f, 0.015, 0.03, harm=0.3)
    mono = _mix([(chirp, 0, -4), (click, 0, -12 + k.h), (thump, 0, -9 + k.lo)], T)
    y = _sum(dsp.pan_to_stereo(mono), tone * 10 ** (-2 / 20))
    y = fx.saturate(y / np.abs(y).max(), 6.0, 'tanh', level='peak')
    return _out(c, dsp.fade(y, 0.0, 0.05), room='plate', spread=0.3)


def _circ(fn, y):
    """a filter applied to a loop as if it had always been playing: run on two copies, keep the second (every
    filter and room here forgets its past within one loop)"""
    n = len(y)
    return fn(np.concatenate([y, y]))[n:2 * n]


def _cnoise(r, n, lo, hi, tilt=0.0):
    """noise that loops without a seam (shaped in the frequency domain of exactly n samples), band lo..hi Hz with
    raised-cosine edges over half an octave, tilt dB per octave, unit RMS"""
    f = sfft.rfftfreq(n, 1 / SR)
    u = np.log2(np.maximum(f, 1.0))
    g = np.clip((u - np.log2(lo) + 0.25) * 2, 0, 1) * np.clip((np.log2(hi) - u + 0.25) * 2, 0, 1)
    g = np.sin(0.5 * np.pi * g) ** 2 * 2.0 ** (tilt * (u - np.log2(lo)) / 6.02)
    x = sfft.irfft((r.standard_normal(len(f)) + 1j * r.standard_normal(len(f))) * g, n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def _cmod(r, n, rates, depth):
    """a periodic random modulation over a loop of n samples: sines at whole cycles per loop near `rates` Hz"""
    L = n / SR
    t = _t(n)
    m = np.zeros(n)
    for f in rates:
        m += r.uniform(0.5, 1.0) * np.sin(2 * np.pi * max(1, round(f * L)) / L * t + r.uniform(0, 2 * np.pi))
    return depth * m / (np.abs(m).max() + 1e-12)


def _loop_out(c, y, room='room'):
    """a loop's finish: every filter and the room applied circularly, so the end runs into the start"""
    w = c.lerp('width', 0.0, 2.0, 1.0)
    y = _circ(lambda x: fx.width(x, w), y)
    y = _circ(lambda x: _shelf(x, 6.0 * c.bright + 2.0 * c.energy), y) * c.db('energy', 6.0)
    w = space.room(y, c.room(room), amount=c.space)        # a room is linear: its tail past the end wraps to the start
    out = np.array(w[:len(y)])
    for i in range(len(y), len(w), len(y)):
        out[:len(w[i:i + len(y)])] += w[i:i + len(y)]
    return out


@design('laser/beam', kind='zap',
        says=('a sustained sci-fi laser beam humming and crackling', 'a continuous energy beam',
              'a steady buzzing ray with a shimmering hum'),
        uses=('laser', 'beam', 'ray', 'plasma', 'charge'), domains=DOM, dur=1.0, axes=AXES, loop=True)
def laser_beam(c, r):
    """a held beam as a seamless loop: a low hum of detuned harmonic voices (whole cycles per loop, so the beating
    repeats exactly), an FM shimmer above it, a plasma band of noise flickering at 8-30 Hz, and arc crackles
    (sparse short bursts, wrapped around the loop)"""
    k = _k(c)
    L = c.seconds(1.0)
    n = dsp.n_of(L)
    t = _t(n)
    q = lambda f: max(1, round(f * L)) / L                  # a frequency with whole cycles per loop
    f0 = q(_vary(r, 110, 0.05) * k.f)
    hum = np.zeros((n, 2))
    for ch, det in enumerate((0.0, q(0.7))):             # the two sides beat slowly against each other
        hum[:, ch] = sum(h ** -1.1 * np.sin(2 * np.pi * h * (f0 + det) * t + r.uniform(0, 6.3))
                         for h in range(1, 18) if h * f0 < 4000)
    hum /= np.abs(hum).max()
    fs = q(_vary(r, 880, 0.05) * k.f)
    wob = _cmod(r, n, (3.0, 7.0), 1.0)
    shimmer = np.sin(2 * np.pi * fs * t + (1.5 + 0.8 * wob) * np.sin(2 * np.pi * q(fs * 1.5) * t))
    plasma = _cnoise(r, n, 1200 * k.f, 7000 * k.f) * (1 + _cmod(r, n, (8.0, 13.0, 21.0, 29.0), 0.7))
    m = r.poisson(30 * L * (1 + 0.5 * k.e))
    arcs = _grains(r, 2 * n, r.uniform(0, L, m), _power(r, m, 0.1, 1.0, 1.5), _logu(r, 2500, 9000, m),
                   r.uniform(0.8, 2.0, m), r.uniform(0.0003, 0.0015, m), spread=0.7)
    arcs = arcs[:n] + arcs[n:]                                   # an arc running past the end goes on at the start
    pulse = 1 + _cmod(r, n, (4.0, 6.0), 0.25)
    y = (hum * 10 ** ((-2 + k.lo) / 20) + dsp.pan_to_stereo(shimmer) * 10 ** (-10 / 20)
         + fx.decorrelate(plasma, 0.8) * 10 ** ((-16 + k.h) / 20)) * pulse[:, None]
    y = y + _norm(arcs) * 10 ** (-12 / 20) * np.abs(y).max()
    y = np.tanh(1.4 * y / np.abs(y).max())
    return _loop_out(c, y, room='room')


# ------------------------------------------------------------------------------------------------ doors and chests
def _slips(r, seconds, rate, modes, jitter=0.15, stick=0.15):
    """a lid's hinge: single slips at a jittered rate that follows the drive, each ringing the lid's modes ((freqs,
    taus, amps), dying between slips). Heard through earshot's ears as a creaking lid (z 3.1), where the same
    model on a door read as a tone: a lid creaks slower and lower. Mono, peak 1"""
    n = dsp.n_of(seconds)
    pulses = np.zeros(n)
    i = 0
    while True:
        f = rate[min(i, n - 1)]
        i += dsp.n_of(0.004) if f <= 1.0 else max(2, int(SR / f * (1 + jitter * r.standard_normal())))
        if i >= n:
            break
        if f > 1.0 and r.uniform() > stick:
            pulses[i] += (1 + 0.3 * r.standard_normal()) * (f / rate.max()) ** 0.5
    fr, ta, am = modes
    ir = dsp.modal(fr, ta, am, max(0.02, 6 * max(ta)), r, jitter=0.0) * _env(dsp.n_of(max(0.02, 6 * max(ta))),
                                                                              0.0007, 1.0)
    return dsp.fade(_norm(signal.fftconvolve(pulses, ir)[:n]), 0.002, 0.04)


def _ping_latch(r, k):
    """the hasp of a chest: a few short metal modes at 1.8-6 kHz struck by steel"""
    f = _logu(r, 1800, 6000, 4) * k.f ** 0.4
    return _strike(r, f, r.uniform(0.004, 0.015, 4), r.uniform(0.4, 1, 4), 0.06, 0.00015 * k.tc, grit=0.6)


def _latch(r, k, size=1.0):
    """a small metal latch: a snap of noise with a few short modes at 1.5-7 kHz in it (a bolt in its housing is
    damped by the wood around it: 2-6 ms), never a clean ping"""
    f = _logu(r, 1500, 7000, 8) * k.f ** 0.4 / size
    y = _strike(r, f, r.uniform(0.002, 0.006, 8), r.uniform(0.4, 1, 8), 0.06, 0.00015 * k.tc, grit=0.6)
    return _norm(y + 0.8 * _burst(r, 0.06, 1000, 9000, 0.004, attack=0.0002))


def _hinge(r, k, rate, ring=1.0):
    """a hinge creaking through a door (stick-slip, texture.stick_slip): the force is a sawtooth at the slip rate
    (Hz per sample), unsteady (10 % jitter per period, drops of uneven size, the hinge now and then holding for
    40-120 ms), and its bright edges ring the door's few strong panel modes (300-1500 Hz, q 7-14: a board's t60 of
    30-60 ms) and the pin and leaf of the hinge itself (2-3.5 kHz). A sawtooth through formants is a pitched voice;
    the jitter, the uneven drops and the holds make it a creak, not a tone. Mono, peak 1"""
    x, _ = _tex().stick_slip(r, rate, jitter=0.1, hold=0.018, hold_s=(0.04, 0.12), rough=0.45)
    x = x - dsp.butter(x, 'low', 20.0)
    modes = np.sort(_logu(r, 300, 1500, 5)) * k.f ** 0.5
    y = sum(r.uniform(0.5, 1.0) * dsp.svf(x, f, r.uniform(7, 14) * ring ** 0.5, 'bandn') for f in modes)
    y = y + 0.6 * dsp.svf(x, 170 * k.f, 2.0, 'bandn')                                  # the whole board
    pin = dsp.svf(x, _vary(r, 2600, 0.15), 6.0, 'bandn')
    edge = dsp.butter(x, 'high', 3000.0)
    y = _norm(y) + 0.25 * _norm(pin) + 0.12 * _norm(edge)
    return dsp.fade(_norm(dsp.butter(y, 'low', 7000.0)), 0.002, 0.03)


@design('door/open', kind='object', says=('a wooden door creaking open with a latch click', 'a creaky old door opening',
                                         'a door handle turning and hinges creaking'),
        uses=('door', 'door open', 'creak', 'open'), domains=DOM, dur=1.1, axes=AXES,
        allows=('rising_glide',))      # the creak climbs as the door swings faster: stick-slip follows the speed
def door_open(c, r):
    """a wooden door: the handle and latch (two metal clicks and the knock of the bolt freeing), then the hinge
    creaking (_hinge: a stick-slip sawtooth whose rate follows the swing, 30 Hz as it starts to 150-200 Hz at speed,
    wandering and now and then jumping as the grip changes, with holds) through the door's panel, and the low air
    the door pushes"""
    k = _k(c)
    T = 1.15 * k.L
    n = dsp.n_of(T)
    t = _t(n)
    t0, t1 = 0.17, 0.17 + _vary(r, 0.8, 0.08) * k.L
    v = np.clip((t - t0) / (t1 - t0), 0, 1)
    speed = np.sin(np.pi * v ** 0.8) ** 1.2 * (t > t0) * (t < t1)
    jumps = np.ones(n)                                       # the grip changes: the creak leaps to another register
    for tj in np.sort(r.uniform(t0 + 0.1, t1 - 0.1, r.integers(1, 4))):
        jumps[t > tj] *= r.choice([0.7, 1.35, 1.5])
    rate = (28 + 150 * speed ** 0.8 * (1 + 0.2 * k.e)) * jumps / k.f ** 0.5
    rate = rate * np.exp(0.2 * dsp.wander(T, r, 3.0)[:n]) * (speed > 0.04)
    creak = _hinge(r, k, rate, k.ring) * (0.35 + 0.65 * speed ** 0.5)
    move = dsp.butter(r.standard_normal(n), 'band', (70 * k.f, 350 * k.f)) * speed
    knock = _cloud(r, 120 * k.f, 1500 * k.f, 20, 0.02, 0.006, 0.15, 0.002 * k.tc)
    y = _mix([(_latch(r, k), 0, -6 + k.h), (_latch(r, k, 1.2), 0.085, -3 + k.h), (knock, 0.09, -9),
              (creak, 0, 0), (_norm(move), 0, -20 + k.lo)], T)
    return _out(c, dsp.fade(y, 0.0, 0.06), room='room', spread=0.3)


@design('door/close', kind='object',
        says=('a wooden door closing with a solid thud and a click', 'a door shutting', 'a door slamming shut'),
        uses=('door', 'door close', 'shut', 'close'), domains=DOM, dur=0.7, axes=AXES)
def door_close(c, r):
    """the swing (air pushed by the door, rising as it speeds up), then the shut: the door's board modes and the
    frame's low thud from a 2 ms wood-on-wood contact, the latch snapping a moment later, a rattle in the frame"""
    k = _k(c)
    T = 0.75 * k.L * k.ring ** 0.5
    n = dsp.n_of(T)
    ts = _vary(r, 0.24, 0.12) * k.L
    m = dsp.n_of(ts)
    move = dsp.butter(r.standard_normal(m), 'band', (80 * k.f, 600 * k.f)) * np.linspace(0, 1, m) ** 2
    door = _cloud(r, 90 * k.f, 1600 * k.f, 34, 0.04 * k.ring, 0.008, 0.4, 0.002 * k.tc, rough=0.6)
    thud = _thump(r, 0.3, 95 * k.f, 70 * k.f, 0.02, 0.06 * k.L, harm=0.4)
    rattle = _grains(r, n, ts + 0.03 + r.uniform(0, 0.06, 6), r.uniform(0.05, 0.3, 6), r.uniform(900, 3000, 6), 4,
                     0.002, spread=0.3)
    y = _mix([(_norm(move), 0, -16), (door, ts, 0), (thud, ts, -3 + k.lo),
              (_latch(r, k), ts + r.uniform(0.008, 0.02), -6 + k.h)], T)
    return _out(c, y, _norm(rattle) * 10 ** (-24 / 20), room='room', spread=0.3)


@design('door/slide', kind='object',
        says=('a sci-fi sliding door opening with a hiss', 'a pneumatic door whooshing open',
              'an automatic spaceship door sliding'),
        uses=('door', 'sliding door', 'airlock', 'hatch', 'open'), domains=DOM, dur=0.9, axes=AXES)
def door_slide(c, r):
    """a powered sliding door: a pneumatic release (a fast hiss falling in brightness), the servo spinning up and
    down under load (a rotation tone with its harmonics and gear buzz, rising under three semitones), the panel's
    low rumble as it slides, then a soft stop: a damped low thunk and a small click"""
    k = _k(c)
    T = 0.95 * k.L
    n = dsp.n_of(T)
    t = _t(n)
    run = _vary(r, 0.62, 0.08) * k.L
    hiss = dsp.svf(r.standard_normal(n), np.geomspace(9000, 2500, n), 0.7, 'high') * _env(n, 0.008, 0.18 * k.ring)
    sp = np.clip(t / run, 0, 1)
    speed = np.sin(np.pi * sp) ** 0.7 * (t < run)
    fm = 150 * k.f * (1 + 0.18 * speed)
    ph = 2 * np.pi * np.cumsum(fm) / SR
    servo = sum(h ** -1.2 * np.sin(h * ph) for h in range(1, 9)) * (1 + 0.3 * np.sin(6 * ph))
    servo = dsp.butter(servo, 'low', 1800) * speed
    slide = dsp.butter(r.standard_normal(n), 'band', (90 * k.f, 700 * k.f)) * speed
    stop = _thump(r, 0.2, 120 * k.f, 85 * k.f, 0.02, 0.04, attack=0.004, harm=0.3)
    y = _mix([(_norm(hiss), 0, -4 + k.h), (_norm(servo), 0.01, -10), (_norm(slide), 0, -10 + k.lo),
              (stop, run, -6 + k.lo), (_latch(r, k, 1.4), run + 0.004, -18 + k.h)], T)
    return _out(c, dsp.fade(y, 0.0, 0.08), room='room', spread=0.4)


@design('chest/open', kind='object', says=('a wooden chest lid creaking open with a soft magical sparkle',
                                          'a treasure chest opening', 'a creaky box lid lifting and a twinkle'),
        uses=('chest', 'treasure chest', 'loot', 'reward', 'open'), domains=DOM, dur=1.3, axes=AXES,
        allows=('ring',))      # sparkle on a pentatonic scale: its notes lie near the harmonics of one low note
def chest_open(c, r):
    """a wooden lid: the hasp clicking free, the hinges creaking low and slow (a heavier lid: a longer slip period),
    the lid knocking back on its stop, and a separate, quieter reward layer: a rising pentatonic sparkle"""
    k = _k(c)
    T = 1.35 * k.L * k.ring ** 0.3
    n = dsp.n_of(T)
    t = _t(n)
    t0, t1 = 0.08, 0.08 + _vary(r, 0.6, 0.1) * k.L
    v = np.clip((t - t0) / (t1 - t0), 0, 1)
    speed = np.sin(np.pi * v) ** 1.3 * (t < t1)
    rate = (28 + 90 * speed) * (speed > 0.03) / k.f ** 0.5 * np.exp(0.35 * dsp.wander(T, r, 5.0)[:n])
    f = np.sort(_logu(r, 140, 2000, 8)) * k.f
    creak = _slips(r, T, rate, (f, 0.008 * (f / 300) ** -0.4, r.uniform(0.3, 1, 8) * (f / 140) ** -0.3))
    knock = _cloud(r, 140 * k.f, 1800 * k.f, 24, 0.025, 0.007, 0.2, 0.002 * k.tc)
    y = _mix([(_ping_latch(r, k), 0, -4 + k.h), (creak * speed, 0, -3), (knock, t1, -5 + k.lo)], T)
    m = r.integers(16, 24)
    ts = np.sort(t1 * 0.45 + r.uniform(0, 0.55, m) ** 1.3 * (T - t1 * 0.45 - 0.35))
    root = 523.25 * 2 ** (r.integers(-2, 3) / 12)
    sp = _sparkle(r, n, ts, root, 1800, 7500, rise=(ts - ts[0]) / (ts[-1] - ts[0] + 1e-9))
    return _out(c, y, _norm(sp) * 10 ** (-13 / 20), room='room', spread=0.3)


# ------------------------------------------------------------------------------------------------ breaking
@design('break/crate', kind='object',
        says=('a wooden crate smashing and splintering', 'wood cracking and breaking apart',
              'planks snapping'),
        uses=('break', 'smash', 'destroy', 'crate', 'wood break'), domains=DOM, dur=0.9, axes=AXES)
def break_crate(c, r):
    """a crate breaks: the blow on the box (its plank modes and a thud), fibres tearing as a cascade of short
    cracks (a self-exciting burst process: one fracture triggers the next), then the planks falling: short wood
    knocks bouncing"""
    k = _k(c)
    T = 0.95 * k.L * k.ring ** 0.4
    n = dsp.n_of(T)
    blow = _cloud(r, 120 * k.f, 2500 * k.f, 30, 0.03, 0.006, 0.3, 0.0015 * k.tc, rough=0.6)
    thud = _dull(r, 0.25, 130 * k.f, 0.05 * k.L)
    tex = _tex()
    tc = tex.arrivals(r, 500.0 * (1 + 0.4 * k.e), 0.3 * k.L, cluster=0.6, spread=0.01)[0] + 0.002
    m = len(tc)
    s = _power(r, m, 0.05, 1.0, 1.7) * np.exp(-tc / 0.12)
    ev = _grains(r, n, tc, s, _logu(r, 500, 6000, m) * k.f ** 0.5, r.uniform(1, 4, m),
                 r.uniform(0.0008, 0.005, m))
    layers = [(blow, 0, 0), (thud, 0, -4 + k.lo)]
    tt = 0.12
    for i in range(r.integers(3, 6)):
        tt += r.uniform(0.05, 0.16)
        if tt > T - 0.12:
            break
        layers.append((_knock(r, 300 * k.f, 2500 * k.f, 0.012), tt, -7 - 3 * i))
    y = _mix(layers, T)
    return _out(c, y, _norm(ev) * np.abs(y).max() * 10 ** ((1 + k.h) / 20), room='room', spread=0.3)


@design('break/glass', kind='object', says=('a glass pane shattering', 'glass breaking into tinkling shards',
                                           'a window smashing'),
        uses=('break', 'shatter', 'glass break', 'smash', 'window'), domains=DOM, dur=1.6, axes=AXES)
def break_glass(c, r):
    """a pane shatters (crackling-noise statistics, Sethna et al. 2001): the crack (a broadband click and the pane's
    modes for a few ms), the flex of the pane (a low whoomp), then shards: power-law sizes, each 3 modes at
    2-14 kHz (smaller is higher: f ~ m^-1/3) decaying at glass damping, arriving at an Omori-like falling rate,
    and the pieces landing and tinkling for a second more"""
    k = _k(c)
    T = 1.6 * k.L * k.ring ** 0.4
    n = dsp.n_of(T)
    crack = _burst(r, 0.03, 1500, 16000, 0.002 * k.tc, attack=0.0001)
    pane = _strike(r, _logu(r, 1200, 6000, 10) * k.f ** 0.5, r.uniform(0.005, 0.02, 10),
                   r.uniform(0.4, 1, 10), 0.1, 0.0002)
    whoomp = _thump(r, 0.12, 160 * k.f, 70 * k.f, 0.015, 0.03, harm=0.3)
    y = _mix([(crack, 0, 0 + k.h), (pane, 0, -4), (whoomp, 0, -8 + k.lo)], T)
    # shards: lambda(t) = l0 (1 + t / t0)^-p, l0 ~ 1500/s, t0 20 ms, p 1.3
    ts = _times(r, lambda tt: 1500 * (1 + tt / 0.02) ** -1.3, 0.001, 0.4 * T, 1500)
    ts = np.concatenate([ts, _times(r, lambda tt: 40 * np.exp(-(tt - 0.25) / 0.35), 0.25, 0.7 * T, 40)])
    m = len(ts)
    s = _power(r, m, 0.02, 1.0, 1.6)
    f1 = np.clip(2400 * s ** (-1 / 3), 2000, 13000) * k.f ** 0.5 * r.uniform(0.9, 1.1, m)
    g = s ** 0.7 * np.exp(-ts / 0.6)
    ev = np.zeros((n, 2))
    pan = r.uniform(-0.8, 0.8, m)
    for ratio in (1.0, r.uniform(1.5, 2.2), r.uniform(2.6, 3.6)):
        f = np.minimum(f1 * ratio * r.uniform(0.97, 1.03, m), 0.42 * SR)
        ev += _pings(r, n, ts, g / ratio, f, 1.0 / (np.pi * 4e-3 * f) * k.ring, pan=pan)    # t60 0.1-0.4 s
    ev += _grains(r, n, ts, 0.3 * g, f1 * 1.5, 1.5, 0.0004, spread=0.8)
    return _out(c, y, _norm(ev) * 10 ** (-3 / 20), room='room', spread=0.3)


@design('break/rock', kind='object', says=('a rock cracking and crumbling', 'stone breaking apart into rubble',
                                          'a boulder smashing'),
        uses=('break', 'crumble', 'rock', 'stone break', 'destroy'), domains=DOM, dur=1.0, axes=AXES)
def break_rock(c, r):
    """stone breaks: a heavy knock and a crack (stone has few modes and they die in under 30 ms), a crumbling
    cascade of grit with power-law sizes, chunks falling and knocking, dust"""
    k = _k(c)
    T = 1.0 * k.L * k.ring ** 0.4
    n = dsp.n_of(T)
    thump = _thump(r, 0.4, 110 * k.f, 55 * k.f, 0.03, 0.08 * k.L, harm=0.4)
    crack = _burst(r, 0.04, 700, 7000, 0.004 * k.tc, attack=0.0002)
    knock = _cloud(r, 250 * k.f, 2500 * k.f, 12, 0.012, 0.005, 0.1, 0.0008 * k.tc, rough=0.8)
    ts = _times(r, lambda tt: 900 * (1 + tt / 0.03) ** -1.2, 0.002, 0.6 * T, 900)
    m = len(ts)
    s = _power(r, m, 0.03, 1.0, 1.7)
    ev = _grains(r, n, ts, s * np.exp(-ts / 0.35), np.clip(1800 * s ** -0.4, 300, 6000) * k.f, r.uniform(0.8, 2.5, m),
                 r.uniform(0.001, 0.005, m), sharp=0.6)
    layers = [(thump, 0, 0 + k.lo), (crack, 0, -3 + k.h), (knock, 0, -5)]
    tt = 0.1
    for i in range(r.integers(3, 6)):
        tt += r.uniform(0.06, 0.18)
        if tt > T - 0.1:
            break
        layers.append((_knock(r, 250 * k.f, 2500 * k.f, 0.008), tt, -7 - 3 * i))
    dust = _burst(r, T, 200, 3000 * k.f, 0.18 * k.ring, attack=0.02)
    y = _mix(layers + [(dust, 0.01, -16)], T)
    return _out(c, y, _norm(ev) * 10 ** (-7 / 20), room='room', spread=0.3)


# ------------------------------------------------------------------------------------------------ water
def _splash(c, r, k, big):
    """a body entering water: the slap of the surface (broad, a few ms), the air cavity pinching off into a large
    bubble whose pitch rises as it collapses (the bloop: Minnaert f = 3.26 / R Hz), droplets falling back (small
    bubbles 1-4 kHz with rising pitch, van den Doel 2005) at a decaying rate, the spray's hiss"""
    T = (1.3 if big else 0.6) * k.L * k.ring ** 0.4
    n = dsp.n_of(T)
    slap = _burst(r, 0.05, 300 * k.f, 6000, (0.009 if big else 0.005) * k.tc, attack=0.0004)
    low = _thump(r, 0.25, (110 if big else 220) * k.f, (60 if big else 130) * k.f, 0.02, 0.04 * k.L, harm=0.3)
    fb = _vary(r, 210 if big else 520, 0.1) * k.f
    tb = r.uniform(0.02, 0.05) * (2 if big else 1)
    bloop = _pings(r, n, [tb], [1.0], [fb], [0.035 * k.L], 0.002, [0.1 / 0.035 * 4], spread=0.1)
    count = r.poisson(70 if big else 22)
    td = _times(r, lambda tt: 200 * np.exp(-tt / (0.3 if big else 0.12)), 0.03, 0.85 * T, 200)[:count]
    rad = _power(r, len(td), 0.0008, 0.004, 2.2)
    f = 3.26 / rad * k.f ** 0.3
    dec = 1.0 / (np.pi * f * (0.013 + 0.0025 * np.sqrt(f / 1000)) + 1e-9) * 1.5
    drops = _pings(r, n, td, (rad / 0.004) ** 0.8 * np.exp(-td / 0.5), f, dec, 0.0003, 0.1 / dec * 3, spread=0.8)
    drops += _grains(r, n, td, 0.15 * (rad / 0.004), 4000, 1.2, 0.0005, spread=0.8)
    spray = _burst(r, T, 1500, 12000, (0.3 if big else 0.1) * k.ring, attack=0.004)
    y = _mix([(slap, 0, 0 + k.h), (low, 0, (-4 if big else -9) + k.lo), (spray, 0.002, -12 if big else -14)], T)
    ev = _sum(_norm(bloop) * 10 ** (-9 / 20), _norm(drops) * 10 ** (-8 / 20))
    return _out(c, y, ev, room='room', spread=0.35)


@design('splash/small', kind='object', says=('a small splash in water', 'a quick watery plop and splash',
                                            'something small dropping into water'),
        uses=('splash', 'water', 'plop', 'drop in water'), domains=DOM, dur=0.6, axes=AXES,
        allows=('rising_glide',))      # the cavity bubble's pitch rises as it collapses: the physics of a plop
def splash_small(c, r):
    """a small object entering water (_splash)"""
    return _splash(c, r, _k(c), False)


@design('splash/big', kind='object',
        says=('a big splash of a body falling into water', 'a heavy watery crash and spray',
              'someone diving into a pool'),
        uses=('splash', 'dive', 'water', 'swim', 'fall in water'), domains=DOM, dur=1.3, axes=AXES,
        allows=('rising_glide',))      # as splash/small
def splash_big(c, r):
    """a body entering water (_splash): a bigger slap with a low whump, a lower bloop, many more droplets"""
    return _splash(c, r, _k(c), True)


@design('bubble/rise', kind='object', says=('a few bubbles rising in water', 'underwater bubbling blips',
                                           'air bubbles gurgling up'),
        uses=('bubble', 'underwater', 'water', 'potion', 'drown'), domains=DOM, dur=0.8, axes=AXES,
        allows=('rising_glide',))      # each bubble's pitch rises (van den Doel 2005): what makes it a bubble
def bubble_rise(c, r):
    """a few bubbles of 2-6 mm (Minnaert: f = 3.26 / R, 550-1600 Hz), each ringing at its own damping with the
    rising pitch of a bubble near the surface, in irregular succession; heard through water, so the top is soft"""
    k = _k(c)
    T = 0.85 * k.L
    n = dsp.n_of(T)
    m = r.integers(5, 9)
    t = np.sort(r.uniform(0.0, 0.6 * k.L, m))
    t[0] = 0.0
    rad = r.uniform(0.002, 0.006, m) / k.f
    f = 3.26 / rad
    dec = 1.0 / (np.pi * f * 0.035) * k.ring ** 0.5
    g = r.uniform(0.5, 1.0, m)
    ev = _pings(r, n, t, g, f, dec, 0.0015, 0.1 / dec * 4, spread=0.5)
    low = dsp.butter(r.standard_normal(n), 'band', (120, 600)) * _env(n, 0.05, 0.3)
    return _out(c, _sum(dsp.butter(ev, 'low', 3500 * k.f), dsp.pan_to_stereo(_norm(low)) * 0.03 * (1 + k.lo / 6)),
                room='room', spread=0.3, tilt=k.h / 3)


# ------------------------------------------------------------------------------------------------ magic
def _pop(r, k, f0=700.0):
    """a soft pop: a cavity snapping shut, a puff of air (noise through the cavity's falling resonance) over a
    short dull click"""
    n = dsp.n_of(0.12)
    t = _t(n)
    f = f0 * k.f * (0.5 + 0.5 * np.exp(-t / 0.012))
    y = dsp.svf(r.standard_normal(n), f, 4.0, 'bandn') * _env(n, 0.0008, 0.02)
    return _mix([(_norm(y), 0, 0), (_burst(r, 0.03, 300, 4000, 0.003), 0, -4),
                 (_dull(r, 0.08, 250 * k.f, 0.015), 0, -6)])


@design('teleport/out', kind='zap',
        says=('a quick whoosh sucked inward ending in a soft pop', 'a magical vanishing whoosh',
              'an inward rush of air and a pop'),
        uses=('teleport', 'vanish', 'warp', 'disappear', 'blink'), domains=DOM, dur=0.5, axes=AXES,
        allows=('rising_glide', 'ring'))   # the suction climbs into the pop; pentatonic glitter reads as a comb
def teleport_out(c, r):
    """vanishing: air drawn inward (a noise band whose level and centre climb toward the end, the shape of a
    reversed decay), a low swell under it, then the pop and a short glitter"""
    k = _k(c)
    T = 0.52 * k.L
    pre = 0.28 * k.L
    m = dsp.n_of(pre)
    t = _t(m)
    suck = dsp.svf(_air(pre, r, center=1500, flutter=0.3), np.geomspace(350 * k.f, 4500 * k.f, m), 2.0, 'bandn')
    suck = dsp.fade(_norm(suck * np.exp((t - pre) / 0.07)), 0.0, 0.004)
    swell = np.sin(2 * np.pi * 90 * k.f * t) * np.exp((t - pre) / 0.1)
    n = dsp.n_of(T)
    ts = pre + 0.01 + np.sort(r.uniform(0, 0.15, 8))
    gl = _sparkle(r, n, ts, 1046.5, 2500, 9000, decay=(0.03, 0.1))
    y = _mix([(suck, 0, 0), (dsp.fade(swell, 0, 0.003), 0, -12 + k.lo), (_pop(r, k), pre, -1 + k.h)], T)
    return _out(c, y, _norm(gl) * 10 ** (-16 / 20), room='room', spread=0.4)


@design('teleport/in', kind='zap',
        says=('a magical sparkle appearing with a soft pop', 'a shimmering sparkly whoosh',
              'a soft pop and a burst of glitter'),
        uses=('teleport', 'appear', 'spawn', 'respawn', 'warp in'), domains=DOM, dur=0.7, axes=AXES,
        allows=('ring',))      # pentatonic glitter: its lines lie near the harmonics of one low note
def teleport_in(c, r):
    """appearing, the reverse gesture: the pop first, then air pushed outward (a noise band falling and dying) and
    a glitter that spreads wider in the stereo field as it fades"""
    k = _k(c)
    T = 0.72 * k.L * k.ring ** 0.4
    n = dsp.n_of(T)
    t = _t(n)
    burst = dsp.svf(_air(T, r, center=2000, flutter=0.3), np.geomspace(5000 * k.f, 500 * k.f, n), 1.8, 'bandn')
    burst = dsp.fade(_norm(burst * _env(n, 0.006, 0.12 * k.ring)), 0.0, 0.05)
    whoomp = _dull(r, 0.25, 140 * k.f, 0.05)
    m = r.integers(26, 36)
    ts = 0.01 + np.sort(r.uniform(0, 0.75, m) ** 1.6) * 0.6 * k.L
    gl = _sparkle(r, n, ts, 1046.5, 2200, 9000, decay=(0.06, 0.22), spread=1.0, rise=1 - ts / ts.max())
    y = _mix([(_pop(r, k, 600), 0, -3 + k.h), (burst, 0.004, -5), (whoomp, 0, -10 + k.lo)], T)
    return _out(c, y, _norm(gl) * np.abs(y).max() * 1.2, room='plate', spread=0.5)


def _pad(r, k, root, seconds, level_db):
    """a soft pad chord under a shimmer (earshot.synth.patches 'warm_pad', short release), as mono"""
    from ..synth import patches       # lazy: only heal/glow and magic/cast use a synth layer
    p = patches.play('warm_pad', [root, root * 1.5, root * 2.5198], vel=0.5, dur=seconds, r=r, bright=-0.2, ring=-1.0,
                     space=-1.0)
    return p.mean(1) * 10 ** (level_db / 20) / (np.abs(p).max() + 1e-12)


@design('heal/glow', kind='rise', says=('a soft rising magical shimmer', 'a gentle sparkling chime that glows upward',
                                       'a warm twinkling swell'),
        uses=('heal', 'healing', 'restore', 'potion', 'regenerate'), domains=DOM, dur=1.0, axes=AXES,
        allows=('rising_glide',))      # the shimmer climbs the scale on purpose: rising reads as restoring
def heal_glow(c, r):
    """restoring: a run of bar-like sparkle grains on a major pentatonic climbing from 1.2 to 5 kHz (density
    swelling then thinning), a soft pad chord fading in under it, and a breath of air"""
    k = _k(c)
    T = 1.05 * k.L * k.ring ** 0.3
    n = dsp.n_of(T)
    m = r.integers(30, 42)
    u = np.sort(r.beta(2.0, 2.4, m))
    ts = 0.03 + u * 0.75 * k.L
    root = 523.25 * k.f ** 0.3
    sp = _sparkle(r, n, ts, root, 1100 * k.f ** 0.3, 5200 * k.f ** 0.3, decay=(0.1, 0.3), glide=0.015, rise=u)
    pad = _pad(r, k, root / 2, 0.6 * T, 0.0)
    pad = dsp.fade(pad[:n] * np.sin(0.5 * np.pi * np.minimum(_t(min(n, len(pad))) / 0.25, 1)), 0.0, 0.2)
    air = _burst(r, T, 3000, 11000, 0.4, attack=0.3)
    y = _mix([(pad, 0, -10 + k.lo), (air, 0, -22 + k.h)], T)
    return _out(c, y, _norm(sp) * 10 ** (-2 / 20), room='hall', spread=0.4)


@design('magic/cast', kind='rise',
        says=('an airy magical swell with sparkles ending in a soft burst',
              'a spell being cast with a shimmering whoosh', 'a mystical rushing rise and release'),
        uses=('magic', 'spell', 'cast', 'enchant', 'summon'), domains=DOM, dur=1.1, axes=AXES,
        allows=('rising_glide', 'pre_swell'))   # a swell into a release is the gesture of a cast
def magic_cast(c, r):
    """casting: air gathering (a breath band rising in level and centre, with sparkle grains thickening), a soft pad
    under it, and the release: a falling air burst, a low soft whoomp and a spray of grains"""
    k = _k(c)
    T = 1.15 * k.L * k.ring ** 0.3
    n = dsp.n_of(T)
    sw = 0.55 * k.L
    m = dsp.n_of(sw)
    t = _t(m)
    swell = dsp.svf(_air(sw, r, center=1200, flutter=0.4), np.geomspace(450 * k.f, 2800 * k.f, m), 1.6, 'bandn')
    swell = dsp.fade(_norm(swell * np.sin(0.5 * np.pi * t / sw) ** 2.5), 0.0, 0.01)
    rel = dsp.n_of(T - sw)
    burst = dsp.svf(_air(T - sw, r, center=1500, flutter=0.3), np.geomspace(3500 * k.f, 500 * k.f, rel), 1.2, 'bandn')
    burst = dsp.fade(_norm(burst * _env(rel, 0.004, 0.14 * k.ring)), 0.0, 0.1)
    whoomp = _thump(r, 0.35, 170 * k.f, 95 * k.f, 0.04, 0.08, attack=0.004, harm=0.2)
    root = 587.33 * k.f ** 0.3
    pad = _pad(r, k, root / 2, 0.5 * k.L, 0.0)[:dsp.n_of(0.8 * k.L)]
    pad = dsp.fade(pad, 0.0, 0.15)
    a = _times(r, lambda tt: 12 + 60 * tt / sw, 0.05, sw, 72)
    b = sw + np.sort(r.uniform(0, 0.25, r.integers(10, 16)) ** 1.4)
    ts = np.concatenate([a, b])
    sp = _sparkle(r, n, ts, root, 1800, 8000, decay=(0.05, 0.2), rise=np.minimum(ts / sw, 1.0) * 0.8)
    y = _mix([(swell, 0, -2), (burst, sw, 0 + k.h), (whoomp, sw, -8 + k.lo), (pad, 0.05, -12)], T)
    return _out(c, y, _norm(sp) * 10 ** (-8 / 20), room='hall', spread=0.4)


# ------------------------------------------------------------------------------------------------ ground
@design('rumble/quake', kind='texture', says=('a low rumbling earthquake', 'a deep ground rumble with rattling',
                                             'the earth shaking'),
        uses=('rumble', 'earthquake', 'quake', 'tremor', 'ground shake'), domains=DOM, dur=2.0, axes=AXES, loop=True)
def rumble_quake(c, r):
    """the ground shaking, as a seamless loop: low noise (25-200 Hz) heaving in slow waves, and on it what makes a
    rumble a quake and not a band of noise: irregular low events (rocks and beams shifting: dull knocks at
    50-250 Hz, a few ringing in the structure) and loose things chattering (a window, a shelf of dishes: each an
    object with its own few modes, knocked 4-12 times at 9-25 Hz while the shaking lifts it), all clustered where
    the shaking is strongest; every event wraps round the loop"""
    tex = _tex()
    k = _k(c)
    L = c.seconds(2.0)
    n = dsp.n_of(L)
    heave = 1 + _cmod(r, n, (0.7, 1.3, 2.1, 3.4), 0.6)
    low = _cnoise(r, n, 25 * k.f, 200 * k.f, tilt=-4.0) * heave
    grind = _cnoise(r, n, 120 * k.f, 2000 * k.f, tilt=-4.5) * heave ** 2
    hv = heave / heave.max()

    def when(rate):          # Poisson times over the loop, thinned by the shaking (louder where it heaves)
        t = r.uniform(0, L, r.poisson(rate * L))
        return t[r.uniform(size=len(t)) < np.interp(t, _t(n), hv) ** 2]
    ev = np.zeros((n, 2))
    tk = when(14.0 * (1 + 0.5 * k.e))                     # rocks and beams shifting
    a = _power(r, len(tk), 0.1, 1.0, 1.8) * np.interp(tk, _t(n), hv)
    pan = r.uniform(-0.8, 0.8, len(tk))
    tex.bursts(ev, tk, 3.0 * a, pan, _logu(r, 50, 250, len(tk)) * k.f, r.uniform(1.2, 3.0, len(tk)),
               r.uniform(0.004, 0.02, len(tk)), r, 0.8)
    big = a > 0.4
    tex.rings(ev, tk[big], 0.6 * a[big], pan[big], _logu(r, 70, 300, big.sum()) * k.f,
              r.uniform(0.03, 0.08, big.sum()), 0.002)
    rat = np.zeros((n, 2))
    for obj in range(4):              # four loose things (a door in its frame, a window, things on a shelf), each
        f0 = _logu(r, 200, 700, 1)[0] if obj < 2 else _logu(r, 900, 2500, 1)[0]      # knocking in bursts: dull,
        p0 = r.uniform(-0.7, 0.7)                                                     # damped knocks, not pings
        for tb in when(1.6 * (1 + 0.5 * k.e)):
            m = int(r.integers(4, 13))
            th = (tb + np.cumsum(r.uniform(0.04, 0.11, m))) % L
            g = r.uniform(0.3, 1.0, m) * np.sin(np.pi * (np.arange(m) + 0.5) / m)
            for f, w in ((f0, 1.0), (2.3 * f0, 0.5)):
                tex.bursts(rat, th, w * g, p0, f * np.exp(0.05 * r.standard_normal(m)), r.uniform(2.0, 4.0),
                           r.uniform(0.002, 0.005, m), r, 0.3)
    y = fx.decorrelate(low * 10 ** ((0 + k.lo) / 20) + grind * 10 ** (-20 / 20), 0.4)
    top = np.abs(y).max()
    ev = _circ(lambda x: dsp.butter(x, 'low', 700.0), ev)          # a knock in the ground has no top
    y = y + _norm(ev) * top * 10 ** ((-3 + k.lo / 2) / 20) + _norm(rat) * top * 10 ** ((-10 + k.h) / 20)
    return _loop_out(c, y, room='room')


# ------------------------------------------------------------------------------------------------ signals
def _beep(c, r, k, f, hold, release):
    """a soft signal tone: a sine with a little body (2nd and 3rd harmonics, -14 / -24 dB) and a sub-octave for
    weight (size), a 5 ms attack, a pitch settling from +15 cents over 25 ms, a breath of noise on the onset"""
    hold, release = _vary(r, hold, 0.1), _vary(r, release, 0.2)
    T = hold + 6 * release
    n = dsp.n_of(T)
    t = _t(n)
    fi = f * (1 + 0.0087 * np.exp(-t / 0.025)) * (1 + 0.0015 * np.sin(2 * np.pi * r.uniform(4, 6) * t))
    ph = 2 * np.pi * (np.cumsum(fi) - fi[0]) / SR
    e = _env(n, 0.005 / k.tc ** 0.5, release * k.ring, hold)
    tone = (np.sin(ph) + 0.2 * np.sin(2 * ph) * np.exp(-t / 0.15) + 0.06 * np.sin(3 * ph) * np.exp(-t / 0.06)
            + 0.12 * 10 ** (k.lo / 20) * np.sin(0.5 * ph)) * e
    tone = dsp.fade(_norm(tone), 0.0, release)
    tick = _burst(r, 0.02, 1500, 6000, 0.003)
    return _out(c, _mix([(tone, 0, 0), (tick, 0, -28 + k.h)], T), room='room', spread=0.15)


@design('countdown/beep', kind='note', says=('a short soft electronic beep', 'a clean round countdown tone',
                                            'a brief pure tone'),
        uses=('countdown', 'timer', 'race start', 'ready', 'tick'), domains=DOM, dur=0.22, pitch='A5', axes=AXES)
def countdown_beep(c, r):
    """one of the three ready beeps of a race start (_beep): 0.12 s held"""
    k = _k(c)
    return _beep(c, r, k, c.hz(), 0.12, 0.025)


@design('countdown/go', kind='note', says=('a longer higher soft electronic tone', 'a bright held start signal tone',
                                          'a clean sustained beep'),
        uses=('countdown', 'go', 'race start', 'start', 'timer'), domains=DOM, dur=0.65, pitch='E6', axes=AXES)
def countdown_go(c, r):
    """the go of a race start (_beep): a fifth above the ready beeps (A5 -> E6) and held four times as long"""
    k = _k(c)
    return _beep(c, r, k, c.hz(), 0.48, 0.05)


@design('ui/error_buzz', kind='figure', says=('a short low double buzz', 'two soft low buzzing tones',
                                             'a dull two-pulse buzzer'),
        uses=('error', 'not allowed', 'denied', 'blocked', 'wrong answer'), domains=DOM, dur=0.26, pitch='D3',
        axes=AXES)
def error_buzz(c, r):
    """'not allowed': two short pulses of a low buzz (a band-limited square and a slightly detuned saw, low-passed
    near 1 kHz so it is soft, never rasping), the second a semitone lower and softer; a sine under each for body"""
    k = _k(c)
    f0 = c.hz()
    gate, gap = _vary(r, 0.085, 0.06), _vary(r, 0.05, 0.1)
    out = []
    for i, (st, g) in enumerate(((0, 0.0), (-1, -2.5))):
        f = f0 * 2 ** (st / 12)
        T = gate + 0.04
        n = dsp.n_of(T)
        x = dsp.osc(f, 'square', T) + 0.6 * dsp.osc(f * 1.006, 'saw', T, phase=r.uniform())
        x = dsp.svf(x, 1100 * k.f ** 0.5 * 10 ** (0.15 * k.h), 0.8, 'low')
        x += 0.8 * np.sin(2 * np.pi * f * _t(n)) * 10 ** (k.lo / 20)
        e = _env(n, 0.004 / k.tc ** 0.5, 0.012 * k.ring, gate - 0.01)
        out.append((dsp.fade(_norm(x * e), 0.0, 0.03), i * (gate + gap), g))
    return _out(c, _mix(out), room='room', spread=0.1)
