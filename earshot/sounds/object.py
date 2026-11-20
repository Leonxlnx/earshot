"""Everyday object sounds: the small real things interfaces, games, animations and films use as foley and as
skeuomorphic feedback (a coin on a table, a page turned, a light switch, a cork, a clock). Each one should be heard
as a recording of a real object handled by a person, never as a tone or a noise burst with an envelope.

Every design is built from the physics of its parts, each layer shaped on its own and glued by one small room:
    contact     a force pulse whose length follows the hardness of the two surfaces and the speed (Hertz: glass on
                glass a few hundredths of a millisecond, a knuckle on wood a millisecond and more), plus the tick
                of the contact itself (modal.contact_noise)
    body        the modes of what is struck, at the ratios its geometry gives (a disc 1 : 1.73 : 2.33, a bowl or
                a glass rim 1 : 2.3-2.7 : 4.3-5.0, a hand bell) and with the decay its material gives each mode
                (modal.decay_rate), or a dense cloud of short wood / plastic / paper modes; round bodies ring in
                beating pairs that mix differently at every strike point (modal.doublets)
    after       the irregular events real objects have: bounces at intervals shrinking by the restitution, each
                weaker and at another strike point; a coin's spin-down whose wobble speeds up as it settles; the
                second contact of an object rocking onto its base; a spring-loaded clapper bouncing on a bell
    friction    stick-slip trains whose rate follows the gesture (a zipper's teeth, a peeling tape, a cork in the
                neck, a drawer's runners), crackle clouds with power-law sizes (crumpling paper, a tear)
    fluids      Minnaert bubbles (f = 3.26 / radius) with the rising pitch of a bubble near the surface, a
                Helmholtz cavity ringing behind a pop, the air column over a filling glass

    from earshot.sounds import registry, object      # object registers its designs on import
    y, meta = registry.render('coin/drop', seed=3, energy=0.4)
    y, meta = registry.render('clock/ticktock', dur=4.0)       # a loop: exactly 4 s, no seam

The shared axes mean the same everywhere: energy = how hard or how many (a higher drop, a harder press, more
coins), size = a bigger object (lower and longer), bright = more top, hard = harder surfaces (a shorter contact),
ring = a longer ring of the resonant parts, space = more room, width = wider. The poured, pulled and looped ones
also read dur. Seeds are performances: two seeds of a design differ in their timing and strike points, not in
what they are.
"""
from types import SimpleNamespace

import numpy as np
from scipy import signal

from ..forge import dsp, fx
from ..synth import modal, space
from .registry import design

SR = dsp.SR
T60 = modal.T60
AXES = ('energy', 'size', 'bright', 'hard', 'ring', 'space', 'width')
DOM = ('ui', 'game', 'motion')
G = 9.81


# ------------------------------------------------------------------------------------------------ controls
def _k(c):
    """the shared axes as factors: f = frequency factor of size (bigger is lower), L = length factor of size,
    tc = contact-time factor (hard and energy: harder or faster is shorter), ring = factor on ring times,
    h = dB of the contact layer, e = energy in -1..1"""
    e = c.energy
    return SimpleNamespace(e=e, f=c.lerp('size', 1.4, 0.7, 1.0, log=True), L=c.lerp('size', 0.8, 1.3, 1.0, log=True),
                           tc=c.lerp('hard', 2.0, 0.5, 1.0, log=True) * 0.85 ** e,
                           ring=c.lerp('ring', 0.5, 2.0, 1.0, log=True), h=5.0 * c.hard + 2.0 * e)


def _vary(r, x, s):
    """x scaled by a log-normal take-to-take variation of s (a fraction)"""
    return x * float(np.exp(s * r.standard_normal()))


def _out(c, r, y, room='room', spread=0.3, tilt=0.0):
    """mono or stereo layers to the finished take: width, a high shelf for bright (energy: harder is brighter),
    energy as level, the shared room"""
    w = c.lerp('width', 0.0, 2.0, 1.0)
    st = fx.decorrelate(y, min(1.0, spread * w)) if np.ndim(y) == 1 else fx.width(y, w)
    st = _shelf(st, 6.0 * c.bright + 2.0 * c.energy + tilt) * c.db('energy', 6.0)
    return space.room(st, c.room(room), amount=c.space, r=r)


# The takes of one design are matched the way a sound designer matches a set of variations: a take whose peak stands
# more than CREST dB over its momentary-max loudness (an unlucky stack of contacts in phase, one big crackle) has
# those few peaks limited, so every seed lands at the same loudness once it is levelled by its peak. The values are
# the lowest crest of six takes of each design as built (a take is limited by 0-3 dB, on its loudest contacts).
CREST = {'coin/drop': 11.4, 'coin/flip': 10.7, 'coins/handful': 13.6, 'glass/clink': 17.0, 'glass/set': 16.7,
         'cup/set': 16.3, 'paper/page': 12.0, 'paper/card': 18.6, 'paper/crumple': 14.0, 'paper/tear': 15.5,
         'book/close': 16.5, 'cards/shuffle': 13.2, 'dice/roll': 16.8, 'chips/stack': 16.0, 'water/drop': 14.9,
         'water/pour': 9.5, 'bottle/open': 14.8, 'cork/pop': 19.1, 'clock/tick': 20.6, 'switch/light': 23.2,
         'switch/push': 20.8, 'pen/click': 19.0, 'zipper/pull': 12.4, 'keys/jingle': 10.8, 'scissors/snip': 15.5,
         'match/strike': 11.5, 'drawer/slide': 13.7, 'knock/door': 14.8, 'bell/desk': 9.4, 'bell/shop': 5.7,
         'bell/bike': 5.5, 'stamp/press': 18.3, 'tape/peel': 11.4, 'stapler/press': 18.2}


def _crest(y):
    """dB of the sample peak over the momentary-max loudness (LUFS, the loudest 400 ms K-weighted), measured above
    25 Hz as master.finish will play it"""
    from ..synth import master
    y = dsp.butter(y, 'high', 25.0)
    return 20 * np.log10(np.abs(y).max() + 1e-12) - master.loudness(y)


def _design(id, **kw):
    """registry.design, with the take matched to the design's CREST (see above)"""
    def wrap(fn):
        def take(c, r):
            y = fn(c, r)
            top = CREST.get(id)
            cut = 0.0
            for _ in range(6):          # limiting a lone transient also lowers its loudness: approach the target
                over = _crest(y) - top if top else 0.0
                if over < 0.2 or cut > 6.0:
                    break
                step = min(2.5 * over, 6.0 - cut + 0.5)
                pk = 20 * np.log10(np.abs(y).max())
                y = fx.limit(y, ceiling_db=pk - step, lookahead=0.002, release=0.03, true_peak=False)
                cut += step
            return y
        take.__doc__, take.__name__ = fn.__doc__, fn.__name__
        design(id, **kw)(take)
        return take
    return wrap


def _loop(c, r, y, period, room='booth', spread=0.3):
    """a loop of `period` s: the take through _out, then everything that rings past the end (the room's tail too)
    folded back onto the start, so the buffer repeats with no seam and no cut"""
    st = _out(c, r, y, room, spread)
    p = dsp.n_of(period)
    out = np.zeros((p, 2))
    for i in range(0, len(st), p):
        seg = st[i:i + p]
        out[:len(seg)] += seg
    if c.space > -1.0:          # the room's floor (space.tone) runs through the whole loop, circular, not faded
        F = np.fft.rfft(r.standard_normal((p, 2)), axis=0)
        f = np.maximum(np.fft.rfftfreq(p, 1 / SR), 1.0)
        F *= ((f > 120.0) / np.sqrt(f))[:, None]
        fl = np.fft.irfft(F, p, axis=0)
        out += fl / (np.sqrt(np.mean(fl ** 2)) + 1e-12) * np.abs(out).max() * 10 ** (space.TONE_DB / 20)
    return out


def _shelf(y, db, fc=2500.0):
    """first-order high shelf: low pass + G * high pass of the same corner sum to a flat line when G = 1"""
    return y + (10 ** (db / 20) - 1) * dsp.butter(y, 'high', fc, order=1) if db else y


def _onto(y, z, i=0):
    """y with the shorter z added from sample i on (whatever of z runs past the end of y is dropped)"""
    m = max(0, min(len(z), len(y) - i))
    y[i:i + m] += z[:m]
    return y


def _norm(x):
    return x / (np.abs(x).max() + 1e-12)


def _mix(layers, seconds=None):
    """layers (x mono or stereo, t s, gain dB[, peak]), each brought to peak 1 first (unless peak=False: a _bed), into
    one buffer (mono when every layer is, else stereo with the mono ones in the centre); with a length, what runs
    past it is faded over 20 ms"""
    ls = [tuple(L) + (True,) * (4 - len(L)) for L in layers]
    st = any(np.ndim(L[0]) == 2 for L in ls)
    ls = [(dsp.pan_to_stereo(x) if st and np.ndim(x) == 1 else np.asarray(x), t, g, p) for x, t, g, p in ls]
    n = dsp.n_of(seconds) if seconds else max(dsp.n_of(t) * (t > 0) + len(x) for x, t, _, _ in ls)
    out = np.zeros((n, 2) if st else n)
    for x, t, g, p in ls:
        if np.abs(x).max() > 0:
            dsp.place(out, _norm(x) if p else x, t, 10 ** (g / 20))
    return dsp.fade(out, 0.0, 0.02) if seconds else out


def _bed(x):
    """a continuous layer (a rustle, a stream, a hiss) at a steady level: RMS 0.25 over where it sounds, so a rare
    peak of the noise does not set how loud the layer is in a mix; _mix then takes it as it is (peak about 1)"""
    live = np.abs(x) > 1e-4 * np.abs(x).max()
    return x * (0.25 / (np.sqrt(np.mean(x[live] ** 2)) + 1e-12))


def _t(n):
    return np.arange(n) / SR


def _env(n, attack, tau):
    """a raised-cosine attack and an exponential decay with time constant tau s"""
    e = np.exp(-np.maximum(_t(n) - attack, 0.0) / max(tau, 1e-4))
    a = min(n, dsp.n_of(attack))
    if a > 1:
        e[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    return e


def _speed(n, t0, t1, shape=1.5):
    """a gesture's speed 0..1 over n samples: rises from t0 and falls to rest at t1 s (sin^shape)"""
    t = _t(n)
    u = np.clip((t - t0) / max(t1 - t0, 1e-3), 0.0, 1.0)
    return np.sin(np.pi * u) ** shape * ((t > t0) & (t < t1))


def _burst(r, seconds, lo, hi, tau, attack=0.0003, order=2):
    """band-passed noise with a fast attack and an exponential decay, peak 1: a puff, a slap of air, a hiss"""
    n = dsp.n_of(seconds)
    x = dsp.butter(r.standard_normal(n), 'band', (lo, min(hi, 0.45 * SR)), order) * _env(n, attack, tau)
    return dsp.fade(_norm(x), 0.0, min(0.01, 0.3 * seconds))


# ------------------------------------------------------------------------------------------------ bodies
def _cloud(r, lo, hi, count, t60_lo, t60_hi, tilt=-3.0):
    """a dense body (a table top, a door, a plastic case, a book): `count` modes log-uniform in lo..hi Hz whose
    ring time falls from t60_lo at lo to t60_hi at hi (wood, plastic, paper: the upper modes die first); signed
    amplitudes tilted by `tilt` dB per octave. Returns (freq, rate 1/s, amp)"""
    f = np.sort(np.exp(r.uniform(np.log(lo), np.log(hi), int(count))))
    u = np.log(f / lo) / np.log(hi / lo)
    t60 = np.exp(np.log(t60_lo) + u * np.log(t60_hi / t60_lo))
    a = 10 ** (tilt * np.log2(f / lo) / 20) * r.uniform(0.3, 1.0, len(f)) * r.choice([-1.0, 1.0], len(f))
    return f, T60 / t60, a


def _modes(f, t60, a):
    """a body from listed modes: Hz, seconds to fall 60 dB, amplitudes"""
    return (np.asarray(f, np.float64), T60 / np.asarray(t60, np.float64), np.asarray(a, np.float64))


def _join(*bodies):
    return tuple(np.concatenate(v) for v in zip(*bodies))


def _ring(r, body, seconds, tc, vary=0.3):
    """one contact on a body (mono): its modes driven by a force pulse tc s long (modal.ring: a shorter contact
    reaches higher modes); vary moves each mode's weight log-normally (another strike point: +-3 dB at 0.35)"""
    f, rate, a = body
    if vary:
        a = a * np.exp(vary * r.standard_normal(len(a)))
    return modal.ring(f, rate, a, dsp.n_of(seconds), float(np.clip(tc, 2e-5, 0.012))).imag[0]


def _round(m, f0, material, support, n, tc, angle, built, loss=1.0, ring=1.0, knee=None):
    """one strike of a round body (a coin, a glass, a cup, a bell, a dome; mono): the mode set m (earshot.synth.modal)
    on f0 Hz, each mode a beating pair whose two members the strike at `angle` drives differently (modal.doublets:
    the pairs' axes belong to the object, drawn from the seed `built`, so the same coin struck elsewhere sounds
    like itself); damping from the material plus `support` 1/s (a hand, a table, a mounting), times loss, ring
    times x ring; below `knee` Hz the body radiates poorly (modal.radiation: the high modes are lifted)"""
    keep = f0 * m['ratio'] < 0.45 * SR
    m = {k: v[keep] for k, v in m.items()}
    index, detune, weight, _ = modal.doublets(m['split'], m['order'], angle, dsp.rng(built))
    f = f0 * m['ratio'][index] * detune
    rate = modal.decay_rate(f, material, support) * loss * m['damp'][index] / ring
    a = m['gain'][index] * weight
    if knee:
        a = a * modal.radiation(f, knee) / modal.radiation(f0, knee)
    return modal.ring(f, rate, a, n, float(np.clip(tc, 2e-5, 0.012))).imag[0]


def _hits(r, n, body, times, vels, tc, seconds, vary=0.3, pans=0.0, noise=None):
    """contacts on one body into a stereo buffer of n samples: each at relative speed v (level v, contact tc v^-0.2:
    Hertz, a faster contact is shorter and so brighter), at its own strike point (vary), panned; noise = (click,
    grit, ring_s) adds the sound of the contact itself (modal.contact_noise)"""
    out = np.zeros((n, 2))
    times, vels, pans = np.broadcast_arrays(np.atleast_1d(times), np.atleast_1d(vels), np.atleast_1d(pans))
    for t, v, p in zip(times, vels, pans):
        i = dsp.n_of(t) if t > 0 else 0
        if i >= n or v <= 0:
            continue
        m = min(n - i, dsp.n_of(seconds))
        tci = float(np.clip(tc * max(v, 1e-3) ** -0.2, 2e-5, 0.012))
        y = v * _ring(r, body, m / SR, tci, vary)
        if noise:
            z = v * modal.contact_noise(tci, r, 1.5, *noise)
            y[:min(m, len(z))] += z[:m]
        th = (np.clip(p, -1, 1) + 1) * np.pi / 4
        out[i:i + m] += y[:m, None] * (np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2))
    return out


def _bounces(r, e, first, floor=0.012, most=12, spread=0.2):
    """impact times and speeds of a body bouncing to rest: flight times shrink by the restitution e (each flight
    2 e v / g), the speed by e, both perturbed (+-spread: a body that is not a ball lands on another edge each time),
    until a flight is shorter than `floor` s. `first` = the first flight in s"""
    t, v, gap = [0.0], [1.0], first
    while gap > floor and len(t) < most:
        t.append(t[-1] + gap * r.uniform(1 - spread, 1 + spread))
        v.append(v[-1] * e * r.uniform(1 - spread, 1 + 0.6 * spread))
        gap *= e
    return np.array(t), np.array(v)


def _train(r, rate, jitter=0.05, stick=0.0):
    """times (s) of a pulse train whose rate (Hz, one value per sample) follows a gesture: one pulse per cycle of its
    integral, each moved by `jitter` of a period, a share `stick` left out (the surfaces hold for a cycle)"""
    ph = np.cumsum(np.maximum(rate, 0.0) + 1e-9) / SR
    k = int(ph[-1])
    if k < 1:
        return np.zeros(0)
    marks = np.sort(np.clip(np.arange(1, k + 1) + jitter * r.standard_normal(k), 0.0, ph[-1]))
    t = np.interp(marks, ph, _t(len(rate)))
    return t[r.uniform(size=k) >= stick]


def _through(r, n, t, a, irs, spread=0.3):
    """events (times s, amplitudes) each sounding one of the impulse responses `irs` (picked at random: the next
    tooth, the next point of contact), panned within +-spread; stereo. Fast for trains of hundreds of contacts"""
    out = np.zeros((n, 2))
    t, a = np.atleast_1d(t), np.atleast_1d(a)
    i = (t * SR).astype(np.int64)
    ok = (i >= 0) & (i < n)
    pick = r.integers(0, len(irs), len(t))
    th = (r.uniform(-spread, spread, len(t)) + 1) * np.pi / 4
    for j, ir in enumerate(irs):
        s = ok & (pick == j)
        if not s.any():
            continue
        for ch, w in ((0, np.cos(th[s])), (1, np.sin(th[s]))):
            imp = np.bincount(i[s], a[s] * w * np.sqrt(2), minlength=n)[:n]
            out[:, ch] += signal.fftconvolve(imp, ir)[:n]
    return out


def _crackle(r, n, t, a, lo, hi, length=0.0006, spread=0.6, bands=3):
    """sparse crackle events into a stereo buffer (paper, fibres, fizz, sparks): each a fresh noise grain about
    `length` s long with a 0.15 ms rise and an exponential fall, of amplitude a, in one of `bands` log-spaced bands
    of lo..hi Hz. A bigger break releases more energy over a longer time and lower down (a fracture's size sets
    its scale), so big events are longer and fall in the lower bands more often; small ones are short and high.
    Vectorised: hundreds of events cost a few ms"""
    out = np.zeros((n, 2))
    t, a = np.atleast_1d(np.asarray(t, np.float64)), np.atleast_1d(np.asarray(a, np.float64))
    k = (t >= 0) & (t < n / SR)
    t, a = t[k], a[k]
    if not len(t):
        return out
    s = a / a.max()
    L = np.maximum(12, length * SR * (0.4 + 1.2 * s) * r.uniform(0.7, 1.4, len(t))).astype(np.int64)
    j = np.arange(L.max())
    env = (j[None, :] < L[:, None]) * np.minimum(j / 7.0, 1.0)[None, :] * np.exp(-3.0 * j[None, :] / L[:, None])
    g = a[:, None] * env * r.standard_normal(env.shape)
    idx = (t * SR).astype(np.int64)[:, None] + j[None, :]
    ok = idx < n
    th = (r.uniform(-spread, spread, len(t)) + 1) * np.pi / 4
    edges = np.geomspace(lo, min(hi, 0.42 * SR), bands + 1)
    band = np.minimum((bands * r.uniform(size=len(t)) * (1 - 0.6 * s)).astype(np.int64), bands - 1)
    for b in range(bands):
        s = band == b
        if not s.any():
            continue
        bus = np.zeros((n, 2))
        for ch, w in ((0, np.cos(th[s])), (1, np.sin(th[s]))):
            bus[:, ch] = np.bincount(idx[s][ok[s]], (g[s] * w[:, None] * np.sqrt(2))[ok[s]], minlength=n)[:n]
        out += dsp.butter(bus, 'band', (edges[b], edges[b + 1]), 2)
    return out


def _power(r, k, lo, hi, shape=1.4):
    """k sizes with density ~ size^-shape in [lo, hi]: many small, few large (crumpling, shards, fizz)"""
    u = r.uniform(size=int(k))
    p = 1 - shape
    return (lo ** p + u * (hi ** p - lo ** p)) ** (1 / p)


def _tex():
    from ..synth import texture        # lazy: only the friction designs need it
    return texture


def _logu(r, lo, hi, k):
    """k values log-uniform in lo..hi"""
    return np.exp(r.uniform(np.log(lo), np.log(hi), k))


def _poisson(r, rate, t0, t1, peak):
    """Poisson arrivals in [t0, t1) at rate(t) events/s (a function, at most `peak`), by thinning"""
    k = r.poisson(max(peak * (t1 - t0), 0.0))
    t = np.sort(r.uniform(t0, t1, k))
    return t[r.uniform(0, peak, k) < rate(t)]


def _wood(r, k, lo=120.0, hi=3200.0, count=26, t60=(0.12, 0.03), scale=1.0):
    """a wooden table top or desk: the knock of the board (its low, heavily damped plate modes: 300, 850, 2100 Hz at
    q 3.5-5, the table earshot.synth.struck's coin sits on) and a cloud of shorter modes above; scale moves it in
    pitch"""
    fk = np.array([300.0, 850.0, 2100.0]) * scale
    knock = (fk, np.pi * fk / np.array([3.5, 4.5, 5.0]), np.array([1.0, 0.7, 0.45]))
    return _join(knock, _cloud(r, lo * scale, hi * scale, count, t60[0] * k.ring ** 0.5, t60[1], -2.0))


def _bubble(out, t, radius, gain, pan=0.0, rise=0.1, decay=1.0):
    """air bubbles in water into a stereo buffer: each rings at the Minnaert frequency 3.26 / radius Hz (radius m),
    damped at 0.043 f + 0.0014 f^1.5 per second (van den Doel 2005; decay scales the time constant) and rising in
    pitch as it nears the surface, f (1 + rise * damping * t)"""
    from ..synth import texture
    f = 3.26 / np.asarray(radius, np.float64)
    d = (0.043 * f + 0.0014 * f ** 1.5) / decay
    return texture.rings(out, t, gain, pan, f, 1 / d, 0.25 / f, rise * d, wrap=False)


# ------------------------------------------------------------------------------------------------ coins
COIN_F = 9000.0    # Hz: the (2,0) mode of a coin about 24 mm across (a measured 2 euro coin: 10.4 kHz, 1 : 1.70 : 2.30)


def _coin(r, f0, n, tc, edge, angle, built, ring=1.0, support=1.5):
    """one contact of a coin (mono): a free disc (modal.plate_modes; edge 1 = on the rim, toward 0 = flat on its
    face), brass, ringing about half a second at 9 kHz (the measured coin: 0.4 s), plus the tick of metal on a
    hard surface"""
    y = _round(modal.plate_modes(edge, False, 0.002), f0, 'brass', support, n, tc, angle, built, 1.17, ring, 9000.0)
    z = modal.contact_noise(tc, r, 1.5, 0.12, 0.06, 0.002)
    y[:min(n, len(z))] += 0.6 * z[:n]
    return y


@_design('coin/drop', kind='object',
        says=('a coin dropped on a wooden table, bouncing and spinning to rest',
              'a metal coin clattering on a table and whirring flat', 'a coin falling on a desk with bright clinks'),
        uses=('coin_object', 'coin', 'drop', 'tip', 'purchase'), tags=('metal', 'foley'), domains=DOM, dur=0.95,
        axes=AXES, allows=('tail', 'ring'))     # bounces and spin-down follow the first hit; the coin's one partial
        # under 12 kHz is read as a comb by the cepstral ring detector: a ringing coin, not a fault
def coin_drop(c, r):
    """a coin dropped from h (energy: 5-22 cm) on a wooden table: it lands on its rim, bounces with restitution
    0.5-0.6 (flight 2 e v / g, each landing at another point of the rim, so the beating pairs of the disc mix
    differently), then rolls on its rim and wobbles down: the contact point circles ever faster as the coin tilts
    flat (Euler's disk: Moffatt's (t0 - t)^(-1/6); the exponent is debated, a 1/3 power gives the audible rise
    of a short spin), and it lands flat with a last clack while the table damps the ring"""
    k = _k(c)
    f0, built = _vary(r, COIN_F, 0.04) * k.f, int(r.integers(1 << 30))
    v0 = np.sqrt(2 * G * 0.1 * 2.2 ** k.e)
    e = r.uniform(0.5, 0.62)
    tb, vb = _bounces(r, e, 2 * e * v0 / G, floor=0.016, spread=0.25)
    t0 = tb[-1] + r.uniform(0.02, 0.04)
    spin = r.uniform(0.35, 0.5) * k.L * k.ring ** 0.3
    u = np.linspace(0.0, 1.0, dsp.n_of(spin))
    tw = _train(r, r.uniform(11.0, 15.0) * (1.0 - 0.985 * u) ** (-1 / 3), 0.1)
    vw = vb[-1] * 0.9 * (1 - 0.55 * tw / spin) * np.exp(0.25 * r.standard_normal(len(tw)))
    flat = t0 + spin
    times = np.concatenate([tb, t0 + tw, [flat]])
    vels = np.concatenate([vb, vw, [0.7 * vb[-1]]])
    edges = np.concatenate([r.uniform(0.55, 1.0, len(tb)), np.full(len(tw), 0.95), [0.35]])
    angles = np.concatenate([r.uniform(0, 2 * np.pi, len(tb)), r.uniform(0, 2 * np.pi) + np.cumsum(
        r.uniform(0.6, 1.2, len(tw))), [r.uniform(0, 2 * np.pi)]])      # rolling, the contact walks round the rim
    n = dsp.n_of(flat + 0.6 * k.ring)
    coin = np.zeros(n)
    for t, v, ed, an in zip(times, vels, edges, angles):
        i = dsp.n_of(t)
        coin[i:] += v * _coin(r, f0, n - i, 6e-5 * k.tc * v ** -0.2, ed, an, built, k.ring)
    coin *= modal.damper(n, flat + 0.002, 0.05 * k.ring)          # lying flat, the table takes the ring
    roll = np.concatenate([np.ones(len(tb)), np.full(len(tw), 0.6), [1.2]])
    table = _hits(r, n, _wood(r, k), times, vels * roll, 1.5e-4 * k.tc, 0.25, 0.3)[:, 0]
    return _out(c, r, _mix([(coin, 0, 0), (table, 0, -8)]), 'room', 0.25)


@_design('coin/flip', kind='object',
        says=('a coin flicked into the air with a thumb, a bright ring wobbling as it spins',
              'a spinning coin shimmering in the air and caught in a hand', 'a coin toss, a metallic ting and whirr'),
        uses=('coin_object', 'coin', 'flip', 'toss', 'decide'), tags=('metal', 'foley'), domains=DOM, dur=0.95,
        axes=AXES, allows=('tail',))     # the catch is a second event after the flick
def coin_flip(c, r):
    """a coin tossed with a thumb: the nail strikes the rim hard (a few hundredths of a millisecond), the thumb snaps
    off the finger, and the coin rings freely in the air while it spins at 15-25 revolutions per second: its face
    turns to the listener twice per turn, so the ring swells and fades at twice the spin (the whirr), with a faint
    edge-on rush of air; then a palm catches it: a soft slap that damps the ring"""
    k = _k(c)
    f0, built = _vary(r, COIN_F, 0.04) * k.f, int(r.integers(1 << 30))
    air = r.uniform(0.5, 0.65) * k.L * k.ring ** 0.2
    n = dsp.n_of(air + 0.35)
    tc = 3e-5 * k.tc
    ring = _coin(r, f0, n, tc, 0.98, r.uniform(0, 2 * np.pi), built, 1.6 * k.ring, 0.3)
    t = _t(n)
    rev = r.uniform(15.0, 22.0) * (1 + 0.15 * k.e) * (1 - 0.2 * np.minimum(t / air, 1.0))
    face = np.abs(np.cos(2 * np.pi * np.cumsum(rev) / SR + r.uniform(0, np.pi)))
    ring *= (0.3 + 0.7 * face) * modal.damper(n, air, 0.03)
    whirr = dsp.butter(r.standard_normal(n), 'band', (2500.0, 7000.0)) * face ** 2 * _speed(n, 0.0, air, 0.3)
    snap = _burst(r, 0.03, 900.0, 5000.0, 0.003)
    palm = _ring(r, _cloud(r, 140.0, 900.0, 10, 0.05, 0.02), 0.12, 0.004, 0.2)
    slap = _burst(r, 0.06, 150.0, 2500.0, 0.008, 0.002)
    y = _mix([(ring, 0, 0), (snap, 0.002, -14 + k.h), (whirr, 0, -30), (palm, air, -12), (slap, air, -14)])
    return _out(c, r, y, 'room', 0.3)


@_design('coins/handful', kind='object',
        says=('a handful of coins poured onto a table, jingling and clattering',
              'many metal coins clinking together in a pile', 'loose change spilling out of a purse'),
        uses=('coin', 'coin_object', 'purchase', 'reward', 'win', 'money'), tags=('metal', 'foley'), domains=DOM,
        dur=1.0, axes=AXES, allows=('tail', 'stacked', 'ring'))     # many contacts; ringing coins read as a comb
def coins_handful(c, r):
    """coins poured from a hand: 8-25 of them (energy) arrive front-loaded over half a second, each a coin of its
    own size (6-14 kHz) striking another coin or the pile at a random point of its rim, sizes of impact following
    a power law (a few loud, many soft); most bounce once more off a neighbour; coins lying on coins ring short
    (0.1-0.2 s), and the pile knocks the table under them"""
    k = _k(c)
    count = int(round(22 * 1.6 ** k.e))
    span = 0.5 * k.L
    t = np.sort(span * r.uniform(0, 1, count) ** 1.7)
    t[0] = 0.0
    v = _power(r, count, 0.15, 1.0, 1.6) * (1 - 0.45 * t / span)
    v[0] = max(v[0], 0.7)
    again = r.uniform(size=count) < 0.8
    t = np.concatenate([t, t[again] + r.uniform(0.006, 0.03, again.sum())])
    v = np.concatenate([v, v[again] * r.uniform(0.2, 0.5, again.sum())])
    n = dsp.n_of(span + 0.45 * k.ring + 0.1)
    out = np.zeros((n, 2))
    for ti, vi in zip(t, v):
        i = dsp.n_of(ti)
        f0 = COIN_F * k.f * 2 ** r.uniform(-1.0, 0.3)
        y = vi * _coin(r, f0, n - i, 5e-5 * k.tc * vi ** -0.2, r.uniform(0.4, 0.9), r.uniform(0, 2 * np.pi),
                       int(r.integers(1 << 30)), k.ring, r.uniform(30.0, 70.0))
        out[i:] += dsp.pan_to_stereo(y, r.uniform(-0.5, 0.5))
    pile = _hits(r, n, _wood(r, k, 200.0, 2500.0), t, 0.5 * v, 3e-4 * k.tc, 0.15, 0.4, r.uniform(-0.4, 0.4, len(t)))
    return _out(c, r, _mix([(out, 0, 0), (pile, 0, -6)]), 'room', 0.3)


# ------------------------------------------------------------------------------------------------ glass and china
GLASS_F = 1150.0   # Hz: the (2,0) rim mode of a stemmed wine glass (earshot.synth.struck's glass_clink sits at D6)


@_design('glass/clink', kind='object',
        says=('two wine glasses clinking together in a toast', 'a bright glass clink that rings on',
              'cheers, glasses touching with a crystal ting'),
        uses=('glass_object', 'cheers', 'propose a toast', 'celebration', 'success'),
        tags=('glass', 'foley'), domains=DOM,
        dur=0.9, axes=AXES)
def glass_clink(c, r):
    """two stemmed glasses touched rim to rim: one contact of a few hundredths of a millisecond (glass on glass)
    drives every rim mode of both alike (modal.rim_modes, fall 0); the second glass is a little higher and is
    struck on its side; held by the stems and with wine in them, they ring about a second. Often the glasses touch
    a second time as they part: a soft echo of the clink 20-50 ms later, at other points of both rims"""
    k = _k(c)
    f1 = _vary(r, GLASS_F, 0.03) * k.f
    f2 = f1 * 2 ** (r.uniform(1.0, 2.6) / 12)
    b1, b2 = int(r.integers(1 << 30)), int(r.integers(1 << 30))
    n = dsp.n_of(1.5 * k.ring + 0.1)
    tc = 3e-5 * k.tc
    m = modal.rim_modes('glass', 0.15, 0.0015, 0.0)
    touch = [(0.0, 1.0)] + ([(r.uniform(0.02, 0.05), r.uniform(0.12, 0.28))] if r.uniform() < 0.75 else [])
    out = np.zeros((n, 2))
    for t, v in touch:
        i = dsp.n_of(t)
        tv = tc * v ** -0.2
        a = _round(m, f1, 'glass', r.uniform(0.6, 1.6), n - i, tv, r.uniform(0, 2 * np.pi), b1, 1.0, k.ring, 2 * f1)
        b = _round(m, f2, 'glass', r.uniform(0.6, 1.6), n - i, tv, r.uniform(0, 2 * np.pi), b2, 1.0, k.ring, 2 * f2)
        z = modal.contact_noise(tv, r, 1.5, 0.15, 0.05, 0.002)[:n - i]
        a[:len(z)] += 0.5 * z
        out[i:] += v * (dsp.pan_to_stereo(a, -0.25) + 0.8 * dsp.pan_to_stereo(b, 0.25))
    return _out(c, r, out, 'room', 0.3)


@_design('glass/set', kind='object',
        says=('a glass set down on a wooden table, a short glassy knock', 'a drinking glass placed on a table',
              'a tumbler tapping down onto wood'),
        uses=('glass_object', 'drop', 'place', 'land', 'tap'), tags=('glass', 'wood', 'foley'), domains=DOM,
        dur=0.45, axes=AXES)
def glass_set(c, r):
    """a tumbler put down: its base lands on one edge and rocks flat 4-12 ms later (sometimes a third tiny
    contact), glass on wood (the wood is the soft side: a tenth of a millisecond). The table knocks, the thick
    base rings briefly high up, and the wall of the glass, struck from its foot (fewer rim modes) and held by a
    hand, rings a few tenths of a second"""
    k = _k(c)
    f0, built = _vary(r, 1600.0, 0.05) * k.f, int(r.integers(1 << 30))
    times = [0.0, r.uniform(0.004, 0.012)] + ([r.uniform(0.02, 0.04)] if r.uniform() < 0.5 else [])
    vels = [1.0, r.uniform(0.45, 0.8)] + [r.uniform(0.1, 0.2)] * (len(times) - 2)
    n = dsp.n_of(0.5 * k.ring + 0.1)
    tc = 1.2e-4 * k.tc
    wall, base = np.zeros(n), np.zeros(n)
    rim = modal.rim_modes('glass', 1.0, 0.0015, 0.7)
    for t, v in zip(times, vels):
        i = dsp.n_of(t)
        wall[i:] += v * _round(rim, f0, 'glass', 14.0, n - i, tc, r.uniform(0, 2 * np.pi), built, 1.0, k.ring)
        base[i:] += v * _round(modal.plate_modes(0.7, False, 0.002), 3.4 * f0, 'glass', 45.0, n - i, tc,
                               r.uniform(0, 2 * np.pi), built + 1, 1.0, k.ring)
    table = _hits(r, n, _wood(r, k), times, vels, tc, 0.25, 0.3)[:, 0]
    y = _mix([(table, 0, 0), (wall, 0, -5), (base, 0, -9 + k.h)])
    return _out(c, r, y, 'room', 0.25)


@_design('cup/set', kind='object',
        says=('a ceramic cup set down on a saucer with a clink', 'a porcelain teacup clattering onto its saucer',
              'china clinking, a coffee cup placed on a plate'),
        uses=('drop', 'place', 'land', 'tap', 'cafe'), tags=('ceramic', 'foley'), domains=DOM, dur=0.35, axes=AXES,
        allows=('ring',))      # the saucer's few strong rim partials read as a comb to the cepstral detector
def cup_set(c, r):
    """a cup lowered onto its saucer: the foot ring lands on one side, rocks onto the other 8-20 ms later and often
    rattles once more; porcelain on porcelain (a contact of a few hundredths of a millisecond: a bright clink).
    The saucer is a plate struck near its middle, resting on the table on its own foot (porcelain: a glass-like
    material with more loss, so it rings a few tenths of a second); the cup, struck at its foot and held by its
    handle, rings shorter; the table under them knocks faintly"""
    k = _k(c)
    fs, fc = _vary(r, 1300.0, 0.06) * k.f, _vary(r, 2100.0, 0.06) * k.f
    bs, bc = int(r.integers(1 << 30)), int(r.integers(1 << 30))
    times = [0.0, r.uniform(0.008, 0.02)] + ([r.uniform(0.03, 0.055)] if r.uniform() < 0.6 else [])
    vels = [1.0, r.uniform(0.4, 0.7)] + [r.uniform(0.15, 0.3)] * (len(times) - 2)
    n = dsp.n_of(0.55 * k.ring + 0.1)
    tc = 4e-5 * k.tc
    saucer, cup = np.zeros(n), np.zeros(n)
    plate, bowl = modal.plate_modes(0.55, False, 0.003), modal.rim_modes('bowl', 1.0, 0.003, 0.7)
    for t, v in zip(times, vels):
        i, tv = dsp.n_of(t), tc * v ** -0.2
        saucer[i:] += v * _round(plate, fs, 'glass', 5.0, n - i, tv, r.uniform(0, 2 * np.pi), bs, 1.8, k.ring, 2500)
        cup[i:] += v * _round(bowl, fc, 'glass', 18.0, n - i, tv, r.uniform(0, 2 * np.pi), bc, 1.8, k.ring)
        z = modal.contact_noise(tv, r, 1.5, 0.12, 0.05, 0.0015)[:n - i]
        saucer[i:i + len(z)] += 0.4 * v * z
    table = _hits(r, n, _wood(r, k), times, 0.3 * np.array(vels), 3e-4, 0.2, 0.3)[:, 0]
    return _out(c, r, _mix([(saucer, 0, 0), (cup, 0, -4), (table, 0, -14)]), 'room', 0.25)


# ------------------------------------------------------------------------------------------------ paper and cards
def _rustle(r, n, v, lo=800.0, hi=9000.0, grain=40.0):
    """paper rubbing on paper: noise whose level goes as speed^1.5 and whose top opens with speed, roughened by a
    fast random flutter (the fibres catch and let go): a texture, not a hiss"""
    x = dsp.butter(r.standard_normal(n), 'high', lo)
    x = dsp.svf(x, np.clip(hi * (0.3 + 0.7 * v), 1200.0, 0.42 * SR), 0.7, 'low')
    return x * v ** 1.5 * np.exp(0.6 * dsp.wander(n / SR, r, grain)[:n])


def _breaks(r, v, rate, lo=1e-3, shape=1.4):
    """the crackle of a deforming sheet: Poisson events at rate * v(t) per s (v the deformation speed, an array),
    energies with density E^-shape (crumpled paper: 1.3-1.6, Houle & Sethna 1996); returns (times s, amplitudes)"""
    n = len(v)
    t = _poisson(r, lambda tt: rate * v[np.minimum((tt * SR).astype(np.int64), n - 1)], 0.0, n / SR, rate)
    a = np.sqrt(_power(r, len(t), lo, 1.0, shape))
    return t, np.minimum(a / (np.quantile(a, 0.97) if len(a) else 1.0), 2.0)    # a microphone's few loudest snaps


@_design('paper/page', kind='object',
        says=('a page of a book being turned, a papery swish and flap', 'paper rustling as a page flips over',
              'turning a page of a magazine'),
        uses=('paper', 'swipe', 'next', 'page turn', 'read'), tags=('paper', 'foley'), domains=DOM, dur=0.6,
        axes=AXES, allows=('clicks',))       # the crackles of a bending sheet are single sharp events
def paper_page(c, r):
    """a page turned: the corner peeled up (a few crackles), the sheet bending over fast (rustle at speed^1.5 with
    its top opening as it speeds up, crackles at a rate that follows the bending, the air it pushes), and the
    page settling on the others: a soft slap of air and a dull knock of the book under it"""
    k = _k(c)
    T = _vary(r, 0.4, 0.05) * k.L
    n = dsp.n_of(T + 0.2)
    v = _speed(n, 0.0, T, 1.2) * np.clip(_t(n) / T, 0, 1) ** 0.4          # it speeds up as it goes over
    t, a = _breaks(r, v, 260.0 * 1.4 ** k.e)
    crack = _crackle(r, n, t, a, 1200.0, 9000.0, 0.0005)
    air = dsp.butter(r.standard_normal(n), 'band', (150.0, 900.0)) * v ** 2
    land = _burst(r, 0.06, 300.0, 4000.0, 0.006, 0.001)
    book = _ring(r, _cloud(r, 90.0 * k.f, 700.0 * k.f, 12, 0.05, 0.015), 0.1, 0.003)
    y = _mix([(_bed(_rustle(r, n, v)), 0, -4, False), (crack, 0, 0 + k.h), (air, 0, -12), (land, T, -6),
              (book, T, -12)])
    return _out(c, r, y, 'booth', 0.4)


@_design('paper/card', kind='object',
        says=('a playing card dealt onto a table, a quick flick and a flat slap',
              'a card snapped off the deck and landing on the felt', 'a single card tossed down on a table'),
        uses=('paper', 'card', 'deal', 'tap', 'swipe'), tags=('paper', 'foley'), domains=DOM, dur=0.35, axes=AXES)
def paper_card(c, r):
    """one card dealt: it slides off the deck (25-45 ms of fine friction crackle), snaps free of the thumb (the
    stiff card's few short modes and a click), flutters through the air spinning (a faint swish whose level
    swings at the spin rate) and lands flat: the air cushion under it squeezed out in a 2-3 ms slap, the card and
    the table knocking, a short slide to rest"""
    k = _k(c)
    n0 = dsp.n_of(0.05)
    sl = _vary(r, 0.035, 0.15)
    t, a = _breaks(r, _speed(n0, 0.0, sl, 0.7), 1500.0)
    slide = _crackle(r, n0, t, a, 2000.0, 9000.0, 0.0003)
    card = _cloud(r, 250.0 * k.f, 3000.0 * k.f, 12, 0.03, 0.01)
    snap = _onto(_ring(r, card, 0.05, 3e-4 * k.tc), 0.4 * modal.contact_noise(2e-4, r, 1.5, 0.2, 0.1, 0.001))
    fly = _vary(r, 0.11, 0.15) * k.L
    m = dsp.n_of(fly)
    swish = dsp.butter(r.standard_normal(m), 'band', (1000.0, 4000.0)) * (
        0.5 + 0.5 * np.abs(np.sin(2 * np.pi * r.uniform(18.0, 30.0) * _t(m)))) * _speed(m, 0.0, fly, 0.5)
    t1 = sl + fly
    slap = _burst(r, 0.03, 400.0, 7000.0, 0.0025, 0.0005)
    body = _onto(0.8 * _ring(r, _wood(r, k), 0.15, 0.002), _ring(r, card, 0.06, 0.0015))
    n2 = dsp.n_of(0.05)
    rest = _rustle(r, n2, _speed(n2, 0.0, _vary(r, 0.035, 0.2), 1.0), 1500.0)
    y = _mix([(slide, 0, -6), (snap, sl, -8 + k.h), (swish, sl, -22), (slap, t1, 0 + k.h), (body, t1, -3),
              (rest, t1 + 0.003, -14)])
    return _out(c, r, y, 'room', 0.3)


@_design('paper/crumple', kind='object',
        says=('a sheet of paper being crumpled into a ball', 'crinkling crackling paper squeezed in a hand',
              'a paper wrapper scrunched up'),
        uses=('paper', 'delete', 'discard', 'trash', 'crumple'), tags=('paper', 'foley'), domains=DOM, dur=1.0,
        axes=AXES)
def paper_crumple(c, r):
    """paper crumpled by a hand: two to four squeezes (energy), each a burst of deformation; the sheet answers
    with sparse crackle events (Poisson at a rate following the squeeze, energies with density E^-1.4: a few loud
    snaps among many ticks, Houle & Sethna 1996), the big ones with a lower crunch of the folding sheet, over a
    rustle of paper sliding on paper"""
    k = _k(c)
    T = 0.95 * k.L
    n = dsp.n_of(T + 0.1)
    v = np.zeros(n)
    count = int(np.clip(round(3 + k.e * 1.2), 2, 4))
    for i in range(count):
        t0 = T * (i / count) + r.uniform(0.0, 0.06)
        v = np.maximum(v, _speed(n, t0, t0 + _vary(r, 0.8 * T / count, 0.15), 0.8) * r.uniform(0.6, 1.0))
    t, a = _breaks(r, v, 900.0 * 1.4 ** k.e)
    crack = _crackle(r, n, t, a, 1000.0, 9000.0, 0.0008)
    big = a > 0.45
    crunch = _crackle(r, n, t[big], a[big], 250.0, 1500.0, 0.002, 0.4, 2)
    y = _mix([(crack, 0, 0 + k.h), (crunch, 0, -8), (_bed(_rustle(r, n, v, 600.0)), 0, -12, False)])
    return _out(c, r, y, 'booth', 0.4)


@_design('paper/tear', kind='object',
        says=('a sheet of paper being torn in two', 'paper ripping with a raspy tearing sound', 'a quick paper rip'),
        uses=('paper', 'delete', 'discard', 'rip', 'tear'), tags=('paper', 'foley'), domains=DOM, dur=0.6,
        axes=AXES, allows=('clicks',))     # a tear is a cloud of fibre breaks: clicks are the sound
def paper_tear(c, r):
    """a tear: the front does not run smoothly, it advances in stick-slip jumps, and every jump snaps a bundle of
    fibres at once: a short grain of crackle within a millisecond, and the force on the sheet drops in a step
    (texture.stick_slip). The jumps come at 90-350 per second, faster as the tear speeds up, very irregular and now
    and then caught for a few ms, so the rip has a rough pitch that climbs with the speed; between them single
    fibres break in a fine crackle. The steps ring the sheet (broad paper resonances at 1-5 kHz) and set the two
    halves fluttering (a low papery flap that follows the jumps); a last bigger crack where it separates, and the
    halves swinging apart"""
    k = _k(c)
    T = _vary(r, 0.42, 0.05) * k.L * 1.15 ** -k.e
    n = dsp.n_of(T + 0.18)
    t_ = _t(n)
    v = np.clip(0.3 + 0.7 * (t_ / T) ** 0.6, 0, 1) * _speed(n, 0.0, T, 0.25)
    rate = (90.0 + 260.0 * v * 1.15 ** k.e) * np.exp(0.15 * dsp.wander(n / SR, r, 15.0)[:n]) * (v > 0.05) / k.L
    x, ts = _tex().stick_slip(r, rate, jitter=0.3, hold=0.04, hold_s=(0.004, 0.02), rough=0.7, edge=5e-5)
    snap = dsp.butter(x, 'high', 600.0)
    sheet = sum(g * dsp.svf(snap, f, q, 'bandn') for f, q, g in
                zip(_vary(r, 1.0, 0.1) * np.array([1300.0, 2600.0, 4300.0]) * k.f ** 0.5, (1.5, 2.0, 2.0),
                    (1.0, 0.8, 0.5)))
    tb = np.repeat(ts, 3) + r.exponential(0.0008, 3 * len(ts))          # a bundle: a few fibres within a ms
    bundles = _crackle(r, n, tb, np.exp(0.6 * r.standard_normal(len(tb))), 800.0, 9000.0, 0.0008)
    t, a = _breaks(r, v, 900.0 * 1.3 ** k.e, shape=1.6)
    fibres = _crackle(r, n, t, a, 1500.0, 10000.0, 0.0003)
    flap = dsp.svf(dsp.butter(x, 'low', 900.0), _vary(r, 350.0, 0.2) * k.f, 1.2, 'bandn')
    end = _burst(r, 0.02, 1500.0, 9000.0, 0.002, 0.0003)
    m = dsp.n_of(0.15)
    part = dsp.butter(r.standard_normal(m), 'band', (300.0, 2500.0)) * _speed(m, 0.0, 0.15, 1.0)
    y = _mix([(bundles, 0, 0 + k.h), (sheet, 0, -8 + k.h), (fibres, 0, -6), (flap, 0, -12), (end, T, -4),
              (part, T, -18)])
    return _out(c, r, y, 'booth', 0.4)



@_design('book/close', kind='object',
        says=('a book closing with a soft thump of pages and cover', 'a heavy book shut on a table',
              'a hardcover book slapped closed'),
        uses=('paper', 'close', 'done', 'book', 'land'), tags=('paper', 'foley'), domains=DOM, dur=0.5, axes=AXES)
def book_close(c, r):
    """a book shut on a table: the cover swings over (the air it pushes), then lands on the pages: a broad slap,
    the air squeezed out from between the pages (a puff that lasts tens of ms), the dull knock of the block of
    paper (very lossy: a few short low modes) and the table under it; the cover settles once more"""
    k = _k(c)
    swing = _vary(r, 0.12, 0.15) * k.L
    m = dsp.n_of(swing)
    air = dsp.butter(r.standard_normal(m), 'band', (100.0, 1200.0)) * np.linspace(0, 1, m) ** 2
    slap = _burst(r, 0.06, 250.0, 6000.0, 0.004 * k.tc ** 0.5, 0.0005)
    puff = _burst(r, 0.12, 150.0, 2500.0, 0.03 * k.L, 0.003)
    block = _ring(r, _cloud(r, 80.0 * k.f, 700.0 * k.f, 40, 0.07 * k.ring, 0.02), 0.25, 0.002 * k.tc)
    table = _ring(r, _wood(r, k), 0.25, 0.003 * k.tc)
    settle = r.uniform(0.02, 0.035)
    y = _mix([(air, 0, -18), (slap, swing, -2 + k.h), (puff, swing, -6), (block, swing, 0), (table, swing, -6),
              (_burst(r, 0.03, 300.0, 3000.0, 0.003), swing + settle, -16)])
    return _out(c, r, y, 'room', 0.3)


@_design('cards/shuffle', kind='object',
        says=('a deck of playing cards riffle shuffled', 'cards fluttering together in a fast riffle',
              'a card shuffle with a bridge and a tap'),
        uses=('card', 'shuffle', 'deal', 'loading', 'paper'), tags=('paper', 'foley'), domains=DOM, dur=1.4,
        axes=AXES, allows=('tail', 'clicks'))       # 52 card flaps are clicks; the bridge and the tap follow
def cards_shuffle(c, r):
    """a riffle: two halves bent under the thumbs release 52 cards in about 0.7 s, at a rate that swells and fades,
    unevenly and mostly alternating between the halves (left, right); every card is a small flap (a short crackle
    and the card's few damped modes). Then the bridge: the cards cascade together in a quick soft burr, and the
    deck is squared with a tap on the table"""
    k = _k(c)
    T = _vary(r, 0.7, 0.08) * k.L
    n = dsp.n_of(T + 0.7)
    sw = _speed(n, 0.0, T, 0.8)
    t = _train(r, sw * 52.0 / (np.sum(sw) / SR), 0.3)
    side = np.cumsum(r.uniform(size=len(t)) < 0.75) % 2 * 2 - 1.0
    card = _cloud(r, 300.0 * k.f, 2500.0 * k.f, 10, 0.02, 0.008)
    irs = []
    for _ in range(4):
        ir = _ring(r, card, 0.03, 2e-4 * k.tc)
        ir = _onto(ir, 0.3 * _burst(r, 0.003, 1500.0, 8000.0, 0.0008, 0.0002) * np.abs(ir).max())
        irs.append(dsp.butter(ir, 'high', 300.0))
    a = np.exp(0.3 * r.standard_normal(len(t)))
    riffle = np.zeros((n, 2))
    for s in (-1.0, 1.0):
        pick = side == s
        riffle += dsp.pan_to_stereo(_through(r, n, t[pick], a[pick], irs, 0.0)[:, 0], 0.35 * s * c.lerp(
            'width', 0.0, 1.5, 1.0))
    bridge_t = T + _vary(r, 0.12, 0.2)
    bd = _vary(r, 0.22, 0.15)
    bv = _speed(n, bridge_t, bridge_t + bd, 0.7)
    tb = _train(r, bv * 52.0 / (np.sum(bv) / SR + 1e-9), 0.4)
    bridge = _through(r, n, tb, 0.35 * np.exp(0.3 * r.standard_normal(len(tb))), irs, 0.2)
    tap = _ring(r, _join(card, _wood(r, k)), 0.2, 0.0015 * k.tc)
    y = _mix([(riffle, 0, 0 + k.h), (_bed(_rustle(r, n, sw, 1000.0)), 0, -16, False), (bridge, 0, -6),
              (tap, bridge_t + bd + r.uniform(0.04, 0.08), -4)])
    return _out(c, r, y, 'room', 0.3)


def _die(r, k, t0, v0):
    """impact times and speeds of one die thrown on a table: a hard first landing, bounces on its corners (flights
    shrinking by 0.65 with +-35 % scatter), often an edge and then a face a few ms apart, then tumbles that slow
    down, and a last rock to rest"""
    t, v = _bounces(r, 0.65, _vary(r, 0.11, 0.2) * k.L ** 0.5, floor=0.03, most=9, spread=0.35)
    pair = r.uniform(size=len(t)) < 0.4
    t = np.concatenate([t, t[pair] + r.uniform(0.003, 0.008, pair.sum())])
    v = np.concatenate([v, v[pair] * r.uniform(0.3, 0.6, pair.sum())])
    end = t.max() + r.uniform(0.04, 0.08)
    t = np.concatenate([t, [end, end + r.uniform(0.02, 0.04)]])
    v = np.concatenate([v, [0.08, 0.04]])
    return t0 + t, v0 * v


@_design('dice/roll', kind='object',
        says=('two dice rolled on a wooden table, clattering and tumbling to a stop',
              'a pair of plastic dice thrown and rattling', 'dice bouncing and rolling on a board game table'),
        uses=('dice', 'roll', 'random', 'game', 'luck'), tags=('plastic', 'wood', 'foley'), domains=DOM, dur=0.55,
        axes=AXES, allows=('tail', 'stacked', 'clicks', 'ring'))      # many hard plastic clacks of two dice: the design
def dice_roll(c, r):
    """two dice: each lands hard, bounces on its corners and tumbles to rest (_die); an acrylic cube is too stiff
    and small to ring in the audible range, so what is heard is the contact (plastic on wood, a few hundredths
    of a millisecond), the die's dense short high modes and the table knocking; once the two dice hit each other"""
    k = _k(c)
    die = _cloud(r, 2500.0 * k.f, 9000.0 * k.f, 10, 0.025, 0.008)
    t1, v1 = _die(r, k, 0.0, 1.0)
    t2, v2 = _die(r, k, r.uniform(0.01, 0.05), r.uniform(0.7, 0.95))
    tx = r.uniform(0.1, 0.3)
    n = dsp.n_of(max(t1.max(), t2.max()) + 0.25)
    tc = 5e-5 * k.tc
    p1, p2 = r.uniform(-0.4, -0.1), r.uniform(0.1, 0.4)
    dice = (_hits(r, n, die, t1, v1, tc, 0.06, 0.4, p1, (0.15, 0.1, 0.001)) +
            _hits(r, n, die, t2, v2, tc, 0.06, 0.4, p2, (0.15, 0.1, 0.001)) +
            _hits(r, n, die, [tx, tx + 0.0004], [0.5, 0.45], 3e-5, 0.06, 0.4, 0.5 * (p1 + p2), (0.2, 0.1, 0.001)))
    table = _wood(r, k)
    knock = (_hits(r, n, table, t1, v1, 1.5e-4 * k.tc, 0.2, 0.3, p1) +
             _hits(r, n, table, t2, v2, 1.5e-4 * k.tc, 0.2, 0.3, p2))
    return _out(c, r, _mix([(dice, 0, 0 + k.h), (knock, 0, -4)]), 'room', 0.3)


@_design('chips/stack', kind='object',
        says=('poker chips clacking as they are stacked', 'clay casino chips dropped onto a pile',
              'a stack of chips clicking together'),
        uses=('chips', 'bet', 'purchase', 'coin', 'add'), tags=('clay', 'foley'), domains=DOM, dur=0.8, axes=AXES,
        allows=('tail', 'stacked', 'ring'))      # several chips, each a contact; one chip's few modes read as a comb
def chips_stack(c, r):
    """chips let go one by one onto a stack (4-8 of them, energy), 60-130 ms apart as the fingers release them;
    each lands flat on the one below with a clack (clay composite: a disc's modes but heavily damped, so it
    clacks and does not ring) and bounces once or twice a few ms later; the stack and the felt under it give
    a short low knock"""
    k = _k(c)
    count = int(round(6 * 1.3 ** k.e))
    t = np.cumsum(np.concatenate([[0.0], r.uniform(0.06, 0.13, count - 1) * k.L]))
    v = r.uniform(0.6, 1.0, count)
    b1 = t + r.uniform(0.004, 0.012, count)
    b2 = b1 + r.uniform(0.003, 0.008, count)
    tt = np.concatenate([t, b1, b2])
    vv = np.concatenate([v, v * r.uniform(0.2, 0.35, count), v * r.uniform(0.05, 0.12, count)])
    n = dsp.n_of(t[-1] + 0.2)
    out = np.zeros(n)
    f0, built = _vary(r, 3300.0, 0.05) * k.f, int(r.integers(1 << 30))
    for ti, vi in zip(tt, vv):
        i = dsp.n_of(ti)
        tc = 6e-5 * k.tc * vi ** -0.2
        y = _round(modal.plate_modes(r.uniform(0.6, 0.9), False, 0.003), f0, 'maple', 10.0, n - i, tc,
                   r.uniform(0, 2 * np.pi), built, 1.0, k.ring)
        z = modal.contact_noise(tc, r, 1.5, 0.12, 0.08, 0.001)[:n - i]
        y[:len(z)] += 0.5 * z
        out[i:] += vi * y
    stack = _hits(r, n, _cloud(r, 700.0 * k.f, 2000.0 * k.f, 10, 0.03, 0.012), t, v, 2e-4, 0.08, 0.3)[:, 0]
    return _out(c, r, _mix([(out, 0, 0 + k.h), (stack, 0, -8)]), 'room', 0.3)


# ------------------------------------------------------------------------------------------------ water
@_design('water/drop', kind='object',
        says=('a single drop of water falling into still water, a liquid plink', 'a water droplet plopping into a bowl',
              'a drip into a sink'),
        uses=('water', 'drop', 'drip', 'pop', 'notify', 'hydration', 'drink logged'),
        tags=('water', 'foley'), domains=DOM, dur=0.3, axes=AXES,
        allows=('rising_glide', 'tail'))    # a bubble's pitch rises as it nears the surface; the jet drop follows
def water_drop(c, r):
    """a drop into still water: the impact itself (a tick of a fraction of a millisecond, high), then 2-5 ms later
    the bubble it entrains rings at its Minnaert frequency (a 2-3 mm bubble: about 1.1-1.6 kHz) with the
    rising pitch of a bubble near the surface (van den Doel 2005; xi 0.15-0.3 here, the bright plink of a
    falling drop), and 70-140 ms later the drop thrown up by the crater's jet falls back: a smaller, higher one"""
    k = _k(c)
    n = dsp.n_of(0.45 * k.L + 0.1)
    out = np.zeros((n, 2))
    rad = _vary(r, 0.0024, 0.15) / k.f
    t1 = r.uniform(0.002, 0.005)
    _bubble(out, t1, rad, 1.0, r.uniform(-0.1, 0.1), r.uniform(0.15, 0.3), 2.0 * k.ring)
    t2 = r.uniform(0.07, 0.14) * k.L
    if r.uniform() < 0.8:
        _bubble(out, t2 + 0.002, rad * r.uniform(0.45, 0.65), r.uniform(0.2, 0.35), r.uniform(-0.3, 0.3),
                r.uniform(0.1, 0.2), 1.5 * k.ring)
    tick = _burst(r, 0.004, 2000.0, 16000.0, 0.0003 * k.tc, 0.0001)
    slosh = dsp.butter(r.standard_normal(n), 'band', (200.0, 1200.0)) * _env(n, 0.01, 0.06)
    y = _mix([(out, 0, 0), (tick, 0, -10 + k.h), (_burst(r, 0.003, 2500.0, 14000.0, 0.0003), t2, -20),
              (slosh, 0, -30)])
    return _out(c, r, y, 'room', 0.3)


@_design('water/pour', kind='object',
        says=('water poured into a glass, the pitch rising as it fills',
              'a drink being poured, liquid gurgling into a cup',
              'pouring water from a bottle into a glass'),
        uses=('water', 'pour', 'fill', 'loading', 'drink'), tags=('water', 'glass', 'foley'), domains=DOM, dur=2.0,
        axes=AXES, allows=('rising_glide', 'tail'))    # the air column shortens as the glass fills: its pitch rises
def water_pour(c, r):
    """a pour of dur s: the stream hitting the water (turbulent noise that wanders with the flow) and entraining
    bubbles (Poisson at a rate following the flow, radii with density r^-2.6 between 0.6 and 5 mm, each a
    Minnaert ring with its rising pitch); all of it heard through the air column over the water, a tube closed by
    the water: f = c / (4 (L + 0.6 a)), rising from about 600 Hz to 1.7 kHz as the glass fills to three quarters;
    the glass rings faintly where the first water hits its bottom, and two or three last drips follow the stream"""
    k = _k(c)
    T = c.seconds(2.0)
    n = dsp.n_of(T + 0.35)
    t_ = _t(n)
    stop = max(0.3, T - 0.15)
    q = np.clip(t_ / 0.1, 0, 1) * np.clip((stop - t_) / 0.15, 0, 1) * (1 + 0.15 * dsp.wander(n / SR, r, 3.0)[:n])
    q = np.maximum(q, 0.0) * 1.3 ** k.e
    jet = dsp.butter(r.standard_normal(n), 'band', (400.0, 6000.0)) * q * np.exp(0.4 * dsp.wander(n / SR, r, 12)[:n])
    bub = np.zeros((n, 2))
    tb = _poisson(r, lambda tt: 260.0 * q[np.minimum((tt * SR).astype(np.int64), n - 1)], 0.0, stop, 400.0)
    rad = _power(r, len(tb), 0.0006, 0.005, 2.6) / k.f ** 0.5
    _bubble(bub, tb, rad, (rad / 0.005) ** 0.7, r.uniform(-0.3, 0.3, len(tb)), 0.1, 1.0)
    H, a = 0.12 * k.L, 0.035                       # m: the glass's height and radius
    fill = 0.75 * np.cumsum(q) / (np.sum(q) + 1e-12)
    col = 343.0 / (4 * (H * (0.97 - fill) + 0.6 * a))
    mono = _norm(jet) * 0.6 + _norm(bub.mean(1))
    tube = dsp.svf(mono, col, 6.0, 'bandn')
    glass = 0.15 * _round(modal.rim_modes('glass', 1.0, 0.0015, 0.7), 1800.0 * k.f, 'glass', 4.0, n, 2e-4,
                          r.uniform(0, 2 * np.pi), int(r.integers(1 << 30)), 1.0, k.ring)
    drips = np.zeros((n, 2))
    td = stop + np.cumsum(r.uniform(0.06, 0.14, int(r.integers(2, 4))))
    _bubble(drips, td, r.uniform(0.0012, 0.002, len(td)) / k.f, r.uniform(0.3, 0.6, len(td)), 0.0, 0.2, 1.5)
    y = _mix([(tube, 0, 0), (mono, 0, -9), (glass, 0, -12), (drips, 0, -12)])
    return _out(c, r, y, 'room', 0.3)


@_design('bottle/open', kind='object',
        says=('a bottle cap prised off with a hiss of gas', 'a beer bottle opened, a metal pop and a fizzy psst',
              'opening a soda bottle with a bottle opener'),
        uses=('open', 'drink', 'bottle', 'unlock', 'fizz'), tags=('glass', 'metal', 'foley'), domains=DOM, dur=0.9,
        axes=AXES, allows=('tail', 'clicks'))       # the cap bounces after the hiss; a fizz is made of clicks
def bottle_open(c, r):
    """an opener on a crown cap: the steel catches the cap's edge (a tick), the cap's crimp bends with a short
    metallic crackle, then the seal breaks: the gas rushes out (a bright hiss, longer with more pressure), the
    headspace rings once as a Helmholtz cavity (a 330 ml bottle with 15 ml of gas under a 17 mm neck: f = c / 2pi
    sqrt(A / V L') = about 1.1 kHz), the cap drops on the table and bounces with small tinny clinks, and the drink
    fizzes (tiny bubbles bursting at the surface)"""
    k = _k(c)
    n = dsp.n_of(0.9 * k.L)
    cap = _cloud(r, 2500.0, 9000.0, 10, 0.06, 0.02)
    tick = _ring(r, cap, 0.08, 3e-5)
    tr = r.uniform(0.03, 0.06)
    tcr = r.uniform(0.0, tr, 30)
    crimp = _crackle(r, n, tcr, np.sqrt(_power(r, 30, 0.01, 1.0, 1.4)), 2000.0, 9000.0, 0.0003)
    # the gas: a jet through the gap under the cap, its noise peaking at St U / D (U falls with the pressure as the
    # headspace empties, so the psst falls from about 6 to 2 kHz as it dies: a jet, not static)
    m = dsp.n_of(0.5 * k.L)
    tg = _t(m)
    tau = _vary(r, 0.05, 0.15) * k.L * 1.4 ** k.e
    p = np.exp(-tg / tau)
    jet = dsp.svf(dsp.butter(r.standard_normal(m), 'high', 700.0), (1800.0 + 4500.0 * p ** 0.5) * k.f ** 0.3, 1.3,
                  'bandn')
    gas = dsp.fade(jet * p ** 0.8 * _env(m, 0.0015, 10.0) * np.exp(0.2 * dsp.wander(m / SR, r, 40.0)[:m]), 0.0, 0.05)
    pulse = _burst(r, 0.003, 200.0, 8000.0, 0.0008, 0.0002)
    m = dsp.n_of(0.06)
    neck = dsp.svf(np.pad(pulse, (0, m - len(pulse))), 1100.0 * k.f, 5.0, 'bandn')
    tb, vb = _bounces(r, 0.45, _vary(r, 0.09, 0.2), floor=0.015, most=6, spread=0.3)
    land = tr + r.uniform(0.15, 0.25)
    clinks = _hits(r, n, cap, land + tb, vb, 4e-5, 0.08, 0.4, r.uniform(-0.3, 0.3), (0.12, 0.08, 0.001))
    thud = _hits(r, n, _wood(r, k, 300.0, 3000.0), land + tb, vb, 1.5e-4, 0.08, 0.3)
    tf = tr + 0.02 + r.exponential(0.25, int(260 * 1.3 ** k.e))
    fizz = _crackle(r, n, tf, np.exp(0.5 * r.standard_normal(len(tf))) * np.exp(-(tf - tr) / 0.4), 3000.0, 12000.0,
                    0.0002, 0.7)
    y = _mix([(tick, 0, -8), (crimp, 0, -12), (gas, tr, -2 + 2 * k.e), (neck, tr, 0 + k.h), (clinks, 0, -8),
              (thud, 0, -14), (fizz, 0, -20)])
    return _out(c, r, y, 'room', 0.3)


@_design('cork/pop', kind='pop',
        says=('a cork popping out of a bottle with a hollow pop', 'a wine bottle uncorked, a squeak and a pop',
              'a champagne cork pop and a hiss'),
        uses=('cork', 'pop', 'celebration', 'open', 'win', 'goal reached', 'champagne'),
        tags=('glass', 'foley'), domains=DOM, dur=0.6,
        axes=AXES, allows=('rising_glide',))     # the cork's squeak climbs as it slides faster: stick-slip
def cork_pop(c, r):
    """a cork eased out: it squeaks in the glass neck (stick-slip at a slip rate climbing from about 120 to 420 Hz as
    it slides faster, with jitter and held moments, through the neck's damped modes), then leaves: a pressure pulse
    of 1-2 ms and the headspace ringing a few cycles as a Helmholtz cavity (q 4-6 at about 520 Hz: bigger
    bottles lower), and the gas hissing out (longer with more pressure: energy)"""
    k = _k(c)
    sq = _vary(r, 0.22, 0.15) * k.L
    m = dsp.n_of(sq)
    u = np.linspace(0, 1, m)
    rate = (120.0 + 300.0 * u ** 1.5) * (1 + 0.1 * dsp.wander(sq, r, 8.0)[:m])
    neck = _cloud(r, 900.0 * k.f, 4000.0 * k.f, 8, 0.03, 0.012)
    irs = [_ring(r, neck, 0.04, 2e-4) for _ in range(3)]
    ts = _train(r, rate, 0.08, 0.1)
    squeak = _through(r, m, ts, (0.4 + 0.6 * np.interp(ts, _t(m), u)) * np.exp(0.2 * r.standard_normal(len(ts))),
                      irs, 0.1)
    fh = _vary(r, 520.0, 0.08) * k.f
    q = r.uniform(4.0, 6.0)
    cavity = _ring(r, _modes([fh, 2.9 * fh], [T60 * q / (np.pi * fh), T60 * q / (np.pi * 2.9 * fh)], [1.0, 0.25]),
                   0.12, 0.0012 * k.tc, 0.0)
    pulse = _burst(r, 0.008, 200.0, 9000.0, 0.0012 * k.tc, 0.0002)
    hiss = _burst(r, 0.4, 1500.0, 9000.0, 0.04 * 1.6 ** k.e, 0.003)
    y = _mix([(squeak, 0, -14), (cavity, sq, 0), (pulse, sq, -4 + k.h), (hiss, sq + 0.002, -10 + 3 * k.e)])
    return _out(c, r, y, 'room', 0.3)


# ------------------------------------------------------------------------------------------------ clocks and pulses
def _clock(r, k):
    """the bodies of one mantel clock, drawn once per take so that every beat is the same clock: the escapement's
    small steel and brass parts (dense high modes, a few tens of ms), the brass movement plate, and the wooden case
    that the movement is screwed into (its low modes are what make a tick a tock)"""
    return {'steel': _cloud(r, 2800.0 * k.f, 9500.0, 10, 0.04, 0.015),
            'plate': _cloud(r, 1100.0 * k.f, 5000.0 * k.f, 14, 0.09 * k.ring, 0.03),
            'case': _cloud(r, 180.0 * k.f, 1800.0 * k.f, 20, 0.12 * k.ring, 0.03, -4.0)}


def _beat(r, k, body, n, pitch=1.0, level=1.0):
    """one beat of an escapement (mono, n samples): a tooth of the escape wheel drops onto the pallet (a hard
    steel-on-steel contact), 2-6 ms later the pallet locks (a second, weaker click), the shock passing into the
    plate and the case; pitch moves the steel parts (the entry and the exit pallet are not the same piece)"""
    out = np.zeros(n)
    lock = r.uniform(0.002, 0.006)
    for t, v in ((0.0, 1.0), (lock, r.uniform(0.3, 0.5))):
        i = dsp.n_of(t) if t else 0
        f, rate, a = body['steel']
        steel = _ring(r, (f * pitch, rate, a), (n - i) / SR, 2.5e-5 * k.tc, 0.15)
        steel = _onto(steel, modal.contact_noise(2.5e-5, r, 1.5, 0.15, 0.08, 0.0008))
        plate = _ring(r, body['plate'], (n - i) / SR, 6e-5 * k.tc, 0.15)
        case = _ring(r, body['case'], (n - i) / SR, 2e-4 * k.tc, 0.15)
        out[i:] += level * v * (_norm(steel) + 0.6 * _norm(plate) + 0.4 * _norm(case))
    return out


@_design('clock/tick', kind='tick',
        says=('one tick of a mechanical clock', 'a single wooden clock tick, a small metallic click in a case',
              'the tick of a mantel clock'),
        uses=('tick', 'timer', 'clock', 'countdown', 'hover'), tags=('metal', 'wood', 'foley'), domains=DOM,
        dur=0.2, axes=AXES)
def clock_tick(c, r):
    """one beat of a mantel clock (_beat): the escapement's double click and the case answering it"""
    k = _k(c)
    n = dsp.n_of(0.2 * k.ring ** 0.5)
    return _out(c, r, _beat(r, k, _clock(r, k), n), 'booth', 0.25)


@_design('clock/ticktock', kind='tick',
        says=('a mechanical clock ticking, tick tock', 'a steady clock ticking in a quiet room',
              'a pendulum clock tick-tocking'),
        uses=('timer', 'clock', 'tick', 'countdown', 'waiting', 'loading'), tags=('metal', 'wood', 'foley'),
        domains=DOM, dur=2.0, axes=AXES, loop=True, allows=('clicks', 'ring'))    # an escapement: clicks, and a beat
def clock_ticktock(c, r):
    """a loop of dur s (2 s: four beats of a mantel clock at 7200 beats an hour; size: a bigger clock beats
    slower, a longcase once a second): tick and tock alternate, made by the two pallets (the tock's steel parts a
    little lower and softer), and no clock is exactly in beat: every second beat comes a few per cent late.
    Every beat is the same clock (_clock), each strike a little different"""
    k = _k(c)
    D = c.seconds(2.0)
    beat = c.lerp('size', 0.33, 1.0, 0.5, log=True)
    nb = max(2, 2 * int(round(D / (2 * beat))))
    beat = D / nb
    body = _clock(r, k)
    late = r.uniform(0.02, 0.05) * beat
    m = dsp.n_of(min(0.25, 0.9 * beat))
    y = np.zeros(dsp.n_of(D) + m)
    for i in range(nb):
        tock = i % 2
        t = 0.03 + i * beat + (late if tock else 0.0) + 0.0004 * r.standard_normal()
        dsp.place(y, _beat(r, k, body, m, 0.88 if tock else 1.0, 0.85 if tock else 1.0), t)
    return _loop(c, r, y, D, 'booth', 0.25)


def _valve(r, k, f, t60, n, tc=0.01):
    """one heart sound (mono): a valve closing sets the blood, the walls and the chest ringing at a few low modes
    (tens of Hz to about 200 Hz, gone in a tenth of a second); soft tissue spreads the onset over ~10 ms"""
    return _ring(r, _modes(np.asarray(f) * k.f, np.asarray(t60) * k.ring ** 0.5, [1.0, 0.8, 0.45, 0.2]),
                 n / SR, tc * k.tc ** 0.5, 0.15)


@_design('heartbeat/pulse', kind='thud',
        says=('a heartbeat, a low lub-dub thumping', 'a beating heart heard up close', 'a slow steady heart pulse'),
        uses=('heartbeat', 'pulse', 'tension', 'suspense', 'alive', 'hurt'), tags=('body', 'foley'), domains=DOM,
        dur=1.0, axes=AXES, loop=True, allows=('stacked',))     # lub and dub are two closures: the heart sound
def heartbeat_pulse(c, r):
    """a loop of dur s at 60 beats a minute (one beat per second of dur): S1, the 'lub' (the mitral and then the
    tricuspid valve closing 20-30 ms apart: two low thuds at 35-190 Hz), and a third of a second later S2, the
    shorter, higher 'dub' (the aortic and pulmonary valves, 20-40 ms apart), as a stethoscope or a film close-up
    hears them; the strength of each beat varies by a fraction of a dB"""
    k = _k(c)
    D = c.seconds(1.0)
    nb = max(1, int(round(D)))
    period = D / nb
    m = dsp.n_of(0.3)
    y = np.zeros(dsp.n_of(D) + m)
    for i in range(nb):
        t = 0.04 + i * period
        g = float(np.exp(0.05 * r.standard_normal()))
        split1, split2 = r.uniform(0.02, 0.03), r.uniform(0.02, 0.04)
        lub = _valve(r, k, [38.0, 62.0, 110.0, 190.0], [0.14, 0.1, 0.06, 0.04], m)
        lub2 = _valve(r, k, [44.0, 70.0, 125.0, 210.0], [0.12, 0.09, 0.05, 0.03], m)
        dub = _valve(r, k, [55.0, 92.0, 150.0, 260.0], [0.09, 0.07, 0.04, 0.025], m, 0.007)
        dub2 = _valve(r, k, [60.0, 100.0, 170.0, 280.0], [0.08, 0.06, 0.035, 0.02], m, 0.007)
        s2 = 0.33 * period ** 0.5
        for x, dt, a in ((lub, 0.0, 1.0), (lub2, split1, 0.6), (dub, s2, 0.75), (dub2, s2 + split2, 0.45)):
            dsp.place(y, x / (np.abs(lub).max() + 1e-12), t + dt, a * g * 1.25 ** k.e)
    flow = dsp.butter(r.standard_normal(len(y)), 'band', (60.0, 300.0)) * 0.004
    return _loop(c, r, y + flow, D, 'booth', 0.15)


# ------------------------------------------------------------------------------------------------ switches and clicks
@_design('switch/light', kind='click',
        says=('a light switch flicked on, a sprung plastic snap', 'a wall rocker switch clicking',
              'flipping a light switch'),
        uses=('toggle', 'switch_on', 'switch_off', 'click', 'light switch'), tags=('plastic', 'foley'), domains=DOM,
        dur=0.2, axes=AXES)
def switch_light(c, r):
    """a wall rocker: the fingertip meets it (soft, dull), the rocker is pushed until its spring goes over centre
    and it snaps (15-40 ms later): the rocker hits its stop (hard plastic on plastic, the faceplate's short modes),
    the contacts close 2-4 ms after (small steel parts), the hollow box in the wall answers, and the rocker
    bounces once on its stop"""
    k = _k(c)
    snap = _vary(r, 0.025, 0.25)
    n = dsp.n_of(0.2)
    tip = _ring(r, _cloud(r, 400.0, 2000.0, 8, 0.03, 0.01), 0.05, 0.0015)
    plate = _cloud(r, 900.0 * k.f, 6000.0 * k.f, 18, 0.035 * k.ring, 0.012)
    hit = _onto(_ring(r, plate, 0.15, 5e-5 * k.tc), modal.contact_noise(5e-5 * k.tc, r, 1.5, 0.25, 0.1, 0.001))
    contact = _ring(r, _cloud(r, 4000.0, 10000.0, 6, 0.04, 0.02), 0.08, 2e-5)
    box = _ring(r, _modes([520.0 * k.f, 1150.0 * k.f], [0.03 * k.ring, 0.02], [1.0, 0.5]), 0.1, 2e-4, 0.0)
    back = _ring(r, plate, 0.08, 8e-5 * k.tc)
    y = _mix([(tip, 0, -16), (hit, snap, 0 + k.h), (contact, snap + r.uniform(0.002, 0.004), -10),
              (box, snap, -7), (back, snap + r.uniform(0.006, 0.012), -15)], 0.2)
    return _out(c, r, y[:n], 'room', 0.2)


@_design('switch/push', kind='click',
        says=('a mechanical push button pressed and released, click-clack', 'a clicky button pushed down and let go',
              'a tactile switch clicking twice'),
        uses=('tap', 'click', 'button press', 'toggle', 'select'), tags=('plastic', 'foley'), domains=DOM, dur=0.3,
        axes=AXES, allows=('tail', 'ring'))        # the release follows the press; the dome's modes read as a comb
def switch_push(c, r):
    """a push button: pressed, its metal dome snaps through (a sharp click) and the plunger bottoms out 1-3 ms
    later (a duller knock in the housing); released 110-180 ms later, the dome springs back (softer, a little
    brighter) and the plunger hits its top stop"""
    k = _k(c)
    house = _cloud(r, 600.0 * k.f, 6000.0 * k.f, 16, 0.03 * k.ring, 0.01)
    dome = _cloud(r, 3000.0 * k.f, 9500.0, 6, 0.025, 0.012)
    up = _vary(r, 0.14, 0.15)
    tc = 4e-5 * k.tc

    def click(v, tcf):
        return _onto(_ring(r, dome, 0.06, tc * tcf), modal.contact_noise(tc * tcf, r, 1.5, 0.2, 0.08, 0.001)) * v

    y = _mix([(click(1.0, 1.0), 0, 0 + k.h), (_ring(r, house, 0.1, 3e-4 * k.tc), r.uniform(0.001, 0.003), -5),
              (click(0.6, 0.8), up, -6 + k.h), (_ring(r, house, 0.1, 2e-4 * k.tc), up + r.uniform(0.002, 0.005), -9)])
    return _out(c, r, y, 'room', 0.2)


@_design('pen/click', kind='click',
        says=('a ballpoint pen clicked, a small plastic click', 'a retractable pen clicking in and out',
              'the click of a clicky pen'),
        uses=('click', 'tap', 'select', 'toggle', 'write'), tags=('plastic', 'foley'), domains=DOM, dur=0.22,
        axes=AXES, allows=('tail', 'ring'))        # press and release of one pen; its spring's partials read as a comb
def pen_click(c, r):
    """a retractable pen: the thumb presses the plunger, the cam rotates and latches with a small hard plastic
    click; the barrel (a plastic tube) rings briefly and the spring inside gives a faint metallic ting; the
    thumb lets go 60-110 ms later and the cam's second click sounds, softer"""
    k = _k(c)
    barrel = _cloud(r, 1400.0 * k.f, 7000.0 * k.f, 12, 0.04 * k.ring, 0.015)
    spring = _modes(np.array([2400.0, 5100.0, 7900.0]) * k.f, [0.08 * k.ring, 0.06, 0.05], [1.0, 0.6, 0.4])
    tc = 4e-5 * k.tc
    off = _vary(r, 0.085, 0.2)

    def click(v):
        y = _onto(_ring(r, barrel, 0.08, tc / v ** 0.2), modal.contact_noise(tc, r, 1.5, 0.2, 0.06, 0.0008))
        return v * _onto(y, 0.08 * _ring(r, spring, 0.08, tc))

    thumb = _ring(r, _cloud(r, 300.0, 1500.0, 6, 0.02, 0.008), 0.04, 0.002)
    y = _mix([(thumb, 0, -20), (click(1.0), 0.004, 0 + k.h), (click(0.7), 0.004 + off, -5 + k.h)])
    return _out(c, r, y, 'booth', 0.2)


@_design('zipper/pull', kind='object',
        says=('a zipper being pulled closed', 'a metal zip zipping up a jacket', 'the rasp of a zipper'),
        uses=('zip', 'close', 'open', 'swipe', 'pack'), tags=('metal', 'cloth', 'foley'), domains=DOM, dur=0.6,
        axes=AXES, allows=('ring',))     # a train of tooth contacts at a steady rate is a comb by nature
def zipper_pull(c, r):
    """a zip pulled for dur s: the slider meets the teeth one by one, so the sound is a train of tiny contacts
    whose rate is the pull speed over the tooth pitch (2 mm; 0.35-0.5 m/s gives 175-250 contacts a second) with a
    little jitter: the teeth are never quite even and the hand never quite steady; every contact rings the
    slider's small metal modes and the fabric; the cloth rustles under it, the slider unlocks with a click at the
    start and knocks the top stop at the end"""
    k = _k(c)
    T = c.seconds(0.6)
    n = dsp.n_of(T + 0.1)
    v = _speed(n, 0.0, T, 0.45) * (1 + 0.3 * dsp.wander(n / SR, r, 4.0)[:n])
    rate = np.maximum(v, 0.0) * _vary(r, 0.42, 0.1) * 1.2 ** k.e / (0.002 * k.L)
    t = _train(r, rate, 0.15)
    a = np.interp(t, _t(n), np.maximum(v, 0.0)) ** 0.8 * np.exp(0.4 * r.standard_normal(len(t)))
    tooth, cloth = _cloud(r, 2200.0 * k.f, 9000.0, 8, 0.012, 0.004), _cloud(r, 400.0 * k.f, 2500.0, 10, 0.02, 0.006)
    irs = [_ring(r, tooth, 0.03, 4e-5 * k.tc) + 0.6 * _ring(r, cloth, 0.03, 3e-4) for _ in range(4)]
    teeth = _through(r, n, t, a, irs, 0.15)
    stop = _ring(r, _join(tooth, cloth), 0.08, 6e-5 * k.tc)
    y = _mix([(teeth, 0, 0 + k.h), (_bed(_rustle(r, n, np.maximum(v, 0.0), 1500.0)), 0, -14, False),
              (stop, 0.0, -10), (stop, T - 0.01, -4)])
    return _out(c, r, y, 'room', 0.25)


@_design('keys/jingle', kind='object',
        says=('a small bunch of keys jingling', 'metal keys on a key ring shaken and clinking',
              'keys rattling in a hand'),
        uses=('keyring', 'unlock', 'lock', 'home', 'keys'), tags=('metal', 'foley'), domains=DOM, dur=0.9, axes=AXES,
        allows=('tail', 'stacked', 'ring'))       # many key contacts; ringing brass keys are a comb to the detector
def keys_jingle(c, r):
    """four or five keys on a ring, shaken two or three times (energy): every shake throws the keys against each
    other in a burst of collisions (a cluster of 4-8 contacts within 70 ms and a few after), each contact
    striking two keys at once; a key is a small brass bar (modal.bar_modes: a 5 cm key at 2-5 kHz, its
    bending modes 1 : 2.76 : 5.4 above) touching the others and the hand, so it rings only a tenth of a second
    or two, after a clack of metal on metal; the steel ring itself rings higher"""
    k = _k(c)
    irs = []
    for f0 in np.sort(r.uniform(2000.0, 4800.0, int(r.integers(4, 6)))) * k.f:
        for x in r.uniform(0.0, 0.3, 2):
            m = modal.bar_modes(f0, x, 0.0022, 0.022, 3450.0, 5, 0.3)
            f = f0 * m['ratio']
            rate = (modal.decay_rate(f, 'brass', 30.0) * m['damp']) / k.ring
            ir = modal.ring(f, rate, m['gain'], dsp.n_of(0.3 * k.ring), 3e-5 * k.tc).imag[0]
            irs.append(_onto(ir, 0.4 * np.abs(ir).max() * _burst(r, 0.004, 2000.0, 12000.0, 0.0006, 0.0002)))
    irs.append(_ring(r, _modes(np.array([3100.0, 7000.0, 11500.0]) * k.f, [0.2 * k.ring, 0.12, 0.08],
                               [1.0, 0.7, 0.4]), 0.5 * k.ring, 3e-5 * k.tc, 0.0))
    shakes = int(np.clip(round(2.5 + k.e), 2, 3))
    n = dsp.n_of(0.3 * shakes + 0.5 * k.ring)
    t, a = [], []
    for s in range(shakes):
        t0 = s * _vary(r, 0.28, 0.1)
        hit = t0 + np.sort(r.uniform(0.0, 0.08, int(r.integers(7, 13))))
        hit = np.concatenate([hit, t0 + 0.07 + r.exponential(0.06, int(r.integers(1, 4)))])
        t.append(np.repeat(hit, 2) + np.tile([0.0, 0.0002], len(hit)))
        a.append(np.repeat(np.sqrt(_power(r, len(hit), 0.02, 1.0, 1.5)) * (1 - 0.3 * s / shakes), 2))
    out = _through(r, n, np.concatenate(t), np.concatenate(a), [i / (np.abs(i).max() + 1e-12) for i in irs], 0.5)
    return _out(c, r, out, 'room', 0.3)


@_design('scissors/snip', kind='object',
        says=('a pair of scissors snipping through paper', 'a quick metallic scissor snip',
              'scissors cutting, a sharp shearing click'),
        uses=('cut', 'snip', 'delete', 'trim', 'crop'), tags=('metal', 'paper', 'foley'), domains=DOM, dur=0.35,
        axes=AXES)
def scissors_snip(c, r):
    """one cut: the blades slide over each other while they close (friction: a dense stick-slip train at 1.8-3 kHz
    through the blades' steel modes: the tips of thin bars held in a hand, damped and dense), the paper's
    fibres part between them, and the blades meet at their stop with a metallic click that sets them ringing
    briefly; opened again a moment later, they scrape once more, softly"""
    k = _k(c)
    close = _vary(r, 0.07, 0.15) * k.L
    f0 = _vary(r, 2400.0, 0.08) * k.f
    m = modal.bar_modes(f0, 0.1, 0.002, 0.012, 5100.0, 6, 0.3)
    f = f0 * m['ratio']
    blade = _join((f, modal.decay_rate(f, 'steel', 90.0) * m['damp'] / k.ring, m['gain']),
                  _cloud(r, 1500.0 * k.f, 9000.0, 14, 0.03, 0.01))
    ir = _ring(r, blade, 0.08, 1e-4, 0.2)
    n = dsp.n_of(close + 0.35 * k.ring)
    v = _speed(n, 0.0, close, 0.6)
    ts = _train(r, (1800.0 + 1200.0 * v) * (v > 0.05), 0.3, 0.2)
    scrape = _through(r, n, ts, np.interp(ts, _t(n), v) * np.exp(0.4 * r.standard_normal(len(ts))), [ir], 0.1)
    tf, af = _breaks(r, v, 1500.0)
    fibres = _crackle(r, n, tf, af, 1500.0, 8000.0, 0.0005, 0.2)
    snip = _onto(_ring(r, blade, 0.3 * k.ring, 3e-5 * k.tc), modal.contact_noise(3e-5, r, 1.5, 0.2, 0.1, 0.001))
    op = close + r.uniform(0.12, 0.18)
    m2 = dsp.n_of(0.05)
    v2 = _speed(m2, 0.0, 0.05, 0.6)
    t2 = _train(r, (1500.0 + 800.0 * v2) * (v2 > 0.05), 0.3, 0.3)
    reopen = _through(r, m2, t2, np.interp(t2, _t(m2), v2), [ir], 0.1)
    y = _mix([(scrape, 0, -4), (fibres, 0, -3), (snip, close, 0 + k.h), (reopen, op, -12)])
    return _out(c, r, y, 'room', 0.25)


@_design('match/strike', kind='object',
        says=('a match struck and flaring into flame', 'a matchstick scraped on a box and igniting with a whoosh',
              'lighting a match'),
        uses=('fire', 'ignite', 'start', 'light', 'magic'), tags=('wood', 'fire', 'foley'), domains=DOM, dur=1.0,
        axes=AXES, allows=('clicks',))     # the scrape and the sparks are made of tiny clicks
def match_strike(c, r):
    """a match: its head scraped across the striker (60-110 ms of rough, bright friction: dense crackle through the
    cardboard box), the head catching (a soft pff of the chemicals), then the flare: the flame grows from the head
    in about 100 ms, a soft whoosh of combustion noise whose level and brightness swell together and fall back, and
    settles into a small flame: low noise fluttering at the flicker rate of a flame a few mm across (buoyant puffing,
    1.5 / sqrt(d) Hz: 10-14 Hz, Cetegen & Ahmed 1993), with a few late sparks"""
    k = _k(c)
    sc = _vary(r, 0.085, 0.15)
    T = 1.1 * k.L
    n = dsp.n_of(T)
    v = _speed(n, 0.0, sc, 0.8)
    t, a = _breaks(r, v, 3000.0)
    scratch = _crackle(r, n, t, a, 1200.0, 9000.0, 0.0006, 0.2)
    box = _ring(r, _cloud(r, 200.0, 1500.0, 10, 0.03, 0.01), 0.06, 3e-4)
    rub = signal.fftconvolve(dsp.butter(r.standard_normal(n), 'band', (1500.0, 7000.0)) * v, box)[:n]
    fl = sc + r.uniform(0.0, 0.015)
    pff = _burst(r, 0.06, 300.0, 4000.0, 0.02, 0.004)
    m = dsp.n_of(T - fl)
    tf = _t(m)
    grow = _vary(r, 0.1, 0.15)
    swell = np.where(tf < grow, np.sin(0.5 * np.pi * tf / grow) ** 2, 0.35 + 0.65 * np.exp(-(tf - grow) / 0.12))
    flick = _vary(r, 12.0, 0.1)
    flutter = 1 + 0.45 * np.sin(2 * np.pi * flick * tf + 1.5 * dsp.wander(m / SR, r, 4.0)[:m]) * (tf > grow)
    roar = dsp.butter(r.standard_normal(m), 'band', (90.0, 6000.0), 1)
    roar = dsp.svf(roar, 350.0 + 1600.0 * (swell - 0.35 * (tf > grow)) * k.f ** -0.3, 0.8, 'low')
    flame = roar * swell * flutter * np.exp(0.15 * dsp.wander(m / SR, r, 20.0)[:m])
    flame *= np.clip((T - fl - tf) / 0.12, 0, 1)
    ts = fl + grow + r.exponential(0.2, int(r.integers(4, 10) * 1.4 ** k.e))
    sparks = _crackle(r, n, ts, np.exp(0.5 * r.standard_normal(len(ts))), 2000.0, 8000.0, 0.0004, 0.4)
    y = _mix([(scratch, 0, 0 + k.h), (rub, 0, -6), (pff, fl, -8), (flame, fl, -1), (sparks, 0, -14)])
    return _out(c, r, y, 'room', 0.3)


@_design('drawer/slide', kind='object',
        says=('a wooden drawer sliding shut with a soft thump', 'a desk drawer pushed closed',
              'a wooden drawer rumbling along its runners'),
        uses=('close', 'drawer', 'open', 'swipe', 'store'), tags=('wood', 'foley'), domains=DOM, dur=0.7, axes=AXES,
        allows=('rising_glide',))      # the rub brightens as the drawer is pushed up to speed
def drawer_slide(c, r):
    """a drawer pushed shut over dur s: wood rubbing on its runners (a rough contact: noise whose spectrum moves up
    with the speed, plus occasional stick-slip catches) heard through the drawer box (a dense cloud of wood modes),
    what lies inside shifting with a few small knocks, and the stop: the box meets the frame softly (a damped low
    thud, a puff of air pushed out) and rebounds a little"""
    k = _k(c)
    T = c.seconds(0.7)
    stop = max(0.2, T - 0.12)
    n = dsp.n_of(T + 0.25)
    t_ = _t(n)
    v = np.clip(t_ / 0.08, 0, 1) * np.clip(1 - 0.5 * t_ / stop, 0.4, 1) * (t_ < stop)
    v = v * (1 + 0.15 * dsp.wander(n / SR, r, 10.0)[:n])
    rough = dsp.svf(r.standard_normal(n), 300.0 + 2500.0 * np.maximum(v, 0.0), 0.8, 'low') * np.maximum(v, 0) ** 1.2
    box = _cloud(r, 90.0 * k.f, 1200.0 * k.f, 22, 0.09 * k.ring, 0.02)
    ir = _ring(r, box, 0.15, 5e-4)
    run = signal.fftconvolve(rough, ir)[:n]
    tc_ = _train(r, (30.0 + 60.0 * np.maximum(v, 0)) * (v > 0.1), 0.3, 0.5)
    catches = _through(r, n, tc_, np.exp(0.4 * r.standard_normal(len(tc_))), [ir], 0.1)
    pencil = _cloud(r, 1500.0, 5000.0, 6, 0.03, 0.01)
    tk = np.concatenate([r.uniform(0.0, 0.1, 2), stop + r.uniform(0.005, 0.04, int(r.integers(1, 3)))])
    things = _hits(r, n, pencil, tk, r.uniform(0.3, 1.0, len(tk)), 1e-4, 0.06, 0.3, r.uniform(-0.4, 0.4, len(tk)))
    thud = _onto(_ring(r, box, 0.25, 0.002 * k.tc), 0.5 * _ring(r, _modes(np.array([75.0, 140.0]) * k.f,
                                                                          [0.12, 0.08], [1.0, 0.6]), 0.25, 0.003))
    back = r.uniform(0.025, 0.04)
    y = _mix([(_bed(run), 0, -6, False), (catches, 0, -16), (things, 0, -14), (thud, stop, 0 + k.h),
              (_burst(r, 0.08, 150.0, 1500.0, 0.015, 0.002), stop, -12), (thud, stop + back, -16)])
    return _out(c, r, y, 'room', 0.3)


# ------------------------------------------------------------------------------------------------ knocks and bells
@_design('knock/door', kind='object',
        says=('knuckles knocking three times on a wooden door', 'someone knocking at the door, knock knock knock',
              'three knocks on a door'),
        uses=('knock', 'door', 'notify', 'visitor', 'attention', 'join request', 'someone at the door'),
        tags=('wood', 'foley'), domains=DOM, dur=0.8,
        axes=AXES, allows=('tail', 'stacked', 'ring'))      # three knocks; the door's low modes ring
def knock_door(c, r):
    """three knuckle knocks on a door: one door (a wide panel: dense modes from 70 Hz to 2 kHz, the low ones ringing
    longest), struck by a knuckle (bone under skin: a contact of about a millisecond, with its own little click)
    at slightly different points; a person knocks unevenly: the gaps differ by 10-20 %, the middle knock is often
    the firmest; a loose latch rattles in the frame after the harder ones"""
    k = _k(c)
    door = _join(_modes(np.array([85.0, 130.0, 190.0]) * k.f, np.array([0.16, 0.12, 0.1]) * k.ring, [1.0, 0.8, 0.6]),
                 _cloud(r, 70.0 * k.f, 2200.0 * k.f, 34, 0.16 * k.ring, 0.03, -3.0))
    g1 = _vary(r, 0.2, 0.1)
    times = np.array([0.0, g1, g1 + g1 * r.uniform(0.85, 1.2)])
    vels = np.array([0.85, 1.0, 0.9]) * np.exp(0.12 * r.standard_normal(3)) * 1.3 ** k.e
    n = dsp.n_of(times[-1] + 0.35 * k.ring)
    knocks = _hits(r, n, door, times, vels / vels.max(), 0.0012 * k.tc, 0.35 * k.ring, 0.35, 0.0, (0.3, 0.4, 0.003))
    latch = _cloud(r, 1800.0, 6000.0, 5, 0.03, 0.01)
    lt = times[r.uniform(size=3) < 0.6] + r.uniform(0.004, 0.009)
    rattle = _hits(r, n, latch, lt, 0.15, 5e-5, 0.05, 0.3, 0.2)
    return _out(c, r, _mix([(knocks, 0, 0), (rattle, 0, -18)]), 'room', 0.3)


DOME = 'bowl'      # a bell dome is a shallow shell struck near its rim: the bowl ratios (modal.rim_modes)


def _dome(r, f0, n, tc, angle, built, k, support=0.8, material='steel', kind=DOME, fall=0.3):
    """one strike of a bell dome or a small bell (mono): beating pairs (modal.doublets) on the rim modes of a
    shell, or kind 'hand' for a small cast bell (modal.bell_modes), plus the hammer's tick"""
    m = modal.rim_modes(kind, 0.0, 0.004, fall) if kind != 'hand' else modal.bell_modes('hand', 0.0, f0, 0.003)
    y = _round(m, f0, material, support, n, tc, angle, built, 1.0, k.ring, 2.0 * f0)
    return _onto(y, 0.3 * modal.contact_noise(tc, r, 1.5, 0.15, 0.05, 0.001))


@_design('bell/desk', kind='object',
        says=('a reception desk bell rung once, a bright ding', 'a hotel service bell, a clear metallic ding',
              'someone tapping a counter bell'),
        uses=('bell', 'service', 'notify', 'attention', 'success', 'order up', 'new order', 'order ready'),
        tags=('metal', 'foley'), domains=DOM,
        dur=2.4, axes=AXES)
def bell_desk(c, r):
    """a counter bell: the palm pushes the plunger (a dull knock in the base), the spring-loaded clapper hits the
    steel dome 4-8 ms later (steel on steel: a few hundredths of a millisecond, every rim mode driven, each a
    beating pair) and bounces back onto it once 10-20 ms after, weakly and on nearly the same spot; the dome
    rings a few seconds; the plunger returns 120-200 ms later with a small mechanical clack"""
    k = _k(c)
    f0, built = _vary(r, 2100.0, 0.03) * k.f, int(r.integers(1 << 30))
    n = dsp.n_of(2.4 * k.ring)
    hit = r.uniform(0.004, 0.008)
    a0 = r.uniform(0, 2 * np.pi)
    tc = 4e-5 * k.tc
    ring = _dome(r, f0, n, tc, a0, built, k)
    reb = r.uniform(0.01, 0.02)
    ring = _onto(ring, r.uniform(0.12, 0.25) * _dome(r, f0, n, tc * 1.3, a0 + 0.05, built, k), dsp.n_of(reb))
    base = _cloud(r, 300.0, 2500.0, 12, 0.04, 0.012)
    y = _mix([(_ring(r, base, 0.08, 0.001), 0, -16), (ring, hit, 0 + k.h),
              (_ring(r, _cloud(r, 900.0, 4000.0, 8, 0.03, 0.01), 0.06, 1e-4), hit + r.uniform(0.12, 0.2), -22)])
    return _out(c, r, y, 'room', 0.3)


@_design('bell/shop', kind='object',
        says=('a small shop door bell jingling as the door opens', 'a little brass bell shaken on a spring',
              'a doorbell tinkling at a shop entrance'),
        uses=('bell', 'notify', 'door', 'welcome', 'visitor', 'shop'), tags=('metal', 'foley'), domains=DOM,
        dur=1.6, axes=AXES, allows=('tail', 'stacked', 'ring'))      # clapper strikes on two ringing bells
def bell_shop(c, r):
    """two small brass bells on a spring bracket, shaken by the door: the spring swings and the clappers strike
    irregularly (4-7 strikes, energy, 50-160 ms apart, each weaker as the swing dies), each strike on one of the
    two bells and at another point of its mouth (the beating pairs mix anew); the bells (modal.bell_modes 'hand':
    the note and its twelfth) ring on through the later strikes"""
    k = _k(c)
    bells = [(_vary(r, 2600.0, 0.04) * k.f, int(r.integers(1 << 30))),
             (_vary(r, 3300.0, 0.04) * k.f, int(r.integers(1 << 30)))]
    count = int(np.clip(round(5 + 1.5 * k.e + r.uniform(-1, 1)), 3, 8))
    t = np.cumsum(np.concatenate([[0.0], r.uniform(0.05, 0.16, count - 1)]))
    v = 0.72 ** np.arange(count) * np.exp(0.25 * r.standard_normal(count))
    n = dsp.n_of(t[-1] + 1.3 * k.ring)
    out = np.zeros((n, 2))
    for ti, vi in zip(t, v):
        b = int(r.uniform() < 0.4)
        f0, built = bells[b]
        i = dsp.n_of(ti)
        y = vi * _dome(r, f0, n - i, 5e-5 * k.tc, r.uniform(0, 2 * np.pi), built, k, 2.0, 'bronze', 'hand')
        out[i:] += dsp.pan_to_stereo(y, -0.2 if b else 0.2)
    tr = t[0] + r.uniform(0.0, 0.4, 10)
    rattle = _crackle(r, n, tr, np.exp(0.4 * r.standard_normal(10)), 3000.0, 8000.0, 0.0004, 0.3, 2)
    return _out(c, r, _mix([(out, 0, 0), (rattle, 0, -26)]), 'room', 0.3)


@_design('bell/bike', kind='object',
        says=('a bicycle bell ringing twice, ring ring', 'a bike bell, a bright metallic trill',
              'someone ringing a bicycle bell'),
        uses=('bell', 'notify', 'attention', 'warning', 'alert'), tags=('metal', 'foley'), domains=DOM, dur=1.5,
        axes=AXES, allows=('tail', 'stacked', 'ring'))      # two trills of clapper strikes on one ringing dome
def bell_bike(c, r):
    """a rotary bike bell rung twice: each thumb push spins a gear that throws two clappers against the steel dome
    in turn, 28-36 strikes a second for about a fifth of a second (faster in the middle of the push), so the
    dome rings as a trill; the lever clicks as it is pushed, and the dome rings on after each ring"""
    k = _k(c)
    f0, built = _vary(r, 3000.0, 0.04) * k.f, int(r.integers(1 << 30))
    a1, a2 = r.uniform(0, 2 * np.pi), r.uniform(0, 2 * np.pi)
    second = r.uniform(0.32, 0.42)
    n = dsp.n_of(second + 0.25 + 1.0 * k.ring)
    out = np.zeros(n)
    lever = []
    for t0, g in ((0.0, 1.0), (second, r.uniform(0.8, 1.0))):
        d = _vary(r, 0.2, 0.12)
        m = dsp.n_of(d)
        sp = _speed(m, 0.0, d, 0.5)
        ts = _train(r, r.uniform(28.0, 36.0) * (0.6 + 0.4 * sp) * 1.15 ** k.e, 0.08)
        for j, t in enumerate(ts):
            i = dsp.n_of(t0 + t)
            v = g * (0.6 + 0.4 * np.interp(t, _t(m), sp)) * float(np.exp(0.15 * r.standard_normal()))
            out[i:] += v * _dome(r, f0, n - i, 4e-5 * k.tc, a1 if j % 2 else a2, built, k, 1.5)
        lever.append(t0)
    click = _ring(r, _cloud(r, 900.0, 5000.0, 10, 0.03, 0.01), 0.05, 1e-4)
    y = _mix([(out, 0, 0)] + [(click, t, -18) for t in lever])
    return _out(c, r, y, 'room', 0.3)


# ------------------------------------------------------------------------------------------------ desk
@_design('stamp/press', kind='object',
        says=('a rubber stamp pressed onto paper on a desk', 'a stamp thudding down on a document',
              'an office stamp, a soft thump and a sticky lift'),
        uses=('stamp', 'approved', 'success', 'confirm', 'done'), tags=('rubber', 'wood', 'paper', 'foley'),
        domains=DOM, dur=0.5, axes=AXES, allows=('tail', 'ring'))      # the lift follows; the desk's few modes ring
def stamp_press(c, r):
    """a rubber stamp: the pad meets paper on a desk (rubber: a long, soft contact of several ms, so a dull thud of
    the desk and a soft slap of the pad and paper), the wooden handle knocks faintly; lifted 0.18-0.28 s later,
    the inked rubber unsticks from the paper with a short tacky crackle"""
    k = _k(c)
    tc = 0.004 * k.tc
    desk = _ring(r, _wood(r, k, scale=k.f), 0.25, tc)
    handle = _ring(r, _cloud(r, 700.0 * k.f, 2600.0 * k.f, 10, 0.05, 0.015), 0.1, tc * 0.5)
    thup = _burst(r, 0.04, 150.0, 2500.0, 0.006 * k.tc ** 0.5, 0.002)
    lift = _vary(r, 0.23, 0.05)
    n = dsp.n_of(0.03)
    tl = r.uniform(0.0, 0.015, int(r.integers(12, 25)))
    tack = _crackle(r, n, tl, np.exp(0.4 * r.standard_normal(len(tl))), 1500.0, 7000.0, 0.0004, 0.2, 2)
    y = _mix([(desk, 0, 0 + k.h), (handle, 0, -9), (thup, 0, -6), (tack, lift, -14)])
    return _out(c, r, y, 'room', 0.25)


@_design('tape/peel', kind='object',
        says=('sticky tape being pulled off a roll', 'adhesive tape peeling with a raspy screech',
              'packing tape stretched off the roll'),
        uses=('tape', 'peel', 'pack', 'unwrap', 'open'), tags=('plastic', 'foley'), domains=DOM, dur=0.8, axes=AXES,
        allows=('rising_glide',))      # the screech climbs as the pull speeds up: the slip rate follows it
def tape_peel(c, r):
    """tape peeled for dur s: the adhesive lets go in stick-slip (the peel front jumps; Cortet et al. 2007), so
    the force on the film is a sawtooth (texture.stick_slip) at a rate that follows the pull, 350 Hz as it starts
    to 1.2-1.6 kHz at speed: a pitched screech. Its roughness is a second, slower stick-slip of the whole strip
    (the front runs in bursts at 40-90 Hz: the screech is gated by them) and 8 % of jitter per period. The
    sawtooth's edges ring the stretched film (bright, 2-6 kHz) and the roll it comes off (a cardboard core under
    many layers: damped modes at 300-1400 Hz); the adhesive's fibrils snap in a fine crackle under it"""
    k = _k(c)
    T = c.seconds(0.8)
    n = dsp.n_of(T + 0.1)
    t_ = _t(n)
    v = np.clip(t_ / 0.15, 0, 1) ** 0.7 * np.clip((T - t_) / 0.05, 0, 1) * (1 + 0.15 * dsp.wander(n / SR, r, 5.0)[:n])
    v = np.maximum(v, 0.0) * 1.15 ** k.e
    tex = _tex()
    x, _ = tex.stick_slip(r, (350.0 + 1000.0 * v) * (v > 0.03) / k.L, jitter=0.08, rough=0.4, edge=4e-5)
    g, _ = tex.stick_slip(r, (40.0 + 50.0 * v) * (v > 0.03), jitter=0.2, hold=0.05, hold_s=(0.01, 0.04), rough=0.4)
    gate = 0.25 + 0.75 * (g / (g.max() + 1e-12)) ** 0.5
    x = dsp.butter(x, 'high', 250.0) * gate
    film = sum(w * dsp.svf(x, f, 3.0, 'bandn') for f, w in zip(_logu(r, 2000, 6000, 3), (1.0, 0.8, 0.6)))
    roll = sum(w * dsp.svf(x, f, 6.0, 'bandn') for f, w in zip(_logu(r, 300 * k.f, 1400 * k.f, 3), (1.0, 0.8, 0.6)))
    peel = _norm(film) + 0.7 * _norm(roll) + 0.3 * _norm(x)
    tf, af = _breaks(r, np.minimum(v, 1.0), 2500.0)
    fibrils = _crackle(r, n, tf, af, 2000.0, 9000.0, 0.0003, 0.4)
    y = _mix([(dsp.fade(dsp.butter(peel, 'low', 9000.0), 0.002, 0.03), 0, 0 + k.h), (fibrils, 0, -20)])
    return _out(c, r, y, 'room', 0.25)



@_design('stapler/press', kind='object',
        says=('a desk stapler punching a staple through paper', 'a stapler pressed, a metallic ka-chunk',
              'stapling papers together'),
        uses=('staple', 'attach', 'confirm', 'done', 'save'), tags=('metal', 'paper', 'foley'), domains=DOM, dur=0.5,
        axes=AXES, allows=('tail', 'stacked', 'ring'))      # press, clinch, return; the steel body rings
def stapler_press(c, r):
    """a desk stapler: the arm is pushed down (its spring creaks briefly), the driver forces the staple through the
    paper (a short crunch of fibres) and onto the anvil, where its legs fold over (two metallic clicks 3-6 ms
    apart through the steel body's modes), the base knocks the desk; released, the arm springs up 120-180 ms
    later with a lighter clack and the spring's faint ting"""
    k = _k(c)
    body = _cloud(r, 500.0 * k.f, 6000.0 * k.f, 20, 0.08 * k.ring, 0.02)
    tp = _vary(r, 0.05, 0.2)
    m = dsp.n_of(tp)
    sp = _speed(m, 0.0, tp, 1.0)
    tcr = _train(r, (180.0 + 120.0 * sp) * (sp > 0.1), 0.1, 0.2)
    creak = _through(r, m, tcr, np.interp(tcr, _t(m), sp), [_ring(r, body, 0.03, 3e-4)], 0.1)
    tc = 3e-5 * k.tc
    punch = _onto(_ring(r, body, 0.2, tc), modal.contact_noise(tc, r, 1.5, 0.2, 0.1, 0.001))
    clinch = _ring(r, body, 0.2, tc * 1.3)
    n0 = dsp.n_of(0.01)
    paper = _crackle(r, n0, r.uniform(0, 0.004, 10), np.exp(0.4 * r.standard_normal(10)), 1500.0, 8000.0, 0.0004)
    desk = _ring(r, _wood(r, k), 0.2, 0.002)
    up = tp + _vary(r, 0.15, 0.12)
    spring = _modes(np.array([3200.0, 6900.0]) * k.f, [0.15 * k.ring, 0.1], [1.0, 0.5])
    y = _mix([(creak, 0, -20), (paper, tp - 0.002, -10), (punch, tp, 0 + k.h), (clinch, tp + r.uniform(0.003, 0.006),
              -4), (desk, tp, -8), (_ring(r, body, 0.15, 1e-4), up, -9), (_ring(r, spring, 0.15, 3e-5, 0.0), up, -18)])
    return _out(c, r, y, 'room', 0.25)
