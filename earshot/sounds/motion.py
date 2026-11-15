"""Motion-design sounds: what a product film, a title sequence or an interface animation is cut with. Air and
movement (whooshes, swooshes, flicks, a sliding panel), hits (booms, a title slam, soft thumps and knocks), builds and
falls (risers, a reversed swell, a sub drop, a suck-in), small arrivals (pops, tocks, a blip), light (a glint, a
shimmer sweep, a sparkle), a glitch, key ticks and three short stingers.

They are shaped after class statistics of licensed film effects (the film study in docs/motion-sfx-study.md and
docs/launch-video-sound.md, and a library of effects filed by film action that the lab measured). Medians over a
class, never one recording:
    air     a film whoosh is a rounded swell, not a pointed one: it stays within 20 dB of its peak for about two
            thirds of its length (0.3-0.6 s for a small move, 1.3 s for a scene pass), peaks at 40-60 % of it, and
            its energy sits low (centroid 0.2-1.8 kHz; a scene pass under 150 Hz) with the air far under the body;
            the spectrum sinks across the pass; near mono
    hits    sub-dominated (three quarters and more under 150 Hz), a 20-30 ms rise instead of a click, 20 dB down
            in 0.2-0.7 s; a landing in a product film is 0.25 s long and dull (centroid about 500 Hz)
    builds  grow almost linearly in amplitude over 90 % of their length, then fall 20 dB within about 100 ms
    small   0.1-0.25 s long, a 10-20 ms rise, 20 dB down in 25-60 ms, mono, little between 2 and 5 kHz
The trailer register (boom/deep, boom/wide, rise/tension) is the rarer, larger one.

How they are built: air is stereo band noise whose bands follow one moving spectral hump and gust together
(texture.band_noise, texture.envelopes); the hump climbs with the speed of the move and sinks behind it, so nothing
sweeps a tone, and a pitched pass gets its fall from a real propagation delay (fx.doppler). Struck parts are modes
behind a contact pulse (modal.ring), sparkle, grit and debris are event clouds (texture.arrivals, rings, bursts),
tonal parts are unison voices and struck instruments (voice, voices), and everything sits in one shared room.

    from earshot.sounds import registry, motion      # motion registers its designs on import
    y, meta = registry.render('whoosh/fast', seed=2, energy=0.4)
    y, meta = registry.render('rise/air', dur=3.0)   # a sustained gesture keeps `dur`; its peak stays at its share

Axes: energy = a faster, harder, brighter gesture; size = bigger (lower, and longer when no dur is asked); bright =
more top; hard = a faster onset; ring = a longer tail; space, width; motion = more travel across the stereo field
or more movement inside (the designs that list it). Texture, voice and voices are imported where they are used.
"""
from types import SimpleNamespace

import numpy as np
from scipy import fft as sfft

from ..forge import dsp, fx
from ..synth import modal, space
from .registry import design

SR = dsp.SR
T60 = modal.T60
BASE = ('energy', 'size', 'bright', 'hard', 'space', 'width')
RING = BASE + ('ring',)
MOVE = BASE + ('motion',)
DOM = ('motion', 'ui', 'game')
FILM = ('motion', 'game')
PENTA = (0, 2, 4, 7, 9)
MEMBRANE = np.array([1.0, 1.59, 2.14, 2.30, 2.65, 2.92, 3.16, 3.50])     # modes of an ideal drum skin (Bessel zeros)
PLATE = np.array([1.0, 1.73, 2.33, 3.91, 4.11, 6.30])                    # a free disc: (2,0), (0,1), (3,0) ...


# ------------------------------------------------------------------------------------------------ controls
def _k(c):
    """the shared axes as factors: f = frequency factor of size (bigger is lower), L = length factor of size, lo = dB
    of the low layer (size), h = dB of the onset layer (hard, energy), tc = contact-time factor (a harder or faster
    contact is shorter), e = energy in -1..1"""
    e = c.energy
    return SimpleNamespace(e=e, f=c.lerp('size', 1.5, 0.66, 1.0, log=True), L=c.lerp('size', 0.75, 1.4, 1.0, log=True),
                           lo=6.0 * c.size, h=6.0 * c.hard + 2.0 * e,
                           tc=c.lerp('hard', 1.9, 0.5, 1.0, log=True) * 0.85 ** e)


def _ring(c):
    """the tail length factor of the ring axis"""
    return c.lerp('ring', 0.5, 2.0, 1.0, log=True)


def _span(c, k, default, least=0.06):
    """the length of a sustained gesture in s: the one asked for, else the design's own scaled by size"""
    T = c.length()
    return max(least, float(T)) if T else default * k.L


def _out(c, r, y, room='room', spread=0.3, tilt=0.0, crest=None):
    """mono or stereo layers to the finished take: width, a high shelf for bright (and energy: harder is
    brighter), the shared room; crest (dB of the peak over the loudness) matches a short sound's takes (_even)"""
    w = c.lerp('width', 0.0, 2.0, 1.0)
    st = fx.decorrelate(y, min(1.0, spread * w)) if np.ndim(y) == 1 else fx.width(y, w)
    st = space.room(_shelf(st, 6.0 * c.bright + 3.0 * c.energy + tilt), c.room(room), amount=c.space, r=r)
    return st if crest is None else _even(st, crest)


def _even(y, crest):
    """a take whose peak stands more than `crest` dB over its loudness (momentary-max LUFS) has its highest peaks
    rounded off (an oversampled soft clip, driven by 6 dB at most): an unlucky stack of contacts, one tall
    half-cycle. A limiter cannot do this to a tick, which is over before a gain can recover. So every take of a
    design lands at about the same loudness once it is levelled by its peak, as a designer matches a set of
    variations (earshot.sounds.object does the same for its longer sounds). The designs pass the lowest crest of
    eight of their takes as built"""
    from ..synth import master
    for _ in range(3):
        pk = np.abs(y).max() + 1e-12
        over = 20 * np.log10(pk) - master.loudness(dsp.butter(y, 'high', 25.0)) - crest
        if over < 0.4:
            break
        y = fx.saturate(y / pk, min(1.6 * over, 6.0), 'tanh', level='peak') * pk
    return y


def _shelf(y, db, fc=2500.0):
    """first-order high shelf: low pass + G * high pass of the same corner sum to a flat line when G = 1"""
    return y + (10 ** (db / 20) - 1) * dsp.butter(y, 'high', fc, order=1) if db else y


def _vary(r, x, s):
    """x scaled by a log-normal take-to-take variation of s (a fraction)"""
    return x * float(np.exp(s * r.standard_normal()))


def _norm(x):
    return x / (np.abs(x).max() + 1e-12)


def _bed(x):
    """a noise layer at its mixing level: its loudest 10 ms at an RMS of 0.3 (peaks near 1), so that one rare sample
    of the noise does not set how loud the layer is"""
    x = np.asarray(x, np.float64)
    p = (x ** 2).mean(axis=1) if x.ndim == 2 else x ** 2
    w = dsp.n_of(0.01)
    c = np.concatenate([[0.0], np.cumsum(p)])
    top = (c[w:] - c[:-w]).max() / w if len(p) > w else p.mean()
    return x * (0.3 / (np.sqrt(top) + 1e-12))


def _mix(layers, seconds=None):
    """layers (x, t s, gain dB), mono or stereo, into one buffer (stereo when any layer is); every layer arrives at
    its mixing level (_norm: a transient or a tone at peak 1, _bed: a noise layer). With a length, whatever runs
    past it is faded out over the last 20 ms, never cut"""
    st = any(np.ndim(x) == 2 for x, _, _ in layers)
    ls = [(dsp.pan_to_stereo(x) if st and np.ndim(x) == 1 else np.asarray(x, np.float64), t, g) for x, t, g in layers]
    n = dsp.n_of(seconds) if seconds else max((dsp.n_of(t) if t > 0 else 0) + len(x) for x, t, _ in ls)
    out = np.zeros((n, 2) if st else n)
    for x, t, g in ls:
        dsp.place(out, x, t, 10 ** (g / 20))
    return dsp.fade(out, 0.0, 0.02) if seconds else out


def _tex():
    from ..synth import texture        # lazy: numba kernels, only where events or band noise are used
    return texture


def _voice():
    from ..synth import voice
    return voice


# ------------------------------------------------------------------------------------------------ sources
def _t(n):
    return np.arange(n) / SR


def _env(n, attack, tau):
    """a raised-cosine attack and an exponential decay with time constant tau s"""
    e = np.exp(-np.maximum(_t(n) - attack, 0.0) / max(tau, 1e-4))
    a = min(n, dsp.n_of(attack))
    if a > 1:
        e[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    return e


def _noise(r, n, color='pink'):
    """pink (1/f) or brown (1/f^2) noise of unit RMS, drawn at a fast FFT length and cut"""
    m = sfft.next_fast_len(n, real=True)
    F = sfft.rfft(r.standard_normal(m))
    f = sfft.rfftfreq(m, 1 / SR)
    f[0] = f[1]
    w = sfft.irfft(F / f ** (0.5 if color == 'pink' else 1.0), m)[:n]
    return w / (np.sqrt(np.mean(w ** 2)) + 1e-12)


def _burst(r, seconds, lo, hi, tau, attack=0.0003):
    """band-passed noise with a fast attack and an exponential decay, peak 1: a crack, a puff, the tick of a
    contact. The decay moves +-15 % from take to take"""
    n = dsp.n_of(seconds)
    x = dsp.butter(r.standard_normal(n), 'band', (lo, min(hi, 0.45 * SR))) * _env(n, attack, _vary(r, tau, 0.15))
    return dsp.fade(_norm(x), 0.0, min(0.01, 0.3 * seconds))


def _thump(r, seconds, f0, f1, drop, tau, attack=0.002, harm=0.25):
    """the weight of a hit: a sine whose pitch falls from f0 to f1 Hz (time constant drop s: a skin or a floor
    relaxing after the blow) with its second harmonic at `harm`, so it still reads on small speakers. Peak 1"""
    n = dsp.n_of(seconds)
    t = _t(n)
    f = f1 + (_vary(r, f0, 0.05) - f1) * np.exp(-t / max(drop, 1e-4))
    ph = 2 * np.pi * (np.cumsum(f) - f[0]) / SR
    y = (np.sin(ph) + harm * np.sin(2 * ph)) * _env(n, attack, _vary(r, tau, 0.12))
    return dsp.fade(_norm(y), 0.0, min(0.25 * seconds, 0.3))


def _dull(r, seconds, f, tau, attack=0.002, q=1.1):
    """a dull thud: noise through a broad resonance at f Hz (a padded body, soft ground, a felt bumper answering a
    blow); with no steady pitch it is never heard as a tone the way a sine thump can be. Peak 1"""
    n = dsp.n_of(seconds)
    x = dsp.svf(r.standard_normal(n), f, q, 'bandn') * _env(n, attack, _vary(r, tau, 0.15))
    return dsp.fade(_norm(dsp.butter(x, 'low', 3 * f, order=2)), 0.0, min(0.03, 0.3 * seconds))


def _rumble(r, seconds, lo, hi, attack, tau):
    """a rolling low rumble: band-limited brown noise whose level wanders slowly (a big space settling). Peak 1"""
    n = dsp.n_of(seconds)
    x = dsp.butter(_noise(r, n, 'brown'), 'band', (lo, hi))
    knots = max(4, int(4 * seconds))                           # its level moves a few dB about four times a second
    x *= np.exp(0.4 * np.interp(np.arange(n), np.linspace(0, n, knots), r.standard_normal(knots)))
    return dsp.fade(_norm(x * _env(n, attack, tau)), 0.0, 0.3 * seconds)


def _cloud(r, lo, hi, count, t60_lo, t60_hi, tilt=-3.0):
    """a dense body (a board, a desk, a plastic case): `count` modes log-uniform in lo..hi Hz whose ring time falls
    from t60_lo at lo to t60_hi at hi (wood and plastic: the upper modes die first), signed amplitudes tilted by
    `tilt` dB per octave. Returns (Hz, decay rates 1/s, gains)"""
    f = np.sort(np.exp(r.uniform(np.log(lo), np.log(hi), int(count))))
    u = np.log(f / lo) / np.log(hi / lo)
    t60 = np.exp(np.log(t60_lo) + u * np.log(t60_hi / t60_lo))
    a = 10 ** (tilt * np.log2(f / lo) / 20) * r.uniform(0.3, 1.0, len(f)) * r.choice([-1.0, 1.0], len(f))
    return f, T60 / t60, a


def _join(*bodies):
    return tuple(np.concatenate(v) for v in zip(*bodies))


def _struck(r, body, seconds, tc, noise=None, vary=0.3, rough=0.0):
    """one contact on a body (Hz, decay rates 1/s, gains), mono, peak 1: its modes behind a force pulse tc s long
    (modal.ring: a longer contact leaves the high modes out, as a softer tip does); vary moves each mode's weight
    from take to take (another strike point); noise = (click, grit, ring_s) adds the sound of the contact itself;
    rough (against the modes' peak) adds the many modes a table of a few dozen leaves out: noise in the body's
    band that outlasts the listed modes by half (their median ring time), so the tail of a dense body (wood,
    plastic, a desk) is a rustle and never a few notes"""
    f, rate, a = body
    tc = float(np.clip(tc, 2e-5, 0.012))
    n = dsp.n_of(seconds)
    y = modal.ring(f, rate, a * np.exp(vary * r.standard_normal(len(a))), n, tc).imag[0].copy()
    if noise:
        z = modal.contact_noise(tc, r, 1.5, *noise)
        m = min(len(y), len(z))
        y[:m] += z[:m] * np.abs(a).mean()
    if rough:
        band = dsp.butter(r.standard_normal(n), 'band', (f.min(), min(f.max(), 0.4 * SR)))
        band = dsp.butter(band, 'low', min(1.5 / tc, 0.4 * SR), order=1) * _env(n, 2.0 * tc, 1.5 / np.median(rate))
        y = _norm(y) + rough * _norm(band)
    return dsp.fade(_norm(y), 0.0, min(0.02, 0.3 * seconds))


def _scale(root, lo, hi, gap=(2100.0, 4700.0)):
    """the notes of the major pentatonic scale on `root` Hz that lie in lo..hi Hz, without those inside `gap`: a
    dense run of small bars between 2 and 5 kHz is the band where sparkle turns harsh"""
    f = np.array([root * 2.0 ** ((s + 12 * o) / 12) for o in range(-4, 7) for s in PENTA])
    return f[(f >= lo) & (f <= hi) & ((f < gap[0]) | (f > gap[1]))]


def _chimes(r, n, t, gain, f, decay, pan, attack=0.002, split=0.004):
    """small struck bars into a stereo buffer of n samples (texture.rings): each event a damped sinusoid with the
    second mode of a free bar (2.76 f, 14 dB down, dying three times as fast) and a slight upward settle. Every
    bar sounds as a pair about `split` apart in frequency, one member in each channel at the level its pan gives
    it, so the channels beat against each other and a cloud of them is wide, not a row of panned points"""
    out = np.zeros((n, 2))
    if not np.size(t):
        return out
    tex = _tex()
    t, gain, f, decay, pan, attack = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, np.float64))
                                                           for v in (t, gain, f, decay, pan, attack)])
    th = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4
    up = 2.756 * f * (1 + 2 * split) < 0.42 * SR
    for side, lvl in ((-1.0, np.cos(th)), (1.0, np.sin(th))):
        fs = f * (1 + side * split * r.uniform(0.5, 1.5, len(f)))
        a = gain * lvl * np.sqrt(2.0)
        tex.rings(out, t, a, side, fs, decay, attack, 0.02 / decay, wrap=False)
        tex.rings(out, t[up], 0.2 * a[up], side, 2.756 * fs[up], 0.35 * decay[up], 0.001, wrap=False)
    return out


# ------------------------------------------------------------------------------------------------ moving air
def _move(c, r, T, peak, shape, depth=54.0):
    """the level of a gesture T s long (0..1 per sample) and where it peaks (a share of its length, a few per cent
    different in every take): `depth` dB down at both ends and in between, in dB, a power law of the distance to
    the peak with the exponents `shape` (in, out). 2 is a Gaussian, the rounded swell measured on film whooshes; 1
    is straight in dB; a harder gesture rises on a smaller exponent, so it accelerates into its peak"""
    n = dsp.n_of(T)
    pk = float(np.clip(peak * np.exp(0.05 * r.standard_normal()), 0.03, 0.97))
    u = (np.arange(n) + 0.5) / n
    x = np.where(u < pk, (pk - u) / pk, (u - pk) / (1.0 - pk))
    k_in = shape[0] * c.lerp('hard', 1.4, 0.6, 1.0, log=True)
    return 10 ** (-depth * x ** np.where(u < pk, k_in, shape[1]) / 20), pk


def _hump(f, centre, lo_db, hi_db):
    """the amplitude at f Hz of a broad spectral hump: rising lo_db per octave up to `centre`, falling hi_db per
    octave above it (the shape of aerodynamic noise, as in dsp.air)"""
    u = np.log2(np.maximum(f, 1.0) / centre)
    return (2.0 ** (-lo_db / 3.0103 * u) + 2.0 ** (hi_db / 3.0103 * u)) ** -0.5


def _air(r, a, centre, lo_db=5.0, hi_db=6.0, span=(60.0, 14000.0), bands=10, gust=0.25, rate=(2.0, 40.0),
         common=0.25, side=(0.05, 0.6), pan=None):
    """moving air as stereo band noise (n, 2): `bands` log-spaced bands over `span` Hz, each under the gesture's
    level a (per sample) times its place on a spectral hump whose centre (Hz, a number or per sample) moves, times
    its own turbulence: log-normal gusts of depth `gust` between rate[0] and rate[1] Hz of which the share `common`
    is one gust for all bands (texture.envelopes; a quarter by default, so the bands live on their own and the
    loudest moment stays where the gesture puts it). So the spectrum itself moves band by band and no filter
    sweeps across a static noise. side = how decorrelated the two channels are at 200 Hz and below .. at 2 kHz and above
    (texture.band_noise); pan (-1..1, a number or per sample) carries the whole pass across"""
    tex = _tex()
    n = len(a)
    m = sfft.next_fast_len(n, real=True)
    cen = span[0] * (span[1] / span[0]) ** (np.arange(bands) / (bands - 1))
    env = tex.envelopes(m / SR, r, bands, gust, common, rate[0], rate[1])
    k = env.shape[1]
    j = np.arange(k) * m // k
    live = j < n
    j = np.minimum(j, n - 1)
    fc = np.broadcast_to(np.asarray(centre, np.float64), (n,))[j]
    env = env * (a[j] * live) * _hump(cen[:, None], fc[None, :], lo_db, hi_db)
    if pan is not None:
        th = (np.clip(np.broadcast_to(np.asarray(pan, np.float64), (n,))[j], -1.0, 1.0) + 1.0) * np.pi / 4
        env = env[:, :, None] * (np.sqrt(2.0) * np.stack([np.cos(th), np.sin(th)], axis=1))[None]
    sd = side[0] + (side[1] - side[0]) * np.clip(np.log(cen / 200.0) / np.log(10.0), 0.0, 1.0)
    return tex.band_noise(m, r, cen, env, sd)[:n]


def _across(r, n, pk, travel):
    """the pan of a pass over n samples: from one side (another in every take) `travel` of the way out, through the
    centre at the peak, to the other side"""
    return r.choice([-1.0, 1.0]) * min(travel, 1.0) * np.tanh(((np.arange(n) + 0.5) / n - pk) / 0.25)


def _wind(c, r, k, a, pk, f_peak, up, down, pan, lo_db=5.0, hi_db=6.0, track=0.6, body=None, q=2.2, **air):
    """the air of one pass at its mixing level: _air under the level a, its hump at f_peak Hz at the peak, `up`
    octaves lower at the start and `down` octaves lower at the end (the spectrum follows the speed of the move, as
    level ** track, and sinks further behind it). Energy and bright lift the hump and flatten its top.
    body (dB against the air, None = none) adds what the moving thing itself sheds: noise through one broad
    resonance of quality q at the hump's peak, in the centre, which gives the pass a colour as the measured film
    whooshes have (a third of each moment's energy within a quarter octave of its strongest band; plain band noise
    has a sixth). The resonance moves a third of an octave at most, so no pitch is heard to sweep"""
    n = len(a)
    u = (np.arange(n) + 0.5) / n
    m = a ** track
    f0 = _vary(r, f_peak, 0.07) * k.f * 2.0 ** (0.3 * k.e + 0.4 * c.bright)
    y = _bed(_air(r, a, f0 * 2.0 ** (-np.where(u < pk, up, down) * (1.0 - m)), lo_db,
                  hi_db * 2.0 ** (-0.4 * c.bright), pan=pan, **air))
    if body is None:
        return y
    res = dsp.svf(r.standard_normal(n), f0 * 2.0 ** (-0.33 * (1.0 - m)), q, 'bandn')
    res = _bed(res * a * np.exp(0.3 * dsp.wander(n / SR, r, 12.0)[:n]))
    return _bed(y + 10 ** (body / 20) * res[:, None])


def _weight(r, a, f, lag=0.0):
    """the weight under a gesture (the air a larger object shoves ahead of it): brown noise between f / 2.5 and f
    Hz under the gesture's level a, arriving `lag` s after it. Mono, at its mixing level"""
    n = len(a)
    x = dsp.butter(dsp.butter(_noise(r, n, 'brown'), 'low', f), 'high', max(28.0, f / 2.5))
    i = min(n - 1, dsp.n_of(lag)) if lag > 0 else 0
    return _bed(x * np.concatenate([np.zeros(i), a[:n - i]]))


def _grit(r, a, f, rate, length=0.0015, spread=0.5, cluster=0.0, sharp=0.6):
    """fine grains riding a gesture (the crispness of fast air, the fibres of paper, a runner's grain): band-passed
    noise events (texture.bursts) arriving at up to `rate` per second where the gesture is loudest (the rate follows
    its level squared), optionally in clusters, sizes by a power law, each at its own pitch around f Hz; sharp
    1 = each full at its onset (a crackle) .. 0 = each a small swell (a rub). Stereo, at its mixing level"""
    tex = _tex()
    n = len(a)
    t, _ = tex.arrivals(r, rate * a[::48] ** 2, n / SR, cluster, 0.004)
    out = np.zeros((n, 2))
    if len(t):
        s = tex.sizes(r, len(t), 0.05, 1.0, 'power', 1.6)
        tex.bursts(out, t, s, r.uniform(-spread, spread, len(t)), f * np.exp(0.35 * r.standard_normal(len(t))), 1.4,
                   length * (0.5 + s), r, sharp=sharp, wrap=False)
    return _bed(out)


@design('whoosh/soft', kind='whoosh',
        says=('a soft short whoosh', 'a gentle low swish of air passing by', 'a quiet puff of air moving past'),
        uses=('whoosh', 'transition', 'element move', 'swipe', 'slide in'), domains=DOM, dur=0.52, axes=MOVE,
        prior=1.05)
def whoosh_soft(c, r):
    """a small element crossing the frame: a rounded swell of dark air (the hump at 620 Hz at its peak, 1.3 octaves
    lower at the start, 1.9 at the end, the rise a little longer than the fall) with a soft body in its centre,
    over the low weight of the air it moves, which arrives 12 ms late"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 0.52), 0.56, (2.0, 1.8))
    air = _wind(c, r, k, a, pk, 620.0, 1.3, 1.9, _across(r, len(a), pk, 0.2 * (1 + c.motion)), lo_db=6.0, hi_db=9.0,
                gust=0.2, body=0.0)
    low = _weight(r, a, 190.0 * k.f, 0.012)
    return _out(c, r, _mix([(air, 0, 0), (low, 0, -6 + k.lo)]), room='room')


@design('whoosh/fast', kind='whoosh',
        says=('a fast crisp whoosh', 'a quick sharp swish of air rushing past', 'a short fast swoosh'),
        uses=('whoosh', 'whip pan', 'fast transition', 'cut', 'fly in', 'title'), domains=DOM, dur=0.4, axes=MOVE,
        prior=1.03)
def whoosh_fast(c, r):
    """a fast element or a whip pan: a shorter, more pointed pass with the hump at 1.2 kHz and a body in it, quick
    gusts, a crisp top of fine grains around 6.5 kHz that ride the peak (above the band where a whoosh turns
    harsh), and a little weight"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 0.4), 0.55, (1.7, 1.5))
    air = _wind(c, r, k, a, pk, 1200.0, 1.6, 2.3, _across(r, len(a), pk, 0.3 * (1 + c.motion)), lo_db=6.0, hi_db=8.0,
                gust=0.25, rate=(6.0, 60.0), body=1.0)
    crisp = _grit(r, a, 6500.0 * k.f ** 0.5, 900.0)
    low = _weight(r, a, 170.0 * k.f, 0.008)
    return _out(c, r, _mix([(air, 0, 0), (crisp, 0, -14 + k.h), (low, 0, -9 + k.lo)]), room='room')


@design('whoosh/deep', kind='whoosh',
        says=('a deep low whoosh, a large heavy swell of air', 'a big dark rumbling whoosh passing',
              'a slow low rush of air'),
        uses=('whoosh', 'big transition', 'scene change', 'pass by', 'swell'), domains=FILM, dur=1.4, axes=MOVE)
def whoosh_deep(c, r):
    """a scene transition: the designed film whoosh is a low swell, not air. A slow rounded pass of noise whose
    hump sits near 105 Hz (nine tenths of the energy under 150 Hz) and is wide even down low, with the air of the
    move 23 dB under it and sinking two octaves behind the peak"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 1.4), 0.66, (2.0, 1.9))
    pan = _across(r, len(a), pk, 0.25 * (1 + c.motion))
    swell = _wind(c, r, k, a, pk, 105.0, 0.7, 1.4, pan, lo_db=6.0, hi_db=10.0, gust=0.2, rate=(1.5, 14.0),
                  span=(32.0, 6000.0), side=(0.55, 0.9))
    air = _wind(c, r, k, a, pk, 900.0, 1.5, 2.2, pan, lo_db=3.0, hi_db=6.0, gust=0.3, rate=(2.0, 20.0),
                span=(200.0, 12000.0), bands=8, side=(0.5, 0.9))
    return _out(c, r, _mix([(swell, 0, k.lo / 2), (air, 0, -23 + k.h / 2)]), room='room')


@design('whoosh/airy', kind='whoosh',
        says=('a wide breathy whoosh of air', 'a slow soft hiss of wind sweeping past', 'a long airy swoosh'),
        uses=('whoosh', 'slow transition', 'air movement', 'reveal', 'breath'), domains=FILM, dur=0.9, axes=MOVE)
def whoosh_airy(c, r):
    """a slow wide breath in two layers that leave the band between 2 and 5 kHz thin: the soft body of the moving
    air (a hump at 800 Hz) and the hiss of the breath above it (a hump at 8 kHz, wide), both with slow deep gusts
    that the bands do not share, so it breathes instead of pulsing; no weight under it"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 0.9), 0.62, (2.2, 2.6))
    pan = _across(r, len(a), pk, 0.35 * (1 + c.motion))
    body = _wind(c, r, k, a, pk, 800.0, 1.0, 1.5, pan, lo_db=4.0, hi_db=8.0, gust=0.4, rate=(1.0, 12.0),
                 side=(0.2, 0.6))
    hiss = _wind(c, r, k, a, pk, 8000.0, 0.5, 0.8, pan, lo_db=7.0, hi_db=4.0, gust=0.45, rate=(1.0, 12.0),
                 span=(2500.0, 16000.0), bands=6, side=(0.5, 0.75))
    return _out(c, r, _mix([(body, 0, 0), (hiss, 0, -4 + k.h / 2)]), room='room')


@design('whoosh/tonal', kind='whoosh',
        says=('a whoosh with a humming tone that drops in pitch as it passes', 'a pitched object rushing past',
              'a tonal swoosh flying by'),
        uses=('whoosh', 'pass by', 'fly by', 'transition', 'tech'), domains=FILM, dur=0.9, pitch='A3', axes=MOVE)
def whoosh_tonal(c, r):
    """air with a pitched body: a blown tone (noise through narrow resonances on the pitch and its next three
    harmonics, with a faint steady partial) carried past the listener at 30 m/s, so its pitch falls by the
    propagation delay alone (fx.doppler: about 3 semitones, fastest at the closest point; never a swept oscillator)
    while the air of the pass crosses with it, from either side"""
    k = _k(c)
    T = _span(c, k, 0.9)
    a, pk = _move(c, r, T, 0.5, (2.0, 1.9))
    n = len(a)
    f0 = c.hz()
    src = sum(g * dsp.svf(r.standard_normal(n), f0 * q, 30.0, 'bandn') for q, g in ((1.0, 1.0), (2.0, 0.6),
                                                                                  (3.0, 0.35), (4.02, 0.2)))
    src = dsp.fade(_norm(src) + 0.25 * np.sin(2 * np.pi * f0 * _t(n) + r.uniform(0, 6.28)), 0.02 * T, 0.05 * T)
    v, d = _vary(r, 30.0, 0.06) * (1 + 0.12 * k.e), 2.4 * T
    tone = fx.doppler(src, speed=v, dist=d, t_close=pk * T, width=min(1.0, 0.6 * (1 + c.motion)))
    lead = dsp.n_of((np.hypot(v * pk * T, d) - d) / 343.0)        # the output starts when the first sound arrives
    tone = np.concatenate([np.zeros((lead, 2)), tone, np.zeros((n, 2))])[:n] * (a[:, None] ** 0.6)
    u = (np.arange(n) + 0.5) / n
    air = _wind(c, r, k, a, pk, 800.0, 1.2, 1.8, min(1.0, 0.3 * (1 + c.motion)) * np.tanh((u - pk) / 0.25),
                lo_db=4.0, hi_db=7.0, gust=0.25)
    low = _weight(r, a, 180.0 * k.f, 0.01)
    y = _mix([(_bed(tone), 0, -1), (air, 0, -3 + k.h / 2), (low, 0, -10 + k.lo)])
    return _out(c, r, y[:, ::-1] if r.uniform() < 0.5 else y, room='room', tilt=-3.0 * c.size)


@design('swoosh/in', kind='whoosh',
        says=('a whoosh that rises and stops, something arriving', 'a quick swell of air rushing in',
              'a short swoosh building to an abrupt end'),
        uses=('whoosh', 'fly in', 'arrive', 'element enters', 'appear'), domains=DOM, dur=0.55, axes=MOVE)
def swoosh_in(c, r):
    """an element arriving: the air climbs for five sixths of the length (the hump from 220 Hz up to 1 kHz as
    the move speeds up), comes in from one side to the centre and stops within the last sixth"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 0.55), 0.84, (1.7, 1.6))
    u = (np.arange(len(a)) + 0.5) / len(a)
    pan = r.choice([-1.0, 1.0]) * min(1.0, 0.45 * (1 + c.motion)) * np.clip(1.0 - u / pk, 0.0, 1.0)
    air = _wind(c, r, k, a, pk, 1000.0, 2.2, 0.8, pan, lo_db=4.0, hi_db=7.0, gust=0.22, body=-1.0)
    low = _weight(r, a, 180.0 * k.f, 0.01)
    return _out(c, r, _mix([(air, 0, 0), (low, 0, -9 + k.lo)]), room='room')


@design('swoosh/out', kind='whoosh',
        says=('a whoosh that starts full and fades away, something leaving', 'air rushing away into the distance',
              'a short swoosh trailing off'),
        uses=('whoosh', 'fly out', 'exit', 'element leaves', 'dismiss'), domains=DOM, dur=0.6, axes=MOVE)
def swoosh_out(c, r):
    """an element leaving: full within the first seventh of the length, then the air falls away, darkening 2.6
    octaves and drifting from the centre to one side"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 0.6), 0.14, (1.5, 1.5))
    u = (np.arange(len(a)) + 0.5) / len(a)
    pan = r.choice([-1.0, 1.0]) * min(1.0, 0.5 * (1 + c.motion)) * np.clip((u - pk) / (1.0 - pk), 0.0, 1.0)
    air = _wind(c, r, k, a, pk, 1000.0, 0.6, 2.6, pan, lo_db=4.0, hi_db=7.0, gust=0.22, body=-1.0)
    low = _weight(r, a, 190.0 * k.f, 0.006)
    return _out(c, r, _mix([(air, 0, 0), (low, 0, -9 + k.lo)]), room='room')


@design('flick/paper', kind='whoosh',
        says=('a quick flick of paper', 'a very short dry swish of a card', 'a light papery flutter'),
        uses=('swish', 'flick', 'text slide', 'page turn', 'card flip'), domains=DOM, dur=0.16, axes=MOVE)
def flick_paper(c, r):
    """a sheet flicked through the air: a short bright puff, the fibres of the paper as clustered fine grains
    around 7 kHz, and the flap of the sheet itself low under it; eased between 2 and 5 kHz"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 0.16, 0.05), 0.4, (1.6, 1.4))
    air = _wind(c, r, k, a, pk, 5200.0, 1.0, 1.4, _across(r, len(a), pk, 0.15 * (1 + c.motion)), lo_db=5.0,
                hi_db=4.0, gust=0.3, rate=(20.0, 200.0), span=(400.0, 16000.0), side=(0.2, 0.6))
    fibres = _grit(r, a, 7000.0 * k.f ** 0.5, 2200.0, 0.0008, cluster=0.5)
    flap = _weight(r, a, 520.0 * k.f, 0.004)
    y = _mix([(air, 0, 0), (fibres, 0, -6 + k.h), (flap, 0, -14 + k.lo)])
    return _out(c, r, dsp.peq(y, 3200.0, -3.0, 0.9), room='booth', crest=15.6)


@design('flick/thin', kind='whoosh',
        says=('a tiny thin high swish', 'a very short light flick of air', 'a quick faint whip'),
        uses=('swish', 'flick', 'text move', 'small move', 'cursor'), domains=DOM, dur=0.11, axes=MOVE, prior=1.02)
def flick_thin(c, r):
    """the smallest move: a tenth of a second of air in a narrow hump at 6.2 kHz (10 dB per octave below it, 7
    above: thin, little under 4 kHz and nothing under 1 kHz) around one broad resonance, a little pointed"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 0.11, 0.04), 0.45, (1.8, 1.5))
    air = _wind(c, r, k, a, pk, 6200.0, 0.8, 1.2, _across(r, len(a), pk, 0.15 * (1 + c.motion)), lo_db=10.0,
                hi_db=7.0, gust=0.15, rate=(20.0, 200.0), span=(1200.0, 16000.0), bands=8, side=(0.2, 0.5), body=0.0,
                q=3.0)
    return _out(c, r, air, room='booth', crest=14.7)


@design('slide/panel', kind='whoosh',
        says=('a panel sliding with a soft stop', 'a smooth low sliding sound ending in a soft bump',
              'a drawer gliding shut'),
        uses=('swipe', 'slide', 'panel', 'drawer', 'card slide'), domains=DOM, dur=0.45, axes=MOVE)
def slide_panel(c, r):
    """a panel on runners: the rolling noise of the slide (a dark hump at 380 Hz with the fast gusts of a runner's
    grain, a low rumble, a faint fine friction) speeds up and slows over five sixths of the length; while it still
    moves the panel meets a felt stop: a dull bump and the small knock of its frame"""
    k = _k(c)
    T = _span(c, k, 0.45, 0.15)
    a, pk = _move(c, r, 0.84 * T, 0.42, (2.0, 1.3))
    roll = _wind(c, r, k, a, pk, 380.0, 1.0, 1.2, _across(r, len(a), pk, 0.3 * (1 + c.motion)), lo_db=4.0,
                 hi_db=8.0, gust=0.35, rate=(8.0, 90.0), span=(50.0, 9000.0), side=(0.1, 0.6))
    low = _weight(r, a, 140.0 * k.f, 0.004)
    grain = _grit(r, a, 1800.0 * k.f, 500.0, 0.004, sharp=0.2)
    bump = _dull(r, 0.12, 150.0 * k.f, 0.02, attack=0.004 * k.tc)
    frame = _struck(r, _cloud(r, 300.0 * k.f, 2400.0 * k.f, 12, 0.06, 0.02), 0.08, 0.0015 * k.tc, rough=0.6)
    y = _mix([(roll, 0, 0), (low, 0, -7 + k.lo), (grain, 0, -20), (bump, 0.68 * T, -8 + k.h / 2),
              (frame, 0.68 * T, -17 + k.h)])
    return _out(c, r, y, room='room')


# ------------------------------------------------------------------------------------------------ hits
@design('boom/deep', kind='boom',
        says=('a deep cinematic boom with a long low tail', 'a huge low drum hit with sub bass ringing out',
              'a deep trailer impact'),
        uses=('impact', 'cinematic hit', 'trailer hit', 'reveal', 'title'), domains=FILM, dur=1.7, axes=RING)
def boom_deep(c, r):
    """a big slack drum in a hall: the beater's dull contact (16 ms to the sub's peak: a rise, not a click), a
    clean sub that falls onto 46 Hz and rings for most of a second as the skin's lowest mode, the skin's upper
    modes over it at the Bessel ratios (starting a little sharp and settling as the stretch relaxes; they die
    within a few tenths of a second), the air shock as a noise band that dulls while it decays, and a low rumble.
    The hall only hears what is above 220 Hz, so the sub stays dry"""
    k = _k(c)
    g = _ring(c)
    T = 1.7 * k.L * g ** 0.5
    f0 = _vary(r, 62.0, 0.04) * k.f
    sub = _thump(r, T, 1.5 * f0, 0.74 * f0, 0.07, 0.3 * k.L * g, attack=0.016 * k.tc, harm=0.1)
    up = MEMBRANE[1:]
    skin = dsp.modal(0.74 * f0 * up, 0.5 * g * k.L * up ** -1.3, up ** -1.0, 1.1 * k.L * g, r, 0.01, 0.05)
    hit = dsp.contact(0.006 * k.tc, r, 0.3)
    skin = dsp.fade(_norm(dsp.excite(skin, hit / np.abs(hit).sum())), 0.0, 0.1)
    slap = _dull(r, 0.2, 170.0 * k.f, 0.03, attack=0.003 * k.tc, q=0.9)
    ns = dsp.n_of(0.45)
    shock = dsp.svf(r.standard_normal(ns), np.geomspace(2400.0, 300.0, ns), 0.7, 'low') * _env(ns, 0.004, 0.07)
    shock = dsp.fade(_norm(dsp.butter(shock, 'high', 180.0)), 0.0, 0.1)
    tick = _burst(r, 0.03, 1400.0, 5500.0, 0.004 * k.tc, attack=0.0005)
    rumble = _rumble(r, T, 28.0, 120.0 * k.f, 0.05, 0.45 * k.L * g)
    y = _mix([(sub, 0, -2 + k.lo / 2), (skin, 0.002, 0), (slap, 0, -5 + k.h / 2), (shock, 0.001, -7),
              (tick, 0, -24 + k.h), (rumble, 0.02, -13)], T)
    return _out(c, r, fx.saturate(y, 3.0 + 2.0 * k.e, 'tanh', level='peak'), room='hall', spread=0.2)


@design('boom/short', kind='boom',
        says=('a short deep sub hit', 'a tight low punchy thud', 'a quick bass drum boom'),
        uses=('impact', 'hit on a cut', 'cut', 'accent'), domains=FILM, dur=0.42, axes=RING)
def boom_short(c, r):
    """a hit on a cut: a sub that rises in 14 ms (the films' sub hits swell in, they do not click), falls from 106
    to 56 Hz within 35 ms and is gone in under half a second, the dull knock of the contact and a faint tick on
    top; nearly dry"""
    k = _k(c)
    g = _ring(c)
    T = min(0.48, 0.42 * k.L * g ** 0.5)
    f0 = _vary(r, 56.0, 0.04) * k.f
    sub = _thump(r, T, 1.9 * f0, f0, 0.035, 0.07 * k.L * g, attack=0.014 * k.tc, harm=0.15)
    knock = _dull(r, 0.1, 200.0 * k.f, 0.012, attack=0.004 * k.tc, q=1.0)
    tick = _burst(r, 0.02, 1500.0, 6000.0, 0.0025 * k.tc)
    y = _mix([(sub, 0, k.lo / 2), (knock, 0, -5 + k.h / 2), (tick, 0, -22 + k.h)], T)
    return _out(c, r, fx.saturate(y, 4.0 + 2.0 * k.e, 'tanh', level='peak'), room='booth', spread=0.1)


@design('boom/wide', kind='boom',
        says=('a wide low impact with a rumbling noisy body and falling debris', 'a big crash with rubble',
              'a heavy boom spreading out'),
        uses=('impact', 'big hit', 'trailer hit', 'crash', 'collapse'), domains=FILM, dur=1.8, axes=RING)
def boom_wide(c, r):
    """something large coming down: the body is noise, not a note (a wide band whose hump falls from 320 to 70 Hz
    in a fifth of a second and whose two channels differ even down low), over a centred sub thump; a dull crack at
    the contact; debris landing at a rate that dies away, power-law sizes, the big pieces lower"""
    k = _k(c)
    g = _ring(c)
    T = 1.8 * k.L * g ** 0.5
    n = dsp.n_of(T)
    t = _t(n)
    body = _bed(_air(r, _env(n, 0.012 * k.tc, 0.3 * k.L * g), k.f * (70.0 + 250.0 * np.exp(-t / 0.2)), 4.0, 9.0,
                     span=(30.0, 6000.0), bands=9, gust=0.3, rate=(3.0, 30.0), side=(0.7, 0.9)))
    sub = _thump(r, 1.2 * k.L, 85.0 * k.f, 36.0 * k.f, 0.1, 0.3 * k.L * g, attack=0.01 * k.tc, harm=0.15)
    crack = _burst(r, 0.12, 220.0, 2600.0, 0.02 * k.tc, attack=0.001)
    tex = _tex()
    rate = 90.0 * 2.0 ** (0.5 * k.e) * np.exp(-np.linspace(0, 0.6 * T, 400) / (0.35 * k.L))
    td, _ = tex.arrivals(r, rate, 0.6 * T, 0.3, 0.02)
    debris = np.zeros((n, 2))
    if len(td):
        s = tex.sizes(r, len(td), 0.05, 1.0, 'power', 1.8)
        f = np.exp(r.uniform(np.log(250.0), np.log(3500.0), len(td))) * k.f * (0.5 / (s + 0.5)) ** 0.5
        tex.bursts(debris, td + 0.06, s ** 1.2, r.uniform(-0.9, 0.9, len(td)), f, r.uniform(0.8, 2.0, len(td)),
                   r.uniform(0.006, 0.014, len(td)), r, sharp=0.3, wrap=False)
    debris = dsp.butter(debris, 'low', 5000.0)        # a lone piece wider than this is a click, not a piece landing
    y = _mix([(body, 0, 6), (sub, 0, -4 + k.lo / 2), (crack, 0, -6 + k.h), (_norm(debris), 0, -10 + 3 * k.e)], T)
    return _out(c, r, fx.saturate(y, 3.0 + 2.0 * k.e, 'tanh', level='peak'), room='hall', spread=0.4)


@design('slam/title', kind='hit',
        says=('a hard dry slam', 'a heavy object slammed onto a desk', 'a short punchy wooden bang'),
        uses=('impact', 'title slam', 'slam', 'text slam', 'stamp'), domains=DOM, dur=0.34, axes=RING)
def slam_title(c, r):
    """a title landing hard: a board struck by a 0.8 ms contact (its 30 modes between 150 Hz and 3 kHz ring
    50-280 ms, the upper ones dying first), the dull crack of the contact, and a short low thump under them that
    falls from 125 to 68 Hz, 12 dB down so that the hit stays a slam and not a kick drum; driven a little, and
    almost no room"""
    k = _k(c)
    g = _ring(c)
    T = 0.34 * k.L * g ** 0.5
    board = _struck(r, _cloud(r, 150.0 * k.f, 3000.0 * k.f, 30, 0.28 * g, 0.05, -1.0), T, 0.0008 * k.tc,
                    noise=(0.2, 0.08, 0.004), rough=0.4)
    thump = _thump(r, 0.2 * k.L, 125.0 * k.f, 68.0 * k.f, 0.02, 0.04 * k.L * g, attack=0.0015, harm=0.3)
    crack = _burst(r, 0.03, 400.0, 2800.0, 0.003 * k.tc, attack=0.0002)
    y = _mix([(board, 0, 0), (thump, 0, -12 + k.lo), (crack, 0, -3 + k.h)], T)
    return _out(c, r, fx.saturate(y, 6.0 + 2.0 * k.e, 'tanh', level='peak'), room='booth', spread=0.15, crest=13.4)


def _soft_thump(r, k, g):
    """something soft and heavy landing, mono: the air it pushes out (a dull band of noise near 105 Hz with a 7 ms
    rise), its weight (a thump falling from 96 to 60 Hz) and the felt of the contact (a soft band under 1 kHz,
    15 dB down, with a 5 ms rise: nothing in it is hard)"""
    T = 0.3 * k.L * g ** 0.5
    air = _dull(r, T, _vary(r, 105.0, 0.1) * k.f, 0.04 * k.L * g, attack=0.009 * k.tc, q=0.9)
    body = _thump(r, T, 96.0 * k.f, 60.0 * k.f, 0.04, 0.045 * k.L * g, attack=0.008 * k.tc, harm=0.2)
    felt = _burst(r, 0.06, 240.0 * k.f, 1000.0, 0.012 * k.tc, attack=0.005 * k.tc)
    return _mix([(air, 0, 0), (body, 0, -5 + k.lo), (felt, 0, -15 + k.h)], T)


@design('thump/soft', kind='thud',
        says=('a soft low thump', 'something soft and heavy landing', 'a dull muffled bump'),
        uses=('thud', 'thump', 'element lands', 'card lands', 'drop'), domains=DOM, dur=0.3, axes=RING, prior=0.9)
def thump_soft(c, r):
    """an element landing in a product film (_soft_thump), in a small room"""
    return _out(c, r, _soft_thump(r, _k(c), _ring(c)), room='room', spread=0.15, tilt=-2.0, crest=15.4)


@design('knock/soft', kind='thud',
        says=('a small rounded knock', 'a soft hollow tock', 'a gentle tap on a small wooden box'),
        uses=('knock', 'lock in', 'snap into place', 'dock', 'land'), domains=DOM, dur=0.3, axes=RING)
def knock_soft(c, r):
    """an element locking in: a small hollow body tapped with a soft tip (a 0.9 ms contact: nothing above 2 kHz is
    struck hard). Three shell modes from 620 Hz ring 55-130 ms inside a cloud of 30 shorter ones, so it is a knock
    with a colour and not a note; the air inside answers low and short. Mono, as small film sounds are"""
    k = _k(c)
    g = _ring(c)
    T = 0.3 * k.L * g ** 0.5
    f0 = _vary(r, 620.0, 0.06) * k.f
    shell = (f0 * np.array([1.0, 2.32, 3.9]), T60 / (np.array([0.13, 0.085, 0.055]) * g), np.array([0.45, 0.3, 0.2]))
    tock = _struck(r, _join(shell, _cloud(r, 300.0 * k.f, 4500.0 * k.f, 30, 0.09 * g, 0.03, -1.5)), T,
                   0.0009 * k.tc, noise=(0.15, 0.25, 0.004), rough=0.4)
    low = _dull(r, 0.09, 185.0 * k.f, 0.014, attack=0.002)
    return _out(c, r, _mix([(tock, 0, 0), (low, 0, -11 + k.lo)], T), room='room', spread=0.03, crest=18.9)


# ------------------------------------------------------------------------------------------------ builds and falls
def _build(c, T, peak=0.93):
    """the level of a build over T s (0..1 per sample) and its progress 0..1: amplitude grows as progress ** p up
    to the cut at the share `peak` of the length (p 1.5: the nearly linear growth measured on film builds; a
    harder build holds back and arrives late, p up to 2.6), then falls to nothing within the rest (cos^2)"""
    n = dsp.n_of(T)
    u = (np.arange(n) + 0.5) / n
    w = np.minimum(u / peak, 1.0)
    p = c.lerp('hard', 1.0, 2.6, 1.5)
    return np.where(u < peak, w ** p, np.cos(0.5 * np.pi * np.clip((u - peak) / (1.0 - peak), 0.0, 1.0)) ** 2), w


def _tremor(n, w, lo, hi, depth):
    """a tremolo that speeds up from lo to hi Hz and deepens to `depth` with the progress w of a build, then lets
    go over its last tenth, so the cut itself is always the loudest moment"""
    ph = np.cumsum(lo * (hi / lo) ** w) / SR
    return 1.0 - depth * w * (1.0 - w ** 10) * (0.5 - 0.5 * np.cos(2 * np.pi * ph))


@design('rise/air', kind='rise',
        says=('a rising rush of air that opens up and gets brighter', 'a noise riser building up',
              'air swelling to a peak'),
        uses=('riser', 'build', 'noise riser', 'reveal', 'transition', 'uplifter'), domains=FILM, dur=2.0, axes=MOVE,
        prior=1.1)
def rise_air(c, r):
    """a noise riser with no pitch in it, in two layers that leave the band between 2 and 5 kHz thin (as the
    measured film builds do): a body of band noise that grows to the cut at 93 % of the length while its hump
    climbs two octaves from 350 Hz, and a top of air around 9 kHz that enters over the second half and is wide.
    A tremolo speeds up from 5 to 13 Hz (motion) and the stereo field opens; after the cut it falls away within
    7 % of the length and the room rings out"""
    k = _k(c)
    T = _span(c, k, 2.0, 0.4)
    a, w = _build(c, T)
    a = a * _tremor(len(a), w, 5.0, 13.0, 0.3 * (1 + c.motion))
    lift = 2.0 ** (0.3 * k.e + 0.4 * c.bright)
    body = _air(r, a, 350.0 * k.f * lift * 4.3 ** (w ** 1.3), 5.0, 9.0 * 2.0 ** (-0.4 * c.bright),
                span=(90.0, 9000.0), bands=9, gust=0.15, rate=(2.0, 25.0), side=(0.1, 0.6))
    top = _air(r, a * w ** 1.5, 9000.0 * k.f ** 0.5 * lift ** 0.5, 12.0, 4.0, span=(3500.0, 16000.0), bands=6,
               gust=0.25, rate=(2.0, 25.0), side=(0.7, 0.9))
    y = _mix([(_bed(body), 0, 0), (_bed(top), 0, -2 + 4 * c.bright)])
    return _out(c, r, fx.width(y, 0.6 + 0.5 * w), room='room')


@design('rise/tension', kind='rise',
        says=('a tense rising synthesizer swell', 'a dark detuned chord growing louder and brighter',
              'a suspenseful build-up'),
        uses=('riser', 'tension', 'build', 'suspense', 'anticipation'), domains=FILM, dur=1.8, pitch='D3', axes=MOVE,
        prior=1.05)
def rise_tension(c, r):
    """a tonal build: a stack of seven saws on the root that crossfades from a tight unison (8 cents) to a wide
    one (36 cents: the beating speeds up), a fifth entering above it, a sub an octave under; the pitch climbs two
    to three semitones in all, a ladder filter opens from 570 Hz to 6 kHz, a tremolo speeds up, and a breath of
    air comes in at the top. It grows to the cut at 93 % of the length and falls away"""
    k = _k(c)
    T = _span(c, k, 1.8, 0.5)
    a, w = _build(c, T)
    n = len(a)
    vo = _voice()
    freq = c.hz() * 2.0 ** ((2.0 + c.motion) * w ** 2 / 12)
    tight = vo.unison(freq, 'saw', r, voices=7, detune=8.0, spread=0.5, drift=3.0)
    wide = vo.unison(freq, 'saw', r, voices=7, detune=36.0, spread=0.9, drift=5.0)
    fifth = vo.unison(freq * 2.0 ** (7 / 12), 'saw', r, voices=5, detune=18.0, spread=0.7, drift=4.0)
    x = np.sqrt(1.0 - w)[:, None] * tight + np.sqrt(w)[:, None] * wide + 0.5 * (w ** 1.5)[:, None] * fifth
    x = x + 10 ** ((-12 + k.lo) / 20) * vo.sine(freq / 2)[:, None]
    x = vo.ladder(x, 570.0 * k.f * 2.0 ** (0.5 * c.bright + 0.3 * k.e + 3.4 * w ** 1.4), 0.18)
    x = x * (a * (0.75 + 0.25 * w ** 6) * _tremor(n, w, 3.0, 10.0, 0.12 * (1 + c.motion)))[:, None]
    air = _air(r, a * w, 3000.0 * 2.0 ** (1.2 * w), 4.0, 5.0, span=(1200.0, 15000.0), bands=7, side=(0.8, 0.95))
    y = vo.chorus(_mix([(_bed(x), 0, 0), (_bed(air), 0, -16)]), mix=0.35)
    return _out(c, r, y, room='room', tilt=-3.0 * c.size)


@design('rise/reverse', kind='rise',
        says=('a reversed cymbal swelling up to an abrupt stop', 'a backwards shimmering swell',
              'a reverse crash building'),
        uses=('reverse', 'reverse swell', 'riser', 'swell into a cut', 'reverse cymbal'), domains=FILM, dur=1.5,
        axes=BASE, prior=1.05)
def rise_reverse(c, r):
    """the tail of a struck sheet of metal played backwards: 44 inharmonic modes between 380 Hz and 11 kHz and the
    wash around them, each decaying at its own rate (the high ones faster), so in reverse the low ones swell first
    and the top arrives last. The loudest moment is the cut at 92 % of the length; the last 8 % are the same tail
    running forward again under a fast fade, so it stops without a step"""
    k = _k(c)
    T = _span(c, k, 1.5, 0.3)
    n = dsp.n_of(0.92 * T)
    tex = _tex()
    cnt = 44
    f = np.exp(r.uniform(np.log(380.0 * k.f), np.log(11000.0), cnt))
    tau = 0.26 * T * (f / 1000.0) ** -0.45 * c.lerp('hard', 1.5, 0.6, 1.0, log=True)
    metal = np.zeros((n, 2))
    tex.rings(metal, r.uniform(0.0, 0.004, cnt), r.uniform(0.3, 1.0, cnt) * (f / 1000.0) ** -0.3,
              r.uniform(-0.8, 0.8, cnt), f, tau, 0.0005, wrap=False)
    cen = 400.0 * k.f * (13000.0 / 400.0) ** (np.arange(8) / 7)
    m = sfft.next_fast_len(n, real=True)
    tw = np.arange(m // 48) * 48 / SR
    fall = np.exp(-tw[None, :] / (0.3 * T * (cen[:, None] / 1000.0) ** -0.5)) * (tw[None, :] < n / SR)
    wash = tex.band_noise(m, r, cen, fall * 2.0 ** (0.2 * (c.bright + k.e) * np.log2(cen[:, None] / 1000.0)), 0.8)[:n]
    tail = _norm(metal) + 0.8 * _bed(wash)
    j = dsp.n_of(T) - n
    y = np.concatenate([tail[::-1], tail[1:j + 1] * (np.cos(np.linspace(0.0, np.pi / 2, j)) ** 2)[:, None]])
    i = dsp.n_of(0.35 * T)
    y[:i] *= (np.linspace(0.0, 1.0, i) ** 2)[:, None]
    return _out(c, r, y, room='room')


@design('fall/air', kind='fall',
        says=('a rush of air falling away and fading out', 'a whoosh trailing off and getting darker',
              'air dying away'),
        uses=('downlifter', 'fall', 'exit', 'after a hit', 'wind down'), domains=FILM, dur=1.6, axes=MOVE,
        prior=1.1)
def fall_air(c, r):
    """what follows a hit: air that is full within the first 7 % of the length and then falls away, in two layers
    that leave the band between 2 and 5 kHz thin: a body whose hump sinks three and a half octaves from 1.6 kHz
    and a wide top around 9 kHz that is gone by the middle; the stereo field closes, and a low rumble under it
    all outlasts the rest"""
    k = _k(c)
    a, pk = _move(c, r, _span(c, k, 1.6, 0.3), 0.07, (1.2, 1.25))
    u = (np.arange(len(a)) + 0.5) / len(a)
    w = np.clip((u - pk) / (1.0 - pk), 0.0, 1.0)
    lift = 2.0 ** (0.3 * k.e + 0.4 * c.bright)
    body = _air(r, a, 1600.0 * k.f * lift * 2.0 ** (-3.5 * w ** 0.6), 5.0, 9.0 * 2.0 ** (-0.4 * c.bright),
                span=(50.0, 9000.0), bands=9, gust=0.3, rate=(1.5, 20.0), side=(0.2, 0.6))
    top = _air(r, a * (1.0 - w) ** 3, 9000.0 * k.f ** 0.5 * lift ** 0.5, 12.0, 4.0, span=(3500.0, 16000.0), bands=6,
               gust=0.25, rate=(2.0, 25.0), side=(0.7, 0.9))
    y = _mix([(_bed(body), 0, 0), (_bed(top), 0, -3 + 4 * c.bright)])
    y = fx.width(y, np.clip(1.0 + 0.5 * c.motion - 0.7 * w, 0.0, 2.0))
    low = _weight(r, a ** 0.7, 130.0 * k.f, 0.03)
    return _out(c, r, _mix([(y, 0, 0), (low, 0, -10 + k.lo)]), room='studio')


@design('fall/sub', kind='boom',
        says=('a deep sub bass drop that swells and fades', 'a very low falling bass tone',
              'a low boom blooming and sinking'),
        uses=('sub drop', 'bass drop', 'drop', 'low drop after a cut', 'impact'), domains=FILM, dur=1.4, axes=RING,
        allows=('pre_swell',))        # the sub blooms a third of a second after the cut: that swell is the design
def fall_sub(c, r):
    """a sub drop under a cut: a soft knock and a faint bright tick mark the cut, then a sine sinking from 66 to
    44 Hz blooms to its peak 0.3 s later and decays straight in dB at about 40 dB per second (the film study's sub
    drop); driven gently, so odd harmonics carry it on small speakers. Centred and nearly dry"""
    k = _k(c)
    g = _ring(c)
    T = _span(c, k, 1.4, 0.4)
    n = dsp.n_of(T)
    t = _t(n)
    s = T / 1.4
    f = k.f * (44.0 + _vary(r, 22.0, 0.08) * np.exp(-t / (0.35 * s)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    bloom = 0.3 * s * k.tc ** 0.5
    amp = np.where(t < bloom, np.sin(0.5 * np.pi * np.minimum(t / bloom, 1.0)) ** 1.6,
                   np.exp(-np.maximum(t - bloom, 0.0) / (0.22 * s * g)))
    sub = fx.saturate((np.sin(ph) + 0.2 * np.sin(2 * ph)) * amp, 6.0 + 3.0 * k.e, 'tanh', level='peak')
    knock = _dull(r, 0.08, 160.0 * k.f, 0.012, attack=0.002)
    tick = _burst(r, 0.025, 1800.0, 7000.0, 0.003 * k.tc)
    y = _mix([(_norm(sub), 0, k.lo / 2), (knock, 0, -13 + k.h / 2), (tick, 0, -22 + k.h)], T)
    return _out(c, r, dsp.fade(y, 0.0, 0.25 * T), room='booth', spread=0.05)


@design('suck/in', kind='rise',
        says=('a short inward suck of air', 'a quick reverse whoosh pulling in and stopping', 'air being sucked in'),
        uses=('suck', 'pre-hit', 'inhale', 'lead in', 'reverse whoosh'), domains=FILM, dur=0.45, axes=MOVE,
        prior=0.9)
def suck_in(c, r):
    """the breath before a hit: air that accelerates into the cut at 93 % of the length (most of its rise in the
    last third), its hump climbing two and a half octaves from 250 Hz while the stereo field closes from wide
    onto the centre, then nothing"""
    k = _k(c)
    T = _span(c, k, 0.45, 0.15)
    n = dsp.n_of(T)
    u = (np.arange(n) + 0.5) / n
    w = np.minimum(u / 0.93, 1.0)
    a = np.where(u < 0.93, 10 ** (-50.0 * (1.0 - w) ** c.lerp('hard', 1.1, 0.55, 0.75) / 20),
                 np.cos(0.5 * np.pi * np.clip((u - 0.93) / 0.07, 0.0, 1.0)) ** 2)
    fc = 250.0 * k.f * 2.0 ** (0.3 * k.e + 0.4 * c.bright) * 6.0 ** (w ** 1.5)
    y = _bed(_air(r, a, fc, 5.0, 8.0 * 2.0 ** (-0.4 * c.bright), gust=0.2, rate=(4.0, 40.0), side=(0.3, 0.9)))
    y = fx.width(y, np.clip(1.3 + 0.4 * c.motion - 1.1 * w, 0.0, 2.0))
    low = _weight(r, a, 170.0 * k.f)
    return _out(c, r, _mix([(y, 0, 0), (low, 0, -9 + k.lo)]), room='booth')


# ------------------------------------------------------------------------------------------------ small arrivals
def _cavity(r, f0, bend, seconds, attack, tau, air=0.3):
    """the pop of a small cavity (a mouth, the neck of a bottle, a bubble of air under a skin), mono, peak 1: the
    release of pressure (a pulse a few ms long) and, `air` 0..1 of the energy, the noise of the air rushing out
    after it (it rises in `attack` s and dies with time constant tau s) through the cavity's first three
    resonances: f0 Hz with a quality of 10, and two weaker ones 2.1 and 3.7 times higher. A cavity that small is
    damped: it rings a dozen cycles, and the puff sets how long the pop is. The resonances start `bend` semitones
    high and settle within 14 ms as the opening relaxes. Nothing in it is an oscillator, so it is heard as a pop
    and not as a beep; recorded pops hold about two thirds of their energy near the pitch"""
    n = dsp.n_of(seconds)
    t = _t(n)
    push = _env(n, 0.3 * attack, 0.5 * attack)
    rush = dsp.butter(r.standard_normal(n), 'low', 4000.0) * _env(n, attack, _vary(r, tau, 0.12))
    exc = (1.0 - air) * push / np.sqrt(np.sum(push ** 2)) + air * rush / np.sqrt(np.sum(rush ** 2))
    f = f0 * 2.0 ** (bend / 12 * np.exp(-t / 0.014))
    y = sum(lv * dsp.svf(exc, f * q, qq, 'bandn')
            for q, lv, qq in ((1.0, 1.0, 10.0), (2.1, 0.2, 7.0), (3.7, 0.06, 5.0)))
    return dsp.fade(_norm(y), 0.0, 0.25 * seconds)


@design('pop/soft', kind='pop',
        says=('a soft rounded pop', 'a gentle mid-pitched bloop', 'a small soft mouth pop'),
        uses=('pop', 'appear', 'bubble', 'chip', 'badge'), domains=DOM, dur=0.3, axes=RING, prior=1.05)
def pop_soft(c, r):
    """a chip or a bubble appearing: a cavity pop at 620 Hz (about 12 ms to its peak, 20 dB down 50 ms later,
    settling 2.5 semitones as it opens, two thirds pulse and one third air), a faint low thump and the tick of
    lips parting far under"""
    k = _k(c)
    g = _ring(c)
    pop = _cavity(r, _vary(r, 620.0, 0.03) * k.f, c.lerp('hard', 1.5, 4.5, 2.5), 0.3 * k.L, 0.009 * k.tc,
                  0.02 * k.L * g)
    thump = _dull(r, 0.08, 150.0 * k.f, 0.014, attack=0.004)
    lip = _burst(r, 0.012, 2500.0, 7000.0, 0.0015 * k.tc)
    return _out(c, r, _mix([(pop, 0, 0), (thump, 0, -17 + k.lo), (lip, 0, -30 + k.h)]), room='room', spread=0.1,
                crest=16.0)


@design('pop/bubble', kind='pop',
        says=('a bubble popping', 'a small watery bloop', 'a liquid plop'),
        uses=('pop', 'bubble', 'bloop', 'appear', 'water drop'), domains=DOM, dur=0.25, axes=RING,
        allows=('ring',))     # a bubble and its daughter are two near-pure tones: the comb detector reads the pair
def pop_bubble(c, r):
    """a bubble of about 4.7 mm radius under a surface (Minnaert: 3.26 / radius, so 700 Hz) whose pitch climbs
    three semitones while it rings 26 ms, as a bubble near the surface does, a smaller daughter bubble 50-80 ms
    later, and the wet splash of the surface"""
    k = _k(c)
    g = _ring(c)
    f0 = _vary(r, 700.0, 0.06) * k.f
    tau = 0.026 * k.L * g
    out = np.zeros((dsp.n_of(0.3 * k.L), 2))
    gap = r.uniform(0.05, 0.08) * k.L
    _tex().rings(out, [0.0, gap], [1.0, r.uniform(0.3, 0.5)], [0.0, r.uniform(-0.3, 0.3)],
                 [f0, f0 * r.uniform(1.5, 1.9)], [tau, 0.6 * tau], [0.003 * k.tc, 0.002 * k.tc],
                 [0.19 / (4 * tau), 0.19 / (2.4 * tau)], wrap=False)
    wet = _burst(r, 0.04, 1200.0, 5000.0, 0.008, attack=0.002)
    return _out(c, r, _mix([(_norm(out), 0, 0), (wet, 0.001, -15 + k.h)]), room='room', spread=0.2)


@design('pop/pluck', kind='pop',
        says=('a soft pop with a short plucked note', 'a small muted pluck with a round body', 'a tiny plucked tine'),
        uses=('pop', 'appear', 'pluck', 'icon lands', 'badge'), domains=DOM, dur=0.35, pitch='A5', voice='kalimba',
        axes=RING)
def pop_pluck(c, r):
    """a pop with a pitch: a short damped note of the voice (a kalimba tine unless another is asked for, stopped
    after 70 ms) on the pitch, over a cavity pop an octave below it"""
    from ..synth import voices
    k = _k(c)
    g = _ring(c)
    f0 = c.hz()
    note = voices.play(c.voice, f0, vel=float(np.clip(0.6 + 0.2 * k.e, 0.1, 1.0)), dur=0.07 * g * k.L, r=r,
                       hard=c.hard, bright=c.bright)
    pop = _cavity(r, 0.5 * f0, 3.0, 0.12 * k.L, 0.005 * k.tc, 0.02 * k.L)
    return _out(c, r, _mix([(_norm(note), 0.004, 0), (pop, 0, -5 + k.lo)]), room='room', tilt=-4.0 * c.size)


@design('tock/wood', kind='tick',
        says=('a tiny wooden tick', 'a small dry tock of wood', 'a light tap on a wooden block'),
        uses=('tock', 'tick', 'tap', 'snap into place', 'detent', 'lock'), domains=DOM, dur=0.2, material='wood',
        axes=RING)
def tock_wood(c, r):
    """a small piece of wood tapped with another (a 0.2 ms contact): a bar's first three modes from 1.5 kHz
    (free-bar ratios 1 : 2.76 : 5.40, 20-60 ms) inside a cloud of 30 short ones up to 7 kHz, the tick of the
    contact, and the hand that holds it low under them"""
    k = _k(c)
    g = _ring(c)
    T = 0.2 * k.L * g ** 0.5
    f0 = _vary(r, 1500.0, 0.07) * k.f
    bar = (f0 * np.array([1.0, 2.76, 5.4]), T60 / (np.array([0.06, 0.035, 0.02]) * g), np.array([0.5, 0.35, 0.2]))
    tick = _struck(r, _join(bar, _cloud(r, 900.0 * k.f, 7000.0 * k.f, 30, 0.06 * g, 0.02, 0.0)), T, 0.0002 * k.tc,
                   noise=(0.3, 0.25, 0.003), rough=0.4)
    low = _dull(r, 0.06, 260.0 * k.f, 0.008, attack=0.001)
    return _out(c, r, _mix([(tick, 0, 0), (low, 0, -16 + k.lo)], T), room='room', spread=0.05, crest=20.3)


@design('tock/glass', kind='tick',
        says=('a tiny glass tick', 'a small bright tink of glass', 'a light tap on a glass'),
        uses=('tick', 'tap', 'tock', 'snap into place', 'detent', 'select'), domains=DOM, dur=0.25, material='glass',
        axes=RING, allows=('ring',))      # glass rings in steady partials: the comb detector reads any such tone as one
def tock_glass(c, r):
    """a small thick piece of glass tapped with something hard (a 0.09 ms contact): the modes of a disc from
    2.9 kHz as beating pairs, damped by the glass itself and by the fingers that hold it (modal.decay_rate), so it
    tinks and is gone in a quarter of a second"""
    k = _k(c)
    g = _ring(c)
    T = 0.3 * k.L * g ** 0.5
    f0 = _vary(r, 2900.0, 0.06) * k.f
    rat = PLATE * (1 + 0.01 * r.standard_normal(len(PLATE)))
    f = np.concatenate([f0 * rat, f0 * rat * (1 + r.uniform(0.002, 0.006, len(rat)))])
    body = (f, modal.decay_rate(f, 'glass', support=28.0) / g, np.tile(rat ** -0.6 * r.uniform(0.5, 1.0, len(rat)), 2))
    ting = _struck(r, body, T, 0.00009 * k.tc, noise=(0.3, 0.04, 0.0015))
    low = _dull(r, 0.05, 420.0 * k.f, 0.006, attack=0.0008)
    return _out(c, r, _mix([(ting, 0, 0), (low, 0, -20 + k.lo)], T), room='booth', spread=0.15, crest=17.0)


@design('blip/soft', kind='tick',
        says=('a soft short high blip', 'a gentle rounded electronic pip', 'a small soft digital tick'),
        uses=('blip', 'counter', 'cursor', 'tick', 'count up'), domains=DOM, dur=0.2, pitch='B5',
        material='digital', axes=RING, prior=0.5)       # a request for a beep or a timer means a tone, not this
def blip_soft(c, r):
    """a counter ticking or a cursor: one FM pair on the pitch (a 6 ms rise, 140 ms decay) whose modulator an
    octave up dies within 30 ms, so the start has the colour of a small struck thing and the rest is a round
    tone; a short breath of noise around the pitch far under it, and a small room"""
    k = _k(c)
    g = _ring(c)
    f0 = c.hz()
    ops = [dict(out=1.0, mod=(1,), env=(0.006 * k.tc, _vary(r, 0.14, 0.15) * g * k.L, 0.0, 0.04 * g)),
           dict(ratio=2.0, index=_vary(r, 0.9, 0.3) * 2.0 ** (c.bright + 0.5 * k.e), env=(0.002, 0.03, 0.0, 0.02))]
    tone = _voice().fm(f0, ops, 0.8, 0.03, r)
    breath = _burst(r, 0.04, 0.7 * f0, 3.0 * f0, 0.008, attack=0.003)
    return _out(c, r, _mix([(_norm(tone), 0, 0), (breath, 0, -18 + k.h)]), room='booth', tilt=-4.0 * c.size, crest=14.9)


# ------------------------------------------------------------------------------------------------ light and detail
@design('shine/glint', kind='chime',
        says=('a short bright glint', 'a quick high metallic shimmer ping', 'a small bright shing'),
        uses=('shimmer', 'glint', 'highlight', 'shine', 'lens flare'), domains=DOM, dur=0.7, pitch='A6', axes=RING,
        prior=0.95)
def shine_glint(c, r):
    """light catching an edge: one clear high partial on the pitch and five inharmonic ones far above it (2.9 to
    6.9 times: over 5 kHz, clear of the band where a ring turns harsh), each a beating pair spread across the
    stereo field, that swell in over 6 ms instead of being struck and die within a few hundred ms (the higher the
    sooner); a breath of air above 6 kHz comes with the onset"""
    k = _k(c)
    g = _ring(c)
    n = dsp.n_of(0.7 * k.L * g ** 0.5)
    rat = np.array([1.0, 2.92, 3.6, 4.4, 5.6, 6.9]) * (1 + 0.01 * r.standard_normal(6))
    rat[0] = 1.0
    f = c.hz() * np.concatenate([rat, rat * (1 + r.uniform(0.001, 0.004, 6))])
    gain = np.tile(np.where(rat < 1.2, 1.0, 0.45 * rat ** (-0.3 + 0.4 * c.bright) * r.uniform(0.5, 1.0, 6)), 2)
    out = np.zeros((n, 2))
    _tex().rings(out, np.tile(r.uniform(0.0, 0.006, 6), 2), gain, r.uniform(-0.7, 0.7, 12), f,
                 np.tile(0.16 * g * k.L * rat ** -0.5, 2), 0.006 * k.tc, wrap=False)
    air = _burst(r, 0.12, 6000.0, 14000.0, 0.03, attack=0.012)
    return _out(c, r, _mix([(_norm(out), 0, 0), (air, 0, -22 + k.h)]), room='plate', tilt=-4.0 * c.size)


@design('shine/sweep', kind='texture',
        says=('a shimmer sweeping across from one side to the other', 'a sparkling glittering sweep',
              'a bright twinkling pass'),
        uses=('shimmer', 'shine', 'highlight sweep', 'glitter', 'light sweep'), domains=DOM, dur=1.3, pitch='A5',
        axes=RING + ('motion',))
def shine_sweep(c, r):
    """a highlight travelling across: small struck bars on a pentatonic scale over the pitch arriving up to 90 a
    second under a rounded swell (in clusters, as glitter catches the light), their pitches climbing and their
    place moving from one side to the other with the sweep (motion = how far): a third of them on the two notes
    just under 2 kHz, the rest above 4.7 kHz; under them a faint band of air that travels along, and one quiet
    held tone on the pitch with its fifth, which is what makes a cloud of grains a shimmer"""
    k = _k(c)
    g = _ring(c)
    T = _span(c, k, 1.3, 0.3)
    a, _ = _move(c, r, T, 0.42, (2.0, 1.8), depth=40.0)
    n = len(a)
    tex = _tex()
    t, _ = tex.arrivals(r, 90.0 * 2.0 ** (0.5 * k.e) * a[::48] ** 1.2, T, 0.2, 0.02)
    t = t[t < T - 0.02]
    u = t / T
    notes = _scale(c.hz(), 1700.0, 12000.0)
    low = int((notes < 2100.0).sum())
    idx = low + np.clip((0.55 * u + 0.45 * r.uniform(size=len(t))) * (len(notes) - low), 0, len(notes) - low - 1)
    idx = np.where(r.uniform(size=len(t)) < 0.3, r.integers(0, max(low, 1), len(t)), idx.astype(int))
    f = notes[idx] * 2.0 ** (5 * r.standard_normal(len(t)) / 1200)
    reach = r.choice([-1.0, 1.0]) * float(np.clip(0.6 + 0.35 * c.motion, 0.0, 1.0))
    way = np.cumsum(a ** 2)
    way = reach * (2 * way / way[-1] - 1)             # across while it is loud: half way at half of the energy
    pan = np.clip(way[np.minimum((t * SR).astype(int), n - 1)] + 0.12 * r.standard_normal(len(t)), -1.0, 1.0)
    gain = a[np.minimum((t * SR).astype(int), n - 1)] * np.exp(0.3 * r.standard_normal(len(t)))
    dec = r.uniform(0.04, 0.12, len(t)) * g * np.sqrt(2500.0 / f)
    bars = _chimes(r, n + dsp.n_of(0.3), t, gain, f, dec, pan, r.uniform(0.001, 0.004, len(t)) * k.tc)
    air = _air(r, a, 5000.0 * 2.0 ** (0.7 * (np.arange(n) + 0.5) / n), 6.0, 5.0, span=(1500.0, 15000.0), bands=7,
               gust=0.3, side=(0.8, 0.9), pan=way)
    held = sum(lv * np.sin(2 * np.pi * q * c.hz() * _t(n) + r.uniform(0, 6.28)) for q, lv in ((1.0, 1.0), (1.5, 0.5)))
    y = _mix([(_norm(bars), 0, 0), (_bed(air), 0, -20), (_norm(held * a ** 0.7), 0, -15 + k.lo)])
    return _out(c, r, y, room='plate', tilt=-4.0 * c.size)


@design('sparkle/burst', kind='chime',
        says=('a small burst of bright sparkles', 'a quick scatter of tiny high chimes', 'a short twinkling flurry'),
        uses=('sparkle', 'shimmer', 'twinkle', 'burst', 'magic'), domains=DOM, dur=0.6, pitch='C7', axes=RING,
        prior=0.95)
def sparkle_burst(c, r):
    """a handful of bright grains thrown at once: about 25 small struck bars (more with energy) whose arrivals thin
    out exponentially after the first and largest one, power-law sizes (a big grain is lower and rings longer), on
    the pentatonic scale of the pitch between 4.7 and 13 kHz, scattered wide"""
    k = _k(c)
    g = _ring(c)
    n = dsp.n_of(0.6 * k.L * g ** 0.5)
    cnt = 6 + int(r.poisson(20.0 * 2.0 ** (0.6 * k.e)))
    t = np.sort(r.exponential(0.09 * k.L, cnt))
    t -= t[0]
    s = _tex().sizes(r, cnt, 0.08, 1.0, 'power', 1.7)
    s[0] = 1.0
    notes = _scale(c.hz(), 4700.0 * k.f, 13000.0, gap=(0.0, 0.0))
    f = notes[np.clip((r.uniform(size=cnt) * (1.0 - 0.4 * s) * len(notes)).astype(int), 0, len(notes) - 1)]
    dec = (0.03 + 0.09 * s) * g * np.sqrt(4000.0 / f)
    bars = _chimes(r, n, t, s ** 0.7, f, dec, r.uniform(-0.9, 0.9, cnt), r.uniform(0.0005, 0.003, cnt) * k.tc,
                   split=0.002)
    return _out(c, r, _norm(bars), room='plate')


@design('glitch/short', kind='noise',
        says=('a short digital stutter', 'a quick glitchy buzz that repeats faster and stops', 'a brief data glitch'),
        uses=('glitch', 'stutter', 'digital glitch', 'error', 'transition'), domains=DOM, dur=0.32, material='digital',
        axes=BASE, allows=('ring',))      # one fragment repeated every 16-36 ms is a comb by construction: the stutter
def glitch_short(c, r):
    """a designed stutter: one 40 ms fragment of crushed FM (a buzzing tone at 640 Hz, 5 bits at 7 kHz, kept under
    2 kHz) stuck and repeated on a gate that speeds up from 36 to about 20 ms, alternating sides, now and then
    replaced by a burst of fizz above 5 kHz or left out, growing by 7 dB into the last slice, which is let ring
    out: every slice has millisecond fades, the band between 2 and 5 kHz stays thin, and it ends with a decay,
    not a cut"""
    k = _k(c)
    T = _span(c, k, 0.32, 0.12)
    n = dsp.n_of(T)
    ops = [dict(out=1.0, mod=(1,), env=(0.0005, 0.3, 0.7, 0.004)),
           dict(ratio=3.01, index=2.0 + k.e, env=(0.0005, 0.06, 0.5, 0.004))]
    data = _norm(_voice().fm(_vary(r, 640.0, 0.12) * k.f, ops, 0.9, 0.05, r).mean(axis=1))
    data = dsp.butter(fx.crush(data, bits=5, rate=7000.0 * 2.0 ** (0.6 * c.bright)), 'low', 1900.0, order=4)
    fizz = _burst(r, 0.05, 5200.0, 13000.0, 0.03, attack=0.0004)
    out = np.zeros((n, 2))
    side = r.choice([-1.0, 1.0])
    starts, t, p = [], 0.0, 0.036
    while t + p < T - 0.03:
        starts.append((t, p))
        t, p = t + p, max(0.016, 0.93 * p)
    for i, (ts, ps) in enumerate(starts):
        pick = r.uniform()
        if 0 < i < len(starts) - 1 and pick > 0.9:
            continue
        seg = (fizz if pick < 0.22 else data)[:dsp.n_of(0.72 * ps)]
        seg = dsp.fade(seg, 0.0012 * k.tc, 0.002) * 10 ** (-7.0 * (1.0 - i / max(1, len(starts) - 1)) / 20)
        dsp.place(out, dsp.pan_to_stereo(seg, side * 0.6 * (-1.0) ** i), ts)
    last = data[:dsp.n_of(0.03)] * np.exp(-_t(dsp.n_of(0.03)) / 0.009)
    dsp.place(out, dsp.pan_to_stereo(dsp.fade(last, 0.0012 * k.tc, 0.004)), t)
    return _out(c, r, _mix([(out, 0, 0), (dsp.pan_to_stereo(fizz), t, -14 + k.h)], T), room='booth')


def _key(r, k, g, heavy=0.0):
    """one soft key press, mono, peak about 1, as a key really sounds: two contacts. Down: the keycap lands on its
    housing, plastic on plastic (a 0.1 ms contact), and rings the whole key for a few tens of ms (a cloud of 36
    short modes spread evenly from 500 Hz to 9 kHz, drawn anew for every key, so no band stands out and none
    rings as a note), with the click of the plastic itself (2 ms of noise above 3.5 kHz) and the case under it
    (a dull knock near 210 Hz). Up, 70-100 ms later: the stem meets the top of the housing, a thinner tick on
    other, higher modes, 14 dB down. heavy 0..1 = a larger key (the space bar: half an octave lower, more
    case)"""
    s = 2.0 ** (-0.5 * heavy)
    tc = _vary(r, 0.0001, 0.2) * k.tc
    down = _struck(r, _cloud(r, 500.0 * k.f * s, 9000.0 * k.f * s, 36, 0.035 * g, 0.015, 0.0), 0.08, tc, vary=0.5,
                   rough=0.4)
    click = _burst(r, 0.012, 3500.0 * k.f ** 0.5, 11000.0, 0.002 * k.tc, attack=0.0002)
    case = _dull(r, 0.05, _vary(r, 210.0, 0.12) * k.f * s, 0.009, attack=0.0015)
    up = _struck(r, _cloud(r, 1500.0 * k.f * s, 9000.0 * k.f * s, 20, 0.03 * g, 0.014, 1.0), 0.05, 0.8 * tc, vary=0.5,
                 rough=0.3)
    return _mix([(down, 0, 0), (click, 0, -11 + k.h), (case, 0.0005, -9 + k.lo + 4 * heavy),
                 (up, r.uniform(0.07, 0.1), -14 + k.h / 2)])


@design('type/soft', kind='tick',
        says=('one soft key press', 'a small muted plastic tick', 'a soft keyboard key'),
        uses=('type', 'key press', 'kinetic type', 'text', 'letter'), domains=DOM, dur=0.18, axes=RING, prior=1.02,
        allows=('ring',))     # the release is a tick of a few ms alone in the tail: the comb detector reads its ripple
def type_soft(c, r):
    """one key for kinetic type (_key: down and up): every seed is another key under another finger"""
    return _out(c, r, _key(r, _k(c), _ring(c)), room='booth', spread=0.02, crest=22.2)


@design('type/run', kind='texture',
        says=('soft typing on a keyboard', 'a run of soft key ticks', 'quick quiet keystrokes'),
        uses=('typing run', 'type', 'text reveal', 'kinetic type', 'typing'), domains=DOM, dur=1.5,
        axes=RING + ('motion',), allows=('ring',))        # as type/soft: the short release ticks, not a comb
def type_run(c, r):
    """a line being typed: keys (_key) at 10 a second (7 to 14 with motion; the film study's runs: 60-150 ms
    apart), the gaps log-normal so it never ticks like a clock, a breath twice as long after every three to seven
    keys (words), every key its own take at its own level (+-3 dB) and place, about one in eight a larger key.
    `dur` is the length of the run"""
    k = _k(c)
    g = _ring(c)
    T = _span(c, k, 1.5, 0.2)
    out = np.zeros((dsp.n_of(T), 2))
    t, word = 0.0, int(r.integers(3, 8))
    while t < T - 0.17:
        key = _key(r, k, g, float(r.uniform() < 0.12)) * 10 ** (3.0 * r.standard_normal() / 20)
        dsp.place(out, dsp.pan_to_stereo(key, r.uniform(-0.2, 0.2)), t)
        word -= 1
        t += (1.0 if word else 2.0) * float(np.exp(0.35 * r.standard_normal())) / (10.0 * 2.0 ** (0.5 * c.motion))
        word = word or int(r.integers(3, 8))
    return _out(c, r, out, room='booth')


# ------------------------------------------------------------------------------------------------ stingers
@design('sting/reveal', kind='stinger',
        says=('a soft swell into a clean low hit with a bright chime on top', 'a short reveal sting with a bell chord',
              'a whoosh, a thump and a ringing chime'),
        uses=('logo', 'reveal', 'sting', 'logo reveal', 'title reveal'), domains=DOM, dur=1.5, pitch='C6',
        voice='celesta', axes=RING, prior=1.05)
def sting_reveal(c, r):
    """a reveal in three parts that share one moment: air that swells over the first 30 % of the length and stops,
    a soft clean hit where it stops (a thump falling to 52 Hz and the puff of air around it), and on the hit a
    chord of the voice (root, fifth and octave, rolled by 12 ms) that rings out in a hall"""
    from ..synth import phrase
    k = _k(c)
    g = _ring(c)
    T = c.seconds()
    lead = 0.3 * T
    ns = dsp.n_of(lead)
    us = (np.arange(ns) + 0.5) / ns
    a = us ** c.lerp('hard', 1.2, 2.6, 1.6) * np.cos(0.5 * np.pi * np.clip((us - 0.95) / 0.05, 0.0, 1.0)) ** 2
    swell = _bed(_air(r, a, 500.0 * k.f * 2.0 ** (0.4 * c.bright) * 8.0 ** (us ** 1.3), 4.0, 6.0, gust=0.25,
                      side=(0.3, 0.9)))
    sub = _thump(r, 0.5, 105.0 * k.f, 52.0 * k.f, 0.05, 0.12 * k.L, attack=0.008 * k.tc, harm=0.2)
    puff = _dull(r, 0.2, 180.0 * k.f, 0.035, attack=0.005)
    root = c.hz()
    chord = phrase.play([(0.0, root, 1.0), (0.012, root * 2.0 ** (7 / 12), 0.75), (0.024, 2.0 * root, 0.65)],
                        c.voice, r, vel=float(np.clip(0.55 + 0.2 * k.e, 0.1, 1.0)), dur=0.8 * (T - lead) * g,
                        spread=0.4, bright=c.bright)
    y = _mix([(swell, 0, -9), (sub, lead, -3 + k.lo / 2), (puff, lead, -9 + k.h / 2), (_norm(chord), lead, 0)])
    return _out(c, r, y, room='hall', tilt=-3.0 * c.size)


@design('sting/tech', kind='stinger',
        says=('a falling electronic chirp landing on a low humming tone', 'a quick zap down into a deep synth note',
              'a short tech logo sting'),
        uses=('logo', 'sting', 'tech reveal', 'reveal', 'ident'), domains=DOM, dur=1.0, pitch='D2',
        material='digital', axes=RING, prior=1.1)
def sting_tech(c, r):
    """the designed sting of the film study: a chirp that falls from 1.5 kHz to the octave over the pitch in 85 ms
    (a sine with one overtone, straight on a log-frequency axis) and lands on a low hum on the pitch (three saws
    and a sub through a ladder filter that closes again) which sinks five semitones while it rings out; a tick
    marks the landing. Near mono"""
    k = _k(c)
    g = _ring(c)
    T = c.seconds()
    vo = _voice()
    f0 = c.hz()
    nc = dsp.n_of(0.085 * k.L)
    ph = 2 * np.pi * np.cumsum(np.geomspace(1500.0, 2.0 * f0, nc)) / SR
    chirp = (np.sin(ph) + 0.3 * np.sin(2 * ph)) * np.sin(np.pi * (np.arange(nc) + 0.5) / nc) ** 0.7
    hum = vo.analog(f0, 0.85, max(0.2, T - 0.5), r, sub=0.7, cutoff=260.0 * 2.0 ** c.bright, keytrack=0.0, env=1.6,
                    res=0.2,
                    osc=(dict(shape='saw', voices=3, detune=9.0, spread=0.3),
                         dict(shape='pulse', pw=0.35, semi=12.0, level=0.3)),
                    fenv=(0.004, 0.25, 0.2, 0.2), amp=(0.006 * k.tc, 0.5 * g, 0.25, 0.35 * g),
                    pitch=lambda t: -5.0 * (1.0 - np.exp(-t / 0.45)))
    tick = _burst(r, 0.02, 2000.0, 8000.0, 0.002 * k.tc)
    land = 0.085 * k.L
    y = _mix([(_norm(chirp), 0, -4), (_norm(hum), land - 0.006, k.lo / 2), (tick, land, -20 + k.h)])
    return _out(c, r, fx.saturate(y, 2.0 + 2.0 * k.e, 'tanh', level='peak'), room='room', spread=0.15,
                tilt=-4.0 * c.size)


@design('sting/soft', kind='stinger',
        says=('a soft thump with a warm piano chord ringing out', 'a mellow piano chord over a low bump',
              'a gentle closing chord on a felt piano'),
        uses=('logo', 'sting', 'end card', 'outro', 'reveal'), domains=DOM, dur=2.2, pitch='C4', voice='felt_piano',
        axes=RING, prior=1.15)
def sting_soft(c, r):
    """an end card: the soft thump of thump/soft and, 6 ms after it, an open chord of the voice (root, fifth and
    the third an octave up, rolled by 15 ms, played softly) that is held for most of the length and released"""
    from ..synth import phrase
    k = _k(c)
    g = _ring(c)
    T = c.seconds()
    root = c.hz()
    chord = phrase.play([(0.0, root, 1.0), (0.015, root * 2.0 ** (7 / 12), 0.8), (0.03, root * 2.0 ** (16 / 12), 0.75)],
                        c.voice, r, vel=float(np.clip(0.42 + 0.2 * k.e, 0.1, 1.0)), dur=0.7 * T * g, spread=0.4,
                        bright=c.bright, hard=c.hard)
    y = _mix([(_soft_thump(r, k, g), 0, -3 + k.lo / 2), (_norm(chord), 0.006, 0)])
    return _out(c, r, y, room='hall', tilt=-3.0 * c.size)
