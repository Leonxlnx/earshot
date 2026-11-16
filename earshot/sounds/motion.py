"""Motion-design sounds: what a product film, a title sequence or an interface animation is cut with. Air and
movement (whooshes, swooshes, flicks, a fly-by, a sliding panel), hits (booms, a title slam, an impact with a
suck-in, soft thumps and knocks), builds and falls (risers, a reversed swell, a drum roll, a sub drop, a suck-in), a
braam, an explosion, small arrivals (pops, tocks, a blip), light (a glint, a shimmer sweep, a sparkle), a glitch, key
ticks and two stingers.

The film sounds (air, hits, builds, falls, the braam, the explosion, the reveal) are built from real recordings in
the public domain, as a sound designer builds them: learned things of earshot.sounds.learned (orchestral bass drums,
timpani, tam-tams, crash cymbals, heavy drops, door slams, swells of air and metal, drum rolls, low brass, whooshes)
reshaped with earshot.synth.design (reverse, layer, move, fade) and glued by one room ("real material" below). The
owner heard the invented booms, slams and risers as unrealistic (1-2 of 5) and could not tell a recording from its
take. The rest is invented, shaped after class statistics of licensed film effects (the film study in
docs/motion-sfx-study.md and docs/launch-video-sound.md, and a library of effects filed by film action that the lab
measured). Medians over a class, never one recording:
    air     a film whoosh is a rounded swell, not a pointed one: it stays within 20 dB of its peak for about two
            thirds of its length (0.3-0.6 s for a small move, 1.3 s for a scene pass), peaks at 40-60 % of it, and
            its energy sits low (centroid 0.2-1.8 kHz; a scene pass under 150 Hz) with the air far under the body;
            the spectrum sinks across the pass; near mono
    hits    sub-dominated (three quarters and more under 150 Hz), a 20-30 ms rise instead of a click, 20 dB down
            in 0.2-0.7 s; a landing in a product film is 0.25 s long and dull (centroid about 500 Hz)
    builds  grow almost linearly in amplitude over 90 % of their length, then fall 20 dB within about 100 ms
    small   0.1-0.25 s long, a 10-20 ms rise, 20 dB down in 25-60 ms, mono, little between 2 and 5 kHz
The invented ones are built so: a sliding panel's air is stereo band noise whose bands follow one moving spectral
hump and gust together (texture.band_noise, texture.envelopes); struck parts are modes behind a contact pulse
(modal.ring), sparkle and grit are event clouds (texture.arrivals, rings, bursts), tonal parts are unison voices and
struck instruments (voice, voices), and everything sits in one shared room.

    from earshot.sounds import registry, motion      # motion registers its designs on import
    y, meta = registry.render('whoosh/fast', seed=2, energy=0.4)     # meta['member']: the main recording played
    y, meta = registry.render('rise/reverse', dur=3.0)               # a build keeps `dur`; it ends on its peak

Axes: energy = a faster, harder, brighter gesture (and more of the lower layers); size = bigger (lower and longer:
the takes 0.8 .. 1.25 times the design's own size); bright = more top (a tilt); hard = a harder strike; ring = a
longer ring; space, width; motion = more travel across the stereo field (the designs that list it). Texture, voice
and voices are imported where they are used.
"""
from types import SimpleNamespace

import numpy as np
from scipy import fft as sfft

from ..forge import dsp, fx
from ..synth import design as ops
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


# ------------------------------------------------------------------------------------------------ real material
# The film designs below are built the way a sound designer builds them: from recordings (the learned things of
# earshot.sounds.learned: orchestral bass drums, timpani, gongs, cymbals, heavy drops, door slams, swells of air and
# metal, drum rolls, low brass, real whooshes and passes), reshaped by earshot.synth.design (reverse, layer, move,
# fade) within the range where a take stays natural, and glued by one room. No oscillator and no synthetic sweep.
# The seed is a performance: it picks the members (another drum, another cymbal) and every take draws fresh noise.
REAL = ('energy', 'size', 'bright', 'hard', 'ring', 'space', 'width')
REAL_AIR = ('energy', 'size', 'bright', 'hard', 'space', 'width')
REAL_MOVE = REAL_AIR + ('motion',)
# the pitch detector reads the noise of a real whoosh or swell as gliding up (a swing speeds up, a swell opens):
# that is the recording, not a synthesized whip
GLIDE = ('rising_glide',)


def _learned():
    from . import learned            # lazy: the store's index is read on the first film take
    return learned


def _real(c, r, sid, k=0, main=False, **tr):
    """a take (stereo) of the learned thing sid (earshot.sounds.learned) through design.take: the member the seed
    picks, or the one k further along its fixed order (a second layer of the same thing is another recording). main:
    the take that says which recording was played (meta['member'])"""
    L = _learned()
    e = L.index()[sid]
    i = L.pick(sid, c.seed + k)
    m = e['members'][i]
    y = ops.take(L.model(m['file']), r, **tr)
    if m.get('polarity', 1) < 0:
        y[:, 1] = -y[:, 1]
    if main:
        c.member = {'n': i + 1, 'of': len(e['members']), 'recording': m['credit']['title'],
                    'author': m['credit']['author']}
    return y


def _natural(sid, c, k=0):
    """the natural length in s of the member the seed picks"""
    L = _learned()
    e = L.index()[sid]
    return float(e['members'][L.pick(sid, c.seed + k)]['dur_s'])


def _tr(c, size=1.0, ring=None, bright=0.0, hard=0.0):
    """the shared axes as the transforms of a take (design.take), around a design's own size, ring, tilt (dB per
    octave) and hardness: size 0.8 .. 1.25 times, ring 0.6 .. 1.7 times (when the design rings), bright 2 dB per
    octave and energy 0.6 more, hard and 0.4 of energy on the strike; a little take-to-take variation (vary)"""
    e = c.energy
    out = dict(size=size * c.lerp('size', 0.8, 1.25, 1.0, log=True), bright=2.0 * c.bright + 0.6 * e + bright,
               hard=float(np.clip(c.hard + 0.4 * e + hard, -1.0, 1.0)), vary=0.5 * c.vary)
    if ring is not None:
        out['ring'] = ring * c.lerp('ring', 0.6, 1.7, 1.0, log=True)
    return out


def _fit(c, natural, share=1.0, size=1.0):
    """the length factor of a take whose natural length is `natural` s: `share` of the length asked for (dur), else
    of the design's own (so the takes of one design, each another recording, come out about as long), within
    design.LENGTH (size already stretches it)"""
    return float(np.clip(share * c.seconds() / (natural * size), *ops.LENGTH))


def _place(c, r, y, room='room', crest=None):
    """a mix to the finished take: its stereo width (width: mid / side, 0 .. 2 times) in the shared room; crest: the
    takes of a short design matched in loudness as a designer matches a set of variations (_even: the peaks of a
    take that stand more than `crest` dB over its loudness rounded off), since each seed is another recording"""
    y = space.room(fx.width(y, c.lerp('width', 0.0, 2.0, 1.0)), c.room(room), amount=c.space, r=r)
    y = y * 10 ** (4.0 * c.energy / 20)                   # energy is also level (+-4 dB), where a take is not levelled
    return y if crest is None else _even(y, crest)


def _whoosh(c, r, sid, share=1.0, size=1.0, bright=0.0, hard=0.0, main=True, k=0, low=None):
    """a take of a real whoosh (a learned thing) at a design's size, tilt and length (`share` of the design's); low:
    the rumble under `low` Hz taken out (a small move has no weight: the wind on a field microphone is not the move)"""
    tr = _tr(c, size=size, bright=bright, hard=hard)
    y = _real(c, r, sid, k=k, main=main, length=_fit(c, _natural(sid, c, k), share, tr['size']), **tr)
    return dsp.butter(y, 'high', low, order=2) if low else y


@design('whoosh/fast', kind='whoosh',
        says=('a fast crisp whoosh of a stick cutting the air', 'a quick sharp swish rushing past',
              'a short fast swoosh'),
        uses=('whoosh', 'whip pan', 'fast transition', 'cut', 'fly in', 'title'), tags=('real',), domains=DOM,
        dur=0.45, axes=REAL_AIR, prior=1.03, allows=GLIDE)
def whoosh_fast(c, r):
    """a fast move: a real stick swung through the air (whoosh/stick: the owner rated one of these 5) and, lined up
    at its loudest moment 7 dB under it (4 more with energy), a real swish (swish/fast) for the density of a
    designed whoosh"""
    a = _whoosh(c, r, 'whoosh/stick', 1.0)
    b = _whoosh(c, r, 'swish/fast', 1.0, main=False)
    y, _ = ops.layer([(a, 0.0, 0.0, ops.peak(a, 20.0)), (b, -7.0 + 4.0 * c.energy, 0.0, ops.peak(b, 20.0))])
    return _place(c, r, y, crest=12.0)


@design('whoosh/soft', kind='whoosh',
        says=('a soft short whoosh', 'a gentle swish of air passing by', 'a quiet breath of air moving past'),
        uses=('whoosh', 'transition', 'element move', 'swipe', 'slide in'), tags=('real',), domains=DOM, dur=0.6,
        axes=REAL_AIR, prior=1.05, allows=GLIDE)
def whoosh_soft(c, r):
    """a small element crossing the frame: a real airy whoosh (whoosh/airy) taken a little smaller, darker by
    1.5 dB per octave, with a softer onset and nothing under 120 Hz, trimmed around its loudest moment to the
    length (a long designed whoosh among the members is cut to a small move, faded at both ends)"""
    y = _whoosh(c, r, 'whoosh/airy', 1.0, size=0.9, bright=-1.5, hard=-0.4, low=120.0)
    T, p = c.seconds(), ops.peak(y)
    y = ops.span(y, max(0.0, p - 0.45 * T), p + 0.55 * T, fin=0.2 * T, fout=0.3 * T)
    return _place(c, r, y, crest=12.0)


@design('whoosh/deep', kind='whoosh',
        says=('a deep low whoosh, a large heavy swell of air', 'a big dark rush of air passing',
              'a slow low rush of air with weight under it'),
        uses=('whoosh', 'big transition', 'scene change', 'pass by', 'swell'), tags=('real',), domains=FILM,
        dur=1.4, axes=REAL_AIR, allows=GLIDE)
def whoosh_deep(c, r):
    """a scene transition: a real deep whoosh (whoosh/low, or the heavy one the owner rated 3: whoosh/heavy, by the
    seed) taken a little bigger, with the weight a designer puts under it: a real bass drum's decay played
    backwards 12 dB under it (more with energy), swelling into the whoosh's loudest moment"""
    sid = ('whoosh/low', 'whoosh/low', 'whoosh/heavy')[c.seed % 3]
    a = _whoosh(c, r, sid, 1.0, size=1.1)
    drum = _real(c, r, 'bass_drum/orchestral', **_tr(c, size=1.3, bright=-2.0))
    drum = ops.reverse(ops.span(drum, 0.0, 0.8, fout=0.2))
    y, _ = ops.layer([(a, 0.0, 0.0, ops.peak(a)), (ops.fade(drum, 0.4, 0.0, 'db'), -12.0 + 4.0 * c.energy, 0.0,
                                                    ops.end(drum))])
    return _place(c, r, y)


@design('whoosh/long', kind='whoosh',
        says=('a long deep whoosh that builds and passes', 'a slow heavy rush of air sweeping through',
              'a big scene-change whoosh'),
        uses=('whoosh', 'scene change', 'big transition', 'long whoosh', 'pass by'), tags=('real',), domains=FILM,
        dur=2.2, axes=REAL_AIR, allows=GLIDE)
def whoosh_long(c, r):
    """a scene pass: a real deep whoosh (whoosh/low) taken bigger and slower, led in by a swell of real air played
    backwards (swell/air, reversed) that ends at the whoosh's loudest moment, 8 dB under it"""
    a = _whoosh(c, r, 'whoosh/low', 0.7, size=1.15)
    lead = ops.reverse(ops.span(_real(c, r, 'swell/air', length=1.2, **_tr(c, bright=-1.0)), 0.0, 1.6))
    y, _ = ops.layer([(a, 0.0, 0.0, ops.peak(a)), (lead, -8.0 + 3.0 * c.energy, 0.0, ops.end(lead))])
    return _place(c, r, y, 'studio')


@design('swoosh/in', kind='whoosh',
        says=('a whoosh that rises and stops, something arriving', 'a quick rush of air swelling in',
              'a short swoosh building to an abrupt end'),
        uses=('whoosh', 'fly in', 'arrive', 'element enters', 'appear'), tags=('real',), domains=DOM, dur=0.5,
        axes=REAL_AIR, allows=GLIDE)
def swoosh_in(c, r):
    """an element arriving: a real whoosh (whoosh/airy) up to its loudest moment and stopped there within 30 ms"""
    a = _whoosh(c, r, 'whoosh/airy', 1.6, size=0.95, low=90.0)
    return _place(c, r, ops.span(a, 0.0, ops.peak(a, 20.0) + 0.03, fout=0.03), crest=12.0)


@design('swoosh/out', kind='whoosh',
        says=('a whoosh that starts full and fades away, something leaving', 'air rushing away into the distance',
              'a short swoosh trailing off'),
        uses=('whoosh', 'fly out', 'exit', 'element leaves', 'dismiss'), tags=('real',), domains=DOM, dur=0.5,
        axes=REAL_AIR, allows=GLIDE)
def swoosh_out(c, r):
    """an element leaving: a real whoosh (whoosh/airy) from just before its loudest moment on, so it is full at once
    and falls away as it was recorded"""
    a = _whoosh(c, r, 'whoosh/airy', 1.4, size=0.95, low=90.0)
    return _place(c, r, ops.span(a, max(ops.peak(a, 20.0) - 0.02, 0.0), fin=0.01), crest=12.0)


@design('flick/thin', kind='whoosh',
        says=('a tiny thin swish', 'a very short light flick of air', 'a quick faint whip'),
        uses=('swish', 'flick', 'text move', 'small move', 'cursor'), tags=('real',), domains=DOM, dur=0.3,
        axes=REAL_AIR, prior=1.02, allows=GLIDE)
def flick_thin(c, r):
    """the smallest move: a real swish (swish/fast) taken smaller (size 0.75), brighter by 1 dB per octave and
    without its rumble under 150 Hz"""
    return _place(c, r, _whoosh(c, r, 'swish/fast', 1.0, size=0.75, bright=1.0, low=150.0), 'booth', crest=12.0)


@design('flick/paper', kind='whoosh',
        says=('a quick flick of paper', 'a short dry papery swish', 'a light page flutter'),
        uses=('swish', 'flick', 'text slide', 'page turn', 'card flip'), tags=('real',), domains=DOM, dur=0.8,
        axes=REAL_AIR, allows=GLIDE)
def flick_paper(c, r):
    """a sheet flicked: a real page turned (page/turn), fitted to the length"""
    return _place(c, r, _whoosh(c, r, 'page/turn', 1.0), 'booth')


@design('pass/air', kind='whoosh',
        says=('something flying past fast, a whoosh that sweeps from one side to the other',
              'a fly-by whoosh dropping a little in pitch as it passes', 'a rush of air passing close by'),
        uses=('fly by', 'pass by', 'whoosh', 'transition', 'flyby'), tags=('real',), domains=FILM, dur=1.4,
        axes=REAL_MOVE, allows=GLIDE)
def pass_air(c, r):
    """a fly-by: a real deep whoosh (whoosh/low) carried past the listener (design.move): from one side to the
    other through the centre at its loudest moment, its level following the distance and its pitch falling 1.5
    semitones as it goes by (motion: 0.5 .. 2.5 semitones and how far it travels across)"""
    a = _whoosh(c, r, 'whoosh/low', 1.0, size=1.05)
    y = ops.move(a, r, travel=c.lerp('motion', 0.4, 1.0, 0.8), fall=c.lerp('motion', 0.5, 2.5, 1.5))
    return _place(c, r, y)


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
def _drum(c, r, size=1.25, ring=1.2, k=0, main=False, bright=0.0):
    """a real orchestral bass drum (bass_drum/orchestral) at a design's size and ring"""
    return _real(c, r, 'bass_drum/orchestral', k=k, main=main, **_tr(c, size=size, ring=ring, bright=bright))


@design('boom/deep', kind='boom',
        says=('a deep cinematic boom with a long low tail', 'a huge orchestral drum hit with a dark gong under it',
              'a deep trailer impact'),
        uses=('impact', 'cinematic hit', 'trailer hit', 'boom', 'reveal', 'title'), tags=('real',), domains=FILM,
        dur=3.6, axes=REAL)
def boom_deep(c, r):
    """a trailer boom: a real orchestral bass drum taken bigger (size 1.25, ring 1.2) for the body, a real timpani
    7 dB under it for the attack and the pitch of the skin, and a real tam-tam 15 dB under for the dark wash that
    outlasts both (more of them with energy), lined up at their strikes, in a hall"""
    e = c.energy
    drum = _drum(c, r, main=True)
    timp = _real(c, r, 'timpani/hit', **_tr(c, size=1.15, ring=1.0))
    gong = _real(c, r, 'gong/hit', **_tr(c, size=1.1, ring=0.8, bright=-1.0))
    y, _ = ops.layer([(drum, 0.0, 0.0), (timp, -7.0 + 3.0 * e, 0.0), (ops.span(gong, 0.0, 4.0, fout=1.5),
                                                                     -15.0 + 3.0 * e, 0.01)])
    return _place(c, r, ops.span(y, 0.0, 4.0, fout=1.2, shape='db'), 'hall')


@design('boom/short', kind='boom',
        says=('a short deep drum hit', 'a tight low punchy boom', 'a quick timpani thud'),
        uses=('impact', 'hit on a cut', 'cut', 'accent', 'boom'), tags=('real',), domains=FILM, dur=0.7,
        axes=REAL)
def boom_short(c, r):
    """a hit on a cut: a real timpani (timpani/hit) with a real bass drum 4 dB under it, both damped (ring 0.6) and
    faded out from 0.25 s as a decay fades (straight in dB), nearly dry"""
    timp = _real(c, r, 'timpani/hit', main=True, **_tr(c, size=1.05, ring=0.6))
    drum = _drum(c, r, size=1.1, ring=0.6)
    y, _ = ops.layer([(timp, 0.0, 0.0), (drum, -4.0 + 3.0 * c.energy, 0.0)])
    return _place(c, r, ops.span(y, 0.0, 0.25 + 0.45 * c.lerp('ring', 0.5, 2.0, 1.0, log=True), fout=0.4,
                                 shape='db'), 'booth')


@design('boom/wide', kind='boom',
        says=('a wide low impact, a struck tam-tam over a bass drum', 'a big metallic boom spreading out',
              'a heavy boom with a shimmering wash'),
        uses=('impact', 'big hit', 'trailer hit', 'boom', 'reveal'), tags=('real',), domains=FILM, dur=4.5,
        axes=REAL)
def boom_wide(c, r):
    """something large coming down: a real tam-tam struck hard (gong/hit) for the wash, a real bass drum under it
    for the weight, both wider than recorded (width 1.6 at width 0)"""
    gong = _real(c, r, 'gong/hit', main=True, **_tr(c, size=1.0, ring=1.0))
    drum = _drum(c, r, size=1.2, ring=1.0)
    y, _ = ops.layer([(gong, -2.0, 0.0), (drum, 0.0 + 2.0 * c.energy, 0.0)])
    y = ops.span(y, 0.0, 5.0, fout=1.5, shape='db')
    return space.room(fx.width(y, c.lerp('width', 0.6, 2.0, 1.6)), c.room('hall'), amount=c.space, r=r)


@design('slam/title', kind='hit',
        says=('a hard heavy slam with a low drum under it', 'a door slammed with a boom', 'a punchy title slam'),
        uses=('impact', 'title slam', 'slam', 'text slam', 'stamp'), tags=('real',), domains=DOM, dur=0.9,
        axes=REAL)
def slam_title(c, r):
    """a title landing hard: a real heavy door slammed (door/slam) over a real bass drum 5 dB under it whose ring
    is cut short (0.5 s, faded as a decay), in a short room"""
    slam = _real(c, r, 'door/slam', main=True, **_tr(c, size=1.1, ring=1.0))
    drum = ops.span(_drum(c, r, size=1.2, ring=0.7), 0.0, 0.5, fout=0.35, shape='db')
    y, _ = ops.layer([(slam, 0.0, 0.0), (drum, -5.0, 0.0)])
    return _place(c, r, y, 'room', crest=11.0)


@design('impact/suck', kind='hit',
        says=('air sucked in backwards, then a heavy hit', 'a reverse whoosh pulling into a big impact',
              'an inhale into a crash'),
        uses=('impact', 'suck in', 'pre-hit', 'reverse whoosh', 'hit with a suck-in'), tags=('real',), domains=FILM,
        dur=1.8, prior=0.9,
        axes=REAL, allows=('pre_swell', 'stacked'))       # the suck-in before the layered hit is the design
def impact_suck(c, r):
    """the breath before a hit: the build of a real airy whoosh (whoosh/airy through design.rise: its tail
    reversed, about half a second, longer with dur) ends where a real heavy drop (heavy_drop/floor) and a real bass
    drum 3 dB under it strike"""
    pre = ops.rise(_real(c, r, 'whoosh/airy', length=1.2, **_tr(c, size=1.1)))
    pre = ops.span(pre, max(0.0, ops.end(pre) - (c.length() or 1.8) * 0.3))
    hit = _real(c, r, 'heavy_drop/floor', main=True, **_tr(c, size=1.1, ring=1.0))
    drum = _drum(c, r, size=1.2, ring=0.8)
    y, _ = ops.layer([(pre, -4.0, 0.0, ops.end(pre)), (hit, 0.0, 0.0), (ops.span(drum, 0.0, 1.2, fout=0.6,
                                                                                shape='db'), -3.0, 0.0)])
    return _place(c, r, y, 'room', crest=11.0)


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
def _riser(y, at):
    """a build cut at `at` s (its peak): everything before it, then 60 ms of what follows faded out as a decay fades,
    so it stops on its loudest moment without a step"""
    return ops.span(y, 0.0, at + 0.06, fout=0.06, shape='db')


@design('rise/reverse', kind='rise',
        says=('a reversed cymbal swelling up to an abrupt stop', 'a backwards shimmering swell',
              'a reverse crash building'),
        uses=('reverse', 'reverse swell', 'riser', 'swell into a cut', 'reverse cymbal'), tags=('real',),
        domains=FILM, dur=2.0, axes=REAL_AIR, prior=1.05, allows=GLIDE)
def rise_reverse(c, r):
    """the classic riser: a real crash cymbal's decay played backwards (cymbal/orchestral, reversed), with a real
    tam-tam's decay played backwards 9 dB under it (more with energy) for a dark floor, both ending together on what
    was their strike; dur sets how much of the decays is heard (2 s by default)"""
    T = c.length() or 2.0
    tr = _tr(c, ring=None)
    cym = ops.reverse(ops.span(_real(c, r, 'cymbal/orchestral', main=True, **tr), 0.0, T + 0.01, fout=0.2))
    gong = ops.reverse(ops.span(_real(c, r, 'gong/hit', **_tr(c, size=1.1, bright=-1.0)), 0.0, T + 0.01,
                                fout=0.2))
    y, _ = ops.layer([(cym, 0.0, 0.0, ops.end(cym)), (gong, -9.0 + 4.0 * c.energy, 0.0, ops.end(gong))])
    return _place(c, r, ops.fade(y, 0.3 * T, 0.0, 'db'), crest=11.0)


@design('rise/air', kind='rise',
        says=('a rising rush of air that swells to a peak', 'a hiss of air and steam building up',
              'air swelling to a cut'),
        uses=('riser', 'build', 'noise riser', 'reveal', 'transition', 'uplifter'), tags=('real',), domains=FILM,
        dur=1.8, axes=REAL_AIR, prior=1.1, allows=GLIDE)
def rise_air(c, r):
    """a noise riser made of real air: the build a recorded swell of air, wind or steam holds (swell/air through
    design.rise: up to its loudest moment, or its decay reversed), with the build of a real airy whoosh (whoosh/airy)
    6 dB under it for the last part, ending on the same moment and sharpening the top"""
    sw = ops.rise(_real(c, r, 'swell/air', main=True, length=_fit(c, _natural('swell/air', c), 1.8),
                        **_tr(c, size=1.0)))
    wh = ops.rise(_real(c, r, 'whoosh/airy', **_tr(c, size=1.05)))
    wh = ops.span(wh, max(0.0, ops.end(wh) - 0.6 * ops.end(sw)), fin=0.1)    # it joins in for the last part
    y, _ = ops.layer([(sw, 0.0, 0.0, ops.end(sw)), (wh, -6.0 + 3.0 * c.energy, 0.0, ops.end(wh))])
    return _place(c, r, ops.fade(y, 0.15 * len(y) / SR, 0.0, 'db'), crest=11.0)


@design('rise/tension', kind='rise',
        says=('a drum roll growing to a peak with a cymbal swelling over it', 'a timpani roll building tension',
              'a rumbling orchestral build-up'),
        uses=('riser', 'tension', 'build', 'suspense', 'anticipation', 'drum roll'), tags=('real',), domains=FILM,
        dur=4.0, axes=REAL, prior=1.05, allows=GLIDE + ('clicks',))     # the strokes of a roll
def rise_tension(c, r):
    """a build that grows as players grow it: a real drum roll in crescendo (roll/drum or roll/timpani, by the seed)
    up to its loudest moment and stopped there, with a real suspended cymbal rolled in crescendo (swell/cymbal) 6 dB
    under it culminating on the same moment"""
    sid = ('roll/drum', 'roll/timpani')[c.seed % 2]
    roll = _real(c, r, sid, main=True, length=_fit(c, _natural(sid, c), 1.1), **_tr(c, size=1.05, ring=1.0))
    roll = _riser(roll, ops.peak(roll, 200.0))
    roll = ops.fade(roll, 0.7 * len(roll) / SR, 0.0, 'power')         # the fader ridden up, as on a steady roll
    cym = _real(c, r, 'swell/cymbal', length=_fit(c, _natural('swell/cymbal', c), 1.1), **_tr(c, ring=1.0))
    cym = _riser(cym, ops.peak(cym, 200.0))
    y, _ = ops.layer([(roll, 0.0, 0.0, ops.end(roll) - 0.06), (cym, -6.0 + 3.0 * c.energy, 0.0, ops.end(cym) - 0.06)])
    return _place(c, r, y, 'hall')


@design('suck/in', kind='rise',
        says=('a short inward suck of air', 'a quick reverse whoosh pulling in and stopping', 'air being sucked in'),
        uses=('suck', 'pre-hit', 'inhale', 'lead in', 'reverse whoosh'), tags=('real',), domains=FILM, dur=0.6,
        axes=REAL_AIR, prior=0.9, allows=('ring',) + GLIDE)     # a reversed resonant tail can read as a comb
def suck_in(c, r):
    """the breath before a hit: a real airy whoosh (whoosh/airy) through design.rise (its tail played backwards
    becomes the in-breath) that stops where its loudest moment was"""
    y = ops.rise(_whoosh(c, r, 'whoosh/airy', 1.4, size=1.05))
    T = c.length() or 0.6
    return _place(c, r, ops.span(y, max(0.0, ops.end(y) - T), fin=0.05), 'booth')


@design('fall/air', kind='fall',
        says=('a rush of air falling away and fading out', 'a hiss of air dying away', 'air trailing off'),
        uses=('downlifter', 'fall', 'exit', 'after a hit', 'wind down'), tags=('real',), domains=FILM, dur=1.4,
        axes=REAL_AIR, prior=1.1, allows=GLIDE)
def fall_air(c, r):
    """what follows a hit: the decay a recorded swell of air, wind or steam holds (swell/air through design.fall:
    from its loudest moment on, or its build reversed), ending in a fade straight in dB"""
    y = ops.fall(_real(c, r, 'swell/air', main=True, length=_fit(c, _natural('swell/air', c), 1.4),
                       **_tr(c, size=1.05, bright=-0.5)))
    return _place(c, r, ops.fade(y, 0.0, 0.4 * len(y) / SR, 'db'), 'studio', crest=11.0)


@design('fall/sub', kind='boom',
        says=('a deep drum boom sinking into a long low tail', 'a very low boom dying away',
              'a low boom blooming and sinking'),
        uses=('sub drop', 'bass drop', 'drop', 'low drop after a cut', 'impact'), tags=('real',), domains=FILM,
        dur=3.6, axes=REAL, allows=('rising_glide',))    # a big drum's skin can settle upward a little
def fall_sub(c, r):
    """a sub drop under a cut: a real orchestral bass drum as big and dark as the engine takes it (size 1.4, 2 dB
    per octave darker, ring 1.4), its long decay as it is, with a real timpani 10 dB under it for the moment of the
    cut"""
    drum = _drum(c, r, size=1.4, ring=1.4, main=True, bright=-2.0)
    timp = _real(c, r, 'timpani/hit', **_tr(c, size=1.3, ring=0.8, bright=-1.0))
    y, _ = ops.layer([(drum, 0.0, 0.0), (timp, -10.0 + 3.0 * c.energy, 0.0)])
    return _place(c, r, ops.span(y, 0.0, c.length() or 4.0, fout=1.2, shape='db'), 'booth')


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
        says=('a reversed cymbal swelling into a deep drum hit with a gong ringing out',
              'a swell that lands on a big orchestral boom', 'a cinematic reveal hit'),
        uses=('logo', 'reveal', 'sting', 'logo reveal', 'title reveal'), tags=('real',), domains=DOM, dur=3.5,
        axes=REAL, prior=1.05, allows=('pre_swell',))       # the swell into the hit is the design
def sting_reveal(c, r):
    """a reveal in two parts that meet on one moment: a real crash cymbal's decay played backwards (cymbal/orchestral,
    reversed, a third of the length) swells into a real bass drum (bass_drum/orchestral) and a real tam-tam 8 dB
    under it (gong/hit), which ring out in a hall"""
    T = c.length() or 3.5
    lead = 0.3 * T
    cym = ops.reverse(ops.span(_real(c, r, 'cymbal/orchestral', **_tr(c)), 0.0, lead + 0.01, fout=0.15))
    drum = _drum(c, r, size=1.2, ring=1.1, main=True)
    gong = _real(c, r, 'gong/hit', **_tr(c, size=1.05, ring=1.0))
    y, _ = ops.layer([(ops.fade(cym, 0.4 * lead, 0.0, 'db'), -4.0, 0.0, ops.end(cym)), (drum, 0.0, 0.0),
                      (gong, -8.0 + 3.0 * c.energy, 0.005)])
    return _place(c, r, ops.span(y, 0.0, T, fout=0.4 * T, shape='db'), 'hall')


# the chord of a braam: the root and the octave in the low brass, the fifth between them (semitones over the root)
BRAAM = (0, 7, 12)


@design('braam/brass', kind='boom',
        says=('a huge low brass blast in octaves and a fifth over a bass drum hit', 'a cinematic braam of real tubas '
              'and trombones', 'a blaring orchestral brass stab that swells'),
        uses=('braam', 'trailer hit', 'impact', 'reveal', 'title', 'brass hit'), tags=('real', 'brass'),
        domains=FILM, dur=3.0, pitch='A#1', axes=REAL, prior=1.2, allows=('pre_swell', 'stacked'))
def braam_brass(c, r):
    """a braam made of real brass: held low brass notes (brass/low: tubas, trombones, horns) moved to the root, its
    fifth and its octave (each the member nearest its note, moved at most 7 semitones), entering together over a
    real bass drum hit and swelling for 0.12 s (0.03 .. 0.4 s with hard), led in by a real cymbal decay played
    backwards 12 dB under them; they are held for the length asked for and fall away as the hall closes"""
    L = _learned()
    e = L.index()['brass/low']
    root = c.hz()
    T = c.length() or 3.0
    hard = c.lerp('hard', 0.4, 0.03, 0.12, log=True)
    parts, first = [], None
    for k, st in enumerate(BRAAM):
        f = root * 2 ** (st / 12)
        order = sorted(range(len(e['members'])), key=lambda i: abs(np.log2(e['members'][i]['hz'] / f)))
        i = order[(c.seed + k) % min(2, len(order))]
        m = e['members'][i]
        tr = _tr(c, bright=1.0)
        tr.pop('size')
        z = ops.take(L.model(m['file']), r, pitch=12 * np.log2(f / m['hz']),
                     length=float(np.clip(T / m['dur_s'], *ops.LENGTH)), **tr)
        first = first or m
        z = ops.fade(ops.span(z, max(0.0, ops.onset(z, 20.0) - 0.01), fout=0.5), hard, 0.0)
        parts.append((z, (0.0, -3.0, -4.0)[k] + c.lerp('size', -3.0, 3.0, 0.0) * (k == 0), 0.0, 0.0))
    c.member = {'n': L.index()['brass/low']['members'].index(first) + 1, 'of': len(e['members']),
                'recording': first['credit']['title'], 'author': first['credit']['author']}
    drum = ops.span(_drum(c, r, size=1.25, ring=0.9), 0.0, 1.5, fout=0.8, shape='db')
    cym = ops.reverse(ops.span(_real(c, r, 'cymbal/orchestral', **_tr(c)), 0.0, 1.0, fout=0.15))
    y, _ = ops.layer(parts + [(drum, -2.0 + 3.0 * c.energy, 0.0, ops.onset(drum)),
                              (ops.fade(cym, 0.5, 0.0, 'db'), -12.0, 0.0, ops.end(cym))])
    return _place(c, r, ops.span(y, 0.0, T + 1.0, fout=0.6, shape='db'), 'hall')


@design('explosion/huge', kind='boom',
        says=('a huge close explosion with a deep boom and a long rolling tail', 'a big cinematic blast',
              'an explosion that shakes the ground'),
        uses=('explosion', 'blast', 'impact', 'big explosion'), tags=('real',), domains=FILM, dur=4.0, axes=REAL,
        prior=1.05)
def explosion_huge(c, r):
    """a film explosion: a real near explosion (explosion/near) for the blast, a real bass drum 3 dB under it for the
    weight the recording lacks, and a real roll of thunder 9 dB under them (thunder/roll) for the tail that rolls
    away, in a hall"""
    blast = _real(c, r, 'explosion/near', main=True, **_tr(c, size=1.15, ring=1.2))
    drum = _drum(c, r, size=1.35, ring=1.2, bright=-1.0)
    roll = _real(c, r, 'thunder/roll', **_tr(c, size=1.05, ring=1.0, bright=-1.0))
    roll = ops.span(roll, max(0.0, ops.peak(roll, 200.0) - 0.4), fin=0.4)
    y, _ = ops.layer([(blast, 0.0, 0.0), (drum, -3.0 + 3.0 * c.energy, 0.0), (roll, -9.0, 0.08, 0.0)])
    return _place(c, r, ops.span(y, 0.0, c.length() or 4.5, fout=1.5, shape='db'), 'hall', crest=11.0)



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
