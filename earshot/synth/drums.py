"""Drums from their physics, for what no recording plays: struck membranes, snare wires, hands, shaken beads, and
the classic synthetic drum voices.

The catalogue's acoustic kit and hand percussion are played from real public-domain recordings (earshot.sounds.drums
and earshot/data/drums/); this engine makes the rest: a brush on a snare, a finger snap, sticks clicked together, the
shaker and the drum-machine voices of the music beds' own kit (earshot.synth.music), and the drum-machine designs.
A drum written in code usually sounds like a sine with a noise burst on top. A real one differs in these ways, and
each is built here and can be measured on the output:

  membrane   a circular head rings in inharmonic modes, (0,1) 1 : (1,1) 1.59 : (2,1) 2.14 : (0,2) 2.30 : (3,1) 2.65
             ... (Bessel zeros), moved by the air on the head and the air shut in the shell (membrane_modes); each
             mode has its own decay, the (0,n) ones radiate best and die fastest, a pillow or a ring on the head takes
             the upper modes first (membrane_rates). A hard hit stretches the head, so every mode starts sharp and
             falls back within tens of ms, more at a higher velocity (tension modulation: the tension follows the
             energy of the head, which decays twice as fast as its amplitude). Where it is struck chooses the modes
             (the centre drives only the (0,n) thud; off the centre the ring). The beater is a Hertz contact whose
             length follows its hardness and speed (contact): a harder or faster hit is brighter, not only louder.
  wires      the snares under a snare drum are driven by the bottom head: they slap against it whenever it swings
             into them, buzz while it moves and bounce a little after it (rattle), a bright noise that follows the
             drum's motion with a 150-250 Hz body, never white noise under an envelope.
  events     a shaker is many small collisions whose rate and size follow the energy of the gesture (Cook's PhISM
             model: shaker); a clap is the slap of two palms exciting the cavity
             between them (Repp 1987, Peltola et al. 2007: 0.6-2.5 kHz by hand shape), a finger snap the same
             cavity struck harder and higher.
  synthetic  the classic drum-machine voices as productions use them: a kick of a sine whose pitch falls fast with a
             click and tape saturation (kick_synth), a snare of two tuned tones and filtered noise (snare_synth), a
             clap of three or four noise bursts and a tail (clap_synth), hats of six inharmonic square waves through
             band-passes (hat_synth).

Every acoustic voice is stereo from a close pair: each mode has its own place in the image (modal.mic_pair) and the
pair hears the very early reflections of the stand and floor (modal.early); no room of its own (a design adds the
shared room, earshot.synth.space).

    from earshot.forge import dsp
    from earshot.synth import drums
    r, built = dsp.rng(1), dsp.rng(7)                       # the take, the instrument
    tom = drums.membrane(110.0, dsp.n_of(1.2), r, built, vel=0.8, strike=0.35, t60=0.9, tune=1, radius=0.2)
    y = tom['y']                                            # (n, 2); tom['motion'] is the head's displacement
    kick = drums.kick_synth(dsp.n_of(1.0), r, f_end=48.0, vel=0.9)
"""
import functools
import math

import numpy as np
from scipy import signal, special

from ..forge import dsp, fx
from ..forge.dsp import njit
from . import modal

SR = dsp.SR
T60 = modal.T60

# ------------------------------------------------------------------------------------------------------ contact
# Beaters: contact time (s) at 3 m/s, the Hertz exponent of the head (1.5 elastic; felt and skin stiffen faster),
# the click level and the grit of modal.contact_noise. The contact time goes as speed^(-(p - 1) / (p + 1))
# (modal.contact_time): a faster hit is a shorter, brighter contact. Values are class values: a felt kick beater
# stays on the head for about 2-4 ms, a hickory stick on a coated head 0.3-0.6 ms, a palm 3-6 ms, fingertips about
# 1.5 ms, a yarn mallet on a cymbal 2-4 ms, wood on wood or a stick on a metal hoop under 0.15 ms.
BEATERS = {
    'felt': (0.0026, 2.5, 0.04, 0.03),
    'plastic': (0.0009, 1.5, 0.30, 0.08),
    'wood': (0.0007, 1.5, 0.35, 0.10),
    'stick': (0.00045, 1.5, 0.30, 0.10),
    'tip': (0.00016, 1.5, 0.30, 0.10),          # a stick's tip on a cymbal
    'shoulder': (0.00028, 1.5, 0.30, 0.10),     # the stick's shoulder on a cymbal's edge
    'rim': (0.00009, 1.5, 0.45, 0.15),          # a stick across a metal hoop
    'clave': (0.00012, 1.5, 0.35, 0.12),        # rosewood on rosewood
    'brush': (0.0016, 2.5, 0.02, 0.15),
    'hand': (0.0045, 2.5, 0.02, 0.05),
    'fingers': (0.0016, 2.0, 0.08, 0.08),
    'mallet': (0.0025, 3.0, 0.00, 0.01),
}


def speed(vel):
    """the beater's speed in m/s for a velocity 0..1 (a whisper about 0.6 m/s, a full stroke about 7 m/s)"""
    return 0.6 + 6.4 * float(np.clip(vel, 0.0, 1.0))


def contact(beater, vel, hard=0.0):
    """(tau s, exponent, click, grit) of a beater at velocity vel (0..1); hard -1..1 halves or doubles the
    stiffness of its face: tau goes as stiffness^(-1 / (p + 1)), so +-1 is about -+30 %"""
    tau, p, click, grit = BEATERS[beater]
    tau *= (speed(vel) / 3.0) ** (-(p - 1.0) / (p + 1.0)) * 2.0 ** (-float(hard) / (p + 1.0))
    return tau, p, click, grit


def pulse(tau, p=1.5):
    """the contact force sin^p(pi t / tau) sampled at SR, unit sum, starting and ending at zero"""
    k = max(3, int(math.ceil(tau * SR)) + 2)
    u = (np.arange(k) - 0.5) / (tau * SR)
    g = np.where((u > 0) & (u < 1), np.sin(np.pi * np.clip(u, 0, 1)) ** p, 0.0)
    return g / g.sum()


def _strike(y, tau, p):
    """y (n,) or (n, 2) driven by a contact of tau s instead of an impulse (low-passes it the way a real contact
    does: a longer contact leaves the high modes out)"""
    g = pulse(tau, p)
    if y.ndim == 2:
        return signal.fftconvolve(y, g[:, None], axes=0)[:len(y)]
    return signal.fftconvolve(y, g)[:len(y)]


def _click(tau, r, p, click, grit, n, low=500.0, ring_s=0.002):
    """the beater's own tick (modal.contact_noise), n samples"""
    k = modal.contact_noise(tau, r, p, click, grit, ring_s, low)
    out = np.zeros(n)
    out[:min(n, len(k))] = k[:n]
    return out


# ------------------------------------------------------------------------------------------------------ membranes
@functools.lru_cache(maxsize=1)
def _bessel():
    """(m, n, j_mn) for the modes of a circular membrane with a fixed edge, sorted by frequency"""
    rows = [(m, k + 1, j) for m in range(14) for k, j in enumerate(special.jn_zeros(m, 6))]
    rows.sort(key=lambda x: x[2])
    return np.array(rows)


def membrane_modes(count=24, load=0.0, seal=0.0, strike=0.3):
    """The modes of a drum head, as a modal mode set ('ratio' to the (0,1) mode, 'gain' for a strike at `strike`,
    'order' = nodal diameters m) plus 'circles' (nodal circles n).

    The ideal membrane rings at the Bessel zeros j_mn / j_01: 1, 1.594, 2.136, 2.296, 2.653, 2.918 ... Air on the
    head adds mass, most to the slow wide modes: f / sqrt(1 + load (1.594 / ideal)^2), which with load = 1 brings
    (1,1) : (2,1) : (3,1) : (4,1) to 1 : 1.52 : 2.02 : 2.53 (timpani, measured 1 : 1.50 : 1.98 : 2.44; Fletcher &
    Rossing ch. 18). Air shut in by a second head or a kettle pushes back only on the modes that change its volume
    (m = 0): sqrt(1 + seal / ideal^2). A kick with load 1 and seal 1 has its lowest modes at 1 : 1.50 : 2.28.
    strike: 0 = the centre (only the (0,n) modes move there) .. 1 = the edge; the gain is the mode shape there,
    J_m(j_mn r) / |J_m+1(j_mn)| (unit mass)"""
    tb = _bessel()[:count]
    m, nn, j = tb[:, 0].astype(np.int64), tb[:, 1].astype(np.int64), tb[:, 2]
    ideal = j / j[0]
    f = ideal / np.sqrt(1.0 + load * (1.594 / ideal) ** 2)
    f = np.where(m == 0, f * np.sqrt(1.0 + seal / ideal ** 2), f)
    x = float(np.clip(strike, 0.0, 0.97))
    gain = np.sqrt(np.where(m > 0, 2.0, 1.0)) * special.jv(m, j * x) / np.abs(special.jv(m + 1, j))
    s = modal.modes(f / f[0], gain, order=m)
    s['circles'] = nn
    return s


def membrane_rates(ratio, order, circles, t60, tune=0, damp=0.0, mono=1.6, air=0.6):
    """Decay rates (1/s) of the modes of a head: the tuned mode falls 60 dB in t60 s; the others as (f / f_tune)^air
    (radiation and internal loss grow with frequency); modes with no nodal diameter (m = 0) other than the tuned
    one radiate as a monopole and lose energy `mono` times faster; damp (a pillow, a ring, a hand) adds
    damp * (0.2 + (f / f_tune)^2 - 1) of the base rate to the modes from the tuned one up: it shortens the upper
    modes most and the note itself a little (t60 / (1 + 0.2 damp))"""
    ratio = np.asarray(ratio, np.float64)
    q = ratio / ratio[tune]
    base = T60 / t60
    rate = base * q ** air * np.where((np.asarray(order) == 0) & (np.arange(len(q)) != tune), mono, 1.0)
    return rate + damp * base * np.where(q >= 1.0, 0.2 + q * q - 1.0, 0.0)


def _tune_index(s, tune):
    """index of the mode the pitch names: 0 = (0,1), 1 = (1,1)"""
    want = (0, 1) if tune == 0 else (1, 1)
    hit = np.nonzero((s['order'] == want[0]) & (s['circles'] == want[1]))[0]
    return int(hit[0])


def membrane(f, n, r, built, vel=0.7, beater='stick', hard=0.0, strike=0.3, t60=0.5, damp=0.0, load=0.5,
             seal=0.0, tune=0, count=24, fall=0.06, radius=0.18, mono=1.6, tilt=0.0, click=1.0, spread=0.3,
             shell=(), early=0.15, glide=True):
    """One stroke on a drum head tuned so that mode `tune` (0 = (0,1), 1 = (1,1)) is at f Hz: a dict with
    'y' (n, 2) the close pair, 'motion' (n,) the head's displacement at its centre region (peak = vel: what the
    snare wires feel), 'freq' (Hz at rest), 'rate' (1/s), 'amp', 'fall' (initial fractional pitch rise),
    'tau_g' (s), 'tau' (contact s).

    Tension modulation: the stroke stretches the head, so each mode starts fall * vel^2 sharp and relaxes as the
    head's energy decays, with the time constant 1 / (2 rate) of the tuned mode: f(t) = f (1 + D e^(-t / tau_g)),
    whose phase integrates exactly. Pressure at the pair: the mode's gain at the strike point times its radiation
    (monopoles radiate fully; a mode with m nodal diameters as modal.radiation of order m / 2 with the knee where
    the head's circumference is a wavelength: a close pair, so half the far-field order) times ratio^tilt.
    shell: (Hz, t60 s, gain re the strongest mode) resonances of the shell and the air inside, driven by the same
    contact and centred.
    glide=False leaves the pitch still (a roll is built from one stroke's response played many times: train)."""
    s = membrane_modes(count, load, seal, strike)
    k = _tune_index(s, tune)
    f01 = f / s['ratio'][k]
    freq = f01 * s['ratio']
    keep = freq < 0.4 * SR
    rate = membrane_rates(s['ratio'], s['order'], s['circles'], t60, k, damp, mono)
    knee = 343.0 / (2 * np.pi * radius)
    rad = np.where(s['order'] == 0, 1.0, modal.radiation(freq, knee, np.maximum(s['order'], 1) * 0.5))
    amp = s['gain'] * rad * s['ratio'] ** tilt * keep
    amp = amp / (np.abs(amp).sum() + 1e-12)
    tau, p, ck, grit = contact(beater, vel, hard)
    D = fall * vel ** 2 if glide else 0.0
    tau_g = 1.0 / (2.0 * rate[k])
    az = np.where(s['order'] == 0, 0.0, built.uniform(-1.0, 1.0, len(freq)))
    A = modal.mic_pair(amp * np.exp(1j * r.uniform(-0.15, 0.15, len(freq))), az, freq, spread)
    y = np.zeros((n, 2))
    motion = np.zeros(n)
    disp = s['gain'] / s['ratio'] ** 2          # displacement per unit impulse goes as 1 / (mass omega)
    disp = disp / (np.abs(disp).max() + 1e-12)
    for i in np.nonzero(amp != 0)[0]:
        life = min(n, int(math.ceil(1.7 * T60 / rate[i] * SR)))         # to about -100 dB
        t = np.arange(life) / SR
        ph = 2 * np.pi * freq[i] * (t + D * tau_g * (1.0 - np.exp(-t / tau_g)))
        z = np.exp(-rate[i] * t + 1j * ph)
        y[:life, 0] += (A[0, i] * z).imag
        y[:life, 1] += (A[1, i] * z).imag
        if i < 8:
            motion[:life] += disp[i] * z.imag
    for fs, ts, gs in shell:
        life = min(n, int(math.ceil(1.7 * ts * SR)))
        t = np.arange(life) / SR
        sh = gs * np.abs(amp).max() * np.exp(-T60 / ts * t) * np.sin(2 * np.pi * fs * t)
        y[:life] += sh[:, None]
    lvl = speed(vel) / 7.0
    y = _strike(y, tau, p) * lvl
    if click:
        y += (click * lvl * _click(tau, r, p, ck, grit, n))[:, None]
    if early:
        y = modal.early(y, built, early)
    motion = _strike(motion, tau, p)
    motion *= vel / (np.abs(motion).max() + 1e-12)
    return {'y': y, 'motion': motion, 'freq': freq, 'rate': rate, 'amp': amp, 'fall': D, 'tau_g': tau_g, 'tau': tau}


# ------------------------------------------------------------------------------------------------------ wires
@njit(cache=True)
def _follow(x, rel):
    """peak follower: jumps up at once, falls by the factor rel per sample"""
    y = np.empty_like(x)
    v = 0.0
    for i in range(x.shape[0]):
        a = x[i]
        v = a if a > v else v * rel
        y[i] = v
    return y


def rattle(drive, r, gap=0.08, release=0.02, bands=((2600.0, 1.4, 1.0), (4800.0, 1.8, 0.8), (8500.0, 2.2, 0.5)),
           body=0.3, body_hz=200.0, grain=3.0, stereo=0.5):
    """Loose metal against a moving surface (snare wires on the bottom head, two hi-hat cymbals, a cajon's wires):
    stereo (n, 2), the buzz in level units of `drive` (displacement, peak about the velocity).

    The wires touch the head on one side: whenever it swings more than `gap` into them they are slapped, as hard
    as the head moves past the gap (the excess: the head's speed at the contact goes with its excursion); every
    slap sets them bouncing, and they ring on with time constant `release` s, so the buzz follows the head, swells
    a little after the stroke and outlasts it by a few tens of ms. Each slap is many coils hitting at once:
    impulsive
    noise (a Gaussian raised to `grain`) under that envelope, ringing in the wires' bands (Hz, q, gain). The slaps
    come once per cycle of the head, so the force itself carries the head's frequency and its harmonics: the
    150-250 Hz body of a snare (body, around body_hz)."""
    n = len(drive)
    c = np.maximum(np.asarray(drive, np.float64) - gap, 0.0)
    if not c.any():
        return np.zeros((n, 2))
    a = math.exp(-1.0 / (max(release, 1e-4) * SR))
    g = signal.lfilter([1.0 - a], [1.0, -a], _follow(c, a ** 4))      # each slap sets them going; they ring on
    out = np.zeros((n, 2))
    for ch in range(2):
        w = r.standard_normal(n)
        w = w * np.abs(w) ** (grain - 1.0)
        w /= np.sqrt(np.mean(w * w)) + 1e-12
        e = g * w
        x = sum(gain * dsp.svf(e, fc, q, 'bandn') for fc, q, gain in bands)
        out[:, ch] = x
    mid = out.mean(1, keepdims=True)
    out = mid + stereo * (out - mid)
    if body:
        b = dsp.butter(c, 'band', (body_hz / 1.6, body_hz * 1.6), 2)
        out += (body * b / (np.abs(b).max() + 1e-12) * np.abs(out).max())[:, None]
    return out


def bottom_head(motion, f, q=8.0, delay=0.0004):
    """the snare-side head as the top head drives it through the air in the shell: a resonance at f Hz (quality
    q, so it fills over q / (pi f) s) a fraction of a millisecond later; peak normalised to that of motion"""
    d = int(delay * SR)
    x = np.concatenate([np.zeros(d), motion[:len(motion) - d]])
    y = dsp.svf(x, f, q, 'bandn')
    return y * (np.abs(motion).max() / (np.abs(y).max() + 1e-12))


# ------------------------------------------------------------------------------------------------------ bodies
def body(freq, t60, gain, n, r, built, vel=0.7, beater='wood', hard=0.0, spread=0.3, click=1.0, early=0.1,
         low=500.0):
    """A struck rigid body (a wooden bar or box, a stick, a hoop) of given modes: freq Hz, t60 s, gain (arrays):
    stereo (n, 2) from the close pair, driven by the beater's contact, with its tick"""
    freq, t60, gain = (np.atleast_1d(np.asarray(v, np.float64)) for v in (freq, t60, gain))
    tau, p, ck, grit = contact(beater, vel, hard)
    az = built.uniform(-1.0, 1.0, len(freq))
    A = modal.mic_pair(gain * np.exp(1j * r.uniform(-0.2, 0.2, len(freq))), az, freq, spread)
    y = modal.ring(freq, T60 / t60, A, n, tau, p).imag.T
    lvl = speed(vel) / 7.0
    y = y / (np.abs(gain).sum() + 1e-12) * lvl
    if click:
        y += (click * lvl * _click(tau, r, p, ck, grit, n, low))[:, None]
    return modal.early(y, built, early) if early else y


# ------------------------------------------------------------------------------------------------------ events
def _impulses(n, t, a, pan):
    """stereo impulse trains: amplitude a at times t s, constant-power panned"""
    i = np.round(np.asarray(t) * SR).astype(np.int64)
    ok = (i >= 0) & (i < n)
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    out = np.zeros((n, 2))
    np.add.at(out[:, 0], i[ok], (a * np.cos(th))[ok])
    np.add.at(out[:, 1], i[ok], (a * np.sin(th))[ok])
    return out


def energy(n, strokes, settle, cr=2000):
    """the energy of shaken beads at cr Hz over n samples: each stroke (t s, e) adds e at the moment the beads hit
    the wall, then it decays with time constant `settle` s (friction and collisions)"""
    m = max(2, int(n / SR * cr))
    t = np.arange(m) / cr
    E = np.zeros(m)
    for ts, e in strokes:
        E += e * np.exp(-np.maximum(t - ts, 0.0) / settle) * (t >= ts)
    return E


def shaker(n, r, strokes, beads=50.0, settle=0.035, pre=0.25, travel=0.05,
           res=((5200.0, 3.0, 1.0), (8400.0, 2.5, 0.6)), grain=0.0004, width=0.6):
    """A shaker (Cook's PhISM shaker, 1997): stereo (n, 2).

    The beads carry the energy of the gesture; at every stroke (t s, energy) they reach the wall together. The
    rate of collisions is beads * 600 * E(t) per second and each collision's size goes as sqrt(E) with a
    log-normal spread, so a fresh stroke is a dense loud burst that thins out into single grains as they settle
    (time constant `settle`). Before each wall hit the beads roll and slide for `travel` s at `pre` of the
    energy. Every collision is a broadband tick `grain` s long, heard through the shell's resonances (Hz, q, gain)
    and directly"""
    dur = n / SR
    pre_strokes = [(ts - travel, e * pre) for ts, e in strokes if ts - travel >= 0] if pre else []
    cr = 2000
    E = energy(n, strokes, settle, cr)
    if pre_strokes:
        Er = np.zeros_like(E)
        tt = np.arange(len(E)) / cr
        for ts, e in pre_strokes:
            Er += e * ((tt >= ts) & (tt < ts + travel)) * np.sin(np.pi * np.clip((tt - ts) / travel, 0, 1))
        E = E + Er
    from .texture import arrivals
    t, _ = arrivals(r, beads * 600.0 * E, dur)
    if not len(t):
        return np.zeros((n, 2))
    Ei = E[np.minimum((t * cr).astype(np.int64), len(E) - 1)]
    a = np.sqrt(Ei) * np.minimum(np.exp(0.45 * r.standard_normal(len(t))), 2.5)
    pan = np.clip(width * r.uniform(-1, 1, len(t)), -1, 1)
    x = _impulses(n, t, a, pan)
    k = max(3, int(grain * SR))
    tick = r.standard_normal(k) * np.hanning(k)
    tick -= tick.mean() * np.hanning(k) / np.hanning(k).mean()
    x = signal.fftconvolve(x, tick[:, None], axes=0)[:n]
    y = 0.3 * dsp.butter(x, 'high', 2500.0, 2)
    for fc, q, gain in res:
        for ch in range(2):
            y[:, ch] += gain * dsp.svf(x[:, ch], fc, q, 'bandn')
    return dsp.butter(y, 'low', 13000.0, 2)          # a bead's tick has little above 13 kHz (it is no impulse)


# ------------------------------------------------------------------------------------------------------ hands
def clap(n, r, at=0.0, cavity=1300.0, q=3.0, vel=0.8, slap=0.6, length=0.0015, flesh=0.35):
    """One hand clap (mono, n samples, the contact at `at` s): the palms meet over a millisecond or two (a slap of
    broadband noise, brighter at a higher velocity), the air pushed out of the cavity between them rings once or
    twice at `cavity` Hz (0.6-2.5 kHz by hand shape, a broad peak: Repp 1987; Peltola et al. 2007) with quality
    q, and the soft flesh of the palms gives a dull short thud under it (250-800 Hz, about 6 ms)"""
    out = np.zeros(n)
    m = min(n - dsp.n_of(at), dsp.n_of(0.06))
    if m <= 8:
        return out
    t = np.arange(m) / SR
    L = length * (1.3 - 0.5 * vel)
    e = np.clip(t / L, 0, 1) ** 2 * np.exp(-np.maximum(t - L, 0.0) / (0.6 * L + 0.0012))
    w = r.standard_normal(m) * e
    sl = dsp.butter(w, 'band', (400.0, 4000.0 + 7000.0 * vel), 2)
    cav = dsp.svf(w, cavity * r.uniform(0.95, 1.05), q, 'bandn') * math.sqrt(q)
    th = dsp.butter(r.standard_normal(m), 'band', (250.0, 800.0), 2) * np.clip(t / L, 0, 1) * np.exp(-t / 0.006)
    x = slap * sl + cav + flesh * th * (np.abs(cav).max() / (np.abs(th).max() + 1e-12))
    i = dsp.n_of(at)
    out[i:i + m] += x
    return out


def snap(n, r, at=0.02, cavity=2800.0, q=6.0, vel=0.8):
    """A finger snap (mono): the thumb's skin slips over the finger first (a soft friction hiss 15-30 ms long),
    then the middle finger strikes the palm and the cavity between palm and thumb pops: a sharper, shorter and
    higher clap (contact about 0.3 ms, cavity 2-3.5 kHz)"""
    t_slip = r.uniform(0.012, 0.022)
    out = clap(n, r, at, cavity, q, vel, slap=0.5, length=0.0004, flesh=0.1)
    k = dsp.n_of(t_slip)
    i = max(0, dsp.n_of(at) - k)
    hiss = dsp.butter(r.standard_normal(k), 'band', (2000.0, 9000.0), 2) * np.sin(np.pi * np.arange(k) / k) ** 2
    out[i:i + k] += 0.04 * hiss / (np.abs(hiss).max() + 1e-12) * np.abs(out).max()
    return out


# ------------------------------------------------------------------------------------------------------ synthetic
def _env(n, attack, tau, hold=0.0):
    t = np.arange(n) / SR
    e = np.exp(-np.maximum(t - attack - hold, 0.0) / max(tau, 1e-5))
    a = min(n, dsp.n_of(attack))
    if a > 1:
        e[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    return e


def kick_synth(n, r, f_end=48.0, vel=0.8, sweep=2.6, drop=0.045, snap_=0.006, decay=0.5, click=0.6, drive=7.0):
    """The synthetic kick (mono): a sine whose pitch starts `sweep` octaves over f_end Hz and falls in two stages
    (snap_ s: the beater's snap; drop s: the body settling; more sweep at a higher velocity), an amplitude that
    holds briefly and decays with time constant `decay` s (the long sub tail), tape saturation that writes the
    harmonics small speakers need (more drive with velocity), and a click: 1-2 ms of bright noise and a 4 kHz blip"""
    t = np.arange(n) / SR
    oc = sweep * (0.75 + 0.25 * vel)
    semis = 12 * oc * (0.62 * np.exp(-t / snap_) + 0.38 * np.exp(-t / drop))
    f = f_end * 2.0 ** (semis / 12)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR)
    y *= _env(n, 0.0004, decay, 0.012) * (1 + 0.35 * np.exp(-t / 0.03))
    y = fx.saturate(y, drive + 6.0 * vel, 'tape', level='peak')
    if click:
        k = min(n, dsp.n_of(0.01))
        tk = t[:k]
        c = dsp.butter(r.standard_normal(k), 'high', 2000.0, 2) * np.exp(-tk / 0.0008)
        c += 0.7 * np.sin(2 * np.pi * 4000.0 * tk) * np.exp(-tk / 0.0015)
        y[:k] += click * (0.5 + 0.5 * vel) * 0.5 * c / (np.abs(c).max() + 1e-12)
    return y


def snare_synth(n, r, f=185.0, vel=0.8, tone=0.5, noise_tau=0.13, snappy=1.0, bright=0.0):
    """The synthetic snare (mono): two tuned tones (f and 1.78 f, a bridged-T pair: each a sine that falls a few
    per cent as it decays, 45 and 30 ms) and noise through a high-pass and a band around 5 kHz with its own
    longer envelope (the 'snappy'), the noise brighter at a higher velocity"""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ff, tau_, g in ((f, 0.045, 1.0), (1.78 * f, 0.03, 0.6)):
        fi = ff * (1 + 0.06 * vel * np.exp(-t / 0.01))
        y += g * np.sin(2 * np.pi * np.cumsum(fi) / SR) * _env(n, 0.0003, tau_)
    w = r.standard_normal(n)
    nz = dsp.butter(w, 'high', 1500.0 + 1000.0 * bright, 2) + 0.7 * dsp.svf(w, 5200.0 * 2 ** (0.4 * vel + 0.5 * bright),
                                                                           1.2, 'bandn')
    nz *= _env(n, 0.0005, noise_tau) * (1 + 1.5 * np.exp(-t / 0.008))
    y = tone * y / (np.abs(y).max() + 1e-12) + snappy * 0.8 * nz / (np.abs(nz).max() + 1e-12)
    return fx.saturate(y, 4.0 + 4.0 * vel, 'tanh', level='peak')


def clap_synth(n, r, vel=0.8, bursts=4, gap=0.0105, centre=1150.0, tail=0.14):
    """The synthetic clap (mono): band-passed noise (one band around `centre` Hz, q 1.6) in `bursts` short bursts
    `gap` s apart (each 5 ms, a sharp attack; the drum machine's sawtooth retriggers), then a tail decaying with
    time constant `tail` s; brighter at a higher velocity"""
    t = np.arange(n) / SR
    env = np.zeros(n)
    for b in range(bursts):
        t0 = b * gap * r.uniform(0.9, 1.1)
        env += (t >= t0) * np.exp(-np.maximum(t - t0, 0) / 0.004) * (0.8 + 0.2 * (b == bursts - 1))
    t0 = (bursts - 1) * gap
    env = np.maximum(env, 0.75 * (t >= t0) * np.exp(-np.maximum(t - t0, 0) / tail))
    a = dsp.n_of(0.0002)
    env[:a] *= np.linspace(0, 1, a)
    w = r.standard_normal(n)
    y = dsp.svf(w, centre * 2 ** (0.3 * vel), 1.6, 'bandn') + 0.35 * dsp.butter(w, 'high', 3000.0 + 2000 * vel, 2)
    return y * env


HAT_OSC = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)    # the six square oscillators of a classic analog drum
#                                                         machine's metal voices (its published circuit values)


def hat_synth(n, r, vel=0.8, decay=0.04, scale=1.0):
    """The synthetic hi-hat (mono): six band-limited square waves at inharmonic frequencies (HAT_OSC times scale)
    summed into a dense metallic cluster, through two band-passes (7 and 10 kHz) and a high-pass at 6 kHz, with a
    sharp exponential decay (decay s); a little noise for the air. Brighter at a higher velocity"""
    x = sum(dsp.osc(fo * scale * (1 + 0.002 * r.standard_normal()), 'square', n / SR, r.uniform())[:n]
            for fo in HAT_OSC)
    x = x + 0.4 * r.standard_normal(n)
    y = dsp.svf(x, 7100.0 * 2 ** (0.2 * vel), 2.5, 'bandn') + 0.7 * dsp.svf(x, 10200.0, 2.0, 'bandn')
    y = dsp.butter(y, 'high', 6000.0, 2)
    return y * _env(n, 0.0002, decay)


# ------------------------------------------------------------------------------------------------------ patterns
def train(n, ir, times, gains):
    """a linear body struck many times (a roll, a swell): the response ir ((m,) or (m, 2)) placed at `times` s with
    `gains`, n samples, stereo"""
    ir = np.stack([ir, ir], 1) if ir.ndim == 1 else ir
    x = np.zeros(n)
    i = np.round(np.asarray(times) * SR).astype(np.int64)
    ok = (i >= 0) & (i < n)
    np.add.at(x, i[ok], np.asarray(gains, np.float64)[ok])
    return signal.fftconvolve(x[:, None], ir, axes=0)[:n]
