"""The physics of struck and plucked bodies: where their modes lie, how fast each one dies, what a mallet does to them.

A bar, a bell or a wine glass rings in modes. Whether the sum is heard as an instrument or as a synthesizer is decided
by three things. The modes sit where the geometry puts them (a free bar 1 : 2.76 : 5.40, a church bell hum / prime /
tierce / quint / nominal). Every mode decays at the rate its material and its frequency give it, never one envelope
for the whole note. And the excitation is a contact: a force pulse a fraction of a millisecond long whose length
follows the hardness, mass and speed of the mallet (Hertz), so a harder or faster strike is brighter, not only louder.

    from earshot.forge import dsp
    from earshot.synth import modal
    m = modal.bar_modes(1047.0, strike=0.42)                        # a steel glockenspiel bar, C6
    f = 1047.0 * m['ratio']
    rate = modal.decay_rate(f, 'steel', support=0.5) * m['damp']    # 1/s per mode; it falls 60 dB in 6.91 / rate s
    tau = modal.contact_time(0.010, 4e8, 2.8)                       # a 10 g hard plastic mallet at 2.8 m/s: 0.15 ms
    y = modal.ring(f, rate, m['gain'], dsp.n_of(2.0), tau).imag[0]  # exact damped sinusoids behind a real attack

Mode sets (bar_modes, tuned_bar_modes, tine_modes, tube_modes, bell_modes, rim_modes, plate_modes) are dicts of
arrays: 'ratio' (frequency over the fundamental), 'gain' (signed, for the given strike position), 'damp' (factor on
the decay law), 'split' (doublet split of a round body, fraction of the frequency), 'order' (how often the pattern
repeats around the body) and 'held' (how much the mode moves where the body is supported: what the cord or felt
takes from it). decay_rate is the damping law, contact_time and ring the excitation, doublets the beating pairs of
bells and bowls, resonator the tube under a bar (and the vibraphone motor), body_response a soundboard,
contact_noise the tick of the mallet itself, mic_pair and early a close stereo pair, damper a hand on the bar.
No reverb here: a shared room is added by another module.
"""
import math

import numpy as np
from scipy import signal

from ..forge import dsp

SR = dsp.SR
T60 = math.log(1000.0)      # a mode decaying at `rate` 1/s falls 60 dB in T60 / rate seconds

# Internal friction per material. eta is the loss factor (1 / Q): alone it gives the rate pi * eta * f, a ring time
# that halves per octave. visc is the coefficient of the f^2 term: viscoelastic loss in wood (Chaigne & Doutaut 1997
# fit xylophone bars with rate = a0 + a2 f^2), thermoelastic and radiation growth in metal and glass. c is the bar
# wave speed sqrt(E / rho) in m/s. Published ranges of eta: steel and aluminium 1e-4 .. 1e-3, bronze and brass
# 1e-4 .. 2e-3, glass 5e-4 .. 3e-3, hardwood along the grain 4e-3 .. 2e-2. The values sit inside those ranges and
# are set so that bodies of concert size ring as long as the instrument tables in struck.py say.
MATERIALS = {
    'steel': {'eta': 1.4e-4, 'visc': 1.1e-7, 'c': 5100.0},
    'aluminium': {'eta': 1.6e-4, 'visc': 3.0e-8, 'c': 5050.0},
    'brass': {'eta': 2.0e-4, 'visc': 6.0e-8, 'c': 3450.0},
    'bronze': {'eta': 1.8e-4, 'visc': 4.0e-8, 'c': 3600.0},
    'crystal': {'eta': 4.5e-4, 'visc': 1.5e-7, 'c': 4500.0},
    'glass': {'eta': 1.4e-3, 'visc': 3.0e-7, 'c': 5200.0},
    'rosewood': {'eta': 4.3e-3, 'visc': 2.1e-6, 'c': 4400.0},
    'padouk': {'eta': 6.5e-3, 'visc': 3.2e-6, 'c': 4200.0},
    'synthetic': {'eta': 3.0e-3, 'visc': 1.4e-6, 'c': 3800.0},
    'maple': {'eta': 9.0e-3, 'visc': 5.0e-6, 'c': 4100.0},
}


# ------------------------------------------------------------------------------------------------------- damping
def decay_rate(f, material, support=0.0, radiation=0.0, f_body=1.0):
    """Amplitude decay rate in 1/s of a mode at f Hz (the mode falls 60 dB in 6.91 / rate seconds):

        rate = support + pi * eta * (1 + radiation * f / f_body) * f + visc * f^2

    support: what the cord, felt or hand takes plus air drag; a number, or one value per mode where the support
    sits at a node of some modes and not of others (the `held` of a mode set times a rate). pi * eta * f:
    internal friction at a constant loss factor. visc * f^2: the viscoelastic / thermoelastic term that makes the
    upper modes of a bar die much faster than its fundamental. radiation: a round body sheds its higher modes into
    the air more efficiently, so its loss factor rises with the ratio of the mode to the fundamental f_body"""
    m = MATERIALS[material]
    f = np.asarray(f, np.float64)
    return support + np.pi * m['eta'] * (1.0 + radiation * f / f_body) * f + m['visc'] * f * f


def ring_time(rate, amp, floor_db=-60.0, longest=40.0):
    """seconds until the sum of the mode envelopes amp * exp(-rate t) has fallen floor_db under its start"""
    a, rate = np.abs(np.asarray(amp)).ravel(), np.asarray(rate, np.float64).ravel()
    target = a.sum() * 10 ** (floor_db / 20)
    lo, hi = 0.0, float(longest)
    if (a * np.exp(-rate * hi)).sum() > target:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if (a * np.exp(-rate * mid)).sum() > target else (lo, mid)
    return hi


def damper(n, at, t60=0.12, land=0.006):
    """Gain over n samples for a damper (felt, a hand, the mallet held on the bar) that lands at `at` seconds: over
    `land` seconds the extra decay rate of every mode rises smoothly to 6.91 / t60, so the note dies in about t60
    with its timbre intact instead of being cut"""
    u = np.clip((dsp.t_of(n) - at) / max(land, 1e-4), 0.0, 1.0)
    return np.exp(-np.cumsum(u * u * (3.0 - 2.0 * u)) * (T60 / t60) / SR)


# ---------------------------------------------------------------------------------------------------- excitation
def contact_time(mass, stiffness, speed, exponent=1.5):
    """Seconds a mallet of `mass` kg stays on the bar when it arrives at `speed` m/s and its head compresses with
    force = stiffness * depth^exponent (Hertz: exponent 1.5 for an elastic ball on a flat, stiffness in N/m^1.5;
    felt and yarn stiffen faster, exponent 2 .. 3). From the energy balance:

        depth = ((p + 1) m v^2 / (2 K))^(1 / (p + 1)),   time = c(p) * depth / v,   c(1.5) = 2.94

    so the time goes as m^(1/(p+1)) K^(-1/(p+1)) v^(-(p-1)/(p+1)): v^-0.2 for Hertz, v^-0.43 for p = 2.5"""
    q = 1.0 / (exponent + 1.0)
    depth = ((exponent + 1.0) * mass * speed * speed / (2.0 * stiffness)) ** q
    return 2.0 * math.sqrt(math.pi) * math.gamma(1.0 + q) / math.gamma(0.5 + q) * depth / speed


def _steps(tau):
    return max(8, int(math.ceil(64.0 / (tau * SR))))


def _pulse(tau, exponent, steps):
    """the contact force sin^p(pi t / tau) on a grid of `steps` points per sample, unit area: (times, force)"""
    k = int(math.ceil(tau * SR)) + 1
    u = np.arange(k * steps + 1) / (steps * SR)
    g = np.sin(np.pi * np.minimum(u / tau, 1.0)) ** exponent
    return u, g * (steps * SR / g.sum())


def force_gain(f, tau, exponent=1.5):
    """How much of a unit impulse reaches a mode at f Hz when the force is a pulse sin^p(pi t / tau) of the same
    momentum: 1 well below 1 / tau, falling steeply above about 1.5 / tau. This is why a soft mallet is dark"""
    f = np.atleast_1d(np.asarray(f, np.float64))
    if tau <= 0:
        return np.ones(len(f))
    steps = _steps(tau)
    u, g = _pulse(tau, exponent, steps)
    return np.abs(np.exp(-2j * np.pi * f[:, None] * u) @ g) / (steps * SR)


def _bank(s, coef, n):
    """sum over modes of coef[c, m] * exp(s[m] t) at t = 0 .. n-1 samples, complex (channels, n). Exact and
    vectorised: exp(s t) at sample k of block b is exp(s b B) * exp(s k), so one matrix product per group of modes
    gives all samples. Modes are grouped by how long they live (down to -140 dB), so the short ones cost little"""
    out = np.zeros((coef.shape[0], n), np.complex128)
    if not len(s):
        return out
    block = int(2 ** np.clip(round(math.log2(math.sqrt(n) + 1.0)), 6, 11))
    life = np.minimum(n, np.ceil(16.1 * SR / np.maximum(-s.real, 1e-3))).astype(np.int64)
    order = np.argsort(-life)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and life[order[j]] * 2 > life[order[i]]:
            j += 1
        idx = order[i:j]
        blocks = int(math.ceil(life[order[i]] / block))
        within = np.exp(s[idx, None] * (np.arange(block) / SR))
        start = np.exp(s[None, idx] * (np.arange(blocks)[:, None] * (block / SR)))
        m = min(n, blocks * block)
        for c in range(coef.shape[0]):
            out[c, :m] += ((start * coef[c, idx]) @ within).reshape(-1)[:m]
        i = j
    return out


def ring(freq, rate, amp, n, tau=0.0, exponent=1.5):
    """The response of a set of modes to one force pulse, as the complex analytic signal of shape (channels, n); its
    imaginary part is the sound. freq Hz, rate 1/s, amp real or complex with shape (modes,) or (channels, modes).

    Each mode answers an impulse with amp * exp(-rate t) * sin(2 pi f t): sine phase, so the sound leaves zero
    without a step. The contact pulse (tau seconds, sin^p shaped, see contact_time) is convolved in analytically:
    after the contact every mode is an exact damped sinusoid whose level and phase carry the spectrum of the pulse
    at its frequency, and during the contact the running integral gives the true rise. So the onset is a real attack
    with no fade-in, and a longer contact removes the high modes by physics. An imaginary amp starts a mode in
    cosine phase (a pluck: the release of a held bend). Modes above 0.45 SR are dropped, those above 0.40 SR fade"""
    freq, rate = np.asarray(freq, np.float64), np.asarray(rate, np.float64)
    amp = np.atleast_2d(np.asarray(amp, np.complex128))
    edge = np.clip((0.45 * SR - freq) / (0.05 * SR), 0.0, 1.0)
    keep = (edge > 0) & (freq > 0) & (np.abs(amp).max(axis=0) > 0)
    s = (-rate + 2j * np.pi * freq)[keep]
    amp = amp[:, keep] * np.sin(0.5 * np.pi * edge[keep]) ** 2
    if tau <= 0 or not len(s):
        return _bank(s, amp, n)
    steps = _steps(tau)
    u, g = _pulse(tau, exponent, steps)
    z = g * np.exp(-s[:, None] * u)
    run = np.zeros(z.shape, np.complex128)
    np.cumsum((z[:, 1:] + z[:, :-1]) * (0.5 / (steps * SR)), axis=1, out=run[:, 1:])
    out = _bank(s, amp * run[:, -1], n)
    k = min(n, (len(u) - 1) // steps)
    out[:, :k] = amp @ (np.exp(s[:, None] * u[::steps][:k]) * run[:, ::steps][:, :k])
    return out


def contact_noise(tau, r, exponent=1.5, click=0.2, grit=0.05, ring_s=0.003, low=400.0):
    """The sound of the contact itself (mono, starts at zero), in the units of ring() for a mode of gain 1.

    Two parts. The click: while they touch, mallet and bar push each other apart, and that acceleration radiates a
    pressure pulse shaped like the derivative of the force, as long as the contact. Its spectrum peaks near 1 / tau
    and its peak level goes as 1 / tau^2, so brass on steel ticks and yarn on rosewood does not (`click` is the peak
    for a 0.1 ms contact). The grit: the many high modes no mode table lists (and those of the mallet head) are
    excited too and die within milliseconds; together they are a noise burst under the force spectrum, level going
    as tau^-0.5, decaying with time constant ring_s. Its top falls only 6 dB per octave above 1.5 / tau: a mallet
    face is not a smooth ball, and yarn, felt and wood grain put more high frequencies into the contact than the
    Hertz pulse has. High-passed at `low` Hz"""
    steps = _steps(tau)
    _, g = _pulse(tau, exponent, steps)
    g = np.concatenate([[0.0], g[:-1].reshape(-1, steps).mean(axis=1) / SR])       # per sample, unit sum, from zero
    n = len(g) + dsp.n_of(7.0 * ring_s) + 1
    out = np.zeros(n)
    out[:len(g) + 1] = np.diff(g, prepend=0.0, append=0.0) * (click * (1e-4 * SR) ** 2 / 3.5)
    rise = np.ones(n)
    rise[:len(g)] = np.cumsum(g)
    w = dsp.butter(r.standard_normal(n), 'low', min(1.5 / tau, 0.42 * SR), 1)
    w = dsp.butter(w, 'high', low, 2)
    env = rise * np.exp(-np.maximum(dsp.t_of(n) - tau, 0.0) / ring_s)
    return out + w * env * (grit * math.sqrt(3e-4 / tau) / (np.sqrt(np.mean(w * w)) + 1e-12))


# ----------------------------------------------------------------------------------------------------- mode sets
def modes(ratio, gain, damp=1.0, split=0.0, order=0, held=0.0):
    """a mode set from arrays or scalars: ratio to the fundamental, signed gain, factor on the decay law, doublet
    split (fraction of the frequency, 0 = a single mode), circumferential order of the pattern, and held: how much
    the mode moves where the body is supported (squared, 0 = the support sits on a node and takes nothing)"""
    ratio = np.atleast_1d(np.asarray(ratio, np.float64))
    one = np.ones(len(ratio))
    return {'ratio': ratio, 'gain': gain * one, 'damp': damp * one, 'split': split * one,
            'order': (order * one).astype(np.int64), 'held': held * one}


def join(*sets):
    """several mode sets of one body as one"""
    return {k: np.concatenate([s[k] for s in sets]) for k in sets[0]}


_FREE = (4.730040745, 7.853204624, 10.995607838, 14.137165491, 17.278759657)       # cos(b) cosh(b) = 1
_CLAMPED = (1.875104069, 4.694091133, 7.854757438, 10.995540735, 14.137168391)    # cos(b) cosh(b) = -1


def _beta(count, free=True):
    k = np.arange(1, count + 1)
    b = (2 * k + 1) * np.pi / 2 if free else (2 * k - 1) * np.pi / 2
    first = _FREE if free else _CLAMPED
    b[:min(count, 5)] = first[:count]
    return b


def beam_shape(beta, x, free=True):
    """Mode shapes of a uniform beam at x (0 .. 1 along its length), one value per beta, normalised to 2 at a free
    end. free=True: both ends free (bars, tubes, rods); the fundamental has its nodes at 0.224 and 0.776 and every
    second mode has a node in the middle. free=False: clamped at x = 0, free at x = 1 (tines)"""
    beta = np.asarray(beta, np.float64)
    b = np.minimum(beta, 30.0)                      # beyond that cosh loses its digits and the limit form is exact
    if free:
        sig = (np.cosh(b) - np.cos(b)) / (np.sinh(b) - np.sin(b))
        exact = np.cosh(b * x) + np.cos(b * x) - sig * (np.sinh(b * x) + np.sin(b * x))
        even = np.where(np.round(beta / np.pi - 0.5) % 2 == 1, 1.0, -1.0)
        far = np.sqrt(2.0) * np.cos(beta * x + np.pi / 4) + np.exp(-beta * x) + even * np.exp(-beta * (1.0 - x))
    else:
        sig = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
        exact = np.cosh(b * x) - np.cos(b * x) - sig * (np.sinh(b * x) - np.sin(b * x))
        even = np.where(np.round(beta / np.pi + 0.5) % 2 == 1, 1.0, -1.0)
        far = np.sqrt(2.0) * np.sin(beta * x - np.pi / 4) + np.exp(-beta * x) + even * np.exp(-beta * (1.0 - x))
    return np.where(beta < 30.0, exact, far)


def bar_modes(f0, strike=0.42, thickness=0.0095, width=0.032, c=5100.0, count=8, off_axis=0.15, rest=0.224):
    """A uniform rectangular bar, free at both ends, tuned to f0 Hz (a glockenspiel bar; any metal offcut).

    Bending modes of a thin bar: 1 : 2.756 : 5.404 : 8.933 ... (the squares of 3.011, 5, 7, 9). A real bar is not
    thin: shear and rotary inertia (Timoshenko) pull the upper modes down by 1 / sqrt(1 + 0.343 (beta h / L)^2),
    more for the short high bars (measured glockenspiel: 1 : 2.71 : 5.15). The length follows from f0, thickness h
    and wave speed c: f1 = 1.028 h c / L^2. Besides bending, a real bar has torsional modes (n c_t / 2L: about 3.6
    and 7.2 times f0 at C6, lower against f0 on the short bars) and stiff lateral modes ((width / thickness) times
    the bending series): both weak, driven only as far as the mallet lands off the centre line (off_axis, fraction
    of the half width) or not quite vertically. strike: position along the bar (0 end .. 0.5 middle). rest: where
    the bar lies on its felt or cord, the nodes of the fundamental; every other mode moves there (`held`) and loses
    energy to the support, which is why the upper modes of a bar are gone long before its fundamental"""
    length = math.sqrt(1.028 * thickness * c / f0)
    beta = _beta(count)
    bend = beta ** 2 / np.sqrt(1.0 + 0.343 * (beta * thickness / length) ** 2)
    h, w = thickness, width
    kt = w * h ** 3 * (1 / 3 - 0.21 * h / w * (1 - h ** 4 / (12 * w ** 4)))      # torsion constant of a rectangle
    ct = c / 1.61 * math.sqrt(kt / (w * h * (w * w + h * h) / 12))               # torsional wave speed
    k = np.arange(1, 4)
    twist = modes(k * ct / (2 * length) / f0, 3.0 * off_axis ** 2 * np.cos(k * np.pi * strike),
                  held=0.5 * np.cos(k * np.pi * rest) ** 2)
    side = beta[:2] ** 2 / np.sqrt(1.0 + 0.343 * (beta[:2] * w / length) ** 2) * (w / h)
    side = modes(side / bend[0], 0.015 * beam_shape(beta[:2], strike), held=0.5)
    return join(modes(bend / bend[0], 0.5 * beam_shape(beta, strike), held=(0.5 * beam_shape(beta, rest)) ** 2),
                twist, side)


def tuned_bar_modes(second=4.0, third=10.0, strike=0.42, count=7, twist=5.0, off_axis=0.15, rest=0.224):
    """A bar with an arch cut under its middle (marimba, vibraphone, xylophone). The cut lowers the fundamental far
    more than the upper modes, and the tuner sets the second mode to 4 (marimba, vibraphone: two octaves) or 3
    (xylophone: a twelfth) and, on the low bars, the third near 10 (or 6): pass the ratios the bar at this pitch
    has. Modes 4 and up are not tuned; measured bars (1 : 3.98 : 10.7 : 18.0 : 23.7 : 33.6, 1 : 4.00 : 9.47 :
    15.6 : 20.9 : 29.4, 1 : 3.93 : 9.54 : 16.7 : 24.6 : 31.1) grow by about 1.69, 1.38, 1.37 from one mode to the
    next, which is what is used. Strike weights and the motion at the cord (rest, see bar_modes) are the shapes of
    the uniform bar, an approximation. twist: ratio of the first torsional mode (weak, see bar_modes)"""
    ratio = [1.0, second, third]
    for grow in (1.69, 1.38, 1.37, 1.30, 1.26, 1.22)[:max(0, count - 3)]:
        ratio.append(ratio[-1] * grow)
    k = np.arange(1, 3)
    beta = _beta(count)
    return join(modes(ratio[:count], 0.5 * beam_shape(beta, strike), held=(0.5 * beam_shape(beta, rest)) ** 2),
                modes(k * twist, 3.0 * off_axis ** 2 * np.cos(k * np.pi * strike),
                      held=0.5 * np.cos(k * np.pi * rest) ** 2))


def tine_modes(second=6.27, pluck=0.95, count=4):
    """A tine clamped at one end (kalimba, music box tooth): 1 : 6.27 : 17.55 : 34.39 for an ideal clamp. A real
    kalimba tine rests on a bridge with a springy back length, which lowers the upper modes (second = 5 .. 6); the
    others scale with it. The gain is the force each mode puts on the bridge after a pluck at `pluck` (0 clamp ..
    1 tip), the release of a static bend: shape(pluck) / sqrt(ratio), all in cosine phase"""
    beta = _beta(count, free=False)
    ratio = (beta / beta[0]) ** 2
    ratio[1:] *= second / ratio[1]
    return modes(ratio, 0.5 * beam_shape(beta, pluck, free=False) / np.sqrt(ratio))


def tube_modes(f0, mode=1, gyration=0.0124, c=3430.0, strike=0.03, count=10, split=0.0015, rest=0.224,
               shear=5.9):
    """A tube or rod free at both ends, with bending mode number `mode` at f0 Hz (wind chime, triangle rod: mode 1;
    a tubular bell is named by its strike tone, an octave below mode 4: pass twice the strike tone and mode=4).

    Thin-beam ratios (2n+1)^2: 9.07 : 25 : 49 : 81 : 121 : 169. A real tube is wide for its length; with shear and
    rotary inertia, 1 / sqrt(1 + shear (beta K / L)^2) with radius of gyration K (m) and shear = 1 + E / (k G) (5.9
    for a thin-walled tube, 3.9 for a solid rod), modes 4, 5, 6 of a 38 mm brass chime come out at 2 : 2.94 : 4.02
    and the ear hears the missing fundamental of that 2 : 3 : 4 (the strike tone). The length follows from f0, K and
    wave speed c. Bending in a round tube comes as a pair of equal modes (two planes); the seam and the hanger hole
    split them by `split`, hence the slow beat of a chime. rest: where the cord holds it (0.224 = at the node of
    mode 1; a tubular bell hangs from its top end, 0)"""
    beta = _beta(count)
    length = beta[mode - 1] * math.sqrt(gyration * c / (2 * np.pi * f0))
    bend = beta ** 2 / np.sqrt(1.0 + shear * (beta * gyration / length) ** 2)
    return modes(bend / bend[mode - 1], 0.5 * beam_shape(beta, strike), split=split, order=1,
                 held=(0.5 * beam_shape(beta, rest)) ** 2)


# name, ratio to the prime, order (meridian pattern cos(m theta)), level dB for a clapper on the soundbow, dB change
# for a strike up on the waist. Ratios: a tuned church bell as measured (Fletcher & Rossing ch. 21 after Perrin and
# Rossing): minor-third tierce, nominal an octave over the prime, upper partials stretched (octave nominal 2471
# cents over the prime, not 2400). The strike note is heard at the prime, set by nominal : superquint : octave
# nominal = 2 : 3.01 : 4.17. The quint has a node at the soundbow and is very quiet.
_CHURCH = (('hum', 0.500, 2, -7.0, 0.0), ('prime', 1.000, 2, -5.0, 3.0), ('tierce', 1.183, 3, -1.0, 2.0),
           ('quint', 1.506, 3, -18.0, 8.0), ('nominal', 2.000, 4, 0.0, -5.0), ('deciem', 2.514, 4, -19.0, 4.0),
           ('undeciem', 2.662, 2, -16.0, 4.0), ('superquint', 3.011, 5, -2.0, -7.0),
           ('octave nominal', 4.166, 6, -3.0, -9.0), ('eleventh', 5.433, 7, -4.0, -11.0),
           ('thirteenth', 6.796, 8, -6.0, -12.0), ('double octave', 8.215, 9, -8.0, -13.0),
           ('(10,1)', 9.70, 10, -10.0, -14.0), ('(11,1)', 11.25, 11, -12.0, -15.0))
# An English handbell: the (2,0) mode is the note and the (3,0) is tuned a twelfth above it; nothing else is tuned.
# (2,1) and (3,1), with a nodal circle, are weak. The higher (m,0) modes sit lower in small bells: the last column
# is the exponent of (f0 / 523 Hz) on the ratio. Approximate class values (Rossing & Sathoff; one measured small
# bell has 1, 1.79, 2.52, 2.99, 3.79, 4.57, 5.03).
_HAND = (('(2,0)', 1.00, 2, 0.0, 0.0, 0.0), ('(2,1)', 1.80, 2, -24.0, 6.0, 0.0), ('(3,0)', 3.00, 3, -8.0, -2.0, 0.0),
         ('(3,1)', 4.20, 3, -20.0, 6.0, -0.04), ('(4,0)', 5.50, 4, -6.0, -5.0, -0.06),
         ('(4,1)', 6.80, 4, -22.0, 6.0, -0.06), ('(5,0)', 8.40, 5, -9.0, -8.0, -0.08),
         ('(6,0)', 11.6, 6, -12.0, -10.0, -0.09), ('(7,0)', 15.1, 7, -15.0, -12.0, -0.10))


def bell_modes(kind='church', height=0.0, f0=523.25, split=0.0015):
    """The partials of a bell relative to its note. 'church': hum 0.5, prime 1, tierce 1.183, quint 1.506, nominal
    2, superquint 3.011, octave nominal 4.166 and the rim series above (see _CHURCH). 'hand': (2,0) = 1, (3,0) = 3,
    then untuned shell modes that depend a little on the size of the bell (f0). height: 0 = struck on the soundbow,
    as rung; +1 = up on the waist (less nominal and rim series, more prime, tierce and quint); -1 = on the lip.
    Every partial is a pair split by about `split` (no casting is round: 1.5 .. 9 cents measured)"""
    if kind == 'church':
        _, ratio, order, db, up = zip(*_CHURCH)
        ratio = np.array(ratio)
    else:
        _, ratio, order, db, up, small = zip(*_HAND)
        ratio = np.array(ratio) * (f0 / 523.25) ** np.array(small)
    gain = 10 ** ((np.array(db) + height * np.array(up)) / 20)
    return modes(ratio, gain, split=split, order=np.array(order))


_RIMS = {   # (m,0) rim modes m = 2, 3, 4 ...: means of three measured singing bowls (140 .. 180 mm; Inacio,
            # Henrique & Antunes 2006) and a measured wine glass (Rossing)
    'bowl': (1.0, 2.735, 5.047, 7.829, 11.03, 14.92, 18.1),
    'glass': (1.0, 2.32, 4.25, 6.63, 9.38, 12.5, 16.0),
}


def rim_modes(kind='bowl', height=0.0, split=0.004, fall=0.7):
    """A bowl or a glass struck near its rim: modes with m = 2, 3, 4 ... wavelengths around the rim. Ratios from
    measured bowls and glasses (_RIMS), close to those of a ring, m (m^2 - 1) / sqrt(m^2 + 1), but lower. All are
    pairs: a hammered bowl splits them by up to 1 % (the slow beat a singing bowl is known for), blown glass by far
    less. height: 0 = struck at the rim, +1 = well below it (the higher modes, which live at the rim, get less).
    fall: the gain goes as (m / 2)^-fall: a padded stick or a fingertip spreads the blow over the rim and favours
    the low modes (0.7); a hard point contact, glass on glass, drives all of them alike (0)"""
    ratio = np.array(_RIMS[kind])
    m = np.arange(2, 2 + len(ratio))
    return modes(ratio, (m / 2.0) ** -fall * (1.0 - 0.45 * max(height, -1.0)) ** (m - 2.0), split=split, order=m)


# (nodal diameters m, nodal circles n): ratio to (2,0) for a free circular plate, Poisson ratio 0.33 (Leissa;
# Fletcher & Rossing section 3.6). A measured 2 euro coin: 1 : 1.70 : 2.30.
_PLATE = ((2, 0, 1.000), (0, 1, 1.73), (3, 0, 2.328), (1, 1, 3.91), (4, 0, 4.11), (5, 0, 6.30), (2, 1, 6.71),
          (0, 2, 7.34), (6, 0, 8.91), (3, 1, 10.07), (1, 2, 11.40), (7, 0, 11.94), (4, 1, 13.9), (8, 0, 15.3))


def plate_modes(strike=1.0, centre_held=False, split=0.002):
    """A flat disc with a free edge (a coin, a crotale): (2,0) 1 : (0,1) 1.73 : (3,0) 2.33 : (1,1) 3.91 : (4,0) 4.11
    : (5,0) 6.30 ... strike: 1 = on the edge, where every mode moves; toward the centre (0) a mode with m nodal
    diameters fades as strike^m (shapes approximated by r^m cos(0.8 n pi (1 - r))). Modes with m > 0 are pairs.
    centre_held: the disc is bolted through its middle (crotales): modes that move the centre (m = 0 and 1) are
    damped away at once, which leaves the clear (2,0), (3,0), (4,0) series"""
    m, n, ratio = (np.array(v, np.float64) for v in zip(*_PLATE))
    gain = strike ** m * np.cos(0.8 * n * np.pi * (1.0 - strike)) / (1.0 + 0.5 * n)
    damp = np.where(m < 2, 40.0, 1.0) if centre_held else 1.0
    return modes(ratio, gain, damp=damp, split=np.where(m > 0, split, 0.0), order=m)


def doublets(split, order, strike_angle, built, listen_angle=None):
    """Turn the modes of a round body into their beating pairs. A mode of `order` m exists twice, as cos(m theta)
    and sin(m theta) patterns about an axis the casting fixed (drawn from `built`, the generator of the instrument,
    not of the take), a few tenths of a percent apart. A strike at strike_angle drives them as cos and sin of
    m * (angle - axis), and a listener at listen_angle (default: where the strike is) hears them with the same
    weights again, so each partial beats at its own rate and depth and the pattern changes when the strike moves.

    Returns (index, detune, weight, side): for every output partial the mode it comes from, its frequency factor,
    its amplitude factor and -1 / +1 for the two members (0 for a single mode)"""
    split, order = np.asarray(split, np.float64), np.asarray(order)
    axis = built.uniform(0.0, np.pi, len(split))
    amount = split * built.uniform(0.4, 1.6, len(split))
    listen = strike_angle if listen_angle is None else listen_angle
    pair = np.nonzero((split > 0) & (order > 0))[0]
    single = np.nonzero(~((split > 0) & (order > 0)))[0]
    a, b = order[pair] * (strike_angle - axis[pair]), order[pair] * (listen - axis[pair])
    index = np.concatenate([single, pair, pair])
    detune = np.concatenate([np.ones(len(single)), 1.0 - 0.5 * amount[pair], 1.0 + 0.5 * amount[pair]])
    weight = np.concatenate([np.ones(len(single)), np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)])
    side = np.concatenate([np.zeros(len(single)), -np.ones(len(pair)), np.ones(len(pair))])
    return index, detune, weight, side


# --------------------------------------------------------------------------------------- radiation and resonators
def radiation(f, knee, order=1.0):
    """Relative radiation efficiency of a small vibrating body: it rises as f^order while the body is small against
    the wavelength (the pressure cancels around its edges) and levels off above `knee` Hz, where its circumference
    is about a wavelength. High modes radiate better; this is the tilt that makes a struck bar bright at first"""
    x = (np.asarray(f, np.float64) / knee) ** order
    return x / np.sqrt(1.0 + x * x)


def air_absorption(f, metres):
    """gain of air absorption over a path (20 C, 50 % humidity: 0.10 dB/m at 8 kHz, growing as f^2)"""
    return 10 ** (-1.6e-9 * np.asarray(f, np.float64) ** 2 * metres / 20)


def resonator(n, f, rate, q=30.0, boost_db=8.0, load=1.0, detune=0.0, motor=0.0, depth=0.85, phase=0.0):
    """What a tuned tube (or box) under a bar does to the fundamental of the bar: a complex gain per sample, to be
    multiplied on the analytic signal of that mode from ring().

    The tube is a second, heavily damped resonator (quality q) driven by the bar. It needs q / (pi f) seconds to
    fill, so its sound blooms a few milliseconds after the strike; once full it radiates far better than the bar
    (boost_db louder at the tuned pitch, a quarter cycle behind) and so drains the bar: the decay rate of the
    fundamental grows by `load` times its own rate (a tuned marimba resonator about halves the ring time). detune:
    tube frequency off the bar (fraction). motor: the discs of a vibraphone turning in the tube mouths open and
    close the tube `motor` times per second (0 = off), so boost, drain and a little phase pulse; depth = how far
    they close it"""
    width = np.pi * f * (1.0 + detune) / q
    opening = np.ones(n)
    if motor > 0:
        opening = 1.0 - depth * np.abs(np.cos(np.pi * motor * dsp.t_of(n) + phase))
    pole = np.exp(-(width + 2j * np.pi * f * detune) / SR)
    tube = signal.lfilter([1.0 - abs(pole)], [1.0, -pole], opening * np.exp(0.6j * (1.0 - opening)))
    drain = np.exp(-np.cumsum(np.abs(tube) ** 2) * (load * rate / SR))
    return drain * (1.0 + 1j * math.sqrt(max(10 ** (boost_db / 10) - 1.0, 0.0)) * tube)


def body_response(f, body, direct=0.2):
    """Complex gain of a soundboard or box at f Hz: `direct` plus one band-pass resonance per (frequency Hz, q,
    gain) in `body`. A tine or a comb radiates almost nothing itself; the board it sits on does, and colours every
    partial with its own few broad modes"""
    fk, qk, gk = (np.asarray(v, np.float64) for v in zip(*body))
    x = np.atleast_1d(np.asarray(f, np.float64))[:, None] / fk
    return direct + (gk * (1j * x / qk) / (1.0 - x * x + 1j * x / qk)).sum(axis=1)


# ---------------------------------------------------------------------------------------------- a close stereo pair
def mic_pair(amp, azimuth, freq, spread=0.2, lag=2.5e-4, phase=0.6):
    """Complex amplitudes (2, modes) for a close pair of microphones. The modes of a body radiate from different
    parts of it and in different directions, so each has its own place in the picture (azimuth -1 .. 1 within the
    source): the pair hears it with a level difference of +-spread and a time difference of up to +-lag seconds.
    The time difference is capped where it would turn the two channels more than `phase` radians apart, so every
    partial keeps cos(phase / 2) of its level in the mono sum (0.4 dB down at 0.6 rad) and none can cancel"""
    az = np.clip(np.asarray(azimuth, np.float64), -1.0, 1.0)
    turn = np.exp(1j * az * np.minimum(np.pi * np.asarray(freq, np.float64) * lag, 0.5 * phase))
    return np.stack([amp * (1.0 - spread * az) * turn, amp * (1.0 + spread * az) / turn])


def early(y, built, level=0.2, spread=1.0):
    """Add the very early reflections a close pair picks up from the frame of the instrument, the floor or the
    table: two that both microphones share and two per channel (0.8 .. 7 ms, each -18 dB or lower, the highs
    absorbed). They are what makes the two channels differ a little in phase. The sum of all gains stays under 0.4,
    so no partial can cancel in mono. y stereo (n, 2); built: the generator of the instrument"""
    mid = dsp.butter(y.mean(axis=1), 'low', 5000.0, 1)
    out = y.copy()
    for ch in (None, 0, 1):
        for lo, hi, gain in ((0.0008, 0.003, 0.5), (0.003, 0.007, 0.35)):
            d = int(built.uniform(lo, hi) * SR)
            g = level * gain * built.uniform(0.7, 1.0) * (1.0 if ch is None else 0.6 * min(spread, 2.0))
            if d >= len(y):
                continue
            if ch is None:
                out[d:] += g * mid[:len(y) - d, None]
            else:
                out[d:, ch] += g * mid[:len(y) - d]
    return out
