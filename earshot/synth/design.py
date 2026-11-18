"""Sound design on recordings: the few operations a film sound designer uses to build booms, risers, passes and
braams out of real material, applied to takes of learned models (earshot.synth.resyn). Nothing here makes a sound:
no oscillator, no noise source, no filter sweep. Everything heard is a take of a recording, reshaped.

    take     a take of a model inside the range where the engine stays natural (size 0.7 .. 1.4, length 0.6 .. 1.8,
             pitch at most 7 semitones): a bigger drum, a slower whoosh, a brass note moved to a chord tone
    stretch  a take of air (a whoosh, a breath, a flame) slowed up to 3 times: a long build from a short recording
    opening  a take growing brighter from start to end, the top arriving last (a build)
    reverse  a decay played backwards: a reversed cymbal or gong swell IS the classic riser, a reversed whoosh the
             suck-in before a hit; it ends where the strike was, on its loudest moment
    rise     the build a take holds (up to its loudest moment, or its decay reversed); fall: the decay it holds
    layer    several takes aligned at their main onsets, with gains and offsets: the transient of one, the body of
             another, the tail of a third, as one hit
    move     a pass: the take carried from one side to the other through the centre at its loudest moment, its level
             rising and falling with the distance and its pitch falling a little as it goes by (the shape of a
             Doppler shift, on the recording itself)
    fade     fade shapes: equal power, linear, or straight in dB (a decay's own shape)
    span     a part of a take between two times (a whoosh up to its peak: something arriving), faded at the cuts
and the room is earshot.synth.space, shared with every other sound.

    from earshot.synth import design as D, resyn
    hit = D.take(drum_model, r, size=1.3, ring=1.4)
    rise = D.reverse(D.take(cymbal_model, r))
    y, t_hit = D.layer([(rise, -6.0, 0.0, D.end(rise)), (hit, 0.0, 0.0)])
"""
from fractions import Fraction

import numpy as np
from scipy import signal

from ..forge import dsp, fx
from . import resyn

SR = dsp.SR
SIZE = (0.7, 1.4)           # resyn size and length factors inside which a take stays a natural performance
LENGTH = (0.6, 1.8)         # (earshot.sounds.learned: measured on the bands' envelope modulation)
PITCH = 7.0                 # semitones a take is moved at most
RING = (0.5, 2.0)
AIR = 3.0                   # how far stretch slows a take of air


def take(model, r, size=1.0, length=1.0, pitch=0.0, bright=0.0, hard=0.0, ring=1.0, weight=0.0, vary=0.0):
    """a take of a model (resyn.render, stereo (n, 2)) with its transforms held inside the natural ranges (SIZE,
    LENGTH, PITCH, RING); r: the generator (the noise of every take is drawn fresh). vary: small random amounts of
    length, pitch, tilt and hardness, as resyn draws them. Size and pitch are done as resyn does them (the take
    rendered at its own pitch, time-warped against the change, then resampled), the resampling polyphase: resyn
    resamples by an FFT of the take's own length, which for a long take of an awkward length costs seconds"""
    if vary > 0:
        length *= float(np.exp(0.05 * vary * r.standard_normal()))
        pitch += 0.3 * vary * float(r.standard_normal())
        bright += 0.4 * vary * float(r.standard_normal())
        hard += 0.3 * vary * float(r.standard_normal())
    size = float(np.clip(size, *SIZE))
    pitch = float(np.clip(pitch, -PITCH, PITCH))
    q = 2.0 ** (pitch / 12.0) / size
    y = resyn.render(model, r, length=float(np.clip(length, *LENGTH)) * 2.0 ** (pitch / 12.0), bright=bright,
                     hard=float(np.clip(hard, -1.0, 1.0)), ring=float(np.clip(ring, *RING)), weight=weight)
    if abs(q - 1.0) > 1e-6:
        ratio = Fraction(1.0 / q).limit_denominator(96)       # within a fraction of a cent of the size asked for
        y = signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=0)
        y = dsp.fade(y, 0.0001, 0.004)
        peak = np.abs(y).max()
        if peak > 0.98:
            y *= 0.98 / peak
    return y


def stretch(model, r, factor, **kw):
    """a take of a model of air (a whoosh, a breath, a burst of flame: noise in bands, no pitch, no strokes) slowed
    `factor` times by resynthesis, up to AIR times: its band envelopes are drawn out and the noise in them drawn
    fresh, so a long build stays air where a struck or tonal thing slowed that far would smear. Other arguments as
    take, inside its ranges"""
    kw = {k: v for k, v in kw.items() if k != 'length'}
    if factor <= LENGTH[1]:
        return take(model, r, length=max(factor, LENGTH[0]), **kw)
    y = resyn.render(model, r, length=float(min(factor, AIR)), bright=kw.get('bright', 0.0),
                     hard=float(np.clip(kw.get('hard', 0.0), -1.0, 1.0)))
    peak = np.abs(y).max()
    return y * (0.98 / peak) if peak > 0.98 else y


def _env(y, ms):
    k = max(1, dsp.n_of(ms / 1000.0))
    p = np.mean(np.atleast_2d(np.asarray(y, np.float64).T).T ** 2, axis=1)
    return np.sqrt(np.convolve(p, np.ones(k) / k, 'same'))


def onset(y, drop=12.0):
    """seconds to a take's main strike: the first moment its 2 ms level comes within `drop` dB of its loudest (the
    time two hits are lined up at)"""
    e = _env(y, 2.0)
    return float(np.argmax(e >= e.max() * 10 ** (-drop / 20)) / SR)


def peak(y, ms=50.0):
    """seconds to a take's loudest moment (its level over `ms` milliseconds): where a swell culminates, where a
    pass is closest"""
    return float(np.argmax(_env(y, ms)) / SR)


def end(y):
    """seconds to the end of a take: where a reversed take culminates, lined up with the hit that follows it"""
    return len(y) / SR


def reverse(y, close=0.006):
    """y played backwards (stereo): what decayed now swells and ends on what was the strike. The last `close` s are
    faded over a quarter cosine, so the cut is sharp but not a step"""
    z = np.array(np.asarray(y, np.float64)[::-1])
    z = z if z.ndim == 2 else np.stack([z, z], axis=1)
    n = min(len(z), dsp.n_of(close))
    z[len(z) - n:] *= np.cos(np.linspace(0.0, np.pi / 2, n))[:, None] ** 0.5
    k = min(len(z), dsp.n_of(0.002))
    z[:k] *= np.linspace(0.0, 1.0, k)[:, None]
    return z


def rise(y, ms=100.0):
    """the build a take holds: when its loudest moment comes in its second half, the take up to that moment (cut 30
    ms after it), else its decay played backwards (a hiss that dies, a whoosh's tail: reversed, they grow); either
    way it ends on its loudest moment"""
    p = peak(y, ms)
    if p >= 0.5 * end(y):
        return span(y, 0.0, p + 0.03, fout=0.03)
    return reverse(span(y, max(0.0, p - 0.005)))


def fall(y, ms=100.0):
    """the decay a take holds: from its loudest moment on when that comes in its first half, else its build played
    backwards; either way it starts full and dies away"""
    p = peak(y, ms)
    if p <= 0.5 * end(y):
        return span(y, max(0.0, p - 0.02), fin=0.02)
    return fade(reverse(span(y, 0.0, p + 0.005)), 0.01, 0.0)


def layer(parts):
    """takes mixed as one sound, each lined up at a common moment: parts are (y, gain dB, offset s) or (y, gain dB,
    offset s, at s) where `at` is the time in y that is lined up (default onset(y)) and the offset moves it after
    (or, negative, before) the common moment. Returns (stereo mix, the common moment in s from the start)"""
    rows = []
    for p in parts:
        y = np.asarray(p[0], np.float64)
        y = y if y.ndim == 2 else np.stack([y, y], axis=1)
        rows.append((y, p[1], p[2], onset(y) if len(p) < 4 or p[3] is None else p[3]))
    hit = max(max(at - off, 0.0) for _, _, off, at in rows)
    starts = [dsp.n_of(hit + off - at) for _, _, off, at in rows]
    n = max(s + len(y) for (y, _, _, _), s in zip(rows, starts))
    out = np.zeros((n, 2))
    for (y, g, _, _), s in zip(rows, starts):
        a = max(s, 0)
        out[a:s + len(y)] += y[a - s:] * 10 ** (g / 20)
    return out, hit


def move(y, r, travel=1.0, fall=1.0, near=0.35, side=None):
    """a pass: y carried past the listener, closest at its loudest moment (peak). The pan runs from one side
    (`side`: -1 left, +1 right, else drawn from r) `travel` of the way out, through the centre, to the other; the
    level follows the distance (1 / distance, the closest point `near` of the take's length from either end: 0 dB there); the
    pitch falls `fall` semitones in all, fastest at the closest point, as a Doppler shift does (the take is read at
    a speed that falls from +fall/2 to -fall/2 semitones). Stereo in, stereo out; the side image is kept"""
    y = np.asarray(y, np.float64)
    y = y if y.ndim == 2 else np.stack([y, y], axis=1)
    n = len(y)
    tc = peak(y, 100.0) * SR
    s = float(side) if side is not None else float(r.choice([-1.0, 1.0]))
    u = (np.arange(n) - tc) / (near * n)
    speed = 2.0 ** (-0.5 * fall * np.tanh(2.0 * u) / 12.0)
    pos = np.concatenate([[0.0], np.cumsum(speed[:-1])])
    pos = pos[pos < n - 1]
    z = fx._hermite(y, pos)
    m = len(z)
    u = (pos - tc) / (near * n)
    gain = 1.0 / np.sqrt(1.0 + u * u)
    th = (-s * float(np.clip(travel, 0.0, 1.0)) * np.tanh(u) + 1.0) * np.pi / 4      # from side s to -s
    mid, side_ = 0.5 * (z[:, 0] + z[:, 1]), 0.5 * (z[:, 0] - z[:, 1])
    out = np.stack([mid * np.cos(th), mid * np.sin(th)], axis=1) * np.sqrt(2) + np.stack([side_, -side_], axis=1)
    out *= gain[:, None]
    return dsp.fade(out, 0.002, min(0.02, 0.1 * m / SR))


def halt(y, seconds, at=None, curve=1.6):
    """y running down to a stop, as a tape or a turntable does when its motor is cut: from `at` s (default: so the
    stop ends with y's end, else its start) the take is read at a speed falling from 1 to 0 over `seconds` s
    (1 - u ** (1 / curve): fast at first, the last slow turns long), so pitch and tempo sink together through the
    recording itself; what follows the stop is silence. Stereo in, stereo out"""
    y = np.asarray(y, np.float64)
    y = y if y.ndim == 2 else np.stack([y, y], axis=1)
    n, k = len(y), dsp.n_of(seconds)
    a = dsp.n_of(at) if at is not None else max(0, n - int(k / (curve + 1.0)) - 2)   # the stop reads k / (curve + 1)
    u = np.linspace(0.0, 1.0, k)
    speed = np.clip(1.0 - u ** (1.0 / curve), 0.0, 1.0)
    pos = a + np.concatenate([[0.0], np.cumsum(speed[:-1])])
    pos = pos[pos < n - 1]
    z = np.concatenate([y[:a], fx._hermite(y, pos)])
    return z


def fade(y, fin=0.0, fout=0.0, shape='power'):
    """y with a fade in over `fin` s and a fade out over `fout` s: shape 'power' (equal power, a quarter sine),
    'lin', or 'db' (straight in dB, from 60 dB down: the shape of a natural decay)"""
    y = np.array(np.asarray(y, np.float64))

    def curve(k):
        u = np.linspace(0.0, 1.0, k)
        if shape == 'lin':
            return u
        if shape == 'db':
            return np.where(u > 0, 10 ** (-60.0 * (1.0 - u) / 20), 0.0)
        return np.sin(0.5 * np.pi * u)
    for sec, head in ((fin, True), (fout, False)):
        k = min(len(y), dsp.n_of(sec))
        if k > 1:
            c = curve(k)
            c = c if y.ndim == 1 else c[:, None]
            if head:
                y[:k] *= c
            else:
                y[len(y) - k:] *= c[::-1]
    return y


def span(y, a=0.0, b=None, fin=0.004, fout=0.03, shape='power'):
    """the part of y from a s to b s (None: to its end), faded at both cuts (fade)"""
    y = np.asarray(y, np.float64)
    i, j = dsp.n_of(max(a, 0.0)), len(y) if b is None else min(len(y), dsp.n_of(b))
    return fade(y[i:j], fin if i > 0 else 0.0, fout if j < len(y) else 0.0, shape)


def opening(y, hz=500.0, power=1.5):
    """y growing brighter from its start to its end, as a build does: what lies over `hz` Hz (a second-order split,
    the two parts summing back to y) comes in along (t / T) ** power while what lies under it is heard throughout,
    so the top arrives last and the take ends on all of itself (a reversed decay already brightens this way; this
    makes it hold for a take that was not recorded as a decay)"""
    y = np.asarray(y, np.float64)
    lo = dsp.butter(y, 'low', hz, order=2)
    u = np.linspace(0.0, 1.0, len(y)) ** power
    return lo + (y - lo) * (u if y.ndim == 1 else u[:, None])


def contour(y, start_db=-24.0, ms=60.0, floor_db=-50.0):
    """y with its level redrawn as a build is drawn on a fader: its own envelope (over `ms` milliseconds) divided
    out and a line straight in dB laid on, from `start_db` at the start to 0 dB at the end, so a reversed decay that
    spends its first second 40 dB down is heard from its start and grows evenly; the grain inside the envelope is
    kept. Moments more than `floor_db` under its loudest are not lifted further than that"""
    y = np.asarray(y, np.float64)
    e = _env(y, ms)
    e = np.maximum(e, e.max() * 10 ** (floor_db / 20))
    line = 10 ** (np.linspace(start_db, 0.0, len(y)) / 20) / e * e.max()
    z = y * (line if y.ndim == 1 else line[:, None])
    return z * (np.abs(y).max() / (np.abs(z).max() + 1e-20))


def pad(y, seconds):
    """y with `seconds` of silence before it (stereo)"""
    y = np.asarray(y, np.float64)
    return np.concatenate([np.zeros((dsp.n_of(seconds),) + y.shape[1:]), y])
