"""The patch library of the earshot synthesizer: finished synth sounds behind one call.

    from earshot.forge import dsp
    from earshot.synth import patches
    y = patches.play('soft_pluck', 'C5', vel=0.8, dur=0.3)                      # stereo (n, 2) with release and room
    y = patches.play('warm_pad', ['C4', 'E4', 'G4'], dur=2.0, bright=0.3)       # a chord: one voice per note
    y = patches.play('braam', 'D2', dur=0.6, hard=-1, r=dsp.rng(4))             # the swell variant, another take
    patches.PATCHES['fm_bell']                                                   # says, kind, default_pitch, range

Every patch is built on earshot.synth.voice the way a preset designer builds it on a polysynth: detuned drifting
oscillators or FM operators whose index dies before the note does, a filter that follows velocity and the keyboard,
exponential envelopes, then the effects that belong to the sound (ensemble chorus, a damped echo, a room from
earshot.synth.space). A note is held for dur s and then released; the result includes the tail.

Controls are numbers in [-1, 1], 0 = the patch as designed: bright (cutoff / FM index), hard (attack sharpness),
ring (decay and release length), size (sub, body), motion (LFO / chorus / detune), width, dirt (drive), space (the
patch's echo and room; -1 = none); vary in [0, 1] moves a take a little away from the design (never its pitch).
Level is not normalised: velocity 1 peaks near 0.9, softer notes are quieter and darker.
"""
import numpy as np

from ..forge import dsp, fx, knobs
from ..forge.dsp import SR, n_of
from . import space, voice

CONTROLS = ('bright', 'hard', 'ring', 'size', 'motion', 'width', 'dirt', 'space')
_MAJOR = (0, 4, 7, 12, 16, 19, 24)


def _x(c, key, octaves=1.0):
    """a control as a ratio: 2^(octaves * value)"""
    return 2.0 ** (octaves * c[key])


def _reg(f, slope):
    """a time that shrinks up the keyboard, (f / C4)^-slope: the high notes of every instrument die sooner"""
    return (f / voice.C4) ** -slope


def _st(y):
    return np.stack([y, y], axis=1)


def _mix(takes):
    """[(t s, stereo)] summed on one timeline"""
    out = np.zeros((max(n_of(t) + len(y) for t, y in takes), 2))
    for t, y in takes:
        dsp.place(out, y, t)
    return out


# ------------------------------------------------------------------------------------------------ plucks and keys
def _soft_pluck(f, vel, dur, r, c):
    """a triangle pair a few cents apart with a saw under a ladder that opens 3 octaves for the first 50 ms and
    settles a little above the fundamental, so the second to fourth harmonics stay (-8 to -20 dB: a round tone
    that is not a bare sine); a sine an octave down gives it body in the low mids, a breath of noise through the
    open filter marks the attack, and the note decays over about 1.5 s and releases over 0.6 s. A darker version
    sounds dull and technical next to bell timbres; this keeps it soft (no
    inharmonic partials, nothing above 4 kHz but the breath) and makes it fuller."""
    g, h = _x(c, 'ring', 1.3) * _reg(f, 0.25), _x(c, 'hard', -2.0)
    return voice.analog(
        f, vel, dur, r, drift=1.5, slew=0.0012,
        osc=(dict(shape='tri', voices=2, detune=4.5 * (1 + c['motion']), spread=0.45, level=0.9, phase=0.25),
             dict(shape='saw', level=0.6, phase=0.5)),
        sub=0.42 * _x(c, 'size', 1.5), noise=0.05 * _x(c, 'bright', 1.0), noise_width=0.6,
        cutoff=900 * _x(c, 'bright', 1.3), keytrack=0.9, res=0.08, env=2.3, vel_cut=1.5,
        fenv=(0.001, 0.3 * g, 0.0, 0.25 * g), fcurve=(2.0, 6.0, 5.0), amp=(0.003 * h, 1.5 * g, 0.0, 0.6 * g))


def _glass_pluck(f, vel, dur, r, c):
    """two FM pairs: 1:3 for the body and an inharmonic 2:5.03 pair that lasts 60 ms, the glass in the attack"""
    g, h, b = _x(c, 'ring', 1.3) * _reg(f, 0.3), _x(c, 'hard', -2.0), _x(c, 'bright', 0.9)
    d = 1 + c['motion']
    return voice.fm(f, [
        dict(cents=-2 * d, out=1.0, pan=-0.3, vel=0.5, mod=(1,), env=(0.001 * h, 0.7 * g, 0.0, 0.3 * g)),
        dict(ratio=3.0, index=1.9 * b, vel=0.8, key=-0.3, env=(0.001, 0.16 * g, 0.04, 0.12)),
        dict(ratio=2.0, cents=2 * d, out=0.4, pan=0.3, vel=0.5, mod=(3,), env=(0.001 * h, 0.35 * g, 0.0, 0.2 * g)),
        dict(ratio=5.03, index=1.5 * b, vel=0.8, key=-0.4, env=(0.001, 0.06 * g, 0.0, 0.05))], vel, dur)


def _muted_pluck(f, vel, dur, r, c):
    """saw and narrow pulse under a ladder that shuts within 70 ms: the thump and short body of a palm mute"""
    g, h = _x(c, 'ring', 1.3) * _reg(f, 0.2), _x(c, 'hard', -1.5)
    return voice.analog(
        f, vel, dur, r, osc=(dict(shape='saw', voices=2, detune=3, spread=0.3, level=0.8, phase=0.5),
                             dict(shape='pulse', pw=0.3, level=0.5, cents=4, phase=0.0)),
        sub=0.3 * _x(c, 'size', 1.5), cutoff=300 * _x(c, 'bright', 1.3), keytrack=0.7, res=0.18, env=3.0,
        vel_cut=2.0, drive=4.0 + 4.0 * min(c['dirt'], 0.0), fenv=(0.001, 0.07 * g, 0.0, 0.05),
        amp=(0.002 * h, 0.3 * g, 0.0, 0.07 * g))


def _bubble(f, vel, dur, r, c):
    """a sine with a little triangle that starts 7 semitones low and rises into its pitch within 40 ms (a bubble's
    resonance rises as it closes), 4 ms attack, gone in 200 ms"""
    g, h = _x(c, 'ring', 1.3), _x(c, 'hard', -1.5)
    return voice.analog(
        f, vel, dur, r, drift=0.0, slew=0.002,
        osc=(dict(shape='sine', phase=0.0), dict(shape='tri', phase=0.0, level=0.35 * _x(c, 'bright', 1.0))),
        bend=(-7.0 * _x(c, 'size', 0.6), 0.018 * (1 + 0.6 * c['motion'])), filter='ladder2',
        cutoff=700 * _x(c, 'bright', 1.3), keytrack=0.8, res=0.1, env=1.8, vel_cut=1.5,
        fenv=(0.001, 0.08 * g, 0.0, 0.05),
        amp=(0.004 * h, 0.2 * g, 0.0, 0.1 * g))


def _warm_keys(f, vel, dur, r, c):
    """saw pair and a pulse whose width drifts, a sub, a ladder that opens with velocity and settles: poly keys"""
    g, h, m = _x(c, 'ring', 1.3) * _reg(f, 0.2), _x(c, 'hard', -2.0), 1 + c['motion']
    return voice.analog(
        f, vel, dur, r, drift=3.0, smear=0.002,
        osc=(dict(shape='saw', voices=2, detune=6 * m, spread=0.5, level=0.7),
             dict(shape='pulse', pw=0.42, pwm=0.08 * m, pwm_rate=0.6, cents=-4, level=0.6)),
        sub=0.25 * _x(c, 'size', 1.5), cutoff=800 * _x(c, 'bright', 1.3), keytrack=0.45, res=0.12, env=1.9,
        vel_cut=1.8, fenv=(0.004 * h, 0.55 * g, 0.3, 0.4 * g), amp=(0.005 * h, 1.5 * g, 0.5, 0.4 * g))


def _fm_keys(f, vel, dur, r, c):
    """three FM pairs a few cents apart, as the digital electric pianos are built: a 1:1 body whose index follows
    velocity (the bark), a 1:14 tine that rings for 70 ms, and a pair with feedback for the low growl; every
    index is gone within a second, so the chord settles into nearly pure, slowly beating sines"""
    g, h, b = _x(c, 'ring', 1.2), _x(c, 'hard', -1.5), _x(c, 'bright', 0.9)
    d = 1 + c['motion']
    long = (0.002 * h, 6.0 * g, 0.0, 0.2 * g)
    return voice.fm(f, [
        dict(cents=-3 * d, out=1.0, pan=-0.25, vel=0.5, rate=0.4, mod=(1,), env=long),
        dict(index=1.3 * b, vel=0.85, key=-0.25, rate=0.4, env=(0.001, 1.5 * g, 0.06, 0.2)),
        dict(cents=4 * d, out=0.4, pan=0.25, vel=0.6, rate=0.4, mod=(3,), env=(0.001 * h, 2.5 * g, 0.0, 0.2 * g)),
        dict(ratio=14.0, index=1.0 * b, vel=0.9, key=-0.45, env=(0.001, 0.35, 0.0, 0.1)),
        dict(cents=1 * d, out=0.7, vel=0.5, rate=0.4, mod=(5,), env=long),
        dict(index=0.75 * b * _x(c, 'size', 0.5), fb=0.5, vel=0.9, key=-0.25, rate=0.4,
             env=(0.001, 1.0 * g, 0.05, 0.2))], vel, dur)


def _fm_bell(f, vel, dur, r, c):
    """two carriers 2.4 cents apart, one at half the level (the slow shallow beat of a bell's doublets) under
    inharmonic modulators, 1:3.5 and 1:1.4, whose index is gone long before the carriers are: struck metal at first,
    two pure sines in the tail"""
    g, h, b = _x(c, 'ring', 1.2) * _reg(f, 0.35), _x(c, 'hard', -1.5), _x(c, 'bright', 0.9)
    d = 1 + c['motion']
    return voice.fm(f, [
        dict(cents=-1.2 * d, out=1.0, pan=-0.2, vel=0.5, mod=(1,), env=(0.001 * h, 7.0 * g, 0.0, 3.5 * g)),
        dict(ratio=3.5, index=1.6 * b, vel=0.7, key=-0.3, env=(0.001, 2.2 * g, 0.0, 1.0 * g)),
        dict(cents=1.2 * d, out=0.45, pan=0.5, vel=0.5, mod=(3,), env=(0.001 * h, 6.0 * g, 0.0, 3.2 * g)),
        dict(ratio=1.4, index=1.2 * b, vel=0.7, env=(0.001, 1.2 * g, 0.0, 0.6 * g)),
        dict(ratio=5.4 * _x(c, 'size', -0.3), out=0.22, vel=0.8, mod=(5,), env=(0.001 * h, 0.4 * g, 0.0, 0.3 * g)),
        dict(ratio=7.1, index=1.4 * b, key=-0.4, env=(0.001, 0.12, 0.0, 0.1))], vel, dur)


def _mallet_synth(f, vel, dur, r, c):
    """a sine carrier knocked by a modulator at 4 times its pitch for 70 ms, with the overtones of a tuned bar (4 and
    10 times the pitch) as partials of their own that die in 0.3 and 0.1 s, a soft octave below, and the mallet's
    thock"""
    g, h, b = _x(c, 'ring', 1.3) * _reg(f, 0.4), _x(c, 'hard', -1.5), _x(c, 'bright', 0.9)
    y = voice.fm(f, [
        dict(out=1.0, vel=0.5, mod=(1,), env=(0.001 * h, 1.4 * g, 0.0, 0.5 * g)),
        dict(ratio=3.98, index=1.1 * b, vel=0.8, key=-0.3, env=(0.0005, 0.07, 0.0, 0.05)),
        dict(ratio=3.98, out=0.2 * b, pan=-0.3, vel=0.8, key=-0.3, env=(0.001 * h, 0.3 * g, 0.0, 0.2 * g)),
        dict(ratio=9.9, out=0.07 * b, pan=0.3, vel=0.9, key=-0.5, env=(0.001 * h, 0.1 * g, 0.0, 0.08 * g)),
        dict(ratio=0.5, out=0.25 * _x(c, 'size', 1.5), vel=0.5, env=(0.002, 0.7 * g, 0.0, 0.4 * g))], vel, dur)
    k = voice.tick(r, 0.012, 700.0, 3500.0, 0.003)
    y[:len(k)] += (0.12 * vel * _x(c, 'hard', 1.0) * k)[:, None]
    return y


def _square_lead(f, vel, dur, r, c):
    g, h, m = _x(c, 'ring', 1.3), _x(c, 'hard', -2.0), 1 + c['motion']
    return voice.analog(
        f, vel, dur, r, drift=2.5, vib=(14 * m, 5.6, 0.28, 0.45),
        osc=(dict(shape='pulse', pwm=0.06 * m, pwm_rate=0.9, voices=2, detune=4, spread=0.25, level=0.9),
             dict(shape='tri', semi=-12, level=0.35 * _x(c, 'size', 1.5))),
        cutoff=1500 * _x(c, 'bright', 1.3), keytrack=0.6, res=0.12, env=1.0, vel_cut=1.2,
        fenv=(0.01 * h, 0.4, 0.5, 0.2), amp=(0.012 * h, 0.4, 0.85, 0.16 * g))


def _saw_lead(f, vel, dur, r, c):
    g, h, m = _x(c, 'ring', 1.3), _x(c, 'hard', -2.0), 1 + c['motion']
    return voice.analog(
        f, vel, dur, r, drift=3.0, smear=0.003, vib=(8 * m, 5.4, 0.4, 0.5),
        osc=(dict(shape='saw', voices=7, detune=16 * m, spread=0.7, blend=0.55),
             dict(shape='saw', semi=-12, level=0.35 * _x(c, 'size', 1.5))),
        cutoff=2600 * _x(c, 'bright', 1.3), keytrack=0.5, res=0.15, env=1.2, vel_cut=1.2,
        fenv=(0.004 * h, 0.5, 0.4, 0.25), amp=(0.006 * h, 0.5, 0.85, 0.22 * g))


# ------------------------------------------------------------------------------------------------ pads and beds
def _warm_pad(f, vel, dur, r, c):
    """five drifting saws and a pulse with slow PWM under a ladder that opens over a second and never sits still"""
    g, h, m = _x(c, 'ring', 1.3), _x(c, 'hard', -2.0), 1 + c['motion']
    return voice.analog(
        f, vel, dur, r, drift=3.5, smear=0.012, curve=(1.0, 5.0, 5.0), slew=0.004,
        osc=(dict(shape='saw', voices=5, detune=13 * m, spread=0.8, blend=0.6, level=0.8),
             dict(shape='pulse', pwm=0.2 * min(m, 1.7), pwm_rate=0.31, voices=2, detune=5, spread=0.3, level=0.5)),
        sub=0.3 * _x(c, 'size', 1.5), cutoff=650 * _x(c, 'bright', 1.3), keytrack=0.4, res=0.1, env=1.4,
        vel_cut=1.2, cut_lfo=(0.3 * m, 0.13, 'random'), fenv=(0.9 * h, 2.5, 0.55, 2.0 * g),
        amp=(0.55 * h, 1.8, 0.85, 1.9 * g))


def _air_pad(f, vel, dur, r, c):
    """breath: noise through narrow resonances on the first three harmonics (each a tone with the flutter of blown
    air), a quiet sine for the pitch, upper partials that come and go, and hiss above 5 kHz"""
    g, h, m = _x(c, 'ring', 1.2), _x(c, 'hard', -2.0), 1 + c['motion']
    b = _x(c, 'bright', 1.0) * (0.4 + 0.6 * vel)               # a harder breath is a brighter one
    amp = voice.adsr(dur, 0.7 * h, 2.0, 0.85, 2.2 * g, curve=(0.5, 5.0, 5.0), slew=0.004)
    n = len(amp)
    t = np.arange(n) / SR
    ns = voice.noise2(n, r, 0.45)
    y = np.zeros((n, 2))
    for k, a in ((1, 1.0), (2, 0.5 * b), (3, 0.28 * b)):
        fk = f * k * 2.0 ** (4.0 * m * voice.wobble(n, 0.4, r) / 1200)
        for ch in (0, 1):
            band = dsp.svf(ns[:, ch], fk, 45.0, 'bandn')
            y[:, ch] += a * band / (band.std() + 1e-12)
    y = dsp.butter(y, 'band', (0.7 * f, min(5.0 * f, 16000.0)), order=2)     # no rumble or hiss from the bands' skirts
    y += 0.7 * _x(c, 'size', 1.0) * voice.sine(f * 2.0 ** (2.0 * voice.wobble(n, 0.3, r) / 1200), r.uniform())[:, None]
    for i, k in enumerate((4, 6, 9)):
        a = 0.1 * b / (i + 1) * (1.0 + np.tanh(voice.wobble(n, 0.2 + 0.15 * m + 0.1 * i, r)))
        p = a * np.sin(2 * np.pi * k * f * t + r.uniform(0, 2 * np.pi)) * (k * f < 0.45 * SR)
        y += p[:, None] * (np.array([1.2, 0.6]) if i % 2 else np.array([0.6, 1.2]))
    hiss = dsp.butter(voice.noise2(n, r, 0.8), 'high', 5000.0, order=1)
    y += 0.035 * b * hiss * (1.0 + 0.5 * np.tanh(voice.wobble(n, 0.5, r)))[:, None]
    return y * (amp * vel)[:, None]


def _glass_pad(f, vel, dur, r, c):
    """FM carriers 10 cents apart, left and right, whose modulators (2, 3 and 7 times the pitch) wander slowly and
    independently, so the upper partials shimmer; the octave pair speaks at once, a glass ping before the swell"""
    g, h, b = _x(c, 'ring', 1.2), _x(c, 'hard', -2.0), _x(c, 'bright', 0.9)
    w = min(0.7 * (1 + c['motion']), 0.95)
    body, crv = (0.5 * h, 2.5, 0.8, 2.4 * g), (0.5, 5.0, 5.0)
    return voice.fm(f, [
        dict(cents=-5, out=1.0, pan=-0.55, vel=0.5, mod=(1,), env=body, curve=crv),
        dict(ratio=2.0, index=0.9 * b, vel=0.5, key=-0.3, wobble=(w, 0.23), env=(1.0 * h, 3.0, 0.7, 2.0 * g)),
        dict(cents=5, out=1.0, pan=0.55, vel=0.5, mod=(3,), env=body, curve=crv),
        dict(ratio=3.0, index=0.7 * b, vel=0.5, key=-0.3, wobble=(w, 0.17), env=(1.4 * h, 3.0, 0.7, 2.0 * g)),
        dict(ratio=2.0, cents=3, out=0.3, vel=0.5, mod=(5,), env=(0.04 * h, 2.5, 0.7, 2.6 * g)),
        dict(ratio=7.0, cents=8, index=0.6 * b, key=-0.5, wobble=(w, 0.31), env=(0.002, 1.2, 0.35, 2.0 * g))],
        vel, dur, r)


def _string_pad(f, vel, dur, r, c):
    g, h, m = _x(c, 'ring', 1.3), _x(c, 'hard', -2.0), 1 + c['motion']
    return voice.analog(
        f, vel, dur, r, drift=3.0, smear=0.008, vib=(7 * m, 5.3, 0.5, 0.6), slew=0.003,
        osc=(dict(shape='saw', voices=2, detune=6, spread=0.35, level=0.9),
             dict(shape='pulse', pw=0.3, semi=12, level=0.25)),
        sub=0.2 * _x(c, 'size', 1.5), filter='low', cutoff=3200 * _x(c, 'bright', 1.3), keytrack=0.3, res=0.05,
        env=0.6, hp=150.0, fenv=(0.3 * h, 1.0, 0.6, 0.8), amp=(0.2 * h, 0.8, 0.85, 0.8 * g))


def _drone(f, vel, dur, r, c):
    g, h, m = _x(c, 'ring', 1.3), _x(c, 'hard', -2.0), 1 + c['motion']
    return voice.analog(
        f, vel, dur, r, drift=5.0, smear=0.02, curve=(0.5, 5.0, 5.0), slew=0.005,
        osc=(dict(shape='saw', voices=3, detune=9 * m, spread=0.9, level=0.8),
             dict(shape='pulse', pw=0.4, pwm=0.25, pwm_rate=0.07, semi=7, voices=2, detune=6, spread=0.8, level=0.22),
             dict(shape='saw', semi=-12, level=0.5)),
        sub=0.5 * _x(c, 'size', 1.5), noise=0.08, noise_width=0.8, cutoff=260 * _x(c, 'bright', 1.3), keytrack=0.3,
        res=0.35, env=1.2, cut_lfo=(0.9 * m, 0.11, 'random'), drive=6.0 + 6.0 * min(c['dirt'], 0.0),
        fenv=(2.5 * h, 4.0, 0.6, 3.0 * g), amp=(1.2 * h, 3.0, 0.9, 2.8 * g))


def _sub_swell(f, vel, dur, r, c):
    """a sine that grows over the whole hold, its harmonics arriving with the level, over band-limited rumble"""
    amp = voice.adsr(dur, 0.97 * dur, 0.05, 1.0, 0.45 * _x(c, 'ring', 1.5), curve=(-3.0 * _x(c, 'hard', 0.6), 5.0, 5.0),
                     slew=0.004)
    n = len(amp)
    u = np.clip(np.arange(n) / (SR * dur), 0.0, 1.0)
    w = 2 * np.pi * np.cumsum(f * 2.0 ** (-1.5 * (1 - u) ** 2 / 12)) / SR
    k = _x(c, 'bright', 1.5) * amp ** 1.5
    y = np.sin(w) + k * (0.3 * np.sin(2 * w) + 0.18 * np.sin(3 * w) + 0.1 * np.sin(4 * w) + 0.06 * np.sin(5 * w))
    rumble = dsp.butter(r.standard_normal(n), 'band', (30.0, 140.0))
    y += 0.3 * _x(c, 'size', 1.0) * rumble / (rumble.std() + 1e-12)
    beat = 1.0 - 0.25 * (1 + c['motion']) * u * (0.5 - 0.5 * np.cos(2 * np.pi * np.cumsum(3.0 * 3.0 ** u) / SR))
    return _st(y * amp * beat * vel)


# ------------------------------------------------------------------------------------------------ basses
def _sub_bass(f, vel, dur, r, c):
    """a sine with harmonics 2-6 at -17 dB and falling (a phone speaker plays nothing under 200 Hz, so the pitch of
    a sub has to live in its overtones too); they are louder for the first 50 ms and on lower notes"""
    amp = voice.adsr(dur, 0.008 * _x(c, 'hard', -1.5), 0.6, 0.85, 0.09 * _x(c, 'ring', 1.5), slew=0.003)
    t = np.arange(len(amp)) / SR
    w = 2 * np.pi * f * t
    k = _x(c, 'bright', 1.5) * (1.0 + 1.5 * np.exp(-t / 0.05)) * (49.0 / f) ** 0.5
    y = np.sin(w) + k * sum(a * np.sin(j * w) for j, a in ((2, 0.14), (3, 0.09), (4, 0.05), (5, 0.035), (6, 0.02)))
    return _st(y * amp * vel)


def _round_bass(f, vel, dur, r, c):
    g, h = _x(c, 'ring', 1.3), _x(c, 'hard', -1.5)
    return voice.analog(
        f, vel, dur, r, drift=1.0,
        osc=(dict(shape='tri', level=0.7, phase=0.25), dict(shape='saw', level=0.6, phase=0.5)),
        sub=0.5 * _x(c, 'size', 1.0), cutoff=700 * _x(c, 'bright', 1.3), keytrack=0.5, res=0.12, env=2.2, vel_cut=1.5,
        drive=3.0 + 3.0 * min(c['dirt'], 0.0), fenv=(0.002, 0.25 * g, 0.15, 0.1),
        amp=(0.004 * h, 0.6, 0.75, 0.09 * g))


def _reese_bass(f, vel, dur, r, c):
    g, h, m = _x(c, 'ring', 1.3), _x(c, 'hard', -1.5), 1 + c['motion']
    y = voice.analog(
        f, vel, dur, r, drift=2.0, osc=(dict(shape='saw', voices=2, detune=14 * m, spread=0.6),),
        sub=0.55 * _x(c, 'size', 1.0), cutoff=1100 * _x(c, 'bright', 1.3), keytrack=0.4, res=0.2,
        cut_lfo=(0.5 * m, 0.23, 'random'), drive=8.0 + 8.0 * min(c['dirt'], 0.0), vel_cut=1.0,
        amp=(0.01 * h, 0.5, 0.9, 0.12 * g))
    return voice.mono_below(y, 120.0)


def _pluck_bass(f, vel, dur, r, c):
    g, h = _x(c, 'ring', 1.3), _x(c, 'hard', -1.5)
    return voice.analog(
        f, vel, dur, r, drift=1.0, osc=(dict(shape='saw', phase=0.5), dict(shape='pulse', pw=0.35, cents=6, phase=0.3)),
        sub=0.45 * _x(c, 'size', 1.0), cutoff=380 * _x(c, 'bright', 1.3), keytrack=0.5, res=0.3, env=3.6, vel_cut=2.0,
        drive=5.0 + 5.0 * min(c['dirt'], 0.0), fenv=(0.001, 0.13 * g, 0.0, 0.08),
        amp=(0.002 * h, 0.5 * g, 0.0, 0.08 * g))


def _kick_808(f, vel, dur, r, c):
    """a sine that starts 2.7 octaves above its pitch and falls onto it in two stages (11 ms: the beater, 55 ms: the
    drum settling), a slow exponential decay, tape saturation for the harmonics small speakers need, and a 2 ms
    click; the phase starts at zero so every hit is the same"""
    g = _x(c, 'ring', 1.3)
    amp = voice.adsr(dur, 0.0006, 1.5 * g, 0.0, 0.3 * g, curve=(2.0, 4.5, 5.0), slew=0.0006)
    t = np.arange(len(amp)) / SR
    semi = (0.7 + 0.3 * vel) * (26.0 * _x(c, 'hard', 0.3) * np.exp(-t / 0.011) + 7.0 * np.exp(-t / 0.055))
    y = fx.saturate(voice.sine(f * 2.0 ** (semi / 12)) * amp, 7.0 + 9.0 * c['dirt'] + 6.0 * c['bright'], 'tape',
                    level='peak')
    k = voice.tick(r, 0.012, 1500.0, 7000.0, 0.0015)
    y[:len(k)] += 0.2 * _x(c, 'hard', 1.5) * _x(c, 'bright', 1.0) * k
    return _st(y * vel)


# ------------------------------------------------------------------------------------------------ hits and fx
def _braam(f, vel, dur, r, c):
    """twelve detuned saws in two octaves and a little noise through a ladder that opens 3.6 octaves with the level
    (brass gets brighter as it gets louder), brass formants at 520 and 1500 Hz, tape saturation after the VCA so the
    rasp follows the dynamics, a clean sub an octave down. hard >= 0 is the hit (30 ms attack, a short scoop up into
    the pitch); toward hard = -1 the attack stretches to a 1.3 s swell that ends in the blast"""
    swell = max(0.0, -c['hard'])
    att = 0.03 * 45.0 ** swell * 2.0 ** -max(c['hard'], 0.0)
    g, m = _x(c, 'ring', 1.2), 1 + c['motion']
    gate = dur + att
    amp = voice.adsr(gate, att, 2.5 * g, 0.55, 1.5 * g, curve=(1.5 - 4.0 * swell, 4.0, 5.0), slew=0.002)
    n = len(amp)
    t = np.arange(n) / SR
    fe = voice.fit(voice.adsr(gate, 0.6 * att + 0.02, 1.0 * g, 0.5, 1.3 * g, curve=(1.0 - 3.0 * swell, 4.0, 4.0)), n)
    semi = (-0.6 * (1 - swell) * np.exp(-t / 0.05) - 1.2 * swell * np.clip(1 - t / att, 0.0, 1.0) ** 2
            - 0.2 * np.clip((t - att) / 3.0, 0.0, 1.0))
    freq = f * 2.0 ** (semi / 12)
    x = (voice.unison(freq, 'saw', r, 7, 20 * m, 0.7, 0.65, 3.0, 0.004)
         + 0.55 * voice.unison(2 * freq, 'saw', r, 5, 14 * m, 0.9, 0.6, 3.0, 0.004) + 0.1 * voice.noise2(n, r, 0.8))
    cut = 190 * _x(c, 'bright', 1.2) * (f / 73.4) ** 0.3 * 2.0 ** (3.6 * fe + 1.2 * (vel - 1.0))
    y = dsp.peq(dsp.peq(voice.ladder(x, cut, 0.18), 520.0, 4.0, 1.0), 1500.0, 3.0, 1.2)
    y = fx.saturate(y * amp[:, None], 12.0 + 10.0 * min(c['dirt'], 0.0), 'tape')
    y += (0.6 * _x(c, 'size', 1.0) * voice.sine(freq / 2) * amp)[:, None]
    return voice.mono_below(y, 110.0) * vel


def _stab(f, vel, dur, r, c):
    g, h = _x(c, 'ring', 1.3), _x(c, 'hard', -1.5)
    return voice.analog(
        f, vel, dur, r, drift=3.0, smear=0.002,
        osc=(dict(shape='saw', voices=5, detune=18 * (1 + c['motion']), spread=0.8, blend=0.6),
             dict(shape='pulse', pw=0.4, semi=12, level=0.3)),
        sub=0.2 * _x(c, 'size', 1.5), cutoff=500 * _x(c, 'bright', 1.3), keytrack=0.4, res=0.22, env=3.5, vel_cut=1.5,
        drive=4.0 + 4.0 * min(c['dirt'], 0.0), fenv=(0.001, 0.16 * g, 0.05, 0.12),
        amp=(0.002 * h, 0.3 * g, 0.0, 0.14 * g))


def _zap(f, vel, dur, r, c):
    """a designed laser: an FM pair (1:2.41, inharmonic, with feedback) whose pitch falls 3.5 octaves in 150 ms,
    through a resonant ladder that tracks three times the pitch (the "pew"), over a 55 Hz body thump and a 1 ms
    crack"""
    g, h, b = _x(c, 'ring', 1.2), _x(c, 'hard', -1.5), _x(c, 'bright', 0.4)
    t = np.arange(n_of(dur + 2.0)) / SR
    freq = f * 2.0 ** ((-12.0 + 42.0 * np.exp(-t / (0.05 * g * (1 + 0.5 * c['motion'])))) / 12)
    y = voice.fm(freq, [
        dict(out=1.0, vel=0.5, mod=(1,), fb=0.4, env=(0.001 * h, 0.3 * g, 0.0, 0.12 * g), curve=(2.0, 4.0, 5.0)),
        dict(ratio=2.41, index=3.0 * b, vel=0.4, env=(0.001, 0.14 * g, 0.12, 0.1))], vel, dur)[:, 0]
    n = len(y)
    y = voice.ladder(y, np.clip(3.0 * _x(c, 'bright', 1.0) * voice.fit(freq, n), 100.0, 16000.0), 0.5)
    t = t[:n]
    y += dsp.fade(0.5 * vel * _x(c, 'size', 1.0) * voice.sine(55.0 + 130.0 * np.exp(-t / 0.03)) * np.exp(-t / 0.07),
                  0.0, 0.05)
    k = voice.tick(r, 0.008, 2500.0, 12000.0, 0.0012)
    y[:len(k)] += 0.3 * vel * _x(c, 'hard', 1.0) * k
    return _st(y)


def _power_up(f, vel, dur, r, c):
    """the patch's own figure: a major arpeggio over two octaves (root, 3rd, 5th, octave, ...) spread over dur,
    slightly accelerating and growing louder, each note a glassy FM pluck; the last note rings, and high pings from
    the chord scatter after it (the shimmer) over a rising breath of noise"""
    g, b, m = _x(c, 'ring', 1.3), _x(c, 'bright', 0.8), 1 + c['motion']
    takes = []
    for k, s in enumerate(_MAJOR):
        u = k / (len(_MAJOR) - 1)
        d = (1.0 if u == 1 else 0.32) * g
        side = 0.45 * (-1) ** k * u
        takes.append((dur * u ** 0.92, voice.fm(f * 2.0 ** (s / 12), [
            dict(out=1.0, pan=side, vel=0.5, mod=(1,), env=(0.001, d, 0.0, d)),
            dict(ratio=3.0, index=1.5 * b, vel=0.6, key=-0.35, env=(0.001, 0.12, 0.05, 0.1)),
            dict(cents=6, out=0.5, pan=-side, vel=0.5, mod=(3,), env=(0.004, 0.8 * d, 0.0, d)),
            dict(ratio=2.0, index=0.8 * b, env=(0.001, 0.2, 0.2, 0.2))], vel * (0.72 + 0.28 * u), 0.06)))
    for _ in range(int(round(5 * m))):
        fk = f * 2.0 ** ((24 + r.choice((0, 4, 7, 12, 16))) / 12)
        takes.append((dur + r.uniform(0.04, 0.5) * g, voice.fm(fk, [
            dict(out=0.13, pan=r.uniform(-0.8, 0.8), mod=(1,), env=(0.002, 0.35 * g, 0.0, 0.3 * g)),
            dict(ratio=2.0, index=0.6 * b, env=(0.001, 0.08, 0.0, 0.05))], vel, 0.03)))
    n = n_of(dur + 0.25)
    u = np.arange(n) / n
    ns = voice.noise2(n, r, 0.8)
    air = np.stack([dsp.svf(ns[:, ch], 1200.0 * 6.0 ** u, 1.2, 'bandn') for ch in (0, 1)], axis=1)
    takes.append((0.0, 0.05 * vel * air * (np.sin(np.pi * u ** 1.5) ** 2)[:, None]))
    return _mix(takes)


def _power_down(f, vel, dur, r, c):
    """the patch's own figure: a minor arpeggio falling two octaves, slowing down, each note darker than the last
    and sagging as it dies, over a tone that runs down like a motor: its pitch falls two octaves, its filter closes
    and the flutter in its level slows from 14 to 3 Hz"""
    g, b, m = _x(c, 'ring', 1.3), _x(c, 'bright', 1.0), 1 + c['motion']
    steps = (0, -5, -9, -12, -17, -24)
    span = dur + 0.6 * g
    bed = voice.analog(
        f, vel, dur + 0.2, r, osc=(dict(shape='saw', voices=3, detune=10 * m, spread=0.5), dict(shape='pulse', pw=0.4)),
        pitch=lambda t: -24.0 * np.clip(t / span, 0.0, 1.0), sub=0.3 * _x(c, 'size', 1.5), cutoff=900 * b, keytrack=0.4,
        res=0.25, env=-2.5, fenv=(span, 0.1, 1.0, 0.1), fcurve=(1.0, 5.0, 5.0), amp=(0.02, span, 0.0, 0.4 * g),
        curve=(2.0, 2.0, 5.0))
    u = np.clip(np.arange(len(bed)) / (SR * span), 0.0, 1.0)
    bed *= (1.0 - 0.3 * m * (0.5 - 0.5 * np.cos(2 * np.pi * np.cumsum(14.0 * (3.0 / 14.0) ** u) / SR)))[:, None]
    takes = [(0.0, 0.6 * bed)]
    for k, s in enumerate(steps):
        u = k / (len(steps) - 1)
        sag = 0.7 + 0.8 * u
        takes.append((dur * u ** 1.15, voice.analog(
            f * 2.0 ** (s / 12), vel * (1 - 0.3 * u), 0.12, r, slew=0.003,
            osc=(dict(shape='pulse', pw=0.45, voices=2, detune=7 * m, spread=0.4), dict(shape='tri', level=0.6)),
            pitch=lambda t, sag=sag: -sag * (1.0 - np.exp(-t / 0.15)), cutoff=1100 * b * 2.0 ** (-1.4 * u),
            keytrack=0.4, res=0.12, env=1.5, fenv=(0.004, 0.2, 0.0, 0.2), amp=(0.008, 0.7 * g, 0.0, 0.5 * g))))
    return _mix(takes)


def _riser_tone(f, vel, dur, r, c):
    """a riser that stays a tone: seven saws whose detune widens, pitch up by only 2 semitones (late), a ladder that
    opens 5.6 octaves, a noise band riding the cutoff, a pulse in the level that speeds up from 4 to 16 Hz, and a
    stereo image that opens as it climbs; it stops at the end of the hold"""
    m = 1 + c['motion']
    amp = voice.adsr(dur, 0.98 * dur, 0.05, 1.0, 0.07 * _x(c, 'ring', 1.5), curve=(-3.0 * _x(c, 'hard', 0.6), 5.0, 5.0),
                     slew=0.003)
    n = len(amp)
    u = np.clip(np.arange(n) / (SR * dur), 0.0, 1.0)
    freq = f * 2.0 ** ((2.0 + c['size']) * u ** 2.2 / 12)
    wide = u ** 1.5
    x = (np.sqrt(1.0 - wide)[:, None] * voice.unison(freq, 'saw', r, 7, 9, 0.5, 0.5, 2.0, 0.006)
         + np.sqrt(wide)[:, None] * voice.unison(freq, 'saw', r, 7, 36 * m, 0.9, 0.7, 3.0, 0.006)
         + 0.5 * voice.unison(freq / 2, 'saw', r, 3, 12, 0.3, drift=2.0))
    cut = 220 * _x(c, 'bright', 1.0) * 2.0 ** (5.6 * u ** 1.6)
    y = voice.ladder(x, cut, 0.22 + 0.2 * u)
    ns = voice.noise2(n, r, 0.9)
    air = np.stack([dsp.svf(ns[:, ch], np.clip(1.6 * cut, 300.0, 15000.0), 1.4, 'bandn') for ch in (0, 1)], axis=1)
    y += 0.35 * (u ** 2)[:, None] * air
    pulse = 1.0 - 0.3 * m * u * (0.5 - 0.5 * np.cos(2 * np.pi * np.cumsum(4.0 * 4.0 ** u) / SR))
    return fx.width(y * (amp * pulse * vel)[:, None], 0.5 + 0.7 * u)


def _noise_sweep(f, vel, dur, r, c):
    amp = voice.adsr(dur, 0.92 * dur, 0.05, 1.0, 0.18 * _x(c, 'ring', 1.5), curve=(-2.5 * _x(c, 'hard', 0.6), 5.0, 5.0),
                     slew=0.003)
    n = len(amp)
    u = np.clip(np.arange(n) / (SR * dur), 0.0, 1.0)
    fc = np.clip(0.5 * f * _x(c, 'bright', 1.0) * 2.0 ** (5.0 * u ** 1.5), 60.0, 16000.0)
    ns = voice.noise2(n, r, 0.5)
    y = np.stack([dsp.svf(ns[:, ch], fc, 1.8, 'bandn') + 0.35 * dsp.svf(ns[:, ch], 0.5 * fc, 0.7, 'low')
                  for ch in (0, 1)], axis=1)
    y = voice.phaser(y, rate=0.35 * (1 + c['motion']), lo=400.0, hi=5000.0, feedback=0.45, mix=0.5)
    return y * (amp * vel)[:, None]


# ------------------------------------------------------------------------------------------------ the library
def _def(note, kind, pitch, lo, hi, gain, says, fx=(), chord=None, own=False):
    return dict(note=note, kind=kind, default_pitch=pitch, range=(lo, hi), gain=gain, says=says, fx=fx, chord=chord,
                own=own)


# each patch: its voice, kind, default pitch, range, the gain that puts velocity 1 near 0.9, what it sounds like, and
# its effects in order: ('chorus' | 'phaser' | 'echo', settings of voice.*) and ('room', (a space of synth.space,
# the level of its tail)). chord: the semitones a single pitch is played as; own: the patch plays its own figure
_DEFS = {
    'soft_pluck': _def(_soft_pluck, 'pluck', 'C5', 'C3', 'C7', 0.638, [
        'a soft rounded synthesizer pluck', 'a gentle mellow synth pluck note from a product film',
        'a warm sine-like plucked synth tone'],
        (('echo', dict(time=0.27, feedback=0.3, mix=0.12, damp=3200.0)), ('room', ('plate', 0.14)))),
    'glass_pluck': _def(_glass_pluck, 'pluck', 'E5', 'C4', 'C7', 0.592, [
        'a bright glassy FM synthesizer pluck', 'a short crystalline digital pluck note',
        'a glass-like bell pluck'],
        (('echo', dict(time=0.2, feedback=0.35, mix=0.13, damp=6000.0)), ('room', ('plate', 0.2)))),
    'muted_pluck': _def(_muted_pluck, 'pluck', 'A3', 'E2', 'E5', 0.8, [
        'a muted synthesizer pluck', 'a short palm-muted synth note', 'a dark filtered sawtooth pluck'],
        (('echo', dict(time=0.1875, feedback=0.4, mix=0.16, damp=2500.0)), ('room', ('studio', 0.08)))),
    'bubble': _def(_bubble, 'pluck', 'G5', 'C4', 'C7', 1.01, [
        'a soft bubble pop', 'a rounded user interface blip that bends upward', 'a gentle water drop tone'],
        (('room', ('room', 0.16)),)),
    'warm_keys': _def(_warm_keys, 'key', 'C4', 'C2', 'C6', 0.611, [
        'warm analog synthesizer keys', 'a soft analog polysynth chord', 'a mellow vintage synth keyboard note'],
        (('chorus', dict(rate=0.513, depth=1.85, mix=0.45)), ('room', ('studio', 0.12)))),
    'fm_keys': _def(_fm_keys, 'key', 'C4', 'C2', 'C7', 0.336, [
        'a glassy FM electric piano note', 'a digital electric piano', 'a bright bell-like electric piano'],
        (('chorus', dict(rate=0.863, depth=1.2, mix=0.3)), ('room', ('studio', 0.1)))),
    'fm_bell': _def(_fm_bell, 'bell', 'A5', 'C4', 'C7', 0.564, [
        'an FM synthesizer bell', 'a clear digital bell tone ringing out', 'a glassy chime with a long pure tail'],
        (('room', ('hall', 0.22)),)),
    'mallet_synth': _def(_mallet_synth, 'key', 'G4', 'C3', 'C7', 0.738, [
        'a synthesizer mallet note', 'a soft marimba-like synth tone', 'a round digital mallet percussion note'],
        (('chorus', dict(mix=0.2)), ('room', ('studio', 0.12)))),
    'square_lead': _def(_square_lead, 'lead', 'A4', 'C3', 'C6', 0.623, [
        'a soft square wave synthesizer lead with vibrato', 'a hollow mellow synth lead note',
        'a warm pulse wave lead'],
        (('echo', dict(time=0.33, feedback=0.35, mix=0.16, damp=3500.0, wow=0.5)), ('room', ('plate', 0.14)))),
    'saw_lead': _def(_saw_lead, 'lead', 'A4', 'C3', 'C6', 0.572, [
        'a sawtooth synthesizer lead', 'a bright detuned supersaw lead note', 'a wide trance synth lead'],
        (('chorus', dict(mix=0.25)), ('echo', dict(time=0.3, feedback=0.4, mix=0.18)), ('room', ('hall', 0.16)))),
    'warm_pad': _def(_warm_pad, 'pad', 'C4', 'C2', 'C6', 0.49, [
        'a warm analog synthesizer pad', 'a slow lush synth pad chord', 'a soft detuned analog pad'],
        (('chorus', dict(rate=0.513, mix=0.55)), ('room', ('hall', 0.3)))),
    'air_pad': _def(_air_pad, 'pad', 'C5', 'C3', 'C6', 0.196, [
        'a breathy airy synthesizer pad', 'a soft airy pad with breath noise', 'an ethereal atmospheric pad'],
        (('chorus', dict(mix=0.25)), ('room', ('hall', 0.4)))),
    'glass_pad': _def(_glass_pad, 'pad', 'C5', 'C3', 'C6', 0.282, [
        'a glassy shimmering FM synthesizer pad', 'a bright crystalline digital pad', 'a shimmering bell-like pad'],
        (('chorus', dict(voices=3, mix=0.4)), ('room', ('hall', 0.35)))),
    'string_pad': _def(_string_pad, 'pad', 'C4', 'C2', 'C6', 0.402, [
        'a string ensemble synthesizer pad', 'a vintage string machine chord', 'lush synthetic strings'],
        (('chorus', dict(voices=3, depth=1.6, delay=4.5, mix=0.85)), ('phaser', dict(rate=0.12, mix=0.2)),
         ('room', ('hall', 0.25)))),
    'drone': _def(_drone, 'pad', 'D2', 'A0', 'D3', 0.765, [
        'a dark low synthesizer drone', 'an ominous evolving low bed', 'a deep dark ambient drone'],
        (('phaser', dict(rate=0.05, lo=200.0, hi=1600.0, mix=0.3)), ('room', ('cathedral', 0.3)))),
    'sub_swell': _def(_sub_swell, 'pad', 'A1', 'E1', 'E2', 0.474, [
        'a low sub bass swell', 'a deep rumbling bass rising in volume', 'a cinematic sub bass build-up'],
        (('room', ('hall', 0.08)),)),
    'sub_bass': _def(_sub_bass, 'bass', 'G1', 'C1', 'C3', 0.849, [
        'a deep clean sub bass note', 'a low sine wave bass', 'a smooth deep bass tone'],
        (('room', ('room', 0.03)),)),
    'round_bass': _def(_round_bass, 'bass', 'E2', 'C1', 'C4', 0.789, [
        'a round warm synthesizer bass note', 'a soft mellow synth bass', 'a smooth analog bass'],
        (('room', ('room', 0.04)),)),
    'reese_bass': _def(_reese_bass, 'bass', 'G1', 'C1', 'C3', 0.761, [
        'a reese bass', 'a detuned growling synthesizer bass', 'a dark moving drum and bass bass note'],
        (('room', ('room', 0.05)),)),
    'pluck_bass': _def(_pluck_bass, 'bass', 'E2', 'C1', 'C4', 0.685, [
        'a plucked synthesizer bass note', 'a punchy analog synth bass', 'a short funky synth bass'],
        (('room', ('room', 0.05)),)),
    'kick_808': _def(_kick_808, 'bass', 'G1', 'C1', 'C3', 0.866, [
        'an 808 kick drum', 'a deep booming sub kick with a long tail', 'a pitched bass drum hit'],
        (('room', ('room', 0.03)),)),
    'braam': _def(_braam, 'hit', 'D2', 'A0', 'A2', 0.522, [
        'a cinematic braam', 'a huge low brass blast from a movie trailer', 'a deep distorted horn-like trailer hit'],
        (('room', ('cathedral', 0.28)),)),
    'stab': _def(_stab, 'hit', 'C4', 'C3', 'C5', 0.589, [
        'a synthesizer chord stab', 'a short house music chord stab', 'a punchy minor chord synth hit'],
        (('chorus', dict(mix=0.35)), ('echo', dict(time=0.28, feedback=0.35, mix=0.15)), ('room', ('plate', 0.2))),
        chord=(0, 3, 7, 10)),
    'zap': _def(_zap, 'fx', 'A4', 'A3', 'A5', 0.677, [
        'a laser zap', 'a science fiction laser gun shot', 'a descending synth zap'],
        (('echo', dict(time=0.09, feedback=0.3, mix=0.2, damp=5000.0)), ('room', ('studio', 0.12)))),
    'power_up': _def(_power_up, 'fx', 'C5', 'C4', 'C6', 0.501, [
        'a power-up sound', 'a rising sparkling arpeggio', 'a magical ascending synth run with shimmer'],
        (('echo', dict(time=0.16, feedback=0.45, mix=0.2, damp=7000.0)), ('room', ('plate', 0.25))), own=True),
    'power_down': _def(_power_down, 'fx', 'C5', 'C4', 'C6', 0.348, [
        'a power-down sound', 'a descending synth arpeggio winding down', 'a machine shutting down tone'],
        (('echo', dict(time=0.2, feedback=0.4, mix=0.18, damp=2500.0)), ('room', ('studio', 0.15))), own=True),
    'riser_tone': _def(_riser_tone, 'fx', 'C4', 'C3', 'C5', 0.441, [
        'a tonal synthesizer riser', 'a rising tension build-up before a drop',
        'a swelling synth rising in pitch and brightness'],
        (('chorus', dict(mix=0.3)), ('room', ('hall', 0.22)))),
    'noise_sweep': _def(_noise_sweep, 'fx', 'C5', 'C4', 'C6', 0.477, [
        'a white noise sweep', 'a filtered noise riser whoosh', 'a rising noise swoosh transition'],
        (('room', ('hall', 0.2)),)),
}
PATCHES = {k: {q: d[q] for q in ('says', 'kind', 'default_pitch', 'range')} for k, d in _DEFS.items()}


def _hz(pitch):
    """a pitch as Hz: a note name or '440hz' (knobs.hz), an int 0..127 as a MIDI note, any other number as Hz"""
    if isinstance(pitch, (int, np.integer)) and not isinstance(pitch, bool) and 0 <= pitch <= 127:
        return 440.0 * 2.0 ** ((int(pitch) - 69) / 12)
    return knobs.hz(pitch)


def _controls(controls, r):
    bad = sorted(set(controls) - set(CONTROLS) - {'vary'})
    if bad:
        raise ValueError(f'unknown control {", ".join(bad)}: one of {", ".join(CONTROLS)}, vary')
    c = {k: float(controls.get(k) or 0.0) for k in CONTROLS}
    vary = float(controls.get('vary') or 0.0)
    if any(abs(v) > 1 for v in c.values()) or not 0 <= vary <= 1:
        raise ValueError('controls are numbers in -1..1 (0 = the patch as designed), vary in 0..1')
    c['level'] = 1.0
    if vary:                       # a take: timbre and level move a little, the pitch never does
        for k, s in (('bright', 0.25), ('hard', 0.25), ('ring', 0.2), ('motion', 0.2), ('width', 0.15)):
            c[k] = float(np.clip(c[k] + vary * s * r.standard_normal(), -1.0, 1.0))
        c['level'] = 10.0 ** (-0.075 * vary * abs(r.standard_normal()))
    return c


def _room(x, name, wet):
    """x with the tail of a shared room (earshot.synth.space) fed from its mono sum: stereo, as long as x + tail"""
    kw = {k: v for k, v in space.SPACES[name].items() if k != 'wet'}
    y = fx.reverb(x.mean(axis=1), None, wet=wet, dry=0.0, stereo=True, ir_seed=space.SEED, **kw)
    y[:len(x)] += x
    return y


def _effects(x, chain, c):
    wet = 1.0 + c['space']
    for kind, kw in chain:
        if kind == 'chorus':
            x = voice.chorus(x, **{**kw, 'mix': min(kw['mix'] * (1 + c['motion']), 1.0)})
        elif kind == 'phaser':
            x = voice.phaser(x, **{**kw, 'mix': min(kw['mix'] * (1 + c['motion']), 1.0)})
        elif kind == 'echo' and wet > 0:
            x = voice.echo(x, **{**kw, 'mix': kw['mix'] * wet})
        elif kind == 'room' and wet > 0:
            x = _room(x, kw[0], kw[1] * wet)
    return x


def play(patch, pitch=None, vel=0.8, dur=0.3, r=None, dry=False, **controls):
    """one note or chord of a patch as stereo float64 (n, 2): held for dur s, then released, with the tail of its
    effects. pitch: a note name ('C5'), a MIDI note (an int 0..127), Hz (a float, or '440hz'), or a list of them for
    a chord; None = the patch's default_pitch. stab plays a minor-seventh chord on a single pitch; power_up and
    power_down play their own figure from the pitch over dur s. vel 0..1 sets level and brightness (1 peaks near 0.9;
    chords are scaled by 1 / sqrt(notes) and limited under 0.95). r: a numpy Generator or a seed; the same arguments
    give the same samples. dry=True leaves out the patch's effects. Controls: see the module docstring; an unknown
    patch or control raises ValueError with the valid names.
        y = patches.play('fm_keys', ['A3', 'C4', 'E4'], vel=0.6, dur=1.2, bright=0.2, vary=0.3, r=dsp.rng(5))"""
    if patch not in _DEFS:
        raise ValueError(f'unknown patch {patch!r}: one of {", ".join(_DEFS)}')
    if not 0 < vel <= 1 or not dur > 0:
        raise ValueError('vel is in (0, 1] and dur is positive seconds')
    d = _DEFS[patch]
    r = r if isinstance(r, np.random.Generator) else dsp.rng(r or 0)
    c = _controls(controls, r)
    many = isinstance(pitch, (list, tuple, np.ndarray))
    fs = [_hz(p) for p in (pitch if many else [d['default_pitch'] if pitch is None else pitch])]
    if not fs:
        raise ValueError('a chord needs at least one pitch')
    if d['chord'] and len(fs) == 1:
        fs = [fs[0] * 2.0 ** (s / 12) for s in d['chord']]
    takes = [d['note'](f, float(vel), float(dur), r, c) for f in (fs[:1] if d['own'] else fs)]
    x = _mix([(0.0, t) for t in takes]) / np.sqrt(len(takes))
    if c['dirt'] > 0:
        x = fx.saturate(x, 15.0 * c['dirt'], 'tape')
    if c['size']:
        x = dsp.peq(dsp.peq(x, 200.0, 5.0 * c['size'], 0.7), 3500.0, -2.5 * c['size'], 0.7)
    if not dry:
        x = _effects(x, d['fx'], c)
    if c['width']:
        x = fx.width(x, 1.0 + c['width'] * (1.0 if c['width'] < 0 else 0.7))
    x = dsp.fade(dsp.trim_tail(x * (d['gain'] * c['level']), -72.0, 0.02), 0.0, 0.03)
    return fx.limit(x, ceiling_db=-0.45, true_peak=False) if np.abs(x).max() > 0.94 else x
