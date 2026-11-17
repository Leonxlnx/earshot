"""Retro / chip game sounds in the manner of the 8-bit and 16-bit console sound chips: coins, jumps, lasers,
explosions, power-ups, menu blips, jingles and fanfares. They should sound like a well-made console game, so the
character is the hardware's and the edges are clean:

- channels: a pulse with the four duty cycles (12.5 / 25 / 50 / 75 %) and duty sweeps; a stepped triangle (a 32-step
  staircase over 16 levels, whose steps are the soft buzz of the low notes); noise from a 15-bit linear-feedback
  shift register, long mode (hiss) or short mode (a 93-step loop: metallic and pitched), clocked at one of 16 stepped
  rates; a 4-bit wavetable channel (32 steps of 0..15) for softer tones; and delta-coded sample-and-hold grit.
- every channel is a train of level steps, and every step is rendered band-limited (a windowed-sinc BLEP, 48 taps,
  cut-off 20 kHz, Kaiser beta 8.6): folded energy of a 3 kHz pulse is about 90 dB down, where a naive square at
  48 kHz folds at -20 to -30 dB.
- a sound is one or more voice dicts in the well-known public parameter model of retro sound generators (slide and
  slide acceleration, vibrato, arpeggio steps at frame rate, duty sweep, retrigger, attack / sustain / punch / decay
  with optional 16-level volume steps, swept low- and high-pass, phaser), in physical units (see DEFAULTS);
  this is an implementation from the description, not a port.
- generators draw their parameters from ranges where every draw sounds right: notes snapped to equal temperament,
  arpeggios on fourths, fifths, octaves and major triads, slides that stop before they turn into sirens.
- polish (0..1) is the console's output stage and a sound designer's finish: a high-pass at 40 Hz, a gentle
  low-pass at 11 kHz (polish 0.5), sub-millisecond edge fades, a quiet slapback, and every generator at one
  momentary-max loudness (LOUD_LUFS), peaks at most PEAK.

    from earshot.synth import chip
    y = chip.render('coin', 3)                          # stereo (n, 2) float64 at 48 kHz
    y = chip.render('one_up', 1, pitch='C6')            # in a key
    y = chip.render('laser', 2, dur=0.4, bits=6)        # longer, through a 6-bit output
    vs = chip.jump(5); vs[0]['duty'] = 0.125; y = chip.render(vs)      # edit the drawn parameters
    y = chip.render({'wave': 'triangle', 'freq': 110, 'sustain': 0.3, 'decay': 0.2})
    chip.SOUNDS['coin']                     # {'says': [...], 'dur_s': .., 'range': (lo, hi), 'pitched': True}
"""
import functools
import re

import numpy as np

from ..forge import dsp
from ..forge.dsp import SR, njit
from ..forge.polish import lufs_max

FRAME = 1 / 60                       # one video frame: the update rate of the sound drivers of the era
ENV_RATE = 240.0                     # Hz: the envelope clock of the classic 8-bit chip (quarter frames)
DUTIES = (0.125, 0.25, 0.5, 0.75)
# the 16 noise clock rates of the classic 8-bit console: its CPU clock (1.789773 MHz) over its noise period table
NOISE_CLOCKS = tuple(1789773.0 / p for p in (4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034,
                                              4068))
# the 16 sample rates of its delta-modulation channel (Hz), from the same published hardware tables
DELTA_RATES = (4181.71, 4709.93, 5264.04, 5593.04, 6257.95, 7046.35, 7919.35, 8363.42, 9419.86, 11186.1, 12604.0,
               13982.6, 16884.6, 21306.8, 24858.0, 33143.9)
LOUD_LUFS = -14.0                    # momentary-max loudness of every rendered sound (polish > 0)
PEAK = 0.9                           # sample peak ceiling (and the target when polish = 0)
BLEP_W, BLEP_OS, BLEP_FC, BLEP_BETA = 24, 512, 20000.0, 8.6

DEFAULTS = {
    'wave': 'pulse',          # 'pulse' | 'triangle' (16-level staircase) | 'noise' (shift register) | 'wave' (4-bit)
    'freq': 440.0,            # Hz: pitch; for 'noise' the shift-register clock rate (short mode sounds at clock / 93)
    'slide': 0.0,             # semitones per second (+ rises)
    'slide_accel': 0.0,       # semitones per second squared (bends the slide; - with + slide = a rise that settles)
    'slide_end': None,        # s: the slide stops there and the pitch holds (None: slides to the end)
    'freq_min': 0.0,          # Hz: the sound stops (3 ms fade) when the pitch falls below this; 0 = off
    'vib_depth': 0.0,         # semitones (peak) of a sine vibrato
    'vib_rate': 6.0,          # Hz
    'vib_delay': 0.0,         # s before the vibrato fades in (over 80 ms)
    'arp': (),                # semitone offsets from freq, one per step: (0, 4, 7) is a major triad
    'arp_step': FRAME,        # s per arpeggio step (default one video frame)
    'arp_loop': False,        # cycle the steps (True) or hold the last one (False)
    'duty': 0.5,              # pulse: fraction of the period that is high
    'duty_sweep': 0.0,        # duty change per second
    'duty_steps': True,       # snap the duty to 12.5 / 25 / 50 / 75 % once per frame, as the chips did (False: PWM)
    'repeat': 0.0,            # Hz: retrigger rate; restarts the slide (not the arpeggio, vibrato or envelope)
    'attack': 0.0,            # s: linear rise
    'sustain': 0.1,           # s: held level (with punch)
    'decay': 0.2,             # s: linear fall to zero
    'punch': 0.0,             # 0..1: extra level at the start of the sustain, falling linearly over it
    'steps': True,            # quantise the envelope to 16 volume levels at 240 Hz like the hardware; False = smooth
    'lpf': 0.0,               # Hz: resonant low-pass cut-off (0 = off)
    'lpf_sweep': 0.0,         # octaves per second
    'lpf_res': 0.707,         # Q of the low-pass
    'hpf': 0.0,               # Hz: high-pass cut-off (0 = off)
    'hpf_sweep': 0.0,         # octaves per second
    'phaser': 0.0,            # ms: phaser / flanger delay offset (0 = off); the delayed copy is subtracted
    'phaser_sweep': 0.0,      # ms per second
    'phaser_mix': 0.6,        # 0..1 share of the delayed copy
    'table': 'soft',          # 'wave' channel: a WAVES name or 32 levels 0..15
    'noise_mode': 'long',     # 'long' (32767-step hiss) | 'short' (93-step loop: metallic, pitched)
    'noise_steps': True,      # snap the noise clock to the 16 NOISE_CLOCKS (slides then step through them)
    'grit': 0.0,              # 0..1 share of a delta-coded (7-bit counter, +-2 per tick) sample-and-hold copy
    'grit_rate': 16884.6,     # Hz: its hold rate (one of DELTA_RATES gives the hardware's)
    'gain': 0.0,              # dB
    't': 0.0,                 # s: start time inside a multi-voice sound
    'pan': 0.0,               # -1..1 (constant power)
    'loop': 0.0,              # s: > 0 makes the whole sound a seamless loop of this length
}
_TIME_KEYS = ('attack', 'sustain', 'decay', 't', 'vib_delay', 'arp_step', 'loop')
_RATE_KEYS = ('slide', 'duty_sweep', 'lpf_sweep', 'hpf_sweep', 'phaser_sweep', 'repeat')


def _wave_tables():
    k = np.arange(32) * 2 * np.pi / 32
    shapes = {'sine': np.sin(k), 'soft': np.sin(k) + 0.25 * np.sin(2 * k), 'organ': np.sin(k) + 0.45 * np.sin(3 * k),
              'hollow': np.sin(k) + 0.33 * np.sin(3 * k) + 0.2 * np.sin(5 * k), 'bell': np.sin(k) + 0.4 * np.sin(4 * k)}
    out = {n: tuple(int(v) for v in np.round(7.5 + 7.5 * s / np.abs(s).max())) for n, s in shapes.items()}
    out['saw'] = tuple(i // 2 for i in range(32))
    return out


WAVES = _wave_tables()                                     # 32-step, 4-bit waves for the 'wave' channel
TRIANGLE = tuple(range(15, -1, -1)) + tuple(range(16))     # the 32-step, 16-level staircase triangle


# ------------------------------------------------------------------------------------------------ band-limited steps
@functools.lru_cache(maxsize=1)
def _blep():
    """the band-limited unit step H(tau), tau in samples from -W to W: the integral of a Kaiser-windowed sinc (cut-off
    BLEP_FC). Every level change of a channel is rendered as this step instead of a jump, so nothing folds back."""
    W, OS = BLEP_W, BLEP_OS
    tau = np.arange(-W * OS, W * OS + 1) / OS
    fc = BLEP_FC / SR
    win = np.i0(BLEP_BETA * np.sqrt(np.clip(1 - (tau / W) ** 2, 0, 1))) / np.i0(BLEP_BETA)
    k = 2 * fc * np.sinc(2 * fc * tau) * win
    H = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2) / OS])
    return np.append(H / H[-1], 1.0)


@njit(cache=True)
def _bl_render(te, h, n, tab, W, OS):
    y = np.zeros(n)
    acc = np.zeros(n + 1)
    for k in range(len(te)):
        t = te[k]
        hk = h[k]
        m0 = int(np.ceil(t))
        if m0 < n:
            acc[max(m0, 0)] += hk
        lo = max(0, int(np.ceil(t - W)))
        hi = min(n - 1, int(np.floor(t + W)))
        for m in range(lo, hi + 1):
            tau = m - t
            pos = (tau + W) * OS
            j = int(pos)
            v = tab[j] + (tab[j + 1] - tab[j]) * (pos - j)
            if tau >= 0.0:
                v -= 1.0
            y[m] += hk * v
    s = 0.0
    for m in range(n):
        s += acc[m]
        y[m] += s
    return y


def _steps_to_audio(te, h, n):
    """steps (times in samples, heights) -> band-limited samples; the level before the first step is 0"""
    return _bl_render(np.ascontiguousarray(te, np.float64), np.ascontiguousarray(h, np.float64), int(n), _blep(),
                      BLEP_W, BLEP_OS)


@njit(cache=True)
def _pulse_steps(freq, duty):
    n = len(freq)
    te = np.empty(2 * n + 2)
    h = np.empty(2 * n + 2)
    d = duty[0]
    te[0] = 0.0
    h[0] = 2.0 * (1.0 - d)                   # levels 2 (1 - d) and -2 d: zero mean in every cycle
    k = 1
    high = True
    ph = 0.0
    dt = min(max(freq[0] / 48000.0, 1e-9), 0.45)
    for i in range(n):
        p1 = ph + dt
        if high and ph < d <= p1:
            te[k] = i + (d - ph) / dt
            h[k] = -2.0
            k += 1
            high = False
        if p1 >= 1.0:                        # a new period: pitch and duty written since take effect here, so every
            fr = (1.0 - ph) / dt             # cycle is whole (no lopsided half-cycle thump on arpeggio steps)
            nd = min(max(freq[i] / 48000.0, 1e-9), 0.45)
            te[k] = i + fr
            h[k] = 2.0 + 2.0 * (d - duty[i]) if not high else 2.0 * (d - duty[i])
            k += 1
            high = True
            d = duty[i]
            p1 = (p1 - 1.0) * nd / dt
            dt = nd
            if d <= p1:
                te[k] = i + fr + d / dt
                h[k] = -2.0
                k += 1
                high = False
        ph = p1
    return te[:k], h[:k]


@njit(cache=True)
def _stair_steps(freq, table, cap):
    K = len(table)
    te = np.empty(cap)
    h = np.empty(cap)
    lev = table[0]
    te[0] = 0.0
    h[0] = lev
    k = 1
    ph = 0.0
    step = min(freq[0] / 48000.0, 0.45) * K
    for i in range(len(freq)):
        p1 = ph + step
        b = np.floor(ph) + 1.0
        while b <= p1 and k < cap:
            nl = table[int(b) % K]
            if nl != lev:
                te[k] = i + (b - ph) / step
                h[k] = nl - lev
                k += 1
                lev = nl
            b += 1.0
        if p1 >= K:                      # a new period: the new pitch takes effect (whole cycles, as above)
            ns = min(freq[i] / 48000.0, 0.45) * K
            p1 = (p1 - K) * ns / step
            step = ns
        ph = p1
    return te[:k], h[:k]


@njit(cache=True)
def _lfsr_steps(clock, tap, cap):
    te = np.empty(cap)
    h = np.empty(cap)
    reg = 1
    lev = -1.0                                       # bit 0 set: the channel is low
    te[0] = 0.0
    h[0] = lev
    k = 1
    ph = 0.0
    for i in range(len(clock)):
        step = max(clock[i] / 48000.0, 1e-9)
        p1 = ph + step
        b = 1.0
        while b <= p1 and k < cap:
            fb = (reg ^ (reg >> tap)) & 1
            reg = (reg >> 1) | (fb << 14)
            nl = 1.0 if (reg & 1) == 0 else -1.0
            if nl != lev:
                te[k] = i + (b - ph) / step
                h[k] = nl - lev
                k += 1
                lev = nl
            b += 1.0
        ph = p1 - np.floor(p1)
    return te[:k], h[:k]


@njit(cache=True)
def _delta_steps(target, tick):
    te = np.empty(len(target))
    h = np.empty(len(target))
    c = 64.0
    k = 0
    for i in range(len(target)):
        d = target[i] - c
        s = 2.0 if d >= 2.0 else (-2.0 if d <= -2.0 else 0.0)      # within one step: hold (no idle whine)
        if s != 0.0 and 0.0 <= c + s <= 127.0:
            c += s
            te[k] = tick[i]
            h[k] = s / 63.5
            k += 1
    return te[:k], h[:k]


# ------------------------------------------------------------------------------------------------ channels
def pulse(freq, duty=0.5):
    """band-limited pulse, peak-to-peak 2 and zero mean (levels 2 (1 - duty) and -2 duty): freq (Hz) and duty are
    per-sample arrays (duty may be a scalar); both take effect at the next period, as the chips' timers reload.
    Duty 0.125 / 0.25 / 0.5 / 0.75 are the hardware's; 0.75 sounds like 0.25 inverted."""
    f = np.ascontiguousarray(freq, np.float64)
    d = np.ascontiguousarray(np.clip(np.broadcast_to(duty, f.shape), 0.01, 0.99), np.float64)
    te, h = _pulse_steps(f, d)
    return _steps_to_audio(te, h, len(f))


def stair(freq, table=TRIANGLE):
    """a band-limited stepped wave: `table` levels (0..15) per period, played at freq (Hz per sample), scaled to
    about +-1 around the table's mean. TRIANGLE gives the 16-level staircase triangle: its stair images sit near 32 x
    the pitch, a soft buzz on low notes and above Nyquist (removed, not folded) on high ones."""
    f = np.ascontiguousarray(freq, np.float64)
    tb = np.asarray(table, np.float64)
    tb = (tb - tb.mean()) / 7.5
    cap = int(np.sum(np.minimum(f, SR * 0.45)) / SR * len(tb)) + len(tb) + 8
    te, h = _stair_steps(f, tb, cap)
    return _steps_to_audio(te, h, len(f))


def snap_clock(clock):
    """the nearest of the 16 hardware noise clock rates (log distance), per sample"""
    c = np.asarray(clock, np.float64)
    lc = np.log(np.asarray(NOISE_CLOCKS))
    i = np.abs(np.log(np.maximum(c, 1.0))[..., None] - lc).argmin(-1)
    return np.asarray(NOISE_CLOCKS)[i]


def lfsr(clock, mode='long'):
    """band-limited shift-register noise, +-1: a 15-bit register clocked at `clock` Hz (per sample), feedback bit 0 xor
    bit 1 ('long': a 32767-step sequence, hiss) or bit 0 xor bit 6 ('short': 93 steps, a metallic tone at clock/93)"""
    c = np.ascontiguousarray(clock, np.float64)
    cap = int(np.sum(c) / SR) + 8
    te, h = _lfsr_steps(c, 6 if mode == 'short' else 1, cap)
    return _steps_to_audio(te, h, len(c))


def delta(x, rate=16884.6):
    """delta-coded sample-and-hold, the grit of the console's sample channel: x (about +-1) is read at `rate` Hz and
    followed by a 7-bit counter that can only step +-2 per tick, so it lags fast edges (slope overload) and rounds to
    64 levels; the held staircase is rendered band-limited"""
    x = np.asarray(x, np.float64)
    tick = np.arange(0, len(x) - 1, SR / rate)
    target = 64.0 + 63.0 * np.clip(np.interp(tick, np.arange(len(x)), x), -1, 1)
    te, h = _delta_steps(np.ascontiguousarray(target), np.ascontiguousarray(tick))
    return _steps_to_audio(te, h, len(x))


def _smooth(c, ms=1.0):
    """a control signal with its jumps rounded off over `ms` (Hann kernel): volume and duty steps zip, not click"""
    c = np.asarray(c, np.float64)
    k = max(3, int(ms * 1e-3 * SR) | 1)
    if len(c) <= k:
        return c
    w = np.hanning(k + 2)[1:-1]
    pad = np.concatenate([np.full(k, c[0]), c, np.full(k, c[-1])])
    return np.convolve(pad, w / w.sum(), 'same')[k:-k]


# ------------------------------------------------------------------------------------------------ one voice
def _freq(p, t):
    """pitch trajectory (Hz per sample): base * 2^((slide + arpeggio + vibrato) / 12)"""
    ts = t % (1.0 / p['repeat']) if p['repeat'] > 0 else t
    if p['slide_end'] is not None:
        ts = np.minimum(ts, p['slide_end'])
    semi = p['slide'] * ts + 0.5 * p['slide_accel'] * ts ** 2
    arp = np.asarray(p['arp'], np.float64)
    if len(arp):
        i = (t / max(p['arp_step'], 1e-4) + 1e-9).astype(np.int64)
        semi = semi + arp[i % len(arp) if p['arp_loop'] else np.minimum(i, len(arp) - 1)]
    if p['vib_depth']:
        ramp = np.clip((t - p['vib_delay']) / 0.08, 0, 1)
        semi = semi + p['vib_depth'] * ramp * np.sin(2 * np.pi * p['vib_rate'] * t)
    return p['freq'] * 2 ** (semi / 12)


def _env(p, t):
    """attack (linear), sustain (1 + punch falling to 1), decay (linear to 0), peak 1; with steps the level is held
    at 16 volume steps that change at 240 Hz (each change rounded over 1 ms)"""
    A, S, D, P = max(p['attack'], 0.0), max(p['sustain'], 0.0), max(p['decay'], 1e-4), p['punch']
    tq = np.floor(t * ENV_RATE) / ENV_RATE if p['steps'] else t
    e = np.where(tq < A, tq / max(A, 1e-9),
                 np.where(tq < A + S, 1 + P * (1 - (tq - A) / max(S, 1e-9)), np.maximum(0, 1 - (tq - A - S) / D)))
    e = e / (1 + P)
    if p['steps']:
        e = _smooth(np.round(e * 15) / 15)
    return e


def _table(p):
    tb = p['table']
    return WAVES[tb] if isinstance(tb, str) else tuple(int(v) for v in tb)


def voice(p):
    """render one voice dict (keys and units in DEFAULTS; missing keys take the defaults) to mono at 48 kHz"""
    p = dict(DEFAULTS, **p)
    n = dsp.n_of(max(p['attack'] + p['sustain'] + p['decay'], 0.002))
    t = np.arange(n) / SR
    f = _freq(p, t)
    end_fade = 0.001
    if p['freq_min'] > 0 and np.any(f < p['freq_min']):
        n = max(dsp.n_of(0.004), int(np.argmax(f < p['freq_min'])))
        t, f, end_fade = t[:n], f[:n], 0.003
    f = np.clip(f, 10.0, SR * 0.45)
    w = p['wave']
    if w == 'pulse':
        d = p['duty'] + p['duty_sweep'] * (np.floor(t / FRAME) * FRAME if p['duty_steps'] else t)
        d = np.clip(d, 0.02, 0.98)
        if p['duty_steps']:
            d = np.asarray(DUTIES)[np.abs(d[:, None] - np.asarray(DUTIES)).argmin(1)]
        y = pulse(f, d)
    elif w == 'triangle':
        y = stair(f, TRIANGLE)
    elif w == 'wave':
        y = stair(f, _table(p))
    elif w == 'noise':
        c = np.clip(f, 20.0, 450000.0)
        y = lfsr(snap_clock(c) if p['noise_steps'] else c, p['noise_mode'])
    else:
        raise ValueError(f'chip: unknown wave {w!r} (pulse, triangle, noise, wave)')
    y = y * _env(p, t)
    if p['grit'] > 0:
        y = (1 - p['grit']) * y + p['grit'] * delta(y, p['grit_rate'])
    if p['lpf'] > 0:
        y = dsp.svf(y, p['lpf'] * 2 ** (p['lpf_sweep'] * t), p['lpf_res'], 'low')
    if p['hpf'] > 0:
        y = dsp.svf(y, p['hpf'] * 2 ** (p['hpf_sweep'] * t), 0.707, 'high')
    if p['phaser'] > 0:
        dl = np.clip(p['phaser'] + p['phaser_sweep'] * t, 0.0, 20.0) * 1e-3 * SR
        y = y - p['phaser_mix'] * np.interp(np.arange(n) - dl, np.arange(n), y, left=0.0)
    return dsp.fade(y, 0.0005, end_fade) * 10 ** (p['gain'] / 20)


# ------------------------------------------------------------------------------------------------ the whole sound
def _voices(x):
    if isinstance(x, dict):
        x = [x]
    return [dict(DEFAULTS, **v) for v in x]


def _slap(y, polish):
    """a quiet slapback off a near wall: 45 / 58 ms (left / right, so it also widens a little), then its echo, dull"""
    g = 10 ** ((-28 + 10 * polish) / 20)
    out = np.concatenate([y, np.zeros((dsp.n_of(0.2), 2))])
    for ch, d in ((0, 0.045), (1, 0.058)):
        e = dsp.butter(dsp.butter(y[:, ch], 'low', 4500), 'high', 250)
        for k, gk in ((1, g), (2, g * 0.35)):
            dsp.place(out[:, ch], e, d * k, gk)
    return out


def _fold(y, L, X=0.05):
    """make y[:L] a seamless loop: its head is crossfaded (equal power) with what followed L"""
    nL, nX = dsp.n_of(L), dsp.n_of(X)
    u = np.linspace(0, 1, nX)[:, None]
    out = y[:nL].copy()
    out[:nX] = y[:nX] * np.sin(u * np.pi / 2) + y[nL:nL + nX] * np.cos(u * np.pi / 2)
    return out


def mix(voices, polish=0.5, bits=None):
    """voices (a dict or a list of dicts) -> stereo (n, 2): placed at their 't', panned, through an optional `bits`
    output (a DAC of that depth), then the output stage and finish (see render)"""
    vs = _voices(voices)
    loop = max(v['loop'] for v in vs)
    ys = [(voice(v), v) for v in vs]
    n = max(dsp.n_of(v['t']) + len(y) for y, v in ys)
    if loop:
        n = max(n, dsp.n_of(loop + 0.06))
    y = np.zeros((n, 2))
    for yi, v in ys:
        dsp.place(y, dsp.pan_to_stereo(yi, v['pan']), v['t'])
    if bits:
        q = 2 ** (int(bits) - 1) - 1
        pk = np.abs(y).max() + 1e-12
        y = np.round(y / pk * q) / q * pk
    if polish > 0:
        y = np.concatenate([y, np.zeros((dsp.n_of(0.03), 2))])
        y = dsp.butter(dsp.butter(y, 'high', 40.0), 'low', 13000 - 4000 * polish)
        if polish >= 0.2:
            y = _slap(y, polish)
    if loop:
        y = _fold(y, loop)
    else:
        y = dsp.fade(dsp.trim_tail(y, -80.0, 0.004), 0.0004, 0.001)
    if polish > 0:
        y = y * 10 ** ((LOUD_LUFS - lufs_max(y)) / 20)
        pk = np.abs(y).max()
        return y * (PEAK / pk) if pk > PEAK else y
    return y * (PEAK / (np.abs(y).max() + 1e-12))


def render(name_or_params, r=None, pitch=None, dur=None, bits=None, polish=0.5):
    """one chip sound as stereo float64 (n, 2) at 48 kHz (dual mono plus a slight slapback width).
    name_or_params: a SOUNDS name (drawn with variation seed / rng r, default 1), or a voice dict, or a list of them.
    pitch: Hz or a note name ('E6'); the key or start note, snapped to a note (names only). dur: seconds; the drawn
    timing is stretched to it (names only; for 'engine' the loop length). bits: output DAC depth (None = off).
    polish 0..1: 0 is the raw generator peaked at PEAK; above 0 the console output stage (40 Hz high-pass,
    13 - 4 x polish kHz low-pass), a slapback at -28 + 10 x polish dB from 0.2 up, and the level set to LOUD_LUFS
    momentary-max (peak at most PEAK): every generator plays at one loudness."""
    if isinstance(name_or_params, str):
        if name_or_params not in SOUNDS:
            raise KeyError(f'chip: no sound {name_or_params!r}; one of {", ".join(SOUNDS)}')
        name_or_params = GENERATORS[name_or_params](1 if r is None else r, pitch, dur)
    return mix(name_or_params, polish, bits)


# ------------------------------------------------------------------------------------------------ helpers for drawing
_NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def midi(pitch):
    """the nearest MIDI note of a frequency in Hz or a name like 'C#5', 'Eb4', 'A4' (= 69)"""
    if isinstance(pitch, str):
        m = re.fullmatch(r'([A-Ga-g])([#b]?)(-?\d)', pitch.strip())
        if not m:
            raise ValueError(f'chip: cannot read the note {pitch!r} (like C5, F#4, Bb3)')
        return 12 * (int(m.group(3)) + 1) + _NOTE[m.group(1).upper()] + {'#': 1, 'b': -1, '': 0}[m.group(2)]
    return int(round(69 + 12 * np.log2(float(pitch) / 440.0)))


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _r(seed):
    return seed if isinstance(seed, np.random.Generator) else dsp.rng(seed)


def _base(r, pitch, lo, hi):
    m = int(r.integers(lo, hi + 1))          # drawn even when pitch is given, so the other draws do not shift
    return midi(pitch) if pitch is not None else m


def _pick(r, seq, w=None):
    w = None if w is None else np.asarray(w, float) / np.sum(w)
    return seq[int(r.choice(len(seq), p=w))]


def _frames(k):
    return max(1, int(round(k))) * FRAME


def _v(**kw):
    return dict(DEFAULTS, **kw)


def _length(vs):
    return max(v['t'] + v['attack'] + v['sustain'] + v['decay'] for v in _voices(vs))


def _fit(vs, dur):
    """stretch a drawn sound to `dur` s: times scale by k, rates by 1/k (the same travel in the new time)"""
    if dur is None:
        return vs
    k = float(np.clip(dur / _length(vs), 0.2, 5.0))
    out = []
    for v in vs:
        v = dict(v)
        for key in _TIME_KEYS:
            v[key] *= k
        for key in _RATE_KEYS:
            v[key] /= k
        v['slide_accel'] /= k * k
        if v['slide_end'] is not None:
            v['slide_end'] *= k
        out.append(v)
    return out


def _glide(travel, seconds, settle=0.25):
    """slide and slide_accel (st/s, st/s^2) that travel `travel` semitones in `seconds` and slow to `settle` x the
    starting speed by then (a slide that arrives instead of running on like a siren)"""
    v0 = travel / (seconds * (1 + settle) / 2)
    return v0, (settle - 1) * v0 / seconds


def _note(m, t, length, duty=0.5, wave='pulse', last=False, gain=0.0, pan=0.0, **kw):
    """one articulated melody note: plucked (punch, decaying) or, when last, held with a late vibrato and a release"""
    if wave == 'triangle':                      # the triangle channel has no volume control: gate it, 20 ms release
        return _v(wave=wave, freq=hz(m), t=t, sustain=length * (1.0 if last else 0.88), decay=0.03 if not last
                  else 0.2, steps=False, gain=gain, pan=pan, **kw)
    if last:
        return _v(wave=wave, freq=hz(m), t=t, duty=duty, sustain=length * 0.65, decay=length * 0.35 + 0.15,
                  punch=0.35, vib_depth=0.1, vib_rate=6.0, vib_delay=0.12, gain=gain, pan=pan, **kw)
    return _v(wave=wave, freq=hz(m), t=t, duty=duty, sustain=length * 0.55, decay=length * 0.35, punch=0.45,
              gain=gain, pan=pan, **kw)


def _line(notes, m0, beat, duty=0.5, wave='pulse', gain=0.0, pan=0.0, hold_last=True, **kw):
    """a melody: notes [(semitones from m0, beats)] on the frame grid (lengths rounded to whole frames)"""
    out, t = [], 0.0
    for i, (s, b) in enumerate(notes):
        L = _frames(b * beat / FRAME)
        out.append(_note(m0 + s, t, L, duty, wave, hold_last and i == len(notes) - 1, gain, pan, **kw))
        t += L
    return out


def _below(s):
    """the next lower tone of the major triad on 0 (harmony a chord tone under the melody)"""
    pc = s % 12
    return s - pc + {0: -5, 4: 0, 7: 4}.get(pc, pc - 3)


# ------------------------------------------------------------------------------------------------ generators
def coin(seed=1, pitch=None, dur=None):
    """coin: two pulse notes a fourth, fifth or octave apart, the first 4-5 frames, the second ringing out"""
    r = _r(seed)
    m = _base(r, pitch, 79, 86)
    iv = _pick(r, (5, 7, 12), (0.5, 0.25, 0.25))
    first = _frames(r.integers(4, 6))
    v = _v(freq=hz(m), arp=(0, iv), arp_step=first, duty=_pick(r, (0.5, 0.25), (0.6, 0.4)),
           sustain=first + r.uniform(0.02, 0.05), decay=r.uniform(0.26, 0.42), punch=r.uniform(0.3, 0.5))
    return _fit([v], dur)


def jump(seed=1, pitch=None, dur=None):
    """jump: a pulse rising 10-16 semitones in 0.12-0.2 s and settling there (not a siren), then decaying"""
    r = _r(seed)
    m = _base(r, pitch, 55, 64)
    te = r.uniform(0.12, 0.2)
    s, a = _glide(r.uniform(10, 16), te)
    duty = _pick(r, (0.25, 0.5, 0.125), (0.45, 0.35, 0.2))
    sweep = r.uniform(0.6, 1.5) if (duty < 0.5 and r.random() < 0.4) else 0.0
    v = _v(freq=hz(m), slide=s, slide_accel=a, slide_end=te, duty=duty, duty_sweep=sweep,
           sustain=r.uniform(0.07, 0.12), decay=r.uniform(0.09, 0.16), punch=r.uniform(0.1, 0.3), hpf=120)
    return _fit([v], dur)


def double_jump(seed=1, pitch=None, dur=None):
    """double jump: a quick thin chirp and, 4-5 frames later, a second one starting higher (the flip in the air)"""
    r = _r(seed)
    m = _base(r, pitch, 60, 67)
    vs = []
    for i, (st, tr, t0) in enumerate(((0, r.uniform(7, 10), 0.0), (r.choice([5, 7]), r.uniform(9, 13),
                                                                       _frames(r.integers(4, 6))))):
        te = r.uniform(0.07, 0.1) * (1 + 0.4 * i)
        s, a = _glide(tr, te)
        vs.append(_v(freq=hz(m + st), slide=s, slide_accel=a, slide_end=te, duty=0.125 if i == 0 else 0.25, t=t0,
                     sustain=te * 0.8, decay=r.uniform(0.06, 0.1) * (1 + i), punch=0.3, gain=-3.0 * (1 - i),
                     hpf=150))
    return _fit(vs, dur)


def laser(seed=1, pitch=None, dur=None):
    """laser: a thin pulse falling 2-3 octaves fast and settling, duty widening as it falls, high-passed"""
    r = _r(seed)
    m = _base(r, pitch, 84, 95)
    te = r.uniform(0.1, 0.22)
    s, a = _glide(-r.uniform(24, 34), te, 0.3)
    v = _v(freq=hz(m), slide=s, slide_accel=a, slide_end=te, duty=_pick(r, (0.125, 0.25)),
           duty_sweep=r.uniform(1.5, 3.0), sustain=r.uniform(0.03, 0.07), decay=te * r.uniform(0.7, 1.0),
           punch=r.uniform(0.3, 0.6), hpf=r.uniform(250, 500))
    if r.random() < 0.35:
        v.update(phaser=r.uniform(0.6, 1.5), phaser_sweep=r.uniform(4, 10), phaser_mix=0.5)
    return _fit([v], dur)


def explosion(seed=1, pitch=None, dur=None):
    """explosion: long-mode noise stepping down the hardware clock rates under a closing low-pass, punchy and
    gritty, over a falling triangle thump"""
    r = _r(seed)
    m = _base(r, pitch, 40, 45)                     # the thump: E2-A2
    k = 1.0 if pitch is None else hz(m) / 98.0
    d = r.uniform(0.55, 0.95)
    noise = _v(wave='noise', freq=NOISE_CLOCKS[int(r.integers(4, 7))] * k, slide=-r.uniform(12, 22),
               sustain=r.uniform(0.06, 0.14), decay=d, punch=r.uniform(0.5, 0.8), lpf=r.uniform(6000, 9000),
               lpf_sweep=-r.uniform(1.5, 2.6), grit=r.uniform(0.2, 0.4), grit_rate=DELTA_RATES[int(r.integers(9, 13))])
    if r.random() < 0.3:
        noise.update(vib_depth=r.uniform(1, 3), vib_rate=r.uniform(14, 22))         # a rattling roll
    s, a = _glide(-r.uniform(14, 20), 0.25, 0.4)
    thump = _v(wave='triangle', freq=hz(m), slide=s, slide_accel=a, slide_end=0.25, sustain=0.02,
               decay=r.uniform(0.25, 0.4), steps=False, gain=-2.0)
    return _fit([noise, thump], dur)


def power_up(seed=1, pitch=None, dur=None):
    """power-up: a fast legato arpeggio climbing two octaves on a major triad, fifths-and-octaves or the major
    pentatonic, 1-3 frames a step, the top note held with a little vibrato. The run starts between F3 and C4, so
    its held top note lands between F5 and C6: a square held above C6 is shrill"""
    r = _r(seed)
    m = _base(r, pitch, 53, 60)
    arp, k = _pick(r, (((0, 4, 7, 12, 16, 19, 24), 2), ((0, 7, 12, 19, 24), 3),
                       ((0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24), 1)))
    step = _frames(k + int(r.integers(0, 2)) * (k < 3))
    v = _v(freq=hz(m), arp=arp, arp_step=step, duty=_pick(r, (0.5, 0.25)), duty_sweep=-r.uniform(0.5, 1.0),
           sustain=len(arp) * step + r.uniform(0.06, 0.14), decay=r.uniform(0.15, 0.28), punch=0.2,
           vib_depth=0.1, vib_rate=7.0, vib_delay=len(arp) * step)
    return _fit([v], dur)


def power_down(seed=1, pitch=None, dur=None):
    """power-down: a falling arpeggio (minor triad, fifths, or chromatic), 2-4 frames a step, the duty thinning"""
    r = _r(seed)
    m = _base(r, pitch, 55, 62)
    arp, k = _pick(r, (((24, 19, 15, 12, 7, 3, 0), 3), ((19, 12, 7, 0, -5), 4),
                       ((12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), 2)))
    step = _frames(k)
    v = _v(freq=hz(m), arp=arp, arp_step=step, duty=0.5, duty_sweep=-r.uniform(0.5, 0.9),
           sustain=len(arp) * step, decay=r.uniform(0.18, 0.3), punch=0.15)
    return _fit([v], dur)


def hurt(seed=1, pitch=None, dur=None):
    """hurt: a square dropping fast with a 25-40 Hz wobble (a buzz), on a short noise crack"""
    r = _r(seed)
    m = _base(r, pitch, 62, 70)
    d = r.uniform(0.12, 0.2)
    s, a = _glide(-r.uniform(10, 18), d, 0.5)
    tone = _v(freq=hz(m), slide=s, slide_accel=a, slide_end=d, duty=_pick(r, (0.5, 0.25)),
              sustain=r.uniform(0.02, 0.05), decay=d, punch=0.5, vib_depth=r.uniform(0.6, 1.5),
              vib_rate=r.uniform(25, 40))
    crack = _v(wave='noise', freq=NOISE_CLOCKS[int(r.integers(3, 6))], sustain=0.0, decay=r.uniform(0.04, 0.07),
               punch=0.0, gain=-5.0)
    return _fit([tone, crack], dur)


def blip(seed=1, pitch=None, dur=None):
    """menu move: one short pulse note, C5-B5"""
    r = _r(seed)
    m = _base(r, pitch, 72, 83)
    v = _v(freq=hz(m), duty=_pick(r, (0.5, 0.25)), sustain=r.uniform(0.025, 0.04), decay=r.uniform(0.03, 0.05),
           punch=0.15, hpf=150)
    return _fit([v], dur)


def select(seed=1, pitch=None, dur=None):
    """menu confirm: two or three notes up (a fifth, an octave, or the major triad), 2-3 frames each"""
    r = _r(seed)
    m = _base(r, pitch, 72, 79)
    arp = _pick(r, ((0, 7), (0, 12), (0, 4, 7)))
    step = _frames(r.integers(2, 4))
    v = _v(freq=hz(m), arp=arp, arp_step=step, duty=_pick(r, (0.5, 0.25)), sustain=(len(arp) - 1) * step + 0.03,
           decay=r.uniform(0.07, 0.11), punch=0.3, hpf=150)
    return _fit([v], dur)


def back(seed=1, pitch=None, dur=None):
    """menu back / cancel: two notes down (a fourth or a fifth), softer and shorter than select"""
    r = _r(seed)
    m = _base(r, pitch, 67, 74)
    arp = _pick(r, ((5, 0), (7, 0)))
    step = _frames(r.integers(2, 4))
    v = _v(freq=hz(m), arp=arp, arp_step=step, duty=0.25, sustain=step + 0.02, decay=r.uniform(0.05, 0.08),
           punch=0.2, hpf=150, gain=-1.5)
    return _fit([v], dur)


def pickup(seed=1, pitch=None, dur=None):
    """item pickup: a one-frame-per-note major-triad sparkle up an octave on a thin pulse or a soft 4-bit wave"""
    r = _r(seed)
    m = _base(r, pitch, 76, 84)
    arp = _pick(r, ((0, 4, 7, 12), (0, 7, 12, 16, 19, 24), (0, 4, 7, 12, 16)))
    wave = _pick(r, ('pulse', 'wave'), (0.6, 0.4))
    v = _v(wave=wave, table='soft', freq=hz(m), arp=arp, duty=0.125, sustain=len(arp) * FRAME + 0.02,
           decay=r.uniform(0.09, 0.15), punch=0.3, hpf=150)
    return _fit([v], dur)


ONE_UP_TUNES = ((0, 4, 7, 12, 16, 19), (0, 7, 12, 16, 19, 24), (0, 2, 4, 7, 9, 12), (0, 4, 7, 4, 7, 12),
                (7, 12, 16, 12, 16, 19))


def one_up(seed=1, pitch=None, dur=None):
    """extra life: a short rising jingle of six articulated chord tones, 4-5 frames each, the last one held, with the
    drivers' echo trick (the same line 3 frames late, quieter, on a thinner duty)"""
    r = _r(seed)
    m = _base(r, pitch, 74, 81)
    tune = ONE_UP_TUNES[int(r.integers(len(ONE_UP_TUNES)))]
    beat = _frames(r.integers(4, 6))
    notes = [(s, 1.0) for s in tune[:-1]] + [(tune[-1], 3.0)]
    duty = _pick(r, (0.5, 0.25))
    vs = _line(notes, m, beat, duty)
    vs += [dict(v, t=v['t'] + 3 * FRAME, gain=-11.0, duty=0.125, pan=0.25) for v in _line(notes, m, beat, duty)]
    return _fit(vs, dur)


LEVEL_TUNES = ((0, 4, 7, 12), (4, 7, 12, 16, 19, 24), (0, 2, 4, 7, 12), (7, 12, 16, 19, 24))


def level_up(seed=1, pitch=None, dur=None):
    """level up: a rising chord-tone line (4-6 notes, 0.07-0.1 s) to a held top note, harmonised a chord tone below
    on the second pulse, the root on the triangle under the held note"""
    r = _r(seed)
    m = _base(r, pitch, 67, 74)
    tune = LEVEL_TUNES[int(r.integers(len(LEVEL_TUNES)))]
    beat = _frames(r.uniform(4.2, 6.0))
    hold = r.uniform(4.5, 6.5)
    notes = [(s, 1.0) for s in tune[:-1]] + [(tune[-1], hold)]
    vs = _line(notes, m, beat, 0.5, pan=-0.15)
    vs += _line([(_below(s), b) for s, b in notes], m, beat, 0.25, gain=-5.0, pan=0.15)
    t_last = vs[len(notes) - 1]['t']
    vs.append(_note(m - 24, t_last, hold * beat, wave='triangle', last=True, gain=-1.0))
    return _fit(vs, dur)


# fanfares: (melody, harmony or None = a chord tone below, bass) as [(semitones from the tonic, beats)]
WIN_TUNES = (
    ([(7, .5), (12, .5), (16, .5), (19, .5), (24, 2.5)], None, [(-24, 2.0), (-17, 0.5), (-24, 2.0)]),
    ([(9, .75), (9, .25), (11, .75), (11, .25), (12, 2.5)], [(5, .75), (5, .25), (7, .75), (7, .25), (7, 2.5)],
     [(-19, 1.0), (-17, 1.0), (-24, 2.5)]),
    ([(0, 1 / 3), (4, 1 / 3), (7, 1 / 3), (12, 1 / 3), (16, 1 / 3), (19, 1 / 3), (24, 2.5)], None,
     [(-24, 1.0), (-17, 1.0), (-24, 2.5)]),
    ([(0, .5), (4, .25), (7, .25), (12, .5), (7, .25), (12, .25), (16, 2.5)], None,
     [(-24, 1.0), (-20, 0.5), (-17, 0.5), (-24, 2.5)]),
)
LOSE_TUNES = (       # (melody, harmony offset in semitones, bass)
    ([(7, 1), (6, 1), (5, 1), (4, 3)], -3, [(-17, 1), (-18, 1), (-19, 1), (-20, 3)]),
    ([(12, .5), (7, .5), (3, .5), (2, .5), (0, 2.5)], -12, [(-24, 2), (-28, 1), (-29, 1.5)]),
    ([(4, .75), (3, .75), (2, .75), (1, .75), (0, 2.5)], -4, [(-24, 1.5), (-25, 1.5), (-29, 2.5)]),
)


def _score(r, m, tune, seconds, duty_sweep=0.0):
    mel, har, bass = tune
    beat = seconds / sum(b for _, b in mel)
    vs = _line(mel, m, beat, _pick(r, (0.5, 0.25)), pan=-0.18)
    vs[-1]['duty_sweep'] = duty_sweep
    if isinstance(har, int):
        har = [(s + har, b) for s, b in mel]
    vs += _line(har or [(_below(s), b) for s, b in mel], m, beat, 0.25, gain=-6.0, pan=0.18)
    vs += _line(bass, m, beat, wave='triangle', gain=-3.0)
    return vs


def win(seed=1, pitch=None, dur=None):
    """victory fanfare (1.3-2 s): a major cadence on two pulse channels (melody and a chord tone below) over a
    triangle bass, frame-locked, the last chord held with vibrato"""
    r = _r(seed)
    m = _base(r, pitch, 67, 74)
    tune = WIN_TUNES[int(r.integers(len(WIN_TUNES)))]
    return _fit(_score(r, m, tune, r.uniform(1.15, 1.5)), dur)


def lose(seed=1, pitch=None, dur=None):
    """game over / fail (1.3-2.2 s): a falling chromatic or minor phrase, parallel harmony, triangle bass under it;
    the last note's duty steps down (the 'wah' of a muted horn, done with the waveform)"""
    r = _r(seed)
    m = _base(r, pitch, 62, 69)
    tune = LOSE_TUNES[int(r.integers(len(LOSE_TUNES)))]
    return _fit(_score(r, m, tune, r.uniform(1.3, 1.8), duty_sweep=-r.uniform(0.8, 1.4)), dur)


TEXT_STEPS = (0, 2, 4, 7, 9)


def text_blip(seed=1, pitch=None, dur=None):
    """dialogue tick: a 45-65 ms note, a pentatonic step from A4-E5 (seeds vary the step, so a line of text chatters
    in tune)"""
    r = _r(seed)
    m = _base(r, pitch, 69, 76)
    m += 0 if pitch is not None else _pick(r, TEXT_STEPS)
    wave = _pick(r, ('pulse', 'wave'), (0.75, 0.25))
    v = _v(wave=wave, table='soft', freq=hz(m), duty=0.5, sustain=r.uniform(0.024, 0.032),
           decay=r.uniform(0.022, 0.032), punch=0.1, hpf=200)
    return _fit([v], dur)


def door(seed=1, pitch=None, dur=None):
    """dungeon door: a short-mode noise grind (metallic, 50-95 Hz) creaking with a stepped wobble, then a thud of
    low noise and a falling triangle"""
    r = _r(seed)
    m = _base(r, pitch, 38, 43)
    grind_s = r.uniform(0.16, 0.28)
    grind = _v(wave='noise', noise_mode='short', freq=NOISE_CLOCKS[int(r.integers(8, 11))], attack=0.02,
               sustain=grind_s, decay=0.04, vib_depth=r.uniform(2, 4), vib_rate=r.uniform(9, 14), lpf=3000,
               gain=-3.0)
    t1 = 0.02 + grind_s
    thud = _v(wave='noise', freq=NOISE_CLOCKS[int(r.integers(10, 13))], t=t1, sustain=0.01,
              decay=r.uniform(0.12, 0.2), punch=0.8, lpf=1300)
    s, a = _glide(-10, 0.12, 0.4)
    low = _v(wave='triangle', freq=hz(m), t=t1, slide=s, slide_accel=a, slide_end=0.12, sustain=0.02,
             decay=r.uniform(0.12, 0.18), steps=False)
    return _fit([grind, thud, low], dur)


def step(seed=1, pitch=None, dur=None):
    """footstep: a 60-100 ms burst of long-mode noise (seeds pick the clock: different surfaces of one floor) over a
    tiny falling triangle knock"""
    r = _r(seed)
    m = _base(r, pitch, 44, 51)
    tick = _v(wave='noise', freq=NOISE_CLOCKS[int(r.integers(5, 9))], sustain=r.uniform(0.015, 0.025),
              decay=r.uniform(0.05, 0.08), punch=0.1, lpf=r.uniform(3500, 6000), hpf=150)
    knock = _v(wave='triangle', freq=hz(m), slide=-60, sustain=0.0, decay=0.04, steps=False, gain=-10.0)
    return _fit([tick, knock], dur)


def splash(seed=1, pitch=None, dur=None):
    """water splash: a bright noise burst closing down, a rising 'bloop' on the 4-bit sine and two little drips"""
    r = _r(seed)
    m = _base(r, pitch, 64, 71)
    burst = _v(wave='noise', freq=NOISE_CLOCKS[int(r.integers(1, 4))], attack=0.003, sustain=r.uniform(0.02, 0.04),
               decay=r.uniform(0.32, 0.48), punch=0.6, lpf=9000, lpf_sweep=-2.2, hpf=300)
    vs = [burst, _v(wave='wave', table='sine', freq=hz(m), slide=60, slide_end=0.05, sustain=0.03, decay=0.06,
                    steps=False, gain=-5.0)]
    for _ in range(2):
        vs.append(_v(wave='wave', table='sine', freq=hz(m + int(r.integers(12, 20))), t=r.uniform(0.12, 0.32),
                     slide=100, sustain=0.01, decay=0.04, steps=False, gain=-14.0, pan=r.uniform(-0.3, 0.3)))
    return _fit(vs, dur)


def alarm(seed=1, pitch=None, dur=None):
    """alarm: a two-tone pulse (a fourth or a fifth, hi-lo) switching every 6-9 frames, 4-6 whole cycles"""
    r = _r(seed)
    m = _base(r, pitch, 76, 83)
    iv = _pick(r, (5, 7))
    step = _frames(r.integers(6, 10))
    cyc = int(r.integers(4, 7)) if dur is None else max(1, int(round(dur / (2 * step))))
    v = _v(freq=hz(m), arp=(iv, 0), arp_step=step, arp_loop=True, duty=_pick(r, (0.5, 0.25)), attack=0.004,
           sustain=cyc * 2 * step - 0.05, decay=0.045)
    return [v]


def engine(seed=1, pitch=None, dur=None):
    """engine (a seamless loop, `dur` its length, default 1.2-1.8 s): a 12.5 / 25 % pulse at 45-70 Hz with a piston
    wobble (vibrato a whole number of cycles per loop) and delta grit, short-mode noise rattle and a triangle sub"""
    r = _r(seed)
    m = _base(r, pitch, 30, 37)
    L = r.uniform(1.2, 1.8) if dur is None else float(dur)
    rate = max(1, round(r.uniform(9, 14) * L)) / L
    n = L + 0.1
    vs = [_v(freq=hz(m), duty=_pick(r, (0.125, 0.25)), sustain=n, decay=0.0, vib_depth=r.uniform(0.5, 1.0),
             vib_rate=rate, grit=0.3, grit_rate=DELTA_RATES[8], lpf=2500, loop=L),
          _v(wave='noise', noise_mode='short', freq=NOISE_CLOCKS[int(r.integers(9, 12))], sustain=n, decay=0.0,
             vib_depth=1.0, vib_rate=rate, lpf=1800, gain=-9.0),
          _v(wave='triangle', freq=hz(m - 12), sustain=n, decay=0.0, steps=False, vib_depth=0.5, vib_rate=rate,
             gain=-3.0)]
    return vs


def teleport(seed=1, pitch=None, dur=None):
    """teleport / warp: rising chirps retriggered 14-20 times a second, each chirp a step higher (chord tones or a
    whole-tone run), through a sweeping phaser"""
    r = _r(seed)
    m = _base(r, pitch, 60, 67)
    rate = r.uniform(14, 20)
    arp = _pick(r, ((0, 4, 7, 12, 16, 19, 24, 28, 31), (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)))
    v = _v(freq=hz(m), repeat=rate, slide=r.uniform(5, 7) * rate, arp=arp, arp_step=1 / rate, duty=0.25,
           attack=0.02, sustain=len(arp) / rate, decay=r.uniform(0.15, 0.25), phaser=1.5,
           phaser_sweep=r.uniform(5, 9), phaser_mix=0.5)
    return _fit([v], dur)


def charge(seed=1, pitch=None, dur=None):
    """charging shot: a pulse climbing 1.5-2 octaves and slowing as it fills (duty stepping 50 -> 12.5 %), a triangle
    an octave under it, then a fast shimmer vibrato on the held top note"""
    r = _r(seed)
    m = _base(r, pitch, 48, 55)
    tc = r.uniform(0.6, 0.95)
    s, a = _glide(r.uniform(19, 26), tc, 0.2)
    hold = r.uniform(0.2, 0.35)
    common = dict(slide=s, slide_accel=a, slide_end=tc, sustain=tc + hold, decay=0.12, attack=0.03,
                  vib_depth=0.2, vib_rate=r.uniform(12, 16), vib_delay=tc)
    vs = [_v(freq=hz(m), duty=0.5, duty_sweep=-0.6 / tc, **common),
          _v(wave='triangle', freq=hz(m - 12), steps=False, gain=-4.0, **common)]
    return _fit(vs, dur)


GENERATORS = {f.__name__: f for f in (coin, jump, double_jump, laser, explosion, power_up, power_down, hurt, blip,
                                      select, back, pickup, one_up, level_up, win, lose, text_blip, door, step,
                                      splash, alarm, engine, teleport, charge)}

SOUNDS = {    # says: plain descriptions; dur_s: typical length (s); range: lengths over seeds at polish 0.5
    'coin': {'says': ['an 8-bit coin pickup sound', 'a retro video game coin'],
              'dur_s': 0.55, 'range': (0.4, 0.75), 'pitched': True},
    'jump': {'says': ['an 8-bit jump sound', 'a retro video game jump'],
              'dur_s': 0.33, 'range': (0.25, 0.45), 'pitched': False},
    'double_jump': {'says': ['an 8-bit double jump sound', 'a retro video game double jump chirp'],
                     'dur_s': 0.45, 'range': (0.33, 0.58), 'pitched': False},
    'laser': {'says': ['an 8-bit laser shot', 'a retro video game laser zap'],
               'dur_s': 0.3, 'range': (0.2, 0.42), 'pitched': False},
    'explosion': {'says': ['an 8-bit explosion', 'a retro video game explosion'],
                   'dur_s': 0.95, 'range': (0.65, 1.3), 'pitched': False},
    'power_up': {'says': ['an 8-bit power-up sound', 'a retro video game power up arpeggio'],
                  'dur_s': 0.68, 'range': (0.45, 0.95), 'pitched': True},
    'power_down': {'says': ['an 8-bit power-down sound', 'a retro video game falling arpeggio'],
                    'dur_s': 0.74, 'range': (0.55, 0.95), 'pitched': True},
    'hurt': {'says': ['an 8-bit hurt sound', 'a retro video game damage hit'],
              'dur_s': 0.32, 'range': (0.23, 0.42), 'pitched': False},
    'blip': {'says': ['an 8-bit menu blip', 'a retro video game menu cursor beep'],
              'dur_s': 0.19, 'range': (0.15, 0.24), 'pitched': True},
    'select': {'says': ['an 8-bit menu select sound', 'a retro video game confirm beep'],
                'dur_s': 0.3, 'range': (0.22, 0.42), 'pitched': True},
    'back': {'says': ['an 8-bit menu back sound', 'a retro video game cancel beep'],
              'dur_s': 0.25, 'range': (0.19, 0.31), 'pitched': True},
    'pickup': {'says': ['an 8-bit item pickup sound', 'a retro video game item collect sparkle'],
                'dur_s': 0.34, 'range': (0.25, 0.43), 'pitched': True},
    'one_up': {'says': ['an 8-bit extra life jingle', 'a retro video game 1-up sound'],
                'dur_s': 0.9, 'range': (0.72, 1.12), 'pitched': True},
    'level_up': {'says': ['an 8-bit level up jingle', 'a retro video game level up fanfare'],
                  'dur_s': 1.1, 'range': (0.8, 1.5), 'pitched': True},
    'win': {'says': ['an 8-bit victory fanfare', 'a retro video game level complete jingle', 'a chiptune win stinger'],
             'dur_s': 1.65, 'range': (1.3, 2.05), 'pitched': True},
    'lose': {'says': ['an 8-bit game over jingle', 'a retro video game lose sound', 'a sad descending chiptune phrase'],
              'dur_s': 1.9, 'range': (1.45, 2.4), 'pitched': True},
    'text_blip': {'says': ['an 8-bit dialogue text blip', 'a retro video game text scroll tick'],
                   'dur_s': 0.17, 'range': (0.14, 0.21), 'pitched': True},
    'door': {'says': ['an 8-bit door opening sound', 'a retro video game dungeon door'],
              'dur_s': 0.53, 'range': (0.4, 0.7), 'pitched': False},
    'step': {'says': ['an 8-bit footstep', 'a retro video game footstep tick'],
              'dur_s': 0.2, 'range': (0.16, 0.25), 'pitched': False},
    'splash': {'says': ['an 8-bit water splash', 'a retro video game splash'],
                'dur_s': 0.54, 'range': (0.4, 0.7), 'pitched': False},
    'alarm': {'says': ['an 8-bit alarm', 'a retro video game two-tone warning siren'],
               'dur_s': 1.4, 'range': (0.8, 2.1), 'pitched': True},
    'engine': {'says': ['an 8-bit engine loop', 'a retro video game engine rumble'],
                'dur_s': 1.5, 'range': (1.2, 1.8), 'pitched': False, 'loop': True},
    'teleport': {'says': ['an 8-bit teleport sound', 'a retro video game warp effect'],
                  'dur_s': 0.9, 'range': (0.65, 1.2), 'pitched': False},
    'charge': {'says': ['an 8-bit charging sound', 'a retro video game power charge up'],
                'dur_s': 1.35, 'range': (1.0, 1.75), 'pitched': False},
}
