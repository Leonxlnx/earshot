"""One shared set of rooms for every made sound, so a family of sounds sits in the same space.

    from earshot.synth import space
    y = space.room(note, 'studio')                 # the space as designed
    y = space.room(note, 'studio', amount=0.5)     # more of it (amount -1 = dry .. 0 = as designed .. 1 = a lot)

A source without a room reads as synthesized; a room that is too long or too loud reads as an effect. The spaces here
are the ones sound designers reach for on short sounds: a booth and a small room are mostly early reflections (they
make a sound "recorded" without an audible tail), the studio and hall have a tail that stays out of the way (pre-delay
so the attack stays dry, a low cut so the room does not rumble, soft walls so it does not hiss), the plate is the
bright dense tail of bells and stingers. Each space is one fixed impulse response (fx.reverb with ir_seed): the same
room for every sound and every take, as a reverb plug-in is.

A room also has a sound of its own. A recording is never a few spectral lines over digital silence: under every
note lies the room's tone and the microphone's noise, far below it. `room` adds that floor (TONE_DB under the peak,
from the first sample until the tail has died, then faded): inaudible at any sensible playback level, and it is what
separates "a recording of an instrument" from "a tone" for earshot's own ears: on vibraphone, marimba and piano notes
and three interface figures the match to their own description rose by 0.6 to 1.1 z and the real-or-synthesized
score from 0.03-0.26 to 0.20-0.66 with a floor at -66 dB; noise that only followed the sound's envelope did less.
The 'dry' space, and amount -1, stay without it.
"""
import numpy as np
from scipy.fft import next_fast_len

from ..forge import dsp, fx

# rt60 s, size 0..1, predelay s, damp 0..1 (soft walls), wet (tail level re the direct sound), lowcut Hz (dry below)
SPACES = {
    'dry': None,
    'booth': dict(rt60=0.16, size=0.1, predelay=0.001, damp=0.75, wet=0.07, early_db=2.0, lowcut=120.0),
    'room': dict(rt60=0.32, size=0.22, predelay=0.004, damp=0.65, wet=0.1, early_db=0.0, lowcut=150.0),
    'studio': dict(rt60=0.55, size=0.38, predelay=0.01, damp=0.55, wet=0.12, early_db=-3.0, lowcut=180.0),
    'hall': dict(rt60=1.7, size=0.8, predelay=0.022, damp=0.45, wet=0.16, early_db=-6.0, lowcut=220.0),
    'plate': dict(rt60=1.3, size=0.45, predelay=0.0, damp=0.25, wet=0.15, early_db=None, lowcut=250.0, diffuse=0.8),
    'cathedral': dict(rt60=4.2, size=1.0, predelay=0.035, damp=0.5, wet=0.2, early_db=-8.0, lowcut=250.0),
}
SEED = 20261115
TONE_DB = -68.0        # the room's own tone and the microphone's noise: RMS under the sound's peak
TONE_END_DB = -60.0    # the floor ends where the sound has fallen this far under its peak


def tone(y, r):
    """y (stereo) with the room's floor under it (module doc): pink noise above 120 Hz, independent per channel, at
    TONE_DB under the peak, from the start until the sound is TONE_END_DB down, then faded over 80 ms and cut"""
    m = np.abs(y).max(1)
    pk = m.max()
    if pk <= 0:
        return y
    live = np.nonzero(m >= pk * 10 ** (TONE_END_DB / 20))[0]
    end = min(len(y), int(live[-1]) + dsp.n_of(0.08))
    out = np.array(y[:end])
    m = next_fast_len(end)                     # pink noise is shaped by an FFT: an awkward length costs 0.2 s
    for ch in range(2):
        nz = dsp.butter(dsp.noise(m / dsp.SR, r, 'pink')[:end], 'high', 120.0)
        nz /= np.sqrt(np.mean(nz ** 2)) + 1e-12            # the high-pass took about 40 % of its energy
        out[:, ch] += dsp.fade(nz, 0.003, 0.08) * pk * 10 ** (TONE_DB / 20)
    k = min(end, dsp.n_of(0.08))
    out[end - k:] *= (np.cos(np.linspace(0, np.pi / 2, k)) ** 2)[:, None]
    return out


def room(y, name='room', amount=0.0, r=None):
    """y (mono or stereo) in the named space, returned stereo: the sound and its tail, ending 80 ms after it has
    fallen 60 dB under its peak (silence at the end of y is not kept). amount -1..1 moves the tail's level (-1 = dry,
    0 = the space as designed, +1 = 8 dB more) and its length (x 0.6 .. x 1.6). r: the generator the room's floor
    is drawn from (default: one fixed by the sound's length)"""
    if name not in SPACES:
        raise ValueError(f'unknown space {name!r}: one of {", ".join(SPACES)}')
    y = np.asarray(y, np.float64)
    st = y if y.ndim == 2 else np.stack([y, y], 1)
    p = SPACES[name]
    if p is None or amount <= -1.0:
        return st
    a = float(np.clip(amount, -1.0, 1.0))
    kw = dict(p)
    wet = kw.pop('wet') * 10 ** (8.0 * a / 20) * (1 + min(a, 0.0))
    kw['rt60'] = kw['rt60'] * 1.6 ** a
    wet_y = fx.reverb(st, dsp.rng(0), wet=wet, stereo=True, ir_seed=SEED, grid=True, **kw)
    return tone(wet_y, r if r is not None else dsp.rng(SEED + len(st)))
