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
"""
import numpy as np

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


def room(y, name='room', amount=0.0):
    """y (mono or stereo) in the named space, returned stereo and as long as y plus the tail. amount -1..1 moves the
    tail's level (-1 = dry, 0 = the space as designed, +1 = 8 dB more) and its length (x 0.6 .. x 1.6)"""
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
    return fx.reverb(st, dsp.rng(0), wet=wet, stereo=True, ir_seed=SEED, grid=True, **kw)
