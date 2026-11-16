"""Learned sounds: real recordings in the public domain, kept as a few kilobytes of numbers each and played as new
takes (earshot.synth.resyn). A mouse click, a cork, a whoosh, a reversed cymbal: things whose sound is their detail
(the grain of every band's envelope, the exact shape of the strike), which a handful of modes and a noise envelope
do not reach.

The store is earshot/data/resyn/: one model per recording (`<family>-<member>.npz`, numbers only, never audio),
`index.json` (per model: id, descriptions, jobs, kind, source, whether it is tonal and its strongest partial, the main
onset, its natural length, the axes that measurably act on it, the measured quality of the recording, the owner's
rating where he gave one and whether it is a default choice, the credit) and CREDITS.md. It is built in the lab
(tools/learn_build.py) from recordings whose licence is CC0, each one checked for a full band, a natural decay, a
clear onset and a single event; nothing else may enter it. Each design of this module is one entry of the index; a
model is read from disk once, on its first take.

The shared axes are transforms of the model, so they stay physical:
    size     a thing 0.7 .. 1.4 times as big: every frequency divided and every duration multiplied by it
    bright   spectral tilt, -TILT .. +TILT dB per octave around 1 kHz
    hard     the strikes 6 dB softer and duller .. louder and brighter (models with a strike that answers)
    ring     the ringing bands decay 0.5 .. 2 times as long (models that ring)
    energy   level +-6 dB, and a harder, slightly brighter strike
    pitch    tonal models only: the take moved to the note asked for (at most PITCH semitones)
    dur      the time between onsets warped (onsets stay as sharp), within LENGTH of the natural length
    space    the shared room (earshot.synth.space); width: mid / side, the low end mono (a mono recording is
             spread by decorrelation)
Every take draws fresh noise and a small variation (VARY times the request's vary) from its seed: seeds are
performances of one recording, not copies of it.

    from earshot.sounds import registry
    y, meta = registry.render('click/mouse_warm', seed=3, bright=0.3)
    y, meta = registry.render('whoosh/heavy', dur=1.0, size=0.5)
"""
import json
import os
from functools import lru_cache

import numpy as np

from ..forge import dsp, fx
from ..synth import resyn, space
from .registry import LEARNED, design

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'resyn')
SIZE = (0.7, 1.4)            # size -1 .. 1: a smaller .. bigger thing (log)
TILT = 2.0                   # dB per octave at bright +-1 (about +-8 dB at 16 kHz, +-6 at 125 Hz)
PITCH = 7.0                  # semitones a tonal model is moved at most
VARY = 0.5                   # resyn's vary per unit of the request's vary (0.35 by default: 0.18)
BASS = 250.0                 # Hz under which a take is mono (image())
SPREAD = 0.3                 # decorrelation per unit of width (as earshot.sounds.object spreads its mono layers)
# the length a take may be warped to, as a factor of the model's natural length. A warp stretches the 2 ms grain of
# every band's envelope with it, which a slower or faster performance of the same thing would not do; measured on
# eight models (whooshes, a slide, a cork, a reversed cymbal, a click, a soft impact; three seeds each) as the
# change of the bands' 64-128 Hz envelope modulation over the spread of plain takes: up to about 0.8 dB at 0.6 and
# 1.1 dB at 1.8 (one soft impact 1.5 at 0.6), 1.1 to 2.4 dB at 0.4 and up to 2.4 dB at 2.2: there the grain turns
# into a buzz (squeezed) or a flutter (stretched). A click barely changes length at all: it is mostly strike.
LENGTH = (0.6, 1.8)
# resyn.finish before master.finish, gently and only on short dry sounds (clicks, taps, ticks, pops, knocks under
# 0.6 s: dry()). Measured on 41 models as the registry ships them: on short struck sounds the finish adds 0.6 to
# 6 dB of loudness and some body with the centroid within a few percent, but its limiter slowed the strike of half
# the clicks by 2 to 5 ms, so it runs here without the limiter (lufs -40: master.finish levels afterwards) and with
# half its EQ; a click has nothing under 150 Hz, where a boost only adds the filter's own ring. On air (whooshes,
# swells) it moved one centroid up by two octaves, and the owner heard an airy whoosh get worse: never there.
FINISH = dict(lufs=-40.0, weight=1.5, presence=1.5, air=1.5, drive=3.0)
CREST, CUT = 14.0, 8.0       # dB: present() (a sound of 0.5 s and more, peaks over its loudest 50 ms)
DEFAULT_OFF = 0.6           # prior factor of a model the index marks as not a default (`default`: false)


def dry(e):
    """True for a short dry sound of the index (the ones FINISH is for)"""
    short = e['dur_s'] < 0.6
    return short and (e['kind'] in ('click', 'tap', 'tick', 'pop') or ('knock' in e['uses'] and e['kind'] == 'hit'))


@lru_cache(maxsize=1)
def index():
    """the store's index: {id: entry}"""
    with open(os.path.join(STORE, 'index.json'), encoding='utf-8') as fh:
        return {e['id']: e for e in json.load(fh)}


@lru_cache(maxsize=None)
def model(sid):
    """the model of a learned design, read once"""
    return resyn.load(os.path.join(STORE, index()[sid]['file']))


def credit(sid):
    """the recording a learned design was made from: {title, author, page, licence, licence_url}"""
    return dict(index()[sid]['credit'])


def take(e, c, r):
    """one take of the index entry e under the controls c (module doc)"""
    ax = set(e['axes'])
    energy = c.energy if 'energy' in ax else 0.0
    size = c.lerp('size', *SIZE, 1.0, log=True) if 'size' in ax else 1.0
    bright = TILT * (c.bright if 'bright' in ax else 0.0) + 0.5 * energy
    hard = float(np.clip((c.hard if 'hard' in ax else 0.0) + 0.35 * energy, -1.0, 1.0))
    ring = c.lerp('ring', 0.5, 2.0, 1.0, log=True) if 'ring' in ax else 1.0
    pitch = 0.0
    if e.get('hz'):
        pitch = float(np.clip(12.0 * np.log2(c.hz() / e['hz']), -PITCH, PITCH))
    want = c.length()
    length = 1.0 if want is None else float(np.clip(want / (e['dur_s'] * size), *LENGTH))
    y = resyn.render(model(e['id']), r, length=length, pitch=pitch, size=size, bright=bright, hard=hard, ring=ring,
                     vary=VARY * c.vary)
    if dry(e):
        y = resyn.finish(y, **FINISH)
    y = present(y)
    y = image(y * 10 ** (6.0 * energy / 20), c.lerp('width', 0.0, 2.0, 1.0) if 'width' in ax else 1.0, e)
    return space.room(y, c.room('room'), amount=c.space if 'space' in ax else 0.0, r=r)


def present(y):
    """a take whose few loudest peaks stand more than CREST dB over its loudest 50 ms has those peaks limited (at
    most CUT dB, a 2 ms look-ahead): levelled by its peak, a sound of half a second and more with a crack in it (a
    whip, a door, dice) otherwise plays 20 dB under everything else (registry.LIFT reaches only sounds under 1 s,
    and not those it cannot fold enough). Shorter takes are left to the registry's lift, which folds the spike"""
    if len(y) < dsp.n_of(0.5):
        return y
    k = dsp.n_of(0.05)
    body = 10 * np.log10(np.convolve(np.mean(y ** 2, axis=1), np.ones(k) / k, 'same').max() + 1e-20)
    pk = 20 * np.log10(np.abs(y).max() + 1e-12)
    over = min(pk - body - CREST, CUT)
    if over < 0.5:
        return y
    return fx.limit(y, ceiling_db=pk - over, lookahead=0.002, release=0.03, true_peak=False)


def image(y, w, e):
    """the take's stereo image at width w (0 mono, 1 as recorded, 2 wider): side times w, the low end in the
    middle (under BASS Hz the side is removed: resyn renders a low band's carrier fully correlated with the sign of
    the band's measured coherence, so the lows of a nearly uncorrelated recording come out at +-1, often in
    antiphase, which cancels on a phone speaker); a recording with one channel inverted is turned back first;
    a mono recording, or a stereo one asked to be wider, is spread by decorrelation (SPREAD per unit of width)"""
    y = np.array(y, np.float64)
    if e.get('polarity', 1) < 0:
        y[:, 1] = -y[:, 1]
    mid, side = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])
    side = w * dsp.butter(side, 'high', BASS, order=2)
    st = fx.decorrelate(mid, min(1.0, SPREAD * (max(w - 1.0, 0.0) if e['stereo'] else w)))
    return st + np.stack([side, -side], axis=1)


def _register(e):
    def fn(c, r):
        return take(e, c, r)
    fn.__name__ = e['id'].replace('/', '_')
    fn.__doc__ = f"{e['says'][0]} (learned from \"{e['credit']['title']}\", {e['credit']['licence']})"
    design(e['id'], kind=e['kind'], says=tuple(e['says']), uses=tuple(e['uses']), tags=tuple(e['tags']) + ('real',),
           domains=tuple(e['domains']), dur=e['dur_s'], pitch=e.get('hz'), axes=tuple(e['axes']),
           allows=tuple(e.get('allows', ())), prior=LEARNED * (1.0 if e.get('default', True) else DEFAULT_OFF))(fn)


for _e in index().values():
    _register(_e)
