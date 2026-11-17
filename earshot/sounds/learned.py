"""Learned sounds: real recordings in the public domain, kept as a few kilobytes of numbers each and played as new
takes (earshot.synth.resyn). A mouse click, a cork, a footstep, a light switch: things whose sound is their detail
(the grain of every band's envelope, the exact shape of the strike), which a handful of modes and a noise envelope
do not reach.

A design is one THING (`step/concrete`, `switch/lamp`, `keyboard/laptop`); the recordings of that thing are its
members. The seed picks the member: seeds 0, 1, 2 ... walk through the members best first (the index's `walk`: a
recording heard as good first, then the full, long enough ones; a game that plays a footstep with seeds 0, 1, 2
gets three different real footsteps), and every take adds fresh noise and a small variation on top, so seeds past
the last member are new performances of a recording, not copies. Seeds 101 apart play the same member (make re-rolls
a take that fails its audit at seed + 101: another performance of the same recording, not the next recording). A request for a
dark or a bright take (bright at -0.5 or below, 0.5 or above) walks the darker or the brighter half of the members
(by their measured centroid) and tilts it as well. `registry.render` reports the member played in meta['member'].

The store is earshot/data/resyn/: one model per recording (`<family>-<name>-<n>.npz`, numbers only, never audio),
`index.json` (per design: id, descriptions, jobs, kind, whether it is a default choice, the interface jobs it plays
(`roles`), the order its seeds play its members in (`walk`) and its members, each with its file, source recording, credit, main onset, natural length, the axes that
measurably act on it, the measured quality of the recording, a listening rating where there is one) and CREDITS.md.
It is built in the lab (tools/learn_build.py) from recordings whose licence is CC0, each one checked for a full band,
a natural decay, a clear onset and a single event; nothing else may enter it. A model is read from disk once, on its
first take.

The shared axes are transforms of the model, so they stay physical:
    size     a thing 0.7 .. 1.4 times as big: every frequency divided and every duration multiplied by it
    bright   spectral tilt, -TILT .. +TILT dB per octave around 1 kHz (and the darker / brighter members, above)
    hard     the strikes 6 dB softer and duller .. louder and brighter
    ring     the ringing bands decay 0.5 .. 2 times as long
    energy   level +-6 dB, and a harder, slightly brighter strike
    pitch    tonal designs only: the take moved to the note asked for (at most PITCH semitones)
    dur      the time between onsets warped (onsets stay as sharp), within LENGTH of the natural length
    space    the shared room (earshot.synth.space); width: mid / side (a mono recording is spread by decorrelation)

A thing a person does several times in a row (footsteps, a knock, typing, coins, a bunch of keys, clinking glasses)
can also be played as that action (`perform`): a recording of the whole action where the store has one (a member
with `events` over 1), else its single recordings sequenced in the thing's `pattern` by earshot.synth.perform (walk,
knock, type, scatter, shake, pair). A plain take never plays a whole-action member of a thing that also has single
ones; the request decides between the two (earshot.brief.performance, make's `events`).

    from earshot.sounds import registry
    y, meta = registry.render('step/pebbles', seed=1)            # meta['member']: which recording was played
    y, meta = registry.render('switch/lamp', seed=0, bright=-0.6)
    y, meta = learned.perform('step/pebbles', seed=0)            # a walk; meta['performance']: what was played
"""
import json
import os
import zlib
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from ..forge import dsp, fx
from ..synth import resyn, space
from .registry import LEARNED, design

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'resyn')
SIZE = (0.7, 1.4)            # size -1 .. 1: a smaller .. bigger thing (log)
TILT = 2.0                   # dB per octave at bright +-1 (about +-8 dB at 16 kHz, +-6 at 125 Hz)
PITCH = 7.0                  # semitones a tonal model is moved at most
VARY = 0.5                   # resyn's vary per unit of the request's vary (0.35 by default: 0.18)
SPREAD = 0.3                 # decorrelation per unit of width (as earshot.sounds.object spreads its mono layers)
CHARACTER = 0.5              # |bright| from which a take is drawn from the darker or the brighter members
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
# swells) it moved one centroid up by two octaves, and an airy whoosh was heard to get worse: never there.
FINISH = dict(lufs=-40.0, weight=1.5, presence=1.5, air=1.5, drive=3.0)
# what follows the strike of a short dry member raised against its loudest 5 ms (body()): gain (level / loudest)
# ** -BODY_EXP, at most BODY_DB. The registry's lift raises a click until its loudest 5 ms sit 3.5 dB under the peak,
# and no further can it go without clipping the strike: a click's energy lies in its first 10-20 ms, so its loudest
# 50 ms stay 7-9 dB under its loudest 5 ms where an interface figure's stay 2 dB under. Measured on the six interface
# members (three seeds): loudest 50 ms +0.5 to +0.9 dB, the loudest 5 ms within 0.2 dB, centroid -1 to -5 %
BODY_EXP, BODY_DB = 0.5, 6.0
CREST, CUT = 14.0, 8.0       # dB: present() (a sound of 0.5 s and more, peaks over its loudest 50 ms)
DEFAULT_OFF = 0.6           # prior factor of a design the index marks as not a default (`default`: false)
RETAKE = 101                # make.take's step between the seeds it tries: seeds this far apart play one member
# how far (a share of its length) a length asked of a performance may lie from a whole-action recording's own for
# that recording to be played (warped that little): further, the pattern fills the length at the natural pace, as a
# warp would slow or hurry the walk, the knock, the typing it holds
WHOLE_FIT = 0.15


def dry(e, m):
    """True for a short dry member (the ones FINISH is for)"""
    short = m['dur_s'] < 0.6
    return short and (e['kind'] in ('click', 'tap', 'tick', 'pop') or ('knock' in e['uses'] and e['kind'] == 'hit'))


@lru_cache(maxsize=1)
def index():
    """the store's index: {id: entry}"""
    with open(os.path.join(STORE, 'index.json'), encoding='utf-8') as fh:
        return {e['id']: e for e in json.load(fh)}


@lru_cache(maxsize=None)
def model(file):
    """the model of one member (its file name in the store), read once"""
    return resyn.load(os.path.join(STORE, file))


def credit(sid, member=0):
    """the recording a member of a learned design was made from: {title, author, page, licence, licence_url}"""
    return dict(index()[sid]['members'][member]['credit'])


@lru_cache(maxsize=None)
def order(sid, whole=False):
    """the members of a design in the order the seeds walk through: the index's `walk` (best first; a member heard
    as poor is left out of it), else a fixed shuffled order. A thing with recordings of single events and of the
    whole action (a member's `events` over 1: a walk of several steps, a knock pattern) plays its single events as a
    take and its whole actions as a performance (`perform`): whole=False gives the first, True the second (each
    falls back on all members when the thing has none of that sort)"""
    e = index()[sid]
    if e.get('walk'):
        walk = tuple(int(k) for k in e['walk'])
    else:
        walk = tuple(int(k) for k in np.random.default_rng(zlib.crc32(sid.encode())).permutation(len(e['members'])))
    keep = tuple(k for k in walk if (e['members'][k].get('events', 1) > 1) == whole)
    return keep or walk


def pick(sid, seed, bright=0.0):
    """the member (its position in the index) played for a seed: seed k plays order(sid)[(k % RETAKE) % n]; a
    request at |bright| >= CHARACTER walks only the darker or the brighter half (members are stored darkest first)"""
    walk = order(sid)
    n = len(index()[sid]['members'])
    if n > 1 and abs(bright) >= CHARACTER:
        half = (n + 1) // 2
        keep = set(range(half)) if bright < 0 else set(range(n - half, n))
        walk = tuple(k for k in walk if k in keep) or walk
    return walk[(int(seed) % RETAKE) % len(walk)]


def take(e, c, r, k=None, room=True, nudge=None):
    """one take of the design e (an index entry) under the controls c (module doc): the member the seed picks, or
    member k. room False leaves the take dry (a performance puts all its events in one room); nudge: offsets added
    to the energy and size axes ({'energy': .., 'size': ..}: one event of a performance a little firmer or smaller)"""
    ax = set(e['axes'])
    nudge = nudge or {}
    tilt = c.bright if 'bright' in ax else 0.0
    if k is None:
        k = pick(e['id'], c.seed, tilt)
        m = e['members'][k]
        c.member = {'n': k + 1, 'of': len(e['members']), 'recording': m['credit']['title'],
                    'author': m['credit']['author']}
    m = e['members'][k]
    energy = float(np.clip((c.energy if 'energy' in ax else 0.0) + nudge.get('energy', 0.0), -1.0, 1.0))
    s = float(np.clip((c.size if 'size' in ax else 0.0) + nudge.get('size', 0.0), -1.0, 1.0))
    size = float(SIZE[1] ** s if s >= 0 else SIZE[0] ** -s) if 'size' in ax else 1.0
    bright = TILT * tilt + 0.5 * energy
    hard = float(np.clip((c.hard if 'hard' in ax else 0.0) + 0.35 * energy, -1.0, 1.0))
    ring = c.lerp('ring', 0.5, 2.0, 1.0, log=True) if 'ring' in ax else 1.0
    pitch = 0.0
    if m.get('hz'):
        pitch = float(np.clip(12.0 * np.log2(c.hz(m['hz']) / m['hz']), -PITCH, PITCH))
    want = c.length() if room else None
    length = 1.0 if want is None else float(np.clip(want / (m['dur_s'] * size), *LENGTH))
    y = resyn.render(model(m['file']), r, length=length, pitch=pitch, size=size, bright=bright, hard=hard, ring=ring,
                     vary=VARY * c.vary)
    if dry(e, m):
        y = body(resyn.finish(y, **FINISH))
    y = present(y)
    y = image(y * 10 ** (6.0 * energy / 20), c.lerp('width', 0.0, 2.0, 1.0) if 'width' in ax else 1.0, m)
    if not room:
        return y
    return space.room(y, c.room('room'), amount=c.space if 'space' in ax else 0.0, r=r)


def body(y):
    """a short dry take with what follows its strike raised (BODY_EXP, BODY_DB): an upward compressor keyed by the
    level of every 5 ms against the loudest 5 ms, so the strike itself keeps its level and its brightness"""
    k = dsp.n_of(0.005)
    e = np.sqrt(np.convolve(np.mean(y ** 2, axis=1), np.ones(k) / k, 'same'))
    e = fx._follow(np.ascontiguousarray(e), fx._coef(0.5), fx._coef(20.0))
    return y * np.clip((e / (e.max() + 1e-12) + 1e-9) ** -BODY_EXP, 1.0, 10 ** (BODY_DB / 20))[:, None]


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


def image(y, w, m):
    """the take's stereo image at width w (0 mono, 1 as recorded, 2 wider): side times w; a recording with one
    channel inverted is turned back first; a mono recording, or a stereo one asked to be wider, is spread by
    decorrelation (SPREAD per unit of width)"""
    y = np.array(y, np.float64)
    if m.get('polarity', 1) < 0:
        y[:, 1] = -y[:, 1]
    mid, side = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])
    st = fx.decorrelate(mid, min(1.0, SPREAD * (max(w - 1.0, 0.0) if m['stereo'] else w)))
    return st + w * np.stack([side, -side], axis=1)


def wholes(e):
    """the members of e that are recordings of the whole action (`events` over 1), best first"""
    return [k for k in order(e['id'], True) if e['members'][k].get('events', 1) > 1]


def performs(sid):
    """True when the learned design sid is a thing a person does several times in a row (its index entry has a
    `pattern`, or recordings of the whole action): a walk, a knock, typing, coins, keys, clinking glasses"""
    e = index().get(sid)
    return bool(e and (e.get('pattern') or wholes(e)))


def perform(sid, seed=0, events=None, peak_db=-1.0, lufs=None, check=True, **request):
    """a performance of the thing sid (earshot.synth.perform): (y, meta) as registry.render gives a take, with
    meta['performance'] = {'pattern' or 'recording', 'events', 'pace' (events a second), 'recordings' (the member
    of each event, 1-based)}. events: how many (None: the pattern's usual number, or as many as fill `dur`);
    request: the controls of registry.render (dur = the length the performance fills at its natural pace).
    A recording of the whole action is played when the thing has one and the request asks for neither another number
    of events nor a length more than WHOLE_FIT away from its own; else the pattern sequences the thing's single events,
    each its own take of the engine with a small variation, all in one room and levelled as one sound"""
    from . import registry
    from ..synth import master
    from ..synth import perform as P
    e = index()[sid]
    d = registry.get(sid)
    dur = request.get('dur')
    c = registry.Controls(d, **request)          # the events of a pattern leave dur unread (take, room=False)
    c.seed = int(seed)
    r = dsp.rng(registry.seed_of(sid, seed) ^ 0x5EED)
    pat = e.get('pattern') or {}
    whole = wholes(e)
    walk = order(sid)
    play = None
    if whole:
        k = whole[(int(seed) % RETAKE) % len(whole)]
        m = e['members'][k]
        fits = ((events is None or events == m['events']) and
                (dur is None or abs(dur / m['dur_s'] - 1.0) <= WHOLE_FIT))
        if fits or not pat:
            y = take(e, c, r, k=k)
            play = {'recording': m['credit']['title'], 'events': m['events'], 'recordings': [k + 1],
                    'pace': m.get('pace')}
            c.member = {'n': k + 1, 'of': len(e['members']), 'recording': m['credit']['title'],
                        'author': m['credit']['author']}
    if play is None:
        p = P.plan(pat.get('name', 'pair'), r, members=walk, events=events, dur=dur, usual=pat.get('events'),
                   pace=pat.get('pace'))
        parts = []
        for k, nudge in zip(p['members'], p['nudge']):
            m = e['members'][k]
            size = SIZE[1] ** nudge['size'] if nudge['size'] >= 0 else SIZE[0] ** -nudge['size']
            parts.append((take(e, c, dsp.rng(int(r.integers(2 ** 32))), k=k, room=False, nudge=nudge),
                          m.get('onset_s', 0.0) * size))
        y = P.mix(parts, p)
        y = space.room(y, c.room('room'), amount=c.space if 'space' in e['axes'] else 0.0, r=r)
        play = {'pattern': p['pattern'], 'events': len(p['members']), 'pace': p['pace'],
                'recordings': [k + 1 for k in p['members']]}
        c.member = {'n': p['members'][0] + 1, 'of': len(e['members']),
                    'recording': e['members'][p['members'][0]]['credit']['title'],
                    'author': e['members'][p['members'][0]]['credit']['author']}
    # levelled as one sound: a performance is longer than the one-shots the registry lifts, so its few loudest
    # strikes are folded the way a long sound held down by a few peaks is (master.finish `sparse`), as far as a
    # single event of the thing is lifted (registry.LIFT)
    y = master.finish(y, peak_db=peak_db, lufs=lufs, cap=registry.LOUDEST[''], lift=registry.LIFT,
                      sparse=registry.LIFT)
    meta = {'id': sid, 'kind': d.kind, 'seed': int(seed), 'dur_s': round(len(y) / dsp.SR, 3),
            'peak_dbfs': round(master.true_peak_db(y), 2), 'lufs_max': round(master.loudness(y), 1),
            'member': dict(c.member), 'performance': play}
    if check:                       # several attacks are the design of a performance, not layers stacked on a hit
        meta['checks'] = registry.audit(y, SimpleNamespace(kind=d.kind, pitch=d.pitch, loop=False,
                                                            allows=d.allows + ('stacked',)))
    return y, meta


def _register(e):
    def fn(c, r):
        return take(e, c, r)
    fn.__name__ = e['id'].replace('/', '_')
    fn.__doc__ = (f"{e['says'][0]} (learned from {len(e['members'])} public-domain recording"
                  f"{'s' if len(e['members']) > 1 else ''}, credits in earshot/data/resyn/CREDITS.md)")
    hz = e['members'][0].get('hz') if e.get('tonal') else None
    design(e['id'], kind=e['kind'], says=tuple(e['says']), uses=tuple(e['uses']), tags=tuple(e['tags']) + ('real',),
           domains=tuple(e['domains']), dur=e['dur_s'], pitch=hz, axes=tuple(e['axes']),
           allows=tuple(e.get('allows', ())), prior=LEARNED * (1.0 if e.get('default', True) else DEFAULT_OFF))(fn)


for _e in index().values():
    _register(_e)
