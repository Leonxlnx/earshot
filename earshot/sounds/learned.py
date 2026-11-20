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
It is built outside the package from recordings whose licence is CC0, each one checked for a full band,
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
ones; the request decides between the two (earshot.brief.performance, make's `events`). A walk alternates the two
single steps its pattern names as its `feet` (one session, alike in loudness, brightness and length: one walker,
not a limp) and is heard from standing height where the pattern says how far (`far`: a gravel walk recorded at
the feet loses a little of its closest crunch), unless the request asks for it close or dry (space under 0).

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
from .registry import LEARNED, design, member

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
# dB the strike's crest may lose after the engine (None: as much as it loses; the sets and recipes were heard and
# chosen so), ms of the bell that restores it
ATTACK, ATTACK_MS = None, 3.0
ATTACK_MAX = 2.0              # dB the bell raises the strike at most (more is heard as a click of its own)
# what the stored envelope cannot hold, taken out of a take by gain (resyn.tidy), from two numbers per member that
# the build measured on its recording: TAIL, the member's `tail` (the take's faint floor of noise fades with the
# recording's own decay, where the envelope stops some 40 to 60 dB down); LEAD, the member's `lead` (what is smeared
# into the few milliseconds before the main strike held as low as the recording has it there). A member carries the
# numbers only where they were measured to bring its takes closer to its recording and change nothing else (the
# loudness, the length, the audit, the sound between the strike and the tail). False: off for every member. TAIL is
# off: chosen on one take of each member it did not hold on the next (it moved some tails away from their
# recording and changed lengths)
TAIL, LEAD = False, True
DEFAULT_OFF = 0.6           # prior factor of a design the index marks as not a default (`default`: false)
RETAKE = 101                # make.take's step between the seeds it tries: seeds this far apart play one member
# how far (a share of its length) a length asked of a performance may lie from a whole-action recording's own for
# that recording to be played (warped that little): further, the pattern fills the length at the natural pace, as a
# warp would slow or hurry the walk, the knock, the typing it holds
WHOLE_FIT = 0.15
LATE_END = 0.03     # s a warped whole recording may run past the length asked (its own end kept, not faded early)
# the different keys a typing run lets be heard at least: with two, a run alternates 1, 2, 1, 2 and reads as a
# machine; a thing with fewer single recordings borrows its neighbours' (borrow) and varies each key more
KEYS = 4


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
    to the energy and size axes and to the tilt ({'energy': .., 'size': .., 'bright': ..}: one event of a
    performance a little firmer, smaller or brighter)"""
    ax = set(e['axes'])
    nudge = nudge or {}
    tilt = c.bright if 'bright' in ax else 0.0
    own = k is None                 # the design's own take (another design that names a member plays it as it was)
    if k is None:
        k = pick(e['id'], c.seed, tilt)
        c.member = member(k + 1, len(e['members']))
    m = e['members'][k]
    energy = float(np.clip((c.energy if 'energy' in ax else 0.0) + nudge.get('energy', 0.0), -1.0, 1.0))
    s = float(np.clip((c.size if 'size' in ax else 0.0) + nudge.get('size', 0.0), -1.0, 1.0))
    size = float(SIZE[1] ** s if s >= 0 else SIZE[0] ** -s) if 'size' in ax else 1.0
    bright = TILT * tilt + 0.5 * energy + nudge.get('bright', 0.0)
    hard = float(np.clip((c.hard if 'hard' in ax else 0.0) + 0.35 * energy, -1.0, 1.0))
    ring = c.lerp('ring', 0.5, 2.0, 1.0, log=True) if 'ring' in ax else 1.0
    pitch = 0.0
    if m.get('hz'):
        pitch = float(np.clip(12.0 * np.log2(c.hz(m['hz']) / m['hz']), -PITCH, PITCH))
    want = c.length() if room else None
    length = 1.0 if want is None else float(np.clip(want / (m['dur_s'] * size), *LENGTH))
    tail = tuple(m['tail']) if TAIL and m.get('tail') else None
    lead = (m['onset_s'],) + tuple(m['lead']) if LEAD and m.get('lead') else None
    y = resyn.render(model(m['file']), r, length=length, pitch=pitch, size=size, bright=bright, hard=hard, ring=ring,
                     vary=VARY * c.vary, tail=tail, lead=lead)
    at = dsp.n_of(max(m.get('onset_s', 0.0) * size - 0.002, 0.0))
    strike = crest(y, at)
    if dry(e, m):
        y = body(resyn.finish(y, **FINISH))
    y = present(y)
    y = image(y * 10 ** (6.0 * energy / 20), c.lerp('width', 0.0, 2.0, 1.0) if 'width' in ax else 1.0, m)
    if room:
        y = space.room(y, c.room('room'), amount=c.space if 'space' in ax else 0.0, r=r)
    return attack(y, strike, at) if own else y


def crest(y, at):
    """dB of the peak over the RMS of the 22 ms from sample `at` (the strike: 2 ms before the main onset to 20 ms
    after it), the channels summed; None when the take is shorter"""
    seg = np.asarray(y)[at:at + dsp.n_of(0.022)]
    seg = seg.mean(axis=1) if seg.ndim == 2 else seg
    if len(seg) < dsp.n_of(0.022) or not np.any(seg):
        return None
    return float(20 * np.log10(np.abs(seg).max()) - 10 * np.log10(np.mean(seg ** 2)))


def attack(y, strike, at):
    """y with its strike as sharp as the engine played it: the steps after the engine (finish's saturation, body,
    present's limiter, the room) may lower the crest of the strike's first 20 ms (crest()) by at most ATTACK dB;
    a take that lost more has its loudest moment there raised by a short bell (ATTACK_MS wide) just as far as
    brings it back, by ATTACK_MAX dB at most. Measured on thirty designs: a short dry strike (a gavel, a crutch, a pen cap, a knock on a
    table) lost 1.2 to 1.7 dB, mostly to the saturation; the others lost under 0.5 dB"""
    if strike is None or ATTACK is None:
        return y
    now = crest(y, at)
    if now is None or strike - now <= ATTACK:
        return y
    n = dsp.n_of(0.022)
    mono = np.asarray(y)[at:at + n].mean(axis=1)
    p = at + int(np.argmax(np.abs(mono)))
    h = dsp.n_of(ATTACK_MS / 1000) // 2
    w = np.zeros(len(y))
    lo, hi = max(p - h, 0), min(p + h + 1, len(y))
    w[lo:hi] = np.hanning(2 * h + 1)[lo - (p - h):hi - (p - h)]
    lo_g, hi_g = 1.0, 10 ** (min(strike - now, ATTACK_MAX) / 4)
    for _ in range(24):                  # the smallest bell that brings the crest back within ATTACK - 0.02 dB
        g = 0.5 * (lo_g + hi_g)
        if strike - crest(y * (1 + (g - 1) * w)[:, None], at) > ATTACK - 0.02:
            lo_g = g
        else:
            hi_g = g
    g = min(hi_g, 10 ** (ATTACK_MAX / 20))
    return y * (1 + (g - 1) * w)[:, None]


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


def feet(e):
    """the recordings a walk alternates (member positions): the pattern's `feet`, the two single steps of one session
    most alike in loudness, brightness and length (one walker in one pair of shoes: the feet then differ only as a
    left and a right do, by each take's own variation and the trailing foot's lighter landing); one recording twice
    when no step of the thing is alike to its first (every step a new take of it). Without `feet`, the first two
    of the walk"""
    return [int(k) for k in ((e.get('pattern') or {}).get('feet') or list(order(e['id']))[:2])]


def wholes(e):
    """the members of e that are recordings of the whole action (`events` over 1), best first"""
    return [k for k in order(e['id'], True) if e['members'][k].get('events', 1) > 1]


def borrow(e, keys):
    """the keys of a typing run when the thing has fewer than KEYS single recordings: its own, then those of its
    neighbours (the other designs of its family and kind: the other keyboards), one from each in turn, until KEYS
    different keys are heard; and the family's deeper key (a design whose description names a space bar) for the
    pauses between words. Returns (keys [(entry, member) ...], position of the deep key or None)"""
    family = e['id'].split('/')[0]
    near = [x for sid, x in sorted(index().items()) if sid != e['id'] and sid.split('/')[0] == family
            and x['kind'] == e['kind']]
    bar = [x for x in near if any('space bar' in s for s in x['says'])]
    others = [[(x, k) for k in order(x['id'])] for x in near if all(x is not b for b in bar)]
    keys, i = list(keys), 0
    while len(keys) < KEYS and any(others):
        if others[i % len(others)]:
            keys.append(others[i % len(others)].pop(0))
        i += 1
    if not bar:
        return keys, None
    return keys + [(bar[0], order(bar[0]['id'])[0])], len(keys)


def stretch(y, dur, hop=0.01, rise=0.02, fall=0.3):
    """the `dur` s of y (a take of a recording of a whole action, stereo) where the action is busiest (the most
    energy, the earliest of equals, on a `hop` s grid), faded in over `rise` s unless it starts at the beginning and
    faded out over a quarter of it (at most `fall` s): a length shorter than the recording keeps the action at its
    own pace and never ends in a cut"""
    n, k = dsp.n_of(dur), dsp.n_of(hop)
    e = np.concatenate([[0.0], np.cumsum(np.mean(y ** 2, axis=1))])
    starts = np.arange(0, len(y) - n + 1, k)
    s = int(starts[np.argmax(e[starts + n] - e[starts])])
    out = np.array(y[s:s + n])
    if s > 0:
        a = min(n, dsp.n_of(rise))
        out[:a] *= (np.sin(np.linspace(0, np.pi / 2, a)) ** 2)[:, None]
    b = min(n, dsp.n_of(min(fall, 0.25 * dur)))
    out[n - b:] *= (np.cos(np.linspace(0, np.pi / 2, b)) ** 2)[:, None]
    return out


def performs(sid):
    """True when the learned design sid is a thing a person does several times in a row (its index entry has a
    `pattern`, or recordings of the whole action): a walk, a knock, typing, coins, keys, clinking glasses"""
    e = index().get(sid)
    return bool(e and (e.get('pattern') or wholes(e)))


def perform(sid, seed=0, events=None, peak_db=-1.0, lufs=None, check=True, **request):
    """a performance of the thing sid (earshot.synth.perform): (y, meta) as registry.render gives a take, with
    meta['performance'] = {'pattern' or 'recording', 'events', 'pace' (events a second), 'recordings' (the member
    of each event, 1-based)}. events: how many (None: the pattern's usual number, or as many as fill `dur`);
    request: the controls of registry.render (dur = the length of the file: the pattern's events are spread over it,
    or played at the closest or widest gap a person keeps, with meta['note'] saying so: earshot.synth.perform.fill).
    A recording of the whole action is played when the thing has one and the request asks for neither another number
    of events nor a length more than WHOLE_FIT away from its own; else the pattern sequences the thing's single events,
    each its own take of the engine with a small variation, all in one room and levelled as one sound. A thing with no
    pattern plays the busiest `dur` s of a recording of the whole action at least that long (`stretch`, never
    warped), or its longest one whole, and meta['note'] says which"""
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
    # a walk is heard from standing height: the pattern's `far` tilts every step (dB per octave around 1 kHz); a
    # request that asks for a close or dry sound (space under 0) hears the recording's own closeness
    near = (request.get('space') or 0.0) < 0
    far = {'bright': float(pat['far'])} if pat.get('name') == 'walk' and pat.get('far') and not near else None
    whole = wholes(e)
    walk = order(sid)
    play, p, cut, long = None, {}, None, False
    if whole:
        k = whole[(int(seed) % RETAKE) % len(whole)]
        m = e['members'][k]
        fits = ((events is None or events == m['events']) and
                (dur is None or abs(dur / m['dur_s'] - 1.0) <= WHOLE_FIT))
        if fits or not pat:
            if dur and abs(dur / m['dur_s'] - 1.0) > WHOLE_FIT:
                # no pattern to spread over the length: a stretch of a recording as long as asked, unwarped (a
                # warp would slow or hurry the action), or the longest whole when none is as long
                pool = [j for j in whole if e['members'][j]['dur_s'] >= dur] or [
                    max(whole, key=lambda j: e['members'][j]['dur_s'])]
                k, cut = pool[(int(seed) % RETAKE) % len(pool)], float(dur)
                m = e['members'][k]
                c = registry.Controls(d, **{a: v for a, v in request.items() if a != 'dur'})
                c.seed = int(seed)
            y = take(e, c, r, k=k, nudge=far)
            long = bool(cut) and len(y) > dsp.n_of(cut)
            if long:
                y = stretch(y, cut)
            c.member = member(k + 1, len(e['members']))
            play = {'recording': c.member['recording'], 'events': m['events'], 'recordings': [k + 1],
                    'pace': m.get('pace')}
    if play is None:
        name = pat.get('name', 'pair')
        keys = [(e, k) for k in (dict.fromkeys(feet(e)) if name == 'walk' else walk)]
        deep, spread = None, 1.0
        if name == 'type' and len(walk) < KEYS:
            keys, deep = borrow(e, keys)
            spread = 2.0
        ring = float(np.clip(np.median([e['members'][k]['dur_s'] for k in walk]), 0.35, 0.6))
        p = P.plan(name, r, members=[i for i in range(len(keys)) if i != deep], events=events, dur=dur,
                   usual=pat.get('events'), pace=pat.get('pace'), deep=deep, spread=spread, ring=ring)
        parts, own = [], []
        for i, nudge in zip(p['members'], p['nudge']):
            ek, k = keys[i]
            m = ek['members'][k]
            size = SIZE[1] ** nudge['size'] if nudge['size'] >= 0 else SIZE[0] ** -nudge['size']
            if far:
                nudge = {**nudge, 'bright': nudge['bright'] + far['bright']}
            z = take(ek, c, dsp.rng(int(r.integers(2 ** 32))), k=k, room=False, nudge=nudge)
            if ek is e:
                own.append(np.abs(z).max())
            parts.append([z, m.get('onset_s', 0.0) * size, ek is e])
        # a borrowed key is played at the level of the thing's own keys (each recording has its own gain)
        ref = float(np.median(own)) if own else None
        for part in parts:
            if ref and not part[2]:
                part[0] = part[0] * (ref / (np.abs(part[0]).max() + 1e-12))
        # a length asked for: the last event placed early enough to ring out (its ring measured in the room it is
        # played in) before the end; measured LIFT dB deeper, as finish below folds the strikes and raises what
        # follows them by up to that
        amount = c.space if 'space' in e['axes'] else 0.0
        if p.get('dur') and p['pattern'] in P.GAPS:
            deep = P.RING_DB + registry.LIFT
            rings = [P.ring_of(space.room(z, c.room('room'), amount=amount, r=dsp.rng(0)), on, deep)
                     for z, on, _ in parts]
            p, kept = P.close(p, rings)
            parts = parts[:kept]
            parts[-1][0] = P.ease(parts[-1][0], parts[-1][1])     # no later event covers the last one's own end
        y = P.mix([(z, on) for z, on, _ in parts], p)
        y = space.room(y, c.room('room'), amount=amount, r=r)
        play = {'pattern': p['pattern'], 'events': len(p['members']), 'pace': p['pace'],
                'recordings': [keys[i][1] + 1 if keys[i][0] is e else f"{keys[i][0]['id']}#{keys[i][1] + 1}"
                               for i in p['members']]}
        p['members'] = [keys[i][1] if keys[i][0] is e else walk[0] for i in p['members']]
        c.member = member(p['members'][0] + 1, len(e['members']))
    # levelled as one sound: a performance is longer than the one-shots the registry lifts, so its few loudest
    # strikes are folded the way a long sound held down by a few peaks is (master.finish `sparse`), as far as a
    # single event of the thing is lifted (registry.LIFT)
    y = master.finish(y, peak_db=peak_db, lufs=lufs, cap=registry.LOUDEST[''], lift=registry.LIFT,
                      sparse=registry.LIFT)
    if p.get('dur'):               # a length asked for is the file's: the events were spread over it (perform.fill)
        # a scatter or a shake whose events ring longer than `ring` is faded over the last one: the motion stops
        y = P.fit(y, p['dur'], fade=ring if p['pattern'] in P.OWN else 0.05)
    if long:                       # the stretch is the length asked (finish may have trimmed its faded end)
        y = P.fit(y, cut)
    elif dur and not p and not cut and not 0 <= len(y) - dsp.n_of(dur) <= dsp.n_of(LATE_END):
        # a whole recording warped to the length: finish may have trimmed its end (padded with silence), or its own
        # end may lie a little later (left to ring out when within LATE_END)
        y = P.fit(y, float(dur))
    meta = {'id': sid, 'kind': d.kind, 'seed': int(seed), 'dur_s': round(len(y) / dsp.SR, 3),
            'peak_dbfs': round(master.true_peak_db(y), 2), 'lufs_max': round(master.loudness(y), 1),
            'member': dict(c.member), 'performance': play}
    note = P.said(p, dur, meta['dur_s']) if p else None
    if cut:
        note = (f"{cut:g} s of a recording of the whole action ({m['dur_s']:.2f} s), its busiest stretch" if long else
                f"{meta['dur_s']:.2f} s, not the {cut:g} s asked: the longest recording of the whole action is "
                f"{m['dur_s']:.2f} s")
    if note:
        meta['note'] = note
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
