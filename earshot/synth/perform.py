"""Performances: several events of one thing played the way a person does it.

A listener hears one footstep, one knuckle on a door or one key as a fragment; what he recognises is the action: a
walk, a knock, a burst of typing, a handful of coins settling, a bunch of keys shaken, two glasses clinked. This
module turns single events into that action. It only plans and mixes: when each event starts, how loud it is and
which recording plays it; the events themselves are rendered by the caller (earshot.sounds.learned), each one its
own take, so no two are the same.

A pattern is data (`PATTERNS`, and per thing in the learned store's index: the pattern's name, its usual number of
events and pace), never a branch per thing:
    walk     two feet alternate (two neighbouring recordings in turn), 100 to 120 steps a minute, each step a little
             early or late and a little louder or softer, the trailing foot slightly softer
    knock    three or four knuckle hits about 0.2 s apart, the first the firmest, the last a touch later
    type     a run of keys at 5 to 8 a second with the irregular gaps of typing (log-normal), a short pause between
             words where a deeper key (the space bar) falls; every key a little different in size, brightness and
             level, and never varied the same way twice in a row (ten fingers, not a machine)
    scatter  things that land and settle: each next impact comes sooner and softer, as a bouncing, spreading heap
    shake    dense irregular impacts in two or three swings (keys on a ring, ice in a glass, a shaker)
    pair     two events, the second the answer: a clink and its reply, a switch on and off
Rules for all: never the same recording twice in a row (when the thing has more than one), never on a grid, tails
overlap and are never gated (an event rings on under the next), one position and one room for the whole.

    from earshot.synth import perform
    p = perform.plan('walk', dsp.rng(1), members=(2, 0, 1))     # {'onsets', 'gains', 'members', 'pace', ...}
    y = perform.mix([(take_k, onset_k) ...], p)                   # stereo, the first event's onset at LEAD s
"""
import numpy as np

from ..forge import dsp

LEAD = 0.02          # s of silence before the first event's onset (the caller fades and levels afterwards)
# name: events (lo, hi) when the request names no count, pace (lo, hi) events per second, level spread (dB, sd),
# timing spread (share of the gap, sd). Paces: walking 100-120 steps a minute (1.7-2.0 Hz, the usual range of
# comfortable gait); knocking about 0.2 s between knuckle hits (4.5-5.5 Hz); typing 5-8 keys a second (an average
# typist's 40-60 words a minute at five letters and a space a word); a shaken bunch of keys 10-20 impacts a second
PATTERNS = {
    'walk': dict(events=(6, 8), pace=(100 / 60, 120 / 60), level=1.0, timing=0.03),
    'knock': dict(events=(3, 4), pace=(4.5, 5.5), level=0.8, timing=0.05),
    'type': dict(events=(10, 16), pace=(5.0, 8.0), level=2.0, timing=0.35),
    'scatter': dict(events=(4, 7), pace=(5.0, 8.0), level=1.5, timing=0.2),
    'shake': dict(events=(10, 18), pace=(10.0, 20.0), level=3.0, timing=0.45),
    'pair': dict(events=(2, 2), pace=(2.0, 4.0), level=0.5, timing=0.1),
}
MOST = 64            # events a performance has at most (a length asked for fills at the pace, up to this)


def count(name, r, events=None, dur=None, rate=None, usual=None):
    """how many events: the number asked for; else as many as fill `dur` s at `rate` events a second; else the
    usual number (`usual` (lo, hi), else the pattern's), drawn from its range. Always 1 .. MOST"""
    p = PATTERNS[name]
    if events:
        return int(np.clip(events, 1, MOST))
    if dur:
        rate = rate or float(np.mean(p['pace']))
        return int(np.clip(round(dur * rate), 2, MOST))
    lo, hi = usual or p['events']
    return int(r.integers(lo, hi + 1))


def sequence(members, n, r, name='walk'):
    """the recording (a position in `members`, the thing's members best first) of each of n events: never the same
    one twice in a row when there are two or more. walk: the first two recordings as the two feet, in turn (one
    walker has one pair of shoes); type: the brighter keys in a shuffled run, with the darkest key (members are stored
    darkest first, so the lowest position) left for the pauses (`plan`); others: the first three in an order that
    never repeats back to back"""
    members = list(members)
    if len(members) == 1:
        return [members[0]] * n
    if name == 'walk':
        return [members[k % 2] for k in range(n)]
    pool = members[:3] if name != 'type' else members[:5]
    out = []
    for _ in range(n):
        choice = [m for m in pool if not out or m != out[-1]]
        out.append(choice[int(r.integers(len(choice)))])
    return out


def _keys(members, n, r, size, tone):
    """the variation of each key of a typing run (size and brightness offsets, added to the plan's): a finger
    never strikes two keys alike, and the same recording heard again is never varied the way it was last time
    (the new offsets are mirrored when they land within a third of their spread of that key's last ones)"""
    last = {}
    ds, db = r.normal(0.0, size, n), r.normal(0.0, tone, n)
    for i, m in enumerate(members):
        if m in last and abs(ds[i] - last[m][0]) < size / 3 and abs(db[i] - last[m][1]) < tone / 3:
            ds[i], db[i] = -ds[i] if abs(ds[i]) > size / 6 else ds[i] + size / 2, -db[i]
        last[m] = (ds[i], db[i])
    return ds, db


def plan(name, r, members=(0,), events=None, dur=None, usual=None, pace=None, deep=None, spread=1.0):
    """the performance of a pattern: {'pattern', 'onsets' (s, the first at LEAD), 'gains' (linear), 'members' (the
    recording of each event), 'nudge' (per event: offsets for the caller's energy, size and bright axes, small),
    'pace' (events per second over the whole)}. events: the number asked for; dur: the length to fill at the natural
    pace; usual (lo, hi) and pace (lo, hi) events a second: the thing's own, in place of the pattern's. type only:
    deep, a deeper key (a space bar) kept for the pauses between words instead of the darkest of `members`; spread,
    how much each key's size, brightness and level differ (1 = a little; more when few recordings stand for a whole
    keyboard, so the same key never comes back the same)"""
    p = PATTERNS[name]
    rate = float(r.uniform(*(pace or p['pace'])))
    n = count(name, r, events, dur, rate, usual)
    gaps = np.full(max(n - 1, 0), 1.0 / rate)
    db = r.normal(0.0, p['level'], n)
    size = np.zeros(n)
    if name == 'walk':
        gaps *= 1.0 + np.clip(r.normal(0.0, p['timing'], n - 1), -0.08, 0.08)
        db[1::2] -= float(r.uniform(0.5, 1.5))            # the trailing foot lands a little softer
    elif name == 'knock':
        gaps *= 1.0 + np.clip(r.normal(0.0, p['timing'], n - 1), -0.12, 0.12)
        if n > 2:
            gaps[-1] *= float(r.uniform(1.0, 1.15))       # the last knock a touch later
        db -= np.linspace(0.0, float(r.uniform(2.0, 4.0)), n)      # the first the firmest
        db[0] = max(db[0], 0.0)
    elif name == 'type':
        gaps *= np.exp(np.clip(r.normal(0.0, p['timing'], n - 1), -0.7, 0.9))
        gaps = np.maximum(gaps, 0.06)
    elif name == 'scatter':
        ratio = float(r.uniform(0.6, 0.78))            # the first gap is the pace's, the next ones ever shorter
        gaps = (float(r.uniform(0.8, 1.2)) / rate * ratio ** np.arange(n - 1)
                * np.exp(r.normal(0.0, p['timing'], n - 1)))
        gaps = np.maximum(gaps, 0.012)
        db = -np.arange(n) * float(r.uniform(1.5, 3.0)) + r.normal(0.0, p['level'], n)
        db[0] = 0.0
        size = -0.15 * np.arange(n) / max(n - 1, 1)       # the later, smaller settling bits
    elif name == 'shake':
        gaps *= np.exp(np.clip(r.normal(0.0, p['timing'], n - 1), -1.0, 1.0))
        swings = int(r.integers(2, 4))
        phase = np.linspace(0.0, swings * np.pi, n)
        db += 6.0 * (np.abs(np.sin(phase + 0.3)) - 1.0)   # louder where each swing turns
        gaps = np.maximum(gaps, 0.015)
    elif name == 'pair':
        gaps *= float(r.uniform(0.8, 1.25))
        db[1:] -= float(r.uniform(1.5, 4.0))               # the answer is softer
    onsets = LEAD + np.concatenate([[0.0], np.cumsum(gaps)])
    order = sequence(members, n, r, name)
    tone = np.zeros(n)
    if name == 'type':
        ds, tone = _keys(order, n, r, 0.05 * spread, 0.15 * spread)
        size += ds
        db += r.normal(0.0, 0.5 * (spread - 1.0), n) if spread > 1.0 else 0.0
    if name == 'type' and (len(members) > 2 or deep is not None):     # a word ends now and then: a pause, then
        dark = min(members) if deep is None else deep                  # the deeper key
        members = list(members) + ([deep] if deep is not None else [])
        k = int(r.integers(3, 6))
        while k < n - 1:
            onsets[k:] += float(r.uniform(0.08, 0.18))
            order[k] = dark
            for j in (k - 1, k + 1):                      # its neighbours: another key, and not each other's
                if order[j] == dark:
                    near = {order[i] for i in (j - 1, j + 1) if 0 <= i < n}
                    order[j] = next(m for m in members if m not in near | {dark})
            db[k] += 1.0
            size[k] = 0.25 if deep is None else size[k]          # a space bar is deeper as recorded
            k += int(r.integers(4, 7))
    nudge = [{'energy': float(np.clip(d / 12.0, -0.3, 0.3)), 'size': float(s + r.normal(0.0, 0.04)),
              'bright': float(t)} for d, s, t in zip(db, size, tone)]
    span = float(onsets[-1] - onsets[0])
    return {'pattern': name, 'onsets': [float(t) for t in onsets], 'gains': [float(10 ** (d / 20)) for d in db],
            'members': [int(m) for m in order], 'nudge': nudge,
            'pace': round((n - 1) / span, 2) if span > 0 else None}


def mix(events, p):
    """the events placed and summed: events = [(y, onset_s) ...] in the plan's order (y mono or stereo, onset_s the
    event's main onset inside y), each scaled by the plan's gain and starting so that its onset falls on the plan's
    onset. Every tail rings out in full under the events that follow. Returns stereo (n, 2)"""
    starts, parts = [], []
    for (y, on), t, g in zip(events, p['onsets'], p['gains']):
        y = np.asarray(y, np.float64)
        y = y if y.ndim == 2 else np.stack([y, y], 1)
        parts.append(g * y)
        starts.append(dsp.n_of(t) - dsp.n_of(on) if on > 0 else dsp.n_of(t))
    shift = max(0, -min(starts))
    starts = [s + shift for s in starts]
    out = np.zeros((max(s + len(y) for s, y in zip(starts, parts)), 2))
    for s, y in zip(starts, parts):
        out[s:s + len(y)] += y
    return out
