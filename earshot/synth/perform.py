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
    row      a counted thing done in a row by hand (claps, taps, clicks, beeps), 2.5 to 3.5 a second, a little
             uneven in time and level; earshot.make plays any one-shot design this way, each event a new seed;
             without a count or a length ("clapping") six to nine events, about 2 to 3 s
    count    a countdown: one event a second, on the clock
Rules for all: never the same recording twice in a row (when the thing has more than one), never on a grid, tails
overlap and are never gated (an event rings on under the next), one position and one room for the whole.
A length asked for is the length of the file (`fill`, `fit`): the first event at the start, the last early enough
to ring out, the gaps even around the pattern's own spread; a length without a count takes as many events as the
pattern's pace fits. Gaps a hand cannot keep (GAPS: closer than its fastest, or so far apart the events are no longer
one action) are held at the limit and `said` gives the plain line; a countdown stays one a second. A scatter and a
shake keep their own timing and are reshaped to the length instead: the heap settles over it (`settle`: more
impacts, thickening, over a longer one; a quicker settle over a shorter one), a shake repeats its swing.

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
    'row': dict(events=(6, 9), pace=(2.5, 3.5), level=0.8, timing=0.05),
    'count': dict(events=(3, 3), pace=(1.0, 1.0), level=0.3, timing=0.0),
}
MOST = 64            # events a performance has at most (a length asked for fills at the pace, up to this)
# the gaps (s, between onsets) a length asked for may spread a pattern's events to: closer than the first a hand
# cannot repeat the stroke (finger tapping and knocking top out at about 8 a second, running steps at about 4), and
# further apart than the second the events no longer make one action. A countdown is on the clock: exactly 1 s.
# scatter and shake keep their own timing (a heap settles and a shaken thing swings as physics has it): a length
# asked of them reshapes it instead (OWN: `settle`, more swings), never evened out
GAPS = {'walk': (0.25, 1.5), 'knock': (0.12, 1.5), 'row': (0.12, 1.5), 'type': (0.06, 1.0), 'pair': (0.1, 1.5),
        'count': (1.0, 1.0)}
OWN = ('scatter', 'shake')
# things a hand or a foot does: their rows keep the GAPS limit. Any other event (a tick, a click, a blip, a counter)
# follows as close as its own length lets it without sounding over itself (`own_gap`): a number counting up in a
# film ticks 15 to 30 times a second
HAND = ('knock', 'clap', 'tap', 'step', 'footstep', 'stomp', 'slap', 'punch', 'kick', 'drum', 'door', 'hand',
        'knuckle', 'finger', 'foot')
OWN_GAP = (0.02, 0.12)   # s: the closest a row of events that no hand plays may follow (its own length, within this)
OWN_DB = 20.0            # dB under its loudest at which an event no longer sounds over the next one
LAST_RING = 0.1          # s the last event of a length asked keeps at least, before the events are held apart
SWING = 0.35         # s one swing of a shake takes (its usual 10-18 impacts at 10-20 a second fall in two or three)
FADE = 0.05         # s over which `fit` fades a performance out at the length asked for
RING_DB = 40.0       # dB under its loudest at which an event has sounded: a fade from there is not heard as a cut
CHIEF = {'walk': 'step', 'knock': 'knock', 'row': 'event', 'type': 'key', 'pair': 'event', 'count': 'beat'}


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


def by_hand(thing=None, sound=None):
    """True when the events of a row are done by a hand or a foot (HAND: the counted word, else the design's family
    or name), so they keep the GAPS limit; False for a tick, a click, a blip, a counter"""
    words = [str(thing or '')] + str(sound or '').replace('/', '_').split('_')
    return any(w.rstrip('s') in HAND or w in HAND for w in words if w)


def own_gap(y):
    """the closest s a row may follow an event y with: its length down to OWN_DB under its loudest, within OWN_GAP"""
    return float(np.clip(ring_of(y, 0.0, OWN_DB), *OWN_GAP))


def fill(name, events, dur, rate, ring, lo=None):
    """how a pattern's events fill a length of `dur` s: (n, gap, fits). The first event at LEAD, the last `ring` s
    before the end so it has sounded, the gaps even (the plan adds the pattern's human spread around them). events:
    the number asked for, else the number that keeps the pattern's natural `rate` (events a second) closest (a
    countdown: as many whole seconds as fit). lo: the closest gap, in place of the pattern's (GAPS; a row no hand
    plays: `own_gap`). Where the usual ring leaves the events closer than that, the last one rings less (at least
    LAST_RING; `close` and the caller's taper measure and shape it) before the gap is held. fits is False when the
    gap had to be held: the events are played at that gap and the length follows from it"""
    lo, hi = (GAPS[name][0] if lo is None else lo), GAPS[name][1]
    room = max(dur - LEAD - ring, 0.0)                    # from the first onset to the last
    whole = np.floor(room * rate + 1e-6) if name == 'count' else np.round(room * rate)
    n = int(np.clip(events or whole + 1, 2, MOST))
    if name != 'count' and room < (n - 1) * lo:
        room = max(room, min((n - 1) * lo, dur - LEAD - LAST_RING))
    gap = room / (n - 1)
    held = float(np.clip(gap, lo, hi))
    if name == 'count':                                   # one a second: a little room after the last is fine
        return n, held, gap >= held and room - (n - 1) * held < 1.0
    return n, held, abs(held - gap) < 1e-9


def fit(y, dur, fade=FADE):
    """y (stereo, finished) at exactly `dur` s: a longer one faded out over its last `fade` s (the last event has
    sounded by then: `fill` leaves it a usual ring, `close` its measured one), a shorter one padded with silence"""
    n = dsp.n_of(dur)
    if len(y) >= n:
        y = np.array(y[:n])
        k = min(n, dsp.n_of(fade))
        y[n - k:] *= (np.cos(np.linspace(0, np.pi / 2, k)) ** 2)[:, None]
        return y
    return np.concatenate([y, np.zeros((n - len(y),) + y.shape[1:])])


def ring_of(y, onset=0.0, db=RING_DB):
    """s from an event's onset (s into y) to the last moment its 5 ms level is within `db` of its loudest: how long
    the event sounds before a fade from there is no longer heard as a cut"""
    y = np.asarray(y, np.float64)
    m = y ** 2 if y.ndim == 1 else np.mean(y ** 2, axis=1)
    k = dsp.n_of(0.005)
    e = 10 * np.log10(np.convolve(m, np.ones(k) / k, 'same') + 1e-20)
    return max(float(np.nonzero(e > e.max() - db)[0][-1]) / dsp.SR - onset, 0.0)


def ease(y, onset=0.0, db=RING_DB, most=0.25):
    """an event y whose own end is a cut (10 ms before its end still within `db` of its loudest: a recording that
    stops while the thing still sounds, as a step on gravel does) faded out over the last third of what follows its
    onset (s into y), at most `most` s; one that has died away is returned as it is. For the last event of a length
    asked for, which no later event covers"""
    y = np.asarray(y, np.float64)
    m = y ** 2 if y.ndim == 1 else np.mean(y ** 2, axis=1)
    k = dsp.n_of(0.005)
    if len(m) < 4 * k:
        return y
    e = np.convolve(m, np.ones(k) / k, 'valid')           # 5 ms levels: its level just before its own end fade
    if 10 * np.log10(e[-3 * k] + 1e-20) < 10 * np.log10(e.max() + 1e-20) - db:
        return y
    j = min(dsp.n_of(most), max((len(y) - dsp.n_of(onset)) // 3, k))
    g = np.cos(np.linspace(0, np.pi / 2, j)) ** 2
    y = np.array(y)
    y[-j:] *= g if y.ndim == 1 else g[:, None]
    return y


def taper(y, room):
    """an event y that would still sound `room` s after its start (the end of a length asked) ended there: faded out
    over the last half of it (cos²), so the file ends on its tail, not on a cut"""
    y = np.array(y, np.float64)
    n = max(dsp.n_of(room), 1)
    if len(y) <= n:
        return y
    y = y[:n]
    j = max(n // 2, 1)
    g = np.cos(np.linspace(0, np.pi / 2, j)) ** 2
    y[-j:] *= g if y.ndim == 1 else g[:, None]
    return y


def close(p, rings, keep=False):
    """the plan p (a length filled by `fill`: p['dur'] set) with its events placed so that the last rings out before
    the end: rings[i] (s, `ring_of`) is how long event i sounds. When an event's onset plus its ring passes the end
    (less the end fade, FADE: the last event, or an earlier one that rings longer), the onsets are drawn together
    toward the first (the human spread kept) until none does; where that would bring them closer on average than a
    hand repeats the stroke (GAPS, or the plan's 'lo'), the last events are left out, unless `keep` (a number of
    events asked for): then all stay at that closest gap and the caller tapers what still sounds at the end (`taper`).
    Returns the plan (the same dict when nothing moves) and how many events it keeps"""
    on = np.asarray(p['onsets'])
    ring = np.asarray(rings[:len(on)], np.float64)
    n, end = len(on), (p.get('dur') or 0.0) - FADE
    if not p.get('dur') or p['pattern'] not in GAPS or n < 2 or (on + ring).max() <= end:
        return p, n
    lo = p.get('lo') or GAPS[p['pattern']][0]
    while True:
        # the most the onsets (from the first) may be scaled by for every event to have sounded by the end
        s = min(1.0, float(((end - LEAD - ring[1:n]) / np.maximum(on[1:n] - LEAD, 1e-9)).min()))
        if keep or n == 2 or s * (on[n - 1] - on[0]) / (n - 1) >= lo:
            break
        n -= 1
    s = max(s, lo * (n - 1) / max(float(on[n - 1] - on[0]), 1e-9))
    on = LEAD + (on[:n] - LEAD) * s
    q = {**p, 'onsets': [float(t) for t in on], 'gains': p['gains'][:n], 'members': p['members'][:n],
         'nudge': p['nudge'][:n], 'gap': round(float(on[-1] - on[0]) / (n - 1), 3)}
    q['pace'] = round((n - 1) / float(on[-1] - on[0]), 2) if on[-1] > on[0] else None
    return q, n


def settle(n, first, ratio, room, events=None):
    """(n, ratio) of a scatter that settles over `room` s (its first onset to its last): the first gap stays and each
    next one is `ratio` times the one before, the ratio set so the gaps add up to the room. A room longer than the
    scatter's own settle takes more impacts (as many as keep the mean gap at two thirds of the first: the heap
    thickens over the length, toward an even trickle at ratio 1); a shorter one loses its last, smallest impacts and
    settles faster (a smaller ratio).
    events: a number asked for, kept. The plan stretches the onsets onto the room exactly afterwards"""
    def span(k, q):
        return first * (k - 1 if q > 1.0 - 1e-9 else (1.0 - q ** (k - 1)) / (1.0 - q))
    if not events and room > span(n, ratio):
        n = int(np.clip(max(n, round(1.5 * room / first) + 1), 2, MOST))
    while not events and n > 2 and span(n - 1, ratio) >= room:
        n -= 1                                            # a short room loses the last, smallest bits first
    lo, hi = 0.2, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if span(n, mid) < room else (lo, mid)
    return n, 0.5 * (lo + hi)


def said(p, dur, real, thing=None):
    """the plain line for a length the events could not fill as asked (p: a plan whose `fits` is False), or None.
    thing: what one event is called ("knock", "clap"), else the pattern's word"""
    if p.get('fits', True) or not dur:
        return None
    n, gap, thing = len(p['onsets']), p['gap'], thing or CHIEF.get(p['pattern'], 'event')
    if p['pattern'] == 'count':
        why = 'a countdown keeps one beat a second'
    elif p.get('lo') and gap <= p['lo']:
        why = f'{thing}s closer than {gap:g} s would sound over each other'
    elif gap <= GAPS[p['pattern']][0]:
        why = f'{thing}s cannot follow closer than {gap:g} s by hand'
    else:
        why = f'{thing}s further apart than {gap:g} s are no longer one action'
    return f'{n} {thing}s {gap:g} s apart: {real:.2f} s, not the {dur:g} s asked ({why})'


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


def plan(name, r, members=(0,), events=None, dur=None, usual=None, pace=None, deep=None, spread=1.0, ring=None,
         lo=None):
    """the performance of a pattern: {'pattern', 'onsets' (s, the first at LEAD), 'gains' (linear), 'members' (the
    recording of each event), 'nudge' (per event: offsets for the caller's energy, size and bright axes, small),
    'pace' (events per second over the whole)}. events: the number asked for; dur: the length asked for. With
    `ring` (s one event needs to sound) and a pattern of GAPS, dur is the length of the file and the events are
    spread over it (`fill`): the plan then also has 'dur' (the length to `fit` the finished sound to, None when
    the gaps were held and the length follows from them), 'gap' and 'fits'; a scatter or a shake keeps its own shape
    and is fitted to end `ring` s before dur (`settle`; a longer shake swings more often); else dur is filled at the
    natural pace.
    usual (lo, hi) and pace (lo, hi) events a second: the thing's own, in place of the pattern's. type only:
    deep, a deeper key (a space bar) kept for the pauses between words instead of the darkest of `members`; spread,
    how much each key's size, brightness and level differ (1 = a little; more when few recordings stand for a whole
    keyboard, so the same key never comes back the same)
    lo: the closest gap of a row no hand plays (`own_gap`), in place of the pattern's GAPS limit; kept as 'lo'"""
    p = PATTERNS[name]
    rate = float(r.uniform(*(pace or p['pace'])))
    filled, room = None, None
    if dur and ring is not None and name in GAPS and (events or 2) > 1:
        n, gap, fits = fill(name, events, dur, rate, ring, lo)
        filled = {'gap': round(gap, 3), 'fits': bool(fits), 'dur': float(dur) if fits else None}
        if lo is not None:
            filled['lo'] = float(lo)
        rate = 1.0 / gap
    elif dur and ring is not None and name in OWN and (events or 2) > 1:
        room = max(dur - LEAD - ring, 0.0)                # from the first onset to the last
        n = count(name, r, events, None, rate, usual)     # a scatter's own number (`settle` may add to it)
        if name == 'shake' and not events:
            n = int(np.clip(round(room * rate) + 1, 2, MOST))     # a shake keeps its pace over the length
    else:
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
        first, own = float(r.uniform(0.8, 1.2)) / rate, n
        if room is not None:
            n, ratio = settle(n, first, ratio, room, events)
        gaps = first * ratio ** np.arange(n - 1) * np.exp(r.normal(0.0, p['timing'], n - 1))
        gaps = np.maximum(gaps, 0.012)
        # the heap falls as far over a longer settle as over its own: the step per impact shrinks with their number
        db = (-np.arange(n) * float(r.uniform(1.5, 3.0)) * (own - 1) / max(n - 1, own - 1, 1)
              + r.normal(0.0, p['level'], n))
        db[0] = 0.0
        size = -0.15 * np.arange(n) / max(n - 1, 1)       # the later, smaller settling bits
    elif name == 'shake':
        gaps *= np.exp(np.clip(r.normal(0.0, p['timing'], n - 1), -1.0, 1.0))
        swings = int(r.integers(2, 4))
        if room is not None:                              # a longer shake repeats its motion
            swings = max(swings, int(round(room / SWING)))
        phase = np.linspace(0.0, swings * np.pi, n)
        db += 6.0 * (np.abs(np.sin(phase + 0.3)) - 1.0)   # louder where each swing turns
        gaps = np.maximum(gaps, 0.015)
    elif name == 'pair':
        gaps *= float(r.uniform(0.8, 1.25))
        db[1:] -= float(r.uniform(1.5, 4.0))               # the answer is softer
    elif name == 'row':
        gaps *= 1.0 + np.clip(r.normal(0.0, p['timing'], n - 1), -0.12, 0.12)
    if room is not None:                 # the shape kept, the last onset `ring` s before the end (stretched below)
        gap = room / max(n - 1, 1)
        filled = {'gap': round(gap, 3), 'fits': True, 'dur': float(dur)}
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
    if filled and span > 0:              # the human spread kept, the whole stretched so the last lands where planned
        onsets = LEAD + (onsets - LEAD) * (n - 1) * gap / span
        span = float(onsets[-1] - onsets[0])
    return {'pattern': name, 'onsets': [float(t) for t in onsets], 'gains': [float(10 ** (d / 20)) for d in db],
            'members': [int(m) for m in order], 'nudge': nudge,
            'pace': round((n - 1) / span, 2) if span > 0 else None, **(filled or {})}


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
