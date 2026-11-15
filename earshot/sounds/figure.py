"""Tonal designs as figures: a few notes in time, played by whichever voice the request or the theme names.

A confirmation, a notification, a toggle or a logo is a gesture (which way it moves, by which interval, how fast,
where the weight falls); the instrument is a second, independent choice. Keeping the two apart is what lets one
catalogue serve a glassy finance app, a wooden meditation timer and an 8-bit game, and what makes a themed set sound
like one product: every figure of the set is played by the theme's voice, in its key, in its room.

    from earshot.sounds import figure
    FIG = dict(steps=(0, 4), gap=0.085, root='G5', last=0.5)       # two notes, a major third up, the last one rings
    y = figure.play(c, r, **FIG)                                    # inside a design function

The shared axes mean the same for every figure: energy = how hard it is played, hard / bright / size go to the voice,
ring scales how long notes sound, motion the speed of the figure, space the room, width the spread of the notes.
"""
import numpy as np

from ..synth import phrase, space, voices

DEFAULT_VOICE = 'vibraphone'


def play(c, r, steps=(0,), gap=0.09, root='C6', vel=0.6, vels=None, gate=None, last=0.6, accent=-1, room='room',
         degrees=False, spread=0.25, voice=None):
    """a figure as stereo audio. steps: semitones above the root (or scale degrees of the theme's scale with
    degrees=True), one per note; gap: seconds between note starts (a number, or one per interval); vel: how hard it
    is played (0..1), vels: each note's share of it; gate: seconds an inner note sounds before it is damped (None =
    until just after the next one starts), last: seconds the final note rings; accent: the note that leads (index,
    -1 = the last, None = none); room: the space it sits in (the theme's space wins); spread: how far the notes sit
    apart in the stereo field; voice: the figure's own default voice"""
    v = c.voice or voice or DEFAULT_VOICE
    hz = c.degrees(steps, root) if degrees else c.semis(steps, root)
    k = len(hz)
    speed = c.lerp('motion', 1.4, 0.7, 1.0, log=True)
    gaps = [float(g) * speed for g in (np.broadcast_to(gap, (max(1, k - 1),)) if k > 1 else [])]
    ring = c.lerp('ring', 0.45, 2.2, 1.0, log=True)
    t, notes = 0.0, []
    for i, f in enumerate(hz):
        if i == k - 1:
            d = last * ring
        else:
            d = (gate if gate is not None else gaps[i] * 1.15) * min(ring, 1.5)
        notes.append((t, f, 1.0 if vels is None else vels[i], d))
        if i < k - 1:
            t += gaps[i]
    level = float(np.clip(vel * c.db('energy', 7.0), 0.05, 1.0))
    y = phrase.play(notes, v, r, vel=level, vary=c.vary, accent=accent if k > 1 else None,
                    spread=spread * c.lerp('width', 0.0, 2.2, 1.0), **c.axes('bright', 'hard', 'size'), ring=c.ring)
    return space.room(y, c.room(room), amount=c.space)


def register(design, table, domains, axes=('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy', 'motion')):
    """register a table of figures {id: dict(kind, says, uses, tags=(), prior=1.0, allows=(), fig=dict(play args))}
    as designs that any voice can play"""
    for sid, f in table.items():
        fig = f['fig']

        def fn(c, r, fig=fig):
            return play(c, r, **fig)
        n = len(fig.get('steps', (0,)))
        g = fig.get('gap', 0.09)
        dur = float(np.sum(np.broadcast_to(g, (max(1, n - 1),)))) * (n > 1) + fig.get('last', 0.6)
        design(sid, kind=f['kind'], says=f['says'], uses=f['uses'], tags=tuple(f.get('tags', ())) + ('tonal',),
               domains=f.get('domains', domains), dur=round(dur, 2), pitch=fig.get('root', 'C6'), axes=axes,
               voice=fig.get('voice') or DEFAULT_VOICE, allows=f.get('allows', ()), prior=f.get('prior', 1.0))(fn)


def voice_names():
    return sorted(voices.table())
