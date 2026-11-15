"""Tonal designs as figures: a few notes in time, played by whichever voice the request or the theme names.

The default voice is a soft rounded pluck with some body under it, the register of product sound. A bell, a
glockenspiel or a chime plays only when the request or the theme asks for one. Taps, toggles and hovers lead with a
tactile touch (a key's thock and click, `touch`) that the note only colours.

A confirmation, a notification, a toggle or a logo is a gesture (which way it moves, by which interval, how fast,
where the weight falls); the instrument is a second, independent choice. Keeping the two apart is what lets one
catalogue serve a glassy finance app, a wooden meditation timer and an 8-bit game, and what makes a themed set sound
like one product: every figure of the set is played by the theme's voice, in its key, in its room.

    from earshot.sounds import figure
    FIG = dict(steps=(0, 4), gap=0.085, root='G5', last=0.5)       # two notes, a major third up, the last one rings
    y = figure.play(c, r, **FIG)                                    # inside a design function

The shared axes mean the same for every figure: energy = how hard it is played, hard / bright / size go to the voice,
ring scales how long notes sound, motion the speed of the figure, space the room, width the spread of the notes.
A length in the request sets how long the figure sounds before its room dies away; a minor key in the request or
theme turns the figure's major thirds, sixths and sevenths into minor ones; a material without a voice picks the
voice made of it.
"""
import numpy as np

from ..forge import dsp
from ..synth import phrase, space, voices

DEFAULT_VOICE = 'soft_pluck'
SUSTAIN_FLOOR = 0.11              # s: the shortest an inner note of a sustaining voice sounds (the last: 1.6 x)
CHORD_GAP = 0.03                  # s: notes this close are one chord (or a grace note): the earlier one rings on
MINOR = ('minor', 'minor_pentatonic')
MINOR_STEP = {4: 3, 9: 8, 11: 10}    # a major third, sixth and seventh as their minor forms


def touch(c, r, times, level, pitch=1.0):
    """the tactile part of a tap, a key or a switch, one per onset in `times` (s), stereo: a low thock (the body of
    a key or a panel, 70-110 Hz, gone in about 60 ms), a short knock of the cap or the lever (a ring near 1.4 kHz,
    5 ms) and a click of the contact (a 1.5 ms noise burst around 4 kHz). The shape follows the re-made keyboard key
    the owner rated 4 of 5 (fit/key/thock: most energy below 150 Hz and above 1 kHz, attack under 1 ms). `pitch`
    (one per onset or one) moves all three: a switch's second part is a little higher or lower than its first.
    hard sharpens and raises the click, size lowers and lengthens the thock, bright opens the click."""
    from ..synth import texture
    t = np.atleast_1d(np.asarray(times, np.float64))
    p = np.broadcast_to(np.asarray(pitch, np.float64), t.shape)
    k = len(t)
    out = np.zeros((dsp.n_of(float(t.max()) + 0.25), 2))
    big, hard, bright = c.lerp('size', 0.7, 1.4, 1.0, log=True), c.db('hard', 5.0), c.db('bright', 5.0)
    j = np.exp(0.08 * r.standard_normal((3, k)))
    texture.rings(out, t, 0.55 * level * j[0], 0.0, 88.0 * p / big * j[1], 0.016 * big, 0.0006, -1.5, wrap=False)
    texture.rings(out, t + 0.0004, 0.22 * level * hard * j[2], r.uniform(-0.2, 0.2, k), 1400.0 * p * j[0], 0.005,
                  0.0002, wrap=False)
    texture.bursts(out, t, 1.6 * level * hard * bright, r.uniform(-0.25, 0.25, k), 4200.0 * p * bright ** 0.3, 1.2,
                   0.0015 / hard ** 0.3, r, wrap=False)
    return out


def play(c, r, steps=(0,), gap=0.09, root='C6', vel=0.6, vels=None, gate=None, last=0.6, accent=-1, room='room',
         degrees=False, spread=0.35, voice=None, avoid=(), longest=None, tap=0.0, tap_pitch=None):
    """a figure as stereo audio. steps: semitones above the root (or scale degrees of the theme's scale with
    degrees=True), one per note; gap: seconds between note starts (a number, or one per interval); vel: how hard it
    is played (0..1), vels: each note's share of it; gate: seconds an inner note sounds before it is damped (None =
    until just after the next one starts), last: seconds the final note rings (and every note of a chord: notes
    closer than CHORD_GAP); accent: the note that leads (index,
    -1 = the last, None = none); room: the space it sits in (the theme's space wins); spread: how far the notes sit
    apart in the stereo field; voice: the figure's own default voice; avoid: materials whose voices cannot carry this
    figure (two short low notes on a metal bar are heard as an electronic beep): a theme's voice of such a material
    gives way to the figure's own; a voice asked for by name is always played; longest: seconds after which the
    figure has faded out whatever the voice (a hover or a tap on a piano must not ring for a second: the fade
    starts at 55 % of it; `ring` scales it, a requested length overrides it); tap: the level of a tactile touch
    (touch()) under every note, against the notes' level (0 = none: a tap, a toggle or a key leads with it and lets
    the note only colour it), tap_pitch: its pitch ratio per note"""
    v = c.voice or voice or DEFAULT_VOICE
    if avoid and c._voice is None and voices.info(v).get('material') in avoid:     # a voice asked for by name
        v = voice or DEFAULT_VOICE                                                 # is played as asked
    c.played = v
    if not degrees and c.scale() in MINOR:       # in a minor key the thirds, sixths and sevenths are the minor ones
        steps = [MINOR_STEP.get(s % 12, s % 12) + 12 * (s // 12) for s in steps]
    hz = c.degrees(steps, root) if degrees else c.semis(steps, root)
    k = len(hz)
    speed = c.lerp('motion', 1.4, 0.7, 1.0, log=True)
    gaps = [float(g) * speed for g in (np.broadcast_to(gap, (max(1, k - 1),)) if k > 1 else [])]
    ring = c.lerp('ring', 0.45, 2.2, 1.0, log=True)
    want = c.length()
    if want:                                     # a length was asked for: the notes fit into the first 60 % of it
        run = sum(gaps)                          # (played faster if they must) and the last one rings for the rest
        if run > 0.6 * want:
            gaps = [g * 0.6 * want / run for g in gaps]
        last, ring = max(0.04, want - sum(gaps)), 1.0
    # a sustaining voice (a metal bar, a bell, a pad) damped after a few tens of ms is a bare tone burst, which is
    # heard as an electronic beep: such a voice keeps each note long enough to be heard as its instrument
    info = voices.info(v)                # an instrument that rings over a second, or a patch that sustains
    sustains = float(info.get('ring_s') or 0.0) > 1.0 or info.get('kind') in ('bell', 'pad', 'key', 'lead')
    floor = SUSTAIN_FLOOR if sustains else 0.0
    t, notes = 0.0, []
    for i, f in enumerate(hz):
        if i == k - 1 or gaps[i] <= CHORD_GAP:           # the last note, and a note the next one joins as a chord
            d = max(last * ring, 1.6 * floor)
        else:
            d = max((gate if gate is not None else gaps[i] * 1.15) * min(ring, 1.5), floor)
        notes.append((t, f, 1.0 if vels is None else vels[i], d))
        if i < k - 1:
            t += gaps[i]
    level = float(np.clip(vel * c.db('energy', 7.0), 0.05, 1.0))
    y = phrase.play(notes, v, r, vel=level, vary=c.vary, accent=accent if k > 1 else None,
                    spread=spread * c.lerp('width', 0.0, 2.2, 1.0), **c.axes('bright', 'hard', 'size'), ring=c.ring)
    if tap:                              # a tactile figure: the touch leads, the notes colour it
        th = touch(c, r, [n[0] for n in notes], tap * level, 1.0 if tap_pitch is None else tap_pitch)
        if len(th) > len(y):
            y = np.pad(y, ((0, len(th) - len(y)), (0, 0)))
        y[:len(th)] += th
    if want and len(y) < 0.75 * want * dsp.SR:
        c.used.discard('dur')            # the voice rings shorter than the length asked for: say it was not kept
    cut = dsp.n_of(longest * ring) if (longest and not want) else None    # a requested length is kept as asked
    if cut and len(y) > cut:
        y = y[:cut]
        k = max(2, int(0.45 * cut))
        y[-k:] *= (np.cos(np.linspace(0, np.pi / 2, k)) ** 2)[:, None]
    return space.room(y, c.room(room), amount=c.space, r=r)


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
        # a figure of pitched notes is what requests call a chime, a ding or a bell, whatever its job
        uses = tuple(f['uses']) + (('chime',) if f['kind'] in ('chime', 'figure', 'note', 'stinger') else ())
        design(sid, kind=f['kind'], says=f['says'], uses=uses, tags=tuple(f.get('tags', ())) + ('tonal',),
               domains=f.get('domains', domains), dur=round(dur, 2), pitch=fig.get('root', 'C6'), axes=axes,
               voice=fig.get('voice') or DEFAULT_VOICE, allows=f.get('allows', ()), prior=f.get('prior', 1.0))(fn)


def voice_names():
    return sorted(voices.table())
