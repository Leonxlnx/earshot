"""Notes in time: the small figures sound design is made of (a two-note confirmation, a notification motif, a chime
run, a rolled chord, a logo) played by any voice.

    from earshot.synth import phrase
    y = phrase.play([(0.0, 'C6'), (0.09, 'E6'), (0.18, 'G6', 1.0)], 'glockenspiel', dsp.rng(1), vel=0.6)
    y = phrase.play(phrase.steps('D5', [0, 7, 12], 0.11), 'kalimba', dsp.rng(2), accent=-1)

A figure played by a machine has every note at one level on a grid; played by a person the first note leads, the
last one lands, timing breathes by a few milliseconds and no two strikes are the same. `play` does that part: each
note is its own take of the voice (its own strike), `vary` moves time and velocity slightly, `accent` lifts one
note, `spread` places the notes across the stereo field.
"""
import numpy as np

from ..forge import dsp, knobs
from . import voices


def steps(root, semitones, step_s, swing=0.0):
    """[(t, hz)] for notes `semitones` above `root`, `step_s` apart (swing 0..0.5 delays every second note)"""
    f0 = knobs.hz(root)
    return [(i * step_s * (1 + (swing if i % 2 else 0.0)), f0 * 2 ** (s / 12)) for i, s in enumerate(semitones)]


def play(notes, voice, r, vel=0.7, dur=None, vary=0.3, accent=None, accent_db=2.5, spread=0.0, tail=None, **controls):
    """mix notes [(t_s, pitch[, vel_scale[, dur_s]])] of one voice into a stereo phrase. vel is the phrase's level
    (each note's vel_scale multiplies it); dur damps every note that gives none (None = ring); accent = the index of
    the note to lift by accent_db (-1 = the last); spread 0..1 pans the notes alternately left and right, wider
    toward the end; tail = seconds the last note may ring (None = fully). Controls go to the voice."""
    notes = [tuple(n) + (None,) * (4 - len(n)) for n in notes]
    k = len(notes)
    if not k:
        raise ValueError('a phrase needs at least one note')
    if accent is not None:
        accent %= k
    takes = []
    for i, (t, pitch, vs, nd) in enumerate(notes):
        j = r.standard_normal(2)
        v = float(np.clip(vel * (1.0 if vs is None else vs) * (1 + 0.06 * vary * j[0]), 0.02, 1.0))
        t = max(0.0, t + (0.004 * vary * j[1] if i else 0.0))
        y = voices.play(voice, pitch, vel=v, dur=nd if nd is not None else dur, r=r, vary=vary, **controls)
        if tail is not None and i == k - 1 and len(y) > dsp.n_of(tail):
            y = dsp.fade(y[:dsp.n_of(tail)], 0.0, min(0.25 * tail, 0.08))
        g = 10 ** (accent_db / 20) if accent is not None and i == accent else 1.0
        pan = spread * ((i % 2) * 2 - 1) * (0.35 + 0.65 * i / max(1, k - 1)) if spread else 0.0
        takes.append((dsp.pan_to_stereo(y, pan) if pan else y, t, g))
    n = max(dsp.n_of(t) + len(y) for y, t, _ in takes)
    out = np.zeros((n, 2))
    for y, t, g in takes:
        dsp.place(out, y, t, g)
    return out
