"""Real instruments played note by note: a marimba, a harp, pizzicato strings and an upright piano, learned from
single-note recordings in the public domain.

Why. Tonal interface sounds played by synthesised voices scored 1-2 of 5 in listening tests (heard as technical
and synthetic), real recorded drums 3-4, and a take out of earshot.synth.resyn is heard as the recording itself.
So the notes of a figure are real notes: each instrument's recordings (VSCO-2 Community Edition
and the Versilian Community Sample Library, CC0) are kept as a few kilobytes of numbers each in earshot/data/notes/
(never audio; credits in CREDITS.md there) and played as new takes, the way earshot.sounds.drums plays real strokes.

How a note is played (note()):
    which    the learned note nearest to the pitch asked for among the recordings of the dynamic nearest to the
             energy (each dynamic has a velocity 0..1 in the index; energy -1..1 asks for 0.5 + 0.5 energy); the
             seed walks the takes (round robin) of that note
    pitch    the take moved to the pitch asked for by at most SHIFT semitones, with the tuning correction measured
             on the recording (its deviation from equal temperament at A = 440 Hz), so a note is in tune within
             5 cents. resyn moves pitch by resampling with its time-warp compensating: the attack is moved with the
             pitch exactly as a resampled recording would be, its length kept
    level    every take is set by its own loudest 50 ms to REF, measured after it was moved to its pitch (a
             recording moved a semitone can lie 2-3 dB from its level at its own pitch), and every learned note's
             brightness was levelled against the others (`tilt` dB per octave in the index, where a recording stood
             out of its neighbours): a chromatic run has no jump where the source note changes; energy then moves
             the level STEP dB per unit around REF
    length   None: the note rings out at its natural decay (the excerpt runs until it has fallen under the noise
             floor, at most about 4 s, its end faded). A length shorter than that damps it as a player's hand or a
             damper would: a fade of DAMP s (150 ms at C3 .. 60 ms at C6) that starts at `length`, never a cut
    onset    a note starts LEAD s before it first comes within 20 dB of its peak (onset()): no finger noise or early
             player ahead of the note, so the notes of a figure land where they are placed
    image    stereo as recorded by a close pair: the direct sound in front; add a room with earshot.synth.space

    from earshot.synth import played
    y = played.note('marimba', 67, energy=0.0, seed=1)          # G4, stereo (n, 2), rings out
    y = played.note('harp', 72, length=0.3, seed=2)              # C5 damped after 0.3 s
    played.instruments()['pizzicato']['range']                   # (lowest, highest) MIDI note it plays

As voices of earshot.synth.voices (VOICES, play()) they take the names of the models they replace: `marimba`,
`harp`, `pizzicato`, and `felt_piano` played by the upright, so a figure, a theme or a request reaches them by name.
"""
import copy
import json
import os
from functools import lru_cache

import numpy as np

from ..forge import dsp, knobs
from . import resyn

SR = dsp.SR
STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'notes')
SHIFT = 3.0                 # semitones a learned note is moved at most
REF = -16.0                 # dBFS: the loudest 50 ms of a note at energy 0
STEP = 4.0                  # dB of level per unit of energy (the dynamic chosen changes the timbre as well)
DAMP = (0.15, 0.06)         # s: the damping fade at MIDI 48 (C3) .. 84 (C6): a low string takes longer to stop
LEAD = 0.01                 # s a note keeps before it first comes within 20 dB of its peak (onset())
TILT = 2.0                 # dB per octave of a voice's bright = +-1 (as earshot.sounds.drums)


# ------------------------------------------------------------------------------------------------------ the store
@lru_cache(maxsize=1)
def index():
    """the store's index: {instrument: entry} (entry: says, dynamics {name: velocity 0..1}, notes [{file, midi,
    dynamic, take, len_s, onset_s, cents, gain_db, tilt, source, credit}])"""
    with open(os.path.join(STORE, 'index.json'), encoding='utf-8') as fh:
        return {e['instrument']: e for e in json.load(fh)}


@lru_cache(maxsize=None)
def model(file):
    """the model of one learned note (its path in the store), read once"""
    return resyn.load(os.path.join(STORE, file))


def instruments():
    """{instrument: {'range': (lowest, highest) MIDI note it plays, 'learned': (lowest, highest) learned note,
    'dynamics': [names, softest first], 'notes': count, 'says': description}}"""
    out = {}
    for name, e in index().items():
        ms = [n['midi'] for n in e['notes']]
        lo, hi = min(ms), max(ms)
        out[name] = {'range': (int(lo - SHIFT), int(hi + SHIFT)), 'learned': (lo, hi),
                     'dynamics': sorted(e['dynamics'], key=e['dynamics'].get), 'notes': len(e['notes']),
                     'says': e['says']}
    return out


def midi_of(pitch):
    """a MIDI number (float) from a MIDI number, a note name ('G4') or Hz (a number over 127)"""
    if isinstance(pitch, str) or float(pitch) > 127.0:
        return 69.0 + 12.0 * np.log2(knobs.hz(pitch) / 440.0)
    return float(pitch)


def choose(instrument, midi, energy=0.0, seed=0):
    """the learned note (an index row) that plays `midi` at `energy` for a seed (module doc)"""
    if instrument not in index():
        raise ValueError(f'unknown instrument {instrument!r}: one of {", ".join(sorted(index()))}')
    e = index()[instrument]
    want = 0.5 + 0.5 * float(np.clip(energy, -1.0, 1.0))
    order = sorted(e['dynamics'], key=lambda d: abs(e['dynamics'][d] - want))
    for dyn in order:                       # the nearest dynamic that holds a note within SHIFT semitones
        ns = [n for n in e['notes'] if n['dynamic'] == dyn]
        best = min(abs(n['midi'] - midi) for n in ns)
        if best <= SHIFT + 1e-9:
            break
    else:
        lo, hi = instruments()[instrument]['range']
        raise ValueError(f'{instrument} plays MIDI {lo} .. {hi}, not {midi:.1f}')
    near = min(ns, key=lambda n: (abs(n['midi'] - midi), n['midi']))
    takes = sorted((n for n in ns if n['midi'] == near['midi']), key=lambda n: n['take'])
    return takes[int(seed) % len(takes)]


def loudest(y, s=0.05):
    """RMS of the loudest s seconds over both channels, dBFS"""
    p = np.mean(np.atleast_2d(np.asarray(y).T).T ** 2, axis=1)
    k = max(2, dsp.n_of(s))
    if len(p) <= k:
        return float(10 * np.log10(p.mean() + 1e-30))
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(10 * np.log10(np.max(c[k:] - c[:-k]) / k + 1e-30))


def damp_s(midi):
    """s a hand or a damper takes to stop a note at `midi`: DAMP[0] at C3 .. DAMP[1] at C6"""
    u = float(np.clip((midi - 48.0) / 36.0, 0.0, 1.0))
    return DAMP[0] + (DAMP[1] - DAMP[0]) * u


def damp(y, length, midi, fade=None):
    """y damped at `length` s: a cos^2 fade of `fade` s (None: damp_s(midi), longer for a low note) from there, then
    the end"""
    a = dsp.n_of(length)
    if a >= len(y):
        return y
    k = min(len(y) - a, dsp.n_of(damp_s(midi) if fade is None else fade))
    y = np.array(y[:a + k])
    g = np.cos(np.linspace(0.0, np.pi / 2, k)) ** 2
    y[a:] *= g[:, None] if y.ndim == 2 else g
    return y


def onset(y):
    """y from LEAD s before it first comes within 20 dB of its peak, the cut faded in over 4 ms: a harpist's finger
    on the string or the first players of a section ahead of the others are a few tens of ms of quiet sound before
    the note, which would put every note of a figure late"""
    e = np.abs(y).max(axis=1)
    a = int(np.argmax(e >= 0.1 * e.max())) - dsp.n_of(LEAD)
    if a <= dsp.n_of(0.003):
        return y
    y = np.array(y[a:])
    k = dsp.n_of(0.004)
    y[:k] *= (np.sin(np.linspace(0.0, np.pi / 2, k)) ** 2)[:, None]
    return y


def render(row, midi, r, energy=0.0, length=None, bright=0.0, hard=0.0, vary=0.0):
    """a take of one learned note (an index row) moved to `midi`: stereo (n, 2) at REF + STEP * energy dBFS (its
    loudest 50 ms). bright: dB per octave on top of the note's own levelling; hard: -1 .. 1 the strike softer and
    duller .. harder and brighter (resyn's onset control); vary: 0 .. 1 small random differences of the take"""
    shift = float(midi) - row['midi']
    if abs(shift) > SHIFT + 1e-9:
        raise ValueError(f'a learned note is moved at most {SHIFT} semitones, not {shift:.2f}')
    hard = float(np.clip(hard, -1.0, 1.0))
    ref = copy.deepcopy(r) if (hard or bright) else None
    y = onset(resyn.render(model(row['file']), r, pitch=shift - row['cents'] / 100.0, bright=row['tilt'] + bright,
                           hard=hard, vary=0.3 * float(vary)))
    # the level of the take as moved: the same recording moved a semitone can lie 2-3 dB from its level at its own
    # pitch, so the gain is measured on the take itself (with bright and hard at 0, so they keep their own effect)
    z = y if ref is None else onset(resyn.render(model(row['file']), ref, pitch=shift - row['cents'] / 100.0,
                                                 bright=row['tilt'], vary=0.3 * float(vary)))
    y = y * 10 ** ((REF + STEP * float(np.clip(energy, -1.0, 1.0)) - loudest(z)) / 20)
    if length is not None:
        y = damp(y, float(length), float(midi))
    return y


def note(instrument, midi, energy=0.0, length=None, seed=0, **controls):
    """one real note: stereo float64 (n, 2), its attack within the first few ms. midi: MIDI number (float: between
    semitones), note name or Hz; energy -1..1: how hard it is played (dynamic and level); length: s before the note
    is damped (None: it rings out); seed: which take and which noise (the same arguments give the same samples);
    controls: bright, hard, vary (render())"""
    m = midi_of(midi)
    row = choose(instrument, m, energy, seed)
    return render(row, m, dsp.rng(int(seed)), energy, length, **controls)


# the real instruments as voices of earshot.synth.voices: voice name -> instrument. `felt_piano` is played by the
# upright (its soft dynamic is the default): the store holds no felted piano, and a theme or a request that asks for
# a soft piano gets a real one
VOICES = {'marimba': 'marimba', 'harp': 'harp', 'pizzicato': 'pizzicato', 'felt_piano': 'upright_piano'}


def _voices():
    out = {}
    for v, inst in VOICES.items():
        if inst not in index():
            continue
        lo, hi = instruments()[inst]['range']
        names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        out[v] = {'says': [index()[inst]['says'], f'a real {inst.replace("_", " ")} note'],
                  'range': (f'{names[lo % 12]}{lo // 12 - 1}', f'{names[hi % 12]}{hi // 12 - 1}'),
                  'default_pitch': 'C5', 'kind': 'real', 'instrument': inst,
                  'ring_s': round(float(np.median([n['len_s'] for n in index()[inst]['notes']])), 2)}
    return out


TABLE = _voices()                # what earshot.synth.voices lists: {voice: says, range, default_pitch, ring_s ...}


def play(voice, hz, vel=0.7, dur=None, r=None, bright=0.0, hard=0.0, vary=0.0):
    """one note of a real instrument as earshot.synth.voices plays a voice: vel 0..1 is the energy 2 vel - 1.2
    (0.6 = as learned), dur None = it rings out, else it is damped at dur s; bright -1..1 is a tilt of -+TILT dB per
    octave; the take is drawn from r"""
    r = r if r is not None else dsp.rng(0)
    seed = int(r.integers(1 << 30))
    energy = float(np.clip(2.0 * float(vel) - 1.2, -1.0, 1.0))
    return note(VOICES[voice], midi_of(hz), energy, dur, seed, bright=TILT * float(bright), hard=hard, vary=vary)


def credit(instrument):
    """the recordings an instrument was learned from: [{title, author, page, licence, licence_url}]"""
    seen, out = set(), []
    for n in index()[instrument]['notes']:
        if n['source'] not in seen:
            seen.add(n['source'])
            out.append(dict(n['credit']))
    return out
