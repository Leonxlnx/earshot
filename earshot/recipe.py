"""earshot recipe: a sound the catalogue has nothing of, built from the catalogue's own parts.

    earshot recipe --example                          # the form and three recipes to start from
    earshot recipe --list                             # the cookbook: recipes that ship with earshot
    earshot recipe dragon_roar                        # one of them, to change
    earshot make --recipe roar.json -o roar.wav       # render a recipe (`-` reads it from standard input)

    from earshot.make import make
    make(recipe={'name': 'dragon_roar', 'says': 'a dragon roaring', 'layers': [
        {'sound': 'lion/roar', 'pitch': -7},
        {'sound': 'fire/crackle', 'gain': -12, 'at': 0.3, 'dur': 2.5},
        {'source': 'noise', 'dur': 2.5, 'lowpass': [70, 160], 'gain': -14, 'swell': 0.6, 'decay': 1.2}]},
        out='roar.wav')

The catalogue is built by layering, slowing, reversing and filtering real recordings (earshot.synth.design). A
recipe opens that to whoever asks: a sound is a few layers, written as JSON. A layer is a catalogue design by its id
(`earshot sounds --search`; the designs learned from recordings are the best material, they are real) or one of two
plain sources for what no recording gives: filtered noise (air, rumble; its cutoff may move) and a low sine (weight
under a hit). Each layer has its controls:

    sound | source   the design's id, or "noise" / "sine"
    seed             another take of the design (a learned design: its next recording)
    size, bright, hard, energy, ring, motion, width
                     the design's own axes, -1 .. 1, where it has them
    dur              the length asked of the design itself (an ambience any length, a whoosh, a recording within
                     0.6 to 1.8 times its own); of a source: its length
    pitch            semitones, -24 .. 12, as a tape does it: lower is also slower and bigger
    reverse          played backwards: a decay becomes a swell that ends on its strike
    cut              [from, to] in s: the part of the take that is used
    halt             s: the take runs down to a stop over that long, as a motor does when its power is cut
    lowpass, highpass
                     Hz; two numbers [from, to] move the cutoff over the layer
    pan              -1 (left) .. 1 (right)
    gain             dB against the other layers: every layer is first brought to the same peak
    at               s: where the layer starts
    fade_in, fade_out, swell, decay
                     s: short fades at its ends (an ambience gets 0.1 and 0.3 s, a source 0.01 and 0.02 s when
                     not said); a slow rise from 30 dB down; a slow fall to nothing
    events           {"count", "gap", "faster", "softer", "loose"}: the layer several times, each a new take: `gap`
                     s from one to the next, each gap `faster` times the one before (under 1: faster and faster),
                     each event `softer` dB against the one before, each up to `loose` (0 .. 1) of half its gap
                     early or late (drops, crackles, clicks that come as they come)
    noise: color     "white", "pink" or "brown"
    sine: hz, to     20 .. 150 Hz, gliding to `to`: weight, never a tone that is heard as a note

The whole has a `name`, `says` (what it is, in words), a `space` (dry, booth, room, studio, hall, plate,
cathedral), a `kind` ("event", or "ambience": levelled and faded as a bed) and a `dur` (the file is cut there with
a fade). A cookbook recipe (earshot/data/recipes) also says the requests it answers (`things`), what it is built
from in one sentence (`built`) and, when earshot's listening model does not hear it as its thing yet, `sketch`:
such a recipe is named by `make` as a start and never given as the answer. `check` answers every mistake with one
sentence that names the field and what it takes. `render` gives the
sound as every sound of earshot is given: levelled like a catalogue sound of its kind and audited (clicks, a cut
tail, silence, clipping), with where its hit is. Nothing here loads a model or a package the base install lacks.

One sound as the body of another ("X on Y": the hits of X ringing through the resonance of Y) is not in the form:
the engine has no way to take a resonance from a design that holds up for any pair of designs.
"""
import difflib
import json
import os
import sys

from . import brief, config

LAYERS = 8                 # layers of one recipe
PEAK = 0.5                 # every layer's peak before its gain: gains say how loud it sits against the others
SOURCES = ('noise', 'sine')
COLORS = ('white', 'pink', 'brown')
KINDS = ('event', 'ambience')
AXES = ('size', 'bright', 'hard', 'energy', 'ring', 'motion', 'width')
TOP = ('name', 'says', 'layers', 'dur', 'space', 'kind', 'things', 'built', 'sketch')
LAYER = ('sound', 'source', 'seed', 'dur', 'pitch', 'reverse', 'cut', 'halt', 'lowpass', 'highpass', 'pan', 'gain',
         'at', 'fade_in', 'fade_out', 'swell', 'decay', 'events', 'color', 'hz', 'to') + AXES
EVENTS = ('count', 'gap', 'faster', 'softer', 'loose')
# a number field -> (lowest, highest, what it is)
RANGE = {'dur': (0.02, 120.0, 'a length in seconds'), 'pitch': (-24.0, 12.0, 'a number of semitones'),
         'pan': (-1.0, 1.0, 'a position'), 'gain': (-40.0, 12.0, 'a level in dB'),
         'at': (0.0, 120.0, 'a start time in seconds'), 'fade_in': (0.0, 30.0, 'a time in seconds'),
         'fade_out': (0.0, 30.0, 'a time in seconds'), 'swell': (0.0, 60.0, 'a time in seconds'),
         'decay': (0.0, 60.0, 'a time in seconds'), 'halt': (0.05, 10.0, 'a time in seconds'),
         'hz': (20.0, 150.0, 'a frequency in Hz'),
         'to': (20.0, 150.0, 'a frequency in Hz'), **{a: (-1.0, 1.0, 'an amount') for a in AXES}}
CUTOFF = (20.0, 20000.0)
EVENT_RANGE = {'count': (2, 64, 'a whole number'), 'gap': (0.02, 30.0, 'a time in seconds'),
               'faster': (0.5, 2.0, 'a factor'), 'softer': (-12.0, 12.0, 'a level in dB'),
               'loose': (0.0, 1.0, 'an amount')}
EXAMPLES = ('dragon_roar', 'giant_footsteps', 'campfire_wind')     # layers and pitch; events; an ambience
FOLDER = os.path.join(config.DATA_DIR, 'recipes')


class RecipeError(ValueError):
    """a recipe that cannot be rendered; str(e) is one plain sentence that names the field and what it takes"""


# ------------------------------------------------------------------------------------------------- reading a recipe
def _number(v, where, lo, hi, what, whole=False):
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and (not whole or float(v).is_integer())
    if not ok or not lo <= v <= hi:
        raise RecipeError(f'{where} is {what} from {lo:g} to {hi:g}, not {v!r}')
    return int(v) if whole else float(v)


def _cutoff(v, where):
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise RecipeError(f'{where} is a frequency in Hz or two of them [from, to], not {list(v)!r}')
        return [_number(x, f'{where}[{i}]', *CUTOFF, 'a frequency in Hz') for i, x in enumerate(v)]
    return _number(v, where, *CUTOFF, 'a frequency in Hz')


def _fields(d, allowed, where, what):
    bad = [k for k in d if k not in allowed]
    if bad:
        raise RecipeError(f'unknown field {bad[0]!r} in {where}: {what} takes {", ".join(allowed)}')


def _layer(lay, i):
    from .sounds import registry
    w = f'layers[{i}]'
    if not isinstance(lay, dict):
        raise RecipeError(f'{w} is an object such as {{"sound": "door/slam", "gain": -6}}, not {lay!r}')
    _fields(lay, LAYER, w, 'a layer')
    if ('sound' in lay) == ('source' in lay):
        raise RecipeError(f'{w} names either a catalogue design ("sound": "door/slam") or a plain source '
                          f'("source": "noise" or "sine"), one of the two')
    out = dict(lay)
    if 'sound' in lay:
        sid = lay['sound']
        if not isinstance(sid, str) or sid not in registry.ids():
            near = difflib.get_close_matches(str(sid), registry.ids(), n=3, cutoff=0.5)
            raise RecipeError(f'{w}.sound {sid!r} is not in the catalogue'
                              + (f'; the nearest ids: {", ".join(near)}' if near else '')
                              + '; `earshot sounds --search <word>` lists what there is')
        for k in ('color', 'hz', 'to'):
            if k in lay:
                raise RecipeError(f'{w}.{k} belongs to a source ("source": "noise" or "sine"), not to a design')
    else:
        src = lay['source']
        if not isinstance(src, str) or src not in SOURCES:
            raise RecipeError(f'{w}.source is one of {", ".join(SOURCES)}, not {src!r}')
        if 'dur' not in lay:
            raise RecipeError(f'{w}.dur is missing: a source needs its length in seconds')
        own = ('color',) if src == 'noise' else ('hz', 'to')
        for k in AXES + ('seed', 'color', 'hz', 'to'):
            if k in lay and k not in own and k != 'seed':
                raise RecipeError(f'{w}.{k} does not belong to the source {src}: it takes dur, {", ".join(own)} and '
                                  'the controls every layer has (pitch, cut, lowpass, highpass, pan, gain, at, fades, '
                                  'events)')
        if src == 'sine' and 'hz' not in lay:
            raise RecipeError(f'{w}.hz is missing: a sine needs its frequency, from 20 to 150 Hz')
        if 'color' in lay and lay['color'] not in COLORS:
            raise RecipeError(f'{w}.color is one of {", ".join(COLORS)}, not {lay["color"]!r}')
    for k, (lo, hi, what) in RANGE.items():
        if k in lay:
            out[k] = _number(lay[k], f'{w}.{k}', lo, hi, what)
    if 'seed' in lay:
        out['seed'] = _number(lay['seed'], f'{w}.seed', 0, 2 ** 31 - 1, 'a whole number', whole=True)
    if 'reverse' in lay and not isinstance(lay['reverse'], bool):
        raise RecipeError(f'{w}.reverse is true or false, not {lay["reverse"]!r}')
    for k in ('lowpass', 'highpass'):
        if k in lay:
            out[k] = _cutoff(lay[k], f'{w}.{k}')
    if 'cut' in lay:
        c = lay['cut']
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            raise RecipeError(f'{w}.cut is two times in seconds [from, to], not {c!r}')
        a, b = (_number(x, f'{w}.cut[{j}]', 0.0, 120.0, 'a time in seconds') for j, x in enumerate(c))
        if b - a < 0.01:
            raise RecipeError(f'{w}.cut is [from, to] with to at least 0.01 s after from, not {list(c)!r}')
        out['cut'] = [a, b]
    if 'events' in lay:
        ev = lay['events']
        if not isinstance(ev, dict):
            raise RecipeError(f'{w}.events is an object such as {{"count": 4, "gap": 0.5}}, not {ev!r}')
        _fields(ev, EVENTS, f'{w}.events', 'events')
        for k in ('count', 'gap'):
            if k not in ev:
                raise RecipeError(f'{w}.events.{k} is missing: events take count (2 to 64) and gap (seconds from '
                                  'one to the next)')
        out['events'] = {k: _number(ev[k], f'{w}.events.{k}', *EVENT_RANGE[k], whole=k == 'count') for k in ev}
    return out


def check(recipe):
    """a recipe checked and returned with its numbers as numbers; the first mistake raises RecipeError with one
    sentence that names the field and what it takes"""
    from .synth import space
    if not isinstance(recipe, dict):
        raise RecipeError('a recipe is an object {"name": ..., "layers": [...]}, not ' + type(recipe).__name__)
    _fields(recipe, TOP, 'the recipe', 'a recipe')
    name = recipe.get('name')
    if not isinstance(name, str) or not name or not all(ch.isalnum() and ch.isascii() or ch == '_' for ch in name):
        raise RecipeError(f'name is the sound\'s name in lower-case letters, digits and "_" (it names the file), '
                          f'not {name!r}')
    layers = recipe.get('layers')
    if not isinstance(layers, list) or not 1 <= len(layers) <= LAYERS:
        raise RecipeError(f'layers is a list of 1 to {LAYERS} layers, each {{"sound": <id>, ...}} or '
                          '{"source": "noise", ...}')
    out = {**recipe, 'name': name.lower(), 'layers': [_layer(lay, i) for i, lay in enumerate(layers)]}
    if 'says' in recipe and not (isinstance(recipe['says'], str) and recipe['says'].strip()):
        raise RecipeError(f'says is what the sound is, in words ("a dragon roaring"), not {recipe["says"]!r}')
    if 'built' in recipe and not isinstance(recipe['built'], str):
        raise RecipeError(f'built is one sentence on what the sound is built from, not {recipe["built"]!r}')
    if 'things' in recipe and not (isinstance(recipe['things'], list)
                                   and all(isinstance(t, str) and t.strip() for t in recipe['things'])):
        raise RecipeError('things is a list of the requests the recipe answers, each in words ("rocket launch")')
    if 'sketch' in recipe and not isinstance(recipe['sketch'], bool):
        raise RecipeError(f'sketch is true or false, not {recipe["sketch"]!r}')
    if 'dur' in recipe:
        out['dur'] = _number(recipe['dur'], 'dur', *RANGE['dur'][:2], 'the length of the sound in seconds')
    if 'space' in recipe and not (isinstance(recipe['space'], str) and recipe['space'] in space.SPACES):
        raise RecipeError(f'space is one of {", ".join(space.SPACES)}, not {recipe["space"]!r}')
    if 'kind' in recipe and not (isinstance(recipe['kind'], str) and recipe['kind'] in KINDS):
        raise RecipeError(f'kind is one of {", ".join(KINDS)}, not {recipe["kind"]!r}')
    return out


def load(source):
    """a recipe from what a caller gives: a dict as it is, the text of one (JSON), "-" (standard input), the path
    of a JSON file, or the name of a cookbook recipe; checked (`check`)"""
    if isinstance(source, dict):
        return check(source)
    if not isinstance(source, str) or not source.strip():
        raise RecipeError('a recipe is an object, the path of a JSON file, "-" for standard input or the name of '
                          'a cookbook recipe (`earshot recipe --list`)')
    text = source.strip()
    try:
        if text == '-':
            return check(json.loads(sys.stdin.read()))
        if text.startswith('{'):
            return check(json.loads(text))
        if os.path.isfile(text):
            with open(text, encoding='utf-8-sig') as fh:
                return check(json.load(fh))
    except json.JSONDecodeError as e:
        raise RecipeError(f'the recipe is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}') from None
    book = cookbook()
    if text in book:
        return check(book[text])
    near = difflib.get_close_matches(text, list(book), n=3, cutoff=0.5)
    raise RecipeError(f'no file and no cookbook recipe is called {text!r}'
                      + (f'; the nearest: {", ".join(near)}' if near else '') + '; `earshot recipe --list` lists them')


# ------------------------------------------------------------------------------------------------------ the cookbook
_book = None


def cookbook():
    """the recipes that ship with earshot, by name: {name: recipe}"""
    global _book
    if _book is None:
        _book = {}
        if os.path.isdir(FOLDER):
            for f in sorted(os.listdir(FOLDER)):
                if f.endswith('.json'):
                    with open(os.path.join(FOLDER, f), encoding='utf-8') as fh:
                        r = json.load(fh)
                    _book[r['name']] = r
    return _book


def things():
    """the requests the cookbook answers: {the thing in words ("rocket launch"): the recipe's name}"""
    return {t.lower(): name for name, r in cookbook().items() for t in r.get('things', ())}


def answer(b, why=(), sketch=False):
    """the cookbook recipe a request (a brief, earshot.brief.read) names, or None: every word of one of the
    recipe's `things` is a word of the request's own sound (by its stem; the thing of most words wins). `why`: the
    words the catalogue's first design was found by; when they hold every word of the thing, the catalogue has
    that sound itself and its design answers. A recipe marked `sketch` (earshot's listening model does not hear
    it as its thing yet) is no answer: sketch=True finds those instead, for `make` to name as a start"""
    ctx = {brief.stem(w) for w in b.get('context_words') or ()}
    said = {brief.stem(w) for w in brief.WORD.findall((b.get('text') or '').lower())} - ctx
    found = {w.split(':')[-1] for w in why or ()}
    found |= {brief.stem(w) for w in found}
    best = None
    for thing, name in things().items():
        if bool(cookbook()[name].get('sketch')) != bool(sketch):
            continue
        stems = {brief.stem(w) for w in brief.WORD.findall(thing) if w not in brief.L.STOP}
        if stems and stems <= said and not stems <= found and (best is None or len(stems) > best[0]):
            best = (len(stems), name)
    return cookbook()[best[1]] if best else None


# -------------------------------------------------------------------------------------------------------- the engine
def _stereo(y):
    import numpy as np
    y = np.asarray(y, np.float64)
    return np.array(y) if y.ndim == 2 else np.stack([y, y], axis=1)


def _moving(y, lo, hi):
    """y through a low-pass at `lo` and / or a high-pass at `hi` (12 dB an octave), each a frequency or [from, to]
    moving over the layer on a logarithmic line: done on the short-time spectrum, so the cutoff glides"""
    import numpy as np
    from scipy import signal
    from .forge import dsp
    seg = 2048
    if len(y) < seg:
        y = np.concatenate([y, np.zeros((seg - len(y), 2))])
    out = np.zeros_like(y)
    for ch in range(2):
        f, t, z = signal.stft(y[:, ch], fs=dsp.SR, nperseg=seg, noverlap=seg * 3 // 4)
        u = np.linspace(0.0, 1.0, z.shape[1])[None, :]
        g = np.ones_like(z, dtype=np.float64)
        ff = np.maximum(f, 1e-3)[:, None]
        for cut, low in ((lo, True), (hi, False)):
            if cut is None:
                continue
            a, b = cut if isinstance(cut, list) else (cut, cut)
            fc = a * (b / a) ** u
            g *= 1.0 / np.sqrt(1.0 + ((ff / fc) if low else (fc / ff)) ** 4)
        out[:, ch] = signal.istft(z * g, fs=dsp.SR, nperseg=seg, noverlap=seg * 3 // 4)[1][:len(y)]
    return out


def _filters(y, lay):
    from .forge import dsp
    lo, hi = lay.get('lowpass'), lay.get('highpass')
    if isinstance(lo, list) or isinstance(hi, list):
        return _moving(y, lo, hi)
    if lo is not None:
        y = dsp.butter(y, 'low', lo, order=2)
    if hi is not None:
        y = dsp.butter(y, 'high', hi, order=2)
    return y


def _slope(y, seconds, db, head):
    """y rising from `db` under its level over its first `seconds` (head) or falling to it over its last, straight
    in dB, the far end of the slope faded to nothing over 20 ms"""
    import numpy as np
    from .forge import dsp
    k = min(len(y), dsp.n_of(seconds))
    if k < 2:
        return y
    w = 10 ** (db * (1.0 - np.linspace(0.0, 1.0, k)) / 20)
    e = min(k, dsp.n_of(0.02))
    w[:e] *= np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, e))
    if head:
        y[:k] *= w[:, None]
    else:
        y[len(y) - k:] *= w[::-1, None]
    return y


def _source(lay, r):
    """a plain source, stereo: noise of a colour (two channels that share most of their sound), or a low sine
    gliding from hz to `to`"""
    import numpy as np
    from .forge import dsp
    if lay['source'] == 'noise':
        mid, a, b = (dsp.noise(lay['dur'], r, lay.get('color', 'pink')) for _ in range(3))
        return np.stack([mid + 0.4 * a, mid + 0.4 * b], axis=1)
    n = dsp.n_of(lay['dur'])
    hz = lay['hz'] * (lay.get('to', lay['hz']) / lay['hz']) ** np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * np.cumsum(hz) / dsp.SR)
    return np.stack([y, y], axis=1)


def _one(lay, seed, i, notes):
    """one event of a layer, stereo, before its gain: the design's take (or the source) under the layer's
    controls, moved in pitch as a tape does, reversed, cut, filtered, panned, brought to PEAK and faded"""
    from fractions import Fraction
    import numpy as np
    from scipy import signal
    from .forge import dsp
    from .sounds import registry
    from .synth import design as D
    if 'sound' in lay:
        d = registry.get(lay['sound'])
        req = {a: lay[a] for a in AXES if a in lay}
        if 'dur' in lay:
            req['dur'] = lay['dur']
        y, meta = registry.render(lay['sound'], seed=seed + lay.get('seed', 0), check=False, **req)
        ign = [k for k in meta.get('ignored', ()) if k in req]
        if ign:
            note = (f'layers[{i}]: {lay["sound"]} has no control of its {", ".join(ign)} and plays as designed'
                    + (f' (its axes: {", ".join(d.axes)})' if d.axes else ''))
            if note not in notes:
                notes.append(note)
        y = _stereo(y)
    else:
        y = _source(lay, dsp.rng(7919 * (i + 1) + 104729 * seed + lay.get('seed', 0)))
    if lay.get('pitch'):
        ratio = Fraction(2.0 ** (-lay['pitch'] / 12.0)).limit_denominator(96)
        y = signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=0)
    if lay.get('reverse'):
        y = D.reverse(y)
    if 'cut' in lay:
        y = D.span(y, lay['cut'][0], lay['cut'][1], fin=0.004, fout=0.02)
        if len(y) < dsp.n_of(0.01):
            raise RecipeError(f'layers[{i}].cut starts at {lay["cut"][0]:g} s, after the end of the take: it is '
                              'shorter than that')
    if lay.get('halt'):             # the motor runs down and grows quiet with it: the stop is no cut
        y = D.fade(D.halt(y, lay['halt']), 0.0, lay['halt'], shape='db')
    y = _filters(y, lay)
    if 'pan' in lay:
        th = (lay['pan'] + 1.0) * np.pi / 4
        y = y * np.array([np.cos(th), np.sin(th)]) * np.sqrt(2)
    peak = float(np.abs(y).max())
    if peak <= 1e-9:
        raise RecipeError(f'layers[{i}] is silent after its cut and filters: nothing of it is left to hear')
    y = y * (PEAK / peak)
    # what has no start and end of its own gets them: a source 10 and 20 ms, an ambience (a looping design) 0.1
    # and 0.3 s, unless the layer says its fades. A fade out falls straight in dB, as a decay does: the audit
    # reads a fade that drops faster at its end than before it as a cut tail
    fin, fout = (0.01, 0.02) if 'source' in lay else (0.1, 0.3) if registry.get(lay['sound']).loop else (0.0, 0.0)
    y = D.fade(D.fade(y, lay.get('fade_in', fin), 0.0), 0.0, lay.get('fade_out', fout), shape='db')
    if lay.get('swell'):
        y = _slope(y, lay['swell'], -30.0, True)
    if lay.get('decay'):
        y = _slope(y, lay['decay'], -50.0, False)
    return y


def _placed(lay, seed, i, notes):
    """the events of a layer as (samples, start s, gain as a factor)"""
    from .forge import dsp
    ev = lay.get('events') or {'count': 1, 'gap': 0.0}
    t, gap, out = lay.get('at', 0.0), ev['gap'], []
    r = dsp.rng(15485863 + seed)      # one hand for every layer: layers with the same events stay together
    for k in range(ev['count']):
        g = 10 ** ((lay.get('gain', 0.0) + k * ev.get('softer', 0.0)) / 20)
        off = ev.get('loose', 0.0) * gap * float(r.uniform(-0.5, 0.5)) if k else 0.0
        out.append((_one(lay, seed + k, i, notes), max(0.0, t + off), g))
        t += gap
        gap *= ev.get('faster', 1.0)
    return out


def render(recipe, seed=0, lufs=None, peak_db=-1.0, dur=None):
    """the sound of a recipe: (y, meta). y: stereo float samples at 48 kHz, levelled like a catalogue sound of its
    kind (the true peak at peak_db under the kind's loudest level, or `lufs`). seed: another take of every layer
    and new noise; the same seed gives the same samples. dur: a length to cut the file at instead of the recipe's.
    meta has the fields of any take of `earshot.make` (id "recipe/<name>", kind, seed, dur_s, peak_dbfs, lufs_max,
    checks: the audit every sound gets, peak_s, onset_s) and source "recipe", `parts` (the designs and sources it
    is built from) and `notes` (a control a design could not follow, a length that could not be kept)"""
    from types import SimpleNamespace
    import numpy as np
    from .forge import dsp
    from .make import marks
    from .sounds import registry
    from .synth import design as D, master, space
    rc = check(recipe)
    seed, notes, rows = int(seed), [], []
    for i, lay in enumerate(rc['layers']):
        rows += _placed(lay, seed, i, notes)
    n = max(dsp.n_of(t) + len(y) for y, t, _ in rows)
    mix = np.zeros((n, 2))
    for y, t, g in rows:
        a = dsp.n_of(t) if t > 0 else 0
        mix[a:a + len(y)] += y * g
    bed = rc.get('kind', 'event') == 'ambience'
    if rc.get('space', 'dry') != 'dry':
        mix = space.room(mix, rc['space'], r=dsp.rng(31337 + seed))
    want = dur if dur is not None else rc.get('dur')
    if want and len(mix) > dsp.n_of(want):
        mix = D.fade(mix[:dsp.n_of(want)], 0.0, min(0.5, max(0.05, 0.25 * want)), shape='db')
    elif want and len(mix) < dsp.n_of(want) - dsp.n_of(0.1):
        notes.append(f'{len(mix) / dsp.SR:.2f} s, not the {want:g} s asked: the layers end there; make one longer '
                     '(dur, events) or move it later (at)')
    if bed:
        mix = D.fade(D.fade(mix, 0.3, 0.0), 0.0, 0.6, shape='db')
    kind = 'bed' if bed else 'object'
    try:
        y = master.finish(mix, peak_db=peak_db, lufs=lufs, cap=registry.LOUDEST.get(kind, registry.LOUDEST['']),
                          lift=0.0 if bed else registry.LIFT, sparse=registry.SPARSE if bed else 0.0)
    except ValueError as e:
        raise RecipeError(f'the recipe {rc["name"]} made no sound: {e}') from None
    # several events are the design of a recipe, with silence between them in a dry space; a reversed layer ends
    # on its strike: that stop is the design, no cut tail; and what the audit allows a part's design (a campfire's
    # crackles, the ring of a part that has a note) it allows the sound built from it
    used = [registry.get(lay['sound']) for lay in rc['layers'] if 'sound' in lay]
    allows = (('stacked', 'silence', 'tail') if len(rows) > 1 else ()) + (
        ('truncated',) if any(lay.get('reverse') for lay in rc['layers']) else ()) + tuple(
        a for d in used for a in d.allows) + (('ring',) if any(d.pitch for d in used) else ())
    checks = registry.audit(y, SimpleNamespace(kind=kind, pitch=None, loop=False, allows=allows))
    meta = {'id': f'recipe/{rc["name"]}', 'kind': kind, 'seed': seed, 'dur_s': round(len(y) / dsp.SR, 3),
            'peak_dbfs': round(master.true_peak_db(y), 2), 'lufs_max': round(master.loudness(y), 1),
            'checks': checks, 'source': 'recipe', 'parts': parts(rc)}
    if rc.get('says'):
        meta['says'] = rc['says']
    if notes:
        meta['notes'] = notes
    meta.update(marks(y))
    return y, meta


def parts(recipe):
    """what a recipe is built from, a few words a layer: the design or source and what is done to it"""
    out = []
    for lay in recipe['layers']:
        bits = []
        if lay.get('pitch'):
            bits.append(f'{lay["pitch"]:+g} semitones')
        if lay.get('reverse'):
            bits.append('reversed')
        if lay.get('events'):
            bits.append(f'{lay["events"]["count"]} times')
        for k, word in (('lowpass', 'under'), ('highpass', 'over')):
            if k in lay:
                v = lay[k]
                bits.append(f'{word} {v[0]:g} to {v[1]:g} Hz' if isinstance(v, list) else f'{word} {v:g} Hz')
        what = lay.get('sound') or (f'{lay.get("color", "pink")} noise' if lay['source'] == 'noise'
                                    else f'a {lay["hz"]:g} Hz sine')
        out.append(what + (f' ({", ".join(bits)})' if bits else ''))
    return out


# ---------------------------------------------------------------------------------------------------- command line
HOW = """A recipe builds a sound the catalogue lacks from its own parts. Say what physically makes the sound, find
two to four real parts (earshot sounds --search <word>), write the layers, render, read the result, change one thing.
  layer:   "sound": a design id   or   "source": "noise" | "sine" (with "dur"; a sine with "hz" 20..150)
  on it:   seed, dur, pitch (semitones, -24..12: lower is slower and bigger), reverse, cut [from, to], halt (s: it
           runs down to a stop), lowpass / highpass (Hz, or [from, to] to move), pan, gain (dB), at (s), fade_in,
           fade_out, swell, decay, size / bright / hard / energy / ring / motion / width (-1..1, where the design
           has them), events {"count", "gap", "faster", "softer", "loose"}
  whole:   name, says, layers, dur, space (dry, booth, room, studio, hall, plate, cathedral), kind (event, ambience)
  render:  earshot make --recipe file.json -o out.wav      (or: earshot make --recipe - -o out.wav < file.json)"""


def listing():
    """the cookbook as lines: each recipe's name, what it is and what it is built from"""
    rows = [f'{name:<24} {r.get("says", "")}{" (a sketch)" if r.get("sketch") else ""}: {", ".join(parts(r))}'
            for name, r in cookbook().items()]
    return {'summary': '\n'.join([f'{len(rows)} recipes in the cookbook'] + rows), 'recipes': list(cookbook())}


def examples():
    """the form in a few lines and three cookbook recipes to start from"""
    book = cookbook()
    picked = [book[n] for n in EXAMPLES if n in book] or list(book.values())[:3]
    text = '\n\n'.join([HOW] + [json.dumps(r, indent=1) for r in picked])
    return {'summary': text, 'recipes': picked}


def add_cli(sub):
    p = sub.add_parser('recipe', help='recipes: sounds the catalogue lacks, built from its parts (the form, examples, '
                                      'the cookbook)')
    p.add_argument('name', nargs='?', help='a cookbook recipe to print (earshot recipe --list)')
    p.add_argument('--example', action='store_true', help='the form in a few lines and three recipes to start from')
    p.add_argument('--list', action='store_true', help='the cookbook: every recipe that ships with earshot')
    p.add_argument('--check', metavar='FILE', help='check a recipe (a JSON file, or - for standard input) without '
                                                   'rendering it')
    from .cli import common
    common(p)
    p.set_defaults(func=_run)


def _run(args):
    from .report import emit
    if args.check:
        rc = load(args.check)
        res = {'summary': f'{rc["name"]}: a recipe of {len(rc["layers"])} layers: {", ".join(parts(rc))}',
               'recipe': rc}
    elif args.name:
        rc = load(args.name)
        res = {'summary': json.dumps(rc, indent=1), 'recipe': rc}
    elif args.list:
        res = listing()
    else:
        res = examples()
    emit(res, args, text=res['summary'])
