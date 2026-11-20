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
(`earshot sounds --search`; the designs learned from recordings are the best material, they are real), a cookbook
recipe by its name, or one of two plain sources for what no recording gives: filtered noise (air, rumble; its cutoff
may move) and a low sine (weight under a hit). Recordings are the material and the recipe gives them a structure:
one clap scattered becomes applause, a drop on tin a rain on a tin roof, a running motor that passes a car going by,
a short creak stretched a long one, a moo bent a whale's song. Each layer has its controls:

    sound | source | recipe
                     the design's id, "noise" / "sine", or a cookbook recipe's name (one level deep: a recipe
                     layer holds no recipe layer)
    seed             another take of the design (a learned design: its next recording)
    size, bright, hard, energy, ring, motion, width
                     the design's own axes, -1 .. 1, where it has them
    dur              the length asked of the design itself (an ambience any length, a whoosh, a recording within
                     0.6 to 1.8 times its own); of a source: its length
    pitch            semitones, -24 .. 12, as a tape does it: lower is also slower and bigger
    glide            semitones, -24 .. 12: where the pitch has moved to at the end of the take, from `pitch` (or 0)
                     at its start, straight in semitones, as a tape that slows or speeds up (a power-down, a sinking)
    reverse          played backwards: a decay becomes a swell that ends on its strike
    cut              [from, to] in s: the part of the take that is used
    stretch          a factor, 0.5 .. 2: the take longer or shorter at the same pitch; the first 30 ms after its
                     attack stay as they are. Windows of the take laid again: aligned by waveform for a take with a
                     pitch, read from places moved at random for a noisy one. Further than that real takes (creaks, a
                     moo, a snore) ring like a comb: for longer, scatter the take or repeat it
    halt             s: the take runs down to a stop over that long, as a motor does when its power is cut
    lowpass, highpass
                     Hz; two numbers [from, to] move the cutoff over the layer
    pan              -1 (left) .. 1 (right); [from, to]: the layer moves across
    gain             dB against the other layers: every layer is first brought to the same peak
    at               s: where the layer starts
    fade_in, fade_out, swell, decay
                     s: short fades at its ends (an ambience gets 0.1 and 0.3 s, a source 0.01 and 0.02 s when
                     not said); a slow rise from 30 dB down; a slow fall to nothing
    events           {"count", "gap", "faster", "softer", "loose"}: the layer several times, each a new take: `gap`
                     s from one to the next, each gap `faster` times the one before (under 1: faster and faster),
                     each event `softer` dB against the one before, each up to `loose` (0 .. 1) of half its gap
                     early or late (drops, crackles, clicks that come as they come)
    scatter          {"rate", "rate_to", "dur", "pitch", "gain", "pan", "grain"}: the layer many times at random
                     moments (a Poisson process): `rate` events a second (0.5 .. 400) moving to `rate_to` by the end,
                     over `dur` s; each event up to `pitch` semitones (0 .. 12) higher or lower, up to `gain` dB
                     (0 .. 40) softer and up to `pan` (0 .. 1) to either side; with `grain` (0.005 .. 2 s) each is a
                     random piece of that length of the take, faded over 5 ms (sizzle, gravel, a crowd's murmur).
                     Eight takes (three for pieces) are rendered once (a learned design: its recordings in turn)
                     and placed as copies, so thousands of events stay fast; takes are kept for the next render.
                     Rain, hail, applause, a swarm, coins pouring. The take's controls shape every event; pan,
                     fades, swell and decay shape the scattered layer as a whole
    level            [[time s, dB], ...]: the layer's loudness over time, straight in dB between the points
                     (-60 .. 12), smoothed over 5 ms
    bend             [[time s, semitones], ...]: the layer's pitch over the heard time as a tape does it (-24 .. 12):
                     lower is slower; glide is its two-point case over a take
    pass             {"at", "speed"}: the layer passes the listener, closest (8 m) at `at` s, at `speed` m/s
                     (1 .. 100): read with the delay of its distance, so its pitch falls through that moment as the
                     Doppler effect makes it; as loud as the distance allows, duller far away, crossing from one side
                     to the other (a car, a train, a jet going by)
    spin             {"rate", "depth", "shape"}: the loudness pulses `rate` times a second (0.5 .. 60, or [from,
                     to]) by `depth` (0 .. 1, default 0.8), "smooth" (a cosine) or "chop" (a gate with soft edges):
                     rotor blades, wings, a fan, a motor's chug, purring
                     The times of level, bend and pass count from the layer's own start (its `at`); they are done to
                     the layer as a whole, once its events are laid, in this order: bend, pass, level, spin, a moving
                     pan
    follow          {"layer", "amount", "smooth"}: the layer's loudness follows another layer's (its index, from 0):
                     its level over `smooth` s (default 0.05) as a share of its loudest moment; amount 1 (default)
                     makes the layer exactly that loud, 0.5 half of the way; under 0 it gives way where the other is
                     loud (-1: silent at the other's loudest moment): twigs that snap with the gusts, a rattle that
                     comes with a motor, a bed that ducks under a hit
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

One sound as the body of another ("X on Y": the hits of X ringing through the resonance of Y) is not in the form.
Built from a struck design's partials and colour, rain through a wooden knock comes no nearer a real recording of rain
on wood for the listening model than the plain rain does, and a metal body moves it no nearer rain on tin: what is
heard of rain on a thing is the colour of the whole, and a knock's colour is not it. Where the catalogue has the real
surface (rain on wood, on a tin roof, on a window), that design is the answer.
"""
import difflib
import json
import os
import sys

from . import brief, config

LAYERS = 12                # layers of one recipe
PEAK = 0.5                 # every layer's peak before its gain: gains say how loud it sits against the others
SOURCES = ('noise', 'sine')
COLORS = ('white', 'pink', 'brown')
KINDS = ('event', 'ambience')
AXES = ('size', 'bright', 'hard', 'energy', 'ring', 'motion', 'width')
TOP = ('name', 'says', 'layers', 'dur', 'space', 'kind', 'things', 'built', 'sketch')
LAYER = ('sound', 'source', 'recipe', 'seed', 'dur', 'pitch', 'reverse', 'cut', 'stretch', 'halt', 'lowpass',
         'highpass', 'pan', 'gain', 'at', 'fade_in', 'fade_out', 'swell', 'decay', 'events', 'scatter', 'follow',
         'glide', 'bend', 'level', 'pass', 'spin', 'color', 'hz', 'to') + AXES
EVENTS = ('count', 'gap', 'faster', 'softer', 'loose')
SCATTER = ('rate', 'rate_to', 'dur', 'pitch', 'gain', 'pan', 'grain')
# a scatter field -> (lowest, highest, what it is)
SCATTER_RANGE = {'rate': (0.5, 400.0, 'a number of events a second'),
                 'rate_to': (0.5, 400.0, 'a number of events a second'), 'dur': (0.05, 120.0, 'a length in seconds'),
                 'pitch': (0.0, 12.0, 'a number of semitones'), 'gain': (0.0, 40.0, 'a level in dB'),
                 'pan': (0.0, 1.0, 'an amount'), 'grain': (0.005, 2.0, 'a length in seconds')}
TAKES = 8                  # takes of a scattered layer's design, rendered once and placed many times
GRAIN_TAKES = 3            # of one scattered in pieces: the pieces come from everywhere in a take
VARIANTS = 5               # pitches of each take when the events of a scatter differ in pitch
PASS = ('at', 'speed')
PASS_RANGE = {'at': (0.0, 120.0, 'a time in seconds'), 'speed': (1.0, 100.0, 'a speed in metres a second')}
NEAR = 8.0                 # m: how close a passing layer comes
SOUND_SPEED = 343.0        # m/s
SPIN = ('rate', 'depth', 'shape')
SPIN_RANGE = {'rate': (0.5, 60.0, 'a rate in Hz'), 'depth': (0.0, 1.0, 'an amount')}
SHAPES = ('smooth', 'chop')
STRETCH = (0.5, 2.0)       # a stretch factor: where real takes stay free of a metallic colour (creaks, a moo and a
                           # snore stretched 3 times ring like a comb; a creak halved does too)
# a list of [time, value] points -> (lowest value, highest value, what the value is)
CURVES = {'level': (-60.0, 12.0, 'a level in dB'), 'bend': (-24.0, 12.0, 'a number of semitones')}
# a number field -> (lowest, highest, what it is)
RANGE = {'dur': (0.02, 120.0, 'a length in seconds'), 'pitch': (-24.0, 12.0, 'a number of semitones'),
         'glide': (-24.0, 12.0, 'a number of semitones'), 'stretch': (*STRETCH, 'a factor'),
         'gain': (-40.0, 12.0, 'a level in dB'),
         'at': (0.0, 120.0, 'a start time in seconds'), 'fade_in': (0.0, 30.0, 'a time in seconds'),
         'fade_out': (0.0, 30.0, 'a time in seconds'), 'swell': (0.0, 60.0, 'a time in seconds'),
         'decay': (0.0, 60.0, 'a time in seconds'), 'halt': (0.05, 10.0, 'a time in seconds'),
         'hz': (20.0, 150.0, 'a frequency in Hz'),
         'to': (20.0, 150.0, 'a frequency in Hz'), **{a: (-1.0, 1.0, 'an amount') for a in AXES}}
CUTOFF = (20.0, 20000.0)
EVENT_RANGE = {'count': (2, 64, 'a whole number'), 'gap': (0.02, 30.0, 'a time in seconds'),
               'faster': (0.5, 2.0, 'a factor'), 'softer': (-12.0, 12.0, 'a level in dB'),
               'loose': (0.0, 1.0, 'an amount')}
EXAMPLES = ('dragon_roar', 'marbles_pouring', 'jet_flyover')     # layers and pitch; scatter; pass
FOLDER = os.path.join(config.DATA_DIR, 'recipes')
FOLLOW = ('layer', 'amount', 'smooth')
# a follow field -> (lowest, highest, what it is, when not said)
FOLLOW_RANGE = {'amount': (-1.0, 1.0, 'an amount', 1.0), 'smooth': (0.005, 2.0, 'a time in seconds', 0.05)}


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
    if sum(k in lay for k in ('sound', 'source', 'recipe')) != 1:
        raise RecipeError(f'{w} names either a catalogue design ("sound": "door/slam"), a plain source '
                          f'("source": "noise" or "sine") or a cookbook recipe ("recipe": "thunder_far"), one of them')
    out = dict(lay)
    if 'recipe' in lay:
        name, book = lay['recipe'], cookbook()
        if not isinstance(name, str) or name not in book:
            near = difflib.get_close_matches(str(name), list(book), n=3, cutoff=0.5)
            raise RecipeError(f'{w}.recipe {name!r} is not in the cookbook' + (
                f'; the nearest: {", ".join(near)}' if near else '') + '; `earshot recipe --list` lists them')
        if any('recipe' in sub for sub in book[name]['layers']):
            raise RecipeError(f'{w}.recipe {name!r} holds a recipe itself: a recipe layer goes one level deep')
        for k in AXES + ('color', 'hz', 'to'):
            if k in lay:
                raise RecipeError(f'{w}.{k} does not belong to a recipe layer: it takes dur and the controls every '
                                  'layer has')
    elif 'sound' in lay:
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
            try:
                out[k] = _number(lay[k], f'{w}.{k}', lo, hi, what)
            except RecipeError as e:
                if k != 'stretch':
                    raise
                raise RecipeError(f'{e}: further than that a real take rings like a comb (for longer, scatter '
                                  'or repeat it)') from None
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
    if 'follow' in lay:
        out['follow'] = _follow_field(lay['follow'], w, i)
    if 'pan' in lay:
        p = lay['pan']
        if isinstance(p, (list, tuple)):
            if len(p) != 2:
                raise RecipeError(f'{w}.pan is a position from -1 to 1 or two of them [from, to], not {list(p)!r}')
            out['pan'] = [_number(x, f'{w}.pan[{j}]', -1.0, 1.0, 'a position') for j, x in enumerate(p)]
        else:
            out['pan'] = _number(p, f'{w}.pan', -1.0, 1.0, 'a position')
    if 'scatter' in lay:
        out['scatter'] = _scatter_field(lay['scatter'], w)
        if 'events' in lay:
            raise RecipeError(f'{w} takes events (a row of up to 64) or scatter (many at random moments), not both')
    for k, (lo, hi, what) in CURVES.items():
        if k in lay:
            out[k] = _curve(lay[k], f'{w}.{k}', lo, hi, what)
    if 'pass' in lay:
        pa = lay['pass']
        if not isinstance(pa, dict):
            raise RecipeError(f'{w}.pass is an object such as {{"at": 2.0, "speed": 25}}, not {pa!r}')
        _fields(pa, PASS, f'{w}.pass', 'pass')
        if 'at' not in pa:
            raise RecipeError(f'{w}.pass.at is missing: pass takes at (s, the closest moment) and speed (m/s)')
        out['pass'] = {k: _number(pa.get(k, 20.0), f'{w}.pass.{k}', *PASS_RANGE[k]) for k in PASS}
    if 'spin' in lay:
        out['spin'] = _spin_field(lay['spin'], w)
    return out


def _scatter_field(sc, w):
    """the `scatter` of a layer checked: {"rate", "rate_to", "dur", "pitch", "gain", "pan", "grain"}"""
    if not isinstance(sc, dict):
        raise RecipeError(f'{w}.scatter is an object such as {{"rate": 40, "dur": 6, "pitch": 3}}, not {sc!r}')
    _fields(sc, SCATTER, f'{w}.scatter', 'scatter')
    for k in ('rate', 'dur'):
        if k not in sc:
            raise RecipeError(f'{w}.scatter.{k} is missing: scatter takes rate (events a second, 0.5 to 400) and '
                              'dur (seconds)')
    return {k: _number(sc[k], f'{w}.scatter.{k}', *SCATTER_RANGE[k]) for k in sc}


def _curve(pts, where, lo, hi, what):
    """a list of [time s, value] points checked: two or more, times from 0 to 120 s and in order"""
    ok = isinstance(pts, (list, tuple)) and len(pts) >= 2 and all(
        isinstance(p, (list, tuple)) and len(p) == 2 for p in pts)
    if not ok:
        raise RecipeError(f'{where} is a list of two or more points [time s, {what.split()[-1]}] such as '
                          f'[[0, 0], [2, -6]], not {pts!r}')
    out = []
    for j, (t, v) in enumerate(pts):
        t = _number(t, f'{where}[{j}][0]', 0.0, 120.0, 'a time in seconds')
        if out and t < out[-1][0]:
            raise RecipeError(f'{where}[{j}][0] is {t:g} s, before the point ahead of it: the times go in order')
        out.append([t, _number(v, f'{where}[{j}][1]', lo, hi, what)])
    return out


def _spin_field(sp, w):
    """the `spin` of a layer checked: {"rate": Hz or [from, to], "depth": 0 .. 1, "shape": "smooth" or "chop"}"""
    if not isinstance(sp, dict):
        raise RecipeError(f'{w}.spin is an object such as {{"rate": 12, "depth": 0.8}}, not {sp!r}')
    _fields(sp, SPIN, f'{w}.spin', 'spin')
    if 'rate' not in sp:
        raise RecipeError(f'{w}.spin.rate is missing: spin takes rate (Hz, 0.5 to 60, or [from, to]), depth and '
                          'shape')
    r = sp['rate']
    if isinstance(r, (list, tuple)):
        if len(r) != 2:
            raise RecipeError(f'{w}.spin.rate is a rate in Hz or two of them [from, to], not {list(r)!r}')
        rate = [_number(x, f'{w}.spin.rate[{j}]', *SPIN_RANGE['rate']) for j, x in enumerate(r)]
    else:
        rate = _number(r, f'{w}.spin.rate', *SPIN_RANGE['rate'])
    shape = sp.get('shape', 'smooth')
    if shape not in SHAPES:
        raise RecipeError(f'{w}.spin.shape is one of {", ".join(SHAPES)}, not {shape!r}')
    return {'rate': rate, 'depth': _number(sp.get('depth', 0.8), f'{w}.spin.depth', *SPIN_RANGE['depth']),
            'shape': shape}


def _follow_field(fo, w, i):
    """the `follow` of layer i checked: {"layer": another layer's index, "amount": -1 .. 1, "smooth": s}"""
    if isinstance(fo, int) and not isinstance(fo, bool):
        fo = {'layer': fo}
    if not isinstance(fo, dict):
        raise RecipeError(f'{w}.follow is an object such as {{"layer": 0, "amount": 0.8}}, not {fo!r}')
    _fields(fo, FOLLOW, f'{w}.follow', 'follow')
    if 'layer' not in fo:
        raise RecipeError(f'{w}.follow.layer is missing: follow names the layer (its index, from 0) whose loudness '
                          'this layer follows')
    k = _number(fo['layer'], f'{w}.follow.layer', 0, LAYERS - 1, 'the index of another layer', whole=True)
    if k == i:
        raise RecipeError(f'{w}.follow.layer is {k}, the layer itself: a layer follows another layer')
    out = {'layer': k}
    for key, (lo, hi, what, default) in FOLLOW_RANGE.items():
        out[key] = _number(fo.get(key, default), f'{w}.follow.{key}', lo, hi, what)
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
    for i, lay in enumerate(out['layers']):
        if 'follow' in lay and lay['follow']['layer'] >= len(layers):
            raise RecipeError(f'layers[{i}].follow.layer is {lay["follow"]["layer"]}, but the recipe has '
                              f'{len(layers)} layers (0 to {len(layers) - 1})')
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


def _glide(y, start, end):
    """y played as a tape whose speed moves from `start` to `end` semitones over the take, straight in semitones
    along the take: the place read at time t has a closed form, read between samples (a low-pass at the higher of
    the two speeds' band edges first keeps what is moved up from folding back)"""
    import numpy as np
    from .forge import dsp
    c, n = np.log(2.0) / 12.0, len(y)
    d = end - start
    if max(start, end) > 0:
        y = dsp.butter(y, 'low', 0.45 * dsp.SR * 2.0 ** (-max(start, end) / 12.0), order=4)
    # read position p(t) solves dp/dt = 2 ** ((start + d * p / n) / 12)
    if abs(d) < 1e-9:
        t_end = n * np.exp(-c * start)
    else:
        t_end = n * (1.0 - np.exp(-c * d)) / (c * d * np.exp(c * start))
    t = np.arange(int(t_end))
    if abs(d) < 1e-9:
        p = np.exp(c * start) * t
    else:
        p = -(n / (c * d)) * np.log(np.maximum(1.0 - c * d * np.exp(c * start) * t / n, 1e-12))
    p = np.clip(p, 0, n - 1)
    return np.stack([np.interp(p, np.arange(n), y[:, ch]) for ch in range(y.shape[1])], axis=1)


_takes = {}
KEEP = 96                  # takes of designs kept in memory: a recipe rendered again (another seed, a number tuned)
                           # renders only what it has not rendered before


def _take(sid, seed, req):
    """a design's take as registry.render gives it (unchecked), kept for the next render of the same take"""
    from .sounds import registry
    key = (sid, int(seed), tuple(sorted(req.items())))
    if key not in _takes:
        if len(_takes) >= KEEP:
            _takes.pop(next(iter(_takes)))
        _takes[key] = registry.render(sid, seed=seed, check=False, **req)
    y, meta = _takes[key]
    return y, dict(meta)


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
        y, meta = _take(lay['sound'], seed + lay.get('seed', 0), req)
        ign = [k for k in meta.get('ignored', ()) if k in req]
        if ign:
            note = (f'layers[{i}]: {lay["sound"]} has no control of its {", ".join(ign)} and plays as designed'
                    + (f' (its axes: {", ".join(d.axes)})' if d.axes else ''))
            if note not in notes:
                notes.append(note)
        y = _stereo(y)
    elif 'recipe' in lay:
        y = _mix(check(cookbook()[lay['recipe']]), seed + lay.get('seed', 0), [], want=lay.get('dur'))[0]
    else:
        y = _source(lay, dsp.rng(7919 * (i + 1) + 104729 * seed + lay.get('seed', 0)))
    if 'glide' in lay:
        y = _glide(y, lay.get('pitch', 0.0), lay['glide'])
    elif lay.get('pitch'):
        ratio = Fraction(2.0 ** (-lay['pitch'] / 12.0)).limit_denominator(96)
        y = signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=0)
    if lay.get('reverse'):
        y = D.reverse(y)
    if 'cut' in lay:
        y = D.span(y, lay['cut'][0], lay['cut'][1], fin=0.004, fout=0.02)
        if len(y) < dsp.n_of(0.01):
            raise RecipeError(f'layers[{i}].cut starts at {lay["cut"][0]:g} s, after the end of the take: it is '
                              'shorter than that')
    if lay.get('stretch', 1.0) != 1.0:
        y = _stretch(y, lay['stretch'], dsp.rng(48611 * (i + 1) + seed))
    if lay.get('halt'):             # the motor runs down and grows quiet with it: the stop is no cut
        y = D.fade(D.halt(y, lay['halt']), 0.0, lay['halt'], shape='db')
    y = _filters(y, lay)
    if isinstance(lay.get('pan'), float):
        th = (lay['pan'] + 1.0) * np.pi / 4
        y = y * np.array([np.cos(th), np.sin(th)]) * np.sqrt(2)
    peak = float(np.abs(y).max())
    if peak <= 1e-9:
        raise RecipeError(f'layers[{i}] is silent after its cut and filters: nothing of it is left to hear')
    y = y * (PEAK / peak)
    # what has no start and end of its own gets them: a source 10 and 20 ms, an ambience (a looping design) 0.1
    # and 0.3 s, unless the layer says its fades. A fade out falls straight in dB, as a decay does: the audit
    # reads a fade that drops faster at its end than before it as a cut tail
    fin, fout = (0.01, 0.02) if 'source' in lay else (0.1, 0.3) if 'sound' in lay and registry.get(
        lay['sound']).loop else (0.0, 0.0)
    y = D.fade(D.fade(y, lay.get('fade_in', fin), 0.0), 0.0, lay.get('fade_out', fout), shape='db')
    if lay.get('swell'):
        y = _slope(y, lay['swell'], -30.0, True)
    if lay.get('decay'):
        y = _slope(y, lay['decay'], -50.0, False)
    return y


def _placed(lay, seed, i, notes):
    """the events of a layer as (samples, start s, gain as a factor)"""
    from .forge import dsp
    if 'scatter' in lay:
        return [(_scatter(lay, seed, i, notes), lay.get('at', 0.0), 10 ** (lay.get('gain', 0.0) / 20))]
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


AS_A_WHOLE = ('pan', 'fade_in', 'fade_out', 'swell', 'decay', 'scatter', 'gain', 'at')


def _active(y, peak):
    """(first, last) sample where y is within 30 dB of its peak"""
    import numpy as np
    on = np.flatnonzero(np.abs(y).max(axis=1) > 0.03 * peak)
    return (int(on[0]), int(on[-1])) if len(on) else (0, len(y) - 1)


def _scatter(lay, seed, i, notes):
    """the layer's design many times at random moments (scatter): a few takes rendered once, each at a few pitches,
    and copies of them (or of random pieces of them, `grain`) laid over each other where a Poisson process of the
    rate puts them, each softer and further to a side by a random amount; brought to PEAK as one layer"""
    from fractions import Fraction
    import numpy as np
    from scipy import signal
    from .forge import dsp
    sc = lay['scatter']
    base = {k: v for k, v in lay.items() if k not in AS_A_WHOLE}
    r = dsp.rng(7907 * (i + 1) + 15485863 + seed)
    pool = []
    shifts = np.linspace(-sc['pitch'], sc['pitch'], VARIANTS) if sc.get('pitch') else [0.0]
    for k in range(GRAIN_TAKES if sc.get('grain') else TAKES):
        y = _one(base, seed + k, i, notes)
        for s in shifts:
            if s:
                ratio = Fraction(2.0 ** (-s / 12.0)).limit_denominator(96)
                pool.append(signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=0))
            else:
                pool.append(y)
    # event times: a Poisson process whose rate moves from `rate` to `rate_to` on a geometric line over `dur`
    a, b_, dur = sc['rate'], sc.get('rate_to', sc['rate']), sc['dur']
    need = int(1.5 * dur * max(a, b_) + 20)
    s = np.cumsum(r.exponential(1.0, need))
    if abs(b_ - a) < 1e-9:
        t = s / a
    else:
        g = np.log(b_ / a) / dur
        t = np.log(np.maximum(1.0 + g * s / a, 1e-12)) / g
    t = t[np.isfinite(t) & (t < dur)]
    m = len(t)
    pick = r.integers(0, len(pool), m)
    gains = 10 ** (-r.uniform(0.0, sc.get('gain', 0.0), m) / 20) if sc.get('gain') else np.ones(m)
    pans = np.clip(r.uniform(-1.0, 1.0, m) * sc.get('pan', 0.0) + (lay['pan'] if isinstance(lay.get('pan'), float)
                                                                    else 0.0), -1.0, 1.0)
    th = (pans + 1.0) * np.pi / 4
    gl, gr = gains * np.cos(th) * np.sqrt(2), gains * np.sin(th) * np.sqrt(2)
    starts = (t * dsp.SR).astype(np.int64)
    if sc.get('grain'):
        size = max(dsp.n_of(sc['grain']), 16)
        f = min(dsp.n_of(0.005), size // 4)
        win = np.ones(size)
        win[:f] = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, f)) ** 2
        win[size - f:] = win[:f][::-1]
        spans = [_active(p, PEAK) for p in pool]
        u = r.uniform(0.0, 1.0, m)
        out = np.zeros((dsp.n_of(dur) + size + 1, 2))
        for e in range(m):
            p = pool[pick[e]]
            if len(p) <= size:
                piece = p * win[:len(p), None]
            else:
                lo, hi = spans[pick[e]]
                hi = max(lo, min(hi, len(p) - size))
                o = lo + int(u[e] * (hi - lo))
                piece = p[o:o + size] * win[:, None]
            d = starts[e]
            out[d:d + len(piece), 0] += piece[:, 0] * gl[e]
            out[d:d + len(piece), 1] += piece[:, 1] * gr[e]
    else:
        out = np.zeros((dsp.n_of(dur) + max(len(p) for p in pool) + 1, 2))
        for e in range(m):
            p, d = pool[pick[e]], starts[e]
            out[d:d + len(p), 0] += p[:, 0] * gl[e]
            out[d:d + len(p), 1] += p[:, 1] * gr[e]
    peak = float(np.abs(out).max())
    if peak <= 1e-9:
        raise RecipeError(f'layers[{i}].scatter placed nothing to hear: give it a higher rate or a longer dur')
    return out * (PEAK / peak)


def _stretch(y, factor, r):
    """y made `factor` times as long without moving its pitch. The first 30 ms after its attack are kept as they are
    (a hit is not smeared); the rest is laid again from windows of 50 ms at half their length apart, read where the
    stretch puts them: for a take with a pitch (a creak, a growl, a voice) each window is moved up to 12 ms to where
    it continues the one before best (waveform similarity), for a noisy take (a hiss, rain, wind) each is read from
    a place moved at random, so no repeat is heard as a buzz"""
    import numpy as np
    from scipy import signal
    from .forge import dsp
    n, size = len(y), dsp.n_of(0.05)
    hop = size // 2
    peak = float(np.abs(y).max())
    if n < 3 * size or peak <= 0:
        return y
    on = int(np.argmax(np.abs(y).max(axis=1) > 0.1 * peak))
    h = min(on + dsp.n_of(0.03) + hop, n - 2 * size)
    x = y[h:]
    mono = x.mean(axis=1)
    f, _, z = signal.stft(mono[:min(len(mono), dsp.n_of(2.0))], nperseg=1024)
    mag = np.abs(z) + 1e-12
    flat = float(np.median(np.exp(np.log(mag).mean(axis=0)) / mag.mean(axis=0)))
    noisy = flat > 0.3
    win = np.sin(np.pi * (np.arange(size) + 0.5) / size)          # power-complementary: for unrelated windows
    if not noisy:
        win = win ** 2                                             # amplitude-complementary: for aligned ones
    step = hop / factor
    want = int(len(x) * factor)
    frames = int(np.ceil(max(want - size, 0) / hop)) + 1
    out = np.zeros((frames * hop + size, 2))
    tol, dec = dsp.n_of(0.012), 4
    m4 = mono[::dec]
    prev = 0
    for k in range(frames):
        nominal = min(int(k * step), len(x) - size)
        if k == 0:
            pos = 0
        elif noisy:
            pos = int(np.clip(nominal + r.integers(-size, size + 1), 0, len(x) - size))
        else:
            lo, hi = max(0, nominal - tol), min(len(x) - size, nominal + tol)
            target = prev + hop
            if hi <= lo or target + size > len(x):
                pos = int(np.clip(nominal, 0, len(x) - size))
            else:
                tpl = m4[target // dec:(target + size) // dec]
                seg = m4[lo // dec:(hi + size) // dec]
                c = np.correlate(seg, tpl, 'valid')
                e = np.sqrt(np.maximum(np.convolve(seg ** 2, np.ones(len(tpl)), 'valid'), 1e-12))
                pos = int(np.clip(lo + dec * int(np.argmax(c / e[:len(c)])), 0, len(x) - size))
        out[k * hop:k * hop + size] += x[pos:pos + size] * win[:, None]
        prev = pos
    out = out[:want]
    # the kept head runs into the stretched rest over half a window
    fall = np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, hop))[:, None] ** (2 if not noisy else 1)
    head = np.array(y[:h])
    tail = np.array(out)
    tail[:hop] += y[h - hop:h] * fall
    return np.concatenate([head[:h - hop], tail])


def _curve_at(pts, n):
    """a [[time s, value], ...] curve read at every sample of n, straight between the points, held at its ends"""
    import numpy as np
    from .forge import dsp
    ts, vs = zip(*pts)
    return np.interp(np.arange(n) / dsp.SR, ts, vs)


def _read(y, p):
    """y read at the (fractional, rising) sample positions p"""
    import numpy as np
    idx = np.arange(len(y))
    return np.stack([np.interp(p, idx, y[:, ch]) for ch in range(y.shape[1])], axis=1)


def _bend(y, pts):
    """y as a tape whose speed follows the curve of semitones over the heard time"""
    import numpy as np
    from .forge import dsp
    top, low = max(v for _, v in pts), min(v for _, v in pts)
    if top > 0:
        y = dsp.butter(y, 'low', 0.45 * dsp.SR * 2.0 ** (-top / 12.0), order=4)
    m = int(len(y) / 2.0 ** (low / 12.0)) + 2
    speed = 2.0 ** (_curve_at(pts, m) / 12.0)
    p = np.concatenate([[0.0], np.cumsum(speed[:-1])])
    p = p[p <= len(y) - 1]
    return _read(y, p)


def _pass(y, pa):
    """y as a thing that passes the listener at `speed` m/s, closest (NEAR m) at `at` s of the heard time: read with
    the delay of its distance (so its pitch falls through that moment as the Doppler effect does), as loud as its
    distance allows, duller far away, and crossing from one side to the other"""
    import numpy as np
    from .forge import dsp
    v, at = pa['speed'], pa['at']
    y = dsp.butter(y, 'low', 0.45 * dsp.SR * (1.0 - v / SOUND_SPEED), order=4)
    m = int(len(y) / (1.0 - v / SOUND_SPEED)) + 2
    t = np.arange(m) / dsp.SR
    x = v * (t - at)
    dist = np.sqrt(x ** 2 + NEAR ** 2)
    p = (t + (dist[0] - dist) / SOUND_SPEED) * dsp.SR
    keep = p <= len(y) - 1
    p, x, dist = p[keep], x[keep], dist[keep]
    z = _read(y, p) * (NEAR / dist)[:, None]
    fc = np.clip(18000.0 * NEAR / dist, 700.0, 18000.0)
    z = np.stack([dsp.svf(z[:, ch], fc, 0.7071, 'low') for ch in range(2)], axis=1)
    return _panned(z, 0.9 * x / dist)


def _panned(y, pan):
    """y with a position (-1 left .. 1 right) for every sample, at equal power"""
    import numpy as np
    th = (np.asarray(pan) + 1.0) * np.pi / 4
    return y * np.stack([np.cos(th), np.sin(th)], axis=-1) * np.sqrt(2)


def _spin(y, sp):
    """y pulsing in loudness `rate` times a second (from, to on a geometric line), by `depth`: a cosine (smooth) or
    a gate with soft edges (chop)"""
    import numpy as np
    from .forge import dsp
    a, b = sp['rate'] if isinstance(sp['rate'], list) else (sp['rate'], sp['rate'])
    rate = a * (b / a) ** np.linspace(0.0, 1.0, len(y))
    ph = 2 * np.pi * np.cumsum(rate) / dsp.SR
    c = np.cos(ph)
    m = 0.5 + 0.5 * (np.clip(4.0 * c, -1.0, 1.0) if sp['shape'] == 'chop' else c)
    return y * (1.0 - sp['depth'] * (1.0 - m))[:, None]


def _moved(y, lay):
    """what is done to a layer as a whole once its events are laid: (of a scatter: its pan, fades, swell, decay),
    bend, pass, level, spin and a moving pan"""
    import numpy as np
    from scipy import ndimage
    from .forge import dsp
    from .synth import design as D
    if 'scatter' in lay:                  # (its pan placed the events already)
        y = D.fade(D.fade(y, lay.get('fade_in', 0.0), 0.0), 0.0, lay.get('fade_out', 0.0), shape='db')
        if lay.get('swell'):
            y = _slope(y, lay['swell'], -30.0, True)
        if lay.get('decay'):
            y = _slope(y, lay['decay'], -50.0, False)
    if 'bend' in lay:
        y = _bend(y, lay['bend'])
    if 'pass' in lay:
        y = _pass(y, lay['pass'])
    if 'level' in lay:
        g = 10 ** (_curve_at(lay['level'], len(y)) / 20)
        y = y * ndimage.uniform_filter1d(g, dsp.n_of(0.005))[:, None]
    if 'spin' in lay:
        y = _spin(y, lay['spin'])
    if isinstance(lay.get('pan'), list):
        y = _panned(y, np.linspace(*lay['pan'], len(y)))
    return y


def _following(lead, fo):
    """the gain a follower gets from the layer it follows: the leader's level (RMS over `smooth` s, centred, as a
    share of its loudest moment, 0 .. 1); amount a over 0: (1 - a) + a * level, the follower as loud as the leader
    is at a = 1; amount under 0: 1 - |a| * level, the follower giving way where the leader is loud"""
    import numpy as np
    from scipy import ndimage
    from .forge import dsp
    p = np.mean(lead ** 2, axis=1)
    env = np.sqrt(np.maximum(ndimage.uniform_filter1d(p, dsp.n_of(fo['smooth'])), 0.0))
    env = env / env.max() if env.max() > 0 else env
    a = fo['amount']
    return (1.0 - a) + a * env if a >= 0 else 1.0 - abs(a) * env


def _mix(rc, seed, notes, want=None):
    """a checked recipe's layers laid, shaped as wholes, followed, mixed, put in its space and cut at `want` s, before
    it is levelled: (stereo samples, how many events were laid)"""
    import numpy as np
    from .forge import dsp
    from .synth import design as D, space
    placed = [_placed(lay, seed, i, notes) for i, lay in enumerate(rc['layers'])]
    own = []                       # each layer from its own start (its `at`): its events, then what is done to it
    for lay, p in zip(rc['layers'], placed):
        a0 = dsp.n_of(lay.get('at', 0.0)) if lay.get('at', 0.0) > 0 else 0
        at = [(dsp.n_of(t) if t > 0 else 0) - a0 for _, t, _ in p]
        tr = np.zeros((max(a + len(y) for a, (y, _, _) in zip(at, p)), 2))
        for a, (y, t, g) in zip(at, p):
            tr[a:a + len(y)] += y * g
        moved = any(k in lay for k in MOVES) or isinstance(lay.get('pan'), list)
        tr = _moved(tr, lay) if moved else tr
        # the length a layer reaches, counted as it always was (a take at 0 s counts one sample more)
        own.append((a0, tr, a0 + len(tr) if moved else max(dsp.n_of(t) + len(y) for y, t, _ in p)))
    n = max(end for _, _, end in own)
    own = [(a0, tr) for a0, tr, _ in own]
    tracks = []
    for a0, y in own:
        tr = np.zeros((n, 2))
        tr[a0:a0 + len(y)] += y
        tracks.append(tr)
    mix = np.zeros((n, 2))
    for lay, tr in zip(rc['layers'], tracks):
        mix += tr * _following(tracks[lay['follow']['layer']], lay['follow'])[:, None] if 'follow' in lay else tr
    if rc.get('space', 'dry') != 'dry':
        mix = space.room(mix, rc['space'], r=dsp.rng(31337 + seed))
    if want and len(mix) > dsp.n_of(want):
        mix = D.fade(mix[:dsp.n_of(want)], 0.0, min(0.5, max(0.05, 0.25 * want)), shape='db')
    elif want and len(mix) < dsp.n_of(want) - dsp.n_of(0.1):
        notes.append(f'{len(mix) / dsp.SR:.2f} s, not the {want:g} s asked: the layers end there; make one longer '
                     '(dur, events, scatter, stretch) or move it later (at)')
    if rc.get('kind', 'event') == 'ambience':
        mix = D.fade(D.fade(mix, 0.3, 0.0), 0.0, 0.6, shape='db')
    many = sum(len(p) for p in placed) > 1 or any('scatter' in lay or 'recipe' in lay for lay in rc['layers'])
    return mix, many


MOVES = ('scatter', 'bend', 'pass', 'level', 'spin')


def render(recipe, seed=0, lufs=None, peak_db=-1.0, dur=None):
    """the sound of a recipe: (y, meta). y: stereo float samples at 48 kHz, levelled like a catalogue sound of its
    kind (the true peak at peak_db under the kind's loudest level, or `lufs`). seed: another take of every layer
    and new noise; the same seed gives the same samples. dur: a length to cut the file at instead of the recipe's.
    meta has the fields of any take of `earshot.make` (id "recipe/<name>", kind, seed, dur_s, peak_dbfs, lufs_max,
    checks: the audit every sound gets, peak_s, onset_s) and source "recipe", `parts` (the designs and sources it
    is built from) and `notes` (a control a design could not follow, a length that could not be kept)"""
    from types import SimpleNamespace
    from .forge import dsp
    from .make import marks
    from .sounds import registry
    from .synth import master
    rc = check(recipe)
    seed, notes = int(seed), []
    mix, many = _mix(rc, seed, notes, want=dur if dur is not None else rc.get('dur'))
    bed = rc.get('kind', 'event') == 'ambience'
    kind = 'bed' if bed else 'object'
    try:
        y = master.finish(mix, peak_db=peak_db, lufs=lufs, cap=registry.LOUDEST.get(kind, registry.LOUDEST['']),
                          lift=0.0 if bed else registry.LIFT, sparse=registry.SPARSE if bed else 0.0)
    except ValueError as e:
        raise RecipeError(f'the recipe {rc["name"]} made no sound: {e}') from None
    # several events are the design of a recipe, with silence between them in a dry space; a reversed layer ends
    # on its strike: that stop is the design, no cut tail; and what the audit allows a part's design (a campfire's
    # crackles, the ring of a part that has a note) it allows the sound built from it
    lays = rc['layers'] + [sub for lay in rc['layers'] if 'recipe' in lay
                           for sub in cookbook()[lay['recipe']]['layers']]
    used = [registry.get(lay['sound']) for lay in lays if 'sound' in lay]
    allows = (('stacked', 'silence', 'tail') if many else ()) + (
        ('truncated',) if any(lay.get('reverse') for lay in lays) else ()) + tuple(
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
        if 'glide' in lay:
            bits.append(f'{lay.get("pitch", 0.0):+g} to {lay["glide"]:+g} semitones')
        elif lay.get('pitch'):
            bits.append(f'{lay["pitch"]:+g} semitones')
        if lay.get('bend'):
            bits.append('bent through ' + ', '.join(f'{v:+g}' for _, v in lay['bend']) + ' semitones')
        if lay.get('reverse'):
            bits.append('reversed')
        if lay.get('stretch'):
            bits.append(f'stretched {lay["stretch"]:g} times')
        if lay.get('events'):
            bits.append(f'{lay["events"]["count"]} times')
        if lay.get('scatter'):
            sc = lay['scatter']
            rate = f'{sc["rate"]:g}' + (f' to {sc["rate_to"]:g}' if 'rate_to' in sc else '')
            bits.append(f'scattered {rate} a second over {sc["dur"]:g} s'
                        + (f' in pieces of {sc["grain"]:g} s' if sc.get('grain') else ''))
        if lay.get('pass'):
            bits.append(f'passing at {lay["pass"].get("speed", 20):g} m/s')
        if lay.get('spin'):
            r = lay['spin']['rate']
            bits.append(f'pulsing {r[0]:g} to {r[1]:g} Hz' if isinstance(r, list) else f'pulsing {r:g} Hz')
        if lay.get('level'):
            bits.append('its level drawn: ' + ', '.join(f'{v:+g}' for _, v in lay['level']) + ' dB')
        if isinstance(lay.get('pan'), list):
            bits.append(f'moving from {lay["pan"][0]:g} to {lay["pan"][1]:g}')
        if lay.get('follow'):
            fo = lay['follow']
            fo = fo if isinstance(fo, dict) else {'layer': fo}
            bits.append(f'{"giving way to" if fo.get("amount", 1.0) < 0 else "following"} layer {fo["layer"]}')
        for k, word in (('lowpass', 'under'), ('highpass', 'over')):
            if k in lay:
                v = lay[k]
                bits.append(f'{word} {v[0]:g} to {v[1]:g} Hz' if isinstance(v, list) else f'{word} {v:g} Hz')
        what = lay.get('sound') or (f'the recipe {lay["recipe"]}' if 'recipe' in lay else
                                    f'{lay.get("color", "pink")} noise' if lay['source'] == 'noise'
                                    else f'a {lay["hz"]:g} Hz sine')
        out.append(what + (f' ({", ".join(bits)})' if bits else ''))
    return out


# ---------------------------------------------------------------------------------------------------- command line
HOW = """A recipe builds a sound the catalogue lacks from its own parts. Say what physically makes the sound, find
two to four real parts (earshot sounds --search <word>), write the layers, render, read the result, change one thing.
Real recordings are the material; these give them a structure: one clap scattered is applause, a drop on tin
scattered is rain on a tin roof, a running motor that passes is a car going by, a moo bent is a whale's song.
  layer:   "sound": a design id   or   "source": "noise" | "sine" (with "dur"; a sine with "hz" 20..150)
           or   "recipe": a cookbook recipe's name (one level deep)
  on it:   seed, dur, pitch (semitones, -24..12: lower is slower and bigger), glide (the pitch at the take's end),
           reverse, cut [from, to], stretch (0.5..2: longer at the same pitch), halt (s: it runs down to a stop),
           lowpass / highpass (Hz, or [from, to] to move), pan (-1..1, or [from, to] to move across), gain (dB),
           at (s), fade_in, fade_out, swell, decay, size / bright / hard / energy / ring / motion / width (-1..1,
           where the design has them), events {"count", "gap", "faster", "softer", "loose"},
           follow {"layer", "amount", "smooth"} (its loudness follows another layer's; amount under 0: it gives
           way to it)
  many:    scatter {"rate" (events a second, 0.5..400), "rate_to", "dur" (s), "pitch" (+- semitones), "gain" (up
           to dB softer), "pan" (0..1 to either side), "grain" (s: random pieces of the take)}
  moving:  level [[s, dB], ...], bend [[s, semitones], ...], pass {"at" (s, closest), "speed" (m/s, 1..100)},
           spin {"rate" (Hz 0.5..60, or [from, to]), "depth" (0..1), "shape" ("smooth" | "chop")}; times from the
           layer's start
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
