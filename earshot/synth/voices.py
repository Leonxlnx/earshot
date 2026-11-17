"""Every pitched voice earshot can play, behind one call: struck instruments (earshot.synth.struck), strings and
pianos (earshot.synth.strings), synthesizer patches (earshot.synth.patches) and real instruments learned note by
note from public-domain recordings (earshot.synth.played).

The real instruments take the names of the voices they replace: `marimba`, `harp`, `pizzicato` and `felt_piano`
(played by a real upright piano at its soft dynamic) are recordings, not models: a synthesised interface figure
sounds technical and thin next to a real note, and a take of a recording is heard as the recording itself.
struck.note('marimba') and
strings.note('harp') still play the physical models directly.

    from earshot.synth import voices
    y = voices.play('kalimba', 'E5', vel=0.6)                    # stereo (n, 2), rings out
    y = voices.play('soft_pluck', 'A4', dur=0.2, bright=-0.3)    # held 0.2 s, a little darker
    voices.table()['vibraphone']                                 # engine, says, range, material ...

A design names a voice instead of an engine, so a figure (a success chime, a notification motif, a logo) can be
played by any of them and a theme can swap the voice of a whole set. Controls are the shared axes in [-1, 1]
(0 = the voice as built); a voice ignores the ones it does not have. A synthesizer patch is played without its
own delay and reverb unless `space` is given, so every voice of a set sits in the one room its design adds.
"""
import importlib

from ..forge import dsp, knobs

# engine module, its play function, its table of voices, the controls it takes
ENGINES = {
    'struck': ('note', 'INSTRUMENTS', ('bright', 'hard', 'ring', 'size', 'position', 'width', 'vary')),
    'strings': ('note', 'INSTRUMENTS', ('bright', 'hard', 'ring', 'size', 'position', 'width', 'vary')),
    'patches': ('play', 'PATCHES', ('bright', 'hard', 'ring', 'size', 'motion', 'width', 'dirt', 'space', 'vary')),
    'played': ('play', 'TABLE', ('bright', 'hard', 'vary')),       # last: its voices replace the models' of one name
}
# what a voice is made of: how a theme ("wood", "glass", "felt") finds its voices
MATERIAL = {
    'glockenspiel': 'metal', 'celesta': 'metal', 'vibraphone': 'metal', 'crotales': 'metal', 'tubular_bell': 'metal',
    'wind_chime': 'metal', 'hand_bell': 'metal', 'church_bell': 'metal', 'triangle': 'metal', 'handpan': 'metal',
    'steel_drum': 'metal', 'cowbell': 'metal', 'coin': 'metal', 'metal_ping': 'metal', 'singing_bowl': 'metal',
    'kalimba': 'tine', 'music_box': 'tine', 'marimba': 'wood', 'xylophone': 'wood', 'wood_block': 'wood',
    'wine_glass': 'glass', 'glass_clink': 'glass',
    'felt_piano': 'felt', 'grand_piano': 'piano', 'electric_piano': 'electric', 'reed_piano': 'electric',
    'harp': 'string', 'nylon_guitar': 'string', 'steel_guitar': 'string', 'muted_guitar': 'string',
    'pizzicato': 'string', 'koto': 'string', 'upright_bass': 'string', 'harpsichord': 'string', 'dulcimer': 'string',
}
# a patch's held time when a design gives none (s), by its kind
HOLD = {'pluck': 0.12, 'key': 0.3, 'bell': 0.15, 'pad': 1.6, 'bass': 0.35, 'lead': 0.3, 'hit': 0.4, 'fx': 0.6}
_table = None


def table():
    """{voice: {'engine', 'says', 'range', 'default_pitch', 'material', ...}} over the engines that import"""
    global _table
    if _table is None:
        t = {}
        for eng, (_, tab, _) in ENGINES.items():
            try:
                mod = importlib.import_module(f'earshot.synth.{eng}')
            except ImportError:
                continue
            for name, info in getattr(mod, tab).items():
                old = t.get(name, {})              # a real instrument keeps how its model was described
                t[name] = {**info, 'says': old.get('says', info['says']), 'engine': eng,
                           'material': MATERIAL.get(name, 'synth' if eng == 'patches' else None)}
        _table = t
    return _table


def names(material=None):
    return [n for n, v in table().items() if material is None or v['material'] == material]


def info(voice):
    t = table()
    if voice not in t:
        raise ValueError(f'unknown voice {voice!r}: one of {", ".join(sorted(t))}')
    return t[voice]


def clamp(voice, hz):
    """hz moved by octaves into the voice's range"""
    lo, hi = (knobs.hz(p) for p in info(voice)['range'])
    while hz < lo * 0.999:
        hz *= 2.0
    while hz > hi * 1.001:
        hz /= 2.0
    return hz


def play(voice, pitch=None, vel=0.7, dur=None, r=None, **controls):
    """one note of a voice: stereo float64 (n, 2), starting at its attack. pitch: note name / MIDI / Hz (None = the
    voice's default; outside its range it is moved by octaves); vel 0..1 (level and timbre); dur None = it rings out
    (a patch is held for its kind's usual time), else it is damped / released at dur s"""
    v = info(voice)
    fn, _, takes = ENGINES[v['engine']]
    mod = importlib.import_module(f'earshot.synth.{v["engine"]}')
    hz = clamp(voice, knobs.hz(v['default_pitch'] if pitch is None else pitch))
    kw = {k: float(c) for k, c in controls.items() if k in takes and c is not None}
    if v['engine'] == 'patches':
        kw.setdefault('space', -1.0)       # a patch comes without its own delay and reverb unless asked: the design
        if dur is None:                    # that plays it puts every voice in one shared room
            dur = HOLD.get(v.get('kind'), 0.3)
    return getattr(mod, fn)(voice, hz, vel=float(vel), dur=dur, r=r if r is not None else dsp.rng(0), **kw)
