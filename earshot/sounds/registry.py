"""The catalogue: every sound earshot can make is a design.

A design is a function that renders one sound from shared controls, plus what an agent needs to choose it: what it
sounds like (`says`: plain descriptions of the sound itself, never of its job), which jobs it plays (`uses`), where
(`domains`), its usual length and pitch, and the controls it answers to.

    from earshot import sounds
    y, meta = sounds.render('coin/bright', seed=2, pitch='B5', bright=0.3)      # stereo float64 at 48 kHz, finished
    sounds.rows(domain='game')                                                  # the catalogue as rows

Writing one (in a module of this package listed in MODULES):

    @design('tap/soft', kind='tap', says=('a soft fingertip tap', 'a small muted tick'), uses=('tap', 'button press'),
            domains=('ui',), dur=0.06, axes=('bright', 'hard', 'size'))
    def tap_soft(c, r):
        return ...                 # c: the controls (Controls), r: a numpy Generator; mono (n,) or stereo (n, 2)

The controls are the same for every design, so a request ("softer", "bigger", "longer", "in D") and a theme mean the
same thing everywhere: pitch and dur, a voice or a material where the design has one, and the axes in [-1, 1] with
0 = the sound as designed:
    bright   darker .. brighter              hard     softer attack .. harder attack
    ring     shorter, drier .. longer ring   size     smaller, lighter .. bigger, heavier
    space    no room .. more room            width    narrower .. wider
    energy   gentler .. more intense         motion   steadier .. more movement
A design reads what it supports; whatever a request set that the design never read is reported as ignored.
"""
import importlib
import zlib

import numpy as np

from ..forge import dsp, knobs

AXES = ('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy', 'motion')
KINDS = ('tap', 'click', 'tick', 'pop', 'note', 'chime', 'figure', 'hit', 'thud', 'whoosh', 'rise', 'fall', 'zap',
         'boom', 'noise', 'texture', 'bed', 'pad', 'stinger', 'voice', 'object')
DOMAINS = ('ui', 'game', 'motion', 'ambience', 'music')
MODULES = ('ui', 'tonal', 'retro', 'bank')   # the design modules of this package, in catalogue order
SCALES = {'major': (0, 2, 4, 5, 7, 9, 11), 'minor': (0, 2, 3, 5, 7, 8, 10), 'pentatonic': (0, 2, 4, 7, 9),
          'minor_pentatonic': (0, 3, 5, 7, 10), 'lydian': (0, 2, 4, 6, 7, 9, 11), 'dorian': (0, 2, 3, 5, 7, 9, 10),
          'whole_tone': (0, 2, 4, 6, 8, 10)}
# the loudest a finished sound may be at full peak (momentary-max LUFS), by kind: beds sit low by nature
LOUDEST = {'': -14.0, 'pad': -18.0, 'texture': -18.0, 'bed': -22.0}
_designs = {}
_loaded = False


class Design:
    """one catalogue entry (module doc); `allows` = audit findings that are this sound's design (e.g. a power-up's
    rising pitch), reported as notes instead of warnings"""

    def __init__(self, id, fn, kind, says, uses, tags, domains, dur, pitch, axes, voice, material, loop, allows,
                 prior):
        self.id, self.fn, self.kind, self.prior = id, fn, kind, float(prior)
        self.says, self.uses, self.tags, self.domains = tuple(says), tuple(uses), tuple(tags), tuple(domains)
        self.dur, self.pitch, self.axes = dur, pitch, tuple(axes)
        self.voice, self.material, self.loop, self.allows = voice, material, bool(loop), tuple(allows)

    def row(self):
        return {'id': self.id, 'kind': self.kind, 'says': list(self.says), 'uses': list(self.uses),
                'tags': list(self.tags), 'domains': list(self.domains), 'dur_s': self.dur, 'pitch': self.pitch,
                'axes': list(self.axes), 'voice': self.voice, 'material': self.material, 'loop': self.loop,
                'prior': self.prior}


def design(id, *, kind, says, uses=(), tags=(), domains=(), dur=None, pitch=None, axes=(), voice=None, material=None,
           loop=False, allows=(), prior=1.0):
    """register a design (module doc). id = 'family/name', lower case; kind one of KINDS; says: 2-3 descriptions of
    the sound; dur: its usual length in s; pitch: its default note when it has one; voice / material: its default
    when it can be played by several; prior: its weight when a request fits several designs equally (1 = normal,
    above 1 = the family's first choice)"""
    fam, _, name = id.partition('/')
    if not (fam and name and id == id.lower() and ' ' not in id):
        raise ValueError(f'bad design id {id!r}: use family/name in lower case')
    if kind not in KINDS:
        raise ValueError(f'{id}: unknown kind {kind!r}')
    bad = [a for a in axes if a not in AXES] + [d for d in domains if d not in DOMAINS]
    if bad or len(says) < 2:
        raise ValueError(f'{id}: unknown axes / domains {bad}' if bad else f'{id}: give at least two `says`')

    def wrap(fn):
        if id in _designs:
            raise ValueError(f'design {id} is defined twice')
        _designs[id] = Design(id, fn, kind, says, uses, tags, domains, dur, pitch, axes, voice, material, loop, allows,
                              prior)
        return fn
    return wrap


def load():
    """import the design modules once"""
    global _loaded
    if not _loaded:
        for m in MODULES:
            importlib.import_module(f'{__package__}.{m}')
        _loaded = True
    return _designs


def ids():
    return list(load())


def get(id):
    d = load().get(id)
    if d is None:
        import difflib
        near = difflib.get_close_matches(id, list(_designs), n=4, cutoff=0.4)
        raise ValueError(f'unknown sound {id!r}' + (f'. Did you mean {", ".join(near)}?' if near else ''))
    return d


def rows(domain=None, kind=None, family=None):
    """the catalogue as rows (Design.row), optionally one domain / kind / family"""
    return [d.row() for d in load().values()
            if (domain is None or domain in d.domains) and (kind is None or d.kind == kind)
            and (family is None or d.id.split('/')[0] == family)]


class Controls:
    """what one render was asked for, as a design reads it (module doc). Axes are attributes (c.bright ...);
    `used` collects what the design read"""

    def __init__(self, d, pitch=None, dur=None, voice=None, material=None, theme=None, vary=0.35, **axes):
        bad = [k for k in axes if k not in AXES]
        if bad:
            raise ValueError(f'unknown controls {bad}: the axes are {", ".join(AXES)}')
        self._d, self._theme = d, theme or {}
        self._pitch = None if pitch is None else knobs.hz(pitch)
        self._dur = None if dur is None else float(dur)
        self._voice, self._material = voice, material
        th = self._theme.get('axes') or {}
        self._ax = {a: float(np.clip((axes.get(a) or 0.0) + (th.get(a) or 0.0), -1.0, 1.0)) for a in AXES}
        self.vary = float(np.clip(vary, 0.0, 1.0))
        self.asked = {k for k, v in dict(pitch=pitch, dur=dur, voice=voice, material=material, **axes).items()
                      if v is not None and v != 0.0}
        self.used = set()

    def __getattr__(self, name):
        if name in AXES:
            self.used.add(name)
            return self._ax[name]
        raise AttributeError(name)

    def axes(self, *names):
        """{axis: value} of the named axes: what a design passes on to a voice"""
        return {a: getattr(self, a) for a in names}

    def lerp(self, axis, lo, hi, mid=None, log=False):
        """the axis mapped onto a parameter: lo at -1, mid at 0 (default: half way, geometric when log), hi at 1"""
        v = getattr(self, axis)
        if mid is None:
            mid = float(np.sqrt(lo * hi)) if log else 0.5 * (lo + hi)
        a, b = (mid, hi) if v >= 0 else (mid, lo)
        return float(a * (b / a) ** abs(v)) if log else float(a + (b - a) * abs(v))

    def db(self, axis, span):
        """the axis as a linear gain: +-span dB at +-1"""
        return float(10 ** (getattr(self, axis) * span / 20))

    def seconds(self, default=None):
        """the length asked for, else the default (the design's own when none is given)"""
        self.used.add('dur')
        return self._dur if self._dur is not None else (default if default is not None else self._d.dur)

    def stretch(self):
        """the length asked for relative to the design's own (1.0 when none was asked): for designs that scale"""
        self.used.add('dur')
        return 1.0 if self._dur is None or not self._d.dur else self._dur / self._d.dur

    def hz(self, default=None):
        """the root pitch in Hz: the pitch asked for; else the default (the design's own when none is given) moved to
        the nearest note of the theme's key"""
        self.used.add('pitch')
        if self._pitch is not None:
            return self._pitch
        f = knobs.hz(default if default is not None else self._d.pitch)
        key = self._theme.get('key')
        if key:
            st = (knobs.midi(f'{key}4') - 12 * np.log2(f / 440.0) - 69) % 12
            f *= 2 ** ((st - 12 if st > 6 else st) / 12)
        return float(f)

    def semis(self, steps, base=None):
        """[Hz] `steps` semitones above the root"""
        f = self.hz(base)
        return [f * 2 ** (s / 12) for s in steps]

    def degrees(self, degs, base=None, scale=None):
        """[Hz] at scale degrees (0 = the root, 7 = an octave up in a seven-note scale) of the theme's scale"""
        sc = SCALES[scale or self._theme.get('scale') or 'major']
        return self.semis([12 * (g // len(sc)) + sc[g % len(sc)] for g in degs], base)

    @property
    def voice(self):
        """the voice that plays a tonal design: asked for, else the theme's, else the design's own"""
        self.used.add('voice')
        return self._voice or self._theme.get('voice') or self._d.voice

    def room(self, default='room'):
        """the space a design sits in: the theme's, else the design's own (earshot.synth.space)"""
        return self._theme.get('space') or default

    @property
    def material(self):
        self.used.add('material')
        return self._material or self._theme.get('material') or self._d.material


def seed_of(id, seed):
    """the generator seed of a take: the same id and seed give the same samples, different ids differ"""
    return (zlib.crc32(id.encode()) + 7919 * int(seed)) & 0xFFFFFFFF


def render(id, seed=0, peak_db=-1.0, lufs=None, check=True, **request):
    """one finished take of a design: (y, meta). request: pitch, dur, voice, material, theme (a dict, see theme.py),
    vary and the axes. y is float64, stereo (n, 2) or mono (n,), levelled (master.finish). meta: id, kind, seed, dur_s,
    peak_dbfs, lufs_max, pitch / voice / material when the design used them, `ignored` (what was asked for and not
    read), and the audit (`checks`: forge.sfx.check, with the design's `allows` as notes). Level: the true peak at
    peak_db, held under the kind's loudest level (LOUDEST); or exactly `lufs` (momentary max) when given."""
    from ..synth import master
    d = get(id)
    c = Controls(d, **request)
    y = d.fn(c, dsp.rng(seed_of(id, seed)))
    y = master.finish(y, peak_db=peak_db, lufs=lufs, loop=d.loop, cap=LOUDEST.get(d.kind, LOUDEST['']))
    meta = {'id': id, 'kind': d.kind, 'seed': int(seed), 'dur_s': round(len(y) / dsp.SR, 3),
            'peak_dbfs': round(master.true_peak_db(y), 2), 'lufs_max': round(master.loudness(y), 1)}
    if 'pitch' in c.used and (d.pitch or c._pitch):
        meta['pitch'] = knobs.note(c.hz())
    if 'voice' in c.used:
        meta['voice'] = c.voice
    if 'material' in c.used and c.material:
        meta['material'] = c.material
    ignored = sorted(c.asked - c.used)
    if ignored:
        meta['ignored'] = ignored
    if check:
        meta['checks'] = audit(y, d)
    return y, meta


def audit(y, d):
    """the model-free audit of a take (forge.sfx.check): {'ok', 'warnings', 'notes'}; findings the design allows
    (its rising pitch, its hard stop) are notes"""
    from ..forge import sfx
    c = sfx.check(y, role='hit' if d.kind in ('hit', 'thud', 'boom') else None)
    allows = d.allows + (('edge', 'truncated') if d.loop else ())     # a loop runs into its own start
    if d.pitch:
        allows += ('ring',)        # the comb detector reads any steady harmonic tone as a comb: a note is one
    warn = [w for w in c.get('warnings', []) if w.get('code') not in allows]
    notes = list(c.get('notes', [])) + [w for w in c.get('warnings', []) if w.get('code') in allows]
    return {'ok': not warn, 'warnings': warn, 'notes': notes}
