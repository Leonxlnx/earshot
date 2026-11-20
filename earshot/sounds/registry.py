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
         'boom', 'noise', 'texture', 'bed', 'pad', 'stinger', 'voice', 'object', 'music')
DOMAINS = ('ui', 'game', 'motion', 'ambience', 'music')
MODULES = ('ui', 'game', 'action', 'object', 'motion', 'drums', 'brand', 'synth', 'nature', 'beds', 'tonal', 'retro',
           'learned', 'bank')
SCALES = {'major': (0, 2, 4, 5, 7, 9, 11), 'minor': (0, 2, 3, 5, 7, 8, 10), 'pentatonic': (0, 2, 4, 7, 9),
          'minor_pentatonic': (0, 3, 5, 7, 10), 'lydian': (0, 2, 4, 6, 7, 9, 11), 'dorian': (0, 2, 3, 5, 7, 9, 10),
          'whole_tone': (0, 2, 4, 6, 8, 10)}
# the voice a material names, for tonal designs asked for "in wood", "glassy", "metallic" ... without a voice
VOICE_OF = {'wood': 'marimba', 'glass': 'glockenspiel', 'metal': 'vibraphone', 'soft': 'felt_piano',
            'digital': 'glass_pluck', 'retro': 'square_lead'}
# the loudest a finished sound may be at full peak (momentary-max LUFS), by kind: beds sit low by nature
LOUDEST = {'': -14.0, 'pad': -18.0, 'texture': -18.0, 'bed': -22.0, 'music': -16.0}
# the most a short, peaky one-shot is raised (dB, its spike folded over: master.finish `lift`): at the same loudness a
# short click sounds barely there next to a longer sound; beds, pads and music keep their own level
LIFT = 14.0
# the prior of a learned design (earshot.sounds.learned: recordings in the public domain played as new takes):
# where an invented design of the same thing exists, the learned one wins a tie
LEARNED = 1.1
# the most a bed held under its cap by a few peaks is raised (dB: master.finish `sparse`): a campfire's cracks
SPARSE = 8.0
# kinds of held tones: asked for a loudness (a set's level, `lufs`) one of these with a pitch is not folded or
# limited to reach it but stops at its peak (master.finish `tonal`: both bend a held tone measurably)
TONAL = ('note', 'chime', 'figure', 'stinger')
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

    def __init__(self, d, pitch=None, dur=None, voice=None, material=None, theme=None, vary=0.35, loop=None,
                 **axes):
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
        self.played = None           # the voice a design really played, when it is not the one asked for
        self.seed = 0                # the take's seed (render sets it): a design of several recordings picks by it
        self.member = None           # the recording a design of several played ({'n', 'of', ...}), for meta
        self.loop = d.loop and loop is not False      # a looping design renders once, with fades, on loop=False
        self.seam = d.loop and loop is True           # the request said "loop" / "seamless": the seam over the length

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
        """the voice that plays a tonal design: the one asked for; else the one a requested material names (wood: a
        marimba, glass: a glockenspiel ...: VOICE_OF); else the theme's; else the design's own"""
        self.used.add('voice')
        if self._voice:
            return self._voice
        if VOICE_OF.get(self._material):
            self.used.add('material')
            return VOICE_OF[self._material]
        return self._theme.get('voice') or self._d.voice

    def length(self):
        """the length asked for in seconds, or None when the request named none (seconds() falls back on the
        design's own; this does not)"""
        if self._dur is not None:
            self.used.add('dur')
        return self._dur

    def scale(self):
        """the scale of the request or the theme ('major' when none was given)"""
        return self._theme.get('scale') or 'major'

    def room(self, default='room'):
        """the space a design sits in: the theme's, else the design's own (earshot.synth.space)"""
        return self._theme.get('space') or default

    @property
    def material(self):
        self.used.add('material')
        return self._material or self._theme.get('material') or self._d.material


def member(n, of):
    """meta['member'] of a design that played recording n (1-based) of its `of`: the number and a plain line. The
    title a recording's uploader gave it (a brand, a file name) stays in the store's index and CREDITS.md: an agent
    needs to see that a seed played another recording, not whose"""
    return {'n': int(n), 'of': int(of), 'recording': f'recording {int(n)} of {int(of)}'}


def seed_of(id, seed):
    """the generator seed of a take: the same id and seed give the same samples, different ids differ"""
    return (zlib.crc32(id.encode()) + 7919 * int(seed)) & 0xFFFFFFFF


HELD = 0.05                  # s: a looping design's length within this of the one asked for is the length asked for
CUT_IN, CUT_OUT, CUT_OUT_MAX = 0.03, 0.1, 0.5   # a cut from a bed: fade in (s), fade out (share of it, at most s)


def fit(y, d, c):
    """(y, note) of a looping design whose length differs from the one asked for (c.length()); note None when it
    does not. A bed or a texture asked shorter than its shortest seamless loop is cut from it to the length asked
    for (a CUT_IN fade in, a fade out of CUT_OUT of the length, at most CUT_OUT_MAX) and is no loop then (c.loop
    False), unless the request asked for a loop (c.seam): then the shortest loop stays, and the note says so. A
    music loop keeps whole bars: the note says the nearest length of whole bars was played"""
    want, got = c._dur, len(y) / dsp.SR
    if not (d.loop and want) or abs(got - want) <= HELD:
        return y, None
    if d.kind == 'music':
        return y, f'{got:.2f} s: a music loop is whole bars, and these are the nearest to the {want:g} s asked'
    if got < want:
        return y, f'{got:.2f} s, shorter than the {want:g} s asked: the longest this sound is made'
    if c.seam:
        return y, (f'a seamless loop of {got:.2f} s, the shortest this sound has; {want:g} s was asked, and a loop '
                   'was asked for: the seam was kept over the length')
    shortest = got if c.loop else None
    c.loop = False
    y = dsp.fade(y[:dsp.n_of(want)], CUT_IN, min(CUT_OUT * want, CUT_OUT_MAX))
    return y, (f'a cut of {want:g} s, not a loop' + (f': the shortest seamless loop of this sound is {shortest:.2f} s'
                                                    if shortest else ''))


def render(id, seed=0, peak_db=-1.0, lufs=None, check=True, **request):
    """one finished take of a design: (y, meta). request: pitch, dur, voice, material, theme (a dict, see theme.py),
    vary, loop (False = a looping design once, with fades) and the axes. y is float64, stereo (n, 2) or mono (n,), levelled (master.finish). meta: id, kind, seed, dur_s,
    peak_dbfs, lufs_max, pitch / voice / material when the design used them, `member` when the design plays one of
    several recordings (earshot.sounds.learned: which one, of how many), `ignored` (what was asked for and not
    read), and the audit (`checks`: forge.sfx.check, with the design's `allows` as notes). Level: the true peak at
    peak_db, held under the kind's loudest level (LOUDEST); or exactly `lufs` (momentary max) when given."""
    from ..synth import master
    d = get(id)
    c = Controls(d, **request)
    c.seed = int(seed)
    y = d.fn(c, dsp.rng(seed_of(id, seed)))
    y = master.finish(y, peak_db=peak_db, lufs=lufs, loop=c.loop, cap=LOUDEST.get(d.kind, LOUDEST['']),
                      lift=0.0 if d.kind in LOUDEST else LIFT, sparse=SPARSE if d.kind == 'bed' else 0.0,
                      tonal=bool(d.pitch) and d.kind in TONAL)
    y, note = fit(y, d, c)
    meta = {'id': id, 'kind': d.kind, 'seed': int(seed), 'dur_s': round(len(y) / dsp.SR, 3),
            'peak_dbfs': round(master.true_peak_db(y), 2), 'lufs_max': round(master.loudness(y), 1)}
    if 'pitch' in c.used and (d.pitch or c._pitch):
        meta['pitch'] = knobs.note(c.hz())
    if 'voice' in c.used:
        meta['voice'] = c.played or c.voice
    if 'material' in c.used and c.material:
        meta['material'] = c.material
    if c.member:
        meta['member'] = dict(c.member)
    # a voice or material that is the design's own was honoured by choosing the design ("a kalimba note", "8-bit"),
    # as is a material read from a word of the thing's own name ("a cork popping out of a bottle": cork/bottle)
    from .lexicon import MATERIALS
    named = {MATERIALS.get(w) for w in id.replace('/', '_').split('_')}
    own = {k for k, v in (('voice', d.voice), ('material', d.material))
           if request.get(k) and (request[k] == v or request[k] in d.tags or (k == 'material' and request[k] in named))}
    if note:
        meta['note'] = note          # the length differs from the one asked for, and why, in plain words
        c.used.add('dur')
    elif c._dur and not (0.75 * c._dur <= meta['dur_s'] <= 1.5 * c._dur + 0.5):
        c.used.discard('dur')        # read but not kept: a struck sound rings as it rings
    ignored = sorted(c.asked - c.used - own)
    if ignored:
        meta['ignored'] = ignored
    if d.loop:
        meta['loop'] = c.loop
    if check:
        meta['checks'] = audit(y, d, c.loop)
    return y, meta


def audit(y, d, loop=None):
    """the model-free audit of a take (forge.sfx.check): {'ok', 'warnings', 'notes'}; findings the design allows
    (its rising pitch, its hard stop) are notes"""
    from ..forge import sfx
    c = sfx.check(y, role='hit' if d.kind in ('hit', 'thud', 'boom') else None)
    allows = d.allows + (('edge', 'truncated') if (d.loop if loop is None else loop) else ())   # a loop runs
    #                                                                                 into its own start
    if d.pitch:
        allows += ('ring',)        # the comb detector reads any steady harmonic tone as a comb: a note is one
    warn = [w for w in c.get('warnings', []) if w.get('code') not in allows]
    notes = list(c.get('notes', [])) + [w for w in c.get('warnings', []) if w.get('code') in allows]
    return {'ok': not warn, 'warnings': warn, 'notes': notes}


def voices_says(voice):
    """the descriptions of a voice itself (earshot.synth.voices), [] for none: what a tonal design sounds like
    besides its figure"""
    if not voice:
        return []
    from ..synth import voices
    return list(voices.info(voice)['says'][:2])
