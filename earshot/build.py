"""earshot build: the workbench on which a sound the catalogue lacks is built from its recorded parts, with earshot as
the ears.

    earshot build "hail on a tin roof" -o hail.wav             # draft, render, hear, tune: hail.wav, hail.recipe.json
    earshot build "a whale singing" --recipe whale.json         # your own recipe: render it, hear it, say what is off
    earshot build "a stampede" --rounds 40 --without step/gravel rain --json

    from earshot.build import build, draft, hear, tune
    build('hail on a tin roof', out='hail.wav')['summary']

Four steps, each its own call:

draft(words)   one to three first recipes (earshot.recipe) from the words alone, without a language model. The reader
               (earshot.brief) says what the thing is and what it does; the words choose a structure: one event, a
               row of events, very many small events (`scatter`: rain, hail, applause, a swarm), something that runs
               (a bed that turns: `spin`), something that passes (`pass`), a creature's voice (moved in pitch, `bend`),
               something drawn out (`stretch`), something made up (a reversed swell into a hit, air), or a place (two
               or three beds). The parts are the recorded designs nearest to the words and to each thing they name:
               by the listening model's text-to-audio atlas when the listen extra is installed, by the reader's
               ranking otherwise; never an instrument, a note or a synthesized design. A cookbook recipe of the thing
               is the first draft. `without` forbids designs (an id) or whole families (a family name).
hear(recipe, words)
               renders and says in plain lines what an agent can act on: clean or what the audit found; length, where
               the hit is, how bright and how dense, against what the words ask and against the nearest real design
               of the catalogue; with the listen extra also what it is heard as (the nearest designs by sound, the
               nearest words), its fit to the words as a z beside the z of the design `make` would hand over (the
               verdict: "sounds like it" when clean, z >= 3.0 and at least 1.0 over that design; else "a sketch"),
               recorded or synthesized (earshot.realness), a steady tone heard as a note, and per layer what it adds
               (the fit with the layer left out). Every remark names the field of the recipe to change.
tune(recipe, words, rounds)
               keeps the structure and moves the numbers (gain, pitch, cutoffs, at, rates, depths, lengths) within the
               ranges earshot.recipe checks, by a seeded search: each round changes one or two numbers, renders,
               hears once and keeps the change when the score rises. The score is the fit to the words minus
               penalties: not clean, heard as synthesized, a steady tone, far from the length asked. The same seed
               gives the same result. Without the listen extra it can only keep the sound clean and near the length.
build(words)   draft, hear the drafts, tune the best, hear it again; writes the sound and the recipe beside it.

The listening model only judges; nothing here generates audio with a model. Every sound is the catalogue's own
designs (most of them learned from public-domain recordings) layered, moved and placed by the recipe.
"""
import copy
import json
import math
import os
import re
import time

from . import brief, recipe as R

ROUNDS = 20             # tuning rounds of a default build: a render and one embedding each
SEED = 0
GOOD_Z = 3.0            # the fit a sound needs to be heard as its words (match.VERDICTS 'strong')
OVER = 1.0              # and its lead over the design `make` would hand over
LISTEN_INSTALL = 'pip install "earshot[listen] @ git+https://github.com/Leonxlnx/earshot"'
PLAYED = {'synth', 'instrument', 'tonal', 'chip', '8-bit', 'arcade', 'retro', 'glitch', 'drums', 'kit', 'music'}
PLAYED_KINDS = {'note', 'chime', 'figure', 'music', 'pad', 'stinger'}
BED_S = 4.0             # a scatter, a running thing or a place is drafted this long when the words give no length
#                         (long enough to hear it, short enough for a round of tuning to stay quick)
STRUCTURES = ('one', 'row', 'scatter', 'run', 'pass', 'voice', 'long', 'magic', 'place')
# words that say what kind of sound a request is, by structure; the first structure in ORDER whose words the
# request holds is drafted first
WORDS = {
    'magic': 'magic magical spell spells portal force field energy laser plasma teleport teleporting shield alien '
             'scifi sci fi spaceship ufo warp hologram enchanted cursed ghost ghostly phaser beam charging',
    'pass': 'passing pass passes past flyover flying overhead by zooming zooms whizzing whizzes driving',
    'place': 'ambience ambient atmosphere atmo tone city street forest jungle beach market office cafe restaurant '
             'station village park underwater cave harbour harbor countryside room',
    'scatter': 'rain raining hail hailstorm drizzle sleet pouring crackle crackling crackles fizz fizzing sizzle '
               'sizzling applause clapping crowd swarm bees insects marbles popcorn stampede rustle rustling '
               'chattering bubbling bubbles shower confetti hooves herd gravel pebbles coins clatter clattering',
    'run': 'motor engine machine fan drill mower lawnmower blender hum humming idle idling running runs treadmill '
           'generator compressor pump turbine printer conveyor saw grinder vacuum washing spinning whirring '
           'rotor helicopter propeller wheel modem projector',
    'voice': 'roar roaring growl growling howl howling call calling song singing sings moo mooing bark barking cry '
             'crying screech screeching whimper whimpering squeal squeak squeaking chirp chirping purr '
             'croak croaking snort grunt grunting bleat neigh whinny bray braying hoot hooting yelp trumpeting',
    'row': 'steps footsteps stomping stomp knocking knocks hammering ticking clockwork typing dripping clicking '
           'galloping gallop chopping sawing marching tapping',
    'long': 'slowly slow drawn creaking creak groan groaning stretching sinking',
}
ORDER = ('magic', 'pass', 'place', 'scatter', 'run', 'voice', 'row', 'long')
# a steady tone heard as a note (earshot.realness.signal_features): nearly every frame one line, most of the energy in
# a few lines, a level that hardly moves; a creature's call drifts and breathes and stays under it
TONE = {'frames': 0.9, 'lines': 0.75, 'crest': 12.0}
BIG = {'giant', 'huge', 'big', 'enormous', 'massive', 'whale', 'monster', 'deep', 'heavy', 'colossal', 'titan'}
SMALL = {'tiny', 'small', 'little', 'mini', 'baby', 'mouse', 'bird', 'chick', 'fairy', 'toy'}


# ------------------------------------------------------------------------------------------------------------ parts
def _family(sid):
    return sid.split('/')[0]


def _forbidden(sid, without):
    """True when `without` names the design itself or its family"""
    return sid in without or _family(sid) in without


def _real(row):
    """a design that is a recording of a thing of the world (learned or tagged real) and not played music"""
    tags = set(row.get('tags') or ())
    return (('real' in tags or row['id'] in brief._roles()) and not tags & PLAYED
            and row['kind'] not in PLAYED_KINDS)


def _without(without):
    if not without:
        return frozenset()
    if isinstance(without, str):
        without = re.split(r'[,\s]+', without)
    return frozenset(w.strip().strip('/') for w in without if w and w.strip())


def _content(text):
    return [w for w in brief.WORD.findall((text or '').lower()) if w not in brief.L.STOP and len(w) > 2]


def _phrases(b, words):
    """the parts of a request to find material for: the whole, the thing it names, what it happens on, and each
    content word on its own"""
    out = [words]
    sur = b.get('surface') or {}
    if sur.get('of'):
        out.append(sur['of'])
    if sur.get('words'):
        out.append(' '.join(sur['words']))
    if b.get('absent'):
        out.append(b['absent'])
    out += _content(b.get('text') or words)
    return list(dict.fromkeys(p for p in out if p and p.strip()))


def candidates(words, without=(), ears=None, k=12):
    """the recorded designs nearest to a request, best first: [{'id', 'score', 'for', 'kind', 'dur', 'loop'}], at most
    one design of a family; each found for the whole request or for one of its parts (`for`). With ears, the
    listening model's atlas scores them (how each design sounds and is described against the words), else the
    reader's ranking does"""
    from .sounds import registry
    without = _without(without)
    b = brief.read(words)
    rows = {r['id']: r for r in registry.rows()}
    ok = {i for i, r in rows.items() if _real(r) and not _forbidden(i, without)}
    best = {}
    phrases = _phrases(b, words)
    for n, ph in enumerate(phrases):
        weight = 1.0 if n == 0 else 0.8
        if ears is not None:
            try:
                zs = ears.atlas_z(ph)
            except Exception:       # noqa: BLE001  (no atlas: the reader ranks)
                zs = None
        else:
            zs = None
        if zs is None:
            cands = brief.rank(brief.read(ph), top=40)
            top = max((c['score'] or 0.0) for c in cands) if cands else 0.0
            zs = {c['id']: 3.0 * (c['score'] or 0.0) / top for c in cands if top > 0}
        for sid, z in zs.items():
            if sid in ok:
                s = weight * float(z)
                if s > best.get(sid, (-9.0, ''))[0]:
                    best[sid] = (s, ph)
    out, fams = [], set()
    for sid, (s, ph) in sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0])):
        if _family(sid) in fams:
            continue
        fams.add(_family(sid))
        r = rows[sid]
        out.append({'id': sid, 'score': round(s, 2), 'for': ph, 'kind': r['kind'], 'dur': r['dur_s'] or 0.5,
                    'loop': bool(r['loop'])})
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------------------------------------------------- structure
def structure(words):
    """the kind of sound a request is (STRUCTURES): what its words say, then how many events it asks for"""
    b = brief.read(words)
    said = set(brief.WORD.findall((b.get('text') or words).lower()))
    for s in ORDER:
        if said & set(WORDS[s].split()):
            if (b.get('events') or 0) > 1 and s in ('voice', 'long', 'magic'):
                return 'row'            # "a dog barking three times": the voice three times
            return s
    if (b.get('events') or 0) > 1:
        return 'row'
    return 'one'


def _has(op):
    return op in R.LAYER


def _name(words):
    s = re.sub(r'[^a-z0-9]+', '_', (words or 'sound').lower()).strip('_')
    return (s[:40].rstrip('_') or 'sound')


def _size(words):
    said = set(brief.WORD.findall(words.lower()))
    return -1 if said & BIG else 1 if said & SMALL else 0


def _layers(kind, parts, b, words):
    """the layers of a draft of one structure from ranked parts (candidates); [] when the parts do not fit it"""
    if not parts:
        return []
    want = b.get('dur')
    p = parts
    shorts = [x for x in p if not x['loop'] and x['dur'] <= 0.8]
    longs = [x for x in p if x['loop'] or x['dur'] >= 1.5]
    size = _size(words)
    lay = []
    if kind == 'one':
        lay.append({'sound': p[0]['id']})
        if len(p) > 1:
            lay.append({'sound': p[1]['id'], 'gain': -9.0})
        if len(p) > 2:
            lay.append({'sound': p[2]['id'], 'pitch': -5.0, 'lowpass': 2500.0, 'gain': -15.0})
    elif kind == 'row':
        n = int(b.get('events') or 4)
        n = max(2, min(n, R.EVENT_RANGE['count'][1]))
        gap = round(max(0.2, min(0.8, 1.1 * min(p[0]['dur'], 0.7))), 3)
        lay.append({'sound': p[0]['id'], 'events': {'count': n, 'gap': gap, 'loose': 0.15, 'softer': -0.5}})
        if len(p) > 1:
            lay.append({'sound': p[1]['id'], 'gain': -10.0,
                        'events': {'count': n, 'gap': gap, 'loose': 0.15, 'softer': -0.5}})
    elif kind == 'scatter':
        total = want or BED_S
        grain = shorts[0] if shorts else None
        bed = longs[0] if longs else None
        if grain and _has('scatter'):
            lay.append({'sound': grain['id'], 'scatter': {'rate': 14.0, 'dur': total, 'pitch': 2.0, 'gain': 9.0,
                                                          'pan': 0.6}})
        elif grain:
            n = int(min(R.EVENT_RANGE['count'][1], 24))
            lay.append({'sound': grain['id'], 'events': {'count': n, 'gap': round(total / n, 3), 'loose': 1.0}})
        if bed:
            lay.append({'sound': bed['id'], 'dur': total, 'gain': -6.0 if lay else 0.0})
        others = [x for x in p if x is not grain and x is not bed]
        if len(lay) < 2 and others:
            o = others[0]
            if not o['loop'] and _has('scatter'):
                lay.append({'sound': o['id'], 'gain': -8.0, 'scatter': {'rate': 6.0, 'dur': total, 'pitch': 2.0,
                                                                        'gain': 9.0, 'pan': 0.6}})
            elif o['loop']:
                lay.append({'sound': o['id'], 'dur': total, 'gain': -8.0})
    elif kind == 'run':
        total = want or BED_S
        main = longs[0] if longs else p[0]
        first = {'sound': main['id'], 'dur': total}
        if _has('spin'):
            first['spin'] = {'rate': 9.0, 'depth': 0.25, 'shape': 'smooth'}
        lay.append(first)
        rest = [x for x in p if x is not main and (x['loop'] or x['dur'] >= 1.0)]
        if rest:
            lay.append({'sound': rest[0]['id'], 'dur': total, 'gain': -10.0})
        lay.append({'source': 'noise', 'color': 'brown', 'dur': total, 'lowpass': 220.0, 'gain': -18.0})
    elif kind == 'pass':
        total = want or 3.5
        main = p[0]
        first = {'sound': main['id']}
        if main['loop'] or main['dur'] >= 1.5:
            first['dur'] = total
        if _has('pass'):
            first['pass'] = {'at': round(total / 2, 2), 'speed': 25.0}
        else:
            first.update({'pitch': 2.0, 'glide': -3.0, 'swell': round(0.4 * total, 2), 'decay': round(0.5 * total, 2)})
        lay.append(first)
        if len(p) > 1:
            lay.append({'sound': p[1]['id'], 'gain': -10.0, **({'dur': total} if p[1]['loop'] else {})})
    elif kind == 'voice':
        shift = {-1: -9.0, 1: 5.0}.get(size, -3.0)
        first = {'sound': p[0]['id'], 'pitch': shift}
        if _has('bend'):
            first['bend'] = [[0.0, 0.0], [round(max(0.4, p[0]['dur'] * 2 ** (-shift / 12)), 2), -2.0]]
        if _has('stretch') and size < 0:
            first['stretch'] = 1.6
        lay.append(first)
        if len(p) > 1:
            lay.append({'sound': p[1]['id'], 'pitch': shift, 'gain': -9.0})
    elif kind == 'long':
        first = {'sound': p[0]['id'], 'pitch': -3.0}
        if _has('stretch'):
            first['stretch'] = 2.0
        lay.append(first)
        if len(p) > 1:
            lay.append({'sound': p[1]['id'], 'pitch': -3.0, 'gain': -10.0})
    elif kind == 'magic':
        swells = [x for x in p if x['kind'] in ('whoosh', 'rise', 'noise', 'zap')] or p
        hits = [x for x in p if x['kind'] in ('hit', 'boom', 'thud', 'impact', 'object') and x is not swells[0]]
        sw = swells[0]
        ln = min(max(sw['dur'], 0.6), 2.5)
        first = {'sound': sw['id'], 'reverse': True}
        if sw['loop'] or sw['dur'] > 2.5:
            first['dur'] = ln
        lay.append(first)
        if hits:
            lay.append({'sound': hits[0]['id'], 'at': round(max(0.0, ln - 0.03), 3), 'gain': -2.0})
        lay.append({'source': 'noise', 'color': 'pink', 'dur': round(ln + 0.6, 2), 'highpass': 2500.0,
                    'swell': round(ln * 0.8, 2), 'decay': 0.5, 'gain': -18.0})
    elif kind == 'place':
        beds = [x for x in p if x['loop'] or x['dur'] >= 3.0][:3]
        if not beds:
            return []
        total = want or BED_S
        for j, x in enumerate(beds):
            lay.append({'sound': x['id'], 'dur': total, 'gain': (0.0, -6.0, -10.0)[j]})
    return lay


def _recipe(words, kind, lay, tag=''):
    whole = {'name': _name(words) + tag, 'says': words, 'layers': lay,
             'built': f'{kind}: ' + ', '.join(x.get('sound') or x.get('source') or x.get('recipe') for x in lay)}
    if kind == 'place':
        whole['kind'] = 'ambience'
    return R.check(whole)


ALTERNATE = {'one': 'row', 'row': 'one', 'scatter': 'place', 'run': 'one', 'pass': 'one', 'voice': 'one',
             'long': 'voice', 'magic': 'one', 'place': 'scatter'}


def _cooked(b, words, without):
    """the cookbook's recipe of the thing (an answer or a sketch), when no layer of it is forbidden"""
    for sketch in (False, True):
        rc = R.answer(b, (), sketch=sketch)
        if rc and not any(_forbidden(x['sound'], without) for x in rc['layers'] if 'sound' in x):
            return R.check(copy.deepcopy(rc))
    return None


def draft(words, without=(), ears=None, n=3, near=None):
    """one to `n` first recipes of a request from its words alone (module doc), each checked by earshot.recipe; its
    `built` says the structure and the parts. ears: an Ears for the words (the atlas finds the parts), or None;
    near: the design `make` hands over for the words, which one draft builds on when it is a recording"""
    if not words or not str(words).strip():
        raise ValueError('say in words what the sound is, e.g. "hail on a tin roof"')
    without = _without(without)
    b = brief.read(words)
    out = []
    cooked = _cooked(b, words, without)
    if cooked:
        out.append({**cooked, 'says': words})
    parts = candidates(words, without, ears)
    kind = structure(words)
    by_near = []
    if near and not _forbidden(near, without):
        from .sounds import registry
        r = registry.get(near).row()
        if _real(r):
            by_near = [{'id': near, 'score': None, 'for': words, 'kind': r['kind'], 'dur': r['dur_s'] or 0.5,
                        'loop': bool(r['loop'])}] + [p for p in parts if _family(p['id']) != _family(near)][:1]
    for tag, k, pp in (('', kind, parts), ('_n', kind, by_near), ('_c', ALTERNATE[kind], parts),
                       ('_b', kind, parts[:2]), ('_d', kind, parts[:1])):
        if len(out) >= n:
            break
        lay = _layers(k, pp, b, words)
        if not lay:
            continue
        try:
            rc = _recipe(words, k, lay, tag)
        except R.RecipeError:
            continue
        if all(rc['layers'] != o['layers'] for o in out):
            out.append(rc)
    if not out:
        raise ValueError(f'no recorded part of the catalogue is near {words!r}'
                         + (f' once {", ".join(sorted(without))} are left out' if without else '')
                         + ': write the recipe yourself (`earshot recipe --example`)')
    return out


# ------------------------------------------------------------------------------------------------------------- ears
class Ears:
    """the listening model as a judge of one request: the text embedding of the words, the vocabulary that turns a
    cosine into a z, the atlas of the catalogue's sounds and the realness model. Loading it costs the model load
    (CLAP, 20-60 s on a laptop); a round of tuning then costs one embedding"""

    def __init__(self, words):
        from . import extras
        extras.need(*extras.LISTEN, what='build with ears')
        from . import vocab
        from .models import clap
        from .sounds import atlas
        self.words = words
        self.v = vocab.get()
        self.q = self.v.embed_text([words])
        self.clap = clap.get()
        try:
            self.atlas = atlas._atlas()
        except atlas.AtlasError:
            self.atlas = None
        self._zall = None

    def embed(self, y):
        import numpy as np
        y = np.asarray(y, np.float32)
        return self.clap.embed_audio([y.mean(axis=1) if y.ndim == 2 else y], pad='repeatpad')[0]

    def fit(self, e):
        """the z of a sound's embedding for the words (match.match's z)"""
        return float(self.v.calibrate(e[None], self.q)['z_q'][0, 0])

    def atlas_z(self, text):
        """{id: z} of every design for a text: half the atlas's ranking (its sound and its descriptions, the ranking
        `make` blends in), half how well its sound alone fits the text as `fit` measures it, each a z over the
        catalogue"""
        import numpy as np
        from .sounds import atlas
        if self.atlas is None:
            return None
        z = self.v.calibrate(self.atlas['vecs'], self.v.embed_text([text]))['z_q'][:, 0]
        z = (z - z.mean()) / (z.std() + 1e-9)
        mixed = atlas.zmap(text)
        return {i: 0.5 * mixed[i] + 0.5 * float(x) for i, x in zip(self.atlas['ids'], np.asarray(z))}

    def catalogue_fits(self):
        """{id: z} of every design's sound in the atlas for the words (the same z as `fit`)"""
        if self._zall is None and self.atlas is not None:
            z = self.v.calibrate(self.atlas['vecs'], self.q)['z_q'][:, 0]
            self._zall = dict(zip(self.atlas['ids'], (float(x) for x in z)))
        return self._zall or {}

    def heard_as(self, e, k=3):
        """the designs that sound most like an embedding: [(id, cosine)]"""
        import numpy as np
        if self.atlas is None:
            return []
        cos = self.atlas['vecs'] @ e
        return [(self.atlas['ids'][j], round(float(cos[j]), 3)) for j in np.argsort(-cos)[:k]]

    def rank_of(self, e, sid):
        """where a design stands among the catalogue's sounds for an embedding (1 = the nearest)"""
        import numpy as np
        if self.atlas is None or sid not in self.atlas['ids']:
            return None
        cos = self.atlas['vecs'] @ e
        return int((cos > cos[self.atlas['ids'].index(sid)]).sum()) + 1

    def words_near(self, e, k=3):
        from .match import SKIP_CATS
        return [(t, z) for t, _, z, _ in self.v.top_terms(e, k=k, exclude=SKIP_CATS)]

    def realness(self, y, e):
        """(p_real, verdict, a steady tone?) of a sound from its embedding and its signal"""
        import numpy as np
        from . import realness as RL
        m = RL.load_model()
        x = np.asarray(y, np.float32)
        x = x.mean(axis=1) if x.ndim == 2 else x
        f = RL.signal_features(RL.lift_quiet(x[:int(RL.LONG_S * 48000)])[0])
        if f is None:
            return None, 'silent', False
        lc, _ = RL.head_logit(m['heads']['clap'], e)
        ls, _ = RL.head_logit(m['heads']['signal'], RL._sig_vector(m['heads']['signal'], f))
        p, _ = RL.fuse(m['fusion'], {'clap': lc, 'signal': ls})
        tone = bool(RL.pure_tone(f)) or (f.get('tonal_frames', 0) >= TONE['frames']
                                         and f.get('line_share', 0) >= TONE['lines']
                                         and f.get('crest_db', 99) <= TONE['crest'])
        if RL.pure_tone(f):
            p = min(p, 0.03)
        return round(float(p), 2), RL.verdict_of(p, m), tone


def ears_for(words, listen=None):
    """(Ears or None, a line saying why there are none): listen=False never loads them; None and True load them
    when the listen extra is installed (True raises when it is not)"""
    from . import extras
    if listen is False:
        return None, 'not listened: tuned for a clean sound and the length only'
    gone = extras.missing(*extras.LISTEN)
    if gone:
        if listen:
            extras.need(*extras.LISTEN, what='build with ears')
        return None, (f'no ears: it was drafted, rendered, audited and measured, not heard; {LISTEN_INSTALL} '
                      'gives it ears')
    try:
        return Ears(words), None
    except Exception as e:      # noqa: BLE001  (a model that is not in the cache: build without ears)
        if listen:
            raise
        return None, f'no ears ({type(e).__name__}: {str(e)[:120]}): measured, not heard'


# -------------------------------------------------------------------------------------------------------- measuring
def measures(y):
    """what a sound is without a model: {'dur', 'hit' (s, the loudest moment), 'onset', 'active' (s within 30 dB of
    the loudest), 'bright' (Hz, the median spectral centroid of its loud frames), 'density' (onsets a second over
    its active part)}"""
    import numpy as np
    x = np.asarray(y, np.float64)
    x = x.mean(axis=1) if x.ndim == 2 else x
    sr = 48000
    hop = int(0.005 * sr)
    n = len(x) // hop
    if n < 4:
        return {'dur': round(len(x) / sr, 3), 'hit': 0.0, 'onset': 0.0, 'active': 0.0, 'bright': 0.0, 'density': 0.0}
    env = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-12
    db = 20 * np.log10(env)
    top = db.max()
    act = np.flatnonzero(db > top - 30.0)
    active = (act[-1] - act[0] + 1) * hop / sr if len(act) else 0.0
    onset = np.flatnonzero(db > top - 40.0)[0] * hop / sr
    # onsets: a rise of 6 dB within 30 ms to within 30 dB of the loudest, at least 50 ms apart
    lag = 6
    rise = db[lag:] - db[:-lag]
    cand = np.flatnonzero((rise > 6.0) & (db[lag:] > top - 30.0)) + lag
    ons = []
    for c in cand:
        if not ons or c - ons[-1] >= 10:
            ons.append(c)
    seg, h2 = 2048, 1024
    k = (len(x) - seg) // h2
    bright = 0.0
    if k > 0:
        fr = np.stack([x[j * h2:j * h2 + seg] for j in range(k)]) * np.hanning(seg)
        mag = np.abs(np.fft.rfft(fr, axis=1))
        f = np.fft.rfftfreq(seg, 1 / sr)
        e = (mag ** 2).sum(axis=1)
        loud = e > e.max() * 10 ** (-3.0)
        cen = (mag * f).sum(axis=1) / (mag.sum(axis=1) + 1e-12)
        bright = float(np.median(cen[loud])) if loud.any() else 0.0
    return {'dur': round(len(x) / sr, 3), 'hit': round(int(np.argmax(env)) * hop / sr, 3), 'onset': round(onset, 3),
            'active': round(active, 3), 'bright': round(bright), 'density': round(len(ons) / max(active, 0.25), 1)}


def _listen(y, meta, ears):
    """the numbers of one rendered sound: audit, measures and (with ears) the fit, realness and its embedding"""
    chk = meta.get('checks') or {}
    out = {'clean': bool(chk.get('ok', True)) and not chk.get('warnings'),
           'warnings': [w['msg'] for w in chk.get('warnings') or ()], 'meta': meta, **measures(y)}
    if ears is not None:
        e = ears.embed(y)
        out['e'] = e
        out['z'] = round(ears.fit(e), 2)
        out['p_real'], out['real'], out['tone'] = ears.realness(y, e)
    return out


def _render(rc, seed=SEED):
    y, meta = R.render(rc, seed=seed)
    return y, meta


def score(h, asked=None):
    """a heard sound as one number: its fit to the words (0 without ears) minus penalties: not clean (1), heard as
    synthesized (up to 2), a steady tone (1.5), far from the length asked (2 a doubling past a quarter off)"""
    s = h.get('z') or 0.0
    if not h['clean']:
        s -= 1.0
    if h.get('p_real') is not None and h['p_real'] < 0.5:
        s -= 4.0 * (0.5 - h['p_real'])
    if h.get('tone'):
        s -= 1.5
    if asked:
        s -= 2.0 * max(0.0, abs(math.log2(max(h['dur'], 0.02) / asked)) - math.log2(1.25))
    return round(s, 3)


# ------------------------------------------------------------------------------------------------------- the tuner
def numbers(rc):
    """the numbers of a recipe the tuner may move, each with its range from earshot.recipe:
    [(path, lowest, highest, how)]; path is a tuple of keys into the recipe. Layer 0's gain and start stay (the
    others are set against it); seeds, sources' kinds, curves and positions stay"""
    out = []
    for i, lay in enumerate(rc['layers']):
        base = ('layers', i)
        for k, rng in R.RANGE.items():
            if k in ('pan', 'hz', 'to') or k in R.AXES:
                continue
            if k == 'gain':
                if i > 0:
                    out.append((base + (k,), rng[0], rng[1], 'db'))
                continue
            if k in lay and isinstance(lay[k], (int, float)) and not (k == 'at' and i == 0):
                out.append((base + (k,), rng[0], rng[1], 'st' if k in ('pitch', 'glide') else 'ratio'))
        for k in ('lowpass', 'highpass'):
            if k in lay:
                if isinstance(lay[k], list):
                    out += [(base + (k, j), R.CUTOFF[0], R.CUTOFF[1], 'ratio') for j in range(2)]
                else:
                    out.append((base + (k,), R.CUTOFF[0], R.CUTOFF[1], 'ratio'))
        for field, val in lay.items():
            if not isinstance(val, dict) or field == 'follow':
                continue
            ranges = getattr(R, f'{field.upper()}_RANGE', None) or getattr(R, f'{field.upper()[:-1]}_RANGE', None)
            if not isinstance(ranges, dict):
                continue
            for sub, v in val.items():
                if sub in ranges and isinstance(v, (int, float)) and not isinstance(v, bool):
                    lo, hi = ranges[sub][0], ranges[sub][1]
                    how = 'count' if field == 'events' and sub == 'count' else (
                        'db' if sub in ('softer',) else 'step' if sub in ('loose', 'faster', 'depth', 'pan')
                        else 'st' if sub == 'pitch' and lo < 0 else 'ratio')
                    out.append((base + (field, sub), lo, hi, how))
    if isinstance(rc.get('dur'), (int, float)):
        out.append((('dur',), R.RANGE['dur'][0], R.RANGE['dur'][1], 'ratio'))
    return out


def _get(rc, path):
    v = rc
    for k in path:
        v = v[k]
    return v


def _set(rc, path, value):
    v = rc
    for k in path[:-1]:
        v = v[k]
    v[path[-1]] = value


def _moved(v, lo, hi, how, r):
    """a number moved one step of its kind, kept in [lo, hi] and rounded as a person would write it"""
    if how == 'db':
        v, nd = v + float(r.choice([-1, 1])) * float(r.uniform(2.0, 6.0)), 1
    elif how == 'st':
        v, nd = v + float(r.choice([-1, 1])) * float(r.uniform(1.0, 3.0)), 1
    elif how == 'count':
        return int(min(hi, max(lo, v + int(r.choice([-3, -2, -1, 1, 2, 3])))))
    elif how == 'step':
        v, nd = v + float(r.choice([-1, 1])) * 0.15 * (hi - lo), 2
    else:
        v = (v if v > 0 else 0.05) * 2.0 ** (float(r.choice([-1, 1])) * float(r.uniform(0.2, 0.7)))
        nd = 0 if v >= 100 else 1 if v >= 10 else 2 if v >= 0.1 else 3
    v = min(hi, max(lo, round(v, nd)))
    return int(v) if nd == 0 else v


def _said(path, rc, a, b):
    """one change in words: 'layer 1 (hail/roof) gain -8 -> -13 dB'"""
    if path[0] == 'dur':
        return f'the whole dur {a:g} -> {b:g} s'
    lay = rc['layers'][path[1]]
    what = lay.get('sound') or lay.get('recipe') or lay.get('source')
    field = '.'.join(str(p) for p in path[2:])
    unit = {'gain': ' dB', 'pitch': ' semitones', 'glide': ' semitones', 'lowpass': ' Hz', 'highpass': ' Hz'}.get(
        str(path[2]), '')
    return f'layer {path[1]} ({what}) {field} {a:g} -> {b:g}{unit}'


def tune(recipe, words, rounds=ROUNDS, seed=SEED, ears=None, asked=None, log=None, start=None):
    """the recipe with its numbers moved to fit the words better (module doc): {'recipe', 'changes' (in words),
    'score' (before, after), 'z' (before, after; None without ears), 'rounds', 'take' (y, meta, numbers) of the
    result}. Keeps the layers, their parts and every field that is not a number; the result always passes
    earshot.recipe.check. start: (y, meta, numbers) of the recipe when it was rendered and heard already"""
    import numpy as np
    rc = R.check(copy.deepcopy(R.load(recipe) if not isinstance(recipe, dict) else recipe))
    if asked is None:
        asked = brief.read(words).get('dur')
    r = np.random.default_rng(int(seed) + 7919)
    if start is None:
        y, meta = _render(rc, seed)
        start = (y, meta, _listen(y, meta, ears))
    take, h = start, start[2]
    best = first_score = score(h, asked)
    z0 = h.get('z')
    zb = z0
    nums = numbers(rc)
    for k in range(int(rounds)):
        if not nums:
            break
        trial = copy.deepcopy(rc)
        for j in r.choice(len(nums), size=min(len(nums), 1 + int(r.random() < 0.35)), replace=False):
            path, lo, hi, how = nums[int(j)]
            _set(trial, path, _moved(_get(trial, path), lo, hi, how, r))
        try:
            trial = R.check(trial)
            y, meta = _render(trial, seed)
        except R.RecipeError:
            continue
        ht = _listen(y, meta, ears)
        s = score(ht, asked)
        if log:
            log(f'round {k + 1}: {s:+.2f} (best {best:+.2f})')
        if s > best + 0.02:
            best, rc, zb, take = s, trial, ht.get('z'), (y, meta, ht)
    first = R.check(copy.deepcopy(R.load(recipe) if not isinstance(recipe, dict) else recipe))
    changes = []
    for path, *_ in numbers(first):
        try:
            a, b = _get(first, path), _get(rc, path)
        except (KeyError, IndexError):
            continue
        if a != b:
            changes.append(_said(path, rc, a, b))
    for i, lay in enumerate(rc['layers']):          # a gain the draft left at 0 and the tuner set
        if i and 'gain' in lay and 'gain' not in first['layers'][i]:
            changes.append(_said(('layers', i, 'gain'), rc, 0.0, lay['gain']))
    return {'recipe': rc, 'changes': changes, 'score': (first_score, best), 'z': (z0, zb), 'rounds': int(rounds),
            'take': take}


# ----------------------------------------------------------------------------------------------------------- hearing
def _drop(rc, i):
    """the recipe with layer i left out (the follows of the others kept pointing at their layers)"""
    out = copy.deepcopy(rc)
    del out['layers'][i]
    keep = []
    for lay in out['layers']:
        fo = lay.get('follow')
        if fo is not None:
            if fo['layer'] == i:
                lay.pop('follow')
            elif fo['layer'] > i:
                fo['layer'] -= 1
        keep.append(lay)
    out['layers'] = keep
    return out


def nearest_design(words, without=(), ears=None):
    """the design `make` would hand over for the words when `without` is missing from the catalogue: (id, its file's
    numbers as _listen gives them), rendered as make renders it"""
    import tempfile
    import soundfile as sf
    from . import make as M
    without = _without(without)
    _, cands = M.choose(words, top=60, listen=True if ears is not None and ears.atlas is not None else False)
    pick = next((c['id'] for c in cands if not _forbidden(c['id'], without)), None)
    if pick is None:
        return None, None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            res = M.make(words, out=os.path.join(tmp, 'nearest.wav'), sound=pick)
            y, _ = sf.read(res['files'][0], always_2d=True)
    except (ValueError, OSError):
        return pick, None
    return pick, _listen(y, res['takes'][0], ears)


def _verdict(h, near):
    if h.get('z') is None:
        return None
    zn = near.get('z') if near else None
    ok = h['clean'] and h['z'] >= GOOD_Z and (zn is None or h['z'] - zn >= OVER)
    return 'sounds like it' if ok else 'a sketch: not clearly it yet'


def _loudest_layer(rc):
    return max(range(len(rc['layers'])), key=lambda i: rc['layers'][i].get('gain', 0.0))


def _brighter(rc):
    """what to change in a recipe that sounds too dull: a low cutoff first, then a part moved far down, then a dark
    layer"""
    lays = rc['layers']
    for i, lay in enumerate(lays):
        lp = lay.get('lowpass')
        if lp is not None and (min(lp) if isinstance(lp, list) else lp) < 6000:
            return f"raise layer {i}'s `lowpass` (now {lp})"
    low = [i for i, lay in enumerate(lays) if min(lay.get('pitch', 0.0), lay.get('glide', 0.0)) <= -5]
    if low:
        i = _loudest_layer({'layers': [lays[k] for k in low]})
        i = low[i]
        return (f"layer {i} is moved {-lays[i].get('pitch', 0.0):g} semitones down, which darkens it: raise its "
                "`pitch` by a few semitones")
    return 'lower the `gain` of the darkest layer or add a brighter part'


def hear(recipe, words=None, without=(), seed=SEED, ears=None, near=None, layers=True, listen=None, take=None):
    """a recipe (a dict, JSON text, a file or a cookbook name) or a sound file, rendered and heard: {'summary' (plain
    lines, each naming the field of the recipe to change), 'clean', 'warnings', 'dur', 'hit', 'bright', 'density',
    and with ears 'z', 'nearest' {'id', 'z'}, 'verdict', 'heard_as', 'words_near', 'p_real', 'real', 'tone',
    'layers' [{'layer', 'part', 'z_without'}]}. near: (id, numbers) of the design make would hand over, when known;
    take: (y, meta) of the recipe when it was rendered already"""
    import numpy as np
    rc = None
    if take is not None:
        rc = R.load(recipe)
        y, meta = take[:2]
    elif isinstance(recipe, str) and os.path.isfile(recipe) and not recipe.lower().endswith('.json'):
        import soundfile as sf
        y, _ = sf.read(recipe, always_2d=True)
        y = np.asarray(y, np.float64)
        from .sounds import registry
        from types import SimpleNamespace
        meta = {'checks': registry.audit(y, SimpleNamespace(kind='object', pitch=None, loop=False, allows=()))}
    else:
        rc = R.load(recipe)
        y, meta = _render(rc, seed)
    words = words or (rc or {}).get('says') or ''
    why_not = None
    if ears is None and words and listen is not False:
        ears, why_not = ears_for(words, listen)
    elif ears is None:
        why_not = 'not listened: measured, not heard' if words else 'no words: measured, not heard'
    if take is not None and len(take) > 2 and (ears is None or 'e' in take[2]):
        h = take[2]
    else:
        h = _listen(y, meta, ears)
    if near is None and words:
        near = nearest_design(words, without, ears)
    nid, nh = near if near else (None, None)
    lines = []
    if h['clean']:
        lines.append('clean')
    else:
        fix = []
        txt = ' '.join(h['warnings']).lower()
        if 'clip' in txt or 'peak' in txt:
            fix.append('lower the loudest layer\'s `gain`')
        if 'tail' in txt or 'cut' in txt or 'truncat' in txt:
            fix.append('give the last layer a `decay` or `fade_out`, or a longer `dur`')
        if 'silen' in txt:
            fix.append('move a late layer\'s `at` earlier or shorten the whole `dur`')
        if 'click' in txt:
            fix.append('give the layer that starts there a `fade_in` of 0.005 s')
        lines.append('audit: ' + '; '.join(h['warnings'])[:200] + (f' ({"; ".join(fix)})' if fix else ''))
    asked = brief.read(words).get('dur') if words else None
    shape = f"{h['dur']:.2f} s, hit at {h['hit']:.2f} s, {h['bright'] / 1000:.1f} kHz bright, {h['density']:g} onsets/s"
    if nh:
        shape += (f"; the nearest real design {nid}: {nh['dur']:.2f} s, hit at {nh['hit']:.2f} s, "
                  f"{nh['bright'] / 1000:.1f} kHz, {nh['density']:g} onsets/s")
    lines.append('measured: ' + shape)
    off = []
    if asked and abs(math.log2(max(h['dur'], 0.02) / asked)) > math.log2(1.25):
        off.append(f"it is {h['dur']:.2f} s, the words ask {asked:g} s: set the whole's `dur` to {asked:g}"
                   + (' and make a layer longer (`dur`, `events`, `stretch`)' if h['dur'] < asked else ''))
    if rc is not None and nh:
        if nh['bright'] > 0 and h['bright'] > 1.7 * nh['bright']:
            i = _loudest_layer(rc)
            off.append(f"brighter than {nid} ({h['bright'] / 1000:.1f} against {nh['bright'] / 1000:.1f} kHz): give "
                       f"layer {i} a `lowpass` near {int(round(nh['bright'] * 3, -2))} Hz")
        elif h['bright'] > 0 and nh['bright'] > 1.7 * h['bright']:
            off.append(f"duller than {nid} ({h['bright'] / 1000:.1f} against {nh['bright'] / 1000:.1f} kHz): "
                       + _brighter(rc))
        if nh['density'] >= 4 and h['density'] < 0.5 * nh['density']:
            off.append(f"sparser than {nid} ({h['density']:g} against {nh['density']:g} onsets a second): raise a "
                       "scattered layer's `scatter.rate` or `events.count`")
        elif h['density'] >= 4 and nh['density'] < 0.4 * h['density'] and structure(words) in ('one', 'voice'):
            off.append(f"busier than {nid} ({h['density']:g} against {nh['density']:g} onsets a second): fewer "
                       "`events`, or a lower `scatter.rate`")
    res = {k: h[k] for k in ('clean', 'warnings', 'dur', 'hit', 'onset', 'active', 'bright', 'density')}
    res['nearest'] = {'id': nid} if nid else None
    if ears is not None:
        res.update(z=h['z'], p_real=h['p_real'], real=h['real'], tone=h['tone'])
        res['heard_as'] = [i for i, _ in ears.heard_as(h['e'])]
        res['words_near'] = [{'term': t, 'z': z} for t, z in ears.words_near(h['e'])]
        if nh:
            res['nearest']['z'] = nh['z']
        res['verdict'] = _verdict(h, nh)
        fits = ears.catalogue_fits()
        allowed = {i: z for i, z in fits.items() if not _forbidden(i, _without(without))}
        best = max(allowed.items(), key=lambda kv: kv[1]) if allowed else None
        lines.append('heard as: ' + ', '.join(res['heard_as']) + '; words: '
                     + ', '.join(f"'{t}' (z {z:+.1f})" for t, z in ears.words_near(h['e'])))
        lines.append(f"fit: z {h['z']:+.2f} for {words!r}" + (f" (the nearest design {nid}: z {nh['z']:+.2f})" if nh
                                                              else '') + f": {res['verdict']}")
        lines.append(f"recorded: p_real {h['p_real']} ({h['real']})" if h['p_real'] is not None else 'silent')
        if nh and nh.get('z') is not None and nh['z'] > h['z']:
            off.append(f"the nearest design {nid} fits the words better (z {nh['z']:+.2f}): use it, or build on it "
                       "as layer 0 (`sound`)")
        elif best and best[1] > h['z'] + 0.5 and best[0] != nid:
            off.append(f"the catalogue's {best[0]} sounds more like the words (z {best[1]:+.1f} by the atlas): try it "
                       "as a layer (`sound`)")
        if h['tone']:
            off.append('a steady tone is heard as a note: lower the pitched layer\'s `gain`, or `cut` the ringing part')
        if h['p_real'] is not None and h['p_real'] < 0.5:
            srcs = [i for i, x in enumerate(rc['layers']) if 'source' in x] if rc else []
            off.append('heard as synthesized' + (f": lower the `gain` of the noise or sine layers ({srcs})" if srcs
                                                 else ': move the parts less far in `pitch`, or use fewer `stretch`'))
        res['layers'] = []
        if rc is not None and layers and len(rc['layers']) > 1:
            for i, lay in enumerate(rc['layers']):
                try:
                    yo, mo = _render(_drop(rc, i), seed)
                except R.RecipeError:
                    continue
                zo = round(ears.fit(ears.embed(yo)), 2)
                part = lay.get('sound') or lay.get('recipe') or lay.get('source')
                res['layers'].append({'layer': i, 'part': part, 'z_without': zo})
                if zo > h['z'] + 0.3:
                    off.append(f"layer {i} ({part}) covers the rest: the fit rises to z {zo:+.2f} without it; lower "
                               "its `gain` by about 6 dB or leave it out")
                elif zo < h['z'] - 0.8:
                    lines.append(f"layer {i} ({part}) carries it: z {zo:+.2f} without it")
    elif why_not:
        lines.append(why_not)
    res['off'] = off
    res['summary'] = '\n'.join(lines + ([f'still off: {x}' for x in off]
                                        or ['still off: nothing the measures find; a person should listen once']))
    return res


# ------------------------------------------------------------------------------------------------------------- build
def build(words, out=None, recipe=None, rounds=None, without=(), seed=SEED, listen=None, drafts=3, layers=True):
    """draft, hear, tune and hear again (module doc); writes the sound to `out` (default <name>.wav) and the recipe
    beside it (<out without extension>.recipe.json). recipe: start from this recipe instead of drafting (tuned only
    when `rounds` is given). layers=False skips hearing each layer left out (a render each). Returns {'summary',
    'sound', 'files', 'recipe', 'recipe_file', 'heard', 'tuned', 'drafts', 'seconds'}"""
    from . import make as M
    from .synth import master
    t0 = time.time()
    if not words or not str(words).strip():
        raise ValueError('say in words what the sound is, e.g. "hail on a tin roof"')
    words = str(words).strip()
    without = _without(without)
    ears, why_not = ears_for(words, listen)
    asked = brief.read(words).get('dur')
    near = nearest_design(words, without, ears)
    tried, best = [], None
    if recipe is not None:
        rc = R.load(recipe)
        y, meta = _render(rc, seed)
        h = _listen(y, meta, ears)
        tried.append({'name': rc['name'], 'built': rc.get('built'), 'score': score(h, asked), 'z': h.get('z')})
        best = (0.0, rc, (y, meta, h))
        rounds = 0 if rounds is None else int(rounds)
    else:
        rounds = ROUNDS if rounds is None else int(rounds)
        for d in draft(words, without, ears, n=drafts, near=near[0] if near else None):
            y, meta = _render(d, seed)
            h = _listen(y, meta, ears)
            s = score(h, asked)
            tried.append({'name': d['name'], 'built': d.get('built'), 'score': s, 'z': h.get('z')})
            if best is None or s > best[0]:
                best = (s, d, (y, meta, h))
    rc, take = best[1], best[2]
    tuned = tune(rc, words, rounds, seed, ears, asked, start=take) if rounds else None
    if tuned:
        rc, take = tuned['recipe'], tuned['take']
    rc = {**rc, 'says': words}
    heard = hear(rc, words, without, seed, ears, near, layers=layers, listen=False if ears is None else None,
                 take=take)
    path = os.path.abspath(out or f'{rc["name"]}.wav')
    rpath = os.path.splitext(path)[0] + '.recipe.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    master.write(path, take[0])
    saved = {k: v for k, v in rc.items() if k in R.TOP}
    with open(rpath, 'w', encoding='utf-8') as fh:
        json.dump(saved, fh, indent=1)
    first = {**take[1], 'file': path}
    lines = [M._line(first, path, 'built')]
    lines.append('built from: ' + '; '.join(first['parts']))
    if heard.get('heard_as'):
        lines.append('heard as: ' + ', '.join(heard['heard_as']) + '; words: '
                     + ', '.join(f"'{w['term']}' (z {w['z']:+.1f})" for w in heard['words_near']))
        n = heard.get('nearest') or {}
        lines.append(f"fit: z {heard['z']:+.2f} for {words!r}"
                     + (f" (the nearest design {n['id']}: z {n['z']:+.2f})" if n.get('z') is not None else '')
                     + f": {heard['verdict']}; recorded: p_real {heard['p_real']}")
    else:
        lines.append(why_not or 'not heard')
        lines.append([x for x in heard['summary'].splitlines() if x.startswith('measured:')][0])
    if tuned and tuned['changes']:
        za, zb = tuned['z']
        lines.append(f"tuned in {tuned['rounds']} rounds: " + '; '.join(tuned['changes'][:4])
                     + (' ...' if len(tuned['changes']) > 4 else '')
                     + (f'; fit z {za:+.2f} -> {zb:+.2f}' if za is not None else ''))
    elif tuned:
        lines.append(f"tuned in {tuned['rounds']} rounds: no change scored better")
    if not heard['clean']:
        lines.append([x for x in heard['summary'].splitlines() if x.startswith('audit:')][0])
    lines += [f'still off: {x}' for x in heard['off']] or ['still off: nothing the measures find; a person should '
                                                            'listen once']
    lines.append(f'next: edit {os.path.basename(rpath)}, then: earshot build "{words}" --recipe '
                 f'{os.path.basename(rpath)}')
    return {'summary': '\n'.join(lines), 'sound': first['id'], 'files': [path], 'recipe': saved,
            'recipe_file': rpath, 'heard': heard, 'tuned': None if not tuned else {
                'changes': tuned['changes'], 'score': tuned['score'], 'z': tuned['z'], 'rounds': tuned['rounds']},
            'drafts': tried, 'takes': [first], 'seconds': round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------------ command line
def add_cli(sub):
    p = sub.add_parser('build', help='build a sound the catalogue lacks from its recorded parts: draft, render, hear, '
                                     'tune (writes the sound and its recipe)')
    p.add_argument('want', help='the sound in words, e.g. "hail on a tin roof"')
    p.add_argument('-o', '--output', help='the file to write (wav, flac, ogg, mp3, m4a, webm); the recipe is written '
                                          'beside it as <name>.recipe.json')
    p.add_argument('--recipe', help='your own recipe (a JSON file, - for standard input, or a cookbook name): rendered '
                                    'and heard instead of drafted; tuned only with --rounds')
    p.add_argument('--rounds', type=int, help=f'tuning rounds, a render and a hearing each (default {ROUNDS}; 0: no '
                                              'tuning)')
    p.add_argument('--without', nargs='+', default=(), help='designs (an id) or families (a name) the build must not '
                                                            'use')
    p.add_argument('--seed', type=int, default=SEED, help='the seed of the takes and of the tuning search')
    p.add_argument('--no-listen', action='store_true', help='do not load the listening model: draft, render, audit '
                                                            'and measure only')
    from .cli import common
    common(p)
    p.set_defaults(func=_run)


def _run(args):
    from .report import emit
    res = build(args.want, args.output, recipe=args.recipe, rounds=args.rounds, without=args.without,
                seed=args.seed, listen=False if args.no_listen else None)
    emit(res, args, text=res['summary'])


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
