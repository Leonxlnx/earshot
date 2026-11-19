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
PLAYED = {'synth', 'instrument', 'tonal', 'chip', '8-bit', 'arcade', 'retro', 'glitch', 'drums', 'kit', 'music'}
PLAYED_KINDS = {'note', 'chime', 'figure', 'music', 'pad', 'stinger'}
PLAYED_FAMILIES = {'braam', 'brass', 'strings', 'piano', 'guitar', 'organ', 'choir'}     # played, whatever the tags say
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
               'chattering bubbling bubbles shower confetti hooves herd gravel pebbles coins clatter clattering flock '
               'pack pile shoal hundreds thousands',
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
# many of one thing: a collective word and the rate a second its single events come at; the thing is what follows
# "of" ("a stampede of horses": horses), or the single event of the word itself (applause: a clap)
COLLECTIVE = {'stampede': 30.0, 'herd': 12.0, 'flock': 8.0, 'swarm': 30.0, 'crowd': 20.0, 'shower': 25.0,
              'hail': 40.0, 'pile': 12.0, 'pack': 1.5, 'shoal': 15.0, 'cloud': 30.0, 'rain': 30.0, 'applause': 40.0,
              'hundreds': 40.0, 'thousands': 80.0, 'dozens': 15.0}
SINGLE = {'stampede': 'hooves galloping', 'herd': 'hooves', 'applause': 'a clap', 'swarm': 'a buzz',
          'hail': 'a hailstone'}
OF = re.compile(r'\b(?:a|an|the)?\s*(' + '|'.join(COLLECTIVE) + r')\s+of\s+([a-z][a-z\s-]*)')
SAID = re.compile(r'\b(' + '|'.join(COLLECTIVE) + r')\b')
# words between the collective's thing and what the request goes on to say ("a pile of plates falling over")
THEN = re.compile(r'\s+(?:falling|falls|fall|running|flying|over|on|in|at|across|into|onto|through|with|and)\b.*$')


def collective(words):
    """(the single thing, its rate a second) when the words ask for many of one thing, else None: "a stampede of
    horses" -> ("horses", 30), "applause" -> ("a clap", 40)"""
    low = (words or '').lower()
    m = OF.search(low)
    if m:
        thing = THEN.sub('', m.group(2)).strip()
        if m.group(1) in SINGLE:
            thing = f'{thing} {SINGLE[m.group(1)]}'
        return thing or SINGLE.get(m.group(1), m.group(1)), COLLECTIVE[m.group(1)]
    m = SAID.search(low)
    if m and m.group(1) in SINGLE:
        return SINGLE[m.group(1)], COLLECTIVE[m.group(1)]
    return None
# a steady tone heard as a note (earshot.realness.signal_features): nearly every frame one line, most of the energy in
# a few lines, a level that hardly moves; a creature's call drifts and breathes and stays under it
TONE = {'frames': 0.9, 'lines': 0.75, 'crest': 12.0}
# the second ear (the AudioSet tagger) in hear's verdict: its top labels, those shown, the labels that are clearly
# another kind of sound than a thing of the world, and how sure it must be of one to take a verdict back
SECOND_K = 10
SHOWN = 4
OTHER = {'Music', 'Musical instrument', 'Speech', 'Synthesizer', 'Sine wave', 'Beep, bleep', 'Silence',
         'Electronic music', 'Ringtone', 'Narration, monologue', 'Singing', 'Male speech, man speaking',
         'Female speech, woman speaking', 'Song', 'Theme music', 'Electronic tuner', 'Dial tone', 'Busy signal',
         'Piano', 'Guitar', 'Organ', 'Electronic organ', 'Keyboard (musical)', 'Bell', 'Chime', 'Tuning fork'}
SPOKEN = {'Speech', 'Narration, monologue', 'Male speech, man speaking', 'Female speech, woman speaking', 'Singing'}
VOICED = {'voice', 'speak', 'speaking', 'talk', 'talking', 'whisper', 'whispering', 'sing', 'singing', 'man', 'woman',
          'child', 'children', 'people', 'person', 'crowd', 'chatter', 'chattering', 'shout', 'shouting', 'someone'}
OTHER_P = 0.3
# what a thing does, which names a tagger class of its own that is not the thing ("running": Run, "singing": Singing)
ACTS = {'running', 'opening', 'singing', 'moving', 'playing', 'working', 'falling', 'rolling', 'climbing', 'apart'}
# words too general to tie a tagger class or a design to the thing
GENERIC = {'sound', 'sounds', 'noise', 'effect', 'small', 'large', 'big', 'giant', 'low', 'high', 'old', 'new',
           'soft', 'loud', 'long', 'short', 'and', 'the', 'over', 'from', 'with', 'into', 'onto', 'moving', 'making'}
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
            and row['kind'] not in PLAYED_KINDS and _family(row['id']) not in PLAYED_FAMILIES)


def _without(without):
    if not without:
        return frozenset()
    if isinstance(without, str):
        without = re.split(r'[,\s]+', without)
    return frozenset(w.strip().strip('/') for w in without if w and w.strip())


def _content(text):
    return [w for w in brief.WORD.findall((text or '').lower()) if w not in brief.L.STOP and len(w) > 2]


def _phrases(b, words):
    """the parts of a request to find material for, each with its weight: the whole, the single thing of a
    collective ("a stampede of horses": horses), the thing it names, what it happens on, and each content word on
    its own; a word that only qualifies the next one ("fire" in "a fire alarm") counts less than the head"""
    out = [(words, 1.0)]
    many = collective(words)
    if many:
        out.append((many[0], 1.0))
    sur = b.get('surface') or {}
    if sur.get('of'):
        out.append((sur['of'], 0.8))
    if sur.get('words'):
        out.append((' '.join(sur['words']), 0.8))
    if b.get('absent'):
        out.append((b['absent'], 0.8))
    toks = brief.WORD.findall((b.get('text') or words).lower())
    content = set(_content(b.get('text') or words))
    for j, w in enumerate(toks):
        if w in content:
            # a compound of two nouns ("fire alarm"): the second is the thing; a noun and what it does ("lion
            # growling") stay equal
            modifier = j + 1 < len(toks) and toks[j + 1] in content and not toks[j + 1].endswith('ing')
            head = not modifier and j > 0 and toks[j - 1] in content and not many and not w.endswith('ing')
            out.append((w, 0.6 if modifier else 0.9 if head else 0.8))
    seen, res = set(), []
    for p, wt in out:
        if p and p.strip() and p not in seen:
            seen.add(p)
            res.append((p, wt))
    return res


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
    for ph, weight in phrases:
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
    if collective(words) and not said & set(WORDS['magic'].split()):
        return 'scatter'
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
        many = collective(words)
        rate = many[1] if many else 14.0
        if many:                    # the single thing of the collective leads: a stampede is hooves before a bed
            p = sorted(p, key=lambda x: x['for'] != many[0])
            shorts = [x for x in p if not x['loop'] and x['dur'] <= 0.8]
            longs = [x for x in p if x['loop'] or x['dur'] >= 1.5]
        grain = shorts[0] if shorts else None
        if grain is not None and many and grain['for'] != many[0] and p[0]['for'] == many[0] \
                and not p[0]['loop'] and p[0]['dur'] <= 3.0:
            grain = None            # the thing itself, even a longer take, before a short part of something else
        if grain is None:           # no short part: the shortest single take, scattered less densely as it is long
            single = sorted((x for x in p if not x['loop'] and x['dur'] <= 3.0), key=lambda x: x['dur'])
            if single:
                grain = single[0]
                rate = max(0.5, round(rate / max(1.0, 2.0 * grain['dur']), 1))
        bed = next((x for x in longs if x is not grain), None)
        if grain and _has('scatter'):
            lay.append({'sound': grain['id'], 'scatter': {'rate': rate, 'dur': total, 'pitch': 2.0, 'gain': 9.0,
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
    loops = {x['id'] for x in parts if x['loop']}
    for x in lay:                   # a loop with no length is drafted as long as a bed, not its own 20 s
        if x.get('sound') in loops and 'dur' not in x and 'scatter' not in x:
            x['dur'] = want or BED_S
    return lay


# the field that makes a structure what it is: a draft is only called by its structure when a layer has that field
KIND_FIELD = {'row': 'events', 'scatter': 'scatter', 'run': 'spin', 'pass': 'pass', 'voice': 'bend', 'long': 'stretch',
              'magic': 'reverse'}


def _built(kind, lay):
    """the recipe's `built`: its structure (only when a layer has the structure's field) and what each layer is"""
    f = KIND_FIELD.get(kind)
    if f and not any(f in x for x in lay):
        kind = 'row' if any('events' in x for x in lay) else 'layers' if len(lay) > 1 else 'one'
    return f'{kind}: ' + '; '.join(R.parts({'layers': lay}))


def _recipe(words, kind, lay, tag=''):
    whole = {'name': _name(words) + tag, 'says': words, 'layers': lay, 'built': _built(kind, lay)}
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
    if near is None:
        near = plain_pick(words, without, ears)
    out = []
    cooked = _cooked(b, words, without)
    if cooked:
        out.append({**cooked, 'says': words})
    parts = candidates(words, without, ears)
    if not parts and not out:
        near = None                 # nothing of the catalogue is near the words: the plain pick is only a guess
    kind = structure(words)
    by_near = []
    if near and not _forbidden(near, without):
        from .sounds import registry
        r = registry.get(near).row()
        if _real(r):
            by_near = [{'id': near, 'score': None, 'for': words, 'kind': r['kind'], 'dur': r['dur_s'] or 0.5,
                        'loop': bool(r['loop'])}] + [p for p in parts if _family(p['id']) != _family(near)][:1]
    plain = None
    if near and not _forbidden(near, without):
        plain = R.check({'name': _name(words) + '_plain', 'says': words, 'layers': [{'sound': near}],
                         'built': f'plain: {near} as make hands it over'})
    for tag, k, pp in (('', kind, parts), ('_n', kind, by_near), ('_c', ALTERNATE[kind], parts),
                       ('_b', kind, parts[:2]), ('_d', kind, parts[:1])):
        if len(out) >= n - (plain is not None):
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
    if out and plain is not None and all(plain['layers'] != o['layers'] for o in out):
        out.append(plain)           # a build never ends below the design make would hand over
    if not out:
        raise ValueError(f'no recorded part of the catalogue is near {words!r}'
                         + (f' once {", ".join(sorted(without))} are left out' if without else '')
                         + ': write the recipe yourself (`earshot recipe --example`)')
    return out


# ----------------------------------------------------------------------------------------------------- parts in words
# The phrase grammar of a part ("hundreds of hoofbeats on dirt, for 6 s, getting louder"): each entry is a pattern
# and what it sets; what the patterns leave is the thing, which earshot's reader and the atlas find a recording of.
# docs/BUILD.md has the same table for people.
_N = r'(\d+(?:\.\d+)?)'
_S = r'\s*(?:s|sec|secs|seconds?)\b'
_COUNTS = {'twice': 2, 'thrice': 3, 'a couple of': 2, 'a few': 3, 'several': 4, 'once': 1, 'a single': 1}
_TIMES = r'(\d+|' + '|'.join(w for w in brief.NUMBER_WORDS if ' ' not in w) + r')\s+times\b'
GRAMMAR = [
    ('sound', re.compile(r'\bsound\s*=\s*([a-z0-9_]+/[a-z0-9_]+)')),
    ('spin', re.compile(r'\b(pulsing|pulses|throbbing|chopping|chopped|fluttering|flapping fast)(?:\s+at\s+' + _N
                        + r'\s*(?:hz|(?:times\s+)?(?:a|per)\s+second))?')),
    ('dur', re.compile(r'\bfor\s+' + _N + _S)),
    ('at', re.compile(r'\b(?:at|after)\s+' + _N + _S)),
    ('start', re.compile(r'\bat the (?:start|beginning)\b|\bfirst\b')),
    ('end', re.compile(r'\bat the end\b|\blast\b')),
    ('after_hit', re.compile(r'\bafter the hit\b')),
    ('bed', re.compile(r'\b(?:under|beneath|behind|below) everything\b|\bin the background\b|\bunderneath\b')),
    ('many', re.compile(r'\b(hundreds|thousands|dozens) of\b')),
    ('many', re.compile(r'\b(?:a|an)\s+(crowd|swarm|shower|stampede|flock|hail|herd|pile|cloud|rain|shoal) of\b')),
    ('many', re.compile(r'\b(lots of|many|countless|constant|constantly|endless|steady)\b')),
    ('count', re.compile(r'\b(twice|thrice|a couple of|a few|several|once|a single)\b')),
    ('times', re.compile(r'\b' + _TIMES)),
    ('pass', re.compile(r'\b(?:passing(?: by)?|going (?:by|past)|goes (?:by|past)|(?:flying|racing|driving|rushing|'
                        r'whizzing|speeding|roaring) (?:over|past|by)|flying overhead|past)\b')),
    ('pan', re.compile(r'\bfrom (left|right) to (left|right)\b')),
    ('louder', re.compile(r'\b(?:getting|growing) (?:louder|closer)\b|\bbuilding up\b|\bcoming closer\b|'
                          r'\bapproaching\b')),
    ('fading', re.compile(r'\b(?:fading(?: out| away)?|dying (?:away|out)|getting (?:quieter|softer)|going away)\b')),
    ('swell', re.compile(r'\b(?:swelling|swells)\b')),
    ('pitch', re.compile(r'\b(much lower|much deeper|much higher|lower|deeper|higher)\b')),
    ('bend', re.compile(r'(?:^|,)\s*(rising|falling)(?:\s+in\s+pitch)?\s*(?=,|$)|\b(?:pitch|pitched)\s+(rising|falling)'
                        r'\b|\b(rising|falling)\s+in\s+pitch\b')),
    ('reverse', re.compile(r'\b(?:reversed|backwards?)\b')),
    ('far', re.compile(r'\bfar(?: away| off)?\b|\bdistant\b|\bin the distance\b')),
    ('close', re.compile(r'\bclose(?: up| by)?\b|\bnearby\b')),
    ('muffled', re.compile(r'\bmuffled\b|\bthrough a wall\b')),
    ('soft', re.compile(r'\bsoft(?:ly)?\b|\bquiet(?:ly)?\b|\bgentle\b|\bfaint\b')),
    ('loud', re.compile(r'\bloud(?:ly)?\b')),
    ('long', re.compile(r'\b(long|short)\b')),
]
MANY_RATE = {'hundreds': 40.0, 'thousands': 80.0, 'dozens': 15.0, 'lots of': 12.0, 'many': 12.0, 'countless': 30.0,
             'constant': 25.0, 'constantly': 25.0, 'endless': 25.0, 'steady': 20.0, **COLLECTIVE}
PITCH = {'much lower': -8.0, 'much deeper': -8.0, 'much higher': 7.0, 'lower': -4.0, 'deeper': -4.0, 'higher': 4.0}
FILLER = {'a', 'an', 'the', 'and', 'then', 'with', 'of', 'some', 'it', 'its', 'that', 'is', 'are', 'sound', 'sounds',
          'noise'}
BED_GAIN = -8.0
FAR = {'lowpass': 2500.0, 'gain': -8.0}


def read_part(phrase):
    """a part's phrase read by the grammar (GRAMMAR): {'thing': the words left for the reader, 'said': {what the
    grammar set}, 'used': [the grammar's words it took]}"""
    low = ' ' + str(phrase or '').lower().strip() + ' '
    said, used = {}, []
    for name, pat in GRAMMAR:
        for m in list(pat.finditer(low)):
            txt = m.group(0).strip(' ,')
            g = [x for x in m.groups() if x] if m.groups() else []
            if name == 'sound':
                said['sound'] = g[0]
            elif name == 'spin':
                said['spin'] = (float(g[1]) if len(g) > 1 else 4.0, 'chop' if 'chop' in g[0] else 'smooth')
            elif name in ('dur', 'at'):
                said[name] = float(g[0])
            elif name in ('start', 'end', 'after_hit'):
                said['at'] = {'start': 0.0, 'end': 'end', 'after_hit': 'hit'}[name]
            elif name == 'many':
                said['many'] = MANY_RATE.get(g[0], 12.0)
            elif name == 'count':
                said['count'] = _COUNTS[g[0]]
                if g[0] in ('a few', 'several'):
                    said['loose'] = 0.5
            elif name == 'times':
                said['count'] = int(g[0]) if g[0].isdigit() else brief.NUMBER_WORDS[g[0]]
            elif name == 'pan':
                a, b = (-0.8 if x == 'left' else 0.8 for x in g[:2])
                said['pan'] = [a, b]
            elif name == 'pitch':
                said['pitch'] = said.get('pitch', 0.0) + PITCH[g[0]]
            elif name == 'bend':
                said['bend'] = g[0]
            elif name == 'long':
                said['stretch'] = 1.6 if g[0] == 'long' else 0.7
            else:
                said[name] = True
            used.append(txt)
            low = low[:m.start()] + ' ' * (m.end() - m.start()) + low[m.end():]
    words = [w for w in brief.WORD.findall(low) if w not in FILLER]
    return {'thing': ' '.join(words), 'said': said, 'used': used}


_WORDS_OF = []


def _catalogue_words():
    """the stems of every word in the catalogue's ids, tags and descriptions: a word of a part that is among them
    was understood, even when the reader's lexicon lacks it"""
    if not _WORDS_OF:
        from .sounds import registry
        out = set()
        for r in registry.rows():
            text = ' '.join([r['id'].replace('/', ' ').replace('_', ' '), ' '.join(r.get('tags') or ()),
                             str(r.get('says') or '')])
            out |= {brief.stem(w) for w in brief.WORD.findall(text.lower())}
        _WORDS_OF.append(out)
    return _WORDS_OF[0]


# A part's thing is also looked up in the designs' own words (ids, tags, uses, descriptions), as `earshot sounds
# --search` reads them: a design whose own words name the thing comes before one the reader reached by a loose word.
# Words that only describe (low, big, soft) never choose a design; they add a little to one that names the thing.
DESCRIBING = {'low', 'high', 'big', 'small', 'soft', 'loud', 'deep', 'heavy', 'light', 'long', 'short', 'little',
              'tiny', 'huge', 'giant', 'strong', 'weak', 'quiet', 'gentle', 'hard', 'fast', 'slow', 'old', 'wet',
              'dry', 'wooden', 'metallic', 'close', 'far', 'distant', 'sharp', 'dull', 'bright', 'dark', 'warm',
              'cold', 'hot', 'thin', 'thick', 'full', 'rich', 'crisp', 'clean', 'sudden', 'massive', 'enormous',
              'great', 'faint', 'rough', 'smooth', 'empty', 'busy', 'lower', 'higher', 'deeper', 'bigger', 'smaller'}
# words that name a sound and not a thing: the thing named beside them counts more
SOUND_WORDS = {'rumble', 'hiss', 'hum', 'roar', 'crackle', 'whoosh', 'thud', 'buzz', 'creak', 'rattle', 'splash',
               'beat', 'bang', 'boom', 'crash', 'clank', 'clang', 'click', 'knock', 'tap', 'thump', 'whistle', 'whir',
               'drone', 'squeak', 'screech', 'chirp', 'drip', 'patter', 'pop', 'snap', 'swish', 'rustle', 'growl',
               'howl', 'chug', 'clatter', 'crunch', 'sizzle', 'whine', 'wail', 'clap', 'rumbling', 'noise', 'sound',
               'scrape', 'squeal', 'chime', 'ring', 'step'}
IRREGULAR = {'hooves': 'hoof', 'wolves': 'wolf', 'leaves': 'leaf', 'knives': 'knife', 'children': 'child',
             'people': 'person', 'mice': 'mouse', 'feet': 'foot', 'teeth': 'tooth', 'geese': 'goose',
             'shelves': 'shelf', 'wooden': 'wood', 'calves': 'calf', 'halves': 'half', 'men': 'man', 'women': 'woman'}
OWN_GENERIC = {'real', 'foley', 'nature', 'ambience', 'loop', 'background', 'sfx', 'effect'}
SETTING = {'on', 'in', 'into', 'onto', 'over', 'through', 'across', 'against', 'at', 'inside', 'along', 'from',
           'under', 'beneath', 'off', 'by', 'near'}
OWN_FIELD = {'id': 1.0, 'tags': 1.0, 'uses': 0.9, 'says': 0.6}
SET_W = 0.05                    # what the thing is on or in only breaks ties ("a car on a wet street" is a car)
_OWN = []


def _norm(w):
    w = w.lower()
    return brief.stem(IRREGULAR.get(w, w))


def _own_index():
    """{id: {stem: weight}} of every design's own words, and the set of all their stems"""
    if not _OWN:
        from .sounds import registry
        idx, vocab = {}, set()
        for r in registry.rows():
            ws = {}
            fields = {'id': r['id'].replace('/', ' ').replace('_', ' '), 'tags': ' '.join(r.get('tags') or ()),
                      'uses': ' '.join(r.get('uses') or ()), 'says': ' '.join(r.get('says') or ())}
            for f, text in fields.items():
                for w in brief.WORD.findall(text.lower()):
                    if len(w) < 3 or w in brief.L.STOP or w in OWN_GENERIC:
                        continue
                    s = _norm(w)
                    ws[s] = max(ws.get(s, 0.0), OWN_FIELD[f])
            idx[r['id']] = ws
            vocab |= set(ws)
        _OWN.append((idx, vocab))
    return _OWN[0]


def _split(w, vocab):
    """a compound the designs do not know whole ("hoofbeats") as the two words they know ("hoof", "beat"), or None"""
    best = None
    for i in range(3, len(w) - 2):
        a, b = _norm(w[:i]), _norm(w[i:])
        if a in vocab and b in vocab and (best is None or min(len(a), len(b)) > min(len(best[0]), len(best[1]))):
            best = (a, b)
    return best


def _thing_words(thing):
    """the words of a part's thing to look up in the designs' own words: [(stem, weight, said word)]. The thing
    itself counts 1; a sound word or a doing word (-ing) beside a thing 0.5; what the thing is on or in 0.2; a
    describing word 0 (it only adds to a design that names the thing)"""
    _, vocab = _own_index()
    sounds = SOUND_WORDS | {_norm(x) for x in SOUND_WORDS}
    toks = [w for w in brief.WORD.findall((thing or '').lower()) if w not in FILLER]
    out, setting = [], False
    for j, w in enumerate(toks):
        if w in SETTING:
            setting = True
            continue
        if w in brief.L.STOP or len(w) < 3:
            continue
        s = _norm(w)
        if w in DESCRIBING or s in DESCRIBING:
            out.append((s, 0.0, w))
            continue
        pieces = [s] if s in vocab else list(_split(w, vocab) or [s])
        nxt = toks[j + 1] if j + 1 < len(toks) else ''
        # a noun before another noun or a sound word names what kind ("train whistle", "car engine"): it counts
        # less than the head, but a design that has both comes first
        modifier = (len(pieces) == 1 and nxt and nxt not in SETTING and nxt not in DESCRIBING and _norm(nxt) not in
                    sounds and not nxt.endswith('ing') and nxt not in brief.L.STOP and len(nxt) > 2)
        for p in pieces:
            sound = p in sounds or (w.endswith('ing') and len(pieces) == 1)
            out.append((p, SET_W if setting else 0.5 if sound else 0.6 if modifier else 1.0, w))
    # a sound word or a doing word beside no head the designs know names the thing itself ("a low rumble",
    # "a hydraulic hiss", "an air conditioner humming")
    if not any(wt == 1.0 and s in vocab for s, wt, _ in out):
        out = [(s, 1.0 if wt == 0.5 else wt, w) for s, wt, w in out]
    return out


def own_words(thing, without=(), real=True):
    """the designs whose own words name a part's thing, best first: [(id, score, [the words it matched], whether
    it names the thing itself and not only its sound)]; real:
    recorded designs only, False: invented ones only (never played music). A design must match a word that names
    the thing or a sound; every further word it matches raises it, so a thing of two words ("train whistle")
    finds a design that has both before one that has one"""
    from .sounds import registry
    idx, _ = _own_index()
    tw = _thing_words(thing)
    if not any(wt > SET_W for _, wt, _ in tw):
        return []
    rows = {r['id']: r for r in registry.rows()}
    without = _without(without)
    out = []
    for sid, ws in idx.items():
        row = rows[sid]
        if _forbidden(sid, without) or _real(row) != real:
            continue
        if not real and (set(row.get('tags') or ()) & PLAYED or row['kind'] in PLAYED_KINDS
                         or _family(sid) in PLAYED_FAMILIES):
            continue
        hit = [(s, wt, w) for s, wt, w in tw if s in ws]
        main = sum(wt * ws[s] for s, wt, _ in hit if wt > SET_W)
        if main <= 0.0:
            continue
        both = len({s for s, wt, _ in hit if wt > SET_W}) - 1        # each further word of the thing it names
        score = (main + 0.5 * both + sum(wt * ws[s] for s, wt, _ in hit if wt == SET_W)
                 + 0.05 * sum(1 for _, wt, _ in hit if not wt))
        out.append((sid, round(score, 3), sorted({w for _, _, w in hit}), any(wt >= 0.6 for _, wt, _ in hit)))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def _weighed(thing, cands, without, k=8):
    """the reader's (or the atlas') candidates and the designs' own words weighed: [candidate rows], best first, one
    a family. A design that names the thing by its own words comes first; the reader breaks ties among them"""
    from .sounds import registry
    own = own_words(thing, without)
    if not own or own[0][1] < 0.5:
        return cands, None
    top = max([(c['score'] or 0.0) for c in cands] + [1e-9])
    voice = {c['id']: max(0.0, (c['score'] or 0.0) / top) for c in cands}
    rows = {r['id']: r for r in registry.rows()}
    scored = {sid: 3.0 * s + 0.6 * voice.get(sid, 0.0) for sid, s, _, _ in own if s >= 0.5}
    for c in cands:             # the reader's own answers stay, below every design that names the thing
        scored.setdefault(c['id'], 0.6 * voice[c['id']])
    words = {sid: w for sid, _, w, _ in own}
    out, fams = [], set()
    for sid, s in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0])):
        if _family(sid) in fams:
            continue
        fams.add(_family(sid))
        r = rows[sid]
        out.append({'id': sid, 'score': round(s, 2), 'for': ' '.join(words[sid]) if sid in words else thing,
                    'kind': r['kind'], 'dur': r['dur_s'] or 0.5, 'loop': bool(r['loop']), 'own': sid in words})
        if len(out) >= k:
            break
    return out, own


def _pick(cands, role):
    """the candidate that fits a part's role, among those near the best: a short take for a scatter or a row, a
    long one or a loop for a bed or a pass"""
    if not cands:
        return None
    top = cands[0]['score'] or 0.0
    near = [c for c in cands if (c['score'] or 0.0) >= top - max(0.6, 0.2 * abs(top))]
    fits = {'grain': lambda c: not c['loop'] and c['dur'] <= 1.2,
            'bed': lambda c: c['loop'] or c['dur'] >= 2.5,
            'pass': lambda c: c['loop'] or c['dur'] >= 1.5}.get(role)
    if fits:
        for c in near:
            if fits(c):
                return c
    return cands[0]


def ground(phrase, without=(), ears=None):
    """one part's phrase grounded in a recording: {'phrase', 'thing', 'said', 'used', 'id', 'cand' (its
    candidate row), 'why' (a line: which design and why), 'missing' (the thing the catalogue has none of, or None),
    'unread' (words neither the grammar nor the reader understood)}"""
    from .sounds import registry
    r = read_part(phrase)
    said = r['said']
    role = 'grain' if said.get('many') or (said.get('count') or 0) > 1 else 'bed' if said.get('bed') else \
        'pass' if said.get('pass') else None
    res = {'phrase': phrase, **r, 'missing': None, 'unread': []}
    if said.get('sound'):
        sid = said['sound']
        try:
            row = registry.get(sid).row()
        except Exception:       # noqa: BLE001  (any lookup failure: the id is not in the catalogue)
            raise ValueError(f'sound={sid} in the part {phrase!r} is not in the catalogue: `earshot sounds --search '
                             '<word>` lists what there is') from None
        res.update(id=row['id'], cand={'id': row['id'], 'score': None, 'for': 'named', 'kind': row['kind'],
                                       'dur': row['dur_s'] or 0.5, 'loop': bool(row['loop'])},
                   why='named by sound=')
        return res
    thing = r['thing']
    if not thing:
        raise ValueError(f'the part {phrase!r} names no thing: say what makes the sound ("a horse neighing, far") '
                         'or name a design with sound=<id>')
    b = brief.read(thing)
    known = _catalogue_words()
    res['unread'] = [w for w in b.get('unknown') or () if w not in FILLER and brief.stem(w) not in known]
    tw = _thing_words(thing)
    if tw and not any(wt > 0.0 for _, wt, _ in tw):
        raise ValueError(f'the part {phrase!r} only describes ({thing!r}): say what makes the sound ("a low '
                         'rumble", "a big drum") or name a design with sound=<id>')
    res['unread'] = [w for w in res['unread'] if not any(x == w and s in _own_index()[1] for s, _, x in tw)]
    # a thing said to pass is looked up as passing too ("a car passing": a car going by before a car's tyres)
    cands, own = _weighed(('passing ' if role == 'pass' else '') + thing, candidates(thing, without, ears, k=8),
                          without)
    how = 'atlas' if ears is not None and ears.atlas is not None else 'reader'
    made = None
    if own is None or not own[0][3]:
        # no recorded design names the thing (at most its sound): an invented design that names the thing itself
        # is taken, and said to be one
        made = own_words(thing, without, real=False)
        made = made[0] if made and made[0][1] >= 1.0 and made[0][3] else None
    if made:
        row = registry.get(made[0]).row()
        c = {'id': made[0], 'score': round(3.0 * made[1], 2), 'for': ' '.join(made[2]), 'kind': row['kind'],
             'dur': row['dur_s'] or 0.5, 'loop': bool(row['loop'])}
        res.update(id=c['id'], cand=c, others=[x['id'] for x in cands][:3],
                   why=f"(no recording of {thing!r}; {c['id']} is synthesized, its own words name {c['for']!r})")
        return res
    c = _pick(cands, role)
    if c is None:
        raise ValueError(f'no recorded part of the catalogue is near {thing!r} (part {phrase!r}): name one with '
                         'sound=<id> (`earshot sounds --search <word>`)')
    gone = b.get('absent')
    found = {s for _, _, ws, _ in (own or ()) for s in ws}
    if gone and gone not in c['id'].replace('/', ' ').replace('_', ' ') and not set(gone.split()) <= found:
        res['missing'] = gone
    res.update(id=c['id'], cand=c, others=[x['id'] for x in cands if x is not c][:3],
               why=(f"names {c['for']!r} in its own words ({how} and own words {c['score']:+.1f}" if c.get('own')
                    else f"nearest to {c['for']!r} ({how} {c['score']:+.1f}")
                   + (', a short take to repeat' if role == 'grain' else ', a long take for a bed'
                      if role == 'bed' else '') + ')')
    return res


def _heard_len(c, lay):
    """about how long a layer is heard, in s"""
    if 'scatter' in lay:
        return lay['scatter']['dur']
    own = lay.get('dur') or c['dur']
    own = own * lay.get('stretch', 1.0) * 2 ** (-lay.get('pitch', 0.0) / 12)
    if 'events' in lay:
        return (lay['events']['count'] - 1) * lay['events']['gap'] + own
    return own


def part_layers(grounded, words=''):
    """the layers of a recipe from grounded parts (ground), in their order: each part's fields from its phrase"""
    asked = brief.read(words).get('dur') if words else None
    durs = [g['said']['dur'] for g in grounded if 'dur' in g['said']]
    total = asked or (max(durs) if durs else None)
    if total is None and any(g['said'].get(k) for g in grounded for k in ('bed', 'many', 'pass')):
        total = BED_S
    lays, lens = [], []
    for j, g in enumerate(grounded):
        s, c = g['said'], g['cand']
        lay = {'sound': g['id']}
        uses = sum(1 for x in grounded[:j] if x['id'] == g['id'])
        if uses:
            lay['seed'] = uses          # the same design again: its next take
        gain = 0.0 if j == 0 else -3.0
        if 'pitch' in s:
            lay['pitch'] = max(-24.0, min(12.0, s['pitch']))
        if 'stretch' in s:
            lay['stretch'] = s['stretch']
        if s.get('reverse'):
            lay['reverse'] = True
        length = s.get('dur') or total or BED_S
        if s.get('many') and (c['loop'] or c['dur'] > 3.0):
            # a long recording of the thing is many already (a fire's crackle, a crowd): played for the length asked,
            # not scattered
            lay['dur'] = length if c['loop'] else round(min(length, 1.8 * c['dur']), 2)
            g['why'] = (g['why'].replace(', a short take to repeat', '')
                        + '; a long recording of many already, so it is played, not scattered')
        elif s.get('many'):
            rate = s['many']
            if c['dur'] > 1.0:           # a long take scattered less densely
                rate = max(0.5, round(rate / c['dur'], 1))
            lay['scatter'] = {'rate': min(400.0, rate), 'dur': length, 'pitch': 2.0, 'gain': 6.0, 'pan': 0.6}
        elif (s.get('count') or 0) > 1:
            n = min(int(s['count']), R.EVENT_RANGE['count'][1])
            gap = round(max(0.25, min(1.5, 1.1 * min(c['dur'], 1.2))), 3)
            if s.get('loose'):
                gap = round(gap * 1.5, 3)
            lay['events'] = {'count': n, 'gap': gap, 'loose': s.get('loose', 0.15)}
        elif s.get('bed') or (s.get('dur') and (c['loop'] or c['dur'] >= 1.5)) or (s.get('pass') and c['loop']):
            lay['dur'] = length if c['loop'] else round(min(length, 1.8 * c['dur']), 2)
        elif s.get('dur') and c['dur'] < s['dur'] and 'stretch' not in lay:
            lay['stretch'] = round(min(R.STRETCH[1], s['dur'] / c['dur']), 2)
        elif c['loop']:                 # a loop said with no length: as long as the sound, not the loop's own 20 s
            lay['dur'] = length
        if s.get('bed'):
            gain = BED_GAIN
        heard = _heard_len(c, lay)
        if s.get('bend'):
            up = s['bend'] == 'rising'
            lay['bend'] = [[0.0, -2.0 if up else 2.0], [round(max(0.2, heard), 2), 3.0 if up else -3.0]]
        if s.get('pass'):
            lay['pass'] = {'at': round(heard / 2, 2), 'speed': 25.0}
        if 'pan' in s:
            lay['pan'] = s['pan']
        if s.get('louder'):
            lay['level'] = [[0.0, -18.0], [round(max(0.2, heard), 2), 0.0]]
        elif s.get('fading'):
            lay['level'] = [[0.0, 0.0], [round(max(0.2, heard), 2), -24.0]]
        if s.get('swell'):
            lay['swell'] = round(max(0.1, 0.6 * heard), 2)
        if 'spin' in s:
            lay['spin'] = {'rate': max(0.5, min(60.0, s['spin'][0])), 'depth': 0.7, 'shape': s['spin'][1]}
        if s.get('far'):
            lay['lowpass'] = FAR['lowpass']
            gain += FAR['gain']
        if s.get('muffled'):
            lay['lowpass'] = 800.0
        gain += -6.0 if s.get('soft') else 3.0 if s.get('loud') else 2.0 if s.get('close') else 0.0
        if j == 0:
            g0 = gain                   # layer 0 is the reference: the others are set against it
        else:
            lay['gain'] = round(max(-40.0, min(12.0, gain - g0)), 1)
        at = s.get('at')
        if isinstance(at, (int, float)) and at:
            lay['at'] = float(at)
        elif at == 'hit':
            lay['at'] = 0.15
        lays.append(lay)
        lens.append(heard)
    whole = max([(x.get('at', 0.0) if isinstance(x.get('at'), float) else 0.0) + h for x, h in zip(lays, lens)]
                + [total or 0.0])
    for g, lay, h in zip(grounded, lays, lens):
        if g['said'].get('at') == 'end':
            lay['at'] = round(max(0.0, whole - min(h, 1.0) - 0.2), 2)
    return lays


def from_parts(words, parts, without=(), ears=None):
    """a recipe from the parts an agent names in words (module doc): (recipe, [grounded part]); each phrase read
    by the grammar (GRAMMAR, docs/BUILD.md) and its thing grounded in a recorded design"""
    if isinstance(parts, str):
        parts = [parts]
    parts = [p for p in parts or () if p and str(p).strip()]
    if not parts:
        raise ValueError('name at least one part, e.g. --part "a horse neighing, twice, far"')
    if len(parts) > R.LAYERS:
        raise ValueError(f'a recipe holds at most {R.LAYERS} layers: name at most {R.LAYERS} parts')
    without = _without(without)
    grounded = [ground(p, without, ears) for p in parts]
    lay = part_layers(grounded, words)
    rc = R.check({'name': _name(words) + '_parts', 'says': words, 'layers': lay, 'built': _built('parts', lay)})
    return rc, grounded


def part_lines(grounded, recipe=None):
    """one line a part: which design it became and why, what the phrase set; then a line for every word that
    was not understood and every thing the catalogue lacks"""
    out = []
    for j, g in enumerate(grounded):
        bits = R.parts({'layers': [recipe['layers'][j]]})[0] if recipe else g['id']
        out.append(f"part {j + 1} '{g['phrase']}': {bits}; {g['id']} {g['why']}"
                   + (f"; next nearest: {', '.join(g['others'])}" if g.get('others') else ''))
        if g.get('missing'):
            out.append(f"part {j + 1}: earshot has no {g['missing']}; {g['id']} is only the nearest it has (swap it: "
                       f"sound=<id> in the part, or edit layer {j}'s `sound`)")
        if g.get('unread'):
            out.append(f"not understood in part {j + 1}: {', '.join(g['unread'])}")
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
        self._ast = None

    def second(self, y, k=SECOND_K):
        """the second ear (the AudioSet tagger, earshot.models.ast): its top labels [(label, p)] for a sound, or None
        when the tagger cannot load. It only judges in hear's verdict; the tuner never sees it"""
        import numpy as np
        if self._ast is False:
            return None
        try:
            if self._ast is None:
                from .models import ast
                self._ast = ast.get()
            x = np.asarray(y, np.float32)
            p = self._ast.probs([x.mean(axis=1) if x.ndim == 2 else x])[0]
            return self._ast.top(p, k)
        except Exception:       # noqa: BLE001  (no tagger in the cache: the verdict rests on the fit alone)
            self._ast = False
            return None

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
        return None, ('no ears: it was drafted, rendered, audited and measured, not heard; the listen extra gives it '
                      f"ears: {extras.command('listen')}")
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


def plain_pick(words, without=(), ears=None):
    """the id of the design `make` would hand over for the words, `without` left out (None when every one is)"""
    from . import make as M
    without = _without(without)
    _, cands = M.choose(words, top=60, listen=True if ears is not None and ears.atlas is not None else False)
    return next((c['id'] for c in cands if not _forbidden(c['id'], without)), None)


def nearest_design(words, without=(), ears=None):
    """the design `make` would hand over for the words when `without` is missing from the catalogue: (id, its file's
    numbers as _listen gives them), rendered as make renders it"""
    import tempfile
    import soundfile as sf
    from . import make as M
    without = _without(without)
    pick = plain_pick(words, without, ears)
    if pick is None:
        return None, None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            res = M.make(words, out=os.path.join(tmp, 'nearest.wav'), sound=pick)
            y, _ = sf.read(res['files'][0], always_2d=True)
    except (ValueError, OSError):
        return pick, None
    return pick, _listen(y, res['takes'][0], ears)


def _stems(text):
    return {brief.stem(w) for w in _content(text) if w not in GENERIC}


PLACE = re.compile(r'\b(?:over|on|in|at|into|onto|across|through|under|near|inside)\b.*$')


def thing_classes(words, labels):
    """the tagger's classes of the thing the words name: the labels that share a word (by stem) with the words
    before where it happens ("a swarm of locusts over a field": not Field recording)"""
    want = _stems(PLACE.sub('', (words or '').lower())) - {brief.stem(w) for w in ACTS}
    out = []
    for lab in labels:
        ws = {brief.stem(w) for w in re.findall(r'[a-z]+', lab.lower()) if len(w) > 2 and w not in GENERIC}
        if ws & want:
            out.append(lab)
    return out


def second_says(words, top, labels):
    """what the second ear says of a sound for the words: (True: a class of the thing is among its top labels,
    False: it is not, or the tagger has no class for the thing and puts a clearly different sound first (music,
    speech, a synth tone) at p >= OTHER_P; None: it has no class and nothing clearly different leads), a reason)"""
    if not top:
        return None, 'no second ear'
    classes = thing_classes(words, labels)
    if classes:
        hit = [l for l, _ in top if l in classes]
        if hit:
            return True, f'it hears {hit[0]}'
        return False, f'none of its classes of the thing ({", ".join(classes[:3])}) is among its top labels'
    lab, p = top[0]
    voiced = bool(_stems(words) & {brief.stem(w) for w in VOICED})
    if p >= OTHER_P and lab in OTHER and not (voiced and lab in SPOKEN):
        return False, f'it hears {lab} first ({p:.2f})'
    return None, 'it has no class of the thing and hears nothing clearly different'


def _verdict(h, near, second=None):
    if h.get('z') is None:
        return None
    zn = near.get('z') if near else None
    ok = h['clean'] and h['z'] >= GOOD_Z and (zn is None or h['z'] - zn >= OVER)
    if ok and second is not None and second[0] is False:
        return 'a sketch: the second ear does not hear it (' + second[1] + ')'
    return 'sounds like it' if ok else 'a sketch: not clearly it yet'


def same_kind(sid, words):
    """True when a design is of the thing the words name (its id shares a word with them by stem): only such a
    design is a fair yardstick for how bright or dense the build should be"""
    if not sid:
        return False
    return bool({brief.stem(w) for w in re.findall(r'[a-z]+', sid.replace('_', ' ').replace('/', ' '))} & _stems(words))


def _loudest_layer(rc):
    return max(range(len(rc['layers'])), key=lambda i: rc['layers'][i].get('gain', 0.0))


def _brighter(rc):
    """what to change in a recipe that sounds too dull: a low cutoff first, then a part moved far down, then a dark
    layer"""
    lays = rc['layers']
    for i, lay in enumerate(lays):
        lp = lay.get('lowpass')
        if lp is not None and (min(lp) if isinstance(lp, list) else lp) < 6000:
            def up(r, i=i):
                v = r['layers'][i]['lowpass']
                r['layers'][i]['lowpass'] = [min(20000.0, 2 * x) for x in v] if isinstance(v, list) else \
                    min(20000.0, 2 * v)
            return f"raise layer {i}'s `lowpass` (now {lp})", up
    low = [i for i, lay in enumerate(lays) if min(lay.get('pitch', 0.0), lay.get('glide', 0.0)) <= -5]
    if low:
        i = _loudest_layer({'layers': [lays[k] for k in low]})
        i = low[i]

        def higher(r, i=i):
            r['layers'][i]['pitch'] = min(12.0, r['layers'][i].get('pitch', 0.0) + 3.0)
        return (f"layer {i} is moved {-lays[i].get('pitch', 0.0):g} semitones down, which darkens it: raise its "
                "`pitch` by a few semitones", higher)
    return 'lower the `gain` of the darkest layer or add a brighter part', None


def _denser(r, factor):
    """a recipe's first scattered layer (or row) made denser (factor > 1) or sparser; False when it has none"""
    for lay in r['layers']:
        if 'scatter' in lay:
            lay['scatter']['rate'] = round(max(0.5, min(400.0, lay['scatter']['rate'] * factor)), 1)
            return True
        if 'events' in lay:
            n = lay['events']['count']
            lay['events']['count'] = int(max(2, min(64, round(n * factor))))
            return True
    return False


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
    advice = []             # (remark, the change it proposes as a function of a recipe, or None)
    if rc is not None and nh and same_kind(nid, words):
        if nh['bright'] > 0 and h['bright'] > 1.7 * nh['bright']:
            i = _loudest_layer(rc)
            cut = int(round(nh['bright'] * 3, -2))
            advice.append((f"brighter than {nid} ({h['bright'] / 1000:.1f} against {nh['bright'] / 1000:.1f} kHz): "
                           f"give layer {i} a `lowpass` near {cut} Hz",
                           lambda r, i=i, cut=cut: r['layers'][i].update(lowpass=float(max(200, min(20000, cut))))))
        elif h['bright'] > 0 and nh['bright'] > 1.7 * h['bright']:
            text, change = _brighter(rc)
            advice.append((f"duller than {nid} ({h['bright'] / 1000:.1f} against {nh['bright'] / 1000:.1f} kHz): "
                           + text, change))
        if nh['density'] >= 4 and h['density'] < 0.5 * nh['density']:
            advice.append((f"sparser than {nid} ({h['density']:g} against {nh['density']:g} onsets a second): raise "
                           "a scattered layer's `scatter.rate` or `events.count`", lambda r: _denser(r, 1.6)))
        elif h['density'] >= 4 and nh['density'] < 0.4 * h['density'] and structure(words) in ('one', 'voice'):
            advice.append((f"busier than {nid} ({h['density']:g} against {nh['density']:g} onsets a second): fewer "
                           "`events`, or a lower `scatter.rate`", lambda r: _denser(r, 0.6)))
    tried = []
    for text, change in advice:
        if change is None and ears is not None:
            continue                # a remark whose change cannot be tried is not given when there are ears
        if ears is None or change is None or rc is None:
            off.append(text)
            continue
        trial = copy.deepcopy(rc)
        try:
            if change(trial) is False:      # nothing to raise: the recipe has no scattered layer or row
                off.append(text.split(':')[0] + ': the recipe has no scattered layer or row; add a part that comes '
                           'many times (`scatter`, `events`)')
                continue
            yt, _ = _render(R.check(trial), seed)
        except R.RecipeError:
            off.append(text)
            continue
        zt = round(ears.fit(ears.embed(yt)), 2)
        if zt >= h['z']:
            off.append(f'{text} (tried: fit z {h["z"]:+.2f} -> {zt:+.2f})')
        else:               # following it would lower the fit: not given as advice
            tried.append(f"{text.split(':')[0]}, but following it lowers the fit (z {h['z']:+.2f} -> {zt:+.2f}): "
                         'leave it')
    res = {k: h[k] for k in ('clean', 'warnings', 'dur', 'hit', 'onset', 'active', 'bright', 'density')}
    res['nearest'] = {'id': nid} if nid else None
    res['not_advised'] = tried
    if ears is not None:
        res.update(z=h['z'], p_real=h['p_real'], real=h['real'], tone=h['tone'])
        res['heard_as'] = [i for i, _ in ears.heard_as(h['e'])]
        res['words_near'] = [{'term': t, 'z': z} for t, z in ears.words_near(h['e'])]
        if nh:
            res['nearest']['z'] = nh['z']
        top2 = ears.second(y)
        res['second'] = None
        if top2 is not None:
            agree, why = second_says(words, top2, ears._ast.labels)
            res['second'] = {'top': [{'label': l, 'p': p} for l, p in top2[:SHOWN]], 'agrees': agree, 'why': why}
        res['verdict'] = _verdict(h, nh, None if top2 is None else (agree, why))
        fits = ears.catalogue_fits()
        allowed = {i: z for i, z in fits.items() if not _forbidden(i, _without(without))}
        best = max(allowed.items(), key=lambda kv: kv[1]) if allowed else None
        lines.append('heard as: ' + ', '.join(res['heard_as']) + '; words: '
                     + ', '.join(f"'{t}' (z {z:+.1f})" for t, z in ears.words_near(h['e']))
                     + ('; second ear: ' + ', '.join(f'{l} {p:.2f}' for l, p in top2[:SHOWN]) if top2 else
                        '; second ear: none (the AudioSet tagger did not load)'))
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
                                        or ['still off: nothing the measures find; a person should listen once'])
                               + [f'not advised: {x}' for x in tried])
    return res


# ------------------------------------------------------------------------------------------------------------- build
def build(words, out=None, recipe=None, rounds=None, without=(), seed=SEED, listen=None, drafts=3, layers=True,
          parts=None):
    """draft, hear, tune and hear again (module doc); writes the sound to `out` (default <name>.wav) and the recipe
    beside it (<out without extension>.recipe.json). recipe: start from this recipe instead of drafting (tuned only
    when `rounds` is given). parts: what the sound is made of, in phrases ("hundreds of hoofbeats on dirt, for 6 s,
    getting louder"): the recipe is built from them (from_parts) instead of drafted, then tuned. layers=False skips
    hearing each layer left out (a render each). Returns {'summary', 'sound', 'files', 'recipe', 'recipe_file',
    'heard', 'tuned', 'drafts', 'parts', 'seconds'}"""
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
    tried, best, grounded, plain = [], None, None, None
    if recipe is not None:
        rc = R.load(recipe)
        y, meta = _render(rc, seed)
        h = _listen(y, meta, ears)
        tried.append({'name': rc['name'], 'built': rc.get('built'), 'score': score(h, asked), 'z': h.get('z')})
        best = (0.0, rc, (y, meta, h))
        rounds = 0 if rounds is None else int(rounds)
    else:
        rounds = ROUNDS if rounds is None else int(rounds)
        if parts:
            rc, grounded = from_parts(words, parts, without, ears)
            todo = [rc]
            if near and near[0]:
                todo.append(R.check({'name': _name(words) + '_plain', 'says': words, 'layers': [{'sound': near[0]}],
                                     'built': f'plain: {near[0]} as make hands it over'}))
        else:
            todo = draft(words, without, ears, n=drafts, near=near[0] if near else None)
        for d in todo:
            y, meta = _render(d, seed)
            h = _listen(y, meta, ears)
            s = score(h, asked)
            tried.append({'name': d['name'], 'built': d.get('built'), 'score': s, 'z': h.get('z')})
            if d['name'].endswith('_plain'):
                plain = (s, d, (y, meta, h))
                if parts:
                    continue        # the agent's parts are built; the plain design is only the yardstick
            if best is None or s > best[0]:
                best = (s, d, (y, meta, h))
    rc, take = best[1], best[2]
    tuned = tune(rc, words, rounds, seed, ears, asked, start=take) if rounds and numbers(rc) else None
    if tuned:
        rc, take = tuned['recipe'], tuned['take']
    plain_line = None
    if plain is not None:
        final = score(take[2], asked)
        pid = plain[1]['layers'][0]['sound']
        if plain[1] is rc:
            plain_line = (f'the plain design {pid} fits best: no draft beat it, so this is {pid} as make hands it '
                          'over (name the parts with --part to build it otherwise)')
        elif plain[0] > final + 0.02 and not parts:
            rc, take, tuned = plain[1], plain[2], None
            plain_line = (f'the plain design {pid} fits best: the tuned draft scored below it, so this is {pid} as '
                          'make hands it over (name the parts with --part to build it otherwise)')
        elif plain[0] > final + 0.02:
            zp = plain[2][2].get('z')
            plain_line = (f'the plain design {pid} fits the words better than these parts (score {plain[0]:+.2f} '
                          f'against {final:+.2f}' + (f', z {zp:+.2f}' if zp is not None else '') + '): swap a part, or '
                          f'hand over `earshot make "{words}" --sound {pid}`')
    rc = {**rc, 'says': words}
    if tuned and rc.get('built') and ':' in rc['built']:       # the tuned numbers, not the draft's
        rc['built'] = rc['built'].split(':')[0] + ': ' + '; '.join(R.parts(rc))
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
    if grounded:
        same = [lay.get('sound') for lay in rc['layers']] == [g['id'] for g in grounded]
        lines += part_lines(grounded, rc if same else None)
    if heard.get('heard_as'):
        sec = heard.get('second')
        lines.append('heard as: ' + ', '.join(heard['heard_as']) + '; words: '
                     + ', '.join(f"'{w['term']}' (z {w['z']:+.1f})" for w in heard['words_near'])
                     + ('; second ear: ' + ', '.join(f"{t['label']} {t['p']:.2f}" for t in sec['top']) if sec else
                        '; second ear: none (the AudioSet tagger did not load)'))
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
    if plain_line:
        lines.append(plain_line)
    lines += [f'still off: {x}' for x in heard['off']] or ['still off: nothing the measures find; a person should '
                                                            'listen once']
    lines += [f'not advised: {x}' for x in heard.get('not_advised') or ()]
    lines.append(f'next: edit {os.path.basename(rpath)}, then: earshot build "{words}" --recipe '
                 f'{os.path.basename(rpath)}')
    return {'summary': '\n'.join(lines), 'sound': first['id'], 'files': [path], 'recipe': saved,
            'recipe_file': rpath, 'heard': heard, 'tuned': None if not tuned else {
                'changes': tuned['changes'], 'score': tuned['score'], 'z': tuned['z'], 'rounds': tuned['rounds']},
            'drafts': tried, 'parts': [{k: g[k] for k in ('phrase', 'thing', 'id', 'why', 'missing', 'unread')}
                                       for g in grounded or ()],
            'takes': [first], 'seconds': round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------------ command line
def add_cli(sub):
    p = sub.add_parser('build', help='build a sound the catalogue lacks from its recorded parts: draft, render, hear, '
                                     'tune (writes the sound and its recipe)')
    p.add_argument('want', help='the sound in words, e.g. "hail on a tin roof"')
    p.add_argument('-o', '--output', help='the file to write (wav, flac, ogg, mp3, m4a, webm); the recipe is written '
                                          'beside it as <name>.recipe.json')
    p.add_argument('--recipe', help='your own recipe (a JSON file, - for standard input, or a cookbook name): rendered '
                                    'and heard instead of drafted; tuned only with --rounds')
    p.add_argument('--part', action='append', dest='parts', metavar='PHRASE',
                   help='a part of the sound in plain words, once a part ("hundreds of hoofbeats on dirt, for 6 s, '
                        'getting louder", "a low rumble under everything", "a horse neighing, twice, far"; sound=<id> '
                        'names the design): the recipe is built from the parts instead of drafted (docs/BUILD.md '
                        'has the words it reads)')
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
                seed=args.seed, listen=False if args.no_listen else None, parts=args.parts)
    emit(res, args, text=res['summary'])


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
