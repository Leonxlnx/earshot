"""earshot make: ask for a sound in words and get it, made in code from designs and from recordings learned as
numbers.

    earshot make "a soft success sound for a finance app" -o public/success.ogg
    earshot make "coin pickup, bright" -o sfx/coin.wav --n 3            # coin_1.wav .. coin_3.wav: three takes
    earshot make "rain on a window" -o rain.ogg --dur 20                # a seamless loop
    earshot make --set ui -o public/sounds --format ogg                 # a ready-made set as one family
    earshot make --spec sounds.json -o public/sounds                    # your own set in one call (make_set)
    earshot sounds --search "notification"                              # the catalogue, compact

    from earshot.make import make, make_set
    make('a deep impact for a logo reveal with a long tail', 'hit.wav')

What happens: the request is read (earshot.brief: the job or sound, the feeling as offsets on the shared axes, a
material or voice, numbers such as length, pitch and key), the catalogue is ranked (earshot.sounds), the best design
is rendered with those controls, audited without models (clicks, cut tails, clipping, DC) and re-rolled on another
seed if the audit fails, then written in the format of the file name. No model is loaded and nothing is downloaded:
a call takes about a second. `--listen` adds earshot's ears (CLAP, 20-40 s once): the ranking also weighs how much
each design sounds like the request (earshot.sounds.atlas, blended by earshot.brief.blend), and the result says what
the written sound is heard as. When the model is already loaded in the process (the MCP server keeps it warm), the
blend is used without asking.

The result says what was chosen and why, the sounds the request named that it leaves out (`not in this sound`),
what else would fit (`also`, always up to three), and which words were not understood, so one call is usually
enough. When the pick is a guess (no job or sound word matched and nothing sounds like the request, or another kind
of sound fits almost as well), the `also` line says so and says what each alternative sounds like, to be asked for
by `sound=<id>`. Everything rendered is the caller's to use, in any product, without credit.
"""
import json
import math
import os

from . import brief
from .cli import common
from .sounds import lexicon as L
from .report import emit, read_json

SET_FIELDS = ('name', 'want', 'sound', 'pitch', 'dur', 'voice', 'material', 'seed', 'n', 'axes', 'lufs', 'events',
              'loop', 'recipe')
TRIES = 4
GUESS_Z, GUESS_HAIR = 2.0, 0.06     # `guess`: semantic support, and how close another family may come (share)
# formats whose encoder pads the start and end of a file and records it only in a note some players ignore (the LAME
# tag of an mp3, the edit list of an m4a); Vorbis (granule positions) and Opus in WebM (pre-skip and discard padding)
# carry it in the stream itself. A 384000-sample loop decodes raw to +1920 samples as mp3 and +1024 as m4a
PADDED = ('mp3', 'm4a')
SEAM_SAFE = 'flac'      # what a loop of a set is written as in place of a padded format
GIST = 40               # characters of a design's description next to an alternative of a guess
MAX_DUR = 120.0         # seconds: the longest sound one call makes (an ambience loops seamlessly: a player repeats it)
MAX_TAKES = 12          # takes of one request in one call (the most a request's words can ask for, brief.read)
MAX_EVENTS = 64         # events of one performance (earshot.synth.perform.MOST)


def _whole(v, what, lo, hi):
    """v as a whole number from lo to hi ("3" counts as 3), else a ValueError that says what is wrong"""
    span = f' from {lo} to {hi}' if hi < 2 ** 31 - 1 else ''
    try:
        if isinstance(v, bool) or isinstance(v, float) and not v.is_integer():
            raise ValueError
        i = int(v.strip()) if isinstance(v, str) else int(v)
    except (TypeError, ValueError):
        raise ValueError(f'{what} is a whole number{span}, not {v!r}') from None
    if not lo <= i <= hi:
        raise ValueError(f'{what} is a whole number from {lo} to {hi}, not {i}')
    return i


def _length(v, what='dur'):
    """a length asked for, in seconds, over 0 and at most MAX_DUR; None stays None"""
    if v is None:
        return None
    try:
        s = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{what} is a length in seconds, not {v!r}') from None
    if not s > 0:
        raise ValueError(f'{what} is a length in seconds over 0, not {v!r}')
    if s > MAX_DUR:
        raise ValueError(f'{s:g} s is longer than earshot makes in one sound (at most {MAX_DUR:g} s); an ambience or '
                         'music loop is seamless, so make it shorter and let the player repeat it')
    return s


def _paths(out, n, fmt):
    base, ext = os.path.splitext(out)
    ext = ext or f'.{fmt or "wav"}'
    return [base + ext] if n == 1 else [f'{base}_{i + 1}{ext}' for i in range(n)]


def _line(meta, path, why):
    bits = [f"{meta['dur_s']:.2f} s"]
    if meta.get('peak_s') is not None:
        bits[0] += f", peak at {meta['peak_s']:.2f} s"
    if meta.get('pitch'):
        bits.append(meta['pitch'])
    if meta.get('voice'):
        bits.append(meta['voice'])
    bits.append(f"{meta['lufs_max']:.0f} LUFS")
    if meta.get('loop'):
        bits.append('seamless loop')
    chk = meta.get('checks') or {}
    bits.append('clean' if chk.get('ok', True) else 'audit: ' + '; '.join(w['msg'] for w in chk['warnings'])[:120])
    return f"{os.path.basename(path)}: {meta['id']} ({', '.join(bits)})" + (f" [{why}]" if why else '')


def semantic(want, listen=None):
    """({id: z} of how much each design sounds like the request, or None; a note when listening was asked for and
    could not be done). listen: True = use earshot.sounds.atlas (loads CLAP once), None = only when the model is
    already loaded in this process (the MCP server keeps it warm), False = never"""
    from .sounds import atlas
    if not want or listen is False or (listen is None and not atlas.warm()):
        return None, None
    try:
        return atlas.zmap(want), None
    except atlas.AtlasError as e:
        return None, (f'not listened: {e}' if listen else None)


def choose(want, sound=None, top=4, listen=False, **over):
    """(brief, candidates) of a request: the brief with explicit arguments laid over it, and the ranked designs
    ([{'id', 'score', 'why'}, + 'lex' and 'z' when blended with how they sound: `semantic`]); `sound` names a design
    and skips the ranking. b['listened'] says whether the semantic blend was used, b['listen_note'] why not"""
    from .sounds import registry
    b = brief.read(want or '')
    for k in ('pitch', 'dur', 'voice', 'material'):
        if over.get(k) is not None:
            b[k] = over[k]
    for a, v in (over.get('axes') or {}).items():
        b['axes'][a] = max(-1.0, min(1.0, b['axes'].get(a, 0.0) + float(v)))
    if sound:
        return b, [{'id': registry.get(sound).id, 'score': None, 'why': ['named']}]
    sem, b['listen_note'] = semantic(want, listen)
    b['listened'] = sem is not None
    return b, brief.rank(b, top=top, semantic=sem)


def doubt(b, cands):
    """why the pick rests on weak evidence, or None: the request asks for a sound the catalogue has nothing of
    (brief.not_made: a dog's bark, a violin; the pick is only the nearest), or no job or sound word matched and the
    request does not sound
    like the pick (z under GUESS_Z, or not listened), or the best design of another family is within GUESS_HAIR (a
    share of the pick's score) of it: members of one family tie often and harmlessly. Thresholds chosen on the
    development request sets (660 requests, top 8 ranked): lexical ranking, 32 % of wrong first picks flagged and
    10 % of right ones; blended, 36 % and 8 %"""
    if not cands or cands[0]['score'] is None:
        return None
    top = cands[0]
    absent = brief.not_made(b, top['why'], top['id'])
    absent = absent or brief.owner_unknown(b, top['id'], top['why'])
    if absent:                          # "a dog barking", "a rocket launch": no design is one; only the nearest
        return f'earshot makes no {absent}; {top["id"]} is only the nearest sound it has'   # "a fax machine" too
    other = surface(b, top['id'])
    if other:                           # "rain on an umbrella": the rain exists, on another surface
        return other
    fam = top['id'].split('/')[0]
    # an invented twin of the pick, or the kit's cymbal under the film's, is the same thing (lexicon.TWINS,
    # FILM_CYMBAL), not another kind of sound; a learned pick
    # whose thing the request names is no guess (brief.thing_named)
    same = {fam} | {k.split('/')[0] for k, v in L.TWINS.items() if v == top['id']}    # door/knock: knock/* too
    sense = L.FAMILY_SENSE.get(fam, fam)    # rise/* and riser/* are both the riser (lexicon.FAMILY_SENSE)
    same |= {c['id'].split('/')[0] for c in cands
             if L.FAMILY_SENSE.get(c['id'].split('/')[0], c['id'].split('/')[0]) == sense}
    # designs found by nothing but the interface job the request asks for ("putting a window away": a close) are each
    # that job: which real thing plays it is no doubt about what was asked
    job = (top['why'][0] if top['why'] and list(top['why']) == list(b.get('concepts') or ())[:1]
           and top['why'][0] in L.INTERFACE_JOBS else None)
    others = [c for c in cands[1:] if c['id'].split('/')[0] not in same and L.TWINS.get(c['id']) != top['id']
              and L.FILM_CYMBAL.get(c['id'].split('/')[0]) != top['id'] and not (job and list(c['why']) == [job])]
    named = (brief.named(top['why']) or brief.thing_named(top['id'], top['why'])
             or brief.use_said(b, top['id']))      # "a bumblebee", "booing and jeering": a whole job of the recording
    heard = top.get('z') is not None and top['z'] >= GUESS_Z
    close = bool(others) and top['score'] - others[0]['score'] <= GUESS_HAIR * abs(top['score'])
    # a request in another language is read only for the English words it holds: the pick is a guess that says
    # how to ask
    # (a German request read whole through the reader's table is read as its English: nothing to say)
    english = 'the request does not read as English, write it in English'
    foreign = b.get('foreign') and not b.get('read_as')
    if (named or heard) and not close:
        return english if foreign else None
    why = 'another kind of sound fits almost as well' if close else 'no job or sound word matched' + (
        ' and it does not clearly sound like the request' if b.get('listened') else '')
    return f'{why}; {english}' if foreign else why


def surface(b, sid):
    """the plain note when the request says what its sound happens on or in (brief.read's `surface`: "rain on an
    umbrella", "footsteps in mud") and the design picked is of that sound on another surface: 'no rain on an
    umbrella: rain/tent is the nearest surface'; None when the design's own name, jobs or descriptions say a word
    of the surface, or its family is not one of a sound on several surfaces (at least two designs named for a
    surface or a ground of lexicon.SURFACE_KIN: rain, steps, landings, waves)"""
    from .sounds import registry
    sur = b.get('surface')
    if not sur or not sur.get('words') or not sur.get('of'):
        return None
    row = registry.get(sid).row()
    fam = sid.split('/')[0]
    grounds = {brief.stem(v) for vs in L.SURFACE_KIN.values() for v in vs}
    names = [{brief.stem(p) for p in r['id'].split('/')[1].split('_')} for r in registry.rows()
             if r['id'].split('/')[0] == fam]
    if sum(bool(n & grounds) for n in names) < 2:
        return None
    own = ' '.join([sid.replace('/', ' ').replace('_', ' ')] + [str(s) for k in ('says', 'uses', 'tags')
                                                                 for s in row.get(k) or ()])
    have = {brief.stem(w) for w in brief.WORD.findall(own.lower())}
    if any(brief.stem(w) in have for w in sur['words']):
        return None
    return f"no {sur['of']} {sur['prep']} {' '.join(sur.get('said') or sur['words'])}: {sid} is the nearest surface"


def alternatives(b, cands, pick, k=3, others_first=False):
    """up to k designs besides the pick: the ranked candidates first (other families first when the pick is a
    guess), then the rest of the catalogue that shares the pick's family, then its kind, then the request's domain
    (the pick's when the request names none), the nearest in usual length first; always as the pick loops or not,
    and chip sound or not"""
    from .sounds import registry
    fam = pick.split('/')[0]
    ranked = [c['id'] for c in cands if c['id'] != pick]
    if others_first:
        ranked = [i for i in ranked if i.split('/')[0] != fam] + [i for i in ranked if i.split('/')[0] == fam]
    out = list(dict.fromkeys(ranked))[:k]
    if len(out) >= k:
        return out
    d = registry.get(pick)
    doms = {b['domain']} if b.get('domain') else set(d.domains)
    rest = [r for r in registry.rows() if r['id'] != pick and r['id'] not in out and r['loop'] == d.loop
            and brief.chip(r) == brief.chip(d.row())]
    rest.sort(key=lambda r: (abs(math.log((r['dur_s'] or 1.0) / (d.dur or 1.0))), r['id']))
    for near in (lambda r: r['id'].split('/')[0] == fam, lambda r: r['kind'] == d.kind,
                 lambda r: doms & set(r['domains'])):
        out += [r['id'] for r in rest if near(r) and r['id'] not in out][:k - len(out)]
    return out


def gist(sid, width=GIST):
    """a design's first description in a few words: its first clause, cut at a word to about `width` characters"""
    from .sounds import registry
    try:
        text = registry.get(sid).says[0].split(',')[0].strip()
    except ValueError:
        return ''
    if len(text) <= width:
        return text
    return text[:width].rsplit(' ', 1)[0].rstrip(' ,;:') + ' ...'


STRONG = 0.7          # a ranked design scoring this share of the nearest one is a start of its own


def starts(b, cands, k=3):
    """k starting points besides the nearest design for a thing no design is ("a trumpet", "a fax machine"): each of
    another family; first the parts of the cookbook's sketch of the thing (brief.sketched), three kinds of sound of
    the world when no word found anything (lexicon.STARTS_ANY), the ranked designs that score near the nearest one
    (STRONG) and the design of its kind of sound (brief.read's `kind`), then the designs described most like the
    nearest one and the request (recordings and the same kind of sound first; no interface sound for a thing of the
    world), the third not of the same size as the other two when another size is near; a synthesized design or an
    instrument's note only when the nearest is one"""
    from .sounds import registry
    pick = cands[0]['id']
    d = registry.get(pick)
    seen, out = {pick.split('/')[0]}, []
    tonal = 'synth' in d.tags or pick.startswith('note/')

    def add(sid):
        f = sid.split('/')[0]
        try:
            played = 'synth' in registry.get(sid).tags or f == 'note'     # no synth or note for a thing of the world
        except ValueError:
            return
        if len(out) < k and f not in seen and sid != pick and (tonal or not played):
            seen.add(f)
            out.append(sid)

    for sid in brief.sketched(b):       # "a dishwasher": the other parts the cookbook's sketch of it is built on
        add(sid)
    if not cands[0]['why'] and not cands[0]['score'] and not b.get('kind'):
        for sid in L.STARTS_ANY:        # no word found anything ("an escalator"): three kinds of sound to try
            add(sid)
    best = cands[0]['score'] or 0.0
    for c in cands[1:]:
        if c['score'] is not None and best > 0 and c['score'] >= STRONG * best:
            add(c['id'])
    if b.get('kind') and b['kind'] != pick:
        try:
            add(registry.get(b['kind']).id)
        except ValueError:
            pass
    roles = brief._roles()
    face = 'interface' in d.tags

    def said(r):
        texts = [r['id'].replace('/', ' ').replace('_', ' ')] + [str(s) for k in ('says', 'tags', 'uses')
                                                                for s in r.get(k) or ()]
        return {brief.stem(w) for s in texts for w in brief.WORD.findall(s.lower()) if w not in L.STOP}

    want = said(d.row()) | set(brief._said_words(b.get('text')))
    rows = [r for r in registry.rows() if r['id'].split('/')[0] not in seen and r['loop'] == d.loop
            and brief.chip(r) == brief.chip(d.row()) and ('interface' in r.get('tags', ()) or False) <= face]

    def like(r):        # how much a design is described as the nearest one and the request are; a recording first
        w = said(r)
        return (len(want & w) / math.sqrt(len(w) + 1) + 0.3 * (r['kind'] == d.kind) + 0.2 * (r['id'] in roles))

    pool = sorted(rows, key=lambda r: (-like(r), r['id']))
    size = d.dur or 1.0
    sizes = []
    for r in pool:              # the most alike first; the third not of the same size as the other two
        if len(out) >= k:
            break
        if r['id'].split('/')[0] in seen:
            continue
        band = round(math.log((r['dur_s'] or 1.0) / size))
        if len(out) == k - 1 and len(sizes) >= 2 and len(set(sizes + [band])) == 1 and any(
                round(math.log((q['dur_s'] or 1.0) / size)) != band for q in pool[:40]
                if q['id'].split('/')[0] not in seen):
            continue
        sizes.append(band)
        add(r['id'])
    for c in cands[1:]:
        add(c['id'])
    return out


def guess(b, cands, also=None):
    """the alternatives line when the pick is a guess (`doubt`), else None: why it is a guess and the alternatives
    (default: the ranked candidates, other families first), each with what it sounds like in a few words (`gist`),
    so one of them can be chosen without looking it up"""
    why = doubt(b, cands)
    if why is None:
        return None
    if why.startswith('earshot makes no ') and cands:
        also = starts(b, cands)
    if also is None:
        fam = cands[0]['id'].split('/')[0]
        also = [c['id'] for c in cands[1:] if c['id'].split('/')[0] != fam]
        also = (also + [c['id'] for c in cands[1:] if c['id'] not in also])[:3]
    alts = ', '.join(f'{a} ({gist(a)})' if gist(a) else a for a in also)
    if why == surface(b, cands[0]['id']):       # the sound exists, on another surface: no guess, its other surfaces
        fam = cands[0]['id'].split('/')[0]
        same = [c['id'] for c in cands[1:] if c['id'].split('/')[0] == fam][:3]
        alts = ', '.join(f'{a} ({gist(a)})' if gist(a) else a for a in same) or alts
        return f'{why}; also: {alts}. Name one with sound=<id>'
    return f'also (the pick is a guess: {why}): {alts}. Name one with sound=<id>'


def _formats(fmt):
    """the formats of a set: 'webm,mp3', ['webm', 'mp3'] or 'ogg' -> a list, in order, each once"""
    from .synth import master
    parts = fmt if isinstance(fmt, (list, tuple)) else str(fmt).split(',')
    out = list(dict.fromkeys(str(f).strip().lstrip('.').lower() for f in parts if str(f).strip()))
    bad = [f for f in out if f not in master.FORMATS]
    if bad or not out:
        raise ValueError(f'unknown audio format {", ".join(bad) or repr(fmt)}: one or more of '
                         f'{", ".join(master.FORMATS)}, with commas (webm,mp3)')
    return out


def _unknown(b, why):
    """the words of the request that found nothing: not those the design was picked for (by their stem)"""
    return [w for w in b['unknown'] if brief.stem(w) not in why]


def _loop_formats(fmts):
    """the formats a loop of a set is written in: each padded one (mp3, m4a) replaced by SEAM_SAFE, once"""
    return list(dict.fromkeys(SEAM_SAFE if f in PADDED else f for f in fmts))


def _seam(names, fmts):
    """the line that warns about loops in a padded format (mp3, m4a), or None"""
    lossy = [f for f in fmts if f in PADDED]
    if not (names and lossy):
        return None
    return (f"{', '.join(names)}: a loop in {' / '.join(lossy)} has a gap at its seam in players that ignore the "
            "encoder's padding note; for loops use ogg, webm, wav or flac")


def _request(b, theme):
    """the render request of a brief: its numbers and axes, with the request's key laid over the theme"""
    from .sounds import theme as T
    th = T.get(theme)
    if b.get('key') or b.get('scale'):
        th = dict(th or {})
        th.update({k: b[k] for k in ('key', 'scale') if b.get(k)})
    req = {k: b[k] for k in ('pitch', 'dur', 'voice', 'material') if b.get(k) is not None}
    if b.get('loop') is not None:            # True: the request said "loop" / "seamless"; False: "once"
        req['loop'] = b['loop']
    return {**req, **b['axes'], 'theme': th}


def performance(b, sound, events=None):
    """(perform, events) of a request for a design that a person performs several times in a row (a walk, a knock,
    typing: earshot.sounds.learned.performs), else None. events: the caller's count (1 = one event, 2 or more = a
    performance of that many), else as the request says (brief.performance)"""
    from .sounds import learned
    if not learned.performs(sound):
        return rowed(b, sound, events)
    if events is not None:
        return int(events) > 1, max(1, int(events))
    return brief.performance(b, sound, learned.index()[sound].get('pattern'))


LASTING = ('bed', 'music', 'pad', 'texture', 'noise')     # kinds that last: never played as a row of events


def rowed(b, sound, events=None):
    """(True, events, pattern, thing) when a one-shot design that is not performed from recordings
    (earshot.sounds.learned.performs) is asked for as a counted thing a person does in a row ("four knocks on the
    table", "three claps", "a countdown 3 2 1": brief's `row`) or with the caller's events over 1; else None. thing:
    the counted word as one ("knock", "clap"), else None"""
    from .sounds import registry
    d = registry.get(sound)
    if d.loop or d.kind in LASTING:
        return None
    thing = (b.get('counted') or [None])[-1] if b.get('row') else None
    if events is not None:
        return (True, int(events), b.get('row') or 'row', thing) if int(events) > 1 else None
    if b.get('row') and (b.get('events') or 0) > 1:
        return True, int(b['events']), b['row'], thing
    act = brief.acted(b)              # "clapping", "beeping, 3 seconds": the action at its pace (`row`)
    if act:
        return True, None, act, L.ACTION_WORDS[b['act']]
    return None


def twin(b, sound):
    """the design that plays the action a request names with an action word (brief.acted): the recorded twin of the
    invented design picked when the twin is performed and the invented one is not (lexicon.TWINS, learned.performs;
    knock/door is three knocks of its own, so "knocking" and "knocking, 4 seconds" are door/knock, whose knocks are
    spread over the length asked), else sound"""
    from .sounds import learned
    t = L.TWINS.get(sound)
    if t and brief.acted(b) and not learned.performs(sound) and learned.performs(t):
        return t
    return sound


def row(sound, seed, events, name, req, lufs=None, peak_db=-1.0, thing=None):
    """a counted action of any one-shot design as one sound: (y, meta) as registry.render gives a take, with
    meta['performance'] = {'pattern', 'events', 'pace', 'seeds'}. The pattern player (earshot.synth.perform) places
    `events` takes of the design, each its own seed (a new variation; a learned design's next recording), each a
    little firmer or softer and early or late as the pattern does it, then the whole is levelled as one sound. A
    length asked for (req's dur) is the length of the file: the events are spread over it (earshot.synth.perform
    `fill`; events None = as many as the pattern's pace fits), the last one left the design's usual length (0.35 to
    0.6 s) to ring out before the end; where the gaps would leave what a hand can do, they are held there, the
    length follows and meta['note'] says so"""
    from types import SimpleNamespace
    from .forge import dsp
    from .sounds import registry
    from .synth import master
    from .synth import perform as P
    d = registry.get(sound)
    dur = req.get('dur')
    one = {k: v for k, v in req.items() if k != 'dur'}
    lo = None           # a row no hand plays (a tick, a blip, a counter) follows as close as its event's own length
    if dur and name in P.GAPS and name != 'count' and not P.by_hand(thing, sound):
        lo = P.own_gap(registry.render(sound, seed=seed, check=False, **one)[0])
    p = P.plan(name, dsp.rng(registry.seed_of(sound, seed) ^ 0x60E5), events=events, dur=dur,
               ring=min(max(d.dur or 0.35, 0.35), 0.6), lo=lo)
    parts, first = [], None
    for i, nudge in enumerate(p['nudge']):
        ax = {**one, 'energy': max(-1.0, min(1.0, float(one.get('energy', 0.0)) + nudge['energy']))}
        y, m = registry.render(sound, seed=seed + i, check=False, **ax)
        first = first or m
        parts.append((y, 0.0))
    # the last event rings out before the end: measured LIFT dB deeper, as finish folds the strikes and raises what
    # follows them by up to that
    rings = [P.ring_of(z, 0.0, P.RING_DB + registry.LIFT) for z, _ in parts]
    p, kept = P.close(p, rings, keep=events is not None)
    parts = parts[:kept]
    if p.get('dur') and p['pattern'] in P.GAPS:           # no later event covers the last one's own end
        parts[-1] = (P.ease(parts[-1][0]), 0.0)
        end = p['dur'] - P.FADE                           # an event still sounding there is tapered to end on it
        parts = [(P.taper(z, end - t), 0.0) if t + r > end else (z, 0.0)
                 for (z, _), t, r in zip(parts, p['onsets'], rings)]
    y = master.finish(P.mix(parts, p), peak_db=peak_db, lufs=lufs, cap=registry.LOUDEST[''], lift=registry.LIFT,
                      sparse=registry.LIFT)
    if p.get('dur'):
        y = P.fit(y, p['dur'])
    meta = {'id': sound, 'kind': d.kind, 'seed': int(seed), 'dur_s': round(len(y) / dsp.SR, 3),
            'peak_dbfs': round(master.true_peak_db(y), 2), 'lufs_max': round(master.loudness(y), 1),
            'performance': {'pattern': p['pattern'], 'events': len(p['onsets']), 'pace': p['pace'],
                            'seeds': [seed + i for i in range(len(p['onsets']))]}}
    note = P.said(p, dur, meta['dur_s'], thing)
    if note:
        meta['note'] = note
    meta.update({k: first[k] for k in ('pitch', 'voice', 'material') if first.get(k)})
    ign = [k for k in first.get('ignored', ()) if k in one and k != 'energy' or k == 'energy' and one.get('energy')]
    if ign:
        meta['ignored'] = ign
    # several attacks are the design of a row, not layers stacked on one hit, and a short dry event ends before the
    # next begins: the gap between them is silence
    meta['checks'] = registry.audit(y, SimpleNamespace(kind=d.kind, pitch=d.pitch, loop=False,
                                                        allows=d.allows + ('stacked', 'silence')))
    return y, meta


def takes_asked(b, why, perf, n=1):
    """how many takes a request asks for: its count of takes ("3 variations"), else the caller's n when not 1, else
    the number of events it names for a design that is not performed ("five coin pickups": five pickups, each its
    own take; the counted thing is one the design was picked for, so "level 2 complete" asks for one), else n"""
    if b.get('n'):
        return int(b['n'])
    ev = b.get('events') or 0
    if int(n) == 1 and ev > 1 and perf is None and set(b.get('counted', ())) & {brief.stem(w) for w in why}:
        return min(12, int(ev))
    return int(n)


def _held(sound, dur, takes):
    """the plain note for a figure of notes (earshot.sounds.figure) whose length differs from the one asked for: a
    struck note is not held longer than it rings, and a figure cannot be shorter than its notes hurried to a hand's
    spacing; None when the length was kept (within 30 ms) or none was asked for"""
    from .sounds import figure, registry
    if not dur or getattr(registry.get(sound).fn, '__module__', '') != figure.__name__:
        return None
    s = [t['dur_s'] for t in takes]
    if min(s) > dur + 0.03:
        return f'as short as this sound can be: {min(s):.2f} s (asked for {dur:g} s)'
    if max(s) < 0.75 * dur:
        return f'rings out in {max(s):.2f} s, shorter than the {dur:g} s asked for: a struck note is not held'
    return None


# an axis as a listener names what it controls
PLAIN = {'bright': 'brightness', 'hard': 'hardness', 'ring': 'ring', 'size': 'size', 'space': 'room',
         'width': 'width', 'energy': 'strength', 'motion': 'speed'}


LENGTH_WORDS = {'short', 'shorter', 'long', 'longer', 'brief', 'briefer', 'longish', 'shortish'}


def words_of(b):
    """the words of a request, lower case"""
    return set(brief.WORD.findall((b.get('text') or '').lower()))


def unfollowed(b, sound, why, ignored, req, given, takes):
    """the plain lines for what a request asked and the sound could not follow (registry.render's `ignored`), each
    with what the sound does instead. An axis is said only when the caller set it (`given`) or a word of the request
    moved it that neither the design was picked for nor its name or tags hold: "fast" in "a fast whoosh" named
    whoosh/fast, nothing was ignored"""
    from .sounds import registry
    d = registry.get(sound)
    own = [w for w in sound.replace('/', '_').split('_')] + list(d.tags)
    picked = set(why) | {w.split(':')[-1] for w in why} | set(own) | {brief.stem(w) for w in own}
    words = [w for w in brief.WORD.findall((b.get('text') or '').lower())
             if brief.stem(w) not in picked and w not in picked]
    if req.get('dur') and 'dur' not in ignored:     # "a short whoosh, 0.3 seconds": the length said it, and was kept
        words = [w for w in words if w not in LENGTH_WORDS]
    out, what, axes = [], [], []
    length = [w for w in words if w in LENGTH_WORDS and brief.modifier(w)
              and set(ignored) & set(L.MODIFIERS[brief.modifier(w)]) and not req.get('dur')]
    if length:          # "a short whoosh": the length in seconds is what sets it
        quoted = ', '.join(dict.fromkeys('"' + w + '"' for w in length))
        out.append(f'not followed: {quoted}; say the length in seconds instead ("0.3 seconds")')
        words = [w for w in words if w not in length]
    for k in [k for k in ignored if k in PLAIN]:
        said = [w for w in words if brief.modifier(w) and k in L.MODIFIERS[brief.modifier(w)]]
        if k in given or said:
            what += [f'"{w}"' for w in said] if said else [f'{k}={req.get(k, 0):+.2g}']
            axes.append(PLAIN[k])
    if axes:
        no = ', '.join(axes[:-1]) + (' or ' if len(axes) > 1 else '') + axes[-1]
        out.append(f"not followed: {', '.join(dict.fromkeys(what))}; this sound has no control of its {no} "
                   'and plays as designed')
    for k in ignored:
        if k == 'dur' and req.get('dur'):
            got = max(t['dur_s'] for t in takes)
            out.append(f"{got:.2f} s, not the {req['dur']:g} s asked: a single struck sound rings as long as it rings")
        elif k in ('pitch', 'voice', 'material') and req.get(k) is not None:
            if brief.stem(str(req[k])) in picked or str(req[k]) in picked:
                continue
            if k != 'pitch' and k not in given and str(req[k]).lower() not in words_of(b):
                continue        # read from another word ("a chair": wood), not asked for: nothing to say
            out.append(f'not followed: {k} {req[k]}; this sound has no {k} to set and plays as designed')
    import re
    text = (b.get('text') or '').lower()
    if 'dur' not in ignored and req.get('dur') and takes and not any(t.get('note') for t in takes):
        got = max(t['dur_s'] for t in takes)        # "a whoosh, 2 seconds long" made at its own 2.9 s
        if abs(got - req['dur']) > max(0.25, 0.2 * req['dur']):
            out.append(f"{got:.2f} s, not the {req['dur']:g} s asked: this sound is made at its own length")
    # a change over the length of the sound ("getting louder", "slowing down"): no design changes as it plays,
    # only a recipe built for it ("a heartbeat getting faster")
    m = re.search(r'\b(?:getting|growing|becoming|going|gets|grows|turning)\s+(?:louder|quieter|softer|faster|'
                  r'slower|closer|nearer|further|farther|bigger|smaller|higher|lower|deeper|brighter|darker)\b'
                  r'|\b(?:speeding up|slowing down|fading (?:in|out|away)|coming closer|moving away)\b', text)
    if m and not sound.startswith('recipe/'):
        out.append(f'not followed: "{m.group(0)}"; this sound stays the same from its start to its end')
    # heard through something: a door, a wall, water
    m = re.search(r'\b(?:behind|through) (?:a |the )?(?:closed |shut |thick )?(?:door|wall|window)\b'
                  r'|\b(?:from|in) the next room\b|\bunder ?water\b|\bmuffled\b', text)
    if m and not {'underwater', 'muffled'} & set(re.findall('[a-z]+', sound)):
        out.append(f'not followed: "{m.group(0)}"; the sound is heard as in the same room, not through anything')
    if b.get('pan'):            # make places no sound between the speakers: a passing sound moves as it was recorded
        out.append(f'not followed: "{b["pan"]}"; earshot does not pan a sound: it plays as designed, pan it in an '
                   'editor')
    return out


def marks(y, events=1):
    """where things happen in a take y, in s from the start of the file: {'peak_s': the loudest moment (the largest
    sample of the loudest 50 ms), 'onset_s': where the sound starts to be heard (its 5 ms level first within 40 dB
    of its loudest)}, and for a row of `events` over 1 'events_s': each event's start (the largest rises of the 5 ms
    level, at least 20 ms apart). The hit goes on the frame at t: start the file at t - peak_s"""
    import numpy as np
    from .forge import dsp
    y = np.asarray(y, np.float64)
    m = y ** 2 if y.ndim == 1 else np.mean(y ** 2, axis=1)
    if not len(m) or not m.any():
        return {'peak_s': 0.0, 'onset_s': 0.0}
    w = max(dsp.n_of(0.05), 1)
    c = np.concatenate([[0.0], np.cumsum(m)])
    i = int(np.argmax(c[w:] - c[:-w])) if len(m) > w else 0
    a = np.abs(y) if y.ndim == 1 else np.abs(y).max(axis=1)
    peak = i + int(np.argmax(a[i:i + w]))
    k = max(dsp.n_of(0.005), 1)
    e = 10 * np.log10(np.convolve(m, np.ones(k) / k, 'same') + 1e-20)
    out = {'peak_s': round(peak / dsp.SR, 3), 'onset_s': round(int(np.argmax(e > e.max() - 40.0)) / dsp.SR, 3)}
    if events and events > 1:
        h = max(dsp.n_of(0.001), 1)
        f = e[::h]
        rise = np.maximum(f[k // h * 2:] - f[:-(k // h * 2)], 0.0)      # how much the level climbs over 10 ms
        rise[f[k // h * 2:] < e.max() - 30.0] = 0.0
        apart, picks = 20, []
        for j in np.argsort(-rise, kind='stable'):
            if rise[j] <= 0 or len(picks) == events:
                break
            if all(abs(int(j) - q) >= apart for q in picks):
                picks.append(int(j))
        out['events_s'] = sorted(round(q * h / dsp.SR, 3) for q in picks)
        out['events_s'][0] = min(out['events_s'][0], out['onset_s'])
    return out


def take(sound, seed, req, lufs=None, peak_db=-1.0, perform=None):
    """one clean take of a design: the first of TRIES seeds (seed, seed + 101, ...) that passes the audit, else the
    first one with its warnings: (y, meta). perform: (True, events) plays the design as a performance
    (earshot.sounds.learned.perform: events None = its natural number, or as many as fill the length asked for);
    (True, events, pattern) plays any one-shot design as a row of that many events (`row`)"""
    from .sounds import learned, registry
    first = None
    for k in range(TRIES):
        if perform and perform[0] and len(perform) > 2:
            y, meta = row(sound, seed + 101 * k, perform[1], perform[2], req, lufs=lufs, peak_db=peak_db,
                          thing=perform[3] if len(perform) > 3 else None)
        elif perform and perform[0]:
            y, meta = learned.perform(sound, seed=seed + 101 * k, events=perform[1], lufs=lufs, peak_db=peak_db,
                                      **req)
        else:
            y, meta = registry.render(sound, seed=seed + 101 * k, lufs=lufs, peak_db=peak_db, **req)
        if meta['checks']['ok']:
            first = (y, meta)
            break
        first = first or (y, meta)
    y, meta = first
    meta.update(marks(y, (meta.get('performance') or {}).get('events') or 1))
    return y, meta


CLOSE_ENOUGH = 0.1      # s: a take within this of the length asked is the length asked: nothing to say


def _note(t, dur):
    """a take's note as it is worth saying: None for a length within CLOSE_ENOUGH of the one asked (a loop cut to
    the nearest whole cycle), and no reason that is not true (a bed can be made longer than one take of it)"""
    note = t.get('note')
    if not note or not dur:
        return note
    if abs(t['dur_s'] - dur) <= CLOSE_ENOUGH and ' asked' in note:
        return None
    return note.replace(': the longest this sound is made', '')


def _heard(path, queries):
    from . import match as M
    m = M.match(path, queries)
    return {'query': m.get('query'), 'z': round(float(m['z']), 2), 'verdict': m['verdict']}


def make(want=None, out=None, sound=None, n=1, seed=0, theme=None, listen=False, fmt=None, mono=False, lufs=None,
         sr=48000, bits=24, pitch=None, dur=None, voice=None, material=None, loop=None, events=None, recipe=None,
         **axes):
    """make the sound a request asks for and write it: {'summary', 'sound', 'files', 'why', 'also', 'not_in_sound',
    'unknown', 'takes': [meta ...]} (+ 'heard' with listen); `not_in_sound`: the sounds or jobs the request named
    that the chosen design was not picked for, as written ("water drop"), `also`: up to three alternatives. want:
    the request in words; sound: a design id instead of (or on top of) the words; out: the file to write (its
    extension picks the format: wav, flac, ogg, mp3, m4a, webm; none = `<sound>.wav` in the working folder); n
    takes get `_1 .. _n` before the extension; theme: a preset name or dict (earshot.sounds.theme); lufs: a
    momentary-max loudness instead of the -1 dBFS peak; pitch / dur / voice / material and the axes (bright, hard,
    ring, size, space, width, energy, motion) override what the words say. events: for a thing a person does
    several times in a row (footsteps, a knock, typing, coins, keys, clinking glasses), 1 = one event, 2 or more =
    the action with that many events; None = as the words say ("footsteps", "knocking", "three knocks", "5 seconds
    of typing" ask for the action, "a footstep" for one step); any other one-shot design plays a counted action done
    in a row ("four knocks on the table", "three claps", "a countdown 3 2 1") or events of 2 or more as one file of
    that many events (`row`), while a count of separate things ("five coin pickups") is that many files; a length
    asked with an action ("three knocks, 2 seconds") is the file's and the events are spread over it; a take's
    meta['performance'] says what was played. recipe: a sound built from the catalogue's parts instead of a
    request (earshot.recipe: a dict, JSON text, the path of a JSON file, "-" for standard input or the name of a
    cookbook recipe), rendered and written the same way (`from_recipe`); a request that names a thing the cookbook
    builds and no design is ("a dragon roar") is answered by that recipe, and the result says so"""
    from .sounds import registry
    from .synth import master
    if recipe is not None:
        if sound or axes or any(v is not None for v in (theme, pitch, voice, material, loop, events)):
            raise ValueError('a recipe is rendered as it is written: with recipe= make takes out, n, seed, fmt, '
                             'mono, lufs, sr, bits, dur (a length to cut it at), listen and want (what it is, in '
                             'words, for listening); change the sound in the recipe itself')
        return from_recipe(recipe, out, n=_whole(n, 'n (takes)', 1, MAX_TAKES),
                           seed=_whole(seed, 'seed', -2 ** 31, 2 ** 31 - 1), fmt=fmt, mono=mono, lufs=lufs, sr=sr,
                           bits=bits, listen=listen, want=want, dur=_length(dur))
    if not ((want and str(want).strip()) or sound):
        raise ValueError('say what sound you want (want=...) or name one (sound=...)')
    bad = [a for a in axes if a not in registry.AXES]
    if bad:
        raise ValueError(f'unknown option {", ".join(bad)}: make takes want, out, sound, n, seed, theme, listen, fmt, '
                         f'mono, lufs, sr, bits, pitch, dur, voice, material, loop, events, recipe and the axes '
                         f'{", ".join(registry.AXES)} (-1 to 1)')
    for a, v in axes.items():
        if v is not None and not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f'{a} is a number from -1 to 1, not {v!r}')
    n, seed, dur = _whole(n, 'n (takes)', 1, MAX_TAKES), _whole(seed, 'seed', -2 ** 31, 2 ** 31 - 1), _length(dur)
    events = None if events is None else _whole(events, 'events', 1, MAX_EVENTS)
    if listen:      # before anything is made: listening needs the models' packages
        from . import extras
        extras.need(*extras.LISTEN, what='make with listen')
    b, cands = choose(want, sound, top=8, listen=True if listen else None, pitch=pitch, dur=dur, voice=voice,
                      material=material, axes=axes)
    if b.get('dur') is not None:        # a length the words asked for ("rain, 10 hours") has the same bounds
        b['dur'] = _length(b['dur'], 'the length asked for')
    if loop is False:
        b['loop'] = False
    if not sound:       # a thing the cookbook builds from parts and no design of the catalogue was found by
        from . import recipe as R
        cooked = R.answer(b, found_by(cands[0]['why'], cands[0]['id']) if cands else ())
        if cooked:
            given = [k for k, v in (('theme', theme), ('pitch', pitch), ('voice', voice), ('material', material))
                     if v is not None] + [a for a, v in axes.items() if v]
            cooked, level, asked = asked_of_recipe(cooked, b, events, seed)
            return from_recipe(cooked, out, n=n, seed=seed, fmt=fmt, mono=mono, lufs=level if lufs is None else lufs,
                               sr=sr, bits=bits, listen=listen, want=want, dur=b.get('dur'), cands=cands, given=given,
                               asked=asked)
    if not cands:
        absent = brief.not_made(b)
        raise ValueError(f'no sound in the catalogue answers {want!r}'
                         + (f' (earshot makes no {absent})' if absent else '')
                         + ': `earshot sounds` lists what there is; name '
                         'the sound itself (a chime, a click, a whoosh, rain) or the job it plays '
                         '(success, coin, logo)')
    req = _request(b, theme)
    heard = None
    pick = cands[0]
    if not sound and twin(b, pick['id']) != pick['id']:
        pick = {**pick, 'id': twin(b, pick['id'])}
    if b.get('dur_max') and 'dur' not in req and (registry.get(pick['id']).dur or 0) > b['dur_max']:
        req['dur'] = b['dur_max']                # "under 150 ms": only shortens a design that is longer
    perf = performance(b, pick['id'], events)
    if b.get('gap') and perf and len(perf) > 2 and perf[1] and not req.get('dur'):
        # "three cards landing, 200 ms apart": the row lasts that many gaps and the last event's usual ring
        ring = min(max(registry.get(pick['id']).dur or 0.35, 0.35), 0.6)
        req['dur'] = round(0.02 + (int(perf[1]) - 1) * float(b['gap']) + ring, 3)
    n = takes_asked(b, pick['why'], perf, n)
    paths = _paths(out or pick['id'].replace('/', '_') + '.' + (fmt or 'wav'), n, fmt)
    takes = []
    for i, p in enumerate(paths):
        y, meta = take(pick['id'], seed + i, req, lufs, perform=perf)
        master.write(p, y, sr=sr, mono=mono, bits=bits)
        takes.append({**meta, 'file': p})
    if listen and heard is None:
        heard = _heard(paths[0] if paths[0].lower().endswith(('.wav', '.flac')) else _tmp_wav(pick['id'], seed, req),
                       ([want] if want else []) + list(registry.get(pick['id']).says))
    why = brief.shown(b, pick['why'])
    lines = [_line(t, t['file'], ', '.join(why) if i == 0 else '') for i, t in enumerate(takes)]
    played = takes[0].get('performance')
    if played:
        how = (f"a recording of the whole action ({played['events']} events)" if played.get('recording') else
               f"{played['events']} events as a {played['pattern']}"
               + (f" at {played['pace']:.1f} a second" if played.get('pace') else ''))
        lines.append(f'played: {how}; one event: events=1 or "a single ..."')
    elif perf is not None and b.get('domain') != 'ui':     # a press of an interface is one event: nothing left open
        lines.append('played: one event; the action: events=<n> or say it ("footsteps", "knocking", "three knocks")')
    missing = [] if sound else brief.uncovered(b, pick['why'])
    if missing:
        lines.append(f"not in this sound: {', '.join(missing)}")
    reason = None if sound else doubt(b, cands)
    also = [] if sound else alternatives(b, cands, pick['id'], others_first=reason is not None)
    if reason:
        lines.append(guess(b, cands, also))
        if reason.startswith('earshot makes no '):      # no design is this thing: it can be built (earshot.recipe)
            lines.append('build it from parts: earshot recipe --example')
            lines.append('or let earshot draft, hear and tune it: earshot build "%s"' % str(want).replace('"', "'"))
    elif also:
        lines.append('also: ' + ', '.join(also))
    start = None if sound else R.answer(b, found_by(cands[0]['why'], cands[0]['id']), sketch=True)
    if start:           # a cookbook recipe of this thing that is a sketch: named as a start, not given as the answer
        lines.append(f'a start from parts: earshot make --recipe {start["name"]} (a sketch of {start["says"]}; '
                     f'`earshot recipe {start["name"]}` prints it to change)')
    if b.get('listen_note'):
        lines.append(b['listen_note'])
    if heard:
        lines.append(f"heard as: {heard['query']!r} (z {heard['z']:+.1f}, {heard['verdict']})")
    ign = sorted(set().union(*(set(t.get('ignored') or ()) for t in takes)))
    held = _held(pick['id'], req.get('dur'), takes)
    if held:
        lines.append(held)
        ign = [k for k in ign if k != 'dur']
    lines += list(dict.fromkeys(_note(t, req.get('dur')) for t in takes if _note(t, req.get('dur'))))
    given = {a for a, v in axes.items() if v} | {k for k, v in (('voice', voice), ('material', material)) if v}
    lines += unfollowed(b, pick['id'], pick['why'], ign, req, given, takes)
    unknown = _unknown(b, pick['why'])
    if unknown:
        lines.append(f"not understood: {', '.join(unknown)}")
    seam = _seam([os.path.basename(t['file']) for t in takes if t.get('loop')],
                 {os.path.splitext(p)[1].lower().lstrip('.') for p in paths})
    if seam:
        lines.append(seam)
    res = {'summary': '\n'.join(lines), 'sound': pick['id'], 'files': paths, 'why': why, 'also': also,
           'not_in_sound': missing, 'unknown': unknown, 'takes': takes}
    if heard:
        res['heard'] = heard
    return res


def _tmp_wav(sound, seed, req):
    import tempfile
    from .synth import master
    y, _ = take(sound, seed, req)
    return master.write(os.path.join(tempfile.mkdtemp(), 'heard.wav'), y)


def found_by(why, sound=None):
    """the words the catalogue's first design was found by, a phrase ("ph:launch rocket") as its words: when they hold
    every word of a cookbook thing the catalogue has that sound itself, and its design answers, not the recipe. With
    the design's id, only the words of what it is (its id and `says`) count: a lion's roar listed for "a dragon roar"
    is the nearest of a dragon, a recording of a rocket launch is the thing"""
    words = [w for x in why or () for w in str(x).split(':')[-1].split()]
    if sound is None:
        return tuple(words)
    from .sounds import registry
    try:
        d = registry.get(sound)
    except ValueError:
        return tuple(words)
    own = {brief.stem(w) for w in brief.WORD.findall(' '.join([d.id.replace('/', ' ').replace('_', ' '), *d.says])
                                                      .lower())}
    return tuple(w for w in words if brief.stem(w) in own or w in own)


FAR_SPACE = 'hall'         # a cookbook sound asked for far away: in this space, duller and softer
FAR_LOWPASS = 3000.0       # Hz: the most of its top a far sound keeps
SOFTER = 6.0               # dB under the loudest level of its kind: a cookbook sound asked for soft, quiet or far away
CLOSE_SPACES = ('dry', 'booth')
REPEAT_GAP = 0.3           # s of quiet between one whole of a recipe and the next ("three times")


def asked_of_recipe(rc, b, events=None, seed=0):
    """what a request asks of a cookbook thing that the recipe form can do: (recipe, lufs, lines). Far away ("in
    the distance"): the whole in a hall, every layer duller, 6 dB softer; up close: no room; soft or quiet: 6 dB
    softer; N times (or events=N): every layer N times, one whole after the other, when no layer is a row of its
    own. What it cannot do (louder: a cookbook sound is made at the loudest level of its kind; a row of rows) is a
    `not followed:` line; the length asked is the whole's `dur` (from_recipe)"""
    import copy
    from . import recipe as R
    from .sounds import registry
    rc, lines, level = copy.deepcopy(R.check(rc)), [], None
    softer = registry.LOUDEST['bed' if rc.get('kind') == 'ambience' else ''] - SOFTER
    ax = b.get('axes') or {}
    if ax.get('space', 0) >= 0.3:
        if rc.get('space', 'dry') in ('dry', 'booth', 'room', 'studio'):
            rc['space'] = FAR_SPACE
        for lay in rc['layers']:
            lp = lay.get('lowpass')
            lay['lowpass'] = [min(x, FAR_LOWPASS) for x in lp] if isinstance(lp, list) else min(lp or 2e4, FAR_LOWPASS)
        level = softer
        lines.append(f'as asked: far away (in a {rc["space"]}, duller and 6 dB softer)')
    elif ax.get('space', 0) <= -0.3:
        if rc.get('space', 'dry') not in CLOSE_SPACES:
            rc['space'] = 'dry'
        lines.append('as asked: up close (no room around it)')
    if ax.get('energy', 0) <= -0.5 or (ax.get('hard', 0) <= -0.3 and ax.get('energy', 0) < 0):
        if level is None:
            level = softer
            lines.append('as asked: softer (6 dB under the usual level)')
    elif ax.get('energy', 0) >= 0.5:
        lines.append('not followed: louder; a cookbook sound is made at the loudest level of its kind')
    count = events if events is not None else (b.get('events') if b.get('row') or b.get('perform') else None)
    if count is not None and int(count) > 1:
        count = int(count)
        if any(lay.get('events') for lay in rc['layers']):
            lines.append(f'not followed: {count} times; this recipe is a row of events already, played once')
        elif count > R.EVENT_RANGE['count'][1]:
            lines.append(f'not followed: {count} times; a recipe repeats at most {R.EVENT_RANGE["count"][1]} times')
        else:
            one = len(R.render(rc, seed=seed)[0]) / 48000.0
            gap = round(min(one + REPEAT_GAP, R.EVENT_RANGE['gap'][1]), 3)
            for lay in rc['layers']:
                lay['events'] = {'count': count, 'gap': gap}
            rc.pop('dur', None)
            lines.append(f'as asked: {count} times, {gap:.2f} s apart')
    return rc, level, lines


def from_recipe(recipe, out=None, n=1, seed=0, fmt=None, mono=False, lufs=None, sr=48000, bits=24, listen=False,
                want=None, dur=None, cands=None, given=(), asked=()):
    """a recipe (earshot.recipe: a dict, JSON text, the path of a JSON file, "-" or a cookbook name) rendered and
    written as `make` writes a sound: the same result ('summary', 'sound': "recipe/<name>", 'files', 'takes' with
    the audit, level, length and peak_s of each) plus 'source': "recipe" and 'parts' (what it is built from). n
    takes are the seeds seed .. seed + n - 1: other takes of every layer. dur: a length to cut the sound at. With
    listen, `heard` says how well the first take matches `want` (else the recipe's `says`). cands: the catalogue's
    ranking of the request, when the cookbook answered it; the result then says so and names the nearest designs;
    given: the controls the caller set with that request, which a recipe does not follow, said in one line; asked:
    the lines of what the request asked of the recipe (asked_of_recipe)"""
    import tempfile
    from . import recipe as R
    from .synth import master
    if listen:
        from . import extras
        extras.need(*extras.LISTEN, what='make with listen')
    rc = R.load(recipe)
    paths = _paths(out or f'{rc["name"]}.{fmt or "wav"}', n, fmt)
    takes, first = [], None
    for i, p in enumerate(paths):
        y, meta = R.render(rc, seed=seed + i, lufs=lufs, dur=dur)
        first = y if first is None else first
        master.write(p, y, sr=sr, mono=mono, bits=bits)
        takes.append({**meta, 'file': p})
    lines = [_line(t, t['file'], 'recipe' if i == 0 else '') for i, t in enumerate(takes)]
    lines.append('built from: ' + '; '.join(takes[0]['parts']))
    also = [c['id'] for c in (cands or [])[:3]]
    if cands is not None:
        lines.append(f'built from parts by the cookbook recipe {rc["name"]}: no design of the catalogue is this '
                     f'sound; `earshot recipe {rc["name"]}` prints the recipe to change'
                     + (f'; the nearest designs: {", ".join(also)} (sound=<id>)' if also else ''))
    lines += list(asked)
    if given:
        lines.append(f'not followed: {", ".join(given)}; a recipe plays as it is written')
    lines += list(dict.fromkeys(note for t in takes for note in t.get('notes', ())))
    res = {'summary': '', 'sound': takes[0]['id'], 'files': paths, 'why': [], 'also': also, 'not_in_sound': [],
           'unknown': [], 'takes': takes, 'source': 'recipe', 'recipe': rc['name'], 'parts': takes[0]['parts']}
    says = (want if want and str(want).strip() else None) or rc.get('says')
    if listen and says:
        with tempfile.TemporaryDirectory() as tmp:
            res['heard'] = _heard(master.write(os.path.join(tmp, 'heard.wav'), first), [says])
        lines.append(f"heard as: {res['heard']['query']!r} (z {res['heard']['z']:+.1f}, {res['heard']['verdict']})")
    res['summary'] = '\n'.join(lines)
    return res


def _set_recipe(it, seed, top, out, fmt):
    """a set's entry that is a recipe ({"name", "recipe", + seed, lufs, dur}) as a row the set levels and writes
    with its other sounds: an event as it is made alone, an ambience by loudness under the set's loudest"""
    from types import SimpleNamespace
    from . import recipe as R
    extra = [k for k in it if k not in ('name', 'recipe', 'seed', 'lufs', 'dur')]
    if extra:
        raise ValueError(f'{it["name"]}: {extra[0]}: a recipe of a set takes name, recipe, seed, lufs and dur; '
                         'change the sound in the recipe itself')
    rc = R.load(it['recipe'])
    bed = rc.get('kind') == 'ambience'
    alone = it.get('lufs') is None and top is None and not bed
    level = None if alone else it.get('lufs', (TOP if top is None else top) + WEIGHT['bed' if bed else 'object'])
    own = _whole(it['seed'], f'{it["name"]}: seed', -2 ** 31, 2 ** 31 - 1) if it.get('seed') is not None else seed
    y, meta = R.render(rc, seed=own, lufs=level, dur=_length(it.get('dur'), f'{it["name"]}: dur'))
    return {'p': os.path.join(out, f'{it["name"]}.{fmt}'), 'y': y, 'meta': meta, 'alone': alone, 'name': it['name'],
            'd': SimpleNamespace(id=meta['id'], kind=meta['kind'], loop=False),
            'why': 'recipe: ' + '; '.join(meta['parts'])}


def make_set(spec, out=None, theme=None, fmt=None, seed=0, mono=None, lufs=None, sprite=None):
    """a whole set of sounds from one spec, as one family: {'summary', 'files', 'manifest', 'sounds': [...]}.
    spec: a dict, the path of a JSON file, or the name of a ready-made set (earshot.sounds.sets: ui, game, arcade,
    film):
        {"set": "ui", "theme": "glass" | {...}, "format": ["webm", "mp3"], "out": "public/sounds", "mono": true,
         "lufs": -18, "sprite": "sprite.webm",
         "sounds": [{"name": "tap", "want": "a soft tap for buttons"},
                    {"name": "success", "want": "a quick success sound", "axes": {"bright": 0.2}}]}
    "set" starts from a ready-made list; "sounds" adds to it, or changes the sound of the same name. Each sound
    takes name (the file name) and want (words) and / or sound (a design id), plus pitch, dur, voice, material,
    seed, n, axes, lufs, events (a thing done in a row, as in `make`) and loop (false: once); or name and recipe (a
    sound built from parts: earshot.recipe, with seed, lufs and dur). Every sound shares the
    theme (one voice, key and room). A name's take is its `seed` when
    given, else the next take of its design (the first name of a learned design gets its best recording). Level:
    every one-shot is made as it is made alone (true peak -1 dBFS) and the set is balanced by the loudest 50 ms
    (balance()): none more than BALANCE dB over the quietest, a ready-made set's `quiet` names at the quietest,
    turned down by gain only. Loops, beds, pads and music, and every sound when `lufs` is given (the loudest
    allowed), are levelled by momentary loudness, each design's own weight under it (WEIGHT; -16 when not given).
    `manifest.json` in the folder lists every file with its sound, length, level and recording, and
    "sprite" also packs them into one file with a map (earshot.pack). format (or fmt): one format or several
    ("webm,mp3" or a list): every sound is written once per format, the manifest's `file` is the first and `files`
    all of them, and a sprite named with one of them (or no extension) is written in each, its map's `src` listing
    them in that order (a browser plays the first it can). A loop is written only in formats that keep its seam:
    mp3 and m4a pad a file's start and end, so a loop gets flac in their place (its `files` say what was written).
    A sprite holds every sound, loops included, in each of its formats: one map serves them all, and a loop inside
    a sprite repeats a region of the decoded audio, away from the file's padded ends. The result also carries
    `also` and `not_in_sound` by the name of each sound asked for in words; the summary shows them only where the
    pick is in doubt, and says when two names got the same sound; two names that play different recordings of one
    learned design (an on and an off of one switch) get a plain note saying which recording each plays."""
    from .sounds import registry
    from .synth import master
    from .sounds import sets
    if isinstance(spec, str):
        if spec not in sets.SETS and not os.path.isfile(spec):
            raise ValueError(f'unknown set {spec!r}: one of {", ".join(sets.SETS)}, or the path of a JSON spec')
        spec = {'set': spec} if spec in sets.SETS else read_json(spec)
    if not isinstance(spec, dict) or not isinstance(spec.get('sounds') or [], list) or not all(
            isinstance(it, dict) for it in spec.get('sounds') or []):
        raise ValueError('a set spec is {"set": "ui"} and / or '
                         '{"sounds": [{"name": "tap", "want": "a soft tap"}, ...]}')
    items, skipped, quiet = list(spec.get('sounds') or []), [], ()
    if spec.get('set'):
        base = sets.get(spec['set'], theme or spec.get('theme'))     # a theme named: the tonal figures
        theme = theme or spec.get('theme') or base['theme']
        named = {it.get('name') for it in items}
        have = set(registry.ids())
        skipped = [n for n, sid in base['sounds'] if sid not in have and n not in named]
        pins, quiet = base.get('seeds', {}), tuple(base.get('quiet', ()))
        items = [{'name': n, 'sound': sid, **({'seed': pins[n]} if n in pins else {})}
                 for n, sid in base['sounds'] if sid in have and n not in named] + items
    if not items:
        raise ValueError('the spec has no "sounds" and names no "set"')
    bad = [f'{it.get("name", i)}: {k}' for i, it in enumerate(items) for k in it if k not in SET_FIELDS]
    names = [it.get('name') for it in items]
    if bad or None in names or len(set(names)) != len(names):
        raise ValueError(f'unknown fields {bad}: a sound takes {", ".join(SET_FIELDS)}' if bad else
                         'every sound of a set needs its own "name"')
    seed = _whole(seed, 'seed', -2 ** 31, 2 ** 31 - 1)
    out = out or spec.get('out') or 'sounds'
    theme = theme or spec.get('theme')
    fmts = _formats(fmt or spec.get('format') or 'wav')
    mono = spec.get('mono', False) if mono is None else mono
    top = lufs if lufs is not None else spec.get('lufs')
    rows, lines, ys, alts, missed, first, played, pending, takes = [], [], [], {}, {}, {}, {}, [], {}
    for it in items:
        if it.get('recipe') is not None:        # a sound built from parts (earshot.recipe), levelled with the set
            lines.append(None)
            pending.append({**_set_recipe(it, seed, top, out, fmts[0]), 'line': len(lines) - 1})
            lines += [f'  {note}' for note in pending[-1]['meta'].get('notes', ())]
            continue
        if not ((it.get('want') and str(it['want']).strip()) or it.get('sound')):
            raise ValueError(f'{it["name"]}: say what sound you want ("want") or name one ("sound")')
        it = {**it, 'dur': _length(it.get('dur'), f'{it["name"]}: dur')}
        if it.get('n') is not None:
            it['n'] = _whole(it['n'], f'{it["name"]}: n (takes)', 1, MAX_TAKES)
        if it.get('seed') is not None:
            it['seed'] = _whole(it['seed'], f'{it["name"]}: seed', -2 ** 31, 2 ** 31 - 1)
        if it.get('events') is not None:
            it['events'] = _whole(it['events'], f'{it["name"]}: events', 1, MAX_EVENTS)
        if it.get('loop') not in (None, True, False):
            raise ValueError(f'{it["name"]}: loop is true or false, not {it["loop"]!r}')
        b, cands = choose(it.get('want'), it.get('sound'), pitch=it.get('pitch'), dur=it.get('dur'),
                          voice=it.get('voice'), material=it.get('material'), axes=it.get('axes'))
        if b.get('dur') is not None:
            b['dur'] = _length(b['dur'], f'{it["name"]}: the length asked for')
        if not cands:
            raise ValueError(f'{it["name"]}: no sound in the catalogue answers {it.get("want")!r}')
        d = registry.get(cands[0]['id'] if it.get('sound') else twin(b, cands[0]['id']))
        if it.get('loop') is False:
            b['loop'] = False
        # a thing done in a row is played as `make` plays it ("three knocks, 2 seconds", footsteps, events=4)
        perf = performance(b, d.id, it.get('events')) if it.get('want') or it.get('events') else None
        alone = it.get('lufs') is None and top is None and not d.loop and d.kind not in registry.LOUDEST
        level = None if alone else it.get('lufs', (TOP if top is None else top) + WEIGHT.get(d.kind, -4.0))
        k = int(it.get('n') or b['n'] or 1)
        mine = {}                         # the recordings this name played: {n: meta['member']}
        for j, p in enumerate(_paths(os.path.join(out, f'{it["name"]}.{fmts[0]}'), k, fmts[0])):
            s = int(it['seed']) + j if it.get('seed') is not None else seed + takes.get(d.id, 0)
            takes[d.id] = takes.get(d.id, 0) + 1
            y, meta = take(d.id, s, _request(b, theme), lufs=level, perform=perf)
            if meta.get('member'):
                mine[meta['member']['n']] = meta['member']
            lines.append(None)            # the file's line, written once the set is balanced
            pending.append({'p': p, 'y': y, 'meta': meta, 'd': d, 'alone': alone, 'name': it['name'],
                            'line': len(lines) - 1, 'why': ', '.join(brief.shown(b, cands[0]['why']))})
            done = meta.get('performance')
            if done and j == 0:
                lines.append(f"  played: {done['events']} events" + (
                    ' (a recording of the whole action)' if done.get('recording') else f" as a {done.get('pattern')}"))
            if meta.get('note'):
                lines.append(f"  {meta['note']}")
            if meta.get('loop') and _loop_formats(fmts) != fmts:
                lines.append(f"  a loop: {SEAM_SAFE} instead of {' / '.join(f for f in fmts if f in PADDED)}, "
                             'which leaves a gap at the seam')
        reason = None if it.get('sound') else doubt(b, cands)
        alts[it['name']] = alternatives(b, cands, d.id, others_first=reason is not None)
        if d.id in first and mine and not set(mine) & played[d.id]:
            # two names, one design of several recordings, each name its own recording (toggle_on / toggle_off): an
            # intended pair, said as a fact
            lines.append(f'  another recording of {d.id} than {first[d.id]}: '
                         + ', '.join(m['recording'] for m in mine.values()))
        elif d.id in first:               # two names, one sound: the second says so and names another
            chip = brief.chip(d.row())
            free = [a for a in alts[it['name']] + alternatives(b, cands[:1], d.id, k=6) if a not in first]
            other = next((a for a in free if brief.chip(registry.get(a).row()) == chip), None)
            lines.append(f'  same design as {first[d.id]}: ask for a different one with "sound"'
                         + (f' (the nearest: {other})' if other else ''))
        first.setdefault(d.id, it['name'])
        played.setdefault(d.id, set()).update(mine)
        if it.get('sound'):
            del alts[it['name']]
            continue
        missing, unknown = brief.uncovered(b, cands[0]['why']), _unknown(b, cands[0]['why'])
        if missing:
            missed[it['name']] = missing
            lines.append(f"  not in this sound: {', '.join(missing)}")
        if (missing or reason) and alts[it['name']]:      # only where the pick is in doubt: a set stays short
            lines.append('  ' + (guess(b, cands, alts[it['name']]) or 'also: ' + ', '.join(alts[it['name']])))
        if unknown:
            lines.append(f"  not understood: {', '.join(unknown)}")
    balance(pending, quiet)
    for q in pending:
        y, meta, d, p = q['y'], q['meta'], q['d'], q['p']
        stem = os.path.splitext(p)[0]
        loop = bool(meta.get('loop', False))
        own = _loop_formats(fmts) if loop else fmts
        files = [master.write(f'{stem}.{f}', y, mono=mono and not d.loop, bits=16 if f == 'wav' else 24)
                 for f in own]
        ys.append(y)
        rows.append({'name': os.path.basename(stem), 'file': os.path.basename(files[0]),
                     'files': [os.path.basename(f) for f in files], 'sound': d.id, 'dur_s': meta['dur_s'],
                     'lufs_max': meta['lufs_max'], 'loop': loop, 'ok': meta['checks']['ok'],
                     **({'pitch': meta['pitch']} if meta.get('pitch') else {}),
                     **({'member': meta['member']} if meta.get('member') else {})})
        shown = p if len(own) == 1 else f'{stem}.{{{",".join(own)}}}'
        lines[q['line']] = _line(meta, shown, q['why'])
    man = os.path.join(out, 'manifest.json')
    with open(man, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'theme': theme, 'format': fmts[0], 'formats': fmts, 'sounds': rows}, fh, indent=1)
        fh.write('\n')
    head = f'{len(rows)} sounds in {out} ({", ".join(fmts)}), manifest.json'
    res = {'files': [os.path.join(out, f) for r in rows for f in r['files']], 'manifest': man, 'sounds': rows,
           'also': alts, 'not_in_sound': missed}
    sprite = sprite or spec.get('sprite')
    if sprite:
        from .pack import pack
        stem, ext = os.path.splitext(os.path.join(out, os.path.basename(sprite)))
        ext = ext.lower().lstrip('.')
        outs = [f'{stem}.{f}' for f in fmts] if not ext or ext in fmts else [f'{stem}.{ext}']
        sp = pack([os.path.join(out, r['file']) for r in rows], outs, mono=mono,
                  loops=[r['name'] for r in rows if r['loop']], ys=ys)
        lines.append(sp['summary'])
        res['sprite'] = sp['file']
        res['sprites'] = sp['files']
    if skipped:
        lines.append(f"not in this catalogue, left out: {', '.join(skipped)}")
    return {'summary': '\n'.join([head] + lines), **res}


# a kind's level under the loudest sound of a set (LU): the more often a sound plays, the quieter it sits
WEIGHT = {'tap': -10.0, 'tick': -12.0, 'click': -9.0, 'pop': -7.0, 'note': -5.0, 'chime': -3.0, 'figure': -2.0,
          'hit': -1.0, 'thud': -3.0, 'whoosh': -5.0, 'rise': -3.0, 'fall': -3.0, 'zap': -3.0, 'boom': 0.0,
          'noise': -8.0, 'texture': -12.0, 'bed': -14.0, 'pad': -10.0, 'stinger': 0.0, 'voice': -5.0, 'object': -6.0,
          'music': -8.0}
TOP = -16.0             # LUFS (momentary max) of a set's loudest loop or bed when the spec names no `lufs`
# a set's one-shots, balanced by their loudest 50 ms: levelled to a common loudness (-16 LUFS + WEIGHT), an
# interface set came out at -21 to -25 LUFS, 6 to 8 dB under the same takes made alone and 8 to 10 dB under the
# sounds of other sets, and quiet reads as far and thin. So each is made as it is made alone (peak -1 dBFS) and only
# turned down: none more than BALANCE dB over the quietest (not counting one under FLOOR, which cannot come up
# without being folded), a `quiet` name (hover) QUIET dB under it. Gain only: nothing is folded or limited again
BALANCE, FLOOR, QUIET = 2.5, -18.0, 0.5


def balance(pending, quiet=()):
    """turn a set's one-shots (the takes made alone, `pending` rows with 'y', 'meta', 'alone', 'name') down so that
    their loudest 50 ms lie within BALANCE dB of the quietest (module constants), in place"""
    from .synth import master
    lv = {i: master.loudest(q['y']) for i, q in enumerate(pending) if q['alone']}
    ref = [v for i, v in lv.items() if pending[i]['name'] not in quiet]
    if not ref:
        return
    low = max(min(ref), FLOOR)
    for i, v in lv.items():
        q = pending[i]
        most = low - QUIET if q['name'] in quiet else low + BALANCE
        if v > most:
            q['y'] = q['y'] * 10 ** ((most - v) / 20)
            q['meta'] = {**q['meta'], 'lufs_max': round(master.loudness(q['y']), 1),
                         'peak_dbfs': round(master.true_peak_db(q['y']), 2)}


def sounds(search=None, domain=None, kind=None, family=None):
    """the catalogue, compact: {'summary', 'sounds': rows}. search: words, ranked as `make` would rank them"""
    from .sounds import registry
    rows = registry.rows(domain=domain, kind=kind, family=family)
    if search:
        by = {r['id']: r for r in rows}
        rows = [by[c['id']] for c in brief.rank(brief.read(search), rows, top=20)]
    lines = [f"{r['id']:<26} {(r['dur_s'] or 0):>5.2f} s  {r['says'][0]}" + (' (loop)' if r['loop'] else '')
             for r in rows]
    fams = sorted({r['id'].split('/')[0] for r in rows})
    head = f'{len(rows)} sounds' + (f' for {search!r}' if search else f' in {len(fams)} families')
    return {'summary': '\n'.join([head] + lines), 'sounds': rows}


def _axes(args):
    return {a: getattr(args, a) for a in ('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy', 'motion')
            if getattr(args, a, None) is not None}


def add_cli(sub):
    p = sub.add_parser('make', help='make a sound from a request in words (or a whole set from a spec)')
    p.add_argument('want', nargs='?', help='the request in words, e.g. "a soft success sound for a finance app"')
    p.add_argument('-o', '--output', help='the file to write; its extension picks the format (wav, flac, ogg, mp3, '
                                          'm4a, webm). With --spec: the folder')
    p.add_argument('--spec', help='a JSON file of a whole set: {"set": "ui", "format": "webm,mp3", '
                                  '"sounds": [{"name": "tap", "want": "a soft tap"}, ...]}; docs/MAKE.md has every '
                                  'field. Re-running a spec into the same folder rewrites its files and manifest.json')
    p.add_argument('--set', dest='set_name', help='a ready-made set: ui, game, arcade or film (one family, one call)')
    p.add_argument('--format', help='with --spec / --set: the format of the set, or several with commas (webm,mp3: '
                                    'every sound in each; a browser plays the first it can). ogg = Vorbis, webm = '
                                    'Opus, m4a = AAC, mp3; wav and flac are lossless. Loops: not mp3 or m4a (their '
                                    'padding leaves a gap at the seam)')
    p.add_argument('--sprite', help='with --spec / --set: also pack the set into this one file with a map')
    p.add_argument('--sound', help='a design id from `earshot sounds` instead of (or on top of) the words')
    p.add_argument('--recipe', help='a sound built from the catalogue\'s parts: a JSON file of a recipe, - to read '
                                    'it from standard input, or the name of a cookbook recipe (`earshot recipe '
                                    '--example` shows the form); the words, if given, say what it is for --listen')
    p.add_argument('--n', type=int, default=1, help='how many takes (file_1 .. file_n)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--theme', help='a theme preset (soft, glass, wood, felt, tine, metal, electric, digital, retro): '
                                   'the voice, key and room of tonal figures; a set with a theme named plays its '
                                   'figures on that instrument, without one its sounds of real things')
    p.add_argument('--pitch', help='a note (C5) or Hz')
    p.add_argument('--dur', type=float, help='seconds, up to 120: how long a figure of notes sounds (damped by the '
                                             'hand to end '
                                             'there; a figure rings no longer than it rings), the hold of a pad, the '
                                             'length of an ambience (shorter than its shortest loop: a cut, not a '
                                             'loop, unless the words say "loop") or a whoosh; a sound learned from '
                                             'recordings within 0.6 to 1.8 times its natural length; a music loop '
                                             'keeps whole bars; a counted action ("three knocks", footsteps) is '
                                             'spread over it; a single struck sound (a tap, a hit) rings as it '
                                             'rings: it ignores --dur and says so')
    p.add_argument('--voice', help='the instrument or patch of a tonal sound, played only when named here or in the '
                                   'words: marimba (the figures\' own voice), harp, pizzicato and felt_piano are '
                                   'real instruments learned note by note; kalimba, glockenspiel, vibraphone, '
                                   'soft_pluck ... are models')
    p.add_argument('--material', help='what a tap, click or knock is made of')
    p.add_argument('--events', type=int, help='a thing a person does several times in a row (footsteps, a knock, '
                                              'typing, coins, keys, clinking glasses, a dog barking): 1 = one '
                                              'event, 2 or more = the action with that many (any other one-shot: a '
                                              'row of that many in one file); default: as the words say ("three '
                                              'knocks", "clapping, 4 seconds")')
    for a in ('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy', 'motion'):
        p.add_argument(f'--{a}', type=float, help='-1 .. 1, 0 = as designed')
    p.add_argument('--lufs', type=float, help='momentary-max loudness instead of a -1 dBFS peak')
    p.add_argument('--mono', action='store_true')
    p.add_argument('--no-loop', action='store_true', help='an ambience once, with a fade in and out, instead of a '
                                                          'seamless loop')
    p.add_argument('--listen', action='store_true', help='also rank by how each design sounds (CLAP against the '
                                                         'sound atlas) and say what the result is heard as (loads '
                                                         'the model: 20-40 s)')
    common(p)
    p.set_defaults(func=_run_make)
    q = sub.add_parser('sounds', help='the catalogue of sounds `earshot make` can make')
    q.add_argument('--search', help='words; ranked as make would rank them')
    q.add_argument('--domain', help='ui, game, motion, ambience or music')
    q.add_argument('--kind')
    q.add_argument('--family')
    common(q)
    q.set_defaults(func=lambda a: _emit(sounds(a.search, a.domain, a.kind, a.family), a))


def _emit(res, args):
    """the summary alone as text (it is the whole answer); everything with --json"""
    emit(res, args, text=res['summary'])


def _run_make(args):
    if args.spec or args.set_name:
        return _emit(make_set(args.spec or args.set_name, out=args.output, theme=args.theme, fmt=args.format,
                              seed=args.seed, mono=True if args.mono else None, lufs=args.lufs,
                              sprite=args.sprite), args)
    return _emit(make(args.want, args.output, sound=args.sound, n=args.n, seed=args.seed, theme=args.theme,
                      listen=args.listen, mono=args.mono, lufs=args.lufs, pitch=args.pitch, dur=args.dur,
                      voice=args.voice, material=args.material, loop=False if args.no_loop else None,
                      events=args.events, recipe=args.recipe, **_axes(args)), args)
