"""earshot make: ask for a sound in words and get it, made in code.

    earshot make "a soft success chime for a finance app" -o public/success.ogg
    earshot make "coin pickup, bright" -o sfx/coin.wav --n 3            # coin_1.wav .. coin_3.wav: three takes
    earshot make "rain on a window" -o rain.ogg --dur 20                # a seamless loop
    earshot make --set ui --theme glass -o public/sounds --format ogg   # a ready-made set as one family
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
from .report import emit, read_json

SET_FIELDS = ('name', 'want', 'sound', 'pitch', 'dur', 'voice', 'material', 'seed', 'n', 'axes', 'lufs')
TRIES = 4
GUESS_Z, GUESS_HAIR = 2.0, 0.06     # `guess`: semantic support, and how close another family may come (share)
# formats whose encoder pads the start and end of a file and records it only in a note some players ignore (the LAME
# tag of an mp3, the edit list of an m4a); Vorbis (granule positions) and Opus in WebM (pre-skip and discard padding)
# carry it in the stream itself. A 384000-sample loop decodes raw to +1920 samples as mp3 and +1024 as m4a
PADDED = ('mp3', 'm4a')
SEAM_SAFE = 'flac'      # what a loop of a set is written as in place of a padded format
GIST = 40               # characters of a design's description next to an alternative of a guess


def _paths(out, n, fmt):
    base, ext = os.path.splitext(out)
    ext = ext or f'.{fmt or "wav"}'
    return [base + ext] if n == 1 else [f'{base}_{i + 1}{ext}' for i in range(n)]


def _line(meta, path, why):
    bits = [f"{meta['dur_s']:.2f} s"]
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
    """why the pick rests on weak evidence, or None: no job or sound word matched and the request does not sound
    like the pick (z under GUESS_Z, or not listened), or the best design of another family is within GUESS_HAIR (a
    share of the pick's score) of it: members of one family tie often and harmlessly. Thresholds chosen on the
    development request sets (660 requests, top 8 ranked): lexical ranking, 32 % of wrong first picks flagged and
    10 % of right ones; blended, 36 % and 8 %"""
    if not cands or cands[0]['score'] is None:
        return None
    top = cands[0]
    fam = top['id'].split('/')[0]
    others = [c for c in cands[1:] if c['id'].split('/')[0] != fam]
    named = brief.named(top['why'])
    heard = top.get('z') is not None and top['z'] >= GUESS_Z
    close = bool(others) and top['score'] - others[0]['score'] <= GUESS_HAIR * abs(top['score'])
    if (named or heard) and not close:
        return None
    return 'another kind of sound fits almost as well' if close else 'no job or sound word matched' + (
        ' and it does not clearly sound like the request' if b.get('listened') else '')


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


def guess(b, cands, also=None):
    """the alternatives line when the pick is a guess (`doubt`), else None: why it is a guess and the alternatives
    (default: the ranked candidates, other families first), each with what it sounds like in a few words (`gist`),
    so one of them can be chosen without looking it up"""
    why = doubt(b, cands)
    if why is None:
        return None
    if also is None:
        fam = cands[0]['id'].split('/')[0]
        also = [c['id'] for c in cands[1:] if c['id'].split('/')[0] != fam]
        also = (also + [c['id'] for c in cands[1:] if c['id'] not in also])[:3]
    alts = ', '.join(f'{a} ({gist(a)})' if gist(a) else a for a in also)
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
    if b.get('loop') is False:
        req['loop'] = False
    return {**req, **b['axes'], 'theme': th}


def take(sound, seed, req, lufs=None, peak_db=-1.0):
    """one clean take of a design: the first of TRIES seeds (seed, seed + 101, ...) that passes the audit, else the
    first one with its warnings: (y, meta)"""
    from .sounds import registry
    first = None
    for k in range(TRIES):
        y, meta = registry.render(sound, seed=seed + 101 * k, lufs=lufs, peak_db=peak_db, **req)
        if meta['checks']['ok']:
            return y, meta
        first = first or (y, meta)
    return first


def _heard(path, queries):
    from . import match as M
    m = M.match(path, queries)
    return {'query': m.get('query'), 'z': round(float(m['z']), 2), 'verdict': m['verdict']}


def make(want=None, out=None, sound=None, n=1, seed=0, theme=None, listen=False, fmt=None, mono=False, lufs=None,
         sr=48000, bits=24, pitch=None, dur=None, voice=None, material=None, loop=None, **axes):
    """make the sound a request asks for and write it: {'summary', 'sound', 'files', 'why', 'also', 'not_in_sound',
    'unknown', 'takes': [meta ...]} (+ 'heard' with listen); `not_in_sound`: the sounds or jobs the request named
    that the chosen design was not picked for, as written ("water drop"), `also`: up to three alternatives. want:
    the request in words; sound: a design id instead of (or on top of) the words; out: the file to write (its
    extension picks the format: wav, flac, ogg, mp3, m4a, webm; none = `<sound>.wav` in the working folder); n
    takes get `_1 .. _n` before the extension; theme: a preset name or dict (earshot.sounds.theme); lufs: a
    momentary-max loudness instead of the -1 dBFS peak; pitch / dur / voice / material and the axes (bright, hard,
    ring, size, space, width, energy, motion) override what the words say."""
    from .sounds import registry
    from .synth import master
    if not (want or sound):
        raise ValueError('say what sound you want (want=...) or name one (sound=...)')
    b, cands = choose(want, sound, top=8, listen=True if listen else None, pitch=pitch, dur=dur, voice=voice,
                      material=material, axes=axes)
    if loop is False:
        b['loop'] = False
    if not cands:
        raise ValueError(f'no sound in the catalogue answers {want!r}: `earshot sounds` lists what there is; name '
                         'the sound itself (a chime, a click, a whoosh, rain) or the job it plays (success, coin, logo)')
    n = int(b['n'] or n)
    req = _request(b, theme)
    heard = None
    pick = cands[0]
    if b.get('dur_max') and 'dur' not in req and (registry.get(pick['id']).dur or 0) > b['dur_max']:
        req['dur'] = b['dur_max']                # "under 150 ms": only shortens a design that is longer
    paths = _paths(out or pick['id'].replace('/', '_') + '.' + (fmt or 'wav'), n, fmt)
    takes = []
    for i, p in enumerate(paths):
        y, meta = take(pick['id'], seed + i, req, lufs)
        master.write(p, y, sr=sr, mono=mono, bits=bits)
        takes.append({**meta, 'file': p})
    if listen and heard is None:
        heard = _heard(paths[0] if paths[0].lower().endswith(('.wav', '.flac')) else _tmp_wav(pick['id'], seed, req),
                       ([want] if want else []) + list(registry.get(pick['id']).says))
    why = brief.shown(b, pick['why'])
    lines = [_line(t, t['file'], ', '.join(why) if i == 0 else '') for i, t in enumerate(takes)]
    missing = [] if sound else brief.uncovered(b, pick['why'])
    if missing:
        lines.append(f"not in this sound: {', '.join(missing)}")
    reason = None if sound else doubt(b, cands)
    also = [] if sound else alternatives(b, cands, pick['id'], others_first=reason is not None)
    if reason:
        lines.append(guess(b, cands, also))
    elif also:
        lines.append('also: ' + ', '.join(also))
    if b.get('listen_note'):
        lines.append(b['listen_note'])
    if heard:
        lines.append(f"heard as: {heard['query']!r} (z {heard['z']:+.1f}, {heard['verdict']})")
    ign = sorted(set().union(*(set(t.get('ignored') or ()) for t in takes)))
    if ign:
        lines.append(f"not used by this sound: {', '.join(ign)}")
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


def make_set(spec, out=None, theme=None, fmt=None, seed=0, mono=None, lufs=None, sprite=None):
    """a whole set of sounds from one spec, as one family: {'summary', 'files', 'manifest', 'sounds': [...]}.
    spec: a dict, the path of a JSON file, or the name of a ready-made set (earshot.sounds.sets: ui, game, arcade,
    film):
        {"set": "ui", "theme": "glass" | {...}, "format": ["webm", "mp3"], "out": "public/sounds", "mono": true,
         "lufs": -18, "sprite": "sprite.webm",
         "sounds": [{"name": "tap", "want": "a soft tap for buttons"},
                    {"name": "success", "want": "a happy two-note success chime", "dur": 0.5, "axes": {"bright": 0.2}}]}
    "set" starts from a ready-made list; "sounds" adds to it, or changes the sound of the same name. Each sound
    takes name (the file name) and want (words) and / or sound (a design id), plus pitch, dur, voice, material,
    seed, n, axes, lufs. Every sound shares the theme (one voice, key and room). The sounds are levelled against
    each other by loudness (`lufs`: the loudest allowed; default -16 for the set's loudest sound, each design's
    own weight under it), `manifest.json` in the folder lists every file with its sound, length and level, and
    "sprite" also packs them into one file with a map (earshot.pack). format (or fmt): one format or several
    ("webm,mp3" or a list): every sound is written once per format, the manifest's `file` is the first and `files`
    all of them, and a sprite named with one of them (or no extension) is written in each, its map's `src` listing
    them in that order (a browser plays the first it can). A loop is written only in formats that keep its seam:
    mp3 and m4a pad a file's start and end, so a loop gets flac in their place (its `files` say what was written).
    A sprite holds every sound, loops included, in each of its formats: one map serves them all, and a loop inside
    a sprite repeats a region of the decoded audio, away from the file's padded ends. The result also carries
    `also` and `not_in_sound` by the name of each sound asked for in words; the summary shows them only where the
    pick is in doubt, and says when two names got the same design."""
    from .sounds import registry
    from .synth import master
    from .sounds import sets
    if isinstance(spec, str):
        spec = {'set': spec} if spec in sets.SETS else read_json(spec)
    items, skipped = list(spec.get('sounds') or []), []
    if spec.get('set'):
        base = sets.get(spec['set'])
        theme = theme or spec.get('theme') or base['theme']
        named = {it.get('name') for it in items}
        have = set(registry.ids())
        skipped = [n for n, sid in base['sounds'] if sid not in have and n not in named]
        items = [{'name': n, 'sound': sid} for n, sid in base['sounds'] if sid in have and n not in named] + items
    if not items:
        raise ValueError('the spec has no "sounds" and names no "set"')
    bad = [f'{it.get("name", i)}: {k}' for i, it in enumerate(items) for k in it if k not in SET_FIELDS]
    names = [it.get('name') for it in items]
    if bad or None in names or len(set(names)) != len(names):
        raise ValueError(f'unknown fields {bad}: a sound takes {", ".join(SET_FIELDS)}' if bad else
                         'every sound of a set needs its own "name"')
    out = out or spec.get('out') or 'sounds'
    theme = theme or spec.get('theme')
    fmts = _formats(fmt or spec.get('format') or 'wav')
    mono = spec.get('mono', False) if mono is None else mono
    top = lufs if lufs is not None else spec.get('lufs', -16.0)
    rows, lines, ys, alts, missed, first = [], [], [], {}, {}, {}
    for it in items:
        b, cands = choose(it.get('want'), it.get('sound'), pitch=it.get('pitch'), dur=it.get('dur'),
                          voice=it.get('voice'), material=it.get('material'), axes=it.get('axes'))
        if not cands:
            raise ValueError(f'{it["name"]}: no sound in the catalogue answers {it.get("want")!r}')
        d = registry.get(cands[0]['id'])
        level = it.get('lufs', top + WEIGHT.get(d.kind, -4.0))
        k = int(it.get('n') or b['n'] or 1)
        for p in _paths(os.path.join(out, f'{it["name"]}.{fmts[0]}'), k, fmts[0]):
            y, meta = take(d.id, int(it.get('seed', seed)) + len(rows), _request(b, theme), lufs=level)
            stem = os.path.splitext(p)[0]
            loop = bool(meta.get('loop', False))
            own = _loop_formats(fmts) if loop else fmts
            files = [master.write(f'{stem}.{f}', y, mono=mono and not d.loop, bits=16 if f == 'wav' else 24)
                     for f in own]
            ys.append(y)
            rows.append({'name': os.path.basename(stem), 'file': os.path.basename(files[0]),
                         'files': [os.path.basename(f) for f in files], 'sound': d.id, 'dur_s': meta['dur_s'],
                         'lufs_max': meta['lufs_max'], 'loop': loop,
                         'ok': meta['checks']['ok'], **({'pitch': meta['pitch']} if meta.get('pitch') else {})})
            shown = p if len(own) == 1 else f'{stem}.{{{",".join(own)}}}'
            lines.append(_line(meta, shown, ', '.join(brief.shown(b, cands[0]['why']))))
            if own != fmts:
                lines.append(f"  a loop: {SEAM_SAFE} instead of {' / '.join(f for f in fmts if f in PADDED)}, "
                             'which leaves a gap at the seam')
        reason = None if it.get('sound') else doubt(b, cands)
        alts[it['name']] = alternatives(b, cands, d.id, others_first=reason is not None)
        if d.id in first:                 # two names, one design: the second says so and names another
            chip = brief.chip(d.row())
            free = [a for a in alts[it['name']] + alternatives(b, cands[:1], d.id, k=6) if a not in first]
            other = next((a for a in free if brief.chip(registry.get(a).row()) == chip), None)
            lines.append(f'  same design as {first[d.id]}: ask for a different one with "sound"'
                         + (f' (the nearest: {other})' if other else ''))
        first.setdefault(d.id, it['name'])
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
    p = sub.add_parser('make', help='make a sound in code from a request in words (or a whole set from a spec)')
    p.add_argument('want', nargs='?', help='the request in words, e.g. "a soft success chime for a finance app"')
    p.add_argument('-o', '--output', help='the file to write; its extension picks the format (wav, flac, ogg, mp3, '
                                          'm4a, webm). With --spec: the folder')
    p.add_argument('--spec', help='a JSON file of a whole set: {"set": "ui", "theme": "felt", "format": "webm,mp3", '
                                  '"sounds": [{"name": "tap", "want": "a soft tap"}, ...]}; docs/MAKE.md has every '
                                  'field. Re-running a spec into the same folder rewrites its files and manifest.json')
    p.add_argument('--set', dest='set_name', help='a ready-made set: ui, game, arcade or film (one family, one call)')
    p.add_argument('--format', help='with --spec / --set: the format of the set, or several with commas (webm,mp3: '
                                    'every sound in each; a browser plays the first it can). ogg = Vorbis, webm = '
                                    'Opus, m4a = AAC, mp3; wav and flac are lossless. Loops: not mp3 or m4a (their '
                                    'padding leaves a gap at the seam)')
    p.add_argument('--sprite', help='with --spec / --set: also pack the set into this one file with a map')
    p.add_argument('--sound', help='a design id from `earshot sounds` instead of (or on top of) the words')
    p.add_argument('--n', type=int, default=1, help='how many takes (file_1 .. file_n)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--theme', help='a theme preset (soft, glass, wood, felt, tine, metal, electric, digital, retro)')
    p.add_argument('--pitch', help='a note (C5) or Hz')
    p.add_argument('--dur', type=float, help='seconds: how long a figure of notes sounds, the hold of a pad, the '
                                             'length of a loop (each has a shortest one) or a whoosh; a single '
                                             'struck sound (a tap, a hit) rings as it rings: it ignores --dur and '
                                             'says so ("not used by this sound: dur")')
    p.add_argument('--voice', help='the instrument or patch of a tonal sound')
    p.add_argument('--material', help='what a tap, click or knock is made of')
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
                      **_axes(args)), args)
