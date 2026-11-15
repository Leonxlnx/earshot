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
a call takes about a second. `--listen` adds earshot's ears (the models, 20-40 s once): the best three designs are
rendered and the one heard closest to the request is kept, with its verdict in the result.

The result says what was chosen and why, what else would fit (`also`), and which words were not understood, so one
call is usually enough. Everything rendered is the caller's to use, in any product, without credit.
"""
import json
import os

from . import brief
from .cli import common
from .report import emit, read_json

SET_FIELDS = ('name', 'want', 'sound', 'pitch', 'dur', 'voice', 'material', 'seed', 'n', 'axes', 'lufs')
TRIES = 4


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


def choose(want, sound=None, top=4, **over):
    """(brief, candidates) of a request: the brief with explicit arguments laid over it, and the ranked designs
    ([{'id', 'score', 'why'}]); `sound` names a design and skips the ranking"""
    from .sounds import registry
    b = brief.read(want or '')
    for k in ('pitch', 'dur', 'voice', 'material'):
        if over.get(k) is not None:
            b[k] = over[k]
    for a, v in (over.get('axes') or {}).items():
        b['axes'][a] = max(-1.0, min(1.0, b['axes'].get(a, 0.0) + float(v)))
    if sound:
        return b, [{'id': registry.get(sound).id, 'score': None, 'why': ['named']}]
    return b, brief.rank(b, top=top)


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
    """make the sound a request asks for and write it: {'summary', 'sound', 'files', 'why', 'also', 'unknown',
    'takes': [meta ...]} (+ 'heard' with listen). want: the request in words; sound: a design id instead of (or on
    top of) the words; out: the file to write (its extension picks the format: wav, flac, ogg, mp3, m4a, webm;
    none = `<sound>.wav` in the working folder); n takes get `_1 .. _n` before the extension; theme: a preset name
    or dict (earshot.sounds.theme); lufs: a momentary-max loudness instead of the -1 dBFS peak; pitch / dur / voice /
    material and the axes (bright, hard, ring, size, space, width, energy, motion) override what the words say."""
    from .sounds import registry
    from .synth import master
    if not (want or sound):
        raise ValueError('say what sound you want (want=...) or name one (sound=...)')
    b, cands = choose(want, sound, pitch=pitch, dur=dur, voice=voice, material=material, axes=axes)
    if loop is False:
        b['loop'] = False
    if not cands:
        raise ValueError(f'no sound in the catalogue answers {want!r}: `earshot sounds` lists what there is; name '
                         'the sound itself (a chime, a click, a whoosh, rain) or the job it plays (success, coin, logo)')
    n = int(b['n'] or n)
    req = _request(b, theme)
    heard = None
    if listen and len(cands) > 1 and not sound:
        import tempfile
        best = None
        with tempfile.TemporaryDirectory() as tmp:
            for c in cands[:3]:
                y, meta = take(c['id'], seed, req, lufs)
                p = master.write(os.path.join(tmp, 'c.wav'), y)
                h = _heard(p, [want] + list(registry.get(c['id']).says))
                if best is None or h['z'] > best[0]['z']:
                    best = (h, c)
        heard, pick = best
    else:
        pick = cands[0]
    paths = _paths(out or pick['id'].replace('/', '_') + '.' + (fmt or 'wav'), n, fmt)
    takes = []
    for i, p in enumerate(paths):
        y, meta = take(pick['id'], seed + i, req, lufs)
        master.write(p, y, sr=sr, mono=mono, bits=bits)
        takes.append({**meta, 'file': p})
    if listen and heard is None:
        heard = _heard(paths[0] if paths[0].lower().endswith(('.wav', '.flac')) else _tmp_wav(pick['id'], seed, req),
                       ([want] if want else []) + list(registry.get(pick['id']).says))
    why = ', '.join(dict.fromkeys(pick['why']))
    lines = [_line(t, t['file'], why if i == 0 else '') for i, t in enumerate(takes)]
    also = [c['id'] for c in cands if c['id'] != pick['id']][:3]
    if also:
        lines.append('also: ' + ', '.join(also))
    if heard:
        lines.append(f"heard as: {heard['query']!r} (z {heard['z']:+.1f}, {heard['verdict']})")
    ign = sorted(set().union(*(set(t.get('ignored') or ()) for t in takes)))
    if ign:
        lines.append(f"not used by this sound: {', '.join(ign)}")
    unknown = [w for w in b['unknown'] if brief.stem(w) not in pick['why']]      # words that found nothing
    if unknown:
        lines.append(f"not understood: {', '.join(unknown)}")
    res = {'summary': '\n'.join(lines), 'sound': pick['id'], 'files': paths, 'why': pick['why'], 'also': also,
           'unknown': unknown, 'takes': takes}
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
        {"set": "ui", "theme": "glass" | {...}, "format": "ogg", "out": "public/sounds", "mono": true, "lufs": -18,
         "sprite": "sprite.webm",
         "sounds": [{"name": "tap", "want": "a soft tap for buttons"},
                    {"name": "success", "want": "a happy two-note success chime", "dur": 0.5, "axes": {"bright": 0.2}}]}
    "set" starts from a ready-made list; "sounds" adds to it, or changes the sound of the same name. Each sound
    takes name (the file name) and want (words) and / or sound (a design id), plus pitch, dur, voice, material,
    seed, n, axes, lufs. Every sound shares the theme (one voice, key and room). The sounds are levelled against
    each other by loudness (`lufs`: the loudest allowed; default -16 for the set's loudest sound, each design's
    own weight under it), `manifest.json` in the folder lists every file with its sound, length and level, and
    "sprite" also packs them into one file with a map (earshot.pack)."""
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
    fmt = (fmt or spec.get('format') or 'wav').lstrip('.')
    mono = spec.get('mono', False) if mono is None else mono
    top = lufs if lufs is not None else spec.get('lufs', -16.0)
    rows, lines = [], []
    for it in items:
        b, cands = choose(it.get('want'), it.get('sound'), pitch=it.get('pitch'), dur=it.get('dur'),
                          voice=it.get('voice'), material=it.get('material'), axes=it.get('axes'))
        if not cands:
            raise ValueError(f'{it["name"]}: no sound in the catalogue answers {it.get("want")!r}')
        d = registry.get(cands[0]['id'])
        level = it.get('lufs', top + WEIGHT.get(d.kind, -4.0))
        k = int(it.get('n') or b['n'] or 1)
        for p in _paths(os.path.join(out, f'{it["name"]}.{fmt}'), k, fmt):
            y, meta = take(d.id, int(it.get('seed', seed)) + len(rows), _request(b, theme), lufs=level)
            master.write(p, y, mono=mono and not d.loop, bits=16 if fmt == 'wav' else 24)
            rows.append({'name': os.path.splitext(os.path.basename(p))[0], 'file': os.path.basename(p),
                         'sound': d.id, 'dur_s': meta['dur_s'], 'lufs_max': meta['lufs_max'], 'loop': d.loop,
                         'ok': meta['checks']['ok'], **({'pitch': meta['pitch']} if meta.get('pitch') else {})})
            lines.append(_line(meta, p, ', '.join(dict.fromkeys(cands[0]['why']))))
            if b['unknown']:
                lines.append(f"  not understood: {', '.join(b['unknown'])}")
    man = os.path.join(out, 'manifest.json')
    with open(man, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'theme': theme, 'format': fmt, 'sounds': rows}, fh, indent=1)
        fh.write('\n')
    head = f'{len(rows)} sounds in {out} ({fmt}), manifest.json'
    res = {'files': [os.path.join(out, r['file']) for r in rows], 'manifest': man, 'sounds': rows}
    sprite = sprite or spec.get('sprite')
    if sprite:
        from .pack import pack
        sp = pack(res['files'], os.path.join(out, os.path.basename(sprite)), mono=mono,
                  loops=[r['name'] for r in rows if r['loop']])
        lines.append(sp['summary'])
        res['sprite'] = sp['file']
    if skipped:
        lines.append(f"not in this catalogue, left out: {', '.join(skipped)}")
    return {'summary': '\n'.join([head] + lines), **res}


# a kind's level under the loudest sound of a set (LU): the more often a sound plays, the quieter it sits
WEIGHT = {'tap': -10.0, 'tick': -12.0, 'click': -9.0, 'pop': -7.0, 'note': -5.0, 'chime': -3.0, 'figure': -2.0,
          'hit': -1.0, 'thud': -3.0, 'whoosh': -5.0, 'rise': -3.0, 'fall': -3.0, 'zap': -3.0, 'boom': 0.0,
          'noise': -8.0, 'texture': -12.0, 'bed': -14.0, 'pad': -10.0, 'stinger': 0.0, 'voice': -5.0, 'object': -6.0}


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
    p.add_argument('--spec', help='a JSON file describing a whole set (earshot.make.make_set)')
    p.add_argument('--set', dest='set_name', help='a ready-made set: ui, game, arcade or film (one family, one call)')
    p.add_argument('--format', help='with --spec / --set: the file format of the set (wav, ogg, mp3, m4a, webm, flac)')
    p.add_argument('--sprite', help='with --spec / --set: also pack the set into this one file with a map')
    p.add_argument('--sound', help='a design id from `earshot sounds` instead of (or on top of) the words')
    p.add_argument('--n', type=int, default=1, help='how many takes (file_1 .. file_n)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--theme', help='a theme preset (soft, glass, wood, felt, tine, metal, electric, digital, retro)')
    p.add_argument('--pitch', help='a note (C5) or Hz')
    p.add_argument('--dur', type=float, help='length in seconds')
    p.add_argument('--voice', help='the instrument or patch of a tonal sound')
    p.add_argument('--material', help='what a tap, click or knock is made of')
    for a in ('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy', 'motion'):
        p.add_argument(f'--{a}', type=float, help='-1 .. 1, 0 = as designed')
    p.add_argument('--lufs', type=float, help='momentary-max loudness instead of a -1 dBFS peak')
    p.add_argument('--mono', action='store_true')
    p.add_argument('--no-loop', action='store_true', help='an ambience once, with a fade in and out, instead of a '
                                                          'seamless loop')
    p.add_argument('--listen', action='store_true', help='let the models pick among the best three and say what '
                                                         'the result is heard as (loads them: 20-40 s)')
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
