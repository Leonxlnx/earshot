"""earshot pack: many sounds in one file, with a map of where each one is (an audio sprite).

    earshot pack public/sounds -o public/sounds/sprite.webm          # + sprite.json next to it
    from earshot.pack import pack
    pack(['tap.wav', 'success.wav'], 'sprite.ogg')

A web page or a game that loads one file instead of twenty starts faster and plays every sound without a request.
The map is the format howler.js reads (`{"src": [...], "sprite": {name: [offset_ms, duration_ms, loop]}}`), which
most web audio libraries accept or convert. Sounds are laid end to end with `gap` seconds of silence between them:
lossy codecs smear a little in time, and without a gap the tail of one sound leaks into the start of the next.
Loops (from a manifest.json written by `earshot make --spec / --set`) are marked as loops and get a longer gap.
"""
import json
import os

import numpy as np

from .cli import common
from .report import emit

GAP_S = 0.25
AUDIO = ('.wav', '.flac', '.ogg', '.mp3', '.m4a', '.webm')


def pack(files, out, gap=GAP_S, mono=False, names=None, loops=(), ys=None):
    """lay `files` (paths, or a folder: every audio file in it, by name) end to end into `out` (format by its
    extension, see earshot.synth.master.write; a list of paths writes the same sprite in each format, and the map's
    `src` lists them in that order, the first playable one wins) and write the map next to the first as
    <out stem>.json: {'summary', 'file' (the first), 'files', 'map', 'sprite'}. names: one per file (default: the
    file names without extension); loops: the names that loop; ys: the sounds themselves, one per file, instead of
    reading the files (a set packs what it rendered, not a lossy file decoded again)"""
    from . import audio
    from .synth import master
    outs = [out] if isinstance(out, str) else list(out)
    if isinstance(files, str):
        folder = files
        man, rows, sprites = os.path.join(folder, 'manifest.json'), [], {os.path.basename(o) for o in outs}
        if os.path.isfile(man):
            with open(man, encoding='utf-8') as fh:
                rows = [s for s in json.load(fh).get('sounds', []) if os.path.isfile(os.path.join(folder, s['file']))]
        for f in os.listdir(folder):               # a sprite made earlier in this folder is not a sound of the set
            if f.lower().endswith('.json') and f != 'manifest.json':
                with open(os.path.join(folder, f), encoding='utf-8') as fh:
                    try:
                        doc = json.load(fh)
                    except ValueError:
                        continue
                if isinstance(doc, dict) and 'sprite' in doc:
                    sprites.update(doc.get('src') or [])
        if rows:                                   # a set: exactly its sounds, in its order
            files = [os.path.join(folder, s['file']) for s in rows if s['file'] not in sprites]
            loops = loops or [s['name'] for s in rows if s.get('loop')]
        else:
            files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(AUDIO) and f not in sprites)
    if not files:
        raise ValueError('nothing to pack: no audio files')
    names = list(names) if names else [os.path.splitext(os.path.basename(f))[0] for f in files]
    if len(set(names)) != len(names):
        raise ValueError('two sounds of a sprite share a name')
    parts, sprite, t = [], {}, 0.0
    for k, (f, name) in enumerate(zip(files, names)):
        y = np.asarray(audio.load(f) if ys is None else ys[k], np.float64)
        y = y if y.ndim == 2 else np.stack([y, y], 1)
        loop = name in loops
        sprite[name] = [round(t * 1000), round(len(y) / master.SR * 1000)] + ([True] if loop else [])
        pad = np.zeros((int(round((gap * 4 if loop else gap) * master.SR)), 2))
        parts += [y, pad]
        t += (len(y) + len(pad)) / master.SR
    y = np.concatenate(parts[:-1])
    for o in outs:
        master.write(o, y, mono=mono, bits=16)
    mp = os.path.splitext(outs[0])[0] + '.json'
    with open(mp, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'src': [os.path.basename(o) for o in outs], 'sprite': sprite}, fh, indent=1)
        fh.write('\n')
    kb = ', '.join(f'{os.path.getsize(o) / 1024:.0f} kB' for o in outs)
    shown = outs[0] if len(outs) == 1 else ' + '.join([outs[0]] + [os.path.basename(o) for o in outs[1:]])
    return {'summary': f'{len(files)} sounds in {shown} ({t - gap:.1f} s, {kb}), map {os.path.basename(mp)}',
            'file': outs[0], 'files': outs, 'map': mp, 'sprite': sprite}


def add_cli(sub):
    p = sub.add_parser('pack', help='many sounds in one file with a map (an audio sprite for the web and games)')
    p.add_argument('files', nargs='+', help='audio files, or one folder (a manifest.json in it marks the loops)')
    p.add_argument('-o', '--output', required=True, help='the sprite file (wav, ogg, mp3, m4a, webm ...)')
    p.add_argument('--gap', type=float, default=GAP_S, help='seconds of silence between sounds')
    p.add_argument('--mono', action='store_true')
    common(p)
    p.set_defaults(func=_run)


def _run(args):
    one = len(args.files) == 1 and os.path.isdir(args.files[0])
    res = pack(args.files[0] if one else args.files, args.output, args.gap, args.mono)
    emit(res, args, text=res['summary'])
