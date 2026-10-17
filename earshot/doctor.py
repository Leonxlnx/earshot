"""earshot doctor: check the install (python packages, ffmpeg, model files, CPU settings, optional extras).

    earshot doctor [--load] [--fetch] [--json]

Without --load it imports nothing heavy (a few seconds). It reports, per model, the pinned revision, what the cache
holds (cached / legacy / incomplete / missing) and the folder that loads, whether an offline flag is set, and the
sound bank (earshot sfx / kit: its version, the files it loads and how many sounds of each type). --load
also loads both models, times one embedding (first load can take ~20-40 s) and reports the revision, weights file and
precision in use (bf16 only on CPUs with native bf16, else fp32). --fetch downloads the pinned files that are not in
the cache yet (~1.1 GB the first time; refused while an offline flag is set).
"""
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import sys
import time

from . import config
from .report import emit

PACKAGES = [('numpy', 'numpy'), ('scipy', 'scipy'), ('soundfile', 'soundfile'), ('soxr', 'soxr'),
            ('librosa', 'librosa'), ('pyloudnorm', 'pyloudnorm'), ('sklearn', 'scikit-learn'), ('PIL', 'pillow'),
            ('matplotlib', 'matplotlib'), ('torch', 'torch'), ('transformers', 'transformers'), ('mcp', 'mcp'),
            ('requests', 'requests')]


def _models():
    """(model id, revision override or None) for CLAP and AST"""
    return ((config.CLAP_MODEL, config.CLAP_REVISION), (config.AST_MODEL, config.AST_REVISION))


def models():
    """per model id: the revision earshot wants ('local' for a local model folder), what the cache holds and the
    folder that loads. Filesystem only"""
    from .models import hub
    out = {}
    for mid, rev in _models():
        s = hub.resolve(mid, rev)
        out[mid] = {'revision': s.revision, 'cached': s.state, 'loads': s.sha, 'weights': s.weights,
                    'snapshot': s.snapshot, 'missing': list(s.missing)}
    return out


def fetch():
    """download the wanted revision of every model whose files are not all in the cache: {model id: what happened}.
    A model running on a legacy snapshot gets its pinned safetensors. Raises nothing; failures are reported"""
    from .models import hub
    out = {}
    for mid, rev in _models():
        s = hub.resolve(mid, rev)
        if s.state == 'cached':
            out[mid] = 'already cached' if not s.folder else 'a local folder (nothing to download)'
            continue
        try:
            out[mid] = 'downloaded ' + hub.fetch(s).sha[:7]
        except RuntimeError as e:
            out[mid] = f'failed: {e}'
    return out


def bank_info():
    """the sound bank as loaded (earshot.forge.bank): {'version', 'sounds', 'types': {forge, fit, ...}, 'pluck',
    'files': [{file, version, entries}], 'extra': EARSHOT_BANK files, 'error'}. Reads the bank files only"""
    try:
        from .forge import bank
        st = bank._load()
        types, files = {}, {}
        for i, e in st['entries'].items():
            t = e.get('type', 'forge')
            types[t] = types.get(t, 0) + 1
            f = st['where'][i]
            files[f] = files.get(f, 0) + 1
        own = os.path.normcase(os.path.abspath(bank.BANK_DIR))
        rows = [{'file': os.path.basename(f) if os.path.normcase(os.path.dirname(f)) == own else f, 'entries': n}
                for f, n in files.items()]
        return {'version': st['version'], 'sounds': len(st['entries']), 'types': types,
                'pluck': sum(1 for i in st['entries'] if i.startswith('pluck/')), 'files': rows,
                'extra': [p for p in (os.environ.get('EARSHOT_BANK') or '').split(os.pathsep) if p.strip()]}
    except Exception as e:  # noqa: BLE001  (a broken bank file: say so, keep checking the rest)
        return {'error': f'{type(e).__name__}: {e}'}


def _load(r):
    """load both models, time one window each and record what loaded (revision, weights file, precision)"""
    import numpy as np
    from .models import ast, clap
    t = time.time()
    c = clap.get()
    r['clap_load_s'] = round(time.time() - t, 1)
    t = time.time()
    c.embed_audio([np.zeros(48000, np.float32)])
    r['clap_one_window_s'] = round(time.time() - t, 2)
    r['clap_precision'] = c.precision
    r['clap_revision'], r['clap_weights'] = c.revision, c.weights
    t = time.time()
    a = ast.get()
    r['ast_load_s'] = round(time.time() - t, 1)
    t = time.time()
    a.probs([np.zeros(48000, np.float32)])
    r['ast_one_window_s'] = round(time.time() - t, 2)
    r['ast_revision'], r['ast_weights'] = a.revision, a.weights
    import torch
    r['native_bf16'] = clap.native_bf16(torch)


def check(load=False, fetch_missing=False):
    r = {'python': sys.version.split()[0], 'earshot_home': config.HOME, 'threads': config.threads(),
         'precision': config.PRECISION, 'clap_model': config.CLAP_MODEL, 'ast_model': config.AST_MODEL}
    pk = {}
    for name, dist in PACKAGES:  # installed versions from package metadata: no import (torch alone takes ~10-20 s)
        try:
            if importlib.util.find_spec(name) is None:
                raise ImportError(name)
            try:
                pk[name] = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                pk[name] = 'ok'
        except Exception as e:  # noqa: BLE001
            pk[name] = f'MISSING ({type(e).__name__})'
    r['packages'] = pk
    r['ffmpeg'] = shutil.which('ffmpeg') or 'MISSING (install ffmpeg and put it on PATH)'
    r['ffprobe'] = shutil.which('ffprobe') or 'MISSING'
    r['torch_cuda_used'] = False
    r['offline'] = config.offline()   # the flag that keeps earshot off the network, or None
    if fetch_missing:
        r['fetched'] = fetch()
    r['models'] = models()
    r['models_cached'] = {m: v['snapshot'] is not None for m, v in r['models'].items()}
    r['llm_keys_present'] = {k: bool(os.environ.get(k)) for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY')}
    data = {f: os.path.exists(os.path.join(config.DATA_DIR, f)) for f in
            ('vocab.json', 'vocab_stats.json', 'realness_model.json', 'launch_profile.json')}
    r['data_files'] = data
    r['bank'] = bank_info()
    problems = [f'{k}: {v}' for k, v in pk.items() if str(v).startswith('MISSING') and k not in ('mcp', 'requests')]
    if r['bank'].get('error'):
        problems.append(f"sound bank: {r['bank']['error']}")
    if 'MISSING' in str(r['ffmpeg']):
        problems.append('ffmpeg missing')
    problems += [f'data file earshot/data/{f} missing' for f, ok in data.items() if not ok]
    failed = [m for m, v in r.get('fetched', {}).items() if v.startswith('failed')]
    problems += [f"{m}: {r['fetched'][m]}" for m in failed]
    folders = {m for m, v in r['models'].items() if v['revision'] == 'local'}
    problems += [f"{m}: the local model folder lacks {' and '.join(r['models'][m]['missing'])} (earshot never "
                 'downloads into a folder)' for m in r['models']
                 if m in folders and not r['models_cached'][m] and m not in failed]
    notes, todo = [], [m for m, ok in r['models_cached'].items() if not ok and m not in failed and m not in folders]
    if todo and r['offline']:
        problems.append(f"{' and '.join(todo)} not in the cache ({config.HF_CACHE}) and {r['offline']} is set, so "
                        'earshot will not download them: run `earshot doctor --fetch` once without it, or copy a '
                        'cache there')
    elif todo:
        notes.append(f"Not cached yet: {' and '.join(todo)}. The first call that needs them downloads the pinned "
                     f'files (safetensors; ~1.1 GB for both models) into {config.HF_CACHE}, or run '
                     '`earshot doctor --fetch` now.')
    for m, v in r['models'].items():
        if v['cached'] == 'legacy':
            notes.append(f"{m} loads the pickle weights of a cache made before earshot 0.2.1 ({v['loads'][:7]}, "
                         f"{v['weights']}), from the cache only; `earshot doctor --fetch` downloads the pinned "
                         f"safetensors ({v['revision'][:7]}), used from then on.")
    if load and not (r['offline'] and not all(r['models_cached'].values())):   # (cannot load: a problem says why)
        try:
            _load(r)
        except RuntimeError as e:   # a model that cannot load (offline and not cached, a damaged cache): say why
            problems.append(f'model load failed: {e}')
    b, head = r['bank'], []
    if not b.get('error'):
        ty = b['types']
        head.append(f"Sound bank {b['version']}: {b['sounds']} sounds ({ty.get('forge', 0) - b['pluck']} forge, "
                    f"{b['pluck']} plucks, {ty.get('fit', 0)} fitted from CC0 recordings"
                    + (f"; extra files from EARSHOT_BANK: {', '.join(b['extra'])}" if b['extra'] else '') + ').')
    r['problems'], r['notes'] = problems, notes
    r['summary'] = ' '.join(head + ['earshot looks healthy.' if not problems else
                                    'Problems: ' + '; '.join(p.rstrip('.') for p in problems) + '.'] + notes)
    return r


def add_cli(sub):
    from .cli import common
    p = common(sub.add_parser('doctor', help='check the install, models and CPU settings'))
    p.add_argument('--load', action='store_true', help='also load both models and time one window')
    p.add_argument('--fetch', action='store_true', help='download the pinned model files that are not in the cache '
                   '(~1.1 GB the first time; refused while an offline flag is set)')
    p.set_defaults(func=lambda a: emit(check(a.load, a.fetch), a) and 0)
