"""earshot doctor: check the install (python packages, ffmpeg, model files, CPU settings, optional extras).

    earshot doctor [--load] [--json]

Without --load it imports nothing heavy (a few seconds). --load also loads both models, times one embedding (first
load can take ~20-40 s) and reports the precision in use (bf16 only on CPUs with native bf16, else fp32).
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


def check(load=False):
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

    def cached(mid):
        d = os.path.join(config.HF_CACHE, 'models--' + mid.replace('/', '--'))
        return os.path.isdir(d)
    r['models_cached'] = {config.CLAP_MODEL: cached(config.CLAP_MODEL), config.AST_MODEL: cached(config.AST_MODEL)}
    r['llm_keys_present'] = {k: bool(os.environ.get(k)) for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY')}
    data = {f: os.path.exists(os.path.join(config.DATA_DIR, f)) for f in
            ('vocab.json', 'vocab_stats.json', 'realness_model.json', 'launch_profile.json')}
    r['data_files'] = data
    problems = [f'{k}: {v}' for k, v in pk.items() if str(v).startswith('MISSING') and k not in ('mcp', 'requests')]
    if 'MISSING' in str(r['ffmpeg']):
        problems.append('ffmpeg missing')
    problems += [f'data file earshot/data/{f} missing' for f, ok in data.items() if not ok]
    if load:
        import numpy as np
        from .models import ast, clap
        t = time.time()
        c = clap.get()
        r['clap_load_s'] = round(time.time() - t, 1)
        t = time.time()
        c.embed_audio([np.zeros(48000, np.float32)])
        r['clap_one_window_s'] = round(time.time() - t, 2)
        r['clap_precision'] = c.precision
        t = time.time()
        a = ast.get()
        r['ast_load_s'] = round(time.time() - t, 1)
        t = time.time()
        a.probs([np.zeros(48000, np.float32)])
        r['ast_one_window_s'] = round(time.time() - t, 2)
        import torch
        r['native_bf16'] = clap.native_bf16(torch)
    r['problems'] = problems
    r['summary'] = ('earshot looks healthy.' if not problems else 'Problems: ' + '; '.join(problems)) + (
        '' if all(r['models_cached'].values()) else ' Models are not cached yet: the first call downloads them '
        '(~1.4 GB) into ' + config.HF_CACHE + '.')
    return r


def add_cli(sub):
    from .cli import common
    p = common(sub.add_parser('doctor', help='check the install, models and CPU settings'))
    p.add_argument('--load', action='store_true', help='also load both models and time one window')
    p.set_defaults(func=lambda a: emit(check(a.load), a) and 0)
