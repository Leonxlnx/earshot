"""Compute the realness features of every clip in out/realness/manifest.json (resumable and time-boxed).

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/extract.py [--budget 480] [--what signal,clap] [--ast N]

Writes out/realness/feats/signal.json (id -> earshot.realness.signal_features, None when silent), clap.npz (ids,
emb: the same CLAP call as earshot.realness at inference) and, with --ast N, ast.npz for a fixed stratified subset of
about N clips (AudioSet probabilities; AST costs ~5x CLAP on this CPU, so it is a baseline / optional head only).
Run it again until it prints "done": each run stops after --budget seconds and keeps what it computed.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from earshot import realness  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'out', 'realness')
FEATS = os.path.join(OUT, 'feats')


def all_clips():
    man = json.load(open(os.path.join(OUT, 'manifest.json')))
    return [c for part in man.values() for c in part]


def load_clip(c):
    x, _ = sf.read(os.path.join(OUT, c['wav']), dtype='float32')
    return x if x.ndim == 1 else x.mean(axis=1)


def ast_subset(clips, n):
    """deterministic stratified subset: the same share of every (label, group, type) cell, picked by id hash"""
    cells = {}
    for c in clips:
        if 'perturb' not in c:
            cells.setdefault((c['label'], c['group'], c['type']), []).append(c)
    frac = n / max(1, sum(len(v) for v in cells.values()))
    out = []
    for k in sorted(cells, key=str):
        v = sorted(cells[k], key=lambda c: hashlib.sha1(c['id'].encode()).hexdigest())
        out += v[: max(1, int(round(frac * len(v))))]
    return out


def save_npz(path, ids, arr):
    tmp = path + '.tmp.npz'
    np.savez(tmp, ids=np.array(ids), emb=np.asarray(arr, np.float32))
    os.replace(tmp, path)


def load_npz(path):
    if not os.path.exists(path):
        return [], []
    z = np.load(path, allow_pickle=False)
    return list(z['ids']), list(z['emb'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--budget', type=float, default=480)
    ap.add_argument('--what', default='signal,clap')
    ap.add_argument('--ast', type=int, default=0, help='AST probabilities for a stratified subset of ~N clips')
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(FEATS, exist_ok=True)
    clips = all_clips()
    what = a.what.split(',')
    if 'signal' in what:
        sp = os.path.join(FEATS, 'signal.json')
        sig = json.load(open(sp)) if os.path.exists(sp) else {}
        todo = [c for c in clips if c['id'] not in sig]
        for c in todo:
            x = load_clip(c)
            sig[c['id']] = realness.signal_features(x) if realness.audible(x) else None
        json.dump(sig, open(sp, 'w'))
        print(f'signal: {len(todo)} new, {len(sig)} total ({time.time() - t0:.0f} s)')
    jobs = []
    if 'clap' in what:
        jobs.append(('clap', clips))
    if a.ast:
        jobs.append(('ast', ast_subset(clips, a.ast)))
    left = 0
    for name, pool in jobs:
        path = os.path.join(FEATS, f'{name}.npz')
        ids, arr = load_npz(path)
        todo = [c for c in pool if c['id'] not in set(ids)]
        if todo and time.time() - t0 < a.budget:
            from earshot.models import ast, clap
            model = clap.get() if name == 'clap' else ast.get()
            fn = model.embed_audio if name == 'clap' else model.probs
            if name == 'ast':
                json.dump(model.labels, open(os.path.join(FEATS, 'ast_labels.json'), 'w'))
            print(f'{name}: {len(todo)} to do, model ready after {time.time() - t0:.0f} s', flush=True)
            while todo and time.time() - t0 < a.budget:
                chunk, todo = todo[:a.batch], todo[a.batch:]
                ids += [c['id'] for c in chunk]
                arr += list(fn([load_clip(c) for c in chunk]))
                save_npz(path, ids, arr)
        left += len(todo)
        print(f'{name}: {len(ids)} stored, {len(todo)} left ({time.time() - t0:.0f} s)', flush=True)
    print('done' if not left else 'run again to continue')


if __name__ == '__main__':
    main()
