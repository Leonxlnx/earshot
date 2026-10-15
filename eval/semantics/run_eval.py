"""Evaluate earshot semantics (calibration, pad mode, prompt ensembling, verdicts, find, query kind) on real recordings.

Data (read-only, never copied):
  tuning library  - the 219 labelled recordings in <assets>/<category>/ + manifests (split by
                    file id into dev / test halves). Variants, the verdict thresholds and find's heuristics were chosen
                    on it, and 67 vocabulary terms were added after seeing its classes: numbers on it are optimistic.
  held-out library - the 126 rejected-for-quality real recordings in .../assets/_rejected/<category>/, labelled from
                    folder + file name only, BEFORE any score on them was computed (heldout_labels.json). Nothing was
                    chosen on it: these are the honest numbers.
  term-bias reference sets (unsupervised: only their mean embedding is used) - for the tuning analysis: the held-out
                    library + 7 pro launch films + the Clawd soundtrack / stems + retired synth stems (disjoint from the
                    219 files, but the Clawd mixes contain some of them); shipped (data/vocab_stats.json): the 219
                    tuning files + films + Clawd mixes + synth stems (disjoint from the held-out library).
Queries: queries.json (written from the manifest needs before scoring). Run from the repo root (EARSHOT_THREADS=2; data: eval/README.md):

    .venv/Scripts/python.exe eval/semantics/run_eval.py embed --set eval|tuneref|heldout [--pad P] [--part 1/3]
    .venv/Scripts/python.exe eval/semantics/run_eval.py stats      # -> earshot/data/vocab_stats.json (shipped refs)
    .venv/Scripts/python.exe eval/semantics/run_eval.py analyze    # tuning library -> results.json (variants, thresholds)
    .venv/Scripts/python.exe eval/semantics/run_eval.py heldout    # held-out library, shipped config -> heldout.json
    .venv/Scripts/python.exe eval/semantics/run_eval.py kind       # find's query-kind reader -> kind.json
    .venv/Scripts/python.exe eval/semantics/run_eval.py find       # find() on labelled key presses + 2 built scenes
    .venv/Scripts/python.exe eval/semantics/run_eval.py timing     # wall-clock per call (warm models, cold caches)
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from earshot import audio, match, vocab  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'out', 'semantics')
CLAWD = EP.FILM
ASSETS = EP.ASSETS
CATS = ['ambience', 'foley', 'garden', 'keyboard', 'magic', 'motion', 'music', 'toy']
FILMS = [EP.ref(rid) for rid in EP.REF_IDS]   # the 7 reference launch films
REF_MAX_WIN = 3
TEMPLATES = {'t1': ['{}'], 't3': ['{}', 'the sound of {}', 'a recording of {}'],
             't5': ['{}', 'the sound of {}', 'a recording of {}', '{} sound effect', 'an audio clip of {}']}
# classes a file can legitimately be described by although its label is another one: 'lenient' scoring leaves these
# pairs out, 'strict' scoring counts them as wrong (an agent asking for a spacebar does not want an Enter key)
CONFUSABLE = [{'snap_lock', 'toy_clicks', 'latch', 'lego', 'switch', 'soft_landing', 'dominoes'},
              {'tiny_steps', 'soft_landing', 'cursor_click', 'key_thock', 'spacebar', 'enter', 'typing'},
              {'wind_grass', 'tree', 'grass_steps', 'meadow', 'tiny_crowd', 'fruit_drop'},
              {'small_whoosh', 'big_whoosh'}, {'fruit_drop', 'soft_landing'}, {'note', 'ting', 'chime_sweep'},
              {'swell', 'chime_sweep', 'impact'}, {'small_pop', 'stretch_pop'}, {'plunge', 'underwater'},
              {'room_tone', 'rain'}]
NEG_MOODS = ['dark and tense music', 'sad and melancholic music', 'mysterious music', 'aggressive music',
             'suspenseful build-up music']


def _json(name):
    with open(os.path.join(HERE, name), encoding='utf-8') as fh:
        return json.load(fh)


def Q():
    return _json('queries.json')


def _write(name, obj):
    with open(os.path.join(HERE, name), 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1)
    print('wrote', name)


def _split(fid):
    return 'dev' if int(hashlib.sha1(fid.encode()).hexdigest(), 16) % 2 == 0 else 'test'


def eval_set():
    """[{id, path, cat, needs, classes, title, split}] for every media file in the 8 tuning asset folders"""
    q = Q()
    need2cls = {n: c for c, d in q['classes'].items() for n in d['needs']}
    by_id, by_name = {}, {}
    for m in glob.glob(f'{ASSETS}/manifests/*.json'):
        j = json.load(open(m, encoding='utf-8'))
        if isinstance(j, list):
            for e in j:
                if 'need' in e and 'file' in e and e.get('category'):
                    by_id.setdefault(e['id'], []).append(e)
                    by_name.setdefault(os.path.basename(e['file']), []).append(e)
    out = []
    for cat in CATS:
        for p in audio.list_media(f'{ASSETS}/{cat}', recursive=False):
            es = by_name.get(os.path.basename(p), [])
            fid = es[0]['id'] if es else os.path.basename(p).split('_')[0]
            es = by_id.get(fid, es)
            needs = sorted({e['need'] for e in es})
            cls = sorted({'music'} if cat == 'music' else {need2cls[n] for n in needs if n in need2cls})
            out.append({'id': fid, 'path': p, 'cat': cat, 'needs': needs, 'classes': cls,
                        'title': es[0].get('title') if es else None, 'split': _split(fid)})
    return out


def heldout_set():
    """the _rejected library with the labels of heldout_labels.json (the lego file in foley/ and toy/ counted once)"""
    lab = _json('heldout_labels.json')
    out, seen = [], set()
    for cat in ['toy'] + [c for c in CATS if c != 'toy']:
        for p in audio.list_media(f'{ASSETS}/_rejected/{cat}', recursive=False):
            fid = os.path.basename(p).split('_')[0]
            if os.path.basename(p) in seen:
                continue
            seen.add(os.path.basename(p))
            cls = ['music'] if cat == 'music' else lab['classes'][fid]
            out.append({'id': fid, 'path': p, 'cat': cat, 'classes': cls, 'split': 'heldout'})
    return sorted(out, key=lambda f: (CATS.index(f['cat']), f['path']))


def _others():
    p = sorted(glob.glob(f'{CLAWD}/out/audio/stems/*.wav')) + [f'{CLAWD}/audio/_retired_synth/bed.wav',
                                                              f'{CLAWD}/out/audio/melody.wav',
                                                              f'{CLAWD}/out/audio/ambience.wav']
    p += [f'{CLAWD}/out/audio/soundtrack.wav'] + sorted(glob.glob(f'{CLAWD}/out/audio/stems_pro/*.wav'))
    p += FILMS
    return [x for x in p if os.path.exists(x)]


def tune_refs():
    """term-bias references for the tuning analysis (disjoint from the 219 tuning files)"""
    return audio.list_media(f'{ASSETS}/_rejected') + _others()


def ship_refs():
    """term-bias references of the shipped stats (disjoint from the held-out library): (paths, max_win per path)"""
    ev = [f['path'] for f in eval_set()]
    ot = _others()
    return ev + ot, [vocab.MAX_WIN] * len(ev) + [REF_MAX_WIN] * len(ot)


def embed_refs(paths, wins, pad=vocab.PAD):
    A = np.zeros((len(paths), 512), np.float32)
    for w in sorted(set(wins)):
        idx = [i for i, x in enumerate(wins) if x == w]
        A[idx] = vocab.embed_files([paths[i] for i in idx], pad=pad, max_win=w)
    return A


def embeddings(which, pad):
    if which == 'eval':
        return vocab.embed_files([f['path'] for f in eval_set()], pad=pad)
    if which == 'heldout':
        return vocab.embed_files([f['path'] for f in heldout_set()], pad=pad)
    return vocab.embed_files(tune_refs(), pad=pad, max_win=REF_MAX_WIN)


def cmd_embed(a):
    paths = {'eval': lambda: [f['path'] for f in eval_set()], 'heldout': lambda: [f['path'] for f in heldout_set()],
             'tuneref': tune_refs}[a.set]()
    i, n = map(int, a.part.split('/'))
    paths = paths[(i - 1) * len(paths) // n: i * len(paths) // n]
    t = time.time()
    vocab.embed_files(paths, pad=a.pad, max_win=REF_MAX_WIN if a.set == 'tuneref' else vocab.MAX_WIN)
    print(f'{a.set} {a.pad} part {a.part}: {len(paths)} files in {time.time() - t:.1f} s')


# ---------------------------------------------------------------------------------------------------- metrics
def ap(scores, rel):
    o = np.argsort(-scores)
    r = rel[o]
    if not r.any():
        return None
    hits = np.cumsum(r)
    return float((hits[r] / (np.nonzero(r)[0] + 1)).mean())


def prec_at(scores, rel, k):
    return float(rel[np.argsort(-scores)[:k]].mean())


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pair_labels(files, cls_list, strict=False):
    """(files x classes) +1 positive, -1 negative, 0 left out (confusable unless strict; files without a class)"""
    L = -np.ones((len(files), len(cls_list)), int)
    for i, f in enumerate(files):
        amb = set()
        for g in CONFUSABLE:
            if g & set(f['classes']):
                amb |= g
        for j, c in enumerate(cls_list):
            if c in f['classes']:
                L[i, j] = 1
            elif not f['classes'] or (c in amb and not strict):
                L[i, j] = 0
    return L


def specific_rel(spec, files):
    if 'classes' in spec:
        return np.array([bool(set(spec['classes']) & set(f['classes'])) for f in files])
    if 'ids' in spec:
        return np.array([f['id'] in spec['ids'] for f in files])
    return np.array([any(f['id'].startswith(p) for p in spec['ids_prefix']) for f in files])


def class_pairs(files, Z, cls, strict):
    """pooled (z, label) over both phrasings of every class query; Z = (files, 2 * classes) in phrasing-major order"""
    L = pair_labels(files, cls, strict)
    PL = np.concatenate([L, L], axis=1)
    return Z[PL != 0], PL[PL != 0]


def evaluate(files, A, V, method, specific=True):
    """retrieval + tagging + pooled separability for one (embeddings, vocab/templates/stats, method)"""
    q = Q()
    cats = list(q['categories'])
    cls = list(q['classes'])
    texts = [q['categories'][c] for c in cats] + [q['classes'][c]['q'][k] for k in (0, 1) for c in cls] + \
        [s['q'] for s in q['specific']]
    r = V.calibrate(A, V.embed_text(texts), method)
    Z = r['z_q']
    nc, ncl = len(cats), len(cls)
    out = {}
    rows = []
    for j, c in enumerate(cats):
        rel = np.array([f['cat'] == c for f in files])
        rows.append({'cat': c, 'ap': ap(Z[:, j], rel), 'p5': prec_at(Z[:, j], rel, 5), 'p10': prec_at(Z[:, j], rel, 10)})
    rows = [x for x in rows if x['ap'] is not None]
    out['category'] = {'map': float(np.mean([x['ap'] for x in rows])), 'p5': float(np.mean([x['p5'] for x in rows])),
                       'p10': float(np.mean([x['p10'] for x in rows])), 'rows': rows}
    aps, top3 = [], []
    for k in (0, 1):
        for j, c in enumerate(cls):
            rel = np.array([c in f['classes'] for f in files])
            if not rel.any():
                continue
            s = Z[:, nc + k * ncl + j]
            aps.append(ap(s, rel))
            top3.append(bool(rel[np.argsort(-s)[:3]].any()))
    out['class'] = {'map': float(np.mean([x for x in aps if x is not None])), 'top3': float(np.mean(top3)),
                    'n_queries': len(aps)}
    if specific:
        spec = []
        for j, s in enumerate(q['specific']):
            rel = specific_rel(s, files)
            if not rel.any():
                continue
            sc = Z[:, nc + 2 * ncl + j]
            o = np.argsort(-sc)
            spec.append({'q': s['q'], 'top1': bool(rel[o[0]]), 'top3': bool(rel[o[:3]].any()), 'ap': ap(sc, rel),
                         'best': os.path.basename(files[o[0]]['path'])[:50]})
        out['specific'] = {'top1': float(np.mean([x['top1'] for x in spec])),
                           'top3': float(np.mean([x['top3'] for x in spec])),
                           'map': float(np.mean([x['ap'] for x in spec])), 'rows': spec}
    U = r['u_q']  # tagging: which class description wins for each file (per-sound z cannot change the order)
    acc, mrr = [], []
    for k in (0, 1):
        S = U[:, nc + k * ncl: nc + (k + 1) * ncl]
        for i, f in enumerate(files):
            if not f['classes']:
                continue
            o = list(np.argsort(-S[i]))
            rk = min(o.index(cls.index(c)) for c in f['classes']) + 1
            acc.append(rk == 1)
            mrr.append(1 / rk)
    out['tagging'] = {'acc': float(np.mean(acc)), 'mrr': float(np.mean(mrr))}
    Zc = Z[:, nc:nc + 2 * ncl]
    for strict in (False, True):
        z, lab = class_pairs(files, Zc, cls, strict)
        out['pooled_auc_strict' if strict else 'pooled_auc'] = auc(z[lab == 1], z[lab == -1])
    out['_Zc'] = Zc
    return out


def _row(ev):
    r = {'cat_map': ev['category']['map'], 'cat_p5': ev['category']['p5'], 'cat_p10': ev['category']['p10'],
         'class_map': ev['class']['map'], 'class_top3': ev['class']['top3'], 'tag_acc': ev['tagging']['acc'],
         'tag_mrr': ev['tagging']['mrr'], 'pooled_auc': ev['pooled_auc'], 'pooled_auc_strict': ev['pooled_auc_strict']}
    if 'specific' in ev:
        r.update(spec_top1=ev['specific']['top1'], spec_top3=ev['specific']['top3'])
    return r


def _clean(ev):
    return {k: v for k, v in ev.items() if not k.startswith('_')}


class SdVocab(vocab.Vocab):
    """the rejected variants: (cos - mu) / sd, with mu, sd the exact (folder-weighted) mean / std of each description's
    cosine over the reference recordings, optionally followed by the per-sound z"""
    def __init__(self, templates, stats, R, paths, per_sound):
        super().__init__(templates, stats)
        w = vocab.group_weights(paths)
        self.R, self.w, self.per_sound = R, w / w.sum(), per_sound
        self.sd_E = self._sd(self.E)

    def _sd(self, Q):
        C = self.R @ np.atleast_2d(Q).T
        return np.sqrt(self.w @ (C - self.w @ C) ** 2) + 1e-9

    def calibrate(self, A, Q=None, method=None):
        A = np.atleast_2d(A)
        C = A @ self.E.T
        U = (C - self.mu_E) / self.sd_E
        mu, sd = (U.mean(1, keepdims=True), U.std(1, keepdims=True) + 1e-9) if self.per_sound else (0.0, 1.0)
        r = {'cos_t': C, 'u_t': U, 'z_t': (U - mu) / sd}
        if Q is not None:
            Cq = A @ np.atleast_2d(Q).T
            Uq = (Cq - self.bias(Q)) / self._sd(Q)
            r.update(cos_q=Cq, u_q=Uq, z_q=(Uq - mu) / sd, pct_q=(U[:, None, :] < Uq[:, :, None]).mean(-1))
        return r


# ---------------------------------------------------------------------------------------------------- verdicts
def grid(z, lab, step=0.25):
    """cumulative (z >= t) recall / false-positive rate / likelihood ratio / precision at the data's base rate"""
    npos, nneg = max(1, int((lab == 1).sum())), max(1, int((lab == -1).sum()))
    out = []
    for t in np.arange(0.0, 6.01, step):
        sel = z >= t
        tp, fp = int(((lab == 1) & sel).sum()), int(((lab == -1) & sel).sum())
        out.append({'z': round(float(t), 2), 'tp': tp, 'fp': fp, 'recall': round(tp / npos, 4),
                    'fpr': round(fp / nneg, 5), 'lr': round((tp / npos) / max(fp / nneg, 1e-9), 1) if fp else None,
                    'precision': round(tp / max(1, tp + fp), 4)})
    return out


def bands(z, lab, cuts):
    """per verdict band (not cumulative): pairs in it, likelihood ratio LR = P(band | true) / P(band | false),
    precision at the data's base rate and at even prior odds (LR / (LR + 1)), cumulative recall"""
    npos, nneg = max(1, int((lab == 1).sum())), max(1, int((lab == -1).sum()))
    edges = [np.inf] + list(cuts) + [-np.inf]
    names = [n for _, n in match.VERDICTS] + ['no match']
    out = {'n_pos': int((lab == 1).sum()), 'n_neg': int((lab == -1).sum()),
           'base_rate': round(npos / (npos + nneg), 4)}
    for name, hi, lo in zip(names, edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        tp, fp = int(((lab == 1) & m).sum()), int(((lab == -1) & m).sum())
        lr = (tp / npos) / (fp / nneg) if fp else None
        out[name] = {'z_range': [None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi],
                     'tp': tp, 'fp': fp, 'lr': None if lr is None else round(lr, 2),
                     'precision': round(tp / max(1, tp + fp), 3),
                     'precision_even_odds': None if lr is None else round(lr / (lr + 1), 3),
                     'recall_at_or_above': round(float(((lab == 1) & (z >= lo)).sum()) / npos, 3)}
    return out


RULE = ('on the 219 tuning files (lenient labels), 0.5 z grid, top-down: strong = lowest t with LR(z >= t) >= 100; '
        'plausible = lowest t below it with band LR >= 20; weak = lowest t below that with band LR >= 5')


def _band_lr(z, lab, lo, hi):
    npos, nneg = max(1, int((lab == 1).sum())), max(1, int((lab == -1).sum()))
    m = (z >= lo) & (z < hi)
    tp, fp = int(((lab == 1) & m).sum()), int(((lab == -1) & m).sum())
    return (tp / npos) / (fp / nneg) if fp else (np.inf if tp else 0.0)


def pick_thresholds(z, lab):
    """the verdict thresholds by RULE"""
    out, hi = {}, np.inf
    for name, lr_min in (('strong', 100), ('plausible', 20), ('weak', 5)):
        t = None
        for c in np.arange(6.0, -0.01, -0.5):
            if _band_lr(z, lab, c, hi) >= lr_min:
                t = float(c)
            elif t is not None:
                break
        out[name] = t
        hi = t if t is not None else hi
    return out


def verdict_tables(files, Zc, cls, cuts):
    return {('strict' if s else 'lenient'): bands(*class_pairs(files, Zc, cls, s), cuts) for s in (False, True)}


def mood_check(V, A, files):
    """music only: share of tracks whose best music_mood tag is a dark / sad / tense one. Every track in both libraries
    was picked for a bright, playful animated film (titles: happy, playful, cheer up, delighted, dreaming big ...),
    so a dark top mood is very likely wrong"""
    moods = V.cats['music_mood']
    idx = [V.names.index(t) for t in moods]
    mus = [i for i, f in enumerate(files) if f['cat'] == 'music']
    Z = V.calibrate(A[mus])['z_t'][:, idx]
    top = [moods[i] for i in np.argmax(Z, axis=1)]
    neg = [t in NEG_MOODS for t in top]
    return {'n_tracks': len(mus), 'top_mood_negative_share': round(float(np.mean(neg)), 3),
            'chance_share': round(len(NEG_MOODS) / len(moods), 3),
            'top_moods': {t: top.count(t) for t in sorted(set(top), key=top.count, reverse=True)}}


def cmd_analyze(a):
    """tuning library: every variant (pad x templates x calibration) on all 219 files and per split, the verdict
    thresholds and the reference-set sensitivity"""
    files = eval_set()
    refs = tune_refs()
    sp = np.array([f['split'] for f in files])
    cls = list(Q()['classes'])
    out = {'n_eval': len(files), 'n_ref': len(refs), 'n_vocab': len(vocab.Vocab().names),
           'chosen': {'pad': vocab.PAD, 'templates': a.templates, 'method': 'both'}, 'variants': []}
    t0 = time.time()
    for pad in vocab.PADS:
        A, R = embeddings('eval', pad), embeddings('tuneref', pad)
        for tk, tp in TEMPLATES.items():
            st = vocab.build_stats(R, refs, path=None, templates=tp)
            V = vocab.Vocab(tp, stats=st)
            vs = [(m, V, m) for m in vocab.METHODS] + [('term_sd', SdVocab(tp, st, R, refs, False), None),
                                                      ('both_sd', SdVocab(tp, st, R, refs, True), None)]
            for name, VV, method in vs:
                ev = evaluate(files, A, VV, method)
                row = dict(pad=pad, templates=tk, method=name, all=_row(ev),
                           **{k: _row(evaluate([f for f in files if f['split'] == k], A[sp == k], VV, method))
                              for k in ('dev', 'test')})
                out['variants'].append(row)
                print(pad, tk, name, ' '.join(f'{k} {v:.3f}' for k, v in row['all'].items()), flush=True)
                if (pad, tk, name) == (vocab.PAD, a.templates, 'both'):
                    out['chosen'].update(row, detail=_clean(ev))
                    Zc = ev['_Zc']
        if pad != vocab.PAD:
            continue
        tp = TEMPLATES[a.templates]
        st = vocab.build_stats(R, refs, path=None, templates=tp)
        g = {s: grid(*class_pairs(files, Zc, cls, s)) for s in (False, True)}
        out['thresholds'] = {'rule': RULE, 'grid_lenient': g[False], 'grid_strict': g[True],
                             'picked': pick_thresholds(*class_pairs(files, Zc, cls, False))}
        out['thresholds']['shipped'] = {n: t for t, n in match.VERDICTS}
        out['thresholds']['bands_tuning'] = verdict_tables(files, Zc, cls, [t for t, _ in match.VERDICTS])
        w_uniform = dict(st, mean=list(R.mean(0)))
        musicish = np.array([p.endswith('.mp4') or '/out/audio' in p.replace(os.sep, '/') or '/music/' in
                             p.replace(os.sep, '/') or '_retired_synth' in p for p in refs])
        sens = {'folder-balanced (used)': st, 'uniform over files': w_uniform,
                'without music-like refs': vocab.build_stats(R[~musicish], [p for p, m in zip(refs, musicish) if not m],
                                                             path=None, templates=tp),
                'transductive (tuning set itself)': vocab.build_stats(A, [f['path'] for f in files], path=None,
                                                                     templates=tp),
                'no term bias': False}
        out['reference_sensitivity'] = {}
        for name, s_ in sens.items():
            VV = vocab.Vocab(tp, stats=s_)
            out['reference_sensitivity'][name] = {k: _row(evaluate([f for f in files if f['split'] == k], A[sp == k],
                                                                   VV, 'both')) for k in ('dev', 'test')}
        mu = V.mu_E
        hub = np.argsort(-mu)
        out['term_bias'] = {'highest': [(V.names[i], round(float(mu[i]), 3)) for i in hub[:10]],
                            'lowest': [(V.names[i], round(float(mu[i]), 3)) for i in hub[-5:]]}
    out['seconds'] = round(time.time() - t0, 1)
    _write('results.json', out)
    print('thresholds picked by the rule:', out['thresholds']['picked'], 'shipped:', out['thresholds']['shipped'])


def cmd_heldout(a):
    """held-out library with exactly the shipped configuration (vocabulary, templates, stats, thresholds), plus the
    pre-registered 217-term vocabulary for comparison (same templates and stats)"""
    files = heldout_set()
    A = embeddings('heldout', vocab.PAD)
    cls = list(Q()['classes'])
    cuts = [t for t, _ in match.VERDICTS]
    out = {'n_files': len(files), 'n_with_class': sum(bool(f['classes']) for f in files),
           'per_category': {c: sum(f['cat'] == c for f in files) for c in CATS}, 'thresholds': {n: t for t, n in match.VERDICTS}}
    V = vocab.get()
    pre = vocab.Vocab(stats=None, categories=_json('vocab_preregistered.json')['categories'])
    for name, VV in (('shipped', V), ('preregistered_217', pre)):
        ev = evaluate(files, A, VV, 'both', specific=False)
        out[name] = dict(n_vocab=len(VV.names), metrics=_row(ev), detail=_clean(ev),
                         verdicts=verdict_tables(files, ev['_Zc'], cls, cuts))
        print(name, len(VV.names), json.dumps({k: round(v, 3) for k, v in _row(ev).items()}), flush=True)
    raw = evaluate(files, A, V, 'raw', specific=False)
    out['raw_cosine_metrics'] = _row(raw)
    out['mood_check'] = {'heldout': mood_check(V, A, files),
                         'tuning': mood_check(V, embeddings('eval', vocab.PAD), eval_set())}
    print('mood', json.dumps(out['mood_check']))
    _write('heldout.json', out)


def cmd_stats(a):
    paths, wins = ship_refs()
    meta = {'max_win': 'tuning files 6, others 3',
            'reference_set': 'the 219 tuning recordings of the Clawd project + 7 professional launch films + Clawd '
                             'soundtrack/stems + retired code-synthesized stems (eval/semantics/run_eval.py ship_refs); '
                             'disjoint from the held-out library'}
    r = vocab.build_stats(embed_refs(paths, wins), paths, meta)
    print(f'wrote {vocab.STATS_PATH}: n_ref={r["n_ref"]} folders={r["n_folders"]}')


# ---------------------------------------------------------------------------------------------------- query kind
def cmd_kind(a):
    """find's query-kind reader on labelled query texts: 'tune' = the class phrasings of the tuning library (the
    thresholds were set on these), 'test' = generic queries written for this check. The costly error is an event
    read as 'texture' (windows only: no tight trims); 'either' scores both candidate kinds"""
    from earshot import search
    v = vocab.get()
    lab = _json('kind_queries.json')
    out = {'threshold': search.KIND_TEXTURE}
    for part in ('tune', 'test'):
        rows = []
        for kind in ('event', 'texture'):
            for q in lab[part][kind]:
                k, d = search.query_kind(v, q)
                rows.append({'query': q, 'truth': kind, 'read': k, 'score': d})
        cnt = {t: {k: sum(1 for r in rows if r['truth'] == t and r['read'] == k) for k in ('event', 'either', 'texture')}
               for t in ('event', 'texture')}
        out[part] = {'events': cnt['event'], 'textures': cnt['texture'],
                     'events_read_as_texture': [(r['query'], r['score']) for r in rows
                                                if r['truth'] == 'event' and r['read'] == 'texture'],
                     'textures_read_as_event': [(r['query'], r['score']) for r in rows
                                                if r['truth'] == 'texture' and r['read'] == 'event'], 'rows': rows}
        print(part, {k: v_ for k, v_ in out[part].items() if k != 'rows'})
    _write('kind.json', out)


# ---------------------------------------------------------------------------------------------------- find
KEYQ = 'a single mechanical keyboard key press'
SCENE = [('fs-170237', 'magic', 'a glass ting'), ('fs-719637', 'motion', 'a fast whoosh'),
         ('fs-16064', 'foley', 'a cork pop'), ('fs-670000', 'toy', 'a lego brick click'),
         ('fs-272392', 'motion', 'an object dropping into water'), ('mk-2585', 'toy', 'a light switch click'),
         ('fs-789630', 'keyboard', 'a spacebar press'), ('fs-452379', 'magic', 'a small bell')]
LONG_EVENTS = [(20.0, 'fs-170237', 'magic'), (150.0, 'fs-170237', 'magic'), (275.0, 'fs-170237', 'magic'),
               (205.0, 'fs-16064', 'foley'), (240.0, 'fs-719637', 'motion')]
LONG_QUERIES = [('a glass ting', [20.0, 150.0, 275.0]), ('a cork pop', [205.0]), ('a fast whoosh', [240.0])]


def _manifest(fid):
    for m in glob.glob(f'{ASSETS}/manifests/*.json'):
        j = json.load(open(m, encoding='utf-8'))
        for e in j if isinstance(j, list) else []:
            if e.get('id') == fid and e.get('category'):
                return e


def _attacks(x, slices):
    """attack inside every labelled slice, measured independently of find's trim rule: the first SAMPLE whose
    magnitude reaches 10% (-20 dB) of the slice's peak sample"""
    out = []
    for s, e in slices:
        seg = np.abs(x[int(s * audio.SR):int(e * audio.SR)])
        out.append(s + int(np.argmax(seg >= 0.1 * seg.max())) / audio.SR)
    return out


def _score_keys(segs, x, slices, limit=None, skip=None):
    """a hit = the segment starts on a labelled key press as a usable trim (lead = attack - start between 0 and 60 ms:
    the attack is not clipped and there is little dead air) and ends before the next press"""
    att = _attacks(x, slices)
    rows = []
    for g in segs:
        k = int(np.argmin([abs(g['start'] - t) for t in att]))
        lead = att[k] - g['start']
        nxt = att[k + 1] if k + 1 < len(att) else 1e9
        labelled = limit is None or g['start'] < limit
        rows.append({'start': g['start'], 'end': g['end'], 'z': g['z'], 'kind': g['kind'], 'tail_db': g['tail_db'],
                     'press': k if labelled else None, 'lead_ms': round(1000 * lead, 1) if labelled else None,
                     'in_skip_region': bool(skip and skip[0] <= g['start'] < skip[1]),
                     'hit': bool(labelled and 0.0 <= lead <= 0.06 and g['end'] <= nxt + 0.005)})
    hits = [r for r in rows if r['hit']]
    return {'hits': len(hits), 'n': len(rows), 'distinct_presses': len({r['press'] for r in hits}),
            'in_unlabelled_part': sum(r['press'] is None for r in rows),
            'false_positives_in_skip_region': sum(r['in_skip_region'] for r in rows),
            'median_lead_ms': float(np.median([r['lead_ms'] for r in hits])) if hits else None,
            'lead_ms_range': [min(r['lead_ms'] for r in hits), max(r['lead_ms'] for r in hits)] if hits else None,
            'rows': rows}


def _oneshot(fid, cat, peak_db=-6.0):
    p = next(iter(glob.glob(f'{ASSETS}/{cat}/{fid}_*')))
    x = audio.load(p, mono=True)
    env = np.abs(x)
    i0 = max(0, int(np.argmax(env >= env.max() * 0.05)) - int(0.01 * audio.SR))
    x = x[i0: i0 + int(2.5 * audio.SR)].copy()
    x[-int(0.05 * audio.SR):] *= np.linspace(1, 0, int(0.05 * audio.SR))
    return x * 10 ** (peak_db / 20) / (np.abs(x).max() + 1e-12)


def _bed(seconds):
    bed = audio.load(f'{ASSETS}/ambience/fs-664152_roomtone-basement-hifi.ogg', mono=True)
    bed = np.tile(bed, int(np.ceil(seconds * audio.SR / len(bed))))[: int(seconds * audio.SR)].copy()
    return bed * 10 ** (-55 / 20) / (np.sqrt(np.mean(bed ** 2)) + 1e-12)


def build_scene(path):
    """real one-shots (first <= 2.5 s, peak -6 dBFS) at known times over a real basement room tone at -55 dBFS RMS"""
    import soundfile as sf
    bed = _bed(44)
    truth = []
    for k, (fid, cat, q) in enumerate(SCENE):
        x = _oneshot(fid, cat)
        t = 2.0 + 5.0 * k
        bed[int(t * audio.SR):int(t * audio.SR) + len(x)] += x
        truth.append({'id': fid, 'query': q, 'start': t, 'end': round(t + len(x) / audio.SR, 3)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, bed, audio.SR, subtype='FLOAT')
    return truth


def build_long(path):
    """300 s: room tone + a real key press every ~1.3 s all the way through (Keychron slices, -12 dBFS) + glass tings
    at 20 / 150 / 275 s, a cork pop at 205 s and a whoosh at 240 s (-6 dBFS): events must be found anywhere"""
    import soundfile as sf
    bed = _bed(300)
    kp = f'{ASSETS}/keyboard/fs-789628_keychron-linear-individual-keys.ogg'
    kx = audio.load(kp, mono=True)
    keys = [kx[int(s * audio.SR):int(e * audio.SR)] for s, e in _manifest('fs-789628')['slices']]
    busy = [(t, t + 2.6) for t, _, _ in LONG_EVENTS]
    t, k = 1.0, 0
    while t < 298:
        if not any(a - 0.7 < t < b for a, b in busy):
            x = keys[k % len(keys)] * 10 ** (-12 / 20) / (np.abs(keys[k % len(keys)]).max() + 1e-12)
            bed[int(t * audio.SR):int(t * audio.SR) + len(x)] += x
        t += 1.3 + 0.1 * (k % 3)
        k += 1
    for t, fid, cat in LONG_EVENTS:
        x = _oneshot(fid, cat)
        bed[int(t * audio.SR):int(t * audio.SR) + len(x)] += x
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, bed, audio.SR, subtype='FLOAT')


def cmd_find(a):
    from earshot import search
    vocab.PAD = a.pad
    out = {'pad': a.pad}
    k1 = f'{ASSETS}/keyboard/fs-789628_keychron-linear-individual-keys.ogg'
    k2 = f'{ASSETS}/keyboard/fs-709460_lubed-wsbrown-creamy-typing.ogg'
    k3 = f'{ASSETS}/keyboard/fs-609534_single-key-presses-multi.ogg'
    for name, p, top, limit, skip in (('keychron_17_keys', k1, 17, None, (9.3, 10.45)),
                                      ('creamy_typing_slow_then_fast', k2, 5, 17.3, None),
                                      ('mars_32_keys', k3, 10, None, None)):
        x = audio.load(p, mono=True)
        sl = _manifest(os.path.basename(p).split('_')[0])['slices']
        t = time.time()
        r = search.find(p, KEYQ, top=top)
        dt = time.time() - t
        t = time.time()
        search.find(p, KEYQ, top=top)
        res = dict(_score_keys(r['segments'], x, sl, limit, skip), seconds=round(dt, 1),
                   seconds_cached=round(time.time() - t, 2), duration=r['duration'], n_candidates=r['n_candidates'],
                   query=KEYQ)
        if limit:
            r2 = search.find(p, 'fast continuous typing on a keyboard', top=5)
            res['continuous_typing_query'] = {'in_fast_part': sum(g['start'] >= limit for g in r2['segments']), 'n': 5,
                                              'segments': [(g['start'], g['end'], g['z'], g['kind']) for g in r2['segments']]}
        out[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
    for name, build, queries in (('scene_44s', build_scene, None), ('long_300s', build_long, LONG_QUERIES)):
        path = os.path.join(OUT, f'find_{name}.wav')
        truth = build(path)
        if queries is None:
            queries = [(tr['query'], [tr['start']]) for tr in truth]
        rows = []
        t = time.time()
        for q, starts in queries:
            r = search.find(path, q, top=max(3, len(starts)))
            segs = r['segments']
            got = [any(g['start'] - 0.05 <= s0 < g['end'] for g in segs) for s0 in starts]
            first = [g['start'] - 0.05 <= starts[0] < g['end'] for g in segs]
            rows.append({'query': q, 'truth_starts': starts, 'found_in_top_n': sum(got), 'n_truth': len(starts),
                         'top1_hit': bool(first and any(g['start'] - 0.05 <= s0 < g['end'] for s0 in starts
                                                        for g in segs[:1])),
                         'query_kind': r['query_kind'], 'n_candidates': r['n_candidates'],
                         'segments': [[g['start'], g['end'], g['z'], g['kind']] for g in segs],
                         'lead_ms': [round(1000 * (s0 + 0.01 - g['start']), 1) for s0 in starts for g in segs
                                     if g['start'] - 0.05 <= s0 < g['end'] and g['kind'] != 'window']})
            print(name, rows[-1]['query'], rows[-1]['found_in_top_n'], '/', len(starts), rows[-1]['segments'][:3],
                  flush=True)
        out[name] = {'found': sum(x['found_in_top_n'] for x in rows), 'n': sum(x['n_truth'] for x in rows),
                     'top1': sum(x['top1_hit'] for x in rows), 'n_queries': len(rows),
                     'tight_trims': sum(len(x['lead_ms']) for x in rows), 'seconds_all_queries': round(time.time() - t, 1),
                     'rows': rows}
    _write(f'find_{a.pad}.json', out)


def cmd_timing(a):
    """wall-clock per call with warm models on fresh copies (nothing cached) in out/semantics/timing_*/"""
    import shutil
    from earshot import config, describe, search
    d = os.path.join(OUT, f'timing_{int(time.time())}')
    os.makedirs(d)
    cp = lambda p: shutil.copy(p, d) and os.path.join(d, os.path.basename(p))  # noqa: E731
    t = time.time()
    vocab.get()
    describe._ast()
    vocab._clap()
    out = {'threads': config.threads(), 'precision': config.PRECISION, 'model_load_s': round(time.time() - t, 1)}

    def tm(name, f, *args, **kw):
        t0 = time.time()
        f(*args, **kw)
        out[name] = round(time.time() - t0, 2)
        print(name, out[name], flush=True)
    key = cp(f'{ASSETS}/keyboard/fs-789629_keychron-linear-enter.ogg')
    tm('describe_short_cold', describe.describe, key)
    tm('match_cached_new_query', match.match, key, 'a deep thock of a heavy key')
    rain = cp(f'{ASSETS}/ambience/fs-346642_rain-on-windows-interior.ogg')
    tm('describe_31s_cold', describe.describe, rain)
    k17 = cp(f'{ASSETS}/keyboard/fs-789628_keychron-linear-individual-keys.ogg')
    tm('find_18s_cold', search.find, k17, KEYQ)
    tm('find_18s_cached_new_query', search.find, k17, 'a soft click')
    lng = cp(os.path.join(OUT, 'find_long_300s.wav'))
    tm('find_300s_cold', search.find, lng, 'a glass ting')
    rk = os.path.join(d, 'rank')
    os.makedirs(rk)
    src = [p for p in audio.list_media(f'{ASSETS}/foley', recursive=False)][:20]
    for p in src:
        shutil.copy(p, rk)
    tm('rank_20_foley_files_cold', search.rank, rk, 'a cork pop')
    out['rank_s_per_file'] = round(out['rank_20_foley_files_cold'] / len(src), 2)
    out['rank_files_total_s'] = round(sum(audio.duration(p) for p in src), 1)
    tm('rank_20_cached_new_query', search.rank, rk, 'a suction cup pop')
    _write('timing.json', out)
    print(json.dumps(out))


def main():
    ap_ = argparse.ArgumentParser()
    sub = ap_.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('embed')
    p.add_argument('--set', choices=['eval', 'tuneref', 'heldout'], required=True)
    p.add_argument('--pad', default='repeatpad', choices=vocab.PADS)
    p.add_argument('--part', default='1/1')
    p.set_defaults(func=cmd_embed)
    p = sub.add_parser('analyze')
    p.add_argument('--templates', default='t5')
    p.set_defaults(func=cmd_analyze)
    for name, fn in (('stats', cmd_stats), ('heldout', cmd_heldout), ('kind', cmd_kind), ('timing', cmd_timing)):
        sub.add_parser(name).set_defaults(func=fn)
    p = sub.add_parser('find')
    p.add_argument('--pad', default=vocab.PAD, choices=vocab.PADS)
    p.set_defaults(func=cmd_find)
    a = ap_.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
