"""Remove hub phrases from timeline's event tags: per-phrase background statistics over real events, and the choice
of scoring rule and confidence threshold. Never uses the Clawd film or the three reference films used for evaluation.

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/tag_bg.py [--per 40]

Background events: the events timeline would tag (its own picking, up to --per per source) in the 8 synthetic tuning
mixes (mixes.py) and in four launch films that the evaluation does not use (ref_3, ref_5,
ref_6, ref_7). Fit half = mixes 0-3 + ref_3 + ref_5; test half = mixes 4-7 + ref_6 + ref_7. Scoring rules over the
CLAP cosines C (events x phrases), with the fit half's per-phrase mean mu and std sd:
  row     z of C within each event (the first build: phrases that score high on everything win: hubs)
  col     (C - mu) / sd: how unusual the phrase's score is for this event compared with typical events
  cent    z within each event of C - mu
Metrics (test half): one-shot top-1 / top-3 kind (FIT, as tag_tune.py), the share of bed events (music / ambience
onsets at least 0.15 s from any one-shot) called KEY or UI, and the share of the single most common phrase over all
test events (hubness). Then confidence thresholds on the top background z. Prints the TAG_BG table (all sources) for
earshot/timeline.py.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from earshot import audio, timeline as tl  # noqa: E402
from earshot.models import clap  # noqa: E402
from tag_tune import FIT  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

FILMS = [EP.ref(rid) for rid in ('ref_3', 'ref_5', 'ref_6', 'ref_7')]   # 4 reference films the evaluation does not use
UIK = {'KEY', 'UI'}


def sources():
    out = [(f'out/timeline/mixes/mix_{k}.wav', 'fit' if k < 4 else 'test') for k in range(8)]
    return out + [(f, 'fit' if k < 2 else 'test') for k, f in enumerate(FILMS)]


def events_of(path, per, c, E):
    """the events timeline would tag (+ every placed one-shot in a mix) with their cosines and ground truth"""
    x = audio.load(path)
    ev = [e for e in tl.detect_events(x) if e['shape'] == 'hit']
    dur = len(x) / tl.SR
    order = sorted(range(len(ev)), key=lambda i: (-ev[i]['prominence_db'], -ev[i]['contrast_db']))
    pick = set(tl._spread(order, [e['t'] for e in ev], dur, per))
    meta = json.load(open(path[:-4] + '.json')) if path.endswith('.wav') else None
    truth = {}
    if meta:
        det = np.array([e['t'] for e in ev])
        for g in meta['events']:
            k = int(np.argmin(np.abs(det - g['t'])))
            if g['align'] == 'attack' and abs(det[k] - g['t']) <= 0.03:
                truth[k] = (g['cat'], g['snr_db'])
                pick.add(k)
    pick = sorted(pick)
    spans = [tl._event_span(ev[i], dur) for i in pick]
    A = tl._cached(c, tl.span_id(path), spans, 'clap-repeat',
                   lambda q: c.embed_audio([tl._seg(x, a, b) for a, b in q], pad='repeatpad'))
    rows = []
    for j, i in enumerate(pick):
        snr = np.nan
        if i in truth:
            role, snr = truth[i]
        elif meta and min(abs(ev[i]['t'] - g['t']) for g in meta['events']) >= 0.15:
            role = 'bed'
        elif meta:
            role = 'other'
        else:
            role = 'film'
        rows.append({'role': role, 'cos': A[j] @ E.T, 'prom': ev[i]['prominence_db'], 'snr': snr})
    return rows


def score(rule, C, mu, sd):
    if rule == 'row':
        return (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-9)
    if rule == 'col':
        return (C - mu) / sd
    D = C - mu
    return (D - D.mean(1, keepdims=True)) / (D.std(1, keepdims=True) + 1e-9)


def report(rule, rows, terms, kinds, mu, sd):
    C = np.stack([r['cos'] for r in rows])
    Z = score(rule, C, mu, sd)
    top = np.argsort(-Z, axis=1)[:, :3]
    one = [j for j, r in enumerate(rows) if r['role'] in FIT]
    bed = [j for j, r in enumerate(rows) if r['role'] == 'bed']
    t1 = [kinds[top[j, 0]] in FIT[rows[j]['role']] for j in one]
    t3 = [any(kinds[k] in FIT[rows[j]['role']] for k in top[j]) for j in one]
    cnt = Counter(terms[top[j, 0]] for j in range(len(rows)))
    res = {'rule': rule, 'n_oneshots': len(one), 'top1': round(float(np.mean(t1)), 3),
           'top3': round(float(np.mean(t3)), 3), 'n_bed': len(bed),
           'bed_called_key_or_ui': round(float(np.mean([kinds[top[j, 0]] in UIK for j in bed])), 3),
           'film_called_key_or_ui': round(float(np.mean([kinds[top[j, 0]] in UIK for j, r in enumerate(rows)
                                                         if r['role'] == 'film'])), 3),
           'most_common_share': round(cnt.most_common(1)[0][1] / len(rows), 3), 'most_common': cnt.most_common(4),
           'per_category': {k: round(float(np.mean([t for t, j in zip(t1, one) if rows[j]['role'] == k])), 2)
                            for k in sorted({rows[j]['role'] for j in one})}}
    print(json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per', type=int, default=40)
    a = ap.parse_args()
    c = clap.get()
    terms, kinds, E = tl._event_matrix(c)
    kinds = [str(k) for k in kinds]
    rows = {'fit': [], 'test': []}
    for path, half in sources():
        rows[half] += events_of(path, a.per, c, E)
    np.savez('out/timeline/tag_bg_rows.npz', **{f'{h}_{k}': np.array([r[k] for r in rows[h]])
                                                for h in rows for k in ('role', 'cos', 'prom', 'snr')})
    print('events: fit', len(rows['fit']), 'test', len(rows['test']), flush=True)
    F = np.stack([r['cos'] for r in rows['fit']])
    mu, sd = F.mean(0), F.std(0) + 1e-6
    for rule in ('row', 'col', 'cent'):
        report(rule, rows['test'], terms, kinds, mu, sd)
    C = np.stack([r['cos'] for r in rows['test']])
    Z = score('col', C, mu, sd)
    top = np.argmax(Z, axis=1)
    for thr in (0.5, 1.0, 1.5, 2.0, 2.5):
        conf = Z[np.arange(len(Z)), top] >= thr
        one = [j for j, r in enumerate(rows['test']) if r['role'] in FIT]
        bed = [j for j, r in enumerate(rows['test']) if r['role'] == 'bed']
        film = [j for j, r in enumerate(rows['test']) if r['role'] == 'film']
        ok = [kinds[top[j]] in FIT[rows['test'][j]['role']] for j in one if conf[j]]
        print(json.dumps({'col_threshold': thr, 'oneshot_coverage': round(float(np.mean(conf[one])), 3),
                          'oneshot_precision': round(float(np.mean(ok)), 3) if ok else None,
                          'bed_confident': round(float(np.mean(conf[bed])), 3),
                          'bed_confident_key_or_ui': round(float(np.mean([conf[j] and kinds[top[j]] in UIK
                                                                          for j in bed])), 3),
                          'film_confident': round(float(np.mean(conf[film])), 3)}), flush=True)
    A = np.stack([r['cos'] for h in rows.values() for r in h])
    mu, sd = A.mean(0), A.std(0)
    print(f'TAG_BG from {len(A)} events:')
    print(json.dumps({t: [round(float(m), 4), round(float(s), 4)] for t, m, s in zip(terms, mu, sd)}))


if __name__ == '__main__':
    main()
