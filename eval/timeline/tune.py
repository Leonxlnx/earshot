"""Tune the timeline event detector on the synthetic mixes (eval/timeline/mixes.py), never on the Clawd soundtrack.

    .venv/Scripts/python.exe eval/timeline/tune.py [--dir out/timeline/mixes] [--grid]

Reports recall (hits found within 30 ms of their attack; whooshes within 120 ms of their peak), precision against
the placed hits, and the share of extra detections explained by onsets in the bed alone (music / ambience notes).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from earshot import audio, timeline  # noqa: E402


def match(gt, det, tol):
    """greedy one-to-one matching by |dt|: list of (gt_index, det_index, dt)"""
    pairs = sorted((abs(d - g), i, j, d - g) for i, g in enumerate(gt) for j, d in enumerate(det) if abs(d - g) <= tol)
    gi, dj, out = set(), set(), []
    for _, i, j, dt in pairs:
        if i not in gi and j not in dj:
            gi.add(i); dj.add(j); out.append((i, j, dt))
    return out


def score(mixes, zthr=None, prom=None):
    """mixes: [(events detected at the loosest thresholds, bed events, meta)]; filter, then score"""
    rows = {'hit': [], 'dt': [], 'snr': [], 'whoosh': [], 'n_det': 0, 'tp': 0, 'explained': 0}
    for ev, bed, meta in mixes:
        if zthr is not None:
            ev = [e for e in ev if e['shape'] == 'swell' or (e['z'] >= zthr and e['contrast_db'] >= prom)]
        det = [e['t'] for e in ev]
        hits = [e for e in meta['events'] if e['align'] == 'attack']
        wh = [e for e in meta['events'] if e['align'] == 'peak']
        m1 = match([e['t'] for e in hits], det, 0.03)
        found = {i for i, _, _ in m1}
        rows['hit'] += [i in found for i in range(len(hits))]
        rows['snr'] += [e['snr_db'] for e in hits]
        rows['dt'] += [dt for _, _, dt in m1]
        rest = [d for j, d in enumerate(det) if j not in {j for _, j, _ in m1}]
        m2 = match([e['t'] for e in wh], rest, 0.12)
        rows['whoosh'] += [i in {i for i, _, _ in m2} for i in range(len(wh))]
        rest_j = [j for j in range(len(det)) if j not in {j for _, j, _ in m1}]
        used = {j for _, j, _ in m1} | {rest_j[j] for _, j, _ in m2}
        rest_idx = [j for j in range(len(det)) if j not in used]
        bed_det = [e['t'] for e in bed]
        m3 = match(bed_det, [det[j] for j in rest_idx], 0.03)
        rows['n_det'] += len(det)
        rows['tp'] += len(m1) + len(m2)
        rows['explained'] += len(m3)
    h, snr = np.array(rows['hit']), np.array(rows['snr'])
    res = {'recall_hits': round(float(h.mean()), 3), 'recall_snr_ge0': round(float(h[snr >= 0].mean()), 3),
           'recall_snr_-6_0': round(float(h[(snr < 0) & (snr >= -6)].mean()), 3),
           'recall_snr_lt-6': round(float(h[snr < -6].mean()), 3),
           'recall_whoosh': round(float(np.mean(rows['whoosh'])), 3) if rows['whoosh'] else None,
           'precision_vs_placed': round(rows['tp'] / max(1, rows['n_det']), 3),
           'precision_incl_bed_onsets': round((rows['tp'] + rows['explained']) / max(1, rows['n_det']), 3),
           'n_det': rows['n_det'], 'n_gt': int(len(h) + len(rows['whoosh'])),
           'median_dt_ms': round(float(np.median(rows['dt'])) * 1000, 1) if rows['dt'] else None,
           'p90_abs_dt_ms': round(float(np.percentile(np.abs(rows['dt']), 90)) * 1000, 1) if rows['dt'] else None}
    return res


def load(d, **kw):
    """detections per mix + every real onset in its bed alone (loosest settings, so extra detections that are
    genuine bed onsets count as explained)"""
    out = []
    for f in sorted(glob.glob(os.path.join(d, 'mix_*.json'))):
        x, b = audio.load(f[:-5] + '.wav'), audio.load(f[:-5] + '_bed.wav')
        bed = timeline.detect_events(b, zthr=2.0, min_contrast_db=1.0)
        out.append((timeline.detect_events(x, **kw), bed, json.load(open(f))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out/timeline/mixes')
    ap.add_argument('--grid', action='store_true')
    a = ap.parse_args()
    print('default', json.dumps(score(load(a.dir))))
    if a.grid:
        loose = load(a.dir, zthr=2.0, min_contrast_db=1.0)
        for z in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
            for p in (1.0, 3.0, 6.0):
                print(f'zthr {z} prom {p}', json.dumps(score(loose, z, p)))


if __name__ == '__main__':
    main()
