"""Tune timeline's section merging on the synthetic mixes (bed switches = true boundaries). Never uses Clawd.

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/section_tune.py [--shipped]

Boundary recall / precision within +-1.5 s for several CLAP merge thresholds and Foote-novelty thresholds (99 = no
novelty split); --shipped runs only the shipped setting (3 s windows, 2 s hop, merge 0.84, novelty z 3.0). CLAP
window embeddings are cached, so only the first run pays for the model. Labels come from CLAP only (no AST), as in
timeline's merge of neighbouring sections that read alike.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from earshot import audio, timeline as tl  # noqa: E402
from tune import match  # noqa: E402


def main(d='out/timeline/mixes', tol=1.5):
    mixes = [(audio.load(f[:-5] + '.wav'), tl.span_id(f[:-5] + '.wav'), json.load(open(f)))
             for f in sorted(glob.glob(os.path.join(d, 'mix_*.json')))]
    grid = [(3.0, 2.0, mc, nz) for mc in (0.80, 0.84, 0.88) for nz in (2.0, 3.0, 4.0, 99.0)]
    if '--shipped' in sys.argv:
        grid = [(3.0, 2.0, 0.84, tl.NOVELTY_Z)]
    else:
        grid += [(4.0, 4.0, 0.84, 3.0), (3.0, 1.5, 0.84, 3.0)]
    for win, hop, mc, nz in grid:
        tp = n_gt = n_det = 0
        err = []
        for x, fid, meta in mixes:
            secs, _ = tl.sections(x, fid, 0.0, win, hop, ast_budget=0, merge_cos=mc, nov_z=nz)
            det = [s['start'] for s in secs[1:]]
            m = match(meta['boundaries'], det, tol)
            tp += len(m)
            n_gt += len(meta['boundaries'])
            n_det += len(det)
            err += [abs(dt) for _, _, dt in m]
        print(json.dumps({'win': win, 'hop': hop, 'merge_cos': mc, 'novelty_z': nz, 'recall': round(tp / n_gt, 3),
                          'precision': round(tp / max(1, n_det), 3), 'n_gt': n_gt, 'n_det': n_det,
                          'median_abs_err_s': round(float(np.median(err)), 2) if err else None}), flush=True)


if __name__ == '__main__':
    main()
