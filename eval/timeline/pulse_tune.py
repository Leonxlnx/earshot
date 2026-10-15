"""Calibrate timeline's pulse-clarity thresholds: music beds should show a pulse, ambience beds should not.

    .venv/Scripts/python.exe eval/timeline/pulse_tune.py

Uses 20 s excerpts (from 10 s in) of the unused / rejected music and ambience recordings of the Clawd library (the
same pools as the synthetic mixes, never the Clawd soundtrack).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from earshot import audio, timeline as tl  # noqa: E402
import mixes  # noqa: E402


def main():
    res = {}
    for cat in ('music', 'ambience'):
        vals = []
        for f in mixes.unused(cat):
            x = audio.load(f, mono=True, start=10.0, end=30.0)
            if len(x) < 8 * audio.SR:
                x = audio.load(f, mono=True, end=20.0)
            g = tl.music_grid(x)
            vals.append((g.get('pulse_clarity') or 0.0, g.get('tempo'), os.path.basename(f)))
        res[cat] = vals
        c = np.array([v[0] for v in vals])
        print(cat, len(c), 'clarity pct10/50/90:', np.round(np.percentile(c, [10, 50, 90]), 3).tolist())
    mu = np.array([v[0] for v in res['music']])
    am = np.array([v[0] for v in res['ambience']])
    for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        print(json.dumps({'thr': thr, 'music_with_pulse': round(float(np.mean(mu >= thr)), 3),
                          'ambience_with_pulse': round(float(np.mean(am >= thr)), 3)}))
    print('lowest-clarity music:', sorted(res['music'])[:5])
    print('highest-clarity ambience:', sorted(res['ambience'])[-5:])


if __name__ == '__main__':
    main()
