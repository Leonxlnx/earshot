"""Choose when timeline prefers the faster of two octave-related tempos (a two-beat pattern makes the autocorrelation
peak at the half tempo). Reference: librosa's beat tracker (not ground truth, an independent second opinion) on the
same 20 s excerpts of the unused / rejected library music as pulse_tune.py; never the Clawd soundtrack.

    .venv/Scripts/python.exe eval/timeline/tempo_tune.py

For each excerpt: timeline's tempo, the autocorrelation at its beat period and at half of it (the faster octave), and
librosa's tempo. Prints how often timeline agrees with librosa (within 4%, or at 2x / 0.5x) for each rule
"faster octave when ac(P/2) >= r * ac(P) and it is <= 180 BPM".
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
    import librosa
    rows = []
    for f in mixes.unused('music'):
        x = audio.load(f, mono=True, start=10.0, end=30.0)
        if len(x) < 8 * audio.SR:
            x = audio.load(f, mono=True, end=20.0)
        cands = [p for env in tl._pulse_envs(x) for p in [tl._period(env, 60, 200)] if p]
        if not cands:
            continue
        P, clar, _, ac = max(cands, key=lambda c: c[2])
        y = audio.resample(x, audio.SR, 22050)
        lt = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=22050)[0])[0])
        rows.append({'file': os.path.basename(f), 'tempo': 6000 / P, 'clarity': clar,
                     'ac_half': float(ac[int(round(P / 2))]), 'librosa': lt})
    print(len(rows), 'music excerpts')

    def agree(a, b, oct_ok=False):
        r = a / b
        return abs(r - 1) <= 0.04 or (oct_ok and (abs(r - 2) <= 0.08 or abs(r - 0.5) <= 0.02))
    for ratio in (None, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        same = octv = faster = 0
        for q in rows:
            t = q['tempo']
            if ratio is not None and 2 * t <= 180 and q['ac_half'] >= ratio * q['clarity']:
                t, faster = 2 * t, faster + 1
            same += int(agree(t, q["librosa"]))
            octv += int(agree(t, q["librosa"], True))
        print(json.dumps({'rule_ratio': ratio, 'switched_to_faster': faster, 'agree_with_librosa': same,
                          'agree_incl_octave': octv, 'n': len(rows)}))
    print('timeline slower than librosa (before):', [(q['file'][:30], round(q['tempo']), round(q['librosa']),
                                                      round(q['ac_half'] / max(q['clarity'], 1e-6), 2))
                                                     for q in rows if agree(2 * q['tempo'], q['librosa'])])


if __name__ == '__main__':
    main()
