"""Check the pure-tone rule in earshot.realness: what it does to a test tone, and that it never fires on a real recording.

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/tone_guard.py      # -> eval/realness/tone_guard.json

1. a 30 s 440 Hz sine at -10.5 dBFS (the eval/launch negative): mean p_real over its 4 s windows from the model alone
   (CLAP head + fusion, no rule) and with the rule.
2. every real recording of the Clawd project (the 219 used + 126 rejected Freesound / Mixkit files under
   EP.ASSETS, see eval/README.md, read-only), clipped exactly as realness() clips it (whole sound up to 6 s, else
   4 s windows): on how many clips does pure_tone() fire, and which audible clips come closest (fewest-failed tests).
No model is needed for 2 (signal measurements only).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from earshot import audio  # noqa: E402
from earshot import realness as R  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

ASSETS = EP.ASSETS


def _closeness(f):
    """how many of the rule's four tests a clip passes (4 = the rule fires)"""
    w = f.get('pitch_wobble', float('nan'))
    return (int(f.get('tonal_frames', 0) >= 0.95) + int(f.get('line_share', 0) >= 0.9) + int(f.get('crest_db', 99) <= 4.5)
            + int(not np.isfinite(w) or w <= -1.0))


def real_files():
    out = []
    for root, dirs, files in os.walk(ASSETS):
        dirs.sort()
        out += [os.path.join(root, f) for f in sorted(files) if os.path.splitext(f)[1].lower() in audio.AUDIO_EXTS]
    return out


def main():
    res = {}
    t = np.arange(30 * audio.SR) / audio.SR
    sine = (10 ** (-10.5 / 20) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    m = R.load_model()
    rows = R.score_clips([c for _, _, c in R._clips(R.lift_quiet(sine)[0])], m)
    model_only = [R.fuse(m['fusion'], r['logits'])[0] for r in rows]
    res['sine_440hz_30s'] = {'p_real_model_only': round(float(np.mean(model_only)), 3),
                             'p_real_with_rule': round(float(np.mean([r['p_real'] for r in rows])), 3),
                             'windows': len(rows), 'rule_fired': sum(bool(r.get('tone')) for r in rows)}
    print('sine', res['sine_440hz_30s'], flush=True)
    files = real_files()
    fired, n_clips, close = [], 0, []
    for i, p in enumerate(files):
        x = R.lift_quiet(audio.load(p, mono=True))[0]
        for a, b, c in R._clips(x):
            if not R.audible(c):
                continue
            f = R.signal_features(c)
            if f is None:
                continue
            n_clips += 1
            k = _closeness(f)
            if R.pure_tone(f):
                fired.append({'file': os.path.relpath(p, ASSETS), 'start': a})
            elif k >= 2:
                close.append({'file': os.path.relpath(p, ASSETS).replace('\\', '/'), 'start': round(a, 1),
                              'tests_passed': k, 'crest_db': round(float(f.get('crest_db', 0)), 1),
                              'line_share': round(float(f.get('line_share', 0)), 2),
                              'tonal_frames': round(float(f.get('tonal_frames', 0)), 2)})
        if i % 50 == 0:
            print(f'{i}/{len(files)} files, {n_clips} clips, fired {len(fired)}', flush=True)
    close.sort(key=lambda r: (-r['tests_passed'], r['crest_db']))
    res['real_recordings'] = {'files': len(files), 'audible_clips': n_clips, 'rule_fired': len(fired),
                              'fired_on': fired[:20], 'closest': close[:5]}
    json.dump(res, open(os.path.join(HERE, 'tone_guard.json'), 'w'), indent=1)
    print(json.dumps(res['real_recordings'], indent=1)[:1500])


if __name__ == '__main__':
    main()
