"""Tune the build-up (whoosh / riser) detector on the synthetic mixes: whooshes are placed with their peak on the
ground-truth time. Never uses the Clawd mix.

    .venv/Scripts/python.exe eval/timeline/swell_tune.py [--peaks]

--peaks tunes the level-peak check sync uses where no onset answers a cut / cue: at every whoosh peak (should be
answered) and at times on a 0.25 s grid at least 0.6 s from any placed sound (nothing placed there), how often each
rise threshold finds a level peak among the times no onset / swell answers.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from earshot import audio, timeline as tl  # noqa: E402
from tune import match  # noqa: E402


def peaks(d='out/timeline/mixes'):
    rows = {'whoosh': [], 'empty': []}
    for f in sorted(glob.glob(os.path.join(d, 'mix_*.json'))):
        x = audio.load(f[:-5] + '.wav')
        meta, ev, lv = json.load(open(f)), tl.detect_events(x), tl.level_envs(x)
        pt = np.array([g['t'] for g in meta['events']])
        grid = [t for t in np.arange(1.0, meta['duration'] - 1.0, 0.25) if np.min(np.abs(pt - t)) >= 0.6]
        for kind, times in (('whoosh', [g['t'] for g in meta['events'] if g['align'] == 'peak']), ('empty', grid)):
            for it in tl.sync(times, ev)['items']:
                rows[kind].append((it['t'], it['status'], lv))
    for kind, r in rows.items():
        un = [(t, lv) for t, st, lv in r if st in ('no sound', 'off')]
        print(kind, json.dumps({'n': len(r), 'answered_by_onset_or_swell': round(1 - len(un) / len(r), 3),
                                'unanswered': len(un)}))
        for rise, local in ((3.0, 0.3), (4.0, 0.3), (5.0, 0.3), (6.0, 0.3), (8.0, 0.3), (3.0, 0.15), (4.0, 0.15)):
            got = [tl.level_peak(lv, t, min_rise_db=rise, local=local) is not None for t, lv in un]
            print(f'  rise >= {rise} dB, local max over +-{local} s: level peak found for {sum(got)}/{len(un)} '
                  f'unanswered ({100 * np.mean(got) if got else 0:.0f}%)')


def main(d='out/timeline/mixes'):
    if '--peaks' in sys.argv:
        return peaks(d)
    mixes = []
    for f in sorted(glob.glob(os.path.join(d, 'mix_*.json'))):
        x = audio.load(f[:-5] + '.wav')
        mixes.append((x, [e for e in tl.detect_events(x) if e['shape'] == 'hit'], json.load(open(f))))
    for rise, mono, dur in [(r, m, d) for r in (5.0, 6.0, 8.0) for m in (0.5, 0.6) for d in (0.12, 0.25)]:
        found = n_gt = n_sw = near_any = 0
        for x, hits, meta in mixes:
            sw = tl._swells(x, hits, min_rise_db=rise, mono_frac=mono, min_dur=dur)
            gt = [e['t'] for e in meta['events'] if e['align'] == 'peak']
            found += len(match(gt, [s['t'] for s in sw], 0.12))
            n_gt += len(gt)
            n_sw += len(sw)
            near_any += sum(any(abs(s['t'] - g['t']) < 0.3 for g in meta['events']) for s in sw)
        print(json.dumps({'min_rise_db': rise, 'mono_frac': mono, 'min_dur_s': dur,
                          'whoosh_peaks_found_120ms': f'{found}/{n_gt}', 'swells': n_sw,
                          'swells_within_300ms_of_any_placed_sound': near_any}), flush=True)


if __name__ == '__main__':
    main()
