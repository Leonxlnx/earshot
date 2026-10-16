"""earshot timeline (detail high) + sheet over each reference film: JSON, transcript and sheet PNG in out/refs_pro/_work.

    EARSHOT_THREADS=2 python eval/forge/pro/run_timeline.py ref_4 ref_1
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from videos import WORK, path  # noqa: E402
from earshot import api  # noqa: E402
from earshot.report import to_json  # noqa: E402

for ref in sys.argv[1:]:
    t0 = time.time()
    p = path(ref)
    r = api.timeline(p, detail='high')
    os.makedirs(f'{WORK}/tl', exist_ok=True)
    with open(f'{WORK}/tl/{ref}.json', 'w', encoding='utf-8') as fh:
        fh.write(to_json(r))
    with open(f'{WORK}/tl/{ref}.txt', 'w', encoding='utf-8') as fh:
        fh.write(r['text'])
    t1 = time.time()
    s = api.sheet(p, f'{WORK}/sheets/{ref}_full.png', detail='high', width=2400, tl=r)
    print(ref, f'timeline {t1 - t0:.0f}s sheet {time.time() - t1:.0f}s', len(r['events']), 'events', flush=True)
