"""Many zoomed crop sheets in one process (python start-up is slow on the shared laptop).

    python eval/forge/pro/crops_batch.py specs.json
specs.json: [{"ref": "ref_2", "out": "ref_2_a", "cols": 3, "specs": [[t_center, half_width, "label"], ...]}, ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viewer  # noqa: E402
from videos import WORK  # noqa: E402

for job in json.load(open(sys.argv[1], encoding='utf-8')):
    out = f"{WORK}/view/{job['out']}.png"
    print(viewer.crops(job['ref'], [tuple(s) for s in job['specs']], cols=job.get('cols', 3), out=out), flush=True)
