"""bf16 vs fp32 CLAP on this CPU: embedding agreement and speed (models load twice; no cache is used or written).

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/precision.py      # -> eval/semantics/precision.json

Input: the first 10 s of 12 real recordings (2 per category folder of the Clawd project's library, read-only), each
embedded in fp32 and in bf16; cosine between the two embeddings of the same clip, and seconds per 10 s window
(batch 4, after one warm-up batch). Whether bf16 is used at all depends on the CPU: earshot.models.clap.native_bf16.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from earshot import audio, config  # noqa: E402
from earshot.models import clap  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

ASSETS = EP.ASSETS
CATS = ['ambience', 'foley', 'keyboard', 'motion', 'music', 'toy']


def clips():
    out = []
    for c in CATS:
        fs = sorted(f for f in os.listdir(os.path.join(ASSETS, c)) if os.path.splitext(f)[1].lower() in audio.AUDIO_EXTS)
        for f in fs[:2]:
            out.append(audio.load(os.path.join(ASSETS, c, f), mono=True, end=10.0))
    return out


def run(precision, xs):
    config.PRECISION = precision
    c = clap.Clap()
    c._forward(xs[:4], 'repeatpad', 4)  # warm-up
    t = time.time()
    e = c._forward(xs, 'repeatpad', 4)
    return c.precision, e, (time.time() - t) / len(xs)


def main():
    import torch
    xs = clips()
    p32, e32, s32 = run('fp32', xs)
    p16, e16, s16 = run('bf16', xs)
    cos = np.sum(e32 * e16, axis=1) / (np.linalg.norm(e32, axis=1) * np.linalg.norm(e16, axis=1))
    res = {'torch': torch.__version__, 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
           'native_bf16': clap.native_bf16(torch), 'threads': config.threads(), 'clips': len(xs),
           'used': [p32, p16], 'cosine_mean': round(float(cos.mean()), 5), 'cosine_min': round(float(cos.min()), 5),
           's_per_window_fp32': round(s32, 3), 's_per_window_bf16': round(s16, 3),
           'speedup_bf16': round(s32 / s16, 2) if s16 else None}
    json.dump(res, open(os.path.join(HERE, 'precision.json'), 'w'), indent=1)
    print(res)


if __name__ == '__main__':
    main()
