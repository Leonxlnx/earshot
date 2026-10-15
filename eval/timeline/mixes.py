"""Tuning set for the timeline event detector: synthetic mixtures of REAL recordings with exact ground truth.

    .venv/Scripts/python.exe eval/timeline/mixes.py [--n 8] [--dur 30] [--out out/timeline/mixes]

Only recordings that the Clawd soundtrack does NOT use (unused files in the Clawd asset library + its _rejected
folder) go into these mixes, so detector thresholds tuned here are evaluated on the Clawd soundtrack without leakage.
Each mix: 2-4 beds (music or ambience, hard switches = section boundaries) and one-shot hits every 0.35-1.6 s at a
random level relative to the bed (seeded, deterministic). Ground truth per hit: attack time (first 1 ms within 20 dB
of the hit's peak, the Clawd mixer's own definition), peak-aligned time for whooshes, and the hit-to-bed SNR in dB.
Writes <out>/mix_<k>.wav, mix_<k>_bed.wav (the beds alone) + mix_<k>.json (media stays in out/, never in git).
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from earshot import audio  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

SR = audio.SR
ASSETS = EP.ASSETS
MIX_REPORT = f'{EP.FILM}/out/qa/mix_report.json'
HITS = ('foley', 'garden', 'keyboard', 'magic', 'toy')


def unused(cat):
    used = {os.path.basename(f) for f in json.load(open(MIX_REPORT))['files_used']}
    out = []
    for d in (os.path.join(ASSETS, cat), os.path.join(ASSETS, '_rejected', cat)):
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if os.path.splitext(f)[1].lower() in audio.AUDIO_EXTS and f not in used]
    return out


def env1(x):
    n = len(x) // 48
    return np.abs(x[: n * 48]).reshape(n, 48).max(axis=1)


def attack_ms(x):
    e = env1(x)
    return int(np.argmax(e >= e.max() * 0.1)) if len(e) else 0


def one_shots(path, kmax=3):
    """single isolated hits inside a recording: [(array, attack_s)]"""
    x = audio.load(path, mono=True)[: 60 * SR]
    if len(x) < SR // 10:
        return []
    r = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(240) / 240, 'same')[::240] + 1e-12)
    db = 20 * np.log10(r)
    out, used = [], np.zeros(len(db), bool)
    for ip in np.argsort(-db):
        if len(out) >= kmax or db[ip] < db.max() - 25:
            break
        if used[max(0, ip - 60): ip + 60].any():
            continue
        used[max(0, ip - 60): ip + 60] = True
        pre, post = db[max(0, ip - 40): max(0, ip - 4)], db[ip + 12: ip + 100]
        if len(pre) < 10 or pre.max() > db[ip] - 20 or (len(post) and post.max() > db[ip] - 12):
            continue
        a = max(0, (ip - 40) * 240)
        seg = x[a: a + int(0.6 * SR)].copy()
        fade = int(0.05 * SR)
        seg[-fade:] *= np.linspace(1, 0, fade)
        out.append((seg, attack_ms(seg) / 1000))
    return out


def whooshes(path):
    x = audio.load(path, mono=True)[: 4 * SR]
    if len(x) < SR // 5:
        return []
    fade = int(0.03 * SR)
    x = x.copy()
    x[-fade:] *= np.linspace(1, 0, fade)
    e = env1(x)
    return [(x, (int(np.argmax(e)) + 0.5) / 1000)]


def bed(path, n, rng):
    x = audio.load(path, mono=True)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / max(1, len(x)))) + 1)
    a = int(rng.uniform(0, max(1, len(x) - n - 1)))
    seg = x[a: a + n].copy()
    return seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9)


def build(k, dur, pools, rng):
    music, amb, hits, swoosh = pools
    nb = int(rng.integers(2, 5))
    cuts = np.sort(rng.uniform(4, dur - 4, nb - 1))
    while len(cuts) > 1 and np.min(np.diff(cuts)) < 4:
        cuts = np.sort(rng.uniform(4, dur - 4, nb - 1))
    edges = [0.0] + [round(float(c), 2) for c in cuts] + [float(dur)]
    y = np.zeros(int(dur * SR), np.float32)
    beds = []
    for j in range(nb):
        src = music if (j % 2 == 0) != (rng.uniform() < 0.3) else amb
        f = src[int(rng.integers(len(src)))]
        lvl = rng.uniform(-24, -16) if src is music else rng.uniform(-40, -28)
        a, b = int(edges[j] * SR), int(edges[j + 1] * SR)
        y[a:b] = bed(f, b - a, rng) * 10 ** (lvl / 20)
        beds.append({'start': edges[j], 'end': edges[j + 1], 'kind': 'music' if src is music else 'ambience',
                     'file': os.path.basename(f), 'rms_dbfs': round(lvl, 1)})
    bed_only = y.copy()
    gt, t = [], rng.uniform(0.3, 0.8)
    while t < dur - 0.8:
        is_w = rng.uniform() < 0.12
        pool = swoosh if is_w else hits
        sig, al, cat, name = pool[int(rng.integers(len(pool)))]
        lo = int(max(0, (t - 0.05) * SR))
        bed_db = 20 * np.log10(np.sqrt(np.mean(bed_only[lo: lo + int(0.3 * SR)] ** 2)) + 1e-9)
        snr = rng.uniform(-14, 14)
        pk = 20 * np.log10(np.sqrt(np.max(np.convolve(sig ** 2, np.ones(480) / 480, 'same'))) + 1e-9)
        g = 10 ** ((bed_db + snr - pk) / 20)
        s0 = int(round(t * SR)) - int(round(al * SR))
        a, b = max(0, s0), min(len(y), s0 + len(sig))
        y[a:b] += (sig[a - s0: b - s0] * g).astype(np.float32)
        gt.append({'t': round(float(t), 4), 'align': 'peak' if is_w else 'attack', 'snr_db': round(float(snr), 1),
                   'cat': cat, 'file': name})
        t += rng.uniform(0.35, 1.6) + (1.0 if is_w else 0.0)
    pk = np.abs(y).max()
    if pk > 0.89:
        y *= 0.89 / pk
        bed_only *= 0.89 / pk
    return y, bed_only, {'beds': beds, 'boundaries': edges[1:-1], 'events': gt, 'duration': dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--dur', type=float, default=30.0)
    ap.add_argument('--out', default='out/timeline/mixes')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20261014)
    music = unused('music')
    amb = unused('ambience')
    hits = [(x, al, c, os.path.basename(f)) for c in HITS for f in unused(c) for x, al in one_shots(f)]
    sw = [(x, al, 'motion', os.path.basename(f)) for f in unused('motion') for x, al in whooshes(f)]
    print(f'pools: {len(music)} music, {len(amb)} ambience, {len(hits)} one-shots, {len(sw)} whooshes')
    for k in range(a.n):
        y, b, meta = build(k, a.dur, (music, amb, hits, sw), rng)
        sf.write(os.path.join(a.out, f'mix_{k}.wav'), y, SR, subtype='FLOAT')
        sf.write(os.path.join(a.out, f'mix_{k}_bed.wav'), b, SR, subtype='FLOAT')
        json.dump(meta, open(os.path.join(a.out, f'mix_{k}.json'), 'w'), indent=1)
        print(f'mix_{k}: {len(meta["events"])} events, boundaries {meta["boundaries"]}')


if __name__ == '__main__':
    main()
