"""The replication benchmark: can `earshot fit` re-make professional sound effects?

    EARSHOT_THREADS=3 python eval/forge/replicate.py --targets <dir> --out <dir>        # fit + measure every target
    python eval/forge/replicate.py --targets <dir> --out <dir> --only pop,key --limit 3       # a subset
    python eval/forge/replicate.py --targets <dir> --out <dir> --report               # tables from the cache only
    python eval/forge/replicate.py --targets <dir> --out <dir> --remeasure            # props from saved renders
    python eval/forge/replicate.py --targets <dir> --out <dir> --split-file held2.json --split heldout   # a frozen list
    PYTHONPATH=<other checkout> python eval/forge/replicate.py ...       # that checkout's engine on the same targets
    (or EARSHOT_REPLICATE_TARGETS / EARSHOT_REPLICATE_OUT instead of --targets / --out)

Targets: a folder of clean one-gesture clips, `<category>/<clip>.wav` (48 kHz), optionally with a `catalog.json`
({'clips': [{'clip': '<category>/<file>.wav', 'category', 'family', ...}]}) whose records are kept with the results.
The clips are licensed third-party audio in the study this was written for: this script never copies them into the
repo, and it refuses an output or render folder inside the repo unless it is under the gitignored `out/`.

Per target, in ONE process (CLAP loads once), resumable (a clip with a result JSON is skipped unless --force):
  fit       `earshot.forge.fit.auto.fit(x, clap=True)`: every candidate fitted, scored with quality.score + CLAP and
            the lowest kept (the same choice as `earshot fit --clap`); the winning kind and every candidate's score
  renders   seed 0 (the identity render: the fit as measured, the target's length), the floor (seeds 11 / 12) and
            four takes (seeds 1-4 at vary 0.3, take=True: what a film would use; a spectral model has no vary knob,
            its takes are fresh noise). Takes are compared to the target over the target's length (CLAP repeat-pads
            a short clip, so another length reads as another rhythm); their property and audit measures use the full
            take
  metrics   (seed 0 against the target, and the mean over the takes)
            clap        CLAP cosine to the target (and the takes' CLAP to the seed-0 render)
            mel_gap     quality mel dB minus the model's own floor (render seed 11 vs 12; docs/FIT.md)
            mel_head / mel_tail   the same mel distance split at 65 ms after the onset (fft 256 / 1024), with floors
            env_r, onset_err (sharpness octaves), tone_db, excess_db / missing_db   (earshot.forge.fit.quality)
            attack      catalog definition: 1 ms envelope (5 ms window, 10 ms smoothing), first ms within 20 dB of
                        the peak to the peak; error = log2((render + 2 ms) / (target + 2 ms))
            centroid    over the frames within 20 dB of the loudest (2048-point STFT); error in octaves. Also the
                        head (first 15 ms) and body centroids, and layering = log2(head / body): a transient on a body
            width       side-minus-mid dB over the gesture (floored at -40), L/R correlation, and the tail's
                        correlation (from peak + 30 ms; a reverb space decorrelates); error = render - target
            glide       the dominant ridge's line fit over tonal frames (kept when r^2 >= 0.6, else 0), semitones
            tail        peak to the last 1 ms frame above the target's threshold (6 dB over its 60 ms pre-roll bed,
                        between -45 and -10 dB re peak; the render uses the same level re its own peak); error =
                        log2((render + 10 ms) / (target + 10 ms)); also t20, the decay rate and the tail's flatness
            pulses      peaks of the 5 ms dB envelope (prominence 6 dB, 8 ms apart, within 25 dB of the top)
            gate        the steepest 5 ms fall of the 1 ms dB envelope from within 30 dB of the top (a hard stop or a
                        gated pulse falls 30-60 dB; capped at 60 dB)
            low share   energy share under 150 Hz
  audit     earshot's polish audit on the takes: `forge.sfx.check` warnings the target itself does not raise, and
            `polish.PolishRef([target]).check` hard artifacts beyond the target's; loudness spread over seed 0 + the
            takes; takes_ok = the bank's take check (docs/SFX.md "Fit entries"): no new warning or hard artifact,
            CLAP >= 0.90 to the seed-0 render and within 0.06 of its CLAP to the target, excess within 1.5 dB,
            spread <= 2 LU
  grade     high: CLAP >= 0.92 and mel gap <= 2.5 dB (docs/SFX.md, the bank's ship bar); good: CLAP >= 0.88 and gap
            <= 3.5 dB (the CC0 fit survey); fair: CLAP >= 0.80 and gap <= 5.0 dB; poor: the rest
  split     deterministic 70 % dev / 30 % held-out by clip id (sha1), stratified by category; categories with one
            clip are pooled by family (catalog 'family') so the held-out set still sees sweeps, builds, ...
            --split-file: a split frozen in a JSON file instead (read_split; its 'clips' records, when present, are
            the targets: a fresh held-out set drawn from a wider pool once the first one has been opened)

Outputs (--out): baseline_fit.json (every clip, per-category medians, dev / held-out / all summaries, failure
analysis numbers), BASELINE.md (tables; `analysis.md` in --out, hand-written, is appended when present),
split.json, clips/<category>/<stem>.json (the per-clip cache), renders under --renders (default
<out>/baseline_renders/<category>/<stem>/: fit_s0.wav, take_s1..4.wav, audition.wav = target | fit | takes,
pair.png, model.json).
"""
import os

for _k in ('EARSHOT_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '3')

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE)).replace(chr(92), '/')
if REPO not in sys.path and not os.environ.get('PYTHONPATH'):
    sys.path.insert(0, REPO)          # this checkout's earshot, not whatever the venv's editable install points at;
                                      # a PYTHONPATH decides instead (another checkout's engine on the same benchmark)

SR = 48000
VARY = 0.3                 # the bank's shipped vary for fit takes
TAKE_SEEDS = (1, 2, 3, 4)
FLOOR_SEEDS = (11, 12)
SPLIT_SALT = 'earshot-replicate-v1'
HELD_FRACTION = 0.3
HEAD_S = 0.065             # mel head / tail split after the onset (the 5 ms lead of quality._align included)
GRADES = ('poor', 'fair', 'good', 'high')
GRADE_BARS = {'high': (0.92, 2.5), 'good': (0.88, 3.5), 'fair': (0.80, 5.0)}
# a property "fails" beyond these (seed 0 vs the target): used for the failure analysis and the "why" lines
TOL = {'env_r': 0.90, 'onset_err': 1.0, 'attack_oct': 1.0, 'centroid_oct': 0.3, 'width_db': 3.0, 'glide_st': 2.0,
       'tail_oct': 0.5, 'pulses': 2, 'excess_db': 1.5, 'missing_db': 1.5, 'mel_tail_gap': 3.0, 'layer_oct': 0.7,
       'tail_corr': 0.3, 'lo150': 0.15, 'gate_db': 15.0}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def jdump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def r(v, nd=3):
    if v is None:
        return None
    v = float(v)
    return None if not math.isfinite(v) else round(v, nd)


# ============================================================================================= targets + split
def find_targets(root, records=None):
    """[{id, path, category, family, rec}] from `records` (catalog records, 'clip' relative to root) when given, else
    root/catalog.json if present, else root/<category>/*.wav. The id is <category>/<file stem>"""
    root = root.replace(chr(92), '/').rstrip('/')
    cat_path = f'{root}/catalog.json'
    out = []
    if records is None and os.path.exists(cat_path):
        with open(cat_path, encoding='utf-8') as fh:
            C = json.load(fh)
        records = C['clips'] if isinstance(C, dict) else C
    if records is not None:
        for rec in records:
            rel = rec['clip'].replace(chr(92), '/')
            cat = rec.get('category') or rel.split('/')[0]
            cid = f"{cat}/{os.path.splitext(os.path.basename(rel))[0]}"
            out.append({'id': cid, 'path': f'{root}/{rel}', 'category': cat, 'family': rec.get('family'),
                        'rec': rec})
    else:
        for p in sorted(glob.glob(f'{root}/*/*.wav')):
            p = p.replace(chr(92), '/')
            cat = p.split('/')[-2]
            if cat.startswith('_'):
                continue
            cid = f"{cat}/{os.path.splitext(os.path.basename(p))[0]}"
            out.append({'id': cid, 'path': p, 'category': cat, 'family': None, 'rec': {}})
    out.sort(key=lambda t: t['id'])
    return out


def _h(cid):
    return hashlib.sha1(f'{SPLIT_SALT}:{cid}'.encode('utf-8')).hexdigest()


def make_split(targets):
    """deterministic stratified split: per stratum (the category; one-clip categories pooled by family) the
    floor(0.3 n + 0.5) clips with the smallest sha1(salt:id) are held out"""
    from collections import defaultdict
    by_cat = defaultdict(list)
    for t in targets:
        by_cat[t['category']].append(t)
    strata = defaultdict(list)
    for cat, ts in by_cat.items():
        if len(ts) >= 2:
            strata[f'category:{cat}'] += ts
        else:
            strata[f"family:{ts[0].get('family') or 'other'}"] += ts
    held, dev, info = [], [], {}
    for s, ts in sorted(strata.items()):
        ts = sorted(ts, key=lambda t: _h(t['id']))
        k = int(math.floor(HELD_FRACTION * len(ts) + 0.5))
        held += [t['id'] for t in ts[:k]]
        dev += [t['id'] for t in ts[k:]]
        info[s] = {'n': len(ts), 'held': k}
    return {'method': f'sha1("{SPLIT_SALT}:" + clip id), per stratum the floor(0.3 n + 0.5) smallest are held out; '
                      'strata = categories with >= 2 clips, one-clip categories pooled by family',
            'dev': sorted(dev), 'heldout': sorted(held), 'strata': info,
            'n_dev': len(dev), 'n_heldout': len(held)}


def read_split(path, targets):
    """a split frozen in a file (--split-file): JSON {'heldout': [ids], 'dev': [ids] (default: every other target),
    'clips': [catalog records] (optional: the targets, in place of the target folder's catalog.json), 'rule' / 'method'
    (how it was chosen)}"""
    with open(path, encoding='utf-8') as fh:
        S = json.load(fh)
    ids = {t['id'] for t in targets}
    held = sorted(i for i in S.get('heldout', []) if i in ids)
    dev = sorted(i for i in S['dev'] if i in ids) if S.get('dev') else sorted(ids - set(held))
    return {'method': f"from {os.path.basename(path)}: " + str(S.get('method') or S.get('rule') or 'a frozen list'),
            'dev': dev, 'heldout': held, 'strata': {'file': {'n': len(ids), 'held': len(held)}},
            'n_dev': len(dev), 'n_heldout': len(held)}


# ============================================================================================= measurements
def _stereo(y):
    y = np.asarray(y, np.float64)
    if y.ndim == 1:
        return np.stack([y, y], 1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, 1)
    return y


def _db(p, floor=1e-20):
    return 10 * np.log10(np.maximum(p, floor))


def env1(m):
    """1 ms hops, 5 ms window power envelope (the catalog's stage A / measure.py)"""
    n = len(m)
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    w, hop = int(0.005 * SR), int(0.001 * SR)
    cen = np.arange(0, n, hop)
    a = np.clip(cen - w // 2, 0, n)
    b = np.clip(cen + w // 2, 0, n)
    return (c[b] - c[a]) / np.maximum(b - a, 1)


def _stft(m, nfft=2048, hop=240):
    from scipy import signal
    seg = np.asarray(m, np.float64)
    if len(seg) < nfft:
        seg = np.pad(seg, (0, nfft - len(seg)))
    nfr = 1 + (len(seg) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    win = signal.get_window('hann', nfft)
    P = (np.abs(np.fft.rfft(seg[idx] * win, axis=1)) ** 2).T          # bins x frames
    return P, np.fft.rfftfreq(nfft, 1 / SR), (np.arange(nfr) * hop + nfft / 2) / SR


def _centroid(spec, F):
    tot = spec.sum()
    return float((spec * F).sum() / tot) if tot > 0 else None


def bed_threshold(x):
    """the tail threshold re peak (dB) of a target: 3 dB over its floor, the lower of the pre-roll level (median 1 ms
    level 5-50 ms) and the level just before the clip's 10 ms fade-out (a clean clip ends where its tail meets the
    bed, or 20 ms before the next event), kept within -45 .. -12 dB"""
    m = _stereo(x).mean(1)
    E1 = env1(m)
    sm = np.convolve(E1, np.ones(10) / 10, mode='same')
    pk = _db(float(sm.max()) + 1e-30)
    if len(E1) < 80:
        return -40.0
    fl = min(_db(float(np.median(E1[5:50])) + 1e-30), _db(float(np.median(E1[-25:-12])) + 1e-30))
    return float(np.clip(fl + 3.0 - pk, -45.0, -12.0))


def props(y, thr_rel=-40.0, on_ms=None):
    """the catalog-style property measures of one sound (module doc). on_ms: the onset (ms into the clip) when it is
    known (a catalogued target and its time-aligned renders); else the first 1 ms frame within 30 dB of the peak"""
    from scipy import signal
    y = _stereo(y)
    m = y.mean(1)
    n = len(m)
    out = {}
    E1 = env1(m)
    sm = np.convolve(E1, np.ones(10) / 10, mode='same')
    ip = int(np.argmax(sm))
    pk = float(sm[ip]) + 1e-30
    if on_ms is None or not 0 <= on_ms < ip + 1:
        on30 = np.nonzero(sm >= pk * 1e-3)[0]
        on = int(on30[0]) if len(on30) else 0
    else:
        on = int(on_ms)
    i0 = max(0, on - 10)
    above = np.nonzero(sm[i0:ip + 1] >= pk * 1e-2)[0]
    out['attack_ms'] = float(ip - (i0 + above[0])) if len(above) else 0.0
    after = sm[ip:]
    w20 = np.nonzero(after < pk * 1e-2)[0]
    out['t20_ms'] = float(w20[0]) if len(w20) else float(len(after))
    thr = pk * 10 ** (thr_rel / 10)
    live = np.nonzero(sm >= thr)[0]
    last = int(live[-1]) if len(live) else ip
    out['tail_ms'] = float(max(0, last - ip))
    out['tail_thr_db'] = round(thr_rel, 1)
    out['dur_ms'] = float(max(1, last - on))
    out['peak_ms'] = float(ip)
    # decay rate: line fit of the smoothed dB envelope from -6 dB to the threshold after the peak
    ed = _db(after / pk + 1e-30)
    sel = np.nonzero((ed <= -6) & (ed >= max(thr_rel, -45)))[0]
    sel = sel[sel <= max(1, last - ip)]
    out['decay_db_s'] = float(np.polyfit(sel * 0.001, ed[sel], 1)[0]) if len(sel) >= 5 else None
    # pulses: gates / stutter / runs in the body
    s5 = _db(np.convolve(E1, np.ones(5) / 5, mode='same') + 1e-14)
    seg = s5[on:last + 1] if last > on + 10 else s5
    pk_i, _ = signal.find_peaks(seg, prominence=6, distance=8)
    pk_i = [p for p in pk_i if seg[p] > seg.max() - 25]
    out['pulses'] = int(max(1, len(pk_i)))
    out['pulse_gap_ms'] = float(np.median(np.diff(pk_i))) if len(pk_i) >= 3 else None
    # gate: the steepest 5 ms fall of the 1 ms dB envelope from a level within 30 dB of the top (a hard stop / a
    # gated pulse falls 30-60 dB in 5 ms; a natural fast decay 5-15 dB), the clip's last 15 ms (its fade) excluded,
    # capped at 60 dB (a fall into digital silence)
    e1 = _db(E1 / (E1.max() + 1e-30) + 1e-30)
    hi_ = max(on + 1, min(len(e1) - 5, len(e1) - 15))
    cand = np.arange(on, hi_)
    cand = cand[e1[cand] >= max(thr_rel, -30.0)]
    out['gate_db'] = float(min(60.0, max(0.0, np.max(e1[cand] - e1[cand + 5])))) if len(cand) else 0.0
    # spectrum over the frames within 20 dB of the loudest
    P, F, tt = _stft(m)
    fe = P.sum(0)
    act = fe >= fe.max() * 1e-2 if fe.max() > 0 else np.zeros(len(fe), bool)
    if act.any():
        spec = P[:, act].mean(1)
        tot = spec.sum() + 1e-30
        out['centroid_hz'] = _centroid(spec, F)
        out['lo150'] = float(spec[F < 150].sum() / tot)
        out['hi4k'] = float(spec[F >= 4000].sum() / tot)
    else:
        out['centroid_hz'], out['lo150'], out['hi4k'] = None, 0.0, 0.0
    # head (first 15 ms from the onset) vs body centroid: a transient layered on a body
    a0, a1 = on * 48, min(n, on * 48 + int(0.015 * SR))
    b1 = min(n, max(a1 + 480, last * 48))
    out['centroid_head_hz'] = out['centroid_body_hz'] = out['layer_oct'] = None
    if a1 - a0 >= 256:
        sh = np.abs(np.fft.rfft(m[a0:a1] * np.hanning(a1 - a0), 4096)) ** 2
        out['centroid_head_hz'] = _centroid(sh, np.fft.rfftfreq(4096, 1 / SR))
    if b1 - a1 >= 1024:
        Pb, Fb, _ = _stft(m[a1:b1], 2048, 480)
        out['centroid_body_hz'] = _centroid(Pb.mean(1), Fb)
    if out['centroid_head_hz'] and out['centroid_body_hz']:
        out['layer_oct'] = float(np.log2(out['centroid_head_hz'] / out['centroid_body_hz']))
    # glide: the dominant ridge over tonal frames (catalog stage A, without a bed to subtract)
    band = (F >= 80) & (F <= 12000)
    Fb_ = F[band]
    rid, ton, fidx = [], [], []
    for j in np.nonzero(act)[0]:
        col = P[band, j]
        if col.max() <= 0:
            continue
        k = int(np.argmax(col))
        fr = Fb_[k]
        share = col[(Fb_ >= fr / 2 ** 0.25) & (Fb_ <= fr * 2 ** 0.25)].sum() / (col.sum() + 1e-30)
        rid.append(fr)
        ton.append(share)
        fidx.append(j)
    out['glide_st'], out['tonal'], out['ridge_hz'] = 0.0, None, None
    if len(rid) >= 2:
        rid, ton, fidx = np.array(rid), np.array(ton), np.array(fidx)
        out['tonal'] = float(ton.mean())
        good = ton >= 0.45
        out['ridge_hz'] = float(np.median(rid[good])) if good.any() else float(np.median(rid))
        # the longest continuous ridge run (< 3 semitones and <= 2 frames between tonal frames: no partial swaps)
        gi = np.nonzero(good)[0]
        runs, cur = [], []
        for k in gi:
            if cur and (fidx[k] - fidx[cur[-1]] > 2 or abs(np.log2(rid[k] / rid[cur[-1]])) >= 0.25):
                runs.append(cur)
                cur = []
            cur.append(k)
        if cur:
            runs.append(cur)
        run = max(runs, key=len) if runs else []
        if len(run) >= 4:
            t_ = fidx[run] * 240 / SR
            lf = np.log2(rid[run])
            A_ = np.vstack([t_, np.ones_like(t_)]).T
            coef, *_ = np.linalg.lstsq(A_, lf, rcond=None)
            pred = A_ @ coef
            ss = ((lf - lf.mean()) ** 2).sum()
            r2 = 1 - ((lf - pred) ** 2).sum() / ss if ss > 1e-9 else 0.0
            if r2 >= 0.6:
                out['glide_st'] = float(12 * (pred[-1] - pred[0]))
    # tail texture: spectral flatness of the frames after peak + 30 ms that are still above the threshold
    tsel = (tt >= (ip + 30) / 1000) & (tt <= max(last, ip + 31) / 1000) & (fe >= fe.max() * 10 ** (thr_rel / 10))
    out['tail_flatness'] = None
    if tsel.sum() >= 2:
        Pt = P[(F >= 100) & (F <= 16000)][:, tsel] + 1e-30
        fl = np.exp(np.mean(np.log(Pt), 0)) / Pt.mean(0)
        out['tail_flatness'] = float(np.average(fl, weights=Pt.sum(0)))
    # stereo over the gesture (onset to the tail end, at most 1.5 s) and over the tail
    xa, xb = on * 48, min(n, max(on * 48 + 480, last * 48 + 48), on * 48 + int(1.5 * SR))
    L, R = y[xa:xb, 0], y[xa:xb, 1]
    mid, side = 0.5 * (L + R), 0.5 * (L - R)
    pm, ps = float((mid ** 2).mean()) + 1e-30, float((side ** 2).mean())
    out['side_minus_mid_db'] = float(max(-40.0, _db(ps / pm + 1e-30)))
    out['corr'] = float((L * R).sum() / (np.sqrt((L * L).sum() * (R * R).sum()) + 1e-30))
    ta, tb = (ip + 30) * 48, min(n, last * 48)
    out['tail_corr'] = None
    if tb - ta >= int(0.02 * SR):
        Lt, Rt = y[ta:tb, 0], y[ta:tb, 1]
        out['tail_corr'] = float((Lt * Rt).sum() / (np.sqrt((Lt * Lt).sum() * (Rt * Rt).sum()) + 1e-30))
    return {k: (r(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def mel_split(a, b):
    """quality's mel distance (levelled, aligned at a's onset) split into the first HEAD_S and the rest"""
    from earshot.forge.fit import quality as Q
    A, B = Q._align(Q.mono(a), Q.mono(b))
    acc = {'head': [], 'tail': []}
    for nf in (256, 1024):
        MA, MB = Q.melspec(A, nf), Q.melspec(B, nf)
        top = max(MA.max(), MB.max())
        MA, MB = np.maximum(MA, top - 80), np.maximum(MB, top - 80)
        act = np.maximum(MA, MB) > top - 60
        t = np.arange(len(MA)) * (nf // 4) / SR
        D = np.abs(MA - MB)
        for key, sel in (('head', t < HEAD_S), ('tail', t >= HEAD_S)):
            mk = act & sel[:, None]
            if mk.sum() >= 8:
                acc[key].append(float(D[mk].mean()))
    return {f'mel_{k}': (r(np.mean(v)) if v else None) for k, v in acc.items()}


def errors(pt, pr):
    """render property errors against the target (signed; module doc)"""
    def ratio(key, floor):
        if pt.get(key) is None or pr.get(key) is None:
            return None
        return r(np.log2((pr[key] + floor) / (pt[key] + floor)))

    def diff(key, nd=3):
        if pt.get(key) is None or pr.get(key) is None:
            return None
        return r(pr[key] - pt[key], nd)
    e = {'attack_oct': ratio('attack_ms', 2.0), 'tail_oct': ratio('tail_ms', 10.0), 't20_oct': ratio('t20_ms', 5.0),
         'centroid_oct': (r(np.log2(pr['centroid_hz'] / pt['centroid_hz']))
                          if pt.get('centroid_hz') and pr.get('centroid_hz') else None),
         'width_db': diff('side_minus_mid_db', 2), 'corr': diff('corr'), 'tail_corr': diff('tail_corr'),
         'glide_st': diff('glide_st', 2), 'pulses': diff('pulses', 0), 'lo150': diff('lo150'),
         'layer_oct': diff('layer_oct'), 'tail_flatness': diff('tail_flatness'), 'gate_db': diff('gate_db', 1),
         'head_oct': (r(np.log2(pr['centroid_head_hz'] / pt['centroid_head_hz']))
                      if pt.get('centroid_head_hz') and pr.get('centroid_head_hz') else None),
         'body_oct': (r(np.log2(pr['centroid_body_hz'] / pt['centroid_body_hz']))
                      if pt.get('centroid_body_hz') and pr.get('centroid_body_hz') else None)}
    if pt.get('decay_db_s') and pr.get('decay_db_s') and pt['decay_db_s'] < 0 and pr['decay_db_s'] < 0:
        e['decay_oct'] = r(np.log2(pt['decay_db_s'] / pr['decay_db_s']))     # > 0: the render decays slower
    else:
        e['decay_oct'] = None
    return e


def grade(clap, gap):
    if clap is None or gap is None:
        return 'poor'
    for g in ('high', 'good', 'fair'):
        c, d = GRADE_BARS[g]
        if clap >= c and gap <= d:
            return g
    return 'poor'


# ============================================================================================= one target
def _trim_to(y, n, fade_s=0.01):
    y = np.array(y[:n], np.float64)
    if len(y) < n:
        pad = np.zeros((n - len(y),) + y.shape[1:])
        y = np.concatenate([y, pad])
    k = min(len(y) // 3, int(fade_s * SR))
    if k > 1:
        ramp = np.linspace(1, 0, k)
        y[-k:] *= ramp[:, None] if y.ndim == 2 else ramp
    return y


def _wav(path, y):
    import soundfile as sf
    sf.write(path, np.asarray(y, np.float32), SR, subtype='FLOAT')


def run_target(t, rdir, eX_cache=None):
    from earshot.forge import polish, sfx
    from earshot.forge.fit import auto
    from earshot.forge.fit import quality as Q
    t0 = time.time()
    X = Q.load(t['path'])
    if X.ndim == 1:
        X = np.stack([X, X], 1)
    n = len(X)
    models = {}
    m = auto.fit(X, clap=True, name=os.path.basename(t['path']), verbose=False, collect=models)
    fe = m['fit_engine']
    t_fit = time.time() - t0
    kind = auto.kind_of(m)
    kn = auto.knobs_of(m)
    y0 = auto.render(m, seed=0)
    yf = [auto.render(m, seed=s) for s in FLOOR_SEEDS]
    has_vary = 'vary' in kn
    takes = [auto.render(m, seed=s, take=True, **({'vary': VARY} if has_vary else {})) for s in TAKE_SEEDS]
    takes_cmp = [_trim_to(_stereo(y) if X.ndim == 2 else y, n) for y in takes]
    E = Q.clap_embed([X, y0] + yf + takes_cmp)
    eX, e0 = E[0], E[1]
    q0 = Q.quality(X, y0)
    q0['clap_cos'] = r(eX @ e0, 4)
    fl = Q.quality(yf[0], yf[1])
    fl['clap_cos'] = r(E[2] @ E[3], 4)
    q0.update(mel_split(X, y0))
    fl.update(mel_split(yf[0], yf[1]))
    thr = bed_threshold(X)
    rec = t.get('rec') or {}
    on_ms = (int(round((rec['onset'] - rec['clip_start']) * 1000)) if rec.get('onset') is not None
             and rec.get('clip_start') is not None else None)
    pT = props(X, thr, on_ms)
    p0 = props(y0, thr, on_ms)
    e0r = errors(pT, p0)
    # takes
    aT = sfx.check(X, category=t['category'])
    t_codes = {w['code'] for w in aT['warnings']}
    PR = polish.PolishRef([X])
    c0 = PR.check(y0)
    trows = []
    for s, y, yc, e in zip(TAKE_SEEDS, takes, takes_cmp, E[4:]):
        qv = Q.quality(X, yc)
        pv = props(y, thr, on_ms)
        au = sfx.check(y, category=t['category'])
        new = [w for w in au['warnings'] if w['code'] not in t_codes]
        c = PR.check(y)
        trows.append({'seed': s, 'vary': VARY if has_vary else None, 'len_s': r(len(y) / SR, 3),
                      'clap': r(eX @ e, 4), 'clap_fit': r(e0 @ e, 4), 'mel_db': qv['mel_db'],
                      'env_r': qv['env_corr'], 'onset_err': qv['onset_err'], 'excess_db': qv['excess_db'],
                      'props': pv, 'err': errors(pT, pv), 'audit_new': new,
                      'audit_notes': [w['code'] for w in au.get('notes', [])], 'hard': c['hard'],
                      'art_pen': c['art_pen'], 'polish_pen': c['polish_pen'], 'words': c['words']})
    spread = polish.loudness_spread([y0] + takes)
    gap = r(q0['mel_db'] - fl['mel_db'])
    g = grade(q0['clap_cos'], gap)
    fails = []                 # the take check (seed 0 is the identity render: it keeps the target's cut, not judged)
    for row in trows:
        if row['hard']:
            fails.append(f"take {row['seed']}: hard artifacts {row['hard']}")
        if row['audit_new']:
            fails.append(f"take {row['seed']}: audit " + ', '.join(w['code'] for w in row['audit_new']))
        if row['clap_fit'] is not None and row['clap_fit'] < 0.90:
            fails.append(f"take {row['seed']}: CLAP to the fit {row['clap_fit']:.3f} < 0.90")
        if row['clap'] is not None and row['clap'] < q0['clap_cos'] - 0.06:
            fails.append(f"take {row['seed']}: CLAP to the target {row['clap']:.3f} drops > 0.06 under the fit")
        if row['excess_db'] > q0['excess_db'] + 1.5:
            fails.append(f"take {row['seed']}: excess {row['excess_db']:.2f} dB (fit {q0['excess_db']:.2f})")
    if (spread.get('spread_lu') or 0) > 2.0:
        fails.append(f"loudness spread {spread['spread_lu']} LU > 2")

    def tmean(key, sub=None):
        v = [(row[sub] if sub else row).get(key) for row in trows]
        v = [a for a in v if a is not None]
        return r(np.mean(v), 4) if v else None
    take_err = {k: tmean(k, 'err') for k in trows[0]['err']}
    take_err_abs = {k: (r(np.mean([abs(row['err'][k]) for row in trows if row['err'].get(k) is not None]), 4)
                        if any(row['err'].get(k) is not None for row in trows) else None) for k in trows[0]['err']}
    # renders for listening
    os.makedirs(rdir, exist_ok=True)
    _wav(f'{rdir}/fit_s0.wav', y0)
    for s, y in zip(TAKE_SEEDS, takes):
        _wav(f'{rdir}/take_s{s}.wav', y)
    gapz = np.zeros((int(0.35 * SR), 2))
    parts = [_stereo(X), gapz, _stereo(y0), gapz] + sum(([_stereo(y), gapz] for y in takes), [])
    _wav(f'{rdir}/audition.wav', np.concatenate(parts))
    auto.save(m, f'{rdir}/model.json')
    try:
        Q.pair_png(f'{rdir}/pair.png', X, y0, title=f"{t['id']} -> fit ({fe['chosen']}) CLAP {q0['clap_cos']:.3f}",
                   q=q0)
    except Exception as ex:                               # the picture is a nice-to-have
        log(f'  pair.png failed: {ex}')
    cands = []
    for row in fe['candidates']:
        c = {'candidate': row['candidate'], 'score': row.get('score'), 'fit_s': row.get('fit_s'),
             'error': row.get('error')}
        if row.get('quality'):
            c.update({'clap': row['quality'].get('clap_cos'), 'mel_db': row['quality'].get('mel_db'),
                      'env_r': row['quality'].get('env_corr'), 'excess_db': row['quality'].get('excess_db')})
        cands.append(c)
    keep_rec = {k: t['rec'].get(k) for k in ('video', 'family', 'tier', 'confidence', 'duration_s', 'over_bed_db',
                                             'centroid_hz', 'attack_ms', 't20_ms', 'glide_semitones', 'tonal_share',
                                             'stereo_corr', 'side_minus_mid_db', 'pulses', 'pulse_gap_ms',
                                             'cut_by_next_event', 'merged_onsets', 'earshot_label', 'shape')
                if k in t['rec']}
    res = {
        'id': t['id'], 'category': t['category'], 'family': t.get('family'), 'len_s': r(n / SR, 4),
        'catalog': keep_rec,
        'chosen': fe['chosen'], 'kind': kind, 'character': fe.get('character'), 'candidates': cands,
        'channels': auto.channels_of(m), 'model_bytes': os.path.getsize(f'{rdir}/model.json'),
        'grade': g,
        'seed0': {'clap': q0['clap_cos'], 'mel_db': q0['mel_db'], 'floor_mel_db': fl['mel_db'], 'mel_gap': gap,
                  'mel_head': q0.get('mel_head'), 'mel_tail': q0.get('mel_tail'),
                  'floor_mel_head': fl.get('mel_head'), 'floor_mel_tail': fl.get('mel_tail'),
                  'mel_tail_gap': (r(q0['mel_tail'] - fl['mel_tail']) if q0.get('mel_tail') is not None
                                   and fl.get('mel_tail') is not None else None),
                  'env_r': q0['env_corr'], 'onset_err': q0['onset_err'], 'sharp_db_ms': q0['sharp_db_ms'],
                  'attack_q_ms': q0['attack_ms'], 'tone_db': q0['tone_db'], 'excess_db': q0['excess_db'],
                  'missing_db': q0['missing_db'], 'level_db': q0['level_db'], 'floor_clap': fl['clap_cos'],
                  'score': r(Q.score(q0)), 'err': e0r, 'props': p0, 'hard': c0['hard']},
        'target': {'props': pT, 'audit': [w['code'] for w in aT['warnings']],
                   'audit_notes': [w['code'] for w in aT.get('notes', [])]},
        'takes': {'vary': VARY if has_vary else None, 'rows': trows,
                  'clap_mean': tmean('clap'), 'clap_min': r(min(row['clap'] for row in trows), 4),
                  'clap_fit_mean': tmean('clap_fit'), 'mel_mean': tmean('mel_db'), 'env_r_mean': tmean('env_r'),
                  'err_mean': take_err, 'err_abs_mean': take_err_abs, 'loudness': spread,
                  'ok': not fails, 'fails': fails,
                  'n_new_warnings': int(sum(len(row['audit_new']) for row in trows)),
                  'n_hard': int(sum(len(row['hard']) for row in trows))},
        'seconds': {'fit': r(t_fit, 1), 'total': r(time.time() - t0, 1)},
    }
    return res


def remeasure(t, c, rdir):
    """recompute the property measures and errors of a cached clip from its saved renders (no fit, no CLAP): for a
    change to props() / errors() without refitting"""
    from earshot.forge.fit import quality as Q
    X = _stereo(Q.load(t['path']))
    rec = t.get('rec') or {}
    on_ms = (int(round((rec['onset'] - rec['clip_start']) * 1000)) if rec.get('onset') is not None
             and rec.get('clip_start') is not None else None)
    thr = bed_threshold(X)
    pT = props(X, thr, on_ms)
    p0 = props(Q.load(f'{rdir}/fit_s0.wav'), thr, on_ms)
    c['target']['props'] = pT
    c['seed0']['props'] = p0
    c['seed0']['err'] = errors(pT, p0)
    for row in c['takes']['rows']:
        row['props'] = props(Q.load(f"{rdir}/take_s{row['seed']}.wav"), thr, on_ms)
        row['err'] = errors(pT, row['props'])
    rows = c['takes']['rows']
    keys = rows[0]['err']
    c['takes']['err_mean'] = {k: (r(np.mean([row['err'][k] for row in rows if row['err'].get(k) is not None]), 4)
                                  if any(row['err'].get(k) is not None for row in rows) else None) for k in keys}
    c['takes']['err_abs_mean'] = {k: (r(np.mean([abs(row['err'][k]) for row in rows
                                                 if row['err'].get(k) is not None]), 4)
                                      if any(row['err'].get(k) is not None for row in rows) else None) for k in keys}
    return c


# ============================================================================================= report
def _med(v):
    v = [float(a) for a in v if a is not None and math.isfinite(float(a))]
    return float(np.median(v)) if v else None


def _q(v, p):
    v = [float(a) for a in v if a is not None]
    return float(np.percentile(v, p)) if v else None


def _absv(rows, path):
    out = []
    for c in rows:
        v = c
        for k in path:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        out.append(None if v is None else abs(float(v)))
    return out


def _get(c, path):
    v = c
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            return None
    return v


METRIC_COLS = [   # (name, path, abs?)
    ('clap', ('seed0', 'clap'), False), ('mel_gap', ('seed0', 'mel_gap'), False),
    ('mel_head', ('seed0', 'mel_head'), False), ('mel_tail_gap', ('seed0', 'mel_tail_gap'), False),
    ('env_r', ('seed0', 'env_r'), False), ('onset_err', ('seed0', 'onset_err'), False),
    ('attack_oct', ('seed0', 'err', 'attack_oct'), True), ('centroid_oct', ('seed0', 'err', 'centroid_oct'), True),
    ('width_db', ('seed0', 'err', 'width_db'), True), ('glide_st', ('seed0', 'err', 'glide_st'), True),
    ('tail_oct', ('seed0', 'err', 'tail_oct'), True), ('pulses', ('seed0', 'err', 'pulses'), True),
    ('excess_db', ('seed0', 'excess_db'), False), ('take_clap', ('takes', 'clap_mean'), False),
    ('take_tail_oct', ('takes', 'err_mean', 'tail_oct'), True),
]


def summarize(rows):
    if not rows:
        return {'n': 0}
    s = {'n': len(rows)}
    for name, path, ab in METRIC_COLS:
        s[name] = r(_med(_absv(rows, path) if ab else [_get(c, path) for c in rows]), 3)
    s['grades'] = {g: sum(1 for c in rows if c['grade'] == g) for g in reversed(GRADES)}
    s['takes_ok'] = sum(1 for c in rows if c['takes']['ok'])
    from collections import Counter
    s['kinds'] = dict(Counter(c['chosen'] for c in rows).most_common())
    return s


def why(c):
    """the properties that miss most (beyond TOL), worst first, in words"""
    s0, e, pT, p0 = c['seed0'], c['seed0']['err'], c['target']['props'], c['seed0']['props']
    out = []

    def add(sev, text):
        out.append((sev, text))
    if e.get('tail_oct') is not None and abs(e['tail_oct']) > TOL['tail_oct']:
        add(abs(e['tail_oct']) / TOL['tail_oct'],
            f"tail {'too long' if e['tail_oct'] > 0 else 'too short'} ({p0['tail_ms']:.0f} vs {pT['tail_ms']:.0f} ms)")
    tk = (c['takes'].get('err_mean') or {}).get('tail_oct')
    if tk is not None and abs(tk) > 1.0 and not c['catalog'].get('cut_by_next_event'):
        add(abs(tk) / 1.0, f"takes' tail {'rings on' if tk > 0 else 'short'} ({2 ** tk:.1f}x the target's)")
    if e.get('width_db') is not None and abs(e['width_db']) > TOL['width_db']:
        add(abs(e['width_db']) / TOL['width_db'],
            f"width: side-mid {p0['side_minus_mid_db']:.0f} vs {pT['side_minus_mid_db']:.0f} dB")
    if e.get('glide_st') is not None and abs(e['glide_st']) > TOL['glide_st']:
        add(abs(e['glide_st']) / TOL['glide_st'], f"glide {p0['glide_st']:+.1f} vs {pT['glide_st']:+.1f} st")
    if e.get('pulses') is not None and abs(e['pulses']) >= TOL['pulses']:
        add(abs(e['pulses']) / TOL['pulses'], f"pulses {p0['pulses']} vs {pT['pulses']}")
    if e.get('centroid_oct') is not None and abs(e['centroid_oct']) > TOL['centroid_oct']:
        add(abs(e['centroid_oct']) / TOL['centroid_oct'],
            f"{'brighter' if e['centroid_oct'] > 0 else 'darker'} ({p0['centroid_hz']:.0f} vs "
            f"{pT['centroid_hz']:.0f} Hz)")
    if e.get('layer_oct') is not None and abs(e['layer_oct']) > TOL['layer_oct']:
        add(abs(e['layer_oct']) / TOL['layer_oct'],
            f"transient/body balance off (head-body {p0['layer_oct']:+.1f} vs {pT['layer_oct']:+.1f} oct)")
    if e.get('attack_oct') is not None and abs(e['attack_oct']) > TOL['attack_oct']:
        add(abs(e['attack_oct']) / TOL['attack_oct'],
            f"attack {p0['attack_ms']:.0f} vs {pT['attack_ms']:.0f} ms")
    if s0.get('onset_err') is not None and s0['onset_err'] > TOL['onset_err']:
        add(s0['onset_err'] / TOL['onset_err'],
            f"onset sharpness {s0['sharp_db_ms'][1]:.0f} vs {s0['sharp_db_ms'][0]:.0f} dB/ms")
    if s0.get('env_r') is not None and s0['env_r'] < TOL['env_r']:
        add((1 - s0['env_r']) / (1 - TOL['env_r']), f"envelope r {s0['env_r']:.2f}")
    if s0.get('excess_db', 0) > TOL['excess_db']:
        add(s0['excess_db'] / TOL['excess_db'], f"excess energy {s0['excess_db']:.1f} dB (hiss / ring / floor)")
    if s0.get('missing_db', 0) > TOL['missing_db']:
        add(s0['missing_db'] / TOL['missing_db'], f"missing energy {s0['missing_db']:.1f} dB")
    if s0.get('mel_tail_gap') is not None and s0['mel_tail_gap'] > TOL['mel_tail_gap']:
        add(s0['mel_tail_gap'] / TOL['mel_tail_gap'], f"tail spectrogram {s0['mel_tail_gap']:.1f} dB over its floor")
    if e.get('tail_corr') is not None and abs(e['tail_corr']) > TOL['tail_corr']:
        add(abs(e['tail_corr']) / TOL['tail_corr'],
            f"tail space: L/R corr {p0['tail_corr']:.2f} vs {pT['tail_corr']:.2f}")
    if e.get('gate_db') is not None and abs(e['gate_db']) > TOL['gate_db']:
        add(abs(e['gate_db']) / TOL['gate_db'],
            f"{'harder' if e['gate_db'] > 0 else 'softer'} stop: steepest 5 ms fall "
            f"{p0['gate_db']:.0f} vs {pT['gate_db']:.0f} dB")
    if not c['takes']['ok']:
        add(0.9, 'takes: ' + '; '.join(c['takes']['fails'][:2]))
    out.sort(key=lambda a: -a[0])
    return [t for _, t in out[:4]] or ['no single property beyond tolerance (overall texture / CLAP)']


# target traits for the failure analysis: (name, test on the target's props / catalog, the property it stresses)
def traits(c):
    pT, cat = c['target']['props'], c['catalog']
    tail = pT.get('tail_ms') or 0
    return {
        'long_tail': tail >= 250,
        'noisy_tail': tail >= 150 and (pT.get('tail_flatness') or 0) >= 0.1,
        'noisy_body': (pT.get('tonal') if pT.get('tonal') is not None else 1.0) < 0.4,
        'gated': (pT.get('gate_db') or 0) >= 30,
        'wide': (pT.get('side_minus_mid_db') if pT.get('side_minus_mid_db') is not None else -40) >= -12,
        'glide': abs(pT.get('glide_st') or 0) >= 2,
        'multi_pulse': (pT.get('pulses') or 1) >= 3,
        'layered': (pT.get('layer_oct') or 0) >= 1.5,
        'reverb_space': tail >= 150 and pT.get('tail_corr') is not None and pT['tail_corr'] <= 0.7,
        'sub_heavy': (pT.get('lo150') or 0) >= 0.5,
        'bright': (pT.get('centroid_hz') or 0) >= 3000,
        'very_short': (c['len_s'] or 0) <= 0.15,
        'merged_run': bool(cat.get('merged_onsets')),
        'cut_by_next': bool(cat.get('cut_by_next_event')),
    }


TRAIT_PROP = {'long_tail': ('tail_oct', ('seed0', 'err', 'tail_oct')),
              'noisy_tail': ('mel_tail_gap', ('seed0', 'mel_tail_gap')),
              'noisy_body': ('excess_db', ('seed0', 'excess_db')),
              'gated': ('gate_db', ('seed0', 'err', 'gate_db')),
              'wide': ('width_db', ('seed0', 'err', 'width_db')),
              'glide': ('glide_st', ('seed0', 'err', 'glide_st')),
              'multi_pulse': ('pulses', ('seed0', 'err', 'pulses')),
              'layered': ('layer_oct', ('seed0', 'err', 'layer_oct')),
              'reverb_space': ('tail_corr', ('seed0', 'err', 'tail_corr')),
              'sub_heavy': ('lo150', ('seed0', 'err', 'lo150')),
              'bright': ('centroid_oct', ('seed0', 'err', 'centroid_oct')),
              'very_short': ('onset_err', ('seed0', 'onset_err')),
              'merged_run': ('pulses', ('seed0', 'err', 'pulses')),
              'cut_by_next': ('take_tail_oct', ('takes', 'err_mean', 'tail_oct'))}


def _fails(c, prop, path):
    v = _get(c, path)
    if v is None:
        return None
    tol = {'take_tail_oct': 1.0}.get(prop, TOL.get(prop))
    if prop == 'env_r':
        return v < tol
    return abs(v) > tol if prop != 'pulses' else abs(v) >= tol


def failure_analysis(rows):
    out = {'traits': {}, 'properties': {}, 'by_category_property_fail': {}, 'spearman_clap': {}}
    T = {c['id']: traits(c) for c in rows}
    for name in next(iter(T.values())):
        inn = [c for c in rows if T[c['id']][name]]
        oth = [c for c in rows if not T[c['id']][name]]
        prop, path = TRAIT_PROP[name]
        fi = [_fails(c, prop, path) for c in inn]
        fo = [_fails(c, prop, path) for c in oth]
        fi, fo = [a for a in fi if a is not None], [a for a in fo if a is not None]
        out['traits'][name] = {
            'n': len(inn), 'clap_med': r(_med([c['seed0']['clap'] for c in inn])),
            'clap_med_others': r(_med([c['seed0']['clap'] for c in oth])),
            'high_share': r(np.mean([c['grade'] == 'high' for c in inn]) if inn else None),
            'high_share_others': r(np.mean([c['grade'] == 'high' for c in oth]) if oth else None),
            'gap_med': r(_med([c['seed0']['mel_gap'] for c in inn])),
            'gap_med_others': r(_med([c['seed0']['mel_gap'] for c in oth])),
            'prop': prop, 'prop_abs_med': r(_med(_absv(inn, path))), 'prop_abs_med_others': r(_med(_absv(oth, path))),
            'prop_fail_share': r(np.mean(fi)) if fi else None, 'prop_fail_share_others': r(np.mean(fo)) if fo else None,
            'categories': sorted({c['category'] for c in inn}),
        }
    props_ = [('env_r', ('seed0', 'env_r')), ('onset_err', ('seed0', 'onset_err')),
              ('attack_oct', ('seed0', 'err', 'attack_oct')), ('centroid_oct', ('seed0', 'err', 'centroid_oct')),
              ('width_db', ('seed0', 'err', 'width_db')), ('glide_st', ('seed0', 'err', 'glide_st')),
              ('tail_oct', ('seed0', 'err', 'tail_oct')), ('pulses', ('seed0', 'err', 'pulses')),
              ('excess_db', ('seed0', 'excess_db')), ('missing_db', ('seed0', 'missing_db')),
              ('mel_tail_gap', ('seed0', 'mel_tail_gap')), ('layer_oct', ('seed0', 'err', 'layer_oct')),
              ('tail_corr', ('seed0', 'err', 'tail_corr')), ('lo150', ('seed0', 'err', 'lo150')),
              ('gate_db', ('seed0', 'err', 'gate_db')), ('take_tail_oct', ('takes', 'err_mean', 'tail_oct'))]
    for prop, path in props_:
        f = [(c, _fails(c, prop, path)) for c in rows]
        f = [(c, a) for c, a in f if a is not None]
        bad = [c for c, a in f if a]
        ok = [c for c, a in f if not a]
        out['properties'][prop] = {
            'tol': {'take_tail_oct': 1.0}.get(prop, TOL.get(prop)), 'n': len(f), 'fail': len(bad),
            'fail_share': r(len(bad) / len(f)) if f else None,
            'clap_med_fail': r(_med([c['seed0']['clap'] for c in bad])),
            'clap_med_ok': r(_med([c['seed0']['clap'] for c in ok])),
            'abs_med': r(_med(_absv(rows, path))), 'abs_p90': r(_q(_absv(rows, path), 90)),
            'signed_med_fail': r(_med([_get(c, path) for c in bad])),
        }
    from collections import defaultdict
    bycat = defaultdict(list)
    for c in rows:
        bycat[c['category']].append(c)
    for cat, cs in sorted(bycat.items()):
        d = {}
        for prop, path in props_:
            f = [a for a in (_fails(c, prop, path) for c in cs) if a is not None]
            if f:
                d[prop] = f'{sum(f)}/{len(f)}'
        out['by_category_property_fail'][cat] = d
    # why takes fail the take check: clips with at least one take failing for each reason
    import re
    kinds = defaultdict(set)
    for c in rows:
        for f in c['takes']['fails']:
            if 'hard artifacts' in f:
                for k in re.findall(r"'(\w+)'", f):
                    kinds[f'hard:{k}'].add(c['id'])
            elif ': audit ' in f:
                for k in f.split(': audit ', 1)[1].split(', '):
                    kinds[f'audit:{k}'].add(c['id'])
            elif 'CLAP to the fit' in f:
                kinds['clap_to_fit<0.90'].add(c['id'])
            elif 'CLAP to the target' in f:
                kinds['clap_drop>0.06'].add(c['id'])
            elif 'excess' in f:
                kinds['excess+1.5dB'].add(c['id'])
            elif 'loudness spread' in f:
                kinds['spread>2LU'].add(c['id'])
    out['take_fail_reasons'] = {k: len(v) for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1]))}
    out['takes_ok'] = int(sum(1 for c in rows if c['takes']['ok']))
    # which candidate kinds win, and what each kind scores when it wins / when it is a candidate
    by_kind = defaultdict(list)
    cand_clap = defaultdict(list)
    for c in rows:
        by_kind[c['chosen']].append(c['seed0']['clap'])
        for k in c['candidates']:
            if k.get('clap') is not None:
                cand_clap[k['candidate']].append(k['clap'])
    out['kinds'] = {k: {'won': len(v), 'clap_med_when_won': r(_med(v)),
                        'clap_med_as_candidate': r(_med(cand_clap.get(k, [])))} for k, v in
                    sorted(by_kind.items(), key=lambda kv: -len(kv[1]))}
    for k, v in cand_clap.items():
        out['kinds'].setdefault(k, {'won': 0, 'clap_med_when_won': None, 'clap_med_as_candidate': r(_med(v))})
    # the best candidate by CLAP alone vs the engine's choice
    gains = [max(k['clap'] for k in c['candidates'] if k.get('clap') is not None) - c['seed0']['clap']
             for c in rows if any(k.get('clap') is not None for k in c['candidates'])]
    out['oracle_clap_gain'] = {'median': r(_med(gains)), 'p90': r(_q(gains, 90)),
                               'n_over_0.02': int(sum(g > 0.02 for g in gains)), 'n': len(gains)}
    # what keeps a clip from 'high': CLAP alone, the mel gap alone, or both
    lim = {'clap_only': 0, 'gap_only': 0, 'both': 0}
    for c in rows:
        if c['grade'] == 'high':
            continue
        a_, b_ = (c['seed0']['clap'] or 0) < GRADE_BARS['high'][0], (c['seed0']['mel_gap'] or 0) > GRADE_BARS['high'][1]
        lim['both' if a_ and b_ else 'clap_only' if a_ else 'gap_only'] += 1
    out['high_limiter'] = lim
    # per winning kind: width on wide targets, onset direction, takes
    sd = lambda c, k: (c[k]['props'] if k == 'target' else c['seed0']['props']).get('side_minus_mid_db')  # noqa: E731
    pk = {}
    for kind in sorted(by_kind, key=lambda k: -len(by_kind[k])):
        cs = [c for c in rows if c['chosen'] == kind]
        wide = [c for c in cs if (sd(c, 'target') if sd(c, 'target') is not None else -40) >= -12]
        sg = [np.log2(c['seed0']['sharp_db_ms'][1] / c['seed0']['sharp_db_ms'][0]) for c in cs
              if c['seed0'].get('sharp_db_ms') and min(c['seed0']['sharp_db_ms']) > 0]
        pk[kind] = {
            'n': len(cs), 'wide_n': len(wide), 'wide_target_sd': r(_med([sd(c, 'target') for c in wide]), 1),
            'wide_render_sd': r(_med([sd(c, 'seed0') for c in wide]), 1),
            'wide_tail_corr_target': r(_med([c['target']['props'].get('tail_corr') for c in wide])),
            'wide_tail_corr_render': r(_med([c['seed0']['props'].get('tail_corr') for c in wide])),
            'onset_blunter': int(sum(x < -1 for x in sg)), 'onset_sharper': int(sum(x > 1 for x in sg)),
            'sharp_target': r(_med([c['seed0']['sharp_db_ms'][0] for c in cs]), 1),
            'sharp_render': r(_med([c['seed0']['sharp_db_ms'][1] for c in cs]), 1),
            'takes_ok': int(sum(c['takes']['ok'] for c in cs)),
            'take_clap_minus_fit': r(_med([c['takes']['clap_mean'] - c['seed0']['clap'] for c in cs])),
            'take_clap_to_fit': r(_med([c['takes']['clap_fit_mean'] for c in cs])),
            'pulses_target_multi': r(_med([c['target']['props']['pulses'] for c in cs
                                           if c['target']['props']['pulses'] >= 3]), 1),
            'pulses_render_multi': r(_med([c['seed0']['props']['pulses'] for c in cs
                                           if c['target']['props']['pulses'] >= 3]), 1)}
    out['per_kind'] = pk
    # by the clip's floor re its peak (a bed under the sound: music, a previous effect's tail) and by length
    fb = {}
    for name, lo, hi in (('floor <= -35 dB', -99, -35), ('floor -35..-25 dB', -35, -25), ('floor >= -25 dB', -25, 0)):
        cs = [c for c in rows if lo <= (c['target']['props'].get('tail_thr_db') or -40) - 3 < hi]
        fb[name] = {'n': len(cs), 'clap': r(_med([c['seed0']['clap'] for c in cs])),
                    'high': int(sum(c['grade'] == 'high' for c in cs)),
                    'takes_ok': int(sum(c['takes']['ok'] for c in cs))}
    for name, lo, hi in (('length < 0.25 s', 0, 0.25), ('length 0.25-0.5 s', 0.25, 0.5), ('length >= 0.5 s', 0.5, 99)):
        cs = [c for c in rows if lo <= c['len_s'] < hi]
        fb[name] = {'n': len(cs), 'clap': r(_med([c['seed0']['clap'] for c in cs])),
                    'high': int(sum(c['grade'] == 'high' for c in cs)),
                    'takes_ok': int(sum(c['takes']['ok'] for c in cs))}
    out['buckets'] = fb
    # rank correlation of target traits with CLAP
    try:
        from scipy.stats import spearmanr
        cl = np.array([c['seed0']['clap'] for c in rows], float)
        for name, key in (('tail_ms', 'tail_ms'), ('len_ms', 'dur_ms'), ('side_minus_mid_db', 'side_minus_mid_db'),
                          ('abs_glide_st', 'glide_st'), ('pulses', 'pulses'), ('layer_oct', 'layer_oct'),
                          ('tail_flatness', 'tail_flatness'), ('tail_corr', 'tail_corr'),
                          ('centroid_hz', 'centroid_hz'),
                          ('lo150', 'lo150'), ('tonal', 'tonal'), ('attack_ms', 'attack_ms'),
                          ('gate_db', 'gate_db')):
            v = np.array([np.nan if c['target']['props'].get(key) is None else
                          (abs(c['target']['props'][key]) if name.startswith('abs_') else c['target']['props'][key])
                          for c in rows], float)
            ok = np.isfinite(v)
            if ok.sum() >= 8:
                rho, p = spearmanr(v[ok], cl[ok])
                out['spearman_clap'][name] = {'rho': r(rho), 'p': r(p, 4), 'n': int(ok.sum())}
    except Exception as ex:
        out['spearman_clap'] = {'error': str(ex)}
    return out


def _f(v, fmt='{:.2f}'):
    return '-' if v is None else fmt.format(v)


def write_report(out_dir, targets, split, rows, errors_, meta):
    from collections import defaultdict
    by_id = {c['id']: c for c in rows}
    dev = [by_id[i] for i in split['dev'] if i in by_id]
    held = [by_id[i] for i in split['heldout'] if i in by_id]
    bycat = defaultdict(list)
    for c in rows:
        bycat[c['category']].append(c)
    cats = {cat: summarize(cs) for cat, cs in sorted(bycat.items())}
    worst = sorted(rows, key=lambda c: (GRADES.index(c['grade']), c['seed0']['clap'] - 0.02 * max(
        0.0, c['seed0']['mel_gap'] or 0)))[:20]
    fa = failure_analysis(rows) if rows else {}
    J = {'meta': meta, 'split': {k: split[k] for k in ('method', 'n_dev', 'n_heldout', 'strata')},
         'grades': {g: list(GRADE_BARS.get(g, ())) for g in GRADES}, 'tolerances': TOL,
         'summary': {'all': summarize(rows), 'dev': summarize(dev), 'heldout': summarize(held)},
         'categories': cats, 'worst20': [{'id': c['id'], 'grade': c['grade'], 'clap': c['seed0']['clap'],
                                          'mel_gap': c['seed0']['mel_gap'], 'chosen': c['chosen'], 'why': why(c)}
                                         for c in worst],
         'failure_analysis': fa, 'errors': errors_,
         'clips': [dict(c, split='heldout' if c['id'] in set(split['heldout']) else 'dev') for c in rows]}
    jdump(f'{out_dir}/baseline_fit.json', J)

    L = []
    A = L.append
    S = J['summary']
    A('# Replication benchmark: the fit baseline (earshot 0.4 work, v0.4 worktree)')
    A('')
    sg = meta.get('script_git') or ''
    A(f"Generated {meta['made']} by `eval/forge/replicate.py` (earshot {meta['earshot']}, fit engine v"
      f"{meta['fit_version']}, engine git {(meta.get('git') or '')[:10] or 'unknown'}"
      + (f", script git {sg[:10]}" if sg and sg != meta.get('git') else '')
      + f"), {len(rows)} of {len(targets)} targets fitted"
      + (f", {len(errors_)} failed" if errors_ else '') + '. Local study numbers: the targets are licensed clips '
      'from third-party films and stay outside the repo; so do these renders.')
    A('')
    A('Per target: `auto.fit(clap=True)` (the `earshot fit --clap` choice), seed 0 measured against the target, the '
      "model's own floor (seed 11 vs 12), four takes (seeds 1-4, vary 0.3, take mode). Grades: **high** CLAP >= 0.92 "
      'and mel gap <= 2.5 dB; **good** >= 0.88 / <= 3.5; **fair** >= 0.80 / <= 5.0; **poor** the rest. Property '
      'errors are render minus target (|median| in the tables); definitions in the script docstring.')
    A('')
    A('## Summary')
    A('')
    A('| set | n | CLAP | mel gap dB | env r | onset err oct | high / good / fair / poor | takes ok | kinds won |')
    A('|---|---|---|---|---|---|---|---|---|')
    for k in ('all', 'dev', 'heldout'):
        s = S[k]
        if not s.get('n'):
            continue
        g = s['grades']
        A(f"| {k} | {s['n']} | {_f(s['clap'], '{:.3f}')} | {_f(s['mel_gap'])} | {_f(s['env_r'], '{:.3f}')} | "
          f"{_f(s['onset_err'])} | {g['high']} / {g['good']} / {g['fair']} / {g['poor']} | {s['takes_ok']} | "
          + ', '.join(f'{a} {b}' for a, b in s['kinds'].items()) + ' |')
    A('')
    A('## Per category (medians)')
    A('')
    A('| category | n | CLAP | mel gap | mel head | tail mel gap | env r | onset err | |attack| oct | |centroid| oct | '
      '|width| dB | |glide| st | |tail| oct | |pulses| | excess dB | take CLAP | |take tail| oct | grades h/g/f/p | '
      'takes ok | kinds |')
    A('|' + '---|' * 20)
    for cat, s in cats.items():
        g = s['grades']
        A(f"| {cat} | {s['n']} | {_f(s['clap'], '{:.3f}')} | {_f(s['mel_gap'])} | {_f(s['mel_head'])} | "
          f"{_f(s['mel_tail_gap'])} | {_f(s['env_r'], '{:.3f}')} | {_f(s['onset_err'])} | {_f(s['attack_oct'])} | "
          f"{_f(s['centroid_oct'])} | {_f(s['width_db'], '{:.1f}')} | {_f(s['glide_st'], '{:.1f}')} | "
          f"{_f(s['tail_oct'])} | {_f(s['pulses'], '{:.0f}')} | {_f(s['excess_db'])} | "
          f"{_f(s['take_clap'], '{:.3f}')} | {_f(s['take_tail_oct'])} | {g['high']}/{g['good']}/{g['fair']}/"
          f"{g['poor']} | {s['takes_ok']}/{s['n']} | " + ', '.join(f'{a} {b}' for a, b in s['kinds'].items()) + ' |')
    A('')
    A('## The worst 20 (lowest grade, then lowest CLAP)')
    A('')
    A('| # | clip | split | grade | CLAP | mel gap | won | why |')
    A('|---|---|---|---|---|---|---|---|')
    hs = set(split['heldout'])
    for i, w in enumerate(J['worst20'], 1):
        A(f"| {i} | `{w['id']}` | {'held' if w['id'] in hs else 'dev'} | {w['grade']} | {_f(w['clap'], '{:.3f}')} | "
          f"{_f(w['mel_gap'])} | {w['chosen']} | " + '; '.join(w['why']) + ' |')
    if fa:
        A('')
        A('## Failure analysis (numbers)')
        A('')
        A('**Properties** (seed 0 vs the target; fail = beyond the tolerance):')
        A('')
        A('| property | tolerance | fail / n | share | CLAP med when failing / ok | |err| median / p90 | signed med '
          'when failing |')
        A('|---|---|---|---|---|---|---|')
        for p, d in sorted(fa['properties'].items(), key=lambda kv: -(kv[1]['fail_share'] or 0)):
            A(f"| {p} | {d['tol']} | {d['fail']}/{d['n']} | {_f(d['fail_share'])} | {_f(d['clap_med_fail'], '{:.3f}')}"
              f" / {_f(d['clap_med_ok'], '{:.3f}')} | {_f(d['abs_med'])} / {_f(d['abs_p90'])} | "
              f"{_f(d['signed_med_fail'])} |")
        A('')
        A('**Target traits** (clips with the trait vs the rest; the property it stresses):')
        A('')
        A('| trait | n | CLAP med (others) | high share (others) | mel gap med (others) | property | |err| med '
          '(others) | fail share (others) | categories |')
        A('|---|---|---|---|---|---|---|---|---|')
        for name, d in fa['traits'].items():
            A(f"| {name} | {d['n']} | {_f(d['clap_med'], '{:.3f}')} ({_f(d['clap_med_others'], '{:.3f}')}) | "
              f"{_f(d['high_share'])} ({_f(d['high_share_others'])}) | {_f(d['gap_med'])} ({_f(d['gap_med_others'])})"
              f" | {d['prop']} | {_f(d['prop_abs_med'])} ({_f(d['prop_abs_med_others'])}) | "
              f"{_f(d['prop_fail_share'])} ({_f(d['prop_fail_share_others'])}) | {', '.join(d['categories'])} |")
        A('')
        A('**Why takes fail the take check** (clips with >= 1 take failing for the reason; '
          f"{fa['takes_ok']} of {len(rows)} clips pass):")
        A('')
        A('| reason | clips |')
        A('|---|---|')
        for k, v in fa['take_fail_reasons'].items():
            A(f'| {k} | {v} |')
        A('')
        og = fa['oracle_clap_gain']
        A(f"**Model kinds** (the best candidate by CLAP alone would gain a median {_f(og['median'], '{:.3f}')} CLAP, "
          f"p90 {_f(og['p90'], '{:.3f}')}, > 0.02 on {og['n_over_0.02']} of {og['n']}):")
        A('')
        A('| kind | won | CLAP med when won | CLAP med as a candidate |')
        A('|---|---|---|---|')
        for k, d in fa['kinds'].items():
            A(f"| {k} | {d['won']} | {_f(d['clap_med_when_won'], '{:.3f}')} | "
              f"{_f(d['clap_med_as_candidate'], '{:.3f}')} |")
        A('')
        hl = fa['high_limiter']
        A(f"**What keeps a clip from high**: CLAP alone {hl['clap_only']}, the mel gap alone {hl['gap_only']}, both "
          f"{hl['both']}.")
        A('')
        A('**Per winning kind** (wide = target side-minus-mid >= -12 dB; onset = sharpness off by more than an octave; '
          'pulses on targets with >= 3):')
        A('')
        A('| kind | n | wide n | wide side-mid target / render dB | wide tail corr target / render | onset blunter / '
          'sharper | sharpness target / render dB/ms | pulses target / render | takes ok | take CLAP - fit | take '
          'CLAP to fit |')
        A('|---|---|---|---|---|---|---|---|---|---|---|')
        for k, d in fa['per_kind'].items():
            A(f"| {k} | {d['n']} | {d['wide_n']} | {_f(d['wide_target_sd'], '{:.1f}')} / "
              f"{_f(d['wide_render_sd'], '{:.1f}')} | {_f(d['wide_tail_corr_target'])} / "
              f"{_f(d['wide_tail_corr_render'])} | {d['onset_blunter']} / {d['onset_sharper']} | "
              f"{_f(d['sharp_target'], '{:.1f}')} / {_f(d['sharp_render'], '{:.1f}')} | "
              f"{_f(d['pulses_target_multi'], '{:.0f}')} / {_f(d['pulses_render_multi'], '{:.0f}')} | "
              f"{d['takes_ok']}/{d['n']} | {_f(d['take_clap_minus_fit'], '{:+.3f}')} | "
              f"{_f(d['take_clap_to_fit'], '{:.3f}')} |")
        A('')
        A("**By the clip's floor re its peak** (the bed under the sound: music or a previous effect's tail) **and by "
          'length:**')
        A('')
        A('| bucket | n | CLAP | high | takes ok |')
        A('|---|---|---|---|---|')
        for k, d in fa['buckets'].items():
            A(f"| {k} | {d['n']} | {_f(d['clap'], '{:.3f}')} | {d['high']} | {d['takes_ok']} |")
        A('')
        A('**Spearman rank correlation of target traits with CLAP** (negative = the trait makes the fit worse):')
        A('')
        A('| trait | rho | p | n |')
        A('|---|---|---|---|')
        for name, d in (fa.get('spearman_clap') or {}).items():
            if isinstance(d, dict):
                A(f"| {name} | {_f(d['rho'])} | {_f(d['p'], '{:.4f}')} | {d['n']} |")
        A('')
        A('**Property fails per category** (fail / measured):')
        A('')
        P_ = list(fa['properties'])
        A('| category | ' + ' | '.join(P_) + ' |')
        A('|---|' + '---|' * len(P_))
        for cat, d in fa['by_category_property_fail'].items():
            A(f'| {cat} | ' + ' | '.join(d.get(p, '-') for p in P_) + ' |')
    if errors_:
        A('')
        A('## Failed fits')
        A('')
        for e in errors_:
            A(f"- `{e['id']}`: {e['error']}")
    A('')
    A('## Split')
    A('')
    A(f"{split['method']}. dev {split['n_dev']}, held-out {split['n_heldout']}: `split.json`.")
    A('')
    A('| stratum | n | held out |')
    A('|---|---|---|')
    for s, d in split['strata'].items():
        A(f"| {s} | {d['n']} | {d['held']} |")
    A('')
    A('## Reproduce')
    A('')
    A('```')
    A('cd <v0.4 worktree>; EARSHOT_THREADS=3 <venv python> eval/forge/replicate.py --targets <sfx dir> --out <out dir>')
    A('<venv python> eval/forge/replicate.py --targets <sfx dir> --out <out dir> --report   # tables from the cache')
    A('```')
    A('')
    A('Renders: `baseline_renders/<category>/<stem>/` (`audition.wav` = target | fit seed 0 | takes 1-4, 0.35 s gaps; '
      '`pair.png`; `model.json`).')
    an = f'{out_dir}/analysis.md'
    if os.path.exists(an):
        with open(an, encoding='utf-8') as fh:
            L += ['', fh.read().rstrip()]
    with open(f'{out_dir}/BASELINE.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


# ============================================================================================= main
def _git_head(path=REPO):
    """the HEAD commit of the checkout at `path` ('' when it is none)"""
    try:
        import subprocess
        return subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ''


def _in_repo(path, repo):
    """is `path` inside the checkout `repo` and not under its gitignored out/ (where target-derived files may not go)"""
    p, r = (os.path.abspath(v).replace(chr(92), '/').lower().rstrip('/') + '/' for v in (path, repo))
    return p.startswith(r) and not p.startswith(r + 'out/')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--targets', default=os.environ.get('EARSHOT_REPLICATE_TARGETS'),
                    help='target folder (<category>/*.wav [+ catalog.json]); or EARSHOT_REPLICATE_TARGETS')
    ap.add_argument('--out', default=os.environ.get('EARSHOT_REPLICATE_OUT'),
                    help='output folder (outside the repo, or under its gitignored out/); or EARSHOT_REPLICATE_OUT')
    ap.add_argument('--renders', help='render folder (default <out>/baseline_renders)')
    ap.add_argument('--only', help='comma-separated categories')
    ap.add_argument('--clips', help='comma-separated substrings of clip ids')
    ap.add_argument('--split', choices=('all', 'dev', 'heldout'), default='all', help='which split to fit')
    ap.add_argument('--split-file', help="a frozen split (JSON {'heldout': [ids], 'dev': [ids], 'clips': [catalog "
                                         "records]}: read_split) in place of the sha1 split; its 'clips', when "
                                         'present, are the targets')
    ap.add_argument('--limit', type=int, help='at most this many targets (after filters)')
    ap.add_argument('--force', action='store_true', help='refit clips that already have a result')
    ap.add_argument('--report', action='store_true', help='only rebuild baseline_fit.json / BASELINE.md from the cache')
    ap.add_argument('--remeasure', action='store_true',
                    help='recompute the property measures of cached clips from their saved renders (no fit), '
                         'then report')
    a = ap.parse_args(argv)
    if not a.targets or not a.out:
        ap.error('--targets and --out (or EARSHOT_REPLICATE_TARGETS / EARSHOT_REPLICATE_OUT) are required')
    out_dir = os.path.abspath(a.out).replace(chr(92), '/')
    rend = os.path.abspath(a.renders or f'{out_dir}/baseline_renders').replace(chr(92), '/')
    for what, d in (('output', out_dir), ('render', rend)):   # the renders hold the targets themselves (audition.wav),
        if _in_repo(d, REPO):                                  # their spectrograms and fits
            ap.error(f'refusing to write target-derived files inside the repo ({what} folder {d}); use a folder '
                     'outside it or under out/')
    records = None
    if a.split_file:
        with open(a.split_file, encoding='utf-8') as fh:
            records = json.load(fh).get('clips')
    targets = find_targets(a.targets, records)
    if not targets:
        ap.error(f'no targets under {a.targets}')
    split = read_split(a.split_file, targets) if a.split_file else make_split(targets)
    os.makedirs(out_dir, exist_ok=True)
    jdump(f'{out_dir}/split.json', split)
    sel = targets
    if a.only:
        cs = set(a.only.split(','))
        sel = [t for t in sel if t['category'] in cs]
    if a.clips:
        subs = a.clips.split(',')
        sel = [t for t in sel if any(s in t['id'] for s in subs)]
    if a.split != 'all':
        keep = set(split['dev' if a.split == 'dev' else 'heldout'])
        sel = [t for t in sel if t['id'] in keep]
    if a.limit:
        sel = sel[:a.limit]
    import earshot
    from earshot.forge.fit import auto
    engine = os.path.dirname(os.path.dirname(os.path.abspath(earshot.__file__)))
    meta = {'made': time.strftime('%Y-%m-%d %H:%M'), 'earshot': getattr(earshot, '__version__', None),
            'earshot_path': os.path.dirname(earshot.__file__).replace(chr(92), '/'), 'fit_version': auto.VERSION,
            'git': _git_head(engine) if os.path.exists(os.path.join(engine, '.git')) else '',   # the engine's
                                                       # checkout (a PYTHONPATH may name another one; none: installed)
            'script_git': _git_head(), 'targets_dir': a.targets.replace(chr(92), '/'), 'n_targets': len(targets),
            'vary': VARY, 'take_seeds': list(TAKE_SEEDS), 'floor_seeds': list(FLOOR_SEEDS),
            'threads': os.environ.get('EARSHOT_THREADS')}
    log(f'earshot {meta["earshot"]} from {meta["earshot_path"]}; {len(targets)} targets, {len(sel)} selected; '
        f'split dev {split["n_dev"]} / held-out {split["n_heldout"]}')
    errors_ = []
    if a.remeasure:
        for t in sel:
            cat, stem = t['id'].split('/', 1)
            cpath = f'{out_dir}/clips/{cat}/{stem}.json'
            if os.path.exists(cpath):
                with open(cpath, encoding='utf-8') as fh:
                    c = json.load(fh)
                jdump(cpath, remeasure(t, c, f'{rend}/{cat}/{stem}'))
        log(f'remeasured {len(sel)} clips from their renders')
    elif not a.report:
        t_all = time.time()
        for i, t in enumerate(sel, 1):
            cat, stem = t['id'].split('/', 1)
            cpath = f'{out_dir}/clips/{cat}/{stem}.json'
            if os.path.exists(cpath) and not a.force:
                continue
            try:
                res = run_target(t, f'{rend}/{cat}/{stem}')
                jdump(cpath, res)
                s0 = res['seed0']
                log(f"[{i}/{len(sel)}] {t['id']}: {res['chosen']} CLAP {s0['clap']:.3f} mel {s0['mel_db']:.2f} "
                    f"(floor {s0['floor_mel_db']:.2f}) env {s0['env_r']:.3f} -> {res['grade']}; takes CLAP "
                    f"{res['takes']['clap_mean']:.3f} {'ok' if res['takes']['ok'] else 'FAIL'} "
                    f"({res['seconds']['total']} s)")
            except Exception as ex:
                log(f"[{i}/{len(sel)}] {t['id']}: FAILED {type(ex).__name__}: {ex}")
                traceback.print_exc()
                jdump(f'{out_dir}/clips/{cat}/{stem}.error.json',
                      {'id': t['id'], 'error': f'{type(ex).__name__}: {ex}', 'trace': traceback.format_exc()})
        log(f'fitting done in {(time.time() - t_all) / 60:.1f} min')
    rows = []
    for t in targets:
        cat, stem = t['id'].split('/', 1)
        p = f'{out_dir}/clips/{cat}/{stem}.json'
        if os.path.exists(p):
            with open(p, encoding='utf-8') as fh:
                rows.append(json.load(fh))
        elif os.path.exists(p.replace('.json', '.error.json')):
            with open(p.replace('.json', '.error.json'), encoding='utf-8') as fh:
                e = json.load(fh)
            errors_.append({'id': e['id'], 'error': e['error']})
    if rows:
        write_report(out_dir, targets, split, rows, errors_, meta)
        log(f"wrote {out_dir}/baseline_fit.json and BASELINE.md ({len(rows)} clips)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
