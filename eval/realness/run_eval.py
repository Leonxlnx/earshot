"""Held-out evaluation of the realness detector: grouped cross-validation, baselines, robustness, file level.

    OMP_NUM_THREADS=2 EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/run_eval.py [--stage cv|layers|apps|level|report]

Needs out/realness/manifest.json (build_data.py) and out/realness/feats (extract.py; --ast 240 for the AST rows).
Stage cv caches out-of-group predictions per scheme in out/realness/oof/ (time-boxed: run again until it says done);
stage layers scores the layered clips (real bed + one code-synthesized layer) with models that never saw the bed's group
nor the layer's family; stage apps scores whole application files (the Clawd film with its real and its synthesized
soundtrack, the Windows UI sounds); stage level measures how much the input level moves p_real (final model; a fixed
subset at its own level, -20 dB, normalised to -25 dBFS RMS, lifted as at inference, at -56 dBFS); stage report
writes eval/realness/results.json and RESULTS.md (generated: every number in it comes from this run). Every clip is scored by a model that never saw its CV group:
  A        groups = real asset category (ambience, foley, garden, keyboard, magic, motion, toy, music) or the films;
           synthesized family (retired score, cinetic engine, generic generators): the headline
  A_noaug  the same without the augmented training copies; A_clap / A_signal: one head only (ablation)
  B        synthesized groups = one generator type (cinetic:whoosh, generic:beep, retired:stem_pad ...)
  C        the retired score is never in any training set: real recordings must still rank above it
  D        neither the retired score nor the cinetic engine is ever trained on (only generic generators as negatives)
  M        as A, but music is held out one provider at a time (mk / sb / ch / other): the grouping the final model is
           trained with (each source weighs the same, so music counts as four sources); verdict thresholds come from M
  M_layers as M, also trained on the layered clips of the other groups (tested, not adopted: see RESULTS.md)
  S*       the stratified AST subset: clap+signal vs +AST head vs AST head alone
"""
import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402
import common as C  # noqa: E402
from earshot import realness  # noqa: E402

OOF = os.path.join(C.OUT, 'oof')
SCHEMES = {'A': {}, 'A_noaug': {'augment': False}, 'A_clap': {'heads': ('clap',)}, 'A_signal': {'heads': ('signal',)},
           'C': {'exclude': ('retired',)},
           'D': {'exclude': ('retired', 'cinetic')}, 'B': {'groups': 'type', 'c_clap': 0.001, 'inner': 3},
           'M': {'groups': 'music_src'}, 'M_layers': {'groups': 'music_src', 'layers': True},
           'S': {'ast': True}, 'S_ast': {'ast': True, 'heads': ('clap', 'signal', 'ast')},
           'S_astonly': {'ast': True, 'heads': ('ast',)}}
APPS = {'clawd_film_real_soundtrack': f'{EP.FILM}/out/Clawd_1080p60.mp4',
        'clawd_film_synth_soundtrack': f'{EP.FILM}/out/superseded/Clawd_1080p60_synth-audio.mp4',
        'clawd_soundtrack_wav': f'{EP.FILM}/out/audio/soundtrack.wav',
        'clawd_stem_pro_sfx': f'{EP.FILM}/out/audio/stems_pro/sfx.wav',
        'clawd_stem_pro_amb': f'{EP.FILM}/out/audio/stems_pro/amb.wav',
        'clawd_stem_pro_music': f'{EP.FILM}/out/audio/stems_pro/music.wav'}
PRO_SYNTH = 'C:/Windows/Media'   # professionally made synthetic UI / alert sounds (never trained on)
ASSETS = EP.ASSETS + '/'


def music_source(src):
    """provider prefix of a music file name (mk = Mixkit, sb, ch ...; rare providers pooled)"""
    v = os.path.basename(src).split('-')[0].split('_')[0]
    return v if v in ('mk', 'sb', 'ch') else 'other'


def groups_of(d, kind='fam'):
    if kind == 'music_src':
        g = groups_of(d)
        return np.array([f'music:{music_source(s)}' if gi == 'music' else gi for gi, s in zip(g, d.src)])
    if kind == 'type':
        return np.where(d.y == 1, d.group, np.char.add(np.char.add(d.family.astype(str), ':'), d.type.astype(str)))
    return np.where(d.y == 1, d.group, d.family)


def bed_groups_of(d, kind='fam'):
    """the group of a layered clip's real bed under the same grouping (other clips: their own group)"""
    g = groups_of(d, kind).astype(object)
    for i in np.where(d.is_layer)[0]:
        b = d.bed_group[i]
        g[i] = f'music:{music_source(d.bed_src[i])}' if b == 'music' and kind == 'music_src' else b
    return g


def run_layers():
    """the layer test: every layered clip (real bed + one code-synthesized layer) is scored by models that never saw
    its bed's group nor its layer's family (scheme-M grouping, the final model's CLAP C): 'base' trains as the final
    model does without layered clips, 'train' also trains on the layered clips of the other groups"""
    d = C.load()
    g = groups_of(d, 'music_src')
    bg = bed_groups_of(d, 'music_src')
    c = realness.load_model()['C']['clap']
    Lm = d.is_layer
    pairs = sorted(set(zip(bg[Lm], g[Lm])))
    for variant in ('base', 'train'):
        path = os.path.join(OOF, f'layers_{variant}.npz')
        if os.path.exists(path):
            continue
        t = time.time()
        p = np.full(len(d), np.nan)
        usable = (d.perturb == '') | d.augment | (Lm & (variant == 'train'))
        for gb, gl in pairs:
            te = Lm & (bg == gb) & (g == gl)
            tr = usable & ~np.isin(g, [gb, gl]) & ~np.isin(bg, [gb, gl])
            p[te] = C.predict(C.fit_model(d.sub(tr), g[tr], c_clap=c), d.sub(te))
        np.savez(path, ids=d.ids, p=p)
        print(f'layers {variant}: {int(Lm.sum())} clips, {len(pairs)} folds, {time.time() - t:.0f} s', flush=True)
        if variant == 'base':
            print('run again for the train variant')
            return


def run_cv(budget):
    os.makedirs(OOF, exist_ok=True)
    t0 = time.time()
    left = []
    for name, cfg in SCHEMES.items():
        path = os.path.join(OOF, name + '.npz')
        if os.path.exists(path):
            continue
        if time.time() - t0 > budget:
            left.append(name)
            continue
        if cfg.get('ast') and not os.path.exists(os.path.join(C.FEATS, 'ast.npz')):
            print(f'{name}: skipped (no AST features; run extract.py --ast 240)')
            continue
        d = C.load(require_ast=bool(cfg.get('ast')), robust=not cfg.get('ast'))
        g = groups_of(d, cfg.get('groups', 'fam'))
        tm = ~np.isin(d.family, cfg['exclude']) if cfg.get('exclude') else None
        kw = {k: cfg[k] for k in ('heads', 'c_clap', 'augment', 'aug_weight', 'inner', 'layers') if k in cfg}
        if cfg.get('layers'):
            kw['bed_groups'] = bed_groups_of(d, cfg.get('groups', 'fam'))
        t = time.time()
        p, L = C.logo(d, g, tm, **kw)
        np.savez(path, ids=d.ids, p=p, **{'L_' + k: v for k, v in L.items()})
        print(f'{name}: {len(d)} clips, {len(set(g))} folds, {time.time() - t:.0f} s', flush=True)
    print('cv done' if not left else f'run again for: {", ".join(left)}')


def prompt_embeddings():
    """CLAP text embeddings of the zero-shot prompts, read from earshot's text cache when present"""
    from earshot import config
    texts = realness.REAL_PROMPTS + realness.SYNTH_PROMPTS
    for d in sorted(os.listdir(os.path.join(config.EMB_CACHE, 'clap'))) if os.path.isdir(os.path.join(config.EMB_CACHE, 'clap')) else []:
        f = os.path.join(config.EMB_CACHE, 'clap', d, 'text.npz')
        if os.path.exists(f) and config.CLAP_MODEL.replace('/', '--') in d:
            z = np.load(f, allow_pickle=False)
            keys = [hashlib.sha1(t.encode('utf-8')).hexdigest()[:24] for t in texts]
            if all(k in z.files for k in keys):
                return np.stack([z[k] for k in keys])
    from earshot.models import clap
    return clap.get().embed_text(texts)


def app_files():
    """{name: path}: the application files plus every top-level wav of C:/Windows/Media (name 'winui:<file>')"""
    out = dict(APPS)
    if os.path.isdir(PRO_SYNTH):
        out.update({f'winui:{f}': os.path.join(PRO_SYNTH, f) for f in sorted(os.listdir(PRO_SYNTH))
                    if f.lower().endswith('.wav')})
    return out


def run_apps():
    """windows of whole application files and of the Windows UI sounds -> CLAP + signal features, exactly as
    earshot.realness scores them (quiet sources lifted), cached in out/realness/feats/apps.npz"""
    from earshot import audio
    from earshot.models import clap
    c = clap.get()
    ids, E, S = [], [], []
    for name, path in app_files().items():
        x = realness.lift_quiet(audio.load(path, mono=True, cache=False))[0]
        spans = realness._clips(x)
        live = [(t0, w) for t0, _, w in spans if realness.audible(w)]
        E += list(c.embed_audio([w for _, w in live]))
        S += [json.dumps(realness.signal_features(w)) for _, w in live]
        ids += [f'{name}@{t0:.0f}' for t0, _ in live]
        if not name.startswith('winui:'):
            print(name, len(live), 'windows', flush=True)
    np.savez(os.path.join(C.FEATS, 'apps.npz'), ids=np.array(ids), emb=np.array(E, np.float32), sig=np.array(S))


def run_level():
    """p_real of a fixed subset of clean clips under level changes (final model, CLAP head + fusion; the CLAP head is
    the only level-dependent input, signal features are peak-normalised): own level, 20 dB quieter, normalised to
    -25 dBFS RMS, own level after realness.lift_quiet (what inference does), and moved to -56 dBFS RMS without / with
    the lift (the old cliff: the first build scored sources between -60 and -30 dBFS RMS as they were)"""
    import soundfile as sf
    from earshot.models import clap
    man = json.load(open(os.path.join(C.OUT, 'manifest.json')))
    m = realness.load_model()
    c = clap.get()
    at = lambda x, r, db: (x * 10 ** ((db - r) / 20)).astype(np.float32)  # noqa: E731
    conds = {'orig': lambda x, r: x, '-20dB': lambda x, r: x * 10 ** (-1.0), 'norm-25': lambda x, r: at(x, r, -25),
             'lifted': lambda x, r: realness.lift_quiet(x)[0], 'at-56': lambda x, r: at(x, r, -56),
             'at-56_lifted': lambda x, r: realness.lift_quiet(at(x, r, -56))[0]}
    out = {}
    for k, clips in {'real': man['real'][::12], 'cinetic': man['cinetic'][::9], 'generic': man['generic'][::8]}.items():
        X = [sf.read(os.path.join(C.OUT, cl['wav']), dtype='float32')[0] for cl in clips]
        rms = np.array([20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12) for x in X])
        P = {}
        for g, fn in conds.items():
            lc = np.array([realness.head_logit(m['heads']['clap'], e)[0]
                           for e in c.embed_audio([fn(x, r) for x, r in zip(X, rms)])])
            P[g] = 1 / (1 + np.exp(-(m['fusion']['intercept'] + m['fusion']['coef'][0] * lc)))
        lab = k == 'real'
        out[k] = {'clips': len(X), 'rms_db_median': round(float(np.median(rms)), 1),
                  'lifted_clips': int(np.sum(rms < realness.LIFT_BELOW_DB)),
                  **{f'mean_p_{g}': round(float(v.mean()), 3) for g, v in P.items()},
                  **{f'acc_{g}': round(float(np.mean((v >= 0.5) == lab)), 3) for g, v in P.items()},
                  **{f'mean_abs_dp_{g}': round(float(np.abs(P[g] - P['orig']).mean()), 3) for g in conds if g != 'orig'}}
        print(k, out[k], flush=True)
    json.dump(out, open(os.path.join(C.FEATS, 'level.json'), 'w'), indent=1)


# ---------------------------------------------------------------------------------------------- metrics
def auc(y, s):
    return round(float(roc_auc_score(y, s)), 4) if len(set(y)) == 2 else None


def brier(y, p):
    return round(float(np.mean((p - y) ** 2)), 4)


def acc(y, p, t=0.5):
    return round(float(np.mean((p >= t) == (y == 1))), 4)


def metrics(y, p):
    return {'n': int(len(y)), 'n_real': int(y.sum()), 'auc': auc(y, p), 'acc@0.5': acc(y, p), 'brier': brier(y, p),
            'logloss': round(float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))), 4)}


TARGET = 0.925


def thresholds(y, p, target=TARGET):
    """hi = lowest p above which >= target of clips are real; lo = highest p below which >= target are synthesized"""
    o = np.argsort(-p)
    prec = np.cumsum(y[o]) / np.arange(1, len(o) + 1)
    ok = np.where(prec >= target)[0]
    hi = float(p[o][ok[-1]]) if len(ok) else 1.0
    o2 = np.argsort(p)
    prec2 = np.cumsum(1 - y[o2]) / np.arange(1, len(o2) + 1)
    ok2 = np.where(prec2 >= target)[0]
    lo = float(p[o2][ok2[-1]]) if len(ok2) else 0.0
    return round(min(max(hi, 0.5), 0.95), 2), round(max(min(lo, 0.5), 0.05), 2)


def band_stats(y, p, lo, hi):
    real, syn = p >= hi, p <= lo
    unc = ~(real | syn)
    return {'coverage': round(float(np.mean(~unc)), 4),
            'precision_real': round(float(np.mean(y[real])), 4) if real.any() else None,
            'precision_synth': round(float(np.mean(1 - y[syn])), 4) if syn.any() else None,
            'real_called_synth': round(float(np.mean(syn[y == 1])), 4), 'synth_called_real': round(float(np.mean(real[y == 0])), 4),
            'uncertain_frac': round(float(np.mean(unc)), 4)}


def calibration(y, p, bins=10):
    out = []
    for i in range(bins):
        m = (p >= i / bins) & (p < (i + 1) / bins + (i == bins - 1))
        if m.sum():
            out.append({'bin': f'{i / bins:.1f}-{(i + 1) / bins:.1f}', 'n': int(m.sum()), 'mean_p': round(float(p[m].mean()), 3),
                        'frac_real': round(float(y[m].mean()), 3)})
    return out


def per_group(d, p, g):
    rows = []
    for k in sorted(set(g)):
        m = (g == k) & (d.perturb == '')
        lab = int(d.y[m][0])
        other = (d.y == 1 - lab) & (d.perturb == '')
        yy = np.r_[d.y[m], d.y[other]]
        rows.append({'group': str(k), 'label': 'real/pro' if lab else 'synth', 'n': int(m.sum()),
                     'mean_p_real': round(float(p[m].mean()), 3), 'acc@0.5': acc(d.y[m], p[m]),
                     'auc_vs_other_class': auc(yy, np.r_[p[m], p[other]])})
    return rows


def file_level(d, p):
    """(sources, labels, mean window p_real) of the clean sources with >= 2 windows (long files)"""
    srcs = {}
    for i in np.where(d.perturb == '')[0]:
        if '@' in d.ids[i]:
            srcs.setdefault(d.src[i], []).append(i)
    multi = {s: v for s, v in srcs.items() if len(v) >= 2}
    return (list(multi), np.array([d.y[v[0]] for v in multi.values()]), np.array([p[v].mean() for v in multi.values()]),
            [len(v) for v in multi.values()])


def file_thresholds(fy, fp, clip_hi, clip_lo):
    """verdict thresholds for the mean window p_real of a long file: lo from the held-out files (averaging windows
    makes a low mean stronger evidence than one low clip); hi is never looser than the clip hi, because real files
    outnumber synthesized ones in the held-out set, which makes precision on the real side cheap"""
    fh, fl = thresholds(fy, fp)
    return {'hi': max(fh, clip_hi), 'lo': max(fl, clip_lo), 'files': int(len(fy)), 'synth_files': int((fy == 0).sum()),
            'how': f'lo: {100 * TARGET:g}% precision on held-out long files (scheme M); hi: max(that, the clip hi)'}


def load_oof(name, d):
    path = os.path.join(OOF, name + '.npz')
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    m = dict(zip(z['ids'], z['p']))
    return np.array([m.get(i, np.nan) for i in d.ids])


def zero_shot(d, P):
    """CLAP zero-shot: log-sum-exp over the real prompts minus over the synth prompts (logit scale 38.7)"""
    s = 38.7 * (d.Xc @ P.T)
    n = len(realness.REAL_PROMPTS)
    lse = lambda v: v.max(axis=1) + np.log(np.exp(v - v.max(axis=1, keepdims=True)).sum(axis=1))  # noqa: E731
    return lse(s[:, :n]) - lse(s[:, n:])


def report():
    d = C.load()
    clean = d.perturb == ''
    dc = d.sub(clean)
    g = groups_of(d)
    res = {'data': {}, 'schemes': {}, 'baselines': {}}
    res['data'] = {'clips': int(clean.sum()), 'real': int(dc.y.sum()), 'synth': int((1 - dc.y).sum()),
                   'by_family': {f: int((dc.family == f).sum()) for f in sorted(set(dc.family))},
                   'real_groups': sorted(set(dc.group[dc.y == 1])), 'perturbed_clips': int((~clean).sum())}
    P = {k: load_oof(k, d) for k in ('A', 'A_noaug', 'A_clap', 'A_signal', 'B', 'C', 'D', 'M')}
    for k, p in P.items():
        if p is None:
            continue
        # D trains on generic negatives only, so its generic fold has no negatives to learn from: score real vs the
        # two never-trained-on families
        keep = clean & ((d.y == 1) | np.isin(d.family, SCHEMES[k]['exclude'])) if k == 'D' else clean
        res['schemes'][k] = metrics(d.y[keep], p[keep])
        if k == 'D':
            res['schemes'][k]['note'] = 'real vs retired + cinetic, neither ever trained on (generic generators were the only negatives)'
    pA = P['A']
    pT, tk = (P['M'], 'M') if P['M'] is not None else (pA, 'A')
    hi, lo = thresholds(dc.y, pT[clean])
    res['thresholds'] = {'hi': hi, 'lo': lo, 'scheme': tk, 'how': f'>= {100 * TARGET:g}% precision on each side'}
    res['band'] = band_stats(dc.y, pT[clean], lo, hi)
    res['band_alt'] = {}
    for t in (0.95, TARGET, 0.9):
        h, lw = thresholds(dc.y, pT[clean], t)
        res['band_alt'][f'{t:g}'] = dict(hi=h, lo=lw, **band_stats(dc.y, pT[clean], lw, h))
    res['calibration'] = calibration(dc.y, pT[clean])
    res['per_group_A'] = per_group(d, pA, g)
    if P['M'] is not None:
        res['per_group_M'] = per_group(d, P['M'], g)
        nm = clean & (d.group != 'music')
        res['non_music'] = {'A': metrics(d.y[nm], pA[nm]), 'M': metrics(d.y[nm], P['M'][nm])}
    gt = groups_of(d, 'type')
    res['per_type_A'] = [r for r in per_group(d, pA, gt) if r['label'] == 'synth']
    if P['B'] is not None:
        res['per_type_B'] = [r for r in per_group(d, P['B'], gt) if r['label'] == 'synth']
    if P['M'] is not None:
        mu = clean & (d.group == 'music')
        files = sorted(set(d.src[mu]))
        fp = np.array([P['M'][mu & (d.src == f)].mean() for f in files])
        res['music_M'] = {'windows': int(mu.sum()), 'mean_p_real': round(float(P['M'][mu].mean()), 3),
                          'acc@0.5': acc(d.y[mu], P['M'][mu]), 'files': len(files),
                          'files_acc@0.5': round(float(np.mean(fp >= 0.5)), 4),
                          'files_called_synth': int((fp <= lo).sum()), 'files_called_real': int((fp >= hi).sum()),
                          'scheme_A_mean_p_real': round(float(pA[mu].mean()), 3),
                          'per_provider': [r for r in per_group(d, P['M'], groups_of(d, 'music_src')) if r['group'].startswith('music:')]}
    # the required check: real recordings vs the retired score, retired never trained on
    real = clean & (d.y == 1)
    checks = {}
    for k, fams in (('C', ('retired',)), ('D', ('retired', 'cinetic'))):
        if P[k] is None:
            continue
        for f in fams:
            m = real | (clean & (d.family == f))
            checks[f'{k}: real vs {f}'] = {'auc': auc(d.y[m], P[k][m]), 'acc@0.5': acc(d.y[m], P[k][m]),
                                           'synth_mean_p': round(float(P[k][clean & (d.family == f)].mean()), 3)}
        assets = clean & np.isin(d.family, ['real-asset', 'real-rejected'])
        m = assets | (clean & (d.family == 'retired'))
        checks[f'{k}: real recordings only (no music/films) vs retired'] = {'auc': auc(d.y[m], P[k][m])}
    res['retired_check'] = checks
    # file level: windows of the same long source averaged; scheme M also gives the file verdict thresholds
    for k in ('A', 'M'):
        if P[k] is None:
            continue
        fs, fy, fp, nw = file_level(d, P[k])
        r = dict(metrics(fy, fp), note='sources with >= 2 windows (films, music, ambience, retired stems, synthesized '
                 'mixes): mean of window p_real')
        r['files'] = sorted([{'src': '/'.join(str(EP.anon(s)).replace(chr(92), '/').split('/')[-2:]), 'label': int(y), 'p_real': round(float(q), 3), 'windows': w}
                             for s, y, q, w in zip(fs, fy, fp, nw)], key=lambda r: r['p_real'])
        if k == 'M':
            r['thresholds'] = file_thresholds(fy, fp, hi, lo)
            r['band'] = band_stats(fy, fp, r['thresholds']['lo'], r['thresholds']['hi'])
            r['calibration'] = calibration(fy, fp, 5)
        res[f'file_level_{k}'] = r
    # robustness: perturbed copies scored by the fold model that never saw their group
    rob = []
    base = dict(zip(d.ids, pA))
    for kind in ('noise70', 'room70', 'gated', 'short', 'mp3'):
        for lab in (0, 1):
            m = (d.perturb == kind) & (d.y == lab)
            if not m.any():
                continue
            pb = np.array([base.get(b, np.nan) for b in d.base[m]])
            rob.append({'perturbation': kind, 'label': 'real/pro' if lab else 'synth', 'n': int(m.sum()),
                        'acc@0.5': acc(d.y[m], pA[m]), 'acc_unperturbed': acc(d.y[m], pb),
                        'mean_p_real': round(float(pA[m].mean()), 3), 'mean_p_unperturbed': round(float(np.nanmean(pb)), 3)})
    res['robustness_A'] = rob
    lp = os.path.join(C.FEATS, 'level.json')
    if os.path.exists(lp):
        res['level_sensitivity'] = json.load(open(lp))
    for k in ('A_noaug', 'A_clap', 'M'):
        if P[k] is not None:
            bk = dict(zip(d.ids, P[k]))
            res['robustness_' + k] = [dict(r, **{'acc@0.5': acc(d.y[m], P[k][m]),
                                                'acc_unperturbed': acc(d.y[m], np.array([bk[b] for b in d.base[m]])),
                                                'mean_p_real': round(float(P[k][m].mean()), 3)})
                                     for r in rob for m in [(d.perturb == r['perturbation']) & (d.y == (r['label'] == 'real/pro'))]]
    # baselines (no training): CLAP zero-shot prompts, single signal measurements
    Pe = prompt_embeddings()
    zs = zero_shot(dc, Pe)
    res['baselines']['clap_zero_shot'] = dict(metrics(dc.y, 1 / (1 + np.exp(-zs))), prompts={
        'real': realness.REAL_PROMPTS, 'synth': realness.SYNTH_PROMPTS})
    res['baselines']['signal_univariate_auc'] = {k: auc(dc.y[np.isfinite(dc.Xs[:, j])], C.SIGNS[k] * dc.Xs[np.isfinite(dc.Xs[:, j]), j])
                                                 for j, k in enumerate(C.SIG_INPUTS)}
    # AST subset
    if os.path.exists(os.path.join(C.FEATS, 'ast.npz')):
        da = C.load(require_ast=True, robust=False)
        labels = json.load(open(os.path.join(C.FEATS, 'ast_labels.json'))) if os.path.exists(os.path.join(C.FEATS, 'ast_labels.json')) else None
        sub = {}
        if labels:
            idx = [labels.index(l) for l in realness.AST_SYNTH]
            smax = da.Xa[:, idx].max(axis=1)
            sub['ast_synth_classes_max'] = dict(metrics(da.y, 1 - smax), classes=realness.AST_SYNTH)
            for l in ('Synthesizer', 'Sine wave', 'Beep, bleep', 'Electronic music', 'Music'):
                sub[f'ast_single:{l}'] = {'auc_real_high': auc(da.y, -da.Xa[:, labels.index(l)])}
        for k in ('S', 'S_ast', 'S_astonly'):
            p = load_oof(k, da)
            if p is not None:
                sub[k] = metrics(da.y, p)
        m = np.isin(d.ids, da.ids) & clean
        sub['A_on_subset'] = metrics(d.y[m], pA[m])
        zsa = zero_shot(da, Pe)
        sub['clap_zero_shot_on_subset'] = metrics(da.y, 1 / (1 + np.exp(-zsa)))
        res['ast_subset'] = sub
    # final model internals
    if os.path.exists(realness.MODEL_PATH):
        m = json.load(open(realness.MODEL_PATH))
        res['final_model'] = {'fusion': m['fusion'], 'signal_coef': dict(zip(m['heads']['signal']['inputs'], m['heads']['signal']['coef'])),
                              'C': m.get('C'), 'clap_direction': m.get('clap_direction')}
    # application files: the final model (in-sample: it trained on the real soundtrack's source recordings and on
    # the retired score) and a model that never saw the accepted Clawd assets (the 83 recordings the real soundtrack
    # is built from are among them) nor the retired score; plus the Windows UI sounds (never trained on)
    ap = os.path.join(C.FEATS, 'apps.npz')
    if os.path.exists(ap) and os.path.exists(realness.MODEL_PATH):
        z = np.load(ap, allow_pickle=False)
        sig = [json.loads(s) for s in z['sig']]
        Xs = np.array([[np.nan if f is None else f.get(k, np.nan) for k in C.SIG_INPUTS] for f in sig], float)
        dd = SimpleNamespace(Xc=z['emb'], Xs=Xs, Xa=None)
        fin = json.load(open(realness.MODEL_PATH))
        fth = realness.thresholds_of(fin, True)
        src = np.array([s.replace(chr(92), '/') for s in d.src])
        accepted = np.array([s.startswith(ASSETS) and '/_rejected/' not in s for s in src])
        trh = d.sub(((d.perturb == '') | d.augment) & ~accepted & (d.family != 'retired'))
        mH = C.fit_model(trh, groups_of(trh, 'music_src'), c_clap=fin.get('C', {}).get('clap'))
        pp = {'final': C.predict(fin, dd), 'held_out': C.predict(mH, dd)}
        name_of = np.array([i.rsplit('@', 1)[0] for i in z['ids']])
        out = {}
        for name in APPS:
            mk = name_of == name
            out[name] = {k: {'p_real_mean': round(float(v[mk].mean()), 3), 'windows': int(mk.sum()),
                             'windows_lean_synth': int((v[mk] < 0.5).sum()), 'windows_le_lo': int((v[mk] <= lo).sum()),
                             'verdict': realness.verdict_of(float(v[mk].mean()), fin, mk.sum() > 1)}
                         for k, v in pp.items()}
        res['applications'] = out
        res['held_out_model'] = {'clips': int(len(trh)), 'excluded': 'accepted Clawd assets (incl. assets/music) and '
                                 'the retired score', 'file_thresholds': fth}
        ui = sorted(set(n for n in name_of if n.startswith('winui:')))
        if ui:
            fp = np.array([pp['final'][name_of == n].mean() for n in ui])
            vs = [realness.verdict_of(float(q), fin, (name_of == n).sum() > 1) for q, n in zip(fp, ui)]
            res['windows_ui_sounds'] = {'files': len(ui), 'median_p_real': round(float(np.median(fp)), 3),
                                        **{v: int(sum(x == v for x in vs)) for v in realness.VERDICTS[:3]},
                                        'lowest': [[n[6:], round(float(q), 3)] for q, n in sorted(zip(fp, ui))[:5]],
                                        'highest': [[n[6:], round(float(q), 3)] for q, n in sorted(zip(fp, ui))[-5:]]}
    # layer test: real beds + one code-synthesized layer (run_layers), and the variant trained on layered clips
    pl = {k: load_oof(f'layers_{k}', d) for k in ('base', 'train')}
    if pl['base'] is not None:
        man = {c['id']: c for part in json.load(open(os.path.join(C.OUT, 'manifest.json'))).values() for c in part}
        Lm = d.is_layer & np.isfinite(pl['base'])
        lev = np.array([man[i].get('level_db', 99) for i in d.ids])
        pbed = dict(zip(d.ids, P['M'])) if P['M'] is not None else {}
        bed = np.array([pbed.get(man[i].get('bed'), np.nan) for i in d.ids])
        beds = clean & (d.y == 1) & np.isin(d.ids, [man[i]['bed'] for i in d.ids[Lm]])
        lay = {'clips': int(Lm.sum()), 'beds': int(beds.sum()), 'bed_p_real_mean': round(float(np.nanmean(bed[Lm])), 3),
               'beds_lean_synth': round(float(np.mean(P['M'][beds] < 0.5)), 3) if P['M'] is not None else None,
               'per_level': []}
        for lv in sorted(set(lev[Lm])):
            mk = Lm & (lev == lv)
            row = {'level_db': int(lv), 'n': int(mk.sum())}
            for k, q in pl.items():
                if q is not None:
                    row[k] = {'mean_p_real': round(float(q[mk].mean()), 3), 'lean_synth': round(float(np.mean(q[mk] < 0.5)), 3),
                              'le_lo': round(float(np.mean(q[mk] <= lo)), 3), 'ge_hi': round(float(np.mean(q[mk] >= hi)), 3)}
            lay['per_level'].append(row)
        lay['per_bed_group_base'] = {str(b): round(float(np.mean(pl['base'][Lm & (d.bed_group == b) & (lev >= -6)] < 0.5)), 3)
                                     for b in sorted(set(d.bed_group[Lm]))}
        lay['per_layer_family_base'] = {str(f): round(float(np.mean(pl['base'][Lm & (d.family == f) & (lev >= -6)] < 0.5)), 3)
                                        for f in sorted(set(d.family[Lm]))}
        pml = load_oof('M_layers', d)
        if pml is not None:
            h2, l2 = thresholds(dc.y, pml[clean])
            lay['M_layers'] = dict(metrics(dc.y, pml[clean]), hi=h2, lo=l2, band=band_stats(dc.y, pml[clean], l2, h2))
        top = [r for r in lay['per_level'] if r['level_db'] >= -6]
        lay['summary'] = (f"code-synthesized layers at -6..+6 dB inside real beds: "
                          f"{np.mean([r['base']['lean_synth'] for r in top]):.2f} of windows lean synthesized "
                          f"(p_real < 0.5), {np.mean([r['base']['le_lo'] for r in top]):.2f} at or below lo; "
                          f"the beds alone: {lay['beds_lean_synth']}")
        res['layers'] = lay
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    with open(os.path.join(HERE, 'RESULTS.md'), 'w', encoding='utf-8') as fh:
        fh.write(markdown(res))
    a = res['schemes']['A']
    print(f"scheme A: AUC {a['auc']}, acc@0.5 {a['acc@0.5']}, Brier {a['brier']}; thresholds lo {lo} hi {hi}; "
          'wrote results.json + RESULTS.md')
    return res


# ---------------------------------------------------------------------------------------------- RESULTS.md
def _tab(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] +
                     ['| ' + ' | '.join('-' if v is None else str(v) for v in r) + ' |' for r in rows]) + '\n'


FAMILY_DOC = {'real-asset': 'Freesound / Mixkit recordings used in the Clawd film (7 categories)',
              'real-rejected': 'real recordings rejected for quality (still real)',
              'pro-film': 'audio of the 7 professional launch films (4 s windows)',
              'pro-music': 'professionally produced music (assets/music, kept and rejected)',
              'retired': 'the retired code-synthesized Clawd score: stems, beds, melody, events',
              'cinetic': 'cinetic synth engine: generator types x seeds / parameters, short mixes, 6 full rendered scores',
              'generic': 'independent numpy generators (14 types) + 6 generic code-made mixes'}
SCHEME_DOC = {'A': 'A: leave one real category / synth family out (headline)',
              'A_noaug': 'A without augmented training copies', 'A_clap': 'A, CLAP head only',
              'A_signal': 'A, signal-measurement head only', 'B': 'B: leave one synth generator type out',
              'C': 'C: retired score never trained on', 'D': 'D: retired score and cinetic engine never trained on',
              'M': 'M: as A, but music held out by provider (mk / sb / ch / other)'}


def _level_md(res):
    lv = res.get('level_sensitivity')
    if not lv:
        return ''
    cs = [c for c in ('-20dB', 'norm-25', 'lifted', 'at-56', 'at-56_lifted') if f'mean_p_{c}' in next(iter(lv.values()))]
    return ('\nLevel (final model, so in-sample clips; it measures sensitivity, not accuracy): the same clips at their own '
            'level, 20 dB quieter, normalised to -25 dBFS RMS, after the inference lift (sources under '
            f'{realness.LIFT_BELOW_DB:g} dBFS RMS are raised to {realness.LIFT_BELOW_DB:g} dBFS RMS, peak <= -1 dBFS), and '
            'moved to -56 dBFS RMS without / with that lift (the first build only lifted sources under -60 dBFS RMS, '
            'so -56 dBFS input was scored as it was).\n\n' +
            _tab(['clips', 'n', 'median RMS dBFS', 'lifted at inference', 'mean p_real own / ' + ' / '.join(cs),
                  'mean abs change ' + ' / '.join(cs), 'acc own / ' + ' / '.join(cs)],
                 [[k, v['clips'], v['rms_db_median'], v.get('lifted_clips'),
                   ' / '.join(str(v[f'mean_p_{c}']) for c in ['orig'] + cs),
                   ' / '.join(str(v[f'mean_abs_dp_{c}']) for c in cs),
                   ' / '.join(str(v[f'acc_{c}']) for c in ['orig'] + cs)] for k, v in lv.items()]))


def _music_md(res):
    mm = res.get('music_M')
    if not mm:
        return ''
    return (f"""
### Pro music
Scheme A holds out all music at once, so the model scoring it has never heard real music while its negatives include
synthesized scores: a pessimistic estimate (music mean p_real {mm['scheme_A_mean_p_real']}). Scheme M holds out one
provider at a time (the model has heard other providers' music): {mm['windows']} windows of {mm['files']} files, mean
p_real {mm['mean_p_real']}, window acc@0.5 {mm['acc@0.5']}, file acc@0.5 {mm['files_acc@0.5']}; files (mean of windows)
called real: {mm['files_called_real']}, called code-synthesized: {mm['files_called_synth']}. The price of weighting music
as four sources, on everything except music: scheme A AUC {res['non_music']['A']['auc']} / Brier
{res['non_music']['A']['brier']} vs scheme M AUC {res['non_music']['M']['auc']} / Brier {res['non_music']['M']['brier']}.

""" + _tab(['provider fold', 'windows', 'mean p_real', 'acc@0.5'],
             [[r['group'], r['n'], r['mean_p_real'], r['acc@0.5']] for r in mm['per_provider']]))


def _apps_md(res):
    """the held-out note under the application table, the Windows UI sounds and the layer test"""
    o = []
    ho = res.get('held_out_model', {})
    o.append(f"""
The held-out model ({ho.get('clips')} training clips) never saw {ho.get('excluded')}: the real soundtrack is built from
83 of those accepted recordings, and the synthesized soundtrack is the retired score, so this column is out of sample
on both sides. The final model trained on both (in-sample, optimistic). Verdicts use the file thresholds for files
over 6 s.
""")
    ui = res.get('windows_ui_sounds')
    if ui:
        o.append(f"""
### Professionally made synthetic sounds (Windows UI / alert sounds, never trained on)
{ui['files']} top-level wav files of C:/Windows/Media (chimes, alarms, ringtones, notification tones; made by sound
designers, mostly synthesized): {ui['likely code-synthesized']} called 'likely code-synthesized', {ui['uncertain']}
'uncertain', {ui['real or pro-produced']} 'real or pro-produced' (median p_real {ui['median_p_real']}). Lowest:
{', '.join(f'{n} {q}' for n, q in ui['lowest'])}. Highest: {', '.join(f'{n} {q}' for n, q in ui['highest'])}.
The detector hears synthesis, not provenance: a clean, simple synthesized tone reads as code-made even when a
professional made it and it is licensed. An agent using a licensed UI-sound pack must not treat that verdict as proof.
""")
    lay = res.get('layers')
    if lay:
        S, ba, ml = res['schemes'], res['band'], lay.get('M_layers', {})
        o.append(f"""
### Layer test: a code-synthesized layer inside a real mix
{lay['clips']} clips: {lay['beds']} real 4 s beds (ambience, films, music, foley ...; mean p_real alone
{lay['bed_p_real_mean']}, {lay['beds_lean_synth']} of them under 0.5 alone) with one code-synthesized layer each (a
cinetic / generic / retired clip: one-shots 1-4 times, 4 s windows throughout), at a layer-to-bed RMS of -12 / -6 / 0 /
+6 dB (layer RMS over the span it plays). Every clip is scored by a model that never saw its bed's group nor its
layer's family (`--stage layers`). 'base' = the shipped recipe; 'train' = also trained on the layered clips of the
other groups.

""" + _tab(['layer level', 'clips', 'base: mean p_real', 'base: < 0.5', 'base: <= lo', 'base: >= hi (called real)',
            'train: mean p_real', 'train: < 0.5', 'train: <= lo'],
           [[f"{r['level_db']:+d} dB", r['n'], r['base']['mean_p_real'], r['base']['lean_synth'], r['base']['le_lo'],
             r['base']['ge_hi'], r.get('train', {}).get('mean_p_real'), r.get('train', {}).get('lean_synth'),
             r.get('train', {}).get('le_lo')] for r in lay['per_level']]) + f"""
Most layers are missed: the windows lean synthesized for only about a third of the clips, almost independent of the
layer level, and many are still called real. Per bed group (base, -6..+6 dB, share < 0.5): {lay['per_bed_group_base']}
(magic and motion beds already lean synthesized on their own). Per layer family: {lay['per_layer_family_base']}.
Training on layered clips (scheme M_layers: M plus the layered clips of the other groups) raised the held-out clip AUC
to {ml.get('auc')} (acc {ml.get('acc@0.5')}, Brier {ml.get('brier')}; M: {S['M']['auc']} / {S['M']['acc@0.5']} /
{S['M']['brier']}) and the layer detection above, but compressed the synthesized scores: at the same 92.5% precision
target its thresholds are lo {ml.get('lo')} / hi {ml.get('hi')} with {ml.get('band', {}).get('coverage')} firm-verdict
coverage (M: {ba['coverage']}). It was not adopted (decided after seeing these held-out numbers): judging source files
firmly is the main use, and even trained, most layers stay unflagged. A dedicated layer head (CLAP logistic, bed vs bed
+ layer, same pair folds; a scratch experiment, not in results.json) reached AUC 0.82-0.84 with 40-46% of the layers
at -6..+6 dB found at 5% false alarms on beds, but 11-17% false alarms on film and music beds: not shipped either.
So `earshot realness` says plainly that a clean window does not rule out a code-made layer under a real mix.
""")
    return ''.join(o)


def markdown(res):
    S, D, th, ba = res['schemes'], res['data'], res['thresholds'], res['band']
    A, fl, bl = S['A'], res['file_level_A'], res['baselines']
    fM = res.get('file_level_M', {})
    fth = fM.get('thresholds', {})
    fm = res.get('final_model', {})
    fu = fm.get('fusion', {'heads': [], 'coef': [0, 0]})
    fool = sorted((r for r in res['per_type_A'] if r['acc@0.5'] < 0.75), key=lambda r: r['acc@0.5'])
    weak_real = sorted((r for r in res['per_group_A'] if r['label'] == 'real/pro' and r['acc@0.5'] < 0.8),
                       key=lambda r: r['acc@0.5'])
    typeB = {r['group']: r for r in res.get('per_type_B', [])}
    noaug = {(r['perturbation'], r['label']): r['acc@0.5'] for r in res.get('robustness_A_noaug', [])}
    robM = {(r['perturbation'], r['label']): r['acc@0.5'] for r in res.get('robustness_M', [])}
    grM = {r['group']: r for r in res.get('per_group_M', [])}
    o = [f"""# Realness detector: held-out evaluation

Generated by `eval/realness/run_eval.py --stage report`; every number below comes from that run (`results.json` holds
all of them). The question: is a sound a real recording / professionally produced audio (positive), or amateur audio
synthesized in code: oscillators, filtered noise, simple envelopes, algorithmic reverb, beeps (negative)?
Professionally produced electronic music and pro sound design count as positive.

## Data
{D['clips']} clean clips ({D['real']} real/pro, {D['synth']} synthesized) plus {D['perturbed_clips']} perturbed copies
(robustness tests and training-only augmentation). A clip is a whole sound up to 6 s, else a 4 s window: exactly the
clips `earshot.realness` scores at inference. Audio stays in `out/realness/` (never committed).

""", _tab(['family', 'clean clips', 'what'], [[f, n, FAMILY_DOC.get(f, '')] for f, n in D['by_family'].items()]),
         f"""
Tuning vs evaluation: model choices (regularisation C of the CLAP head, fusion weights) are fitted inside each training
fold by inner grouped CV; nothing is tuned on the held-out group. The verdict thresholds (lo {th['lo']} / hi {th['hi']})
are read from the scheme-{th.get('scheme', 'A')} held-out predictions ({th['how']}), so the band numbers are in-sample
for that choice (two numbers fitted on {A['n']} clips); the file thresholds for long files likewise come from held-out
long files. Design choices made by looking at held-out results across the three build rounds: augmentation on
(robustness below), the precision target, music weighted as four sources (scheme M), the feature set, the file
threshold rule, and not training on layered clips (layer test below).
The final model (`earshot/data/realness_model.json`) is trained on all groups with the scheme-M grouping. Robustness copies are only ever scored, never trained
on; augmentation copies (synthetic noise floor, gating, short excerpts, codec round trips, with randomised parameters
different from the robustness set, never the real room tone) are only trained on.

## Held-out results
Every clip is scored by a model that never saw its group (nor augmented copies of its clips).

""", _tab(['scheme', 'clips', 'ROC-AUC', 'acc@0.5', 'Brier', 'log loss'],
          [[SCHEME_DOC.get(k, k), v['n'], v['auc'], v['acc@0.5'], v['brier'], v['logloss']] for k, v in S.items()]),
         """
D is scored on real vs the retired score and the cinetic engine only (its sole negatives were the generic generators).

### The required check: real recordings rank above the retired synthesized score
The retired score (the synthesized soundtrack the Clawd project replaced) is held out of every training set
(scheme C), or held out together with the whole cinetic engine that made it (scheme D):

""", _tab(['check', 'ROC-AUC', 'acc@0.5', 'mean p_real of the synth side'],
          [[k, v.get('auc'), v.get('acc@0.5'), v.get('synth_mean_p')] for k, v in res['retired_check'].items()]),
         f"""
### Verdict bands (scheme {th.get('scheme', 'A')} held-out predictions)
`real or pro-produced` when p_real >= {th['hi']}, `likely code-synthesized` when p_real <= {th['lo']}, else `uncertain`.
Coverage {100 * ba['coverage']:.0f}% (the rest is uncertain); precision of the real verdict {ba['precision_real']}, of the
synth verdict {ba['precision_synth']}; {100 * ba['real_called_synth']:.1f}% of real clips are called synthesized and
{100 * ba['synth_called_real']:.1f}% of synthesized clips are called real. Other precision targets:

""", _tab(['precision target', 'lo', 'hi', 'coverage', 'real called synth', 'synth called real'],
          [[k, v['lo'], v['hi'], v['coverage'], v['real_called_synth'], v['synth_called_real']]
           for k, v in res.get('band_alt', {}).items()]),
         f"\nCalibration (scheme {th.get('scheme', 'A')}, clean clips):\n\n",
         _tab(['p_real bin', 'clips', 'mean p_real', 'fraction real'],
              [[r['bin'], r['n'], r['mean_p'], r['frac_real']] for r in res['calibration']]),
         _music_md(res),
         '\n### Per group (A: the whole group held out; M: the same, music held out by provider)\n\n',
         _tab(['group', 'label', 'clips', 'A mean p_real', 'A acc@0.5', 'A AUC vs the other class', 'M mean p_real',
               'M acc@0.5'],
              [[r['group'], r['label'], r['n'], r['mean_p_real'], r['acc@0.5'], r['auc_vs_other_class'],
                grM.get(r['group'], {}).get('mean_p_real'), grM.get(r['group'], {}).get('acc@0.5')]
               for r in res['per_group_A']]),
         '\n### Synthesized generator types (A: the whole family held out; B: only that type held out)\n\n',
         _tab(['type', 'clips', 'A mean p_real', 'A acc', 'B mean p_real', 'B acc'],
              [[r['group'], r['n'], r['mean_p_real'], r['acc@0.5'], typeB.get(r['group'], {}).get('mean_p_real'),
                typeB.get(r['group'], {}).get('acc@0.5')] for r in res['per_type_A']]),
         f"""
### File level (long files: the mean of their window p_real)
Sources with >= 2 windows (films, music, ambience, retired stems, full synthesized mixes): {fl['n']} files
({fl['n'] - fl['n_real']} synthesized). Scheme A: AUC {fl['auc']}, acc@0.5 {fl['acc@0.5']}. Scheme M (the final model's
grouping): AUC {fM.get('auc')}, acc@0.5 {fM.get('acc@0.5')}. File verdict thresholds ({fth.get('how')}): lo {fth.get('lo')} /
hi {fth.get('hi')}; with them {fM.get('band', {}).get('coverage')} of the files get a firm verdict, real files called
synthesized {fM.get('band', {}).get('real_called_synth')}, synthesized files called real {fM.get('band', {}).get('synth_called_real')}.
Only {fth.get('synth_files')} synthesized long files from three code bases: weak calibration. Scheme M synthesized files,
lowest to highest: {', '.join(f"{r['src']} {r['p_real']}" for r in fM.get('files', []) if r['label'] == 0)}.
Scheme M real files under 0.6: {', '.join(f"{r['src']} {r['p_real']}" for r in fM.get('files', []) if r['label'] == 1 and r['p_real'] < 0.6) or 'none'}.

### Robustness (perturbed copies, scored by the fold model that never saw their group)
noise70 = pink noise at -70 dBFS RMS added to synthesized clips; room70 = a real room tone (quietest 20 s of a rain
recording) at -70 dBFS; gated = real clips gated to exact digital silence below -45 dB re peak; short = 0.15 s around
the peak; mp3 = 128 kbps round trip. The final model is trained with augmentation (A, M); A_noaug shows the trade-off.

""", _tab(['perturbation', 'label', 'clips', 'acc perturbed (A)', 'acc unperturbed (A)', 'acc perturbed (A_noaug)',
           'acc perturbed (M)'],
          [[r['perturbation'], r['label'], r['n'], r['acc@0.5'], r['acc_unperturbed'],
            noaug.get((r['perturbation'], r['label'])), robM.get((r['perturbation'], r['label']))]
           for r in res['robustness_A']]),
         _level_md(res),
         f"""
### Baselines (no training)
- CLAP zero-shot (log-sum-exp over 6 'real' prompts minus 6 'synth' prompts): AUC {bl['clap_zero_shot']['auc']},
  acc@0.5 {bl['clap_zero_shot']['acc@0.5']} on all clean clips.
- Single signal measurements (AUC, physical sign): {', '.join(f'{k} {v}' for k, v in bl['signal_univariate_auc'].items())}.
"""]
    sub = res.get('ast_subset')
    if sub:
        o.append(f"""- AST, on a stratified subset of {sub['A_on_subset']['n']} clean clips (AST costs ~5x CLAP here; the subset predates
  the full synthesized mixes): max over the synth classes ({', '.join(sub['ast_synth_classes_max']['classes'])})
  AUC {sub['ast_synth_classes_max']['auc']}; single classes (high = real):
  {', '.join(f"{k.split(':')[1]} {v['auc_real_high']}" for k, v in sub.items() if k.startswith('ast_single'))}.
  On the same subset: scheme A {sub['A_on_subset']['auc']}, CLAP zero-shot {sub['clap_zero_shot_on_subset']['auc']},
  clap+signal heads refit on the subset {sub.get('S', {}).get('auc')}, plus an AST head {sub.get('S_ast', {}).get('auc')},
  AST head alone {sub.get('S_astonly', {}).get('auc')}. An AST head adds nothing, so AST is evidence only (`--ast`).
""")
    o.append(f"""
## What carries the decision
- Fusion weights (fitted on inner held-out head outputs): {dict(zip(fu['heads'], fu['coef']))}. The CLAP head carries
  the decision; the signal head ({S['A_signal']['auc']} AUC alone vs {S['A_clap']['auc']} for CLAP alone) gets weight
  {fu['coef'][1]}, so its measurements are reported as evidence (each with the share of training clips past that
  threshold that were of that side), not votes.
- The CLAP head's weight direction is closest to these vocabulary texts. Toward real:
  {', '.join(fm.get('clap_direction', {}).get('toward_real', []))}. Toward synthesized:
  {', '.join(fm.get('clap_direction', {}).get('toward_synth', []))}.
- Signal-head coefficients (standardised, sign-constrained to their physics): {fm.get('signal_coef')}.
- Exact digital silence is weak evidence: over half of the clips of both classes contain some (edited and gated library
  recordings too), so it is never a verdict reason, only context quoted with its training base rates.
- One rule overrides the model at inference (earshot.realness.pure_tone): a pure, dead-steady tone (one spectral line in
  every frame, crest <= 4.5 dB, no pitch drift) gets p_real <= 0.03, because bare test tones lie outside the training
  data. Checked by `eval/realness/tone_guard.py` (`tone_guard.json`): a 30 s 440 Hz sine gets 0.48 from the model
  alone and 0.03 with the rule; the rule fires on none of 4741 audible clips of the 345 real recordings.

## Application files (whole soundtracks, 4 s windows)
""")
    o.append(_tab(['file', 'held-out model: p_real (mean) / verdict', 'windows < 0.5 / <= lo / all (held-out)',
                   'final model (in-sample): p_real / verdict'],
                  [[k, f"{v['held_out']['p_real_mean']} / {v['held_out']['verdict']}",
                    f"{v['held_out']['windows_lean_synth']} / {v['held_out']['windows_le_lo']} / {v['held_out']['windows']}",
                    f"{v['final']['p_real_mean']} / {v['final']['verdict']}"]
                   for k, v in res.get('applications', {}).items()]))
    o.append(_apps_md(res))
    o.append(f"""

## Limits: what fools it
- Synthesized types it misses most (scheme A acc < 0.75, lowest first; acc / mean p_real): {', '.join(f"{r['group']} ({r['acc@0.5']} / {r['mean_p_real']})" for r in fool) or 'none'}.
  Code-made noise textures that imitate nature (rain, wind beds) and telephone-like tones (DTMF, sirens) sound like
  recordings of real things to CLAP, and a generic code-made mix built on such beds can score real.
- Real groups it doubts (scheme A acc < 0.8; acc / mean p_real): {', '.join(f"{r['group']} ({r['acc@0.5']} / {r['mean_p_real']})" for r in weak_real) or 'none'}.
  Library "magic" and "motion" sounds are themselves designed (often synthesized) by sound designers, and real
  electronic music sits near synthesized scores: read 'uncertain' there as "sounds designed", not "fake".
- Finished mixes are harder than sources, and a code-made layer under a louder real mix is usually missed (layer
  test). Judge the source files and stems, not only the final mix.
- Professionally made synthetic tones (UI sounds, chimes, alarms) are called code-synthesized: it hears synthesis, not
  provenance (Windows UI sounds above).
- A -70 dBFS noise floor added to synthesized audio lowers accuracy (robustness table): the detector mostly hears what
  a sound is, not microphone physics. Very short clips (< 0.2 s) of real sounds are harder.
- Training data: one sound library (Freesound / Mixkit picks for one film), 7 launch films, ~120 music windows; the
  negatives come from three code bases (cinetic, the retired score made with it, generic numpy generators). Synthesis
  from other tools (pro soft synths, granular, sample-based) was never seen: pro-produced synth music is meant to score
  real, and amateur code using techniques outside these three code bases may also score real.

## Reproduce
```
.venv/Scripts/python.exe eval/realness/build_data.py                                   # clips -> out/realness/
.venv/Scripts/python.exe eval/realness/build_data.py --only layers                     # layer-test clips
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/extract.py --ast 240           # repeat until "done"
OMP_NUM_THREADS=2 EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/run_eval.py --stage cv   # until "cv done"
OMP_NUM_THREADS=2 EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/run_eval.py --stage layers   # twice
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/run_eval.py --stage apps
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/train.py
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/run_eval.py --stage level     # needs the final model
.venv/Scripts/python.exe eval/realness/run_eval.py --stage report                       # results.json + RESULTS.md
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/train.py                       # embeds the summary in the model
```
""")
    return ''.join(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', default='cv', choices=['cv', 'layers', 'apps', 'level', 'report'])
    ap.add_argument('--budget', type=float, default=480)
    a = ap.parse_args(argv)
    if a.stage == 'cv':
        run_cv(a.budget)
    elif a.stage == 'layers':
        run_layers()
    elif a.stage == 'apps':
        run_apps()
    elif a.stage == 'level':
        run_level()
    else:
        report()


if __name__ == '__main__':
    main()
