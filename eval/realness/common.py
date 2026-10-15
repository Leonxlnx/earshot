"""Shared training code for the realness detector: dataset loading, the two logistic heads, the fusion layer and
grouped cross-validation. Used by train.py (final model) and run_eval.py (held-out evaluation).

Model: head_clap = standardised L2 logistic regression on the 512-d CLAP embedding; head_signal = the same on the
signal measurements of earshot.realness.signal_features; fusion = logistic regression on the two head logits, fitted on
inner out-of-group head predictions (so its weights reflect how well each head generalises to unseen groups).
Sample weights: each class weighs 1 in total, split equally over its CV groups, equally over the clips of a group.
Layered clips (a real bed + one code-synthesized layer, label 0) belong to two groups: the layer's and the bed's. They
are never test clips of the clip-level schemes, and a model never trains on one whose layer group or bed group is held
out.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'out', 'realness')
FEATS = os.path.join(OUT, 'feats')
SIG_INPUTS = ['silence_frac', 'floor_db', 'spec_floor_db', 'log_flatness', 'peakiness_db', 'line_share', 'bandwidth_khz',
              'quiet_flat', 'tonal_frames', 'pitch_wobble', 'env_resid_db', 'texture_kurt', 'crest_db']
# physical direction of each measurement (+1: larger reads as a recording): the signal head's coefficients are
# constrained to these signs, so a measurement can never vote against its physics because of a dataset quirk
SIGNS = {'silence_frac': -1, 'floor_db': 1, 'spec_floor_db': 1, 'log_flatness': 1, 'peakiness_db': -1, 'line_share': -1,
         'bandwidth_khz': 1, 'quiet_flat': 1, 'tonal_frames': -1, 'pitch_wobble': 1, 'env_resid_db': 1, 'texture_kurt': 1,
         'crest_db': 1}
C_CLAP = (0.001, 0.003, 0.01)
C_SIGNAL = 0.1
C_AST = 0.01
AUG_WEIGHT = 0.35


class Data:
    """clips with features, aligned arrays; .sub(mask) for a subset"""

    def __init__(self, clips, clap, sig, ast=None):
        self.clips = clips
        self.ids = np.array([c['id'] for c in clips])
        self.y = np.array([c['label'] for c in clips])
        self.family = np.array([c['family'] for c in clips])
        self.group = np.array([c['group'] for c in clips])
        self.type = np.array([c['type'] for c in clips])
        self.src = np.array([c['src'] for c in clips])
        self.perturb = np.array([c.get('perturb', '') for c in clips])
        self.augment = np.array([bool(c.get('augment')) for c in clips])
        self.base = np.array([c.get('base', c['id']) for c in clips])
        self.is_layer = self.perturb == 'layer'
        self.bed_group = np.array([c.get('bed_group', c['group']) for c in clips])
        self.bed_src = np.array([c.get('bed_src', c['src']) for c in clips])
        self.Xc = np.stack([clap[i] for i in self.ids]) if len(clips) else np.zeros((0, 512))
        self.Xs = np.array([[np.nan if sig[i] is None else sig[i].get(k, np.nan) for k in SIG_INPUTS] for i in self.ids], float)
        self.Xa = np.stack([ast[i] for i in self.ids]) if ast is not None and len(clips) else None

    def sub(self, mask):
        d = object.__new__(Data)
        for k, v in self.__dict__.items():
            d.__dict__[k] = (v[mask] if isinstance(v, np.ndarray) else [c for c, m in zip(v, mask) if m]) if v is not None else None
        return d

    def __len__(self):
        return len(self.ids)


def load(require_ast=False, robust=True):
    man = json.load(open(os.path.join(OUT, 'manifest.json')))
    clips = [c for k, part in man.items() for c in part if robust or k != 'robust']
    # a few recordings sit in two asset folders (e.g. foley and toy): keep them in the first group only, so no
    # recording is ever on both sides of a held-out split
    owner, seen, drop = {}, set(), set()
    for c in clips:
        if c['label'] == 1 and 'perturb' not in c:
            key = os.path.basename(c['src']).split('_')[0]
            if owner.setdefault(key, c['group']) != c['group'] or c['id'] in seen:
                drop.add(c['id'])
            seen.add(c['id'])
    clips = [c for c in clips if c['id'] not in drop and c.get('base') not in drop]
    sig = json.load(open(os.path.join(FEATS, 'signal.json')))
    z = np.load(os.path.join(FEATS, 'clap.npz'), allow_pickle=False)
    clap = dict(zip(z['ids'], z['emb']))
    ast = None
    if require_ast:
        za = np.load(os.path.join(FEATS, 'ast.npz'), allow_pickle=False)
        ast = dict(zip(za['ids'], za['emb']))
    keep = [c for c in clips if c['id'] in clap and c['id'] in sig and sig[c['id']] is not None and (ast is None or c['id'] in ast)]
    return Data(keep, clap, sig, ast)


def weights(d, groups, aug_weight=AUG_WEIGHT):
    """each class weighs 1, split equally over its groups; inside a group an augmented or layered clip weighs
    aug_weight of a clean clip"""
    w = np.zeros(len(d))
    u = np.where(d.augment | d.is_layer, aug_weight, 1.0)
    for lab in (0, 1):
        gs = sorted(set(groups[d.y == lab]))
        for g in gs:
            m = (d.y == lab) & (groups == g)
            w[m] = u[m] / (len(gs) * u[m].sum())
    return w * len(d) / w.sum()


def logreg(X, y, w, C, signs=None):
    """weighted L2 logistic regression (sklearn's objective: C * sum w*logloss + |b|^2 / 2, free intercept);
    signs (+1 / -1 / 0 per column) bound coefficients to that sign (0 = free)"""
    n, k = X.shape
    s = 2 * y - 1.0

    def f(b):
        m = s * (X @ b[:k] + b[k])
        lo = np.logaddexp(0, -m)
        g = -s * w / (1 + np.exp(np.clip(m, -50, 50)))
        return C * np.sum(w * lo) + 0.5 * b[:k] @ b[:k], np.r_[C * (X.T @ g) + b[:k], C * g.sum()]
    bounds = None
    if signs is not None:
        bounds = [(0, None) if v > 0 else (None, 0) if v < 0 else (None, None) for v in signs] + [(None, None)]
    r = minimize(f, np.zeros(k + 1), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 3000})
    return r.x[:k], float(r.x[k])


def fit_head(X, y, w, C, signs=None):
    mean = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, mean)
    lo, hi = np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0)
    X = np.clip(X, lo, hi)
    scale = X.std(axis=0) + 1e-6
    coef, b0 = logreg((X - mean) / scale, y, w, C, signs)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'clip': [lo.tolist(), hi.tolist()],
            'coef': coef.tolist(), 'intercept': b0}


def head_logits(h, X):
    mean = np.asarray(h['mean'])
    X = np.where(np.isfinite(X), X, mean)
    X = np.clip(X, np.asarray(h['clip'][0]), np.asarray(h['clip'][1]))
    return ((X - mean) / np.asarray(h['scale'])) @ np.asarray(h['coef']) + h['intercept']


def head_signs(name):
    return [SIGNS[k] for k in SIG_INPUTS] if name == 'signal' else None


def head_inputs(d, name):
    return {'clap': d.Xc, 'signal': d.Xs, 'ast': None if d.Xa is None else np.log(np.clip(d.Xa, 1e-4, 1))}[name]


def inner_folds(groups, k=5):
    """split the distinct groups into k folds (deterministic, round-robin over sorted names)"""
    gs = sorted(set(groups))
    return [np.isin(groups, gs[i::k]) for i in range(min(k, len(gs)))]


def fit_model(d, groups, heads=('clap', 'signal'), c_clap=None, inner=5, aug_weight=AUG_WEIGHT):
    """heads + fusion on data d. c_clap: fixed C for the CLAP head, else chosen by inner grouped CV (log loss on
    clean clips: augmented copies help training but are not what the model is scored on)."""
    w = weights(d, groups, aug_weight)
    clean = ~d.augment & ~d.is_layer
    folds = inner_folds(groups, inner)
    cs = {'signal': [C_SIGNAL], 'ast': [C_AST], 'clap': [c_clap] if c_clap else list(C_CLAP)}
    oof = {}
    chosen = {}
    for name in heads:
        X = head_inputs(d, name)
        best = None
        for C in cs[name]:
            z = np.zeros(len(d))
            for te in folds:
                tr = ~te
                if len(set(d.y[tr])) < 2:
                    z[te] = 0.0
                    continue
                z[te] = head_logits(fit_head(X[tr], d.y[tr], w[tr], C, head_signs(name)), X[te])
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            ll = -np.sum((w * (d.y * np.log(p + 1e-9) + (1 - d.y) * np.log(1 - p + 1e-9)))[clean]) / w[clean].sum()
            if best is None or ll < best[0]:
                best = (ll, C, z)
        chosen[name], oof[name] = best[1], best[2]
    F = np.stack([oof[h] for h in heads], axis=1)
    coef, b0 = logreg(F[clean], d.y[clean], w[clean], 1.0, [1] * len(heads))
    model = {'heads': {}, 'fusion': {'heads': list(heads), 'coef': coef.tolist(), 'intercept': b0}, 'C': chosen}
    for name in heads:
        model['heads'][name] = fit_head(head_inputs(d, name), d.y, w, chosen[name], head_signs(name))
    return model


def predict(model, d, return_heads=False):
    fu = model['fusion']
    L = {h: head_logits(model['heads'][h], head_inputs(d, h)) for h in fu['heads']}
    s = fu['intercept'] + sum(c * L[h] for h, c in zip(fu['heads'], fu['coef']))
    p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
    return (p, L) if return_heads else p


def logo(d, groups, train_mask=None, augment=True, layers=False, bed_groups=None, **kw):
    """leave-one-group-out predictions: every clip is scored by a model that never saw its group (nor augmented
    copies of its clips). train_mask limits which clips may ever be used for training (e.g. never the retired score);
    augment=False trains on clean clips only; layers=True also trains on layered clips whose layer group and bed group
    (bed_groups) are both outside the held-out group. Returns p (n,) and head logits (augmented and layered clips: NaN)."""
    p = np.full(len(d), np.nan)
    L = {}
    tm = np.ones(len(d), bool) if train_mask is None else train_mask
    bg = groups if bed_groups is None else bed_groups
    usable = (d.perturb == '') | (d.augment & augment) | (d.is_layer & layers)
    for g in sorted(set(groups[~d.is_layer])):
        te = (groups == g) & ~d.augment & ~d.is_layer
        tr = (groups != g) & (bg != g) & tm & usable
        sub = d.sub(tr)
        m = fit_model(sub, groups[tr], **kw)
        pt, Lt = predict(m, d.sub(te), True)
        p[te] = pt
        for h, v in Lt.items():
            L.setdefault(h, np.full(len(d), np.nan))[te] = v
    return p, L
