"""Fit the final realness model on every group and write earshot/data/realness_model.json (plain JSON, no pickle).

    OMP_NUM_THREADS=2 EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/train.py

Uses all clean clips plus their augmented training copies (common.fit_model: C of the CLAP head and the fusion weights
from inner grouped CV), grouped like run_eval scheme M (each source weighs the same; music = four providers). Verdict
thresholds come from the held-out scheme-M predictions (out/realness/oof/M.npz, made by run_eval.py --stage cv):
hi / lo = the p_real above / below which >= 92.5% (run_eval.TARGET) of held-out clips were real / synthesized; the
file thresholds (for the mean of a long file's windows) come from the same predictions averaged per long source
(run_eval.file_thresholds). Layered clips (part 'layers') are not trained on: run_eval.py --stage layers tested it.
Also stores which vocabulary descriptions the CLAP head's direction points to (an explanation of what it learned).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import run_eval as R  # noqa: E402
from earshot import config, realness  # noqa: E402


def rounded(h, nd=6):
    return {k: ([round(float(v), nd) for v in x] if isinstance(x, list) and x and not isinstance(x[0], list) else
                [[round(float(v), nd) for v in r] for r in x] if isinstance(x, list) and x and isinstance(x[0], list) else
                round(float(x), nd) if isinstance(x, float) else x) for k, x in h.items()}


def clap_direction(head, k=8):
    """vocabulary terms most aligned with the CLAP head's real (+) and synth (-) directions"""
    from earshot.models import clap
    voc = json.load(open(os.path.join(config.DATA_DIR, 'vocab.json')))
    terms = sorted({t for v in voc['categories'].values() for t in v} | set(realness.REAL_PROMPTS + realness.SYNTH_PROMPTS))
    T = clap.get().embed_text(terms)
    w = np.asarray(head['coef']) / np.asarray(head['scale'])
    s = T @ (w / np.linalg.norm(w))
    o = np.argsort(s)
    return {'toward_real': [terms[i] for i in o[::-1][:k]], 'toward_synth': [terms[i] for i in o[:k]],
            'how': 'cosine between each vocabulary text embedding and the CLAP head weight direction (coef / scale)'}


def facts(d, min_prec=0.9, min_support=25):
    """per measurement and side, the loosest threshold past which >= min_prec of the clean training clips belong to
    that side (support >= min_support clips): the human-readable signal facts realness() reports. Exact digital
    silence is never a fact: edited / gated library recordings have it as often as synthesizers (see silence_base)"""
    out = []
    for j, k in enumerate(C.SIG_INPUTS):
        if k == 'silence_frac':
            continue
        v = d.Xs[:, j]
        ok = np.isfinite(v)
        v, y = v[ok], d.y[ok]
        for side in ('synth', 'real'):
            toward_real_is_big = C.SIGNS[k] > 0
            big = (side == 'real') == toward_real_is_big
            best = None
            for t in np.unique(np.percentile(v, np.arange(1, 50) if not big else np.arange(51, 100))):
                m = v >= t if big else v <= t
                prec = float(np.mean(y[m] == (side == 'real'))) if m.any() else 0.0
                if m.sum() >= min_support and prec >= min_prec and (best is None or m.sum() > best[2]):
                    best = (float(t), prec, int(m.sum()))
            if best:
                out.append({'feature': k, 'side': side, 'op': '>=' if big else '<=', 'value': round(best[0], 4),
                            'precision': round(best[1], 3), 'support': best[2]})
    return out


def silence_base(d, levels=(0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """share of clean training clips per class with at least this fraction of exact digital silence"""
    v = d.Xs[:, C.SIG_INPUTS.index('silence_frac')]
    return [{'at_least': t, 'real': round(float(np.mean(v[d.y == 1] >= t)), 3),
             'synth': round(float(np.mean(v[d.y == 0] >= t)), 3)} for t in levels]


def main():
    d = C.load()
    train = (d.perturb == '') | d.augment
    dt = d.sub(train)
    g = R.groups_of(dt, 'music_src')
    m = C.fit_model(dt, g)
    clean = d.perturb == ''
    oof = R.load_oof('M', d)
    if oof is None:
        raise SystemExit('run run_eval.py --stage cv first (thresholds come from held-out predictions)')
    hi, lo = R.thresholds(d.y[clean], oof[clean])
    _, fy, fp, _ = R.file_level(d, oof)
    fth = R.file_thresholds(fy, fp, hi, lo)
    heads = {'clap': dict(rounded(m['heads']['clap']), inputs='clap_embedding_512'),
             'signal': dict(rounded(m['heads']['signal']), inputs=list(C.SIG_INPUTS))}
    ids = sorted(dt.ids)
    res_path = os.path.join(HERE, 'results.json')
    ev = {}
    if os.path.exists(res_path):
        r = json.load(open(res_path))
        ev = {'scheme_A': r['schemes'].get('A'), 'scheme_M': r['schemes'].get('M'), 'band': r.get('band'),
              'retired_check': r.get('retired_check'), 'layers': r.get('layers', {}).get('summary'),
              'file_level_M': {k: r.get('file_level_M', {}).get(k) for k in ('n', 'auc', 'acc@0.5', 'band')},
              'see': 'eval/realness/RESULTS.md'}
    model = {
        'version': 1,
        'trained': hashlib.sha1('|'.join(ids).encode()).hexdigest()[:12],
        'about': 'p_real = sigmoid(fusion over head logits); heads are standardised logistic regressions. '
                 'clap head: laion/larger_clap_general audio embedding (repeat-padded, one window per clip <= 6 s, else '
                 '4 s windows). signal head: earshot.realness.signal_features, coefficients sign-constrained to their '
                 'physical direction. Positives: real recordings (Freesound/Mixkit), pro launch-film soundtracks, pro '
                 'music; negatives: code-synthesized audio (cinetic synth engine, a retired synthesized film score, '
                 'generic numpy generators).',
        'data': {'clips': int(len(dt)), 'clean': int((~dt.augment).sum()), 'augmented': int(dt.augment.sum()),
                 'real': int(dt.y[~dt.augment].sum()), 'synth': int((1 - dt.y[~dt.augment]).sum()),
                 'groups': sorted(set(g))},
        'long_s': realness.LONG_S, 'win_s': realness.WIN_S, 'silent_db': realness.SILENT_DB,
        'heads': heads,
        'fusion': {'heads': m['fusion']['heads'], 'coef': [round(c, 6) for c in m['fusion']['coef']],
                   'intercept': round(m['fusion']['intercept'], 6)},
        'C': m['C'],
        'thresholds': {'hi': hi, 'lo': lo},
        'file_thresholds': fth,
        'prompts': {'real': realness.REAL_PROMPTS, 'synth': realness.SYNTH_PROMPTS},
        'facts': facts(d.sub(clean)),
        'silence_base': silence_base(d.sub(clean)),
        'clap_direction': clap_direction(m['heads']['clap']),
        'eval': ev,
    }
    with open(realness.MODEL_PATH, 'w', encoding='utf-8') as fh:
        json.dump(model, fh, indent=1)
    print(f"wrote {realness.MODEL_PATH}: {len(dt)} clips, C {m['C']}, fusion {model['fusion']}, thresholds lo {lo} hi {hi}, "
          f"file thresholds lo {fth['lo']} hi {fth['hi']}")
    print('signal coef:', dict(zip(C.SIG_INPUTS, heads['signal']['coef'])))
    print('clap direction:', model['clap_direction'])
    for q in model['facts']:
        print('fact:', q)
    print('silence base rates:', model['silence_base'])


if __name__ == '__main__':
    main()
