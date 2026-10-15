"""The listen-and-improve loop: an evolutionary search over a recipe's parameters, scored by the Critic.

    from earshot.forge.search import optimize
    res = optimize(recipe, critic, budget=160, pop=16, seed=0)
    res['best'][0]['params'], res['best'][0]['score']['summary'], res['history']

A recipe is a module (or object) with
    SPACE    {name: (lo, hi)} | {name: (lo, hi, 'log')} | {name: (lo, hi, 'int')} | {name: [choice, ...]}
    DEFAULTS {name: value}            (a sensible starting sound; optional)
    render(params, r) -> np.ndarray   (48 kHz mono or stereo float; r = numpy Generator for natural variation)
Each generation renders `pop` candidates (elites mutated with a shrinking step, plus a few fresh random ones), peak-
normalises them to -1 dBFS (so level never decides), scores them in one batch, and keeps the best. Deterministic for a
given seed. Every candidate is kept in the history, so an agent can read what helped.

`space=` restricts the search to a family (a narrower {name: spec} dict; parameters it leaves out keep the recipe's
DEFAULTS). `robust()` re-renders a result on new noise seeds: the best single render of a search is a winner's curse
(the forge makers measured robust means 0.1-2.2 below the reported best), so judge finals by their robust mean.
`reps=N` fights that inside the search: every candidate is rendered on N seeds and ranked by its mean total (it passes
when at least half of its renders pass); the row kept is the first render's, with 'total_reps' listing all N.
"""
import math
import sys
import traceback

import numpy as np

from . import dsp


def _decode(space, u):
    p = {}
    for (k, spec), v in zip(space.items(), u):
        v = float(np.clip(v, 0.0, 1.0))
        if isinstance(spec, list):
            p[k] = spec[min(len(spec) - 1, int(v * len(spec)))]
        else:
            lo, hi = spec[0], spec[1]
            kind = spec[2] if len(spec) > 2 else 'lin'
            if kind == 'log':
                p[k] = float(math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo))))
            elif kind == 'int':
                p[k] = int(round(lo + v * (hi - lo)))
            else:
                p[k] = float(lo + v * (hi - lo))
    return p


def _encode(space, params):
    u = []
    for k, spec in space.items():
        v = params.get(k)
        if v is None:
            u.append(0.5)
        elif isinstance(spec, list):
            u.append((spec.index(v) + 0.5) / len(spec) if v in spec else 0.5)
        else:
            lo, hi = spec[0], spec[1]
            kind = spec[2] if len(spec) > 2 else 'lin'
            if kind == 'log':
                u.append((math.log(max(v, 1e-12)) - math.log(lo)) / (math.log(hi) - math.log(lo)))
            else:
                u.append((v - lo) / (hi - lo))
    return np.clip(np.array(u, float), 0, 1)


_failed = set()


def render(recipe, params, seed):
    """render one candidate, peak-normalised to -1 dBFS; None if it fails or is silent. The first exception of each
    recipe is printed to stderr (a recipe bug must not look like 'render failed or silent' candidates)."""
    try:
        x = np.asarray(recipe.render(dict(params), dsp.rng(seed)), np.float64)
    except Exception as e:
        name = getattr(recipe, '__name__', type(recipe).__name__)
        if name not in _failed:
            _failed.add(name)
            tb = traceback.extract_tb(e.__traceback__)[-1]
            sys.stderr.write(f'forge: recipe {name} raised {type(e).__name__}: {e} ({tb.filename}:{tb.lineno}); '
                             'later failures of this recipe are not printed' + chr(10))
        return None
    if not len(x) or not np.isfinite(x).all() or np.abs(x).max() < 1e-7:
        return None
    return dsp.normalize(x, -1.0).astype(np.float32)


def optimize(recipe, critic, budget=160, pop=16, elite=4, seed=0, init=None, sigma0=0.22, sigma1=0.04,
             fresh=0.15, log=None, space=None, reps=1):
    """returns {'best': [top results], 'history': [every result], 'generations': n}; a result = {params, seed, score}.
    space: optional narrower {name: spec} (a family); parameters it leaves out keep the recipe's DEFAULTS.
    reps: renders per candidate (ranked by the mean total; budget counts candidates, so reps=2 costs twice the time)"""
    base = {}
    if space is None:
        space = recipe.SPACE
    else:
        base = dict(getattr(recipe, 'DEFAULTS', {}) or {})
    R = np.random.default_rng(seed)
    d = len(space)
    starts = [dict(getattr(recipe, 'DEFAULTS', {}) or {})] + [dict(p) for p in (init or [])]
    gens = max(1, budget // pop)
    history, elites = [], []
    for g in range(gens):
        sig = sigma0 * (sigma1 / sigma0) ** (g / max(1, gens - 1))
        U, seeds = [], []
        for j in range(pop):
            if g == 0:
                u = _encode(space, starts[j]) if j < len(starts) else R.uniform(0, 1, d)
            elif R.uniform() < fresh or not elites:
                u = R.uniform(0, 1, d)
            else:
                parent = elites[min(len(elites) - 1, int(R.integers(0, max(1, min(elite, len(elites))))))]
                u = np.clip(parent['u'] + R.normal(0, sig, d), 0, 1)
            U.append(u)
            seeds.append(int(R.integers(0, 2 ** 31)))
        params = [{**base, **_decode(space, u)} for u in U]
        jobs = [(j, s + k * 7919) for j, s in enumerate(seeds) for k in range(max(1, reps))]
        xs = [render(recipe, params[j], s) for j, s in jobs]
        live = [i for i, x in enumerate(xs) if x is not None]
        scores = critic.score([xs[i] for i in live]) if live else []
        got = {}
        for i, sc in zip(live, scores):
            got.setdefault(jobs[i][0], []).append(sc)
        by = {}
        for j, rows in got.items():
            if len(rows) < max(1, reps):          # a failed render among the reps: the candidate is unreliable
                continue
            if reps > 1:
                tot = [float(r['total']) for r in rows]
                row = dict(rows[0])
                row['total_reps'] = [round(t, 3) for t in tot]
                row['total'] = round(float(np.mean(tot)), 3) if all(np.isfinite(tot)) else -np.inf
                row['passes'] = sum(1 for r in rows if r.get('passes')) * 2 >= len(rows)
                by[j] = row
            else:
                by[j] = rows[0]
        for j in range(pop):
            sc = by.get(j, {'total': -np.inf, 'passes': False, 'summary': 'render failed or silent'})
            history.append({'gen': g, 'u': U[j], 'params': params[j], 'seed': seeds[j], 'score': sc})
        ranked = sorted([h for h in history if np.isfinite(h['score']['total'])], key=lambda h: -h['score']['total'])
        elites = ranked[:elite]
        if log and elites:
            b = elites[0]['score']
            log(f"gen {g + 1}/{gens}: best total {b['total']:.2f} ({'PASS' if b['passes'] else 'no pass'}) "
                f"z {b.get('match_z')} p_real {b.get('p_real')} typ {b.get('ref_typ_z')} pen {b.get('feat_pen')}")
    ranked = sorted([h for h in history if np.isfinite(h['score']['total'])], key=lambda h: -h['score']['total'])
    return {'best': ranked[:max(elite, 8)], 'history': history, 'generations': gens}


def robust(recipe, critic, params, seeds=4, seed0=5000):
    """re-render `params` on `seeds` new noise seeds (seed0, seed0+1, ...) and score them: {mean, min, max, sd,
    pass_rate, n, totals}. A sound that only scores well on one seed is a lucky render, not a good recipe."""
    ss = list(range(seed0, seed0 + seeds)) if isinstance(seeds, int) else list(seeds)
    xs = [render(recipe, params, s) for s in ss]
    live = [x for x in xs if x is not None]
    rows = critic.score(live) if live else []
    fin = [float(r['total']) for r in rows if np.isfinite(r['total'])]
    return {'n': len(ss), 'seeds': ss, 'totals': [round(t, 3) for t in fin], 'failed': len(ss) - len(fin),
            'mean': round(float(np.mean(fin)), 3) if fin else None, 'min': round(float(min(fin)), 3) if fin else None,
            'max': round(float(max(fin)), 3) if fin else None, 'sd': round(float(np.std(fin)), 3) if fin else None,
            'pass_rate': round(sum(1 for r in rows if r.get('passes')) / len(ss), 3)}


def diverse(results, recipe, k=4, min_dist=0.08):
    """pick up to k strong results whose parameter vectors differ (normalised distance >= min_dist), passing first"""
    out = []
    for h in sorted(results, key=lambda h: (not h['score'].get('passes'), -h['score']['total'])):
        if all(np.linalg.norm(h['u'] - o['u']) / math.sqrt(len(h['u'])) >= min_dist for o in out):
            out.append(h)
        if len(out) == k:
            break
    return out
