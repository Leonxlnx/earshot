"""earshot forge: an agent makes its own sounds in code and improves them by listening with earshot.

    earshot forge keypress --refs kit/keyboard --budget 160 --keep 4 --dir made/keypress
    earshot forge whoosh --target "a fast crisp air whoosh" --alts "a wind gust" "a sword swing" --refs kit/whoosh

Loop: a recipe (earshot/forge/recipes/<name>.py: physical synthesis written in code, no recorded audio) renders
candidates from parameters; the Critic scores each one (description match z, real-or-synthetic p_real, closeness to
real reference recordings, measured shape); an evolutionary search keeps the best and mutates them; repeat. The agent's
outer loop reads the critic's words ("sounds like: a wooden knock", "decay +2.4 outside the refs") and rewrites the
recipe itself. Output: the best diverse sounds as WAVs + report.json (scores, parameters, the whole history) + the same
critic's scores of the real references (leave-one-out, so a reference is not its own neighbour) and their held-out
AudioSet tags, so the gap to real is always visible.

Honest limits: the critic's models can be pleased without the sound getting better (Goodhart). earshot's realness model
is easy to fool with modal resonators, a room and a mic noise floor, so a high p_real proves little. Worse, once a
search has tuned against them, every model check saturates: the forge test's made sounds land inside or above the real
recordings on held-out phrasings, the AudioSet tagger and independent real sets too, while a plain 1/3-octave
measurement still finds them 8-13 dB off the real balance. So the critic always reports the model-free tone check
(earshot/forge/tone.py: tonal balance against the refs, the bands outside every ref, the tail), and `tone=W` (CLI
--tone W, 1.5 is a good start) puts it into the objective. Judge finished sounds by those measurements, robust
re-renders (`robust=N`), spectrograms and, last, human ears. AudioSet tags are a held-out check only while nobody picks
or steers by them: once they decide which candidate is kept, they are part of the objective.
"""
import importlib
import json
import os
import pkgutil

import numpy as np

from .. import audio
from ..report import emit


def recipes():
    pkg = importlib.import_module(f'{__name__}.recipes')   # not `from . import recipes`: this function shadows it
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.name.startswith('_'))


def load_recipe(name):
    return importlib.import_module(f'earshot.forge.recipes.{name}')


def forge(recipe, target=None, alts=None, refs=None, budget=160, pop=16, keep=4, seed=0, out=None, log=print,
          init=None, holdout=True, robust=0, space=None, sigma0=0.22, sigma1=0.04, critic=None, tone=0.0,
          tone_target=None, reps=1):
    """run the loop for one recipe; writes WAVs + report.json into `out` when given; returns the report dict.
    robust=N re-renders every pick on N new seeds (mean / min / pass rate in pick['robust']); space= a narrower
    {name: spec} family; sigma0/sigma1 the first / last mutation step; critic= a prebuilt Critic (reuse one across
    several forge() calls in a round script, or pass one with an `extra` check); tone= the weight of the model-free
    tonal-balance penalty (Critic tone=, tone_target=); reps= renders per candidate inside the search (ranked by
    their mean, against lucky single renders)"""
    import soundfile as sf
    from .critic import Critic
    from .search import diverse, optimize, render
    from .search import robust as robust_stats
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    name = rec.__name__.rsplit('.', 1)[-1]
    if critic is None:
        target = target or rec.TARGET
        alts = list(alts if alts is not None else getattr(rec, 'ALTS', []))
        critic = Critic(target, alts, refs, tone=tone, tone_target=tone_target)
    else:
        target, alts = critic.target, list(critic.alts)
    res = optimize(rec, critic, budget=budget, pop=pop, seed=seed, init=init, log=log, space=space, sigma0=sigma0,
                   sigma1=sigma1, reps=reps)
    picks = diverse(res['best'] + sorted(res['history'], key=lambda h: -h['score']['total'])[:64], rec, keep)
    ref_scores = critic.score_refs(loo=True) if critic.refs else []
    report = {'recipe': name, 'target': target, 'alts': alts, 'refs': critic.refs, 'budget': budget, 'seed': seed,
              'tone_weight': getattr(critic, 'tone', 0.0), 'tone_target': getattr(critic, 'tone_target', None),
              'reps': reps,
              'generations': res['generations'], 'picks': [], 'ref_scores': [], 'ref_scores_loo': True}
    if space is not None:
        report['space'] = space
    xs = [render(rec, h['params'], h['seed']) for h in picks]
    hold = critic.holdout([x for x in xs if x is not None]) if holdout and xs else []
    ref_hold = critic.holdout([audio.load(p) for p in critic.refs]) if holdout and critic.refs else []
    if out:
        os.makedirs(out, exist_ok=True)
    for i, (h, x) in enumerate(zip(picks, xs)):
        item = {'name': f'{name}_{i + 1:02d}', 'params': h['params'], 'seed': h['seed'],
                'score': {k: v for k, v in h['score'].items() if k != 'i'}}
        if i < len(hold):
            item['holdout'] = hold[i]
        if robust:
            item['robust'] = robust_stats(rec, critic, h['params'], robust)
        if out and x is not None:
            item['file'] = os.path.join(out, item['name'] + '.wav').replace(chr(92), '/')
            sf.write(item['file'], x, 48000, subtype='PCM_24')
        report['picks'].append(item)
    for i, (p, s) in enumerate(zip(critic.refs, ref_scores)):
        report['ref_scores'].append({'file': os.path.basename(p), **{k: s.get(k) for k in
                                     ('total', 'passes', 'match_z', 'alt_margin', 'alt_zs', 'p_real', 'ref_typ_z',
                                      'feat_pen', 'tone_db', 'tone_ratio', 'sounds_like')}})
        if i < len(ref_hold):
            report['ref_scores'][-1]['holdout'] = ref_hold[i]
    best_ref = max((r['total'] for r in report['ref_scores'] if r['total'] is not None and np.isfinite(r['total'])),
                   default=None)
    n_pass = sum(1 for p in report['picks'] if p['score'].get('passes'))
    top = report['picks'][0]['score'] if report['picks'] else {}
    report['summary'] = (f"forge {name}: '{target}' - {n_pass} of {len(report['picks'])} kept sounds pass "
                         f"(best total {top.get('total')}, match z {top.get('match_z')}, p_real {top.get('p_real')})"
                         + (f"; the real references score up to {best_ref:.2f} with the same critic" if best_ref is not None
                            else '') + ' (references scored leave-one-out).')
    report['history'] = [{'gen': h['gen'], 'params': h['params'], 'seed': h['seed'],
                          'score': {k: h['score'].get(k) for k in ('total', 'passes', 'match_z', 'alt_margin', 'p_real',
                                                                    'ref_typ_z', 'feat_pen', 'tone_db', 'tone_ratio',
                                                                    'sounds_like')}}
                         for h in res['history']]
    if out:
        with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return report


def add_cli(sub):
    from ..cli import common
    p = common(sub.add_parser('forge', help='make sounds in code and improve them by listening (recipe + critic + search)'))
    p.add_argument('recipe', nargs='?', help='recipe name (earshot/forge/recipes/); omit to list them')
    p.add_argument('--target', help="description to aim for (default: the recipe's)")
    p.add_argument('--alts', nargs='*', help='confusions it must sound less like')
    p.add_argument('--refs', nargs='*', help='real reference recordings (files / folders) to compare with')
    p.add_argument('--budget', type=int, default=160, help='candidates to render and score (default 160)')
    p.add_argument('--pop', type=int, default=16)
    p.add_argument('--keep', type=int, default=4, help='diverse best sounds to write')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dir', dest='outdir', help='folder for the WAVs + report.json')
    p.add_argument('--robust', type=int, default=0, help='re-render every kept sound on N new seeds and report the mean')
    p.add_argument('--tone', type=float, default=0.0,
                   help='weight of the model-free tonal-balance penalty vs the refs (0 = report only; 1.5 is a good start)')
    p.add_argument('--reps', type=int, default=1,
                   help='renders per candidate inside the search, ranked by their mean (2 = half as lucky, twice as slow)')
    p.add_argument('--tone-target', type=float, default=None,
                   help="dB distance from the refs' median balance that counts as typical (default: the refs' own spread)")

    def run(a):
        if not a.recipe:
            print('recipes: ' + ', '.join(recipes()))
            return 0
        r = forge(a.recipe, a.target, a.alts, a.refs, a.budget, a.pop, a.keep, a.seed, a.outdir, robust=a.robust,
                  tone=a.tone, tone_target=a.tone_target, reps=a.reps)
        lines = [r['summary']] + [f"  {p['name']}: {p['score'].get('summary')}" for p in r['picks']]
        emit({k: v for k, v in r.items() if k != 'history'}, a, text='\n'.join(lines))
        return 0
    p.set_defaults(func=run)
