"""earshot forge: an agent makes its own sounds in code and improves them by listening with earshot.

    earshot forge keypress --refset typing_key --variants 4 --dir made/keypress          # ES + pre-screen (CLI)
    earshot forge whoosh --target "a fast crisp air whoosh" --refs kit/whoosh --strategy cma --prescreen 64
    earshot forge --list                                       # recipes with their TARGET (+ refset categories)
    earshot forge --render keypress --params made/keypress/report.json --seeds 1 2 3 --dir made/again
    earshot forge --check out/forge_v2/*/final                # re-render every pick from report.json, compare WAVs

Loop: a recipe (earshot/forge/recipes/<name>.py: physical synthesis written in code, no recorded audio) renders
candidates from parameters; a model-free pre-screen ranks many renders against the real references (tonal balance,
envelope, artifacts: earshot.forge.search.Prescreen) and the Critic scores the best (description match z,
real-or-synthetic p_real, closeness to real reference recordings, measured shape, `tone=` the model-free balance); a
search (v1 evolution strategy, or separable CMA-ES with restarts) keeps the best and moves on; repeat. The agent's
outer loop reads the critic's words ("sounds like: a wooden knock", "decay +2.4 outside the refs") and rewrites the
recipe itself.

Output folder (`out=` / --dir): the best diverse sounds as WAVs; report.json (scores, parameters, per-generation stats,
timing, the whole history, the same critic's scores of the real references leave-one-out and their held-out AudioSet
tags); report.md (what was asked, budget, per-generation best, the critic's words for each pick, the measured gap to
the references); compare.png (each pick's spectrogram, envelope and 1/3-octave balance next to its nearest reference:
earshot.forge.viz); audition.wav + audition.txt (made, ref, made, ref ... peak-matched, to listen in pairs).
Every pick of report.json (forge() and palette(), schema 2 since round 6) carries a `render` block: recipe module,
render function, family member, the complete params, seed, critic mode, peak / gain (earshot.forge.schema), so each
WAV re-renders from its report alone (`--check`, schema.check_final; older reports: schema.legacy_block / upgrade).
The seed bank (earshot/forge/seeds/<recipe>.json) keeps good parameter sets; `bank=True` (CLI default) starts the
search from them and adds the new picks. `shortlist=S` with `robust=N` re-renders S x keep diverse candidates on N new
seeds and keeps the `keep` best robust means (the best single render of a search is a winner's curse).

Families (a category's palette of characters, round 3): `forge(rec, families={name: params | {'init', 'space',
'budget'}}, ...)` searches each member inside its own box (`space`, else family_box(): +-box of each numeric range
around the preset, choices fixed) with keep=1, then palette() scores the exact member set together (no diverse()
drop: a family's members may sound alike on purpose), reports the family variety (mean / max pairwise CLAP cosine of
the members next to the refs' own, and the closest pair's 1/3-octave distance), loudness-matches the palette (every
member to the quietest momentary-max LUFS, then one common gain so the loudest peak sits at -1 dBFS: swappable in an
edit without re-levelling) and writes <category>_NN.wav, report.json / report.md and _audition.wav (members at their
matched levels, each followed by its nearest ref; the leading underscore keeps bench.py from counting it).
`palette(rec, members, critic, out=...)` does the second half alone for members chosen any other way.

Python API defaults reproduce v1 (strategy 'es', no pre-screen, no bank), so earlier round scripts stay comparable;
the CLI defaults to the v2 setup (es + prescreen 3 x pop, bank, robust 3, shortlist 2, critic v2 with its own weights,
and when neither --refs nor --refset is given the recipe's REFSET, else its REFS minus held-out files). The critic is
v2 (earshot/forge/critic.py): the v1 model terms minus model-free tone, envelope, artifact and polish penalties; the
pre-screen ranks with the same tone / polish measurements, so CLAP only hears the best `pop` of each generation. Measured on keypress at 96 scored candidates: the pre-screen raised the
passing candidates from 10.5 to 36.5 and the top-5 mean total from 5.77 to 6.17; CMA-ES did not beat the v1 ES at
96-240 candidates (earshot/forge/search.py has the numbers).

Honest limits: the critic's models can be pleased without the sound getting better (Goodhart). earshot's realness model
is easy to fool with modal resonators, a room and a mic noise floor, so a high p_real proves little. Worse, once a
search has tuned against them, every model check saturates: the forge test's made sounds land inside or above the real
recordings on held-out phrasings, the AudioSet tagger and independent real sets too, while a plain 1/3-octave
measurement still finds them 8-13 dB off the real balance. So the critic always reports the model-free tone check
(earshot/forge/tone.py), the pre-screen ranks by it, and critic v2 always puts it into the objective (weight 1.0;
`tone=W` / --tone W changes it, --tone 0 makes it report-only). Judge finished sounds by those measurements, robust
re-renders, compare.png, the held-out benchmark (eval/forge/bench.py) and, last, human ears (audition.wav). AudioSet
tags are a held-out check only while nobody picks or steers by them.
"""
import importlib
import json
import os
import pkgutil
import time

import numpy as np

from .. import audio
from ..report import emit
from . import recipes as _recipes_pkg   # first import binds earshot.forge.recipes to the package; the function below
                                         # rebinds it once, for good (later imports hit sys.modules and leave it)


def recipes():
    pkg = _recipes_pkg
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.name.startswith('_'))


def load_recipe(name):
    return importlib.import_module(f'earshot.forge.recipes.{name}')


def render_block(rec, params, seed, member=None, mode=None, gain=None):
    """a pick's render block (earshot.forge.schema: recipe module, render function, member, complete params, seed,
    critic mode, peak / gain) for a recipe module or stand-in object; a variant_recipe() pick names its member's own
    params and render function. None when the render function cannot be named (a driver's lambda: the driver adds
    the block itself, schema.make_block)."""
    from . import schema
    members = getattr(rec, 'MEMBERS', None)
    if members is not None and set(params) == {'variant'}:
        v = int(params['variant'])
        names = getattr(rec, 'MEMBER_NAMES', None) or [None] * len(members)
        return schema.make_block(getattr(rec, 'RENDER_FN', None), members[v], seed,
                                 member=names[v] if member is None else member, mode=mode, gain=gain)
    return schema.make_block(rec.render, params, seed, member=member, mode=mode, gain=gain)


def variant_recipe(render_fn, members, names=None, name=None, **attrs):
    """a stand-in recipe whose only searched key 'variant' indexes `members` (complete param dicts for the module-level
    `render_fn`): forge(variant_recipe(...), budget=len(members), pop=len(members), keep=len(members), init=[{'variant':
    k} ...]) scores an exact hand-made member set (the tonal drivers' final pass); each pick's render block names the
    member's own params and render_fn, so the picks re-render from report.json. attrs: TARGET, ALTS, REFSET, ..."""
    import types
    members = [dict(m) for m in members]
    mod = getattr(getattr(render_fn, 'func', render_fn), '__module__', None) or 'earshot.forge.recipes.variant'
    return types.SimpleNamespace(__name__=name or mod, SPACE={'variant': list(range(len(members)))},
                                 DEFAULTS={'variant': 0}, render=lambda p, r: render_fn(dict(members[p['variant']]), r),
                                 MEMBERS=members, MEMBER_NAMES=list(names) if names else None, RENDER_FN=render_fn,
                                 **attrs)


def pick_refs(refs=None, refset=None, recipe=None, log=None):
    """(reference files, refset category used) for a run: `refset` (a category of earshot.forge.refsets, its 'forge'
    split only: the 'bench' split stays held out), else `refs` (files / folders), else the recipe's REFSET when this
    machine has a registry with refs in it, else the recipe's REFS that exist here. Held-out files (a registry bench
    ref anywhere, or for the recipe's REFS a kit file cut from a bench source: refsets.held_out) are dropped."""
    from . import refsets as _rs

    def keep(paths, kit):
        bad = set(_rs.held_out(paths, kit=kit))
        if bad and log:
            log(f"dropped {len(bad)} held-out reference(s) (benchmark side): "
                + ', '.join(sorted(os.path.basename(p) for p in bad)))
        return [p for p in paths if p not in bad]

    if refset:
        return _rs.refset(refset, split='forge'), refset
    if refs:
        return keep(audio.list_media(refs, exts=audio.AUDIO_EXTS), kit=False) or None, None
    cat = getattr(recipe, 'REFSET', None)
    if cat:
        try:
            rr = _rs.refset(cat, split='forge')
            if rr:
                return rr, cat
        except (_rs.RefsetsMissing, KeyError):
            pass
    rr = keep([p for p in (getattr(recipe, 'REFS', None) or []) if os.path.exists(p)], kit=True)
    return rr or None, None


def resolve_refs(refs=None, refset=None, recipe=None, log=None):
    """the reference files of pick_refs() (None when there are none)"""
    return pick_refs(refs, refset, recipe, log)[0]


# score keys copied into report['history'] and report['ref_scores'] (v1 model terms + the v2 model-free components)
SCORE_KEYS = ('total', 'passes', 'match_z', 'alt_margin', 'p_real', 'ref_typ_z', 'feat_pen', 'tone_db', 'tone_ratio',
              'sounds_like', 'total_v1', 'mode', 'tone_pen', 'env_db', 'env_ratio', 'env_out', 'env_pen', 'art_pen',
              'polish_pen', 'hard', 'polish_words')


def _fmt(v, nd=2):
    return '-' if v is None else (f'{v:.{nd}f}' if isinstance(v, (float, np.floating)) else str(v))


def audition(path, made, refs, pairs, gap_s=0.6, pair_gap_s=1.2, peak_db=-3.0, keep_made_levels=False):
    """write made, ref, made, ref ... (each peak-normalised to peak_db, stereo 48 kHz) + a .txt cue list next to it;
    made / refs = [(name, array)], pairs = a ref index (or None) per made. keep_made_levels=True plays the made
    sounds at their own (e.g. loudness-matched) levels and peak-normalises only the refs. Returns {'file', 'txt',
    'cues'}"""
    import soundfile as sf
    from . import dsp
    made_names = {id(x) for _, x in made}

    def st(x):
        keep = keep_made_levels and id(x) in made_names
        x = np.asarray(x, np.float64)
        x = np.stack([x, x], 1) if x.ndim == 1 else x
        return x if keep else dsp.normalize(x, peak_db)

    parts, cues, t = [], [], 0.0
    for i, (name, x) in enumerate(made):
        seq = [('made', name, x)]
        j = pairs[i] if i < len(pairs) else None
        if j is not None:
            seq.append(('ref', refs[j][0], refs[j][1]))
        for k, (kind, nm, y) in enumerate(seq):
            y = st(y)
            cues.append({'t': round(t, 2), 'kind': kind, 'name': nm})
            parts.append(y)
            t += len(y) / 48000
            g = gap_s if k < len(seq) - 1 else pair_gap_s
            parts.append(np.zeros((int(g * 48000), 2)))
            t += g
    x = np.concatenate(parts) if parts else np.zeros((4800, 2))
    sf.write(path, x.astype(np.float32), 48000, subtype='PCM_24')
    txt = os.path.splitext(path)[0] + '.txt'
    lines = [f'{os.path.basename(path)}: each made sound, then its nearest real reference (measured: tonal balance + '
             + (f'envelope), the made sounds at their own matched levels, the refs peak-normalised to {peak_db:g} dBFS'
                if keep_made_levels else
                f'envelope), every sound peak-normalised to {peak_db:g} dBFS (peak-matched, not loudness-matched)')]
    lines += [f"{c['t']:8.2f} s  {c['kind']:<4}  {c['name']}" for c in cues]
    with open(txt, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return {'file': path.replace(chr(92), '/'), 'txt': txt.replace(chr(92), '/'), 'cues': cues}


def report_md(r):
    """the run as Markdown: what was asked, the budget, per-generation best, each pick in the critic's words and the
    measured gap to the real references"""
    L = [f"# forge {r['recipe']}: '{r['target']}'", '']
    refs = [os.path.basename(p) for p in r.get('refs') or []]
    tm = r.get('timing') or {}
    L.append(f"**Asked:** recipe `{r['recipe']}`, target '{r['target']}'"
             + (f", alternatives: {', '.join(repr(a) for a in r['alts'])}" if r.get('alts') else '')
             + (f", refset `{r['refset']}`" if r.get('refset') else '')
             + f"; {len(refs)} reference recording(s)" + (f": {', '.join(refs)}" if refs else '')
             + (f"; critic v{r['critic']['version']}, mode {r['critic']['mode']}" if r.get('critic') else '') + '.')
    L.append('')
    L.append(f"**Budget:** strategy `{r.get('strategy', 'es')}`, {r['budget']} critic-scored candidates "
             f"(pop {r.get('pop')}, {r['generations']} generations)"
             + (f", pre-screen: {r['prescreen']} renders per generation ranked by measurement, the best "
                f"{r.get('pop')} scored" if r.get('prescreen') else ', no pre-screen')
             + f", reps {r.get('reps', 1)}, tone weight {r.get('tone_weight')}, seed {r['seed']}"
             + (f", {r.get('n_bank', 0)} start(s) from the seed bank" if r.get('n_bank') else '')
             + (f", {r.get('restarts')} restart(s)" if r.get('restarts') else '') + '.')
    if tm:
        L.append(f"Took {tm.get('total_s', 0) / 60:.1f} min for the search: {tm.get('rendered')} renders, "
                 f"{tm.get('scored')} scored ({tm.get('scored_per_min')} scored / min, {tm.get('rendered_per_min')} "
                 f"rendered / min; render {tm.get('render_s')} s, measurements {tm.get('pre_s')} s, critic "
                 f"{tm.get('score_s')} s).")
    L += ['', '## Search', '', '| gen | best total | this gen best | this gen median | passes | best tone dB | sigma | note |',
          '|---|---|---|---|---|---|---|---|']
    for g in r.get('gens', []):
        L.append(f"| {g['gen']} | {_fmt(g.get('best'))} | {_fmt(g.get('gen_best'))} | {_fmt(g.get('gen_median'))} | "
                 f"{g.get('passes')}/{g.get('scored')} | {_fmt(g.get('best_tone_db'), 1)} | {_fmt(g.get('sigma'), 3)} | "
                 f"{('restart from ' + g['restart']) if g.get('restart') else ''} |")
    L += ['', '## Picks', '', r.get('pick_rule', ''), '']
    for p in r['picks']:
        L.append(f"### {p['name']}" + (f" ({os.path.basename(p['file'])})" if p.get('file') else ''))
        L.append(f"- critic: {p['score'].get('summary')}")
        rb = p.get('robust')
        if rb:
            L.append(f"- robust ({rb['n']} new seeds): mean {_fmt(rb['mean'])}, min {_fmt(rb['min'])}, sd {_fmt(rb['sd'])}, "
                     f"pass rate {_fmt(rb['pass_rate'])}" + (f", tone {rb['tone_db_mean']} dB" if 'tone_db_mean' in rb else ''))
        m = p.get('measured') or {}
        if m:
            L.append(f"- measured vs the refs: tone {_fmt(m.get('tone_db'), 1)} dB from their median (a typical real one "
                     f"sits {_fmt(m.get('ref_spread'), 1)} dB away), t40 {_fmt(m.get('t40_ms'), 0)} ms (refs "
                     f"{m.get('t40_ref')}), length {_fmt(m.get('length_s'))} s (refs {m.get('length_ref')}), nearest ref by "
                     f"measurement: {m.get('nearest_ref')}, by CLAP: {p['score'].get('nearest_ref')}"
                     + (f"; artifacts: {m['art']}" if m.get('art') else ''))
        if p.get('holdout'):
            L.append(f"- AudioSet (held out, decided nothing): {p['holdout'].get('audioset_top')}")
        L.append('- params: `' + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                              for k, v in p['params'].items()}) + f"` seed {p['seed']}")
        L.append('')
    if r.get('ref_scores'):
        L += ['## Gap to the real references', '',
              'The same critic scores the references leave-one-out (a reference is not its own neighbour). Model '
              'checks saturate once a search has tuned against them: read the measured columns first.', '',
              '| sound | total | passes | match z | p_real | tone dB | env pen | art pen | polish pen | hard | t40 ms | '
              'length s |', '|---|---|---|---|---|---|---|---|---|---|---|---|']

        def v2(s):
            return (f"{_fmt(s.get('env_pen'))} | {_fmt(s.get('art_pen'))} | {_fmt(s.get('polish_pen'))} | "
                    f"{', '.join(s.get('hard') or []) or '-'}")
        for p in r['picks']:
            s, m = p['score'], p.get('measured') or {}
            L.append(f"| made {p['name']} | {_fmt(s.get('total'))} | {s.get('passes')} | {_fmt(s.get('match_z'))} | "
                     f"{_fmt(s.get('p_real'))} | {_fmt(m.get('tone_db'), 1)} | {v2(s)} | {_fmt(m.get('t40_ms'), 0)} | "
                     f"{_fmt(m.get('length_s'))} |")
        for s in r['ref_scores']:
            L.append(f"| real {s['file']} | {_fmt(s.get('total'))} | {s.get('passes')} | {_fmt(s.get('match_z'))} | "
                     f"{_fmt(s.get('p_real'))} | {_fmt(s.get('tone_loo_db'), 1)} | {v2(s)} | {_fmt(s.get('t40_ms'), 0)} | "
                     f"{_fmt(s.get('length_s'))} |")
        if r.get('gap'):
            L += ['', r['gap']]
    L += ['', '## Files', '', '- WAVs: ' + ', '.join(os.path.basename(p['file']) for p in r['picks'] if p.get('file')),
          '- compare.png: each pick next to its nearest reference (spectrograms, envelopes, 1/3-octave balance)',
          '- audition.wav + audition.txt: made, ref, made, ref ... peak-matched',
          '- report.json: everything above plus the whole search history', '']
    return '\n'.join(L)


def forge(recipe, target=None, alts=None, refs=None, budget=160, pop=16, keep=4, seed=0, out=None, log=print,
          init=None, holdout=True, robust=0, space=None, sigma0=0.22, sigma1=0.04, critic=None, tone=0.0,
          tone_target=None, reps=1, strategy='es', prescreen=0, workers=None, bank=False, shortlist=1, refset=None,
          outputs=True, bank_dir=None, weights=None, mode='auto', families=None, box=0.15):
    """run the loop for one recipe; writes WAVs + report.json (+ report.md, compare.png, audition.wav when outputs)
    into `out` when given; returns the report dict.
    robust=N re-renders every pick on N new seeds (mean / min / pass rate in pick['robust']); shortlist=S (with
    robust) robust-scores S x keep diverse candidates and keeps the keep best robust means; space= a narrower
    {name: spec} family; sigma0/sigma1 the first / last ES mutation step; critic= a
    prebuilt Critic; tone= the weight of the model-free tonal-balance penalty (Critic tone=, tone_target=); reps=
    renders per candidate inside the search; strategy 'es' | 'cma' (sigma0 / sigma1 are the ES steps; CMA starts at
    0.15); prescreen=K renders per generation ranked by
    measurement before the critic (K > pop); workers= render threads; bank=True starts from the recipe's seed bank and
    adds the picks to it (bank_dir= another folder); refset= a refsets category (its 'forge' split) instead of refs
    (held-out bench files among explicit refs are dropped: pick_refs); weights= / mode= go to the Critic (v2 weights,
    'auto' | 'real' | 'designed'; weights={'tone': 0} makes the tone check report-only; a critic built here also gets
    category= the refset (or the recipe's REFSET): absolute study targets where the refs are thin, critic.py).
    families= {name: params | {'init': params, 'space': {...}, 'fixed': {...}, 'budget': n}}: one keep=1 search per
    member (inside `space`, else family_box(rec, params, box); `fixed` keys are never searched; every key outside
    the box keeps the member's own value, not the recipe's DEFAULTS), then palette() of the exact members (module
    doc: Families); returns the palette report with 'runs'."""
    import soundfile as sf
    from .critic import Critic
    from .search import diverse, optimize, prescreen_for, render, robust as robust_stats, sound_dist
    from .schema import SCHEMA
    t0 = time.time()
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    name = rec.__name__.rsplit('.', 1)[-1]
    if critic is None:
        target = target or rec.TARGET
        alts = list(alts if alts is not None else getattr(rec, 'ALTS', []))
        refs = resolve_refs(refs, refset, log=log) if (refs or refset) else refs
        critic = Critic(target, alts, refs, tone=tone, tone_target=tone_target, weights=weights, mode=mode,
                        category=refset or getattr(rec, 'REFSET', None))
    else:
        target, alts = critic.target, list(critic.alts)
    if families:
        return _forge_families(rec, families, critic, out=out, budget=budget, pop=pop, seed=seed, log=log,
                               holdout=holdout, robust=robust, sigma0=sigma0, sigma1=sigma1, reps=reps,
                               strategy=strategy, prescreen=prescreen, workers=workers, shortlist=shortlist,
                               refset=refset, outputs=outputs, box=box, t0=t0)
    real = name in recipes()
    starts = list(init or [])
    n_bank = 0
    if bank and real:
        from .search import load_seeds
        bs = load_seeds(name, target, critic.refs, folder=bank_dir)
        n_bank = len(bs)
        starts += bs
    pre = prescreen_for(critic)
    res = optimize(rec, critic, budget=budget, pop=pop, seed=seed, init=starts, log=log, space=space, sigma0=sigma0,
                   sigma1=sigma1, reps=reps, strategy=strategy, prescreen=prescreen, workers=workers, pre=pre)
    pool = res['best'] + sorted(res['history'], key=lambda h: -h['score']['total'])[:64]
    S = max(1, int(shortlist)) if robust else 1
    picks = diverse(pool, rec, keep * S)
    rstats = [robust_stats(rec, critic, h['params'], robust, workers=workers) for h in picks] if robust else []
    if robust and S > 1:
        order = sorted(range(len(picks)), key=lambda i: (-(rstats[i]['pass_rate'] >= 0.5),
                                                          -(rstats[i]['mean'] if rstats[i]['mean'] is not None else -1e9)))
        order = order[:keep]
        picks, rstats = [picks[i] for i in order], [rstats[i] for i in order]
        rule = (f'Chosen by a fixed rule: {keep * S} diverse strong candidates (parameters and sound differ), re-rendered '
                f'on {robust} new seeds each, the {keep} with the best robust mean kept (passing at least half first).')
    else:
        rule = ('Chosen by a fixed rule: the strongest candidates whose parameters and sound differ (passing first)'
                + (f', each re-rendered on {robust} new seeds for the robust columns.' if robust else '.'))
    ref_scores = critic.score_refs(loo=True) if critic.refs else []
    report = {'schema': SCHEMA, 'recipe': name, 'target': target, 'alts': alts, 'refs': critic.refs, 'refset': refset,
              'budget': budget,
              'seed': seed, 'pop': pop, 'tone_weight': (critic.W.get('tone') if getattr(critic, 'version', 1) >= 2 and hasattr(critic, 'W')
                                                  else getattr(critic, 'tone', 0.0)),
              'tone_target': getattr(critic, 'tone_target', None), 'reps': reps, 'strategy': strategy,
              'critic': {'version': getattr(critic, 'version', 1), 'mode': getattr(critic, 'mode', None),
                         'weights': getattr(critic, 'W', None)},
              'prescreen': res['prescreen'], 'restarts': res.get('restarts', 0), 'n_bank': n_bank,
              'generations': res['generations'], 'gens': res['gens'], 'timing': res['timing'], 'pick_rule': rule,
              'picks': [], 'ref_scores': [], 'ref_scores_loo': True}
    if space is not None:
        report['space'] = space
    if res.get('fill_warnings'):
        report['fill_warnings'] = res['fill_warnings']
    xs = [render(rec, h['params'], h['seed']) for h in picks]
    hold = critic.holdout([x for x in xs if x is not None]) if holdout and xs else []
    ref_x = [audio.load(p) for p in critic.refs] if critic.refs else []
    ref_hold = critic.holdout(ref_x) if holdout and ref_x else []
    if out:
        os.makedirs(out, exist_ok=True)
    pairs = []
    for i, (h, x) in enumerate(zip(picks, xs)):
        item = {'name': f'{name}_{i + 1:02d}', 'params': h['params'], 'seed': h['seed'],
                'score': {k: v for k, v in h['score'].items() if k != 'i'}}
        rb = render_block(rec, h['params'], h['seed'], mode=getattr(critic, 'mode', None))
        if rb is not None:
            item['render'] = rb
        if i < len(hold):
            item['holdout'] = hold[i]
        if robust:
            item['robust'] = rstats[i]
        j = None
        if x is not None and pre.n:
            row = pre.row(x)
            j = pre.nearest(row['desc'])
            d = row['desc']
            item['measured'] = {'tone_db': row['tone_db'], 'tone_ratio': row['tone_ratio'],
                                'ref_spread': round(pre.spread, 2), 'env_z': row['env_z'],
                                't40_ms': d['env']['t40_ms'], 'length_s': round(len(x) / 48000, 3),
                                't40_ref': [round(min(r['env']['t40_ms'] for r in pre.descs)),
                                            round(max(r['env']['t40_ms'] for r in pre.descs))],
                                'length_ref': [round(min(len(r) for r in ref_x) / 48000, 3),
                                               round(max(len(r) for r in ref_x) / 48000, 3)],
                                'nearest_ref': os.path.basename(critic.refs[j]),
                                'nearest_dist': round(sound_dist(d, pre.descs[j]), 2), 'art': row['art']['flags']}
        pairs.append(j)
        if out and x is not None:
            item['file'] = os.path.join(out, item['name'] + '.wav').replace(chr(92), '/')
            sf.write(item['file'], x, 48000, subtype='PCM_24')
        report['picks'].append(item)
    if not all('render' in p for p in report['picks']):  # schema 2 = every pick names its render (a driver's lambda
        report.pop('schema')                              # stand-in cannot: schema.legacy_block infers those)
    for i, (p, s) in enumerate(zip(critic.refs, ref_scores)):
        report['ref_scores'].append({'file': os.path.basename(p), 'alt_zs': s.get('alt_zs'),
                                     **{k: s.get(k) for k in SCORE_KEYS}})
        if pre.n > i:
            others = [pre.descs[k]['profile'] for k in range(pre.n) if k != i]
            if others:
                report['ref_scores'][-1]['tone_loo_db'] = round(float(np.sqrt(np.mean(
                    (pre.descs[i]['profile'] - np.median(others, 0)) ** 2))), 2)
            report['ref_scores'][-1]['t40_ms'] = pre.descs[i]['env']['t40_ms']
            report['ref_scores'][-1]['length_s'] = round(len(ref_x[i]) / 48000, 3)
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
    md = [p['measured']['tone_db'] for p in report['picks'] if p.get('measured')]
    rl = [r['tone_loo_db'] for r in report['ref_scores'] if r.get('tone_loo_db') is not None]
    if md and rl:
        report['gap'] = (f"Measured gap: the picks sit {min(md):.1f}-{max(md):.1f} dB from the references' median "
                         f"1/3-octave balance; the references themselves sit {min(rl):.1f}-{max(rl):.1f} dB from the "
                         f"others' median (leave-one-out). Inside that range = as typical as a real recording.")
        report['summary'] += ' ' + report['gap']
    report['history'] = [{'gen': h['gen'], 'params': h['params'], 'seed': h['seed'],
                          'score': {k: h['score'].get(k) for k in SCORE_KEYS},
                          **({'pre': h['pre']} if 'pre' in h else {})}
                         for h in res['history']]
    report['timing']['forge_s'] = round(time.time() - t0, 1)
    if out:
        report['files'] = {}
        if outputs:
            from .viz import compare_png
            made = [(p['name'], x) for p, x in zip(report['picks'], xs) if x is not None]
            refn = [(os.path.basename(p), x) for p, x in zip(critic.refs, ref_x)]
            pr = [j for j, x in zip(pairs, xs) if x is not None]
            try:
                report['files']['compare'] = compare_png(os.path.join(out, 'compare.png'), made, refn, pairs=pr,
                                                         title=f"forge {name}: made (left) vs nearest real reference")['file']
            except Exception as e:                    # a plotting problem must not lose the run
                report['files']['compare_error'] = f'{type(e).__name__}: {e}'
            report['files']['audition'] = audition(os.path.join(out, 'audition.wav'), made, refn, pr)['file']
            with open(os.path.join(out, 'report.md'), 'w', encoding='utf-8') as fh:
                fh.write(report_md(report))
            report['files']['report_md'] = os.path.join(out, 'report.md').replace(chr(92), '/')
        with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    if bank and real and report['picks']:
        from .search import save_seeds
        try:
            save_seeds(name, report['picks'], target, critic.refs, folder=bank_dir,
                       source=f"forge {strategy} budget {budget} seed {seed}" + (f' refset {refset}' if refset else ''))
        except OSError as e:
            if log:
                log(f'seed bank not updated: {e}')
    return report


def family_box(recipe, params, box=0.15, keys=None):
    """a narrow search space around one preset: each numeric SPACE entry (or only `keys`) shrunk to +-box of its
    range (log ranges in log units) around the preset's value, choices and keys outside `keys` fixed at the preset"""
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    out = {}
    for k, spec in rec.SPACE.items():
        if (keys is not None and k not in keys) or isinstance(spec, list) or k not in params:
            continue
        lo, hi = float(spec[0]), float(spec[1])
        kind = spec[2] if len(spec) > 2 else 'lin'
        v = float(params[k])
        if kind == 'log' and lo > 0:
            a, b, c = np.log(lo), np.log(hi), np.log(min(max(v, lo), hi))
            w = box * (b - a)
            out[k] = (float(np.exp(max(a, c - w))), float(np.exp(min(b, c + w))), 'log')
        else:
            w = box * (hi - lo)
            q = (max(lo, v - w), min(hi, v + w))
            out[k] = (q[0], q[1], 'int') if kind == 'int' else q
        if out[k][0] >= out[k][1]:
            del out[k]
    return out


def _forge_families(rec, families, critic, out=None, budget=96, pop=16, seed=0, log=print, holdout=True, robust=0,
                    sigma0=0.22, sigma1=0.04, reps=1, strategy='es', prescreen=0, workers=None, shortlist=1,
                    refset=None, outputs=True, box=0.15, t0=None):
    t0 = time.time() if t0 is None else t0
    items = list(families.items()) if isinstance(families, dict) else list(families)
    members, runs = [], []
    import types
    for k, (fam, spec) in enumerate(items):
        spec = dict(spec)
        plain = not ({'init', 'space', 'fixed', 'budget'} & set(spec))
        init = dict(spec if plain else spec.get('init', {}))
        fixed = dict(spec.get('fixed', {}) if not plain else {})
        base = {**(getattr(rec, 'DEFAULTS', {}) or {}), **init, **fixed}
        space = {kk: v for kk, v in (spec.get('space') or family_box(rec, base, box)).items() if kk not in fixed}
        # keys outside the member's box keep the MEMBER's values (optimize() fills them from DEFAULTS)
        mrec = types.SimpleNamespace(**{a: getattr(rec, a) for a in ('TARGET', 'ALTS', 'REFSET', 'REFS')
                                        if hasattr(rec, a)})
        mrec.__name__, mrec.SPACE, mrec.DEFAULTS, mrec.render = rec.__name__, rec.SPACE, base, rec.render
        rd = os.path.join(out, 'runs', str(fam)) if out else None
        rep = forge(mrec, critic=critic, space=space, init=[base], keep=1,
                    budget=int(spec.get('budget', budget)), pop=pop, seed=seed + 101 * k, out=rd, log=log,
                    holdout=False, robust=robust, sigma0=sigma0, sigma1=sigma1, reps=reps, strategy=strategy,
                    prescreen=prescreen, workers=workers, shortlist=shortlist, refset=refset, outputs=False,
                    bank=False)
        if rep['picks']:
            pk = rep['picks'][0]
            members.append({'params': pk['params'], 'seed': pk['seed'], 'family': str(fam),
                            **({'robust': pk['robust']} if pk.get('robust') else {})})
        runs.append({'family': str(fam), 'out': rd, 'budget': rep['budget'], 'timing': rep['timing'],
                     'total': rep['picks'][0]['score'].get('total') if rep['picks'] else None})
    rep = palette(rec, members, critic, out=out, name=refset, holdout=holdout, log=log, outputs=outputs)
    rep['runs'] = runs
    rep['timing'] = {'forge_s': round(time.time() - t0, 1)}
    if out:
        with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return rep


def _variety(E):
    """mean / max pairwise cosine of embeddings (rows), None with fewer than 2"""
    if E is None or len(E) < 2:
        return None
    E = np.asarray(E, np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    C = E @ E.T
    off = C[~np.eye(len(E), dtype=bool)]
    return {'cos_mean': round(float(off.mean()), 3), 'cos_max': round(float(off.max()), 3)}


def palette(recipe, members, critic, out=None, name=None, peak_db=-1.0, loudness=True, robust=0, workers=None,
            holdout=True, log=print, outputs=True):
    """score an exact member set together and write it as one swappable palette (module doc: Families).
    members: [{'params', 'seed'[, 'family', 'name', 'robust']}] (a report's picks work) or bare params dicts (seed 0);
    params are merged over the recipe's DEFAULTS. Every member is rendered (search.render) and kept: no diverse()
    drop. Returns {'recipe', 'category', 'picks', 'variety', 'loudness', 'refs', ...}; writes <name>_NN.wav,
    report.json, report.md and _audition.wav into `out` when given (compare.png too when outputs)."""
    import soundfile as sf
    from .polish import loudness_spread
    from .schema import SCHEMA
    from .search import prescreen_for, render, robust as robust_stats, sound_dist
    from .tone import profile
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    rname = rec.__name__.rsplit('.', 1)[-1]
    name = name or getattr(rec, 'REFSET', None) or rname
    base = dict(getattr(rec, 'DEFAULTS', {}) or {})
    ms = []
    for i, m in enumerate(members):
        m = dict(m) if isinstance(m, dict) and 'params' in m else {'params': dict(m), 'seed': 0}
        m['params'] = {**base, **m['params']}
        m['seed'] = int(m.get('seed') or 0)
        ms.append(m)
    xs = [render(rec, m['params'], m['seed']) for m in ms]
    bad = [i for i, x in enumerate(xs) if x is None]
    if bad and log:
        log(f'palette: {len(bad)} member(s) failed to render or were silent: {bad}')
    keep = [i for i, x in enumerate(xs) if x is not None]
    ms, xs = [ms[i] for i in keep], [np.asarray(xs[i], np.float64) for i in keep]
    lu0 = loudness_spread(xs)
    gains = [1.0] * len(xs)
    if loudness and xs:
        gains = [10 ** ((min(lu0['lufs']) - v) / 20) for v in lu0['lufs']]
    c = 10 ** (peak_db / 20) / max((float(np.abs(x).max()) * g for x, g in zip(xs, gains)), default=1.0)
    xs = [(x * g * c).astype(np.float32) for x, g in zip(xs, gains)]
    rows = critic.score(xs) if xs else []
    E = critic.embed(xs) if hasattr(critic, 'embed') and len(xs) > 1 else None
    refs = list(getattr(critic, 'refs', None) or [])
    ref_x = [audio.load(p) for p in refs]
    E_r = critic.embed(ref_x) if hasattr(critic, 'embed') and len(ref_x) > 1 else None
    P = [profile(x) for x in xs]
    tone_d = [float(np.sqrt(np.mean((P[i] - P[j]) ** 2))) for i in range(len(P)) for j in range(i + 1, len(P))]
    variety = {'members': _variety(E), 'refs': _variety(E_r),
               'tone_min_db': round(min(tone_d), 2) if tone_d else None,
               'tone_mean_db': round(float(np.mean(tone_d)), 2) if tone_d else None}
    hold = critic.holdout(xs) if holdout and xs and hasattr(critic, 'holdout') else []
    pre = prescreen_for(critic)
    report = {'schema': SCHEMA, 'recipe': rname, 'category': name, 'target': getattr(critic, 'target', None),
              'refs': refs,
              'critic': {'version': getattr(critic, 'version', 1), 'mode': getattr(critic, 'mode', None),
                         'category': getattr(critic, 'category', None)},
              'pick_rule': ('palette: the exact member set, scored together (no diverse() drop), loudness-matched to '
                            f'the quietest member, then one common gain to a {peak_db:g} dBFS peak'),
              'variety': variety, 'loudness': {'before': lu0, 'after': loudness_spread(xs)}, 'picks': []}
    if out:
        os.makedirs(out, exist_ok=True)
    pairs = []
    for i, (m, x, row, g) in enumerate(zip(ms, xs, rows, gains)):
        item = {'name': m.get('name') or f'{name}_{i + 1:02d}', 'family': m.get('family'), 'params': m['params'],
                'seed': m['seed'], 'gain_db': round(float(20 * np.log10(g * c)), 2),
                'score': {k: v for k, v in row.items() if k != 'i'}}
        rb = render_block(rec, m['params'], m['seed'], member=m.get('family') or m.get('member') or m.get('preset'),
                          mode=getattr(critic, 'mode', None), gain=float(g * c))
        if rb is not None:
            item['render'] = rb
        if m.get('robust'):
            item['robust'] = m['robust']
        elif robust:
            item['robust'] = robust_stats(rec, critic, m['params'], robust, workers=workers)
        if i < len(hold):
            item['holdout'] = hold[i]
        j = None
        if pre.n:
            r = pre.row(x)
            j = pre.nearest(r['desc'])
            item['measured'] = {'tone_db': r['tone_db'], 't40_ms': r['desc']['env']['t40_ms'],
                                'length_s': round(len(x) / 48000, 3), 'nearest_ref': os.path.basename(refs[j]),
                                'nearest_dist': round(sound_dist(r['desc'], pre.descs[j]), 2)}
        pairs.append(j)
        if out:
            item['file'] = os.path.join(out, item['name'] + '.wav').replace(chr(92), '/')
            sf.write(item['file'], x, 48000, subtype='PCM_24')
        report['picks'].append(item)
    if not all('render' in p for p in report['picks']):  # schema 2 only when every pick names its render
        report.pop('schema')
    n_pass = sum(1 for p in report['picks'] if p['score'].get('passes'))
    vm = variety['members'] or {}
    vr = variety['refs'] or {}
    report['summary'] = (f"palette {name}: {len(report['picks'])} members, {n_pass} pass; variety cos "
                         f"{vm.get('cos_mean', '-')} (refs {vr.get('cos_mean', '-')}), closest pair "
                         f"{variety['tone_min_db']} dB apart (1/3 oct); loudness spread "
                         f"{lu0['spread_lu']} -> {report['loudness']['after']['spread_lu']} LU")
    if out:
        report['files'] = {}
        made = [(p['name'], x) for p, x in zip(report['picks'], xs)]
        refn = [(os.path.basename(p), x) for p, x in zip(refs, ref_x)]
        report['files']['audition'] = audition(os.path.join(out, '_audition.wav'), made, refn, pairs,
                                               keep_made_levels=True)['file']
        if outputs:
            from .viz import compare_png
            try:
                report['files']['compare'] = compare_png(os.path.join(out, 'compare.png'), made, refn, pairs=pairs,
                                                         title=f'palette {name}: made (left) vs nearest reference')['file']
            except Exception as e:                    # a plotting problem must not lose the palette
                report['files']['compare_error'] = f'{type(e).__name__}: {e}'
            with open(os.path.join(out, 'report.md'), 'w', encoding='utf-8') as fh:
                fh.write(palette_md(report))
        with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    if log:
        log(report['summary'])
    return report


def palette_md(r):
    """a palette report as Markdown"""
    L = [f"# palette {r['category']} ({r['recipe']})", '', r['summary'], '', r['pick_rule'] + '.', '',
         '| member | family | total | passes | tone dB | env pen | art pen | polish pen | hard | gain dB | nearest ref |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for p in r['picks']:
        s, m = p['score'], p.get('measured') or {}
        L.append(f"| {p['name']} | {p.get('family') or '-'} | {_fmt(s.get('total'))} | {s.get('passes')} | "
                 f"{_fmt(s.get('tone_db'), 1)} | {_fmt(s.get('env_pen'))} | {_fmt(s.get('art_pen'))} | "
                 f"{_fmt(s.get('polish_pen'))} | {', '.join(s.get('hard') or []) or '-'} | {p['gain_db']} | "
                 f"{m.get('nearest_ref', '-')} |")
    L += ['', '## Critic words', '']
    L += [f"- {p['name']}: {p['score'].get('summary')}" for p in r['picks']]
    return '\n'.join(L) + '\n'


def render_params(recipe, params, seeds=None, out='.', refs=None):
    """render parameter sets without searching. params: a dict, a list of dicts, a report.json dict (its picks, with
    their own seeds unless `seeds` is given) or a seed bank dict (its entries). Writes <out>/<recipe>_r<i>_s<seed>.wav;
    with refs, each file also gets the model-free measurements (tone dB, t40, nearest ref). Returns {'recipe', 'files'}
    A report's pick renders through its render block (earshot.forge.schema: its own render function and complete
    params, e.g. typing.render_run for a pro_typing_run pick, a 'variant' pick's member params) when it has one, or
    when the block inferred for an older report names this recipe's module; else rec.render over its DEFAULTS."""
    import soundfile as sf
    import types
    from . import schema
    from .search import Prescreen, render
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    name = rec.__name__.rsplit('.', 1)[-1]
    if isinstance(params, dict) and 'picks' in params:
        sets = []
        for p in params['picks']:
            try:
                b = schema.block(params, p)
                if isinstance(p.get('render'), dict) or b['recipe'] == name:
                    fn = schema.resolve(b)
                    sets.append((b['params'], p.get('seed'), types.SimpleNamespace(__name__=fn.__module__, render=fn)))
                    continue
            except (schema.NotReproducible, KeyError, TypeError, ValueError):
                pass
            sets.append((p['params'], p.get('seed')))
    elif isinstance(params, dict) and 'entries' in params:
        sets = [(e['params'], None) for e in params['entries']]
    elif isinstance(params, dict):
        sets = [(params, None)]
    else:
        sets = [(p, None) for p in params]
    ref_files = audio.list_media(refs, exts=audio.AUDIO_EXTS) if refs else []
    pre = Prescreen(ref_files) if ref_files else None
    os.makedirs(out, exist_ok=True)
    files = []
    for i, (p, own, *blk) in enumerate(sets):
        for s in (seeds or [own if own is not None else 0]):
            x = (render(blk[0], p, int(s)) if blk else                  # a render block: complete params, own fn
                 render(rec, {**(getattr(rec, 'DEFAULTS', {}) or {}), **p}, int(s)))
            item = {'set': i, 'seed': int(s), 'file': None}
            if x is not None:
                item['file'] = os.path.join(out, f'{name}_r{i + 1:02d}_s{int(s)}.wav').replace(chr(92), '/')
                sf.write(item['file'], x, 48000, subtype='PCM_24')
                if pre is not None and pre.n:
                    row = pre.row(x)
                    item['measured'] = {'tone_db': row['tone_db'], 'tone_ratio': row['tone_ratio'],
                                        't40_ms': row['desc']['env']['t40_ms'], 'art': row['art']['flags'],
                                        'nearest_ref': os.path.basename(ref_files[pre.nearest(row['desc'])])}
            files.append(item)
    return {'recipe': name, 'files': files}


def add_cli(sub):
    from ..cli import common
    p = common(sub.add_parser('forge', help='make sounds in code and improve them by listening (recipe + critic + search)'))
    p.add_argument('recipe', nargs='?', help='recipe name (earshot/forge/recipes/); omit to list them')
    p.add_argument('--list', action='store_true', help='list the recipes with their TARGET (and refset categories)')
    p.add_argument('--render', nargs='?', const=True, default=None, metavar='RECIPE',
                   help='render --params without searching (the recipe may follow --render or be the positional)')
    p.add_argument('--params', help='JSON: a params dict, a list of them, a report.json or a seed bank file')
    p.add_argument('--check', nargs='+', metavar='DIR',
                   help='re-render every pick of these output folders from their report.json and compare with the '
                        'WAVs (earshot.forge.schema; within 1e-6)')
    p.add_argument('--seeds', nargs='*', type=int, help='noise seeds for --render (default: each pick\'s own, else 0)')
    p.add_argument('--target', help="description to aim for (default: the recipe's)")
    p.add_argument('--alts', nargs='*', help='confusions it must sound less like')
    p.add_argument('--refs', nargs='*', help='real reference recordings (files / folders) to compare with')
    p.add_argument('--refset', help="a reference-set category (earshot.forge.refsets; its 'forge' split)")
    p.add_argument('--budget', type=int, default=160, help='candidates to score with the critic (default 160)')
    p.add_argument('--pop', type=int, default=16)
    p.add_argument('--variants', '--keep', dest='keep', type=int, default=4, help='diverse best sounds to write')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dir', dest='outdir', help='folder for the WAVs, report.json / .md, compare.png, audition.wav')
    p.add_argument('--strategy', choices=('es', 'cma'), default='es',
                   help='search: es (v1 evolution strategy, default: it beat CMA-ES at these budgets) or cma')
    p.add_argument('--prescreen', type=int, default=None,
                   help='renders per generation ranked by measurement before the critic (default 3 x pop; 0 = off)')
    p.add_argument('--robust', type=int, default=3, help='re-render every kept sound on N new seeds (default 3; 0 = off)')
    p.add_argument('--shortlist', type=int, default=2,
                   help='with --robust: robust-score this many x --variants candidates, keep the best robust means')
    p.add_argument('--no-bank', action='store_true', help='do not start from / add to the seed bank')
    p.add_argument('--workers', type=int, default=None, help='render threads (default EARSHOT_THREADS or 2)')
    p.add_argument('--tone', type=float, default=None,
                   help="weight of the model-free tonal-balance penalty vs the refs (default: the critic's, 1.0 in v2; "
                        "0 = report only)")
    p.add_argument('--mode', choices=('auto', 'real', 'designed'), default='auto',
                   help='critic mode: real refs (realness counts) or designed SFX (realness reported, not required)')
    p.add_argument('--reps', type=int, default=1,
                   help='renders per candidate inside the search, ranked by their mean (2 = half as lucky, twice as slow)')
    p.add_argument('--tone-target', type=float, default=None,
                   help="dB distance from the refs' median balance that counts as typical (default: the refs' own spread)")

    def run(a):
        if a.check:
            from . import schema
            return schema.main(list(a.check))
        if a.render is not None:
            name = a.render if isinstance(a.render, str) else a.recipe
            if not name or not a.params:
                print('usage: earshot forge --render <recipe> --params file.json [--seeds 1 2 3] [--dir out]')
                return 2
            with open(a.params, encoding='utf-8') as fh:
                params = json.load(fh)
            refs = resolve_refs(a.refs, a.refset, log=print)
            r = render_params(name, params, a.seeds, a.outdir or '.', refs)
            lines = [f"rendered {sum(1 for f in r['files'] if f['file'])} sound(s) of {r['recipe']}:"]
            for f in r['files']:
                m = f.get('measured')
                lines.append(f"  {f['file'] or 'failed or silent'}" + (
                    f"  tone {m['tone_db']} dB, t40 {m['t40_ms']:.0f} ms, nearest {m['nearest_ref']}" if m else ''))
            emit(r, a, text='\n'.join(lines))
            return 0
        if a.list or not a.recipe:
            lines = ['recipes (name: TARGET):']
            for n in recipes():
                try:
                    m = load_recipe(n)
                    lines.append(f'  {n}: {m.TARGET}' + (f'  [refset {m.REFSET}]' if getattr(m, 'REFSET', None) else ''))
                except Exception as e:
                    lines.append(f'  {n}: (does not load: {type(e).__name__})')
            try:
                from .refsets import categories
                lines.append('refsets: ' + ', '.join(categories()))
            except Exception:
                lines.append('refsets: none on this machine (see earshot/forge/refsets.py)')
            emit({'recipes': recipes()}, a, text='\n'.join(lines))
            return 0
        rec = load_recipe(a.recipe)
        refs, cat = pick_refs(a.refs, a.refset, rec, log=print)
        if not refs:
            print('note: no reference recordings (--refs / --refset): the critic cannot compare with real sounds')
        elif cat and not a.refset:
            print(f"refs: the recipe's refset '{cat}' ({len(refs)} forge-split recordings)")
        pre = a.prescreen if a.prescreen is not None else 3 * a.pop
        tone, weights = (a.tone or 0.0), ({'tone': 0.0} if a.tone == 0 else None)
        r = forge(rec, a.target, a.alts, refs, a.budget, a.pop, a.keep, a.seed, a.outdir, robust=a.robust,
                  tone=tone, tone_target=a.tone_target, reps=a.reps, strategy=a.strategy, prescreen=pre,
                  workers=a.workers, bank=not a.no_bank, shortlist=a.shortlist, refset=cat, weights=weights,
                  mode=a.mode)
        lines = [r['summary']] + [f"  {p['name']}: {p['score'].get('summary')}" for p in r['picks']]
        if a.outdir:
            lines.append(f"wrote {a.outdir}: WAVs, report.json, report.md, compare.png, audition.wav")
        emit({k: v for k, v in r.items() if k != 'history'}, a, text='\n'.join(lines))
        return 0
    p.set_defaults(func=run)
