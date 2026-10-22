"""earshot fit: translate a real recording into code (analysis by synthesis), then vary it.

    from earshot.forge import fit
    m = fit.fit('click.wav')                    # kind='auto' | 'modal' | 'spectral' | 'sines' | 'gesture' | 'hybrid'
    y = fit.render(m, seed=0)                   # the resynthesis (seeds change only noise / jitter); knobs vary it
    from earshot.forge.fit.quality import quality
    q = quality('click.wav', y)                 # mel dB, envelope r, tone dB, onset error, ... (clap=True: CLAP cos)
    src = fit.to_recipe(m, name='click_fit', source='click.wav')   # a forge recipe module (Python source)
    mm = fit.morph(m_a, m_b, 0.5)               # a new model between two fits of the same kind

    earshot fit click.wav -o made/click_fit.py --report       # recipe + resynthesis WAV + spectrogram PNG + metrics
    earshot fit --morph a.wav b.wav --amount 0.5 -o made/ab_morph.py
    earshot fit --forge made/click_fit.py --budget 96 --dir made/click_variations   # critic-searched variations

The fitters (one module each, plain-JSON models, deterministic per seed, pure numpy / scipy):
  modal.py     struck / ringing sounds -> strikes of damped sinusoids (ESPRIT + least squares) + band noise
  sines.py     tonal sounds -> tracked partials (McAulay-Quatieri / SMS) + strike atoms + 1/3-octave noise
  spectral.py  noisy sounds -> a time-varying 48-band noise gesture + stereo + texture + tonal ridges
  gesture.py   designed gestures (whooshes, swells, risers) -> tones (a sines / modal fit's partials) + air (noise)
  space.py     a stereo recording's decorrelated tail -> a seeded diffuse side layer (the chosen fit's post stage)
  auto.py      the engine: character -> candidates -> fit, render, score, keep the best; hybrids; morph
  quality.py   the metrics every fit is judged by (and the side-by-side spectrogram picture)
  codegen.py   model -> a forge recipe module (TARGET / SPACE / DEFAULTS / MODEL / render(p, r))
docs/FIT.md has how it works, the measured quality and the limits.
"""
import os


def fit(x, kind='auto', **kw):
    from . import auto
    return auto.fit(x, kind=kind, **kw)


def render(model, seed=0, **knobs):
    from . import auto
    return auto.render(model, seed=seed, **knobs)


def morph(a, b, amount=0.5):
    from . import auto
    return auto.morph(a, b, amount)


def to_recipe(model, **kw):
    from . import codegen
    return codegen.to_recipe(model, **kw)


# ------------------------------------------------------------------------------------------------------- CLI
def _paths(a, stem):
    """output paths: -o recipe.py names the folder and stem, else --dir (default '.') and <stem>_fit"""
    if a.output:
        d = os.path.dirname(os.path.abspath(a.output))
        base = os.path.splitext(os.path.basename(a.output))[0]
    else:
        d = os.path.abspath(a.dir or '.')
        base = f'{stem}_fit'
    os.makedirs(d, exist_ok=True)
    j = lambda ext: os.path.join(d, base + ext).replace(chr(92), '/')
    return {'recipe': j('.py'), 'wav': (a.render or j('.wav')).replace(chr(92), '/'), 'png': j('.png'),
            'model': j('.json'), 'report': j('.report.json'), 'name': base}


def _write_wav(path, y):
    import numpy as np
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, np.asarray(y, np.float32), 48000, subtype='FLOAT')
    return path


def run_fit(path, kind='auto', layers=None, output=None, out_dir=None, render_path=None, report=False, clap=False,
            describe=False, seed=0, target=None, fast=False, verbose=True, log=print, takes=False, bed=True):
    """the `earshot fit <file>` pipeline as a function: fit, write the recipe / WAV / PNG / model JSON, measure.
    takes=True scores the candidates on seeds 0, 1, 2 (auto.TAKE_SEEDS), not seed 0 alone. bed=False fits a
    recording with a bed under it as it is (auto.fit strips the bed by default). Returns the result dict (paths,
    quality, floor, chosen kind, timings)."""
    import time

    import numpy as np

    from . import auto, codegen
    from . import quality as Q

    class A:
        pass
    a = A()
    a.output, a.dir, a.render = output, out_dir, render_path
    stem = os.path.splitext(os.path.basename(path))[0]
    P = _paths(a, stem)
    t0 = time.time()
    X = Q.load(path)
    m = auto.fit(X, kind=kind, layers=layers, clap=clap, fast=fast, name=os.path.basename(path),
                 verbose=verbose and kind == 'auto', seeds=auto.TAKE_SEEDS if takes else (0,), bed=bed)
    t_fit = time.time() - t0
    codegen.write_recipe(P['recipe'], m, name=P['name'], source=path, target=target)
    rec = codegen.load_recipe(P['recipe'])
    y = rec.render_seed(seed)                       # the code's render, not the model's: what the file really does
    _write_wav(P['wav'], y)
    q = Q.quality(X, y, clap=report or clap, describe=describe)
    fl = Q.floor(lambda s: rec.render_seed(s), clap=False) if report else None
    codegen.write_recipe(P['recipe'], m, name=P['name'], source=path, target=target, quality=q, floor=fl)
    rec = codegen.load_recipe(P['recipe'])
    y2 = rec.render_seed(seed)
    same = bool(np.array_equal(np.asarray(y2), np.asarray(auto.render(m, seed=seed))) and np.array_equal(y2, y))
    auto.save(m, P['model'])
    Q.pair_png(P['png'], X, y, title=f"{os.path.basename(path)} -> {P['name']}.py ({auto.kind_of(m)}"
               + (f", auto: {m['fit_engine']['chosen']}" if m.get('fit_engine', {}).get('kind') == 'auto' else '')
               + ')')
    res = {'source': path.replace(chr(92), '/'), 'kind': auto.kind_of(m),
           'chosen': (m.get('fit_engine') or {}).get('chosen', kind), 'quality': q, 'floor': fl,
           'files': {k: P[k] for k in ('recipe', 'wav', 'png', 'model')}, 'code_renders_fit': same,
           'seconds': {'fit': round(t_fit, 2), 'total': round(time.time() - t0, 2)},
           'candidates': [{k: r.get(k) for k in ('candidate', 'score', 'fit_s', 'error')}
                          for r in (m.get('fit_engine') or {}).get('candidates', [])]}
    if report:
        import json
        with open(P['report'], 'w', encoding='utf-8') as fh:
            json.dump({**res, 'fit_engine': m.get('fit_engine'), 'summary': auto.summary(m)}, fh, indent=1,
                      default=float)
        res['files']['report'] = P['report']
    return res


def run_morph(path_a, path_b, amount=0.5, output=None, out_dir=None, render_path=None, clap=False, seed=0,
              fast=False, verbose=True):
    """`earshot fit --morph a b`: both fitted with one kind, a morph recipe (knob 'morph' in SPACE) + its render"""
    import numpy as np

    from . import auto, codegen
    from . import quality as Q

    class A:
        pass
    a = A()
    a.output, a.dir, a.render = output, out_dir, render_path
    sa = os.path.splitext(os.path.basename(path_a))[0]
    sb = os.path.splitext(os.path.basename(path_b))[0]
    P = _paths(a, f'{sa}__{sb}_morph'.replace('_fit', ''))
    if not output:
        P = {k: (v.replace('_morph_fit', '_morph') if isinstance(v, str) else v) for k, v in P.items()}
    ma, mb, kinds = auto.fit_pair(path_a, path_b, clap=clap, fast=fast, verbose=verbose)
    mm = auto.morph_model(ma, mb, amount)
    src = codegen.to_recipe(mm, name=P['name'], source=None, licence=(
        f'Licence: a morph of two recordings ({os.path.basename(path_a)}, {os.path.basename(path_b)}); keep both '
        'sources\' attribution / licence;'))
    os.makedirs(os.path.dirname(P['recipe']), exist_ok=True)
    with open(P['recipe'], 'w', encoding='utf-8') as fh:
        fh.write(src)
    rec = codegen.load_recipe(P['recipe'])
    y = rec.render_seed(seed)
    _write_wav(P['wav'], y)
    Xa, Xb = Q.load(path_a), Q.load(path_b)
    qa, qb = Q.quality(Xa, y, clap=clap), Q.quality(Xb, y, clap=clap)
    n = max(len(Xa), len(Xb), len(y))
    padt = lambda v: np.pad(Q.mono(v), (0, n - len(Q.mono(v))))
    Q.pair_png(P['png'], padt(Xa), padt(y), title=f'morph {amount:.2f}: {os.path.basename(path_a)} | morph | '
               f'{os.path.basename(path_b)}', extra=(os.path.basename(path_b), padt(Xb)))
    auto.save(mm, P['model'])
    return {'a': path_a, 'b': path_b, 'amount': amount, 'kind': auto.kind_of(ma), 'kinds_tried': kinds,
            'vs_a': qa, 'vs_b': qb, 'files': {k: P[k] for k in ('recipe', 'wav', 'png', 'model')}}


def add_cli(sub):
    from ...cli import common
    p = common(sub.add_parser('fit', help='translate a recording into code: fit a model, write a forge recipe, '
                                         'resynthesis + spectrogram + quality (analysis by synthesis)'))
    p.add_argument('file', nargs='?', help='the recording to fit')
    p.add_argument('-o', '--output', help='recipe path (.py); the WAV / PNG / model JSON go next to it')
    p.add_argument('--dir', help='output folder when -o is not given (default: current folder)')
    p.add_argument('--kind', default='auto', choices=('auto', 'modal', 'spectral', 'sines', 'gesture', 'hybrid'))
    p.add_argument('--layers', help='hybrid layers, e.g. sines,modal,spectral (default modal,spectral)')
    p.add_argument('--render', help='resynthesis WAV path (default next to the recipe)')
    p.add_argument('--report', action='store_true',
                   help='also measure the render-vs-render floor and CLAP cosine and write <name>.report.json')
    p.add_argument('--clap', action='store_true', help='let CLAP cosine take part in choosing the fit (loads CLAP)')
    p.add_argument('--describe', action='store_true', help='run earshot describe on the original and the resynthesis')
    p.add_argument('--target', help='critic description for the recipe (default: from the refset category / sound)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fast', action='store_true', help='cheaper fitter settings (previews)')
    p.add_argument('--takes', action='store_true',
                   help='a fit meant to make takes: score every candidate on seeds 0, 1, 2 (not seed 0 alone)')
    p.add_argument('--keep-bed', action='store_true',
                   help='fit a recording with a bed under it (music in its pre-roll) as it is; by default the bed is '
                        'stripped and the sound alone is fitted')
    p.add_argument('--morph', nargs=2, metavar=('A', 'B'),
                   help='fit two recordings with one kind and write a morph recipe')
    p.add_argument('--amount', type=float, default=0.5, help='with --morph: 0 = A, 1 = B (default 0.5)')
    p.add_argument('--forge', metavar='RECIPE_PY',
                   help='search variations of a fitted recipe file with the forge critic')
    p.add_argument('--budget', type=int, default=96, help='with --forge: candidates the critic scores')
    p.add_argument('--pop', type=int, default=16, help='with --forge: population per generation')
    p.add_argument('--variants', type=int, default=4, help='with --forge: diverse sounds to keep')
    p.add_argument('--refset', help="with --forge: the critic's refset category (default the recipe's REFSET)")
    p.add_argument('--quiet', action='store_true', help='do not print the candidates while fitting')

    def run(a):
        from ...report import emit
        from . import quality as Q
        if a.forge:
            from .. import forge
            from . import codegen
            rec = codegen.load_recipe(a.forge)
            refset = a.refset or getattr(rec, 'REFSET', None)
            refs = None if refset else (getattr(rec, 'REFS', None) or None)
            out = a.dir or os.path.splitext(a.forge)[0] + '_forge'
            r = forge(rec, refs=refs, refset=refset, budget=a.budget, pop=a.pop, keep=a.variants, seed=a.seed, out=out,
                      robust=3, prescreen=3 * a.pop, shortlist=2, strategy='es')
            lines = [r['summary']] + [f"  {p['name']}: {p['score'].get('summary')}" for p in r['picks']]
            lines.append(f'wrote {out}: WAVs, report.json, report.md, compare.png, audition.wav')
            emit({k: v for k, v in r.items() if k != 'history'}, a, text='\n'.join(lines))
            return 0
        if a.morph:
            r = run_morph(a.morph[0], a.morph[1], a.amount, a.output, a.dir, a.render, clap=a.clap, seed=a.seed,
                          fast=a.fast, verbose=not a.quiet)
            lines = [f"morph {a.amount:.2f} of {os.path.basename(a.morph[0])} and {os.path.basename(a.morph[1])} "
                     f"({r['kind']} models)",
                     f"  vs A: {Q.text(r['vs_a'])}", f"  vs B: {Q.text(r['vs_b'])}"]
            lines += [f'  wrote {k}: {v}' for k, v in r['files'].items()]
            emit(r, a, text='\n'.join(lines))
            return 0
        if not a.file:
            print('usage: earshot fit <file> [-o recipe.py] [--kind auto] [--render out.wav] [--report] | '
                  '--morph A B [-o morph.py] | --forge recipe.py [--dir out]')
            return 2
        layers = tuple(a.layers.split(',')) if a.layers else None
        r = run_fit(a.file, a.kind, layers, a.output, a.dir, a.render, a.report, a.clap, a.describe, a.seed, a.target,
                    a.fast, verbose=not a.quiet and not a.json, takes=a.takes, bed=not a.keep_bed)
        lines = [f"{os.path.basename(a.file)} -> {r['kind']} model" + (f" (auto chose {r['chosen']})"
                                                                      if a.kind == 'auto' else '')
                 + f" in {r['seconds']['fit']:.1f} s",
                 '  ' + Q.text(r['quality'], r['floor'])]
        if r['floor'] and r['floor'].get('mel_db') is not None:
            lines.append(f"  floor = two renders of the model with different seeds (a noise model cannot beat it)")
        lines.append(f"  the recipe's own render equals the fit render: {r['code_renders_fit']}")
        lines += [f'  wrote {k}: {v}' for k, v in r['files'].items()]
        emit(r, a, text='\n'.join(lines))
        return 0
    p.set_defaults(func=run)
