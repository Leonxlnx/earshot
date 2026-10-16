"""The forge report schema: every delivered sound re-renders from its report.json.

    from earshot.forge import schema
    rows = schema.check_final('out/forge_v2/pro_shimmer/final')          # re-render every pick, compare to its WAV
    x = schema.rerender(schema.block(report, report['picks'][0]))       # one pick, at its written level
    schema.upgrade('out/forge_v2/pop/final/report.json', write=True)    # add the render blocks to an older report
    python -m earshot.forge.schema out/forge_v2/*/final [--upgrade --write]

A pick's `render` block (schema 2, written by forge() / palette() since round 6) holds everything a re-render needs:
    recipe   the module in earshot.forge.recipes that defines the render function ('typing')
    fn       the render function's name in that module ('render_run' for typing.RUN, 'render' for most)
    member   the family member / preset / character name (None when the pick is no family member)
    params   the complete dict the render function was called with, filled keys included (the stand-in recipe's
             DEFAULTS, a family member's own values, the keys a final pass filled), so a later change of a
             recipe's DEFAULTS cannot change the re-render
    seed     the noise seed (dsp.rng(seed))
    mode     the critic mode the pick was chosen in ('designed' | 'real' | ...; None when unknown)
    peak_db  every render is peak-normalised to this level first (search.render: -1 dBFS)
    gain_db  the gain after that (palette loudness match; 0 for a plain forge() pick), rounded to 0.01 dB
    gain     the exact linear gain when the writer knew it (palette()); else 10 ** (gain_db / 20) is the gain
The written WAV = float32(normalise(fn(params, dsp.rng(seed)), peak_db) * gain) as PCM_24, so a re-render matches
it to one 24-bit step (1.2e-7).

Reports written before round 6 have no render block; block() infers it (legacy_block(): the report's
`recipe_module` or `recipe`, or for the drivers' stand-in names the LEGACY table; a 'variant' pick's own
family_params or the report's family members; the pick's gain_db). upgrade() writes the inferred blocks into the
report after re-rendering every pick against its WAV (nothing is written unless every pick matches), with params
filled from the render function's DEFAULTS only where that renders bit-identically.
"""
import importlib
import json
import os
import sys

import numpy as np

SCHEMA = 2
TOL = 1e-6
PEAK_DB = -1.0
BLOCK_KEYS = ('recipe', 'fn', 'member', 'params', 'seed', 'mode', 'peak_db', 'gain_db', 'gain')
_PKG = 'earshot.forge.recipes'

# reports whose `recipe` names a driver's stand-in object (its __name__), not a module: (module, render function)
LEGACY = {'pro_glitch': ('glitch', 'render'), 'pro_pop': ('pop', 'render'), 'pro_riser': ('riser', 'render'),
          'pro_shimmer': ('shimmer', 'render'), 'pro_sub_drop': ('subdrop', 'render'),
          'sub_drop': ('subdrop', 'render'), 'pro_typing_key': ('typing', 'render'),
          'pro_typing_run': ('typing', 'render_run'), 'typing_key': ('keypress', 'render')}


class NotReproducible(ValueError):
    """a pick whose render cannot be resolved from its report"""


def _module(name):
    return importlib.import_module(f'{_PKG}.{name}')


def _short(modname):
    return modname[len(_PKG) + 1:] if modname and modname.startswith(_PKG + '.') else None


def render_ref(fn):
    """(module short name, function name) for a render function defined at module level in earshot.forge.recipes
    (functools.partial unwrapped), else None (a lambda / closure: the writer must say what it rendered)"""
    fn = getattr(fn, 'func', fn)
    short = _short(getattr(fn, '__module__', None))
    name = getattr(fn, '__name__', None)
    if not short or not name or name == '<lambda>':
        return None
    try:
        return (short, name) if getattr(_module(short), name, None) is fn else None
    except ImportError:
        return None


def resolve(block):
    """the render function a block names"""
    try:
        fn = getattr(_module(block['recipe']), block.get('fn') or 'render')
    except (ImportError, AttributeError, KeyError, TypeError) as e:
        raise NotReproducible(f"render {block.get('recipe')}.{block.get('fn')} not found: {e}") from e
    return fn


def make_block(fn, params, seed, member=None, mode=None, gain=None, gain_db=None, peak_db=PEAK_DB, ref=None):
    """a pick's render block (module doc); fn = the render function (or ref=(module, fn name) when fn is a stand-in's
    lambda); gain = the exact linear gain (gain_db is derived from it when not given). None when the render function
    cannot be named."""
    ref = ref or render_ref(fn)
    if ref is None:
        return None
    g = None if gain is None else float(gain)
    gdb = gain_db if gain_db is not None else (round(float(20 * np.log10(g)), 2) if g else 0.0)
    b = {'recipe': ref[0], 'fn': ref[1], 'member': None if member is None else str(member), 'params': dict(params),
         'seed': int(seed), 'mode': mode, 'peak_db': float(peak_db), 'gain_db': float(gdb)}
    if g is not None:
        b['gain'] = g
    return b


def _mode(report):
    c = report.get('critic')
    m = c.get('mode') if isinstance(c, dict) else None
    m = m or report.get('mode')
    return m if isinstance(m, str) else None


def _member(pick):
    for k in ('member', 'family_name', 'family', 'preset'):
        v = pick.get(k)
        if isinstance(v, (str, int)) and not isinstance(v, bool):
            return str(v)
    return str(pick['family_member']) if pick.get('family_member') is not None else None


def legacy_block(report, pick):
    """infer the render block of a report written before round 6 (module doc)"""
    rec = report.get('recipe')
    if report.get('recipe_module'):
        mod, fn = report['recipe_module'], 'render'
    elif rec in LEGACY:
        mod, fn = LEGACY[rec]
    else:
        mod, fn = rec, 'render'
    try:
        _module(mod)
    except (ImportError, TypeError) as e:
        raise NotReproducible(f'recipe {rec!r} is no module of {_PKG} and not in schema.LEGACY') from e
    params = dict(pick.get('params') or {})
    if set(params) == {'variant'}:                         # a driver's final pass over hand-made members
        if pick.get('family_params'):
            params = dict(pick['family_params'])
        else:
            members = (report.get('family') or {}).get('members') if isinstance(report.get('family'), dict) else None
            if not members:
                raise NotReproducible(f"{pick.get('name')}: params are only a 'variant' index and the report has "
                                      'neither family_params nor family members')
            params = dict(members[int(params['variant'])])
    member = _member(pick)
    fam = report.get('family')
    labels = fam.get('labels') if isinstance(fam, dict) else None
    if member is None and isinstance(labels, list):        # the digital drivers: family.labels[i] names pick i
        i = next((k for k, q in enumerate(report.get('picks') or []) if q is pick), None)
        if i is not None and i < len(labels) and labels[i] is not None:
            member = str(labels[i])
    return make_block(None, params, pick['seed'], member=member, mode=_mode(report),
                      gain_db=float(pick.get('gain_db') or 0.0), ref=(mod, fn))


def block(report, pick):
    """the pick's render block: its own (schema 2) or the inferred one (legacy_block)"""
    b = pick.get('render')
    return dict(b) if isinstance(b, dict) and b.get('recipe') else legacy_block(report, pick)


def raw(b):
    """the render function's output for a block (float64, before normalisation); None if it fails or is silent"""
    from . import dsp
    x = np.asarray(resolve(b)(dict(b['params']), dsp.rng(b['seed'])), np.float64)
    if not len(x) or not np.isfinite(x).all() or np.abs(x).max() < 1e-7:
        return None
    return x


def level(x, b):
    """a raw render at the block's written level: peak_db normalisation (as search.render), then the gain"""
    from . import dsp
    x = dsp.normalize(x, float(b.get('peak_db', PEAK_DB))).astype(np.float32)
    g = b.get('gain')
    g = float(g) if g is not None else 10 ** (float(b.get('gain_db') or 0.0) / 20)
    return (x.astype(np.float64) * g).astype(np.float32) if g != 1.0 else x


def rerender(b):
    """re-render one block at its written level (float32, mono (n,) or (n, ch)); NotReproducible when it fails"""
    x = raw(b)
    if x is None:
        raise NotReproducible(f"{b.get('recipe')}.{b.get('fn')} failed or was silent")
    return level(x, b)


def compare(x, y):
    """max |x - y| over every sample and channel (x a render, y a WAV read with always_2d); inf when the shapes
    differ (a mono render matches a 1-channel WAV)"""
    x = np.asarray(x, np.float64)
    x = x[:, None] if x.ndim == 1 else x
    y = np.asarray(y, np.float64)
    if x.shape != y.shape:
        return float('inf')
    return float(np.abs(x - y).max()) if x.size else 0.0


def _wav(folder, pick):
    f = pick.get('file')
    return os.path.join(folder, os.path.basename(f)) if f else os.path.join(folder, pick['name'] + '.wav')


def _load(report):
    if isinstance(report, str):
        path = report if report.endswith('.json') else os.path.join(report, 'report.json')
        with open(path, encoding='utf-8') as fh:
            return json.load(fh), os.path.dirname(os.path.abspath(path))
    return report, None


def check_final(report, folder=None, tol=TOL, workers=None):
    """re-render every pick of a final (a folder, a report.json path or a report dict + folder) from its report and
    compare with the delivered WAV. Returns rows {'name', 'file', 'err', 'ok', 'recipe', 'fn', 'member', 'source'}
    ('source': 'block' = the report's own render block, 'legacy' = inferred); err inf = shape differs / missing."""
    import soundfile as sf
    from .search import _workers
    report, f0 = _load(report)
    folder = folder or f0

    def one(pk):
        row = {'name': pk.get('name'), 'file': os.path.basename(_wav(folder, pk)),
               'source': 'block' if isinstance(pk.get('render'), dict) else 'legacy'}
        try:
            b = block(report, pk)
            row.update(recipe=b['recipe'], fn=b['fn'], member=b.get('member'))
            if not os.path.exists(_wav(folder, pk)):
                row.update(err=float('inf'), ok=False, why='WAV missing')
                return row
            y, _ = sf.read(_wav(folder, pk), always_2d=True)
            row['err'] = compare(rerender(b), y)
        except NotReproducible as e:
            row.update(err=float('inf'), why=str(e))
        row['ok'] = bool(row['err'] <= tol)
        return row
    picks = list(report.get('picks') or [])
    w = _workers(workers)
    if w > 1 and len(picks) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(w) as ex:
            return list(ex.map(one, picks))
    return [one(pk) for pk in picks]


def _defaults_for(b):
    """candidate DEFAULTS tables for filling a block's params: the module's DEFAULTS first when fn is its render,
    then the DEFAULTS of every module-level object whose render is fn (typing.RUN, pop.V2), then the module's other
    *DEFAULTS* dicts"""
    mod = _module(b['recipe'])
    fn = resolve(b)
    out = []
    if b['fn'] == 'render' and isinstance(getattr(mod, 'DEFAULTS', None), dict):
        out.append(mod.DEFAULTS)
    for v in vars(mod).values():
        if getattr(v, 'render', None) is fn and isinstance(getattr(v, 'DEFAULTS', None), dict):
            out.append(v.DEFAULTS)
    for k, v in vars(mod).items():
        if k.endswith('DEFAULTS') and isinstance(v, dict):
            out.append(v)
    seen, uniq = set(), []
    for d in out:
        if id(d) not in seen:
            seen.add(id(d))
            uniq.append(d)
    return uniq


def fill(b, x=None):
    """the block with its params completed from the render function's DEFAULTS tables (_defaults_for), one table at
    a time, each step kept only when the filled params render bit-identically (a table the render does not merge
    that way is skipped). x = the block's raw render when already made."""
    x = raw(b) if x is None else x
    if x is None:
        return b
    cur = b
    for d in _defaults_for(b):
        missing = {k: v for k, v in d.items() if k not in cur['params']}
        if not missing:
            continue
        c = {**cur, 'params': {**missing, **cur['params']}}
        try:
            xc = raw(c)
        except Exception:
            continue
        if xc is not None and xc.shape == x.shape and np.array_equal(xc, x):
            cur = c
    return cur


def upgrade(report, write=False, tol=TOL, folder=None, log=print):
    """add a render block (filled params) and 'schema': 2 to every pick of a final's report (a folder / report.json
    path, or a dict + folder) that lacks one, after re-rendering each pick against its WAV. Returns (report, rows);
    writes report.json only with write=True and when every pick matches within tol (a pick that does not keeps the
    report untouched: fix the recipe's render path first)."""
    import soundfile as sf
    rep, f0 = _load(report)
    folder = folder or f0
    rows = []
    for pk in rep.get('picks') or []:
        has = isinstance(pk.get('render'), dict)
        b = block(rep, pk)
        x = raw(b)
        bf = b if has else fill(b, x)
        y, _ = sf.read(_wav(folder, pk), always_2d=True)
        err = compare(level(x, bf), y) if x is not None else float('inf')   # fill() kept bf bit-identical to b
        rows.append({'name': pk.get('name'), 'err': err, 'ok': err <= tol, 'filled': len(bf['params']) - len(b['params']),
                     'had_block': has})
        if not has:
            pk['render'] = bf
    ok = all(r['ok'] for r in rows)
    if ok:
        rep['schema'] = SCHEMA
    if write:
        if not ok:
            raise NotReproducible('not written: ' + ', '.join(f"{r['name']} err {r['err']:.2e}" for r in rows
                                                               if not r['ok']))
        path = os.path.join(folder, 'report.json')
        tmp = path + f'.tmp{os.getpid()}'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        os.replace(tmp, path)
        if log:
            log(f'wrote {path}: {len(rows)} render blocks (schema {SCHEMA})')
    return rep, rows


def main(argv=None):
    import argparse
    import glob
    ap = argparse.ArgumentParser(prog='python -m earshot.forge.schema',
                                 description='re-render finals from their report.json and compare with their WAVs')
    ap.add_argument('dirs', nargs='+', help='final folders (globs work)')
    ap.add_argument('--upgrade', action='store_true', help='add render blocks (schema 2) where missing')
    ap.add_argument('--write', action='store_true', help='with --upgrade: write report.json (only when all match)')
    ap.add_argument('--tol', type=float, default=TOL)
    a = ap.parse_args(argv)
    dirs = sorted({d for p in a.dirs for d in (glob.glob(p) or [p]) if os.path.exists(os.path.join(d, 'report.json'))})
    bad = 0
    for d in dirs:
        if a.upgrade:
            try:
                _, rows = upgrade(d, write=a.write, tol=a.tol)
            except NotReproducible as e:
                print(f'{d}: {e}')
                bad += 1
                continue
        else:
            rows = check_final(d, tol=a.tol)
        worst = max((r['err'] for r in rows), default=0.0)
        n_ok = sum(1 for r in rows if r['ok'])
        bad += n_ok < len(rows)
        src = sorted({r['source'] if 'source' in r else ('had a block' if r.get('had_block') else 'block added' if a.write else 'block inferred, not written')
                      for r in rows})
        print(f"{d}: {n_ok} / {len(rows)} re-render within {a.tol:g} (max err {worst:.2e}; {'/'.join(src)})"
              + ''.join(f"\n    {r['name']}: err {r['err']:.2e} {r.get('why', '')}" for r in rows if not r['ok']))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
