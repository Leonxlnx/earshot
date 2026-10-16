"""Fitted model -> a readable, self-contained forge recipe module (Python source).

    from earshot.forge.fit import auto, codegen
    m = auto.fit('click.wav')
    src = codegen.to_recipe(m, name='click_fit', source='click.wav', quality=q)   # q: quality dict (optional)
    open('click_fit.py', 'w').write(src)
    rec = codegen.load_recipe('click_fit.py')        # a module: TARGET, ALTS, SPACE, DEFAULTS, MODEL, render(p, r)
    y = rec.render(rec.DEFAULTS, dsp.rng(0))          # DEFAULTS re-render the fit; the knobs in SPACE vary it
    # search variations with the forge critic: earshot fit --forge click_fit.py --budget 96 --dir made/click_fit
    # (or copy the file into earshot/forge/recipes/ and run `earshot forge click_fit`)

The module holds: a docstring (what the fit found, the source and its licence note, the measured fit quality, how to
use it), TARGET / ALTS (the critic's description: the refset category's recipe when the source is a refset file,
else the category or a description from the measured character), REFSET / REFS (the critic's real references),
SPACE (the model kind's knobs with search ranges around the identity: auto.KNOBS), DEFAULTS (the identity: the
fit itself), MODEL (the fitted parameters as a readable Python literal, one mode / partial / keyframe per line), and
render(p, r) / render_seed(seed, **knobs), which call earshot.forge.fit.auto.render (the fitters' renderers: numpy /
scipy). A morph model makes a recipe whose SPACE has a 'morph' knob (where between the two sources), so a search
explores sounds between two recordings rather than copies of one.
"""
import importlib.util
import os
import re

import numpy as np

from . import auto

COMMENTS = {
    'strikes': 'one per onset: time s, contact ramp ms, modes, contact (attack) modes',
    'modes': '[Hz, decay time constant s, amplitude, phase rad] referenced at the strike',
    'attack': 'the contact: fast modes from the onset [Hz, tau s, amp, phase]',
    'noise': 'band noise: ERB-spaced bands (edges Hz), per band an envelope [[t s, dB], ...]',
    'sub': 'the residual under 20 Hz as a 100 Hz waveform (a pressure pulse) or None',
    'side': 'per noise band, side / mid amplitude ratio of the stereo original',
    'partials': 'strongest first: birth t0 s, rise s, last breakpoint t1 s, release s, phase, [t, Hz], [t, dBFS]',
    'transients': 'per strike: damped sinusoid atoms [Hz, tau s, amplitude, phase]',
    'residual': 'noise in 1/3-octave bands (Hz centres): dB envelopes at `times`',
    'env': 'the gesture: keyframe times s -> 48-band power density dB (re full-scale white noise)',
    'stereo': 'per band group and keyframe: L/R level difference dB, inter-channel correlation',
    'texture': 'what the smooth gesture does not hold: flutter (am) and crackle (grains)',
    'ridges': 'tonal ridges: keyframes of frequency Hz and level dB',
    'layers': 'fitted one after another on what the previous layers leave',
    'fit_engine': None,
}
DROP = ('fit_engine',)


def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return repr(bool(v))
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v != v or v in (float('inf'), float('-inf')):
            return f"float('{v}')"
        return repr(v)
    return repr(v)


def _is_scalar(v):
    return v is None or isinstance(v, (bool, int, float, str, np.integer, np.floating, np.bool_))


def _lit(v, ind, width=116):
    pad = ' ' * ind
    if isinstance(v, dict):
        if not v:
            return '{}'
        out = ['{']
        for k, x in v.items():
            s = _lit(x, ind + 4, width)
            c = COMMENTS.get(k)
            if c and '\n' in s:
                out.append(f'{pad}    # {c}')
            out.append(f'{pad}    {k!r}: {s},' + (f'  # {c}' if c and '\n' not in s else ''))
        out.append(pad + '}')
        return '\n'.join(out)
    if isinstance(v, (list, tuple)):
        if not v:
            return '[]'
        if all(_is_scalar(x) for x in v):
            items = [_scalar(x) for x in v]
            one = '[' + ', '.join(items) + ']'
            if len(one) + ind <= width:
                return one
            lines, cur = [], ''
            for it in items:
                if cur and len(cur) + len(it) + ind + 6 > width:
                    lines.append(cur)
                    cur = ''
                cur += it + ', '
            lines.append(cur)
            return '[\n' + '\n'.join(pad + '    ' + ln.rstrip() for ln in lines) + '\n' + pad + ']'
        return '[\n' + '\n'.join(pad + '    ' + _lit(x, ind + 4, width) + ',' for x in v) + '\n' + pad + ']'
    return _scalar(v)


def literal(model, name='MODEL'):
    """`name = {...}`: the model as a readable Python literal that evaluates to an equal dict (fit diagnostics left
    out)"""
    m = {k: v for k, v in model.items() if k not in DROP}
    return f'{name} = ' + _lit(m, 0)


def _category_of(path):
    """(refset category, split) of a registry file, else (None, None)"""
    if not path:
        return None, None
    try:
        from ..refsets import load
        reg = load()
    except Exception:
        return None, None
    p = os.path.normcase(os.path.abspath(path))
    for c, e in reg.get('categories', {}).items():
        for s in ('forge', 'bench'):
            if any(os.path.normcase(os.path.abspath(q)) == p for q in e.get(s, [])):
                return c, s
    return None, None


def critic_words(source=None, category=None, character=None):
    """(TARGET, ALTS, REFSET) for a fitted recipe: a forge recipe of the same refset category lends its TARGET /
    ALTS; else the category's description; else a description from the measured character"""
    if category is None:
        category, _ = _category_of(source)
    if category:
        try:
            from .. import load_recipe, recipes
            for nm in recipes():
                try:
                    r = load_recipe(nm)
                except Exception:
                    continue
                if getattr(r, 'REFSET', None) == category:
                    t = (getattr(r, 'TARGETS', None) or {}).get(category, r.TARGET)
                    return t, list(getattr(r, 'ALTS', []) or []), category
        except Exception:
            pass
        from ..refsets import CATEGORIES
        if category in CATEGORIES:
            return re.sub(r'\s*\(.*?\)\s*$', '', CATEGORIES[category]), [], category
    c = character or {}
    if c.get('gliding'):
        t = 'a short round pop with a pitch glide'
    elif c.get('tonal'):
        t = 'a bright ringing chime' if c.get('t40_ms', 0) > 400 else 'a short pitched ding'
    elif c.get('impulsive') and c.get('dur_s', 1) < 0.5:
        t = 'a short crisp click'
    elif c.get('impulsive'):
        t = 'a deep hit with a long tail'
    elif c.get('noisy'):
        t = 'an air whoosh'
    else:
        t = 'a sound effect'
    return t, [], None


def _space_lines(knobs):
    lines = ['SPACE = {']
    for n, (spec, ident, doc) in knobs.items():
        lines.append(f'    {n!r}: {tuple(spec)!r},   # {doc}')
    lines.append('}')
    return lines


def to_recipe(model, name='fitted', source=None, quality=None, floor=None, target=None, alts=None, refset=None,
              refs=None, licence=None):
    """the recipe module source (module doc). source: the recording's path (for the critic refs and the docstring);
    quality / floor: quality dicts measured on this model's render (written into the docstring)"""
    kind = auto.kind_of(model)
    knobs = auto.knobs_of(model)
    fe = model.get('fit_engine') or {}
    t, a, cat = critic_words(source, refset, fe.get('character'))
    target = target or t
    alts = a if alts is None else list(alts)
    refset = refset or cat
    if refs is None:
        refs = [os.path.abspath(source).replace(chr(92), '/')] if source and os.path.exists(source) else []
    from .quality import text as qtext
    summ = auto.summary(model)
    src_name = os.path.basename(source) if source else (model.get('source') or 'unknown')
    doc = [f'{name}: a forge recipe fitted from a real recording by `earshot fit` ({kind} model).', '']
    doc += [ln.rstrip() for ln in summ.splitlines()]
    doc += ['', f'Source: {src_name}' + (f' (refset {refset})' if refset else '') + '.',
            licence or ('Licence: a fit of a recording is a derivative of it. Keep the source\'s attribution / licence '
                        'with this file;'), 'prefer CC0 sources for shipped seeds (a morph of several sources is less '
                                            'derivative of any one).']
    if quality:
        doc += ['', 'Fit quality (original vs this recipe at DEFAULTS, seed 0):', '  ' + qtext(quality, floor)]
    doc += ['', 'Use:',
            '    from earshot.forge import dsp',
            '    y = render(DEFAULTS, dsp.rng(0))                     # one take of the fit (r = the take\'s noise)',
            '    y = render_seed(0)                                   # the fit as measured (seed 0)',
            f"    y = render(dict(DEFAULTS, {next(iter(knobs))}=...), dsp.rng(3))   # a variation",
            f'    earshot fit --forge {name}.py --budget 96 --dir made/{name}   # variations searched with the critic',
            f'    (or copy this file into earshot/forge/recipes/ and run `earshot forge {name}`)',
            'Knobs: SPACE (search ranges); DEFAULTS = the identity (the fit itself).']
    body = ['"""' + '\n'.join(doc).replace('"""', "'''").replace('\\', '/') + '\n"""',
            'import numpy as np', '', 'from earshot.forge.fit import auto as _engine', '',
            f'TARGET = {target!r}', f'ALTS = {alts!r}']
    if refset:
        body.append(f'REFSET = {refset!r}   # the critic\'s default refs: this refsets category (forge split)')
    body.append(f'REFS = {refs!r}')
    body.append(f'KIND = {kind!r}')
    body += [''] + _space_lines(knobs)
    body.append('DEFAULTS = ' + repr({n: v[1] for n, v in knobs.items()}) + '   # the identity: re-renders the fit')
    body += ['', literal(model), '', '',
             'def render(p, r):',
             '    """one take: p = knob values (missing keys take DEFAULTS), r = numpy Generator (noise / jitter)"""',
             '    q = dict(DEFAULTS)',
             '    q.update({k: v for k, v in p.items() if k in DEFAULTS})',
             '    return _engine.render(MODEL, seed=int(r.integers(0, 2 ** 31 - 1)), **q)', '', '',
             'def render_seed(seed=0, **knobs):',
             '    """render with an explicit seed: render_seed(0) is the fit as measured"""',
             '    q = dict(DEFAULTS)',
             '    q.update(knobs)',
             '    return _engine.render(MODEL, seed=seed, **q)', '', '',
             "if __name__ == '__main__':",
             '    import sys',
             '    import soundfile as sf',
             '    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0',
             f"    out = sys.argv[2] if len(sys.argv) > 2 else '{name}.wav'",
             "    sf.write(out, np.asarray(render_seed(seed), np.float32), 48000, subtype='FLOAT')",
             '    print(out)', '']
    return '\n'.join(body)


def write_recipe(path, model, **kw):
    """to_recipe() written to `path` (the module name defaults to the file stem); returns the path"""
    kw.setdefault('name', re.sub(r'\W', '_', os.path.splitext(os.path.basename(path))[0]))
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(to_recipe(model, **kw))
    return path


def load_recipe(path):
    """import a recipe module from a file path (a fitted recipe outside earshot/forge/recipes/)"""
    name = re.sub(r'\W', '_', os.path.splitext(os.path.basename(path))[0])
    spec = importlib.util.spec_from_file_location(f'earshot_fit_recipe_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__name__ = name
    return mod
