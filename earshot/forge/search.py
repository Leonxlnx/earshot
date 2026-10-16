"""The listen-and-improve loop: a search over a recipe's parameters, scored by the Critic.

    from earshot.forge.search import optimize
    res = optimize(recipe, critic, budget=160, pop=16, seed=0)                         # v1 evolution strategy
    res = optimize(recipe, critic, budget=160, pop=16, strategy='cma', prescreen=48)   # v2: CMA-ES + model-free pre-screen
    res['best'][0]['params'], res['best'][0]['score']['summary'], res['history'], res['gens'], res['timing']

A recipe is a module (or object) with
    SPACE    {name: (lo, hi)} | {name: (lo, hi, 'log')} | {name: (lo, hi, 'int')} | {name: [choice, ...]}
    DEFAULTS {name: value}            (a sensible starting sound; optional)
    render(params, r) -> np.ndarray   (48 kHz mono or stereo float; r = numpy Generator for natural variation)
Candidates are peak-normalised to -1 dBFS (so level never decides) and scored in batches. Deterministic for a given
seed (renders run on `workers` threads, default 2, each on its own seed: the same samples as a serial run). Every
scored candidate is kept in the history, so an agent can read what helped.

Strategies (`budget` always counts critic-scored candidates):
  'es'   (v1, the default) elites mutated with a shrinking step (sigma0 -> sigma1) plus a few fresh random ones.
  'cma'  separable CMA-ES in the unit cube (diagonal covariance, mirrored at the bounds) with restarts: when the step
         collapses or the best has not moved for `patience` generations it restarts from the strongest unexplored
         elite, then from random points, with the step reset (initial step 0.15, `cma=` options).
Measured (keypress, 2 render threads): at 96-240 scored candidates CMA did NOT beat the v1 ES - on a model-free proxy
objective (the pre-screen score, 3-5 seeds) ES -1.23 vs CMA -1.26..-1.36 at 96, -1.136 vs -1.154 at 240; with the real
critic (2 seeds, 96) best 6.55 vs 6.03, top-5 mean 5.77 vs 5.30. Elitist and small-mu CMA variants were worse. The
pre-screen is what helps: ES + prescreen 48 raised the passing candidates from 10.5 to 36.5 of 96 and the top-5 mean
from 5.77 to 6.17 (CMA + prescreen: 25.5 and 5.92). So 'es' + prescreen is the default the CLI uses.
`prescreen=K` (K > pop): render K candidates per generation, rank them with model-free measurements against the
critic's reference recordings (Prescreen: 1/3-octave tonal balance in units of the refs' own spread, envelope stats,
artifact checks) and embed only the best `pop` with CLAP. The model checks are the expensive part (CLAP is ~70 % of a
generation), the measurements cost ~10 ms a sound, and they are the signal that still separates made from real once
the model checks saturate. `reps=N` renders each scored candidate on N seeds and ranks it by the mean (robust to lucky
renders). Starting points: the recipe's DEFAULTS, `init` (e.g. a seed bank, `load_seeds()`) are scored in generation 0,
exactly as given; a SPACE key a start leaves out takes the recipe's DEFAULTS (the object's own, then those of the module
that defines its render: recipe_defaults()), clipped into the space if they lie outside it, and the spec's mid-range
only when no DEFAULTS has the key (a MidRangeFill warning; res['fill_warnings'] lists every such value). Before round 5
a missing key always got the mid-range (search._encode's 0.5): a hidden value (pro_impact round 3: snap 0.75).

`space=` restricts the search to a family (a narrower {name: spec} dict; parameters it leaves out keep the recipe's
DEFAULTS). `robust()` re-renders a result on new noise seeds (the best single render of a search is a winner's curse:
judge finals by their robust mean). `diverse()` picks strong results that differ in parameters AND in sound (tonal
balance + envelope). The seed bank (earshot/forge/seeds/<recipe>.json, small, committed) keeps good parameter sets
found so far: `load_seeds()` / `save_seeds()`.
"""
import json
import math
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import dsp

SR = 48000
SEEDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seeds')


class MidRangeFill(UserWarning):
    """a start point's SPACE key had no value in the start or in any DEFAULTS and was set to the spec's mid-range"""


# ---------------------------------------------------------------------------------------------- parameters
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


def _inside(spec, v):
    if isinstance(spec, list):
        return v in spec
    try:
        return spec[0] <= v <= spec[1] and (len(spec) < 3 or spec[2] != 'int' or float(v).is_integer())
    except TypeError:
        return False


def _clip(spec, v):
    """the nearest value inside a numeric spec (None for a choice list or a value that is not a number)"""
    if isinstance(spec, list):
        return None
    try:
        lo, hi = float(spec[0]), float(spec[1])
        c = min(max(float(v), lo), hi)
    except (TypeError, ValueError):
        return None
    return int(round(c)) if len(spec) > 2 and spec[2] == 'int' else c


def midrange(spec):
    """the value a unit-cube 0.5 decodes to for one SPACE spec (what a key missing from a start point used to get)"""
    return _decode({'k': spec}, [0.5])['k']


def recipe_defaults(recipe):
    """the values a key missing from a start point falls back to: the recipe object's own DEFAULTS over the DEFAULTS
    of the module that defines its render(). A stand-in object (a family member whose DEFAULTS is one preset, a
    driver's final-pass object whose DEFAULTS is its first member) still reaches the recipe module's DEFAULTS."""
    own = dict(getattr(recipe, 'DEFAULTS', None) or {})
    fn = getattr(recipe, 'render', None)
    fn = getattr(fn, 'func', fn)                          # functools.partial
    mod = sys.modules.get(getattr(fn, '__module__', None) or '')
    if mod is not None and mod is not recipe and isinstance(getattr(mod, 'SPACE', None), dict):
        md = getattr(mod, 'DEFAULTS', None)
        if isinstance(md, dict):
            return {**md, **own}
    return own


def fill_start(space, start, defaults, own=None):
    """complete one start point for the searched `space`: each SPACE key keeps the start's own value, else the
    recipe's DEFAULTS (`defaults`, recipe_defaults()), clipped into the space when it lies outside, and only when
    neither has a value the spec's mid-range. Returns (full start dict, notes): notes = [{'key', 'value', 'from'}] for
    every value not taken from the start or the object's own DEFAULTS (`own`): 'module DEFAULTS', 'DEFAULTS clipped'
    (a default outside the space: the nearest value inside, for a choice the list does not hold its middle entry),
    'mid-range' (no value anywhere: a hidden value; round 4 found one that put a 0 dB onset tick into pro_impact) or
    'start clipped' (a start value outside the space: the search scores the decoded, clipped value as before)."""
    full, notes = dict(start), []
    own = own if own is not None else defaults
    for k, spec in space.items():
        v = start.get(k)
        src = 'start'
        if v is None:
            v, src = defaults.get(k), ('DEFAULTS' if own.get(k) is not None else 'module DEFAULTS')
        if v is None:
            full[k] = midrange(spec)
            notes.append({'key': k, 'value': full[k], 'from': 'mid-range'})
            continue
        if not _inside(spec, v):
            to = _clip(spec, v)
            if to is None:                                 # a choice the space's list does not hold: the list is
                to = midrange(spec)                        # the author's narrower choice, take its middle entry
            if src == 'start':
                notes.append({'key': k, 'value': to, 'from': 'start clipped', 'was': v})
                continue                                   # the search keeps today's rule: decoded (clipped)
            full[k] = to
            notes.append({'key': k, 'value': to, 'from': 'DEFAULTS clipped', 'was': v})
            continue
        full[k] = v
        if src == 'module DEFAULTS':
            notes.append({'key': k, 'value': v, 'from': src})
    return full, notes


def _mirror(x):
    """fold any real vector into [0, 1] by mirroring at the bounds (keeps the sampling distribution's shape)"""
    r = np.mod(x, 2.0)
    return np.where(r > 1.0, 2.0 - r, r)


# ---------------------------------------------------------------------------------------------- rendering
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


def _workers(workers):
    if workers is None:
        workers = int(os.environ.get('EARSHOT_THREADS') or 2)
    return max(1, min(int(workers), 8))


def render_many(recipe, jobs, workers=None):
    """render [(params, seed), ...] on `workers` threads (default EARSHOT_THREADS or 2); same order and samples as
    a serial loop (each render owns its seed). Returns a list of arrays / None."""
    jobs = list(jobs)
    w = _workers(workers)
    if w == 1 or len(jobs) < 2:
        return [render(recipe, p, s) for p, s in jobs]
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(lambda j: render(recipe, j[0], j[1]), jobs))


# ---------------------------------------------------------------------------------------------- model-free ears
ENV_KEYS = ('attack_ms', 't10_ms', 't20_ms', 't40_ms', 'centroid_ms', 'crest_db', 'end_db')
_DB_KEYS = 2                                       # the last two are levels in dB (6 dB units), the rest log times


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def envelope_stats(x, hop_ms=2.0):
    """envelope measurements of one sound (mono sum): attack (onset at -30 dB to peak), t10 / t20 / t40 (peak until the
    2 ms RMS envelope first falls 10 / 20 / 40 dB under it, or the file end), temporal centroid from the onset, and
    the crest factor (peak / RMS of the part within 40 dB of the peak), end_db (mean level of the last quarter of the
    file re the peak, floored at -90: a noise-floor plateau where real edits fade to silence), all in ms or dB"""
    m = _mono(x)
    n = max(1, int(SR * hop_ms / 1000))
    k = len(m) // n
    if k < 2:
        return {key: (-90.0 if key == 'end_db' else 0.0) for key in ENV_KEYS}
    pw = (m[:k * n].reshape(k, n) ** 2).mean(1)
    e = 10 * np.log10(pw + 1e-18)
    ip = int(np.argmax(e))
    pk = e[ip]
    on = int(np.argmax(e > pk - 30))
    out = {'attack_ms': (ip - on) * hop_ms + hop_ms / 2}
    for db in (10, 20, 40):
        idx = np.nonzero(e[ip:] < pk - db)[0]
        out[f't{db}_ms'] = float((idx[0] if len(idx) else k - ip) * hop_ms)
    w = pw[on:]
    out['centroid_ms'] = float((np.arange(len(w)) * w).sum() / (w.sum() + 1e-18) * hop_ms + hop_ms / 2)
    act = m[on * n:k * n][np.repeat(e[on:] > pk - 40, n)]
    out['crest_db'] = float(20 * np.log10(np.abs(m).max() / (np.sqrt((act ** 2).mean()) + 1e-12) + 1e-12)) \
        if len(act) else 0.0
    out['end_db'] = float(max(-90.0, 10 * np.log10(pw[int(0.75 * k):].mean() + 1e-18) - pk))
    return out


def artifacts(x):
    """cheap artifact checks on one sound: {'flags': {name: value}, 'penalty': float}. flat_top (clipped runs at the
    peak), dc (mean offset vs RMS), start_jump (the first sample already loud: an unfaded click), cut_end (the last
    5 ms within 45 dB of the peak: a tail cut off), gap (digital silence inside the sound), ultrasonic (energy share
    above 20 kHz). The fallback when earshot.forge.polish (finer detectors, ref-relative) is not available."""
    m = _mono(x)
    flags, pen = {}, 0.0
    pk = np.abs(m).max() + 1e-12
    flat = np.concatenate([[False], np.abs(np.diff(m)) <= 1e-6 * pk])   # clipped samples repeat exactly; a 60 Hz
    hot = (np.abs(m) > 0.98 * pk) & flat                                  # sine near its crest does not
    if hot.sum() >= 4:
        d = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
        runs = np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]
        if runs.max() >= 4:
            flags['flat_top'] = int(runs.max())
            pen += 1.0
    rms = np.sqrt((m ** 2).mean()) + 1e-12
    if abs(m.mean()) / rms > 0.1:
        flags['dc'] = round(float(m.mean() / rms), 3)
        pen += 0.5
    if abs(m[0]) / pk > 0.05:
        flags['start_jump'] = round(float(abs(m[0]) / pk), 3)
        pen += 0.5
    tail = m[-int(0.005 * SR):]
    end_db = 20 * np.log10(np.sqrt((tail ** 2).mean()) / pk + 1e-12)
    if end_db > -45:
        flags['cut_end'] = round(float(end_db), 1)
        pen += 1.0
    z = (m == 0).astype(np.int8)
    if z.any():
        d = np.diff(np.concatenate([[0], z, [0]]))
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        inner = [(a, b) for a, b in zip(st, en) if a > 0 and b < len(m) and b - a > 0.005 * SR]
        if inner:
            flags['gap'] = len(inner)
            pen += 1.0
    F = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    us = 10 * np.log10(F[f > 20000].sum() / (F.sum() + 1e-18) + 1e-12)
    if us > -25:
        flags['ultrasonic'] = round(float(us), 1)
        pen += 0.5
    return {'flags': flags, 'penalty': round(pen, 3)}


def describe(x):
    """the model-free descriptor of one sound: 1/3-octave profile (earshot.forge.tone), envelope stats, artifacts"""
    from .tone import profile
    return {'profile': profile(x), 'env': envelope_stats(x), 'art': artifacts(x)}


def _envvec(env):
    return np.array([math.log(max(env[k], 0.5)) for k in ENV_KEYS[:-_DB_KEYS]]
                    + [env.get(k, -90.0 if k == 'end_db' else 0.0) / 6.0 for k in ENV_KEYS[-_DB_KEYS:]])


def sound_dist(a, b):
    """distance between two descriptors in dB-like units: RMS 1/3-octave difference plus 4 dB per e-fold of envelope
    timing difference (RMS over attack / t10 / t20 / t40 / centroid, crest and end level in 6 dB units)"""
    pa, pb = a['profile'], b['profile']
    return float(np.sqrt(np.mean((pa - pb) ** 2)) + 4.0 * np.sqrt(np.mean((_envvec(a['env']) - _envvec(b['env'])) ** 2)))


class Prescreen:
    """ranks sounds against reference recordings with measurements only (no model), cheap enough to run on many
    renders before the critic embeds a few. score(xs) -> rows {'pre', 'tone_db', 'tone_ratio', 'env_pen', 'env_z',
    'art', 'desc'[, 'polish']}; pre (higher is better) = -(tone_ratio + env_w * env_pen + artifacts), env_pen = mean
    over env_keys of the robust z beyond 1.5 (both directions: a tail that dies too fast counts too). artifacts =
    earshot.forge.polish's art_pen + 0.75 polish_pen + 1 per hard artifact, as the critic weighs them (`polish_ref=`
    the critic's PolishRef, else one built from the refs; ~45 ms a sound), or this module's artifacts() without polish.
    Measured against the critic (tone=1.5) on random candidates around DEFAULTS: without polish, Spearman 0.60-0.63
    (keypress, 96), picking 16 of 48 raised the mean critic total in all four halves tested; with polish vs critic
    v2 (72 candidates, 12 of 36 picked): whoosh Spearman 0.53 -> 0.69, selected mean 6.66 -> 7.71 and 4.35 -> 6.02
    (critic's best kept in both halves); keypress 0.60 -> 0.71, 1.43 -> 2.40 and -1.99 -> 0.56 (best kept in one
    half). Without refs only the artifact checks rank. study= {key: (lo, hi)} (the critic's absolute fallback
    targets, critic.STUDY_TARGETS, for a category with fewer than 3 refs; widened to the refs' own values): its
    tone / env / polish penalties are subtracted from pre too, as the critic counts them."""

    def __init__(self, refs=None, tone_target=None, env_w=0.5, env_keys=('t40_ms', 'crest_db', 'end_db'),
                 polish_ref=None, polish=True, study=None):
        from .. import audio
        xs = [audio.load(r) if isinstance(r, str) else np.asarray(r) for r in (refs or [])]
        self.study = None
        if study and len(xs) < 3:
            from .critic import study_measures, widen_targets
            self.study = widen_targets(study, [study_measures(x) for x in xs])
        self.pol = polish_ref
        if self.pol is None and polish:
            try:
                from .polish import PolishRef
                self.pol = PolishRef(xs)
            except Exception:                         # polish missing or broken: the built-in checks rank alone
                self.pol = None
        self.n = len(xs)
        self.tone_target, self.env_w = tone_target, float(env_w)
        self.env_idx = [ENV_KEYS.index(k) for k in env_keys]
        self.descs = [describe(x) for x in xs]
        if self.n:
            P = np.stack([d['profile'] for d in self.descs])
            self.med = np.median(P, 0)
            if self.n >= 3:
                loo = [float(np.sqrt(np.mean((P[i] - np.median(np.delete(P, i, 0), 0)) ** 2))) for i in range(self.n)]
                self.spread = max(float(np.median(loo)), 0.5)
            else:
                self.spread = 4.0
            if tone_target is not None:
                self.spread = float(tone_target)
            V = np.stack([_envvec(d['env']) for d in self.descs])
            self.env_med = np.median(V, 0)
            q1, q3 = np.percentile(V, [25, 75], axis=0)
            floor = np.array([0.2] * (len(ENV_KEYS) - _DB_KEYS) + [2.0 / 6.0] * _DB_KEYS)
            self.env_sd = np.maximum((q3 - q1) / 1.35, floor)

    def row(self, x, polish=True):
        d = describe(x)
        r = {'art': d['art'], 'desc': d}
        art = d['art']['penalty']
        if polish and self.pol is not None:
            try:
                pm = self.pol.check(x)
                r['polish'] = {'art_pen': round(float(pm['art_pen']), 3), 'polish_pen': round(float(pm['polish_pen']), 3),
                               'hard': list(pm.get('hard') or [])}
                art = r['polish']['art_pen'] + 0.75 * r['polish']['polish_pen'] + 1.0 * len(r['polish']['hard'])
            except Exception:
                pass
        if self.study:
            from .critic import study_measures, study_penalties
            pen, off, _ = study_penalties(study_measures(x), self.study)
            r['study'] = off
            art += pen['tone'] + pen['env'] + 0.75 * pen['polish']
        if self.n:
            r['tone_db'] = round(float(np.sqrt(np.mean((d['profile'] - self.med) ** 2))), 2)
            r['tone_ratio'] = round(r['tone_db'] / self.spread, 3)
            z = (_envvec(d['env']) - self.env_med) / self.env_sd
            r['env_z'] = {k: round(float(v), 2) for k, v in zip(ENV_KEYS, z)}
            r['env_pen'] = round(float(np.mean(np.maximum(0.0, np.abs(z[self.env_idx]) - 1.5))), 3) if self.env_idx else 0.0
            r['pre'] = round(-(r['tone_ratio'] + self.env_w * r['env_pen'] + art), 3)
        else:
            r['pre'] = round(-art, 3)
        return r

    def score(self, xs, polish=True):
        return [self.row(x, polish) if x is not None else {'pre': -np.inf, 'art': {'flags': {'silent': 1}, 'penalty': 9.0}}
                for x in xs]

    def nearest(self, desc):
        """index of the nearest reference to a descriptor (sound_dist), or None without refs"""
        if not self.n:
            return None
        return int(np.argmin([sound_dist(desc, r) for r in self.descs]))


def prescreen_for(critic):
    """a Prescreen built from a critic's references (reuses nothing model-based; None-safe for critics without refs)"""
    return Prescreen(getattr(critic, 'refs', None) or [], tone_target=getattr(critic, 'tone_target', None),
                     polish_ref=getattr(critic, 'polish_ref', None), study=getattr(critic, 'study', None))


# ---------------------------------------------------------------------------------------------- the search
def _evaluate(recipe, critic, params, seeds, reps, workers, first=None):
    """render (unless `first` holds the first renders) and score candidates; returns ({j: row}, timing dict)"""
    reps = max(1, reps)
    t0 = time.time()
    jobs = [(j, k, s + k * 7919) for j, s in enumerate(seeds) for k in range(reps)]
    todo = [(params[j], s) for j, k, s in jobs if first is None or k > 0]
    it = iter(render_many(recipe, todo, workers))
    xs = [first[j] if (first is not None and k == 0) else next(it) for j, k, _ in jobs]
    t1 = time.time()
    live = [i for i, x in enumerate(xs) if x is not None]
    scores = critic.score([xs[i] for i in live]) if live else []
    t2 = time.time()
    got = {}
    for i, sc in zip(live, scores):
        got.setdefault(jobs[i][0], []).append((sc, i))
    by = {}
    for j, rows in got.items():
        if len(rows) < reps:          # a failed render among the reps: the candidate is unreliable
            continue
        if reps > 1:
            tot = [float(r['total']) for r, _ in rows]
            row = dict(rows[0][0])
            row['total_reps'] = [round(t, 3) for t in tot]
            row['total'] = round(float(np.mean(tot)), 3) if all(np.isfinite(tot)) else -np.inf
            row['passes'] = sum(1 for r, _ in rows if r.get('passes')) * 2 >= len(rows)
            by[j] = row
        else:
            by[j] = rows[0][0]
    xs0 = [xs[j * reps] for j in range(len(seeds))]
    return by, xs0, {'render_s': t1 - t0, 'score_s': t2 - t1, 'rendered': len(todo), 'scored': len(live)}


class _CMA:
    """separable CMA-ES state in the unit cube (Ros & Hansen 2008), mirrored at the bounds"""

    def __init__(self, d, lam, m, sigma, mu_frac=0.5, elite=0.0):
        self.d, self.lam, self.elite = d, lam, float(elite)
        mu = max(1, int(round(lam * mu_frac)))
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        n, me = d, self.mueff
        self.cs = (me + 2) / (n + me + 5)
        self.ds = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        sep = (n + 2) / 3.0
        self.c1 = min(1.0, sep * 2 / ((n + 1.3) ** 2 + me))
        self.cmu = min(1 - self.c1, sep * 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.reset(m, sigma)

    def reset(self, m, sigma):
        self.m, self.sigma = np.array(m, float), float(sigma)
        self.C = np.ones(self.d)
        self.ps, self.pc = np.zeros(self.d), np.zeros(self.d)
        self.g = 0

    def ask(self, R, n, elites=()):
        """n points: around the mean, a fraction `elite` around the best points so far (elitist, for short noisy runs)"""
        z = R.standard_normal((n, self.d))
        ne = int(round(self.elite * n)) if len(elites) else 0
        cen = [elites[i % len(elites)] for i in range(ne)] + [self.m] * (n - ne)
        return [_mirror(c + self.sigma * np.sqrt(self.C) * zi) for c, zi in zip(cen, z)]

    def tell(self, U, totals):
        """U: evaluated points (unit cube), totals: their scores (higher is better)"""
        order = np.argsort(-np.asarray(totals, float), kind='stable')[:len(self.w)]
        Y = np.array([(U[i] - self.m) / self.sigma for i in order])
        lim = 2.0 * self.chi                              # clip injected / mirrored outliers (Hansen 2011)
        nrm = np.sqrt(((Y / np.sqrt(self.C)) ** 2).sum(1))
        Y *= np.minimum(1.0, lim / np.maximum(nrm, 1e-12))[:, None]
        w = self.w[:len(Y)] / self.w[:len(Y)].sum()
        yw = (w[:, None] * Y).sum(0)
        self.m = np.clip(self.m + self.sigma * yw, 0.0, 1.0)
        self.g += 1
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw / np.sqrt(self.C)
        hs = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (self.d + 1)) * self.chi
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (w[:, None] * Y ** 2).sum(0))
        self.C = np.clip(self.C, 1e-4, 16.0)
        self.sigma = float(np.clip(self.sigma * math.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi - 1)),
                                   1e-3, 0.6))


def optimize(recipe, critic, budget=160, pop=16, elite=4, seed=0, init=None, sigma0=0.22, sigma1=0.04,
             fresh=0.15, log=None, space=None, reps=1, strategy='es', prescreen=0, workers=None, patience=4,
             sigma_stop=0.03, pre=None, cma=None):
    """returns {'best': [top results], 'history': [every scored result], 'generations': n, 'gens': [per-generation
    stats], 'timing': {...}, 'strategy', 'prescreen', 'fill_warnings'}; a result = {gen, u, params, seed, score[, pre]}.
    space: optional narrower {name: spec} (a family); parameters it leaves out keep the recipe's DEFAULTS (a start
    point keeps its own; a start's missing SPACE keys are filled by fill_start(): DEFAULTS, never a silent mid-range).
    reps: renders per candidate (ranked by the mean total; budget counts candidates, so reps=2 costs twice the time).
    strategy: 'es' (v1) | 'cma' (separable CMA-ES with restarts). prescreen=K > pop: render K per generation, rank
    them model-free against the critic's refs (`pre=` a prebuilt Prescreen) and score only the best pop.
    workers: render threads (default EARSHOT_THREADS or 2). patience / sigma_stop: CMA restart triggers; cma= {'sigma0':
    0.15 initial / restart step, 'mu': 0.5 parent fraction, 'elite': 0 share of samples around the best so far}."""
    base = dict(getattr(recipe, 'DEFAULTS', {}) or {})   # keys outside the searched space keep their DEFAULTS
    if space is None:                                     # (also for a recipe object whose SPACE is narrower)
        space = recipe.SPACE
    full = space is recipe.SPACE
    R = np.random.default_rng(seed)
    d = len(space)
    defaults = dict(getattr(recipe, 'DEFAULTS', {}) or {})
    starts = [defaults] + [{**defaults, **dict(p)} for p in (init or [])]
    # a SPACE key a start leaves out takes the recipe's DEFAULTS (the object's, then its render module's), never the
    # mid-range _encode() gives a missing key (round 4: a driver's final pass over members built off DEFAULTS got
    # snap 0.75 / sheen 1.0 that way, a 0 dB onset tick); mid-range only when no DEFAULTS has the key, with a warning
    fill_defaults = recipe_defaults(recipe)
    fill_warnings = []
    for j, st in enumerate(starts):
        starts[j], notes = fill_start(space, st, fill_defaults, own=defaults)
        for nt in notes:
            if nt['from'] != 'module DEFAULTS':
                fill_warnings.append({'start': j, **nt})
    if fill_warnings:
        msg = (f"forge: {len(fill_warnings)} start value(s) not from the start or DEFAULTS: "
               + '; '.join(f"start {w['start']} {w['key']} = {w['value']!r} ({w['from']})" for w in fill_warnings[:8])
               + (' ...' if len(fill_warnings) > 8 else ''))
        if log:
            log(msg)
        if any(w['from'] == 'mid-range' for w in fill_warnings):  # a driver that silences log still hears this
            warnings.warn(msg, MidRangeFill, stacklevel=2)
    gens = max(1, budget // pop)
    K = int(prescreen or 0)
    use_pre = K > pop
    if use_pre and pre is None:
        pre = prescreen_for(critic)
    history, elites, gstats = [], [], []
    timing = {'render_s': 0.0, 'pre_s': 0.0, 'score_s': 0.0, 'rendered': 0, 'scored': 0}
    co = {'sigma0': 0.15, 'mu': 0.5, 'elite': 0.0, **(cma or {})}
    cma = _CMA(d, pop, _encode(space, starts[0]), co['sigma0'], co['mu'], co['elite']) if strategy == 'cma' else None
    restarts, best_seen, stale, tried = 0, -np.inf, 0, []
    t_start = time.time()
    for g in range(gens):
        n_new = K if use_pre else pop
        U, seeds, fixed = [], [], 0
        if strategy == 'cma':
            if g == 0:                                    # the starting points are always scored
                U = [_encode(space, p) for p in starts[:pop]]
                fixed = len(U)
            U += cma.ask(R, n_new - len(U), [h['u'] for h in elites])
            seeds = [int(R.integers(0, 2 ** 31)) for _ in U]
        else:                                             # v1 draw order: each seed right after its candidate
            sig = sigma0 * (sigma1 / sigma0) ** (g / max(1, gens - 1))
            for j in range(n_new):
                if g == 0:
                    u = _encode(space, starts[j]) if j < len(starts) and j < pop else R.uniform(0, 1, d)
                    fixed += int(j < len(starts) and j < pop)
                elif R.uniform() < fresh or not elites:
                    u = R.uniform(0, 1, d)
                else:
                    parent = elites[min(len(elites) - 1, int(R.integers(0, max(1, min(elite, len(elites))))))]
                    u = np.clip(parent['u'] + R.normal(0, sig, d), 0, 1)
                U.append(u)
                seeds.append(int(R.integers(0, 2 ** 31)))
        params = [{**base, **_decode(space, u)} for u in U]
        for j in range(fixed):                            # starting points are scored as given (filled: fill_start):
            params[j].update({k: v for k, v in starts[j].items()        # exact values inside the space, their own
                              if k not in space or _inside(space[k], v)})  # values for keys outside it
        prows, first = None, None
        if use_pre:
            t0 = time.time()
            xs = render_many(recipe, list(zip(params, seeds)), workers)
            t1 = time.time()
            prows = pre.score(xs)
            t2 = time.time()
            timing['render_s'] += t1 - t0
            timing['pre_s'] += t2 - t1
            timing['rendered'] += len(xs)
            order = list(range(fixed)) + sorted(range(fixed, len(U)), key=lambda i: (-prows[i]['pre'], i))
            sel = order[:pop]
            U, seeds, params = [U[i] for i in sel], [seeds[i] for i in sel], [params[i] for i in sel]
            prows, first = [prows[i] for i in sel], [xs[i] for i in sel]
        by, xs0, tm = _evaluate(recipe, critic, params, seeds, reps, workers, first=first)
        timing['render_s'] += tm['render_s']
        timing['score_s'] += tm['score_s']
        timing['rendered'] += tm['rendered']
        timing['scored'] += tm['scored']
        if prows is None:                                 # measurements for diverse picks and the report (cheap)
            t0 = time.time()
            prows = pre.score(xs0, polish=False) if pre is not None else [{'desc': describe(x)} if x is not None else {}
                                                                          for x in xs0]
            timing['pre_s'] += time.time() - t0
        rows = []
        for j in range(len(U)):
            sc = by.get(j, {'total': -np.inf, 'passes': False, 'summary': 'render failed or silent'})
            h = {'gen': g, 'u': U[j], 'params': params[j], 'seed': seeds[j], 'score': sc}
            pr = prows[j] if j < len(prows) else {}
            if pr.get('desc') is not None:
                h['desc'] = pr['desc']
            if 'pre' in pr and pre is not None and pre.n:
                h['pre'] = {k: pr.get(k) for k in ('pre', 'tone_db', 'tone_ratio', 'env_pen')}
                h['pre']['art'] = pr['art']['flags']
                if 'polish' in pr:
                    h['pre']['polish'] = pr['polish']
            rows.append(h)
        history += rows
        ranked = sorted([h for h in history if np.isfinite(h['score']['total'])], key=lambda h: -h['score']['total'])
        elites = ranked[:elite]
        fin = [h['score']['total'] for h in rows if np.isfinite(h['score']['total'])]
        gs = {'gen': g + 1, 'best': round(float(elites[0]['score']['total']), 3) if elites else None,
              'gen_best': round(max(fin), 3) if fin else None,
              'gen_median': round(float(np.median(fin)), 3) if fin else None,
              'passes': sum(1 for h in rows if h['score'].get('passes')), 'scored': len(rows),
              'rendered': (K if use_pre else len(rows)), 'secs': round(time.time() - t_start, 1)}
        if strategy != 'cma':
            gs['sigma'] = round(sig, 4)
        if elites and 'tone_db' in elites[0]['score']:
            gs['best_tone_db'] = elites[0]['score']['tone_db']
        if strategy == 'cma':
            fu = [h['u'] for h in rows if np.isfinite(h['score']['total'])]
            ft = [h['score']['total'] for h in rows if np.isfinite(h['score']['total'])]
            if fu:
                cma.tell(fu, ft)
            gs['sigma'] = round(cma.sigma, 4)
            if elites and elites[0]['score']['total'] > best_seen + 1e-3:
                best_seen, stale = elites[0]['score']['total'], 0
            else:
                stale += 1
            if g < gens - 1 and (cma.sigma < sigma_stop or stale >= patience):
                restarts += 1
                tried.append(cma.m.copy())
                far = [h for h in ranked[:24] if min(np.linalg.norm(h['u'] - t) for t in tried) / math.sqrt(d) > 0.2]
                m0 = far[0]['u'] if far and restarts % 2 == 1 else R.uniform(0, 1, d)
                cma.reset(m0, co['sigma0'])
                stale = 0
                gs['restart'] = 'elite' if (far and restarts % 2 == 1) else 'random'
        gstats.append(gs)
        if log and elites:
            b = elites[0]['score']
            extra = f" sigma {gs['sigma']:.3f}" if 'sigma' in gs else ''
            extra += f" (restart from {gs['restart']})" if gs.get('restart') else ''
            extra += f' tone {b["tone_db"]:.1f} dB' if b.get('tone_db') is not None else ''
            log(f"gen {g + 1}/{gens}: best total {b['total']:.2f} ({'PASS' if b['passes'] else 'no pass'}) "
                f"z {b.get('match_z')} p_real {b.get('p_real')} typ {b.get('ref_typ_z')} pen {b.get('feat_pen')}"
                + extra + (f' [{K} rendered, {pop} scored]' if use_pre else ''))
    ranked = sorted([h for h in history if np.isfinite(h['score']['total'])], key=lambda h: -h['score']['total'])
    timing['total_s'] = round(time.time() - t_start, 2)
    for k in ('render_s', 'pre_s', 'score_s'):
        timing[k] = round(timing[k], 2)
    timing['scored_per_min'] = round(len(history) / max(timing['total_s'], 1e-9) * 60, 1)
    timing['rendered_per_min'] = round(timing['rendered'] / max(timing['total_s'], 1e-9) * 60, 1)
    return {'best': ranked[:max(elite, 8)], 'history': history, 'generations': gens, 'gens': gstats,
            'timing': timing, 'strategy': strategy, 'prescreen': K if use_pre else 0, 'restarts': restarts,
            'full_space': full, 'fill_warnings': fill_warnings}


def robust(recipe, critic, params, seeds=4, seed0=5000, workers=None):
    """re-render `params` on `seeds` new noise seeds (seed0, seed0+1, ...) and score them: {mean, min, max, sd,
    pass_rate, n, totals}. A sound that only scores well on one seed is a lucky render, not a good recipe."""
    ss = list(range(seed0, seed0 + seeds)) if isinstance(seeds, int) else list(seeds)
    xs = render_many(recipe, [(params, s) for s in ss], workers)
    live = [x for x in xs if x is not None]
    rows = critic.score(live) if live else []
    fin = [float(r['total']) for r in rows if np.isfinite(r['total'])]
    out = {'n': len(ss), 'seeds': ss, 'totals': [round(t, 3) for t in fin], 'failed': len(ss) - len(fin),
           'mean': round(float(np.mean(fin)), 3) if fin else None, 'min': round(float(min(fin)), 3) if fin else None,
           'max': round(float(max(fin)), 3) if fin else None, 'sd': round(float(np.std(fin)), 3) if fin else None,
           'pass_rate': round(sum(1 for r in rows if r.get('passes')) / len(ss), 3)}
    tdb = [r['tone_db'] for r in rows if r.get('tone_db') is not None]
    if tdb:
        out['tone_db_mean'] = round(float(np.mean(tdb)), 2)
    return out


def diverse(results, recipe, k=4, min_dist=0.08, min_sound=2.0):
    """pick up to k strong results, passing first, whose parameter vectors differ (normalised distance >= min_dist)
    and, when the results carry sound descriptors ('desc', from optimize), whose sounds differ too (sound_dist >=
    min_sound dB-like units: tonal balance + envelope). If that leaves fewer than k, the rest are filled with the
    parameter rule alone (the v1 rule)."""
    order = sorted(results, key=lambda h: (not h['score'].get('passes'), -h['score']['total']))
    order = [h for h in order if np.isfinite(h['score']['total'])] or order

    def pdist(a, b):
        return np.linalg.norm(a['u'] - b['u']) / math.sqrt(len(a['u']))

    def ok(h, out, sound):
        for o in out:
            if o is h or pdist(h, o) < min_dist:
                return False
            if sound and 'desc' in h and 'desc' in o and sound_dist(h['desc'], o['desc']) < min_sound:
                return False
        return True

    out = []
    for sound in ((True, False) if min_sound else (False,)):
        for h in order:
            if len(out) == k:
                break
            if all(h is not o for o in out) and ok(h, out, sound):
                out.append(h)
    return sorted(out, key=lambda h: (not h['score'].get('passes'), -h['score']['total']))


# ---------------------------------------------------------------------------------------------- seed bank
def _ctx(target, refs):
    return (target or '') + ' | ' + ','.join(sorted(os.path.basename(str(r)) for r in (refs or [])))


def _round_params(p):
    return {k: (float(f'{v:.6g}') if isinstance(v, float) else v) for k, v in p.items()}


def seeds_path(recipe, folder=None):
    name = recipe if isinstance(recipe, str) else getattr(recipe, '__name__', 'recipe').rsplit('.', 1)[-1]
    return os.path.join(folder or SEEDS_DIR, f'{name}.json')


def _rank(e):
    v = e.get('rank_score')
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -1e9
    return v if np.isfinite(v) else -1e9


def load_seeds(recipe, target=None, refs=None, n=6, folder=None):
    """good parameter sets found so far for a recipe (earshot/forge/seeds/<recipe>.json): up to n param dicts, the
    ones found with the same target + refs first (the delivered members of that context first: save_seeds
    delivered=), then the rest by score. [] when there is no bank."""
    path = seeds_path(recipe, folder)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        bank = json.load(fh)
    ctx = _ctx(target, refs)
    ents = sorted(bank.get('entries', []), key=lambda e: (e.get('context') != ctx, not e.get('delivered'), -_rank(e)))
    return [dict(e['params']) for e in ents[:n]]


def save_seeds(recipe, picks, target=None, refs=None, source='', folder=None, per_context=6, cap=48, delivered=None):
    """merge picks ({params, score[, robust, seed, family / preset / name]} dicts) into the recipe's seed bank; ranked
    by the robust mean when there is one, else the total. Keeps it small: per_context entries per target + refs, cap
    in all (the newest contexts win; 48 leaves room for one recipe serving several categories, e.g. whoosh: fast / big
    / pro_big), near-duplicate parameter sets merged. Atomic write. Returns the bank dict.
    delivered= a tag (a category name; True = the context) when the picks are a delivered family: those entries are
    pinned, always kept (outside per_context and cap, never merged into an ordinary entry, kept without a finite
    score), and the entries pinned earlier under the same tag lose their pin (a re-delivery replaces the delivery; the
    old members stay as ordinary entries while they rank). Before round 5 a delivered member could be pushed out by
    higher-scoring attempts of the same context that were never delivered (logo_sting kept 1 of its 5)."""
    path = seeds_path(recipe, folder)
    bank = {'recipe': os.path.basename(path)[:-5], 'entries': []}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            bank = json.load(fh)
    ctx = _ctx(target, refs)
    tag = (ctx if delivered is True else str(delivered)) if delivered else None
    new = []
    for p in picks:
        sc = p.get('score', {}) or {}
        rb = p.get('robust') or {}
        rank = rb.get('mean') if rb.get('mean') is not None else sc.get('total')
        ok = rank is not None and np.isfinite(rank)
        if not ok and not tag:
            continue
        e = {'params': _round_params(p['params']), 'rank_score': round(float(rank), 3) if ok else None,
             'context': ctx, 'total': sc.get('total'), 'tone_db': sc.get('tone_db'), 'passes': bool(sc.get('passes')),
             'source': source}
        if rb:
            e['robust_mean'], e['robust_pass_rate'] = rb.get('mean'), rb.get('pass_rate')
        if tag:
            e['delivered'] = tag
            member = p.get('family') or p.get('preset') or p.get('member') or p.get('name')
            if member is not None:
                e['member'] = str(member)
            if p.get('seed') is not None:
                e['seed'] = int(p['seed'])
        new.append(e)
    old = [dict(e) for e in bank.get('entries', [])]
    if tag:
        for e in old:
            if e.get('delivered') == tag:
                e.pop('delivered')
                e['was_delivered'] = tag
    ents = new + old
    pinned = [e for e in ents if e.get('delivered')]
    keep, seen_ctx = [], []
    for e in ents:
        if e['context'] not in seen_ctx:
            seen_ctx.append(e['context'])
    for c in seen_ctx:
        mine = sorted([e for e in ents if e['context'] == c and not e.get('delivered')], key=lambda e: -_rank(e))
        picked = [q for q in pinned if q['context'] == c]
        n0 = len(picked)
        for e in mine:
            if len(picked) - n0 >= per_context:
                break
            if all(e['params'] != q['params'] for q in picked):
                picked.append(e)
        keep += picked[n0:]
    bank['entries'] = pinned + keep[:max(0, cap - len(pinned))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(bank, fh, indent=1)
    os.replace(tmp, path)
    return bank


def report_members(report):
    """the delivered members of a final's report.json (dict or path) as save_seeds picks: params with the tonal /
    pop finals' family_params merged under them, seed, score, robust and the member name"""
    if isinstance(report, str):
        with open(report, encoding='utf-8') as fh:
            report = json.load(fh)
    out = []
    for pk in report.get('picks', []):
        m = {k: pk[k] for k in ('score', 'robust', 'seed', 'family', 'preset', 'member', 'name') if pk.get(k) is not None}
        m['params'] = {**(pk.get('family_params') or {}), **(pk.get('params') or {})}
        m.setdefault('score', {})
        out.append(m)
    return out


def pin_delivered(recipe, report, category, target=None, refs=None, folder=None, source=None):
    """pin a delivered final's members in the recipe's seed bank (save_seeds delivered=category): `report` = the
    final's report.json (dict or path); the context is the report's target + refs unless given. Returns the bank."""
    path = report if isinstance(report, str) else None
    if path:
        with open(path, encoding='utf-8') as fh:
            report = json.load(fh)
    return save_seeds(recipe, report_members(report), target if target is not None else report.get('target'),
                      refs if refs is not None else report.get('refs'), folder=folder, delivered=category,
                      source=source or f"delivered final ({category})" + (f', {path}' if path else ''))
