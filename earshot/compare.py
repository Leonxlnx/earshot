"""Compare sounds with each other and with reference recordings: nearest references, typicality, A/B in words.

    earshot compare candidate.ogg --refs refs/approved_keys/ [--start 1.2 --end 1.6]
    earshot ab take1.ogg take2.ogg --query "a deep thocky key press"
    earshot ab pack.ogg pack.ogg --a-start 1.217 --a-end 1.615 --b-start 7.881 --b-end 8.285   # two trims (find output)
    earshot ab mine.wav pro.wav --refs refs/launch_whooshes/
    from earshot.compare import compare, ab

compare: CLAP cosine to every reference, the k nearest, and typicality = how close the file is to its k nearest
references compared with how close each reference is to its own k nearest neighbours in the set (z and percentile:
z >= -1 fits the set, z < -2 is an outlier). Feature level: robust z of each measurement (loudness, brightness,
attack, tail, low end, noise floor, ...) against the references' median / spread, in words.
ab: which of two sounds is closer to a description (calibrated z of each) and/or to the references, how similar they
are to each other, and how B differs from A in measurable terms (only differences big enough to hear; 'closer' is
None when the z difference is below 0.3).
Trims: compare --start/--end and ab --a-start/--a-end/--b-start/--b-end judge only that part of a file (both the
CLAP embedding and the measurements), so a take found by `earshot find` in a multi-take pack can be A/B-ed alone;
without them the whole file is compared (every take plus the silence between them).
Measurements are trim-proof, so two trims of the same sample read as the same: loudness = gated integrated LUFS of
the sound padded to at least 0.4 s (silence before / after it does not count), sounding length = first to last
10 ms frame within 40 dB of the loudest, noise floor only for sounds of 1 s or more (floored at -120 dBFS), width
floored at -40 dB side/mid (= mono). Unreadable files are skipped and listed under 'skipped'.
"""
import math
import os

import numpy as np

from . import audio, features, vocab
from .cli import common

LOG = {'brightness_hz', 'bandwidth_hz', 'attack_ms', 'decay_ms', 'length_s'}
MIN_SCALE = {'loudness_lufs': 1.0, 'peak_dbfs': 1.0, 'noise_floor_dbfs': 2.0, 'low_end_db': 1.0, 'width_db': 2.0,
             'brightness_hz': 0.05, 'bandwidth_hz': 0.05, 'attack_ms': 0.1, 'decay_ms': 0.1, 'length_s': 0.05}
# (key, threshold, kind) in order of how much they matter to the ear
DIFFS = [('loudness_lufs', 1.5, 'db'), ('brightness_hz', 1.15, 'ratio'), ('attack_ms', 1.5, 'ratio'),
         ('decay_ms', 1.5, 'ratio'), ('low_end_db', 3.0, 'db'), ('length_s', 1.25, 'ratio'),
         ('noise_floor_dbfs', 6.0, 'db'), ('width_db', 6.0, 'db'), ('bandwidth_hz', 1.4, 'ratio')]
WORDS = {'loudness_lufs': ('louder', 'quieter'), 'brightness_hz': ('brighter', 'darker'),
         'bandwidth_hz': ('more top end', 'less top end'), 'attack_ms': ('a softer attack', 'a sharper attack'),
         'decay_ms': ('a longer tail', 'a shorter tail'), 'noise_floor_dbfs': ('noisier', 'cleaner'),
         'length_s': ('longer', 'shorter'), 'low_end_db': ('more low end', 'less low end'),
         'width_db': ('wider', 'narrower'), 'peak_dbfs': ('hotter peaks', 'lower peaks')}
WITH = {'attack_ms', 'decay_ms', 'low_end_db', 'bandwidth_hz'}
LABEL = {'brightness_hz': 'brightness', 'bandwidth_hz': 'bandwidth', 'noise_floor_dbfs': 'noise floor',
         'width_db': 'side/mid', 'low_end_db': 'energy below 250 Hz', 'loudness_lufs': 'loudness',
         'length_s': 'sounding length'}
MONO_DB = -40.0
CLOSE_DZ = 0.3


def _hz(f):
    return f'{f / 1000:.1f} kHz' if f >= 1000 else f'{f:.0f} Hz'


def _ms(v):
    return f'{v / 1000:.1f} s' if v >= 2000 else f'{v:.0f} ms'


def active_lufs(x):
    """gated integrated loudness of the sound padded to >= 0.4 s: trailing / leading silence does not change it"""
    x = np.asarray(x, np.float32)
    n = int(0.4 * audio.SR)
    if len(x) < n:
        x = np.concatenate([x, np.zeros((n - len(x),) + x.shape[1:], np.float32)])
    return round(features.integrated_lufs(x), 1)


def active_length(x, floor_db=40.0):
    """seconds from the first to the last 10 ms frame within floor_db of the loudest frame"""
    e = features.frame_rms(audio.to_mono(np.asarray(x, np.float32)), int(0.01 * audio.SR))
    on = np.nonzero(20 * np.log10(e) >= 20 * np.log10(e.max()) - floor_db)[0]
    return round(min(len(x) / audio.SR, (on[-1] - on[0] + 1) * 0.01), 3)


def feats(m, x):
    """the comparable, trim-proof measurements of a features.measure() dict and its audio x"""
    b = m.get('bands_db') or {}
    sm, nf = m.get('side_to_mid_db'), m.get('noise_floor_dbfs')
    f = {'loudness_lufs': active_lufs(x), 'peak_dbfs': m.get('peak_dbfs'), 'brightness_hz': m.get('brightness_hz'),
         'bandwidth_hz': m.get('bandwidth_hz'), 'attack_ms': m.get('attack_ms'), 'decay_ms': m.get('decay_ms'),
         'length_s': active_length(x), 'duration': m.get('duration'),
         'noise_floor_dbfs': max(nf, -120.0) if nf is not None and m['duration'] >= 1.0 else None,
         'width_db': MONO_DB if sm is None else max(sm, MONO_DB)}
    if 'sub' in b and 'low' in b:
        f['low_end_db'] = round(10 * math.log10(10 ** (b['sub'] / 10) + 10 ** (b['low'] / 10)), 1)
    return {k: v for k, v in f.items() if v is not None}


def _val(k, v):
    return math.log(max(v, 1e-3)) if k in LOG else v


def _fmt(k, v):
    if k in ('brightness_hz', 'bandwidth_hz'):
        return _hz(v)
    if k in ('attack_ms', 'decay_ms'):
        return _ms(v)
    if k in ('length_s', 'duration'):
        return f'{v:.2f} s'
    if k == 'width_db' and v <= MONO_DB:
        return 'mono'
    return f'{v:.0f} ' + ('LUFS' if k == 'loudness_lufs' else 'dBFS' if k in ('noise_floor_dbfs', 'peak_dbfs') else 'dB')


def _phrase(k, up, a_txt, b_txt):
    lab = LABEL.get(k)
    return ('with ' if k in WITH else '') + WORDS[k][0 if up else 1] + f" ({(lab + ' ') if lab else ''}{b_txt} vs {a_txt})"


def differences(fa, fb, limit=4):
    """how B differs from A in words, from two feats() dicts: the `limit` most important differences big enough to
    hear (level, brightness, attack, tail, low end, length, noise, width, top end)"""
    out = []
    for k, thr, kind in DIFFS:
        a, b = fa.get(k), fb.get(k)
        if a is None or b is None:
            continue
        if kind == 'db':
            if abs(b - a) < thr:
                continue
            out.append((k, f'{abs(b - a):.0f} dB ' + WORDS[k][0 if b > a else 1] if k == 'loudness_lufs'
                        else _phrase(k, b > a, _fmt(k, a), _fmt(k, b))))
        elif min(a, b) > 0 and max(a, b) / min(a, b) >= thr and not (k == 'attack_ms' and abs(a - b) < 3):
            out.append((k, _phrase(k, b > a, _fmt(k, a), _fmt(k, b))))
    out = out[:limit]
    parts = [p for _, p in out if not p.startswith('with ')] + [p for _, p in out if p.startswith('with ')]
    if not parts:
        return [], 'no measurable difference big enough to hear (level, brightness, envelope, low end, noise)'
    return [k for k, _ in out], parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]


def measure_path(p, start=None, end=None):
    x = audio.load(p, start=start, end=end)
    return feats(features.measure(x, light=True), x)


def _name(p, start=None, end=None):
    """file name plus the trim, when one is given"""
    from .report import fmt_t
    rng = f' [{fmt_t(start or 0.0)}-{fmt_t(end) if end is not None else "end"}]' if start is not None or end is not None \
        else ''
    return os.path.basename(p) + rng


def _typicality(a_x, A_r, k):
    """(mean cos of x to its k nearest refs, z, percentile) vs the refs' own k-nearest-neighbour similarity"""
    S = A_r @ A_r.T
    np.fill_diagonal(S, -np.inf)
    kk = max(1, min(k, len(A_r) - 1))
    ref_s = np.sort(S, axis=1)[:, ::-1][:, :kk].mean(1)
    sx = float(np.sort(A_r @ a_x)[::-1][:kk].mean())
    sd = float(ref_s.std()) + 1e-6
    return sx, (sx - float(ref_s.mean())) / sd, 100 * float((ref_s <= sx).mean())


def _ordinal(v):
    n = int(round(v))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _fit_words(z):
    if z >= -1:
        return 'fits the set'
    if z >= -2:
        return 'somewhat unusual for the set'
    return 'an outlier: unlike the references'


def _refs(refs, exclude=()):
    """(readable reference files minus `exclude`, [{path, error}] skipped)"""
    audio.check_paths(refs, 'refs')
    ex = {os.path.normcase(os.path.abspath(p)) for p in exclude}
    return vocab.readable([p for p in audio.list_media(refs) if os.path.normcase(os.path.abspath(p)) not in ex])


def _skipped_text(bad):
    return f' Skipped {len(bad)} unreadable file(s): ' + ', '.join(os.path.basename(b['path']) for b in bad[:3]) + \
        ('...' if len(bad) > 3 else '') + '.' if bad else ''


def compare(path, refs, k=5, start=None, end=None):
    """nearest references, typicality vs the reference set, feature-level differences; start / end (s) = judge only
    that part of `path`"""
    vocab.check(path, start, end)
    R, bad = _refs(refs, exclude=[path])
    if len(R) < 2:
        return {'summary': f'need at least 2 readable reference files, found {len(R)} in {refs}.' + _skipped_text(bad),
                'path': path, 'n_refs': len(R), 'nearest': [], 'skipped': bad}
    A = vocab.embed_files([path] + R, ranges=[(start, end)] + [(None, None)] * len(R))
    ax, AR = A[0], A[1:]
    cos = AR @ ax
    order = np.argsort(-cos)
    nearest = [{'path': R[i], 'name': os.path.basename(R[i]), 'cosine': round(float(cos[i]), 3)} for i in order[:k]]
    sx, z, pct = _typicality(ax, AR, k)
    fx = measure_path(path, start, end)
    FR = [measure_path(p) for p in R]
    fz = {}
    for key, fxv in fx.items():
        vals = np.array([_val(key, f[key]) for f in FR if key in f])
        if len(vals) < 2 or key == 'duration':
            continue
        med = float(np.median(vals))
        scale = max(1.4826 * float(np.median(np.abs(vals - med))), MIN_SCALE.get(key, 0.1))
        fz[key] = {'value': fxv, 'refs_median': round(math.exp(med) if key in LOG else med, 3),
                   'robust_z': round((_val(key, fxv) - med) / scale, 2),
                   'refs_p10_p90': [round(float(np.percentile([f[key] for f in FR if key in f], q)), 3) for q in (10, 90)]}
    order_k = [d[0] for d in DIFFS]
    outl = sorted([(kk, d) for kk, d in fz.items() if abs(d['robust_z']) >= 2 and kk in order_k],
                  key=lambda t: order_k.index(t[0]))[:4]
    words = []
    for kk, d in outl:
        up = d['robust_z'] > 0
        if kk == 'loudness_lufs':
            words.append(f"{abs(d['value'] - d['refs_median']):.0f} dB {WORDS[kk][0 if up else 1]} than their median")
        else:
            words.append(_phrase(kk, up, 'median ' + _fmt(kk, d['refs_median']), _fmt(kk, d['value'])))
    fdist = float(np.median([abs(d['robust_z']) for d in fz.values()])) if fz else None
    s = (f"{_name(path, start, end)} vs {len(R)} references. Nearest: " +
         ', '.join(f"{n['name']} (cos {n['cosine']:.2f})" for n in nearest[:3]) +
         f". Typicality z {vocab.zf(z)} ({_ordinal(pct)} percentile of how close the references are to each other): "
         f"{_fit_words(z)}.")
    s += (' Measured against the references it is ' + '; '.join(words) + '.') if words else \
        " Every measurement is within the references' spread."
    return {'summary': s + _skipped_text(bad), 'path': path, 'range': [start, end], 'n_refs': len(R), 'nearest': nearest,
            'typicality': {'mean_cos_knn': round(sx, 3), 'z': round(z, 2), 'percentile': round(pct, 1), 'k': k,
                           'verdict': _fit_words(z)},
            'features': fz, 'feature_distance': None if fdist is None else round(fdist, 2), 'skipped': bad}


def _closer(za, zb):
    """('a' | 'b' | None, words): None below CLOSE_DZ"""
    d = abs(za - zb)
    if d < CLOSE_DZ:
        return None, 'Neither is closer'
    w = 'B' if zb > za else 'A'
    return w.lower(), f"{w} is {'clearly' if d >= 1 else 'slightly'} closer"


def ab(a, b, query=None, refs=None, k=5, a_start=None, a_end=None, b_start=None, b_end=None):
    """which of two sounds is closer to a description and/or reference set, and how B differs from A; *_start /
    *_end (s) = use only that part of a file (e.g. two trims from `find`, even of the same file)"""
    vocab.check(a, a_start, a_end)
    vocab.check(b, b_start, b_end)
    A = vocab.embed_files([a, b], ranges=[(a_start, a_end), (b_start, b_end)])
    res = {'a': a, 'b': b, 'a_range': [a_start, a_end], 'b_range': [b_start, b_end],
           'cosine_ab': round(float(A[0] @ A[1]), 3)}
    s = [f'A = {_name(a, a_start, a_end)}, B = {_name(b, b_start, b_end)}.']
    if query:
        sc = vocab.get().scores(A, [query])
        za, zb = sc[0][0]['z'], sc[1][0]['z']
        w, txt = _closer(za, zb)
        res['query'] = {'query': query, 'a': sc[0][0], 'b': sc[1][0], 'closer': w, 'dz': round(abs(za - zb), 2)}
        s.append(f"{txt} to '{query}' (z A {vocab.zf(za)}, B {vocab.zf(zb)}).")
    if refs:
        R, bad = _refs(refs, exclude=[a, b])
        if len(R) >= 2:
            AR = vocab.embed_files(R)
            ta, tb = _typicality(A[0], AR, k), _typicality(A[1], AR, k)
            w, txt = _closer(ta[1], tb[1])
            res['refs'] = {'n_refs': len(R), 'a': {'mean_cos_knn': round(ta[0], 3), 'z': round(ta[1], 2)},
                           'b': {'mean_cos_knn': round(tb[0], 3), 'z': round(tb[1], 2)}, 'closer': w, 'skipped': bad}
            s.append(f'{txt} to the {len(R)} references (typicality z A {vocab.zf(ta[1])}, B {vocab.zf(tb[1])}; '
                     f'>= -1 fits the set).' + _skipped_text(bad))
        else:
            s.append(f'Need at least 2 readable reference files, found {len(R)}.' + _skipped_text(bad))
    sim = res['cosine_ab']
    same = ('nearly the same sound' if sim >= 0.9 else 'similar' if sim >= 0.75 else 'related' if sim >= 0.5
            else 'different sounds')
    s.append(f'CLAP hears them as {same} (cosine {sim:.2f}).')
    fa, fb = measure_path(a, a_start, a_end), measure_path(b, b_start, b_end)
    keys, txt = differences(fa, fb)
    res['features'] = {'a': fa, 'b': fb}
    res['differences'] = keys
    s.append(f'B is {txt}.' if keys else f'Measured: {txt}.')
    res['summary'] = ' '.join(s)
    return res


def _compare_text(r):
    lines = [r['summary'], '', 'nearest:'] + [f"  cos {n['cosine']:.3f}  {n['name']}" for n in r.get('nearest', [])]
    if r.get('features'):
        lines.append('features (value / refs median / robust z):')
        for k, d in r['features'].items():
            lines.append(f"  {k:<17} {d['value']:>9} / {d['refs_median']:>9} / {d['robust_z']:+.1f}")
    return '\n'.join(lines)


def add_cli(sub):
    p = sub.add_parser('compare', help='nearest reference recordings, typicality and feature differences')
    p.add_argument('path')
    p.add_argument('--refs', nargs='+', required=True, help='reference files / folders')
    p.add_argument('-k', type=int, default=5, help='how many nearest references to list (default 5)')
    p.add_argument('--start', type=float, help='judge only from this time of PATH (s), e.g. a trim from find')
    p.add_argument('--end', type=float, help='judge only up to this time of PATH (s)')
    common(p)
    p.set_defaults(func=lambda a: vocab.run_cli(lambda: compare(a.path, a.refs, a.k, a.start, a.end), a, _compare_text))
    p = sub.add_parser('ab', help='A/B two sounds (or two trims): closer to a description / references, differences '
                                  'in words')
    p.add_argument('a', help='sound A (e.g. the current version)')
    p.add_argument('b', help='sound B (e.g. the candidate / new render; may be the same file as A with other trims)')
    p.add_argument('--query', help='a description both should match: which one is closer (calibrated z)')
    p.add_argument('--refs', nargs='+', help='reference files / folders both should resemble')
    for w in ('a', 'b'):
        p.add_argument(f'--{w}-start', type=float, help=f'use {w.upper()} only from this time (s)')
        p.add_argument(f'--{w}-end', type=float, help=f'use {w.upper()} only up to this time (s)')
    common(p)
    p.set_defaults(func=lambda a: vocab.run_cli(lambda: ab(a.a, a.b, a.query, a.refs, a_start=a.a_start, a_end=a.a_end,
                                                           b_start=a.b_start, b_end=a.b_end), a))


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
