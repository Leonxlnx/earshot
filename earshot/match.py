"""How well does a sound match a text description? Calibrated scores and a verdict whose evidence was measured.

    earshot match key.ogg "a deep thocky mechanical keyboard press" --alts "a clicky blue switch" "a typewriter"
    earshot match whoosh.wav "a fast whoosh" "a soft air whoosh" "a deep cinematic whoosh"      # several queries
    from earshot.match import match
    match('key.ogg', 'a deep thock', alts=['a clicky switch'])

Fields per query: cosine (raw CLAP similarity, NOT a probability; only comparable for the same sound and query), z
(calibrated: how far the description stands out among the vocabulary descriptions for this sound after removing the
description's own bias, see earshot/vocab.py), percentile + rank_in_vocab (where the query would rank among the
vocabulary descriptions of this sound), p_vs_alts (softmax of logit_scale * cosine over the query and its alternatives:
a forced choice, only as meaningful as the alternatives; with several queries and no alts the queries compete with
each other), verdict and the closest vocabulary descriptions (what else it could be).

Verdicts are bands of z (not cumulative). Thresholds chosen on 219 labelled real recordings by a fixed likelihood-ratio
rule; the evidence below was measured on a separate held-out library of 125 real recordings, labelled from file names
before scoring, that nothing was tuned on (eval/semantics/RESULTS.md). LR = how much more often true matches land in
the band than wrong ones; precision_library = share of pairs in the band that are true when ~3% of candidate pairs are
true (searching a library), precision_even_odds = when you already half-expect it to fit. 'lenient' leaves out pairs
of related classes (an Enter key for 'a spacebar press', wind for grass steps); 'strict' counts them as wrong, so it is
the number to use when the exact variant matters. 'weak' is evidence (LR ~7-12), not proof; below 2 there is none.
The strong band's LR (1289) rests on only 3 wrong pairs (95% range roughly 440-6200; 170 on the tuning library), so
the text output quotes precision, not the ratio.
Music genre / mood, instrument and voice descriptions were never validated: treat their verdicts as hints.
"""
import os

import numpy as np

from . import audio, vocab
from .cli import common

VERDICTS = [(3.0, 'strong'), (2.5, 'plausible'), (2.0, 'weak')]
# held-out library, shipped configuration (eval/semantics/heldout.json): 248 true / 7614 wrong pairs (lenient),
# 8222 wrong (strict); base rate 3.2%
VERDICT_EVIDENCE = {
    'strong': {'z': '>= 3', 'lr': 1289, 'precision_library': 0.98, 'precision_even_odds': 1.0,
               'lr_strict': 51, 'precision_library_strict': 0.61, 'recall_at_or_above': 0.51},
    'plausible': {'z': '2.5 - 3', 'lr': 21, 'precision_library': 0.41, 'precision_even_odds': 0.95,
                  'lr_strict': 9.1, 'precision_library_strict': 0.22, 'recall_at_or_above': 0.63},
    'weak': {'z': '2 - 2.5', 'lr': 12, 'precision_library': 0.28, 'precision_even_odds': 0.92,
             'lr_strict': 6.6, 'precision_library_strict': 0.17, 'recall_at_or_above': 0.76},
    'no match': {'z': '< 2', 'lr': 0.25, 'precision_library': 0.008, 'precision_even_odds': 0.2,
                 'lr_strict': 0.25, 'precision_library_strict': 0.008},
}
SKIP_CATS = ('quality', 'production')
SILENT_DBFS = -70.0


def verdict(z):
    for t, name in VERDICTS:
        if z >= t:
            return name
    return 'no match'


def softmax(x, scale):
    e = np.exp(scale * (np.asarray(x, np.float64) - np.max(x)))
    return e / e.sum()


def _p(p):
    return '>99%' if p > 0.995 else '<1%' if p < 0.005 else f'{100 * p:.0f}%'


def score_queries(a, queries, alts=None, v=None):
    """per-query score dicts for one audio embedding a (512,)"""
    v = v or vocab.get()
    rows = v.scores(a[None], queries)[0]
    scale = vocab.logit_scale()
    alt_rows = v.scores(a[None], list(alts))[0] if alts else []
    multi = softmax([r['cosine'] for r in rows], scale) if len(rows) > 1 and not alts else None
    kinds = v.nearest_category(v.embed_text([r['query'] for r in rows]))
    for i, r in enumerate(rows):
        r['verdict'] = verdict(r['z'])
        r['validated'] = kinds[i] not in vocab.UNVALIDATED
        if alts:
            p = softmax([r['cosine']] + [x['cosine'] for x in alt_rows], scale)
            r['p_vs_alts'] = round(float(p[0]), 3)
            r['alts'] = [dict(query=x['query'], p=round(float(pi), 3), z=x['z'], cosine=x['cosine'])
                         for x, pi in zip(alt_rows, p[1:])]
        else:
            r['p_vs_alts'] = None if multi is None else round(float(multi[i]), 3)
    return rows


def peak_dbfs(path, start=None, end=None):
    """sample peak of the file as it plays (stereo decode; mono files are duplicated at unity gain)"""
    x = audio.load(path, start=start, end=end)
    return audio.db(np.abs(x).max() if len(x) else 0.0)


def match(path, query, alts=None, start=None, end=None):
    """calibrated match of one or several text descriptions against a sound (or a [start, end) segment of it)"""
    queries = [query] if isinstance(query, str) else list(query or [])
    if not queries or not all(isinstance(q, str) and q.strip() for q in queries):
        raise ValueError('no query given: pass one or more non-empty descriptions')
    vocab.check(path, start, end)
    v = vocab.get()
    a = vocab.embed_file(path, start, end)
    rows = sorted(score_queries(a, queries, alts, v), key=lambda r: -r['z'])
    best = rows[0]
    near = [dict(term=t, category=c, z=z, cosine=cos) for t, c, z, cos in v.top_terms(a, k=4, exclude=SKIP_CATS)]
    seg = f' [{start or 0:.3f}-{end:.3f} s]' if end is not None else (f' [from {start:.3f} s]' if start else '')
    unval = ' (unvalidated: music / instrument / voice description)'
    parts = [f"'{r['query']}' -> {r['verdict']}{'' if r['validated'] else unval} (z {vocab.zf(r['z'])}; would rank "
             f"#{r['rank_in_vocab']} among {r['vocab_size']} vocabulary descriptions; cosine {r['cosine']:.2f})"
             for r in rows]
    s = f'{os.path.basename(path)}{seg}: ' + '; '.join(parts) + '.'
    if best.get('alts'):
        s += f" Forced choice vs your alternatives: '{best['query']}' {_p(best['p_vs_alts'])}, " + ', '.join(
            f"'{x['query']}' {_p(x['p'])}" for x in best['alts']) + '.'
    elif len(rows) > 1:
        s += ' Forced choice among your queries: ' + ', '.join(f"'{r['query']}' {_p(r['p_vs_alts'])}" for r in rows) + '.'
    alt_txt = ', '.join(f"'{x['term']}' (z {vocab.zf(x['z'])})" for x in near if x['term'] not in queries)
    if alt_txt:
        s += f' It sounds most like: {alt_txt}.'
    ev = VERDICT_EVIDENCE[best['verdict']]
    if best['verdict'] != 'no match' and best['validated']:
        s += (f" Evidence for '{best['verdict']}' (held-out library search, ~3% of candidate pairs true): the right kind "
              f"of sound {100 * ev['precision_library']:.0f}% of the time, the exact variant "
              f"{100 * ev['precision_library_strict']:.0f}% (a close variant such as an Enter key for a spacebar counts as "
              'wrong then).')
    pk = peak_dbfs(path, start, end)
    if pk < SILENT_DBFS:
        s = f'WARNING: near-silent input (peak {pk:.0f} dBFS), scores are meaningless. ' + s
    res = {'summary': s, 'path': path, 'start': start, 'end': end, 'peak_dbfs': round(pk, 1)}
    res.update({k: best[k] for k in ('query', 'cosine', 'z', 'percentile', 'rank_in_vocab', 'p_vs_alts', 'verdict',
                                     'validated')})
    if best.get('alts'):
        res['alts'] = best['alts']
    res.update(results=rows, closest_vocab=near, verdict_thresholds_z={n: t for t, n in VERDICTS},
               verdict_evidence=VERDICT_EVIDENCE)
    return res


def add_cli(sub):
    p = sub.add_parser('match', help='how well does a sound match a text description (calibrated z + verdict)')
    p.add_argument('path', help='audio file or video')
    p.add_argument('query', nargs='+', help='one or more descriptions')
    p.add_argument('--alts', nargs='+', help='alternative descriptions for a forced choice (p_vs_alts)')
    p.add_argument('--start', type=float, help='judge only from this time (s)')
    p.add_argument('--end', type=float, help='judge only up to this time (s)')
    common(p)
    p.set_defaults(func=lambda a: vocab.run_cli(lambda: match(a.path, a.query, a.alts, a.start, a.end), a))


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
