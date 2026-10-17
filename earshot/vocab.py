"""Listening vocabulary and score calibration: raw CLAP cosines -> numbers that mean the same for every sound and query.

    from earshot import vocab
    v = vocab.get()                                   # data/vocab.json terms, embedded once (cached on disk)
    a = vocab.embed_file('hit.ogg')                   # (512,) audio embedding, cached
    s = v.scores(a[None], ['a glass ting'])           # cosine, z, percentile, rank of the query among the vocab terms
    v.top_terms(a, k=8)                               # best vocab terms for this sound, calibrated

Why calibrate: a raw CLAP cosine is not comparable across sounds (dense, busy sounds score high on every text) or
across descriptions (some phrases score high for every sound). Every score here is
    u = cos(a, q) - m.q               term bias removed: m.q = mean cosine of description q over 241 reference
                                      recordings (the 219-file tuning library, 7 pro launch films, a film soundtrack +
                                      stems, synth stems; every source folder weighted equally). Same as centring audio
                                      embeddings on the reference mean m (data/vocab_stats.json), so it works for ANY
                                      query, not only vocabulary terms.
    z = (u - mean_t u_t) / std_t u_t  per-sound z over all 322 vocab terms t: how far q stands out among the
                                      vocabulary descriptions for this very sound.
Chosen by measurement (eval/semantics/RESULTS.md). Tuning library (219 labelled real recordings): the per-sound z is
the big win (pooled AUC of (recording, description) pairs 0.951 -> 0.963), centring adds a little (AUC 0.966,
category MAP 0.770 -> 0.778; class MAP 0.650 -> 0.642); dividing by each description's std over the references hurt
and was dropped; 5 prompt templates beat 1 (top-1 tagging 0.553 -> 0.594). Held-out library (125 real recordings,
nothing tuned on it): pooled AUC 0.980 (raw cosine 0.978), class MAP 0.646 (raw cosine 0.700: the z helps absolute
verdicts more than ranking files). Audio: 10 s or shorter = one CLAP window, repeat-padded (PAD; zero padding: AUC
0.964, class top-3 0.843 vs 0.886); longer = mean of 10 s windows (hop 5 s), at most MAX_WIN evenly spaced.

Rebuild the reference statistics after changing the model or the reference set (vocab edits need no rebuild):
    python -m earshot.vocab --build-stats REF_FOLDERS_OR_FILES...
"""
import hashlib
import json
import os
import sys
import threading

import numpy as np

from . import audio, config

VOCAB_PATH = os.path.join(config.DATA_DIR, 'vocab.json')
STATS_PATH = os.path.join(config.DATA_DIR, 'vocab_stats.json')
PAD = 'repeatpad'
PADS = ('repeatpad', 'zero', 'spaced')
SPACED_S = 1.0
MAX_WIN = 6
WIN, HOP = 10.0, 5.0
METHODS = ('raw', 'sound', 'center', 'both')
# categories whose tags were never validated (the evaluation has no labelled music genre / mood / instrument / speech
# data): reported as hints, never with a verdict
UNVALIDATED = ('music_genre', 'music_mood', 'instruments', 'voice')
_lock = threading.Lock()
_inst = None


def load_vocab(path=VOCAB_PATH):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def load_stats(path=STATS_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _key(*parts):
    return hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:24]


def _clap():
    from .models import clap
    return clap.get()


def _cache_dir():
    from .models import hub   # filesystem only: another CLAP revision or a local folder gets its own cache
    d = os.path.join(config.EMB_CACHE, 'semantics',
                     hub.cache_name(config.CLAP_MODEL, config.CLAP_REVISION) + '--' + config.PRECISION)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------- input checks
def check(path, start=None, end=None):
    """duration of `path` after validating [start, end); raises audio.AudioError (missing / undecodable) or ValueError"""
    dur = audio.duration(path)
    name = os.path.basename(path)
    if start is not None and start < 0:
        raise ValueError(f'start {start:.3f} s is negative')
    if start is not None and start >= dur:
        raise ValueError(f'start {start:.3f} s is at or past the end of {name} ({dur:.3f} s long)')
    if end is not None and end <= (start or 0.0):
        raise ValueError(f'end {end:.3f} s must be after start {start or 0.0:.3f} s')
    return dur


def readable(paths):
    """(decodable paths, [{path, error}] for the rest): one bad file must not sink a whole batch"""
    ok, bad = [], []
    for p in paths:
        try:
            audio.duration(p)
            ok.append(p)
        except (audio.AudioError, OSError) as e:
            bad.append({'path': p, 'error': str(e).strip().splitlines()[0][:200]})
    return ok, bad


def zf(z):
    """z for text, floored to 0.1 so it can never contradict a verdict threshold (2.96 -> '+2.9', not '+3.0')"""
    return f'{np.floor(float(z) * 10 + 1e-9) / 10:+.1f}'


def peak_db(path, start=None, end=None):
    """sample peak of the mono decode in dB (for the near-silence guard; levels proper come from the stereo decode)"""
    x = audio.load(path, mono=True, start=start, end=end)
    return audio.db(float(np.abs(x).max()) if len(x) else 0.0)


def run_cli(fn, a, text=None):
    """run a CLI handler: expected problems (missing / undecodable file, bad range) as one plain line, also as JSON
    with --json, exit 1; anything else is left to earshot.cli"""
    from .report import emit, error_line
    try:
        r = fn()
    except (audio.AudioError, ValueError, OSError) as e:
        msg = ' '.join(str(e).split()) or type(e).__name__
        error_line(getattr(a, 'command', None) or '', msg)
        if getattr(a, 'json', False):  # the JSON error stays on stdout for a caller parsing --json
            sys.stdout.write(json.dumps({'summary': f'error: {msg}', 'error': msg}) + '\n')
        return 1
    emit(r, a, text(r) if text else r['summary'])
    return 0


# ---------------------------------------------------------------------------------------------------- audio embedding
def spans_for(start, end, max_win=MAX_WIN):
    """CLAP windows [(s, e)] covering [start, end): one span if <= 10 s, else 10 s windows hop 5 s (last one flush with
    the end), at most max_win of them evenly spaced"""
    n, w, h = int(round((end - start) * audio.SR)), int(WIN * audio.SR), int(HOP * audio.SR)
    if n <= w:
        return [(start, end)]
    st = list(range(0, n - w + 1, h))
    if (n - w) % h:
        st.append(n - w)
    if max_win and len(st) > max_win:
        st = [st[int(round(i))] for i in np.linspace(0, len(st) - 1, max_win)]
    return [(start + s / audio.SR, start + (s + w) / audio.SR) for s in st]


def _pad_key(pad, dur):
    """cache key of the padding actually applied to a span of `dur` seconds"""
    if dur >= WIN - 1e-3:
        return '-'
    return 'repeatpad' if pad == 'spaced' and dur >= SPACED_S - 1e-3 else pad


def embed_spans(items, pad=PAD, batch=8):
    """items = [(path, start, end)] -> (n, 512): every span is one CLAP window (<= 10 s), padded to 10 s with `pad`:
    'repeatpad' (tile the sound), 'zero' (silence after it) or 'spaced' (silence up to 1 s, then tile: one hit per
    second); cached per span on disk and computed in batches"""
    d = _cache_dir()
    out = [None] * len(items)
    todo = []
    for i, (p, s, e) in enumerate(items):
        f = os.path.join(d, _key(audio.file_id(p), round(s, 4), round(e, 4), _pad_key(pad, e - s)) + '.npy')
        if os.path.exists(f):
            try:
                out[i] = np.load(f)
                continue
            except Exception:
                pass
        todo.append((i, p, s, e, f))
    loaded = {}
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        segs = []
        for _, p, s, e, _ in chunk:
            if p not in loaded:
                loaded = {p: audio.load(p, mono=True)}
            seg = loaded[p][int(round(s * audio.SR)):int(round(e * audio.SR))]
            if pad == 'spaced' and len(seg) < SPACED_S * audio.SR:
                seg = np.pad(seg, (0, int(SPACED_S * audio.SR) - len(seg)))
            segs.append(seg)
        es = _clap().embed_audio(segs, pad='repeatpad' if pad == 'spaced' else pad, batch=batch)
        for (i, _, _, _, f), v in zip(chunk, es):
            out[i] = v
            tmp = f'{f}.{os.getpid()}.tmp.npy'
            np.save(tmp, v)
            audio.replace_quietly(tmp, f)
    return np.stack(out) if out else np.zeros((0, 512), np.float32)


def _norm(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)


def embed_files(paths, pad=PAD, max_win=MAX_WIN, ranges=None):
    """(n, 512) file (or [start, end) range) embeddings = mean of their CLAP windows; ranges: [(start, end)] or None"""
    items, owner = [], []
    for j, p in enumerate(paths):
        s, e = (ranges[j] if ranges else (None, None))
        dur = audio.duration(p)
        s = max(0.0, s or 0.0)
        e = dur if e is None else min(dur, e)
        for sp in spans_for(s, max(e, s + 0.01), max_win):
            items.append((p, sp[0], sp[1]))
            owner.append(j)
    E = embed_spans(items, pad)
    owner = np.array(owner)
    return _norm(np.stack([E[owner == j].mean(axis=0) for j in range(len(paths))]))


def embed_file(path, start=None, end=None, pad=PAD, max_win=MAX_WIN):
    return embed_files([path], pad, max_win, [(start, end)])[0]


# ---------------------------------------------------------------------------------------------------- vocabulary
class Vocab:
    def __init__(self, templates=None, stats=None, categories=None):
        d = load_vocab()
        self.templates = list(templates or d['templates'])
        self.cats = {c: list(ts) for c, ts in (categories or d['categories']).items()}
        self.names = [t for ts in self.cats.values() for t in ts]
        self.cat_of = {t: c for c, ts in self.cats.items() for t in ts}
        self.E = self.embed_text(self.names)
        self.stats = load_stats() if stats is None else (stats or None)
        if self.stats and self.stats.get('model') != config.CLAP_MODEL:
            self.stats = None
        self.m = np.array(self.stats['mean'], np.float32) if self.stats else None
        self.mu_E = self.bias(self.E)

    def embed_text(self, texts):
        """(n, 512): each phrase through every template, averaged and renormalised. Cached on disk (one file per
        phrase list > 16, else one per phrase), so a repeated query needs no model load"""
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts:
            return np.zeros((0, 512), np.float32)
        d = _cache_dir()
        keys = ([_key(json.dumps([self.templates, texts]))] if len(texts) > 16 else
                [_key(json.dumps([self.templates, [t]])) for t in texts])
        files = [os.path.join(d, 't_' + k + '.npy') for k in keys]
        got = []
        for f in files:
            try:
                got.append(np.load(f) if os.path.exists(f) else None)
            except Exception:
                got.append(None)
        if all(g is not None for g in got):
            v = np.concatenate([np.atleast_2d(g) for g in got])
            if v.shape == (len(texts), 512):
                return v
        T = len(self.templates)
        e = _clap().embed_text([tp.format(t) for t in texts for tp in self.templates])
        v = _norm(e.reshape(len(texts), T, -1).mean(axis=1)).astype(np.float32)
        for f, part in zip(files, [v] if len(files) == 1 and len(texts) > 16 else v[:, None, :]):
            tmp = f'{f}.{os.getpid()}.tmp.npy'
            np.save(tmp, part if len(texts) > 16 else part[0])
            audio.replace_quietly(tmp, f)
        return v

    def nearest_category(self, Q):
        """category of the vocabulary term whose text is closest to each query embedding (what kind of description)"""
        return [self.cat_of[self.names[i]] for i in np.argmax(np.atleast_2d(Q) @ self.E.T, axis=1)]

    def bias(self, Q):
        """term bias = mean cosine of each row of Q (n, 512) over the reference recordings (= Q . their mean embedding)"""
        Q = np.atleast_2d(Q)
        return Q @ self.m if self.m is not None else np.zeros(len(Q), np.float32)

    def calibrate(self, A, Q=None, method='both'):
        """A (n, 512) audio, Q (m, 512) query embeddings -> dict of (n, T) term arrays and (n, m) query arrays:
        cos, u (bias removed when method is 'center' / 'both'), z (per-sound z when 'sound' / 'both', else u),
        pct (share of vocab terms the query beats)"""
        A = np.atleast_2d(A)
        C = A @ self.E.T
        U = C - self.mu_E if method in ('center', 'both') else C
        r = {'cos_t': C, 'u_t': U}
        if method in ('sound', 'both'):
            mu, sd = U.mean(1, keepdims=True), U.std(1, keepdims=True) + 1e-9
        else:
            mu, sd = 0.0, 1.0
        r['z_t'] = (U - mu) / sd
        if Q is not None:
            Q = np.atleast_2d(Q)
            Cq = A @ Q.T
            Uq = Cq - self.bias(Q) if method in ('center', 'both') else Cq
            r['cos_q'], r['u_q'], r['z_q'] = Cq, Uq, (Uq - mu) / sd
            r['pct_q'] = (U[:, None, :] < Uq[:, :, None]).mean(-1)
        return r

    def scores(self, A, queries, method='both'):
        """per sound (rows of A) and query: cosine, z, percentile among vocab terms, rank_in_vocab"""
        Q = self.embed_text(list(queries))
        r = self.calibrate(A, Q, method)
        out = []
        for i in range(len(np.atleast_2d(A))):
            row = []
            for j, q in enumerate(queries):
                beats = int((r['u_t'][i] > r['u_q'][i, j]).sum())
                row.append({'query': q, 'cosine': round(float(r['cos_q'][i, j]), 3), 'z': round(float(r['z_q'][i, j]), 2),
                            'percentile': round(100 * float(r['pct_q'][i, j]), 1), 'rank_in_vocab': beats + 1,
                            'vocab_size': len(self.names)})
            out.append(row)
        return out

    def top_terms(self, a, k=8, cats=None, method='both', exclude=()):
        """[(term, category, z, cosine)] best calibrated vocab terms for one sound"""
        r = self.calibrate(a, method=method)
        z, c = r['z_t'][0], r['cos_t'][0]
        idx = [i for i in np.argsort(-z) if (cats is None or self.cat_of[self.names[i]] in cats)
               and self.cat_of[self.names[i]] not in exclude]
        return [(self.names[i], self.cat_of[self.names[i]], round(float(z[i]), 2), round(float(c[i]), 3)) for i in idx[:k]]


def logit_scale():
    """CLAP's softmax temperature (stored with the reference stats, so no model load is needed)"""
    st = load_stats()
    return float(st['logit_scale']) if st and st.get('logit_scale') else float(_clap().logit_scale)


def get(templates=None):
    global _inst
    with _lock:
        if _inst is None or (templates and list(templates) != _inst.templates):
            _inst = Vocab(templates)
        return _inst


# ---------------------------------------------------------------------------------------------------- reference stats
def group_weights(paths):
    """weight 1 / (files in the same folder): every source folder counts equally, whatever its size"""
    folders = [os.path.normcase(os.path.dirname(os.path.abspath(p))) for p in paths]
    n = {f: folders.count(f) for f in set(folders)}
    return np.array([1.0 / n[f] for f in folders])


def build_stats(A, paths, meta=None, path=STATS_PATH, templates=None):
    """reference statistics from audio embeddings A (n, 512) of `paths`: the folder-balanced mean embedding (all the
    calibration uses) + the resulting mean / std cosine of every vocab term (derived, for inspection); written to
    `path` unless it is None"""
    A = np.asarray(A, np.float64)
    w = group_weights(paths)
    w /= w.sum()
    m = (w[:, None] * A).sum(0)
    v = Vocab(templates, stats=False)
    C = A @ v.E.T.astype(np.float64)
    mu = (w[:, None] * C).sum(0)
    sd = np.sqrt((w[:, None] * (C - mu) ** 2).sum(0))
    res = dict(meta or {}, model=config.CLAP_MODEL, precision=config.PRECISION, pad=PAD, templates=v.templates,
               logit_scale=round(float(_clap().logit_scale), 4), n_ref=len(A),
               n_folders=len({os.path.dirname(os.path.abspath(p)) for p in paths}),
               mean=[round(float(x), 5) for x in m],
               term_cos={t: [round(float(a), 4), round(float(b), 4)] for t, a, b in zip(v.names, mu, sd)})
    if path:
        lines = [f' {json.dumps(k)}: {json.dumps(x)}' for k, x in res.items() if k != 'term_cos']
        lines.append(' "term_cos": {' + ','.join(f'\n  {json.dumps(t)}: {json.dumps(x)}'
                                                 for t, x in res['term_cos'].items()) + '\n }')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('{\n' + ',\n'.join(lines) + '\n}\n')
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m earshot.vocab', description=__doc__.splitlines()[0])
    ap.add_argument('--build-stats', nargs='+', metavar='REF', help='reference recordings (files / folders)')
    a = ap.parse_args(argv)
    if not a.build_stats:
        v = get()
        print(f'{len(v.names)} terms in {len(v.cats)} categories; reference stats: '
              f'{"%d recordings" % v.stats["n_ref"] if v.stats else "none"}')
        return 0
    paths = audio.list_media(a.build_stats)
    r = build_stats(embed_files(paths, max_win=3), paths, {'reference_set': ' '.join(a.build_stats), 'max_win': 3})
    print(f'wrote {STATS_PATH}: {r["n_ref"]} reference recordings from {r["n_folders"]} folders')
    return 0


if __name__ == '__main__':
    sys.exit(main())
