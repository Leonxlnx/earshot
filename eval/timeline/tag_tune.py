"""Choose how timeline tags events: excerpt treatment x vocabulary, scored on the synthetic mixes (mixes.py; every
one-shot carries its library category). Never uses the Clawd mix.

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/tag_tune.py --vocab compact --variants delta,repeat,zero,iso

vocab     json = all event categories of earshot/data/vocab.json (kind from the category), compact = timeline's own
          EVENT_VOCAB (kind = its key)
variants  zero / repeat = CLAP on the event excerpt, zero- or repeat-padded to 10 s (repeat: what timeline ships);
          delta = zero-padded cosine minus the same for the bed just before the event (the first build);
          deltarep = the same with repeat padding;
          win = minus the 3 s section window around the event; debed = excerpt after subtracting the bed's average
          spectrum; iso = the one-shot alone (an upper bound: no bed at all)
--center  subtract this share of each phrase's mean cosine over the file's events (hub-phrase correction)
A tag is plausible when the kind of the top phrase fits the one-shot's library category (FIT below).
Scores here are the first build's (z within each event); timeline now also damps hub phrases (tag_bg.py).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from earshot import audio, timeline as tl  # noqa: E402
from earshot.models import clap  # noqa: E402
import mixes  # noqa: E402

FIT = {'keyboard': {'KEY'}, 'motion': {'WHOOSH', 'RISER'}, 'garden': {'STEP', 'FOLEY'},
       'toy': {'FOLEY', 'UI'}, 'magic': {'NOTE', 'UI'}, 'foley': {'FOLEY', 'STEP', 'UI'}}
JSON_CATS = ('transitions', 'impacts', 'ui', 'foley', 'instruments')


def json_kind(term, cat):
    """kind of a vocab.json phrase, for comparing against the compact vocabulary"""
    if cat == 'transitions':
        return 'RISER' if any(w in term for w in ('riser', 'uplifter', 'swell')) else 'WHOOSH'
    if cat == 'impacts':
        return 'HIT'
    if cat == 'ui':
        return 'KEY' if any(w in term for w in ('keyboard', 'key press', 'spacebar', 'typing')) else 'UI'
    if cat == 'foley':
        return 'STEP' if 'footsteps' in term else 'FOLEY'
    return 'NOTE'


def vocabulary(c, which):
    if which == 'compact':
        terms, kinds, E = tl._event_matrix(c)
        return terms, [str(k) for k in kinds], E
    terms, cats, E = tl._vocab_matrix(c)
    keep = [i for i, k in enumerate(cats) if k in JSON_CATS]
    return [terms[i] for i in keep], [json_kind(terms[i], cats[i]) for i in keep], E[keep]


def debed(x, a, b, pre_s=0.5):
    """excerpt with the bed's average magnitude spectrum (pre_s before it) subtracted, -20 dB floor, original phase"""
    seg = tl._seg(x, a, b).astype(np.float64)
    pre = tl._seg(x, max(0.0, a - pre_s), max(0.0, a - 0.01)).astype(np.float64) if a > 0.1 else np.zeros(0)
    if len(pre) < 2048 or len(seg) < 1024:
        return seg.astype(np.float32)
    _, _, P = signal.stft(pre, nperseg=1024, noverlap=768)
    _, _, S = signal.stft(seg, nperseg=1024, noverlap=768)
    mag = np.abs(S)
    y = signal.istft(np.maximum(mag - np.abs(P).mean(axis=1, keepdims=True), 0.1 * mag) * np.exp(1j * np.angle(S)),
                     nperseg=1024, noverlap=768)[1]
    return y[: len(seg)].astype(np.float32)


def iso(g):
    for d in (os.path.join(mixes.ASSETS, g['cat']), os.path.join(mixes.ASSETS, '_rejected', g['cat'])):
        f = os.path.join(d, g['file'])
        if os.path.exists(f):
            got = mixes.whooshes(f) if g['cat'] == 'motion' else mixes.one_shots(f)
            return got[0][0] if got else np.zeros(4800, np.float32)
    return np.zeros(4800, np.float32)


def cosines(v, c, E, x, fid, spans, picks):
    def emb(kind, sp, make, pad):
        return tl._cached(c, fid, sp, kind, lambda q: c.embed_audio([make(p, r) for p, r in q], pad=pad))

    def seg(p, r):
        return tl._seg(x, p, r)
    pre = [(max(0.0, p - (q - p) - 0.01), max(0.01, p - 0.01)) for p, q in spans]
    if v == 'iso':
        return c.embed_audio([iso(g) for g, _ in picks], pad='zero') @ E.T
    if v == 'delta':
        return emb('clap-zero', spans, seg, 'zero') @ E.T - emb('clap-zero', pre, seg, 'zero') @ E.T
    if v == 'deltarep':
        return emb('clap-repeat', spans, seg, 'repeatpad') @ E.T - emb('clap-repeat', pre, seg, 'repeatpad') @ E.T
    if v == 'win':
        dur = len(x) / tl.SR
        starts = list(np.arange(0.0, max(dur - 3.0, 0.0) + 1e-6, 2.0))
        if starts[-1] + 3.0 < dur - 0.5:
            starts.append(max(0.0, dur - 3.0))
        wsp = [(round(a, 3), round(min(dur, a + 3.0), 3)) for a in starts]
        W = emb('clap-repeat', wsp, seg, 'repeatpad')
        mids = np.array([(a + b) / 2 for a, b in wsp])
        ref = np.stack([W[int(np.argmin(np.abs(mids - (p + q) / 2)))] for p, q in spans])
        return emb('clap-repeat', spans, seg, 'repeatpad') @ E.T - ref @ E.T
    if v == 'debed':
        return emb('clap-debed', spans, lambda p, r: debed(x, p, r), 'zero') @ E.T
    return emb(f'clap-{v}', spans, seg, 'repeatpad' if v == 'repeat' else 'zero') @ E.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out/timeline/mixes')
    ap.add_argument('--vocab', default='compact', choices=('compact', 'json'))
    ap.add_argument('--variants', default='delta,repeat,zero')
    ap.add_argument('--center', type=float, default=0.0)
    a = ap.parse_args()
    c = clap.get()
    terms, kinds, E = vocabulary(c, a.vocab)
    rows = {v: [] for v in a.variants.split(',')}
    for f in sorted(glob.glob(os.path.join(a.dir, 'mix_*.json'))):
        meta = json.load(open(f))
        x = audio.load(f[:-5] + '.wav')
        ev = tl.detect_events(x)
        det = np.array([e['t'] for e in ev])
        picks = [(g, ev[int(np.argmin(np.abs(det - g['t'])))]) for g in meta['events']
                 if np.min(np.abs(det - g['t'])) <= (0.12 if g['align'] == 'peak' else 0.03)]
        spans = [tl._event_span(e, len(x) / tl.SR) for _, e in picks]
        for v in rows:
            cos = cosines(v, c, E, x, audio.file_id(f[:-5] + '.wav'), spans, picks)
            if a.center:
                cos = cos - a.center * cos.mean(axis=0, keepdims=True)
            z = (cos - cos.mean(axis=1, keepdims=True)) / (cos.std(axis=1, keepdims=True) + 1e-9)
            for j, (g, _) in enumerate(picks):
                idx = np.argsort(-z[j])[:3]
                rows[v].append({'cat': g['cat'], 'snr': g['snr_db'], 'top1': kinds[idx[0]] in FIT[g['cat']],
                                'top3': any(kinds[i] in FIT[g['cat']] for i in idx), 'tag': terms[idx[0]]})
    for v, r in rows.items():
        t1, t3 = np.array([q['top1'] for q in r]), np.array([q['top3'] for q in r])
        snr = np.array([q['snr'] for q in r])
        print(v, json.dumps({'vocab': a.vocab, 'center': a.center, 'n': len(r), 'top1': round(float(t1.mean()), 3),
                             'top3': round(float(t3.mean()), 3), 'top1_snr_ge0': round(float(t1[snr >= 0].mean()), 3),
                             'top1_snr_lt0': round(float(t1[snr < 0].mean()), 3),
                             'per_category': {k: round(float(np.mean([q['top1'] for q in r if q['cat'] == k])), 2)
                                              for k in sorted({q['cat'] for q in r})},
                             'most_common_tags': Counter(q['tag'] for q in r).most_common(4)}), flush=True)


if __name__ == '__main__':
    main()
