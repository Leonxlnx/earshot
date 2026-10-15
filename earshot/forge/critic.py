"""The ears of the forge: scores sounds made in code against a description and real reference recordings.

    from earshot.forge.critic import Critic
    c = Critic('a single deep thocky mechanical keyboard key press', alts=['a wooden knock', 'a mouse click'],
               refs='refs/keyboard/')
    rows = c.score([x1, x2])          # 48 kHz arrays (mono or stereo)
    rows[0]['total'], rows[0]['passes'], rows[0]['summary']

One CLAP embedding per sound feeds every model judgement (the vocabulary calibration and the realness head use the
same padding), so a batch of 16 sounds costs about one batch of CLAP windows. Components:
  match_z     calibrated z of the description among earshot's vocabulary (strong >= 3, as `earshot match`)
  alt_margin  match_z minus the best alternative's z (must be > 0: it sounds more like the target than the confusions)
  p_real      earshot realness (clip thresholds: real >= 0.77; the pure-tone rule applies)
  ref_typ_z   how close it sits to the nearest real references compared with how close the references sit to each other
  feat_pen    how far its measurements (attack, decay, brightness, length, low end, bandwidth) fall outside the
              references' spread (robust z beyond 1.5, averaged)
  total       min(match_z, 6) + 2 * (2 * p_real - 1) + 0.5 * clip(ref_typ_z, -3, 2) - 0.7 * feat_pen
              - max(0, 0.5 - alt_margin)
  passes      match_z >= 3 and alt_margin > 0 and p_real >= 0.77 and ref_typ_z >= -2 and feat_pen <= 1
Optimising against earshot can find sounds that please its models without sounding better (Goodhart). `holdout()`
uses the AudioSet tagger, which is not in the objective, but it stays held out only while nobody picks or steers by
its tags (in the forge test it did steer four of five sounds, which biased their tags). Once a search has run, the
model checks saturate: held-out phrasings and independent real sets agree too. The model-free `tone` check below is
the signal that still separates made from real. The realness
model in particular is easy to fool: modal resonators plus a room and a mic noise floor read "real" (p_real > 0.9)
whether or not they sound like the target, so trust reference closeness, the held-out tags and spectrograms more.

Fair comparison with the references: `score_refs()` scores every reference leave-one-out (its typicality and the
feature spread are computed from the OTHER references), so a real recording is not its own nearest neighbour. With
fewer than 3 other references there is no spread to compare with, and ref_typ_z is None. Each row also carries
`alt_zs` (the z of every alternative, so you can see which confusion is closest). `extra=fn(x, row) -> float` adds a
recipe-specific check (a model-free distance to the refs, say) to the total; its value is kept in row['extra'].

Model-free ears (`tone=`): every row also carries row['tone'] from earshot.forge.tone (1/3-octave tonal balance against
the refs' median, which bands fall outside every ref, the tail's t40), measured from the samples, no model. The model
checks above saturate once a search has tuned against them (made sounds land inside or above the real recordings on
every one, held-out ones included), while these measurements still show the gap. tone=W (default 0, the scores the
forge test reported) subtracts W * ToneRef.penalty() from the total (0 while the sound is no farther from the refs'
median balance than a typical real recording, then linear) and, with W > 0, `passes` also needs tone_db no larger
than the farthest real reference's. tone_target= sets that "typical" distance in dB (and the pass limit to 1.6x it,
the median : farthest proportion of the six real key presses) when the refs are too few or too different from each
other for their own spread to mean anything (the four kit pops sit 15-24 dB apart).
"""
import os

import numpy as np

from .. import audio, compare, features, realness, vocab
from .tone import ToneRef

FEATS = ('attack_ms', 'decay_ms', 'brightness_hz', 'length_s', 'low_end_db', 'bandwidth_hz')
LOG = ('attack_ms', 'decay_ms', 'brightness_hz', 'length_s', 'bandwidth_hz')


def _mono(x):
    x = np.asarray(x, np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def _stereo(x):
    x = np.asarray(x, np.float32)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def _fv(k, v):
    return np.log(max(v, 1e-3)) if k in LOG else float(v)


class Critic:
    def __init__(self, target, alts=(), refs=None, k=3, extra=None, tone=0.0, tone_target=None):
        self.target, self.alts = target, list(alts or [])
        self.extra = extra
        self.tone, self.tone_target = float(tone), tone_target
        self.v = vocab.get()
        self.Q = self.v.embed_text([target] + self.alts)
        self.m = realness.load_model()
        self.k = k
        self.refs = audio.list_media(refs, exts=audio.AUDIO_EXTS) if refs else []
        self.A_r, self.ref_feats, self.ref_stats, self.tone_ref = None, [], {}, None
        if self.refs:
            self.A_r = vocab.embed_files(self.refs)
            xs = []
            for p in self.refs:
                x = audio.load(p)
                xs.append(x)
                self.ref_feats.append(compare.feats(features.measure(x, light=True), x))
            self.ref_stats = self._stats(self.ref_feats)
            self.tone_ref = ToneRef(xs)

    @staticmethod
    def _stats(ref_feats):
        st = {}
        for key in FEATS:
            vals = np.array([_fv(key, f[key]) for f in ref_feats if f.get(key) is not None])
            if len(vals) >= 2:
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                st[key] = (float(med), max(float(q3 - q1) / 1.35, 0.15 if key in LOG else 1.5))
        return st

    def embed(self, xs, batch=16):
        from ..models import clap
        return clap.get().embed_audio([_mono(x) for x in xs], pad=vocab.PAD, batch=batch)

    def score(self, xs, E=None, exclude=None):
        """one result dict per sound (silent or broken sounds get total -inf). exclude: per sound, the index of a
        reference to leave out of its comparison (leave-one-out scoring of the references themselves) or None"""
        xs = list(xs)
        E = self.embed(xs) if E is None else E
        cal = self.v.calibrate(E, self.Q)
        hs = self.m['heads']
        rows = []
        exclude = list(exclude) if exclude is not None else [None] * len(xs)
        loo = {}
        for i, x in enumerate(xs):
            ex = exclude[i]
            if ex is None:
                A_r, refs, stats, min_refs = self.A_r, self.refs, self.ref_stats, 2
            else:
                if ex not in loo:
                    keep = [j for j in range(len(self.refs)) if j != ex]
                    loo[ex] = (self.A_r[keep] if self.A_r is not None else None, [self.refs[j] for j in keep],
                               self._stats([self.ref_feats[j] for j in keep]))
                (A_r, refs, stats), min_refs = loo[ex], 3
            mono = _mono(x)
            row = {'i': i}
            if not len(mono) or not realness.audible(mono):
                rows.append({**row, 'total': -np.inf, 'passes': False, 'summary': 'silent'})
                continue
            z = cal['z_q'][i]
            mz = float(z[0])
            alt = float(z[1:].max()) if len(z) > 1 else -np.inf
            f = realness.signal_features(mono)
            lc, _ = realness.head_logit(hs['clap'], E[i])
            ls, _ = realness.head_logit(hs['signal'], realness._sig_vector(hs['signal'], f)) if f else (0.0, None)
            p, _ = realness.fuse(self.m['fusion'], {'clap': lc, 'signal': ls})
            if f and realness.pure_tone(f):
                p = min(p, 0.03)
            st = _stereo(x)
            fx = compare.feats(features.measure(st, light=True), st)
            pen, dev = 0.0, {}
            for key, (med, sd) in stats.items():
                if fx.get(key) is None:
                    continue
                zz = (_fv(key, fx[key]) - med) / sd
                dev[key] = round(float(zz), 2)
                pen += max(0.0, abs(zz) - 1.5)
            pen = pen / max(1, len(dev))
            typ, near = None, None
            if A_r is not None and len(A_r) >= min_refs:
                _, typ, _ = compare._typicality(E[i], A_r, self.k)
                near = os.path.basename(refs[int(np.argmax(A_r @ E[i]))])
            margin = mz - alt if np.isfinite(alt) else 9.0
            total = (min(mz, 6.0) + 2.0 * (2 * p - 1) + (0.5 * float(np.clip(typ, -3, 2)) if typ is not None else 0.0)
                     - 0.7 * pen - max(0.0, 0.5 - margin))
            passes = bool(mz >= 3 and margin > 0 and p >= 0.77 and (typ is None or typ >= -2) and pen <= 1.0)
            top = self.v.top_terms(E[i], k=3)
            row.update(match_z=round(mz, 2), alt_z=round(alt, 2) if np.isfinite(alt) else None,
                       alt_zs={a: round(float(z[j + 1]), 2) for j, a in enumerate(self.alts)},
                       alt_margin=round(margin, 2), p_real=round(float(p), 3),
                       ref_typ_z=None if typ is None else round(float(typ), 2), nearest_ref=near,
                       feat_pen=round(pen, 2), feat_dev=dev, feats={k: fx.get(k) for k in FEATS if fx.get(k) is not None},
                       total=round(float(total), 3), passes=passes, sounds_like=[t[0] for t in top])
            if ex is not None:
                row['left_out'] = os.path.basename(self.refs[ex])
            if self.tone_ref is not None and self.tone_ref.n:
                tm = self.tone_ref.check(x, exclude=ex)
                if self.tone_target is not None:
                    tm['tone_ratio'] = round(tm['tone_db'] / float(self.tone_target), 2)
                row['tone'] = {k: tm[k] for k in ('tone_db', 'tone_ratio', 'outside', 't40_ms', 't40_ref', 'length_s',
                                                  'length_ref', 'shares', 'shares_ref_med', 'words')}
                row['tone_db'], row['tone_ratio'] = tm['tone_db'], tm['tone_ratio']
                if self.tone > 0:
                    tpen = self.tone_ref.penalty(tm)
                    row['tone_pen'] = round(tpen, 3)
                    row['total'] = round(row['total'] - self.tone * tpen, 3)
                    far = max(tm['tone_ref']) if tm['tone_ref'] else None
                    if self.tone_target is not None:          # median : farthest of a typical real set is ~1 : 1.6
                        far = 1.6 * float(self.tone_target)
                    if far is not None and tm['tone_db'] > far:
                        row['passes'] = False
            if self.extra is not None:
                adj = float(self.extra(x, row))
                row['extra'] = round(adj, 3)
                row['total'] = round(float(row['total']) + adj, 3)
            row['summary'] = self._summary(row)
            rows.append(row)
        return rows

    def _summary(self, r):
        bits = [f"match z {r['match_z']:+.2f}" + (f" (best alternative {r['alt_z']:+.2f})" if r['alt_z'] is not None else ''),
                f"p_real {r['p_real']:.2f}"]
        if r['ref_typ_z'] is not None:
            bits.append(f"typicality vs refs z {r['ref_typ_z']:+.2f} (nearest {r['nearest_ref']})")
        off = [f'{k} {v:+.1f}' for k, v in r['feat_dev'].items() if abs(v) > 1.5]
        if off:
            bits.append('outside the refs: ' + ', '.join(off))
        bits.append('sounds like: ' + ', '.join(r['sounds_like']))
        if r.get('tone'):
            bits.append(r['tone']['words'])
        return ('PASS ' if r['passes'] else '') + f"total {r['total']:.2f}: " + '; '.join(bits)

    def score_files(self, paths):
        return self.score([audio.load(p) for p in paths])

    def score_refs(self, loo=True):
        """the same critic's scores of the real references, in self.refs order. loo=True (fair): each reference is
        compared with the OTHER references only; loo=False counts it as its own nearest neighbour (inflated)"""
        xs = [audio.load(p) for p in self.refs]
        return self.score(xs, exclude=list(range(len(xs))) if loo else None)

    def holdout(self, xs):
        """checks that are NOT in the objective: AudioSet top labels and its synth-class probability (held out only
        while they decide nothing: fix the selection rule before reading them)"""
        from ..models import ast
        t = ast.get()
        P = t.probs([_mono(x) for x in xs])
        out = []
        for p in P:
            synth = max(t.prob_of(p, lab) for lab in realness.AST_SYNTH if lab in t.index)
            out.append({'audioset_top': t.top(p, 3), 'audioset_synth_max': round(float(synth), 3)})
        return out
