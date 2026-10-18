"""The ears of the forge: scores sounds made in code against a description and real reference recordings.

    from earshot.forge.critic import Critic
    c = Critic('a single deep thocky mechanical keyboard key press', alts=['a wooden knock', 'a mouse click'],
               refs='refs/keyboard/')
    rows = c.score([x1, x2])          # 48 kHz arrays (mono or stereo)
    rows[0]['total'], rows[0]['passes'], rows[0]['summary']
    Critic('a glitchy digital stutter', mode='designed')     # no refs / designed refs: realness not required
    c.consistency([x1, x2, x3])       # do the variants play at one loudness?

One CLAP embedding per sound feeds every model judgement (the vocabulary calibration and the realness head use the
same padding), so a batch of 16 sounds costs about one batch of CLAP windows. Model components (v1):
  match_z     calibrated z of the description among earshot's vocabulary (strong >= 3, as `earshot match`)
  alt_margin  match_z minus the best alternative's z (must be > 0: it sounds more like the target than the confusions)
  p_real      earshot realness (clip thresholds: real >= 0.77; the pure-tone rule applies)
  ref_typ_z   how close it sits to the nearest real references compared with how close the references sit to each other
  feat_pen    how far its measurements (attack, decay, brightness, length, low end, bandwidth) fall outside the
              references' spread (robust z beyond 1.5, averaged)
Model-free components (v2, measured from the samples; each its own number in the row):
  tone        earshot.forge.tone.ToneRef: 1/3-octave balance vs the refs' median (tone_db, tone_ratio, tone_pen)
  env         earshot.forge.tone.EnvRef: the spectro-temporal envelope (28 band trajectories in 10 ms frames) against
              the nearest ref after onset / peak alignment with a small stretch / shift tolerance (env_db, env_ratio),
              plus the cells (segment x band group) beyond every ref (env_out); env_pen = EnvRef.penalty
  artifacts   earshot.forge.polish: clicks, edge jumps, aliasing, DC / infrasound, harsh narrow resonances, comb ring,
              truncated tail, digital silence inside, clipping, frozen spectrum (art_pen; `hard` lists the defects
              that fail `passes`)
  polish      attack crispness, onset snap, tail smoothness, tail length, body / crest vs the refs (polish_pen)
Modes: 'real' (refs are real recordings: realness counts) or 'designed' (refs are designed SFX or missing: glitch,
sub drop, UI blips, risers; realness is reported but not required, clicks / digital silence are soft). mode='auto'
picks 'designed' when there are no refs or the target names a designed category (DESIGNED_WORDS).

Totals. version=1 reproduces the forge test's critic exactly:
  total = min(match_z, 6) + 2 * (2 * p_real - 1) + 0.5 * clip(ref_typ_z, -3, 2) - 0.7 * feat_pen - max(0, 0.5 - margin)
          [- tone * tone_pen when tone > 0]
  passes = match_z >= 3 and margin > 0 and p_real >= 0.77 and ref_typ_z >= -2 and feat_pen <= 1 [and the tone rule]
version=2 (default) keeps those terms (weights W2, realness weight 0 in designed mode) and subtracts the model-free
penalties; why each weight:
  match 1, real 2, typ 0.5, feat 0.7, alt 1   unchanged from v1 (calibrated in the forge test; the model checks saturate
                                              after a search, so they mostly decide early rounds)
  tone 1.0    always on in v2 (v1 recommended 1.5 when it was the only model-free check; env now carries part of the
              same evidence, band levels over time include the average balance, so 1.0 avoids counting it twice)
  env 1.0     one point per typical ref-to-ref spread of excess (env_ratio - 1) plus 1 per 6 dB beyond every ref; the
              same scale as tone. The shape over time is what the blind test heard (the forged impact's tail)
  art 1.0     each detector is capped at 2, so one blatant defect costs as much as the whole realness term; hard
              defects also fail `passes` (a click or a clipped peak disqualifies a library sound)
  polish 0.75 relative to the refs' ranges, which 3-6 refs pin down loosely: a little less than the artifacts
  passes (v2) = the v1 model rules (realness only in real mode) and no hard artifact and env_pen <= 1.5 and
              polish_pen <= 1.5 and tone_db <= 1.25 x the farthest ref's (or 2 x tone_target)
`weights=` overrides any W2 entry (weights={'tone': 0} to drop the tone term); tone= > 0 sets the tone weight (v1
callers keep their meaning).

Thin ref sets (category= / study=). Tone and envelope penalties are relative to the refs' own spread, which needs 3
refs: with 0-2 (impact, logo_sting, riser, sub_drop, glitch, downlifter and most pro_* categories) tone_pen / env_pen
were 0 and polish had no range, so the search was steered by CLAP and the artifact rules alone. Critic(category=...)
adds absolute fallback targets for those categories (STUDY_TARGETS: docs/motion-sfx-study.md "Targets for synthesis
recipes"; downlifter / glitch / logo_sting / riser / impact / sub_drop from the category definitions in refsets.py and
the recipes' study rows), measured by study_measures(): band shares (tone.shares regions, dB of the total) go into
tone_pen (1 per 6 dB outside the range, each capped at 2, sum at 3) when the row has no tone_ratio; length / attack /
t20 / t40 (log2 octaves outside the range) and the peak position (1 per 25 % outside), each capped at 1.5, sum at 2,
go into env_pen when the row has no env_ratio; stereo correlation (1 per 0.3 outside, capped at 1.5) goes into
polish_pen when there are fewer than 3 refs. With 1-2 refs every range is first widened to take in the refs' own
values (widen_targets; the other refs only when a ref is scored leave-one-out): the forge refs disagree with the
films the study measured in places (pro_whoosh_big refs fall 20 dB in 16 ms, pro_click / riser refs are wide stereo
trims). row['study'] keeps the measures, the targets used, the penalties and words. study= a {key: (lo, hi)} dict
overrides the table (study={} switches the fallback off); without category= / study= nothing changes (round-2
behaviour). gated= (default: category in GATED, designed mode only) lets a designed sound stop on a hard gate
(polish.GATE_DB). forge() passes its refset category when it builds the critic itself.

Fair comparison with the references: `score_refs()` scores every reference leave-one-out (its typicality, feature
spread, tone, envelope and polish ranges are computed from the OTHER references), so a real recording is not its own
nearest neighbour. With fewer than 3 other references there is no spread to compare with, and ref_typ_z / env_ratio
are None. Each row also carries `alt_zs` (the z of every alternative). `extra=fn(x, row) -> float` adds a
recipe-specific check to the total; its value is kept in row['extra'].

Honest limits: optimising against earshot finds sounds that please its models without sounding better (Goodhart);
the realness model is easy to fool (modal resonators plus a room and a mic floor read "real"). Nearest-ref distances
(typicality, env_db) flatter averages: with few, diverse refs a sound in their middle sits closer to each than they
sit to each other, so the v1 finals score env_ratio < 1 for pops, whooshes and impacts. The 'beyond every ref' cells,
the artifact detectors and the polish ranges are what still separate made from real there. `holdout()` uses the
AudioSet tagger, held out only while nobody picks or steers by its tags.
"""
import os

import numpy as np

from .. import audio, compare, features, realness, vocab
from . import polish
from .tone import EnvRef, ToneRef, bandgram

FEATS = ('attack_ms', 'decay_ms', 'brightness_hz', 'length_s', 'low_end_db', 'bandwidth_hz')
LOG = ('attack_ms', 'decay_ms', 'brightness_hz', 'length_s', 'bandwidth_hz')
W2 = {'match': 1.0, 'real': 2.0, 'typ': 0.5, 'feat': 0.7, 'alt': 1.0, 'tone': 1.0, 'env': 1.0, 'art': 1.0,
      'polish': 0.75}
DESIGNED_WORDS = ('glitch', 'stutter', 'sub drop', 'sub-drop', 'subdrop', 'blip', 'beep', 'bleep', 'synth', 'riser',
                  'downlifter', 'uplifter', 'stinger', 'logo sting', 'braam', 'laser', 'zap', 'notification', 'shimmer',
                  'bitcrush', 'digital')
ENV_PASS, POLISH_PASS, TONE_PASS = 1.5, 1.5, 1.25
GATED = ('glitch', 'pro_glitch', 'riser', 'pro_riser', 'zip', 'pro_zip', 'pro_impact')   # the stop is the design

# absolute fallback targets per category (used only where the refs give no spread, see the module doc). Keys:
# length s (active: first to last 10 ms frame within 50 dB of the loudest), attack / t20 / t40 ms (envelope_stats:
# onset at -30 dB to the peak; peak to -20 / -40 dB), peak_frac (peak time / active length), sh150 / sh1k / sh4k /
# sh12k / shhi (tone.shares '<150', '150-1k', '1-4k', '4-12k', '>12k' in dB of the total), corr (L/R correlation, 1 =
# mono). Ranges are the study's numbers widened by about a third (one or two films measured per row).
STUDY_TARGETS = {
    'pro_whoosh_big': {'length': (0.5, 1.2), 'attack': (100, 350), 't20': (60, 160), 't40': (180, 400),
                       'sh150': (-1.5, 0.0), 'corr': (0.8, 1.0)},
    'riser': {'length': (1.0, 6.0), 'peak_frac': (0.7, 1.0), 't40': (0, 400), 'corr': (0.5, 1.0)},
    'pro_riser': {'length': (0.8, 2.5), 'peak_frac': (0.75, 1.0), 't40': (0, 300), 'corr': (0.8, 1.0)},
    'impact': {'length': (1.0, 8.0), 't40': (1000, 3000), 'sh150': (-2.5, -0.2)},
    'pro_impact': {'length': (0.6, 1.3), 'attack': (40, 300), 't40': (250, 1100), 'sh150': (-1.25, 0.0),
                   'corr': (-0.1, 0.6)},
    'sub_drop': {'length': (0.8, 3.0), 'attack': (0, 450), 'sh150': (-1.0, 0.0), 'corr': (0.8, 1.0)},
    'pro_sub_drop': {'length': (0.8, 1.6), 'attack': (250, 460), 't40': (600, 1600), 'sh150': (-0.7, 0.0),
                     'corr': (0.8, 1.0)},
    'logo_sting': {'length': (0.8, 2.5), 't40': (250, 1200), 'corr': (0.6, 1.0)},
    'pro_logo_sting': {'length': (0.5, 1.5), 'attack': (0, 90), 't20': (30, 110), 't40': (150, 450),
                       'corr': (0.9, 1.0)},
    'glitch': {'length': (0.2, 1.5), 'sh150': (-200.0, -10.0), 'corr': (-0.3, 0.6)},
    'pro_glitch': {'length': (0.3, 1.0), 'sh150': (-200.0, -12.0), 'sh12k': (-8.0, 0.0), 'corr': (-0.3, 0.4)},
    'pro_shimmer': {'length': (1.4, 2.6), 'attack': (150, 800), 'sh150': (-200.0, -30.0), 'corr': (-0.2, 0.45)},
    'pro_pop': {'length': (0.05, 0.2), 'attack': (0, 60), 't20': (3, 30), 'corr': (0.9, 1.0)},
    'pro_tap': {'length': (0.06, 0.2), 'attack': (2, 15), 't20': (10, 30), 't40': (18, 50), 'corr': (0.85, 1.0)},
    'pro_click': {'length': (0.02, 0.08), 'attack': (0, 4), 't20': (4, 20), 'sh150': (-200.0, -20.0),
                  'sh12k': (-10.0, 0.0), 'corr': (0.9, 1.0)},
    'pro_typing_run': {'sh150': (-200.0, -25.0), 'sh1k': (-200.0, -20.0), 'corr': (0.9, 1.0)},
    'pro_zip': {'length': (0.12, 0.4), 'attack': (10, 50), 'corr': (0.85, 1.0)},
    'downlifter': {'length': (1.0, 4.0), 'attack': (0, 80), 't40': (500, 3500), 'corr': (0.3, 1.0)},
}
STUDY_TARGETS['zip'] = STUDY_TARGETS['pro_zip']
SHARE_KEYS = {'sh150': '<150', 'sh1k': '150-1k', 'sh4k': '1-4k', 'sh12k': '4-12k', 'shhi': '>12k'}


def study_measures(x):
    """the measures STUDY_TARGETS speaks in, for one 48 kHz sound (mono or stereo)"""
    from .search import envelope_stats
    from .tone import shares
    xa = np.asarray(x, np.float64)
    m = xa.mean(axis=1) if xa.ndim == 2 else xa
    k = len(m) // 480
    out = {}
    if k >= 2:
        e = 10 * np.log10((m[:k * 480].reshape(k, 480) ** 2).mean(1) + 1e-24)
        act = np.nonzero(e > e.max() - 50)[0]
        out['length'] = round(float((act[-1] - act[0] + 1) * 0.01), 3)
        out['peak_frac'] = round(float((int(np.argmax(e)) - act[0] + 0.5) / (act[-1] - act[0] + 1)), 3)
    env = envelope_stats(m)
    out.update(attack=round(env['attack_ms'], 1), t20=round(env['t20_ms'], 1), t40=round(env['t40_ms'], 1))
    sh = shares(m)
    out.update({kk: sh[v] for kk, v in SHARE_KEYS.items()})
    if xa.ndim == 2 and xa.shape[1] == 2:
        a, b = xa[:, 0] - xa[:, 0].mean(), xa[:, 1] - xa[:, 1].mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        out['corr'] = round(float((a * b).sum() / den), 3) if den > 0 else 1.0
    else:
        out['corr'] = 1.0
    return out


def widen_targets(targets, ref_measures):
    """the targets widened to take in every ref's own value (a trait the refs share is not a defect)"""
    out = {}
    for key, (lo, hi) in targets.items():
        vals = [s[key] for s in ref_measures if s.get(key) is not None]
        out[key] = (min([lo] + vals), max([hi] + vals))
    return out


def study_penalties(sm, targets):
    """({'tone', 'env', 'polish'} fallback penalties, {key: penalty} off target, words) for measures sm"""
    pen = {'tone': 0.0, 'env': 0.0, 'polish': 0.0}
    off, words = {}, []
    for key, (lo, hi) in targets.items():
        v = sm.get(key)
        if v is None:
            continue
        if key in SHARE_KEYS:
            p, part = min(2.0, max(lo - v, v - hi, 0.0) / 6.0), 'tone'
            say = f'{SHARE_KEYS[key]} share {v:+.1f} dB (target {lo:g}..{hi:g})'
        elif key == 'corr':
            p, part = min(1.5, max(lo - v, v - hi, 0.0) / 0.3), 'polish'
            say = f'stereo corr {v:.2f} (target {lo:g}..{hi:g})'
        elif key == 'peak_frac':
            p, part = min(1.5, max(lo - v, v - hi, 0.0) / 0.25), 'env'
            say = f'peak at {v:.0%} of the sound (target {lo:.0%}..{hi:.0%})'
        else:
            fl = 1e-3 if key == 'length' else 1.0                     # log ratios; ms below 1 count as 1
            vv, lo_, hi_ = max(v, fl), max(lo, fl), max(hi, lo, fl)
            p, part = min(1.5, max(np.log2(lo_ / vv), np.log2(vv / hi_), 0.0)), 'env'
            u = ' s' if key == 'length' else ' ms'
            say = f'{key} {v:g}{u} (target {lo:g}..{hi:g})'
        if p > 0:
            off[key] = round(float(p), 3)
            pen[part] += float(p)
            words.append(say)
    pen = {'tone': round(min(3.0, pen['tone']), 3), 'env': round(min(2.0, pen['env']), 3),
           'polish': round(min(1.5, pen['polish']), 3)}
    return pen, off, ('study targets: ' + ', '.join(words)) if words else 'study targets met'


def _mono(x):
    x = np.asarray(x, np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def _stereo(x):
    x = np.asarray(x, np.float32)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def _fv(k, v):
    return np.log(max(v, 1e-3)) if k in LOG else float(v)


_HOLD = {}


def holdout_key(mono):
    """cache key of one sound for Critic.holdout: sha1 of its mono float32 samples + the AudioSet model's cache name
    (org--name for the pinned model, so existing caches stay valid; a revision or a local folder gets its own)"""
    import hashlib
    from .. import config
    from ..models import hub
    m = np.ascontiguousarray(mono, np.float32)
    return hub.cache_name(config.AST_MODEL, config.AST_REVISION) + '/' + hashlib.sha1(m.tobytes()).hexdigest()


def _holdout_dir():
    from .. import config
    return os.path.join(config.EMB_CACHE, 'forge_holdout')


def _holdout_get(key):
    if key in _HOLD:
        return _HOLD[key]
    p = os.path.join(_holdout_dir(), key + '.json')
    if os.path.exists(p):
        import json
        try:
            with open(p, encoding='utf-8') as fh:
                v = json.load(fh)
            v['audioset_top'] = [tuple(a) if isinstance(a, list) else a for a in v['audioset_top']]
            _HOLD[key] = v
            return v
        except (OSError, ValueError, KeyError):
            return None
    return None


def _holdout_put(key, v):
    import json
    _HOLD[key] = v
    p = os.path.join(_holdout_dir(), key + '.json')
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f'.tmp{os.getpid()}'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(v, fh)
        os.replace(tmp, p)
    except OSError:
        pass


def resolve_mode(mode, target, refs):
    """'real' / 'designed' from mode='auto': designed when there are no refs or the target names a designed category"""
    if mode in ('real', 'designed'):
        return mode
    t = (target or '').lower()
    return 'designed' if not refs or any(w in t for w in DESIGNED_WORDS) else 'real'


def total_v2(r, W=None, mode='real'):
    """the v2 total and pass verdict from a row's components (pure; rows from Critic.score carry every input)"""
    W = {**W2, **(W or {})}
    mz, margin, p, typ = r['match_z'], r['alt_margin'], r['p_real'], r.get('ref_typ_z')
    real = W['real'] if mode == 'real' else 0.0
    total = (W['match'] * min(mz, 6.0) + real * (2 * p - 1) + (W['typ'] * float(np.clip(typ, -3, 2)) if typ is not None
             else 0.0) - W['feat'] * r['feat_pen'] - W['alt'] * max(0.0, 0.5 - margin)
             - W['tone'] * r.get('tone_pen', 0.0) - W['env'] * r.get('env_pen', 0.0)
             - W['art'] * r.get('art_pen', 0.0) - W['polish'] * r.get('polish_pen', 0.0))
    passes = bool(mz >= 3 and margin > 0 and (mode != 'real' or p >= 0.77) and (typ is None or typ >= -2)
                  and r['feat_pen'] <= 1.0 and not r.get('hard') and r.get('env_pen', 0.0) <= ENV_PASS
                  and r.get('polish_pen', 0.0) <= POLISH_PASS and not r.get('tone_far', False))
    return float(total), passes


class Critic:
    def __init__(self, target, alts=(), refs=None, k=3, extra=None, tone=0.0, tone_target=None, mode='auto',
                 version=2, weights=None, category=None, study=None, gated=None):
        self.target, self.alts = target, list(alts or [])
        self.extra = extra
        self.tone, self.tone_target = float(tone), tone_target
        self.version = int(version)
        self.W = {**W2, **(weights or {})}
        if self.tone > 0 and not (weights and 'tone' in weights):
            self.W['tone'] = self.tone
        self.v = vocab.get()
        self.Q = self.v.embed_text([target] + self.alts)
        self.m = realness.load_model()
        self.k = k
        self.refs = audio.list_media(refs, exts=audio.AUDIO_EXTS) if refs else []
        self.mode = resolve_mode(mode, target, self.refs)
        self.category = category
        self.study = dict(STUDY_TARGETS.get(category) or {}) if study is None else dict(study)
        self.gated = (category in GATED) if gated is None else bool(gated)
        self.A_r, self.ref_feats, self.ref_stats, self.tone_ref, self.env_ref = None, [], {}, None, None
        xs = []
        if self.refs:
            self.A_r = vocab.embed_files(self.refs)
            for p in self.refs:
                x = audio.load(p)
                xs.append(x)
                self.ref_feats.append(compare.feats(features.measure(x, light=True), x))
            self.ref_stats = self._stats(self.ref_feats)
            self.tone_ref = ToneRef(xs)
            self.env_ref = EnvRef(xs, names=self.refs)
        self.polish_ref = polish.PolishRef(xs, designed=self.mode == 'designed',
                                           gated=self.gated and self.mode == 'designed')

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
                A_r, refs, stats, min_refs = self.A_r, self.refs, self.ref_stats, 3   # 2 refs: one pair, no spread
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
                       total=round(float(total), 3), passes=passes, sounds_like=[t[0] for t in top], mode=self.mode)
            if ex is not None:
                row['left_out'] = os.path.basename(self.refs[ex])
            tm = self.tone_ref.check(x, exclude=ex) if self.tone_ref is not None and self.tone_ref.n else {}
            if tm:                            # {} when the only ref is the one left out (score_refs with 1 ref)
                if self.tone_target is not None:
                    tm['tone_ratio'] = round(tm['tone_db'] / float(self.tone_target), 2)
                row['tone'] = {k: tm[k] for k in ('tone_db', 'tone_ratio', 'outside', 't40_ms', 't40_ref', 'length_s',
                                                  'length_ref', 'shares', 'shares_ref_med', 'words')}
                row['tone_db'], row['tone_ratio'] = tm['tone_db'], tm['tone_ratio']
                tpen = self.tone_ref.penalty(tm)
                row['tone_pen'] = round(tpen, 3)
                far = max(tm['tone_ref']) if tm['tone_ref'] else None
                if self.tone_target is not None:          # median : farthest of a typical real set is ~1 : 1.6
                    far = 1.6 * float(self.tone_target)
                if self.version == 1 and self.tone > 0:
                    row['total'] = round(row['total'] - self.tone * tpen, 3)
                    if far is not None and tm['tone_db'] > far:
                        row['passes'] = False
                row['tone_far'] = bool(far is not None and tm['tone_db'] > TONE_PASS * far)
            self._model_free(x, row, ex)
            row['total_v1'] = row['total']
            if self.version >= 2:
                t2, p2 = total_v2(row, self.W, self.mode)
                row['total'], row['passes'] = round(t2, 3), p2
            if self.extra is not None:
                adj = float(self.extra(x, row))
                row['extra'] = round(adj, 3)
                row['total'] = round(float(row['total']) + adj, 3)
            row['summary'] = self._summary(row)
            rows.append(row)
        return rows

    def _model_free(self, x, row, ex):
        """the v2 measurements: spectro-temporal envelope, artifacts and polish (in place on the row)"""
        G = bandgram(x)
        if self.env_ref is not None and self.env_ref.n:
            em = self.env_ref.check(x, exclude=ex, G=G)
            if em:
                row['env'] = {k: em[k] for k in ('env_db', 'env_ratio', 'env_ref', 'env_out', 'env_out_ref', 'nearest',
                                                 'anchor', 'stretch', 'shift_ms', 'outside', 'words')}
                row['env_db'], row['env_ratio'], row['env_out'] = em['env_db'], em['env_ratio'], em['env_out']
                row['env_pen'] = round(EnvRef.penalty(em), 3)
        pm = self.polish_ref.check(x, exclude=ex, G=G)
        row['artifacts'] = {'pen': pm['art'], 'hard': pm['hard'], 'limits': {k: v for k, v in pm['limits'].items()
                                                                              if k in pm['art']}}
        row['polish'] = {'pen': pm['polish'], 'limits': {k: v for k, v in pm['limits'].items() if k in pm['polish']}}
        row['measures'] = {k: v for k, v in pm['measures'].items() if k not in ('res_peaks',)}
        row['art_pen'], row['polish_pen'], row['hard'] = pm['art_pen'], pm['polish_pen'], pm['hard']
        row['polish_words'] = pm['words']
        self._study(x, row, ex)

    def _study(self, x, row, ex):
        """absolute fallback targets where the refs give no spread (module doc: thin ref sets), in place"""
        n_refs = len(self.refs) - (ex is not None)
        if not self.study or n_refs >= 3:
            return
        if getattr(self, '_ref_sm', None) is None:
            self._ref_sm = [study_measures(audio.load(p)) for p in self.refs]
        tg = widen_targets(self.study, [s for j, s in enumerate(self._ref_sm) if j != ex])
        sm = study_measures(x)
        pen, off, words = study_penalties(sm, tg)
        used = {}
        if row.get('tone_ratio') is None and pen['tone'] > 0:
            row['tone_pen'] = round(row.get('tone_pen', 0.0) + pen['tone'], 3)
            used['tone'] = pen['tone']
        if row.get('env_ratio') is None and pen['env'] > 0:
            row['env_pen'] = round(row.get('env_pen', 0.0) + pen['env'], 3)
            used['env'] = pen['env']
        if pen['polish'] > 0:
            row['polish_pen'] = round(row.get('polish_pen', 0.0) + pen['polish'], 3)
            used['polish'] = pen['polish']
        row['study'] = {'measures': sm, 'targets': tg, 'off': off, 'pen': used, 'words': words}

    def _summary(self, r):
        bits = [f"match z {r['match_z']:+.2f}" + (f" (best alternative {r['alt_z']:+.2f})" if r['alt_z'] is not None else ''),
                f"p_real {r['p_real']:.2f}" + (' (not required: designed)' if r.get('mode') == 'designed' else '')]
        if r['ref_typ_z'] is not None:
            bits.append(f"typicality vs refs z {r['ref_typ_z']:+.2f} (nearest {r['nearest_ref']})")
        off = [f'{k} {v:+.1f}' for k, v in r['feat_dev'].items() if abs(v) > 1.5]
        if off:
            bits.append('outside the refs: ' + ', '.join(off))
        bits.append('sounds like: ' + ', '.join(r['sounds_like']))
        if r.get('tone'):
            bits.append(r['tone']['words'])
        if r.get('env'):
            bits.append(r['env']['words'])
        if r.get('polish_words'):
            bits.append(r['polish_words'])
        if r.get('study') and r['study'].get('off'):
            bits.append(r['study']['words'])
        return ('PASS ' if r['passes'] else '') + f"total {r['total']:.2f}: " + '; '.join(bits)

    def score_files(self, paths):
        return self.score([audio.load(p) for p in paths])

    def score_refs(self, loo=True):
        """the same critic's scores of the real references, in self.refs order. loo=True (fair): each reference is
        compared with the OTHER references only; loo=False counts it as its own nearest neighbour (inflated).
        Memoised per critic (a family run calls forge() once per member with one critic: the refs were re-embedded
        and re-scored every time); the memo is keyed by loo, the weights, mode, version and the extra term"""
        import copy
        key = (bool(loo), tuple(sorted(self.W.items())), self.mode, self.version, id(self.extra), tuple(self.refs))
        memo = getattr(self, '_ref_memo', None)
        if memo is not None and memo[0] == key:
            return copy.deepcopy(memo[1])
        xs = [audio.load(p) for p in self.refs]
        rows = self.score(xs, exclude=list(range(len(xs))) if loo else None)
        self._ref_memo = (key, copy.deepcopy(rows))
        return rows

    def consistency(self, xs):
        """loudness consistency of variants meant to be swapped in an edit: momentary-max LUFS per variant at its own
        level, spread and std (polish.loudness_spread), with the refs' own spread when there are refs"""
        out = polish.loudness_spread(xs)
        if self.refs:
            out['refs'] = polish.loudness_spread(self.refs)
        return out

    def holdout(self, xs, cache=True):
        """checks that are NOT in the objective: AudioSet top labels and its synth-class probability (held out only
        while they decide nothing: fix the selection rule before reading them). cache=True keeps each result by a
        hash of the exact mono float32 samples (in memory and under EARSHOT_HOME/emb/forge_holdout/<model>/), so the
        refs' tags, which every forge() call used to recompute, cost nothing after the first run"""
        out = [None] * len(xs)
        monos = [np.ascontiguousarray(_mono(x), np.float32) for x in xs]
        keys = [holdout_key(m) for m in monos] if cache else [None] * len(xs)
        for i, k in enumerate(keys):
            if k is not None:
                out[i] = _holdout_get(k)
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            from ..models import ast
            t = ast.get()
            P = t.probs([monos[i] for i in todo])
            for i, p in zip(todo, P):
                synth = max(t.prob_of(p, lab) for lab in realness.AST_SYNTH if lab in t.index)
                out[i] = {'audioset_top': t.top(p, 3), 'audioset_synth_max': round(float(synth), 3)}
                if keys[i] is not None:
                    _holdout_put(keys[i], out[i])
        return [dict(o) for o in out]
