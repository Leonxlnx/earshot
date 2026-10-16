"""The fixed forge benchmark: made sounds per category against HELD-OUT real refs, with checks the forge never optimises.

    python eval/forge/bench.py --made made/ --out eval/forge/results_round3.json      # made/<category>/*.wav
    python eval/forge/bench.py --files typing_key="demo/forge-test/keypress/final/*.wav" click=... --out ...
    python eval/forge/bench.py --real-only --out eval/forge/results_real.json        # only the real ceilings

Refs: earshot.forge.refsets (the 'bench' split: source recordings the forge never sees; see eval/forge/build_refsets.py).
Every made sound gets, against its category's bench refs (no model in the objective of any forge recipe here):
  typ_z       CLAP typicality: closeness to the nearest 3 bench refs vs how close the bench refs sit to each other
              (compare._typicality; None with fewer than 3 refs)
  tone_db     1/3-octave tonal balance distance to the bench refs' median (eval/forge/frozen_tone.py: earshot/forge/
              tone.py frozen at round 0, so the forge's own evolving tone.py never moves the benchmark), tone_ratio = that /
              the refs' own leave-one-out distance, n_outside = bands beyond every ref by > 3 dB
  env_oct     envelope shape: mean |log2(made / bench median)| of attack, t40 (peak to -40 dB) and active length
              (1.0 = typically 2x off); curve_db = RMS dB distance of the peak-aligned 10 ms envelope (floored at -60 dB)
              to the bench median envelope, which is where a tail that dies too fast shows
  phrase_z    calibrated CLAP z of 2-3 independent phrasings per category (PHRASES, never a recipe's TARGET / ALTS:
              checked at start), margin = phrase z minus the best of the category's CONFUSIONS, wins = share of
              phrasings with z >= 2.5 and above every confusion
  as_names    whether an AudioSet top-3 label names the category (AS_LABELS), as_p its best label probability
  artifacts   clipped, dc offset, start click (first sample non-zero), end click / tail cut (still sounding at the
              end; not for risers, which stop on their peak), late start (nothing within 50 dB of the peak in the
              first 100 ms: bad for frame sync); calibrated on the real trims (see artifacts())
  diversity   among a category's variants: CLAP cosine mean / max, tonal-balance distance min, near duplicates
              (var cos in the table, next to the forge refs' and the bench refs' own: real sets sit ~0.5-0.8)
Context ("real ceiling"), with the very same functions: real_cross = the category's FORGE refs scored against the
bench refs (real recordings the bench never saw, the same yardstick as the made sounds); real_loo = each bench ref
against the other bench refs; real_rev = the bench refs against the forge refs (stored in the JSON only).
texture     floor_db (the recording floor; digital silence = -120), stereo_corr, darken (does the decay darken)
inside = per made sound, how many checks land inside the real rows' 80% limits (the 80th percentile of real_cross +
real_loo for lower-is-better checks, the 20th for higher-is-better), next to the same count for the real rows.
Writes the JSON (--out) and a markdown table next to it (.md). Models: CLAP + AudioSet (EARSHOT_THREADS=2 is gentle).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE)).replace(chr(92), '/')
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import frozen_tone as tone  # noqa: E402  (a frozen copy of earshot/forge/tone.py: the benchmark must not drift)

SR = 48000
PHRASES = {   # independent of the recipes' TARGET / ALTS and of build_refsets' find queries
    'whoosh_fast': ['a quick swoosh as something flies past', 'a short sharp swoosh of air'],
    'whoosh_big': ['a heavy swooping rush of air', 'a big slow swoosh in a movie trailer'],
    'swish': ['a thin airy swish', 'a light quick swoosh of a thrown object'],
    'swipe': ['fabric rustling in one quick swipe', 'a sleeve brushing past quickly'],
    'riser': ['a tension build-up getting louder and louder', 'a whooshing crescendo that swells to a climax'],
    'downlifter': ['a descending whoosh that fades out', 'a falling pitch sweep winding down'],
    'impact': ['a huge low thud for a movie trailer', 'a massive hit with a rumbling tail'],
    'sub_drop': ['a low frequency boom that drops in pitch', 'a deep rumbling bass drop'],
    'click': ['a crisp tiny plastic click', 'a short clean click of a button being pressed'],
    'tap': ['a fingertip tapping on glass', 'a light tap of a finger on a phone screen'],
    'toggle': ['a light switch clicking on', 'a small switch being flipped'],
    'hover_tick': ['a very short faint tick', 'a tiny dry click'],
    'pop': ['a popping sound like a bubble bursting', 'a quick plop'],
    'notify': ['a pleasant chime alert sound', 'a single bell ding on a phone'],
    'shimmer': ['sparkling twinkling chimes', 'a glittering magical sound'],
    'typing_key': ['one key being pressed on a computer keyboard', 'a single keystroke on a mechanical keyboard'],
    'glitch': ['a digital glitch with stuttering noise', 'a broken electronic signal stutter'],
    'logo_sting': ['a short musical logo with a ringing finish', 'a bright struck bell ringing out'],
}
CONFUSIONS = {
    'whoosh_fast': ['a steam hiss', 'a car driving past', 'tearing paper', 'a bird flapping its wings'],
    'whoosh_big': ['a strong wind storm', 'an airplane taking off', 'ocean waves crashing', 'a car driving past'],
    'swish': ['a short hiss', 'a sword being drawn', 'a page being turned', 'a bird flying past'],
    'swipe': ['paper crumpling', 'a broom sweeping', 'a gust of wind', 'a zipper'],
    'riser': ['a jet engine starting', 'a siren', 'steady rain', 'an orchestra tuning'],
    'downlifter': ['a siren', 'a car passing by', 'a falling bomb whistle', 'wind'],
    'impact': ['a gunshot', 'a door being slammed', 'a bass drum', 'distant fireworks'],
    'sub_drop': ['an earthquake rumble', 'a kick drum', 'a truck engine idling', 'thunder in the distance'],
    'click': ['a wooden stick hit', 'a finger snap', 'a drop of water', 'a stapler'],
    'tap': ['a knock on a door', 'a raindrop', 'a keyboard key', 'a ballpoint pen click'],
    'toggle': ['a door latch', 'a stapler', 'a seatbelt buckle', 'a ball bouncing'],
    'hover_tick': ['a clock ticking', 'a drop of water', 'a typewriter', 'a cricket chirp'],
    'pop': ['a hand clap', 'a gunshot', 'a kiss', 'a tongue click'],
    'notify': ['a doorbell', 'a microwave beep', 'a church bell', 'a cash register'],
    'shimmer': ['a harp glissando', 'breaking glass', 'a music box', 'a sleigh bell jingle'],
    'typing_key': ['a ballpoint pen click', 'a stapler', 'a light switch', 'a dripping tap'],
    'glitch': ['radio static', 'a modem dialing', 'a scratched record', 'an electric buzz'],
    'logo_sting': ['a church bell', 'a doorbell', 'a gong', 'a cymbal crash'],
}
AS_LABELS = {   # AudioSet classes that name the category (riser / downlifter / glitch have none exact: nearest)
    'whoosh_fast': ['Whoosh, swoosh, swish', 'Whip'], 'whoosh_big': ['Whoosh, swoosh, swish'],
    'swish': ['Whoosh, swoosh, swish', 'Whip'], 'swipe': ['Whoosh, swoosh, swish', 'Rustle'],
    'riser': ['Whoosh, swoosh, swish', 'Cymbal', 'Drum roll'], 'downlifter': ['Whoosh, swoosh, swish'],
    'impact': ['Boom', 'Thump, thud', 'Explosion', 'Bang', 'Bass drum'], 'sub_drop': ['Boom', 'Rumble', 'Thump, thud'],
    'click': ['Clicking', 'Tick', 'Tap'], 'tap': ['Tap', 'Knock', 'Clicking', 'Tick'],
    'toggle': ['Clicking', 'Mechanisms', 'Tick'], 'hover_tick': ['Tick', 'Clicking', 'Tick-tock'],
    'pop': ['Burst, pop', 'Plop'],
    'notify': ['Ding', 'Chime', 'Bell', 'Glockenspiel', 'Jingle, tinkle', 'Ding-dong', 'Ringtone', 'Marimba, xylophone',
               'Vibraphone'],
    'shimmer': ['Chime', 'Wind chime', 'Jingle, tinkle', 'Glockenspiel', 'Bell', 'Jingle bell'],
    'typing_key': ['Computer keyboard', 'Typing', 'Typewriter', 'Clicking'],
    'glitch': ['Static', 'Noise', 'Beep, bleep', 'White noise', 'Crackle'],
    'logo_sting': ['Chime', 'Bell', 'Ding', 'Jingle, tinkle', 'Gong', 'Cymbal', 'Glockenspiel'],
}
# Round 2 additions (2026-10-16): the PRO designed categories measured from the reference launch films
# (docs/motion-sfx-study.md). Additions only: the 18 rows above are unchanged. Phrasings written for the benchmark,
# independent of the recipes' TARGET / ALTS (checked at start); pro_riser / pro_zip phrasings from the sweep area's
# wrapper (out/forge_v2/pro_zip/bench_pro.py). Categories without a bench split still get phrase / AudioSet /
# artifact / diversity rows.
PRO_PHRASES = {
    'pro_click': ['a crisp bright digital click', 'a tiny sharp interface click'],
    'pro_tap': ['a soft muted tap on a screen', 'a short dull interface tap'],
    'pro_pop_in': ['a soft round thump as something appears on screen', 'a short muffled pop of an element appearing'],
    'pro_hover_tick': ['a tiny bright tick of a cursor', 'a faint high interface tick'],
    'pro_pop': ['a quick electronic blip', 'a tiny synthetic pop'],
    'pro_notify': ['a soft digital notification ding', 'a short pure alert tone'],
    'pro_shimmer': ['a glittering digital sparkle', 'a shimmering burst of tiny tones'],
    'pro_typing_key': ['a light clean keystroke', 'a soft tick of a keyboard key'],
    'pro_typing_run': ['someone typing a short line quickly', 'a rapid run of keystrokes'],
    'pro_glitch': ['a short burst of digital stutter', 'a crackling data glitch that cuts off'],
    'pro_whoosh_big': ['a deep heavy swoosh with a low rumble', 'a slow powerful whoosh of air'],
    'pro_riser': ['a rising stutter effect that speeds up', 'a pulsing electronic build-up'],
    'pro_zip': ['a fast rising electronic tone', 'a quick swooping synth glide'],
    'pro_impact': ['a deep designed boom that cuts off', 'a heavy low rumble hit in a launch video'],
    'pro_sub_drop': ['a deep sub bass boom that sinks', 'a low electronic bass drop'],
    'pro_logo_sting': ['a short electronic logo sound', 'a falling synth tone landing on a hum'],
}
PRO_CONFUSIONS = {
    'pro_click': CONFUSIONS['click'], 'pro_tap': CONFUSIONS['tap'],
    'pro_pop_in': ['a kick drum', 'a knock on a door', 'a ball bouncing', 'a hand clap'],
    'pro_hover_tick': CONFUSIONS['hover_tick'],
    'pro_pop': ['a microwave beep', 'a bird chirp', 'a tongue click', 'a kiss'],
    'pro_notify': CONFUSIONS['notify'], 'pro_shimmer': CONFUSIONS['shimmer'],
    'pro_typing_key': CONFUSIONS['typing_key'],
    'pro_typing_run': ['rain on a roof', 'a clock ticking', 'popcorn popping', 'a horse galloping'],
    'pro_glitch': CONFUSIONS['glitch'], 'pro_whoosh_big': CONFUSIONS['whoosh_big'],
    'pro_riser': ['a helicopter', 'a machine gun', 'a dial-up modem', 'a ticking clock'],
    'pro_zip': ['a whistle', 'a bird singing', 'a mosquito buzzing', 'a slide whistle'],
    'pro_impact': CONFUSIONS['impact'], 'pro_sub_drop': CONFUSIONS['sub_drop'],
    'pro_logo_sting': ['a doorbell', 'a car horn', 'an error buzzer', 'a church bell'],
}
PRO_AS = {
    'pro_click': AS_LABELS['click'], 'pro_tap': AS_LABELS['tap'], 'pro_pop_in': ['Thump, thud', 'Burst, pop', 'Plop'],
    'pro_hover_tick': AS_LABELS['hover_tick'], 'pro_pop': ['Burst, pop', 'Plop', 'Beep, bleep', 'Chirp tone'],
    'pro_notify': AS_LABELS['notify'] + ['Beep, bleep'], 'pro_shimmer': AS_LABELS['shimmer'],
    'pro_typing_key': AS_LABELS['typing_key'], 'pro_typing_run': AS_LABELS['typing_key'],
    'pro_glitch': AS_LABELS['glitch'], 'pro_whoosh_big': AS_LABELS['whoosh_big'],
    'pro_riser': ['Whoosh, swoosh, swish', 'Drum roll', 'Beep, bleep'],
    'pro_zip': ['Chirp tone', 'Sine wave', 'Beep, bleep', 'Whoosh, swoosh, swish'],
    'pro_impact': AS_LABELS['impact'] + ['Rumble'], 'pro_sub_drop': AS_LABELS['sub_drop'],
    'pro_logo_sting': AS_LABELS['logo_sting'] + ['Beep, bleep', 'Synthesizer'],
}
for _c in PRO_PHRASES:
    PHRASES.setdefault(_c, PRO_PHRASES[_c])
    CONFUSIONS.setdefault(_c, PRO_CONFUSIONS[_c])
    AS_LABELS.setdefault(_c, PRO_AS[_c])
METRICS = (('typ_z', 'high'), ('tone_db', 'low'), ('env_oct', 'low'), ('curve_db', 'low'), ('phrase_z', 'high'),
           ('margin', 'high'), ('as_p', 'high'))


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _frames_db(m, ms):
    n = max(1, int(SR * ms / 1000))
    k = max(1, len(m) // n)
    return 20 * np.log10(np.sqrt((np.pad(m, (0, max(0, k * n - len(m))))[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-12)


# ------------------------------------------------------------------------------------------- model-free checks
def env_feats(x):
    """onset (first 1 ms frame within 20 dB of the peak), attack (onset to peak), t40 (tone.t40_ms), active length
    (onset to the last 5 ms frame above peak - 40 dB), peak time; seconds / ms"""
    m = _mono(x)
    e1 = _frames_db(m, 1.0)
    p1 = int(np.argmax(e1))
    on = int(np.nonzero(e1 >= e1[p1] - 20)[0][0])
    e5 = _frames_db(m, 5.0)
    last = int(np.nonzero(e5 >= e5.max() - 40)[0][-1])
    return {'onset_s': round(on / 1000, 4), 'attack_ms': float(max(1, p1 - on)), 't40_ms': tone.t40_ms(m),
            'active_s': round(max(0.005, (last + 1) * 0.005 - on / 1000), 3), 'peak_s': round(p1 / 1000, 4)}


def _curve(x, pre, post):
    e = _frames_db(_mono(x), 10.0)
    p = int(np.argmax(e))
    e = np.maximum(e - e[p], -60.0)
    out = np.full(pre + post, -60.0)
    a, b = max(0, p - pre), min(len(e), p + post)
    out[pre - (p - a):pre + (b - p)] = e[a:b]
    return out


class EnvRef:
    """envelope shape of a ref set and the distance of any sound to it (see the module docstring: env_oct, curve_db)"""
    PRE = 10   # 100 ms before the peak

    def __init__(self, refs):
        xs = [_load(r) for r in refs]
        self.n = len(xs)
        self.F = [env_feats(x) for x in xs]
        tail = max([f['onset_s'] + f['active_s'] - f['peak_s'] for f in self.F], default=0.1)
        self.post = int(min(800, max(10, np.ceil(1.25 * tail / 0.01))))
        self.C = np.stack([_curve(x, self.PRE, self.post) for x in xs]) if xs else None

    def check(self, x, exclude=None):
        keep = [j for j in range(self.n) if j != exclude]
        if not keep:
            return {}
        f = env_feats(x)
        med = {k: float(np.median([self.F[j][k] for j in keep])) for k in ('attack_ms', 't40_ms', 'active_s')}
        floor = {'attack_ms': 1.0, 't40_ms': 5.0, 'active_s': 0.01}
        logs = {k: float(np.log2(max(f[k], floor[k]) / max(med[k], floor[k]))) for k in med}
        c = _curve(x, self.PRE, self.post)
        return {'attack_ms': f['attack_ms'], 't40_ms': f['t40_ms'], 'active_s': f['active_s'], 'onset_s': f['onset_s'],
                'env_oct': round(float(np.mean(np.abs(list(logs.values())))), 3),
                'curve_db': round(float(np.sqrt(np.mean((c - np.median(self.C[keep], 0)) ** 2))), 2),
                't40_ratio': round(2 ** logs['t40_ms'], 3), 'attack_ratio': round(2 ** logs['attack_ms'], 3),
                'active_ratio': round(2 ** logs['active_s'], 3), 'ref_median': {k: round(v, 3) for k, v in med.items()}}


ENDS_ON_PEAK = ('riser', 'pro_riser')   # grows out of silence and stops on its peak: no late-start / tail-cut flags


def artifacts(x, cat=None):
    """artifact flags and the numbers behind them: clipped (>= 3 samples at full scale), dc offset (mean > -30 dB re
    the peak), start click (first sample > 5 % of the peak), end click (last 1 ms RMS > 2 %), tail cut (last 10 ms
    within 30 dB of the loudest 5 ms), late start (lead_s: the first 1 ms frame within 50 dB of the peak comes after
    100 ms, so the sound cannot be placed on a frame). Calibrated so the real bench / forge trims rarely trip them."""
    xa = np.asarray(x, np.float64)
    m = _mono(xa)
    pk = float(np.abs(m).max()) + 1e-12
    e5 = _frames_db(m, 5.0)
    e1 = _frames_db(m, 1.0)
    out = {'clipped_samples': int((np.abs(xa) >= 0.999).sum()), 'dc_db': round(float(20 * np.log10(abs(m.mean()) / pk + 1e-12)), 1),
           'first_sample': round(abs(float(m[0])) / pk, 3), 'tail_db': round(float(_frames_db(m[-int(0.01 * SR):], 10.0)[0] - e5.max()), 1),
           'last_ms_rel': round(float(np.sqrt(np.mean(m[-48:] ** 2))) / pk, 3),
           'lead_s': round(int(np.nonzero(e1 >= e1.max() - 50)[0][0]) / 1000, 3),
           'peak_dbfs': round(float(20 * np.log10(float(np.abs(xa).max()) + 1e-12)), 2)}
    fl = []
    if out['clipped_samples'] >= 3:
        fl.append('clipped')
    if out['dc_db'] > -30:
        fl.append('dc offset')
    if out['first_sample'] > 0.05:
        fl.append('start click')
    if out['last_ms_rel'] > 0.02 and cat not in ENDS_ON_PEAK:
        fl.append('end click')
    if out['tail_db'] > -30 and cat not in ENDS_ON_PEAK:
        fl.append('tail cut')
    if out['lead_s'] > 0.1 and cat not in ENDS_ON_PEAK:
        fl.append('late start')
    out['flags'] = fl
    return out


def diversity(xs, E):
    """variety among one category's variants: CLAP cosine, tonal-balance distance, near duplicates (cos > 0.985 and
    tone < 1.5 dB apart)"""
    n = len(xs)
    if n < 2:
        return {'n': n}
    E = np.asarray(E, np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    P = [tone.profile(x) for x in xs]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cos = [float(E[i] @ E[j]) for i, j in pairs]
    td = [float(np.sqrt(np.mean((P[i] - P[j]) ** 2))) for i, j in pairs]
    return {'n': n, 'cos_mean': round(float(np.mean(cos)), 3), 'cos_max': round(max(cos), 3),
            'tone_min_db': round(min(td), 2), 'tone_mean_db': round(float(np.mean(td)), 2),
            'near_duplicates': [[i, j] for (i, j), c, t in zip(pairs, cos, td) if c > 0.985 and t < 1.5]}


def texture(x):
    """model-free texture: floor_db (10th percentile of the 5 ms frames re the loudest: a recording's room / mic floor,
    -120 = digital silence), stereo_corr (L / R correlation, 1 = mono), darken (log2 of the spectral centroid 20 ms at
    half the t40 tail over the centroid at the peak: real decays usually darken, < 0)"""
    xa = np.asarray(x, np.float64)
    m = _mono(xa)
    e5 = _frames_db(m, 5.0)
    out = {'floor_db': round(float(max(np.percentile(e5, 10) - e5.max(), -120.0)), 1)}
    if xa.ndim == 2 and xa.shape[1] == 2:
        a, b = xa[:, 0] - xa[:, 0].mean(), xa[:, 1] - xa[:, 1].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        out['stereo_corr'] = round(float((a * b).sum() / den), 3) if den > 0 else 1.0
    else:
        out['stereo_corr'] = 1.0
    t40 = tone.t40_ms(m)
    n = int(0.02 * SR)
    p = int(np.argmax(np.abs(m)))
    late = p + int(t40 / 2000 * SR)
    if t40 >= 40 and late + n <= len(m):
        f = np.fft.rfftfreq(n, 1 / SR)

        def cen(seg):
            F = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            return float((F * f).sum() / (F.sum() + 1e-12))
        out['darken'] = round(float(np.log2(max(cen(m[late:late + n]), 20.0) / max(cen(m[p:p + n]), 20.0))), 2)
    return out


def inside_real(made, real, metrics=METRICS, q=0.8):
    """per made row, how many metrics land inside the real rows' q-quantile limit (q = 0.8: the 80th percentile of a
    lower-is-better metric, the 20th of a higher-is-better one; q = 1: the worst real value)"""
    lim = {}
    for k, d in metrics:
        v = [r[k] for r in real if r.get(k) is not None]
        if v:
            lim[k] = (float(np.quantile(v, q if d == 'low' else 1 - q)), d)
    per = [sum(1 for k, (b, d) in lim.items() if r.get(k) is not None and (r[k] <= b if d == 'low' else r[k] >= b))
           for r in made]
    return {'per_sound': per, 'n_checks': len(lim), 'limits': {k: round(float(b), 3) for k, (b, _) in lim.items()}}


def _load(x):
    if isinstance(x, str):
        from earshot import audio
        return audio.load(x)
    return x


# ------------------------------------------------------------------------------------------- the benchmark
def recipe_phrases():
    """every TARGET / ALT string of the forge recipes (the benchmark's phrasings must not be among them)"""
    from earshot.forge import load_recipe, recipes
    out = set()
    for r in recipes():
        try:
            m = load_recipe(r)
        except Exception:      # a recipe being edited by someone else must not stop the benchmark
            continue
        out |= {s.strip().lower() for s in [getattr(m, 'TARGET', '')] + list(getattr(m, 'ALTS', [])) if s}
    return out


class Yardstick:
    """one ref set's CLAP embeddings, tonal balance and envelope shape"""

    def __init__(self, paths, xs, E):
        self.paths, self.E = paths, E
        self.tone = tone.ToneRef(xs) if xs else None
        self.env = EnvRef(xs) if xs else None

    def check(self, x, e, exclude=None):
        from earshot import compare
        keep = [j for j in range(len(self.paths)) if j != exclude]
        r = {}
        if len(keep) >= 3:
            _, z, _ = compare._typicality(e, self.E[keep], 3)
            r['typ_z'] = round(float(z), 2)
        if keep:
            r['nn_cos'] = round(float((self.E[keep] @ e).max()), 3)
            tm = self.tone.check(x, exclude=exclude)
            r.update(tone_db=tm['tone_db'], tone_ratio=tm['tone_ratio'], n_outside=len(tm['outside']),
                     outside=sorted(tm['outside'], key=lambda o: -abs(o[1]))[:3])
            r.update(self.env.check(x, exclude=exclude))
        return r


def bench_category(cat, made_paths, reg, models, real=True, log=print):
    from earshot import audio
    v, t = models
    e_ = reg['categories'].get(cat, {'forge': [], 'bench': [], 'note': ''})
    B, F = list(e_.get('bench', [])), list(e_.get('forge', []))
    allp = list(made_paths) + B + F
    if not allp:
        return {'n_bench': 0, 'n_forge': 0, 'note': e_.get('note', ''), 'made': [], 'summary': {}}
    X = [audio.load(p) for p in allp]
    from earshot import vocab
    E = vocab.embed_files(allp)
    phr, conf = PHRASES.get(cat, []), CONFUSIONS.get(cat, [])
    Z = v.calibrate(E, v.embed_text(phr + conf))['z_q'] if phr else None
    labs = [lab for lab in AS_LABELS.get(cat, []) if lab in t.index]
    nm, nb = len(made_paths), len(B)
    ys_b = Yardstick(B, X[nm:nm + nb], E[nm:nm + nb])
    ys_f = Yardstick(F, X[nm + nb:], E[nm + nb:])

    def row(i, ys, exclude=None):
        x, e = X[i], E[i]
        r = {'file': allp[i].replace(chr(92), '/'), 'length_s': round(len(x) / SR, 3)}
        r.update(ys.check(x, e, exclude) if ys.paths else {})
        if Z is not None:
            zp, zc = Z[i, :len(phr)], Z[i, len(phr):]
            best_c = float(zc.max()) if len(zc) else -9.0
            r.update(phrase_z=round(float(zp.mean()), 2), phrase_min=round(float(zp.min()), 2),
                     margin=round(float(np.mean(zp - best_c)), 2),
                     wins=round(float(np.mean((zp >= 2.5) & (zp > best_c))), 2),
                     best_confusion=conf[int(np.argmax(zc))] if len(zc) else None)
        p = t.probs_file(allp[i])
        top = t.top(p, 3)
        r.update(as_top=top, as_names=any(lab in labs for lab, _ in top),
                 as_p=round(max([t.prob_of(p, lab) for lab in labs], default=0.0), 3))
        r.update(texture(x))
        a = artifacts(x, cat)
        r['artifacts'] = a['flags']
        r['artifact_detail'] = {k: a[k] for k in ('clipped_samples', 'dc_db', 'first_sample', 'tail_db', 'lead_s',
                                                  'peak_dbfs')}
        return r

    out = {'n_bench': nb, 'n_forge': len(F), 'note': e_.get('note', ''), 'bench_refs': B,
           'made': [row(i, ys_b) for i in range(nm)]}
    out['diversity'] = diversity(X[:nm], E[:nm]) if nm else {'n': 0}
    if real:
        out['real_cross'] = [row(nm + nb + j, ys_b) for j in range(len(F))] if nb else []
        out['real_loo'] = [row(nm + j, ys_b, exclude=j) for j in range(nb)] if nb >= 2 else []
        out['real_rev'] = [row(nm + j, ys_f) for j in range(nb)] if F else []
        out['diversity_bench'] = diversity(X[nm:nm + nb], E[nm:nm + nb])
        out['diversity_forge'] = diversity(X[nm + nb:], E[nm + nb:])
        realrows = out['real_cross'] + out['real_loo']
        if nm and realrows:
            out['inside'] = inside_real(out['made'], realrows)
            out['inside']['real_per_sound'] = inside_real(realrows, realrows)['per_sound']
    out['summary'] = {k: _summ(out.get(k, [])) for k in ('made', 'real_cross', 'real_loo')}
    for k, d in (('made', 'diversity'), ('real_cross', 'diversity_forge'), ('real_loo', 'diversity_bench')):
        if out['summary'].get(k) and (out.get(d) or {}).get('n', 0) >= 2:
            out['summary'][k]['var_cos'] = out[d]['cos_mean']
    log(f"  {cat}: {nm} made, {nb} bench, {len(F)} forge refs" + (
        f"; inside the real 80% limits on {np.median(out['inside']['per_sound']):g} of {out['inside']['n_checks']} "
        f"checks (a real recording: {np.median(out['inside']['real_per_sound']):g})" if out.get('inside') else ''))
    return out


def _summ(rows):
    if not rows:
        return {}
    s = {'n': len(rows)}
    for k, _ in METRICS + (('tone_ratio', 'low'), ('t40_ratio', ''), ('wins', 'high'), ('floor_db', ''),
                           ('stereo_corr', ''), ('darken', '')):
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            s[k] = round(float(np.median(v)), 2)
    s['as_names'] = round(float(np.mean([bool(r.get('as_names')) for r in rows])), 2)
    s['artifacts'] = sum(len(r.get('artifacts', [])) for r in rows)
    return s


def gather(made=None, files=None):
    """{category: [paths]} from a folder of <category>/*.wav and / or cat=glob specs"""
    out = {}
    if made:
        for c in sorted(os.listdir(made)):
            d = os.path.join(made, c)
            if os.path.isdir(d):
                if os.path.isdir(os.path.join(d, 'final')):     # forge v2 layout: <category>/final/<category>_NN.wav
                    d = os.path.join(d, 'final')
                ps = [p for p in sorted(glob.glob(os.path.join(d, '*.wav'))) if not _skip(p)]
                if ps:
                    out[c] = [p.replace(chr(92), '/') for p in ps]
    for spec in files or []:
        c, g = spec.split('=', 1)
        out.setdefault(c, []).extend(sorted(p.replace(chr(92), '/') for p in glob.glob(g) if not _skip(p)))
    return out


def _skip(p):
    """forge's audition.wav (made and refs interleaved) and _-prefixed scratch files are never made sounds"""
    b = os.path.basename(p).lower()
    return b.startswith('audition') or b.startswith('_')


def table(res):
    """markdown: per category, the medians of made vs the real ceilings"""
    cols = [('typ_z', 'typ z'), ('tone_db', 'tone dB'), ('env_oct', 'env oct'), ('curve_db', 'curve dB'),
            ('t40_ratio', 't40 x'), ('phrase_z', 'phrase z'), ('margin', 'margin'), ('as_names', 'AS names'),
            ('artifacts', 'artifacts'), ('var_cos', 'var cos')]
    L = [f"# forge benchmark: {res['tag']}", '', f"made: {res.get('made_from') or '-'}; refs: {res['registry']}; "
         f"{res['created']}", '',
         'Medians per set. made = the made sounds vs the bench refs; real x = the forge refs (real, never in the bench) '
         'vs the bench refs; real loo = each bench ref vs the other bench refs. typ z, phrase z, margin, AS names: '
         'higher is better; tone dB, env oct (octaves off in attack / t40 / length), curve dB: lower is better; t40 x '
         '= tail length / bench median (1 = same). inside = checks (of n) a made sound lands inside the real '
         '80% limit, median over the made sounds (real = the same count for the real rows). var cos = mean pairwise CLAP '
         'cosine within the set (made variants / forge refs / bench refs): high = the variants are near clones.', '',
         '| category | set | n | ' + ' | '.join(c for _, c in cols) + ' | inside |',
         '|---|---|---|' + '---|' * (len(cols) + 1)]
    for cat, r in res['categories'].items():
        ins = r.get('inside')
        first = True
        for k, name in (('made', 'made'), ('real_cross', 'real x'), ('real_loo', 'real loo')):
            s = r.get('summary', {}).get(k) or {}
            if not s:
                continue
            cells = ['' if s.get(c) is None else (f'{s[c]:g}' if isinstance(s[c], (int, float)) else str(s[c]))
                     for c, _ in cols]
            inside = (f"{np.median(ins['per_sound']):g} / {ins['n_checks']} (real {np.median(ins['real_per_sound']):g})"
                      if k == 'made' and ins else '')
            L.append(f"| {cat if first else ''} | {name} | {s['n']} | " + ' | '.join(cells) + f' | {inside} |')
            first = False
        if first:
            L.append(f"| {cat} | no rows | 0 | " + ' | ' * len(cols) + ' |')
        if r.get('note'):
            L.append(f"| | note: {r['note']} | | " + ' | ' * len(cols) + ' |')
    L += ['', 'Texture (medians; model-free, not in any forge objective): floor dB = 10th percentile 5 ms frame re the '
          'loudest (-120 = digital silence), stereo corr (1 = mono), darken = log2 centroid at half the tail / at the '
          'peak (< 0: the decay darkens).', '', '| category | set | floor dB | stereo corr | darken |', '|---|---|---|---|---|']
    for cat, r in res['categories'].items():
        for k, name in (('made', 'made'), ('real_cross', 'real x'), ('real_loo', 'real loo')):
            s = r.get('summary', {}).get(k) or {}
            if s:
                L.append(f"| {cat if k == 'made' else ''} | {name} | {s.get('floor_db', '')} | {s.get('stereo_corr', '')} | "
                         f"{s.get('darken', '')} |")
    L += ['', 'Diversity among the made variants (CLAP cosine mean / max, tonal distance min dB, near duplicates):', '']
    for cat, r in res['categories'].items():
        d = r.get('diversity') or {}
        if d.get('n', 0) >= 2:
            db = r.get('diversity_bench') or {}
            L.append(f"- {cat}: cos {d['cos_mean']} / {d['cos_max']}, tone min {d['tone_min_db']} dB, "
                     f"{len(d['near_duplicates'])} near duplicate pair(s)"
                     + (f" (bench refs: cos {db['cos_mean']} / {db['cos_max']}, tone min {db['tone_min_db']} dB)"
                        if db.get('n', 0) >= 2 else ''))
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--made', help='folder with <category>/*.wav (or <category>/final/*.wav; audition*.wav skipped)')
    ap.add_argument('--files', nargs='*', help='category=glob (quote the glob)')
    ap.add_argument('--out', required=True, help='results JSON (a .md table is written next to it)')
    ap.add_argument('--tag', help='name in the report (default: from --out)')
    ap.add_argument('--cats', nargs='*', help='only these categories')
    ap.add_argument('--real-only', action='store_true', help='only the real ceilings of every category')
    ap.add_argument('--no-real', action='store_true', help='skip the real ceilings (faster)')
    ap.add_argument('--registry', help='refsets registry (default earshot.forge.refsets.registry_path())')
    ap.add_argument('--retable', action='store_true', help='only rewrite the .md table from an existing --out JSON')
    a = ap.parse_args(argv)
    if a.retable:
        with open(a.out, encoding='utf-8') as fh:
            res = json.load(fh)
        for r in res['categories'].values():
            r.setdefault('summary', {})
        md = os.path.splitext(a.out)[0] + '.md'
        with open(md, 'w', encoding='utf-8') as fh:
            fh.write(table(res))
        return print(f'wrote {md}')
    from earshot.forge import refsets
    reg = refsets.load(a.registry)
    clash = {c: [p for p in ps if p.strip().lower() in recipe_phrases()] for c, ps in PHRASES.items()}
    clash = {c: v for c, v in clash.items() if v}
    if clash:
        raise SystemExit(f'benchmark phrasings coincide with recipe TARGET / ALTS (no longer independent): {clash}')
    made = {} if a.real_only else gather(a.made, a.files)
    cats = a.cats or (sorted(reg['categories']) if a.real_only else sorted(made))
    unknown = [c for c in made if c not in PHRASES]
    if unknown:
        raise SystemExit(f'unknown categories {unknown}; known: {sorted(PHRASES)}')
    from earshot import vocab
    from earshot.models import ast
    models = (vocab.get(), ast.get())
    t0 = time.time()
    res = {'tag': a.tag or os.path.splitext(os.path.basename(a.out))[0].replace('results_', ''),
           'made_from': a.made or a.files, 'registry': a.registry or refsets.registry_path(),
           'created': time.strftime('%Y-%m-%d %H:%M'), 'phrases': {c: PHRASES[c] for c in cats},
           'confusions': {c: CONFUSIONS[c] for c in cats}, 'as_labels': {c: AS_LABELS[c] for c in cats},
           'categories': {}}
    for c in cats:
        res['categories'][c] = bench_category(c, made.get(c, []), reg, models, real=not a.no_real)
    res['seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    from scrub_paths import scrub   # no machine paths in a results file (refs relative to named roots)
    res = scrub(json.loads(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))))
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    md = os.path.splitext(a.out)[0] + '.md'
    with open(md, 'w', encoding='utf-8') as fh:
        fh.write(table(res))
    print(f'wrote {a.out} and {md} ({res["seconds"]} s)')


if __name__ == '__main__':
    main()
