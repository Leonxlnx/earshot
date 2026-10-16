"""Designed motion-design typing: the key tick of launch films (one key, `render`) and a whole typed run (`RUN`).

Not a keyboard recording: the study of the reference films (docs/motion-sfx-study.md, 'Typing') measured designed
typing as high-passed noise and nothing else. Each key is a noise burst at 3-12 kHz (spectral ridge 8.6-11 kHz),
essentially nothing under 2 kHz (2 kHz -28, 4 kHz -13, 8 kHz -7, 16 kHz -14 dB re total; bands up to 1 kHz at the
floor), tilt +12..+19 dB/oct, attack 2-5 ms, t20 11-18 ms, t40 about 60 ms, a convex end (holds, then stops), mono,
50-80 ms. The spectrograms of the six forge trims show the shape inside that: a sharp onset (two close peaks, touch then
hit), a drop of ~20 dB within 5-8 ms of the peak, then a long faint tail (-25 .. -60 dB over 50 ms) with a few small
rattle bumps, sometimes a faint release tick 40-80 ms after the hit.

How each part is built (every sample computed, no recorded audio):
  burst    white noise -> high-pass (order 6 at `lowcut`, so < 2 kHz sits 30+ dB down) -> a peaking boost at the
           ridge (`ridge`, `ridge_db`, `ridge_q`) -> a gentle low-pass (`hicut`); envelope = attack (`att` ms, power
           curve) then two decays: a fast one (`tau1`) that drops ~20 dB, a faint long tail (`tail_db`, `tau2`)
  touch    a smaller first peak `touch_dt` ms before the hit (`touch_db`): the finger on the cap, then the cap on the
           plate (the 'two peaks in ~10 ms' of a real keystroke, kept as a designed micro-structure)
  rattle   1-3 small bumps in the tail (`rattle_db`), positions fixed per `key`
  tink     an optional faint tonal layer: 2-3 high partials (5-12 kHz, fixed per key) that decay in a few ms (`tink_db`)
  release  an optional faint second tick `rel_dt` ms after the hit (`rel_db`), brighter
  edit     the file is `length` s, the last `gate` ms a smooth gate (convex end); starts on the first sample; mono
           (two identical channels, as the films' keys: correlation 1.0)

Character (what makes a family of audibly different but consistent keys): ridge (pitch of the tick), lowcut +
ridge_db (thin / full), att + touch (soft / snappy), tau1 + tail_db + length (tight / long), tink (noise / tonal),
rel_db (single / double). `key` fixes one key's layout; each render varies force, timing and noise only.
Round 3 (read only when present; DEFAULTS hold them at 0, which renders the r2 key exactly): `grain` / `grain_rate`
replace the smooth hiss by sparse impulses (velvet noise: a crackly key in the same band), `tink_glide` glides the tink
partials +-12 semitones over ~5 ms (a pitched digital blip), and `tink_db` reaches +2 dB (a clearly tonal glass key).
They exist for variety: the r2 family was near clones (CLAP cosine 0.926 vs the film keys' 0.849).

RUN: a typed run as one file, the films' way: keys at jittered spacing (62-148 ms in the reference run, median about
87 ms = 11.5 keys/s), level and timing jittered, never the same key layout twice in a row, an optional soft low swell
rising under the end of the run (the reference run has one; `swell_db`, -60 = off). forge(typing.RUN, refset=
'pro_typing_run', mode='designed'); its seed-bank context differs by target, so key and run entries share
seeds/typing.json without mixing.
"""
import types

import numpy as np

from .. import dsp

SR = dsp.SR
TARGET = 'a short crisp high-passed digital typing tick for a motion graphics title'
ALTS = ['a mouse click', 'a drop of water', 'a hi-hat', 'a spray can hiss']
REFSET = 'pro_typing_key'

SPACE = {
    'ridge': (6500, 13500, 'log'),     # spectral ridge (refs 8.6-11 kHz)
    'ridge_db': (0.0, 12.0),           # boost at the ridge
    'ridge_q': (0.5, 2.5, 'log'),
    'lowcut': (2500, 5000, 'log'),     # order-6 high-pass (refs: < 2 kHz at least 30 dB down)
    'hicut': (12000, 20000, 'log'),    # gentle low-pass (refs: 16 kHz -14 dB re total)
    'att': (0.0015, 0.005, 'log'),     # attack to the peak (refs 2-5 ms)
    'att_curve': (0.6, 3.0, 'log'),    # its shape: < 1 snaps up, > 1 swells
    'tau1': (0.0012, 0.006, 'log'),    # the fast drop after the peak (refs: ~20 dB in 5-8 ms)
    'tail_db': (-40.0, -12.0),         # the long faint tail's level re the peak
    'tau2': (0.008, 0.05, 'log'),      # its decay (refs: t40 about 60 ms; the tail holds near -35 dB to ~60 ms)
    'touch_db': (-30.0, -4.0),         # the first (touch) peak re the hit
    'touch_dt': (0.0015, 0.008, 'log'),
    'rattle_db': (-40.0, -18.0),       # small bumps in the tail
    'tink_db': (-40.0, 2.0),           # tonal partials re the burst (-40 = practically noise only; r3: up to +2)
    'rel_db': (-40.0, -14.0),          # release tick (-40 = none)
    'rel_dt': (0.035, 0.075, 'log'),
    'length': (0.05, 0.1),             # file length (refs 59-99 ms)
    'gate': (0.004, 0.02, 'log'),      # the final gate (convex end)
    'key': (0, 999, 'int'),
    # round 3 (only read when present: r2 parameter sets render exactly as before)
    'grain': (0.0, 1.0),               # share of sparse impulse (velvet) excitation: smooth hiss -> crackly sizzle
    'grain_rate': (300.0, 4000.0, 'log'),   # impulses per second of that excitation
    'tink_glide': (-12.0, 12.0),       # the tink partials glide this many semitones over ~5 ms (a digital blip)
}
DEFAULTS = {'ridge': 9600.0, 'ridge_db': 6.0, 'ridge_q': 1.0, 'lowcut': 3400.0, 'hicut': 16000.0, 'att': 0.003,
            'att_curve': 1.2, 'tau1': 0.0025, 'tail_db': -24.0, 'tau2': 0.016, 'touch_db': -12.0, 'touch_dt': 0.003,
            'rattle_db': -30.0, 'tink_db': -32.0, 'rel_db': -40.0, 'rel_dt': 0.05, 'length': 0.075, 'gate': 0.008,
            'key': 7, 'grain': 0.0, 'grain_rate': 1500.0, 'tink_glide': 0.0}   # r3 keys at 0: the r2 sound


def _t(n):
    return np.arange(n) / SR


def _db(v):
    return 10 ** (v / 20)


def _key_rng(p, part):
    """one key's fixed layout (tink partials, rattle positions): the same key sounds the same twice"""
    return dsp.rng(int(p['key']) * 7919 + part * 104729 + 31)


def _band(w, p, ridge=None, lowcut=None):
    """the designed band: steep high-pass, a ridge boost, a gentle top roll-off"""
    lc = p['lowcut'] if lowcut is None else lowcut
    y = dsp.butter(w, 'high', lc, order=6)
    y = dsp.peq(y, p['ridge'] if ridge is None else ridge, p['ridge_db'], p['ridge_q'])
    y = dsp.butter(y, 'low', p['hicut'], order=2)
    return dsp.butter(y, 'low', 18500, order=8)          # nothing above ~19 kHz (polish: energy > 20 kHz is a defect)


def _burst_env(n, att, curve, tau1, tail, tau2):
    """attack (power curve) to a peak of 1, then a fast drop to a faint long tail"""
    t = _t(n)
    a = max(2, dsp.n_of(att))
    e = np.empty(n)
    k = min(a, n)
    e[:k] = (np.arange(k) / a) ** curve
    td = t[k:] - t[k - 1] if n > k else t[:0]
    e[k:] = (1 - tail) * np.exp(-td / tau1) + tail * np.exp(-td / tau2)
    return e


def key_mono(p, r, force=1.0, ridge_mul=1.0):
    """one designed key as a mono array (peak about 1), length p['length'] (+ a release tick if it falls later)"""
    kr = _key_rng(p, 0)
    L = p['length'] * (1 + 0.06 * r.standard_normal())
    rel = p['rel_db'] > -39.5
    t_rel = p['touch_dt'] + p['rel_dt'] * (1 + 0.08 * r.standard_normal())
    n = dsp.n_of(max(L, (t_rel + 0.025) if rel else 0.0))
    ridge = p['ridge'] * ridge_mul * (1 + 0.015 * r.standard_normal())
    pp = dict(p, ridge=min(ridge, 16000.0))
    w = r.standard_normal(n)
    if p.get('grain', 0.0) > 0.0:      # r3: sparse impulses (velvet noise) instead of smooth hiss: a crackly key
        k = max(1, int(SR / p.get('grain_rate', 1500.0)))   # (tried: a smooth hit with a crackling tail; CLAP and
        v = np.zeros(n)                                       #  the ear then hear the plain key again)
        idx = np.arange(0, n, k) + r.integers(0, k, len(range(0, n, k)))
        idx = idx[idx < n]
        v[idx] = r.choice([-1.0, 1.0], len(idx)) * r.uniform(0.3, 1.0, len(idx))
        v /= np.sqrt(np.mean(v ** 2)) + 1e-12
        w = (1 - p['grain']) * w + p['grain'] * v
    # touch, then hit: two bursts of the same band (the hit starts touch_dt later)
    i_hit = dsp.n_of(p['touch_dt'] * (1 + 0.1 * r.standard_normal()))
    tail = _db(p['tail_db'] + 1.5 * r.standard_normal())
    e_hit = np.zeros(n)
    e_hit[i_hit:] = _burst_env(n - i_hit, p['att'] * 0.6, p['att_curve'], p['tau1'], tail, p['tau2'])
    e_touch = _db(p['touch_db']) * _burst_env(n, p['att'], p['att_curve'], p['tau1'] * 0.7, tail * 0.3,
                                               p['tau2'] * 0.5)
    env = np.maximum(e_hit, e_touch)
    # rattle: small bumps in the tail (positions fixed per key, timing breathes per press)
    if p['rattle_db'] > -39.5:
        for j in range(int(kr.integers(1, 4))):
            tj = i_hit / SR + kr.uniform(0.012, 0.045) * (1 + 0.1 * r.standard_normal())
            i = dsp.n_of(tj)
            if i < n - 8:
                m = n - i
                env[i:] += _db(p['rattle_db'] + kr.uniform(-4, 2)) * np.exp(-_t(m) / kr.uniform(0.001, 0.003)) \
                    * np.minimum(1, _t(m) / 0.0004)
    y = _band(w * env, pp)            # excite, then filter: the band stays clean under the envelope's fast edges
    # tonal tink: a few high partials of this key, struck by the hit
    if p['tink_db'] > -39.5:
        fs = np.sort(kr.uniform(5000, 12500, 3)) * ridge_mul
        rms = np.sqrt(np.mean(y[: dsp.n_of(0.01)] ** 2)) + 1e-12
        tk = np.zeros(n)
        m = n - i_hit
        gl = p.get('tink_glide', 0.0)
        for f, a, tau in zip(fs, (1.0, 0.6, 0.4), kr.uniform(0.002, 0.006, 3)):
            f = f * (1 + 0.003 * r.standard_normal())
            if gl:                     # r3: the partials glide (exponential, ~5 ms): a pitched digital blip
                fi = np.minimum(f * 2 ** (gl / 12 * (1 - np.exp(-_t(m) / 0.005))), 17000.0)
                ph = 2 * np.pi * np.cumsum(fi) / SR
            else:
                ph = 2 * np.pi * f * _t(m)
            tk[i_hit:] += a * np.exp(-_t(m) / tau) * np.sin(ph + r.uniform(0, 6.28))
        tk[i_hit:i_hit + 24] *= np.linspace(0, 1, min(24, m))
        y += _db(p['tink_db']) * tk * rms / (np.sqrt(np.mean(tk[i_hit:i_hit + dsp.n_of(0.01)] ** 2)) + 1e-12)
    if rel:
        i = dsp.n_of(t_rel)
        m = n - i
        er = _burst_env(m, 0.0008, 1.0, p['tau1'] * 0.6, tail * 0.5, p['tau2'] * 0.4)
        rr = _band(r.standard_normal(m) * er, pp, ridge=min(ridge * 1.15, 16000.0))
        y[i:] += _db(p['rel_db']) * rr / (np.abs(rr).max() + 1e-12) * np.abs(y).max()
    # the edit: convex end (a smooth gate over the last `gate` s of the key; a release tick keeps its own short tail)
    g = min(n - 2, dsp.n_of(p['gate']))
    end = n if rel else dsp.n_of(L)
    gw = np.ones(n)
    gw[end - g:end] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, g))
    gw[end:] = 0.0
    y = y * gw
    y = y[:end]
    return force * y / (np.abs(y).max() + 1e-12)


def render(p, r):
    p = {**DEFAULTS, **p}
    force = float(np.exp(0.08 * r.standard_normal()))
    y = key_mono(p, r, force)
    y = dsp.fade(y, 0.0, 0.0015)
    return np.stack([y, y], 1)


# ------------------------------------------------------------------------------------------------ the typed run
RUN_SPACE = {**SPACE,
             'rate': (8.5, 14.0),          # keys per second (reference run: about 11.5)
             'jit': (0.12, 0.35),          # timing jitter (sd of log interval)
             'lvl_jit': (1.0, 4.5),        # level jitter, dB sd
             'accent': (0.0, 0.35),        # share of accented keys (+3..+5 dB)
             'gap': (0.0, 0.2),            # share of word gaps (x1.4-1.7 the interval)
             'var': (0.01, 0.08),          # ridge spread between the keys of the run
             'dur': (1.1, 1.9),            # run length, s (reference run 1.55 s)
             'swell_db': (-60.0, -26.0),   # the soft low swell under the end (-60 = off)
             'swell_len': (0.2, 0.5)}
RUN_DEFAULTS = {**DEFAULTS, 'rate': 11.5, 'jit': 0.22, 'lvl_jit': 2.5, 'accent': 0.15, 'gap': 0.08, 'var': 0.04,
                'dur': 1.5, 'swell_db': -42.0, 'swell_len': 0.3}
RUN_TARGET = 'a quick run of crisp high-passed digital typing ticks for an animated title'


def _swell(n, r, length):
    """a soft low noise swell (100 Hz-1 kHz) that rises, accelerating, into the end of the run"""
    m = dsp.n_of(length)
    w = dsp.noise(m / SR, r, 'brown')
    w = dsp.butter(w, 'band', (90, 900), 2)
    u = np.linspace(0, 1, m)
    w *= u ** 2.5
    out = np.zeros(n)
    out[n - m:] = w[: m] / (np.abs(w).max() + 1e-12)
    return out


def render_run(p, r):
    p = {**RUN_DEFAULTS, **p}
    med = 1.0 / p['rate']
    times, t = [], 0.0
    while t < p['dur'] - 0.07:
        times.append(t)
        d = med * float(np.exp(p['jit'] * r.standard_normal()))
        if r.uniform() < p['gap']:
            d *= r.uniform(1.4, 1.7)
        t += float(np.clip(d, 0.62 * med, 1.75 * med))
    n = dsp.n_of(times[-1] + p['length'] + 0.06)
    y = np.zeros(n)
    layouts = [int(p['key']) + 101 * j for j in range(5)]
    last = None
    lvl = 0.0
    for i, t0 in enumerate(times):
        k = int(r.integers(0, len(layouts)))
        if k == last:
            k = (k + 1 + int(r.integers(0, len(layouts) - 1))) % len(layouts)
        last = k
        kp = dict(p, key=layouts[k])
        lvl = 0.6 * lvl + p['lvl_jit'] * r.standard_normal()
        g = lvl + (r.uniform(3, 5) if r.uniform() < p['accent'] else 0.0)
        mul = float(np.exp(p['var'] * (k - 2) / 2 + 0.3 * p['var'] * r.standard_normal()))
        dsp.place(y, key_mono(kp, r, 1.0, mul), t0, _db(min(g, 6.0)))
    if p['swell_db'] > -59:
        y += _db(p['swell_db']) * np.abs(y).max() * _swell(n, r, min(p['swell_len'], times[-1]))
    # a faint air bed (-68 dB re peak, the band of the keys) so the gaps between keys are not digital silence
    bed = dsp.butter(r.standard_normal(n), 'band', (p['lowcut'], 16000), 2)
    y += _db(-68.0) * np.abs(y).max() * bed / (np.std(bed) + 1e-12)
    y = dsp.fade(y, 0.002, 0.004)
    return np.stack([y, y], 1)


RUN = types.SimpleNamespace(
    __name__='earshot.forge.recipes.typing', __doc__='the typed run of earshot.forge.recipes.typing (one file)',
    TARGET=RUN_TARGET, ALTS=['rain on a window', 'a hi-hat pattern', 'a clock ticking', 'popcorn popping'],
    REFSET='pro_typing_run', SPACE=RUN_SPACE, DEFAULTS=RUN_DEFAULTS, render=render_run)


# ------------------------------------------------------------------------------------------------ study checks
def _env_db(m, ms=1.0):
    k = max(1, int(SR * ms / 1000))
    n = len(m) // k
    e = 10 * np.log10((m[:n * k].reshape(n, k) ** 2).mean(1) + 1e-14)
    return e - e.max()


def _share_below(m, f_hz):
    P = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return 10 * np.log10(P[f < f_hz].sum() / (P.sum() + 1e-14) + 1e-14)


def key_stats(x):
    """the study's designed-key numbers, model-free: attack (first ms within 20 dB of the peak -> peak), t20 / t40
    (peak -> first fall 20 / 40 dB under it, 10 ms smoothing as in the study), share under 2 kHz (dB re total)"""
    m = np.asarray(x, np.float64)
    m = m.mean(1) if m.ndim == 2 else m
    e1 = _env_db(m, 1.0)
    ip = int(np.argmax(e1))
    i20 = int(np.nonzero(e1[:ip + 1] >= -20)[0][0])
    k = int(SR * 0.001)
    sm = np.convolve(m ** 2, np.ones(10 * k) / (10 * k), 'same')
    ed = 10 * np.log10(sm[::k] + 1e-14)
    ed -= ed.max()
    jp = int(np.argmax(ed))

    def fall(db):
        j = np.nonzero(ed[jp:] <= -db)[0]
        return float(j[0]) if len(j) else None
    return {'attack_ms': float(ip - i20), 't20_ms': fall(20), 't40_ms': fall(40),
            'lo2k_db': round(float(_share_below(m, 2000)), 1), 'length_s': round(len(m) / SR, 3)}


def _outside(v, lo, hi, scale):
    if v is None:
        return 1.0
    return max(0.0, lo - v, v - hi) / scale


def study_extra(x, row=None):
    """Critic(extra=...) for one designed key: minus the distance to the study's target box (attack 2-5 ms, t20 11-18 ms,
    t40 45-70 ms, < 2 kHz at least 30 dB down, 50-95 ms), each term capped at 1, so at most -3 in all"""
    s = key_stats(x)
    pen = (min(1.0, _outside(s['attack_ms'], 1.5, 5.5, 3.0)) + min(1.0, _outside(s['t20_ms'], 10.0, 19.0, 6.0))
           + min(1.0, _outside(s['t40_ms'], 42.0, 72.0, 25.0)) + min(1.0, max(0.0, s['lo2k_db'] + 30) / 10)
           + min(1.0, _outside(s['length_s'], 0.05, 0.1, 0.03)))
    return -min(3.0, 0.75 * pen)


def run_stats(x):
    """onsets of a typed run (1 ms envelope crossing -32 dB re peak, 45 ms apart at least): spacing median / min / max"""
    m = np.asarray(x, np.float64)
    m = m.mean(1) if m.ndim == 2 else m
    e = _env_db(m, 1.0)
    on = [i for i in range(1, len(e)) if e[i] > -32 >= e[i - 1]]
    keep = []
    for i in on:
        if not keep or i - keep[-1] >= 45:
            keep.append(i)
    d = np.diff(keep) if len(keep) > 1 else np.array([0.0])
    return {'keys': len(keep), 'med_ms': float(np.median(d)), 'min_ms': float(d.min()), 'max_ms': float(d.max()),
            'lo2k_db': round(float(_share_below(m, 2000)), 1)}


def run_extra(x, row=None):
    """Critic(extra=...) for a run: spacing median 75-100 ms (reference 87), no key closer than 45 ms, < 2 kHz quiet"""
    s = run_stats(x)
    pen = (min(1.0, _outside(s['med_ms'], 72.0, 100.0, 20.0)) + min(1.0, max(0.0, s['lo2k_db'] + 28) / 10)
           + (0.5 if s['keys'] < 10 else 0.0))
    return -min(3.0, 0.75 * pen)
