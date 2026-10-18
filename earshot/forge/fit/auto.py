"""The fit engine: one entry point that turns a recording into a model (and back), whatever kind of sound it is.

    from earshot.forge.fit import auto
    m = auto.fit('click.wav')                   # kind='auto': measure the sound, try the fitters, keep the best
    m = auto.fit('pop.wav', kind='hybrid', layers=('sines', 'modal', 'spectral'))
    y = auto.render(m, seed=0)                  # the resynthesis; seeds change only the noise / jitter
    y = auto.render(m, seed=3, vary=0.4)        # knobs: knobs_of(m) (a fitted recipe's SPACE)
    mm = auto.morph(m_a, m_b, 0.5)              # a new model between two fits of the same kind
    auto.character('x.wav')                     # the measurements the candidates are chosen from
    print(auto.summary(m))

Kinds (each a fitter in this package, all analysis by synthesis, all plain JSON models):
  modal     struck / ringing sounds: strikes of damped sinusoids + band noise (fit/modal.py)
  sines     tonal sounds: tracked partials (glides kept) + strike atoms + 1/3-octave noise (fit/sines.py)
  spectral  noisy sounds: a time-varying 48-band noise gesture + stereo + texture + ridges (fit/spectral.py)
  hybrid    layers fitted one after another on what the previous layers leave: every layer but the last keeps only
            its deterministic part (sines partials + atoms, or modal modes + contact + sub), the last (spectral)
            models the rest. A pop = chirp (sines) + body (modal) + air (spectral); a key press = modal + spectral.
  morph     {'kind': 'morph', 'a': model, 'b': model, 'amount': w}: rendered as morph(a, b, w) (the recipe knob
            'morph' moves w), so a variation sits between two recordings instead of copying one.

kind='auto' (fit):
  1. character() measures attack, decay, crest, tonal share (energy in spectral peaks 12 dB over the local median),
     partial persistence, the glide of the strongest partial and flatness, and names a class (glide, ringing,
     struck, gesture, mixed); plan() orders the candidates by class. Under 1 s every candidate is fitted (each fit
     takes 0.5-4 s); longer sounds get the class's four.
  2. every candidate is fitted, then its post stage (fit_post) is measured from two renders: a smooth third-octave
     EQ onto the original's balance, the level, and the attack: the original's 0.25 ms envelope before the onset
     (renders may not stand above it: band noise and window smear leak energy before a strike) and over the first
     6-25 ms after a sharp onset (renders follow its contour within +-9 dB). render() always applies it.
  3. each candidate is rendered (seed 0) and scored against the original (quality.score: mel dB, tone, envelope,
     onset, excess / missing energy, + CLAP with clap=True); the lowest score wins. Hybrids reuse the single-kind
     fits as their first layer. The model gets a 'fit_engine' entry: the character, every candidate's metrics, the
     choice. seeds=TAKE_SEEDS (`earshot fit --takes`) scores the mean over seeds 0, 1, 2 instead: a candidate with
     noise in it can win on one lucky seed-0 draw and then make weaker takes (on 10 of the 12 validation fits the
     chosen candidate's seed-0 score sat 0.1-2.4 under its mean over seeds 0-7).
Pure numpy / scipy (CLAP only with clap=True). Deterministic: render(model, seed, knobs) is a pure function.
"""
import copy
import json
import time

import numpy as np

from . import quality as Q

SR = 48000
TAKE_DC_HZ = 8.0          # a take (render(take=True)) is high-passed here: no DC offset, no infrasound
TAKE_SEEDS = (0, 1, 2)    # fit(seeds=TAKE_SEEDS): score candidates on these seeds (a fit meant to make takes)
KINDS = ('modal', 'spectral', 'sines', 'hybrid')
ROLE = {'sines': 'tone', 'modal': 'body', 'spectral': 'air'}
HYBRIDS = {'modal+spectral': ('modal', 'spectral'), 'sines+spectral': ('sines', 'spectral'),
           'sines+modal+spectral': ('sines', 'modal', 'spectral')}
VERSION = 1

# knobs per kind: name -> (search range spec for a forge SPACE, identity value, what it does)
KNOBS = {
    'modal': {
        'size': ((0.8, 1.25, 'log'), 1.0, 'object size: mode frequencies / size, decays x size'),
        'damping': ((0.7, 1.43, 'log'), 1.0, 'decays / damping'),
        'brightness': ((-3.0, 3.0), 0.0, 'dB / octave about 1 kHz on modes and noise'),
        'hardness': ((0.7, 1.43, 'log'), 1.0, 'contact: shorter and brighter attack above 1'),
        'noise': ((-6.0, 4.0), 0.0, 'dB on the noise bands'),
        'width': ((0.5, 1.5), 1.0, 'x the side signal'),
        'room': ((0.0, 0.25), 0.0, 'wet of a small stochastic room'),
        'vary': ((0.0, 0.6), 0.0, 'seeded jitter of modes, timing and noise: new takes of the same object'),
        'spacing': ((0.8, 1.25, 'log'), 1.0, 'x the time between strikes'),
    },
    'sines': {
        'transpose': ((-3.0, 3.0), 0.0, 'semitones (partials and the residual spectrum together)'),
        'stretch': ((0.85, 1.18, 'log'), 1.0, 'time factor about the first strike'),
        'decay': ((0.7, 1.43, 'log'), 1.0, 'factor on every decay after its peak'),
        'brightness': ((-3.0, 3.0), 0.0, 'dB / octave about the partial centroid'),
        'vibrato': ((0.0, 20.0), 0.0, 'cents of vibrato'),
        'drift': ((0.0, 10.0), 0.0, 'cents of slow random wander'),
        'residual': ((0.5, 2.0, 'log'), 1.0, 'gain on the noise'),
        'transient': ((0.5, 2.0, 'log'), 1.0, 'gain on the strike noise (first 15 ms)'),
        'vary': ((0.0, 0.6), 0.0, 'seeded per-partial detune / level / decay jitter'),
        'inharm': ((-0.03, 0.03), 0.0, 'stretch of the partial ratios'),
    },
    'spectral': {
        'length': ((0.8, 1.25, 'log'), 1.0, 'time-stretch of the gesture'),
        'speed': ((0.8, 1.25, 'log'), 1.0, 'movement speed: shorter and higher above 1'),
        'sweep': ((0.7, 1.43, 'log'), 1.0, 'x the spectral centroid excursion'),
        'shift': ((-0.4, 0.4), 0.0, 'octaves'),
        'tilt': ((-2.5, 2.5), 0.0, 'dB / octave about 1 kHz'),
        'roughness': ((0.5, 1.5), 1.0, 'texture depth (flutter, crackle)'),
        'width': ((0.5, 1.5), 1.0, 'side / pan amount'),
        'ridges': ((0.5, 1.5), 1.0, 'tonal ridge gain'),
    },
    'hybrid': {
        'pitch': ((-3.0, 3.0), 0.0, 'semitones on every layer (modal size, sines transpose, spectral shift)'),
        'decay': ((0.7, 1.43, 'log'), 1.0, 'decay factor of the tonal / body layers'),
        'bright': ((-3.0, 3.0), 0.0, 'dB / octave on every layer'),
        'vary': ((0.0, 0.6), 0.0, 'seeded jitter of the tonal / body layers'),
        'width': ((0.5, 1.5), 1.0, 'stereo width of the air layer'),
        'tone_db': ((-6.0, 6.0), 0.0, 'level of the tonal (sines) layer'),
        'body_db': ((-6.0, 6.0), 0.0, 'level of the body (modal) layer'),
        'air_db': ((-6.0, 6.0), 0.0, 'level of the air (spectral) layer'),
    },
}


def _fitters():
    from . import modal, sines, spectral
    return {'modal': modal, 'sines': sines, 'spectral': spectral}


def kind_of(model):
    """'modal' | 'sines' | 'spectral' | 'hybrid' | 'morph'"""
    k = model.get('kind') if isinstance(model, dict) else None
    if k == 'spectral_noise':
        return 'spectral'
    if k not in ('modal', 'sines', 'hybrid', 'morph'):
        raise ValueError(f'not a fit model (kind {k!r})')
    return k


def knobs_of(model):
    """the knob table (KNOBS entry) of a model; a morph adds 'morph' (0..1) to its kind's"""
    k = kind_of(model)
    if k == 'morph':
        kn = dict(knobs_of(model['a']))
        kn['morph'] = ((0.2, 0.8), float(model.get('amount', 0.5)), 'where between the two recordings (0 = a, 1 = b)')
        return kn
    if k == 'hybrid':
        roles = {L['role'] for L in model['layers']}
        return {n: v for n, v in KNOBS['hybrid'].items()
                if not n.endswith('_db') or n[:-3] in roles}
    return KNOBS[k]


def channels_of(x_or_model):
    if isinstance(x_or_model, dict):
        m = x_or_model
        k = kind_of(m)
        if k == 'morph':
            return max(channels_of(m['a']), channels_of(m['b']))
        if k == 'spectral':
            return 2 if m.get('stereo') else int(m.get('channels', 1))
        if k == 'sines':
            return int(m.get('channels', 2 if m.get('width', -120.0) > -50.0 else 1))
        return int(m.get('channels', 1))
    x = np.asarray(x_or_model)
    return 1 if x.ndim == 1 or x.shape[1] == 1 or np.allclose(x[:, 0], x[:, 1]) else 2


# ================================================================================================ character
def character(x):
    """measurements that decide which fitter suits a sound (mono sum): duration, attack / t40 (ms), crest (dB),
    tonal share (0..1, energy in peaks 12 dB over the local median), persistence (0..1, peaks that continue into
    the next frame), glide (octaves the strongest partial moves), flatness (0..1, geometric / arithmetic mean
    power), strikes (onsets), and the class flags impulsive / tonal / noisy / gliding"""
    from scipy.ndimage import maximum_filter1d, median_filter
    X = Q.load(x)
    m = Q.mono(X)
    n = len(m)
    e = Q.env_db(m, 1.0)
    pk = int(np.argmax(e))
    top = float(e[pk])
    on = int(np.argmax(e > top - 30))
    below = np.nonzero(e[pk:] < top - 40)[0]
    t40 = float(below[0] if len(below) else len(e) - pk)
    act = e > top - 40
    eh = Q.env_db(m, 0.5)                           # the steepest rise anywhere loud (dB per 2 ms): a strike's onset
    loud = eh > eh.max() - 40
    rise = float(np.max(np.where(loud[4:], eh[4:] - eh[:-4], 0.0))) if len(eh) > 4 else 0.0
    crest = float(20 * np.log10(np.abs(m).max() + 1e-12) - 10 * np.log10(np.mean(10 ** (e[act] / 10)) + 1e-20))
    nfft, hop = 2048, 256
    T = max(1, (n + hop - 1) // hop)
    xp = np.zeros(T * hop + nfft)
    xp[nfft // 2:nfft // 2 + n] = m
    fr = np.lib.stride_tricks.sliding_window_view(xp, nfft)[::hop][:T]
    P = np.abs(np.fft.rfft(fr * np.hanning(nfft + 2)[1:-1], axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / SR)
    band = (f >= 60) & (f <= 16000)
    P = P[:, band]
    fb = f[band]
    fe = P.sum(1)
    af = fe > fe.max() * 10 ** -3.5
    L = 10 * np.log10(P + 1e-20)
    med = median_filter(L, size=(1, 21), mode='nearest')
    lmax = maximum_filter1d(L, 3, axis=1)
    peaks = (L >= lmax) & (L - med > 12) & (L > L.max(1, keepdims=True) - 50)
    mask = maximum_filter1d(peaks.astype(np.uint8), 5, axis=1).astype(bool)
    share = (P * mask).sum(1) / (fe + 1e-30)
    w = fe * af
    tonal = float((share * w).sum() / (w.sum() + 1e-30))
    flat = np.exp(np.mean(np.log(P + 1e-30), 1)) / (P.mean(1) + 1e-30)
    flatness = float(np.exp((np.log(flat + 1e-12) * w).sum() / (w.sum() + 1e-30)))
    num = den = 0.0
    idx = np.nonzero(af)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1 or not peaks[a].any():
            continue
        pw = P[a][peaks[a]]
        num += float((pw * mask[b][peaks[a]]).sum())
        den += float(pw.sum())
    persist = num / den if den > 0 else 0.0
    # glide: the strongest partial's frequency over the frames where it stands out
    tr = []
    for t in idx:
        if not peaks[t].any():
            continue
        j = int(np.argmax(np.where(peaks[t], L[t], -1e9)))
        if L[t, j] - med[t, j] < 15:
            continue
        a, b, c = L[t, max(j - 1, 0)], L[t, j], L[t, min(j + 1, len(fb) - 1)]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
        tr.append((t * hop / SR, np.log2(max(fb[j] + d * (fb[1] - fb[0]), 20.0)), b))
    glide = 0.0
    track_ms = 0.0
    if len(tr) >= 4:
        tt, lf, lv = map(np.asarray, zip(*tr))
        strong = lv > lv.max() - 15
        tt, lf = tt[strong], lf[strong]
        if len(tt) >= 4:
            jumps = np.abs(np.diff(lf))
            run = np.concatenate([[True], jumps < 1 / 6])          # a continuous track: < 2 semitones per frame
            k0 = int(np.argmax(run))
            seg = slice(k0, k0 + max(1, int(np.argmin(run[k0:])) if not run[k0:].all() else len(run) - k0))
            st, sl = tt[seg], lf[seg]
            track_ms = float((st[-1] - st[0]) * 1000) if len(st) > 1 else 0.0
            if len(st) >= 4:
                q = max(1, len(st) // 5)
                glide = float(np.median(sl[-q:]) - np.median(sl[:q]))
    from .modal import onsets
    strikes = len(onsets(m))
    c = {'dur_s': round(n / SR, 4), 'attack_ms': float(pk - on), 'rise_db': round(rise, 1), 't40_ms': t40,
         'crest_db': round(crest, 2),
         'tonal_share': round(tonal, 3), 'persistence': round(persist, 3), 'glide_oct': round(glide, 3),
         'track_ms': round(track_ms, 1), 'flatness': round(flatness, 4), 'strikes': strikes,
         'channels': channels_of(X)}
    c['impulsive'] = c['attack_ms'] <= 40.0 and (c['t40_ms'] <= 400.0 or c['crest_db'] >= 18.0)
    c['tonal'] = c['tonal_share'] >= 0.45 and c['persistence'] >= 0.5
    c['noisy'] = c['tonal_share'] < 0.25
    c['gliding'] = abs(c['glide_oct']) >= 0.3 and c['track_ms'] >= 25 and c['tonal_share'] >= 0.25
    if c['gliding']:
        c['class'] = 'glide'
    elif c['tonal_share'] >= 0.8 and c['persistence'] >= 0.9 and c['t40_ms'] > 300:
        c['class'] = 'ringing'
    elif c['impulsive']:
        c['class'] = 'struck'
    elif c['attack_ms'] > 40.0 and c['crest_db'] < 18.0:
        c['class'] = 'gesture'
    else:
        c['class'] = 'mixed'
    return c


PLANS = {   # class -> candidates, expected best first
    'glide': ['sines', 'hybrid:sines+modal+spectral', 'hybrid:sines+spectral', 'modal'],
    'ringing': ['sines', 'modal', 'hybrid:sines+spectral', 'spectral'],
    'struck': ['modal', 'hybrid:sines+modal+spectral', 'hybrid:modal+spectral', 'sines'],
    'gesture': ['spectral', 'hybrid:sines+spectral', 'modal', 'sines'],
    'mixed': ['modal', 'spectral', 'hybrid:modal+spectral', 'sines'],
}
ALL = ('modal', 'sines', 'spectral', 'hybrid:modal+spectral', 'hybrid:sines+spectral', 'hybrid:sines+modal+spectral')


def plan(c, max_candidates=None):
    """ordered candidate kinds for a character (first = the expected best); hybrids as 'hybrid:<a>+<b>[+<c>]'.
    The measured quality decides among them; the class orders them and prunes the slow fits of long sounds: under
    1 s every candidate is tried (each fit takes 0.5-4 s), longer sounds the class's four (a long struck sound:
    modal, sines, modal + spectral, spectral; the three-layer hybrid alone took 147 s on a 4.5 s impact)."""
    order = list(PLANS[c['class']])
    if c['dur_s'] >= 1.0 and c['class'] == 'struck':
        order = ['modal', 'sines', 'hybrid:modal+spectral', 'spectral']
    order += [k for k in ALL if k not in order]
    if max_candidates is None:
        max_candidates = len(ALL) if c['dur_s'] < 1.0 else 4
    return order[:max_candidates]


# ================================================================================================ fitting
def fit_kind(x, kind, fast=False, name=None, **opts):
    """one fitter on x ((n,) / (n, 2) 48 kHz array or path); fast=True uses cheaper settings (tests, previews)"""
    F = _fitters()
    X = Q.load(x)
    if kind == 'modal':
        kw = dict(passes=1) if fast else {}
        kw.update(opts)
        m = F['modal'].fit(X, **kw)
    elif kind == 'sines':
        kw = dict(windows=(1024,)) if fast else {}
        kw.update(opts)
        m = F['sines'].fit(X, **kw)
        m['channels'] = channels_of(X)
    elif kind == 'spectral':
        kw = dict(texture=False, iters=1, seeds=(1,)) if fast else {}
        kw.update(opts)
        m = F['spectral'].fit(X, name=name, **kw)
        m['channels'] = channels_of(X)
    else:
        raise ValueError(f'unknown fitter {kind!r} (modal, sines, spectral; hybrids via fit_hybrid)')
    if name:
        m['source'] = name
    return m


def deterministic(model):
    """a copy of a modal / sines model that renders only its deterministic part (no noise): a hybrid layer"""
    k = kind_of(model)
    m = copy.deepcopy(model)
    if k == 'modal':
        L = float(m['length_s'])
        m['noise'] = dict(m['noise'], env=[[[0.0, -240.0], [L, -240.0]] for _ in m['noise']['env']])
        m['side'] = None
        m['channels'] = 1
    elif k == 'sines':
        m['residual'] = {'bands': [1000.0], 'times': [0.0, float(m['duration'])], 'db': [[-150.0, -150.0]]}
        m['width'] = -120.0
        m['channels'] = 1
    else:
        raise ValueError(f'a {k} model has no deterministic part')
    m.pop('post', None)
    m.pop('fit_engine', None)
    m['layer'] = 'deterministic'
    return m


def _fit_n(X, y):
    """y (mono or stereo) as (len(X), channels of X)"""
    y = np.asarray(y, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] != X.shape[1]:
        y = np.repeat(y.mean(1, keepdims=True), X.shape[1], 1)
    out = np.zeros_like(X)
    k = min(len(X), len(y))
    out[:k] = y[:k]
    return out


def fit_hybrid(x, layers=('modal', 'spectral'), fast=False, base=None, name=None):
    """layers fitted in turn on what the previous ones leave (module doc). base: {kind: model} single-kind fits of
    the same x to reuse for the first layer"""
    X = Q.load(x)
    if X.ndim == 1:
        X = X[:, None]
    layers = tuple(layers)
    if len(layers) < 2 or layers[-1] == 'sines' or any(k not in ROLE for k in layers):
        raise ValueError(f'bad hybrid layers {layers}: 2-3 of modal / sines / spectral, deterministic ones first')
    resid = X.copy()
    out = []
    for i, k in enumerate(layers):
        last = i == len(layers) - 1
        if i == 0 and base and k in base:
            m = base[k]
        else:
            m = fit_kind(resid, k, fast=fast)
        if not last:
            m = deterministic(m)
            resid = resid - _fit_n(X, render(m, seed=0))
        out.append({'role': ROLE[k], 'gain_db': 0.0, 'model': m})
    return {'kind': 'hybrid', 'version': VERSION, 'sr': SR, 'length_s': round(len(X) / SR, 6),
            'channels': channels_of(X), 'layers': out, 'source': name}


def _cand_fit(X, cand, fast, base, name):
    if cand.startswith('hybrid'):
        layers = HYBRIDS[cand.split(':', 1)[1]] if ':' in cand else ('modal', 'spectral')
        return fit_hybrid(X, layers, fast=fast, base=base, name=name)
    m = fit_kind(X, cand, fast=fast, name=name)
    base[cand] = m
    return m


PRE_HOP = 0.00025         # s: the attack envelope grid of the post stage
ATTACK_S = 0.006          # s: the attack window after the onset whose contour the post stage follows, at least,
ATTACK_MAX_S = 0.025      # or up to 4 ms past the loudest frame within this much after the onset (a later main strike)


def _env_frames(m, hop):
    k = max(1, int(round(hop * SR)))
    nb = len(m) // k
    return 10 * np.log10((m[:nb * k].reshape(nb, k) ** 2).mean(1) + 1e-20) if nb else np.zeros(0)


def fit_post(X, renders, iters=2, max_db=12.0, envelope=False):
    """the post stage of a model, measured by analysis by synthesis (renders = a few seeds of the model):
      eq_db    a smooth third-octave EQ (minimum phase, forge.fx.eq_bands; 28 gains at tone.CEN) that moves the
               renders' mean balance onto the original's
      gain_db  the gain that matches the loudest 50 ms
      pre      the original's 0.25 ms envelope (dB) before its first onset (the first frame within 20 dB of the
               loudest) and, for a sharp onset (>= 15 dB/ms, quality.attack), over the ATTACK_S after it (or to
               4 ms past the loudest frame of the next ATTACK_MAX_S: a key's bottom-out after its first touch):
               {'t_on' s, 'db' [...], 'attack_db' [...]}. Band noise and window smear leak energy before a strike
               and soften it (a pre-echoed, blunt attack); _pre_clamp keeps the attack as clean as the recording's
      env      (envelope=True only) after the attack: the renders' mean 5 ms envelope error as a gain curve ({'hop',
               'db'}, -12..+6 dB, smoothed): a tail that rings too long, a fade-out the model missed. Off by default:
               on the 10 short validation refs it lowered mel dB 3.69 -> 3.62 and raised envelope r 0.977 -> 0.986,
               but CLAP fell 0.925 -> 0.918 (a key press 0.955 -> 0.905)"""
    from ..fx import eq_bands
    from ..tone import profile
    xm = Q.mono(X)
    target = profile(xm)
    ys = [Q.mono(y) for y in renders]
    tot = np.zeros(len(target))
    for _ in range(iters):
        cur = 10 * np.log10(np.mean([10 ** (profile(eq_bands(y, tot) if tot.any() else y) / 10) for y in ys], 0))
        d = target - cur
        w = 10 ** (cur / 10)
        d -= np.sum(w * d) / np.sum(w)                     # level-neutral: centred on the loud bands
        d = np.where(cur < -50, np.minimum(d, 6.0), d)     # EQ cannot create what is not there
        d = np.convolve(np.pad(d, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
        tot = np.clip(tot + d, -max_db, max_db)
    g = Q.peak_rms_db(xm) - float(np.mean([Q.peak_rms_db(eq_bands(y, tot)) for y in ys]))   # mono vs mono
    post = {'eq_db': [round(float(v), 2) for v in tot], 'gain_db': round(float(g), 2)}
    e = _env_frames(xm, PRE_HOP)
    if len(e) > 4:
        on = int(np.argmax(e > e.max() - 20))
        pk = on + int(np.argmax(e[on:on + int(round(ATTACK_MAX_S / PRE_HOP))]))   # the main strike, if it is later
        n_att = max(int(round(ATTACK_S / PRE_HOP)), pk - on + int(round(0.004 / PRE_HOP)))
        struck = Q.attack(xm)[1] >= 15.0              # a sharp onset (dB/ms): its contour is worth keeping
        if on >= 2 or (struck and len(e) > on + 4):
            post['pre'] = {'t_on': round(on * PRE_HOP, 6), 'db': [round(float(v), 1) for v in e[:on]],
                           'attack_db': [round(float(v), 1) for v in e[on:on + n_att]] if struck else []}
            # the clamp pins the attack to the original's contour: re-level what it leaves
            g2 = Q.peak_rms_db(xm) - float(np.mean([Q.peak_rms_db(apply_post(y, post)) for y in ys]))
            post['gain_db'] = round(float(g + g2), 2)
    if envelope:
        post['env'] = _fit_env(xm, [apply_post(y, post) for y in ys], post)
        # the gain curve moves the balance (tail and attack differ in colour): one more EQ + level pass on top
        cur = 10 * np.log10(np.mean([10 ** (profile(apply_post(y, post)) / 10) for y in ys], 0))
        d = target - cur
        w = 10 ** (cur / 10)
        d -= np.sum(w * d) / np.sum(w)
        d = np.where(cur < -50, np.minimum(d, 6.0), d)
        d = np.convolve(np.pad(d, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
        post['eq_db'] = [round(float(v), 2) for v in np.clip(np.asarray(post['eq_db']) + d, -max_db, max_db)]
        g3 = Q.peak_rms_db(xm) - float(np.mean([Q.peak_rms_db(apply_post(y, post)) for y in ys]))
        post['gain_db'] = round(float(post['gain_db'] + g3), 2)
    return post


ENV_HOP = 0.005           # s: the envelope calibration grid


def _fit_env(xm, ys, post):
    """the seed-averaged envelope error left after the attack (5 ms frames, dB, smoothed, -12..+6): a tail that
    rings too long or too loud, a missing fade-out. Applied as a gain curve (time-scaled with the decay knobs)"""
    eo = Q.env_db(xm, ENV_HOP * 1000)
    er = 10 * np.log10(np.mean([10 ** (Q.env_db(Q.mono(y), ENV_HOP * 1000)[:len(eo)] / 10) for y in ys
                                if len(y) >= len(xm)], 0) + 1e-20)
    n = min(len(eo), len(er))
    d = np.clip(eo[:n] - er[:n], -12.0, 6.0)
    top = max(eo.max(), er.max())
    d = np.where((eo[:n] < top - 70) & (er[:n] < top - 70), 0.0, d)        # both inaudible: leave it
    pre = post.get('pre') or {}
    t_start = float(pre.get('t_on', 0.0)) + PRE_HOP * len(pre.get('attack_db') or []) + 0.002
    i0 = int(np.ceil(t_start / ENV_HOP))
    d[:i0] = 0.0
    for _ in range(2):
        d = np.convolve(np.pad(d, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
    d[:i0] = 0.0
    d[i0:i0 + 2] *= [0.33, 0.67][:len(d[i0:i0 + 2])]                       # fade the correction in
    return {'hop': ENV_HOP, 'db': [round(float(v), 2) for v in d]}


def _env_gain(n, env, scale=1.0):
    d = np.asarray(env['db'], float)
    hop = float(env['hop']) * scale
    tc = (np.arange(len(d)) + 0.5) * hop * SR
    return 10 ** (np.interp(np.arange(n), tc, d, left=0.0, right=float(d[-1]) if len(d) else 0.0) / 20)


def _pre_clamp(y, pre, scale=1.0):
    """the attack through the original's envelope (0.25 ms frames): before the onset y may not stand above the
    original (+1.5 dB; only cut), over the ATTACK_S after it y follows the original's contour (+-9 dB, the correction
    fading out over the last 2 ms), so a smeared or pre-echoed attack comes back as sharp as the recording's.
    scale: time factor of the render (spectral length / speed)"""
    k = int(round(PRE_HOP * SR))
    ref_pre = np.asarray(pre.get('db') or [], float)
    ref_att = np.asarray(pre.get('attack_db') or [], float)
    n_pre = int(round(float(pre['t_on']) * scale / PRE_HOP))
    n_att = len(ref_att) if scale == 1.0 else int(round(len(ref_att) * scale))
    nb = min(len(y) // k, n_pre + n_att)
    if nb < 2:
        return y
    m = y.mean(1) if y.ndim == 2 else y
    e = _env_frames(m[:nb * k], PRE_HOP)
    gdb = np.zeros(nb)
    if n_pre > 0 and len(ref_pre):
        want = np.interp(np.arange(min(n_pre, nb)) / max(n_pre - 1, 1), np.linspace(0, 1, len(ref_pre)), ref_pre)
        gdb[:len(want)] = np.minimum(0.0, want + 1.5 - e[:len(want)])
    if n_att > 0 and len(ref_att) and nb > n_pre:
        j = np.arange(n_pre, nb)
        want = np.interp((j - n_pre) / max(n_att - 1, 1), np.linspace(0, 1, len(ref_att)), ref_att)
        fade = np.clip((n_pre + n_att - 1 - j) / max(1, int(round(0.002 / PRE_HOP))), 0.0, 1.0)
        gdb[n_pre:] = np.clip(want - e[n_pre:nb], -9.0, 9.0) * fade
    neg = np.minimum(gdb, 0.0)                                       # widen each cut by a frame (smooth edges)
    neg = np.minimum(neg, np.minimum(np.r_[neg[1:], 0.0], np.r_[0.0, neg[:-1]]))
    gdb = np.where(gdb < 0, neg, gdb)
    gdb = np.convolve(np.pad(gdb, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
    n_end = nb * k
    tt = (np.arange(nb) + 0.5) * k
    gs = 10 ** (np.interp(np.arange(n_end), np.r_[tt, n_end], np.r_[gdb, 0.0]) / 20)
    y = np.array(y, copy=True)
    if y.ndim == 2:
        y[:n_end] *= gs[:, None]
    else:
        y[:n_end] *= gs
    return y


def apply_post(y, post, scale=1.0, tail=1.0):
    """a render through a model's post stage (EQ, gain, attack clamp, envelope calibration). scale: the render's
    time factor (spectral length / speed); tail: the decay knobs' factor on the envelope after the attack"""
    if not post:
        return y
    from ..fx import eq_bands
    if any(post.get('eq_db') or ()):
        y = eq_bands(y, post['eq_db'])
    y = y * 10 ** (float(post.get('gain_db', 0.0)) / 20)
    if post.get('pre'):
        y = _pre_clamp(y, post['pre'], scale)
    if post.get('env') and post['env'].get('db'):
        g = _env_gain(len(y), post['env'], scale * tail)
        y = y * (g[:, None] if y.ndim == 2 else g)
    return y


def fit(x, kind='auto', layers=None, candidates=None, clap=False, fast=False, seeds=(0,), name=None, verbose=False,
        collect=None, post=True):
    """a model of x (path or 48 kHz array). kind: 'auto' | 'modal' | 'sines' | 'spectral' | 'hybrid' (layers=
    ('modal', 'spectral') by default). auto: candidates (default plan(character(x))) are all fitted, rendered with
    `seeds` and scored (quality.score; clap=True adds CLAP; the mean over seeds), the best is returned with a
    'fit_engine' entry. collect: a dict that receives every candidate's model ({candidate: model})"""
    import os
    if name is None and isinstance(x, (str, os.PathLike)):
        name = os.path.basename(str(x))
    X = Q.load(x)
    t0 = time.time()
    if kind != 'auto':
        if kind == 'hybrid':
            m = fit_hybrid(X, layers or ('modal', 'spectral'), fast=fast, name=name)
        else:
            m = fit_kind(X, kind, fast=fast, name=name)
        if post:
            m['post'] = fit_post(X, [render(m, seed=s) for s in (0, 1)])
        m['fit_engine'] = {'kind': kind, 'seconds': round(time.time() - t0, 2)}
        return m
    c = character(X)
    cands = list(candidates or plan(c))
    base, rows, best = {}, [], None
    clap_e = None
    seeds = tuple(int(s) for s in seeds) or (0,)
    if clap:
        clap_e = Q.clap_embed([X])[0]
    for cand in cands:
        t1 = time.time()
        try:
            m = _cand_fit(X, cand, fast, base, name)
        except Exception as e:                       # one fitter failing must not stop the others
            rows.append({'candidate': cand, 'error': f'{type(e).__name__}: {e}'})
            if verbose:
                print(f'  {cand}: failed ({type(e).__name__}: {e})')
            continue
        tf = time.time() - t1
        if collect is not None:
            collect[cand] = m
        qs = []
        if post:
            r0 = [render(m, seed=s) for s in (0, 1)]
            m['post'] = fit_post(X, r0)
        ys = [render(m, seed=s) for s in seeds]
        E = Q.clap_embed(ys) if clap_e is not None else None
        for i, y in enumerate(ys):
            q = Q.quality(X, y)
            if E is not None:
                q['clap_cos'] = round(float(clap_e @ E[i]), 4)
            qs.append(q)
        sc = float(np.mean([Q.score(q) for q in qs]))
        row = {'candidate': cand, 'score': round(sc, 3), 'fit_s': round(tf, 2), 'quality': qs[0]}
        rows.append(row)
        if verbose:
            print(f'  {cand}: score {sc:.2f} ({Q.text(qs[0])}) in {tf:.1f} s')
        if best is None or sc < best[0]:
            best = (sc, cand, m)
    if best is None:
        raise RuntimeError('every fitter failed: ' + '; '.join(r.get('error', '') for r in rows))
    m = best[2]
    m['fit_engine'] = {'kind': 'auto', 'chosen': best[1], 'character': c, 'candidates': rows,
                       'seeds': list(seeds), 'seconds': round(time.time() - t0, 2)}
    return m


# ================================================================================================ rendering
def _layer_render(m, seed, two, **kn):
    F = _fitters()
    k = kind_of(m)
    if k == 'sines':
        y = F['sines'].render(m, seed, knobs=kn)
    elif k == 'modal':
        y = F['modal'].render(m, seed, **kn)
    else:
        y = np.asarray(F['spectral'].render(m, seed, limit=False, steady=False, **kn), np.float64)
    y = np.asarray(y, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if two and y.shape[1] == 1:
        y = np.repeat(y, 2, 1)
    if not two and y.shape[1] == 2:
        y = y.mean(1, keepdims=True)
    return y


def _render_hybrid(model, seed=0, pitch=0.0, decay=1.0, bright=0.0, vary=0.0, width=1.0, tone_db=0.0,
                   body_db=0.0, air_db=0.0, stereo=None, ring=None):
    two = (model.get('channels', 1) == 2) if stereo is None else bool(stereo)
    ring = vary > 0 if ring is None else bool(ring)
    gains = {'tone': tone_db, 'body': body_db, 'air': air_db}
    ys = []
    for i, L in enumerate(model['layers']):
        m = L['model']
        k = kind_of(m)
        s = (int(seed) * 7 + 101 * (i + 1)) & 0x7FFFFFFF
        if k == 'sines':
            kn = {'transpose': pitch, 'decay': decay, 'brightness': bright, 'vary': vary, 'ring': ring}
        elif k == 'modal':
            size = 2.0 ** (-pitch / 12.0)
            kn = {'size': size, 'damping': size / max(decay, 1e-3), 'brightness': bright, 'vary': vary,
                  'stereo': False if m.get('side') is None else two, 'width': width, 'ring': ring}
        else:
            kn = {'shift': pitch / 12.0, 'tilt': bright, 'width': width}
        y = _layer_render(m, s, two, **kn) * 10 ** ((float(L.get('gain_db', 0.0)) + gains[L['role']]) / 20)
        ys.append(y)
    n = max(len(y) for y in ys)
    out = np.zeros((n, ys[0].shape[1]))
    for y in ys:
        out[:len(y)] += y
    return out if two else out[:, 0]


def _render_raw(model, seed=0, **knobs):
    """the sound of a model (float64, 48 kHz): (n, 2) when the source was stereo, else (n,). knobs: knobs_of(model)
    names (identity values re-render the fit); unknown names raise"""
    k = kind_of(model)
    if k == 'morph':
        w = knobs.pop('morph', model.get('amount', 0.5))
        return render(morph(model['a'], model['b'], w), seed, **knobs)
    allowed = set(knobs_of(model)) | {'stereo', 'gain', 'gain_db', 'ring'}
    bad = set(knobs) - allowed
    if bad:
        raise ValueError(f'unknown knobs for a {k} model: {sorted(bad)} (known: {sorted(allowed)})')
    F = _fitters()
    two = channels_of(model) == 2
    if k == 'hybrid':
        g = knobs.pop('gain_db', knobs.pop('gain', 0.0))
        return _render_hybrid(model, seed, **knobs) * 10 ** (g / 20)
    if k == 'modal':
        if 'gain_db' in knobs:
            knobs['gain'] = knobs.pop('gain_db')
        y = F['modal'].render(model, seed, **knobs)
    elif k == 'sines':
        if 'gain' in knobs:
            knobs['gain_db'] = knobs.pop('gain')
        knobs.setdefault('stereo', two)
        y = F['sines'].render(model, seed, knobs=knobs)
    else:
        if 'gain' in knobs:
            knobs['gain_db'] = knobs.pop('gain')
        knobs.pop('ring', None)                     # a noise gesture has no ringing modes
        y = np.asarray(F['spectral'].render(model, seed, limit=False, **knobs), np.float64)
        if not two:
            y = y.mean(1)
    return np.asarray(y, np.float64)


def render(model, seed=0, take=None, **knobs):
    """the sound of a model (float64, 48 kHz): (n, 2) when the source was stereo, else (n,), through the model's post
    stage (fitted EQ + gain) when it has one. knobs: knobs_of(model) names (identity values re-render the fit);
    unknown names raise.
    take (not a search knob; default None = True when vary > 0): render a take to use, not the fit to measure.
    Modes and partials ring out past the recording's length to 60 dB down with a fade instead of stopping where
    the trimmed recording stopped (modal / sines ring=True), and the result is high-passed at TAKE_DC_HZ (no DC
    offset, no infrasound: a noise band modulated by its envelope spreads energy down to DC, and a new seed of a
    fit could carry an offset of 1-2 % of its peak). take=False keeps the recording's length and every bit of its
    low end: what the fit's quality is measured on (the identity render, seed 0, is that)."""
    knobs = dict(knobs)
    take = float(knobs.get('vary') or 0.0) > 0 if take is None else bool(take)
    if kind_of(model) == 'morph':
        w = knobs.pop('morph', model.get('amount', 0.5))
        return render(morph(model['a'], model['b'], w), seed, take=take, **knobs)
    knobs.setdefault('ring', take)
    y = _render_raw(model, seed, **knobs)
    k = kind_of(model)
    scale = float(knobs.get('length', 1.0)) / float(knobs.get('speed', 1.0)) if k == 'spectral' else 1.0
    tail = {'modal': lambda: float(knobs.get('size', 1.0)) / float(knobs.get('damping', 1.0)),
            'sines': lambda: float(knobs.get('stretch', 1.0)) * float(knobs.get('decay', 1.0)),
            'hybrid': lambda: float(knobs.get('decay', 1.0))}.get(k, lambda: 1.0)()
    y = apply_post(y, model.get('post'), scale, tail)
    return _dc_block(y) if take else y


def _dc_block(y, hz=TAKE_DC_HZ):
    """a causal 2nd-order Butterworth high-pass (per channel)"""
    from scipy import signal
    return signal.sosfilt(signal.butter(2, hz, 'highpass', fs=SR, output='sos'), np.asarray(y, np.float64), axis=0)


def identity(model):
    """the knob values that re-render the fit (a recipe's DEFAULTS)"""
    return {n: v[1] for n, v in knobs_of(model).items()}


# ================================================================================================ morph
def _assign(fa, ea, fb, eb, max_oct=1.0):
    """pairs (i, j) of a-rows and b-rows by log-frequency distance (optimal assignment; pairs further apart than
    max_oct octaves stay unpaired), plus the unpaired a and b indices"""
    from scipy.optimize import linear_sum_assignment
    fa, fb = np.asarray(fa, float), np.asarray(fb, float)
    if not len(fa) or not len(fb):
        return [], list(range(len(fa))), list(range(len(fb)))
    C = np.abs(np.log2(np.maximum(fa, 1.0))[:, None] - np.log2(np.maximum(fb, 1.0))[None, :])
    da = 10 * np.log10(np.asarray(ea, float) / (np.max(ea) + 1e-30) + 1e-12)
    db = 10 * np.log10(np.asarray(eb, float) / (np.max(eb) + 1e-30) + 1e-12)
    cost = C + 0.003 * np.abs(da[:, None] - db[None, :])
    cost = np.where(C > max_oct, 1e6, cost)
    r, c = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if C[i, j] <= max_oct]
    pa, pb = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [i for i in range(len(fa)) if i not in pa], [j for j in range(len(fb)) if j not in pb]


def _geo(a, b, w):
    return float(np.exp((1 - w) * np.log(max(a, 1e-30)) + w * np.log(max(b, 1e-30))))


def _circ(pa, pb, w):
    return float(np.angle((1 - w) * np.exp(1j * pa) + w * np.exp(1j * pb)))


def _mix_rows(ra, rb, w):
    """damped-sinusoid rows [f, tau, amp, phase] of two sounds -> rows between them: paired rows interpolate
    (frequency, decay, amplitude geometric; phase circular), unpaired rows fade with their side's weight"""
    a = np.array(ra, float).reshape(-1, 4)
    b = np.array(rb, float).reshape(-1, 4)
    for q in (a, b):                                   # a negative amplitude is a phase of pi
        neg = q[:, 2] < 0
        q[neg, 2] *= -1
        q[neg, 3] += np.pi
    pairs, ua, ub = _assign(a[:, 0], a[:, 2] ** 2 * a[:, 1], b[:, 0], b[:, 2] ** 2 * b[:, 1])
    out = []
    for i, j in pairs:
        out.append([_geo(a[i, 0], b[j, 0], w), _geo(a[i, 1], b[j, 1], w), _geo(a[i, 2], b[j, 2], w),
                    _circ(a[i, 3], b[j, 3], w)])
    if w < 1:
        out += [[float(a[i, 0]), float(a[i, 1]), float(a[i, 2] * (1 - w)), float(a[i, 3])] for i in ua]
    if w > 0:
        out += [[float(b[j, 0]), float(b[j, 1]), float(b[j, 2] * w), float(b[j, 3])] for j in ub]
    return out


def _anchors(ta, tb, w):
    """piecewise-linear time maps a -> morph and b -> morph through paired anchor times"""
    ta, tb = np.asarray(ta, float), np.asarray(tb, float)
    tm = (1 - w) * ta + w * tb
    for t in (ta, tb, tm):                             # strictly increasing
        for i in range(1, len(t)):
            t[i] = max(t[i], t[i - 1] + 1e-6)

    def mk(src):
        def f(t):
            t = np.asarray(t, float)
            y = np.interp(t, src, tm)
            return np.where(t > src[-1], tm[-1] + (t - src[-1]), np.where(t < src[0], tm[0] + (t - src[0]), y))
        return f
    return mk(ta), mk(tb), tm


def _mix_curves(ta, va, tb, vb, w):
    """two piecewise-linear curves (already on the morph time axis) -> their weighted mix on the union grid"""
    ta, va, tb, vb = (np.asarray(v, float) for v in (ta, va, tb, vb))
    g = np.unique(np.round(np.concatenate([ta, tb]), 6))
    return g, (1 - w) * np.interp(g, ta, va) + w * np.interp(g, tb, vb)


def _morph_modal(A, B, w):
    from .modal import _rdp
    SA, SB = A['strikes'], B['strikes']
    k = min(len(SA), len(SB))
    wa, wb, tm = _anchors([0.0] + [s['t'] for s in SA[:k]] + [A['length_s']],
                          [0.0] + [s['t'] for s in SB[:k]] + [B['length_s']], w)
    strikes = []
    for i in range(k):
        strikes.append({'t': round(float(tm[i + 1]), 6),
                        'ramp_ms': round((1 - w) * SA[i]['ramp_ms'] + w * SB[i]['ramp_ms'], 4),
                        'modes': _mix_rows(SA[i]['modes'], SB[i]['modes'], w),
                        'attack': _mix_rows(SA[i].get('attack', []), SB[i].get('attack', []), w)})
    for S, wt, wf in ((SA[k:], 1 - w, wa), (SB[k:], w, wb)):
        for s in S:
            if wt <= 0:
                continue
            strikes.append({'t': round(float(wf(s['t'])), 6), 'ramp_ms': s['ramp_ms'],
                            'modes': [[q[0], q[1], q[2] * wt, q[3]] for q in s['modes']],
                            'attack': [[q[0], q[1], q[2] * wt, q[3]] for q in s.get('attack', [])]})
    strikes.sort(key=lambda s: s['t'])
    na, nb = A['noise'], B['noise']
    if list(na['edges']) != list(nb['edges']):
        raise ValueError('modal models with different noise bands cannot be morphed')
    env = []
    for ea, eb in zip(na['env'], nb['env']):
        ea, eb = np.asarray(ea, float), np.asarray(eb, float)
        g, v = _mix_curves(wa(ea[:, 0]), ea[:, 1], wb(eb[:, 0]), eb[:, 1], w)
        keep = _rdp(g, v, 0.25)
        env.append([[round(float(g[j]), 6), round(float(v[j]), 3)] for j in keep])
    sub = None
    xa = (A.get('sub') or {}).get('x')
    xb = (B.get('sub') or {}).get('x')
    if xa or xb:
        xa, xb = np.asarray(xa or [0.0], float) * (1 - w), np.asarray(xb or [0.0], float) * w
        n = max(len(xa), len(xb))
        sx = np.pad(xa, (0, n - len(xa))) + np.pad(xb, (0, n - len(xb)))
        if np.abs(sx).max() > 0:
            sub = {'rate': (A.get('sub') or B.get('sub'))['rate'], 'x': [float(v) for v in sx]}
    side = None
    if A.get('side') or B.get('side'):
        sa = np.asarray(A.get('side') or np.zeros(len(B['side'])), float)
        sb = np.asarray(B.get('side') or np.zeros(len(A['side'])), float)
        side = [round(float(v), 4) for v in (1 - w) * sa + w * sb]
    return {'kind': 'modal', 'version': max(A['version'], B['version']), 'sr': SR,
            'length_s': round(float(tm[-1]), 6), 'channels': max(A.get('channels', 1), B.get('channels', 1)),
            'fade_out_s': (1 - w) * float(A.get('fade_out_s', 0.0)) + w * float(B.get('fade_out_s', 0.0)),
            'strikes': strikes, 'sub': sub, 'noise': {'edges': list(na['edges']), 'h_erb': na['h_erb'], 'env': env},
            'side': side, 'fit': {'modes': int(sum(len(s['modes']) for s in strikes))}}


def _morph_partial(p, q, w, wa, wb):
    t0p, t1p = float(wa(p['t0'])), float(wa(p['t1']))
    t0q, t1q = float(wb(q['t0'])), float(wb(q['t1']))
    t0, t1 = (1 - w) * t0p + w * t0q, (1 - w) * t1p + w * t1q
    t1 = max(t1, t0 + 1e-4)

    def u(pts, warp, a, b):
        pts = np.asarray(pts, float)
        return (warp(pts[:, 0]) - a) / max(b - a, 1e-6), pts[:, 1]

    out = {}
    for key, tf in (('freq', np.log2), ('amp', lambda v: v)):
        up, vp = u(p[key], wa, t0p, t1p)
        uq, vq = u(q[key], wb, t0q, t1q)
        g, v = _mix_curves(up, tf(vp), uq, tf(vq), w)
        if key == 'freq':
            v = 2.0 ** v
        out[key] = [[round(float(t0 + gg * (t1 - t0)), 6), round(float(vv), 4)] for gg, vv in zip(g, v)]
    return {'t0': round(t0, 6), 'rise': (1 - w) * p['rise'] + w * q['rise'], 't1': round(t1, 6),
            'rel': (1 - w) * p['rel'] + w * q['rel'], 'ph': _circ(p['ph'], q['ph'], w), **out}


def _shift_partial(p, warp, gain_db):
    return {'t0': float(warp(p['t0'])), 'rise': p['rise'], 't1': float(warp(p['t1'])), 'rel': p['rel'], 'ph': p['ph'],
            'freq': [[float(warp(t)), f] for t, f in p['freq']],
            'amp': [[float(warp(t)), a + gain_db] for t, a in p['amp']]}


def _morph_sines(A, B, w):
    from .sines import partial_freq, summary
    oa = list(A.get('onsets') or [A.get('onset', 0.0)])
    ob = list(B.get('onsets') or [B.get('onset', 0.0)])
    k = min(len(oa), len(ob))
    wa, wb, tm = _anchors([0.0] + oa[:k] + [A['duration']], [0.0] + ob[:k] + [B['duration']], w)
    PA, PB = A['partials'], B['partials']
    en = lambda p: 10 ** (max(v for _, v in p['amp']) / 10) * (p['t1'] - p['t0'] + 0.01)
    pairs, ua, ub = _assign([partial_freq(p) for p in PA], [en(p) for p in PA],
                            [partial_freq(p) for p in PB], [en(p) for p in PB])
    parts = [_morph_partial(PA[i], PB[j], w, wa, wb) for i, j in pairs]
    if w < 1:
        parts += [_shift_partial(PA[i], wa, 20 * np.log10(1 - w)) for i in ua]
    if w > 0:
        parts += [_shift_partial(PB[j], wb, 20 * np.log10(w)) for j in ub]
    parts.sort(key=lambda p: -max(v for _, v in p['amp']))
    TA, TB = A.get('transients', []), B.get('transients', [])
    kt = min(len(TA), len(TB))
    trans = [{'t': (1 - w) * float(wa(TA[i]['t'])) + w * float(wb(TB[i]['t'])),
              'atoms': _mix_rows(TA[i]['atoms'], TB[i]['atoms'], w)} for i in range(kt)]
    for T, wt, wf in ((TA[kt:], 1 - w, wa), (TB[kt:], w, wb)):
        trans += [{'t': float(wf(tr['t'])), 'atoms': [[f, tau, a * wt, ph] for f, tau, a, ph in tr['atoms']]}
                  for tr in T if wt > 0]
    ra, rb = A['residual'], B['residual']
    if list(ra['bands']) != list(rb['bands']):
        raise ValueError('sines models with different residual bands cannot be morphed')
    ga = wa(np.asarray(ra['times'], float))
    gb = wb(np.asarray(rb['times'], float))
    g = np.unique(np.round(np.concatenate([ga, gb]), 6))
    da, dbb = np.asarray(ra['db'], float), np.asarray(rb['db'], float)
    lev = np.array([(1 - w) * np.interp(g, ga, da[i]) + w * np.interp(g, gb, dbb[i]) for i in range(len(da))])
    ons = sorted([(1 - w) * oa[i] + w * ob[i] for i in range(k)]
                 + ([float(wa(o)) for o in oa[k:]] if w < 1 else [])
                 + ([float(wb(o)) for o in ob[k:]] if w > 0 else []))
    m = {'kind': 'sines', 'version': max(A['version'], B['version']), 'sr': SR, 'duration': round(float(tm[-1]), 6),
         'onset': float(ons[0]) if ons else 0.0, 'onsets': ons, 'partials': parts, 'transients': trans,
         'residual': {'bands': list(ra['bands']), 'times': [float(v) for v in g],
                      'db': [[round(float(v), 2) for v in row] for row in lev]},
         'width': (1 - w) * float(A.get('width', -120.0)) + w * float(B.get('width', -120.0)),
         'channels': max(int(A.get('channels', 1)), int(B.get('channels', 1))), 'analysis': {}}
    m['summary'] = summary(m)
    return m


def _morph_spectral(A, B, w):
    if A['bands'] != B['bands']:
        raise ValueError('spectral models with different bands cannot be morphed')

    def peak(M):
        E = 10 ** (np.asarray(M['env']['db'], float) / 10)
        return float(M['env']['t'][int(np.argmax(E.sum(1)))])
    wa, wb, tm = _anchors([0.0, peak(A), A['dur']], [0.0, peak(B), B['dur']], w)
    ta, tb = wa(np.asarray(A['env']['t'], float)), wb(np.asarray(B['env']['t'], float))
    g = np.unique(np.round(np.concatenate([ta, tb]), 5))
    Ea, Eb = np.asarray(A['env']['db'], float), np.asarray(B['env']['db'], float)
    E = np.stack([(1 - w) * np.interp(g, ta, Ea[:, j]) + w * np.interp(g, tb, Eb[:, j]) for j in range(Ea.shape[1])], 1)
    stereo = None
    if A.get('stereo') or B.get('stereo'):
        def st(M, tt):
            s = M.get('stereo')
            if not s:
                G = len((A.get('stereo') or B.get('stereo'))['groups']) - 1
                z = np.zeros((len(g), G))
                return z, z + 1.0
            ild, icc = np.asarray(s['ild'], float), np.asarray(s['icc'], float)
            return (np.stack([np.interp(g, tt, ild[:, j]) for j in range(ild.shape[1])], 1),
                    np.stack([np.interp(g, tt, icc[:, j]) for j in range(icc.shape[1])], 1))
        (ia, ca), (ib, cb) = st(A, ta), st(B, tb)
        stereo = {'groups': list((A.get('stereo') or B.get('stereo'))['groups']),
                  'ild': [[round(float(v), 2) for v in r] for r in (1 - w) * ia + w * ib],
                  'icc': [[round(float(v), 3) for v in r] for r in (1 - w) * ca + w * cb]}
    ridges = []
    for M, wt, wf in ((A, 1 - w, wa), (B, w, wb)):
        if wt <= 0:
            continue
        for rd in M.get('ridges') or []:
            ridges.append(dict(rd, t=[float(v) for v in wf(np.asarray(rd['t'], float))],
                               db=[float(v) + 20 * np.log10(wt) for v in rd['db']]))
    tex = _mix_texture(A.get('texture'), B.get('texture'), w)
    fine = None
    if A.get('fine') and B.get('fine'):
        fa, fb = A['fine'], B['fine']
        try:
            fine = dict(fa, db=[(1 - w) * u + w * v for u, v in zip(fa['db'], fb['db'])])
        except (KeyError, TypeError):
            fine = fa if w < 0.5 else fb
    return {'kind': 'spectral_noise', 'version': max(A['version'], B['version']), 'source': None, 'sr': SR,
            'dur': round(float(tm[-1]), 4), 'level_db': round((1 - w) * A['level_db'] + w * B['level_db'], 2),
            'bands': A['bands'], 'fit': {}, 'env': {'t': [float(v) for v in g],
                                                    'db': [[round(float(v), 2) for v in r] for r in E]},
            'stereo': stereo, 'fine': fine, 'texture': tex, 'ridges': ridges,
            'channels': max(int(A.get('channels', 1)), int(B.get('channels', 1)))}


def _mix_texture(ta, tb, w):
    if not ta and not tb:
        return None
    dom = copy.deepcopy(ta if (w < 0.5 and ta) or not tb else tb)
    ga, gb = (ta or {}).get('grains'), (tb or {}).get('grains')
    if ga and gb:
        dom['grains'] = dict(ga, density=_geo(ga['density'], gb['density'], w))
    elif ga or gb:
        dom['grains'] = (ga if (1 - w) >= 0.5 else None) if ga else (gb if w >= 0.5 else None)
    aa, ab = (ta or {}).get('am'), (tb or {}).get('am')
    if aa and ab and len(aa.get('depth', [])) == len(ab.get('depth', [])):
        am = copy.deepcopy(aa if w < 0.5 else ab)
        am['depth'] = [(1 - w) * u + w * v for u, v in zip(aa['depth'], ab['depth'])]
        dom['am'] = am
    elif aa or ab:
        am = copy.deepcopy(aa or ab)
        wt = (1 - w) if aa else w
        am['depth'] = [v * wt for v in am.get('depth', [])]
        dom['am'] = am
    return dom


def _mix_post(pa, pb, w):
    if not pa and not pb:
        return None
    za = {'eq_db': [0.0] * 28, 'gain_db': 0.0}
    pa, pb = pa or za, pb or za
    out = {'eq_db': [round((1 - w) * u + w * v, 3) for u, v in zip(pa['eq_db'], pb['eq_db'])],
           'gain_db': round((1 - w) * pa.get('gain_db', 0.0) + w * pb.get('gain_db', 0.0), 3)}
    if pa.get('pre') and pb.get('pre'):            # the pre-onset envelopes on a common normalised time axis
        ra, rb = np.asarray(pa['pre']['db'], float), np.asarray(pb['pre']['db'], float)
        t_on = (1 - w) * pa['pre']['t_on'] + w * pb['pre']['t_on']
        nb = max(1, int(round(t_on / PRE_HOP)))
        u = np.linspace(0, 1, nb)
        mix = (1 - w) * np.interp(u, np.linspace(0, 1, len(ra)), ra) + w * np.interp(u, np.linspace(0, 1, len(rb)), rb)
        out['pre'] = {'t_on': round(t_on, 6), 'db': [round(float(v), 1) for v in mix], 'attack_db': []}
        aa = np.asarray(pa['pre'].get('attack_db') or [], float)
        ab = np.asarray(pb['pre'].get('attack_db') or [], float)
        if len(aa) and len(ab):
            na = max(2, int(round((1 - w) * len(aa) + w * len(ab))))
            u = np.linspace(0, 1, na)
            mix = ((1 - w) * np.interp(u, np.linspace(0, 1, len(aa)), aa)
                   + w * np.interp(u, np.linspace(0, 1, len(ab)), ab))
            out['pre']['attack_db'] = [round(float(v), 1) for v in mix]
    return out


def morph(a, b, amount=0.5):
    """a model between two fits (see _morph); the post stages (EQ, gain) interpolate too"""
    w = float(np.clip(amount, 0.0, 1.0))
    if w in (0.0, 1.0) and kind_of(a) == kind_of(b) != 'morph':   # the endpoints are the sources, exactly
        return copy.deepcopy({k: v for k, v in (a if w == 0.0 else b).items() if k != 'fit_engine'})
    m = _morph(a, b, amount)
    p = _mix_post(a.get('post'), b.get('post'), w)
    if p:
        m['post'] = p
    return m


def _morph(a, b, amount=0.5):
    """a model between two fitted models of the same kind (0 = a, 1 = b): paired modes / partials interpolate
    (frequency, decay, level geometric), unpaired ones fade with their side's weight, envelopes interpolate in dB on
    a time axis warped through the strikes (modal, sines) or the loudness peak (spectral); hybrids morph layer by
    layer (a role only one side has fades)"""
    w = float(np.clip(amount, 0.0, 1.0))
    ka, kb = kind_of(a), kind_of(b)
    if ka != kb:
        raise ValueError(f'cannot morph a {ka} model with a {kb} model: refit one with kind={ka!r}')
    if ka == 'morph':
        raise ValueError('morph the source models, not a morph')
    if ka == 'modal':
        return _morph_modal(a, b, w)
    if ka == 'sines':
        return _morph_sines(a, b, w)
    if ka == 'spectral':
        return _morph_spectral(a, b, w)
    la = {L['role']: L for L in a['layers']}
    lb = {L['role']: L for L in b['layers']}
    layers = []
    for role in [r for r in ('tone', 'body', 'air') if r in la or r in lb]:
        A, B = la.get(role), lb.get(role)
        if A and B:
            layers.append({'role': role, 'gain_db': (1 - w) * A.get('gain_db', 0.0) + w * B.get('gain_db', 0.0),
                           'model': morph(A['model'], B['model'], w)})
        else:
            L, wt = (A, 1 - w) if A else (B, w)
            if wt > 0:
                layers.append(dict(L, gain_db=L.get('gain_db', 0.0) + 20 * np.log10(wt)))
    return {'kind': 'hybrid', 'version': VERSION, 'sr': SR,
            'length_s': round((1 - w) * a['length_s'] + w * b['length_s'], 6),
            'channels': max(a.get('channels', 1), b.get('channels', 1)), 'layers': layers}


def morph_model(a, b, amount=0.5):
    """a 'morph' model: both sources kept, rendered as morph(a, b, amount) (the 'morph' knob moves amount)"""
    if kind_of(a) != kind_of(b):
        raise ValueError(f'cannot morph a {kind_of(a)} model with a {kind_of(b)} model')
    strip = lambda m: {k: v for k, v in m.items() if k != 'fit_engine'}
    return {'kind': 'morph', 'version': VERSION, 'amount': float(amount), 'a': strip(a), 'b': strip(b)}


def fit_pair(xa, xb, clap=False, fast=False, verbose=False):
    """two recordings fitted with ONE kind (so they can be morphed): each is fitted with kind='auto', and if the
    winners differ, both are fitted with each winning kind and the kind with the lower summed score is kept.
    Returns (model a, model b, {kind: (score a, score b)})"""
    import os
    ma = fit(xa, clap=clap, fast=fast, verbose=verbose)
    mb = fit(xb, clap=clap, fast=fast, verbose=verbose)
    ka, kb = ma['fit_engine']['chosen'], mb['fit_engine']['chosen']
    if ka == kb:
        return ma, mb, {ka: (ma['fit_engine']['candidates'], mb['fit_engine']['candidates'])}
    Xa, Xb = Q.load(xa), Q.load(xb)
    out = {}
    names = [os.path.basename(str(v)) if isinstance(v, (str, os.PathLike)) else None for v in (xa, xb)]
    for kd in (ka, kb):
        pair = []
        for X, m, nm in ((Xa, ma, names[0]), (Xb, mb, names[1])):
            if m['fit_engine']['chosen'] == kd:
                pair.append((m, next(r['score'] for r in m['fit_engine']['candidates'] if r['candidate'] == kd)))
                continue
            mm = fit(X, candidates=[kd], clap=clap, fast=fast, name=nm)
            pair.append((mm, mm['fit_engine']['candidates'][0].get('score', 1e9)))
        out[kd] = pair
    kd = min(out, key=lambda k: out[k][0][1] + out[k][1][1])
    return out[kd][0][0], out[kd][1][0], {k: (v[0][1], v[1][1]) for k, v in out.items()}


# ================================================================================================ io
def save(model, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(model, fh, separators=(',', ':'))
    return path


def load(path):
    with open(path, encoding='utf-8') as fh:
        m = json.load(fh)
    kind_of(m)
    return m


def summary(model):
    """a few readable lines on a model"""
    F = _fitters()
    k = kind_of(model)
    fe = model.get('fit_engine') or {}
    head = ''
    if fe.get('kind') == 'auto':
        head = f"auto chose {fe['chosen']} of " + ', '.join(
            f"{r['candidate']} ({r['score']:.2f})" if 'score' in r else f"{r['candidate']} (failed)"
            for r in fe['candidates']) + '\n'
    if k == 'modal':
        return head + F['modal'].summary(model)
    if k == 'sines':
        return head + F['sines'].describe(model)
    if k == 'spectral':
        return head + F['spectral'].summary(model)
    if k == 'morph':
        ind = lambda t: t.replace('\n', '\n     ')
        return (f"morph {model['amount']:.2f} between two {kind_of(model['a'])} models:\n  a: "
                + ind(summary(model['a'])) + '\n  b: ' + ind(summary(model['b'])))
    lines = [head + f"hybrid of {len(model['layers'])} layers ({model['length_s'] * 1000:.0f} ms):"]
    for L in model['layers']:
        lines.append(f"  {L['role']} ({kind_of(L['model'])}{', deterministic part' if L['model'].get('layer') else ''}"
                     f", {L.get('gain_db', 0.0):+.1f} dB): " + summary(L['model']).replace('\n', '\n    '))
    return '\n'.join(lines)
