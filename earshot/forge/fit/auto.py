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
  0. a recording with a bed under it (bed_of: a steady pre-roll 12-40 dB under the sound, a film's music) is
     stripped (strip_bed) and the sound alone is fitted, from its onset; post['bed'] keeps the bed's spectrum and
     render() puts a steady noise with it back under the sound (bed=False: the sound alone, for a kit or a bank).
     Analysis by synthesis decides whether it was a bed (BED_CHECK): the winning candidate's kind is fitted on the
     recording as it is too, and the bed is dropped when that scores lower on the same seeds (a whoosh's own steady
     approach is not a bed).
  1. character() measures attack, decay, crest, tonal share (energy in spectral peaks 12 dB over the local median),
     partial persistence, the glide of the strongest partial (2 semitones over 15 ms count) and flatness, and names
     a class (glide, ringing, struck, gesture, mixed); plan() orders the candidates by class. Under 1 s every
     candidate is fitted (each fit takes 0.5-4 s); longer sounds get the class's four.
  2. every candidate is fitted, then its post stage (fit_post) is measured from two renders: a smooth third-octave
     EQ onto the original's balance, the level, and the attack: the original's 0.25 ms envelope before the onset
     (renders may not stand above it: band noise and window smear leak energy before a strike) and over the first
     6-25 ms after a sharp onset (renders follow its contour within +-9 dB; for a noise model high-passed, and kept
     only when it lowers the score); then the gate: a cut-only gain curve
     where the renders stand over the original after the onset (gated pulses, hard stops, merged runs: modes and
     partials cannot stop), kept when it lowers the score (with CLAP: and costs none). render() always applies the
     post stage, then a fade-in and a peak limit at -1 dBFS (a gain).
  3. each candidate is rendered (seed 0) and scored against the original (quality.score: mel dB, tone, envelope,
     onset, excess / missing energy, + CLAP with clap=True); the lowest score wins. Hybrids reuse the single-kind
     fits as their first layer. The model gets a 'fit_engine' entry: the character, every candidate's metrics, the
     choice. seeds=TAKE_SEEDS (`earshot fit --takes`) scores the mean over seeds 0, 1, 2 instead: a candidate with
     noise in it can win on one lucky seed-0 draw and then make weaker takes (on 10 of the 12 validation fits the
     chosen candidate's seed-0 score sat 0.1-2.4 under its mean over seeds 0-7). With seed 0 alone the best three
     are scored on two more seeds (CONFIRM, CONFIRM_SEEDS) and the choice is made on the mean; a candidate whose CLAP
     is CLAP_MARGIN higher on seed 0 and on that mean, at most CLAP_TOL worse on the score and CLAP_ENV_TOL on the
     envelope, is chosen instead (the model-free terms must not decide a near-tie against a clearly closer sound;
     each confirmed row keeps 'score2' and 'clap2', the one chosen so 'chosen_by': 'clap'). A modal / hybrid
     winner gets 'vary_scale' 0.5 (TAKE_VARY): its takes at vary 0.3 jitter as 0.15 did. A stereo recording with a
     decorrelated tail gives the chosen model fit.space's diffuse side layer (post['space']: a seeded room in the
     side, so the mono sum and every mono measure stay as they were).
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
KINDS = ('modal', 'spectral', 'sines', 'hybrid', 'gesture')
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
        'vary': ((0.0, 0.6), 0.0, 'seeded jitter of the modes and the noise: new takes of the same object'),
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
        'width': ((0.5, 1.5), 1.0, 'x the fitted side'),
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
        'width': ((0.5, 1.5), 1.0, 'stereo width (x every layer\'s side)'),
        'tone_db': ((-6.0, 6.0), 0.0, 'level of the tonal (sines) layer'),
        'body_db': ((-6.0, 6.0), 0.0, 'level of the body (modal) layer'),
        'air_db': ((-6.0, 6.0), 0.0, 'level of the air (spectral) layer'),
    },
}


def _fitters():
    from . import gesture, modal, sines, spectral
    return {'modal': modal, 'sines': sines, 'spectral': spectral, 'gesture': gesture}


def kind_of(model):
    """'modal' | 'sines' | 'spectral' | 'gesture' | 'hybrid' | 'morph'"""
    k = model.get('kind') if isinstance(model, dict) else None
    if k == 'spectral_noise':
        return 'spectral'
    if k not in ('modal', 'sines', 'gesture', 'hybrid', 'morph'):
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
    if k == 'gesture':                               # fit/gesture.py's own table (the same format)
        kn = _fitters()['gesture'].KNOBS
        return {n: v for n, v in kn.items() if model.get('tones') or n not in ('tone_db', 'vary')}
    return KNOBS[k]


def channels_of(x_or_model):
    if isinstance(x_or_model, dict):
        m = x_or_model
        k = kind_of(m)
        if k == 'morph':
            return max(channels_of(m['a']), channels_of(m['b']))
        if k == 'spectral':
            return 2 if m.get('stereo') else int(m.get('channels', 1))
        if k == 'gesture':
            return int(m.get('channels', 1))
        if k == 'sines':
            return int(m.get('channels', 2 if m.get('width', -120.0) > -50.0 else 1))
        return int(m.get('channels', 1))
    x = np.asarray(x_or_model)
    return 1 if x.ndim == 1 or x.shape[1] == 1 or np.allclose(x[:, 0], x[:, 1]) else 2


# ================================================================================================ character
GLIDE_OCT = 2.0 / 12      # character(): a glide is the strongest partial moving this much (2 semitones) ...
GLIDE_TRACK_MS = 15.0     # ... along a track this long (before 0.4: 0.3 octave over 25 ms on 43 ms frames, which
                          # never fired on the films' 2-6 semitone pops and dings)
GLIDE_NFFT, GLIDE_HOP = 1024, 128   # the glide's own analysis: 21 ms frames every 2.7 ms


def _glide(m):
    """(octaves, track ms): the strongest partial's frequency over the frames where it stands out (a peak 15 dB over
    the local median, within 15 dB of its loudest), along the longest track that moves < 2 semitones per frame; the
    glide is the median of the track's last fifth minus its first fifth (0 for a track under 4 frames)"""
    from scipy.ndimage import maximum_filter1d, median_filter
    n, nfft, hop = len(m), GLIDE_NFFT, GLIDE_HOP
    T = max(1, (n + hop - 1) // hop)
    xp = np.zeros(T * hop + nfft)
    xp[nfft // 2:nfft // 2 + n] = m
    fr = np.lib.stride_tricks.sliding_window_view(xp, nfft)[::hop][:T]
    P = np.abs(np.fft.rfft(fr * np.hanning(nfft + 2)[1:-1], axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / SR)
    band = (f >= 60) & (f <= 16000)
    P, fb = P[:, band], f[band]
    fe = P.sum(1)
    if fe.max() <= 0:
        return 0.0, 0.0
    L = 10 * np.log10(P + 1e-20)
    med = median_filter(L, size=(1, 11), mode='nearest')
    peaks = (L >= maximum_filter1d(L, 3, axis=1)) & (L - med > 15) & (L > L.max(1, keepdims=True) - 50)
    tr = []
    for t in np.nonzero(fe > fe.max() * 10 ** -3.5)[0]:
        if not peaks[t].any():
            continue
        j = int(np.argmax(np.where(peaks[t], L[t], -1e9)))
        a, b, c = L[t, max(j - 1, 0)], L[t, j], L[t, min(j + 1, len(fb) - 1)]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
        tr.append((t * hop / SR, np.log2(max(fb[j] + d * (fb[1] - fb[0]), 20.0)), b))
    if len(tr) < 4:
        return 0.0, 0.0
    tt, lf, lv = map(np.asarray, zip(*tr))
    strong = lv > lv.max() - 15
    tt, lf = tt[strong], lf[strong]
    if len(tt) < 4:
        return 0.0, 0.0
    cut = np.nonzero((np.abs(np.diff(lf)) >= 1 / 6) | (np.diff(tt) > 2.5 * hop / SR))[0] + 1   # track breaks
    segs = np.split(np.arange(len(tt)), cut)
    s = max(segs, key=len)
    st, sl = tt[s], lf[s]
    track_ms = float((st[-1] - st[0]) * 1000) if len(st) > 1 else 0.0
    if len(st) < 4:
        return 0.0, round(track_ms, 1)
    q = max(1, len(st) // 5)
    return float(np.median(sl[-q:]) - np.median(sl[:q])), track_ms


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
    glide, track_ms = _glide(m)
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
    c['gliding'] = (abs(c['glide_oct']) >= GLIDE_OCT and c['track_ms'] >= GLIDE_TRACK_MS
                    and c['tonal_share'] >= 0.25)
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


PLANS = {   # class -> candidates, expected best first (a sound under 1 s gets every candidate in ALL after these)
    'glide': ['sines', 'hybrid:sines+modal+spectral', 'hybrid:sines+spectral', 'modal'],
    'ringing': ['sines', 'modal', 'hybrid:sines+spectral', 'spectral'],
    'struck': ['modal', 'hybrid:sines+modal+spectral', 'hybrid:modal+spectral', 'sines'],
    'gesture': ['spectral', 'hybrid:sines+spectral', 'modal', 'sines'],
    'mixed': ['modal', 'spectral', 'hybrid:modal+spectral', 'sines'],
}
LONG_PLANS = {   # a sound of 1 s or more: the class's four only (0.4: the gesture model in place of the spectral one,
    'glide': ['sines', 'gesture', 'hybrid:sines+modal+spectral', 'hybrid:sines+spectral'],   # which won none of the
    'ringing': ['sines', 'gesture', 'modal', 'hybrid:sines+spectral'],                       # films' whooshes, swells,
    'struck': ['modal', 'sines', 'hybrid:modal+spectral', 'spectral'],                       # risers, camera moves)
    'gesture': ['sines', 'gesture', 'modal', 'hybrid:sines+spectral'],
    'mixed': ['modal', 'sines', 'gesture', 'hybrid:modal+spectral'],
}
ALL = ('modal', 'sines', 'gesture', 'spectral', 'hybrid:modal+spectral', 'hybrid:sines+spectral',
       'hybrid:sines+modal+spectral')   # 'gesture' reuses the sines candidate's fit: it always comes after 'sines'


def plan(c, max_candidates=None):
    """ordered candidate kinds for a character (first = the expected best); hybrids as 'hybrid:<a>+<b>[+<c>]'.
    The measured quality decides among them; the class orders them and prunes the slow fits of long sounds: under
    1 s every candidate is tried (each fit takes 0.5-4 s; the gesture 3-60 s), longer sounds the class's four
    (LONG_PLANS: a long struck sound: modal, sines, modal + spectral, spectral; the three-layer hybrid alone took
    147 s on a 4.5 s impact). A long gesture, mixed, glide or ringing sound gets the gesture model (fit/gesture.py:
    tones + air) second, in place of the spectral one (0.4). 'gesture' always comes after 'sines' (it takes the sines
    candidate's partials as its tones)."""
    long_ = c['dur_s'] >= 1.0
    order = list((LONG_PLANS if long_ else PLANS)[c['class']])
    order += [k for k in ALL if k not in order]
    if max_candidates is None:
        max_candidates = 4 if long_ else len(ALL)
    order = order[:max_candidates]
    if 'gesture' in order and 'sines' in order and order.index('gesture') < order.index('sines'):
        order.remove('gesture')
        order.insert(order.index('sines') + 1, 'gesture')
    return order


# ================================================================================================ bed
BED_DB = -40.0            # a pre-roll this loud re the loudest 50 ms (or louder) is a bed under the sound (the film's
                          # music, a room): fit() strips it; a quieter one is the recording's own floor and stays
BED_MIN_S = 0.03          # the pre-roll used (3 ms after the clip's start to 2 ms before the first onset): this long
BED_PRE_MAX_S = 0.12      # ... and the onset within this (a later one is a slow sound's peak, not its start)
BED_MAX_DB = -12.0        # ... but not louder than this (a pre-roll that loud is the sound's own start, or a bed
                          # the sound cannot be told from)
BED_STEADY_DB = 6.0       # ... and steady: its quarters within this, the last at most half of it over the first (a
                          # decaying tail, a swell's start is not a bed)
BED_RISE_DB = 10.0        # ... and the first 10 ms from the onset this much louder (the sound stands out of it)
BED_NFFT = 1024           # the bed's spectrum: STFT power per bin (Hann, hop 1/4), the median over the pre-roll (47 Hz
                          # bins: a music bed's partials stand apart)
BED_OVER = 2.0            # strip_bed subtracts this x the bed's power from every cell ...
BED_FLOOR_DB = -25.0      # ... keeping at least this much of it (no musical-noise holes)
BED_LINE_DB = 10.0        # bed_noise: a bin this far over the median of its 15 neighbours is a steady partial (a sine)


def _stft(m, nfft=BED_NFFT, pad=True):
    from scipy import signal
    return signal.stft(m, SR, window='hann', nperseg=nfft, noverlap=nfft - nfft // 4,
                       **({} if pad else {'boundary': None, 'padded': False}))[2]


def bed_of(x):
    """the bed under a recording, from its pre-roll (the films' clips carry 60 ms before the onset): {'t_on' s (the
    onset: the first 1 ms frame within 30 dB of the loudest that stands BED_RISE_DB over the median of the frames
    before it, refined to a modal.onsets strike within 3 ms), 'level_db' (the pre-roll re the loudest 50 ms), 'n'
    (the recording's length, samples), 'nfft', 'psd_db' (the median STFT power per bin over the pre-roll,
    scipy.signal.stft scaling)}, or None: no such onset between BED_MIN_S and BED_PRE_MAX_S, a pre-roll quieter than
    BED_DB (the recording's own floor) or louder than BED_MAX_DB, or not steady or rising (BED_STEADY_DB)"""
    from .modal import onsets
    m = Q.mono(Q.load(x))
    if len(m) < int(0.03 * SR) or not np.any(m):
        return None
    e = _env_frames(m, 0.001)                        # 1 ms frames: the onset is the first one BED_RISE_DB over the
    top = e.max()                                    # median of those before it (from 3 ms: the edit fade-in)
    k0 = int(round((BED_MIN_S + 0.005) / 0.001))
    k = next((k for k in range(k0, min(len(e), int(BED_PRE_MAX_S / 0.001))) if e[k] >= top - 30
              and e[k] >= np.median(e[3:k - 2]) + BED_RISE_DB), None)
    if k is None:
        return None
    near = [int(o) for o in onsets(m) if abs(o - k * 48) <= 144]      # a strike onset within 3 ms refines it
    on = min(near, key=lambda o: abs(o - k * 48)) if near else k * 48
    a, b = int(0.003 * SR), on - int(0.002 * SR)
    if b - a < int(BED_MIN_S * SR):
        return None
    pre = m[a:b]
    db = lambda v: 10 * np.log10(np.mean(v ** 2) + 1e-20)
    lv = db(pre) - Q.peak_rms_db(m)
    q = [db(v) for v in np.array_split(pre, 4)]
    if (not BED_DB <= lv <= BED_MAX_DB or max(q) - min(q) > BED_STEADY_DB
            or q[-1] - q[0] > BED_STEADY_DB / 2):   # rising: a swell's or a whoosh's own start, not a bed
        return None
    P = np.median(np.abs(_stft(pre, pad=False)) ** 2, axis=1)       # whole frames only (no zero-padded edges)
    return {'t_on': round(on / SR, 6), 'level_db': round(float(lv), 2), 'n': len(m), 'nfft': BED_NFFT,
            'psd_db': [round(float(v), 1) for v in 10 * np.log10(P + 1e-30)]}


def strip_bed(x, bed):
    """x without its bed (bed_of): every STFT cell keeps sqrt(max(1 - BED_OVER x bed power / its power, BED_FLOOR))
    of itself (the gain from the mono sum, smoothed over 3 frames, the same on both channels), and the pre-roll is
    silenced up to 0.5 ms before the onset (a 1.5 ms raised-cosine fade-in): the sound alone, from its onset"""
    from scipy import signal
    X = Q.load(x)
    two = X.ndim == 2
    X2 = X if two else X[:, None]
    n, nfft = len(X2), int(bed['nfft'])
    B = 10 ** (np.asarray(bed['psd_db'], float) / 10)[:, None]
    Zm = _stft(X2.mean(1), nfft)
    G = np.sqrt(np.maximum(1.0 - BED_OVER * B / (np.abs(Zm) ** 2 + 1e-30), 10 ** (BED_FLOOR_DB / 10)))
    G = signal.lfilter(np.ones(3) / 3, [1.0], np.pad(G, ((0, 0), (1, 1)), mode='edge'), axis=1)[:, 2:]
    out = np.zeros_like(X2)
    for c in range(X2.shape[1]):
        y = signal.istft(_stft(X2[:, c], nfft) * G, SR, window='hann', nperseg=nfft, noverlap=nfft - nfft // 4)[1]
        out[:, c] = np.pad(y, (0, max(0, n - len(y))))[:n]
    on = int(round(float(bed['t_on']) * SR))
    e, f = max(0, on - int(0.0005 * SR)), int(0.0015 * SR)
    ramp = np.zeros(n)
    ramp[e:] = 1.0
    s = max(0, e - f)
    ramp[s:e] = 0.5 - 0.5 * np.cos(np.pi * np.arange(e - s) / max(e - s, 1))
    out *= ramp[:, None]
    return out if two else out[:, 0]


def bed_noise(bed, n, seed=0):
    """(n,) the bed layer of a fit of a stripped recording: a steady noise with the bed's spectrum, its lines
    (BED_LINE_DB) as steady sines, over the recording's length (bed['n'] samples; a 2 ms fade-in, a 10 ms fade-out,
    like the films' clips), silent after it"""
    from scipy import signal
    from scipy.ndimage import median_filter
    nfft = int(bed['nfft'])
    nb = min(int(n), int(bed.get('n', n)))
    if nb <= 0:
        return np.zeros(int(n))
    w = np.hanning(nfft + 1)[:-1]
    c = (w ** 2).sum() / w.sum() ** 2 * np.log(2)    # white noise's median STFT power per bin (scipy scaling)
    L = np.asarray(bed['psd_db'], float)
    med = median_filter(L, 15, mode='nearest')
    B = 10 ** (L / 10)
    r = np.random.default_rng([int(seed) & 0xFFFFFFFF, 77])
    t = np.arange(nb) / SR
    lines = np.zeros(nb)                             # a music bed's steady partials (a drone, a held chord) as sines:
    for j in range(1, len(L) - 1):                   # as noise they would flicker 5-10 dB where the bed is steady
        if L[j] - med[j] >= BED_LINE_DB and L[j] >= L[j - 1] and L[j] >= L[j + 1]:
            a, b_, c_ = L[j - 1], L[j], L[j + 1]
            d = 0.5 * (a - c_) / (a - 2 * b_ + c_) if (a - 2 * b_ + c_) < 0 else 0.0
            amp = np.sqrt(4.0 * B[j - 1:j + 2].sum() / 1.5)       # Hann: the peak bin A^2/4, its neighbours A^2/16
            lines += amp * np.cos(2 * np.pi * (j + d) * SR / nfft * t + r.uniform(0, 2 * np.pi))
            B[j - 1:j + 2] = 10 ** (med[j - 1:j + 2] / 10)
    Z = _stft(r.standard_normal(nb + nfft), nfft) * np.sqrt(B[:, None] / c)
    y = signal.istft(Z, SR, window='hann', nperseg=nfft, noverlap=nfft - nfft // 4)[1]
    y = np.pad(y, (0, max(0, nb - len(y))))[:nb] + lines
    fi, fo = min(nb, int(0.002 * SR)), min(nb, int(0.01 * SR))
    y[:fi] *= np.linspace(0.0, 1.0, fi)
    y[nb - fo:] *= np.linspace(1.0, 0.0, fo)
    return np.pad(y, (0, int(n) - nb))


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
    elif kind == 'gesture':                          # opts: tones= a sines fit of the same x to take the tones from
        kw = dict(fast=True) if fast else {}
        kw.update(opts)
        m = F['gesture'].fit(X, name=name, **kw)
    else:
        raise ValueError(f'unknown fitter {kind!r} (modal, sines, spectral, gesture; hybrids via fit_hybrid)')
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
        m.pop('side_t', None)
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


HYBRID_SIDE_SEEDS = 8      # fit_hybrid: stereo renders of a tone layer with a side tried for the residual's side


def _layer_take(resid, X, m):
    """what a deterministic hybrid layer takes out of the residual: its render (seed 0) as (len(X), channels of X).
    A sines layer with a fitted side (0.4) renders that side in the hybrid (_render_hybrid), so for a stereo recording
    the residual leaves it out too, or the air layer models the tones' side again as noise and the side is counted
    twice. The side's quadrature part has a random sign per seed (shaped_side): of HYBRID_SIDE_SEEDS stereo renders
    (the mid the same in all) the one that leaves the least side in the residual is taken, and the mono render when
    none leaves less than it (the tone layer's side does not match the recording's: nothing to take out)"""
    y = _fit_n(X, _render(m, seed=0, final=False))
    if X.shape[1] != 2 or kind_of(m) != 'sines' or not m.get('side'):
        return y
    left = lambda r: float(np.sum((resid[:, 0] - r[:, 0] - resid[:, 1] + r[:, 1]) ** 2))
    best, e = y, left(y)
    for s in range(HYBRID_SIDE_SEEDS):
        r = _fit_n(X, _render(m, seed=s, final=False, stereo=True))
        es = left(r)
        if es < e:
            best, e = r, es
    return best


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
            resid = resid - _layer_take(resid, X, m)
        out.append({'role': ROLE[k], 'gain_db': 0.0, 'model': m})
    return {'kind': 'hybrid', 'version': VERSION, 'sr': SR, 'length_s': round(len(X) / SR, 6),
            'channels': channels_of(X), 'layers': out, 'source': name}


def _cand_fit(X, cand, fast, base, name):
    if cand.startswith('hybrid'):
        layers = HYBRIDS[cand.split(':', 1)[1]] if ':' in cand else ('modal', 'spectral')
        return fit_hybrid(X, layers, fast=fast, base=base, name=name)
    if cand == 'gesture':                            # the tones of the sines candidate, when there is one
        return fit_kind(X, cand, fast=fast, name=name, tones=base.get('sines'))
    m = fit_kind(X, cand, fast=fast, name=name)
    base[cand] = m
    return m


PRE_HOP = 0.00025         # s: the attack envelope grid of the post stage
PRE_HP_HZ = 200.0         # a noise model's attack clamp (fit_post(noise=True), pre['hp_hz']): its correction is
                          # high-passed here (the clamp follows the noise's own 0.25 ms flicker: its +-9 dB steps
                          # spread energy into the empty low bands, a whoosh +8 dB at 31-99 Hz), and the stage is
                          # kept only when the clamp lowers the score (on a smooth whoosh it blunts the onset)
NOISE_KINDS = ('spectral', 'gesture')   # the kinds whose post stage is fitted with noise=True
ATTACK_S = 0.006          # s: the attack window after the onset whose contour the post stage follows, at least,
ATTACK_MAX_S = 0.025      # or up to 4 ms past the loudest frame within this much after the onset (a later main strike)
GATE_HOP = 0.0005         # s: the gate's envelope grid (1 ms windows, half overlapping)
GATE_TOL_DB = 4.0         # the gate cuts where the post-processed renders stand more than this over the original ...
GATE_KEEP_DB = 1.0        # ... down to this much over it
GATE_MIN_CUT_DB = 6.0     # a cut region is kept when it reaches at least this deep (shallower: the EQ's business)
GATE_FALL_DB = 12.0       # ... and where the original lies this far under its loudest of the GATE_LOOK_S before
GATE_LOOK_S = 0.02        # (a gap, a stop, a fast decay; not the random dips of a sustained noise)
GATE_QUIET_DB = 40.0      # (or lies this far under its top: a long silence after a stop is still a gap)
GATE_MIN_S = 0.003        # ... and that lasts this long (shorter notches are a noise's or a beat's flicker)
GATE_DIP_DB = 6.0         # a run of pulses: between two of the original's pulses (peaks of its 5 ms envelope, 6 dB
GATE_DIP_TOL_DB = 2.0     # prominence) the frames this far under the lower pulse are a dip, cut where the renders
                          # stand GATE_DIP_TOL_DB over the original (a smoother render merges the pulses)
GATE_DEPTH_DB = -80.0     # the deepest cut (a hard stop into digital silence)
GATE_EDIT_S = 0.012       # the clip's own edit fade-out (its last 10 ms): not a gate, the cut before it holds
GATE_RDP_DB = 0.5         # breakpoint tolerance of the stored gain curve
GATE_CLAP_TOL = 0.005     # a gate that lowers the score is still dropped when it costs more CLAP than this
GATE_MAP_S = 0.002        # _gate_map: under a modal spacing the gate's last GATE_MAP_S before a strike (the cut's
                          # reopening, a frame before the original rises) moves with that strike, the cut before it
                          # with the strike before (held across a longer gap, cut off where a shorter one ends)
GATE_SMOOTH_S = 0.001     # render: the gate's gain (linear) smoothed by a Hann window this long: its breakpoints sit
                          # 0.5 ms apart, and a 30-80 dB step between two of them put a kink into a loud tone or ring
                          # that the takes' audit heard as a click (a pop's gesture takes: 4 of 4 clicked at the cuts)


def _env_frames(m, hop):
    k = max(1, int(round(hop * SR)))
    nb = len(m) // k
    return 10 * np.log10((m[:nb * k].reshape(nb, k) ** 2).mean(1) + 1e-20) if nb else np.zeros(0)


def fit_post(X, renders, iters=2, max_db=12.0, envelope=False, gate=True, judge=None, sharp=True, noise=False):
    """the post stage of a model, measured by analysis by synthesis (renders = a few seeds of the model):
      eq_db    a smooth third-octave EQ (minimum phase, forge.fx.eq_bands; 28 gains at tone.CEN) that moves the
               renders' mean balance onto the original's
      gain_db  the gain that matches the loudest 50 ms
      pre      the original's 0.25 ms envelope (dB) before its first onset (the first frame within 20 dB of the
               loudest) and, for a sharp onset (>= 15 dB/ms, quality.attack), over the ATTACK_S after it (or to
               4 ms past the loudest frame of the next ATTACK_MAX_S: a key's bottom-out after its first touch):
               {'t_on' s, 'db' [...], 'attack_db' [...], 'v': 2 (sharp=True)}. Band noise and window smear leak energy
               before a strike and soften it (a pre-echoed, blunt attack); _pre_clamp keeps the attack as clean as
               the recording's. noise=True (a NOISE_KINDS model; 0.4): 'hp_hz' PRE_HP_HZ (the clamp's correction
               high-passed: gain steps that follow a noise's own flicker spread energy into empty low bands), and
               'pre' is dropped unless it lowers the score (as the gate is judged). On the films' dev clips: a
               whoosh's gesture fit 10.46 -> 8.67 (no clamp), a swish's 12.68 -> 11.37 (high-passed), a glitch run
               keeps the plain clamp's sharp first burst (9.29, 10.09 without it)
      env      (envelope=True only) after the attack: the renders' mean 5 ms envelope error as a gain curve ({'hop',
               'db'}, -12..+6 dB, smoothed): a tail that rings too long, a fade-out the model missed. Off by default:
               on the 10 short validation refs it lowered mel dB 3.69 -> 3.62 and raised envelope r 0.977 -> 0.986,
               but CLAP fell 0.925 -> 0.918 (a key press 0.955 -> 0.905)
      gate     (gate=True) a cut-only gain curve after the onset (_fit_gate): where the renders stand GATE_TOL_DB over
               the original (the gaps of a run of pulses, a hard stop, a tail the model lets ring on) they are cut to
               it, with edges as sharp as the original's ({'pts': [[t s, dB], ...]}). Modes and partials decay
               exponentially and cannot stop; a gated pulse falls 30-60 dB in 5 ms. Kept only when it lowers the
               renders' mean quality.score, or with judge(list of post-processed renders) -> (score, CLAP cos), when
               it lowers that score and costs at most GATE_CLAP_TOL of CLAP (fit(clap=True) passes one: on the
               films' dev clips a gate that helped every model-free measure still cost CLAP on 4 of 67)"""
    from ..fx import eq_bands
    from ..tone import profile
    xm = Q.mono(X)
    target = profile(xm)
    ys = [Q.mono(y) for y in renders]

    def sc(p):                                       # (score, CLAP) of the renders through a post stage p
        if judge is None:
            return float(np.mean([Q.score(Q.quality(xm, apply_post(y, p))) for y in ys])), 0.0
        return judge([apply_post(y, p) for y in renders])
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
            if sharp:
                post['pre']['v'] = 2
            if noise:
                post['pre']['hp_hz'] = PRE_HP_HZ
            # the clamp pins the attack to the original's contour: re-level what it leaves
            g2 = Q.peak_rms_db(xm) - float(np.mean([Q.peak_rms_db(apply_post(y, post)) for y in ys]))
            post['gain_db'] = round(float(g + g2), 2)
            if noise:                                # a noise's clamp only where it helps (judged like the gate)
                bare = {'eq_db': post['eq_db'], 'gain_db': round(float(g), 2)}
                if sc(bare)[0] < sc(post)[0]:
                    post = bare
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
    if gate:
        gt = _fit_gate(xm, [apply_post(y, post) for y in ys], post)
        if gt:
            cand = dict(post, gate=gt)
            g4 = Q.peak_rms_db(xm) - float(np.mean([Q.peak_rms_db(apply_post(y, cand)) for y in ys]))
            cand['gain_db'] = round(float(post['gain_db'] + g4), 2)
            (s1, c1), (s0, c0) = sc(cand), sc(post)
            if s1 < s0 and c1 >= c0 - GATE_CLAP_TOL:
                post = cand
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


def _env_half(m, hop=GATE_HOP):
    """power (dB) of 2-hop windows on a `hop` grid: frame k covers [k hop, (k + 2) hop), its centre is (k + 1) hop"""
    k = max(1, int(round(hop * SR)))
    nb = len(m) // k
    if nb < 3:
        return np.zeros(0)
    p = (m[:nb * k].reshape(nb, k) ** 2).mean(1)
    return 10 * np.log10(0.5 * (p[:-1] + p[1:]) + 1e-20)


def _runs(mask):
    """[(first, last)] index pairs of the runs of True in a boolean array"""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def _fit_gate(xm, ys, post, pulses=True):
    """the gate of a post stage (fit_post): a cut-only gain curve, {'pts': [[t s, dB], ...]} (linear in dB between
    breakpoints, 0 dB before the first; the last value holds past the end, so a hard stop stays shut in a take that
    rings out), or None when nothing needs cutting. On 1 ms windows (GATE_HOP grid, a 3-frame median: one frame's
    flicker is not a gap) from the original's onset: where the original has fallen GATE_FALL_DB under its loudest of
    the GATE_LOOK_S before (or lies GATE_QUIET_DB under its top) and the renders' mean (ys, through the rest of the
    post stage) stands more than GATE_TOL_DB over it, or (pulses=True) in a dip between two of its pulses where they
    stand GATE_DIP_TOL_DB over it, the renders are cut to GATE_KEEP_DB over the original (at most GATE_DEPTH_DB). A
    run of cut frames survives when it reaches GATE_MIN_CUT_DB and lasts GATE_MIN_S; inside it the cut is smoothed (a
    5-frame median, then 1-2-1), its edges are not: the gain falls within a frame of the original's fall and is back
    to 0 dB a frame before the original rises again (the next pulse's onset is never cut). The clip's own edit fade
    (its last GATE_EDIT_S) is not a gate: the cut before it holds through it"""
    from scipy.ndimage import maximum_filter1d, median_filter
    from .modal import _rdp
    eo = _env_half(xm)
    if len(eo) < 8 or not ys:
        return None
    er = []
    for y in ys:
        e = _env_half(Q.mono(y))
        er.append(10 ** (np.pad(e, (0, max(0, len(eo) - len(e))), constant_values=-200.0)[:len(eo)] / 10))
    er = 10 * np.log10(np.mean(er, 0) + 1e-20)
    eo, er = median_filter(eo, 3, mode='nearest'), median_filter(er, 3, mode='nearest')   # a 1-frame flicker of a
    n = len(eo)                                                                          # noise is not a gap
    look = int(round(GATE_LOOK_S / GATE_HOP))
    fell = ((eo <= maximum_filter1d(eo, look + 1, origin=look // 2) - GATE_FALL_DB)      # under its loudest just before
            | (eo <= eo.max() - GATE_QUIET_DB))                                         # or quiet, however long
    top = max(eo.max(), er.max())
    pre = post.get('pre') or {}
    t_on = pre.get('t_on')
    if t_on is None:
        t_on = float(np.argmax(eo > eo.max() - 20)) * GATE_HOP
    i0 = int(np.ceil(t_on / GATE_HOP))
    d = eo - er
    dip = np.zeros(n, bool)                          # the dips between the original's pulses
    if pulses:
        from scipy.signal import find_peaks
        e5 = np.convolve(eo, np.ones(10) / 10, mode='same')
        pk = [p for p in find_peaks(e5, prominence=6.0, distance=16)[0] if e5[p] > e5.max() - 25]
        for p, q in zip(pk[:-1], pk[1:]):
            seg = np.arange(p, q)
            dip[seg[e5[seg] < min(e5[p], e5[q]) - GATE_DIP_DB]] = True
    hit = ((d < -GATE_TOL_DB) & fell) | ((d < -GATE_DIP_TOL_DB) & dip)
    cut = np.where(hit & (er > top - 90.0), np.maximum(d + GATE_KEEP_DB, GATE_DEPTH_DB), 0.0)
    cut = np.minimum(cut, 0.0)
    cut[:i0] = 0.0
    i_edit = max(i0, n - int(round(GATE_EDIT_S / GATE_HOP)))
    cut[i_edit:] = cut[i_edit - 1] if i_edit > i0 else 0.0
    out = np.zeros(n)
    n_min = int(round(GATE_MIN_S / GATE_HOP))
    for a, b in _runs(cut < 0):
        seg = cut[a:b + 1]
        if seg.min() > -GATE_MIN_CUT_DB or (len(seg) < n_min and b + 1 < n):
            continue
        if len(seg) >= 5:
            seg = median_filter(seg, 5, mode='nearest')
            seg = np.convolve(np.pad(seg, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
        seg = np.minimum(seg, 0.0)
        if b + 1 < n and len(seg) > 1:              # open a frame early: the original rises in frame b + 1
            seg[-1] = 0.0
        out[a:b + 1] = seg
    if not (out < -0.5).any():
        return None
    t = (np.arange(n) + 1) * GATE_HOP                # frame centres
    keep = _rdp(t, out, GATE_RDP_DB)
    return {'pts': [[round(float(t[i]), 5), round(float(out[i]), 1)] for i in keep]}


def _gate_gain(n, gate, tmap=None):
    """the gate's gain per sample (linear, GATE_SMOOTH_S smooth); tmap: the knobs' time map of the gate's breakpoints
    (render)"""
    pts = np.asarray(gate['pts'], float).reshape(-1, 2)
    t, g = pts[:, 0], pts[:, 1]
    if tmap is not None:
        t, g = tmap(t, g)
    a = 10 ** (np.interp(np.arange(n) / SR, t, g, left=0.0, right=float(g[-1]) if len(g) else 0.0) / 20)
    k = int(round(GATE_SMOOTH_S * SR)) | 1
    if k > 1 and n > k:
        w = np.hanning(k + 2)[1:-1]
        a = np.convolve(np.pad(a, k // 2, mode='edge'), w / w.sum(), mode='valid')
    return a


def _env_gain(n, env, scale=1.0, anchor=0.0):
    d = np.asarray(env['db'], float)
    hop = float(env['hop']) * scale
    tc = (np.arange(len(d)) + 0.5) * hop * SR
    if anchor and scale != 1.0:                      # time scaled about the anchor (a gesture's first strike)
        tc = (anchor + ((np.arange(len(d)) + 0.5) * float(env['hop']) - anchor) * scale) * SR
    return 10 ** (np.interp(np.arange(n), tc, d, left=0.0, right=float(d[-1]) if len(d) else 0.0) / 20)


def _smooth_cuts(gdb):
    """widen each cut by a frame and smooth with 1-2-1 (smooth edges)"""
    neg = np.minimum(gdb, 0.0)
    neg = np.minimum(neg, np.minimum(np.r_[neg[1:], 0.0], np.r_[0.0, neg[:-1]]))
    gdb = np.where(gdb < 0, neg, gdb)
    return np.convolve(np.pad(gdb, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid') if len(gdb) else gdb


def _pre_clamp(y, pre, scale=1.0, anchor=0.0):
    """the attack through the original's envelope (0.25 ms frames): before the onset y may not stand above the
    original (+1.5 dB; only cut), over the ATTACK_S after it y follows the original's contour (+-9 dB, the correction
    fading out over the last 2 ms), so a smeared or pre-echoed attack comes back as sharp as the recording's.
    scale: time factor of the render (spectral length / speed, gesture length) about `anchor` (s; a gesture with
    tones stretches about their first strike, tones['onset'], so its onset stays where it was: the pre-onset window
    ends at anchor + (t_on - anchor) x scale, and the original's pre-roll envelope is read through the same map;
    0: about the start). A stage fitted with sharp=True (0.4, pre['v'] = 2)
    smooths the pre-onset cut and the attack contour apart: the edge smoothing carried the pre-onset cut into the
    strike's first frame (half of a -20 dB cut, all but silencing it after a stripped bed's -200 dB pre-roll).
    pre['hp_hz'] (a noise model's stage, 0.4): the correction (y x (gain - 1)) is high-passed there (4th-order
    Butterworth, zero phase) before it is added: it cannot put energy into the bands under it"""
    v2 = pre.get('v') == 2
    k = int(round(PRE_HOP * SR))
    ref_pre = np.asarray(pre.get('db') or [], float)
    ref_att = np.asarray(pre.get('attack_db') or [], float)
    t_on = float(pre['t_on'])
    about = bool(anchor) and scale != 1.0
    n_pre = int(round((anchor + (t_on - anchor) * scale if about else t_on * scale) / PRE_HOP))
    n_att = len(ref_att) if scale == 1.0 else int(round(len(ref_att) * scale))
    nb = min(len(y) // k, n_pre + n_att)
    if nb < 2:
        return y
    m = y.mean(1) if y.ndim == 2 else y
    e = _env_frames(m[:nb * k], PRE_HOP)
    gdb = np.zeros(nb)
    if n_pre > 0 and len(ref_pre):
        u = np.arange(min(n_pre, nb)) / max(n_pre - 1, 1)
        if about:                                    # the new pre-onset frames back to the original's time
            t_new = u * (anchor + (t_on - anchor) * scale)
            u = np.clip((anchor + (t_new - anchor) / scale) / max(t_on, 1e-9), 0.0, 1.0)
        want = np.interp(u, np.linspace(0, 1, len(ref_pre)), ref_pre)
        gdb[:len(want)] = np.minimum(0.0, want + 1.5 - e[:len(want)])
    if n_att > 0 and len(ref_att) and nb > n_pre:
        j = np.arange(n_pre, nb)
        want = np.interp((j - n_pre) / max(n_att - 1, 1), np.linspace(0, 1, len(ref_att)), ref_att)
        fade = np.clip((n_pre + n_att - 1 - j) / max(1, int(round(0.002 / PRE_HOP))), 0.0, 1.0)
        gdb[n_pre:] = np.clip(want - e[n_pre:nb], -9.0, 9.0) * fade
    if v2 and 0 < n_pre < nb:
        gdb = np.r_[_smooth_cuts(gdb[:n_pre]), _smooth_cuts(gdb[n_pre:])]
    else:
        gdb = _smooth_cuts(gdb)
    n_end = nb * k
    tt = (np.arange(nb) + 0.5) * k
    gs = 10 ** (np.interp(np.arange(n_end), np.r_[tt, n_end], np.r_[gdb, 0.0]) / 20)
    y = np.array(y, copy=True)
    g2 = gs[:, None] if y.ndim == 2 else gs
    hp = float(pre.get('hp_hz') or 0.0)
    if hp > 0:                                       # the correction high-passed (zero phase): no energy below hp
        from scipy import signal
        n_pad = min(len(y), n_end + int(0.02 * SR))
        d = np.zeros_like(y[:n_pad])
        d[:n_end] = y[:n_end] * (g2 - 1.0)
        sos = signal.butter(4, hp, 'highpass', fs=SR, output='sos')
        y[:n_pad] += signal.sosfiltfilt(sos, d, axis=0)
        return y
    y[:n_end] *= g2
    return y


def apply_post(y, post, scale=1.0, tail=1.0, tmap=None, anchor=0.0):
    """a render through a model's post stage (EQ, gain, attack clamp, envelope calibration, gate). scale: the
    render's time factor (spectral length / speed, gesture length) about `anchor` (s: a gesture with tones stretches
    about their first strike; 0 = the start); tail: the decay knobs' factor on the envelope after the attack;
    tmap: (t, dB) -> (t, dB), where the knobs moved the gate's breakpoints (default: scaled by `scale`)"""
    if not post:
        return y
    from ..fx import eq_bands
    if any(post.get('eq_db') or ()):
        y = eq_bands(y, post['eq_db'])
    y = y * 10 ** (float(post.get('gain_db', 0.0)) / 20)
    if post.get('pre'):
        y = _pre_clamp(y, post['pre'], scale, anchor)
    if post.get('env') and post['env'].get('db'):
        g = _env_gain(len(y), post['env'], scale * tail, anchor)
        y = y * (g[:, None] if y.ndim == 2 else g)
    if post.get('gate') and post['gate'].get('pts'):
        if tmap is None and scale != 1.0:
            tmap = lambda t, g: (t * scale, g)
        g = _gate_gain(len(y), post['gate'], tmap)
        y = y * (g[:, None] if y.ndim == 2 else g)
    return y


def fit(x, kind='auto', layers=None, candidates=None, clap=False, fast=False, seeds=(0,), name=None, verbose=False,
        collect=None, post=True, bed=True, space=True, confirm=None):
    """a model of x (path or 48 kHz array). kind: 'auto' | 'modal' | 'sines' | 'spectral' | 'hybrid' (layers=
    ('modal', 'spectral') by default). auto: candidates (default plan(character(x))) are all fitted, rendered with
    `seeds` and scored (quality.score; clap=True adds CLAP; the mean over seeds), the best is returned with a
    'fit_engine' entry. collect: a dict that receives every candidate's model ({candidate: model}).
    bed=True: a recording with a bed under it (bed_of: a film's music) is stripped first (strip_bed) and the sound
    alone is fitted, from its onset, and the post stage is measured on it; the bed is kept as post['bed']: render
    puts a steady noise with its spectrum back under the sound (render(bed=False): the sound alone), and the
    candidates are scored that way against the original (energy the bed covers is not excess; energy the fit adds
    over it is). space=True: a stereo recording whose tail is decorrelated gets the diffuse side layer of fit.space
    (post['space']), fitted on the chosen model only, on its dry render (seed 0, through its post stage): it costs
    0.6-1.6 s, and fitted per candidate it changed no choice on the films' dev clips (the stereo score terms only
    separate candidates that already sound alike). confirm (default CONFIRM; 0 = off; with one seed only): the best
    this many candidates are scored on CONFIRM_SEEDS too and the choice is made on the mean (each row's 'score2'),
    then the CLAP margin rule (CLAP_MARGIN, module doc) may pick a candidate CLAP clearly prefers. With a bed, the
    winner's kind is also fitted on the recording as it is (BED_CHECK; a 'plain:<kind>' row) and the bed is dropped
    when that scores lower (fit_engine['bed']: 'score_stripped', 'score_plain', 'kept')"""
    import os
    if name is None and isinstance(x, (str, os.PathLike)):
        name = os.path.basename(str(x))
    X0 = X = Q.load(x)
    t0 = time.time()
    bd = bed_of(X) if bed else None
    if bd:
        X = strip_bed(X, bd)                         # the fitters and the post stage see the sound alone; the
    if kind != 'auto':                               # candidates are scored with the bed back on the original
        if kind == 'hybrid':
            m = fit_hybrid(X, layers or ('modal', 'spectral'), fast=fast, name=name)
        else:
            m = fit_kind(X, kind, fast=fast, name=name)
        if post:
            m['post'] = fit_post(X, [_render(m, seed=s, final=False) for s in (0, 1)],
                                 noise=kind_of(m) in NOISE_KINDS)
            _add_space(m, X, space)
        _set_take(m, bd)
        m['fit_engine'] = {'kind': kind, 'seconds': round(time.time() - t0, 2)}
        return m
    c = character(X)
    cands = list(candidates or plan(c))
    base, rows, best = {}, [], None
    kept = {}
    clap_e = None
    seeds = tuple(int(s) for s in seeds) or (0,)
    if clap:
        clap_e = Q.clap_embed([X0])[0]
    judge = _judge(X0, clap_e, bd)                   # the post stage's gate is kept by the same score, with CLAP

    def scored(m, ss):                               # the scores of a model's renders on seeds ss against the
        ys = [render(m, seed=s) for s in ss]         # original, the first one's quality, the CLAP cosines
        E = Q.clap_embed(ys) if clap_e is not None else None
        qs = []
        for i, y in enumerate(ys):
            q = Q.quality(X0, y)
            if E is not None:
                q['clap_cos'] = round(float(clap_e @ E[i]), 4)
            qs.append(q)
        return [Q.score(q) for q in qs], qs[0], [q.get('clap_cos') for q in qs]
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
        if post:
            r0 = [_render(m, seed=s, final=False) for s in (0, 1)]
            m['post'] = fit_post(X, r0, judge=judge, noise=kind_of(m) in NOISE_KINDS)
        _set_take(m, bd, vary=False)                 # the bed under the renders scored against the original
        scs, q0, _ = scored(m, seeds)
        sc = float(np.mean(scs))
        row = {'candidate': cand, 'score': round(sc, 3), 'fit_s': round(tf, 2), 'quality': q0}
        rows.append(row)
        if verbose:
            print(f'  {cand}: score {sc:.2f} ({Q.text(q0)}) in {tf:.1f} s')
        kept[cand] = (m, row)
        if best is None or sc < best[0]:
            best = (sc, cand, m)
    if best is None:
        raise RuntimeError('every fitter failed: ' + '; '.join(r.get('error', '') for r in rows))
    confirm = CONFIRM if confirm is None else int(confirm)
    more = [s for s in CONFIRM_SEEDS if s not in seeds]
    confirmed = confirm > 1 and len(seeds) == 1 and len(kept) > 1 and bool(more)
    if confirmed:
        top = sorted(kept, key=lambda k: kept[k][1]['score'])[:confirm]     # the winner's curse: a candidate with a
        for k in top:                                                       # lot of noise in it wins on one lucky
            s2, _, c2 = scored(kept[k][0], more)                            # draw (module doc)
            kept[k][1]['score2'] = round(float(np.mean([kept[k][1]['score']] + s2)), 3)
            if clap_e is not None:
                kept[k][1]['clap2'] = round(float(np.mean([kept[k][1]['quality']['clap_cos']] + c2)), 4)
        k = min(top, key=lambda k: kept[k][1]['score2'])
        best = (kept[k][1]['score2'], k, kept[k][0])
        if verbose:
            print(f'  confirmed on seeds {more}: ' + ', '.join(f"{k} {kept[k][1]['score2']:.2f}" for k in top))
    if CLAP_MARGIN and confirmed and clap_e is not None:  # a candidate CLAP clearly prefers, close on the score
        rows_ = {k: v[1] for k, v in kept.items()}
        alts = _clap_alts(rows_, best[1])
        for k in alts:                               # on the confirm seeds too (a lucky seed-0 draw is not enough)
            r_ = kept[k][1]
            if r_.get('clap2') is None:
                s2, _, c2 = scored(kept[k][0], more)
                r_['score2'] = round(float(np.mean([r_['score']] + s2)), 3)
                r_['clap2'] = round(float(np.mean([r_['quality']['clap_cos']] + c2)), 4)
        k = _clap_pick(rows_, best[1], alts)
        if k:
            kept[k][1]['chosen_by'] = 'clap'
            if verbose:
                print(f"  {k} chosen by CLAP ({kept[k][1]['clap2']:.3f} against {kept[best[1]][1]['clap2']:.3f})")
            best = (kept[k][1]['score2'], k, kept[k][0])
    bed_info = None
    if bd:
        bed_info = {'level_db': bd['level_db'], 't_on': bd['t_on']}
        if BED_CHECK:                                # analysis by synthesis decides whether the bed is one
            t1 = time.time()
            try:
                b0 = {}
                if best[1] == 'gesture':             # its tones come from a sines fit of the same recording
                    _cand_fit(X0, 'sines', fast, b0, name)
                mp = _cand_fit(X0, best[1], fast, b0, name)
                if post:
                    mp['post'] = fit_post(X0, [_render(mp, seed=s, final=False) for s in (0, 1)],
                                          judge=_judge(X0, clap_e, None), noise=kind_of(mp) in NOISE_KINDS)
                scs, q0, _ = scored(mp, list(seeds) + (more if confirmed else []))
                sp_ = round(float(np.mean(scs)), 3)
                rows.append({'candidate': f'plain:{best[1]}', 'score': sp_, 'fit_s': round(time.time() - t1, 2),
                             'quality': q0})
                bed_info.update(score_stripped=round(float(best[0]), 3), score_plain=sp_, kept=True)
                if verbose:
                    print(f'  {best[1]} on the recording as it is (no bed stripped): score {sp_:.2f} against '
                          f'{best[0]:.2f}')
                if sp_ < best[0]:                    # the "bed" was the sound's own start (a whoosh's approach)
                    best, bd, X = (sp_, best[1], mp), None, X0
                    bed_info['kept'] = False
            except Exception as e:                   # the stripped fit stands
                rows.append({'candidate': f'plain:{best[1]}', 'error': f'{type(e).__name__}: {e}'})
    m = best[2]
    if post:
        _add_space(m, X, space)
    _set_take(m, bd)
    m['fit_engine'] = {'kind': 'auto', 'chosen': best[1], 'character': c, 'candidates': rows,
                       'seeds': list(seeds), 'seconds': round(time.time() - t0, 2)}
    sp = (m.get('post') or {}).get('space')
    if sp:
        m['fit_engine']['space'] = dict(sp.get('fit') or {}, tail_corr=sp.get('tail_corr'))
    if bed_info:
        m['fit_engine']['bed'] = bed_info
    return m


CLAP_MARGIN = 0.02        # fit(): a candidate whose CLAP beats the choice's by this much, on seed 0 and on the mean over
CLAP_TOL = 1.5            # the confirm seeds, with a score at most this much higher (seed 0 and the mean) and an
CLAP_ENV_TOL = 0.05       # envelope r at most this much lower, is chosen instead (the highest such mean CLAP). The
                          # model-free terms (mel, onset, envelope, excess) decided near-ties against a clearly closer
                          # sound: on the held-out check 5 of 6 regressions were a gesture fit chosen with 0.015-0.06
                          # less CLAP than another candidate. On the dev clips (0.015-0.03 margins, score tolerances 1-4
                          # tried): 5 choices change, all to a higher CLAP, grades 30/26/20/2 -> 33/24/20/1, takes 46
                          # -> 46; a tolerance of 4 also took a candidate with an envelope r of 0.58 (the guard)
BED_CHECK = True          # fit(): a recording with a bed (bed_of) is also fitted as it is with the winning candidate's
                          # kind, and the bed is dropped when that scores lower on the same seeds (analysis by
                          # synthesis): a whoosh's own steady approach 13.6 dB under its peak read as a music bed, and
                          # every candidate fitted on the stripped rest fell (CLAP 0.64 against 0.88)


def _clap_alts(rows, best):
    """the candidates the CLAP margin rule weighs against the choice `best` ({candidate: row}, rows with 'score' and
    'quality' as fit() keeps them): CLAP_MARGIN more CLAP on seed 0, a seed-0 score at most CLAP_TOL higher"""
    rb = rows[best]
    c0 = rb['quality'].get('clap_cos')
    if c0 is None:
        return []
    return [k for k, r in rows.items() if k != best and r['quality'].get('clap_cos') is not None
            and r['quality']['clap_cos'] >= c0 + CLAP_MARGIN and r['score'] <= rb['score'] + CLAP_TOL]


def _clap_pick(rows, best, alts):
    """the alternative (of _clap_alts) the CLAP margin rule chooses, or None: its mean CLAP over the confirm seeds
    ('clap2') CLAP_MARGIN over the choice's, its mean score ('score2') at most CLAP_TOL over it and its envelope r at
    most CLAP_ENV_TOL under it; the highest such mean CLAP"""
    rb = rows[best]
    ok = [k for k in alts if rows[k].get('clap2') is not None and rb.get('clap2') is not None
          and rows[k]['clap2'] >= rb['clap2'] + CLAP_MARGIN and rows[k]['score2'] <= rb['score2'] + CLAP_TOL
          and rows[k]['quality']['env_corr'] >= rb['quality']['env_corr'] - CLAP_ENV_TOL]
    return max(ok, key=lambda k: rows[k]['clap2']) if ok else None


def _judge(X0, clap_e, bd):
    """the post stage's judge (fit_post(judge=)): (mean score, mean CLAP) of renders against the original X0 with the
    bed bd under them (None: none), or None without CLAP"""
    if clap_e is None:
        return None

    def judge(ys):
        ys = [_with_bed(y, bd, s) for s, y in enumerate(ys)]
        c = [float(clap_e @ e) for e in Q.clap_embed(ys)]
        return (float(np.mean([Q.score(dict(Q.quality(X0, y), clap_cos=v)) for y, v in zip(ys, c)])),
                float(np.mean(c)))
    return judge


CONFIRM = 3               # fit(): the best this many candidates (seed-0 score) are scored on CONFIRM_SEEDS too, chosen
CONFIRM_SEEDS = (2, 3)    # by the mean (each row's 'score2'): on short noise-led sounds (keys, ticks) one seed's CLAP
                          # spread 0.84-0.92 and its score 7-13, and the seed-0 choice took the lucky draws (a key's
                          # gesture fit scored 7.11 on seed 0 and 8.0-12.9 on seeds 1-5; an air-only gesture fit at CLAP
                          # 0.906 on seed 0 averaged 0.885 over seeds 1-4, and its takes failed the take check). Not
                          # seeds 0 and 1: the post stage is fitted on those. Costs 2-30 s a fit (6 renders + CLAP)
TAKE_VARY = {'modal': 0.5, 'hybrid': 0.5}   # fit(): a model's 'vary_scale' (render multiplies the vary knob by it).
                                            # Modal / hybrid takes drifted from their fit at the bank's vary 0.3 (on
                                            # the films' clips CLAP to the target -0.047 / -0.075, 10 / 0 of 38 / 9
                                            # passing the take check; sines +0.003, 38 of 65): 0.3 now jitters them
                                            # as 0.15 did


def _set_take(m, bd, vary=True):
    """the fit's take settings: the vary scale of its kind (TAKE_VARY) and its stripped bed (post['bed'])"""
    s = TAKE_VARY.get(kind_of(m))
    if vary and s is not None:
        m['vary_scale'] = s
    if bd:
        m['post'] = dict(m.get('post') or {}, bed=bd)


def _with_bed(y, bd, seed=0):
    """a dry render with the bed layer under it (render's default for a fit of a stripped recording)"""
    if not bd:
        return y
    b = bed_noise(bd, len(y), seed)
    return y + (b[:, None] if np.ndim(y) == 2 else b)


def _add_space(m, X, on=True):
    """the diffuse side layer (fit.space) of a stereo recording with a decorrelated tail (X: the sound alone, a
    stripped bed's too), fitted on the model's dry render (seed 0, through its post stage, without a bed, before the
    final fade and limit): post['space'], or nothing"""
    m.get('post', {}).pop('space', None)
    if not on or channels_of(X) != 2 or 'post' not in m:
        return
    from . import space as SP
    if not SP.needs_space(X):
        return
    sp = SP.fit_space(X, SR, dry=_render(m, seed=0, bed=False, final=False, space=False))
    if sp:
        m['post']['space'] = sp


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
        if k == 'sines':                             # the tonal layer keeps its fitted side (0.4: sines 'side')
            kn = {'transpose': pitch, 'decay': decay, 'brightness': bright, 'vary': vary, 'ring': ring,
                  'stereo': two and bool(m.get('side')), 'width': width}
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
    elif k == 'gesture':
        if 'gain' in knobs:
            knobs['gain_db'] = knobs.pop('gain')
        y = F['gesture'].render(model, seed, **knobs)
    else:
        if 'gain' in knobs:
            knobs['gain_db'] = knobs.pop('gain')
        knobs.pop('ring', None)                     # a noise gesture has no ringing modes
        y = np.asarray(F['spectral'].render(model, seed, limit=False, **knobs), np.float64)
        if not two:
            y = y.mean(1)
    return np.asarray(y, np.float64)


def render(model, seed=0, take=None, bed=None, **knobs):
    """the sound of a model (float64, 48 kHz): (n, 2) when the source was stereo, else (n,), through the model's post
    stage (fitted EQ + gain) when it has one. knobs: knobs_of(model) names (identity values re-render the fit);
    unknown names raise.
    take (not a search knob; default None = True when vary > 0): render a take to use, not the fit to measure.
    Modes and partials ring out past the recording's length to 60 dB down with a fade instead of stopping where
    the trimmed recording stopped (modal / sines ring=True), and the result is high-passed at TAKE_DC_HZ (no DC
    offset, no infrasound: a noise band modulated by its envelope spreads energy down to DC, and a new seed of a
    fit could carry an offset of 1-2 % of its peak). take=False keeps the recording's length and every bit of its
    low end: what the fit's quality is measured on (the identity render, seed 0, is that).
    A model's 'vary_scale' (fit: TAKE_VARY) multiplies the vary knob. post['space'] (fit.space) adds the diffuse side
    after the post stage (the width knob scales it; a take lets it ring out). bed (not a knob; default None = when the
    model has one): a fit of a stripped recording (post['bed']) puts a steady noise with its bed's spectrum under the
    sound over the recording's length (bed_noise: the original's context, so the fit and its takes compare with the
    original as before); bed=False renders the sound alone (a kit's or a bank's take).
    Every render of a model fitted by 0.4 ends with sines.finish: a fade-in (FADE_IN_S, shorter when the sound's onset
    comes earlier: 0.25 ms at least) and a gain down to PEAK_DB dBFS when it peaks over that (no clipping; the
    reference clips peak at about -3). A model fitted before 0.4 (finished(): a modal, sines or hybrid model without
    the 'coherent' flag, the 0.3 bank's fit entries, or a spectral model without 'finish') renders as it did, sample
    for sample: the presets' takes stay the same"""
    return _render(model, seed, take, bed, **knobs)


FADE_IN_S = 0.002         # render(): the fade-in at the start of every render (it starts at zero, not mid-bed) ...
FADE_MIN_S = 0.00025      # ... shortened to 0.5 ms before the onset (the first 1 ms frame 40 dB under the loudest),
                          # to this at least, so a clip cut into its own attack keeps the attack
PEAK_DB = -1.0            # ... and the peak limit (a gain only): 18 of the films' 464 takes had peaked over 0 dBFS


def _render(model, seed=0, take=None, bed=None, final=True, space=True, **knobs):
    """render() with the internal switches: final=False leaves out the fade-in and the peak limit (renders the fit
    engine measures a model's level and post stage from), space=False the space layer (its dry)"""
    knobs = dict(knobs)
    take = float(knobs.get('vary') or 0.0) > 0 if take is None else bool(take)
    if kind_of(model) == 'morph':
        w = knobs.pop('morph', model.get('amount', 0.5))
        return _render(morph(model['a'], model['b'], w), seed, take, bed, final, space, **knobs)
    if knobs.get('vary') and model.get('vary_scale') is not None:
        knobs['vary'] = float(knobs['vary']) * float(model['vary_scale'])
    knobs.setdefault('ring', take)
    y = _render_raw(model, seed, **knobs)
    k = kind_of(model)
    scale = {'spectral': lambda: float(knobs.get('length', 1.0)) / float(knobs.get('speed', 1.0)),
             'gesture': lambda: float(knobs.get('length', 1.0))}.get(k, lambda: 1.0)()
    tail = {'modal': lambda: float(knobs.get('size', 1.0)) / float(knobs.get('damping', 1.0)),
            'sines': lambda: float(knobs.get('stretch', 1.0)) * float(knobs.get('decay', 1.0)),
            'hybrid': lambda: float(knobs.get('decay', 1.0))}.get(k, lambda: 1.0)()
    post = model.get('post') or {}
    # gesture.render stretches a gesture with tones about their first strike (its onset stays): the post stage too
    anchor = float((model.get('tones') or {}).get('onset', 0.0)) if k == 'gesture' else 0.0
    y = apply_post(y, post, scale, tail, _gate_map(model, knobs) if post.get('gate') else None, anchor)
    if space and post.get('space'):                  # fit.space: the diffuse side, after the post stage
        from . import space as SP
        y = SP.render_space(post['space'], y, seed=seed, width=float(knobs.get('width', 1.0)), ring=take,
                            limit=False)
    if post.get('bed') and bed is not False:
        b = bed_noise(post['bed'], len(y), seed)
        y = y + (b[:, None] if y.ndim == 2 else b)
    y = _dc_block(y) if take else y
    if not final or not finished(model):
        return y
    from .sines import finish
    fade = float(np.clip(Q.onset_index(Q.mono(y), -40.0) / SR - 0.0005, FADE_MIN_S, FADE_IN_S))
    return finish(y, SR, fade_s=fade, peak_db=PEAK_DB)


def finished(model):
    """does a render of this model end with the 0.4 finish (render())? Every fit made by 0.4 does (its modal and sines
    models carry 'coherent', its spectral models 'finish', a morph of them the flag of either side; gesture models
    are new in 0.4); a modal, sines, hybrid or spectral model fitted before 0.4 does not"""
    k = kind_of(model)
    if k in ('modal', 'sines'):
        return bool(model.get('coherent'))
    if k == 'hybrid':
        return any((L.get('model') or {}).get('coherent') for L in model['layers'])
    if k == 'spectral':
        return bool(model.get('finish'))
    return True


def _gate_map(model, knobs):
    """the gate's time map under a model's knobs ((t, dB) -> (t, dB)), None when the knobs leave time alone: a sines
    stretch scales time about the first onset, a spectral length / speed scales it; a modal spacing moves every
    strike, and the cuts after a strike move with it (decays do not move a gate: a gate is the designer's cut, a
    longer ring is cut at the same place)"""
    k = kind_of(model)
    if k == 'sines' and float(knobs.get('stretch', 1.0)) != 1.0:
        on, st = float(model.get('onset', 0.0)), float(knobs['stretch'])
        return lambda t, g: (on + (t - on) * st, g)
    if k == 'spectral':
        sc = float(knobs.get('length', 1.0)) / float(knobs.get('speed', 1.0))
        return None if sc == 1.0 else (lambda t, g: (t * sc, g))
    if k == 'gesture' and float(knobs.get('length', 1.0)) != 1.0:     # the whole gesture about the tones' first strike
        on, st = float((model.get('tones') or {}).get('onset', 0.0)), float(knobs['length'])
        return lambda t, g: (on + (t - on) * st, g)
    if k == 'modal' and float(knobs.get('spacing', 1.0)) != 1.0 and len(model['strikes']) > 1:
        t0 = np.array([s['t'] for s in model['strikes']], float)
        tn = [t0[0]]
        for i in range(1, len(t0)):                  # modal._plan's strike times
            tn.append(tn[-1] + max(0.003, (t0[i] - t0[i - 1]) * float(knobs['spacing'])))
        tn = np.asarray(tn)

        def f(t, g):
            t, g = np.asarray(t, float), np.asarray(g, float)
            if not len(t):
                return t, g
            G = lambda u: float(np.interp(u, t, g, left=0.0, right=float(g[-1])))
            j = np.clip(np.searchsorted(t0, t, side='right') - 1, 0, len(t0) - 1)
            ot, og = [], []
            for i in range(len(t0)):
                d, sel = tn[i] - t0[i], j == i
                if i == len(t0) - 1:
                    ot.append(t[sel] + d)
                    og.append(g[sel])
                    break
                s, ns = max(t0[i + 1] - GATE_MAP_S, t0[i]), max(tn[i + 1] - GATE_MAP_S, tn[i])
                a = sel & (t < s) & (t + d < ns)     # the cut after strike i moves with it, up to the new split
                ot.append(t[a] + d)
                og.append(g[a])
                e = min(s + d, ns)                   # held at the split's value across a longer gap; a shorter gap
                ot.append([e - 1e-6, ns])            # ends the cut at its value there
                og.append([G(e - d), G(s)])
                b = sel & (t >= s)                   # the reopening before strike i + 1 moves with that strike
                ot.append(t[b] + (tn[i + 1] - t0[i + 1]))
                og.append(g[b])
            tt, gg = np.concatenate(ot), np.concatenate(og)
            ok = tt > np.r_[-np.inf, np.maximum.accumulate(tt)[:-1]]     # strictly increasing for np.interp
            return tt[ok], gg[ok]
        return f
    return None


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


def _anchor_times(A, B):
    """the paired anchor times a morph of two models of one kind warps time through: the strikes (modal), the onsets
    (sines), the loudness peak (spectral), between the start and the end of each"""
    k = kind_of(A)
    if k == 'modal':
        SA, SB = A['strikes'], B['strikes']
        n = min(len(SA), len(SB))
        return ([0.0] + [s['t'] for s in SA[:n]] + [A['length_s']], [0.0] + [s['t'] for s in SB[:n]] + [B['length_s']])
    if k == 'sines':
        oa = list(A.get('onsets') or [A.get('onset', 0.0)])
        ob = list(B.get('onsets') or [B.get('onset', 0.0)])
        n = min(len(oa), len(ob))
        return [0.0] + oa[:n] + [A['duration']], [0.0] + ob[:n] + [B['duration']]
    if k == 'spectral':
        def peak(M):
            E = 10 ** (np.asarray(M['env']['db'], float) / 10)
            return float(M['env']['t'][int(np.argmax(E.sum(1)))])
        return [0.0, peak(A), A['dur']], [0.0, peak(B), B['dur']]
    raise ValueError(f'a {k} model has no anchors of its own')


def _morph_maps(a, b, w):
    """(a -> morph, b -> morph): the time maps a morph of a and b puts its strikes, onsets or loudness peak on
    (_anchors through _anchor_times), where the post stage's gate follows them; a gesture: its tones' when both have
    tones of one kind, else its air's; a hybrid: its first deterministic layer both have, else its air's. None when
    there is none (the gates then mix on their own times)"""
    k = kind_of(a)
    if k in ('modal', 'sines', 'spectral'):
        return _anchors(*_anchor_times(a, b), w)[:2]
    if k == 'gesture':
        ta, tb = a.get('tones'), b.get('tones')
        if ta and tb and ta.get('kind') == tb.get('kind'):
            return _morph_maps(ta, tb, w)
        return _morph_maps(a['air'], b['air'], w)
    if k == 'hybrid':
        la = {L['role']: L['model'] for L in a['layers']}
        lb = {L['role']: L['model'] for L in b['layers']}
        for role in ('tone', 'body', 'air'):
            if role in la and role in lb and kind_of(la[role]) == kind_of(lb[role]):
                return _morph_maps(la[role], lb[role], w)
    return None


def _morph_modal(A, B, w):
    from .modal import _rdp
    SA, SB = A['strikes'], B['strikes']
    k = min(len(SA), len(SB))
    wa, wb, tm = _anchors(*_anchor_times(A, B), w)
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
    from .sines import morph_side
    side_t = morph_side(A.get('side_t'), B.get('side_t'), w, wa, wb)
    return {'kind': 'modal', 'version': max(A['version'], B['version']), 'sr': SR,
            'length_s': round(float(tm[-1]), 6), 'channels': max(A.get('channels', 1), B.get('channels', 1)),
            'fade_out_s': (1 - w) * float(A.get('fade_out_s', 0.0)) + w * float(B.get('fade_out_s', 0.0)),
            'strikes': strikes, 'sub': sub, 'noise': {'edges': list(na['edges']), 'h_erb': na['h_erb'], 'env': env},
            'side': side, **({'side_t': side_t} if side_t else {}),
            **({'coherent': True} if A.get('coherent') or B.get('coherent') else {}),
            'fit': {'modes': int(sum(len(s['modes']) for s in strikes))}}


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
    from .sines import morph_side, partial_freq, summary
    oa = list(A.get('onsets') or [A.get('onset', 0.0)])
    ob = list(B.get('onsets') or [B.get('onset', 0.0)])
    k = min(len(oa), len(ob))
    wa, wb, tm = _anchors(*_anchor_times(A, B), w)
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
         'side': morph_side(A.get('side'), B.get('side'), w, wa, wb),
         **({'coherent': True} if A.get('coherent') or B.get('coherent') else {}),
         'channels': max(int(A.get('channels', 1)), int(B.get('channels', 1))), 'analysis': {}}
    m['summary'] = summary(m)
    return m


def _morph_spectral(A, B, w):
    if A['bands'] != B['bands']:
        raise ValueError('spectral models with different bands cannot be morphed')
    wa, wb, tm = _anchors(*_anchor_times(A, B), w)
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
            'channels': max(int(A.get('channels', 1)), int(B.get('channels', 1))),
            **({'finish': True} if A.get('finish') or B.get('finish') else {})}


def _morph_gesture(A, B, w):
    """tones of one kind morph as sines or modal models do (tones only one side has, or tones of two kinds, which
    cannot be paired: the nearer side's, faded with its weight by gesture.gain_tones), the air as spectral models
    morph; the trajectory is measured again on the morphed air"""
    from .gesture import gain_tones
    from .spectral import trajectory
    ta, tb = A.get('tones'), B.get('tones')
    tones = None
    if ta and tb and ta.get('kind') == tb.get('kind'):
        tones = dict(_morph(ta, tb, w), layer='deterministic')
        if tones['kind'] == 'modal':
            tones['onset'] = round(float(tones['strikes'][0]['t']), 6) if tones.get('strikes') else 0.0
    else:
        if ta and tb:                                # sines tones against modal tones: the nearer side's
            ta, tb = (ta, None) if w < 0.5 else (None, tb)
        if (ta and w < 1) or (tb and w > 0):
            tones = gain_tones(ta, 20 * np.log10(1 - w)) if ta else gain_tones(tb, 20 * np.log10(w))
    air = _morph_spectral(A['air'], B['air'], w)
    two = max(int(A.get('channels', 1)), int(B.get('channels', 1)))
    if tones:
        tones['channels'] = two
    return {'kind': 'gesture', 'version': max(A['version'], B['version']), 'sr': SR, 'dur': air['dur'],
            'channels': two, 'source': None, 'tones': tones, 'air': air, 'traj': trajectory(air), 'fit': {}}


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


def _mix_post(pa, pb, w, maps=None):
    """the post stages of a morph's two sources mixed with weight w; maps: (a -> morph, b -> morph) time maps
    (_morph_maps), through which each gate's breakpoints move onto the morph's timeline before they mix (a cut sits
    where the morph's strikes and gaps are, not where its source's were)"""
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
        if pa['pre'].get('v') == 2 and pb['pre'].get('v') == 2:
            out['pre']['v'] = 2
        if pa['pre'].get('hp_hz') and pb['pre'].get('hp_hz'):
            out['pre']['hp_hz'] = max(float(pa['pre']['hp_hz']), float(pb['pre']['hp_hz']))
        aa = np.asarray(pa['pre'].get('attack_db') or [], float)
        ab = np.asarray(pb['pre'].get('attack_db') or [], float)
        if len(aa) and len(ab):
            na = max(2, int(round((1 - w) * len(aa) + w * len(ab))))
            u = np.linspace(0, 1, na)
            mix = ((1 - w) * np.interp(u, np.linspace(0, 1, len(aa)), aa)
                   + w * np.interp(u, np.linspace(0, 1, len(ab)), ab))
            out['pre']['attack_db'] = [round(float(v), 1) for v in mix]
    ga, gb = (pa.get('gate') or {}).get('pts'), (pb.get('gate') or {}).get('pts')
    if ga or gb:                                   # the gates: cuts mix in dB (one side's alone fades with its weight)
        A = np.asarray(ga or [[0.0, 0.0]], float).reshape(-1, 2)
        B = np.asarray(gb or [[0.0, 0.0]], float).reshape(-1, 2)
        if maps is not None:                       # each source's cuts onto the morph's timeline
            A[:, 0], B[:, 0] = maps[0](A[:, 0]), maps[1](B[:, 0])
        t = np.unique(np.r_[A[:, 0], B[:, 0]])
        g = ((1 - w) * np.interp(t, A[:, 0], A[:, 1], left=0.0) + w * np.interp(t, B[:, 0], B[:, 1], left=0.0))
        out['gate'] = {'pts': [[round(float(a), 5), round(float(b), 1)] for a, b in zip(t, g)]}
    for key in ('space', 'bed'):                   # the space layer and the bed: the nearer source's
        src = (pa if w < 0.5 else pb).get(key)
        if src:
            out[key] = copy.deepcopy(src)
    return out


def morph(a, b, amount=0.5):
    """a model between two fits (see _morph); the post stages (EQ, gain) interpolate too"""
    w = float(np.clip(amount, 0.0, 1.0))
    if w in (0.0, 1.0) and kind_of(a) == kind_of(b) != 'morph':   # the endpoints are the sources, exactly
        return copy.deepcopy({k: v for k, v in (a if w == 0.0 else b).items() if k != 'fit_engine'})
    m = _morph(a, b, amount)
    p = _mix_post(a.get('post'), b.get('post'), w, _morph_maps(a, b, w) if _gated(a, b) else None)
    if p:
        m['post'] = p
    if a.get('vary_scale') is not None or b.get('vary_scale') is not None:
        m['vary_scale'] = round((1 - w) * float(a.get('vary_scale', 1.0)) + w * float(b.get('vary_scale', 1.0)), 4)
    return m


def _gated(a, b):
    return any(((m.get('post') or {}).get('gate') or {}).get('pts') for m in (a, b))


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
    if ka == 'gesture':
        return _morph_gesture(a, b, w)
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
    if k == 'gesture':
        return head + F['gesture'].summary(model)
    if k == 'morph':
        ind = lambda t: t.replace('\n', '\n     ')
        return (f"morph {model['amount']:.2f} between two {kind_of(model['a'])} models:\n  a: "
                + ind(summary(model['a'])) + '\n  b: ' + ind(summary(model['b'])))
    lines = [head + f"hybrid of {len(model['layers'])} layers ({model['length_s'] * 1000:.0f} ms):"]
    for L in model['layers']:
        lines.append(f"  {L['role']} ({kind_of(L['model'])}{', deterministic part' if L['model'].get('layer') else ''}"
                     f", {L.get('gain_db', 0.0):+.1f} dB): " + summary(L['model']).replace('\n', '\n    '))
    return '\n'.join(lines)
