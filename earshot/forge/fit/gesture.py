"""Gesture fitter: a long designed gesture (whoosh, swish, riser, reverse swell, camera move, glitch run) -> a tones +
air model -> fresh takes of it.

    from earshot.forge.fit import gesture
    m = gesture.fit('whoosh.wav')                 # or a 48 kHz array; tones=<a sines fit of the same x> reuses it
    m = gesture.fit(x, tones=modal_fit)           # the tones from a modal fit instead (modes, contact, sub)
    y = gesture.render(m, seed=3)                 # float64 at 48 kHz, (n, 2) for a stereo source: new air, same tones
    y = gesture.render(m, seed=4, length=1.2, sweep=1.3, doppler=2.0, motion=0.6, proximity=0.5, vary=0.2)
    print(gesture.summary(m))
    a = gesture.air(x, y_det)                     # the forced air layer of any deterministic render y_det: a detail
                                                  # noise model of what y_det leaves in x (magnitude domain)

Why a model of its own: the films' whooshes, swells, risers and camera moves are not noise. They are designed: steady
and gliding tonal lines (a reverse cymbal's partials, a riser's whistle, the bed under a camera move), short tonal
grains, clicks, stutter gates and a noise body, panned and moving. Fitted alone, the sines fitter turns all of it into
60-180 partials (the noise as partials: takes ring metallic) and the noise model into a smooth wash (CLAP 0.44-0.73
on them). The hybrid of the two subtracts the partials' render from the waveform, which leaves every tone it does not
cancel in phase in the residual, where the noise layer smears it over its band. The gesture model splits the sound in
the magnitude domain instead:

  tones    the deterministic part of a sines fit (partials with their glides, strike atoms and, for a stereo source,
           the fitted side: panned and decorrelated per band), or of a modal fit (modes, contact modes, sub wave, the
           side over time); dropped when it carries under TONE_MIN of the energy. tones['onset'] is the first strike
  air      a detail noise model (spectral.fit over spectral.DETAIL and AIR_FIT: a 5 ms gesture kept to 0.5 dB with
           keyframes that follow cells down to 70 dB under the loudest, so gaps and stutters stay dark; no denoise, up
           to 100 tonal ridges, the texture statistics) of the RESIDUAL: the recording through a soft STFT
           mask that takes the tones' magnitude out, |R|^2 = max(|X|^2 - ALPHA |T|^2, 0) per bin with the recording's
           phase. ALPHA 3 over-subtracts: a partial's skirt and its estimation error stay out of the air, where the
           ridges would re-make them as jittered whistles. The band envelopes carry the sweep, the per-group L/R level
           difference and correlation per keyframe the pan and width over time, the texture (flutter, crackle) the
           grain statistics: a take is fresh air around the same tones, not a re-jittered bank of noise partials, so
           takes do not ring
  traj     the air's trajectory (spectral.trajectory: level, centroid, brightness, pan and width per keyframe, the peak
           and its 20 dB rise / fall), what the motion knobs read
  fit      {'alpha', 'tones_kind', 'tone_share', 'partials' (the tones' partials or modes), 'air_keys', 'ridges',
           'seconds'}

Measured on the 7 dev gesture clips of the replication benchmark (camera_air, glitch, reverse_swell x2, riser, swish,
whoosh_fast; eval/forge/replicate.py's dev split, scored the way auto.fit(clap=True) scores a candidate: bed stripped,
the post stage with the CLAP judge, seed 0 with the bed back on) against the 0.4 fit engine's choice without this
kind: the gesture scores lower on 4 of the 7, with a higher CLAP on each (camera move 0.855 -> 0.880, reverse swells
0.858 -> 0.935 and 0.808 -> 0.842, riser 0.871 -> 0.921), and its takes pass the take check. It loses on the short or
bursty ones: a 0.24 s swish (CLAP 0.845 against modal's 0.884), a grainy whoosh (0.795 against the sines fit's
0.796) and a glitch run of hard bursts (0.905 against 0.885, but a blunter first onset on the 0.125 ms sharpness
measure costs it the score). ALPHA 3, the 5 ms gesture and the 70 dB keyframe weights against the first version's 1,
10 ms and 50 dB: mean score 10.21 -> 9.47, CLAP 0.857 -> 0.875. Modal tones scored worse on 6 of the 7: an option,
not the default. In the fit engine's candidate plans (auto.plan: every sound under 1 s, and second in the long
gesture / mixed / glide / ringing plans) the gesture model won 38 of the 78 dev clips of the replication benchmark,
short ones too (keys, pops, ticks: tones plus a little air beat the sines fit's residual noise).

The fit engine's use of this module (fit/auto.py), kept stable:
  KIND, VERSION, KNOBS (auto.KNOBS' format: name -> (search range spec, identity value, what it does); without tones
      tone_db and vary do nothing and auto.knobs_of leaves them out)
  fit(x, name=None, tones=None, fast=False, **opts) -> model. tones: the sines (or modal) candidate of the same x
      (None: fit a sines model here, tones_kind='modal' a modal one; False: air only); fast=True for tests and
      previews. model['channels'], ['dur'], ['tones'] (a deterministic sines or modal model, or None), ['air'] (a
      spectral_noise model), ['traj']
  render(model, seed=0, **knobs) -> float64 (n, 2) or (n,); knobs: KNOBS plus gain_db, stereo, ring (a take). The
      length knob stretches about tones['onset'] (auto._gate_map moves a post-stage gate with it)
  air(x, y_det=None, alpha=ALPHA, fast=False, **spectral_fit_opts) -> a spectral_noise model (the forced air layer)
  gain_tones(det, gain_db) (a morph fading the tones only one side has), summary, save, load, check
auto.py: KINDS, kind_of, knobs_of, channels_of, fit_kind('gesture', tones=...), _cand_fit('gesture': the tones of the
sines candidate, so 'gesture' comes after 'sines' in ALL and in every plan), _render_raw, the post stage (a NOISE_KINDS
member: the attack clamp high-passed and kept only where it helps; the length knob's time scale), _gate_map,
_morph_gesture (tones of two kinds, sines and modal, cannot be interpolated: the nearer side's, faded with gain_tones),
summary; knobs.FIT_LENGTH maps a bank entry's dur knob to `length`.

render(model, seed, **knobs) knobs (identity values re-render the fit; KNOBS has the search ranges):
    length (time-stretch: the tones about their first strike, the air with them), sweep (x the air's centroid
    excursion), doppler (semitones of a pass-by pitch bend of the whole sound, a time warp turning at the air's peak),
    proximity (-1..1: the air's quieter keyframes duller and quieter still), pan (x the air's L/R level differences),
    motion (-1..1: a pass-by pan of the whole sound), width (x the tones' side and the air's width), roughness (the
    air's texture depth), tone_db / air_db (layer levels), vary (seeded jitter of the tones; the air is new noise on
    every seed anyway), gain_db, stereo (None = the source's channels), ring (a take: partials ring out with their
    release, the air decays 60 dB over at least RING_S past the clip's end, and the end gets a 10 ms fade instead of
    stopping on the recording's last sample).
Pure numpy / scipy. Deterministic: render(model, seed, knobs) is a pure function.
"""
import copy
import json
import os

import numpy as np

KIND = 'gesture'
VERSION = 1
SR = 48000
ALPHA = 3.0               # mask over-subtraction: |R|^2 = max(|X|^2 - ALPHA |T|^2, 0)
AIR_FIT = {'gesture_ms': 5.0, 'quiet_db': 70.0,   # the air's spectral.fit options over spectral.DETAIL: a 5 ms
           'grain_norm_ms': 5.0, 'grain_halves': True}   # gesture, keyframes that follow cells down to 70 dB under the
                          # loudest (the gaps stay dark); a grain texture levelled over 5 ms (>= 20 clicks) and only
                          # when stationary (0.4 round C2: short noise-led fits, keys and ticks fitted as air, had takes
                          # whose CLAP spread 0.84-0.92 from seed to seed because one draw put its loudest grains where
                          # the sound is quiet; on 20 dev clips the takes' median gap under seed 0 0.046 -> 0.031, 'CLAP
                          # drops > 0.06' 7 -> 5, grades 9/6/5/0 -> 13/3/4/0)
MASK_FFT = 1024           # the mask's STFT (hop a quarter)
TONE_MIN = 0.05           # tones under this share of the energy are dropped (a pure noise gesture)
MIN_PARTIAL_MS = 0.0      # partials shorter than this (birth to release end) are left to the air
FADE_S = 0.010            # a take's end fade (ring=True)
RING_S = 0.06             # a take's air decays 60 dB over at least this past the clip's end (ring=True)

# knob -> (search range spec for a forge SPACE, identity value, what it does): the format of auto.KNOBS
KNOBS = {
    'length': ((0.8, 1.25, 'log'), 1.0, 'time-stretch of the gesture (tones about their first strike)'),
    'sweep': ((0.7, 1.43, 'log'), 1.0, "x the air's spectral centroid excursion"),
    'doppler': ((-2.0, 2.0), 0.0, 'semitones of a pass-by pitch bend turning at the peak (+ = approaching higher)'),
    'proximity': ((-0.5, 1.0), 0.0, 'quieter keyframes duller and quieter still (closer feel above 0)'),
    'pan': ((0.5, 1.5), 1.0, "x the air's L/R level differences"),
    'motion': ((-0.6, 0.6), 0.0, 'a pass-by pan of the whole sound (-1 left .. +1 right)'),
    'width': ((0.5, 1.5), 1.0, "x the tones' side and the air's width"),
    'roughness': ((0.5, 1.5), 1.0, "the air's texture depth (flutter, crackle)"),
    'tone_db': ((-6.0, 6.0), 0.0, 'level of the tones'),
    'air_db': ((-6.0, 6.0), 0.0, 'level of the air'),
    'vary': ((0.0, 0.4), 0.0, 'seeded jitter of the tones: new takes of the same gesture'),
}


def _load(x):
    from . import quality as Q
    X = Q.load(x)
    if X.ndim == 1:
        X = np.stack([X, X], 1)
    elif X.shape[1] == 1:
        X = np.repeat(X, 2, 1)
    return np.asarray(X[:, :2], np.float64)


def _two(y, n):
    """y (mono or stereo, any length) as (n, 2)"""
    y = np.asarray(y, np.float64)
    if y.ndim == 1:
        y = np.stack([y, y], 1)
    out = np.zeros((n, 2))
    k = min(n, len(y))
    out[:k] = y[:k, :2]
    return out


def check(model):
    if not isinstance(model, dict) or model.get('kind') != KIND:
        raise ValueError(f"not a {KIND} model (kind={model.get('kind') if isinstance(model, dict) else None!r})")
    if int(model.get('version', 0)) > VERSION:
        raise ValueError(f"model version {model.get('version')} is newer than this renderer ({VERSION})")


# ---------------------------------------------------------------------------------------------------- split


def residual(X, Y, alpha=ALPHA, n_fft=MASK_FFT):
    """X minus Y in the STFT magnitude domain, per channel: X through the soft mask sqrt(max(1 - alpha |Y|^2 / |X|^2,
    0)) with X's phase (Hann, hop n_fft / 4). Where Y explains a bin the bin goes; where Y is quiet X stays as it is;
    nothing Y has and X has not is ever added (a waveform subtraction adds every tone it does not cancel in phase)"""
    from scipy import signal
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    one = X.ndim == 1
    X2 = X[:, None] if one else X
    Y2 = (Y[:, None] if Y.ndim == 1 else Y)
    n = len(X2)
    kw = dict(fs=SR, window='hann', nperseg=n_fft, noverlap=n_fft - n_fft // 4)
    out = np.zeros_like(X2)
    for c in range(X2.shape[1]):
        _, _, ZX = signal.stft(X2[:, c], **kw)
        yc = Y2[:n, min(c, Y2.shape[1] - 1)]
        _, _, ZY = signal.stft(np.pad(yc, (0, n - len(yc))), **kw)
        g = np.sqrt(np.clip(1.0 - alpha * np.abs(ZY) ** 2 / (np.abs(ZX) ** 2 + 1e-30), 0.0, 1.0))
        _, r = signal.istft(ZX * g, **kw)
        out[:min(n, len(r)), c] = r[:n]
    return out[:, 0] if one else out


def _deterministic(ms, two, min_ms=0.0):
    """the tones of a sines or modal model, without their noise: partials + strike atoms (sines; partials that last
    under min_ms, birth to release end, are left to the air) or modes + contact + sub (modal); the fitted side kept for
    a stereo source"""
    m = copy.deepcopy({k: v for k, v in ms.items() if k not in ('post', 'fit_engine')})
    m['layer'] = 'deterministic'
    m['channels'] = 2 if two else 1
    if m.get('kind') == 'modal':
        L = float(m['length_s'])
        m['noise'] = dict(m['noise'], env=[[[0.0, -240.0], [L, -240.0]] for _ in m['noise']['env']])
        m['onset'] = round(float(m['strikes'][0]['t']), 6) if m.get('strikes') else 0.0
        if not two:
            m['side'] = None
            m.pop('side_t', None)
        return m
    if min_ms > 0:
        m['partials'] = [p for p in m['partials'] if (p['t1'] + p['rel'] - p['t0']) * 1000.0 >= min_ms]
    m['residual'] = {'bands': [1000.0], 'times': [0.0, float(m['duration'])], 'db': [[-150.0, -150.0]]}
    if not two:
        m['side'] = None
    return m


def _tones(det, seed, two, n=None, stretch=1.0, vary=0.0, width=1.0, gain_db=0.0, ring=False):
    """a render of the tones: (n, 2) when n is given, else as the fitter renders it. stretch: time factor about the
    first strike (a modal model: strike spacing x stretch, decays x stretch)"""
    if det.get('kind') == 'modal':
        from . import modal
        s = max(float(stretch), 1e-3)
        y = modal.render(det, seed, spacing=s, damping=1.0 / s, vary=float(vary), width=float(width),
                         gain=float(gain_db), ring=bool(ring), stereo=bool(two))
    else:
        from . import sines
        y = sines.render(det, seed, knobs=dict(stretch=float(stretch), vary=float(vary), width=float(width),
                                               gain_db=float(gain_db), ring=bool(ring), stereo=bool(two)))
    y = np.asarray(y, np.float64)
    return y if n is None else _two(y, n)


def gain_tones(det, gain_db):
    """a copy of a gesture's tones (a deterministic sines or modal model) gain_db louder: sines partial levels and
    strike atoms; modal modes, contact modes and the sub wave (a morph fades the tones only one side has with it)"""
    t = copy.deepcopy(det)
    g = 10 ** (float(gain_db) / 20)
    if t.get('kind') == 'modal':
        for st in t.get('strikes') or ():
            for key in ('modes', 'attack'):
                rows = np.asarray(st.get(key) or [], float)
                if rows.size:
                    nested = rows.ndim == 2
                    rows = rows.reshape(-1, 4)
                    rows[:, 2] *= g
                    st[key] = [[float(v) for v in r] for r in rows] if nested else [float(v) for v in rows.ravel()]
        if t.get('sub'):
            t['sub'] = dict(t['sub'], x=[float(v) * g for v in t['sub']['x']])
        return t
    for p in t['partials']:
        p['amp'] = [[a, v + float(gain_db)] for a, v in p['amp']]
    for tr in t.get('transients', []):
        tr['atoms'] = [[f, tau, a * g, ph] for f, tau, a, ph in tr['atoms']]
    return t


def _n_tones(det):
    """how many tones a deterministic model holds: partials (sines) or modes over every strike (modal)"""
    if not det:
        return 0
    if det.get('kind') == 'modal':
        return int(sum(np.asarray(s['modes'], float).reshape(-1, 4).shape[0] for s in det.get('strikes') or ()))
    return len(det['partials'])


DETAIL_FAST = dict(iters=1, texture=False, seeds=(1,))


def _air_tail(am, tail_s):
    """a copy of an air model that goes on for tail_s past its end, falling 60 dB (linear in dB) from its last
    keyframe: a take's air decays under the ringing tones"""
    am = dict(am)
    t_end = float(am['dur']) + float(tail_s)
    env = am['env']
    am['env'] = {'t': list(env['t']) + [round(t_end, 4)], 'db': list(env['db']) + [[v - 60.0 for v in env['db'][-1]]]}
    if am.get('stereo'):
        st = am['stereo']
        am['stereo'] = dict(st, ild=list(st['ild']) + [st['ild'][-1]], icc=list(st['icc']) + [st['icc'][-1]])
    am['dur'] = round(t_end, 4)
    return am


def air(x, y_det=None, alpha=ALPHA, fast=False, name=None, mask_fft=MASK_FFT, **kw):
    """the air layer of a gesture or a hybrid: a detail noise model (spectral.DETAIL; kw override it) of x, or of what
    the deterministic render y_det leaves in x (residual(): the magnitude domain, never a waveform subtraction). The
    forced air layer of a hybrid: fitted even when the deterministic layers explain almost all of x (the air under a
    tone is what CLAP and ears miss first)"""
    from . import spectral
    X = _load(x)
    R = X if y_det is None else residual(X, _two(y_det, len(X)), alpha, mask_fft)
    opts = dict(spectral.DETAIL, **AIR_FIT)
    if fast:
        opts.update(DETAIL_FAST)
    opts.update(kw)
    m = spectral.fit(R, name=name, **opts)
    m['channels'] = 1 if np.allclose(X[:, 0], X[:, 1]) else 2
    return m


# ------------------------------------------------------------------------------------------------------ fit


def fit(x, name=None, tones=None, alpha=ALPHA, tone_min=TONE_MIN, min_partial_ms=MIN_PARTIAL_MS, fast=False,
        tones_kind='sines', verbose=False, **air_kw):
    """a gesture model of x (path or 48 kHz array, mono or (n, 2)). tones: a sines or modal model of the same x to take
    the tones from (the fit engine passes its candidate of that kind), None = fit one here (tones_kind 'sines' or
    'modal'), False = air only. Partials shorter than min_partial_ms go to the air. air_kw: options for the air's
    spectral.fit (over AIR_FIT, over spectral.DETAIL)"""
    import time
    from . import spectral
    if name is None and isinstance(x, (str, os.PathLike)):
        name = os.path.basename(str(x))
    t0 = time.time()
    X = _load(x)
    n = len(X)
    two = not np.allclose(X[:, 0], X[:, 1])
    e_x = float((X ** 2).sum()) + 1e-30
    det, share, yt = None, 0.0, None
    if tones is not False:
        ms = tones if isinstance(tones, dict) else _fit_tones(X if two else X[:, 0], tones_kind, fast)
        det = _deterministic(ms, two, min_partial_ms)
        yt = _tones(det, 0, two, n)
        share = float((yt ** 2).sum()) / e_x
        if share < tone_min:
            det, yt = None, None
    t1 = time.time()
    am = air(X, yt, alpha, fast=fast, name=name, **air_kw)
    m = {'kind': KIND, 'version': VERSION, 'sr': SR, 'dur': round(n / SR, 6), 'channels': 2 if two else 1,
         'source': name, 'tones': det, 'air': am, 'traj': spectral.trajectory(am),
         'fit': {'alpha': alpha, 'tones_kind': det.get('kind') if det else None, 'tone_share': round(share, 4),
                 'partials': _n_tones(det), 'air_keys': len(am['env']['t']), 'ridges': len(am.get('ridges') or ()),
                 'seconds': {'tones': round(t1 - t0, 2), 'air': round(time.time() - t1, 2)}}}
    if verbose:
        print('  ' + summary(m))
    return m


def _fit_tones(x, kind, fast):
    if kind == 'modal':
        from . import modal
        return modal.fit(x, **({'passes': 1} if fast else {}))
    if kind != 'sines':
        raise ValueError(f"tones_kind {kind!r}: 'sines' or 'modal'")
    from . import sines
    return sines.fit(x, **({'windows': (1024,)} if fast else {}))


# --------------------------------------------------------------------------------------------------- render


def render(model, seed=0, length=1.0, sweep=1.0, doppler=0.0, proximity=0.0, pan=1.0, motion=0.0, width=1.0,
           roughness=1.0, tone_db=0.0, air_db=0.0, vary=0.0, gain_db=0.0, stereo=None, ring=False):
    """the sound of a gesture model (float64, 48 kHz): (n, 2) for a stereo source (or stereo=True), else (n,). See the
    module doc for the knobs; identity values re-render the fit (seed 0 = the fit as measured)"""
    from . import spectral
    check(model)
    two = (model.get('channels', 1) == 2) if stereo is None else bool(stereo)
    s = max(float(length), 1e-3)
    n = max(64, int(round(model['dur'] * s * SR)))
    det = model.get('tones')
    t = None
    if det:
        t = _tones(det, seed, two, None, stretch=s, vary=float(vary), width=float(width), gain_db=float(tone_db),
                   ring=bool(ring))
    am = model['air']
    if ring:                 # a take: the air decays under the ringing tones instead of stopping on the clip's end
        am = _air_tail(am, max(RING_S, (len(t) / SR / s - model['dur']) if t is not None else 0.0))
    a = np.asarray(spectral.render(am, (int(seed) * 7 + 101) & 0x7FFFFFFF, length=s, sweep=sweep,
                                   roughness=roughness, width=width, pan=pan, proximity=proximity, gain_db=air_db,
                                   limit=False), np.float64)
    if t is not None:
        # the tones stretch about their first strike, the air from 0: move the air onto the tones
        sh = int(round(float(det.get('onset', 0.0)) * (1.0 - s) * SR))
        a = np.concatenate([np.zeros((sh, 2)), a]) if sh > 0 else a[-sh:]
        m = max(n, len(t), len(a)) if ring else n
        y = _two(a, m) + _two(t, m)
    else:
        y = _two(a, max(n, len(a)) if ring else n)
    if doppler or motion:
        tr = model['traj']
        t_pk, tau = tr['peak_s'] * s, spectral._motion_tau(tr, s)
        y = spectral.pass_pan(spectral.bend(y, t_pk, tau, doppler), t_pk, tau, motion)
    if gain_db:
        y = y * 10 ** (float(gain_db) / 20)
    if ring:
        k = min(len(y) // 3, int(FADE_S * SR))
        if k > 1:
            y[-k:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, k)))[:, None]
    return y if two else y.mean(1)


# ---------------------------------------------------------------------------------------------- words, io


def summary(model):
    """one paragraph: length, tones, air, trajectory, size"""
    from . import spectral
    check(model)
    det = model.get('tones')
    f = model.get('fit') or {}
    tr = model['traj']
    lv = np.asarray(tr['level_db'])
    pk = int(np.argmax(lv))
    w = [f"{model.get('source') or 'gesture'}: {model['dur']:.2f} s gesture"]
    if det and det.get('kind') == 'modal':
        fr = [float(v) for st in det['strikes'] for v in np.asarray(st['modes'], float).reshape(-1, 4)[:, 0]] or [0.0]
        w.append(f"tones: modal, {len(det['strikes'])} strike(s), {len(fr)} modes {min(fr):.0f}-{max(fr):.0f} Hz "
                 f"({100 * f.get('tone_share', 0.0):.0f} % of the energy)")
    elif det:
        fr = [p['freq'][0][1] for p in det['partials']] or [0.0]
        w.append(f"tones: {len(det['partials'])} partials {min(fr):.0f}-{max(fr):.0f} Hz "
                 f"({100 * f.get('tone_share', 0.0):.0f} % of the energy)")
    else:
        w.append('no tones (air only)')
    w.append('air: ' + spectral.summary(model['air']).split(': ', 1)[-1])
    w.append(f"peak at {tr['peak_s']:.2f} s (rise {1000 * tr['rise_s']:.0f} ms, fall {1000 * tr['fall_s']:.0f} ms), "
             f"centroid {tr['centroid_hz'][0]:.0f} -> {tr['centroid_hz'][pk]:.0f} -> {tr['centroid_hz'][-1]:.0f} Hz, "
             f"pan {min(tr['pan_db']):+.1f}..{max(tr['pan_db']):+.1f} dB, width {np.median(tr['width']):.2f}")
    w.append(f"{len(json.dumps(model)) / 1024:.0f} kB JSON")
    return '; '.join(w)


def save(model, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(model, fh, separators=(',', ':'))
    return path


def load(path):
    with open(path, encoding='utf-8') as fh:
        m = json.load(fh)
    check(m)
    return m
