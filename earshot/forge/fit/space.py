"""Space: the stereo picture of a sound, and a fitted diffuse tail layer (a small room) for recordings whose tail is
decorrelated.

    from earshot.forge.fit import space
    sp = space.fit_space(x, 48000)               # None when the tail is not decorrelated (L/R correlation > 0.7)
    sp = space.fit_space(x, 48000, dry=y0)       # only the diffuse side the dry render y0 lacks
    y = space.render_space(sp, y0, seed=0)       # y0 + the wet layer: (n, 2)
    space.tail_corr(x)                           # the tail's L/R correlation (from the peak + 30 ms)
    P = space.side_powers(x)                     # per band and frame: mid, side and cross power

Why: a reverb tail is two different signals in L and R (a diffuse field: tail L/R correlation 0.1-0.5 in the films'
dings, glass, bells, swells), while a model of a sound's mid rings the same signal in both channels (a render's tail
correlation 0.9 and more). The wet layer is the dry render convolved with seeded noise impulse responses: a SIDE part
(L + w, R - w), which leaves the mono sum alone (so every mono measure and CLAP), and a MID part (both channels), fitted
only when the dry's own mono tail falls short of the recording's. A fit of the whole recording already rings its tail
in mono, so the side part is usually all there is: then the width and the tail's L/R correlation change and nothing
else. Convolving the dry render (not adding band noise) keeps a ringing tail tonal in the side too: the reverb of a bell
is the bell's partials smeared, not hiss.

Stereo measures (shared with fit/sines.py and fit/quality.py), on mid M = (L + R) / 2 and side S = (L - R) / 2:
  side_powers  per octave band (triangles on log frequency that sum to one, sines.band_masks) and STFT frame: PM, PS
               and C = Re(S M*). C / PM is the side's part coherent with the mid (a pan: L = M (1 + g), R = M (1 - g));
               PS - C^2 / PM what is left, a side uncorrelated with the mid (diffuse; rendered as the mid's quadrature
               or as this module's wet layer)
  active_span / tail_span   onset to the last 1 ms frame above the clip's floor (replicate.props' definition: 3 dB
               over the lower of the pre-roll level and the level before the clip's end, kept 12-45 dB under the
               peak); the tail starts TAIL_START_S after the loudest 10 ms
  tail_corr    L/R correlation (zero lag) over the tail span

Model (plain JSON): {'kind': 'space', 'version': 1, 'predelay': s, 'build': s (the impulse responses' rise),
  'bands': [Hz octave centres], 'rt60': [s per band], 'side_db': [dB per band re a unit-energy band IR; -150 = off],
  'mid_db': [the same for the mid part] (all -150: side only), 'tail_corr': the recording's tail correlation,
  'fit': {'need_db', 'mid_need_db', 'iters', 'dry', 'side_offset_db', 'tail_corr_dry', 'tail_corr_wet'}}

fit_space (analysis by synthesis), over the tail span, per band: the side part must supply the recording's diffuse side
(its side uncorrelated with its mid) minus what the dry render already has (its own diffuse side: a sines model's side,
a modal side); the mid part the recording's mid energy minus the dry's, where the dry lacks at least MID_NEED_DB of it.
No layer when the dry's tail correlation is within 0.1 of the recording's already, or when the side need is NEED_DB
under the recording's own diffuse tail. Starting rt60s come from the decay of the recording's diffuse side per band;
then unit-gain wet renders of the dry (the recording's own mid when no dry is given) are measured with the same
analysis and each band's levels and rt60 corrected toward the needs (`iters` passes: the dry's own decay lengthens the
convolved tail; rt60 at most RT_SPAN x the tail span). The mid part is dropped if it makes the mono mel distance
(quality.mel_distance) worse. Last, the side part's overall level moves (+-OFFSET_DB, closed form over the tail, the
products pooled over FIT_SEEDS renders) to where the tail correlation is closest to the recording's; the layer is kept
only if that lands closer than the dry's. The wet side is made orthogonal to the dry's side and mid per band
(_orthogonal), so a random wet never cancels the dry's own width.
On the 29 wide-category dev clips of the replication benchmark it was fitted on 3 winners (a modal, two hybrids: tail
correlation off by 0.21 median -> 0.02) and on none of the sines winners (their side already matches). Seeds spread
around the fitted correlation by +-0.02 on a 230 ms bell tail but +-0.25 on 40-70 ms tails (their correlation is
itself that uncertain; the mean over seeds still lands within 0.15 of the recording's, the dry's was 0.23-0.29 off).
render_space(model, dry, seed, width, ring): the dry (mono or (n, 2)) plus the wet; width scales the side part;
ring=True lets the wet ring out past the dry's end (a take). The result is peak-limited to -1 dBFS (sines.finish).
Deterministic: render_space(model, dry, seed) is a pure function of its arguments. Pure numpy / scipy.
"""
import numpy as np
from scipy import fft as sfft
from scipy import signal

from .sines import band_masks, finish

SR = 48000
KIND = 'space'
VERSION = 1
BANDS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)   # octave centres
NFFT = 1024                # stereo analysis frames (Hann), every HOP samples
HOP = 240
TAIL_START_S = 0.03        # the tail starts this long after the loudest 10 ms (replicate.props)
TAIL_CORR_MAX = 0.7        # fit_space models a tail only when the recording's tail L/R correlation is at most this
MAX_RT = 3.0               # s: the longest rt60 a band may get (and the impulse response's length cap)
RT_SPAN = 4.0              # and at most this many times the recording's tail span
OFFSET_DB = 12.0           # the side's overall level may move this far (dB) onto the recording's tail correlation
FLAT_ITERS = 4             # passes that flatten each band of the impulse responses (impulse())
FIT_SEEDS = (0, 1, 2, 3)   # renders whose tail products are pooled when the side level is set
NEED_DB = -15.0            # no layer when what is missing is this far under the recording's own tail side energy
MID_NEED_DB = -10.0        # a mid part only where the dry lacks at least this share of the tail's mid energy (dB)


# ================================================================================================ helpers
def _stereo(x):
    """float64 (n, 2): a mono signal (n,) or (n, 1) is duplicated"""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        return np.stack([x, x], 1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, 1)
    return x[:, :2]


def _resample(x, sr):
    x = np.asarray(x, np.float64)
    if int(sr) == SR:
        return x
    from math import gcd
    g = gcd(int(sr), SR)
    return signal.resample_poly(x, SR // g, int(sr) // g, axis=0)


def _db(p, floor=1e-30):
    return 10 * np.log10(np.maximum(p, floor))


def is_stereo(x, tol_db=-60.0):
    """True when x has two channels that differ (side energy above `tol_db` re the mid)"""
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        return False
    m, s = 0.5 * (x[:, 0] + x[:, 1]), 0.5 * (x[:, 0] - x[:, 1])
    return bool(_db((s ** 2).sum()) - _db((m ** 2).sum()) > tol_db)


def _frames_fft(m, nfft=NFFT, hop=HOP):
    """Hann frames of m centred at k * hop (k = 0 .. len(m) // hop), rfft: (K, nfft // 2 + 1)"""
    K = len(m) // hop + 1
    xp = np.zeros((K - 1) * hop + nfft)
    xp[nfft // 2:nfft // 2 + len(m)] = m[:len(xp) - nfft // 2]
    fr = np.lib.stride_tricks.sliding_window_view(xp, nfft)[::hop][:K]
    return np.fft.rfft(fr * np.hanning(nfft + 2)[1:-1], axis=1)


def band_powers(m, centers=BANDS, nfft=NFFT, hop=HOP):
    """per band (log-frequency triangles that sum to one) and frame: the power of the mono signal m, (B, K)"""
    W = band_masks(centers, nfft, SR)
    return (np.abs(_frames_fft(np.asarray(m, np.float64), nfft, hop)) ** 2 @ W.T).T


def side_powers(x, centers=BANDS, nfft=NFFT, hop=HOP):
    """{'t': frame centres (s), 'PM', 'PS', 'C': (B, K)}: per band and frame the mid power, the side power and the
    real cross power Re(S M*) of mid M = (L + R) / 2 and side S = (L - R) / 2 (module doc)"""
    X = _stereo(x)
    M, S = 0.5 * (X[:, 0] + X[:, 1]), 0.5 * (X[:, 0] - X[:, 1])
    W = band_masks(centers, nfft, SR)
    FM, FS = _frames_fft(M, nfft, hop), _frames_fft(S, nfft, hop)
    return {'t': np.arange(len(FM)) * hop / SR, 'PM': (np.abs(FM) ** 2 @ W.T).T, 'PS': (np.abs(FS) ** 2 @ W.T).T,
            'C': (np.real(FS * np.conj(FM)) @ W.T).T}


def diffuse_power(P):
    """the side power uncorrelated with the mid, PS - C^2 / PM (>= 0), from side_powers()"""
    return np.maximum(P['PS'] - P['C'] ** 2 / np.maximum(P['PM'], 1e-30), 0.0)


def _env1(m):
    """1 ms hops, 5 ms window power envelope (replicate.env1)"""
    n = len(m)
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    w, hop = int(0.005 * SR), int(0.001 * SR)
    cen = np.arange(0, n, hop)
    a = np.clip(cen - w // 2, 0, n)
    b = np.clip(cen + w // 2, 0, n)
    return (c[b] - c[a]) / np.maximum(b - a, 1)


def active_span(x):
    """(onset, peak, end) sample indices of a sound (module doc): the onset is the first 1 ms frame within 30 dB of the
    loudest 10 ms, the end the last frame above the floor threshold"""
    m = _stereo(x).mean(1)
    E1 = _env1(m)
    if len(E1) < 3:
        return 0, 0, len(m)
    sm = np.convolve(E1, np.ones(10) / 10, mode='same')
    ip = int(np.argmax(sm))
    pk = float(sm[ip]) + 1e-30
    if len(E1) >= 80:
        fl = min(_db(float(np.median(E1[5:50]))), _db(float(np.median(E1[-25:-12]))))
        thr = float(np.clip(fl + 3.0 - _db(pk), -45.0, -12.0))
    else:
        thr = -40.0
    on = int(np.argmax(sm >= pk * 1e-3))
    live = np.nonzero(sm >= pk * 10 ** (thr / 10))[0]
    last = int(live[-1]) if len(live) else ip
    return on * 48, ip * 48, min(len(m), last * 48 + 48)


def tail_span(x, start_s=TAIL_START_S, min_s=0.02):
    """(first, last) sample of a sound's tail (peak + start_s to the end of active_span); None if shorter than min_s"""
    _, ip, end = active_span(x)
    a = ip + int(start_s * SR)
    return (a, end) if end - a >= int(min_s * SR) else None


def corr(L, R):
    """zero-lag correlation of two channels (None when either is silent)"""
    eL, eR = float((L * L).sum()), float((R * R).sum())
    if eL < 1e-20 or eR < 1e-20:
        return None
    return float((L * R).sum() / np.sqrt(eL * eR))


def tail_corr(x, span=None):
    """L/R correlation over the tail (tail_span unless `span` = (first, last) samples is given); None for no tail"""
    X = _stereo(x)
    span = tail_span(X) if span is None else span
    if span is None:
        return None
    a, b = span
    return corr(X[a:b, 0], X[a:b, 1])


def shaped_side(mid, bands, times, pan, diffuse, sign=1.0, sr=SR):
    """a side signal made from the mid: per band (band_masks of `bands`) the band's mid times its pan gain plus the
    band's quadrature (Hilbert transform: uncorrelated with the mid at zero lag, the same envelope) times its diffuse
    gain, both gains linear between keyframes at `times` (pan[b], diffuse[b]: one value per time)"""
    mid = np.asarray(mid, np.float64)
    n = len(mid)
    if n == 0:
        return np.zeros(0)
    nfft = sfft.next_fast_len(n + 8192)
    F = np.fft.rfft(mid, nfft)
    F[0] = 0.0
    W = band_masks(bands, nfft, sr)
    t = np.arange(n) / sr
    tk = np.asarray(times, np.float64)
    side = np.zeros(n)
    for b in range(len(bands)):
        pb = np.asarray(pan[b], np.float64)
        db = np.asarray(diffuse[b], np.float64)
        if np.abs(pb).max() > 1e-4:
            side += np.interp(t, tk, pb) * np.fft.irfft(F * W[b], nfft)[:n]
        if db.max() > 1e-4:
            side += np.interp(t, tk, db) * np.fft.irfft(F * W[b] * (-1j * sign), nfft)[:n]
    return side


# ================================================================================================ the wet layer
PARTS = {'side': ('side_db', 4099), 'mid': ('mid_db', 8191)}   # gains key, seed salt (independent noises)


def _gains(model, part):
    g = model.get(PARTS[part][0])
    return np.full(len(model['bands']), -150.0) if g is None else np.asarray(g, np.float64)


def impulse(model, part='side', seed=0, flat_iters=FLAT_ITERS):
    """one part's impulse response: per band seeded noise (band_masks), a rise over `build` after `predelay`, an
    exponential decay to -60 dB at the band's rt60, unit energy per band times 10^(gain / 20); zeros(1) when off.
    Each band is flattened (`flat_iters` passes: its spectrum's magnitude set to the band's shape, its phase kept, the
    envelope put back): plain decayed noise has a random power at every frequency (exponentially distributed), so a
    tonal dry's few partials got a wet 2-3 dB (std) louder or softer from seed to seed; flattened, 0.3-1 dB"""
    bands = np.asarray(model['bands'], np.float64)
    rt = np.clip(np.asarray(model['rt60'], np.float64), 0.02, MAX_RT)
    g = _gains(model, part)
    on = g > -140
    if not on.any():
        return np.zeros(1)
    pd, build = float(model.get('predelay', 0.0)), max(float(model.get('build', 0.005)), 1e-4)
    L = int((pd + build + min(MAX_RT, float(rt[on].max()))) * SR) + 1
    r = np.random.default_rng((int(seed) * 7919 + PARTS[part][1]) & 0xFFFFFFFF)
    nfft = sfft.next_fast_len(2 * L)
    F = np.fft.rfft(r.standard_normal(nfft))
    W = band_masks(bands, nfft, SR)
    t = np.arange(L) / SR - pd
    rise = np.where(t >= 0, 1.0 - np.exp(-np.maximum(t, 0.0) / build), 0.0)
    ir = np.zeros(L)
    for b in np.nonzero(on)[0]:
        env = rise * np.exp(-6.907755 * np.maximum(t, 0.0) / rt[b])
        u = np.fft.irfft(F * W[b], nfft)[:L] * env
        for _ in range(int(flat_iters)):
            U = np.fft.rfft(u, nfft)
            u = np.fft.irfft(W[b] * U / (np.abs(U) + 1e-20), nfft)[:L] * env
        ir += u * 10 ** (g[b] / 20) / (np.sqrt((u ** 2).sum()) + 1e-20)
    return ir


def wet(model, dry, seed=0, n=None, part='side'):
    """one part of the wet layer, (n,): the dry's mono sum convolved with impulse(model, part, seed), cut or padded
    to n samples (default the dry's length)"""
    m = _stereo(dry).mean(1)
    n = len(m) if n is None else int(n)
    ir = impulse(model, part, seed)
    if not np.any(ir) or not len(m):
        return np.zeros(n)
    w = signal.fftconvolve(m, ir)[:n]
    return np.pad(w, (0, n - len(w)))


def tail_s(model):
    """how long the wet rings past the dry (s): predelay + build + the longest rt60 of a part that is on"""
    rt = np.asarray(model['rt60'], np.float64)
    on = (_gains(model, 'side') > -140) | (_gains(model, 'mid') > -140)
    return 0.0 if not on.any() else float(model.get('predelay', 0.0)) + float(model.get('build', 0.005)) + min(
        MAX_RT, float(rt[on].max()))


def _orthogonal(w, refs, bands=BANDS):
    """w with, per band (band_masks), its zero-lag projection on each reference signal removed (Gram-Schmidt): a
    random wet that happens to line up with the dry's own side or mid would cancel it or lean the image"""
    n = len(w)
    nfft = sfft.next_fast_len(n + 8192)                 # padded: the band split must not wrap around in time
    Wm = band_masks(bands, nfft, SR)
    Fw = np.fft.rfft(w, nfft)
    Fr = [np.fft.rfft(r, nfft) for r in refs]
    out = np.zeros(n)
    for b in range(len(bands)):
        wb = np.fft.irfft(Fw * Wm[b], nfft)[:n]
        basis = []
        for F in Fr:
            v = np.fft.irfft(F * Wm[b], nfft)[:n]
            for u in basis:
                v = v - (v @ u) * u
            nv = np.sqrt(v @ v)
            if nv > 1e-12:
                basis.append(v / nv)
        for u in basis:
            wb = wb - (wb @ u) * u
        out += wb
    return out


def render_space(model, dry, seed=0, width=1.0, ring=False, limit=True):
    """the dry render (mono or (n, 2)) plus the wet layer: (n, 2). The wet side is made orthogonal (per band, zero
    lag) to the dry's side and mid, so it only ever adds width. width scales the side part; ring=True extends the
    result until the wet has rung out (a take); limit: peak-limit to -1 dBFS (sines.finish, no fade-in: the dry
    already starts at zero)"""
    y = _stereo(dry).copy()
    if not model:
        return y
    if ring:
        y = np.vstack([y, np.zeros((int(tail_s(model) * SR) + 1, 2))])
    ws, wm = _wet_parts(model, y, seed)
    ws = ws * float(width)
    y[:, 0] += wm + ws
    y[:, 1] += wm - ws
    return finish(y, SR, fade_s=0.0, limit=limit)


def _wet_parts(model, y, seed):
    """(side, mid) wet signals for the stereo dry y (same length): the side orthogonal to the dry's side and mid"""
    ws = wet(model, y, seed, len(y), 'side')
    wm = wet(model, y, seed, len(y), 'mid')
    if np.any(ws):
        ws = _orthogonal(ws, [0.5 * (y[:, 0] - y[:, 1]), y.mean(1) + wm])
    return ws, wm


# ================================================================================================ fitting
def _decay_rate(e_db, t, depth=30.0, min_pts=4):
    """dB / s of the line through a band's tail envelope (frames within `depth` dB of its tail maximum), None
    when it does not fall"""
    if len(e_db) < min_pts:
        return None
    sel = e_db >= e_db.max() - depth
    if sel.sum() < min_pts:
        return None
    s = float(np.polyfit(t[sel], e_db[sel], 1)[0])
    return s if s < -1.0 else None


def _smooth(P, k=5):
    return np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode='same'), -1, P)


def needs_space(x, max_corr=TAIL_CORR_MAX):
    """True when x is stereo and its tail's L/R correlation is at most max_corr (a decorrelated, reverberant tail)"""
    if not is_stereo(x):
        return False
    c = tail_corr(x)
    return c is not None and c <= max_corr


def fit_space(x, sr=SR, dry=None, max_corr=TAIL_CORR_MAX, iters=3, need_db=NEED_DB, mid_need_db=MID_NEED_DB):
    """a space model for recording x (path or array, (n, 2)), or None when x is mono, has no tail or its tail
    correlation is above max_corr, or when the dry render (mono or (n, 2), the model's seed-0 render through its
    post stage) already carries the tail's diffuse side to within need_db (module doc)"""
    if isinstance(x, str):
        from ... import audio
        x, sr = audio.load(x, sr=SR), SR
    X = _stereo(_resample(x, sr))
    if not is_stereo(X):
        return None
    span = tail_span(X)
    tc = tail_corr(X, span)
    if tc is None or tc > max_corr:
        return None
    n = len(X)
    P = side_powers(X)
    dif = _smooth(diffuse_power(P))
    t = P['t']
    tail = (t >= span[0] / SR) & (t <= span[1] / SR)
    if tail.sum() < 3:
        return None
    E_dif = dif[:, tail].sum(1)
    E_mid = _smooth(P['PM'])[:, tail].sum(1)
    need = {'side': E_dif.copy(), 'mid': np.zeros(len(BANDS))}
    D, tc_dry = None, 1.0
    if dry is not None:
        D = _stereo(dry)[:n]
        D = np.vstack([D, np.zeros((n - len(D), 2))])
        tc_dry = tail_corr(D, span)
        if tc_dry is None or tc_dry <= tc + 0.1:          # the dry is about as decorrelated already
            return None
        PD = side_powers(D)
        need['side'] = np.maximum(E_dif - _smooth(diffuse_power(PD))[:, tail].sum(1), 0.0)
        short = np.maximum(E_mid - _smooth(PD['PM'])[:, tail].sum(1), 0.0)       # the dry's mono tail falls short
        need['mid'] = np.where(short >= E_mid * 10 ** (mid_need_db / 10), short, 0.0)
    total = float(need['side'].sum())
    need_rel = _db(total) - _db(float(E_dif.sum()))
    if total <= 0 or need_rel < need_db:
        return None
    B = len(BANDS)
    live = {p: v > max(float(v.max()), 1e-30) * 1e-4 for p, v in need.items()}
    # starting rt60s: the decay of the recording's diffuse side per band over the tail; a tail that falls 12-45 dB
    # over its span cannot ring much longer than RT_SPAN x the span (a band that hardly falls would run to MAX_RT)
    tt = t[tail]
    rt_max = float(np.clip(RT_SPAN * (span[1] - span[0]) / SR, 0.1, MAX_RT))
    rate_need = [(_decay_rate(_db(dif[b, tail]), tt) if live['side'][b] else None) for b in range(B)]
    rts = [(-60.0 / s if s else None) for s in rate_need]
    known = [v for v in rts if v]
    fill = float(np.median(known)) if known else 0.5
    rt = np.clip([v if v else fill for v in rts], 0.05, rt_max)
    # the build-up: how far the diffuse side's loudest frame lags the mid's
    lag = float(t[int(np.argmax(dif.sum(0)))] - t[int(np.argmax(P['PM'].sum(0)))])
    model = {'kind': KIND, 'version': VERSION, 'predelay': 0.002, 'build': round(float(np.clip(lag, 0.005, 0.08)), 4),
             'bands': list(BANDS), 'rt60': [round(float(v), 4) for v in rt],
             'side_db': [0.0 if live['side'][b] else -150.0 for b in range(B)],
             'mid_db': [0.0 if live['mid'][b] else -150.0 for b in range(B)], 'tail_corr': round(float(tc), 4)}
    src = X.mean(1) if D is None else D               # no dry: the recording's own mid stands in for it
    for _ in range(int(iters)):
        r_ = np.asarray(model['rt60'], float)
        for part in ('side', 'mid'):
            if not live[part].any():
                continue
            Pw = _smooth(band_powers(wet(model, src, 0, n, part)))
            Ew = Pw[:, tail].sum(1)
            g = _gains(model, part)
            for b in range(B):
                if not live[part][b] or Ew[b] <= 0:
                    continue
                g[b] += 10 * np.log10(need[part][b] / Ew[b])
                if part == 'side':
                    sw, sn = _decay_rate(_db(Pw[b, tail]), tt), rate_need[b]
                    if sw and sn:            # the render decays slower (sw > sn) -> shorter rt60
                        r_[b] = np.clip(r_[b] * np.clip(sw / sn, 0.5, 2.0), 0.05, rt_max)
            model[PARTS[part][0]] = [round(float(np.clip(v, -150.0, 40.0)), 2) if live[part][b] else -150.0
                                     for b, v in enumerate(g)]
        model['rt60'] = [round(float(v), 4) for v in r_]
    mid_on = bool(live['mid'].any())
    if mid_on:                            # the mid part changes the mono sum: only if the mono gets no worse
        from .quality import mel_distance
        if mel_distance(X, render_space(model, src, seed=0, limit=False)) > mel_distance(X, src) + 0.02:
            model['mid_db'] = [-150.0] * B
            mid_on = False
    # the side's overall level onto the recording's tail correlation (the energy match lands near it; the correlation
    # is what is heard): L = dry + mid + g side, R = dry + mid - g side over the tail, g on a 0.5 dB grid of +-OFFSET_DB
    # A short tail holds few independent cycles (a 40 ms tail at 125 Hz: about five), so one render's tail correlation
    # is a noisy statistic: the products are pooled over FIT_SEEDS renders, and their correlation is what is matched
    i, j = span
    S2 = _stereo(src)
    acc = np.zeros(6)
    for s in FIT_SEEDS:
        ws, wm = _wet_parts(model, S2, s)
        ws, wm = ws[i:j], wm[i:j]
        a, b_ = S2[i:j, 0] + wm, S2[i:j, 1] + wm
        acc += [a @ a, b_ @ b_, ws @ ws, a @ b_, a @ ws, b_ @ ws]
    aa, bb, ss, ab, as_, bs = acc
    offs = np.arange(-OFFSET_DB, OFFSET_DB + 1e-9, 0.5)
    g = 10 ** (offs / 20)
    num = ab - g * as_ + g * bs - g * g * ss
    den = np.sqrt(np.maximum((aa + 2 * g * as_ + g * g * ss) * (bb - 2 * g * bs + g * g * ss), 1e-30))
    k = int(np.argmin(np.abs(num / den - tc)))
    off, tc_wet = float(offs[k]), float(num[k] / den[k])
    model['side_db'] = [round(v + off, 2) if v > -140 else -150.0 for v in model['side_db']]
    # analysis by synthesis: keep the layer only if it brings the tail's correlation closer to the recording's
    if abs(tc_wet - tc) >= abs(tc_dry - tc):
        return None
    model['fit'] = {'need_db': round(float(need_rel), 2),
                    'mid_need_db': (round(float(_db(need['mid'].sum()) - _db(E_mid.sum())), 2) if mid_on else None),
                    'iters': int(iters), 'dry': D is not None, 'side_offset_db': off,
                    'tail_corr_dry': round(float(tc_dry), 4), 'tail_corr_wet': round(float(tc_wet), 4)}
    return model


def describe(model):
    """one line of words about a space model"""
    if not model:
        return 'no space layer'
    side, mid = _gains(model, 'side') > -140, _gains(model, 'mid') > -140
    on = side | mid
    if not on.any():
        return 'space layer (silent)'
    rt = np.asarray(model['rt60'], float)[on]
    bands = np.asarray(model['bands'], float)[on]
    return (f"diffuse tail ({'side + mid' if mid.any() else 'side only'}): rt60 {rt.min():.2f}-{rt.max():.2f} s over "
            f"{bands.min():.0f}-{bands.max():.0f} Hz, build {model['build'] * 1000:.0f} ms, recording's tail corr "
            f"{model.get('tail_corr', float('nan')):.2f}")
