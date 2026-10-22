"""Shared building blocks of the earshot 0.4 tonal / UI recipes (ui_pop, knock, mallet, drip, shutter): a soft onset
with a real decay, a gliding tone with soft partials, noise that follows a pitch, short contact bursts, struck modes,
and the finish (band limits, a small room, width, a clean start and end). Every sample is computed.

    from earshot.forge import dsp
    from earshot.forge.recipes import _tonal as T
    t = dsp.t_of(dsp.n_of(0.3))
    e = T.env(t, onset=0.03, tau=0.035)                    # 30 ms soft rise, then 20 dB down in ~80 ms
    y = T.tone(T.glide(t, 560.0, 3.0, 0.02), dsp.rng(1), [(2, 0.3, 1.5)], e)
    x = T.finish(y, dsp.rng(1), hp=120.0, lp=5000.0, room=0.08, width=0.1)

Why a shared module: the films' UI sounds differ in body (a tone, a struck bar, a wooden block, a bubble, a shutter's
two contacts) but share the parts that the 0.3 bank got wrong in all of them (the earshot 0.4 film study): the onset
(20-60 ms in the films' pops and dings, 5 ms in the bank), the decay (a real ring-down to 0.2-0.4 s, not a fade at the
end of a 2 s ring) and the width (corr 0.3-0.7 on mallet notes, ~1.0 in the bank).
"""
import numpy as np

from .. import dsp, fx

SR = dsp.SR


def env(t, onset, tau, hold=0.0, curve=1.0, dcurve=1.0):
    """a soft rise from 0 to 1 over `onset` s (a raised cosine; curve > 1 starts slower, < 1 faster), `hold` s at the
    top, then a decay exp(-(t' / tau) ** dcurve): dcurve 1 = exponential (20 dB down in 2.3 tau), > 1 = a body that is
    damped as it rings (slower at first, then faster: t40 / t20 = 2 ** (1 / dcurve))"""
    u = np.clip(t / max(onset, 1e-4), 0, 1) ** curve
    rise = 0.5 - 0.5 * np.cos(np.pi * u)
    x = np.maximum(t - onset - hold, 0) / max(tau, 1e-4)
    return rise * np.exp(-(x if dcurve == 1.0 else x ** dcurve))


def two_rate(t, onset, fast, tf, ts, curve=1.0):
    """a struck body's decay: a soft rise over `onset`, then `fast` of the level dying with tf, the rest with ts (most
    of a strike dies in tens of ms, a quiet ring carries on)"""
    u = np.clip(t / max(onset, 1e-4), 0, 1) ** curve
    rise = 0.5 - 0.5 * np.cos(np.pi * u)
    tm = np.maximum(t - onset, 0)
    return rise * (fast * np.exp(-tm / max(tf, 1e-4)) + (1 - fast) * np.exp(-tm / max(ts, 1e-4)))


def glide(t, f0, bend_st=0.0, ptau=0.02):
    """a pitch track (Hz per sample) that starts bend_st semitones off f0 (> 0: above, so it falls) and settles on f0
    exponentially with time constant ptau s"""
    return f0 * 2 ** (bend_st / 12 * np.exp(-t / max(ptau, 1e-4)))


def rise_track(t, f0, rise_st, rtau):
    """a pitch that RISES from f0 by rise_st semitones, most of it within ~3 rtau s (a bubble's cavity shrinking)"""
    return f0 * 2 ** (rise_st / 12 * (1 - np.exp(-t / max(rtau, 1e-4))))


def _alias(f, q):
    """a partial's gain as it nears Nyquist (faded out over the last 2.4 kHz below 0.45 SR)"""
    return np.clip((0.45 * SR - q * f) / 2400.0, 0, 1)


def tone(f, r, parts=(), e=None):
    """a sine on the pitch track f (Hz per sample) plus partials [(ratio, amp, decay)], each following the envelope
    e ** decay (decay > 1: dies faster than the body, a natural rounded tone); ratio may be inharmonic"""
    n = len(f)
    e = np.ones(n) if e is None else e
    ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 2 * np.pi)
    y = np.sin(ph) * e
    for q, a, d in parts:
        if a <= 0:
            continue
        y = y + a * _alias(f, q) * e ** d * np.sin(q * ph + r.uniform(0, 2 * np.pi))
    return y


def follow_noise(n, r, fc, q, e):
    """turbulence through a band-pass that follows fc (Hz per sample or a number): a breath or cavity hiss, scaled to
    the RMS of a unit sine over the envelope's live part (so 0 dB = as strong as the tone), times e"""
    fc = np.minimum(np.broadcast_to(np.asarray(fc, float), (n,)), 16000.0)
    b = dsp.svf(r.standard_normal(n), fc, q, 'band')
    live = e > 0.1 * (e.max() + 1e-12)
    b = b / (np.sqrt(np.mean(b[live] ** 2)) + 1e-12) * np.sqrt(0.5)
    return b * e


def burst(n, r, lo, hi, tau, rise=0.0003, order=2):
    """a short contact noise of length n samples: white noise band-passed lo-hi Hz, a `rise` s rise and an exponential
    decay (tau s); peak 1"""
    k = min(n, dsp.n_of(max(6 * tau + rise, 0.002)))
    tk = dsp.t_of(k)
    w = dsp.butter(r.standard_normal(k), 'band', (max(lo, 20.0), min(hi, 0.45 * SR)), order=order)
    w *= (1 - np.exp(-tk / max(rise, 1e-5))) * np.exp(-tk / max(tau, 1e-5))
    y = np.zeros(n)
    y[:k] = w / (np.abs(w).max() + 1e-12)
    return y


def thump(n, f, tau, rise=0.002):
    """a soft low thump of length n: a damped sine at f Hz with a `rise` s onset; peak ~1"""
    t = dsp.t_of(n)
    y = np.sin(2 * np.pi * f * t) * (1 - np.exp(-t / max(rise, 1e-4))) * np.exp(-t / max(tau, 1e-4))
    return y / (np.abs(y).max() + 1e-12)


def modes(n, r, freqs, amps, taus, jitter=0.0, bend=0.0, bend_tau=0.004):
    """struck modes: damped sines (Hz, linear amplitudes, time constants s); `bend` = a fraction the pitch starts above
    (a hard strike stiffens the body for a few ms), settling with bend_tau"""
    t = dsp.t_of(n)
    y = np.zeros(n)
    b = 1 + bend * np.exp(-t / max(bend_tau, 1e-4))
    for f, a, tau in zip(freqs, amps, taus):
        if a <= 0 or f >= 0.45 * SR:
            continue
        f = f * (1 + jitter * r.standard_normal())
        ph = 2 * np.pi * np.cumsum(f * b) / SR + r.uniform(0, 2 * np.pi)
        y += a * np.exp(-t / max(tau, 1e-5)) * np.sin(ph)
    return y


def strike(y, r, contact_s, bright=0.5):
    """a resonator's response driven by a contact (dsp.contact: a raised-cosine force pulse of contact_s s plus a little
    grit); a longer contact (a soft mallet, a fingertip) excites the high modes less"""
    return dsp.excite(y, dsp.contact(contact_s, r, bright))


def room(y, r, wet, rt60, size=0.15, width=0.3):
    """a small room on a stereo (n, 2) signal (fx.reverb, stereo); wet 0 = none (no rng draw)"""
    if wet <= 0.005:
        return y
    return fx.reverb(y, r, rt60=rt60, size=size, predelay=0.004, damp=0.6, wet=wet, stereo=True, width=width)


def finish(y, r, hp=80.0, lp=12000.0, room_wet=0.0, rt60=0.15, width=0.0, cut=500.0, end_db=-60.0, fade=0.006,
           pan=0.0, size=0.15):
    """mono -> stereo: band limits (4th-order high-pass at hp, 2nd-order low-pass at lp; lp >= 17.5 kHz = none), width
    (fx.decorrelate: a decorrelated side above `cut` Hz), a constant-power pan, a small room, the tail trimmed where it
    stays end_db under the peak, a clean start (0.3 ms) and a convex end fade. The dry signal's own end is faded first
    (20 ms, at most 15 % of it): it is ~60 dB down there, but the room rings on past it, and a low body cut within
    one of its cycles (a 95 Hz desk thump: 10.5 ms) leaves a step in the quiet tail that the audit hears as a click"""
    y = np.array(y, np.float64)
    kd = min(len(y), dsp.n_of(0.02), int(0.15 * len(y)))
    y[len(y) - kd:] *= np.cos(np.linspace(0, np.pi / 2, kd)) ** 2
    y = dsp.butter(y, 'high', hp, order=4)
    if lp < 17500:
        y = dsp.butter(y, 'low', lp, order=2)
    st = fx.decorrelate(y, width, cut=cut) if width > 0.005 else np.stack([y, y], 1)
    if abs(pan) > 0.005:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = st * np.array([np.cos(th), np.sin(th)]) * np.sqrt(2)
    st = room(st, r, room_wet, rt60, size=size, width=0.3 + width)
    st = dsp.trim_tail(st, end_db, 0.01)
    k = min(len(st), dsp.n_of(fade))
    w = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    st[-k:] *= w[:, None]
    a = min(len(st), dsp.n_of(0.0003))
    st[:a] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a)))[:, None]
    return st


def measure(x):
    """the catalog's model-free measurements of one render on silence (the same definitions the films were measured
    with: the earshot 0.4 film catalog, stage A): dur_s (onset at -30 dB to the last frame within 45 dB), attack_ms
    (first ms within 20 dB of the peak to the peak, 5 ms RMS smoothed 10 ms), t20_ms / t40_ms, centroid_hz, lo150 / hi4k
    (energy shares under 150 Hz / over 4 kHz), tonal (mean share within a quarter octave of the strongest bin), glide_st
    and corr (L / R correlation)"""
    y = np.asarray(x, np.float64)
    if y.ndim == 1:
        y = np.stack([y, y], 1)
    m = y.mean(1)
    n = len(m)
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    w, hop = int(0.005 * SR), int(0.001 * SR)
    cen = np.arange(0, n, hop)
    a, b = np.clip(cen - w // 2, 0, n), np.clip(cen + w // 2, 0, n)
    sm = np.convolve((c[b] - c[a]) / np.maximum(b - a, 1), np.ones(10) / 10, mode='same')
    ip = int(np.argmax(sm))
    pk = sm[ip]
    above = np.nonzero(sm[:ip + 1] >= pk * 1e-2)[0]
    on = np.nonzero(sm >= pk * 1e-3)[0]
    after = sm[ip:]
    w20, w40 = np.nonzero(after < pk * 1e-2)[0], np.nonzero(after < pk * 1e-4)[0]
    a45 = np.nonzero(sm >= pk * 10 ** -4.5)[0]
    out = {'dur_s': round((a45[-1] if len(a45) else n / hop) * 0.001 - (on[0] * 0.001 if len(on) else 0.0), 3),
           'attack_ms': float(ip - above[0]) if len(above) else None,
           't20_ms': float(w20[0]) if len(w20) else None, 't40_ms': float(w40[0]) if len(w40) else None}
    nf, hp_ = 2048, 240
    seg = np.pad(m, (0, max(0, nf - n)))
    k = 1 + (len(seg) - nf) // hp_
    idx = np.arange(nf)[None, :] + hp_ * np.arange(k)[:, None]
    P = (np.abs(np.fft.rfft(seg[idx] * np.hanning(nf), axis=1)) ** 2).T
    F = np.fft.rfftfreq(nf, 1 / SR)
    fe = P.sum(0)
    act = fe >= fe.max() * 1e-2
    spec = P[:, act].mean(1)
    tot = spec.sum() + 1e-20
    out['centroid_hz'] = round(float((spec * F).sum() / tot))
    out['lo150'] = round(float(spec[F < 150].sum() / tot), 3)
    out['hi4k'] = round(float(spec[F >= 4000].sum() / tot), 3)
    band = (F >= 80) & (F <= 12000)
    rid, ton = [], []
    for j in np.nonzero(act)[0]:
        col = P[band, j]
        if col.max() <= 0:
            continue
        fr = F[band][int(np.argmax(col))]
        rid.append(fr)
        ton.append(col[(F[band] >= fr / 2 ** 0.25) & (F[band] <= fr * 2 ** 0.25)].sum() / (col.sum() + 1e-20))
    out['tonal'] = round(float(np.mean(ton)), 2) if len(ton) >= 2 else None
    if len(rid) >= 2:
        k3 = max(1, min(3, len(rid) // 3))
        out['glide_st'] = round(float(12 * np.log2(np.median(rid[-k3:]) / np.median(rid[:k3]))), 1)
    else:
        out['glide_st'] = None
    L, R = y[:, 0], y[:, 1]
    out['corr'] = round(float((L * R).sum() / (np.sqrt((L * L).sum() * (R * R).sum()) + 1e-20)), 2)
    return out


def outside(m, targets):
    """how far measurements fall outside target ranges {key: (lo, hi, scale)}: the sum of (distance / scale), each
    capped at 2 (0 = every target met)"""
    pen = 0.0
    for k, (lo, hi, sc) in targets.items():
        v = m.get(k)
        if v is None:
            pen += 1.0
        elif v < lo:
            pen += min(2.0, (lo - v) / sc)
        elif v > hi:
            pen += min(2.0, (v - hi) / sc)
    return pen


def family_space(space, params, fine):
    """a member's search space: its `fine` keys free (the SPACE ranges), every other key pinned to its value (a
    one-element choice list: the forge's way of fixing a parameter)"""
    return {k: (spec if k in fine else [params.get(k)]) for k, spec in space.items() if k in params}
