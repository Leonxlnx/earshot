"""Endless beds from a learned texture: takes of the resynthesis models (earshot.synth.resyn) of a real recording,
cut into pieces of one to three seconds and stitched in a new order where the texture is in a similar state, so a
bed of any length plays without an audible repeat and loops without a seam.

Why. A recording of rain or a fire is ten to sixteen seconds of numbers here (a few hundred kB); a game or a page
plays it for minutes. Looping the take repeats every event at the same place, which the ear finds within two or
three rounds. A take of a resyn model already differs from the recording in its noise (the band envelopes are the
recording's, the fine structure is drawn fresh, `vary` redraws part of the envelope's grain and nudges pitch,
length and tilt); quilting moves the events as well. Pieces are cut where nothing happens (no crack, chirp or
drop under the crossfade) and joined to pieces that start in a similar state: the band levels of the 150 ms each
side of the cut are compared, so the level and the spectrum carry on across the joint as they do inside the
texture. A crossfade of 80-150 ms keeps the power constant for the two signals as they are (equal power for
independent noise, less of a dip where the two happen to be alike). Within twenty seconds of the output a piece of
the model is not played again where the material allows it. A periodic texture (waves) is stitched from whole
periods: each stored chunk is one wave from trough to trough, and the joints sit in the troughs.

A loop closes the same way: the last piece fades into the first one, the joint chosen like any other.

What it measures (the tests use these): seams(), the level and spectrum jump across each joint
against the same measure inside the takes (the texture's own variation), and whether an event lies under a fade;
repeats(), how alike the closest two two-second stretches 2-20 s apart are (band envelopes at 50 ms).

    from earshot.forge import dsp
    from earshot.synth import quilt, resyn
    models = [resyn.load(p) for p in paths]                   # the chunks of one learned texture
    r = dsp.rng(3)
    takes = quilt.takes(models, 20.0, r)                      # renders: fresh noise and a little vary each
    y, info = quilt.stitch(takes, 20.0, r)                     # (n, 2): a seamless loop of 20 s
    quilt.seams(y, info)['ratio'], quilt.repeats(y)['max']
"""
import numpy as np
from scipy import ndimage

from ..forge import dsp
from . import resyn

SR = dsp.SR
HOP = 480                                    # 10 ms: the frame of the state and event measures
NFFT = 2048
EDGES = np.geomspace(50.0, 18000.0, 25)      # 24 bands, a third of an octave and wider
MARGIN = 0.12                                # s at each end of a take never used (the render's edge fades)
WINDOW = 15                                  # frames each side of a cut whose band levels are its state (150 ms)
MEMORY = 20.0                                # s within which a piece of the model is not played again if avoidable
FADE = (0.08, 0.15)                          # s: the crossfades
PIECE = (1.0, 3.0)                           # s: the pieces of a texture that is not periodic
EDGE = 0.4                                   # s of a wave's chunk before its first trough and after its last


# ------------------------------------------------------------------------------------------------ measures
def features(y):
    """band levels in dB of the mono sum: (24, frames) at 10 ms, frame k centred on sample k * HOP"""
    x = np.asarray(y, np.float64)
    x = x.mean(axis=1) if x.ndim == 2 else x
    k = len(x) // HOP + 1
    pad = np.pad(x, (NFFT // 2, NFFT // 2 + HOP))
    f = np.fft.rfftfreq(NFFT, 1.0 / SR)
    edges = np.searchsorted(f, EDGES)
    win = np.hanning(NFFT)
    out = np.empty((len(EDGES) - 1, k))
    for a in range(0, k, 512):
        b = min(k, a + 512)
        idx = np.arange(NFFT)[None, :] + HOP * np.arange(a, b)[:, None]
        P = np.abs(np.fft.rfft(pad[idx] * win, axis=1)) ** 2
        out[:, a:b] = np.add.reduceat(P, edges[:-1], axis=1)[:, :len(EDGES) - 1].T / NFFT
    return 10 * np.log10(out + 1e-12)


def _level(F):
    return 10 * np.log10(np.sum(10 ** (F / 10), axis=0) + 1e-12)


def onsets(F):
    """onset strength per frame: the mean rise of the band levels over 20 ms (dB)"""
    o = np.zeros(F.shape[1])
    o[2:] = np.maximum(F[:, 2:] - F[:, :-2], 0.0).mean(axis=0)
    return o


def events(F):
    """frames that hold an event that stands out of the texture (a loud crack, a chirp, a drop, a wave's break):
    an onset far above the texture's usual ones (over median + 10 median deviations and among the strongest 2 %)
    or a level 8 dB over the running median of the second around it (and among the 2 % that stand out most),
    widened by 30 ms. The small crackles and drops that make the texture are not events: under a crossfade of
    80-150 ms one of them only comes out a little softer"""
    o = onsets(F)
    mad = np.median(np.abs(o - np.median(o))) + 1e-9
    L = _level(F)
    up = L - ndimage.median_filter(L, 101, mode='nearest')
    ev = (o > max(np.median(o) + 10.0 * mad, np.percentile(o, 98))) | (up > max(8.0, np.percentile(up, 98)))
    return ndimage.binary_dilation(ev, np.ones(7, bool))


def _states(F):
    """the state of the texture at every frame: its band levels over 150 ms and over half a second (a gust that
    is rising or falling is not the same state as one at its top)"""
    return np.concatenate([ndimage.uniform_filter1d(F, 2 * WINDOW + 1, axis=1, mode='nearest'),
                           ndimage.uniform_filter1d(F, 101, axis=1, mode='nearest')])


def _jumps(F, at, gap):
    """the band-level jump (dB RMS over the audible bands) and the level jump (dB) between the 100 ms before
    at - gap / 2 and the 100 ms after at + gap / 2 (frames)"""
    live = F.mean(axis=1) > F.mean(axis=1).max() - 45.0
    out, lv = [], []
    L = _level(F)
    for j, g in zip(at, gap):
        a, b = int(j - g // 2), int(j + (g + 1) // 2)
        if a - 10 < 0 or b + 10 > F.shape[1]:
            continue
        x, y = F[live, a - 10:a].mean(axis=1), F[live, b:b + 10].mean(axis=1)
        out.append(float(np.sqrt(np.mean((x - y) ** 2))))
        lv.append(float(abs(L[a - 10:a].mean() - L[b:b + 10].mean())))
    return np.array(out), np.array(lv)


# ------------------------------------------------------------------------------------------------ takes
def takes(models, dur, r, vary=0.4, periods=None, energy=0.0, pair=False, **transform):
    """renders of a texture's chunk models (resyn.render with fresh noise, `vary` and `transform`) for a bed of
    `dur` s, each chunk rendered once before any twice. Not periodic: about 1.4 dur + 3 s of material, at least
    two takes (the more material, the better the joints match). Periodic (`periods`: each chunk's length from
    trough to trough, s): one take per wave, none of the last few waves again, the louder waves more often with
    `energy` above 0 and the quieter below, all warped by one factor so the waves fill `dur`. pair: a model of a
    mono recording heard as a spaced pair of microphones would hear it: left and right are two renders with the
    same pace and size and their own noise, so every crackle comes at the same moment on both sides while the
    noise between them differs (`vary` is then 0: it would move the two sides apart). A list of {'y', 'chunk'}, a
    wave's with 'troughs': the samples of its first and its last trough"""
    if pair:
        vary = 0.0
    out, order = [], []
    if periods is not None:
        periods = np.asarray(periods, np.float64) * transform.get('size', 1.0)     # a bigger sea: slower waves
        top = np.array([float(m['top']) for m in models])
        w = np.exp(energy * (top - top.mean()) / 2.0)
        best = None
        for _ in range(24):                            # the draw of waves whose lengths come nearest to dur
            k = max(1, int(round(dur / periods.mean() * r.uniform(0.85, 1.15))))
            seq = []
            while len(seq) < k:
                p = w.copy()
                if seq and len(models) > 1:            # none of the last few waves again (one if only two)
                    p[seq[-max(1, len(models) - 2):]] = 0.0
                if len(seq) == k - 1 and k > 2 and len(models) > 2:
                    p[seq[0]] = 0.0                    # a loop: the last wave runs into the first
                seq.append(int(r.choice(len(models), p=p / p.sum())))
            miss = abs(np.log(dur / periods[seq].sum()))
            if best is None or miss < best[0]:
                best = (miss, seq)
        seq = best[1]
        stretch = float(np.clip(dur / periods[seq].sum(), 0.8, 1.25))
        size = transform.get('size', 1.0)
        for c in seq:
            y = _take(models[c], r, vary, dict(transform, length=stretch), 0.04, pair)
            a = int(round(len(y) * EDGE * size / (periods[c] + 2 * EDGE * size)))
            out.append({'y': y, 'chunk': c, 'troughs': (a, len(y) - a)})
        return out
    need, total = 1.4 * dur + 3.0, 0.0
    while total < need or len(out) < 2:
        if not order:
            order = [int(i) for i in r.permutation(len(models))]
        c = order.pop(0)
        out.append({'y': _take(models[c], r, vary, transform, 0.1, pair), 'chunk': c})
        total += len(out[-1]['y']) / SR
    return out


def _take(model, r, vary, transform, spread, pair=False):
    """one render with its own small differences of pace (time between events, `spread` in log units) and of
    source size (3 %: a little lower and slower or higher and quicker), so a stretch of the model heard twice
    is not the same stretch twice; pair: two renders of the same pace and size, one per side"""
    tr = dict(transform)
    tr['length'] = tr.get('length', 1.0) * float(np.exp(spread * r.standard_normal()))
    tr['size'] = tr.get('size', 1.0) * float(np.exp(0.03 * r.standard_normal()))
    y = resyn.render(model, dsp.rng(int(r.integers(2 ** 31))), vary=vary, **tr)
    if pair:
        z = resyn.render(model, dsp.rng(int(r.integers(2 ** 31))), vary=vary, **tr)
        k = min(len(y), len(z))
        y = np.stack([y[:k, 0], z[:k, 1]], axis=1)
    return y


def _prep(tk):
    """per take: features, states, the frames a cut may sit on, the busyness of the moment"""
    F = features(tk['y'])
    o = onsets(F)
    ev = events(F)
    m = int(np.ceil((MARGIN + FADE[1] / 2) * SR / HOP))
    ok = ~ndimage.binary_dilation(ev, np.ones(2 * int(np.ceil((FADE[1] / 2 + 0.03) * SR / HOP)) + 1, bool))
    ok[:m], ok[-m - 1:] = False, False
    busy = ndimage.uniform_filter1d(o, 200, mode='nearest') + 0.05 * ndimage.uniform_filter1d(
        _level(F), 200, mode='nearest')
    return dict(F=F, S=_states(F), ok=ok, ev=ev, busy=busy, n=len(tk['y']), chunk=tk['chunk'],
                troughs=tk.get('troughs'))


def _troughs(p):
    """the cut frames of one wave take: where its chunk was cut from the recording, trough to trough (EDGE s from
    each end, as warped), or, for a take that does not say, the quietest moments (level smoothed over 300 ms) in its
    first and its last 40 %. A wave is played whole: a quiet moment inside the wash would cut a third of it away,
    and the bed would then be resampled by as much to fill its length, every sound in it that much lower"""
    if p.get('troughs'):
        return tuple(int(round(v / HOP)) for v in p['troughs'])
    L = ndimage.uniform_filter1d(_level(p['F']), 31, mode='nearest')
    k = len(L)
    m = int(np.ceil((MARGIN + FADE[1] / 2) * SR / HOP)) + 1
    a = m + int(np.argmin(L[m:max(m + 1, int(0.4 * k))]))
    b = int(0.6 * k) + int(np.argmin(L[int(0.6 * k):max(int(0.6 * k) + 1, k - m)]))
    return a, b


# ------------------------------------------------------------------------------------------------ the plan
def _dist(Sa, Sb, live):
    return np.sqrt(np.mean((Sa[live, None] - Sb[live]) ** 2, axis=0)) if Sb.ndim == 2 else float(
        np.sqrt(np.mean((Sa[live] - Sb[live]) ** 2)))


def _choose(cost, r, keep=6, temp=0.7):
    best = np.argsort(cost)[:keep]
    w = np.exp(-(cost[best] - cost[best[0]]) / temp)
    return int(best[r.choice(len(best), p=w / w.sum())])


def _plan(preps, n, r, loop, energy, piece):
    """pieces [(take, first sample, last sample)] whose lengths sum to n, chosen as the module text says"""
    lo, hi = int(piece[0] * SR), int(piece[1] * SR)
    cand = [(i, f) for i, p in enumerate(preps) for f in np.nonzero(p['ok'])[0][::2]]
    T = np.array([c[0] for c in cand])
    Fr = np.array([c[1] for c in cand])
    S = np.stack([preps[i]['S'][:, f] for i, f in cand], axis=1)
    busy = np.array([preps[i]['busy'][f] for i, f in cand])
    busy = (busy - busy.mean()) / (busy.std() + 1e-9)
    room = np.array([preps[i]['n'] for i in T]) - Fr * HOP - int((MARGIN + FADE[1]) * SR)
    oks = [np.concatenate([[0], np.cumsum(p['ok'])]) for p in preps]

    def ends(span):                                # whether a cut is allowed somewhere span[0]..span[1] s on
        a = np.minimum(Fr + int(span[0] * SR) // HOP, [len(preps[i]['ok']) for i in T])
        b = np.minimum(Fr + int(span[1] * SR) // HOP + 1, [len(preps[i]['ok']) for i in T])
        return np.array([oks[i][y] - oks[i][x] for i, x, y in zip(T, a, b)]) > 0
    chunk = np.array([preps[i]['chunk'] for i in T])
    allF = np.concatenate([p['F'].mean(axis=1, keepdims=True) for p in preps], axis=1).mean(axis=1)
    live = np.tile(allF > allF.max() - 45.0, 2)
    used, pieces, pos = [], [], 0
    first = None
    while pos < n:
        rem = n - pos
        final = rem <= hi
        if final:
            want = rem
        elif rem <= hi + lo:                       # leave a short last piece: it has to fit the most constraints
            want = rem - lo
        else:
            want = int(r.uniform(lo, min(hi, rem - lo)))
        tries = [(want, final)] + ([(lo, False)] if not final and want > lo else []) + (
            [(rem - lo, False)] if final and rem >= 2 * lo else [])
        for want, final in tries:                  # short of fresh material: a shorter piece, or the last one in two
            fits = room >= want
            cost = np.where(fits if fits.any() else room >= min(want, lo), 0.0, np.inf)
            cost -= 4.0 * energy * busy
            if not final:
                cost += np.where(ends((lo / SR, min(hi, rem - lo) / SR)), 0.0, 20.0)
            hard = np.zeros(len(T), bool)
            for i, k, a, b, t0, t1 in used:    # the same samples never again; a piece of the model already played: the
                near = min(pos - t1, t0 + n - pos - want) if loop else pos - t1      # more often and the more recently
                over = np.clip(np.minimum(Fr * HOP + want, b) - np.maximum(Fr * HOP, a), 0, None) / want    # the worse
                same = (T == i) * over
                hard |= same * want > 0.2 * SR
                cost += 1000.0 * (same + (same > 0)) + 8.0 * over * (chunk == k)
                if near < MEMORY * SR:
                    cost += 30.0 * (1.0 - near / (MEMORY * SR)) * over * (chunk == k)
            if pieces:
                t, e = pieces[-1][0], pieces[-1][2] // HOP
                cost += _dist(preps[t]['S'][:, e], S, live)
                cost += 50.0 * ((T == t) & (np.abs(Fr * HOP - pieces[-1][2]) < 2 * SR))   # not where it just was
            if final and loop:
                last = np.minimum((Fr * HOP + want) // HOP, np.array([preps[i]['S'].shape[1] - 1 for i in T]))
                cost += np.where([preps[i]['ok'][e] for i, e in zip(T, last)], 0.0, 20.0)
                if first is not None:                      # the last piece runs into the first: alike there too
                    Se = np.stack([preps[i]['S'][:, e] for i, e in zip(T, last)], axis=1)
                    cost += _dist(preps[first[0]]['S'][:, first[1]], Se, live)
            if (np.isfinite(cost) & ~hard).any():
                break
        if (np.isfinite(cost) & ~hard).any():          # a fifth of a second of the same samples: only if nothing else
            cost[hard] = np.inf
        if not np.isfinite(cost).any():
            raise ValueError('the takes are too short for this texture')
        if not pieces:
            j = int(r.choice(np.nonzero(np.isfinite(cost))[0], p=_soft(-cost[np.isfinite(cost)])))
        else:
            j = _choose(cost, r)
        t, s0 = int(T[j]), int(Fr[j]) * HOP
        if final:
            s1 = s0 + want
        else:                                          # the end: an allowed frame near the length drawn
            p = preps[t]
            fr = np.arange((s0 + lo) // HOP, min((s0 + min(hi, rem - lo)) // HOP, (p['n'] - int(
                (MARGIN + FADE[1]) * SR)) // HOP) + 1)
            fr = fr[(fr * HOP - s0 >= lo) & (fr * HOP - s0 <= rem - lo)]
            good = fr[p['ok'][fr]] if len(fr) else fr
            pick = good if len(good) else fr
            if not len(pick):
                pick = np.array([(s0 + want) // HOP])
            s1 = int(pick[np.argmin(np.abs(pick * HOP - s0 - want))]) * HOP
        if first is None:
            first = (t, s0 // HOP)
        pieces.append((t, s0, s1))
        used.append((t, preps[t]['chunk'], s0, s1, pos, pos + s1 - s0))
        pos += s1 - s0
    return pieces


def _soft(score, temp=1.0):
    w = np.exp((score - score.max()) / temp)
    return w / w.sum()


def _wave_plan(preps, n, loop):
    """whole waves in the order of the takes, each joint placed within 0.6 s of the two troughs where the end of
    one wave and the start of the next are most alike (and the waves come nearest to n in all; stitch() resamples
    the rest away)"""
    tr = [_troughs(p) for p in preps]
    allF = np.concatenate([p['F'].mean(axis=1, keepdims=True) for p in preps], axis=1).mean(axis=1)
    live = np.tile(allF > allF.max() - 45.0, 2)
    m = int(np.ceil((MARGIN + FADE[1] / 2) * SR / HOP)) + 1
    k = len(preps)
    starts, ends = [a for a, _ in tr], [b for _, b in tr]
    nominal = sum(b - a for a, b in tr) * HOP
    for i in range(k if loop else k - 1):
        j = (i + 1) % k
        pa, pb = preps[i], preps[j]
        e = np.arange(max(m, ends[i] - 60), min(pa['F'].shape[1] - m, ends[i] + 61))
        s = np.arange(max(m, starts[j] - 60), min(pb['F'].shape[1] - m, starts[j] + 61))
        if j == 0 and k > 1:
            s = np.array([starts[0]])                  # the first wave's start is already joined to the one before
        D = np.sqrt(np.mean((pa['S'][live][:, e][:, :, None] - pb['S'][live][:, s][:, None, :]) ** 2, axis=0))
        D += 20.0 * np.abs((e[:, None] - ends[i]) - (s[None, :] - starts[j])) * HOP / SR / max(1, k)
        if loop and j == 0:                            # the closing joint also brings the total towards n
            D += 3.0 * np.abs(nominal + (e[:, None] - ends[i] - (s[None, :] - starts[j])) * HOP - n) / SR
        x, y = np.unravel_index(int(np.argmin(D)), D.shape)
        ends[i], starts[j] = int(e[x]), int(s[y])
    return [(i, starts[i] * HOP, ends[i] * HOP) for i in range(k)]


# ------------------------------------------------------------------------------------------------ stitching
def stitch(tks, dur, r, loop=True, periodic=False, energy=0.0, piece=PIECE):
    """the takes stitched into a bed of exactly round(dur * SR) samples (stereo float64): a seamless loop, or
    (loop=False) a stretch faded in over 0.5 s and out over 1 s. energy -1 .. 1: calmer or busier pieces first.
    Returns (y, info): info['joints'] = [(output sample, fade samples, event under the fade)], info['own'] =
    the jumps measured inside the takes (seams() compares the joints with these)"""
    n = int(round(dur * SR))
    preps = [_prep(t) for t in tks]
    pieces = _wave_plan(preps, n, loop) if periodic else _plan(preps, n, r, loop, float(energy), piece)
    total = sum(b - a for _, a, b in pieces)
    out = np.zeros((total, 2))
    k = len(pieces)
    fades = [int(r.uniform(*FADE) * SR) // 2 * 2 for _ in range(k)]   # fades[i]: the joint after piece i
    joints, at = [], 0
    for i, (t, a, b) in enumerate(pieces):
        y = tks[t]['y']
        fin = fades[i - 1] if (i > 0 or loop) else 0
        fout = fades[i] if (i < k - 1 or loop) else 0
        seg = y[a - fin // 2:b + fout // 2]
        g = np.ones(len(seg))
        if fin:
            g[:fin] = np.sin(np.linspace(0.0, np.pi / 2, fin))
        if fout:
            g[len(seg) - fout:] = np.cos(np.linspace(0.0, np.pi / 2, fout))
        idx = (at - fin // 2 + np.arange(len(seg))) % total       # a loop's first and last fades go round
        if len(seg) > total:
            np.add.at(out, idx, seg * g[:, None])
        else:
            out[idx] += seg * g[:, None]
        if fout:
            nt, na, _ = pieces[(i + 1) % k]
            A = y[b - fout // 2:b + fout // 2].mean(axis=1)
            B = tks[nt]['y'][na - fout // 2:na + fout // 2].mean(axis=1)
            rho = float(np.sum(A * B) / (np.sqrt(np.sum(A * A) * np.sum(B * B)) + 1e-12))
            th = np.linspace(0.0, np.pi / 2, fout)
            out[(at + b - a - fout // 2 + np.arange(fout)) % total] /= np.sqrt(
                1.0 + 2.0 * max(rho, 0.0) * np.cos(th) * np.sin(th))[:, None]
            joints.append(((at + b - a) % total, fout, _under(preps[t], b, fout) or _under(preps[nt], na, fout)))
        at += b - a
    if total != n:          # waves that came out a little long or short: the whole bed resampled (circularly)
        from scipy import signal
        out = signal.resample(out, n, axis=0)
        joints = [(int(j * n / total), f, e) for j, f, e in joints]
    if not loop:
        out = dsp.fade(out, min(0.5, dur / 4), min(1.0, dur / 4))
    return out, {'joints': joints, 'own': _own(preps, r), 'loop': loop, 'pieces': pieces}


def _under(p, s, f):
    """whether an event lies under a fade of f samples centred on sample s of a take"""
    a, b = max(0, (s - f // 2) // HOP - 1), min(len(p['ok']), (s + f // 2) // HOP + 2)
    return bool(p['ev'][a:b].any())


def _own(preps, r, count=200):
    """the texture's own variation: the jumps of seams() at random places inside the takes"""
    js, ls = [], []
    for _ in range(count):
        p = preps[int(r.integers(len(preps)))]
        m = int((MARGIN + 0.3) * SR / HOP)
        if p['F'].shape[1] <= 2 * m + 1:
            continue
        at = int(r.integers(m, p['F'].shape[1] - m))
        g = int(r.uniform(*FADE) * SR / HOP)
        a, b = _jumps(p['F'], [at], [g])
        js += list(a)
        ls += list(b)
    js, ls = np.array(js), np.array(ls)
    return {'mean': float(js.mean()), 'p95': float(np.percentile(js, 95)), 'p99': float(np.percentile(js, 99)),
            'level_mean': float(ls.mean()), 'level_p99': float(np.percentile(ls, 99))}


def seams(y, info):
    """the joints of a stitched bed against the texture's own variation: {'jumps', 'levels' (dB, per joint),
    'ratio' (mean jump over the mean jump inside the takes), 'worst' (largest jump over their 99th percentile),
    'level_worst', 'events' (joints with an event under the fade)}. A loop is measured round its seam."""
    y = np.asarray(y, np.float64)
    pad = int(0.5 * SR)
    z = np.concatenate([y[-pad:], y, y[:pad]]) if info['loop'] else np.pad(y, ((pad, pad), (0, 0)))
    F = features(z)
    at = [(j + pad) / HOP for j, _, _ in info['joints']]
    gap = [f / HOP for _, f, _ in info['joints']]
    js, ls = _jumps(F, at, gap)
    own = info['own']
    return {'jumps': js.round(2).tolist(), 'levels': ls.round(2).tolist(),
            'ratio': float(js.mean() / own['mean']) if len(js) else 0.0,
            'worst': float(js.max() / own['p99']) if len(js) else 0.0,
            'level_worst': float(ls.max() / max(own['level_p99'], 0.5)) if len(ls) else 0.0,
            'events': int(sum(e for _, _, e in info['joints']))}


def repeats(y, stretch=2.0, lags=(2.0, MEMORY), loop=False):
    """whether a stretch of `stretch` s comes back lags[0]..lags[1] s later, two ways. 'max': the closest two
    stretches by their band envelopes (24 bands at 50 ms, each band's own mean within the stretch taken out, the
    rest correlated): two takes of one stretch of a model read 0.9, a stretch played twice 1, different moments of
    a texture 0.1-0.4. 'copy': the largest normalised cross-correlation of the waveform of half-second stretches
    with anything 1-20 s away (a stretch played twice reads 1, fresh noise under 0.1). A loop is compared round its
    seam, at lags shorter than itself. {'max', 'lag' (s), 'p99', 'copy'}"""
    y = np.asarray(y, np.float64)
    n = len(y)
    F = features(y)
    k = F.shape[1] // 5
    E = F[:, :k * 5].reshape(F.shape[0], k, 5).mean(axis=2)
    E = E[E.mean(axis=1) > E.mean(axis=1).max() - 45.0]
    w, hi = int(round(stretch * 20)), int(round(lags[1] * 20))
    if loop:
        hi = min(hi, k - w)
        E = np.concatenate([E, E[:, :w + hi]], axis=1)
    starts = np.arange(0, E.shape[1] - w + 1, 2)
    V = np.stack([E[:, s:s + w] - E[:, s:s + w].mean(axis=1, keepdims=True) for s in starts]).reshape(len(starts), -1)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    d = starts[None, :] - starts[:, None]
    sel = (d >= int(lags[0] * 20)) & (d <= hi) & ((starts[:, None] < k) if loop else True)
    out = {'max': 0.0, 'lag': 0.0, 'p99': 0.0, 'copy': _copy(y, lags[1], loop)}
    if sel.any():
        vals = (V @ V.T)[sel]
        i = int(np.argmax(vals))
        out.update(max=float(vals[i]), lag=float(d[sel][i] / 20.0), p99=float(np.percentile(vals, 99)))
    return out


def _copy(y, far, loop, rate=12000):
    """the largest normalised cross-correlation between a half-second stretch of y (mono, at `rate` Hz, whitened
    over the whole file and without the band under 200 Hz: a narrow band has few degrees of freedom and two
    stretches of it correlate by chance) and y 1 .. far s later"""
    from scipy import signal
    x = signal.resample_poly(np.asarray(y, np.float64).mean(axis=1), rate, SR)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    x = np.fft.irfft(X / (ndimage.uniform_filter1d(np.abs(X), max(3, len(X) // 400)) + 1e-12) * (f > 200.0), len(x))
    m, h = len(x), rate // 2
    z = np.concatenate([x, x[:int(far * rate) + h]]) if loop else x
    best = 0.0
    nf = 1 << int(np.ceil(np.log2(len(z) + h)))
    Z = np.fft.rfft(z, nf)
    e = np.concatenate([[0.0], np.cumsum(z * z)])
    for s in range(0, m - h, rate):
        c = np.fft.irfft(Z * np.conj(np.fft.rfft(z[s:s + h], nf)), nf)[:len(z) - h + 1]
        lag = np.arange(len(c)) - s
        ok = (lag >= rate) & (lag <= far * rate)
        if not ok.any():
            continue
        en = np.sqrt((e[h:len(z) + 1] - e[:len(z) - h + 1]) * (e[s + h] - e[s])) + 1e-12
        best = max(best, float(np.max(np.abs(c[ok]) / en[ok])))
    return best
