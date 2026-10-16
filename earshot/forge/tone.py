"""Model-free ears for the forge: tonal balance and tail, measured straight from the samples.

    from earshot.forge.tone import ToneRef
    tr = ToneRef(['real_01.wav', 'real_02.wav', ...])     # files or 48 kHz arrays
    m = tr.check(x)
    m['tone_db'], m['tone_ratio'], m['words']    # e.g. 'tone 11.2 dB from the refs (they sit 3.1-6.0 apart):
                                                 #  +16 dB at 500-630 Hz (above every ref), -14 dB at 10 kHz (below ...)'

Why this exists: earshot's model signals saturate. After a search tunes a sound against CLAP (match z, typicality) and
the realness head, the sound lands inside or above the real recordings on every model check, including held-out
phrasings, the AudioSet tagger and independent real sets (demo/forge-test, the judge's re-scoring). The models can no
longer tell made from real, but a 1/3-octave measurement still can (a key press 8-13 dB off the real keys' balance
while the real keys sit 3-6 dB from each other). These numbers cannot be pleased by a room and a mic noise floor.

Measures (mono sum, the whole file):
  profile    power per 1/3-octave band, 31.5 Hz - 16 kHz (28 bands), in dB relative to the file's total power
             (so level never decides), floored at -60 dB
  tone_db    RMS over bands of (profile - the refs' median profile)
  tone_ref   the refs' own distances, each to the median of the OTHER refs (leave-one-out): what "real" spreads to
  tone_ratio tone_db / median(tone_ref): 1 = as far from the median as a typical real recording
  outside    bands where the sound is beyond the refs' min / max by more than `margin` dB (the words name the worst)
  t40_ms     time from the peak until the 5 ms RMS envelope first falls 40 dB under it (tail length), with the refs'
             range; length_s the file length
  shares     energy share of 5 broad regions (<150 Hz, 150 Hz-1 kHz, 1-4, 4-12, >12 kHz) in dB of the total, with the
             refs' median share; region_penalty() counts the excess beyond median + 6 dB (a low puff, a bright hiss)

Spectro-temporal envelope (EnvRef): the average spectrum above cannot hear a tail that dies too fast, a swell where
the refs hit, or a band that rings on after the rest has gone (the blind test caught the forged impact by its tail).

    from earshot.forge.tone import EnvRef
    er = EnvRef(['real_01.wav', ...])
    m = er.check(x)
    m['env_db'], m['env_ratio'], m['env_out'], m['words']   # e.g. 'envelope 9.8 dB from the nearest ref
        # kb_space_01.wav (the refs sit 6.3-8.1 apart); beyond every ref: tail 4-6.3 kHz +14 dB, tail 1.6-3.2 kHz +12 dB'
    EnvRef.penalty(m)
  bandgram   28 third-octave band energies (the CEN bands, 3rd-order Butterworth, so low bands smear in time like an
             ear's) in 10 ms frames, dB re the loudest frame's total power, floored at -70 dB
  env_dist   the made sound's bandgram against one ref's: aligned at the onset (first frame within 20 dB of the
             peak frame) or at the peak frame, whichever fits better, with a small tolerance (time-stretch 0.85-1.18
             about the anchor, then a shift of +-30 ms around the best; stretch costs a little). Distance = mean
             |dB difference| over every band x frame of both active spans (cells weighted by how loud the louder of
             the two is, so a missing tail or an extra swell counts, silence does not; each cell's error capped at
             30 dB). The tolerance is deliberately small: a tail half as long as the ref's cannot be stretched away.
  env_db     distance to the NEAREST ref; env_ref: each ref's distance to its nearest other ref (what real spreads
             to); env_ratio = env_db / median(env_ref). Nearest-ref distances flatter averages: with few, diverse refs
             a sound in their middle sits closer to each than they sit to each other (the v1 pops, whooshes and
             impacts do), so env_ratio alone cannot find a gap. Hence:
  cells      per segment (pre < -20 ms, attack -20..30 ms, body 30-200 ms, tail 0.2-1 s, late > 1 s from the anchor)
             x 8 band groups (31-80 Hz ... 12.5-16 kHz): the signed weighted difference (made - ref) and the cell's
             weight share, against each ref at its own best alignment
  outside    cells where the sound is beyond EVERY kept ref in the same direction by more than max(6 dB, the refs'
             own median pairwise difference in that cell): a consistent gap that no real recording shares, in a cell
             where the refs agree with each other. env_out = the sum of the excess dB beyond that tolerance, scaled
             by min(1, share / 5 %); env_out_ref = the refs' own median env_out (each against the others; the cell
             spreads use all refs, a small leak in leave-one-out scoring): diverse real refs are each unique
             somewhere, so only the excess over it counts
"""
import numpy as np

SR = 48000
CEN = 1000 * 2 ** (np.arange(-15, 13) / 3)          # 31.5 Hz .. 15.9 kHz
EDGE = 2 ** (1 / 6)
REGIONS = {'<150': (0, 150), '150-1k': (150, 1000), '1-4k': (1000, 4000), '4-12k': (4000, 12000), '>12k': (12000, 24000)}


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _load(x):
    if isinstance(x, str):
        from .. import audio
        return audio.load(x)
    return x


def label(c):
    return f'{c / 1000:.3g} kHz' if c >= 1000 else f'{c:.0f} Hz'


def profile(x):
    """1/3-octave power profile in dB relative to the total power (28 bands, floored at -60)"""
    m = _mono(x)
    F = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    p = np.array([F[(f >= c / EDGE) & (f < c * EDGE)].sum() for c in CEN]) + 1e-20
    return np.maximum(10 * np.log10(p / p.sum()), -60.0)


def shares(x):
    """energy share of each broad region (REGIONS) in dB of the total"""
    m = _mono(x)
    F = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    tot = F.sum() + 1e-20
    return {k: round(float(10 * np.log10(F[(f >= a) & (f < b)].sum() / tot + 1e-12)), 1) for k, (a, b) in REGIONS.items()}


def t40_ms(x, db=40.0, ms=5.0):
    """ms from the peak of the 5 ms RMS envelope until it first falls `db` under it"""
    m = _mono(x)
    n = int(SR * ms / 1000)
    k = len(m) // n
    if k < 2:
        return 0.0
    e = 20 * np.log10(np.sqrt((m[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-12)
    i = int(np.argmax(e))
    after = e[i:] - e[i]
    idx = np.nonzero(after < -db)[0]
    return float((idx[0] if len(idx) else len(after)) * ms)


class ToneRef:
    """the real references' tonal balance and tail, and a check of any sound against them"""

    def __init__(self, refs, margin=3.0):
        xs = [_load(r) for r in refs]
        self.n = len(xs)
        self.margin = margin
        self.P = np.stack([profile(x) for x in xs]) if xs else np.zeros((0, len(CEN)))
        self.med = np.median(self.P, 0) if xs else None
        self.lo, self.hi = (self.P.min(0), self.P.max(0)) if xs else (None, None)
        self.t40 = np.array([t40_ms(x) for x in xs])
        self.S = [shares(x) for x in xs]
        self.len_s = np.array([len(_mono(x)) / SR for x in xs])
        self.ref_dist = np.array([self._loo(i) for i in range(self.n)]) if self.n >= 3 else np.zeros(0)

    def _loo(self, i):
        keep = [j for j in range(self.n) if j != i]
        return float(np.sqrt(np.mean((self.P[i] - np.median(self.P[keep], 0)) ** 2)))

    @property
    def spread(self):
        """(median, max) of the refs' leave-one-out distances, or None with fewer than 3 refs"""
        if not len(self.ref_dist):
            return None
        return float(np.median(self.ref_dist)), float(self.ref_dist.max())

    def check(self, x, exclude=None):
        """measure one sound against the refs. exclude = index of a ref to leave out (scoring a ref fairly)"""
        x = _load(x)
        if self.n == 0:
            return {}
        keep = [j for j in range(self.n) if j != exclude]
        if not keep:                       # a single ref scored leave-one-out: nothing left to compare with
            return {}
        P = self.P[keep]
        med, lo, hi = np.median(P, 0), P.min(0), P.max(0)
        p = profile(x)
        dev = p - med
        dist = float(np.sqrt(np.mean(dev ** 2)))
        sp = self.ref_dist[keep] if len(self.ref_dist) else np.zeros(0)
        ratio = dist / max(float(np.median(sp)), 0.5) if len(sp) >= 2 else None
        over = np.maximum(p - hi - self.margin, 0) - np.maximum(lo - p - self.margin, 0)
        outside = [(label(c), round(float(d), 1), 'above' if o > 0 else 'below')
                   for c, d, o in zip(CEN, dev, over) if o != 0]
        t40 = t40_ms(x)
        t40r = self.t40[keep]
        out = {'tone_db': round(dist, 2), 'tone_ratio': None if ratio is None else round(ratio, 2),
               'tone_ref': [round(float(v), 2) for v in sp], 'band_dev': [round(float(v), 1) for v in dev],
               'outside': outside, 't40_ms': t40, 't40_ref': (float(t40r.min()), float(t40r.max())),
               'length_s': round(len(_mono(x)) / SR, 3), 'shares': shares(x),
               'shares_ref_med': {k: round(float(np.median([self.S[j][k] for j in keep])), 1) for k in REGIONS},
               'length_ref': (round(float(self.len_s[keep].min()), 3), round(float(self.len_s[keep].max()), 3))}
        out['words'] = self.words(out)
        return out

    @staticmethod
    def words(m):
        sp = m['tone_ref']
        s = f"tone {m['tone_db']:.1f} dB from the refs' median"
        if sp:
            s += f" (the refs sit {min(sp):.1f}-{max(sp):.1f} apart)"
        worst = sorted(m['outside'], key=lambda o: -abs(o[1]))[:3]
        if worst:
            s += ': ' + ', '.join(f'{d:+.0f} dB at {b} ({w} every ref)' for b, d, w in worst)
        a, b = m['t40_ref']
        if m['t40_ms'] > 1.5 * b + 10 or m['t40_ms'] < 0.6 * a - 5:
            s += f"; tail t40 {m['t40_ms']:.0f} ms (refs {a:.0f}-{b:.0f})"
        loud = [f"{k} {m['shares'][k] - m['shares_ref_med'][k]:+.0f} dB" for k in REGIONS
                if abs(m['shares'][k] - m['shares_ref_med'][k]) > 9]
        if loud:
            s += '; energy share vs the refs: ' + ', '.join(loud)
        lo, hi = m['length_ref']
        if m['length_s'] > 1.3 * hi:
            s += f"; {m['length_s']:.2f} s long (refs {lo:.2f}-{hi:.2f})"
        return s

    @staticmethod
    def region_penalty(m, regions=('<150', '4-12k', '>12k'), margin=6.0, scale=6.0):
        """model-free check of broad regions: 1 per `scale` dB that a region's energy share exceeds the refs' median
        share by more than `margin` dB (a low puff or a bright hiss / tick on a mid-band sound)"""
        return sum(max(0.0, m['shares'][k] - m['shares_ref_med'][k] - margin) / scale for k in regions)

    def penalty(self, m):
        """0 while the sound is no farther from the refs' median than a typical real recording, then linear in the
        excess (tone_ratio - 1), plus a share of the tail excess beyond the refs' longest t40"""
        if not m or m.get('tone_ratio') is None:
            return 0.0
        pen = max(0.0, m['tone_ratio'] - 1.0)
        a, b = m['t40_ref']
        if b > 0 and m['t40_ms'] > b:
            pen += 0.5 * min(2.0, (m['t40_ms'] - b) / b)
        return pen



# ------------------------------------------------------------------------------- spectro-temporal envelope
FLOOR = -70.0
HOP_MS = 10.0
STRETCH = (0.85, 0.92, 1.0, 1.08, 1.18)
SHIFT = (-3, -2, -1, 0, 1, 2, 3)                     # frames (10 ms)
SEGMENTS = (('pre', -1e9, -2), ('attack', -2, 3), ('body', 3, 20), ('tail', 20, 100), ('late', 100, 1e9))  # frames
GROUPS = (('31-80 Hz', 0, 90), ('100-200 Hz', 90, 224), ('250-500 Hz', 224, 560), ('630-1250 Hz', 560, 1400),
          ('1.6-3.2 kHz', 1400, 3550), ('4-6.3 kHz', 3550, 7100), ('8-10 kHz', 7100, 11200), ('12.5-16 kHz', 11200, 1e9))
OUT_MARGIN = 6.0
_SOS = []


def _bank():
    if not _SOS:
        from scipy import signal
        for c in CEN:
            _SOS.append(signal.butter(3, [c / EDGE, min(c * EDGE, 0.49 * SR)], 'bandpass', fs=SR, output='sos'))
    return _SOS


def bandgram(x, hop_ms=HOP_MS):
    """third-octave band energy over time: (frames, 28) dB re the loudest frame's total power, floored at FLOOR"""
    from scipy import signal
    m = _mono(x)
    n = int(SR * hop_ms / 1000)
    k = max(1, -(-len(m) // n))
    m = np.pad(m, (0, k * n - len(m)))
    G = np.empty((k, len(CEN)))
    for j, sos in enumerate(_bank()):
        G[:, j] = (signal.sosfilt(sos, m).reshape(k, n) ** 2).mean(1)
    return np.maximum(10 * np.log10(G / (G.sum(1).max() + 1e-30) + 1e-30), FLOOR)


def _total(G):
    return 10 * np.log10((10 ** (G / 10)).sum(1) + 1e-30)


def _anchors(G):
    tot = _total(G)
    ip = int(np.argmax(tot))
    on = int(np.nonzero(tot >= tot[ip] - 20)[0][0])
    act = np.nonzero(tot >= tot[ip] - 50)[0]
    return {'onset': on, 'peak': ip}, int(act[0]), int(act[-1])


def _rows(G, t):
    """G sampled at fractional frames t (linear in dB), FLOOR outside"""
    i0 = np.floor(t).astype(int)
    fr = (t - i0)[:, None]
    n = len(G)
    a = np.where(((i0 >= 0) & (i0 < n))[:, None], G[np.clip(i0, 0, n - 1)], FLOOR)
    b = np.where(((i0 + 1 >= 0) & (i0 + 1 < n))[:, None], G[np.clip(i0 + 1, 0, n - 1)], FLOOR)
    return a * (1 - fr) + b * fr


def env_dist(A, B, stretch=STRETCH, shift=SHIFT, detail=False):
    """distance in dB between two bandgrams (A = the sound judged, B = a ref), best over the anchors / stretch /
    shift tolerance; detail=True also returns the alignment and, per segment x band group, the signed weighted mean
    difference (A - B, dB) and the cell's share of the weight: cells {'tail 630-1250 Hz': (diff, share)}"""
    ancA, a_lo, a_hi = _anchors(A)
    ancB, b_lo, b_hi = _anchors(B)
    frames = {}
    for name in ('onset', 'peak'):
        a0, b0 = ancA[name], ancB[name]
        lo = max(-300, min(a_lo - a0, b_lo - b0)) - 3
        hi = min(600, max(a_hi - a0, b_hi - b0)) + 4
        tt = np.arange(lo, hi, dtype=float)
        frames[name] = (a0, tt, _rows(B, b0 + tt))

    def cost(name, s, sh):
        a0, tt, Bv = frames[name]
        Av = _rows(A, a0 + tt / s + sh)
        w = np.clip((np.maximum(Av, Bv) - FLOOR) / 40.0, 0.0, 1.0)           # how loud the louder of the two is
        d = float((w * np.minimum(np.abs(Av - Bv), 30.0)).sum() / (w.sum() + 1e-9))
        return d * (1.0 + 0.5 * abs(np.log(s))), name, s, sh, tt, Av, Bv, w  # stretching costs a little

    best = min((cost(n, s, 0) for n in frames for s in stretch), key=lambda c: c[0])    # coarse: no shift
    best = min([best] + [cost(best[1], best[2], sh) for sh in shift if sh], key=lambda c: c[0])   # then shift
    d, name, s, sh, tt, Av, Bv, w = best
    if not detail:
        return d
    cells, W = {}, w.sum() + 1e-9
    for seg, t0, t1 in SEGMENTS:
        rs = (tt >= t0) & (tt < t1)
        for grp, f0, f1 in GROUPS:
            cs = (CEN >= f0) & (CEN < f1)
            ww = w[np.ix_(rs, cs)]
            if ww.sum() < 0.01 * W:
                continue
            diff = float((ww * (Av[np.ix_(rs, cs)] - Bv[np.ix_(rs, cs)])).sum() / ww.sum())
            cells[f'{seg} {grp}'] = (round(diff, 1), round(float(ww.sum() / W), 3))
    return d, {'anchor': name, 'stretch': s, 'shift_ms': sh * HOP_MS, 'cells': cells}


class EnvRef:
    """the real references' band-energy trajectories, and a check of any sound's shape over time against them"""

    def __init__(self, refs, names=None, margin=OUT_MARGIN):
        xs = [_load(r) for r in refs]
        self.names = list(names) if names is not None else [r if isinstance(r, str) else f'ref{i}' for i, r in
                                                            enumerate(refs)]
        self.margin = margin
        self.G = [bandgram(x) for x in xs]
        self.n = len(self.G)
        self.D = np.full((self.n, self.n), np.inf)        # D[i, j] = env_dist(ref i judged against ref j)
        det = {(i, j): env_dist(self.G[i], self.G[j], detail=True)
               for i in range(self.n) for j in range(self.n) if i != j}
        for (i, j), r in det.items():
            self.D[i, j] = r[0]
        pair = {}                                         # how far the refs sit from each other, cell by cell
        for r in det.values():
            for key, (v, _) in r[1]['cells'].items():
                pair.setdefault(key, []).append(abs(v))
        self.cell_spread = {key: float(np.median(v)) for key, v in pair.items() if len(v) >= 2}
        self.out_ref = np.array([self._outside([det[i, j] for j in range(self.n) if j != i])[1] for i in range(self.n)])

    def _outside(self, res):
        """cells beyond EVERY compared ref in one direction by more than the margin, and their weighted excess"""
        outside, out_db = [], 0.0
        spread = getattr(self, 'cell_spread', {})
        if len(res) >= 2:
            for key in res[0][1]['cells']:
                got = [r[1]['cells'].get(key) for r in res]
                if any(g is None for g in got):
                    continue
                diffs = np.array([g[0] for g in got])
                share = float(np.mean([g[1] for g in got]))
                lo = diffs.min() if diffs.min() > 0 else (-diffs.max() if diffs.max() < 0 else 0.0)
                tol = max(self.margin, spread.get(key, self.margin))     # where the refs disagree, allow as much
                if lo > tol and share >= 0.02:
                    outside.append((key, round(float(np.sign(diffs[0]) * lo), 1), round(share, 3)))
                    out_db += (lo - tol) * min(1.0, share / 0.05)
        return sorted(outside, key=lambda o: -abs(o[1]) * o[2]), float(out_db)

    def spread(self, exclude=None):
        """each kept ref's distance to its nearest OTHER kept ref (needs 3+ refs after the exclusion)"""
        keep = [j for j in range(self.n) if j != exclude]
        if len(keep) < 3:
            return np.zeros(0)
        return np.array([min(self.D[i, j] for j in keep if j != i) for i in keep])

    def check(self, x, exclude=None, G=None):
        """measure one sound (or its precomputed bandgram G) against the refs; exclude = a ref index to leave out"""
        keep = [j for j in range(self.n) if j != exclude]
        if not keep:
            return {}
        G = bandgram(_load(x)) if G is None else G
        res = [env_dist(G, self.G[j], detail=True) for j in keep]
        k = int(np.argmin([r[0] for r in res]))
        d, info = res[k]
        sp = self.spread(exclude)
        ratio = d / max(float(np.median(sp)), 1.0) if len(sp) else None
        outside, out_db = self._outside(res)
        oref = self.out_ref[keep] if self.n >= 3 else np.zeros(0)
        out = {'env_db': round(d, 2), 'env_ratio': None if ratio is None else round(ratio, 2),
               'env_ref': [round(float(v), 2) for v in sp],
               'nearest': self.names[keep[k]].replace(chr(92), '/').rsplit('/', 1)[-1], 'nearest_i': keep[k],
               'anchor': info['anchor'], 'stretch': info['stretch'], 'shift_ms': info['shift_ms'],
               'cells': info['cells'], 'outside': outside, 'env_out': round(out_db, 2),
               'env_out_ref': round(float(np.median(oref)), 2) if len(oref) else None}
        out['words'] = self.words(out)
        return out

    @staticmethod
    def words(m):
        s = f"envelope {m['env_db']:.1f} dB from the nearest ref {m['nearest']}"
        if m['env_ref']:
            s += f" (the refs sit {min(m['env_ref']):.1f}-{max(m['env_ref']):.1f} apart)"
        if m['outside']:
            s += '; beyond every ref: ' + ', '.join(f'{k} {v:+.0f} dB' for k, v, _ in m['outside'][:3])
        else:
            worst = sorted(((k, v) for k, v in m['cells'].items() if abs(v[0]) >= 8),
                           key=lambda kv: -abs(kv[1][0]) * kv[1][1] ** 0.5)[:2]
            if worst:
                s += ': vs it ' + ', '.join(f'{k} {v[0]:+.0f} dB' for k, v in worst)
        return s

    @staticmethod
    def penalty(m, cap=2.0):
        """nearest part: 0 while the sound is no farther from its nearest ref than a typical ref from its nearest
        neighbour, then linear in (env_ratio - 1); outside part: (env_out - max(3, the refs' own median env_out)) / 6,
        so with diverse refs (each real one is unique somewhere) only an excess over what real recordings show
        counts (6 dB beyond every ref in a cell holding 5 % or more of the weight = 1). Capped."""
        if not m:
            return 0.0
        near = max(0.0, m['env_ratio'] - 1.0) if m.get('env_ratio') is not None else 0.0
        tol = max(3.0, m.get('env_out_ref') or 0.0)
        return float(min(cap, near + max(0.0, m.get('env_out', 0.0) - tol) / 6.0))
