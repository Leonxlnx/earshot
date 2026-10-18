"""FROZEN copy of earshot/forge/tone.py (profile, shares, t40_ms, ToneRef) as of branch forge-v2 round 0 (git HEAD
108763c), so the forge benchmark measures the same thing every round while the forge's own tone.py evolves.

    from frozen_tone import ToneRef, profile, t40_ms      (eval/forge/bench.py imports it from this folder)

Do not edit: a change here changes every benchmark number. Add new measurements in bench.py instead (and record them in
docs/history/FORGE_V2_LOG.md). The original docstring follows.

Model-free ears for the forge: tonal balance and tail, measured straight from the samples.

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
        from earshot import audio
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
