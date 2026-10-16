"""Artifact detectors and polish measures for sound effects, measured straight from the samples (no model).

    from earshot.forge.polish import PolishRef, measure, loudness_spread
    m = measure(x)                           # every detector's raw number for one 48 kHz sound (mono or stereo)
    pr = PolishRef(['ref_01.wav', ...])      # the refs' own numbers loosen the tolerances (PolishRef([]): absolute)
    c = pr.check(x)
    c['art_pen'], c['polish_pen'], c['hard'], c['words']   # e.g. 'artifacts: 2 clicks (0.212 s, 0.305 s); polish:
                                                           #  attack 41 ms (refs 1-9), crest 19.5 dB (refs 11-15)'
    loudness_spread([x1, x2, x3])            # do the variants of one sound play at one loudness?

Why: a forged sound effect is judged in a motion-design edit next to pro library sounds. A click from a hard gate, a
foldover whistle on a sweep, a 3 kHz ring, a tail chopped at -30 dB or a spectrum frozen while the refs move are heard
at once and none of earshot's model checks reliably hears them (they were trained to tell real from synthetic, not
good from broken). Each detector below is a separate function returning plain data.

Artifacts (`measure` keys):
  clicks      isolated discontinuities inside the sound: 1 ms frames of the 2nd difference (+12 dB/oct, so a jump
              stands far above smooth content) more than 18 dB above both neighbourhoods while the neighbourhoods
              agree within 8 dB, with 60 % or more of that frame's energy in 2 samples (a jump, not a physical tick,
              which spreads over a fraction of a ms); an onset, where the sound starts and stays, is not a click and
              the loudest transient +-3 ms is the sound's own attack. edge_db: the first / last sample re the peak (a file
              that starts or ends off zero clicks when played)
  fold        aliasing / foldover: in clearly harmonic frames (4096-point STFT; >= 70 % of the 60 Hz - 4 kHz energy
              within +-2 bins of one harmonic grid, f0 150 Hz - 3 kHz, refined by least squares on the low partials),
              the share of the 4-20 kHz energy in spectral PEAKS where a naive oscillator's partials above Nyquist fold
              back (k f0 mirrored at multiples of 48 kHz, off the grid). A peak counts when it stands 10 dB over the
              local floor (median over ~750 Hz) and over the neighbouring harmonic's window leakage, is no louder than
              6 dB over the grid's strongest partial in the band's top half (an alias of partial k is weaker than the
              partials below it), and when half of the first-fold positions in that frame show one (a family). Each
              window is widened by its sweep smear (from the f0 change between frames); harmonics whose window would
              exceed half the spacing are not judged. Round 2 counted ALL off-grid energy (aliasing(legacy=True)): a
              noise layer under a harmonic tone (downlifter synth_fall 0.47) and inharmonic bell partials over a
              chance grid (notify / shimmer / logo_sting 0.4-0.99) read as aliasing; they now measure 0-0.08 while a
              naive 440 Hz saw keeps 0.057 (FM run without oversampling, logo_sting_01, still 0.53: real foldover).
              Noise, modal and inharmonic sounds have no grid and are not judged.
              nyq_db: energy share above 20 kHz (recordings roll off; naive oscillators and white noise do not)
  dc_db       |mean| re the peak; infra_db: energy share below 15 Hz (a one-sided puff, an offset, drift)
  res_db      the harshest narrow resonance: energy spectrum (4096-point frames) minus its median over +-1/3 octave,
              weighted by where the ear is touchy (1 for 2-6 kHz, 0.7 for 1-2 and 6-9 kHz, 0.5 elsewhere), peaks
              within 30 dB of the spectrum's maximum only; res_hz its frequency
  evolution   how much the spectral SHAPE changes: per 10 ms frame (30 ms smoothed) the octave-group profile re the
              frame's own total (so a fade does not count), change over 50 ms as an RMS over groups weighted by their
              energy share (quiet groups' estimation noise does not count), energy-weighted over active frames (dB);
              centroid_oct: the 10-90 % range of the spectral centroid in octaves. Frozen = below 0.35 x the refs'
              median where the refs move (median >= 1.5 dB); penalty 2 x the relative shortfall
  ring_z      comb-like metallic ringing in the tail (from 30 ms after the peak to -50 dB): cepstral peak of the
              tail's log spectrum (300 Hz - 12 kHz, trend removed) at 2-40 ms quefrency, as a robust z against the
              rest of that range; ring_ms its delay. A dense natural tail has no repeating delay. Not judged (None,
              ring_src 'narrow' / 'pitch', ring_raw_z the raw z): a tail narrower than 2 kHz (a sub drop's few low
              partials over an empty band put z 120-390 at the 2 ms edge) and a tail whose delay is the sound's own
              pitch (its partial levels are smooth, median 2nd difference <= 2 dB, where noise-fed comb teeth scatter
              6-7 dB; or a body around the peak periodic at the same delay, z >= 100: buzzy hums z 35-1000 before)
  trunc_db    a truncated tail: the level re the peak at which the decay was cut (the last 10 ms if the sound was
              still decaying there, or the level before a final fade that falls much faster than the decay before
              it); None when the sound reaches -55 dB or its own floor. Designed mode allows a gate once the level
              is 30 dB down (GATE_DB); gated=True (glitch, riser, zip, pro_impact: the stop is the design) allows it
              from 12 dB down and never makes it hard
  silence_ms  digital silence inside the sound (runs of |x| < 1e-6 of >= 2 ms between the first and last audible
              sample), total ms; a recording always has a floor, a gap of exact zeros is an edit or a gate
  clip_runs   runs of >= 3 equal samples at the peak (flat tops), per channel; overs: samples beyond 1.0
Polish:
  attack_ms   10 -> 90 % rise of the 0.5 ms RMS envelope to its first near-peak (sample-accurate crispness)
  onset_hf_db energy share above 6 kHz in the first 10 ms from the onset (the snap / top end of the attack)
  bump_db     the largest re-rise of the 10 ms envelope above its running minimum after the peak (frames above
              -45 dB): a smooth tail falls monotonically; decay_db_s the fitted decay rate, t40_ms the tail length
  crest_db    peak - the loudest 50 ms of K-weighted RMS (high = thin, spiky, little body at the same peak)
  loudness_spread(xs): momentary-max LUFS (K-weighted, 400 ms) per variant at its own level, spread and std (LU)

Penalties (`PolishRef.check`): RULES give each measure a bad direction, an absolute tolerance, a margin, a scale and a
cap. limit = the absolute tolerance, loosened to the refs' worst value + margin when refs are given (a trait the real
recordings share is not a defect), or for the relative-only measures (attack, onset_hf, t40, crest, evolution) set by
the refs alone (dB margins widen to a quarter of the refs' own range: 3-6 diverse refs pin a range down loosely);
penalty = min(cap, excess / scale) (log2 ratios for times). 'hard' artifacts at penalty >= 0.5 are listed
in check()['hard'] and fail the critic's `passes`; designed=True (glitch, stutter) makes clicks and digital silence
soft at half weight and allows gates (trunc_db, GATE_DB); gated=True loosens the gate further. targets= {polish key:
(lo, hi)} stands in for the refs' range of a relative polish measure when fewer than 2 refs carry it (the critic's
absolute STUDY_TARGETS). art_pen = sum of artifact penalties; polish_pen = sum of polish penalties.
"""
import numpy as np

SR = 48000
K_SOS = np.array([[1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
                  [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]])     # ITU-R BS.1770 K-weighting, 48 kHz


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _chans(x):
    x = np.asarray(x, np.float64)
    return x.T if x.ndim == 2 else x[None]


def _load(x):
    if isinstance(x, str):
        from .. import audio
        return audio.load(x)
    return x


def _fdb(m, n):
    """dB power of consecutive n-sample frames"""
    k = len(m) // n
    if k < 1:
        return np.full(1, -240.0)
    return 10 * np.log10((m[:k * n].reshape(k, n) ** 2).mean(1) + 1e-24)


def _stft(m, nfft, hop):
    if len(m) < nfft:
        m = np.pad(m, (0, nfft - len(m)))
    k = 1 + (len(m) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(k)[:, None]
    return np.abs(np.fft.rfft(m[idx] * np.hanning(nfft), axis=1)) ** 2


# ------------------------------------------------------------------------------------------------ artifacts
def clicks(x, thr=18.0):
    """isolated discontinuities inside the sound (count, times) and the edge jump (dB re peak)"""
    m = _mono(x)
    peak = np.abs(m).max() + 1e-12
    edge = 20 * np.log10(max(abs(m[0]), abs(m[-1])) / peak + 1e-12)
    d2 = np.diff(m, 2)
    e = _fdb(d2, 48)
    lvl = _fdb(m, 48)
    k = min(len(e), len(lvl))
    times = []
    if k >= 24:
        e, lvl = e[:k], lvl[:k]
        ip = int(np.argmax(lvl))
        P = np.pad(e, 12, mode='edge')
        W = np.lib.stride_tricks.sliding_window_view(P, 25)         # W[i] = e[i-12 .. i+12]
        pre, post = np.median(W[:, 2:10], 1), np.median(W[:, 15:23], 1)
        q = np.sort(d2[:k * 48].reshape(k, 48) ** 2, axis=1)
        imp = q[:, -2:].sum(1) / (q.sum(1) + 1e-30)                 # a jump puts its energy in 2 samples
        hit = ((e - np.maximum(pre, post) > thr) & (np.abs(post - pre) < 8) & (lvl > lvl.max() - 70)
               & (np.abs(np.arange(k) - ip) > 3) & (imp > 0.6))
        i = 0
        while i < k:
            if hit[i]:
                j = i
                while j + 1 < k and hit[j + 1]:
                    j += 1
                times.append(round((i + j + 1) / 2 / 1000, 3))
                i = j + 1
            else:
                i += 1
    return {'clicks': len(times), 'click_times': times[:8], 'edge_db': round(float(edge), 1)}


def _f0(P, Pw, f, c0, low, G):
    """harmonic grid of one spectrum frame: (f0 refined by least squares on the low partials, grid share) or None"""
    tl = P[(f >= 60) & (f < 4000)].sum()
    if tl <= 0:
        return None
    on = np.where(low, Pw[G], 0.0).sum(1) / tl - 0.02 * low.sum(1)   # a dense grid covers more by chance
    i = int(np.argmax(on))
    if on[i] < 0.7:
        return None
    df = f[1]
    hs, fs, ws = [], [], []
    for h in range(1, int(4000 / c0[i]) + 1):
        k = int(round(h * c0[i] / df))
        lo, hi = max(1, k - 3), min(len(P) - 2, k + 3)
        j = lo + int(np.argmax(P[lo:hi + 1]))
        a, b, c = np.log(P[j - 1:j + 2] + 1e-30)
        hs.append(h)
        fs.append((j + 0.5 * (a - c) / (a - 2 * b + c - 1e-12)) * df)
        ws.append(P[j])
    hs, fs, ws = np.array(hs, float), np.array(fs), np.array(ws)
    return float((ws * hs * fs).sum() / (ws * hs * hs).sum()), float(on[i])


ALIAS_PEAK_DB = 10.0       # an alias line must stand this far above the local floor (median over ~750 Hz)
ALIAS_FAMILY = 0.5         # ... and this share of the judged alias positions must show one (a family, not a partial)
ALIAS_LOUDEST = 4.0        # ... and be no louder than 6 dB over the grid's strongest partial in the band's top half


def aliasing(x, nfft=4096, hop=1024, frames=64, legacy=False):
    """fold: in clearly harmonic frames, the share of the 4-20 kHz energy in spectral PEAKS at the positions where a
    naive oscillator's partials above Nyquist fold back (k f0 mirrored at multiples of 48 kHz, off the harmonic grid,
    each window widened by its sweep smear); nyq_db: energy share above 20 kHz. A peak counts only when it stands
    ALIAS_PEAK_DB over the local floor and at least ALIAS_FAMILY of the judged alias positions in that frame show one:
    a noise layer under a harmonic tone (no lines) or an inharmonic bell partial that happens to sit near one alias
    position (no family) is not aliasing. legacy=True: the round-2 measure (all off-grid 4-20 kHz energy)."""
    m = _mono(x)
    S = _stft(m, nfft, hop)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    df = f[1]
    nyq = 10 * np.log10(S[:, f >= 20000].sum() / (S.sum() + 1e-30) + 1e-12)
    lvl = 10 * np.log10(S.sum(1) + 1e-30)
    act = np.nonzero(lvl > lvl.max() - 30)[0][:frames]
    Pw = sum(np.roll(S, k, axis=1) for k in (-2, -1, 0, 1, 2))           # energy within +-2 bins
    c0 = 150 * 2 ** (np.arange(0, np.log2(3000 / 150), 1 / 96))
    H = np.arange(1, int(4000 / c0[0]) + 1)
    hf = c0[:, None] * H[None, :]
    G = np.clip(np.rint(hf / df).astype(int), 0, len(f) - 1)
    low = (hf < 4000) & (hf >= 60)
    f0 = {int(t): _f0(S[t], Pw[t], f, c0, low, G) for t in act}
    C = np.cumsum(S, axis=1)
    num = den = 0.0
    harm = 0
    from scipy.ndimage import median_filter
    for t, r in f0.items():
        if r is None:
            continue
        nb = [f0.get(t + d) for d in (-1, 1)]
        rates = [abs(q[0] - r[0]) / (hop / SR) for q in nb if q is not None]
        rate = max(rates) if rates else 0.0                               # Hz / s of the fundamental
        smear = 0.5 * rate * (nfft / SR) / df                              # extra half-width in bins per harmonic
        hs = np.arange(int(np.ceil(4000 / r[0])), int(20000 / r[0]) + 1)
        hw = 2 + hs * smear                                                # half-width in bins, with smear
        ok = 2 * hw * df < 0.5 * r[0]
        if not ok.any() or not ok[0]:
            continue
        hs, hw = hs[ok], hw[ok]
        if len(hs) > 1 and (np.diff(hs) > 1).any():
            cut = int(np.nonzero(np.diff(hs) > 1)[0][0]) + 1
            hs, hw = hs[:cut], hw[:cut]
        top = min(20000.0, (hs[-1] + 0.5) * r[0])
        b0, b1 = int(np.searchsorted(f, 4000)), int(np.searchsorted(f, top))
        th = C[t, b1] - C[t, b0]
        if b1 - b0 < 20 or th < 1e-3 * C[t, -1]:
            continue
        if legacy:
            k = hs * r[0] / df
            lo = np.clip(np.floor(k - hw).astype(int), b0, b1)
            hi = np.clip(np.ceil(k + hw).astype(int), b0, b1)
            on = float((C[t, hi] - C[t, lo]).sum())
            harm += 1
            num += (1.0 - min(1.0, on / th)) * th
            den += th
            continue
        ka = np.arange(int(SR / 2 / r[0]) + 1, int(4 * SR / r[0]) + 1)    # partials above Nyquist (to 4 x SR)
        fa = np.abs(np.mod(ka * r[0] + SR / 2, SR) - SR / 2)               # where they fold back to
        aw = 1 + ka * smear
        sel = (fa >= 4000) & (fa <= top)
        ka, fa, aw = ka[sel], fa[sel], aw[sel]
        hn = np.maximum(1, np.rint(fa / r[0]))                             # nearest harmonic and its window
        hwn = 2 + hn * smear
        gap = np.abs(fa - hn * r[0]) / df
        judge = gap >= hwn + 1.0                                           # the alias centre is off the grid window
        first = ka < SR / r[0]                                             # the first fold: the strongest aliases
        if (judge & first).sum() < 4:
            continue
        ka, fa, hn, hwn, gap, first = ka[judge], fa[judge], hn[judge], hwn[judge], gap[judge], first[judge]
        aw = np.minimum(aw[judge], np.maximum(0.5, gap - hwn - 0.5))      # never reach into the harmonic window
        seg = S[t, b0:b1 + 1]
        fl = median_filter(seg, size=65, mode='nearest')                  # the local floor
        kb = fa / df - b0
        lo = np.clip(np.ceil(kb - aw).astype(int), 0, b1 - b0)
        hi = np.clip(np.floor(kb + aw).astype(int), 0, b1 - b0)
        kh = hn * r[0] / df - b0                                           # the neighbouring harmonic's own peak
        hlo = np.clip(np.floor(kh - hwn).astype(int), 0, b1 - b0)
        hhi = np.clip(np.ceil(kh + hwn).astype(int), 0, b1 - b0)
        up = hs * r[0] >= 0.5 * top                                       # an alias is never louder than the
        kk = hs[up] * r[0] / df - b0                                      # grid's own partials near the top
        href = max([float(seg[max(0, int(np.floor(c - w))):int(np.ceil(c + w)) + 1].max(initial=0.0))
                    for c, w in zip(kk, hw[up])], default=0.0)
        exc, peaks, nf = 0.0, 0, 0
        for j in range(len(lo)):
            a, b = int(lo[j]), int(max(lo[j], hi[j]))
            w = seg[a:b + 1]
            hp = float(seg[hlo[j]:hhi[j] + 1].max()) if hhi[j] >= hlo[j] else 0.0
            d = max(2.5, gap[j] - hwn[j] + 2.0)                            # Hann sidelobes: -31.5 dB at 2.5 bins,
            leak = hp * 10 ** ((-31.5 - 18 * np.log2(d / 2.5)) / 10)       # -18 dB / octave beyond
            thr = 10 ** (ALIAS_PEAK_DB / 10) * max(float(fl[(a + b) // 2]), leak)
            hit = bool(len(w)) and thr < float(w.max()) <= ALIAS_LOUDEST * href
            if first[j]:
                nf += 1
                peaks += int(hit)
            if hit:
                exc += max(0.0, float(w.sum()) - float(fl[(a + b) // 2]) * len(w))
        harm += 1
        num += min(th, exc) * min(1.0, peaks / max(1, nf) / ALIAS_FAMILY)
        den += th
    fold = num / den if den > 0 and harm >= 2 else 0.0
    return {'fold': round(float(fold), 3), 'harmonic_frames': harm, 'nyq_db': round(float(nyq), 1)}


def dc(x):
    m = _mono(x)
    peak = np.abs(m).max() + 1e-12
    from scipy.fft import next_fast_len
    n = next_fast_len(len(m), real=True)
    F = np.abs(np.fft.rfft(m, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    return {'dc_db': round(float(20 * np.log10(abs(m.mean()) / peak + 1e-12)), 1),
            'infra_db': round(float(10 * np.log10(F[f < 15].sum() / (F.sum() + 1e-30) + 1e-12)), 1)}


def resonances(x, top=3):
    """the harshest narrow peak above the 2/3-octave median of the energy spectrum (weighted dB, Hz)"""
    m = _mono(x)
    nfft = 4096 if len(m) >= 4096 else 1 << max(10, int(np.ceil(np.log2(max(len(m), 2)))))
    S = _stft(m, nfft, nfft // 4).sum(0)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    L = 10 * np.log10(S + 1e-30)
    g = 100 * 2 ** (np.arange(0, np.log2(16000 / 100), 1 / 48))       # 48 points per octave, 100 Hz - 16 kHz
    edges = np.searchsorted(f, np.concatenate([g / 2 ** (1 / 96), [g[-1] * 2 ** (1 / 96)]]))
    Lg = np.array([L[a:max(b, a + 1)].max() for a, b in zip(edges[:-1], edges[1:])])
    from scipy.ndimage import median_filter
    prom = Lg - median_filter(Lg, size=33, mode='nearest')
    w = np.where((g >= 2000) & (g <= 6000), 1.0, np.where((g >= 1000) & (g <= 9000), 0.7, 0.5))
    ok = (Lg > Lg.max() - 30) & (prom > 0)
    loc = np.r_[False, (prom[1:-1] >= prom[:-2]) & (prom[1:-1] >= prom[2:]), False] & ok
    idx = np.nonzero(loc)[0]
    if not len(idx):
        return {'res_db': 0.0, 'res_hz': None, 'res_peaks': []}
    order = idx[np.argsort(-(prom[idx] * w[idx]))][:top]
    return {'res_db': round(float(prom[order[0]] * w[order[0]]), 1), 'res_hz': round(float(g[order[0]])),
            'res_peaks': [(round(float(g[i])), round(float(prom[i]), 1)) for i in order]}


def evolution(x, G=None):
    """how much the spectral shape changes over the sound (dB), and the centroid's range (octaves)"""
    from .tone import CEN, bandgram
    G = bandgram(x) if G is None else G
    P = 10 ** (G / 10)
    groups = [P[:, (CEN >= a) & (CEN < b)].sum(1) for a, b in
              ((0, 112), (112, 224), (224, 447), (447, 894), (894, 1789), (1789, 3578), (3578, 7155), (7155, 1e5))]
    Q = np.stack(groups, 1) + 1e-12
    Q = np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, mode='same'), 0, Q)   # 30 ms smoothing
    tot = Q.sum(1)
    act = 10 * np.log10(tot + 1e-30) > 10 * np.log10(tot.max()) - 30
    if act.sum() < 8:
        return {'evolution_db': None, 'centroid_oct': None}
    R = 10 * np.log10(Q / tot[:, None])
    R = np.maximum(R, -40)
    lag = 5
    i = np.nonzero(act)[0]
    i = i[(i >= lag) & act[np.clip(i - lag, 0, None)]]
    if len(i) < 3:
        return {'evolution_db': None, 'centroid_oct': None}
    sh = 0.5 * (Q[i] / tot[i, None] + Q[i - lag] / tot[i - lag, None])          # groups weighted by their share
    ch = np.sqrt((sh * (R[i] - R[i - lag]) ** 2).sum(1) / sh.sum(1))
    w = tot[i]
    cen = np.log2((P[act] * CEN).sum(1) / P[act].sum(1))
    lo, hi = np.percentile(cen, [10, 90])
    return {'evolution_db': round(float((ch * w).sum() / w.sum()), 2), 'centroid_oct': round(float(hi - lo), 2)}


RING_SPAN_HZ = 2000.0      # a comb needs this much tail bandwidth (within 40 dB of its loudest, 300 Hz - 12 kHz)
RING_PITCH_Z = 100.0       # a body this periodic at the tail's delay (+-30 %, or 2x / 3x) is the sound's own pitch
RING_SMOOTH_DB = 2.0       # tail partials this smooth (median 2nd difference of their levels) are an oscillator


def _cepstrum_peak(seg, span=False):
    """robust z (and quefrency in ms) of the largest cepstral peak at 2-40 ms of one segment's log spectrum (300 Hz -
    12 kHz, 2 kHz trend removed); with span=True also the bandwidth (Hz) where the 2 kHz-smoothed spectrum lies within
    40 dB of its loudest. None when the segment is too short to resolve the range"""
    N = 8192 if len(seg) <= 8192 else 16384
    L = 10 * np.log10(np.abs(np.fft.rfft(seg * np.hanning(len(seg)), N)) ** 2 + 1e-30)
    f = np.fft.rfftfreq(N, 1 / SR)
    sel = (f >= 300) & (f <= 12000)
    df = f[1]
    from scipy.ndimage import uniform_filter1d
    Lc = np.maximum(L[sel], L[sel].max() - 60)
    trend = uniform_filter1d(Lc, max(3, int(2000 / df)), mode='nearest')
    r = Lc - trend
    C = np.abs(np.fft.rfft(r * np.hanning(len(r))))
    q = np.arange(len(C)) / (len(r) * df)
    band = (q >= 0.002) & (q <= 0.04)
    if band.sum() < 8:
        return None
    c = C[band]
    mad = np.median(np.abs(c - np.median(c))) * 1.4826 + 1e-12
    j = int(np.argmax(c))
    out = (float((c[j] - np.median(c)) / mad), float(q[band][j] * 1000))
    if span:
        sm = uniform_filter1d(L[sel], max(3, int(2000 / df)), mode='nearest')
        out += (float((sm > sm.max() - 40).sum() * df),)
    return out


def ring(x, legacy=False):
    """comb-like periodicity of the tail: cepstral peak z at 2-40 ms quefrency (None when the tail is too short).
    Not judged (ring_z None, ring_src says why): a tail narrower than RING_SPAN_HZ ('narrow': a sub drop's few low
    partials over an empty band put a huge z at the 2 ms edge) or a tail whose delay is the sound's own pitch ('pitch':
    the body around the peak is periodic at the same delay, z >= RING_PITCH_Z: a buzzy hum, a pitched tone). A comb
    on a noise burst's tail, or on a pitched body at another delay, is still flagged. legacy=True: round 2."""
    m = _mono(x)
    e = _fdb(m, 480)
    ip = int(np.argmax(e))
    s = ip + 3
    below = np.nonzero(e[s:] < e[ip] - 50)[0]
    end = s + (int(below[0]) if len(below) else len(e) - s)
    if end - s < 6:
        return {'ring_z': None, 'ring_ms': None}
    seg = m[s * 480:end * 480][:16384]
    got = _cepstrum_peak(seg, span=True)
    if got is None:
        return {'ring_z': None, 'ring_ms': None}
    z, ms, bw = got
    out = {'ring_z': round(z, 1), 'ring_ms': round(ms, 2)}
    if legacy or z < 8.0:
        return out
    if bw < RING_SPAN_HZ:
        return {'ring_z': None, 'ring_ms': round(ms, 2), 'ring_raw_z': round(z, 1), 'ring_src': 'narrow'}
    pitch = _teeth_roughness(seg, ms) <= RING_SMOOTH_DB                  # deterministic partials, not noisy teeth
    if not pitch:
        body = m[max(0, (ip - 2) * 480):(ip + 6) * 480]
        bp = _cepstrum_peak(body) if len(body) >= 1920 else None
        if bp is not None and bp[0] >= RING_PITCH_Z:
            ratio = ms / bp[1]
            pitch = any(abs(ratio / k - 1) < (0.3 if k == 1 else 0.1) for k in (1, 2, 3, 0.5, 1 / 3))
    if pitch:
        return {'ring_z': None, 'ring_ms': round(ms, 2), 'ring_raw_z': round(z, 1), 'ring_src': 'pitch'}
    return out


def _teeth_roughness(seg, ms, N=16384):
    """median |2nd difference| (dB) of the peak levels at the multiples of 1 / delay (300 Hz - 12 kHz, within 50 dB
    of the loudest bin): an oscillator's partials follow a smooth envelope (0.1-1.4 dB on buzzy hums), the teeth of a
    comb fed with noise scatter like the noise (6-7 dB); inf with fewer than 8 teeth"""
    L = 10 * np.log10(np.abs(np.fft.rfft(seg * np.hanning(len(seg)), N)) ** 2 + 1e-30)
    df = SR / N
    f0 = 1000.0 / ms
    ks = np.arange(max(1, int(np.ceil(300 / f0))), int(12000 / f0) + 1)
    lv = np.array([L[max(0, int(k * f0 / df) - 2):int(k * f0 / df) + 3].max() for k in ks])
    lv = lv[lv > L.max() - 50]
    if len(lv) < 8:
        return np.inf
    return float(np.median(np.abs(np.diff(lv, 2))))


def truncated(x):
    """trunc_db: the level (dB re peak) at which a still-decaying tail was cut, or None"""
    m = _mono(x)
    e = _fdb(m, 240)                                                   # 5 ms frames
    if len(e) < 12:
        return {'trunc_db': None, 'end_db': None}
    ip = int(np.argmax(e))
    e = e - e[ip]
    end = float(np.mean(e[-2:]))
    if len(e) - ip < 12 or end < -55:
        return {'trunc_db': None, 'end_db': round(end, 1)}
    before = e[max(ip, len(e) - 38):len(e) - 8]                        # 150 ms .. 40 ms before the end
    slope = np.polyfit(np.arange(len(before)) * 0.005, before, 1)[0] if len(before) >= 4 else 0.0  # dB/s
    fade_from = float(np.mean(e[-10:-7]))
    expect = max(3.0, -slope * 0.04)                                   # natural drop over the last 40 ms
    cut = None
    if fade_from - end > 3 * expect + 6 and fade_from > -50:           # a final fade much faster than the decay
        cut = fade_from
    elif slope < -20 and end > -50:                                    # still decaying when the file stops
        cut = end
    return {'trunc_db': None if cut is None else round(cut, 1), 'end_db': round(end, 1)}


def silence(x):
    m = _mono(x)
    a = np.abs(m)
    peak = a.max() + 1e-12
    aud = np.nonzero(a > peak * 1e-3)[0]
    if len(aud) < 2:
        return {'silence_ms': 0.0, 'silence_max_ms': 0.0}
    z = a[aud[0]:aud[-1]] < 1e-6
    runs = np.diff(np.r_[0, z.astype(np.int8), 0])
    st, en = np.nonzero(runs == 1)[0], np.nonzero(runs == -1)[0]
    L = (en - st)[(en - st) >= 96]
    return {'silence_ms': round(float(L.sum() / 48), 1),
            'silence_max_ms': round(float(L.max() / 48), 1) if len(L) else 0.0}


def clipping(x):
    runs, overs = 0, 0
    for c in _chans(x):
        a = np.abs(c)
        pk = a.max()
        if pk <= 0:
            continue
        overs += int((a > 1.0).sum())
        flat = (a[1:] >= 0.99 * pk) & (np.abs(np.diff(c)) < 2e-6)
        d = np.diff(np.r_[0, flat.astype(np.int8), 0])
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        runs += int(((en - st) >= 2).sum())
    return {'clip_runs': runs, 'overs': overs}


# ------------------------------------------------------------------------------------------------ polish
def attack(x):
    """10 -> 90 % rise (ms) to the first near-peak of the 0.5 ms RMS envelope, and the HF share of the first 10 ms"""
    m = _mono(x)
    n = len(m) // 12
    if n < 4:
        return {'attack_ms': None, 'onset_hf_db': None}
    p = (m[:n * 12].reshape(n, 12) ** 2).mean(1)                       # 0.25 ms hops
    env = np.sqrt(np.convolve(p, np.ones(2) / 2, mode='same'))         # 0.5 ms RMS
    top = env.max()
    i90 = int(np.nonzero(env >= 0.9 * top)[0][0])
    pre = np.nonzero(env[:i90 + 1] < 0.1 * top)[0]
    i10 = int(pre[-1]) if len(pre) else 0
    on = i10 * 12
    seg = m[on:on + 480]
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1024)) ** 2 if len(seg) >= 32 else None
    hf = None
    if F is not None:
        f = np.fft.rfftfreq(1024, 1 / SR)
        hf = round(float(10 * np.log10(F[f >= 6000].sum() / (F.sum() + 1e-30) + 1e-12)), 1)
    return {'attack_ms': round((i90 - i10) * 0.25, 2), 'onset_hf_db': hf}


def tail(x):
    """bump_db (re-rise above the running minimum after the peak), decay_db_s, t40_ms"""
    m = _mono(x)
    e = _fdb(m, 480)
    if len(e) < 4:
        return {'bump_db': 0.0, 'decay_db_s': None, 't40_ms': None}
    p = np.convolve(10 ** (e / 10), np.ones(3) / 3, mode='same')
    e = 10 * np.log10(p + 1e-24)
    ip = int(np.argmax(e))
    e = e - e[ip]
    after = e[ip:]
    below = np.nonzero(after < -40)[0]
    t40 = float(below[0] * 10) if len(below) else float(len(after) * 10)
    run = np.minimum.accumulate(after)
    live = after > -45
    live[:2] = False
    bump = float(np.max((after - run)[live])) if live.any() else 0.0
    sel = np.nonzero((after <= -6) & (after >= -45))[0]
    rate = None
    if len(sel) >= 3:
        rate = round(float(np.polyfit(sel * 0.01, after[sel], 1)[0]), 1)
    return {'bump_db': round(bump, 1), 'decay_db_s': rate, 't40_ms': t40}


def _movmax(p, n):
    """the largest mean of n consecutive values (the whole mean when shorter)"""
    if len(p) < n:
        return float(p.sum() / n)
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float((c[n:] - c[:-n]).max() / n)


def _kweight(m):
    from scipy import signal
    return signal.sosfilt(K_SOS, m)


def body(x):
    """crest_db: peak (dBFS) minus the loudest 50 ms of K-weighted RMS"""
    m = _mono(x)
    k = _kweight(m)
    loud = 10 * np.log10(_movmax(k ** 2, 2400) + 1e-24)
    return {'crest_db': round(float(20 * np.log10(np.abs(m).max() + 1e-12) - loud), 1)}


def lufs_max(x):
    """momentary-max loudness (LUFS, K-weighted 400 ms windows over the channel sum of powers; zero-padded when
    the sound is shorter than 400 ms, as a meter would show it)"""
    tot = sum(_kweight(c) ** 2 for c in _chans(x))
    return float(-0.691 + 10 * np.log10(_movmax(tot, 19200) + 1e-24))


def loudness_spread(xs):
    """do variants of one sound play at one loudness? per-variant momentary-max LUFS at their own levels"""
    L = [lufs_max(_load(x)) for x in xs]
    if not L:
        return {'lufs': [], 'spread_lu': None, 'std_lu': None}
    return {'lufs': [round(v, 2) for v in L], 'spread_lu': round(float(max(L) - min(L)), 2),
            'std_lu': round(float(np.std(L)), 2)}


def measure(x, G=None):
    """every detector's raw numbers for one sound (G: its tone.bandgram, when already computed)"""
    x = _load(x)
    out = {}
    for fn in (clicks, aliasing, dc, resonances, ring, truncated, silence, clipping, attack, tail, body):
        out.update(fn(x))
    out.update(evolution(x, G))
    return out


# ------------------------------------------------------------------------------------------------ penalties
#        key            bad   abs     margin scale  kind    cap   (bad: +1 high is bad, -1 low is bad)
RULES = (('clicks',     +1,   0.0,    0.0,   1.0,   'hard', 2.0),
         ('edge_db',    +1,   -40.0,  6.0,   10.0,  'hard', 2.0),
         ('fold',       +1,   0.02,   0.02,  0.03,  'hard', 2.0),
         ('nyq_db',     +1,   -35.0,  6.0,   8.0,   'soft', 1.0),    # inaudible itself: a sign of naive synthesis
         ('dc_db',      +1,   -40.0,  6.0,   10.0,  'hard', 2.0),
         ('infra_db',   +1,   -20.0,  6.0,   8.0,   'soft', 1.5),    # wasted headroom, thumps on real systems
         ('res_db',     +1,   15.0,   3.0,   5.0,   'soft', 2.0),
         ('ring_z',     +1,   12.0,   3.0,   6.0,   'soft', 2.0),
         ('trunc_db',   +1,   -45.0,  6.0,   10.0,  'hard', 2.0),
         ('silence_ms', +1,   2.0,    2.0,   10.0,  'hard', 2.0),
         ('clip_runs',  +1,   0.0,    1.0,   4.0,   'hard', 2.0),
         ('evolution_db', -1, None,   0.35,  None,  'soft', 2.0))    # frozen: below 0.35 x the refs' median
#        key           bad  kind  margin scale   (refs only; dB margins widen to a quarter of the refs' range)
POLISH = (('attack_ms',  +1,  'log', 1.5, 1.0),                   # slower than 1.5 x the refs' slowest (+1 ms)
          ('onset_hf_db', -1, 'db', 4.0, 6.0),                    # duller snap than the refs' dullest - 4 dB
          ('bump_db',    +1,  'db', 2.0, 6.0),                    # bumpier tail than the refs' bumpiest + 2 dB (abs 4)
          ('t40_ms',     -1,  'log', 0.6, 1.0),                   # tail shorter than 0.6 x the refs' shortest
          ('crest_db',   +1,  'db', 2.0, 4.0))                    # thinner than the refs' thinnest + 2 dB
HARD_AT = 0.5
GATE_DB = {'designed': -30.0, 'gated': -12.0}   # designed mode: a tail may stop once it is 30 dB down (a gate, not a
                                                # cut); gated categories (glitch, riser, zip, pro_impact): a gate
                                                # anywhere 12 dB under the peak is the design, soft at most


class PolishRef:
    """the refs' artifact and polish numbers, and a check of any sound against them (or absolute limits alone)"""

    def __init__(self, refs=(), designed=False, gated=False, targets=None):
        self.M = [measure(_load(r)) for r in refs]
        self.n = len(self.M)
        self.designed = designed
        self.gated = bool(gated)
        self.targets = dict(targets or {})          # {polish key: (lo, hi)}: stands in for refs when < 2 refs

    def _vals(self, key, keep):
        v = [self.M[j].get(key) for j in keep]
        return [float(a) for a in v if a is not None]

    def check(self, x, exclude=None, G=None, m=None):
        """measure one sound (or pass its measure() dict as m) and score it; exclude = a ref index to leave out"""
        m = measure(x, G) if m is None else m
        keep = [j for j in range(self.n) if j != exclude]
        pens, limits, hard = {}, {}, []
        for key, bad, ab, margin, scale, kind, cap in RULES:
            v = m.get(key)
            if v is None:
                continue
            rv = self._vals(key, keep)
            if bad < 0:                                               # frozen: relative to moving refs only
                if len(rv) < 2 or np.median(rv) < 1.5:
                    continue
                lim = margin * float(np.median(rv))
                p = min(cap, max(0.0, (lim - v) / max(lim, 1e-6) * 2))
            else:
                lim = max(ab, max(rv) + margin) if rv else ab
                p = min(cap, max(0.0, (v - lim) / scale))
            if self.designed and key in ('clicks', 'silence_ms'):
                kind, p = 'soft', 0.5 * p
            if self.designed and key == 'trunc_db':       # a gate after the level has already fallen is a choice
                lim = max(lim, GATE_DB['gated' if self.gated else 'designed'])
                p = min(cap, max(0.0, (v - lim) / scale))
                if self.gated:
                    kind = 'soft'
            limits[key] = round(lim, 2)
            if p > 0:
                pens[key] = round(p, 3)
                if kind == 'hard' and p >= HARD_AT:
                    hard.append(key)
        art = sum(pens.values())
        ppen = {}
        for key, bad, kind, margin, scale in POLISH:
            v = m.get(key)
            rv = self._vals(key, keep)
            if v is None:
                continue
            if key != 'bump_db' and len(rv) < 2 and key in self.targets:    # absolute fallback (critic
                rv = [float(a) for a in self.targets[key]]                       # STUDY_TARGETS) for < 2 refs
            if key == 'bump_db':
                lim = max(4.0, max(rv) + margin) if rv else 4.0
                p = (v - lim) / scale
            elif len(rv) < 2:
                continue
            elif kind == 'log':
                lim = margin * max(rv) + 1.0 if bad > 0 else margin * min(rv)
                p = np.log2(max(v, 1e-3) / lim) if bad > 0 else np.log2(lim / max(v, 1e-3))
                p /= scale
            else:
                mg = max(margin, 0.25 * (max(rv) - min(rv)))
                lim = max(rv) + mg if bad > 0 else min(rv) - mg
                p = (v - lim) / scale if bad > 0 else (lim - v) / scale
            limits[key] = round(float(lim), 2)
            if p > 0:
                ppen[key] = round(float(min(2.0, p)), 3)
        out = {'measures': m, 'limits': limits, 'art': pens, 'polish': ppen, 'hard': hard,
               'art_pen': round(float(art), 3), 'polish_pen': round(float(sum(ppen.values())), 3)}
        out['words'] = self.words(out, keep)
        return out

    def words(self, c, keep=None):
        m, keep = c['measures'], keep if keep is not None else range(self.n)

        def rng(key, fmt='{:.0f}'):
            rv = self._vals(key, keep)
            return f" (refs {fmt.format(min(rv))}-{fmt.format(max(rv))})" if rv else ''
        say = {'clicks': lambda: f"{m['clicks']} click{'s' if m['clicks'] != 1 else ''} "
                                 f"({', '.join(f'{t:.3f} s' for t in m['click_times'][:3])})",
               'edge_db': lambda: f"starts / ends off zero ({m['edge_db']:.0f} dB re peak)",
               'fold': lambda: f"aliasing: {m['fold']:.0%} of the moving partials fold back",
               'nyq_db': lambda: f"energy above 20 kHz {m['nyq_db']:.0f} dB" + rng('nyq_db'),
               'dc_db': lambda: f"DC offset {m['dc_db']:.0f} dB re peak",
               'infra_db': lambda: f"infrasound below 15 Hz {m['infra_db']:.0f} dB" + rng('infra_db'),
               'res_db': lambda: f"narrow resonance at {m['res_hz']} Hz +{m['res_db']:.0f} dB" + rng('res_db'),
               'ring_z': lambda: f"metallic comb ring every {m['ring_ms']:.1f} ms (z {m['ring_z']:.0f})" + rng('ring_z'),
               'trunc_db': lambda: f"tail cut at {m['trunc_db']:.0f} dB",
               'silence_ms': lambda: f"{m['silence_ms']:.0f} ms of digital silence inside",
               'clip_runs': lambda: f"clipped ({m['clip_runs']} flat tops)",
               'evolution_db': lambda: (f"frozen spectrum ({m['evolution_db']:.1f} dB of change, refs median "
                                        f"{np.median(self._vals('evolution_db', keep)):.1f})"),
               'attack_ms': lambda: f"attack {m['attack_ms']:.0f} ms" + rng('attack_ms', '{:.1f}'),
               'onset_hf_db': lambda: f"dull onset (>6 kHz share {m['onset_hf_db']:.0f} dB" + rng('onset_hf_db') + ')',
               'bump_db': lambda: f"bumpy tail (+{m['bump_db']:.0f} dB re-rise)",
               't40_ms': lambda: f"tail dies fast (t40 {m['t40_ms']:.0f} ms" + rng('t40_ms') + ')',
               'crest_db': lambda: f"thin body (crest {m['crest_db']:.1f} dB" + rng('crest_db', '{:.1f}') + ')'}
        a = [say[k]() for k in sorted(c['art'], key=lambda k: -c['art'][k])]
        p = [say[k]() for k in sorted(c['polish'], key=lambda k: -c['polish'][k])]
        bits = (['artifacts: ' + ', '.join(a)] if a else []) + (['polish: ' + ', '.join(p)] if p else [])
        return '; '.join(bits) if bits else 'clean (no artifact or polish flag)'


if __name__ == '__main__':                                            # python -m earshot.forge.polish files/folders
    import sys
    from .. import audio
    for p in audio.list_media(sys.argv[1:], exts=audio.AUDIO_EXTS):
        r = measure(p)
        print(p.replace(chr(92), '/').rsplit('/', 1)[-1], ' '.join(f'{k}={r[k]}' for k in
              ('clicks', 'edge_db', 'fold', 'nyq_db', 'dc_db', 'res_db', 'res_hz', 'ring_z', 'trunc_db', 'silence_ms',
               'clip_runs', 'evolution_db', 'attack_ms', 'onset_hf_db', 'bump_db', 't40_ms', 'crest_db')))
