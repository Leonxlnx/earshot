"""Measurements of one sound-design event inside a finished film mix (music bed underneath), bed-subtracted.

    from measure import measure
    m = measure(x, t0, t1, bed=(b0, b1), cut=6.38)      # x = stereo float array at 48 kHz, times in seconds

The bed is the mix just before the event (default: the quietest 150 ms in the 0.6 s before t0, quiet_bed()). Everything that can be is measured
on (event power - bed power), so the music under the event mostly drops out; where the event does not rise over the bed
in a band, that band is 'masked' (floored) and says nothing about the event.

Returned (all plain floats / lists, JSON-ready):
  dur_s, bed_lufs (K-weighted mean level of the bed, ~LUFS; bed_silent when < -60), ev_lufs (same over the event's loudest 100 ms),
  over_bed_db (ev_lufs - bed_lufs, a silent bed counted as -70: 'how far it sticks out of the music'),
  snr_db (bed-subtracted event power / bed power over the whole event window, broadband),
  third_oct: {centres, ev_db (event power per band, dB re its total), bed_db (same scale), excess_db (ev - bed in each
             band), masked (bands where the event adds < 1 dB)}, tilt_db_oct (slope of the bed-subtracted balance,
             100 Hz - 10 kHz, least squares), shares (5 broad regions, dB of total: <150, 150-1k, 1-4k, 4-12k, >12k),
  attack_ms (from the first ms within 20 dB of the peak to the peak, 1 ms bed-subtracted envelope), rise_ms (10 -> 90 %
             amplitude), t_peak (s, absolute), peak_frac (where in [t0, t1] the peak sits),
  t20_ms / t40_ms (peak -> first fall 20 / 40 dB under it; None when the bed comes first), decay_db_s (fit over the
             first 20 dB of the fall), decay_shape ('exp' = straight in dB, 'convex' = holds then drops (gated / faded),
             'concave' = drops fast then a long tail (reverb)),
  centroid: start / peak / end (Hz, bed-subtracted, frames >= -20 dB re peak), sweep_oct (end - start in octaves),
            sweep_oct_s, the 20 ms track (t, Hz)
  f0_track (dominant peak of the 20-250 Hz band per 20 ms frame, for sub drops / booms: start, end, octaves),
  stereo: side_mid_db (bed-subtracted side / mid power), corr (L/R correlation of the event window), pan_db range
          (L-R level difference over 20 ms frames: min, max, travel), width words
  cut_ms: t_peak - cut and t_attack_start - cut (ms) when a cut is given
"""
import numpy as np
from scipy import signal

SR = 48000
CENTRES = [1000 * 2 ** (k / 3) for k in range(-16, 13)]          # 25 Hz ... 16 kHz (29 bands)
KW = [((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585)),
      ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))]
REGIONS = [(0, 150), (150, 1000), (1000, 4000), (4000, 12000), (12000, 24000)]


def kweight(m):
    y = m
    for b, a in KW:
        y = signal.lfilter(b, a, y)
    return y


def _seg(x, a, b):
    return x[max(0, int(round(a * SR))):max(0, int(round(b * SR)))]


def _psd(m, nfft=None):
    """mean power per sample spectrum (Hann, zero-padded), returns f, P such that sum(P) ~ mean(m**2)"""
    n = len(m)
    nfft = nfft or max(8192, 1 << int(np.ceil(np.log2(max(n, 2)))))
    w = np.hanning(n)
    X = np.fft.rfft(m * w, nfft)
    P = np.abs(X) ** 2 / (np.sum(w ** 2) * n) * n / 1.0
    P[1:-1] *= 2
    P /= n
    f = np.fft.rfftfreq(nfft, 1 / SR)
    return f, P


def _bands(f, P):
    out = []
    for c in CENTRES:
        lo, hi = c * 2 ** (-1 / 6), c * 2 ** (1 / 6)
        k = (f >= lo) & (f < hi)
        out.append(P[k].sum() if k.any() else P[np.argmin(np.abs(f - c))])
    return np.array(out)


def _env_db(m, ms=1.0):
    n = max(1, int(SR * ms / 1000))
    k = len(m) // n
    return 10 * np.log10((m[:k * n].reshape(k, n) ** 2).mean(1) + 1e-14)


def _frames(m, ms=20.0):
    n = int(SR * ms / 1000)
    k = len(m) // n
    return m[:k * n].reshape(k, n), n


def quiet_bed(x, t0, search=0.6, win=0.15, gap=0.015):
    """the bed under an event: the quietest `win` s (K-weighted power, 10 ms hop) in the `search` s before it (so a
    run of clicks or a previous hit does not count as bed); shorter at the start of a file"""
    a, b = max(0.0, t0 - search), max(0.0, t0 - gap)
    if b - a <= win:
        return (a, b) if b - a >= 0.03 else (a, max(b, a + 0.03))
    k = kweight(_seg(x, a, b).mean(1)) ** 2
    n, h = int(win * SR), int(0.01 * SR)
    c = np.cumsum(np.concatenate([[0.0], k]))
    starts = np.arange(0, len(k) - n + 1, h)
    p = (c[starts + n] - c[starts]) / n
    i = int(starts[int(np.argmin(p))])
    return (round(a + i / SR, 4), round(a + (i + n) / SR, 4))


def measure(x, t0, t1, bed=None, cut=None):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    if bed is None:
        bed = quiet_bed(x, t0)
    ev = _seg(x, t0, t1)
    bd = _seg(x, *bed)
    if len(bd) < int(0.02 * SR):
        bd = np.zeros((int(0.05 * SR), 2)) + 1e-7
    mev, mbd = ev.mean(1), bd.mean(1)
    r = {'t0': round(t0, 3), 't1': round(t1, 3), 'dur_s': round(t1 - t0, 3), 'bed': [round(bed[0], 3), round(bed[1], 3)]}

    # levels (K-weighted)
    pre = _seg(x, max(0, bed[0] - 0.5), t1).mean(1)
    kw = kweight(pre)
    off = len(pre) - len(mev)
    kb = kw[int(round((bed[0] - max(0, bed[0] - 0.5)) * SR)):][:len(mbd)]
    ke = kw[off:]
    bed_p = np.mean(kb ** 2) + 1e-14
    win = int(0.1 * SR)
    if len(ke) > win:
        c = np.convolve(ke ** 2, np.ones(win) / win, 'valid')
        ev_p = c.max()
    else:
        ev_p = np.mean(ke ** 2)
    r['bed_lufs'] = round(-0.691 + 10 * np.log10(bed_p), 1)
    r['ev_lufs'] = round(-0.691 + 10 * np.log10(ev_p + 1e-14), 1)
    r['bed_silent'] = bool(r['bed_lufs'] < -60)
    r['over_bed_db'] = round(r['ev_lufs'] - max(r['bed_lufs'], -70.0), 1)   # a silent bed counts as -70
    pe, pb = np.mean(mev ** 2), np.mean(mbd ** 2) + 1e-14
    r['snr_db'] = round(10 * np.log10(max(pe - pb, 1e-14) / pb), 1)

    # third-octave balance, bed-subtracted
    f, Pe = _psd(mev)
    # bed spectrum: average of bed chunks of the event's length (or the whole bed when shorter)
    L = min(len(mbd), len(mev))
    chunks = [mbd[i:i + L] for i in range(0, len(mbd) - L + 1, max(1, L // 2))] or [mbd]
    Pb = np.mean([_psd(c, len(f) * 2 - 2)[1] for c in chunks], 0)
    Be, Bb = _bands(f, Pe), _bands(f, Pb)
    D = np.maximum(Be - Bb, 1e-14)
    masked = (10 * np.log10(Be + 1e-14) - 10 * np.log10(Bb + 1e-14)) < 1.0
    D[masked] = np.minimum(D[masked], Be[masked] * 0.05)
    tot = D.sum() + 1e-14
    evdb = 10 * np.log10(D / tot)
    r['third_oct'] = {'centres': [round(c, 1) for c in CENTRES],
                      'ev_db': [round(max(v, -60), 1) for v in evdb],
                      'bed_db': [round(max(v, -60), 1) for v in 10 * np.log10(Bb / tot + 1e-14)],
                      'excess_db': [round(v, 1) for v in 10 * np.log10(Be + 1e-14) - 10 * np.log10(Bb + 1e-14)],
                      'masked': [bool(v) for v in masked]}
    lc = np.log2(np.array(CENTRES))
    k = (np.array(CENTRES) >= 100) & (np.array(CENTRES) <= 10000) & ~masked
    if k.sum() >= 4:
        r['tilt_db_oct'] = round(float(np.polyfit(lc[k] - lc[k].mean(), evdb[k], 1)[0]), 2)
    else:
        r['tilt_db_oct'] = None
    sh = []
    for lo, hi in REGIONS:
        kk = (f >= lo) & (f < hi)
        sh.append(round(10 * np.log10(max(Pe[kk].sum() - Pb[kk].sum(), 1e-14) / max(Pe.sum() - Pb.sum(), 1e-14)), 1))
    r['shares'] = sh
    # spectral centroid of the bed-subtracted spectrum
    Dp = np.maximum(Pe - Pb, 0)
    r['centroid_hz'] = round(float((f * Dp).sum() / (Dp.sum() + 1e-14)), 0)

    # envelope (1 ms, bed-subtracted power)
    e = 10 ** (_env_db(mev) / 10) - pb
    e = np.maximum(e, pb * 1e-3)
    edb = 10 * np.log10(e)
    ip = int(np.argmax(edb))
    pk = edb[ip]
    i20 = int(np.argmax(edb[:ip + 1] >= pk - 20))
    r['attack_ms'] = float(ip - i20)
    amp = np.sqrt(e)
    i10 = int(np.argmax(amp[:ip + 1] >= 0.1 * amp[ip]))
    i90 = int(np.argmax(amp[:ip + 1] >= 0.9 * amp[ip]))
    r['rise_ms'] = float(i90 - i10)
    r['t_peak'] = round(t0 + ip / 1000, 4)
    r['t_attack'] = round(t0 + i20 / 1000, 4)
    r['peak_frac'] = round(ip / max(1, len(edb)), 3)
    bedlvl = 10 * np.log10(pb)
    # smoothed envelope for decay (10 ms)
    es = np.convolve(e, np.ones(10) / 10, 'same')
    esdb = 10 * np.log10(es + 1e-14)
    after = esdb[ip:]
    pk_s = esdb[ip]

    def fall(db):
        j = np.nonzero(after <= pk_s - db)[0]
        return float(j[0]) if len(j) else None
    r['t20_ms'], r['t40_ms'] = fall(20), fall(40)
    r['peak_over_bed_db'] = round(float(min(pk_s - bedlvl, pk_s - 10 * np.log10(1e-7))), 1)
    j20 = r['t20_ms']
    if j20 and j20 >= 8:
        seg = after[:int(j20) + 1]
        tt = np.arange(len(seg)) / 1000
        p1 = np.polyfit(tt, seg, 1)
        r['decay_db_s'] = round(float(p1[0]), 1)
        mid = seg[len(seg) // 2] - (seg[0] + seg[-1]) / 2
        r['decay_shape'] = 'convex' if mid > 2.5 else ('concave' if mid < -2.5 else 'exp')
        r['decay_mid_dev_db'] = round(float(mid), 1)
    else:
        r['decay_db_s'] = None
        r['decay_shape'] = None
    # rise shape (onset -> peak) for swells / whooshes: mid-point deviation from a straight dB ramp
    if ip - i20 >= 20:
        seg = esdb[i20:ip + 1]
        mid = seg[len(seg) // 2] - (seg[0] + seg[-1]) / 2
        r['rise_shape'] = 'convex' if mid > 2.5 else ('concave' if mid < -2.5 else 'exp')
        r['rise_mid_dev_db'] = round(float(mid), 1)

    # centroid track (20 ms frames, bed-subtracted)
    Fm, n = _frames(mev, 20)
    if len(Fm) >= 2:
        w = np.hanning(n)
        S = np.abs(np.fft.rfft(Fm * w, 2048)) ** 2
        ff = np.fft.rfftfreq(2048, 1 / SR)
        Bs = np.abs(np.fft.rfft(chunks[0][:n] * w if len(chunks[0]) >= n else np.zeros(n), 2048)) ** 2
        if len(mbd) >= n:
            Bf, _ = _frames(mbd, 20)
            Bs = (np.abs(np.fft.rfft(Bf * w, 2048)) ** 2).mean(0)
        Sd = np.maximum(S - Bs, 0)
        en = Sd.sum(1)
        ok = en >= en.max() * 0.01
        cen = (Sd * ff).sum(1) / (en + 1e-14)
        idx = np.nonzero(ok)[0]
        tr = [(round(t0 + (i + 0.5) * 0.02, 3), round(float(cen[i]), 0)) for i in idx]
        r['centroid'] = {'start': tr[0][1], 'peak': round(float(cen[int(np.argmax(en))]), 0), 'end': tr[-1][1],
                         'track': tr}
        if len(tr) >= 3 and tr[0][1] > 0 and tr[-1][1] > 0:
            # robust: medians of first / last quarter of active frames
            q = max(1, len(tr) // 4)
            c0 = np.median([v for _, v in tr[:q]])
            c1 = np.median([v for _, v in tr[-q:]])
            r['centroid']['sweep_oct'] = round(float(np.log2(c1 / c0)), 2)
            span = tr[-1][0] - tr[0][0]
            r['centroid']['sweep_oct_s'] = round(float(np.log2(c1 / c0) / max(span, 0.02)), 2)
            # rise-to-peak and peak-to-end sweeps
            ipk = int(np.argmax([en[i] for i in idx]))
            cp = np.median([v for _, v in tr[max(0, ipk - 1):ipk + 2]])
            r['centroid']['to_peak_oct'] = round(float(np.log2(cp / c0)), 2)
            r['centroid']['from_peak_oct'] = round(float(np.log2(c1 / cp)), 2)
        # low f0 track (20-250 Hz peak) on 80 ms frames for sub content
        F8, n8 = _frames(mev, 80)
        if len(F8) >= 2:
            S8 = np.abs(np.fft.rfft(F8 * np.hanning(n8), 16384)) ** 2
            f8 = np.fft.rfftfreq(16384, 1 / SR)
            kb = (f8 >= 20) & (f8 <= 250)
            pk8 = f8[kb][np.argmax(S8[:, kb], 1)]
            lv8 = 10 * np.log10(S8[:, kb].max(1) + 1e-14)
            okk = lv8 >= lv8.max() - 25
            ii = np.nonzero(okk)[0]
            r['f0_track'] = {'track': [(round(t0 + (i + 0.5) * 0.08, 2), round(float(pk8[i]), 1)) for i in ii],
                             'start': round(float(pk8[ii[0]]), 1), 'end': round(float(pk8[ii[-1]]), 1),
                             'oct': round(float(np.log2(pk8[ii[-1]] / max(pk8[ii[0]], 1))), 2)}

    # ridge: the dominant (bed-subtracted) peak frequency per 10 ms frame, 80 Hz - 16 kHz (tonal sweeps: zips, pops,
    # stings, the tonal layer of a riser); frames within 20 dB of the loudest frame
    Fr, nr = _frames(mev, 10)
    if len(Fr) >= 3:
        wr = np.hanning(nr)
        Sr = np.abs(np.fft.rfft(Fr * wr, 8192)) ** 2
        fr = np.fft.rfftfreq(8192, 1 / SR)
        if len(mbd) >= nr:
            Bfr, _ = _frames(mbd, 10)
            Sr = np.maximum(Sr - (np.abs(np.fft.rfft(Bfr * wr, 8192)) ** 2).mean(0) * 1.5, 0)
        kb = (fr >= 80) & (fr <= 16000)
        pkf = fr[kb][np.argmax(Sr[:, kb], 1)]
        pe_ = Sr[:, kb].max(1)
        ok = pe_ >= pe_.max() * 0.01
        ii = np.nonzero(ok)[0]
        tr = [(round(t0 + (i + 0.5) * 0.01, 3), round(float(pkf[i]), 0)) for i in ii]
        q = max(1, len(tr) // 5)
        f0_, f1_ = float(np.median([v for _, v in tr[:q]])), float(np.median([v for _, v in tr[-q:]]))
        r['ridge'] = {'start_hz': round(f0_, 0), 'end_hz': round(f1_, 0),
                      'min_hz': round(float(min(v for _, v in tr)), 0), 'max_hz': round(float(max(v for _, v in tr)), 0),
                      'peak_hz': round(float(pkf[int(np.argmax(pe_))]), 0),
                      'sweep_oct': round(float(np.log2(max(f1_, 1) / max(f0_, 1))), 2), 'track': tr}
    # amplitude modulation of the 1 ms envelope (stutter / gate / typing rate): strongest 4-60 Hz component
    if len(edb) >= 200:
        env = np.sqrt(e)
        trend = np.convolve(env, np.ones(101) / 101, 'same')
        mod = env - trend
        Fm_ = np.abs(np.fft.rfft(mod * np.hanning(len(mod)), 1 << 15))
        fm = np.fft.rfftfreq(1 << 15, 1 / 1000)
        km = (fm >= 4) & (fm <= 60)
        if km.any():
            j = int(np.argmax(Fm_[km]))
            r['am_rate_hz'] = round(float(fm[km][j]), 1)
            r['am_depth'] = round(float(np.std(mod) / (np.mean(env) + 1e-12)), 2)

    # stereo
    Lc, Rc = ev[:, 0], ev[:, 1]
    M, Sd_ = (Lc + Rc) / 2, (Lc - Rc) / 2
    Mb, Sb = mbd, (bd[:, 0] - bd[:, 1]) / 2
    pm = max(np.mean(M ** 2) - np.mean(Mb ** 2), 1e-14)
    ps = max(np.mean(Sd_ ** 2) - np.mean(Sb ** 2), 1e-14)
    r['stereo'] = {'side_mid_db': round(max(10 * np.log10(ps / pm), -40.0), 1),
                   'side_mid_raw_db': round(10 * np.log10((np.mean(Sd_ ** 2) + 1e-14) / (np.mean(M ** 2) + 1e-14)), 1),
                   'bed_side_mid_db': round(10 * np.log10((np.mean(Sb ** 2) + 1e-14) / (np.mean(Mb ** 2) + 1e-14)), 1),
                   'corr': round(float(np.corrcoef(Lc, Rc)[0, 1]) if np.std(Lc) > 0 and np.std(Rc) > 0 else 1.0, 3)}
    FL, _ = _frames(Lc, 20)
    FR, _ = _frames(Rc, 20)
    if len(FL):
        el, er = (FL ** 2).mean(1), (FR ** 2).mean(1)
        act = (el + er) >= (el + er).max() * 0.05
        pan = 10 * np.log10((el[act] + 1e-14) / (er[act] + 1e-14))
        r['stereo']['pan_db'] = {'min': round(float(pan.min()), 1), 'max': round(float(pan.max()), 1),
                                 'travel': round(float(pan.max() - pan.min()), 1),
                                 'start': round(float(pan[0]), 1), 'end': round(float(pan[-1]), 1)}
    if cut is not None:
        r['cut'] = round(cut, 3)
        r['peak_vs_cut_ms'] = round((r['t_peak'] - cut) * 1000, 0)
        r['attack_vs_cut_ms'] = round((r['t_attack'] - cut) * 1000, 0)
        r['end_vs_cut_ms'] = round((t1 - cut) * 1000, 0)
    return r
