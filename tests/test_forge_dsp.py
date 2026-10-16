"""Fast tests for the forge sound toolkit v2 (earshot/forge/dsp.py sources + layering, earshot/forge/fx.py effects):
shape, finite, level (no clipping unless asked), determinism and that each tool does what it claims, measured on
synthetic signals. No models are loaded."""
import numpy as np
import pytest
from scipy import ndimage, signal

from earshot.forge import dsp, fx, tone

SR = 48000


def ok(y, peak=1.0):
    """finite, non-silent, and no sample above `peak` (1.0 = 0 dBFS)"""
    y = np.asarray(y)
    assert np.isfinite(y).all()
    assert np.abs(y).max() > 1e-6
    assert np.abs(y).max() <= peak + 1e-9, np.abs(y).max()
    return y


def same(f):
    """f(rng) twice with the same seed gives identical samples"""
    a, b = f(dsp.rng(5)), f(dsp.rng(5))
    assert np.array_equal(a, b)
    return a


def peak_hz(x, lo=20, hi=20000):
    F = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    s = (f >= lo) & (f <= hi)
    return float(f[s][np.argmax(F[s])])


def ring_stats(ir, rt60, t1=0.1, band=(100, 6000)):
    """decay-compensated late tail: 99.9th percentile (dB) of the long-FFT spectrum over its local mean. Gaussian
    noise (a real late field) sits near 8 dB; resolvable modes (metallic ringing) push it to 11-16 dB"""
    m = ir if ir.ndim == 1 else ir[:, 0]
    seg = m[int(t1 * SR):int((t1 + 1.5 * rt60) * SR)]
    seg = seg * np.exp(6.91 * (np.arange(len(seg)) / SR + t1) / rt60)
    P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    sm = ndimage.uniform_filter1d(P, max(9, int(60 / f[1])))
    s = (f > band[0]) & (f < band[1])
    return float(np.percentile(10 * np.log10(P[s] / sm[s]), 99.9))


def band_rt60(x, lo, hi):
    b = dsp.butter(x, 'band', (lo, hi), order=3)
    e = np.cumsum((b ** 2)[::-1])[::-1]
    e = 10 * np.log10(e / e[0] + 1e-30)
    return (np.argmax(e < -25) - np.argmax(e < -5)) / SR * 3


IMP = np.eye(1, 4800, 0)[0]


# ------------------------------------------------------------------------------------------------ dsp: sources
def test_curve_and_sweep():
    n = dsp.n_of(1.0)
    assert np.allclose(dsp.curve(3.0, 10), 3.0)
    c = dsp.curve([(0, 0.0), (1.0, 1.0)], n)
    assert len(c) == n and abs(c[n // 2] - 0.5) < 0.01
    assert len(dsp.curve(np.array([0.0, 1.0]), 100)) == 100
    f = dsp.sweep(100, 1600, 1.0)
    assert abs(f[0] - 100) < 1e-9 and abs(f[-1] - 1600) < 1e-6
    assert abs(f[n // 2] - 400) < 1                     # exponential: half time = half the octaves
    e = dsp.sweep(100, 1600, 1.0, 'ease', power=2.0)
    assert e[n // 2] < f[n // 2] and np.all(np.diff(e) >= 0)


@pytest.mark.parametrize('shape', ['sine', 'saw', 'square', 'tri'])
def test_osc_shapes(shape):
    y = dsp.osc(440, shape, 0.5)
    assert len(y) == SR // 2 and np.isfinite(y).all()
    assert 0.9 < np.abs(y).max() < 1.15
    assert abs(peak_hz(y) - 440) < 3


def test_osc_saw_is_band_limited():
    """a 5 kHz saw: harmonics land on 5, 10, 15, 20 kHz; everything else would be aliasing"""
    y = dsp.osc(5000, 'saw', 1.0)
    F = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    harm = np.zeros(len(f), bool)
    for k in range(1, 5):
        harm |= np.abs(f - 5000 * k) < 20
    alias = 10 * np.log10(F[~harm & (f > 30)].sum() / F[harm].sum())
    naive = 2 * ((5000 * np.arange(SR) / SR) % 1.0) - 1
    Fn = np.abs(np.fft.rfft(naive * np.hanning(SR))) ** 2
    alias_naive = 10 * np.log10(Fn[~harm & (f > 30)].sum() / Fn[harm].sum())
    assert alias < alias_naive - 8 and alias < -20


def test_shepard_unit_rms_deterministic():
    for kind in ('tone', 'noise'):
        y = same(lambda r: dsp.shepard(1.0, r, kind=kind))
        assert abs(np.sqrt(np.mean(y ** 2)) - 1) < 1e-6 and np.isfinite(y).all()
    up, dn = dsp.shepard(1.0, dsp.rng(1), up=True), dsp.shepard(1.0, dsp.rng(1), up=False)
    assert not np.allclose(up, dn)


def test_shepard_voices_glide_in_the_right_direction():
    """a single voice region: the strongest frequency inside 400-800 Hz moves up for up=True, down for up=False"""
    for up, sgn in ((True, 1), (False, -1)):
        y = dsp.shepard(2.0, dsp.rng(2), up=up, rate=0.5, f_lo=50, f_hi=12800)
        a, b = peak_hz(y[:8192], 400, 800), peak_hz(y[SR // 4:SR // 4 + 8192], 400, 800)
        d = np.log2(b / a) % 1.0                         # octave-wrapped move of the nearest voice
        assert (d < 0.5) == (sgn > 0)


def test_wander_and_air():
    c = same(lambda r: dsp.wander(2.0, r, rate=5))
    assert abs(c.mean()) < 1e-6 and abs(c.std() - 1) < 1e-6 and len(c) == 2 * SR
    assert np.abs(np.diff(c)).max() < 0.01               # smooth (a control signal, not audio noise)
    y = same(lambda r: dsp.air(1.5, r, center=1000, flutter=0.0, bright=0.0))
    assert abs(np.sqrt(np.mean(y ** 2)) - 1) < 1e-6
    f, P = signal.welch(y, SR, nperseg=8192)
    psd = lambda hz: 10 * np.log10(np.mean(P[(f > hz / 1.2) & (f < hz * 1.2)]))
    shape = lambda u: -10 * np.log10(2 ** (-6.0 / 3.0103 * u) + 2 ** (4.5 / 3.0103 * u))   # the documented curve
    for u in (-2, 2, 3):                                  # rises 6 dB / oct below 1 kHz, falls 4.5 dB / oct above
        assert abs((psd(1000 * 2 ** u) - psd(1000)) - (shape(u) - shape(0))) < 1.5
    g = dsp.air(2.0, dsp.rng(3), flutter=0.8, whistle=0.3)
    env = np.sqrt(ndimage.uniform_filter1d(g ** 2, 2400))
    assert np.percentile(env, 95) / np.percentile(env, 5) > 2.5   # it gusts


def test_sub_drop_pitch_falls_and_peaks_at_minus_1():
    y = ok(same(lambda r: dsp.sub_drop(1.5, r, f0=150, f1=35, drop=0.2, click=0.0)), 0.8913)
    assert abs(20 * np.log10(np.abs(y).max()) + 1) < 0.01
    early, late = peak_hz(y[:4096], 20, 400), peak_hz(y[int(0.8 * SR):int(0.8 * SR) + 16384], 20, 400)
    assert early > 90 and 30 < late < 45


def test_bell_partials_and_decay():
    y = ok(same(lambda r: dsp.bell(1000, 1.5, r, 'bar', decay=0.6, strike=0.0)), 0.8913)
    assert abs(peak_hz(y, 500, 1500) - 1000) < 5
    assert abs(peak_hz(y, 2000, 4000) - 2756) < 15      # the free-free bar's inharmonic second mode
    e = lambda a, b: np.mean(y[int(a * SR):int(b * SR)] ** 2)
    assert e(0.6, 0.9) < e(0.0, 0.3) * 0.2
    hi = dsp.butter(y, 'high', 4000)
    assert np.mean(hi[int(0.5 * SR):] ** 2) < np.mean(hi[:int(0.2 * SR)] ** 2) * 0.05   # highs die first
    for name in dsp.BELLS:
        ok(dsp.bell(440, 0.5, dsp.rng(1), name))


def test_fm_inharmonic():
    y = ok(dsp.fm(500, 1.4, 4, 1.0, decay=0.5, index_decay=0.2), 1.0)
    F = np.abs(np.fft.rfft(y[:16384] * np.hanning(16384)))
    f = np.fft.rfftfreq(16384, 1 / SR)
    side = F[np.abs(f - 700) < 10].max() + F[np.abs(f - 1200) < 10].max()   # fc + fm, fc + 2 fm - ...
    assert side > 0.05 * F.max()


# ------------------------------------------------------------------------------------------------ dsp: layering
def test_onset_align_stack_xfade():
    x = np.zeros(4800)
    x[1000:1200] = 1.0
    assert abs(dsp.onset(x) * SR - 1000) < 1
    assert abs(dsp.onset(dsp.align(x, 0.05)) - 0.05) < 1 / SR
    assert abs(dsp.onset(dsp.align(x, 0.0))) < 1 / SR
    m = dsp.stack([(x, 0.0, 0.0), (x, 0.01, -6.0)])
    assert m.ndim == 1 and len(m) == 4800 + 480
    s = dsp.stack([(x, 0.0, 0.0, -1.0), {'x': x, 't': 0.0, 'gain': 0.0, 'pan': 1.0}])
    assert s.shape == (4800, 2) and abs(s[1100, 0] - s[1100, 1]) < 1e-9   # one hard left + one hard right
    al = dsp.stack([(x, 0.02, 0.0), (np.r_[np.zeros(300), x], 0.02, 0.0)], onset_db=-24)
    assert abs(dsp.onset(al) - 0.02) < 1 / SR and al.max() == pytest.approx(2.0)   # onsets line up exactly
    ok(dsp.stack([(dsp.sub_drop(0.5), 0, 0), (dsp.sub_drop(0.5), 0, 0)], peak_db=-1.0), 0.8913)
    a, b = np.ones(1000), np.ones(2000)
    lin = dsp.xfade(a, b, 0.005, power=False)
    assert len(lin) == 3000 - 240 and np.allclose(lin, 1.0)
    pw = dsp.xfade(np.ones((1000, 2)), np.ones((500, 2)), 0.005)
    assert pw.shape == (1500 - 240, 2)


# ------------------------------------------------------------------------------------------------ fx: reverb
def test_reverb_shapes_and_determinism():
    x = dsp.sub_drop(0.3)
    m = same(lambda r: fx.reverb(x, r, rt60=0.8, predelay=0.02, wet=0.3))
    assert m.ndim == 1 and len(m) == len(x) + dsp.n_of(0.02) + dsp.n_of(1.2 * 0.8 * 1.2) - 1
    st = fx.reverb(x, dsp.rng(1), rt60=0.8, stereo=True)
    assert st.shape[1] == 2 and np.isfinite(st).all()
    s2 = fx.reverb(np.stack([x, -x], axis=1), dsp.rng(1), rt60=0.5)
    assert s2.ndim == 2
    assert np.allclose(fx.reverb(x, dsp.rng(1), wet=0.0, rt60=0.5)[:len(x)], x)   # dry path untouched
    assert np.abs(fx.reverb(x, dsp.rng(1), rt60=0.5, wet=0.2)[-100:]).max() < 1e-3  # faded, no cut-off click


def test_reverb_decay_and_unit_energy():
    """band rt60s follow the design (mid, damped highs, longer bass) and the wet IR has about unit energy"""
    ir = fx.reverb(IMP, dsp.rng(2), rt60=1.5, damp=0.0, bass=1.0, wet=1.0, dry=0.0, early_db=None, stereo=True,
                   air=0.0)
    assert 0.7 < (ir ** 2).sum(axis=0).mean() < 1.25
    assert abs(band_rt60(ir[:, 0], 707, 1414) - 1.5) < 0.2
    ir = fx.reverb(IMP, dsp.rng(2), rt60=1.5, damp=0.6, bass=1.5, wet=1.0, dry=0.0, stereo=True)
    mid, hi8, low = band_rt60(ir[:, 0], 707, 1414), band_rt60(ir[:, 0], 5657, 11314), band_rt60(ir[:, 0], 60, 120)
    assert hi8 < 0.75 * mid and low > 1.2 * mid
    assert abs(np.corrcoef(ir[:, 0], ir[:, 1])[0, 1]) < 0.3            # a wide, decorrelated stereo tail
    dry_air = fx.reverb(IMP, dsp.rng(2), rt60=3.0, damp=0.3, wet=1.0, dry=0.0, stereo=True, air=0.0)
    wet_air = fx.reverb(IMP, dsp.rng(2), rt60=3.0, damp=0.3, wet=1.0, dry=0.0, stereo=True, air=1.0)
    assert band_rt60(wet_air[:, 0], 8500, 17000) < 0.45 * band_rt60(dry_air[:, 0], 8500, 17000)   # air absorption
    assert band_rt60(wet_air[:, 0], 8500, 17000) < 1.0                   # 12 kHz cannot ring for seconds in air
    assert abs(band_rt60(wet_air[:, 0], 707, 1414) - band_rt60(dry_air[:, 0], 707, 1414)) < 0.3   # mids: walls
    lc = fx.reverb(IMP, dsp.rng(2), rt60=2.0, wet=1.0, dry=0.0, stereo=True, lowcut=200)
    assert (dsp.butter(lc[:, 0], 'low', 80) ** 2).sum() < 0.01 * (lc[:, 0] ** 2).sum()


def test_reverb_tail_does_not_ring():
    """the long tail must read like a real late field (Gaussian noise), not like a metallic resonator: its long-FFT
    peakiness stays within 1.5 dB of a pure noise tail, where a pure FDN shows ringing modes"""
    rt = 3.0
    ref = ring_stats(dsp.room_ir(dsp.rng(1), rt60=rt, damp=0.0, seconds=rt * 1.6), rt)
    kw = dict(rt60=rt, damp=0.0, bass=1.0, wet=1.0, dry=0.0, early_db=None, stereo=True, air=0.0)
    hyb = fx.reverb(IMP, dsp.rng(3), **kw)
    fdn = fx.reverb(IMP, dsp.rng(3), late='fdn', **kw)
    assert ring_stats(hyb, rt) < ref + 1.5
    assert ring_stats(fdn, rt) > ring_stats(hyb, rt) + 1.0             # the measurement does see FDN modes
    late = hyb[int(0.2 * SR):int(2.0 * SR), 0]
    fr = late[:len(late) // 480 * 480].reshape(-1, 480)
    assert abs(np.median((fr ** 4).mean(1) / (fr ** 2).mean(1) ** 2) - 3) < 0.3   # dense: Gaussian, no flutter


def test_early_reflections():
    er = fx.early(IMP, dsp.rng(1), size=0.5)
    assert er.shape[1] == 2 and np.allclose((er ** 2).sum(axis=0), 1.0, atol=0.05)
    first = np.nonzero(np.abs(er[:, 0]) > 0.05 * np.abs(er[:, 0]).max())[0][0] / SR
    assert 0.002 < first < 0.01


# ------------------------------------------------------------------------------------------------ fx: dynamics
def alias_db(y, f0=7000):
    F = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    harm = f < 20
    for k in range(1, 4):
        harm |= np.abs(f - f0 * k) < 30
    return 10 * np.log10(F[~harm].sum() / F[harm].sum())


@pytest.mark.parametrize('mode', ['tanh', 'tape', 'hard', 'fold'])
def test_saturate_oversampled(mode):
    x = 0.9 * np.sin(2 * np.pi * 7000 * np.arange(SR // 2) / SR)
    y = fx.saturate(x, 18, mode)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert abs(np.sqrt(np.mean(y ** 2)) - np.sqrt(np.mean(x ** 2))) < 1e-6        # level='rms'
    assert alias_db(y) < -35 and alias_db(y) < alias_db(fx.saturate(x, 18, mode, os=1)) - 20
    pk = fx.saturate(dsp.sub_drop(0.4), 12, mode, level='peak')
    assert abs(np.abs(pk).max() - 0.8913) < 1e-3
    st = fx.saturate(np.stack([x, x], axis=1), 6, mode)
    assert st.shape == (len(x), 2)


def test_saturate_adds_harmonics():
    x = 0.8 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)
    y = fx.saturate(x, 12, 'tanh')
    F = np.abs(np.fft.rfft(y))
    assert F[600] > 0.05 * F[200]                       # odd harmonic (1 Hz bins)
    y = fx.saturate(x, 12, 'tape')
    F = np.abs(np.fft.rfft(y))
    assert F[400] > 0.005 * F[200]                      # tape bias: even harmonics too


def seg_db(y, a, b):
    return 10 * np.log10(np.mean(y[int(a * SR):int(b * SR)] ** 2) + 1e-20)


def test_transient_shaper():
    r = dsp.rng(3)
    x = dsp.excite(dsp.modal([300, 800, 1900], [0.08, 0.05, 0.02], [1, 0.6, 0.3], 0.8, r), dsp.contact(0.001, r))
    y = fx.transient(x, attack_db=6)
    assert seg_db(y, 0, 0.004) - seg_db(x, 0, 0.004) > 5 and abs(seg_db(y, 0.1, 0.3) - seg_db(x, 0.1, 0.3)) < 1
    y = fx.transient(x, attack_db=-6)
    assert seg_db(y, 0, 0.004) - seg_db(x, 0, 0.004) < -4.5
    y = fx.transient(x, attack_db=0, sustain_db=6)
    assert abs(seg_db(y, 0, 0.004) - seg_db(x, 0, 0.004)) < 0.5 and seg_db(y, 0.1, 0.3) - seg_db(x, 0.1, 0.3) > 4
    assert tone.t40_ms(y) > tone.t40_ms(x) + 30          # a longer tail (the v1 impact's weakness)
    sine = np.sin(2 * np.pi * 440 * np.arange(SR) / SR) * np.minimum(1, np.arange(SR) / 480)
    y = fx.transient(sine, 6, 6)
    assert abs(seg_db(y, 0.2, 0.9) - seg_db(sine, 0.2, 0.9)) < 1.0     # a steady tone is left alone


def test_limit_ceiling_true_peak():
    x = dsp.sub_drop(1.0) * 6 + 0.3 * dsp.noise(1.0, dsp.rng(1))
    y = fx.limit(x, -1.0)
    assert y.shape == x.shape and np.abs(y).max() <= 10 ** (-1 / 20) + 1e-12
    up = signal.resample_poly(y, 4, 1)
    assert 20 * np.log10(np.abs(up).max()) < -0.8
    q = 0.3 * np.sin(2 * np.pi * 100 * np.arange(SR) / SR)
    assert np.allclose(fx.limit(q, -1.0), q)             # nothing over the ceiling: untouched
    st = fx.limit(np.stack([x, 0.5 * x], axis=1), -3.0)
    assert np.abs(st).max() <= 10 ** (-3 / 20) + 1e-12
    assert np.allclose(st[:, 1], 0.5 * st[:, 0])         # linked: the stereo image does not shift


def test_split_sums_flat():
    bands = fx.split(np.eye(1, 16384, 100)[0], [150, 1200, 6000])
    assert len(bands) == 4
    H = np.abs(np.fft.rfft(sum(bands)))
    assert np.abs(20 * np.log10(H[5:])).max() < 0.05     # an all-pass: flat magnitude
    lo = np.abs(np.fft.rfft(bands[0]))
    f = np.fft.rfftfreq(16384, 1 / SR)
    assert lo[np.argmin(np.abs(f - 2000))] < 0.01 * lo[np.argmin(np.abs(f - 50))]


def test_eq_match_moves_tone_toward_target():
    pk, wh = dsp.noise(1.0, dsp.rng(1), 'pink'), dsp.noise(1.0, dsp.rng(2))
    d = lambda a, b: np.sqrt(np.mean((tone.profile(a) - tone.profile(b)) ** 2))
    for ph in ('min', 'linear'):
        y = fx.eq_match(pk, wh, phase=ph)
        assert len(y) == len(pk) and np.isfinite(y).all()
        assert d(y, wh) < 0.25 * d(pk, wh)
        assert abs(np.sqrt(np.mean(y ** 2)) - np.sqrt(np.mean(pk ** 2))) < 1e-6
    half = fx.eq_match(pk, wh, amount=0.5)
    assert d(wh, wh) < d(half, wh) < d(pk, wh)
    g = fx.match_curve(pk, [wh, dsp.noise(1.0, dsp.rng(3))])
    assert len(g) == len(tone.CEN) and g[-1] > g[0] + 10          # pink -> white: tilt up
    assert max(abs(v) for v in fx.match_curve(pk, wh, max_db=6)) <= 6.0
    hit = dsp.excite(dsp.modal([620, 1490], [0.03, 0.02], [1, .5], 0.2, dsp.rng(4)), dsp.contact(0.001, dsp.rng(4)))
    y = fx.eq_bands(hit, [0.0] * len(tone.CEN))
    assert np.max(np.abs(y - hit)) < 1e-3 * np.abs(hit).max()      # a flat curve changes nothing


# ------------------------------------------------------------------------------------------------ fx: motion
def test_doppler_pitch_level_pan():
    src = np.sin(2 * np.pi * 1000 * np.arange(3 * SR) / SR)
    y = fx.doppler(src, speed=40, dist=3)
    assert y.ndim == 2 and np.isfinite(y).all()
    m = y.mean(axis=1)
    assert abs(peak_hz(m[:8192]) - 1000 * 343 / (343 - 40)) < 12        # approaching: higher
    assert abs(peak_hz(m[-8192:]) - 1000 * 343 / (343 + 40)) < 12       # receding: lower
    rms = lambda a: np.sqrt(np.mean(a ** 2))
    assert rms(y[:8192, 0]) > 3 * rms(y[:8192, 1]) and rms(y[-8192:, 1]) > 3 * rms(y[-8192:, 0])
    mid = np.argmax(np.abs(m)) / len(m)
    assert 0.35 < mid < 0.65                             # loudest when closest
    far = fx.doppler(dsp.noise(3, dsp.rng(1)), speed=40, dist=3, stereo=False, air=3.0)
    near = dsp.butter(far, 'high', 8000)
    a, b = rms(near[:SR // 4]), rms(near[len(far) // 2 - SR // 8:len(far) // 2 + SR // 8])
    assert a < 0.3 * b                                   # far away: the highs are absorbed


def test_pan_width_decorrelate():
    x = dsp.noise(1.0, dsp.rng(1))
    p = fx.pan(x, [(0, -1.0), (1.0, 1.0)])
    assert p.shape == (len(x), 2)
    assert np.abs(p[:100, 1]).max() < 0.03 and np.abs(p[-100:, 0]).max() < 0.03
    assert np.allclose(fx.pan(x, 0.0), np.stack([x, x], axis=1))
    d = fx.decorrelate(x, 1.0)
    assert d.shape == (len(x), 2) and abs(np.corrcoef(d[:, 0], d[:, 1])[0, 1]) < 0.7
    lo = dsp.butter(d, 'low', 100)
    assert np.corrcoef(lo[:, 0], lo[:, 1])[0, 1] > 0.95  # the low end stays mono
    w = fx.width(d, [(0, 0.0), (1.0, 2.0)])
    assert np.allclose(w[:50, 0], w[:50, 1], atol=1e-2)
    assert np.allclose(fx.width(d, 1.0), d)


def test_sweep_filter_moves():
    y = fx.sweep_filter(dsp.noise(2.0, dsp.rng(1)), 300, 6000, q=8)
    c = lambda s: np.sum(np.abs(np.fft.rfft(s)) * np.fft.rfftfreq(len(s), 1 / SR)) / np.sum(np.abs(np.fft.rfft(s)))
    assert c(y[:SR // 4]) < 0.5 * c(y[-SR // 4:])
    st = fx.sweep_filter(np.ones((1000, 2)), 9000, 200)
    assert st.shape == (1000, 2)


def test_varispeed_and_tape_stop():
    x = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    y = fx.varispeed(x, 2.0)
    assert abs(len(y) - SR / 2) < 3 and abs(peak_hz(y[:8192]) - 2000) < 10
    y = fx.varispeed(x, [(0, 1.0), (0.5, 0.5)])
    assert abs(len(y) - SR // 2) < 2                     # breakpoints set the output length
    assert abs(peak_hz(y[:4096]) - 1000 * (1 - 0.5 * 2048 / 24000)) < 15 and abs(peak_hz(y[-4096:]) - 543) < 15
    t = fx.tape_stop(x, start=0.5, seconds=0.5)
    assert np.isfinite(t).all() and len(t) > SR * 0.6
    assert peak_hz(t[int(0.35 * SR):int(0.35 * SR) + 4096]) > 950
    assert peak_hz(t[-6000:-1500]) < 400                 # sliding down to a stop
    assert abs(t[-1]) < 1e-3


# ------------------------------------------------------------------------------------------------ fx: glitch + grains
def test_crush():
    y = fx.crush(np.linspace(-1, 1, 1000), bits=3)
    assert len(np.unique(y)) <= 9
    s = fx.crush(np.sin(2 * np.pi * 100 * np.arange(4800) / SR), bits=None, rate=8000)
    assert np.all(s[::6] == s[1::6])                     # held for 48000 / 8000 = 6 samples
    assert np.allclose(fx.crush(s, 4, mix=0.0), s)


def test_stutter_and_glitch():
    x = dsp.fm(300, 1.5, 2, 0.6, decay=0.5)
    y = fx.stutter(x, 0.1, 0.05, repeats=4, gate=0.8, decay_db=-3)
    assert len(y) == len(x) + 3 * dsp.n_of(0.05) and np.isfinite(y).all()
    i0, L = dsp.n_of(0.1), dsp.n_of(0.05)
    r1, r2 = y[i0:i0 + L], y[i0 + L:i0 + 2 * L]
    assert np.sqrt(np.mean(r2 ** 2)) < np.sqrt(np.mean(r1 ** 2)) * 0.8
    assert np.abs(y[i0 + int(0.85 * L):i0 + L]).max() == 0      # gated
    g = ok(same(lambda r: fx.glitch(x, r, grid=0.03)), np.abs(x).max() + 1e-9)
    assert len(g) == len(x) and not np.allclose(g, x)
    st = fx.glitch(np.stack([x, x], axis=1), dsp.rng(1))
    assert st.shape == (len(x), 2)


def test_granular_cloud():
    src = dsp.bell(880, 1.0, dsp.rng(1), 'glass', decay=0.6)
    c = same(lambda r: fx.granular(src, 2.0, r, grain=0.05, density=50, pitch=12, pitch_jitter=0.1))
    assert c.shape == (2 * SR, 2) and np.isfinite(c).all()
    assert abs(peak_hz(c.mean(axis=1), 1000, 3000) - 1760) < 40          # an octave up
    assert 0.2 < np.sqrt(np.mean(c ** 2)) / np.sqrt(np.mean(src[:SR // 2] ** 2)) < 3
    assert abs(np.corrcoef(c[:, 0], c[:, 1])[0, 1]) < 0.95               # spread across the stereo field


def test_reverse_swell_lands_on_the_hit():
    hit = dsp.sub_drop(0.8, dsp.rng(1), click=0.5)
    y = fx.reverse_swell(hit, dsp.rng(2), seconds=0.9, rt60=1.5, swell_db=-6)
    assert y.ndim == 2 and np.isfinite(y).all()
    m = np.abs(y).max(axis=1)
    env = ndimage.uniform_filter1d(m[:dsp.n_of(0.9)], 960)
    assert env[-2000:].mean() > 4 * env[:5000].mean()     # it swells up into the hit
    assert abs(dsp.onset(y[dsp.n_of(0.85):], -6) + 0.85 - 0.9) < 0.01  # the hit lands at `seconds`
    sw = np.abs(y[:dsp.n_of(0.89)]).max()
    assert abs(20 * np.log10(sw / np.abs(y).max()) + 6) < 1.5
