"""earshot.forge.fit.gesture (tones + air) and the spectral additions it rides on (detail fit, trajectory, motion
knobs, the ridge cache): known structure recovered from synthetic signals, render knobs, determinism, serialisation."""
import json

import numpy as np
import pytest

from earshot.forge.fit import gesture as G
from earshot.forge.fit import spectral as sp

SR = 48000


def _band(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def _whoosh(n_s=0.7, seed=0, tone=True, stereo=True):
    """band noise swelling to a peak at 0.3 s, a gliding tone (1.5 -> 3 kHz) over it; the right channel quieter"""
    rng = np.random.default_rng(seed)
    n = int(n_s * SR)
    t = np.arange(n) / SR
    env = np.clip(t / 0.3, 0, 1) ** 2 * np.exp(-np.maximum(t - 0.3, 0) / 0.1)
    x = 0.1 * _band(rng.standard_normal(n), 300, 12000) * env
    if tone:
        f = 1500 * 2 ** (t / t[-1])
        fade = np.clip(t / 0.2, 0, 1) * np.clip((n_s - 0.05 - t) / 0.1, 0, 1)
        x = x + 0.05 * np.sin(2 * np.pi * np.cumsum(f) / SR) * fade
    return np.stack([x, 0.6 * x], 1) if stereo else x


def _balance_db(x, y, within_db=40.0):
    """RMS third-octave balance difference over the bands within within_db of x's loudest (the synthetic sources are
    brick-wall band-limited: a band 50 dB under the rest, empty in x, would rule an unweighted RMS)"""
    from earshot.forge.tone import profile
    px, py = profile(x.mean(1) if x.ndim == 2 else x), profile(y.mean(1) if y.ndim == 2 else y)
    sel = px > px.max() - within_db
    return float(np.sqrt(np.mean((px[sel] - py[sel]) ** 2)))


def _band_db(y, lo, hi):
    m = y.mean(1) if y.ndim == 2 else y
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return 10 * np.log10(P[(f >= lo) & (f < hi)].sum() + 1e-30)


def test_residual_takes_a_tone_out_and_leaves_the_rest():
    rng = np.random.default_rng(1)
    n = SR // 2
    t = np.arange(n) / SR
    noise = 0.05 * _band(rng.standard_normal(n), 200, 12000)
    tone = 0.2 * np.sin(2 * np.pi * 2000 * t)
    r = G.residual(noise + tone, tone)
    assert _band_db(noise + tone, 1950, 2050) - _band_db(r, 1950, 2050) > 12      # the tone is gone
    assert abs(_band_db(r, 4000, 10000) - _band_db(noise, 4000, 10000)) < 0.5      # the noise stays
    assert np.abs(G.residual(noise, noise)).max() < 1e-9                          # nothing left of itself
    assert np.allclose(G.residual(noise, np.zeros(n)), noise, atol=1e-9)           # a silent Y changes nothing
    # never adds what Y has and X has not (a waveform subtraction would)
    assert _band_db(G.residual(noise, tone), 1950, 2050) <= _band_db(noise, 1950, 2050) + 0.01


def test_gesture_fit_splits_tones_and_air():
    x = _whoosh()
    m = G.fit(x, fast=True)
    assert m['kind'] == G.KIND and m['channels'] == 2
    assert m['tones'] is not None and m['fit']['tone_share'] > 0.2
    assert m['air']['kind'] == sp.KIND and m['air']['stereo'] is not None
    tr = m['traj']
    assert abs(tr['peak_s'] - 0.3) < 0.05 and tr['rise_s'] > 0.1
    assert np.median(tr['pan_db']) == pytest.approx(10 * np.log10(1 / 0.36), abs=1.5)   # L 4.4 dB over R
    y = G.render(m, seed=0)
    assert y.shape == x.shape
    q = sp.compare(x, y)
    assert q['env_corr'] > 0.9 and _balance_db(x, y) < 2.0
    # the tones keep the glide: the 1.5-3 kHz band holds the render's tonal energy like the original's
    assert abs(_band_db(y, 1400, 3200) - _band_db(x, 1400, 3200)) < 2.0


def test_air_only_and_tone_floor():
    x = _whoosh(tone=False, seed=2)
    m = G.fit(x, fast=True)
    assert m['tones'] is None or m['fit']['tone_share'] < 0.5
    m0 = G.fit(x, fast=True, tones=False)
    assert m0['tones'] is None and m0['fit']['partials'] == 0
    m1 = G.fit(x, fast=True, tone_min=1.1)              # a floor nothing reaches: air only
    assert m1['tones'] is None
    assert G.render(m0, seed=1).shape == x.shape


def test_render_determinism_fresh_air_and_json(tmp_path):
    m = G.fit(_whoosh(seed=3), fast=True, name='synthetic')
    p = tmp_path / 'g.json'
    G.save(m, str(p))
    m2 = G.load(str(p))
    assert json.loads(json.dumps(m)) == m2
    a, b = G.render(m, seed=5), G.render(m2, seed=5)
    assert np.array_equal(a, b)
    c = G.render(m, seed=6)
    d = a - c                                            # the tones are the same, the air is new noise
    assert 0.05 < np.sqrt((d ** 2).mean() / (a ** 2).mean()) < 1.5
    assert G.summary(m).startswith('synthetic:')
    with pytest.raises(ValueError):
        G.render({'kind': 'sines', 'version': 1})


def test_render_knobs():
    m = G.fit(_whoosh(seed=4), fast=True)
    base = G.render(m, seed=1)
    assert len(G.render(m, seed=1, length=1.5)) == pytest.approx(1.5 * len(base), abs=4)
    assert np.abs(G.render(m, seed=1, gain_db=-6.0)).max() < np.abs(base).max()
    quiet_air = G.render(m, seed=1, air_db=-20.0)
    assert _band_db(quiet_air, 8000, 12000) < _band_db(base, 8000, 12000) - 6      # the air carries the highs
    mono = G.render(m, seed=1, stereo=False)
    assert mono.ndim == 1
    take = G.render(m, seed=2, vary=0.3, ring=True)
    assert len(take) >= len(base) and np.abs(take[-8:]).max() < 1e-3 * np.abs(take).max() + 1e-9   # faded end
    # motion: a pass-by pan moves the energy from the left channel to the right
    y = G.render(m, seed=1, motion=0.8, pan=0.0)
    h = int(m['traj']['peak_s'] * SR)
    lr = lambda s: 10 * np.log10((s[:, 0] ** 2).sum() / (s[:, 1] ** 2).sum())
    assert lr(y[:h]) > lr(y[h:]) + 3


def test_bend_is_a_doppler_pitch_turn():
    n = SR // 2
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * 1000 * t)
    b = sp.bend(y, 0.25, 0.03, 2.0)

    def freq(s):
        z = np.nonzero(np.diff(np.signbit(s)))[0]
        return 0.5 * SR / np.mean(np.diff(z))
    early, late = freq(b[int(0.05 * SR):int(0.15 * SR)]), freq(b[int(0.35 * SR):int(0.45 * SR)])
    assert early == pytest.approx(1000 * 2 ** (2 / 12), rel=0.02)
    assert late == pytest.approx(1000 * 2 ** (-2 / 12), rel=0.02)
    assert len(b) > n                                    # the receding tail is read out more slowly
    assert sp.bend(y, 0.25, 0.03, 0.0) is not None and np.array_equal(sp.bend(y, 0.25, 0.03, 0.0), y)


def test_spectral_detail_fit_keeps_a_stutter_gate():
    rng = np.random.default_rng(5)
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    gate = (np.floor(t / 0.04) % 2 == 0).astype(float)          # 40 ms on, 40 ms off
    x = 0.1 * _band(rng.standard_normal(n), 300, 12000) * gate
    lv = lambda m: sp._db((10 ** (np.asarray(m['env']['db']) / 10) * sp.bandwidths(sp.centers())).sum(1))
    smooth = sp.fit(x, texture=False, ridges=False)
    detail = sp.fit(x, texture=False, ridges=False, **{k: v for k, v in sp.DETAIL.items() if k != 'max_ridges'})
    assert detail['fit']['rank'] == 0
    assert np.ptp(lv(detail)) > np.ptp(lv(smooth)) + 6          # the gaps stay deep
    q = sp.compare(x, sp.render(detail, seed=1))
    assert q['env_corr'] > sp.compare(x, sp.render(smooth, seed=1))['env_corr']


def test_trajectory_reads_sweep_pan_and_width():
    rng = np.random.default_rng(6)
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    lo = _band(rng.standard_normal(n), 300, 1200) * np.clip(1 - t / 0.8, 0, 1)
    hi = _band(rng.standard_normal(n), 3000, 9000) * np.clip(t / 0.8, 0, 1)
    x = 0.1 * (lo + hi)
    m = sp.fit(np.stack([x, 0.5 * x], 1), texture=False, ridges=False)
    tr = sp.trajectory(m)
    c = np.asarray(tr['centroid_hz'])
    assert c[-1] > 2 * c[0]                                      # an upward sweep
    assert np.median(tr['pan_db']) == pytest.approx(6.0, abs=1.0)
    assert np.median(tr['width']) < 0.05                         # fully correlated
    y = sp.render(m, seed=2)
    assert np.array_equal(y, sp.render(m, seed=2, doppler=0.0, proximity=0.0, pan=1.0, motion=0.0))
    far = sp.render(m, seed=2, proximity=1.0)
    k = int(0.1 * SR)                                            # the start sits under the peak: duller when far
    dull = lambda s: _band_db(s[:k], 300, 1200) - _band_db(s[:k], 3000, 9000)
    assert dull(far) > dull(y)
    flat = sp.render(m, seed=2, pan=0.0).astype(np.float64)
    assert abs(10 * np.log10((flat[:, 0] ** 2).sum() / (flat[:, 1] ** 2).sum())) < 1.0


def test_trajectory_tilt_reads_the_slope_where_the_energy_is():
    rng = np.random.default_rng(8)
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    P = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    P[1:] /= np.sqrt(f[1:] / 1000.0)
    P[0] = 0
    pink = np.fft.irfft(P, n)
    x = 0.05 * (pink * np.clip(1 - t / 0.8, 0, 1) + 0.3 * rng.standard_normal(n) * np.clip(t / 0.8, 0, 1))
    tilt = np.asarray(sp.trajectory(sp.fit(x, texture=False, ridges=False))['tilt_db_oct'])
    assert tilt[0] == pytest.approx(-3.0, abs=1.0)               # pink: -3 dB / octave
    assert tilt[-1] == pytest.approx(0.0, abs=1.0)               # white: flat
    assert tilt[-1] > tilt[0] + 2


def test_ridge_cache_renders_the_same():
    rng = np.random.default_rng(7)
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    x = 0.05 * _band(rng.standard_normal(n), 200, 15000) + 0.05 * np.sin(2 * np.pi * 2500 * t)
    m = sp.fit(x, texture=False)
    assert m['ridges']
    a = sp._mono_render(m, 31, None)
    b = sp._mono_render(m, 31, None, sp._ridge_sig(m, 31))
    assert np.allclose(a, b, atol=1e-9 * np.abs(a).max())


def _struck(n_s=0.4, seed=10):
    """a struck object (three decaying modes from 50 ms) in a band-noise swell; the right channel quieter"""
    rng = np.random.default_rng(seed)
    n = int(n_s * SR)
    t = np.arange(n) / SR
    tt = np.maximum(t - 0.05, 0)
    modes = sum(a * np.sin(2 * np.pi * f * tt) * np.exp(-tt / d) for f, a, d in ((620, 0.3, 0.12), (1710, 0.15, 0.06),
                                                                                (3380, 0.08, 0.03))) * (t >= 0.05)
    x = modes + 0.03 * _band(rng.standard_normal(n), 300, 12000) * np.clip(t / 0.2, 0, 1)
    return np.stack([x, 0.7 * x], 1)


def test_modal_tones_and_gain_tones():
    x = _struck()
    m = G.fit(x, fast=True, tones_kind='modal')
    det = m['tones']
    assert m['fit']['tones_kind'] == 'modal' and det['kind'] == 'modal' and m['fit']['partials'] > 0
    assert det['onset'] == pytest.approx(det['strikes'][0]['t'], abs=1e-5)
    assert det['onset'] == pytest.approx(0.05, abs=0.005)
    y = G.render(m, seed=0)
    assert y.shape == x.shape and sp.compare(x, y)['env_corr'] > 0.9
    assert len(G.render(m, seed=1, length=1.3)) == pytest.approx(1.3 * len(x), abs=4)
    assert 'modal' in G.summary(m)
    level = lambda d: 10 * np.log10((G._tones(d, 0, True, len(x)) ** 2).sum())
    assert level(det) - level(G.gain_tones(det, -20.0)) == pytest.approx(20.0, abs=0.1)
    ts = G.fit(x, fast=True, tones_kind='sines')['tones']   # sines tones: partial levels and strike atoms
    assert level(ts) - level(G.gain_tones(ts, -6.0)) == pytest.approx(6.0, abs=0.1)
    with pytest.raises(ValueError):
        G.fit(x, fast=True, tones_kind='spectral')
