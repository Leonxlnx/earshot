"""earshot.forge.fit.spectral: recover KNOWN parameters from synthetic signals, render knobs, serialisation.
Real-recording checks are marked slow (they need the local refsets registry)."""
import json

import numpy as np
import pytest

from earshot.forge.fit import spectral as sp

SR = 48000


def _band(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def _tilted(n, slope_db_oct, rng, lo=200.0, hi=12000.0):
    """white noise with a power-density slope of `slope_db_oct` dB per octave about 1 kHz, band-limited"""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= 10 ** (slope_db_oct * np.log2(np.maximum(f, 1.0) / 1000.0) / 20)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n)


def _env(n, rise=0.08, tau=0.12, start=0.02):
    t = np.arange(n) / SR
    e = np.clip((t - start) / rise, 0, 1) ** 2
    return e * np.exp(-np.maximum(t - start - rise, 0) / tau)


def test_bands_partition_and_widths():
    c = sp.centers()
    assert len(c) == sp.N_BANDS and np.all(np.diff(c) > 0)
    f = np.linspace(c[0], sp.SR / 2, 5000)
    assert np.allclose(sp.weights(f, c).sum(0), 1.0, atol=1e-9)
    bw = sp.bandwidths(c)
    assert np.all(bw > 0) and bw[-1] > bw[5]
    assert np.all((sp._band_nfft(c) >= 512) & (sp._band_nfft(c) <= 4096))


def test_recovers_spectral_tilt():
    rng = np.random.default_rng(1)
    x = 0.05 * _tilted(int(0.8 * SR), -6.0, rng)
    m = sp.fit(x, texture=False, ridges=False)
    c = sp.centers()
    E = np.asarray(m['env']['db'])
    tk = np.asarray(m['env']['t'])
    mid = E[(tk > 0.15) & (tk < 0.65)].mean(0)
    sel = (c > 500) & (c < 8000)
    slope = np.polyfit(np.log2(c[sel]), mid[sel], 1)[0]
    assert slope == pytest.approx(-6.0, abs=1.0)


def test_recovers_envelope_and_attack():
    rng = np.random.default_rng(2)
    n = int(0.6 * SR)
    e = _env(n)
    x = 0.2 * _band(rng.standard_normal(n), 100, 15000) * e
    m = sp.fit(x, texture=False, ridges=False)
    c = sp.centers()
    bw = sp.bandwidths(c)
    tk = np.asarray(m['env']['t'])
    lv = sp._db((10 ** (np.asarray(m['env']['db']) / 10) * bw).sum(1))
    truth = 20 * np.log10(np.interp(tk, np.arange(n) / SR, e) + 1e-9)
    act = truth > truth.max() - 25
    err = (lv - lv.max()) - (truth - truth.max())
    assert np.median(np.abs(err[act])) < 2.0
    # peak time within 20 ms of the true peak (start + rise)
    assert abs(tk[int(np.argmax(lv))] - 0.10) < 0.02
    y = sp.render(m, seed=4)
    q = sp.compare(x, y)
    assert q['env_corr'] > 0.95 and q['tone_db'] < 3.0


def test_stereo_correlation_and_level_difference():
    rng = np.random.default_rng(3)
    n = int(0.8 * SR)
    a, b = [_band(rng.standard_normal(n), 300, 12000) for _ in range(2)]
    rho = 0.3
    th = 0.5 * np.arccos(rho)
    L = np.cos(th) * a + np.sin(th) * b
    R = np.cos(th) * a - np.sin(th) * b
    g = 10 ** (6.0 / 20)                       # L 6 dB louder
    x = 0.1 * np.stack([L * np.sqrt(2 * g * g / (1 + g * g)), R * np.sqrt(2 / (1 + g * g))], 1)
    m = sp.fit(x, texture=False, ridges=False)
    icc = np.asarray(m['stereo']['icc'])[:, 1:5]
    ild = np.asarray(m['stereo']['ild'])[:, 1:5]
    assert np.median(icc) == pytest.approx(rho, abs=0.12)
    assert np.median(ild) == pytest.approx(6.0, abs=1.0)
    y = sp.render(m, seed=1).astype(np.float64)
    assert np.corrcoef(y[:, 0], y[:, 1])[0, 1] == pytest.approx(rho, abs=0.15)
    assert 10 * np.log10((y[:, 0] ** 2).sum() / (y[:, 1] ** 2).sum()) == pytest.approx(6.0, abs=1.5)


def test_tonal_ridge_tracked():
    rng = np.random.default_rng(4)
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    noise = 0.05 * _band(rng.standard_normal(n), 200, 15000)
    f = 2000 * (3000 / 2000) ** (t / t[-1])
    tone = 0.05 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    m = sp.fit(noise + tone, texture=False)
    assert len(m['ridges']) >= 1
    rd = m['ridges'][0]
    assert rd['f'][0] == pytest.approx(2000, rel=0.04)
    assert rd['f'][-1] == pytest.approx(3000, rel=0.04)
    assert rd['t'][-1] - rd['t'][0] > 0.6
    # pure noise: no ridge
    assert sp.fit(noise, texture=False)['ridges'] == []


def test_flutter_found_and_gaussian_noise_left_plain():
    rng = np.random.default_rng(5)
    n = SR
    t = np.arange(n) / SR
    env = np.minimum(1, t / 0.1) * np.minimum(1, (1 - t) / 0.2)
    plain = 0.1 * _band(rng.standard_normal(n), 200, 12000) * env
    m0 = sp.fit(plain, ridges=False)
    assert m0['texture'] is None or (m0['texture']['am'] is None and m0['texture']['grains'] is None)
    am = plain * (1 + 0.8 * np.sin(2 * np.pi * 40 * t))
    m = sp.fit(am, ridges=False)
    tx = m['texture']
    assert tx['am'] is not None and tx['grains'] is None
    peak = tx['am']['rates'][int(np.argmax(tx['am']['spec']))]
    assert 28 <= peak <= 64
    assert 0.3 < max(tx['am']['depth']) <= sp.AM_MAX_DEPTH      # truth: std of ln(1 + 0.8 sin) = 0.6
    # roughness 0 renders without the flutter: less low-rate envelope variance
    var = lambda y: np.var(sp._env_db(sp._mono(y))[20:-40])
    assert var(sp.render(m, seed=2, roughness=0.0)) < var(sp.render(m, seed=2))


def test_crackle_density_order_of_magnitude():
    rng = np.random.default_rng(6)
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    env = np.minimum(1, t / 0.1) * np.minimum(1, (t[-1] - t) / 0.2)
    k = rng.poisson(300 * n / SR)
    c = np.zeros(n)
    np.add.at(c, rng.integers(0, n, k), np.exp(rng.standard_normal(k)) * rng.choice([-1, 1], k))
    x = _band(c, 200, 12000) * env
    m = sp.fit(0.2 * x / np.abs(x).max(), ridges=False)
    gr = m['texture']['grains']
    assert gr is not None and 100 <= gr['density'] <= 1500


def test_json_roundtrip_determinism_and_fresh_noise(tmp_path):
    rng = np.random.default_rng(7)
    n = int(0.5 * SR)
    x = 0.2 * _band(rng.standard_normal(n), 150, 14000) * _env(n)
    m = sp.fit(x, name='synthetic')
    assert m['kind'] == sp.KIND and m['version'] == sp.VERSION
    p = tmp_path / 'm.json'
    sp.save(m, str(p))
    m2 = sp.load(str(p))
    assert json.loads(json.dumps(m)) == m2
    a, b = sp.render(m, seed=3), sp.render(m2, seed=3)
    assert a.shape == (n, 2) and a.dtype == np.float32
    assert np.array_equal(a, b)
    c = sp.render(m, seed=4)
    assert abs(np.corrcoef(a[:, 0], c[:, 0])[0, 1]) < 0.2          # fresh noise, not the same samples
    q, fl = sp.compare(x, a), sp.compare(a, c)
    assert q['mel_1024'] < fl['mel_1024'] + 1.5                     # as close to the original as to itself


def test_render_knobs():
    rng = np.random.default_rng(8)
    n = int(0.5 * SR)
    x = 0.2 * _band(rng.standard_normal(n), 150, 14000) * _env(n)
    m = sp.fit(x, texture=False)
    base = sp.render(m, seed=1)
    assert len(sp.render(m, seed=1, length=2.0)) == pytest.approx(2 * len(base), abs=2)
    assert len(sp.render(m, seed=1, speed=2.0)) == pytest.approx(len(base) / 2, abs=2)

    def centroid(y):
        P = np.abs(np.fft.rfft(sp._mono(y))) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return (P * f).sum() / P.sum()
    assert centroid(sp.render(m, seed=1, tilt=4.0)) > 1.3 * centroid(base)
    assert centroid(sp.render(m, seed=1, shift=1.0)) > 1.3 * centroid(base)
    assert centroid(sp.render(m, seed=1, speed=2.0)) > centroid(base)
    w0 = sp.render(m, seed=1, width=0.0).astype(np.float64)
    assert np.corrcoef(w0[:, 0], w0[:, 1])[0, 1] > 0.99
    assert np.abs(sp.render(m, seed=1, gain_db=-6.0)).max() < np.abs(base).max()


def test_to_code_executes_and_matches_render():
    rng = np.random.default_rng(9)
    n = int(0.3 * SR)
    x = 0.2 * _band(rng.standard_normal(n), 150, 14000) * _env(n, rise=0.03, tau=0.06)
    m = sp.fit(x, name='tiny', texture=False)
    src = sp.to_code(m)
    ns = {}
    exec(compile(src, 'fitted_tiny.py', 'exec'), ns)
    assert np.array_equal(ns['make'](seed=5), sp.render(m, seed=5))
    assert 'keyframe' in src and sp.summary(m).startswith('tiny:')


def test_compare_identity_and_check():
    rng = np.random.default_rng(10)
    x = 0.1 * rng.standard_normal(SR // 4)
    q = sp.compare(x, x)
    assert q['mel_db'] == 0.0 and q['env_corr'] == pytest.approx(1.0) and q['tone_db'] == 0.0
    with pytest.raises(ValueError):
        sp.render({'kind': 'modal', 'version': 1})


@pytest.mark.slow
def test_real_refs_resynthesis_near_floor():
    """on real forge refs the resynthesis sits near the render-vs-render floor (spectro-temporal statistics)"""
    from earshot.forge import refsets
    try:
        files = refsets.refset('whoosh_fast', 'forge')[:2] + refsets.refset('whoosh_big', 'forge')[:1]
    except refsets.RefsetsMissing:
        pytest.skip('no refsets registry on this machine')
    for f in files:
        r = sp.fit_report(f)
        assert r['fit']['mel_db'] < r['floor']['mel_db'] + 2.5, (f, r['fit'], r['floor'])
        assert r['fit']['env_corr'] > 0.9
