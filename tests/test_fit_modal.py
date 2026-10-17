"""Tests for the modal fitter (earshot/forge/fit/modal.py): synthetic struck sounds of KNOWN parameters (modes, onsets,
fades) must be recovered; the model must be JSON-plain, render deterministically, follow its knobs, and turn into a
standalone script that renders the same samples. The slow test fits real forge-split refs (needs the registry)."""
import json

import numpy as np
import pytest

from earshot.forge.fit import modal

SR = 48000


def struck(modes, t0=0.010, length=0.3, noise=1e-4, seed=0):
    """damped sinusoids [(f, tau, amp, phase)] starting at t0, plus white noise"""
    n = int(length * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    o = int(round(t0 * SR))
    for f, tau, a, ph in modes:
        tt = t[o:] - t[o]
        x[o:] += a * np.exp(-tt / tau) * np.cos(2 * np.pi * f * tt + ph)
    return x + noise * np.random.default_rng(seed).standard_normal(n)


TRUE = [(440.0, 0.08, 0.5, 0.3), (1234.5, 0.03, 0.3, -1.0), (3150.0, 0.012, 0.2, 2.0), (95.0, 0.15, 0.25, 0.0)]


@pytest.fixture(scope='module')
def fitted():
    x = struck(TRUE)
    return x, modal.fit(x)


def test_recovers_known_modes(fitted):
    x, md = fitted
    assert md['kind'] == 'modal' and md['version'] == modal.VERSION
    assert len(md['strikes']) == 1
    assert abs(md['strikes'][0]['t'] - 0.010) < 0.0003
    rows = md['strikes'][0]['modes']
    for f, tau, a, ph in TRUE:
        best = min(rows, key=lambda q: abs(q[0] - f))
        assert abs(best[0] - f) / f < 0.002, (f, best)
        assert abs(best[1] - tau) / tau < 0.05, (tau, best)
        assert abs(best[2] - a) / a < 0.05, (a, best)
    # the four true modes carry nearly all the energy: nothing big was invented
    E = sorted((q[2] ** 2 * q[1] for q in rows), reverse=True)
    assert sum(E[4:]) < 0.01 * sum(E[:4])


def test_resynthesis_is_close(fitted):
    x, md = fitted
    y = modal.render(md, seed=0)
    assert y.shape == x.shape
    mel, per = modal.mel_distance(x, y)
    assert mel < 3.0, per
    assert modal.env_corr(x, y) > 0.98
    assert np.sqrt(np.mean((x - y) ** 2)) < 0.05 * np.sqrt(np.mean(x ** 2))   # waveform-level match for pure modes


def test_model_is_plain_json_and_deterministic(fitted, tmp_path):
    x, md = fitted
    back = json.loads(json.dumps(md))
    assert back == md
    a, b = modal.render(back, seed=3), modal.render(md, seed=3)
    assert np.array_equal(a, b)
    c = modal.render(md, seed=4)
    assert not np.array_equal(a, c)
    assert np.sqrt(np.mean((a - c) ** 2)) < 0.05 * np.sqrt(np.mean(a ** 2))  # seeds change the noise only
    p = modal.save(md, str(tmp_path / 'm.json'))
    assert modal.load(p) == md
    with pytest.raises(ValueError):
        modal.render({'kind': 'other', 'version': 1})


def test_two_strikes_and_a_damping_contact():
    a = struck([(700.0, 0.2, 0.4, 0.0), (1900.0, 0.05, 0.2, 1.0)], t0=0.01, length=0.35, noise=0)
    n = len(a)
    o2 = int(0.12 * SR)
    a[o2:] *= np.exp(-np.arange(n - o2) / (0.004 * SR))         # a finger stops the ring at 120 ms
    b = struck([(2600.0, 0.03, 0.3, 0.5)], t0=0.12, length=0.35, noise=0)
    x = a + b + 1e-4 * np.random.default_rng(1).standard_normal(n)
    md = modal.fit(x)
    ts = [s['t'] for s in md['strikes']]
    assert len(ts) == 2 and abs(ts[0] - 0.01) < 0.0003 and abs(ts[1] - 0.12) < 0.0005, ts
    assert any(abs(q[0] - 2600) < 5 for q in md['strikes'][1]['modes'])
    y = modal.render(md, 0)
    late = slice(int(0.16 * SR), int(0.3 * SR))                  # the stopped ring must stay stopped
    assert np.sqrt(np.mean(y[late] ** 2)) < 2.0 * np.sqrt(np.mean(x[late] ** 2)) + 1e-4
    assert modal.mel_distance(x, y)[0] < 4.0


def test_onsets_and_fade():
    n = int(0.4 * SR)
    x = 1e-4 * np.random.default_rng(2).standard_normal(n)
    for t0 in (0.02, 0.15, 0.27):
        o = int(t0 * SR)
        k = np.arange(n - o)
        x[o:] += 0.5 * np.exp(-k / (0.01 * SR)) * np.random.default_rng(int(t0 * 1000)).standard_normal(n - o)
    ons = modal.onsets(x)
    assert len(ons) == 3
    assert all(abs(o / SR - t) < 0.0015 for o, t in zip(ons, (0.02, 0.15, 0.27))), ons
    y = struck([(300.0, 0.4, 0.5, 0.0)], t0=0.01, length=0.5, noise=1e-5)
    nf = int(0.03 * SR)
    y[-nf:] *= np.linspace(1, 0, nf)
    assert modal.fade_out(y) in (0.02, 0.03, 0.05)
    assert modal.fade_out(struck([(300.0, 0.4, 0.5, 0.0)], length=0.5)) == 0.0


def test_knobs(fitted):
    x, md = fitted

    def peak_hz(y):
        F = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 16))
        return np.fft.rfftfreq(1 << 16, 1 / SR)[np.argmax(F)]

    def centroid(y):
        F = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return float((F * f).sum() / F.sum())

    y0 = modal.render(md, 0)
    assert abs(peak_hz(modal.render(md, 0, size=2.0)) - peak_hz(y0) / 2) < 2.0
    assert len(modal.render(md, 0, size=2.0)) > len(y0)                    # bigger rings longer: the tail grows
    late = slice(int(0.1 * SR), int(0.2 * SR))
    e0 = np.mean(y0[late] ** 2)
    assert np.mean(modal.render(md, 0, damping=2.0)[late] ** 2) < 0.2 * e0
    assert centroid(modal.render(md, 0, brightness=6.0)) > 1.3 * centroid(y0)
    assert centroid(modal.render(md, 0, brightness=-6.0)) < centroid(y0)
    yv = modal.render(md, 5, vary=0.6, ring=False)
    assert len(yv) == len(y0) and not np.allclose(yv, y0)
    assert len(modal.render(md, 5, vary=0.6)) >= len(y0)                   # a vary take rings out (ring=None)
    assert np.allclose(modal.render(md, 0, gain=-6.0), y0 * 10 ** (-6 / 20))
    s = modal.render(md, 0, stereo=True, room=0.3)
    assert s.shape == (len(y0), 2) and np.isfinite(s).all()


def test_stereo_side_follows_the_original():
    m = struck(TRUE, noise=3e-3)
    side = 0.3 * struck([(440.0, 0.08, 0.5, 1.3), (1234.5, 0.03, 0.3, 0.2)], noise=3e-3, seed=5)
    x = np.stack([m + side, m - side], axis=1)
    md = modal.fit(x)
    assert md['channels'] == 2 and md['side'] and len(md['side']) == len(md['noise']['env'])
    y = modal.render(md, 0)
    assert y.shape == x.shape
    cx = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    cy = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
    assert abs(cx - cy) < 0.15, (cx, cy)


def test_to_code_renders_the_same_samples(fitted, tmp_path):
    x, md = fitted
    src = modal.to_code(md, name='synthetic_strike')
    assert 'def render(' in src and 'MODEL = {' in src and 'earshot' not in src.split('"""', 2)[2]
    ns = {'__name__': 'emitted'}
    exec(compile(src, 'emitted.py', 'exec'), ns)
    for knobs in ({}, {'size': 1.3, 'damping': 0.8, 'brightness': -2.0, 'vary': 0.4}):
        assert np.array_equal(ns['render'](7, **knobs), modal.render(md, 7, **knobs))
    # a stereo, faded take with a sub-bass bump: every model field must reach the script
    y = struck([(300.0, 0.4, 0.5, 0.0), (2100.0, 0.02, 0.2, 1.0)], t0=0.01, length=0.4, noise=1e-3)
    y[int(0.01 * SR):int(0.08 * SR)] += 0.3 * np.hanning(int(0.07 * SR))      # a slow pressure pulse
    nf = int(0.03 * SR)
    y[-nf:] *= np.linspace(1, 0, nf)
    st = np.stack([y, 0.8 * y + 0.01 * np.random.default_rng(3).standard_normal(len(y))], axis=1)
    m2 = modal.fit(st)
    assert m2['fade_out_s'] > 0 and m2['sub'] is not None and m2['channels'] == 2
    ns2 = {'__name__': 'emitted'}
    exec(compile(modal.to_code(m2), 'emitted2.py', 'exec'), ns2)
    assert np.array_equal(ns2['render'](3, size=1.2, room=0.2), modal.render(m2, 3, size=1.2, room=0.2))


def test_band_masks_are_power_complementary():
    W = modal.band_masks(1 << 14)
    assert W.shape[0] == len(modal.NOISE_EDGES) + 1
    assert np.allclose((W ** 2).sum(0), 1.0, atol=1e-9)


def test_esprit_recovers_poles():
    n = np.arange(200)
    z_true = np.array([0.98 * np.exp(0.3j), 0.95 * np.exp(-1.1j), 0.99 * np.exp(2.0j)])
    s = sum(np.exp(n * np.log(z)) * c for z, c in zip(z_true, (1.0, 0.5, 0.25)))
    z, k = modal.esprit(s, kmax=8)
    assert k == 3
    for zt in z_true:
        assert np.min(np.abs(z - zt)) < 1e-6


def test_mel_distance_basics():
    x = struck(TRUE)
    assert modal.mel_distance(x, x)[0] == 0.0
    assert 4.5 < modal.mel_distance(x, 0.5 * x)[0] <= 6.03          # -6 dB, less where cells sit on the floor


@pytest.mark.slow
def test_real_refs_resynthesise_closely():
    from earshot import audio
    from earshot.forge import refsets
    try:
        refs = refsets.refset('click', 'forge')[:2] + refsets.refset('typing_key', 'forge')[:1]
    except refsets.RefsetsMissing:
        pytest.skip('no reference-set registry on this machine')
    for p in refs:
        x = audio.load(p)
        md = modal.fit(x)
        y = modal.render(md, 0)
        floor = modal.mel_distance(y, modal.render(md, 1))[0]          # two noise draws of the same model
        mel = modal.mel_distance(x, y)[0]
        assert mel < floor + 1.5, (p, mel, floor)
        assert modal.env_corr(x, y) > 0.8, p
