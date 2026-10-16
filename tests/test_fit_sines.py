"""earshot.forge.fit.sines: recover known parameters from synthetic tonal sounds, and the render / code contract."""
import json
import os

import numpy as np
import pytest

from earshot.forge.fit import sines

SR = 48000


def _bell(dur=1.0, on=0.05, parts=((440.0, -12.0, 1.0, 0.3), (1213.0, -18.0, 0.6, 1.1), (2390.5, -24.0, 0.3, 2.0)),
          floor_db=-70.0, seed=1):
    """struck inharmonic partials (f, dBFS at the strike, T60 s, phase) with a white noise floor"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    tt = np.clip(t - on, 0, None)
    x = np.zeros(n)
    for f, a, t60, ph in parts:
        x += (t >= on) * 10 ** (a / 20) * 10 ** (-3 * tt / t60) * np.cos(2 * np.pi * f * tt + ph)
    return x + 10 ** (floor_db / 20) * np.random.default_rng(seed).standard_normal(n)


def _near(model, f, tol_hz):
    """the strongest partial whose median frequency is within tol_hz of f"""
    best = None
    for p in model['partials']:
        fm = sines.partial_freq(p)
        if abs(fm - f) <= tol_hz:
            pk = max(v for _, v in p['amp'])
            if best is None or pk > best[0]:
                best = (pk, fm, p)
    assert best is not None, f'no partial near {f} Hz'
    return best


@pytest.fixture(scope='module')
def bell_model():
    x = _bell()
    return x, sines.fit(x, windows=(1024, 2048))


def test_recovers_partials_strike_and_decays(bell_model):
    x, m = bell_model
    assert m['kind'] == 'sines' and m['version'] == sines.VERSION
    assert abs(m['onset'] - 0.05) < 0.0015                        # the strike, to the sample region
    for f, a, t60, _ in ((440.0, -12.0, 1.0, 0), (1213.0, -18.0, 0.6, 0), (2390.5, -24.0, 0.3, 0)):
        pk, fm, p = _near(m, f, 5.0)
        assert abs(fm - f) < 0.5                                   # frequency within half a hertz
        assert abs(pk - a) < 1.0                                   # strike level within 1 dB (no pre-echo smear)
        assert abs(p['t0'] - 0.05) < 0.0015                        # each partial starts at the strike
        assert abs(sines._t60(p) - t60) / t60 < 0.1                # decay within 10 %
    s = m['summary']
    assert abs(s['f_low'] - 440) < 1 and s['inharmonicity'] > 0.04  # no harmonic series forced on it
    strong = [p for p in m['partials'] if max(v for _, v in p['amp']) > -40]
    assert len(strong) == 3                                        # no spurious strong partials


def test_resynthesis_is_close(bell_model):
    x, m = bell_model
    y = sines.render(m)
    assert len(y) == len(x)
    assert sines._lsd(x, y) < 3.0
    e = lambda z: 10 * np.log10((z[int(0.06 * SR):int(0.5 * SR)] ** 2).sum())
    assert abs(e(x) - e(y)) < 0.5                                  # level of the body


def test_chirp_trajectory_tracks():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    on = 0.02
    tt = np.clip(t - on, 0, None)
    f0, f1, T = 600.0, 1500.0, 0.15
    k = np.log(f1 / f0) / T
    ph = 2 * np.pi * f0 * np.where(tt < T, (np.exp(k * tt) - 1) / k, (np.exp(k * T) - 1) / k + np.exp(k * T) * (tt - T))
    x = (t >= on) * 0.3 * np.exp(-tt / 0.08) * np.sin(ph) + 1e-4 * np.random.default_rng(0).standard_normal(n)
    m = sines.fit(x, windows=(512, 1024))
    p = m['partials'][0]
    fq = np.asarray(p['freq'])
    for tq in (0.06, 0.10, 0.14):
        true = f0 * np.exp(k * (tq - on))
        got = 2 ** np.interp(tq, fq[:, 0], np.log2(fq[:, 1]))
        assert abs(1200 * np.log2(got / true)) < 40                 # within 40 cents mid-glide
    assert m['summary']['glide_cents'][0] > 1200                    # reads as an upward glide


def test_transient_click_is_modelled():
    """a struck partial plus a short click: the click must come back as a click (atoms), not as smeared noise"""
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    on = 0.03
    tt = np.clip(t - on, 0, None)
    click = (t >= on) * 0.5 * np.exp(-tt / 0.0008) * np.cos(2 * np.pi * 3100 * tt)
    x = (t >= on) * 0.2 * np.exp(-tt / 0.1) * np.cos(2 * np.pi * 700 * tt) + click
    x += 1e-4 * np.random.default_rng(2).standard_normal(n)
    m = sines.fit(x, windows=(512, 1024))
    assert m['transients'], 'no transient atoms at the strike'
    y = sines.render(m)
    a, b = int((on - 0.001) * SR), int((on + 0.004) * SR)
    assert abs(np.abs(y[a:b]).max() - np.abs(x[a:b]).max()) < 0.15  # the click's peak survives
    pre = slice(0, int((on - 0.002) * SR))
    assert np.abs(y[pre]).max() < 0.01                               # nothing before the strike


def test_residual_noise_band_levels():
    """a decaying tone over a steady band of noise: the noise comes back in the right band at the right level"""
    from scipy import signal
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    r = np.random.default_rng(3)
    nz = signal.sosfiltfilt(signal.butter(4, [2000, 4000], 'band', fs=SR, output='sos'), r.standard_normal(n)) * 0.02
    x = 0.3 * np.exp(-t / 0.3) * np.cos(2 * np.pi * 500 * t) + nz
    m = sines.fit(x, windows=(2048,))
    y = sines.render(m, knobs={'residual': 1.0}) - sines.synth_partials(m['partials'], n) \
        - sines.synth_atoms(m['transients'], n)
    band = lambda z: signal.sosfiltfilt(signal.butter(4, [2300, 3600], 'band', fs=SR, output='sos'), z)
    e = lambda z: 10 * np.log10((z[int(0.1 * SR):int(0.7 * SR)] ** 2).mean())
    assert abs(e(band(y)) - e(band(nz))) < 2.0
    lo = lambda z: signal.sosfiltfilt(signal.butter(4, 300, 'low', fs=SR, output='sos'), z)
    assert e(lo(y)) < e(band(y)) - 20                                # no noise where there was none


def test_json_round_trip_and_determinism(bell_model, tmp_path):
    _, m = bell_model
    s = json.dumps(m)
    m2 = json.loads(s)
    assert np.array_equal(sines.render(m), sines.render(m2))
    path = sines.save(m, str(tmp_path / 'm.json'))
    assert np.array_equal(sines.render(sines.load(path), seed=4), sines.render(m, seed=4))
    a, b = sines.render(m, seed=1), sines.render(m, seed=2)
    assert not np.array_equal(a, b)                                   # the seed changes the noise
    assert np.array_equal(sines.render(m, seed=1, knobs={'vary': 0.4}), sines.render(m, seed=1, knobs={'vary': 0.4}))


def _peak_hz(y, lo, hi):
    F = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    s = (f >= lo) & (f <= hi)
    return f[s][np.argmax(F[s])]


def test_knobs(bell_model):
    x, m = bell_model
    up = sines.render(m, knobs={'transpose': 12})
    assert abs(_peak_hz(up, 700, 1000) - 880) < 3                    # an octave up, partials keep their ratios
    assert abs(_peak_hz(up, 2200, 2700) - 2426) < 6
    st = sines.render(m, knobs={'stretch': 2.0})
    assert abs(len(st) / len(x) - 2.0) < 0.05
    from earshot.forge.tone import t40_ms
    assert t40_ms(sines.render(m, knobs={'decay': 2.0})) > 1.6 * t40_ms(sines.render(m))
    hi_ = sines.render(m, knobs={'brightness': 6.0})
    lo_ = sines.render(m, knobs={'brightness': -6.0})
    assert _peak_hz(hi_, 2000, 3000) == pytest.approx(_peak_hz(lo_, 2000, 3000), abs=3)
    e = lambda z: (np.abs(np.fft.rfft(z))[int(2300 * len(z) / SR):int(2500 * len(z) / SR)] ** 2).sum()
    assert e(hi_) > 10 * e(lo_)
    few = sines.render(m, knobs={'partials': 1})
    assert e(few) < 1e-3 * e(sines.render(m))
    vib = sines.render(m, knobs={'vibrato': 50, 'drift': 10})
    assert np.isfinite(vib).all() and len(vib) == len(x)
    st2 = sines.render(m, knobs={'stereo': True})
    assert st2.shape == (len(x), 2)
    with pytest.raises(ValueError):
        sines.render(m, knobs={'nope': 1})


def test_to_code_reproduces_render(bell_model, tmp_path):
    _, m = bell_model
    src = sines.to_code(m, name='bell')
    assert 'def render(seed=0)' in src and 'import numpy as np' in src
    ns = {}
    exec(compile(src, 'bell.py', 'exec'), ns)
    for seed in (0, 3):
        assert np.allclose(ns['render'](seed), sines.render(m, seed=seed), atol=1e-9)
    p = tmp_path / 'bell.py'
    p.write_text(src, encoding='utf-8')
    assert os.path.getsize(p) > 1000


def test_describe_and_stereo_input():
    x = _bell(dur=0.6)
    st = np.column_stack([x, 0.9 * x])
    m = sines.fit(st, windows=(1024,))
    assert m['width'] > -40                                           # side energy measured
    words = sines.describe(m)
    assert 'partials' in words and 'Hz' in words


def test_noise_only_input_does_not_crash():
    x = 0.05 * np.random.default_rng(0).standard_normal(int(0.2 * SR)) * np.exp(-np.arange(int(0.2 * SR)) / 2000)
    m = sines.fit(x, windows=(512,))
    y = sines.render(m)
    assert np.isfinite(y).all() and len(y) == len(x)


@pytest.mark.slow
def test_real_tonal_refs_fit_closely():
    """manual / slow check on the real forge-split refs (needs the licensed reference registry)"""
    from earshot.forge.refsets import refset, RefsetsMissing
    from earshot import audio
    try:
        refs = refset('notify')[:2] + refset('pop')[:2]
    except (RefsetsMissing, KeyError):
        pytest.skip('no reference registry on this machine')
    for p in refs:
        x = audio.load(p).mean(1)
        m = sines.fit(p)
        y = sines.render(m)
        assert sines._lsd(x, y) < 6.0, os.path.basename(p)
