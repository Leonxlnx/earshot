"""Stereo in the fit engine: the sines side (0.4), quality's width terms and the diffuse tail layer (fit/space.py)."""
import numpy as np
import pytest
from scipy import signal

from earshot.forge.fit import quality as Q
from earshot.forge.fit import sines, space

SR = 48000


def _hilbert(v):
    return np.imag(signal.hilbert(v))


def _partials(n, on=0.03, parts=((440.0, -12.0, 1.0, 0.3), (1213.0, -18.0, 0.6, 1.1), (2390.5, -24.0, 0.3, 2.0))):
    t = np.arange(n) / SR
    tt = np.clip(t - on, 0, None)
    return [(t >= on) * 10 ** (a / 20) * 10 ** (-3 * tt / t60) * np.cos(2 * np.pi * f * tt + ph)
            for f, a, t60, ph in parts]


@pytest.fixture(scope='module')
def wide_bell():
    """440 Hz panned left (pan 0.5), 1213 Hz decorrelated (the mid's quadrature at 0.8), 2390 Hz centred"""
    n = int(0.8 * SR)
    p = _partials(n)
    mid = p[0] + p[1] + p[2] + 1e-4 * np.random.default_rng(0).standard_normal(n)
    side = 0.5 * p[0] + 0.8 * _hilbert(p[1])
    X = np.stack([mid + side, mid - side], 1)
    return X, sines.fit(X, windows=(1024,))


def _reverb_target(n=int(0.9 * SR), rt=0.5, wet_db=-6.0, tau=0.01, seed=3):
    """a dry struck tone + click (centred) and two independent decaying noise reverbs, one per channel (tail L/R
    correlation 0.27 at the defaults)"""
    t = np.arange(n) / SR
    on = 0.02
    tt = np.clip(t - on, 0, None)
    dry = (t >= on) * (0.3 * np.exp(-tt / tau) * np.cos(2 * np.pi * 800 * tt) + 0.2 * np.exp(-tt / 0.002))
    r = np.random.default_rng(seed)
    k = np.arange(int(rt * SR)) / SR
    env = np.exp(-6.9 * k / rt) * (1 - np.exp(-k / 0.004))
    wet = [signal.fftconvolve(dry, r.standard_normal(len(k)) * env)[:n] for _ in range(2)]
    g = 10 ** (wet_db / 20) / np.sqrt(np.mean(wet[0] ** 2) / np.mean(dry ** 2))
    return dry, np.stack([dry + g * wet[0], dry + g * wet[1]], 1)


# ------------------------------------------------------------------------------------------------ measures
def test_side_powers_split_pan_from_diffuse():
    m = signal.lfilter([1.0], [1, -0.5], np.random.default_rng(1).standard_normal(SR // 2))
    P = space.side_powers(np.stack([m * 1.4, m * 0.6], 1))        # a pan: S = 0.4 M
    live = P['PM'].sum(1) > 1e-6 * P['PM'].sum()
    assert np.allclose((P['C'].sum(1) / P['PM'].sum(1))[live], 0.4, atol=0.02)
    assert (space.diffuse_power(P).sum(1) / P['PM'].sum(1))[live].max() < 1e-3
    s = 0.7 * _hilbert(m)                                          # a diffuse side: the mid's quadrature
    P = space.side_powers(np.stack([m + s, m - s], 1))
    assert np.allclose(np.sqrt(space.diffuse_power(P).sum(1) / P['PM'].sum(1))[2:-1], 0.7, atol=0.05)
    assert np.abs(P['C'].sum(1) / P['PM'].sum(1))[2:-1].max() < 0.05


def test_tail_corr_and_span():
    dry, X = _reverb_target()
    a, b = space.tail_span(X)
    _, ip, end = space.active_span(X)
    assert a == ip + int(0.03 * SR) and b == end and b > a
    assert space.tail_corr(np.stack([dry, dry], 1)) == pytest.approx(1.0)
    assert space.tail_corr(X) < 0.5                                # the independent reverbs decorrelate the tail
    click = np.zeros(SR // 10)
    click[100:110] = 1.0
    assert space.tail_span(click) is None and space.tail_corr(click) is None
    assert not space.is_stereo(np.stack([dry, dry], 1)) and space.is_stereo(X)


# ------------------------------------------------------------------------------------------------ sines side
def test_sines_side_matches_source_width(wide_bell):
    X, m = wide_bell
    sd = m['side']
    assert sd['bands'] == list(sines.SIDE_BANDS) and len(sd['pan']) == len(sd['bands'])
    j = int(np.searchsorted(sd['times'], 0.2))
    assert sd['pan'][2][j] == pytest.approx(0.5, abs=0.05)          # 500 Hz band: the panned 440 Hz partial
    assert sd['diffuse'][3][j] == pytest.approx(0.8, abs=0.08)      # 1 kHz band: the decorrelated 1213 Hz partial
    y = sines.render(m, stereo=True)
    q = Q.quality(X, y)
    assert abs(q['side_db'][0] - q['side_db'][1]) < 1.5 and q['width_db'] < 2.5
    assert abs(q['tail_corr'][0] - q['tail_corr'][1]) < 0.1
    assert np.allclose(y.mean(1), sines.render(m), atol=1e-12)      # the side never changes the mono sum
    ym = Q.quality(X, sines.render(m))                              # a mono render of the same model
    assert ym['width_db'] > 15 and Q.score(ym) > Q.score(q) + 0.5


def test_sines_side_follows_the_tail():
    """a centred strike whose tail decorrelates: the keyframes carry the change"""
    n = int(0.8 * SR)
    p = _partials(n)
    mid = p[0] + p[1] + 1e-4 * np.random.default_rng(2).standard_normal(n)
    grow = np.clip((np.arange(n) / SR - 0.06) / 0.06, 0, 1)
    side = 0.95 * grow * _hilbert(mid)
    X = np.stack([mid + side, mid - side], 1)
    y = sines.render(sines.fit(X, windows=(1024,)), stereo=True)
    head = slice(int(0.03 * SR), int(0.055 * SR))
    assert space.corr(y[head, 0], y[head, 1]) > 0.95
    assert space.tail_corr(X) < 0.45
    assert abs(space.tail_corr(y) - space.tail_corr(X)) < 0.1


def test_legacy_model_keeps_its_side(wide_bell):
    _, m = wide_bell
    old = dict(m)
    old.pop('side')
    y = sines.render(old, seed=2, stereo=True)
    mid = sines.render(old, seed=2)
    res = old['residual']
    s = sines.synth_noise(np.asarray(res['bands']), np.asarray(res['times']), np.asarray(res['db']), len(mid),
                          2 + 1000003, SR, hf=res.get('hf')) * 10 ** (old['width'] / 20)
    s = sines.finish(s, SR, limit=False)                            # the fade-in applies to both channels
    assert np.allclose(y[:, 0] - y[:, 1], 2 * s, atol=1e-9)


def test_morph_side(wide_bell):
    _, m = wide_bell
    sd = m['side']
    assert sines.morph_side(None, None, 0.5) is None
    assert sines.morph_side(sd, sd, 0.3) == sd                       # the same side either way
    half = sines.morph_side(sd, None, 0.5)                          # against a centred (mono) model: half the gains
    j = int(np.searchsorted(half['times'], 0.2))
    assert half['diffuse'][3][j] == pytest.approx(0.5 * np.interp(half['times'][j], sd['times'], sd['diffuse'][3]),
                                                  abs=2e-3)
    late = sines.morph_side(sd, sd, 0.5, map_a=lambda t: t + 0.1, map_b=lambda t: t + 0.1)
    assert late['times'][0] == pytest.approx(sd['times'][0] + 0.1)
    mm = dict(m, side=half)
    y = sines.render(mm, stereo=True)
    assert y.shape[1] == 2 and Q.stereo(sines.render(m, stereo=True), y)['side_db'][1] < Q.stereo(
        sines.render(m, stereo=True), y)['side_db'][0]


def test_side_knobs_and_determinism(wide_bell):
    X, m = wide_bell
    a, b = sines.render(m, seed=1, stereo=True), sines.render(m, seed=1, stereo=True)
    assert np.array_equal(a, b)
    narrow = sines.render(m, seed=1, stereo=True, width=0.0)
    assert np.allclose(narrow[:, 0], narrow[:, 1])
    st = sines.render(m, seed=1, stereo=True, stretch=1.5, transpose=3)
    assert st.shape[1] == 2 and np.isfinite(st).all()
    assert Q.stereo(st, st)['width_db'] == 0.0


def test_renders_start_at_zero_and_peak_limited(wide_bell):
    _, m = wide_bell
    y = sines.render(m, gain_db=24.0)                              # far over full scale
    assert y[0] == 0.0 and abs(y[1]) < 1e-3 * np.abs(y).max()
    assert np.abs(y).max() == pytest.approx(10 ** (sines.PEAK_DB / 20), rel=1e-9)
    ys = sines.render(m, gain_db=24.0, stereo=True)
    assert np.abs(ys).max() == pytest.approx(10 ** (sines.PEAK_DB / 20), rel=1e-9)
    layer = dict(m, layer='deterministic')                          # a hybrid's layer: the hybrid limits the sum
    assert np.abs(sines.render(layer, gain_db=24.0)).max() > 1.0
    code = {}
    exec(compile(sines.to_code(dict(m, partials=m['partials'][:3])), 'b.py', 'exec'), code)
    assert np.allclose(code['render'](0), sines.render(dict(m, partials=m['partials'][:3])), atol=1e-9)


# ------------------------------------------------------------------------------------------------ quality
def test_quality_width_terms():
    dry, X = _reverb_target()
    q = Q.quality(X, X)
    assert q['width_db'] == 0.0 and q['tail_corr_err'] == 0.0
    qm = Q.quality(X, dry)
    assert qm['side_db'][1] == -40.0 and qm['width_db'] > 10 and qm['tail_corr'][1] == pytest.approx(1.0)
    assert qm['tail_corr_err'] > 0.5 and Q.score(qm) > Q.score(dict(qm, width_db=0.0, tail_corr_err=0.0)) + 0.9
    q1 = Q.quality(dry, dry * 0.5)                                 # mono vs mono: no stereo terms
    assert q1['width_db'] == 0.0 and q1['tail_corr_err'] == 0.0
    assert 'tail corr' in Q.text(qm)


# ------------------------------------------------------------------------------------------------ space layer
def test_fit_space_decorrelates_a_mono_tail():
    """a mono re-make that rings the whole tail (the recording's own mid): a side part only, the mono sum unchanged"""
    _, X = _reverb_target()
    mono = X.mean(1)
    sp = space.fit_space(X, SR, dry=mono)
    assert sp is not None and sp['kind'] == 'space' and len(sp['rt60']) == len(space.BANDS)
    assert max(sp['mid_db']) == -150.0 and sp['fit']['mid_need_db'] is None
    live = [r for r, g in zip(sp['rt60'], sp['side_db']) if g > -140]
    assert live and 0.2 < float(np.median(live)) < 1.0               # the reverbs' rt60 is 0.5 s
    a, b = space.tail_span(X)
    assert max(sp['rt60']) <= space.RT_SPAN * (b - a) / SR + 1e-9   # never rings far past the recording's tail
    y = space.render_space(sp, mono, seed=0)
    assert np.allclose(y.mean(1), mono, atol=1e-12)                 # side only: the mono sum is the dry
    tc = space.tail_corr(X)
    assert abs(space.tail_corr(y) - tc) < 0.2 < abs(1.0 - tc)
    q0, q1 = Q.quality(X, mono), Q.quality(X, y)
    assert q1['width_db'] < q0['width_db'] - 5 and q1['tail_corr_err'] < q0['tail_corr_err'] - 0.4
    assert q1['mel_db'] == pytest.approx(q0['mel_db'], abs=1e-6)
    assert np.array_equal(y, space.render_space(sp, mono, seed=0))  # deterministic
    assert not np.array_equal(y, space.render_space(sp, mono, seed=1))
    long = space.render_space(sp, mono, seed=0, ring=True)
    assert len(long) > len(mono) and np.allclose(long[:len(mono)], y, atol=1e-5)   # the same up to -100 dB
    assert 'side only' in space.describe(sp)
    alone = space.fit_space(X, SR)                                  # no dry: the recording's mid stands in
    assert {k: v for k, v in alone.items() if k != 'fit'} == {k: v for k, v in sp.items() if k != 'fit'}


def test_fit_space_fills_a_short_dry_tail():
    """a dry that stops where the recording's reverb goes on: a mid part as well, and the mono gets closer"""
    dry, X = _reverb_target()
    sp = space.fit_space(X, SR, dry=dry)
    assert sp is not None and max(sp['mid_db']) > -140 and 'side + mid' in space.describe(sp)
    y = space.render_space(sp, dry, seed=0)
    tc = space.tail_corr(X)
    assert abs(space.tail_corr(y) - tc) < 0.25
    q0, q1 = Q.quality(X, dry), Q.quality(X, y)
    assert q1['mel_db'] < q0['mel_db'] and q1['tail_corr_err'] < q0['tail_corr_err'] - 0.4


def test_fit_space_declines():
    dry, X = _reverb_target()
    assert space.fit_space(np.stack([dry, dry], 1), SR) is None     # mono
    assert space.fit_space(dry, SR) is None
    _, Xc = _reverb_target(wet_db=-30.0, tau=0.02)                  # a barely reverberant, correlated tail
    assert space.tail_corr(Xc) > 0.7 and space.fit_space(Xc, SR) is None
    assert space.fit_space(X, SR, dry=X) is None                    # the dry already has the tail's width
    assert space.needs_space(X) and not space.needs_space(Xc)


def test_hybrid_counts_the_tone_layers_side_once():
    """review: a sines layer keeps its fitted side in a 0.4 hybrid, but fit_hybrid took only its mono render out of
    the residual, so the air layer modelled the tones' side again as noise and a panned tone came out about 3 dB too
    wide; the residual now leaves the tone layer's side out (its stereo render)"""
    from earshot.forge.fit import auto
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    r = np.random.default_rng(3)
    env = np.minimum(t / 0.002, 1) * np.exp(-t / 0.12)
    tone = 0.4 * (np.sin(2 * np.pi * 880 * t) + 0.5 * np.sin(2 * np.pi * 1760 * t + 0.4)) * env
    nz = 0.03 * np.exp(-t / 0.05)
    X = np.stack([tone * 1.5 + nz * r.standard_normal(n), tone * 0.5 + nz * r.standard_normal(n)], 1)
    X *= 0.8 / np.abs(X).max()
    m = auto.fit_hybrid(X, ('sines', 'spectral'), fast=True)
    assert m['layers'][0]['model'].get('side')
    for s in range(3):
        a, b = Q.stereo(X, auto.render(m, seed=s))['side_db']
        assert abs(a - b) < 1.0, (s, a, b)                          # (2.4-3.1 dB too wide before)
