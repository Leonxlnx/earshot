"""The fit engine (earshot/forge/fit: auto, quality, codegen, CLI): synthetic signals of known structure only."""
import json

import numpy as np
import pytest

from earshot.forge import dsp
from earshot.forge.fit import auto, codegen
from earshot.forge.fit import quality as Q

SR = 48000


def struck(f=(1200.0, 2900.0), tau=(0.03, 0.012), amp=(0.6, 0.3), n_s=0.15, t0=0.01, seed=0, noise=0.02):
    """a click: damped modes from t0 plus a short decaying noise burst and a faint floor"""
    r = np.random.default_rng(seed)
    n = int(n_s * SR)
    t = np.arange(n) / SR - t0
    on = t >= 0
    y = np.zeros(n)
    for fi, ti, ai in zip(f, tau, amp):
        y += on * ai * np.exp(-np.clip(t, 0, None) / ti) * np.sin(2 * np.pi * fi * np.clip(t, 0, None))
    y += on * noise * 10 * r.standard_normal(n) * np.exp(-np.clip(t, 0, None) / 0.004)
    y += 1e-4 * r.standard_normal(n)
    return y


def swell(n_s=0.5, seed=1):
    """a noise gesture: band-limited noise swelling over 300 ms, then fading"""
    r = np.random.default_rng(seed)
    n = int(n_s * SR)
    t = np.arange(n) / SR
    env = np.minimum(t / 0.3, 1.0) ** 2 * np.exp(-np.clip(t - 0.3, 0, None) / 0.05)
    return 0.3 * dsp.butter(r.standard_normal(n), 'band', (800.0, 5000.0)) * env


def tone(f=(660.0, 1320.0), n_s=1.2, decay=0.4):
    t = np.arange(int(n_s * SR)) / SR
    y = sum(a * np.sin(2 * np.pi * fi * t) for fi, a in zip(f, (0.5, 0.2)))
    return y * np.exp(-t / decay) * np.minimum(t / 0.003, 1.0)


def chirp(n_s=0.2, f0=400.0, f1=1400.0):
    t = np.arange(int(n_s * SR)) / SR
    f = f0 * (f1 / f0) ** (t / n_s)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return 0.5 * np.sin(ph) * np.minimum(t / 0.002, 1.0) * np.exp(-t / 0.08)


# --------------------------------------------------------------------------------------------------- quality
def test_quality_identity_gain_and_shift():
    x = struck()
    q = Q.quality(x, x)
    assert q['mel_db'] == 0 and q['tone_db'] == 0 and q['onset_err'] == 0 and q['env_corr'] > 0.999
    q2 = Q.quality(x, 0.5 * x)                        # each sound is levelled: only level_db sees a gain
    assert abs(q2['level_db'] + 6.02) < 0.05 and q2['mel_db'] < 1e-6
    q3 = Q.quality(x, np.roll(x, int(0.03 * SR)))     # a timing error counts (no re-alignment on the resynthesis)
    assert q3['mel_db'] > 5 and q3['env_corr'] < q['env_corr']
    assert Q.score(q) < Q.score(q3)


def test_quality_onset_sharpness_and_tone():
    x = struck()
    t = np.arange(len(x)) / SR - 0.01
    soft = Q.quality(x, x * np.clip(t / 0.004, 0, 1))['onset_err']             # a 4 ms fade-in over the strike
    softer = Q.quality(x, x * np.clip(t / 0.012, 0, 1))['onset_err']           # a 12 ms one
    assert Q.quality(x, x.copy())['onset_err'] < 0.05 and 0.4 < soft < softer and softer > 1.0
    dark = dsp.butter(x, 'low', 1500.0)
    assert Q.quality(x, dark)['tone_db'] > 3


# --------------------------------------------------------------------------------------------------- character
@pytest.mark.parametrize('sig,cls,first', [
    (struck(), 'struck', 'modal'),
    (swell(), 'gesture', 'spectral'),
    (tone(), 'ringing', 'sines'),
    (chirp(), 'glide', 'sines'),
])
def test_character_classes(sig, cls, first):
    c = auto.character(sig)
    assert c['class'] == cls, c
    p = auto.plan(c)
    assert p[0] == first
    assert len(p) == len(set(p))
    if c['dur_s'] < 1.0:
        assert set(p) == set(auto.ALL)               # short sounds: every candidate is cheap enough to try


# --------------------------------------------------------------------------------------------------- fitting
def test_auto_picks_by_quality_and_post_stage():
    x = struck()
    got = {}
    m = auto.fit(x, candidates=['modal', 'spectral'], fast=True, collect=got)
    fe = m['fit_engine']
    assert fe['kind'] == 'auto' and set(got) == {'modal', 'spectral'}
    sc = {r['candidate']: r['score'] for r in fe['candidates']}
    assert fe['chosen'] == min(sc, key=sc.get) == 'modal'
    assert set(m['post']) >= {'eq_db', 'gain_db', 'pre'} and len(m['post']['eq_db']) == 28
    y = auto.render(m, seed=0)
    assert len(y) == len(x)
    q = Q.quality(x, y)
    assert q['mel_db'] < 6 and q['env_corr'] > 0.85 and abs(q['level_db']) < 1.5
    assert np.array_equal(auto.render(m, seed=3), auto.render(m, seed=3))       # deterministic per seed


def test_post_stage_cleans_the_pre_onset_and_matches_balance():
    x = struck(t0=0.02)
    r = np.random.default_rng(5)
    leak = x + 0.05 * r.standard_normal(len(x)) * (np.arange(len(x)) < int(0.02 * SR))   # noise before the strike
    leak = dsp.butter(leak, 'low', 6000.0)                                      # and a darker balance
    post = auto.fit_post(x, [leak])
    y = auto.apply_post(leak, post)
    n_on = int(0.019 * SR)
    p = lambda v: 10 * np.log10(np.mean(v[:n_on] ** 2) + 1e-20)
    assert p(y) < p(leak) - 15                                                  # pre-onset leak removed
    assert Q.quality(x, y)['tone_db'] < Q.quality(x, leak)['tone_db']


def test_hybrid_layers_and_deterministic_part():
    x = struck(noise=0.05)
    m = auto.fit_hybrid(x, ('modal', 'spectral'), fast=True)
    assert m['kind'] == 'hybrid' and [L['role'] for L in m['layers']] == ['body', 'air']
    det = m['layers'][0]['model']
    assert det['layer'] == 'deterministic' and 'post' not in det
    assert all(max(v for _, v in env) <= -200 for env in det['noise']['env'])   # no noise in the deterministic layer
    y = auto.render(m, seed=1)
    assert y.shape[0] == len(x) and Q.quality(x, y)['mel_db'] < 7
    y2 = auto.render(m, seed=1, body_db=-60.0)                                  # the air layer alone is quieter
    assert np.sqrt(np.mean(y2 ** 2)) < np.sqrt(np.mean(y ** 2))
    with pytest.raises(ValueError):
        auto.fit_hybrid(x, ('spectral', 'sines'))


def test_render_rejects_unknown_knobs():
    m = auto.fit(struck(), kind='modal', fast=True)
    with pytest.raises(ValueError):
        auto.render(m, seed=0, transpose=2)
    assert len(auto.render(m, seed=0, size=1.2, vary=0.3)) > 0


# --------------------------------------------------------------------------------------------------- codegen
def test_codegen_recipe_round_trip(tmp_path):
    x = struck()
    m = auto.fit(x, kind='modal', fast=True)
    src = codegen.to_recipe(m, name='synth_click', target='a short test click')
    assert 'fit_engine' not in src.split('MODEL = ', 1)[1]
    ns = {}
    exec(codegen.literal(m), ns)
    assert ns['MODEL'] == {k: v for k, v in m.items() if k != 'fit_engine'}       # the literal is exact
    p = codegen.write_recipe(str(tmp_path / 'synth_click.py'), m, target='a short test click')
    rec = codegen.load_recipe(p)
    assert rec.TARGET == 'a short test click' and rec.KIND == 'modal'
    assert set(rec.SPACE) == set(rec.DEFAULTS) == set(auto.KNOBS['modal'])
    for spec in rec.SPACE.values():                    # valid forge SPACE specs
        assert len(spec) in (2, 3) and spec[0] < spec[1]
    assert np.array_equal(rec.render_seed(0), auto.render(m, seed=0))          # the code renders the fit, bitwise
    seed = int(dsp.rng(7).integers(0, 2 ** 31 - 1))
    assert np.array_equal(rec.render(rec.DEFAULTS, dsp.rng(7)), auto.render(m, seed=seed))
    var = rec.render(dict(rec.DEFAULTS, size=1.2, vary=0.4), dsp.rng(7))
    assert len(var) > 0 and not np.allclose(var[:len(x)], rec.render_seed(0)[:len(var)])


def test_codegen_sines_and_spectral_recipes(tmp_path):
    for sig, kind in ((tone(n_s=0.5, decay=0.15), 'sines'), (swell(), 'spectral')):
        m = auto.fit(sig, kind=kind, fast=True)
        rec = codegen.load_recipe(codegen.write_recipe(str(tmp_path / f'r_{kind}.py'), m))
        assert np.array_equal(rec.render_seed(2), auto.render(m, seed=2))
        assert set(rec.SPACE) == set(auto.KNOBS[kind])


# --------------------------------------------------------------------------------------------------- morph
def _strongest_mode(m):
    md = np.array(m['strikes'][0]['modes'])
    return md[np.argmax(md[:, 2] ** 2 * md[:, 1]), 0]


def test_morph_modal_interpolates_frequency_and_keeps_endpoints():
    a = auto.fit(struck(f=(1000.0,), tau=(0.04,), amp=(0.6,), noise=0.002), kind='modal', fast=True)
    b = auto.fit(struck(f=(1600.0,), tau=(0.04,), amp=(0.6,), noise=0.002, seed=3), kind='modal', fast=True)
    mid = auto.morph(a, b, 0.5)
    assert abs(_strongest_mode(mid) / np.sqrt(1000.0 * 1600.0) - 1) < 0.03
    assert np.allclose(auto.render(auto.morph(a, b, 0.0), 1), auto.render(a, 1), atol=1e-9)
    assert np.allclose(auto.render(auto.morph(a, b, 1.0), 1), auto.render(b, 1), atol=1e-9)
    mm = auto.morph_model(a, b, 0.3)
    assert 'morph' in auto.knobs_of(mm)
    assert np.array_equal(auto.render(mm, 2), auto.render(auto.morph(a, b, 0.3), 2))
    assert np.array_equal(auto.render(mm, 2, morph=0.5), auto.render(mid, 2))
    s = auto.fit(swell(), kind='spectral', fast=True)
    with pytest.raises(ValueError):
        auto.morph(a, s, 0.5)


def test_morph_sines_and_spectral():
    from earshot.forge.fit.sines import partial_freq
    a = auto.fit(tone(f=(440.0, 880.0), n_s=0.4, decay=0.15), kind='sines', fast=True)
    b = auto.fit(tone(f=(660.0, 1320.0), n_s=0.4, decay=0.15), kind='sines', fast=True)
    mid = auto.morph(a, b, 0.5)
    f0 = partial_freq(mid['partials'][0])
    assert abs(f0 / np.sqrt(440.0 * 660.0) - 1) < 0.03
    assert len(auto.render(mid, 0)) > 0
    r = np.random.default_rng(2)
    lo = auto.fit(0.2 * dsp.butter(r.standard_normal(SR // 3), 'low', 1000.0), kind='spectral', fast=True)
    hi = auto.fit(0.2 * dsp.butter(r.standard_normal(SR // 3), 'high', 4000.0), kind='spectral', fast=True)
    cen = lambda y: float(np.sum(np.abs(np.fft.rfft(Q.mono(y))) ** 2 * np.fft.rfftfreq(len(Q.mono(y)), 1 / SR))
                          / np.sum(np.abs(np.fft.rfft(Q.mono(y))) ** 2))
    c_lo, c_mid, c_hi = (cen(auto.render(m, 0)) for m in (lo, auto.morph(lo, hi, 0.5), hi))
    assert c_lo < c_mid < c_hi


# --------------------------------------------------------------------------------------------------- CLI
def test_cli_fit_and_morph(tmp_path, wav):
    from earshot.cli import main
    a = wav(struck(), 'a.wav')
    b = wav(struck(f=(1500.0, 3600.0), seed=4), 'b.wav')
    out = tmp_path / 'made' / 'a_fit.py'
    assert main(['fit', a, '-o', str(out), '--kind', 'modal', '--fast', '--quiet', '--json',
                 '--out', str(tmp_path / 'r.json')]) == 0
    r = json.load(open(tmp_path / 'r.json'))
    for k in ('recipe', 'wav', 'png', 'model'):
        assert (tmp_path / 'made').joinpath(r['files'][k].rsplit('/', 1)[-1]).exists()
    assert r['code_renders_fit'] is True and r['quality']['mel_db'] < 6
    assert main(['fit', '--morph', a, b, '--amount', '0.4', '-o', str(tmp_path / 'ab.py'), '--fast', '--quiet']) == 0
    rec = codegen.load_recipe(str(tmp_path / 'ab.py'))
    assert 'morph' in rec.SPACE and rec.DEFAULTS['morph'] == 0.4
    assert len(rec.render(rec.DEFAULTS, dsp.rng(0))) > 0


@pytest.mark.slow
def test_fitted_recipe_runs_in_the_forge(tmp_path):
    from earshot.forge import forge
    m = auto.fit(struck(), kind='modal', fast=True)
    rec = codegen.load_recipe(codegen.write_recipe(str(tmp_path / 'f.py'), m, target='a short plastic click'))
    r = forge(rec, refs=None, budget=8, pop=4, keep=1, seed=0, out=None, log=lambda *a: None)
    assert r['picks'] and set(r['picks'][0]['params']) >= set(rec.SPACE)
