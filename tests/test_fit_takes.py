"""Takes of fitted models (earshot 0.3 renderer fixes), on synthetic sounds only:
- a modal take rings out past a trimmed recording's end instead of stopping on a ringing tail (ring / take);
- the fit itself (identity render, seed 0) keeps the recording's length;
- a sines fit's residual noise follows the recording's roll-off above 16 kHz (no noise above 20 kHz it lacked);
- takes carry no DC offset; spectral takes keep one loudness across seeds; candidates can be scored on 3 seeds;
- 0.4: vary and the knobs keep a coherent (0.4) model's structure (a strike that stops the ring still stops it), and
  models without the flag render as before."""
import json

import numpy as np
import pytest

from earshot.forge import polish
from earshot.forge.fit import auto, modal, sines, spectral

SR = 48000


def _trimmed_strike(length=0.15, fade=0.03, seed=0):
    """a struck object whose slowest mode (tau 0.25 s) is still ringing where the 'recording' was trimmed, with the
    trim's linear fade-out (what a refset cut does)"""
    n = int(length * SR)
    t = np.arange(n) / SR
    o = int(0.005 * SR)
    tt = np.clip(t - t[o], 0, None)
    x = np.zeros(n)
    for f, tau, a in ((420.0, 0.25, 0.4), (1310.0, 0.04, 0.3), (2875.0, 0.015, 0.2)):
        x += (t >= t[o]) * a * np.exp(-tt / tau) * np.cos(2 * np.pi * f * tt)
    x += 1e-4 * np.random.default_rng(seed).standard_normal(n)
    nf = int(fade * SR)
    x[-nf:] *= np.linspace(1, 0, nf)
    return x


@pytest.fixture(scope='module')
def strike_model():
    x = _trimmed_strike()
    return x, modal.fit(x)


def test_modal_fit_keeps_the_recording_length(strike_model):
    x, m = strike_model
    y = modal.render(m, 0)
    assert len(y) == len(x)                                       # the fit, as measured against the recording
    assert np.array_equal(y, modal.render(m, 0, ring=False))      # ring is off at vary 0 (None -> vary > 0)
    assert np.array_equal(auto.render(m, seed=0), auto.render(m, seed=0, take=False))


def test_modal_take_rings_out_and_ends_clean(strike_model):
    x, m = strike_model
    y = modal.render(m, 0, ring=True)
    longest = max(modal.RING_LOUD_S, modal.RING_EXT * len(x) / SR)
    assert len(x) + int(modal.RING_MIN_S * SR) - 2 <= len(y) <= len(x) + int(longest * SR) + 2
    e = 10 * np.log10(np.convolve(y ** 2, np.ones(240) / 240, 'valid') + 1e-20)
    assert e[-1] < e.max() - 55                                    # faded out, not cut on a ringing mode
    assert polish.truncated(y)['trunc_db'] is None
    yv = modal.render(m, 3, vary=0.3)                              # a vary take rings out by default
    assert len(yv) > len(x) and polish.truncated(yv)['trunc_db'] is None


def test_modal_ring_in_the_standalone_script(strike_model):
    _, m = strike_model
    ns = {}
    exec(compile(modal.to_code(m, 'strike'), 'strike.py', 'exec'), ns)
    assert np.array_equal(ns['render'](2, ring=True, vary=0.2), modal.render(m, 2, ring=True, vary=0.2))


def _lossy_tone(dur=0.4, cut_hz=16000.0, seed=3):
    """a decaying tone over noise, band-limited at cut_hz like a lossy-coded recording"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    env = np.exp(-t / 0.08)
    x = 0.3 * env * np.sin(2 * np.pi * 880 * t) + 0.05 * env * rng.standard_normal(n)
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(n, 1 / SR) > cut_hz] = 0
    return np.fft.irfft(X, n)


def _above_20k_db(y):
    m = y.mean(1) if y.ndim == 2 else y
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return 10 * np.log10(P[f >= 20000].sum() / P.sum() + 1e-30)


def test_hf_rolloff_is_monotone_and_finds_a_band_limit():
    hf = sines.hf_rolloff(_lossy_tone())
    lv = [v for _, v in hf]
    assert [f for f, _ in hf] == list(sines.HF_POINTS) and lv[0] == 0.0
    assert all(b <= a for a, b in zip(lv, lv[1:]))
    assert lv[-1] < -40                                            # nothing above the 16 kHz cut


@pytest.fixture(scope='module')
def tone_model():
    return sines.fit(_lossy_tone(), windows=(1024,))


def test_sines_residual_follows_the_recordings_rolloff(tone_model):
    m = tone_model
    assert m['residual']['hf'][0] == [16000.0, 0.0]
    flat = dict(m, residual={k: v for k, v in m['residual'].items() if k != 'hf'})   # a model fitted before 0.3
    y, y_flat = sines.render(m, 0), sines.render(flat, 0)
    assert _above_20k_db(y) < _above_20k_db(y_flat) - 10
    assert _above_20k_db(y) < -60
    for model, ref in ((m, y), (flat, y_flat)):                    # the standalone script, with and without hf
        ns = {}
        exec(compile(sines.to_code(model, 'tone'), 'tone.py', 'exec'), ns)
        assert np.allclose(ns['render'](0), ref, atol=1e-9)


def test_sines_take_fades_its_noise_out(tone_model):
    """a trimmed tone whose noise is still sounding at the cut: the take fades it over RING_FADE_S, the fit keeps
    the recording's length"""
    m = json.loads(json.dumps(tone_model))
    n_id = int(round(m['duration'] * SR))
    res = m['residual']
    res['db'] = [row[:-1] + [max(row)] if max(row) > -149 else row for row in res['db']]   # loud to the last time
    assert len(sines.render(m, 0)) == n_id
    y = sines.render(m, 3, knobs={'vary': 0.3})                                              # ring=None -> on
    assert len(y) >= n_id + int(sines.RING_FADE_S * SR)
    e = 10 * np.log10(np.convolve(y ** 2, np.ones(240) / 240, 'valid') + 1e-20)
    assert e[-1] < e.max() - 55


def test_takes_have_no_dc_offset(tone_model):
    m = json.loads(json.dumps(tone_model))
    res = m['residual']
    res['db'][0] = [v + 30.0 if v > -149 else v for v in res['db'][0]]      # a loud lowest band (flat to DC)
    dc = lambda y: polish.dc(y)['dc_db']
    plain = [dc(auto.render(m, seed=s, vary=0.3, take=False)) for s in (1, 2, 3)]
    takes = [dc(auto.render(m, seed=s, vary=0.3)) for s in (1, 2, 3)]          # take=None -> True at vary > 0
    assert max(takes) < -60 and max(takes) < max(plain) - 10


def _gesture(n, rng):
    """a low-heavy noise swell (a cloth / air pass): 40 Hz - 3 kHz, a 0.1 s rise and a decay"""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= np.where((f > 40) & (f < 3000), (np.maximum(f, 40) / 100.0) ** -0.7, 0.0)
    t = np.arange(n) / SR
    env = np.clip(t / 0.1, 0, 1) ** 2 * np.exp(-np.maximum(t - 0.1, 0) / 0.08)
    return np.fft.irfft(X, n) * env


def test_spectral_takes_keep_one_loudness():
    x = _gesture(int(0.4 * SR), np.random.default_rng(5))
    x = 0.5 * x / np.abs(x).max()
    m = spectral.fit(x, texture=False, iters=1, seeds=(1,))
    lu = [polish.lufs_max(auto.render(m, seed=s)) for s in range(6)]
    assert max(lu) - min(lu) < 1.0
    # the expected level stays the original's (within the noise of one take)
    assert abs(np.mean([spectral._peak_rms_db(spectral.render(m, s, limit=False)) for s in range(6)])
               - m['level_db']) < 2.0


def test_fit_scores_candidates_on_three_seeds():
    x = _trimmed_strike(0.12)
    m = auto.fit(x, candidates=['modal', 'spectral'], fast=True, seeds=auto.TAKE_SEEDS)
    assert m['fit_engine']['seeds'] == [0, 1, 2]
    assert auto.fit(x, candidates=['modal'], fast=True)['fit_engine']['seeds'] == [0]


# ------------------------------------------------------------------------------------- 0.4: coherent takes
def _stopped_strike(coherent):
    """a two-strike modal model whose second strike stops the first one's ring (the same modes, opposite phase: what a
    least squares fit makes of a gated pulse or a damped hit), as a fit would hold it"""
    t1, t2, L = 480 / SR, 1920 / SR, 0.2
    modes = ((600.0, 0.5, 1.0), (1700.0, 0.4, 0.6), (3100.0, 0.3, 0.3))
    s1 = [[f, tau, a, 0.0] for f, tau, a in modes]
    s2 = [[f, tau, a * np.exp(-(t2 - t1) / tau), float(np.mod(2 * np.pi * f * (t2 - t1) + np.pi, 2 * np.pi))]
          for f, tau, a in modes]
    m = {'kind': 'modal', 'version': modal.VERSION, 'sr': SR, 'length_s': L, 'channels': 1, 'fade_out_s': 0.0,
         'strikes': [{'t': t1, 'ramp_ms': 0.0, 'modes': s1, 'attack': []},
                     {'t': t2, 'ramp_ms': 0.0, 'modes': s2, 'attack': []}],
         'sub': None, 'side': None,
         'noise': {'edges': list(modal.NOISE_EDGES), 'h_erb': modal.NOISE_H,
                   'env': [[[0.0, -240.0], [L, -240.0]] for _ in range(len(modal.NOISE_EDGES) + 1)]}}
    if coherent:
        m['coherent'] = True
    return m


def _tail_db(y):
    """the energy after the second strike (+ 5 ms) re the first strike's loudest 5 ms (dB)"""
    p = np.convolve(y ** 2, np.ones(240) / 240, 'valid')
    return float(10 * np.log10(np.mean(y[2160:SR // 10] ** 2) / p[:1900].max() + 1e-30))


def test_coherent_modal_vary_keeps_the_fit_structure():
    old, new = _stopped_strike(False), _stopped_strike(True)
    y_old, y_new = modal.render(old, 0), modal.render(new, 0)
    keep = np.ones(len(y_old), bool)                                          # the fit itself: the same samples, but
    keep[1920:1920 + int(round(modal.STRIKE_RAMP_S * SR)) + 1] = False        # the later strike's 0.15 ms onset ramp
    assert np.array_equal(y_old[keep], y_new[keep])                           # (round C2: no step, no click)
    assert np.max(np.abs(np.diff(y_new[1910:1940]))) < np.max(np.abs(np.diff(y_old[1910:1940])))
    assert _tail_db(modal.render(new, 0)) < -100                             # the second strike stops the ring
    t_old = [_tail_db(modal.render(old, s, vary=0.3, ring=False)) for s in (1, 2, 3)]
    t_new = [_tail_db(modal.render(new, s, vary=0.3, ring=False)) for s in (1, 2, 3)]
    assert max(t_new) < -40                                                   # coherent takes still stop
    assert min(t_old) > max(t_new) + 15                                       # per-mode jitter undid the stop
    t_size = [_tail_db(modal.render(m, 0, size=1.1)) for m in (old, new)]     # a pitch knob keeps it too (the
    assert t_size[1] < -35 and t_size[1] < t_size[0] - 20                     # decays it stretches leave a trace)
    assert not np.allclose(modal.render(new, 1, vary=0.3), modal.render(new, 2, vary=0.3))
    ns = {}
    exec(compile(modal.to_code(new, 'stop'), 'stop.py', 'exec'), ns)
    assert np.array_equal(ns['render'](3, vary=0.3), modal.render(new, 3, vary=0.3))


def test_fits_are_coherent_and_old_models_render_as_before(strike_model, tone_model, monkeypatch):
    """a model fitted before 0.4 (no 'coherent': the 0.3 bank's fit entries) renders as 0.3 did: no fade-in, no peak
    limit (sines and auto), and the flag changes nothing else at vary 0"""
    from earshot.forge.fit import auto
    assert strike_model[1].get('coherent') is True and tone_model.get('coherent') is True
    old = json.loads(json.dumps(tone_model))
    old.pop('coherent')
    assert sines.fade_in_s(old) == 0.0 and not sines._limited(old) and sines.fade_in_s(tone_model) > 0
    assert not auto.finished(old) and auto.finished(tone_model)
    hyb = {'kind': 'hybrid', 'layers': [{'role': 'tone', 'model': old}, {'role': 'air', 'model': {'kind': 'spectral_noise'}}]}
    assert not auto.finished(hyb)
    hyb['layers'][0]['model'] = tone_model
    assert auto.finished(hyb)
    y_old = sines.render(old, 2)
    monkeypatch.setattr(sines, 'finish', lambda y, sr=48000, fade_s=0.002, peak_db=-1.0, limit=True: np.array(y, float))
    assert np.array_equal(y_old, sines.render(tone_model, 2))
    assert not np.allclose(sines.render(old, 2, knobs={'vary': 0.3}), sines.render(tone_model, 2, knobs={'vary': 0.3}))


def test_spectral_fits_carry_the_finish_flag_and_old_ones_render_as_before():
    """a spectral model fitted by 0.3 has no 'finish' flag: auto.render leaves out the 0.4 fade-in and peak limit
    (its renders stay 0.3's, sample for sample); every 0.4 spectral fit carries it, and a morph keeps it"""
    x = _gesture(int(0.3 * SR), np.random.default_rng(9))
    m = spectral.fit(0.9 * x / np.abs(x).max(), texture=False, iters=1, seeds=(1,))
    assert m['finish'] is True and auto.finished(m)
    old = json.loads(json.dumps(m))
    old.pop('finish')
    assert not auto.finished(old)
    for s in (0, 3):
        assert np.array_equal(auto.render(old, seed=s), auto._render(old, seed=s, final=False))
    y = auto.render(m, seed=0)
    assert abs(y[0]) < 1e-12 and np.abs(y).max() <= 10 ** (auto.PEAK_DB / 20) + 1e-9
    assert auto.finished(auto.morph(m, m, 0.5)) and not auto.finished(auto.morph(old, old, 0.5))


def test_old_stereo_sines_side_keeps_the_0_3_arithmetic_under_gain(tone_model, monkeypatch):
    """a 0.3 stereo sines model (no 'side', a real width) with a gain knob: the side is scaled in 0.3's two steps,
    width then gain, so it stays bit-identical to 0.3.0 (one combined step differs in the last bits)"""
    old = json.loads(json.dumps(tone_model))
    for k in ('coherent', 'side'):
        old.pop(k, None)
    old.update(width=-6.4, channels=2)
    seen = {}
    real = sines.synth_noise

    def spy(*a, **kw):
        out = real(*a, **kw)
        seen[a[4]] = out.copy()
        return out
    monkeypatch.setattr(sines, 'synth_noise', spy)
    y2 = sines.render(old, 5, knobs={'gain_db': 3.0, 'stereo': True})
    y1 = sines.render(old, 5, knobs={'gain_db': 3.0, 'stereo': False})
    side = seen[5 + 1000003] * 10 ** (-6.4 / 20)
    side *= 10 ** (3.0 / 20)
    assert np.array_equal(y2[:, 0], y1 + side) and np.array_equal(y2[:, 1], y1 - side)


def test_vary_field_is_smooth_in_log_frequency():
    r = np.random.default_rng(4)
    fs = [sines.vary_field(r) for _ in range(200)]
    h = 200.0 * np.arange(1, 41)                                              # a harmonic series
    v = np.array([[F(f) for f in h] for F in fs])
    assert abs(v.std() - 1.0) < 0.15                                          # unit variance over draws
    step = np.abs(np.diff(v[:, 20:], axis=1)).mean()                          # neighbouring upper partials
    assert step < 0.25                                                        # independent draws differ by ~1.13
