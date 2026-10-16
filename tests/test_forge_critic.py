"""Fast tests for the forge critic v2's model-free ears: every artifact detector and polish measure on synthetic
signals (a phase-jump click, a naive vs band-limited oscillator, a resonant ring, a comb-filtered tail, a truncated
tail, digital silence, clipping, a slow attack, a bumpy tail ...), the spectro-temporal envelope match (EnvRef),
PolishRef's ref-relative tolerances, and the v2 total. No models are loaded."""
import numpy as np
import pytest
from scipy import signal

from earshot.forge import polish as P
from earshot.forge.critic import W2, resolve_mode, total_v2
from earshot.forge.tone import CEN, EnvRef, bandgram, env_dist

SR = 48000


def t_(s):
    return np.arange(int(s * SR)) / SR


def decay_sine(f=440.0, tau=0.1, dur=0.6, fade=0.001):
    t = t_(dur)
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    n = int(fade * SR)
    x[:n] *= np.linspace(0, 1, n)
    return x


def burst(dur=0.6, tau=0.12, seed=0, floor=1e-4):
    r = np.random.default_rng(seed)
    t = t_(dur)
    x = r.standard_normal(len(t)) * np.exp(-t / tau)
    x[:48] *= np.linspace(0, 1, 48)
    return x + floor * r.standard_normal(len(t))


def saw(f0=440.0, dur=1.0, bandlimited=False):
    t = t_(dur)
    ph = f0 * t
    if bandlimited:
        x = sum(np.sin(2 * np.pi * h * ph) / h for h in range(1, int(20000 / f0) + 1)) * 2 / np.pi
    else:
        x = 2 * (ph % 1.0) - 1
    return 0.5 * x * np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.05)


# ------------------------------------------------------------------------------------------------ artifacts
def test_clicks_finds_a_phase_jump_not_a_clean_decay():
    x = decay_sine()
    y = x.copy()
    y[int(0.3013 * SR):] *= -1                       # a jump off a zero crossing
    assert P.clicks(x)['clicks'] == 0
    c = P.clicks(y)
    assert c['clicks'] == 1 and abs(c['click_times'][0] - 0.301) < 0.003
    assert P.clicks(x)['edge_db'] < -60
    cut = decay_sine(tau=1.0, dur=0.2)               # ends while loud: the file jumps to zero
    assert P.clicks(cut)['edge_db'] > -40


def test_clicks_ignores_a_physical_tick():
    x = burst(dur=0.4, tau=0.05)
    tick = decay_sine(5000, 0.0004, 0.004) * 0.05    # a 0.4 ms tick at 200 ms spreads over many samples
    x[int(0.2 * SR):int(0.2 * SR) + len(tick)] += tick
    assert P.clicks(x)['clicks'] == 0


def test_aliasing_naive_vs_bandlimited_oscillator():
    a, b = P.aliasing(saw()), P.aliasing(saw(bandlimited=True))
    assert a['fold'] > 0.05 and b['fold'] < 0.01
    assert a['nyq_db'] > -30 and b['nyq_db'] < -40
    n = P.aliasing(burst())
    assert n['fold'] == 0.0 and n['harmonic_frames'] == 0     # noise has no harmonic grid: not judged


def test_dc_offset_and_infrasound():
    x = burst()
    assert P.dc(x)['dc_db'] < -60
    assert P.dc(x + 0.05)['dc_db'] > -40 and P.dc(x + 0.05)['infra_db'] > -20


def test_resonance_finds_a_ring_in_noise():
    x = burst(dur=0.5, tau=0.1)
    r0 = P.resonances(x)
    r1 = P.resonances(x + 0.5 * decay_sine(3000, 0.1, 0.5))
    assert r0['res_db'] < 10
    assert r1['res_db'] > 15 and abs(r1['res_hz'] - 3000) < 200


def test_evolution_frozen_vs_sweeping():
    r = np.random.default_rng(1)
    t = t_(0.8)
    env = np.sin(np.pi * t / 0.8)
    frozen = r.standard_normal(len(t)) * env
    w = r.standard_normal(len(t))
    sos_lo = signal.butter(4, 400, 'lowpass', fs=SR, output='sos')
    sos_hi = signal.butter(4, 6000, 'highpass', fs=SR, output='sos')
    mix = np.clip((t - 0.2) / 0.4, 0, 1)            # crossfade from a dark to a bright noise
    moving = (signal.sosfilt(sos_lo, w) * (1 - mix) * 4 + signal.sosfilt(sos_hi, w) * mix) * env
    ef, em = P.evolution(frozen), P.evolution(moving)
    assert em['evolution_db'] > 2 * ef['evolution_db']
    assert em['centroid_oct'] > ef['centroid_oct'] + 1


def test_ring_comb_tail_vs_natural_tail():
    natural = burst(dur=1.5, tau=0.25)
    b = burst(dur=0.1, tau=0.02)
    d = int(0.005 * SR)
    a = np.zeros(d + 1)
    a[0], a[d] = 1.0, -0.9
    comb = signal.lfilter([1.0], a, np.concatenate([b, np.zeros(SR)]))
    assert P.ring(natural)['ring_z'] < 8
    c = P.ring(comb)
    assert c['ring_z'] > 20 and abs(c['ring_ms'] - 5.0) < 0.3


def test_truncated_tail():
    assert P.truncated(burst(dur=2.0))['trunc_db'] is None
    t = P.truncated(burst(dur=0.3))
    assert t['trunc_db'] is not None and t['trunc_db'] > -30
    faded = burst(dur=0.3)
    faded[-480:] *= np.linspace(1, 0, 480)           # a 10 ms fade does not hide a cut tail
    assert P.truncated(faded)['trunc_db'] is not None


def test_digital_silence_inside():
    x = burst()
    gap = np.concatenate([x[:4800], np.zeros(4800), x[4800:9600]])
    assert abs(P.silence(gap)['silence_ms'] - 100) < 2
    assert P.silence(x)['silence_ms'] == 0.0


def test_clipping_flat_tops():
    s = np.sin(2 * np.pi * 200 * t_(0.3))
    assert P.clipping(np.clip(2 * s, -1, 1))['clip_runs'] > 10
    assert P.clipping(s)['clip_runs'] == 0
    assert P.clipping(np.stack([s, 1.5 * s], 1))['overs'] > 0


def test_attack_tail_body_and_loudness():
    x = burst()
    slow = x.copy()
    slow[:2400] *= np.linspace(0, 1, 2400)
    assert P.attack(slow)['attack_ms'] > 20 > P.attack(x)['attack_ms']
    bumpy = x.copy()
    bumpy[9600:] += 0.6 * burst(dur=0.4, tau=0.05, seed=3)
    assert P.tail(bumpy)['bump_db'] > 4 > P.tail(x)['bump_db']
    assert P.tail(x)['decay_db_s'] < -40
    assert P.body(decay_sine(2000, 0.003, 0.1))['crest_db'] > P.body(x)['crest_db'] + 2
    ls = P.loudness_spread([x, 0.5 * x])
    assert abs(ls['spread_lu'] - 6.02) < 0.05


def test_measure_has_every_detector_and_plain_data():
    m = P.measure(burst())
    for k in ('clicks', 'edge_db', 'fold', 'nyq_db', 'dc_db', 'infra_db', 'res_db', 'ring_z', 'trunc_db', 'silence_ms',
              'clip_runs', 'evolution_db', 'attack_ms', 'onset_hf_db', 'bump_db', 't40_ms', 'crest_db'):
        assert k in m
    assert all(isinstance(v, (int, float, str, list, type(None))) for v in m.values())


# ------------------------------------------------------------------------------------------------ PolishRef
def test_polishref_flags_hard_defects_and_passes_clean():
    refs = [decay_sine(f, 0.12, 0.8) for f in (400, 500, 630)]
    pr = P.PolishRef(refs)
    clean = pr.check(decay_sine(520, 0.12, 0.8))
    assert clean['art_pen'] == 0 and clean['hard'] == [] and clean['words'].startswith('clean')
    y = decay_sine(520, 0.12, 0.8)
    y[int(0.2013 * SR):] *= -1
    c = pr.check(y)
    assert 'clicks' in c['hard'] and c['art_pen'] >= 1 and 'click' in c['words']
    cut = pr.check(decay_sine(520, 0.12, 0.25))
    assert 'trunc_db' in cut['hard']
    clip = pr.check(np.clip(3 * decay_sine(520, 0.12, 0.8), -1, 1))
    assert 'clip_runs' in clip['hard']


def test_polishref_refs_loosen_tolerances_and_designed_softens():
    refs = [burst(seed=s) + 0.1 for s in range(3)]    # every ref carries the same offset: not a defect here
    pr = P.PolishRef(refs)
    assert 'dc_db' not in pr.check(burst(seed=7) + 0.1)['art']
    assert 'dc_db' in P.PolishRef([]).check(burst(seed=7) + 0.1)['hard']
    y = decay_sine(520, 0.12, 0.8)
    y[int(0.2013 * SR):] *= -1
    d = P.PolishRef([], designed=True).check(y)
    assert 'clicks' not in d['hard'] and d['art'].get('clicks', 0) > 0


def test_polishref_relative_polish_and_leave_one_out():
    refs = [burst(seed=s, tau=0.1) for s in range(4)]
    pr = P.PolishRef(refs)
    slow = burst(seed=8, tau=0.1)
    slow[:4800] *= np.linspace(0, 1, 4800)
    assert pr.check(slow)['polish'].get('attack_ms', 0) > 0
    r = pr.check(None, exclude=0, m=pr.M[0])
    assert r['polish_pen'] == 0


# ------------------------------------------------------------------------------------------------ envelope
def test_bandgram_shape_and_level_independence():
    x = burst()
    G = bandgram(x)
    assert G.shape == (int(np.ceil(len(x) / 480)), len(CEN))
    tot = 10 * np.log10((10 ** (G / 10)).sum(1))
    assert abs(tot.max()) < 0.01
    assert np.allclose(G, bandgram(0.1 * x), atol=1e-6)


def test_env_dist_tolerates_small_stretch_but_not_a_cut_tail():
    x = burst(dur=1.0, tau=0.15)
    G = bandgram(x)
    st = signal.resample(x, int(len(x) * 1.05))          # 5 % slower (and 5 % lower): inside the tolerance
    cut = x.copy()
    cut[int(0.25 * SR):] *= np.linspace(1, 0, len(cut) - int(0.25 * SR)) ** 8
    assert env_dist(G, G) < 0.5
    assert env_dist(bandgram(st), G) < 3.0
    assert env_dist(bandgram(cut), G) > env_dist(bandgram(st), G) + 2


def test_envref_nearest_ratio_and_outside_cells():
    refs = [burst(dur=0.8, tau=tau, seed=s) for s, tau in enumerate((0.08, 0.1, 0.12, 0.15))]
    er = EnvRef(refs)
    m = er.check(refs[1])
    assert m['env_db'] < 0.5 and m['nearest_i'] == 1
    loo = er.check(None, exclude=1, G=er.G[1])
    assert loo['nearest_i'] != 1 and loo['env_ratio'] is not None
    assert EnvRef.penalty(loo) < 0.5
    odd = burst(dur=0.8, tau=0.1, seed=7) + 2.0 * decay_sine(3000, 0.3, 0.8)   # a long 3 kHz ring the refs lack
    o = er.check(odd)
    assert o['outside'] and any('1.6-3.2 kHz' in k for k, _, _ in o['outside'])
    assert o['env_out'] > 0 and EnvRef.penalty(o) > 0.5
    assert 'beyond every ref' in o['words']


# ------------------------------------------------------------------------------------------------ the v2 total
def _row(**kw):
    r = {'match_z': 4.0, 'alt_margin': 1.0, 'p_real': 0.9, 'ref_typ_z': 0.5, 'feat_pen': 0.2, 'tone_pen': 0.0,
         'env_pen': 0.0, 'art_pen': 0.0, 'polish_pen': 0.0, 'hard': []}
    r.update(kw)
    return r


def test_total_v2_arithmetic_and_modes():
    t, ok = total_v2(_row())
    assert abs(t - (4.0 + 2.0 * 0.8 + 0.5 * 0.5 - 0.7 * 0.2)) < 1e-9 and ok
    t2, _ = total_v2(_row(tone_pen=1.0, env_pen=0.5, art_pen=1.0, polish_pen=1.0))
    assert abs((t - t2) - (W2['tone'] + 0.5 * W2['env'] + W2['art'] + W2['polish'])) < 1e-9
    td, okd = total_v2(_row(p_real=0.1), mode='designed')    # realness neither counts nor gates in designed mode
    assert abs(td - (4.0 + 0.25 - 0.14)) < 1e-9 and okd
    assert not total_v2(_row(p_real=0.1))[1]
    assert not total_v2(_row(hard=['clicks']))[1]
    assert not total_v2(_row(env_pen=2.0))[1]
    assert total_v2(_row(p_real=0.1), W={'real': 0.0})[0] == pytest.approx(td, abs=1e-9)


def test_resolve_mode():
    assert resolve_mode('auto', 'a key press', []) == 'designed'
    assert resolve_mode('auto', 'a glitchy digital stutter', ['r.wav']) == 'designed'
    assert resolve_mode('auto', 'a deep sub drop', ['r.wav']) == 'designed'
    assert resolve_mode('auto', 'a single key press', ['r.wav']) == 'real'
    assert resolve_mode('real', 'a glitch', []) == 'real'
