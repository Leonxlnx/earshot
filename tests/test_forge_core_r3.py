"""Fast tests for the round-3 core changes (no models): polish false flags (aliasing peaks only, comb ring guards,
designed / gated gates), critic absolute fallback targets for thin ref sets, the reverb IR cache, the holdout cache
key and the palette (first-class families) helpers."""
import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp, fx
from earshot.forge import polish as P

SR = 48000


def t_(s):
    return np.arange(int(s * SR)) / SR


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


def hum(f0=70.0, dur=1.0, lp=2500.0, sink=0.0):
    t = t_(dur)
    ph = np.cumsum(f0 * 2 ** (-sink * t / dur)) / SR
    x = dsp.butter(2 * (ph % 1) - 1, 'low', lp, order=4)
    return x * np.minimum(1, t / 0.01) * np.exp(-t / 0.25)


def subdrop(f0=64.0, f1=44.0, dur=1.2, parts=(1, 2, 3, 4, 5, 6)):
    t = t_(dur)
    ph = 2 * np.pi * np.cumsum(f1 + (f0 - f1) * np.exp(-t / 0.3)) / SR
    x = sum(np.sin(k * ph) / k ** 1.5 for k in parts)
    return x * np.minimum(1, t / 0.3) * np.exp(-np.maximum(t - 0.3, 0) / 0.25)


def comb(x, ms=5.0, g=0.9):
    d = int(ms / 1000 * SR)
    a = np.zeros(d + 1)
    a[0], a[d] = 1.0, -g
    return signal.lfilter([1.0], a, np.concatenate([x, np.zeros(SR)]))


# ------------------------------------------------------------------------------------------------ aliasing
def test_aliasing_counts_alias_peaks_not_a_noise_layer():
    r = np.random.default_rng(0)
    tone_noise = saw(300, bandlimited=True) + 0.125 * r.standard_normal(SR)     # noise 12 dB under the tone
    assert P.aliasing(tone_noise, legacy=True)['fold'] > 0.1                    # round 2: read as aliasing
    assert P.aliasing(tone_noise)['fold'] < 0.01
    assert P.aliasing(saw())['fold'] > 0.04                                    # a naive saw still folds
    assert P.aliasing(saw(bandlimited=True))['fold'] < 0.005


def test_aliasing_ignores_inharmonic_bell_partials_over_a_chance_grid():
    t = t_(1.5)
    bell = sum(a * np.sin(2 * np.pi * 1760 * q * t) * np.exp(-t / (0.8 / q ** 0.7))
               for q, a in ((1.0, 1.0), (2.76, 0.5), (5.40, 0.3), (8.93, 0.2), (11.34, 0.12)))
    bell *= np.minimum(1, t / 0.002)
    assert P.aliasing(bell)['fold'] < 0.02


# ------------------------------------------------------------------------------------------------ ring
def test_ring_guards_band_limited_subs_and_pitched_hums():
    s = P.ring(subdrop())
    assert s['ring_z'] is None and s['ring_src'] == 'narrow' and s['ring_raw_z'] > 50   # the 2 ms edge
    h = P.ring(hum())
    assert h['ring_z'] is None and h['ring_src'] == 'pitch'
    assert P.ring(hum(sink=0.6))['ring_z'] is None
    assert P.ring(subdrop(), legacy=True)['ring_z'] > 50                      # round 2's number
    assert P.ring(comb(burst(dur=0.1, tau=0.02)))['ring_z'] > 20              # a comb on noise still rings
    assert P.ring(comb(burst(dur=0.1, tau=0.02), ms=2.1))['ring_z'] > 20      # ... also at the 2 ms edge
    assert P.ring(comb(hum(70, 0.15), ms=5.0))['ring_z'] > 12                 # a comb at another delay than the pitch


# ------------------------------------------------------------------------------------------------ gates
def gated(level_db, tau=0.06):
    """a decaying noise burst stopped by a 1 ms gate where its envelope is level_db under the start"""
    x = burst(dur=1.0, tau=tau, floor=0.0)
    y = x[:int((-level_db / 8.686 * tau) * SR)].copy()
    y[-48:] *= np.linspace(1, 0, 48)
    return y


def test_designed_mode_allows_a_gate_once_the_level_is_low():
    low = gated(-36.0)
    assert -40 < P.truncated(low)['trunc_db'] < -31
    assert 'trunc_db' in P.PolishRef([]).check(low)['hard']                   # real mode: a cut tail
    d = P.PolishRef([], designed=True).check(low)
    assert 'trunc_db' not in d['art']
    high = gated(-18.0)
    assert -22 < P.truncated(high)['trunc_db'] < -13
    assert 'trunc_db' in P.PolishRef([], designed=True).check(high)['hard']   # a gate at -16 dB is still a cut ...
    g = P.PolishRef([], designed=True, gated=True).check(high)
    assert 'trunc_db' not in g['hard']                                         # ... unless the category is gated


def test_polish_targets_stand_in_for_missing_refs():
    slow = burst()
    slow[:2400] *= np.linspace(0, 1, 2400)
    assert 'attack_ms' not in P.PolishRef([]).check(slow)['polish']            # no refs: not judged (round 2)
    c = P.PolishRef([], targets={'attack_ms': (0.5, 3.0)}).check(slow)
    assert c['polish'].get('attack_ms', 0) > 0 and c['limits']['attack_ms'] == 5.5


# ------------------------------------------------------------------------------------------------ critic fallback
def test_study_targets_measures_and_penalties():
    from earshot.forge.critic import STUDY_TARGETS, study_measures, study_penalties, widen_targets
    t = t_(1.0)
    sub = np.sin(2 * np.pi * 50 * t) * np.minimum(1, t / 0.3) * np.exp(-np.maximum(t - 0.3, 0) / 0.2)
    m = study_measures(np.stack([sub, sub], 1))
    assert m['corr'] == 1.0 and m['sh150'] > -1 and 0.5 < m['length'] < 1.01 and 250 < m['attack'] < 330
    pen, off, words = study_penalties(m, STUDY_TARGETS['pro_sub_drop'])
    assert pen['tone'] == 0 and 'sh150' not in off
    wide = np.stack([sub, -sub], 1)                                            # anti-phase: corr -1
    pen2, off2, w2 = study_penalties(study_measures(wide), STUDY_TARGETS['pro_sub_drop'])
    assert off2['corr'] == 1.5 and pen2['polish'] == 1.5 and 'stereo corr' in w2
    hiss = np.random.default_rng(0).standard_normal(SR) * np.exp(-t / 0.2)
    pen3, off3, _ = study_penalties(study_measures(hiss), STUDY_TARGETS['pro_sub_drop'])
    assert pen3['tone'] == 2.0                                                  # no sub energy at all: capped
    tg = widen_targets({'t40': (150, 450)}, [{'t40': 86.0}, {'t40': None}])
    assert tg['t40'] == (86.0, 450)


def test_critic_study_fallback_fills_tone_env_polish_for_thin_ref_sets():
    from earshot.forge.critic import Critic
    c = Critic.__new__(Critic)                                                   # no models: the fallback alone
    c.refs, c.study, c._ref_sm = [], {'sh150': (-1.0, 0.0), 'length': (0.8, 1.6), 'corr': (0.8, 1.0)}, None
    t = t_(0.3)
    hiss = np.random.default_rng(1).standard_normal(len(t)) * np.exp(-t / 0.05)
    row = {'tone_ratio': None, 'env_ratio': None, 'polish_pen': 0.2}
    c._study(np.stack([hiss, hiss[::-1]], 1), row, None)
    assert row['tone_pen'] > 1 and row['env_pen'] > 0.5 and row['polish_pen'] > 1
    assert set(row['study']['pen']) == {'tone', 'env', 'polish'} and 'study targets' in row['study']['words']
    full = {'tone_ratio': 1.2, 'env_ratio': 0.9, 'tone_pen': 0.2}
    c._study(hiss, full, None)                                                  # refs gave a spread: tone / env kept
    assert full['tone_pen'] == 0.2 and 'env_pen' not in full
    c.refs = ['a.wav', 'b.wav', 'c.wav']
    untouched = {'tone_ratio': None}
    c._study(hiss, untouched, None)                                             # 3 refs: no fallback at all
    assert 'study' not in untouched


def test_holdout_key_is_content_based():
    from earshot.forge.critic import holdout_key
    x = np.random.default_rng(2).standard_normal(4800).astype(np.float32)
    assert holdout_key(x) == holdout_key(x.copy()) == holdout_key(np.stack([x, x], 1).mean(1))
    assert holdout_key(x) != holdout_key(x * 0.5)


# ------------------------------------------------------------------------------------------------ reverb IR
def test_reverb_ir_cache_replays_exactly_and_fixed_rooms_leave_r_alone():
    fx.ir_cache_clear()
    x = dsp.rng(5).standard_normal(2400) * np.exp(-np.arange(2400) / 400)
    kw = dict(rt60=0.4, size=0.3, lines=8, stereo=True, tail=0.3)
    r1 = dsp.rng(3)
    y1, a1 = fx.reverb(x, r1, **kw), r1.uniform()
    r2 = dsp.rng(3)
    y2, a2 = fx.reverb(x, r2, **kw), r2.uniform()
    assert np.array_equal(y1, y2) and a1 == a2 and fx.ir_cache_stats()['hits'] == 1
    r0 = dsp.rng(3)
    ref = fx._reverb_ir(r0, 0.4, 0.3, 0.01, 0.5, 1.2, 1.0, 0.7, -6.0, 1.0, 'noise', 8, 0.3, 1)
    assert np.array_equal(fx.reverb_ir(dsp.rng(3), rt60=0.4, size=0.3, lines=8, tail=0.3, cache=False), ref)
    r3 = dsp.rng(9)
    s0 = r3.bit_generator.state
    f1 = fx.reverb(x, r3, ir_seed=4, **kw)
    assert r3.bit_generator.state == s0                                         # a fixed room draws nothing from r
    assert np.array_equal(f1, fx.reverb(x, dsp.rng(10), ir_seed=4, **kw))
    g1, g2 = fx.ir_grid(rt60=0.400), fx.ir_grid(rt60=0.401)
    assert g1['rt60'] == g2['rt60'] and abs(g1['rt60'] / 0.4 - 1) < 0.015


def test_reverb_ir_eight_lines_is_finite():
    for s in range(40):
        ir = fx.reverb_ir(dsp.rng(s), rt60=0.3, size=0.45, lines=8, tail=0.2, inputs=2, cache=False)
        assert np.isfinite(ir).all()


# ------------------------------------------------------------------------------------------------ palette
def test_palette_keeps_every_member_and_matches_loudness(tmp_path):
    import json
    from earshot.forge import family_box, palette
    from test_forge import FakeCritic, toy
    rec = toy()
    members = [{'params': {'f': 1000.0, 'tau': 0.03, 'shape': 'hard', 'n': 1}, 'seed': 1, 'family': 'a'},
               {'params': {'f': 1000.0, 'tau': 0.031, 'shape': 'hard', 'n': 1}, 'seed': 2, 'family': 'b'},
               {'params': {'f': 600.0, 'tau': 0.1, 'shape': 'soft', 'n': 1}, 'seed': 3, 'family': 'c'}]
    rep = palette(rec, members, FakeCritic(), out=str(tmp_path), name='ping', log=None)
    assert [p['family'] for p in rep['picks']] == ['a', 'b', 'c']                # near twins are all kept
    assert rep['loudness']['after']['spread_lu'] < 0.05
    assert rep['variety']['members'] is None and rep['variety']['tone_min_db'] is not None   # FakeCritic: no CLAP
    assert (tmp_path / 'ping_03.wav').exists() and (tmp_path / '_audition.wav').exists()
    disk = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert len(disk['picks']) == 3 and 'palette' in disk['summary']
    box = family_box(rec, {'f': 1000.0, 'tau': 0.03, 'shape': 'hard', 'n': 5}, 0.1)
    assert box['f'][0] < 1000 < box['f'][1] and box['f'][2] == 'log' and 'shape' not in box and box['n'][2] == 'int'


def test_forge_families_runs_one_search_per_member(tmp_path):
    from earshot.forge import forge
    from test_forge import FakeCritic, toy
    fam = {'ping': {'f': 1000.0, 'tau': 0.03, 'shape': 'hard', 'n': 1},
           'low': {'init': {'f': 400.0, 'tau': 0.05, 'shape': 'hard', 'n': 1}, 'budget': 16}}
    rep = forge(toy(), budget=32, pop=16, seed=1, out=str(tmp_path), log=None, critic=FakeCritic(), families=fam)
    assert [p['family'] for p in rep['picks']] == ['ping', 'low'] and len(rep['runs']) == 2
    assert rep['runs'][1]['budget'] == 16
    assert 250 < rep['picks'][1]['params']['f'] < 630          # the box (+-0.15 of the log range) kept the character
    assert rep['picks'][1]['params']['shape'] == 'hard'                      # the member's choice, not DEFAULTS' soft
    fixed = {'f': {'init': {'f': 900.0, 'tau': 0.03, 'shape': 'hard', 'n': 1}, 'fixed': {'f': 900.0}}}
    rep2 = forge(toy(), budget=16, pop=16, seed=2, log=None, critic=FakeCritic(), families=fixed, outputs=False)
    assert rep2['picks'][0]['params']['f'] == 900.0


def test_prescreen_counts_the_study_fallback_without_refs():
    from earshot.forge.search import Prescreen
    t = t_(0.3)
    hiss = np.random.default_rng(1).standard_normal(len(t)) * np.exp(-t / 0.05)
    plain = Prescreen([], polish=False).row(hiss)
    pre = Prescreen([], polish=False, study={'sh150': (-1.0, 0.0), 'length': (0.8, 1.6)}).row(hiss)
    assert pre['pre'] < plain['pre'] - 1 and set(pre['study']) == {'sh150', 'length'}


def test_score_refs_is_memoised_per_critic(monkeypatch):
    from earshot.forge import critic as C
    c = C.Critic.__new__(C.Critic)
    c.refs, c.W, c.mode, c.version, c.extra = ['a.wav', 'b.wav'], dict(C.W2), 'real', 2, None
    calls = []
    monkeypatch.setattr(C.audio, 'load', lambda p: np.zeros(480))
    c.score = lambda xs, exclude=None: calls.append(exclude) or [{'total': 1.0, 'i': i} for i in range(len(xs))]
    a = c.score_refs()
    a[0]['total'] = 99.0                                                        # callers may edit their copy
    b = c.score_refs()
    assert len(calls) == 1 and b[0]['total'] == 1.0
    c.W = {**c.W, 'tone': 0.0}
    c.score_refs()
    assert len(calls) == 2                                                      # new weights: scored again
