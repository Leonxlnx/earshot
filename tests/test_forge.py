"""Fast tests for earshot forge: a toy recipe and a fake critic exercise the search, diverse picks, rendering, robust
re-scoring, family spaces, forge() end to end, and the dsp building blocks. No models are loaded."""
import json
import math
import types

import numpy as np
import pytest

from earshot.forge import dsp, forge
from earshot.forge import search
from earshot.forge.search import _decode, _encode, diverse, optimize, render, robust

SR = 48000
# the pop recipe's v1 (bubble model) DEFAULTS as of round 2, pinned here so the module may switch its DEFAULTS to the
# v2 tonal pop without this test changing meaning (render() dispatches on 'style': these keys take the v1 path)
POP_V1_DEFAULTS = {
    'f0': 440.159, 'q': 22.1207, 'rise': 0.991892, 'rise_tau': 0.0159722, 'att': 0.000357098, 'h2': 0.00847939,
    'wobble': 0.0112906, 'breath_amp': 0.0, 'breath_tau': 0.00848874, 'f2_ratio': 3.82514, 'f2_amp': 0.8, 'f2_q': 25.0,
    'f2_follow': 0.948223, 'sub_amp': 0.00328062, 'sub_ratio': 2.08913, 'sub_delay': 0.0111736, 'click_amp': 0.685048,
    'click_w': 7.72837e-05, 'tear': 0.000171419, 'jerks': 1, 'click_lp': 4881.31, 'shred_amp': 0.200779,
    'shred_tau': 0.00153019, 'shred_hp': 5000.0, 'hiss_amp': 0.0659931, 'hiss_f': 2418.88, 'hiss_q': 2.86992,
    'hiss_tau': 0.00113073, 'puff_amp': 0.0, 'puff_f': 121.063, 'ring_delay': 0.00105542, 'rt60': 0.249181,
    'wet': 0.291252, 'damp': 0.490701, 'size': 0.05, 'air_db': -0.476997, 'floor_db': -64.4367, 'swell_amp': 0.160452,
    'swell_t': 0.00252584, 'pre': 0.0269276, 'hp': 92.6866, 'lp': 10181.4, 'clip': 0.274425, 'cut_fade': 0.0153059}


def _toy_render(p, r):
    """a decaying sine at p['f'] Hz with time constant p['tau'], plus a little noise (r gives natural variation)"""
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * p['f'] * t) * np.exp(-t / p['tau']) + 0.001 * r.standard_normal(n)
    return x * {'soft': 0.2, 'hard': 1.0}[p['shape']]


def toy(**over):
    m = types.SimpleNamespace(__name__='earshot.forge.recipes.toy',
                              SPACE={'f': (200.0, 4000.0, 'log'), 'tau': (0.005, 0.2, 'log'), 'shape': ['soft', 'hard'],
                                     'n': (1, 9, 'int')},
                              DEFAULTS={'f': 300.0, 'tau': 0.15, 'shape': 'soft', 'n': 3}, render=_toy_render)
    for k, v in over.items():
        setattr(m, k, v)
    return m


class FakeCritic:
    """scores by closeness to 1 kHz and a 30 ms decay (no models); counts how many sounds it heard"""
    target, alts, refs = 'a 1 kHz ping', ['a low hum'], []

    def __init__(self):
        self.heard = 0

    def score(self, xs):
        rows = []
        for i, x in enumerate(xs):
            self.heard += 1
            m = np.asarray(x).mean(axis=1) if np.ndim(x) == 2 else np.asarray(x)
            F = np.abs(np.fft.rfft(m))
            f = np.fft.rfftfreq(len(m), 1 / SR)[int(np.argmax(F))]
            w = SR // 1000
            env = np.abs(m[:len(m) // w * w]).reshape(-1, w).max(axis=1)           # 1 ms envelope
            tau = max(1e-3, np.argmax(env < env.max() * math.exp(-1)) / 1000)
            total = 5.0 - 4 * abs(math.log2(max(f, 1.0) / 1000)) - abs(math.log(tau / 0.03))
            rows.append({'i': i, 'total': total, 'passes': total > 3.5, 'match_z': total, 'alt_margin': 1.0,
                         'p_real': 0.9, 'ref_typ_z': None, 'feat_pen': 0.0, 'sounds_like': ['a ping'],
                         'summary': f'total {total:.2f}'})
        return rows

    def holdout(self, xs):
        return [{'audioset_top': [['Ping', 0.5]], 'audioset_synth_max': 0.0} for _ in xs]


# ---------------------------------------------------------------------------------------------- search
def test_encode_decode_round_trip():
    space = toy().SPACE
    p = {'f': 1000.0, 'tau': 0.03, 'shape': 'hard', 'n': 7}
    q = _decode(space, _encode(space, p))
    assert abs(q['f'] - 1000.0) < 1e-6 and abs(q['tau'] - 0.03) < 1e-9
    assert q['shape'] == 'hard' and q['n'] == 7


def test_render_peak_normalises_and_rejects_silence():
    x = render(toy(), {'f': 1000.0, 'tau': 0.03, 'shape': 'soft', 'n': 1}, 1)
    assert x.dtype == np.float32
    assert abs(20 * np.log10(np.abs(x).max()) + 1.0) < 1e-3
    silent = toy(render=lambda p, r: np.zeros(4800))
    assert render(silent, {}, 1) is None


def test_render_logs_the_first_exception_once(capsys):
    search._failed.discard('broken')

    def boom(p, r):
        raise ValueError('bad mode count')
    bad = types.SimpleNamespace(__name__='broken', SPACE={}, render=boom)
    assert render(bad, {}, 1) is None and render(bad, {}, 2) is None
    err = capsys.readouterr().err
    assert err.count('bad mode count') == 1 and 'broken' in err


def test_optimize_improves_on_the_default_and_is_deterministic():
    c = FakeCritic()
    rec = toy()
    base = c.score([render(rec, rec.DEFAULTS, 0)])[0]['total']
    a = optimize(rec, c, budget=96, pop=16, seed=3)
    b = optimize(rec, FakeCritic(), budget=96, pop=16, seed=3)
    assert len(a['history']) == 96 and a['generations'] == 6
    assert a['best'][0]['score']['total'] > base + 2
    assert a['best'][0]['params'] == b['best'][0]['params']
    tot = [h['score']['total'] for h in a['best']]
    assert tot == sorted(tot, reverse=True)


def test_space_override_keeps_defaults_for_the_rest():
    rec = toy()
    res = optimize(rec, FakeCritic(), budget=32, pop=16, seed=1, space={'f': (900.0, 1100.0), 'shape': ['hard']})
    for h in res['history']:
        assert 900.0 <= h['params']['f'] <= 1100.0
        assert h['params']['shape'] == 'hard'
        assert h['params']['tau'] == rec.DEFAULTS['tau'] and h['params']['n'] == rec.DEFAULTS['n']


def test_diverse_picks_differ_and_passing_come_first():
    res = optimize(toy(), FakeCritic(), budget=64, pop=16, seed=2)
    picks = diverse(res['history'], toy(), k=3, min_dist=0.08)
    assert len(picks) == 3
    for i in range(3):
        for j in range(i):
            assert np.linalg.norm(picks[i]['u'] - picks[j]['u']) / math.sqrt(len(picks[i]['u'])) >= 0.08
    passes = [p['score']['passes'] for p in picks]
    assert passes == sorted(passes, reverse=True)


def test_robust_re_renders_on_new_seeds():
    c = FakeCritic()
    st = robust(toy(), c, {'f': 1000.0, 'tau': 0.03, 'shape': 'hard', 'n': 1}, seeds=4)
    assert c.heard == 4 and st['n'] == 4 and st['seeds'] == [5000, 5001, 5002, 5003] and st['failed'] == 0
    assert st['min'] <= st['mean'] <= st['max'] and st['pass_rate'] == 1.0
    flaky = toy(render=lambda p, r: _toy_render(p, r) if r.uniform() < 0.5 else np.zeros(100))
    st2 = robust(flaky, FakeCritic(), {'f': 1000.0, 'tau': 0.03, 'shape': 'hard', 'n': 1}, seeds=8)
    assert st2['failed'] + len(st2['totals']) == 8 and st2['pass_rate'] <= len(st2['totals']) / 8


def test_forge_end_to_end_with_a_prebuilt_critic(tmp_path):
    rep = forge(toy(), budget=48, pop=16, keep=2, seed=4, out=str(tmp_path), log=None, critic=FakeCritic(), robust=3)
    assert rep['target'] == 'a 1 kHz ping' and len(rep['picks']) == 2
    for p in rep['picks']:
        assert p['robust']['n'] == 3 and p['holdout']['audioset_top'][0][0] == 'Ping'
        assert (tmp_path / (p['name'] + '.wav')).exists()
    disk = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert disk['ref_scores'] == [] and len(disk['history']) == 48


def test_recipes_are_listed_and_render_their_defaults():
    from earshot.forge import load_recipe, recipes
    names = recipes()
    assert {'keypress', 'click', 'pop', 'whoosh', 'impact'} <= set(names)
    for name in names:
        rec = load_recipe(name)
        x = render(rec, dict(getattr(rec, 'DEFAULTS', {}) or {}), 7)
        assert x is not None, name
        assert np.isfinite(x).all() and 0.01 < len(x) / SR < 12, name


# ---------------------------------------------------------------------------------------------- dsp
def test_room_and_mic_accept_stereo():
    r = dsp.rng(1)
    hit = dsp.modal([620, 1490], [0.03, 0.02], [1, 0.5], 0.2, r)
    st = np.stack([hit, hit], axis=1)
    y = dsp.room(st, dsp.rng(2), rt60=0.2, wet=0.2)
    ym = dsp.room(hit, dsp.rng(2), rt60=0.2, wet=0.2)
    assert y.shape == (len(ym), 2)
    assert np.allclose(y[:, 0], ym) and np.allclose(y[:, 1], ym)       # same room on both channels as on mono
    z = dsp.mic(y, dsp.rng(3), floor_db=-40)
    assert z.shape == y.shape
    assert not np.allclose(z[:, 0], z[:, 1])                             # independent noise floor per channel
    zm = dsp.mic(ym, dsp.rng(3), floor_db=-40)
    assert zm.ndim == 1 and len(zm) == len(ym)


def test_svf_band_gain_and_normalised_band():
    t = np.arange(SR // 2) / SR
    x = np.sin(2 * np.pi * 1000 * t)
    q = 8.0
    band = dsp.svf(x, 1000.0, q, 'band')[SR // 4:]
    bandn = dsp.svf(x, 1000.0, q, 'bandn')[SR // 4:]
    assert abs(np.abs(band).max() - q) < 0.1 * q
    assert abs(np.abs(bandn).max() - 1.0) < 0.05


@pytest.mark.parametrize('mode', ['low', 'band', 'high'])
def test_svf_modes_are_unchanged_by_the_new_mode(mode):
    r = dsp.rng(5)
    x = r.standard_normal(4800)
    y = dsp.svf(x, 2000.0, 0.9, mode)
    assert np.isfinite(y).all() and y.shape == x.shape


# ---------------------------------------------------------------------------------------------- after the judge
def test_optimize_reps_scores_each_candidate_on_several_seeds():
    c = FakeCritic()
    a = optimize(toy(), c, budget=32, pop=16, seed=5, reps=2)
    b = optimize(toy(), FakeCritic(), budget=32, pop=16, seed=5, reps=2)
    assert c.heard == 64 and len(a['history']) == 32
    h = a['best'][0]
    assert len(h['score']['total_reps']) == 2
    assert abs(h['score']['total'] - float(np.mean(h['score']['total_reps']))) < 1e-3
    assert a['best'][0]['params'] == b['best'][0]['params']
    one = optimize(toy(), FakeCritic(), budget=32, pop=16, seed=5)          # reps=1 is the old search, unchanged
    assert 'total_reps' not in one['best'][0]['score']


def _burst(r, lo, hi, n=SR // 4, boost=None):
    from earshot.forge import dsp
    t = np.arange(n) / SR
    x = dsp.butter(r.standard_normal(n), 'band', (lo, hi), 2) * np.exp(-t / 0.04)
    if boost is not None:
        x = x + 8.0 * dsp.butter(r.standard_normal(n), 'band', boost, 4) * np.exp(-t / 0.04)
    return x


def test_tone_ref_finds_a_band_no_reference_has():
    from earshot.forge.tone import ToneRef, profile, shares
    r = np.random.default_rng(0)
    refs = [_burst(r, 300 * (1 + 0.1 * k), 6000) for k in range(5)]
    tr = ToneRef(refs)
    med, far = tr.spread
    assert 0 < med <= far
    good = tr.check(_burst(r, 320, 6000))
    bad = tr.check(_burst(r, 320, 6000, boost=(450, 650)))
    assert bad['tone_db'] > good['tone_db'] + 1.5
    assert any(w == 'above' and b in ('500 Hz', '630 Hz') for b, d, w in bad['outside'])
    assert tr.penalty(bad) > tr.penalty(good)
    assert 'above every ref' in bad['words']
    assert len(profile(refs[0])) == 28 and set(shares(refs[0])) == {'<150', '150-1k', '1-4k', '4-12k', '>12k'}
    loo = tr.check(refs[0], exclude=0)                     # a reference scored against the others only
    assert loo['tone_db'] == round(tr.ref_dist[0], 2)


def test_keypress_and_pop_new_chains_and_old_params_unchanged():
    """round 13-15 / 8-9 keys change the sound; parameters without them (rounds 1-12 / 1-7) keep the old code path"""
    from earshot.forge import load_recipe
    from earshot.forge.tone import profile
    kp, pop = load_recipe('keypress'), load_recipe('pop')
    new = render(kp, kp.DEFAULTS, 3)
    old = render(kp, {k: v for k, v in kp.DEFAULTS.items() if k != 'hpf'}, 3)
    assert profile(new)[0] < profile(old)[0] - 6           # the session high-pass empties the 31 Hz band
    p = dict(getattr(pop, 'DEFAULTS_V1', None) or POP_V1_DEFAULTS)
    assert 'style' not in p
    x = render(pop, p, 3)
    assert len(x) / SR < 0.45                              # the edit: the kit's pops are 0.19-0.41 s
    legacy = {k: v for k, v in p.items() if k not in ('hp', 'lp', 'clip', 'cut_fade', 'cut_db')}
    assert len(render(pop, legacy, 3)) > len(x)            # no edit keys: the old room-tail length
