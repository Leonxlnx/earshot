"""Core round 5 (forge v2): start points are filled from DEFAULTS, never silently at mid-range; delivered family
members are always kept in the seed bank. Fake critics and toy recipes only: no models are loaded."""
import json
import sys
import types
import warnings

import numpy as np
import pytest

from earshot.forge import search
from earshot.forge.search import MidRangeFill, fill_start, midrange, optimize, recipe_defaults, save_seeds

SR = 48000


class Crit:
    refs = []
    target, alts = 'a ping', []

    def score(self, xs, E=None, exclude=None):
        return [{'total': float(np.abs(x).mean()), 'passes': False} for x in xs]


def _module(name, space, defaults):
    """a recipe module registered in sys.modules (what recipe_defaults() reaches through render.__module__)"""
    seen = []
    mod = types.ModuleType(name)

    def render(p, r):
        seen.append(dict(p))
        t = np.arange(SR // 20) / SR
        return np.sin(2 * np.pi * p['f'] * t) * np.exp(-t / 0.01) * (1 + p.get('tick', 0.0))

    render.__module__ = name
    mod.SPACE, mod.DEFAULTS, mod.render = space, defaults, render
    sys.modules[name] = mod
    return mod, seen


SPACE = {'f': (200.0, 2000.0, 'log'), 'tick': (0.0, 1.5), 'shape': [0, 1], 'n': (1, 9, 'int')}
DEFAULTS = {'f': 500.0, 'tick': 0.0, 'shape': 0, 'n': 3}


def test_final_pass_fills_a_members_missing_keys_from_the_module_defaults_not_mid_range():
    # the round-3 pro_impact bug: a driver's final pass over members that do not set every key, with a stand-in
    # recipe whose DEFAULTS is its first member; the missing keys came back at mid-range (snap 0.75, a 0 dB tick)
    mod, seen = _module('earshot_test_r5_recipe_a', SPACE, DEFAULTS)
    members = [{'f': 700.0, 'n': 5}, {'f': 1200.0, 'n': 2, 'shape': 1}]
    recF = types.SimpleNamespace(__name__='final_pass', SPACE=mod.SPACE, DEFAULTS=dict(members[0]), render=mod.render)
    with warnings.catch_warnings():
        warnings.simplefilter('error', MidRangeFill)
        res = optimize(recF, Crit(), budget=3, pop=3, init=members, log=None, seed=7)
    got = [h['params'] for h in res['history']]
    assert got[1] == {'f': 700.0, 'n': 5, 'tick': 0.0, 'shape': 0}               # exact member + DEFAULTS
    assert got[2] == {'f': 1200.0, 'n': 2, 'shape': 1, 'tick': 0.0}
    assert all(p['tick'] == 0.0 for p in got) and midrange(SPACE['tick']) == 0.75
    assert res['fill_warnings'] == []
    assert recipe_defaults(recF) == {**DEFAULTS, **members[0]}


def test_mid_range_only_without_any_defaults_and_it_warns():
    mod, seen = _module('earshot_test_r5_recipe_b', SPACE, {'f': 500.0, 'shape': 0, 'n': 3})   # no 'tick' anywhere
    with pytest.warns(MidRangeFill, match='tick'):
        res = optimize(mod, Crit(), budget=2, pop=2, init=[{'f': 900.0}], log=None, seed=1)
    assert res['history'][0]['params']['tick'] == 0.75
    assert {(w['start'], w['key'], w['from']) for w in res['fill_warnings']} == {(0, 'tick', 'mid-range'),
                                                                                 (1, 'tick', 'mid-range')}


def test_defaults_outside_the_space_are_clipped_into_it_and_reported():
    narrow = {'f': (800.0, 1600.0, 'log'), 'tick': (0.2, 0.6)}
    full, notes = fill_start(narrow, {'f': 1000.0}, {'f': 500.0, 'tick': 0.0}, own={})
    assert full == {'f': 1000.0, 'tick': 0.2}
    assert notes[0]['from'] == 'DEFAULTS clipped' and notes[0]['was'] == 0.0
    full, notes = fill_start(narrow, {'f': 100.0, 'tick': 0.3}, {}, own={})
    assert full['f'] == 100.0 and notes == [{'key': 'f', 'value': 800.0, 'from': 'start clipped', 'was': 100.0}]


def test_es_starts_are_scored_exactly_as_given_including_keys_outside_the_space():
    mod, seen = _module('earshot_test_r5_recipe_c', SPACE, DEFAULTS)
    start = {'f': 733.3333333333, 'tick': 0.1, 'shape': 1, 'n': 4, 'extra': 'kept'}
    res = optimize(mod, Crit(), budget=4, pop=4, init=[start], log=None, seed=3, space={'f': (200.0, 2000.0, 'log')})
    assert res['history'][1]['params'] == start                       # no encode / decode round trip, own extras
    assert all(h['params']['tick'] == 0.0 for h in res['history'][2:])  # searched candidates: DEFAULTS outside space


def test_forge_report_carries_the_fill_warnings(tmp_path):
    from earshot.forge import forge
    mod, seen = _module('earshot_test_r5_recipe_d', SPACE, {'f': 500.0, 'shape': 0, 'n': 3})
    mod.TARGET, mod.ALTS = 'a ping', []
    with pytest.warns(MidRangeFill):
        rep = forge(mod, critic=Crit(), budget=2, pop=2, keep=1, log=None, holdout=False, outputs=False)
    assert rep['fill_warnings'][0]['key'] == 'tick'


# ---------------------------------------------------------------------------------------------- seed bank
def _picks(scores, f0, **extra):
    return [{'params': {'f': f0 + i}, 'score': {'total': s}, 'family': f'm{i}', 'seed': 10 + i, **extra}
            for i, s in enumerate(scores)]


def test_delivered_members_are_always_kept_even_when_outscored_and_over_the_limits(tmp_path):
    folder = str(tmp_path)
    save_seeds('toy', _picks([9.0] * 6, 100.0), 'a sting', ['r.wav'], folder=folder)          # strong attempts
    bank = save_seeds('toy', _picks([1.0, 0.5, -np.inf, 2.0, 0.1], 500.0), 'a sting', ['r.wav'], folder=folder,
                      delivered='logo_sting', per_context=6, cap=8)
    pinned = [e for e in bank['entries'] if e.get('delivered') == 'logo_sting']
    assert sorted(e['params']['f'] for e in pinned) == [500.0, 501.0, 502.0, 503.0, 504.0]     # all 5, -inf too
    assert [e['member'] for e in pinned] == ['m0', 'm1', 'm2', 'm3', 'm4'] and pinned[2]['rank_score'] is None
    assert len(bank['entries']) == 8                                                           # 3 ordinary left
    # more attempts in the same context never push a delivered member out
    for k in range(3):
        bank = save_seeds('toy', _picks([20.0 + k] * 6, 1000.0 * (k + 2)), 'a sting', ['r.wav'], folder=folder,
                          cap=8)
    assert sum(1 for e in bank['entries'] if e.get('delivered') == 'logo_sting') == 5
    # the delivered members are the first starts of their context
    got = search.load_seeds('toy', 'a sting', ['r.wav'], n=5, folder=folder)
    assert sorted(p['f'] for p in got) == [500.0, 501.0, 502.0, 503.0, 504.0]


def test_a_re_delivery_unpins_the_previous_delivery_of_that_category(tmp_path):
    folder = str(tmp_path)
    save_seeds('toy', _picks([1.0, 1.0], 100.0), 't', [], folder=folder, delivered='impact')
    save_seeds('toy', _picks([1.0], 900.0), 't', [], folder=folder, delivered='pro_impact')
    bank = save_seeds('toy', _picks([0.5, 0.5], 300.0), 't', [], folder=folder, delivered='impact')
    tags = {e['params']['f']: e.get('delivered') for e in bank['entries']}
    assert tags[300.0] == tags[301.0] == 'impact' and tags[900.0] == 'pro_impact'
    assert tags.get(100.0) is None and all(e.get('was_delivered') == 'impact' for e in bank['entries']
                                           if e['params']['f'] in (100.0, 101.0))


def test_pin_delivered_reads_a_final_report_with_family_params(tmp_path):
    rep = {'target': 'a ding', 'refs': ['a/b/r1.wav'],
           'picks': [{'name': 'notify_01', 'family_member': 'low', 'params': {'f0': 880.0},
                      'family_params': {'f0': 440.0, 'decay': 0.3}, 'seed': 4, 'score': {'total': None}}]}
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(rep), encoding='utf-8')
    bank = search.pin_delivered('toy', str(path), 'notify', folder=str(tmp_path / 'seeds'))
    e = bank['entries'][0]
    assert e['params'] == {'f0': 880.0, 'decay': 0.3} and e['delivered'] == 'notify' and e['seed'] == 4
    assert e['context'] == 'a ding | r1.wav' and e['member'] == 'notify_01'


def test_the_pro_impact_round_3_case_on_the_real_recipe_gets_no_tick():
    # members built off impact's _PRO presets do not set snap / sheen; the round-3 driver's final pass (a stand-in whose
    # DEFAULTS is the first member) filled them at mid-range (snap 0.75, sheen 1.0: a 0 dB onset tick)
    from earshot.forge.recipes import impact
    gone = ('snap', 'sheen', 'sheen_ms', 'sheen_f', 'pre_shape', 'hall_tone')
    members = [{k: v for k, v in dict(p).items() if k not in gone} for _, p in impact.FAMILIES['pro_impact'][:2]]
    seen = []

    def render(p, r):                                   # records the parameters instead of rendering 2-7 s hits
        seen.append(dict(p))
        return np.sin(np.arange(4800) / 7.0) * np.exp(-np.arange(4800) / 900.0)

    render.__module__ = impact.__name__
    recF = types.SimpleNamespace(__name__='pro_impact', SPACE=impact.SPACE, DEFAULTS=dict(members[0]), render=render)
    with warnings.catch_warnings():
        warnings.simplefilter('error', MidRangeFill)
        optimize(recF, Crit(), budget=3, pop=3, init=members, log=None, seed=7)
    assert len(seen) == 3
    for p in seen:
        assert p['snap'] == 0.0 and p['sheen'] == 0.0 and p['pre_shape'] == impact.DEFAULTS['pre_shape']
        assert p['hall_tone'] == impact.DEFAULTS['hall_tone']
