"""Round 6 (forge v2, code): every delivered sound re-renders from its report.json (earshot.forge.schema).
Unit tests use fake critics and fast real recipes (no models). test_final_rerenders re-renders every pick of every
delivered final in out/forge_v2/*/final/ from its report.json and compares it with the WAV (max |error| <= 1e-6 over
every sample and channel); it skips cleanly when out/ is absent (a fresh clone)."""
import glob
import json
import os

import numpy as np
import pytest
import soundfile as sf

from earshot.forge import dsp, schema
from earshot.forge.recipes import shimmer, typing as ty

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINALS = sorted(d for d in glob.glob(os.path.join(ROOT, 'out', 'forge_v2', '*', 'final'))
                if os.path.exists(os.path.join(d, 'report.json')))


class Crit:
    refs = []
    target, alts, mode = 'a tick', [], 'designed'

    def score(self, xs, E=None, exclude=None):
        return [{'total': float(np.abs(x).mean()), 'passes': False} for x in xs]


# ------------------------------------------------------------------------------------------------ blocks
def test_render_ref_names_module_functions_only():
    assert schema.render_ref(ty.RUN.render) == ('typing', 'render_run')
    assert schema.render_ref(ty.render) == ('typing', 'render')
    assert schema.render_ref(lambda p, r: None) is None
    b = schema.make_block(ty.RUN.render, {'rate': 11.0}, 7, member='m', mode='designed', gain=0.5)
    assert set(b) == set(schema.BLOCK_KEYS) and b['gain_db'] == -6.02 and b['seed'] == 7 and b['member'] == 'm'


def test_rerender_is_the_search_render_at_its_gain():
    from earshot.forge.search import render
    p = dict(ty.DEFAULTS)
    x0 = render(ty, p, 11)
    b = schema.make_block(ty.render, p, 11)
    assert np.array_equal(schema.rerender(b), x0)
    b2 = schema.make_block(ty.render, p, 11, gain=0.7)
    assert np.abs(schema.rerender(b2) - (x0.astype(np.float64) * 0.7)).max() < 1e-7


def test_legacy_block_variant_labels_and_stand_in_names():
    rep = {'recipe': 'pro_typing_run', 'critic': {'mode': 'designed'}, 'family': {'labels': ['blip', 'glass']},
           'picks': [{'name': 'a', 'params': {'rate': 10.0}, 'seed': 3}, {'name': 'b', 'params': {'rate': 12.0}, 'seed': 4}]}
    b = schema.legacy_block(rep, rep['picks'][1])
    assert (b['recipe'], b['fn'], b['member'], b['mode'], b['gain_db']) == ('typing', 'render_run', 'glass', 'designed', 0.0)
    rep2 = {'recipe': 'pro_shimmer', 'mode': 'designed',
            'picks': [{'name': 'c', 'params': {'variant': 1}, 'family_member': 1, 'seed': 5, 'gain_db': -1.5,
                       'family_params': {'style': 'run', 'notes': 9}}]}
    b2 = schema.block(rep2, rep2['picks'][0])
    assert b2['params'] == {'style': 'run', 'notes': 9} and b2['member'] == '1' and b2['gain_db'] == -1.5
    rep3 = {'recipe': 'pro_x', 'family': {'members': [{'f': 1}, {'f': 2}]},
            'picks': [{'name': 'd', 'params': {'variant': 1}, 'seed': 1}]}
    with pytest.raises(schema.NotReproducible):
        schema.legacy_block(rep3, rep3['picks'][0])           # 'pro_x' is neither a module nor in LEGACY
    rep3['recipe'] = 'pro_typing_key'
    assert schema.legacy_block(rep3, rep3['picks'][0])['params'] == {'f': 2}
    own = {'recipe': 'nothing', 'picks': [{'name': 'e', 'params': {}, 'seed': 1, 'render': b}]}
    assert schema.block(own, own['picks'][0]) == b             # a schema-2 pick's own block wins


def test_fill_completes_params_only_when_bit_identical():
    b = schema.make_block(ty.RUN.render, {'rate': 12.0, 'dur': 1.2}, 9)
    f = schema.fill(b)
    assert set(ty.RUN_DEFAULTS) <= set(f['params']) and f['params']['rate'] == 12.0
    assert np.array_equal(schema.raw(f), schema.raw(b))


# ------------------------------------------------------------------------------------------------ writers
def test_forge_and_palette_picks_rerender_from_report(tmp_path):
    from earshot.forge import forge, palette
    rep = forge(ty, critic=Crit(), budget=2, pop=2, keep=2, log=None, holdout=False, outputs=False,
                out=str(tmp_path / 'f'))
    disk = json.loads((tmp_path / 'f' / 'report.json').read_text(encoding='utf-8'))
    assert disk['schema'] == schema.SCHEMA and all(p['render']['fn'] == 'render' for p in disk['picks'])
    assert all(p['render']['mode'] == 'designed' and p['render']['params'] == p['params'] for p in disk['picks'])
    assert all(r['ok'] for r in schema.check_final(str(tmp_path / 'f'))) and len(rep['picks']) == 2
    members = [{'params': {**ty.RUN_DEFAULTS, 'rate': 9.0}, 'seed': 1, 'family': 'slow'},
               {'params': {**ty.RUN_DEFAULTS, 'rate': 13.0, 'swell_db': -30.0}, 'seed': 2, 'family': 'fast'}]
    pal = palette(ty.RUN, members, Crit(), out=str(tmp_path / 'p'), name='run', holdout=False, log=None,
                  outputs=False)
    rows = schema.check_final(str(tmp_path / 'p'))
    assert [p['render']['member'] for p in pal['picks']] == ['slow', 'fast']
    assert all(p['render']['fn'] == 'render_run' and 'gain' in p['render'] for p in pal['picks'])
    assert all(r['ok'] and r['source'] == 'block' for r in rows), rows


def test_variant_recipe_picks_name_their_member(tmp_path):
    from earshot.forge import forge, variant_recipe
    members = [{**ty.DEFAULTS, 'ridge': 7000.0}, {**ty.DEFAULTS, 'ridge': 11000.0}, {**ty.DEFAULTS, 'tink_db': 0.0}]
    F = variant_recipe(ty.render, members, names=['low', 'high', 'tink'], TARGET='a tick', ALTS=[])
    forge(F, critic=Crit(), budget=3, pop=3, keep=3, init=[{'variant': 1}, {'variant': 2}], log=None,
          holdout=False, outputs=False, out=str(tmp_path))
    disk = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    for p in disk['picks']:
        v = p['params']['variant']
        assert p['render']['params'] == members[v] and p['render']['member'] == ['low', 'high', 'tink'][v]
    assert all(r['ok'] for r in schema.check_final(str(tmp_path)))


def test_a_lambda_stand_in_gets_no_block_and_no_schema_claim():
    import types
    from earshot.forge import forge
    F = types.SimpleNamespace(__name__='earshot.forge.recipes.typing', SPACE={'v': [0, 1]}, DEFAULTS={'v': 0},
                              render=lambda p, r: ty.render({'ridge': 7000.0 + 3000.0 * p['v']}, r))
    rep = forge(F, critic=Crit(), budget=2, pop=2, keep=2, log=None, holdout=False, outputs=False)
    assert 'schema' not in rep and not any('render' in p for p in rep['picks'])


def test_upgrade_writes_only_when_every_pick_matches(tmp_path):
    from earshot.forge.search import render
    p = {'rate': 11.0, 'dur': 1.2}
    x = render(ty.RUN, {**ty.RUN_DEFAULTS, **p}, 4)
    sf.write(str(tmp_path / 'r_01.wav'), x, 48000, subtype='PCM_24')
    rep = {'recipe': 'pro_typing_run', 'picks': [{'name': 'r_01', 'params': p, 'seed': 4, 'file': 'r_01.wav'}]}
    (tmp_path / 'report.json').write_text(json.dumps(rep), encoding='utf-8')
    _, rows = schema.upgrade(str(tmp_path), write=True, log=None)
    disk = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert rows[0]['ok'] and disk['schema'] == 2 and set(ty.RUN_DEFAULTS) <= set(disk['picks'][0]['render']['params'])
    rep['picks'][0]['seed'] = 5                                # the WAV no longer matches: nothing is written
    (tmp_path / 'report.json').write_text(json.dumps(rep), encoding='utf-8')
    with pytest.raises(schema.NotReproducible):
        schema.upgrade(str(tmp_path), write=True, log=None)
    assert 'schema' not in json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))


def test_render_params_follows_the_render_block(tmp_path):
    from earshot.forge import render_params
    b = schema.make_block(ty.RUN.render, {**ty.RUN_DEFAULTS, 'rate': 13.0}, 6)
    rep = {'recipe': 'pro_typing_run', 'picks': [{'name': 'a', 'params': {'variant': 0}, 'seed': 6, 'render': b}]}
    r = render_params('typing', rep, out=str(tmp_path))
    y, _ = sf.read(r['files'][0]['file'], always_2d=True)
    assert schema.compare(schema.rerender(b), y) < 1e-6        # the run, not a single key from typing.render


# ------------------------------------------------------------------------------------------------ shimmer r2_path
def test_shimmer_r2_path_is_off_unless_asked():
    run = dict(shimmer.FAMILY['pro_shimmer'][2], length=0.6)
    a = shimmer.render(dict(run), dsp.rng(3))
    assert np.array_equal(a, shimmer.render({**run, 'r2_path': 0}, dsp.rng(3)))     # absent = 0 = today's code
    assert not np.array_equal(a, shimmer.render({**run, 'r2_path': 1}, dsp.rng(3)))  # 1 = the round-2 draws
    sp = dict(shimmer.FAMILY['pro_shimmer'][0], length=0.5)                          # sparkle: tone bed on, same
    assert np.array_equal(shimmer.render(dict(sp), dsp.rng(2)), shimmer.render({**sp, 'r2_path': 1}, dsp.rng(2)))


# ------------------------------------------------------------------------------------------------ delivered finals
@pytest.mark.skipif(not FINALS, reason='no out/forge_v2/*/final/ here (a fresh clone): nothing delivered to re-render')
@pytest.mark.parametrize('final', FINALS, ids=lambda d: os.path.basename(os.path.dirname(d)))
def test_final_rerenders(final):
    rows = schema.check_final(final, tol=1e-6)
    assert rows, f'{final}: report.json has no picks'
    bad = [f"{r['name']}: err {r['err']:.2e} {r.get('why', '')}" for r in rows if not r['ok']]
    assert not bad, f'{final}: ' + '; '.join(bad)
