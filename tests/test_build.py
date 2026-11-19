"""earshot build: drafts from words, the tuner, the printed lines; the tests that hear need the listen extra and
--slow."""
import json
import os

import numpy as np
import pytest

from earshot import build as B, recipe as R
from earshot.sounds import registry

# twenty requests of every structure that the reader finds recorded parts for without the listening model
REQUESTS = ['hail on a tin roof', 'a whale singing', 'a car alarm', 'a treadmill running', 'a jet flying over',
            'a dog barking three times', 'a magic portal opening', 'a busy train station', 'a snake hissing',
            'a door slam', 'rain on a car roof', 'a chainsaw cutting a log', 'a horse galloping', 'a crowd cheering',
            'a giant bear growling', 'an old wooden ship creaking', 'a force field humming', 'a forest at night',
            'a motorbike passing', 'footsteps in snow']


def _real_layers(rc):
    for lay in rc['layers']:
        if 'sound' in lay:
            row = registry.get(lay['sound']).row()
            assert B._real(row), (rc['name'], lay['sound'])


@pytest.mark.parametrize('words', REQUESTS)
def test_draft_gives_checked_recipes_of_recorded_parts(words):
    drafts = B.draft(words)
    assert 1 <= len(drafts) <= 3
    for d in drafts:
        assert R.check(d) == d or R.check(d)['layers'] == d['layers']
        assert d['says'] == words
        if d['name'] not in R.cookbook():       # a cookbook recipe of the thing is drafted as it is written
            _real_layers(d)
    assert len({json.dumps(d['layers'], sort_keys=True) for d in drafts}) == len(drafts)


def test_the_structure_follows_the_kind_of_sound():
    assert B.structure('hail on a tin roof') == 'scatter'
    assert B.structure('a lawn mower running') == 'run'
    assert B.structure('a jet flying over') == 'pass'
    assert B.structure('a wolf howling') == 'voice'
    assert B.structure('a dog barking three times') == 'row'
    assert B.structure('a magic portal opening') == 'magic'
    assert B.structure('a busy train station') == 'place'
    assert B.structure('a door slam') == 'one'


def test_without_is_obeyed_for_designs_and_families():
    first = B.draft('hail on a tin roof')
    used = {lay['sound'] for d in first for lay in d['layers'] if 'sound' in lay}
    assert used
    fam = sorted(used)[0].split('/')[0]
    one = sorted(used)[-1]
    for d in B.draft('hail on a tin roof', without=[fam, one]):
        for lay in d['layers']:
            if 'sound' in lay:
                assert lay['sound'] != one and lay['sound'].split('/')[0] != fam
    assert all(c['id'].split('/')[0] != fam for c in B.candidates('hail on a tin roof', without=fam))


def test_draft_says_plainly_when_nothing_is_near():
    with pytest.raises(ValueError, match='earshot recipe --example'):
        B.draft('zorp frobnicator')


def test_the_tuner_reads_its_ranges_from_the_recipe_form():
    rc = R.check({'name': 't', 'layers': [{'sound': 'door/slam', 'pitch': -2},
                                          {'sound': 'lock/latch', 'gain': -6, 'at': 0.1, 'lowpass': [3000, 9000],
                                           'events': {'count': 3, 'gap': 0.2}}]})
    nums = {path: (lo, hi) for path, lo, hi, _ in B.numbers(rc)}
    assert nums[('layers', 0, 'pitch')] == R.RANGE['pitch'][:2]
    assert ('layers', 0, 'gain') not in nums and ('layers', 0, 'at') not in nums
    assert nums[('layers', 1, 'gain')] == R.RANGE['gain'][:2]
    assert nums[('layers', 1, 'lowpass', 0)] == R.CUTOFF
    assert nums[('layers', 1, 'events', 'count')] == R.EVENT_RANGE['count'][:2]
    assert nums[('layers', 1, 'events', 'gap')] == R.EVENT_RANGE['gap'][:2]


def test_tune_is_deterministic_and_always_valid():
    rc = R.check({'name': 'knock_row', 'says': 'three knocks, 2 seconds',
                  'layers': [{'sound': 'door/slam', 'pitch': -2},
                             {'sound': 'lock/latch', 'gain': -6, 'at': 0.05, 'lowpass': 6000}]})
    a = B.tune(rc, 'a door slam, 2 seconds', rounds=3, seed=3)
    b = B.tune(rc, 'a door slam, 2 seconds', rounds=3, seed=3)
    assert a['recipe'] == b['recipe'] and a['changes'] == b['changes'] and a['score'] == b['score']
    assert R.check(a['recipe'])['layers'][0]['sound'] == 'door/slam'
    assert [lay.get('sound') for lay in a['recipe']['layers']] == ['door/slam', 'lock/latch']
    assert a['score'][1] >= a['score'][0] and a['z'] == (None, None)


def test_moved_numbers_stay_in_range():
    r = np.random.default_rng(0)
    for _ in range(200):
        assert -40.0 <= B._moved(-39.0, -40.0, 12.0, 'db', r) <= 12.0
        assert 20.0 <= B._moved(25.0, 20.0, 20000.0, 'ratio', r) <= 20000.0
        assert 2 <= B._moved(3, 2, 64, 'count', r) <= 64
        assert isinstance(B._moved(3, 2, 64, 'count', r), int)


def test_measures_of_a_click(click):
    m = B.measures(np.stack([click, click], axis=1))
    assert m['dur'] == pytest.approx(0.25, abs=0.01) and m['hit'] < 0.02 and m['bright'] > 1000


def test_build_writes_the_sound_and_its_recipe_and_says_how_to_go_on(tmp_path):
    out = str(tmp_path / 'slam.wav')
    res = B.build('a door slam', out=out, rounds=1, listen=False)
    lines = res['summary'].splitlines()
    assert lines[0].startswith('slam.wav: recipe/') and '[built]' in lines[0]
    assert lines[1].startswith('built from: ')
    assert any(x.startswith('not listened') for x in lines)
    assert any(x.startswith('measured: ') for x in lines)
    assert any(x.startswith('still off: ') for x in lines)
    assert lines[-1] == 'next: edit slam.recipe.json, then: earshot build "a door slam" --recipe slam.recipe.json'
    assert os.path.isfile(out) and res['files'] == [os.path.abspath(out)]
    with open(res['recipe_file'], encoding='utf-8') as fh:
        rc = R.check(json.load(fh))
    assert rc['says'] == 'a door slam'
    again = B.build('a door slam', out=str(tmp_path / 'again.wav'), recipe=res['recipe_file'], listen=False)
    assert again['tuned'] is None and again['recipe']['layers'] == rc['layers']


def test_hear_without_ears_names_fields(tmp_path):
    rc = {'name': 'long_slam', 'says': 'a door slam, 0.5 seconds',
          'layers': [{'sound': 'door/slam'}, {'source': 'noise', 'dur': 3.0, 'lowpass': 400, 'gain': -10}]}
    h = B.hear(rc, listen=False)
    assert 'z' not in h and h['dur'] > 2.5
    assert any('`dur`' in x for x in h['off'])


def test_the_command_line_builds(tmp_path, capsys):
    from earshot import cli
    out = str(tmp_path / 'cli.wav')
    assert cli.main(['build', 'a door slam', '-o', out, '--rounds', '0', '--no-listen']) == 0
    text = capsys.readouterr().out
    assert text.startswith('cli.wav: recipe/') and 'earshot build "a door slam" --recipe cli.recipe.json' in text


@pytest.mark.slow
def test_build_with_ears_says_the_fit_and_what_it_is_heard_as(tmp_path):
    pytest.importorskip('torch')
    res = B.build('a whale singing', out=str(tmp_path / 'whale.wav'), rounds=2)
    text = res['summary']
    assert 'heard as: ' in text and 'fit: z ' in text and 'p_real' in text
    assert res['heard']['verdict'] in ('sounds like it', 'a sketch: not clearly it yet')
    assert res['heard']['nearest']['id'] and res['heard']['z'] is not None
