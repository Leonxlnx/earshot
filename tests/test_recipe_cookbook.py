"""The cookbook (earshot/data/recipes) and the catalogue together: every recipe is valid; a recipe that is not a
sketch answers plain requests of its thing, and never a request the catalogue has a design of itself; a sketch is
never the answer; what a request asks of a cookbook thing (a length, far away, up close, softer, N times) is
followed where the recipe form can do it and said plainly where not. Slow tier: every recipe, answered through
`make`, renders clean."""
import json
import os

import pytest

from earshot import brief
from earshot import recipe as R
from earshot.make import asked_of_recipe, choose, found_by, make

BOOK = R.cookbook()
AUTOMATIC = sorted(n for n, r in BOOK.items() if not r.get('sketch'))
SKETCHES = sorted(n for n, r in BOOK.items() if r.get('sketch'))


def answered(request, sketch=False):
    """the cookbook recipe `make` answers a request with (None: the catalogue answers)"""
    b, cands = choose(request, top=8)
    r = R.answer(b, found_by(cands[0]['why'], cands[0]['id']) if cands else (), sketch=sketch)
    return r and r['name']


def test_every_cookbook_file_is_a_valid_recipe_named_as_its_file():
    files = sorted(f for f in os.listdir(R.FOLDER) if f.endswith('.json'))
    assert len(files) == len(BOOK) >= 60
    for f in files:
        with open(os.path.join(R.FOLDER, f), encoding='utf-8') as fh:
            r = json.load(fh)
        rc = R.check(r)
        assert rc['name'] == f[:-5], f
        assert rc.get('says') and rc.get('built') and len(rc.get('things') or ()) >= 3, f
        assert rc['built'].endswith('.') and len(rc['built']) <= 240, f
    owners = {}
    for name, r in BOOK.items():
        for t in r['things']:
            assert owners.setdefault(t.lower(), name) == name, f'{t!r} is a thing of {owners[t.lower()]} and {name}'


@pytest.mark.parametrize('name', AUTOMATIC)
def test_an_automatic_recipe_answers_three_plain_requests_of_its_thing(name):
    things = BOOK[name]['things']
    asked = [things[0], 'a ' + things[0], things[1]]
    assert [answered(q) for q in asked] == [name] * 3, asked


def test_a_thing_the_catalogue_has_a_recording_of_is_answered_by_the_recording(tmp_path):
    assert answered('a rocket launch') is None and answered('a rocket launch', sketch=True) is None
    assert answered('footsteps in mud') is None and 'footsteps_mud' not in BOOK
    r = make('a rocket launch', str(tmp_path / 'r.wav'))
    assert r['sound'] == 'rocket/launch' and r.get('source') != 'recipe'
    assert 'a start from parts' not in r['summary']
    assert found_by(['ph:launch rocket', 'engine']) == ('launch', 'rocket', 'engine')
    # a recording listed for a thing it is not (a lion's roar for a dragon) is only its nearest: the recipe is named
    # (here as a start: the dragon built on a lion is heard no better than the lion alone)
    assert found_by(['dragon', 'roar', 'ph:dragon roar'], 'lion/roar') == ('roar', 'roar')
    assert answered('a dragon roar') is None and answered('a dragon roar', sketch=True) == 'dragon_roar'


def test_a_sketch_is_never_the_answer():
    assert {'wolf_pack', 'helicopter_landing', 'typewriter_return', 'heartbeat_faster', 'punch_whoosh',
            'time_travel', 'dragon_roar', 'rocket_launch'} <= set(SKETCHES)
    for name in SKETCHES:
        for t in BOOK[name]['things']:
            assert answered(t) != name, (name, t)


def test_a_sketch_is_named_as_a_start_where_the_catalogue_has_only_a_nearest(tmp_path):
    r = make('a heartbeat getting faster', str(tmp_path / 'h.wav'))
    lines = r['summary'].splitlines()
    assert r['sound'] == 'heart/beat'
    assert any(ln.startswith('a start from parts: earshot make --recipe heartbeat_faster') for ln in lines)


def test_far_away_close_softer_and_louder_on_a_cookbook_thing():
    rc = BOOK['dragon_roar']
    far, level, lines = asked_of_recipe(rc, brief.read('a dragon roar far away'))
    assert far['space'] == 'hall' and level is not None and level < -14
    assert all(max(lay['lowpass'] if isinstance(lay['lowpass'], list) else [lay['lowpass']]) <= 3000
               for lay in far['layers'])
    assert lines == ['as asked: far away (in a hall, duller and 6 dB softer)']
    near, level, lines = asked_of_recipe(rc, brief.read('a dragon roar up close'))
    assert near['space'] == 'dry' and level is None and lines == ['as asked: up close (no room around it)']
    soft, level, lines = asked_of_recipe(rc, brief.read('a quiet dragon roar'))
    assert soft['layers'] == R.check(rc)['layers'] and level is not None and level < -14
    assert lines == ['as asked: softer (6 dB under the usual level)']
    loud, level, lines = asked_of_recipe(rc, brief.read('a loud dragon roar'))
    assert level is None and lines == ['not followed: louder; a cookbook sound is made at the loudest level of '
                                       'its kind']
    assert asked_of_recipe(rc, brief.read('a dragon roar'))[2] == []


def test_n_times_repeats_the_whole_or_says_it_cannot():
    rc = BOOK['dragon_roar']
    three, _, lines = asked_of_recipe(rc, brief.read('a dragon roar three times'))
    assert all(lay['events']['count'] == 3 for lay in three['layers'])
    gaps = {lay['events']['gap'] for lay in three['layers']}
    assert len(gaps) == 1 and gaps.pop() > 1.0 and lines[0].startswith('as asked: 3 times, ')
    rows = next(n for n in BOOK if any(lay.get('events') for lay in BOOK[n]['layers']))
    same, _, lines = asked_of_recipe(BOOK[rows], brief.read('it three times'), events=3)
    assert same['layers'] == R.check(BOOK[rows])['layers']
    assert lines == ['not followed: 3 times; this recipe is a row of events already, played once']


def test_a_length_far_away_and_twice_through_make(tmp_path):
    r = make('a werewolf howl far away, 2 seconds', str(tmp_path / 'w.wav'))
    lines = r['summary'].splitlines()
    assert r['sound'] == 'recipe/werewolf_howl' and r['takes'][0]['checks']['ok']
    assert abs(r['takes'][0]['dur_s'] - 2.0) <= 0.1
    assert 'as asked: far away (in a hall, duller and 6 dB softer)' in lines
    near = make('a werewolf howl', str(tmp_path / 'n.wav'))
    assert near['takes'][0]['lufs_max'] - r['takes'][0]['lufs_max'] >= 4.0
    two = make('a car crash twice', str(tmp_path / 'c.wav'))
    assert two['sound'] == 'recipe/car_crash' and two['takes'][0]['checks']['ok']
    assert any(ln.startswith('as asked: 2 times, ') for ln in two['summary'].splitlines())
    assert two['takes'][0]['dur_s'] > 1.8 * make('a car crash', str(tmp_path / 'o.wav'))['takes'][0]['dur_s']


def test_far_close_a_length_and_twice_on_things_built_from_parts(tmp_path):
    far = make('a machine gun burst far away', str(tmp_path / 'g.wav'))
    assert far['sound'] == 'recipe/machine_gun' and far['takes'][0]['checks']['ok']
    assert 'as asked: far away (in a hall, duller and 6 dB softer)' in far['summary'].splitlines()
    close = make('a dragon breathing fire up close', str(tmp_path / 'd.wav'))
    assert close['sound'] == 'recipe/dragon_fire'
    assert 'as asked: up close (no room around it)' in close['summary'].splitlines()
    short = make('a fireball spell, 1.5 seconds', str(tmp_path / 'f.wav'))
    assert short['sound'] == 'recipe/fireball_spell' and abs(short['takes'][0]['dur_s'] - 1.5) <= 0.1
    two = make('a plasma shot twice', str(tmp_path / 'b.wav'))
    assert two['sound'] == 'recipe/plasma_shot' and two['takes'][0]['checks']['ok']
    assert any(ln.startswith('as asked: 2 times, ') for ln in two['summary'].splitlines())


@pytest.mark.slow
@pytest.mark.parametrize('name', AUTOMATIC)
def test_every_automatic_recipe_is_made_clean_from_its_words(name, tmp_path):
    r = make('a ' + BOOK[name]['things'][0], str(tmp_path / f'{name}.wav'))
    assert r['sound'] == f'recipe/{name}' and r['takes'][0]['checks']['ok'], r['summary']
    assert not [ln for ln in r['summary'].splitlines() if ' asked: the layers end there' in ln]
