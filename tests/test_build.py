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
        # a cookbook recipe of the thing is drafted as it is written, the plain design as make hands it over
        if d['name'] not in R.cookbook() and not d['name'].endswith('_plain'):
            _real_layers(d)
    assert len({json.dumps(d['layers'], sort_keys=True) for d in drafts}) == len(drafts)


@pytest.mark.parametrize('words', REQUESTS)
def test_the_plain_design_is_among_the_drafts_and_built_names_only_what_is_there(words):
    drafts = B.draft(words)
    plain = B.plain_pick(words)
    assert any(d['layers'] == [{'sound': plain}] for d in drafts)
    for d in drafts:
        kind = d['built'].split(':')[0]
        if kind in B.KIND_FIELD:
            assert any(B.KIND_FIELD[kind] in lay for lay in d['layers']), d['built']
        for op, word in (('scatter', 'scattered'), ('pass', 'passing'), ('spin', 'pulsing'), ('stretch', 'stretched')):
            assert (word in d['built']) == any(op in lay for lay in d['layers']), (op, d['built'])


def test_many_of_one_thing_is_scattered():
    assert B.collective('a stampede of horses') == ('horses hooves galloping', 30.0)
    assert B.collective('a shower of books falling')[0] == 'books'
    assert B.collective('applause') == ('a clap', 40.0) and B.collective('a door slam') is None
    for words in ('a stampede of horses', 'a flock of geese', 'a swarm of flies', 'applause'):
        assert B.structure(words) == 'scatter'
    first = B.draft('a stampede of horses')[0]
    assert any('scatter' in lay for lay in first['layers']) and 'scattered' in first['built']


# a part's phrase -> what the grammar sets (docs/BUILD.md has the table)
PHRASES = [
    ('hundreds of hoofbeats on dirt, for 6 s, getting louder', {'many': 40.0, 'dur': 6.0, 'louder': True}),
    ('a low rumble under everything', {'bed': True}),
    ('a horse neighing, twice, far', {'count': 2, 'far': True}),
    ('a dog barking 3 times', {'count': 3}),
    ('a door slam, once, at 1.5 s', {'count': 1, 'at': 1.5}),
    ('branches snapping, a few', {'count': 3, 'loose': 0.5}),
    ('a swarm of bees', {'many': 30.0}),
    ('a crowd of people talking', {'many': 20.0}),
    ('rain, constant', {'many': 25.0}),
    ('a thunder crack at the end', {'at': 'end'}),
    ('a hiss at the start', {'at': 0.0}),
    ('a short knock after the hit', {'at': 'hit', 'stretch': 0.7}),
    ('a car passing', {'pass': True}),
    ('a jet flying over', {'pass': True}),
    ('a fly buzzing from left to right', {'pan': [-0.8, 0.8]}),
    ('wind, fading', {'fading': True}),
    ('a roar, much lower, swelling', {'pitch': -8.0, 'swell': True}),
    ('a creak, higher, rising', {'pitch': 4.0, 'bend': 'rising'}),
    ('a cymbal, reversed, muffled, soft', {'reverse': True, 'muffled': True, 'soft': True}),
    ('a fan, pulsing at 6 a second, close, loud', {'spin': (6.0, 'smooth'), 'close': True, 'loud': True}),
    ('rotor blades chopping at 12 hz, long', {'spin': (12.0, 'chop'), 'stretch': 1.6}),
    ('a knock sound=knock/door', {'sound': 'knock/door'}),
]


@pytest.mark.parametrize('phrase,said', PHRASES)
def test_a_part_phrase_becomes_the_fields_the_table_promises(phrase, said):
    r = B.read_part(phrase)
    assert r['said'] == said, r
    assert r['thing'] and not set(r['thing'].split()) & {'once', 'twice', 'far', 'under', 'getting'}


def test_parts_become_layers_and_say_which_design_each_became():
    rc, grounded = B.from_parts('a stampede of horses', ['hundreds of hoofbeats on dirt, for 6 s, getting louder',
                                                         'a low rumble under everything',
                                                         'a horse neighing, twice, far'])
    a, b, c = rc['layers']
    assert a['scatter']['dur'] == 6.0 and a['level'][0][1] < a['level'][-1][1]
    assert 'dur' in b and b['gain'] < 0
    assert c['events']['count'] == 2 and c['lowpass'] <= 3000 and c['gain'] < b['gain'] + 6
    assert c['sound'].startswith('horse/')
    for lay in rc['layers']:
        assert B._real(registry.get(lay['sound']).row())
    lines = B.part_lines(grounded, rc)
    assert lines[0].startswith("part 1 'hundreds of hoofbeats on dirt, for 6 s, getting louder': ")
    assert any(x.startswith('part 3 ') and 'horse/' in x for x in lines)
    assert 'scattered' in rc['built']


def test_a_part_the_catalogue_lacks_is_named_and_unread_words_are_said():
    g = B.ground('hundreds of armadillos on dirt')
    assert g['missing'] and 'armadillo' in g['missing']
    lines = B.part_lines([g])
    assert any('earshot has no' in x and 'only the nearest' in x for x in lines)
    g = B.ground('a horse neighing, twice, frobnicated')
    assert 'frobnicated' in g['unread'] and 'horse' not in g['unread']
    assert any(x == 'not understood in part 1: frobnicated' for x in B.part_lines([g]))
    with pytest.raises(ValueError, match='not in the catalogue'):
        B.ground('a knock sound=knock/nothing_like_it')


def test_a_part_finds_the_design_whose_own_words_name_its_thing():
    # "hoofbeats" is not a word of the reader, but a horse's hooves are in the catalogue: no "has no hoofbeats"
    g = B.ground('hundreds of hoofbeats on dirt')
    assert g['id'] == 'horse/trot' and not g['missing'] and 'hoofbeats' not in g['unread']
    assert not any('earshot has no' in x for x in B.part_lines([g]))
    # a word that names a sound is looked up the same way; the describing word does not choose
    g = B.ground('a low rumble under everything')
    assert 'rumbl' in B._own_index()[0][g['id']], g['id']
    # a thing of two words: the design that has both comes first
    g = B.ground('a train whistle, far')
    assert {'train', 'whistl'} <= set(B._own_index()[0][g['id']]), g['id']
    g = B.ground('a car engine idling')
    assert {'car', 'engin'} <= set(B._own_index()[0][g['id']]), g['id']
    for phrase in ('hundreds of hoofbeats on dirt', 'a low rumble under everything', 'a train whistle, far'):
        assert B._real(registry.get(B.ground(phrase)['id']).row())


def test_a_synthesized_design_is_taken_only_without_a_recording_and_named_as_such():
    g = B.ground('a spaceship humming')
    assert not B._real(registry.get(g['id']).row())
    line = B.part_lines([g])[0]
    assert "no recording of 'spaceship humming'" in line and f"{g['id']} is synthesized" in line
    assert B._real(registry.get(B.ground('a fridge humming')['id']).row())
    with pytest.raises(ValueError, match='only describes'):
        B.ground('something low and big')


def test_parts_are_deterministic_and_named_designs_are_kept():
    parts = ['a fly buzzing from left to right, for 3 s', 'a knock sound=door/slam, at the end']
    a, b = B.from_parts('a fly in a room', parts), B.from_parts('a fly in a room', parts)
    assert a[0] == b[0]
    assert a[0]['layers'][1]['sound'] == 'door/slam' and a[0]['layers'][1]['at'] > 1.0
    assert a[0]['layers'][0]['pan'] == [-0.8, 0.8]


def test_a_long_recording_of_many_is_played_not_scattered():
    rc, g = B.from_parts('a cheering crowd', ['many people cheering sound=crowd/cheer, for 6 s'])
    assert 'scatter' not in rc['layers'][0] and rc['layers'][0]['dur'] == 6.0
    assert 'not scattered' in g[0]['why']


def test_the_second_ear_takes_a_verdict_back_only_on_a_contradiction():
    labels = ['Horse', 'Clip-clop', 'Music', 'Speech', 'Rain', 'Wind', 'Whale vocalization']
    assert B.second_says('a stampede of horses', [('Horse', 0.4), ('Wind', 0.1)], labels)[0] is True
    assert B.second_says('a stampede of horses', [('Rain', 0.5), ('Wind', 0.2)], labels)[0] is False
    assert B.second_says('a dragon roaring', [('Wind', 0.5)], labels)[0] is None
    assert B.second_says('a dragon roaring', [('Music', 0.6)], labels)[0] is False
    assert B.second_says('a man whispering', [('Speech', 0.6)], labels)[0] is None
    near = {'z': 1.0}
    h = {'clean': True, 'z': 4.0}
    assert B._verdict(h, near, (True, 'it hears Horse')) == 'sounds like it'
    assert B._verdict(h, near, None) == 'sounds like it'
    assert B._verdict(h, near, (False, 'it hears Music first (0.60)')).startswith('a sketch: the second ear')


def test_a_comparison_is_only_with_a_design_of_the_same_thing():
    assert B.same_kind('whale/song', 'a whale singing') and not B.same_kind('wolf/howl', 'a whale singing')
    assert B.same_kind('rain/tin_roof', 'heavy rain on a roof')


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
