"""earshot.synth.perform and learned.perform: a performance is several events of one thing played the way a person
does it. Deterministic; never the same recording twice in a row; as many events as asked for; a pace inside the
pattern's range and never on a grid; clean start and end; peak at -1 dBFS; the request decides between one event and
the action (brief.performance, make's `events`). Model-free and fast: a few performances of short things."""
import numpy as np
import pytest

from earshot import brief
from earshot.forge import dsp
from earshot.make import performance
from earshot.sounds import learned
from earshot.synth import master, perform

SR = dsp.SR


@pytest.mark.parametrize('name', sorted(perform.PATTERNS))
def test_a_plan_is_deterministic_and_has_the_events_asked_for(name):
    a = perform.plan(name, dsp.rng(3), members=(2, 0, 1), events=5)
    b = perform.plan(name, dsp.rng(3), members=(2, 0, 1), events=5)
    assert a == b
    assert len(a['onsets']) == len(a['gains']) == len(a['members']) == 5
    assert all(np.diff(a['onsets']) > 0)
    assert all(x != y for x, y in zip(a['members'], a['members'][1:]))


@pytest.mark.parametrize('name', sorted(perform.PATTERNS))
def test_the_usual_number_and_a_length_filled_at_the_pace(name):
    lo, hi = perform.PATTERNS[name]['events']
    for seed in range(4):
        n = len(perform.plan(name, dsp.rng(seed), members=(0, 1))['onsets'])
        assert lo <= n <= hi
    p = perform.plan(name, dsp.rng(1), members=(0, 1), dur=4.0)
    assert len(p['onsets']) >= 2


def test_a_walk_alternates_two_feet_at_a_walking_pace_and_off_the_grid():
    p = perform.plan('walk', dsp.rng(7), members=(4, 1, 2), events=8)
    assert p['members'] == [4, 1] * 4
    gaps = np.diff(p['onsets'])
    assert 100 / 60 * 0.9 <= 1 / gaps.mean() <= 120 / 60 * 1.1
    assert gaps.std() > 0.002                       # not on a grid
    lead, trail = [], []                            # the trailing foot lands a little softer, on average
    for seed in range(12):
        g = np.log(perform.plan('walk', dsp.rng(seed), members=(0, 1), events=8)['gains'])
        lead.append(g[0::2].mean())
        trail.append(g[1::2].mean())
    assert np.mean(trail) < np.mean(lead)


def test_a_knock_is_three_or_four_hits_about_a_fifth_of_a_second_apart_the_first_the_firmest():
    for seed in range(5):
        p = perform.plan('knock', dsp.rng(seed), members=(0,))
        assert len(p['onsets']) in (3, 4)
        assert 0.15 <= float(np.mean(np.diff(p['onsets']))) <= 0.26
        assert p['gains'][0] == max(p['gains'])


def test_typing_runs_at_five_to_eight_keys_a_second_with_a_deeper_key_at_the_pauses():
    p = perform.plan('type', dsp.rng(2), members=(3, 1, 4, 0, 2), events=16)
    rate = (len(p['onsets']) - 1) / (p['onsets'][-1] - p['onsets'][0])
    assert 3.5 <= rate <= 8.5
    assert 0 in p['members']                        # the darkest key falls where a word ends


def test_typing_varies_every_key_and_keeps_the_deep_key_for_the_pauses():
    p = perform.plan('type', dsp.rng(5), members=(0, 1, 2, 3), events=16, deep=4, spread=2.0)
    keys = p['members']
    assert all(a != b for a, b in zip(keys, keys[1:]))
    assert 4 in keys and len(set(keys)) == 5
    for m in set(keys) - {4}:                       # the same key never comes back with the same variation
        seen = [(round(n['size'], 3), round(n['bright'], 3)) for k, n in zip(keys, p['nudge']) if k == m]
        assert all(a != b for a, b in zip(seen, seen[1:]))


def test_a_keyboard_with_two_recordings_borrows_keys():
    _, meta = learned.perform('keyboard/mechanical', seed=0, events=14)
    rec = meta['performance']['recordings']
    assert len(set(rec)) >= 4
    assert all(a != b for a, b in zip(rec, rec[1:]))
    assert any('space' in str(k) for k in rec)


def test_a_scatter_settles():
    p = perform.plan('scatter', dsp.rng(4), members=(0, 1, 2), events=6)
    gaps = np.diff(p['onsets'])
    assert gaps[-1] < gaps[0]
    assert p['gains'][-1] < p['gains'][0]


def test_mix_places_each_onset_and_keeps_every_tail():
    ev = np.zeros((SR // 2, 2))
    ev[dsp.n_of(0.01):, :] = 0.1
    p = {'onsets': [0.02, 0.3], 'gains': [1.0, 1.0]}
    y = perform.mix([(ev, 0.01), (ev, 0.01)], p)
    assert y.shape == (dsp.n_of(0.3) - dsp.n_of(0.01) + SR // 2, 2)      # the second onset 0.3 s in
    assert abs(y[dsp.n_of(0.31) + 10, 0] - 0.2) < 1e-9     # the first tail rings on under the second event


PLAYED = ('door/knock', 'step/pebbles', 'glass/cheers')


@pytest.mark.parametrize('sid', PLAYED)
def test_a_performance_renders_clean_at_full_peak(sid):
    y, meta = learned.perform(sid, seed=0)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    assert abs(meta['peak_dbfs'] + 1.0) < 0.3
    assert np.abs(y[:2]).max() < 0.02 and np.abs(y[-8:]).max() < 1e-3          # from zero, faded out
    assert meta['checks']['ok'], meta['checks']['warnings']
    p = meta['performance']
    assert p['events'] >= 2
    if len(learned.order(sid)) > 1 and p.get('pattern'):
        assert all(a != b for a, b in zip(p['recordings'], p['recordings'][1:]))


def test_a_performance_is_deterministic_and_seeds_differ():
    a, ma = learned.perform('door/knock', seed=1, events=3)
    b, _ = learned.perform('door/knock', seed=1, events=3)
    c, _ = learned.perform('door/knock', seed=2, events=3)
    assert np.array_equal(a, b)
    assert ma['performance']['events'] == 3
    assert len(a) != len(c) or not np.allclose(a, c)


def test_a_performance_is_longer_than_one_event():
    one, _ = learned.perform('door/knock', seed=0, events=1)
    four, _ = learned.perform('door/knock', seed=0, events=4)
    assert len(four) > len(one) + dsp.n_of(0.4)


def test_a_length_is_filled_at_the_natural_pace():
    y, meta = learned.perform('door/knock', seed=0, dur=1.5)
    assert meta['performance']['events'] >= 5
    assert master.true_peak_db(y) < -0.5


def test_a_plain_take_never_plays_a_recording_of_the_whole_action():
    for sid, e in learned.index().items():
        whole = set(learned.wholes(e))
        if whole and len(whole) < len(e['members']):
            assert not whole & set(learned.order(sid))


@pytest.mark.parametrize('text, sid, want', [
    ('a footstep on gravel', 'step/pebbles', (False, None)),
    ('footsteps on gravel', 'step/pebbles', (True, None)),
    ('someone walking on gravel', 'step/pebbles', (True, None)),
    ('two steps on a wooden floor', 'step/floorboard', (True, 2)),
    ('a single footstep on concrete', 'step/concrete', (False, 1)),
    ('a knock', 'door/knock', (False, None)),
    ('a knock on the door', 'door/knock', (True, None)),
    ('knocking on a wooden door', 'door/knock', (True, None)),
    ('three knocks on a door', 'door/knock', (True, 3)),
    ('a single knock on the door', 'door/knock', (False, 1)),
    ('a key press', 'keyboard/mechanical', (False, None)),
    ('typing on a mechanical keyboard', 'keyboard/mechanical', (True, None)),
    ('glasses clinking', 'glass/cheers', (True, None)),
    ('a light switch', 'switch/wall', (False, None)),
    ('a light switch flipped on and off', 'switch/wall', (True, None)),
    ('5 seconds of footsteps on wood', 'step/floorboard', (True, None)),
    ('an 8 bit footstep', 'step/concrete', (False, None)),
])
def test_the_request_decides_between_one_event_and_the_action(text, sid, want):
    assert performance(brief.read(text), sid) == want


def test_the_count_argument_overrides_the_words():
    assert performance(brief.read('footsteps on gravel'), 'step/pebbles', events=1) == (False, 1)
    assert performance(brief.read('a footstep on gravel'), 'step/pebbles', events=4) == (True, 4)
    assert performance(brief.read('a whoosh'), 'whoosh/stick') is None


def test_a_row_is_a_little_uneven_and_a_countdown_keeps_the_clock():
    row = perform.plan('row', dsp.rng(5), events=4)
    gaps = np.diff(row['onsets'])
    assert len(row['onsets']) == 4 and 1 / 3.5 * 0.85 < gaps.min() and gaps.max() < 1 / 2.5 * 1.15
    assert np.ptp(gaps) > 0.001                                              # off the grid
    count = perform.plan('count', dsp.rng(5), events=3)
    assert np.allclose(np.diff(count['onsets']), 1.0)                       # one a second, on the clock
