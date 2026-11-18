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
    assert meta['performance']['events'] >= 5 and meta['dur_s'] == pytest.approx(1.5, abs=0.001)
    assert master.true_peak_db(y) < -0.5 and np.abs(y[-1]).max() < 1e-4


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


def test_a_length_spreads_the_events_and_holds_them_where_a_hand_must():
    """the first event at LEAD, the last `ring` before the end, the human spread kept; no count: the pace's;
    gaps held at the fastest and the widest a person keeps, with the line that says so; a countdown on the clock"""
    p = perform.plan('knock', dsp.rng(3), events=3, dur=2.0, ring=0.3)
    on = np.array(p['onsets'])
    assert p['fits'] and p['dur'] == 2.0 and on[0] == perform.LEAD and on[-1] == pytest.approx(1.7)
    assert np.ptp(np.diff(on)) > 0.001 and perform.said(p, 2.0, 2.0) is None
    walk = perform.plan('walk', dsp.rng(3), dur=5.0, ring=0.4)
    assert 8 <= len(walk['onsets']) <= 10 and walk['onsets'][-1] == pytest.approx(4.6)
    near = perform.plan('knock', dsp.rng(3), events=10, dur=0.5, ring=0.3)
    assert not near['fits'] and near['dur'] is None and near['gap'] == 0.12
    assert perform.said(near, 0.5, 1.4) == ('10 knocks 0.12 s apart: 1.40 s, not the 0.5 s asked (knocks cannot '
                                            'follow closer than 0.12 s by hand)')
    far = perform.plan('row', dsp.rng(3), events=3, dur=10.0, ring=0.3)
    assert not far['fits'] and far['gap'] == 1.5 and 'claps further apart than 1.5 s' in perform.said(
        far, 10.0, 3.3, 'clap')
    count = perform.plan('count', dsp.rng(3), events=3, dur=3.0, ring=0.22)
    assert count['fits'] and np.allclose(np.diff(count['onsets']), 1.0)
    assert not perform.plan('count', dsp.rng(3), events=3, dur=6.0, ring=0.22)['fits']
    assert len(perform.plan('count', dsp.rng(3), dur=5.0, ring=0.22)['onsets']) == 5
    y = np.ones((dsp.n_of(1.0), 2))
    assert len(perform.fit(y, 0.5)) == dsp.n_of(0.5) and perform.fit(y, 0.5)[-1, 0] < 1e-12
    assert len(perform.fit(y, 2.0)) == dsp.n_of(2.0) and perform.fit(y, 2.0)[-1, 0] == 0.0


def test_a_length_reshapes_a_scatter_and_a_shake_without_evening_them_out():
    """the last onset `ring` s before the end; a longer settle takes more impacts and still shortens its gaps and
    falls in level; a shorter one loses its last bits; a longer shake swings more often at its own pace"""
    for dur, more in ((3.0, True), (0.6, False)):
        p = perform.plan('scatter', dsp.rng(4), members=(0, 1, 2), dur=dur, ring=0.4)
        on, own = np.array(p['onsets']), perform.plan('scatter', dsp.rng(4), members=(0, 1, 2))
        assert p['fits'] and p['dur'] == dur and on[-1] == pytest.approx(dur - 0.4)
        assert (len(on) > len(own['onsets'])) == more
        gaps = np.diff(on)
        k = max(1, len(gaps) // 3)
        assert np.mean(gaps[-k:]) < np.mean(gaps[:k]) and p['gains'][-1] < p['gains'][0]
    n, ratio = perform.settle(5, 0.15, 0.7, 3.0)
    assert n > 5 and 0.7 < ratio < 1.0
    assert perform.settle(5, 0.15, 0.7, 3.0, events=5)[0] == 5
    s = perform.plan('shake', dsp.rng(2), members=(0, 1), dur=3.0, ring=0.5)
    assert s['onsets'][-1] == pytest.approx(2.5) and 10.0 <= s['pace'] <= 20.0 * 1.2
    assert len(s['onsets']) > perform.PATTERNS['shake']['events'][1]


def test_a_recording_of_the_whole_action_plays_the_stretch_a_length_asks_for():
    """a thing with no pattern (a bunch of keys): the busiest stretch of a recording, unwarped and faded out; a
    length over every recording plays the longest whole and says so"""
    y, meta = learned.perform('keys/bunch', seed=0, dur=1.0)
    assert meta['dur_s'] == pytest.approx(1.0, abs=0.001) and 'busiest stretch' in meta['note']
    assert np.abs(y[:2]).max() < 0.02 and np.abs(y[-8:]).max() < 1e-3 and meta['checks']['ok']
    y, meta = learned.perform('coins/floor', seed=0, dur=3.0)
    assert meta['dur_s'] < 2.5 and 'not the 3 s asked' in meta['note']
    s = learned.stretch(np.concatenate([np.zeros((SR, 2)), np.ones((SR // 2, 2)), np.zeros((SR, 2))]), 1.0)
    assert len(s) == SR and s[SR // 2 + 10, 0] == pytest.approx(1.0) and s[SR // 2 - 1, 0] == 0.0     # the earliest


def test_the_last_event_is_placed_to_ring_out_before_the_end():
    """a length asked for: an event whose measured ring passes the end draws the onsets together (the spread kept,
    the first where it was); none that passes leaves the plan as it is; a gap closer than a hand repeats drops the
    last events; an event that stops while still sounding is faded over its own end, one that died away is not"""
    p = perform.plan('walk', dsp.rng(4), members=(0, 1), dur=4.0, ring=0.4)
    n = len(p['onsets'])
    same, k = perform.close(p, [0.3] * n)
    assert same is p and k == n
    q, k = perform.close(p, [0.3] * (n - 1) + [1.2])
    assert k == n and q['onsets'][0] == p['onsets'][0]
    assert q['onsets'][-1] + 1.2 == pytest.approx(4.0 - perform.FADE)
    ratio = np.diff(q['onsets']) / np.diff(p['onsets'])
    assert np.allclose(ratio, ratio[0])                                    # the human spread kept
    q, k = perform.close(p, [3.0] * n)
    assert 2 <= k < n and len(q['gains']) == len(q['members']) == len(q['nudge']) == k
    assert q['onsets'][-1] + 3.0 <= 4.0 - perform.FADE + 1e-9
    assert perform.close({**p, 'dur': None}, [5.0] * n)[1] == n             # no length: nothing moves
    t = np.arange(SR) / SR
    ring = np.sin(2 * np.pi * 500 * t) * np.exp(-t / 0.05)                  # -40 dB at about 0.23 s
    assert perform.ring_of(ring) == pytest.approx(0.05 * 40 / 8.686, abs=0.01)
    assert perform.ring_of(ring, onset=0.1) == pytest.approx(perform.ring_of(ring) - 0.1, abs=1e-9)
    assert np.array_equal(perform.ease(ring), ring)
    cut = np.sin(2 * np.pi * 500 * t[:SR // 2])
    eased = perform.ease(cut)
    assert np.abs(eased[-100:]).max() < 1e-3 and np.array_equal(eased[:SR // 4], cut[:SR // 4])


@pytest.mark.parametrize('text,sid', [('footsteps on gravel', 'step/pebbles'), ('knocking on a door', 'door/knock')])
def test_a_length_lets_the_last_event_ring_out(text, sid):
    for dur in (2.0, 4.0):
        y, meta = learned.perform(sid, seed=0, dur=dur)
        assert meta['dur_s'] == pytest.approx(dur, abs=0.03) and meta['checks']['ok'], (text, dur, meta['checks'])


def test_a_row_no_hand_plays_follows_its_own_length():
    assert perform.by_hand('knock') and perform.by_hand(None, 'step/pebbles') and perform.by_hand('claps')
    assert not perform.by_hand('tick', 'tick/metal') and not perform.by_hand(None, 'ui_pop/blip')
    n, gap, fits = perform.fill('row', 12, 1.5, 3.0, 0.35, lo=0.03)
    assert n == 12 and fits and gap >= 0.03
    n, gap, fits = perform.fill('row', 15, 1.5, 3.0, 0.35)          # a hand: held at its limit
    assert not fits and gap == perform.GAPS['row'][0]


def test_a_short_length_takes_its_ring_from_the_last_event_before_it_holds_the_gaps():
    n, gap, fits = perform.fill('row', 3, 0.6, 3.0, 0.5)
    assert n == 3 and fits and gap >= perform.GAPS['row'][0]
    y = np.ones((dsp.n_of(0.5), 2))
    z = perform.taper(y, 0.2)
    assert len(z) == dsp.n_of(0.2) and abs(z[-1]).max() < 1e-3
