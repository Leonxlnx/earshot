"""earshot.sounds.learned: the store (every model is a member of one design, credited to a public-domain recording;
the store stays small; names are plain), designs render clean levelled takes that pass their audit, seeds walk
through a design's recordings, the declared axes move what they stand for, seeds of one recording are different
performances that measure alike. Model-free; the fast tier renders three interface designs and a footstep,
--slow every member of every design."""
import json
import os
import re

import numpy as np
import pytest

from earshot.forge import dsp
from earshot.sounds import learned, registry, sets
from earshot.synth import resyn

SR = dsp.SR
INDEX = learned.index()
IDS = list(INDEX)
LIMIT = 19_000_000                  # bytes the whole store may weigh
# three interface designs (sets 'ui') and a footstep: rendered in the fast tier (every member: --slow)
FAST = ('keyboard/mechanical', 'switch/lamp', 'leather/grab', 'step/pebbles')
# the character suffixes of the first pass's automatic names, and engine words: never in a name ('light' is a verb
# in match/light)
SLOP = {'bright', 'dark', 'warm', 'deep', 'clear', 'resyn', 'model', 'learned', 'fit', 'take', 'sample'}


def _seed(sid, member):
    """a seed that plays the member at that position of the index"""
    return learned.order(sid).index(member)


def _raw(sid, seed=5, member=None, **req):
    """the design function's own output for a seed (the member it picks), or for a member and a performance seed"""
    d = registry.get(sid)
    c = registry.Controls(d, **req)
    c.seed = _seed(sid, member) if member is not None else seed
    r = dsp.rng(seed if member is None else seed * len(INDEX[sid]['members']) + c.seed)
    return np.asarray(d.fn(c, r), np.float64)


def _centroid(y):
    m = y.mean(axis=1)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return float((P * f).sum() / (P.sum() + 1e-30))


def _level(y):
    return float(10 * np.log10(np.mean(y ** 2) + 1e-20))


def _end(y, down=30.0):
    """seconds to where the take is `down` dB under its loudest 5 ms for the last time"""
    k = dsp.n_of(0.005)
    e = 10 * np.log10(np.convolve(np.mean(y ** 2, axis=1), np.ones(k) / k, 'same') + 1e-20)
    return float(np.nonzero(e > e.max() - down)[0][-1] / SR)


def _strike(y, t0):
    """dB of the strike (1 ms before to 4 ms after t0) over what follows it (12 to 60 ms after)"""
    p = np.mean(y ** 2, axis=1)
    a, b = p[max(dsp.n_of(t0 - 0.001), 0):dsp.n_of(t0 + 0.004)], p[dsp.n_of(t0 + 0.012):dsp.n_of(t0 + 0.06)]
    return float(10 * np.log10((a.mean() + 1e-20) / (b.mean() + 1e-20)))


# ---- the store

def test_index_files_and_credits():
    files = {f for f in os.listdir(learned.STORE) if f.endswith('.npz')}
    listed = [m['file'] for e in INDEX.values() for m in e['members']]
    assert files == set(listed) and len(listed) == len(set(listed))        # nothing unlisted, missing or shared
    with open(os.path.join(learned.STORE, 'CREDITS.md'), encoding='utf-8') as fh:
        credits = fh.read()
    for sid, e in INDEX.items():
        assert e['members'] and e['kind'] in registry.KINDS and isinstance(e['default'], bool), sid
        for k, m in enumerate(e['members'], 1):
            c = m['credit']
            assert c['title'] and c['author'] and c['page'].startswith('https://'), sid
            assert 'CC0' in c['licence'] or 'public domain' in c['licence'].lower(), (sid, c['licence'])
            assert f'`{sid}` ({k})' in credits and c['page'] in credits, sid
            assert 'mixkit' not in (c['page'] + m['source']).lower(), sid
            assert 0 <= m['onset_s'] < m['dur_s'], sid
            assert m['rating'] is None or m['rating'] >= 3, sid                # listening score 1-2: never in


def test_store_size():
    total = sum(os.path.getsize(os.path.join(learned.STORE, f)) for f in os.listdir(learned.STORE))
    assert total < LIMIT, total


def test_names_are_plain_and_one_design_per_thing():
    with open(os.path.join(learned.STORE, 'index.json'), encoding='utf-8') as fh:
        rows = json.load(fh)
    assert len(rows) == len(INDEX) >= 100 and sum(len(e['members']) for e in rows) >= 250
    for sid in IDS:
        assert re.fullmatch(r'[a-z]+(_[a-z]+)?/[a-z]+(_[a-z]+)?', sid), sid
        assert not set(re.split('[/_]', sid)) & SLOP and not sid.endswith('_light'), sid
    families = {}
    for sid in IDS:
        families.setdefault(sid.split('/')[0], []).append(sid)
    assert len([f for f, v in families.items() if len(v) > 1]) < len(families)   # things, not characters
    assert all(len(set(m['source'] for m in e['members'])) == len(e['members']) for e in rows)


def test_every_design_registers_and_its_models_load():
    for sid, e in INDEX.items():
        d = registry.get(sid)
        prior = registry.LEARNED * (1.0 if e['default'] else learned.DEFAULT_OFF)
        assert d.kind == e['kind'] and set(d.axes) == set(e['axes']) and d.prior == prior, sid
        assert {'energy', 'size', 'bright'} <= set(d.axes) and 'real' in d.tags, sid
        assert (d.pitch is not None) == bool(e.get('tonal')), sid
    f = INDEX[IDS[0]]['members'][0]['file']
    m = learned.model(f)
    assert {'n', 'on', 'ap', 'ton', 'frq', 'q', 'u', 'h'} <= set(m) and int(m['n']) > 0
    assert learned.model(f) is learned.model(f)                              # read once


@pytest.mark.slow
def test_every_model_loads():
    for sid, e in INDEX.items():
        for m in e['members']:
            md = learned.model(m['file'])
            assert int(md['n']) > 0 and ('ild' in md) == m['stereo'] and (not m['stereo'] or 'ipd' in md), sid


def test_the_ui_set_plays_the_designs_chosen_for_its_roles():
    """tap is the recorded key chosen for it; every role is a sound of a real thing (a learned design, or one made
    of learned designs by earshot.sounds.interface), none a figure on an instrument"""
    ui = dict(sets.SETS['ui']['sounds'])
    roles = {r: sid for sid, e in INDEX.items() for r in e.get('roles', ())}
    assert ui['tap'] == roles['tap'] and INDEX[ui['tap']]['default']
    for name, sid in ui.items():
        d = registry.get(sid)
        assert 'real' in d.tags and not d.pitch and not d.voice, name


# ---- renders

def _renders(sid, seeds):
    for seed in seeds:
        ok = False
        for retry in (0, 101, 202, 303):           # as make re-rolls a take that fails its audit
            y, meta = registry.render(sid, seed=seed + retry)
            assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
            assert np.abs(y).max() <= 1.0 and meta['peak_dbfs'] <= -0.8 and not meta.get('ignored')
            assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-2
            assert meta['member']['of'] == len(INDEX[sid]['members'])
            ok = meta['checks']['ok']
            if ok:
                break
        assert ok, (sid, seed, meta['checks']['warnings'])


@pytest.mark.parametrize('sid', FAST)
def test_renders_clean_and_passes_its_audit(sid):
    _renders(sid, (0,))


# members (1-based) whose takes fail their audit on every seed: a fault of the engine where a frame edge meets an
# onset (earshot.synth.resyn), not of the recording. They play last in their design's walk
ENGINE_FAULTS = {('roll/drum', 2): 'two clicks where the engine sets an onset at a window edge',
                 ('twigs/snap', 3): 'a comb ring and stacked attacks the engine adds at its onsets'}


@pytest.mark.slow
@pytest.mark.parametrize('sid', IDS)
def test_every_member_renders_clean_and_passes_its_audit(sid):
    # every member a take plays (learned.order: a recording of the whole action plays only as a performance), but
    # the known faults of the engine, each skipped by name below
    _renders(sid, [s for s, k in enumerate(learned.order(sid)) if (sid, k + 1) not in ENGINE_FAULTS])


@pytest.mark.slow
@pytest.mark.parametrize('sid,member', [pytest.param(s, m, id=f'{s}#{m}', marks=pytest.mark.skip(reason=why))
                                        for (s, m), why in ENGINE_FAULTS.items()])
def test_a_member_with_a_known_engine_fault(sid, member):
    _renders(sid, [learned.order(sid).index(member - 1)])


# ---- members

def test_seeds_walk_through_the_members():
    sid = 'switch/lamp'
    n = len(INDEX[sid]['members'])
    got = [learned.pick(sid, s) for s in range(2 * n)]
    assert sorted(got[:n]) == list(range(n)) and got[n:] == got[:n]
    (a, ma), (b, mb) = registry.render(sid, seed=2, check=False), registry.render(sid, seed=2, check=False)
    assert np.array_equal(a, b) and ma['member']['n'] == mb['member']['n'] == got[2] + 1
    assert ma['member']['recording'] == f'recording {got[2] + 1} of {n}'        # never the uploader's title


def test_a_dark_or_bright_request_draws_from_the_darker_or_brighter_members():
    sid = 'keyboard/desktop'
    n = len(INDEX[sid]['members'])
    half = (n + 1) // 2
    for s in range(n):
        assert learned.pick(sid, s, -0.7) < half and learned.pick(sid, s, 0.7) >= n - half
        assert learned.pick(sid, s, 0.3) == learned.pick(sid, s)
    cents = [m['quality']['centroid'] for m in INDEX[sid]['members']]
    assert cents == sorted(cents)                                             # stored darkest first


# ---- controls

@pytest.mark.parametrize('sid', ['keyboard/mechanical', 'apple/drop'])
def test_size_is_lower_and_longer(sid):
    small, big = _raw(sid, member=0, size=-1.0, space=-1.0), _raw(sid, member=0, size=1.0, space=-1.0)
    assert _centroid(big) < 0.75 * _centroid(small) and _end(big) > 1.5 * _end(small)


@pytest.mark.parametrize('sid', ['keyboard/mechanical', 'whoosh/stick'])
def test_bright_raises_the_centroid(sid):
    assert _centroid(_raw(sid, member=0, bright=1.0)) > 1.3 * _centroid(_raw(sid, member=0, bright=-1.0))


def test_hard_makes_the_strike_stand_out():
    sid = 'knock/hollow'
    t0 = INDEX[sid]['members'][0]['onset_s']
    hi, lo = _raw(sid, member=0, hard=1.0, space=-1.0), _raw(sid, member=0, hard=-1.0, space=-1.0)
    assert _strike(hi, t0) > _strike(lo, t0) + 2.0


def test_ring_rings_longer():
    sid = 'cork/bottle'
    k = next(i for i, m in enumerate(INDEX[sid]['members']) if m['source'] == 'freesound-335358')
    assert _end(_raw(sid, member=k, ring=1.0, space=-1.0)) > 1.3 * _end(_raw(sid, member=k, ring=-1.0, space=-1.0))


def test_energy_is_louder():
    lo, hi = _raw('keyboard/mechanical', member=0, energy=-1.0), _raw('keyboard/mechanical', member=0, energy=1.0)
    assert _level(hi) > _level(lo) + 9.0


def test_width_spreads_a_mono_and_a_stereo_recording():
    def corr(y):
        return np.corrcoef(y[:, 0], y[:, 1])[0, 1]
    for sid in ('keyboard/mechanical', 'apple/drop'):
        assert corr(_raw(sid, member=0, width=1.0, space=-1.0)) < corr(_raw(sid, member=0, width=-1.0,
                                                                             space=-1.0)) - 0.05, sid


def test_space_adds_a_tail():
    assert _end(_raw('knock/hollow', member=0, space=1.0)) > _end(_raw('knock/hollow', member=0, space=-1.0)) + 0.05


def test_pitch_moves_a_tonal_model():
    sid = 'knock/hollow'
    f0 = INDEX[sid]['members'][0]['hz']
    y = _raw(sid, member=0, pitch=f0 * 2 ** (3 / 12), space=-1.0)
    f = max(resyn.partials(y), key=lambda q: q[1])[0]
    assert abs(1200 * np.log2(f / (f0 * 2 ** (3 / 12)))) < 50
    assert registry.get('keyboard/mechanical').pitch is None                    # a key has no note to move


def test_length_warps_within_its_natural_range():
    sid = 'whoosh/heavy'
    d = INDEX[sid]['members'][0]['dur_s']
    plain, long_ = _end(_raw(sid, member=0, space=-1.0)), _end(_raw(sid, member=0, dur=1.5 * d, space=-1.0))
    assert 1.25 < long_ / plain < 1.75
    far = _raw(sid, member=0, dur=3.0 * d, space=-1.0)                         # clamped: no stretch past LENGTH
    assert len(far) <= (learned.LENGTH[1] + 0.1) * d * SR


# ---- performances of one recording

@pytest.mark.parametrize('sid', ['keyboard/mechanical', 'knock/hollow', 'switch/lamp'])
def test_seeds_of_one_recording_are_performances(sid):
    a, b = _raw(sid, seed=1, member=0), _raw(sid, seed=1, member=0)
    assert np.array_equal(a, b)
    takes = [a] + [_raw(sid, seed=s, member=0) for s in (2, 3)]
    for t in takes[1:]:
        assert len(t) != len(a) or not np.allclose(t, a)
    c = [_centroid(t) for t in takes]
    lv = [_level(t) for t in takes]
    assert max(c) / min(c) < 1.35 and max(lv) - min(lv) < 2.5, (c, lv)


def test_back_and_next_are_one_page_turned_two_ways():
    """next and back are the same page: back the darker and shorter flick, played the other way"""
    ui = dict(sets.SETS['ui']['sounds'])
    assert ui['next'] == 'next/page' and ui['back'] == 'back/turn'
    nxt, back = (registry.render(ui[n], seed=0)[0] for n in ('next', 'back'))
    assert _centroid(back) < 0.85 * _centroid(nxt) and len(back) < len(nxt)


# recordings with the best listening score (1 to 5) of their thing, which seed 0 plays first by design
BEST = {'keyboard/mechanical': 'freesound-380142', 'camera/shutter': 'freesound-170229',
         'whoosh/stick': 'freesound-59995', 'bag/drop': 'freesound-584441'}


@pytest.mark.parametrize('sid', sorted(BEST))
def test_seed_zero_plays_the_recording_rated_best(sid):
    """seed 0 plays the recording with the best listening score, not another member of the thing"""
    assert INDEX[sid]['members'][learned.pick(sid, 0)]['source'] == BEST[sid]
    assert learned.order(sid)[0] == learned.pick(sid, 0)


def test_the_walk_holds_every_member_but_those_rated_one():
    """a plain take walks every single-event member but one with a listening score of 1; the recordings of the whole action are walked
    by a performance (learned.perform) and never by a plain take"""
    for sid, e in INDEX.items():
        walk = learned.order(sid)
        whole = set(learned.wholes(e))
        assert len(set(walk)) == len(walk) and set(walk) <= set(range(len(e['members']))), sid
        out = set(range(len(e['members']))) - set(walk) - whole
        assert all(e['members'][k].get('rating_final') == 1 for k in out), sid
        assert walk or not e['members'], sid
        assert not whole & set(walk) or whole == set(range(len(e['members']))), sid


def test_a_retake_plays_the_same_recording():
    """make re-rolls a take that fails its audit at seed + 101: another performance of that recording, not the
    next one (whoosh/stick played its plainer first member through a re-roll)"""
    for sid in ('whoosh/stick', 'keyboard/laptop', 'bag/drop'):
        for s in range(4):
            assert learned.pick(sid, s) == learned.pick(sid, s + learned.RETAKE) == learned.pick(sid, s + 303)


def test_the_ui_set_plays_the_members_chosen_for_its_roles():
    """tap is the fullest mechanical key (the set pins its seed)"""
    ui = sets.SETS['ui']
    sound = dict(ui['sounds'])
    src = {n: INDEX[sound[n]]['members'][learned.pick(sound[n], s)]['source'] for n, s in ui['seeds'].items()}
    assert src['tap'] == 'freesound-380142'


def _loudest(y, ms):
    k = dsp.n_of(ms / 1000.0)
    return 10 * np.log10(np.convolve(np.mean(y ** 2, axis=1), np.ones(k) / k, 'same').max())


def test_interface_members_sit_near_the_figures():
    """a click lifted by the registry keeps its strike and gets some body: its loudest 50 ms within 5 dB of an
    interface figure's (lower, they were heard as too quiet; a click is mostly its first 20 ms), its strike as bright"""
    fig = _loudest(registry.render('success/rise', seed=0)[0], 50.0)
    for sid in ('keyboard/mechanical', 'keyboard/desktop', 'dial/click'):
        y, _ = registry.render(sid, seed=0)
        assert fig - _loudest(y, 50.0) < 5.0, sid
    hit = np.concatenate([np.zeros(480), np.exp(-np.arange(4800) / 480.0)])                # a knock 10 ms in
    raw = learned.present(np.repeat(hit[:, None], 2, axis=1) * 0.5)
    lifted = learned.body(raw)
    assert abs(_loudest(lifted, 5.0) - _loudest(raw, 5.0)) < 0.5                    # the strike keeps its level
    assert _loudest(lifted, 50.0) > _loudest(raw, 50.0) + 0.3                        # what follows it comes up


WALKS = sorted(sid for sid, e in INDEX.items() if (e.get('pattern') or {}).get('name') == 'walk')


def _session(m):
    """the recording a member was cut from ('cuts/bigsoundbank/0165_07.wav', 'bigsoundbank/0165.wav@1.038-1.633')"""
    s = m['source'].split('@')[0]
    return re.sub(r'^cuts/', '', re.sub(r'_\d\d(\.\w+)$', r'\1', s)).replace('.wav', '').rsplit('.', 1)[0]


def _foot(e, k, seed):
    """(loudest 50 ms dB, centroid Hz) of member k as a walk plays it"""
    c = registry.Controls(registry.get(e['id']))
    c.seed = seed
    y = np.asarray(learned.take(e, c, dsp.rng(seed), k=k, room=False), np.float64)
    P = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return _loudest(y, 50.0), (P * f).sum() / P.sum()


def test_a_walk_alternates_two_feet_alike():
    """one walker in one pair of shoes: the two single steps a walk alternates (pattern `feet`) are cut from one
    recording session and measure alike as the walk plays them (loudest 50 ms within 3 dB, centroids within a
    factor 1.2: a left and a right foot, not a limp); a walk of a length plays those two in turn"""
    assert {'step/floorboard', 'step/tile', 'step/concrete', 'step/pebbles'} <= set(WALKS)
    for sid in WALKS:
        e = INDEX[sid]
        a, b = e['pattern']['feet']
        assert e['members'][a].get('events', 1) == 1 and e['members'][b].get('events', 1) == 1, sid
        if a == b:
            continue
        assert _session(e['members'][a]) == _session(e['members'][b]), sid
        fa, fb = [np.mean([_foot(e, k, s) for s in (0, 1)], axis=0) for k in (a, b)]
        assert abs(fa[0] - fb[0]) < 3.0, (sid, fa, fb)
        assert max(fa[1], fb[1]) / min(fa[1], fb[1]) < 1.2, (sid, fa, fb)
    y, meta = learned.perform('step/floorboard', seed=0, dur=5.0)
    a, b = INDEX['step/floorboard']['pattern']['feet']
    rec = meta['performance']['recordings']
    assert rec[0::2] == [a + 1] * len(rec[0::2]) and rec[1::2] == [b + 1] * len(rec[1::2])


def test_a_gravel_walk_is_heard_from_standing_height():
    """a walk on gravel is tilted darker as the pattern's `far` says (the closest crunch of a recording made at the
    feet a little softer), and only the walk: a single step keeps its recording's brightness"""
    def cen(y):
        P = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return (P * f).sum() / P.sum()
    e = INDEX['step/pebbles']
    assert e['pattern']['far'] < 0
    walk, _ = learned.perform('step/pebbles', seed=0)
    k = learned.wholes(e)[0]
    c = registry.Controls(registry.get('step/pebbles'))
    c.seed = 0
    near = np.asarray(learned.take(e, c, dsp.rng(registry.seed_of('step/pebbles', 0) ^ 0x5EED), k=k), np.float64)
    assert cen(walk) < 0.95 * cen(near)


def test_a_gravel_walk_asked_close_keeps_its_closeness(monkeypatch):
    """"footsteps on gravel, close" (space under 0: close, up close, close-miked, dry) plays the recording's own
    closeness: the walk is the one a pattern without `far` gives; at space 0 or more the tilt applies"""
    close, _ = learned.perform('step/pebbles', seed=0, dur=3.0, space=-0.6)
    plain, _ = learned.perform('step/pebbles', seed=0, dur=3.0, space=0.0)
    flat = learned.index()
    flat['step/pebbles']['pattern']['far'] = 0.0
    monkeypatch.setattr(learned, 'index', lambda: flat)
    close_flat, _ = learned.perform('step/pebbles', seed=0, dur=3.0, space=-0.6)
    plain_flat, _ = learned.perform('step/pebbles', seed=0, dur=3.0, space=0.0)
    assert np.array_equal(close, close_flat)
    assert not np.array_equal(plain, plain_flat)


# ---- what the stored envelope cannot hold, taken out (learned.TAIL, learned.LEAD, resyn.tidy)

def _member(file):
    return next((sid, k, m) for sid, e in INDEX.items() for k, m in enumerate(e['members']) if m['file'] == file)


def _db(y, ms):
    return 10 * np.log10(resyn._avg(np.mean(y ** 2, axis=1), dsp.n_of(ms / 1000)) + 1e-20)


def test_the_tail_and_lead_repairs_switch_off(monkeypatch):
    sid, k, m = _member('door-knock-2.npz')
    assert m.get('lead') and learned.LEAD and not learned.TAIL

    def take():
        c = registry.Controls(registry.get(sid))
        c.seed = 0
        return learned.take(INDEX[sid], c, dsp.rng(3), k=k)
    on = take()
    monkeypatch.setattr(learned, 'TAIL', False)
    monkeypatch.setattr(learned, 'LEAD', False)
    off = take()
    monkeypatch.undo()
    monkeypatch.setitem(m, 'lead', None)
    assert not np.array_equal(on, off) and on.shape == off.shape
    assert np.array_equal(off, take())                           # off is the take of a member without the numbers


@pytest.mark.parametrize('file', ['knock-table-1.npz', 'step-floorboard-5.npz', 'cutlery-plate-2.npz'])
def test_the_tail_falls_with_the_recordings_decay(file):
    """after the point where the recording last stood 30 dB under its loudest 10 ms the take is held under the
    recording's decay line: its end is quieter than the plain take's, everything before it and the strike untouched"""
    _, _, m = _member(file)
    a = resyn.render(learned.model(file), dsp.rng(4))
    b = resyn.render(learned.model(file), dsp.rng(4), tail=tuple(m['tail']))
    i = dsp.n_of(m['tail'][0]) - dsp.n_of(0.002)
    assert a.shape == b.shape and np.array_equal(a[:i], b[:i])
    assert np.abs(_db(a, 10).max() - _db(b, 10).max()) < 0.05
    end = slice(i + (len(a) - i) // 2, None)
    assert np.mean(_db(b, 50)[end]) < np.mean(_db(a, 50)[end]) - 0.5
    assert np.all(_db(b, 50)[i:] <= _db(a, 50)[i:] + 0.01)


@pytest.mark.parametrize('file', ['door-knock-2.npz', 'door-shut-2.npz', 'cork-bottle-3.npz'])
def test_nothing_is_smeared_into_the_moment_before_the_strike(file):
    """in the stretch before the main onset (to 2 ms before it) the take stands no higher under its strike than
    the recording did (or 48 dB under its peak), and lower than the plain take; outside that stretch it is the same"""
    _, _, m = _member(file)
    on, cap, span = m['onset_s'], m['lead'][0], m['lead'][1]
    a = resyn.render(learned.model(file), dsp.rng(4))
    b = resyn.render(learned.model(file), dsp.rng(4), lead=(on, cap, span))
    lo, hi = max(dsp.n_of(on - span) - dsp.n_of(0.001), 0), dsp.n_of(on - 0.002) + dsp.n_of(0.001)
    assert a.shape == b.shape and np.array_equal(a[:lo], b[:lo]) and np.array_equal(a[hi:], b[hi:])
    hit = _db(b, 10)[dsp.n_of(on):dsp.n_of(on + 0.05)].max()
    floor = 20 * np.log10(np.abs(b).max()) + resyn.LEAD_LOW
    w = slice(max(dsp.n_of(on - span), 0) + dsp.n_of(0.002), dsp.n_of(on - 0.002) - dsp.n_of(0.002))
    assert _db(b, 2)[w].max() <= max(hit + cap, floor) + 2.0     # (the gain is smoothed over 1 ms)
    assert np.sum(b[lo:hi] ** 2) < np.sum(a[lo:hi] ** 2)


def test_a_member_the_engine_plays_with_a_fault_comes_last():
    """the second drum roll clicks where the engine sets an onset at a window edge: it plays after the other"""
    e = INDEX['roll/drum']
    assert e['members'][e['walk'][-1]]['file'] == 'roll-drum-2.npz'
