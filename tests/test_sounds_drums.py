"""The drums and percussion of the catalogue (earshot.sounds.drums): the store (credits for every member, the
budget), the designs (clean, deterministic, a seed is another recording, takes measure alike) and what a listener
expects of real drums (a kick is low, a hat is high and short, open rings longer than closed, energy is louder and
brighter, size is lower, a tuned drum follows the pitch asked for). Fast tier under 20 s; --slow renders every member
of every design."""
import os

import numpy as np
import pytest

from earshot.forge import dsp
from earshot.sounds import drums, registry

SR = dsp.SR
INDEX = drums.index()
IDS = sorted(INDEX)
BUDGET = 3_000_000
KEPT = ('kick/soft', 'kick/punch', 'kick/deep', 'kick/electro', 'snare/tight', 'snare/fat', 'snare/brush',
        'snare/electro', 'rim/shot', 'rim/click', 'tom/high', 'tom/mid', 'tom/floor', 'hat/closed', 'hat/open',
        'hat/pedal', 'hat/electro', 'crash/bright', 'crash/dark', 'ride/ping', 'clap/hands', 'clap/group',
        'clap/electro', 'snap/finger', 'shaker/tick', 'shaker/roll', 'tambourine/hit', 'tambourine/shake',
        'conga/open', 'bongo/high', 'cajon/bass', 'clave/click', 'sticks/count', 'drum/roll')
ADDED = ('conga/muted', 'bongo/low', 'darbuka/hit', 'frame_drum/deep', 'cabasa/hit', 'slapstick/crack')
REMOVED = ('ride/bell', 'cymbal/swell', 'conga/slap', 'cajon/slap')
# what the music beds play (earshot.sounds.beds.ACOUSTIC)
BEDS = ('kick/soft', 'snare/brush', 'rim/click', 'shaker/tick', 'clap/hands')


def _env_db(x, win=0.005):
    x = x.mean(1) if x.ndim == 2 else x
    w = int(win * SR)
    k = len(x) // w
    e = np.sqrt(np.mean(x[:k * w].reshape(k, w) ** 2, 1)) + 1e-12
    return 20 * np.log10(e / e.max())


def _t_down(x, db=-40.0):
    e = _env_db(x)
    i = int(np.argmax(e))
    below = np.nonzero(e[i:] < db)[0]
    return (i + below[0]) * 0.005 if len(below) else len(e) * 0.005


def _centroid(x, t0=0.0, t1=0.1):
    x = x.mean(1) if x.ndim == 2 else x
    s = x[int(t0 * SR):int(t1 * SR)]
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    f = np.fft.rfftfreq(len(s), 1 / SR)
    return float((S * f).sum() / S.sum())


def _raw(sid, seed=1, **axes):
    """a design's own output, before the registry levels it"""
    d = registry.get(sid)
    c = registry.Controls(d, **axes)
    c.seed = seed
    return d.fn(c, dsp.rng(registry.seed_of(sid, seed)))


# ------------------------------------------------------------------------------------------------------ the store
def test_the_store_is_within_its_budget_and_ships_no_audio():
    files = os.listdir(drums.STORE)
    assert sum(os.path.getsize(os.path.join(drums.STORE, f)) for f in files) <= BUDGET
    assert {os.path.splitext(f)[1] for f in files} <= {'.npz', '.json', '.md'}
    used = {m['file'] for e in INDEX.values() for m in e['members']}
    assert used == {f for f in files if f.endswith('.npz')}                   # nothing stale, nothing missing


def test_every_member_is_credited_and_public_domain():
    text = open(os.path.join(drums.STORE, 'CREDITS.md'), encoding='utf-8').read()
    for sid, e in INDEX.items():
        assert e['members'], sid
        for k, m in enumerate(e['members']):
            c = drums.credit(sid, k)
            assert c['title'] and c['author'] and c['page'].startswith('https://') and 'CC0' in c['licence'], sid
            assert c['title'] in text, (sid, c['title'])


def test_members_are_stored_softest_first_and_tuned_drums_have_a_note():
    for sid, e in INDEX.items():
        v = [m['vel'] for m in e['members']]
        assert v == sorted(v) and all(0.0 <= x <= 1.0 for x in v), sid
        assert (e['note'] is None) == (not any(m.get('hz') for m in e['members'])), sid
        assert all(e['note'] is None or m.get('hz') for m in e['members']), sid


def test_every_recording_is_played():
    """the seeds and the energies reach every member of a design (kick/punch plays the recordings of kick/soft at a
    harder velocity: its softest layers are kick/soft's); recordings of one velocity are equals"""
    files = {}
    for sid, e in INDEX.items():
        n = len(e['members'])
        got = {drums.pick(sid, s, en) for s in range(n) for en in np.linspace(-1.0, 1.0, 5)}
        if sid != 'kick/punch':
            assert got == set(range(n)), (sid, sorted(set(range(n)) - got))
        for k in got:
            files[e['members'][k]['file']] = sid
    assert set(files) == {m['file'] for e in INDEX.values() for m in e['members']}
    clap = INDEX['clap/group']
    assert len({m['vel'] for m in clap['members']}) == 1                     # six claps of one velocity
    assert {drums.pick('clap/group', s) for s in range(len(clap['members']))} == set(range(len(clap['members'])))


def test_the_catalogue_keeps_its_drums_and_names_what_left():
    ids = set(registry.ids())
    assert set(KEPT) | set(ADDED) <= ids
    assert not set(REMOVED) & ids
    for sid in IDS:
        d = registry.get(sid)
        assert 'real' in d.tags and d.axes == drums.AXES and (d.pitch is None) == (INDEX[sid]['note'] is None)


# ------------------------------------------------------------------------------------------------------ takes
_TAKES = {}


def _take(sid, seed=1, **request):
    """registry.render, each (id, seed, request) rendered once for the whole file"""
    key = (sid, seed, tuple(sorted(request.items())))
    if key not in _TAKES:
        _TAKES[key] = registry.render(sid, seed=seed, **request)
    return _TAKES[key]


def _raw_once(sid, seed=1, **axes):
    key = ('raw', sid, seed, tuple(sorted(axes.items())))
    if key not in _TAKES:
        _TAKES[key] = _raw(sid, seed, **axes)
    return _TAKES[key]


@pytest.mark.parametrize('sid', ['kick/soft', 'hat/closed'])
def test_designs_are_clean_and_deterministic(sid):
    y, meta = _take(sid, 1)
    y2, _ = registry.render(sid, seed=1)
    y3, _ = _take(sid, 2)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    assert np.array_equal(y, y2) and not np.array_equal(y, y3)
    assert np.abs(y).max() < 1.0
    assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-3
    assert meta['checks']['ok'], meta['checks']['warnings']
    assert meta['member']['of'] == len(INDEX[sid]['members']) and not meta.get('ignored')


def test_a_drummer_never_hits_twice_the_same_and_takes_measure_alike():
    """three seeds of a kick: three recordings, as loud as each other within 3 dB (loudest 50 ms)"""
    got = [_take('kick/soft', s) for s in range(3)]
    assert len({m['member']['n'] for _, m in got}) == 3
    lv = [drums.loudest(y) for y, _ in got]
    assert max(lv) - min(lv) < 3.0


def _band(y, lo, hi):
    return drums.loudest(dsp.butter(y, 'band', (lo, hi), order=2))


def test_energy_picks_harder_recordings_and_plays_them_louder_and_brighter():
    """a harder stroke is a harder recording, louder, and louder in the top (2-12 kHz) too. In these recordings a
    soft stroke is mostly the stick (its centroid is the higher), a hard one drives the whole drum: the body under
    500 Hz rises most"""
    sid = 'snare/tight'
    soft = [INDEX[sid]['members'][drums.pick(sid, s, -1.0)]['vel'] for s in range(4)]
    hard = [INDEX[sid]['members'][drums.pick(sid, s, 1.0)]['vel'] for s in range(4)]
    assert max(soft) < min(hard)
    lo, hi = _raw_once(sid, energy=-1.0, space=-1.0), _raw_once(sid, energy=1.0, space=-1.0)
    assert drums.loudest(hi) > drums.loudest(lo) + 6.0
    top = _band(hi, 2000.0, 12000.0) - _band(lo, 2000.0, 12000.0)
    assert top > 3.0 and _band(hi, 100.0, 500.0) - _band(lo, 100.0, 500.0) > top


def test_a_kick_is_low_and_a_hat_is_high_and_short():
    k, _ = _take('kick/soft', 0)
    h, _ = _take('hat/closed', 1, space=-1.0)
    assert _centroid(k) < 400.0 and _centroid(h) > 4000.0
    assert _t_down(h) < 0.5


def test_open_hat_rings_longer_than_closed():
    a, _ = _take('hat/closed', 1, space=-1.0)
    b, _ = _take('hat/open', 1, space=-1.0)
    assert _t_down(a) < 0.5 * _t_down(b)


def _f0(y, hi=400.0):
    s = y.mean(1)[int(0.03 * SR):int(0.5 * SR)]
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)), 1 << 17))
    f = np.fft.rfftfreq(1 << 17, 1 / SR)
    return f[np.argmax(S * (f > 40) * (f < hi))]


def test_size_lowers_a_drum_pitch_tunes_it_and_ring_draws_its_tail_out():
    a = _raw_once('conga/open', space=-1.0)
    assert _f0(_raw_once('conga/open', size=1.0, space=-1.0)) < 0.85 * _f0(a)
    hz = INDEX['conga/open']['members'][drums.pick('conga/open', 1)]['hz']
    c = _raw_once('conga/open', pitch=hz * 2 ** (3 / 12), space=-1.0)
    assert _f0(c) == pytest.approx(hz * 2 ** (3 / 12), rel=0.06)
    assert _t_down(_raw_once('conga/open', ring=1.0, space=-1.0), -30.0) > 1.4 * _t_down(a, -30.0)


def test_the_bed_kit_plays_real_drums_and_the_kick_follows_the_bed_tuning():
    for sid in BEDS:
        assert sid in registry.ids()
    assert 'real' in registry.get('kick/soft').tags and 'real' in registry.get('rim/click').tags
    hz = INDEX['kick/soft']['members'][drums.pick('kick/soft', 1)]['hz']           # the bed asks for D2 (73.4 Hz)
    a, b = _raw_once('kick/soft', space=-1.0), _raw_once('kick/soft', pitch='D2', space=-1.0)
    assert _ratio(a, b) == pytest.approx(73.4 / hz, rel=0.05)


def _ratio(a, b, lo=30.0, hi=2000.0):
    """the factor b's spectrum is a's moved by: the best match of their log-frequency spectra (first 0.5 s)"""
    grid = np.geomspace(lo, hi, 1200)
    spec = []
    for y in (a, b):
        s = y.mean(1)[:int(0.5 * SR)]
        S = np.abs(np.fft.rfft(s * np.hanning(len(s)), 1 << 17))
        spec.append(np.log(np.interp(grid, np.fft.rfftfreq(1 << 17, 1 / SR), S) + 1e-9))
    step = np.log2(hi / lo) / (len(grid) - 1)
    lags = np.arange(-int(1.0 / step), int(1.0 / step) + 1)
    k = max(lags, key=lambda g: np.corrcoef(spec[0][max(0, -g):len(grid) - max(0, g)],
                                            spec[1][max(0, g):len(grid) - max(0, -g)])[0, 1])
    return float(2 ** (k * step))


def test_a_roll_is_as_long_as_asked_and_ends_on_its_accent():
    y, meta = _take('drum/roll', 0, dur=1.2)
    assert meta['checks']['ok'], meta['checks']['warnings']
    e = _env_db(y, 0.05)
    assert 1.1 < len(y) / SR < 3.5 and int(np.argmax(e)) * 0.05 > 1.0         # the loudest moment is the last hit
    s, _ = _take('shaker/roll', 0, dur=1.0)
    assert 0.9 < len(s) / SR < 2.0


@pytest.mark.parametrize('sid', ['kick/electro', 'snare/brush'])
def test_the_invented_ones_still_render(sid):
    y, meta = _take(sid, 1)
    assert np.isfinite(y).all() and meta['checks']['ok'], meta['checks']['warnings']


# ------------------------------------------------------------------------------------------------------ slow
@pytest.mark.slow
@pytest.mark.parametrize('sid', IDS)
def test_every_member_renders_clean_and_passes_its_audit(sid):
    """every member seeds 0 .. n-1 reach at energy -1, 0 and 1, once each (a take that fails its audit re-rolled as
    make re-rolls it)"""
    seen = set()
    for energy in (-1.0, 0.0, 1.0):
        for seed in range(len(INDEX[sid]['members'])):
            k = drums.pick(sid, seed, energy)
            if k in seen:
                continue
            seen.add(k)
            ok = False
            for retry in (0, 101, 202):
                y, meta = registry.render(sid, seed=seed + retry, energy=energy)
                assert np.isfinite(y).all() and np.abs(y).max() <= 1.0, (sid, k)
                assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-2, (sid, k)
                ok = meta['checks']['ok']
                if ok:
                    break
            assert ok, (sid, k, meta['checks']['warnings'])
