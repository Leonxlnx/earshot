"""earshot.sounds.motion: the film vocabulary renders clean, repeats per seed, varies between seeds, keeps the length
and the peak position it is asked for, answers to its controls and never sweeps a pitch upward (model-free)."""
import functools

import numpy as np
import pytest

from earshot.forge import dsp, sfx
from earshot.sounds import motion, registry
from earshot.synth import master

registry.load()
IDS = sorted(d for d, v in registry._designs.items() if v.fn.__module__ == motion.__name__)
AIR = ['whoosh/soft', 'whoosh/fast', 'whoosh/deep', 'whoosh/airy', 'whoosh/tonal', 'swoosh/in', 'swoosh/out',
       'flick/paper', 'flick/thin', 'slide/panel', 'rise/air', 'rise/tension', 'rise/reverse', 'fall/air', 'suck/in']
# sustained gestures and the share of the length around which their loudest stretch sits
GESTURES = {'whoosh/soft': 0.52, 'whoosh/fast': 0.52, 'whoosh/tonal': 0.5, 'swoosh/in': 0.78, 'swoosh/out': 0.2,
            'rise/air': 0.86, 'suck/in': 0.86}


@functools.lru_cache(maxsize=None)
def _take(sid, seed=1, **req):
    return registry.render(sid, seed=seed, check=False, **req)


@functools.lru_cache(maxsize=None)
def _raw(sid, seed=1, **ax):
    """the design function's own output (before levelling), dry so the comparisons see the source"""
    d = registry.get(sid)
    c = registry.Controls(d, **{'space': -1.0, **ax})
    return np.asarray(d.fn(c, dsp.rng(registry.seed_of(sid, seed))), np.float64)


def _mono(y):
    return y.mean(1) if y.ndim == 2 else y


def _spectrum(y):
    m = dsp.butter(_mono(y), 'high', 25.0)                   # as a finished take: nothing sub-audible
    return np.abs(np.fft.rfft(m)) ** 2, np.fft.rfftfreq(len(m), 1 / dsp.SR)


def _centroid(y):
    P, f = _spectrum(y)
    return float((P * f).sum() / P.sum())


def _share(y, lo, hi):
    """the share of the energy between lo and hi Hz"""
    P, f = _spectrum(y)
    return float(P[(f >= lo) & (f < hi)].sum() / P.sum())


def _env(y, hop=480):
    m = _mono(y)
    k = len(m) // hop
    return np.sqrt((m[:k * hop].reshape(k, hop) ** 2).mean(1))


def _peak_share(y, window=0.15):
    """the centre of the loudest stretch of a take (`window` of its length), as a share of the length"""
    e = _env(y, 240) ** 2
    w = max(1, int(window * len(e)))
    c = np.concatenate([[0.0], np.cumsum(e)])
    return float((np.argmax(c[w:] - c[:-w]) + w / 2) / len(e))


def _ridge(y, t0, t1, lo, hi):
    """the strongest frequency between lo and hi Hz in the stretch t0..t1 s"""
    m = _mono(y)[int(t0 * dsp.SR):int(t1 * dsp.SR)]
    P = np.abs(np.fft.rfft(m * np.hanning(len(m)), 1 << 17))
    f = np.fft.rfftfreq(1 << 17, 1 / dsp.SR)
    k = (f >= lo) & (f <= hi)
    return float(f[k][np.argmax(P[k])])


def test_catalogue_entries():
    assert len(IDS) == 37
    assert len({i.split('/')[0] for i in IDS}) == 19
    for sid in IDS:
        d = registry.get(sid)
        assert d.domains[0] == 'motion', sid
        assert 2 <= len(d.says) <= 3 and len(d.uses) >= 4 and d.dur, sid
        assert {'energy', 'size', 'bright', 'hard', 'space', 'width'} <= set(d.axes), sid


@pytest.mark.parametrize('sid', IDS)
def test_renders_clean(sid):
    d = registry.get(sid)
    c = registry.Controls(d)
    y = master.finish(d.fn(c, dsp.rng(registry.seed_of(sid, 1))))            # as registry.render, keeping the controls
    assert np.isfinite(y).all() and y.ndim == 2
    assert np.abs(y).max() <= 1.0
    assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-3          # starts at zero, its tail is faded
    checks = registry.audit(y, d)
    assert checks['ok'], checks['warnings']
    assert 0.6 * d.dur <= len(y) / dsp.SR <= 1.6 * d.dur + 0.4, (len(y) / dsp.SR, d.dur)
    assert c.used & set(registry.AXES) == set(d.axes), sid                  # the axes it lists are the ones it reads


@pytest.mark.parametrize('sid', ['whoosh/fast', 'boom/wide', 'knock/soft', 'pop/bubble', 'shine/sweep', 'glitch/short',
                                 'type/run'])
def test_deterministic_per_seed(sid):
    d = registry.get(sid)
    a, b, c = (np.asarray(d.fn(registry.Controls(d, space=-1.0), dsp.rng(registry.seed_of(sid, s))))
               for s in (4, 4, 5))
    assert np.array_equal(a, b)
    assert len(a) != len(c) or not np.allclose(a, c)


@pytest.mark.parametrize('sid', sorted(GESTURES))
def test_dur_is_kept_and_the_peak_stays_at_its_share(sid):
    d = registry.get(sid)
    shares = []
    _, meta = _take(sid, 2, dur=round(1.7 * d.dur, 3))
    assert 'dur' not in meta.get('ignored', ())
    assert 0.85 * 1.7 * d.dur <= meta['dur_s'] <= 1.2 * 1.7 * d.dur + 0.35, meta['dur_s']
    for dur in (round(0.8 * d.dur, 3), round(1.7 * d.dur, 3)):
        raw = _raw(sid, 2, dur=dur)
        assert abs(len(raw) / dsp.SR - dur) < 0.02 * dur + 0.01                 # the gesture itself is that long
        shares.append(_peak_share(raw))
    assert abs(shares[0] - shares[1]) < 0.15, shares
    assert abs(np.mean(shares) - GESTURES[sid]) < 0.15, shares


@pytest.mark.parametrize('sid', ['type/run', 'fall/sub', 'slide/panel', 'flick/paper'])
def test_other_lengths_are_kept(sid):
    dur = round(1.6 * registry.get(sid).dur, 3)
    _, meta = _take(sid, 2, dur=dur)
    assert 0.8 * dur <= meta['dur_s'] <= 1.25 * dur + 0.4, (dur, meta['dur_s'])


@pytest.mark.parametrize('sid', ['whoosh/soft', 'whoosh/tonal', 'flick/thin', 'slide/panel', 'boom/short', 'slam/title',
                                 'knock/soft', 'rise/tension', 'pop/soft', 'tock/glass', 'blip/soft', 'shine/glint',
                                 'sparkle/burst', 'glitch/short', 'type/soft', 'sting/tech'])
def test_size_lowers_centroid(sid):
    assert _centroid(_raw(sid, 2, size=0.8)) < _centroid(_raw(sid, 2, size=-0.8))


@pytest.mark.parametrize('sid', ['whoosh/soft', 'flick/thin', 'boom/short', 'slam/title', 'suck/in', 'pop/soft',
                                 'tock/wood', 'blip/soft', 'shine/glint', 'glitch/short', 'type/soft'])
def test_energy_and_bright_act(sid):
    def top(**ax):
        y = _raw(sid, 2, **ax)
        return _centroid(y), _share(y, 2000.0, 24000.0)
    assert top(bright=0.8) > top() > top(bright=-0.8)
    assert top(energy=0.8) > top(energy=-0.8)
    assert top(bright=0.8)[1] > top(bright=-0.8)[1] and top(energy=0.8)[1] > top(energy=-0.8)[1]


@pytest.mark.parametrize('sid', ['boom/short', 'slam/title', 'knock/soft', 'tock/wood', 'thump/soft'])
def test_hard_is_a_faster_onset(sid):
    def early(y):
        m = _mono(y)
        return float(np.sum(m[:dsp.n_of(0.004)] ** 2) / np.sum(m ** 2))       # how much of it arrives within 4 ms
    assert early(_raw(sid, 2, hard=0.9)) > early(_raw(sid, 2, hard=-0.9))
    assert _centroid(_raw(sid, 2, hard=0.9)) > _centroid(_raw(sid, 2, hard=-0.9))


@pytest.mark.parametrize('sid', ['whoosh/soft', 'whoosh/fast', 'thump/soft', 'pop/soft', 'tock/wood', 'type/soft',
                                 'sparkle/burst'])
def test_seeds_vary_but_measure_alike(sid):
    (a, ma), (b, mb) = _take(sid, 1), _take(sid, 2)
    k = min(len(a), len(b))
    assert abs(np.corrcoef(_mono(a)[:k], _mono(b)[:k])[0, 1]) < 0.9           # another take, not a copy
    assert abs(ma['dur_s'] - mb['dur_s']) <= 0.25 * max(ma['dur_s'], mb['dur_s'])
    assert abs(ma['lufs_max'] - mb['lufs_max']) <= 2.5
    assert abs(np.log2(_centroid(a) / _centroid(b))) < 0.6


def test_keys_differ_from_take_to_take():
    cen = [_centroid(_raw('type/soft', s)) for s in range(1, 7)]
    assert max(cen) / min(cen) > 1.05                               # another key under another finger
    assert max(cen) / min(cen) < 3.0                                # of the same keyboard


@pytest.mark.parametrize('sid', AIR)
def test_no_rising_glide_in_air_and_builds(sid):
    assert sfx.rising_glide(_raw(sid, 3))['rise_st'] <= 4.0


@pytest.mark.parametrize('sid', ['whoosh/tonal', 'rise/tension'])
def test_no_rising_glide_when_pushed(sid):
    assert sfx.rising_glide(_raw(sid, 3, energy=1.0, motion=1.0))['rise_st'] <= 4.0


def test_tonal_whoosh_falls_by_doppler_alone():
    y = _raw('whoosh/tonal', 2, pitch=220.0)
    n = len(y) / dsp.SR
    before, after = _ridge(y, 0.1 * n, 0.4 * n, 170, 290), _ridge(y, 0.6 * n, 0.9 * n, 170, 290)
    assert 1.0 < 12 * np.log2(before / after) <= 4.0                 # it falls, by less than four semitones


def test_tension_riser_climbs_a_few_semitones_at_most():
    y = _raw('rise/tension', 2, pitch=146.83, motion=1.0)
    n = len(y) / dsp.SR
    start, end = _ridge(y, 0.05 * n, 0.3 * n, 60, 85), _ridge(y, 0.75 * n, 0.92 * n, 60, 100)     # the sub octave
    assert 0.5 < 12 * np.log2(end / start) <= 3.2


def test_builds_end_on_the_cut_and_die_after_it():
    for sid in ('rise/air', 'rise/tension', 'rise/reverse', 'suck/in'):
        raw = _raw(sid, 1)
        e = _env(raw, 240)
        assert _peak_share(raw) > 0.75, sid                         # its loudest stretch is its last
        assert e[-max(2, len(e) // 50):].max() < 0.2 * e.max(), sid  # and the gesture itself falls away after it
        _, meta = _take(sid)
        assert meta['dur_s'] < len(raw) / dsp.SR + 0.5, sid         # the room dies within half a second


def test_booms_are_low_and_swell_in():
    for sid in ('boom/deep', 'boom/short', 'fall/sub'):
        assert _share(_take(sid)[0], 0, 150) > 0.7, sid
    assert _take('boom/short')[1]['dur_s'] < 0.5
    e = _env(_raw('boom/short', 1), 48)
    assert np.argmax(e >= 0.9 * e.max()) >= 5                        # at least 5 ms to its peak: a rise, not a click
    assert _share(_take('slam/title')[0], 0, 150) < 0.6             # a slam is a board, not a kick drum
    wide, short = _take('boom/wide')[0], _take('boom/short')[0]
    assert np.corrcoef(wide[:, 0], wide[:, 1])[0, 1] < np.corrcoef(short[:, 0], short[:, 1])[0, 1] - 0.1


def test_sub_drop_blooms_after_the_cut_and_sinks():
    y = _raw('fall/sub', 1)
    assert 0.2 < _peak_share(y) * len(y) / dsp.SR < 0.6             # the sub's loudest stretch comes after the cut
    assert _ridge(y, 0.02, 0.25, 30, 100) > 1.15 * _ridge(y, 0.7, 1.2, 30, 100)


def test_small_film_sounds_are_short_and_near_mono():
    for sid in ('pop/soft', 'knock/soft', 'tock/wood', 'type/soft', 'blip/soft'):
        y, meta = _take(sid)
        assert meta['dur_s'] < 0.5, sid
        assert np.corrcoef(y[:, 0], y[:, 1])[0, 1] > 0.85, sid


def test_typing_run_follows_motion_and_dur():
    def keys(**req):
        e = _env(_raw('type/run', 2, **req), 480)
        return int(((e[1:] > 0.4 * e.max()) & (e[:-1] <= 0.4 * e.max())).sum())
    slow, fast = keys(dur=2.0, motion=-1.0), keys(dur=2.0, motion=1.0)
    assert 8 <= slow < fast <= 40
    assert keys(dur=1.0) < keys(dur=3.0)


def test_shimmer_sweeps_from_one_side_to_the_other():
    y = dsp.butter(_raw('shine/sweep', 2, motion=0.5), 'high', 3000.0)       # the grains, without the held tone
    i = int(_peak_share(y, 0.3) * len(y))
    k = len(y) // 4
    lean = [float(np.log(np.sqrt(np.mean(p[:, 1] ** 2)) / np.sqrt(np.mean(p[:, 0] ** 2))))
            for p in (y[max(0, i - k):i], y[i:i + k])]
    assert lean[0] * lean[1] < 0 and abs(lean[0] - lean[1]) > 0.5         # one side before its middle, the other after


def test_swooshes_arrive_and_leave():
    assert _peak_share(_raw('swoosh/in', 1)) > 0.65
    assert _peak_share(_raw('swoosh/out', 1)) < 0.3
    y = _raw('swoosh/out', 1)
    k = len(y) // 2
    assert _centroid(y[:k]) > 1.5 * _centroid(y[k:])                # it darkens as it leaves


def test_stingers_take_a_voice_and_a_pitch():
    _, meta = _take('sting/reveal', 1, voice='glockenspiel', pitch='D6')
    assert meta['voice'] == 'glockenspiel' and meta['pitch'] == 'D6' and not meta.get('ignored')
    _, meta = _take('sting/soft', 1, pitch='A3')
    assert meta['voice'] == 'felt_piano' and meta['pitch'] == 'A3'
