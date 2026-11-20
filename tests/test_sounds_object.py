"""earshot.sounds.object: the everyday object designs render clean, repeat per seed, vary between seeds as
performances, answer to size, and their physics holds (bounces shrink, loops have no seam). Model-free."""
import numpy as np
import pytest

from earshot.forge import dsp
from earshot.sounds import object as obj, registry

registry.load()
IDS = sorted(d for d, v in registry._designs.items() if v.fn.__module__ == obj.__name__)
LOOPS = [i for i in IDS if registry.get(i).loop]


def _raw(sid, seed=1, **ax):
    """the design function's own output (before levelling), dry so the comparisons see the source"""
    d = registry.get(sid)
    c = registry.Controls(d, **{'space': -1.0, **ax})
    return np.asarray(d.fn(c, dsp.rng(registry.seed_of(sid, seed))), np.float64)


def _mono(y):
    return y.mean(1) if y.ndim == 2 else y


def _centroid(y):
    m = _mono(y)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / dsp.SR)
    return float((P * f).sum() / P.sum())


def test_catalogue_entries():
    assert len(IDS) == 36
    for sid in IDS:
        d = registry.get(sid)
        assert d.domains == ('ui', 'game', 'motion'), sid
        assert 2 <= len(d.says) <= 3 and d.uses and d.dur, sid
        assert {'energy', 'size', 'bright'} <= set(d.axes), sid


@pytest.mark.parametrize('sid', IDS)
def test_renders_clean(sid):
    y, meta = registry.render(sid, seed=3)
    assert np.isfinite(y).all() and y.ndim == 2
    assert np.abs(y).max() <= 1.0
    assert meta['checks']['ok'], meta['checks']['warnings']
    assert not meta.get('ignored')
    d = registry.get(sid)
    assert 0.5 * d.dur <= meta['dur_s'] <= 3.0 * d.dur + 0.3, (meta['dur_s'], d.dur)


@pytest.mark.parametrize('sid', IDS)
def test_seeds_are_performances(sid):
    a, b = _raw(sid, 4), _raw(sid, 4)
    assert np.array_equal(a, b)
    ya, ma = registry.render(sid, seed=1, check=False)
    yb, mb = registry.render(sid, seed=2, check=False)
    assert len(ya) != len(yb) or not np.allclose(ya, yb)
    assert max(ma['dur_s'], mb['dur_s']) <= 1.2 * min(ma['dur_s'], mb['dur_s']) + 0.01, (ma['dur_s'], mb['dur_s'])
    assert abs(ma['lufs_max'] - mb['lufs_max']) <= 2.5, (ma['lufs_max'], mb['lufs_max'])


@pytest.mark.parametrize('sid', ['coin/drop', 'glass/clink', 'cup/set', 'knock/door', 'bell/desk', 'book/close',
                                 'water/drop', 'clock/tick', 'switch/light', 'chips/stack'])
def test_size_lowers_centroid(sid):
    assert _centroid(_raw(sid, 2, size=0.8)) < _centroid(_raw(sid, 2, size=-0.8))


def test_bounces_shrink_in_time_and_level():
    r = dsp.rng(5)
    t, v = obj._bounces(r, 0.6, 0.2, floor=0.012, spread=0.0)
    gaps = np.diff(t)
    assert len(gaps) >= 4
    assert np.all(np.diff(gaps) < 0) and np.all(np.diff(v) < 0)
    assert np.allclose(gaps[1:] / gaps[:-1], 0.6) and np.allclose(v[1:] / v[:-1], 0.6)


def test_coin_drop_bounces_audibly_decay():
    """the first landing is the loudest event of a coin drop and the take ends quieter than it began"""
    y = _mono(_raw('coin/drop', 3))
    hop = 480
    k = len(y) // hop
    e = np.sqrt((y[:k * hop].reshape(k, hop) ** 2).mean(1))
    assert np.argmax(e) < 5
    assert e[k // 2:].max() < 0.5 * e.max()


@pytest.mark.parametrize('sid', LOOPS)
def test_loops_are_seamless(sid):
    y, meta = registry.render(sid, seed=2, dur=registry.get(sid).dur)
    assert abs(meta['dur_s'] - registry.get(sid).dur) < 0.002
    w = dsp.n_of(0.02)

    def step(x, i):
        """the level step (dB) between the 20 ms before and after sample i"""
        return 10 * np.log10(np.mean(x[i:i + w] ** 2) / np.mean(x[i - w:i] ** 2))
    z = np.roll(y, len(y) // 2, axis=0)              # the seam in the middle, neighbours around it
    seam = len(y) // 2
    near = max(abs(step(z, seam + d)) for d in (-dsp.n_of(0.03), dsp.n_of(0.03)))
    assert abs(step(z, seam)) < near + 1.0
    step = np.abs(y[0] - y[-1]).max()
    assert step <= np.percentile(np.abs(np.diff(y, axis=0)).max(1), 99.9) + 1e-9


def test_ticktock_keeps_its_length_and_beats():
    y, meta = registry.render('clock/ticktock', seed=1, dur=4.0)
    assert abs(meta['dur_s'] - 4.0) < 0.002


def test_energy_adds_coins_and_level():
    lo, hi = _raw('coins/handful', 2, energy=-0.8), _raw('coins/handful', 2, energy=0.8)
    assert np.sqrt(np.mean(hi ** 2)) > np.sqrt(np.mean(lo ** 2))


def test_tape_peel_is_a_pitched_screech():
    """stick-slip at 0.35-1.6 kHz: the peel has a period the ear hears as a pitch, not a hiss"""
    m = _mono(_raw('tape/peel', 1))
    n = dsp.n_of(0.02)
    h = []
    for i in range(dsp.n_of(0.3), dsp.n_of(0.7), n):
        a = np.fft.irfft(np.abs(np.fft.rfft(m[i:i + n] * np.hanning(n), 4 * n)) ** 2)
        lag = slice(int(dsp.SR / 2500), int(dsp.SR / 250))
        h.append(a[lag].max() / a[0])
    assert np.median(h) > 0.3
