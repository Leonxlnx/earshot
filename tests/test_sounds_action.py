"""earshot.sounds.action: the physical action designs render clean, repeat per seed, vary between seeds and answer
to their controls (model-free)."""
import numpy as np
import pytest

from earshot.forge import dsp
from earshot.sounds import action, registry

registry.load()
IDS = sorted(d for d, v in registry._designs.items() if v.fn.__module__ == action.__name__)
STEPS = [i for i in IDS if i.startswith('step/')]


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


def _env(y, hop=480):
    m = np.abs(_mono(y))
    k = len(m) // hop
    return np.sqrt((m[:k * hop].reshape(k, hop) ** 2).mean(1))


def test_catalogue_entries():
    assert len(IDS) == 47
    for sid in IDS:
        d = registry.get(sid)
        assert d.domains == ('game', 'motion'), sid
        assert 2 <= len(d.says) <= 3 and d.uses and d.dur, sid
        assert {'energy', 'size'} <= set(d.axes), sid


@pytest.mark.parametrize('sid', IDS)
def test_renders_clean(sid):
    y, meta = registry.render(sid, seed=3)
    assert np.isfinite(y).all() and y.ndim == 2
    assert np.abs(y).max() <= 1.0
    assert meta['checks']['ok'], meta['checks']['warnings']
    assert not meta.get('ignored')
    d = registry.get(sid)
    assert 0.5 * d.dur <= meta['dur_s'] <= 3.0 * d.dur + 0.3, (meta['dur_s'], d.dur)


@pytest.mark.parametrize('sid', ['jump/spring', 'step/gravel', 'hit/metal', 'explosion/small', 'shot/pistol',
                                 'laser/beam', 'door/open', 'break/glass', 'splash/small', 'heal/glow', 'swing/blade',
                                 'ui/error_buzz'])
def test_deterministic_per_seed(sid):
    a, b, c = _raw(sid, 4), _raw(sid, 4), _raw(sid, 5)
    assert np.array_equal(a, b)
    assert len(a) != len(c) or not np.allclose(a, c)


@pytest.mark.parametrize('sid', ['hit/punch', 'explosion/big', 'step/wood', 'swing/blade', 'door/close', 'laser/pulse',
                                 'splash/big', 'countdown/beep', 'rumble/quake', 'jump/spring'])
def test_size_lowers_centroid(sid):
    big, _ = registry.render(sid, seed=2, size=0.8, space=-1.0)          # finished: no sub-audible content
    small, _ = registry.render(sid, seed=2, size=-0.8, space=-1.0)
    assert _centroid(big) < _centroid(small)


@pytest.mark.parametrize('sid', ['hit/metal', 'explosion/small', 'step/gravel', 'whoosh/pass', 'magic/cast',
                                 'ui/error_buzz', 'break/glass', 'laser/beam'])
def test_energy_raises_level_and_brightness(sid):
    lo, hi = _raw(sid, 2, energy=-0.8), _raw(sid, 2, energy=0.8)
    assert np.sqrt(np.mean(hi ** 2)) > 1.5 * np.sqrt(np.mean(lo ** 2))
    assert _centroid(hi) > _centroid(lo)


@pytest.mark.parametrize('sid', STEPS)
def test_steps_vary_but_measure_alike(sid):
    (a, ma), (b, mb) = registry.render(sid, seed=1), registry.render(sid, seed=2)
    ea, eb = _env(a), _env(b)
    k = min(len(ea), len(eb))
    assert np.corrcoef(ea[:k], eb[:k])[0, 1] < 0.98
    assert abs(ma['dur_s'] - mb['dur_s']) <= 0.2 * max(ma['dur_s'], mb['dur_s'])
    assert abs(ma['lufs_max'] - mb['lufs_max']) <= 2.0


@pytest.mark.parametrize('sid', ['laser/beam', 'rumble/quake'])
def test_loops_join_without_a_seam(sid):
    y, meta = registry.render(sid, seed=1, dur=1.5)
    assert abs(meta['dur_s'] - 1.5) < 0.01
    m = _mono(y)
    step = abs(m[0] - m[-1])
    assert step < 4 * np.percentile(np.abs(np.diff(m)), 99.9)


def test_countdown_go_is_a_fifth_above_and_longer():
    beep, go = registry.get('countdown/beep'), registry.get('countdown/go')
    assert abs(12 * np.log2(registry.Controls(go).hz() / registry.Controls(beep).hz()) - 7) < 0.01
    assert go.dur > 2 * beep.dur


def _f0(m, lo, hi, t0, t1, frame=0.04):
    """(f0 Hz, harmonicity 0..1) every half frame from t0 to t1 s: the autocorrelation peak between 1 / hi and 1 / lo"""
    out = []
    n = int(frame * dsp.SR)
    for s in np.arange(t0, t1 - frame, frame / 2):
        i = int(s * dsp.SR)
        a = np.fft.irfft(np.abs(np.fft.rfft(m[i:i + n] * np.hanning(n), 4 * n)) ** 2)
        l0, l1 = int(dsp.SR / hi), int(dsp.SR / lo)
        j = l0 + np.argmax(a[l0:l1])
        out.append((dsp.SR / j, a[j] / a[0]))
    return np.array(out)


@pytest.mark.parametrize('seed', [1, 2])
def test_door_creak_is_pitched_but_wanders(seed):
    """a stick-slip creak: most of the swing has a slip rate (20-250 Hz) the ear hears as pitch, and it wanders"""
    p = _f0(dsp.butter(_mono(_raw('door/open', seed)), 'low', 2000.0), 20, 250, 0.3, 0.8)
    voiced = p[p[:, 1] > 0.3]
    assert len(voiced) >= 0.6 * len(p)
    assert voiced[:, 0].max() / voiced[:, 0].min() > 1.5


def test_slaps_differ_from_take_to_take():
    a, b = _env(_raw('hit/slap', 1)), _env(_raw('hit/slap', 2))
    k = min(len(a), len(b))
    assert np.corrcoef(a[:k], b[:k])[0, 1] < 0.98


def test_quake_has_events_over_its_rumble():
    m = _mono(_raw('rumble/quake', 1))
    e = _env(dsp.butter(m, 'band', (150, 700), order=4), 240)
    e = e - e.mean()
    assert np.mean(e ** 4) / np.mean(e ** 2) ** 2 > 5.0          # knocks and rattles stand out (noise: about 3)
