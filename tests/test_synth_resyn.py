"""earshot.synth.resyn on synthetic signals: the round trip keeps band envelopes, onsets and partials, takes differ
sample by sample but not in what is heard, transforms do what they say, models are small and hold no audio."""
import functools
import io

import numpy as np

from earshot.forge import dsp
from earshot.synth import master, resyn

SR = dsp.SR
PARTIALS = ((440.0, 0.25, 1.0), (1213.0, 0.15, 0.5), (2378.0, 0.08, 0.3))      # Hz, time constant s, amplitude
CLICKS = (2400, 7200, 9600, 16800)                                             # samples


def _tone():
    t = np.arange(SR // 2) / SR
    return 0.5 * sum(a * np.exp(-t / d) * np.sin(2 * np.pi * f * t) for f, d, a in PARTIALS)


def _clicks():
    x = np.zeros(SR // 2)
    x[list(CLICKS)] = 0.8
    return x


def _noise():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    shape = np.exp(-t / 0.15) * (1 - np.exp(-t / 0.01))
    return dsp.butter(dsp.rng(1).standard_normal(n), 'band', (500, 6000)) * shape * 0.2


def _sweep():
    n = int(0.6 * SR)
    f = 300 * (8000 / 300) ** (np.arange(n) / n)
    return dsp.svf(dsp.rng(2).standard_normal(n), f, 4.0, 'bandn') * np.hanning(n) * 0.3


def _rumble():
    n = int(0.4 * SR)
    return dsp.butter(dsp.rng(3).standard_normal(n), 'band', (50, 1500)) * np.hanning(n) * 0.2


SIGNALS = {'tone': _tone, 'clicks': _clicks, 'noise': _noise, 'sweep': _sweep, 'rumble': _rumble}


@functools.lru_cache(maxsize=None)
def _x(name):
    return SIGNALS[name]()


@functools.lru_cache(maxsize=None)
def _model(name):
    return resyn.analyze(_x(name), SR)


@functools.lru_cache(maxsize=None)
def _take(name, seed=1, **kw):
    return resyn.render(_model(name), dsp.rng(seed), **kw)


def _centroid(y):
    F = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    return float(np.sum(np.fft.rfftfreq(len(y), 1 / SR) * F) / np.sum(F))


def _low_db(y, hz=120.0):
    F = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    return float(10 * np.log10(F[np.fft.rfftfreq(len(y), 1 / SR) < hz].sum() + 1e-30))


def test_takes_are_clean_stereo_and_deterministic():
    for name in ('tone', 'clicks', 'noise', 'sweep'):
        y = _take(name)
        assert y.shape == (len(_x(name)), 2) and y.dtype == np.float64
        assert np.isfinite(y).all() and np.abs(y).max() < 1.0
        assert np.abs(y[-1]).max() < 1e-3 and abs(y.mean()) < 1e-3
        assert np.array_equal(y, resyn.render(_model(name), dsp.rng(1)))


def test_round_trip_keeps_the_band_envelopes():
    for name, env, shape in (('tone', 3.0, 2.0), ('clicks', 3.0, 1.6), ('noise', 4.0, 1.4), ('sweep', 4.0, 1.4)):
        f = resyn.fidelity(_x(name), _take(name), SR)
        assert f['env_db'] < env, (name, f)
        assert f['shape_db'] < shape, (name, f)
        assert abs(f['level_db']) < 1.0, (name, f)
        assert f['mod_db'] < 3.0, (name, f)


def test_onsets_are_found_and_stay_sharp():
    m = _model('clicks')
    assert np.array_equal(np.asarray(m['on']) * resyn.DEC, CLICKS)
    f = resyn.fidelity(_x('clicks'), _take('clicks'), SR)
    assert abs(f['rise_ms'][0] - f['rise_ms'][1]) <= 1.0, f
    y = _take('clicks')[:, 0]
    for c in CLICKS:                                           # each click is still one pulse, where it was
        seg = np.abs(y[c - 96:c + 96])
        assert abs(int(np.argmax(seg)) - 96) <= 12
        assert seg.max() > 0.45                                # of 0.8: the bands start together
        assert np.sqrt(np.mean(y[c - 480:c - 96] ** 2)) < 0.02 * seg.max()      # nothing audible before it


def test_partials_stay_within_cents_and_keep_their_decay():
    px, py = resyn.partials(_x('tone'), SR, count=3), resyn.partials(_take('tone'), SR, count=3)
    assert len(px) == len(py) == 3
    for (f, _, _), (fx, lx, tx), (fy, ly, ty) in zip(PARTIALS, px, py):
        assert abs(fx - f) < 1.0
        assert abs(1200 * np.log2(fy / fx)) < 3.0, (fx, fy)
        assert abs(ly - lx) < 2.0
        assert 0.8 < ty / tx < 1.25, (tx, ty)


def test_two_takes_differ_in_samples_but_not_in_envelopes():
    for name in ('noise', 'sweep'):
        a, b = _take(name, 1)[:, 0], _take(name, 2)[:, 0]
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.6
        f = resyn.fidelity(_take(name, 1), _take(name, 2))
        assert f['shape_db'] < 1.5 and abs(f['level_db']) < 0.7, (name, f)


def test_length_pitch_and_size():
    base = _take('tone')
    longer = _take('tone', length=1.5)
    assert 1.35 < len(longer) / len(base) < 1.6
    up = _take('tone', pitch=12.0)
    assert abs(len(up) / len(base) - 1.0) < 0.06
    assert abs(1200 * np.log2(resyn.partials(up, SR, count=1)[0][0] / 880.0)) < 5.0
    big = _take('tone', size=2.0)
    assert abs(len(big) / len(base) - 2.0) < 0.02
    assert abs(1200 * np.log2(resyn.partials(big, SR, count=1)[0][0] / 220.0)) < 5.0


def test_bright_hard_ring_and_weight():
    assert _centroid(_take('noise', bright=3.0)) > 1.25 * _centroid(_take('noise'))
    assert _centroid(_take('noise', bright=-3.0)) < 0.8 * _centroid(_take('noise'))
    peak = np.abs(_take('clicks')).max()
    assert np.abs(_take('clicks', hard=1.0)).max() > 1.3 * peak
    assert np.abs(_take('clicks', hard=-1.0)).max() < 0.8 * peak
    t60 = resyn.partials(_take('tone'), SR, count=1)[0][2]
    rung = _take('tone', ring=2.0)
    assert 1.5 < resyn.partials(rung, SR, count=1)[0][2] / t60 < 2.6
    assert len(rung) > 1.5 * len(_take('tone'))
    assert 4.0 < _low_db(_take('rumble', weight=6.0)) - _low_db(_take('rumble')) < 7.5


def test_vary_gives_another_good_take():
    a, b = _take('noise', 1, vary=1.0), _take('noise', 2, vary=1.0)
    assert np.isfinite(a).all() and len(a) != len(b)
    f = resyn.fidelity(_x('noise'), a, SR, align=True)
    assert f['shape_db'] < 4.0 and abs(f['level_db']) < 3.0, f


def test_stereo_level_difference_and_coherence():
    r = dsp.rng(5)
    n = int(0.3 * SR)
    mid, side = [dsp.butter(r.standard_normal(n), 'band', (300, 5000)) * np.hanning(n) * 0.1 for _ in range(2)]
    x = np.stack([2.0 * (mid + 0.6 * side), mid - 0.6 * side], axis=1)
    m = resyn.analyze(x, SR)
    y = resyn.render(m, dsp.rng(1))
    db = lambda v: 10 * np.log10(np.mean(v[:, 0] ** 2) / np.mean(v[:, 1] ** 2))      # noqa: E731
    assert abs(db(y) - db(x)) < 1.5
    assert abs(np.corrcoef(y[:, 0], y[:, 1])[0, 1] - np.corrcoef(x[:, 0], x[:, 1])[0, 1]) < 0.2


def test_ringing_low_bands_keep_their_channel_correlation():
    """low ringing modes whose channels are 0.3, 0.5 and 0.8 of half a turn apart (correlation about 0.6, 0, -0.8)
    keep that correlation: the coherent carrier of a band is not copied into both channels with the sign of its
    coherence (which put the lows of a wide recording fully in or out of phase: 1.0 and -1.0 for the first and the
    last)"""
    t = np.arange(int(0.6 * SR)) / SR
    env = np.exp(-t / 0.2) * (1 - np.exp(-t / 0.002))

    def corr(v):
        lo = np.stack([dsp.butter(v[:, k], 'low', 250.0, order=4) for k in range(2)], axis=1)
        return float(np.corrcoef(lo[:, 0], lo[:, 1])[0, 1])
    for turn in (0.3 * np.pi, 0.5 * np.pi, 0.8 * np.pi):
        x = 0.4 * env[:, None] * np.stack([np.sin(2 * np.pi * f * t[:, None] + np.array([0.0, turn]))
                                           for f in (110.0, 180.0)]).sum(axis=0)
        y = resyn.render(resyn.analyze(x, SR), dsp.rng(1))
        assert abs(corr(y) - corr(x)) < 0.15, (turn, corr(x), corr(y))


def test_models_are_small_numbers_and_survive_a_file(tmp_path):
    for name, most in (('tone', 6000), ('clicks', 9000), ('noise', 20000), ('sweep', 20000)):
        m = _model(name)
        assert resyn.size(m) < most, (name, resyn.size(m))
        assert resyn.size(m) < 0.3 * 2 * len(_x(name))            # of the bytes of the signal as 16-bit audio
        assert all(np.asarray(v).dtype.kind in 'iuf' for v in m.values())
        # the largest array is the envelope's grain: one small integer per band and frame, nearly all within 2 of zero
        assert np.abs(m['q']).max() <= 40 and np.mean(np.abs(m['q']) <= 2) > 0.9
    path = str(tmp_path / 'tone.npz')
    resyn.save(path, _model('tone'))
    back = resyn.load(path)
    assert set(back) == set(_model('tone'))
    assert all(np.array_equal(back[k], _model('tone')[k]) for k in back)
    assert np.array_equal(resyn.render(back, dsp.rng(4)), resyn.render(_model('tone'), dsp.rng(4)))
    with np.load(path) as z:
        assert z.files == ['model']
    assert io.BytesIO(open(path, 'rb').read()).getbuffer().nbytes == resyn.size(_model('tone'))


def _knock(f0, tau, bright, seed):
    """a synthetic struck thing: a click, two decaying modes and a burst of contact noise"""
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    x = np.exp(-t / tau) * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2.76 * f0 * t))
    x += bright * dsp.butter(dsp.rng(seed).standard_normal(n), 'high', 2000) * np.exp(-t / 0.004)
    x = np.concatenate([np.zeros(480), 0.4 * x])
    x[480] += 0.5
    return x


def test_morph_and_class_make_members_of_the_kind():
    knocks = [resyn.analyze(_knock(f0, tau, br, i), SR) for i, (f0, tau, br) in
              enumerate(((300.0, 0.03, 0.5), (360.0, 0.04, 0.3), (420.0, 0.025, 0.6), (330.0, 0.05, 0.4)))]
    same = resyn.render(resyn.morph(knocks[0], knocks[0], 0.5), dsp.rng(1))
    f = resyn.fidelity(resyn.render(knocks[0], dsp.rng(1)), same, align=True)
    assert f['shape_db'] < 3.0 and abs(f['level_db']) < 2.0, f
    mid = resyn.render(resyn.morph(knocks[0], knocks[2], 0.5), dsp.rng(1))
    assert np.isfinite(mid).all() and 0.5 * 0.25 * SR < len(mid) < 1.5 * 0.27 * SR
    cm = resyn.fit_class(knocks)
    assert 1 <= len(cm['sd']) <= 3 and cm['pc'].shape[0] == len(cm['sd'])
    a, b = resyn.sample(cm, dsp.rng(7)), resyn.sample(cm, dsp.rng(8))
    assert np.array_equal(a, resyn.sample(cm, dsp.rng(7)))
    assert np.isfinite(a).all() and np.isfinite(b).all() and 0 < np.abs(a).max() < 1.0
    assert len(a) != len(b) or not np.array_equal(a, b)
    lens = [int(k['n']) for k in knocks]
    assert 0.5 * min(lens) < len(a) < 2.0 * max(lens)
    pk = int(np.argmax(np.abs(a[:, 0])))
    assert pk < 0.2 * len(a)                                  # the event comes first, the ring after it
    assert np.sqrt(np.mean(a[pk + 4800:] ** 2)) < 0.3 * np.sqrt(np.mean(a[pk:pk + 2400] ** 2))


def test_finish_is_louder_and_limited():
    y = _take('noise')
    z = resyn.finish(y)
    assert master.loudness(z) > master.loudness(y) + 3.0
    assert master.true_peak_db(z) <= -0.9
    assert np.isfinite(z).all() and np.abs(z[0]).max() < 1e-3 and np.abs(z[-1]).max() < 1e-3
    assert np.array_equal(_take('noise', full=True), resyn.finish(_take('noise')))


def _long():
    """a struck thing that rings for two seconds: two decaying partials over decaying band noise"""
    n = 2 * SR
    t = np.arange(n) / SR
    x = np.exp(-t / 0.5) * (np.sin(2 * np.pi * 110.0 * t) + 0.4 * np.sin(2 * np.pi * 263.0 * t))
    x += dsp.butter(dsp.rng(5).standard_normal(n), 'band', (200, 5000)) * np.exp(-t / 0.3)
    x[:4] += 0.6
    return 0.3 * x


def test_late_frames_cost_little():
    x = _long()
    fine, late = resyn.analyze(x, SR), resyn.analyze(x, SR, late=4)
    assert resyn.size(late) < 0.75 * resyn.size(fine), (resyn.size(late), resyn.size(fine))
    a = resyn.fidelity(x, resyn.render(fine, dsp.rng(1)))
    b = resyn.fidelity(x, resyn.render(late, dsp.rng(1)))
    assert b['env_db'] < a['env_db'] + 0.8 and b['shape_db'] < a['shape_db'] + 0.4, (a, b)
    assert abs(b['rise_ms'][1] - a['rise_ms'][1]) < 0.5                  # the onset keeps its full resolution
    y = resyn.render(late, dsp.rng(1))
    tail = slice(int(1.2 * SR), int(1.5 * SR))
    assert abs(10 * np.log10(np.mean(y[tail] ** 2) / np.mean(x[tail] ** 2))) < 1.5   # the tail is where it was
    assert np.array_equal(resyn.analyze(x, SR, late=1)['q'], fine['q'])            # late=1 is the model as before


def test_models_of_the_store_still_load():
    import os
    store = os.path.join(os.path.dirname(resyn.__file__), '..', 'data', 'resyn')
    m = resyn.load(os.path.join(store, 'whoosh-stick-1.npz'))
    y = resyn.render(m, dsp.rng(1))
    assert np.isfinite(y).all() and y.shape[1] == 2 and 0 < np.abs(y).max() < 1.0
