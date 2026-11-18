"""earshot.synth.quilt: a bed stitched from takes is seamless by its own seam measure, a loop closes, nothing comes
back within 20 s as a copy, seeds differ, energy picks busier pieces, waves join in their troughs. The takes here
stand in for resyn takes: per chunk one envelope (slow swells, cracks at fixed moments), fresh noise per take.
The 20 s copy test and the waves run with --slow."""
import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp
from earshot.synth import quilt, resyn

SR = dsp.SR


def _chunk(r, secs, busy):
    """the envelope of one chunk: a slow swell and crack times (`busy`: crack rate rising over the chunk)"""
    n = int(secs * SR)
    t = np.arange(n) / SR
    swell = 1.0 + 0.3 * np.sin(2 * np.pi * r.uniform(0.1, 0.3) * t + r.uniform(0, 6))
    rate = 2.0 + busy * 18.0 * t / secs
    cracks = np.nonzero(r.uniform(size=n // 480) < rate[::480] * 0.01)[0] * 480
    return swell, cracks, r.uniform(0.5, 3.0, len(cracks))


def _take(env, r):
    swell, cracks, size = env
    n = len(swell)
    sos = signal.butter(2, (300.0, 9000.0), 'band', fs=SR, output='sos')
    y = signal.sosfilt(sos, r.standard_normal((n, 2)), axis=0) * 0.05 * swell[:, None]
    k = np.arange(int(0.006 * SR))
    for c, a in zip(cracks, size):
        m = min(len(k), n - c)
        y[c:c + m] += (a * 0.2 * np.exp(-k[:m] / (0.0015 * SR)) * r.standard_normal(m))[:, None]
    return y


def _takes(seed=0, chunks=3, secs=6.0, per=3, busy=0.0):
    r = dsp.rng(100 + seed)
    envs = [_chunk(r, secs, busy) for _ in range(chunks)]
    return [{'y': _take(envs[i % chunks], r), 'chunk': i % chunks} for i in range(chunks * per)]


@pytest.fixture(scope='module')
def bed():
    tk = _takes()
    y, info = quilt.stitch(tk, 24.0, dsp.rng(1))
    return tk, y, info


def test_a_loop_is_exact_seamless_and_closes(bed):
    _, y, info = bed
    assert y.shape == (24 * SR, 2) and np.isfinite(y).all()
    s = quilt.seams(y, info)
    assert len(s['jumps']) == len(info['joints']) >= 8
    assert s['ratio'] < 1.3 and s['worst'] < 1.2 and s['level_worst'] < 1.2      # within the texture's own variation
    assert s['events'] == 0                                                     # no crack under a fade
    assert any(j < quilt.HOP for j, _, _ in info['joints']) or info['joints'][-1][0] == 0   # the seam is a joint
    edge = np.concatenate([y[-SR // 10:], y[:SR // 10]])
    inner = y[SR:SR + SR // 5]
    assert np.sqrt(np.mean(edge ** 2)) < 2.0 * np.sqrt(np.mean(inner ** 2))      # no bump or hole at the seam


@pytest.mark.slow
def test_nothing_comes_back_as_a_copy_within_20_s(bed):
    tk, y, _ = bed
    rp = quilt.repeats(y, loop=True)
    plain = np.tile(tk[0]['y'], (3, 1))[:14 * SR]
    loop = quilt.repeats(plain, lags=(2.0, 8.0))
    assert rp['copy'] < 0.3 < 0.95 < loop['copy']
    assert rp['max'] < loop['max']


def test_seeds_differ_and_the_same_seed_is_the_same_bed(bed):
    tk, y, info = bed
    y2, info2 = quilt.stitch(tk, 24.0, dsp.rng(2))
    y1, _ = quilt.stitch(tk, 24.0, dsp.rng(1))
    assert np.array_equal(y, y1)
    assert info2['pieces'] != info['pieces'] and not np.allclose(y, y2)


def test_pieces_are_one_to_three_seconds_and_never_the_same_samples_twice(bed):
    _, _, info = bed
    lens = [(b - a) / SR for _, a, b in info['pieces']]
    assert min(lens[:-1]) >= 0.98 and max(lens) <= 3.01
    for i, (t, a, b) in enumerate(info['pieces']):
        for u, c, d in info['pieces'][i + 1:]:
            assert t != u or min(b, d) <= max(a, c)


def test_energy_picks_the_busier_moments():
    tk = _takes(1, busy=1.0)

    def rate(y):
        return quilt.onsets(quilt.features(y)).mean()
    calm = rate(quilt.stitch(tk, 10.0, dsp.rng(1), energy=-1.0)[0])
    busy = rate(quilt.stitch(tk, 10.0, dsp.rng(1), energy=1.0)[0])
    assert busy > 1.1 * calm


@pytest.mark.slow
def test_waves_join_in_their_troughs_and_fill_the_length():
    r = dsp.rng(5)
    t = np.arange(int(6.0 * SR)) / SR
    tk = []
    for i in range(4):
        period = 5.0 + 0.4 * i % 1.2
        swell = 0.05 + np.sin(np.pi * np.clip((t - 0.4) / period, 0, 1)) ** 2
        y = signal.sosfilt(signal.butter(2, 200.0, 'high', fs=SR, output='sos'), r.standard_normal((len(t), 2)),
                           axis=0) * swell[:, None] * 0.1
        tk.append({'y': y, 'chunk': i % 2})
    y, info = quilt.stitch(tk, 20.0, r, periodic=True)
    assert y.shape == (20 * SR, 2) and np.isfinite(y).all()
    lv = np.sqrt(np.mean(y ** 2, axis=1))
    level = np.array([lv[max(0, j - 2400):j + 2400].mean() for j, _, _ in info['joints']])
    assert level.max() < 0.5 * lv.mean()                                       # joints sit in the quiet troughs


def test_a_pair_of_takes_shares_its_crackles_and_not_its_noise():
    r = dsp.rng(9)
    n = SR // 2
    x = 0.01 * r.standard_normal(n)
    for c in r.integers(SR // 20, n - SR // 20, 12):                    # a few sharp cracks over a soft noise
        x[c:c + 96] += 0.6 * r.standard_normal(96) * np.exp(-np.arange(96) / 24.0)
    m = resyn.analyze(x)
    (tk,) = quilt.takes([m], 0.1, dsp.rng(3), vary=0.4, pair=True)[:1]
    y = tk['y']
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    hi = signal.sosfilt(signal.butter(4, 2000.0, 'high', fs=SR, output='sos'), y, axis=0)
    e = np.sqrt(np.convolve(hi[:, 0] ** 2, np.ones(48) / 48, 'same')), np.sqrt(
        np.convolve(hi[:, 1] ** 2, np.ones(48) / 48, 'same'))
    assert np.corrcoef(e)[0, 1] > 0.8                                    # the cracks come at the same moments
    assert np.corrcoef(y.T)[0, 1] < 0.98                                 # two renders, not two copies


def test_played_once_it_fades_in_and_out():
    y, info = quilt.stitch(_takes(2, per=1), 8.0, dsp.rng(3), loop=False)
    assert y.shape == (8 * SR, 2) and abs(y[0]).max() < 1e-6 and abs(y[-1]).max() < 1e-3
    assert not info['loop'] and len(info['joints']) == len(info['pieces']) - 1
