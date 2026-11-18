"""earshot.synth.design: the operations on takes of recordings (reverse, layer, move, fade, span) do what a sound
designer means by them, and takes stay inside the natural ranges (model-free; synthetic models)."""
import functools

import numpy as np

from earshot.forge import dsp
from earshot.synth import design, resyn

SR = dsp.SR


def _hit(f0=180.0, tau=0.12, at=0.05, n=0.6):
    """a synthetic struck thing: a click and a decaying partial, `at` s into a signal n s long"""
    k = int(n * SR)
    t = np.arange(k) / SR - at
    x = np.where(t >= 0, np.exp(-np.maximum(t, 0) / tau) * np.sin(2 * np.pi * f0 * np.maximum(t, 0)), 0.0)
    x[int(at * SR)] += 0.6
    return 0.4 * x


@functools.lru_cache(maxsize=None)
def _model():
    return resyn.analyze(_hit(), SR)


def _env(y, ms=2.0):
    k = max(1, int(ms / 1000 * SR))
    return np.sqrt(np.convolve(np.mean(np.atleast_2d(y.T).T ** 2, axis=1), np.ones(k) / k, 'same'))


def test_take_stays_in_the_natural_range():
    big = design.take(_model(), dsp.rng(1), size=3.0)
    assert np.array_equal(big, design.take(_model(), dsp.rng(1), size=design.SIZE[1]))
    long_ = design.take(_model(), dsp.rng(1), length=9.0)
    assert np.array_equal(long_, design.take(_model(), dsp.rng(1), length=design.LENGTH[1]))
    assert np.isfinite(big).all() and big.shape[1] == 2


def test_reverse_ends_on_the_strike():
    y = design.take(_model(), dsp.rng(2))
    z = design.reverse(y)
    e = _env(z)
    assert len(z) == len(y) and z.shape[1] == 2
    assert np.argmax(e) > 0.85 * len(z)                       # the loudest moment comes last
    assert np.abs(z[0]).max() < 1e-3 and np.abs(z[-1]).max() < 1e-3     # from silence, closed without a step
    assert e[: len(e) // 4].max() < 0.2 * e.max()             # it swells: quiet first


def test_layer_lines_up_the_onsets():
    a, b = _hit(at=0.05), _hit(f0=90.0, at=0.21)
    y, hit = design.layer([(a, 0.0, 0.0), (b, -6.0, 0.0)])
    assert abs(design.onset(y) - hit) < 0.003                 # one strike, not two
    assert abs(hit - max(design.onset(a), design.onset(b))) < 0.002
    y2, hit2 = design.layer([(a, 0.0, 0.0), (b, 0.0, 0.1)])
    e = _env(y2)
    late = int((hit2 + 0.1) * SR)
    assert e[late - 48:late + 96].max() > 2.0 * e[late - 480:late - 240].max()   # the second one 100 ms later
    rise = design.reverse(np.stack([a, a], axis=1))
    y3, hit3 = design.layer([(rise, 0.0, 0.0, design.end(rise)), (b, 0.0, 0.0)])
    assert abs(hit3 - design.end(rise)) < 1e-3 and len(y3) >= int((hit3 + 0.35) * SR)


def test_move_pans_across_and_falls_in_pitch():
    n = 2 * SR
    t = np.arange(n) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t) * np.exp(-((t - 1.0) / 0.5) ** 2)
    y = design.move(np.stack([tone, tone], axis=1), dsp.rng(1), travel=1.0, fall=2.0, side=-1.0)
    half = len(y) // 2
    lean = [np.log(np.mean(p[:, 1] ** 2) / np.mean(p[:, 0] ** 2))
            for p in (y[half - SR // 2:half], y[half:half + SR // 2])]
    assert lean[0] < -0.5 and lean[1] > 0.5                   # from the left to the right
    def pitch(seg):
        P = np.abs(np.fft.rfft(seg.mean(axis=1) * np.hanning(len(seg)), 1 << 18))
        return np.fft.rfftfreq(1 << 18, 1 / SR)[np.argmax(P)]
    before, after = pitch(y[half - SR // 2:half - SR // 4]), pitch(y[half + SR // 4:half + SR // 2])
    assert 1.0 < 12 * np.log2(before / after) < 2.5           # it falls by about the two semitones asked for
    e = _env(y, 50.0)
    assert abs(np.argmax(e) / len(e) - 0.5) < 0.1            # closest in the middle, where it was loudest


def test_fades_and_spans():
    y = np.ones((SR, 2))
    f = design.fade(y, 0.0, 1.0, 'db')
    assert abs(20 * np.log10(f[SR // 2, 0]) + 30.0) < 1.0     # straight in dB: half way is 30 dB down
    p = design.fade(y, 0.5, 0.0, 'power')
    assert abs(p[SR // 4, 0] - np.sin(np.pi / 4)) < 0.01
    s = design.span(y, 0.25, 0.5, fin=0.01, fout=0.01)
    assert len(s) == SR // 4 and s[0, 0] == 0.0 and s[-1, 0] == 0.0 and s[len(s) // 2, 0] == 1.0
    assert len(design.pad(y, 0.1)) == SR + SR // 10


def _noise_decay(n=1.0, tau=0.25, seed=3):
    """a synthetic burst of air: white noise dying away with time constant tau"""
    r = np.random.default_rng(seed)
    t = np.arange(int(n * SR)) / SR
    return 0.3 * r.standard_normal(len(t)) * np.exp(-t / tau)


def _centroid(y):
    P = np.abs(np.fft.rfft(np.atleast_2d(y.T).T.mean(axis=1))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float((P * f).sum() / P.sum())


def test_stretch_draws_air_out_beyond_the_natural_range():
    m = resyn.analyze(_noise_decay(), SR)
    a = design.take(m, dsp.rng(1))
    b = design.stretch(m, dsp.rng(1), 2.6)
    assert 2.0 < len(b) / len(a) < 3.2 and np.isfinite(b).all() and np.abs(b).max() <= 0.98
    assert np.array_equal(design.stretch(m, dsp.rng(1), 1.2), design.take(m, dsp.rng(1), length=1.2))


def test_opening_and_contour_make_a_build():
    y = design.reverse(np.stack([_noise_decay()] * 2, axis=1))
    z = design.contour(design.opening(y, 800.0), -24.0)
    q = np.array_split(z, 4)
    lv = [float(np.sqrt(np.mean(p ** 2))) for p in q]
    assert lv[0] < lv[1] < lv[2] < lv[3] and 15.0 < 20 * np.log10(lv[3] / lv[0]) < 30.0     # a straight line in dB
    assert _centroid(q[3]) > 1.5 * _centroid(q[0])                                          # the top arrives last


def test_halt_slows_pitch_and_time_together_to_silence():
    t = np.arange(int(1.0 * SR)) / SR
    y = np.stack([0.5 * np.sin(2 * np.pi * 440.0 * t)] * 2, axis=1)
    z = design.halt(y, 0.6)
    assert 1.25 < len(z) / SR < 1.45                              # 0.77 s as it was, then 0.6 s running down
    a, b = z[int(0.1 * SR):int(0.3 * SR), 0], z[-int(0.35 * SR):-int(0.15 * SR), 0]
    zc = [np.sum(np.abs(np.diff(np.sign(x))) > 0) / len(x) for x in (a, b)]
    assert zc[1] < 0.6 * zc[0]                                   # the tone sinks as the reading slows
