"""chip engine: band-limited channels, the hardware character, every generator clean, in tune and at one loudness"""
import time

import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp
from earshot.forge.polish import clicks, lufs_max
from earshot.synth import chip

SR = 48000
NAMES = list(chip.SOUNDS)


def folded_db(y, f0):
    """energy off the harmonic grid of f0 (what a naive oscillator folds back), dB re the total"""
    S = np.abs(np.fft.rfft(y * signal.windows.blackmanharris(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    keep = f < 30
    for k in range(1, int(SR / 2 / f0) + 1):
        keep |= np.abs(f - k * f0) <= 6 * f[1]
    return 10 * np.log10(S[~keep].sum() / S.sum())


def peak_cents(y):
    """the strongest partial (Welch, parabolic peak) and its distance to the nearest equal-tempered note, cents"""
    m = y.mean(1)
    f, P = signal.welch(m, SR, nperseg=min(len(m), 32768), nfft=1 << 17)
    P[f < 50] = 0
    i = int(np.argmax(P))
    a, b, c = np.log(P[i - 1:i + 2] + 1e-30)
    fp = f[i] + 0.5 * (a - c) / (a - 2 * b + c) * f[1]
    mm = 69 + 12 * np.log2(fp / 440)
    return 100 * (mm - round(mm))


def amp_at(y, fr):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return S[int(round(fr * len(y) / SR))]


@pytest.mark.parametrize('f0', [3009.0, 2960.0])
def test_channels_do_not_alias(f0):
    F = np.full(SR // 2, f0)
    for d in (0.125, 0.25, 0.5):
        assert folded_db(chip.pulse(F, d)[1000:], f0) < -60
    assert folded_db(chip.stair(F)[1000:], f0) < -60
    assert folded_db(chip.stair(F, chip.WAVES['saw'])[1000:], f0) < -60
    ph = (np.arange(SR // 2) * f0 / SR) % 1
    assert folded_db(np.where(ph < 0.5, 1.0, -1.0)[1000:], f0) > -30          # the naive square the test would catch


def test_duty_cycles_shape_the_spectrum():
    F = np.full(SR // 2, 500.0)
    y = chip.pulse(F, 0.25)[1000:]
    assert amp_at(y, 2000) < amp_at(y, 500) * 10 ** (-40 / 20)            # 25 %: every 4th harmonic missing
    y = chip.pulse(F, 0.125)[1000:]
    assert amp_at(y, 4000) < amp_at(y, 500) * 10 ** (-40 / 20)            # 12.5 %: every 8th
    assert amp_at(y, 1000) > amp_at(y, 500) * 0.8                         # and a thin, bright stack below it
    assert abs(chip.pulse(F, 0.125).mean()) < 1e-3                        # zero mean at any duty


def test_triangle_is_a_16_level_staircase():
    y = chip.stair(np.full(SR // 2, 110.0))[1000:]
    a1 = amp_at(y, 110)
    assert amp_at(y, 31 * 110) > a1 / 100 > 3 * a1 / 31 ** 2              # stair images near 32 f (the buzz, -30 dB),
    assert amp_at(y, 33 * 110) > a1 / 100                                  # a true triangle has -60 dB there
    assert amp_at(y, 220) < amp_at(y, 110) * 1e-3                          # still a triangle: odd harmonics only
    assert set(chip.TRIANGLE) == set(range(16)) and len(chip.TRIANGLE) == 32


def test_noise_modes():
    clock = chip.NOISE_CLOCKS[7]
    lag = int(round(93 * SR / clock))
    for mode, lo, hi in (('short', 0.8, 1.01), ('long', -0.3, 0.3)):
        y = chip.lfsr(np.full(SR // 2, clock), mode)[1000:]
        y = y - y.mean()
        c = np.dot(y[:-lag], y[lag:]) / np.dot(y, y)
        assert lo < c < hi, mode                                           # short mode repeats every 93 clocks
    assert np.allclose(chip.snap_clock([chip.NOISE_CLOCKS[3] * 1.1]), chip.NOISE_CLOCKS[3])


def test_parameter_model():
    v = {'freq': 440.0, 'arp': (0, 12), 'sustain': 0.1, 'decay': 0.01, 'steps': False}
    y = chip.voice(v)
    up = lambda a, b: np.sum(np.diff(np.sign(y[int(a * SR):int(b * SR)])) > 0)
    assert up(0.003, 0.015) * 2 == pytest.approx(up(0.020, 0.032), abs=2)   # the step lands one frame in
    y = chip.voice({'freq': 800.0, 'slide': -100.0, 'freq_min': 200.0, 'sustain': 1.0, 'decay': 0.1})
    assert len(y) / SR == pytest.approx(24 / 100, abs=0.01)                  # two octaves at 100 st/s, then stops
    e = chip._env(dict(chip.DEFAULTS, sustain=0.1, decay=0.3), np.arange(dsp.n_of(0.4)) / SR)
    mid = e[(np.arange(len(e)) % 200) == 100]                              # mid-way between the 240 Hz updates
    assert np.allclose(mid * 15, np.round(mid * 15), atol=1e-6)            # 16 volume steps
    y = chip.render({'freq': 300.0, 'sustain': 0.2, 'decay': 0.0}, polish=0.0, bits=4)[300:-300, 0]
    assert len(np.unique(np.round(y, 9))) <= 15                            # a 4-bit output


def test_names_and_notes():
    assert chip.midi('A4') == 69 and chip.midi('C#5') == 73 and chip.midi('Bb3') == 58 and chip.midi(261.0) == 60
    assert set(chip.GENERATORS) == set(chip.SOUNDS)
    for info in chip.SOUNDS.values():
        assert 2 <= len(info['says']) <= 3 and info['range'][0] <= info['dur_s'] <= info['range'][1]
    with pytest.raises(KeyError):
        chip.render('kazoo')


def test_deterministic():
    a, b = chip.render('explosion', 4), chip.render('explosion', 4)
    assert a.shape[1] == 2 and a.dtype == np.float64 and np.array_equal(a, b)
    assert not np.array_equal(chip.render('coin', 1), chip.render('coin', 2))


@pytest.mark.parametrize('name', NAMES)
def test_generator_over_seeds(name):
    info = chip.SOUNDS[name]
    lo, hi = info['range']
    for s in range(1, 13):                       # 20 seeds pass too (about 20 s for the file)
        y = chip.render(name, s)
        assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
        pk = np.abs(y).max()
        assert 0.1 < pk <= chip.PEAK + 1e-9
        assert lo <= len(y) / SR <= hi, (s, len(y) / SR)
        if info.get('loop'):
            seam = np.concatenate([y[-480:], y[:480]])
            assert clicks(seam)['clicks'] == 0 and np.abs(np.diff(seam, axis=0)).max() < 0.25 * pk
        else:
            assert np.abs(y[0]).max() < 1e-3 * pk and np.abs(y[-1]).max() < 1e-3 * pk
            assert np.abs(y[:10]).max() < 0.2 * pk                             # starts with a ramp, not a step
            assert clicks(y)['edge_db'] < -60


@pytest.mark.parametrize('name', [n for n in NAMES if chip.SOUNDS[n]['pitched']])
def test_tonal_generators_in_tune(name):
    for s, p in ((1, 'C5'), (2, 'G4'), (3, 'A5')):
        assert abs(peak_cents(chip.render(name, s, pitch=p, polish=0.0))) < 15, (s, p)


def test_one_loudness_and_speed():
    L, T = [], []
    for name in NAMES:
        chip.render(name, 1)
        t0 = time.perf_counter()
        y = chip.render(name, 1)
        T.append(time.perf_counter() - t0)
        L.append(lufs_max(y))
    assert max(L) - min(L) < 2.5 and abs(np.median(L) - chip.LOUD_LUFS) < 0.5
    assert max(T) < 1.5        # about 0.1 s on one thread; the bound only catches a render gone wrong


def test_dur_and_pitch():
    y = chip.render('laser', 1, dur=0.6)
    assert 0.6 <= len(y) / SR <= 0.9
    y = chip.render('engine', 1, dur=1.0)
    assert len(y) == SR
    a, b = chip.render('jump', 2, pitch='C4', polish=0.0), chip.render('jump', 2, pitch='C5', polish=0.0)
    ca = signal.welch(a.mean(1), SR, nperseg=4096)
    cb = signal.welch(b.mean(1), SR, nperseg=4096)
    cen = lambda w: np.sum(w[0] * w[1]) / np.sum(w[1])
    assert cen(cb) > 1.4 * cen(ca)
