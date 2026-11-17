"""earshot.synth.space and earshot.synth.master: the shared rooms (stereo out, a tail for the reverberant ones, the
room's own tone at TONE_DB under the peak that ends with the file, dry stays untouched, deterministic) and the last
stage of every made sound (refusals, DC, clean edges, tail trim, true peak, the loudness cap, a loudness target) and
the file writer (wav, flac, mono, the ffmpeg formats). Synthetic signals only, model-free."""
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from earshot.forge import dsp
from earshot.synth import master, space

SR = dsp.SR
WET = [s for s, p in space.SPACES.items() if p is not None]


def _burst(lead=0.2, after=0.3, seed=0):
    """stereo: `lead` s of silence, a 40 ms decaying noise burst, `after` s of silence"""
    r = dsp.rng(seed)
    n = dsp.n_of(0.04)
    b = r.standard_normal((n, 2)) * np.exp(-np.arange(n) / (0.008 * SR))[:, None] * 0.5
    return np.concatenate([np.zeros((dsp.n_of(lead), 2)), b, np.zeros((dsp.n_of(after), 2))])


def _tone(f=440.0, sec=0.5, amp=0.5, decay=0.15):
    t = np.arange(dsp.n_of(sec)) / SR
    return amp * np.sin(2 * np.pi * f * t) * np.exp(-t / decay)


def _db(x):
    return 20 * np.log10(x + 1e-30)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# ---- space.room

@pytest.mark.parametrize('name', list(space.SPACES))
def test_every_space_is_stereo(name):
    x = _burst(after=0.0)
    y = space.room(x[:, 0], name, r=dsp.rng(1))
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    if name in WET:
        assert len(y) > len(x)


@pytest.mark.parametrize('name', ['booth', 'room', 'studio'])
def test_a_room_ends_with_its_sound_not_with_the_input(name):
    """silence after the sound is not kept: the file ends 80 ms after the tail is 60 dB down"""
    x = _burst(after=0.3)
    y = space.room(x, name, r=dsp.rng(1))
    assert dsp.n_of(0.1) < len(y) < len(x) + dsp.n_of(1.0)
    assert np.abs(y[-1]).max() < 1e-6


def test_dry_and_amount_minus_one_return_the_input():
    x = _burst()
    assert np.array_equal(space.room(x, 'dry'), x)
    for name in WET:
        assert np.array_equal(space.room(x, name, amount=-1.0), x)
    m = x[:, 0]
    assert np.array_equal(space.room(m, 'dry'), np.stack([m, m], 1))


def test_unknown_space_is_refused():
    with pytest.raises(ValueError, match='studio'):
        space.room(_burst(), 'garage')


def _floor(name, seeds=(3, 4)):
    """(take, the room's floor alone, peak): the floor is the only part that changes with the generator, so two takes
    differ by it alone (their difference / sqrt 2 has its level)"""
    x = _burst(lead=0.3)
    a, b = space.room(x, name, r=dsp.rng(seeds[0])), space.room(x, name, r=dsp.rng(seeds[1]))
    assert len(a) == len(b)
    return a, (a - b) / np.sqrt(2), np.abs(a).max()


FLOOR_LOW = ('the floor sits 2.1-3.2 dB under TONE_DB (-70.1 .. -71.2 dB re the peak, before the sound and under the '
             'tail, booth / studio / hall): dsp.noise sets the pink noise to unit RMS before space.tone high-passes it '
             'at 120 Hz, which takes away about 40 % of its energy')


@pytest.mark.parametrize('name', ['booth', 'studio', 'hall'])
def test_room_tone_floor_before_the_sound(name):
    """before the burst arrives there is nothing but the room's floor: its RMS sits TONE_DB under the peak"""
    a, _, pk = _floor(name)
    for ch in range(2):
        floor = _db(_rms(a[dsp.n_of(0.02):dsp.n_of(0.28), ch]) / pk)
        assert abs(floor - space.TONE_DB) <= 2.0, (name, ch, floor)


@pytest.mark.parametrize('name', ['booth', 'studio', 'hall'])
def test_room_tone_floor_through_the_tail(name):
    """the floor alone, from the sound to the closing fade, is TONE_DB under the peak"""
    a, fl, pk = _floor(name)
    span = slice(dsp.n_of(0.3), len(a) - dsp.n_of(0.09))
    for ch in range(2):
        floor = _db(_rms(fl[span, ch]) / pk)
        assert abs(floor - space.TONE_DB) <= 2.0, (name, ch, floor)


@pytest.mark.parametrize('name', ['booth', 'studio', 'hall'])
def test_room_tone_lasts_to_the_end_and_ends_at_zero(name):
    a, fl, pk = _floor(name)
    near_end = fl[len(a) - dsp.n_of(0.12):len(a) - dsp.n_of(0.08)]
    assert _db(_rms(near_end) / pk) > space.TONE_DB - 8.0                   # still there at the tail's end
    assert np.abs(a[-1]).max() < 1e-9 and np.abs(fl[-dsp.n_of(0.002):]).max() < pk * 10 ** (space.TONE_DB / 20)


@pytest.mark.parametrize('name', WET)
def test_room_is_deterministic(name):
    x = _burst()
    assert np.array_equal(space.room(x, name, r=dsp.rng(5)), space.room(x, name, r=dsp.rng(5)))
    assert np.array_equal(space.room(x, name), space.room(x, name))           # the default generator is fixed too


def test_amount_lengthens_the_tail():
    x = _burst()
    lo, hi = space.room(x, 'hall', amount=-0.5, r=dsp.rng(1)), space.room(x, 'hall', amount=0.5, r=dsp.rng(1))
    assert len(hi) > len(lo)


# ---- master.finish

def test_finish_refuses_silent_and_non_finite():
    with pytest.raises(ValueError, match='silent'):
        master.finish(np.zeros(1000))
    with pytest.raises(ValueError, match='silent'):
        master.finish(np.zeros(0))
    x = _tone()
    x[100] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        master.finish(x)
    x[100] = np.inf
    with pytest.raises(ValueError, match='non-finite'):
        master.finish(x)


def test_finish_removes_dc():
    t = np.arange(SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.4
    y = master.finish(x)
    assert abs(y.mean()) < 1e-3 * np.abs(y).max()


@pytest.mark.parametrize('stereo', [False, True])
def test_finish_starts_and_ends_at_zero(stereo):
    x = np.full(dsp.n_of(0.3), 0.5) * np.sin(2 * np.pi * 300 * np.arange(dsp.n_of(0.3)) / SR + 1.0)
    x = master.stereo(x) if stereo else x
    y = master.finish(x)
    assert np.abs(y[0]).max() == 0.0
    assert np.abs(y[-1]).max() < 1e-6


def test_finish_drops_leading_silence_and_trims_a_silent_tail():
    x = np.concatenate([np.zeros(dsp.n_of(0.25)), _tone(sec=0.4, decay=0.05), np.zeros(SR)])
    y = master.finish(x)
    assert len(y) < dsp.n_of(0.45)
    assert np.argmax(np.abs(y) > 0.1 * np.abs(y).max()) < dsp.n_of(0.004)        # starts when the sound does


def test_finish_true_peak():
    for pk in (-1.0, -6.0):
        y = master.finish(_tone(decay=0.05), peak_db=pk)
        assert abs(master.true_peak_db(y) - pk) <= 0.2


def test_finish_holds_a_dense_sound_at_the_cap():
    x = dsp.rng(1).standard_normal((SR, 2)) * 0.3
    y = master.finish(x, cap=-14.0)
    assert abs(master.loudness(y) - (-14.0)) <= 0.5
    assert master.true_peak_db(y) < -1.0 + 0.2
    free = master.finish(x)
    assert master.loudness(free) > -14.0 + 1.0                        # the cap did something


def test_finish_at_a_loudness():
    x = dsp.rng(2).standard_normal(SR) * 0.1 * np.exp(-np.arange(SR) / (0.3 * SR))
    for target in (-20.0, -26.0):
        y = master.finish(x, lufs=target)
        assert abs(master.loudness(y) - target) <= 0.5
        assert master.true_peak_db(y) <= -1.0 + 0.2


def test_finish_does_not_bend_a_tone_to_reach_a_loudness():
    """a tonal figure asked for more loudness than its peak allows stops at its peak (no fold, no limiter: both
    leave a residual a held tone shows); quieter targets are met as before"""
    x = _tone(sec=0.6, decay=0.3)
    free = master.finish(x)
    y = master.finish(x, lufs=master.loudness(free) + 4.0, tonal=True)
    g = float((y * free[:len(y)]).sum() / (free[:len(y)] ** 2).sum())
    assert abs(master.true_peak_db(y) - (-1.0)) <= 0.2
    assert np.abs(y - g * free[:len(y)]).max() < 1e-6 * np.abs(y).max() + 1e-9    # the same take, only gain
    lo = master.finish(x, lufs=master.loudness(free) - 6.0, tonal=True)
    assert abs(master.loudness(lo) - (master.loudness(free) - 6.0)) <= 0.3


def test_loudest_window():
    y = np.zeros((SR, 2))
    y[1000:1000 + dsp.n_of(0.05)] = 0.5
    assert abs(master.loudest(y) - 20 * np.log10(0.5)) < 0.1
    assert abs(master.loudest(y, 0.005) - 20 * np.log10(0.5)) < 0.1


def test_finish_keeps_a_loop_whole():
    x = 0.3 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)
    y = master.finish(x, loop=True)
    assert len(y) == len(x) and np.abs(y[0]) == 0.0                 # sin(0); no fade, no trim
    assert np.abs(y[-1]) > 1e-3


# ---- master.write

def _stereo_take():
    x = master.stereo(_tone(sec=0.3, decay=0.08))
    x[:, 1] *= 0.5
    return x


@pytest.mark.parametrize('ext,bits,subtype', [('wav', 16, 'PCM_16'), ('wav', 24, 'PCM_24'), ('flac', 24, 'PCM_24'),
                                              ('flac', 16, 'PCM_16')])
def test_write_wav_and_flac_round_trip(tmp_path, ext, bits, subtype):
    x = _stereo_take()
    p = master.write(str(tmp_path / f'a.{ext}'), x, bits=bits)
    info = sf.info(p)
    assert info.samplerate == SR and info.channels == 2 and info.subtype == subtype
    y, sr = sf.read(p, always_2d=True)
    assert y.shape == x.shape
    assert abs(np.abs(y).max() - np.abs(x).max()) < 2.0 / 2 ** (bits - 1)


def test_write_other_sample_rate(tmp_path):
    pytest.importorskip('soxr')
    p = master.write(str(tmp_path / 'a.wav'), _stereo_take(), sr=44100)
    info = sf.info(p)
    assert info.samplerate == 44100 and abs(info.frames - round(dsp.n_of(0.3) * 44100 / SR)) <= 2


def test_write_mono_sums_and_keeps_the_peak(tmp_path):
    x = _stereo_take()
    p = master.write(str(tmp_path / 'm.wav'), x, mono=True)
    y, _ = sf.read(p)
    assert y.ndim == 1 and len(y) == len(x)
    assert abs(np.abs(y).max() - np.abs(x).max()) < 1e-4
    ref = x.mean(1) * np.abs(x).max() / np.abs(x.mean(1)).max()
    assert np.allclose(y, ref, atol=1e-4)


def test_write_makes_the_folder(tmp_path):
    p = master.write(str(tmp_path / 'deep' / 'er' / 'a.wav'), _stereo_take())
    assert os.path.isfile(p)


@pytest.mark.parametrize('ext', ['ogg', 'mp3', 'm4a', 'webm'])
def test_write_lossy_through_ffmpeg(tmp_path, ext):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is not on PATH')
    p = master.write(str(tmp_path / f'a.{ext}'), _stereo_take())
    assert os.path.isfile(p) and os.path.getsize(p) > 200
    if ext == 'ogg':
        info = sf.info(p)
        assert info.samplerate == SR and info.channels == 2


def test_write_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match='wav'):
        master.write(str(tmp_path / 'a.aiff'), _stereo_take())
    with pytest.raises(ValueError):
        master.write(str(tmp_path / 'a'), _stereo_take())
