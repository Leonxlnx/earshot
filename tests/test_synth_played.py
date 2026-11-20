"""earshot.synth.played (real instrument notes learned from public-domain recordings): the store, the choice of a
note, tuning, a clean start and end, damping instead of a cut, determinism. Fast tier: a few notes per instrument;
--slow plays every learned note once and checks that neighbours of a chromatic run are levelled."""
import numpy as np
import pytest

from earshot.forge import dsp
from earshot.synth import played

SR = dsp.SR
INSTR = sorted(played.index())


def _f0(y, hz):
    """the pitch near hz over the sustained part (from 30 ms after the loudest 5 ms, up to 1.2 s): the power-weighted
    mean frequency of the spectrum within 40 cents of its peak near hz and 20 dB under it (a string section's
    players differ by a few cents; for one partial this is the peak), as tools/notes_build.py measures it"""
    m = y.mean(1)
    k = dsp.n_of(0.005)
    e = np.convolve(m ** 2, np.ones(k) / k, 'same')
    s = m[int(np.argmax(e)) + dsp.n_of(0.03):][:dsp.n_of(1.2)]
    n = 1 << int(np.ceil(np.log2(max(8 * len(s), 1 << 18))))
    mag = 20 * np.log10(np.abs(np.fft.rfft(s * np.hanning(len(s)), n)) + 1e-12)
    f = np.fft.rfftfreq(n, 1 / SR)
    sel = np.nonzero(np.abs(1200 * np.log2(np.maximum(f, 1.0) / hz)) < 80)[0]
    j = int(sel[np.argmax(mag[sel])])
    near = sel[(np.abs(1200 * np.log2(f[sel] / f[j])) < 40) & (mag[sel] > mag[j] - 20.0)]
    p = 10 ** (mag[near] / 10)
    return float((p * f[near]).sum() / p.sum())


def _cents(y, midi):
    hz = 440.0 * 2 ** ((midi - 69) / 12)
    return 1200 * np.log2(_f0(y, hz) / hz)


def test_store_covers_the_warm_register():
    info = played.instruments()
    assert set(info) >= {'marimba', 'harp'}
    for name, v in info.items():
        lo, hi = v['range']
        assert lo <= 48 and hi >= 84, (name, v['range'])
        assert v['notes'] >= 8 and v['dynamics']


def test_no_note_is_moved_more_than_three_semitones():
    for inst in INSTR:
        for m in range(48, 85):
            for en in (-1.0, 0.0, 1.0):
                row = played.choose(inst, m, en, 0)
                assert abs(row['midi'] - m) <= played.SHIFT, (inst, m, row['midi'])
    with pytest.raises(ValueError):
        played.render(played.choose('marimba', 60), 60 + played.SHIFT + 1, dsp.rng(0))
    with pytest.raises(ValueError):
        played.note('marimba', 20)


@pytest.mark.parametrize('inst', INSTR)
def test_in_tune_clean_damped_and_deterministic(inst):
    m = 57                                        # A3: moved from its learned note, through the tuning correction
    y = played.note(inst, m, 0.0, None, 0)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    pk = np.abs(y).max()
    assert 0.01 < pk < 1.0
    assert abs(_cents(y, m)) < 5.0, (inst, _cents(y, m))
    assert np.abs(y[:3]).max() < 0.05 * pk                            # starts at (near) zero: no click
    assert np.abs(y[-dsp.n_of(0.005):]).max() < 0.01 * pk             # its natural end is faded
    assert np.abs(y.mean(0)).max() < 0.01 * pk                        # no DC
    e = np.abs(y).max(1)                  # the onset is immediate (a low bar's resonator peaks later, after 0.1 s)
    assert int(np.argmax(e >= 0.1 * pk)) < dsp.n_of(0.02)
    d = played.note(inst, m, 0.0, 0.2, 0)                             # the same take, damped after 0.2 s
    assert dsp.n_of(0.2 + 0.055) <= len(d) <= dsp.n_of(0.2 + 0.16) < len(y)   # a 60-150 ms fade, never a cut
    assert np.array_equal(d[:dsp.n_of(0.2)], y[:dsp.n_of(0.2)])       # deterministic: same arguments, same samples
    assert np.abs(d[-dsp.n_of(0.005):]).max() < 0.02 * pk
    o = played.note(inst, m, 0.0, 0.2, 1)                             # another seed: another take
    assert not np.allclose(o[:dsp.n_of(0.2)], d[:dsp.n_of(0.2)])


def test_energy_plays_louder():
    soft = played.note('pizzicato', 64, -0.8, None, 0)
    hard = played.note('pizzicato', 64, 0.8, None, 0)
    assert played.loudest(hard) > played.loudest(soft) + 3.0


@pytest.mark.slow
def test_every_learned_note_plays_in_tune():
    for inst in INSTR:
        for row in played.index()[inst]['notes']:
            y = played.render(row, row['midi'], dsp.rng(0))
            assert abs(_cents(y, row['midi'])) < 5.0, row['file']
            assert np.isfinite(y).all() and np.abs(y).max() < 1.0


@pytest.mark.slow
def test_neighbours_are_levelled():
    """a chromatic run C4..C5: the loudest 50 ms of neighbouring notes within 2 dB"""
    for inst in INSTR:
        lv = [played.loudest(played.note(inst, m, 0.0, 0.3, 0)) for m in range(60, 73)]
        assert np.abs(np.diff(lv)).max() < 2.0, (inst, np.round(lv, 1))
