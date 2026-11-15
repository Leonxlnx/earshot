"""Fast, model-free tests for earshot/synth/strings.py: the interface (shape, level, determinism, clean start and end,
errors) and the physics each note promises, measured on the rendered samples: stretched partials, a partial's own
decay, the piano's two-stage decay, a shorter hammer contact and a brighter note when played harder, the damper."""
import time

import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp
from earshot.synth import strings as S

SR = 48000
CHECKED = ('felt_piano', 'grand_piano', 'electric_piano', 'harp', 'nylon_guitar', 'pizzicato', 'muted_guitar')


def mono(y):
    return y.mean(axis=1)


def centroid(y, seconds=0.5):
    seg = np.pad(mono(y), (0, SR))[:int(seconds * SR)]
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return float((F * f).sum() / F.sum())


def partial_hz(y, guess, span=0.03):
    """the strongest spectral peak within +-span of guess Hz, from 0.05 .. 0.75 s"""
    seg = mono(y)[int(0.05 * SR):int(0.75 * SR)]
    nf = 1 << 20
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nf))
    f = np.fft.rfftfreq(nf, 1 / SR)
    lo, hi = np.searchsorted(f, [guess * (1 - span), guess * (1 + span)])
    j = lo + int(np.argmax(F[lo:hi]))
    a, b, c = np.log(F[j - 1:j + 2])
    return f[j] + 0.5 * (a - c) / (a - 2 * b + c) * f[1], F[j]


def band_db(y, lo, hi):
    sos = signal.butter(4, [lo, hi], 'bandpass', fs=SR, output='sos')
    return 20 * np.log10(np.abs(signal.hilbert(signal.sosfiltfilt(sos, mono(y)))) + 1e-12)


def onset(y, f):
    """seconds until the band around f Hz comes within 12 dB of its loudest"""
    db = band_db(y, 0.97 * f, 1.03 * f)
    return float(np.argmax(db > db.max() - 12.0)) / SR


def rate(db, t0, t1):
    """decay in dB per second of a level curve between t0 and t1 s (straight-line fit)"""
    i = np.arange(int(t0 * SR), int(t1 * SR), 48)
    return -np.polyfit(i / SR, db[i], 1)[0]


# ---------------------------------------------------------------------------------------------------- interface
def test_table_names_every_instrument():
    need = {'felt_piano', 'grand_piano', 'electric_piano', 'harp', 'nylon_guitar', 'steel_guitar', 'muted_guitar',
            'pizzicato', 'koto', 'upright_bass', 'harpsichord', 'dulcimer'}
    assert need <= set(S.INSTRUMENTS)
    for name, info in S.INSTRUMENTS.items():
        lo, hi = (S._pitch(p) for p in info['range'])
        assert lo < S._pitch(info['default_pitch']) < hi, name
        assert 2 <= len(info['says']) <= 3 and info['ring_s'] > 0, name


@pytest.mark.parametrize('name', sorted(S.INSTRUMENTS))
def test_every_instrument_gives_a_clean_note(name):
    y = S.note(name, vel=0.7, dur=0.25)
    assert y.dtype == np.float64 and y.ndim == 2 and y.shape[1] == 2 and len(y) > 0.25 * SR
    assert np.isfinite(y).all()
    assert abs(np.abs(y).max() - 0.63) < 1e-9                       # the level law: 0.9 * vel
    assert np.abs(y[0]).max() < 1e-6 and np.abs(y[1]).max() < 0.02  # starts at zero, without a step
    assert np.abs(y[-1]).max() < 1e-6                               # the end is faded, not cut
    assert abs(mono(y).mean()) < 2e-3                               # no DC
    assert np.array_equal(y, S.note(name, vel=0.7, dur=0.25))       # the same arguments, the same samples


def test_level_follows_velocity_and_nothing_is_normalised():
    for vel in (0.2, 1.0):
        assert abs(np.abs(S.note('felt_piano', 'C4', vel=vel, dur=0.3)).max() - 0.9 * vel) < 1e-9
    assert abs(np.abs(S.chord('harp', ['C4', 'E4', 'G4'], vel=0.5, dur=0.3)).max() - 0.45) < 1e-9


def test_pitch_as_name_midi_or_hz():
    a = S.note('nylon_guitar', 'A3', dur=0.2)
    assert np.array_equal(a, S.note('nylon_guitar', 57, dur=0.2))
    assert np.array_equal(a, S.note('nylon_guitar', 220.0, dur=0.2))
    assert abs(partial_hz(S.note('harp', 'A4', dur=1.0), 440.0)[0] - 440.0) < 0.4


def test_errors_say_what_is_valid():
    with pytest.raises(ValueError, match='felt_piano'):
        S.note('kazoo', 'C4')
    with pytest.raises(ValueError, match='bright.*hard.*ring'):
        S.note('harp', 'C4', sparkle=0.5)
    with pytest.raises(ValueError, match='E2'):
        S.note('nylon_guitar', 'C8')
    with pytest.raises(ValueError):
        S.note('harp', 'C4', tremolo=0.5)                           # only the electric pianos have one


# ------------------------------------------------------------------------------------------------------ physics
def test_piano_partials_are_stretched_by_the_string_stiffness():
    y = S.note('grand_piano', 'C4', vel=0.9, dur=1.0)
    f1 = partial_hz(y, 261.63)[0]
    assert abs(1200 * np.log2(f1 / 261.63)) < 1.5                   # the note is in tune
    k = np.array([2, 4, 6, 8, 10])
    fk = np.array([partial_hz(y, f1 * j * np.sqrt(1 + 3.8e-4 * j * j), 0.012)[0] for j in k])
    B = np.polyfit(k * k, (fk / k) ** 2, 1)
    assert 3.0e-4 < B[0] / B[1] < 4.6e-4                            # Chaigne & Askenfelt 1994: 3.8e-4 at C4
    assert fk[-1] / (10 * f1) > 1.012                               # partial 10 is a fifth of a semitone sharp


def test_guitar_and_harp_strings_are_far_less_stiff_than_piano_wire():
    for name, pitch in (('nylon_guitar', 'E3'), ('harp', 'C4')):
        y = S.note(name, pitch, vel=0.9, dur=1.0)
        f1 = partial_hz(y, S._pitch(pitch))[0]
        assert 1.0 <= partial_hz(y, 8 * f1, 0.01)[0] / (8 * f1) < 1.004, name


def test_higher_partials_die_faster():
    for name, pitch in (('grand_piano', 'C4'), ('nylon_guitar', 'A3'), ('harp', 'G4')):
        f0 = S._pitch(pitch)
        y = S.note(name, pitch, vel=0.9)
        slow = rate(band_db(y, 0.9 * f0, 1.1 * f0), 0.05, 0.8)
        fast = rate(band_db(y, 8.5 * f0, 14.5 * f0), 0.05, 0.35)
        assert fast > 1.5 * slow > 0, (name, slow, fast)


def test_piano_note_has_a_prompt_sound_and_an_aftersound():
    y = S.note('grand_piano', 'C4', vel=0.7)
    assert len(y) > 8 * SR                                          # it rings for many seconds
    db = band_db(y, 246, 278)
    prompt, after = rate(db, 0.1, 1.0), rate(db, 4.0, 8.0)
    assert 8 < prompt < 25                                          # about 10 dB per second at first (Martin 1947)
    assert after < 0.5 * prompt                                     # then a slower stage (Weinreich 1977)
    assert db[int(0.1 * SR)] - db[int(4.0 * SR)] < 0.8 * 3.9 * prompt   # one exponential would have fallen further


def test_hammer_contact_shortens_with_speed_and_towards_the_treble():
    soft, loud = S.contact_ms('grand_piano', 'C4', 0.2), S.contact_ms('grand_piano', 'C4', 1.0)
    assert 1.2 < loud < soft < 4.0                                  # about 2 ms at middle C
    assert S.contact_ms('grand_piano', 'C7', 0.5) < 1.2 < 3.0 < S.contact_ms('grand_piano', 'C2', 0.5)
    assert S.contact_ms('felt_piano', 'C4', 0.5) > S.contact_ms('grand_piano', 'C4', 0.5)


@pytest.mark.parametrize('name', CHECKED)
def test_louder_is_brighter(name):
    soft, loud = (centroid(S.note(name, vel=v, dur=0.6)) for v in (0.2, 1.0))
    assert loud > 1.08 * soft, (soft, loud)


def test_damper_stops_the_string_in_tens_of_milliseconds():
    free = S.note('felt_piano', 'C4', vel=0.6)
    held = S.note('felt_piano', 'C4', vel=0.6, dur=0.5)
    assert len(held) < 0.3 * len(free)
    a, b = band_db(np.pad(held, ((0, SR), (0, 0))), 200, 2000), band_db(free, 200, 2000)
    assert np.allclose(held[:int(0.49 * SR)], free[:int(0.49 * SR)])          # the same note until the key is let go
    assert a[int(0.48 * SR)] - a[int(0.8 * SR)] > 30                # 0.3 s later it is gone
    assert b[int(0.48 * SR)] - b[int(0.8 * SR)] < 8                 # which the free note is not
    step = np.abs(np.diff(mono(held)))[int(0.495 * SR):int(0.52 * SR)].max()
    assert step < 1.5 * np.abs(np.diff(mono(held)))[int(0.4 * SR):int(0.49 * SR)].max()    # no click as it lands
    assert len(S.note('grand_piano', 'C7', dur=0.2)) > 0.7 * SR     # the top of a piano has no dampers


def test_muted_and_pizzicato_are_short_and_a_harp_rings():
    assert len(S.note('muted_guitar')) < 0.8 * SR
    assert len(S.note('pizzicato')) < 1.6 * SR
    assert len(S.note('harp', 'C3')) > 3 * SR


# ----------------------------------------------------------------------------------------------------- controls
def test_controls_move_what_they_name():
    base = S.note('nylon_guitar', 'A3', dur=0.6)
    assert centroid(S.note('nylon_guitar', 'A3', dur=0.6, bright=0.8)) > 1.1 * centroid(base)
    assert centroid(S.note('nylon_guitar', 'A3', dur=0.6, hard=0.8)) > 1.05 * centroid(base)
    assert centroid(S.note('nylon_guitar', 'A3', dur=0.6, position=-0.9)) > 1.1 * centroid(base)   # at the bridge
    felt = centroid(S.note('felt_piano', 'C4', dur=0.6))
    assert centroid(S.note('felt_piano', 'C4', dur=0.6, hard=0.8)) > 1.05 * felt
    short, usual, long = (len(S.note('harp', 'G4', ring=v)) for v in (-0.8, 0.0, 0.8))
    assert long > 1.4 * usual > 1.96 * short
    flat = S.note('felt_piano', 'C4', dur=0.4, width=-1.0)
    assert np.allclose(flat[:, 0], flat[:, 1])                      # width -1 is mono


def test_stereo_is_a_pair_of_microphones_and_folds_to_mono():
    for name in CHECKED:
        y = S.note(name, dur=1.0)
        c = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
        assert 0.55 < c < 0.995, (name, c)
        lost = 10 * np.log10(np.mean(mono(y) ** 2) / np.mean(y ** 2))
        assert lost > -1.5, (name, lost)                            # the mono sum keeps the note


def test_vary_changes_the_take_but_never_the_pitch():
    a = S.note('grand_piano', 'A3', dur=0.8, vary=1.0, r=dsp.rng(1))
    b = S.note('grand_piano', 'A3', dur=0.8, vary=1.0, r=dsp.rng(2))
    assert not np.allclose(a[:20000], b[:20000], atol=1e-3)
    assert abs(1200 * np.log2(partial_hz(a, 220.0)[0] / partial_hz(b, 220.0)[0])) < 0.5
    assert np.abs(a).max() <= 0.63 + 1e-9 and np.abs(b).max() <= 0.63 + 1e-9


def test_electric_piano_barks_when_hit_and_its_tremolo_trades_sides():
    f0 = S._pitch('C3')
    def second(vel):
        y = S.note('electric_piano', 'C3', vel=vel, dur=1.0)
        return partial_hz(y, 2 * f0)[1] / partial_hz(y, f0)[1]
    assert second(1.0) > 2.5 * second(0.15)                         # the pickup's curve: harmonics grow with the swing
    y = S.note('electric_piano', 'C4', dur=1.5, tremolo=1.0)
    k = SR // 100
    e = np.sqrt(signal.fftconvolve(y ** 2, np.ones((k, 1)) / k, mode='same'))[int(0.2 * SR):int(1.4 * SR):48]
    ratio = np.log(e[:, 0] / e[:, 1])
    F = np.abs(np.fft.rfft((ratio - ratio.mean()) * np.hanning(len(ratio))))
    assert 4.0 <= np.fft.rfftfreq(len(ratio), 1e-3)[np.argmax(F)] <= 6.5 and np.ptp(ratio) > 1.0


# -------------------------------------------------------------------------------------------------------- chords
def test_chord_is_rolled_in_the_order_given():
    y = S.chord('harp', ['C4', 'D4', 'F#4', 'B4'], vel=0.6, strum_s=0.3, dur=1.0, r=dsp.rng(3))
    assert y.shape[1] == 2 and np.isfinite(y).all() and abs(np.abs(y).max() - 0.54) < 1e-9
    assert np.array_equal(y, S.chord('harp', ['C4', 'D4', 'F#4', 'B4'], vel=0.6, strum_s=0.3, dur=1.0, r=dsp.rng(3)))
    first = [onset(y, f) for f in (261.63, 293.66, 369.99, 493.88)]
    assert first == sorted(first) and 0.2 < first[-1] - first[0] < 0.4
    down = S.chord('harp', ['C4', 'D4', 'F#4', 'B4'], vel=0.6, strum_s=-0.3, dur=1.0, r=dsp.rng(3))
    assert onset(down, 493.88) < onset(down, 261.63)
    block = S.chord('felt_piano', ['C3', 'G3', 'E4'], vel=0.5, dur=0.5)
    assert len(block) < 1.5 * SR                                    # dur releases every note together


def test_a_four_second_piano_note_renders_fast():
    S.note('grand_piano', 'C4', dur=4.0)
    best = 9.0
    for _ in range(3):
        t = time.perf_counter()
        y = S.note('grand_piano', 'C4', dur=4.0)
        best = min(best, time.perf_counter() - t)
    assert len(y) >= 4 * SR and best < 0.6, best
