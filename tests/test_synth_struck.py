"""Fast, model-free tests of the struck-instrument engine (earshot/synth/modal.py, earshot/synth/struck.py): shape,
determinism, clean start and end, level, and the physical facts the engine promises (partial ratios, per-mode decay
that shortens with frequency and pitch, a brighter spectrum from a harder or faster stroke, beating doublets, a
stereo picture that survives the mono sum)."""
import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp
from earshot.forge.knobs import hz
from earshot.synth import modal, struck

SR = 48000


def peak_near(x, f, tol=0.02, seconds=None):
    """(frequency, level dB) of the strongest spectral peak within +-tol of f"""
    n = len(x) if seconds is None else min(len(x), int(seconds * SR))
    nfft = 1 << int(np.ceil(np.log2(n * 4)))
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n), nfft))
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    band = np.nonzero((fr > f * (1 - tol)) & (fr < f * (1 + tol)))[0]
    k = band[np.argmax(spec[band])]
    return float(fr[k]), float(20 * np.log10(spec[k] + 1e-30))


def t60_of(x, f, win=0.05):
    """T60 of the partial at f from a line through the log envelope of its band (maximum down to -25 dB)"""
    w = int(win * SR)
    het = x * np.exp(-2j * np.pi * f * np.arange(len(x)) / SR)
    env = np.abs(np.lib.stride_tricks.sliding_window_view(het, w)[::w // 8] @ np.hanning(w))
    t = np.arange(len(env)) * (w // 8) / SR
    i0 = int(np.argmax(env))
    low = np.nonzero(env[i0:] < env[i0] * 10 ** (-25 / 20))[0]
    i1 = i0 + (low[0] if len(low) else len(env) - i0)
    return -60.0 / np.polyfit(t[i0:i1], 20 * np.log10(env[i0:i1] + 1e-30), 1)[0]


def centroid(x, seconds=0.1):
    n = min(len(x), int(seconds * SR))
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(2 * n)[n:])) ** 2
    return float((np.fft.rfftfreq(n, 1 / SR) * spec).sum() / spec.sum())


def test_catalogue():
    assert len(struck.INSTRUMENTS) == 22
    for name, entry in struck.INSTRUMENTS.items():
        lo, hi = (hz(p) for p in entry['range'])
        assert hi >= 2 * lo * 0.99 or name == 'tubular_bell', name
        assert lo <= hz(entry['default_pitch']) <= hi, name
        assert 2 <= len(entry['says']) <= 3 and entry['ring_s'] > 0 and entry['position'], name
    for name in ('marimba', 'xylophone', 'kalimba', 'steel_drum', 'wood_block', 'coin', 'glass_clink'):
        ring = len(struck.note(name)) / SR                           # ring_s is what the note does when left alone
        assert 0.7 * ring < struck.INSTRUMENTS[name]['ring_s'] < 1.4 * ring + 0.05, (name, ring)


@pytest.mark.parametrize('name', list(struck.INSTRUMENTS))
def test_every_instrument_is_clean(name):
    y = struck.note(name, None, 0.7, 0.3, r=dsp.rng(4))
    assert y.ndim == 2 and y.shape[1] == 2 and y.dtype == np.float64
    assert np.isfinite(y).all()
    assert 0.2 < np.abs(y).max() < 0.95                              # vel 0.7 of a 0.9 full stroke
    assert np.abs(y[0]).max() == 0.0 and np.abs(y[-1]).max() < 1e-4  # starts at zero, tail faded out
    assert abs(y.mean()) < 2e-3                                      # no DC
    assert len(y) < (0.3 + 1.2) * SR                                 # damped at 0.3 s, not left ringing
    assert np.array_equal(y, struck.note(name, None, 0.7, 0.3, r=dsp.rng(4)))


@pytest.mark.parametrize('name', ['glockenspiel', 'marimba', 'kalimba', 'hand_bell', 'coin'])
def test_level_follows_velocity(name):
    full = np.abs(struck.note(name, None, 1.0, 0.5)).max()
    half = np.abs(struck.note(name, None, 0.5, 0.5)).max()
    assert 0.7 < full <= 0.9 + 1e-4
    for seed in (5, 6):                                              # no take and no control setting clips
        assert np.abs(struck.note(name, None, 1.0, 0.5, r=dsp.rng(seed), bright=1.0, vary=1.0)).max() <= 0.98 + 1e-9
    assert 0.25 * full < half < 0.7 * full


def test_bad_names_list_the_valid_ones():
    with pytest.raises(ValueError, match='glockenspiel'):
        struck.note('theremin', 'C5')
    with pytest.raises(ValueError, match='bright'):
        struck.note('marimba', 'C4', sparkle=1.0)
    with pytest.raises(ValueError, match='motor'):
        struck.note('vibraphone', 'C4', buzz=1.0)


def test_pitch_forms_and_range():
    a = struck.note('glockenspiel', 'C6', dur=0.2)
    assert np.array_equal(a, struck.note('glockenspiel', 84, dur=0.2))        # a whole number 0 .. 127 is MIDI
    assert np.array_equal(a, struck.note('glockenspiel', hz('C6'), dur=0.2))  # any other number is Hz
    low = struck.note('glockenspiel', 'C5', dur=0.2)
    assert np.array_equal(low, struck.note('glockenspiel', 'C2', dur=0.2))    # moved up by octaves into C5 .. C8
    f, _ = peak_near(struck.note('marimba', 'A3', dur=1.0).mean(axis=1), 220.0)
    assert abs(f - 220.0) < 0.5


@pytest.mark.parametrize('name, pitch, ratios', [
    ('glockenspiel', 'C6', {2.71: 0.03, 5.15: 0.05}),                # measured bar; thin bar 2.756, 5.404
    ('marimba', 'C3', {4.0: 0.005, 9.9: 0.01}),
    ('vibraphone', 'F3', {3.99: 0.005, 10.2: 0.01}),
    ('xylophone', 'C5', {3.0: 0.005}),
    ('church_bell', 'C4', {0.5: 0.004, 1.183: 0.004, 2.0: 0.004, 3.011: 0.004, 4.166: 0.004}),
    ('tubular_bell', 'C5', {2.0: 0.004, 2.94: 0.02, 4.0: 0.04}),
    ('singing_bowl', 'A3', {2.735: 0.008, 5.047: 0.008}),
    ('hand_bell', 'C5', {3.0: 0.005}),
    ('kalimba', 'C5', {5.6: 0.01}),
])
def test_partials_sit_where_the_geometry_puts_them(name, pitch, ratios):
    x = struck.note(name, pitch, 0.9, 1.5, r=dsp.rng(1), position=0.3).mean(axis=1)
    f0 = hz(pitch)
    _, top = peak_near(x, f0 * min(1.0, min(ratios)), 0.01, 0.4)
    for ratio, tol in ratios.items():
        f, level = peak_near(x, f0 * ratio, tol * 1.5, 0.4)
        assert abs(f / (f0 * ratio) - 1) < tol, (name, ratio, f / f0)
        assert level > top - 60, (name, ratio, level - top)


def test_upper_modes_die_first_and_high_notes_die_faster():
    x = struck.note('glockenspiel', 'C6', 0.9, position=0.6).mean(axis=1)
    f0 = hz('C6')
    f2, _ = peak_near(x, 2.72 * f0, 0.02, 0.3)
    assert t60_of(x, f2, 0.02) < 0.5 * t60_of(x, f0)
    rings = [t60_of(struck.note('marimba', p).mean(axis=1), hz(p)) for p in ('C3', 'C4', 'C5', 'C6')]
    assert all(a > 1.4 * b for a, b in zip(rings, rings[1:])), rings
    assert 1.5 < rings[0] < 4.0 and 0.25 < rings[-1] < 0.7           # rosewood over tubes: seconds low, a blip high
    metal = t60_of(struck.note('vibraphone', 'C5').mean(axis=1), hz('C5'))
    assert metal > 5 * rings[2]                                      # aluminium rings far longer than rosewood


@pytest.mark.parametrize('name', ['glockenspiel', 'marimba', 'vibraphone', 'xylophone', 'kalimba', 'church_bell',
                                  'singing_bowl', 'steel_drum'])
def test_a_harder_or_faster_stroke_is_brighter(name):
    soft = centroid(struck.note(name, None, 0.25, 0.5).mean(axis=1))
    loud = centroid(struck.note(name, None, 1.0, 0.5).mean(axis=1))
    assert 1.03 * soft < loud < 3.0 * soft, (soft, loud)             # brighter, by less than an octave and a half
    dull = centroid(struck.note(name, None, 0.7, 0.5, hard=-1.0).mean(axis=1))
    sharp = centroid(struck.note(name, None, 0.7, 0.5, hard=1.0).mean(axis=1))
    assert sharp > 1.1 * dull, (dull, sharp)


def test_contact_physics():
    t1, t2 = modal.contact_time(0.02, 1e8, 1.0), modal.contact_time(0.02, 1e8, 2.0)
    assert abs(t2 / t1 - 2 ** -0.2) < 1e-9 and 2e-4 < t1 < 6e-4     # Hertz: duration ~ v^-1/5
    assert modal.contact_time(0.04, 1e8, 1.0) > t1 > modal.contact_time(0.02, 1e9, 1.0)
    felt = modal.contact_time(0.006, 2e10, 2.0, 2.5) / modal.contact_time(0.006, 2e10, 1.0, 2.5)
    assert abs(felt - 2 ** (-1.5 / 3.5)) < 1e-9                      # felt stiffens: v^-3/7
    g = modal.force_gain([100.0, 2000.0, 6000.0], 4e-4)
    assert abs(g[0] - 1) < 0.01 and g[0] > g[1] > g[2] and g[2] < 0.1
    rate = modal.decay_rate(np.array([500.0, 1000.0, 4000.0]), 'steel')
    assert np.all(np.diff(rate) > 0) and np.all(modal.decay_rate(1000.0, 'rosewood') > 10 * rate[1])


def test_ring_is_the_exact_response_to_the_force_pulse():
    f = np.array([440.0, 1210.0, 3333.0, 9100.0])
    rate = np.array([3.0, 20.0, 60.0, 200.0])
    amp = np.array([1.0, -0.6, 0.4, 0.3])
    tau, n = 3.7e-4, 2400
    y = modal.ring(f, rate, amp, n, tau).imag[0]
    steps = 64
    t = np.arange(n * steps) / (SR * steps)
    impulse = (amp[:, None] * np.exp(-rate[:, None] * t) * np.sin(2 * np.pi * f[:, None] * t)).sum(axis=0)
    u = np.arange(int(tau * SR * steps) + 1) / (SR * steps)
    pulse = np.sin(np.pi * u / tau) ** 1.5
    ref = np.convolve(impulse, pulse / pulse.sum())[:n * steps:steps]
    assert y[0] == 0.0 and np.abs(y - ref).max() < 2e-3 * np.abs(ref).max()
    long = modal.ring(f, rate, amp, SR, tau).imag[0]                 # the block synthesis equals the closed form
    k = np.arange(SR - 64, SR)
    gain = np.exp(-rate[0] * k / SR)
    assert np.abs(long[k]).max() < 1.2 * gain.max() and np.isfinite(long).all()
    assert np.allclose(long[:n], y, rtol=0, atol=1e-10)


def test_modes_above_the_band_are_dropped():
    y = modal.ring([1000.0, 20500.0, 23000.0, 30000.0], [5.0] * 4, [1.0] * 4, 4800).imag[0]
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    assert spec[fr > 22000].max() < 1e-3 * spec.max()
    assert 0.02 < spec[(fr > 20000) & (fr < 21000)].max() / spec.max() < 0.9      # faded, not gone


def test_damping_at_dur_keeps_the_timbre():
    free = struck.note('vibraphone', 'C4', 0.8)
    held = struck.note('vibraphone', 'C4', 0.8, 0.4)
    assert len(free) > 4 * SR and 0.45 * SR < len(held) < 1.0 * SR
    k = int(0.39 * SR)
    assert np.allclose(free[:k], held[:k])                           # untouched until the damper lands
    tail = np.abs(held[int(0.6 * SR):]).max()
    assert tail < 0.1 * np.abs(held[k - 2400:k]).max()               # 20 dB down 0.2 s later
    assert np.abs(np.diff(held[k:, 0])).max() < 2 * np.abs(np.diff(free[k - 4800:k, 0])).max()     # no click
    late = struck.note('vibraphone', 'C4', 0.8, 3.0)                 # a long hold is not faded before its damper
    assert np.allclose(free[:int(2.99 * SR)], late[:int(2.99 * SR)])


def test_round_bodies_beat():
    x = struck.note('singing_bowl', 'A3', 0.7, 8.0).mean(axis=1)[: 8 * SR]
    w = int(0.05 * SR)
    het = x * np.exp(-2j * np.pi * 220.0 * np.arange(len(x)) / SR)
    env = np.abs(np.lib.stride_tricks.sliding_window_view(het, w)[::w // 4] @ np.hanning(w))[20:560]
    flat = env / np.exp(np.polyval(np.polyfit(np.arange(len(env)), np.log(env), 1), np.arange(len(env))))
    assert flat.max() / flat.min() > 1.15                            # the pair of (2,0) modes beats
    turns = np.sum(np.diff(np.sign(np.diff(flat))) != 0)
    assert 2 <= turns <= 60                                          # a slow warble: 0.2 .. 4 Hz over 6.7 s
    y = struck.note('marimba', 'C4', 0.7, 1.0).mean(axis=1)
    het = y * np.exp(-2j * np.pi * hz('C4') * np.arange(len(y)) / SR)
    env = np.abs(np.lib.stride_tricks.sliding_window_view(het, w)[::w // 4] @ np.hanning(w))
    peak = int(np.argmax(env))
    assert 0.004 * SR < peak * (w // 4) + w // 2 < 0.12 * SR        # the resonator blooms after the strike
    assert np.all(np.diff(env[peak + 2:peak + 40]) < 0)              # then a bar decays without any beat


def test_stereo_survives_mono():
    for name in ('glockenspiel', 'marimba', 'church_bell', 'singing_bowl', 'kalimba', 'tubular_bell'):
        y = struck.note(name, None, 0.7, 1.5)
        corr = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
        assert 0.55 < corr < 0.98, (name, corr)
        f0 = hz(struck.INSTRUMENTS[name]['default_pitch'])
        f, mono = peak_near(y.mean(axis=1), f0, 0.01)
        left, right = peak_near(y[:, 0], f, 0.002)[1], peak_near(y[:, 1], f, 0.002)[1]
        assert mono > 20 * np.log10(0.5 * (10 ** (left / 20) + 10 ** (right / 20))) - 1.5, name
        narrow = struck.note(name, None, 0.7, 1.5, width=-1.0)
        assert np.allclose(narrow[:, 0], narrow[:, 1])
        wide = struck.note(name, None, 0.7, 1.5, width=1.0)
        assert np.corrcoef(wide[:, 0], wide[:, 1])[0, 1] < corr


def test_controls_do_what_they_say():
    base = struck.note('marimba', 'C4', 0.7)
    long, short = (struck.note('marimba', 'C4', 0.7, ring=v) for v in (1.0, -1.0))
    assert len(long) > 2 * len(base) > 4 * len(short)
    x0 = struck.note('glockenspiel', 'C6', 0.8, 0.5, position=-1.0).mean(axis=1)     # dead centre: no second mode
    x1 = struck.note('glockenspiel', 'C6', 0.8, 0.5, position=0.5).mean(axis=1)
    f0 = hz('C6')
    second = [peak_near(x, 2.72 * f0, 0.02, 0.2)[1] - peak_near(x, f0, 0.01, 0.2)[1] for x in (x0, x1)]
    assert second[1] > second[0] + 12
    big = struck.note('marimba', 'C3', 0.7, 0.8, size=1.0).mean(axis=1)
    small = struck.note('marimba', 'C3', 0.7, 0.8, size=-1.0).mean(axis=1)
    assert centroid(big, 0.3) < centroid(small, 0.3)
    assert centroid(struck.note('vibraphone', None, 0.7, 0.5, bright=1.0).mean(axis=1)) > \
        1.2 * centroid(struck.note('vibraphone', None, 0.7, 0.5, bright=-1.0).mean(axis=1))


def test_takes_vary_but_pitch_never_moves():
    a = struck.note('hand_bell', 'C5', 0.7, 1.0, r=dsp.rng(1), vary=0.8)
    b = struck.note('hand_bell', 'C5', 0.7, 1.0, r=dsp.rng(2), vary=0.8)
    assert a.shape != b.shape or not np.allclose(a, b)
    for ratio in (1.0, 3.0):
        fa, _ = peak_near(a.mean(axis=1), 523.25 * ratio, 0.004)
        fb, _ = peak_near(b.mean(axis=1), 523.25 * ratio, 0.004)
        assert abs(fa - fb) < 0.004 * 523.25 * ratio                 # within the split of the doublet


def test_options_and_chord():
    plain = struck.note('vibraphone', 'F4', 0.7, 2.0)
    motor = struck.note('vibraphone', 'F4', 0.7, 2.0, motor=5.0)
    w = int(0.02 * SR)

    def wobble(y):
        e = np.sqrt(np.convolve(y.mean(axis=1) ** 2, np.ones(w) / w, 'same'))[int(0.3 * SR):int(1.9 * SR):w]
        e = e / np.exp(np.polyval(np.polyfit(np.arange(len(e)), np.log(e), 1), np.arange(len(e))))
        return e.std()
    assert wobble(motor) > 5 * wobble(plain) and wobble(motor) > 0.08

    def top(y):
        return np.sqrt(np.mean(dsp.butter(y.mean(axis=1), 'high', 3000.0, 4)[int(0.02 * SR):int(0.2 * SR)] ** 2))
    assert top(struck.note('kalimba', 'G4', 0.8, 0.6, buzz=1.0)) > 3 * top(struck.note('kalimba', 'G4', 0.8, 0.6))
    one = struck.note('coin', None, 0.8)
    many = struck.note('coin', None, 0.8, bounce=0.7)
    assert len(many) > len(one) and np.isfinite(many).all() and np.abs(many).max() < 1.0
    env = np.convolve(np.abs(many).max(axis=1), np.ones(96) / 96, 'same')
    hits, _ = signal.find_peaks(env, height=0.02, distance=int(0.015 * SR), prominence=0.02)
    gaps = np.diff(hits)
    assert len(hits) >= 3 and gaps[1] < gaps[0]                       # it lands again and again, faster each time
    c = struck.chord('marimba', ['C3', 'E3', 'G3'], 0.8, 0.05)
    assert c.shape[1] == 2 and np.abs(c).max() <= 0.9 + 1e-9 and np.abs(c[0]).max() == 0.0
    solo = struck.note('marimba', 'G3', 0.8)
    assert len(c) >= len(solo) + int(0.1 * SR) - 2


# ----------------------------------------------------------------------------- laws calibrated against recordings
# Anechoic single notes of the University of Iowa MIS (marimba, vibraphone, xylophone) and notes of VSCO-2 CE
# (glockenspiel, marimba, xylophone): the [cal] comments in struck.py. No recording is needed to check them.
def mode_db(name, pitch, vel, ratio, tol=0.03):
    """level of the mode near ratio x the pitch in the first 100 ms, dB re the fundamental"""
    x = struck.note(name, pitch, vel, 1.0).mean(axis=1)
    return peak_near(x, ratio * hz(pitch), tol, 0.1)[1] - peak_near(x, hz(pitch), 0.01, 0.1)[1]


def test_tuned_bars_are_struck_near_the_middle():
    assert -20 < mode_db('marimba', 'C4', 0.9, 4.0) < -8             # recorded with yarn at ff: -13.5 dB
    assert -14 < mode_db('marimba', 'C3', 0.9, 4.0) < -3             # -8 dB
    assert mode_db('marimba', 'C3', 0.9, 9.9) > -8                   # mode 3 of a low bar stands as high as mode 1
    loud = mode_db('vibraphone', 'C4', 0.95, 3.99)
    assert -30 < loud < -14                                          # recorded at ff: -21 dB
    assert mode_db('vibraphone', 'C4', 0.95, 9.9) > loud             # and mode 3 above it: -4 dB
    assert mode_db('xylophone', 'C5', 0.9, 3.0) < -6                 # recorded: -21 dB (anechoic), -5 .. -37 (far)
    assert mode_db('xylophone', 'C6', 0.9, 5.9) > -32                # the hard mallet still reaches mode 3: -21 dB


def test_a_wound_mallet_stiffens_with_the_stroke():
    assert mode_db('marimba', 'C4', 0.9, 4.0) > mode_db('marimba', 'C4', 0.3, 4.0) + 8     # recorded mf to ff: 17 dB


def test_vibraphone_rings_long_to_the_top_and_its_tube_fills_in_some_20_periods():
    rings = [t60_of(struck.note('vibraphone', p, 0.5).mean(axis=1), hz(p)) for p in ('C4', 'C5', 'C6')]
    assert 15 < rings[0] < 27 and 11 < rings[1] < 21 and 9 < rings[2] < 18, rings     # recorded: 20, 15, 14 s
    for p, lo, hi in (('C4', 40, 95), ('C5', 22, 62)):               # recorded: the envelope peaks after 63, 41 ms
        x = struck.note('vibraphone', p, 0.5, 1.0).mean(axis=1)
        w = int(0.02 * SR)
        het = x * np.exp(-2j * np.pi * hz(p) * np.arange(len(x)) / SR)
        env = np.abs(np.lib.stride_tricks.sliding_window_view(het, w)[::w // 8] @ np.hanning(w))
        assert lo < (np.argmax(env) * (w // 8) + w // 2) / SR * 1e3 < hi, p


def test_glockenspiel_attack_is_its_upper_modes():
    x = struck.note('glockenspiel', 'C6', 0.6).mean(axis=1)
    f0 = hz('C6')
    assert abs(mode_db('glockenspiel', 'C6', 0.6, 2.71)) < 9         # recorded: mode 2 at -2 dB (C6), +5 dB (G6)
    assert mode_db('glockenspiel', 'C6', 0.6, 5.2, 0.04) > -9        # mode 3 at 0 dB
    f2, _ = peak_near(x, 2.71 * f0, 0.03, 0.3)
    assert 35 < 60 / t60_of(x, f2, 0.02) < 130                       # mode 2 falls 50 dB/s at C6, 90 at G6
    assert centroid(x, 0.02) > 4000 > 1.5 * centroid(x[int(0.3 * SR):], 0.1)      # 6 kHz at first, then the note
