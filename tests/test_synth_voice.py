"""The synthesizer voice (earshot.synth.voice) and its patch library (earshot.synth.patches): alias-free oscillators,
click-free envelopes, a ladder filter in tune, FM that goes pure, effects that survive mono, and the promises of
every patch (level, clean edges, velocity, register, controls, stereo). No models, no audio fixtures."""
import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp
from earshot.synth import patches, voice

SR = dsp.SR
ALL = list(patches.PATCHES)


def _off_grid_db(y, f0):
    """the strongest component that is not a harmonic of f0, in dB under the strongest one"""
    w = signal.windows.blackmanharris(len(y))
    S = np.abs(np.fft.rfft(y * w))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    off = (fr > 20) & (np.abs((fr + f0 / 2) % f0 - f0 / 2) > 6 * SR / len(y))
    return 20 * np.log10((S * off).max() / S.max() + 1e-15)


def _peak_hz(y, lo=20.0):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    S[fr < lo] = 0
    return fr[int(np.argmax(S))]


def _centroid(y):
    y = y.mean(axis=1) if y.ndim == 2 else y
    S = np.abs(np.fft.rfft(y)) ** 2
    return float((S * np.fft.rfftfreq(len(y), 1 / SR)).sum() / (S.sum() + 1e-20))


def _above(y, hz):
    """the share of the energy above hz"""
    y = y.mean(axis=1) if y.ndim == 2 else y
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return float(S[np.fft.rfftfreq(len(y), 1 / SR) > hz].sum() / (S.sum() + 1e-20))


def _corr(y):
    return float(np.corrcoef(y[:, 0], y[:, 1])[0, 1])


def _mono_loss_db(y):
    return 10 * np.log10(np.mean(y.mean(axis=1) ** 2) / np.mean(y ** 2))


# ------------------------------------------------------------------------------------------------ engine
@pytest.mark.parametrize('shape', ['saw', 'pulse', 'tri', 'sine'])
def test_oscillators_do_not_alias(shape):
    for f0 in (1244.51, 4186.01, 6271.93):
        y = voice.osc(f0, shape, pw=0.31, seconds=0.5)[2400:-2400]
        assert _off_grid_db(y, f0) < -70
        assert abs(y.mean()) < 1e-3


def test_hard_sync_and_fm_between_oscillators_stay_clean():
    assert _off_grid_db(voice.osc(523.25, 'saw', sync=3.3, seconds=0.5)[2400:-2400], 523.25) < -70
    mod = voice.osc(880.0, 'sine', seconds=0.5)
    y = voice.osc(880.0 * (1 + 0.5 * mod), 'saw')[2400:-2400]              # linear FM at ratio 1: a harmonic spectrum
    assert _off_grid_db(y, 880.0) < -70


def test_a_swept_saw_keeps_its_aliases_70_db_down():
    f = dsp.sweep(300, 9000, 2.0)
    y = voice.osc(f, 'saw')
    nf, worst = 8192, 0.0
    for s in range(0, len(y) - nf, nf):
        S = np.abs(np.fft.rfft(y[s:s + nf] * signal.windows.blackmanharris(nf)))
        fr = np.fft.rfftfreq(nf, 1 / SR)
        off = fr > 30
        for k in range(1, int(SR / 2 / f[s]) + 2):                         # everything a harmonic passed through
            off &= ~((fr > k * f[s] - 8 * SR / nf) & (fr < k * f[s + nf] + 8 * SR / nf))
        worst = max(worst, (S * off).max() / S.max())
    assert 20 * np.log10(worst + 1e-15) < -70


def test_unison_is_a_drifting_stack_around_the_pitch():
    f = np.full(SR, 440.0)
    a = voice.unison(f, 'saw', dsp.rng(1), voices=7, detune=20, spread=0.8, drift=2, smear=0.004)
    b = voice.unison(f, 'saw', dsp.rng(1), voices=7, detune=20, spread=0.8, drift=2, smear=0.004)
    c = voice.unison(f, 'saw', dsp.rng(2), voices=7, detune=20, spread=0.8, drift=2, smear=0.004)
    assert a.shape == (SR, 2) and np.array_equal(a, b) and not np.allclose(a, c)
    assert 0.6 < np.sqrt(np.mean(a ** 2) * 2) / np.sqrt(2 / 3) < 1.6        # about the power of one saw per channel
    assert _corr(a) < 0.9                                                  # the voices are spread
    assert abs(_peak_hz(a.mean(axis=1)) - 440.0) < 3.0                     # the centre does not move
    env = np.abs(signal.hilbert(dsp.butter(a.mean(axis=1), 'band', (400, 480))))[4800:-4800]
    assert env.std() / env.mean() > 0.15                                   # the stack beats; it is not one static wave


def test_envelope_starts_and_ends_at_zero_without_steps():
    e = voice.adsr(0.3, 0.0, 0.5, 0.6, 0.2)
    assert e[0] == 0.0 and e[-1] == 0.0 and e.min() >= 0.0 and 0.95 < e.max() <= 1.0
    assert np.abs(np.diff(e)).max() < 0.06                                 # the fastest attack still takes about 1 ms
    assert e[1] < 1e-2
    hold = voice.adsr(1.0, 0.01, 0.2, 0.7, 0.3)
    assert abs(hold[SR // 2] - 0.7) < 1e-6
    rel = hold[SR:SR + 14400]
    assert np.all(np.diff(rel) <= 1e-12) and rel[4800] < 0.7 * 0.25         # exponential: most of the fall comes first


def test_ladder_resonance_is_in_tune_and_stable():
    for fc in (880.0, 3520.0):
        x = np.zeros(SR)
        x[0] = 1e-3
        y = voice.ladder(x, fc, 1.05)[SR // 2:]
        assert abs(1200 * np.log2(_peak_hz(y) / fc)) < 5                   # self-oscillation at the cutoff
        assert 0.3 < np.abs(y).max() < 1.5                                 # and tame
    r = dsp.rng(3)
    w = r.standard_normal(2 * SR)
    f, P = signal.welch(voice.ladder(w, 1000.0), SR, nperseg=4096)
    g = lambda hz: 10 * np.log10(P[np.argmin(np.abs(f - hz))] / P[np.argmin(np.abs(f - 100))])
    assert -27 < g(4000) - g(2000) < -19                                   # 24 dB per octave
    y = voice.ladder(3.0 * r.standard_normal((SR // 2, 2)), 10 ** r.uniform(1.3, 4.33, SR // 2),
                     r.uniform(0, 1.1, SR // 2))
    assert np.isfinite(y).all() and np.abs(y).max() < 100


def test_fm_goes_from_bright_to_pure_without_aliases_or_dc():
    bell = [dict(out=1.0, mod=(1,), env=(0.001, 3.0, 0.0, 1.0)),
            dict(ratio=3.5, index=3.0, env=(0.001, 0.4, 0.0, 0.3))]
    y = voice.fm(880.0, bell, 1.0, 1.5)
    assert y.shape[1] == 2 and abs(y[0, 0]) < 1e-5 and np.isfinite(y).all()
    assert _centroid(y[:4800]) > 1.5 * _centroid(y[SR:SR + 9600])           # the index dies before the carrier
    assert abs(_centroid(y[SR:SR + 9600]) - 880.0) < 60                     # what is left is the carrier
    on = (0.001, 0.5, 1.0, 0.05)
    held = [dict(out=1.0, mod=(1,), env=on), dict(ratio=3.0, index=6.0, mod=(2,), env=on), dict(index=3.0, env=on)]
    for f0 in (523.25, 4186.01):
        assert _off_grid_db(voice.fm(f0, held, 1.0, 0.6)[4800:24000, 0], f0) < -70
    one = voice.fm(220.0, [dict(out=1.0, mod=(1,), env=on), dict(index=2.0, env=on)], 1.0, 1.0)[9600:43200, 0]
    assert abs(one.mean()) < 0.01 * one.std()                               # ratio 1:1 has a sideband at 0 Hz
    soft, loud = (voice.fm(220.0, [dict(out=1.0, mod=(1,)), dict(index=2.0, vel=1.0)], v, 0.2) for v in (0.3, 1.0))
    assert _centroid(loud) > _centroid(soft)
    with pytest.raises(ValueError):
        voice.fm(220.0, [dict(out=1.0), dict(index=1.0, mod=(0,))])


def test_effects_keep_the_mono_sum_and_decay():
    x = voice.unison(np.full(SR, 220.0), 'saw', dsp.rng(1), drift=2)
    for v in (2, 3):
        c = voice.chorus(x, voices=v, mix=0.6)
        assert c.shape == x.shape and _corr(c) < 0.95 and _mono_loss_db(c) > -1.5
    click = np.zeros((2400, 2))
    click[:240] = dsp.rng(2).standard_normal((240, 1))
    e = voice.echo(click, time=0.1, feedback=0.5, mix=1.0)
    first, third = e[4800:7200], e[14400:16800]
    assert np.abs(e[4800:7200, 0]).max() > 10 * np.abs(e[4800:7200, 1]).max()   # ping (left) ...
    assert np.abs(e[9600:12000, 1]).max() > 10 * np.abs(e[9600:12000, 0]).max()  # ... pong (right)
    assert _centroid(third) < _centroid(first) and np.abs(third).max() < 0.5 * np.abs(first).max()
    assert np.abs(e[-480:]).max() < 2e-3 * np.abs(e).max()
    assert np.isfinite(voice.phaser(x)).all()
    wide = voice.unison(np.full(SR, 55.0), 'saw', dsp.rng(1), 3, 20, 1.0)
    lo = dsp.butter(voice.mono_below(wide, 120.0), 'low', 90.0, order=6)
    assert _corr(lo[4800:]) > 0.99


# ------------------------------------------------------------------------------------------------ patches
def test_the_library_is_complete_and_described():
    assert len(ALL) == 28
    for name, p in patches.PATCHES.items():
        assert set(p) == {'says', 'kind', 'default_pitch', 'range'} and 2 <= len(p['says']) <= 3
        assert p['kind'] in ('pluck', 'key', 'pad', 'bass', 'lead', 'bell', 'hit', 'fx')
        lo, hi = (patches._hz(q) for q in p['range'])
        assert lo <= patches._hz(p['default_pitch']) <= hi


@pytest.mark.parametrize('name', ALL)
def test_every_patch_is_clean(name):
    y = patches.play(name, vel=1.0, dur=0.2, r=dsp.rng(4), dry=True)
    pk = np.abs(y).max()
    assert y.ndim == 2 and y.shape[1] == 2 and y.dtype == np.float64 and np.isfinite(y).all()
    assert 0.05 < pk <= 0.95                                               # a pad released after 0.2 s is quiet
    assert np.abs(y[0]).max() < 1e-3 * pk and np.abs(y[-1]).max() < 1e-3 * pk   # starts and ends at zero: -60 dB
    assert np.abs(y[:3]).max() < 0.05 * pk                                 # and rises from it without a step
    assert abs(y.mean()) < 0.05 * pk
    lo, hi = patches.PATCHES[name]['range']
    for p in (lo, hi):
        z = patches.play(name, p, vel=1.0, dur=0.05, r=dsp.rng(4), dry=True)
        assert np.isfinite(z).all() and 0.005 < np.abs(z).max() <= 0.95


@pytest.mark.parametrize('name,dur', [('soft_pluck', 0.3), ('warm_pad', 1.2), ('kick_808', 0.5), ('power_up', 0.4),
                                      ('braam', 0.3), ('fm_keys', 0.5)])
def test_patches_are_deterministic_and_velocity_matters(name, dur):
    dry = name != 'soft_pluck'                                             # one patch with its echo and room
    a = patches.play(name, vel=1.0, dur=dur, r=dsp.rng(7), dry=dry)
    b = patches.play(name, vel=1.0, dur=dur, r=7, dry=dry)
    soft = patches.play(name, vel=0.3, dur=dur, r=7, dry=dry)
    assert np.array_equal(a, b)
    assert 0.6 < np.abs(a).max() <= 0.95                                   # velocity 1 peaks near 0.9
    assert np.abs(soft).max() < 0.6 * np.abs(a).max()
    if name != 'fm_keys':                                                  # FM operators start in phase: no chance
        assert not np.array_equal(a[:len(a) // 2], patches.play(name, vel=1.0, dur=dur, r=8, dry=dry)[:len(a) // 2])


def test_a_held_bass_has_no_dc():
    for name in ('sub_bass', 'round_bass', 'pluck_bass', 'reese_bass'):
        y = patches.play(name, vel=1.0, dur=1.5, r=3, dry=True, ring=1)[14400:62400]
        assert abs(y.mean()) < 0.03 * y.std()


def test_pitch_forms_chords_and_errors():
    a, b, c = (patches.play('sub_bass', p, dur=0.2, dry=True) for p in ('A1', 33, 55.0))
    assert np.array_equal(a, b) and np.array_equal(a, c)                   # a note name, a MIDI note, Hz
    one = patches.play('warm_keys', 'C4', dur=0.2, dry=True)
    chord = patches.play('warm_keys', ['C4', 'E4', 'G4'], dur=0.2, dry=True)
    S = np.abs(np.fft.rfft(chord.mean(axis=1)))
    fr = np.fft.rfftfreq(len(chord), 1 / SR)
    for hz in (261.63, 329.63, 392.0):
        assert S[np.abs(fr - hz) < 6].max() > 0.2 * S.max()
    assert np.abs(chord).max() <= 0.95 and len(chord) == len(one)
    with pytest.raises(ValueError, match='soft_pluck'):
        patches.play('nope', 'C4')
    with pytest.raises(ValueError, match='bright'):
        patches.play('soft_pluck', 'C4', shiny=1)
    with pytest.raises(ValueError):
        patches.play('soft_pluck', 'C4', bright=2)


def test_controls_do_what_they_say():
    base = dict(vel=0.8, dur=0.3, r=1)
    play = patches.play
    lit = lambda b: _above(play('soft_pluck', bright=b, dry=True, **base)[:2400], 1500.0)
    assert lit(1) > 2 * lit(0) > 4 * lit(-1)
    dry = patches.play('soft_pluck', space=-1, **base)
    assert len(patches.play('soft_pluck', ring=1, space=-1, **base)) > 1.3 * len(dry) > 1.3 * 1.3 * len(
        patches.play('soft_pluck', ring=-1, space=-1, **base))
    assert len(patches.play('soft_pluck', **base)) > len(dry)              # space = -1 has no echo and no room
    assert np.array_equal(dry, patches.play('soft_pluck', dry=True, **base))
    assert _corr(play('warm_keys', width=-1, dry=True, **base)) > 0.999
    assert _corr(play('warm_keys', width=1, dry=True, **base)) < _corr(play('warm_keys', dry=True, **base))
    soft, hard = (play('warm_keys', hard=h, dry=True, **base) for h in (-1, 1))
    assert np.abs(hard[:240]).max() > 2 * np.abs(soft[:240]).max()          # hard = a faster attack
    assert _centroid(play('round_bass', dirt=1, dry=True, **base)) > _centroid(play('round_bass', dry=True, **base))
    take = play('fm_bell', vary=1.0, dry=True, **base)
    plain = play('fm_bell', dry=True, **base)
    assert not np.array_equal(take[:9600], plain[:9600])
    assert abs(_peak_hz(take.mean(axis=1)) - _peak_hz(plain.mean(axis=1))) < 2.0    # a take keeps the pitch


def test_plucks_close_pads_move_and_basses_stay_in_the_middle():
    for name in ('soft_pluck', 'glass_pluck', 'muted_pluck'):
        y = patches.play(name, vel=1.0, dur=0.5, dry=True)
        top = 3 * patches._hz(patches.PATCHES[name]['default_pitch'])
        assert _above(y[:1440], top) > 4 * _above(y[9600:19200], top)       # the spectrum closes as it decays
    pad = patches.play('warm_pad', ['C4', 'E4', 'G4'], dur=1.2, r=2)
    assert 0.2 < _corr(pad) < 0.75 and _mono_loss_db(pad) > -1.5
    env = np.abs(signal.hilbert(dsp.butter(pad.mean(axis=1), 'band', (240, 285))))[SR // 2:int(1.2 * SR)]
    assert env.std() / env.mean() > 0.1                                    # unison beating on the lowest note
    for name in ('sub_bass', 'round_bass', 'reese_bass', 'pluck_bass', 'kick_808'):
        lo = dsp.butter(patches.play(name, vel=1.0, dur=0.3), 'low', 120.0, order=4)
        assert _corr(lo) > 0.95


def test_designed_pitch_movements():
    sub = patches.play('sub_bass', 'G1', vel=1.0, dur=1.0, dry=True).mean(axis=1)[9600:43200]
    S = np.abs(np.fft.rfft(sub * np.hanning(len(sub))))
    fr = np.fft.rfftfreq(len(sub), 1 / SR)
    h1, h2 = S[np.abs(fr - 49.0) < 4].max(), S[np.abs(fr - 98.0) < 4].max()
    assert -30 < 20 * np.log10(h2 / h1) < -10                              # a little harmonic for small speakers
    kick = patches.play('kick_808', 'G1', vel=1.0, dur=1.0, dry=True).mean(axis=1)
    early = np.argmax(np.abs(kick[:480]))                                  # the click and the top of the pitch drop
    assert early < 240 and abs(_peak_hz(kick[9600:33600]) - 49.0) < 6       # settles on the note
    up = np.nonzero(np.diff(np.sign(kick[:4800])) > 0)[0]
    assert np.diff(up)[0] < 0.5 * np.diff(up)[-1]                           # the first cycles are shorter: it falls
    bub = patches.play('bubble', 'G5', vel=1.0, dur=0.2, dry=True).mean(axis=1)
    z = np.nonzero(np.diff(np.sign(bub[:2400])) > 0)[0]
    assert np.diff(z)[0] > 1.15 * np.diff(z)[-1]                            # it bends upward into its pitch
    rise = patches.play('riser_tone', 'C4', vel=1.0, dur=2.0, dry=True)
    a, b = _peak_hz(rise[4800:24000].mean(axis=1), 200), _peak_hz(rise[76800:96000].mean(axis=1), 200)
    assert 0.5 < 12 * np.log2(b / a) < 3.2                                 # a tonal riser: 1-3 semitones, not a sweep
    assert _centroid(rise[72000:96000]) > 3 * _centroid(rise[:24000])       # the brightness does the rising
    assert np.abs(rise[76800:96000]).max() > 3 * np.abs(rise[:19200]).max()
    pu = patches.play('power_up', 'C5', vel=1.0, dur=0.6, dry=True).mean(axis=1)
    assert _peak_hz(pu[:4000], 200) < _peak_hz(pu[13000:17000], 200) < _peak_hz(pu[28000:36000], 200)
    hit, swell = (patches.play('braam', 'D2', vel=1.0, dur=0.4, hard=h, dry=True) for h in (0, -1))
    peak_at = lambda y: np.argmax(np.abs(y).max(axis=1)) / SR
    assert peak_at(hit) < 0.5 and peak_at(swell) > 0.6                      # the hit and the swell variant
