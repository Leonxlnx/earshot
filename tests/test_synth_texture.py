"""earshot.synth.texture: the event and noise engines keep their statistics and physics, every scene renders clean,
quiet and wide, loops close without a seam."""
import functools

import numpy as np
import pytest

from earshot.forge import dsp
from earshot.synth import texture as tx

SR = dsp.SR
BEDS = [k for k, v in tx.SCENES.items() if v['loop']]


@functools.lru_cache(maxsize=None)
def _bed(scene, dur=2.0, seed=1, **kw):
    return tx.render(scene, dur, dsp.rng(seed), **kw)


def _centroid(y):
    F = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(np.sum(f * F) / np.sum(F))


def _share(y, lo, hi):
    F = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(F[(f >= lo) & (f < hi)].sum() / F.sum())


def _kurtosis(v):
    v = v - v.mean()
    return float(np.mean(v ** 4) / np.mean(v ** 2) ** 2)


def _envelope(y, lo, hi, frame=0.005):
    x = dsp.butter(y.mean(axis=1), 'band', (lo, hi), order=4)
    k = int(frame * SR)
    e = np.sqrt(np.mean(x[:len(x) // k * k].reshape(-1, k) ** 2, axis=1))
    return e[int(0.15 / frame):-int(0.35 / frame)]                    # without the fades of render()


# ------------------------------------------------------------------------------------------------ the engines
def test_arrivals_keep_the_rate_and_cluster():
    r = dsp.rng(1)
    plain, gen = tx.arrivals(r, 200.0, 20.0)
    assert abs(len(plain) - 4000) < 4 * np.sqrt(4000) and not gen.any()
    assert np.all(np.diff(plain) >= 0) and plain.min() >= 0 and plain.max() < 20.0
    bursty, gen = tx.arrivals(r, 200.0, 20.0, cluster=0.6, spread=0.01)
    assert abs(len(bursty) - 4000) < 0.15 * 4000                     # the mean rate stays
    assert gen.max() >= 2                                            # triggered events trigger more
    fano = lambda t: np.var(np.histogram(t, 400, (0, 20.0))[0]) / np.mean(np.histogram(t, 400, (0, 20.0))[0])
    assert fano(plain) < 1.5 < 2.5 < fano(bursty)                    # Poisson is 1; a Hawkes process far above
    rate = np.where(np.arange(20000) < 10000, 300.0, 20.0)           # a rate that changes over the buffer
    t, _ = tx.arrivals(r, rate, 20.0)
    assert np.sum(t < 10.0) > 8 * np.sum(t >= 10.0)


def test_sizes_follow_their_laws():
    r = dsp.rng(2)
    s = tx.sizes(r, 200000, 1.0, 100.0, 'power', 2.5)
    assert s.min() >= 1.0 and s.max() <= 100.0
    assert np.mean(s > 2.0) == pytest.approx(2.0 ** -1.5, rel=0.05)  # P(size > x) = x^-(shape - 1)
    e = tx.sizes(r, 200000, 0.5, 50.0, 'exp', 2.0)
    assert np.mean(e) - 0.5 == pytest.approx(0.5, rel=0.03)
    g = tx.sizes(r, 200000, 0.1, 10.0, 'lognormal', 0.4)
    assert np.median(g) == pytest.approx(1.0, rel=0.03) and np.std(np.log(g)) == pytest.approx(0.4, rel=0.05)
    with pytest.raises(ValueError, match='power, exp, lognormal'):
        tx.sizes(r, 3, 1, 2, 'gauss')


def test_a_ring_has_its_level_pitch_decay_and_glide():
    out = np.zeros((SR, 2))
    tx.rings(out, 0.1, 0.5, -1.0, 2000.0, 0.02, attack=0.002)
    assert np.abs(out[:, 0]).max() == pytest.approx(0.5, rel=0.03) and not out[:, 1].any()       # peak, hard left
    seg = out[4800:4800 + 8192, 0]
    f = np.fft.rfftfreq(8192, 1 / SR)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(seg * np.hanning(8192))))] - 2000.0) < 12
    env = np.abs(seg[480:2400].reshape(-1, 24)).max(axis=1)          # 10 ms .. 50 ms after the onset
    tau = -1 / np.polyfit(np.arange(len(env)) * 24 / SR, np.log(env), 1)[0]
    assert tau == pytest.approx(0.02, rel=0.1)
    assert out[:4799].max() == 0 and abs(out[4800 + int(6.3 * 0.02 * SR) + 400:]).max() == 0    # starts and ends at 0
    up = np.zeros((SR, 2))
    tx.rings(up, 0.1, 0.5, 0.0, 2000.0, 0.02, glide=5.0)             # +5 per second: 10 % higher after 20 ms
    late = up[4800 + 960:4800 + 960 + 1024, 0]
    assert f[np.argmax(np.abs(np.fft.rfft(late * np.hanning(1024), 8192)))] > 2150


def test_bubbles_ring_at_the_minnaert_frequency():
    out = np.zeros((SR // 2, 2))
    tx._bubbles(out, 0.01, 1.0, 0.0, 2.0, rise=0.0)                  # radius 2 mm -> 1630 Hz
    f = np.fft.rfftfreq(len(out), 1 / SR)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(out[:, 0])))] - 1630.0) < 25


def test_events_wrap_around_a_loop_or_stop_at_the_end():
    for wrap in (True, False):
        out = np.zeros((SR, 2))
        tx.rings(out, 0.999, 1.0, 0.0, 500.0, 0.05, wrap=wrap)
        tx.bursts(out, 0.9995, 1.0, 0.0, 800.0, 20.0, 0.002, dsp.rng(1), wrap=wrap)
        assert (np.abs(out[:2000]).max() > 0.1) == wrap


def test_a_burst_carries_the_energy_its_gain_says():
    r = dsp.rng(3)
    for f, q, length, sharp in ((5000, 1.0, 0.001, 1.0), (4000, 1.0, 0.003, 0.0), (1000, 4.0, 0.0005, 1.0),
                                (3000, 20.0, 0.001, 0.5)):
        out = np.zeros((4 * SR, 2))
        tx.bursts(out, 0.01 + 0.02 * np.arange(190), 2.0, -1.0, f, q, length, r, sharp)
        energy = np.sum(out[:, 0] ** 2) / 190                       # gain 1 = 1 ms of unit-RMS noise = 48
        assert abs(10 * np.log10(energy / (4 * 48))) < 2.0


def test_band_noise_follows_its_envelopes_and_stays_mono_safe():
    n, r = 4 * SR, dsp.rng(4)
    centres = tx._bands(100.0, 6400.0, 7)
    env = np.ones((7, 4000))
    env[5, 2000:] = 0.1                                              # the 3.2 kHz band drops 20 dB half way
    y = tx.band_noise(n, r, centres, env, side=0.7)
    assert y.shape == (n, 2)
    hi = dsp.butter(y[:, 0], 'band', (3100, 3300), order=4)
    assert 20 * np.log10(tx._rms(hi[SR:2 * SR - 4800]) / tx._rms(hi[2 * SR + 4800:3 * SR])) == pytest.approx(19, abs=3)
    lo = dsp.butter(y[:, 0], 'band', (150, 260), order=4)
    assert abs(20 * np.log10(tx._rms(lo[SR:2 * SR]) / tx._rms(lo[2 * SR:3 * SR]))) < 2.0
    corr = np.corrcoef(y.T)[0, 1]
    assert corr == pytest.approx((1 - 0.49) / (1 + 0.49), abs=0.08)
    mono, each = tx._rms(y.mean(axis=1)), np.sqrt(np.mean(y ** 2))
    assert mono ** 2 / each ** 2 == pytest.approx((1 + corr) / 2, abs=0.02)     # nothing cancels in the mono sum


def test_envelopes_have_the_marginal_and_the_correlation_asked_for():
    r = dsp.rng(5)
    together = np.log(tx.envelopes(30.0, r, 6, sigma=0.5, rho=1.0, slope=0.0))
    apart = np.log(tx.envelopes(30.0, r, 6, sigma=0.5, rho=0.0, slope=0.0))
    off = ~np.eye(6, dtype=bool)
    assert np.corrcoef(together)[off].min() > 0.99
    assert np.abs(np.corrcoef(apart)[off]).mean() < 0.25
    assert together.std(axis=1).mean() == pytest.approx(0.5, rel=0.02)
    assert np.mean(tx.envelopes(30.0, r, 40, sigma=0.5, rho=0.0, slope=0.0) ** 2) == pytest.approx(1.0, rel=0.1)
    slow, fast = tx.drift(30.0, r, 0.05, 50.0, 2.0), tx.drift(30.0, r, 0.05, 50.0, 0.0)
    assert slow.std() == pytest.approx(1.0) and np.abs(np.diff(slow)).mean() < 0.2 * np.abs(np.diff(fast)).mean()


# ------------------------------------------------------------------------------------------------ the scenes
def test_the_scene_table_is_complete():
    assert set(tx.SCENES) == set(tx._BUILD)
    for name, spec in tx.SCENES.items():
        assert 2 <= len(spec['says']) <= 3 and all(isinstance(s, str) for s in spec['says'])
        assert isinstance(spec['loop'], bool) and spec['controls']
        assert (spec['min_loop'] is not None) == spec['loop']
        assert set(spec.get('choices', {})) <= set(spec['controls'])
    assert set(tx.SCENES['rain']['choices']['surface']) == {'leaves', 'roof', 'window', 'street', 'water', 'tent'}
    assert set(tx.SCENES['wind']['choices']['place']) == {'open', 'trees', 'window', 'mountain'}
    assert tx.SCENES['rain']['min_loop'] == 8 and tx.SCENES['forest']['min_loop'] == 20
    assert not {'waves', 'underwater', 'room_tone', 'city'} & set(tx.SCENES)     # learned from recordings now


@pytest.mark.parametrize('scene', BEDS)
def test_every_bed_is_clean_quiet_and_wide(scene):
    y = _bed(scene)
    assert y.shape == (2 * SR, 2) and y.dtype == np.float64 and np.isfinite(y).all()
    peak = np.abs(y).max()
    assert 0.3 * tx.BED_PEAK < peak <= tx.BED_PEAK * (1 + 1e-9)       # a quarter of -1 dBFS
    assert not y[0].any() and not y[-1].any()                         # fades in from zero and out to zero
    assert np.abs(y.mean(axis=0)).max() < 5e-3 * peak                 # no DC
    corr = np.corrcoef(y.T)[0, 1]
    assert -0.1 < corr < 0.85
    mono, each = tx._rms(y.mean(axis=1)), np.sqrt(np.mean(y ** 2))
    assert 20 * np.log10(mono / each) > -3.6                          # the mono sum does not hollow out


def test_thunder_is_an_event_with_a_clap_and_a_roll():
    near, far = _bed('thunder', 3.0, distance=-1.0), _bed('thunder', 3.0, distance=1.0)
    for y in (near, far):
        assert y.shape == (3 * SR, 2) and np.isfinite(y).all()
        assert np.abs(y).max() == pytest.approx(tx.EVENT_PEAK, rel=1e-6)
        assert np.abs(y[:int(0.2 * SR)]).max() < 0.02 * np.abs(y).max() and not y[-1].any()
    assert _centroid(near) > 2 * _centroid(far)                       # distance takes the highs
    assert tx._rms(near[SR:2 * SR]) > 0.02 * tx._rms(near[:SR])       # the roll goes on after the clap
    with pytest.raises(ValueError, match='event'):
        tx.loop('thunder', 3.0)


def test_same_arguments_same_samples():
    for scene, kw in (('rain', dict(surface='roof')), ('forest', {})):
        a, b = tx.render(scene, 1.5, dsp.rng(7), **kw), tx.render(scene, 1.5, dsp.rng(7), **kw)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, tx.render(scene, 1.5, dsp.rng(8), **kw))
    assert np.array_equal(tx.render('stream', 1.0), tx.render('stream', 1.0, dsp.rng(0)))


def test_unknown_names_list_the_valid_ones():
    with pytest.raises(ValueError, match='rain, wind, fire'):
        tx.render('hail', 1.0)
    with pytest.raises(ValueError, match='intensity, surface, distance'):
        tx.render('rain', 1.0, loudness=1)
    with pytest.raises(ValueError, match='leaves, roof, window'):
        tx.render('rain', 1.0, surface='car')
    with pytest.raises(ValueError, match='open, trees, window, mountain'):
        tx.render('wind', 1.0, place='desert')
    with pytest.raises(ValueError, match='add m for minor'):
        tx.render('sparkle', 1.0, key='H')
    assert np.array_equal(tx.render('fire', 1.0, dsp.rng(1), size=5), tx.render('fire', 1.0, dsp.rng(1), size=1))


@pytest.mark.parametrize('scene, kw', [('rain', dict(surface='window')), ('fire', {}), ('stream', {}),
                                       ('wind', dict(place='window', howl=1.0)), ('night', {})])
def test_loops_close_without_a_seam(scene, kw):
    y = tx.loop(scene, 3.0, dsp.rng(3), **kw)
    assert y.shape == (3 * SR, 2) and np.abs(y).max() == pytest.approx(tx.BED_PEAK, rel=1e-9)
    seam = tx.seam_db(y)
    other = [tx.seam_db(np.roll(y, k, axis=0))['spectral'] for k in range(SR // 3, 3 * SR, SR // 3)]
    assert abs(seam['level']) < 1.0 and seam['click'] < 3.0
    assert seam['spectral'] < max(3.0, 1.1 * np.max(other))           # no rougher than a cut anywhere else
    cut = y.copy()
    cut[-int(0.4 * SR):] *= 0.5                                       # a 6 dB step at the seam is found
    assert abs(tx.seam_db(cut)['level']) > 4.0
    hard = y.copy()
    hard[-1] += 0.5                                                   # a click at the loop point is found
    assert tx.seam_db(hard)['click'] > 6.0


def test_rain_gets_denser_and_the_surface_changes_the_drops():
    drizzle = _bed('rain', intensity=-1.0, surface='street')
    pour = _bed('rain', intensity=1.0, surface='street')
    crest = lambda y: 20 * np.log10(np.abs(y).max() / tx._rms(y))
    assert crest(drizzle) > crest(pour) + 3                           # single drops over a faint wash / mostly wash
    tent, street = _bed('rain', surface='tent'), _bed('rain', surface='street')
    assert _share(tent, 100, 500) > 4 * _share(street, 100, 500)      # the membrane thuds, asphalt only ticks
    assert _centroid(_bed('rain', surface='window')) < _centroid(street)        # heard from inside
    hi = _envelope(street, 3000, 9000, 0.001)
    assert _kurtosis(hi) > 4.0                                        # drops stand out of the wash (noise: about 3)
    far = _bed('rain', surface='street', distance=1.0)
    assert _centroid(far) < 0.8 * _centroid(street)


def test_wind_is_one_gust_across_all_bands_and_howls_at_narrow_tones():
    calm, gale = _bed('wind', intensity=-0.8), _bed('wind', intensity=0.8)
    assert _centroid(gale) > 1.5 * _centroid(calm)                    # the spectrum moves up with the wind speed
    y = _bed('wind', 3.0, place='open')
    lo, hi = _envelope(y, 100, 300, 0.2), _envelope(y, 1000, 3000, 0.2)
    assert np.corrcoef(np.log(lo), np.log(hi))[0, 1] > 0.4            # co-modulated
    tone, none = _bed('wind', 3.0, place='window', howl=1.0), _bed('wind', 3.0, place='window', howl=0.0)

    def tonality(y):                  # how far the 200-900 Hz band still agrees with itself 8-12 ms later, in gusts
        x = dsp.butter(y.mean(axis=1), 'band', (200, 900), order=4)
        w = x[SR // 4:SR // 4 + (len(x) - SR) // 4800 * 4800].reshape(-1, 4800)           # 100 ms windows
        a = np.fft.irfft(np.abs(np.fft.rfft(w, 9600, axis=1)) ** 2, axis=1)
        return np.percentile(np.abs(a[:, 384:576]).max(axis=1) / a[:, 0], 90)
    assert tonality(tone) > 0.33 and tonality(tone) > 1.3 * tonality(none)       # a narrow band rings on; noise not
    trees = _bed('wind', 3.0, place='trees')
    assert _share(trees, 2500, 10000) > 2 * _share(y, 2500, 10000)    # leaves


def test_fire_crackles_in_bursts_over_a_roar_that_grows_with_the_fire():
    camp, large = _bed('fire', 4.0), _bed('fire', 4.0, size=1.0)
    assert _kurtosis(_envelope(camp, 2000, 9000)) > 6                 # sparse (noise: about 3)
    assert _share(camp, 2000, 12000) > 2 * _share(large, 2000, 12000)  # a large fire is mostly roar
    lo, hi = _envelope(camp, 1500, 3000), _envelope(camp, 5000, 9000)
    assert np.corrcoef(lo, hi)[0, 1] > 0.3                            # a crackle is broadband: bands rise together


def test_stick_slip_is_a_sawtooth_at_the_drive_rate_with_holds():
    rate = np.full(SR, 150.0)
    rate[:SR // 10] = 0.0                                             # not moving yet: silent
    f, ts = tx.stick_slip(dsp.rng(3), rate, jitter=0.05)
    g, _ = tx.stick_slip(dsp.rng(3), rate, jitter=0.05)
    assert np.array_equal(f, g) and np.isfinite(f).all()
    assert abs(len(ts) - 0.9 * 150) < 8 and ts.min() > 0.1
    assert np.abs(f[:SR // 10]).max() == 0.0
    d = np.diff(f)
    assert -d.min() > 5 * d.max()                                     # slow rise, sudden drop: the bright edge
    _, held = tx.stick_slip(dsp.rng(3), rate, jitter=0.05, hold=0.05, hold_s=(0.03, 0.06))
    assert np.diff(held).max() > 0.025 > np.diff(ts).max()            # the surfaces now and then hold


def test_water_bands_move_on_their_own_and_size_goes_from_plinks_to_a_roar():
    brook = _bed('stream')
    lo, hi = _envelope(brook, 1000, 1500), _envelope(brook, 4000, 6000)
    assert abs(np.corrcoef(lo, hi)[0, 1]) < 0.25
    trickle, river = _bed('stream', 3.0, size=-1.0), _bed('stream', size=1.0)
    k = lambda y: _kurtosis(_envelope(y, 1000, 6000))
    assert k(trickle) > 2 * k(brook)                                  # single plinks / a dense babble
    assert _share(river, 60, 400) > 3 * _share(brook, 60, 400)        # and a river roars underneath


def test_shared_controls_do_what_they_say():
    y = _bed('stream')
    mono = _bed('stream', width=-1.0)
    assert np.allclose(mono[:, 0], mono[:, 1])
    assert np.corrcoef(_bed('stream', width=1.0).T)[0, 1] < np.corrcoef(y.T)[0, 1]
    assert _centroid(_bed('stream', bright=1.0)) > _centroid(y) > _centroid(_bed('stream', bright=-1.0))
    assert _centroid(_bed('stream', distance=1.0)) < 0.9 * _centroid(y)
    assert _centroid(_bed('night', distance=1.0)) < _centroid(_bed('night'))
    assert _share(_bed('space_hum', size=1.0), 20, 60) > _share(_bed('space_hum', size=-1.0), 20, 60)
    hum = np.abs(np.fft.rfft(_bed('space_hum').mean(axis=1) * np.hanning(2 * SR)))
    f = np.fft.rfftfreq(2 * SR, 1 / SR)
    assert abs(f[np.argmax(hum)] - 55.0) < 3 or abs(f[np.argmax(hum)] - 110.0) < 3      # the stack on 55 Hz


def test_a_bed_has_no_period():
    k = int(0.02 * SR)
    for scene in ('rain',):
        y = _bed(scene, 10.0)
        e = np.sqrt(np.mean(y[:len(y) // k * k].reshape(-1, k, 2) ** 2, axis=(1, 2)))[10:-20]
        e = e - e.mean()
        a = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)[:len(e) // 2]
        a = a[50:] / a[0]                                             # lags past 1 s
        peaks = a[1:-1][(a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])]
        assert peaks.max() < 0.3                                      # nothing comes back
