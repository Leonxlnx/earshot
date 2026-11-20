"""earshot.synth.drums (what plays the drums no recording covers: the brush, the snap, the sticks, the shaker and the
drum-machine voices): shape, determinism, and the physics it promises (membrane mode ratios, the pitch drop growing
with velocity, wires that follow the head, damping shortening the ring). The designs of the catalogue are tested in
test_sounds_drums.py."""
import numpy as np
import pytest

from earshot.forge import dsp
from earshot.synth import drums as D

SR = dsp.SR


def _env_db(x, win=0.005):
    x = x.mean(1) if x.ndim == 2 else x
    w = int(win * SR)
    k = len(x) // w
    e = np.sqrt(np.mean(x[:k * w].reshape(k, w) ** 2, 1)) + 1e-12
    return 20 * np.log10(e / e.max())


def _t_down(x, db=-40.0):
    e = _env_db(x)
    i = int(np.argmax(e))
    below = np.nonzero(e[i:] < db)[0]
    return (i + below[0]) * 0.005 if len(below) else len(e) * 0.005


def _centroid(x, t0, t1):
    x = x.mean(1) if x.ndim == 2 else x
    s = x[int(t0 * SR):int(t1 * SR)]
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    f = np.fft.rfftfreq(len(s), 1 / SR)
    return float((S * f).sum() / S.sum())


def _freq(x, t0, t1):
    """mean frequency from zero crossings of a single partial between t0 and t1 s"""
    s = x[int(t0 * SR):int(t1 * SR)]
    z = np.nonzero((s[:-1] <= 0) & (s[1:] > 0))[0]
    frac = z - s[z] / (s[z + 1] - s[z])
    return (len(frac) - 1) / ((frac[-1] - frac[0]) / SR)


# ------------------------------------------------------------------------------------------------------ membrane
def test_ideal_membrane_ratios_are_bessel_zeros():
    s = D.membrane_modes(count=6, load=0.0)
    assert np.allclose(s['ratio'], [1.0, 1.594, 2.136, 2.296, 2.653, 2.918], rtol=2e-3)
    assert list(zip(s['order'][:4], s['circles'][:4])) == [(0, 1), (1, 1), (2, 1), (0, 2)]


def test_air_loading_gives_timpani_and_kick_ratios():
    s = D.membrane_modes(count=12, load=1.0)
    n1 = [s['ratio'][np.nonzero((s['order'] == m) & (s['circles'] == 1))[0][0]] for m in (1, 2, 3, 4)]
    q = np.array(n1) / n1[0]
    assert np.allclose(q, [1.0, 1.50, 1.98, 2.44], rtol=0.04)            # measured timpani (Fletcher & Rossing)
    k = D.membrane_modes(count=6, load=1.0, seal=1.0)
    assert np.allclose(np.sort(k['ratio'])[:3], [1.0, 1.5, 2.28], rtol=0.03)


def test_centre_strike_drives_only_axisymmetric_modes():
    s = D.membrane_modes(count=12, strike=0.0)
    assert np.all(np.abs(s['gain'][s['order'] > 0]) < 1e-12)
    assert np.all(np.abs(s['gain'][s['order'] == 0]) > 0.1)


def test_render_has_its_modes_where_the_law_puts_them():
    m = D.membrane(200.0, dsp.n_of(1.0), dsp.rng(1), dsp.rng(2), vel=0.5, strike=0.45, t60=1.0, load=0.5,
                   count=6, glide=False, click=0.0, early=0.0)
    y = m['y'].mean(1)[int(0.02 * SR):]
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 18))
    f = np.fft.rfftfreq(1 << 18, 1 / SR)
    for fm, a in zip(m['freq'], m['amp']):
        if abs(a) > 0.1 * np.abs(m['amp']).max():
            band = (f > fm * 0.97) & (f < fm * 1.03)
            assert abs(f[band][np.argmax(S[band])] / fm - 1) < 0.004


def test_pitch_drop_grows_with_velocity():
    drops = []
    for vel in (0.3, 0.6, 1.0):
        m = D.membrane(100.0, dsp.n_of(0.6), dsp.rng(1), dsp.rng(2), vel=vel, beater='felt', strike=0.0, t60=0.6,
                       count=1, fall=0.1, click=0.0, early=0.0)
        y = m['y'].mean(1)
        drops.append(12 * np.log2(_freq(y, 0.004, 0.03) / _freq(y, 0.3, 0.5)))
    assert 0.0 < drops[0] < drops[1] < drops[2]
    assert 0.8 < drops[2] < 3.0                       # about a semitone and a half at full velocity, fall 0.1


def test_damping_shortens_the_upper_modes_and_the_ring():
    r0 = D.membrane_rates([1.0, 1.594, 2.136], [0, 1, 2], [1, 1, 1], 0.5, damp=0.0)
    r1 = D.membrane_rates([1.0, 1.594, 2.136], [0, 1, 2], [1, 1, 1], 0.5, damp=2.0)
    assert r1[0] / r0[0] == pytest.approx(1.4) and r1[2] / r0[2] > r1[1] / r0[1] > 1.5
    late = []
    for damp in (0.0, 2.0):
        m = D.membrane(180.0, dsp.n_of(1.0), dsp.rng(1), dsp.rng(2), vel=0.7, strike=0.4, t60=0.6, damp=damp,
                       click=0.0, early=0.0)
        y = m['y'].mean(1)
        late.append(10 * np.log10(np.sum(y[int(0.1 * SR):int(0.3 * SR)] ** 2)))
    assert late[1] < late[0] - 3.0                   # the same stroke: the damped head is far quieter by 0.1-0.3 s


def test_harder_and_faster_contacts_are_shorter_and_brighter():
    assert D.contact('stick', 1.0)[0] < D.contact('stick', 0.3)[0]
    assert D.contact('felt', 0.7, hard=1.0)[0] < D.contact('felt', 0.7, hard=-1.0)[0]
    c = []
    for hard in (-1.0, 1.0):
        m = D.membrane(180.0, dsp.n_of(0.4), dsp.rng(1), dsp.rng(2), vel=0.7, beater='felt', hard=hard,
                       strike=0.5, t60=0.4, click=0.0, early=0.0)
        c.append(_centroid(m['y'], 0.0, 0.05))
    assert c[1] > 1.1 * c[0]


# ------------------------------------------------------------------------------------------------------ wires
def test_wires_follow_the_head_and_ring_a_little_longer():
    m = D.membrane(230.0, dsp.n_of(0.7), dsp.rng(1), dsp.rng(2), vel=0.8, strike=0.35, t60=0.4, click=0.0,
                   early=0.0)
    head = D.bottom_head(m['motion'], 330.0)
    w = D.rattle(head, dsp.rng(3), gap=0.015, release=0.03)
    eh, ew = _env_db(head), _env_db(w)
    live = (ew > -40) & (eh > -40)
    assert np.corrcoef(eh[live], ew[live])[0, 1] > 0.8
    for db in (-10.0, -20.0):                        # the buzz lingers a little longer than the head's swing
        assert _t_down(head, db) < _t_down(w, db) < 2.0 * _t_down(head, db)
    assert not D.rattle(head * 0.01, dsp.rng(3), gap=0.015).any()       # a head that never reaches them: silence


# ------------------------------------------------------------------------------------------------------ voices
@pytest.mark.parametrize('fn', [
    lambda r: D.shaker(dsp.n_of(0.3), r, [(0.05, 1.0)]),
    lambda r: D.clap(dsp.n_of(0.1), r),
    lambda r: D.snap(dsp.n_of(0.1), r),
    lambda r: D.kick_synth(dsp.n_of(0.5), r),
    lambda r: D.snare_synth(dsp.n_of(0.3), r),
    lambda r: D.clap_synth(dsp.n_of(0.3), r),
    lambda r: D.hat_synth(dsp.n_of(0.1), r)])
def test_voices_are_finite_deterministic_and_vary_by_seed(fn):
    a, b, c = fn(dsp.rng(1)), fn(dsp.rng(1)), fn(dsp.rng(2))
    assert np.isfinite(a).all() and np.abs(a).max() > 0
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_shaker_density_follows_the_energy():
    """collisions come thick right after the wall hit and thin out as the beads settle"""
    y = D.shaker(dsp.n_of(0.3), dsp.rng(1), [(0.02, 1.0)], pre=0.0)
    e = _env_db(y)
    assert e[int(0.03 / 0.005)] > e[int(0.2 / 0.005)] + 20


def test_kick_synth_pitch_falls_onto_its_note():
    y = D.kick_synth(dsp.n_of(0.8), dsp.rng(1), f_end=50.0, click=0.0, drive=0.0)
    assert _freq(y, 0.25, 0.6) == pytest.approx(50.0, rel=0.03)
    assert _freq(y, 0.0, 0.03) > 1.5 * 50.0
