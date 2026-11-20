"""The learned ambience beds of earshot.sounds.nature: the store holds numbers with a public-domain credit for every
model and stays under its budget; a bed renders as a finite stereo loop of its exact length at the level of beds,
two seeds are two beds, and the axes move what they stand for (seeds, bright and size with --slow)."""
import os

import numpy as np
import pytest
from scipy import ndimage, signal, stats

from earshot.forge import dsp
from earshot.sounds import nature, registry
from earshot.synth import quilt, resyn

SR = dsp.SR
BUDGET = 9_500_000


def test_the_store_is_numbers_with_credits_and_under_budget():
    tex = nature.store()
    assert set(n for n, *_ in nature.LEARNED.values()) <= set(tex)
    total = 0
    with open(os.path.join(nature.STORE, 'CREDITS.md'), encoding='utf-8') as fh:
        credits = fh.read()
    for name, t in tex.items():
        assert 'CC0' in t['licence'] and t['author'] and t['page'].startswith('https://')
        assert f'`{name}`' in credits and t['page'] in credits
        assert 10.0 <= t['seconds'] <= (40.0 if t['periodic'] or name in ('campfire', 'cornfield', 'deck', 'tinroof',
                                                                               'umbrella') else 17.5)
        assert set(t['ids']) <= set(nature.LEARNED) | set(nature.SCENES)    # or answered by an invented scene
        assert not t['periodic'] or len(t['chunks']) >= 4 or t['seconds'] < 17.5     # the sea: four waves or more
        for c in t['chunks']:
            path = os.path.join(nature.STORE, c['file'])
            total += os.path.getsize(path)
            with np.load(path) as z:
                assert list(z.keys()) == ['model'] and z['model'].dtype == np.uint8      # resyn's numbers
        assert (t['periodic'] and all(c['period'] for c in t['chunks'])) or not t['periodic']
    assert total < BUDGET
    files = [f for f in os.listdir(nature.STORE) if f.endswith('.npz')]
    assert len(files) == sum(len(t['chunks']) for t in tex.values())          # nothing stored without a credit


def test_the_learned_ids_replace_the_invented_ones():
    ids = registry.ids()
    for sid in nature.LEARNED:
        assert sid in ids and sid not in nature.SCENES
        d = registry.get(sid)
        assert d.kind == 'bed' and d.loop and set(d.axes) == set(nature.LEARNED_AXES)
    assert 'rain/window' in nature.SCENES and 'forest/morning' in nature.SCENES     # invented scenes that are kept


@pytest.fixture(scope='module')
def bed():
    return [registry.render('underwater/deep', seed=1, dur=4.0)]


def test_a_bed_is_a_finite_stereo_loop_at_the_bed_level(bed):
    for y, meta in bed:
        assert y.shape == (4 * SR, 2) and np.isfinite(y).all() and meta['loop'] is True
        assert abs(meta['lufs_max'] - registry.LOUDEST['bed']) < 1.5 and meta['peak_dbfs'] <= -0.9
        assert 'ignored' not in meta and meta['checks']['ok'], meta['checks']['warnings']
        assert 0.05 < np.corrcoef(y.T)[0, 1] < 0.98                          # stereo, not two copies


@pytest.mark.slow
def test_two_seeds_are_two_beds(bed):
    (a, _), (b, _) = bed[0], registry.render('underwater/deep', seed=2, dur=4.0)
    assert not np.allclose(a, b)
    assert abs(np.corrcoef(a[:SR, 0], b[:SR, 0])[0, 1]) < 0.2               # fresh noise, other pieces


def _top(y, f0=2000.0):
    """the share of the power above f0 Hz"""
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    p = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    return float(p[f > f0].sum() / p.sum())


def _centroid(y):
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    p = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    return float(np.sum(f * p) / np.sum(p))


@pytest.mark.slow
def test_bright_and_size_move_the_spectrum():
    m = nature.models('froth')[0]
    lo, plain, hi = (resyn.render(m, dsp.rng(1), bright=b) for b in (-1.5, 0.0, 1.5))
    assert _centroid(hi) > 1.3 * _centroid(lo)
    big = resyn.render(m, dsp.rng(1), size=1.25)
    assert _centroid(big) < 0.9 * _centroid(plain) and len(big) > 1.15 * len(plain)   # a bigger sea: lower, slower


def test_space_and_width_place_the_bed(bed):
    take = bed[0][0]                                                  # a loop: the room goes round
    d = registry.get('underwater/deep')

    def placed(**ax):
        return nature.place(take, registry.Controls(d, **ax), mono=False, loop=True)
    near, far = placed(space=-1.0), placed(space=1.0)
    assert _top(far) < 0.5 * _top(near)                                    # the air takes the top
    wide, narrow = placed(width=1.0), placed(width=-1.0)
    assert np.corrcoef(wide.T)[0, 1] < np.corrcoef(narrow.T)[0, 1] - 0.1
    for z in (near, far, wide):
        assert z.shape == take.shape and np.isfinite(z).all()
    mono = nature.place(take, registry.Controls(d, width=0.0), mono=True, loop=True)
    assert np.corrcoef(mono.T)[0, 1] < 0.95                                # a mono recording opened up
    step = np.abs(np.diff(quilt._level(quilt.features(np.concatenate([far[-SR // 2:], far[:SR // 2]])))))
    inside = np.abs(np.diff(quilt._level(quilt.features(far))))
    assert step[45:56].max() <= np.percentile(inside, 99.5)                 # no step where the loop closes


# ------------------------------------------------------------------------------------------------ what each bed is
# The numbers are measured on the public-domain recordings the beds are learned from and on two or three others of
# the same thing, against what a rebuild that lost the detail measured.
def _band(y, lo, hi):
    m = y.mean(axis=1) if y.ndim == 2 else y
    return signal.sosfiltfilt(signal.butter(4, (lo, hi), 'band', fs=SR, output='sos'), m)


def _crackles(y):
    """crackles per second and the median 10-90 % rise (ms) of 2-12 kHz peaks 15 dB over the 50 ms around them.
    Recordings of fires: 23-41 per second, rising in 1.5-2.3 ms; a fire whose crackles were smeared: 13, 2.8 ms"""
    h = _band(y, 2000.0, 12000.0)
    e = np.sqrt(ndimage.uniform_filter1d(h * h, 24))[::12]                 # 0.25 ms frames
    fs = SR / 12
    L = 20 * np.log10(e + 1e-9)
    pk, _ = signal.find_peaks(L - ndimage.median_filter(L, int(0.05 * fs) | 1, mode='nearest'), height=15.0,
                              distance=int(0.004 * fs))
    rises = []
    for p in pk:
        seg = e[max(0, p - int(0.004 * fs)):p + 1]
        lo, top = seg.min(), seg[-1]
        rises.append((np.argmax(seg >= lo + 0.9 * (top - lo)) - np.argmax(seg >= lo + 0.1 * (top - lo))) / fs * 1e3)
    return len(pk) / (len(y) / SR), float(np.median(rises))


def _env_kurtosis(y, lo, hi):
    """kurtosis of a band's envelope at 1 ms: dry leaves and foam are many small sharp events (high), a hiss whose
    level is shaped from outside is not"""
    e = np.sqrt(ndimage.uniform_filter1d(_band(y, lo, hi) ** 2, 48))[::48]
    return float(stats.kurtosis(e))


@pytest.mark.slow
def test_the_fire_crackles_as_densely_and_sharply_as_a_real_one():
    y, meta = registry.render('fire/camp', dur=10.0, seed=3)
    rate, rise = _crackles(y)
    assert rate >= 22.0 and rise <= 2.0, (rate, rise)
    assert np.corrcoef(y.T)[0, 1] < 0.95                                    # two sides, not one fire twice
    assert meta['checks']['ok']


@pytest.mark.slow
def test_the_wind_in_the_trees_is_a_bright_dry_rustle_that_swells_with_the_gusts():
    y, _ = registry.render('wind/trees', dur=12.0, seed=3)
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    p = np.abs(np.fft.rfft(y.mean(axis=1))) ** 2
    assert p[(f > 2000) & (f < 10000)].sum() / p.sum() >= 0.55    # the leaves: recording 0.64-0.68; hiss 0.45
    assert _env_kurtosis(y, 4000.0, 8000.0) >= 10.0                         # dry: recording 22; filtered hiss 6-7
    e = np.sqrt(np.mean(_band(y, 2500.0, 6000.0)[: len(y) // 480 * 480].reshape(-1, 480) ** 2, axis=1))
    gust = ndimage.uniform_filter1d(20 * np.log10(e + 1e-9), 100)
    assert gust.std() >= 2.5                                      # swells and falls: recording 2.9-4.7 dB


@pytest.mark.slow
def test_the_waves_come_whole_one_after_another_with_their_foam():
    y, _ = registry.render('waves/sand', dur=15.0, seed=3)
    e = np.sqrt(np.mean(y.mean(axis=1)[: len(y) // 4800 * 4800].reshape(-1, 4800) ** 2, axis=1))
    s = np.convolve(20 * np.log10(e + 1e-9), np.ones(10) / 10, 'valid')
    assert np.percentile(s, 95) - np.percentile(s, 5) >= 9.0      # breaks and troughs: recording 9-17 dB, heard 7
    f, t, Z = signal.stft(y.mean(axis=1), SR, nperseg=512, noverlap=384)
    P = np.abs(Z[(f >= 2000) & (f < 4000)]) ** 2 + 1e-20
    P = P[:, P.sum(axis=0) > np.percentile(P.sum(axis=0), 30)]
    bubbles = np.mean(10 * np.log10(P.max(axis=0) / np.median(P, axis=0)) > 15.0)
    assert bubbles >= 0.012, bubbles                              # foam: recording 0.023-0.035; smoothed 0.002


@pytest.mark.slow
def test_the_rain_on_a_window_holds_no_lone_tonal_drop():
    for dur, most in ((4.0, 0), (15.0, 1)):                                 # with the gutter's bubble: 3 and 4
        y, _ = registry.render('rain/window', dur=dur, seed=0)
        f, t, Z = signal.stft(y.mean(axis=1), SR, nperseg=1024, noverlap=784)
        P = np.abs(Z[(f > 400) & (f < 8000)]) ** 2 + 1e-20
        line = 10 * np.log10(P.max(axis=0) / np.median(P, axis=0))        # a ring 22-26 dB over the rain around it
        assert ndimage.label(line > 22.0)[1] <= most, dur


def test_a_pair_of_takes_opens_above_a_kilohertz_only():
    r = dsp.rng(4)
    common = r.standard_normal(SR)
    y = np.stack([common + 0.3 * r.standard_normal(SR), common + 0.3 * r.standard_normal(SR)], axis=1)
    z = nature.spread(y)
    lo = [np.corrcoef(_band(v[:, :1], 100.0, 500.0), _band(v[:, 1:], 100.0, 500.0))[0, 1] for v in (y, z)]
    hi = [np.corrcoef(_band(v[:, :1], 3000.0, 8000.0), _band(v[:, 1:], 3000.0, 8000.0))[0, 1] for v in (y, z)]
    assert abs(lo[1] - lo[0]) < 0.05 and hi[1] < hi[0] - 0.25
    assert np.allclose(z.mean(axis=1), y.mean(axis=1))                      # the middle is left as it was
