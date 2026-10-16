"""The sweep recipes (riser, downlifter, zip): every character family renders, hits its category's measured targets
(docs/motion-sfx-study.md) and sounds different from its siblings; renders are deterministic and use no recorded
audio."""
import inspect

import numpy as np
import pytest

from earshot.forge import load_recipe
from earshot.forge.search import describe, render, sound_dist

SR = 48000
NAMES = ('riser', 'downlifter', 'zip')


def families(rec):
    out = dict(rec.FAMILIES)
    for cat, fams in getattr(rec, 'FAMILIES_BY_CAT', {}).items():
        out.update({f'{cat}:{k}': v for k, v in fams.items()})
    return out


@pytest.fixture(scope='module')
def renders():
    out = {}
    for name in NAMES:
        rec = load_recipe(name)
        out[name] = {k: render(rec, p, 11) for k, p in families(rec).items()}
    return out


@pytest.mark.parametrize('name', NAMES)
def test_recipe_contract_and_no_recorded_audio(name):
    rec = load_recipe(name)
    for k in ('TARGET', 'ALTS', 'REFSET', 'SPACE', 'DEFAULTS', 'FAMILIES', 'render', 'measure', 'target_check'):
        assert hasattr(rec, k), (name, k)
    src = inspect.getsource(rec)
    for bad in ('soundfile', 'sf.read', 'audio.load', 'wavfile', 'librosa.load'):
        assert bad not in src, (name, bad)
    for fam, p in families(rec).items():                     # presets stay inside the search space
        for key, spec in rec.SPACE.items():
            v = p.get(key, rec.DEFAULTS.get(key))
            if isinstance(spec, list):
                assert v in spec, (name, fam, key, v)
            else:
                assert spec[0] - 1e-9 <= v <= spec[1] + 1e-9, (name, fam, key, v)


@pytest.mark.parametrize('name', NAMES)
def test_families_render_clean_and_deterministic(name, renders):
    rec = load_recipe(name)
    for fam, x in renders[name].items():
        assert x is not None and np.isfinite(x).all(), (name, fam)
        assert x.ndim == 2 and x.shape[1] == 2, (name, fam)
        assert abs(float(x[0].max())) < 0.05, (name, fam)          # no start click
        assert np.abs(x).max() <= 0.9 and np.abs(x.mean(axis=0)).max() < 0.01, (name, fam)
    fam0, p0 = next(iter(families(rec).items()))
    a, b = render(rec, p0, 11), render(rec, p0, 11)
    assert np.array_equal(a, b)
    c = render(rec, p0, 12)
    assert not np.array_equal(a[:len(c)], c[:len(a)])               # new seed, new noise


def test_risers_build_and_stop_on_their_peak(renders):
    rec = load_recipe('riser')
    for fam, x in renders['riser'].items():
        m = rec.measure(x)
        assert m['end_db'] > -6, (fam, m)                         # the loudest moment is the end (the cut)
        assert m['last_third'] >= 0.4, (fam, m)                   # concave: most of the rise comes late
        assert m['hf_rise_db'] > 25, (fam, m)                     # the top opens up
        assert 0.75 <= m['length_s'] <= 4.05, (fam, m)
        assert rec.target_check(x) > -1.0, (fam, m)


def test_downlifters_push_first_fall_and_fade(renders):
    rec = load_recipe('downlifter')
    for fam, x in renders['downlifter'].items():
        m = rec.measure(x)
        assert m['peak_frac'] < 0.15, (fam, m)
        assert m['cent_oct'] < -0.3, (fam, m)                     # the spectrum falls
        assert m['tail_db'] < -50, (fam, m)                       # faded to silence, never cut
        assert rec.target_check(x) > -1.5, (fam, m)


def test_zips_are_fast_exponential_sweeps(renders):
    rec = load_recipe('zip')
    for fam, x in renders['zip'].items():
        m = rec.measure(x)
        assert m['length_s'] < 0.5, (fam, m)
        assert m['oct_span'] >= 1.3, (fam, m)
        if families(rec)[fam]['shape'] != 'arc':
            assert 15 <= abs(m['oct_per_s']) <= 75, (fam, m)      # pro zips: 20-60 oct/s
        assert m['side_db'] < -12, (fam, m)                       # near mono
    f = rec.trajectory({**rec.DEFAULTS, 'shape': 'up', 'f0': 100.0, 'f1': 1600.0, 'bend': 0.0}, 4801)
    d = np.diff(np.log2(f))
    assert np.allclose(d, d[0], rtol=1e-6)                        # equal octaves per second


@pytest.mark.parametrize('name', NAMES)
def test_families_sound_different(name, renders):
    """the v1 finals were clones (CLAP cos 0.89-0.97 between variants): each family must differ from every other in
    tonal balance + envelope (search.sound_dist, the measure diverse() uses; 2.0 is its default threshold)"""
    xs = renders[name]
    rec = load_recipe(name)
    groups = [[f'{c}:{k}' for k in f] for c, f in getattr(rec, 'FAMILIES_BY_CAT', {}).items()] or [list(rec.FAMILIES)]
    for keys in groups:
        _differ(name, xs, keys)


def _differ(name, xs, keys):
    D = {k: describe(xs[k]) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert sound_dist(D[a], D[b]) >= 2.0, (name, a, b, sound_dist(D[a], D[b]))


def test_round3_no_steady_lines_and_air(renders):
    """round 3 (the judge: riser_05's steady 2.5 kHz metal line, riser_02's sub drone line; pro_riser heard as a
    chiptune arpeggio; pro_zip a toy 'boing'): drones glide up with the build or become a noise rumble, the cymbal's
    metal glides, the pro stutter builds carry their steps in filtered noise, the pro zips ride a noise band"""
    rec = load_recipe('riser')
    for k, p in rec.FAMILIES_BY_CAT['riser'].items():
        assert p['drone_oct'] >= 0.5 or p['drone_db'] <= -40, k
    assert rec.FAMILIES_BY_CAT['riser']['cymbal']['metal_glide'] >= 0.3
    pro = rec.FAMILIES_BY_CAT['pro_riser']
    assert sum(p['tmode'] == 'nblips' for p in pro.values()) >= 2
    assert all(p['tmode'] != 'blips' and p.get('wave') != 'saw' or p['tone_db'] <= -12 for p in pro.values())
    for k in pro:
        x = renders['riser'][f'pro_riser:{k}']
        assert rec.measure(x)['end_db'] > -6, k                    # still stops on its peak (the cut)
    # the drone pitch really moves: its strongest low partial in the last fifth sits >= 0.4 oct above the first fifth
    p = {**rec.FAMILIES_BY_CAT['riser']['tension'], 'noise_db': -36.0, 'tone_db': -36.0, 'oct_db': -36.0,
         'drone_db': -3.0, 'drone_noise': 0.0, 'verb': 0.0}
    x = render(rec, p, 3).mean(axis=1)
    n = len(x) // 5

    def lowpk(seg):
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        sel = (f > 30) & (f < 200)
        return f[sel][np.argmax(S[sel])]
    assert np.log2(lowpk(x[-n:]) / lowpk(x[:n])) >= 0.4
    z = load_recipe('zip')
    assert all(v['noise_db'] > -20 for v in z.FAMILIES_BY_CAT['pro_zip'].values())


def test_round6_build_risers_no_rumble_concave_gated_end_on_peak():
    """round 6 (the judge: CLAP's nearest confusion for the risers is 'a jet engine starting'): the 'build' render
    (riser.R6, p['r6'] = 1) has no drone / rumble (nothing under 150 Hz), a concave rise that stops on its peak, a
    noise band that opens up, and the old render is untouched when r6 is off"""
    rec = load_recipe('riser')
    for k, p in rec.R6.items():
        assert p['r6'] == 1.0 and p['drone_db'] <= -47 and p['metal_db'] <= -47, k
        x = render(rec, p, 5)
        assert x is not None and np.isfinite(x).all() and x.shape[1] == 2, k
        m = rec.measure(x)
        assert m['end_db'] > -3 and m['last_third'] >= 0.45 and m['hf_rise_db'] > 30, (k, m)
        assert rec.low_share_db(x, 150.0) < -30, (k, rec.low_share_db(x, 150.0))
        assert m['side_db'] < -3, (k, m)
    g = {**rec.R6['rush'], 'tone_db': -36.0, 'oct_db': -36.0, 'fifth_db': -48.0, 'verb': 0.0}   # the gate rate (the strongest
    y = render(rec, g, 2).mean(axis=1)                       # 3-60 Hz line of the 2 ms envelope) is higher late
    h = 96
    e = np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1))

    def rate(seg):
        s = (seg - np.convolve(seg, np.ones(15) / 15, 'same')) * np.hanning(len(seg))
        S = np.abs(np.fft.rfft(s, 8192))
        f = np.fft.rfftfreq(8192, h / SR)
        sel = (f > 3) & (f < 60)
        return f[sel][np.argmax(S[sel])]
    n = len(e)
    assert rate(e[int(0.7 * n):int(0.97 * n)]) > 1.5 * rate(e[int(0.15 * n):int(0.45 * n)])
    p0 = rec.FAMILIES_BY_CAT['riser']['air']
    assert np.array_equal(render(rec, p0, 4), render(rec, {**p0, 'r6': 0.0}, 4))
