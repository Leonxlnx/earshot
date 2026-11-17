"""The learned ambience beds of earshot.sounds.nature: the store holds numbers with a public-domain credit for every
model and stays under its budget; a bed renders as a finite stereo loop of its exact length at the level of beds,
two seeds are two beds, and the axes move what they stand for (seeds, bright and size with --slow)."""
import os

import numpy as np
import pytest

from earshot.forge import dsp
from earshot.sounds import nature, registry
from earshot.synth import quilt, resyn

SR = dsp.SR
BUDGET = 4_600_000


def test_the_store_is_numbers_with_credits_and_under_budget():
    tex = nature.store()
    assert set(n for n, *_ in nature.LEARNED.values()) <= set(tex)
    total = 0
    with open(os.path.join(nature.STORE, 'CREDITS.md'), encoding='utf-8') as fh:
        credits = fh.read()
    for name, t in tex.items():
        assert 'CC0' in t['licence'] and t['author'] and t['page'].startswith('https://')
        assert f'`{name}`' in credits and t['page'] in credits
        assert 10.0 <= t['seconds'] <= (34.0 if t['periodic'] or name == 'campfire' else 17.5)
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
    assert 'rain/window' in nature.SCENES and 'forest/morning' in nature.SCENES     # rated 5 of 5: kept


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
    m = nature.models('beach')[0]
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
