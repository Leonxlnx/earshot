"""earshot.sounds.learned: the store (every model loads, is credited to a public-domain recording, the store stays
small), every design renders a clean levelled take that passes its audit, the declared axes move what they stand
for, seeds are different performances that measure alike. Model-free; the fast tier renders one short model per
family, --slow all of them."""
import json
import os

import numpy as np
import pytest

from earshot.forge import dsp
from earshot.sounds import learned, registry
from earshot.synth import resyn

SR = dsp.SR
INDEX = learned.index()
IDS = list(INDEX)
LIMIT = 6_000_000                     # bytes the whole store may weigh for now


def _raw(sid, seed=5, **req):
    d = registry.get(sid)
    return np.asarray(d.fn(registry.Controls(d, **req), dsp.rng(seed)), np.float64)


def _centroid(y):
    m = y.mean(axis=1)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return float((P * f).sum() / (P.sum() + 1e-30))


def _level(y):
    return float(10 * np.log10(np.mean(y ** 2) + 1e-20))


def _end(y, down=30.0):
    """seconds to where the take is `down` dB under its loudest 5 ms for the last time"""
    k = dsp.n_of(0.005)
    e = 10 * np.log10(np.convolve(np.mean(y ** 2, axis=1), np.ones(k) / k, 'same') + 1e-20)
    return float(np.nonzero(e > e.max() - down)[0][-1] / SR)


def _strike(y, t0):
    """dB of the strike (1 ms before to 4 ms after t0) over what follows it (12 to 60 ms after)"""
    p = np.mean(y ** 2, axis=1)
    a, b = p[max(dsp.n_of(t0 - 0.001), 0):dsp.n_of(t0 + 0.004)], p[dsp.n_of(t0 + 0.012):dsp.n_of(t0 + 0.06)]
    return float(10 * np.log10((a.mean() + 1e-20) / (b.mean() + 1e-20)))


# ---- the store

def test_index_files_and_credits():
    files = {f for f in os.listdir(learned.STORE) if f.endswith('.npz')}
    assert files == {e['file'] for e in INDEX.values()}                     # nothing unlisted, nothing missing
    with open(os.path.join(learned.STORE, 'CREDITS.md'), encoding='utf-8') as fh:
        credits = fh.read()
    for sid, e in INDEX.items():
        c = e['credit']
        assert c['title'] and c['author'] and c['page'].startswith('https://'), sid
        assert 'CC0' in c['licence'] or 'public domain' in c['licence'].lower(), (sid, c['licence'])
        assert f'`{sid}`' in credits and c['page'] in credits, sid
        assert 'mixkit' not in (c['page'] + e['source']).lower(), sid
        assert 0 <= e['onset_s'] < e['dur_s'] and e['kind'] in registry.KINDS, sid


def test_store_size():
    total = sum(os.path.getsize(os.path.join(learned.STORE, f)) for f in os.listdir(learned.STORE))
    assert total < LIMIT, total


def test_every_model_loads_and_is_a_design():
    with open(os.path.join(learned.STORE, 'index.json'), encoding='utf-8') as fh:
        rows = json.load(fh)
    assert len(rows) == len(INDEX) >= 250                                    # ids are unique
    for sid, e in INDEX.items():
        m = learned.model(sid)
        assert {'n', 'on', 'ap', 'ton', 'frq', 'q', 'u', 'h'} <= set(m) and int(m['n']) > 0, sid
        assert ('ild' in m) == e['stereo'], sid
        assert 'rating' in e and isinstance(e['default'], bool), sid
        assert e['rating'] is None or e['rating'] >= 3, sid                  # rated 1-2 by the owner: never in
        d = registry.get(sid)
        prior = registry.LEARNED * (1.0 if e['default'] else learned.DEFAULT_OFF)
        assert d.kind == e['kind'] and set(d.axes) == set(e['axes']) and d.prior == prior, sid
        assert {'energy', 'size', 'bright'} <= set(d.axes), sid
        assert (d.pitch is not None) == bool(e.get('hz')), sid
    assert learned.model(IDS[0]) is learned.model(IDS[0])                    # read once


def _fast(sid):
    """the fast tier renders the first short model of each family; the rest need --slow"""
    fam = sid.split('/')[0]
    first = next((i for i in IDS if i.split('/')[0] == fam and INDEX[i]['dur_s'] <= 0.4), None)
    return sid == first


@pytest.mark.parametrize('sid', [pytest.param(i, id=i, marks=[] if _fast(i) else [pytest.mark.slow]) for i in IDS])
def test_renders_clean_and_passes_its_audit(sid):
    ok = False
    for seed in (0, 101, 202, 303):            # as make re-rolls a take that fails its audit
        y, meta = registry.render(sid, seed=seed)
        assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
        assert np.abs(y).max() <= 1.0 and meta['peak_dbfs'] <= -0.8 and not meta.get('ignored')
        assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-2
        ok = meta['checks']['ok']
        if ok:
            break
    assert ok, meta['checks']['warnings']


# ---- controls

@pytest.mark.parametrize('sid', ['typing_key/office', 'thud/drop'])
def test_size_is_lower_and_longer(sid):
    small, big = _raw(sid, size=-1.0, space=-1.0), _raw(sid, size=1.0, space=-1.0)
    assert _centroid(big) < 0.75 * _centroid(small) and _end(big) > 1.5 * _end(small)


@pytest.mark.parametrize('sid', ['typing_key/office', 'whoosh/heavy'])
def test_bright_raises_the_centroid(sid):
    assert _centroid(_raw(sid, bright=1.0)) > 1.3 * _centroid(_raw(sid, bright=-1.0))


@pytest.mark.parametrize('sid', ['knock/hollow', 'click/mouse_light'])
def test_hard_makes_the_strike_stand_out(sid):
    t0 = INDEX[sid]['onset_s']
    assert _strike(_raw(sid, hard=1.0, space=-1.0), t0) > _strike(_raw(sid, hard=-1.0, space=-1.0), t0) + 2.0


def test_ring_rings_longer():
    assert _end(_raw('cork/wine', ring=1.0, space=-1.0)) > 1.3 * _end(_raw('cork/wine', ring=-1.0, space=-1.0))


def test_energy_is_louder_and_harder():
    lo, hi = _raw('typing_key/office', energy=-1.0), _raw('typing_key/office', energy=1.0)
    assert _level(hi) > _level(lo) + 9.0


def test_width_spreads_a_mono_and_a_stereo_recording():
    for sid in ('typing_key/office', 'thud/drop'):
        def corr(y):
            return np.corrcoef(y[:, 0], y[:, 1])[0, 1]
        assert corr(_raw(sid, width=1.0, space=-1.0)) < corr(_raw(sid, width=-1.0, space=-1.0)) - 0.1, sid


def test_space_adds_a_tail():
    assert _end(_raw('knock/hollow', space=1.0)) > _end(_raw('knock/hollow', space=-1.0)) + 0.05


def test_pitch_moves_a_tonal_model():
    sid = 'knock/hollow'
    f0 = INDEX[sid]['hz']
    y = _raw(sid, pitch=f0 * 2 ** (3 / 12), space=-1.0)
    f = max(resyn.partials(y), key=lambda q: q[1])[0]
    assert abs(1200 * np.log2(f / (f0 * 2 ** (3 / 12)))) < 50
    assert registry.get('typing_key/office').pitch is None                       # a click has no note to move


def test_length_warps_within_its_natural_range():
    sid = 'whoosh/heavy'
    d = INDEX[sid]['dur_s']
    plain, long_ = _end(_raw(sid, space=-1.0)), _end(_raw(sid, dur=1.5 * d, space=-1.0))
    assert 1.25 < long_ / plain < 1.75
    far = _raw(sid, dur=6.0 * d, space=-1.0)                                  # clamped: no stretch past LENGTH
    assert len(far) <= (learned.LENGTH[1] + 0.1) * d * SR


# ---- seeds

@pytest.mark.parametrize('sid', ['typing_key/office', 'knock/hollow', 'swish/bamboo'])
def test_seeds_are_performances(sid):
    a, b = _raw(sid, seed=1), _raw(sid, seed=1)
    assert np.array_equal(a, b)
    takes = [a] + [_raw(sid, seed=s) for s in (2, 3)]
    for t in takes[1:]:
        assert len(t) != len(a) or not np.allclose(t, a)
    c = [_centroid(t) for t in takes]
    lv = [_level(t) for t in takes]
    assert max(c) / min(c) < 1.35 and max(lv) - min(lv) < 2.5, (c, lv)
