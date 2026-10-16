"""forge v2 whoosh-area recipes: whoosh (fast / big / pro_big families + the v1 model), swish, swipe."""
import numpy as np
import pytest

from earshot.forge import dsp, load_recipe, recipes
from earshot.forge.search import render as search_render

SR = 48000
KINDS = [('whoosh', 'fast'), ('whoosh', 'big'), ('whoosh', 'pro_big'), ('swish', 'swish'), ('swipe', 'swipe')]
LENGTH = {'fast': (0.2, 0.9), 'big': (0.6, 2.6), 'pro_big': (0.45, 1.6), 'swish': (0.12, 0.8), 'swipe': (0.3, 1.4)}


def _mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def _db_env(m, ms=5):
    h = int(SR * ms / 1000)
    k = len(m) // h
    return 20 * np.log10(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12)


def test_recipes_listed():
    assert {'whoosh', 'swish', 'swipe'} <= set(recipes())


@pytest.mark.parametrize('module,kind', KINDS)
def test_every_family_member_renders_clean(module, kind):
    mod = load_recipe(module)
    assert set(mod.PRESETS[kind]) and len(mod.PRESETS[kind]) >= 4
    lo, hi = LENGTH[kind]
    for name in mod.PRESETS[kind]:
        p = mod.preset(kind, name)
        assert set(p) <= set(mod.SPACE), (name, set(p) - set(mod.SPACE))
        x = mod.render(p, dsp.rng(3))
        assert x.ndim == 2 and x.shape[1] == 2 and np.isfinite(x).all(), name
        assert lo <= len(x) / SR <= hi, (name, len(x) / SR)
        m = _mono(x)
        pk = np.abs(m).max()
        assert abs(m[0]) < 1e-3 * pk, name                       # no start click
        e = _db_env(m)
        assert e[-1] < e.max() - 45, name                          # the tail fades out, not chopped
        assert e[:20].max() > e.max() - 50, name                   # sound within 100 ms: frame-accurate onset


@pytest.mark.parametrize('module,kind', KINDS)
def test_recipe_object_and_local_space(module, kind):
    mod = load_recipe(module)
    first = list(mod.PRESETS[kind])[0]
    p = mod.preset(kind, first)
    rec = mod.recipe(kind, defaults=p)
    assert rec.__name__.rsplit('.', 1)[-1] == module and rec.REFSET and rec.TARGET
    sp = mod.local_space(p)
    for k, spec in sp.items():
        lo, hi = spec[0], spec[1]
        full = mod.SPACE[k]
        assert full[0] - 1e-9 <= lo <= p[k] + 1e-9 and p[k] - 1e-9 <= hi <= full[1] + 1e-9, k
    x1 = search_render(rec, p, 5)
    x2 = search_render(rec, p, 5)
    assert x1 is not None and np.array_equal(x1, x2)              # deterministic per seed


def test_families_differ():
    """members of a family are audibly different: their 1/3-octave balance or length differ clearly"""
    from earshot.forge.search import describe
    mod = load_recipe('whoosh')
    for kind in ('fast', 'big'):
        descs = [describe(mod.render(mod.preset(kind, n), dsp.rng(2))) for n in mod.PRESETS[kind]]
        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                d = np.sqrt(np.mean((descs[i]['profile'] - descs[j]['profile']) ** 2))
                assert d > 1.5, (kind, i, j, d)


def _share_below(x, f_hz):
    m = _mono(x)
    F = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return F[f < f_hz].sum() / F.sum()


def test_pro_big_is_a_low_mono_swell():
    """the pro study's designed whoosh: most energy under 150 Hz, (near) mono, peak inside the file"""
    mod = load_recipe('whoosh')
    for name in mod.PRESETS['pro_big']:
        x = mod.render(mod.preset('pro_big', name), dsp.rng(4))
        assert _share_below(x, 150) > 0.6, name
        corr = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
        assert corr > 0.9, (name, corr)
        e = _db_env(_mono(x))
        assert 0.15 < np.argmax(e) / len(e) < 0.7, name


def test_big_centroid_falls_across_the_pass():
    mod = load_recipe('whoosh')
    x = _mono(mod.render(mod.preset('big', 'sweep'), dsp.rng(1)))
    e = _db_env(x, 20)
    i = int(np.argmax(e))
    h = int(SR * 0.02)

    def cen(k):
        seg = x[k * h:(k + 1) * h]
        F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return (F * f).sum() / F.sum()
    after = [k for k in range(i + 1, len(e)) if e[k] > e[i] - 20]
    assert after and cen(after[-1]) < cen(i)


def test_v1_parameter_sets_still_render_the_v1_model():
    mod = load_recipe('whoosh')
    x = mod.render(dict(mod.DEFAULTS_V1), dsp.rng(3))
    y = mod.render_v1(dict(mod.DEFAULTS_V1), dsp.rng(3))
    assert np.array_equal(x, y) and x.shape[1] == 2
