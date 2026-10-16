"""forge v2 whoosh-area recipes: whoosh (fast / big / pro_big families + the v1 model), swish, swipe."""
import numpy as np
import pytest

from earshot.forge import dsp, load_recipe, recipes
from earshot.forge.search import render as search_render

SR = 48000
KINDS = [('whoosh', 'fast'), ('whoosh', 'big'), ('whoosh', 'pro_big'), ('swish', 'swish'), ('swipe', 'swipe')]
LENGTH = {'fast': (0.2, 1.0), 'big': (0.6, 2.6), 'pro_big': (0.45, 1.6), 'swish': (0.12, 0.8), 'swipe': (0.3, 1.4)}


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
    x = _mono(mod.render(mod.preset('big', 'swell'), dsp.rng(1)))
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


# ---------------------------------------------------------------------------------------------- round 3
def _shares(x):
    m = _mono(x)
    F = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    tot = F.sum()
    return {k: 10 * np.log10(F[(f >= a) & (f < b)].sum() / tot + 1e-12)
            for k, (a, b) in {'lo': (0, 150), 'hi': (4000, 12000)}.items()}


def _t40(x):
    e = _db_env(_mono(x))
    i = int(np.argmax(e))
    below = np.nonzero(e[i:] < e[i] - 40)[0]
    return (below[0] if len(below) else len(e) - i) * 5.0


def test_round3_options_off_render_as_before():
    """the round-3 keys (fabric hiss, shelves, low lag) default to off: a parameter set without them renders exactly as
    the same set with them at their off values (so every round-1/2 report and seed-bank entry still reproduces)"""
    mod = load_recipe('whoosh')
    p = dict(mod.DEFAULTS_R2)
    off = {**p, 'hiss_rustle': 0.0, 'eq_lo_db': 0.0, 'eq_hi_db': 0.0, 'low_lag': 0.0}
    assert np.array_equal(mod.render(p, dsp.rng(5)), mod.render(off, dsp.rng(5)))
    assert set(mod.OPTIONAL) <= set(mod.SPACE)


def test_shelf_moves_only_its_side():
    mod = load_recipe('whoosh')
    x = dsp.rng(1).standard_normal(SR)
    for high, f_in, f_out in ((True, 9000, 200), (False, 60, 4000)):
        y = mod.shelf(x, 1500 if high else 300, -12.0, high=high)
        F, G = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(SR, 1 / SR)
        band = lambda c: (f > c / 1.2) & (f < c * 1.2)
        assert 10 * np.log10(G[band(f_in)].sum() / F[band(f_in)].sum()) < -10
        assert abs(10 * np.log10(G[band(f_out)].sum() / F[band(f_out)].sum())) < 1.5


def test_round3_fast_and_swish_tails_and_balance():
    """round-3 judge fixes: t40 >= ~180 ms (round 2: ~130), < 150 Hz share under -11 dB, 4-12 kHz under -22 dB
    (family medians over the presets)"""
    for module, kind in (('whoosh', 'fast'), ('swish', 'swish')):
        mod = load_recipe(module)
        xs = [mod.render(mod.preset(kind, n), dsp.rng(3)) for n in mod.PRESETS[kind]]
        sh = [_shares(x) for x in xs]
        assert np.median([_t40(x) for x in xs]) >= 180, module
        assert np.median([s['lo'] for s in sh]) < -11, module
        assert np.median([s['hi'] for s in sh]) < -22, module


def test_round3_swipe_is_bright_without_a_low_onset():
    """round-3 judge fix: the swipe family carries a fabric top (4-12 kHz share above -26 dB, round 2 -36) and its
    first 15 ms after the -20 dB crossing are not a low thump (> 6 kHz share above -35 dB there)"""
    mod = load_recipe('swipe')
    his, ons = [], []
    for n in mod.PRESETS['swipe']:
        x = mod.render(mod.preset('swipe', n), dsp.rng(3))
        his.append(_shares(x)['hi'])
        m = _mono(x)
        e = _db_env(m)
        i = int(np.argmax(e))
        a = int(np.nonzero(e[:i + 1] < e[i] - 20)[0][-1]) + 1
        seg = m[a * 240:a * 240 + 720]
        F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        ons.append(10 * np.log10(F[f >= 6000].sum() / F.sum()))
    assert np.median(his) > -26 and np.median(ons) > -35, (his, ons)


@pytest.mark.parametrize('module,kind', KINDS)
def test_presets_inside_space(module, kind):
    """every family member's parameters lie inside SPACE (local_space clips a value outside, so a forge run would
    start from a different sound than the preset says)"""
    mod = load_recipe(module)
    for name in mod.PRESETS[kind]:
        p = mod.preset(kind, name)
        for k, v in p.items():
            spec = mod.SPACE[k]
            if isinstance(spec, list) or v is None:
                continue
            assert spec[0] - 1e-9 <= v <= spec[1] + 1e-9, (name, k, v, spec[:2])
