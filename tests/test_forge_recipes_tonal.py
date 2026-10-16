"""forge v2 tonal recipes (pop v2, notify, shimmer): families render, hit the pro-study targets they were built for,
differ from each other, stay deterministic, and pop's v1 parameter sets still take the v1 path. No models."""
import numpy as np
import pytest

from earshot.forge import dsp
from earshot.forge.search import _decode, _encode, render

SR = 48000


def _mods():
    from earshot.forge.recipes import notify, pop, shimmer
    return notify, pop, shimmer


def _fam(cat):
    notify, pop, shimmer = _mods()
    if cat.endswith('pop'):
        return pop, pop.FAMILY_V2[cat]
    if cat.endswith('notify'):
        return notify, notify.FAMILY[cat]
    return shimmer, shimmer.FAMILY[cat]


@pytest.mark.parametrize('cat', ['pro_notify', 'notify', 'pro_pop', 'pop', 'pro_shimmer', 'shimmer'])
def test_families_render_clean_and_deterministic(cat):
    mod, fam = _fam(cat)
    assert len(fam) >= 4
    for i, p in enumerate(fam[:3] if 'shimmer' in cat else fam):
        x = render(mod, p, i + 1)
        assert x is not None and np.isfinite(x).all()
        y = render(mod, p, i + 1)
        assert np.array_equal(x, y)                               # same seed, same samples
        m = x if x.ndim == 1 else x.mean(1)
        assert abs(m[0]) < 0.05 and abs(m[-1]) < 0.01             # no start / end click


def test_pro_notify_hits_the_study_targets():
    notify, _, _ = _mods()
    for i, p in enumerate(notify.FAMILY['pro_notify']):
        x = render(notify, p, i + 3)
        m = notify.measure(x)
        assert 0.18 <= m['length_s'] <= 0.45
        assert m['attack_ms'] <= 15
        lo = 10 * np.log10(sum(10 ** (s / 10) for s in m['shares'][:2]))
        assert lo < -38                                          # nothing under 500 Hz
        assert 1700 <= m['head_hz'] <= 2400                      # the head's fundamental
    f0s = [p['f0'] for p in notify.FAMILY['pro_notify'] if p['style'] == 'ding']
    steps = np.diff(np.log2(sorted(set(f0s)))) * 12
    assert np.allclose(steps, np.round(steps), atol=0.01)        # the ladder steps in whole semitones


def test_pro_pop_chirp_falls_about_five_octaves_and_is_mono():
    _, pop, _ = _mods()
    p = pop.FAMILY_V2['pro_pop'][0]
    x = render(pop, p, 2)
    m = pop.measure(x)
    assert x.ndim == 1 and m['length_s'] < 0.17
    assert m['ridge']['sweep_oct'] < -4.0                         # 9.8 kHz -> 360 Hz
    b = render(pop, pop.FAMILY_V2['pro_pop'][2], 2)              # the film's blip: a few ms
    assert len(b) / SR < 0.04


def test_pop_v1_parameters_take_the_v1_path():
    _, pop, _ = _mods()
    v1 = render(pop, pop.DEFAULTS, 3)
    direct = dsp.normalize(np.asarray(pop.render_v1(dict(pop.DEFAULTS), dsp.rng(3)), np.float64), -1.0)
    assert np.allclose(v1, direct.astype(np.float32))
    assert 'style' not in pop.DEFAULTS and 'style' in pop.DEFAULTS_V2
    v2 = render(pop.V2, pop.DEFAULTS_V2, 3)
    assert v2 is not None and len(v2) != len(v1)


def test_pro_shimmer_sparkle_is_wide_and_swells():
    _, _, shimmer = _mods()
    x = render(shimmer, shimmer.FAMILY['pro_shimmer'][0], 1)
    m = shimmer.measure(x)
    assert m['corr'] < 0.5                                        # a texture: wide
    e = 20 * np.log10(np.sqrt((x.mean(1) ** 2)[:len(x) // 4800 * 4800].reshape(-1, 4800).mean(1)) + 1e-9)
    pk = int(np.argmax(e))
    assert 3 <= pk <= len(e) - 4 and e[pk] - e[0] > 15 and e[pk] - e[-1] > 15   # swell, then fade
    notes = shimmer.scale_freqs(1000, 9000, 'penta', 2)
    pcs = {int(round(12 * np.log2(f / 16.3516))) % 12 for f in notes}
    assert pcs <= {(2 + s) % 12 for s in shimmer.SCALES['penta']}


@pytest.mark.parametrize('cat', ['pro_notify', 'pro_pop', 'pro_shimmer'])
def test_family_space_pins_the_character(cat):
    mod, fam = _fam(cat)
    p = fam[1]
    sp = mod.family_space(p)
    pinned = [k for k, v in sp.items() if isinstance(v, list) and len(v) == 1]
    free = [k for k in sp if k not in pinned]
    assert free and pinned
    u = _encode(sp, p)
    q = _decode(sp, np.clip(u + 0.3, 0, 1))
    for k in pinned:
        assert q[k] == p.get(k, q[k])                              # a search can never move a character key


def test_family_members_differ():
    """palette members are audibly different: their 1/3-octave balance or envelope differs clearly"""
    from earshot.forge.tone import profile
    for cat in ('pro_notify', 'pro_pop'):
        mod, fam = _fam(cat)
        xs = [render(mod, p, 1) for p in fam]
        prof = [profile(x) for x in xs]
        lens = [len(x) for x in xs]
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                d = float(np.sqrt(np.mean((prof[i] - prof[j]) ** 2)))
                assert d > 1.5 or abs(lens[i] - lens[j]) / max(lens[i], lens[j]) > 0.15, (cat, i, j, d)
