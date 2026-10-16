"""forge v2 digital recipes: keypress (v2 keys), typing (designed key + RUN), glitch. Fast, model-free."""
import numpy as np

from earshot.forge import dsp
from earshot.forge.recipes import glitch, keypress, typing
from earshot.forge.search import render

SR = 48000


def _mono(x):
    return x.mean(1) if x.ndim == 2 else x


def test_recipes_render_finite_and_deterministic():
    for rec in (keypress, typing, typing.RUN, glitch):
        a = render(rec, rec.DEFAULTS, 11)
        b = render(rec, rec.DEFAULTS, 11)
        assert a is not None and np.isfinite(a).all()
        assert np.array_equal(a, b)
        assert set(rec.DEFAULTS) >= set(rec.SPACE)          # every searched parameter has a start value


def test_keypress_v1_params_keep_the_v1_path():
    """parameter sets without the v2 keys (top_lvl, jacket_*, pre_gap) render the v1 sound: the long v1 lead"""
    v1 = render(keypress, keypress.V1_DEFAULTS, 3)
    v2 = render(keypress, keypress.DEFAULTS, 3)
    assert 'top_lvl' not in keypress.V1_DEFAULTS
    assert len(v2) < len(v1)                                    # v2 edit is shorter (clip 0.24 vs 0.38 s)
    assert keypress.key_stats(v2)['t40_ms'] <= 230
    dull = keypress.key_stats(render(keypress, dict(keypress.DEFAULTS, top_lvl=0.0), 3))['top_db']
    crisp = keypress.key_stats(render(keypress, dict(keypress.DEFAULTS, top_lvl=2.2), 3))['top_db']
    assert crisp > dull + 1                                     # the keycap rattle layer lifts the 2-12 kHz share


def test_keypress_jacket_adds_a_pre_click():
    p = dict(keypress.DEFAULTS, jacket_lvl=0.6, jacket_dt=0.01)
    x = _mono(render(keypress, p, 5))
    e = 20 * np.log10(np.abs(x[: int(0.013 * SR)]).max() / np.abs(x).max() + 1e-12)
    assert e > -30                                              # something audible before the bottom-out


def test_designed_key_meets_the_study_box():
    for s in range(4):
        x = render(typing, typing.DEFAULTS, s)
        st = typing.key_stats(x)
        assert st['lo2k_db'] < -30                              # nothing under 2 kHz
        assert 1 <= st['attack_ms'] <= 6
        assert 0.045 <= st['length_s'] <= 0.11
        c = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
        assert c > 0.999                                        # mono, as the films' keys


def test_typed_run_spacing_and_bed():
    x = render(typing.RUN, typing.RUN.DEFAULTS, 2)
    st = typing.run_stats(x)
    assert st['keys'] >= 12 and 65 <= st['med_ms'] <= 110      # ~11.5 keys/s (reference run: median ~87 ms)
    assert np.abs(_mono(x)).min() >= 0 and (np.abs(_mono(x)[SR // 10:-SR // 10]) > 0).all()   # no digital silence


def test_glitch_wide_bright_accelerating_hard_end():
    x = render(glitch, glitch.DEFAULTS, 1)
    st = glitch.glitch_stats(x)
    assert st['corr'] < 0.4                                     # decorrelated texture
    assert st['centroid_hz'] > 5000
    assert -75 < st['end_db'] < -30                             # stops on the cut, but not chopped
    ts = glitch._grid(glitch.DEFAULTS, 0.2, 0.65)
    assert ts[0][1] > ts[-1][1]                                 # the stutter speeds up
    assert len(x) / SR < glitch.DEFAULTS['dur'] + 0.12


def test_study_extras_are_bounded_penalties():
    xs = [render(typing, typing.DEFAULTS, 0), render(glitch, glitch.DEFAULTS, 0), render(keypress, keypress.DEFAULTS, 0)]
    for fn, x in ((typing.study_extra, xs[0]), (glitch.study_extra, xs[1]), (glitch.generic_extra, xs[1]),
                  (keypress.key_extra, xs[2])):
        v = fn(x, {})
        assert -3.0 <= v <= 0.0
    silence = np.zeros((SR // 4, 2))
    silence[100:200] = dsp.rng(0).standard_normal((100, 2)) * 0.1
    assert glitch.study_extra(silence, {}) < -0.5               # a click is not a glitch
