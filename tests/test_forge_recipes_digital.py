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


# ------------------------------------------------------------------------------------------------ round 3
def _corr(x):
    a, b = x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def test_keypress_width_folds_the_pair_and_keeps_mid():
    """r3 `width`: the spaced pair folded towards mono (r2 finals: corr 0.04-0.47); the mid (what mono checks hear)
    is unchanged, and parameter sets without the key keep the r2 pair"""
    wide = {k: v for k, v in keypress.DEFAULTS.items() if k != 'width'}
    a, b = render(keypress, wide, 4), render(keypress, dict(wide, width=0.2), 4)
    assert _corr(b) > 0.9 and _corr(b) >= _corr(a)
    m0 = render(keypress, dict(wide, width=0.0), 4)
    assert _corr(m0) > 0.99                                     # width 0: mono but for the mic floor
    ma, mb = _mono(a), _mono(b)
    assert np.allclose(ma / np.abs(ma).max(), mb / np.abs(mb).max(), atol=1e-3)
    assert 'width' in keypress.SPACE and keypress.DEFAULTS['width'] <= 0.3


def test_designed_key_r3_characters_stay_in_the_band():
    """grain (velvet crackle) and tink_glide (pitched blip) change the texture, never the band: < 2 kHz stays 30 dB
    down; the r3 keys at 0 render the r2 key exactly"""
    base = typing.render(typing.DEFAULTS, dsp.rng(5))
    r2 = typing.render({k: v for k, v in typing.DEFAULTS.items() if k not in ('grain', 'grain_rate', 'tink_glide')},
                       dsp.rng(5))
    assert np.array_equal(base, r2)
    for over in ({'grain': 0.9, 'grain_rate': 400}, {'tink_db': 0.0, 'tink_glide': -12.0}):
        x = render(typing, dict(typing.DEFAULTS, **over), 6)
        assert typing.key_stats(x)['lo2k_db'] < -30
        assert not np.allclose(x, render(typing, typing.DEFAULTS, 6)[: len(x)])


def test_glitch_r3_drama_and_acceleration():
    """skip + accent_db + short bursts: the median 2 ms level sits far under the bursts (film -35 dB, r2 finals
    -20..-25) and the burst onsets show the stutter speeding up (late intervals shorter than early ones)"""
    p = dict(glitch.DEFAULTS, decay=0.22, gate=0.5, crackle=0.33, crackle_db=-21, floor_db=-44, lvl_jit=2.5, skip=0.2,
             accent_db=6, grid0=0.034, grid1=0.022, accel=1.2, low_db=-32)
    meds, acc = [], 0
    for s in range(3):
        st = glitch.glitch_stats(render(glitch, p, 40 + s))
        meds.append(st['med2_db'])
        acc += st['early_ms'] is not None and st['late_ms'] < st['early_ms']
        assert st['corr'] < 0.4 and -75 < st['end_db'] < -30
    dense = glitch.glitch_stats(render(glitch, glitch.DEFAULTS, 40))['med2_db']
    assert np.median(meds) < -30 and np.median(meds) < dense - 5
    assert acc >= 2
    assert glitch.study_extra(render(glitch, p, 40), {}) > -1.0


# ---------------------------------------------------------------------------------------------- round 6 (glitch)
def _r6(name):
    return dict(dict(glitch.FAMILY6)[name])


def test_glitch_r6_older_sets_keep_their_path():
    """only parameter sets with `roll0` take render6; an r3-style set renders exactly as before (no r6 key leaks)"""
    assert 'roll0' not in glitch.DEFAULTS and set(glitch.DEFAULTS) <= set(glitch.SPACE)
    x = glitch.render(dict(glitch.DEFAULTS), dsp.rng(5))
    y = glitch.render(dict(glitch.DEFAULTS), dsp.rng(5))
    assert np.array_equal(x, y) and len(x) / SR < glitch.DEFAULTS['dur'] + 0.12


def test_glitch_r6_family_renders_wide_hard_ended_and_clean():
    assert 5 <= len(glitch.FAMILY6) <= 6
    for name, p in glitch.FAMILY6:
        assert 'roll0' in p and all(k in glitch.SPACE6 for k in p), name
        x = glitch.render(dict(p), dsp.rng(3))
        assert np.array_equal(x, glitch.render(dict(p), dsp.rng(3))), name           # pure function of params + seed
        assert x.ndim == 2 and np.isfinite(x).all() and 0.5 < np.abs(x).max() <= 0.9, name
        st = glitch.glitch_stats(x)
        assert st['corr'] < 0.5, (name, st['corr'])                                    # wide (the film: 0.04)
        assert st['end_db'] < -45, (name, st['end_db'])                                # ends on the cut, not chopped
        assert st['bursts'] >= 6, (name, st['bursts'])
        assert abs(float(x.mean())) < 1e-3, name                                      # no DC from the low thumps
        m = np.abs(x.mean(1))
        aud = np.nonzero(m > m.max() * 1e-3)[0]
        assert (m[aud[0]:aud[-1]] < 1e-6).sum() < 96, name                            # dropouts are not digital silence


def test_glitch_r6_roll_speeds_up_and_sample_rate_drops():
    p = {**glitch.DEFAULTS6, **_r6('stutter')}
    ts = glitch._grid(p, p['dur'] * p['crackle'], p['dur'])
    assert ts[0][1] > 2 * ts[-1][1]                                                    # slots 70 -> 20 ms
    assert p['roll1'] < p['roll0'] / 4 and p['srate1'] < p['srate0'] / 4 and p['bits1'] < p['bits0']
    # the sample-and-hold drop shows as lost top end: the last 150 ms is darker than the first 150 ms of the roll
    x = glitch.render(_r6('dropout'), dsp.rng(3)).mean(1)

    def hi_share(seg):
        F = np.abs(np.fft.rfft(seg)) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return F[f > 4000].sum() / (F.sum() + 1e-20)
    n = int(0.15 * SR)
    assert hi_share(x[-n - int(0.03 * SR):-int(0.03 * SR)]) < hi_share(x[:n])
