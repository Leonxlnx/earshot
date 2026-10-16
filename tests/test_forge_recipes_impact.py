"""forge v2 round 2, impact area: impact v3 (cinematic + designed), subdrop, sting. Model-free checks only (fast, no
CLAP): every family preset renders deterministically and hits the pro study's measured targets it was designed for
(docs/motion-sfx-study.md), the families are audibly different, and v2 impact parameter sets still render."""
import numpy as np
import pytest

from earshot.forge import dsp
from earshot.forge.recipes import impact, sting, subdrop

SR = dsp.SR


def _render(mod, p, seed=3):
    return np.asarray(mod.render(dict(p), dsp.rng(seed)), np.float64)


def _members(mod, cat):
    return [(name, _render(mod, p)) for name, p in mod.FAMILIES[cat]]


@pytest.mark.parametrize('mod,cat', [(impact, 'impact'), (impact, 'pro_impact'), (subdrop, 'sub_drop'),
                                     (subdrop, 'pro_sub_drop'), (sting, 'logo_sting'), (sting, 'pro_logo_sting')])
def test_family_presets_render_clean_and_deterministic(mod, cat):
    assert 4 <= len(mod.FAMILIES[cat]) <= 6
    for name, p in mod.FAMILIES[cat]:
        x = _render(mod, p)
        assert x.ndim == 2 and x.shape[1] == 2, (cat, name)
        assert np.isfinite(x).all() and 0.5 < np.abs(x).max() <= 1.0, (cat, name)
        assert 0.3 < len(x) / SR < 8.0, (cat, name)
        assert abs(float(np.mean(x))) < 1e-3, (cat, name)                    # no DC
        assert np.abs(x[:16]).max() < 0.2 and np.abs(x[-16:]).max() < 0.05, (cat, name)   # no edge jumps
        y = _render(mod, p)
        assert np.array_equal(x, y), (cat, name)                               # a pure function of params + seed
    # the search space of the category contains every preset (the family stays searchable around itself)
    sp = mod.space_for(cat)
    for name, p in mod.FAMILIES[cat]:
        for k, spec in sp.items():
            assert (p[k] in spec) if isinstance(spec, list) else (spec[0] <= p[k] <= spec[1]), (cat, name, k)
    rec = mod.recipe_for(cat, 1)
    assert rec.DEFAULTS == dict(mod.FAMILIES[cat][1][1]) and rec.SPACE == sp


def test_cinematic_impacts_have_long_natural_tails():
    """v1's tell was its short tail (t40 x 0.5 of the held-out median): the cinematic family decays straight in dB
    over 1-3 s like the real kit hits (forge refs: t40 1.4-2.0 s, -12..-18 dB/s)"""
    for name, x in _members(impact, 'impact'):
        m = impact.measure(x)
        assert 1000 <= m['t40'] <= 3500, (name, m['t40'])
        assert -25 <= m['slope_dBs'] <= -7, (name, m['slope_dBs'])
        assert m['gate_db40ms'] < 6, name                                      # rings out, no gate
        assert m['sh150'] > 0.6 and m['tilt'] < -2.5, name


def test_designed_impacts_are_gated_sub_heavy_and_grainy():
    for name, x in _members(impact, 'pro_impact'):
        m = impact.measure(x)
        assert 0.5 <= m['len50'] <= 1.4, (name, m['len50'])
        assert m['gate_db40ms'] >= 10, (name, m['gate_db40ms'])               # a hard gate, not a ring-out
        assert m['sh150'] >= 0.75, (name, m['sh150'])
        assert m['corr'] < 0.7, (name, m['corr'])                              # decorrelated rumble / air tail
    # the granular rumble: the envelope of the 'rumble' preset pulses at ~25-45 per second
    x = dict(_members(impact, 'pro_impact'))['rumble'].mean(axis=1)
    e = np.sqrt(np.convolve(x ** 2, np.ones(96) / 96, 'same'))[::48]           # 1 ms frames
    seg = e[150:900] - np.convolve(e[150:900], np.ones(60) / 60, 'same')
    f = np.fft.rfftfreq(len(seg), 1e-3)
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    band = (f > 5) & (f < 120)
    assert 22 <= f[band][np.argmax(S[band])] <= 50


def test_sub_drops_fall_and_bloom():
    for name, x in _members(subdrop, 'pro_sub_drop'):
        m = subdrop.measure(x)
        r0, _, r1 = m['ridge']
        assert r0 > r1 and 34 <= r1 <= 52, (name, m['ridge'])                  # 64 -> 44 Hz class drop
        assert 200 <= m['attack_ms'] <= 470, (name, m['attack_ms'])           # blooms after the click
        assert -60 <= m['slope_dBs'] <= -30, (name, m['slope_dBs'])           # ~40 dB/s, straight in dB
        assert m['sh150'] >= 0.85 and m['corr'] >= 0.9, name
        assert subdrop.study_penalty(m, subdrop.STUDY_SUB['pro_sub_drop']) < 1.0, name
    for name, x in _members(subdrop, 'sub_drop'):
        m = subdrop.measure(x)
        assert m['ridge'][0] >= m['ridge'][2] and m['sh150'] >= 0.85, name


def test_stings():
    for name, x in _members(sting, 'pro_logo_sting'):
        m = sting.measure(x)
        assert m['corr'] > 0.99, name                                          # mono
        assert m['t40'] <= 750 and m['ridge'][0] >= 600 and m['ridge'][1] <= 300, (name, m['t40'], m['ridge'])  # chirp -> hum
    for name, x in _members(sting, 'logo_sting'):
        m = sting.measure(x)
        assert m['tilt'] > -1.5 and 0.6 <= m['t40'] / 1000 <= 5.0, (name, m['tilt'], m['t40'])   # a bright ring-out


def test_families_are_not_clones():
    """round 1's variants were near clones; each family's members differ in balance or envelope (model-free)"""
    for mod, cat in ((impact, 'impact'), (impact, 'pro_impact'), (subdrop, 'sub_drop'), (subdrop, 'pro_sub_drop'),
                     (sting, 'logo_sting'), (sting, 'pro_logo_sting')):
        ms = [mod.measure(x) for _, x in _members(mod, cat)]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                bal = np.sqrt(np.mean([(ms[i]['oct'][k] - ms[j]['oct'][k]) ** 2 for k in ms[i]['oct']]))
                env = abs(np.log2(max(ms[i]['t40'], 1) / max(ms[j]['t40'], 1)))
                ln = abs(np.log2(ms[i]['len50'] / max(ms[j]['len50'], 1e-3)))
                assert bal > 1.5 or env > 0.25 or ln > 0.25, (cat, i, j, bal, env, ln)


def test_v2_parameter_sets_still_render():
    x = _render(impact, impact.DEFAULTS_V2, seed=1)
    assert x.ndim == 2 and np.isfinite(x).all() and len(x) / SR > 2.0
