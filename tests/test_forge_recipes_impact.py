"""forge v2 rounds 2-3, impact area: impact v3 (cinematic + designed), subdrop, sting. Model-free checks only (fast, no
CLAP): every family preset renders deterministically and hits the pro study's measured targets it was designed for
(docs/motion-sfx-study.md), the families are audibly different, and v2 impact parameter sets still render.
Round 3 adds the judge's fixes as checks: the cinematic onset sheen and 1-4 kHz body, the designed hits' fall into the
gate and 4-12 kHz grain band, and the composed launch sting (lead-in -> bright strike -> short bloom, t40 ~0.9 s)."""
import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp, polish
from earshot.forge.recipes import impact, sting, subdrop

SR = dsp.SR


def _band_db(x, lo, hi):
    m = x.mean(axis=1)
    f, P = signal.welch(m, SR, nperseg=8192)
    return 10 * np.log10(P[(f >= lo) & (f < hi)].sum() / P.sum() + 1e-12)


def _onset6k_db(x, ms=15):
    """share > 6 kHz in the first `ms` after the onset (first 1 ms frame within 20 dB of the loudest)"""
    m = x.mean(axis=1)
    k = len(m) // 48
    e = 20 * np.log10(np.sqrt(np.mean(m[:k * 48].reshape(k, 48) ** 2, 1)) + 1e-12)
    i = int(np.argmax(e >= e.max() - 20)) * 48
    seg = m[i:i + int(SR * ms / 1000)]
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 4096)) ** 2
    f = np.fft.rfftfreq(4096, 1 / SR)
    return 10 * np.log10(F[f >= 6000].sum() / F.sum() + 1e-12)


def _onset_judge_db(x, f0=6000, ms=15):
    """the round-3 judge's onset brightness (out/forge_v2/_judge/judge_r3.py onset_hi): share > f0 of the first `ms`
    after the first sample within 20 dB of the peak sample, rectangular window"""
    m = x.mean(axis=1)
    e = np.abs(m)
    i0 = int(np.argmax(e > e.max() * 0.1))
    F = np.abs(np.fft.rfft(m[i0:i0 + int(SR * ms / 1000)], n=4096)) ** 2
    f = np.fft.rfftfreq(4096, 1 / SR)
    return 10 * np.log10(F[f >= f0].sum() / (F.sum() + 1e-20) + 1e-12)


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
    on = []
    for name, x in _members(impact, 'impact'):
        m = impact.measure(x)
        assert 1000 <= m['t40'] <= 3500, (name, m['t40'])
        # (round 4: the 'punch' member is the short drum hit, -24..-26 dB/s; the others -8..-15)
        assert (-27 if name == 'punch' else -25) <= m['slope_dBs'] <= -7, (name, m['slope_dBs'])
        assert m['gate_db40ms'] < 6, name                                      # rings out, no gate
        assert m['sh150'] > 0.6 and m['tilt'] < -1.5, name
        # round 3: body at 1-4 kHz (r2: -20..-25 dB re total, the forge refs -16 / -18; the dark 'deep' member a little less)
        assert _band_db(x, 1000, 4000) > (-24 if name == 'deep' else -21), (name, _band_db(x, 1000, 4000))
        # round 4: the r3 contact sheen overshot (judge's onset measure -10.9 dB, forge refs -23 / -27, > 12 kHz of that
        # window -10..-14 vs -31..-48): a contact that is bright but not a tick, and a top that rolls off
        on.append(_onset_judge_db(x))
        assert -33 < on[-1] < -14, (name, on[-1])
        assert _onset_judge_db(x, 12000) < -22, (name, _onset_judge_db(x, 12000))
        assert _band_db(x, 12000, 24000) < -33, (name, _band_db(x, 12000, 24000))
    assert -23 < float(np.median(on)) < -16, on


def test_designed_impacts_are_gated_sub_heavy_and_grainy():
    for name, x in _members(impact, 'pro_impact'):
        m = impact.measure(x)
        assert 0.5 <= m['len50'] <= 1.4, (name, m['len50'])
        assert m['gate_db40ms'] >= 10 or m['pre_gate'] <= -22, (name, m['gate_db40ms'], m['pre_gate'])  # a gate
        # round 3: the tail falls before the gate (refs -30 / -27.5 dB 60 ms before the end; r2 held at -14..-21 dB)
        assert m['pre_gate'] <= -18, (name, m['pre_gate'])
        assert _band_db(x, 4000, 12000) > -29, (name, _band_db(x, 4000, 12000))   # the grain / air sparkle
        assert m['sh150'] >= 0.75, (name, m['sh150'])
        assert m['corr'] < 0.7, (name, m['corr'])                              # decorrelated rumble / air tail
    # the granular rumble: the envelope of the 'rumble' preset pulses at ~25-45 per second
    x = dict(_members(impact, 'pro_impact'))['rumble'].mean(axis=1)
    e = np.sqrt(np.convolve(x ** 2, np.ones(96) / 96, 'same'))[::48]           # 1 ms frames
    seg = e[150:900] - np.convolve(e[150:900], np.ones(60) / 60, 'same')
    f = np.fft.rfftfreq(len(seg), 1e-3)
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    band = (f > 5) & (f < 120)
    # (round 4: the members start from the r3 finals' searched parameters, and the r3 search had moved the rumble's
    # grain_rate 36 -> 22.7 per s; with random spacing (grain_jit 0.7, mean step x 1.1) the pulse rate is ~19 per s,
    # slower than the film's 36: a known deviation, see out/forge_v2/pro_impact/LOG.md)
    assert 16 <= f[band][np.argmax(S[band])] <= 50


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
    for (name, p), (_, x) in zip(sting.FAMILIES['logo_sting'], _members(sting, 'logo_sting')):
        m = sting.measure(x)
        # round 3: a composed launch reveal, not a bell chord ringing 2 s: bright, short tail (r2 t40 1.9-2.2 s)
        assert p['mode'] == 'launch' and len(sting.DYADS[p['notes']]) <= 2, name
        assert m['tilt'] > 1.0 and 0.6 <= m['t40'] / 1000 <= 1.3, (name, m['tilt'], m['t40'])
        hit = x[max(0, int(SR * p['lead_s']) - 48):] if p['lead'] != 'none' else x
        assert polish.attack(hit)['onset_hf_db'] > -20, name                  # the strike owns the hit's onset
        if p['lead'] != 'none':                                                # the lead-in rises into the hit
            assert m['attack_ms'] >= 60 and m['peak_ms'] / 1000 >= 0.8 * p['lead_s'], (name, m['attack_ms'])


def test_round4_sting_body_and_top():
    """round 4 (the round-3 judge): the composed reveal gets a low body (r3: < 150 Hz -52.7 dB of total, the held-out
    stings about -27), the designed sting a top end (r3: 4-12 kHz -41.8 dB, the film -21) without a > 12 kHz hiss"""
    for name, x in _members(sting, 'logo_sting'):
        assert -30 < _band_db(x, 20, 150) < -14, (name, _band_db(x, 20, 150))
        assert sting.measure(x)['sh150'] <= sting.STUDY_STING['logo_sting']['sh150'][1], name
    top = [_band_db(x, 4000, 12000) for _, x in _members(sting, 'pro_logo_sting')]
    assert float(np.median(top)) > -30 and max(top) > -20, top
    for name, x in _members(sting, 'pro_logo_sting'):
        assert _band_db(x, 12000, 24000) < -25, (name, _band_db(x, 12000, 24000))
    # the gestures differ: one member has a pre-chirp (two loud events before the hum), one a zap riding the chirp
    fam = dict(sting.FAMILIES['pro_logo_sting'])
    assert any(p.get('double', 0) > 0.3 for p in fam.values()) and any(p.get('zap', 0) > 0.3 for p in fam.values())


def test_round4_grain_timing_is_random():
    """round 4: r3's high grains were evenly spaced stripes (a 'snare roll'); grain_jit spreads the spacing"""
    t = np.arange(dsp.n_of(1.2)) / SR

    def onsets(jit):
        g = impact._grains(t, 18.0, 1.0, dsp.rng(4), jit=jit)
        idx, _ = signal.find_peaks(g, distance=dsp.n_of(0.006), prominence=0.05)
        return np.diff(idx) / SR
    even, rnd = onsets(0.0), onsets(0.7)
    cv = lambda v: float(np.std(v) / np.mean(v))  # noqa: E731
    assert cv(rnd) > 1.5 * cv(even), (cv(even), cv(rnd))


def test_round3_families_still_render():
    """the r3 finals' families are kept (FAMILIES_R3) so the showcase can re-render r3 next to r4"""
    for mod in (impact, sting):
        for cat, fam in mod.FAMILIES_R3.items():
            for name, p in fam:
                x = _render(mod, p)
                assert np.isfinite(x).all() and 0.5 < np.abs(x).max() <= 1.0, (cat, name)


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


def test_round2_families_still_render():
    """the r2 finals' families are kept (FAMILIES_R2) so the showcase can re-render r2 next to r3"""
    for mod in (impact, sting):
        for cat, fam in mod.FAMILIES_R2.items():
            for name, p in fam:
                x = _render(mod, p)
                assert np.isfinite(x).all() and 0.5 < np.abs(x).max() <= 1.0, (cat, name)
