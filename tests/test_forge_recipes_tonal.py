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
    """the round-2..4 film chirp / blip (kept in FAMILY_V2_R4: the r4 finals, final_r4/)"""
    _, pop, _ = _mods()
    p = pop.FAMILY_V2_R4['pro_pop'][0]
    x = render(pop, p, 2)
    m = pop.measure(x)
    assert x.ndim == 1 and m['length_s'] < 0.17
    assert m['ridge']['sweep_oct'] < -4.0                         # 9.8 kHz -> 360 Hz
    b = render(pop, pop.FAMILY_V2_R4['pro_pop'][2], 2)           # the film's blip: a few ms
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


def test_round3_shimmer_rings_out_wide_without_a_bed_line():
    """round 3 (the judge: shimmer tails half the bench's, mono chimes, steady lines): the real-style members ring
    out freely (t40 >= 1.5 s), are wide, and an 'off' tone bed adds no line under an air-only bed"""
    _, _, shimmer = _mods()
    fam = shimmer.FAMILY['shimmer']
    for i in (0, 2):                                              # the bar-chime sweep and the wind chimes
        x = render(shimmer, fam[i], i + 2)
        m = x.mean(1)
        h = 240
        e = 20 * np.log10(np.sqrt((m[:len(m) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-12)
        pk = int(np.argmax(e))
        below = np.nonzero(e[pk:] < e[pk] - 40)[0]
        assert below[0] * h / SR >= 1.5                            # t40
        assert shimmer.measure(x)['corr'] < 0.5                    # wide
        assert abs(m[0]) < 1e-3                                    # starts on zero
    p = {**fam[0], 'air_db': -30.0, 'tone_db': -40.0, 'verb': 0.0}
    x = render(shimmer, p, 4).mean(1)
    S = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    assert 10 * np.log10(S[(f > 150) & (f < 1000)].sum() / S.sum()) < -35   # (round 2 left a -40 dB 500 Hz line)


def test_round3_notify_bells_are_lower_struck_and_ring():
    """round 3 (the judge: no strike transient, rings too even, members 1-2 octaves above the bench bells)"""
    notify, _, _ = _mods()
    for i, p in enumerate(notify.FAMILY['notify']):
        assert p['f0'] <= 1400 and p['strike'] > -40 and p['early_db'] > 0
        if i in (0, 2):
            x = render(notify, p, i + 1)
            m = notify.measure(x)
            assert 1000 <= m['t40_ms'] <= 2600, m
            e = x.mean(1)
            pk = int(np.argmax(np.abs(e)))
            lev = lambda a, b: 20 * np.log10(np.sqrt(np.mean(e[a:b] ** 2)) + 1e-12)
            first = lev(pk, pk + 480)
            assert first - lev(pk + 4800, pk + 5280) > 6             # the strike dies fast: > 6 dB down at 0.1 s


def test_round3_pro_notify_head_has_no_drive_comb():
    """round 3: the pro ding's head is clean (no harmonic comb from the drive), its richness is inharmonic"""
    notify, _, _ = _mods()
    p = notify.FAMILY_R3['pro_notify'][2]
    assert p['head_drive'] == 0.0 and p['p3'] > 0 and p['air'] > -40
    q = {**p, 'body_db': -60.0, 'air': -60.0, 'tick_db': -40.0, 'verb': 0.0, 'p3': 0.0, 'oct': 0.0, 'bright': 0.05}
    x = render(notify, q, 2).mean(1)
    S = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    band = lambda c: S[(f > c * 0.97) & (f < c * 1.03)].sum()
    f0 = p['f0']
    assert band(3 * f0) < 1e-3 * band(f0) and band(5 * f0) < 1e-3 * band(f0)   # no odd-harmonic comb


def test_round3_pro_pop_chirps_start_on_zero_without_digital_silence():
    _, pop, _ = _mods()
    from earshot.forge import polish
    for i in (0, 1, 3, 5):
        p = pop.FAMILY_V2_R4['pro_pop'][i]
        x = render(pop, p, i + 1)
        m = x if x.ndim == 1 else x.mean(1)
        assert abs(m[0]) < 1e-3, i
        assert polish.silence(x)['silence_ms'] < 2.0, i


def test_round4_new_keys_default_to_round_3():
    """round 4 keys (notify strike_max / air_lo, pop chirp c_air) are off by default: the round-3 members render
    sample for sample; strike_max 12 removes high strike energy; R4_EDITS keys exist in the SPACEs"""
    notify, pop, _ = _mods()
    from earshot.forge.recipes import riser
    for mod, fam, keys in ((notify, notify.FAMILY['notify'][2], {'strike_max': 20.0}),
                           (notify, notify.FAMILY_R3['pro_notify'][0], {'air_lo': 0.0}),
                           (pop, pop.FAMILY_V2_R4['pro_pop'][0], {'c_air': -60.0})):
        base = {k: v for k, v in fam.items() if k not in keys}
        assert np.array_equal(render(mod, base, 3), render(mod, dict(base, **keys), 3))
    bell = dict(notify.FAMILY['notify'][2], strike=-8.0)
    hi = lambda x: np.sum(np.abs(np.fft.rfft((x if x.ndim == 1 else x.mean(1))[:dsp.n_of(0.05)]))[int(0.05 * 12000):] ** 2)
    assert hi(render(notify, dict(bell, strike_max=12.0), 3)) < hi(render(notify, bell, 3))
    for mod, eds, space in ((notify, notify.R4_EDITS, notify.SPACE), (pop, pop.R4_EDITS, pop.SPACE_V2),
                            (riser, riser.R4_EDITS, riser.SPACE)):
        for cat, ed in eds.items():
            for e in (ed if isinstance(ed, list) else [ed]):
                assert set(e) <= set(space), (cat, set(e) - set(space))


def _band(x, c, w=0.03):
    m = x if x.ndim == 1 else x.mean(1)
    S = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return S[(f > c * (1 - w)) & (f < c * (1 + w))].sum() + 1e-30


def test_round5_pro_pop_is_a_natural_pop():
    """round 5 (polish-pro; the round-4 judge: CLAP still says '8-bit'): every pro_pop member is the natural chirp
    (c_nat): no drive, no stepped gate; a rounded body with no square-ish (odd) harmonics; the pitch falls fast and
    settles; the level decays exponentially (straight in dB); a tiny air puff; near mono, pro_pop length"""
    _, pop, _ = _mods()
    from earshot.forge import polish
    from earshot.forge.critic import study_measures
    fam = pop.FAMILY_V2_R5['pro_pop']                              # round 6: the round-5 family moved here (final_r5/)
    assert len(fam) == len(pop.FAMILY_V2_NAMES_R5['pro_pop'])
    for i, p in enumerate(fam):
        assert p['c_nat'] == 1.0 and p['drive'] <= 0.2 and p['steps'] == 0.0, i
        x = render(pop, p, i + 1)
        m = x if x.ndim == 1 else x.mean(1)
        assert abs(m[0]) < 1e-3 and polish.silence(x)['silence_ms'] < 2.0, i
        sm = study_measures(x)
        assert 0.05 <= sm['length'] <= 0.2 and sm['corr'] >= 0.95, (i, sm['length'], sm['corr'])
        # the dry body (no puff / room / pre-blip): after the fall, no 3rd / 5th harmonic, a quiet 2nd
        q = {**p, 'c_puff': -60.0, 'c_thump': -60.0, 'c_room': 0.0, 'c_pre': -60.0, 'f_hi': p['f_lo'] * 1.0001}
        y = render(pop, q, 1)
        lo = p['f_lo']
        assert _band(y, 3 * lo) < 1e-3 * _band(y, lo) and _band(y, 5 * lo) < 1e-3 * _band(y, lo), i
        assert _band(y, 2 * lo) < 0.05 * _band(y, lo), i
        # an exponential decay: the 4 ms envelope from -3 to -36 dB under the peak lies on a line (r^2 > 0.95)
        h = 4 * SR // 1000
        k = len(y) // h
        yy = y if y.ndim == 1 else y.mean(1)
        e = 20 * np.log10(np.sqrt((yy[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12)
        pk = int(np.argmax(e))
        sel = np.nonzero((e[pk:] < e[pk] - 3) & (e[pk:] > e[pk] - 36))[0] + pk
        sel = sel[sel < (np.nonzero(e[pk:] < e[pk] - 36)[0][0] + pk if (e[pk:] < e[pk] - 36).any() else k)]
        cc = np.corrcoef(sel, e[sel])[0, 1]
        assert cc ** 2 > 0.95, (i, cc)
    # the pitch falls (start above end) in the chirps
    for i in (0, 1, 3, 5):
        r = pop.measure(render(pop, fam[i], 1))['ridge']
        assert r['start_hz'] > 1.3 * r['end_hz'], (i, r)


def test_round5_pro_notify_is_the_clean_film_ding():
    """round 5 (polish-pro): the film's two-stage ding made clean: a 2 kHz-ish head on the semitone ladder with the
    ~3.4x partial, no drive comb, no noise breath, no struck-bar modes; attack <= 15 ms, t20 22-45 ms, t40 150-340 ms
    (the film dings: attack 1-15, t20 22-30 / 27-42 by the study, t40 190-330)"""
    notify, _, _ = _mods()
    fam = notify.FAMILY['pro_notify']
    f0s = []
    for i, p in enumerate(fam):
        assert p['style'] == 'ding' and p['drive'] == 0.0 and p['head_drive'] == 0.0, i
        assert p['air'] <= -59 and p['p3'] == 0.0, i
        x = render(notify, p, i + 3)
        m = notify.measure(x)
        assert m['attack_ms'] <= 15 and 22 <= m['t20_ms'] <= 45 and 150 <= m['t40_ms'] <= 340, (i, m)
        assert m['shares'][5] < -40, (i, m['shares'])                       # nothing up high (no breath)
        f0 = p['f0']
        head = x[:int(0.03 * SR)]
        pp = _band(head, p['p2'] * f0, 0.02) / _band(head, f0, 0.02)
        assert 10 ** (-3.2) < pp < 10 ** (-0.6), (i, 10 * np.log10(pp))      # the 3.4x partial: -32 .. -6 dB
        assert _band(head, 3 * f0, 0.02) < 1e-3 * _band(head, f0, 0.02), i  # no odd-harmonic comb
        f0s.append(f0)
    steps = np.diff(np.log2(sorted(f0s))) * 12
    assert np.allclose(steps, 1.0, atol=0.01)                                # a semitone family
    for i, p in enumerate(notify.FAMILY_R3['pro_notify']):                   # the round-3 members still render
        assert render(notify, p, 1) is not None, i


def test_round6_pro_pop_natural_and_varied():
    """round 6 (pop-notify; the round-5 judge: natural but four near-steady tones, variety lost): every member is a
    natural pop (n6: no drive, no stepped gate, soft partials) whose pitch moves through its decay; the family covers
    falling and rising pitches, low and high registers and a double pop; the round-5 sets render exactly as before"""
    _, pop, _ = _mods()
    from earshot.forge import polish
    from earshot.forge.critic import study_measures
    fam = pop.FAMILY_V2['pro_pop']
    assert fam is pop.FAMILY_V2_R6['pro_pop'] and len(fam) == len(pop.FAMILY_V2_NAMES['pro_pop']) == 6
    for i, p in enumerate(fam):
        assert p['n6'] == 1.0 and p['drive'] <= 0.2 and p['steps'] == 0.0 and p['n_h3'] <= 0.05, i
        assert abs(np.log2(p['n_f0'] / p['n_f1'])) >= 0.25, i                   # the pitch moves (no steady beep)
        x = render(pop, p, i + 1)
        m = x if x.ndim == 1 else x.mean(1)
        assert abs(m[0]) < 1e-3 and abs(m[-1]) < 1e-3 and polish.silence(x)['silence_ms'] < 2.0, i
        sm = study_measures(x)
        assert 0.05 <= sm['length'] <= 0.2 and sm['corr'] >= 0.95, (i, sm['length'], sm['corr'])
        assert pop.pro_penalty(x, 'pro_pop') <= 1.0, i                          # (the chirp's t40 ~80 ms vs 70)
    up = [p for p in fam if p['n_f1'] > p['n_f0']]
    down = [p for p in fam if p['n_f1'] < p['n_f0']]
    assert len(up) >= 2 and len(down) >= 3
    assert any(p['n_dbl'] > -59 for p in fam)                                   # a double pop
    lo = min(min(p['n_f0'], p['n_f1']) for p in fam)
    hi = max(max(p['n_f0'], p['n_f1']) for p in fam)
    assert lo < 300 and hi > 8000                                               # registers far apart
    for p in pop.FAMILY_V2_R5['pro_pop'][:2] + pop.FAMILY_V2_R4['pro_pop'][:2]:
        base = {k: v for k, v in p.items() if not k.startswith('n_') and k != 'n6'}
        assert np.array_equal(render(pop, base, 3), render(pop, dict(base, n6=0.0), 3))


def test_round6_pro_notify_chime_family():
    """round 6 (not delivered; kept as notify.FAMILY_R6 / final_r6/): five distinct dings around the film targets:
    heads on the semitone ladder, attack <= 15 ms, t20 20-56 ms, t40 70-360 ms (the wood bar is short), nothing under
    500 Hz, no drive"""
    notify, _, _ = _mods()
    fam = notify.FAMILY_R6['pro_notify']
    assert len(fam) == 5 and len({p.get('mat', 'film') + p['style'] + str(p.get('c_motif')) for p in fam}) == 5
    for i, p in enumerate(fam):
        assert p['drive'] == 0.0 and p['head_drive'] == 0.0 and p['air'] <= -59, i
        x = render(notify, p, i + 3)
        assert render(notify, p, i + 3).tolist() == x.tolist()
        m = notify.measure(x)
        assert m['attack_ms'] <= 15 and 20 <= m['t20_ms'] <= 56 and 70 <= m['t40_ms'] <= 360, (i, m)  # fm_bell t20 50-54; wood t40 78-142 (short)
        lo = 10 * np.log10(sum(10 ** (s / 10) for s in m['shares'][:2]))
        assert lo < -38 and 1700 <= m['head_hz'] <= 2400, (i, m)
        assert notify.pro_penalty(x, 'pro_notify') <= 1.0, i
    assert notify.FAMILY['pro_notify'] is not notify.FAMILY_R6['pro_notify']   # delivered: round 5
    assert notify.FAMILY['pro_notify'][0]['style'] == 'ding'
