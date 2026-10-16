"""Fast tests for the forge v2 UI recipes (click, toggle, hover, popin): every family renders a clean, short sound that
hits the pro-study targets it was designed for (docs/motion-sfx-study.md), deterministically, and the families of
one category are audibly different (a palette, not clones). No models are loaded."""
import numpy as np
import pytest

from earshot.forge import load_recipe
from earshot.forge.search import _decode, _encode, render

SR = 48000
UI = {'click': ('click', 'pro_click'), 'toggle': ('toggle',), 'hover': ('hover_tick', 'pro_hover_tick'),
      'popin': ('pro_pop_in', 'pro_tap', 'tap')}
CASES = [(r, c, f) for r, cats in UI.items() for c in cats for f in load_recipe(r).FAMILIES[c]]


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _bands(x):
    """energy share (dB re total) below 150 Hz, below 500 Hz and the spectral centroid (Hz)"""
    m = _mono(x)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    tot = P.sum() + 1e-30
    return (10 * np.log10(P[f < 150].sum() / tot + 1e-12), 10 * np.log10(P[f < 500].sum() / tot + 1e-12),
            float((f * P).sum() / tot))


def _env_db(x, ms=1.0):
    m = _mono(x)
    h = int(SR * ms / 1000)
    k = len(m) // h
    return 10 * np.log10(np.mean(m[:k * h].reshape(k, h) ** 2, 1) + 1e-20)


def _t_fall(x, db):
    """ms from the (10 ms smoothed) peak until the level first falls db under it"""
    e = 10 ** (_env_db(x) / 10)
    s = 10 * np.log10(np.convolve(e, np.ones(10) / 10, 'same') + 1e-20)
    i = int(np.argmax(s))
    j = np.nonzero(s[i:] < s[i] - db)[0]
    return float(j[0]) if len(j) else float(len(s) - i)


@pytest.mark.parametrize('rec_name,cat,fam', CASES)
def test_every_family_renders_clean_and_short(rec_name, cat, fam):
    rec = load_recipe(rec_name)
    p = rec.family_params(cat, fam)
    x = render(rec, p, 3)
    assert x is not None and np.isfinite(x).all()
    assert x.ndim == 2 and x.shape[1] == 2                       # stereo out (mono objects are L == R)
    assert 0.02 < len(x) / SR < 0.45
    assert np.abs(x).max() <= 0.9 and abs(x[0]).max() < 1e-3 and abs(x[-1]).max() < 1e-3   # no edge clicks
    assert abs(float(_mono(x).mean())) < 0.01                    # no DC
    on = int(np.argmax(np.abs(_mono(x)) > 0.03))
    assert on < int(0.006 * SR), 'starts on (almost) the first sample: no lag for frame sync'


@pytest.mark.parametrize('rec_name', list(UI))
def test_space_and_families_are_consistent(rec_name):
    rec = load_recipe(rec_name)
    assert set(rec.DEFAULTS) == set(rec.SPACE)
    for cats in rec.FAMILIES.values():
        for fam in cats.values():
            assert set(fam) <= set(rec.SPACE)
    for cat in rec.FAMILIES:
        assert cat in rec.TARGETS
        for fam in rec.FAMILIES[cat]:
            sp = rec.family_space(cat, fam)
            p = rec.family_params(cat, fam)
            assert set(sp) == set(rec.SPACE)
            u = _encode(sp, p)
            q = _decode(sp, u)
            for k, v in q.items():                                # the family centre lies inside its own space
                spec = sp[k]
                if isinstance(spec, list):
                    assert v in spec
                else:
                    assert spec[0] - 1e-9 <= v <= spec[1] + 1e-9, (cat, fam, k)


@pytest.mark.parametrize('rec_name', list(UI))
def test_deterministic_for_a_seed(rec_name):
    rec = load_recipe(rec_name)
    a, b = render(rec, rec.DEFAULTS, 11), render(rec, rec.DEFAULTS, 11)
    assert np.array_equal(a, b)
    c = render(rec, rec.DEFAULTS, 12)
    assert not np.array_equal(a, c)                              # round robin: a new seed is a new take


def test_pro_click_is_short_and_crisp_with_a_body_where_the_family_has_one():
    """round 3: ref_2's bright idiom stays in tink / glass; the other films' clicks carry a body (the study's ref_5
    mouse click; the round-2 judge: 150 Hz-1 kHz -41 dB of total made vs -6 in the held-out film clicks)"""
    rec = load_recipe('click')
    for fam in rec.FAMILIES['pro_click']:
        x = render(rec, rec.family_params('pro_click', fam), 1)
        lo150, lo500, cen = _bands(x)
        if fam in ('tink', 'glass'):
            assert cen > 4500, fam                               # study: centroid 5-10 kHz
        if fam in ('mouse', 'soft', 'snap'):
            assert _share(x, 150, 1000) > -12, fam              # a body
        assert lo150 < -15, fam                                  # no low thump
        assert _t_fall(x, 20) <= 25 and len(x) / SR <= 0.07, fam  # t20 6-16 ms (+ smoothing), 30-60 ms
        e = _env_db(x)
        assert int(np.argmax(e)) - int(np.argmax(e > e.max() - 20)) <= 3, fam   # attack 0-3 ms


def _share(x, lo, hi):
    m = _mono(x)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return 10 * np.log10(P[(f >= lo) & (f < hi)].sum() / (P.sum() + 1e-30) + 1e-12)


def test_click_fills_1_4_khz_and_rings_no_narrow_line():
    """round 3: the real clicks' balance is 1-4 kHz -3.7 / > 12 kHz -29 dB of total (forge refs); the round-2 clicks
    had a 1-4 kHz hole and narrow 6 kHz / 250-500 Hz lines ringing through the room tail"""
    rec = load_recipe('click')
    sh = []
    for fam in rec.FAMILIES['click']:
        x = render(rec, rec.family_params('click', fam), 2)
        sh.append((_share(x, 1000, 4000), _share(x, 12000, 24000)))
        m = _mono(x)
        on = int(np.argmax(np.abs(m) > 0.1 * np.abs(m).max()))
        seg = m[on + int(0.02 * SR):]                            # 20 ms after the onset: the room tail
        F = 10 * np.log10(np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8192)) ** 2 + 1e-20)
        f = np.fft.rfftfreq(8192, 1 / SR)
        sel = (f > 200) & (f < 10000)
        prom = max(F[i] - np.median(F[(f > f[i] / 1.26) & (f < f[i] * 1.26)]) for i in np.nonzero(sel)[0][::3])
        assert prom < 30, (fam, round(float(prom), 1))
    sh = np.array(sh)
    assert np.median(sh[:, 0]) > -8 and np.median(sh[:, 1]) < -18, sh


def test_pro_hover_tick_peaks_on_a_contact_spike():
    """round 3: the pro ticks' peak is a broadband contact impulse that falls 20 dB within ~1-3 ms"""
    rec = load_recipe('hover')
    for fam in rec.FAMILIES['pro_hover_tick']:
        x = render(rec, rec.family_params('pro_hover_tick', fam), 2)
        e = _env_db(x)
        i = int(np.argmax(e))
        assert int(np.argmax(e[i:] < e[i] - 20)) <= 4, fam
        L, R = x[:, 0], x[:, 1]
        corr = float(np.sum(L * R) / np.sqrt(np.sum(L * L) * np.sum(R * R)))
        assert corr > -0.3, fam                                  # wide, never anti-phase


def test_toggle_first_contact_is_the_loudest_in_most_families():
    rec = load_recipe('toggle')
    first = 0
    for fam in rec.FAMILIES['toggle']:
        p = rec.family_params('toggle', fam)
        x = render(rec, p, 3)
        e = _env_db(x)
        cut = int((0.002 + p['gap'] * 0.7) * 1000)
        first += int(e[:cut].max() >= e[cut:].max())
        assert _share(x, 150, 1000) < -10, fam                   # round 3: no heavy body
    assert first >= 4


def test_taps_have_a_contact_and_a_two_stage_decay():
    rec = load_recipe('popin')
    for fam in rec.FAMILIES['tap']:
        x = render(rec, rec.family_params('tap', fam), 4)
        assert _t_fall(x, 20) <= 25 and _t_fall(x, 40) >= 40, fam


def test_hover_has_nothing_under_500_hz():
    rec = load_recipe('hover')
    for cat in rec.FAMILIES:
        for fam in rec.FAMILIES[cat]:
            x = render(rec, rec.family_params(cat, fam), 2)
            lo150, lo500, cen = _bands(x)
            lim = -30 if cat == 'pro_hover_tick' else -35            # round 3: the contact spike's skirt (the pro refs
            assert lo500 < lim and cen > 2800, (cat, fam)           # themselves: 150 Hz-1 kHz -23 dB of total)
            assert len(x) / SR <= 0.12


def test_pop_in_is_a_low_thump_with_a_thin_tick():
    rec = load_recipe('popin')
    for fam in rec.FAMILIES['pro_pop_in']:
        x = render(rec, rec.family_params('pro_pop_in', fam), 4)
        lo150, lo500, cen = _bands(x)
        assert lo150 > -3.0, fam                                 # study: < 150 Hz holds all but 0.5 dB
        assert 0.09 <= len(x) / SR <= 0.28, fam                  # the sound is ~50-80 ms; the file holds a double hit
        assert _t_fall(x, 20) <= 60 and _t_fall(x, 40) <= 130, fam   # t20 16-35, t40 30-80 ms (hold, then stop)
        L, R = x[:, 0], x[:, 1]
        corr = float(np.sum(L * R) / np.sqrt(np.sum(L * L) * np.sum(R * R)))
        assert 0.3 < corr < 0.95, fam                            # moderately wide, mono-safe


def test_toggle_on_rises_and_off_falls():
    rec = load_recipe('toggle')

    def contacts(x):
        """spectral centroid of the first and second contact (split at the gap midpoint)"""
        p = rec.family_params('toggle', fam)
        m = _mono(x)
        cut = int((0.002 + p['gap'] / 2) * SR)
        e = np.abs(m)
        b0 = cut + int(np.argmax(e[cut:] > 0.3 * e[cut:].max()))   # contact B's onset (the gap is jittered per take)
        out = []
        for seg in (m[:cut], m[b0:b0 + int(0.02 * SR)]):
            P = np.abs(np.fft.rfft(seg, 4096)) ** 2
            f = np.fft.rfftfreq(4096, 1 / SR)
            out.append(float((f * P).sum() / P.sum()))
        return out
    for fam, up in (('on', True), ('off', False)):
        x = render(rec, rec.family_params('toggle', fam), 5)
        a, b = contacts(x)
        assert (b > a) == up, (fam, a, b)


def test_families_of_a_category_differ():
    """a palette, not clones: the family centres of one category differ in balance or envelope"""
    for rec_name, cats in UI.items():
        rec = load_recipe(rec_name)
        for cat in cats:
            rows = []
            for fam in rec.FAMILIES[cat]:
                x = render(rec, rec.family_params(cat, fam), 6)
                lo150, lo500, cen = _bands(x)
                rows.append((np.log2(cen), _t_fall(x, 20) / 10, len(x) / SR * 20, lo500 / 10))
            rows = np.array(rows)
            d = [np.linalg.norm(rows[i] - rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
            assert min(d) > 0.1, (rec_name, cat, round(min(d), 3))


def _share(x, lo, hi):
    m = _mono(x)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return 10 * np.log10(P[(f >= lo) & (f < hi)].sum() / (P.sum() + 1e-30) + 1e-12)


def test_round4_toggle_spring_is_a_rattle_not_a_ring():
    """round 4: the 18 ms tonal spring rang as a narrow line after B (round-3 toggle_01 / 03); hp_steep 1 makes it a
    5 ms rattle, so the energy 40 ms after B is far lower; hp_steep 0 keeps round 3's render sample for sample"""
    rec = load_recipe('toggle')
    p = dict(rec.family_params('toggle', 'on'), spring=0.08, spring_f=2600.0, tail=0.12, verb=0.0, room=0.0)
    ring = np.asarray(render(rec, dict(p, hp_steep=0), 3))
    ratt = np.asarray(render(rec, dict(p, hp_steep=1), 3))
    b = int((0.002 + p['gap'] + 0.04) * SR)                       # 40 ms after contact B (gap jitter is +-4 %)
    late = lambda x: 10 * np.log10(np.mean(_mono(x)[b:b + int(0.03 * SR)] ** 2) / np.max(_mono(x) ** 2) + 1e-20)
    assert late(ratt) < late(ring) - 10
    q = {k: v for k, v in p.items() if k != 'hp_steep'}
    assert np.array_equal(np.asarray(render(rec, q, 3)), ring)    # the default is round 3


def test_round4_tap_contact_lowpass_darkens_only_the_top():
    """round 4: the contact's 2nd-order skirt was the taps' 4-12 kHz; contact_lp_x keeps the contact (1-4 kHz) and
    removes the skirt; 0 (default) is round 3"""
    rec = load_recipe('popin')
    p = dict(rec.family_params('tap', 'table'), contact_db=0.0, contact_f=1600.0)
    a = np.asarray(render(rec, dict(p, contact_lp_x=0.0), 2))
    b = np.asarray(render(rec, dict(p, contact_lp_x=1.3), 2))
    assert _share(b, 4000, 12000) < _share(a, 4000, 12000) - 6
    assert _share(b, 1000, 4000) > _share(a, 1000, 4000) - 3
    q = {k: v for k, v in p.items() if k != 'contact_lp_x'}
    assert np.array_equal(np.asarray(render(rec, dict(q, contact_lp_x=0.0), 2)), np.asarray(render(rec, q, 2)))
