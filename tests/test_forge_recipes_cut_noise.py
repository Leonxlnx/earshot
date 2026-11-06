"""The earshot 0.4 impact / noise recipes (cuthit, thump, revcymbal, darkriser, brightswish, burstglitch): every family
member renders deterministic, clean stereo audio from parameters inside its search space, passes the model-free audit
in its kit role, lands on the character its category measured in the films (short low cut hits, a thump-whoosh with a
long low tail, a bright wide reverse cymbal that peaks at the end of its swell, a dark riser that never sweeps a tone
up, bright swishes, a glitch whose first burst is its loudest), and loads no recorded audio."""
import inspect

import numpy as np
import pytest

from earshot.forge import dsp, load_recipe, recipes, sfx

SR = 48000
MODULES = ('cuthit', 'thump', 'revcymbal', 'darkriser', 'brightswish', 'burstglitch')
# category -> the kit role its takes are audited in (a hit warns on stacked attacks and long pre-swells)
ROLE = {'cut_hit': 'hit', 'whoosh_big': 'whoosh.long', 'reverse_swell': 'swell', 'riser': 'riser',
        'swish': 'whoosh.short', 'whoosh_fast': 'whoosh.short', 'glitch': 'glitch'}
DROP = {'sub', 'kick', 'bloom'}       # cut_hit members meant for the drop role (a bloom is not a hit's pre-swell)


def members():
    for mod in MODULES:
        m = load_recipe(mod)
        for cat, fam in m.FAMILIES.items():
            for name, p in fam:
                full = m.preset_full(cat, p) if hasattr(m, 'preset_full') else {**m.DEFAULTS, **p}
                yield mod, cat, name, full


MEMBERS = list(members())
IDS = [f'{c}/{n}' for _, c, n, _ in MEMBERS]


@pytest.fixture(scope='module')
def takes():
    return {f'{c}/{n}': np.asarray(load_recipe(mod).render(dict(p), dsp.rng(21)), np.float64)
            for mod, c, n, p in MEMBERS}


def _mono(x):
    return x.mean(axis=1)


def _env_db(x, ms=5.0):
    m = _mono(x)
    h = int(SR * ms / 1000)
    k = len(m) // h
    e = 20 * np.log10(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12)
    return e - e.max()


def _centroid(x):
    F = np.abs(np.fft.rfft(_mono(x))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float((F * f).sum() / (F.sum() + 1e-30))


def _low_share(x, hz=150.0):
    F = np.abs(np.fft.rfft(_mono(x))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(F[f < hz].sum() / (F.sum() + 1e-30))


def _corr(x):
    L, R = x[:, 0], x[:, 1]
    return float((L * R).sum() / (np.sqrt((L * L).sum() * (R * R).sum()) + 1e-20))


def _attack_ms(x):
    """the catalog's definition: the first ms within 20 dB of the loudest (5 ms RMS, 10 ms smoothed) to it"""
    m = _mono(x)
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    cen = np.arange(0, len(m), SR // 1000)
    a, b = np.clip(cen - 120, 0, len(m)), np.clip(cen + 120, 0, len(m))
    sm = np.convolve((c[b] - c[a]) / np.maximum(b - a, 1), np.ones(10) / 10, mode='same')
    ip = int(np.argmax(sm))
    return float(ip - np.nonzero(sm[:ip + 1] >= sm[ip] * 1e-2)[0][0])


def test_listed_with_the_recipe_contract_and_no_recorded_audio():
    assert set(MODULES) <= set(recipes())
    for mod in MODULES:
        m = load_recipe(mod)
        for k in ('TARGET', 'ALTS', 'REFSET', 'SPACE', 'DEFAULTS', 'FAMILIES', 'render'):
            assert hasattr(m, k), (mod, k)
        src = inspect.getsource(m)
        for bad in ('soundfile', 'sf.read', 'audio.load', 'wavfile', 'librosa.load'):
            assert bad not in src, (mod, bad)
    for mod, cat, name, p in MEMBERS:                      # every member stays inside its search space
        space = load_recipe(mod).SPACE
        for key, spec in space.items():
            if key in p and isinstance(p[key], (int, float)) and not isinstance(spec, list):
                assert spec[0] - 1e-9 <= p[key] <= spec[1] + 1e-9, (cat, name, key, p[key])


def test_member_names_are_unique_bank_ids():
    assert len(IDS) == len(set(IDS))


@pytest.mark.parametrize('mod,cat,name,p', MEMBERS, ids=IDS)
def test_members_render_clean_deterministic_stereo(mod, cat, name, p, takes):
    x = takes[f'{cat}/{name}']
    assert x.ndim == 2 and x.shape[1] == 2 and np.isfinite(x).all()
    assert 0.08 < len(x) / SR < 3.0
    assert np.abs(x).max() <= 0.9 and abs(float(x[0].max())) < 0.02 and np.abs(x[-1]).max() < 0.01
    m = load_recipe(mod)
    assert np.array_equal(x, np.asarray(m.render(dict(p), dsp.rng(21))))           # one seed, one take
    assert not np.array_equal(x, np.asarray(m.render(dict(p), dsp.rng(22))))       # another seed, another take
    role = 'drop' if cat == 'cut_hit' and name in DROP else ROLE[cat]
    c = sfx.check(x.astype(np.float32), category=cat, role=role)
    assert c['ok'], (cat, name, c['summary'])
    g = sfx.rising_glide(x)                                  # never a rising synth sweep (the audit's definition)
    assert g['rise_st'] <= sfx.GLIDE_ST or g['to_s'] - g['from_s'] < sfx.GLIDE_MIN_S, (cat, name, g)


def test_cut_hits_are_short_low_and_punchy(takes):
    for name in ('sub', 'kick', 'punch'):
        x = takes[f'cut_hit/{name}']
        e = _env_db(x)
        live = (np.nonzero(e > -45)[0][-1] - np.nonzero(e > -30)[0][0]) * 0.005
        assert 0.1 <= live <= 0.7, (name, live)                     # the films: 0.11-0.59 s (sub), 0.11-0.88 (punch)
        assert 120 <= _centroid(x) <= 700, name                     # a kick-lifted sub, not a 50 Hz bloom
        assert 0.2 <= _low_share(x) <= 0.9, name
        assert _corr(x) > 0.8, name                                 # mono like the music films'
    assert _corr(takes['cut_hit/rumble']) < 0.5 and _corr(takes['cut_hit/air']) < 0.5   # the designed ones are wide


def test_thump_whoosh_hits_fast_and_rings_low(takes):
    for name in ('thump', 'thump_deep', 'thump_airy'):
        x = takes[f'whoosh_big/{name}']
        e = _env_db(x)
        ip = int(np.argmax(e))
        assert ip * 0.005 < 0.3, name                               # loud within ~0.1-0.2 s
        assert (e[ip:] > -20).sum() * 0.005 >= 0.15, name           # a long tail, not a blip
        assert _low_share(x) >= 0.4 and _centroid(x) < 800, name


def test_reverse_cymbal_swells_into_its_peak(takes):
    for name in ('cymbal', 'ring', 'dark'):
        x = takes[f'reverse_swell/{name}']
        e = _env_db(x)
        ip = int(np.argmax(e))
        assert ip * 0.005 >= 0.25 and e[0] < -12, name             # it rises into the peak
    for name in ('cymbal', 'ring'):
        x = takes[f'reverse_swell/{name}']
        assert _centroid(x) > 1500 and _corr(x) < 0.4, name         # bright and wide
    assert _centroid(takes['reverse_swell/dark']) < 1500


def test_dark_riser_peaks_at_its_end_and_stays_dark(takes):
    m = load_recipe('darkriser')
    for name, p in m.FAMILIES['riser']:
        x = takes[f'riser/{name}']
        e = _env_db(x)
        T = m.preset_full('riser', p)['length'] if hasattr(m, 'preset_full') else {**m.DEFAULTS, **p}['length']
        assert abs(int(np.argmax(e)) * 0.005 - T) < 0.05, name      # its loudest moment is the end of the build
        assert _centroid(x) < 2500, name


def test_bright_swishes_are_bright(takes):
    for name in ('bright', 'flick'):
        assert _centroid(takes[f'swish/{name}']) > 1500, name
    w = takes['whoosh_fast/whip']
    assert _centroid(w) > 1200 and _corr(w) < 0.5


@pytest.mark.parametrize('mod,cat,name,dur,apex', [('thump', 'whoosh_big', 'thump', 0.8, 0.12),
                                                   ('brightswish', 'swish', 'bright', 0.3, 0.1),
                                                   ('darkriser', 'riser', 'dark', 0.6, None),
                                                   ('burstglitch', 'glitch', 'mid', 0.3, None)])
def test_bank_knobs_aim_the_members(mod, cat, name, dur, apex):
    """as bank entries in their proposed categories the members take the category's knobs: the pass engine's dur /
    apex (thump, swishes), the recipe's own length key (riser: length, glitch: dur)"""
    from earshot.forge import bank, knobs
    m = load_recipe(mod)
    p = dict(dict(m.FAMILIES[cat])[name])
    full = m.preset_full(cat, p) if hasattr(m, 'preset_full') else {**m.DEFAULTS, **p}
    e = {'id': f'{cat}/{name}', 'type': 'forge', 'category': cat, 'member': name, 'recipe': mod, 'fn': 'render',
         'params': full, 'seed': 5, 'peak_db': -1.0, 'gain_db': 0.0}
    y, meta = bank.render(e, dur=dur, apex=apex, checks=False)
    assert abs(len(y) / SR - dur) < 0.002
    method = meta['edits']['dur']['method']
    assert method == ('shape' if cat in knobs.WHOOSH else 'param'), method
    if apex is not None:
        assert meta['edits']['apex']['ok'], meta['edits']['apex']


def test_burst_glitch_hits_first(takes):
    for name in ('burst', 'mid', 'chirp'):
        assert _attack_ms(takes[f'glitch/{name}']) < 40, name       # the films: 16-22 ms (earshot 0.3: 300-450)
    assert _centroid(takes['glitch/burst']) > 5000 and _corr(takes['glitch/burst']) < 0.5
    assert _centroid(takes['glitch/mid']) < 2500
