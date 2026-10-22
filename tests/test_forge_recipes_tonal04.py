"""earshot 0.4 tonal / UI recipes (ui_pop, knock, mallet + runs, drip, shutter; notify's decay knob) and the tonal knobs
(decay / onset / bright / width): families render clean and deterministic, land in the films' measured ranges they
were built for (the earshot 0.4 film catalog, section 1), pass the model-free audit on new takes, the knobs land their
targets and refuse what a sound cannot take, and the old notify members still render sample for sample. No models."""
import numpy as np
import pytest

from earshot.forge import dsp, knobs, sfx
from earshot.forge.search import _decode, _encode, render

SR = 48000
NEW = ('ui_pop', 'knock', 'mallet', 'drip', 'shutter')


def _mod(name):
    import importlib
    return importlib.import_module(f'earshot.forge.recipes.{name}')


def _members(name):
    m = _mod(name)
    return [(m, p) for fam in m.FAMILY.values() for p in fam]


def _m(x):
    from earshot.forge.recipes._tonal import measure
    return measure(x)


@pytest.mark.parametrize('name', NEW)
def test_new_families_render_clean_and_deterministic(name):
    mod = _mod(name)
    fams = mod.FAMILY
    assert fams and all(len(v) == len(mod.FAMILY_NAMES[k]) for k, v in fams.items())
    for i, (m, p) in enumerate(_members(name)):
        assert set(p) <= set(m.SPACE) | {'style'}, set(p) - set(m.SPACE)
        x = render(m, p, i + 1)
        assert x is not None and np.isfinite(x).all() and x.ndim == 2
        assert np.array_equal(x, render(m, p, i + 1))                    # same seed, same samples
        y = render(m, p, i + 11)
        assert not np.array_equal(x[:min(len(x), len(y))], y[:min(len(x), len(y))])   # a new seed is a new take
        mono = x.mean(1)
        assert abs(mono[0]) < 1e-3 and abs(mono[-1]) < 1e-3, (name, i)   # starts and ends on zero
        assert 0.03 < len(x) / SR < 1.2, (name, i)


@pytest.mark.parametrize('name', NEW)
def test_new_takes_pass_the_audit(name):
    """every member's takes (two new seeds) pass earshot's model-free audit (what kit check runs), the drip's rising
    pitch included: it completes within the audit's 50 ms window"""
    cat = name
    for i, (m, p) in enumerate(_members(name)):
        for s in (101 + i, 202 + i):
            c = sfx.check(render(m, p, s), category=cat)
            assert c['ok'], (name, i, s, c['summary'])


def test_ui_pop_family_sits_in_the_films_pop():
    """the palette's typical member lands inside the films' pop IQRs on onset, decay, register, tonal share and width"""
    mod = _mod('ui_pop')
    p = mod.FAMILY['ui_pop'][0]
    m = _m(render(mod, p, 3))
    assert 18 <= m['attack_ms'] <= 62 and 57 <= m['t20_ms'] <= 129, m
    assert 475 <= m['centroid_hz'] <= 830 and m['lo150'] <= 0.14 and m['hi4k'] <= 0.03, m
    assert 0.45 <= m['tonal'] <= 0.68 and m['corr'] >= 0.9, m
    for i, q in enumerate(mod.FAMILY['ui_pop']):                          # falling or flat: never a rising whip
        assert sfx.rising_glide(render(mod, q, 7))['rise_st'] <= 2.0, i
    soft = [_m(render(mod, p, 5)) for p in mod.FAMILY['ui_pop'][:2]]       # the typical pops sit near mono
    assert all(m['corr'] >= 0.9 and m['glide_st'] <= 1.0 for m in soft), soft


def test_knock_is_mid_band_and_short():
    mod = _mod('knock')
    for i, p in enumerate(mod.FAMILY['knock']):
        m = _m(render(mod, p, i + 2))
        assert 300 <= m['centroid_hz'] <= 3500 and m['hi4k'] < 0.2, (i, m)
        assert m['t20_ms'] <= 140 and m['dur_s'] <= 0.45, (i, m)
    wood = _m(render(mod, mod.FAMILY['knock'][0], 3))                     # the films' wooden knock: inside the IQRs
    assert 10 <= wood['attack_ms'] <= 28 and 12 <= wood['t20_ms'] <= 131, wood
    assert 422 <= wood['centroid_hz'] <= 1327 and 0.42 <= wood['tonal'] <= 0.58, wood


def test_knock_bounce_adds_a_second_contact():
    mod = _mod('knock')
    p = dict(mod.DEFAULTS, jit=0.0, room=0.0, decay=0.006, body=-60.0)    # a short strike: quiet 25 ms later
    one = render(mod, p, 1).mean(1)
    two = render(mod, dict(p, bounce=-6.0, bounce_ms=25.0), 1).mean(1)
    k, w = int(0.025 * SR), int(0.004 * SR)
    assert np.abs(two[k:k + w]).max() > 4 * np.abs(one[k:k + w]).max()       # a contact 25 ms after the strike
    assert np.array_equal(render(mod, p, 1), render(mod, dict(p, bounce=-60.0), 1))   # off: the same samples


def test_mallet_octave_double_rounds_the_note_and_runs_render_clean():
    """the bell note doubled an octave under sits in the films' bell-note register and tonal share (a bare bar is
    purer); every chime run renders deterministic and passes the audit"""
    mod = _mod('mallet')
    glock = mod.FAMILY['mallet'][0]
    ms = [_m(render(mod, glock, s)) for s in (3, 4, 5, 6)]
    bare = [_m(render(mod, dict(glock, oct=-60.0), s)) for s in (3, 4, 5, 6)]
    assert all(550 <= m['centroid_hz'] <= 1505 and m['dur_s'] <= 0.5 for m in ms), ms
    ton, ton_bare = np.mean([m['tonal'] for m in ms]), np.mean([m['tonal'] for m in bare])
    assert ton <= 0.72 and ton_bare >= ton + 0.08, (ton, ton_bare)    # the two lines an octave apart: less pure
    assert len(mod.RUNS['chime_run']) == len(mod.RUN_NAMES['chime_run'])
    for i, p in enumerate(mod.RUNS['chime_run']):
        x = np.asarray(mod.render_run(dict(p), dsp.rng(i + 3)), np.float64)
        assert np.array_equal(x, np.asarray(mod.render_run(dict(p), dsp.rng(i + 3)), np.float64))
        c = sfx.check(dsp.normalize(x, -1.0), category='mallet')
        assert c['ok'], (i, c['summary'])


def test_mallet_notes_are_gated_and_wide_and_runs_step():
    mod = _mod('mallet')
    for i, p in enumerate(mod.FAMILY['mallet']):
        x = render(mod, p, i + 2)
        m = _m(x)
        assert m['dur_s'] <= 0.6, (i, m)                                  # gated: no 2 s ring
        if p['style'] == 'note' and p['width'] >= 0.4:
            assert m['corr'] <= 0.9, (i, m)                               # the films' wide bell notes
    run = dict(mod.RUN_DEFAULTS, f0=1046.5, notes=4, step=0.11, pattern='up', swing=0.0, jit=0.0)
    x = np.asarray(mod.render_run(run, dsp.rng(1)), np.float64)
    fs = mod.run_freqs(run)
    assert len(fs) == run['notes'] and all(b > a for a, b in zip(fs, fs[1:]))   # 'up': rising scale notes
    nf, hop = 1024, 96                                                   # each note's own band: when it starts
    m = np.pad(x.mean(1), (nf, nf))
    fr = np.lib.stride_tricks.sliding_window_view(m, nf)[::hop] * np.hanning(nf)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    F = np.fft.rfftfreq(nf, 1 / SR)
    starts = []
    for f in fs:
        e = P[:, (F > f / 2 ** (1 / 24)) & (F < f * 2 ** (1 / 24))].sum(1)
        starts.append(int(np.argmax(e >= e.max() * 10 ** -1.2)))
    gaps = np.diff(starts) * hop / SR
    assert np.allclose(gaps, run['step'], atol=0.008), gaps                # discrete notes `step` apart
    L, R = x[:, 0], x[:, 1]
    k = int(run['step'] * SR)
    bal = [np.sum(L[i * k:(i + 1) * k] ** 2) / (np.sum(R[i * k:(i + 1) * k] ** 2) + 1e-12) for i in range(2)]
    assert (bal[0] - 1) * (bal[1] - 1) < 0                                # panned note by note (L, then R)


def test_drip_rises_and_says_so():
    """every drop rises; all but the bloop inside the audit's 50 ms window, the bloop's slower climb is one the drip
    allowance accepts"""
    mod = _mod('drip')
    assert mod.RISES is True
    names = mod.FAMILY_NAMES['drip']
    for i, p in enumerate(mod.FAMILY['drip']):
        assert p['rise'] > 0, i
        g = sfx.rising_glide(render(mod, p, i + 4))
        inside = g['rise_st'] <= sfx.GLIDE_ST or (g['to_s'] - g['from_s']) < sfx.GLIDE_MIN_S
        assert inside or (names[i] == 'bloop' and mod.allows(g)), (names[i], g)


def test_drip_allowance_accepts_a_bubble_not_a_whip():
    """a slow bubble (a big drop's 'bloop') trips the plain audit's rising-glide check; for the drip category
    sfx.check (and drip.check) turn that into a note, and keep flagging a climb no bubble makes"""
    mod = _mod('drip')
    slow = dict(mod.DEFAULTS, f0=420.0, rise=10.0, rtau=0.025, jit=0.0)
    x = render(mod, slow, 2)
    plain = sfx.check(x)                                                    # no category: a rising tone is a warning
    assert 'rising_glide' in [w['code'] for w in plain['warnings']], plain['summary']
    for c in (sfx.check(x, category='drip'), mod.check(x)):
        assert c['ok'] and 'rising_glide' in [n['code'] for n in c['notes']], c['summary']
        assert 'drip allowance' in c['summary']
    assert not sfx.check(x, category='pop')['ok']                           # the allowance is the drip's only
    assert not mod.allows({'rise_st': 20.0, 'from_s': 0.01, 'to_s': 0.2})          # too large for a bubble
    assert not mod.allows({'rise_st': 8.0, 'from_s': 0.2, 'to_s': 0.35})           # a late sweep
    assert not mod.allows({'rise_st': 8.0, 'from_s': 0.01, 'to_s': 0.5})           # too slow


def test_shutter_has_two_stages():
    """stage A (the release) starts the file, stage B (the landing) jumps in `gap` later (+-10 %: each take moves
    the mechanism by ~3 %)"""
    mod = _mod('shutter')
    for i, p in enumerate(mod.FAMILY['shutter']):
        x = render(mod, p, i + 2).mean(1)
        h = SR // 1000
        e = 20 * np.log10(np.sqrt((x[:len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
        assert e[:3].max() > e.max() - 40, i                              # stage A at the start
        g = p['gap'] * 1000
        lo, hi = int(0.88 * g), int(np.ceil(1.12 * g)) + 2
        jump = np.max(e[lo:hi] - np.maximum.accumulate(e)[lo - 4:hi - 4])
        assert jump > 6, (i, p['gap'], jump)                              # stage B: a new contact, not a decay


@pytest.mark.parametrize('name', NEW)
def test_family_space_pins_the_character(name):
    mod = _mod(name)
    p = next(iter(mod.FAMILY.values()))[0]
    sp = mod.family_space(p)
    pinned = [k for k, v in sp.items() if isinstance(v, list) and len(v) == 1]
    free = [k for k in sp if k not in pinned]
    assert free and pinned and set(free) <= set(mod.FINE)
    q = _decode(sp, np.clip(_encode(sp, p) + 0.3, 0, 1))
    assert all(q[k] == p[k] for k in pinned)


# ---------------------------------------------------------------------------------------------------- notify decay
def test_notify_ring_and_soft_default_to_the_old_render():
    notify = _mod('notify')
    for p in (notify.FAMILY['notify'][2], notify.FAMILY['pro_notify'][0], notify.FAMILY_R6['pro_notify'][1]):
        old = {k: v for k, v in p.items() if k not in ('ring', 'soft', 'bell_len')}
        assert np.array_equal(render(notify, old, 3), render(notify, dict(old, ring=1.0, soft=0.0, bell_len=0.0), 3))


def test_notify_ring_shortens_the_ring_down_not_the_file():
    notify = _mod('notify')
    p = dict(notify.FAMILY['notify'][1], bell_len=0.8)
    long_, short = _m(render(notify, p, 2)), _m(render(notify, dict(p, ring=0.3), 2))
    assert short['t20_ms'] < 0.6 * long_['t20_ms'], (long_, short)
    c = sfx.check(render(notify, dict(p, ring=0.3), 2), category='notify')
    assert 'truncated' not in [w['code'] for w in c['warnings']], c['summary']   # it rings out inside the file
    soft = _m(render(notify, dict(p, ring=0.3, soft=0.03), 2))
    assert soft['attack_ms'] > short['attack_ms'] + 15, (short, soft)


def test_mid_ding_member_hits_the_films_ding():
    notify = _mod('notify')
    for i, p in enumerate(notify.FAMILY_MID['mid_ding']):
        x = render(notify, p, i + 5)
        m = _m(x)
        assert 0.2 <= m['dur_s'] <= 0.55 and 25 <= m['t20_ms'] <= 230, (i, m)
        assert 437 <= m['centroid_hz'] <= 1014, (i, m)
        assert sfx.check(x, category='notify')['ok'], i


# ---------------------------------------------------------------------------------------------------- tonal knobs
def _entry(cat, recipe, params, seed=1):
    return {'id': f'{cat}/test', 'type': 'forge', 'category': cat, 'recipe': recipe, 'fn': 'render',
            'params': dict(params), 'seed': seed, 'peak_db': -1.0, 'gain_db': 0.0}


def test_decay_knob_lands_t20_on_a_bank_style_notify_entry():
    from earshot.forge import bank
    notify = _mod('notify')
    p = {k: v for k, v in notify.FAMILY['notify'][1].items() if k not in ('ring', 'soft', 'bell_len')}
    e = _entry('notify', 'notify', dict(p, bell_len=0.8))
    x, ed = knobs.apply(e, bank.renderer(e), decay=120.0)
    d = ed['decay']
    assert d["param"] == "ring" and d["value"] < 1.0 and d["renders"] <= 1 + knobs.TONAL_FIX, d
    assert d['ok'] and abs(_m(x)['t20_ms'] - 120.0) <= 12.0, d


def test_onset_and_width_knobs_land_their_targets():
    from earshot.forge import bank
    mod = _mod('ui_pop')
    e = _entry('ui_pop', 'ui_pop', mod.FAMILY['ui_pop'][0])
    x, ed = knobs.apply(e, bank.renderer(e), onset=45.0, width=0.6)
    assert ed['onset']['ok'] and ed['width']['ok'], ed
    m = _m(x)
    assert abs(m['attack_ms'] - 45.0) <= 7 and abs(m['corr'] - 0.6) <= 0.05, m


def test_tonal_knobs_refuse_what_a_sound_cannot_take():
    from earshot.forge import bank
    with pytest.raises(knobs.KnobError, match='no decay knob'):
        knobs.validate(bank.get('whoosh_fast/crisp'), decay=100.0)
    with pytest.raises(knobs.KnobError, match='decay must be'):
        knobs.validate(_entry('ui_pop', 'ui_pop', {}), decay=0.5)
    fit = next(i for i in bank.ids() if i.startswith('fit/'))
    with pytest.raises(knobs.KnobError, match='fitted recording'):
        knobs.validate(bank.get(fit), width=0.5)
    d = knobs.describe(bank.get('notify/m0'))
    assert d['tonal']['decay'] == 'ring' and d['tonal']['onset'] == 'soft' and d['dur'] == 'param'
    with pytest.raises(knobs.KnobError, match='no bright knob'):             # a knock's brightness is its material
        knobs.validate(_entry('knock', 'knock', {}), bright=800.0)
    notify, mallet = _mod('notify'), _mod('mallet')
    with pytest.raises(knobs.KnobError, match='chime style has no bright'):
        knobs.validate(_entry('notify', 'notify', notify.FAMILY_MID['mid_ding'][0]), bright=900.0)
    with pytest.raises(knobs.KnobError, match='pluck style has no bright'):
        knobs.validate(_entry('mallet', 'mallet', mallet.FAMILY['mallet'][1]), bright=500.0)


def test_bright_knob_moves_a_shutter_box():
    from earshot.forge import bank
    shutter = _mod('shutter')
    e = _entry('shutter', 'shutter', shutter.FAMILY['shutter'][0])
    x, ed = knobs.apply(e, bank.renderer(e), bright=1500.0)
    assert ed['bright']['param'] == 'b_hz' and ed['bright']['ok'], ed
    assert abs(_m(x)['centroid_hz'] - 1500.0) <= 120, ed


def test_impact_members_take_their_knobs():
    """the impact agent's requests: a reverse cymbal's dur moves its own `length`; a cut hit's sub lands on a note
    (sub_f1, its start sub_f0 keeping the glide ratio, not measured) and its dur stays a trim"""
    from earshot.forge.recipes import cuthit, revcymbal
    rev = _entry('reverse_swell', 'revcymbal', revcymbal.DEFAULTS)
    assert knobs.describe(rev)['dur'] == 'param'
    cut = _entry('cut_hit', 'cuthit', cuthit.DEFAULTS)
    d = knobs.describe(cut)
    assert d['dur'] == 'trim' and d['pitch']['param'] == 'sub_f1', d
    upd, _, _ = knobs._pitch(cut, 'A1')
    assert abs(upd['sub_f1'] - 55.0) < 0.01, upd
    assert abs(upd['sub_f0'] / upd['sub_f1'] - cuthit.DEFAULTS['sub_f0'] / cuthit.DEFAULTS['sub_f1']) < 1e-6, upd
    with pytest.raises(knobs.KnobError, match='outside'):
        knobs.validate(cut, pitch='C5')
