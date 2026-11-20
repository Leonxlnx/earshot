"""earshot.sounds: the contract of every design in the catalogue (id, descriptions, kind, a clean levelled render
that passes its audit, the axes it declares are the axes it reads, one take per seed, takes that vary, energy that
does something) and the promises of the frame around the designs (themes, voices, materials, minor keys, figure
lengths, the smallest figures and sustaining voices, loops, loudness caps, what a render reports as ignored).

The per-design renders run for the short designs; the long ones (beds, pads, notes that ring, logos) and the bank
need --slow, since rendering the whole catalogue takes minutes on one thread. Model-free."""
import re

import numpy as np
import pytest

from earshot.forge import dsp, knobs
from earshot.sounds import figure, registry, sets, theme
from earshot.synth import phrase

SR = dsp.SR
ALL = registry.ids()
BANK = {i for i in ALL if registry.get(i).fn.__module__.endswith('.bank')}
MADE = [i for i in ALL if i not in BANK]
HALF = {a: 0.5 for a in registry.AXES}
TRIES = (0, 101, 202, 303)                     # the seeds make.take tries before it gives up
# designs whose audit fails on every seed make tries, at default settings (none at the time of writing)
AUDIT_FAILS = {}
# designs whose controls test is a known failure (none at the time of writing)
CONTROL_FAILS = {}


def _fast(sid):
    d = registry.get(sid)
    return d.fn.__module__.endswith('.retro') or ((d.dur or 9.0) <= 0.4 and d.kind not in ('bed', 'pad', 'texture'))


def _params(ids, fails=None):
    """ids as pytest params: the long renders behind the slow marker, known failures as strict xfails"""
    out = []
    for i in ids:
        marks = [] if _fast(i) else [pytest.mark.slow]
        if fails and i in fails:
            marks.append(pytest.mark.xfail(strict=True, reason=fails[i]))
        out.append(pytest.param(i, id=i, marks=marks))
    return out


def _short(d):
    """the length a raw render is asked for: beds 3 s, anything longer than 0.6 s 0.6 s, the rest as designed"""
    return 3.0 if d.kind == 'bed' else (0.6 if (d.dur or 0) > 0.6 else None)


def _raw(sid, seed=1, **ax):
    """the design function's own output and its controls, before finishing"""
    d = registry.get(sid)
    c = registry.Controls(d, dur=_short(d), loop=False if d.loop else None, **{**HALF, **ax})
    return np.asarray(d.fn(c, dsp.rng(registry.seed_of(sid, seed))), np.float64), c


def _strongest(y, lo, hi, start=0.0, sec=0.5):
    m = y.mean(1) if y.ndim == 2 else y
    m = m[dsp.n_of(start):dsp.n_of(start) + dsp.n_of(sec)]
    n = len(m)
    nfft = 1 << int(np.ceil(np.log2(n * 4)))
    S = np.abs(np.fft.rfft(m * np.hanning(n), nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    b = np.nonzero((f >= lo) & (f <= hi))[0]
    return float(f[b[np.argmax(S[b])]])


def _cents(f, ref):
    return 1200 * np.log2(f / ref)


# ---- every design

@pytest.mark.parametrize('sid', ALL)
def test_entry(sid):
    d = registry.get(sid)
    assert d.id == sid and re.fullmatch(r'[a-z0-9_]+/[a-z0-9_]+', sid), sid
    assert len(d.says) >= 2 and all(isinstance(s, str) and s.strip() for s in d.says)
    assert len(set(d.says)) == len(d.says)
    assert d.kind in registry.KINDS
    assert set(d.axes) <= set(registry.AXES) and len(set(d.axes)) == len(d.axes)
    assert d.domains and set(d.domains) <= set(registry.DOMAINS)
    assert d.uses and d.dur and d.dur > 0
    if d.pitch:
        assert 20 < knobs.hz(d.pitch) < 20000
    if sid in BANK:
        assert d.axes == ()                       # the axes are not wired to bank recipes
    row = d.row()
    assert row['id'] == sid and row['axes'] == list(d.axes) and row['loop'] is d.loop


def test_catalogue_counts_and_rows():
    assert len(ALL) == len(set(ALL)) and len(MADE) > 200 and len(BANK) > 150
    assert {r['id'] for r in registry.rows(domain='ambience')} <= set(ALL)
    assert all(r['kind'] == 'bed' for r in registry.rows(kind='bed'))
    assert {r['id'] for r in registry.rows(family='coin')} == {i for i in ALL if i.startswith('coin/')}
    with pytest.raises(ValueError, match='success/rise'):
        registry.get('success/rize')


@pytest.mark.parametrize('sid', _params(MADE, AUDIT_FAILS))
def test_renders_clean_and_levelled(sid):
    d = registry.get(sid)
    kw = {'dur': 3.0} if d.kind == 'bed' else {}
    y, meta = registry.render(sid, seed=0, **kw)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    assert np.abs(y).max() <= 1.0 and meta['peak_dbfs'] <= -1.0 + 0.2
    assert meta['lufs_max'] <= registry.LOUDEST.get(d.kind, registry.LOUDEST['']) + 0.3, meta['lufs_max']
    assert meta['id'] == sid and meta['kind'] == d.kind and not meta.get('ignored')
    cut = 'not a loop' in meta.get('note', '')       # a bed asked shorter than its shortest loop is a cut of it
    assert ('loop' in meta) == d.loop and meta.get('loop', False) == (d.loop and not cut)
    assert not cut or meta['dur_s'] == kw['dur']
    assert np.abs(y[0]).max() < 1e-3 or d.loop                       # a one-shot starts from zero
    ok = meta['checks']['ok']
    for s in TRIES[1:]:                                               # make re-rolls a take that fails its audit
        if ok:
            break
        ok = registry.render(sid, seed=s, **kw)[1]['checks']['ok']
    assert ok, meta['checks']['warnings']


@pytest.mark.parametrize('sid', _params(MADE, CONTROL_FAILS))
def test_controls(sid):
    """every axis at 0.5: the axes read are exactly the declared ones; the same seed gives the same samples, another
    seed another take; energy changes the sound (and raises its level, where the design is not levelled inside)"""
    d = registry.get(sid)
    a, c = _raw(sid, 1)
    assert c.used & set(registry.AXES) == set(d.axes), (sorted(c.used & set(registry.AXES)), d.axes)
    b, _ = _raw(sid, 1)
    assert np.array_equal(a, b)
    o, _ = _raw(sid, 2)
    assert len(o) != len(a) or not np.allclose(o, a)
    if 'energy' in d.axes:
        e, _ = _raw(sid, 1, energy=-1.0)
        assert len(e) != len(a) or not np.allclose(e, a)
        if d.kind != 'bed' and not d.fn.__module__.endswith('.retro'):  # beds and chip level themselves
            pk, rms = np.abs(a).max() / np.abs(e).max(), np.sqrt(np.mean(a ** 2) / np.mean(e ** 2))
            assert max(pk, rms) > 1.0, (pk, rms)


# ---- themes, voices, materials, keys

def test_theme_moves_the_root_to_its_key_and_swaps_the_voice():
    y, meta = registry.render('notify/single', theme=theme.get('wood'))       # A5 on a glockenspiel, by design
    assert meta['voice'] == 'marimba' and meta['pitch'] == 'G5'
    f0 = knobs.hz('G5')
    assert abs(_cents(_strongest(y, 100, 5000), f0)) <= 30
    y, meta = registry.render('notify/single', theme=theme.get('glass'))      # key D: A5 goes up to D6
    assert meta['voice'] == 'glockenspiel' and meta['pitch'] == 'D6'
    assert abs(_cents(_strongest(y, 100, 5000), knobs.hz('D6'))) <= 30


def test_voice_overrides_the_theme():
    _, meta = registry.render('notify/single', theme=theme.get('wood'), voice='kalimba')
    assert meta['voice'] == 'kalimba' and 'voice' not in meta.get('ignored', [])


def test_a_material_without_a_voice_picks_its_voice():
    d = registry.get('success/rise')
    for m, v in registry.VOICE_OF.items():
        assert registry.Controls(d, material=m).voice == v
    assert registry.Controls(d, material='wood', voice='harp').voice == 'harp'
    _, meta = registry.render('success/rise', material='wood')
    assert meta['voice'] == 'marimba' and meta['material'] == 'wood' and not meta.get('ignored')


def test_a_minor_key_turns_the_major_third_minor():
    """success/rise climbs from the major third over G4 (B4) by way of D5 onto G5; in a minor scale it starts on the
    minor third, B flat"""
    out = {}
    for sc in ('major', 'minor'):
        y, _ = registry.render('success/rise', voice='marimba', theme={'scale': sc}, space=-1.0)
        third = _strongest(y, 440, 520, 0.0, 0.08)
        top = _strongest(y, 700, 870, 0.19, 0.3)
        assert abs(_cents(top, knobs.hz('G5'))) <= 20
        out[sc] = _cents(top, third)
    assert out['major'] == pytest.approx(800, abs=20) and out['minor'] == pytest.approx(900, abs=20)


def _figure_raw(sid, **req):
    d = registry.get(sid)
    return np.asarray(d.fn(registry.Controls(d, **req), dsp.rng(registry.seed_of(sid, 0))), np.float64)


@pytest.mark.parametrize('voice', [None, 'square_lead', 'warm_pad'])
def test_a_shorter_length_damps_the_figure(voice):
    """success/rise rings about 1.8 s on the marimba: asked for 0.4 s it keeps its spacing and is damped by the hand,
    a fade that ends at 0.4 s, never a cut; until the fade it is the unshortened figure sample for sample. A
    modelled voice (a chip lead, a pad) is held to the length the same way"""
    y, meta = registry.render('success/rise', voice=voice, dur=0.4)
    assert 0.37 <= meta['dur_s'] <= 0.43, meta['dur_s']
    assert 'dur' not in meta.get('ignored', [])
    pk = np.abs(y).max()
    assert np.abs(y[-dsp.n_of(0.005):]).max() < 0.03 * pk                 # faded to (near) zero
    e = np.abs(y).max(1)
    tail = [e[-dsp.n_of(s):][:dsp.n_of(0.01)].max() for s in (0.05, 0.03, 0.01)]
    assert tail[0] > tail[1] > tail[2]                                    # falling over its last 50 ms: a fade
    a, b = _figure_raw('success/rise', voice=voice), _figure_raw('success/rise', voice=voice, dur=0.4)
    assert len(a) > len(b) and np.array_equal(a[:dsp.n_of(0.1)], b[:dsp.n_of(0.1)])


def test_a_figure_hurried_to_fit_and_its_shortest_length():
    """the last note starts at least LAST_ROOM s before the end: shorter, the notes come faster, down to HAND s
    apart; success/rise (two gaps of 90 ms) is then 2 x 70 ms + 120 ms long at the least"""
    assert figure.squeeze([0.09, 0.09], 0.16) == pytest.approx([0.08, 0.08], abs=1e-6)
    assert figure.squeeze([0.09, 0.09], 0.05) == pytest.approx([figure.HAND] * 2)
    assert figure.squeeze([0.07, 0.012], 0.0) == pytest.approx([figure.HAND, 0.012])      # a chord stays a chord
    assert figure.squeeze([0.09, 0.09], 0.5) == [0.09, 0.09]
    _, meta = registry.render('success/rise', dur=0.3, check=False)
    assert 0.27 <= meta['dur_s'] <= 0.33
    _, meta = registry.render('success/rise', dur=0.1, check=False)
    assert meta['dur_s'] == pytest.approx(2 * figure.HAND + figure.LAST_ROOM, abs=0.03)


def test_no_length_or_a_longer_one_plays_the_figure_as_it_rings():
    """a struck note is not held: a length longer than the figure rings gives the same samples as none"""
    a, _ = registry.render('success/rise', voice='marimba')
    b, meta = registry.render('success/rise', voice='marimba', dur=5.0)
    assert np.array_equal(a, b) and 'dur' in meta.get('ignored', [])


@pytest.mark.parametrize('sid', ['tap/tone', 'hover/tone', 'menu/move'])
def test_the_smallest_figures_avoid_sustaining_voices(sid):
    for name in ('glass', 'metal', 'electric'):
        assert registry.render(sid, theme=theme.get(name), check=False)[1]['voice'] == 'marimba', name
    assert registry.render(sid, theme=theme.get('tine'), check=False)[1]['voice'] == 'kalimba'
    assert registry.render('error/double', theme=theme.get('metal'), check=False)[1]['voice'] == 'marimba'


def _notes_of(monkeypatch, sid, **req):
    """the notes figure.play hands to phrase.play: [(t, hz, vel, dur)]"""
    seen = []
    play = phrase.play

    def spy(notes, voice, r, **kw):
        seen.append(list(notes))
        return play(notes, voice, r, **kw)
    monkeypatch.setattr(phrase, 'play', spy)
    registry.render(sid, check=False, **req)
    return seen[0]


@pytest.mark.parametrize('voice', ['vibraphone', 'glockenspiel',
                                   'fm_bell', 'warm_pad'])
def test_inner_notes_of_a_sustaining_voice_keep_the_floor(monkeypatch, voice):
    notes = _notes_of(monkeypatch, 'toggle/up', voice=voice)            # gate 0.04 s, gap 0.045 s
    assert len(notes) == 2
    assert notes[0][3] >= figure.SUSTAIN_FLOOR - 1e-9
    assert notes[-1][3] >= 1.6 * figure.SUSTAIN_FLOOR - 1e-9


@pytest.mark.parametrize('voice', ['xylophone', 'pizzicato'])
def test_a_short_voice_keeps_its_short_notes(monkeypatch, voice):
    notes = _notes_of(monkeypatch, 'toggle/up', voice=voice)
    assert notes[0][3] < figure.SUSTAIN_FLOOR


# ---- loops, levels, what is ignored

@pytest.mark.parametrize('sid,dur', [('engine/chip', 2.0), ('clock/ticktock', 2.0), ('rain/tent', 8.0)])
def test_loops_report_the_loop_and_keep_their_length(sid, dur):
    assert registry.get(sid).loop
    y, meta = registry.render(sid, dur=dur, check=False)
    assert meta['loop'] is True and abs(len(y) - dur * SR) <= 1


@pytest.mark.parametrize('sid', ['engine/chip', 'clock/ticktock', 'rain/tent'])
def test_a_loop_once_has_fades(sid):
    y, meta = registry.render(sid, dur=3.0, loop=False)
    assert meta['loop'] is False and meta['checks']['ok'], meta['checks']['warnings']
    assert np.abs(y[0]).max() == 0.0 and np.abs(y[-1]).max() < 1e-4
    assert len(y) <= 3.0 * SR + 1


def test_loudness_caps():
    for sid, kw in (('rumble/quake', {}), ('engine/chip', {'dur': 2.0}), ('explosion/chip', {}),
                    ('laser/chip', {})):
        d = registry.get(sid)
        _, meta = registry.render(sid, check=False, **kw)
        cap = registry.LOUDEST.get(d.kind, registry.LOUDEST[''])
        assert meta['lufs_max'] <= cap + 0.3, (sid, meta['lufs_max'], cap)
        assert meta['lufs_max'] >= cap - 0.5 or meta['peak_dbfs'] >= -1.2   # held at the cap, or at full peak
    _, meta = registry.render('laser/chip', lufs=-30.0, check=False)
    assert meta['lufs_max'] == pytest.approx(-30.0, abs=0.5)


def test_ignored_names_what_was_not_read():
    def ign(sid, **req):
        return registry.render(sid, check=False, **req)[1].get('ignored', [])
    assert ign('coin/chip', size=0.5, bright=0.3) == ['size']
    assert ign('tick/wood', material='glass') == ['material']
    assert ign('zap/laser', voice='marimba') == ['voice']
    assert ign('whoosh_fast/crisp', bright=0.4) == ['bright']               # bank sounds read no axes
    # a voice or a material that is the design's own was honoured by choosing the design
    assert ign('tick/wood', material='wood') == []
    assert ign('coin/chip', material='retro') == []
    assert ign('note/kalimba', voice='kalimba') == []
    assert ign('success/rise', bright=0.3, voice='harp', pitch='E5') == []


def test_unknown_controls_are_refused():
    with pytest.raises(ValueError, match='bright'):
        registry.render('tap/tone', loudness=0.3)


# ---- themes and sets as data

def test_theme_presets_and_changes():
    for name, t in theme.THEMES.items():
        assert t['key'] in theme.KEYS and t['scale'] in registry.SCALES and t['voice']
        g = theme.get(name)
        assert g['name'] == name and g['voice'] == t['voice']
    t = theme.get({'base': 'wood', 'key': 'D', 'axes': {'hard': 0.2}})
    assert t['voice'] == 'marimba' and t['key'] == 'D' and t['axes'] == {'bright': -0.1, 'ring': -0.2, 'hard': 0.2}
    assert theme.get(None) is None
    assert theme.get({'key': 'E'}) == {'key': 'E', 'axes': {}}               # a dict without a base starts empty
    for bad in ('opera', {'colour': 'red'}, {'key': 'H'}):
        with pytest.raises(ValueError):
            theme.get(bad)


def test_an_unknown_base_theme_is_a_value_error():
    with pytest.raises(ValueError, match='glass'):
        theme.get({'base': 'opera'})


def test_ready_made_sets_name_designs_of_the_catalogue():
    have = set(ALL)
    for name, s in sets.SETS.items():
        files = [f for f, _ in s['sounds']]
        assert len(files) == len(set(files)), name
        assert [i for _, i in s['sounds'] if i not in have] == [], name
        assert s['theme'] is None or s['theme'] in theme.THEMES


def test_sets_for_films_and_interfaces_play_no_note():
    """a set that is not for a game (and no theme named) plays real things: no design with a note in it (a figure
    on an instrument, a bell, a synthesised tone) answers a motion or interface job unless music is asked for"""
    for name, s in sets.SETS.items():
        if name in ('game', 'arcade'):
            continue
        noted = [i for _, i in s['sounds'] if registry.get(i).pitch or 'tonal' in registry.get(i).tags]
        assert noted == [], name
