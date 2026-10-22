"""earshot sfx (the model-free audit and the CLI) and earshot kit (presets, render + lock + cache, check, audition),
plus the MCP tool functions of the sound bank. Synthetic signals and short bank sounds only; no models."""
import json
import os
import re

import numpy as np
import pytest
import soundfile as sf

from earshot import cli, mcp_server
from earshot.forge import bank, kit, sfx

SR = 48000
rng = np.random.RandomState(7)


def _sweep(f0, f1, dur=0.35):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    return (0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(len(t)) ** 0.3).astype(np.float32)


def _burst(t0, tau=0.06, dur=0.8):
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    on = t >= t0
    y[on] = rng.randn(on.sum()) * np.exp(-(t[on] - t0) / tau)
    y[:int(t0 * SR) + 1] = 0.0
    return y / np.abs(y).max()                     # peak 1: callers scale it under full scale


def _codes(c, key='warnings'):
    return {w['code'] for w in c[key]}


# ----------------------------------------------------------------------------------------------------- the audit
def test_rising_glide_is_flagged_and_falling_or_steady_is_not():
    up = sfx.check(_sweep(400, 1200))
    assert 'rising_glide' in _codes(up) and not up['ok'] and up['glide']['rise_st'] > 15
    down = sfx.check(_sweep(1200, 400))
    assert 'rising_glide' not in _codes(down) and down['glide']['rise_st'] < 1
    small = sfx.check(_sweep(500, 500 * 2 ** (3 / 12)))           # 3 semitones: a natural inflection
    assert 'rising_glide' not in _codes(small)
    t = np.arange(SR) / SR
    bell = sfx.check((0.5 * np.sin(2 * np.pi * 880 * t) * np.exp(-t / 0.2)).astype(np.float32))
    assert bell['glide']['rise_st'] < 1 and 'rising_glide' not in _codes(bell)
    noise = sfx.check((_burst(0.01, 0.1) * 0.5).astype(np.float32))
    assert noise['glide']['tonal_share'] < 0.2 and 'rising_glide' not in _codes(noise)


def test_stacked_hit_and_pre_swell_are_warnings_for_a_hit_role():
    double = (0.4 * (_burst(0.01) + _burst(0.13))).astype(np.float32)
    assert 'stacked' in _codes(sfx.check(double, role='hit'))
    assert 'stacked' not in _codes(sfx.check(double)) | _codes(sfx.check(double), 'notes')   # no context: silent
    assert 'stacked' in _codes(sfx.check(double, category='impact'), 'notes')                # a hit category: a note
    one = (0.4 * _burst(0.01)).astype(np.float32)
    c = sfx.check(one, role='hit')
    assert c['layers']['count'] == 1 and c['ok'], c['summary']
    t = np.arange(int(0.8 * SR)) / SR
    lead = np.where(t < 0.25, (t / 0.25) ** 3 * 0.3 * rng.randn(len(t)), 0.0) + _burst(0.25)
    assert 'pre_swell' in _codes(sfx.check((0.4 * lead / np.abs(lead).max()).astype(np.float32), role='hit'))


def test_artifacts_are_flagged():
    t = np.arange(int(0.3 * SR)) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.05)
    assert sfx.check(tone.astype(np.float32))['ok']
    spike = tone.copy()
    spike[int(0.15 * SR)] += 0.3                                   # a one-sample spike inside the decay
    assert 'clicks' in _codes(sfx.check(spike.astype(np.float32)))
    clip = np.clip(3 * np.sin(2 * np.pi * 440 * t), -1, 1) * np.hanning(len(t))
    clip = np.clip(clip * 1.5, -0.9, 0.9)
    assert 'clipping' in _codes(sfx.check(clip.astype(np.float32)))
    assert 'edge' in _codes(sfx.check((0.5 * np.ones(4800)).astype(np.float32)))
    cut = (0.5 * rng.randn(int(0.5 * SR)) * np.exp(-np.arange(int(0.5 * SR)) / (0.3 * SR)))[:int(0.15 * SR)]
    cut[-24:] *= np.linspace(1, 0, 24)
    assert 'truncated' in _codes(sfx.check(cut.astype(np.float32)))
    assert 'truncated' in _codes(sfx.check(cut.astype(np.float32), category='riser'), 'notes')   # the design


def test_silence_is_flagged(wav, capsys):
    """review: sfx.check(np.zeros(4800)) and a kit take faded to nothing by a tiny max_s were called 'clean'"""
    z = sfx.check(np.zeros(4800, np.float32))
    assert not z['ok'] and 'silent' in _codes(z) and 'digital silence' in z['summary']
    t = np.arange(int(0.2 * SR)) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.05)
    assert 'silent' in _codes(sfx.check((tone * 1e-4).astype(np.float32)))              # peak -86 dBFS
    blip = np.zeros(int(0.2 * SR))
    blip[1000:1060] = np.hanning(60) * 0.5                                             # ~1 ms of sound
    assert 'silent' in _codes(sfx.check(blip.astype(np.float32)))
    y, _ = bank.render('pro_hover_tick/air', checks=False)                             # the shortest bank sound
    assert 'silent' not in _codes(sfx.check(y)) and sfx.live_ms(y) > 1.5 * sfx.SILENT_MS
    assert cli.main(['sfx', 'check', wav(np.zeros(4800))]) == 1
    assert 'digital silence' in capsys.readouterr().out


# --------------------------------------------------------------------------------------------------------- kits
def test_presets_are_valid_restrained_and_explained():
    assert kit.presets() == ['calm', 'crisp', 'launch', 'minimal']
    for name in kit.presets():
        k = kit.preset(name)
        assert kit.problems(k) == [], name
        assert set(k['roles']) == set(kit.ROLES), (name, set(kit.ROLES) ^ set(k['roles']))
        for r in kit.RESTRAINED:
            if name == 'launch' and r in ('riser', 'swell'):     # launch films use one, at a section change
                assert 'at most one' in k['roles'][r]['note'], r
                continue
            assert k['roles'][r] is None and k['off'][r], (name, r)
        for role, spec in k['roles'].items():
            if spec is None:
                assert role in k['off'], (name, role)
                continue
            assert len(spec['note']) > 20, (name, role)
            cat = spec['sound'].split('/')[0]
            assert spec['sound'] not in kit.NEVER and cat not in kit.NEVER_CATEGORIES, (name, role)
        hits = [s['sound'] for r, s in k['roles'].items() if s and r == 'hit']
        assert len(hits) == 1
        files = [t['file'] for t in kit.plan(k)]
        assert len(files) == len({f.lower() for f in files})


def _midi(p):
    from earshot.forge import knobs
    return knobs.midi(p)


@pytest.mark.parametrize('name', ['calm', 'crisp', 'minimal', 'launch'])
def test_presets_have_the_rungs_a_progress_motif_needs(name):
    """review (cinetic 5.5b): with 5 bell takes a climbing progress motif ran out of rungs; the bell now spans two
    octaves of one pentatonic (10 notes), and a tonal text role offers the 5 pentatonic degrees of one octave so a
    chain can step through the chord's tones. An untuned text role (crisp, minimal: a soft tap, untuned on purpose)
    rotates its takes instead"""
    k = kit.preset(name)
    bell = [_midi(p) for p in k['roles']['bell']['pitches']]
    assert len(bell) == 10 and bell == sorted(bell) and len(set(bell)) == 10
    pcs = {m % 12 for m in bell}
    assert len(pcs) == 5 and max(bell) - min(bell) >= 21                    # two octaves of a pentatonic
    steps = sorted({(b - a) for a, b in zip(sorted(pcs), sorted(pcs)[1:] + [min(pcs) + 12])})
    assert steps == [2, 3]                                                  # whole steps and minor thirds only
    text = k['roles']['text']
    if text.get('pitches'):
        tm = [_midi(p) for p in text['pitches']]
        assert len({m % 12 for m in tm}) >= 5 and {m % 12 for m in tm} <= pcs and max(tm) - min(tm) < 12
    else:
        assert text['takes'] >= 3 and 'untuned' in text['note']


LISTENING_WORDS = re.compile(r'listener|blind test|picked out|taken for recordings|heard as|to the ear', re.I)


def test_presets_reference_bank_sounds_and_state_measured_facts():
    """every preset sound is in the bank (forge, pluck or fit); notes describe the sound and what was measured,
    never a listening test as a rule"""
    ids = set(bank.ids())
    fits = 0
    for name in kit.presets():
        k = kit.preset(name)
        notes = [k.get('note', '')] + list((k.get('off') or {}).values())
        for role, spec in k['roles'].items():
            if not spec:
                continue
            assert spec['sound'] in ids, (name, role, spec['sound'])
            fits += spec['sound'].startswith('fit/')
            notes.append(spec['note'])
        bad = [n for n in notes if LISTENING_WORDS.search(n)]
        assert not bad, (name, bad)
    assert fits >= 6                      # the presets use the shipped fits where the comparison favoured them


def _tiny(**edit):
    k = {'kit': 1, 'name': 'tiny', 'sr': 48000,
         'roles': {'tick': {'sound': 'tap/finger', 'takes': 2, 'level_db': -6.0},
                   'bell': {'sound': 'pro_notify/ding_1822', 'pitches': ['A6', 'C7']},
                   'whoosh.short': {'sound': 'whoosh_fast/tight', 'edits': {'dur': 0.5, 'apex': 0.2}},
                   'riser': None}}
    for role, spec in edit.items():
        k['roles'][role] = spec
    return k


def test_kit_render_lock_and_cache(tmp_path):
    out = tmp_path / 'kit'
    lock = kit.render(_tiny(), str(out))
    assert lock['stats']['rendered'] == 5 and lock['stats']['cached'] == 0
    disk = json.loads((out / 'kit.lock.json').read_text(encoding='utf-8'))
    assert disk['kit'] == 1 and disk['bank'] == bank.VERSION and disk['sr'] == 48000 and disk['roles']['riser'] is None
    tick = disk['roles']['tick']
    assert tick['sound'] == 'tap/finger' and tick['align'] == 'onset' and tick['level_db'] == -6.0
    assert [t['file'] for t in tick['takes']] == ['tick_1.wav', 'tick_2.wav']
    e = bank.get('tap/finger')
    assert [t['seed'] for t in tick['takes']][0] in (e['seed'], e['seed'] + 1000)
    for t in tick['takes']:
        for k in ('file', 'seed', 'pitch', 'onset_s', 'peak_s', 'end_s', 'dur_s', 'peak_db', 'lufs_m_max', 'sha1'):
            assert k in t
        y, fs = sf.read(str(out / t['file']))
        assert fs == 48000 and abs(len(y) / fs - t['dur_s']) < 1e-3
    assert [t['file'] for t in disk['roles']['bell']['takes']] == ['bell_A6.wav', 'bell_C7.wav']
    w = disk['roles']['whoosh.short']
    assert w['align'] == 'peak' and abs(w['takes'][0]['peak_s'] - 0.2) <= 0.005 and w['takes'][0]['dur_s'] == 0.5
    again = kit.render(_tiny(), str(out))
    assert again['stats']['rendered'] == 0 and again['stats']['cached'] == 5
    moved = kit.render(_tiny(tick={'sound': 'tap/finger', 'takes': 2, 'edits': {'dur': 0.1}}), str(out))
    assert moved['stats']['rendered'] == 2 and moved['stats']['cached'] == 3
    fewer = kit.render(_tiny(bell=None), str(out))
    assert fewer['stats']['removed'] == 2 and not (out / 'bell_A6.wav').exists()
    forced = kit.render(_tiny(bell=None), str(out), force=True)
    assert forced['stats']['rendered'] == 3


def test_kit_check_and_problems(tmp_path):
    k = _tiny()
    assert kit.check(k)['ok']
    bad = _tiny(tick={'sound': 'tap/fingr', 'takes': 0, 'edits': {'speed': 2}})
    probs = kit.problems(bad)
    assert any('tap/finger' in p for p in probs) and any('takes' in p for p in probs)
    assert any('edits' in p for p in probs)
    r = kit.check(bad)
    assert not r['ok'] and r['problems']
    loud = _tiny(riser={'sound': 'pro_zip/up'})
    r = kit.check(loud)
    assert not r['ok'] and any('rising_glide' in {w['code'] for w in row['warnings']} for row in r['roles']['riser'])
    assert any('riser is sounded' in n for n in r['notes']) and any('synth whip' in n for n in r['notes'])
    with pytest.raises(kit.KitError, match='unknown sound'):
        kit.load(bad)


@pytest.mark.parametrize('spec,match', [
    ({'sound': 'tap/finger', 'level_db': None}, 'level_db must be a number'),
    ({'sound': 'tap/finger', 'level_db': True}, 'level_db must be a number'),
    ({'sound': 'tap/finger', 'max_s': -1}, 'max_s must be a number of seconds, 0.01 or more'),
    ({'sound': 'tap/finger', 'max_s': 0.001}, 'max_s must be a number of seconds, 0.01 or more'),
    ({'sound': 'tap/finger', 'edits': {'peak_db': 6}}, 'over 0 dBFS'),
    ({'sound': 'tap/finger', 'edits': {'weight': 2.0}}, 'over 0 dBFS|lower peak_db'),
    ({'sound': 'tap/finger', 'edits': {'dur': 'long'}}, 'edits.dur must be a number'),
    ({'sound': 'tap/finger', 'edits': {'dur': 0.001}}, 'dur must be 0.01-60'),
    ({'sound': 'pluck/kalimba', 'pitches': ['C5', 'H5']}, "pitch 'H5'"),
    ({'sound': 'pluck/kalimba', 'pitches': ['C1']}, 'outside its f0 range'),
    ({'sound': 'whoosh_fast/crisp', 'edits': {'pitch': 'C5'}}, 'noise pass-by'),
    ({'sound': 'click/soft', 'edits': {'apex': 0.05}}, 'whoosh family'),
    ({'sound': 'whoosh_fast/tight', 'edits': {'dur': 0.3, 'apex': 0.29}}, 'must lie inside the sound'),
    ({'sound': 'pluck/kalimba', 'pitches': ['440', '440hz']}, 'are one pitch'),
    ({'sound': 'pluck/kalimba', 'pitches': ['C#5', 'Db5']}, 'are one pitch'),
    ({'sound': 'pluck/kalimba', 'pitches': ['C5', 'c5']}, 'are one pitch'),
])
def test_kit_check_refuses_what_kit_render_cannot_use(tmp_path, spec, match):
    """review: problems() let through values that made `kit render` crash (level_db null, max_s -1, a bad pitch after
    a first take was written), render silence (max_s 0.001), clip (peak_db 6) or overwrite one take with another
    (two spellings of one pitch): check and render now refuse the same things"""
    k = {'kit': 1, 'name': 'bad', 'sr': 48000, 'roles': {'bell': spec}}
    probs = kit.problems(k)
    assert probs and re.search(match, '; '.join(probs)), probs
    r = kit.check(k)
    assert not r['ok'] and r['problems'] and not r['roles']
    with pytest.raises(kit.KitError):
        kit.render(k, str(tmp_path / 'takes'))
    assert not (tmp_path / 'takes').exists() or not os.listdir(tmp_path / 'takes')


def test_kit_sr_and_role_name_collisions_are_problems():
    base = {'kit': 1, 'roles': {'tick': {'sound': 'tap/finger'}}}
    for sr in ('48000x', True, 22050, None):
        assert any('sr' in p for p in kit.problems(dict(base, sr=sr))), sr
    two = {'kit': 1, 'roles': {'bell': {'sound': 'pluck/kalimba', 'pitches': ['C5'], 'takes': 2},
                               'bell_c5': {'sound': 'tap/finger', 'takes': 2}}}
    assert any('case-insensitive' in p for p in kit.problems(two))
    with pytest.raises(kit.KitError, match='case-insensitive'):
        kit.plan(two)


def test_kit_take_labels_are_canonical(tmp_path):
    """one label per pitch (knobs.label): 'Db5' and 'Cs5' render to bell_Cs5.wav, Hz that equal a note take its name;
    kit render accepts every pitch kit check accepts (review: 'Cs5' passed the check and failed the render)"""
    k = {'kit': 1, 'name': 'labels', 'roles': {'bell': {'sound': 'pluck/kalimba', 'pitches': ['Db5', '587.33', 'Fs5']},
                                               'text': {'sound': 'pluck/kalimba_soft', 'pitches': ['Cs5']}}}
    assert kit.problems(k) == []
    assert [t['file'] for t in kit.plan(k)] == ['bell_Cs5.wav', 'bell_D5.wav', 'bell_Fs5.wav', 'text_Cs5.wav']
    lock = kit.render(k, str(tmp_path / 'takes'))
    assert lock['stats']['rendered'] == 4 and 'checks clean' in lock['summary']
    assert [t['pitch'] for t in lock['roles']['bell']['takes']] == ['Db5', '587.33', 'Fs5']   # as written


def test_kit_pitch_off_is_reported_by_render_and_check(tmp_path, monkeypatch):
    from earshot.forge import knobs
    k = {'kit': 1, 'name': 'off', 'roles': {'bell': {'sound': 'pluck/kalimba', 'pitches': ['E5']}}}
    monkeypatch.setattr(knobs, 'PITCH_TOL_CENTS', -1)           # nothing lands: every take misses
    lock = kit.render(k, str(tmp_path / 'takes'))
    tk = lock['roles']['bell']['takes'][0]
    assert 'pitch_off' in tk['checks'] and 'bell:pitch_off' in lock['summary']
    r = kit.check(k, str(tmp_path / 'takes'))                    # from the rendered file + the lock
    assert not r['ok'] and r['roles']['bell'][0]['source'] == 'rendered file'
    assert 'pitch_off' in {w['code'] for w in r['roles']['bell'][0]['warnings']}
    monkeypatch.setattr(knobs, 'PITCH_TOL_CENTS', 50)
    assert kit.check(k)['ok']


def test_kit_audition(tmp_path):
    r = kit.audition(_tiny(), str(tmp_path / 'aud.wav'), takes=str(tmp_path / 'takes'))
    y, fs = sf.read(str(tmp_path / 'aud.wav'))
    assert fs == 48000 and y.ndim == 2 and len(r['index']) == 5 and r['index'][0]['role'] == 'tick'
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-4 and (tmp_path / 'aud.txt').is_file()


# ---------------------------------------------------------------------------------------------- CLI and the API
def test_cli_sfx(tmp_path, capsys, wav):
    assert cli.main(['sfx', 'list', '--kind', 'hit', '--json']) == 0
    rows = json.loads(capsys.readouterr().out)['sounds']
    assert rows and all(any(k.split('.')[0] == 'hit' for k in r['kinds']) for r in rows)
    f = str(tmp_path / 't.wav')
    assert cli.main(['sfx', 'render', 'tap/finger', '-o', f, '--dur', '0.1', '--seed', '3']) == 0
    assert 'wrote' in capsys.readouterr().out and abs(sf.info(f).duration - 0.1) < 1e-4
    assert cli.main(['sfx', 'check', f]) == 0
    assert 'clean' in capsys.readouterr().out
    assert cli.main(['sfx', 'check', wav(_sweep(300, 1500))]) == 1
    assert 'semitones' in capsys.readouterr().out
    assert cli.main(['sfx', 'render', 'whoosh_fast/crisp', '-o', f, '--pitch', 'C5']) == 1   # a readable error
    assert 'noise pass-by' in capsys.readouterr().err


def test_cli_kit(tmp_path, capsys):
    k = str(tmp_path / 'audio' / 'kit.json')
    assert cli.main(['kit', 'init', '--preset', 'minimal', '-o', k]) == 0
    assert json.loads(open(k, encoding='utf-8').read())['name'] == 'minimal'
    assert cli.main(['kit', 'init', '--preset', 'minimal', '-o', k]) == 1                  # exists: no overwrite
    capsys.readouterr()
    small = str(tmp_path / 'small.json')
    with open(small, 'w', encoding='utf-8') as fh:
        json.dump(_tiny(bell=None, **{'whoosh.short': None}), fh)
    assert cli.main(['kit', 'render', small, '--out', str(tmp_path / 'takes'), '--json']) == 0
    r = json.loads(capsys.readouterr().out)
    assert r['stats']['takes'] == 2 and os.path.isfile(r['file'])
    assert cli.main(['kit', 'check', small, '--out', str(tmp_path / 'takes')]) == 0
    assert '2 / 2 takes clean' in capsys.readouterr().out


def test_api_and_mcp_tools(tmp_path):
    from earshot import api
    assert api.sfx_list(category='pluck')['sounds'][0]['id'] == 'pluck/kalimba'
    text = mcp_server.tool_sfx_list(kind='bell')
    assert 'sounds in the bank' in text.split('\n')[0] and 'pluck/kalimba' in text
    f = str(tmp_path / 'mcp.wav')
    text = mcp_server.tool_sfx_render('tap/finger', f, dur=0.12)
    assert os.path.isfile(f) and 'onset' in text
    with pytest.raises(mcp_server.ToolFailure, match='unknown sound id'):
        mcp_server.tool_sfx_render('tap/fingr', f)
    k = tmp_path / 'k.json'
    k.write_text(json.dumps(_tiny(bell=None, **{'whoosh.short': None})), encoding='utf-8')
    text = mcp_server.tool_kit_render(str(k), str(tmp_path / 'takes'))
    assert '2 takes' in text and (tmp_path / 'takes' / 'kit.lock.json').is_file()


# ------------------------------------------------------------------------------- the mix keys and launch (0.4)
def test_old_presets_render_as_before():
    """calm / crisp / minimal carry no mix key: their plans, take keys and lock rows are the 0.3 ones (the version
    aside), so their takes render sample for sample as before"""
    from earshot import __version__
    for name in ('calm', 'crisp', 'minimal'):
        k = kit.preset(name)
        assert 'mix' not in k and not any(set(s) & set(kit.MIX_KEYS) for s in k['roles'].values() if s), name
        for t in kit.plan(k)[:6]:
            assert set(t) == {'role', 'file', 'sound', 'seed', 'pitch', 'edits', 'max_s', 'k'}
            old = kit._digest({'sound': t['sound'], 'entry': kit._digest(bank.get(t['sound'])), 'seed': t['seed'],
                               'pitch': t['pitch'], 'edits': t['edits'], 'max_s': t['max_s'], 'sr': 48000,
                               'bank': bank.VERSION, 'earshot': __version__})
            assert kit.take_key(t) == old


def test_launch_preset_levels_sync_and_runs():
    k = kit.preset('launch')
    on = {r: s for r, s in k['roles'].items() if s}
    assert set(on) == set(kit.ROLES) - {'suck'} and k['off']['suck']
    for role, spec in on.items():                                   # every sounded role is placed against a bed
        assert 0.5 <= spec['bed_db'] <= 12 and -100 <= spec['offset_ms'] <= 100, role
        assert spec['level_db'] <= 0 and spec['sound'] in bank.ids()
    assert on['land.light']['offset_ms'] < on['tock']['offset_ms'] < on['click']['offset_ms'] < 0   # touches lead
    assert on['whoosh.short']['offset_ms'] > 0 and on['swell']['offset_ms'] > 0 and on['land']['offset_ms'] > 0
    assert on['riser']['offset_ms'] < 0 and on['riser']['align'] == 'peak'     # its climax before the cut, its
    for r in ('riser', 'swell'):                                    # fall across it (the members ring on past it)
        assert on[r]['align'] == 'peak' and not any('aligned on its end' in n for n in kit.check(
            {'kit': 1, 'roles': {r: dict(on[r], takes=1)}})['notes']), r
    assert on['pop']['min_gap_ms'] > on['key']['min_gap_ms'] > on['tick']['min_gap_ms']
    assert on['tick']['width'] < 0.5 and 0.3 <= on['bell']['width'] <= 0.7
    assert max(s['level_db'] for s in on.values()) == 0.0
    new = {'cut_hit', 'ui_pop', 'knock', 'mallet', 'shutter', 'reverse_swell'}
    assert new <= {s['sound'].split('/')[0] for s in on.values()}
    mix = k['mix']
    assert mix['duck_db'] == 0 and set(mix['stop_down']['before']) <= set(on)


def test_check_notes_an_end_aligned_riser_that_rings_past_its_peak():
    """riser/dark peaks 0.25 s before its end (the last 5 ms within 50 dB): aligned on the end, its climax would land
    a quarter second before the cue; forge_v2's risers stop on their loudest sample and pass"""
    m = bank.get('riser/dark')['meta']
    assert m['end_s'] - m['peak_s'] > 0.2
    r = kit.check({'kit': 1, 'roles': {'riser': {'sound': 'riser/dark'}}})       # ALIGN's default for riser: end
    assert any(n.startswith('riser: aligned on its end') and "'peak'" in n for n in r['notes']), r['notes']
    r = kit.check({'kit': 1, 'roles': {'riser': {'sound': 'riser/dark', 'align': 'peak'}}})
    assert not any('aligned on its end' in n for n in r['notes'])
    r = kit.check({'kit': 1, 'roles': {'riser': {'sound': 'riser/rush'}}})
    assert not any('aligned on its end' in n for n in r['notes'])


@pytest.mark.parametrize('spec,match', [
    ({'sound': 'tap/finger', 'bed_db': 0.0}, 'bed_db must be a number in 0.5..30'),
    ({'sound': 'tap/finger', 'bed_db': 'loud'}, 'bed_db must be a number'),
    ({'sound': 'tap/finger', 'offset_ms': 900}, 'offset_ms must be a number in -500..500'),
    ({'sound': 'tap/finger', 'min_gap_ms': -5}, 'min_gap_ms must be a number in 0..2000'),
    ({'sound': 'tap/finger', 'width': 1.5}, 'width must be a number in 0..1'),
])
def test_mix_keys_are_checked(spec, match):
    probs = kit.problems({'kit': 1, 'roles': {'tick': spec}})
    assert probs and re.search(match, '; '.join(probs)), probs


def test_mix_block_is_checked():
    base = {'kit': 1, 'roles': {'hit': {'sound': 'cut_hit/punch'}}}
    assert kit.problems(dict(base, mix={'duck_db': 0, 'stop_down': {'before': ['hit'], 'depth_db': [6, 12]}})) == []
    assert any('unknown' in p for p in kit.problems(dict(base, mix={'stop_down': {'before': ['drop']}})))
    assert any('[lo, hi]' in p for p in kit.problems(dict(base, mix={'stop_down': {'length_s': [0.5, 0.3]}})))
    assert any('an object' in p for p in kit.problems(dict(base, mix=3)))
    for before in ([['hit']], [{'role': 'hit'}], [3]):                 # reported, not a TypeError
        probs = kit.problems(dict(base, mix={'stop_down': {'before': before}}))
        assert any('mix.stop_down.before' in p and 'unknown' in p for p in probs), before
    assert any('a list of roles' in p for p in kit.problems(dict(base, mix={'stop_down': {'before': 'hit'}})))
    assert not kit.check(dict(base, mix={'stop_down': {'before': [['hit']]}}))['ok']
    with pytest.raises(kit.KitError, match='mix.stop_down.before'):
        kit.load(dict(base, mix={'stop_down': {'before': [['hit']]}}))


def test_stereo_width_lands_the_correlation():
    rng = np.random.default_rng(3)
    t = np.arange(int(0.3 * SR)) / SR
    mono = (np.sin(2 * np.pi * 1200 * t) + 0.3 * rng.standard_normal(len(t))) * np.exp(-t / 0.05)
    mono *= 0.8 / np.abs(mono).max()
    y, info = kit.stereo_width(mono, 0.3)                            # a mono take comes back wide
    assert y.shape == (len(t), 2) and info['ok'] and abs(kit.corr(y) - 0.3) <= kit.WIDTH_TOL
    assert abs(np.abs(y).max() - 0.8) < 1e-4                         # at its own sample peak
    assert np.array_equal(y, kit.stereo_width(mono, 0.3)[0])         # deterministic
    z, info = kit.stereo_width(y, 0.9)                               # a wide take narrowed
    assert info['ok'] and abs(kit.corr(z) - 0.9) <= kit.WIDTH_TOL and info['before'] < 0.4
    low = np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.1)              # all under 250 Hz: as far as it goes
    _, info = kit.stereo_width(low, 0.0)
    assert not info['ok'] and info['measured'] > 0.5


def test_gain_place_and_runs():
    take = {'onset_s': 0.002, 'peak_s': 0.03, 'end_s': 0.4, 'l50_lufs': -10.0, 'peak_db': -1.0}
    role = {'align': 'onset', 'level_db': -6.0, 'bed_db': 8.0, 'offset_ms': -40}
    g = kit.gain_db(take, role, bed_lufs=-24.0)
    assert abs(g - (-24.0 + kit.over_bed(8.0) + 10.0)) < 0.01 and abs(kit.over_bed(8.0) - 7.25) < 0.01
    assert kit.gain_db(take, role, bed_lufs=-2.0) == 0.0             # never over the ceiling (-1 dBFS at its peak)
    assert kit.gain_db(take, role) == -6.0 and kit.gain_db(take, {'level_db': -3.0}, -24.0) == -3.0
    assert kit.place(take, role, 2.0) == round(2.0 - 0.04 - 0.002, 4)
    assert kit.place(take, {'align': 'peak', 'offset_ms': 37}, 2.0) == round(2.0 + 0.037 - 0.03, 4)
    assert kit.place(take, {'align': 'end'}, 2.0) == 1.6
    assert kit.space_run([1.0, 1.05, 1.06, 1.5], 87) == [1.0, 1.087, 1.174, 1.5]
    assert kit.space_run([0.3, 0.1], 0) == [0.1, 0.3]


def test_bed_level_reads_the_music_before_the_cue():
    rng = np.random.default_rng(5)
    music = np.concatenate([0.01 * rng.standard_normal(SR), 0.1 * rng.standard_normal(SR)])
    quiet, loud = kit.bed_level(music, 1.0), kit.bed_level(music, 2.0)
    assert 18 < loud - quiet < 22                                    # 20 dB apart: the second before each cue
    assert abs(loud - kit.loud_lufs(music[SR:])) < 1.5
    with pytest.raises(kit.KitError, match='no music'):
        kit.bed_level(music, 0.01)


def test_a_silent_bed_is_no_bed():
    """the second before a cue in digital silence (a cold open, a gap in the score) used to read -240.69 LUFS and
    set the effect ~220 dB down: it is no bed (None), and gain_db falls back to the role's level_db"""
    rng = np.random.default_rng(6)
    music = np.concatenate([np.zeros(2 * SR), 0.1 * rng.standard_normal(SR), 1e-5 * rng.standard_normal(SR)])
    assert kit.bed_level(music, 1.5) is None and kit.bed_level(music, 4.0) is None     # silence; a -100 dB hiss
    assert kit.bed_level(music, 3.0) > -30
    take = {'onset_s': 0.002, 'peak_s': 0.03, 'end_s': 0.4, 'l50_lufs': -12.0, 'peak_db': -1.0}
    role = {'align': 'onset', 'level_db': -8.0, 'bed_db': 8.5}
    assert kit.gain_db(take, role, kit.bed_level(music, 1.5)) == -8.0
    assert kit.gain_db(take, role, -240.69) == -8.0                 # a caller's own silent measure: no bed either
    assert kit.gain_db(take, role, kit.BED_FLOOR_LUFS) < -30


def test_mono_takes_and_beds_measure_as_dual_mono():
    """a mono take plays on both channels at unity (audition, any dual-mono mixer) and a mono file loads as two
    channels: both measures count it so, or a mono take with bed_db lands 3 dB hot and a 1-D bed reads 3 dB low"""
    rng = np.random.default_rng(8)
    t = np.arange(int(0.3 * SR)) / SR
    mono = 0.3 * np.sin(2 * np.pi * 900 * t) * np.exp(-t / 0.05)
    dual = np.stack([mono, mono], 1)
    assert kit.loud_lufs(mono) == kit.loud_lufs(dual) == kit.loud_lufs(mono[:, None])
    music = 0.05 * rng.standard_normal(2 * SR)
    assert kit.bed_level(music, 1.5) == kit.bed_level(np.stack([music, music], 1), 1.5)
    assert abs(kit.bed_level(music[:, None], 1.5) - kit.bed_level(music, 1.5)) < 1e-9


def test_a_take_mixed_at_gain_db_lands_bed_db_over_its_bed():
    """the level model end to end: a mono take set by gain_db and played dual mono over a steady stereo bed puts the
    mix's loudest 50 ms bed_db over bed_level (a mono take measured on one channel landed 3 dB hot)"""
    rng = np.random.default_rng(11)
    bed = 0.02 * rng.standard_normal((3 * SR, 2))
    t = np.arange(int(0.3 * SR)) / SR
    take = 0.5 * np.sin(2 * np.pi * 700 * t) * np.exp(-t / 0.06) * np.minimum(t / 0.002, 1.0)
    row = {'l50_lufs': kit.loud_lufs(take), 'peak_db': float(20 * np.log10(np.abs(take).max()))}
    for bed_db in (4.0, 8.0):
        cue = 2.0
        lv = kit.bed_level(bed, cue)
        g = 10 ** (kit.gain_db(row, {'bed_db': bed_db}, lv) / 20)
        mix = bed.copy()
        i = int(cue * SR)
        mix[i:i + len(take)] += (g * take)[:, None]                 # dual mono at unity
        assert abs(kit.loud_lufs(mix[i:i + len(take)]) - (lv + bed_db)) < 1.0, bed_db


def test_kit_render_with_mix_keys(tmp_path):
    k = {'kit': 1, 'name': 'mixed', 'roles': {
        'pop': {'sound': 'ui_pop/soft', 'takes': 2, 'bed_db': 8.5, 'offset_ms': 4, 'min_gap_ms': 114, 'width': 0.5},
        'tock': {'sound': 'knock/wood', 'offset_ms': -40, 'level_db': -3.0}},
        'mix': {'level': 'bed', 'duck_db': 0, 'stop_down': {'before': ['tock'], 'depth_db': [6, 12]}}}
    out = tmp_path / 'kit'
    lock = kit.render(k, str(out))
    assert 'checks clean' in lock['summary'] and lock['mix']['duck_db'] == 0
    pop, tock = lock['roles']['pop'], lock['roles']['tock']
    assert pop['bed_db'] == 8.5 and pop['offset_ms'] == 4.0 and pop['min_gap_ms'] == 114.0 and pop['width'] == 0.5
    assert 'bed_db' not in tock and tock['offset_ms'] == -40.0 and 'width' not in tock
    for tk in pop['takes']:
        y, _ = sf.read(str(out / tk['file']))
        assert abs(tk['corr'] - 0.5) <= kit.WIDTH_TOL and abs(kit.corr(y) - tk['corr']) < 0.01
        assert abs(tk['l50_lufs'] - kit.loud_lufs(y)) < 0.05
    assert 'l50_lufs' not in tock['takes'][0] and 'corr' not in tock['takes'][0]
    again = kit.render(k, str(out))
    assert again['stats']['cached'] == 3 and again['roles']['pop']['takes'][0]['l50_lufs'] == pop['takes'][0]['l50_lufs']
    r = kit.audition(k, str(tmp_path / 'aud.wav'), takes=str(out))
    assert len(r['index']) == 3
