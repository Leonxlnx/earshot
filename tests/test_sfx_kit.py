"""earshot sfx (the model-free audit and the CLI) and earshot kit (presets, render + lock + cache, check, audition),
plus the MCP tool functions of the sound bank. Synthetic signals and short bank sounds only; no models."""
import json
import os

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


# --------------------------------------------------------------------------------------------------------- kits
def test_presets_are_valid_restrained_and_explained():
    assert kit.presets() == ['calm', 'crisp', 'minimal']
    for name in kit.presets():
        k = kit.preset(name)
        assert kit.problems(k) == [], name
        assert set(k['roles']) == set(kit.ROLES), (name, set(kit.ROLES) ^ set(k['roles']))
        for r in kit.RESTRAINED:
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
        assert len(files) == len(set(files))


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
