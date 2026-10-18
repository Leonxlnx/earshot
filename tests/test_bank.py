"""the shipped sound bank (earshot.forge.bank) and its knobs (earshot.forge.knobs): faithful re-renders against the
stored fingerprints, list / get, levels, the build's refusal of picks that do not re-render, add-fit, and dur / apex /
pitch landing on their targets. No models; every sound is rendered here from code."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

from earshot.forge import bank, knobs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SR = 48000
# a dozen short entries across recipes (the long designed sounds take seconds each; bank.check covers them all)
SHORT = ['click/soft', 'pro_click/tink', 'hover_tick/ping', 'pro_hover_tick/soft', 'tap/finger', 'pro_tap/soft',
         'pro_pop_in/round', 'pro_notify/ding_1822', 'pro_typing_key/glass', 'typing_key/crisp', 'pop/1',
         'toggle/micro', 'pro_sub_drop/quick', 'pro_zip/down']


def test_bank_contents():
    v = bank.VERSION
    assert v.startswith('forge_v2r6') and 'pluck1' in v and 'fit1' in v
    ids = bank.ids()
    forge = [i for i in ids if bank.get(i)['type'] == 'forge' and not i.startswith('pluck/')]
    assert len(forge) == 183 and len({i.split('/')[0] for i in forge}) == 34
    assert {'pluck/kalimba', 'pluck/kalimba_soft', 'pluck/kalimba_ring'} <= set(ids)
    for i in ids:
        e = bank.get(i)
        assert e['id'] == i and e['category'] == i.split('/')[0]
        m = e['meta']
        assert m['dur_s'] > 0 and 0 <= m['onset_s'] <= m['peak_s'] <= m['dur_s'] and m['fp']['env_db'][0] <= 0
        if e['type'] == 'fit':
            assert isinstance(e['model'], dict) and isinstance(e['seed'], int)
            continue
        assert e['recipe'] and e['fn'] and isinstance(e['params'], dict) and isinstance(e['seed'], int)
    tk = bank.get('typing_key/crisp')                    # the category is the public name, the module is inside
    assert tk['recipe'] == 'keypress' and bank.get('pro_typing_run/blip')['fn'] == 'render_run'


@pytest.mark.parametrize('sid', SHORT)
def test_rerender_matches_fingerprint(sid):
    """a fresh render equals what the bank stored: length exact, peak within 5 ms, envelope within 0.5 dB"""
    e = bank.get(sid)
    y, meta = bank.render(sid)
    d = bank.compare_fp(e['meta'], y)
    assert d['len_diff'] == 0 and abs(d['peak_diff_ms']) <= 5 and d['env_db'] <= 0.5, d
    assert meta['seed'] == e['seed'] and abs(meta['peak_db'] + 1.0) < 0.01
    assert meta['checks']['ok'], meta['checks']['summary']


# ----------------------------------------------------------------------------------------- shipped fit entries
FITS = ['fit/click/floss_lid', 'fit/tick/domino', 'fit/land.light/fingertip_book', 'fit/pop/mouth_plop',
        'fit/snap/switch_flip']


def test_fit_entries_load_with_their_cc0_sources():
    fits = [i for i in bank.ids() if i.startswith('fit/')]
    assert len(fits) >= 10 and set(FITS) <= set(fits)
    with open(os.path.join(REPO, 'docs', 'CREDITS_FIT.md'), encoding='utf-8') as fh:
        credits = fh.read()
    for i in fits:
        e = bank.get(i)
        assert e['type'] == 'fit' and e['category'] == 'fit' and bank.FIT_ID.match(i)
        src = e['source']
        assert src['license'] == 'CC0 1.0' and src['url'].startswith('https://freesound.org/')
        text = ' '.join(str(src[k]) for k in ('title', 'author', 'url')).lower()
        assert not any(c in text for c in ('zapsplat', 'soundsnap', 'sound-ideas', 'boom library'))
        assert 0.0 <= e['vary'] <= 0.6 and i.split('/')[1] in e['kinds'] and len(e['notes']) > 20
        assert 'fit_engine' not in e['model'] and e['meta']['fp']['env_db']
        assert i in credits and src['url'] in credits
    rows = bank.list(kind='key')
    assert any(r['id'].startswith('fit/key/') and r['source']['license'] == 'CC0 1.0' for r in rows)


@pytest.mark.parametrize('sid', FITS)
def test_fit_entry_rerenders_to_its_fingerprint(sid):
    """a fit entry re-renders as stored, as a clean take; another seed is another clean take of the same sound"""
    e = bank.get(sid)
    y, meta = bank.render(sid)
    d = bank.compare_fp(e['meta'], y)
    assert d['len_diff'] == 0 and abs(d['peak_diff_ms']) <= 5 and d['env_db'] <= 0.5, d
    assert meta['checks']['ok'], meta['checks']['summary']
    y2, m2 = bank.render(sid, seed=7)
    assert m2['checks']['ok'], m2['checks']['summary']
    assert abs(len(y2) - len(y)) <= 0.1 * len(y) and not np.allclose(y2[:min(len(y), len(y2))], y[:len(y2)])


def test_render_final_is_the_delivered_level():
    """render_final keeps the palette gain: the same samples as render() up to one constant factor"""
    e = bank.get('click/soft')
    x = bank.render_final('click/soft')
    y, _ = bank.render('click/soft', checks=False)
    g = 10 ** (e['gain_db'] / 20)
    assert abs(np.abs(x).max() - 10 ** (-1 / 20) * g) < 1e-3
    assert np.allclose(x / np.abs(x).max(), y / np.abs(y).max(), atol=1e-5)


def test_list_filters_and_unknown_ids():
    rows = bank.list(kind='whoosh')
    kinds = {k for r in rows for k in r['kinds']}
    assert {'whoosh.short', 'whoosh.long'} <= kinds
    longs = bank.list(kind='whoosh.long')
    assert longs and all('whoosh.long' in r['kinds'] for r in longs) and not any(
        r['category'] == 'whoosh_fast' for r in longs)
    one = bank.list(category='pluck')
    assert [r['id'] for r in one] == ['pluck/kalimba', 'pluck/kalimba_ring', 'pluck/kalimba_soft']
    assert one[0]['knobs']['pitch']['note'] == 'C5' and one[0]['knobs']['dur'] == 'trim'
    w = bank.list(category='whoosh_fast')[0]
    assert w['knobs']['apex'] and w['knobs']['pitch'] is None and 'note' in w['knobs']['why']
    with pytest.raises(bank.BankError, match='whoosh_fast/crisp'):
        bank.get('whoosh_fast/crisps')


def test_levels_weight_and_sr():
    y, m = bank.render('tap/finger', peak_db=-6.0, checks=False)
    assert abs(20 * np.log10(np.abs(y).max()) + 6.0) < 0.01 and m['checks'] is None
    y, m = bank.render('tap/finger', peak_db=-1.0, weight=0.5, checks=False)
    assert abs(m['peak_db'] + 7.02) < 0.02 and m['edits']['weight']['weight'] == 0.5
    with pytest.raises(bank.BankError, match='over 0 dBFS|lower peak_db'):
        bank.render('tap/finger', weight=4.0)
    y44, m44 = bank.render('tap/finger', sr=44100, checks=False)
    assert abs(len(y44) - round(len(y) * 44100 / SR)) <= 2 and m44['sr'] == 44100


def test_levels_over_0_dbfs_and_odd_sample_rates_are_refused(tmp_path, capsys):
    """review: `sfx render --peak 3` wrote a clipped WAV (exit 0) and any sr was resampled to, while --weight, the MCP
    tool and kits refused or restricted them"""
    for bad in (3.0, 0.5, float('nan'), True, '-1'):
        with pytest.raises(bank.BankError, match='over 0 dBFS|must be a number'):
            bank.render('tap/finger', peak_db=bad, checks=False)
    y, m = bank.render('tap/finger', peak_db=0.0, checks=False)            # 0 dBFS itself is fine
    assert abs(m['peak_db']) < 0.01
    for sr in (1000, 22050, 48000.5, True, '48000'):
        with pytest.raises(bank.BankError, match='sr must be one of'):
            bank.render('tap/finger', sr=sr, checks=False)
    from earshot import cli
    f = str(tmp_path / 'clip.wav')
    assert cli.main(['sfx', 'render', 'tap/finger', '-o', f, '--peak', '3']) == 1 and not os.path.exists(f)
    assert 'over 0 dBFS' in capsys.readouterr().err
    with pytest.raises(SystemExit):                                         # argparse: not one of the choices
        cli.main(['sfx', 'render', 'tap/finger', '-o', f, '--sr', '1000'])


def test_earshot_bank_missing_file_is_a_readable_error(monkeypatch, tmp_path, capsys):
    """review: EARSHOT_BANK naming a missing file broke every bank command with a raw FileNotFoundError; the docstring
    gave ';' as the separator, which is os.pathsep only on Windows"""
    missing = str(tmp_path / 'nope.json')
    monkeypatch.setenv('EARSHOT_BANK', missing)
    with pytest.raises(bank.BankError, match='EARSHOT_BANK file .*nope.json not found'):
        bank.ids()
    from earshot import cli
    assert cli.main(['sfx', 'list']) == 1
    err = capsys.readouterr().err
    assert 'BankError' in err and 'not found' in err and 'Traceback' not in err
    junk = tmp_path / 'junk.json'
    junk.write_text('{not json', encoding='utf-8')
    monkeypatch.setenv('EARSHOT_BANK', str(junk))
    with pytest.raises(bank.BankError, match='not readable as a bank file'):
        bank.ids()
    monkeypatch.delenv('EARSHOT_BANK')
    assert 'os.pathsep' in bank.__doc__ and len(bank.ids()) > 180


def _final(tmp_path, sid, scale=1.0):
    """a forge final folder holding one bank entry: report.json with its render block + the WAV it renders"""
    e = bank.get(sid)
    d = tmp_path / e['category'] / 'final'
    d.mkdir(parents=True)
    block = {k: e[k] for k in ('recipe', 'fn', 'member', 'params', 'seed', 'peak_db', 'gain_db') if k in e}
    if 'gain' in e:
        block['gain'] = e['gain']
    x = bank.render_final(sid) * scale
    sf.write(str(d / 'pick_01.wav'), x, SR, subtype='PCM_24')
    rep = {'schema': 2, 'recipe': e['recipe'], 'target': 'test',
           'picks': [{'name': 'pick_01', 'file': 'pick_01.wav', 'seed': e['seed'], 'render': block}]}
    (d / 'report.json').write_text(json.dumps(rep), encoding='utf-8')
    return d


def test_build_keeps_faithful_picks_and_refuses_the_rest(tmp_path):
    good = _final(tmp_path / 'a', 'click/soft')
    out = tmp_path / 'bank.json'
    doc = bank.build([str(good)], str(out), version='test1', log=None)
    assert list(doc['entries']) == ['click/soft'] and out.is_file()
    stored = json.loads(out.read_text(encoding='utf-8'))
    assert stored['entries']['click/soft']['params'] == bank.get('click/soft')['params']   # full precision
    assert stored['entries']['click/soft']['meta']['fp'] == bank.get('click/soft')['meta']['fp']
    bad = _final(tmp_path / 'b', 'click/soft', scale=0.5)        # a WAV the block does not render
    out2 = tmp_path / 'bank2.json'
    with pytest.raises(bank.BankError, match='refused, nothing written'):
        bank.build([str(bad)], str(out2), log=None)
    assert not out2.exists()


# ---------------------------------------------------------------------------------------------------------- knobs
def test_note_parsing():
    assert abs(knobs.hz('A4') - 440.0) < 1e-9 and abs(knobs.hz('C5') - 523.2511) < 1e-3
    assert abs(knobs.hz('Eb6') - knobs.hz('D#6')) < 1e-9 and abs(knobs.hz('440hz') - 440) < 1e-9
    assert knobs.hz(75) == 75.0 and knobs.note(659.26) == 'E5' and knobs.note(450).startswith('A4 +')
    for bad in ('H5', 'loud', True, 30000):
        with pytest.raises(knobs.KnobError):
            knobs.hz(bad)


def test_s_sharps_and_canonical_labels():
    """review: take files spell sharps 's' (bell_Cs5.wav) and cinetic's kit.py accepts 'Cs5', but knobs.midi refused
    it, so `kit render` died on a kit that `kit check` passed"""
    assert knobs.midi('Cs5') == knobs.midi('C#5') == knobs.midi('Db5') == 73 and knobs.midi('fs4') == 66
    assert abs(knobs.hz('Cs5') - knobs.hz('C#5')) < 1e-9
    y, m = bank.render('pluck/kalimba', pitch='Cs5', checks=False)
    assert m['edits']['pitch']['note'] == 'C#5' and m['edits']['pitch']['ok']
    assert {knobs.label(p) for p in ('C#5', 'Db5', 'Cs5', 554.365)} == {'Cs5'}
    assert knobs.label('440') == knobs.label('440hz') == knobs.label('A4') == 'A4'
    assert knobs.label(445) == '445hz' and knobs.label('Eb6') == 'Eb6'
    with pytest.raises(knobs.KnobError):
        knobs.label('H5')


def test_dur_lands_exactly():
    y, m = bank.render('pro_riser/air_gate', dur=1.0)          # the riser's own length key: exact, ends on its peak
    assert len(y) == SR and m['edits']['dur']['method'] == 'param' and m['peak_s'] > 0.97
    y, m = bank.render('tap/finger', dur=0.08)                  # shorter than the take: faded to the length
    assert len(y) == int(0.08 * SR) and m['edits']['dur']['fit'].startswith(('trim', 'exact'))
    assert np.abs(y[-48:]).max() < 0.05 * np.abs(y).max()
    y, m = bank.render('toggle/micro', dur=0.3)                 # no length key, longer: padded with silence
    assert m['edits']['dur']['method'] == 'trim' and m['edits']['dur']['fit'].startswith('pad')
    assert len(y) == int(0.3 * SR) and np.abs(y[-480:]).max() == 0.0
    y, m = bank.render('click/soft', dur=0.06)                  # the click's end gate: its own length key
    assert len(y) == int(0.06 * SR) and m['edits']['dur']['method'] == 'param'


def test_fit_length_fade_starts_after_keep_s():
    t = np.arange(int(0.6 * SR)) / SR
    x = np.minimum(t / 0.5, 1.0)                     # a ramp that peaks at 0.5 s, then holds
    y, how = knobs.fit_length(x, 0.52)
    assert y[int(0.5 * SR)] < 0.9                    # the default fade (a quarter, 130 ms) covers the peak
    y, how = knobs.fit_length(x, 0.52, keep_s=0.505)
    assert np.array_equal(y[:int(0.505 * SR)], x[:int(0.505 * SR)]) and 'fade 15 ms' in how
    assert len(y) == int(0.52 * SR) and abs(y[-1]) < 1e-3


@pytest.mark.parametrize('sid,apex,dur', [('whoosh_big/swell', 0.5, 0.6), ('whoosh_fast/heavy', 0.5, 0.6),
                                          ('swipe/velvet', 0.4, 0.43)])
def test_apex_with_a_short_tail_lands_and_edits_say_what_ships(sid, apex, dur):
    """review: the fade that fits a whoosh to dur covered an apex 0.03-0.1 s before the end, so the delivered peak
    came 34-189 ms early while edits['apex'] (measured before the fit) said ok"""
    y, m = bank.render(sid, apex=apex, dur=dur)
    a = m['edits']['apex']
    assert abs(m['peak_s'] - apex) <= knobs.APEX_TOL, a
    assert abs(a['measured'] - m['peak_s']) <= 0.001 and a['ok'] is True
    assert len(y) == int(round(dur * SR)) and 'fade' in m['edits']['dur']['fit']


def test_apex_ok_is_false_when_the_delivered_file_misses(monkeypatch):
    """edits['apex'] describes the delivered file: with the old fit (a fade over the apex) the miss is reported"""
    real = knobs.fit_length

    def old_fit(x, dur, keep_s=None):
        return real(x, dur)
    monkeypatch.setattr(knobs, 'fit_length', old_fit)
    y, m = bank.render('whoosh_big/swell', apex=0.5, dur=0.6)
    a = m['edits']['apex']
    assert abs(a['measured'] - m['peak_s']) <= 0.001
    assert abs(m['peak_s'] - 0.5) > knobs.APEX_TOL and a['ok'] is False


@pytest.mark.parametrize('sid,apex,dur', [('whoosh_fast/tight', 0.25, None), ('whoosh_fast/tight', 0.3, 0.7),
                                          ('swipe/velvet', 0.12, 0.5)])
def test_apex_lands_within_5_ms(sid, apex, dur):
    y, m = bank.render(sid, apex=apex, dur=dur)
    assert abs(m['peak_s'] - apex) <= 0.005, m['edits']
    assert m['edits']['apex']['ok'] and m['edits']['apex']['renders'] <= 1 + knobs.MAX_FIX
    if dur:
        assert len(y) == int(round(dur * SR))


def test_pitch_goes_through_the_recipe():
    y, m = bank.render('pluck/kalimba', pitch='E5')
    p = m['edits']['pitch']
    assert p['param'] == 'f0' and abs(p['cents_off']) <= 30 and p['semitones'] == 4.0 and p['ok'] is True
    assert 'pitch_off' not in {w['code'] for w in m['checks']['warnings']}
    y, m = bank.render('pro_sub_drop/quick', pitch='G1')
    p = m['edits']['pitch']
    e = bank.get('pro_sub_drop/quick')['params']
    assert abs(p['params']['f1'] - 49.0) < 0.01
    assert abs(p['params']['f0'] / p['params']['f1'] - e['f0'] / e['f1']) < 1e-3          # the glide ratio kept
    assert p['measured_hz'] and abs(p['cents_off']) <= 50


def test_click_pitch_lands_on_the_heard_partial():
    """review: a click's `pitch` is its tink centre, not the partial heard (click/soft 2454 Hz, heard 2903 Hz), so
    --pitch C7 measured 362 cents off with no warning while describe advertised a native note. The note is now the
    measured partial, a target is landed by measurement, and edits.pitch.ok says whether it landed"""
    d = knobs.describe(bank.get('click/soft'))['pitch']
    assert d['mode'] == 'partial' and abs(d['native_hz'] - 2903.4) < 1 and d['note'] == knobs.note(d['native_hz'])
    assert abs(d['param_hz'] - 2454.3) < 1 and d['range_hz'][0] > 1200 * 2903 / 2455 - 1
    for sid in ('click/soft', 'pro_click/glass', 'pro_click/mouse'):
        y, m = bank.render(sid, pitch='C7')
        p = m['edits']['pitch']
        assert p['ok'] is True and abs(p['cents_off']) <= knobs.PITCH_TOL_CENTS, (sid, p)
        assert 'pitch_off' not in {w['code'] for w in m['checks']['warnings']}
    with pytest.raises(knobs.KnobError, match='no clear tonal partial'):
        bank.render('click/thock', pitch='C7')
    assert knobs.describe(bank.get('click/thock'))['pitch'] is None
    with pytest.raises(knobs.KnobError, match='outside its pitch range'):
        bank.render('toggle/latch', pitch='C7')        # heard 3370 Hz: its range starts at ~2266 Hz


def test_a_missed_pitch_is_a_pitch_off_warning(monkeypatch):
    """edits.pitch.ok False puts 'pitch_off' in the render's checks (and a kit take's), so a take that missed its note
    is re-rolled or reported, never silently kept"""
    assert knobs.pitch_warning({'pitch': {'ok': True}}) is None and knobs.pitch_warning({}) is None
    w = knobs.pitch_warning({'pitch': {'ok': False, 'note': 'C7', 'measured_hz': 2579.7, 'cents_off': 362}})
    assert w['code'] == 'pitch_off' and '+362 cents' in w['msg']
    monkeypatch.setattr(knobs, 'PITCH_TOL_CENTS', -1)           # nothing lands now
    y, m = bank.render('pluck/kalimba', pitch='E5')
    assert m['edits']['pitch']['ok'] is False and not m['checks']['ok']
    assert 'pitch_off' in {w['code'] for w in m['checks']['warnings']} and 'pitch E5' in m['checks']['summary']


def test_pitch_and_apex_refusals_say_why():
    with pytest.raises(knobs.KnobError, match='noise pass-by'):
        bank.render('whoosh_fast/crisp', pitch='C5')
    with pytest.raises(knobs.KnobError, match='outside its f1 range'):
        bank.render('pro_sub_drop/quick', pitch='C4')
    with pytest.raises(knobs.KnobError, match='tonal layer is off'):
        bank.render('hover_tick/air', pitch='C7')
    with pytest.raises(knobs.KnobError, match='whoosh family'):
        bank.render('click/soft', apex=0.05)


# ------------------------------------------------------------------------------------------------------- add-fit
@pytest.fixture(scope='module')
def tone_model():
    from earshot.forge import fit
    t = np.arange(int(0.3 * SR)) / SR
    x = (np.sin(2 * np.pi * 660 * t) + 0.3 * np.sin(2 * np.pi * 1980 * t)) * np.exp(-t / 0.08) * 0.5
    x[:48] *= np.linspace(0, 1, 48)
    return fit.fit(x.astype(np.float32), kind='sines')


def test_add_fit_writes_a_fit_entry_and_checks_its_licence(tmp_path, monkeypatch, tone_model):
    src = {'id': 'test:1', 'title': 'a test tone', 'author': 'the test', 'license': 'CC0 1.0', 'url': 'https://x.test'}
    path = tmp_path / 'fit.json'
    for bad_id in ('fit/bell', 'hit/tone', 'fit/Bell/Tone'):
        with pytest.raises(bank.BankError, match='fit/<role>/<name>'):
            bank.add_fit(tone_model, bad_id, src, str(path), log=None)
    with pytest.raises(bank.BankError, match='forbids'):
        bank.add_fit(tone_model, 'fit/bell/tone', {**src, 'license': 'CC BY-NC 4.0'}, str(path), log=None)
    with pytest.raises(bank.BankError, match='not CC0'):
        bank.add_fit(tone_model, 'fit/bell/tone', {**src, 'license': 'CC BY 4.0'}, str(path), log=None)
    with pytest.raises(bank.BankError, match='missing author'):
        bank.add_fit(tone_model, 'fit/bell/tone', {**src, 'author': ''}, str(path), log=None)
    e = bank.add_fit(tone_model, 'fit/bell/tone', src, str(path), log=None)
    assert e['type'] == 'fit' and e['source'] == src and 'fit_engine' not in e['model']
    assert e['pitch_hz'] and abs(1200 * np.log2(e['pitch_hz'] / 660.0)) < 30
    with pytest.raises(bank.BankError, match='already in'):
        bank.add_fit(tone_model, 'fit/bell/tone', src, str(path), log=None)
    monkeypatch.setenv('EARSHOT_BANK', str(path))
    bank.reload()
    try:
        assert bank.VERSION.endswith('+fit1') and 'fit/bell/tone' in bank.ids()
        y, m = bank.render('fit/bell/tone', pitch='A5')
        assert abs(m['edits']['pitch']['cents_off']) <= 30 and m['checks'] is not None
        assert bank.list(kind='bell')[0]['id'] in bank.ids()
    finally:
        monkeypatch.delenv('EARSHOT_BANK')
        bank.reload()
    assert 'fit/bell/tone' not in bank.ids()
