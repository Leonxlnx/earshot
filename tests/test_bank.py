"""the shipped sound bank (earshot.forge.bank) and its knobs (earshot.forge.knobs): faithful re-renders against the
stored fingerprints, list / get, levels, the build's refusal of picks that do not re-render, add-fit, and dur / apex /
pitch landing on their targets. No models; every sound is rendered here from code."""
import json

import numpy as np
import pytest
import soundfile as sf

from earshot.forge import bank, knobs

SR = 48000
# a dozen short entries across recipes (the long designed sounds take seconds each; bank.check covers them all)
SHORT = ['click/soft', 'pro_click/tink', 'hover_tick/ping', 'pro_hover_tick/soft', 'tap/finger', 'pro_tap/soft',
         'pro_pop_in/round', 'pro_notify/ding_1822', 'pro_typing_key/glass', 'typing_key/crisp', 'pop/1',
         'toggle/micro', 'pro_sub_drop/quick', 'pro_zip/down']


def test_bank_contents():
    v = bank.VERSION
    assert v.startswith('forge_v2r6') and 'pluck1' in v
    ids = bank.ids()
    forge = [i for i in ids if bank.get(i)['type'] == 'forge' and not i.startswith('pluck/')]
    assert len(forge) == 183 and len({i.split('/')[0] for i in forge}) == 34
    assert {'pluck/kalimba', 'pluck/kalimba_soft', 'pluck/kalimba_ring'} <= set(ids)
    for i in ids:
        e = bank.get(i)
        assert e['id'] == i and e['category'] == i.split('/')[0]
        m = e['meta']
        assert m['dur_s'] > 0 and 0 <= m['onset_s'] <= m['peak_s'] <= m['dur_s'] and m['fp']['env_db'][0] <= 0
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
    assert p['param'] == 'f0' and abs(p['cents_off']) <= 30 and p['semitones'] == 4.0
    y, m = bank.render('pro_sub_drop/quick', pitch='G1')
    p = m['edits']['pitch']
    e = bank.get('pro_sub_drop/quick')['params']
    assert abs(p['params']['f1'] - 49.0) < 0.01
    assert abs(p['params']['f0'] / p['params']['f1'] - e['f0'] / e['f1']) < 1e-3          # the glide ratio kept
    assert p['measured_hz'] and abs(p['cents_off']) <= 50


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
