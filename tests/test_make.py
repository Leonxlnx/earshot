"""earshot.make and earshot.pack: a request in words becomes a file (format by extension, takes, the one-line summary,
`also`, `unknown`, the error for a request nothing answers, explicit arguments over words, a theme, a loop once),
a whole set (files, manifest, levels by kind, a sprite whose map covers every sound), specs as dicts and as JSON
files with a byte order mark, the catalogue listing, the command line, and packing a set's folder. Model-free; every
file goes to tmp_path."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

from earshot import cli
from earshot.make import WEIGHT, make, make_set, sounds
from earshot.pack import pack
from earshot.sounds import registry, sets


def _read(p):
    y, sr = sf.read(p, always_2d=True)
    return y, sr


# ---- make

def test_make_writes_the_file_it_names_in_its_format(tmp_path):
    for ext, fmt in (('wav', 'WAV'), ('flac', 'FLAC')):
        p = str(tmp_path / f'ok.{ext}')
        res = make('a soft success chime', p)
        assert res['files'] == [p] and os.path.isfile(p)
        info = sf.info(p)
        assert info.format == fmt and info.samplerate == 48000 and info.channels == 2
        assert info.duration == pytest.approx(res['takes'][0]['dur_s'], abs=0.002)


def test_make_n_takes_are_numbered_and_differ(tmp_path):
    res = make('coin pickup', str(tmp_path / 'coin.wav'), n=3)
    want = [str(tmp_path / f'coin_{i}.wav') for i in (1, 2, 3)]
    assert res['files'] == want and all(os.path.isfile(p) for p in want)
    assert not os.path.exists(tmp_path / 'coin.wav')
    ys = [_read(p)[0] for p in want]
    for a, b in ((0, 1), (1, 2), (0, 2)):
        assert ys[a].shape != ys[b].shape or not np.allclose(ys[a], ys[b])
    assert [t['seed'] for t in res['takes']][0] in (0, 101, 202, 303)


def test_make_summary_also_and_unknown(tmp_path):
    p = str(tmp_path / 'sq.wav')
    res = make('a squishy success chime', p)
    first = res['summary'].splitlines()[0]
    t = res['takes'][0]
    assert first.startswith(f"sq.wav: {res['sound']} ({t['dur_s']:.2f} s")
    assert res['sound'] == t['id'] and registry.get(res['sound'])
    assert res['also'] and res['sound'] not in res['also'] and len(res['also']) <= 3
    assert all(registry.get(a) for a in res['also'])
    assert 'also: ' + ', '.join(res['also']) in res['summary']
    assert res['unknown'] == ['squishy']
    assert 'not understood: squishy' in res['summary']


def test_make_default_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = make(sound='tick/wood')
    assert res['files'] == ['tick_wood.wav'] and os.path.isfile(tmp_path / 'tick_wood.wav')


def test_make_nothing_answers(tmp_path):
    with pytest.raises(ValueError, match='earshot sounds'):
        make('a frobnicating zorp', str(tmp_path / 'x.wav'))
    with pytest.raises(ValueError, match='say what sound'):
        make(None, str(tmp_path / 'x.wav'))
    with pytest.raises(ValueError, match='unknown sound'):
        make(sound='no/such', out=str(tmp_path / 'x.wav'))


def test_explicit_arguments_override_the_words(tmp_path):
    words = 'a success chime in D, 2 seconds'
    base = make(words, str(tmp_path / 'a.wav'))['takes'][0]
    assert base['pitch'].startswith('D') and base['dur_s'] >= 2.0
    res = make(words, str(tmp_path / 'b.wav'), sound=base['id'], pitch='A5', voice='kalimba', dur=0.5)
    t = res['takes'][0]
    assert t['pitch'] == 'A5' and t['voice'] == 'kalimba'
    assert t['dur_s'] < base['dur_s'] - 0.8


def test_make_applies_a_theme(tmp_path):
    t = make(out=str(tmp_path / 'g.wav'), sound='success/rise', theme='glass')['takes'][0]
    assert t['voice'] == 'glockenspiel' and t['pitch'] == 'D5'
    t = make(out=str(tmp_path / 'w.wav'), sound='success/rise', theme={'base': 'wood', 'key': 'A'})['takes'][0]
    assert t['voice'] == 'marimba' and t['pitch'] == 'A5'


def test_make_a_loop_once(tmp_path):
    res = make(out=str(tmp_path / 'r.wav'), sound='rain/tent', dur=3.0, loop=False)
    t = res['takes'][0]
    assert t['loop'] is False and 'seamless loop' not in res['summary']
    assert t['dur_s'] <= 3.05
    y, _ = _read(res['files'][0])
    assert np.abs(y[0]).max() < 1e-4 and np.abs(y[-1]).max() < 1e-4


# ---- make_set

def _check_manifest(folder, n):
    with open(os.path.join(folder, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    rows = man['sounds']
    assert len(rows) == n
    for r in rows:
        assert {'name', 'file', 'sound', 'dur_s', 'lufs_max', 'loop'} <= set(r), r
        assert os.path.isfile(os.path.join(folder, r['file']))
        assert r['name'] == os.path.splitext(r['file'])[0]
        assert registry.get(r['sound']) and isinstance(r['loop'], bool)
        assert sf.info(os.path.join(folder, r['file'])).duration == pytest.approx(r['dur_s'], abs=0.002)
    return man


def test_make_set_ready_made_with_sprite_and_pack_it_again(tmp_path):
    out = str(tmp_path / 'arcade')
    res = make_set('arcade', out=out, sprite='sprite.wav')
    names = [n for n, _ in sets.SETS['arcade']['sounds']]
    man = _check_manifest(out, len(names))
    assert [r['name'] for r in man['sounds']] == names
    assert [r['sound'] for r in man['sounds']] == [s for _, s in sets.SETS['arcade']['sounds']]
    assert sorted(res['files']) == sorted(os.path.join(out, f'{n}.wav') for n in names)
    assert sf.info(res['files'][0]).subtype == 'PCM_16'                  # a set's wav files are 16-bit
    assert res['summary'].splitlines()[0].startswith(f'{len(names)} sounds in {out} (wav), manifest.json')
    with open(os.path.join(out, 'sprite.json'), encoding='utf-8') as fh:
        mp = json.load(fh)
    assert mp['src'] == ['sprite.wav'] and set(mp['sprite']) == set(names)
    spans = sorted(mp['sprite'].values())
    for a, b in zip(spans, spans[1:]):
        assert a[0] + a[1] <= b[0], (a, b)                                # no overlap
    total = sf.info(os.path.join(out, 'sprite.wav')).duration * 1000
    assert spans[-1][0] + spans[-1][1] <= total + 2
    for r in man['sounds']:
        assert abs(mp['sprite'][r['name']][1] - 1000 * r['dur_s']) <= 2

    # pack on the folder: exactly the manifest's sounds, not the earlier sprite, not a stray file
    sf.write(os.path.join(out, 'stray.wav'), np.zeros((4800, 2)) + 0.1, 48000)
    again = pack(out, os.path.join(out, 'again.wav'))
    assert list(again['sprite']) == names


def test_make_set_levels_follow_the_weights(tmp_path):
    spec = {'sounds': [{'name': 'tap', 'sound': 'tap/tone'}, {'name': 'success', 'sound': 'success/rise'}]}
    res = make_set(spec, out=str(tmp_path / 's'))
    lv = {r['name']: r['lufs_max'] for r in res['sounds']}
    k = {r['name']: registry.get(r['sound']).kind for r in res['sounds']}
    assert (k['tap'], k['success']) == ('tap', 'chime')
    assert lv['success'] - lv['tap'] == pytest.approx(WEIGHT['chime'] - WEIGHT['tap'], abs=1.0)
    assert lv['success'] == pytest.approx(-16.0 + WEIGHT['chime'], abs=0.6)
    _check_manifest(str(tmp_path / 's'), 2)


def test_make_set_spec_file_with_a_byte_order_mark(tmp_path):
    spec = {'theme': 'wood', 'format': 'flac', 'lufs': -20,
            'sounds': [{'name': 'ok', 'want': 'a soft success chime'}, {'name': 'knock', 'sound': 'tick/wood'}]}
    p = tmp_path / 'spec.json'
    p.write_bytes(b'\xef\xbb\xbf' + json.dumps(spec).encode('utf-8'))
    res = make_set(str(p), out=str(tmp_path / 'set'))
    assert sorted(os.path.basename(f) for f in res['files']) == ['knock.flac', 'ok.flac']
    man = _check_manifest(str(tmp_path / 'set'), 2)
    assert man['theme'] == 'wood' and man['format'] == 'flac'
    assert {r['name']: r.get('pitch') for r in man['sounds']}['ok'][0] == 'G'      # the wood theme's key


def test_make_set_refuses_unknown_fields_and_missing_names(tmp_path):
    with pytest.raises(ValueError, match='colour'):
        make_set({'sounds': [{'name': 'a', 'sound': 'tap/tone', 'colour': 'red'}]}, out=str(tmp_path / 'a'))
    with pytest.raises(ValueError, match='name'):
        make_set({'sounds': [{'sound': 'tap/tone'}]}, out=str(tmp_path / 'b'))
    with pytest.raises(ValueError, match='name'):
        make_set({'sounds': [{'name': 'a', 'sound': 'tap/tone'}, {'name': 'a', 'sound': 'tick/wood'}]},
                 out=str(tmp_path / 'c'))
    with pytest.raises(ValueError, match='no "sounds"'):
        make_set({}, out=str(tmp_path / 'd'))
    with pytest.raises(ValueError, match='unknown set'):
        make_set({'set': 'opera'}, out=str(tmp_path / 'e'))


# ---- sounds

def test_sounds_filters():
    every = sounds()['sounds']
    assert len(every) == len(registry.ids())
    s = sounds(search='notification')
    assert 0 < len(s['sounds']) <= 20 and s['sounds'][0]['id'].startswith('notify/')
    assert s['summary'].splitlines()[0] == f"{len(s['sounds'])} sounds for 'notification'"
    assert all('ambience' in r['domains'] for r in sounds(domain='ambience')['sounds'])
    beds = sounds(kind='bed')['sounds']
    assert beds and all(r['kind'] == 'bed' for r in beds)
    steps = sounds(family='step')
    assert steps['sounds'] and all(r['id'].startswith('step/') for r in steps['sounds'])
    assert steps['summary'].splitlines()[0] == f"{len(steps['sounds'])} sounds in 1 families"
    assert len(steps['summary'].splitlines()) == len(steps['sounds']) + 1


# ---- the command line

def test_cli_make_sounds_and_pack(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv('EARSHOT_DEBUG', raising=False)
    p = str(tmp_path / 'c.wav')
    assert cli.main(['make', 'a soft success chime', '-o', p]) == 0
    out = capsys.readouterr().out
    assert out.startswith('c.wav: ') and out.count('c.wav: ') == 1 and os.path.isfile(p)
    assert cli.main(['sounds', '--family', 'step']) == 0
    out = capsys.readouterr().out
    assert out.count(' sounds in ') == 1 and out.splitlines()[0].endswith('in 1 families')
    q = str(tmp_path / 't.wav')
    assert cli.main(['make', '--sound', 'tick/wood', '-o', q]) == 0
    capsys.readouterr()
    sp = str(tmp_path / 'sp' / 'sprite.wav')
    assert cli.main(['pack', p, q, '-o', sp]) == 0
    out = capsys.readouterr().out
    assert out.count('2 sounds in ') == 1 and os.path.isfile(sp)
    assert os.path.isfile(str(tmp_path / 'sp' / 'sprite.json'))


def test_cli_bad_request_is_one_line_and_exit_1(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv('EARSHOT_DEBUG', raising=False)
    assert cli.main(['make', 'a frobnicating zorp', '-o', str(tmp_path / 'x.wav')]) == 1
    cap = capsys.readouterr()
    lines = cap.err.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith('earshot make: error: ValueError: no sound in the catalogue')
    assert cap.out == '' and not os.path.exists(tmp_path / 'x.wav')
