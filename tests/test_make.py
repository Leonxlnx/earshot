"""earshot.make and earshot.pack: a request in words becomes a file (format by extension, takes, the one-line summary,
`also`, `unknown`, the error for a request nothing answers, explicit arguments over words, a theme, a loop once),
a whole set (files, manifest, levels by kind, a sprite whose map covers every sound), specs as dicts and as JSON
files with a byte order mark, the catalogue listing, the command line, and packing a set's folder. Model-free; every
file goes to tmp_path."""
import json
import os
import shutil

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


def test_make_says_what_the_sound_leaves_out(tmp_path):
    """usability log: "a water drop into a warm chime" gave a logo bell and never mentioned the water drop"""
    res = make('a short calm logo sound, a water drop into a warm chime', str(tmp_path / 'b.wav'))
    assert res['sound'].startswith('logo') and res['not_in_sound'] == ['water drop']
    assert 'not in this sound: water drop' in res['summary'].splitlines()
    res = make('a soft success chime for a finance app', str(tmp_path / 's.wav'))
    assert res['not_in_sound'] == [] and 'not in this sound' not in res['summary']


def test_make_why_and_unknown_never_show_a_stem_or_a_word_twice(tmp_path):
    """usability log: `[lak, water]` next to `not understood: lake`"""
    want = 'gentle water, small waves on a lake shore, calm'
    res = make(want, str(tmp_path / 'w.wav'), dur=2, loop=False)          # once, short: the words are the point
    shown = res['summary'].splitlines()[0].rsplit('[', 1)[1].rstrip(']').split(', ')
    assert shown == res['why'] and 'lak' not in res['why'] and not set(res['why']) & set(res['unknown'])
    assert 'lake' in res['why'] and 'lake' not in res['unknown']
    s = make_set({'sounds': [{'name': 'amb', 'want': want + ', one-shot', 'dur': 2}]}, out=str(tmp_path / 's'))
    lines = s['summary'].splitlines()
    why = lines[1].rsplit('[', 1)[1].rstrip(']').split(', ')
    assert 'lake' in why and 'lak' not in why
    assert not any(ln.strip().startswith('not understood:') and 'lake' in ln for ln in lines)


def test_also_is_filled_from_the_catalogue(tmp_path):
    """usability log: no `also:` when the ranking had a single candidate"""
    from earshot import brief
    from earshot.make import alternatives, guess
    b = brief.read('crickets at night')
    one = [{'id': 'night/crickets', 'score': 5.0, 'why': ['night']}]
    alts = alternatives(b, one, 'night/crickets')
    assert len(alts) == 3 and 'night/crickets' not in alts and len(set(alts)) == 3
    assert all(registry.get(a).loop for a in alts)                     # a loop's alternatives loop
    two = one + [{'id': 'rain/tent', 'score': 1.0, 'why': ['ambience']}]
    assert alternatives(b, two, 'night/crickets')[0] == 'rain/tent'     # the ranked candidates first
    tap = alternatives(brief.read('a tap'), [{'id': 'tap/tone', 'score': 1.0, 'why': ['tap']}], 'tap/tone')
    assert len(tap) == 3 and not any(registry.get(a).loop for a in tap)
    # a guess says so on the alternatives' line, each with what it sounds like (game log: bare ids cost a search)
    vague = [{'id': 'a/x', 'score': 2.0, 'why': ['airy']}]
    line = guess({'listened': False}, vague, ['open/lift', 'chest/open'])
    assert line.startswith('also (the pick is a guess: no job or sound word matched): '
                           'open/lift (two soft notes rising a fifth), chest/open (') and '\n' not in line
    for part in line.split('): ', 1)[1].split('), ')[:-1]:
        assert len(part.split(' (', 1)[1]) <= 44, part                 # about 40 characters each
    res = make('pause menu open', str(tmp_path / 'p.wav'))
    s = res['summary']
    assert 'also (the pick is a guess' in s and 'open/lift (two soft notes rising a fifth)' in s
    res = make('crickets at night', str(tmp_path / 'n.wav'), dur=2, loop=False)
    assert len(res['also']) == 3 and 'also: ' + ', '.join(res['also']) in res['summary']
    assert sum(1 for ln in res['summary'].splitlines() if 'also' in ln) == 1


def test_a_struck_sound_says_it_ignores_dur(tmp_path):
    res = make(out=str(tmp_path / 'h.wav'), sound='hit/metal', dur=2.5)
    assert res['takes'][0]['dur_s'] < 1.5 and 'not used by this sound: dur' in res['summary']
    res = make(out=str(tmp_path / 'f.wav'), sound='success/rise', dur=1.0)
    assert 'dur' not in res['takes'][0].get('ignored', [])


def test_help_names_the_codecs_and_what_dur_does(capsys):
    with pytest.raises(SystemExit):
        cli.main(['make', '--help'])
    text = ' '.join(capsys.readouterr().out.split())
    for bit in ('ogg = Vorbis', 'webm = Opus', 'm4a = AAC', 'webm,mp3', 'a figure of notes', 'hold of a pad',
                'length of a loop', 'ignores --dur'):
        assert bit in text, bit


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


def test_make_set_in_several_formats_with_a_sprite_in_each(tmp_path):
    """usability log: Ogg Vorbis alone does not play in older Safari; a set is written in every format asked for"""
    for i, fmt in enumerate(('flac,wav', ['flac', 'wav'])):
        out = str(tmp_path / f's{i}')
        spec = {'format': fmt, 'sprite': 'sprite.flac',
                'sounds': [{'name': 'tap', 'sound': 'tap/tone'}, {'name': 'ok', 'want': 'a soft success chime'}]}
        res = make_set(spec, out=out)
        man = _check_manifest(out, 2)
        assert man['format'] == 'flac' and man['formats'] == ['flac', 'wav']
        for r in man['sounds']:
            assert r['file'] == r['name'] + '.flac' and r['files'] == [r['name'] + '.flac', r['name'] + '.wav']
            a, b = (_read(os.path.join(out, f))[0] for f in r['files'])
            assert a.shape == b.shape
        assert sorted(os.path.basename(f) for f in res['files']) == ['ok.flac', 'ok.wav', 'tap.flac', 'tap.wav']
        assert res['summary'].splitlines()[0].endswith('(flac, wav), manifest.json')
        assert res['summary'].splitlines()[1].startswith('tap.{flac,wav}: tap/tone')
        with open(os.path.join(out, 'sprite.json'), encoding='utf-8') as fh:
            mp = json.load(fh)
        assert mp['src'] == ['sprite.flac', 'sprite.wav'] and set(mp['sprite']) == {'tap', 'ok'}
        assert all(os.path.isfile(os.path.join(out, s)) for s in mp['src'])
        assert sf.info(os.path.join(out, 'sprite.flac')).frames == sf.info(os.path.join(out, 'sprite.wav')).frames
    out, p = str(tmp_path / 'cli'), tmp_path / 'spec.json'
    p.write_text(json.dumps({'sounds': [{'name': 'coin', 'sound': 'coin/chip'}]}), encoding='utf-8')
    assert cli.main(['make', '--spec', str(p), '--format', 'wav,flac', '-o', out]) == 0
    assert os.path.isfile(os.path.join(out, 'coin.flac')) and os.path.isfile(os.path.join(out, 'coin.wav'))
    with pytest.raises(ValueError, match='unknown audio format aiff'):
        make_set({'format': 'wav,aiff', 'sounds': [{'name': 'a', 'sound': 'tap/tone'}]}, out=str(tmp_path / 'x'))


def test_a_single_loop_in_a_padded_format_is_warned_about(tmp_path):
    from earshot.make import _seam
    assert _seam(['rain'], ['ogg', 'webm', 'wav']) is None and _seam([], ['mp3']) is None
    line = _seam(['rain'], ['webm', 'mp3', 'm4a'])
    assert line.startswith('rain: a loop in mp3 / m4a has a gap at its seam') and 'ogg, webm, wav or flac' in line
    if shutil.which('ffmpeg'):
        res = make(out=str(tmp_path / 'r.mp3'), sound='rain/tent', dur=8)
        assert res['summary'].splitlines()[-1].startswith('r.mp3: a loop in mp3 has a gap at its seam')
        res = make(out=str(tmp_path / 'once.mp3'), sound='rain/tent', dur=3, loop=False)
        assert 'gap at its seam' not in res['summary']


def test_a_set_writes_loops_only_in_formats_that_keep_the_seam(tmp_path):
    """game log: `webm,mp3` wrote mp3 loops and then warned; the user made flac by hand"""
    from earshot.make import _loop_formats
    assert _loop_formats(['webm', 'mp3']) == ['webm', 'flac'] and _loop_formats(['mp3', 'm4a']) == ['flac']
    assert _loop_formats(['flac', 'mp3', 'wav']) == ['flac', 'wav'] and _loop_formats(['ogg']) == ['ogg']
    if not shutil.which('ffmpeg'):
        return
    out = str(tmp_path / 's')
    spec = {'format': 'wav,mp3', 'sprite': 'sprite.wav',
            'sounds': [{'name': 'tap', 'sound': 'tap/tone'}, {'name': 'rain', 'sound': 'rain/tent', 'dur': 8}]}
    res = make_set(spec, out=out)
    rows = {r['name']: r for r in _check_manifest(out, 2)['sounds']}
    assert rows['tap']['files'] == ['tap.wav', 'tap.mp3'] and rows['rain']['files'] == ['rain.wav', 'rain.flac']
    assert not os.path.exists(os.path.join(out, 'rain.mp3')) and os.path.isfile(os.path.join(out, 'rain.flac'))
    lines = res['summary'].splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith('rain.{wav,flac}: rain/tent'))
    assert lines[k + 1] == '  a loop: flac instead of mp3, which leaves a gap at the seam'
    assert 'gap at its seam' not in res['summary']
    with open(os.path.join(out, 'sprite.json'), encoding='utf-8') as fh:
        mp = json.load(fh)
    assert mp['src'] == ['sprite.wav', 'sprite.mp3'] and mp['sprite']['rain'][2] is True    # one map for both


def test_a_set_says_when_two_names_get_one_design(tmp_path):
    """game log: "losing a life" and "game over" both became lose/descend, and nothing said so"""
    spec = {'sounds': [{'name': 'lose_life', 'want': 'losing a life, a sad falling figure'},
                       {'name': 'game_over', 'want': 'game over, gentle and sad'},
                       {'name': 'tap', 'sound': 'tap/tone'}, {'name': 'tap2', 'sound': 'tap/tone'}]}
    res = make_set(spec, out=str(tmp_path / 's'))
    lines = res['summary'].splitlines()
    assert [r['sound'] for r in res['sounds']][:2] == ['lose/descend', 'lose/descend']
    k = next(i for i, ln in enumerate(lines) if ln.startswith('game_over.wav'))
    assert lines[k + 1].startswith('  same design as lose_life: ask for a different one with "sound" (the nearest: ')
    other = lines[k + 1].rsplit(': ', 1)[1].rstrip(')')
    assert other != 'lose/descend' and not _chip(other)
    k = next(i for i, ln in enumerate(lines) if ln.startswith('tap2.wav'))
    assert lines[k + 1].startswith('  same design as tap: ')
    assert sum('same design' in ln for ln in lines) == 2


def _chip(sid):
    from earshot import brief
    return brief.chip(registry.get(sid).row())


def test_help_says_what_a_spec_is(capsys):
    with pytest.raises(SystemExit):
        cli.main(['make', '--help'])
    text = ' '.join(capsys.readouterr().out.split())
    for bit in ('"sounds": [{"name": "tap", "want": "a soft tap"}', 'docs/MAKE.md has every field',
                'rewrites its files and manifest.json'):
        assert bit in text, bit
    assert 'earshot.make.make_set' not in text


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
