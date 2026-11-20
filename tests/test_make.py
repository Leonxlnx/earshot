"""earshot.make and earshot.pack: a request in words becomes a file (format by extension, takes, the one-line summary,
`also`, `unknown`, the error for a request nothing answers, explicit arguments over words, a theme, a loop once),
a whole set (files, manifest, levels by kind, a sprite whose map covers every sound), specs as dicts and as JSON
files with a byte order mark, the catalogue listing, the command line, and packing a set's folder. Model-free; every
file goes to tmp_path."""
import json
import os
import re
import shutil

import numpy as np
import pytest
import soundfile as sf

from earshot import cli
from earshot.forge import dsp
from earshot.make import WEIGHT, make, make_set, sounds
from earshot.pack import pack
from earshot.sounds import registry, sets


def _read(p):
    y, sr = sf.read(p, always_2d=True)
    return y, sr


# ---- make

def test_the_request_or_the_count_decides_between_one_event_and_the_action(tmp_path):
    """"knocking on a door" plays the knock pattern, "a single knock on a door" one knock, events=2 two knocks; the
    summary and the take's meta say what was played"""
    many = make('knocking on a wooden door', str(tmp_path / 'k.wav'))
    one = make('a single knock on a wooden door', str(tmp_path / 'k1.wav'))
    two = make('a single knock on a wooden door', str(tmp_path / 'k2.wav'), events=2)
    assert many['sound'] == one['sound'] == 'door/knock'
    p = many['takes'][0]['performance']
    assert p['events'] >= 3 and 'played:' in many['summary']
    assert 'performance' not in one['takes'][0] and 'played: one event' in one['summary']
    assert two['takes'][0]['performance']['events'] == 2
    assert many['takes'][0]['dur_s'] > one['takes'][0]['dur_s'] + 0.4
    assert abs(many['takes'][0]['peak_dbfs'] + 1.0) < 0.3


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
    with pytest.raises(ValueError, match='say what sound'):
        make(None, str(tmp_path / 'x.wav'))
    with pytest.raises(ValueError, match='unknown sound'):
        make(sound='no/such', out=str(tmp_path / 'x.wav'))


def test_explicit_arguments_override_the_words(tmp_path):
    words = 'a success chime on a vibraphone in D, 400 ms'
    base = make(words, str(tmp_path / 'a.wav'))['takes'][0]
    assert base['pitch'].startswith('D') and base['voice'] == 'vibraphone' and abs(base['dur_s'] - 0.4) <= 0.03
    res = make(words, str(tmp_path / 'b.wav'), sound=base['id'], pitch='A5', voice='kalimba', dur=0.3)
    t = res['takes'][0]
    assert t['pitch'] == 'A5' and t['voice'] == 'kalimba'
    assert abs(t['dur_s'] - 0.3) <= 0.03


def test_make_applies_a_theme(tmp_path):
    t = make(out=str(tmp_path / 'g.wav'), sound='success/rise', theme='glass')['takes'][0]
    assert t['voice'] == 'glockenspiel' and t['pitch'] == 'D4'
    t = make(out=str(tmp_path / 'w.wav'), sound='success/rise', theme={'base': 'wood', 'key': 'A'})['takes'][0]
    assert t['voice'] == 'marimba' and t['pitch'] == 'A4'


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
    assert 'lake' in ' '.join(res['why']).split() and 'lake' not in res['unknown']
    s = make_set({'sounds': [{'name': 'amb', 'want': want + ', one-shot', 'dur': 2}]}, out=str(tmp_path / 's'))
    lines = s['summary'].splitlines()
    why = lines[1].rsplit('[', 1)[1].rstrip(']').split(', ')
    assert 'lake' in ' '.join(why).split() and 'lak' not in why
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
    assert s.startswith('p.wav: open/drawer')                      # "menu open" is a job of open/drawer: the pick
    res = make('crickets at night', str(tmp_path / 'n.wav'), dur=2, loop=False)
    assert len(res['also']) == 3 and 'also: ' + ', '.join(res['also']) in res['summary']
    assert sum(1 for ln in res['summary'].splitlines() if 'also' in ln) == 1


def test_a_struck_sound_says_it_ignores_dur(tmp_path):
    res = make(out=str(tmp_path / 'h.wav'), sound='hit/metal', dur=2.5)
    assert res['takes'][0]['dur_s'] < 1.5 and 'not the 2.5 s asked: a single struck sound rings' in res['summary']
    res = make(out=str(tmp_path / 'f.wav'), sound='success/rise', dur=1.0)
    assert 'dur' not in res['takes'][0].get('ignored', [])


def test_a_figure_says_how_short_it_can_be_and_that_it_is_not_held(tmp_path):
    """a figure asked shorter than its notes can be hurried says its shortest length; asked longer than it rings, it
    says it rings out sooner, in plain words instead of `not used by this sound: dur`"""
    s = make(out=str(tmp_path / 'a.wav'), sound='success/rise', dur=0.1)['summary']
    assert 'as short as this sound can be: 0.26 s (asked for 0.1 s)' in s and 'not used' not in s
    s = make(out=str(tmp_path / 'b.wav'), sound='success/rise', dur=3.0)['summary']
    assert 'shorter than the 3 s asked for: a struck note is not held' in s and 'not used' not in s
    res = make(out=str(tmp_path / 'c.wav'), sound='success/rise', dur=0.4)
    assert 0.37 <= res['takes'][0]['dur_s'] <= 0.43 and 'not used' not in res['summary']


def test_help_names_the_codecs_and_what_dur_does(capsys):
    with pytest.raises(SystemExit):
        cli.main(['make', '--help'])
    text = ' '.join(capsys.readouterr().out.split())
    for bit in ('ogg = Vorbis', 'webm = Opus', 'm4a = AAC', 'webm,mp3', 'a figure of notes', 'hold of a pad',
                'length of an ambience', 'whole bars', 'ignores --dur'):
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


def test_make_set_levels_follow_the_weights_when_a_loudness_is_given(tmp_path):
    spec = {'lufs': -16, 'sounds': [{'name': 'tap', 'sound': 'tap/tone'}, {'name': 'success', 'sound': 'success/rise'}]}
    res = make_set(spec, out=str(tmp_path / 's'))
    lv = {r['name']: r['lufs_max'] for r in res['sounds']}
    k = {r['name']: registry.get(r['sound']).kind for r in res['sounds']}
    assert (k['tap'], k['success']) == ('tap', 'chime')
    assert lv['success'] - lv['tap'] == pytest.approx(WEIGHT['chime'] - WEIGHT['tap'], abs=1.0)
    assert lv['success'] == pytest.approx(-16.0 + WEIGHT['chime'], abs=0.6)
    _check_manifest(str(tmp_path / 's'), 2)


def _loudest50(path):
    from earshot.synth import master
    return master.loudest(_read(path)[0])


def test_a_set_of_one_shots_is_full_and_balanced(tmp_path):
    """levelled to a common loudness, an interface set came out 6-8 dB under the same takes made alone (taps at -26
    LUFS) and was heard as far and thin. Without a `lufs` a set's one-shots are made as they are alone (the quietest
    at full peak) and only turned down to within BALANCE dB of the quietest by their loudest 50 ms; a ready-made
    set's quiet names (hover) sit lowest; the learned roles play the recordings chosen for them"""
    from earshot.make import BALANCE, QUIET
    out = str(tmp_path / 'ui')
    res = make_set('ui', out=out)
    lv = {r['name']: _loudest50(os.path.join(out, r['file'])) for r in res['sounds']}
    pk = {r['name']: _read(os.path.join(out, r['file']))[0] for r in res['sounds']}
    assert max(lv.values()) - min(lv.values()) <= BALANCE + QUIET + 0.2, lv
    assert min(lv, key=lv.get) == 'hover'
    loudest_peak = max(20 * np.log10(np.abs(y).max()) for y in pk.values())
    assert loudest_peak > -2.0                                     # full: the quietest one-shots stay at full peak
    assert all(v > -20.0 for v in lv.values()), lv                 # nothing far and thin
    from earshot.sounds import learned
    tap = next(r for r in res['sounds'] if r['name'] == 'tap')
    src = learned.index()[tap['sound']]['members'][tap['member']['n'] - 1]['credit']['title']
    assert src.startswith('Computer Keyboard - single key')          # the fullest mechanical key
    assert tap['member']['recording'] == f"recording {tap['member']['n']} of {tap['member']['of']}"
    assert {r['name']: r['sound'] for r in res['sounds']}['hover'] == 'hover/fingertip'


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


def test_a_set_names_two_recordings_of_one_design_without_a_warning(tmp_path):
    """make --set ui printed "same design as toggle_on" under toggle_off, an intended pair of two recordings"""
    spec = {'sounds': [{'name': 'on', 'sound': 'switch/lamp'}, {'name': 'off', 'sound': 'switch/lamp'}]}
    res = make_set(spec, out=str(tmp_path / 's'))
    lines = res['summary'].splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith('off.wav'))
    assert lines[k + 1].startswith('  another recording of switch/lamp than on: ')
    assert [r['sound'] for r in res['sounds']] == ['switch/lamp', 'switch/lamp']
    assert 'same design' not in res['summary']


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
    assert cli.main(['make', 'a click', '--sound', 'no/such', '-o', str(tmp_path / 'x.wav')]) == 1
    cap = capsys.readouterr()
    lines = cap.err.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith('earshot make: error: ')
    assert cap.out == '' and not os.path.exists(tmp_path / 'x.wav')


def test_a_request_nothing_knows_gets_the_nearest_and_says_so(tmp_path):
    """no request comes back without a sound: words nothing knows get a nearest design, said to be only that, with
    alternatives on the same line"""
    res = make('a frobnicating zorp', out=str(tmp_path / 'x.wav'))
    assert os.path.isfile(tmp_path / 'x.wav')
    line = next(x for x in res['summary'].splitlines() if x.startswith('also (the pick is a guess: '))
    assert f"earshot makes no frobnicating zorp; {res['sound']} is only the nearest sound it has" in line
    assert 'Name one with sound=<id>' in line and res['also']
    assert 'or let earshot draft, hear and tune it: earshot build "a frobnicating zorp"' in res['summary']


@pytest.mark.parametrize('text,start', [
    ('a lightsaber', 'earshot makes no lightsaber; laser/beam is only the nearest'),
    ('a zorp squealing', 'earshot makes no zorp squealing; tyres/screech is only the nearest'),
    ('a blorb fizzing', 'earshot makes no blorb fizzing; spray_can/spray is only the nearest'),
    ('rain on a tarp', 'no rain on a tarp: rain/tent is the nearest surface'),
    ('a zorp door rolling up', 'earshot makes no zorp door'),
])
def test_a_missing_thing_and_a_missing_surface_are_said_with_alternatives(text, start):
    """the thing or what it does is unknown: the nearest kind of sound, said to be only that; the sound exists but
    not on that surface: the nearest surface, with the sound's other surfaces as alternatives"""
    from earshot import brief
    from earshot.make import doubt, guess
    b = brief.read(text)
    cands = brief.rank(b, top=8)
    assert doubt(b, cands).startswith(start), doubt(b, cands)
    line = guess(b, cands)
    assert line.endswith('Name one with sound=<id>') and start in line
    if start.startswith('no '):
        fam = cands[0]['id'].split('/')[0]
        assert line.startswith(start + '; also: ' + fam + '/'), line


@pytest.mark.parametrize('text', ['a trumpet', 'a zorp machine', 'a frobnicator', 'a blorb fizzing', 'a zorp squealing'])
def test_the_starts_for_a_missing_thing_are_of_other_families(text):
    """the three alternatives offered for a thing no design is are three starting points of three other families,
    not three members of one"""
    from earshot import brief
    from earshot.make import doubt, starts
    b = brief.read(text)
    cands = brief.rank(b, top=8)
    assert doubt(b, cands).startswith('earshot makes no ')
    fams = [s.split('/')[0] for s in starts(b, cands)]
    assert len(fams) == 3 and len(set(fams)) == 3 and cands[0]['id'].split('/')[0] not in fams, fams


def test_the_starts_of_a_missing_thing_are_of_its_kind():
    """a thing no word found offers three kinds of sound of the world, not three thuds; a thing the cookbook
    sketches offers the other parts its sketch is built on first"""
    from earshot import brief, recipe
    from earshot.make import doubt, starts
    from earshot.sounds import lexicon as L
    b = brief.read('a frobnicator')
    cands = brief.rank(b, top=8)
    assert cands[0]['id'] == L.NEAREST_ANY[0] and starts(b, cands) == list(L.STARTS_ANY), starts(b, cands)
    for name, rc in recipe.cookbook().items():
        b = brief.read(rc['things'][0])
        cands = brief.rank(b, top=8)
        if not rc.get('sketch') or not (doubt(b, cands) or '').startswith('earshot makes no '):
            continue
        want, fams = [], {cands[0]['id'].split('/')[0]}
        for s in brief.sketched(b):          # one part of each family, as the starts take them
            if s.split('/')[0] not in fams and 'synth' not in registry.get(s).tags and s.split('/')[0] != 'note':
                fams.add(s.split('/')[0])
                want.append(s)
        got = starts(b, cands)
        assert got[:len(want[:3])] == want[:3], (name, got, want)


def test_a_pan_asked_for_is_said_not_followed():
    from earshot import brief
    from earshot.make import unfollowed
    b = brief.read('a car passing from left to right')
    lines = unfollowed(b, 'pass/car', [], [], {}, set(), [])
    assert any(x.startswith('not followed: "from left to right"; earshot does not pan') for x in lines), lines


@pytest.mark.parametrize('text', ['rain on a tin roof', 'footsteps on snow', 'a ball bouncing on wood',
                                  'a dog barking', 'a heartbeat getting faster', 'a typewriter', 'a slap'])
def test_what_exists_is_answered_without_a_note(text):
    from earshot import brief
    from earshot.make import doubt
    b = brief.read(text)
    assert doubt(b, brief.rank(b, top=8)) is None


TITLED = ('a laptop keyboard key', 'a light switch', 'three knocks on a door', 'footsteps on gravel', 'a cap gun shot')


def _titles():
    """every title an uploader gave a recording of the stores (learned things, drums, beds, instrument notes), but
    those that are plain words of a request or of the catalogue itself ("Light switch")"""
    from earshot.sounds import drums, learned, nature
    from earshot.synth import played
    out = {m['credit']['title'] for ix in (learned.index(), drums.index()) for e in ix.values() for m in e['members']}
    out |= {t['title'] for t in nature.store().values() if t.get('title')}
    out |= {n['credit']['title'] for e in played.index().values() for n in e['notes']}
    ours = ' | '.join(list(TITLED) + [' '.join([r['id']] + list(r['says']) + list(r['uses']))
                                      for r in registry.rows()]).lower()
    return {t.lower() for t in out if len(t) >= 6 and t.lower() not in ours}


def test_no_output_names_a_recording_by_its_title(tmp_path):
    """the lines, meta and manifest say "recording 2 of 3", never the title its uploader gave a recording (a brand, a
    file name): those stay in CREDITS.md and the stores' indexes"""
    titles = _titles()
    assert len(titles) > 50
    out = [make_set('ui', out=str(tmp_path / 'ui'))]
    for i, want in enumerate(TITLED):
        out += [make(want, str(tmp_path / f'{i}.wav'), n=2)]
    text = json.dumps(out, default=str).lower()
    with open(tmp_path / 'ui' / 'manifest.json', encoding='utf-8') as fh:
        text += fh.read().lower()
    assert not [t for t in titles if t in text]
    assert 'recording 1 of ' in text or 'recording 2 of ' in text


def test_an_asked_length_holds_for_an_ambience_and_a_loop_says_its_bars(tmp_path):
    """who asks for 4 seconds of rain gets 4 seconds, a cut that says it is no loop; "a seamless loop" keeps the
    shortest loop and says so; a music loop keeps whole bars and says so; nothing says `not used by this sound`"""
    cut = make('rain on a window, 4 seconds', str(tmp_path / 'r.wav'))
    t = cut['takes'][0]
    assert t['dur_s'] == pytest.approx(4.0, abs=0.001) and t['loop'] is False and t['checks']['ok']
    y, _ = _read(cut['files'][0])
    assert np.abs(y[0]).max() == 0.0 and np.abs(y[-1]).max() < 1e-4
    assert 'a cut of 4 s, not a loop: the shortest seamless loop of this sound is 8.00 s' in cut['summary']
    seam = make('a seamless loop of rain on a window, 4 seconds', str(tmp_path / 's.wav'))
    assert seam['takes'][0]['loop'] is True and seam['takes'][0]['dur_s'] == pytest.approx(8.0, abs=0.001)
    assert 'a seamless loop of 8.00 s, the shortest this sound has; 4 s was asked' in seam['summary']
    bars = make('background music, 8 seconds', str(tmp_path / 'm.wav'))
    assert 'a music loop is whole bars, and these are the nearest to the 8 s asked' in bars['summary']
    fast = make('a fast whoosh', str(tmp_path / 'w.wav'))
    for res in (cut, seam, bars, fast):
        assert 'not used by this sound' not in res['summary'] and 'not followed' not in res['summary']


def test_a_counted_action_in_a_row_is_one_file_and_a_count_of_things_is_many(tmp_path):
    """"four knocks on the table" picks the recorded knuckles on a table (knock/table) and plays one
    file of four knocks; "three claps" one file of three; "five coin pickups" stays five files; "a single" one event"""
    knocks = make('four knocks on the table', str(tmp_path / 'k.wav'))
    p = knocks['takes'][0]['performance']
    assert knocks['sound'] == 'knock/table' and len(knocks['files']) == 1
    rec = p['recordings']
    assert p['pattern'] == 'knock' and p['events'] == 4 and all(a != b for a, b in zip(rec, rec[1:]))
    assert 'played: 4 events as a knock' in knocks['summary'] and knocks['takes'][0]['checks']['ok']
    claps = make('three claps', str(tmp_path / 'c.wav'))
    assert len(claps['files']) == 1 and claps['takes'][0]['performance']['events'] == 3
    count = make('a countdown 3 2 1', str(tmp_path / 'n.wav'))
    assert count['takes'][0]['performance']['pattern'] == 'count' and count['takes'][0]['dur_s'] > 2.0
    coins = make('five coin pickups', str(tmp_path / 'p.wav'))
    assert len(coins['files']) == 5 and 'not understood' not in coins['summary']     # "five" is the count
    one = make('a single knock on the table', str(tmp_path / 'o.wav'))
    assert 'performance' not in one['takes'][0]
    two = make(out=str(tmp_path / 't.wav'), sound='tick/wood', events=2)
    assert two['takes'][0]['performance']['events'] == 2 and two['takes'][0]['performance']['pattern'] == 'row'


def test_a_length_asked_with_a_count_is_the_files_and_the_events_spread_over_it(tmp_path):
    """"three claps, 3 seconds": one file of 3.00 s with three claps across it; "footsteps on gravel, 5 seconds": the
    count from the walking pace; a countdown keeps one a second and says when the length does not fit; knocks never
    further apart than 1.5 s (the line says what was played); the file ends faded, after the last event sounded"""
    claps = make('three claps, 3 seconds', str(tmp_path / 'c.wav'))
    t = claps['takes'][0]
    assert t['dur_s'] == pytest.approx(3.0, abs=0.001) and t['performance']['events'] == 3
    assert 0.5 < 1 / t['performance']['pace'] < 1.5 and '(3.00 s' in claps['summary']
    y, _ = _read(claps['files'][0])
    env = np.abs(y).max(1)
    assert np.abs(y[-1]).max() < 1e-4 and env[-dsp.n_of(0.05):].max() < 0.1 * env.max()
    assert env[dsp.n_of(2.0):].max() > 0.2 * env.max()                       # the third clap sounds after 2 s
    steps = make('footsteps on gravel, 5 seconds', str(tmp_path / 's.wav'))
    t = steps['takes'][0]
    assert t['dur_s'] == pytest.approx(5.0, abs=0.001) and 7 <= t['performance']['events'] <= 10
    beeps = make('a countdown of three beeps over 3 seconds', str(tmp_path / 'b.wav'))
    t = beeps['takes'][0]
    assert t['performance']['pattern'] == 'count' and t['dur_s'] == pytest.approx(3.0, abs=0.001)
    assert t['performance']['pace'] == pytest.approx(1.0) and 'not the' not in beeps['summary']
    short = make('a countdown 3 2 1, 1 second', str(tmp_path / 'n.wav'))
    assert short['takes'][0]['performance']['events'] == 3 and short['takes'][0]['dur_s'] > 2.0
    assert 'not the 1 s asked (a countdown keeps one beat a second)' in short['summary']
    far = make('three knocks on the table, 10 seconds', str(tmp_path / 'k.wav'))
    t = far['takes'][0]
    assert t['performance']['events'] == 3 and t['dur_s'] < 4.0
    assert '3 knocks 1.5 s apart' in far['summary'] and 'not the 10 s asked' in far['summary']


def test_the_action_form_of_a_row_word_is_the_action(tmp_path):
    """"clapping, 4 seconds" is claps at a clapping pace over 4.00 s (the length is the row's, so it does not steer
    the pick to a longer single clap); "clapping" alone six to nine claps; "knocking" is the recorded door knock,
    spread over a length asked; a comparison or the event a sound plays for stays one event; applause is no row"""
    claps = make('clapping, 4 seconds', str(tmp_path / 'c.wav'))
    t = claps['takes'][0]
    assert claps['sound'] == 'clap/hands' and t['dur_s'] == pytest.approx(4.0, abs=0.001)
    assert t['performance']['pattern'] == 'row' and 10 <= t['performance']['events'] <= 14 and t['checks']['ok']
    t = make('clapping', str(tmp_path / 'd.wav'))['takes'][0]
    assert 6 <= t['performance']['events'] <= 9 and 1.8 < t['dur_s'] < 3.6
    knock = make('knocking, 4 seconds', str(tmp_path / 'k.wav'))
    t = knock['takes'][0]
    assert knock['sound'] == 'door/knock' and t['performance']['pattern'] == 'knock'
    assert t['dur_s'] == pytest.approx(4.0, abs=0.001)
    beeps = make('beeping, 3 seconds', str(tmp_path / 'b.wav'))
    assert beeps['sound'] == 'countdown/beep' and beeps['takes'][0]['performance']['pattern'] == 'row'
    for text in ('applause', 'a single clear ding like tapping a little bell', 'a clap'):
        assert 'performance' not in make(text, str(tmp_path / 'o.wav'))['takes'][0]


def test_a_dog_barks_once_or_in_a_run(tmp_path):
    """"a dog bark" is one bark; "a dog barking" a recorded run of barks; "three barks" three, whatever plays them"""
    one = make('a dog bark', str(tmp_path / 'a.wav'))
    assert one['sound'] == 'dog/bark' and 'performance' not in one['takes'][0] and one['takes'][0]['checks']['ok']
    run = make('a dog barking', str(tmp_path / 'b.wav'))['takes'][0]
    assert run['performance']['events'] >= 2
    three = make('three barks', str(tmp_path / 'c.wav'))
    assert three['sound'] == 'dog/bark' and three['takes'][0]['performance']['events'] == 3


def test_a_length_asked_of_a_scattered_or_shaken_action_holds(tmp_path):
    """the pattern reshaped over the length, or the stretch of a recording of the whole action, said plainly"""
    ice = make('ice cubes dropped in a glass, 3 seconds', str(tmp_path / 'i.wav'))['takes'][0]
    assert ice['performance']['pattern'] == 'scatter' and ice['dur_s'] == pytest.approx(3.0, abs=0.001)
    keys = make('keys jingling, 1 second', str(tmp_path / 'k.wav'))
    assert keys['takes'][0]['dur_s'] == pytest.approx(1.0, abs=0.001) and 'busiest stretch' in keys['summary']
    coins = make('coins dropping, 3 seconds', str(tmp_path / 'c.wav'))
    assert coins['takes'][0]['dur_s'] < 2.5 and 'not the 3 s asked' in coins['summary']


def test_a_design_named_inside_its_family_wins_over_the_default():
    """"tape stop" is a downlifter, whose default is fall/air; the request names a tape stop: the recordings run
    down (stop/tape), not the swept tone of downlifter/tape"""
    from earshot.make import choose
    for want in ('tape stop', 'a tape stop effect'):
        b, cands = choose(want, top=4)
        assert cands[0]['id'] == 'stop/tape'
        assert cands[1]['score'] < 0.9 * cands[0]['score']                 # no near tie: not a guess


# ---- the edges of the call: each ends in a result or one plain sentence, before anything is rendered

@pytest.mark.parametrize('kw, said', [
    (dict(want='   '), 'say what sound you want'),
    (dict(want='rain on a window', dur=0), 'length in seconds over 0'),
    (dict(want='rain on a window', dur=-3), 'length in seconds over 0'),
    (dict(want='rain on a window', dur=600), 'at most 120 s'),
    (dict(want='rain on a window', dur=36000), 'at most 120 s'),
    (dict(want='a click', n=0), 'n (takes) is a whole number from 1 to 12'),
    (dict(want='a click', n=1000), 'n (takes) is a whole number from 1 to 12'),
    (dict(want='a click', seed='abc'), 'seed is a whole number'),
    (dict(want='footsteps on wood', events='three'), 'events is a whole number from 1 to 64'),
    (dict(want='footsteps on wood', events=-2), 'events is a whole number'),
    (dict(want='footsteps on wood', events=2.5), 'events is a whole number'),
    (dict(want='footsteps on wood', events=1000), 'events is a whole number'),
    (dict(want='a click', colour='red'), 'unknown option colour'),
    (dict(want='a click', bright='very'), 'bright is a number from -1 to 1'),
])
def test_a_hurried_call_gets_one_plain_sentence(tmp_path, kw, said):
    out = tmp_path / 'x.wav'
    with pytest.raises(ValueError, match=re.escape(said)):
        make(out=str(out), **kw)
    assert not out.exists()


def test_a_seed_given_as_text_is_the_number_and_a_long_request_is_read(tmp_path):
    a = make('a click', str(tmp_path / 'a.wav'), seed='3')
    b = make('a click', str(tmp_path / 'b.wav'), seed=3)
    assert a['takes'][0]['seed'] == b['takes'][0]['seed'] == 3
    long = make(('a soft click for a button, ' * 200)[:5000], str(tmp_path / 'long.wav'))
    assert os.path.exists(long['files'][0])


def test_a_path_with_spaces_and_umlauts_in_a_new_folder_and_over_an_existing_file(tmp_path):
    p = tmp_path / 'neuer Ordner' / 'Tür knallt zu.wav'
    make('a door slam', str(p))
    first = p.stat().st_size
    make('a door slam', str(p), seed=1)
    assert p.exists() and first > 0


def test_a_set_names_the_sets_that_exist_and_checks_its_sounds(tmp_path):
    with pytest.raises(ValueError, match="unknown set 'nope': one of ui, game, arcade, film"):
        make_set('nope', out=str(tmp_path / 's'))
    with pytest.raises(ValueError, match='a set spec is'):
        make_set({'sounds': 'tap'}, out=str(tmp_path / 's'))
    with pytest.raises(ValueError, match='tap: n'):
        make_set({'sounds': [{'name': 'tap', 'want': 'a soft tap', 'n': 0}]}, out=str(tmp_path / 's'))
    with pytest.raises(ValueError, match='at most 120 s'):
        make_set({'sounds': [{'name': 'rain', 'want': 'rain', 'dur': 36000}]}, out=str(tmp_path / 's'))
    with pytest.raises(ValueError, match='tap: say what sound'):
        make_set({'sounds': [{'name': 'tap', 'want': ' '}]}, out=str(tmp_path / 's'))
    assert not (tmp_path / 's').exists()


def test_a_material_the_reader_took_from_another_word_is_not_reported_as_not_followed(tmp_path):
    r = make('a wet coat taken off and dropped over a chair, heavy cloth', str(tmp_path / 'c.wav'))
    assert 'material wood' not in r['summary']


def test_a_length_that_was_kept_is_not_reported_as_not_followed(tmp_path):
    r = make('a short whoosh, 0.3 seconds', str(tmp_path / 'w.wav'))
    assert '"short"' not in r['summary']


def test_a_set_plays_a_thing_done_in_a_row_as_make_does(tmp_path):
    r = make_set({'sounds': [{'name': 'steps', 'want': 'footsteps on wet pavement', 'events': 4},
                             {'name': 'knocks', 'want': 'three knocks on a door, 2 seconds'},
                             {'name': 'rain', 'want': 'rain on a window', 'dur': 6, 'loop': False}]},
                 out=str(tmp_path / 's'))
    by = {s['name']: s for s in r['sounds']}
    assert 'played: 4 events' in r['summary']
    assert by['knocks']['dur_s'] == pytest.approx(2.0, abs=0.01)
    assert by['rain']['loop'] is False
    with pytest.raises(ValueError, match='steps: events is a whole number'):
        make_set({'sounds': [{'name': 'steps', 'want': 'footsteps', 'events': 0}]}, out=str(tmp_path / 't'))


def test_an_interface_press_gets_no_hint_about_events(tmp_path):
    r = make('a tap on a button', str(tmp_path / 't.wav'))
    assert 'played: one event' not in r['summary']
    one = make('a single knock on a wooden door', str(tmp_path / 'k.wav'))
    assert 'played: one event' in one['summary']


@pytest.mark.parametrize('argv, said', [
    (['--dur', '0'], 'dur is a length in seconds over 0'),
    (['--dur', '36000'], 'at most 120 s'),
    (['--n', '0'], 'n (takes) is a whole number'),
    (['--events', '0'], 'events is a whole number'),
])
def test_cli_edges_are_one_line_and_exit_1(tmp_path, capsys, monkeypatch, argv, said):
    monkeypatch.delenv('EARSHOT_DEBUG', raising=False)
    assert cli.main(['make', 'rain on a window', '-o', str(tmp_path / 'x.wav')] + argv) == 1
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and said in lines[0]
    assert cli.main(['make', '--set', 'nope', '-o', str(tmp_path / 's')]) == 1
    assert "unknown set 'nope': one of ui" in capsys.readouterr().err


def test_a_take_says_where_its_peak_and_onset_are(tmp_path):
    res = make('a fast whoosh', out=str(tmp_path / 'w.wav'))
    t = res['takes'][0]
    y, sr = sf.read(t['file'])
    a = np.abs(y).max(axis=1) if y.ndim == 2 else np.abs(y)
    assert 0.0 <= t['onset_s'] <= t['peak_s'] <= t['dur_s']
    assert a[int(round(t['peak_s'] * sr))] == pytest.approx(a.max(), rel=0.5)
    assert f"peak at {t['peak_s']:.2f} s" in res['summary'] and 'events_s' not in t


def test_a_row_says_when_each_event_starts(tmp_path):
    t = make('a soft thump', out=str(tmp_path / 'r.wav'), events=3, dur=0.6)['takes'][0]
    assert len(t['events_s']) == 3 and t['events_s'] == sorted(t['events_s'])
    assert t['events_s'][0] == pytest.approx(t['onset_s'], abs=0.03)


def test_a_row_of_thumps_keeps_its_length_and_rings_out(tmp_path):
    res = make('a soft thump', out=str(tmp_path / 'c.wav'), events=3, dur=0.6)
    t = res['takes'][0]
    assert t['dur_s'] == pytest.approx(0.6, abs=0.01) and t['performance']['events'] == 3
    assert t['checks']['ok'], t['checks']
    assert 'by hand' not in res['summary']


def test_ticks_follow_closer_than_a_hand(tmp_path):
    res = make('a counter ticking up, 1.5 seconds', out=str(tmp_path / 't.wav'), events=12)
    t = res['takes'][0]
    assert t['dur_s'] == pytest.approx(1.5, abs=0.01) and t['performance']['events'] == 12
    assert 'by hand' not in res['summary']
    res = make('three knocks on a door, 0.2 seconds', out=str(tmp_path / 'k.wav'), events=3)
    assert res['takes'][0]['dur_s'] > 0.25          # a hand's knocks keep their limit


def test_a_double_knock_is_two_strikes_by_design(tmp_path):
    for seed in range(3):
        t = make(sound='knock/double', out=str(tmp_path / f'd{seed}.wav'), seed=seed)['takes'][0]
        assert t['checks']['ok'], t['checks']


def test_a_gap_between_events_spaces_the_row(tmp_path):
    t = make('a soft thump, 200 ms apart', out=str(tmp_path / 'g.wav'), events=3)['takes'][0]
    assert t['checks']['ok'] and len(t['events_s']) == 3
    assert np.diff(t['events_s']) == pytest.approx([0.2, 0.2], abs=0.04)


def test_a_length_within_a_tenth_of_a_second_says_nothing(tmp_path):
    from earshot.make import _note
    t = {'dur_s': 7.93, 'note': '7.93 s, shorter than the 8 s asked: the longest this sound is made'}
    assert _note(t, 8.0) is None
    t = {'dur_s': 5.0, 'note': '5.00 s, shorter than the 8 s asked: the longest this sound is made'}
    assert _note(t, 8.0) == '5.00 s, shorter than the 8 s asked'
