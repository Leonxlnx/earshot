"""Expanded set names must not overwrite another take or duplicate manifest identities."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from earshot.make import make_set
from earshot.sounds import registry


@pytest.fixture
def tone_catalogue(monkeypatch):
    # A controlled catalogue; the real renderer, mastering, audit and file writer still run.
    monkeypatch.setattr(registry, '_designs', {})
    monkeypatch.setattr(registry, '_loaded', True)

    @registry.design('test/tone', kind='note', says=('a test tone', 'a short note'), dur=0.2, pitch='A4')
    def tone(c, r):
        t = np.arange(round(c.seconds() * 48000)) / 48000
        return np.sin(2 * np.pi * 440 * t) * np.sin(np.linspace(0, np.pi, len(t))) ** 2


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('fmt', ['wav', ['wav', 'flac']])
@pytest.mark.parametrize('count', [{'n': 2}, {'want': 'two variations of a note'}])
def test_expanded_take_names_are_rejected_before_writes(tmp_path, tone_catalogue, reverse, fmt, count):
    items = [{'name': 'tap', 'sound': 'test/tone', 'dur': 0.2, **count},
             {'name': 'tap_1', 'sound': 'test/tone', 'dur': 0.4}]
    if reverse:
        items.reverse()
    # Existing output must survive, including its manifest and the colliding filename.
    sentinel = tmp_path / 'tap_1.wav'
    sentinel.write_bytes(b'existing take')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"existing": true}', encoding='utf-8')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='take name'):
        make_set({'sounds': items, 'format': fmt}, out=str(tmp_path))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_duplicate_manifest_basenames_are_rejected(tmp_path, tone_catalogue):
    items = [{'name': 'first/tap', 'sound': 'test/tone'},
             {'name': 'second/tap', 'sound': 'test/tone'}]
    with pytest.raises(ValueError, match='take name'):
        make_set({'sounds': items}, out=str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fmt', ['wav', ['wav', 'flac']])
def test_distinct_expanded_names_keep_files_and_metadata(tmp_path, tone_catalogue, fmt):
    result = make_set({'sounds': [{'name': 'tap', 'sound': 'test/tone', 'n': 2, 'dur': 0.2},
                                 {'name': 'other', 'sound': 'test/tone', 'dur': 0.4}],
                       'format': fmt}, out=str(tmp_path))
    assert [r['name'] for r in result['sounds']] == ['tap_1', 'tap_2', 'other']
    assert len(set(result['files'])) == len(result['files'])
    assert json.loads((tmp_path / 'manifest.json').read_text())['sounds'] == result['sounds']
    for row in result['sounds']:
        for path in row['files']:
            data, sr = sf.read(tmp_path / path)
            assert len(data) / sr == pytest.approx(row['dur_s'], abs=0.002)
            assert np.isfinite(data).all() and np.max(np.abs(data)) > 0


def test_explicit_take_count_overrides_words(tmp_path, tone_catalogue):
    result = make_set({'sounds': [{'name': 'tap', 'sound': 'test/tone', 'want': 'two variations of a note', 'n': 1},
                                 {'name': 'tap_1', 'sound': 'test/tone'}]}, out=str(tmp_path))
    assert [r['name'] for r in result['sounds']] == ['tap', 'tap_1']
    assert len(set(result['files'])) == 2


def test_loop_format_replacement_is_unchanged(tmp_path, tone_catalogue):
    d = registry.get('test/tone')
    registry.design('test/loop', kind='note', says=('a looping tone', 'a sustained note'), dur=0.2,
                    pitch='A4', loop=True)(d.fn)
    result = make_set({'sounds': [{'name': 'loop', 'sound': 'test/loop', 'n': 2}],
                       'format': ['mp3', 'flac']}, out=str(tmp_path))
    assert [r['files'] for r in result['sounds']] == [['loop_1.flac'], ['loop_2.flac']]
    assert all(r['loop'] for r in result['sounds'])
    assert sorted(p.name for p in tmp_path.iterdir()) == ['loop_1.flac', 'loop_2.flac', 'manifest.json']


@pytest.mark.parametrize('sprite,fmt', [
    ('tap.wav', 'wav'), ('tap', 'wav'), ('tap.flac', ['wav', 'flac']),
    ('nested/tap.wav', 'wav'), ('manifest.wav', 'wav'), ('manifest', ['wav', 'flac']),
])
@pytest.mark.parametrize('argument', [False, True])
def test_sprite_destinations_cannot_overwrite_set_files(tmp_path, tone_catalogue, sprite, fmt, argument):
    items = [{'name': 'tap', 'sound': 'test/tone'}, {'name': 'other', 'sound': 'test/tone'}]
    spec = {'sounds': items, 'format': fmt}
    kwargs = {'sprite': sprite} if argument else {}
    if not argument:
        spec['sprite'] = sprite
    (tmp_path / 'tap.wav').write_bytes(b'existing take')
    (tmp_path / 'manifest.json').write_bytes(b'existing manifest')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='sprite output'):
        make_set(spec, out=str(tmp_path), **kwargs)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_sprite_checks_the_loop_replacement_format(tmp_path, tone_catalogue):
    registry.design('test/loop', kind='note', says=('a looping tone', 'a sustained note'), dur=0.2,
                    pitch='A4', loop=True)(registry.get('test/tone').fn)
    with pytest.raises(ValueError, match='sprite output'):
        make_set({'sounds': [{'name': 'tap', 'sound': 'test/loop'}], 'format': 'mp3',
                  'sprite': 'tap.flac'}, out=str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('sprite,fmt', [('sprite', ['wav', 'flac']), ('tap.flac', 'wav')])
def test_nonconflicting_sprite_keeps_takes_manifest_and_map(tmp_path, tone_catalogue, sprite, fmt):
    result = make_set({'sounds': [{'name': 'tap', 'sound': 'test/tone', 'dur': 0.2},
                                 {'name': 'other', 'sound': 'test/tone', 'dur': 0.4}],
                       'format': fmt, 'sprite': sprite}, out=str(tmp_path))
    assert json.loads((tmp_path / 'manifest.json').read_text())['sounds'] == result['sounds']
    for row in result['sounds']:
        assert sf.info(tmp_path / row['file']).duration == pytest.approx(row['dur_s'], abs=0.002)
    sprite_path = result['sprite']
    doc = json.loads(Path(sprite_path).with_suffix('.json').read_text())
    assert set(doc['sprite']) == {'tap', 'other'}
    assert doc['src'] == [Path(p).name for p in result['sprites']]
    for p in result['sprites']:
        assert sf.info(p).duration > 0.8


def test_sprite_override_takes_precedence_over_spec(tmp_path, tone_catalogue):
    result = make_set({'sounds': [{'name': 'tap', 'sound': 'test/tone'}], 'sprite': 'manifest.wav'},
                      out=str(tmp_path), sprite='safe.wav')
    assert result['sprite'] == str(tmp_path / 'safe.wav')
    assert 'sounds' in json.loads((tmp_path / 'manifest.json').read_text())


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='mp3 writing needs ffmpeg')
def test_loop_rendered_once_reserves_its_requested_format(tmp_path, tone_catalogue):
    registry.design('test/loop', kind='note', says=('a looping tone', 'a sustained note'), dur=0.2,
                    pitch='A4', loop=True)(registry.get('test/tone').fn)
    result = make_set({'sounds': [{'name': 'tap', 'sound': 'test/loop', 'want': 'one-shot'}],
                       'format': 'mp3', 'sprite': 'tap.flac'}, out=str(tmp_path))
    assert result['sounds'][0]['loop'] is False
    assert result['sounds'][0]['files'] == ['tap.mp3']
    assert (tmp_path / 'tap.mp3').is_file() and (tmp_path / 'tap.flac').is_file()
