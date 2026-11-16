"""Expanded set names must not overwrite another take or duplicate manifest identities."""
import json

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
