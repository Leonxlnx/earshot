"""Timeline cache identity follows parsed cues, not a second file observation."""
import json
import numpy as np
import soundfile as sf
import pytest
import earshot.timeline as tl              # the module itself: `from earshot import timeline` asks for the listen extra
from earshot import config

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    p = tmp_path / 'silent.wav'
    sf.write(p, np.zeros(96000), 48000)
    cues = tmp_path / 'cues.json'
    cues.write_text('[0.5]')
    return (str(p), cues)

def times(r):
    return [v['t'] for v in r['sync']['cues']['items']]

def test_whitespace_path_edit(setup):
    p, cues = setup
    arg = ' ' + str(cues) + ' '
    assert times(tl.timeline(p, cues=arg)) == [0.5]
    cues.write_text('[1.5]')
    r = tl.timeline(p, cues=arg)
    assert not r['cached'] and times(r) == [1.5]
    assert tl.timeline(p, cues=arg)['cached']

def test_snapshot_read_before_change(setup, monkeypatch):
    p, cues = setup
    original = tl.read_cues

    def changing(arg):
        result = original(arg)
        cues.write_text('[1.5]')
        return result
    with monkeypatch.context() as m:
        m.setattr(tl, 'read_cues', changing)
        assert times(tl.timeline(p, cues=str(cues))) == [0.5]
    r = tl.timeline(p, cues=str(cues))
    assert not r['cached'] and times(r) == [1.5]

def test_equivalent_inputs(setup):
    p, cues = setup
    first = tl.timeline(p, cues=str(cues))
    for value in ([0.5], '[0.5]', '0.5', {'events': [{'time': 0.5}]}):
        r = tl.timeline(p, cues=value)
        assert r['cached'] and times(r) == times(first)

def test_fps_is_part_of_effective_input(setup):
    p, _ = setup
    first = tl.timeline(p, cues={'fps': 30, 'events': [{'f': 30}]})
    r = tl.timeline(p, cues={'fps': 60, 'events': [{'f': 30}]})
    assert not r['cached'] and times(first) == [1] and (times(r) == [0.5])

def test_invalid_file_never_hidden_by_cache(setup):
    p, cues = setup
    tl.timeline(p, cues=str(cues))
    cues.write_text('broken')
    with pytest.raises(json.JSONDecodeError):
        tl.timeline(p, cues=str(cues))

def test_same_time_different_fps_preserves_tolerance(setup):
    p, _ = setup
    first = tl.timeline(p, cues={'fps': 30, 'events': [{'t': 1}]})
    second = tl.timeline(p, cues={'fps': 60, 'events': [{'t': 1}]})
    assert times(first) == times(second) == [1]
    assert not second['cached']
    assert first['sync']['cues']['stats']['frame_ms'] != second['sync']['cues']['stats']['frame_ms']
