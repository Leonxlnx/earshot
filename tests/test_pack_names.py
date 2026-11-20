"""Names must not silently truncate an audio sprite's inputs."""
import json

import numpy as np
import pytest
import soundfile as sf

from earshot.pack import pack
from earshot.synth import master


@pytest.fixture
def inputs(tmp_path):
    paths = []
    for name, frequency in [('first', 440), ('second', 880)]:
        path = tmp_path / f'{name}.wav'
        t = np.arange(4800) / master.SR
        sf.write(path, .2 * np.sin(2 * np.pi * frequency * t), master.SR)
        paths.append(str(path))
    return paths


@pytest.mark.parametrize('names', [['only'], ['a', 'b', 'c']])
@pytest.mark.parametrize('source', ['paths', 'folder', 'arrays'])
def test_mismatched_names_leave_existing_outputs_untouched(tmp_path, inputs, names, source):
    out = tmp_path / 'sprite.wav'
    mapping = tmp_path / 'sprite.json'
    out.write_bytes(b'existing audio')
    mapping.write_bytes(b'existing map')
    files = str(tmp_path) if source == 'folder' else inputs
    kwargs = {'ys': [np.zeros(4800), np.ones(4800) * .1]} if source == 'arrays' else {}
    with pytest.raises(ValueError, match='one name per audio file'):
        pack(files, str(out), names=names, **kwargs)
    assert out.read_bytes() == b'existing audio'
    assert mapping.read_bytes() == b'existing map'


@pytest.mark.parametrize('names', [['only'], ['a', 'b', 'c'], iter(())])
def test_count_checked_before_loading_missing_inputs(tmp_path, names):
    out = tmp_path / 'sprite.wav'
    with pytest.raises(ValueError, match='one name per audio file'):
        pack(['missing-a.wav', 'missing-b.wav'], str(out), names=names)
    assert not out.exists()
    assert not out.with_suffix('.json').exists()


@pytest.mark.parametrize('kind', ['list', 'tuple', 'generator', 'default', 'empty'])
def test_matching_and_default_names_keep_all_audio(tmp_path, inputs, kind):
    names = {'list': ['first', 'second'], 'tuple': ('first', 'second'),
             'generator': iter(['first', 'second']), 'default': None, 'empty': []}[kind]
    out = tmp_path / 'sprite.wav'
    result = pack(inputs, str(out), names=names, gap=.05)
    assert list(result['sprite']) == ['first', 'second']
    assert result['sprite'] == {'first': [0, 100], 'second': [150, 100]}
    assert result['summary'].startswith('2 sounds in ')
    assert sf.info(out).frames == 12000
    doc = json.loads(out.with_suffix('.json').read_text())
    assert doc['sprite'] == result['sprite']
    control = tmp_path / 'control.wav'
    pack(inputs, str(control), gap=.05)
    assert out.read_bytes() == control.read_bytes()


def test_duplicate_names_still_rejected(tmp_path, inputs):
    with pytest.raises(ValueError, match='share a name'):
        pack(inputs, str(tmp_path / 'sprite.wav'), names=['same', 'same'])
