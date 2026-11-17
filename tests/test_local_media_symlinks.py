"""FFmpeg filename disambiguation must preserve symlink/.. traversal."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from earshot import audio, video
from earshot.synth import master


@pytest.fixture
def linked(tmp_path, monkeypatch):
    target = tmp_path / 'target'
    nested = target / 'nested'
    nested.mkdir(parents=True)
    try:
        (tmp_path / 'link').symlink_to(nested, target_is_directory=True)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f'directory symlinks unavailable: {e}')
    monkeypatch.chdir(tmp_path)
    return tmp_path, target


def test_audio_symlink_parent_components(linked):
    root, target = linked
    for folder, level in [(root, .1), (target, .3)]:
        sf.write(folder / 'clip.wav', np.full(4800, level, np.float32), 48000, subtype='FLOAT')
    expected = audio.load(str(target / 'clip.wav'), cache=False)
    np.testing.assert_array_equal(audio.load('link/../clip.wav', cache=False), expected)


def test_video_symlink_parent_components(linked):
    root, target = linked
    for folder, color in [(root, 'white'), (target, 'black')]:
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c={color}:s=160x90:r=25:d=1', '-c:v', 'mpeg4', str(folder / 'clip.mp4')],
                       check=True, capture_output=True)
    assert np.mean(video.frame('link/../clip.mp4', .5)) < 1


def test_export_symlink_parent_components(linked):
    root, target = linked
    sentinel = root / 'out.mp3'
    sentinel.write_bytes(b'keep the unrelated normalized-path file')
    requested = Path('link/../out.mp3')
    assert master.write(requested, np.zeros((4800, 2), np.float32)) == requested
    assert sentinel.read_bytes() == b'keep the unrelated normalized-path file'
    assert audio.probe(str(target / 'out.mp3'))['codec'] == 'mp3'
