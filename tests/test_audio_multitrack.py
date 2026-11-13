"""Real synthetic media: preserve FFmpeg's selected track, stereo image and mono unity gain."""
import hashlib
import subprocess

import numpy as np
import pytest
import soundfile as sf

from earshot import audio, timeline


def _wav(tmp_path, channels, sr=48000):
    t = np.arange(sr) / sr
    x = np.column_stack([0.1 * np.sin(2 * np.pi * (220 + i * 110) * t) for i in range(channels)])
    p = tmp_path / f'{channels}_{sr}.wav'
    sf.write(p, x, sr, subtype='FLOAT')
    return str(p)


def _reference(path, channels, sr=48000):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-vn', '-f', 'f32le',
                        '-acodec', 'pcm_f32le', '-ac', str(channels), '-ar', str(sr), '-'],
                       capture_output=True, check=True)
    x = np.frombuffer(p.stdout, np.float32)
    return x if channels == 1 else x.reshape(-1, channels)


@pytest.mark.parametrize('first_channels,second_channels,default', [
    (1, 2, None), (1, 2, 0), (1, 2, 1), (2, 1, 1), (2, 1, None), (1, 2, 'both'),
])
def test_multitrack_stereo_and_mono(tmp_path, first_channels, second_channels, default):
    sources = [_wav(tmp_path, first_channels), _wav(tmp_path, second_channels)]
    path = str(tmp_path / 'tracks.mkv')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', sources[0], '-i', sources[1],
                    '-map', '0:a', '-map', '1:a', '-c:a', 'pcm_f32le',
                    '-disposition:a:0', 'default' if default in (0, 'both') else '0',
                    '-disposition:a:1', 'default' if default in (1, 'both') else '0', path], check=True)
    # With these populated streams, FFmpeg prioritizes default, then channel count.
    selected = default if isinstance(default, int) else (0 if first_channels > second_channels else 1)
    count = [first_channels, second_channels][selected]
    expected = _reference(sources[selected], 1 if count == 1 else 2)
    if count == 1:
        expected = np.repeat(expected[:, None], 2, axis=1)
    actual = audio.load(path, cache=False)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    # Model input convention must remain byte-for-byte the original FFmpeg mono downmix.
    np.testing.assert_array_equal(audio.load(path, mono=True, cache=False), _reference(path, 1))


@pytest.mark.parametrize('channels', [1, 2, 6])
@pytest.mark.parametrize('sr', [44100, 48000])
def test_single_track_and_resampling(tmp_path, channels, sr):
    path = _wav(tmp_path, channels, sr)
    expected = _reference(path, 1 if channels == 1 else 2)
    if channels == 1:
        expected = np.repeat(expected[:, None], 2, axis=1)
    np.testing.assert_allclose(audio.load(path, cache=False), expected, atol=1e-7)
    np.testing.assert_array_equal(audio.load(path, mono=True, cache=False), _reference(path, 1))


def test_decode_and_span_keys_cannot_reuse_v2_samples(tmp_path):
    path = _wav(tmp_path, 1)
    old_decode = hashlib.sha1(f'{audio._state(path)}|48000|2|v2'.encode()).hexdigest()[:20]
    assert audio._key(path, 48000, 2) != old_decode
    assert timeline.span_id(path) != audio.file_id(path) + '|mono2'
    assert timeline.span_id(path).endswith(f'|decode{audio.DECODE_VERSION}')
