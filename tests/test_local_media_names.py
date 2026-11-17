"""Local filename punctuation must not become an FFmpeg protocol or option."""
import functools
import http.server
import os
from pathlib import Path
import subprocess
import threading

import numpy as np
import pytest
import soundfile as sf

from earshot import audio, config, video
from earshot.llm import encode
from earshot.synth import master


@pytest.fixture(params=['take:01', '-take', 'ordinary'])
def media(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'DEC_CACHE', str(tmp_path / 'decoded'))
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    wav = tmp_path / f'{request.param}.wav'
    # ':' is not a legal native Windows filename. Ordinary names still test URI behavior there.
    if os.name == 'nt' and ':' in request.param:
        pytest.skip('colon is not a legal Windows filename')
    x = (.2 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    sf.write(wav, x, 48000, subtype='FLOAT')
    film = wav.with_suffix('.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=black:s=160x90:r=25:d=1', '-c:v', 'mpeg4', str(film)],
                   check=True, capture_output=True)
    return wav, film, x


@pytest.mark.parametrize('spelling', ['bare', 'dot', 'absolute', 'pathlib'])
def test_local_audio_name(media, spelling):
    wav, _, x = media
    name = {'bare': wav.name, 'dot': './' + wav.name, 'absolute': str(wav), 'pathlib': Path(wav.name)}[spelling]
    assert audio.probe(name)['channels'] == 1
    decoded = audio.load(name, cache=False)
    np.testing.assert_array_equal(decoded[:, 0], x)
    np.testing.assert_array_equal(decoded[:, 1], x)
    np.testing.assert_array_equal(audio.load(name), decoded)
    np.testing.assert_array_equal(audio.load(name), decoded)
    assert audio._key(name, 48000, 2) == audio._key(str(wav), 48000, 2)
    encoded, mime, info = encode(name)
    assert encoded and mime == 'audio/mp3' and info['seconds'] == 1


@pytest.mark.parametrize('spelling', ['bare', 'dot', 'absolute', 'pathlib'])
def test_local_video_name(media, spelling):
    _, film, _ = media
    name = {'bare': film.name, 'dot': './' + film.name, 'absolute': str(film), 'pathlib': Path(film.name)}[spelling]
    assert audio.probe(name)['has_video'] is True
    assert video.cuts(name, cache=False) == []
    assert video.cuts(name, method='scene', cache=False) == []
    assert video.frame(name, .5) is not None


@pytest.mark.parametrize('name', ['export:01', '-export', 'ordinary'])
@pytest.mark.parametrize('ext', ['wav', 'flac', 'mp3', 'ogg', 'm4a', 'webm'])
def test_local_export_name(tmp_path, monkeypatch, name, ext):
    if os.name == 'nt' and ':' in name:
        pytest.skip('colon is not a legal Windows filename')
    monkeypatch.chdir(tmp_path)
    path = Path(f'{name}.{ext}')
    output = master.write(path, np.zeros((4800, 2), np.float32))
    assert output == path and path.is_file()
    assert audio.probe(str(path.resolve()))['codec']


def test_existing_file_uri_and_http_inputs(media):
    wav, film, _ = media
    assert audio.probe('file:' + str(wav))['channels'] == 1
    assert video.frame('file:' + str(film), .5) is not None
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(wav.parent))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        from urllib.parse import quote
        base = f'http://127.0.0.1:{server.server_port}/'
        assert audio.probe(base + quote(wav.name))['channels'] == 1
        assert video.frame(base + quote(film.name), .5) is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_timeline_preserves_public_path_spelling(media):
    from earshot.timeline import timeline
    _, film, _ = media
    first = timeline(film.name)
    second = timeline(film.name)
    assert first['cached'] is False and second['cached'] is True
    for result in (first, second):
        assert result['file'] == result['video'] == film.name
