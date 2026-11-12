"""The optional example never needs a real key or network in tests."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

spec = importlib.util.spec_from_file_location(
    'monid_voice_design', Path(__file__).resolve().parents[1] / 'docs' / 'monid_voice_design.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def response(data, status=200):
    return Mock(status_code=status, json=Mock(return_value=data))


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setenv('MONID_API_KEY', 'fake-key')
    post, get = Mock(), Mock()
    monkeypatch.setattr(example.requests, 'post', post)
    monkeypatch.setattr(example.requests, 'get', get)
    monkeypatch.setattr(example.time, 'sleep', lambda seconds: None)
    return post, get


def test_request_and_output(http):
    post, get = http
    post.return_value = response({'output': {'candidate_1': {'url': 'preview.wav'}}})
    assert example.voice_design('温暖的声音') == {'candidate_1': {'url': 'preview.wav'}}
    args, kwargs = post.call_args
    assert args == ('https://api.monid.ai/v1/run',)
    assert kwargs['headers'] == {'Authorization': 'Bearer fake-key',
                                  'Content-Type': 'application/json; charset=utf-8'}
    assert json.loads(kwargs['data']) == {
        'provider': 'fishaudio', 'endpoint': '/voice-design',
        'input': {'body': {'instruction': '温暖的声音'}, 'queryParams': {}, 'pathParams': {}}}
    assert '温暖的声音'.encode() in kwargs['data']
    get.assert_not_called()


def test_no_key(http, monkeypatch):
    monkeypatch.delenv('MONID_API_KEY')
    with pytest.raises(ValueError, match='set MONID_API_KEY'):
        example.voice_design('warm voice')
    http[0].assert_not_called()


def test_async(http):
    post, get = http
    post.return_value = response({'runId': 'run-1'}, 202)
    get.side_effect = [response({'status': 'RUNNING'}),
                       response({'status': 'COMPLETED', 'output': {'voices': []}})]
    assert example.voice_design('warm voice') == {'voices': []}
    assert get.call_count == 2
    assert get.call_args.args == ('https://api.monid.ai/v1/runs/run-1',)
    assert get.call_args.kwargs['headers'] == {'Authorization': 'Bearer fake-key'}


@pytest.mark.parametrize('status', ['FAILED', 'TIMED_OUT', 'STOPPED', 'BLOCKED'])
def test_terminal_failure(http, status):
    http[0].return_value = response({'runId': 'run-1'}, 202)
    http[1].return_value = response({'status': status})
    with pytest.raises(RuntimeError, match=status):
        example.voice_design('warm voice')


def test_provider_error(http):
    http[0].return_value = response({'status': 'COMPLETED', 'providerResponse': {
        'httpStatus': 400, 'error': {'message': 'invalid instruction'}}})
    with pytest.raises(example.VoiceDesignError, match='Fish Audio HTTP 400.*invalid instruction'):
        example.voice_design('warm voice')


@pytest.mark.parametrize('poll', [False, True])
def test_http_error(http, poll):
    failing = response({})
    failing.raise_for_status.side_effect = requests.HTTPError('unauthorized')
    http[0].return_value = response({'runId': 'run-1'}, 202) if poll else failing
    http[1].return_value = failing
    with pytest.raises(requests.HTTPError, match='unauthorized'):
        example.voice_design('warm voice')


def test_timeout(http):
    http[0].return_value = response({'runId': 'run-1'}, 202)
    with pytest.raises(TimeoutError, match='timed out'):
        example.voice_design('warm voice', timeout=0)
    http[1].assert_not_called()
