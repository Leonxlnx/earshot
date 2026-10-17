"""interfaces: the optional audio-LLM critique (HTTP mocked: no network, no key needed) and the MCP server (in-process
tool functions + a real stdio round trip with the mcp client)."""
import base64
import glob
import json
import os
import re
import sys

import numpy as np
import pytest

from earshot import cli, llm, mcp_server
from earshot.report import to_json

SR = 48000
KEY = 'unit-test-key-0123456789'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSWER = """```json
{"overall": {"score": 6, "verdict": "Real foley, but the riser is a preset."},
 "scores": {"sync": null, "realism": "5/10", "mix_balance": 7, "loudness": 8, "music": 6},
 "notes": {"realism": "the riser sounds synthetic"},
 "synthetic_or_cheap": [{"start": "0:01.5", "end": 2.0, "what": "riser", "why": "plain saw sweep"}],
 "issues": [{"start": 0.2, "end": 0.4, "severity": "low", "category": "mix", "problem": "hat too bright",
             "fix": "-3 dB shelf at 8 kHz"},
            {"start": 1.5, "end": 2.0, "severity": "HIGH", "category": "realism", "problem": "synthetic riser",
             "fix": "replace with a recorded whoosh"},],
 "strengths": ["clean key foley"], "top_fixes": ["replace the riser"]}
```"""


def _bursts(seconds=3.0):
    """decaying noise bursts every 0.5 s over a faint floor (a recording-like test signal, made here)"""
    rng = np.random.RandomState(1)
    n = int(seconds * SR)
    x = rng.randn(n) * 1e-3
    for t0 in np.arange(0.1, seconds - 0.2, 0.5):
        i = int(t0 * SR)
        k = np.arange(int(0.15 * SR))
        x[i:i + len(k)] += rng.randn(len(k)) * np.exp(-k / (0.02 * SR)) * 0.4
    return x.astype(np.float32)


class Resp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self.payload, self.headers = status, payload or {}, headers or {}
        self.text = json.dumps(self.payload)
        self.ok = status < 400

    def json(self):
        return self.payload


class FakeRequests:
    """stands in for the requests module: records calls, replays queued responses, never touches the network"""
    class RequestException(Exception):
        pass

    class Timeout(RequestException):
        pass

    def __init__(self, posts=(), gets=()):
        self.posts, self.gets, self.calls = list(posts), list(gets), []

    def post(self, url, headers=None, timeout=None, **kw):
        self.calls.append(('POST', url, dict(headers or {}), kw))
        r = self.posts.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def get(self, url, headers=None, timeout=None):
        self.calls.append(('GET', url, dict(headers or {}), {}))
        return self.gets.pop(0)

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(('DELETE', url, dict(headers or {}), {}))
        return Resp(200)


def _answer(text=ANSWER):
    return Resp(200, {'candidates': [{'content': {'parts': [{'text': text}]}, 'finishReason': 'STOP'}],
                      'usageMetadata': {'promptTokenCount': 900, 'candidatesTokenCount': 300, 'totalTokenCount': 1200},
                      'modelVersion': 'gemini-3.8-flash'})


@pytest.fixture
def fake(monkeypatch):
    def install(*posts, gets=()):
        f = FakeRequests(posts, gets)
        monkeypatch.setattr(llm, '_requests', lambda: f)
        monkeypatch.setattr(llm.time, 'sleep', lambda s: f.calls.append(('SLEEP', s, {}, {})))
        return f
    return install


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    monkeypatch.delenv('EARSHOT_GEMINI_MODEL', raising=False)
    monkeypatch.setenv('GEMINI_API_KEY', KEY)


# ---------------------------------------------------------------------------------------------------------- llm
def test_llm_off_without_key(monkeypatch, wav, fake):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    f = fake()
    r = llm.critique(wav(_bursts()), consent=True)
    assert r['enabled'] is False and r['ok'] is False
    assert 'GEMINI_API_KEY' in r['summary'] and '--yes-upload' in r['text']
    assert f.calls == []


def test_llm_unexpanded_config_placeholder_is_no_key(monkeypatch, wav, fake):
    monkeypatch.setenv('GEMINI_API_KEY', '${GEMINI_API_KEY:-}')
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    f = fake()
    r = llm.critique(wav(_bursts()), consent=True)
    assert r['enabled'] is False and 'no GEMINI_API_KEY' in r['summary'] and f.calls == []


def test_llm_needs_consent(keyed, wav, fake):
    f = fake()
    r = llm.critique(wav(_bursts()))
    assert r['enabled'] is False and 'consent' in r['summary'] and '--yes-upload' in r['text']
    assert f.calls == [] and KEY not in to_json(r)


def test_llm_inline_round_trip(keyed, wav, fake):
    f = fake(_answer())
    r = llm.critique(wav(_bursts(3.0)), consent=True, start=0.5)
    assert r['ok'] and r['uploaded']['method'] == 'inline' and abs(r['uploaded']['seconds'] - 2.5) < 0.01
    method, url, headers, kw = f.calls[0]
    assert url.endswith('/v1beta/models/gemini-3.8-flash:generateContent') and KEY not in url
    assert headers['x-goog-api-key'] == KEY
    part, prompt = kw['json']['contents'][0]['parts']
    mp3 = base64.b64decode(part['inline_data']['data'])
    assert part['inline_data']['mime_type'] == 'audio/mp3' and (mp3[:3] == b'ID3' or mp3[0] == 0xFF)
    assert 'Reply with ONLY this JSON' in prompt['text'] and 'LUFS' in prompt['text']
    assert kw['json']['generationConfig']['responseMimeType'] == 'application/json'
    c = r['critique']
    assert c['scores']['realism'] == 5.0 and c['scores']['sync'] is None and c['overall']['score'] == 6.0
    assert c['issues'][0]['severity'] == 'high' and c['issues'][0]['start'] == 2.0   # clip time 1.5 + start 0.5
    assert c['synthetic_or_cheap'][0]['start'] == 2.0
    assert r['text'].startswith(r['summary']) and 'replace the riser' in r['summary']
    assert KEY not in to_json(r)


def test_llm_question_is_answered_in_json(keyed, wav, fake):
    f = fake(_answer('{"answer": "Yes, about 100 ms late.", "overall": {"score": 7, "verdict": "ok"}}'))
    r = llm.critique(wav(_bursts()), prompt='Is the hit at 1.1 s late?', consent=True)
    prompt = f.calls[0][3]['json']['contents'][0]['parts'][1]['text']
    assert 'Is the hit at 1.1 s late?' in prompt and '"answer"' in prompt
    assert r['critique']['answer'].startswith('Yes') and 'answer: Yes' in r['text']


def test_llm_video_sends_audio_plus_cut_times(keyed, tmp_path, fake, monkeypatch):
    import subprocess
    vid = str(tmp_path / 'cut.mp4')   # black 1 s -> white 1 s (one hard cut at 1.0 s) with a quiet tone
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=25:d=1', '-f', 'lavfi',
                    '-i', 'color=c=white:s=64x64:r=25:d=1', '-f', 'lavfi', '-i', 'sine=f=440:d=2', '-filter_complex',
                    '[0:v][1:v]concat=n=2:v=1[v]', '-map', '[v]', '-map', '2:a', '-shortest', vid], check=True)
    import earshot.timeline as tl
    monkeypatch.setattr(tl, 'timeline', lambda path, **kw: {'text': f'TRANSCRIPT {kw["start"]}-{kw["end"]}'})
    f = fake(_answer(), _answer())
    llm.critique(vid, consent=True)
    part, prompt = f.calls[0][3]['json']['contents'][0]['parts']
    assert 'Picture cuts' in prompt['text'] and ': 1.' in prompt['text']
    mp3 = base64.b64decode(part['inline_data']['data'])
    assert mp3[:3] == b'ID3' or mp3[0] == 0xFF
    llm.critique(vid, consent=True, with_timeline=True)
    prompt = [c for c in f.calls if c[0] == 'POST'][1][3]['json']['contents'][0]['parts'][1]['text']
    assert 'TRANSCRIPT 0.0-2.0' in prompt and 'Picture cuts' not in prompt   # the transcript already has the cuts

    def broken(path, **kw):
        raise IndexError('pop index out of range')
    monkeypatch.setattr(tl, 'timeline', broken)
    fake(_answer('{"overall": {"score": 6, "verdict": "ok"}}'))
    r = llm.critique(vid, consent=True, with_timeline=True)
    assert 'timeline context was NOT sent' in r['text'] and 'rates the soundtrack 6/10: ok. 0 issues' in r['summary']


def test_llm_files_api_above_inline_limit(keyed, wav, fake, monkeypatch):
    monkeypatch.setattr(llm, 'INLINE_MAX_B64', 16)
    f = fake(Resp(200, {}, {'x-goog-upload-url': 'https://upload.test/u1'}),
             Resp(200, {'file': {'name': 'files/abc', 'uri': 'https://g.test/v1beta/files/abc',
                                 'state': 'PROCESSING'}}),
             _answer(), gets=[Resp(200, {'name': 'files/abc', 'uri': 'https://g.test/v1beta/files/abc',
                                         'state': 'ACTIVE'})])
    r = llm.critique(wav(_bursts()), consent=True)
    kinds = [(c[0], c[1].rsplit('/', 1)[-1]) for c in f.calls if c[0] != 'SLEEP']
    assert kinds == [('POST', 'files'), ('POST', 'u1'), ('GET', 'abc'), ('POST', 'gemini-3.8-flash:generateContent'),
                     ('DELETE', 'abc')]
    start = f.calls[0]
    assert start[2]['X-Goog-Upload-Protocol'] == 'resumable' and start[2]['x-goog-api-key'] == KEY
    assert start[3]['json']['file']['display_name'] == 'earshot-review'   # the user's file name is not sent
    gen = [c for c in f.calls if c[0] == 'POST' and c[1].endswith(':generateContent')][0]
    assert gen[3]['json']['contents'][0]['parts'][0]['file_data']['file_uri'].endswith('files/abc')
    assert r['ok'] and r['uploaded']['method'] == 'files_api' and r['uploaded']['files_api_deleted'] is True


def test_llm_files_api_copy_deleted_when_processing_fails(keyed, wav, fake, monkeypatch):
    monkeypatch.setattr(llm, 'INLINE_MAX_B64', 16)
    f = fake(Resp(200, {}, {'x-goog-upload-url': 'https://upload.test/u1'}),
             Resp(200, {'file': {'name': 'files/abc', 'state': 'PROCESSING'}}),
             gets=[Resp(200, {'name': 'files/abc', 'state': 'FAILED'})])
    r = llm.critique(wav(_bursts()), consent=True)
    assert [c[0] for c in f.calls if c[0] != 'SLEEP'] == ['POST', 'POST', 'GET', 'DELETE']
    assert r['ok'] is False and 'FAILED' in r['summary'] and 'uploaded copy was deleted' in r['summary']
    assert r['text'] == r['summary']


def test_llm_http_errors_are_readable(keyed, wav, fake):
    path = wav(_bursts())
    fake(Resp(404, {'error': {'message': 'models/gemini-x is not found'}}))
    r = llm.critique(path, consent=True, model='gemini-x')
    assert r['ok'] is False and 'HTTP 404' in r['summary'] and 'EARSHOT_GEMINI_MODEL' in r['summary']
    f = fake(Resp(429, {'error': {'message': 'quota'}}), _answer())
    r = llm.critique(path, consent=True)
    assert r['ok'] and [c[1] for c in f.calls if c[0] == 'SLEEP'] == [llm.RETRY_WAIT[0]]
    fake(Resp(200, {'promptFeedback': {'blockReason': 'SAFETY'}}))
    r = llm.critique(path, consent=True)
    assert r['ok'] is False and 'SAFETY' in r['summary'] and r['text'].startswith('LLM critique failed')
    bad = Resp(200)
    bad.json, bad.text = (lambda: json.loads('<html>')), '<html>proxy</html>'
    fake(bad)
    r = llm.critique(path, consent=True)
    assert r['ok'] is False and 'not JSON' in r['summary'] and 'proxy' in r['summary']


def test_llm_network_failures_are_readable(keyed, wav, fake):
    path = wav(_bursts())
    err = FakeRequests.RequestException(f'connection refused for key {KEY}')
    f = fake(err, err, err)
    r = llm.critique(path, consent=True)
    assert r['ok'] is False and 'network error' in r['summary'] and len([c for c in f.calls if c[0] == 'SLEEP']) == 2
    f = fake(FakeRequests.Timeout())
    r = llm.critique(path, consent=True)
    assert r['ok'] is False and 'did not answer within' in r['summary'] and len(f.calls) == 1
    assert KEY not in to_json(r)


def test_llm_unparseable_answer_keeps_raw(keyed, wav, fake):
    fake(_answer('I think the mix is fine overall.'))
    r = llm.critique(wav(_bursts()), consent=True)
    assert r['ok'] is False and r['enabled'] and r['raw'].startswith('I think') and 'parseable' in r['summary']


def test_llm_model_from_env(keyed, wav, fake, monkeypatch):
    monkeypatch.setenv('EARSHOT_GEMINI_MODEL', 'gemini-test-model')
    f = fake(_answer())
    llm.critique(wav(_bursts()), consent=True)
    assert ':generateContent' in f.calls[0][1] and 'gemini-test-model' in f.calls[0][1]


def test_parse_json_and_normalize():
    assert llm.parse_json('```json\n{"a": [1, 2,],}\n```') == {'a': [1, 2]}
    assert llm.parse_json('Here you go: {"x": "a } b", "y": {"z": 1}} hope it helps') == {'x': 'a } b', 'y': {'z': 1}}
    assert llm.parse_json('no json at all') is None
    assert [llm._sec(v) for v in ('1:02.5', 12, '3.5s', 'soon', True)] == [62.5, 12.0, 3.5, None, None]
    n = llm.normalize({'overall': 8, 'scores': {'realism': '9/10', 'sync': 14},
                       'issues': [{'start': 5, 'severity': 'low'}, {'start': 9, 'severity': 'bogus'},
                                  {'start': 99, 'end': 1, 'severity': 'high'}]}, offset=10, clip_len=20)
    assert n['overall']['score'] == 8.0 and n['scores']['realism'] == 9.0 and n['scores']['sync'] == 10.0
    got = [(i['severity'], i['start'], i['end']) for i in n['issues']]
    assert got == [('high', 30.0, 30.0), ('medium', 19.0, 19.0), ('low', 15.0, 15.0)]


def test_encode_trims_and_is_small(wav, tmp_path):
    data, mime, info = llm.encode(wav(_bursts(5.0)), start=1.0, max_seconds=2.0)
    assert info['trimmed'] and info['start'] == 1.0 and info['seconds'] == 2.0 and mime == 'audio/mp3'
    p = tmp_path / 'clip.mp3'
    p.write_bytes(data)
    from earshot import audio
    pr = audio.probe(str(p))
    assert pr['channels'] == 1 and abs(pr['duration'] - 2.0) < 0.1 and len(data) < 2.0 * 64000 / 8 * 1.3


def test_llm_cli_explains_how_to_enable(monkeypatch, wav, capsys):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    assert cli.main(['llm', wav(_bursts())]) == 1
    out = capsys.readouterr().out
    assert 'GEMINI_API_KEY' in out and 'aistudio.google.com' in out


# ---------------------------------------------------------------------------------------------------------- mcp
def _json_part(text):
    return json.loads(text.split('JSON (compact): ', 1)[1])


def test_mcp_measure_tool_summary_first(wav):
    text = mcp_server.tool_measure(wav(_bursts(3.0)), light=True)
    first = text.splitlines()[0]
    assert first.startswith('3.000 s, ') and 'LUFS integrated' in first and 'dBTP' in first
    j = _json_part(text)
    assert 'true_peak_dbtp' in j and 'summary' not in j


def test_measure_summary_wording():
    m = {'duration': 0.25, 'rms_dbfs': -20.0, 'true_peak_dbtp': 0.4, 'clipped_samples': 12, 'noise_floor_dbfs': -120.0,
         'centroid_hz': 880, 'attack_ms': 0.0, 'decay_ms': 5.0, 'onset_count': 1, 'digital_silence_frac': 0.4}
    s = mcp_server.measure_summary(m)
    assert s.startswith('0.250 s, RMS -20.0 dBFS (too short for LUFS), true peak 0.4 dBTP, 12 clipped samples')
    assert 'attack 0 ms, decay 5 ms' in s and '40.0% exact digital zeros' in s and 'weak evidence' in s
    assert 'code-generated' not in s
    assert mcp_server.measure_summary({'duration': 0.001, 'error': 'too short'}) == '0.001 s: too short.'
    silent = mcp_server.measure_summary({'duration': 10.0, 'peak_dbfs': -240.0, 'lufs': -70.0, 'true_peak_dbtp': -240.0,
                                         'noise_floor_dbfs': -200.0, 'digital_silence_frac': 1.0, 'attack_ms': 0.0})
    assert silent.startswith('10.000 s, digitally silent (peak -240 dBFS): nothing to measure') and 'LUFS' not in silent


def test_mcp_measure_tool_on_digital_silence(wav):
    first = mcp_server.tool_measure(wav(np.zeros(SR * 2, np.float32)), light=True).splitlines()[0]
    assert first.startswith('2.000 s, digitally silent') and 'kHz' not in first


def test_mcp_errors_are_readable(tmp_path):
    with pytest.raises(mcp_server.ToolFailure, match='not found'):
        mcp_server.tool_measure(str(tmp_path / 'missing.wav'))
    with pytest.raises(mcp_server.ToolFailure, match='folder'):
        mcp_server.tool_measure(str(tmp_path))
    with pytest.raises(mcp_server.ToolFailure, match='no audio/video files'):
        mcp_server.tool_rank([str(tmp_path)], 'a door slam')
    good = tmp_path / 'a.wav'
    good.write_bytes(b'RIFF')
    assert mcp_server._many(str(tmp_path)) == [str(good)]   # one string is fine
    with pytest.raises(mcp_server.ToolFailure, match=r'no audio/video files found in .*typo_folder'):
        mcp_server._many([str(tmp_path), str(tmp_path / 'typo_folder')], 'refs')
    with pytest.raises(mcp_server.ToolFailure, match='not part of this install yet. Tools with their module present: '):
        mcp_server._run('nonexistent_tool_xyz', {})


def test_mcp_module_that_fails_to_import_is_reported(monkeypatch):
    real = mcp_server.importlib.import_module

    def broken(name):
        if name == 'earshot.describe':
            raise SyntaxError('unterminated string literal')
        return real(name)
    monkeypatch.setattr(mcp_server.importlib, 'import_module', broken)
    with pytest.raises(mcp_server.ToolFailure, match=r'could not be loaded \(SyntaxError: unterminated'):
        mcp_server._api('describe')


def test_mcp_tool_exception_becomes_readable_failure(wav):
    def boom(path):
        raise ValueError('decoder exploded')
    with pytest.raises(mcp_server.ToolFailure, match='describe failed: ValueError: decoder exploded'):
        mcp_server._run('describe', {'path': wav(_bursts())}, fn=boom)


def test_mcp_call_passes_only_supported_kwargs():
    def f(path, top=3):
        return {'summary': f'{path} {top}'}
    r, ignored = mcp_server._call(f, {'path': 'x', 'top': None, 'win': 2.0})
    assert r['summary'] == 'x 3' and ignored == ['win']
    text = mcp_server._run('fake', {'path': 'x', 'win': 2.0}, fn=f)
    assert text.startswith('x 3') and 'ignored: win' in text


def test_mcp_compact_json_is_bounded():
    r = {'summary': 's', 'text': 't' * 50, 'curve': list(np.linspace(0, 1, 5000)),
         'events': [{'t': i / 7, 'label': 'x' * 200} for i in range(500)]}
    s = mcp_server.compact(r)
    assert len(s) <= mcp_server.JSON_MAX + 60 and '"summary"' not in s and 'more' in s
    out = mcp_server._format(r)
    assert out.startswith('s\n\nttt') and 'JSON (compact): ' in out
    wide = json.loads(mcp_server.compact({'a': {f'k{i}': 'x' * 100 for i in range(200)}, 'b': 1}))  # valid JSON
    assert wide['b'] == 1 and wide['cut_fields'] == ['a'] and 'a' not in wide


def test_mcp_sheet_returns_png_path_and_image(wav, tmp_path, monkeypatch):
    from PIL import Image

    def fake_sheet(path, out_png, video=None, detail='normal'):
        Image.new('RGB', (2400, 900), (20, 30, 40)).save(out_png)
        return {'summary': f'sheet drawn ({detail})', 'png': out_png}
    monkeypatch.setattr(mcp_server, '_api', lambda name: fake_sheet)
    out = str(tmp_path / 'sheet.png')
    text, img = mcp_server.tool_sheet(wav(_bursts()), out_png=out, detail='low')
    assert text.startswith(f'PNG: {out}') and 'sheet drawn (low)' in text and 'ignored' not in text
    with pytest.raises(mcp_server.ToolFailure, match='detail must be'):
        mcp_server.tool_sheet(wav(_bursts()), detail='max')
    import io
    im = Image.open(io.BytesIO(img))
    assert max(im.size) == mcp_server.IMG_INLINE_MAX and im.format == 'JPEG'
    Server, Img, ToolError, _ = mcp_server._sdk()
    res = mcp_server._adapt(mcp_server.tool_sheet, Img, ToolError)(path=wav(_bursts()), out_png=out)
    assert isinstance(res[0], str) and isinstance(res[1], Img)
    assert mcp_server.tool_sheet(wav(_bursts()), out_png=out, include_image=False).startswith('PNG: ')


def test_mcp_realness_folder_rows_and_missing_model(wav, tmp_path, monkeypatch):
    import types
    a, b = wav(_bursts(), 'a.wav'), wav(_bursts(2.0), 'b.wav')
    seen = {}

    def many(paths, ast=None, start=None, end=None):
        seen['range'] = (start, end)
        return {'summary': f'{len(paths)} files judged', 'files': [
            {'path': p, 'p_real': 0.91, 'verdict': 'real or pro-produced', 'why': 'a continuous noise floor'}
            for p in paths]}
    mod = types.SimpleNamespace(MODEL_PATH=a, realness_many=many)
    monkeypatch.setitem(sys.modules, 'earshot.realness', mod)
    monkeypatch.setattr(mcp_server, '_api', lambda name: lambda path, start=None, end=None, ast=None: {'summary': '1'})
    text = mcp_server.tool_realness(str(tmp_path))
    assert text.startswith('2 files judged') and '0.91  real or pro-produced' in text and 'b.wav: a continuous' in text
    assert mcp_server.tool_realness(b).startswith('1')
    mcp_server.tool_realness(str(tmp_path), start=1.0)  # a range applies to every file of a folder
    assert seen['range'] == (1.0, None)
    with pytest.raises(mcp_server.ToolFailure, match='not found'):
        mcp_server.tool_realness(str(tmp_path / 'nope.wav'))
    mod.MODEL_PATH = str(tmp_path / 'realness_model.json')
    with pytest.raises(mcp_server.ToolFailure, match='NOT checked') as e:
        mcp_server.tool_realness(a)
    assert 'train' not in str(e.value)


def test_mcp_llm_tool_without_key(monkeypatch, wav):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    text = mcp_server.tool_llm_critique(wav(_bursts()), consent=True)
    assert text.startswith('LLM critique is off') and 'GEMINI_API_KEY' in text


def test_mcp_server_registers_all_tools():
    import anyio
    srv = mcp_server.build()
    tools = anyio.run(srv.list_tools)
    names = {t.name for t in tools}
    assert names == {'measure', 'describe', 'match', 'rank', 'find', 'compare', 'ab', 'realness', 'timeline', 'sheet',
                     'critique', 'doctor', 'llm_critique', 'sfx_list', 'sfx_render', 'kit_render'}
    for t in tools:
        schema = getattr(t, 'input_schema', None) or t.inputSchema
        assert len(t.description) > 120 and 'Use ' in t.description
        assert all(p.get('description') for p in schema['properties'].values()), t.name
    llm_tool = next(t for t in tools if t.name == 'llm_critique')
    ann = llm_tool.annotations.model_dump()   # snake_case in mcp 2.x
    assert ann.get('read_only_hint', ann.get('readOnlyHint')) is False
    assert ann.get('open_world_hint', ann.get('openWorldHint')) is True


# one person's taste written as a rule for every agent (issue #1); conditional, generic phrasing such as "when the user
# wants real sound" stays allowed
PREFERENCE_AS_RULE = re.compile(
    r"\bthe user (?:rejects|rejected|hates|hated)\b|\buser-rejected\b"
    r"|\bthe user's (?:own )?(?:reference (?:profile|films)|ear)\b"
    r"|\bthe user (?:chose|picked) as\b|\bfilms the user (?:chose|picked)\b|random ass", re.I)


def test_agent_text_states_no_preference_as_a_rule():
    """issue #1: tool descriptions, server instructions, skills and docs say what earshot measures and leave the
    decision to the current user's own instructions (synthesized audio may be the point of a film)"""
    import anyio
    texts = {f'tool {t.name}': t.description for t in anyio.run(mcp_server.build().list_tools)}
    texts['INSTRUCTIONS'] = mcp_server.INSTRUCTIONS
    texts['llm.RUBRIC'] = llm.RUBRIC   # the Gemini rubric is fed back to agents through llm_critique
    for pattern in ('README.md', 'docs/*.md', 'skills/**/*.md', 'eval/*/RESULTS.md'):
        for path in glob.glob(os.path.join(REPO, pattern), recursive=True):
            with open(path, encoding='utf-8') as f:
                texts[os.path.relpath(path, REPO)] = f.read()
    assert len(texts) > 30 and any(k.startswith('skills') for k in texts)
    hits = [f'{k}: {m.group(0)!r}' for k, v in texts.items()
            for m in PREFERENCE_AS_RULE.finditer(' '.join(v.split()))]   # joined lines: phrases split by a wrap count
    assert not hits, hits
    realness = ' '.join(texts['tool realness'].split())
    assert 'Use when the user wants real or professionally produced sound' in realness
    assert 'Synthesized audio may be intended' in realness
    assert 'realness when the user wants real' in ' '.join(mcp_server.INSTRUCTIONS.split())
    skill = ' '.join(texts[os.path.join('skills', 'earshot', 'SKILL.md')].split())
    assert "Reject 'likely code-synthesized'" not in skill and 'realness is information, not a gate' in skill


def test_llm_rubric_does_not_count_synthesis_as_a_flaw(keyed, wav, fake):
    """issue #1 leftover (review): the Gemini rubric set 'real recorded foley' as the bar, scored '3 = obviously cheap
    or synthetic' and suggested 'replace the beep with a recorded UI tap', so a film synthesized on purpose got
    'replace your synthesized sounds' advice. Synthesis is a flaw only where a sound should read as recorded, and
    synth_intended (CLI --synth-intended, MCP synth_intended) says so in the prompt"""
    rubric = ' '.join(llm.RUBRIC.split())
    for gone in ('real recorded foley', 'cheap or synthetic', 'recorded UI tap'):
        assert gone not in rubric, gone
    assert 'do not count synthesis itself as a flaw' in rubric and 'Synthesis is a flaw only where' in rubric
    assert llm.SYNTH_INTENT not in llm.build_prompt('ctx')
    assert llm.SYNTH_INTENT in llm.build_prompt('ctx', synth_intended=True)
    f = fake(_answer(), _answer())
    assert llm.critique(wav(_bursts()), consent=True, synth_intended=True)['ok']
    assert llm.SYNTH_INTENT in f.calls[0][3]['json']['contents'][0]['parts'][1]['text']
    assert cli.main(['llm', wav(_bursts()), '--yes-upload', '--synth-intended', '--json']) == 0
    assert llm.SYNTH_INTENT in f.calls[-1][3]['json']['contents'][0]['parts'][1]['text']
    import anyio
    [tool] = [t for t in anyio.run(mcp_server.build().list_tools) if t.name == 'llm_critique']
    schema = getattr(tool, 'input_schema', None) or tool.inputSchema
    assert 'synth_intended' in schema['properties']


async def _stdio_session(calls, log, *args):
    """start `python -m earshot mcp` (or `python -m <args>`) as a subprocess and run calls(session) over stdio"""
    import anyio
    from mcp import ClientSession, StdioServerParameters, stdio_client
    params = StdioServerParameters(command=sys.executable, args=['-m', *(args or ('earshot', 'mcp'))], cwd=REPO,
                                   env={'EARSHOT_THREADS': '2', 'PYTHONIOENCODING': 'utf-8'})
    with anyio.fail_after(900), open(log, 'w', encoding='utf-8') as errlog:
        async with stdio_client(params, errlog=errlog) as (rd, wr):
            async with ClientSession(rd, wr) as s:
                with anyio.fail_after(300):   # a server that answers on the wrong stream hangs here
                    await s.initialize()
                return await calls(s)


def _text(res):
    return '\n'.join(getattr(c, 'text', '') for c in res.content)


def _is_error(res):
    return bool(getattr(res, 'is_error', None) or getattr(res, 'isError', None))


def test_mcp_stdio_end_to_end(wav, tmp_path):
    """fast round trip over a real stdio connection (doctor imports torch: it is in the slow test below)"""
    import anyio
    path = wav(_bursts(3.0))

    async def calls(s):
        tools = await s.list_tools()
        return (tools, await s.call_tool('measure', {'path': path, 'light': True}),
                await s.call_tool('measure', {'path': path + '.missing'}),
                await s.call_tool('rank', {'paths': path + '.missing', 'query': 'a door'}),
                await s.call_tool('llm_critique', {'path': path}))
    tools, meas, missing, rank, llm_res = anyio.run(_stdio_session, calls, str(tmp_path / 'server.log'))
    assert len(tools.tools) == 16   # 13 listening tools + sfx_list, sfx_render, kit_render (the sound bank)
    assert not _is_error(meas) and _text(meas).startswith('3.000 s, ') and 'dBTP' in _text(meas)
    assert _is_error(missing) and 'not found' in _text(missing)
    assert _is_error(rank) and 'no audio/video files found' in _text(rank)   # a plain string passes the schema
    assert not _is_error(llm_res) and 'LLM critique is off' in _text(llm_res)


@pytest.mark.slow
def test_mcp_stdio_describe_keeps_models_warm(wav, tmp_path):
    import importlib.util
    import time
    import anyio
    if importlib.util.find_spec('earshot.describe') is None:
        pytest.skip('earshot.describe not built yet')
    p1, p2 = wav(_bursts(3.0), 'a.wav'), wav(_bursts(2.0)[::-1].copy(), 'b.wav')

    async def calls(s):
        doc = await s.call_tool('doctor', {})
        assert not _is_error(doc) and 'earshot' in _text(doc).lower()
        t0 = time.time()
        a = await s.call_tool('describe', {'path': p1})
        t1 = time.time()
        b = await s.call_tool('describe', {'path': p2, 'top': 3})
        t2 = time.time()
        sheet = None
        if importlib.util.find_spec('earshot.sheet'):
            sheet = await s.call_tool('sheet', {'path': p1, 'out_png': str(tmp_path / 'a_sheet.png')})
        return a, b, t1 - t0, t2 - t1, sheet
    a, b, cold, warm, sheet = anyio.run(_stdio_session, calls, str(tmp_path / 'server.log'))
    assert not _is_error(a) and not _is_error(b), _text(a) + _text(b)
    assert len(_text(a)) > 40 and warm < cold   # the second file needs inference but no model load
    if sheet is not None:
        kinds = [c.type for c in sheet.content]
        assert not _is_error(sheet) and kinds == ['text', 'image'] and (tmp_path / 'a_sheet.png').is_file()
        assert (getattr(sheet.content[1], 'mime_type', None) or sheet.content[1].mimeType) == 'image/jpeg'


@pytest.mark.slow
def test_mcp_stdio_warm_start_answers_and_preloads(wav, tmp_path):
    """--warm loads the models in a thread while the transport starts (this once sent the initialize answer to
    stderr and the client hung); also covers the fast-start entry point python -m earshot.mcp_server"""
    import time
    import anyio
    log = tmp_path / 'server.log'
    path = wav(_bursts(2.0))

    async def calls(s):
        meas = await s.call_tool('measure', {'path': path, 'light': True})
        t0 = time.time()
        while 'models warm' not in log.read_text(encoding='utf-8', errors='replace') and time.time() - t0 < 300:
            await anyio.sleep(1)
        return meas, await s.call_tool('describe', {'path': path})
    meas, desc = anyio.run(_stdio_session, calls, str(log), 'earshot.mcp_server', '--warm')
    assert not _is_error(meas) and not _is_error(desc) and 'models warm' in log.read_text(encoding='utf-8')
