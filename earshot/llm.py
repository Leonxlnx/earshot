"""Optional audio-LLM critique: a second opinion in words from a hosted audio model (Google Gemini).

    earshot llm film.mp4 --yes-upload                    # film-sound rubric: sync, realism, mix, loudness, fixes
    earshot llm mix.wav --yes-upload --with-timeline     # also send earshot's own listening transcript as context
    earshot llm mix.wav --yes-upload --prompt "Does the whoosh at 0:12 feel late?"

    from earshot import llm
    r = llm.critique('mix.wav', consent=True)    # r['summary'], r['text'], r['critique']['scores' | 'issues' | ...]
    r['critique']['issues']      # [{'start', 'end', 'severity', 'category', 'problem', 'fix'}] in file seconds

Off unless BOTH hold: GEMINI_API_KEY (or GOOGLE_API_KEY) is set in the environment, and the caller consents to the
upload (--yes-upload / consent=True), because the audio leaves this machine and goes to Google. Without them it
returns a message explaining how to enable it; the rest of earshot never needs it. Only audio is sent (never the
picture of a video): mono 64 kbps MP3, trimmed to --max-seconds (default 600). Requests up to ~18 MB of base64 go
inline; larger ones use the Files API and the uploaded copy is deleted right after the answer (or the failure). The
key is read from the environment, sent only in the x-goog-api-key header and never printed, logged or stored.

Model: EARSHOT_GEMINI_MODEL (default gemini-3.8-flash). Gemini API docs used (checked 2026-10-14):
  https://ai.google.dev/gemini-api/docs/audio               audio input, 32 tokens/s, downsampled to 16 kbps mono
  https://ai.google.dev/gemini-api/docs/file-input-methods  inline limit (100 MB; the audio page still says 20 MB)
  https://ai.google.dev/gemini-api/docs/files               Files API resumable upload, 48 h retention, DELETE
  https://ai.google.dev/api/generate-content                v1beta models/{model}:generateContent, JSON mode

Trust: an opinion from a general model, not a measurement, and not deterministic. Gemini hears a reduced copy (its docs:
downsampled to 16 kbps, one channel), so fine top-end detail and stereo are beyond it. Timestamps are approximate (a
few seconds off is common); loudness numbers in the answer come from earshot's own measurements, passed in as context.
Verify every suggested fix with earshot timeline / ab before and after.
"""
import base64
import json
import os
import re
import subprocess
import time

from . import audio
from .report import emit, fmt_t

DEFAULT_MODEL = 'gemini-3.8-flash'
BASE = 'https://generativelanguage.googleapis.com'
INLINE_MAX_B64 = 18 * 1024 * 1024   # stay under the strictest documented inline request limit (20 MB)
KBPS = 64
RETRY_STATUS = (429, 500, 502, 503, 504)
RETRY_WAIT = (4, 12)
KEY_URL = 'https://aistudio.google.com/apikey'

RUBRIC = """You are a senior re-recording mixer and sound supervisor reviewing the soundtrack of a short product launch
film (or one stem of it). The bar is a premium launch film from a top studio: every sound crafted and intentional,
music that is produced and mixed, sounds that land exactly on picture events, a clean, controlled master that is loud
but not crushed. Effects may be recorded foley or designed / synthesized on purpose: judge each by whether it fits the
film and is well made, not by how it was made. Listen to the whole clip. Be specific and blunt; praise only what is
genuinely good.

Judge:
1. sync: do hits, whooshes and accents land where they seem to belong? You cannot see the picture: use the picture
   cut / cue times in the context if given, otherwise judge rhythmic tightness and whether sounds feel late, early or
   smeared. Use null for the sync score when there is nothing to judge it against.
2. realism: which sounds read as real recordings and which as code-synthesized, stock-cheesy or generic (pure sine
   beeps, plain filtered-noise whooshes, chiptune blips, preset risers, the identical sample machine-gunned)?
   Synthesis is a flaw only where a sound is meant to read as recorded. If the context or the editor's question says
   the film uses synthesized or procedural sound on purpose, judge those sounds as designed sound (clean, crafted,
   fitting the picture) and do not count synthesis itself as a flaw.
3. mix balance: music vs effects vs ambience, masking, elements too loud or buried, low-end mud, harsh 2-5 kHz
   (the clip is a mono reduction: do not judge stereo width).
4. loudness and dynamics: use the measured numbers in the context, never guess numbers; squashed, too quiet, peaky,
   pumping, clipped?
5. what is cheap: the moments that would make a professional wince.
6. concrete fixes, each tied to a time range and saying exactly what to change (for example: soften the harsh beep
   or swap it for a gentler tap that suits the film, -4 dB on the music under 12.0-15.0, move the hit 80 ms earlier,
   high-pass the pad at 150 Hz).

Scores 1-10: 10 = indistinguishable from the premium reference, 7 = good professional work with small flaws,
5 = competent amateur, 3 = obviously cheap, 1 = broken. realism rates how convincing and well made the sounds are for
what the film intends (recorded, or designed on purpose), not whether they were recorded.
Times are seconds from the start of THIS clip, as numbers (12.4, not "0:12"). Reply with ONLY this JSON object:
{"overall": {"score": <1-10>, "verdict": "<one sentence>"},
 "scores": {"sync": <1-10 or null>, "realism": <1-10>, "mix_balance": <1-10>, "loudness": <1-10>,
            "music": <1-10 or null when there is no music>},
 "notes": {"sync": "", "realism": "", "mix_balance": "", "loudness": "", "music": ""},
 "synthetic_or_cheap": [{"start": <s>, "end": <s>, "what": "<the sound>",
                         "why": "<what gives it away: cheap, generic, or synthetic where it should read as recorded>"}],
 "issues": [{"start": <s>, "end": <s>, "severity": "high|medium|low",
             "category": "sync|realism|mix|loudness|music|noise|edit", "problem": "", "fix": ""}],
 "strengths": [""],
 "top_fixes": ["<most important first, at most 5>"]%s}"""

MEASURED = [('integrated loudness', 'lufs', ' LUFS'), ('true peak', 'true_peak_dbtp', ' dBTP'),
            ('loudness range', 'lra_lu', ' LU'), ('short-term max', 'short_term_max_lufs', ' LUFS'),
            ('quietest 20 ms level (10th percentile)', 'noise_floor_dbfs', ' dBFS'),
            ('clipped samples', 'clipped_samples', '')]
SEV = {'high': 0, 'medium': 1, 'low': 2}
SCORE_KEYS = ('sync', 'realism', 'mix_balance', 'loudness', 'music')


class LLMError(RuntimeError):
    pass


class Provider:
    """one hosted audio model: subclasses implement generate(audio_bytes, mime, prompt, model, key) -> dict"""
    name = ''
    env_keys = ()
    default_model = ''

    def key(self):
        """the first key set in the environment; an unexpanded "${VAR}" placeholder from an MCP config counts as none"""
        return next((v for v in (os.environ.get(k, '').strip() for k in self.env_keys)
                     if v and not v.startswith('${')), None)

    def model(self, model=None):
        return (model or self.default_model).removeprefix('models/')

    def generate(self, data, mime, prompt, model, key):
        raise NotImplementedError


class Gemini(Provider):
    name = 'gemini'
    env_keys = ('GEMINI_API_KEY', 'GOOGLE_API_KEY')

    @property
    def default_model(self):
        return os.environ.get('EARSHOT_GEMINI_MODEL') or DEFAULT_MODEL

    def generate(self, data, mime, prompt, model, key):
        rq = _requests()
        b64 = base64.b64encode(data).decode()
        stored = {}   # the Files API name, as soon as Google holds a copy: deleted whatever happens next
        try:
            if len(b64) <= INLINE_MAX_B64:
                part, method = {'inline_data': {'mime_type': mime, 'data': b64}}, 'inline'
            else:
                f = self._upload(rq, data, mime, key, stored)
                part, method = {'file_data': {'mime_type': mime, 'file_uri': f['uri']}}, 'files_api'
            body = {'contents': [{'role': 'user', 'parts': [part, {'text': prompt}]}],
                    'generationConfig': {'responseMimeType': 'application/json', 'maxOutputTokens': 16384}}
            j = _json(_post(rq, f'{BASE}/v1beta/models/{model}:generateContent', key, json=body, timeout=(20, 600)))
        except Exception as e:
            if not stored:
                raise
            gone = self._delete(rq, stored, key)
            raise LLMError(f'{_msg(e)} (the uploaded copy was ' +
                           ('deleted)' if gone else 'NOT deleted: Google removes it after 48 h)')) from None
        deleted = self._delete(rq, stored, key) if stored else None
        cands = j.get('candidates') or []
        if not cands:
            fb = j.get('promptFeedback') or {}
            raise LLMError(f'Gemini returned no answer (blockReason: {fb.get("blockReason", "unknown")}).')
        c = cands[0]
        parts = (c.get('content') or {}).get('parts') or []
        text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
        if not text.strip():
            raise LLMError(f'Gemini returned an empty answer (finishReason: {c.get("finishReason", "unknown")}).')
        u = j.get('usageMetadata') or {}
        return {'text': text, 'finish_reason': c.get('finishReason'), 'method': method, 'files_api_deleted': deleted,
                'model_version': j.get('modelVersion') or model,
                'usage': {k: u[k] for k in ('promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount',
                                            'totalTokenCount') if k in u}}

    def _upload(self, rq, data, mime, key, stored):
        h = {'X-Goog-Upload-Protocol': 'resumable', 'X-Goog-Upload-Command': 'start',
             'X-Goog-Upload-Header-Content-Length': str(len(data)), 'X-Goog-Upload-Header-Content-Type': mime}
        r = _post(rq, f'{BASE}/upload/v1beta/files', key, headers=h, json={'file': {'display_name': 'earshot-review'}},
                  timeout=(20, 120))
        url = r.headers.get('x-goog-upload-url')
        if not url:
            raise LLMError('Gemini Files API did not return an upload URL.')
        r = _post(rq, url, None, headers={'Content-Length': str(len(data)), 'X-Goog-Upload-Offset': '0',
                                          'X-Goog-Upload-Command': 'upload, finalize'}, data=data, timeout=(20, 900))
        f = (_json(r) or {}).get('file') or {}
        if f.get('name'):
            stored['name'] = f['name']
        for _ in range(60):
            if f.get('state') != 'PROCESSING':
                break
            time.sleep(2)
            g = rq.get(f'{BASE}/v1beta/{f["name"]}', headers={'x-goog-api-key': key}, timeout=(20, 60))
            f = _json(_check(g))
        if f.get('state') not in (None, 'ACTIVE') or not f.get('uri'):
            raise LLMError(f'Gemini Files API upload did not become ACTIVE (state {f.get("state")}).')
        return f

    def _delete(self, rq, f, key):
        try:
            return rq.delete(f'{BASE}/v1beta/{f["name"]}', headers={'x-goog-api-key': key}, timeout=(20, 60)).ok
        except Exception:  # noqa: BLE001
            return False


PROVIDERS = {'gemini': Gemini}


def _requests():
    try:
        import requests
    except ImportError as e:
        raise LLMError('the requests package is missing: pip install "earshot[llm]" (or pip install requests)') from e
    return requests


def _msg(e):
    return str(e) if isinstance(e, LLMError) else f'{type(e).__name__}: {e}'


def _json(r):
    try:
        return r.json()
    except ValueError:
        raise LLMError(f'Gemini answered HTTP {r.status_code} with something that is not JSON (a proxy or firewall '
                       f'page?): {(r.text or "")[:120]!r}') from None


def _check(r):
    if r.status_code < 400:
        return r
    try:
        msg = (r.json().get('error') or {}).get('message') or ''
    except Exception:  # noqa: BLE001
        msg = (r.text or '')[:300]
    hint = {400: 'bad request (often: the model does not accept audio, or the request is malformed)',
            401: 'the API key was rejected', 403: 'the key has no access to this API / model (or the API is disabled)',
            404: 'model not found: set EARSHOT_GEMINI_MODEL or --model to a current model id',
            413: 'request too large: use --max-seconds or --start/--end', 429: 'rate limit or quota exhausted'}
    raise LLMError(f'Gemini HTTP {r.status_code}: {hint.get(r.status_code, "request failed")}. {msg}'.strip())


def _post(rq, url, key, headers=None, timeout=(20, 300), **kw):
    h = dict(headers or {})
    if key:
        h['x-goog-api-key'] = key
    if 'json' in kw:
        h.setdefault('Content-Type', 'application/json')
    for i in range(len(RETRY_WAIT) + 1):
        try:
            r = rq.post(url, headers=h, timeout=timeout, **kw)
        except rq.Timeout:
            raise LLMError(f'Gemini did not answer within {timeout[1]} s (try a shorter clip)') from None
        except rq.RequestException as e:
            if i == len(RETRY_WAIT):
                raise LLMError(f'network error talking to Gemini: {type(e).__name__}') from None
            time.sleep(RETRY_WAIT[i])
            continue
        if r.status_code in RETRY_STATUS and i < len(RETRY_WAIT):
            time.sleep(RETRY_WAIT[i])
            continue
        return _check(r)


def encode(path, start=None, end=None, max_seconds=600.0, kbps=KBPS):
    """the audio of `path` (or [start, end)) as small mono MP3 bytes -> (bytes, mime, info)"""
    dur = float(audio.probe(path).get('duration') or 0) or audio.duration(path)
    a = max(0.0, float(start or 0.0))
    b = min(dur, float(end)) if end is not None else dur
    if b <= a:
        raise LLMError(f'empty range {a:.3f}-{b:.3f} s (file is {dur:.3f} s)')
    trimmed = b - a > max_seconds
    if trimmed:
        b = a + max_seconds
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-ss', f'{a:.3f}', '-i', path, '-t', f'{b - a:.3f}', '-vn', '-ac', '1',
           '-ar', '44100', '-c:a', 'libmp3lame', '-b:a', f'{kbps}k', '-f', 'mp3', '-']
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise LLMError(f'ffmpeg could not encode {path}: {p.stderr.decode(errors="replace")[:300]}')
    return p.stdout, 'audio/mp3', {'start': round(a, 3), 'end': round(b, 3), 'seconds': round(b - a, 3),
                                   'file_duration': round(dur, 3), 'trimmed': trimmed, 'bytes': len(p.stdout),
                                   'codec': f'mp3 mono {kbps} kbps'}


def context(path, clip, video=None, with_timeline=False, cuts=True):
    """what earshot itself measured, as plain text for the prompt (numbers the model must not guess)"""
    from . import features
    lines = [f'Clip under review: {clip["seconds"]:.1f} s (file seconds {clip["start"]:.1f}-{clip["end"]:.1f} of a '
             f'{clip["file_duration"]:.1f} s file), sent as {clip["codec"]} for this review.']
    m = features.measure(path, start=clip['start'], end=clip['end'], light=True)
    got = [f'{label} {m[k]}{unit}' for label, k, unit in MEASURED if k in m]
    if got:
        lines.append('Measured by earshot (exact; use these, never estimate loudness): ' + ', '.join(got) + '.')
    vid = video or (path if os.path.splitext(path)[1].lower() in audio.VIDEO_EXTS else None)
    notes = []
    if with_timeline:
        try:
            from .timeline import timeline
            tl = timeline(path, video=vid, start=clip['start'], end=clip['end'], detail='low')
            txt = (tl.get('text') or tl.get('summary') or '')[:16000]
            lines.append('earshot listening transcript (machine-made; file times' +
                         (f', clip starts at file second {clip["start"]:.1f}' if clip['start'] else '') + '):\n' + txt)
            vid = None
        except Exception as e:  # noqa: BLE001
            notes.append(f'the earshot timeline context was NOT sent: timeline failed ({type(e).__name__}: {e})')
    if vid and cuts:
        try:
            from . import video as vmod
            cs = [round(c - clip['start'], 2) for c in vmod.cuts(vid) if clip['start'] <= c < clip['end']]
            if cs:
                lines.append('Picture cuts / big visual changes, clip seconds (frame-difference detector: hard cuts '
                             'and whips reliable, slow crossfades missed or late): ' +
                             ', '.join(f'{c:g}' for c in cs[:200]) + '.')
        except Exception as e:  # noqa: BLE001
            notes.append(f'picture cut times were NOT sent: cut detection failed ({type(e).__name__})')
    return '\n'.join(lines), notes


SYNTH_INTENT = ('Intent: this film uses synthesized / procedural sound on purpose. Judge those sounds as designed '
                'sound; list a sound under synthetic_or_cheap only when it is cheap, harsh, generic or badly made, '
                'never only because it is synthesized.')


def build_prompt(ctx_text, question=None, synth_intended=False):
    rubric = RUBRIC % (',\n "answer": "<direct answer to the editor\'s question>"' if question else '')
    q = (f'\n\nThe editor\'s specific question (answer it in "answer"; still fill the rest):\n{question}'
         if question else '')
    intent = f'\n{SYNTH_INTENT}' if synth_intended else ''
    return f'{rubric}\n\nContext:\n{ctx_text}{intent}{q}'


def parse_json(text):
    """the JSON object inside a model answer (tolerates code fences, prose around it, trailing commas) or None"""
    s = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text or '')
    for cand in (s, _outer_object(s)):
        if not cand:
            continue
        for c in (cand, re.sub(r',\s*([}\]])', r'\1', cand)):
            try:
                j = json.loads(c)
                if isinstance(j, dict):
                    return j
            except ValueError:
                pass
    return None


def _outer_object(s):
    i = s.find('{')
    if i < 0:
        return None
    depth, instr, esc = 0, False, False
    for k in range(i, len(s)):
        ch = s[k]
        if instr:
            esc = ch == '\\' and not esc
            if ch == '"' and not esc:
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[i:k + 1]
    return s[i:]


def _sec(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r'\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*s?\s*', v)
        if m:
            return int(m.group(1) or 0) * 60 + float(m.group(2))
    return None


def _score(v):
    if isinstance(v, str):
        m = re.match(r'\s*(\d+(?:\.\d+)?)', v)
        v = float(m.group(1)) if m else None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(min(10.0, max(0.0, float(v))), 1)
    return None


def _times(items, offset, clip_len, keys):
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        a, b = _sec(it.get('start')), _sec(it.get('end'))
        if a is not None:
            a = min(max(0.0, a), clip_len)
            b = a if b is None else min(max(a, b), clip_len)
        d = {'start': None if a is None else round(a + offset, 2), 'end': None if b is None else round(b + offset, 2)}
        d.update({k: str(it.get(k, '')).strip() for k in keys})
        out.append(d)
    return out


def normalize(j, offset=0.0, clip_len=1e9):
    """make a parsed answer predictable: numeric scores, file-time seconds, issues sorted by severity then time"""
    ov = j.get('overall') if isinstance(j.get('overall'), dict) else {'score': j.get('overall')}
    sc = j.get('scores') if isinstance(j.get('scores'), dict) else {}
    notes = j.get('notes') if isinstance(j.get('notes'), dict) else {}
    issues = _times(j.get('issues'), offset, clip_len, ('severity', 'category', 'problem', 'fix'))
    for it in issues:
        it['severity'] = it['severity'].lower() if it['severity'].lower() in SEV else 'medium'
    issues.sort(key=lambda d: (SEV[d['severity']], d['start'] if d['start'] is not None else 1e9))
    r = {'overall': {'score': _score(ov.get('score')), 'verdict': str(ov.get('verdict') or '').strip()},
         'scores': {k: _score(sc.get(k)) for k in SCORE_KEYS},
         'notes': {k: str(notes.get(k) or '').strip() for k in SCORE_KEYS if notes.get(k)},
         'synthetic_or_cheap': _times(j.get('synthetic_or_cheap'), offset, clip_len, ('what', 'why')),
         'issues': issues,
         'strengths': [str(s) for s in j.get('strengths') or [] if str(s).strip()][:8],
         'top_fixes': [str(s) for s in j.get('top_fixes') or [] if str(s).strip()][:5]}
    if j.get('answer'):
        r['answer'] = str(j['answer']).strip()
    return r


def _span(it):
    if it['start'] is None:
        return '(no time)'
    return fmt_t(it['start']) if it['end'] in (None, it['start']) else f'{fmt_t(it["start"])}-{fmt_t(it["end"])}'


def render(r):
    """agent-readable text of a critique result"""
    lines = [r['summary']] + [f'(note: {n})' for n in r.get('notes') or []]
    c = r.get('critique')
    if not c:
        return '\n'.join(lines + ([r['raw'][:3000]] if r.get('raw') else []))
    if c.get('answer'):
        lines.append(f'answer: {c["answer"]}')
    lines.append('scores (1-10, LLM opinion): ' + ', '.join(f'{k} {v:g}' for k, v in c['scores'].items()
                                                            if v is not None))
    for k, v in c['notes'].items():
        lines.append(f'  {k}: {v}')
    if c['issues']:
        lines.append('issues (most severe first; times = file seconds, approximate):')
        lines += [f'  {_span(i)} {i["severity"]} {i["category"]}: {i["problem"]} -> fix: {i["fix"]}'
                  for i in c['issues']]
    if c['synthetic_or_cheap']:
        lines.append('sounds that read as synthetic / cheap:')
        lines += [f'  {_span(i)} {i["what"]}: {i["why"]}' for i in c['synthetic_or_cheap']]
    if c['strengths']:
        lines.append('strengths: ' + '; '.join(c['strengths']))
    if c['top_fixes']:
        lines.append('top fixes: ' + ' | '.join(f'{n}. {f}' for n, f in enumerate(c['top_fixes'], 1)))
    return '\n'.join(lines)


def _scrub(s, key):
    return s.replace(key, '***') if key and isinstance(s, str) else s


def _off(path, provider, why, steps):
    r = {'enabled': False, 'ok': False, 'path': path, 'provider': provider.name, 'model': provider.model(),
         'summary': why, 'how_to_enable': steps}
    r['text'] = why + '\n' + '\n'.join(f'  {n}. {s}' for n, s in enumerate(steps, 1))
    return r


def critique(path, prompt=None, provider='gemini', model=None, consent=False, video=None, with_timeline=False,
             start=None, end=None, max_seconds=600.0, cuts=True, synth_intended=False):
    """audio-LLM critique of a soundtrack against the film-sound rubric (only with an API key AND consent).
    synth_intended=True tells the model that synthesized / procedural sound is the point of this film, so synthesis
    itself is not counted as a flaw (the rubric also says so when the editor's question mentions it)"""
    if provider not in PROVIDERS:
        raise ValueError(f'unknown provider {provider!r}; available: {", ".join(PROVIDERS)}')
    if not os.path.isfile(path):
        raise audio.AudioError(f'no such file: {path}')
    pv = PROVIDERS[provider]()
    key = pv.key()
    if not key:
        return _off(path, pv, 'LLM critique is off: no GEMINI_API_KEY or GOOGLE_API_KEY in the environment. Everything '
                    'else in earshot works without it.',
                    [f'Create a Gemini API key at {KEY_URL} (the user\'s own account; usage may be billed).',
                     'Set it in the shell that runs earshot: PowerShell $env:GEMINI_API_KEY="..." / bash export '
                     'GEMINI_API_KEY=... (never paste a key into code, files or chat).',
                     'Rerun with --yes-upload (consent=True), only if the user agrees that the audio is uploaded '
                     'to Google.'])
    if not consent:
        return _off(path, pv, f'Not uploaded: this would send the audio of {os.path.basename(path)} to Google Gemini '
                    f'({pv.model(model)}). It needs explicit consent.',
                    ['Ask the user whether this audio may be uploaded to Google for an AI critique.',
                     'If yes, rerun with --yes-upload (CLI) or consent=True (Python / MCP).'])
    m = pv.model(model)
    t0 = time.time()
    clip = None
    try:
        data, mime, clip = encode(path, start, end, max_seconds)
        ctx_text, notes = context(path, clip, video, with_timeline, cuts)
        out = pv.generate(data, mime, build_prompt(ctx_text, prompt, synth_intended), m, key)
    except Exception as e:  # noqa: BLE001 - any failure becomes a readable result, never a traceback with the key
        msg = _scrub(_msg(e), key)
        return {'enabled': True, 'ok': False, 'path': path, 'provider': pv.name, 'model': m, 'uploaded': clip,
                'summary': f'LLM critique failed: {msg}', 'text': f'LLM critique failed: {msg}', 'error': msg}
    j = parse_json(out['text'])
    r = {'enabled': True, 'ok': j is not None, 'path': path, 'provider': pv.name, 'model': out['model_version'],
         'uploaded': dict(clip, method=out['method']), 'seconds': round(time.time() - t0, 1), 'usage': out['usage'],
         'finish_reason': out['finish_reason'], 'context_sent': ctx_text, 'notes': notes}
    if out.get('files_api_deleted') is not None:
        r['uploaded']['files_api_deleted'] = out['files_api_deleted']
    trim = (f' (trimmed to the first {clip["seconds"]:g} s; use --start/--end for other parts)'
            if clip['trimmed'] else '')
    if j is None:
        r['raw'] = _scrub(out['text'], key)[:20000]
        r['summary'] = (f'{pv.name} answered but not in parseable JSON (finish {out["finish_reason"]}); raw answer '
                        f'attached{trim}.')
    else:
        c = r['critique'] = normalize(j, clip['start'], clip['seconds'])
        high = sum(1 for i in c['issues'] if i['severity'] == 'high')
        ov = c['overall']
        lead = f'{ov["score"]:g}/10' if ov['score'] is not None else 'unscored'
        verdict = (ov['verdict'] or 'no verdict').rstrip()
        verdict += '' if verdict[-1] in '.!?' else '.'
        r['summary'] = (f'{pv.name} ({r["model"]}) rates the soundtrack {lead}: {verdict} '
                        f'{len(c["issues"])} issues ({high} high).' +
                        (f' Top fix: {c["top_fixes"][0]}' if c['top_fixes'] else '') +
                        f' LLM opinion, timestamps approximate; verify with earshot timeline/ab{trim}.')
    r['text'] = render(r)
    return r


def add_cli(sub):
    from .cli import common
    p = common(sub.add_parser('llm', help='optional audio-LLM critique (Gemini; needs your key + --yes-upload)',
                              description='Second opinion from a hosted audio model. Uploads the audio to Google: '
                              'needs GEMINI_API_KEY/GOOGLE_API_KEY and --yes-upload.'))
    p.add_argument('file', help='soundtrack, stem or video (only its audio is sent)')
    p.add_argument('--prompt', help='a specific question to answer in addition to the rubric')
    p.add_argument('--yes-upload', action='store_true', help='consent: upload this audio to Google Gemini')
    p.add_argument('--model', help=f'Gemini model id (default EARSHOT_GEMINI_MODEL or {DEFAULT_MODEL})')
    p.add_argument('--video', help='the film, for picture-cut context (only cut times are sent, never frames)')
    p.add_argument('--with-timeline', action='store_true',
                   help="also send earshot's listening transcript (loads models)")
    p.add_argument('--no-cuts', action='store_true', help='do not compute picture cuts for a video')
    p.add_argument('--start', type=float, help='send only from here (s)')
    p.add_argument('--end', type=float, help='send only up to here (s)')
    p.add_argument('--max-seconds', type=float, default=600.0, help='trim longer audio to this (default 600)')
    p.add_argument('--synth-intended', action='store_true',
                   help='the film uses synthesized / procedural sound on purpose: do not count synthesis as a flaw')
    p.set_defaults(func=lambda a: 0 if emit(critique(a.file, a.prompt, model=a.model, consent=a.yes_upload,
                                                     video=a.video, with_timeline=a.with_timeline, start=a.start,
                                                     end=a.end, max_seconds=a.max_seconds, cuts=not a.no_cuts,
                                                     synth_intended=a.synth_intended),
                                            a)['ok'] else 1)
