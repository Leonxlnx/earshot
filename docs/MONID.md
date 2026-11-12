# Voice previews through Monid

Monid lets agents and developers call data providers with one API key, pay per call.
This standalone example calls Fish Audio voice design through Monid and prints
its JSON output, including the candidate WAV preview links.
It does not change earshot's commands or require a Fish Audio key.

## Run

Use an earshot installation with its existing `llm` extra (which supplies requests).
Set your Monid key in the shell, then run the example from the repository root:

```bash
export MONID_API_KEY='your-key'
python docs/monid_voice_design.py 'Warm, confident studio narrator with a natural tone'
```

With no key, the example exits with a message before making a request.
The voice description is sent to Fish Audio through Monid.
A request can return immediately or start a run; the example polls for up to
60 seconds and reports HTTP errors, failed runs and Fish Audio errors.

## Listen with earshot

Download a candidate WAV from the returned link, then measure it locally using
earshot's public Python API (no models needed):

```python
from earshot import api

print(api.measure('candidate.wav', light=True))
```

The generated preview is a new voice candidate, separate from earshot's local
sound bank and analysis features.

[Monid docs](https://docs.monid.ai?utm_source=github&utm_medium=oss-pr&utm_campaign=leonxlnx-earshot)
· [Get a key](https://monid.ai?utm_source=github&utm_medium=oss-pr&utm_campaign=leonxlnx-earshot&aid=OSS-PR)
