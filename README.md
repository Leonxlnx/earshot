# earshot

**Ears for coding agents.**

<!-- launch video -->
[**Watch the 22 s launch film**](https://github.com/Leonxlnx/earshot/releases/tag/v0.1.0) - every sound in it was made
in code by `earshot forge`; no recordings, no music track.

![Frames from the launch film](docs/media/launch_frames.png)

AI coding agents can now write the code that renders a film, a product demo or a game. They still cannot hear what
they made. They cannot tell a premium key press from a cheap one, notice that a hit lands 80 ms after the cut, or
hear that the music swells in the wrong place.

earshot turns sound into evidence an agent can reason about: what a file sounds like in words, how well it fits a
description, the best moment inside a long recording, whether it is a real recording or code-synthesized, a
second-by-second listening transcript of a whole film, an annotated picture of it, and a scorecard against
professional launch films. It runs locally on CPU, as a CLI, a Python API and an MCP server.

With ears, an agent can also **make its own sounds in code** and improve them by listening (`earshot forge`, below).

```
$ earshot describe enter.ogg
  0.56 s short transient (attack 3 ms, decays in 190 ms). Sounds most like: a single mechanical keyboard key press
  (z +5.4), a soft key release click (z +3.6) ... [clear reading]. peak -0.1 dBFS, brightness 1.5 kHz ...

$ earshot match key.ogg "a deep thocky mechanical keyboard press" --alts "a clicky blue switch keyboard" "a mouse click"
  -> strong (z +3.7; would rank #2 among 322 vocabulary descriptions) ...

$ earshot find keys.ogg "a single mechanical key press" --top 4
  0:03.94-0:04.19 (0.24 s stroke, z +4.1 strong); 0:10.60-0:10.86 ...

$ earshot critique film.mp4
  VERDICT: close - 21 of 23 scored metrics in range ... FIX FIRST: 1. [major] nothing reaches body level until 0:01.01 ...
```

## What it does

| command | what an agent gets |
|---|---|
| `describe FILE` | what it sounds like: calibrated tags (z) with the closest alternative readings, AudioSet labels, measurements |
| `match FILE QUERY [--alts ...]` | how well a description fits: cosine, z among the vocabulary, a verdict (strong / plausible / weak / no match) with held-out evidence |
| `rank QUERY PATHS...` | many candidate files sorted by a description (pick the best download) |
| `find FILE QUERY` | the best moments inside a long recording, as sample trims (start / end seconds) |
| `compare FILE --refs DIR` / `ab A B` | nearest reference recordings and typicality; A/B two sounds or two trims, the differences in words |
| `realness FILE...` | real recording / pro-produced vs code-synthesized, with evidence and per-window scores |
| `timeline FILE [--video V] [--cues C]` | the listening transcript: events (time, level, tag), sections, loudness, beat grid, sync of sound to picture cuts and cue frames |
| `sheet FILE -o out.png` | an annotated PNG (video frames, sections, labelled events, spectrogram, waveform, loudness, sync markers) for agents that read images |
| `critique FILE [--video V]` | a launch-video scorecard against a profile of 7 professional launch films, with timestamped fixes |
| `profile REFS... -o p.json` | build your own reference profile from films you admire |
| `measure FILE` | objective measurements only (no models): LUFS, true peak, LRA, envelope, spectrum, tempo / key |
| `forge RECIPE --refs DIR` | make a sound in code and improve it by listening (see below) |
| `llm FILE --yes-upload` | optional audio-LLM critique (Gemini), only with your own key and explicit consent |
| `mcp` | the same tools as an MCP server for Claude Code and other agents |
| `doctor [--load]` | check the install, models, data files and CPU settings |

Every command prints a plain-language summary first; `--json` gives the full result, `--out f.json` saves it. Errors
are one line on stderr (`earshot <command>: error: ...`, exit 1). Each CLI call is a new process and loads the models
(~20-40 s) before it answers: for more than a few calls use the MCP server (models stay loaded) or one Python script
with `from earshot import api`.

What an agent sees when it listens to that film with `earshot sheet` (frames, labelled sound events, spectrogram,
loudness, sync to the beat):

![earshot sheet of the launch film](docs/media/launch_sheet.png)

## Make sounds: earshot forge

Listening makes iteration possible. The agent writes a sound as code: a **recipe** of physical synthesis (modal
resonators, contact excitations, filtered noise, a computed room and a microphone; no recorded audio inside). The
forge renders candidates from the recipe's parameters, and a **critic** built from earshot scores each one:
description match against named confusions, realness, closeness to real reference recordings, measured shape, and a
model-free tonal-balance and tail check against the references. An evolutionary **search** keeps the best and mutates
them. Then the agent reads the critic's words ("sounds like: a cork pop, a glass ting", "decay +2.4 outside the
refs"), looks at spectrograms next to the references, rewrites the recipe, and runs it again.

```bash
earshot forge                                    # list the recipes: click, impact, keypress, pop, whoosh
earshot forge keypress --refs path/to/real/keypresses/ --budget 144 --reps 2 --tone 1.5 --keep 4 --robust 8 --dir made/keypress
earshot forge whoosh --target "a fast crisp air whoosh" --alts "a wind gust" "a sword swing" \
  --refs path/to/real/whooshes/ --tone 1.5 --dir made/whoosh
```

Output: the best diverse sounds as WAVs plus `report.json` with every candidate's parameters and scores, the same
critic's scores of the real references (leave-one-out), held-out AudioSet tags, and with `--robust N` each pick
re-rendered on N new seeds. Every score also carries model-free measurements (`earshot/forge/tone.py`): the
1/3-octave tonal balance against the references' median ("+16 dB at 500-630 Hz (above every ref)"), the tail and the
length. In Python: `from earshot.forge import forge`.

**Blind test.** A key press, a UI click, a pop, a fast whoosh and an impact were made from scratch in code and
played blind next to real recordings of the same kind of sound. The author picked the made one correctly **8 times out
of 15**: a coin flip. Only the made impact was spotted (its tail dies too fast). One listener and 15 trials is a small
test, not a proof; it says the made sounds are no longer easy to tell from real ones.

**Honest limits (Goodhart).** The search optimises earshot's own models, and those can be pleased without the sound
getting better. After a search the made sounds pass every model check, held-out ones included, so the model checks
stop telling made from real; the 1/3-octave measurement still can. The realness model is easy to fool (a modal
resonator plus a room and a noise floor reads "real"). Judge a made sound by the model-free measurements,
spectrograms next to the references, robust re-renders, and a human blind test.

## Install

Python 3.10+ and ffmpeg / ffprobe on PATH. CPU only; no GPU is used.

```bash
git clone https://github.com/Leonxlnx/earshot && cd earshot
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows: .venv/Scripts/python
.venv/bin/python -m pip install -e ".[mcp,llm,dev]"
.venv/bin/python -m earshot doctor --load
```

The first model use downloads two open models (~1.4 GB) into `.cache/hf` inside the repo (or `EARSHOT_HOME`): CLAP
`laion/larger_clap_general` (text-audio embeddings) and the AudioSet tagger `MIT/ast-finetuned-audioset-10-10-0.4593`.
After that earshot works offline.

Settings (environment variables): `EARSHOT_THREADS` (torch CPU threads, default min(4, cores/2)), `EARSHOT_HOME`
(cache root), `EARSHOT_PRECISION` (`bf16` by default, used only on CPUs with native bf16 such as AVX512-BF16 / AMX,
fp32 everywhere else; `fp32` forces fp32), `EARSHOT_OFFLINE=1`. `earshot doctor --load` reports what is in use.

### MCP server for Claude Code

```bash
claude mcp add earshot --scope user -e EARSHOT_THREADS=3 -- /path/to/earshot/.venv/bin/python -m earshot mcp
```

Or in any client's `.mcp.json`:

```json
{"mcpServers": {"earshot": {"command": "/path/to/earshot/.venv/bin/python", "args": ["-m", "earshot", "mcp"],
  "env": {"EARSHOT_THREADS": "3"}}}}
```

Models load on the first call and stay warm. On a slow machine start Claude Code with `MCP_TIMEOUT=60000`. The
optional `llm_critique` tool reads `GEMINI_API_KEY` from the environment; never write a literal key into a file.

### Python

```python
import earshot
earshot.describe('whoosh.wav')
earshot.find('typing.ogg', 'a single isolated key press')
earshot.timeline('soundtrack.wav', video='film.mp4')
from earshot import api   # everything in one namespace: api.rank, api.critique, api.realness_many, ...
```

## Agent skills

Two skills ship in `skills/`. Copy or symlink them into `~/.claude/skills/` (or a project's `.claude/skills/`).

- **`skills/earshot`**: how an agent should listen with earshot: which command answers which question, how to read
  the verdicts, when not to trust them.
- **`skills/real-sound`**: a full process for premium, realistic soundtracks for code-made films from real licensed
  recordings: sourcing with licence checks, a music edit, a JSON sound design, a deterministic mixer, measured QA, a
  multi-lens critique, mastering, credits and mux. earshot chooses, trims and verifies the sounds.

## How good is it?

Every number below comes with the command that measured it in [docs/VALIDATION.md](docs/VALIDATION.md). What
earshot cannot judge is in [docs/LIMITS.md](docs/LIMITS.md).

- **Matching descriptions** (a held-out library of 125 real recordings, nothing tuned on it): in a library search a
  "strong" verdict is the right kind of sound 98% of the time and the exact variant 61% of the time.
- **Finding the moment**: on a recording of 17 single key presses the found trims land on real key presses; in a
  300 s scene every planted event is found as a tight trim.
- **Real vs code-synthesized**: held-out ROC-AUC 0.92 with whole sound families held out. Firm verdicts call 6.7% of
  real clips synthesized and 5.5% of synthesized clips real; a fifth of clips get "uncertain".
- **Listening through a soundtrack**: 93% of the sound effects placed in a 60 s film mix are detected as events (89%
  of the quiet ones under music). A 60-90 s film takes about 1.5 minutes on a laptop CPU.
- **Launch scorecard**: each reference film critiqued against the other six falls in range on 86% of the metrics
  (the no-overfit baseline is about 71%; some rules changed after the first run, so it is not a blind test).

## Limits

earshot hears through two open models and a set of measurements. Trustworthy: loudness, timing and sync to picture,
"is this the kind of sound I asked for" on single exposed sounds, picking among candidates, obvious code-synthesized
sources. Hints only: tags of events inside a busy mix, music genre and mood. It cannot judge taste or emotion,
provenance or licences, speech content, or anything above 14 kHz from the models. The launch scorecard describes the
style of the 7 films behind it, not every good film. Details in [docs/LIMITS.md](docs/LIMITS.md).

## Repo map

```
earshot/            the package: CLI, Python API, MCP server, CPU model wrappers
  forge/            make sounds in code: DSP building blocks, critic, model-free tone checks, search, recipes/
  data/             vocabulary + calibration stats, realness model weights, launch profile (numbers only, no media)
skills/earshot/     agent skill: how to listen with earshot
skills/real-sound/  agent skill: real-recordings soundtracks (scripts, templates, prompts, references)
docs/               VALIDATION, LIMITS, launch-video sound playbook, integration notes, design notes
eval/               evaluation scripts and results per area (see eval/README.md)
tests/              pytest (fast by default; --slow loads the models)
```

No audio or video is stored in this repo. The evaluation data (licensed recordings and third-party films) is not
redistributed; only derived numbers are committed.

## Licence

MIT for the code (see [LICENSE](LICENSE)). The models are downloaded from their authors under their own licences:
CLAP (`laion/larger_clap_general`) Apache-2.0, AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) BSD-3-Clause.
`skills/real-sound/scripts/master.py` is vendored from the Cinetic skill by the same author.
