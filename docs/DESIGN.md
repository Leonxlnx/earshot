# earshot: design notes

earshot gives coding agents "ears": an agent cannot hear, so earshot turns sound into things an agent can reason
about: calibrated text-audio match scores, the best moment inside a long recording, similarity to real reference
recordings, real-vs-synthesized judgement, a second-by-second listening transcript of a soundtrack, an annotated
picture of it (agents can look at images), a launch-video scorecard against professional references, and (only with
the user's own key) an audio-LLM critique. Everything runs on CPU; no GPU is ever used.

The repo also ships `skills/real-sound/`: the agent skill for building premium soundtracks for code-made films from
real licensed recordings (sourcing + licence audit + design JSON + mixer + mastering + credits), which uses earshot to
pick and verify sounds.

## Principles
1. **Honest numbers.** Every score says what it is (cosine, z vs background, probability) and how much to trust it.
   Never present CLAP cosine as a probability. Validation results live in `docs/VALIDATION.md` with the exact commands.
2. **Agent-readable first.** Every command prints a short plain-language `summary` first, then details; `--json` for
   the full dict. Timestamps as seconds (3 decimals) and m:ss.ss in text. Levels in dBFS / LUFS / dBTP.
3. **Deterministic and cached.** No random crops; fixed windows; embeddings cached on disk (EARSHOT_HOME/emb).
4. **Gentle on the machine.** CPU only, bf16 by default, `EARSHOT_THREADS` (default min(4, cores/2)). Batch work.
   This laptop (Ryzen 7 260, 8 cores / 16 threads) is shared with other jobs: when developing set `EARSHOT_THREADS=3`,
   keep any single job under ~10 minutes, run jobs in the foreground, one heavy job at a time per agent.
5. **No media in git.** Never commit audio/video (reference films, downloaded recordings, renders). Derived numbers
   (profiles, trained weights, eval tables) are fine.
6. **No synthesized deliverables.** earshot may synthesize test signals for tests/evaluation only.

## Layout and ownership (one owner per file during the build)
| file | owner | purpose |
|---|---|---|
| `earshot/config.py`, `audio.py`, `features.py`, `video.py`, `report.py`, `cli.py`, `doctor.py`, `__init__.py`, `models/clap.py`, `models/ast.py` | core (lead) | shared base: read-only for builders. Need a change? Work around it in your module and report it. |
| `earshot/data/vocab.json`, `earshot/vocab.py`, `describe.py`, `match.py`, `search.py`, `compare.py` | semantics | describe / match / rank / find / compare / ab |
| `earshot/timeline.py`, `earshot/sheet.py` | timeline | listening transcript, sync analysis, annotated PNG |
| `earshot/realness.py`, `earshot/data/realness*.json`, `eval/realness/*` | realness | real vs synthesized detector + its evaluation |
| `earshot/profile.py`, `earshot/critique.py`, `earshot/data/launch_profile.json`, `docs/launch-video-sound.md` | launch | reference profile from pro launch films + scorecard |
| `earshot/llm.py`, `earshot/mcp_server.py`, `skills/earshot/SKILL.md` | interfaces | optional audio-LLM critique, MCP server, agent skill for using earshot |
| `skills/real-sound/**`, `docs/INTEGRATION.md` | real-sound | the soundtrack skill (moved from the Clawd project) + Cinetic integration notes |
| `tests/test_<module>.py` | that module's owner | fast tests (synthetic fixtures, no network). Model tests marked `@pytest.mark.slow`. |
| `README.md`, `docs/VALIDATION.md`, `docs/LIMITS.md` | lead (integration) | builders send their numbers in their final report |

## Core API (already built; use it)
- `earshot.audio`: `load(path, sr=48000, mono=False, start=None, end=None)` -> float32 (n,2)|(n,), any ffmpeg-readable
  file incl. the audio of mp4; `probe(path)`; `windows(x, win, hop)`; `list_media(paths)`; `resample`; `to_mono`;
  `file_id(path)`; `SR = 48000`.
- `earshot.features`: `measure(path_or_array, music=False, light=False)` -> dict (see module doc: lufs, true peak,
  noise floor, digital_silence_frac, attack_ms, decay_ms, centroid, bands_db, flatness, tonal_peak_ratio_db,
  harmonic_ratio, f0, onsets, tempo/beats/key); `loudness_curve(x, win, hop)`; `integrated_lufs`; `lra`; `onsets`.
- `earshot.models.clap.get()` -> `Clap`: `embed_text(list)` (n,512) unit; `embed_audio(list_of_arrays, pad=
  'repeatpad'|'zero', per_window=False)`; `embed_file(path, start, end, pad)` (cached); `embed_files(paths)`;
  `logit_scale` (~38.7). ~0.3-1.2 s per 10 s window on 4 threads (bf16). Every input costs one 10 s window (short
  sounds are padded), so batch and avoid needless tiny windows.
- `earshot.models.ast.get()` -> `AST`: `probs(list_of_arrays, reduce='max'|'mean')` (n,527) sigmoid; `probs_file`;
  `top(p, k)`; `labels`; `prob_of(p, label)`. AudioSet ontology labels. Hears up to 8 kHz (16 kHz input).
- `earshot.video`: `cuts(path, threshold=0.3)`, `frame(path, t, width)`, `frames`, `info`, `has_video`.
- `earshot.report`: `emit(result, args)`, `fmt_t`, `to_json`. `earshot.cli.common(parser)` adds `--json --out`.

## Public functions (the contract other modules and the MCP server rely on)
All return JSON-serialisable dicts with a `summary` string. Paths may be audio or video files.
- `describe(path, start=None, end=None, top=8)` -> tags per vocab category with calibrated scores, AST top labels,
  key measurements, one-paragraph `summary`.
- `match(path, query, alts=None, start=None, end=None)` -> `{'query', 'cosine', 'z', 'percentile', 'rank_in_vocab',
  'p_vs_alts' (softmax over query+alts with logit_scale), 'verdict' ('strong'|'plausible'|'weak'|'no match'), ...}`.
- `rank(paths, query, top=None)` -> files sorted by calibrated match. `find(path, query, win=None, hop=None, top=5,
  start=None, end=None, kind='auto')` -> best segments inside a long recording `[{'start','end','score','z','kind',
  'head_db','tail_db','cut'}]`: sample trims, except one marked 'head cut' / 'tail cut'.
- `compare(path, refs, k=5, start=None, end=None)` -> nearest references + typicality vs the reference set;
  `ab(a, b, query=None, refs=None, k=5, a_start=None, a_end=None, b_start=None, b_end=None)` -> which is closer to the
  query / refs and how they differ (feature deltas in words); the trims judge only part of a file.
- `realness(path)` -> `{'p_real', 'verdict', 'evidence': [...]}` (+ per-segment for long files).
- `timeline(path, video=None, cues=None, ...)` -> events (time, level, tags, AST labels), windows tags, loudness
  curve, cuts, sync stats (event-to-cut / event-to-cue offsets), `text` = the listening transcript.
- `sheet(path, out_png, video=None, ...)` -> writes an annotated PNG an agent can open; returns its path + summary.
- `critique(path, profile=None, video=None)` -> scorecard: each metric, the soundtrack's value, the reference range,
  status (ok/low/high), what to do.
- `profile` CLI builds a profile JSON from a folder of reference videos/soundtracks.
- `llm.critique(path, prompt=None, provider='gemini')` -> only when GEMINI_API_KEY / GOOGLE_API_KEY is set and the
  caller passes consent (`--yes-upload`); otherwise a clear message how to enable it. Never hardcode keys.

CLI: each module implements `add_cli(sub)` registering its subcommand(s) (names in `earshot/cli.py COMMANDS`; `search`
registers `rank` and `find`; `compare` registers `compare` and `ab`). Use `from .cli import common` for `--json/--out`.

## Evaluation data (not redistributed)
Evaluation data: licensed recordings and third-party films, not redistributed. The evaluation scripts in `eval/`
read their locations from environment variables or a gitignored `eval/local_paths.json` (see `eval/README.md`).
What was used:
- A library of real licensed recordings (Freesound / Mixkit) in 8 categories (ambience, foley, garden, keyboard,
  magic, motion, music, toy): 219 files used for tuning, plus 126 real recordings rejected for quality, used as the
  held-out set. Labels come from per-file manifests (category, title, need, notes).
- Code-synthesized audio: a retired synthesized film score with its stems, and one-shots rendered by an open synth
  engine (kick, sub_boom, clap, hat, tick, tock, key_click, thock, ui_click, snap, blip, fm_bell, whoosh, riser,
  swell, pluck, bass, drone, keys ...).
- A finished 60 s real-recordings film soundtrack with ground truth: stems (sfx, amb, music), every placement with
  time and sample in a mix report, picture cue events at 60 fps, and the same film with the old synthesized
  soundtrack.
- 7 professional launch films as the quality bar (measured only, never copied), referred to as ref_1..ref_7 with
  neutral style descriptions in `eval/launch/refs.json`.

## Environment
- Python 3.10+ with torch (CPU), transformers, librosa, scikit-learn, mcp. Run from the repo root (package importable
  as `earshot`), e.g. `.venv/bin/python -m earshot describe file.wav`.
- Models are cached in `.cache/hf` inside the repo or `EARSHOT_HOME` (CLAP laion/larger_clap_general, AST
  MIT/ast-finetuned-audioset).
- Scratch output: `out/<area>/` (gitignored).
- ffmpeg / ffprobe on PATH.
