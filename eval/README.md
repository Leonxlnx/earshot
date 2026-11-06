# earshot evaluations

Each folder holds the scripts that measured one part of earshot, the raw numbers (`*.json`) and a written report
(`RESULTS.md`) with the exact commands. The headline numbers in the main README and in `docs/VALIDATION.md` come from
here.

| folder | what it measures | report |
|---|---|---|
| `semantics/` | text-audio matching (`describe`, `match`, `rank`, `find`): verdict precision, likelihood ratios, top-k retrieval on a tuning library and a held-out library labelled before scoring, query-kind reading, find trims, bf16 vs fp32 precision, timing | [RESULTS.md](semantics/RESULTS.md) |
| `timeline/` | the listening transcript and the sheet: event recall on a real 60 s film mix with known placements (quiet events under music too), tags, sections, tempo / beat grid, sync of hits to picture cuts and cue frames, run time on 30-90 s films | [RESULTS.md](timeline/RESULTS.md) |
| `realness/` | the real-vs-code-synthesized detector: held-out ROC-AUC with whole sound families held out, robustness to noise, room, gating, length and mp3, firm-verdict error rates, a tone guard on real recordings | [RESULTS.md](realness/RESULTS.md) |
| `launch/` | the launch profile and `critique` scorecard: leave-one-out over the 7 reference films, a film with a real vs a synthesized soundtrack, known-bad mixes (quiet, hot, mono, muffled, late, dead air, fade-out, hole, over-compressed ...), the visual-event detector vs scene cuts | [RESULTS.md](launch/RESULTS.md) |
| `forge/` | the forge's sound recipes against real references (`bench.py`, `blind_v2.py`); `replicate.py`, the replication benchmark of `earshot fit`: every clip of a folder of film sound effects fitted with `--clap`, its seed-0 render and four takes measured against the clip (CLAP, mel gap over the model's own floor, attack, width, glide, tail, pulses, gate, the take check), on a fixed 70 / 30 dev / held-out split by clip id (targets from `--targets` or `EARSHOT_REPLICATE_TARGETS`, results to a folder outside the repo or under `out/`) | [RESULTS.md](forge/RESULTS.md) |

## The data is not redistributed

The evaluations ran on licensed recordings and third-party films. None of it is in this repo, and none of it may be
redistributed from here. Only derived numbers are committed. What was used:

- **A library of real licensed recordings** (Freesound / Mixkit) in 8 categories: ambience, foley, garden, keyboard,
  magic, motion, music, toy. 219 files were used for tuning; 126 real recordings that were rejected for quality form
  the held-out set. Each file has a manifest entry (category, title, need, notes).
- **A finished short film project** with a real-recordings soundtrack: the 60 s mix, its stems (sfx, ambience, music),
  a mix report with every placement (time and sample), picture cue events at 60 fps, and the same film with an older
  code-synthesized soundtrack and its stems.
- **Code-synthesized one-shots** rendered with an open synth engine (kick, clap, tick, key click, UI click, FM bell,
  whoosh, riser, drone ...).
- **7 professional launch films** as the quality bar, measured only. They are called `ref_1` .. `ref_7` everywhere,
  with a neutral style line each in `launch/refs.json`. The shipped profile (`earshot/data/launch_profile.json`) holds
  their numbers only: no media, no file names.

## Running the scripts on your own data

The scripts read the data locations from environment variables, or from a gitignored `eval/local_paths.json`
(environment variables win). See `evalpaths.py` for the defaults.

| variable | `local_paths.json` key | what it points to |
|---|---|---|
| `EARSHOT_EVAL_ASSETS` | `assets` | a library of real recordings: `<category>/`, `_rejected/<category>/`, `manifests/*.json` |
| `EARSHOT_EVAL_FILM` | `film_project` | a film project: `out/audio/soundtrack.wav`, `out/audio/stems_pro/{sfx,amb,music}.wav`, `out/qa/mix_report.json`, `out/cues_all.json`, the film as `out/Clawd_1080p60.mp4`, the synthesized versions under `audio/_retired_synth/`, `out/audio/stems/` and `out/superseded/` |
| `EARSHOT_EVAL_REFS` | `refs_dir`, or `refs: {"ref_1": "/any/file.mp4", ...}` | your reference launch films, named `ref_1.mp4` .. `ref_7.mp4` |
| `EARSHOT_EVAL_SYNTH` | `synth_scripts` | a folder with the synth engine scripts (`synth.py`, `score.py`), only for `realness/build_data.py` |

Example `eval/local_paths.json`:

```json
{"assets": "/data/sfx-library",
 "film_project": "/data/my-film",
 "refs": {"ref_1": "/data/films/launch_a.mp4", "ref_2": "/data/films/launch_b.mp4"}}
```

To build a launch profile from films you admire you do not need any of this: `earshot profile a.mp4 b.mp4 ... -o
my_profile.json`, then `earshot critique film.mp4 --profile my_profile.json`.

Run every script from the repo root with the repo's Python, e.g.
`EARSHOT_THREADS=2 .venv/bin/python eval/launch/run_eval.py --help`. Scratch audio goes to `out/` (gitignored). Your
numbers will differ from the committed ones unless your data matches the original.
