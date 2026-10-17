# Changelog

## 0.2.1 - 2026-10-17

Two fixes from a careful outside review (thanks, @markblundeberg).

### Fixed

- **Agent-facing text no longer states one user's preferences as instructions** (#1). The `realness` tool description
  and the MCP server instructions told every agent to check every sound because "the user rejects synthetic-sounding
  audio"; the earshot skill said to reject 'likely code-synthesized' sounds outright, and the real-sound skill and
  several docs called the bundled launch-film profile "the user's". They now say what realness measures and leave the
  decision to the current user: use it when real or professionally produced sound is wanted, and treat a synthesized
  verdict as information when synthesized sound is the point. `critique`'s realness fix is scoped the same way. A
  test keeps such phrasing out of tool descriptions, the server instructions, skills and docs.
- **Model loading respects the user's offline setting, never downloads after a failed load, and pins its weights**
  (#2). Before, earshot set `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` once a model's cache folder existed, cleared
  them and downloaded again after any failed local load (a damaged cache, a transformers mismatch, a typo in
  `EARSHOT_CLAP`), and fetched whatever each repo's main held, for CLAP a pickle. Now (`earshot/models/hub.py`):
  - `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` mean the same as `EARSHOT_OFFLINE=1`: no download, a clear error
    naming the flag when a model is not cached. earshot never sets or clears any of them.
  - Models load from a complete local snapshot folder only (transformers gets a folder, not a repo id), so a cached
    load never contacts the Hub (no update checks, no safetensors conversion requests). A failed load is an error
    that names the folder; the only download is for files that are not in the cache.
  - Pinned revisions with safetensors weights: CLAP `16c8cc3` (the Hub's safetensors conversion of main `ada0c23`),
    AST `f826b80`; overrides `EARSHOT_CLAP_REVISION` / `EARSHOT_AST_REVISION`. A cache made before 0.2.1 (CLAP
    `ada0c23`, `pytorch_model.bin`, the same tensors) keeps working from the cache, without a forced 776 MB download.
  - `earshot doctor` reports, per model, the pinned revision, what the cache holds (cached / legacy / incomplete /
    missing), the folder that loads and the offline flag; `--load` adds the revision and weights file that loaded.
    The first download is ~1.1 GB (the docs said 1.4 GB).

### Added

- `earshot doctor --fetch`: download the pinned model files that are not in the cache (also upgrades a legacy CLAP
  cache to the pinned safetensors).

### Changed

- A custom model (`EARSHOT_CLAP` / `EARSHOT_AST`) or another revision keeps its embedding caches per commit, so they
  never mix with the pinned model's; the default models keep their existing caches.

## 0.2.0 - 2026-10-16

forge v2 and `earshot fit`.

### Added

- **`earshot fit`: audio to code.** Fits a small synthesis model (ringing modes, tracked partials, a time-varying
  noise gesture, or a layered hybrid of them) to a recording by analysis by synthesis, and writes it as a readable
  forge recipe module whose `render()` makes the sound. The module stores the structure of the sound, not its
  samples: new takes per seed, knobs for variations, `--morph A B` between two fits of one kind, `--forge` for
  critic-searched variations. On 12 real references: median CLAP cosine 0.945 between recording and render (minimum
  0.881), the same top `earshot describe` tag in 10 of 12. In our 12-trial blind test the listener could not tell
  the recordings from the renders (one listener: a small test). Method, numbers and limits:
  [docs/FIT.md](docs/FIT.md).
- **forge v2.**
  - Reference sets (`earshot/forge/refsets.py`): 18 motion-design categories, split by source into a 'forge' side the
    search may use (81 refs) and a 'bench' side it never hears (71 refs). The recordings stay local.
  - A fixed held-out benchmark (`eval/forge/bench.py`): tone, envelope, independent phrasings, AudioSet names,
    artifacts and variety, next to the same checks for real recordings. Results per round in `eval/forge/`.
  - Critic v2: model-free penalties for tonal balance, the spectro-temporal envelope, artifacts and polish; a designed
    mode for categories without real references. `version=1` reproduces v1.
  - Search v2: model-free pre-screen, CMA-ES option, threaded renders, robust re-renders, seed bank,
    `report.md` / `compare.png` / audition reel, `forge(families=)` and `palette()`.
  - DSP toolkit: `earshot/forge/dsp.py` (sources) and `earshot/forge/fx.py` (effects).
  - 34 categories as families of named characters, 4-6 variants each: 18 real-recording categories and 16 `pro_*`
    designed ones.
  - One report schema (`earshot/forge/schema.py`) and `earshot forge --check`: every pick re-renders from its
    `report.json`.
  - The pro motion-SFX study: 58 designed effects measured in 8 professional launch films (anonymous ids, nothing
    redistributed) and a cited technique note (`docs/motion-sfx-study.md`, `docs/motion-sfx-techniques.md`,
    `eval/forge/pro/`).
  - A clean blind-test kit (`eval/forge/blind_v2.py`): loudness-matched, randomised trials of made sounds against
    held-out recordings, answer sheet, sealed key, scorer, and a cleanliness report that stops the build on a fail.
  - `eval/forge/scrub_paths.py`: removes machine paths from results; `bench.py` writes scrubbed paths.
- **`earshot gallery`**: one static HTML listening page for forge outputs (players, spectrograms, loudness, critic and
  benchmark numbers, parameters), with `--compare`, `--embed` and `--embed-compressed`.

### Changed

- The `earshot forge` CLI defaults are the v2 setup: pre-screen, seed bank, robust re-renders with a shortlist, the
  critic v2 tone weight, and the recipe's reference set as default refs. The Python `forge()` defaults still
  reproduce v1.
- forge v1 -> v2 on the held-out benchmark, in the five categories v1 had: variants are no longer near copies (CLAP
  cosine between variants 0.89-0.97 -> 0.71-0.85), tails that died too fast reach the real length (click 0.56 -> 1.44
  x the bench median, fast whoosh 0.51 -> 0.88), tone balance closer in 3 of 5. All numbers:
  [docs/FORGE_V2.md](docs/FORGE_V2.md).
- The launch film was re-scored with forge v2 sounds (in its own project; no media here): `earshot critique` 16 -> 19
  of 22 metrics in range, realness 0.73 -> 0.84, cues with an onset within one frame 86 -> 98 % of 43; -14.0 LUFS,
  true peak -2.2 dBTP.

### Known limits

- No blind-test score for the forge v2 sounds is published yet.
- `earshot fit`'s selection weights were set on the same 12 references they were measured on; risers, zips and long
  designed sounds are untested.

## 0.1.0 - 2026-10-16

First public release. `describe`, `match`, `rank`, `find`, `compare` / `ab`, `realness`, `timeline`, `sheet`,
`critique`, `profile`, `measure`, `forge` (v1: five recipes), the optional `llm` critique (own key, explicit consent),
`mcp` and `doctor`, as a CLI, a Python API and an MCP server, on CPU. The `earshot` and `real-sound` agent skills,
held-out evaluations ([docs/VALIDATION.md](docs/VALIDATION.md)) and limits ([docs/LIMITS.md](docs/LIMITS.md)). The
22 s launch film, every sound made with `earshot forge`.
