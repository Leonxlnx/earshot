# Changelog

## 0.3.0 - 2026-10-19

The sound engine of a film: the forge's delivered sounds as a bank, one-sound renders aimed at the picture, a
model-free audit, and film kits. Reference: [docs/SFX.md](docs/SFX.md). This release also ships the 0.2.1 fixes
below (#1, #2), which were not released on their own. No new dependencies.

### Added

- **The sound bank** (`earshot/forge/bank.py`, `earshot/forge/bank/*.json`, shipped as package data): the 183 forge
  v2 finals as their schema-2 render blocks at full precision (`forge_v2r6`), plus three tuned kalimba voices from the
  launch film (`pluck1`). Ids are `<category>/<member>`, `pluck/<name>` and `fit/<role>/<name>`; categories are the
  public names (no recipe module names needed). `python -m earshot.forge.bank build out/forge_v2/*/final -o ...`
  refuses any pick whose block does not re-render its WAV (183 of 183 within 1.2e-7); each entry stores its measured
  onset, peak, end, attack, centroid, loudness and a 50 ms envelope fingerprint, and `bank check` re-renders against
  it (length exact, peak within 5 ms, envelope within 0.5 dB). Rendering needs numpy / scipy only.
- **Knobs** (`earshot/forge/knobs.py`): `dur` (exact: the whoosh family scales its pass, length keys are searched by
  secant, then the file is faded or padded to the sample), `apex` (whoosh family: closed form, up to two measured
  corrections, then the peak is placed by trimming the quiet approach or padding silence: within 2.5 ms on every
  tested target), `pitch` (a note name or Hz into each recipe's own pitch parameter, co-scaling the parameters of the
  same body, inside the parameter's designed range; noise sounds refuse it with the reason), `weight` / `peak_db`.
- **`earshot sfx list / render / check`** and the audit (`earshot/forge/sfx.py`): polish's artifact detectors as
  warnings with context (gated categories may stop on their peak), a rising pitch-glide detector (a tonal sound
  climbing more than 4 semitones over 50 ms or more) and, for hits, a stacked-attack / pre-swell heuristic. Every bank
  render runs it.
- **`earshot kit init / render / check / audition`** (`earshot/forge/kit.py`): kit.json (schema 1) names one bank
  sound per role; `render` writes deterministic, cached takes and `kit.lock.json` with each take's measured onset /
  peak / end / loudness; a take that fails the audit is re-rolled on a new seed. Presets `calm`, `crisp`, `minimal`
  (`earshot/forge/kits/`) with a reason per role; riser, swell and suck off in all three.
- **`python -m earshot.forge.bank add-fit MODEL.json --id fit/<role>/<name> --source SOURCE.json`**: a fitted recipe
  as a bank entry with its source (CC0 / public domain unless `--allow-license`; NC / ND refused), rendered through
  `earshot.forge.fit`, tunable when the fit is tonal. Extra bank files load from `EARSHOT_BANK`.
- **Python API** `earshot.api.sfx_list / sfx_render / sfx_check / kit_init / kit_render / kit_check / kit_audition`;
  **MCP tools** `sfx_list`, `sfx_render`, `kit_render` (16 tools in all).
- **Fitted sounds in the bank** (`earshot/forge/bank/fit.json`, `fit1`; `bank.VERSION` is now
  `forge_v2r6+fit1+pluck1`): 15 recipes fitted from CC0 recordings on Freesound, `fit/<role>/<name>`, each with its
  source (title, author, URL, licence) and the take-to-take `vary` at which its takes passed the take check (seeds
  0-4 as the bank renders them: no hard artifact beyond the recording's, every audit warning also on the recording,
  CLAP >= 0.90 to the fit's own render and within 0.06 of the fit's CLAP to the recording, excess energy within
  1.5 dB, loudness spread <= 2 LU), at a fit grade of high. Credits: `docs/CREDITS_FIT.md` (CC0 needs none).
- **Fit takes** (`earshot/forge/fit`, details in [docs/FIT.md](docs/FIT.md#takes-a-fit-rendered-for-use-03)):
  `fit.render(model, seed, vary=v)` with `vary > 0`, or `take=True`, renders a take. Modal modes and sines partials
  ring out past the trimmed recording's end instead of stopping on a ringing tail (13 of 46 CC0 fits had cut tails in
  their takes), takes are high-passed at 8 Hz (no random DC offset), a spectral model's takes keep one loudness
  (a cloth whoosh's 8 seeds: 7.3 -> 0.45 LU apart), and a sines residual follows the recording's roll-off above 16 kHz
  (`residual.hf`; no more noise above 20 kHz the recording lacked). `earshot fit --takes` scores candidates on seeds
  0, 1, 2. The fit itself is unchanged: on the 12 validation refs every seed-0 and seed-averaged quality number is
  the same as in 0.2.
- `earshot doctor` reports the sound bank: its version, the files it loaded and how many sounds of each type.
- Docs: `docs/SFX.md`; README "Sound for films"; `docs/INTEGRATION.md` "cinetic uses earshot"; the earshot skill's
  Loop 5 "make a film's sound kit"; `docs/FIT.md` "Takes"; `docs/CREDITS_FIT.md`.
- Tests: `tests/test_bank.py`, `tests/test_sfx_kit.py`, `tests/test_fit_takes.py` (no models).

### Changed

- Wheels now ship the forge seed banks (`earshot/forge/seeds/*.json`) next to the bank and kit presets; before, a
  non-editable install had none.
- **Presets use the fitted sounds where the measured comparison favours them** (earshot match of the role's
  description, also against synth-blip descriptions, and earshot realness): calm tick, pop, short whoosh, space bar,
  click and snap; crisp tick, light touch, text, click, snap, key and space bar; minimal light touch, text and snap
  (16 role changes). A changed role keeps its loudness (`level_db` moved by the measured difference at the same
  peak). Every note now says what the sound is and what was measured; no note cites a listening test. All takes of
  the three presets pass `earshot kit check` (64, 64 and 63 takes).
- **The presets' bell spans two octaves of one pentatonic** (calm, crisp: kalimba G4-E6; minimal: the ding A5-G7; 10
  notes each), so a climbing progress motif has a rung for every step; calm's tonal text holds the five pentatonic
  degrees of one octave (C5 D5 E5 G5 A5), so a chain of fragments can step through the chord's tones. crisp and
  minimal keep an untuned soft tap for text.

### Fixed (review before release)

- **apex with a short tail**: the fade that fits a whoosh to `dur` covered an apex 0.03-0.1 s before the end, so the
  delivered peak came 8-189 ms early while `edits['apex']` (measured before that fit) said ok. The fade now starts at
  least 5 ms after the apex, candidates are ranked on apex plus tail error, the peak is placed in whole 1 ms steps of
  its measure, and `edits['apex']` is measured on the delivered file: 59 whoosh-family targets with 0.1-0.12 s tails
  land within 2.5 ms.
- **`kit check` refuses what `kit render` cannot use**, with the same checks (`knobs.validate`, `bank.level`): a null
  or non-numeric `level_db`, `max_s` under 0.01 s (a silent take), `peak_db` / `weight` over 0 dBFS, knob values and
  pitches the sound cannot take, a bad `sr`, and take files that would collide (two spellings of one pitch, or names
  that differ only in case). Pitch labels are canonical (`C#5`, `Db5`, 554.37 Hz -> `Cs5`).
- `knobs.midi` accepts `s` sharps (`Cs5`), the spelling of take file names and of cinetic's kit.py.
- The audit flags **silence** (`silent`: a peak under -60 dBFS or less than 5 ms within 40 dB of it).
- `bank.render` / `earshot sfx render` refuse a peak over 0 dBFS and sample rates other than 44100 / 48000 / 88200 /
  96000 (the CLI wrote clipped WAVs).
- **Click and toggle pitch**: their `pitch` parameter is the tink centre, not the partial heard (click/soft +291
  cents), so `--pitch C7` landed 362 cents off with no warning. Their note is now the measured partial (`meta.pitch_hz`
  in the bank; no pitch knob where there is no clear partial), a target is landed by measurement, every measured pitch
  edit reports `ok` (within 50 cents), and a miss adds a `pitch_off` warning to the take's checks.
- `EARSHOT_BANK`: the docstring gives `os.pathsep` as the separator, and a missing or unreadable file is a readable
  error naming it instead of a traceback.
- The README and docs/SFX.md agree on the blind test: no v2 score is published; the README's test is the v1 one (one
  listener, the author, 15 trials).

## 0.2.1 - 2026-10-17

Not released separately: these fixes ship in 0.3.0.

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
  - `EARSHOT_CLAP` / `EARSHOT_AST` may still name a local model folder (`config.json` plus `model.safetensors`, or an
    older checkpoint's `pytorch_model.bin`): it loads from disk as it is, with no network and no revision, has its own
    embedding caches, and earshot never downloads into it (an incomplete folder is an error naming what is missing).
  - A custom Hub model downloads its top-level config, tokenizer and safetensors files by name; the old `*.json` /
    `*.safetensors` globs also matched every subfolder and weight variant of a repo.
- **The optional Gemini critique no longer treats synthesized sound as a defect by itself** (#1). Its rubric set
  "real recorded foley" as the bar and scored "obviously cheap or synthetic"; it now judges each sound by whether it
  fits the film and is well made, counts synthesis as a flaw only where a sound should read as recorded, and
  `earshot llm --synth-intended` (MCP `synth_intended`) tells the model that synthesized sound is the point.

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
