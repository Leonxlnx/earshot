# Validation

Every number here comes from a command in this repo, run on a Ryzen 7 260 laptop CPU (bf16, `EARSHOT_THREADS` 2-3,
other jobs running). Full tables, data descriptions and "what was tuned on what" live in each area's results file;
this page is the summary. Data: the Clawd film's licensed recordings (219 used + 125 rejected Freesound/Mixkit files),
its finished 60 s soundtrack with every placement logged, the code-synthesized score that project retired, renders
of the Cinetic synth engine, independent numpy generators, and 7 professional launch films (measured locally, never
committed).

## 1. Matching descriptions, ranking, finding moments ([eval/semantics/RESULTS.md](../eval/semantics/RESULTS.md))
`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py analyze` (tuning library) and `... heldout`.

- Calibration (shipped: 5 prompt templates, term-bias removal + per-sound z): pooled AUC of (recording, description)
  pairs 0.966 on the tuning library (raw cosine 0.951). Tuning-library numbers are optimistic: 67 of the 322
  vocabulary terms were added after seeing its classes.
- **Held-out library (125 files labelled from file names before any scoring, nothing tuned on it)**, per verdict band:

| verdict | z | true : wrong (lenient) | precision at a 3% base rate | at even odds | strict (close variant = wrong) |
|---|---|---|---|---|---|
| strong | >= 3 | ~1300 : 1 (126 true vs 3 wrong pairs) | 0.98 | 1.00 | 51 : 1 (precision 0.61 at 3%) |
| plausible | 2.5-3 | 21 : 1 | 0.41 | 0.95 | 9.1 : 1 |
| weak | 2-2.5 | 12 : 1 | 0.28 | 0.92 | 6.6 : 1 |
| no match | < 2 | 0.25 : 1 | 0.008 | 0.20 | 0.25 : 1 |

  Half of the true matches reach "strong" (recall 0.51), 76% reach at least "weak". The strong band's ratio rests on
  only 3 wrong pairs: with that count's 95% Poisson range (0.6-8.8) it lies anywhere from ~440:1 to ~6000:1, and the
  tuning library gave 170:1. In a library search (~3% of candidate pairs true) a "strong" hit is the right kind of
  sound 98% of the time but the exact variant only 61% of the time (an Enter key for a spacebar counts as wrong).
- Concrete needs, **tuning library only** (24 queries written from its manifests, such as "a deep thocky keyboard
  press", "a glass ting", "rain on a window"; optimistic, never scored on the held-out library): a correct file in the
  top 3 for 23/24 (miss: "a cork pop", mouth pops rank first), rank 1 for 19/24.
- `find`: on a recording of 17 single key presses, 16 usable trims on 16 distinct presses (the 17th is a false positive
  on a double hit the curator marked to skip), each starting 5-54 ms before the attack, never after; 9/10 on a 32-press
  board; in a 44 s scene of 8 real one-shots the right event is top-1 for 8/8 (7 tight trims, one 5 s window); in a
  300 s scene every planted event is found as a tight trim (the first build never looked past ~95 s; fixed with
  gap-free windows and a two-stage search). Re-run after the final review's find changes (snapped windows for
  one-shot queries, no events across edit gaps, `--kind event` = onset-based trims only): the same numbers.
- Speed (`eval/semantics/timing.json`, models already loaded, nothing cached): model load 27 s once per process;
  `describe` of a 0.56 s key 17 s; `match`, new query on a cached file 2.0 s; `find` on an 18 s file 29.9 s for a new
  file, 1.2 s for a new query; `rank` 2.8 s per new file. Every CLI call is a new process and pays the model load, so
  for more than a few calls use the MCP server or one Python script (`from earshot import api`).

## 2. Real vs code-synthesized ([eval/realness/RESULTS.md](../eval/realness/RESULTS.md))
`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/realness/run_eval.py --stage report` (data and training scripts in
`eval/realness/`; `train.py` reproduces the shipped weights bit for bit).

- 1484 clean clips (691 real / pro, 793 synthesized), grouped cross-validation that never tests on a trained group:

| scheme | ROC-AUC | acc@0.5 |
|---|---|---|
| leave one real category / synth family out (headline) | 0.922 | 0.860 |
| leave one synth generator type out | 0.958 | 0.902 |
| retired score never trained on: real vs retired | **0.948** | 0.862 |
| retired score + the whole engine that made it never trained on: real vs retired | 0.909 | 0.839 |
| zero-shot CLAP prompts (no training, baseline) | 0.858 | 0.749 |

- Verdict bands (held-out): "real or pro-produced" p_real >= 0.77 and "likely code-synthesized" <= 0.17 are each right
  ~92% of the time on a pool that is 47% real / 53% synthesized, with both thresholds chosen on these same held-out
  predictions (so the 92% is in-sample for that choice); 79% of clips get a firm verdict, the rest "uncertain". The
  base-rate-free error rates: 6.7% of real clips are called synthesized, 5.5% of synthesized clips are called real.
  Long files use file-level thresholds (lo 0.40) on the mean of their 4 s windows, read from only 23 synthesized long
  files (weak calibration; the tool says so).
- Pro-produced music held out by provider: mean p_real 0.86, 44 of 52 files called real, none called synthesized.
- A deterministic rule catches pure, dead-steady test tones: a 30 s 440 Hz sine gets 0.48 from the model alone and 0.03
  with the rule; the rule fired on none of 4741 audible clips of the 345 real recordings
  (`eval/realness/tone_guard.py` -> `tone_guard.json`).

## 3. Listening through a soundtrack ([eval/timeline/RESULTS.md](../eval/timeline/RESULTS.md))
`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py` (ground truth: the Clawd mixer's placement log).

- Event detection vs the 147 placed sound effects: recall 93.2% (96.4% of those louder than the bed, 89.1% of those
  under it; 64.7% of the ones more than 6 dB under the music).
- Sync against the picture cue frames: for the 140 of the 153 cues with an attack-aligned placement, measured offsets
  within 5 ms of the true placement offset for 84%, within one frame for 96%. The "without a hit" list is the weak
  part: both cues it named on Clawd were sounded whooshes. Beat grid of the 120 BPM score found at 120.0 BPM.
- Event tags are hints: right kind for about 1 in 2 exposed events and 1 in 4 events buried in a bed; whooshes 56%,
  keyboard presses 58-82%, grass footsteps under music 0%.
- Time: 60 s soundtrack 81 s, 93 s film 93 s (detail normal, 2 threads, 2-4 other processes busy).

## 4. Launch scorecard ([eval/launch/RESULTS.md](../eval/launch/RESULTS.md))
`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py`

- Leave-one-out: each of the 7 reference films vs a profile of the other six: 120/140 reference-range metric checks in
  range (86%; the no-overfit baseline for min-max ranges of six is ~71%). Not a blind test: some rules were changed
  after the first leave-one-out run (listed in the results file). `critique` on one of these 7 films now detects it
  and scores it leave-one-out automatically.
- The Clawd film: with its real-recordings soundtrack "close" (21/23 metrics, p_real 0.94; flags a quiet first second
  and one loud stretch), with the old code-synthesized soundtrack "needs work" (critical: p_real 0.33, the
  code-synthesized windows listed; major: fades away instead of ending on a hit, sparse).
- Known-bad variants written before the run (quiet, hot, mono, muffled, late start, dead air, long fade, hole,
  squashed, SFX-only, ambience-only, synth bed, sine, pink noise) and how each is flagged: in the results file.

## 5. Interfaces
`.venv/Scripts/python.exe -m pytest tests/test_interfaces.py -q` (+ `--slow`)

- All 13 MCP tools exercised over a real stdio connection on real files (silence, clicks, mono/stereo, ogg, mp3, the
  60 s soundtrack, an mp4 film); bad input returns readable errors, never a crash; every JSON part valid.
- Models stay loaded between MCP calls (the stdio handshake takes a few seconds; not benchmarked).
- The Gemini critique has never been called live (no key on this machine, and a live call uploads audio): every path
  is tested against a fake HTTP layer, including the key never being printed and uploads deleted after use.

## 6. The real-sound skill reproduces the Clawd soundtrack exactly
```
.venv/Scripts/python.exe -I skills/real-sound/scripts/mix.py --root "<film project>" --design audio/sounddesign.json \
  --cues out/cues_all.json --out out/repro/soundtrack.wav --stems out/repro/stems --report out/repro/mix_report.json \
  --cues-out out/repro/cues_sounded.json --cache out/repro/cache
```
Bit-identical to the shipped `out/audio/soundtrack.wav` (sha1 `b48c5e8872fe96765624f88db18a0b49a0a05d68`), with an empty
decode cache, and `mix_report.json` byte-identical. (`--cache` must point to a writable folder.)

## 7. Tests
`.venv/Scripts/python.exe -m pytest -q` (fast; synthetic fixtures, no network) and `--slow` (loads the models / long
end-to-end runs). At release: 128 passed, 14 skipped (slow) in ~65 s; the 14 slow tests also pass (~3 min).
