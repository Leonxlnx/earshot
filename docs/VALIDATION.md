# Validation

Every number here comes from a command in this repo, run on a Ryzen 7 260 laptop CPU (bf16, `EARSHOT_THREADS` 2-3,
other jobs running). Full tables, data descriptions and "what was tuned on what" live in each area's results file;
this page is the summary. Data: the Clawd film's licensed recordings (219 used + 125 rejected Freesound/Mixkit files),
its finished 60 s soundtrack with every placement logged, the code-synthesized score that project retired, renders
of the Cinetic synth engine, independent numpy generators, and 7 professional launch films (measured locally, never
committed). Section 8 covers the sounds earshot makes (1.0): measured by its own models and audits, not by people.

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

  Half of the true matches reach "strong" (recall 0.51); 76% reach at least "weak". The strong band's ratio rests on
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
- **Real sound effects** (195 real licensed one-shots filed by the film job they play, `eval/semantics/real_effects.py`):
  asked by its own title, a real effect passes (weak or better) 168 times of 195, 124 strong. Asked by job words ("a
  soft rounded pop, something appearing", "one clean low hit under a logo") real pops and logo hits pass none; the
  best of two phrasings of the sound itself plus the job's description passes 176 of 195 (152 with the description
  alone) at AUC 0.922 own vs other jobs (0.912). Describe the sound, never its job; a forced choice against
  "a video game blip" is not a realness test.
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
(Measured for 0.4; the 1.0 test files are listed in `tests/`.)

## 8. Made sounds (1.0)

What these numbers are: measurements of earshot's own listening models (CLAP and the realness head, the same ones
`match` and `realness` use) and model-free audits, taken in the project's lab on 2026-11-15 on one laptop thread. They
are not a listening test by people. Nobody has run a listening test with people on the 1.0 sounds yet, so "premium" is
a claim about measurements and construction, not about a blind test. The models can be pleased without a sound
getting better (section [Goodhart](../README.md#search-by-listening-earshot-forge)); the designs were built from
physics and checked against the models, not tuned to them. The lab scripts and renders are not in the repository; the
method is below so it can be repeated with `earshot match`.

**Method.** Every design new in 1.0 (248: the modules `ui`, `figure` via ui / game / brand, `tonal`, `synth`,
`retro`, `action`, `object`, `nature`) rendered once at its defaults on seed 1 (loops at 10 s), written as 16-bit WAV.
Each render was matched against its own `says` (two or three plain descriptions of the sound, plus the descriptions
of the voice it played) and, in a forced choice, against "a cheap synthesizer beep", "a video game blip", "an 8-bit
chiptune sound", "a harsh buzzing tone" and "white noise static" (chip designs against "white noise static", "a dial
tone" and "a harsh buzzing tone": chip sound is meant to sound like chip sound). A design is *heard* as itself when
its best own description reaches z >= 2, *strongly* at z >= 3 (the verdict bands of `earshot match`, section 1), and
it *wins* when its own description takes at least half of the forced choice (`p_vs_alts` >= 0.5).

| group | n | heard (z >= 2) | strong (z >= 3) | wins the forced choice |
|---|---|---|---|---|
| figures (interface, game, brand) | 45 | 45 | 42 | 42 |
| notes (36 instruments, 15 patches) | 51 | 51 | 45 | 51 |
| synthesizer | 14 | 14 | 14 | 14 |
| chip | 24 | 24 | 23 | 24 |
| action | 47 | 43 | 24 | 44 |
| objects | 36 | 35 | 19 | 32 |
| ambience | 28 | 25 | 19 | 23 |
| material ticks | 3 | 3 | 1 | 3 |
| **all new designs** | **248** | **240** | **187** | **233** |

For comparison, the 0.4 bank's glockenspiel and marimba notes scored z 2.2 and 2.6 for their own description on the
same test; the 1.0 notes of those instruments 4.6 and 5.1. The 209 bank sounds in the catalogue were not re-heard for
1.0 (their own measurements are in [SFX.md](SFX.md) and [FORGE_V2.md](FORGE_V2.md)).

Below z 2 or losing the forced choice: an opening door (`door/open`), a soft landing (`land/soft`), a slap
(`hit/slap`), a struck match (`match/strike`), tearing paper (`paper/tear`), peeling tape (`tape/peel`), a bottle
opening (`bottle/open`), rain on a roof (`rain/roof`), a small fire (`fire/small`), distant city traffic
(`city/distant`, heard as noise) and a low rumble (`rumble/quake`). Steady noise beds in general sit close to "noise"
for the models. [LIMITS.md](LIMITS.md) lists them for users.

**Calibration of the pianos and mallets.** The struck and string models were compared note by note with freely usable
recordings: the University of Iowa Musical Instrument Samples (a grand piano, a vibraphone, a marimba, a xylophone)
and VSCO-2 Community Edition (CC0: an upright piano, a glockenspiel, a marimba, a xylophone). The laws of the models
(partial levels, decay by partial, the soundboard knock, rise times) were set from several notes each, never fitted
to one; the comments marked `[cal]` in `earshot/synth/struck.py` and `strings.py` give the measured values. No
recording and no per-recording data is in the repository. Before, the two pianos were heard as "an electric piano"
(z 2.5 and 2.8 for their own description); after, z 3.7-4.4, and they win the forced choice against "an electric
piano note" on 6 of 8 notes. The realness model scores them 0.82-0.98 ("real or pro-produced").

**A room has a tone.** Every sound placed in one of the shared rooms gets a noise floor 68 dB under its peak (pink
noise above 120 Hz, per channel, until the sound has died: `earshot/synth/space.py`). Without it the models heard the
same notes as less real and less like themselves: on vibraphone, marimba and piano notes and three interface figures
the match to their own description rose by 0.6 to 1.1 z, and the realness score from 0.03-0.26 to 0.20-0.66, with a
floor (measured at -66 dB); noise that only followed the sound's envelope did less.

**Lessons the lab measured, now in the code.** A sustaining metal bar or synth note damped after less than about
110 ms is heard as an electronic beep: figures keep such notes at least that long (`figure.SUSTAIN_FLOOR`), and the
smallest taps (`tap/tone`, `hover/tone`) use wood unless the theme's voice is short. Small metal plates and bars
struck once read as blips, while a coin and a glass left to ring read as objects: `tick/metal` is a coin and
`tick/glass` rings.

**Understanding requests.** A test set of 440 requests over 110 intents, written by someone who had not seen the
catalogue, scores how often `make` picks a design that answers the request.

<!-- RANKING NUMBERS: lead fills in -->

**The audit.** Every take `make` writes passes `forge.sfx.check` (clicks, cut tails, clipping, DC, comb ring, rising
glides, stacked attacks) or is re-rolled on up to three more seeds; the line says `clean` or names the warnings.

**Speed** (one thread, no models): a single `make` takes about 1-2 s including Python start-up; a 16-sound interface
set with ogg files and a sprite about 25 s; ambience loops of 12-20 s take 3-10 s to render. `--listen` loads the two
models once (20-40 s).
