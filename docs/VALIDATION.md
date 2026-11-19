# Validation

Every number here comes from a command in this repo, run on a Ryzen 7 260 laptop CPU (bf16, `EARSHOT_THREADS` 2-3,
other jobs running). Full tables, data descriptions and "what was tuned on what" live in each area's results file;
this page is the summary. Data: the Clawd film's licensed recordings (219 used + 125 rejected Freesound/Mixkit files),
its finished 60 s soundtrack with every placement logged, the code-synthesized score that project retired, renders
of the Cinetic synth engine, independent numpy generators, and 7 professional launch films (measured locally, never
committed). Section 8 covers the sounds earshot makes (1.0). Their yardstick is listening by ear: earshot's own
listening models did not predict how a person judged the made sounds, so their scores are not used as a measure of
quality there.

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

- The server lists 20 tools: the 13 listening tools, the three of the sound bank and `make`, `make_set`, `sounds`
  and `build`. The 13 listening tools were exercised over a real stdio connection on real files (silence, clicks, mono/stereo, ogg, mp3, the
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
end-to-end runs). What the tests of the made sounds check is in section 8. On the released tree the fast run has
4059 passed, 1685 skipped (the 1684 slow ones and one that needs renders a fresh clone lacks) and none failed;
the slow tests alone (`--slow -m slow`) 1672 passed, 12 skipped (two known faults of the learning engine, ten real
whooshes the pitch detector reads as gliding) and none failed.

## 8. Made sounds (1.0)

### Listening by ear

The made sounds were judged by ear by one listener, the maintainer, on listening pages in a browser. The sounds were
labelled on the page and were not loudness-matched beyond each page's own levelling: a strict judgement by one
person, not a blind test. What it found, in plain words:

- **Passed.** Chip sounds. The soft tonal figures (success, notify, error, level up, coin). Rain on a window and the
  forest bed, both made from rules. Takes of learned recordings: on 24 public-domain recordings, each played next to
  a take of its learned model and a varied take, the take was judged exactly as good as its recording in every row,
  so a learned design is as good as the recording it was learned from.
- **Failed.** Sounds invented from formulas for real things (footsteps, jumps, landings, punches, explosions, doors,
  switches, taps and toggles), whether written by hand or fitted from a recording. Nine of twelve music beds. Most
  drums invented from physics. Bells, glass and glockenspiel timbres for interface sounds. New sounds sampled from a
  model of several recordings. Many short sounds were too quiet to be heard well.
- **What changed because of it.** Realistic sounds are learned from public-domain recordings; the interface sounds
  are things a hand does with an object, and instruments answer only when music is asked for; weak ambience beds were
  replaced by learned textures; nine music beds were removed; the acoustic drums are played from recorded strokes;
  the figures are played on real instruments learned note by note; short sounds are raised (`master.finish`).
- **Not yet judged by ear.** The sounds built last: the `ui` set as it is now, the rebuilt film and motion hits,
  thumps and risers, the drums from recorded strokes, the figures on the real instruments, the newer learned things
  and beds (animals, people, street, house and kitchen sounds; rain on wood, on a tin roof and on an umbrella), and
  the cookbook's recipes. They were measured against their recordings and checked by the audit.

**The listening models did not predict the ear.** The CLAP match scores and the realness score that earshot uses to
listen did not predict these judgements: sounds that failed by ear had passed the models' checks. These docs no
longer quote model scores as the quality of a made sound. A predictor trained on the judgements (nearest neighbours
on CLAP embeddings) ranks kinds of sound apart but cannot tell a good recording from a poor one of the same kind; it
was used only to order the recordings of a learned design, never to admit or drop a design.

### Learned sounds, measured

- **Fidelity** (`earshot/synth/resyn.py`, 41 CC0 sources): the band envelopes of a take sit a median 2.4-3.2 dB
  from the recording's at 2 ms resolution (the earlier fitting engine: 5.9 dB), the spectral shape within 0.7-1.3 dB,
  a click's rise within 1 ms; a model is 3.6-35 kB (median 9.1 kB); a render takes 0.26-0.33 s per second of sound.
  Weak: soft onsets smear (a bag drop's rise went from 61 to 99 ms); a fast swipe; impacts whose mid band is partly
  carried as noise (takes differ by about 3 dB there).
- **The store**: 309 designs, 692 recorded members, 17.93 MB (some designs are material of the film sounds, below;
  30 members hold several events, such as a whole walk or a knock, and 21 designs have such members). No duplicates,
  nothing judged poor by ear. Seed 0 plays the fullest clean member. Admission rules of the build: no chopped start
  or end, a full band, a natural tail, not faint, a clear onset; the build tool is not in the repository.
- **Ambience textures** (`earshot/synth/quilt.py`, 24 textures, 9.46 MB; `wind/trees` is learned from
  wind through a field of dry leaves (maize), whose leaf band carries two thirds of the power and is dry, after no
  recording of trees passed that check; the fire is learned with an onset at every crackle and its two sides are two
  takes sharing the crackles; the waves from a recording at the water's edge, joined where they were cut; the rain
  on a window without its gutter's ringing drip). Measured on the first twelve (the rain on wood, on a tin roof and
  on an umbrella came later and are checked by the audit only), a take differs from its recording by 2.4-3.6 dB at
  band-envelope level, the same as two takes differ from each other, except the crickets (4.4 dB against 2.2: two crickets in one band lose their beating). The joints of a stitched bed differ by 0.9-1.5
  times the texture's own variation (the pebble and rock shores about 1.5). The sea beds are 26-30 s with five waves
  each, the fire 24 s. A bed renders in 1.0-1.7 s of CPU per second of sound.
- **Drums** (`earshot/sounds/drums.py`, `earshot/data/drums/`, 1.31 MB): 130 strokes from VCSL, Karoryfer
  (Big Rusty Drums) and, for `clap/hands`, single claps cut from one person clapping (Freesound), stored softest
  first with their velocity within their instrument; the seed walks through a design's strokes and `energy` picks
  softer or harder ones.
- **Sources**: BigSoundBank, Kenney, VCSL, VSCO-2 CE, Karoryfer and CC0 uploads on Freesound, all CC0 or an explicit
  public-domain dedication; each recording is credited in the `CREDITS.md` of `earshot/data/resyn/`, `textures/`,
  `drums/` and `notes/`.
- **Instrument notes** (`earshot/synth/played.py`, `earshot/data/notes/`, 0.83 MB): 97 single notes from VSCO-2 CE
  and VCSL (marimba 14, harp 24, pizzicato 45, upright piano 14), each played as a take moved to the pitch asked for
  with the tuning measured on the recording.
- **Film sounds** (`earshot/sounds/motion.py` on `earshot/synth/design.py`): 32 motion designs are built from 25
  learned things of the store and the drum store's strokes, reversed, stretched, layered, faded and glued
  by one room, and follow measurements of 39 professional motion videos ([motion-sfx-study.md](motion-sfx-study.md)):
  the title hit about a third of a second with a 0.15 s swell of air into it and its centroid at 312-432 Hz
  (reference middle half from 388 Hz), the logo hit 0.2-0.3 s, the thump of something appearing about 50 ms, the
  click a tick at 5-10 kHz cut 40 ms after it, a transition plain air gone within tens of ms of its peak. The hits
  are a heavy bag dropped on a real explosion's low end (no drum, no door); the car-door build stays by name
  (`slam/door`, `boom/door`).
- **Interface sounds** (`earshot/sounds/interface.py`): the seventeen roles of `ui` and the interface jobs asked in
  words are things a hand does, built from fixed recordings of the learned store; measured, their tonality (how far a
  steady partial stands over the spectrum around it) is 6 to 18 dB, against 20 to 33 dB for the marimba figures
  they replace.
- **Recipes** (`earshot/recipe.py`, `earshot/data/recipes/`): a cookbook recipe answers a request on its own only when
  the listening model hears it as its thing (a match of z 3.0 or more), clearly better than the catalogue's nearest
  design (by 1.0 or more), and the audit finds it clean: 59 of the 99 do. The other 40 are named only as starting
  points. This is a model's judgement, not a listener's; no recipe has been judged by ear as a set. One sound
  ringing through the body of another ("rain on a tin can") was built and measured: it came no nearer the real
  recordings of rain on wood or on a tin roof than plain rain did, so it is not part of the recipe form.

### Understanding requests

Sets of requests written by someone who never saw the catalogue measure the reader (`earshot/brief.py`, the
lexicon, the designs' `uses`). A request counts as right when the first pick belongs to a family accepted for its
intent (several families can be right: a coin pickup may be `coin/*` or `pickup/*`). The development sets were used
to build the reader and are optimistic by construction; set C was held out. The request sets and the scoring script
are not in the repository.

| set | what it is | first pick right | right among the top three |
|---|---|---|---|
| A (development) | 440 requests for 110 intents, in four styles | 91.8 % | 95.9 % |
| B (development) | 220 requests | 96.8 % | 99.1 % |
| D (development) | 160 requests | 93.1 % | 96.2 % |
| E (development) | 200 requests for the film, motion and newest sounds | 93.0 % | 98.0 % |
| G (development) | 60 requests that also say where the sound plays ("for my running club app") | 95.0 % | 96.7 % |
| C (held out) | 220 requests: half long descriptions of a sound that never name it, half casual requests | 69.1 % | 76.4 % |

C was scored once, on an earlier reader and catalogue than the one released; it has not been scored on the
released reader. Its accept lists were written before the interface and motion answers became sounds of real
things. When the learned recordings arrived, 68 answers were added to the accept lists of A and B by a written rule
(a learned recording of the thing the request names is a right answer). Three requests (two of A, one of B) are
answered first by a design added for this release that is the thing they ask for, which their lists do not yet
accept; counted right, A scores 92.3 % / 96.1 % and B 97.3 % / 99.5 %. About one held-out request in three or four
(long descriptions and casual paraphrases) needs a second call, which is what `also`, the guess note and
`sound=<id>` are for. `--listen` (the sound atlas) was not re-measured on the released catalogue.

**The long tail.** Two lists of 100 everyday requests each, ten in each of ten fields (fantasy, science fiction,
animals, vehicles and machines, household and kitchen, tools and workshop, sports and play, weather and surfaces,
people, office and devices), ask for things the catalogue may not have. Each answer is counted as right, as an honest
nearest (the pick is not the thing and the answer says so), or as wrong without saying so. The first list was used
for development: 55 right, 33 honest nearest, 12 wrong. The second was scored once: 61 right, 30 honest nearest, 9
wrong. Both were measured before the reader's last changes of the release.

**The audit.** Every take `make` writes passes `forge.sfx.check` (clicks, cut tails, clipping, DC, comb ring, rising
glides, stacked attacks) or is re-rolled on up to three more seeds; the line says `clean` or names the warnings.

**Speed** (one thread, no models): a one-shot `make` takes about a second plus Python start-up; a learned ambience
bed 1.0-1.7 s of CPU per second of sound. `--listen` loads CLAP once (20-40 s).

**Tests.** The making half has model-free tests for the catalogue contract (a clean levelled render that passes its
audit, the axes a design declares are the axes it reads, takes that vary, themes, materials, minor keys, figure
lengths, loops, loudness caps), the learned sounds and their engine, the stitched beds, the rooms and finishing, the
voices, drums, music loops, `make` / sets / `pack` and the atlas (`tests/test_sounds_catalogue.py`,
`test_sounds_learned.py`, `test_synth_resyn.py`, `test_synth_quilt.py`, `test_sounds_nature.py`,
`test_synth_drums.py`, `test_synth_music.py`, `test_make.py`, `test_brief.py`, `test_sounds_atlas.py` and one file
per engine). The fast run renders the short designs; the long ones, the bank and the atlas tests that load CLAP run
with `--slow`. Tests check construction (no clicks, no clipping, the physics promised), not whether a sound is good.
