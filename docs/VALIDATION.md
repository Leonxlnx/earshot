# Validation

Every number here comes from a command in this repo, run on a Ryzen 7 260 laptop CPU (bf16, `EARSHOT_THREADS` 2-3,
other jobs running). Full tables, data descriptions and "what was tuned on what" live in each area's results file;
this page is the summary. Data: the Clawd film's licensed recordings (219 used + 125 rejected Freesound/Mixkit files),
its finished 60 s soundtrack with every placement logged, the code-synthesized score that project retired, renders
of the Cinetic synth engine, independent numpy generators, and 7 professional launch films (measured locally, never
committed). Section 8 covers the sounds earshot makes (1.0). Their yardstick is the owner's listening: earshot's
own listening models did not predict his ratings, so their scores are not used as a measure of quality there.

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

### The owner's listening

The yardstick for the sounds earshot makes is one listener, the project's owner, rating each sound 1 (bad) to 5
(very good) on listening pages in a browser, with a word or a sentence where he had one. Six pages were rated on
2026-11-15 and 16 (279 rated files). It is one listener, the sounds were labelled on the page (on page 5 the
recording and its takes stood in named columns), and nothing was loudness-matched beyond each page's own levelling:
a strict judgement, not a blind test.

| page | what was on it | his ratings |
|---|---|---|
| 1 | one success sound in six timbres; the interface set; instrument notes; game sounds; chip sounds; film sounds of the bank; ambience; objects | timbres: soft pluck 4, kalimba 3-4, felt piano 3, marimba 2.5, vibraphone 2, glass / glockenspiel 1 ("AI slop"). Interface: notify 4, error 4, tap, toggle, success and send 3 ("a bit too technical"). Instruments 2-3, hand bell and bowl 1. Game: coin and pickup 3, level up 3, lose 2, step on gravel 2, explosion 2 ("not an explosion"), jump and landing 1.5, punch 1 ("not a hit"). Chip: 4, 4, 4, 5. Bank film sounds: fast crisp whoosh 4, deep impact, riser, braam and logo 3, warm pad 2, soft pop 1. Ambience: rain on a window 5, waves 3, forest 3, night 3, campfire 2. Objects: coin drop 3, paper 2.5, glass clink, light switch and desk bell 2, door 1 |
| 2 | motion sounds invented from film statistics; old against new; drums; the soft default voice | motion 1-3 (booms, slams and thumps 1, risers and falls 1-2; fast and deep whoosh, glass tock, glint and soft sting 3); drums 1-3 (tight snare 3; soft kick, closed hi-hat and shaker 2; the other eight 1); success, notify and error figures 3, tap 2, toggle 2 ("does not sound like a toggle"), level up and coin figures 4 |
| 3 | 14 sounds fitted from CC0 recordings by an earlier engine (`earshot fit`) against 6 invented ones | two keyboard keys 4, a space bar and a mouth plop 3, the other fits 1-2; the invented ones 1-2 |
| 5 | 24 CC0 recordings, each next to a take of its learned model and a varied take; 6 morphs; 6 sounds sampled from class models; 6 takes raw and finished | the recording, its take and its varied take got the same rating in all 24 rows (the sources themselves 1-5); morphs 4, 4, 4, 3, 2, 1; class samples 1-2; the finish helped short dry sounds (a wood knock 2 -> 3, a mouse click 1 -> 2) and made an airy whoosh worse (3 -> 2) |
| 6 | twelve music beds | 8-bit 5, warm 3, playful 3 ("a bit cheap"), the other nine 1-2 |
| 7 | before and after: louder figures, invented taps and toggles, ambience, jingles | success 3 -> 4, notify 2 -> 3, error 3 -> 4; taps, hovers and toggles 1-2 ("you cannot recognise these sounds"; two real keys beside them 4); forest 3 -> 5 (the dove taken out), campfire 2 -> 3, waves 2 -> 3, night 3 -> 2 (the changed crickets; reverted); jingles 3 |

Two of his notes changed the work more than any number: "much simply does not sound like what it should, not
recognisable, not professional, some too technical", and "some should be LOUDER and FULLER: many of your sounds are
short, you barely hear them". The learned family (page 8) and the learned ambience beds (page 9) had not been rated
when this was written.

**What passed.** Chip sounds (4-5). The soft tonal figures: notify and error 4, success 3-4, level up and coin 4.
Rain on a window (5) and the forest without its dove (5), both made from rules. The learned recordings: he rated each
take exactly as he rated its source, so a learned design is as good as the recording it was learned from. Morphs
between two good recordings of one kind (4).

**What failed.** Sounds invented from formulas for real things: footsteps, jumps, landings, punches, explosions,
doors, switches, taps and toggles (1-2), in every round, whether written by hand or fitted from a recording with the
earlier engine. Most music beds (1-2; three kept). Most drums invented from physics (1-2; "big problems, does not
sound good"). Bells, glass and glockenspiel timbres for interface sounds (1). New sounds sampled from a class model
of several recordings (1-2): class sampling was dropped.

**What followed.** Realistic sounds are learned from public-domain recordings (`earshot/sounds/learned.py`), the
interface touches are real keys and switches, the ambience beds he rated 2-3 were replaced by learned textures,
nine music beds were removed, the acoustic drums and hand percussion are played from 130 recorded strokes (VCSL and
Karoryfer; not rated yet), the figures are played on real instruments learned note by note (a marimba by default;
a harp, pizzicato strings and an upright piano by name), and short sounds are
raised (`master.finish` lift; the loudest 50 ms of the tap, toggle and hover figures went from -19.5, -20.4 and
-19.5 to -13.2, -9.4 and -13.5 dBFS).

**The listening models did not predict him.** The CLAP match scores and the realness score that earshot uses to
listen did not predict his ratings: sounds he rated 1-2 had passed the models' checks, and the invented sounds landed
at 1-3 for him whatever the models said. Before v1 these docs quoted those scores as the quality of the made sounds;
they are no longer quoted for that.

**A taste predictor, and what it is good for** (a report in the project's lab, not in the repository). From his
279 rated files (231 units, 144 groups that are always held out together), a nearest-neighbour vote on CLAP
embeddings (k = 10) predicts the order of his ratings with a Spearman of 0.60 in grouped cross-validation, 0.54
(p 0.003) on the 24 real recordings of page 5 when trained without that page, and 0.56 (p 0.002) when trained
without anything recorded. Its top fifth got 3 or more in 74 % of cases (40 % for all), its bottom fifth 2 or less
in 87 % (59 % for all). It ranks; its absolute values are off; within one family it swings from -0.02 to 0.82; on
pairs he rated differently inside one group (before and after, raw and finished) it is at chance (47-57 %). Features
read from the signal alone did not predict him. Verdict: a listening order and a tie-breaker inside a class, never
a judge; no design entered or left the catalogue on its word. It was used to order recordings for the learned
store and to drop members it scored lowest where a design had others.

### Learned sounds, measured

- **Fidelity** (`earshot/synth/resyn.py`, 41 CC0 sources): the band envelopes of a take sit a median 2.4-3.2 dB
  from the recording's at 2 ms resolution (the earlier fitting engine: 5.9 dB), the spectral shape within 0.7-1.3 dB,
  a click's rise within 1 ms; a model is 3.6-35 kB (median 9.1 kB); a render takes 0.26-0.33 s per second of sound.
  Weak: soft onsets smear (a bag drop's rise went from 61 to 99 ms); a fast swipe; impacts whose mid band is partly
  carried as noise (takes differ by about 3 dB there).
- **The store**: 153 designs, 364 members, 7.00 MB (21 of the designs and 66 of the members are the film's
  material, below; 13 members are recordings of a whole action, played by a performance); nothing the owner rated
  1-2, no duplicates, at most five single-event members per thing (six notes of low brass for the braam). Seed 0
  plays the member with the best listening score. Admission rules in the lab build: no chopped start or end, a full band, a natural tail, not faint, a
  clear onset, a single event.
- **Ambience textures** (`earshot/synth/quilt.py`, 11 BigSoundBank textures, 3.87 MB; the wind in trees was
  removed: its recording's leaves do not follow its gusts, and `wind/trees` is made by the texture engine): a take differs from its
  recording by 2.4-3.6 dB at band-envelope level, the same as two takes differ from each other, except the crickets
  (4.4 dB against 2.2: two crickets in one band lose their beating). The joints of a stitched bed differ by 0.9-1.5
  times the texture's own variation (the pebble and rock shores about 1.5). The sea beds are 26-30 s with five waves
  each, the fire 24 s. A bed renders in 1.0-1.7 s of CPU per second of sound.
- **Drums** (`earshot/sounds/drums.py`, `earshot/data/drums/`, 1.47 MB): 130 strokes from VCSL and Karoryfer
  (Big Rusty Drums), stored softest first with their velocity within their instrument; the seed walks through a
  design's strokes and `energy` picks softer or harder ones. Nobody has rated them yet.
- **Sources**: BigSoundBank, Kenney, VCSL, VSCO-2 CE, Karoryfer and CC0 uploads on Freesound, all CC0 or an explicit
  public-domain dedication; each recording is credited in the `CREDITS.md` of `earshot/data/resyn/`, `textures/`,
  `drums/` and `notes/`.
- **Instrument notes** (`earshot/synth/played.py`, `earshot/data/notes/`, 0.85 MB): 100 single notes from VSCO-2 CE
  and VCSL (marimba 14, harp 24, pizzicato 48, upright piano 14), each played as a take moved to the pitch asked for
  with the tuning measured on the recording. Not rated by the owner yet.
- **Film sounds** (`earshot/sounds/motion.py` on `earshot/synth/design.py`): 23 motion designs are layers of 21
  learned film things (66 recordings) (orchestral bass drums, timpani, tam-tams, crash cymbals, heavy drops, a door slam, swells
  of air, cymbal and gong, drum and timpani rolls, low brass, deep, airy and fast whooshes, passes, thunder),
  reversed, moved, faded and glued by one room. The owner rated the invented film sounds they replace 1 to 3 of 5;
  the ones built from recordings have not been rated by anyone yet.

### Understanding requests

Three sets of requests were written by someone who never saw the catalogue: A (440 requests for 110 intents, in
four styles) and B (220) were used to develop the reader (`earshot/brief.py`, the lexicon, the designs' `uses`) and
the guess thresholds; C (220: half long descriptions of a sound that never name it, half casual requests) is held
out. A request counts as right when the first pick belongs to a family accepted for its intent (several families
can be right: a coin pickup may be `coin/*` or `pickup/*`). The request sets and the scoring script are in the lab,
not in the repository.

| set | first pick right | right among the top three |
|---|---|---|
| A (development, 440) | 92.3 % | 96.1 % |
| B (development, 220) | 96.4 % | 99.1 % |
| C (held out, 220), scored once on the v1 catalogue | 69.5 % | 76.8 % |

When the learned recordings arrived, 68 answers were added to the accept lists of A and B by written rules: a learned
recording of the thing the request names is a right answer. Without those additions A is 89.1 % / 95.2 % and B
93.6 % / 98.2 %. C was scored once with the same code; the code before the learned family scored 70.9 % / 78.2 % on
the same catalogue, so the learned designs cost C about one and a half points. Development numbers are optimistic by
construction; C is the number to quote. Four of the development misses (three taps, one hover) are requests whose
accepted learned answer was the interface set's earlier tap or hover recording; the reader now gives them the
touches the `ui` set plays (`keyboard/mechanical`, `leather/grab`). About one held-out request in three or four (long descriptions and
casual paraphrases) needs a second call, which is what `also`, the guess note and `sound=<id>` are for.
`--listen` (the sound atlas) was not re-measured on the v1 catalogue.

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
