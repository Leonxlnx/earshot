# earshot semantics: evaluation results

All numbers come from `eval/semantics/run_eval.py` (commands below) with CLAP `laion/larger_clap_general` (bf16, CPU,
`EARSHOT_THREADS=2`). Raw outputs: `results.json` (tuning library), `heldout.json` (held-out library), `kind.json`
(query-kind reader), `find_repeatpad.json` (find), `timing.json`.

## Data, and what was tuned on what
| set | files | labels | used for |
|---|---|---|---|
| tuning library: `<assets>/<8 folders>/` | 219 | manifests (need -> 35 classes, `queries.json`) | choosing pad / templates / calibration, the verdict thresholds, find's heuristics. 67 of the 322 vocabulary terms were added after seeing its classes, so its numbers are **optimistic** |
| held-out library: `.../assets/_rejected/<8 folders>/` | 125 (126 minus a duplicate) | folder + file name only, written **before** any score on these files (`heldout_labels.json`) | nothing: the honest test of the shipped configuration |
| term-bias references (unsupervised: only their mean embedding) | tuning analysis: held-out library + 7 launch films + Clawd mixes + synth stems (148); shipped `vocab_stats.json`: tuning library + films + mixes + synth stems (241) | none | always disjoint from the files being scored (the Clawd mixes do contain some tuning clips) |

Vocabulary history: 217 terms written before any evaluation (`vocab_preregistered.json`), +67 after seeing the tuning
classes, +38 general everyday terms (vehicles, household, tools, animals, people) from general knowledge; frozen
before the held-out library was scored.

## Calibration variants (tuning library, 219 files)
Pooled AUC = one threshold for every (recording, class description) pair; lenient leaves out pairs of related classes
(Enter key vs 'a spacebar press'), strict counts them as wrong. Per-split (dev/test) rows for every variant are in
`results.json`.

| pad | templates | calibration | cat MAP | class MAP | class top-3 | tagging acc | AUC | AUC strict |
|---|---|---|---|---|---|---|---|---|
| repeatpad | 1 | raw cosine | 0.698 | 0.597 | 0.843 | 0.553 | 0.944 | 0.927 |
| repeatpad | 5 | raw cosine | 0.728 | 0.627 | 0.886 | 0.594 | 0.951 | 0.936 |
| repeatpad | 5 | per-sound z | 0.770 | **0.650** | **0.900** | 0.594 | 0.963 | 0.948 |
| repeatpad | 5 | term bias only | 0.728 | 0.627 | 0.886 | 0.580 | 0.954 | 0.939 |
| **repeatpad** | **5** | **term bias + per-sound z (shipped)** | **0.778** | 0.642 | 0.886 | 0.580 | **0.966** | **0.951** |
| repeatpad | 5 | std-divided + per-sound z | 0.768 | 0.629 | 0.900 | 0.578 | 0.964 | 0.949 |
| zero | 5 | shipped calibration | 0.758 | 0.631 | 0.843 | 0.591 | 0.964 | 0.949 |
| spaced | 5 | shipped calibration | 0.784 | 0.644 | 0.871 | 0.589 | 0.965 | 0.950 |

The per-sound z does most of the work; the term bias adds a little to the pooled AUC (the thing verdicts rely on) and
costs a little class MAP. Dev / test for the shipped variant: AUC 0.973 / 0.956, class MAP 0.678 / 0.639. Specific
needs (24 concrete queries): a correct file in the top 3 for 23/24 (miss: 'a cork pop', mouth pops rank first), at
rank 1 for 19/24. Reference-set sensitivity (dev / test AUC): folder-balanced 0.973 / 0.956, uniform 0.972 / 0.957,
without music-like refs 0.974 / 0.959, transductive upper bound 0.974 / 0.961, no term bias 0.971 / 0.953.

## Verdicts
Rule (on the tuning library, lenient labels, 0.5 z grid, top-down): strong = lowest z with LR(z >= t) >= 100,
plausible = lowest z below it whose band LR >= 20, weak = lowest z below that whose band LR >= 5. It gives
**strong z >= 3, plausible 2.5-3, weak 2-2.5, no match < 2**. LR = P(band | true pair) / P(band | wrong pair);
precision at a library base rate of ~3% and at even prior odds follow from it.

Per band (not cumulative), held-out library, shipped configuration (248 true pairs; 7614 wrong lenient / 8222 strict):

| verdict | z | LR lenient | precision @3% | precision even odds | LR strict | precision @3% strict | recall at or above |
|---|---|---|---|---|---|---|---|
| strong | >= 3 | 1289 (126 true / 3 wrong) | 0.98 | 1.00 | 51 | 0.61 | 0.51 |
| plausible | 2.5-3 | 21 | 0.41 | 0.95 | 9.1 | 0.22 | 0.63 |
| weak | 2-2.5 | 12 | 0.28 | 0.92 | 6.6 | 0.17 | 0.76 |
| no match | < 2 | 0.25 | 0.008 | 0.20 | 0.25 | 0.008 | - |

Tuning library, same bands (lenient / strict LR): strong 170 / 40, plausible 29 / 12, weak 7.1 / 4.1, no match 0.18.
The first build's 'weak' band (z 1-2) had LR 0.98: no evidence at all, which is why it moved to 2-2.5.

## Held-out library (125 files, nothing tuned on it)
| vocabulary | cat MAP | class MAP | class top-3 | tagging acc | AUC | AUC strict |
|---|---|---|---|---|---|---|
| shipped (322 terms) | 0.691 | 0.646 | 0.827 | 0.707 | **0.980** | **0.965** |
| pre-registered (217 terms) | 0.685 | **0.669** | 0.827 | 0.707 | 0.976 | 0.960 |
| raw cosine (no calibration) | 0.646 | 0.700 | 0.788 | 0.727 | 0.978 | 0.964 |

The vocabulary extension carries over to new recordings for verdicts (strong precision 0.926 -> 0.977 lenient,
0.543 -> 0.606 strict) but not for ranking files (class MAP 0.669 -> 0.646). Raw cosine ranks files for a class query
better here (class MAP 0.700) though worse on the tuning library (0.627); calibration is kept for its absolute verdicts.
Category AP: ambience 0.92, keyboard 0.98, magic 0.98, music 1.00, motion 0.82, garden 0.45, toy 0.20, foley 0.18.

Music mood (unvalidated tags): on 45 held-out music tracks, all chosen for a bright, playful film, the top mood tag was
a dark / sad / tense one for 18% (chance 31%; tuning library 0 of 8). A music-specific term bias did not help (20%),
so mood stays an explicitly unvalidated hint.

## find()
| test | result |
|---|---|
| Keychron, 17 single presses (`fs-789628`, manifest slices), top 17 | 16 usable trims on 16 distinct presses; the 17th is a false positive on the stray double hit at 9.54 s the manifest says to skip. Attack measured independently (first sample at -20 dB of the slice peak): trims start 5-54 ms before it (median 16 ms), never after |
| Mars board, 32 presses (`fs-609534`), top 10 | 9/10 usable trims, lead 6-8 ms |
| typing recording (`fs-709460`: slow presses 0-17 s, then fast typing), top 5 | single-press query: 2 on labelled slow presses, 3 strokes cut from the fast part; continuous-typing query: 5/5 windows in the fast part |
| 44 s scene: 8 real one-shots over room tone | top-1 on the right event 8/8; 7 tight trims (lead 5-66 ms), the small bell came back as a 5 s window |
| 300 s scene: a key press every ~1.3 s throughout + glass tings at 20 / 150 / 275 s, cork pop 205 s, whoosh 240 s | 5/5 found as tight trims (the first build never looked past ~95 s) |

Query kind (`kind.json`): thresholds from the tuning phrasings (event = lowest event score, texture = median texture
score). Generic test queries: 25/30 one-shots read as one-shots, 4 as either (both candidate kinds scored), 1 as
continuous ('a bird chirp': windows only); 14/20 continuous sounds windows-only, 6 either, none read as a one-shot.

## Timings (EARSHOT_THREADS=2, bf16, shared Ryzen 7 260; models warm, nothing cached)
model load 27 s; describe 0.56 s key 17.0 s; describe 31 s rain 52.5 s; match, new query on a cached file 2.0 s;
find 18 s file 29.9 s cold, 1.2 s new query; find 300 s file 61.8 s cold; rank 20 foley files (804 s of audio) 55.8 s
cold (2.8 s per file), 3.4 s new query. A CLI `match` on a cached file and query loads no model: 1.5 s wall.

## Real sound effects: what they are vs the job they play (`real_effects.py`, 2026-11-06)
A second library: 195 real licensed one-shot effects (Freesound CC0 / Mixkit, 148 shorter than 1 s), each with its
title as published (a person's label of what it is) and filed under the film jobs it plays (26 jobs: appear, toggle,
logo-lock, text, whip ...) with the description an agent had judged that job by. Nothing in earshot was tuned on it.
Repeatpad (shipped) unless the row says otherwise; "passed" = own job at z >= 2 (weak or better).

| asked with | strong | plausible | weak | no match | AUC own vs other | other jobs passed | cheap synth won |
|---|---|---|---|---|---|---|---|
| its own title (vs the other 194 titles) | 124 | 27 | 17 | 27 | 0.919 | - | - |
| the job's description as filed | 95 | 33 | 24 | 43 | 0.912 | 10.4 % | 14 |
| best of two sound-only phrasings (written before scoring) | 108 | 34 | 23 | 30 | 0.914 | 12.5 % | 22 |
| best of the description and the two phrasings | 119 | 36 | 21 | 19 | **0.922** | 14.9 % | 10 |
| its own title, spaced padding | 135 | 24 | 16 | 20 | 0.921 | - | - |
| the description and the two phrasings, spaced padding | 125 | 31 | 20 | 19 | 0.920 | 15.4 % | 15 |

- **earshot hears real effects as what they are**: asked by its own title, 168 of 195 pass (124 strong). The
  failures as filed come from job words, which the model cannot hear: "a soft rounded pop, something appearing"
  passed 0 of 9 real pops, 8 of which pass by their own titles ("a soft mouth pop" z 4.8); "one clean low hit under a logo"
  0 of 4; "a light finger tap on a screen" 0 of 4; "a soft plucked or tapped note" 3 of 8 kalimba, marimba and
  sansula notes (titles z 3.2-6.2).
- Asking the way the skill says to (two or three phrasings of the sound itself, keep the best) passes 176 instead of
  152 and separates the jobs slightly better (AUC 0.922 vs 0.912). The cost: a sound passes 15 % of the other jobs
  instead of 10 % (jobs overlap: one soft tap can play a letter or a touch).
- The forced choice against four cheap-synth descriptions ("a video game blip" ...) is not a realness test: on short
  real transients CLAP's closest of the four is usually "a video game blip", and with the job words it won 14 times,
  mostly on taps, pops and soft thumps. Use `realness` to ask whether a sound is a recording.
- Spaced padding (silence to 1 s, then tiled: one hit per second instead of the hit repeated back to back) lifts the
  title matches (strong 124 -> 135, no match 27 -> 20) at the same AUC; zero padding is worse on every row (title no
  match 38). Not shipped: the verdict bands were set with repeatpad, so a switch needs the held-out library run again.

## Commands
```
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py embed --set heldout --part 1/2   # and 2/2
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py stats
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py analyze
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py heldout
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py kind
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py find
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/run_eval.py timing
EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/real_effects.py   # the effect library (EARSHOT_EVAL_SFX_LIBRARY)
```
(tuning-library and tuning-reference embeddings for the three pad modes were cached by `embed --set eval|tuneref --pad
repeatpad|zero|spaced` in the first build.)
