# Forge benchmark (eval/forge)

A fixed yardstick for sounds the forge makes, so the overnight rounds cannot fool themselves. The forge optimises
against **forge** refs; this benchmark measures against **bench** refs: real / pro recordings from source files (and
authors) the forge never sees, using checks no recipe optimises.

```
python eval/forge/build_refsets.py                     # dev machine: cut the refs, write <EARSHOT_HOME>/refsets.json
python eval/forge/bench.py --made made/ --out eval/forge/results_<tag>.json       # made/<category>/*.wav
python eval/forge/bench.py --files typing_key="demo/forge-test/keypress/final/*.wav" ... --out ...
python eval/forge/bench.py --real-only --out eval/forge/results_real.json        # only the real ceilings
python eval/forge/bench.py --out eval/forge/results_<tag>.json --retable          # rewrite the .md from the JSON
```

Each run writes a JSON (every number per sound) and a markdown table next to it. Run it with `EARSHOT_THREADS=2`. With
the embeddings already cached, the v1 run took 12-25 s and the real-only run of all 18 categories 19-49 s. The first
real-only run, with nothing cached, took 14 min.

## Reference sets (`earshot.forge.refsets`)

`refset(category, split='forge' | 'bench')` returns the paths from the local registry (`EARSHOT_REFSETS`, otherwise
`<EARSHOT_HOME>/refsets.json`). If the registry is missing, it raises a clear error, because public users bring their
own refs: put them in `<root>/<category>/{forge,bench}/*.wav`, then run `refsets.write(refsets.scan(root))`.

The refs on this machine (`out/refs/<category>/{forge,bench}/`, with `out/refs/manifest.json`) are cut from
`demo/launch-sfx-kit/sources.json` recordings. They come either from the kit's own trims (re-cut from the source,
without the kit's high-pass) or from `earshot find` one-shot segments with clean edges and z >= 2. Processing is trim,
fades and peak gain to -1 dBFS, nothing else. The manifest stores each trim's source, trim, licence and earshot
evidence (find z, rank z of the category query, t40, length), plus every rejected candidate and the reason.

**Split rule.** A source *family* is either forge or bench, never both, in any category. On Freesound the family is
the author, which keeps sister takes and sample packs together. VSCO counts as one library; Mixkit goes file by file.
The build refuses to write a registry that breaks this rule. Sources the v1 recipes used as refs are on the forge side.

| category | forge refs (source files) | bench refs (source files) | rank z of the category query, median forge / bench | licences |
|---|---|---|---|---|
| whoosh_fast | 5 (4) | 5 (3) | 4.35 / 4.19 | CC0, CC BY |
| whoosh_big | 3 (3) | 3 (2) | 4.67 / 3.58 | CC0, CC BY, Mixkit |
| swish | 7 (3) | 3 (3) | 3.31 / 3.62 | CC0, CC BY |
| swipe | 4 (1) | 4 (1) | 4.40 / 4.51 | CC0, CC BY |
| riser | 1 (1) | 2 (2) | 2.66 / 3.42 | CC0, Mixkit |
| downlifter | 0 | 0 | - | - |
| impact | 2 (2) | 1 (1) | 3.67 / 3.26 | CC0, CC BY, Mixkit |
| sub_drop | 0 | 1 (1) | - / 4.55 | CC0 |
| click | 8 (5) | 7 (5) | 2.69 / 2.61 | CC0, CC BY, Mixkit |
| tap | 7 (2) | 6 (2) | 1.69 / 1.62 | CC0 |
| toggle | 4 (4) | 4 (3) | 2.10 / 1.92 | CC0, CC BY, Mixkit |
| hover_tick | 6 (2) | 6 (3) | 1.73 / 1.48 | CC0 |
| pop | 7 (6) | 6 (6) | 3.08 / 2.88 | CC0, CC BY, Mixkit |
| notify | 5 (5) | 5 (5) | 1.27 / 1.93 | CC0, CC BY |
| shimmer | 5 (4) | 4 (4) | 3.37 / 4.16 | CC0, CC BY, Mixkit |
| typing_key | 15 (10) | 12 (9) | 5.13 / 3.94 | CC0, CC BY |
| glitch | 0 | 0 | - | - |
| logo_sting | 2 (2) | 2 (2) | 2.38 / 1.87 | CC0, Mixkit |

Total: 81 forge refs from 40 families, 71 bench refs from 37 families.

Thin or empty sets, honestly:
- **downlifter** and **glitch**: no material in the library, real or designed. Made sounds in these categories get
  only the phrasings, AudioSet and artifact checks.
- **sub_drop**: one designed source, held out as bench. The forge has no sub-drop refs.
- **impact**: the library has 3 cinematic hits, and v1 used all three as refs. One (`kit impact_03`) is now the only
  bench ref, so the v1 impact numbers are optimistic and typicality is undefined (it needs 3 refs).
- **riser**: one designed trailer riser (forge); the bench is two real cymbal swells.
- **logo_sting**: stings are composed, so there is no real one. The refs are struck metal / glass rings, useful for
  tail shape and tone only.
- **swipe**: cloth only, no paper.
- **Weak CLAP evidence**: tap, hover_tick and notify bench refs have rank z around 1.5-1.9. CLAP hears finger taps,
  ticks and kalimba / celesta "dings" only weakly.

**Kit WAVs that come from bench sources.** A forge recipe must not use these as refs: `ui_notify_02`,
`whoosh_med_01`, `whoosh_med_02`, `whoosh_big_02`, `swipe_01`, `riser_02`, `impact_03`, `shimmer_01`. Use
`refset(cat)` (forge split) instead of kit paths. In rounds 1-11, v1's whoosh used `whoosh_med_01/02`; one bench take
(`fs-463763` 1.217-1.615 s) overlaps the `whoosh_med_01` trim, so v1's whoosh numbers are slightly optimistic too.

## What the benchmark measures (all against the category's bench refs)

| check | what | better |
|---|---|---|
| typ z | CLAP typicality: closeness to the nearest 3 bench refs, compared with how close the refs sit to each other (needs 3+ refs) | higher |
| tone dB | 1/3-octave tonal-balance distance to the bench median (`frozen_tone.py`: `earshot/forge/tone.py` frozen at round 0, so the forge's evolving copy never moves the benchmark) | lower |
| env oct | mean \|log2(made / bench median)\| of attack, t40 and active length (1 = typically 2x off) | lower |
| curve dB | RMS dB distance of the peak-aligned 10 ms envelope to the bench median envelope: where a tail that dies too fast shows | lower |
| t40 x | tail length / bench median | ~1 |
| phrase z, margin, wins | CLAP z of 2 independent phrasings per category, never a recipe TARGET / ALT (checked at start); margin over 4 category confusions | higher |
| AS names, as_p | an AudioSet top-3 label names the category; best label probability | higher |
| artifacts | clipped, dc offset, start click, end click, tail cut, late start (calibrated: 5 of 150 real trims trip one, all ticks cut from a flurry or plain 5 s windows) | none |
| var cos | mean pairwise CLAP cosine among a category's variants (real sets: 0.48-0.86, logo_sting's two forge refs 0.19) | lower = more varied |
| texture | floor dB (recording floor), stereo corr, darken (does the decay darken) | context |

Real ceilings, scored by the same functions:
- **real x**: the forge refs (real, but never in the bench) scored against the bench refs. This is the yardstick to
  compare made sounds with.
- **real loo**: each bench ref scored against the others.
- **real_rev** (JSON only): the bench refs scored against the forge refs.

**inside** counts, per made sound, how many checks land inside the real rows' 80% limits, shown next to the same
count for the real rows.

## Baseline: v1 finals (`results_v1.json`, `results_v1.md`)

Mapping: keypress → typing_key, click → click, pop → pop, whoosh → whoosh_fast, impact → impact. Medians:

| category | set | n | typ z | tone dB | env oct | curve dB | t40 x | phrase z | margin | AS names | artifacts | var cos | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| click | made | 3 | 1.89 | 7.6 | 0.56 | 8.85 | 0.56 | 3.12 | 1.13 | 0 | 0 | 0.886 | 7 / 7 (real 6) |
|  | real x | 8 | 0.16 | 11.33 | 0.22 | 9.92 | 1.33 | 2.77 | 0.53 | 0.12 | 0 | 0.651 |  |
|  | real loo | 7 | -0.1 | 9.85 | 0.46 | 13.19 | 0.86 | 2.67 | 0.27 | 0 | 0 | 0.576 |  |
| impact | made | 3 |  | 6.7 | 1.54 | 24.86 | 0.5 | 3.68 | 1.81 | 1 | 0 | 0.931 | 5 / 6 (real 3) |
|  | real x | 2 |  | 6.19 | 2.5 | 31.98 | 0.46 | 3.04 | 1.09 | 0.5 | 0 | 0.659 |  |
| pop | made | 3 | 1.33 | 8.59 | 0.68 | 6.29 | 1 | 3.09 | -0.1 | 1 | 0 | 0.917 | 7 / 7 (real 6) |
|  | real x | 7 | -0.09 | 16.06 | 1.51 | 20.54 | 0.91 | 2.46 | -0.5 | 0.29 | 0 | 0.501 |  |
|  | real loo | 6 | 1.13 | 13.66 | 0.41 | 12.5 | 1.02 | 1.96 | -1.06 | 0.67 | 0 | 0.522 |  |
| typing_key | made | 3 | 1.2 | 8.2 | 1.07 | 14.32 | 1.67 | 5.56 | 3.57 | 0.33 | 0 | 0.965 | 7 / 7 (real 6) |
|  | real x | 15 | 0.26 | 8.88 | 1.01 | 12.2 | 1.29 | 4.75 | 2.83 | 0.53 | 0 | 0.721 |  |
|  | real loo | 12 | 0.23 | 5.47 | 1.52 | 16.7 | 1 | 3.71 | 1.53 | 0.42 | 0 | 0.673 |  |
| whoosh_fast | made | 3 | 2.05 | 8.58 | 0.89 | 15.93 | 0.51 | 4.46 | 3.31 | 0.33 | 0 | 0.957 | 7 / 7 (real 6.5) |
|  | real x | 5 | 0.57 | 10.31 | 1.22 | 20.2 | 0.6 | 4.14 | 2.78 | 0.4 | 0 | 0.826 |  |
|  | real loo | 5 | 1.52 | 4.58 | 0.4 | 8.21 | 1.08 | 4.14 | 2.62 | 0.8 | 0 | 0.772 |  |

How to read it:
1. **Per sound, v1 cannot be told from real on these checks.** Each made sound lands inside the real 80% limits on
   7 of 7 checks (impact 5 of 6), while a typical real recording lands inside on 6. This includes the model-free tone
   and envelope distances against held-out refs. The reason: real recordings from different sources sit far from each
   other (real x tone 9-16 dB, curve 10-32 dB), so "as close as another real recording" is a loose bar. This is the
   same saturation the forge-test judge found, now measured on a held-out set.
2. **The clear made-vs-real gap is variety.** The 3 variants of each made category sit at pairwise CLAP cosine 0.89-0.97;
   real sets sit at 0.50-0.83. The forge's "diverse" picks sound alike. A user who hears three made clicks in a row
   will notice it.
3. **Made sounds are prototypes.** In every category, the made median typ z is above both real medians: click 1.89
   vs 0.16 / -0.1, pop 1.33 vs -0.09 / 1.13, typing_key 1.2 vs 0.26 / 0.23, whoosh_fast 2.05 vs 0.57 / 1.52. They sit
   nearer the centre of the held-out cluster than real recordings from other sources do. This fits a search pulled toward
   CLAP closeness. It is not a defect by itself, but it means typ z and phrase z cannot be pushed higher to show
   progress: they are already above real.
4. **Tails.** Made clicks have t40 0.56x the bench median, against the real refs' 0.86-1.33. Made whooshes are at
   0.51x, against 0.6-1.08. Both are short tails. Made impacts are at 0.35-0.62x the single held-out impact's 3.45 s
   tail. But the two forge impacts are just as short (0.38x, 0.54x), so one bench ref cannot settle it. One listener's
   note ("the tail died too fast") points the same way.
5. **Phrasings are saturated.** Phrase z and margin are above real in every category, as expected, so don't read
   progress from them. AudioSet naming is mixed: made sounds beat real for pop (1 vs 0.29 / 0.67) and impact (1 vs
   0.5), but fall below it for click (0 vs 0.12 / 0), typing_key (0.33 vs 0.53 / 0.42) and whoosh_fast (0.33 vs 0.4 /
   0.8). AudioSet rarely names a UI click at all.

So for the overnight rounds, progress is shown by: lower var cos (towards 0.5-0.8) with no tone / env regressions;
t40 x moving towards 1 (clicks, whooshes); tone / curve distances at or under real x; zero artifacts; and the new
categories getting rows at all. The final word is the blind test.

## Round 2: forge v2 families (`results_v2r2.json`, `results_v2r2.md`)

168 made sounds in 34 categories (18 real-recording + 16 pro designed), 4-6 variants each, from
`out/forge_v2/<category>/final/`. The full table, per-category verdicts and v1 comparison are in `results_v2r2.md`.
In short:
- **Clones fixed.** var cos v1 -> v2: click 0.886 -> 0.681, typing_key 0.965 -> 0.705, pop 0.917 -> 0.735, impact
  0.931 -> 0.881, whoosh_fast 0.957 -> 0.856 (real sets 0.48-0.86). Still clone-like: pro_whoosh_big 0.931,
  pro_typing_run 0.930, pro_typing_key 0.926, whoosh_big 0.909, pro_pop_in 0.895.
- **Tails fixed for clicks and keys, not for fast whooshes.** t40 x v1 -> v2: click 0.56 -> 1.22, typing_key 1.67
  -> 1.29 (real x 1.29), impact 0.50 -> 0.62, whoosh_fast 0.51 -> 0.49 (real 0.60 / 1.08). Shimmer 0.48 and notify
  0.63 are short; logo_sting 2.48 is long.
- **Model checks back to real level.** typ z fell from above real (v1 prototypes) to real-like (click -0.25,
  typing_key -0.15, pop -0.02), except whoosh_fast -4.85 and swish -2.8 (below real).
- **Artifacts:** 3 flags in 168 sounds, all deliberate (two pre-swells, one end-on-cut).
- **inside the real 80% limits** (made median vs a real recording): whoosh_big 7/7 (5), notify 7/7 (5.5), pop 7/7
  (6), typing_key 6/7 (6), hover_tick 6/7 (5), shimmer 6/7 (6), click 5/7 (6), toggle 5/7 (6), impact 5/6 (3),
  pro_zip 5/7 (5), whoosh_fast 4/7 (6.5; v1 7/7), swish 4/7 (6), swipe 4/7 (6), tap 3/7 (6), riser 3/6 (4).

## Limits

- The sets are small: 1-15 refs per split, and 1-3 source files in several categories. Real-limit quantiles from 2-3
  rows are rough.
- Real refs are trims cut by earshot find with fades. Their tails and floors partly reflect the cut. `floor dB` on
  short trims is mostly the trim's own tail, not a room floor.
- tone dB and the envelope checks come from the same family of measurements the forge can optimise (`tone=`). On
  bench refs they are still held out, but they are not independent of the forge's own ears in kind.
- AudioSet has no class for riser, downlifter or glitch; the label sets use the nearest classes. AudioSet hears only
  up to 8 kHz.
- Judge sounds by these numbers plus spectrograms plus ears. Never select or steer per candidate by the benchmark: it
  is held out only while it decides nothing.

## Benchmark changes (only additions, recorded)

| when | change |
|---|---|
| 2026-10-16 round 0 | refsets v1 (18 categories, 152 refs); bench.py v1 (checks above); frozen_tone.py from git 108763c; baseline results_v1 + results_real |
| 2026-10-16 round 2 | additions only: PHRASES / CONFUSIONS / AS_LABELS for the 16 pro_* categories (pro refsets from the reference films; the 18 original rows unchanged); `pro_riser` in ENDS_ON_PEAK; `gather()` reads `<category>/final/` when present and skips `audition*.wav` / `_*.wav` (forge's made+ref audition file had polluted four areas' first runs); results_v2r2 |
