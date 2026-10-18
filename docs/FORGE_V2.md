# forge v2: review notes for the `forge-v2` branch

> Released in 0.2.0 ([CHANGELOG.md](../CHANGELOG.md)). These notes were written on the branch before the merge, so
> "not merged" and "nobody has listened" below describe that time. Paths under `out/` are local to the dev machine:
> the made sounds, galleries and blind-test kits hold audio, and the reference recordings are licensed.

What changed in `earshot forge` during the overnight iteration of 2026-10-16, how to try it, what the held-out
measurements say, what is still weak, and what to check before merging. The branch is not merged; `main` is
untouched. The round-by-round log with every number is [the forge v2 log](history/FORGE_V2_LOG.md). Per-category verdicts are in
[eval/forge/results_v2r6.md](../eval/forge/results_v2r6.md) (Judge notes, round 6),
[results_v2r5.md](../eval/forge/results_v2r5.md) (round 5),
[results_v2r4.md](../eval/forge/results_v2r4.md) (round 4) and [results_v2r3.md](../eval/forge/results_v2r3.md)
(round 3).

**Status (round 6 benchmarked at 19:03; round 5 at 15:58):**
- 34 categories, 183 sounds made in code, 0 artifact flags on the held-out benchmark.
- Every judgement so far is a measurement or a model check: **nobody had listened to the sounds yet.**
- The clean blind test is ready in `out/showcase/blind_v2/`: 20 trials, loudness-matched, randomised, real refs from
  the held-out side, rebuilt on the final finals. Its cleanliness report passes: both clips of a trial are trimmed
  the same way, play 0.00 LU apart at their loudest moment, and peak at most -1 dBTP. It is the deciding test, and
  its result belongs in this document and in the README before a merge.
- The overnight showcase is `out/showcase/SHOWCASE.md` (local).

## 1. What was built

| piece | where | what it does |
|---|---|---|
| reference sets | `earshot/forge/refsets.py`, `eval/forge/build_refsets.py` | 18 motion-design categories, 81 'forge' refs (the search may use them) and 71 'bench' refs (only the benchmark hears them), split by source family (author / library / file) so sister takes never cross the split; 0 leaks. `held_out()` keeps bench files and 8 bench-source kit WAVs out of every search. The licensed audio and the registry stay local. |
| held-out benchmark | `eval/forge/bench.py`, `frozen_tone.py`, `results_*.json / .md` | per category, made sounds against the bench refs with checks the forge never optimises: 1/3-octave tone distance (a frozen copy of round 0's tone.py), envelope (attack, t40, length, curve), independent CLAP phrasings, AudioSet names, artifacts, variety between variants; next to the same checks for real recordings ("real x" = forge refs vs bench, "loo" = bench vs bench) and an "inside the real 80% limits" count. Run once per round; additions only. |
| critic v2 | `earshot/forge/critic.py`, `polish.py`, `tone.py` | the v1 model terms (CLAP match, realness, typicality) plus model-free penalties: tonal balance, spectro-temporal envelope (a tail that dies too fast), artifacts (clicks, aliasing, DC, harsh resonance, comb ring, cut tails, clipping) and polish (attack, onset, tail smoothness); a designed mode for categories without real refs; study targets where refs are thin. `version=1` reproduces v1. |
| search v2 | `earshot/forge/search.py`, `earshot/forge/__init__.py` | model-free pre-screen (many renders ranked by tone / envelope / polish, CLAP only hears the best), CMA-ES option, threaded renders, robust re-renders on new seeds with a shortlist, seed bank, `report.md` / `compare.png` / audition reel; `forge(families=)` searches each member of a character family in its own box and `palette()` loudness-matches the set. |
| DSP toolkit | `earshot/forge/dsp.py`, `fx.py` | sources (air, Shepard risers, sub drops, bells, FM, sweeps, layering) and effects (hybrid FDN / noise-tail reverb, saturation, transient shaping, limiter, EQ match, Doppler, glitch, granular). |
| pro study | `docs/motion-sfx-study.md`, `docs/motion-sfx-techniques.md`, `eval/forge/pro/` | 58 designed SFX events measured in 8 professional launch films (anonymous ids ref_1..ref_8; one SFX-only film supplies 39 of them), bed-subtracted balance, attack, decay shape, level over music; plus a cited research note on how pros build each kind of effect. Gives the 16 `pro_*` categories their targets and local refs. |
| recipes and families | `earshot/forge/recipes/` (20 modules) | 34 categories (18 real-recording categories + 16 `pro_*` designed ones), each a family of 4-6 named characters (crisp / heavy / sleeve ..., length, brightness, sweep, weight, texture). 183 finals in `out/forge_v2/<category>/final/` (local). |
| gallery | `earshot/gallery.py` (`earshot gallery`) | one static listening page per run: players, spectrogram thumbnails, loudness, critic and benchmark numbers, parameters, before / after rows (`--compare`), one self-contained file (`--embed-compressed`). Bench refs never reach a page. |
| blind test + progress | `eval/forge/blind_v2.py` | builds the 20-trial blind test (made finals vs held-out real recordings, loudness-matched, answer sheet, scorer, sealed key) and the v1 -> v2 report with a listening reel, from whatever is in `final/` now. |
| path scrub | `eval/forge/scrub_paths.py` | rewrites machine paths in benchmark results to `<refs>/...`, `<refs_pro>/...`, `<EARSHOT_HOME>/...` or repo-relative; `--check` for a pre-merge test. |

Rounds: 1 = tooling (refsets, benchmark, critic v2, search v2, DSP); 2 = families for 34 categories (168 finals);
3 = fixes for the round-2 judge's faults in 29 categories (183 finals; round-2 finals kept in `final_r2/`), core
fixes (polish false flags, critic thin-ref fallback, reverb IR cache), `forge(families=)`, the gallery. Round 4
re-delivered the finals of 14 categories: whoosh_fast, whoosh_big, swish, swipe, riser, impact, pro_impact,
logo_sting, pro_logo_sting, tap, pro_tap, toggle, pro_pop and notify. Their round-3 finals are in `final_r3/`. The
full benchmark (`results_v2r4.json`) kept all 14; the close calls are in its Judge notes. Round 5 re-delivered
whoosh_fast, tap, logo_sting, pro_logo_sting, pro_pop and pro_notify (round-4 finals in `final_r4/`) and fixed the
core fill bug (section 4). The full benchmark (`results_v2r5.json`) kept 4 of the 6 and put round 4 back for tap and
pro_pop (their round-5 attempts are in `final_r5/`). Everything measured below is the final round-5 tree.

## 2. How to try it

On the dev machine (the outputs in `out/` and the licensed refs are local; on a fresh clone the forge runs in designed
mode or against your own refs, see `earshot/forge/refsets.py`):

```bash
# listen: one self-contained page (MP3 previews of the WAVs, every number measured on the WAVs), opens anywhere
out/showcase/gallery_embedded.html
# v1 (main) above v2 (final) for the five categories v1 had, with both benchmark runs, one self-contained file
out/showcase/v1_vs_v2_embedded.html
# private pages: the nearest forge ref under each sound (plays licensed refs), and round 4 vs round 5 rows for the 6
# categories round 5 touched (links local files)
out/showcase/gallery.html  out/showcase/compare.html
# the blind test: 20 trials, real (held-out) vs made, loudness-matched; write A / B per trial, then score
out/showcase/blind_v2/answer_sheet.txt  blind_test.wav  (score_answers.py; do not open KEY_open_after.json first)
# v1 -> v2 per check, per category, with a listening reel
out/showcase/progress/progress.md  v1_vs_v2.wav

# rebuild the pages
earshot gallery out/forge_v2 -o out/showcase/gallery_embedded.html --embed-compressed \
  --results eval/forge/results_v2r5.json --title "earshot forge v2: sound gallery"
earshot gallery out/forge_v2 -o out/showcase/gallery.html --refs forge --results eval/forge/results_v2r5.json
earshot gallery --compare r4:out/forge_v2/whoosh_fast/final_r4 r5:out/forge_v2/whoosh_fast/final ... \
  -o out/showcase/compare.html --results r4:eval/forge/results_v2r4.json r5:<the as-run round-5 results>
  # (one r4: / r5: pair per round-5 category; for tap and pro_pop r4 = final, r5 = final_r5)
earshot gallery --compare "v1:demo/forge-test/*/final" v2:out/forge_v2/typing_key/final v2:out/forge_v2/click/final \
  v2:out/forge_v2/pop/final v2:out/forge_v2/whoosh_fast/final v2:out/forge_v2/impact/final \
  --alias keypress=typing_key whoosh=whoosh_fast --results v1:eval/forge/results_v1.json \
  v2:eval/forge/results_v2r5.json --embed-compressed -o out/showcase/v1_vs_v2_embedded.html

# re-render a delivered final from its parameters (same samples up to the palette's level match, max error ~1e-7);
# the argument is the recipe MODULE: logo_sting -> sting, typing_key -> keypress, sub_drop -> subdrop (section 4)
earshot forge --render click --params out/forge_v2/click/final/report.json --dir made/click_again

# forge new sounds
earshot forge --list                                                       # recipes, their targets, refsets
earshot forge whoosh --refset whoosh_fast --budget 96 --variants 4 --dir made/whoosh
python -c "from earshot.forge import forge; forge('click', refset='click', budget=48, robust=2, \
  families={'soft': {'pitch': 3000.0, 'body': 0.2}, 'tink': {'pitch': 5000.0}}, out='made/click_palette')"
#   (a palette: one search per member in a narrow box around its preset, budget per member, then the members scored
#    together and loudness-matched; the recipes' own presets are in FAMILIES / PRESETS in each recipe module)

# the held-out benchmark (loads CLAP + AudioSet; a few minutes at EARSHOT_THREADS=2; writes scrubbed paths)
EARSHOT_THREADS=2 python eval/forge/bench.py --made out/forge_v2 --out eval/forge/results_v2rN.json
python eval/forge/scrub_paths.py --check

# tests (fast suite, no models; 5-9 min on this laptop)
python -m pytest -q
```

## 3. Before / after on the held-out benchmark

v1 = the five recipes of `main` (3 sounds each, `eval/forge/results_v1.json`). v2 = the final round-5 delivery
(`results_v2r5.json`); pop is the round-2 family, which the round-3 judge kept. Both use the same frozen benchmark and
the same bench refs. The brackets show what real recordings score on the same check: the bench refs among themselves
for variety; real x = the forge refs scored against the bench; loo = bench vs bench.

| category | variety: CLAP cos between variants, lower = more varied | tail t40 / bench median (1 = same) | tone distance dB, lower = closer | curve dB | checks inside the real limits |
|---|---|---|---|---|---|
| typing_key | 0.965 -> **0.713** (0.673) | 1.67 -> **1.00** (1.29 / 1.00) | 8.20 -> **7.44** (8.88) | 14.3 -> **9.1** | 7 -> 4 of 7 (real 6) |
| click | 0.886 -> **0.723** (0.576) | 0.56 -> **1.44** (1.33 / 0.86) | 7.60 -> 8.79 (11.33) | 8.9 -> 8.8 | 7 -> 7 of 7 (6) |
| pop | 0.917 -> **0.735** (0.522) | 1.00 -> 0.82 (0.91 / 1.02) | 8.59 -> 11.42 (16.06) | 6.3 -> 11.0 | 7 -> 7 of 7 (6) |
| whoosh_fast | 0.957 -> **0.831** (0.772) | 0.51 -> **0.88** (0.60 / 1.08) | 8.58 -> **6.21** (10.31) | 15.9 -> **10.1** | 7 -> 4 of 7 (6.5) |
| impact | 0.931 -> **0.849** (forge refs 0.659) | 0.50 -> 0.58 (0.46 / -) | 6.70 -> **4.38** (6.19) | 24.9 -> 30.1 | 5 -> 6 of 6 (3) |

How to read it:
- **Clones fixed.** v1's variants were near copies (0.89-0.97). v2's sit at or near the real sets' spread: whoosh_fast
  (0.83) is inside its real rows (bench 0.77, forge refs 0.83). impact is still more alike than its forge refs.
- **Tails fixed** where v1 died too fast: click 0.56 -> 1.44, whoosh_fast 0.51 -> 0.88. typing_key's long tail went
  1.67 -> 1.00. impact's tail is still short against the one long bench hit; the forge refs score 0.46 there too.
- **Tone.** Closer in 3 of 5: typing_key, whoosh_fast, impact. click and pop are further, but still closer than a
  different set of real recordings of the same kind (real x).
- **The model checks fell, on purpose.** v1 sat *above* real recordings on CLAP typicality and phrasings: the
  Goodhart problem in the README. It was tuned until the models were pleased, partly against kit refs that are bench
  refs now, so v1's benchmark numbers are optimistic. impact most of all: its bench hit was a v1 forge ref.
  typing_key's and whoosh_fast's lower "inside" counts are CLAP / AudioSet misses.

Breadth (no v1 counterpart): 29 new categories.
- **Tone.** 21 categories have a bench tone number. In 18 of them (round 3: 17) the made median is at least as close
  to the held-out balance as the real forge refs. The exceptions are pro_logo_sting (15.6 vs 15.5 dB), pro_pop
  (22.9 vs 18.4) and shimmer (10.7 vs 9.7).
- **Inside the real limits.** In 11 of 22 categories the made median lands inside on at least as many checks as a
  real recording.
- **Artifact flags:** 0 of 183 (round 2: 3 of 168).
- **Thin coverage.** Ten categories have no bench refs at all (downlifter, glitch, pro_glitch, pro_impact, pro_notify,
  pro_pop_in, pro_shimmer, pro_tap, pro_typing_run, pro_whoosh_big). They are measured on phrasings, AudioSet,
  artifacts and variety only. Three more (sub_drop, pro_sub_drop, pro_zip) have bench refs but no forge refs to compare
  against.

Round 3 -> round 4, in the 14 categories round 4 changed (full table in `results_v2r4.md`):
- **whoosh_fast:** variety 0.886 -> 0.768. AudioSet's top label is door-like on 6 of 6 -> 3 of 6.
- **swish:** AudioSet names it 0 -> 0.8, and typ z goes -3.1 -> 0.6.
- **swipe:** stereo corr 0.90 -> 0.64 (bench 0.68), and the tail no longer darkens.
- **whoosh_big:** variety 0.854 -> 0.760.
- **toggle:** tone 10.4 -> 7.8 dB.
- **tap:** tone 8.2 -> 5.8 dB, t40 x 1.82 -> 1.00.
- **notify:** tone 18.3 -> 15.6 dB.
- **impact:** variety 0.905 -> 0.849, tone 5.3 -> 4.4 dB, and the onset tick is 6.5 dB darker.
- **pro_logo_sting:** tone 24.4 -> 15.9 dB.
- **logo_sting:** tone 16.4 -> 13.1 dB, but env oct is worse.
- **riser:** tone 11.3 -> 9.3 dB, but env oct and curve are worse.
- **pro_impact:** the model checks prefer round 3. It was kept on the spectrogram, because round 3 carried an
  accidental onset tick.

Round 5 -> round 6 (18:00-19:40, after the showcase), full table and reasons in `results_v2r6.md`:
- **pro_pop (round 6 kept):** six natural pops with a gliding pitch (`n6` in pop.py); variety 0.649 -> 0.617, tone
  22.9 -> 19.5 dB (real 18.4); CLAP '8-bit' in the top 3 on 0 of 6 (was 3). Phrase z and margin fell.
- **riser (round 6 kept):** a new render for 3 of 5 members (`r6=1`: rising band without a resonant peak, rising chord,
  accelerating gate, reverse swell); margin -0.30 -> +0.08, variety 0.801 -> 0.772, env oct 2.35 -> 2.05; tone
  9.3 -> 12.8 dB (inside the real rows).
- **tap (round 6 kept):** the better of round 4 / 5 per member; variety 0.782 -> 0.761, env oct 1.40 -> 1.22; margin and
  typicality slightly down.
- **glitch (round 6 kept):** a chopped-repeat / bit-crush / dropout path (`roll0`); all three model checks up (no real
  glitch refs exist).
- **pro_impact (round 4 kept):** round 6's boom made it recognisable (AudioSet 0.6 -> 1.0) but less varied (0.788 ->
  0.820) and narrower (corr 0.35 -> 0.67); round 6 is in `final_r6/`.
- **pro_notify (round 5 kept):** five struck materials read acoustic; both model checks fell (not delivered).
- **Reports:** one schema for every final (`earshot/forge/schema.py`, `earshot forge --check DIRS`): all 183 picks
  re-render from their `report.json` within 1.19e-7.

Round 4 -> round 5, in the 6 categories round 5 touched (full table and reasons in `results_v2r5.md`):
- **pro_logo_sting (kept):**
  - Better: variety 0.908 -> 0.642 (five distinct launch gestures), tone 15.9 -> 15.6 dB, env oct 2.50 -> 2.15,
    margin -1.30 -> -0.68. CLAP 'gunshot' went from 2 of 5 to 0.
  - Worse: stereo corr 1.00 -> 0.88, against one mono music chord. The wide tail is deliberate.
- **whoosh_fast (kept):**
  - Better: tail darkening -0.05 -> -0.24 (bench -0.2); AudioSet names it 0.5 -> 1.0.
  - Costs: variety 0.768 -> 0.831 (inside the real rows); CLAP typicality -4.25 -> -5.44.
- **logo_sting (kept):** curve and darkening are slightly better. Decay roughness 0.46 -> 0.34 dB; reverse_bloom's
  pre-hit dip 14.6 -> 2.0 dB. env oct is unchanged (1.68).
- **pro_notify (kept on the spectrogram):**
  - The film dings' 2 kHz head -> 1 kHz body, without round 4's broadband hiss. Phrase z and margin went up.
  - Variety is worse, 0.822 -> 0.936: one patch on a semitone ladder.
- **tap (round 4 put back):** round 5 lost on variety, curve and stereo width, and won only env oct. Its model checks
  went up.
- **pro_pop (round 4 put back):** round 5 removed the '8-bit' reading, but lost on variety (0.649 -> 0.780) and on both
  model checks. Four of its six members became near-steady beeps.

Measured best:
- whoosh_big, swish, shimmer, click, pop, impact, notify
- typing_key, pro_typing_key / run, pro_pop_in, pro_hover_tick, pro_click, pro_whoosh_big
- logo_sting, swipe, whoosh_fast, pro_logo_sting

Weakest:
- pro_pop: two members still read '8-bit'. The round-5 fix was not kept.
- tap: the model checks are low.
- pro_notify: clean, but near clones (0.936).
- riser: AudioSet's nearest confusion is 'a jet engine starting'.
- pro_impact, glitch, pro_tap.

## 4. What is still weak

- **Ears.** All of the above is measurement. A forge result is only good when a blind listener cannot pick it out
  next to real recordings and a motion designer would use it; that test had not been taken at the time of writing.
- **The models and the measurements disagree.** AudioSet still hears 'Door' / 'Knock' in whoosh_fast and swish whose
  balance, tail and envelope now sit inside the real limits; CLAP checks fell in typing_key and notify after fixes the
  model-free checks liked. The trust order used overnight (model-free first, models as held-out hints) is a choice.
- **Thin references.** impact has 2 forge refs and 1 bench ref; riser 1 / 2; logo_sting has no real stings (the refs
  are the rings a sting ends on); downlifter, glitch and sub_drop have no refs (designed mode only); most `pro_*`
  categories rest on 1-10 events, mostly from one film. Numbers there describe little more than those few sounds.
- **A core bug found in round 4, fixed in round 5.**
  - The bug: `forge()`'s final pass filled the SPACE keys a member's preset leaves out at mid-range
    (`search._encode` used 0.5 for a missing key), not from the recipe's DEFAULTS. pro_impact's round-3 finals got an
    unintended 0 dB onset tick this way (snap 0.75 + sheen 1.0).
  - The fix: `search.fill_start` fills from the start point, then the recipe object's DEFAULTS, then the DEFAULTS of
    the module that defines its `render()`. It falls back to mid-range only when no DEFAULTS has the key, and then
    raises a `MidRangeFill` warning (also listed in `report['fill_warnings']`). Start points are scored exactly as
    given.
  - An audit of all 34 finals found hidden mid-range values only in pro_impact (four keys). It stays as delivered,
    because its swell member measures worse with the DEFAULTS values: `pre_shape` 1 is the shape it needs.
  - The seed banks now pin every delivered member (`save_seeds(..., delivered=)`, `pin_delivered()`): 183 pins.
- **Critic pass counts mean little across rounds** (58 / 183 now, 55 / 183 in round 3, 60 / 168 in round 2): the
  critic changed while the makers were using it, and areas ran different critic setups.
- **Held-out discipline was good, not perfect.** Each area ran the benchmark once at the end of its work; the UI area
  ran it four times during round 3 (recorded). In round 5 one area ran it twice (candidates, then the final tree).
  The core area spent its one run on a typing_key re-render that it then withdrew; typing_key's final was restored
  from its original files and measures exactly as in round 4. The seed banks (`earshot/forge/seeds/*.json`) still
  hold v1 finals that were partly tuned against kit refs that are bench refs now (start points, not refs: a soft
  leak).
- **Reproducibility.** Re-rendering every round-3 final from its `report.json` (parameters + seed) with the recipes
  as they stood at 12:00 gave the same samples (max error 3e-7 after the palette's level match) in **31 of 34
  categories**. Not reproducible that way: pro_typing_run (composed runs: the length differs), pro_shimmer (3 of 5
  differ; its finals are from round 2 and the recipe changed in round 3) and pro_notify (1 of 5). The 14 categories
  round 4 re-delivered were checked again on the final code (13:55, all 75 members): every one re-renders to max
  3e-7 after peak normalisation. Round 5's deliveries re-render to max 2.5e-7, and pro_notify now reproduces 5 of 5,
  so 32 of 34 categories reproduce (not pro_typing_run or pro_shimmer). The restored round-4 tap and pro_pop finals
  re-render on the final code to max 1.6e-7. **Round 6:** one report schema (`earshot/forge/schema.py`): every pick
  carries a `render` block (module, function, member, full params, seed, peak, gain), `check_final()` re-renders a
  final sample by sample, and all 34 finals (183 picks) re-render within 1.19e-7 (`tests/test_forge_rerender.py`).
  pro_typing_run and pro_shimmer reproduce again. Before round 6 the report schema
  was not uniform: 12 reports name the category in `recipe` instead of the module (some add
  `recipe_module`), and the tonal and pop finals keep their parameters in `family_params` (merge them under
  `params`), so `earshot forge --render` alone does not reproduce every category yet.
  The per-category driver scripts that ran the family searches live in `out/forge_v2/<category>/` and are not
  committed; `forge(families=)` (round 3) is the committed way to run a palette.
- **Code shape.** Family presets are not uniform across recipes (`FAMILIES`, `FAMILY`, `PRESETS`, `FAMILIES_BY_CAT`,
  plus `FAMILIES_R2` / `_R3` history in impact, sting, riser, zip); `impact.py` and some others carry a lot of round
  history in comments and constants. Worth a consolidation pass before this becomes public API.
- **Repo weight.** The four benchmark JSONs in `eval/forge/` are ~1.7 MB (every row of every run). Keep v1, real and
  the last round, or move the per-row detail out.

## 5. Checklist before merging

1. **Listen.** `out/showcase/gallery_embedded.html` for the whole set; the blind test (`out/showcase/blind_v2/`,
   20 trials, chance = 10, 15 or more right has p = 0.021) for the deciding call.
   When a v2 result exists, publish it in the README's "Blind test" paragraph and in docs/SFX.md "Limits" together,
   with the number of listeners and trials (today both say that no v2 score is published; the README's paragraph is
   the v1 test, 8 of 15, one listener).
2. **No media, no names.** `git ls-files | grep -iE '\.(wav|flac|mp3|ogg|m4a|mp4)$'` returns nothing (checked
   12:10); the pro films appear only as ref_1..ref_8 (the map is `out/refs_pro/films_map.json`, local; at 12:10 no
   tracked file named a film file or its folder).
3. **No machine paths.** `python eval/forge/scrub_paths.py --check` exits 0, and
   `git grep -nE "E:/|C:/Users|Downloads"` shows only deliberate examples. `bench.py` now calls
   `scrub_paths.scrub()` before it writes; `tests/test_gallery.py` fails if a committed results file still has a
   machine path.
4. **Tests.** `python -m pytest -q` (fast suite) passes on the final tree: 433 passed, 15 skipped (16:20, run in
   two halves: forge 292 passed / 1 skipped in 7.4 min, the rest 141 passed / 14 skipped in 1.3 min). `--slow` loads the models.
5. **Compatibility.** The Python `forge()` defaults still reproduce v1 (strategy 'es', no pre-screen, no bank); the
   CLI defaults changed to the v2 setup (pre-screen 3 x pop, seed bank, robust 3, shortlist 2, critic v2 tone weight,
   the recipe's REFSET as default refs). That is a user-visible change: say so in the release notes.
6. **Scope.** Decide which results JSONs to keep, whether the seed banks ship in the package, and whether the
   `pro_*` categories (targets from one film for most) are public categories or study material.
7. **Docs.** README forge / gallery sections, this file and the forge v2 log agree on the final round's numbers
   (`results_v2r5.json`, 15:58).
