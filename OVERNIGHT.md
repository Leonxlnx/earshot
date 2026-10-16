# forge v2: overnight iteration log (branch `forge-v2`, not merged)

Goal (from the user, 2026-10-16 ~02:40): iterate ~12 h to make earshot much better, above all **forge**: sound
effects that are genuinely great for motion-design launch videos (whooshes, swishes, risers, downlifters, impacts,
sub drops, UI clicks / taps / toggles / hovers, pops, notification dings, shimmers, glitches, typing, logo stings),
and better outputs. At the end (~15 h): a showcase of the progress and a new clean blind test. Work only on this
branch; the user verifies before anything reaches `main`.

## How progress is measured (so the overnight work cannot fool itself)
- A fixed **forge benchmark** (`eval/forge/`): per category, held-out real / pro reference recordings that the forge
  never optimises against, independent text phrasings, model-free tone and envelope distances, AudioSet labels,
  artifact checks and variant diversity. Run at the start (baseline = the v1 recipes) and after every round.
- The forge's own critic may change; the benchmark does not (only additions, recorded here).
- Human ears decide at the end: a clean, loudness-matched, randomised blind test.

## Rounds
| round | when | what | result |
|---|---|---|---|
| 0 | 02:40 | branch, plan | |
| 1 | 02:50-05:15 | **forge v2 tooling.** Refsets: 18 motion-design categories, 81 forge / 71 bench refs, split by source family, 0 leaks (`earshot/forge/refsets.py`, `eval/forge/build_refsets.py`). Fixed benchmark `eval/forge/bench.py` + baseline. Critic v2: model-free envelope (EnvRef), artifact detectors and polish (`polish.py`), designed mode; `version=1` reproduces v1. Search v2: model-free pre-screen (tone + envelope + polish, CLAP only on the best), CMA-ES option, threaded renders, robust shortlist, seed bank, report.md / compare.png / audition.wav. DSP v2: `dsp.py` sources (air, shepard, sub_drop, bell, fm, sweeps, stack) + `fx.py` (hybrid FDN / noise-tail reverb, saturation, transient, limiter, eq_match, doppler, glitch, granular). Integration: recipes get `REFSET` (CLI default refs = its forge split), held-out guard (`refsets.held_out`: bench files and the 8 bench-source kit WAVs never reach a search), v2 components in report history / ref rows / report.md, CLI `--mode`, `--tone` default = critic v2 weight (0 = report only), critic typicality needs >= 3 refs (2 refs gave z ~1e5), compare.png envelopes smoothed for long sounds. | Tests: 236 passed, 15 skipped (full fast suite, 197 s). Smoke (critic v2 + pre-screen + forge refsets, budget 24-32): whoosh 0.7 min, impact 2.4 min; all outputs written. Baseline (v1 finals vs bench, `eval/forge/results_v1.md`): variety (CLAP cos between variants) 0.89-0.97 vs real 0.50-0.83; tail t40 x click 0.56 / whoosh 0.51 (real 0.86-1.33 / 0.60-1.08); tone dB click 7.6, pop 8.6, key 8.2, whoosh 8.6, impact 6.7 (real forge-vs-bench 11.3 / 16.1 / 8.9 / 10.3 / 6.2); model checks saturated (typ z, phrasings above real everywhere). |

### Round 1 notes (integrator)
- What works end to end: `earshot forge <recipe> [--refset CAT]` -> critic v2 inside search v2 (pre-screen reuses the
  critic's PolishRef) -> picks re-rendered on new seeds -> WAVs + report.md + compare.png + audition.wav; then
  `eval/forge/bench.py --made made_<tag>` against the held-out side. The smoke picks ran through the benchmark too
  (out/forge-r1/results_smoke.md; tiny budgets, not a progress claim): whoosh var cos 0.966, t40 x 0.47 (still clones,
  still short); impact var cos 0.72, t40 x 0.67, tone 9.1 dB (v1's 6.7 was optimistic: it used the bench impact).
- Gaps the smoke runs show (compare.png): made whooshes reach -70 dB by ~270 ms where refs still sit at -50 dB at
  300-400 ms, but the critic sees no problem because the whoosh_fast *forge* refs are themselves short (real x t40 x
  0.6): optimising against them cannot fix the tail. Made impacts are 5-15 dB thin at 1-10 kHz and carry a narrow
  ~70-80 Hz partial; the impact refs' tails end in the trim fade (a curved drop at 4.5-6 s), which the envelope check
  reads as natural.
- Known limits for round 2: impact has 2 forge refs (no typicality, LOO ref rows get no env / polish ranges, so real
  refs score 0 penalties by construction); downlifter / glitch / sub_drop have no forge refs (designed mode, the
  pre-screen ranks artifacts only); polish on 7 s sounds costs ~1 s each (impact pre-screen 53 s of a 2.4 min run);
  the seed bank's impact / whoosh entries are v1 finals tuned partly against now-bench kit refs (parameters only, a
  soft leak: start points, not refs). `eval/forge/pro/` (a pro motion-SFX study by another agent, still in progress)
  is not part of this commit.
