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
| 2 | 05:20-08:55 | **forge v2 families.** Six maker areas (whoosh, sweep, impact, ui, tonal, digital) rewrote or added 22 recipes with CHARACTER parameters (length, brightness, sweep / pitch range, weight, texture) and FAMILIES of presets per category: 34 categories (18 real + 16 pro designed), 168 finals, 4-6 variants each, every member searched inside a narrow box around its preset, then one final run scoring the members together with robust re-renders; per-recipe model-free study checks (`measure` / `target_check` / `study_extra`) from docs/motion-sfx-study.md. Judge / integrator: benchmark additions (pro_* phrasings, `<cat>/final/` layout, audition.wav skipped), five core fixes (8-line reverb IR NaN, single-ref `score_refs` crash, flat_top false positive on subs, seed-bank cap 12 -> 48, `optimize` keeps DEFAULTS outside a narrow recipe SPACE), notify_04 start click re-rendered away, logo_sting audition rebuilt without the rejected variants. | Tests: 355 passed, 15 skipped (full fast suite, 416 s). Held-out bench `eval/forge/results_v2r2.md`: clones fixed (var cos v1 -> v2: click 0.886 -> 0.681, typing_key 0.965 -> 0.705, pop 0.917 -> 0.735, impact 0.931 -> 0.881, whoosh_fast 0.957 -> 0.856; real 0.48-0.86); tails fixed for click (t40 x 0.56 -> 1.22) and typing_key (1.67 -> 1.29 = real x), not for whoosh_fast (0.51 -> 0.49); 3 artifact flags in 168 sounds, all deliberate; critic passes 60 / 168. Best: whoosh_big (inside 7/7, real 5; AudioSet 'Whoosh' 5/5), typing_key, pro_pop_in. Worst: swipe (4-12 kHz 17 dB under the bench cloth), whoosh_fast + swish (tail 2x short, 8 dB boom, 15 dB hiss), logo_sting (rings 2.5x long). |

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

### Round 2 notes (judge + integrator, 08:15-08:55)
Per category on the held-out bench (made median; real x = forge refs vs bench, loo = bench vs bench; v1 = round-0
baseline, same frozen benchmark). Variety = mean pairwise CLAP cosine (lower = more varied); t40 x = tail / bench
median; tone = 1/3-octave distance to the bench median (dB). Pro categories without a bench split get variety and
artifact rows only (plus phrasings / AudioSet in the JSON).

| category | n | var cos made (forge / bench) | t40 x made (real x / loo) | tone dB made (real x / loo) | artifacts | inside (real) | v1: var / t40 x / tone / inside |
|---|---|---|---|---|---|---|---|
| click | 5 | 0.681 (0.651 / 0.576) | 1.22 (1.33 / 0.86) | 9.86 (11.33 / 9.85) | 0 | 5/7 (6) | 0.886 / 0.56 / 7.60 / 7/7 |
| downlifter | 6 | 0.749 (- / -) | - (- / -) | - (- / -) | 0 | - |  |
| glitch | 4 | 0.799 (- / -) | - (- / -) | - (- / -) | 0 | - |  |
| hover_tick | 5 | 0.807 (0.799 / 0.626) | 1.10 (0.90 / 1.20) | 20.99 (21.20 / 15.46) | 0 | 6/7 (5) |  |
| impact | 5 | 0.881 (0.659 / -) | 0.62 (0.46 / -) | 6.29 (6.19 / -) | 1 | 5/6 (3) | 0.931 / 0.50 / 6.70 / 5/6 |
| logo_sting | 4 | 0.733 (0.185 / 0.507) | 2.48 (1.70 / 1.12) | 16.29 (15.52 / 23.22) | 1 | 3.5/6 (5.5) |  |
| notify | 5 | 0.810 (0.593 / 0.476) | 0.63 (0.34 / 1.19) | 18.65 (21.30 / 16.70) | 0 | 7/7 (5.5) |  |
| pop | 5 | 0.735 (0.501 / 0.522) | 0.82 (0.91 / 1.02) | 11.42 (16.06 / 13.66) | 0 | 7/7 (6) | 0.917 / 1.00 / 8.59 / 7/7 |
| pro_click | 5 | 0.870 (- / 0.450) | 0.67 (0.53 / 1.08) | 25.79 (29.35 / 18.53) | 0 | 4/6 (4) |  |
| pro_glitch | 5 | 0.875 (- / -) | - (- / -) | - (- / -) | 0 | - |  |
| pro_hover_tick | 5 | 0.878 (0.820 / 0.684) | 1.17 (1.00 / 0.86) | 7.89 (6.51 / 8.32) | 0 | 4/7 (6) |  |
| pro_impact | 5 | 0.733 (0.612 / -) | - (- / -) | - (- / -) | 0 | - |  |
| pro_logo_sting | 4 | 0.947 (- / -) | 0.15 (0.06 / -) | 24.42 (15.49 / -) | 0 | 3/6 (6) |  |
| pro_notify | 5 | 0.882 (0.868 / -) | - (- / -) | - (- / -) | 0 | - |  |
| pro_pop | 5 | 0.663 (0.488 / -) | 0.81 (0.06 / -) | 23.68 (18.43 / -) | 0 | 4/6 (3) |  |
| pro_pop_in | 5 | 0.895 (0.696 / -) | - (- / -) | - (- / -) | 0 | - |  |
| pro_riser | 5 | 0.676 (- / -) | 0.05 (0.68 / -) | 20.80 (21.35 / -) | 0 | 3/6 (6) |  |
| pro_shimmer | 5 | 0.646 (0.712 / -) | - (- / -) | - (- / -) | 0 | - |  |
| pro_sub_drop | 5 | 0.835 (- / -) | 1.64 (- / -) | 24.84 (- / -) | 0 | - |  |
| pro_tap | 5 | 0.854 (- / -) | - (- / -) | - (- / -) | 0 | - |  |
| pro_typing_key | 5 | 0.926 (0.849 / -) | 0.64 (0.71 / -) | 26.38 (26.20 / -) | 0 | 4/6 (5.5) |  |
| pro_typing_run | 5 | 0.930 (- / -) | - (- / -) | - (- / -) | 1 | - |  |
| pro_whoosh_big | 5 | 0.931 (0.523 / -) | - (- / -) | - (- / -) | 0 | - |  |
| pro_zip | 5 | 0.761 (- / 0.529) | 0.84 (- / 1.15) | 11.89 (- / 10.40) | 0 | 5/7 (5) |  |
| riser | 5 | 0.780 (- / 0.767) | 0.06 (0.50 / 3.35) | 12.50 (17.61 / 13.22) | 0 | 3/6 (4) |  |
| shimmer | 5 | 0.807 (0.590 / 0.644) | 0.48 (0.73 / 1.05) | 10.49 (9.66 / 15.52) | 0 | 6/7 (6) |  |
| sub_drop | 5 | 0.727 (- / -) | 0.31 (- / -) | 8.84 (- / -) | 0 | - |  |
| swipe | 5 | 0.855 (0.845 / 0.857) | 0.86 (0.79 / 1.02) | 13.47 (16.83 / 3.64) | 0 | 4/7 (6) |  |
| swish | 5 | 0.860 (0.811 / 0.790) | 0.68 (0.64 / 1.16) | 7.48 (11.08 / 9.01) | 0 | 4/7 (6) |  |
| tap | 5 | 0.856 (0.713 / 0.716) | 1.27 (1.64 / 1.00) | 6.10 (9.35 / 16.78) | 0 | 3/7 (6) |  |
| toggle | 5 | 0.720 (0.477 / 0.725) | 1.26 (2.11 / 1.02) | 8.82 (12.48 / 6.25) | 0 | 5/7 (6) |  |
| typing_key | 5 | 0.705 (0.721 / 0.673) | 1.29 (1.29 / 1.00) | 8.15 (8.88 / 5.47) | 0 | 6/7 (6) | 0.965 / 1.67 / 8.20 / 7/7 |
| whoosh_big | 5 | 0.909 (0.700 / 0.495) | 1.04 (3.00 / 0.97) | 7.58 (8.80 / 4.25) | 0 | 7/7 (5) |  |
| whoosh_fast | 5 | 0.856 (0.826 / 0.772) | 0.49 (0.60 / 1.08) | 9.43 (10.31 / 4.58) | 0 | 4/7 (6.5) | 0.957 / 0.51 / 8.58 / 7/7 |

Judged by eye and measurement (spectrogram / envelope / 1/3-octave sheets of finals vs bench, forge and pro refs;
AudioSet as a held-out sanity check; nobody has listened yet, `final/audition.wav` is ready per category):
- **Best**: whoosh_big (inside 7/7 vs a real recording's 5, tone 7.6 dB vs real x 8.8, t40 x 1.04, AudioSet
  'Whoosh' 5/5; fault: variants near clones, 0.909), typing_key (variety, tail and curve at the real level; faults:
  stereo corr 0.34 vs bench 1.00, 5 dB extra <150 Hz thock), pro_pop_in (balance and envelopes overlay the 10 film
  refs; fault: near clones 0.895). Close behind: pro_whoosh_big, pro_typing_key / run, downlifter, pro_zip, sub_drop.
- **Worst**: swipe (muffled: 4-12 kHz share -35.9 vs bench cloth -19.2 dB, onset >6 kHz -27 vs -11.5 dB; AudioSet
  'Knock' / 'Door' on 2 of 5; the forge cloth refs are as dark, so the forge split steered it), whoosh_fast + swish
  (t40 130 vs 265 ms; <150 Hz 8 dB heavy; 4-12 kHz 15 dB hissy; AudioSet 'Knock' / 'Whack, thwack' / 'Door' on 6 of 10;
  whoosh_fast regressed vs v1: inside 4/7 vs 7/7), logo_sting (t40 2.14 vs 0.86 s; bell chords vs single-ring refs;
  the refset is not a composed launch sting).
- Next weakest: tap (phrase z 0.52 vs real 1.6-1.9), pro_hover_tick (phrase z 0.85 vs real 1.8-3.0), pro_glitch
  (median level -19..-24 dB re peak vs the film's -31: too dense, little burst drama), impact (dull onset: >6 kHz share
  of the first 15 ms -28.9 vs bench -21.6 dB), click (narrow 6 kHz / 250-500 Hz partial lines ring in 3 of 5).

Round 3 (from 08:55): fix the worst three first (swipe brightness + no thump; whoosh_fast / swish tail x2, -8 dB
<150 Hz, -10..15 dB 4-12 kHz, softer onset; logo_sting shorter ring or a redefined category), then clones in the pro
families (pro_whoosh_big, pro_typing_key / run, pro_pop_in, whoosh_big: real sets 0.45-0.85), impact onset, click
ring, typing_key width.

Core requests applied this round: `fx.reverb_ir` (8 lines: equal / opposite L-R sign columns gave a NaN IR on ~1-3 %
of seeds; now flipped without an extra rng draw, so every finite render is unchanged), `tone.ToneRef.check` + critic
(scoring the only ref leave-one-out crashed; now no tone row), `search.artifacts` flat_top (exact repeats only; a 60 Hz
sine dwelt ~50 samples near its crest), `search.save_seeds` cap 12 -> 48, `search.optimize` (DEFAULTS kept for keys a
recipe object's narrower SPACE leaves out), bench `gather()` / pro rows. Deferred to round 3:
- polish.aliasing reads a noise layer under a harmonic tone as aliasing (downlifter synth_fall 0.40-0.48, 0.036 with
  the noise muted): test only peaks off the harmonic grid.
- polish.ring flags band-limited designed sounds (sub drops z 300-675 at the 2 ms quefrency edge) and buzzy hums as
  'metallic comb ring': needs an edge / empty-band guard.
- polish.truncated and the 'hard' artifact rule penalise deliberate hard gates (glitch, riser, zip, pro_impact):
  designed-mode allowance for a gate that ends on silence.
- critic: real-mode pass rule ref_typ_z >= -2 blocks designed whoosh_fast / swish / swipe against small close-mic
  sets; with < 3 refs tone / env / polish have no spread (penalty 0 or capped): absolute fallback targets from the
  study table in designed mode.
- speed: fx.reverb rebuilds its IR every call (~0.3 s of 0.7 s); critic holdout re-runs AudioSet on the refs every
  forge() call.
- forge(): a first-class families= / variants= mode (score an exact member set, no diverse() drop, loudness-matched
  palette, per-category pick names, fixed= params); write audition as `_audition.wav`.
- pop module defaults still v1 (pinned by tests/test_forge.py and demo/forge-test/pop): switch to SPACE_V2 / DEFAULTS_V2.
- refsets: bed-subtracted pro refs (pro_notify's 13-14.5 kHz bed line), pro_pop split by sub-style, held-out trims
  for the 8 pro categories without a bench split, brighter (bench-like) cloth / paper swipes and designed fast
  whooshes on the forge side, a composed logo-sting definition; doc: pro_pop_in's 0.14-0.27 s is the trim window, the
  sound is 50-80 ms.
