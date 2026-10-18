# forge v2: overnight iteration log (historical; forge v2 shipped in 0.2.0)

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
| 3 | 08:55-11:30 | **forge v2 fixes + core r3.** Five maker areas fixed the round-2 judge's faults in 29 categories (183 finals, round-2 finals kept in `<category>/final_r2/`): whoosh engine (continuous hiss band, long concave tail, `low_lag`, shelves, fabric rustle) for whoosh_fast / swish / swipe / whoosh_big / pro_whoosh_big; logo_sting redefined as a composed reveal (lead-in -> strike -> note / dyad -> short tail); impact contact sheen; UI (click room fed only the transient, toggle first contact loudest, tap contact + two-stage body, pro_click bodies, pro_hover_tick spike + grains, pro_pop_in families, pop snap); tonal (shimmer / notify free rings, pro_notify head, pro_pop chirp controls, riser / pro_riser / pro_zip noise layers); digital (typing_key width + thock, pro_typing grain / blip members, pro_glitch burst drama). Core: polish false flags (aliasing on the fold grid, ring narrow / pitch guards, designed / gated gates), critic thin-ref fallback (`Critic(category=)`, STUDY_TARGETS), reverb IR cache + fixed rooms, holdout / score_refs caches, `palette()` and `forge(families=)`. New: `earshot gallery` (static listening page, compare mode). Judge: full bench, own spectrogram sheets, round-2 finals put back for pop and pro_zip (r3 worse on the bench; r3 kept in `final_r3/`). | Tests: 400 passed, 15 skipped (full fast suite, 316 s). Held-out bench `eval/forge/results_v2r3.md`: swipe tone 13.5 -> 6.1 dB (4-12 kHz -35.9 -> -20.7, bench -19.2); whoosh_fast t40 x 0.49 -> 0.90, tone 9.4 -> 5.8 dB; swish t40 x 0.68 -> 1.04; logo_sting t40 x 2.48 -> 1.03, inside 3.5 -> 6 of 6; shimmer t40 x 0.48 -> 0.97, curve 14.1 -> 4.1 dB; typing_key stereo corr 0.34 -> 0.96; pro_click tone 25.8 -> 15.4 dB; clones fixed in pro_typing_key / run (0.93 -> 0.81) and pro_pop_in (0.90 -> 0.75); 0 artifact flags in 183 (r2: 3). Costs: AudioSet still hears door / knock in whoosh_fast / swish, impact contact now 11 dB brighter than the bench, model checks fell in typing_key / notify. |
| 4 | 11:30-14:00 | **forge v2 polish + finalize.** Three polish areas re-delivered 14 categories (round-3 finals in `<category>/final_r3/`). whoosh: the 'door' reading measured to the tail (a dark room ringing on the peak band), hiss that outlives the body, an accelerating 130 ms approach for whoosh_fast, side-signal width for swipe, new optional keys (`hiss_pre/tail/mfloor/width`, `low_width`, `room_damp`). impact: a quieter band-limited sheen, `plate` / `scatter` layers, a low strike under logo_sting, `zap` / `ctick` and five gestures for pro_logo_sting, `grain_jit` for pro_impact (its r3 0 dB onset tick came from forge's mid-range fill of missing SPACE keys). ui-tonal: toggle spring -> noisy rattle + steep high-pass (`hp_steep`), tap contact low-pass (`contact_lp_x`), pro_tap kick cue capped, notify `strike_max`, riser without drones, pro_pop `c_air` on one member; pro_notify tried twice, not delivered. Every new key defaults to the old behaviour. Also: the launch film re-scored with forge v2 (`earshot_launch_v2.mp4` in the local earshot-launch project). Then a clean blind test and a v1 -> v2 progress file (`eval/forge/blind_v2.py`). The gallery now has `--embed-compressed` and `--alias`. New: `docs/FORGE_V2.md` and `eval/forge/scrub_paths.py` (`bench.py` now writes scrubbed paths). The finalizer ran one full bench, kept r4 in all 14 categories and rebuilt the blind test, progress file, galleries and film. | Tests: 416 passed, 15 skipped (fast suite, final tree). Held-out bench `eval/forge/results_v2r4.md`, r3 -> r4: whoosh_fast variety 0.886 -> 0.768, darken -0.64 -> -0.05, AudioSet door-like top-1 6/6 -> 3/6. swish: AudioSet names 0 -> 0.8, typ z -3.1 -> 0.6. swipe: stereo corr 0.90 -> 0.64 (bench 0.68). toggle tone 10.4 -> 7.8 dB; tap tone 8.2 -> 5.8 dB, t40 x 1.82 -> 1.00. impact tone 5.3 -> 4.4 dB, variety 0.905 -> 0.849, onset tick 6.5 dB darker. pro_logo_sting tone 24.4 -> 15.9 dB; logo_sting 16.4 -> 13.1 dB (env oct worse); riser 11.3 -> 9.3 dB (env / curve worse). pro_impact's model checks fell; it was kept on the spectrogram. Breadth: tone at least as close as real forge refs in 18 of 21 categories (r3 17); 0 artifact flags in 183. Film: critique 16 -> 19 of 22 in range, realness 0.73 -> 0.85, cues within 1 frame 86 -> 95 %, -14.0 LUFS / TP -2.3. |
| 5 | 14:05-16:30 | **forge v2 polish 2 + clean blind test + finalize.** polish-ui re-delivered whoosh_fast (the hiss band keeps sinking through the tail: new optional `tail_sweep` / `tail_tau` / `hiss_q` / `tail_gust` keys; per-member door fixes), tap (round-3 picks with per-member overrides) and logo_sting (smooth shimmer wash `shimmer_smooth`, round-3 leads, reverse_bloom `swell_trim`). polish-pro re-delivered pro_logo_sting (new sting mode 'gesture': five launch gestures from sines, filtered noise and a reverb fed above 250 Hz), pro_pop (natural chirp `c_nat`: no drive, no stepped gates) and pro_notify (the film ding made clean: no drive, no noise breath, a semitone ladder). Core fixed the round-4 fill bug (`search.fill_start`: start, recipe DEFAULTS, render-module DEFAULTS; mid-range only with a `MidRangeFill` warning), pinned every delivered member in the seed banks (`save_seeds(delivered=)`, `pin_delivered()`; 183 pins) and audited all 34 finals (hidden mid-range values only in pro_impact, kept). The blind test was rebuilt with symmetric trims and a cleanliness report that stops the build on a fail (`blind_v2.py --dry-run / --allow-fail`). Film: computed presence harmonics for phones (`v2lib.presence()`), a gentler logo limiter, the riser apex on its cue, a round-1 bug that skipped the sub drop / sting gates. The finalizer ran one full bench, kept r5 in 4 categories and put r4 back for tap and pro_pop (r5 in `final_r5/`), re-pinned the seed banks and rebuilt the blind test, progress file, galleries and film. | Tests: 433 passed, 15 skipped (fast suite, final tree, two halves 7.4 + 1.3 min). Held-out bench `eval/forge/results_v2r5.md`, r4 -> r5: pro_logo_sting variety 0.908 -> 0.642, tone 15.9 -> 15.6 dB, env oct 2.50 -> 2.15, margin -1.30 -> -0.68; whoosh_fast darken -0.05 -> -0.24 (bench -0.2), AudioSet names 0.5 -> 1.0 (variety 0.768 -> 0.831, inside the real rows); logo_sting decay roughness 0.46 -> 0.34 dB, pre-hit dip 14.6 -> 2.0 dB (reverse_bloom); pro_notify without the round-4 hiss, variety 0.822 -> 0.936. Not kept: tap (variety, curve, width worse) and pro_pop (variety 0.649 -> 0.780, phrase z and margin down; four members became steady beeps). Breadth unchanged: tone at least as close as real forge refs in 18 of 21, inside counts 11 of 22, 0 artifact flags in 183. Blind test: cleanliness PASS, 20 trials, 0.00 LU apart at the loudest moment, true peaks <= -1 dBTP, 118 s. Film: cues within 1 frame 95 -> 98 %, logo limiting 7.1 -> 3.3 dB, logo rise on a phone curve +1.1 -> +8.8 dB, critique 19 of 22, -14.0 LUFS / TP -2.2. |
| 6 | 18:00-19:40 | **forge v2 round 6 (after the showcase).** Four areas. pop-notify: pro_pop rebuilt as natural pops (`n6` in pop.py: a gliding sine body, soft partials, exponential decay, optional puff / thump / pre-blip / room / double pop); pro_notify as five struck materials (`chime` in notify.py: film, glass, bowl, tine, wood, FM bell), not delivered by the area. tap-riser: a new riser render (`r6=1`: a rising noise band without a resonant peak, a rising root-fifth-octave chord, an accelerating gate, a reverse swell from a computed tail) for 3 of 5 members; tap = the better of round 4 / 5 per family. impact-glitch: pro_impact gets a designed boom, a delayed rumble and denser grains; glitch a new chopped-repeat / bit-crush / dropout path (`roll0`). code: one report schema (`earshot/forge/schema.py`: a `render` block per pick, `check_final`, `upgrade`, `earshot forge --check`), `forge.variant_recipe()`, pro_typing_run and pro_shimmer re-render again (`r2_path`). Every new key defaults to the old render. The finalizer ran one full bench, kept r6 for pro_pop, riser, tap and glitch, put r4 back for pro_impact (r6 in `final_r6/`), applied the schema to all 34 finals, re-pinned the seed banks and rebuilt the gallery and a round-5 vs round-6 page. | Tests: 485 passed, 15 skipped (fast suite, final tree, 11.4 min; the untracked fit tests left out). Held-out bench `eval/forge/results_v2r6.md`, r5 -> r6: pro_pop variety 0.649 -> 0.617, tone 22.93 -> 19.51 dB (real x 18.43), CLAP '8-bit' top-3 3/6 -> 0/6 (phrase z 3.29 -> 2.68, margin 1.76 -> 1.27); riser margin -0.30 -> +0.08 (jet-engine reading), variety 0.801 -> 0.772, env oct 2.35 -> 2.05 (tone 9.33 -> 12.75 dB, inside the real rows); tap variety 0.782 -> 0.761, env oct 1.40 -> 1.22, inside 4 -> 5 of 7 (margin 0.77 -> 0.55); glitch phrase z 2.03 -> 2.62, margin -0.13 -> 0.88, AudioSet 0 -> 0.17. Not kept: pro_impact (variety 0.788 -> 0.820, width 0.35 -> 0.67, though AudioSet 0.6 -> 1.0), pro_notify (phrase z 2.16 -> 1.45). Reproducibility: 183 of 183 picks re-render from report.json within 1.19e-7 (round 5: 32 of 34 categories). |

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

### Round 3 notes (judge + integrator, 11:00-11:30)
Full table and per-category verdicts: `eval/forge/results_v2r3.md` (Judge notes). Made medians on the held-out bench,
r2 -> r3; the judge's own model-free measurements and r2 / r3 / forge-ref spectrogram sheets decided (sheets in
`out/forge_v2/_judge/r3/`, not committed). Nobody has listened yet.

| category | verdict | key numbers (r2 -> r3, reference) |
|---|---|---|
| swipe | fixed | 4-12 kHz -35.9 -> -20.7 dB (bench -19.2); tone 13.47 -> 6.06 dB; onset still 11 dB dull |
| whoosh_fast | fixed (model-free) | t40 130 -> 238 ms (bench 265); <150 Hz -9.7 -> -14.4; 4-12 kHz -17.4 -> -26.4; tone 9.43 -> 5.79; AudioSet still 'Door' |
| swish | fixed (model-free) | t40 150 -> 230 ms (bench 220); <150 Hz -15.8 -> -23.4; 4-12 kHz -18.2 -> -29.1; AudioSet 'Door' / 'Whack' |
| logo_sting | fixed | t40 2135 -> 885 ms (bench 862); inside 3.5 -> 6 of 6; no body under ~800 Hz |
| whoosh_big | kept best | inside 7 of 7; variety 0.909 -> 0.854; curve 5.8 -> 9.0 dB |
| pro_whoosh_big | better | variety 0.931 -> 0.882; air fused (1-4 kHz -46.5 -> -34.8, film -38.0) |
| impact | better, overshoot | onset >6 kHz -28.9 -> -10.9 dB (bench -21.6: a bright contact tick now); tone 6.29 -> 5.31; inside 6 of 6 |
| pro_impact | mixed | 4-12 kHz -30.4 -> -17.7 (film -15.4); margin 0.60 -> -0.07 |
| pro_logo_sting | still weak | clones 0.943; nothing above ~5 kHz (4-12 kHz -41.8 vs film -21.0) |
| click | fixed | 1-4 kHz -14.0 -> -5.3 dB (bench -4.9); inside 5 -> 7 of 7; variety 0.681 -> 0.723 |
| toggle | mixed | env oct 2.58 -> 0.62; tone 8.82 -> 10.36; ringing lines in 01 / 03 |
| tap | better, bright | phrase z 0.52 -> 1.13; inside 3 -> 5 of 7; 4-12 kHz 8 dB over the bench; t40 x 1.82 |
| pro_tap | flat | phrase z 1.73 -> 1.65; variety 0.854 -> 0.872 |
| pro_click | fixed | tone 25.8 -> 15.4 dB; 150 Hz-1 kHz -41.2 -> -6.8 (bench -6.2) |
| pro_hover_tick | fixed | phrase z 0.85 -> 1.61 (loo 1.79); inside 4 -> 6 of 7 |
| pro_pop_in | fixed | variety 0.895 -> 0.750 |
| pop | **r2 kept** | r3: tone 11.42 -> 12.38, t40 x 0.82 -> 0.55, phrase z 2.32 -> 1.83 |
| pro_pop | flat | inside 4 -> 3 of 6 (real 3); CLAP still '8-bit' |
| notify | better, model checks fell | t40 x 0.63 -> 0.85; curve 21.8 -> 13.6; phrase z 2.21 -> 1.40; AudioSet 'Clang' |
| pro_notify | better, still far | >12 kHz -29.6 -> -19.6 (films -17.3); variety 0.882 -> 0.822 |
| shimmer | fixed | t40 x 0.48 -> 0.97; curve 14.1 -> 4.1 dB; inside 7 of 7 |
| riser | mixed | tone 12.5 -> 11.3; AudioSet names 0.8 -> 0.4 |
| pro_riser | better | tone 20.8 -> 17.9; phrase z 2.00 -> 2.71; inside 3 -> 4 of 6 |
| pro_zip | **r2 kept** | r3: tone 11.89 -> 12.72, env 0.71 -> 0.97, t40 x 0.84 -> 0.63 |
| typing_key | fixed (model-free) | stereo corr 0.34 -> 0.96; <150 Hz -5.5 -> -10.2 (bench -10.9); inside 6 -> 4 of 7 (model checks) |
| pro_typing_key / run | fixed | variety 0.926 -> 0.815 / 0.930 -> 0.813 |
| glitch | mixed | variety 0.799 -> 0.728; AudioSet names 0.5 -> 0 |
| pro_glitch | fixed | median 2 ms level -22 -> -36 dB re peak (film -33); accelerating stutter 6 of 6 |
| downlifter, hover_tick, sub_drop, pro_sub_drop, pro_shimmer | unchanged | (round 2 finals) |

Showcase-ready (measured): whoosh_big, pro_whoosh_big, shimmer, click, typing_key, pro_typing_key / run, pro_pop_in,
pro_hover_tick, pro_click, logo_sting, swipe, impact, whoosh_fast, swish, pop (r2), pro_zip (r2), downlifter, sub_drop,
pro_glitch, pro_riser. Weakest: pro_logo_sting, pro_tap, pro_pop, toggle, tap, notify, riser, glitch, pro_notify.

Integration this round (judge): bench run on all finals (`eval/forge/results_v2r3.json` / `.md`); pop / pro_zip
`final/` = the round-2 delivery again (`KEPT_R2.txt`; round-3 attempt in `final_r3/`, `NOT_DELIVERED.txt`; pro_zip
re-renders from today's zip.py at max error 1.4e-7); gallery pages `out/showcase/gallery.html` and
`out/showcase/compare.html` (r2 / r3 / r3x rows; private: they play licensed forge refs; checked in headless Chrome at
1440 px). Process notes: the ui area ran bench.py 4 times (numbers seen between fix rounds); the core critic changed
under the fixers mid-round, so critic totals are not comparable across areas or rounds (55 / 183 pass vs r2 60 / 168).

Round 4 (next): impact contact sheen -8..10 dB; logo_sting body (a low strike or sub under the note); pro_logo_sting
top end + distinct members; whoosh_fast / swish AudioSet 'door' reading (onset shape, not balance) and variety; made
tails darken faster than real ones (whoosh_fast -0.64 vs -0.2, swipe -1.96 vs +0.61); swipe onset brightness (>6 kHz
-22 vs -12 dB) and width (corr 0.90 vs 0.70); toggle ring lines; tap 4-12 kHz; notify 'Clang'; core: fx.saturate
(level='peak') normalise before the curve, fx.limit release between hits, polish.tail false 'bumpy tail' on
louder-first toggles, the deferred real-mode ref_typ_z >= -2 pass rule. Human listening (blind, loudness-matched) is
still the deciding test.

### Round 4 notes (finalizer, 13:35-14:00)
- **Benchmark.** One full run on the final tree at 13:35 (`eval/forge/results_v2r4.json` / `.md`); nothing was tuned
  on it. Each polish area had run it once to confirm, at 12:30, 12:42 and 12:49.
- **Round 3 vs round 4.** The rule is the progress file's (`blind_v2.verdict`): a check counts only when it moves
  relative to the real rows, the floor is left out, and the areas' spectrogram sheets break ties. **No category
  went back to round 3.**
- **Close calls** (reasons in `results_v2r4.md` and each category's LOG.md, under "Finalizer"):
  - pro_impact: no bench refs; the model checks prefer r3, whose onset tick was a bug.
  - riser: env oct / curve are worse against real cymbal swells.
  - logo_sting: env oct is worse because the bench refs are bare rings.
- **Reproducibility.** All 75 members of the 14 re-delivered categories re-render from their `report.json` on the
  final code to max 3e-7. The film's `v2lib.MODULE` maps pro_tap to a missing 'tap' module; the film does not use
  pro_tap, and 'popin' is right.
- **Rebuilt from the final finals:**
  - the blind test (`out/showcase/blind_v2/`: 20 trials, 115 s, real = A on 10; toggle moved from weak to middle
    in the tiers);
  - the progress file (`out/showcase/progress/`, now against v2r4);
  - `gallery_embedded.html` (9.3 MB), `gallery.html` (with forge refs, private), `compare.html` (r3 vs r4 for the 14
    categories) and `v1_vs_v2_embedded.html`;
  - the film (`build.py --listen --mux`; nothing it uses changed after 12:49; checked with `earshot timeline` and
    `earshot measure` on the mp4).
- **Showcase:** `out/showcase/SHOWCASE.md`.
- **Fixed on the way:**
  - `bench.py` scrubs machine paths before writing;
  - `blind_v2.py` usage lines no longer name a machine path;
  - `FORGE_V2.md` and the README carry the round-4 numbers.
- **Open:**
  - forge's final pass fills missing SPACE keys at mid-range (`search._encode` gives 0.5), not from DEFAULTS;
  - the logo_sting `swell_trim` fix for a -20 dB pre-hit dip is not delivered;
  - whoosh_fast's weaker hiss tail is the next step;
  - the seed banks also hold not-delivered attempts (ui-tonal) and only 1 of 5 logo_sting members.
- **Human listening (the blind test) is still the deciding test.** Nobody has listened.

### Round 5 notes (finalizer, 15:56-16:30)
- **Benchmark.** One full run at 15:58 on the delivered tree (`eval/forge/results_v2r5.json` / `.md`); nothing was
  tuned on it. polish-ui had run it once (14:56), polish-pro twice (14:47 on candidates, 15:05 on the final tree).
  The core area spent its one run on a typing_key re-render that it withdrew at 14:44; typing_key's final is the
  original (it measures exactly as in round 4).
- **Round 4 vs round 5.** Same rule as round 4 (`blind_v2.verdict`, floor left out, spectrogram sheets break ties).
  - Kept r5: whoosh_fast, logo_sting, pro_logo_sting (clear win), pro_notify (on the spectrogram: round 4 has
    broadband hiss through every ring; round 5's variety is worse, 0.936).
  - **Put r4 back: tap and pro_pop.** `final/` is the round-4 delivery again (`KEPT_R4.txt`), the round-5 delivery is
    in `final_r5/`. tap: model-free better 1 / worse 3 (its model checks went up). pro_pop: variety 0.649 -> 0.780,
    phrase z and margin down, and four of six members became near-steady beeps; the '8-bit' fault of round-4
    members 01 / 02 stays open. The results file carries the round-4 rows for both (`kept_r4`); the as-run rows are
    in the Judge notes.
- **Reproducibility.** The restored round-4 tap and pro_pop finals re-render on the final code to max 1.6e-7; the
  round-5 deliveries to 2.5e-7 (the areas' checks). 32 of 34 categories reproduce from `report.json`.
- **Rebuilt from the final finals:**
  - seed-bank pins (`out/forge_v2/_core_r5/pin_finals.py`);
  - the blind test (`out/showcase/blind_v2/`: cleanliness PASS, 20 trials, 0 fails, 12 pairs with warnings, 118.1 s,
    real = A on 10);
  - the progress file (`out/showcase/progress/`, against v2r5; no stale rows);
  - `gallery_embedded.html` (9.3 MB, checked in headless Chrome at 1440 px and at 504 px, Chrome's minimum),
    `gallery.html`, `compare.html` (r4 vs r5 for the 6 round-5 categories) and `v1_vs_v2_embedded.html`;
  - the film (`build.py --listen --mux`: sample-identical to the 15:46 build; `earshot timeline` on the mp4: 98 % of
    43 cues within 1 frame; `earshot measure`: -14.0 LUFS, TP -2.2 dBTP).
- **Showcase:** `out/showcase/SHOWCASE.md`.
- **Open:**
  - pro_pop's '8-bit' members (round 5's natural chirp was the right direction for 01 / 04 only);
  - tap's model checks;
  - pro_notify variety (a one-patch ladder);
  - the report schema (module vs category in `recipe`, `family_params`) is still not uniform;
  - stand-in recipes with mode-dependent defaults (sting) still fill from `sting.DEFAULTS`, not the mode presets.
- **Human listening (the blind test) is still the deciding test.** Nobody has listened.

### Round 6 notes (finalizer, 19:02-19:40; after the user got the showcase)
- **Benchmark.** One full run at 19:03 on the delivered tree (`eval/forge/results_v2r6.json` / `.md`, 112 s); nothing
  was tuned on it. Each area had run it once (pop-notify on candidates with `--no-real`). The 29 categories round 6
  did not touch give exactly the round-5 numbers.
- **Round 5 vs round 6.** Same rule as rounds 4-5 (`blind_v2.verdict`, floor left out, model-free first, spectrogram
  sheets break ties).
  - Kept r6: pro_pop (model-free better 2 / worse 0: variety, tone; the '8-bit' stepped gates are gone; phrase z and
    margin fell), riser (2 / 0: variety, env oct; margin up, the jet-engine reading down), tap (2 / 0: variety, env
    oct; two of three model checks slightly down), glitch (no refs; all three model checks up).
  - **Put r4 back: pro_impact.** Model-free 0 / 1 (variety 0.788 -> 0.820 against the film refs' 0.612) and a
    narrower image (corr 0.35 -> 0.67), while AudioSet names it on 5 of 5 (was 3 of 5) and phrase z rose. A close
    call, decided by the trust order; `final/` is round 4 (`KEPT_R4.txt`), round 6 is in `final_r6/`.
  - pro_notify: not delivered by its area (both model checks fell; the five materials read acoustic, not digital).
    Candidate in `final_r6/`.
- **Reproducibility.** The code area's schema (`earshot/forge/schema.py`) was applied to all 34 finals: every
  `report.json` has `schema: 2` and a `render` block per pick, and all 183 picks re-render within 1.19e-7
  (`tests/test_forge_rerender.py` checks each final). Pre-upgrade reports: `out/forge_v2/_final_r6/reports_pre_schema/`.
- **Rebuilt:** seed-bank pins for all 34 finals; `out/showcase/gallery_embedded.html` (9.3 MB, 0 warnings);
  `out/showcase/compare_r6.html` (new, 2.4 MB, self-contained, no refs: round 5 vs round 6 for the 6 categories,
  including the two round-6 candidates that were not kept; checked in headless Chrome at 1440 px and 504 px).
- **Not rebuilt on purpose:** the blind test (`out/showcase/blind_v2/`, the user is taking it), the progress file and
  the film (the film keeps its round-5 sounds).
- **Tests:** 485 passed, 15 skipped (fast suite on the final tree, 681 s, one run).
- **Not in this commit:** `earshot/forge/fit/` and `tests/test_fit_*.py` are another session's work in progress
  (untracked, still being written at 19:16); the fast-suite run left them out.
- **Open:**
  - pro_impact: the boom with the sub / body ducked under it and a decorrelated (wide) boom tail, to keep r6's
    recognition without losing the width and variety;
  - pro_pop 05 (tink) is a near-steady 2 kHz line; phrase z / margin favour the old synthetic idiom;
  - riser: the kept shepard / rush members still sit nearest 'a jet engine'; the swell member stops at its peak
    (the gallery flags a tail cut at -2 dB);
  - tap: AudioSet names it on 1 of 5; pro_tap not attempted;
  - pro_notify: digital, not acoustic, variety (the next try: one digital patch with per-member interval, brightness
    and envelope moves rather than new materials);
  - glitch: tonal members still read '8-bit' at times; centroid 1.4-2.9 kHz vs the film's 9 kHz;
  - the finals sweep in `test_forge_rerender.py` adds ~90 s to the fast suite (not marked slow).
- **Human listening (the blind test) is still the deciding test.** Nobody has listened.
