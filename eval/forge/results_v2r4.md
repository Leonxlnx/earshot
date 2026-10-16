# forge benchmark: v2r4

made: out/forge_v2; refs: <EARSHOT_HOME>/refsets.json; 2026-10-16 13:35

Medians per set. made = the made sounds vs the bench refs; real x = the forge refs (real, never in the bench) vs the bench refs; real loo = each bench ref vs the other bench refs. typ z, phrase z, margin, AS names: higher is better; tone dB, env oct (octaves off in attack / t40 / length), curve dB: lower is better; t40 x = tail length / bench median (1 = same). inside = checks (of n) a made sound lands inside the real 80% limit, median over the made sounds (real = the same count for the real rows). var cos = mean pairwise CLAP cosine within the set (made variants / forge refs / bench refs): high = the variants are near clones.

| category | set | n | typ z | tone dB | env oct | curve dB | t40 x | phrase z | margin | AS names | artifacts | var cos | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| click | made | 5 | 0.21 | 8.79 | 0.23 | 8.83 | 1.44 | 3.03 | 0.24 | 0 | 0 | 0.723 | 7 / 7 (real 6) |
|  | real x | 8 | 0.16 | 11.33 | 0.22 | 9.92 | 1.33 | 2.77 | 0.53 | 0.12 | 0 | 0.651 |  |
|  | real loo | 7 | -0.1 | 9.85 | 0.46 | 13.19 | 0.86 | 2.67 | 0.27 | 0 | 0 | 0.576 |  |
| downlifter | made | 6 |  |  |  |  |  | 3.23 | 0.92 | 0.33 | 0 | 0.749 |  |
| | note: empty: the licensed library has no downlifter, real or designed (no reversing or pitching of other sounds here: trim + fade only). Benchmark made downlifters on phrasings, AudioSet and artifacts only. | |  |  |  |  |  |  |  |  |  |  |  |
| glitch | made | 6 |  |  |  |  |  | 2.03 | -0.13 | 0 | 0 | 0.728 |  |
| | note: empty: no glitch material, real or designed, in the licensed library. Benchmark made glitches on phrasings, AudioSet and artifacts only. | |  |  |  |  |  |  |  |  |  |  |  |
| hover_tick | made | 5 | 0.13 | 20.99 | 1.51 | 6.27 | 1.1 | 2.41 | 0.66 | 0 | 0 | 0.807 | 6 / 7 (real 5) |
|  | real x | 6 | 0.09 | 21.2 | 1.64 | 9.88 | 0.9 | 2.51 | 0.9 | 0 | 0 | 0.799 |  |
|  | real loo | 6 | 0.47 | 15.46 | 1.16 | 19.39 | 1.2 | 2 | 0.58 | 0 | 4 | 0.626 |  |
| impact | made | 6 |  | 4.38 | 2.39 | 30.05 | 0.58 | 3.25 | 0.9 | 1 | 0 | 0.849 | 6 / 6 (real 3) |
|  | real x | 2 |  | 6.19 | 2.5 | 31.98 | 0.46 | 3.04 | 1.09 | 0.5 | 0 | 0.659 |  |
| | note: thin: the library has three cinematic hits (all three were the v1 impact forge refs); one is held out as bench, so v1 impact bench numbers are optimistic. | |  |  |  |  |  |  |  |  |  |  |  |
| logo_sting | made | 5 |  | 13.06 | 1.68 | 14.29 | 1 | 1.25 | -0.48 | 1 | 0 | 0.714 | 5 / 6 (real 5.5) |
|  | real x | 2 |  | 15.52 | 1.46 | 21.27 | 1.7 | 1.4 | -0.82 | 0.5 | 0 | 0.185 |  |
|  | real loo | 2 |  | 23.22 | 0.81 | 8.99 | 1.12 | 1.36 | -0.58 | 1 | 0 | 0.507 |  |
| | note: no real logo stings exist in the library (they are composed); the refs are the struck metal / glass rings a sting ends on: a tail-shape and tone reference only. | |  |  |  |  |  |  |  |  |  |  |  |
| notify | made | 6 | 0.15 | 15.63 | 0.66 | 13.09 | 0.86 | 1.54 | -0.5 | 1 | 0 | 0.867 | 7 / 7 (real 5.5) |
|  | real x | 5 | -0.24 | 21.3 | 1.3 | 28.92 | 0.34 | 1.25 | -0.24 | 1 | 0 | 0.593 |  |
|  | real loo | 5 | 3.84 | 16.7 | 0.81 | 17.41 | 1.19 | 1.76 | -0.91 | 1 | 0 | 0.476 |  |
| | note: struck bells / glass / kalimba (forge) and celesta / glass / triangle (bench), rings capped at 2 s with a 0.25 s fade; no designed UI dings in the library. | |  |  |  |  |  |  |  |  |  |  |  |
| pop | made | 5 | -0.02 | 11.42 | 0.55 | 11.01 | 0.82 | 2.32 | -0.2 | 0.4 | 0 | 0.735 | 7 / 7 (real 6) |
|  | real x | 7 | -0.09 | 16.06 | 1.51 | 20.54 | 0.91 | 2.46 | -0.5 | 0.29 | 0 | 0.501 |  |
|  | real loo | 6 | 1.13 | 13.66 | 0.41 | 12.5 | 1.02 | 1.96 | -1.06 | 0.67 | 0 | 0.522 |  |
| pro_click | made | 5 |  | 15.43 | 1.28 | 13.47 | 0.8 | 3.91 | 2.28 | 0 | 0 | 0.786 | 5 / 6 (real 4) |
|  | real x | 1 |  | 29.35 | 1.06 | 10.68 | 0.53 | 4.25 | 2.96 | 0 | 0 |  |  |
|  | real loo | 2 |  | 18.53 | 1.73 | 8.99 | 1.08 | 2.19 | 0.51 | 0 | 2 | 0.45 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_glitch | made | 6 |  |  |  |  |  | 2.53 | 0.24 | 0 | 0 | 0.853 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only, one example. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_hover_tick | made | 5 | -1.32 | 6.25 | 0.82 | 10.43 | 1 | 1.61 | 0.45 | 0 | 0 | 0.862 | 6 / 7 (real 6) |
|  | real x | 3 | -2.23 | 6.51 | 1.11 | 20.12 | 1 | 3.04 | 0.97 | 0 | 0 | 0.82 |  |
|  | real loo | 3 |  | 8.32 | 0.67 | 5.84 | 0.86 | 1.79 | 0.71 | 0 | 1 | 0.684 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_impact | made | 5 |  |  |  |  |  | 2.17 | -0.09 | 0.6 | 0 | 0.788 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only (one SFX-only film); designed, sub-heavy, gated tails. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_logo_sting | made | 5 |  | 15.93 | 2.5 | 23.34 | 0.12 | 0.95 | -1.3 | 0 | 0 | 0.908 | 4 / 6 (real 6) |
|  | real x | 1 |  | 15.49 | 2.84 | 24.51 | 0.06 | 0.99 | -0.25 | 0 | 0 |  |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge: falling chirp into a tonal hum; bench: a music end chord with a 4 s tail. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_notify | made | 5 |  |  |  |  |  | 1.91 | 0.15 | 1 | 0 | 0.822 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_pop | made | 6 |  | 22.93 | 1.41 | 35.59 | 0.75 | 3.29 | 1.76 | 0 | 0 | 0.649 | 3 / 6 (real 3) |
|  | real x | 2 |  | 18.43 | 2.78 | 42.94 | 0.06 | 3.54 | 2.33 | 0 | 0 | 0.488 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_pop_in | made | 5 |  |  |  |  |  | 0.93 | -3.13 | 0 | 0 | 0.75 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). not a forge category yet: element-in micro hits (thump + tick), forge only. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_riser | made | 5 |  | 17.88 | 1.54 | 36.57 | 0.05 | 2.71 | 0.47 | 0.2 | 0 | 0.712 | 4 / 6 (real 6) |
|  | real x | 1 |  | 21.35 | 0.79 | 18.43 | 0.68 | 1.03 | -0.72 | 0 | 0 |  |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge: one designed stutter riser; bench: one noise riser at a film end. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_shimmer | made | 5 |  |  |  |  |  | 3.29 | 1 | 0.2 | 0 | 0.646 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only; one granular sparkle + one melodic chime run. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_sub_drop | made | 5 |  | 24.84 | 0.54 | 16.11 | 1.64 | 2.44 | 0.68 | 0 | 0 | 0.835 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). bench only, one example under music + voice. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_tap | made | 5 |  |  |  |  |  | 1.98 | 1.93 | 0.2 | 0 | 0.861 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only, one example. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_typing_key | made | 6 |  | 26.09 | 0.43 | 13.62 | 0.79 | 1.6 | -0.96 | 0 | 0 | 0.815 | 4 / 6 (real 5.5) |
|  | real x | 6 |  | 26.2 | 0.52 | 11.54 | 0.71 | 1.88 | -0.42 | 0 | 0 | 0.849 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_typing_run | made | 6 |  |  |  |  |  | 1.79 | 0.59 | 0 | 0 | 0.813 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). not a forge category yet: one whole designed typing run. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_whoosh_big | made | 6 |  |  |  |  |  | 3.71 | 1.56 | 0.83 | 0 | 0.882 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only (one SFX-only film); the bench films bury their whooshes under music + voice. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_zip | made | 5 | -2.41 | 11.89 | 0.71 | 20.81 | 0.84 | 2.34 | -1.39 | 0 | 0 | 0.761 | 5 / 7 (real 5) |
|  | real loo | 4 | 1.71 | 10.4 | 1 | 34.47 | 1.15 | 1.4 | 0.54 | 0 | 0 | 0.529 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). not a forge category yet: tonal pitch sweeps, bench only (two films over music). | |  |  |  |  |  |  |  |  |  |  |  |
| riser | made | 5 |  | 9.33 | 2.35 | 23.23 | 0.09 | 2.65 | -0.3 | 0.6 | 0 | 0.801 | 3 / 6 (real 4) |
|  | real x | 1 |  | 17.61 | 1.01 | 20.04 | 0.5 | 3.52 | 1.06 | 1 | 0 |  |  |
|  | real loo | 2 |  | 13.22 | 0.95 | 36.07 | 3.35 | 2.7 | 1.15 | 0.5 | 0 | 0.767 |  |
| | note: thin: one designed trailer riser (forge); the bench is two real cymbal swells, no synth risers. | |  |  |  |  |  |  |  |  |  |  |  |
| shimmer | made | 6 | 1.07 | 10.68 | 0.35 | 4.06 | 0.97 | 3.46 | 0.91 | 0.67 | 0 | 0.845 | 7 / 7 (real 6) |
|  | real x | 5 | 1.69 | 9.66 | 0.73 | 8.58 | 0.73 | 2.87 | 0.03 | 0.6 | 2 | 0.59 |  |
|  | real loo | 4 | 1.01 | 15.52 | 1.33 | 12.89 | 1.05 | 3.4 | 0.56 | 1 | 0 | 0.644 |  |
| | note: kit shimmer_01 (bar chimes sweep) is bench: its author also recorded the celesta notify bench. | |  |  |  |  |  |  |  |  |  |  |  |
| sub_drop | made | 5 |  | 8.84 | 1.76 | 37.38 | 0.31 | 3.31 | 0.7 | 0 | 0 | 0.727 |  |
| | note: one designed source (a tom pitched down 24 semitones by its author), held out as bench; the forge has no sub-drop refs. | |  |  |  |  |  |  |  |  |  |  |  |
| swipe | made | 5 | -9.49 | 5.15 | 0.38 | 7.67 | 0.92 | 1.88 | 0.11 | 0.6 | 0 | 0.812 | 4 / 7 (real 6) |
|  | real x | 4 | -10.07 | 16.83 | 0.43 | 14.12 | 0.79 | 2.7 | 0.59 | 1 | 0 | 0.845 |  |
|  | real loo | 4 | 1.25 | 3.64 | 0.22 | 10.07 | 1.02 | 3.42 | 2.08 | 0.25 | 0 | 0.857 |  |
| | note: cloth only: the library has no paper slides. | |  |  |  |  |  |  |  |  |  |  |  |
| swish | made | 5 | 0.62 | 6.69 | 0.23 | 11.69 | 1.02 | 2.94 | 1.62 | 0.8 | 0 | 0.796 | 5 / 7 (real 6) |
|  | real x | 7 | 1.94 | 11.08 | 0.64 | 12.88 | 0.64 | 3.22 | 1.6 | 0.57 | 0 | 0.811 |  |
|  | real loo | 3 |  | 9.01 | 0.91 | 16.45 | 1.16 | 3.5 | 1.92 | 0.67 | 0 | 0.79 |  |
| tap | made | 5 | -2.31 | 5.81 | 1.4 | 11.65 | 1 | 1.12 | 0.77 | 0.2 | 0 | 0.782 | 4 / 7 (real 6) |
|  | real x | 7 | -1.17 | 9.35 | 1.59 | 13.67 | 1.64 | 1.85 | 0.92 | 0.14 | 0 | 0.713 |  |
|  | real loo | 6 | 1.66 | 16.78 | 1.19 | 11.59 | 1 | 1.62 | 0.35 | 0.17 | 0 | 0.716 |  |
| toggle | made | 5 | -3.4 | 7.82 | 0.42 | 10.11 | 0.74 | 1.72 | -0.38 | 0 | 0 | 0.729 | 5 / 7 (real 6) |
|  | real x | 4 | -2.84 | 12.48 | 1.87 | 18.78 | 2.11 | 2.1 | -0.02 | 0 | 0 | 0.477 |  |
|  | real loo | 4 | 1.54 | 6.25 | 0.28 | 4.92 | 1.02 | 1.94 | -0.28 | 0 | 0 | 0.725 |  |
| typing_key | made | 6 | -0.32 | 7.44 | 1.04 | 9.08 | 1 | 3.74 | 1.45 | 0 | 0 | 0.713 | 4 / 7 (real 6) |
|  | real x | 15 | 0.26 | 8.88 | 1.01 | 12.2 | 1.29 | 4.75 | 2.83 | 0.53 | 0 | 0.721 |  |
|  | real loo | 12 | 0.23 | 5.47 | 1.52 | 16.7 | 1 | 3.71 | 1.53 | 0.42 | 0 | 0.673 |  |
| whoosh_big | made | 6 | 0.51 | 6.87 | 0.4 | 10.21 | 1.05 | 2.44 | -0.02 | 0.83 | 0 | 0.76 | 7 / 7 (real 5) |
|  | real x | 3 | -0.27 | 8.8 | 1.13 | 19.46 | 3 | 2.92 | 0.04 | 1 | 0 | 0.7 |  |
|  | real loo | 3 |  | 4.25 | 0.43 | 6.67 | 0.97 | 1.88 | 0.19 | 0.33 | 0 | 0.495 |  |
| | note: forge = three designed Mixkit whooshes; bench = real air / cape swoops. | |  |  |  |  |  |  |  |  |  |  |  |
| whoosh_fast | made | 6 | -4.25 | 6.28 | 0.32 | 10.32 | 0.85 | 3.23 | 1.66 | 0.5 | 0 | 0.768 | 4 / 7 (real 6.5) |
|  | real x | 5 | 0.57 | 10.31 | 1.22 | 20.2 | 0.6 | 4.14 | 2.78 | 0.4 | 0 | 0.826 |  |
|  | real loo | 5 | 1.52 | 4.58 | 0.4 | 8.21 | 1.08 | 4.14 | 2.62 | 0.8 | 0 | 0.772 |  |

Texture (medians; model-free, not in any forge objective): floor dB = 10th percentile 5 ms frame re the loudest (-120 = digital silence), stereo corr (1 = mono), darken = log2 centroid at half the tail / at the peak (< 0: the decay darkens).

| category | set | floor dB | stereo corr | darken |
|---|---|---|---|---|
| click | made | -45.0 | 0.99 | 0.26 |
|  | real x | -42.85 | 0.99 | -0.12 |
|  | real loo | -44.8 | 1.0 | -0.07 |
| downlifter | made | -54.45 | 0.67 | -1.81 |
| glitch | made | -44.95 | 0.09 | 0.04 |
| hover_tick | made | -44.0 | 0.88 | -0.09 |
|  | real x | -43.55 | 1.0 | -0.09 |
|  | real loo | -24.9 | 0.52 | -0.01 |
| impact | made | -68.95 | 0.68 | -0.96 |
|  | real x | -70.55 | 0.54 | -1.3 |
| logo_sting | made | -61.9 | 0.8 | 0.41 |
|  | real x | -52.1 | 0.59 | -0.85 |
|  | real loo | -44.1 | 0.81 | -1.43 |
| notify | made | -43.7 | 0.98 | -1.53 |
|  | real x | -41.2 | 0.99 | -1.06 |
|  | real loo | -24.4 | 0.61 | -0.53 |
| pop | made | -62.8 | 1.0 | 0.25 |
|  | real x | -53.2 | 1.0 | 0.21 |
|  | real loo | -47.2 | 1.0 | 0.12 |
| pro_click | made | -83.1 | 1.0 |  |
|  | real x | -108.5 | 0.0 |  |
|  | real loo | -30.6 | 0.95 | 0.26 |
| pro_glitch | made | -44.85 | 0.04 | 0.04 |
| pro_hover_tick | made | -37.8 | 0.26 |  |
|  | real x | -47.0 | -0.0 |  |
|  | real loo | -37.4 | 0.16 |  |
| pro_impact | made | -34.7 | 0.35 | 0.41 |
| pro_logo_sting | made | -60.2 | 1.0 | -0.65 |
|  | real x | -120.0 | 1.0 | -0.38 |
| pro_notify | made | -47.7 | 0.94 | 0.39 |
| pro_pop | made | -69.5 | 1.0 | -0.31 |
|  | real x | -120.0 | 1.0 |  |
| pro_pop_in | made | -61.5 | 0.58 | -0.24 |
| pro_riser | made | -32.5 | 0.82 |  |
|  | real x | -41.8 | 0.96 | 0.22 |
| pro_shimmer | made | -49.0 | 0.33 | -0.53 |
| pro_sub_drop | made | -38.3 | 1.0 | 0.13 |
| pro_tap | made | -83.3 | 0.42 | 0.07 |
| pro_typing_key | made | -38.95 | 1.0 | 0.02 |
|  | real x | -53.55 | 1.0 | -0.04 |
| pro_typing_run | made | -57.95 | 1.0 |  |
| pro_whoosh_big | made | -55.05 | 1.0 | -1.31 |
| pro_zip | made | -52.9 | 0.99 | -1.67 |
|  | real loo | -21.95 | 0.93 | 0.24 |
| riser | made | -37.6 | 0.61 |  |
|  | real x | -49.9 | 0.0 |  |
|  | real loo | -30.65 | 1.0 | 0.12 |
| shimmer | made | -57.5 | 0.07 | -0.64 |
|  | real x | -46.8 | 0.17 | -0.29 |
|  | real loo | -36.1 | 0.68 | -0.33 |
| sub_drop | made | -67.0 | 1.0 | -0.59 |
| swipe | made | -61.9 | 0.64 | 0.19 |
|  | real x | -56.85 | 0.42 | 0.07 |
|  | real loo | -38.05 | 0.68 | 0.61 |
| swish | made | -57.9 | 0.96 | -0.02 |
|  | real x | -55.6 | 0.87 | -0.16 |
|  | real loo | -44.6 | 1.0 | -0.45 |
| tap | made | -49.8 | 0.99 | 0.0 |
|  | real x | -43.6 | 1.0 | 1.31 |
|  | real loo | -38.4 | 1.0 | 0.94 |
| toggle | made | -67.3 | 1.0 | 0.32 |
|  | real x | -48.4 | 1.0 | -0.11 |
|  | real loo | -49.1 | 0.48 | -0.18 |
| typing_key | made | -51.8 | 0.96 | -0.43 |
|  | real x | -43.7 | 0.76 | 0.04 |
|  | real loo | -39.65 | 1.0 | 0.09 |
| whoosh_big | made | -59.3 | 0.94 | -0.1 |
|  | real x | -43.2 | 0.66 | -0.11 |
|  | real loo | -42.9 | 1.0 | 0.28 |
| whoosh_fast | made | -64.9 | 0.97 | -0.05 |
|  | real x | -65.4 | 0.81 | -0.16 |
|  | real loo | -32.2 | 1.0 | -0.2 |

Diversity among the made variants (CLAP cosine mean / max, tonal distance min dB, near duplicates):

- click: cos 0.723 / 0.837, tone min 7.41 dB, 0 near duplicate pair(s) (bench refs: cos 0.576 / 0.792, tone min 6.13 dB)
- downlifter: cos 0.749 / 0.955, tone min 3.12 dB, 0 near duplicate pair(s)
- glitch: cos 0.728 / 0.871, tone min 1.8 dB, 0 near duplicate pair(s)
- hover_tick: cos 0.807 / 0.923, tone min 3.74 dB, 0 near duplicate pair(s) (bench refs: cos 0.626 / 0.847, tone min 9.21 dB)
- impact: cos 0.849 / 0.955, tone min 1.01 dB, 0 near duplicate pair(s)
- logo_sting: cos 0.714 / 0.911, tone min 10.41 dB, 0 near duplicate pair(s) (bench refs: cos 0.507 / 0.507, tone min 23.22 dB)
- notify: cos 0.867 / 0.959, tone min 5.35 dB, 0 near duplicate pair(s) (bench refs: cos 0.476 / 0.905, tone min 11.33 dB)
- pop: cos 0.735 / 0.877, tone min 5.92 dB, 0 near duplicate pair(s) (bench refs: cos 0.522 / 0.832, tone min 7.62 dB)
- pro_click: cos 0.786 / 0.902, tone min 6.93 dB, 0 near duplicate pair(s) (bench refs: cos 0.45 / 0.45, tone min 18.53 dB)
- pro_glitch: cos 0.853 / 0.912, tone min 1.35 dB, 0 near duplicate pair(s)
- pro_hover_tick: cos 0.862 / 0.942, tone min 1.79 dB, 0 near duplicate pair(s) (bench refs: cos 0.684 / 0.776, tone min 6.36 dB)
- pro_impact: cos 0.788 / 0.898, tone min 5.23 dB, 0 near duplicate pair(s)
- pro_logo_sting: cos 0.908 / 0.966, tone min 3.82 dB, 0 near duplicate pair(s)
- pro_notify: cos 0.822 / 0.902, tone min 6.02 dB, 0 near duplicate pair(s)
- pro_pop: cos 0.649 / 0.822, tone min 5.8 dB, 0 near duplicate pair(s)
- pro_pop_in: cos 0.75 / 0.823, tone min 5.62 dB, 0 near duplicate pair(s)
- pro_riser: cos 0.712 / 0.808, tone min 2.42 dB, 0 near duplicate pair(s)
- pro_shimmer: cos 0.646 / 0.808, tone min 7.57 dB, 0 near duplicate pair(s)
- pro_sub_drop: cos 0.835 / 0.885, tone min 4.46 dB, 0 near duplicate pair(s)
- pro_tap: cos 0.861 / 0.939, tone min 7.57 dB, 0 near duplicate pair(s)
- pro_typing_key: cos 0.815 / 0.952, tone min 1.1 dB, 0 near duplicate pair(s)
- pro_typing_run: cos 0.813 / 0.951, tone min 1.36 dB, 0 near duplicate pair(s)
- pro_whoosh_big: cos 0.882 / 0.966, tone min 3.24 dB, 0 near duplicate pair(s)
- pro_zip: cos 0.761 / 0.922, tone min 3.83 dB, 0 near duplicate pair(s) (bench refs: cos 0.529 / 0.686, tone min 8.63 dB)
- riser: cos 0.801 / 0.872, tone min 2.95 dB, 0 near duplicate pair(s) (bench refs: cos 0.767 / 0.767, tone min 13.22 dB)
- shimmer: cos 0.845 / 0.938, tone min 6.82 dB, 0 near duplicate pair(s) (bench refs: cos 0.644 / 0.765, tone min 10.66 dB)
- sub_drop: cos 0.727 / 0.889, tone min 4.89 dB, 0 near duplicate pair(s)
- swipe: cos 0.812 / 0.897, tone min 2.37 dB, 0 near duplicate pair(s) (bench refs: cos 0.857 / 0.918, tone min 2.76 dB)
- swish: cos 0.796 / 0.946, tone min 3.54 dB, 0 near duplicate pair(s) (bench refs: cos 0.79 / 0.908, tone min 9.83 dB)
- tap: cos 0.782 / 0.892, tone min 3.68 dB, 0 near duplicate pair(s) (bench refs: cos 0.716 / 0.98, tone min 2.68 dB)
- toggle: cos 0.729 / 0.782, tone min 4.99 dB, 0 near duplicate pair(s) (bench refs: cos 0.725 / 0.846, tone min 5.33 dB)
- typing_key: cos 0.713 / 0.885, tone min 6.37 dB, 0 near duplicate pair(s) (bench refs: cos 0.673 / 0.919, tone min 4.64 dB)
- whoosh_big: cos 0.76 / 0.902, tone min 3.3 dB, 0 near duplicate pair(s) (bench refs: cos 0.495 / 0.768, tone min 3.17 dB)
- whoosh_fast: cos 0.768 / 0.903, tone min 3.49 dB, 0 near duplicate pair(s) (bench refs: cos 0.772 / 0.951, tone min 2.28 dB)

## Judge notes (round 4, finalizer, 13:35-14:00)

Made = `out/forge_v2/<category>/final/` after round 4 (183 sounds, 34 categories). Round 4 re-delivered 14 categories
(whoosh area: whoosh_fast, swish, swipe, whoosh_big; impact area: impact, pro_impact, logo_sting, pro_logo_sting;
ui-tonal area: toggle, tap, pro_tap, notify, pro_pop (one member), riser). The round-3 finals of those 14 are in
`<category>/final_r3/`. pro_notify and pro_whoosh_big were tried and not changed. The other 18 categories are byte-identical to round 3 and give the same numbers.
Each area ran the benchmark once, at the end of its own work, to confirm (`out/forge_v2/bench_r4_<area>.json`, not
committed). This file is one full run at 13:35 on the final tree. Nothing was tuned on it. It only decided round 3
vs round 4 per category. Nobody has listened yet.

Rule for r3 vs r4: a check counts as better or worse only when it moves relative to what real recordings score on it.
This is the same rule as `out/showcase/progress/progress.md` (`blind_v2.verdict`). The recording floor is left out,
because made sounds sit near digital silence. Model checks count as up or down. The spectrogram sheets of each area
(`out/forge_v2/_judge/r4_whoosh/`, `_uitonal_r4/`, `impact/r4/`) were the tie-breaker.

### Round 3 -> round 4 (made medians on the held-out bench; v2r3 vs v2r4)

| category | var cos | tone dB | env oct | curve dB | t40 x | phrase z | margin | AS names | darken | inside | model-free better / worse | models up / down |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whoosh_fast | 0.886 -> 0.768 | 5.79 -> 6.28 | 0.51 -> 0.32 | 12.27 -> 10.32 | 0.9 -> 0.85 | 3.49 -> 3.23 | 2.56 -> 1.66 | 0.17 -> 0.5 | -0.64 -> -0.05 | 4.5 -> 4 | 4 / 1 | 2 / 2 |
| swish | 0.901 -> 0.796 | 7.37 -> 6.69 | 0.25 -> 0.23 | 10.49 -> 11.69 | 1.04 -> 1.02 | 2.71 -> 2.94 | 1.4 -> 1.62 | 0 -> 0.8 | -0.73 -> -0.02 | 4 -> 5 | 2 / 0 | 4 / 0 |
| swipe | 0.885 -> 0.812 | 6.06 -> 5.15 | 0.23 -> 0.38 | 9.81 -> 7.67 | 0.99 -> 0.92 | 1.44 -> 1.88 | 0.22 -> 0.11 | 0.6 -> 0.6 | -1.96 -> 0.19 | 4 -> 4 | 3 / 1 | 2 / 1 |
| whoosh_big | 0.854 -> 0.76 | 6.67 -> 6.87 | 0.44 -> 0.4 | 8.97 -> 10.21 | 0.97 -> 1.05 | 2.38 -> 2.44 | 0.03 -> -0.02 | 0.83 -> 0.83 | -1.2 -> -0.1 | 7 -> 7 | 2 / 1 | 1 / 1 |
| impact | 0.905 -> 0.849 | 5.31 -> 4.38 | 2.47 -> 2.39 | 30.34 -> 30.05 | 0.56 -> 0.58 | 3.34 -> 3.25 | 0.88 -> 0.9 | 1 -> 1 | -1.21 -> -0.96 | 6 -> 6 | 2 / 1 | 0 / 1 |
| pro_impact | 0.745 -> 0.788 | - | - | - | - | 2.55 -> 2.17 | -0.07 -> -0.09 | 0.6 -> 0.6 | -0.25 -> 0.41 | - -> - | 0 / 1 | 0 / 1 |
| logo_sting | 0.722 -> 0.714 | 16.35 -> 13.06 | 1.22 -> 1.68 | 14.81 -> 14.29 | 1.03 -> 1 | 1.32 -> 1.25 | -0.69 -> -0.48 | 1 -> 1 | 0.18 -> 0.41 | 6 -> 5 | 2 / 2 | 1 / 1 |
| pro_logo_sting | 0.943 -> 0.908 | 24.43 -> 15.93 | 2.32 -> 2.5 | 23.36 -> 23.34 | 0.13 -> 0.12 | 0.83 -> 0.95 | -1.15 -> -1.3 | 0.2 -> 0 | -0.43 -> -0.65 | 3 -> 4 | 1 / 1 | 1 / 2 |
| toggle | 0.718 -> 0.729 | 10.36 -> 7.82 | 0.62 -> 0.42 | 14.64 -> 10.11 | 1.7 -> 0.74 | 1.45 -> 1.72 | -1.35 -> -0.38 | 0 -> 0 | 0.45 -> 0.32 | 5 -> 5 | 4 / 1 | 2 / 0 |
| tap | 0.807 -> 0.782 | 8.23 -> 5.81 | 1.4 -> 1.4 | 14.51 -> 11.65 | 1.82 -> 1 | 1.13 -> 1.12 | 0.34 -> 0.77 | 0.6 -> 0.2 | 0 -> 0 | 5 -> 4 | 4 / 0 | 1 / 2 |
| pro_tap | 0.872 -> 0.861 | - | - | - | - | 1.65 -> 1.98 | 1.81 -> 1.93 | 0.2 -> 0.2 | -0.76 -> 0.07 | - -> - | 0 / 0 | 2 / 0 |
| notify | 0.864 -> 0.867 | 18.29 -> 15.63 | 0.81 -> 0.66 | 13.55 -> 13.09 | 0.85 -> 0.86 | 1.4 -> 1.54 | -0.59 -> -0.5 | 1 -> 1 | -2 -> -1.53 | 7 -> 7 | 2 / 0 | 2 / 0 |
| pro_pop | 0.686 -> 0.649 | 23.63 -> 22.93 | 1.41 -> 1.41 | 35.59 -> 35.59 | 0.75 -> 0.75 | 3.32 -> 3.29 | 1.81 -> 1.76 | 0 -> 0 | -1.45 -> -0.31 | 3 -> 3 | 2 / 0 | 0 / 0 |
| riser | 0.82 -> 0.801 | 11.3 -> 9.33 | 2.08 -> 2.35 | 21.7 -> 23.23 | 0.09 -> 0.09 | 2.51 -> 2.65 | -0.3 -> -0.3 | 0.4 -> 0.6 | -0.03 -> - | 3 -> 3 | 1 / 2 | 2 / 0 |

Named faults the round-3 judge set for round 4, as measured model-free by the areas (their LOG.md; r3 -> r4,
reference):
- impact contact sheen: onset >6 kHz share -10.9 -> -17.4 dB on the delivered renders (median over 4 seeds -19.3),
  against a bench value of -21.6. Onset >12 kHz went from -14.3 to -30.7 dB.
- logo_sting body: share under 150 Hz -52.7 -> -23.5 dB (bench -26.8).
- pro_logo_sting top end: 4-12 kHz -41.8 -> -23.2 dB (film -21.0); variety 0.943 -> 0.908.
- whoosh_fast / swish 'door' reading:
  - Cause: the tail, measured. Tail alone: AudioSet door-like sum 0.26-1.19. Head alone: 0.02-0.10.
  - whoosh_fast: door-like top-1 on 6/6 -> 3/6.
  - swish: 'Door' top-1 on 3/5 -> none.
- Tails that darken too fast (darken; the bench median is about -0.2): whoosh_fast -0.64 -> -0.05, swipe -1.96 -> +0.19,
  whoosh_big -1.2 -> -0.1.
- swipe: onset >6 kHz -23.0 -> -17.9 dB; stereo corr 0.90 -> 0.64 (bench 0.68).
- toggle ring lines: the strongest late narrow line went from 15.5 to 13.1 dB, and the 18 ms spring rings are gone.
- tap 4-12 kHz: -18.4 -> -25.0 dB (forge refs -29.6, held-out taps -26.4).
- notify: AudioSet top label 'Ding' on 2/6 -> 3/6; the three low bells still read 'Clang'.

**No category went back to round 3.** Every one of the 14 round-4 deliveries stays. One-line reasons are in each
category's `LOG.md` ("Finalizer"). The close calls:
- **pro_impact.** This category has no bench refs, so its only held-out numbers are model checks. They prefer round 3:
  phrase z 2.55 -> 2.17, margin -0.07 -> -0.09, variety 0.745 -> 0.788. Round 4 stays anyway. Round 3 carried an
  accidental 0 dB onset tick: forge's final pass filled the SPACE keys missing from a member at mid-range (see
  'Core bug' below). Round 4 removes that tick, and its irregular grain stripes match the film ref on the sheet. The
  model checks probably liked the tick.
- **riser.** Better: tone 11.3 -> 9.3 dB, variety, phrase z, AudioSet names 0.4 -> 0.6. Worse: env oct 2.08 -> 2.35
  and curve 21.7 -> 23.2 dB. Those two are measured against two real cymbal swells, and the made risers are synth /
  noise risers.
- **logo_sting.** Better: tone 16.4 -> 13.1 dB, curve, margin. Worse: env oct 1.22 -> 1.68, and 6 -> 5 of 6 inside.
  The bench refs are bare rings, so the low body the judge asked for costs envelope checks by construction.

### v1 -> v2 (final), the five categories v1 had

v1 = the recipes on `main` (`results_v1.json`); v2 = this file. The bracketed values are what real recordings score
on the same check.

| category | var cos v1 -> v2 (bench refs) | t40 x v1 -> v2 (real x / loo) | tone dB v1 -> v2 (real x / loo) | curve dB v1 -> v2 | env oct v1 -> v2 | inside v1 -> v2 (real) |
|---|---|---|---|---|---|---|
| typing_key | 0.965 -> 0.713 (0.673) | 1.67 -> 1 (1.29 / 1) | 8.2 -> 7.44 (8.88 / 5.47) | 14.32 -> 9.08 | 1.07 -> 1.04 | 7 -> 4 of 7 (6) |
| click | 0.886 -> 0.723 (0.576) | 0.56 -> 1.44 (1.33 / 0.86) | 7.6 -> 8.79 (11.33 / 9.85) | 8.85 -> 8.83 | 0.56 -> 0.23 | 7 -> 7 of 7 (6) |
| pop | 0.917 -> 0.735 (0.522) | 1 -> 0.82 (0.91 / 1.02) | 8.59 -> 11.42 (16.06 / 13.66) | 6.29 -> 11.01 | 0.68 -> 0.55 | 7 -> 7 of 7 (6) |
| whoosh_fast | 0.957 -> 0.768 (0.772) | 0.51 -> 0.85 (0.6 / 1.08) | 8.58 -> 6.28 (10.31 / 4.58) | 15.93 -> 10.32 | 0.89 -> 0.32 | 7 -> 4 of 7 (6.5) |
| impact | 0.931 -> 0.849 (0.659) | 0.5 -> 0.58 (0.46 / -) | 6.7 -> 4.38 (6.19 / -) | 24.86 -> 30.05 | 1.54 -> 2.39 | 5 -> 6 of 6 (3) |

- **Variety.** Better in all five. v1 0.89-0.97 (near clones); v2 0.71-0.85. Real sets score 0.52-0.77.
- **Tails.** click 0.56 -> 1.44, whoosh_fast 0.51 -> 0.85, typing_key's long tail 1.67 -> 1.00. impact stays short
  against its one long bench hit, and the real forge refs score 0.46 there too.
- **Tone.** Closer in 3 of 5: typing_key, whoosh_fast, impact. click and pop are further, but still closer than the
  real forge refs.
- **Model checks fell.** v1 sat above the real rows on CLAP typicality and phrasings. It was tuned to please the
  models, partly against kit refs that are now bench refs. typing_key's and whoosh_fast's lower inside counts are
  CLAP / AudioSet misses.

### Breadth

- **Scope.** 34 categories and 183 finals, against v1's 5 and 15.
- **Artifacts.** 0 flags on the bench (round 3: 0; round 2: 3 of 168).
- **Tone against the real forge refs.** 21 categories have a bench tone number. In 18 of them the made median is at
  least as close to the held-out balance as the real forge refs (round 3: 17). The exceptions:
  - pro_logo_sting: 15.9 vs 15.5 dB
  - pro_pop: 22.9 vs 18.4 dB
  - shimmer: 10.7 vs 9.7 dB
- **Inside the real limits.** 22 categories have the count. In 11 the made median is at or above a real recording's
  count: click, hover_tick, impact, notify, pop, pro_click, pro_hover_tick, pro_pop, pro_zip, shimmer, whoosh_big.
- **Variety.** Made sets score 0.65-0.91. The most clone-like are pro_logo_sting 0.908 and pro_whoosh_big 0.882.

### Verdicts after round 4

Showcase-ready, measured:
- whoosh_big (7/7 inside), swish (now 'Whoosh', not 'Door'), swipe, whoosh_fast
- shimmer, click, pop (round-2 family), notify
- impact, typing_key, pro_typing_key / run, pro_pop_in, pro_hover_tick, pro_click, pro_whoosh_big
- logo_sting, downlifter, sub_drop, pro_glitch, pro_riser, pro_zip
- toggle (now middle)

Weakest:
- pro_logo_sting: clones, and CLAP hears 'error buzz'.
- pro_pop: '8-bit'.
- pro_notify: unchanged; the 'clean' attempts sounded cheaper.
- tap: model checks fell with the darker contact.
- riser: AudioSet's nearest confusion is still 'a jet engine starting'.
- pro_impact: the model checks prefer round 3.
- glitch, pro_tap.

Core bug found in round 4, not fixed: `forge()`'s final pass fills the SPACE keys a member leaves out at mid-range
instead of from DEFAULTS. Families that are not built on DEFAULTS can carry hidden values; pro_impact round 3 was one.
