# forge benchmark: v2r2

made: out/forge_v2; refs: <EARSHOT_HOME>/refsets.json; 2026-10-16 08:44

Medians per set. made = the made sounds vs the bench refs; real x = the forge refs (real, never in the bench) vs the bench refs; real loo = each bench ref vs the other bench refs. typ z, phrase z, margin, AS names: higher is better; tone dB, env oct (octaves off in attack / t40 / length), curve dB: lower is better; t40 x = tail length / bench median (1 = same). inside = checks (of n) a made sound lands inside the real 80% limit, median over the made sounds (real = the same count for the real rows). var cos = mean pairwise CLAP cosine within the set (made variants / forge refs / bench refs): high = the variants are near clones.

| category | set | n | typ z | tone dB | env oct | curve dB | t40 x | phrase z | margin | AS names | artifacts | var cos | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| click | made | 5 | -0.25 | 9.86 | 0.24 | 8.61 | 1.22 | 2.27 | 0.56 | 0 | 0 | 0.681 | 5 / 7 (real 6) |
|  | real x | 8 | 0.16 | 11.33 | 0.22 | 9.92 | 1.33 | 2.77 | 0.53 | 0.12 | 0 | 0.651 |  |
|  | real loo | 7 | -0.1 | 9.85 | 0.46 | 13.19 | 0.86 | 2.67 | 0.27 | 0 | 0 | 0.576 |  |
| downlifter | made | 6 |  |  |  |  |  | 3.23 | 0.92 | 0.33 | 0 | 0.749 |  |
| | note: empty: the licensed library has no downlifter, real or designed (no reversing or pitching of other sounds here: trim + fade only). Benchmark made downlifters on phrasings, AudioSet and artifacts only. | |  |  |  |  |  |  |  |  |  |  |  |
| glitch | made | 4 |  |  |  |  |  | 2.14 | 0.08 | 0.5 | 0 | 0.799 |  |
| | note: empty: no glitch material, real or designed, in the licensed library. Benchmark made glitches on phrasings, AudioSet and artifacts only. | |  |  |  |  |  |  |  |  |  |  |  |
| hover_tick | made | 5 | 0.13 | 20.99 | 1.51 | 6.27 | 1.1 | 2.41 | 0.66 | 0 | 0 | 0.807 | 6 / 7 (real 5) |
|  | real x | 6 | 0.09 | 21.2 | 1.64 | 9.88 | 0.9 | 2.51 | 0.9 | 0 | 0 | 0.799 |  |
|  | real loo | 6 | 0.47 | 15.46 | 1.16 | 19.39 | 1.2 | 2 | 0.58 | 0 | 4 | 0.626 |  |
| impact | made | 5 |  | 6.29 | 2.11 | 32.62 | 0.62 | 3.61 | 1.21 | 1 | 1 | 0.881 | 5 / 6 (real 3) |
|  | real x | 2 |  | 6.19 | 2.5 | 31.98 | 0.46 | 3.04 | 1.09 | 0.5 | 0 | 0.659 |  |
| | note: thin: the library has three cinematic hits (all three were the v1 impact forge refs); one is held out as bench, so v1 impact bench numbers are optimistic. | |  |  |  |  |  |  |  |  |  |  |  |
| logo_sting | made | 4 |  | 16.29 | 2.01 | 25.16 | 2.48 | 1.55 | -0.85 | 1 | 1 | 0.733 | 3.5 / 6 (real 5.5) |
|  | real x | 2 |  | 15.52 | 1.46 | 21.27 | 1.7 | 1.4 | -0.82 | 0.5 | 0 | 0.185 |  |
|  | real loo | 2 |  | 23.22 | 0.81 | 8.99 | 1.12 | 1.36 | -0.58 | 1 | 0 | 0.507 |  |
| | note: no real logo stings exist in the library (they are composed); the refs are the struck metal / glass rings a sting ends on: a tail-shape and tone reference only. | |  |  |  |  |  |  |  |  |  |  |  |
| notify | made | 5 | -0.02 | 18.65 | 0.57 | 21.83 | 0.63 | 2.21 | -0.65 | 1 | 0 | 0.81 | 7 / 7 (real 5.5) |
|  | real x | 5 | -0.24 | 21.3 | 1.3 | 28.92 | 0.34 | 1.25 | -0.24 | 1 | 0 | 0.593 |  |
|  | real loo | 5 | 3.84 | 16.7 | 0.81 | 17.41 | 1.19 | 1.76 | -0.91 | 1 | 0 | 0.476 |  |
| | note: struck bells / glass / kalimba (forge) and celesta / glass / triangle (bench), rings capped at 2 s with a 0.25 s fade; no designed UI dings in the library. | |  |  |  |  |  |  |  |  |  |  |  |
| pop | made | 5 | -0.02 | 11.42 | 0.55 | 11.01 | 0.82 | 2.32 | -0.2 | 0.4 | 0 | 0.735 | 7 / 7 (real 6) |
|  | real x | 7 | -0.09 | 16.06 | 1.51 | 20.54 | 0.91 | 2.46 | -0.5 | 0.29 | 0 | 0.501 |  |
|  | real loo | 6 | 1.13 | 13.66 | 0.41 | 12.5 | 1.02 | 1.96 | -1.06 | 0.67 | 0 | 0.522 |  |
| pro_click | made | 5 |  | 25.79 | 1.46 | 13.82 | 0.67 | 4.06 | 2.64 | 0 | 0 | 0.87 | 4 / 6 (real 4) |
|  | real x | 1 |  | 29.35 | 1.06 | 10.68 | 0.53 | 4.25 | 2.96 | 0 | 0 |  |  |
|  | real loo | 2 |  | 18.53 | 1.73 | 8.99 | 1.08 | 2.19 | 0.51 | 0 | 2 | 0.45 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_glitch | made | 5 |  |  |  |  |  | 3.02 | 0.95 | 0.6 | 0 | 0.875 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only, one example. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_hover_tick | made | 5 | -1.81 | 7.89 | 0.88 | 10.7 | 1.17 | 0.85 | -0.54 | 0 | 0 | 0.878 | 4 / 7 (real 6) |
|  | real x | 3 | -2.23 | 6.51 | 1.11 | 20.12 | 1 | 3.04 | 0.97 | 0 | 0 | 0.82 |  |
|  | real loo | 3 |  | 8.32 | 0.67 | 5.84 | 0.86 | 1.79 | 0.71 | 0 | 1 | 0.684 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_impact | made | 5 |  |  |  |  |  | 2.71 | 0.6 | 0.8 | 0 | 0.733 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only (one SFX-only film); designed, sub-heavy, gated tails. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_logo_sting | made | 4 |  | 24.42 | 2.33 | 22.75 | 0.15 | 0.73 | -1.3 | 0 | 0 | 0.947 | 3 / 6 (real 6) |
|  | real x | 1 |  | 15.49 | 2.84 | 24.51 | 0.06 | 0.99 | -0.25 | 0 | 0 |  |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge: falling chirp into a tonal hum; bench: a music end chord with a 4 s tail. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_notify | made | 5 |  |  |  |  |  | 1.71 | 0.08 | 1 | 0 | 0.882 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_pop | made | 5 |  | 23.68 | 1.25 | 37.49 | 0.81 | 3.29 | 1.89 | 0 | 0 | 0.663 | 4 / 6 (real 3) |
|  | real x | 2 |  | 18.43 | 2.78 | 42.94 | 0.06 | 3.54 | 2.33 | 0 | 0 | 0.488 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_pop_in | made | 5 |  |  |  |  |  | 0.62 | -3.53 | 0 | 0 | 0.895 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). not a forge category yet: element-in micro hits (thump + tick), forge only. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_riser | made | 5 |  | 20.8 | 1.61 | 36.34 | 0.05 | 2 | -0.78 | 0 | 0 | 0.676 | 3 / 6 (real 6) |
|  | real x | 1 |  | 21.35 | 0.79 | 18.43 | 0.68 | 1.03 | -0.72 | 0 | 0 |  |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge: one designed stutter riser; bench: one noise riser at a film end. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_shimmer | made | 5 |  |  |  |  |  | 3.29 | 1 | 0.2 | 0 | 0.646 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only; one granular sparkle + one melodic chime run. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_sub_drop | made | 5 |  | 24.84 | 0.54 | 16.11 | 1.64 | 2.44 | 0.68 | 0 | 0 | 0.835 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). bench only, one example under music + voice. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_tap | made | 5 |  |  |  |  |  | 1.73 | 1.9 | 0 | 0 | 0.854 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only, one example. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_typing_key | made | 5 |  | 26.38 | 0.51 | 15.74 | 0.64 | 1.84 | -0.85 | 0 | 0 | 0.926 | 4 / 6 (real 5.5) |
|  | real x | 6 |  | 26.2 | 0.52 | 11.54 | 0.71 | 1.88 | -0.42 | 0 | 0 | 0.849 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md).  | |  |  |  |  |  |  |  |  |  |  |  |
| pro_typing_run | made | 5 |  |  |  |  |  | 2 | 0.77 | 0 | 1 | 0.93 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). not a forge category yet: one whole designed typing run. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_whoosh_big | made | 5 |  |  |  |  |  | 3.93 | 1.6 | 1 | 0 | 0.931 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only (one SFX-only film); the bench films bury their whooshes under music + voice. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_zip | made | 5 | -2.41 | 11.89 | 0.71 | 20.81 | 0.84 | 2.34 | -1.39 | 0 | 0 | 0.761 | 5 / 7 (real 5) |
|  | real loo | 4 | 1.71 | 10.4 | 1 | 34.47 | 1.15 | 1.4 | 0.54 | 0 | 0 | 0.529 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). not a forge category yet: tonal pitch sweeps, bench only (two films over music). | |  |  |  |  |  |  |  |  |  |  |  |
| riser | made | 5 |  | 12.5 | 2.12 | 24.81 | 0.06 | 2.57 | -0.28 | 0.8 | 0 | 0.78 | 3 / 6 (real 4) |
|  | real x | 1 |  | 17.61 | 1.01 | 20.04 | 0.5 | 3.52 | 1.06 | 1 | 0 |  |  |
|  | real loo | 2 |  | 13.22 | 0.95 | 36.07 | 3.35 | 2.7 | 1.15 | 0.5 | 0 | 0.767 |  |
| | note: thin: one designed trailer riser (forge); the bench is two real cymbal swells, no synth risers. | |  |  |  |  |  |  |  |  |  |  |  |
| shimmer | made | 5 | 0.12 | 10.49 | 0.99 | 14.09 | 0.48 | 3.42 | 0.7 | 0.4 | 0 | 0.807 | 6 / 7 (real 6) |
|  | real x | 5 | 1.69 | 9.66 | 0.73 | 8.58 | 0.73 | 2.87 | 0.03 | 0.6 | 2 | 0.59 |  |
|  | real loo | 4 | 1.01 | 15.52 | 1.33 | 12.89 | 1.05 | 3.4 | 0.56 | 1 | 0 | 0.644 |  |
| | note: kit shimmer_01 (bar chimes sweep) is bench: its author also recorded the celesta notify bench. | |  |  |  |  |  |  |  |  |  |  |  |
| sub_drop | made | 5 |  | 8.84 | 1.76 | 37.38 | 0.31 | 3.31 | 0.7 | 0 | 0 | 0.727 |  |
| | note: one designed source (a tom pitched down 24 semitones by its author), held out as bench; the forge has no sub-drop refs. | |  |  |  |  |  |  |  |  |  |  |  |
| swipe | made | 5 | -14.89 | 13.47 | 0.34 | 6.68 | 0.86 | 1.82 | 0.34 | 0.4 | 0 | 0.855 | 4 / 7 (real 6) |
|  | real x | 4 | -10.07 | 16.83 | 0.43 | 14.12 | 0.79 | 2.7 | 0.59 | 1 | 0 | 0.845 |  |
|  | real loo | 4 | 1.25 | 3.64 | 0.22 | 10.07 | 1.02 | 3.42 | 2.08 | 0.25 | 0 | 0.857 |  |
| | note: cloth only: the library has no paper slides. | |  |  |  |  |  |  |  |  |  |  |  |
| swish | made | 5 | -2.8 | 7.48 | 0.64 | 13.78 | 0.68 | 2.68 | 1.62 | 0 | 0 | 0.86 | 4 / 7 (real 6) |
|  | real x | 7 | 1.94 | 11.08 | 0.64 | 12.88 | 0.64 | 3.22 | 1.6 | 0.57 | 0 | 0.811 |  |
|  | real loo | 3 |  | 9.01 | 0.91 | 16.45 | 1.16 | 3.5 | 1.92 | 0.67 | 0 | 0.79 |  |
| tap | made | 5 | -2.69 | 6.1 | 1.05 | 13.46 | 1.27 | 0.52 | 0.13 | 0 | 0 | 0.856 | 3 / 7 (real 6) |
|  | real x | 7 | -1.17 | 9.35 | 1.59 | 13.67 | 1.64 | 1.85 | 0.92 | 0.14 | 0 | 0.713 |  |
|  | real loo | 6 | 1.66 | 16.78 | 1.19 | 11.59 | 1 | 1.62 | 0.35 | 0.17 | 0 | 0.716 |  |
| toggle | made | 5 | -3.59 | 8.82 | 2.58 | 25.45 | 1.26 | 1.39 | -0.32 | 0 | 0 | 0.72 | 5 / 7 (real 6) |
|  | real x | 4 | -2.84 | 12.48 | 1.87 | 18.78 | 2.11 | 2.1 | -0.02 | 0 | 0 | 0.477 |  |
|  | real loo | 4 | 1.54 | 6.25 | 0.28 | 4.92 | 1.02 | 1.94 | -0.28 | 0 | 0 | 0.725 |  |
| typing_key | made | 5 | -0.15 | 8.15 | 0.97 | 11.12 | 1.29 | 4.46 | 2.26 | 0.2 | 0 | 0.705 | 6 / 7 (real 6) |
|  | real x | 15 | 0.26 | 8.88 | 1.01 | 12.2 | 1.29 | 4.75 | 2.83 | 0.53 | 0 | 0.721 |  |
|  | real loo | 12 | 0.23 | 5.47 | 1.52 | 16.7 | 1 | 3.71 | 1.53 | 0.42 | 0 | 0.673 |  |
| whoosh_big | made | 5 | 0.78 | 7.58 | 0.41 | 5.82 | 1.04 | 2.58 | 0.13 | 1 | 0 | 0.909 | 7 / 7 (real 5) |
|  | real x | 3 | -0.27 | 8.8 | 1.13 | 19.46 | 3 | 2.92 | 0.04 | 1 | 0 | 0.7 |  |
|  | real loo | 3 |  | 4.25 | 0.43 | 6.67 | 0.97 | 1.88 | 0.19 | 0.33 | 0 | 0.495 |  |
| | note: forge = three designed Mixkit whooshes; bench = real air / cape swoops. | |  |  |  |  |  |  |  |  |  |  |  |
| whoosh_fast | made | 5 | -4.85 | 9.43 | 0.95 | 18.4 | 0.49 | 3.28 | 2.46 | 0.6 | 0 | 0.856 | 4 / 7 (real 6.5) |
|  | real x | 5 | 0.57 | 10.31 | 1.22 | 20.2 | 0.6 | 4.14 | 2.78 | 0.4 | 0 | 0.826 |  |
|  | real loo | 5 | 1.52 | 4.58 | 0.4 | 8.21 | 1.08 | 4.14 | 2.62 | 0.8 | 0 | 0.772 |  |

Texture (medians; model-free, not in any forge objective): floor dB = 10th percentile 5 ms frame re the loudest (-120 = digital silence), stereo corr (1 = mono), darken = log2 centroid at half the tail / at the peak (< 0: the decay darkens).

| category | set | floor dB | stereo corr | darken |
|---|---|---|---|---|
| click | made | -45.2 | 1.0 | -0.12 |
|  | real x | -42.85 | 0.99 | -0.12 |
|  | real loo | -44.8 | 1.0 | -0.07 |
| downlifter | made | -54.45 | 0.67 | -1.81 |
| glitch | made | -39.5 | 0.13 | 0.01 |
| hover_tick | made | -44.0 | 0.88 | -0.09 |
|  | real x | -43.55 | 1.0 | -0.09 |
|  | real loo | -24.9 | 0.52 | -0.01 |
| impact | made | -71.8 | 0.74 | -0.8 |
|  | real x | -70.55 | 0.54 | -1.3 |
| logo_sting | made | -44.75 | 0.69 | -0.39 |
|  | real x | -52.1 | 0.59 | -0.85 |
|  | real loo | -44.1 | 0.81 | -1.43 |
| notify | made | -28.6 | 0.9 | -1.01 |
|  | real x | -41.2 | 0.99 | -1.06 |
|  | real loo | -24.4 | 0.61 | -0.53 |
| pop | made | -62.8 | 1.0 | 0.25 |
|  | real x | -53.2 | 1.0 | 0.21 |
|  | real loo | -47.2 | 1.0 | 0.12 |
| pro_click | made | -75.2 | 1.0 |  |
|  | real x | -108.5 | 0.0 |  |
|  | real loo | -30.6 | 0.95 | 0.26 |
| pro_glitch | made | -36.1 | 0.04 | 0.02 |
| pro_hover_tick | made | -41.8 | 0.97 | -0.14 |
|  | real x | -47.0 | -0.0 |  |
|  | real loo | -37.4 | 0.16 |  |
| pro_impact | made | -24.4 | 0.59 | 0.45 |
| pro_logo_sting | made | -55.25 | 1.0 | 0.2 |
|  | real x | -120.0 | 1.0 | -0.38 |
| pro_notify | made | -45.7 | 0.94 | -0.27 |
| pro_pop | made | -96.9 | 1.0 | -1.9 |
|  | real x | -120.0 | 1.0 |  |
| pro_pop_in | made | -87.2 | 0.58 | -0.89 |
| pro_riser | made | -29.9 | 0.91 |  |
|  | real x | -41.8 | 0.96 | 0.22 |
| pro_shimmer | made | -49.0 | 0.33 | -0.53 |
| pro_sub_drop | made | -38.3 | 1.0 | 0.13 |
| pro_tap | made | -101.6 | 0.46 | -0.27 |
| pro_typing_key | made | -42.7 | 1.0 | 0.02 |
|  | real x | -53.55 | 1.0 | -0.04 |
| pro_typing_run | made | -56.4 | 1.0 |  |
| pro_whoosh_big | made | -59.1 | 1.0 | -0.87 |
| pro_zip | made | -52.9 | 0.99 | -1.67 |
|  | real loo | -21.95 | 0.93 | 0.24 |
| riser | made | -33.4 | 0.61 |  |
|  | real x | -49.9 | 0.0 |  |
|  | real loo | -30.65 | 1.0 | 0.12 |
| shimmer | made | -53.9 | 0.33 | -0.16 |
|  | real x | -46.8 | 0.17 | -0.29 |
|  | real loo | -36.1 | 0.68 | -0.33 |
| sub_drop | made | -67.0 | 1.0 | -0.59 |
| swipe | made | -60.5 | 0.93 | -1.27 |
|  | real x | -56.85 | 0.42 | 0.07 |
|  | real loo | -38.05 | 0.68 | 0.61 |
| swish | made | -63.0 | 0.95 | -1.15 |
|  | real x | -55.6 | 0.87 | -0.16 |
|  | real loo | -44.6 | 1.0 | -0.45 |
| tap | made | -65.9 | 0.98 | 0.07 |
|  | real x | -43.6 | 1.0 | 1.31 |
|  | real loo | -38.4 | 1.0 | 0.94 |
| toggle | made | -49.5 | 0.99 | -0.48 |
|  | real x | -48.4 | 1.0 | -0.11 |
|  | real loo | -49.1 | 0.48 | -0.18 |
| typing_key | made | -49.0 | 0.34 | -1.17 |
|  | real x | -43.7 | 0.76 | 0.04 |
|  | real loo | -39.65 | 1.0 | 0.09 |
| whoosh_big | made | -61.6 | 0.95 | -1.04 |
|  | real x | -43.2 | 0.66 | -0.11 |
|  | real loo | -42.9 | 1.0 | 0.28 |
| whoosh_fast | made | -64.4 | 0.98 | -1.14 |
|  | real x | -65.4 | 0.81 | -0.16 |
|  | real loo | -32.2 | 1.0 | -0.2 |

Diversity among the made variants (CLAP cosine mean / max, tonal distance min dB, near duplicates):

- click: cos 0.681 / 0.83, tone min 6.69 dB, 0 near duplicate pair(s) (bench refs: cos 0.576 / 0.792, tone min 6.13 dB)
- downlifter: cos 0.749 / 0.955, tone min 3.12 dB, 0 near duplicate pair(s)
- glitch: cos 0.799 / 0.88, tone min 5.31 dB, 0 near duplicate pair(s)
- hover_tick: cos 0.807 / 0.923, tone min 3.74 dB, 0 near duplicate pair(s) (bench refs: cos 0.626 / 0.847, tone min 9.21 dB)
- impact: cos 0.881 / 0.966, tone min 2.34 dB, 0 near duplicate pair(s)
- logo_sting: cos 0.733 / 0.901, tone min 9.37 dB, 0 near duplicate pair(s) (bench refs: cos 0.507 / 0.507, tone min 23.22 dB)
- notify: cos 0.81 / 0.92, tone min 7.59 dB, 0 near duplicate pair(s) (bench refs: cos 0.476 / 0.905, tone min 11.33 dB)
- pop: cos 0.735 / 0.877, tone min 5.92 dB, 0 near duplicate pair(s) (bench refs: cos 0.522 / 0.832, tone min 7.62 dB)
- pro_click: cos 0.87 / 0.904, tone min 3.66 dB, 0 near duplicate pair(s) (bench refs: cos 0.45 / 0.45, tone min 18.53 dB)
- pro_glitch: cos 0.875 / 0.94, tone min 2.69 dB, 0 near duplicate pair(s)
- pro_hover_tick: cos 0.878 / 0.949, tone min 3.83 dB, 0 near duplicate pair(s) (bench refs: cos 0.684 / 0.776, tone min 6.36 dB)
- pro_impact: cos 0.733 / 0.827, tone min 4.73 dB, 0 near duplicate pair(s)
- pro_logo_sting: cos 0.947 / 0.968, tone min 2.38 dB, 0 near duplicate pair(s)
- pro_notify: cos 0.882 / 0.947, tone min 5.93 dB, 0 near duplicate pair(s)
- pro_pop: cos 0.663 / 0.849, tone min 9.48 dB, 0 near duplicate pair(s)
- pro_pop_in: cos 0.895 / 0.942, tone min 5.0 dB, 0 near duplicate pair(s)
- pro_riser: cos 0.676 / 0.802, tone min 5.68 dB, 0 near duplicate pair(s)
- pro_shimmer: cos 0.646 / 0.808, tone min 7.57 dB, 0 near duplicate pair(s)
- pro_sub_drop: cos 0.835 / 0.885, tone min 4.46 dB, 0 near duplicate pair(s)
- pro_tap: cos 0.854 / 0.892, tone min 7.57 dB, 0 near duplicate pair(s)
- pro_typing_key: cos 0.926 / 0.966, tone min 1.06 dB, 0 near duplicate pair(s)
- pro_typing_run: cos 0.93 / 0.958, tone min 1.36 dB, 0 near duplicate pair(s)
- pro_whoosh_big: cos 0.931 / 0.954, tone min 3.66 dB, 0 near duplicate pair(s)
- pro_zip: cos 0.761 / 0.922, tone min 3.83 dB, 0 near duplicate pair(s) (bench refs: cos 0.529 / 0.686, tone min 8.63 dB)
- riser: cos 0.78 / 0.887, tone min 5.02 dB, 0 near duplicate pair(s) (bench refs: cos 0.767 / 0.767, tone min 13.22 dB)
- shimmer: cos 0.807 / 0.907, tone min 8.08 dB, 0 near duplicate pair(s) (bench refs: cos 0.644 / 0.765, tone min 10.66 dB)
- sub_drop: cos 0.727 / 0.889, tone min 4.89 dB, 0 near duplicate pair(s)
- swipe: cos 0.855 / 0.924, tone min 2.77 dB, 0 near duplicate pair(s) (bench refs: cos 0.857 / 0.918, tone min 2.76 dB)
- swish: cos 0.86 / 0.942, tone min 3.97 dB, 0 near duplicate pair(s) (bench refs: cos 0.79 / 0.908, tone min 9.83 dB)
- tap: cos 0.856 / 0.892, tone min 5.15 dB, 0 near duplicate pair(s) (bench refs: cos 0.716 / 0.98, tone min 2.68 dB)
- toggle: cos 0.72 / 0.877, tone min 4.1 dB, 0 near duplicate pair(s) (bench refs: cos 0.725 / 0.846, tone min 5.33 dB)
- typing_key: cos 0.705 / 0.87, tone min 6.24 dB, 0 near duplicate pair(s) (bench refs: cos 0.673 / 0.919, tone min 4.64 dB)
- whoosh_big: cos 0.909 / 0.96, tone min 3.02 dB, 0 near duplicate pair(s) (bench refs: cos 0.495 / 0.768, tone min 3.17 dB)
- whoosh_fast: cos 0.856 / 0.931, tone min 3.86 dB, 0 near duplicate pair(s) (bench refs: cos 0.772 / 0.951, tone min 2.28 dB)

## Judge notes (round 2, integrator)

Made = `out/forge_v2/<category>/final/<category>_NN.wav` (168 sounds, 34 categories; `audition*.wav` skipped).
Everything below comes from this run, `eval/forge/results_v1.json` (v1 finals, same frozen benchmark), the critic
reports in each `final/report.json`, and spectrogram / envelope / 1/3-octave sheets of finals vs bench, forge and pro
refs (`out/forge_v2/_judge/<category>.png`, not committed). Band shares are dB of total energy (frozen_tone.shares),
medians per set. Nobody has listened yet: `final/audition.wav` per category is ready for that.

### v1 vs v2 round 2 (the five categories v1 covered; made medians on the held-out bench)

| category | var cos v1 -> v2 (real) | t40 x v1 -> v2 (real x / loo) | tone dB v1 -> v2 (real x / loo) | env oct | curve dB | typ z | AS names | inside v1 -> v2 (real) |
|---|---|---|---|---|---|---|---|---|
| click | 0.886 -> **0.681** (0.65 / 0.58) | 0.56 -> **1.22** (1.33 / 0.86) | 7.60 -> 9.86 (11.33 / 9.85) | 0.56 -> 0.24 | 8.85 -> 8.61 | 1.89 -> -0.25 | 0 -> 0 | 7 -> 5 of 7 (6) |
| typing_key | 0.965 -> **0.705** (0.72 / 0.67) | 1.67 -> **1.29** (1.29 / 1.00) | 8.20 -> 8.15 (8.88 / 5.47) | 1.07 -> 0.97 | 14.32 -> 11.12 | 1.20 -> -0.15 | 0.33 -> 0.20 | 7 -> 6 of 7 (6) |
| pop | 0.917 -> **0.735** (0.50 / 0.52) | 1.00 -> 0.82 (0.91 / 1.02) | 8.59 -> 11.42 (16.06 / 13.66) | 0.68 -> 0.55 | 6.29 -> 11.01 | 1.33 -> -0.02 | 1.0 -> 0.4 | 7 -> 7 of 7 (6) |
| impact | 0.931 -> 0.881 (0.66) | 0.50 -> 0.62 (0.46) | 6.70 -> 6.29 (6.19) | 1.54 -> 2.11 | 24.86 -> 32.62 | - | 1.0 -> 1.0 | 5 -> 5 of 6 (3) |
| whoosh_fast | 0.957 -> 0.856 (0.83 / 0.77) | 0.51 -> **0.49** (0.60 / 1.08) | 8.58 -> 9.43 (10.31 / 4.58) | 0.89 -> 0.95 | 15.93 -> 18.40 | 2.05 -> -4.85 | 0.33 -> 0.6 | 7 -> **4** of 7 (6.5) |

Read: the round-1 clone finding is fixed in all five (var cos now at or near the real sets), the click and key tails
are fixed (t40 x at the real level), and typ z / phrase z dropped from "above real" (prototypes) to real-like. The fast
whoosh is the exception: its tail is still half the bench median and it now misses 3 of 7 real limits that v1 met.
"inside" falls for click because v1 sat above the real rows on the saturated model checks.

### Verdicts per category (is it good for a motion-design launch video?)

Critic passes count the final members passing their own category critic (different refs per category, so totals are
not comparable across categories). 60 of 168 finals pass.

| category | verdict | key numbers (bench, made median) | concrete fault |
|---|---|---|---|
| whoosh_big | **best** | inside 7/7 (real 5), tone 7.58 (real x 8.80), t40 x 1.04, curve 5.8 dB, AudioSet "Whoosh" 5/5, critic 5/5 | variants near clones (var cos 0.909 vs bench 0.495) |
| typing_key | **good** | inside 6/7 (real 6), var 0.705, t40 x 1.29 = real x, curve 11.1 (real x 12.2), critic 3/5 | too wide (stereo corr 0.34, bench 1.00); <150 Hz share -5.5 vs -10.9 dB (thock boom); AudioSet names it 0.2 |
| pro_pop_in | **good** | 1/3-oct balance and envelopes overlay the 10 film refs, double hits 60-70 ms, 0 artifacts | near clones (0.895 vs refs 0.696); critic 0/5 (CLAP hears every low thump as a kick drum) |
| pro_whoosh_big | good | critic 5/5, AudioSet "Whoosh" 4/5, <150 Hz share -0.3 vs refs -0.2 | detached 5-15 kHz air island over a 1-4 kHz hole (1-4k share -46.5 vs -38.0); near clones 0.931 |
| pro_typing_key / pro_typing_run | good | balance overlays the 6 film keys; run 01 matches the film's spacing (87 ms median); critic 1/5, 2/5 | near clones 0.926 / 0.930 (the film keys themselves 0.849) |
| downlifter | good (no refs) | critic 6/6, var 0.749, 0 artifacts, phrase z 3.23 | judged on models + gesture only; 06 sub_sink reads "Bang" (0.51) |
| pro_zip | good | inside 5/7 (real 5), tone 11.9 (bench loo 10.4), var 0.761, critic 5/5 | AudioSet "Boing" / "Plop" on 2 of 5: toy-like pure tone |
| sub_drop / pro_sub_drop | good | critic 4/5, 3/5; 0 artifacts; harmonics put 150 Hz-1 kHz at -15 dB share (survives phones) | pro_sub_drop bench not independent (targets from that trim) |
| hover_tick | good | inside 6/7 (real 5), tone 21.0 (real x 21.2), var 0.807 (forge 0.799) | critic 0/5 (totals -0.18..1.31 vs refs 0.78..2.44) |
| glitch | ok (no refs) | critic 3/4, var 0.799, decorrelated, 0 artifacts | 04 hit weak; only 4 finals |
| click | ok | t40 x 1.22, env 0.24, var 0.681 (v1 0.886) | narrow partial lines (6 kHz tink, 250-500 Hz body) ring for the whole 80-130 ms in 3 of 5; 1-4 kHz share -14 vs bench -4.9 dB; critic 1/5 |
| pop | ok | inside 7/7, var 0.735, env 0.55 | no bright click layer: 1-4 kHz share -17.2 vs bench -2.0 dB; tone and AudioSet worse than v1 (11.4 vs 8.6 dB, 0.4 vs 1.0) |
| notify | ok | inside 7/7 (real 5.5), AudioSet "Ding" top-1 5/5, curve 21.8 (real x 28.9) | pitched 1-2 oct above the bench bells (1-4 kHz share -0.1 vs 150 Hz-1 kHz in the bench), no strike transient, t40 x 0.63; critic 0/5 |
| pro_notify | ok | balance overlays the 4 film dings (2 kHz + 3.4x partial), AudioSet "Ding" 5/5 | critic -5.7..-7.2 vs films 0.1..3.0 (CLAP "cheap beep"); var 0.882 |
| toggle | ok | inside 5/7, tone 8.82, var 0.72 | 150 Hz-1 kHz share -5.4 vs bench -19.2 dB (14 dB too much body); second contact is the peak (env oct 2.58) |
| impact | ok | tone 6.29 (real 6.19), t40 x 0.62 (v1 0.50), AudioSet "Explosion" 4/5 | dull onset: >6 kHz share of the first 15 ms -28.9 vs bench -21.6 dB; 1-4 kHz 8 dB thin; critic 1/5 |
| pro_impact | ok | critic 3/5, granular sub body matches the films | 15 dB less grain sparkle (4-12 kHz share -30.4 vs films -15.4); the body decays too little before the gate (last 50 ms at -14..-25 dB re peak vs the films' -29..-30) |
| riser | ok | inside 3/6 (real 4), var 0.78 (bench 0.767), AudioSet 0.8, critic 5/5 | riser_05 has a steady 2.5 kHz line for 2.5 s; riser_02 a steady sub drone line |
| pro_riser | ok- | var 0.676, critic 4/5 (maker) | inside 3/6 (real 6); AudioSet "Music" 4/5, CLAP "retro game": reads as a chiptune arpeggio |
| shimmer | ok- | inside 6/7, tone 10.5 (real x 9.7) | tails half the bench (t40 1.16 vs 2.42 s); steady narrow partial lines; chimes corr 0.97 |
| pro_shimmer | ok- | var 0.646, sparkle 01 looks like the film | runs flagged (fold, resonance); critic 1/5 |
| pro_pop | ok- | inside 4/6 (real 3), stepped chirp matches the film | critic 0/5 (best 1.57 vs film refs 1.56 / 0.19); CLAP "chiptune" |
| pro_click | ok- | critic 5/5 vs its one forge ref | vs the two held-out film clicks: 150 Hz-1 kHz share -41 vs -6 dB: bright ticks without body (tone 25.8 vs bench loo 18.5); var 0.870 |
| pro_tap | ok- | study targets met (maker) | critic 0/5, one ref, no bench |
| pro_logo_sting | ok- | chirp-to-hum timing matches the film sting | critic 0/4; inside 3/6 vs the bench end chord (a different sound) |
| pro_hover_tick | weak | inside 4/7 (real 6), tone 7.9 (real x 6.5) | phrase z 0.85; critic 0/5; var 0.878 |
| pro_glitch | weak | critic 3/5, balance matches the film | too even and dense: median 2 ms level -19..-24 dB re peak vs the film's -31 (less burst-to-burst drama); 01 adds a 50-200 Hz block; var 0.875 |
| tap | weak | tone 6.1 (real x 9.35), t40 x 1.27 | phrase z 0.52 (CLAP does not hear a fingertip on glass), inside 3/7 (real 6), var 0.856 (bench 0.716); critic 0/5 |
| swish | **worst 3 (with whoosh_fast)** | inside 4/7 (real 6), t40 x 0.68 | 4-12 kHz share -18.2 vs bench -33.5 dB (15 dB hiss); AudioSet top-1 "Whack, thwack" 3/5 and "Knock" 0.76: an impact-like onset; critic 0/5 |
| whoosh_fast | **worst 3** | inside 4/7 (v1 7/7), typ z -4.85 | tail dies 2x too fast (t40 130 vs 265 ms, x 0.49, v1 0.51); <150 Hz share -9.7 vs -17.7 dB (8 dB boom); 4-12 kHz -17.4 vs -32.3 dB (15 dB hiss); AudioSet "Knock" 2/5; critic 0/5 |
| logo_sting | **worst 3** | inside 3.5/6 (real 5.5), tone 16.3 (real x 15.5) | rings 2.5x too long (t40 2.14 vs 0.86 s); bell chords where single-ring refs have one partial set; critic 0/4; the refset (struck rings) is not a composed launch sting |
| swipe | **worst** | inside 4/7 (real 6), tone 13.5 vs bench loo 3.6 | muffled: 4-12 kHz share -35.9 vs bench cloth -19.2 dB, onset >6 kHz -27 vs -11.5 dB; AudioSet "Knock" / "Door" on 2 of 5 (reads as a thump); critic 0/5; the forge refs are just as dark (-39 dB), so the forge split steered it away from the bench |

Best three: **whoosh_big** (+ pro_whoosh_big), **typing_key** (+ pro_typing_key / run), **pro_pop_in**.
Worst three: **swipe**, **whoosh_fast + swish** (one engine, one fault), **logo_sting**.

### Integration changes in this round (judge)
- bench.py additions (no change to the 18 existing rows): `--made` reads `<category>/final/` when present and skips
  `audition*.wav` / `_*.wav` (four areas' first bench runs had counted audition.wav as a made sound); phrasings,
  confusions and AudioSet labels for the 16 pro_* categories; `pro_riser` ends on its peak like `riser`.
- After the first full run, `notify_04.wav` was re-rendered from its report params with the current recipe (identical after 10 ms; the fade-in
  removes the start click the bench flagged). logo_sting / pro_logo_sting `audition.wav` rebuilt without the rejected
  05 variants. The table above is the second full run, after that fix (24 s with cached embeddings; the first took
  375 s). Remaining artifact flags: 3, all by design: impact_01 and logo_sting_01 "late start" (a reverse swell /
  pre-swell before the hit: place them by the hit, not the file start) and pro_typing_run_04 "tail cut" (the run's
  low swell ends on the cut, as in the film).
