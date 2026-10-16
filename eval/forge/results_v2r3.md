# forge benchmark: v2r3

made: out/forge_v2; refs: <EARSHOT_HOME>/refsets.json; 2026-10-16 11:15

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
| impact | made | 6 |  | 5.31 | 2.47 | 30.34 | 0.56 | 3.34 | 0.88 | 1 | 0 | 0.905 | 6 / 6 (real 3) |
|  | real x | 2 |  | 6.19 | 2.5 | 31.98 | 0.46 | 3.04 | 1.09 | 0.5 | 0 | 0.659 |  |
| | note: thin: the library has three cinematic hits (all three were the v1 impact forge refs); one is held out as bench, so v1 impact bench numbers are optimistic. | |  |  |  |  |  |  |  |  |  |  |  |
| logo_sting | made | 5 |  | 16.35 | 1.22 | 14.81 | 1.03 | 1.32 | -0.69 | 1 | 0 | 0.722 | 6 / 6 (real 5.5) |
|  | real x | 2 |  | 15.52 | 1.46 | 21.27 | 1.7 | 1.4 | -0.82 | 0.5 | 0 | 0.185 |  |
|  | real loo | 2 |  | 23.22 | 0.81 | 8.99 | 1.12 | 1.36 | -0.58 | 1 | 0 | 0.507 |  |
| | note: no real logo stings exist in the library (they are composed); the refs are the struck metal / glass rings a sting ends on: a tail-shape and tone reference only. | |  |  |  |  |  |  |  |  |  |  |  |
| notify | made | 6 | 0.15 | 18.29 | 0.81 | 13.55 | 0.85 | 1.4 | -0.59 | 1 | 0 | 0.864 | 7 / 7 (real 5.5) |
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
| pro_impact | made | 5 |  |  |  |  |  | 2.55 | -0.07 | 0.6 | 0 | 0.745 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only (one SFX-only film); designed, sub-heavy, gated tails. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_logo_sting | made | 5 |  | 24.43 | 2.32 | 23.36 | 0.13 | 0.83 | -1.15 | 0.2 | 0 | 0.943 | 3 / 6 (real 6) |
|  | real x | 1 |  | 15.49 | 2.84 | 24.51 | 0.06 | 0.99 | -0.25 | 0 | 0 |  |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge: falling chirp into a tonal hum; bench: a music end chord with a 4 s tail. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_notify | made | 5 |  |  |  |  |  | 1.91 | 0.15 | 1 | 0 | 0.822 |  |
| | note: pro designed SFX from the reference launch films (bed under some trims; see docs/motion-sfx-study.md). forge only. | |  |  |  |  |  |  |  |  |  |  |  |
| pro_pop | made | 6 |  | 23.63 | 1.41 | 35.59 | 0.75 | 3.32 | 1.81 | 0 | 0 | 0.686 | 3 / 6 (real 3) |
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
| pro_tap | made | 5 |  |  |  |  |  | 1.65 | 1.81 | 0.2 | 0 | 0.872 |  |
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
| riser | made | 5 |  | 11.3 | 2.08 | 21.7 | 0.09 | 2.51 | -0.3 | 0.4 | 0 | 0.82 | 3 / 6 (real 4) |
|  | real x | 1 |  | 17.61 | 1.01 | 20.04 | 0.5 | 3.52 | 1.06 | 1 | 0 |  |  |
|  | real loo | 2 |  | 13.22 | 0.95 | 36.07 | 3.35 | 2.7 | 1.15 | 0.5 | 0 | 0.767 |  |
| | note: thin: one designed trailer riser (forge); the bench is two real cymbal swells, no synth risers. | |  |  |  |  |  |  |  |  |  |  |  |
| shimmer | made | 6 | 1.07 | 10.68 | 0.35 | 4.06 | 0.97 | 3.46 | 0.91 | 0.67 | 0 | 0.845 | 7 / 7 (real 6) |
|  | real x | 5 | 1.69 | 9.66 | 0.73 | 8.58 | 0.73 | 2.87 | 0.03 | 0.6 | 2 | 0.59 |  |
|  | real loo | 4 | 1.01 | 15.52 | 1.33 | 12.89 | 1.05 | 3.4 | 0.56 | 1 | 0 | 0.644 |  |
| | note: kit shimmer_01 (bar chimes sweep) is bench: its author also recorded the celesta notify bench. | |  |  |  |  |  |  |  |  |  |  |  |
| sub_drop | made | 5 |  | 8.84 | 1.76 | 37.38 | 0.31 | 3.31 | 0.7 | 0 | 0 | 0.727 |  |
| | note: one designed source (a tom pitched down 24 semitones by its author), held out as bench; the forge has no sub-drop refs. | |  |  |  |  |  |  |  |  |  |  |  |
| swipe | made | 5 | -14.03 | 6.06 | 0.23 | 9.81 | 0.99 | 1.44 | 0.22 | 0.6 | 0 | 0.885 | 4 / 7 (real 6) |
|  | real x | 4 | -10.07 | 16.83 | 0.43 | 14.12 | 0.79 | 2.7 | 0.59 | 1 | 0 | 0.845 |  |
|  | real loo | 4 | 1.25 | 3.64 | 0.22 | 10.07 | 1.02 | 3.42 | 2.08 | 0.25 | 0 | 0.857 |  |
| | note: cloth only: the library has no paper slides. | |  |  |  |  |  |  |  |  |  |  |  |
| swish | made | 5 | -3.11 | 7.37 | 0.25 | 10.49 | 1.04 | 2.71 | 1.4 | 0 | 0 | 0.901 | 4 / 7 (real 6) |
|  | real x | 7 | 1.94 | 11.08 | 0.64 | 12.88 | 0.64 | 3.22 | 1.6 | 0.57 | 0 | 0.811 |  |
|  | real loo | 3 |  | 9.01 | 0.91 | 16.45 | 1.16 | 3.5 | 1.92 | 0.67 | 0 | 0.79 |  |
| tap | made | 5 | -1.61 | 8.23 | 1.4 | 14.51 | 1.82 | 1.13 | 0.34 | 0.6 | 0 | 0.807 | 5 / 7 (real 6) |
|  | real x | 7 | -1.17 | 9.35 | 1.59 | 13.67 | 1.64 | 1.85 | 0.92 | 0.14 | 0 | 0.713 |  |
|  | real loo | 6 | 1.66 | 16.78 | 1.19 | 11.59 | 1 | 1.62 | 0.35 | 0.17 | 0 | 0.716 |  |
| toggle | made | 5 | -3.36 | 10.36 | 0.62 | 14.64 | 1.7 | 1.45 | -1.35 | 0 | 0 | 0.718 | 5 / 7 (real 6) |
|  | real x | 4 | -2.84 | 12.48 | 1.87 | 18.78 | 2.11 | 2.1 | -0.02 | 0 | 0 | 0.477 |  |
|  | real loo | 4 | 1.54 | 6.25 | 0.28 | 4.92 | 1.02 | 1.94 | -0.28 | 0 | 0 | 0.725 |  |
| typing_key | made | 6 | -0.32 | 7.44 | 1.04 | 9.08 | 1 | 3.74 | 1.45 | 0 | 0 | 0.713 | 4 / 7 (real 6) |
|  | real x | 15 | 0.26 | 8.88 | 1.01 | 12.2 | 1.29 | 4.75 | 2.83 | 0.53 | 0 | 0.721 |  |
|  | real loo | 12 | 0.23 | 5.47 | 1.52 | 16.7 | 1 | 3.71 | 1.53 | 0.42 | 0 | 0.673 |  |
| whoosh_big | made | 6 | 0.77 | 6.67 | 0.44 | 8.97 | 0.97 | 2.38 | 0.03 | 0.83 | 0 | 0.854 | 7 / 7 (real 5) |
|  | real x | 3 | -0.27 | 8.8 | 1.13 | 19.46 | 3 | 2.92 | 0.04 | 1 | 0 | 0.7 |  |
|  | real loo | 3 |  | 4.25 | 0.43 | 6.67 | 0.97 | 1.88 | 0.19 | 0.33 | 0 | 0.495 |  |
| | note: forge = three designed Mixkit whooshes; bench = real air / cape swoops. | |  |  |  |  |  |  |  |  |  |  |  |
| whoosh_fast | made | 6 | -5.43 | 5.79 | 0.51 | 12.27 | 0.9 | 3.49 | 2.56 | 0.17 | 0 | 0.886 | 4.5 / 7 (real 6.5) |
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
| impact | made | -69.45 | 0.71 | -1.21 |
|  | real x | -70.55 | 0.54 | -1.3 |
| logo_sting | made | -61.5 | 0.68 | 0.18 |
|  | real x | -52.1 | 0.59 | -0.85 |
|  | real loo | -44.1 | 0.81 | -1.43 |
| notify | made | -44.75 | 0.98 | -2.0 |
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
| pro_impact | made | -35.8 | 0.38 | -0.25 |
| pro_logo_sting | made | -60.2 | 1.0 | -0.43 |
|  | real x | -120.0 | 1.0 | -0.38 |
| pro_notify | made | -47.7 | 0.94 | 0.39 |
| pro_pop | made | -69.5 | 1.0 | -1.45 |
|  | real x | -120.0 | 1.0 |  |
| pro_pop_in | made | -61.5 | 0.58 | -0.24 |
| pro_riser | made | -32.5 | 0.82 |  |
|  | real x | -41.8 | 0.96 | 0.22 |
| pro_shimmer | made | -49.0 | 0.33 | -0.53 |
| pro_sub_drop | made | -38.3 | 1.0 | 0.13 |
| pro_tap | made | -80.8 | 0.44 | -0.76 |
| pro_typing_key | made | -38.95 | 1.0 | 0.02 |
|  | real x | -53.55 | 1.0 | -0.04 |
| pro_typing_run | made | -57.95 | 1.0 |  |
| pro_whoosh_big | made | -55.05 | 1.0 | -1.31 |
| pro_zip | made | -52.9 | 0.99 | -1.67 |
|  | real loo | -21.95 | 0.93 | 0.24 |
| riser | made | -35.8 | 0.6 | -0.03 |
|  | real x | -49.9 | 0.0 |  |
|  | real loo | -30.65 | 1.0 | 0.12 |
| shimmer | made | -57.5 | 0.07 | -0.64 |
|  | real x | -46.8 | 0.17 | -0.29 |
|  | real loo | -36.1 | 0.68 | -0.33 |
| sub_drop | made | -67.0 | 1.0 | -0.59 |
| swipe | made | -65.2 | 0.9 | -1.96 |
|  | real x | -56.85 | 0.42 | 0.07 |
|  | real loo | -38.05 | 0.68 | 0.61 |
| swish | made | -61.5 | 0.94 | -0.73 |
|  | real x | -55.6 | 0.87 | -0.16 |
|  | real loo | -44.6 | 1.0 | -0.45 |
| tap | made | -45.0 | 0.93 | 0.0 |
|  | real x | -43.6 | 1.0 | 1.31 |
|  | real loo | -38.4 | 1.0 | 0.94 |
| toggle | made | -59.4 | 0.99 | 0.45 |
|  | real x | -48.4 | 1.0 | -0.11 |
|  | real loo | -49.1 | 0.48 | -0.18 |
| typing_key | made | -51.8 | 0.96 | -0.43 |
|  | real x | -43.7 | 0.76 | 0.04 |
|  | real loo | -39.65 | 1.0 | 0.09 |
| whoosh_big | made | -59.6 | 0.93 | -1.2 |
|  | real x | -43.2 | 0.66 | -0.11 |
|  | real loo | -42.9 | 1.0 | 0.28 |
| whoosh_fast | made | -63.6 | 0.96 | -0.64 |
|  | real x | -65.4 | 0.81 | -0.16 |
|  | real loo | -32.2 | 1.0 | -0.2 |

Diversity among the made variants (CLAP cosine mean / max, tonal distance min dB, near duplicates):

- click: cos 0.723 / 0.837, tone min 7.41 dB, 0 near duplicate pair(s) (bench refs: cos 0.576 / 0.792, tone min 6.13 dB)
- downlifter: cos 0.749 / 0.955, tone min 3.12 dB, 0 near duplicate pair(s)
- glitch: cos 0.728 / 0.871, tone min 1.8 dB, 0 near duplicate pair(s)
- hover_tick: cos 0.807 / 0.923, tone min 3.74 dB, 0 near duplicate pair(s) (bench refs: cos 0.626 / 0.847, tone min 9.21 dB)
- impact: cos 0.905 / 0.967, tone min 2.22 dB, 0 near duplicate pair(s)
- logo_sting: cos 0.722 / 0.876, tone min 10.32 dB, 0 near duplicate pair(s) (bench refs: cos 0.507 / 0.507, tone min 23.22 dB)
- notify: cos 0.864 / 0.957, tone min 5.63 dB, 0 near duplicate pair(s) (bench refs: cos 0.476 / 0.905, tone min 11.33 dB)
- pop: cos 0.735 / 0.877, tone min 5.92 dB, 0 near duplicate pair(s) (bench refs: cos 0.522 / 0.832, tone min 7.62 dB)
- pro_click: cos 0.786 / 0.902, tone min 6.93 dB, 0 near duplicate pair(s) (bench refs: cos 0.45 / 0.45, tone min 18.53 dB)
- pro_glitch: cos 0.853 / 0.912, tone min 1.35 dB, 0 near duplicate pair(s)
- pro_hover_tick: cos 0.862 / 0.942, tone min 1.79 dB, 0 near duplicate pair(s) (bench refs: cos 0.684 / 0.776, tone min 6.36 dB)
- pro_impact: cos 0.745 / 0.871, tone min 3.72 dB, 0 near duplicate pair(s)
- pro_logo_sting: cos 0.943 / 0.976, tone min 1.26 dB, 0 near duplicate pair(s)
- pro_notify: cos 0.822 / 0.902, tone min 6.02 dB, 0 near duplicate pair(s)
- pro_pop: cos 0.686 / 0.876, tone min 7.51 dB, 0 near duplicate pair(s)
- pro_pop_in: cos 0.75 / 0.823, tone min 5.62 dB, 0 near duplicate pair(s)
- pro_riser: cos 0.712 / 0.808, tone min 2.42 dB, 0 near duplicate pair(s)
- pro_shimmer: cos 0.646 / 0.808, tone min 7.57 dB, 0 near duplicate pair(s)
- pro_sub_drop: cos 0.835 / 0.885, tone min 4.46 dB, 0 near duplicate pair(s)
- pro_tap: cos 0.872 / 0.95, tone min 5.16 dB, 0 near duplicate pair(s)
- pro_typing_key: cos 0.815 / 0.952, tone min 1.1 dB, 0 near duplicate pair(s)
- pro_typing_run: cos 0.813 / 0.951, tone min 1.36 dB, 0 near duplicate pair(s)
- pro_whoosh_big: cos 0.882 / 0.966, tone min 3.24 dB, 0 near duplicate pair(s)
- pro_zip: cos 0.761 / 0.922, tone min 3.83 dB, 0 near duplicate pair(s) (bench refs: cos 0.529 / 0.686, tone min 8.63 dB)
- riser: cos 0.82 / 0.88, tone min 4.7 dB, 0 near duplicate pair(s) (bench refs: cos 0.767 / 0.767, tone min 13.22 dB)
- shimmer: cos 0.845 / 0.938, tone min 6.82 dB, 0 near duplicate pair(s) (bench refs: cos 0.644 / 0.765, tone min 10.66 dB)
- sub_drop: cos 0.727 / 0.889, tone min 4.89 dB, 0 near duplicate pair(s)
- swipe: cos 0.885 / 0.949, tone min 2.29 dB, 0 near duplicate pair(s) (bench refs: cos 0.857 / 0.918, tone min 2.76 dB)
- swish: cos 0.901 / 0.957, tone min 4.36 dB, 0 near duplicate pair(s) (bench refs: cos 0.79 / 0.908, tone min 9.83 dB)
- tap: cos 0.807 / 0.923, tone min 4.53 dB, 0 near duplicate pair(s) (bench refs: cos 0.716 / 0.98, tone min 2.68 dB)
- toggle: cos 0.718 / 0.849, tone min 3.98 dB, 0 near duplicate pair(s) (bench refs: cos 0.725 / 0.846, tone min 5.33 dB)
- typing_key: cos 0.713 / 0.885, tone min 6.37 dB, 0 near duplicate pair(s) (bench refs: cos 0.673 / 0.919, tone min 4.64 dB)
- whoosh_big: cos 0.854 / 0.914, tone min 2.86 dB, 0 near duplicate pair(s) (bench refs: cos 0.495 / 0.768, tone min 3.17 dB)
- whoosh_fast: cos 0.886 / 0.97, tone min 3.28 dB, 0 near duplicate pair(s) (bench refs: cos 0.772 / 0.951, tone min 2.28 dB)

## Judge notes (round 3, integrator, 11:00-11:30)

Made = `out/forge_v2/<category>/final/` as delivered after judging (183 sounds, 34 categories). Run: one full benchmark of
the fixers' deliveries at 11:03 (kept as `out/forge_v2/_judge/results_v2r3_asdelivered.json`, not committed), then the
same run again at 11:15 after the judge put the round-2 finals back for two categories (this file; the other 32
categories are byte-identical between the two runs and give the same numbers). Per-area fixers had each run the bench
once at the end of their own work (the ui area 4 times, see its report); the judge read bench numbers only to decide
r2 vs r3, never to tune. Model-free columns (band shares, onset brightness, t40 ms, stereo corr) are the judge's own
measurements (`out/forge_v2/_judge/judge_r3.py`, sheets `out/forge_v2/_judge/r3/<category>.png`: r2 / r3 / forge-ref
spectrograms, envelopes, 1/3-octave profiles with the bench median as a dashed line; not committed). Nobody has
listened yet.

### Round 2 -> round 3 (made medians on the held-out bench; v2r2 vs v2r3)

real = the same count for real recordings; real x = forge refs vs bench; loo = bench vs bench; var cos lower = more
varied, (bench) = the bench refs' own; AS names = share of sounds whose AudioSet top 3 names the category. Named fault =
the round-2 judge's fault (or the fixer's target) measured model-free, r2 -> r3 (reference median); band shares are dB
of total energy, onset >6 kHz = share of the first 15 ms.

| category | n | inside (real) | tone dB (real x) | t40 x (real x / loo) | var cos (bench) | phrase z | AS names | named fault, model-free (r2 -> r3, ref) | artifacts |
|---|---|---|---|---|---|---|---|---|---|
| click | 5 -> 5 | 5 -> 7 /7 (6) | 9.86 -> 8.79 (11.33) | 1.22 -> 1.44 (1.33 / 0.86) | 0.681 -> 0.723 (0.576) | 2.27 -> 3.03 | 0 -> 0 | 1-4 kHz share -14 -> -5.3 (bench -4.9) | 0 -> 0 |
| downlifter | 6 -> 6 | - | - | - | 0.749 -> 0.749 (-) | 3.23 -> 3.23 | 0.33 -> 0.33 | - | 0 -> 0 |
| glitch | 4 -> 6 | - | - | - | 0.799 -> 0.728 (-) | 2.14 -> 2.03 | 0.50 -> 0 | - | 0 -> 0 |
| hover_tick | 5 -> 5 | 6 -> 6 /7 (5) | 20.99 -> 20.99 (21.20) | 1.10 -> 1.10 (0.90 / 1.20) | 0.807 -> 0.807 (0.626) | 2.41 -> 2.41 | 0 -> 0 | - | 0 -> 0 |
| impact | 5 -> 6 | 5 -> 6 /6 (3) | 6.29 -> 5.31 (6.19) | 0.62 -> 0.56 (0.46 / -) | 0.881 -> 0.905 (-) | 3.61 -> 3.34 | 1 -> 1 | onset >6 kHz (15 ms) -28.9 -> -10.9 (bench -21.6) | 1 -> 0 |
| logo_sting | 4 -> 5 | 3.5 -> 6 /6 (5.5) | 16.29 -> 16.35 (15.52) | 2.48 -> 1.03 (1.70 / 1.12) | 0.733 -> 0.722 (0.507) | 1.55 -> 1.32 | 1 -> 1 | t40 ms 2135 -> 885 (bench 862.5) | 1 -> 0 |
| notify | 5 -> 6 | 7 -> 7 /7 (5.5) | 18.65 -> 18.29 (21.30) | 0.63 -> 0.85 (0.34 / 1.19) | 0.810 -> 0.864 (0.476) | 2.21 -> 1.40 | 1 -> 1 | 150 Hz-1 kHz -41.3 -> -1 (bench 0) | 0 -> 0 |
| pop | 5 -> 5 | 7 -> 7 /7 (6) | 11.42 -> 11.42 (16.06) | 0.82 -> 0.82 (0.91 / 1.02) | 0.735 -> 0.735 (0.522) | 2.32 -> 2.32 | 0.40 -> 0.40 | 1-4 kHz -17.2 -> -17.2 (bench -2) | 0 -> 0 |
| pro_click | 5 -> 5 | 4 -> 5 /6 (4) | 25.79 -> 15.43 (29.35) | 0.67 -> 0.80 (0.53 / 1.08) | 0.870 -> 0.786 (0.450) | 4.06 -> 3.91 | 0 -> 0 | 150 Hz-1 kHz -41.2 -> -6.8 (bench -6.2) | 0 -> 0 |
| pro_glitch | 5 -> 6 | - | - | - | 0.875 -> 0.853 (-) | 3.02 -> 2.53 | 0.60 -> 0 | <150 Hz -39.9 -> -21.9 (forge -21.6) | 0 -> 0 |
| pro_hover_tick | 5 -> 5 | 4 -> 6 /7 (6) | 7.89 -> 6.25 (6.51) | 1.17 -> 1 (1 / 0.86) | 0.878 -> 0.862 (0.684) | 0.85 -> 1.61 | 0 -> 0 | stereo corr 1 -> 0.3 (bench 0.2) | 0 -> 0 |
| pro_impact | 5 -> 5 | - | - | - | 0.733 -> 0.745 (-) | 2.71 -> 2.55 | 0.80 -> 0.60 | 4-12 kHz -30.4 -> -17.7 (forge -15.4) | 0 -> 0 |
| pro_logo_sting | 4 -> 5 | 3 -> 3 /6 (6) | 24.42 -> 24.43 (15.49) | 0.15 -> 0.13 (0.06 / -) | 0.947 -> 0.943 (-) | 0.73 -> 0.83 | 0 -> 0.20 | 1-4 kHz -13.3 -> -11 (bench -9.2) | 0 -> 0 |
| pro_notify | 5 -> 5 | - | - | - | 0.882 -> 0.822 (-) | 1.71 -> 1.91 | 1 -> 1 | >12 kHz -29.6 -> -19.6 (forge -17.3) | 0 -> 0 |
| pro_pop | 5 -> 6 | 4 -> 3 /6 (3) | 23.68 -> 23.63 (18.43) | 0.81 -> 0.75 (0.06 / -) | 0.663 -> 0.686 (-) | 3.29 -> 3.32 | 0 -> 0 | 1-4 kHz -3.1 -> -1 (bench -2.4) | 0 -> 0 |
| pro_pop_in | 5 -> 5 | - | - | - | 0.895 -> 0.750 (-) | 0.62 -> 0.93 | 0 -> 0 | 1-4 kHz -29.6 -> -24.7 (forge -27.5) | 0 -> 0 |
| pro_riser | 5 -> 5 | 3 -> 4 /6 (6) | 20.80 -> 17.88 (21.35) | 0.05 -> 0.05 (0.68 / -) | 0.676 -> 0.712 (-) | 2 -> 2.71 | 0 -> 0.20 | <150 Hz -9.4 -> -24.1 (bench -42.5) | 0 -> 0 |
| pro_shimmer | 5 -> 5 | - | - | - | 0.646 -> 0.646 (-) | 3.29 -> 3.29 | 0.20 -> 0.20 | - | 0 -> 0 |
| pro_sub_drop | 5 -> 5 | - | 24.84 -> 24.84 (-) | 1.64 -> 1.64 (- / -) | 0.835 -> 0.835 (-) | 2.44 -> 2.44 | 0 -> 0 | - | 0 -> 0 |
| pro_tap | 5 -> 5 | - | - | - | 0.854 -> 0.872 (-) | 1.73 -> 1.65 | 0 -> 0.20 | >12 kHz -57.8 -> -47.9 (forge -36.6) | 0 -> 0 |
| pro_typing_key | 5 -> 6 | 4 -> 4 /6 (5.5) | 26.38 -> 26.09 (26.20) | 0.64 -> 0.79 (0.71 / -) | 0.926 -> 0.815 (-) | 1.84 -> 1.60 | 0 -> 0 | - | 0 -> 0 |
| pro_typing_run | 5 -> 6 | - | - | - | 0.930 -> 0.813 (-) | 2 -> 1.79 | 0 -> 0 | - | 1 -> 0 |
| pro_whoosh_big | 5 -> 6 | - | - | - | 0.931 -> 0.882 (-) | 3.93 -> 3.71 | 1 -> 0.83 | 1-4 kHz -46.5 -> -34.8 (forge -38); 4-12 kHz -31.7 -> -41.7 (forge -38.5) | 0 -> 0 |
| pro_zip | 5 -> 5 | 5 -> 5 /7 (5) | 11.89 -> 11.89 (-) | 0.84 -> 0.84 (- / 1.15) | 0.761 -> 0.761 (0.529) | 2.34 -> 2.34 | 0 -> 0 | 4-12 kHz -13.1 -> -13.1 (bench -16.1) | 0 -> 0 |
| riser | 5 -> 5 | 3 -> 3 /6 (4) | 12.50 -> 11.30 (17.61) | 0.06 -> 0.09 (0.50 / 3.35) | 0.780 -> 0.820 (0.767) | 2.57 -> 2.51 | 0.80 -> 0.40 | <150 Hz -16.2 -> -19.6 (bench -28.1) | 0 -> 0 |
| shimmer | 5 -> 6 | 6 -> 7 /7 (6) | 10.49 -> 10.68 (9.66) | 0.48 -> 0.97 (0.73 / 1.05) | 0.807 -> 0.845 (0.644) | 3.42 -> 3.46 | 0.40 -> 0.67 | t40 ms 1155 -> 2340 (bench 2420) | 0 -> 0 |
| sub_drop | 5 -> 5 | - | 8.84 -> 8.84 (-) | 0.31 -> 0.31 (- / -) | 0.727 -> 0.727 (-) | 3.31 -> 3.31 | 0 -> 0 | - | 0 -> 0 |
| swipe | 5 -> 5 | 4 -> 4 /7 (6) | 13.47 -> 6.06 (16.83) | 0.86 -> 0.99 (0.79 / 1.02) | 0.855 -> 0.885 (0.857) | 1.82 -> 1.44 | 0.40 -> 0.60 | 4-12 kHz share -35.9 -> -20.7 (bench -19.2); onset >6 kHz -27.4 -> -22.4 (bench -11.6) | 0 -> 0 |
| swish | 5 -> 5 | 4 -> 4 /7 (6) | 7.48 -> 7.37 (11.08) | 0.68 -> 1.04 (0.64 / 1.16) | 0.860 -> 0.901 (0.790) | 2.68 -> 2.71 | 0 -> 0 | t40 ms 150 -> 230 (bench 220); <150 Hz -15.8 -> -23.4 (bench -17.4); 4-12 kHz -18.2 -> -29.1 (bench -33.5) | 0 -> 0 |
| tap | 5 -> 5 | 3 -> 5 /7 (6) | 6.10 -> 8.23 (9.35) | 1.27 -> 1.82 (1.64 / 1) | 0.856 -> 0.807 (0.716) | 0.52 -> 1.13 | 0 -> 0.60 | 4-12 kHz -27.5 -> -18.4 (bench -26.4) | 0 -> 0 |
| toggle | 5 -> 5 | 5 -> 5 /7 (6) | 8.82 -> 10.36 (12.48) | 1.26 -> 1.70 (2.11 / 1.02) | 0.720 -> 0.718 (0.725) | 1.39 -> 1.45 | 0 -> 0 | 150 Hz-1 kHz -5.4 -> -9.8 (bench -19.2) | 0 -> 0 |
| typing_key | 5 -> 6 | 6 -> 4 /7 (6) | 8.15 -> 7.44 (8.88) | 1.29 -> 1 (1.29 / 1.00) | 0.705 -> 0.713 (0.673) | 4.46 -> 3.74 | 0.20 -> 0 | stereo corr 0.3 -> 1 (bench 1); <150 Hz -5.5 -> -10.2 (bench -10.9) | 0 -> 0 |
| whoosh_big | 5 -> 6 | 7 -> 7 /7 (5) | 7.58 -> 6.67 (8.80) | 1.04 -> 0.97 (3 / 0.97) | 0.909 -> 0.854 (0.495) | 2.58 -> 2.38 | 1 -> 0.83 | - | 0 -> 0 |
| whoosh_fast | 5 -> 6 | 4 -> 4.5 /7 (6.5) | 9.43 -> 5.79 (10.31) | 0.49 -> 0.90 (0.60 / 1.08) | 0.856 -> 0.886 (0.772) | 3.28 -> 3.49 | 0.60 -> 0.17 | t40 ms 130 -> 237.5 (bench 265); <150 Hz -9.7 -> -14.4 (bench -17.7); 4-12 kHz -17.4 -> -26.4 (bench -32.3) | 0 -> 0 |

Round-3 attempts the judge did NOT deliver (kept in `<category>/final_r3/`, round-2 finals back in `final/`):

| category | n | inside | tone dB | t40 x | env oct | curve dB | phrase z | var cos | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| pop | 5 | 7 /7 (r2 7) | 12.38 (r2 11.42) | 0.55 (r2 0.82; real x 0.91) | 0.66 (0.55) | 11.43 (11.01) | 1.83 (2.32) | 0.665 (0.735) | worse on every single-sound check that moved; the snap adds 4-12 kHz (-25.7 dB vs bench -35.5) and shortens the tail |
| pro_zip | 6 | 4.5 /7 (r2 5) | 12.72 (r2 11.89) | 0.63 (r2 0.84; loo 1.15) | 0.97 (0.71) | 22.95 (20.81) | 1.52 (2.34) | 0.768 (0.761) | worse on all four model-free checks; its one gain (margin -1.39 -> -0.74) came after tuning on an AudioSet reading |

### Verdicts per category (good for a motion-design launch video?)

Fixed as asked (the named fault measurably moved to the reference, no new fault found):
- **swipe**: muffle fixed. 4-12 kHz -35.9 -> -20.7 dB (bench cloth -19.2), tone 13.47 -> 6.06 dB (real x 16.83), t40 x 0.99.
  Onset still 11 dB duller than the bench cloth (>6 kHz -22.4 vs -11.6); L/R corr 0.90 vs 0.70; AudioSet 'Door' on 1 of 5.
- **whoosh_fast / swish**: tails x1.8 (t40 130 -> 238 / 150 -> 230 ms; bench 265 / 220), boom down 5-8 dB, hiss down 9-11 dB;
  whoosh_fast tone 9.43 -> 5.79 dB (real x 10.31), curve 18.4 -> 12.3. Every sound is inside the real limits on the three
  model-free checks; the misses are CLAP / AudioSet (AudioSet still hears Door / Knock / Slam first on 6 of 6 and 5 of 5),
  and variety got worse (0.856 -> 0.886, 0.860 -> 0.901). Still under v1's 7/7 inside (v1 sat above the real rows on the
  saturated model checks, tuned on kit refs that are bench refs now).
- **logo_sting**: redefined as a composed reveal (lead-in -> strike -> one note or dyad -> short tail). t40 2135 -> 885 ms
  (bench 862), curve 25.2 -> 14.8 dB, inside 3.5 -> 6 of 6 (real 5.5), artifact gone, AudioSet 'Ding' first on 5 of 5. New
  weakness seen on the sheet: no body at all (<150 Hz -52.7 dB of total vs bench -26.8; nothing under ~800 Hz).
- **typing_key**: width fixed (stereo corr 0.34 -> 0.96, bench 1.00), thock down (<150 Hz -5.5 -> -10.2, bench -10.9), tone
  7.44, t40 x 1.00, curve 9.1; the model checks fell (inside 6 -> 4 of 7, phrase z 4.46 -> 3.74): CLAP hears less low end,
  not a fault on the trust order.
- **click**: 1-4 kHz -14.0 -> -5.3 dB (bench -4.9), late partial line 18.7 -> 15.0 dB (refs 15.1, fixer's measure), inside
  5 -> 7 of 7, phrase z 2.27 -> 3.03. Variety slipped 0.681 -> 0.723 (bench 0.576).
- **shimmer**: rings out freely: t40 1155 -> 2340 ms (bench 2420), curve 14.1 -> 4.1 dB, inside 7 of 7, AudioSet 0.4 -> 0.67.
- **pro_hover_tick**: spike + grains: phrase z 0.85 -> 1.61 (loo 1.79), margin -0.54 -> 0.45, tone 6.25 (real x 6.51), inside 6 of 7.
- **pro_click**: bodies added: 150 Hz-1 kHz -41.2 -> -6.8 dB (bench -6.2), tone 25.8 -> 15.4 dB, inside 5 of 6, var 0.786.
- **pro_glitch**: burst drama: median 2 ms level re peak -22 -> -36 dB (judge's own measure; forge film -33), the stutter
  speeds up in 6 of 6, <150 Hz -21.9 (film -21.6). AudioSet no longer says static (0.6 -> 0), phrase z 3.02 -> 2.53.
- **pro_typing_key / pro_typing_run / pro_pop_in**: clones fixed: var cos 0.926 -> 0.815, 0.930 -> 0.813, 0.895 -> 0.750 (film
  sets 0.70-0.85); the run's tail-cut flag is gone. pro_typing_key 06 'crackle' is outside the study's envelope box.
- **whoosh_big / pro_whoosh_big**: variety 0.909 -> 0.854 and 0.931 -> 0.882; whoosh_big still inside 7 of 7 (real 5), tone
  7.58 -> 6.67. Costs: whoosh_big curve 5.8 -> 9.0 dB, AudioSet names 5 of 5 -> 5 of 6. pro_whoosh_big's air layer is fused
  (1-4 kHz -46.5 -> -34.8, film -38.0).
- **pro_riser**: tone 20.8 -> 17.9 dB, phrase z 2.00 -> 2.71, margin -0.78 -> 0.47, inside 3 -> 4 of 6; the gaps dip 10-20 dB
  where the film's drop 40; AudioSet hears 'Fart' / 'Machine gun' on two members.

Better with a cost (delivered, the cost noted):
- **impact**: onset fixed and then overshot: >6 kHz share of the first 15 ms -28.9 -> -10.9 dB (bench -21.6; the >12 kHz share
  of that window is -10..-14 dB vs bench -28, forge -33..-45), so the contact now carries a bright tick; after 100 ms the air
  is back at the r2 / bench level. Tone 6.29 -> 5.31 dB (better than real x 6.19), inside 5 -> 6 of 6, late-start flag gone;
  variety worse 0.881 -> 0.905. Next step: the sheen 8-10 dB down.
- **toggle**: envelope fixed (env oct 2.58 -> 0.62, curve 25.4 -> 14.6: the first contact is now the loudest), but tone 8.82
  -> 10.36 dB, t40 x 1.26 -> 1.70 and margin -0.32 -> -1.35 got worse; toggle_01 / 03 show narrow ringing lines at 3-10 kHz.
- **tap**: reads more like a tap (phrase z 0.52 -> 1.13, AudioSet 0 -> 0.6, inside 3 -> 5 of 7) but is 8 dB brighter at 4-12 kHz
  than the bench (-18.4 vs -26.4), tone 6.10 -> 8.23, tails long (t40 x 1.82).
- **notify**: lower, longer bells (150 Hz-1 kHz -41 -> -1 dB, bench 0; t40 x 0.63 -> 0.85; curve 21.8 -> 13.6), but phrase z
  2.21 -> 1.40 and AudioSet 'Clang' first on 4 of 6 (r2 'Ding'); variety 0.81 -> 0.86.
- **pro_impact**: brighter grains toward the film (4-12 kHz -30.4 -> -17.7, film -15.4), straight dB fall before the gate; the
  grains are evenly spaced stripes (as in one of the film refs) and margin fell 0.60 -> -0.07.
- **pro_notify**: cleaner head, breath and >12 kHz toward the films (-29.6 -> -19.6, films -17.3), var 0.882 -> 0.822; still far
  under the films on today's critic (-5.97 vs +0.1..3.0, fixer's rescore).
- **riser**: tone 12.5 -> 11.3, curve 24.8 -> 21.7, longest steady line 0.22 -> 0.11 s (fixer); AudioSet names 0.8 -> 0.4.
- **glitch**: 6 members, variety 0.799 -> 0.728; AudioSet names 0.5 -> 0, margin 0.08 -> -0.13 (no refs: designed mode only).
- **pro_logo_sting**: more weight (1-4 kHz -13.3 -> -11.0) but still near clones (0.943) and band-limited: nothing above ~5 kHz
  (4-12 kHz -41.8 dB of total vs the film -21.0).
- **pro_pop / pro_tap**: roughly flat (pro_pop inside 4 -> 3 of 6 = real 3; pro_tap phrase z 1.73 -> 1.65, variety 0.854 -> 0.872).

Round 2 kept: **pop**, **pro_zip** (above). Unchanged in round 3: downlifter, hover_tick, sub_drop, pro_sub_drop, pro_shimmer.

Showcase-ready (measured, not yet listened): whoosh_big, pro_whoosh_big, shimmer, click, typing_key, pro_typing_key,
pro_typing_run, pro_pop_in, pro_hover_tick, pro_click, logo_sting, swipe, impact (mind the contact tick), whoosh_fast and
swish (tail and balance fixed; AudioSet still says door), pop (r2), pro_zip (r2), downlifter, sub_drop, pro_glitch,
pro_riser. Weakest now: pro_logo_sting (clones, no top end), pro_tap, pro_pop, toggle (ring, tone), tap (bright, long),
notify ('Clang'), riser (AudioSet), glitch (no refs), pro_notify (far under the films on the critic).

Critic passes 55 / 183 (r2 60 / 168), not comparable: the core critic changed during round 3 and the areas used
different critic setups (see each LOG). Tests: full fast suite 400 passed, 15 skipped (316 s).

Pages for the morning (private: they play licensed forge refs): `out/showcase/gallery.html` (round-3 finals with these
numbers) and `out/showcase/compare.html` (r2 / r3 / r3x rows per category).
