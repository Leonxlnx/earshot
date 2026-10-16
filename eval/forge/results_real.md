# forge benchmark: real

made: -; refs: <EARSHOT_HOME>/refsets.json; 2026-10-16 04:07

Medians per set. made = the made sounds vs the bench refs; real x = the forge refs (real, never in the bench) vs the bench refs; real loo = each bench ref vs the other bench refs. typ z, phrase z, margin, AS names: higher is better; tone dB, env oct (octaves off in attack / t40 / length), curve dB: lower is better; t40 x = tail length / bench median (1 = same). inside = checks (of n) a made sound lands inside the real 80% limit, median over the made sounds (real = the same count for the real rows). var cos = mean pairwise CLAP cosine within the set (made variants / forge refs / bench refs): high = the variants are near clones.

| category | set | n | typ z | tone dB | env oct | curve dB | t40 x | phrase z | margin | AS names | artifacts | var cos | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| click | real x | 8 | 0.16 | 11.33 | 0.22 | 9.92 | 1.33 | 2.77 | 0.53 | 0.12 | 0 | 0.651 |  |
|  | real loo | 7 | -0.1 | 9.85 | 0.46 | 13.19 | 0.86 | 2.67 | 0.27 | 0 | 0 | 0.576 |  |
| downlifter | no rows | 0 |  |  |  |  |  |  |  |  |  |  |  |
| | note: empty: the licensed library has no downlifter, real or designed (no reversing or pitching of other sounds here: trim + fade only). Benchmark made downlifters on phrasings, AudioSet and artifacts only. | |  |  |  |  |  |  |  |  |  |  |  |
| glitch | no rows | 0 |  |  |  |  |  |  |  |  |  |  |  |
| | note: empty: no glitch material, real or designed, in the licensed library. Benchmark made glitches on phrasings, AudioSet and artifacts only. | |  |  |  |  |  |  |  |  |  |  |  |
| hover_tick | real x | 6 | 0.09 | 21.2 | 1.64 | 9.88 | 0.9 | 2.51 | 0.9 | 0 | 0 | 0.799 |  |
|  | real loo | 6 | 0.47 | 15.46 | 1.16 | 19.39 | 1.2 | 2 | 0.58 | 0 | 4 | 0.626 |  |
| impact | real x | 2 |  | 6.19 | 2.5 | 31.98 | 0.46 | 3.04 | 1.09 | 0.5 | 0 | 0.659 |  |
| | note: thin: the library has three cinematic hits (all three were the v1 impact forge refs); one is held out as bench, so v1 impact bench numbers are optimistic. | |  |  |  |  |  |  |  |  |  |  |  |
| logo_sting | real x | 2 |  | 15.52 | 1.46 | 21.27 | 1.7 | 1.4 | -0.82 | 0.5 | 0 | 0.185 |  |
|  | real loo | 2 |  | 23.22 | 0.81 | 8.99 | 1.12 | 1.36 | -0.58 | 1 | 0 | 0.507 |  |
| | note: no real logo stings exist in the library (they are composed); the refs are the struck metal / glass rings a sting ends on: a tail-shape and tone reference only. | |  |  |  |  |  |  |  |  |  |  |  |
| notify | real x | 5 | -0.24 | 21.3 | 1.3 | 28.92 | 0.34 | 1.25 | -0.24 | 1 | 0 | 0.593 |  |
|  | real loo | 5 | 3.84 | 16.7 | 0.81 | 17.41 | 1.19 | 1.76 | -0.91 | 1 | 0 | 0.476 |  |
| | note: struck bells / glass / kalimba (forge) and celesta / glass / triangle (bench), rings capped at 2 s with a 0.25 s fade; no designed UI dings in the library. | |  |  |  |  |  |  |  |  |  |  |  |
| pop | real x | 7 | -0.09 | 16.06 | 1.51 | 20.54 | 0.91 | 2.46 | -0.5 | 0.29 | 0 | 0.501 |  |
|  | real loo | 6 | 1.13 | 13.66 | 0.41 | 12.5 | 1.02 | 1.96 | -1.06 | 0.67 | 0 | 0.522 |  |
| riser | real x | 1 |  | 17.61 | 1.01 | 20.04 | 0.5 | 3.52 | 1.06 | 1 | 0 |  |  |
|  | real loo | 2 |  | 13.22 | 0.95 | 36.07 | 3.35 | 2.7 | 1.15 | 0.5 | 0 | 0.767 |  |
| | note: thin: one designed trailer riser (forge); the bench is two real cymbal swells, no synth risers. | |  |  |  |  |  |  |  |  |  |  |  |
| shimmer | real x | 5 | 1.69 | 9.66 | 0.73 | 8.58 | 0.73 | 2.87 | 0.03 | 0.6 | 2 | 0.59 |  |
|  | real loo | 4 | 1.01 | 15.52 | 1.33 | 12.89 | 1.05 | 3.4 | 0.56 | 1 | 0 | 0.644 |  |
| | note: kit shimmer_01 (bar chimes sweep) is bench: its author also recorded the celesta notify bench. | |  |  |  |  |  |  |  |  |  |  |  |
| sub_drop | no rows | 0 |  |  |  |  |  |  |  |  |  |  |  |
| | note: one designed source (a tom pitched down 24 semitones by its author), held out as bench; the forge has no sub-drop refs. | |  |  |  |  |  |  |  |  |  |  |  |
| swipe | real x | 4 | -10.07 | 16.83 | 0.43 | 14.12 | 0.79 | 2.7 | 0.59 | 1 | 0 | 0.845 |  |
|  | real loo | 4 | 1.25 | 3.64 | 0.22 | 10.07 | 1.02 | 3.42 | 2.08 | 0.25 | 0 | 0.857 |  |
| | note: cloth only: the library has no paper slides. | |  |  |  |  |  |  |  |  |  |  |  |
| swish | real x | 7 | 1.94 | 11.08 | 0.64 | 12.88 | 0.64 | 3.22 | 1.6 | 0.57 | 0 | 0.811 |  |
|  | real loo | 3 |  | 9.01 | 0.91 | 16.45 | 1.16 | 3.5 | 1.92 | 0.67 | 0 | 0.79 |  |
| tap | real x | 7 | -1.17 | 9.35 | 1.59 | 13.67 | 1.64 | 1.85 | 0.92 | 0.14 | 0 | 0.713 |  |
|  | real loo | 6 | 1.66 | 16.78 | 1.19 | 11.59 | 1 | 1.62 | 0.35 | 0.17 | 0 | 0.716 |  |
| toggle | real x | 4 | -2.84 | 12.48 | 1.87 | 18.78 | 2.11 | 2.1 | -0.02 | 0 | 0 | 0.477 |  |
|  | real loo | 4 | 1.54 | 6.25 | 0.28 | 4.92 | 1.02 | 1.94 | -0.28 | 0 | 0 | 0.725 |  |
| typing_key | real x | 15 | 0.26 | 8.88 | 1.01 | 12.2 | 1.29 | 4.75 | 2.83 | 0.53 | 0 | 0.721 |  |
|  | real loo | 12 | 0.23 | 5.47 | 1.52 | 16.7 | 1 | 3.71 | 1.53 | 0.42 | 0 | 0.673 |  |
| whoosh_big | real x | 3 | -0.27 | 8.8 | 1.13 | 19.46 | 3 | 2.92 | 0.04 | 1 | 0 | 0.7 |  |
|  | real loo | 3 |  | 4.25 | 0.43 | 6.67 | 0.97 | 1.88 | 0.19 | 0.33 | 0 | 0.495 |  |
| | note: forge = three designed Mixkit whooshes; bench = real air / cape swoops. | |  |  |  |  |  |  |  |  |  |  |  |
| whoosh_fast | real x | 5 | 0.57 | 10.31 | 1.22 | 20.2 | 0.6 | 4.14 | 2.78 | 0.4 | 0 | 0.826 |  |
|  | real loo | 5 | 1.52 | 4.58 | 0.4 | 8.21 | 1.08 | 4.14 | 2.62 | 0.8 | 0 | 0.772 |  |

Texture (medians; model-free, not in any forge objective): floor dB = 10th percentile 5 ms frame re the loudest (-120 = digital silence), stereo corr (1 = mono), darken = log2 centroid at half the tail / at the peak (< 0: the decay darkens).

| category | set | floor dB | stereo corr | darken |
|---|---|---|---|---|
|  | real x | -42.85 | 0.99 | -0.12 |
|  | real loo | -44.8 | 1.0 | -0.07 |
|  | real x | -43.55 | 1.0 | -0.09 |
|  | real loo | -24.9 | 0.52 | -0.01 |
|  | real x | -70.55 | 0.54 | -1.3 |
|  | real x | -52.1 | 0.59 | -0.85 |
|  | real loo | -44.1 | 0.81 | -1.43 |
|  | real x | -41.2 | 0.99 | -1.06 |
|  | real loo | -24.4 | 0.61 | -0.53 |
|  | real x | -53.2 | 1.0 | 0.21 |
|  | real loo | -47.2 | 1.0 | 0.12 |
|  | real x | -49.9 | 0.0 |  |
|  | real loo | -30.65 | 1.0 | 0.12 |
|  | real x | -46.8 | 0.17 | -0.29 |
|  | real loo | -36.1 | 0.68 | -0.33 |
|  | real x | -56.85 | 0.42 | 0.07 |
|  | real loo | -38.05 | 0.68 | 0.61 |
|  | real x | -55.6 | 0.87 | -0.16 |
|  | real loo | -44.6 | 1.0 | -0.45 |
|  | real x | -43.6 | 1.0 | 1.31 |
|  | real loo | -38.4 | 1.0 | 0.94 |
|  | real x | -48.4 | 1.0 | -0.11 |
|  | real loo | -49.1 | 0.48 | -0.18 |
|  | real x | -43.7 | 0.76 | 0.04 |
|  | real loo | -39.65 | 1.0 | 0.09 |
|  | real x | -43.2 | 0.66 | -0.11 |
|  | real loo | -42.9 | 1.0 | 0.28 |
|  | real x | -65.4 | 0.81 | -0.16 |
|  | real loo | -32.2 | 1.0 | -0.2 |

Diversity among the made variants (CLAP cosine mean / max, tonal distance min dB, near duplicates):

