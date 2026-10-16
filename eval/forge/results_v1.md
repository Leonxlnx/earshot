# forge benchmark: v1

made: ['typing_key=demo/forge-test/keypress/final/*.wav', 'click=demo/forge-test/click/final/*.wav', 'pop=demo/forge-test/pop/final/*.wav', 'whoosh_fast=demo/forge-test/whoosh/final/*.wav', 'impact=demo/forge-test/impact/final/*.wav']; refs: <EARSHOT_HOME>/refsets.json; 2026-10-16 04:11

Medians per set. made = the made sounds vs the bench refs; real x = the forge refs (real, never in the bench) vs the bench refs; real loo = each bench ref vs the other bench refs. typ z, phrase z, margin, AS names: higher is better; tone dB, env oct (octaves off in attack / t40 / length), curve dB: lower is better; t40 x = tail length / bench median (1 = same). inside = checks (of n) a made sound lands inside the real 80% limit, median over the made sounds (real = the same count for the real rows). var cos = mean pairwise CLAP cosine within the set (made variants / forge refs / bench refs): high = the variants are near clones.

| category | set | n | typ z | tone dB | env oct | curve dB | t40 x | phrase z | margin | AS names | artifacts | var cos | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| click | made | 3 | 1.89 | 7.6 | 0.56 | 8.85 | 0.56 | 3.12 | 1.13 | 0 | 0 | 0.886 | 7 / 7 (real 6) |
|  | real x | 8 | 0.16 | 11.33 | 0.22 | 9.92 | 1.33 | 2.77 | 0.53 | 0.12 | 0 | 0.651 |  |
|  | real loo | 7 | -0.1 | 9.85 | 0.46 | 13.19 | 0.86 | 2.67 | 0.27 | 0 | 0 | 0.576 |  |
| impact | made | 3 |  | 6.7 | 1.54 | 24.86 | 0.5 | 3.68 | 1.81 | 1 | 0 | 0.931 | 5 / 6 (real 3) |
|  | real x | 2 |  | 6.19 | 2.5 | 31.98 | 0.46 | 3.04 | 1.09 | 0.5 | 0 | 0.659 |  |
| | note: thin: the library has three cinematic hits (all three were the v1 impact forge refs); one is held out as bench, so v1 impact bench numbers are optimistic. | |  |  |  |  |  |  |  |  |  |  |  |
| pop | made | 3 | 1.33 | 8.59 | 0.68 | 6.29 | 1 | 3.09 | -0.1 | 1 | 0 | 0.917 | 7 / 7 (real 6) |
|  | real x | 7 | -0.09 | 16.06 | 1.51 | 20.54 | 0.91 | 2.46 | -0.5 | 0.29 | 0 | 0.501 |  |
|  | real loo | 6 | 1.13 | 13.66 | 0.41 | 12.5 | 1.02 | 1.96 | -1.06 | 0.67 | 0 | 0.522 |  |
| typing_key | made | 3 | 1.2 | 8.2 | 1.07 | 14.32 | 1.67 | 5.56 | 3.57 | 0.33 | 0 | 0.965 | 7 / 7 (real 6) |
|  | real x | 15 | 0.26 | 8.88 | 1.01 | 12.2 | 1.29 | 4.75 | 2.83 | 0.53 | 0 | 0.721 |  |
|  | real loo | 12 | 0.23 | 5.47 | 1.52 | 16.7 | 1 | 3.71 | 1.53 | 0.42 | 0 | 0.673 |  |
| whoosh_fast | made | 3 | 2.05 | 8.58 | 0.89 | 15.93 | 0.51 | 4.46 | 3.31 | 0.33 | 0 | 0.957 | 7 / 7 (real 6.5) |
|  | real x | 5 | 0.57 | 10.31 | 1.22 | 20.2 | 0.6 | 4.14 | 2.78 | 0.4 | 0 | 0.826 |  |
|  | real loo | 5 | 1.52 | 4.58 | 0.4 | 8.21 | 1.08 | 4.14 | 2.62 | 0.8 | 0 | 0.772 |  |

Texture (medians; model-free, not in any forge objective): floor dB = 10th percentile 5 ms frame re the loudest (-120 = digital silence), stereo corr (1 = mono), darken = log2 centroid at half the tail / at the peak (< 0: the decay darkens).

| category | set | floor dB | stereo corr | darken |
|---|---|---|---|---|
| click | made | -49.0 | 1.0 | 0.41 |
|  | real x | -42.85 | 0.99 | -0.12 |
|  | real loo | -44.8 | 1.0 | -0.07 |
| impact | made | -64.7 | 0.41 | -1.61 |
|  | real x | -70.55 | 0.54 | -1.3 |
| pop | made | -65.0 | 1.0 | 0.19 |
|  | real x | -53.2 | 1.0 | 0.21 |
|  | real loo | -47.2 | 1.0 | 0.12 |
| typing_key | made | -54.7 | 0.94 | 0.4 |
|  | real x | -43.7 | 0.76 | 0.04 |
|  | real loo | -39.65 | 1.0 | 0.09 |
| whoosh_fast | made | -43.9 | 0.9 | -0.32 |
|  | real x | -65.4 | 0.81 | -0.16 |
|  | real loo | -32.2 | 1.0 | -0.2 |

Diversity among the made variants (CLAP cosine mean / max, tonal distance min dB, near duplicates):

- click: cos 0.886 / 0.93, tone min 7.24 dB, 0 near duplicate pair(s) (bench refs: cos 0.576 / 0.792, tone min 6.13 dB)
- impact: cos 0.931 / 0.943, tone min 3.76 dB, 0 near duplicate pair(s)
- pop: cos 0.917 / 0.96, tone min 4.49 dB, 0 near duplicate pair(s) (bench refs: cos 0.522 / 0.832, tone min 7.62 dB)
- typing_key: cos 0.965 / 0.967, tone min 4.27 dB, 0 near duplicate pair(s) (bench refs: cos 0.673 / 0.919, tone min 4.64 dB)
- whoosh_fast: cos 0.957 / 0.973, tone min 4.11 dB, 0 near duplicate pair(s) (bench refs: cos 0.772 / 0.951, tone min 2.28 dB)
