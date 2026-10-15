# timeline + sheet: evaluation

All numbers below come from the commands listed with them (results in `eval/timeline/results.json` and
`eval/timeline/label_tune.json`; media and scratch output stay in `out/timeline/`). Machine: Ryzen 7 260 laptop shared
with other builders running models, CPU only, `EARSHOT_THREADS=2`. Second round (after review); the `timing93low` and
`probe` entries of results.json are from the first round.

**Tuning vs evaluation.** Every threshold and design choice was tuned on data that the Clawd soundtrack does not
contain: synthetic mixes of real recordings (`mixes.py`: unused / rejected files of the Clawd asset library, 8 mixes of
30 s, music and ambience beds with real one-shots on top at known times and levels), the library's unused music
(`label_tune.py`, `pulse_tune.py`, `tempo_tune.py`) and, for the tag background only, four launch films the evaluation
does not use (ref_3, ref_5, ref_6, ref_7). The Clawd film (soundtrack, stems, `mix_report.json`
placements, `cues_all.json`) was used only for the evaluation below. The ref_1, ref_2 and ref_4 films were used
only to check that the output reads sensibly. One caveat: `earshot/data/vocab_stats.json` (owned by semantics)
includes the Clawd soundtrack among 148 references; timeline does not use it.

## Event detection (Clawd soundtrack vs every placed sound effect)

`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py --parts detect`

147 unique placement onsets (attack / start aligned layers within 30 ms merged), a hit = a detection within 30 ms.
SNR = the placement's level over music + ambience at that moment (from the stems).

| | recall | n |
|---|---|---|
| all placements | **93.2%** | 147 |
| loud (SNR >= 0 dB) | 96.4% | 83 |
| quiet (SNR < 0 dB) | 89.1% | 64 |
| SNR >= +6 / 0..+6 / -6..0 / < -6 dB | 96.4% / 96.3% / 97.9% / 64.7% | 56 / 27 / 47 / 17 |
| the same detector on the sfx stem alone | 95.2% | 147 |

- 270 detections: **50.7%** are a placement onset, **81.1%** a placement or a music / ambience onset (found in the
  bed stems), 95.6% if the extra hits inside multi-hit effect samples also count (found by re-running the detector at
  looser settings on the sfx stem); 12 (4.4%) unexplained. (The first build's docstring quoted only the 96%.)
- Timing error vs the mixer's attack times: median 0.0 ms, median |error| 0.5 ms, 90th percentile |error| 11.0 ms.
- Peak-aligned whooshes: 9 of 14 have an onset or swell event within 120 ms of their peak.

Tuning (`.venv/Scripts/python.exe eval/timeline/tune.py`, synthetic mixes, 204 hits, first round): robust z 3.0 and
band contrast 3 dB give recall 80.7% (SNR >= 0: 97.9%), 95.4% of detections are hits or bed onsets.

## Event tags (CLAP, compact event vocabulary)

### Hub phrases (review finding) and the fix

The first build ranked phrases by z within each event. Phrases that score high on any short sound then win
everywhere: "a chat message pop" and "a spacebar press" labelled syllables, music notes and ambience.
`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/tag_bg.py` collects the events timeline would tag (its own
picking, plus every placed one-shot of the mixes) from the 8 tuning mixes and the 4 unused launch films: 536 events.
The per-phrase mean cosine of the fit half (mixes 0-3, ref_3, ref_5; 270 events) is subtracted before ranking,
scaled by TAG_CENTER; scored on the test half (mixes 4-7, ref_6, ref_7; 266 events: 69 placed one-shots,
92 bed events = music / ambience onsets >= 0.15 s from any one-shot, 80 film events):

| scoring | one-shot right kind | bed events called KEY / UI | film events called KEY / UI | most common phrase |
|---|---|---|---|---|
| z within the event (first build) | 42.0% | 60.9% | 27.5% | 26.7% "chat message pop" |
| minus 0.25 x background mean | 34.8% | 46.7% | 16.3% | 17.7% "chat message pop" |
| **minus 0.5 x background mean (shipped)** | 30.4% | **31.5%** | **1.3%** | 11.7% "orchestral stab" |
| minus 0.75 x | 29.0% | 17.4% | 0% | 9.4% "uplifter sweep" |
| minus 1.0 x | 34.8% | 9.8% | 0% | 14.3% "uplifter sweep" (a new hub) |
| (C - mean) / std per phrase | 29.0% | 9.8% | 0% | 15.4% "uplifter sweep" |

Honest reading: in a mix no rule beats a constant guess on the one-shots (always "UI" would be right for 50.7% of
them, always "FOLEY" for 63.8%, because the library's toys and foley fit those kinds); the first build's 42% was
largely the hub phrase being lucky. The shipped rule gives up some of those lucky hits for far fewer false KEY / UI
tags on music, ambience and voice. Scoring each kind by its phrases' mean was worse (20-26%); by the mean of its
two best phrases it gave 34.8% / 30.4% / 3.7% at the shipped damping, within noise of the shipped rule, so it was not
adopted (scratch analysis of `out/timeline/tag_bg_rows.npz`, which tag_bg.py writes). The only signal that predicted a
right tag was exposure: with the shipped scoring over all 536 events (background fitted on half of them),
one-shots at least 12 dB over their bed were right **50%** of the time (n=46) vs **25%** below (n=91); the tag's
margin over the runner-up did not predict correctness (31% vs 34% overall). So the transcript prints the kind plain
only for events >= 12 dB over their bed (EXPOSED_DB) and with '?' otherwise, and the sheet draws those labels pale.

### Clawd (evaluation only)

`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py --parts tags`

Each placement time is tagged exactly as timeline tags a detected event; "kind" = the kind of the top phrase.
Strict = the right kind (KEY / WHOOSH or RISER / STEP); lenient also accepts UI for keys and FOLEY for steps.
Top-1 strict / lenient / top-3:

| placements | n | mix, shipped | mix, first build | sfx stem, shipped | sfx stem, first build |
|---|---|---|---|---|---|
| keyboard presses | 40 | 57.5% / 92.5% / 72.5% | 60.0% / 97.5% / 80.0% | 62.5% / 65.0% / 82.5% | 67.5% / 72.5% / 92.5% |
| keyboard, no celesta layer | 27 | 81.5% / 96.3% / 92.6% | 85.2% / 96.3% / 96.3% | 81.5% / 81.5% / 96.3% | 81.5% / 81.5% / 96.3% |
| whooshes | 9 | **55.6%** / 55.6% / 77.8% | 11.1% / 11.1% / 66.7% | 77.8% / 77.8% / 88.9% | 77.8% / 77.8% / 88.9% |
| grass footsteps | 21 | 0% / 0% / 0% | 0% / 0% / 0% | 0% / 4.8% / 33.3% | 0% / 0% / 9.5% |

False tags where there is no keyboard and no UI sound (20 tagged events each, timeline's own picking):

| | KEY / UI, shipped | KEY / UI, first build | most common, shipped |
|---|---|---|---|
| Clawd ambience stem (hum, rain, wind in grass, birds) | **35%** | 70% | "spacebar press" 4 |
| Clawd music stem (orchestral, celesta) | **5%** | 15% | "orchestral stab" 15 |

Grass footsteps read as "hi-hat" / "snare drum hit" (on the stem 19 of 21 "hi-hat"). Read tags as hints; the
measurements (time, level, prominence, length, centroid) are the facts.

On the reference films (refs part below, normal detail, 20 tagged each): ref_1 (tech house + voice-over) KEY / UI
30%, 9 VOICE, most common "spoken word" 6 (the review saw "chat message pop" 12 + "spacebar press" 4); ref_2 KEY / UI
60%, "chat message pop" 6 (review: 9); ref_4 5%, "drum hit" 8. ref_2 0-15 s at `--detail high`: "chat message
pop" 37 of 96 tagged (review: 71 of 96); 5 of the 16 AST-labelled events are now VOICE (AST speech >= 0.5), e.g.
`0:01.11  VOICE  speech (AST 0.85) / chat message pop`.

Budget: one tag per burst (same band within 0.5 s) before the rest; on the Clawd soundtrack the first typing burst
(2.0-2.9 s) now takes 2 of the 20 tags instead of 10, and the summary's accents are at least 0.5 s apart.

Excerpt treatment (`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/tag_tune.py --vocab compact --variants
delta,repeat`, first round, first-build scoring): repeat-padded excerpt 39.3% / 63.8% top-1 / top-3 (kept), bed
subtracted 34.4% / 55.2%, zero-padded 25.2% / 59.5%, the one-shot alone without any bed (upper bound) 61.3% / 92.0%.

## Sync, beat grid, sections (Clawd soundtrack with its cue list)

`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py --parts sync`

`timeline(soundtrack.wav, cues=cues_all.json)`: 153 cues at 60 fps.

- Reported: 88.9% of cues have an onset within 1 frame (chance 15.2%), 96.7% within 100 ms (chance 60.8%), median
  |offset| 0.5 ms; 30.1% land on a strong accent within 1 frame (chance 3.1%); 47.7% on the beat grid (chance 20%).
- Offsets against the truth (140 cues with an attack-aligned placement): within 5 ms of the true placement offset for
  **84.3%**, within 1 frame (16.7 ms) for 95.7%, within 40 ms for 98.6% (median error 0.0 ms, p90 7.1 ms).
- **"Without a hit" over all 153 cues** (review finding: the first build listed 5, of which 4 were sounded whooshes /
  risers; its "0 sounded cues reported as unsounded" excluded the 11 peak / end-aligned cues). Now a cut / cue with
  no onset is first checked for a level peak (a soft whoosh / swell) and reported as "level peak only, CHECK"
  instead. Result: **2 reported without a hit** (28.86 s and 50.50 s, both sounded whooshes, so 0 of 2 correct), 3
  reported "level peak only" (39.75 s riser end and 50.99 s whoosh: right; 48.12 s: the mixer lists it unsounded,
  though a lock-ripple layer starts at 48.115 s). The other unsounded cue (40.48 s) gets an unrelated onset within
  100 ms ("near"). So the no-hit list is still not trustworthy for build-ups: the transcript says so on every
  whoosh / riser / swell cue ("a soft build-up can still be missed: look at the sheet").
- The 11 peak / end-aligned cues: 6.0 swell OK, 11.5 riser OK, 12.0 whoosh OK, 17.75 whoosh near, 27.5 whoosh near,
  28.86 whoosh off, 39.75 riser peak, 48.0 pop OK, 50.5 whoosh no sound, 50.99 whoosh peak, 53.9 swell near.
- Beat grid: 120.0 BPM, "clear" (pulse clarity 0.48); the score is at 120 BPM: 87.6% of the beats from 0-52 s are
  within 40 ms of the 0.5 s grid and 92.3% of the downbeat guesses fall on even seconds.
- Sections: true music / ambience edits at 10.5, 12.0, 28.0, 40.0, 51.0, 52.2 s; detected boundaries 10.75, 28.65,
  40.0, 55.2 s: 3 of 6 matched within 1.5 s, 3 of 4 detections correct. Genres are mostly wrong under the busy
  effects ("glitch hop music?", "lo-fi hip hop beat?"); instruments, rain and birds are right.

Level-peak tuning (`.venv/Scripts/python.exe eval/timeline/swell_tune.py --peaks`, synthetic mixes): 10 of 33 whoosh
peaks have no onset / swell within reach; a level peak rising >= 3 dB (local maximum over +-0.3 s, within 150 ms)
is found for 6 of those 10, and for 13 of the 56 unanswered times on a 0.25 s grid at least 0.6 s from any placed
sound (23%). At 4 dB: 3 of 10 vs 7 of 56; a +-0.15 s local maximum found no more whooshes and more empty times
(21 of 56). Shipped: 3 dB, +-0.3 s, labelled as a check, never as a hit.

Section merging (`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/section_tune.py --shipped`, re-run this
round with the new merge of neighbours that read alike): recall 88.9%, precision 84.2% (18 true bed switches, 19
detected), median boundary error 0.04 s. A single 146 s pop song (cheer-up, rejected library music) now reads as 8
sections instead of 11 (the review saw 13 before the label merge existed).

Section labels (`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/label_tune.py`, first round, 52 unused
library tracks + 26 mix beds): centring the section phrases made dark / aggressive moods the top mood for 78% of
music beds with effects on top vs 44% raw, so raw scoring was kept and a mood is named only at z >= 2.5. Music vs
ambience was right for 26 of 26 beds; the ambience keyword was in the top 2 for 16 of 17.

Pulse thresholds (`.venv/Scripts/python.exe eval/timeline/pulse_tune.py`, first round): clarity >= 0.15 for 94.2%
of 52 music excerpts and 0% of 35 ambience excerpts. Tempo octave (`.venv/Scripts/python.exe
eval/timeline/tempo_tune.py`, same 52 music excerpts, librosa's tracker as a second opinion, not ground truth):
timeline agrees with librosa within 4% on 33 of 52 (38 counting octave errors); switching to the double tempo when
it pulses nearly as clearly changed at most 1 agreement (34 at ratio 0.5), so the tempo is not switched; the double
tempo is reported as `tempo_alt` when it pulses at least half as clearly ("~68 or 135 BPM" for the cheer-up song,
which librosa calls 136).

## Film: Clawd_1080p60.mp4 (picture cuts + cues)

`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py --parts film`

- `earshot.video.cuts` (frame difference, core) finds 26 visual changes; 6 are the blinking terminal cursor in the
  first 1.5 s. Of the 5 act changes (8, 14, 28, 40, 52 s) one has a detected change within 100 ms (8.0 s).
- Cut sync: 46.2% have an onset within 1 frame (chance 15.1%), 80.8% within 100 ms, median |offset| 18 ms; 26.9% on a
  strong accent (chance 3.0%); 65.4% on the beat grid (chance 20%); without a hit: 0.30, 0.80, 1.30 (cursor blinks),
  30.00 and 51.85 s (51.85 has a whoosh peaking on it, mo.bubble.up at -8.5 dB: still missed; no level peak rises
  3 dB there). Cue sync on the film's own audio: 86.3% within 1 frame, 4 level peaks only, 1 without a hit (50.50).
- Zoom: `earshot sheet Clawd_1080p60.mp4 --cues cues_all.json --from 44 --to 54` lists the same cues without a hit
  / level peak as the full analysis (1 without a hit, 50.50; cue 53.90 no longer flips to "without a hit"), because
  the zoom is analysed inside +-30 s of context; its accent threshold is ranked over that context and says so.
- Sheets: `out/timeline/eval/clawd_film_sheet.png`, `clawd_zoom_14-24.png`, `out/timeline/r2/film_zoom.png`, opened
  and checked: header lines no longer run off the edge (one sync line each for cuts and cues), labels never overlap,
  masked tags drawn pale with '?', frames at cuts, cue markers coloured by sync (violet = level peak only).

## Reference launch films (read-through check, measure only)

`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py --parts refs` (detail normal; transcripts and
sheets in `out/timeline/eval/{ref_1,ref_2,ref_4}_{timeline.txt,sheet.png}`)

| film | loudness | events | beat grid | sections | cut sync: onset 1 frame (chance) / accent (chance) |
|---|---|---|---|---|---|
| ref_1, 93.1 s | -12.1 LUFS, TP +0.1 dBTP, LRA 1.7; lead-in 0.6 s, tail 3.5 s | 662 (7.1/s) | none (clarity 0.11) | 2: "tech house music, vocal chop", then "... bass guitar \| voice: voice-over narration" | 28 cuts: 71% (60%) / 25% (11%) |
| ref_2, 28.9 s | -14.8 LUFS, TP +1.0 dBTP, 1 clipped run at 4.94 s | 209 | none (0.11) | 3: house / techno, vocal chop + voice-over | 9 cuts: 67% (46%) / 22% (9%) |
| ref_4, 32.3 s | -14.8 LUFS, TP -0.8 dBTP, LRA 7.0; lead-in 1.1 s | 319 (9.9/s) | 122 BPM, clear (0.61) | 3 | 41 cuts: 80% (71%) / 22% (15%); on beat 24% (20%) |

They read sensibly: loudness, true-peak overs and clipping are flagged; intro / outro quiet is a lead-in / tail,
not a stop-down; voice-over is recognised; the ref_1 sheet at `--width 2400` (`out/timeline/r2/ref_1_2400.png`)
shows every lane title in full. Cut sync in these dense mixes is close to chance, which the chance levels make
visible instead of hiding.

## Edge cases (review findings), each run on a real file

- 10 s of digital silence (`ffmpeg -f lavfi -i anullsrc ...`): "DIGITAL SILENCE: every sample is zero.", one
  "digital silence" section, the gap "= whole file", integrated / true peak None, no models run; the sheet's
  spectrogram is black and the waveform says "digital silence" (`out/timeline/r2/silence.png`).
- 0.05 s noise burst: "click.wav is 0.05 s long: too short for a timeline (>= 0.1 s); for a one-shot use earshot
  measure / describe". 0.3 s tone: true peak -18.1 dBTP ("too short for a loudness measurement") and one event at
  0:00.00 (a file that starts with sound now has an event at its first millisecond).
- mp3 with cover art (cheer-up): no longer treated as a video (0 frames, no frames lane, no cut method).
- Picture-only mp4: analysed as silence of the same length, its cuts listed and all unsounded ("NO AUDIO STREAM").
  An audio file as `--video`: "--video tone3.wav has no picture stream (an audio file?)".
- Frame cues without fps / with key "frame": "cues have frame numbers but no fps: pass {"fps": 60, "events": ...}";
  "frame", "time", "sec", "seconds" are accepted with fps.
- `--from 5` past the end of a 3 s file: "nothing to analyse between 5 s and the end: tone3.wav is 3.00 s long".
- Non-ASCII file names are backslash-escaped in console text (`t\xf6n.wav`), exact in JSON.
- Result cache: keyed also on the core decode / features / cut code; `--fresh` recomputes and stores (a sheet after
  a fresh timeline reuses it: `cached: True`, internal total 86.0 s of the fresh run).

## Wall-clock time (target: under ~2 minutes at detail normal, EARSHOT_THREADS=2)

`EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py --parts timing60,timing93`
(a fresh CLI process each: imports, model loads, `--fresh` = no cached result and no cached model outputs; decoded
audio and the film's frame differences were cached)

| input | detail | wall clock | internal total | other Python processes | model windows (CLAP events / sections, AST events / sections) |
|---|---|---|---|---|---|
| Clawd soundtrack, 60 s | normal | **81.4 s** | 74.6 s | 2 | 20 / 30, 1 / 3 |
| ref_1 film, 93 s | normal | **93.3 s** | 86.0 s | 4 | 20 / 32, 1 / 3 |
| ref_1 film, 93 s (first round) | low | 76.6 s | | 6 | 12 / 24, 0 / 0 |

In the first round, with 8-12 other Python processes, the same normal-detail runs took 113.0 s and 101.3 s (+ ~9 s
when the frame differences are not cached). Model cost then (12 processes): CLAP load 23.3 s, 0.49 s per window;
AST load 1.8 s, 4.75 s per window. Cached results return immediately; after a code change, cached model outputs make a
re-run take 5-35 s.
