# Launch-film sound: the playbook for coding agents

You cannot hear the film you are making. This page tells you what professional launch films do with sound, how to
build that from real recordings, and how to check your result with earshot until the numbers say it is there. It is
written for films meant to sound like those references; when synthesized sound is the point of a film, read the
realness numbers as information, not as defects.

How to read it:

- **[measured]** = measured by earshot on the 7 professional launch films the author chose as the quality bar (modern
  SaaS / AI product launches and motion reels, web-delivered MP4s with 128 kbps AAC audio). Numbers come from
  `earshot/data/launch_profile.json`; the commands and the leave-one-out check are in `eval/launch/RESULTS.md`.
  Seven films are a small sample: treat the numbers as "what pros did here", not as laws.
- **[practice]** = established professional practice for web video and trailer / motion-design sound. earshot did
  not measure these claims.

The references (anonymised as in the profile):

| id | style |
|---|---|
| ref_1 | AI video-search SaaS launch: white UI walkthrough, chat prompts and cursor, kinetic type, logo end card; voice-over (93 s) |
| ref_2 | AI workspace explainer: white minimal kinetic type, app icons, chat UI, photographic logo reveal; voice-over (29 s) |
| ref_3 | editorial kinetic collage: grainy paper, typewriter text, photo cut-outs, pixel icons, fast cuts; voice-over (20 s) |
| ref_4 | fintech brand film: bold black / green kinetic typography, grids, numerals, flicker cuts, logo stings (32 s) |
| ref_5 | AI image-generation launch: gradients, typed prompt, photo collage, sound design with hard silences (30 s) |
| ref_6 | creator-commerce SaaS: blue UI walkthrough, notifications, cursor clicks, tagline end card; voice-over (49 s) |
| ref_7 | gaming motion-design reel: logo stings, 3D explosions, very fast cuts, brickwall-loud music (42 s) |

## The short version

1. **Music leads.** Music runs under (nearly) the whole film: AudioSet hears it in 89-100 % of 5 s windows in 6 of 7
   films [measured]. Pick the track first and cut the picture to it [practice].
2. **Loud, steady body.** Median -14.8 LUFS integrated; the loudest 3 s sit only 1.1-3.5 LU above the integrated
   level (median 2.3 LU); loudness range 1.7-12.2 LU, median 5.5 [measured].
3. **Start at once.** First sound within 0.04-0.48 s (median 0.13 s); 6 of 7 open within -1.8..+2.2 LU of the
   integrated level [measured]. No silent cold open.
4. **Land the ending.** The last 3 s of sound reach body level (median +0.2 LU vs integrated, range -2.2..+5.6);
   fades are short (0.08-2.54 s, median 0.27 s); dead air after the sound is short (median 0.04 s) [measured].
5. **Dense rhythm, a few big accents.** 2.5-8.2 onsets per second (median 5.7) but only ~1.2-1.7 salient hits per
   second (6 of 7 films) [measured]: a busy groove with a handful of moments that jump out.
6. **Not a hit on every cut.** Sound on visual events is no better than chance in most references (details below)
   [measured]. Dense music covers the cuts; deliberate accents go on the moments that matter (reveal, logo, key UI
   action) [practice].
7. **Real recordings, produced music.** Synthesized beeps and oscillator whooshes read as cheap next to films like
   these [practice]. Build from licensed recordings (`skills/real-sound/`) and check with `earshot realness`.
8. **Master for the web.** -14 LUFS integrated, true peak <= -1 dBTP on the master [practice]. 6 of 7 references
   clip after AAC decoding (true peak +0.1..+2.0 dBTP) [measured]: a common flaw, do not copy it.

## 1. Music leads the edit

- [measured] Music presence (AudioSet 'Music' >= 0.3 per 5 s window; music under voice or SFX scores 0.4-0.7, a
  bare track ~0.9): 100 % in 5 films, 89 % in the reel (all but its last, 1.8 s window), 67 % in ref_5, whose
  5-15 s stretch is sound design with hard silences.
- [measured] Tempo 108-136 BPM, median 123 (librosa beat tracker; double / half tempo errors are possible).
- [practice] Choose the track before animating; place section changes, the reveal and the logo on downbeats or bar
  lines; edit music on bar lines with short crossfades, never mid-phrase. If the film is a different length than the
  track, cut whole bars out of the middle, and keep the track's own ending.
- [practice] One track, or one composed edit of a track. Hopping between unrelated cues breaks the energy.

## 2. Loudness and peaks for web delivery

- [measured] Integrated loudness -22.2 .. -7.3 LUFS, median -14.8 (4 of 7 inside -16..-9). The quiet outlier (ref_5,
  -22.2) is the sound-design film with silences; the loud one (ref_7, -7.3, LRA 3.3) is a brickwalled reel.
- [measured] True peak of the decoded AAC: -0.8 .. +2.0 dBTP; 6 of 7 above 0 dBTP, i.e. they clip on playback.
- [practice] YouTube normalises playback to about -14 LUFS and only turns loud uploads down (a quiet master stays
  quiet); Apple's players aim at about -16; X, LinkedIn and website autoplay players do not normalise at all. So: about -14 LUFS integrated (up to about -10
  for a punchy social cut), true peak <= -1 dBTP on the master (AAC / MP3 encoding adds up to ~1 dB of overshoot),
  limiter last in the chain.
- earshot: `lufs_i` (web window -16..-9 LUFS), `true_peak` (<= -1 dBTP for a lossless master, <= 0 dBTP for an already
  decoded MP4 / MP3), `lra`, `crest_lu`.

## 3. The arc: open strong, hold the body, land the ending

- [measured] Opening level (first 10 % of the film vs integrated): -1.8 .. +2.2 LU in 6 films; ref_4 builds from
  -17.5 LU over ~4 s. Where the loudest 3 s sit varies (0.03-0.88 of the duration); 5 of 7 put it after 60 %.
- [measured] Body: loudness range median 5.5 LU, loudest 3 s median +2.3 LU over integrated. The references stay
  loud; they do not make you wait for a climax.
- [measured] The median loudness arc (LU vs integrated at 0 %, 10 % .. 100 % of the film) is flat: -1.3, -0.8, +0.3,
  -0.3, -0.2, +0.2, 0.0, -0.4, +0.9, -0.3, then -7.4 at the very end (ring-out / fade); between 10 % and 90 % the
  p10-p90 band is -5.8..+2.0 LU. A story-film arc (quiet start, climax at 70-80 %) is the opposite: the Clawd film's
  real soundtrack starts at -9.8 LU and peaks at +3.9 LU. `earshot critique` prints your arc next to the median.
- [measured] Silent holes: none longer than 0.04 s except ref_5's deliberate silences (1.66 s).
- [practice] Shape the energy with arrangement (layers in / out, filter sweeps, a stop-down before the reveal) rather
  than with big level changes.
- earshot: `intro_lu`, `end_lu`, `peak_pos`, `first_sound_s`, `max_gap_s`.

## 4. Picture and sound: what the references actually sync

earshot detects visual events (hard cuts and big picture changes: a frame-difference spike >= 4x its surroundings, or a
mean change >= 15 % of the grey scale) and compares them with audio onsets and salient hits. "Chance" = the same events
shifted off the picture by fixed offsets, so it is what random edit points would score with that music.

| [measured] share of visual events with ... | median of 7 | chance median | films >= 10 points above their chance |
|---|---|---|---|
| an onset within +-100 ms | 81 % | 86 % | ref_3 88 vs 68, ref_7 60 vs 44 |
| a salient hit within +-100 ms | 29 % | 27 % | ref_1 46 vs 30, ref_3 53 vs 35 |
| an onset within +-1 frame | 44 % | 44 % | ref_3 47 vs 36, ref_5 38 vs 24 |
| a tracked beat within +-70 ms | 22 % | 28 % | ref_4 40 vs 28, ref_5 38 vs 25, ref_6 52 vs 31 |

For comparison, the code-made Clawd film (its sound placed on picture cues by a script) scores 38 % vs 10 % chance
within +-1 frame and 69 % vs 25 % on the beat grid: earshot does see tight sync when it is there.

What that means: in these films the music is so dense (median 5.7 onsets/s) that almost every cut has *some* sound
under it, and they do **not** put a measurable hit on every cut. Some lock part of the edit to the beat. Caveat:
earshot measures the final mix; quiet SFX under loud music can be present without being separable.

- [measured] 4 of 7 use stop-downs: the sound drops >= 10 dB for 0.15-1.5 s and comes back on a hit (1.2-18.3 per
  minute).
- [measured] Visual events: 0.3-1.6 per second (median 0.63).
- [practice] **Whoosh into transitions**: the whoosh peaks on the cut (or on the fastest frame of the move) and starts
  0.2-0.6 s before it; match its length to the move.
- [practice] **Riser -> stop-down -> impact** for the big reveal: a 1-4 s riser, cut everything for 0.1-0.5 s, then an
  impact (with a sub drop) exactly on the reveal frame. Use it once or twice per film, not on every scene.
- [practice] **Accents with priority**: the reveal, the logo, the key UI action and the section changes get a
  designed sound; small motion rides on the music.
- [practice] For a sparse soundtrack (no dense music) every cut needs sound: `earshot critique` scores the share of
  visual events with an onset (a new attack) within +-100 ms (`cuts_with_sound`, the references' definition) and
  lists the events with no sound at all - no onset, no whoosh / riser peak, no level peak - by the same matcher as
  `earshot timeline` (an action when `cuts_with_sound` fails, a tip otherwise). Soft build-ups can still be missed:
  look at the sheet before adding a sound. Align samples on their attack, to the frame.
- earshot: `cuts_with_sound` (scored), `cut_hit_frac`, `cut_frame_sync`, `cuts_on_beat`, `stopdowns_min` (context).

## 5. UI foley (clicks, typing, notifications)

- [practice] One click per on-screen click, on the frame the cursor presses (not when the UI reacts). Typing: one key
  sound per typed character, from a multi-take recording so no two consecutive keys are the same sample; slightly
  irregular timing. Notifications: short, soft, tonal; avoid alarm-like beeps.
- [practice] Keep UI foley short, bright (2-6 kHz) and low: it should read as detail under the music, ~10-15 dB below
  the music peaks, panned toward where it happens on screen.
- earshot: `earshot find typing.wav "a single soft key press"` to cut takes; `earshot match` to check a sample is what
  its name says; the critique's `onset_rate` tells you if the track is too sparse.

## 6. Logo sting and endings

- [measured] The final moment lands at body level (median +0.2 LU); fade / ring-out median 0.27 s, longest 2.54 s;
  the strongest hit of the last 4 s rings out 2.0-3.6 s in 5 films; ref_7 ends without a final hit.
- [practice] End on a designed sting: a hit plus a tonal tail (2-4 s), with the music resolving into it or stopping
  for it, finishing within ~0.5 s of the last frame. Avoid long fade-outs and dead air.
- earshot: `final_lu`, `fade_s`, `end_silence_s` (scored), `tail_s` (context).

## 7. Frequency space

- [measured] Energy share per band (median, dB of the total): sub 20-60 Hz -6.6, low 60-250 Hz -4.2, low-mid
  250 Hz-1 kHz -5.3, mid 1-4 kHz -12.4, presence 4-8 kHz -17.8, air 8-16 kHz -23.6; brightness (median
  magnitude-weighted spectral centroid) 1169-3633 Hz, median 1966 Hz. Bass-led mixes, as expected for music-led films.
  (Above 16 kHz is not compared: AAC cuts it.)
- [measured] Stereo: side 14.5-5.0 dB below mid (median -8.5 dB), L/R correlation 0.52-0.93.
- [practice] Give every layer its own place: high-pass SFX and ambience (unless they are impacts), keep one element
  in charge of the sub at a time, dip the music 2-5 kHz and duck it 3-6 dB under voice, keep UI sounds out of the
  music's lead range. Check the mono fold-down; phones are mono.
- earshot: `band_*`, `side_to_mid_db` (scored minor; a mono mix, identical left and right, counts as -40 dB and is
  flagged), `brightness_hz`, `correlation` (context).

## 8. Density and restraint

- [measured] 2.5-8.2 onsets per second; salient hits ~1.2-1.7 per second in 6 films (the brickwalled reel 0.1).
- [practice] More sounds is not better. Sound the moments that carry meaning, let the music carry the rest, leave
  room (a stop-down, a breath before the logo). A cluttered SFX layer on top of busy music turns into noise.
- earshot: `onset_rate`, `hit_rate` (scored both ways: too sparse and too busy; salient hits need at least 0.05 /s,
  half the sparsest reference).

## 9. Voice-over

- [measured] 4 of 7 have a voice-over (AudioSet hears speech in 83-100 % of windows); 3 are music-only.
- [practice] If you use VO: duck the music 3-6 dB under it (sidechain or automation), keep SFX out of 1-4 kHz while it
  speaks, and never put a big hit on top of a key word.

## 10. Real recordings, not synthesis

- [practice] Code-synthesized "random" sounds (oscillator beeps, filtered-noise whooshes) are the fastest way to
  sound amateur next to films like these. Use licensed real recordings and produced music, with a licence audit and
  credits: `skills/real-sound/`.
- [measured] The Clawd film scores p_real 0.94 with its real-recordings soundtrack and 0.33 with the retired
  synthesized one (eval/realness/RESULTS.md, application files; eval/launch/RESULTS.md); that line is what separates them in the critique: the real one
  reads "close" (90 %), the synthesized one "needs work" (80 %, 1 critical). The structural metrics alone rate both
  close. Not an independent test: the realness model was trained on related material (eval/launch/RESULTS.md).
- earshot: `earshot realness mix.wav` gives p_real, a verdict and the 4 s windows that sound synthesized;
  `earshot critique` uses the same thresholds: the film's mean p_real against the file thresholds (ok at "real or
  pro-produced", critical below 0.5, where synthesized is the more likely reading, minor in between; a minor miss
  with no window leaning synthesized is only a tip), and its listed 4 s windows against the window threshold, as
  `earshot realness` lists them. A finished mix is harder to judge than its sources:
  run `earshot realness` on each stem.

## Checklist (run it, read it, fix, repeat)

```
earshot critique film.mp4                        # scorecard + concrete fixes with timestamps (start here)
earshot critique film.mp4 --fast                 # quick pass, no AudioSet / realness: a PARTIAL verdict
earshot critique mix.wav --video film.mp4        # a separate soundtrack checked against its picture
earshot timeline film.mp4                        # listening transcript: what sounds when, next to the cuts
earshot sheet film.mp4 -o out/sheet.png          # annotated picture of the soundtrack (open the PNG)
earshot realness mix.wav                         # which 4 s windows sound code-synthesized
earshot profile my_refs/ -o my_refs.json         # measure your own reference films ...
earshot critique film.mp4 --profile my_refs.json # ... and score against them instead of the bundled 7
```

1. Pick the music first; cut the picture to it. Check `music_frac` (whole film) and `tempo`.
2. Lay the accents: reveal (riser -> stop-down -> impact), logo sting, key UI actions, section changes. Check
   `hit_rate`, `stopdowns_min` and the critique's list of the biggest visual events without a hit.
3. Add UI foley on the on-screen interactions, aligned to the frame. Check `cuts_with_sound` and `onset_rate`.
4. Shape the arc: sound from 0:00, steady body, ending at body level. Check `first_sound_s`, `intro_lu`, `lra`,
   `crest_lu`, `final_lu`, `fade_s`, `end_silence_s`, `max_gap_s`.
5. Mix: balance bands and width. Check `band_*`, `side_to_mid_db`.
6. Master: about -14 LUFS, true peak <= -1 dBTP. Check `lufs_i`, `true_peak`.
7. Run `earshot critique` (without `--fast`) on the final MP4 and fix every critical and major line, not just the
   verdict: one structural defect on an otherwise good mix usually keeps it "close" (eval/launch/RESULTS.md, known-bad
   mixes: a 12 s fade-out, 4 s of dead air or a mono fold each stay "close" but are flagged with the fix). The
   verdict is coarse; the FIX FIRST lines are the signal. "launch-grade" or "close" with no open critical / major
   line is the goal; the 7 references score launch-grade (1), close (5) and needs work (1: the -22 LUFS
   sound-design film) in the leave-one-out check.

## What the numbers cannot tell you

- Taste: whether the music fits the brand, whether a whoosh sounds expensive. Use `earshot describe` / `match` /
  `compare` against good reference recordings and listen-proxy with `earshot sheet`.
- Sync of individual layers inside a dense mix (see section 4).
- Over-compression: a mix crushed by a 20:1 compressor and a limiter is not flagged, because two references are as
  dense (LRA 1.7 / 3.3 LU, loudest 3 s 1.1-1.2 LU above integrated). Listen-proxy with `earshot sheet` if in doubt.
- Clips under 10 s: loudness range, crest and opening / closing / final levels are not scored (they would measure the
  clip, not a film); the references are 20-93 s films.
- Styles outside the profile: for a cinematic trailer or a calm explainer, build your own profile from references
  of that style (`earshot profile`).
