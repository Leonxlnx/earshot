# What professional motion-design SFX are made of

A measured study of the sound design in eight professional launch / motion films (third-party, kept local, called
ref_1 .. ref_8 here), done to give earshot forge concrete targets: what a "pro designed" whoosh, riser, impact, click,
pop or sting actually looks like in numbers, and how it sits against music and picture.

Tools: `eval/forge/pro/` (earshot timeline at detail high + sheet per film, paged spectrogram overviews, zoomed crops,
bed-subtracted measurements, trims). Local data: `out/refs_pro/` (never committed: trims, `manifest.json`,
`registry.json`, `stats.json`, `films.json`, sheets and crops under `_work/`).

## How to read the numbers (and how far to trust them)

- **58 events** were picked by looking: earshot sheets, 8 s overview pages, then zoomed crops of every candidate.
  Only events that stand out over the bed were kept (an exposed accent, or an SFX-only stretch).
- **One film dominates.** ref_2 is a pure sound-design piece with no music bed; it supplies 39 of the 58 events and
  most of the clean ones. The other seven are music- or voice-led, and their effects are sparse or buried. The
  per-category numbers below are often 1-10 examples from 1-2 films: they are measurements of real pro work, not
  population statistics.
- **Bed subtraction.** Each event is measured as (event power - bed power), with the bed taken as the quietest 150 ms
  in the 0.6 s before the event (`measure.quiet_bed`, so a run of clicks or a previous hit does not count as bed). Bands
  where the event adds under 1 dB are marked masked and left out of the spectral medians. Over a music bed this
  removes most of the music, not all of it.
- **Levels** are K-weighted (BS.1770 filter): `over_bed` = loudest 100 ms of the event minus the quiet bed. Because
  the bed is the *quietest* stretch between beats, the figure reads a few dB higher than "over the average music level".
  A silent bed counts as -70 LUFS.
- **Balance** = bed-subtracted power per 1/3-octave band in dB relative to the event's own total (level never decides).
  `tilt` = least-squares slope of that balance from 100 Hz to 10 kHz, in dB per octave.
- **Attack** = from the first millisecond within 20 dB of the peak to the peak (1 ms bed-subtracted envelope).
  `t20` / `t40` = peak to the first fall 20 / 40 dB under it (10 ms smoothing). Decay shape: `convex` = holds, then drops
  (a gate or a fade), `exp` = straight in dB, `concave` = drops fast, then a long tail.
- **Pitch** = the dominant spectral ridge (or the centroid for noise). Where a film's tonal music sat under a zip, the
  automatic ridge locked onto the music, and the sweep was read off the zoomed spectrogram instead (marked *read*).
- All audio is the films' decoded AAC at 48 kHz stereo.

## The films: music first, effects placed

| film | dur s | LUFS | LRA | cuts/s | beat | voice-over | cuts with an onset within 1 frame (chance) | strong accent within 1 frame (chance) |
|---|---|---|---|---|---|---|---|---|
| ref_1 | 20.5 | -10.4 | 6.4 | 0.78 | 107 BPM, weak | yes | 31% (35%) | 6% (8%) |
| ref_2 | 29.5 | -22.2 | 12.2 | 0.54 | none | no | 50% (44%) | 19% (10%) |
| ref_3 | 49.3 | -17.4 | 5.5 | 0.63 | 133 BPM, weak | yes | 58% (46%) | 16% (9%) |
| ref_4 | 8.7 | -23.1 | 0.7 | 1.71 | 111 BPM, clear | no | 20% (20%) | 7% (4%) |
| ref_5 | 28.9 | -14.8 | 3.6 | 0.31 | none | yes | 67% (46%) | 22% (9%) |
| ref_6 | 32.3 | -14.8 | 7.0 | 1.27 | 122 BPM, clear | no | 81% (71%) | 22% (15%) |
| ref_7 | 41.8 | -7.3 | 3.3 | 1.60 | 92 BPM | no | 33% (31%) | 4% (7%) |
| ref_8 | 93.1 | -12.1 | 1.7 | 0.30 | none | yes | 71% (60%) | 25% (11%) |

What this says:
- **Six of eight films let the music (and the voice) carry the cut rhythm.** Their cuts have some onset within a
  frame 20-81% of the time against a chance level of 20-71% (a few points above chance; ref_5 is the clearest at 67%
  vs 46%), and a strong accent within a frame only 4-25% of the time. Designed effects show up
  at a few chosen moments: opening UI ticks over a near-silent bed (ref_6), tonal zips into cuts (ref_5, ref_6), a noise
  riser at the very end (ref_1), a sub drop under a cut and an end chord (ref_8). ref_4 (a cut-down) has no designed effects at
  all; ref_3 and ref_7 bury theirs under voice or a -7 LUFS wall of music.
- **ref_2 is the other model:** a film scored entirely with effects. No beat, 12 stop-downs (drops to silence
  between ideas), LRA 12 LU, integrated -22 LUFS. Every effect is exposed, and every one is short, band-limited and
  shaped (see below).
- So "pro" is mostly **restraint and placement**: few sounds, each with one clear gesture (a pitch move, a gate, a
  thump), cut on the picture, sitting about 13-17 dB over the music when there is music.

## Per category

Counts are measured events (exported trims in brackets: forge / bench). Ranges are min-max, "med" = median.

### Whoosh / swell into a cut (`whoosh_big`), 5 (2 / 0)
- **Made of:** in the SFX-only film a whoosh is a **low swell, not air**: energy almost all under 150 Hz (bed-subtracted
  octave balance, re total: 31 Hz -19, 63 Hz -14, 125 Hz -14, 250 Hz -19, 1 kHz -28, 4 kHz -31, 16 kHz -37 dB),
  centroid 51-150 Hz, tilt -3 to -13 dB/oct, a faint high air layer on top at -20..-30 dB. The two buried whooshes in
  ref_1 are brighter (centroid 0.5-0.9 kHz) but the music makes that uncertain.
- **Envelope:** 0.5-1.0 s, symmetric to slightly front-loaded: rise (-20 dB -> peak) 190 ms, fall to -20 dB 90 ms,
  to -40 dB 240 ms; rise shape straight-to-concave in dB (it accelerates into the peak).
- **Pitch:** the centroid falls about 2.3 octaves across the pass (8.28-9.30 in ref_2): a pass-by, not a static noise.
- **Stereo:** mono or near-mono (side 30-40 dB under mid); no pan travel in any example. The width in these films
  comes from textures, not from whooshes.
- **Timing:** peak 127 ms *after* the cut (ref_2 8.64), or a swell that peaks 313 ms before the cut and dies into it
  (ref_2 2.88); the buried one in ref_1 peaks 100 ms before its cut. So: peak within about +-150 ms of the cut, with
  the energy straddling it.
- **Level over music:** +15 to +21 dB (ref_1, two examples).

### Riser (`riser`), 3 (1 / 1; one music riser measured only)
- **Designed riser (ref_2, 0.84 s):** three layers you can see: (1) a **stutter gate**, 34 pulses at 22 ms median
  spacing (45/s), speeding up from 24 ms to 20 ms toward the cut; (2) a **rising tonal layer**, about 2 kHz -> 8 kHz
  (+2 oct; the last 0.22 s climbs 4.2 -> 8.1 kHz, about +4 oct/s); (3) a steady low drone (50-500 Hz) under it all.
  It ends on the cut (+20 ms) and hands over to a low whoosh / boom that blooms just after.
- **Noise riser (ref_1, 1.15 s, film end):** high band (> 4 kHz) climbs about 35 dB, centroid +1.4 oct
  (360 Hz -> 1.1 kHz ridge), rise concave in dB (slow start, fast finish), side 14 dB under mid.
- **Music riser (ref_7, measured only):** high band +20 dB over 0.45 s into the drop, only +5 dB over the music.
- **Rise shape:** concave in all three: the level accelerates; most of the rise happens in the last third.
- **Level over music:** +15 dB (ref_1).

### Impact (`impact`), 3 (2 / 0)
- **Made of:** sub-dominated: 31 Hz -15, 63 Hz -9, 125 Hz -12, 250 Hz -18, 500 Hz -36, 1 kHz -45 dB re total; under 150 Hz
  holds all but 0.7 dB of the energy in ref_2 4.71; tilt -3.6 to -5.7 dB/oct; ridge at 82 Hz.
- **Layers:** a broadband transient on top (full-band vertical line at the onset), the sub body, and **a textured tail**:
  ref_2 4.71 has a granular low rumble (pulses every 28 ms, 36/s, slowing slightly) lasting 1.2 s; ref_2 14.02 has a
  sustained high air band (5-10 kHz) over the body.
- **Envelope:** attack 50-290 ms to the loudest point (they swell into the body rather than peak on the transient);
  tail: t40 230 ms to 1.08 s, decay 38 dB/s for the rumble; **both designed impacts end with a hard gate** (the tail
  is cut, not left to ring).
- **Stereo:** the body is centred, the rumble / air tails are decorrelated (correlation -0.01 and 0.24, side about
  level with mid, pan travel 28-30 dB).
- **Timing:** the swell starts 110-200 ms *before* the cut; the loudest point lands 65 ms before to 82 ms after it.

### Sub drop (`sub_drop`), 1 (0 / 1)
- ref_8 85.58: a **sub tone 64 -> 44 Hz** (-0.55 oct over 0.75 s) entering on a cut with a bright transient on top;
  energy share under 150 Hz -0.5 dB of total; rise to the sub peak 430 ms (it blooms after the cut: peak +380 ms),
  decay 42 dB/s (straight in dB), t20 360 ms; narrow (side 13.5 dB under mid); +13 dB over the voice + music.

### Logo / end sting (`logo_sting`), 2 (1 / 1)
- **Designed (ref_2 28.62):** a **falling chirp 1.5 kHz -> 150 Hz in 85 ms** (-3.3 oct, about -38 oct/s) landing on a
  **low tonal hum** (fundamental about 64-80 Hz with harmonics to 160 Hz, sinking -0.6 oct) that rings about 0.8 s;
  mono (correlation 1.0); balance peaks at 250-500 Hz (-13 / -6 dB re total), tilt -6.9 dB/oct; t20 58 ms, t40 260 ms
  with a concave tail. It starts 28 ms after the cut that ends the glitch.
- **Music ending (ref_8):** a struck end chord with a 4 s ringing tail (centroid 1.9 kHz, decays about 130 dB/s at
  first, then the tail).

### Glitch / stutter (`glitch`), 1 (1 / 0)
- ref_2 27.90-28.60: crackle first (sparse high grains, 4-12 kHz), then a dense broadband stutter: 11 bursts at 28 ms
  median spacing (36/s), **speeding up from 32 ms to 22 ms** into the cut; bright (centroid 9.2 kHz, tilt +3.8
  dB/oct, 8 kHz band strongest); **fully decorrelated** (correlation 0.04, side as loud as mid, pan travel 9 dB).
  **It ends exactly on the cut (0 ms)** and the sting takes over.

### Shimmer (`shimmer`), 2 (2 / 0)
- **Granular sparkle (ref_2 6.15, 1.95 s):** dense short partial grains every 20 ms (50/s) between 1 and 10 kHz over one
  steady tone at about 500 Hz; swells 0.5 s, fades about 0.8 s; balance 500 Hz -11, 1 kHz -15, 2 kHz -9, 4 kHz -15,
  8 kHz -20, 16 kHz -36 dB; tilt +6.5 dB/oct; **wide** (correlation 0.08, side 0.3 dB under mid, pan travel 14 dB).
- **Melodic chime run (ref_2 24.25, 1.95 s):** glockenspiel-like notes (270 Hz - 10 kHz partials) panned note by note
  (pan travel 94 dB = hard-panned notes), cut off by a gate at the end.

### Notification ding (`notify`), 4 (4 / 0)
- **A struck tone with one inharmonic partial:** fundamental about 1.8-2.2 kHz plus a partial near 3.4x (6.3 kHz),
  attack 1-15 ms, t20 27-42 ms, t40 30-180 ms (short, convex: they ring briefly and stop), 0.23-0.36 s in all.
- **Successive dings step up a semitone:** fundamentals 1822, 1934, 2045, 2168 Hz (ratios 1.061, 1.057, 1.060).
- Balance: 1 kHz -6, 2 kHz -5 dB, nothing under 500 Hz (< -38 dB); stereo moderate (side 5.5-13 dB under mid).
- +16 to +17 dB over the bed they ride on.

### Pop (`pop`), 3 (2 / 1)
- **Chirp pop (ref_2 13.10):** a sine **falling 9.8 kHz -> 350 Hz in 110 ms** (-4.8 oct, about -43 oct/s), preceded by a
  20 ms blip at 3 kHz; attack 32 ms (the loudest point is mid-sweep, around 1-6 kHz), t20 14 ms after it.
- **Blip pop (ref_2 13.30):** a 1.2 kHz tone with harmonics, gated on and off (70 ms box, 2 ms attack, 7 ms release).
- **Over music (ref_5 16.84):** a tonal blip falling 1.1 kHz -> 100 Hz with a click, +14 dB over the music.
- All three are **mono** (correlation 1.0 or side 40 dB down).

### Element pop-in (`pop_in`, not a forge category yet), 10 (10 / 0)
- The most used effect in ref_2: one per element appearing, fired in a run (17 events over 3.3 s, spacing 120-310 ms,
  median about 205 ms, settling into a steady 200 ms pulse).
- **Made of:** a **thump body at 80-150 Hz** (ridge peak about 105 Hz; under 150 Hz holds all but 0.5 dB of the
  energy; tilt -4 to -6 dB/oct) plus a **short noise tick** (1-10 kHz, about 30 ms, 30-40 dB under the body), often
  as a quick double hit (two bodies 60-70 ms apart).
- **Envelope:** 0.14-0.27 s, attack 6-19 ms (median 16), t20 16-35 ms (median 20), t40 30-80 ms; convex (holds, then
  stops) in 7 of 10.
- **Stereo:** moderately wide (side 3.5-9 dB under mid, correlation 0.38-0.78).
- **Timing:** lands 25-40 ms after its cut (first and last of the run).

### Tap (`tap`), 1 (1 / 0)
- ref_2 18.70: a 150 Hz thump (ridge 130-160 Hz) with a faint broadband click, 0.1 s, attack 11 ms, t20 18 ms, t40 27 ms,
  tilt -6.5 dB/oct, side 5 dB under mid; 86 ms before its cut.

### Click (`click`), 5 (1 / 2; two measured only)
- **Bright UI clicks (ref_2, ref_6, ref_8):** attack 0-3 ms, t20 6-16 ms; centroid 5-10 kHz, tilt +4 to +12 dB/oct; the
  ref_2 click has tonal partials at 5-8 kHz (a "tink" rather than a noise snap).
- **Mouse-like click over music (ref_5):** broadband with a low body (centroid 430 Hz, tilt -3.7 dB/oct), attack 10 ms,
  t20 11 ms, t40 37 ms, +13 dB over the music.
- Mostly mono (side 10-14 dB under mid). ref_6's opening click lands 31 ms before its cut.

### Hover tick (`hover_tick`), 6 (3 / 3)
- Tiny high ticks: 35-80 ms windows, attack 0-5 ms, t20 6-25 ms; centroid 2.8-9.5 kHz; tilt +2.4 to +6.2 dB/oct;
  balance: 1 kHz -23, 2 kHz -18, 4 kHz -13, 8 kHz -8 dB re total, almost nothing under 500 Hz.
- Two kinds: **tonal pings** (ref_2: a single partial at 6-15 kHz, about 10 ms) and **filtered noise ticks**
  (ref_6: 2-12 kHz noise, 25-40 ms, the spectral peak falling as it decays).
- Over a quiet bed: +15 to +20 dB.

### Typing (`typing_key`, `typing_run`), 7 keys (6 / 1) + 1 run
- **Designed typing is high-passed noise, nothing else:** each key a noise burst at 3-12 kHz (ridge 8.6-11 kHz),
  **essentially nothing under 2 kHz** (bands up to 1 kHz at -60 dB, the floor; 2 kHz -28, 4 kHz -13, 8 kHz -7,
  16 kHz -14 dB re total), tilt +12 to +19 dB/oct; attack 2-5 ms, t20 11-18 ms, t40 about 60 ms; convex; mono.
- **The run (ref_2 10.40-11.95):** keys at 62-148 ms spacing (median about 87 ms, about 11.5 keys/s), level and
  timing jittered, a soft low swell rising under the end of the run.
- The bench key (ref_8, in a voice-over gap) is a more realistic mid-range key (centroid 3.2 kHz).

### Zip (`zip`, not a forge category yet), 4 (0 / 4)
- **Fast tonal pitch sweeps** (a sine-like glide plus harmonics), 0.18-0.28 s, sitting only **+7 to +9 dB over the
  music** (lower than the hits):
  - up: about 100 Hz -> 1.5 kHz in 0.2 s (*read*, about +20 oct/s), starting 39 ms before a cut (ref_6 1.60);
  - up: about 100-150 Hz -> 10 kHz in 0.22-0.25 s (*read*, about +30 oct/s), **ending on the cut** (+20 ms, ref_6 3.78);
  - down: 7.2 kHz -> 420 Hz in about 60 ms (*read*, about -60 oct/s), across a cut (peak 65 ms before it, ref_5 22.80);
  - arc: about 1 -> 3 -> 1 kHz over 0.25 s (*read*, ref_5 28.22).
- Straight lines on a log-frequency axis: the sweeps are exponential (constant octaves per second).

## What the pro sounds have in common

1. **One gesture each, and it is mostly pitch or gate.** Zips at 20-60 oct/s, pop chirps at about -43 oct/s, the sting
   chirp at -38 oct/s, dings stepping up a semitone, a riser's tonal layer climbing 2 octaves. Risers and glitches are
   **gated at 20-45 ms, and the gate speeds up toward the cut** (24 -> 20 ms, 32 -> 22 ms).
2. **Band-limited on purpose.** Typing has nothing under 2 kHz; pop-ins, taps, impacts and the designed whooshes live
   under 150-250 Hz with only a thin tick or air layer on top (20-40 dB down); dings have nothing under 500 Hz. The UI
   and transition sounds mostly leave 300 Hz-1 kHz (where the voice and the music's chords are) to the music; only the
   sting, the zips and the chirps pass through it.
3. **Hard ends.** Short UI sounds hold, then stop (convex decay in 7 of 10 pop-ins, 5 of 6 keys, 4 of 4 dings); both
   designed impacts and the chime run end on a gate, not a natural ring-out. Long natural tails are the exception (the
   music end chord).
4. **Mono for objects, wide for textures.** Pops, keys, clicks, whooshes and the sting are mono to near-mono; shimmer and
   glitch textures are decorrelated (correlation 0.04-0.36) and move (pan travel 9-94 dB); pop-ins and impact tails are in between.
5. **Placed on the cut, sound a little early.** Attacks of impacts, sub drops, taps and clicks start 30-200 ms
   before the cut, with the body or peak landing within about 80 ms of it (sub bloom up to +380 ms); whoosh peaks
   within about +-150 ms; risers, the zip-up into a cut and the glitch **end** on the cut (0-20 ms) and hand over to
   the next sound; pop-ins land 25-40 ms after their cut.
6. **Levels.** Over music, hits, clicks, pops and dings mostly sit +13 to +17 dB over the quiet bed between beats
   (K-weighted, loudest 100 ms; up to +21 dB for a whoosh), tonal zips +7 to +9 dB. In the SFX-only film, effects peak around -16 to -38 LUFS (loudest 100 ms)
   with silence between ideas.

## Targets for synthesis recipes

Concrete numbers a recipe should hit to read as "pro designed" in this idiom (measured above; ranges = what the films
do; judge with `eval/forge/pro/measure.py` on the render, alone, bed = silence).

| category | length | attack | tail | balance | pitch / motion | stereo |
|---|---|---|---|---|---|---|
| whoosh_big (designed swell) | 0.5-1.0 s | -20 dB -> peak 150-250 ms, accelerating | -20 dB in 80-120 ms, -40 dB in about 250 ms | 80% + of energy < 150 Hz, tilt -3..-12 dB/oct, air layer 20-30 dB down | centroid falls about 2 oct across the pass | mono |
| riser | 0.8-1.5 s (longer allowed) | concave: half the rise in the last third | ends on the hit / cut, hands over | drone + tonal layer + high noise | tonal layer +2 oct (2 -> 8 kHz); gate 24 -> 20 ms (40-50 pulses/s) accelerating; high band +30-35 dB over the build | mostly centre (side about 14-18 dB down) |
| impact | 0.6-1.3 s | transient on top, body peaks 50-290 ms later | textured tail (grains every 25-35 ms) then a **hard gate**; t40 0.25-1.1 s | > 75% < 150 Hz, tilt -4..-6 dB/oct, ridge about 80 Hz | tail grains slow down slightly | centre body, decorrelated tail |
| sub_drop | 0.8-1.5 s | 30-50 ms transient, sub blooms 0.3-0.4 s after | 40 dB/s straight-in-dB decay | > 85% < 150 Hz | 64 -> 44 Hz (-0.5 oct over 0.75 s) | narrow |
| logo_sting (designed) | 0.8-1.5 s | 70 ms chirp | t40 about 260 ms, concave | peak 250-500 Hz, tilt about -7 dB/oct | chirp 1.5 kHz -> 150 Hz in 85 ms into a hum at 64-80 Hz sinking 0.6 oct | mono |
| glitch | 0.4-0.8 s | crackle then stutter | ends on the cut | bright: centroid about 9 kHz, tilt about +4 | bursts every 32 -> 22 ms (accelerating) | fully decorrelated |
| shimmer | 1.5-2.5 s | swell 0.5 s | fade 0.8 s or gate | 0.5-8 kHz, tilt +2..+6.5 | grains every 15-25 ms over one steady tone | wide (corr < 0.4), moving |
| notify | 0.2-0.4 s | 1-15 ms | t20 25-45 ms, convex (stops) | fundamental 1.8-2.2 kHz + inharmonic partial near 3.4x; nothing < 500 Hz | a sequence steps up 1 semitone per ding | near mono |
| pop (chirp) | 0.1-0.15 s | loudest mid-sweep | t20 about 15 ms after it | mid-high | sine falling about 5 oct in 110 ms (about -43 oct/s), optional 20 ms pre-blip | mono |
| pop (blip) | 50-80 ms | 2 ms | 5-10 ms release (box) | 1-1.5 kHz tone + harmonics | none | mono |
| pop_in | 0.14-0.25 s | 6-20 ms | t20 about 20 ms, t40 30-80 ms, convex | thump 80-150 Hz, > 85% < 150 Hz; tick 1-10 kHz 30-40 dB down | optional double hit 60-70 ms apart; runs every about 200 ms | slightly wide (corr 0.4-0.8) |
| tap | about 0.1 s | about 10 ms | t20 about 18 ms, t40 about 27 ms | 130-160 Hz thump + faint click, tilt about -6.5 | none | near mono |
| click (UI) | 30-60 ms | 0-3 ms | t20 6-16 ms | centroid 5-10 kHz, tilt +4..+12, tonal partials 5-8 kHz allowed | none | mono |
| hover_tick | 20-40 ms | 0-5 ms | t20 6-25 ms | centroid 3-9.5 kHz, < 500 Hz empty | noise tick with a falling peak, or a single 6-15 kHz ping | mono |
| typing_key (designed) | 50-80 ms | 2-5 ms | t20 11-18 ms, t40 about 60 ms, convex | noise 3-12 kHz only: < 2 kHz at least 30 dB down, tilt +12..+19 | none; a run jitters spacing 60-150 ms (about 11 keys/s) and level | mono |
| zip | 0.15-0.3 s | 20-35 ms | ends on the cut | tonal glide + a few harmonics | exponential sweep 20-60 oct/s: up about 100 Hz -> 1.5-10 kHz, down about 7 kHz -> 400 Hz, or an arc | near mono |

Placement targets (for anything that scores a film): attacks 30-200 ms before the cut with the peak within about 80 ms
of it; whoosh peaks within +-150 ms; risers / zips-up / glitches end on the cut (0-20 ms) and hand straight over to the
next sound; over music, accents at +13..+17 dB over the quiet bed and tonal sweeps at +7..+9 dB; leave silence or the
music alone between ideas.

## The reference sets this produced (local only)

`out/refs_pro/registry.json` is in the `earshot.forge.refsets` format (`{'version', 'categories': {cat: {forge, bench,
note}}}`, paths relative to the file), so `EARSHOT_REFSETS=out/refs_pro/registry.json` (or `refsets.load(path)`)
reads it directly. The split is **by film**: ref_2, ref_3 and ref_7 are forge for every category; ref_1, ref_4, ref_5,
ref_6 and ref_8 are bench (checked with `refsets.check_split`). In practice every forge trim comes from ref_2 (ref_3 and
ref_7 gave nothing clean enough to export) and the bench trims come from ref_1, ref_5, ref_6 and ref_8. 51 trims (37 forge, 14 bench), trim + fade only, original
gain, stereo 48 kHz 24-bit, with the music bed still underneath where there was one (each manifest entry carries its
bed level, over-bed and SNR, the earshot timeline events inside the window and the nearby cuts, and every measurement).

| category | forge | bench | note |
|---|---|---|---|
| click | 1 | 2 | |
| hover_tick | 3 | 3 | |
| typing_key | 6 | 1 | |
| pop | 2 | 1 | |
| riser | 1 | 1 | |
| logo_sting | 1 | 1 | |
| whoosh_big, impact, shimmer, notify, glitch, tap | 2, 2, 2, 4, 1, 1 | 0 | only the SFX-only film has them exposed |
| sub_drop | 0 | 1 | |
| pop_in, typing_run (new) | 10, 1 | 0 | |
| zip (new) | 0 | 4 | |

The benchmark can only use categories with bench trims (click, hover_tick, typing_key, pop, riser, logo_sting,
sub_drop, zip); the rest are forge-side targets. `whoosh_fast`, `swish`, `swipe`, `toggle` and `downlifter` were not
found exposed in any of the eight films.

Rebuild: `python eval/forge/pro/run_timeline.py ref_1 ...` (earshot timeline + sheet), `viewer.py overview|crops`,
`study.py` (measure, export, manifest, registry, stats, category sheets), `extra.py` (ridges, pulse rates),
`film_table.py`. The curated event list with every time and note is `eval/forge/pro/events.py`.
