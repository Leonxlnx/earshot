# Sound design rules: launch videos and code-made films

The rules the designers, critics and fixer work from. Numbers marked *(norm)* are measured norms of top-tier launch
films (from the Cinetic skill's measurements, `references/sound.md` §13 there); the rest is practice that held up on
shipped films. The earshot repo's `docs/launch-video-sound.md` profiles 7 professional reference launch films in depth
(earshot's bundled launch profile); use it for the targets of a specific film, this file stands alone for the craft.

## 1. The bar

- **Every sound has a visible cause**, at a loudness that matches the cause: a caret blink whispers, the hero reveal
  hits. Not every picture event gets a sound; once a class of action is sounded, every instance is, at one level.
- **Real and close.** Professional recordings: close-miked, dry, clean (floor under -55 dBFS between events), full
  bandwidth (95 % rolloff above 12 kHz), no clipping, no room of their own, no background events.
- **One palette.** One material family per film (felt + wood, glass + soft plastic, metal + air), one click voice, one
  keyboard, one music track. A different material for every sound is the fastest way to sound cheap.
- **The music carries the film.** Effects punctuate; the music's own events (drop, stop, re-entry) carry the big
  moments and most cuts.
- **Restraint.** About 5 designed effects per 10 s *(norm; 0-28)*, clustered where the music is sparse (cold open,
  typing, logo build) and thinning out after the drop. Good sound design mostly disappears.

## 2. Launch-video grammar (picture event -> sound)

| picture | sound (real source) | where it sits | level / notes |
|---|---|---|---|
| cursor moves, hovers, scrolls, progress bars, streaming text, rolling counters | nothing | | the music carries continuous motion |
| cursor click / tap / press | one real click voice for the whole film: mouse, trackpad, phone tap on glass; press + release 60-110 ms later (about 85, release 0-3 dB quieter) *(norm)* | the press frame (clicks land 10-30 ms before the visible press in pro films) | about 3 dB under the music's short-term level, 12-20 dB over it above 3 kHz *(norm)*; bright but short (-20 dB within 6-10 ms); centred; the same level every time |
| typing | sliced real keyboard (mechanical or laptop, one board per film); space / return deeper and longer | each sounded key on its character frame; at most ~12 keys/s; above ~25 chars/s one key per word or silence *(norm)* | music dips 6-10 dB under the run; vary the key sample and level (sigma ~2 dB, no jump over 6 dB), not random pitch; most on-screen typing in pro films is silent *(norm: ~75 %)* |
| a card / chip / toast / badge appears | a soft real pop or tick: cork, mouth pop, plastic click, wood block, pitched up for size | 50 % of its travel | a run climbs (`pitch_ramp` +1..+3 st); +2..+9 dB over the music for word-like pops *(norm)* |
| toggle, switch, lock-in, snap into place | a real latch, switch, magnet clack, Lego click | the contact frame | short; in sync within a frame |
| a counter or metric locks on its value | one crisp tick on the final value only | the lock frame | about 8 dB under the music *(norm)* |
| a camera-scale move, whip, zoom into the product, logo zoom | a real swish (bamboo, stick, cloth, cape, air past the mic) | apex on the move's velocity peak, onset 100-450 ms ahead *(norm)* | about 2 per film, never on every cut; may be wide; travels with the motion (pan) |
| hard cuts | nothing | | about 85 % of cuts in pro films have no sound of their own *(norm)* |
| tension into the reveal | a real riser: cymbal roll (yarn mallets), orchestral swell, the music's own build | ends 30-250 ms before the hit (a beat of air), or stops on it | in about 40 % of films *(norm)*; never a crash cymbal on the hit |
| the hero reveal / the turn | the music's DROP lands on it (edit the track so its downbeat is on the frame); optionally one real impact: drum, taiko, kick, felt sub hit, with a short top transient | the contact / cut frame | the loudest moment of the film, at 30-60 % of the runtime, +2..+7 LU over integrated *(norm)* |
| the logo / lockup | the music stops on a bar line or filters down, 0.2-0.7 s of near-silence, then ONE soft resolved element: a bell, celesta or piano note in the track's key, a soft swell | the frame the wordmark becomes readable (its settle), never later | 8-20 dB under the film's peak *(norm)*; a 0.5-2.5 s tail to below -60 dBFS, then silence |
| product objects (a phone, a card, a box) | real foley of the object's material | contacts | dry, centred |

## 3. Music

- **One licensed track** at 110-130 BPM (or half-time 55-65) whose structure fits the film: soft intro, groove, a
  breakdown or stop at 40-80 % of the runtime, a drop for the hero reveal, a resolve. Steady tempo, 4/4,
  professionally mastered, full bandwidth, instrumental.
- **Avoid:** vocals, corporate ukulele + whistling, EDM brostep, lo-fi hiss, trailer braams on a product film,
  dark/tense moods for a friendly product, famous overused stock tracks.
- **Match the brand's personality:** calm/literary -> soft piano, felt, strings, no booms; fast/exact -> tight drums,
  plucks, a drop on true silence; bright/friendly -> marimba, claps, pops, major key; precise/expensive -> sparse piano,
  low strings, air; raw/mechanical -> ticks, latches, metallic hits; warm/kind -> plucked strings, soft pads.
- **Edit at bar lines** (`arrange.py`): the picture is cut on the grid (120 BPM = 2 s bars at 4/4), the music is cut on
  its downbeats, splices continue the harmony (V -> I, or the same chord), and the drop's downbeat lands on the hero
  frame. Stretch only when the tempo is off by more than ~0.2 %, and by at most ~6 %.
- **Shape with rides, not with new music:** dip under typing, lift into the drop, settle under the logo (`buses.music.auto`).

## 4. Levels, arc and mix

- -14 LUFS integrated for web and social; true peak at most -1 dBTP after the AAC encode (`master.ceiling` 0.77 gives
  that). -15/-16 only for a stated reason (a dynamic orchestral track the limiter would squash).
- LRA 5-9 LU for a film with an arc *(norm: 1-2 sounds crushed)*; the intro 5-12 LU under the body for its first
  2-3 s, entering with intent on frame 0 (no fade over the first bar, no click on sample 0).
- The loudest momentary sits on the drop / hero reveal, not on the logo, not in the last 15 %.
- Effects are dry, mono and centred (+-0.15 pan); only whooshes, risers, ambiences and the music are wide. Low end
  below ~120 Hz is mono (master.py folds it).
- Every effect at least +3 dB over the bed in its own best band (qa.py `read_db`; >= 6 is clear); a layer that adds
  nothing measurable is a wasted placement (qa.py `--layers`).
- Nothing in the 2-6 kHz band much louder than the music's own loudest presence peaks (qa.py harshness): that band
  hurts first on phones and laptops. Effect centroids at or below 6.5 kHz *(norm)*; pings at 9-14 kHz fatigue.

## 5. Sync

- Impacts, clicks, keys, pops: `align: attack` on the contact / press / 50 % frame. Whooshes: `align: peak` on the
  velocity peak. Risers: `align: end` on (or 30-250 ms before) the hit. Beds and crowds: `align: start`.
- A sound that leads its picture by more than ~45 ms reads as "sound first"; one that peaks more than ~75 ms late reads
  as late *(norm: accents land -45..+75 ms, median +8)*. Move the picture to the beat, not the sound off its cause.
- Check slow-attack samples: an attack-aligned sample whose peak comes 80 ms after its first transient lands late
  (qa.py sync reports attack and peak per event).
- Cue frames are computed from the picture code; typed-in times land 8-11 frames off *(Cinetic, in practice)*.

## 6. Variation without randomness

- A repeated action draws from a pool of 4+ takes (`mode: cycle`: never the same take twice in a row), +-1-2 dB jitter,
  +-0.3-0.5 st pitch at most (real keyboards do not change pitch per key: vary the take, not the pitch).
- The UI click is the exception: one voice, one level, every time.
- A progression climbs: `pitch_ramp` across a run of pops or stacking blocks, `gain_ramp` into a build.
- Machine-gun (one sample, no variation, regular rhythm) is the second-fastest cheap tell; qa.py density flags it.

## 7. Tuning

- Tuned elements (chimes, bells, celesta, a motif) play in the track's key (`analyze.py --music` and `arrange.py`
  estimate it); tune a recording by resampling (`pitch`) within +-2 semitones; choose a recording near the right note
  rather than shifting far.
- Booms and subs on the track's root, or neutral (unpitched) at least; an untuned boom under a resolving chord sounds
  cheap even to people who cannot say why.

## 8. Film foley (character pieces, miniatures, worlds)

- **Scale through pitch and closeness:** a 1 cm character's steps are fingertip taps pitched +2..+6 st, close and dry;
  a giant's are slowed (-3..-6 st) heavy foley with a longer tail.
- **Layer by role:** body (the weight, 100-400 Hz) + transient (the crisp contact, 2-6 kHz, short) + detail (material:
  rattle, creak, rustle). Each layer is one rule on the same cue; check each adds something (`--layers`).
- **Ambience zones:** one bed per location, from clean windows of real field recordings (no voices, traffic, planes,
  dogs unless the story wants them), cut with the picture's cuts, ducked under key actions with `auto`.
- **Materials stay consistent:** the same key always sounds like the same key; the same surface like the same surface.

## 9. Choosing a recording (agents cannot listen: measure)

- `analyze.py` (format, floor, bandwidth, clipping, onsets, spikes in an ambience, tempo/key for music) and the
  spectrogram PNG; `earshot describe / match` (does it sound like the description? use earshot's verdict words:
  strong z >= 3, plausible 2.5-3, weak 2-2.5, no match below 2),
  `earshot rank` (sort candidates), `earshot find` (the best take inside a long recording: ready-made trims),
  `earshot compare --refs` / `earshot ab` (closer to approved references?), `earshot realness` (a real recording, not
  code-synthesized: require "real or pro-produced" on the trim you use; an "uncertain" needs its evidence read and
  written down).
- Prefer: lossless or HQ previews (Freesound -hq.ogg, Mixkit WAV), 44.1/48 kHz or higher, many takes of the same object
  (variation for free), a clean floor between takes.
- Reject: clipped, low-bitrate (a cliff above 16 kHz), roomy foley, anything with voices or bumps inside the window you
  need, anything whose licence you cannot read on its own page.

## 10. The cheap tells (what makes a soundtrack sound like random stock effects)

Code-synthesized beeps, sine pings and noise sweeps; one sample machine-gunned; a whoosh on every cut; a boom on every
logo; sparkle / shimmer sweeps standing in for "magic"; reverb on tiny sounds; random pitch on typing; partial
coverage (some presses heard, others silent); clicks at cuts; music chopped mid-note at a splice; an audible ambience
loop; a different room on every sound; the logo louder than the drop; an ending that stops mid-note instead of a tail
into silence.
