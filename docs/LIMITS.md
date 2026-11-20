# What earshot can and cannot judge, and cannot make

earshot gives an agent measured evidence about sound, not taste. Read this before trusting a verdict. What `make`
cannot do and where it is weak is [below](#making-sounds-make), and what `build` cannot do
[at the end](#building-sounds-build).

## Trustworthy
- **Measurements**: loudness (LUFS, true peak, LRA), timing (onsets, offsets to cuts and cue frames, to the
  millisecond), levels, spectrum, envelope, clipping, silence, tempo of steady music. These are meters, not opinions.
- **"Is this the kind of sound I asked for?"** on single, exposed sounds (a key press, a whoosh, rain, a glass ting):
  a "strong" match is very reliable on held-out recordings; "plausible" and "weak" are hints worth a second check.
- **Picking among candidates** (`rank`) and **finding the clean moment** (`find`) in real recordings.
- **Obvious code-synthesized sources** (noise risers, simple FM bells, steady test tones): flagged with ~92% precision
  when the verdict is firm (6.7% of real clips get called synthesized, 5.5% of synthesized clips real). Bare beeps
  are less reliable than that list suggests: with the beep family held out, beeps were among the most-missed types
  (accuracy 0.71).
- **Whether sounds land on the picture**: offsets of hard hits to cuts / cue frames (median 0 ms on Clawd, 96% within
  a frame).

## Hints only
- **Tags of events inside a busy mix** (what each hit "is"): right about half the time for exposed events, a quarter for
  events under the music. Times and levels are right; the label is a guess.
- **Music genre and mood**: often wrong (an upbeat track's top mood was a dark one 18% of the time); describe keeps
  them apart as unvalidated hints. Instruments, rain, birds and voice are usually right.
- **"Uncertain" from realness** on designed library sounds ("magic" chimes/celesta, "motion" whooshes) and on real
  electronic music: they sit between real and synthesized. Read it as "sounds designed", not "fake".
- **The launch scorecard ranges** come from 7 films the author picked. They describe that style (modern SaaS / AI
  launch films, kinetic type, UI demos), not every good film. A cinematic short or a quiet piano piece will break
  rules that do not apply to it. Build your own profile from films you admire (`earshot profile`).
- **The "cuts without a hit" list** (timeline, sheet, critique tips): soft build-ups (whooshes, risers, swells) can
  still be missed - on Clawd both cues it named were sounded whooshes. Look at the sheet before adding a sound.
- **critique's "onset on visual events" share** counts only new attacks within +-100 ms (the definition the reference
  ranges were measured with), so a whoosh swelling through a cut does not count; its fixes and tips name only the cuts
  with no sound at all (timeline's matcher). critique's stop-downs (short drop-outs before a hit, per minute) and
  timeline's gaps (silences of any kind) are different counts.
- **realness verdicts depend on length**: a sound up to 6 s is judged with the clip thresholds (lo 0.17), a longer file
  by the mean of its 4 s windows with the weakly calibrated file thresholds (lo 0.40), so a lower p_real can get the
  milder verdict. Judge the trim you will actually use (`--start / --end`), not a multi-take file.
- **AudioSet labels on one-shots under ~0.5 s**: often 'unsure' or a near class (Zipper for a swish, Finger snapping
  for a click). Treat those as neutral; only an outright different class (Speech, Music, Engine) is a red flag.

## Cannot judge
- **Taste and emotion**: whether a sound is beautiful, whether the music fits the brand, whether a moment feels
  satisfying. earshot can say "a strong match for a deep thocky key press, 3 ms attack, lands 4 ms after the cut"; it
  cannot say the film feels premium.
- **A code-made layer hidden under a louder real mix**: usually missed. Judge source files and stems, not only the
  final mix.
- **Provenance**: realness hears synthesis, not licences. A professionally made, licensed synthetic UI sound can read
  "likely code-synthesized"; code-made noise beds that imitate nature (rain, wind) or telephone tones can read real.
- **Anything above 14 kHz from the models** (CLAP's front end stops there; the AudioSet tagger at 8 kHz). Use
  `measure` (bandwidth, bands) for air, hiss and codec cut-offs.
- **Speech content**: no transcription; voice is detected, words are not.
- **Licences**: earshot never decides whether a file may be used. The real-sound skill's `audit.py` checks manifests;
  the licence itself must be read on the item's page.

## Known engineering limits
- Cold start: every CLI call is a fresh process that loads the models (~20-40 s on a busy laptop), even for a cached
  file; the MCP server or one Python script (`from earshot import api`) loads them once. `doctor` without `--load`
  loads nothing heavy.
- bf16 is used only on CPUs with native bf16 (AVX512-BF16 / AMX); elsewhere earshot runs fp32 (bf16 emulation was
  measured 16x slower). The bundled calibration was measured with bf16 embeddings (fp32 agrees to cosine >= 0.9993 on
  12 real clips, `eval/semantics/precision.json`).
- Caches are keyed on path, size, modification time and 16 sampled 4 KB blocks of the file: a re-render that differs
  only between those blocks and keeps the old size and timestamp would still read stale. Use `timeline --fresh` or
  touch the file when in doubt.
- The models were evaluated on one sound library (Freesound / Mixkit picks for one film), 7 launch films and three
  synthesis code bases. Other domains (orchestral recording sessions, field recordings in other climates, other synth
  engines) are untested.
- The optional Gemini critique has not been validated against real answers (it needs the user's key and consent to
  upload).
- Cut detection marks big visual changes (a blinking cursor counts); slow crossfades are missed.

## Making sounds (`make`)

**What was measured and what was judged by ear.** Measured, by code: every take's audit (clicks, cut tails,
clipping, DC), its loudness and length, how close a learned take stays to its recording, and how well requests are
read ([VALIDATION.md](VALIDATION.md#8-made-sounds-10)). Judged by ear, by one listener (the maintainer) on pages that
were not blind: which kinds of sound are good enough ([VALIDATION.md](VALIDATION.md#listening-by-ear)). On 24
recordings each learned take was judged as good as its source, but the store as a whole has not been judged. The
sounds built last (the `ui` set as it is now, the film and motion hits, the drums from recorded strokes, the figures
on real instruments, the newer learned things and beds, and the cookbook's recipes) were measured and checked by the
audit, and no one has judged them by ear yet. earshot's own models did not predict the judgements by ear and are not
a measure of how good a made sound is; a `heard as` line says what a model hears, nothing more.

**What it cannot do.**
- Make a sound no design covers from a free description. `make` chooses one design and shapes it with the controls,
  or renders a recipe that layers designs. A request for a thing the catalogue lacks still gets a file: the nearest
  design, with a line that says `earshot makes no X; Y is only the nearest sound it has`, three starting points of
  other families, and the `earshot build` line that builds it (see [Building sounds](#building-sounds-build)). A
  cookbook recipe answers on its own for 59 things; the other 40 of the 99 are sketches that are starting points, and
  a recipe is judged by a model and the audit, not by ear.
- Realistic sounds for things nobody recorded. A learned sound exists only where someone recorded the thing and put
  the recording in the public domain; the store has 309 things. For other real things the catalogue has designs
  invented from physics (`action`, `object`), which sound clearly less real.
- Pan a catalogue sound. A request to `make` cannot place or move a sound in the stereo field: "from left to right" is
  said not followed. A layer of a recipe can be set to a side or moved across (`pan` [from, to]), and a part of a
  build that says "from left to right" gets that move.
- One sound ringing through the body of another ("rain on a tin can"). It was tried and came no nearer the real
  thing; where the catalogue has the real surface (`rain/wood`, `rain/tin_roof`, `step/floorboard`), that design is
  the answer.
- New sounds from a class of recordings. Sampling new members from a model of several recordings was tried and
  judged poor by ear; it was dropped. A learned design plays its own recordings and new takes of them.
- Music. earshot is a sound-effects tool: three short loops (`bed/warm`, `bed/playful`, `bed/8bit`), notes, chords
  of a pad, short motifs and logos. No melodies, songs or scores, and no music generation.
- Speech, singing and the melodies of instruments. Animals, people without words and vehicles exist as learned
  recordings: a dog barking (a big and a small one), a cat, a robin, a crow, a rooster, an owl, a horse neighing
  and trotting, a cow, a wolf; applause, a crowd cheering, laughter, a gasp, a cough, a kiss, a heartbeat, a baby
  crying; a car horn, a siren passing, an engine starting, a train's horn and a train passing, a helicopter, a
  bicycle bell; a woman's scream, a sneeze, a yawn, a sigh, snoring, a burp, a bite into an apple; a sheep, a goat,
  a chicken, a cat purring, a duck, a pig, a frog, a fly; cutlery on a plate, a broom, a running tap, a spray can,
  velcro, a glass and a plate breaking, a cash register; tyres screeching, a motorcycle passing, a gun reloaded; in
  the kitchen one knife chop on a board, a pot boiling, a pot lid set down, a gas stove lit, a spoon stirring in a
  mug, a plate set on a table, a toaster popping, a blender, an egg cracked, a whisk beating in a bowl; a crowd
  booing, a cat hissing, a lion roaring, an elephant trumpeting, a bee, a rocket launch, a record scratch, a squeaky
  toy, an elevator's arrival ding; steps in mud, through puddles and in dry leaves; machines started, run and
  stopped (a chainsaw, a lawn mower, a tractor, an angle grinder, a jackhammer, a paper shredder, a coffee machine,
  an electric kettle, a microwave, a hard drive, a fax), a kettle whistling, a truck reversing, a ship's horn, a
  steam engine, a metro arriving, wipers, a turn signal, a garage door, a motorboat; a monkey, a parrot, a donkey,
  a bear, geese, a puppy, a mouse, a rattlesnake, a mosquito; knuckles, teeth chattering, a hiccup, a whistle,
  panting, hands rubbed, a puff of breath, sitting down on a sofa; a skateboard, ping pong, pinball, darts, a
  crossbow, ice skates; a screwdriver, a ratchet, a trap, a binder clip, a party popper, a curtain, an office
  chair, a bed, a window, a swing, an envelope, a crisp packet, a drain, a garden hose, a sprinkler, a bicycle
  freewheeling, a tree falling; beds of hail on a roof and on a car, a dishwasher, a washing machine, a fridge, a
  shower, a jungle at night, a factory workshop and a gale.
  Instruments play single notes, chords of a pad and short figures, no melody. To a request for speech, a voice
  that says or sings words, a shout or a yell, sobbing, a violin, a cello, a viola, a saxophone, a flute, a
  clarinet, an oboe, bagpipes, a trumpet, an electric guitar, a guitar or drum solo, an air horn or a ringtone,
  `make` answers "earshot makes no ...". Crying is a baby's: asked of another person it is not made. Other everyday
  things have no recording yet and get the nearest design (a few of them a cookbook recipe): a gulp, chewing, a
  seagull, a pigeon, a snake's hiss, a whale's song, a dial-up modem, a padlock, a pan set on a stove, a fridge
  door, a quick row of chops (one chop is made), a printer, a shopping cart, a bus, an airplane passing close, a
  tennis or basketball hit, a clockwork, a car alarm, a hole punch, a throat cleared; steps on sand, on carpet and
  on metal stairs. No public-domain recording of a real snake hissing or of a whale was found (the ones offered are
  voices or made sounds); a modem's tones and a clockwork's ticks were learned and smeared (a take further from the
  recording than two takes from each other), so they were left out.
- Make two sounds in one call. A request that names two ("a water drop and a coin") makes one of them and says
  `not in this sound: ...` for the other; ask again for it, or make a set.
- Follow every word. Words that matched nothing come back as `not understood`; controls a design does not read, or
  a length it cannot keep, come back in plain words (`not followed: ...`, `0.43 s, not the 2 s asked ...`). A material picks the voice of a tonal sound only
  among six (wood, glass, metal, soft, digital, retro); other materials ("plastic", "ceramic") only steer the
  ranking. A theme's scale changes a figure only when it is minor. Bank sounds read pitch and length where their
  recipe has those knobs, and no axes. A learned sound keeps a length only within 0.6-1.8 times its natural one and
  a pitch only on tonal recordings (7 semitones at most).
- Exact lengths. A length sets how long a figure sounds before its room dies away, so the file runs longer than the
  number; a struck one-shot rings as it rings, and a voice cannot ring longer than it does (a marimba chime asked
  for 2 s came out 0.55 s); both say so. A loop has a shortest length per scene: an ambience asked shorter is a
  cut, not a loop, and a music loop keeps whole bars ("8 seconds" is 8.57 s).
- Choose by taste. The ranking reads words, not the product; two good designs for one job can differ a lot. Use
  `also`, `--sound` and a person's ears.

**What is weak or unverified.**
- **The rebuilt motion sounds are not judged by ear.** The motion and film sounds follow measurements of professional
  motion videos: short, dry hits, thumps, ticks and whooshes of air, built from learned recordings. The `film` set
  plays small real actions cut to those measurements (a door falling shut for the thump, the top of a mechanical key
  for the tick, a heavy bag dropped under the title and the logo, a stick and a broom through the air, a breath of
  air rising, a sword drawn). Of the earlier film sounds the fast whoosh was judged good and the hits, booms, risers
  and the tape stop poor; the rebuilt ones have not been judged. The measurements come from 39 videos and find
  effects only where they stand out of the music, so quieter effects under the music are not counted. 14 of the 69
  motion designs (pops, tocks, a blip, glints, a sparkle, key ticks, a sliding panel, a soft knock, the soft sting)
  are still invented.
- **The interface set is not judged by ear as a set.** The seventeen roles of `ui` are things a hand does, built from
  learned recordings. Which object a role uses is a choice of taste: the other candidates are in the catalogue by
  name.
- **Drums.** The acoustic kit and the hand percussion are takes of 130 recorded strokes (VCSL and Karoryfer; the hand
  claps from one person clapping, Freesound); they have not been judged by ear as a kit. The drums invented from
  physics that they replace were judged poor. Still invented: `kick/electro`, `snare/electro`, `hat/electro`,
  `clap/electro` (drum-machine voices), `snare/brush`, `snap/finger` and `sticks/count`, which no public-domain
  recording covers.
- **One carrier per band.** The learned model keeps one tone per band: two crickets in one band lose their beating
  against each other (the cricket bed's take differs from its recording by 4.4 dB against 2.2 for two takes).
- **Soft onsets smear a little**: a slow rise gets slower (a bag drop's rise went from 61 to 99 ms). What the
  rebuild smeared into the moments before a strike is held as low as the recording has it there on 110 recordings
  (`learned.LEAD`, on), each measured closer to its recording. The same repair for the faint hiss after a sound
  (`learned.TAIL`) is built and off: it did not hold on a second take of each recording.
- **Sea joints.** A stitched bed joins pieces where the texture is in a similar state; on the pebble and rock
  shores the joints differ by about 1.5 times the texture's own variation, more than in the other beds.
- **Takes of a learned design differ in length and level**, because they are different recordings. Make
  pick-at-random takes as a set, where none lies more than 2.5 dB over the quietest.
- **Some learned things still rest on a thin recording** (a bare strike, or a lossy file with a noisy floor): a
  scream, a yawn, a fly, a frog, a pot lid or a plate set down, a leather grab, an airy whoosh, a dial, a latch, a
  stopwatch, a mouse click, a flame whoosh. For a book shut, a glass breaking and a punch the fuller recording is a
  later take (another `seed`), not the first.
- **The newest learned things are measured, not heard.** The 60 things and 9 beds learned last (a chainsaw to a
  tree falling, hail to a gale) are clean at their first seed and their takes stay as close to their recordings
  as two takes to each other, but no one has judged them by ear; 36 of the 60 rest only on Freesound's MP3
  previews, whose floor is higher than a lossless recording's. Machines are whole actions of 4 to 10 s (started,
  run, stopped), not loops; hail, a dishwasher, a washing machine, a fridge and a shower are beds. A tornado or a
  hurricane gets the gale of a village, the strongest real wind in the store; the washing machine is a washer-dryer
  turning, not a spin cycle at full speed.
- **Performances.** Only the learned things whose index names a pattern or holds a recording of the whole action
  (walks, knocks, typing, coins, keys, clinking glasses ...) can be played as the action; the reader decides from
  the words (a plural, an action word, a count or a length), and `events` overrides it. A walk or a typing run made
  from single recordings repeats the few recordings the thing has.
- **The real instruments** have 14 to 45 recorded notes each; a note between them is a neighbour moved by a few
  semitones, and the takes of one instrument can differ in level by 2-3 dB from note to note.
- **Invented realistic sounds** remain in the catalogue for things no learned design covers (jumps, landings, game
  hits, shots, magic). Their kind was judged poor by ear. A learned design wins a tie with them, but a request that
  names only an invented one gets it.
- **Instrument notes and bells** from models were judged weak by ear, bells poor; figures on instruments were judged
  only fair as interface signals, which is why interface and motion requests get no notes unless the words ask for
  music. Figures play a real marimba unless asked otherwise.
- **`--listen` and the atlas** were not re-measured on the v1 catalogue; the atlas is built from each design's
  default render and is stale after designs change until it is rebuilt (`python -m earshot.sounds.atlas build
  --work DIR`, repeated until done).

**Requests.** The reader is a lexicon and rules, not a language model (`--listen` adds CLAP). On 220 held-out
requests by someone who never saw the catalogue, scored once on an earlier reader, the first pick was right for
69.1 % and one of the top three for 76.4 % ([VALIDATION.md](VALIDATION.md#understanding-requests)). It reads
English; German only through a table of common sound words. Requests that name the thing, the job or the sound
did far better than long descriptions of a sound and casual paraphrases. A learned design answers only when its
thing is named, so "a heavy impact" never reaches a recorded dumbbell, and a feeling alone reaches no recording.
The guess note catches only part of the wrong picks, so a pick without it can still be wrong: read the design id
against the job. `make --set` and `make_set` rank by words only, even in the MCP server. One request is one sound; a
set is a spec.

**Formats.** Ogg Vorbis does not play in Safari before 18.4: for the web write `webm,mp3` (or `webm,m4a`). An mp3 or
m4a loop has a gap at its seam in players that ignore the encoder's padding note: a single `make` to mp3 warns, and
a set with several formats writes its loops as flac in place of mp3 / m4a; loops belong in ogg, webm, wav or flac.
The sprite holds every sound in every sprite format, so a loop inside an mp3 sprite keeps the gap. ffmpeg 8 prints
"Error parsing Opus packet header" when it decodes any webm / Opus file it wrote itself (a plain sine tone too); the
files decode in full, so the message is not a sign of a broken file. Re-running a spec into the same folder rewrites
its files and `manifest.json` but does not delete the files of names removed from the spec.

**Engineering.** No models and no network at render time; the base install is numpy, scipy, soundfile, soxr and
numba, and the listening stack (torch, transformers) comes only with the `listen` extra. ogg, mp3, m4a and webm need
ffmpeg on PATH. A learned bed renders in 1.0-1.7 s of CPU per second of sound; a recipe of many layers takes up to
half a minute on one core. The learned store adds 17.93 MB, the textures 9.46 MB, the drums 1.31 MB and the
instrument notes 0.83 MB to the package. The recordings, the listening pages, the request sets and the tool that
builds the learned store are not in the repository: the store can be used and credited, not rebuilt, from the
repository alone, and the request numbers cannot be re-run from it.

**Licence of the learned material.** Every learned model comes from a recording in the public domain: BigSoundBank,
Kenney, VCSL, VSCO-2 CE, Karoryfer and CC0 uploads on Freesound, all CC0 or an explicit public-domain dedication
(the Freesound ones partly read from MP3 previews found through the Openverse catalogue). Each recording's licence
was taken from its source (the sound's page, the pack's licence file, or Openverse's licence field for a Freesound
page) and is written next to it in the four credits files (`earshot/data/resyn/`, `textures/`, `drums/` and
`notes/`);
earshot cannot prove that an uploader had the right to give a recording away.

## Building sounds (`build`)

`earshot build` makes a sound from recorded parts; it is not a way to any sound. What the measurements found, with
the listening model as the judge ([BUILD.md](BUILD.md#how-well-it-does)):
- **It works for textures of many similar events and for pulsing.** Rebuilt from other recordings, a purr, a swarm
  of bees, a boiling pot, a campfire and a helicopter came out nearest their own real design among the 922;
  applause and a brook among the nearest three.
- **It does not work for rain or hail on a new surface.** Close single drops scattered never became the dense far
  wash of real rain: the real designs of rain on a tin roof, on a window, heavy rain and hail came out far down the
  list (places 17 to 124).
- **Nor for a vehicle passing built from another motor.** `pass` carries a recording by and bends its pitch, but the
  material is not the vehicle: a car, a motorcycle and a jet built so were far from their real designs.
- **Nor for voices or pure tones.** There is no voice material: a whale is an owl and a wolf moved down, a whisper
  has nothing to start from. A beep or an alarm is a recorded beeper or horn, never a synthesized tone.
- **A part is found by its words** and can come back as the wrong recording; the `part N` lines say which design each
  part became and the next nearest, and `sound=<id>` swaps it.
- **The fit is a listening model, not a person's ear.** The tuner raises that model's fit, and a sound tuned to it
  can please it more than a person: builds fit their words as well as real recordings do. The second ear (the
  AudioSet tagger), which the tuner never sees, takes some "sounds like it" verdicts back, but it also objected to 6
  of 27 real recordings (a kettle heard as a train), hears only up to 8 kHz and has no class for most made-up things.
  An objection is a reason to listen, and "sounds like it" is not a verdict by ear.
- **Naming the parts is not always better.** On twelve missing things the parts an agent named fit no better on
  average than the automatic draft: they won where the structure was known and the material existed, and lost where
  a part was found by the wrong word or no material existed.
- **Time.** A build takes about a minute after the listening model has loaded (20 to 60 s once per process); a build
  of several named parts with long beds one to two minutes.
