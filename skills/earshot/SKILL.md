---
name: earshot
description: Make sounds with earshot whenever a product, game, page, animation or film needs a sound and no recording is at hand - interface sounds (taps, toggles, hovers, typing, success, error, notifications), everyday foley for games, 3D and video (footsteps, doors, drawers, switches, keys, cloth, glass, water, metal, swings, punches, explosions), ambience beds (fire, sea, rain, wind, crickets, a cafe, a street), game sounds (coins, fanfares, 8-bit), drums, motion-design effects and a few short music loops, one at a time (earshot make) or as one family (earshot make --set / make_set). Listen with earshot whenever you must judge, choose, trim, verify or compare sound you cannot hear - picking a sound effect, music or ambience from candidates, finding the cleanest moment in a recording, checking whether audio reads as a real recording or as code-synthesized, reviewing a soundtrack or a video's audio (what happens when, sync to picture, loudness, mix), A/B-ing a mix change, searching a recipe by listening (earshot forge), or giving a code-made film its sound effects from the shipped bank (earshot sfx / kit). Use it before telling anyone how audio sounds.
---

# earshot: ears for an agent

## Making sounds
When a product, game, page or film needs a sound and no recording is given, make it: earshot picks a design from its
catalogue and renders a new take (no models, about a second for a one-shot, nothing downloaded). This part is
complete on its own.

**When a sound, and when none.** Most moments get none. A sound belongs where something happened that a person
should notice without looking at it. What follows is common practice; only the motion video numbers further down
are measured.
- A page or an app: a sound for what the user did on purpose and for what came of it (a switch thrown, a form
  sent, a payment gone through, an error, a message arriving). None for what happens by itself or all the time: a
  page loading (a browser plays nothing before the first click anyway), scrolling, an animation, the pointer
  crossing the links of a marketing page. A page that changes after a click has had its sound with the click. A
  landing page is served by three to six of the `ui` roles, not by all 17; `hover` is for tools in which hovering
  is a choice, and it is the set's quietest sound.
- A game: every action the player repeats gets a short sound in several takes (steps, jumps, hits, coins), a
  reward and the end of a level get the longer figures on an instrument, a menu gets one small sound for moving and
  one for choosing.
- A motion video: two to five effects in a film that music leads ("Scoring a motion video" below says which).
- A scene of a film, or a clip of real footage: one bed for the place (rain, a room, a street), then the real sound
  of each thing the picture shows being done (a door, steps, keys, a pour, a sizzle), asked for as that thing. A
  designed sound (a whoosh, a pop) only on the few cuts or captions the edit wants felt. A thing the catalogue has
  no recording of (the answer says so) is better recorded or left silent than replaced by a stand-in.

**What the catalogue holds.** Realistic things are learned from recordings in the public domain (235 designs:
footsteps on eight surfaces, doors, drawers, locks, switches, keyboards, a mouse click, a camera shutter, cloth,
paper, books, glass, ice, water, metal, swings, punches, falls, explosions, animals such as a dog, a cat and birds,
people without words such as applause, laughter and a cough, a car horn, a siren, a phone ringing or vibrating
...) and 13 ambience beds learned from
real textures (fire, three shores, crickets, a brook, a cafe, a street, distant city, room tone, underwater, rain on
a terrace, wind in dry leaves). The interface sounds are things a hand does, from those recordings: a key, a rocker
switch, a page, a pen cap and air, knocks, a pop, a pen click, a stapler, crumpled paper. The film and motion
effects are short, dry hits, thumps, ticks and whooshes of air built from recordings. The acoustic drum kit and hand
percussion are learned too (130 recorded strokes; seeds walk through a drum's strokes, `energy` picks softer or
harder ones), and so are the instruments that play the tonal figures when music is asked for: a marimba (the
default voice), a harp, pizzicato strings and an upright piano (`felt_piano`), 97 recorded single notes. Only
numbers ship, no audio file: every sound is computed when it is asked for. Written in code: the figures themselves
(success, error, notify, coin, level up: the notes, their timing and the room), chip sound, the drum-machine voices,
a brush snare, a finger snap and stick counts, rain on a window and a forest, the other instruments' notes, pads,
three music loops, and the film bank.
Realistic things that no one recorded (jumps, magic, game hits) are invented designs and were rated low by the
owner: prefer a learned thing when one fits.

**The three calls** (CLI, or the MCP tools of the same names; MCP paths absolute):
1. One sound: `earshot make "WORDS" -o sounds/success.ogg` (MCP `make`: `want`, `out`; optional `sound`, `n`,
   `theme`, `pitch`, `dur`, `voice`, `material`, `loop`, `events`, `seed`). One call per sound.
2. Several that belong together (an interface, a game, a film's kit): ONE call, never a loop of `make`.
   `earshot make --set ui|game|arcade|film -o FOLDER --format webm,mp3 --sprite sprite`, or MCP `make_set` with
   `set`, `theme`, `format`, `sprite` and your own `sounds: [{"name": "tap", "want": "a soft tap"}, ...]` (fields:
   name, want, sound, pitch, dur, voice, material, seed, n, axes, lufs, events, loop). One voice, key and room for
   the tonal sounds; every one-shot as loud as made alone, within 2.5 dB of the others; loops and beds levelled by
   kind; `manifest.json` written. Themes: soft (the default), glass, wood, felt, tine, metal, electric, digital,
   retro. From the CLI your own list is a JSON file: `{"set": "ui", "format": "webm,mp3",
   "sounds": [{"name": "tap", "want": "a soft tap"}, ...]}`, then `earshot make --spec FILE -o FOLDER` (a re-run
   rewrites the files and `manifest.json`; it does not delete files of names you removed).
   Your own `sounds` come on top of the set named in `set` (the 17 of `ui` and yours; one with a name of the set
   replaces that sound). Leave `set` out and only your list is made: that is the call for the handful of sounds a
   landing page or a scene needs. An entry's `events` and `loop` are the single call's `--events` and `--no-loop`
   (`{"name": "steps", "want": "footsteps on wet pavement", "events": 4}` answered `step/tile (2.28 s ...)`,
   `played: 4 events as a walk`; `"loop": false` makes a bed once); a count, a length and "no loop" said in the
   `want` do the same. An unknown field is an error that lists the fields. A spec of five entries without a set:
   ```
   $ earshot make --spec landing.json -o sounds
   5 sounds in sounds (wav), manifest.json
   open.wav: open/air (0.36 s, -18 LUFS, clean) [opening]
   toast.wav: notify/knock (0.40 s, -19 LUFS, clean) [toast notification]
   error.wav: error/thud (0.49 s, -18 LUFS, clean) [error]
     not understood: address
   card.wav: thump/soft (0.09 s, -24 LUFS, clean) [thump, card, settles]
     not understood: pricing
   paid.wav: success/click (0.31 s, -17 LUFS, clean) [payment success]
   ```
   (the `want`s were "a modal opening", "a toast notification", "an error sound for a wrong email address", "a soft
   low thump when the pricing card settles", "a payment success sound").
3. The catalogue, only when a pick was wrong and `also` does not help: `earshot sounds --search WORDS` (top 20) or
   `earshot sounds --family step` lists designs with their length and what they are; then make the one you chose
   with `earshot make "WORDS" --sound ID` (`sounds` itself takes no `--sound`). Never list the whole catalogue
   (hundreds of lines).

**The ready-made sets.** `ui` (17), every role a normal sound, no instrument: tap a mechanical key
(keyboard/mechanical), hover a fingertip barely touching wood (hover/fingertip), toggle_on and toggle_off a rocker
switch (toggle/rocker_on smaller and brighter, toggle/rocker_off bigger and duller), type a desktop key (type/key),
next and back a page flicked forward and dropped back (next/page, back/page), open a pen cap pulled and air let out
(open/air), close air drawn in onto the cap (close/air), send a stick swung through the air (send/swoosh), receive a
pop of the lips (receive/pop), notify two light knocks (notify/knock), success a pen clicked in two parts
(success/click), complete a stapler pressed down (complete/stamp), warning two firm knocks on a door
(warning/knock), error two dull thuds of a bag (error/thud), delete paper crumpled (delete/crumple). A request for
one of these jobs in words gets the same design. With a theme named, `ui` plays the tonal figures in that theme's
voice instead.
`game` (22: menus, rewards, jingles, and invented action sounds: replace jump, land, hit, explosion, door with
learned ones in a spec where a real sound matters, e.g. `{"name": "door", "sound": "door/shut"}`; the others are
`step/floorboard`, `person/fall`, `punch/body`, `explosion/near`, `splash/plunge`), `arcade` (16 chip
sounds), `film` (motion-design starter; whooshes, hits and risers layered from learned film recordings).
A game's rewards and results are figures on an instrument without being asked: "a coin pickup" is glockenspiel
notes of 0.7 s (`coin/bright`), "a level complete fanfare" 2.9 s on a marimba (`win/fanfare`). One enemy defeated
is a blow (`hit/crit`, 0.6 s), a boss defeated the fanfare.

**How to ask.** Name the thing, the sound or the job first, then feelings, material, numbers, key:
`"a footstep on gravel"`, `"a wooden drawer closing"`, `"a camera shutter"`, `"a soft success chime for a finance
app"`, `"coin pickup, 8-bit"`, `"waves on a beach, 30 seconds"`, `"a wooden notification, 400 ms, in D minor"`.
The moment may follow the sound: "a soft thump when the first headline lands" (thump/soft), "a tiny tick when new
content drops into the feed after refreshing" (tick/soft); keep the sound or the job as the first thing named and
the moment after "when", "as" or "each time", or the words of the moment are taken for the sound.
A learned sound answers only when its thing is named: "a heavy impact" gets an invented boom, "a dumbbell dropped"
the recording. Feelings: "softer", "darker", "bigger", "longer", "drier", "not harsh". Interface and motion
requests get normal sounds; notes on an instrument answer only when the words ask for music (an instrument, a
chime, a jingle, a melody, a motif) or for chip sound. An instrument named as the player sets the voice of a
figure: "a notification on a harp", "a success sound with pizzicato strings", "an error sound on a soft piano" (or
`voice`: marimba, harp, pizzicato, felt_piano, kalimba, glockenspiel ...); a theme sets one voice for many calls. A
material picks the voice of a tonal sound (wooden: marimba); a bell or glass timbre only when you ask for one.
**One event or the action.** A thing a person does several times in a row (footsteps, knocking, typing, coins,
keys, clinking glasses) is played as the action when the words ask for it: "footsteps on gravel", "knocking",
"three knocks", "typing on a keyboard", "5 seconds of typing". "a footstep on gravel" or "a single knock on a door"
is one event. `--events N` (MCP `events`) overrides: 1 = one event, 2 or more = the action with N events. A
counted action in a row is one file whatever the design ("four knocks on the table", "three claps", "a countdown
3 2 1"); a count of separate things is that many files ("five coin pickups", "3 variations"). A length with an
action ("three knocks, 2 seconds", "footsteps, 5 seconds") is the file's length, the events spread over it; a
countdown stays one a second. "clapping", "clicking", "beeping" (the action form of a row word as the sound) are a
row at the pattern's pace, over the length asked or about 2 to 3 s; scattered and shaken actions ("ice cubes in a
glass, 3 seconds", "keys jingling, 1 second") keep the length too. The answer's `played: ...` line says what was
played. Chip sound only with "8-bit" /
"retro" / "chiptune". Naming the thing or the job does far better than a mood or a paraphrase; on held-out requests
written that loosely the first pick was right 69.1 % of the time (76.4 % in the top three). To steer, re-ask with
adjectives, not by browsing.

**Seeds are variations.** For a learned design, seeds 0, 1, 2 ... play its different recordings first, the best one
at seed 0 (three seeds
of `step/pebbles` are three real footsteps), then new takes of them. Ask for 6-12 takes of anything a game repeats
(`--n 8`, or "eight variations"), and make them in a set (`"n": 8` in a spec) so they share one level: single takes
of different recordings differ in length and level. `--json` gives the number of the recording each take played
(`member`).

**Read the answer** (no extra call; ask for `--json` / `full=true` only if you need measurements):
- `x.ogg: design/id (0.88 s, G5, marimba, -14 LUFS, clean) [words]`: report this line. `audit: ...` instead of
  `clean` means not usable: make it again with another `seed`.
- `also: a, b, c`: alternatives. If the chosen id does not fit the job, call again with `--sound ID`.
- `also (the pick is a guess: ...): id (a few words), ...`: treat the pick as unverified; choose an alternative
  by its words and call again with `--sound ID`, or re-ask naming the sound or the job.
- `same design as NAME: ... (the nearest: ID)` (sets): two names got one sound; give the second `"sound": ID` in
  your spec. `another recording of ID than NAME: ...` is not a problem: the two play different recordings of one
  learned design.
- `not in this sound: WORDS`: part of the request is not in this file (one call, one sound). Read the words: a
  second thing or job wants its own call; a verb or an adjective of the same thing is no missing sound ("a
  checkbox being ticked" answered `toggle/rocker_on (0.22 s, -19 LUFS, clean) [checkbox]` and `not in this sound:
  ticked`: the switch is the whole answer).
- `not followed: ...` or `0.43 s, not the 2 s asked: ...`: the design has no such control or could not keep that
  length, and the line says what it does instead; fine unless it mattered, then pick another design.
- `a cut of 4 s, not a loop: ...` / `a seamless loop of 8.00 s, ...` / `8.57 s: a music loop is whole bars ...`:
  the length asked for an ambience or a loop, and what was made of it; ask "a seamless loop of ..." when the
  sound must repeat without a seam.
- `3 knocks 1.5 s apart: 3.25 s, not the 10 s asked (...)`: a counted action could not fill the length at a pace a
  person keeps (or a countdown at one a second); the file is as long as the line says.
- `not understood: WORDS`: those words did nothing; rephrase plainly if they mattered.
- an error `no sound in the catalogue answers '...' (earshot makes no THING): ...`: there is no such thing; say so
  and offer the nearest thing `earshot sounds --search` finds, never a stand-in passed off as the thing.
- `recording 2 of 3` (in `--json` `member`, a performance and the manifest): which of a learned design's recordings
  played; the recording's own title is only in the store's `CREDITS.md`.
- `... a loop in mp3 has a gap at its seam ...`: rewrite that loop as ogg, webm, wav or flac.

**How long, how loud, and where a file starts.** A timeline or a mix needs these three; the answer's line carries
the first two.
- Length (seed 0, read off real answers): an interface sound is 0.2 to 0.6 s (toggle 0.22, success 0.31, hover
  0.34, notify 0.40, error 0.49, delete 0.61); in a motion video a tick is 0.04 s, a thump 0.06 to 0.09 s, a logo
  hit 0.29 s, a whoosh 0.39 s, a riser 2.5 s; a game's figures run from 0.7 s (a coin) to 2.9 s (the win fanfare;
  game over 2.1 s, a power-up 2.8 s); real things take their time: a door falling shut 2.5 s, a phone vibrating
  1.6 s, keys jingling 2.4 s, distant thunder 7.7 s, six footsteps 3.5 s; a bed is 20 s unless a length is asked.
  Seconds in the words, or `--dur`, set the length of a whoosh, a riser, a figure, a bed and an action
  ("footsteps on concrete, 4 seconds" answered `step/concrete (4.00 s ...)`, `played: 7 events as a walk`); a
  single struck sound rings as it rings and says so.
- Level: a single `make` makes each file as loud as its kind allows (true peak -1 dBFS, at most -14 LUFS, a bed
  -22), so single files are not a mix. `--lufs L` (`"lufs"` in a spec entry) sets one file's loudness:
  `earshot make "a logo hit" -o logo.wav --lufs -24` answered `logo.wav: boom/short (0.29 s, -24 LUFS, clean)
  [logo, hit]`, the plain call -17 LUFS. Sounds made as ONE spec are levelled against each other by kind, a first
  balance to mix from. A scene's spec of five entries answered `rain.wav: rain/street (7.93 s, -30 LUFS, clean)`,
  `door.wav: door/shut (2.54 s, -20 LUFS, clean)`, `thunder.wav: thunder/distant (7.70 s, -21 LUFS, clean)`,
  `keys.wav: keys/bunch (2.38 s, -14 LUFS, clean)`, `phone.wav: phone/vibrate (1.64 s, -15 LUFS, clean)`: the bed
  sits 9 to 16 LU under the things done in it. Against a music bed the motion section below has measured numbers.
- Start: every file starts when its sound does (within 2 ms) and ends faded, but its loudest moment is not always
  its start. Measured on files made for this text: `thump/soft` (0.09 s) is loudest 0.02 s in; `boom/short`
  (0.29 s) 0.15 s in, after its short swell of air, so its file starts about 80 ms before the frame the hit lands
  on; `whoosh/soft` (0.39 s) at 0.24 s, three fifths through; `rise/air` asked with `--dur 1.2` is loudest at 1.16
  of its 1.17 s, so a riser starts its own length before the reveal. `--json` gives a take's length, peak level
  and loudness, not the time of its loudest moment. earshot does not put files on a timeline: the editor, ffmpeg
  or the page's code does, and `earshot timeline mix.wav --video film.mp4 --cues cues.json` (Loop 2) then shows
  whether each cue has its sound.

**Formats.** The extension picks it (wav, flac, ogg, webm, mp3, m4a; all but wav and flac need ffmpeg). Web:
`earshot make --set ui -o public/sounds --format webm,mp3 --sprite sprite`, then load `sprite.json` (howler.js
format, `src` lists webm first); Ogg Vorbis does not play in Safari before 18.4.
**Loops**: ambience and the music loops loop seamlessly by default (`--dur` sets the length; a learned bed has no
loop point at all); say "no loop" (or `--no-loop`) for a film's bed. Never ship a loop as mp3 or m4a: a set with
`webm,mp3` writes loops as flac instead (`a loop: flac instead of mp3` is expected, not an error). A learned bed
renders in about 1-2 s per second of sound: a 30 s bed takes up to a minute.
**`--listen`** (`listen=true`): also ranks by how designs sound and adds `heard as: ... (z)`; it loads a model
(20-40 s). Use it only when a describe-the-sound request came back as a guess.

**What not to expect.** No music beyond three short loops (`bed/warm`, `bed/playful`, `bed/8bit`): earshot is for
sound effects. No speech. No sound for a thing that is neither in the catalogue nor recorded: the answer is an
error that says so (often "earshot makes no X"). The film and motion sounds (whooshes, hits, thumps, ticks,
risers, the braam) are built from learned recordings; the owner rated the fast whoosh 4 of 5 and the deep whoosh
and the braam 3 to 3.5; the hits, thumps, risers, the tape stop and the suck-in impact were rebuilt after his
ratings and are not rated yet; so are the interface sounds without instruments, the relearned ambiences and the
animals, people, street and house sounds (measured, no listener yet). The drums invented from physics and the
invented realistic sounds were rated low by
the owner. Do not say how a made sound sounds beyond its descriptions: no one but the owner has listened, and earshot's
models did not predict his ratings. Say plainly that a learned sound is a new take of a public-domain recording,
not a sound invented from nothing.

**Scoring a motion video** (a launch film, product demo, explainer or social clip). Measured on 39 reference motion
design videos (2026-11-18, `docs/motion-sfx-study.md`); [measured] is a number from them, [convention] is common
practice they did not test.
- Music carries the film. 34 of the 35 videos with sound are music-led, about 11 have a voice-over, one is scored with
  effects alone [measured]. In 21 of the 35 no effect stands out of the music at all; where effects do stand out,
  a music-led video has 1 to 4 of them, 0.2 to 1.3 per 10 seconds [measured]. Do not put a sound on every cut:
  pick the two to five moments that matter (a reveal, the logo, a key UI action, a title change).
- What gets a sound, and what to ask for (each phrase checked with the reader today):
  - a stop: the most used move [measured: 47 in 16 videos]. Cut the music and everything for 0.15 to 0.5 s just
    before a reveal or a title, then come back on a hit. It is an edit of the bed, not a file.
  - the hit after the stop: ask "an impact after silence" or "a drop in the video" (hit/drop, about 0.25 s: a
    heavy body with a puff of air).
  - something appears or a card flips: a short low thump, about 50 ms, 130 to 320 Hz, dry [measured].
    Ask "a pop when an element appears", "a card flip" or "an element pops in on screen" (thump/appear, 0.06 s);
    "a soft thump" (thump/soft, 0.09 s) for something landing or settling ("a soft low thump when the pricing card
    settles" answered thump/soft). In a run, one per element, about 200 ms apart: make the thump once (`--n 3`
    gives three takes of it) and place it once per element, so the gaps are the picture's.
  - a UI click, a cursor press, a word highlighted: a bright dry tick, 7 to 10 kHz, nothing under 1 kHz, mono, a
    few tens of ms [measured]. Ask "a cursor click" or "a soft ui tick" (tick/soft, 0.04 s); "a mouse click"
    (click/mouse) is the whole mouse button, 0.2 s.
  - a number counting up [not in the measurements]: a row of the same tick as one file, asked with a count and a
    length. `earshot make "a soft ui tick" -o count.wav --events 8 --dur 1.5` answered `count.wav: tick/soft
    (1.50 s, -15 LUFS, clean) [ui tick]` and `played: 8 events as a row at 6.2 a second`. Ticks, clicks and blips
    follow as close as their own length lets them: `--events 12` on "a counter ticking up, 1.5 seconds" answered
    `tick/metal (1.50 s, ...)`; knocks and steps a hand or foot plays keep 0.12 s at the closest.
  - where the hit is: every take says `peak_s` (the loudest moment; also on the plain line as "peak at"),
    `onset_s` (where it starts to be heard) and for a row `events_s` (each event's start), in seconds from the start
    of the file. The hit goes on the frame at t: start the file at t minus peak_s. "a whoosh for a transition"
    answered `whoosh/soft (0.39 s, peak at 0.24 s, ...)`: for its peak on 12.000 s, start it at 11.76 s.
  - a title card changes, a big transition: a mid-low hit of about 0.3 s that swells in over about 0.1 s, half of
    it under 100 Hz [measured]. Ask "a title hit" or "a title slam" (slam/title: a low body in a short burst of
    air, about a third of a second, then its air dies away).
  - the logo end card: the logo lands on a short low hit, 0.2 to 0.3 s, body around 200 to 300 Hz, fast end
    [measured: 9 of 11 end cards looked at; in some it may be the track's own last hit]. Ask "a logo hit", "a
    logo reveal hit" or "a short boom" (boom/short, about 0.3 s) on the logo frame and let the music ring out under
    it; "a logo reveal" (sting/reveal, 0.8 s) swells into the same hit; "a deep boom" (boom/deep) is darker.
  - a move across the screen: a noise whoosh, mono, as long as the move (0.3 to 1 s), falling off fast after its
    peak [measured, mostly one video]. Ask "a whoosh for a transition" (whoosh/soft, 0.4 s; it holds a length
    asked: "a short whoosh, 0.3 seconds") or "a fast whoosh" (whoosh/fast, rated 4 of 5 by the owner); "a deep
    whoosh" for a big slow move. A small fast move: "a quick swish" or "a soft swish" (flick/thin, 0.2 s).
  - a text or image breaking apart: "a glitch transition" (glitch/short, 0.3 s) [measured: 3 in 3 videos].
  - a reveal after a build: "a reverse swell into a cut" (rise/reverse, about 1.4 s, ends at its peak); put its
    end on the reveal frame [measured: once]. "a riser before the drop" (rise/air, 2.5 s) is air growing to a cut.
  - a sparkle on a product: "sparkle on the product reveal" (shine/air: bright air, no note).
  - typing on screen: "typing on a keyboard, 2 seconds" for a real keyboard; the designed typing in the
    references is brighter (keys around 9 kHz) than this recording [measured].
- What does not get a sound: small motion that rides the music (text easing in word by word, icons drifting,
  backgrounds), and most cuts in a music-led film. The editors cut the picture on the music's accents and let the
  music carry the cut [measured: 8 of 9 hard cuts in random windows have an onset within 150 ms, 6 of them on
  the beat grid]. So cut on the music's beats first; add an effect only where the music has no accent.
- Never as an effect in a motion video: marimba, bells, chimes, glockenspiel, piano or synth notes, unless the video
  is about music or is a game [measured: no instrument note used as an effect in 35 videos; the owner's rule].
  Not used in any of the 39 either: braams, explosions, fire whooshes, tape stops, impacts with a reverse suck-in,
  long risers [measured]. Use them only when the picture shows that thing.
- Where the sound sits against the frame [measured, few cases]: a hit starts about 80 ms before the picture
  changes and peaks about 60 ms after it; the hit after a stop lands about 50 ms after the change; a whoosh peaks
  on the fastest frame of the move [convention]. Where the loudest moment of a made file lies is measured above
  ("How long, how loud, and where a file starts"): `boom/short` 0.15 s in, `whoosh/soft` three fifths through.
- Levels against the music bed [measured]: hits about 18 dB over the music's median level at that moment, a swish
  about 20 dB, the hit after a stop and word ticks about 8 dB, a logo hit as loud as the music was (the music has
  thinned or stopped). Clicks, whooshes and hits near mono; thumps slightly wide; everything dry, no reverb tail.
  A made file's loudness is in its answer's line; set it with `--lufs`, or with a gain in the mix, after measuring
  the music at the cue (`earshot measure`, or `kit.bed_level` in Loop 5).
- Endings [measured]: 26 of 35 end at body level (the last 3 s within 3 LU of the whole), 7 fade out. End with the
  logo hit and the music's last note, no long silence.

**Scoring a scene or a clip of real footage** (a film scene, a cooking short). Nothing here is measured; the
requests are quoted as they were answered on 2026-11-18. Ask for each thing the picture shows as that thing, with
its place or surface, and make them as one spec for a first balance (above): "steady rain on a terrace, 8
seconds, no loop" (rain/street), "the heavy wooden front door falls shut" (door/shut, 2.5 s), "keys jingling"
(keys/bunch), "a phone vibrating on a table" (phone/vibrate), "distant thunder" (thunder/distant, 7.7 s),
"footsteps on concrete, 4 seconds" (step/concrete, 7 steps), "pouring water into a glass" (drink/pour, 4.9 s). A
bed loops unless the words say "no loop". The catalogue has no recording of many kitchen and street things. Then
either the answer says it is a guess ("a whisk beating eggs in a bowl": `earshot makes no whisk; water/drop is
only the nearest sound it has`), or an app's sound answers for the real thing without a doubt: "a kitchen timer
ringing" answered `timer/done (1.68 s, A5, marimba ...)`, three notes on a marimba, an app's timer and not a
kitchen's bell. Read what the picked design is before it goes under a picture (`earshot sounds --search "kitchen
timer"` prints each design's words), and leave a moment silent or record it rather than use a stand-in.

Everything earshot renders is free to use in any product, commercial too, without credit. The learned material
comes from CC0 recordings credited in `earshot/data/resyn/CREDITS.md`, `earshot/data/textures/CREDITS.md`,
`earshot/data/drums/CREDITS.md` and `earshot/data/notes/CREDITS.md`.

The rest of this file is about listening: judging, choosing and checking sound you cannot hear.

## Listen
You cannot hear. Never say how audio sounds, or that a sound "fits", without an earshot reading behind it, and quote
the number when you report ("strong match, z +3.9; p_real 0.96; -14.2 LUFS"). earshot runs locally on CPU (audio
never leaves the machine, except with the optional `llm` command) and caches every embedding and result on disk.
Cost: each CLI call is a new process that loads the models first (~20-40 s on a busy laptop, even for a cached file);
inside the MCP server or one Python script the models load once and repeat calls take seconds. For more than a
handful of calls, use MCP or a short script (`from earshot import api; api.find(...); api.realness(...)`).

## Setup (once)
```
pip install -e "/path/to/earshot[mcp,llm]"   # or the repo venv: /path/to/earshot/.venv/bin/python -m earshot ...
earshot doctor                           # packages, ffmpeg, models (pinned revision, cache state); --load times them
```
Two ways in, same functions:
- **CLI**: `earshot <command> ... [--json] [--out result.json]`. Text output starts with a plain summary; times are
  m:ss.ss in text and seconds in JSON.
- **MCP** (models stay loaded between calls): `claude mcp add earshot --scope user -e EARSHOT_THREADS=3 --
  /path/to/earshot/.venv/bin/python -m earshot mcp`, or in any client's `.mcp.json`:
  `{"mcpServers": {"earshot": {"command": "/path/to/earshot/.venv/bin/python", "args": ["-m", "earshot", "mcp"],
  "env": {"EARSHOT_THREADS": "3", "GEMINI_API_KEY": "${GEMINI_API_KEY:-}"}}}}` (the key line only for the optional
  llm tool; Claude Code fills it from the user's environment, other clients may need the key set in the environment
  that launches them: never write a literal key into a project file).
  Tools: make, make_set, sounds (no models; above); measure, describe, match, rank, find, compare, ab, realness,
  timeline, sheet, critique, doctor, llm_critique; the sound bank (no models): sfx_list, sfx_render, kit_render
  (Loop 5). Pass absolute paths; `full=true` returns all JSON. The server answers within a few seconds, longer on a
  cold or
  busy machine; if Claude Code shows no earshot tools, start it with `MCP_TIMEOUT=60000` (PowerShell:
  `$env:MCP_TIMEOUT=60000; claude`). `earshot mcp --warm` preloads the models at startup.
If the machine is shared, set `EARSHOT_THREADS=2-3` (or `--threads 2` on any command). The first model call of a
process loads CLAP + AST (~20-60 s); later calls in the same process are fast. Any ffmpeg-readable file works,
including the audio of an mp4 (a video with no audio track says so). Errors are one line on stderr:
`earshot <command>: error: ...`, exit 1.

## Loop 1: choose an asset
1. `earshot rank "a soft felt mallet hit on a wooden block" downloads/sfx/ --top 5` - shortlist from a folder.
2. `earshot find winner.ogg "a single clean mallet hit" --top 3` - the best moments as sample trims (start / end in
   seconds). Each segment has a kind: `stroke` (one onset until it decays) and `event` (strokes that belong together,
   e.g. a key's down + up) start 5 ms before the attack and end after the decay; `window` is a stretch whose edges
   find snapped onto the sound for one-shot queries (a plain stretch, to be crossfaded, for beds). `head` / `tail` =
   how loud the first 5 ms / last 10 ms are against the trim's peak (-30 dB or lower = clean); a trim marked
   'head cut' or 'tail cut' starts or ends inside a sound: do not use it as is. `--kind event` returns onset-based
   trims only (hits, clicks, steps), `--kind texture` plain windows only (beds); the default reads the query.
3. `earshot describe winner.ogg --start 1.12 --end 1.48` - what it actually sounds like; check the top tags fit what
   you need and the shape (attack / decay) fits the picture. AudioSet labels are weak on one-shots under ~0.5 s:
   'tagger unsure' or a near class (Zipper for a swish, Finger snapping for a click) is neutral; only an outright
   different class (Speech, Music, Engine, Vehicle) is a red flag.
4. `earshot realness winner.ogg --start 1.12 --end 1.48` (or a whole folder) - judge the trim you will use: a
   multi-take file's p_real is a mean over every take and the silence between them. When the user wants real or
   professionally produced sound, do not use a 'likely code-synthesized' trim, and read the evidence of an
   'uncertain' one before using it (write why you kept it). When synthesized sound is the point (procedural music, UI
   tones, sounds made with earshot forge), realness is information, not a gate. The verdict thresholds depend on
   length (each summary names them): a sound up to 6 s uses the clip thresholds (lo 0.17 / hi 0.77), a longer file
   the mean of its 4 s windows with weakly calibrated file thresholds (lo 0.40 / hi 0.77), so compare verdicts, not
   raw p_real, across files of different lengths. Real recordings and professionally produced music and sound design
   usually read real, but realness hears how a sound sounds, not where it came from: clean professionally synthesized
   tones (UI sounds, chimes, synth stabs) also read as code-synthesized. If realness is unavailable (model missing),
   say plainly that real-vs-synthetic was not checked; describe's AudioSet classes (Sine wave, Beep, Synthesizer) are
   weaker evidence. Exact digital zeros alone prove nothing (59% of the project's licensed real recordings have some
   edit or fade padding).
5. To choose between two candidates: `earshot ab a.ogg b.ogg --query "a crisp fast air whoosh"`, with
   `--a-start/--a-end/--b-start/--b-end` for trims from find (A and B may be the same multi-take file). Optional:
   `earshot match winner.ogg "a deep thocky key press" --alts "a clicky blue switch" "a typewriter key"` to test one
   claim, or `earshot compare winner.ogg --refs approved/ [--start --end]` to check it fits the palette.

## Loop 2: review a mix or a film
1. `earshot timeline film.mp4 --cues cues.json` (or `soundtrack.wav --video film.mp4`) - the listening transcript:
   events with time, level and tags, sections, loudness curve, picture cuts, sync per cut / cue. `--detail low` is
   fast, `high` tags up to 96 events (most important first; the summary says how many were tagged, so an untagged
   event is not a missing sound). Read the sync stats: offset_ms = sound minus picture (negative = early), and the
   'chance' fraction says what random edit points would score; only better-than-chance sync means anything. A cut
   listed with no hit can still carry a soft whoosh: check it on the sheet before adding a sound.
2. `earshot sheet film.mp4 -o sheet.png --cues cues.json` (`--from 10 --to 25` zooms, `--detail low` is faster) -
   an annotated PNG: frames, sections, labelled events, spectrogram, loudness, beats, cue markers coloured by sync
   (green = within a frame, amber = within 100 ms, red = no sound). OPEN THE PNG with your image reader: gaps,
   pile-ups, masking and late hits are obvious at a glance.
3. `earshot critique film.mp4` (`--fast` skips the AudioSet / realness passes) - scorecard against the measured
   range of professional launch films: each metric ok / low / high with its weight and actions with timestamps, and
   a verdict (launch-grade / close / needs work / not launch-grade). Fix critical rows first, re-run 1-3 on the new
   render. Its "onset on visual events" share counts new attacks within +-100 ms only (the references' definition);
   its fixes and tips name the cuts with no sound at all, by the same matcher as timeline. If the file is one of the
   profile's own reference films, critique says so and scores it against the others.
4. Exact meters any time (LUFS, true peak dBTP, LRA, noise floor, attack / decay, brightness, bands):
   `earshot measure mix.wav [--music]` or the MCP `measure` tool.

## Loop 3: A/B a change
- `earshot ab before.wav after.wav --query "warm punchy kick"` or `--refs pro_refs/` - which is closer, and how B
  differs from A in measured words (louder, brighter, longer decay). For sounds inside longer files pass the trims
  (`--a-start 1.217 --a-end 1.615 --b-start 7.881 --b-end 8.285`): a whole-file A/B of multi-take files compares every
  take plus the silence between them. Then `earshot timeline after.wav` and compare the event lines and sync stats
  with the old run. Change one thing at a time.

## Loop 4: write a new recipe and improve it by listening (earshot forge)
Use this when `earshot make` has no design for the sound and the user wants one written and refined against real
reference recordings: an AI that hears can write synthesis code, listen, and improve it. It takes minutes per round;
for an ordinary sound use `earshot make` instead. Every sample must be computed (no recorded audio inside, not even
an impulse response); real recordings
are only the references the critic compares with.
1. Write or pick a recipe in `earshot/forge/recipes/<name>.py`: `SPACE` (parameter ranges), `DEFAULTS`, `TARGET`,
   `ALTS` (realistic confusions) and `render(params, r) -> 48 kHz array`, built from `earshot.forge.dsp` (modal
   resonators, contact excitations, noise colours, a time-varying SVF, a computed room, a mic with a noise floor). Model
   the physics of the real thing, measured on the references: what strikes what, what rings, how long, in what room.
2. Run one round: `earshot forge <recipe> --refs <real refs> --budget 144 --reps 2 --tone 1.5 --keep 4 --robust 8
   --dir <round dir>`, or in one Python script per round that loads the models once:
   `forge(recipe, critic=Critic(target, alts, refs, tone=1.5), reps=2, space={...})`.
   - `--tone` puts the model-free tonal balance and tail into the objective.
   - `--reps 2` ranks each candidate by 2 renders, against lucky renders.
   - It takes about 3-5 minutes per 160 renders on a busy laptop CPU, including a 30-60 s model warm-up.
3. Read what it heard. Each pick's summary says, for example, "sounds like: a cork pop, a glass ting", "decay +2.4
   outside the refs" or "tone 11.2 dB from the refs' median: +16 dB at 500-630 Hz (above every ref)". Then check the
   picks four ways:
   - the model-free `tone` words: which bands fall outside every reference, the energy shares, the tail, the length
     (the most trustworthy signal once the model scores are high)
   - `robust`: the mean over new seeds, not the lucky best render
   - `alt_zs`: which confusion is closest
   - spectrograms next to the references: `ffmpeg -i x.wav -lavfi showspectrumpic=s=900x360:scale=log:legend=0 x.png`

   The AudioSet tags in report.json are held out only while you do not pick or steer by them. Fix a selection rule
   (for example, the robust mean) before you look, and report the tags afterwards.
4. Change the recipe's structure, not only its ranges, for what is wrong (a missing release tick, a metallic ring, a
   tail too long). Then repeat. Keep a LOG of each round's change and numbers.
5. Report honestly, next to the real references scored by the same critic (`report.json` ref_scores,
   leave-one-out).

Goodhart, say it plainly:
- The search optimises earshot's own models. Made sounds routinely out-score real recordings on them, which means the
  critic is satisfied, not that the sound is better.
- **p_real is easy to fool**: modal resonators plus a room and a mic noise floor read "real".
- **Typicality rewards being an average** of the references.
- **The model checks saturate**: after a search, every model check, held-out phrasings and independent real sets
  included, puts the made sound inside the real ones. Only measurements and ears can still tell them apart.
- Judge by measured closeness to the references: tonal balance, bands outside every reference, tails, lengths. Also
  use robust re-renders and spectrograms.
- Ask for a human blind test (random order, loudness-matched, hidden key) before calling a made sound "as good as
  real".

In the first test (demo/forge-test) an independent judge found the key press 8-13 dB off the real tonal balance and
the pops band-unlimited with long hissy tails, while every model check called them real. Re-made by steering with the
model-free check, they now sit inside the real recordings' measured spread. The whoosh was re-made as a truly fast
one. The impact (darker, swell-led) and the click (short tails) were not re-made. No human has listened yet.

## Loop 5: make a film's sound kit (earshot sfx / earshot kit)
Use this when a code-made film (a cinetic film, any cue sheet) needs its effects: render them from earshot's shipped
bank instead of writing synth voices. No models load; a whole kit renders in about a minute and re-renders in seconds.
1. Start from a preset: `earshot kit init --preset calm -o audio/kit.json` (calm for text-led films, crisp for
   energetic product films, minimal for designed effects-only films, launch for films cut to music). Each role's
   `note` says why its sound was chosen. calm / crisp / minimal keep riser, swell and suck off: a reveal resolves on
   one element, and the lockup is one clean hit. launch sounds one riser or reverse swell at a section change and
   uses the 0.4 sounds made for music-led films (below).
2. Fit it to the film: `earshot sfx list --kind hit` (or `--category`) shows the candidates with lengths, peak times
   and knobs. Change a role's `sound`, set `edits` (`pitch` to the film's root for the hit or drop, `dur` / `apex` for
   a fixed move), `takes` (12 or more for typing, 2-4 elsewhere), `level_db`. Audition one candidate with
   `earshot sfx render ID -o x.wav [--pitch E5] [--dur 0.8 --apex 0.32]` and read its measured onset / peak / end.
3. Render: `earshot kit render audio/kit.json --out audio/kit`. The takes and `kit.lock.json` (measured onset, peak,
   end, loudness per take) are what the mixer places on cue frames: onset on the frame for hits and touches, the
   peak on a whoosh's apex, the end on a riser's hit (its peak when the role aligns on `peak`, as launch's riser and
   swell do: those members fall after their climax; `kit check` notes an end-aligned one).
4. Check: `earshot kit check audio/kit.json --out audio/kit`. It refuses what a render cannot use (a knob or pitch
   the sound cannot take, a level over 0 dBFS, two takes that would share a file). Every warning names a take and a
   reason (clicks, a cut tail, comb ring, silence, a pitch that climbs more than 4 semitones, a hit with two attacks
   or a build before it, `pitch_off`: a note the knob did not land); swap the sound or the knob rather than keeping a
   warned take. Read the notes too (a sounded riser, a composite sting).
5. Listen through the finished film with Loop 2 (`timeline --cues`, `sheet`) and check that each cue landed.

**The 0.4 families** (`earshot sfx list --category ui_pop`, ...), for the soft, short, mid-register sound of
music-led launch films: `ui_pop` (rounded mid pops for chips, bubbles and rows arriving: the most used effect in
those films), `knock` (mid-band wood and plastic), `cut_hit` (short sub hits with a kick, a punchy cut hit),
`mallet` (gated glockenspiel / celesta / marimba notes, a soft pluck, chime runs; `--pitch` tunes them), `drip`
(water drops whose pitch rises: the audit allows a bubble its climb), `shutter` (snap zooms, whip-cuts),
`reverse_swell` (peaks just after a cut, one a film), plus whoosh_big thump*, riser dark / build / snatch, swish
bright / flick, whoosh_fast whip, glitch burst / mid / chirp and notify mid_tine / mid_wood. Prefer them over the
bright 5 ms blips and long cinematic tails of the older categories when the film is cut to music.

**Levels against the music, not fixed** (launch, or any role with `bed_db`). In films cut to music effects ride 3-10
dB over the music and read by timing and spectrum: a fixed `level_db` is either buried in loud passages or sticks
out in quiet ones. For each cue, measure the bed and set the take from the lock:
```python
from earshot.forge import kit
lock = kit.render('audio/kit.json', 'audio/kit')                 # takes carry l50_lufs; roles bed_db, offset_ms
role = lock['roles']['pop']; take = role['takes'][i]
bed = kit.bed_level(music, t_cue)                                # the music's median 50 ms level, second before
gain = kit.gain_db(take, role, bed)                              # dB on the take file (never over -1 dBFS peak)
start = kit.place(take, role, t_cue)                             # taps / knocks / clicks lead, whoosh peaks land after
cues = kit.space_run(run_times, role['min_gap_ms'])              # a run keeps its gap (pops 114 ms, keys 87)
```
Pull the music down 6-12 dB for 0.3-0.5 s before a hit, a drop, a big whoosh or a swell and bring it back on the hit
(the lock's `mix.stop_down`); do not duck the music under every effect. Without a music bed (an effects-led film)
`level_db` is the fallback; where the music is silent before a cue (a cold open, a gap in the score) `bed_level`
returns `None` and `gain_db` uses `level_db`. Report the gains you set and the bed levels you measured.

Say plainly what the bank is: forge sounds made in code (0.4's tuned toward the measured style of a private set of
licensed reference clips, and checked to stay originals: CLAP 0.84 at most against any of them), and 15 sounds fitted
as code from CC0 recordings (`fit/<role>/<name>`, each with its source; the presets use them for clicks, ticks, keys,
touches and the snap), all chosen and checked by measurement, none by a listening test. They are not recordings: when
the brief asks for recorded sound, use `skills/real-sound` instead. To add a fit of your own (`earshot fit --takes`,
then `bank add-fit`), see `docs/SFX.md` ("Add your own fit").

## Optional: a second opinion in words (audio LLM)
`earshot llm mix.wav --yes-upload [--with-timeline] [--prompt "Is the riser at 0:12 cheap?"]` asks Google Gemini for
a rubric critique (sync, realism, mix, loudness, what sounds cheap, fixes with times). It only runs when the user has
set GEMINI_API_KEY (or GOOGLE_API_KEY) AND explicitly agreed that the audio is uploaded to Google; ask first, never
pass --yes-upload / consent=true on your own. Only audio is sent (mono 64 kbps). Its timestamps are approximate and
its scores are one model's opinion: confirm every fix with timeline / ab.

## Reading the numbers
| number | what it is | trust |
|---|---|---|
| LUFS, dBTP, LRA, dBFS, ms, Hz | measurements (BS.1770 meter, spectrum, envelope) | exact; use them for levels |
| z (match / rank / find / describe tags) | how far a description stands out among the few hundred vocabulary descriptions scored on the same sound | verdicts: z >= 3 strong, 2.5-3 plausible, 2-2.5 weak, < 2 no match (held-out evidence per band in eval/semantics/RESULTS.md) |
| p_vs_alts | uncalibrated, sharp forced choice between your query and your alternatives | only says which wording fits best; judge by z |
| cosine | raw CLAP similarity | never a probability; compare only for the same sound and query |
| AudioSet labels (0..1) | independent per-class probabilities from the AST tagger | good for broad classes; hears only up to 8 kHz |
| p_real / verdict | real or pro-produced vs code-synthesized, from a small fitted model | strong evidence, not proof; read its evidence list; thresholds differ for clips (<= 6 s) and long files (named in each summary) |
| digital_silence_frac | share of sample frames with every channel below 1e-6 in absolute amplitude | weak alone: edit / fade padding is common in real recordings |
| typicality z (compare) | distance to the reference set vs the refs' own spread | >= -1 fits the set, < -2 outlier |
| sync offset_ms, chance | sound minus picture per cut / cue; what random edits would score | trust when well above chance |
| critique status | value vs the range of 7 professional launch films (loudness: web-delivery practice; true peak <= -1 dBTP for a master, <= 0 dBTP for decoded AAC / MP3, which already holds the codec overshoot) | a range, not a target: ok = inside what pros do |
Prefer measurements, then verdicts / z / probabilities; never present cosine as confidence.

## Writing queries
Describe the sound, not its use: source + material + action + character ("a heavy wooden door slamming in a small
room", not "a sound for the logo"). Give alternatives that are realistic confusions (thock vs click vs typewriter).
When a z is borderline, try two or three phrasings: trust a reading that holds up under all of them.
Job words are not heard: "a soft rounded pop, something appearing" passed none of 9 real pops that their own titles
matched. To check a sound for a job, describe the job's sound two or three ways and keep the best reading (on 195 real
effects: 176 passed instead of 152, the jobs told apart slightly better; docs/VALIDATION.md).

## Limits (say these plainly when they matter)
- Taste is not measured: earshot says what a sound is and how it measures, not whether it is beautiful or right for
  the story. The user decides on taste; give them the evidence and the shortlist.
- No speech recognition (it can tell there is speech, not what is said) and no lyric or language judgement.
- CLAP hears 50 Hz - 14 kHz and AST up to 8 kHz: check "air" and low-pass problems with measure (bandwidth_hz, bands).
- Very short sounds are padded to a 10 s window; isolated hits read best with find / describe on a trimmed segment.
- attack_ms / decay_ms come from smoothed envelopes (5 ms / 10 ms RMS): an instant attack reads ~4 ms, and very
  short clicks cannot read below ~10-20 ms of decay.
- Picture cuts come from a frame-difference detector: hard cuts and whips are reliable, slow crossfades are missed
  or found late, and motion inside a shot is not an event, so pass cue times (--cues) for animation.
- Mixed, dense material blurs tags: describe a stem or a segment rather than the whole mix.
- Licence, provenance and attribution are not audio questions: earshot cannot check them.
