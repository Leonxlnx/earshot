---
name: earshot
description: Make sounds with earshot whenever a product, game, page, animation or film needs a sound and no recording is at hand - interface sounds (taps, toggles, hovers, typing, success, error, notifications), everyday foley for games, 3D and video (footsteps, doors, drawers, switches, keys, cloth, glass, water, metal, swings, punches, explosions, animals, kitchen and street sounds), ambience beds (fire, sea, rain, wind, crickets, a cafe, a street), game sounds (coins, fanfares, 8-bit), drums, motion-design effects and a few short music loops, one at a time (earshot make) or as one family (earshot make --set / make_set), and build what the catalogue lacks from recorded parts named in words, heard and tuned by earshot's ears (earshot build, recipes). Listen with earshot whenever you must judge, choose, trim, verify or compare sound you cannot hear - picking a sound effect, music or ambience from candidates, finding the cleanest moment in a recording, checking whether audio reads as a real recording or as code-synthesized, reviewing a soundtrack or a video's audio (what happens when, sync to picture, loudness, mix), A/B-ing a mix change, searching a recipe by listening (earshot forge), or giving a code-made film its sound effects from the shipped bank (earshot sfx / kit). Use it before telling anyone how audio sounds.
---

# earshot: making and hearing sound

## Making sounds

earshot turns a request in words into a sound file: it picks a design from its catalogue and renders a new take. No
model is loaded and nothing is downloaded; a one-shot takes a few seconds. When the catalogue lacks the thing, the
answer says so, and you build it from recorded parts with `earshot build` (below). If the `earshot` command is
missing, see Setup at the end.

The shortest path: `earshot make "WORDS" -o FILE`; read the first line (`clean` or not) and any `earshot makes no X`
line; if it says that, `earshot build "WORDS" -o FILE --part "..." --part "..."`, read `fit:` and `still off:`, and
hand it over as what `fit` says.

The catalogue holds 922 designs. Realistic things are learned from recordings in the public domain (309 things:
footsteps on many surfaces, doors, drawers, switches, keyboards, cloth, paper, glass, water, metal, punches, falls,
explosions, animals, people without words, street, house and kitchen sounds) and so are 25 ambience beds (fire,
shores, crickets, a brook, a cafe, a street, rain on wood, on a tin roof and on an umbrella), the acoustic drums and
the four real instruments that play tonal figures (marimba, harp, pizzicato strings, upright piano). The interface
sounds and the film and motion effects are built from those recordings. Written in code: the tonal figures (success,
coin, level up), chip sound, the drum-machine voices, some beds made from rules, three short music loops and the
film sound bank. Only numbers ship, no audio file: every sound is computed when it is asked for. Invented designs of
real things (jumps, magic, game hits) sound clearly less real: prefer a learned thing when one fits.

### When a sound, and when none

Most moments get none. A sound belongs where something happened that a person should notice without looking.
What follows is common practice; only the motion-video numbers are measured.

- A page or an app: a sound for what the user did on purpose and for what came of it (a switch thrown, a form sent,
  a payment gone through, an error, a message arriving). None for what happens by itself or all the time: the page
  loading, scrolling, an animation, the pointer crossing links. A landing page needs three to six of the `ui` roles,
  not all 17; `hover` is for tools in which hovering is a choice.
- A game: every action the player repeats gets a short sound in several takes (steps, jumps, hits, coins); a reward
  and the end of a level get the longer figures on an instrument; a menu gets one small sound for moving and one for
  choosing.
- A motion video (a launch film, a product demo): two to five effects in a film that music leads (below).
- A scene of real footage: one bed for the place, then the real sound of each thing the picture shows being done,
  asked for as that thing. A thing the catalogue has no recording of is better recorded, built or left silent than
  replaced by a stand-in.

### How to ask

**One sound**: `earshot make "WORDS" -o sounds/success.ogg` (MCP `make`: `want`, `out`; optional `sound`, `n`,
`theme`, `pitch`, `dur`, `voice`, `material`, `loop`, `events`, `seed`, `recipe`; MCP paths absolute). One call per
sound.

```
$ earshot make "a soft success sound for a finance app" -o success.ogg
success.ogg: success/shutter (0.36 s, peak at 0.10 s, -15 LUFS, clean) [success]
also: complete/shutter, success/click, complete/chop
```

- Name the thing, the sound or the job first, then feelings, material, numbers: "a footstep on gravel", "a wooden
  drawer closing", "a camera shutter", "coin pickup, 8-bit", "waves on a beach, 30 seconds".
- The moment comes after the sound, behind "when", "as" or "each time": "a soft low thump when the pricing card
  settles" (thump/soft). Put first, the words of the moment are taken for the sound. Words for the product or the
  place ("in my running app, a success sound") choose nothing.
- What follows "on" or "in" is the surface or the place: "rain on wood" is `rain/wood`. A surface the sound has no
  design on gets the nearest and says so ("rain on a tarp": `no rain on a tarp: rain/tent is the nearest surface`).
- A learned sound answers only when its thing is named: "a heavy impact" gets a designed film hit, "a dumbbell
  dropped" the recording.
- Feelings steer: "softer", "darker", "bigger", "longer", "drier", "not harsh". To steer, ask again with adjectives
  rather than browsing.
- Interface and motion requests get normal sounds. Notes on an instrument answer only when the words ask for music
  (an instrument, a chime, a jingle, a melody) or a theme is named; an instrument named as the player sets the voice
  ("a notification on a harp", `voice`: marimba, harp, pizzicato, felt_piano, kalimba ...). Chip sound only with
  "8-bit", "retro" or "chiptune". A bell or glass timbre only when you ask for one.
- English; German is read through a table of common sound words ("Regen auf Holz" is rain on wood). Other languages
  get "write it in English".

**One event or the action.** A thing a person does several times in a row (footsteps, knocking, typing, coins,
keys, clinking glasses) is played as the action when the words ask for it: "footsteps on gravel", "knocking",
"three knocks", "5 seconds of typing". "a footstep on gravel" or "a single knock on a door" is one event. `--events N`
(MCP `events`) overrides: 1 is one event, 2 or more the action with N events. A counted action is one file ("four
knocks on the table", "three claps", "a countdown 3 2 1"); a count of separate things is that many files ("five coin
pickups", "3 variations"). A length with an action ("three knocks, 2 seconds") is the file's length, the events
spread over it; "200 ms apart" sets the gap. The `played:` line says what was played.

**Takes.** For a learned design, seeds 0, 1, 2 play its different recordings, the fullest clean one first, then new
takes of them. Ask for 6 to 12 takes of anything a game repeats (`--n 8` or "eight variations"), and make them in a
set (`"n": 8` in a spec) so they share one level.

**Several sounds that belong together** (an interface, a game, a film's kit): ONE call, never a loop of `make`.

- A ready-made set: `earshot make --set ui|game|arcade|film -o FOLDER --format webm,mp3 --sprite sprite` (MCP
  `make_set` with `set`, `theme`, `format`, `sprite`). `ui` (17): every role a thing a hand does, no instrument:
  tap a mechanical key (keyboard/mechanical), hover a fingertip on wood (hover/fingertip), toggle_on and toggle_off
  that key going down and coming up (toggle/key_down, toggle/key_up), type (type/mechanical), next and back a page
  (next/page, back/turn), open and close a wooden drawer (open/drawer, close/drawer), send a stick swung through the
  air (send/swoosh), receive a cork (receive/cork), notify two light knocks (notify/knock), success a camera shutter
  (success/shutter), complete a bigger shutter (complete/shutter), warning two firm knocks (warning/knock), error two
  heavier knocks that fall (error/knock), delete a cut of scissors (delete/snip). A plain request for one of these
  jobs gets the same design. With a theme named (soft, glass, wood, felt, tine, metal, electric, digital, retro),
  `ui` plays tonal figures in that theme's voice instead. `game` (22: menus, rewards, jingles, actions; its jump,
  land, hit, explosion and door are invented, so name learned ones in a spec where a real sound matters, such as
  `{"name": "door", "sound": "door/shut"}`), `arcade` (16 chip sounds), `film` (17 motion-design sounds).
- A spec: your own list, on top of a set or alone. `{"sounds": [{"name": "tap", "want": "a soft tap"}, ...]}` in a
  JSON file, then `earshot make --spec FILE -o FOLDER` (MCP `make_set` with `sounds`). Fields of an entry: name, want,
  sound, recipe, pitch, dur, voice, material, seed, n, axes, lufs, events, loop; an unknown field is an error that
  lists them. With `"set": "ui"` your entries add to the set (an entry with a set's name replaces it); without `set`
  only your list is made, which is the call for the handful of sounds a landing page or a scene needs. Every one-shot
  is as loud as it is made alone, within 2.5 dB of the others; loops and beds are levelled by kind. A re-run rewrites
  the files and `manifest.json`; it does not delete files of names you removed.

```
$ earshot make --spec landing.json -o sounds
5 sounds in sounds (wav), manifest.json
open.wav: open/drawer (0.68 s, peak at 0.39 s, -14 LUFS, clean) [opening]
toast.wav: notify/knock (0.40 s, peak at 0.01 s, -18 LUFS, clean) [toast notification]
error.wav: error/knock (0.58 s, peak at 0.19 s, -18 LUFS, clean) [error]
  not understood: address
card.wav: thump/soft (0.09 s, peak at 0.03 s, -24 LUFS, clean) [thump, card, settles, soft]
  not understood: pricing
paid.wav: success/shutter (0.36 s, peak at 0.10 s, -15 LUFS, clean) [payment success]
```

(the `want`s: "a modal opening", "a toast notification", "an error sound for a wrong email address", "a soft low
thump when the pricing card settles", "a payment success sound").

**The catalogue**, only when a pick was wrong and `also` does not help: `earshot sounds --search WORDS` (top 20) or
`earshot sounds --family step`, then `earshot make "WORDS" --sound ID`. Never list the whole catalogue.

### Reading the answer

The answer is the whole result; ask for `--json` (MCP `full=true`) only for measurements.

- `x.ogg: design/id (0.88 s, peak at 0.05 s, G5, marimba, -14 LUFS, clean) [words]`: the design, its length, its
  loudest moment, the note and voice of a tonal sound, the loudness, and `clean` or what the audit found. Report this
  line. Anything other than `clean` means not usable: make it again with another `seed`.
- `also: a, b, c`: alternatives. If the chosen id does not fit the job, call again with `--sound ID`.
- `also (the pick is a guess: ...): id (a few words), ...`: treat the pick as unverified. Choose an alternative by
  its words (`--sound ID`), or ask again naming the sound or the job.
- `earshot makes no THING; ID is only the nearest sound it has`: the catalogue lacks the thing, and the file is a
  stand-in. Build it with `earshot build` (below), take one of the three starting points by id, or tell the user it
  is missing. Never pass a stand-in off as the thing. There is no error for an
  unknown thing: every request gets a file, and this line is the only sign.
- `built from parts by the cookbook recipe NAME`: a recipe answered; `earshot recipe NAME` prints it to change.
  `a start from parts: earshot make --recipe NAME`: the catalogue answered, and a cookbook sketch is a start.
- `not in this sound: WORDS`: part of the request is not in this file. A second thing wants its own call; a verb or
  an adjective of the same thing is no missing sound.
- `not followed: ...` or `0.43 s, not the 2 s asked: ...`: the design has no such control or could not keep that
  length, and the line says what it did instead. "from left to right" is never followed: earshot does not pan.
- `a cut of 4 s, not a loop: ...`, `a seamless loop of 8.00 s, ...`, `8.57 s: a music loop is whole bars ...`: what
  was made of a length asked for a bed or a loop; ask "a seamless loop of ..." when it must repeat without a seam.
- `not understood: WORDS`: those words did nothing; rephrase plainly if they mattered.
- `same design as NAME: ...` (sets): two names got one sound; give the second `"sound": ID`. `another recording of
  ID than NAME` is not a problem.
- `... a loop in mp3 has a gap at its seam ...`: write that loop as ogg, webm, wav or flac.

### Length, level and where the hit is

- **Length.** An interface sound is 0.2 to 0.7 s; a motion tick 0.04 s, a thump 0.06 to 0.09 s, a logo hit about
  0.3 s, a whoosh about 0.4 s, a riser 2.5 s; a game's figures 0.7 s (a coin) to 2.9 s (the win fanfare); real things
  take their time (a door falling shut 2.5 s, distant thunder 7.7 s); a bed is 20 s unless a length is asked.
  Seconds in the words, or `--dur`, set the length of a whoosh, a riser, a figure, a bed and an action; a single
  struck sound rings as it rings and says so.
- **Level.** A single file is as loud as its kind allows (true peak -1 dBFS, at most -14 LUFS, a bed -22), so single
  files are not a mix. `--lufs L` (`"lufs"` in a spec entry) sets one file's loudness. Sounds made as ONE spec are
  levelled against each other, a first balance to mix from: in a scene's spec the bed came out 9 to 16 LU under the
  things done in it.
- **Where the hit is.** Every file starts when its sound does (within 2 ms), but its loudest moment is often later:
  the line says `peak at`, and `--json` gives `peak_s`, `onset_s` and, for a row, `events_s` (each event's start), in
  seconds from the start of the file. To put the hit on the frame at t, start the file at t minus `peak_s`: "a
  whoosh for a transition" answered `whoosh/sweep (0.42 s, peak at 0.30 s, ...)`, so for its peak on 12.000 s start
  it at 11.70 s. A riser ends on its peak, so it starts its own length before the reveal. earshot does not place files
  on a timeline: the editor, ffmpeg or the page's code does.
- **A row in one file**: `earshot make "a soft ui tick" -o count.wav --events 8 --dur 1.5` answered `tick/key (1.50
  s, peak at 0.81 s, -16 LUFS, clean)` and `played: 8 events as a row at 6.2 a second`: a number counting up.

### Scoring a motion video

Measured on 39 reference motion-design videos (`docs/motion-sfx-study.md`); [measured] is a number from them,
[convention] common practice they did not test.

- Music carries the film. 34 of the 35 videos with sound are music-led [measured]; where effects stand out of the
  music at all, a video has 1 to 4 of them. Pick the two to five moments that matter (a reveal, the logo, a key UI
  action, a title change); most cuts get no sound, because editors cut the picture on the music's accents.
- What to ask for (`earshot make --set film` makes these as one family; one alone with
  `earshot make "WORDS" --sound ID`):
  - a stop before a reveal [measured: the most used move]: cut the music for 0.15 to 0.5 s, then come back on a hit.
    It is an edit of the bed; the hit is drop/sack (a heavy bag dropped) or hit/drop.
  - something appears or a card flips: a short low thump, about 50 ms [measured]: thump/door (a door falling shut,
    cut short and low; the set's `pop`). In a run, make it once and place it once per element.
  - a UI click or a word highlighted: a bright dry tick [measured]: tick/key (the top of a mechanical key);
    click/cursor is a whole mouse button.
  - a title change: title/bag (a swish and a breath of air into a heavy bag dropped, about a third of a second).
  - the logo end card: logo/bag (a heavy bag dropped, low, about a quarter second) on the logo frame; sting/reveal
    swells longer into a hit; impact/car is heavier.
  - a move across the screen: whoosh/fast (a stick swung) for a fast move, whoosh/sweep (a broom's stroke) for a
    soft one, slide/paper for a card sliding in; "a quick swish" (flick/thin) for a small fast move.
  - text breaking apart: "a glitch transition" (glitch/short). A reveal after a build: rise/breath (a breath of air
    growing to the cut) or rise/hit (a slam played backwards). A shine: shine/blade (a sword drawn, only its edge).
- Never as an effect in a motion video: marimba, bells, chimes, piano or synth notes, unless the video is about music
  or is a game [measured: none in 35 videos]. Braams, explosions, tape stops and long risers were not used in any
  of them either: use them only when the picture shows that thing.
- Where it sits [measured, few cases]: a hit starts about 80 ms before the picture changes and peaks about 60 ms
  after it; a whoosh peaks on the fastest frame [convention]. Levels against the music [measured]: hits about 18 dB
  over the music's median level at that moment, the hit after a stop and word ticks about 8 dB; near mono, dry.
  Measure the music at the cue (`earshot measure`, or `kit.bed_level` in Loop 5) and set the gain.
- Endings [measured]: most end at body level; end with the logo hit and the music's last note, no long silence.

### Scoring a scene of real footage

Ask for each thing the picture shows as that thing, with its place or surface, and make them as one spec for a first
balance: "steady rain on a terrace, 8 seconds, no loop" (rain/street), "the heavy wooden front door falls shut"
(door/shut), "keys jingling" (keys/bunch), "a phone vibrating on a table" (phone/vibrate), "distant thunder"
(thunder/distant), "footsteps on concrete, 4 seconds" (step/concrete, a walk). Read what the picked design is
before it goes under a picture: "a kitchen timer ringing" answers `timer/done`, the catalogue's timer, which is an
app's timer figure on a marimba, not a wind-up bell. Leave a moment silent, record it, or build it, rather than use a stand-in.

### When earshot does not have the sound: build it

When the answer says `earshot makes no X`, build it on earshot's workbench (MCP `build`, `parts`). You know what
physically makes the sound; say it, one part a thing, and earshot finds a recording for each and writes the recipe:

```
$ earshot build "a stampede of horses" -o stampede.wav --part "hundreds of hoofbeats, for 6 s, getting louder" --part "a low rumble under everything, soft" --part "a horse neighing, twice, far"
stampede.wav: recipe/a_stampede_of_horses_parts (8.53 s, peak at 6.20 s, -14 LUFS, clean) [built]
built from: horse/trot (scattered 26.2 a second over 6 s, its level drawn: -18, +0 dB); thunder/roll; horse/neigh (2 times, under 3915 Hz)
part 1 'hundreds of hoofbeats, for 6 s, getting louder': horse/trot (scattered 26.2 a second over 6 s, its level drawn: -18, +0 dB); horse/trot names 'hoofbeats' in its own words (atlas and own words +3.4, a short take to repeat); next nearest: heart/beat, whisk/beat, step/floorboard
part 2 'a low rumble under everything, soft': thunder/roll; thunder/roll names 'low rumble' in its own words (atlas and own words +3.2, a long take for a bed); next nearest: roll/drum, explosion/far, cat/purr
part 3 'a horse neighing, twice, far': horse/neigh (2 times, under 3915 Hz); horse/neigh names 'horse neighing' in its own words (atlas and own words +6.6, a short take to repeat); next nearest: cow/moo, elephant/trumpet, bear/growl
heard as: horse/neigh, wolf/howl, wind/window; words: 'a horse neighing' (z +4.1), 'a gust of wind rushing past' (z +3.2), 'a cow mooing' (z +3.1); second ear: Neigh, whinny 0.92, Horse 0.71, Animal 0.57, Clip-clop 0.25
fit: z +3.72 for 'a stampede of horses' (the nearest design horse/neigh: z +2.61): sounds like it; recorded: p_real 0.97
tuned in 20 rounds: layer 0 (horse/trot) scatter.rate 16.5 -> 26.2; layer 0 (horse/trot) scatter.pitch 2 -> 1.33; layer 0 (horse/trot) scatter.gain 6 -> 9.4; layer 1 (thunder/roll) dur 6 -> 5.04 ...; fit z +3.16 -> +3.72
still off: nothing the measures find; a person should listen once
next: edit stampede.recipe.json, then: earshot build "a stampede of horses" --recipe stampede.recipe.json
```

(A first try without "soft" on the rumble was "a sketch", and its `still off` line said the thunder covered the rest.)

1. A phrase says the thing and how: how many (twice, a few, hundreds of), when (at 2 s, at the end, under
   everything), how long (for 4 s), motion (passing, from left to right, getting louder, fading), pitch (lower,
   rising), distance and force (far, close, soft, muffled), reversed, pulsing at N a second. The table is in
   `docs/BUILD.md`. Without `--part` it drafts the parts itself.
2. Read the `part N` lines: the design each part became, the next nearest, `earshot has no ...`, `no recording of
   ...; ID is synthesized` and `not understood in part N`. Swap a part with `sound=<id>` in its phrase.
3. Read `fit:` ("sounds like it" needs the fit and no contradiction from the second ear) and `heard as:`.
4. Read `still off:`: each line names the field of `x.recipe.json` to change; then `earshot build "X" --recipe
   x.recipe.json` hears it again (`--rounds 20` tunes it again).
5. Hand it over as what `fit` says: a sketch is a sketch, and a person should listen once.

`--without ID_OR_FAMILY` keeps parts out. It loads the listening model (the listen extra) and takes about a minute;
without the extra it drafts, renders, audits and measures, and says that it did not hear. `docs/BUILD.md` has the
loop and worked examples.

What builds well, measured by the listening model: textures of many similar events and pulsing (a swarm, a
campfire, a purr). Applause is a sketch: `earshot build applause` fits the words (z +4.69) but not 1.0 over the
catalogue's own applause (z +4.50), so it prints "a sketch: not clearly it yet". What does not: rain or hail on a
surface the catalogue has no recording of, a vehicle passing built from another motor, voices (a whale is an owl
and a wolf moved down), pure tones. A part is found by its words and can come back as the wrong recording: read
the `part N` lines, and if a better recording exists (`earshot sounds --search WORDS`), swap it with `sound=<id>`.
`fit` is a listening model's judgement and the second ear also objects to some real recordings: neither is a
person's ear.

### Recipes by hand

A build writes a recipe; you can also write or edit one yourself and render it without models: `earshot recipe
--example` prints the form, `earshot recipe NAME` a cookbook recipe to change, `earshot recipe --check roar.json`
checks it, `earshot make --recipe roar.json -o roar.wav` renders it (a few seconds for a hit, up to half a minute for
a long bed of many layers). Learned designs are the best parts because they are real (`earshot sounds --search`).

```
{"name": "car_crash_far", "says": "a car crash far away", "space": "hall",
 "layers": [
  {"sound": "tyres/screech", "cut": [0, 1.6], "decay": 0.2, "gain": -2},
  {"sound": "car_door/slam", "pitch": -7, "at": 1.5},
  {"sound": "metal/drop", "pitch": -5, "at": 1.52, "gain": -3},
  {"source": "noise", "color": "brown", "dur": 1.5, "at": 1.5, "lowpass": 150, "gain": -16, "decay": 1.2}]}

$ earshot make --recipe crash.json -o crash.wav
crash.wav: recipe/car_crash_far (3.02 s, peak at 1.58 s, -14 LUFS, clean) [recipe]
built from: tyres/screech; car_door/slam (-7 semitones); metal/drop (-5 semitones); brown noise (under 150 Hz)
```

A layer is a catalogue design (`"sound"`) or a plain source (`"source": "noise"` with `dur` and `color`, or `"sine"`
with `dur` and `hz` from 20 to 150). Fields of a layer: `seed`, `dur`, `pitch` (semitones, -24 to 12, as on a tape:
lower is also slower and bigger), `glide` (where the pitch has moved by the end: a motor winding down), `reverse` (a
decay becomes a swell), `cut` ([from, to] s), `halt` (the take runs down to a stop), `lowpass` / `highpass` (Hz, or
[from, to] to move the cutoff), `pan` (a side, or [from, to] to move), `gain` (dB), `at` (start, s), `fade_in`,
`fade_out`, `swell`, `decay`, the design's own axes, `events` (the layer repeated as new takes), `scatter` (one take
many times at random moments: drops into rain, claps into applause), `level` and `bend` (loudness and pitch drawn
over time), `pass` (a thing going by), `spin` (a pulsing: rotor, wings, purring), `stretch` (longer at the same
pitch, 0.5 to 2), `glide`, `halt`, `follow`, and `recipe` (a cookbook recipe as a layer). The whole takes `name`,
`says`, `layers` (1 to 12), `dur`, `space` (dry, booth, room, studio, hall, plate, cathedral) and `kind` (`event`,
or `ambience` for a bed). A mistake is answered in one sentence that names the field. The same recipe goes into
Python (`make(recipe={...}, out=...)`), the MCP `make` tool (`recipe`) and a spec entry. Every field: `docs/MAKE.md`.

What works: real recordings first, and few of them; a recording moved down is bigger (`"pitch": -12` turns a horse
into a beast); noise only for air and rumble, the low sine only for weight under a hit; no synth tone and no bell
for a thing that is not one. Layers are brought to the same peak before their gain: the body at 0, what supports it
at -4 to -10, air and rumble at -12 to -18.

The cookbook (`earshot recipe --list`) holds 99 recipes. 59 answer a request that names their thing on their own (a
car crash, a cannon, a woodpecker, a fireball spell, an energy shield going up, marbles pouring); the other 40 are
sketches that an answer names as a start. A recording of the thing itself answers first ("a rocket launch" is
`rocket/launch`). `clean` means no clicks, no cut tail and no clipping; it does not mean the sound is the thing. Say
what it is built from when you hand it over, and that a person should listen once.

### What not to expect

No music beyond three short loops (`bed/warm`, `bed/playful`, `bed/8bit`): earshot is for sound effects. No speech,
singing or melodies; instruments play single notes and short figures. `make` does not pan; a recipe layer can. No
realistic sound for a thing nobody recorded, except what a build makes of other recordings. Every sound earshot makes
is measured and audited, but the sounds built last (the `ui` set, the film hits, the newer learned things and beds,
the recipes and builds) have not been judged
by a listener: do not say how a made sound sounds beyond its descriptions, because earshot's listening models did not
predict how a person judged its sounds. Say plainly that a learned sound is a new take of a public-domain recording,
not a sound invented from nothing. Everything earshot renders is free to use, commercial too, without credit.

### Listening to what you made (the listen extra)

`--listen` (MCP `listen=true`) also ranks by how the designs sound and adds `heard as: ... (z)`; it loads a model
(20 to 40 s). Use it when a request that describes a sound came back as a guess, or on a recipe with its words. The
listening tools below check any file you made or found.

## Listen

You cannot hear. Never say how audio sounds, or that a sound "fits", without an earshot reading behind it, and quote
the number when you report ("strong match, z +3.9; p_real 0.96; -14.2 LUFS"). earshot runs locally on CPU (audio
never leaves the machine, except with the optional `llm` command) and caches every embedding and result on disk.
Each CLI call is a new process that loads the models first (~20-40 s, even for a cached file); inside the MCP server
or one Python script the models load once and repeat calls take seconds. For more than a handful of calls, use MCP
or a short script (`from earshot import api; api.find(...); api.realness(...)`). Any ffmpeg-readable file works,
including the audio of an mp4.

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
   raw p_real, across files of different lengths. Realness hears how a sound sounds, not where it came from: clean
   professionally synthesized tones (UI sounds, chimes, synth stabs) also read as code-synthesized. If realness is
   unavailable (model missing), say plainly that real-vs-synthetic was not checked. Exact digital zeros alone prove
   nothing (most real recordings have some edit or fade padding).
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

## Loop 4: write a synthesis recipe and improve it by listening (earshot forge)
Use this when the user wants a sound written as synthesis code and refined against real reference recordings. It
takes minutes per round; for an ordinary sound use `earshot make`, and for a sound built from the catalogue's parts
use a recipe (above). Every sample must be computed (no recorded audio inside); real recordings are only the
references the critic compares with.
1. Write or pick a recipe in `earshot/forge/recipes/<name>.py`: `SPACE` (parameter ranges), `DEFAULTS`, `TARGET`,
   `ALTS` (realistic confusions) and `render(params, r) -> 48 kHz array`, built from `earshot.forge.dsp` (modal
   resonators, contact excitations, noise colours, a time-varying SVF, a computed room, a mic with a noise floor).
   Model the physics of the real thing, measured on the references.
2. Run one round: `earshot forge <recipe> --refs <real refs> --budget 144 --reps 2 --tone 1.5 --keep 4 --robust 8
   --dir <round dir>`, or in one Python script per round that loads the models once:
   `forge(recipe, critic=Critic(target, alts, refs, tone=1.5), reps=2, space={...})`. `--tone` puts the model-free
   tonal balance and tail into the objective; `--reps 2` ranks each candidate by 2 renders. About 3-5 minutes per
   160 renders on a laptop CPU, including a 30-60 s model warm-up.
3. Read what it heard ("sounds like: a cork pop, a glass ting", "decay +2.4 outside the refs", "tone 11.2 dB from the
   refs' median: +16 dB at 500-630 Hz"). Check the picks four ways: the model-free `tone` words (the most trustworthy
   once the model scores are high), `robust` (the mean over new seeds), `alt_zs` (the closest confusion), and
   spectrograms next to the references (`ffmpeg -i x.wav -lavfi showspectrumpic=s=900x360:scale=log:legend=0 x.png`).
   The AudioSet tags in report.json are held out only while you do not pick or steer by them.
4. Change the recipe's structure, not only its ranges, for what is wrong (a missing release tick, a metallic ring, a
   tail too long), and repeat. Keep a log of each round's change and numbers.
5. Report honestly, next to the real references scored by the same critic (`report.json` ref_scores,
   leave-one-out).

Goodhart, say it plainly: the search optimises earshot's own models, so made sounds routinely out-score real
recordings on them, which means the critic is satisfied, not that the sound is better. p_real is easy to fool (modal
resonators plus a room and a noise floor read "real"); typicality rewards being an average; after a search every
model check puts the made sound inside the real ones. Judge by measured closeness to the references (tonal balance,
bands outside every reference, tails, lengths), robust re-renders and spectrograms, and ask for a human blind test
(random order, loudness-matched, hidden key) before calling a made sound "as good as real".

## Loop 5: make a film's sound kit (earshot sfx / earshot kit)
Use this when a code-made film (a cinetic film, any cue sheet) needs its effects: render them from earshot's shipped
bank instead of writing synth voices. No models load; a whole kit renders in about a minute and re-renders in seconds.
1. Start from a preset: `earshot kit init --preset calm -o audio/kit.json` (calm for text-led films, crisp for
   energetic product films, minimal for designed effects-only films, launch for films cut to music). Each role's
   `note` says why its sound was chosen. calm / crisp / minimal keep riser, swell and suck off: a reveal resolves on
   one element, and the lockup is one clean hit. launch sounds one riser or reverse swell at a section change and
   uses the sounds made for music-led films (below).
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

**The families for films cut to music** (`earshot sfx list --category ui_pop`, ...), for the soft, short,
mid-register sound of music-led launch films: `ui_pop` (rounded mid pops for chips, bubbles and rows arriving: the
most used effect in those films), `knock` (mid-band wood and plastic), `cut_hit` (short sub hits with a kick),
`mallet` (gated glockenspiel / celesta / marimba notes, a soft pluck, chime runs; `--pitch` tunes them), `drip`
(water drops whose pitch rises), `shutter` (snap zooms, whip-cuts), `reverse_swell` (peaks just after a cut, one a
film), plus whoosh_big thump*, riser dark / build / snatch, swish bright / flick, whoosh_fast whip, glitch burst / mid
/ chirp and notify mid_tine / mid_wood. Prefer them over the bright 5 ms blips and long cinematic tails of the older
categories when the film is cut to music.

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

Say plainly what the bank is: forge sounds made in code (some tuned toward the measured style of a private set of
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
When a z is borderline, try two or three phrasings: trust a reading that holds up under all of them. Job words are
not heard: "a soft rounded pop, something appearing" passed none of 9 real pops. To check a sound for a job, describe
the job's sound two or three ways and keep the best reading (docs/VALIDATION.md).

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

## Setup (once)
This skill is the instructions; the tool is a Python package. If the `earshot` command is missing, install it from
GitHub (the name `earshot` on PyPI belongs to another project):
```
pip install "earshot @ git+https://github.com/Leonxlnx/earshot"                 # makes sounds: no model, no GPU
pip install "earshot[listen,mcp] @ git+https://github.com/Leonxlnx/earshot"     # adds listening and the MCP server
earshot doctor                           # packages, ffmpeg, models (pinned revision, cache state); --load times them
```
Making sounds needs only the first line; ffmpeg on PATH is needed for ogg, webm, mp3 and m4a. Listening downloads two
open models (about 1.1 GB to download, about 2.2 GB on disk) on first use. In a clone of the repo,
`pip install -e ".[all]"` does the same from the folder.
- **CLI**: `earshot <command> ... [--json] [--out result.json]`. Text output starts with a plain summary; times are
  m:ss.ss in text and seconds in JSON. Errors are one line on stderr: `earshot <command>: error: ...`, exit 1.
- **MCP** (models stay loaded between calls): `claude mcp add earshot --scope user -e EARSHOT_THREADS=3 --
  /path/to/earshot/.venv/bin/python -m earshot mcp`, or in any client's `.mcp.json`:
  `{"mcpServers": {"earshot": {"command": "/path/to/earshot/.venv/bin/python", "args": ["-m", "earshot", "mcp"],
  "env": {"EARSHOT_THREADS": "3", "GEMINI_API_KEY": "${GEMINI_API_KEY:-}"}}}}` (the key line only for the optional
  llm tool; never write a literal key into a project file). Tools: make, make_set, sounds (no models); build; measure,
  describe, match, rank, find, compare, ab, realness, timeline, sheet, critique, doctor, llm_critique; the sound
  bank (no models): sfx_list, sfx_render, kit_render. Pass absolute paths; `full=true` returns all JSON. If Claude
  Code shows no earshot tools, start it with `MCP_TIMEOUT=60000` (PowerShell: `$env:MCP_TIMEOUT=60000; claude`).
  `earshot mcp --warm` preloads the models at startup.
- If the machine is shared, set `EARSHOT_THREADS=2-3` (or `--threads 2` on any command).
