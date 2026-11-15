# Making sounds: `make`, `sounds`, `pack`

An agent asks for a sound in words; earshot reads the request, ranks its catalogue of designs, renders the best one
in code, audits it without models and writes the file. No model is loaded, nothing is downloaded and no recorded
audio is used. Everything rendered belongs to whoever rendered it: free for any use, commercial too, without credit.

- [make](#make) - [make --set / --spec](#a-whole-set) - [sounds](#sounds) -
  [pack](#pack)
- [The request language](#the-request-language) - [The eight axes](#the-eight-axes) - [Themes](#themes)
- [The catalogue](#the-catalogue-family-by-family) - [Voices](#voices)
- Recipes: [a web page](#recipe-a-web-page) - [a game](#recipe-a-game) - [a film](#recipe-a-film) -
  [add a design](#add-a-design)

## make

```
earshot make "WORDS" [-o FILE] [--n N] [--seed S] [--theme NAME] [--sound ID] [--pitch NOTE] [--dur S]
             [--voice NAME] [--material NAME] [--bright X] [--hard X] [--ring X] [--size X] [--space X]
             [--width X] [--energy X] [--motion X] [--lufs L] [--mono] [--no-loop] [--listen] [--json]
```

```
$ earshot make "a soft success chime for a finance app" -o success.ogg
success.ogg: success/rise (0.88 s, G5, vibraphone, -14 LUFS, clean) [success, bell]
also: success/done, success/fifth, success/triad

$ earshot make "coin pickup, bright" -o coin.wav --n 3
coin_1.wav: coin/bright (0.65 s, B5, glockenspiel, -14 LUFS, clean) [coin, pickup]
coin_2.wav: coin/bright (0.65 s, B5, glockenspiel, -14 LUFS, clean)
coin_3.wav: coin/bright (0.67 s, B5, glockenspiel, -14 LUFS, clean)
also: coin/soft, coin/drop, coin/flip
```

The answer is the whole result. Each file gets one line: the design chosen, its length, the note and voice it
played, its momentary-max loudness, `seamless loop` for a loop, and `clean` or the audit's warnings; the first line
ends with the words that decided (`[success, bell]`, or `[named]` for `--sound`). Then up to three more lines:

| line | meaning |
|---|---|
| `also: ...` | the next designs in the ranking; ask again with `--sound ID` to take one |
| `not used by this sound: ...` | controls the request set that this design does not read (bank sounds read no axes) |
| `not understood: ...` | words that matched nothing in the catalogue or the lexicon |
| `heard as: '...' (z +x.x, verdict)` | with `--listen` only (below) |

| option | what it does |
|---|---|
| `-o FILE` | the file; its extension picks the format (wav, flac, ogg, mp3, m4a, webm). None: `<family>_<name>.wav` here |
| `--n N` | N takes on seeds S .. S+N-1, written `name_1 .. name_N`; "three variations" in the words does the same |
| `--seed S` | another take of the same sound (default 0); the same arguments always give the same samples |
| `--sound ID` | a design id from `earshot sounds` instead of the ranking; the words still set feel and numbers |
| `--theme NAME` | a theme preset ([Themes](#themes)): voice, key, room and axis offsets shared by a product's sounds |
| `--pitch`, `--dur` | a note (`C5`) or Hz; seconds. Override the words |
| `--voice`, `--material` | the instrument or patch of a tonal sound ([Voices](#voices)); a material (ranking only, below) |
| `--bright` ... `--motion` | the axes, -1 .. 1, added to what the words say ([The eight axes](#the-eight-axes)) |
| `--lufs L` | a momentary-max loudness instead of the default level |
| `--mono` | sum to one channel |
| `--no-loop` | an ambience once, with a fade in and out, instead of a seamless loop |
| `--listen` | render the best three designs, keep the one earshot's models hear closest to the request (below) |
| `--json` | the full result: files, chosen id, `why`, `also`, `unknown`, every take's measurements and audit |

**What happens.** `earshot.brief` reads the words into a brief (the job or sound, the feel as axis offsets, a
material or voice, numbers, the key); `brief.rank` scores every design; the best is rendered with those controls
(`earshot.sounds.registry.render`), finished (low cut at 25 Hz, leading silence dropped, a 0.4 ms fade-in, the
silent end trimmed and faded) and audited by `forge.sfx.check` (clicks, cut tails, clipping, DC, comb ring, a pitch
that climbs more than 4 semitones, a hit with stacked attacks or a build before it; findings a design declares as
its own, such as a power-up's rising pitch, are notes). A take that fails is rendered again on seed + 101, + 202,
+ 303; if all four fail the first is written and its line names the warnings. A request that matches nothing is an
error that says what to ask for:

```
$ earshot make "a frobnicating zorp" -o x.wav
earshot make: error: ValueError: no sound in the catalogue answers 'a frobnicating zorp': `earshot sounds` lists what
there is; name the sound itself (a chime, a click, a whoosh, rain) or the job it plays (success, coin, logo)
```

**`--listen`** adds earshot's ears: the best three designs are rendered and matched (`earshot match`) against the
request and their own descriptions; the one with the highest z is written, and the result gains `heard as: '<the
description that fit best>' (z +x.x, verdict)`. It loads the two listening models (20-40 s once per process; in the
MCP server they stay loaded). With `--sound` there is nothing to choose, and it only reports the reading.

**Level.** A single sound is set to a true peak of -1 dBFS, but no louder than -14 LUFS momentary max (pads and
textures -18, beds -22): a dense sound at full peak is far louder than a tap at full peak. `--lufs` sets the
momentary-max loudness exactly, with the peaks limited to -1 dBFS. wav and flac are written directly (24-bit); ogg
(Vorbis 96k), mp3 (128k), m4a (AAC 96k) and webm (Opus 64k) go through ffmpeg, which must be on PATH for them. Files
are 48 kHz stereo unless `--mono`; every file starts when its sound does (within 2 ms) and ends faded, never cut.

**Speed** (one thread, no models): a single `make` takes about 1-2 s including Python start-up; ambience loops of
12-20 s take 3-10 s to render.

In Python: `from earshot.make import make; make('a deep impact for a logo reveal', 'hit.wav', size=0.3)` returns the
same result as a dict (`summary`, `sound`, `files`, `why`, `also`, `unknown`, `takes`, and `heard` with
`listen=True`). `make()` also takes `sr` and `bits` for the written file.

## A whole set

```
earshot make --set ui|game|arcade|film [-o FOLDER] [--theme NAME] [--format ogg] [--sprite sprite.webm] [--mono]
             [--lufs L] [--seed S]
earshot make --spec sounds.json [-o FOLDER] [--theme NAME] [--format ...] [--sprite ...]
```

```
$ earshot make --set ui --theme glass -o public/sounds --format ogg --sprite sprite.webm
16 sounds in public/sounds (ogg), manifest.json
tap.ogg: tap/tone (0.47 s, D6, marimba, -26 LUFS, clean) [named]
hover.ogg: hover/tone (0.43 s, D6, marimba, -28 LUFS, clean) [named]
toggle_on.ogg: toggle/up (0.52 s, D6, glockenspiel, -19 LUFS, clean) [named]
...
success.ogg: success/rise (0.82 s, D5, glockenspiel, -19 LUFS, clean) [named]
complete.ogg: success/triad (1.07 s, D5, glockenspiel, -18 LUFS, clean) [named]
warning.ogg: warning/pulse (0.72 s, D5, glockenspiel, -19 LUFS, clean) [named]
error.ogg: error/fall (0.69 s, D5, glockenspiel, -19 LUFS, clean) [named]
delete.ogg: delete/drop (0.54 s, D5, glockenspiel, -19 LUFS, clean) [named]
16 sounds in public/sounds\sprite.webm (14.5 s, 152 kB), map sprite.json
```

Every sound of a set shares the theme: one voice, one key, one room. The tap and hover above play marimba, not the
theme's glockenspiel: one short note of a sustaining metal bar is heard as an electronic beep, so the smallest
figures keep to struck wood unless the theme's voice is itself short. The set took about 25-35 s on one thread.

**The ready-made sets** (`earshot.sounds.sets`; file name -> design):

| set | theme | sounds |
|---|---|---|
| `ui` | soft | 16: tap, hover, toggle_on, toggle_off, open, close, next, back, send, receive, notify, success, complete, warning, error, delete |
| `game` | tine | 22: menu_move, menu_select, menu_back, coin, pickup, gem, heal, power_up, power_down, level_up, checkpoint, secret, win, lose, jump, land, step, hit, hurt, explosion, splash, door |
| `arcade` | retro | 16 chip sounds: menu_move, menu_select, menu_back, coin, pickup, jump, laser, explosion, hurt, power_up, power_down, one_up, level_up, win, lose, text |
| `film` | none | 15 from the bank and the figures: whoosh, swish, swipe, pop, tick, click, type, shutter, hit, impact, drop, riser, swell, chime, logo |

**Levels in a set.** The sounds are levelled against each other by momentary-max loudness: the loudest allowed is
`lufs` (default -16), and each sound sits under it by its kind, so frequent sounds are quieter:

| kind | LU under the top | kind | LU under the top |
|---|---|---|---|
| boom, stinger | 0 | chime, rise, fall, zap, thud | -3 |
| hit | -1 | note, whoosh, voice | -5 |
| figure | -2 | object | -6 |
| pop | -7 | noise | -8 |
| click | -9 | tap, pad | -10 |
| tick, texture | -12 | bed | -14 |

A set's wav files are 16-bit; `--mono` leaves loops stereo. `manifest.json` in the folder lists every file with its
design, length, loudness, loop flag, pitch and whether its audit was clean.

**The spec** (a JSON file for `--spec`, or a dict for `make_set`; the MCP tool `make_set` takes `set`, `sounds`,
`theme`, `format` and `sprite`):

```json
{"set": "ui", "theme": {"base": "wood", "key": "D"}, "format": "ogg", "sprite": "sprite.webm",
 "sounds": [{"name": "success", "want": "a happy two-note success chime", "dur": 0.5, "axes": {"bright": 0.2}},
            {"name": "coin", "want": "coin pickup", "n": 2},
            {"name": "rain", "sound": "rain/window", "dur": 12}]}
```

| key | meaning |
|---|---|
| `set` | start from a ready-made set; `sounds` then adds to it or replaces a sound of the same name |
| `theme` | a preset name or a theme dict (`base` + changes) |
| `format`, `out`, `mono`, `lufs`, `sprite` | file format, folder, mono, the loudest level, a sprite file name |
| `sounds[]` | each: `name` (the file name, required, unique) and `want` (words) and / or `sound` (an id), plus `pitch`, `dur`, `voice`, `material`, `seed`, `n`, `axes` ({axis: -1 .. 1}), `lufs` (this sound's own level) |

```
$ earshot make --spec spec.json -o myset
19 sounds in myset (ogg), manifest.json
tap.ogg: tap/tone (0.28 s, D6, marimba, -26 LUFS, clean) [named]
...
success.ogg: success/rise (0.54 s, D5, marimba, -19 LUFS, clean) [2-note, success, bell]
coin_1.ogg: coin/bright (0.34 s, D6, marimba, -19 LUFS, clean) [coin, pickup]
coin_2.ogg: coin/bright (0.33 s, D6, marimba, -19 LUFS, clean) [coin, pickup]
rain.ogg: rain/window (12.00 s, -30 LUFS, seamless loop, clean) [named]
19 sounds in myset\sprite.webm (24.3 s, 200 kB), map sprite.json
```

An unknown field is an error that lists the fields. A design id of a set that this catalogue does not have is left
out and named on the last line.

## sounds

```
earshot sounds [--search WORDS] [--domain ui|game|motion|ambience|music] [--kind KIND] [--family FAMILY] [--json]
```

```
$ earshot sounds
457 sounds in 143 families
success/rise                0.64 s  two quick rising notes, a small bright chime
success/fifth               0.69 s  two notes a fifth apart rising, an open clear chime
...

$ earshot sounds --search notification
20 sounds for 'notification'
notify/pair                 0.90 s  two notes falling by a major third, a gentle ding-dong
notify/motif                1.06 s  three notes: up a fifth, then settling on the third, a small melodic call
notify/lift                 0.76 s  two notes rising by a fourth, a light upward chime
notify/single               0.90 s  one clear ringing note
...

$ earshot sounds --family step
6 sounds in 1 families
step/grass                  0.25 s  a soft crunchy footstep in dry grass
step/wood                   0.25 s  a footstep on a wooden floor, heel then toe
step/stone                  0.25 s  a hard footstep on stone, a sharp heel click
step/gravel                 0.25 s  a crunchy footstep on gravel
step/snow                   0.25 s  a crunchy squeaky footstep in snow
step/metal                  0.25 s  a heavy boot step on a metal floor
```

One line per design: id, usual length, the first of its descriptions, `(loop)` for a loop. `--search` ranks exactly
as `make` would (top 20). `--json` adds every field of a design: kind, all descriptions (`says`), jobs (`uses`),
tags, domains, default pitch, the axes it reads, voice, material, loop. The MCP tool without a filter answers with
the families only. Kinds: tap, click, tick, pop, note, chime, figure, hit, thud, whoosh, rise, fall, zap, boom,
noise, texture, bed, pad, stinger, voice, object.

## pack

```
earshot pack FILES... | FOLDER -o sprite.EXT [--gap 0.25] [--mono]
```

```
$ earshot make --set arcade -o assets/arcade --format wav
16 sounds in assets/arcade (wav), manifest.json
...
$ earshot pack assets/arcade -o assets/arcade.ogg
16 sounds in assets/arcade.ogg (14.2 s, 113 kB), map arcade.json
```

`pack` lays sounds end to end in one file with `gap` seconds of silence between them (lossy codecs smear a little
in time; without a gap the tail of one sound leaks into the next) and writes the map next to it as
`<name>.json` in the format howler.js reads:

```json
{"src": ["arcade.ogg"], "sprite": {"coin": [0, 537], "explosion": [787, 974], "hurt": [2011, 333], ...}}
```

Each entry is `[offset_ms, duration_ms]`, and `[offset_ms, duration_ms, true]` for a loop. Given a folder, it packs
every audio file in it sorted by name and reads `manifest.json` for the loops, which get four times the gap. It
skips only the file it is writing: pack into a new name or another folder, or an earlier sprite in that folder is
packed too. In Python: `from earshot.pack import pack; pack('public/sounds', 'public/sprite.webm')`.

## The request language

`earshot.brief.read` turns words into a brief without any model, from the tables in `earshot/sounds/lexicon.py`.
How the words were read (`earshot.brief.read`, run on each request; `brief.rank` gave the top design):

| request | read as | top design |
|---|---|---|
| `a soft wooden success chime for a finance app, 400 ms, in D` | concepts success, bell; hard -0.5, energy -0.3; material wood; domain ui; dur 0.4; key D | success/rise |
| `an error sound but not too loud` | error; energy -0.42 | error/fall |
| `the opposite of a success sound` | error | error/fall |
| `a much bigger, darker whoosh` | whoosh; size +0.84, bright -0.6 | whoosh/pass |
| `a kalimba note in E` | note, voice kalimba; domain music; key E | note/kalimba |
| `two-note rising chime in D minor` | bell; 2 notes; rising; key D, scale minor | success/rise |
| `three variations of a footstep on gravel` | footstep; n 3 | step/gravel |
| `8-bit coin pickup` | coin; material retro (chip designs only) | coin/chip |
| `rain on a tent, no loop` | rain; loop off | rain/tent |
| `seamless forest ambience, 30 seconds` | forest, ambience; loop on; dur 30 | forest/morning |
| `a click under 50 ms` | click; a ceiling of 0.05 s | click/snap |
| `a bell with an airy whoosh` | bell, with the part whoosh; bright +0.3, hard -0.3 | bell/bike |

What is recognised:

- **The job or the sound.** A job ("success", "coin pickup", "logo reveal", "level complete") or a sound ("chime",
  "whoosh", "rain"), often both, mapped to canonical words, the longest phrase first. The first one named is the
  head of the request. What follows "with" describes a part of it, what follows "for", "when", "during" ... once
  the head is named is where it plays: both count less.
- **Feelings as axes.** Adjectives become offsets on [the eight axes](#the-eight-axes) ("soft" lowers hard and
  energy, "darker" lowers bright, "bigger" raises size, "longer" raises ring, "drier" lowers space), and also
  hints that break ties inside a family ("crisp" finds `whoosh_fast/crisp`). Intensifiers scale them ("a little",
  "much", "very").
- **Negation and comparison.** "not harsh", "not too loud" flip the adjective; "not an error" turns a named job into
  one to avoid; "the opposite of X" and "X but the other way" turn X into its opposite; "like a ..." is a
  comparison; what follows "than" is a reference and is skipped.
- **Material and voice.** An instrument named as the thing itself ("a kalimba note", "marimba hit") asks for a
  note of that voice; one named as the player ("a success chime on a kalimba") sets the voice of a figure.
  Material words ("wooden", "glass", "metal") steer the ranking toward designs of that material; no design reads a
  material as a control in 1.0, so a figure keeps its voice and the line says `not used by this sound: material`.
  For a wooden chime, ask with `--theme wood` or `--voice marimba`.
- **Numbers.** Lengths ("400 ms", "2 seconds", "half a second"; "under 80 ms" is a ceiling that prefers shorter
  designs and cuts nothing), a pitch ("C6", "440 Hz", "middle C"), a key ("in D", "in the key of Bb minor"), how
  many takes ("three variations", "5 versions", "a few takes"; at most 12) and how many notes ("a two-note chime",
  "three rising notes", "twice"). A key moves a tonal design's root to the nearest note of that key; the scale is
  recorded, and no design in 1.0 builds its steps from it.
- **Direction.** "rising", "falling", "going up" ... prefer designs whose gesture moves that way.
- **Loop or not.** "loop", "seamless", "looping", "endless" ask for a loop; "no loop", "one-shot" for the scene
  once.
- **8-bit.** "8-bit", "retro", "chiptune", "arcade" make chip sound a hard filter: only chip designs answer, and
  chip designs answer nothing else.
- **Where it plays.** Domain words ("app", "game", "film", "platformer") weigh designs made for that domain.
- Words nothing knows come back under `not understood` and count only a little:

```
$ earshot make "a squishy success chime, longer" -o sq.wav
sq.wav: success/rise (1.31 s, G5, vibraphone, -14 LUFS, clean) [success, bell]
also: success/done, success/fifth, success/triad
not understood: squishy
```

What reached the render and what did not, on two real calls:

```
$ earshot make "a soft wooden success chime for a finance app, 400 ms, in D" -o wood.wav
wood.wav: success/rise (0.91 s, D5, vibraphone, -14 LUFS, clean) [success, bell]
also: success/done, success/fifth, success/triad
not used by this sound: dur, material

$ earshot make "a success chime on a kalimba, softer" -o kal.wav
kal.wav: success/rise (0.69 s, G5, kalimba, -14 LUFS, clean) [success, bell]
also: success/done, success/fifth, success/triad
```

Figures read no `dur` (their length comes from their notes; "longer" raises ring, which lets the notes sound
longer); the key D moved the root from G5 to D5.

<!-- RANKING NUMBERS: lead fills in -->

## The eight axes

Every design reads the same controls, so a word and a theme mean the same thing everywhere. Each axis runs from -1
to 1, with 0 = the sound as designed; a design reads only some of them, and the rest are reported as not used.

| axis | -1 .. +1 | what it does in the engines |
|---|---|---|
| `bright` | darker .. brighter | a harder mallet and more of the high modes; a higher filter cutoff or FM index; more top in noise |
| `hard` | softer .. harder attack | mallet stiffness, attack time; a faster and louder onset layer |
| `ring` | shorter, drier .. longer ring | damping (struck: a quarter to four times the ring time); note lengths in a figure; tail length |
| `size` | smaller, lighter .. bigger, heavier | lower and longer, more low layer; a larger resonator or body |
| `space` | no room .. more room | the shared room's tail: -1 dry, +1 8 dB more and 1.6 times longer |
| `width` | narrower .. wider | stereo spread of the notes, the microphone pair, decorrelation |
| `energy` | gentler .. more intense | how hard it is played: level and timbre together; more events in a texture |
| `motion` | steadier .. more movement | the speed of a figure (0.7 to 1.4 times); LFOs, chorus and detune in a patch |

## Themes

A theme is what makes a set of sounds one product: one voice, one key, one room and offsets on the axes, applied to
every design that honours them (`earshot/sounds/theme.py`).

| preset | voice | key | space | axes | says |
|---|---|---|---|---|---|
| `soft` | soft_pluck | C major | room | bright -0.2, hard -0.4 | rounded, quiet, mid-register: the default of product UI |
| `glass` | glockenspiel | D major | studio | bright +0.2, ring +0.1 | clear, bright, bell-like |
| `wood` | marimba | G pentatonic | room | bright -0.1, ring -0.2 | warm, dry, organic |
| `felt` | felt_piano | F major | room | bright -0.3, hard -0.3 | an intimate soft piano |
| `tine` | kalimba | A pentatonic | room | ring -0.1 | plucked tines: friendly, playful |
| `metal` | vibraphone | E major | studio | ring +0.2 | smooth sustained metal bars |
| `electric` | electric_piano | Eb major | room | bright -0.1 | a mellow electric piano |
| `digital` | glass_pluck | B major | dry | bright +0.2, hard +0.2 | clean synthetic tones: precise, modern |
| `retro` | square_lead | C major | dry | hard +0.3 | chip-style: 8-bit games |

The fields of a theme dict (`{"base": "wood", "key": "D"}` changes a preset; a dict without `base` starts empty):

| field | meaning |
|---|---|
| `voice` | the instrument or patch that plays every figure ([Voices](#voices)) |
| `key`, `scale` | the key every tonal design is moved into (C, C#, Db ... B); the scale (major, minor, pentatonic, minor_pentatonic, lydian, dorian, whole_tone) |
| `space` | the room: dry, booth, room, studio, hall, plate, cathedral (`earshot/synth/space.py`) |
| `axes` | offsets added to every request's axes |
| `material` | the material of the theme's taps and clicks |
| `says` | a description, for people |

In 1.0 no design reads `material` or `scale` as a control: the voice, key, space and axes are what a theme changes.
The words of a request lay their key over the theme's. A figure whose shape a voice cannot carry gives way to its
own voice (a tap or hover on a metal or glass theme plays marimba, as above).

## The catalogue, family by family

`earshot sounds` lists 457 designs in 143 families: 248 written for 1.0 and the 209 named sounds of the earlier bank.
Ids are `family/name`; the family is what the sound is or does, the name its character.

| group | designs | ids | for | controls |
|---|---|---|---|---|
| interface figures (`sounds/ui.py`) | 25 | `success/rise`, `error/fall`, `notify/pair`, `toggle/up`, `open/lift`, `send/lift`, `tap/tone`, `hover/tone`, `purchase/chime`, `like/twinkle` ... | confirmations, errors, warnings, notifications, toggles, navigation | all 8 axes, pitch (the root), voice |
| game figures (`sounds/game.py`) | 15 | `coin/bright`, `pickup/blip`, `gem/sparkle`, `power_up/rise`, `level_up/fanfare`, `win/fanfare`, `lose/descend`, `menu/select` ... | rewards, signals, jingles, menus | all 8 axes, pitch, voice |
| brand figures (`sounds/brand.py`) | 5 | `logo/bloom`, `logo/duo`, `logo/motif`, `startup/rise`, `shutdown/fall` | sonic logos, openings, endings | all 8 axes, pitch, voice |
| material ticks (`sounds/ui.py`) | 3 | `tick/wood`, `tick/glass`, `tick/metal` | taps and detents that should not be a note | 7 axes (not motion), pitch |
| notes (`sounds/tonal.py`) | 51 | `note/<voice>`: 36 instruments and 15 synth patches | a named instrument, a product's own motifs | all 8 axes, pitch, dur |
| synthesizer (`sounds/synth.py`) | 14 | `pad/warm`, `pad/air`, `pad/glass`, `pad/strings`, `drone/dark`, `sub/swell`, `braam/hit`, `braam/swell`, `riser/tone`, `sweep/noise`, `stab/chord`, `zap/laser`, `power_up/synth`, `power_down/synth` | finished synth sounds for motion design and games | all 8 axes, pitch (root), dur (held) |
| chip (`sounds/retro.py`) | 24 | `coin/chip`, `jump/chip`, `laser/chip`, `explosion/chip`, `win/chip`, `text/chip`, `engine/chip` (a loop) ... | 8-bit games; only when asked for | bright (output filter), energy (>= 0.5: a 6-bit output), pitch, dur |
| action (`sounds/action.py`) | 47 in 23 families | `jump/*`, `land/*`, `step/grass` .. `step/metal` (6 surfaces), `swing/*`, `hit/punch` ..., `explosion/*`, `shot/*`, `door/*`, `break/*`, `splash/*`, `teleport/*`, `magic/*`, `countdown/*`, `laser/beam` and `rumble/quake` (loops) | games and animation | energy, size, bright, hard, ring, space, width; pitch for the tonal ones; dur for loops |
| objects (`sounds/object.py`) | 36 in 26 families | `coin/drop`, `glass/clink`, `paper/*`, `cards/*`, `dice/*`, `water/drop`, `clock/ticktock` (loop), `switch/*`, `keys/*`, `bell/desk`, `zipper/*`, `cork/*` ... | foley and skeuomorphic feedback | the same 7 axes; dur for poured, pulled and looped ones |
| ambience (`sounds/nature.py`) | 28 in 14 families | `rain/window`, `rain/roof`, `wind/trees`, `fire/camp`, `stream/*`, `waves/*`, `forest/morning`, `night/crickets`, `room_tone/quiet`, `city/distant`, `thunder/clap` ... (26 loops) | beds for games, pages, timers and films | energy (how much), size, space (how far), bright, width; dur = loop length; `--no-loop` |
| bank (`sounds/bank.py`) | 209 in 38 families | `whoosh_fast/crisp`, `riser/tension`, `impact/deep`, `ui_pop/soft`, `cut_hit/punch`, `typing_key/light`, `logo_sting/*`, `pro_*` ... | motion design and film ([SFX.md](SFX.md)) | pitch and dur where the recipe has those knobs; no axes |

Bank sounds rank under the designs written for the catalogue (prior 0.8, the `pro_` families 0.7), so a request
that fits both gets the newer design; members that carry only a number in the bank (`pop/3`) stay reachable through
`earshot sfx`. Chip designs answer only requests that ask for chip sound. 31 designs are loops. By domain: ui 217,
game 263, motion 391, ambience 32, music 70 (a design can serve several).

## Voices

`earshot.synth.voices.table()`: every voice a figure or a note can be played by. Ranges are the playable notes; a
pitch outside is moved by octaves.

| voice | engine | material | range | default | sounds like |
|---|---|---|---|---|---|
| glockenspiel | struck | metal | C5-C8 | C6 | a glockenspiel note |
| celesta | struck | metal | C4-C8 | C5 | a celesta note |
| vibraphone | struck | metal | F3-F6 | F4 | a vibraphone note |
| marimba | struck | wood | C2-C7 | C4 | a marimba note |
| xylophone | struck | wood | F4-C8 | C6 | a xylophone note |
| crotales | struck | metal | C6-C8 | C7 | a crotale note |
| kalimba | struck | tine | G3-E6 | C5 | a kalimba note |
| music_box | struck | tine | C4-C8 | C6 | a music box note |
| tubular_bell | struck | metal | C4-G5 | C5 | a tubular bell |
| wind_chime | struck | metal | C5-C8 | E6 | a wind chime |
| triangle | struck | metal | D3-D5 | A3 | a triangle |
| hand_bell | struck | metal | C3-C8 | C5 | a handbell |
| church_bell | struck | metal | G2-C6 | G4 | a church bell |
| singing_bowl | struck | metal | C3-C6 | A3 | a singing bowl |
| wine_glass | struck | glass | C5-C7 | G5 | a wine glass ringing |
| handpan | struck | metal | C3-C5 | A3 | a handpan note |
| steel_drum | struck | metal | C4-E6 | C5 | a steel drum note |
| wood_block | struck | wood | C5-C7 | C6 | a wood block |
| cowbell | struck | metal | C4-C6 | C#5 | a cowbell |
| coin | struck | metal | C8-C10 | E9 | a coin ringing |
| glass_clink | struck | glass | C5-C7 | D6 | two glasses clinking |
| metal_ping | struck | metal | C5-C8 | C7 | a metal ping |
| felt_piano | strings | felt | A0-C8 | C4 | a soft felt piano note |
| grand_piano | strings | piano | A0-C8 | C4 | a grand piano note |
| electric_piano | strings | electric | E1-E7 | C4 | an electric piano note |
| reed_piano | strings | electric | A1-C7 | C4 | a reed electric piano note |
| harp | strings | string | C1-G7 | G4 | a harp note |
| nylon_guitar | strings | string | E2-B5 | A3 | a classical guitar note |
| steel_guitar | strings | string | E2-E6 | A3 | an acoustic guitar note |
| muted_guitar | strings | string | E2-E5 | E3 | a palm-muted guitar note |
| pizzicato | strings | string | C2-E6 | D4 | a pizzicato string note |
| koto | strings | string | C3-C6 | D4 | a koto note |
| upright_bass | strings | string | E1-G4 | D2 | an upright bass note |
| harpsichord | strings | string | F1-F6 | C4 | a harpsichord note |
| dulcimer | strings | string | D3-D6 | A4 | a hammered dulcimer note |
| banjo | strings | - | C3-D6 | D4 | a banjo note |
| soft_pluck | patches (pluck) | synth | C3-C7 | C5 | a soft rounded synthesizer pluck |
| glass_pluck | patches (pluck) | synth | C4-C7 | E5 | a bright glassy FM synthesizer pluck |
| muted_pluck | patches (pluck) | synth | E2-E5 | A3 | a muted synthesizer pluck |
| bubble | patches (pluck) | synth | C4-C7 | G5 | a soft bubble pop |
| warm_keys | patches (key) | synth | C2-C6 | C4 | warm analog synthesizer keys |
| fm_keys | patches (key) | synth | C2-C7 | C4 | a glassy FM electric piano note |
| fm_bell | patches (bell) | synth | C4-C7 | A5 | an FM synthesizer bell |
| mallet_synth | patches (key) | synth | C3-C7 | G4 | a synthesizer mallet note |
| square_lead | patches (lead) | synth | C3-C6 | A4 | a soft square wave synthesizer lead with vibrato |
| saw_lead | patches (lead) | synth | C3-C6 | A4 | a sawtooth synthesizer lead |
| warm_pad | patches (pad) | synth | C2-C6 | C4 | a warm analog synthesizer pad |
| air_pad | patches (pad) | synth | C3-C6 | C5 | a breathy airy synthesizer pad |
| glass_pad | patches (pad) | synth | C3-C6 | C5 | a glassy shimmering FM synthesizer pad |
| string_pad | patches (pad) | synth | C2-C6 | C4 | a string ensemble synthesizer pad |
| drone | patches (pad) | synth | A0-D3 | D2 | a dark low synthesizer drone |
| sub_swell | patches (pad) | synth | E1-E2 | A1 | a low sub bass swell |
| sub_bass | patches (bass) | synth | C1-C3 | G1 | a deep clean sub bass note |
| round_bass | patches (bass) | synth | C1-C4 | E2 | a round warm synthesizer bass note |
| reese_bass | patches (bass) | synth | C1-C3 | G1 | a reese bass |
| pluck_bass | patches (bass) | synth | C1-C4 | E2 | a plucked synthesizer bass note |
| kick_808 | patches (bass) | synth | C1-C3 | G1 | an 808 kick drum |
| braam | patches (hit) | synth | A0-A2 | D2 | a cinematic braam |
| stab | patches (hit) | synth | C3-C5 | C4 | a synthesizer chord stab |
| zap | patches (fx) | synth | A3-A5 | A4 | a laser zap |
| power_up | patches (fx) | synth | C4-C6 | C5 | a power-up sound |
| power_down | patches (fx) | synth | C4-C6 | C5 | a power-down sound |
| riser_tone | patches (fx) | synth | C3-C5 | C4 | a tonal synthesizer riser |
| noise_sweep | patches (fx) | synth | C4-C6 | C5 | a white noise sweep |

The pads, hits and effects patches are finished sounds (`pad/*`, `braam/*`, `zap/laser` ...), not `note/*` designs.
A sustaining voice (a metal bar, a bell, a pad) keeps every note of a figure at least 110 ms long, the last 1.6
times that: damped earlier it is heard as an electronic beep.

## Recipe: a web page

```bash
earshot make --set ui --theme soft -o public/sounds --format webm --sprite sprite.webm
```

One request loads every sound. With the Web Audio API and the map:

```js
const ctx = new AudioContext();
const map = await (await fetch('/sounds/sprite.json')).json();
const buf = await ctx.decodeAudioData(await (await fetch('/sounds/' + map.src[0])).arrayBuffer());
function play(name) {
  const [start, len, loop] = map.sprite[name];
  const src = ctx.createBufferSource();
  src.buffer = buf; src.connect(ctx.destination);
  if (loop) Object.assign(src, {loop: true, loopStart: start / 1000, loopEnd: (start + len) / 1000});
  src.start(0, start / 1000, loop ? undefined : len / 1000);
  return src;
}
document.querySelector('#save').addEventListener('click', () => { ctx.resume(); play('success'); });
```

With howler.js the map is the `sprite` option as it is: `new Howl({src: ['/sounds/sprite.webm'], sprite:
map.sprite}).play('success')`. Browsers start audio only after a user gesture, hence `ctx.resume()` in the click.
Not every browser decodes every codec: `pack` writes the same sounds as .mp3 or .m4a too (`earshot pack
public/sounds -o public/sprite.mp3`). Keep frequent sounds (taps, hovers) rare in practice; the set levels them low
already.

## Recipe: a game

```
$ earshot make "three variations of a footstep on gravel" -o step.ogg
step_1.ogg: step/gravel (0.30 s, -21 LUFS, clean) [gravel, footstep]
step_2.ogg: step/gravel (0.32 s, -18 LUFS, clean)
step_3.ogg: step/gravel (0.30 s, -19 LUFS, clean)
also: step/grass, step/metal, step/snow

$ earshot make "campfire" -o fire.ogg --dur 16
fire.ogg: fire/camp (16.00 s, -22 LUFS, seamless loop, clean) [fire]
also: fire/large, fire/small, match/strike
```

- **Variations.** A game plays a footstep hundreds of times: ask for 6-12 takes (`--n`, or "six variations") and
  pick one at random per play. Seeds are performances: two seeds differ in timing, strike points and events, not in
  what the sound is. Single takes are levelled to their own peak, so pick-at-random sets are better made as a set
  (`"n": 8` in a spec), where every take gets the kind's level.
- **Loops for levels.** Ambience is a seamless loop by default: the file repeats with no seam and no cut event
  (rendered around a circle). Longer loops repeat less obviously; each scene has a shortest sensible length.
- **One family.** `--set game` (modern, kalimba theme) or `--set arcade` (chip); add or replace sounds with a spec.
  Levels inside a set follow the kind table above: a coin sits under an explosion, a step under a coin.
- **Chip sound** only when the request asks for it ("8-bit", "retro", "chiptune", "arcade") or the theme is retro.

## Recipe: a film

```
$ earshot make "logo reveal" --voice felt_piano --pitch D4 -o logo.wav
logo.wav: logo/bloom (3.79 s, D4, felt_piano, -14 LUFS, clean) [logo]
also: logo/duo, logo/motif, logo_sting/reverse_bloom

$ earshot make "a forest at night" -o night.wav --no-loop --dur 10
night.wav: night/crickets (10.00 s, -22 LUFS, clean) [forest, night]
also: forest/morning, rain/leaves, wind/trees
```

- **Beds once, not looped**: `--no-loop` gives a scene with a fade in and out at the length the shot needs.
- **In the film's key**: `--pitch` or "in D" for logos, stingers and notes, so they sit with the music.
- **The bank roles with sync points**: `earshot make --set film` renders a starter kit as files. For effects placed
  on cue frames use the bank's kits (`earshot kit init --preset launch`, `kit render`): every take carries its
  measured onset, peak and end, and `kit.place`, `kit.gain_db`, `kit.bed_level` and `kit.space_run` place and level
  it against the music ([SFX.md](SFX.md)). A made file starts when its sound does, so a hit's file starts on its cue.
- **Check it landed**: `earshot timeline mix.wav --video film.mp4 --cues cues.json` and `earshot sheet` (the
  listening half, README).

## Add a design

A design is a function plus what an agent needs to choose it (`earshot/sounds/registry.py` holds the contract).
Put it in a module of `earshot/sounds/` and add the module's name to `registry.MODULES`:

```python
from earshot.sounds.registry import design
from earshot.synth import space, voices

@design('bell/tiny', kind='chime', says=('a tiny bright bell rung once', 'a small high ding'),
        uses=('notify', 'bell'), domains=('ui',), dur=0.6, pitch='E6', axes=('bright', 'energy', 'space'))
def bell_tiny(c, r):
    y = voices.play('hand_bell', c.hz(), vel=0.5 * c.db('energy', 6.0), dur=c.seconds(), r=r, vary=c.vary,
                    bright=c.bright)
    return space.room(y, c.room('studio'), amount=c.space, r=r)
```

`c` holds the controls (`c.hz()`, `c.seconds()`, `c.bright` ... in [-1, 1], `c.lerp(axis, lo, hi)`, `c.db(axis,
span)`, `c.voice`, `c.room()`), `r` is a numpy Generator (every random number comes from it: the same seed gives the
same samples). Return mono `(n,)` or stereo `(n, 2)` at 48 kHz at any level; `render` adds the low cut, the edge
fades, the tail trim, the level and the audit. `says` describes the sound as a listener would (never its job), `uses`
lists the jobs it plays, `axes` names exactly the axes the function reads, and `allows` lists audit findings that are
the design (`rising_glide`, `stacked`, `pre_swell`, `tail`, `truncated`). Every design should answer to `energy` and
two more axes in a way a listener would agree with, and two seeds should give different, equally good takes.
