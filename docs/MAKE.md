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
ends with the words that decided, as you wrote them (`[success, bell]`, or `[named]` for `--sound`). Then:

| line | meaning | what to do |
|---|---|---|
| `not in this sound: ...` | sounds or jobs the request named that the chosen design does not cover | ask for them in a second call, or pick from `also` |
| `also: ...` | up to three alternatives, always: the next ranked designs, then the nearest of the same family, kind or domain | take one with `--sound ID` if the pick does not fit the job |
| `also: ... (the pick is a guess: ...)` | no job or sound word matched (and, when listening, the request does not clearly sound like the pick), or another kind of sound scored almost as high; the alternatives then come from other families first | read the alternatives and name one with `--sound ID`, or re-ask naming the sound or the job |
| `not used by this sound: ...` | controls the request set that this design does not read, or a length it could not keep (a struck sound rings as it rings; a marimba cannot ring 2 s) | leave it, or ask for a design that has that control |
| `not understood: ...` | words that matched nothing in the catalogue or the lexicon | rephrase if they mattered |
| `x.mp3: a loop in mp3 has a gap at its seam ...` | a loop written as mp3 or m4a | write loops as ogg, webm, wav or flac |
| `heard as: '...' (z +x.x, verdict)` | with `--listen` only (below) | |
| `not listened: ...` | `--listen` was asked and the model or the atlas is missing | `earshot doctor` |

```
$ earshot make "a water drop and a coin" -o dropcoin.wav
dropcoin.wav: coin/bright (0.69 s, B5, glockenspiel, -14 LUFS, clean) [coin, pickup]
not in this sound: water drop
also: water/drop, water/pour, coins/handful (the pick is a guess: another kind of sound fits almost as well; name the one you want with sound=<id>)

$ earshot make "a marimba success chime, 2 seconds" -o mar2.wav
mar2.wav: success/rise (0.55 s, G5, marimba, -14 LUFS, clean) [success, bell]
also: success/triad, success/done, success/fifth
not used by this sound: dur
```

| option | what it does |
|---|---|
| `-o FILE` | the file; its extension picks the format (wav, flac, ogg, mp3, m4a, webm). None: `<family>_<name>.wav` here |
| `--n N` | N takes on seeds S .. S+N-1, written `name_1 .. name_N`; "three variations" in the words does the same |
| `--seed S` | another take of the same sound (default 0); the same arguments always give the same samples |
| `--sound ID` | a design id from `earshot sounds` instead of the ranking; the words still set feel and numbers |
| `--theme NAME` | a theme preset ([Themes](#themes)): voice, key, room and axis offsets shared by a product's sounds |
| `--pitch` | a note (`C5`) or Hz. Overrides the words |
| `--dur` | seconds: how long a figure of notes sounds, the hold of a pad, the length of a loop (each has a shortest one) or a whoosh; a single struck sound (a tap, a hit) rings as it rings and says `not used by this sound: dur` |
| `--voice` | the instrument or patch of a tonal sound ([Voices](#voices)); a voice asked for by name is always played |
| `--material` | what a tap, click or knock is made of; on a tonal sound without a voice it picks the voice made of it (wood: marimba, glass: glockenspiel, metal: vibraphone, soft: felt_piano, digital: glass_pluck, retro: square_lead) |
| `--bright` ... `--motion` | the axes, -1 .. 1, added to what the words say ([The eight axes](#the-eight-axes)) |
| `--lufs L` | a momentary-max loudness instead of the default level |
| `--mono` | sum to one channel |
| `--no-loop` | an ambience once, with a fade in and out, instead of a seamless loop |
| `--listen` | also rank by how each design sounds (the sound atlas, below) and say what the result is heard as |
| `--json` | the full result: files, chosen id, `why`, `also`, `not_in_sound`, `unknown`, every take's measurements and audit |

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

**`--listen` and the sound atlas.** `earshot/data/sounds_atlas.npz` (numbers only) holds a CLAP embedding of every
design's default render and of its own descriptions. With `--listen`, the request's CLAP text embedding is scored
against it (a z per design over the whole atlas) and blended into the ranking (`earshot.brief.blend`): a named job
keeps its word-based winner, a request that only describes a sound is decided by how the designs sound, and the hard
filters (chip sound, loop, a job ruled out) still hold. The written sound is then matched (`earshot match`) against
the request and its own descriptions, and the result gains `heard as: '<the description that fit best>' (z +x.x,
verdict)`. It loads CLAP once (20-40 s); the MCP server's `make` blends the atlas in without being asked once the
model is loaded in that process. `make --set` and `make_set` rank by words only. The atlas must be rebuilt after
designs are added or re-voiced: `python -m earshot.sounds.atlas build --work DIR`, repeated until it says it is done
(`python -m earshot.sounds.atlas status` checks it).

**Level.** A single sound is set to a true peak of -1 dBFS, but no louder than -14 LUFS momentary max (pads and
textures -18, beds -22): a dense sound at full peak is far louder than a tap at full peak. `--lufs` sets the
momentary-max loudness exactly, with the peaks limited to -1 dBFS. Files are 48 kHz stereo unless `--mono`; every
file starts when its sound does (within 2 ms) and ends faded, never cut.

**Formats.** wav and flac are written directly (24-bit; 16-bit wav in a set) and are lossless; ogg (Vorbis, 96k),
mp3 (128k), m4a (AAC, 96k) and webm (Opus, 64k) go through ffmpeg, which must be on PATH for them. For the web: Ogg
Vorbis does not play in Safari before 18.4, so write `webm,mp3` (or `webm,m4a`) and let the browser take the first
it can play. Loops: mp3 and m4a encoders pad the start and end and record it only in a note some players ignore, so
a loop written as mp3 or m4a has a gap at its seam there (`make` warns); write loops as ogg, webm, wav or flac.

```
$ earshot make "campfire" -o fire.mp3 --dur 16
fire.mp3: fire/camp (16.00 s, -22 LUFS, seamless loop, clean) [fire]
also: fire/large, fire/small, match/strike
fire.mp3: a loop in mp3 has a gap at its seam in players that ignore the encoder's padding note; for loops use ogg, webm, wav or flac
```

**Speed** (one thread, no models): a single `make` takes about 1-2 s including Python start-up; ambience loops of
12-20 s take 3-10 s to render.

In Python: `from earshot.make import make; make('a deep impact for a logo reveal', 'hit.wav', size=0.3)` returns the
same result as a dict (`summary`, `sound`, `files`, `why`, `also`, `not_in_sound`, `unknown`, `takes`, and `heard`
with `listen=True`). `make()` also takes `sr` and `bits` for the written file.

## A whole set

```
earshot make --set ui|game|arcade|film [-o FOLDER] [--theme NAME] [--format webm,mp3] [--sprite sprite] [--mono]
             [--lufs L] [--seed S]
earshot make --spec sounds.json [-o FOLDER] [--theme NAME] [--format ...] [--sprite ...]
```

```
$ earshot make --set ui --theme glass -o public/sounds --format webm,mp3 --sprite sprite
16 sounds in public/sounds (webm, mp3), manifest.json
tap.{webm,mp3}: tap/tone (0.47 s, D6, marimba, -26 LUFS, clean) [named]
hover.{webm,mp3}: hover/tone (0.42 s, D6, marimba, -28 LUFS, clean) [named]
toggle_on.{webm,mp3}: toggle/up (0.51 s, D6, glockenspiel, -19 LUFS, clean) [named]
...
success.{webm,mp3}: success/rise (0.83 s, D5, glockenspiel, -19 LUFS, clean) [named]
complete.{webm,mp3}: success/triad (1.08 s, D5, glockenspiel, -18 LUFS, clean) [named]
warning.{webm,mp3}: warning/pulse (0.72 s, D5, glockenspiel, -19 LUFS, clean) [named]
error.{webm,mp3}: error/fall (0.69 s, D5, glockenspiel, -19 LUFS, clean) [named]
delete.{webm,mp3}: delete/drop (0.54 s, D5, glockenspiel, -19 LUFS, clean) [named]
16 sounds in public/sounds\sprite.webm + sprite.mp3 (14.4 s, 150 kB, 226 kB), map sprite.json
```

Every sound of a set shares the theme: one voice, one key, one room. The tap and hover above play marimba, not the
theme's glockenspiel: one short note of a sustaining metal, glass or electric voice is heard as an electronic beep,
so the tap, the hover and the menu move use struck wood when the default or theme voice is one of those (a voice
asked for by name is always played). The smallest figures (tap, hover, toggles, next / back steps, menu move) also
fade out within a fixed time (0.14-0.32 s before the room, scaled by `ring`) whatever the voice, so a hover on a
piano does not ring for a second. The set above took about 20 s on one thread.

**Several formats.** `--format webm,mp3` (a list or a comma string in a spec, in the MCP tool) writes every sound
once per format; `manifest.json` names each sound's `file` (the first format) and `files` (all), and the set's
`formats`. A sprite named without an extension, or with one of the set's formats, is written in each format, and
its map's `src` lists them in that order: a browser or howler.js plays the first it can.

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
design, length, loudness, loop flag, pitch and whether its audit was clean. A sound asked for in words gets its
`not in this sound`, guess and `also` lines only where the pick is in doubt, so a set's answer stays short; the
result (`--json`) carries `also` and `not_in_sound` for every such sound.

**The spec** (a JSON file for `--spec`, or a dict for `make_set`; the MCP tool `make_set` takes `set`, `sounds`,
`theme`, `format` and `sprite`):

```json
{"set": "ui", "theme": {"base": "wood", "key": "D"}, "format": ["webm", "mp3"], "sprite": "sprite",
 "sounds": [{"name": "success", "want": "a happy two-note success chime", "dur": 0.5, "axes": {"bright": 0.2}},
            {"name": "coin", "want": "coin pickup", "n": 2},
            {"name": "rain", "sound": "rain/window", "dur": 12}]}
```

| key | meaning |
|---|---|
| `set` | start from a ready-made set; `sounds` then adds to it or replaces a sound of the same name |
| `theme` | a preset name or a theme dict (`base` + changes) |
| `format`, `out`, `mono`, `lufs`, `sprite` | file format(s) (a string, "webm,mp3" or a list), folder, mono, the loudest level, a sprite file name |
| `sounds[]` | each: `name` (the file name, required, unique) and `want` (words) and / or `sound` (an id), plus `pitch`, `dur`, `voice`, `material`, `seed`, `n`, `axes` ({axis: -1 .. 1}), `lufs` (this sound's own level) |

```
$ earshot make --spec spec.json -o myset
19 sounds in myset (webm, mp3), manifest.json
tap.{webm,mp3}: tap/tone (0.29 s, D6, marimba, -26 LUFS, clean) [named]
...
success.{webm,mp3}: success/rise (0.54 s, D5, marimba, -19 LUFS, clean) [2-note, success, bell]
coin_1.{webm,mp3}: coin/bright (0.34 s, D6, marimba, -19 LUFS, clean) [coin, pickup]
coin_2.{webm,mp3}: coin/bright (0.34 s, D6, marimba, -19 LUFS, clean) [coin, pickup]
rain.{webm,mp3}: rain/window (12.00 s, -30 LUFS, seamless loop, clean) [named]
rain: a loop in mp3 has a gap at its seam in players that ignore the encoder's padding note; for loops use ogg, webm, wav or flac
19 sounds in myset\sprite.webm + sprite.mp3 (24.3 s, 205 kB, 369 kB), map sprite.json
```

The rain loop plays without a seam from `rain.webm` and from the sprite's webm; in the mp3 it may not.

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
$ earshot make --set arcade -o assets/arcade --format ogg --sprite sprite.ogg
16 sounds in assets/arcade (ogg), manifest.json
...
16 sounds in assets/arcade\sprite.ogg (14.2 s, 111 kB), map sprite.json

$ earshot pack assets/arcade -o assets/arcade_all.webm
16 sounds in assets/arcade_all.webm (14.2 s, 150 kB), map arcade_all.json
```

`pack` lays sounds end to end in one file with `gap` seconds of silence between them (lossy codecs smear a little
in time; without a gap the tail of one sound leaks into the next) and writes the map next to it as
`<name>.json` in the format howler.js reads:

```json
{"src": ["arcade_all.webm"], "sprite": {"menu_move": [0, 192], "menu_select": [442, 280], "menu_back": [972, 234], ...}}
```

Each entry is `[offset_ms, duration_ms]`, and `[offset_ms, duration_ms, true]` for a loop; loops get four times the
gap. Given a folder with a `manifest.json` (a set made by `make`), it packs exactly the manifest's sounds in the
manifest's order and leaves out an earlier sprite (any file a sprite map in the folder names), as above: 16 sounds,
not the earlier `sprite.ogg` as a seventeenth. A folder without a manifest packs every audio file in it by name,
again leaving out sprites. `make --set ... --sprite` packs what it rendered, not the lossy files decoded again. In
Python: `from earshot.pack import pack; pack('assets/arcade', 'assets/arcade_all.webm')`, or a list of outputs
(`['sprite.webm', 'sprite.mp3']`) for the same sprite in several formats.

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
  note of that voice; one named as the player ("a success chime on a kalimba") sets the voice of a figure, and a
  voice asked for by name is always played. Material words ("wooden", "glass", "metal", "soft", "digital",
  "8-bit") steer the ranking toward designs of that material, and on a tonal design with no voice asked for they
  pick the voice made of it: wood a marimba, glass a glockenspiel, metal a vibraphone, soft a felt piano, digital a
  glass pluck, retro a square lead (`registry.VOICE_OF`; this wins over the theme's voice). A voice or material
  that chose the design is not reported as unused.
- **Numbers.** Lengths ("400 ms", "2 seconds", "half a second") set how long a figure sounds before its room dies
  away, how long a pad is held, how long a loop or a whoosh is; a length a design cannot keep (a marimba cannot
  ring 2 s, a struck sound has no length) is reported as `not used by this sound: dur`. "Under 300 ms" is a
  ceiling: it prefers shorter designs and shortens only a design that is longer, as a length. A pitch ("C6", "440
  Hz", "middle C"), a key ("in D", "in the key of Bb minor"), how many takes ("three variations", "5 versions", "a
  few takes"; at most 12) and how many notes ("a two-note chime", "three rising notes", "twice"). A key moves a
  tonal design's root to the nearest note of that key; a minor key (in the request or the theme) turns a figure's
  major thirds, sixths and sevenths into minor ones.
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

What reached the render, on real calls:

```
$ earshot make "a soft wooden success chime for a finance app, 400 ms, in D" -o wood.wav
wood.wav: success/rise (0.49 s, D5, marimba, -14 LUFS, clean) [success, bell]
also: success/done, success/fifth, success/triad

$ earshot make "a success chime on a kalimba, softer" -o kal.wav
kal.wav: success/rise (0.69 s, G5, kalimba, -14 LUFS, clean) [success, bell]
also: success/done, success/fifth, success/triad

$ earshot make "a two-note success chime in D minor" -o minor.wav
minor.wav: success/rise (0.90 s, D5, vibraphone, -14 LUFS, clean) [2-note, success, bell]
also: success/fifth, success/done, success/triad

$ earshot make "a kalimba note in E" -o kalimba.wav
kalimba.wav: note/kalimba (1.06 s, E5, -14 LUFS, clean) [kalimba, note]
also: pluck/kalimba, pluck/kalimba_ring, pluck/kalimba_soft
```

"Wooden" played a marimba, "400 ms" made the figure 0.49 s with its room, and the key D moved the root from G5 to
D5. In D minor, `success/rise` (a major third up) rises a minor third.

**How well requests are understood.** Three sets of requests were written by someone who never saw the catalogue: A
(440 requests for 110 intents, in four styles) and B (220) were used to develop the reader; C (220: half long
descriptions of a sound that never name it, half casual requests) was held out and scored once, at the end. A
request counts as right when the first pick belongs to a family accepted for its intent.

| set | first pick right | right among the top three |
|---|---|---|
| A (development, 440) | 92.5 % | 95.9 % |
| B (development, 220) | 97.3 % | 99.1 % |
| C (held out, 220) | 71.4 % | 80.5 % |
| C with the atlas blended in (`--listen`) | 74.1 % | 82.3 % |

Terse requests that name the job or the sound ("coin pickup", "error beep") are right for all 110 intents of set A.
On C, by group (first pick): interface 54.5 %, game 70.9 %, motion 60.0 %, ambience 87.5 %, instruments 83.3 %,
objects 100 %. Name the sound or the job and the pick is right; describe a mood or paraphrase and one request in
four needs a second call, which is what `also`, the guess note and `--sound ID` are for. The guess note flags about
a third of wrong first picks and about one in ten right ones.

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
| `material` | the material of the theme's taps and clicks (recorded; the voice is what plays the figures) |
| `says` | a description, for people |

A minor scale (`minor`, `minor_pentatonic`) turns every figure's major thirds, sixths and sevenths into minor ones;
the other scales leave the figures' intervals as designed. The words of a request lay their key and scale over the
theme's, and a material or voice in the request wins over the theme's voice. The tap, hover and menu move give way
to struck wood on a metal, glass or electric theme voice (as above).

## The catalogue, family by family

`earshot sounds` lists 457 designs in 143 families: 248 written for 1.0 and the 209 named sounds of the earlier bank.
Ids are `family/name`; the family is what the sound is or does, the name its character.

| group | designs | ids | for | controls |
|---|---|---|---|---|
| interface figures (`sounds/ui.py`) | 25 | `success/rise`, `error/fall`, `notify/pair`, `toggle/up`, `open/lift`, `send/lift`, `tap/tone`, `hover/tone`, `purchase/chime`, `like/twinkle` ... | confirmations, errors, warnings, notifications, toggles, navigation | all 8 axes, pitch (the root), voice or material, dur |
| game figures (`sounds/game.py`) | 15 | `coin/bright`, `pickup/blip`, `gem/sparkle`, `power_up/rise`, `level_up/fanfare`, `win/fanfare`, `lose/descend`, `menu/select` ... | rewards, signals, jingles, menus | all 8 axes, pitch, voice or material, dur |
| brand figures (`sounds/brand.py`) | 5 | `logo/bloom`, `logo/duo`, `logo/motif`, `startup/rise`, `shutdown/fall` | sonic logos, openings, endings | all 8 axes, pitch, voice or material, dur |
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
A sustaining voice (an instrument that rings over a second, or a bell, pad, key or lead patch) keeps every note of a
figure at least 110 ms long, the last 1.6 times that: damped earlier it is heard as an electronic beep.

## Recipe: a web page

```bash
earshot make --set ui --theme soft -o public/sounds --format webm,mp3 --sprite sprite
```

Ogg Vorbis does not play in Safari before 18.4; webm (Opus) with an mp3 (or m4a) beside it plays everywhere, and the
map lists `sprite.webm` first. One request loads every sound. With the Web Audio API and the map:

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

The snippet loads the first file of `src`; to fall back, try `map.src[1]` when decoding fails. With howler.js the
map is the options as they are: `new Howl({src: map.src.map(f => '/sounds/' + f), sprite:
map.sprite}).play('success')`, and howler picks the first format the browser plays. Browsers start audio only after
a user gesture, hence `ctx.resume()` in the click. Keep frequent sounds (taps, hovers) rare in practice; the set
levels them low already. A loop in a sprite is safe in the webm; not in the mp3 (the seam).

## Recipe: a game

```
$ earshot make "three variations of a footstep on gravel" -o step.ogg
step_1.ogg: step/gravel (0.31 s, -21 LUFS, clean) [gravel, footstep]
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
  (rendered around a circle). Longer loops repeat less obviously; each scene has a shortest sensible length. Ship
  loops as ogg, webm, wav or flac, not mp3 or m4a.
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
two more axes in a way a listener would agree with, and two seeds should give different, equally good takes. Then
rebuild the sound atlas (`python -m earshot.sounds.atlas build --work DIR`, repeated until done) so `--listen` knows
the new design, and run the catalogue's contract tests (`tests/test_sounds_catalogue.py`).
