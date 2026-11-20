# Making sounds: `make`, `sounds`, `pack`

An agent asks for a sound in words; earshot reads the request, ranks its catalogue of designs, renders a new take of
the best one, audits it without models and writes the file. No model is loaded and nothing is downloaded. A design is
either written in code (figures, chip sound, the drum-machine voices, ambience from rules, notes, the music loops)
or learned from recordings in the public domain and kept as numbers ([Learned sounds](#learned-sounds)): everyday
things, the interface sounds, the film and motion effects, most ambience beds, the acoustic drums and hand
percussion. Either way every sound is computed when it is asked for; no audio file is shipped or played back.
Everything rendered belongs to whoever rendered it: free for any use, commercial too, without credit.

- [make](#make) - [learned sounds](#learned-sounds) - [make --set / --spec](#a-whole-set) - [sounds](#sounds) -
  [pack](#pack)
- [The request language](#the-request-language) - [The eight axes](#the-eight-axes) - [Themes](#themes)
- [Recipes: build what is missing](#when-earshot-has-no-such-sound-build-it)
- [The catalogue](#the-catalogue-family-by-family) - [Voices](#voices) - [The engines](#the-engines)
- Recipes: [a web page](#recipe-a-web-page) - [a game](#recipe-a-game) - [a film](#recipe-a-film) -
  [add a design](#add-a-design)

## make

```
earshot make "WORDS" [-o FILE] [--n N] [--seed S] [--theme NAME] [--sound ID] [--pitch NOTE] [--dur S]
             [--voice NAME] [--material NAME] [--bright X] [--hard X] [--ring X] [--size X] [--space X]
             [--width X] [--energy X] [--motion X] [--lufs L] [--mono] [--no-loop] [--listen] [--json] [--out F]
```

```
$ earshot make "a soft success sound for a finance app" -o success.ogg
success.ogg: success/shutter (0.36 s, peak at 0.10 s, -15 LUFS, clean) [success]
also: complete/shutter, success/click, complete/chop

$ earshot make "a mouse click" -o click.wav --n 2
click_1.wav: click/mouse (0.19 s, peak at 0.01 s, -20 LUFS, clean) [mouse click]
click_2.wav: click/mouse (0.36 s, peak at 0.14 s, -19 LUFS, clean)
also (the pick is a guess: another kind of sound fits almost as well): mouse/squeak (a mouse squeaking), pen/click (a ballpoint pen clicked), switch/small (a small switch flipped). Name one with sound=<id>
```

The answer is the whole result. Each file gets one line: the design chosen, its length, where its loudest moment
is (`peak at`, seconds from the start of the file), the note and voice it played (tonal designs), its momentary-max
loudness, `seamless loop` for a loop, and `clean` or the audit's warnings;
the first line ends with the words that decided, as you wrote them (`[success, chime]`, or `[named]` for `--sound`).
Then:

| line | meaning | what to do |
|---|---|---|
| `not in this sound: ...` | sounds or jobs the request named that the chosen design does not cover | ask for them in a second call, or pick from `also` |
| `also: ...` | up to three alternatives, always: the next ranked designs, then the nearest of the same family, kind or domain | take one with `--sound ID` if the pick does not fit the job |
| `also (the pick is a guess: ...): id (a few words), ... Name one with sound=<id>` | no job or sound word matched (and, when listening, the request does not clearly sound like the pick), or another kind of sound scored almost as high, or the thing or what it does is a word earshot does not know (`earshot makes no dial up modem; dial/click is only the nearest sound it has`); the alternatives then come from other families first, each with the start of its description | name one with `--sound ID`, or describe the sound by what it is made of ("a screeching run of electronic tones and static") instead of by its name, or tell the user that this thing is missing |
| `no rain on a tarp: rain/tent is the nearest surface; also: ...` | the sound exists, but not on the surface, ground or place said after "on", "in", "against" or "into"; the pick is the same sound on the nearest surface it has, the alternatives its other surfaces | leave it, or name another surface with `--sound ID` |
| `not followed: "slow"; this sound has no control of its speed ...` | a control the request asked for (in words or as an argument) that this design does not have; the line says what it does instead. A word that named the design ("fast" in "a fast whoosh") is never listed | leave it, or ask for a design that has that control |
| `0.43 s, not the 2 s asked: a single struck sound rings as long as it rings` | a length a one-shot cannot keep | leave it, or ask for a whoosh, a figure or a bed |
| `a cut of 4 s, not a loop: the shortest seamless loop of this sound is 8.00 s` | an ambience asked shorter than its shortest loop: the length asked, cut from the bed with a 30 ms fade in and a fade out (10 % of it, at most 0.5 s) | say "a seamless loop of ..." if it must repeat |
| `a seamless loop of 8.00 s, the shortest this sound has; 4 s was asked ...` | the request said "loop" or "seamless": the seam wins over the length | nothing, or drop "loop" for a cut |
| `8.57 s: a music loop is whole bars, and these are the nearest to the 8 s asked` | a music loop keeps its bar grid | nothing |
| `9.87 s, shorter than the 10 s asked: the longest this sound is made` | a looping design came out shorter than the length asked | nothing, or ask for a loop |
| `3 knocks 1.5 s apart: 3.25 s, not the 10 s asked (knocks further apart than 1.5 s are no longer one action)` | a counted action whose events cannot fill the length asked: they are played at the widest (or closest) gap a person keeps, or a countdown at one a second, and the file is as long as that ([Performances](#performances)) | nothing, or ask for more or fewer events |
| `not understood: ...` | words that matched nothing in the catalogue or the lexicon | rephrase if they mattered |
| `x.mp3: a loop in mp3 has a gap at its seam ...` | a single `make` of a loop to mp3 or m4a | write loops as ogg, webm, wav or flac |
| `  a loop: flac instead of mp3, which leaves a gap at the seam` | in a set with several formats, a loop was written as flac in place of mp3 / m4a | nothing |
| `  same design as NAME: ask for a different one with "sound" (the nearest: ID)` | in a set, two names got the same sound | in your own spec: give the second one `"sound": ID`, or reword its `want` |
| `  another recording of ID than NAME: recording N of M` | in a set, two names play different recordings of one learned design (toggle_on and toggle_off, back and next in `ui`) | nothing: the pair is meant |
| `heard as: '...' (z +x.x, verdict)` | with `--listen` only (below) | |
| `not listened: ...` | `--listen` was asked and the model or the atlas is missing | `earshot doctor` |

A guess, as `make` writes it (the alternatives without a guess stay bare ids):

```
$ earshot make "a frobnicating zorp" -o x.wav
x.wav: thump/soft (0.09 s, peak at 0.02 s, -19 LUFS, clean)
also (the pick is a guess: earshot makes no frobnicating zorp; thump/soft is only the nearest sound it has): knock/wood (a knock on a wooden block), drill/electric (an electric drill whirring), whoosh/soft (a soft short whoosh). Name one with sound=<id>
build it from parts: earshot recipe --example
not understood: frobnicating, zorp
```

```
$ earshot make "a marimba success chime, 2 seconds" -o mar2.wav
mar2.wav: success/rise (1.85 s, peak at 0.20 s, G4, marimba, -14 LUFS, clean) [success, chime]
also: success/triad, success/done, success/fifth
```

| option | what it does |
|---|---|
| `-o FILE` | the file; its extension picks the format (wav, flac, ogg, mp3, m4a, webm). None: `<family>_<name>.wav` here |
| `--n N` | N takes on seeds S .. S+N-1, written `name_1 .. name_N`; "three variations" in the words does the same. For a learned design these are its other recordings first |
| `--seed S` | another take of the same sound (default 0); the same arguments always give the same samples |
| `--sound ID` | a design id from `earshot sounds` instead of the ranking; the words still set feel and numbers |
| `--theme NAME` | a theme preset ([Themes](#themes)): voice, key, room and axis offsets shared by a product's sounds |
| `--pitch` | a note (`C5`) or Hz. Overrides the words |
| `--dur` | seconds: how long a figure of notes sounds, the hold of a pad, the length of a loop (each has a shortest one) or a whoosh; a learned sound within 0.6-1.8 times its natural length; a counted action ("three knocks", footsteps) is a file of that length with its events spread over it; a single struck sound (a tap, a hit) rings as it rings and says so (`0.43 s, not the 2 s asked ...`); an ambience asked shorter than its shortest loop is a cut, not a loop (unless the words say "loop"), a music loop keeps whole bars |
| `--voice` | the instrument or patch of a tonal sound ([Voices](#voices)); a voice asked for by name is always played |
| `--material` | what a tap, click or knock is made of; on a tonal sound without a voice it picks the voice made of it (wood: marimba, glass: glockenspiel, metal: vibraphone, soft: felt_piano, digital: glass_pluck, retro: square_lead) |
| `--events N` | for a thing a person does several times in a row (footsteps, a knock, typing, coins, keys, clinking glasses): 1 = one event, 2 or more = the action with that many events; without it the words decide ([Performances](#performances)) |
| `--bright` ... `--motion` | the axes, -1 .. 1, added to what the words say ([The eight axes](#the-eight-axes)) |
| `--lufs L` | a momentary-max loudness instead of the default level |
| `--mono` | sum to one channel |
| `--no-loop` | an ambience once, with a fade in and out, instead of a seamless loop |
| `--listen` | also rank by how each design sounds (the sound atlas, below) and say what the result is heard as |
| `--json` | the full result: files, chosen id, `why`, `also`, `not_in_sound`, `unknown`, every take's measurements, audit and, for a learned design, the recording it played (`member`) |
| `--out F` | also write that JSON result to the file F |

**What happens.** `earshot.brief` reads the words into a brief (the thing, job or sound, the feel as axis offsets, a
material or voice, numbers, the key); `brief.rank` scores every design; the best is rendered with those controls
(`earshot.sounds.registry.render`), finished (low cut at 25 Hz, leading silence dropped, a 0.4 ms fade-in, the
silent end trimmed and faded, the level below) and audited by `forge.sfx.check` (clicks, cut tails, clipping, DC,
comb ring, a pitch that climbs more than 4 semitones, a hit with stacked attacks or a build before it; findings a
design declares as its own, such as a power-up's rising pitch, are notes). A take that fails is rendered again on
seed + 101, + 202, + 303; if all four fail the first is written and its line names the warnings. A request that
matches nothing is no error: it gets the nearest sound there is, and the guess line says that it is only that
(`earshot make "a frobnicating zorp" -o x.wav` writes `thump/soft` and says `also (the pick is a guess: earshot
makes no frobnicating zorp; thump/soft is only the nearest sound it has): ...`).

**`--listen` and the sound atlas.** `earshot/data/sounds_atlas.npz` (numbers only) holds a CLAP embedding of every
design's default render and of its own descriptions. With `--listen`, the request's CLAP text embedding is scored
against it (a z per design over the whole atlas) and blended into the ranking (`earshot.brief.blend`): a named job
keeps its word-based winner, a request that only describes a sound is decided by how the designs sound, and the hard
filters (chip sound, loop, a job ruled out) still hold. The written sound is then matched (`earshot match`) against
the request and its own descriptions, and the result gains `heard as: '<the description that fit best>' (z +x.x,
verdict)`. It loads CLAP once (20-40 s); the MCP server's `make` blends the atlas in without being asked once the
model is loaded in that process. `make --set` and `make_set` rank by words only. The atlas must be rebuilt after
designs are added or changed: `python -m earshot.sounds.atlas build --work DIR`, repeated until it says it is done
(`python -m earshot.sounds.atlas status` checks it). The `heard as` line is what a model hears, not a judgement of
quality: judgements by ear did not follow the models' scores ([VALIDATION.md](VALIDATION.md#listening-by-ear)).

**Level.** A single sound is set to a true peak of -1 dBFS, but no louder than a momentary-max loudness by kind:
-14 LUFS, pads and textures -18, ambience beds -22, music loops -16. A dense sound at full peak is far louder than a
tap at full peak, so short sounds are brought forward: a one-shot under 1 s whose one spike would hold its body far
down is raised by up to 14 dB, until its loudest 5 ms sit 3.5 dB under the peak, and the spike above is folded over
smoothly rather than limited, so the strike stays as bright as it was (`master.finish`, `registry.LIFT`). An
ambience bed held under its cap by a few peaks (a fire's cracks) is raised the same way by up to 8 dB. `--lufs` sets
the momentary-max loudness exactly, with the peaks limited to -1 dBFS. Files are 48 kHz stereo unless `--mono`;
every file starts when its sound does (within 2 ms) and ends faded, never cut.

**Formats.** wav and flac are written directly (24-bit; 16-bit wav in a set) and are lossless; ogg (Vorbis, 96k),
mp3 (128k), m4a (AAC, 96k) and webm (Opus, 64k) go through ffmpeg, which must be on PATH for them. For the web: Ogg
Vorbis does not play in Safari before 18.4, so write `webm,mp3` (or `webm,m4a`) and let the browser take the first
it can play. Loops: mp3 and m4a encoders pad the start and end and record it only in a note some players ignore, so
a loop written as mp3 or m4a has a gap at its seam there. A single `make` to mp3 or m4a warns; a set with several
formats writes its loops as flac in place of mp3 / m4a. Write loops as ogg, webm, wav or flac.

```
$ earshot make "campfire" -o fire.mp3 --dur 16
fire.mp3: fire/camp (16.00 s, peak at 14.33 s, -23 LUFS, seamless loop, clean) [campfire]
also: fire/large, fire/small, fire/whoosh
fire.mp3: a loop in mp3 has a gap at its seam in players that ignore the encoder's padding note; for loops use ogg, webm, wav or flac
```

**Speed** (one thread, no models): a one-shot takes about a second plus Python start-up. A learned ambience bed
renders in 1.0-1.7 s of CPU per second of sound, so a 30 s bed takes 30-50 s; the beds from rules are faster.

In Python: `from earshot.make import make; make('a door shutting', 'door.wav', size=0.3)` returns the same result
as a dict (`summary`, `sound`, `files`, `why`, `also`, `not_in_sound`, `unknown`, `takes`, and `heard` with
`listen=True`). `make()` also takes `sr` and `bits` for the written file.

Every take (`takes` in Python, `--json`, the MCP result) also says where things happen, in seconds from the start
of the file: `peak_s` (the loudest moment), `onset_s` (where the sound starts to be heard) and, for a row of events
(footsteps, knocks, `--events N`), `events_s` (each event's start); the plain line shows the peak. The hit goes on
the frame at t: start the file at t minus `peak_s`. "a whoosh for a transition" answered `whoosh/sweep (0.42 s, peak
at 0.30 s, ...)`: for its loudest moment on 12.000 s, start it at 11.70 s.

## Learned sounds

309 designs of everyday, interface, game and film sounds, animals, people, street, house and kitchen sounds, 25
ambience beds, 33 drum and percussion designs and the notes of four instruments are learned from recordings whose authors put them in the public domain. They are not
invented and they are not the recordings: each recording is kept
as a few kilobytes of numbers (`earshot/synth/resyn.py`: 56 bands from 40 Hz to 20 kHz, each band's level every
2 ms and every 0.25 ms around an onset, how much of each band is one tone and at which frequency, the stereo image),
and every render draws fresh noise on those numbers. A take is a new performance of that recording; it is not a
sound made from nothing, and it exists only because someone recorded the thing and gave the recording away.

| | one-shots | ambience beds | drums | instrument notes |
|---|---|---|---|---|
| designs | 309 (`earshot/sounds/learned.py`) | 25 (`earshot/sounds/nature.py`) | 33 (`earshot/sounds/drums.py`) | the voices `marimba`, `harp`, `pizzicato`, `felt_piano` (`earshot/synth/played.py`) |
| recordings | 692 recorded members (44 hold several events, such as a whole walk) | 24 textures | 130 strokes | 97 single notes (marimba 14, harp 24, pizzicato 45, upright piano 14) |
| store | `earshot/data/resyn/`: 17.93 MB, `index.json`, `CREDITS.md` | `earshot/data/textures/`: 9.46 MB, `index.json`, `CREDITS.md` | `earshot/data/drums/`: 1.31 MB, `index.json`, `CREDITS.md` | `earshot/data/notes/`: 0.83 MB, `index.json`, `CREDITS.md` |
| sources | BigSoundBank, Kenney, VCSL, VSCO-2 CE, Freesound | BigSoundBank, Freesound | VCSL, Karoryfer, Freesound (the hand claps) | VSCO-2 CE, VCSL |
| a take | the member the seed picks, fresh noise, the request's controls | takes stitched endlessly (`earshot/synth/quilt.py`) | the stroke the seed picks near the velocity `energy` asks for | the note nearest the pitch and dynamic asked for, moved to the pitch |

All the sources together: BigSoundBank, Kenney, VCSL, VSCO-2 CE, Karoryfer and CC0 uploads on Freesound, all CC0 or
an explicit public-domain dedication; every recording is credited in the `CREDITS.md` of `earshot/data/resyn/`,
`textures/`, `drums/` and `notes/`. No audio file is shipped or played back.

**A design is one thing; its recordings are its variations.** `step/pebbles` holds nine recordings of footsteps
on gravel, `switch/lamp` three recordings of lamp switches, `keyboard/laptop` several keys. Seeds 0, 1, 2 ... walk
through the members in a fixed order (`walk` in the index: the member with the best listening score first, a member
scored 1 of 5 left out), so `--n 3` gives three different real footsteps; seeds past the last member are new takes
of the same recordings, and a take that fails its audit is rendered again from the same recording. A request at bright -0.5 or below (or 0.5 and above) walks only the
darker (brighter) half of the members, by their measured centroid, and tilts the take as well. `--json` says which
recording each take played, by its number (the titles stay in the credits):

```
$ earshot make "a footstep on pebbles" -o peb.wav --n 3 --json
... "member": {"n": 6, "of": 9, "recording": "recording 6 of 9"}
... "member": {"n": 7, "of": 9, "recording": "recording 7 of 9"}
... "member": {"n": 8, "of": 9, "recording": "recording 8 of 9"}
```

In Python: `y, meta = sounds.render('step/heels', seed=1)` and `meta['member']`.

**The controls are transforms of the model**, so they stay physical:

| control | what it does to a learned sound |
|---|---|
| `size` | the thing 0.7 .. 1.4 times as big: every frequency divided and every duration multiplied by it |
| `bright` | a spectral tilt of up to 2 dB per octave around 1 kHz, and the darker or brighter members |
| `hard` | the strikes 6 dB softer and duller .. louder and brighter |
| `ring` | the ringing bands decay 0.5 .. 2 times as long |
| `energy` | level +-6 dB and a harder, slightly brighter strike |
| `dur` | the time between onsets warped, onsets as sharp as before, within 0.6-1.8 times the natural length (beyond that the grain of the envelopes turns into a buzz or a flutter) |
| `pitch` | tonal designs only: moved to the note asked for, at most 7 semitones |
| `space`, `width` | the shared room; the stereo image (a mono recording is spread by decorrelation) |

Short dry members (clicks, taps, ticks, pops, knocks under 0.6 s) get a gentle finish (some body, presence and air,
no limiter); airy sounds never do: an airy whoosh sounded worse with it.

**Which requests reach them.** A learned design answers only a request that names its thing ("a footstep on gravel",
"a camera shutter", "a wooden drawer"), the interface job it plays (through the interface designs: "hover" is a
fingertip on wood, "toggle" a rocker switch, "next" a page flicked forward), or its family's own word. A feeling
alone never reaches a recording, and a request for "a heavy impact" does not reach a recorded dumbbell. Where a
learned and an invented design answer the same request, the learned one wins a tie (`registry.LEARNED` = 1.1);
learned designs that the store marks as not a default choice (`"default": false` in its index) rank lower (0.6
times that prior).

### Performances

A footstep, one knuckle on a door or one key is heard as a fragment; what a listener recognises is the action. A
learned thing a person does several times in a row (a walk, a knock, typing, coins, keys, clinking glasses) is
played as that action when the request asks for it (`earshot/synth/perform.py`, `learned.perform`): a recording of
the whole action where the store has one (walks on gravel, wood, tiles and in heels, a knock on a door and on a table, toasts, keys, coin handfuls, a dog barking, a heartbeat, a car horn, a
hammer on a nail), else its single recordings in
a pattern: `walk` (two feet in turn, 100-120 steps a minute; the feet are the two single steps of one session most
alike in loudness, brightness and length, so one walker does not limp, and a gravel walk recorded at the feet is
heard from standing height), `knock` (three or four hits about 0.2 s apart),
`type` (5-8 keys a second, irregular gaps, a deeper space bar between words), `scatter` (impacts that come sooner
and softer), `shake` (dense impacts in two or three swings), `pair` (an event and its answer). Never the same
recording twice in a row, never on a grid, every event its own take, one room for all.

| the request | what is played |
|---|---|
| a plural, an action word, a count or a length: "footsteps on gravel", "knocking", "three knocks", "5 seconds of typing", "typing on a keyboard" | the action (a length is the file's, below) |
| a bare singular or "a single": "a footstep on gravel", "a single knock on a door" | one event |
| a singular that names the action: "a knock on the door", "a toast" | the action |
| `--events N` (`events=` in Python and in the MCP tool) | 1: one event; 2 or more: the action with N events |

The answer says what was played (`played: 7 events as a walk at 1.8 a second; one event: events=1 or "a single
..."`) and a take's `meta['performance']` holds it.

A counted action is one file for any one-shot design, recorded as an action or not (`make.row`): a count of
knocks, steps or footsteps, claps, taps, clicks, beeps or bleeps ("four knocks on the table", "three claps", "two
mouse clicks"), or a countdown "3 2 1", plays that many events in one file, each event a new seed of the design and
a little firmer or softer, early or late, in the `knock`, `walk`, `row` (2.5-3.5 a second) or `count` (one a
second, on the clock) pattern (`lexicon.ROW_WORDS`). A count of anything else is that many separate files, each its
own take: "five coin pickups", "3 variations", "two versions", "a few takes". `--events N` plays any one-shot as a
row of N; "a single ..." and `--events 1` stay one event. Loops, beds, pads and music are never played as a row.

```
$ earshot make "four knocks on the table" -o knocks.wav
knocks.wav: knock/table (0.96 s, peak at 0.02 s, -14 LUFS, clean) [knocks table]
played: 4 events as a knock at 4.8 a second; one event: events=1 or "a single ..."
also: knock/desk, knock/door, knock/soft
```

A length asked with an action is the length of the file, recorded design or not: the first event at the start,
the last early enough to ring out (the design's usual length, 0.35 to 0.6 s, before the end), the gaps even around
the pattern's own small spread (`perform.fill`). Without a count ("footsteps on gravel, 5 seconds") the count is
what the pattern's natural pace fits into the length. A countdown stays one a second. Where the gaps would be closer
than a hand plays (0.12 s for knocks, claps, taps and clicks, 0.25 s for steps) or further apart than 1.5 s, they
are held there, the file is as long as the events then need, and one line says so. The result line shows the real
length; `played:` the count and pace.

The action form of a row word as the sound ("clapping, 4 seconds", "clicking", "knocking") is the action: a row at
the pattern's pace over the length asked, or about 2 to 3 s; a scatter or a shake settles or swings over a length, and
a recording of the whole action plays its busiest stretch of it, or says it is shorter than asked.

```
$ earshot make "three knocks on the table, 2 seconds" -o knocks2.wav
knocks2.wav: knock/table (2.00 s, peak at 0.02 s, -18 LUFS, clean) [knocks table]
played: 3 events as a knock at 1.2 a second; one event: events=1 or "a single ..."
also: knock/desk, knock/door, knock/soft
```

**Credits.** Every member is credited with its title, author, page and licence in
`earshot/data/resyn/CREDITS.md`, `earshot/data/textures/CREDITS.md`, `earshot/data/drums/CREDITS.md` and
`earshot/data/notes/CREDITS.md` (`learned.credit(id, member)` in Python). CC0 asks for no credit; the files keep it.

## A whole set

```
earshot make --set ui|game|arcade|film [-o FOLDER] [--theme NAME] [--format webm,mp3] [--sprite sprite] [--mono]
             [--lufs L] [--seed S]
earshot make --spec sounds.json [-o FOLDER] [--theme NAME] [--format ...] [--sprite ...]
```

```
$ earshot make --set ui -o public/sounds --format webm,mp3 --sprite sprite
17 sounds in public/sounds (webm, mp3), manifest.json
tap.{webm,mp3}: keyboard/mechanical (0.32 s, peak at 0.02 s, -18 LUFS, clean) [named]
hover.{webm,mp3}: hover/fingertip (0.34 s, peak at 0.07 s, -22 LUFS, clean) [named]
toggle_on.{webm,mp3}: toggle/key_down (0.24 s, peak at 0.03 s, -19 LUFS, clean) [named]
toggle_off.{webm,mp3}: toggle/key_up (0.21 s, peak at 0.04 s, -18 LUFS, clean) [named]
type.{webm,mp3}: type/mechanical (0.30 s, peak at 0.01 s, -19 LUFS, clean) [named]
back.{webm,mp3}: back/turn (0.51 s, peak at 0.05 s, -14 LUFS, clean) [named]
next.{webm,mp3}: next/page (0.54 s, peak at 0.33 s, -15 LUFS, clean) [named]
open.{webm,mp3}: open/drawer (0.68 s, peak at 0.39 s, -14 LUFS, clean) [named]
close.{webm,mp3}: close/drawer (0.73 s, peak at 0.48 s, -17 LUFS, clean) [named]
send.{webm,mp3}: send/swoosh (0.33 s, peak at 0.03 s, -21 LUFS, clean) [named]
receive.{webm,mp3}: receive/cork (0.41 s, peak at 0.03 s, -19 LUFS, clean) [named]
notify.{webm,mp3}: notify/knock (0.40 s, peak at 0.01 s, -18 LUFS, clean) [named]
success.{webm,mp3}: success/shutter (0.36 s, peak at 0.10 s, -15 LUFS, clean) [named]
complete.{webm,mp3}: complete/shutter (0.59 s, peak at 0.08 s, -14 LUFS, clean) [named]
warning.{webm,mp3}: warning/knock (0.48 s, peak at 0.01 s, -19 LUFS, clean) [named]
error.{webm,mp3}: error/knock (0.58 s, peak at 0.19 s, -18 LUFS, clean) [named]
delete.{webm,mp3}: delete/snip (0.66 s, peak at 0.39 s, -14 LUFS, clean) [named]
17 sounds in public/sounds\sprite.webm + sprite.mp3 (11.7 s, 101 kB, 183 kB), map sprite.json
```

Every sound of the `ui` set is a normal sound, a thing a hand does with an object, built from learned recordings
(`earshot/sounds/interface.py`); no note, no instrument, no synthesised tone. A row of pitched notes on a website
reads as a toy or a ringtone. The seventeen roles:

| role | design | made of |
|---|---|---|
| tap | `keyboard/mechanical` | the fullest key of a mechanical keyboard |
| hover | `hover/fingertip` | a fingertip touching a wooden table very lightly, no click (the quietest of the set) |
| toggle_on, toggle_off | `toggle/key_down`, `toggle/key_up` | the tap's key going down (bigger, darker) and coming back up (smaller, brighter): the two halves of one keystroke |
| type | `type/mechanical` | the tap's mechanical key pressed more softly |
| next, back | `next/page`, `back/turn` | a page flicked forward; the same turn darker and played the other way |
| open, close | `open/drawer`, `close/drawer` | a wooden drawer pulled out; another pushed shut, the end of its slide and the knock |
| send | `send/swoosh` | a thin stick swung fast through the air |
| receive | `receive/cork` | a small cork pulled out of a bottle, a short hollow plop |
| notify | `notify/knock` | two light knuckle knocks on hollow wood |
| success | `success/shutter` | a camera shutter fired, a quick two-part mechanical click |
| complete | `complete/shutter` | a bigger camera's shutter, a fuller and slower clack |
| warning | `warning/knock` | two firm, dull knocks on a door |
| error | `error/knock` | two heavy knocks on a door that fall: lower than the warning's, the second lower than the first |
| delete | `delete/snip` | one cut of a pair of scissors, the blades sliding shut and their snip |

A pair is two actions of one thing, so a listener tells them apart without seeing the name; a signal of two strikes
keeps them 90 to 180 ms apart, the second of a positive one brighter, of a negative one lower. Each design plays
fixed recordings of the learned store, so a set always sounds like the same objects. One other candidate per
rebuilt role takes it by name in a spec: `toggle/shutter_open` and `toggle/shutter_shut` (a film camera's shutter
opening and closing), `type/light`, `back/leaf` (a heavier page falling back), `close/door`, `complete/chop` (one
chop of a knife on a board), `error/chop`, `delete/sweep` (a stroke of a broom), `notify/pop`, `success/click`,
`toggle/rocker_on` and `toggle/rocker_off`. A request for one interface
job in words ("a success sound", "a notification", "like button tapped") gets the same design as the set, and a thing
of the world named in words ("a door closing", "a camera shutter") gets the recording of that thing, not the
interface sound made of it (`door/shut`, `camera/shutter`, not `close/door`, `success/shutter`). A like ("like
button tapped") gets `like/pop`, a small bright pop of the lips over a fingertip; a figure on an
instrument answers only when the words ask for music (an instrument, a chime, a jingle, a melody) or for chip sound,
or when a theme is named: a set with a theme plays the tonal figures (`success/rise`, `notify/pair` ...) in that
theme's voice, key and room.

**Several formats.** `--format webm,mp3` (a list or a comma string in a spec, in the MCP tool) writes every sound
once per format; `manifest.json` names each sound's `file` (the first format) and `files` (all), and the set's
`formats`. A sprite named without an extension, or with one of the set's formats, is written in each format, and
its map's `src` lists them in that order: a browser or howler.js plays the first it can. A loop is the exception:
in place of mp3 or m4a it is written once as flac (below).

**The ready-made sets** (`earshot.sounds.sets`; file name -> design):

| set | theme | sounds |
|---|---|---|
| `ui` | none | 17: tap, hover, toggle_on, toggle_off, type, back, next, open, close, send, receive, notify, success, complete, warning, error, delete (things a hand does, from recordings; with a theme: tonal figures in its voice) |
| `game` | tine | 22: menu_move, menu_select, menu_back, coin, pickup, gem, heal, power_up, power_down, level_up, checkpoint, secret, win, lose, jump, land, step, hit, hurt, explosion, splash, door |
| `arcade` | retro | 16 chip sounds: menu_move, menu_select, menu_back, coin, pickup, jump, laser, explosion, hurt, power_up, power_down, one_up, level_up, win, lose, text |
| `film` | none | 17: a product film's moves, arrivals, hits, builds and a logo ([Film and motion](#film-and-motion)) |

`earshot.sounds.sets.SETS` has the design of every name. The `game` set's jump, land, step, hit, hurt, explosion,
splash and door are still invented designs, which sound less real than the learned ones; for a game that wants them realistic, name the
learned ones in a spec (`step/floorboard`, `person/fall`, `punch/body`, `explosion/near`, `splash/plunge`,
`door/shut`).

**Levels in a set.** Every one-shot is made as a single `make` makes it (true peak -1 dBFS, a short one raised as
under Level above) and then only turned down: by its loudest 50 ms none lies more than 2.5 dB over the quietest
(one quieter than -18 LUFS does not count), and a ready-made set's `quiet` names (`hover` in `ui`) sit 0.5 dB
under the quietest (`make.balance`). Gain only: nothing is folded or limited again, so a set's sounds are as full
as the same takes made alone. Loops, beds, pads and music, and every sound when the spec or `--lufs` gives a
loudness, are levelled by momentary-max loudness: the loudest allowed is `lufs` (default -16), and each sound sits
under it by its kind:

| kind | LU under the top | kind | LU under the top |
|---|---|---|---|
| boom, stinger | 0 | chime, rise, fall, zap, thud | -3 |
| hit | -1 | note, whoosh, voice | -5 |
| figure | -2 | object | -6 |
| pop | -7 | noise, music | -8 |
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
| `sounds[]` | each: `name` (the file name, required, unique) and `want` (words) and / or `sound` (an id), plus `pitch`, `dur`, `voice`, `material`, `seed`, `n`, `axes` ({axis: -1 .. 1}), `lufs` (this sound's own level), `events` (as `--events`), `loop` (false: an ambience once, as `--no-loop`) |

```
$ earshot make --spec spec.json -o myset
20 sounds in myset (webm, mp3), manifest.json
tap.{webm,mp3}: tap/tone (0.22 s, peak at 0.03 s, D6, marimba, -16 LUFS, clean) [named]
...
type.{webm,mp3}: tap/tone (0.23 s, peak at 0.04 s, D6, marimba, -16 LUFS, clean) [named]
  same design as tap: ask for a different one with "sound" (the nearest: tap/glass)
...
open.{webm,mp3}: open/lift (2.85 s, peak at 0.14 s, D4, marimba, -14 LUFS, clean) [named]
...
success.{webm,mp3}: success/rise (0.50 s, peak at 0.21 s, D4, marimba, -14 LUFS, clean) [success, chime]
coin_1.{webm,mp3}: coin/bright (2.02 s, peak at 0.09 s, D6, marimba, -14 LUFS, clean) [coin pickup]
coin_2.{webm,mp3}: coin/bright (1.80 s, peak at 0.09 s, D6, marimba, -14 LUFS, clean) [coin pickup]
rain.{webm,flac}: rain/window (12.00 s, peak at 8.98 s, -30 LUFS, seamless loop, clean) [named]
  a loop: flac instead of mp3, which leaves a gap at the seam
20 sounds in myset\sprite.webm + sprite.mp3 (38.6 s, 333 kB, 593 kB), map sprite.json
```

With a theme, the tonal `ui` set plays `tap/tone` for both tap and type, and the answer flags it: give `type` its own
`"sound"` in the spec when the two must differ.

With a theme named, `ui` is its tonal figures: the wood theme plays them on a marimba in D. A
loop in a set with several formats is written as flac in place of mp3 or m4a, once, and says so under its line; its
manifest `files` list what was written. The sprite still holds every sound in every sprite format: the loop plays
without a seam from the sprite's webm, and may not from its mp3.

Without `set`, only the listed sounds are made (a landing page's or a scene's own handful: the recipes below). An
entry's `events` and `loop` are the single call's `--events` and `--no-loop`: `{"name": "steps", "want":
"footsteps on wet pavement", "events": 4}` is played as `make("footsteps on wet pavement", events=4)` plays it:
`step/concrete (2.55 s, peak at 2.20 s, -18 LUFS, clean) [footsteps]`, `played: 4 events as a walk`, and `no
footsteps on wet pavement: step/concrete is the nearest surface`. `"loop": false` makes a bed once, with a fade in
and out. A count, a length and "no loop" said in the `want` do the same as in a single `make`.

Re-running a spec into the same folder rewrites its files and `manifest.json`; it does not delete the files of names
removed from the spec. An unknown field is an error that lists the fields. A design id of a set that this catalogue
does not have is left out and named on the last line.

## sounds

```
earshot sounds [--search WORDS] [--domain ui|game|motion|ambience|music] [--kind KIND] [--family FAMILY] [--json]
```

```
$ earshot sounds
922 sounds in 384 families
success/rise                0.73 s  three quick rising notes, a short soft confirmation
success/fifth               0.69 s  two notes a fifth apart rising, an open clear signal
...

$ earshot sounds --search notification
20 sounds for 'notification'
notify/knock                0.55 s  two light knuckle knocks on hollow wood
notify/pop                  0.55 s  two soft pops of the lips, the second a little higher
notify/pair                 0.92 s  two notes falling by a major third, a gentle two-note call
notify/motif                1.06 s  three notes: up a fifth, then settling on the third, a small melodic call
...

$ earshot sounds --family step
17 sounds in 1 families
step/concrete               0.28 s  a hard shoe stepping on concrete
step/lawn                   0.67 s  a footstep on a grass lawn
step/pebbles                1.20 s  footsteps on gravel
step/heels                  0.47 s  a high-heeled shoe stepping on a hard floor
step/powder                 0.37 s  a footstep in fresh snow
step/tile                   0.51 s  hard shoes stepping on a tiled floor
step/twigs                  0.61 s  a footstep on dry twigs in a forest
step/floorboard             0.46 s  a footstep on an old wooden floor
step/mud                    0.66 s  a footstep in thick wet mud
step/puddle                 0.78 s  a footstep splashing in shallow water
step/leaves                 0.76 s  a footstep in dry fallen leaves
step/grass                  0.25 s  a soft crunchy footstep in dry grass
step/wood                   0.25 s  a footstep on a wooden floor, heel then toe
step/stone                  0.25 s  a hard footstep on stone, a sharp heel click
step/gravel                 0.25 s  a crunchy footstep on gravel
step/snow                   0.25 s  a crunchy squeaky footstep in snow
step/metal                  0.25 s  a heavy boot step on a metal floor
```

One line per design: id, usual length, the first of its descriptions, `(loop)` for a loop. The first eleven steps
are learned (`--json` shows the tag `real`); the last six are invented. `--search` ranks exactly as `make` would (top
20). `--json` adds every field of a design: kind, all descriptions (`says`), jobs (`uses`), tags, domains, default
pitch, the axes it reads, voice, material, loop, prior. The MCP tool without a filter answers with the families only.
Kinds: tap, click, tick, pop, note, chime, figure, hit, thud, whoosh, rise, fall, zap, boom, noise, texture, bed,
pad, stinger, voice, object, music.

## pack

```
earshot pack FILES... | FOLDER -o sprite.EXT [--gap 0.25] [--mono]
```

```
$ earshot make --set arcade -o assets/arcade --format ogg --sprite sprite.ogg
16 sounds in assets/arcade (ogg), manifest.json
...
16 sounds in assets/arcade\sprite.ogg (14.5 s, 114 kB), map sprite.json

$ earshot pack assets/arcade -o assets/arcade_all.webm
16 sounds in assets/arcade_all.webm (14.5 s, 152 kB), map arcade_all.json
```

`pack` lays sounds end to end in one file with `gap` seconds of silence between them (lossy codecs smear a little
in time; without a gap the tail of one sound leaks into the next) and writes the map next to it as
`<name>.json` in the format howler.js reads:

```json
{"src": ["arcade_all.webm"], "sprite": {"menu_move": [0, 192], "menu_select": [442, 264], "menu_back": [955, 243], ...}}
```

Each entry is `[offset_ms, duration_ms]`, and `[offset_ms, duration_ms, true]` for a loop; loops get four times the
gap. Given a folder with a `manifest.json` (a set made by `make`), it packs exactly the manifest's sounds in the
manifest's order and leaves out an earlier sprite (any file a sprite map in the folder names), as above: 16 sounds,
not the earlier `sprite.ogg` as a seventeenth. A folder without a manifest packs every audio file in it by name,
again leaving out sprites. `make --set ... --sprite` packs what it rendered, not the lossy files decoded again. In
Python: `from earshot.pack import pack; pack('assets/arcade', 'assets/arcade_all.webm')`, or a list of outputs
(`['sprite.webm', 'sprite.mp3']`) for the same sprite in several formats.

## When earshot has no such sound: build it

The catalogue cannot hold everything. For a thing no design is, `make` answers with the nearest design and says so
(`earshot makes no fire alarm; timer/done is only the nearest sound it has`), and adds `build it from parts: earshot
recipe --example` and `or let earshot draft, hear and tune it: earshot build "a fire alarm"`. A recipe is a sound
built from the catalogue's own parts, written as a small JSON form. Whoever asks designs it, or `earshot build`
drafts it from parts named in words and tunes it by listening ([BUILD.md](BUILD.md)); earshot renders it and checks
it like every other sound. Rendering a recipe loads no model.

```
earshot recipe --example                        # the form in a few lines and three recipes to start from
earshot recipe --list                           # the cookbook: the recipes that ship with earshot
earshot recipe dragon_roar                      # one of them, as JSON, to change
earshot recipe --check roar.json                # check a recipe without rendering it
earshot make --recipe roar.json -o roar.wav     # render it; --recipe - reads it from standard input
earshot make "a dragon roaring" --recipe roar.json -o roar.wav --listen     # and say how well it matches the words
```

```python
from earshot.make import make
make(recipe={'name': 'dragon_roar', 'says': 'a dragon roaring', 'space': 'hall', 'layers': [
    {'sound': 'lion/roar', 'pitch': -5},
    {'sound': 'horse/neigh', 'pitch': -12, 'at': 0.1, 'gain': -9, 'lowpass': 3500},
    {'sound': 'fire/whoosh', 'pitch': -5, 'at': 0.5, 'gain': -10},
    {'source': 'noise', 'color': 'brown', 'dur': 3.5, 'lowpass': [70, 150], 'gain': -18, 'swell': 0.6,
     'decay': 1.5}]}, out='roar.wav')
```

The method: say what physically makes the sound (the source, what it hits or moves through, its size, how far away
it is); find two to four real parts with `earshot sounds --search <word>`; write the layers; render; read the result
line and, with the listening extra, how well the sound matches the words; change one thing and render again. Real
recordings first: the designs learned from recordings are the best material. Noise is for air and rumble only, the
sine for weight under a hit; a tone that sounds like a synthesiser belongs only in a thing that is one.

### The form

The whole:

| field | what it is |
|---|---|
| `name` | the sound's name: lower-case letters, digits and `_`; it names the file when `-o` is not given |
| `says` | what the sound is, in words; `--listen` matches the sound against it |
| `layers` | 1 to 12 layers |
| `dur` | seconds: the file is cut there with a fade (a length the layers do not reach is said in a note) |
| `space` | `dry` (default), `booth`, `room`, `studio`, `hall`, `plate`, `cathedral` |
| `kind` | `event` (default), or `ambience`: levelled like a bed and faded in and out |
| `things`, `built`, `sketch` | cookbook fields: the requests the recipe answers, one sentence on what it is built from, and `true` for a recipe that is only a start (below) |

A layer is a catalogue design (`"sound": "door/slam"`), a plain source (`"source": "noise"` or `"sine"`) or a
cookbook recipe (`"recipe": "thunder_far"`, one level deep: a storm can hold a built thunder):

| field | what it does |
|---|---|
| `seed` | another take of the design; of a learned design, its next recording |
| `size`, `bright`, `hard`, `energy`, `ring`, `motion`, `width` | the design's own axes, -1 to 1, where it has them; one it lacks is said in a note |
| `dur` | the length asked of the design itself: an ambience any length, a whoosh, a recording within 0.6 to 1.8 times its own; of a source, its length (required) |
| `pitch` | semitones, -24 to 12, as a tape does it: lower is also slower and bigger |
| `glide` | semitones, -24 to 12: where the pitch has moved to by the end of the take, from `pitch` (or 0) at its start, straight in semitones, as a tape that slows down or speeds up: `{"sound": "vacuum/cleaner", "cut": [0.8, 2.6], "glide": -14, "decay": 1.2}` is a motor winding down |
| `reverse` | played backwards: a decay becomes a swell that ends on its strike |
| `cut` | `[from, to]` in seconds: the part of the take that is used |
| `halt` | seconds, 0.05 to 10: the take runs down to a stop, as a motor does when its power is cut |
| `lowpass`, `highpass` | Hz, 20 to 20000; two numbers `[from, to]` move the cutoff over the layer |
| `pan` | -1 (left) to 1 (right); two numbers `[from, to]` move the layer across |
| `stretch` | a factor, 0.5 to 2: the take longer or shorter at the same pitch (a creak drawn long, a growl held); its first 30 ms after the attack stay as they are, so a hit is not smeared. Further than that real takes (creaks, a moo, a snore) ring like a comb, so the form stops there |
| `gain` | dB, -40 to 12, against the other layers: every layer is first brought to the same peak |
| `at` | seconds: where the layer starts |
| `fade_in`, `fade_out` | seconds: fades at its ends; when not said, an ambience gets 0.1 and 0.3 s and a source 0.01 and 0.02 s |
| `swell`, `decay` | seconds: a slow rise from 30 dB down at its start, a slow fall to nothing at its end |
| `events` | `{"count", "gap", "faster", "softer", "loose"}`: the layer 2 to 64 times, each a new take, `gap` seconds apart; each gap `faster` times the one before (under 1: faster and faster); each event `softer` dB against the one before; `loose` 0 to 1 moves each up to half its gap early or late |
| `scatter` | `{"rate", "rate_to", "dur", "pitch", "gain", "pan", "grain"}`: the layer many times at random moments, `rate` events a second (0.5 to 400) moving to `rate_to` by the end, over `dur` seconds; each event up to `pitch` semitones (0 to 12) higher or lower, up to `gain` dB (0 to 40) softer, up to `pan` (0 to 1) to either side; with `grain` (0.005 to 2 s) each event is a random piece of that length of the take. Every event another take where the design has several recordings. Rain from drops, applause from claps, hail, gravel, a crowd, a swarm |
| `level` | `[[time, dB], ...]`: the layer's loudness over time, from its start, straight in dB between the points (-60 to 12) |
| `bend` | `[[time, semitones], ...]`: the layer's pitch over the heard time as a tape does it (-24 to 12): a moo bent into a whale's song |
| `pass` | `{"at", "speed"}`: the layer passes the listener, closest at `at` seconds, at `speed` m/s (1 to 100): its pitch falls through that moment (the Doppler effect), it grows loud and bright as it nears and dull and soft as it goes, and it crosses from one side to the other |
| `spin` | `{"rate", "depth", "shape"}`: its loudness pulses `rate` times a second (0.5 to 60, or `[from, to]`) by `depth` (0 to 1, default 0.8), `"smooth"` or `"chop"`: rotor blades, wings, a fan, a motor's chug, purring |
| `follow` | `{"layer", "amount", "smooth"}`: the layer's loudness follows another layer's, named by its index from 0 (`"follow": 0` is the short form). The other layer's level is measured over `smooth` seconds (0.005 to 2, default 0.05) as a share of its loudest moment; `amount` 1 (the default) makes this layer exactly that loud, 0.5 half of the way, and an amount under 0 makes it give way where the other is loud (-1: silent at the other's loudest moment) |
| `color` | of noise: `white`, `pink` (default) or `brown` |
| `hz`, `to` | of a sine: 20 to 150 Hz, gliding to `to` |

Every mistake is answered in one sentence that names the field and what it takes: `layers[0].sound 'lion/raor' is
not in the catalogue; the nearest ids: ...`, `layers[1].pitch is a number of semitones from -24 to 12, not 40`,
`unknown field 'loud' in layers[0]: a layer takes sound, source, seed, ...`.

The result has the fields of every `make` result (`summary`, `files`, `takes` with `checks`, `dur_s`, `lufs_max`,
`peak_s`, `onset_s`) and `source: "recipe"`, `sound: "recipe/<name>"` and `parts` (what it is built from). `--n 3`
writes three takes: the seeds give other takes of every layer and new noise; one seed always gives the same
samples. A recipe is rendered as written: the request's controls (`--pitch`, the axes, `--theme`) do not apply to
it, `--dur` cuts it. In a set, a recipe is one of its sounds: `{"name": "roar", "recipe": {...}}` (or the name of a
cookbook recipe), with `seed`, `lufs` and `dur`; it is levelled with the set.

`follow` ties one layer to another. Twigs that snap as the gusts come, in the wind's own rise and fall:

```json
{"name": "twigs_in_the_gusts", "says": "wind gusting through trees, dry twigs snapping as the gusts come",
 "kind": "ambience", "dur": 8.0, "layers": [
  {"sound": "wind/trees", "dur": 8.5, "swell": 0.5},
  {"sound": "twigs/snap", "gain": -14, "at": 0.3, "events": {"count": 24, "gap": 0.33, "loose": 0.9},
   "follow": {"layer": 0, "amount": 0.9, "smooth": 0.4}}]}
```

The same field with an amount under 0 makes a bed give way to a hit (rain that ducks under a thunder clap:
`"follow": {"layer": 1, "amount": -0.7, "smooth": 0.25}` on the rain), and with a short `smooth` a rattle comes and
goes with a motor.

Recordings are the material and the recipe gives them a structure. `scatter` makes many of one: a drop on metal,
scattered 90 times a second over eight seconds, each a little higher or lower, softer and to a side, is rain on a tin
roof; one clap is applause; with `grain` a crackle becomes a fine sizzle. Eight takes of the design are rendered once
and placed as copies, so 200 events a second over ten seconds render in about two seconds. `level`, `bend`, `pass`
and `spin` work on the layer as a whole once its events are laid, with times from the layer's start: a moo bent
an octave down and back is a whale's song, a running motor that `pass`es at 22 m/s is a car going by (its pitch falls
as it passes, it grows loud and bright and crosses from one side to the other), wind chopped 11 times a second is a
rotor. `stretch` draws a take longer at its own pitch, its attack kept:

```json
{"name": "tin_roof_rain", "says": "rain on a tin roof", "kind": "ambience", "dur": 8, "layers": [
  {"sound": "tick/metal", "scatter": {"rate": 90, "dur": 8.5, "pitch": 5, "gain": 14, "pan": 0.9}},
  {"sound": "water/drop", "scatter": {"rate": 40, "dur": 8.5, "pitch": 4, "pan": 0.9}, "gain": -5}]}
```

```json
{"name": "car_passing", "says": "a car passing by", "dur": 6, "layers": [
  {"sound": "tractor/idle", "pitch": 4, "cut": [0, 6.5], "pass": {"at": 3.0, "speed": 22}}]}
```

A layer may also be a cookbook recipe (`{"recipe": "thunder_far", "at": 2}` in a storm), one level deep.

One sound as the body of another ("rain on a tin can": the hits of one thing ringing through another) is not in the
form. It was built and measured: plain rain driven through the partials and colour of a struck design (a knock on a
table, a dropped tin can) comes no nearer the real recordings of rain on wood and on a tin roof for the listening
model than the plain rain does, and a metal body moves it no nearer rain on tin. What is heard of rain on a thing is
the colour of the whole, and a knock's colour is not that colour. Where the catalogue has the real surface
(`rain/wood`, `rain/tin_roof`, `rain/window`, `rain/umbrella`, `step/floorboard`, `step/tile`), use that design.

### Ten worked recipes

Each is in the cookbook (`earshot recipe <name>` prints it whole). The last column is the line `earshot make
--recipe <name> -o <name>.wav` printed for it; `clean` means the audit found no click, no cut tail, no silence
inside and no clipping, not that the sound was judged by ear.

| the thing | the thought | the layers | the result |
|---|---|---|---|
| a dragon's roar, `dragon_roar` (a sketch) | a huge animal's voice with fire in it, in a cave | `lion/roar` -5 semitones as the voice, `horse/neigh` -12 inside it, `fire/whoosh` -5 as the breath, brown noise under 70 to 150 Hz, space hall | 3.69 s, peak at 0.19 s, -14 LUFS, clean |
| a car crash, `car_crash` | the skid, one impact of metal and weight, the debris after | `tyres/screech` cut to 1.6 s, then at 1.5 s `car_door/slam` -7, `metal/drop` -5, `glass/break`, `can/drop` -3 | 3.11 s, peak at 1.58 s, -14 LUFS, clean |
| a monster's growl, `monster_growl` | a big animal's voice far down, a wet throat in it | `lion/roar` -9 under 3000 Hz, `snore/person` -10 cut to 2.8 s, `pig/oink` -15, space room | 3.37 s, peak at 0.24 s, -14 LUFS, clean |
| a spaceship door, `spaceship_door` | air released, a motor that runs and stops, a panel, a stop | `spray_can/spray` over 800 Hz, `drill/electric` -9 with `halt` 0.7, `door/slide`, `lock/latch` -5, space studio | 1.67 s, peak at 0.46 s, -14 LUFS, clean |
| a robot powering down, `robot_power_down` | a switch, motors that lose speed, the body settling | `switch/industrial`, `drill/electric` -5 with `halt` 1.6, `vacuum/cleaner` -7 with `halt` 1.8, `spray_can/spray`, `dumbbell/place` -3 | 2.91 s, peak at 0.25 s, -14 LUFS, clean |
| a sword on a shield, `sword_on_shield` | the clang, the wood, the boss and the weight at one moment, the swing before | `sword/swing`, then at 0.2 s `clash/sword`, `knock/hollow` -7, `saucepan/hit` -7 under 3000 Hz, `punch/body` | 0.90 s, peak at 0.27 s, -14 LUFS, clean |
| a campfire in the wind, `campfire_wind` | two real beds and what happens in them | `fire/camp`, `wind/trees` with a swell, pink noise under 500 to 1400 Hz as a gust, `twigs/snap` five times with `loose` 0.9, kind ambience | 7.65 s, peak at 0.54 s, -22 LUFS, clean |
| a bowling strike, `bowling_strike` | a ball rolling nearer, one hit, pins that scatter | brown noise under 180 to 420 Hz with a swell, `land/wood` -5, `dice/throw` -10, `knock/hollow` -3 six times, each softer and later, space studio | 3.07 s, peak at 1.25 s, -16 LUFS, clean |
| a woodpecker, `woodpecker` | a hard beak on a hollow trunk, very fast, twice, in a forest | `knock/hollow` +4 twelve times 0.045 s apart, again ten times at 1.4 s, `forest/morning` at -20 | 3.01 s, peak at 0.05 s, -14 LUFS, clean |
| a giant's footsteps, `giant_footsteps` | weight, slowness, the ground answering | `step/concrete` -12, `heavy_drop/floor` -7, `rumble/quake` -3 cut to 0.9 s, each four times, 1.1 s apart, space hall | 4.68 s, peak at 3.34 s, -14 LUFS, clean |

### The cookbook

`earshot/data/recipes` holds 99 recipes for things people ask for and no design is. 59 of them answer a request
that names their thing (`"a car crash"`, `"a werewolf howl"`, `"a welding torch"`, `"a woodpecker"`, `"a cannon
firing"`, `"marbles pouring"`, `"rocks tumbling down"`, `"a tornado"`, and for what no recording can hold, `"a fireball spell"`, `"an energy shield going up"`, `"a plasma
shot"`, `"a robot arm moving"`, `"a zombie horde"`); the answer says `built from parts by the cookbook recipe
car_crash` and names the nearest designs, which `--sound <id>` still gives. `earshot recipe --list` prints each
recipe with its parts. Fewer layers often sound more real: a champagne bottle is a cork and a pour, nothing more.

The catalogue comes first. When the design the words found is the thing itself (its id or its description holds
every word of the thing), the design answers and the recipe does not: `"a rocket launch"` is the recording
`rocket/launch`, `"footsteps in mud"` the walk `step/mud`. A recording only listed for a thing it is not stays its
nearest, and the cookbook's recipe answers if it is not a sketch. When a recording of a thing joins the catalogue,
its recipe leaves the cookbook (a chainsaw, a hiccup, a fax machine and a dishwasher are recordings now).

A recipe answers on its own only when earshot's listening model hears it as its thing (a match of z 3.0 or more)
and clearly better than the catalogue's nearest design (by 1.0 or more), and the audit finds it clean. The other 41
carry `"sketch": true` (a dragon roar, a wolf pack, a helicopter landing, a heartbeat getting faster, a time travel
whoosh, a giant stone door, a cryo pod opening and more). The dragon is one: built on a
lion's roar a fourth lower, it matches "a dragon roar" no better than the lion's roar alone (`lion/roar`, which
answers). A sketch is never given as the answer: the
catalogue answers as before, and one more line names the sketch as a start (`a start from parts: earshot make
--recipe heartbeat_faster`).

What the request asks of a cookbook thing is followed where the form can do it, and each is said in the answer:

| asked | done | the line |
|---|---|---|
| a length (`"a werewolf howl, 3 seconds"`) | the file is cut there with a fade; a length the layers do not reach is said | (a note when it is not kept) |
| far away, in the distance | the whole in a hall, every layer under 3000 Hz, 6 dB under the usual level | `as asked: far away (in a hall, duller and 6 dB softer)` |
| up close | no room around it | `as asked: up close (no room around it)` |
| soft, quiet | 6 dB under the usual level | `as asked: softer (6 dB under the usual level)` |
| loud, louder | not followed: a cookbook sound is made at the loudest level of its kind | `not followed: louder; ...` |
| three times, twice, `--events 3` | every layer that many times, one whole after the other | `as asked: 2 times, 3.41 s apart` (a car crash twice) |
| three times, of a recipe that is a row already | not followed | `not followed: 3 times; this recipe is a row of events already, played once` |

`earshot make "a werewolf howl far away"` printed:

```
werewolf_howl.wav: recipe/werewolf_howl (7.71 s, peak at 1.49 s, -20 LUFS, clean) [recipe]
built from: lion/roar (-4 semitones, under 3000 Hz); wolf/howl (-4 semitones, under 3000 Hz); wolf/howl (-8 semitones, under 2500 Hz)
built from parts by the cookbook recipe werewolf_howl: no design of the catalogue is this sound; `earshot recipe werewolf_howl` prints the recipe to change; the nearest designs: wind/window, wind/mountain, wind/open (sound=<id>)
as asked: far away (in a hall, duller and 6 dB softer)
```

## The request language

`earshot.brief.read` turns words into a brief without any model, from the tables in `earshot/sounds/lexicon.py`.
How the words were read (`earshot.brief.read`, run on each request; `brief.rank` gave the top design):

| request | read as | top design |
|---|---|---|
| `a soft wooden success chime for a finance app, 400 ms, in D` | concepts success, bell; hard -0.5, energy -0.3; material wood; dur 0.4; key D | success/rise |
| `an error sound but not too loud` | error; energy -0.42 | error/knock |
| `the opposite of a success sound` | error | error/knock |
| `a much bigger, darker whoosh` | whoosh; size +0.84, bright -0.6 | whoosh/deep |
| `a kalimba note in E` | note, voice kalimba; key E | note/kalimba |
| `a two-note success chime in D minor` | success, bell; key D, scale minor | success/rise |
| `three variations of a footstep on gravel` | footstep; n 3 | step/pebbles (learned) |
| `8-bit coin pickup` | coin; material retro (chip designs only) | coin/chip |
| `rain on a tent, no loop` | rain | rain/tent |
| `seamless forest ambience, 30 seconds` | forest, ambience; loop on; dur 30 | forest/morning |
| `toggle dark mode` | toggle; bright -0.6 | toggle/key_down |
| `a wooden drawer closing` | close; material wood | drawer/wood (learned) |
| `a sword being drawn` | swing | sword/draw (learned) |
| `a kick drum` | kick drum | kick/soft |

What is recognised:

- **The thing, the job or the sound.** A thing ("a light switch", "a cork", "a drawer"), a job ("success", "coin
  pickup", "logo reveal", "level complete") or a sound ("chime", "whoosh", "rain"), often several, mapped to
  canonical words, the longest phrase first. The first one named is the head of the request. What follows "with"
  describes a part of it, what follows "for", "when", "during", "each time" ... once the head is named is where it
  plays: both count less ("a tiny bright tick each time a step number changes in the corner" is `tick/soft`; the
  step is no footstep). A word of a moment is read as what the thing does: a card or a panel that settles lands
  softly ("a soft low thump when the pricing card settles": `thump/soft`), while energy or dust that settles
  winds down; "careful" or "cautious" before a thing say how it is done ("one careful step on creaky wooden
  planks": `step/floorboard`) and alone are a warning. A multi-word job of a design ("page turn", "saw lead")
  matches only as the whole phrase; its single words count no more than a tag. A learned design answers only when
  its thing is named ([Learned sounds](#learned-sounds)).
- **Feelings as axes.** Adjectives become offsets on [the eight axes](#the-eight-axes) ("soft" lowers hard and
  energy, "darker" lowers bright, "bigger" raises size, "longer" raises ring, "drier" lowers space), and also
  hints that break ties inside a family ("crisp" finds `whoosh_fast/crisp`). Intensifiers scale them ("a little",
  "much", "very").
- **Negation and comparison.** "not harsh", "not too loud" flip the adjective; "not an error" turns a named job into
  one to avoid; "the opposite of X" and "X but the other way" turn X into its opposite; "like a ..." is a
  comparison; what follows "than" is a reference and is skipped.
- **Material and voice.** An instrument named as the thing itself ("a kalimba note", "marimba hit") asks for a
  note of that voice; one named as the player ("a success chime on a kalimba", "a notification on a harp", "a
  success sound with pizzicato strings", "an error sound on a soft piano") sets the voice of a figure, and a
  voice asked for by name is always played. Material words ("wooden", "glass", "metal", "soft", "digital",
  "8-bit") steer the ranking toward designs of that material, and on a tonal design with no voice asked for they
  pick the voice made of it: wood a marimba, glass a glockenspiel, metal a vibraphone, soft a felt piano, digital a
  glass pluck, retro a square lead (`registry.VOICE_OF`; this wins over the theme's voice). A voice or material
  that chose the design is not reported as unused.
- **Numbers.** Lengths ("400 ms", "2 seconds", "half a second") set how long a pad is held, how long a loop or a
  whoosh is, and how long a figure of notes sounds: shorter than it rings, it keeps its spacing and the hand damps
  it so that it ends at that length (a figure that does not fit is played faster, down to 70 ms between notes;
  shorter still, the result says `as short as this sound can be: 0.26 s`); a longer length changes nothing, a
  struck note rings as it rings (`rings out in 1.85 s, shorter than the 3 s asked for`). Any other length a design
  cannot keep (a struck sound has no length) is said in plain words (`0.43 s, not the 2 s asked: ...`). "Under 300 ms" is a
  ceiling: it prefers shorter designs and shortens only a design that is longer, as a length. A pitch ("C6", "440
  Hz", "middle C"), a key ("in D", "in the key of Bb minor"), how many takes ("three variations", "5 versions", "a
  few takes"; at most 12) and how many notes ("a two-note chime", "three rising notes", "twice"). A key moves a
  tonal design's root to the nearest note of that key; a minor key (in the request or the theme) turns a figure's
  major thirds, sixths and sevenths into minor ones.
- **One event or the action.** For a thing a person does several times in a row, a plural, an action word, a count
  or a length ("footsteps", "knocking", "three knocks", "5 seconds of typing") asks for the action, a bare singular
  or "a single" for one event ([Performances](#performances)). A counted action of any one-shot ("four knocks on
  the table", "three claps", "a countdown 3 2 1") is one file of that many events; a count of separate things ("five
  coin pickups", "3 variations") is that many files. The action form of a row word ("clapping", "clicking",
  "beeping", "tapping") is a row at the pattern's pace. A length asked with an action ("three knocks, 2 seconds",
  "clapping, 4 seconds") is the file's length, the events spread over it at a pace a hand keeps.
- **Interface and motion jobs.** A request for an interface job ("a success sound", "a notification", "toggle dark
  mode") gets the interface design of that job, a normal sound (`lexicon.INTERFACE_JOBS`); a motion job ("a title
  hit", "a logo reveal", "a transition") gets the film group's hits, thumps, ticks and whooshes. Neither gets a
  design made of notes unless the words ask for music (an instrument, a chime, a jingle, a melody, a motif, a pad)
  or for chip sound, or the request is for a game.
- **Things earshot does not make.** A request always gets a sound. When the thing that makes it, or what the thing
  does, is a word neither the lexicon nor any design's name or job knows ("a dial-up modem", "a vault door"), the
  answer is the nearest design and says so: `also (the pick is a guess: earshot makes no dial up modem; dial/click
  is only the nearest sound it has): ...`. The nearest is chosen by
  the kind of sound the word names (a growl is a low rumble, a hiss a spray of air, a whisper a sigh:
  `lexicon.SOUND_KINDS`), else by the words of the request that are known, never passed off as the thing.
- **On, in, against, into.** What follows these words after the sound is its surface, ground, material or place,
  not a second sound: "rain on wood" is rain and no forest, "a ball bouncing on wood" the ball. The surface picks
  among the sound's own designs ("footsteps on gravel"), a surface none of them has picks the nearest one and says
  so ("rain on a tarp": `no rain on a tarp: rain/tent is the nearest surface`; `lexicon.SURFACE_KIN`).
- **German.** A request in German is read word by word through a table of about 250 sound words
  (`lexicon.GERMAN`): "Regen auf Holz" is read as "rain on wood", "Schritte auf Kies" as "footsteps on gravel".
  A request with a word the table lacks, or in another language, is still answered as a guess that says "write it
  in English".
- **Direction.** "rising", "falling", "going up" ... prefer designs whose gesture moves that way.
- **Loop or not.** "loop", "seamless", "looping", "endless" ask for a loop; "no loop", "one-shot" for the scene
  once.
- **8-bit.** "8-bit", "retro", "chiptune", "arcade" make chip sound a hard filter: only chip designs answer, and
  chip designs answer nothing else.
- **Where it plays.** Domain words ("app", "game", "film", "platformer") weigh designs made for that domain.
- Words nothing knows come back under `not understood` and count only a little:

```
$ earshot make "a squishy success chime, longer" -o sq.wav
sq.wav: success/rise (1.85 s, peak at 0.20 s, G4, marimba, -14 LUFS, clean) [success, chime]
also: success/triad, success/done, success/fifth
not understood: squishy
```

What reached the render, on real calls:

```
$ earshot make "a soft wooden success chime for a finance app, 400 ms, in D" -o wood.wav
wood.wav: success/rise (0.40 s, peak at 0.21 s, D4, marimba, -14 LUFS, clean) [success, chime]
also: success/done, success/fifth, success/triad

$ earshot make "a success chime on a kalimba, softer" -o kal.wav
kal.wav: success/rise (0.79 s, peak at 0.09 s, G4, kalimba, -14 LUFS, clean) [success, chime]
also: success/done, success/fifth, success/triad

$ earshot make "a two-note success chime in D minor" -o minor.wav
minor.wav: success/rise (2.64 s, peak at 0.22 s, D4, marimba, -14 LUFS, clean) [success, chime]
also: success/fifth, success/done, success/triad

$ earshot make "a kalimba note in E" -o kalimba.wav
kalimba.wav: note/kalimba (1.06 s, peak at 0.00 s, E5, -14 LUFS, clean) [kalimba, note]
also: pluck/kalimba, pluck/kalimba_ring, pluck/kalimba_soft
```

"Wooden" played a marimba and the key D moved the root from G4 to D4; "400 ms" is kept the way a player keeps it:
the three notes at their spacing, then the bars damped by the hand so the sound ends at 0.40 s (without a length,
the last note rings out for about 2.6 s). In D minor, `success/rise` (a major third up) rises a minor third.

**How well requests are understood.** On five development sets of requests (A, B, D, E and G) the first
pick is right for 91.8 to 96.8 % and one of the top three for 95.9 to 99.1 %. On 220 held-out requests written by
someone who never saw the catalogue (half long descriptions of a sound that never name it, half casual requests),
scored once on an earlier reader, the first pick was right for 69.1 % and one of the top three for 76.4 %.
Development numbers are optimistic by construction. Requests that name the thing or the job do far better than
descriptions of a mood or paraphrases; about one held-out request in three or four needs a second call, which is
what `also`, the guess note and `--sound ID` are for. Every set and how it was scored:
[VALIDATION.md](VALIDATION.md#understanding-requests).

## The eight axes

Every design reads the same controls, so a word and a theme mean the same thing everywhere. Each axis runs from -1
to 1, with 0 = the sound as designed; a design reads only some of them, and the rest are reported as not used.

| axis | -1 .. +1 | what it does in the engines |
|---|---|---|
| `bright` | darker .. brighter | a harder mallet and more of the high modes; a higher filter cutoff or FM index; more top in noise; a tilt and the brighter members of a learned sound |
| `hard` | softer .. harder attack | mallet stiffness, attack time; a faster and louder onset layer; a learned sound's strikes |
| `ring` | shorter, drier .. longer ring | damping (struck: a quarter to four times the ring time); note lengths in a figure; tail length |
| `size` | smaller, lighter .. bigger, heavier | lower and longer, more low layer; a larger resonator or body; a learned thing 0.7-1.4 times as big |
| `space` | no room .. more room | the shared room's tail: -1 dry, +1 8 dB more and 1.6 times longer |
| `width` | narrower .. wider | stereo spread of the notes, the microphone pair, decorrelation |
| `energy` | gentler .. more intense | how hard it is played: level and timbre together; more events in a texture; the busier moments of a learned bed |
| `motion` | steadier .. more movement | the speed of a figure (0.7 to 1.4 times); LFOs, chorus and detune in a patch; the tempo of a music loop |

## Themes

A theme is what makes a set of sounds one product: one voice, one key, one room and offsets on the axes, applied to
every design that honours them (`earshot/sounds/theme.py`). Recorded sounds have no voice: a theme sets their room
and axes only.

| preset | voice | key | space | axes | says |
|---|---|---|---|---|---|
| `soft` | marimba (real) | C major | room | bright -0.2, hard -0.4 | a real marimba played softly: warm, rounded, mid-register: the default voice of the figures |
| `glass` | glockenspiel | D major | studio | bright +0.2, ring +0.1 | clear, bright, bell-like |
| `wood` | marimba (real) | G pentatonic | room | bright -0.1, ring -0.2 | a real marimba: warm, dry, organic |
| `felt` | felt_piano (a real upright piano) | F major | room | bright -0.3, hard -0.3 | an intimate soft piano |
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
theme's, and a material or voice in the request wins over the theme's voice. A request for an interface or motion
job gets a figure only when its words ask for music (an instrument, a chime, a jingle, a melody, a motif) or for
chip sound, or a theme is named; otherwise it gets the normal sounds of the interface set or the film group. Without
a theme or a voice in the request, figures play the real marimba: a bell, a glockenspiel or a glass timbre only
when asked for (glass, glockenspiel and synthesised voices were judged poor by ear for interface sounds). An instrument
named in the request plays the figure: "a notification on a harp", "a success sound with pizzicato strings", "an
error sound on a soft piano" (`felt_piano`), or `--voice harp`. A few game figures keep the voice they were written
for (`coin/bright`: a glockenspiel).

## The catalogue, family by family

`earshot sounds` lists the designs and prints how many there are. Ids are `family/name`; the family is what the
sound is or does, the name its character.

| group | designs | ids | for | controls |
|---|---|---|---|---|
| learned (`sounds/learned.py`) | 309 | air and swings: `whoosh/stick`, `whoosh/rope`, `swipe/hand`, `sword/swing`, `whip/swish`, `wings/flap`, `cloth/flag`; action: `punch/body`, `person/fall`, `sword/draw`, `whip/crack`, `explosion/far`, `explosion/near`, `match/light`; steps: `step/concrete`, `step/pebbles`, `step/heels`, `step/tile`, `step/floorboard` ...; doors and furniture: `door/shut`, `door/creak`, `drawer/wood`, `lock/key`, `floor/creak`; objects: `bag/drop`, `book/shut`, `page/turn`, `cork/bottle`, `dice/throw`, `zipper/jacket`, `scissors/cut`, `stapler/desk`; glass and water: `glass/cheers`, `ice/cubes`, `water/plop`, `splash/plunge`; metal: `anvil/strike`, `pot/clang`, `chain/rattle`, `can/drop`; interface: `click/mouse`, `switch/lamp`, `switch/wall`, `switch/small`, `dial/click`, `keyboard/laptop`, `keyboard/desktop`, `keyboard/mechanical`, `keyboard/space`, `typewriter/key`, `camera/shutter`, `buzzer/deny` | foley for games, 3D, video and interfaces | size, bright, hard, ring, energy, space, width; dur within 0.6-1.8x; pitch on tonal ones |
| interface (`sounds/interface.py`) | 29 | `hover/fingertip`, `toggle/key_down`, `toggle/key_up`, `type/mechanical`, `next/page`, `back/turn`, `open/drawer`, `close/drawer`, `send/swoosh`, `receive/cork`, `notify/knock`, `success/shutter`, `complete/shutter`, `warning/knock`, `error/knock`, `delete/snip`; other candidates `toggle/shutter_open`, `toggle/shutter_shut`, `toggle/rocker_on`, `toggle/rocker_off`, `type/light`, `back/leaf`, `close/door`, `complete/chop`, `error/chop`, `delete/sweep`, `notify/pop`, `success/click`, `like/pop` | the `ui` set and interface jobs asked in words: things a hand does, from learned recordings | the axes of the recordings they play |
| interface figures (`sounds/ui.py` on `figure.py`) | 25 | `success/rise`, `error/fall`, `notify/pair`, `open/lift`, `send/lift`, `warning/pulse`, `delete/drop`, `purchase/chime` ... | confirmations, errors, warnings, notifications, navigation, when the words ask for music or a theme is named | all 8 axes, pitch (the root), voice or material, dur |
| game figures (`sounds/game.py`) | 15 | `coin/bright`, `pickup/blip`, `gem/sparkle`, `power_up/rise`, `level_up/fanfare`, `win/fanfare`, `lose/descend`, `menu/select` ... | rewards, signals, jingles, menus | all 8 axes, pitch, voice or material, dur |
| brand figures (`sounds/brand.py`) | 5 | `logo/bloom`, `logo/duo`, `logo/motif`, `startup/rise`, `shutdown/fall` | sonic logos, openings, endings | all 8 axes, pitch, voice or material, dur |
| material ticks (`sounds/ui.py`) | 3 | `tick/wood`, `tick/glass`, `tick/metal` | taps and detents that should not be a note | 7 axes (not motion), pitch |
| ambience (`sounds/nature.py`) | 43 in 22 families | learned from real textures (25): `fire/camp`, `waves/sand`, `waves/pebbles`, `waves/rocks`, `night/crickets`, `stream/brook`, `room_tone/quiet`, `city/street`, `city/distant`, `cafe/busy`, `underwater/deep`, `rain/street`, `rain/wood`, `rain/tin_roof`, `rain/umbrella`, `wind/trees`; from rules (18): `rain/window`, `rain/leaves`, `rain/roof`, `rain/tent`, `rain/water`, `wind/open`, `wind/window`, `wind/mountain`, `fire/small`, `fire/large`, `stream/trickle`, `stream/river`, `forest/morning`, `space_hum/ship`, `crackle/vinyl`, `sparkle/magic`, `thunder/clap`, `thunder/distant` | beds for games, pages, timers and films | learned: energy (calmer or busier moments), size, space, bright, width; rules: energy, size, space, bright, width; dur = loop length; `--no-loop` |
| drums (`sounds/drums.py`) | 40 in 21 families | from recorded strokes (33): `kick/soft`, `kick/punch`, `kick/deep`, `snare/tight`, `snare/fat`, `rim/click`, `rim/shot`, `tom/high`, `tom/mid`, `tom/floor`, `hat/closed`, `hat/open`, `hat/pedal`, `crash/bright`, `crash/dark`, `ride/ping`, `clap/hands`, `clap/group`, `shaker/tick`, `tambourine/hit`, `tambourine/shake`, `conga/open`, `conga/muted`, `bongo/high`, `bongo/low`, `cajon/bass`, `darbuka/hit`, `frame_drum/deep`, `clave/click`, `cabasa/hit`, `slapstick/crack`, `drum/roll`, `shaker/roll`; invented (7): `kick/electro`, `snare/electro`, `hat/electro`, `clap/electro`, `snare/brush`, `snap/finger`, `sticks/count` | music loops, motion design, games | energy (velocity: softer or harder strokes), size, bright, hard, ring, space, width; pitch on tuned drums |
| film and motion (`sounds/motion.py`) | see below | | product films, title sequences, interface animation | |
| action (`sounds/action.py`) | 47 in 23 families | `jump/*`, `land/*`, `step/grass` .. `step/metal` (6 surfaces), `swing/*`, `hit/punch` ..., `explosion/*`, `shot/*`, `door/*`, `break/*`, `splash/*`, `teleport/*`, `magic/*`, `countdown/*`, `laser/beam` and `rumble/quake` (loops) | games and animation; invented from physics | energy, size, bright, hard, ring, space, width; pitch for the tonal ones; dur for loops |
| objects (`sounds/object.py`) | 36 in 26 families | `coin/drop`, `glass/clink`, `paper/*`, `cards/*`, `dice/*`, `water/drop`, `clock/ticktock` (loop), `switch/*`, `keys/*`, `bell/desk`, `zipper/*`, `cork/*` ... | foley; invented from physics | the same 7 axes; dur for poured, pulled and looped ones |
| notes (`sounds/tonal.py`) | 51 | `note/<voice>`: 36 instruments and 15 synth patches | a named instrument, a product's own motifs | all 8 axes, pitch, dur |
| synthesizer (`sounds/synth.py`) | 14 | `pad/warm`, `pad/air`, `drone/dark`, `sub/swell`, `braam/hit`, `riser/tone`, `sweep/noise`, `stab/chord`, `zap/laser`, `power_up/synth` ... | finished synth sounds | all 8 axes, pitch (root), dur (held) |
| chip (`sounds/retro.py`) | 24 | `coin/chip`, `jump/chip`, `laser/chip`, `explosion/chip`, `win/chip`, `text/chip`, `engine/chip` (a loop) ... | 8-bit games; only when asked for | bright (output filter), energy (>= 0.5: a 6-bit output), pitch, dur |
| music loops (`sounds/beds.py`) | 3 | `bed/warm` (electric piano, soft drums), `bed/playful` (muted guitar), `bed/8bit` (chip; only when asked for) | a game menu, a demo, under a voice-over | pitch (the key), dur (whole bars), motion (tempo), energy, voice, bright, space, width; `--no-loop` ends on a held chord |
| bank (`sounds/bank.py`) | 209 in 38 families | `whoosh_fast/crisp`, `riser/tension`, `impact/deep`, `ui_pop/soft`, `cut_hit/punch`, `typing_key/light`, `logo_sting/*`, `pro_*` ... | motion design and film ([SFX.md](SFX.md)) | pitch and dur where the recipe has those knobs; no axes |

Where a learned and an invented design answer the same request, the learned one wins a tie. Bank sounds rank under
the designs written for the catalogue (prior 0.8, the `pro_` families 0.7), so a request that fits both gets the
newer design; members that carry only a number in the bank (`pop/3`) stay reachable through `earshot sfx`. Chip
designs answer only requests that ask for chip sound.

**Music.** earshot is a sound-effects tool and does not make music. Of twelve short music styles that were written,
three were kept after listening; the other nine are gone, and nothing more is planned. The three loops are a whole number of bars in a key and a tempo;
each seed is another progression and voicing with the same instruments.

```
$ earshot make "a warm music bed" -o bed.ogg --dur 16
bed.ogg: bed/warm (17.14 s, peak at 0.73 s, F3, electric_piano, -16 LUFS, seamless loop, clean) [music bed]
also: bed/playful, bed/creak, bed/8bit
17.14 s: a music loop is whole bars, and these are the nearest to the 16 s asked
```

**Drums.** The acoustic kit and the hand percussion are takes of real strokes: 130 recordings in the public domain
(VCSL and Karoryfer Big Rusty Drums), kept as numbers in `earshot/data/drums/` like the other learned sounds and
played softest first by velocity. The seed walks through a drum's strokes (a drummer never hits twice the same) and
`energy` picks softer or harder ones; `drum/roll` and `shaker/roll` are performances of those strokes. Invented,
because no public-domain recording covers them or because they are synthesizers by nature: `kick/electro`,
`snare/electro`, `hat/electro`, `clap/electro`, `snare/brush`, `snap/finger`, `sticks/count`
(`earshot/synth/drums.py`: a beater's contact, modes, wires and jingles as collisions). The earlier drums invented
from physics were judged poor by ear; the recorded ones have not been judged by ear yet.

```
$ earshot make "a tight snare drum hit" -o snare.wav
snare.wav: snare/tight (0.38 s, peak at 0.00 s, -18 LUFS, clean) [snare drum, hit]
also: snare/brush, snare/fat, snare/electro
```

### Film and motion

The sounds a product film, a title sequence or an interface animation is cut with, by role: air and movement (a
fast and a soft whoosh, a deep pass, a sliding panel, a flick), hits (a short hit under a title or a logo, the hit
after a stop, a long boom, a thump when something appears), builds and falls (a riser of air, a reversed swell, a
tape stop), small arrivals (a tick, pops, key presses), light (a shine of air, a glint), a glitch and short
stingers.

They follow measurements of professional motion videos ([motion-sfx-study.md](motion-sfx-study.md)): such films
are led by music and add few effects, and those are short, dry, near mono and either low or very bright. The `film`
set and a plain film request play small real actions cut to those shapes, each naming its recordings by file: a
thing appearing gets a door falling shut, deep and short (`thump/door`), a click the bright top of a mechanical key
(`tick/key`) or a whole mouse button (`click/cursor`), a title a stick swung into a heavy bag dropped (`title/bag`),
the logo a heavy bag dropped, low and short (`logo/bag`), the moment after the music stops a heavy bag on a deep thud
(`drop/sack`), an impact a car door with the bag under it (`impact/car`), a move a stick swung (`whoosh/fast`) or a
broom's stroke (`whoosh/sweep`), a card sliding in a sheet of paper (`slide/paper`), a reveal a breath of air
growing to a cut (`rise/breath`) or a slam played backwards (`rise/hit`), a glint a sword drawn, only its edge
(`shine/blade`). None of them uses an instrument note, a bell or a synthesised tone, and a motion request gets no
design made of notes unless its words ask for music (an instrument, a melody, a chord) or for a game.

55 of the 69 designs of `earshot/sounds/motion.py` are built from real material, the way a sound designer builds
them: 23 named small actions (those above and a book shut, a lamp switch, a sword swung, a cloth flapping, a match
struck, a zip and a snare roll running down), and the 32 designs below, made of takes of learned recordings (car
doors and a bag slammed and dropped, explosions, whooshes, swells of air, flames, rolls, low brass, thunder, a gong
and cymbals, a mouse click and switches, and the close-miked strokes of the drum store) reshaped by
`earshot/synth/design.py` (reverse, layer at a moment, stretch, contour, carry past the listener, halt, fade) within
the range where a take stays natural, and glued by one room. The seed is a performance: it picks the recordings and
every take draws fresh noise; `meta['member']` names the main recording.

| family | designs | made of |
|---|---|---|
| air | `whoosh/fast` | a real stick swung through the air, as recorded: close and dry, nothing layered over it |
| | `whoosh/soft` | a real whoosh that is noise and not a whistle (a heavy whoosh, a deep whoosh or a calm burst of flame, by the seed), swelling for three quarters of the length and gone within tens of ms after its peak, near mono, dry |
| | `whoosh/deep` | a real deep or heavy whoosh, a real bass drum's decay played backwards 12 dB under it |
| | `whoosh/long` | a real deep whoosh, bigger and slower, led in by a swell of real air played backwards |
| | `swoosh/in`, `swoosh/out` | a real airy whoosh up to its peak and stopped there; from just before its peak on |
| | `flick/thin` | a short real whip of air (a whip or a swung stick, by the seed) with nothing under 1.5 kHz, mono, dry: "a soft swish" |
| | `flick/paper` | a real page turned, fitted to the length |
| | `pass/air` | a real deep whoosh carried past the listener: across the stereo field, level by distance, pitch falling 1.5 semitones |
| hits | `slam/title` | 0.15 s of a real low whoosh's build swelling into a real heavy bag dropped on the low end of a real close explosion; the whoosh's decay goes on after the strike as the air of the hit; about a third of a second, close, near mono. No door, no drum |
| | `boom/short` | a real heavy bag dropped (nothing over 900 Hz) on the low end of a real close explosion, led in by 0.1 s of a low whoosh's build: a quarter second |
| | `boom/deep` | the low end of a real close explosion with the heavy bag a quarter bigger 3 dB under it, led in by a whoosh's build: a short deep hit |
| | `hit/drop` | the heavy bag on the explosion's low end with a puff of real air after the strike, 0.2 s; the silence before it is the edit |
| | `slam/door`, `boom/door` | the same hits built of a real car door and a car boot slammed shut, by name only |
| | `thump/appear` | a real soft kick drum with a muted conga 9 dB under it and a faint real mouse click, cut 50 ms after the strike |
| | `thump/soft` | a real soft kick drum and a cajon's bass stroke, 80 ms, dry |
| | `impact/suck` | real air drawn in backwards (a breath, a burst of flame or a deep whoosh) ending on the heavy bag a fifth bigger with the explosion's low end under it |
| | `fall/sub` | the long boom by name: the low end of a real deep kick drum with a real close explosion's low end under it, only falling, 2 s |
| | `boom/wide` | a real tam-tam struck hard over a real bass drum, wider than recorded |
| | `explosion/huge` | a real near explosion, a real bass drum 3 dB and a real roll of thunder 9 dB under it, in a hall |
| builds | `rise/reverse` | real air played backwards (a breath, a deep whoosh or a burst of flame), louder and brighter to a 10 ms close |
| | `rise/air` | two builds of real air played backwards, lined up at their ends, growing louder and brighter to a clean cut; no cymbal, no tone |
| | `rise/tension` | a real drum or timpani roll in crescendo, a real suspended cymbal rolled 6 dB under it |
| | `suck/in` | a real airy whoosh's tail played backwards, stopped where its peak was |
| | `stop/tape` | a real timpani or snare roll with a held low brass note under it, read at a speed that falls to a halt |
| falls | `fall/air` | the decay of a recorded swell of air, fading straight in dB |
| small | `tick/soft` | a real small click (a mouse button or a small switch) with nothing under 1.2 kHz, cut 40 ms after the strike |
| | `shine/air` | the build of real air above 2.5 kHz with a real shaker struck once on its top: light without a note |
| | `glitch/short` | slices of a real electric buzzer, an electronic switch and a mouse click on a gate that speeds up |
| stingers | `sting/reveal` | a build of real air played backwards into the low hit of `boom/short` |
| | `braam/brass` | real low brass (tubas, trombones, horns) on the root, fifth and octave over a low end of real recordings, led in by a reversed cymbal, no added room |

Invented, from class statistics of small film sounds (0.1-0.25 s, a 10-20 ms rise, mono): `slide/panel`,
`knock/soft`, `pop/soft`, `pop/bubble`, `pop/pluck`, `tock/wood`, `tock/glass`, `blip/soft`, `shine/glint`,
`shine/sweep`, `sparkle/burst`, `type/soft`, `type/run`, `sting/soft`. The learned film things answer by their own
names too (`timpani/hit`, `gong/hit`, `cymbal/orchestral`, `brass/low`, `thunder/roll`, `pass/car`,
`car_door/slam`, `explosion/far` ...), and the bank's film sounds (`whoosh_fast/crisp`) stay reachable. "An
explosion far away" is `explosion/far`, a long rolling blast; "a fire whoosh" plays a real burst of flame first.

Controls: energy (a harder, brighter strike and more of the lower layers), size (the takes 0.8-1.25 times the
design's own size: bigger is lower and longer), bright (2 dB per octave), hard (the strike; not on `slam/title`,
`boom/short`, `boom/deep` and `hit/drop`, whose body has no strike to raise), ring (on the hits that ring), space,
width, and motion on `pass/air` (how far it travels, 0.5-2.5 semitones of fall). `dur` sets a build's length (it
ends on its peak), a whoosh's length and how much of a tail is heard. A cymbal in a film request is the film's
crash cymbal (`cymbal/orchestral`); with a word of the drum kit ("fill", "kit", "beat", "drums") it is the kit's
`crash/*`.

```
$ earshot make "a fast whoosh as the title flies in" -o whoosh.wav
whoosh.wav: whoosh/fast (0.30 s, peak at 0.04 s, -19 LUFS, clean) [whoosh, fast]
also: whoosh/blade, whoosh/soft, whoosh/deep
not understood: flies
$ earshot make "a deep cinematic boom for the logo" -o boom.wav
boom.wav: boom/deep (0.37 s, peak at 0.13 s, -16 LUFS, clean) [cinematic boom]
also: boom/short, boom/door, boom/wide
$ earshot make "a trailer braam" -o braam.wav
braam.wav: braam/brass (3.17 s, peak at 0.66 s, Bb1, -14 LUFS, clean) [braam]
also: braam/hit, braam/swell, boom/wide
```

`--json` gives the number of the main recording each take played (`member`: `n` of `of`); `learned.credit(id, n -
1)` names the recording and its author.

## Voices

`earshot.synth.voices.table()`: every voice a figure or a note can be played by. Ranges are the playable notes; a
pitch outside is moved by octaves.

| voice | engine | material | range | default | sounds like |
|---|---|---|---|---|---|
| glockenspiel | struck | metal | C5-C8 | C6 | a glockenspiel note |
| celesta | struck | metal | C4-C8 | C5 | a celesta note |
| vibraphone | struck | metal | F3-F6 | F4 | a vibraphone note |
| marimba | played (real recordings) | wood | A2-D6 | C5 | a marimba note: a rosewood marimba struck with yarn mallets |
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
| felt_piano | played (real recordings) | felt | A2-D#6 | C5 | a soft felt piano note: an upright piano played softly |
| grand_piano | strings | piano | A0-C8 | C4 | a grand piano note |
| electric_piano | strings | electric | E1-E7 | C4 | an electric piano note |
| reed_piano | strings | electric | A1-C7 | C4 | a reed electric piano note |
| harp | played (real recordings) | string | A2-F6 | C5 | a harp note |
| nylon_guitar | strings | string | E2-B5 | A3 | a classical guitar note |
| steel_guitar | strings | string | E2-E6 | A3 | an acoustic guitar note |
| muted_guitar | strings | string | E2-E5 | E3 | a palm-muted guitar note |
| pizzicato | played (real recordings) | string | G#2-C6 | C5 | a pizzicato string note: a cello section, a solo violin above B4 |
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
figure at least 110 ms long, the last 1.6 times that: damped earlier it is heard as an electronic beep. The voices
other than the four real instruments, the bells and bowls above all, were judged weak by ear: they are for someone
making real music, which is rarely what an interface or a game needs.

## The engines

**Learned sounds** (`earshot/synth/resyn.py`): analysis and resynthesis at the resolution of hearing
([Learned sounds](#learned-sounds)). On 41 public-domain sources the band envelopes of a take sit a median 2.4 to
3.2 dB from the recording's at 2 ms, the spectral shape within 0.7 to 1.3 dB, a click's rise within 1 ms; a render
takes 0.26 to 0.33 s per second of sound on one thread.

**Learned beds** (`earshot/synth/quilt.py`): fresh takes of a texture's models, cut into pieces of 1 to 3 s (whole
waves for the sea) and stitched where the band levels on both sides match; a piece is not played again within 20 s
where the material allows it.

**Real instruments** (`earshot/synth/played.py`, `earshot/data/notes/`): single notes of a marimba, a harp,
pizzicato strings and an upright piano learned through the learned-sound engine; the take nearest in pitch and
dynamic is moved to the note asked for and damped as a player damps it. Inner notes of a figure ring on under the
next one, the last rings out, and a figure whose top note lies above C6 is played an octave lower.

**Performances** (`earshot/synth/perform.py`): when each event of a walk, a knock, typing, a scatter, a shake or a
pair starts, how loud it is and which recording plays it; never the same recording twice in a row, never on a grid.

**Drums** (`earshot/sounds/drums.py`, `earshot/data/drums/`): the acoustic kit and hand percussion are takes of
recorded strokes through the learned-sound engine; the drum-machine voices, the brush snare, the finger snap and the
stick count are invented (`earshot/synth/drums.py`).

**Struck instruments and strings** (`modal.py`, `struck.py`, `strings.py`): modes at the ratios their geometry
gives, each dying at the rate its material gives it, struck by a contact pulse whose length follows the mallet's
hardness; stiff strings, felt hammers, soundboards and electric pianos. They were compared note by note with freely
usable recordings (University of Iowa Musical Instrument Samples; VSCO-2 Community Edition, CC0); no recording is in
the repository.

**Synthesizer** (`voice.py`, `patches.py`): band-limited oscillators, unison with drift, a ladder filter, FM
operators, chorus and echo; 28 patches.

**Chip** (`chip.py`): pulse, triangle, noise and wavetable channels with every level step band-limited.

**Textures from rules** (`texture.py`): rain, wind, forests and nights from many small events per second with
physical size statistics and band-wise noise, rendered around a circle so they loop.

**The shared room** (`space.py`): seven spaces (dry, booth, room, studio, hall, plate, cathedral) that every design
uses, so a family of sounds sits in one room.

**Recipes** (`earshot/recipe.py`): layers of catalogue designs or plain sources, each a take of its own, shaped and
mixed in one room ([When earshot has no such sound](#when-earshot-has-no-such-sound-build-it)).

## Recipe: a web page

```bash
earshot make --set ui -o public/sounds --format webm,mp3 --sprite sprite
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
levels them low already. A loop's own file is webm and flac; inside the sprite it is safe in the webm, not in the
mp3 (the seam).

**Which moments get a sound.** A page plays a sound for what the user did on purpose and for what came of it: a
switch thrown, a form sent, a payment gone through, an error, a message arriving. It plays none for what happens by
itself or all the time: the page loading (no browser plays audio before the first click), scrolling, an animation,
the pointer crossing its links. This is common practice, not a measurement. The seventeen roles are a palette, not
a list to use up: a landing page with a pricing toggle, a sign-up form, a modal and a toast is served by three to
six sounds. A spec without `set` makes only the ones it lists:

```json
{"sounds": [{"name": "open", "want": "a modal opening"},
            {"name": "toast", "want": "a toast notification"},
            {"name": "error", "want": "an error sound for a wrong email address"},
            {"name": "card", "want": "a soft low thump when the pricing card settles"},
            {"name": "paid", "want": "a payment success sound"}]}
```

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

## Recipe: a game

```
$ earshot make "three variations of a footstep on gravel" -o step.ogg
step_1.ogg: step/pebbles (0.51 s, peak at 0.10 s, -15 LUFS, clean) [gravel, footstep]
step_2.ogg: step/pebbles (0.40 s, peak at 0.07 s, -16 LUFS, clean)
step_3.ogg: step/pebbles (0.60 s, peak at 0.02 s, -16 LUFS, clean)
played: one event; the action: events=<n> or say it ("footsteps", "knocking", "three knocks")
also: step/gravel, step/floorboard, step/heels

$ earshot make "a campfire" -o fire.ogg --dur 24
fire.ogg: fire/camp (24.00 s, peak at 19.96 s, -22 LUFS, seamless loop, clean) [campfire]
also: fire/large, fire/small, fire/whoosh
```

- **Variations.** A game plays a footstep hundreds of times: ask for 6-12 takes (`--n`, or "six variations") and
  pick one at random per play. For a learned design the first takes are its different recordings, then new takes
  of them; for an invented one, two seeds differ in timing, strike points and events. The three takes above are
  three single steps cut from one recording, of different length and level: single takes are levelled to
  their own peak, so pick-at-random sets are better made as a set (`"n": 8` in a spec), where every take gets the
  kind's level.
- **Loops for levels.** Ambience is a seamless loop by default. A learned bed is stitched from its recording without
  a loop point; a longer loop repeats less. Ship loops as ogg, webm, wav or flac, not mp3 or m4a.
- **One family.** `--set game` (modern, kalimba theme) or `--set arcade` (chip); add or replace sounds with a spec,
  for example the learned `person/fall`, `punch/body` or `explosion/near` in place of the invented action sounds.
  Levels inside a set follow the kind table above: a coin sits under an explosion, a step under a coin.
- **Chip sound** only when the request asks for it ("8-bit", "retro", "chiptune", "arcade") or the theme is retro.
- **Rewards are figures, blows are blows.** A coin, a power-up and the end of a level come as figures on an
  instrument without being asked: "a coin pickup" is `coin/bright` (0.69 s of glockenspiel), "a level complete
  fanfare" `win/fanfare` (2.90 s on a marimba). One enemy defeated is a blow (`hit/crit`, 0.59 s) and a boss
  defeated the fanfare; the player's own death is the falling tune (`lose/descend`, 2.14 s). Every action the
  player repeats (steps, jumps, hits, coins) wants several takes; a menu one small sound for moving and one for
  choosing.

## Recipe: a film

```
$ earshot make "a logo reveal" -o logo2.wav
logo2.wav: logo/bag (0.47 s, peak at 0.30 s, -14 LUFS, clean) [logo reveal]
also: logo/door, sting/reveal, boom/short

$ earshot make "logo reveal" --voice felt_piano --pitch D4 -o logo.wav
logo.wav: logo/bloom (3.97 s, peak at 0.29 s, D4, felt_piano, -14 LUFS, clean) [logo reveal]
also: logo/duo, logo/motif, logo/bag

$ earshot make "a forest at night" -o night.wav --no-loop --dur 10
night.wav: jungle/night (9.96 s, peak at 7.11 s, -22 LUFS, clean) [night, forest]
also: night/crickets, forest/morning, rain/wood
```

The night is a learned bed (a recording of a jungle at night), rendered once with a fade in and out.
For the film's moves and hits see [Film and motion](#film-and-motion). A logo with notes is asked for in words that
name music (a voice, "a sonic logo", "a motif"); "a logo hit" or "a logo reveal" without them gets a short low hit
or a swell of air into one (`logo/bag`, `sting/reveal`).

- **Few effects**: motion films are led by music; give two to five moments a sound (a stop and the hit after it,
  something appearing, a click, the title, the logo) and leave the rest to the music. The skill's "Scoring a motion
  video" says which words give which sound.

- **Beds once, not looped**: `--no-loop` gives a scene with a fade in and out at the length the shot needs.
- **In the film's key**: `--pitch` or "in D" for logos, stingers and notes, so they sit with the music.
- **The bank roles with sync points**: `earshot make --set film` renders a starter kit as files. For effects placed
  on cue frames use the bank's kits (`earshot kit init --preset launch`, `kit render`): every take carries its
  measured onset, peak and end, and `kit.place`, `kit.gain_db`, `kit.bed_level` and `kit.space_run` place and level
  it against the music ([SFX.md](SFX.md)). A made file starts when its sound does (within 2 ms); where its loudest
  moment lies is measured below.
- **Check it landed**: `earshot timeline mix.wav --video film.mp4 --cues cues.json` and `earshot sheet` (the
  listening half, README).

**Where a made file's loudest moment lies.** Every result line says it (`peak at`), and `--json` gives `peak_s`,
`onset_s` and, for a row, `events_s`. earshot does not put files on a timeline: the editor, ffmpeg or the page's
code does. To put a hit on the frame at t, start the file at t minus `peak_s`. A whoosh's loudest moment goes on the
fastest frame of the move; a riser ends on its loudest moment, so it starts its own length before the reveal.

**A scene of real footage** (a film scene, a cooking clip) takes one bed for the place and the real sound of each
thing the picture shows being done, asked for as that thing. Made as one spec, the bed comes out 9 to 16 LU under
the things done in it, a first balance to mix from (`"lufs"` in an entry sets one sound's own level):

```
$ earshot make --spec scene.json -o scene
5 sounds in scene (wav), manifest.json
rain.wav: rain/street (7.93 s, peak at 4.12 s, -30 LUFS, clean) [terrace, rain]
  7.93 s, shorter than the 8 s asked: the longest this sound is made
door.wav: door/shut (2.54 s, peak at 1.27 s, -20 LUFS, clean) [door, shut]
keys.wav: keys/bunch (4.21 s, peak at 1.57 s, -15 LUFS, clean) [keys jingling]
  played: 18 events (a recording of the whole action)
phone.wav: phone/vibrate (1.64 s, peak at 0.38 s, -15 LUFS, clean) [phone, table, vibrating]
thunder.wav: thunder/distant (7.70 s, peak at 3.22 s, -21 LUFS, clean) [thunder, distant]
```

The door is the right recording: a word that says which one or in which room ("front", "kitchen") leaves the thing
known, while one that names another thing ("a vault door") is answered as a guess that says so.

The entries asked for "steady rain on a terrace, 8 seconds, no loop", "the heavy wooden front door falls shut",
"keys jingling", "a phone vibrating on a table" and "distant thunder". A thing the catalogue has no recording of
comes back as a guess that says so: "a fire alarm" answered `also (the pick is a guess: earshot makes no fire
alarm; timer/done is only the nearest sound it has)`. Leave such a moment silent or record it; a stand-in
is heard as one. A designed sound (a whoosh between cuts, a pop for a caption) belongs on the few cuts or captions
the edit wants felt, not on each.

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
span)`, `c.vary`, `c.voice`, `c.room()`), `r` is a numpy Generator (every random number comes from it: the same seed
gives the same samples). Return mono `(n,)` or stereo `(n, 2)` at 48 kHz at any level; `render` adds the low cut, the
edge fades, the tail trim, the level and the audit. `says` describes the sound as a listener would (never its job),
`uses` lists the jobs it plays, `axes` names exactly the axes the function reads, and `allows` lists audit findings
that are the design (`rising_glide`, `stacked`, `pre_swell`, `tail`, `truncated`). Every design should answer to
`energy` and two more axes in a way a listener would agree with, and two seeds should give different, equally good
takes. Then rebuild the sound atlas (`python -m earshot.sounds.atlas build --work DIR`, repeated until done) so
`--listen` knows the new design, and run the catalogue's contract tests (`tests/test_sounds_catalogue.py`).

Learned designs are not written by hand: each is an entry of `earshot/data/resyn/index.json` with its member models,
built from CC0 recordings that pass checks for a full band, a natural decay, a clear onset and
a single event; the build tool is not in the repository. A design written from formulas for a real thing should be
listened to by a person before it ships: such designs were judged poor by ear while the models scored them well.
