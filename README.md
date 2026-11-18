# earshot

**Ears for coding agents.**

AI coding agents write the code for interfaces, games and films, and cannot hear any of it. earshot gives them both
halves of sound. An agent asks for a sound in words and earshot makes it: it picks a design from a catalogue and
renders a new take. Interface sounds, game sounds, everyday foley, ambience beds, drums, chip sound and film effects.
Some designs are written in code from physics and synthesis; the realistic ones are learned from recordings in the
public domain and rendered anew each time (no audio is shipped). And it listens: what a file sounds like in words, how
well it fits a description, the best moment in a long recording, and a second-by-second account of a film's
soundtrack. It runs locally on CPU as a CLI, a Python API and an MCP server; everything it makes is yours to use.

```
$ earshot make "a soft success sound for a finance app" -o success.ogg
success.ogg: success/click (0.30 s, -17 LUFS, clean) [success]
also: complete/stamp, complete/latch, success/shutter

$ earshot make "three variations of a footstep on gravel" -o step.ogg
step_1.ogg: step/pebbles (0.51 s, -15 LUFS, clean) [gravel, footstep]
step_2.ogg: step/pebbles (0.40 s, -16 LUFS, clean)
step_3.ogg: step/pebbles (0.60 s, -16 LUFS, clean)
played: one event; the action: events=<n> or say it ("footsteps", "knocking", "three knocks")
also: step/gravel, step/floorboard, step/heels

$ earshot make "a campfire" -o fire.ogg --dur 24
fire.ogg: fire/camp (24.00 s, -22 LUFS, seamless loop, clean) [campfire]
also: fire/large, fire/small, fire/whoosh
```

The three footsteps are three single steps cut from one real recording (recordings 6, 7 and 8 of the nine that
`step/pebbles` holds, from BigSoundBank), each rendered as a new take; the campfire is a recording of a real fire stitched into a
24 s loop that has no seam. `make` loads no model and takes about a second for a one-shot.

## Make sounds

`earshot make` reads a request, ranks the catalogue, renders the best design, audits it without models (clicks, cut
tails, clipping, DC) and writes the file. One call answers in a few lines: what was chosen and why, what else would
fit (`also`, with a note when the pick is a guess), what the request named that the sound does not cover, and which
words were not understood or not used. The full reference: [docs/MAKE.md](docs/MAKE.md).

### What it can make

`earshot sounds` lists every design (`--search`, `--domain`, `--family` narrow it). Ids are `family/name`: what the
sound is, then its character.

| group | designs | how it is made | examples |
|---|---|---|---|
| learned things | 201 | learned from 517 recorded events in the public domain (animals, people without words, street, house and devices among them) | `keyboard/desktop`, `switch/lamp`, `step/pebbles`, `door/shut`, `bag/drop`, `cork/bottle`, `camera/shutter`, `sword/draw`, `explosion/near`, `page/turn` |
| interface | 26 | things a hand does, built from learned recordings: a key, a rocker switch, a page, a pen cap and air, knocks, a pop, a pen click, a stapler, crumpled paper; no note | `keyboard/mechanical`, `toggle/rocker_on`, `next/page`, `notify/knock`, `success/click`, `error/thud` |
| interface and game figures | 45 (and 3 ticks) | notes in time, played by a voice (by default a real marimba); for interfaces only when the words ask for music or a theme is named | `success/rise`, `error/fall`, `notify/pair`, `send/lift`, `coin/bright`, `level_up/fanfare`, `logo/bloom` |
| ambience | 31 | 13 learned from 12 real textures, 18 from rules | `fire/camp`, `waves/sand`, `night/crickets`, `cafe/busy`, `city/street`; `rain/window`, `forest/morning` |
| chip | 24 | the channels of 8-bit consoles, band-limited | `coin/chip`, `jump/chip`, `laser/chip`, `explosion/chip` |
| drums | 40 | 33 played from 130 recorded strokes in the public domain (the acoustic kit and hand percussion); 7 invented (drum-machine voices, a brush snare, a finger snap, stick counts) | `kick/soft`, `snare/tight`, `hat/closed`, `conga/muted`, `darbuka/hit`, `drum/roll`; `kick/electro` |
| film and motion | 46 | 32 built from learned recordings as motion films use them: short dry hits (a heavy bag dropped on a real explosion's low end), low thumps, bright ticks, whooshes and risers of real air, a tape stop, a braam of real brass; 14 small ones invented (pops, tocks, glints, key ticks) | `slam/title`, `boom/short`, `thump/appear`, `tick/soft`, `whoosh/soft`, `rise/air`, `hit/drop`; `pop/soft` |
| action and objects | 83 | invented from physics | `jump/soft`, `hit/punch`, `coin/drop`, `paper/tear`, `bell/desk` |
| notes, synthesizer | 51, 14 | real instruments learned note by note (marimba, harp, pizzicato, upright piano), struck and string models, synth patches | `note/kalimba`, `pad/warm`, `braam/hit` |
| music loops | 3 | a small grammar of harmony and feel | `bed/warm`, `bed/playful`, `bed/8bit` |
| bank | 209 | the 0.2-0.4 sound bank, stored as code | `whoosh_fast/crisp`, `riser/tension`, `typing_key/light` |

Where a learned design and an invented one answer the same request, the learned one wins the tie: the owner rated
the invented realistic sounds 1-2 of 5. Where an invented design and a learned one are the same thing (`knock/door`
and `door/knock`, `paper/tear` and `paper/rip`: 33 invented designs in `lexicon.TWINS`), a request that names the thing gets
the learned one; the invented one is still there by its id. Chip designs answer only requests that ask for chip sound. earshot is a
sound-effects tool: the three music loops are short beds for a game menu or a demo, not music generation.

### How it understands requests

| the words | what they do | example |
|---|---|---|
| the thing, the job or the sound | choose the design: things ("a light switch", "a cork"), jobs ("success", "coin pickup", "toggle") and sounds ("chime", "whoosh", "rain") | `"a notification"` -> `notify/knock` |
| feelings | offsets on eight shared axes: bright, hard, ring, size, space, width, energy, motion | `"a much bigger, darker whoosh"`: size +0.84, bright -0.6 |
| negation, comparison | "not harsh", "not too loud", "the opposite of a success sound", "like a ..." | `"the opposite of a success sound"` -> `error/thud` |
| voice | the instrument that plays a tonal sound, or a single note of it | `"a notification on a harp"`, `"a success sound with pizzicato strings"`, `"a kalimba note in E"` |
| one event or the action | a plural, an action word, a count or a length plays the action a person performs; a singular or "a single" one event | `"footsteps on gravel"` (a walk), `"a footstep on gravel"` (one step), `"three knocks"` |
| material | picks the voice of a tonal sound (wooden: a marimba) and steers the ranking; "8-bit", "retro", "chiptune" allow only chip sound | `"8-bit coin pickup"` -> `coin/chip` |
| numbers | length, pitch, key (a minor key makes a figure minor), how many takes, how many notes | `"400 ms"`, `"C6"`, `"in D minor"`, `"three variations"` |
| loop or not | ambience loops by default; "no loop" or `--no-loop` renders it once with fades | `"rain on a tent, no loop"` |

A learned design answers only a request that names its thing ("a footstep on gravel", "a camera shutter", "a light
switch"); a feeling alone never reaches a recording. Options win over the words: `--pitch`,
`--dur`, `--voice`, `--events` (1 = one event, 2 or more = the action with that many) and `--bright` ...
`--motion` (each -1 .. 1) are laid over what the words say, and `--sound ID` takes a design by name. Re-ask with adjectives ("softer", "drier", "bigger", "longer") rather than browsing.

On 220 held-out requests written by someone who never saw the catalogue (half long descriptions that never name the
sound, half casual), scored once on the v1 catalogue, the first pick is right for 69.1 % and one of the top three
for 76.4 %. Requests that name the thing or the job do far better than descriptions of a mood or paraphrases; about
one request in three or four of the held-out kind needs a second call, which is what `also` and `--sound` are for
([docs/VALIDATION.md](docs/VALIDATION.md#understanding-requests)).

### Learned sounds: what they are

The 201 learned designs, 13 ambience beds, the recorded drums, the instrument notes and the film sounds layered
from them are not invented. Each comes from recordings in
the public domain: BigSoundBank, Kenney, VCSL, VSCO-2 CE, Karoryfer and CC0 uploads on Freesound, all CC0 or an
explicit public-domain dedication. earshot keeps each recording as numbers, not audio: the level of 56 bands on a
2 ms grid (0.25 ms around an onset), how tonal each band is and at which frequency, and the stereo image
(`earshot/synth/resyn.py`). The store is 11.05 MB for 517 recorded events (`earshot/data/resyn/`; a few are several
events cut from one recording, 23 are recordings of a whole action such as a walk), 5.40 MB for 12 textures
(`earshot/data/textures/`), 1.31 MB for 130 drum strokes (`earshot/data/drums/`) and 0.83 MB for 97 single notes
of four instruments (`earshot/data/notes/`), with a credits file in each. No audio file is shipped or played back. A render draws fresh
noise and applies the request's controls (size, brightness, attack, ring, length within 0.6-1.8 times the natural
one, energy, room, width), so every take is new. But it is a new take of that recording: a compact description of a
real sound rendered again, not a sound made from nothing.

A design is one thing and its recordings are its variations: seeds 0, 1, 2 walk through them, from the one with
the best listening score down, so three seeds of `step/pebbles` are three real footsteps; seeds past the last
recording are new takes of the same ones. A thing a person does several times in a row (a walk, a knock, typing,
coins, keys, clinking glasses) is played as that action when the words ask for it: a recording of the whole action
where there is one, else its single recordings in the pattern a person plays them (`earshot/synth/perform.py`); a
length asked with it ("three knocks, 2 seconds", "footsteps, 5 seconds") is the file's, the events spread over it
at gaps a person keeps. A request for
a darker or brighter take draws from the darker or brighter recordings.

Why this way: in the owner's listening test of 24 recordings, each next to two renders of it, he gave the recording
and its renders the same rating in every row. Sounds invented from formulas for real things (footsteps, doors,
punches, explosions, switches, taps) he rated 1-2 of 5, whatever earshot's listening models said of them. So for
real things earshot renders real recordings anew. A realistic sound exists only for things someone recorded and put
in the public domain.

### Themes and sets

A theme is what makes a product's sounds one family: one voice, one key, one room and offsets on the axes, shared by
every design that honours them. Presets: `soft` (a real marimba played softly, the default of the figures), `glass`
(glockenspiel), `wood` (a real marimba, pentatonic in G), `felt` (a real upright piano played softly), `tine`
(kalimba), `metal` (vibraphone), `electric` (electric piano), `digital` (glass pluck), `retro` (square lead); a dict
changes one (`{"base": "wood", "key": "D"}`). An interface or motion request gets no notes unless its words ask
for music (an instrument, a chime, a jingle, a melody) or a theme is named. Figures play the real marimba unless a
request or a theme names another voice (`--voice harp`, "on a harp", "with pizzicato strings", "on a soft
piano"): a bell or a glass timbre only when asked. A few game figures keep the voice they were written for (`coin/bright`: a glockenspiel).

The real instruments (`earshot/synth/played.py`) are a marimba with yarn mallets, a harp, pizzicato strings (a cello
section up to B4, a solo violin above) and an upright piano played softly (the voice `felt_piano`): 97 single notes
from VSCO-2 CE and VCSL (CC0), each played as a new take moved to the pitch asked for. Inner notes of a figure ring
on under the next one, the last rings out, and a figure whose top note lies above C6 is played an octave lower.

A ready-made set makes the sounds a kind of product needs in one call with a `manifest.json`: `ui` (17 interface
sounds), `game` (22: menus, rewards, jingles, actions), `arcade` (16 chip sounds), `film` (17 motion-design sounds).
Every one-shot of a set is as loud as it is made alone and only turned down so that none lies more than 2.5 dB over
the quietest (by its loudest 50 ms; `hover` in `ui` sits just under); loops and beds are levelled by kind. Every
role of `ui` is a normal sound, a thing a hand does: tap a mechanical key, hover a fingertip barely touching wood,
the toggles a rocker switch (on smaller and brighter, off bigger and duller), type a desktop key, next and back a
page flicked forward and dropped back, open and close a pen cap with air let out or drawn in, send a stick swung
through the air, receive a pop of the lips, notify two light knocks, success a pen clicked in two parts, complete
a stapler, warning two firm knocks on a door, error two dull thuds of a bag, delete paper crumpled. A plain
interface request in words ("a success sound", "toggle dark mode") gets the same; the tonal figures play only when
the words ask for music or a theme is named. A JSON spec adds to a set or describes your own (`earshot make --spec
sounds.json`, MCP `make_set`).

```
$ earshot make --set ui -o public/sounds --format webm,mp3 --sprite sprite
17 sounds in public/sounds (webm, mp3), manifest.json
tap.{webm,mp3}: keyboard/mechanical (0.32 s, -18 LUFS, clean) [named]
hover.{webm,mp3}: hover/fingertip (0.34 s, -23 LUFS, clean) [named]
toggle_on.{webm,mp3}: toggle/rocker_on (0.22 s, -19 LUFS, clean) [named]
...
success.{webm,mp3}: success/click (0.31 s, -17 LUFS, clean) [named]
error.{webm,mp3}: error/thud (0.49 s, -18 LUFS, clean) [named]
17 sounds in public/sounds\sprite.webm + sprite.mp3 (10.5 s, 92 kB, 165 kB), map sprite.json
```

### Sprites, loops, formats

- **Sprites**: `--sprite sprite` on a set, or `earshot pack FOLDER -o sprite.webm`, lays every sound in one file
  with a map in the format howler.js reads (`{"src": [...], "sprite": {name: [offset_ms, duration_ms, loop]}}`). A
  page loads one file instead of twenty. `pack` on a set's folder packs exactly the manifest's sounds.
- **Loops**: the ambience beds, the three music loops, a clock, a heartbeat, a laser beam, an earthquake rumble and a
  chip engine are seamless loops. A learned bed is stitched from pieces of its recording where the texture is in a
  similar state (`earshot/synth/quilt.py`), so it plays for any length without a loop point; the beds from rules are
  rendered around a circle. `--dur` sets the length; `--no-loop` (or "no loop") renders the scene once with a fade
  in and out, for a film. Write loops as ogg, webm, wav or flac: an mp3 or m4a loop has a gap at its seam in players
  that ignore the encoder's padding: a single `make` warns, and a set written as `webm,mp3` writes its loops as flac
  in place of the mp3.
- **Formats**: the extension picks it: wav and flac (24-bit; 16-bit in sets) directly; ogg (Vorbis), webm (Opus),
  m4a (AAC) and mp3 through ffmpeg. A set takes several at once (`--format webm,mp3`), and its sprite map lists them
  in order. For the web write `webm,mp3` (or `webm,m4a`): Ogg Vorbis does not play in Safari before 18.4. 48 kHz
  stereo, or `--mono`.
- **Loudness**: a single sound peaks at -1 dBFS true peak but no louder than -14 LUFS momentary max (pads and
  textures -18, ambience beds -22, music loops -16); `--lufs` sets it exactly. A short one-shot whose one spike
  would hold its body far down is raised up to 14 dB and the spike folded over, so a tap or a click is heard. In a
  set the one-shots keep that level within 2.5 dB of each other; loops and beds sit by kind under -16 LUFS (a bed
  14 LU under).
- **Takes**: the same arguments give the same samples; `--seed` and `--n` give other takes (for a learned design,
  other recordings of the thing first).

### The engines

**Learned sounds** (`earshot/synth/resyn.py`): analysis and resynthesis at the resolution of hearing, above. Its
fidelity on 41 public-domain sources: the band envelopes of a take sit a median 2.4-3.2 dB from the recording's at
2 ms, the spectral shape within 0.7-1.3 dB, a click's rise within 1 ms; a render takes 0.26-0.33 s per second of
sound on one thread.

**Learned beds** (`earshot/synth/quilt.py`): fresh takes of a texture's models, cut into pieces of 1-3 s (whole
waves for the sea) and stitched where the band levels on both sides match; a piece is not played again within
20 s where the material allows it.

**Struck instruments and strings** (`modal.py`, `struck.py`, `strings.py`): modes at the ratios their geometry
gives, each dying at the rate its material gives it, struck by a contact pulse whose length follows the mallet's
hardness (a harder strike is brighter, not only louder); stiff strings, felt hammers, soundboards and electric
pianos. The pianos and mallets were compared note by note with freely usable recordings (University of Iowa Musical
Instrument Samples; VSCO-2 Community Edition, CC0); no recording is in the repository.

**Synthesizer** (`voice.py`, `patches.py`): band-limited oscillators, unison with drift, a ladder filter, FM
operators, chorus, echo; 28 patches, among them the soft pluck and the square lead.

**Real instruments** (`played.py`, `earshot/data/notes/`): single notes of a marimba, a harp, pizzicato strings and
an upright piano learned through the learned-sound engine, the take nearest in pitch and dynamic moved to the note
asked for, damped like a player damps it.

**Performances** (`perform.py`): when each event of a walk, a knock, typing, a scatter, a shake or a pair starts, how
loud it is and which recording plays it; never the same recording twice in a row, never on a grid.

**Drums** (`earshot/sounds/drums.py`, `earshot/data/drums/`): the acoustic kit and hand percussion are takes of
130 recorded strokes (VCSL, Karoryfer, and one person clapping for the hand clap) through the learned-sound
engine; the seed walks through a drum's strokes and `energy` picks softer or harder ones. The drum-machine voices,
the brush snare, the finger snap and the stick count are invented (`earshot/synth/drums.py`: a beater's contact, modes, wires and jingles as collisions).

**Chip** (`chip.py`): pulse, triangle, noise and wavetable channels with every level step band-limited.

**Textures from rules** (`texture.py`): rain, wind, forests and nights from many small events per second with
physical size statistics and band-wise noise, rendered around a circle.

**The shared room** (`space.py`): seven spaces (dry, booth, room, studio, hall, plate, cathedral) that every design
uses, so a family of sounds sits in one room.

## Listen

| command | what an agent gets |
|---|---|
| `describe FILE` | what it sounds like: calibrated tags (z) with the closest alternative readings, AudioSet labels, measurements |
| `match FILE QUERY [--alts ...]` | how well a description fits: cosine, z among the vocabulary, a verdict (strong / plausible / weak / no match) with held-out evidence |
| `rank QUERY PATHS...` | many candidate files sorted by a description (pick the best download) |
| `find FILE QUERY` | the best moments inside a long recording, as sample trims (start / end seconds) |
| `compare FILE --refs DIR` / `ab A B` | nearest reference recordings and typicality; A/B two sounds or two trims, the differences in words |
| `realness FILE...` | real recording / pro-produced vs code-synthesized, with evidence and per-window scores |
| `timeline FILE [--video V] [--cues C]` | the listening transcript: events (time, level, tag), sections, loudness, beat grid, sync of sound to picture cuts and cue frames |
| `sheet FILE -o out.png` | an annotated PNG (video frames, sections, labelled events, spectrogram, waveform, loudness, sync markers) for agents that read images |
| `critique FILE [--video V]` | a launch-video scorecard against a profile of 7 professional launch films, with timestamped fixes |
| `profile REFS... -o p.json` | build your own reference profile from films you admire |
| `measure FILE` | objective measurements only (no models): LUFS, true peak, LRA, envelope, spectrum, tempo / key |
| `make WORDS -o FILE` / `make --set` | a sound made from a request in words, or a whole themed set (no models; above) |
| `sounds [--search WORDS]` / `pack` | the catalogue `make` draws on; many sounds in one sprite file with a map |
| `forge RECIPE --refset CAT` | search a recipe's parameters by listening (see below) |
| `fit FILE -o recipe.py` | translate a recording into a forge recipe module that re-renders it, with knobs and morphs (see below) |
| `sfx render ID -o x.wav` | a sound from the bank of 252, aimed at the picture (`--dur`, `--apex`, `--pitch`) and audited; `sfx list`, `sfx check` (no models) |
| `kit init / render / check` | a film's sound kit: one bank sound per role, rendered into cached takes with measured onset / peak / end; the `launch` preset places each role against the music bed and the cut (no models) |
| `gallery DIR -o page.html` | one static HTML listening page of forge outputs: players, spectrograms, numbers |
| `llm FILE --yes-upload` | optional audio-LLM critique (Gemini), only with your own key and explicit consent |
| `mcp` | the same tools as an MCP server for Claude Code and other agents |
| `doctor [--load]` | check the install, models, data files and CPU settings |

Every command prints a plain-language summary first; `--json` gives the full result, `--out f.json` saves it. Errors
are one line on stderr (`earshot <command>: error: ...`, exit 1). Each CLI call is a new process and loads the models
(~20-40 s) before it answers (`make`, `sounds`, `pack`, `measure`, `sfx` and `kit` load none): for more than a few
calls use the MCP server (models stay loaded) or one Python script with `from earshot import api`.

```
$ earshot match key.ogg "a deep thocky mechanical keyboard press" --alts "a clicky blue switch keyboard" "a mouse click"
  -> strong (z +3.7; would rank #2 among 322 vocabulary descriptions) ...

$ earshot find keys.ogg "a single mechanical key press" --top 4
  0:03.94-0:04.19 (0.24 s stroke, z +4.1 strong); 0:10.60-0:10.86 ...

$ earshot critique film.mp4
  VERDICT: close - 21 of 23 scored metrics in range ... FIX FIRST: 1. [major] nothing reaches body level until 0:01.01 ...
```

What an agent sees when it listens to a film with `earshot sheet` (frames, labelled sound events, spectrogram,
loudness, sync to the beat):

![earshot sheet of the launch film](docs/media/launch_sheet.png)

### Search by listening: earshot forge

Listening makes iteration possible. The agent writes a sound as code: a **recipe** of physical synthesis (modal
resonators, contact excitations, filtered noise, a computed room and a microphone; no recorded audio inside). The
forge renders candidates from the recipe's parameters, and a **critic** built from earshot scores each one:
description match against named confusions, realness, closeness to real reference recordings and measured shape, then
model-free penalties against the references: 1/3-octave tonal balance, the spectro-temporal envelope (a tail that dies
too fast shows here), artifacts (clicks, aliasing, DC / infrasound, harsh resonances, comb ring, cut tails, clipping)
and polish (attack, onset snap, tail smoothness). A cheap model-free **pre-screen** ranks many renders with the same
measurements, so the models only hear the best; an evolutionary **search** keeps the best and mutates them. Then the
agent reads the critic's words ("sounds like: a cork pop, a glass ting", "decay +2.4 outside the refs", "bumpy tail"),
looks at spectrograms next to the references, rewrites the recipe, and runs it again.

```bash
earshot forge --list                             # the recipes (click, impact, keypress, pop, whoosh, ...) + refsets
earshot forge keypress --refs path/to/real/keypresses/ --budget 144 --variants 4 --dir made/keypress
earshot forge whoosh --refset whoosh_fast --budget 96 --dir made/whoosh      # a reference set's 'forge' split
earshot forge whoosh --target "a fast crisp air whoosh" --alts "a wind gust" "a sword swing" \
  --refs path/to/real/whooshes/ --mode real --dir made/whoosh
earshot forge --render keypress --params made/keypress/report.json --seeds 1 2 3 --dir made/again
```

Output (`--dir`): the best diverse sounds as WAVs; `report.md` (what was asked, the search per generation, each pick in
the critic's words, the measured gap to the references); `compare.png` (each pick's spectrogram, envelope and
1/3-octave balance next to its nearest reference); `audition.wav` (made, ref, made, ref ... to listen in pairs);
`report.json` with every candidate's parameters and scores, the same critic's scores of the real references
(leave-one-out) and held-out AudioSet tags. Picks are re-rendered on new seeds (`--robust`, default 3) and must differ
in parameters and in sound. Reference sets (`earshot/forge/refsets.py`) are split by source into a 'forge' side the
search may use and a 'bench' side only the fixed benchmark (`eval/forge/bench.py`) hears. The DSP toolkit for recipes:
`earshot/forge/dsp.py` (sources: modal, noise, air, sweeps, Shepard risers, sub drops, bells, FM; layering) and
`earshot/forge/fx.py` (reverb, saturation, transient shaping, limiter, EQ match, Doppler, filters, glitch, granular).
In Python: `from earshot.forge import forge`.

**forge v2 (0.2.0).** Recipes come as families of named characters: 34 motion-design categories (whooshes, swishes,
swipes, risers, downlifters, zips, impacts, sub drops, logo stings, clicks, taps, toggles, hover ticks, pops,
notification dings, shimmers, glitches, key presses), 16 of them modelled on the designed effects of professional
launch films ([docs/motion-sfx-study.md](docs/motion-sfx-study.md)), 4-6 variants each. A fixed benchmark scores them
against held-out recordings the forge never heard. Against v1 on it: variants are no longer near copies (CLAP cosine
between variants 0.89-0.97 -> 0.71-0.85 in the five categories v1 had; real sets 0.52-0.77), tails that died too fast
now reach the real length (click t40 0.56 -> 1.44 x the bench median, fast whoosh 0.51 -> 0.88), and the tonal
balance is closer in 3 of those 5 (key press 8.2 -> 7.4 dB, fast whoosh 8.6 -> 6.2, impact 6.7 -> 4.4). Across all
34, the made balance is at least as close to the held-out recordings as a second set of real recordings in 18 of the
21 categories that have a number, and every final re-renders from its `report.json` (`earshot forge --check`). No v2
blind-test score is published yet; the blind test below is the v1 one. Every number and what is still weak:
[docs/FORGE_V2.md](docs/FORGE_V2.md).

The reference recordings are licensed and not in this repo, so `--refset` needs a local registry: bring your own refs
with `--refs DIR`, or lay them out as described in `earshot/forge/refsets.py`. Without refs the critic runs in
designed mode (realness reported, not required). The CLI defaults changed in 0.2.0 (pre-screen, seed bank, robust
re-renders, the recipe's reference set); the Python `forge()` defaults still reproduce v1.

**Blind test (v1 recipes).** A key press, a UI click, a pop, a fast whoosh and an impact were made from scratch in
code and played blind next to real recordings of the same kind of sound. The author picked the made one correctly **8
times out of 15**: a coin flip. Only the made impact was spotted (its tail dies too fast). One listener and 15 trials
is a small test, not a proof.

**Limits (Goodhart).** The search optimises earshot's own models, and those can be pleased without the sound getting
better. After a search the made sounds pass every model check, held-out ones included, so the model checks stop
telling made from real; the 1/3-octave measurement still can. The realness model is easy to fool (a modal resonator
plus a room and a noise floor reads "real"). Judge a made sound by the model-free measurements, spectrograms next to
the references, robust re-renders, and a human blind test.

### earshot fit: audio to code

`earshot fit` translates a recording into code by analysis by synthesis. It fits a small synthesis model with readable
parameters (ringing modes, tracked partials, a time-varying noise gesture, or a layered hybrid of them) until its
render matches the recording, then writes the model as a forge recipe module whose `render()` makes the sound. The
module stores the structure of the sound, not its samples, so it can make new takes, variations (size, damping,
pitch, brightness, decay, ...) and morphs between two recordings.

```bash
earshot fit click.wav -o made/click_fit.py --report    # recipe + its render + side-by-side spectrogram + metrics
python made/click_fit.py 3 take3.wav                   # the module renders on its own (seed 3)
earshot fit --morph a.wav b.wav --amount 0.5 -o made/ab_morph.py
earshot fit --forge made/click_fit.py --budget 96 --dir made/click_variations   # critic-searched variations
```

Measured with the earlier engine (0.2-0.3) on 12 real recordings (2 each of click, key press, pop, impact, fast
whoosh, notification): median CLAP cosine 0.945 between each recording and the render of its generated code (minimum
0.881), and the same top `earshot describe` tag in 10 of 12. In a 12-trial blind test of that engine (recording vs
render, loudness-matched, random order) the listener could not tell which clip was the recording; one listener is a
small test, not a proof. Neither was re-run on 0.4.0, whose choice changed (the gesture model, the CLAP margin rule,
the gate, the finish); its benchmark below is measurement only (CLAP and the mel gap are proxies; nobody listened).
The renders are close, not identical: onsets, tone balance and noise texture still differ measurably; the selection
weights were set on the same 12 recordings; zips, long glides and swipes are untested (the benchmark below covers
risers, shimmers, swells and camera moves). A fit is a derivative of its recording, so keep the source's licence with
the recipe. Limits:
[docs/FIT.md](docs/FIT.md#limits-honest).

**Re-making professional sound effects (0.4).** The replication benchmark (`eval/forge/replicate.py`) fits a private
set of licensed reference clips (116 clean one-gesture effects in 24 categories; not in the repo: no audio, no clip
name and no per-clip model ships; aggregate measurements of them (medians, IQRs), CLAP cosines to them and parameters
tuned toward them do) and measures each fit and four takes against its clip. 0.4 adds a noise-gesture model (tones plus
air), stereo width and decorrelated tails, music-bed stripping decided by analysis by synthesis, gates for pulses
and hard stops, and takes that keep the fit's structure. It was tuned on 78 dev clips; two held-out sets were each
opened once:

| clips | earshot 0.3 | earshot 0.4.0 |
|---|---|---|
| dev (78): median CLAP, fits graded high | 0.884, 19 | 0.921, 40 |
| held-out (38): median CLAP, high | 0.878, 7 | 0.909, 14 when opened (first-round engine); 0.903, 14 for 0.4.0 on the opened set |
| fresh held-out (28, harder: more music under the effects): median / mean CLAP | 0.853 / 0.844 | 0.851 / 0.860 |
| fresh held-out: mel gap to the clip, grades high / good / fair / poor | 0.87 dB, 2 / 6 / 14 / 6 | 0.58 dB, 5 / 6 / 13 / 4 |
| fresh held-out: takes that pass the take check, take CLAP | 8, 0.809 | 7, 0.863 |

On the harder fresh set 0.4.0 is level with 0.3 on the median and ahead on the mean, the spectrogram gap and the
grades (up on 7 clips, down on 2), but fewer of its takes pass, the bed check can drop a bed the fit needed, and
onsets came out blunter (1.00 octave off against 0.53). Every number and what is still weak:
[docs/FIT.md](docs/FIT.md#the-replication-benchmark-04).

### Sound for films: bank, sfx, kits

The forge's delivered sounds ship as a bank of 252 that renders by id, with no models and no network: 183 forge v2
finals (34 categories), each stored as its render block and re-rendering its delivered WAV to one 24-bit step; 51
sounds made in 0.4 for music-led launch films (`ui_pop`, `knock`, `cut_hit`, `mallet`, `drip`, `shutter`,
`reverse_swell` and new members of older families); three tuned kalimba voices; and 15 sounds fitted from CC0
recordings (credits in [docs/CREDITS_FIT.md](docs/CREDITS_FIT.md)). Its 209 named members are also in the `make`
catalogue.

```bash
earshot sfx list --kind whoosh                                      # ids, lengths, peak times, cue kinds, knobs
earshot sfx render whoosh_fast/crisp -o w.wav --dur 0.8 --apex 0.32  # a new take aimed at a move (--seed N)
earshot sfx render pluck/kalimba -o e5.wav --pitch E5                # tuned through the recipe's own f0
earshot sfx check w.wav                                              # clicks, cut tails, clipping, ring, rising glides
earshot kit init --preset calm -o audio/kit.json                     # calm | crisp | minimal | launch
earshot kit render audio/kit.json --out audio/kit                    # takes + kit.lock.json (measured onset / peak / end)
```

- **Knobs** edit each recipe's own parameters, never resample: `dur` lands exactly, `apex` puts a whoosh's loudest
  moment within 5 ms of the target (measured within 2.5 ms), `pitch` takes a note name or Hz; noise sounds refuse a
  pitch and say why.
- **The audit** (`earshot sfx check`, run on every render) is measurement only: clicks, cut tails, clipping, DC, comb
  ring, a re-rising tail, a tonal pitch that climbs more than 4 semitones, and for hits, stacked attacks or a build
  before the hit. 236 of the 252 sounds are clean at their own seed (all 51 of 0.4's); the presets use none of the
  others. `make` runs the same audit on everything it renders.
- **Kits** name one sound per role (tick, hit, whoosh.short, key, bell, text ...). `kit render` writes deterministic,
  cached takes and a lock with each take's measured sync points; a take that fails the audit is re-rolled on a new
  seed. calm, crisp and minimal sound every role but riser, swell and suck, use one clean hit for the lockup, and use
  the fitted recordings for clicks, ticks, keys, touches and the snap where they measure closer to the role than the
  forge member (earshot match and realness); they render byte for byte as in 0.3.0.
- **The launch preset (0.4)** is for films cut to music. Each role names its level over the music bed (`bed_db`: the
  mix's loudest 50 ms at the effect over the music's median level in the second before it, 3-10 dB), where it sits
  against the cut (`offset_ms`: a tap -60 ms, a knock -40, a click -11, a pop-in thump +30, a whoosh peak +37, a swell
  peak +42) and how close a run may go (`min_gap_ms`: ticks 66, keys 87, pops 114), and sets ticks and bells wide
  (`width`, an L / R correlation). The numbers are medians measured on the reference clips. The music steps down
  before a hit instead of ducking under every effect (the lock's `mix` block). All 72 takes pass the audit. A mixer
  turns them into placements:

```python
from earshot.forge import kit
lock = kit.render('audio/kit.json', 'audio/kit')                     # from `earshot kit init --preset launch`
role = lock['roles']['pop']; take = role['takes'][0]
start = kit.place(take, role, cue_s=4.20)                            # file start: its onset 4 ms after the cue
gain = kit.gain_db(take, role, kit.bed_level('music.wav', 4.20))     # dB on the take: 8.5 dB over that bed
times = kit.space_run([5.00, 5.05, 5.10, 5.16], role['min_gap_ms'])  # a run of pops, at least 114 ms apart
```

In Python: `from earshot.forge import bank; y, meta = bank.render('pro_sub_drop/ref', pitch='G1')`;
`earshot.api.sfx_render / kit_render`; MCP tools `sfx_list`, `sfx_render`, `kit_render`. Full reference (every knob,
the audit, the kit and lock formats, the presets role by role, measured cost):
[docs/SFX.md](docs/SFX.md). How cinetic uses it: [docs/INTEGRATION.md](docs/INTEGRATION.md#cinetic-uses-earshot).

### Gallery

`earshot gallery` turns forge outputs into one static HTML page for people with ears: no server, no network, no
external scripts, inline CSS. One card per category, grouped by family (whooshes, sweeps, impacts, UI, tonal,
digital); per sound a player, a spectrogram thumbnail (log frequency, one time span per card so lengths compare),
duration, peak and momentary-max LUFS, the critic's total / pass, and the recipe parameters of every variant in a
collapsible table. The repo holds no audio, so point it at your own forge runs (`out/forge_v2` below is one folder
per category).

```bash
earshot gallery out/forge_v2 -o out/gallery/gallery.html                  # <category>/final/*.wav (+ report.json)
earshot gallery out/forge_v2 -o g.html --results eval/forge/results_v2r6.json   # + held-out benchmark numbers
earshot gallery out/forge_v2 -o g.html --refs forge                        # + A/B row: each sound's nearest forge ref
earshot gallery --compare "r2:out/forge_v2/*/final_r2" "r3:out/forge_v2/*/final" -o compare.html \
  --results r2:eval/forge/results_v2r2.json                                # before / after, variant under variant
earshot gallery out/forge_v2 -o portable.html --embed                      # audio (FLAC) + images inside the file
earshot gallery out/forge_v2 -o share.html --embed-compressed              # one small file: MP3 previews, plays anywhere
earshot gallery --compare "v1:old/*/final" "v2:out/forge_v2/*/final" --alias keypress=typing_key -o v1_v2.html
```

Audio is referenced by relative path (or embedded with `--embed`); thumbnails go to `<page>_files/` next to the page
and are cached by content. `--embed-compressed` makes one self-contained page to send or open on a phone: each sound
encoded by ffmpeg (MP3 192 kbps by default; `--embed-codec opus` or `vorbis` for Ogg, `--kbps` for the rate), JPEG
thumbnails, no links to local files; 183 sounds come to about 9 MB. The numbers on it are still measured on the WAVs.
`--alias OLD=NEW` puts a folder under another category's card (an older recipe's `keypress` under `typing_key`).
A "match loudness" switch (on by default) turns louder players down so a riser does not blast after a click; it
changes the player volume only, never the files. Benchmark numbers are shown only for files the benchmark actually
measured: a file newer than the results JSON, or with another length / peak, gets none (renamed files are matched by
length and peak, and the page lists every such case). In `--compare`, files identical to an earlier version are marked
`= r2`. Bench references never go on a page: `--refs bench` is refused, every reference passes `refsets.held_out()`,
and only the benchmark's numbers are shown, never its file names. Forge references are licensed recordings: keep a
page that plays them private. In Python: `from earshot.gallery import gallery`.

## Install

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/) and ffmpeg / ffprobe on PATH (`make` writes wav
and flac without ffmpeg; ogg, mp3, m4a and webm need it). CPU only; no GPU is used.

```bash
uv tool install "earshot[mcp] @ git+https://github.com/Leonxlnx/earshot"
earshot doctor --fetch     # downloads the two listening models now (~1.1 GB); make, sounds and pack need none
```

Update with `uv tool upgrade earshot`. For the optional Gemini critique (`earshot llm`, needs your own Gemini API
key), install `"earshot[mcp,llm] @ git+https://github.com/Leonxlnx/earshot"` instead.

To work on earshot itself, install from a clone (Python 3.10+):

```bash
git clone https://github.com/Leonxlnx/earshot && cd earshot
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows: .venv/Scripts/python
.venv/bin/python -m pip install -e ".[mcp,llm,dev]"
.venv/bin/python -m earshot doctor --load
```

The first model use downloads two open models (~1.1 GB) into `.cache/hf` inside a clone, `~/.cache/earshot/hf`
otherwise (or `EARSHOT_HOME`), each at a pinned revision with safetensors weights: CLAP `laion/larger_clap_general` @
`16c8cc3` (text-audio embeddings; the Hub's safetensors conversion of its main revision) and the AudioSet tagger
`MIT/ast-finetuned-audioset-10-10-0.4593` @ `f826b80`. `earshot doctor --fetch` downloads them up front. After that
earshot loads them from that cache only and never contacts the Hub; a damaged cache is an error that names the folder,
not a silent new download.

Offline: `EARSHOT_OFFLINE=1`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` all mean the same to earshot: it never
downloads, and a model missing from the cache is an error that says so. earshot never sets or clears these variables.

A cache made before 0.2.1 holds CLAP's main revision `ada0c23`, whose only weights are a pickle (`pytorch_model.bin`,
the same tensors). earshot still loads it from the cache (torch `weights_only`), never downloads it, and `earshot
doctor` says so; `earshot doctor --fetch` downloads the pinned safetensors (776 MB), used from then on.

Settings (environment variables): `EARSHOT_THREADS` (torch CPU threads, default min(4, cores/2); the CLI and the MCP
server hold numpy / scipy's OpenBLAS to it too, unless `OPENBLAS_NUM_THREADS` is set), `EARSHOT_HOME`
(cache root), `EARSHOT_PRECISION` (`bf16` by default, used only on CPUs with native bf16 such as AVX512-BF16 / AMX,
fp32 everywhere else; `fp32` forces fp32), `EARSHOT_OFFLINE=1` (above), `EARSHOT_CLAP` / `EARSHOT_AST` (another
Hub model id, or a local model folder with `config.json` and `model.safetensors`, which loads from disk as it is with
no network and is never downloaded into) and `EARSHOT_CLAP_REVISION` / `EARSHOT_AST_REVISION` (another Hub revision;
safetensors weights only). A custom Hub model downloads its top-level config, tokenizer and safetensors files by name,
never a repo's subfolders or weight variants.
`earshot doctor` reports the revisions and what the cache holds; `--load` also what loads (revision, weights,
precision).

### MCP server for Claude Code

```bash
claude mcp add earshot --scope user -e EARSHOT_THREADS=3 -- earshot mcp     # uv install
claude mcp add earshot --scope user -e EARSHOT_THREADS=3 -- /path/to/earshot/.venv/bin/python -m earshot mcp   # clone
```

Or in any client's `.mcp.json`:

```json
{"mcpServers": {"earshot": {"command": "/path/to/earshot/.venv/bin/python", "args": ["-m", "earshot", "mcp"],
  "env": {"EARSHOT_THREADS": "3"}}}}
```

Tools: `make` (want, out, sound, n, theme, pitch, dur, voice, material, loop, seed, listen), `make_set` (out, set,
sounds, theme, format: one or several such as `"webm,mp3"`, sprite) and `sounds` (no models; their answer is the
summary text alone unless `full=true`; once a listening tool or `earshot mcp --warm` has loaded CLAP in the server,
`make` also ranks by how the designs sound); `measure`, `describe`, `match`, `rank`, `find`, `compare`, `ab`,
`realness`, `timeline`, `sheet`, `critique`, `doctor`, `llm_critique`; and the bank, `sfx_list`, `sfx_render`,
`kit_render`. Models load on the first listening
call and stay warm. On a slow machine start Claude Code with `MCP_TIMEOUT=60000`. The optional `llm_critique` tool
reads `GEMINI_API_KEY` from the environment; never write a literal key into a file.

### Claude Code plugin

The repository is also a Claude Code plugin and its own marketplace. Install the tool first (above): the plugin
starts the `earshot` on your PATH and installs nothing itself. Then, in Claude Code:

```
/plugin marketplace add Leonxlnx/earshot
/plugin install earshot@earshot
```

(or `/plugin install earshot --marketplace Leonxlnx/earshot` in one step). It brings the two skills below, the MCP
server above (`earshot mcp` with `EARSHOT_THREADS=3`; skip `claude mcp add` then) and two commands:
`/earshot:make WORDS` makes one sound into the project, `/earshot:make-set ui|game|arcade|film [theme]` a whole set.

### Python

```python
from earshot.make import make, make_set, sounds
make('a door shutting', 'door.wav')                       # the same result as the CLI, as a dict
make_set('ui', out='public/sounds', fmt='webm,mp3', sprite='sprite')   # + sprite.json
from earshot import sounds as catalogue
y, meta = catalogue.render('coin/bright', seed=2, pitch='B5', bright=0.3)   # stereo float64 at 48 kHz, audited
y, meta = catalogue.render('step/heels', seed=1)          # meta['member']: which recording this take came from
from earshot.pack import pack
pack('assets/arcade', 'assets/arcade.ogg')                                  # a folder of sounds -> one sprite

import earshot
earshot.describe('whoosh.wav')
earshot.find('typing.ogg', 'a single isolated key press')
earshot.timeline('soundtrack.wav', video='film.mp4')
from earshot import api   # everything in one namespace: api.make, api.rank, api.critique, api.realness_many, ...
```

## Agent skills

Two skills ship in `skills/`. Copy or symlink them into `~/.claude/skills/` (or a project's `.claude/skills/`).

- **`skills/earshot`**: how an agent should make sounds with earshot (one `make` per sound, one set per product, how
  to ask and what to check) and how it should listen: which command answers which question, how to read the
  verdicts, when not to trust them.
- **`skills/real-sound`**: a full process for premium, realistic soundtracks for code-made films from real licensed
  recordings: sourcing with licence checks, a music edit, a JSON sound design, a deterministic mixer, measured QA, a
  multi-lens critique, mastering, credits and mux. earshot chooses, trims and verifies the sounds.

## How good is it?

Every number below comes with its method in [docs/VALIDATION.md](docs/VALIDATION.md). What earshot cannot do or
judge is in [docs/LIMITS.md](docs/LIMITS.md).

- **Made sounds, rated by the owner.** The yardstick for the sounds earshot makes is one person's listening, 1 to 5,
  on six listening pages (279 rated files). What passed: chip sounds (4-5), the soft tonal figures (success, notify,
  error: 3-4; level up and coin 4), rain on a window (5), the forest without its dove (5), and takes of learned
  recordings, which he rated exactly as he rated the recordings themselves (24 of 24). What failed: sounds invented from formulas for real
  things (1-2: jumps, landings, punches, explosions, doors, taps and toggles), most music beds (1-2), most drums
  invented from physics (1-2; the drums now played from recorded strokes are not rated yet), bells and glass
  timbres for interface sounds (1), and new sounds sampled from a class model of several recordings (1-2). earshot's two listening models did not predict his ratings; their scores are not quoted as
  quality any more.
- **Understanding requests**: on 220 held-out requests by someone who never saw the catalogue, scored once on the
  v1 catalogue, the first pick is right for 69.1 % and one of the top three for 76.4 %; on the development sets
  92.3 % and 96.4 % (with 68 answers added to their accept lists when the learned sounds arrived).
- **Matching descriptions** (a held-out library of 125 real recordings, nothing tuned on it): in a library search a
  "strong" verdict is the right kind of sound 98% of the time and the exact variant 61% of the time.
- **Finding the moment**: on a recording of 17 single key presses the found trims land on real key presses; in a
  300 s scene every planted event is found as a tight trim.
- **Real vs code-synthesized**: held-out ROC-AUC 0.92 with whole sound families held out. Firm verdicts call 6.7% of
  real clips synthesized and 5.5% of synthesized clips real; a fifth of clips get "uncertain".
- **Listening through a soundtrack**: 93% of the sound effects placed in a 60 s film mix are detected as events (89%
  of the quiet ones under music). A 60-90 s film takes about 1.5 minutes on a laptop CPU.
- **Launch scorecard**: each reference film critiqued against the other six falls in range on 86% of the metrics
  (the no-overfit baseline is about 71%; some rules changed after the first run, so it is not a blind test).

## Limits

`make` draws on a fixed catalogue: it chooses and shapes designs; it does not compose music, speak, or invent a
sound no design covers. Realistic sounds exist only for things someone recorded and put in the public domain; the
invented ones for other real things were rated low. The sounds built last (the interface sounds without instruments,
the film and motion hits, the relearned ambiences, the everyday sounds from other recordings, the animals, people,
street and house sounds) were measured and checked by models but have not been rated by
anyone yet. A request that
describes a mood or paraphrases a sound is understood less well than one that names it. earshot hears through two
open models and a set of measurements. Trustworthy: loudness, timing and sync to picture, "is this the kind of sound
I asked for" on single exposed sounds, picking among candidates, obvious code-synthesized sources. Hints only: tags
of events inside a busy mix, music genre and mood. It cannot judge taste or emotion, provenance or licences, speech
content, or anything above 14 kHz from the models. Details in [docs/LIMITS.md](docs/LIMITS.md).

## History

- **1.0** adds making: `make`, `make --set`, `sounds`, `pack`, the catalogue (learned sounds from public-domain
  recordings, learned ambience beds, figures, drums, chip, notes, the bank), and the Claude Code plugin.
- **0.4** added the sound of music-led launch films to the bank (51 sounds, the `launch` kit preset) and re-made `fit`
  against professional sound effects. [The 0.4 launch film](https://github.com/Leonxlnx/earshot/releases/tag/v0.4.0)
  (35 s) uses bank takes placed with the launch kit; [the earshot site](https://leonxlnx.github.io/earshot-site/)
  plays every bank sound.
- **0.3** added the bank of sound effects stored as code (`sfx`, `kit`).
- **0.2** added forge v2 and `earshot fit`.
- **0.1** was the listening tools, the first forge, the MCP server and the two agent skills.

Every change, with its numbers: [CHANGELOG.md](CHANGELOG.md).

## Repo map

```
earshot/            the package: CLI, Python API, MCP server, CPU model wrappers
  make.py           earshot make / sounds: a request in words -> a design rendered and written; make_set for sets
  brief.py          reading a request: things, jobs, sounds, feelings as axes, voice, material, numbers, key
  pack.py           earshot pack: many sounds in one file with a howler-style map
  sounds/           the catalogue: registry (the @design contract), themes, sets, the lexicon, the design modules
                    (learned, nature, figure / ui / game / brand, drums, motion, action, object, retro, tonal,
                    synth, beds, bank)
  synth/            the engines: resyn (learned sounds), quilt (learned beds), played (real instruments), perform
                    (performances), modal / struck, strings, voice / patches, drums, chip, texture, music; the
                    shared rooms, phrases, voices, finishing, file formats
  data/resyn/       the learned sounds: one model per recording (numbers), index.json, CREDITS.md
  data/textures/    the learned ambience textures: models, index.json, CREDITS.md
  data/drums/       the recorded drum and percussion strokes: models, index.json, CREDITS.md
  data/notes/       the real instruments' single notes (marimba, harp, pizzicato, upright piano): models, index.json,
                    CREDITS.md
  forge/            make sounds in code by listening: DSP building blocks, critic, model-free tone checks, search
    fit/            recording -> code: modal / sines / spectral fitters, quality metrics, recipe codegen
    bank/, kits/    the sound bank (render blocks + fingerprints, no audio) and the kit presets: earshot sfx / kit
  gallery.py        the static listening page
  data/             also: vocabulary + calibration stats, realness model weights, launch profile, the sound atlas
                    (sounds_atlas.npz: CLAP embeddings of every design; numbers only, no media)
skills/earshot/     agent skill: how to make sounds and listen with earshot
skills/real-sound/  agent skill: real-recordings soundtracks (scripts, templates, prompts, references)
docs/               MAKE, VALIDATION, LIMITS, FORGE_V2, FIT, SFX, the motion-SFX study, launch-video playbook,
                    design notes
.claude-plugin/     the Claude Code plugin and its marketplace; commands/: /earshot:make, /earshot:make-set
eval/               evaluation scripts and results per area (see eval/README.md)
tests/              pytest: fast by default, no models; --slow loads the models and renders the long designs
```

No audio or video is stored in this repo. The learned sounds are numbers analysed from public-domain recordings, not
the recordings. The evaluation data (licensed recordings and third-party films) and the reference recordings used to
calibrate the instruments are not redistributed; only derived numbers are committed.

## Licence

MIT for the code (see [LICENSE](LICENSE)). Everything earshot renders (`make`, `sfx`, `kit`, `forge`) belongs to
whoever rendered it: free for any use, commercial too, without credit. The learned sounds, beds, drums and
instrument notes come from recordings in the public domain: BigSoundBank, Kenney, VCSL, VSCO-2 CE, Karoryfer and CC0
uploads on Freesound, all CC0 or an explicit public-domain dedication, each recording credited in
[earshot/data/resyn/CREDITS.md](earshot/data/resyn/CREDITS.md),
[earshot/data/textures/CREDITS.md](earshot/data/textures/CREDITS.md),
[earshot/data/drums/CREDITS.md](earshot/data/drums/CREDITS.md) and
[earshot/data/notes/CREDITS.md](earshot/data/notes/CREDITS.md); CC0 asks for no credit, and the files keep it
anyway. The bank's fitted sounds come from CC0 recordings ([docs/CREDITS_FIT.md](docs/CREDITS_FIT.md)); a fit you
make of your own recording carries that recording's licence. The models are downloaded from their authors under
their own licences: CLAP (`laion/larger_clap_general`) Apache-2.0, AST (`MIT/ast-finetuned-audioset-10-10-0.4593`)
BSD-3-Clause. `skills/real-sound/scripts/master.py` is vendored from the Cinetic skill by the same author.
