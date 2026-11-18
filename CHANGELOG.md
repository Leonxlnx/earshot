# Changelog

## 1.0.0 - unreleased

earshot makes sounds. An agent asks for a sound in words and earshot renders a new take of a design from its
catalogue, with no model and no network: interface sounds, everyday foley, ambience beds, game sounds, chip sound,
drums, motion-design effects and three short music loops, one at a time or as a family with a sprite for the web.
Realistic things (everyday sounds, the interface sounds, the motion effects, ambiences, the acoustic drums) are
learned from recordings in the public domain and kept as numbers, not audio, and computed anew at every request;
tonal figures, chip sound and the music loops are written in code. The yardstick for what was kept is the owner's listening, 1 to 5 on six pages, not earshot's own
listening models, which did not predict his ratings ([docs/VALIDATION.md](docs/VALIDATION.md#the-owners-listening)).
Fixes from the community, thanks to @Bortlesboat, @anascherif and @liugedragon. No new required dependencies.

### Added

- Learned from public-domain recordings: a crowd booing, a cat hissing, a lion roaring, an elephant, a bee, a rocket
  launch, a record scratch, a squeaky toy, an elevator ding, a toaster, a blender, an egg cracked, a whisk, and walks
  in mud, through puddles and in dry leaves; a pan set on a stove and a fridge door are no longer made.
- Rain on wood, on a tin roof and on an umbrella are learned beds from public-domain recordings (`rain/wood`,
  `rain/tin_roof`, `rain/umbrella`).
- A plain request for an interface job gets the design the `ui` set plays ("a success sound": `success/shutter`,
  "open a menu": `open/drawer`, "delete": `delete/snip`); `open/air`, `close/air`, `error/thud`, `receive/pop`,
  `type/key`, `next/swipe`, `back/page`, `back/swipe`, `complete/stamp`, `complete/latch`, `delete/crumple`,
  `delete/swish` and `hover/pad` are removed.
- A thing of the world is answered by its own recording, never by the interface sound made of it: "a door closing"
  is `door/shut`, not `close/door`; the interface designs no longer weigh on how rare a word is.
- A head whose sound depends on its owner ("a fax machine", "a seagull cry", "an elevator ding") says it is only
  the nearest when the design picked answers neither the owner nor the head.
- "ten seconds" in words is a length; "twice" is no unknown thing; "in a bathroom", "in a stairwell" set the room.
- A request that asks for a change as the sound plays ("getting louder"), for a sound heard through something
  ("behind a closed door", "underwater"), or for a length the design does not keep, says so in the result.
- **`earshot make "WORDS" -o FILE`** (`earshot/make.py`, MCP `make`, `api.make`): the request is read, the catalogue
  ranked, the best design rendered with the request's controls, audited (clicks, cut tails, clipping, DC, comb ring,
  rising glides, stacked attacks; a failed take is re-rendered on up to three more seeds) and written as wav, flac,
  ogg, mp3, m4a or webm. The answer is a few lines: the design and the words that chose it (as written), the sounds
  or jobs of the request it does not cover (`not in this sound`), up to three alternatives (`also`, always, with a
  note when the pick is a guess, to be taken with `sound=<id>`), controls `not used by this sound` (also a length a
  design could not keep) and words `not understood`; a loop written as mp3 or m4a is warned about. `--n` takes,
  `--seed`, `--sound ID`, `--theme`, `--pitch`, `--dur`, `--voice`, `--material`, the eight axes, `--lufs`,
  `--mono`, `--no-loop`. A one-shot takes about a second on one thread.
- Every take of `make` says where things happen in the file: `peak_s` (the loudest moment, also "peak at" on the
  plain line), `onset_s` (where it starts to be heard) and, for a row of events, `events_s` (each event's start).
- A row of events no hand plays (ticks, clicks, blips, a counter) follows as close as its own length lets it, so
  "a counter ticking up, 1.5 seconds" with 12 events is 1.5 s long; knocks and steps keep their 0.12 s limit.
- A row with a count and a length too short for the usual ring ("a soft thump", 3 events, 0.6 s) keeps the length,
  and the last event is faded out to end with the file instead of being cut.
- A gap said with a count ("three soft thumps, 200 ms apart") is the time from one event to the next, and the row
  is as long as that makes it.
- A request that does not read as English is answered as a guess that says so ("write it in English"); a job word
  it still holds ("Erfolgston für die App") keeps its pick.
- Every request gets a sound: when no word of it finds a design ("a frobnicating zorp"), the answer is a nearest
  one that says it is only that, with alternatives, where it was an error before.
- A request whose thing, or what the thing does, is a word earshot does not know ("a mouse squeak", "a curtain
  being pulled") is answered "earshot makes no X; ID is only the nearest sound it has"; the nearest is chosen by
  the kind of sound the word names (a roar is a low rumble, a hiss a spray of air), then by the known words.
- **Recipes** (`earshot/recipe.py`, `earshot make --recipe FILE`, `earshot recipe --example`, MCP `make` with
  `recipe`, a set's `{"name", "recipe"}`): a sound no design is can be built from the catalogue's parts as a small
  JSON form (layers of designs moved in pitch, reversed, cut, filtered, repeated as events, with noise for air and
  rumble), rendered and audited like every sound; a cookbook of such recipes (`earshot/data/recipes`) answers the
  requests that name its things ("a dragon roar"), and the answer says what it is built from.
- What follows "on", "in", "against" or "into" after the sound is its surface, not a second sound: "rain on wood"
  is no forest, "a ball bouncing on wood" is the ball, "water dripping on metal" no sizzling pan.
- A surface the sound has no design on picks the nearest one and says so ("rain on a tarp": `no rain on a tarp:
  rain/tent is the nearest surface`, with the other surfaces as alternatives).
- A request in German is read through a table of about 250 sound words: "Regen auf Holz" as "rain on wood",
  "Schritte auf Kies" as "footsteps on gravel"; a word the table lacks still gets "write it in English".
- "a scratch" is no injury ("scratched" still is): "a vinyl scratch" no longer plays a hurt sound.
- `knock/double` is two strikes by design: the audit no longer marks its two attacks as stacked layers.
- A length within 0.1 s of the one asked prints no note; a length word that is not followed ("a short soft whoosh")
  says to give the length in seconds.
- `earshot.api` raises the plain missing-extra sentence (MissingExtra) instead of a bare ModuleNotFoundError.
- **Learned sounds** (`earshot/synth/resyn.py`, `earshot/sounds/learned.py`, `earshot/data/resyn/`): 249 designs of
  everyday, interface, game and film sounds learned from 604 recorded events in the public domain (the sources of all
  learned material, beds, drums and instrument notes included: BigSoundBank, Kenney, VCSL, VSCO-2 CE, Karoryfer and
  CC0 uploads on Freesound, all CC0 or an explicit public-domain dedication; credits in the `CREDITS.md` of
  `earshot/data/resyn/`, `textures/`, `drums/` and `notes/`). Each recording is analysed at the
  resolution of hearing (56 bands, 2 ms envelopes, 0.25 ms around onsets, measured carriers, the stereo image) and
  stored as numbers: 13.81 MB in all, no audio. Every render draws fresh noise, so every take is new, but it is a
  take of that recording, not a sound invented from nothing. A design is one thing and its recordings are its
  variations: seeds walk through them (three seeds of `step/pebbles`, three real footsteps), `meta['member']` names
  the recording. The controls are transforms of the model: size 0.7-1.4, a tilt of up to 2 dB per octave, strikes
  softer or harder, ring 0.5-2 times, energy, length within 0.6-1.8 times, pitch on tonal recordings. A learned
  design answers only a request that names its thing and wins a tie with an invented one. Why: the owner gave a
  recording and its renders the same rating on all 24 rows of his test, and rated sounds invented from formulas for
  real things 1-2 of 5.
- Animals, people without words, street, house, desk and sport sounds are recordings now (44 designs: a dog barking).
- More everyday things are recordings now (26 designs: a scream, a sneeze, a sheep, a duck, a glass breaking).
- Kitchen things are recordings now (8 designs: a knife chop, a pot boiling, a gas stove lit, a fridge door shut).
- **Learned ambience beds** (`earshot/synth/quilt.py`, `earshot/data/textures/`, `earshot/sounds/nature.py`): 13
  beds from 12 BigSoundBank textures (5.40 MB): `fire/camp`, `waves/sand`, `waves/pebbles`, `waves/rocks`,
  `night/crickets`, `stream/brook`, `room_tone/quiet`, `city/street`, `city/distant`, `cafe/busy`,
  `underwater/deep`, `rain/street`, `wind/trees`. Takes of the texture's models are cut into pieces and stitched
  where the texture is in a similar state, so a bed plays for any length without a loop point. `rain/window`,
  `forest/morning` and the other scenes are made from rules.
- **Interface sounds without instruments** (`earshot/sounds/interface.py`): the seventeen roles of `make --set ui`
  are things a hand does, built from learned recordings: tap a mechanical key, hover a fingertip on wood, the
  toggles that key going down and coming up, type the same key pressed softly, next and back a page flicked
  forward and turned back darker, open and close a wooden drawer, send a stick through the air, receive a cork,
  notify two light knocks, success a camera shutter, complete a bigger, slower shutter, warning two firm knocks,
  error two heavier knocks that fall, delete a cut of scissors; one more candidate per role by name, and a like
  (`like/pop`, a small bright pop of the lips over a fingertip) outside the set. A request for an
  interface job gets the same; the tonal figures answer only when the words ask for music or a theme is named.
- **The best recording plays first** (`earshot/sounds/learned.py`, `walk` in `earshot/data/resyn/index.json`): a
  learned design's seeds walk its recordings from the one with the best listening score down; a recording scored 1
  of 5 is left out of the walk; a take that fails its audit is rendered again from the same recording.
- **A set is as full as its sounds made alone** (`earshot/make.py` `balance`): every one-shot of a set is made as a
  single `make` makes it (true peak -1 dBFS) and only turned down, so that by its loudest 50 ms none lies more than
  2.5 dB over the quietest; a `quiet` name (`hover` in `ui`) sits 0.5 dB under it. Loops, beds, pads and music, and
  every sound of a spec that gives `lufs`, are levelled by kind as before (-16 LUFS and each kind's weight under it).
- **Tonal figures on real instruments** (`earshot/synth/played.py`, `earshot/data/notes/`): a marimba played with
  yarn mallets (14 notes), a harp (24), pizzicato strings (45: a cello section up to B4, a solo violin above) and an
  upright piano played softly (14), 97 single notes from VSCO-2 CE and VCSL (CC0), kept as numbers (0.83 MB,
  credits in `earshot/data/notes/CREDITS.md`) and played as new takes. The voices `marimba`, `harp`, `pizzicato` and
  `felt_piano` play them (`struck.note` and `strings.note` still play the models). The marimba is the default voice
  of figures and of the `soft` and `wood` themes. A real voice lets inner notes ring under the next one, rings the
  last one out and plays a figure whose top note is above C6 an octave lower. The interface figures were written
  for the marimba (`ui.py`: between A3 and G5).
- **Performances** (`earshot/synth/perform.py`, `learned.perform`, `make(events=)`, `--events`, MCP `make`
  `events`): a thing a person does several times in a row is played as the action, from its single recordings by a
  pattern (walk, knock, type, scatter, shake, pair) or from one of 23 recordings of the whole action (walks on
  gravel, wood, tiles and in heels, a knock on a door and on a table, toasts, keys, coin handfuls, a dog barking, a
  heartbeat, a car horn, a hammer on a nail); 14 learned things have such recordings. The words decide: a plural, an action word, a count or a length
  asks for the action ("footsteps on gravel", "knocking", "three knocks", "5 seconds of typing"), a bare singular
  or "a single" for one event ("a footstep on gravel", "a single knock on a door"), a singular that names the
  action ("a knock on the door", "a toast") for the action. `events` overrides the words; the answer says what was
  played. New thing: `keys/bunch`.
- **Short sounds come forward** (`earshot/synth/master.py` `finish(lift=)`, `registry.LIFT`): a one-shot under 1 s
  whose one spike holds its body far down is raised by up to 14 dB and the spike folded over, not limited; an
  ambience bed held down by a few peaks by up to 8 dB (`registry.SPARSE`). The owner had heard many short sounds as
  barely there. Loudest levels by kind: -14 LUFS momentary max, pads and textures -18, ambience beds -22, music
  loops -16.
- **Drums** (`earshot/sounds/drums.py`, `earshot/data/drums/`, `earshot/synth/drums.py`): 40 designs in 21
  families. The acoustic kit and the hand percussion (33: kicks, snares, rims, toms, hi-hats, crashes, a ride,
  claps, shaker, tambourine, congas, bongos, cajon, darbuka, frame drum, clave, cabasa, slapstick, a snare roll and a
  shaker roll) are takes of 130 recorded strokes in the public domain (VCSL, Karoryfer and, for the hand claps, one
  person clapping on Freesound; 1.31 MB of numbers): the seed walks through a design's strokes and `energy` picks
  softer or harder ones. Still invented: `kick/electro`,
  `snare/electro`, `hat/electro`, `clap/electro`, `snare/brush`, `snap/finger`, `sticks/count`. New:
  `conga/muted`, `bongo/low`, `darbuka/hit`, `frame_drum/deep`, `cabasa/hit`, `slapstick/crack`; removed:
  `ride/bell`, `cymbal/swell`, `conga/slap`, `cajon/slap`. The owner rated the drums invented from physics 1-2
  ("big problems, does not sound good"); the recorded ones are not rated yet.
- **Film and motion sounds** (`earshot/sounds/motion.py`, `earshot/synth/design.py`): 46 designs. 32 are built
  the way a sound designer layers them, from 25 learned things of the store (recordings in the public domain: heavy
  drops, car doors and boots slammed, explosions, swells of air, breaths and flames, drum and timpani rolls, low
  brass, deep, airy and fast whooshes, passes, thunder, orchestral drums, tam-tams and cymbals) and the drum
  store's strokes, reversed, stretched, layered, faded and glued by one room, following measurements of
  professional motion videos (short, dry, near mono, low or very bright). 14 small ones stay invented (pops,
  tocks, a blip, a glint, a shimmer sweep, a sparkle, key ticks, a sliding panel, a soft knock, the soft sting).
  Removed: `whoosh/tonal`, `sting/tech` and the invented `whoosh/airy` (the id is now the learned airy whoosh).
  The owner rated the earlier film sounds 1-3 of 5 (the fast whoosh 4); the rebuilt ones are not rated yet. A
  film request for a cymbal gets the film's crash cymbal, a request with a word of the drum kit the kit's.
- **Invented and learned twins** (`lexicon.TWINS`): 33 invented designs of real things have a learned design of
  the same thing (`knock/door` and `door/knock`, `coins/handful` and `coins/floor`, `paper/tear` and `paper/rip`
  ...); a request that names the thing gets the learned one, the invented one answers to its id. A single coin
  ("a coin dropped on the table") keeps the invented `coin/drop`; coins in the plural get the recorded handful.
  An arrow is the recorded arrow hitting its target, not a gunshot; "a heart" is a heartbeat and a heart tapped
  in an app a like; "vibrate" or "haptic" is the recorded phone vibrating; the click of a cursor or a mouse is
  the recorded mouse button.
- **Three music loops** (`earshot/synth/music.py`, `earshot/sounds/beds.py`): `bed/warm`, `bed/playful`, `bed/8bit`,
  a whole number of bars in a key, seamless or once. earshot is a sound-effects tool; nine other styles were
  written and removed after the owner's listening.
- **Voices, keys and lengths from the words**: a material without a voice picks the voice made of it (wood a
  marimba, glass a glockenspiel, metal a vibraphone, soft a felt piano, digital a glass pluck, retro a square lead);
  an instrument named in the request plays the figure ("a notification on a harp", "a success sound with
  pizzicato strings", "an error sound on a soft piano");
  a voice asked for by name is always played; a minor key in the request or theme turns a figure's major thirds,
  sixths and sevenths into minor ones; a length shorter than a figure rings keeps its spacing and damps it by the
  hand so it ends there (a figure that does not fit is played faster, down to 70 ms between notes, and the result
  says the shortest it can be), a longer one changes nothing; "under 300 ms" shortens only a design that is longer.
- **The sound atlas** (`earshot/sounds/atlas.py`, `earshot/data/sounds_atlas.npz`, numbers only): a CLAP embedding of
  every design's default render and of its descriptions. `--listen` (MCP `listen`) blends how each design sounds into
  the ranking and reports what the written sound is heard as; the MCP server's `make` blends it in without being
  asked once CLAP is loaded. Rebuilt with `python -m earshot.sounds.atlas build --work DIR`.
- **Understanding requests, measured**: on 220 held-out requests written by someone who never saw the catalogue,
  scored once on the v1 catalogue, the first pick is right for 69.1 % and one of the top three for 76.4 %; 92.3 %
  and 96.4 % on the two development sets, whose accept lists gained 68 answers when the learned sounds arrived
  ([docs/VALIDATION.md](docs/VALIDATION.md#understanding-requests)).
- **Reading requests** (`earshot/brief.py`, `earshot/sounds/lexicon.py`): things, jobs and sounds to canonical
  words, the head of a request and its parts and context, feelings as offsets on eight shared axes (bright, hard,
  ring, size, space, width, energy, motion) with intensifiers, negation ("not harsh", "not an error"), "the opposite
  of", comparisons, voices named as the instrument or as the player, materials, "8-bit" as a hard filter for chip
  sound, lengths and ceilings, pitches, keys, counts of takes and notes, loop or one-shot, domains. No model.
- The app, site, store or video a sound is for names no sound once the request names one: "in my running app, a
  success sound" is a success sound, not footsteps, and "the Thunder app needs a notification sound" no thunder.
- **The catalogue** (`earshot/sounds/`: `registry.py` with the `@design` contract, `earshot sounds`, MCP `sounds`):
  the learned sounds and beds above; 45 figures for interfaces, games and brands that any voice can play (`ui.py`,
  `game.py`, `brand.py` on `figure.py`) and 3 material ticks; 51 single notes of 36 instruments and 15 patches
  (`tonal.py`); 14 synthesizer sounds (`synth.py`); 24 chip sounds (`retro.py`; the power-up's held top note lands between F5
  and C6); drums; motion-design sounds; 47
  invented action sounds (`action.py`) and 36 object sounds (`object.py`); 31 ambience scenes (`nature.py`); three
  music loops. The bank's 209 named members join it (`bank.py`) and rank under the newer designs.
- **Themes and sets** (`earshot/sounds/theme.py`, `sets.py`): nine theme presets (soft, glass, wood, felt, tine,
  metal, electric, digital, retro) give a family one voice, key, room and axis offsets; four ready-made sets (`ui`
  17, `game` 22, `arcade` 16, `film` 17; `ui` and `film` play real things, no note). `earshot make --set NAME` / `--spec FILE` and MCP `make_set` make a
  whole set in one call, its one-shots as loud as made alone and within 2.5 dB of each other, its loops and beds
  levelled by kind (the loudest at -16 LUFS momentary max, a bed 14 LU under), with `manifest.json`. Several formats at once (`--format webm,mp3`, a list in a spec or the MCP tool):
  every sound in each, the manifest's `files` and `formats`, the sprite in each format with its map's `src` in that
  order. For the web, `webm,mp3`: Ogg Vorbis does not play in Safari before 18.4. A loop in such a set is written
  as flac in place of mp3 / m4a (their padding leaves a gap at the seam); a second name that gets the same sound
  as an earlier one is flagged, with the nearest other design, and a second name that plays another recording of
  the same learned design (toggle_off after toggle_on) is named as such, without a warning. A guess's alternatives
  carry a few words each.
- **`earshot pack`** (`earshot/pack.py`, `--sprite` on a set): many sounds in one file with a map in the format
  howler.js reads; loops are marked and get a longer gap. On a set's folder it packs exactly the manifest's sounds
  and leaves out an earlier sprite.
- **The engines** (`earshot/synth/`): learned sounds (`resyn.py`), stitched beds (`quilt.py`), real instruments
  note by note (`played.py`) and performances (`perform.py`); struck
  instruments from modes, damping laws and a Hertz contact (`modal.py`, `struck.py`); strings, pianos and electric
  pianos (`strings.py`); a synthesizer with band-limited oscillators, a ladder filter, FM operators, chorus and echo,
  and 28 patches (`voice.py`, `patches.py`); drums (`drums.py`); band-limited chip sound (`chip.py`); ambience from
  event statistics and band-wise noise (`texture.py`); music loops (`music.py`); every voice behind one call
  (`voices.py`), figures played like a person (`phrase.py`), seven shared rooms (`space.py`) and the finish and
  file formats (`master.py`).
- **Calibration**: the pianos and mallet instruments were compared note by note with freely usable recordings
  (University of Iowa Musical Instrument Samples; VSCO-2 Community Edition, CC0) and the laws of the models set from
  several notes; no recording and no per-recording data is in the repository.
- **Docs**: [docs/MAKE.md](docs/MAKE.md) (the reference of `make`, learned sounds, performances, sets and the spec,
  `sounds`,
  `pack`, the request language, axes, themes, the catalogue family by family, the voices, recipes for a web page, a
  game and a film, how to add a design); VALIDATION section 8 (the owner's listening, the learned sounds measured,
  understanding requests); LIMITS "Making sounds"; the earshot skill opens with a self-contained making part.
- **Claude Code plugin** (`.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `commands/`): the
  repository is a plugin and its own marketplace (`/plugin marketplace add Leonxlnx/earshot`, `/plugin install
  earshot@earshot`): the two skills, the MCP server from the installed `earshot`, `/earshot:make` and
  `/earshot:make-set`.
- **Tests** for the catalogue contract, the learned sounds and their engine, the stitched beds, drums, music loops,
  the rooms and finishing, the voices, `make` / sets / `pack` and the atlas (`tests/test_sounds_catalogue.py`,
  `test_sounds_learned.py`, `test_synth_resyn.py`, `test_synth_quilt.py`, `test_synth_played.py`,
  `test_synth_perform.py`, `test_sounds_nature.py`,
  `test_synth_drums.py`, `test_synth_music.py`, `test_make.py`, `test_sounds_atlas.py` ...); the long renders and the
  tests that load CLAP run with `--slow`.
- **CI** (#32; thanks, @liugedragon): a Linux job runs the fast tests without the models
  (`.github/workflows/fast-tests.yml`, `docs/TESTING.md`).
- **Motion effects as motion films use them**, all from recordings: a 50 ms low thump for something appearing
  (`thump/appear`), short dry hits of a heavy bag dropped on a real explosion's low end with a swell of air into
  them under a title (`slam/title`) and a logo (`boom/short`, `boom/deep`, `sting/reveal`; the car-door build by
  name: `slam/door`, `boom/door`), the hit after a stop (`hit/drop`;
  "a drop" in a film), a big far explosion that rolls (`explosion/far`), a bright click (`tick/soft`),
  whooshes and risers of plain air that keep the length asked for, a glitch cut from real buzzes and clicks, a shine
  of air (`shine/air`), a tape stop of a real roll running down (`stop/tape`); a motion job gets no instrument note
  or synthesised tone unless its words ask for music (`brief.FILM_MUSIC`).

### Changed

- **A light install**: the base install is numpy, scipy, soundfile, soxr and numba, all that making sounds needs;
  torch, transformers and librosa moved to the `listen` extra, librosa also to `measure`, matplotlib and pillow to
  `images`, and `all` brings every extra.
- pyloudnorm and scikit-learn are no longer dependencies: nothing in earshot imports them.
- A command, MCP tool or Python function that needs a missing extra stops with one sentence that names the package
  and the `pip install` command (`earshot.extras`); importing earshot and making sounds never load torch or librosa.
- `earshot doctor` reports which extras are installed and does not count a missing extra as a problem.
- `make` refuses a length over 120 s, a length of 0 or less, a count of takes outside 1 to 12, events outside 1 to 64,
  a seed that is not a whole number and an unknown option, each with one plain sentence (a seed given as "3" is 3).
- A request of only spaces is treated as an empty one.
- `make_set` with an unknown set name names the sets that exist (it looked for a file of that name), and checks each
  sound's length, takes and seed in the same way as `make`.
- A sound of a set takes `events` and `loop` as `make` does, and a thing done in a row ("three knocks, 2 seconds",
  footsteps) is played as the action in a set too, with a `played:` line.
- `make` no longer reports a material it read from another word ("a chair": wood) as not followed, nor "short" when
  the length asked in the same request was kept, and an interface press gets no hint about events.
- **The README** puts making first and listening second; the 0.4 and 0.3 notes moved to a short history.
- **Package metadata**: version 1.0.0, development status Beta, a description and keywords for both halves; the
  learned models, textures and their credits ship as package data.
- **Picture-only timelines and frame differences use far less memory** (#31; thanks, @liugedragon).
- **Licence of rendered audio** (answers #21): the code is MIT; everything earshot renders (`make`, `sfx`, `kit`,
  `forge`) belongs to whoever rendered it: free for any use, commercial too, without credit. The README says so.
  The learned material comes from CC0 recordings, credited in `earshot/data/resyn/CREDITS.md`,
  `earshot/data/textures/CREDITS.md`, `earshot/data/drums/CREDITS.md` and `earshot/data/notes/CREDITS.md`.

The changes of the last two days, in the order they were built:

- A design of the same name stands in for a request only when its family is the job asked for ("a deep impact" no
  longer reaches `kick/deep`).
- Seed 0 of a learned design plays its best recording; a set's one-shots are made as loud as alone and only balanced
  by gain.
- Tonal figures are played on real instruments learned note by note: marimba, harp, pizzicato strings, an upright
  piano.
- Everyday things answer with the action a person performs (a walk, knocking, typing) when the words ask for it;
  `events` sets it.
- The fast whoosh is the recorded stick swoosh as it is; the interface figures sit between A3 and G5; the chip
  power-up starts a fifth lower.
- A typing run on a keyboard with few recordings borrows the other keyboards' keys and keeps the space bar for the
  pauses.
- An instrument named after a job chooses the voice ("a success sound on a soft piano"); `make` over MCP takes
  `events`.
- A note is levelled as it is played, so a run of notes has no jump in level between neighbours.
- A tonal figure asked shorter than it rings keeps its spacing and is damped by the hand to end at that length.
- Three pizzicato C5 takes without a steady pitch left the notes store; every remaining note plays within 5 cents.
- The source archive carries the changelog, the docs, the plugin's files and the tests' own files; numba is declared.
- A request for a thing no design is answers "earshot makes no X"; "five coin pickups" is five takes.
- An asked length holds for ambiences: shorter than the shortest seamless loop it is a cut with fades that says so.
- `not used by this sound` is gone: only a control asked for in words that the sound cannot follow is named, in plain
  words.
- A counted action in a row ("four knocks on the table", "three claps", "a countdown 3 2 1") is one file for any
  one-shot.
- A design named inside its family wins over the family's default ("tape stop").
- A learned take says "recording 2 of 3", never the title of its source recording; titles stay in the indexes and
  credits.
- A length asked with a count is the file's length, the events spread over it at a pace a hand keeps.
- The action form of a row word is the action: "clapping", "clicking", "beeping" are a row; "clapping, 4 seconds"
  claps for 4 s.
- Scattered and shaken actions keep an asked length; a recording of a whole action plays its busiest stretch of that
  length.
- A recording that passes its audit plays first; the last event of a walk or a row rings out before the length asked.
- The `ui` set and plain interface requests are normal sounds, things a hand does (`sounds/interface.py`); no
  instrument.
- Instruments answer an interface request only when its words ask for music (an instrument, a chime, a jingle, a
  melody) or a theme is named.
- The skill says which moments of a motion video get which sound, from measurements of 39 professional motion videos.
- `fire/camp` keeps every crackle: learned with an onset at each, its two sides two takes that share them.
- `waves/sand` is learned from a recording at the water's edge and keeps its foam; waves are joined where they were
  cut.
- `wind/trees` is learned from wind through a field of dry leaves ("Wind in a cornfield", BigSoundBank, CC0).
- `rain/window` drops the gutter's ringing drip, the one tonal, regular sound in it.
- Motion sounds are the short, dry, low things motion videos use: a 50 ms thump (`thump/appear`), a bright tick
  (`tick/soft`), whooshes of air.
- A motion request gets no design made of notes unless its words ask for music or for a game.
- A whoosh holds an asked length, a riser is real air played backwards to a clean cut, a tape stop a real roll run
  down to a halt (`stop/tape`).
- A fire whoosh plays a real burst of flame first.
- The title hit, the logo hit, the booms and the hit after a stop are heavy real things slammed, no drum stroke left.
- "An explosion far away" plays a long rolling blast first (`explosion/far`).
- In a film or video "a drop" is the hit after the silence (`hit/drop`); "pops in on screen" is the thump; "a soft
  swish" is real air.
- The default title and logo hits are a heavy bag dropped on a real explosion's low end with real air into them;
  `slam/door` and `boom/door` keep a car door's build.
- Knocks on a table are real knuckles on a wooden table (`knock/table`).
- A single knock on a door is one whole knuckle knock with the door's body (`door/knock` plays it first).
- Footsteps on tiles are hard shoes on a tiled floor, footsteps on concrete hard shoes on stone.
- A walk alternates the two single steps of a surface that are most alike, so it does not limp.
- A walk on gravel is heard from standing height.
- A light switch, a punch and a book slammed shut play new close, dry recordings first.
- A tap outside an interface is a fingertip on a table (`tap/fingertip`); "a tap" stays a key.
- A bunch of keys plays a recording of metal keys first.
- The second drum roll recording plays last: the engine clicks at one of its onsets.
- A hand clap is a single clap of one person clapping (`clap/hands`, from "Solo Clap" on Freesound, CC0); "three
  claps" and "clapping" play it.
- The reader understands more requests that describe the moment instead of naming the job ("a title lands", "the
  camera pushes in", "a swoosh as the message goes out", "grab a card to drag it", "the logo locks up", "the numbers
  count up"); a thud for a film or a title is never a drum of the kit unless the words name one.
- Requests written for a real job are read as what they name: a low thump is a thump and no sub drop, a card or a
  panel that settles lands softly, "careful" before a thing says how it is done, one enemy defeated is a blow and no
  fanfare, what follows "each time" is where the sound plays, toast that is bitten is no notification, and the
  seconds asked of footsteps no longer choose the surface. The skill and `docs/MAKE.md` say when a page, a game, a
  film or a clip needs a sound and when none, what a set does with your own list, how long and how loud sounds come
  out and where a file's loudest moment lies.
- The reader reaches the newest recordings by their plain requests ("a tap running", "a frog croaking", "glass
  breaking", a shooter's reload; a bee is said to be missing, with the fly as the nearest), reads "twice" as two
  events of one row, "200 ms apart" as the gap and hours as a length, gives a request that only asks for a cue the
  notification's knocks and a purchase in an app the success click, and no longer answers a request in another
  language from a look-alike word (German "die").

### Fixed

- **`sheet` names the PNG by its real path** (#22; thanks, @Bortlesboat): the summary printed it ASCII-escaped.
- **Cue, kit and params JSON files with a byte order mark or in UTF-16 load** (#24, closes #23; thanks,
  @anascherif). Windows PowerShell 5.1 writes UTF-8 with a byte order mark (`Out-File -Encoding utf8`) and UTF-16
  (`>`); `timeline --cues`, `kit render / check` and `forge --render --params` failed on such files.
- **A damaged scene-cut cache is scanned again** and written atomically (#26; thanks, @liugedragon).
- **`timeline --fresh` bypasses the decode and frame caches** too: it decodes the audio and scans the picture again
  (#27; thanks, @liugedragon).
- **Digital silence is counted across all channels** (#29; thanks, @liugedragon): `digital_silence_frac` is the share
  of frames where every channel is under 1e-6; one silent channel or phase cancellation no longer counts.
- **An empty picture-only range is an error** (#30; thanks, @liugedragon), not an empty timeline.

## 0.4.1 - 2026-11-06

Fixes from the community, thanks to @SmitMachhi, @yinbowen-dc, @kyexyz and @lunar-me, and a measurement of how
earshot hears real sound effects. On valid input every command gives the same results as 0.4.0. No new dependencies.

### Added

- **Install and upgrade with one uv command** (#5, #11; thanks, @SmitMachhi): `uv tool install "earshot[mcp] @
  git+https://github.com/Leonxlnx/earshot"`, then `uv tool upgrade earshot`. Under uv, torch comes from PyTorch's
  CPU-only wheel index (`[tool.uv.sources]` in `pyproject.toml`), so a Linux install skips the CUDA libraries earshot
  never uses; pip ignores that table and installs as before. The README puts this install first, says where such an
  install keeps its models (`~/.cache/earshot/hf`) and gives the short `earshot mcp` line for Claude Code. The clone
  install stays for working on earshot itself.

### Fixed

- **A NaN or Inf sample is one clear error** (#12, #13; thanks, @SmitMachhi). One non-finite sample crashed most
  commands with an error that did not name the file (`cannot convert float NaN to integer`, a librosa
  `ParameterError`, an `IndexError` in `sfx check`), and `match`, `rank` and `realness` exited 0 with a wrong answer
  (`z +nan`, a whoosh called silent). `audio.load()` now refuses such a file, also when it comes from the decode
  cache, with one line that names the file, the count and the time of the first bad sample, so the CLI, the MCP
  server and the Python API stop alike; `rank` skips it like any unreadable file.
- **Media globs match files only** (#16, #17; thanks, @yinbowen-dc). A directory whose name ends in a media extension
  (`folder.wav`) counted as an unreadable file when a glob matched it, so `rank 'a click' '*.wav'` with no real files
  exited 0. A glob that matches only directories is now "nothing playable" (exit 1); a directory given by name still
  expands to the media files inside it.
- **`video.cuts(..., cache=False)` scans again** (#14; thanks, @yinbowen-dc): the default frame-difference method
  reused its saved differences anyway. A forced scan still refreshes the cache.
- **A failed scene scan is an error, not a film without cuts** (#15; thanks, @yinbowen-dc). `video.cuts(method='scene')`
  parsed a failed ffmpeg run as no cuts and cached that empty list. It now raises with the file and ffmpeg's reason
  and caches only a scan that ran; the cache key moved to v2, so older scene scans run once more.
- **BLAS threads follow `EARSHOT_THREADS`** (#8; thanks, @kyexyz). numpy and scipy each bundle an OpenBLAS that
  started a worker thread per logical CPU and, on Windows, committed a buffer for each when first imported: with 12
  logical CPUs an idle MCP server reserved 411 MB and ran 21 threads (154 MB and 13 threads now, at the default limit
  of 4), and the gap grew once a tool loaded scipy. Memory in use and speed are unchanged (`measure` median 600 ms
  either way). The CLI and the MCP server now set `OPENBLAS_NUM_THREADS` to `--threads` / `EARSHOT_THREADS` /
  min(4, cores/2) before numpy loads, unless it is already set. The Python API leaves the environment alone.
- **The interface tests' fake Gemini clock stays inside `llm`** (#18, #19; thanks, @yinbowen-dc). The fixture replaced
  `time.sleep` for the whole process, so a subprocess timeout could log fake sleeps before the first request and fail
  `test_llm_inline_round_trip` / `test_llm_files_api_above_inline_limit` at random. Tests only.

### Docs

- The README opens with the 0.4 launch film and links the earshot site, where every bank sound plays (#10).
- Two comma splices, in `docs/VALIDATION.md` and `docs/motion-sfx-techniques.md` (#6, #7; thanks, @lunar-me).
- **Real sound effects, measured** (`eval/semantics/real_effects.py`, `docs/VALIDATION.md` §1): on 195 real licensed
  one-shots filed by the film job they play, a sound asked by its own title passes 168 times (124 strong), while job
  words are not heard ("a soft rounded pop, something appearing" passed none of 9 real pops). Describing the job's
  sound two or three ways and keeping the best reading passes 176 instead of 152 at a slightly better AUC (0.922 vs
  0.912); the skill says so. Spaced padding of short sounds lifts the title matches (no match 27 -> 20) at the same
  AUC; it is not shipped, because the verdict bands were set with repeatpad.

## 0.4.0 - 2026-10-22

Sound for music-led launch films: 51 new recipe-made sounds in the bank (seven new families), a `launch` kit preset
mixed against the music bed instead of at fixed levels, and the fit engine re-made and measured with a replication
benchmark on a private set of licensed reference clips (not in the repo: no audio, no clip name and no per-clip
model ships; aggregate measurements of them (medians, IQRs), CLAP cosines to them and parameters tuned toward them
do). Tuned on a fixed 78-clip dev split; the 38 held-out clips were opened once, at the end of the first round, and a
fresh 28-clip held-out set, frozen before any further tuning, at the end of round C2. No new dependencies.

### Added

- **Bank 0.4** (`earshot/forge/bank/forge04.json`, `forge04r1`; `bank.VERSION` is now
  `forge_v2r6+forge04r1+fit1+pluck1`, 252 sounds): 51 recipe-made sounds, built from the recipe palettes by
  `python -m earshot.forge.bank build --forge04`. New categories: `cut_hit` (sub, kick, punch, rumble, air),
  `reverse_swell` (cymbal, ring, dark, short, breath), `ui_pop` (soft, step, boop, blip, fall, wide), `knock` (wood,
  plastic, snap, desk, double), `mallet` (glock, pluck, celesta, marimba; chime runs flourish, up4), `drip` (drop,
  bubble, plink, bloop, tiny), `shutter` (dslr, snap, mirror, film, quick). New members of old categories: whoosh_big
  thump / thump_deep / thump_airy, riser dark / build / snatch, swish bright / flick, whoosh_fast whip, glitch burst /
  mid / chirp, notify mid_tine / mid_wood. Every entry carries, besides its render block, meta and fingerprint: the
  catalog profile (`profile`: length, attack, t20 / t40, centroid, low / high shares, tonal share, glide, L/R
  correlation), its take audit in its kit role (`audit`: seeds n..n+5; 50 members 6 of 6 clean, glitch/mid 5 of 6)
  and its originality (`original.max_ref_clap`: the largest CLAP cosine of its render to any of the reference clips,
  three excerpt conventions, 0.53-0.84; the bar is 0.96). The build refuses a member whose own take is not clean,
  with fewer than 5 of 6 clean takes, or without a measured originality under the bar; a recorded value holds only
  for the render it was measured on (`bank.FORGE04_REF_ON`: the member's render block, and the shipped fingerprint),
  so a member changed since needs a fresh one (`build --forge04 --ref-clap`).
- **Recipes** (originals made from recipes, none copied from the reference clips): `ui_pop`, `knock`, `mallet` (with
  chime runs), `drip`, `shutter` (shared parts in `recipes/_tonal.py`), notify's mid dings (`notify.FAMILY_MID`;
  `ring` / `soft` / `bell_len` keys, off by default: earlier members render sample for sample); `cuthit`, `thump`,
  `revcymbal`, `darkriser`, `brightswish`, `burstglitch`. Eight members came last, each found by a model-free search
  from a hand-set corner toward a designed aim with every candidate's takes audited (drip bloop / tiny, shutter film /
  quick, mallet celesta / marimba, reverse_swell short / breath).
- **The launch preset** (`earshot kit init --preset launch`): music-led launch films. The 0.4 members where they fit
  (ui_pop/soft for the pop, knock/wood, cut_hit/punch and kick, whoosh_big/thump, riser/dark, reverse_swell/dark,
  notify/mid_tine, shutter/dslr, a wide gated mallet/glock bell and a soft mallet/pluck text), every role placed
  against the music bed. riser and swell are sounded (at most one or two a film, at a section change), suck is off;
  both are aligned on their peak (the 0.4 members fall after their climax: aligned on the end, the dark riser's
  climax would come 0.27 s before the cut), and `kit check` notes an end-aligned role whose sound rings on more than
  30 ms past its peak. Its 72 takes are clean in `kit check`.
- **Kit mix keys** (optional; a kit without them plans, keys and locks its takes as before): `bed_db` (the mix's
  loudest 50 ms at the effect, K-weighted, over the music's median 50 ms level in the second before the cue; the takes
  of such a role carry `l50_lufs`), `offset_ms` (the sync point against the picture: taps, knocks and clicks lead,
  pop-in thumps and whoosh and swell peaks land after), `min_gap_ms` (run spacing), `width` (the L / R correlation
  every take is set to by mid / side, measured into the lock as `corr`), and a kit-level `mix` block (the stop-down:
  the music steps down before a hit and comes back on it; no ducking). Helpers for a mixer: `kit.bed_level(music, t)`
  (None where the music is silent, under -60 LUFS: `gain_db` then falls back to `level_db`; a mono take or bed
  measured as the dual mono it plays as), `kit.gain_db(take, role, bed_lufs)` (the take alone lands at bed + 10
  log10(10^(bed_db/10) - 1), never over -1 dBFS at its peak), `kit.place(take, role, cue_s)`, `kit.space_run(times, min_gap_ms)`, `kit.stereo_width(y, corr)`,
  `kit.loud_lufs(y)`. The launch preset's numbers are medians measured on the reference clips (effects 3-10 dB over
  their bed; touches lead the cut by 11-60 ms, pop-in thumps and whoosh and swell peaks land 30-42 ms after it; runs
  66-159 ms apart).
- **The audit gives a drip its rise**: `sfx.check` applies `recipes/drip.allows()` to the drip category, so a climb a
  bubble makes (at most 14 st, starting within 60 ms, done within 250 ms) is a note, while any other rising tone is
  still the rising-glide warning (drip/bloop needs it).
- **The gesture model** (`earshot/forge/fit/gesture.py`, kind `gesture`): tones (the sines candidate's partials) +
  air (a detail noise model of what the tones leave, split in the magnitude domain), with motion knobs (length,
  sweep, doppler, proximity, pan, motion, width, roughness, tone_db / air_db, vary). Tried on every sound under 1 s
  and second in the long gesture / mixed / glide / ringing plans; it won 38 of the 78 dev clips. `earshot fit --kind
  gesture`; a bank fit entry's dur knob maps to its `length`.
- **Stereo and space.** Sines and modal fits keep the recording's side over time per octave band (the part coherent
  with the mid as a pan, the rest as the mid's quadrature): on 42 wide or decorrelated dev clips the side / mid error
  fell from 13.4 to 0.15 dB, the tail correlation error from 0.39 to 0.04. A hybrid's sines layer keeps its side and
  the residual the next layers are fitted on leaves it out (`fit_hybrid` takes the layer's stereo render out), so
  the air does not model the tones' side a second time. `earshot/forge/fit/space.py`: a fitted
  diffuse side layer for a stereo recording whose tail is decorrelated (`post['space']`). `quality` measures
  `side_db`, `width_db`, `tail_corr`, `tail_corr_err`, and `score` weighs width and tail correlation.
- **Beds.** A recording with a steady music bed in its pre-roll is stripped and the sound alone is fitted; the bed is
  rendered back under it by default (`render(bed=False)`: the sound alone, which is what a bank entry's and a kit's
  takes render: `bank add-fit` of such a fit keeps the bed out of its takes). `earshot fit --keep-bed` turns it off.
  Analysis by synthesis decides whether it is a bed: the winning candidate's kind is also fitted on the recording as
  it is, and the bed is dropped when that scores lower (`auto.BED_CHECK`; a whoosh's own steady approach had been
  stripped as music).
- **CLAP margin rule.** After the confirm stage a candidate whose CLAP is 0.02 higher on seed 0 and on the confirm
  seeds' mean, within 1.5 of the score and 0.05 of the envelope r, is chosen instead (`auto.CLAP_MARGIN`; rows keep
  `clap2`).
- **Time shape.** A post-stage gate (cut-only, kept when it lowers the score and costs no CLAP) for gated pulses,
  hard stops and merged runs (it follows the knobs that move time: a modal spacing moves each reopening with the
  strike it opens for; a gesture length moves the gate and the attack clamp about the tones' first strike; a morph
  warps each source's gate onto its timeline); modal noise bands may not rise before a strike (onset error 0.39 -> 0.25 octave on the
  modal dev fits); the glide gate now fires at 2 semitones over 15 ms; the best three candidates are confirmed on two
  more seeds.
- **Spectral**: a detail fit (`spectral.DETAIL`, `quiet_db`), `trajectory()` and the motion knobs doppler,
  proximity, pan and motion.
- **Tonal knobs** (`knobs.apply(..., decay=, onset=, bright=, width=)`): t20, attack, centroid and L/R correlation
  targets landed by measurement through one recipe parameter; pitch knobs for ui_pop, mallet, knock, drip and
  cut_hit; a dur knob for reverse_swell.
- **`eval/forge/replicate.py`**, the replication benchmark (targets read from a folder you pass, or a frozen split
  file with `--split-file`; results go outside the repo or under `out/`; an explicit `PYTHONPATH` picks the engine).

### Changed

- Every render of a model fitted by 0.4 starts at zero (a fade-in of 2 ms, shorter when the onset is earlier) and
  peaks at -1 dBFS at most. Models fitted before 0.4 (modal, sines and hybrid models without the `coherent` flag: the
  0.3 bank's fit entries; spectral models without the `finish` flag, which every 0.4 spectral fit carries) render as
  0.3.0 did, sample for sample (`auto.finished()`).
- **calm, crisp and minimal sound as in 0.3.0**: rendered with the 0.3.0 code and with 0.4.0, all 191 of their takes
  are byte-identical. The bank version is part of every take's cache key, so a kit re-renders once after the update.
- Takes of fits made by 0.4 (models marked `coherent`): modal vary moves the modes by a smooth random function of
  log frequency, the same in every strike, with phases kept on the first strike's timeline, and strike times no
  longer jitter (independent per-mode detune undid the fit's cancellations: takes rang on); multi-strike modal fits
  keep their strikes coherent under the size and spacing knobs too; sines vary moves neighbouring partials together.
  A modal or hybrid fit's `vary_scale` 0.5 halves the bank's vary for these kinds.
- The attack clamp of a noise model's post stage is high-passed and kept only where it helps.
- Takes without clicks: the post gate's gain is smoothed over 1 ms, a later modal strike rises over 0.15 ms (coherent
  models), and a coherent sines model jitters a strike's atoms together.
- The gesture air's grain texture is levelled over 5 ms windows and kept only when it is stationary
  (`spectral.fit(grain_norm_ms=, grain_halves=)`, `grains['norm_ms']`; older models render as before).
- `kit check`'s restraint note says which presets keep riser, swell and suck off (launch sounds riser and swell).
- Recipe comments no longer name a source film.

### Measured (replication benchmark)

Tuned on the dev split only; each held-out set opened once. Details and the regressions in
[docs/FIT.md](docs/FIT.md#the-replication-benchmark-04). The fresh held-out set is harder and more bedded than the
benchmark (its isolation bar is lower): compare engines on it, not its numbers with the other rows. The 0.4.0 rows were
measured before the release review's hybrid side fix (a hybrid's residual now leaves its tone layer's stereo side
out), which can change hybrid candidates on stereo tonal clips; they were not measured again.

| split | engine | CLAP (median) | mel gap dB | high / good / fair / poor | takes pass |
|---|---|---|---|---|---|
| dev (78) | 0.3 | 0.884 | 1.43 | 19 / 20 / 31 / 8 | 29 |
| dev (78) | 0.4, first round | 0.907 | 0.90 | 30 / 26 / 20 / 2 | 46 |
| dev (78) | 0.4.0 | 0.921 | 0.85 | 40 / 20 / 17 / 1 | 42 |
| held-out (38) | 0.3 | 0.878 | 1.73 | 7 / 8 / 19 / 4 | 21 |
| held-out (38) | 0.4, first round | 0.909 | 0.98 | 14 / 9 / 12 / 3 | 25 |
| held-out (38), opened | 0.4.0 | 0.903 | 1.04 | 14 / 11 / 12 / 1 | 25 |
| fresh held-out (28) | 0.3 | 0.853 | 0.87 | 2 / 6 / 14 / 6 | 8 |
| fresh held-out (28) | 0.4, first round | 0.830 | 0.75 | 4 / 3 / 14 / 7 | 12 |
| fresh held-out (28) | 0.4.0 | 0.851 | 0.58 | 5 / 6 / 13 / 4 | 7 |

On the fresh held-out set 0.4.0 is level with 0.3 on the median CLAP (0.851 against 0.853) and ahead on the mean
(0.860 against 0.844), the mel gap, the grades (grade up on 7 clips, down on 2) and the take CLAP (0.863 against
0.809); fewer takes pass the take check (7 against 8), and the onset error grew (1.00 octave against 0.53).

### Not done (known gaps)

- The four stereo sines fit entries (fit/click/floss_lid, fit/key.space/thock_space, fit/snap/switch_flip,
  fit/whoosh.short/bamboo_swish) keep their 0.3 fits with the old, too quiet side: calm uses all four, crisp three
  (not bamboo_swish), minimal switch_flip and launch floss_lid and thock_space, and a refit would change those
  presets' takes (calm, crisp and minimal stay byte-identical to 0.3.0).
- The bed check has no CLAP guard: on the fresh held-out set it dropped a knock's bed although the plain fit had 0.087
  less CLAP, and a plain fit of a bedded clip keeps the bed in its sound (four of the six new take failures there).
- Onsets: the onset error grew against 0.3 on the fresh held-out set (14 of 28 clips off by more than an octave).
- Take clicks left: a loud grain inside an air-only key's attack and one modal-layer click in a hybrid take; a comb
  ring in the ring-out tails of short modal clips (7-8 dev clips).
- Multi-pulse counts on long gesture wins (7 of the 28 dev clips with 3 or more pulses miss by more than 2) and clips
  of 0.5 s or longer (3 of 13 high on dev).
- Takes passing fell this round (dev 46 -> 42, fresh held-out 12 -> 7) although the take CLAP rose.
- Both held-out sets are now opened; a next round needs a new one.
- No listening test of the 0.4 sounds: they were chosen and checked by measurement (profiles, audits, CLAP).

## 0.3.0 - 2026-10-19

The sound engine of a film: the forge's delivered sounds as a bank, one-sound renders aimed at the picture, a
model-free audit, and film kits. Reference: [docs/SFX.md](docs/SFX.md). This release also ships the 0.2.1 fixes
below (#1, #2), which were not released on their own. No new dependencies.

### Added

- **The sound bank** (`earshot/forge/bank.py`, `earshot/forge/bank/*.json`, shipped as package data): the 183 forge
  v2 finals as their schema-2 render blocks at full precision (`forge_v2r6`), plus three tuned kalimba voices from the
  launch film (`pluck1`). Ids are `<category>/<member>`, `pluck/<name>` and `fit/<role>/<name>`; categories are the
  public names (no recipe module names needed). `python -m earshot.forge.bank build out/forge_v2/*/final -o ...`
  refuses any pick whose block does not re-render its WAV (183 of 183 within 1.2e-7); each entry stores its measured
  onset, peak, end, attack, centroid, loudness and a 50 ms envelope fingerprint, and `bank check` re-renders against
  it (length exact, peak within 5 ms, envelope within 0.5 dB). Rendering needs numpy / scipy only.
- **Knobs** (`earshot/forge/knobs.py`): `dur` (exact: the whoosh family scales its pass, length keys are searched by
  secant, then the file is faded or padded to the sample), `apex` (whoosh family: closed form, up to two measured
  corrections, then the peak is placed by trimming the quiet approach or padding silence: within 2.5 ms on every
  tested target), `pitch` (a note name or Hz into each recipe's own pitch parameter, co-scaling the parameters of the
  same body, inside the parameter's designed range; noise sounds refuse it with the reason), `weight` / `peak_db`.
- **`earshot sfx list / render / check`** and the audit (`earshot/forge/sfx.py`): polish's artifact detectors as
  warnings with context (gated categories may stop on their peak), a rising pitch-glide detector (a tonal sound
  climbing more than 4 semitones over 50 ms or more) and, for hits, a stacked-attack / pre-swell heuristic. Every bank
  render runs it.
- **`earshot kit init / render / check / audition`** (`earshot/forge/kit.py`): kit.json (schema 1) names one bank
  sound per role; `render` writes deterministic, cached takes and `kit.lock.json` with each take's measured onset /
  peak / end / loudness; a take that fails the audit is re-rolled on a new seed. Presets `calm`, `crisp`, `minimal`
  (`earshot/forge/kits/`) with a reason per role; riser, swell and suck off in all three.
- **`python -m earshot.forge.bank add-fit MODEL.json --id fit/<role>/<name> --source SOURCE.json`**: a fitted recipe
  as a bank entry with its source (CC0 / public domain unless `--allow-license`; NC / ND refused), rendered through
  `earshot.forge.fit`, tunable when the fit is tonal. Extra bank files load from `EARSHOT_BANK`.
- **Python API** `earshot.api.sfx_list / sfx_render / sfx_check / kit_init / kit_render / kit_check / kit_audition`;
  **MCP tools** `sfx_list`, `sfx_render`, `kit_render` (16 tools in all).
- **Fitted sounds in the bank** (`earshot/forge/bank/fit.json`, `fit1`; `bank.VERSION` is now
  `forge_v2r6+fit1+pluck1`): 15 recipes fitted from CC0 recordings on Freesound, `fit/<role>/<name>`, each with its
  source (title, author, URL, licence) and the take-to-take `vary` at which its takes passed the take check (seeds
  0-4 as the bank renders them: no hard artifact beyond the recording's, every audit warning also on the recording,
  CLAP >= 0.90 to the fit's own render and within 0.06 of the fit's CLAP to the recording, excess energy within
  1.5 dB, loudness spread <= 2 LU), at a fit grade of high. Credits: `docs/CREDITS_FIT.md` (CC0 needs none).
- **Fit takes** (`earshot/forge/fit`, details in [docs/FIT.md](docs/FIT.md#takes-a-fit-rendered-for-use-03)):
  `fit.render(model, seed, vary=v)` with `vary > 0`, or `take=True`, renders a take. Modal modes and sines partials
  ring out past the trimmed recording's end instead of stopping on a ringing tail (13 of 46 CC0 fits had cut tails in
  their takes), takes are high-passed at 8 Hz (no random DC offset), a spectral model's takes keep one loudness
  (a cloth whoosh's 8 seeds: 7.3 -> 0.45 LU apart), and a sines residual follows the recording's roll-off above 16 kHz
  (`residual.hf`; no more noise above 20 kHz the recording lacked). `earshot fit --takes` scores candidates on seeds
  0, 1, 2. The fit itself is unchanged: on the 12 validation refs every seed-0 and seed-averaged quality number is
  the same as in 0.2.
- `earshot doctor` reports the sound bank: its version, the files it loaded and how many sounds of each type.
- Docs: `docs/SFX.md`; README "Sound for films"; `docs/INTEGRATION.md` "cinetic uses earshot"; the earshot skill's
  Loop 5 "make a film's sound kit"; `docs/FIT.md` "Takes"; `docs/CREDITS_FIT.md`.
- Tests: `tests/test_bank.py`, `tests/test_sfx_kit.py`, `tests/test_fit_takes.py` (no models).

### Changed

- Wheels now ship the forge seed banks (`earshot/forge/seeds/*.json`) next to the bank and kit presets; before, a
  non-editable install had none.
- **Presets use the fitted sounds where the measured comparison favours them** (earshot match of the role's
  description, also against synth-blip descriptions, and earshot realness): calm tick, pop, short whoosh, space bar,
  click and snap; crisp tick, light touch, text, click, snap, key and space bar; minimal light touch, text and snap
  (16 role changes). A changed role keeps its loudness (`level_db` moved by the measured difference at the same
  peak). Every note now says what the sound is and what was measured; no note cites a listening test. All takes of
  the three presets pass `earshot kit check` (64, 64 and 63 takes).
- **The presets' bell spans two octaves of one pentatonic** (calm, crisp: kalimba G4-E6; minimal: the ding A5-G7; 10
  notes each), so a climbing progress motif has a rung for every step; calm's tonal text holds the five pentatonic
  degrees of one octave (C5 D5 E5 G5 A5), so a chain of fragments can step through the chord's tones. crisp and
  minimal keep an untuned soft tap for text.

### Fixed (review before release)

- **apex with a short tail**: the fade that fits a whoosh to `dur` covered an apex 0.03-0.1 s before the end, so the
  delivered peak came 8-189 ms early while `edits['apex']` (measured before that fit) said ok. The fade now starts at
  least 5 ms after the apex, candidates are ranked on apex plus tail error, the peak is placed in whole 1 ms steps of
  its measure, and `edits['apex']` is measured on the delivered file: 59 whoosh-family targets with 0.1-0.12 s tails
  land within 2.5 ms.
- **`kit check` refuses what `kit render` cannot use**, with the same checks (`knobs.validate`, `bank.level`): a null
  or non-numeric `level_db`, `max_s` under 0.01 s (a silent take), `peak_db` / `weight` over 0 dBFS, knob values and
  pitches the sound cannot take, a bad `sr`, and take files that would collide (two spellings of one pitch, or names
  that differ only in case). Pitch labels are canonical (`C#5`, `Db5`, 554.37 Hz -> `Cs5`).
- `knobs.midi` accepts `s` sharps (`Cs5`), the spelling of take file names and of cinetic's kit.py.
- The audit flags **silence** (`silent`: a peak under -60 dBFS or less than 5 ms within 40 dB of it).
- `bank.render` / `earshot sfx render` refuse a peak over 0 dBFS and sample rates other than 44100 / 48000 / 88200 /
  96000 (the CLI wrote clipped WAVs).
- **Click and toggle pitch**: their `pitch` parameter is the tink centre, not the partial heard (click/soft +291
  cents), so `--pitch C7` landed 362 cents off with no warning. Their note is now the measured partial (`meta.pitch_hz`
  in the bank; no pitch knob where there is no clear partial), a target is landed by measurement, every measured pitch
  edit reports `ok` (within 50 cents), and a miss adds a `pitch_off` warning to the take's checks.
- `EARSHOT_BANK`: the docstring gives `os.pathsep` as the separator, and a missing or unreadable file is a readable
  error naming it instead of a traceback.
- The README and docs/SFX.md agree on the blind test: no v2 score is published; the README's test is the v1 one (one
  listener, the author, 15 trials).

## 0.2.1 - 2026-10-17

Not released separately: these fixes ship in 0.3.0.

Two fixes from a careful outside review (thanks, @markblundeberg).

### Fixed

- **Agent-facing text no longer states one user's preferences as instructions** (#1). The `realness` tool description
  and the MCP server instructions told every agent to check every sound because "the user rejects synthetic-sounding
  audio"; the earshot skill said to reject 'likely code-synthesized' sounds outright, and the real-sound skill and
  several docs called the bundled launch-film profile "the user's". They now say what realness measures and leave the
  decision to the current user: use it when real or professionally produced sound is wanted, and treat a synthesized
  verdict as information when synthesized sound is the point. `critique`'s realness fix is scoped the same way. A
  test keeps such phrasing out of tool descriptions, the server instructions, skills and docs.
- **Model loading respects the user's offline setting, never downloads after a failed load, and pins its weights**
  (#2). Before, earshot set `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` once a model's cache folder existed, cleared
  them and downloaded again after any failed local load (a damaged cache, a transformers mismatch, a typo in
  `EARSHOT_CLAP`), and fetched whatever each repo's main held, for CLAP a pickle. Now (`earshot/models/hub.py`):
  - `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` mean the same as `EARSHOT_OFFLINE=1`: no download, a clear error
    naming the flag when a model is not cached. earshot never sets or clears any of them.
  - Models load from a complete local snapshot folder only (transformers gets a folder, not a repo id), so a cached
    load never contacts the Hub (no update checks, no safetensors conversion requests). A failed load is an error
    that names the folder; the only download is for files that are not in the cache.
  - Pinned revisions with safetensors weights: CLAP `16c8cc3` (the Hub's safetensors conversion of main `ada0c23`),
    AST `f826b80`; overrides `EARSHOT_CLAP_REVISION` / `EARSHOT_AST_REVISION`. A cache made before 0.2.1 (CLAP
    `ada0c23`, `pytorch_model.bin`, the same tensors) keeps working from the cache, without a forced 776 MB download.
  - `earshot doctor` reports, per model, the pinned revision, what the cache holds (cached / legacy / incomplete /
    missing), the folder that loads and the offline flag; `--load` adds the revision and weights file that loaded.
    The first download is ~1.1 GB (the docs said 1.4 GB).
  - `EARSHOT_CLAP` / `EARSHOT_AST` may still name a local model folder (`config.json` plus `model.safetensors`, or an
    older checkpoint's `pytorch_model.bin`): it loads from disk as it is, with no network and no revision, has its own
    embedding caches, and earshot never downloads into it (an incomplete folder is an error naming what is missing).
  - A custom Hub model downloads its top-level config, tokenizer and safetensors files by name; the old `*.json` /
    `*.safetensors` globs also matched every subfolder and weight variant of a repo.
- **The optional Gemini critique no longer treats synthesized sound as a defect by itself** (#1). Its rubric set
  "real recorded foley" as the bar and scored "obviously cheap or synthetic"; it now judges each sound by whether it
  fits the film and is well made, counts synthesis as a flaw only where a sound should read as recorded, and
  `earshot llm --synth-intended` (MCP `synth_intended`) tells the model that synthesized sound is the point.

### Added

- `earshot doctor --fetch`: download the pinned model files that are not in the cache (also upgrades a legacy CLAP
  cache to the pinned safetensors).

### Changed

- A custom model (`EARSHOT_CLAP` / `EARSHOT_AST`) or another revision keeps its embedding caches per commit, so they
  never mix with the pinned model's; the default models keep their existing caches.

## 0.2.0 - 2026-10-16

forge v2 and `earshot fit`.

### Added

- **`earshot fit`: audio to code.** Fits a small synthesis model (ringing modes, tracked partials, a time-varying
  noise gesture, or a layered hybrid of them) to a recording by analysis by synthesis, and writes it as a readable
  forge recipe module whose `render()` makes the sound. The module stores the structure of the sound, not its
  samples: new takes per seed, knobs for variations, `--morph A B` between two fits of one kind, `--forge` for
  critic-searched variations. On 12 real references: median CLAP cosine 0.945 between recording and render (minimum
  0.881), the same top `earshot describe` tag in 10 of 12. In our 12-trial blind test the listener could not tell
  the recordings from the renders (one listener: a small test). Method, numbers and limits:
  [docs/FIT.md](docs/FIT.md).
- **forge v2.**
  - Reference sets (`earshot/forge/refsets.py`): 18 motion-design categories, split by source into a 'forge' side the
    search may use (81 refs) and a 'bench' side it never hears (71 refs). The recordings stay local.
  - A fixed held-out benchmark (`eval/forge/bench.py`): tone, envelope, independent phrasings, AudioSet names,
    artifacts and variety, next to the same checks for real recordings. Results per round in `eval/forge/`.
  - Critic v2: model-free penalties for tonal balance, the spectro-temporal envelope, artifacts and polish; a designed
    mode for categories without real references. `version=1` reproduces v1.
  - Search v2: model-free pre-screen, CMA-ES option, threaded renders, robust re-renders, seed bank,
    `report.md` / `compare.png` / audition reel, `forge(families=)` and `palette()`.
  - DSP toolkit: `earshot/forge/dsp.py` (sources) and `earshot/forge/fx.py` (effects).
  - 34 categories as families of named characters, 4-6 variants each: 18 real-recording categories and 16 `pro_*`
    designed ones.
  - One report schema (`earshot/forge/schema.py`) and `earshot forge --check`: every pick re-renders from its
    `report.json`.
  - The pro motion-SFX study: 58 designed effects measured in 8 professional launch films (anonymous ids, nothing
    redistributed) and a cited technique note (`docs/motion-sfx-study.md`, `docs/motion-sfx-techniques.md`,
    `eval/forge/pro/`).
  - A clean blind-test kit (`eval/forge/blind_v2.py`): loudness-matched, randomised trials of made sounds against
    held-out recordings, answer sheet, sealed key, scorer, and a cleanliness report that stops the build on a fail.
  - `eval/forge/scrub_paths.py`: removes machine paths from results; `bench.py` writes scrubbed paths.
- **`earshot gallery`**: one static HTML listening page for forge outputs (players, spectrograms, loudness, critic and
  benchmark numbers, parameters), with `--compare`, `--embed` and `--embed-compressed`.

### Changed

- The `earshot forge` CLI defaults are the v2 setup: pre-screen, seed bank, robust re-renders with a shortlist, the
  critic v2 tone weight, and the recipe's reference set as default refs. The Python `forge()` defaults still
  reproduce v1.
- forge v1 -> v2 on the held-out benchmark, in the five categories v1 had: variants are no longer near copies (CLAP
  cosine between variants 0.89-0.97 -> 0.71-0.85), tails that died too fast reach the real length (click 0.56 -> 1.44
  x the bench median, fast whoosh 0.51 -> 0.88), tone balance closer in 3 of 5. All numbers:
  [docs/FORGE_V2.md](docs/FORGE_V2.md).
- The launch film was re-scored with forge v2 sounds (in its own project; no media here): `earshot critique` 16 -> 19
  of 22 metrics in range, realness 0.73 -> 0.84, cues with an onset within one frame 86 -> 98 % of 43; -14.0 LUFS,
  true peak -2.2 dBTP.

### Known limits

- No blind-test score for the forge v2 sounds is published yet.
- `earshot fit`'s selection weights were set on the same 12 references they were measured on; risers, zips and long
  designed sounds are untested.

## 0.1.0 - 2026-10-16

First public release. `describe`, `match`, `rank`, `find`, `compare` / `ab`, `realness`, `timeline`, `sheet`,
`critique`, `profile`, `measure`, `forge` (v1: five recipes), the optional `llm` critique (own key, explicit consent),
`mcp` and `doctor`, as a CLI, a Python API and an MCP server, on CPU. The `earshot` and `real-sound` agent skills,
held-out evaluations ([docs/VALIDATION.md](docs/VALIDATION.md)) and limits ([docs/LIMITS.md](docs/LIMITS.md)). The
22 s launch film, every sound made with `earshot forge`.
