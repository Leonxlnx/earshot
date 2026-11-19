# earshot build: a sound the catalogue lacks, built with earshot as the ears

`earshot make` answers from a catalogue of 922 designs. For a thing it has nothing of it says so (`earshot makes no
whale song; wolf/howl is only the nearest sound it has`). `earshot build` is the workbench for that case: it drafts a
recipe (`earshot.recipe`) from the words, renders it, listens to it with earshot's listening model, moves its numbers
until it fits the words better, and says what is still off in lines an agent can act on.

```
earshot build "hail on a tin roof" -o hail.wav                # draft, render, hear, tune: hail.wav, hail.recipe.json
earshot build "a stampede of horses" -o stampede.wav \
  --part "hundreds of hoofbeats on dirt, for 6 s, getting louder" \
  --part "a low rumble under everything" \
  --part "a horse neighing, twice, far"                        # you name the parts, earshot finds the recordings
earshot build "hail on a tin roof" --recipe hail.recipe.json   # your edited recipe: render it, hear it, say what is off
earshot build "a stampede" --rounds 40 --without step/gravel rain --json
```

```python
from earshot.build import build, draft, hear, tune, from_parts
r = build('hail on a tin roof', out='hail.wav')
print(r['summary'])                      # the lines below; r['heard'] holds every number
r = build('a stampede of horses', out='stampede.wav', parts=['hundreds of hoofbeats on dirt, for 6 s',
                                                             'a low rumble under everything'])
recipe, parts = from_parts('a stampede of horses', ['a horse neighing, twice, far'])   # the recipe, not rendered
drafts = draft('a whale singing')        # one to three recipes from the words alone
heard = hear('hail.recipe.json', 'hail on a tin roof')
better = tune(drafts[0], 'a whale singing', rounds=20)['recipe']
```

The MCP server has the same as the tool `build` (`want`, `out`, `parts`, `recipe`, `rounds`, `without`, `seed`).

No model generates audio here. The material is the catalogue's designs, most of them learned from public-domain
recordings; the recipe gives the structure; the listening model (CLAP, the `listen` extra) only judges.

## The loop an agent runs

1. Say what physically makes the sound, one `--part` a thing: `earshot build "X" -o x.wav --part "..." --part "..."`.
   Without `--part` earshot drafts the parts itself (below). It writes `x.wav` and `x.recipe.json`.
2. Read the `part N` lines: which design each part became and why, the next nearest designs, the thing earshot has
   no recording of (`earshot has no ...`) and the words it did not understand (`not understood in part N: ...`).
   Swap a part by naming its design in the phrase (`"hoofbeats sound=horse/trot, many"`) or by editing the layer's
   `sound`.
3. Read `fit:`: the z of the sound for the words, beside the z of the design `make` would hand over, and the verdict:
   "sounds like it" when the sound is clean, its z is at least 3.0 and at least 1.0 over that design, and the second
   ear does not contradict it; otherwise "a sketch". `heard as:` names the designs it sounds most like, the words it
   is nearest to and the second ear's top labels.
4. Read `still off:`. Each line names the field of the recipe to change ("layer 1 (rain/tin_roof) covers the rest: the
   fit rises to z +4.10 without it; lower its `gain` by about 6 dB or leave it out").
5. Change one thing in `x.recipe.json` and hear it again: `earshot build "X" --recipe x.recipe.json` (add `--rounds
   20` to let the tuner move its numbers again from there).
6. Hand it over as what `fit` says. A sketch is a sketch. A person should listen once.

## Parts in words

A part is one plain phrase: the thing that sounds, and how. earshot reads the phrase with this grammar; what the
grammar leaves is the thing, which earshot's reader and the listening model's atlas find the nearest recorded design
for (never an instrument or a synthesized design; for a many, a row or a bed it prefers a take of that shape among
the near ones). `sound=<id>` names the design instead.

| the phrase says | the layer gets |
|---|---|
| once, a single | one event |
| twice, a few, several, three times, 5 times | `events` (count 2, 3, 4, 3, 5; a few and several looser) |
| many, lots of, constant, steady, dozens / hundreds / thousands of, a crowd / swarm / shower / stampede / flock / hail / herd / pile of | `scatter` at a rate that fits (many 12 a second, hundreds 40, a swarm 30, a flock 8; a long take less densely) |
| at 1.5 s, after 2 s; at the start; at the end; after the hit | `at` |
| under everything, in the background, underneath | a bed: the whole length, `gain` -8 dB |
| for 4 s | `dur` (a bed or a long take), the scatter's `dur`, or `stretch` up to 2 |
| long, short | `stretch` 1.6, 0.7 |
| passing, going by, flying over, racing past | `pass` (closest at the middle of the layer, 25 m/s) |
| from left to right, from right to left | `pan` [-0.8, 0.8] |
| getting louder, approaching, building up; fading, dying away | `level` from -18 dB up; down to -24 dB |
| swelling | `swell` |
| lower, deeper, higher; much lower, much higher | `pitch` -4, +4; -8, +7 |
| rising, falling (on their own between commas, or "in pitch") | `bend` up or down by about 5 semitones |
| reversed, backwards | `reverse` |
| far, distant | `lowpass` 2500 Hz and -8 dB |
| close, loud, soft, quiet, muffled | +2 dB, +3 dB, -6 dB, -6 dB, `lowpass` 800 Hz |
| pulsing, throbbing, chopping, fluttering (at N a second, at N Hz) | `spin` (4 a second when no rate; chopping: `chop`) |
| sound=<id> | that design |

Gains are set against the first part. Everything else stays with the thing ("a low rumble", "hoofbeats on dirt").
A word neither the grammar, the reader nor the catalogue knows is answered with `not understood in part N: ...`; a
thing the catalogue has no recording of with `part N: earshot has no X; Y is only the nearest it has`. The thing is
looked up in the designs' own words too (ids, tags, uses, descriptions, as `earshot sounds --search` reads them): a
recording whose own words name it comes before one the reader reached by a single loose word, one that names both
words of a thing ("train whistle") before one that names one, and a word that only describes ("low", "big") never
chooses. A sound word ("a rumble", "a hiss") is looked up the same way. A synthesized design is taken only when no
recording names the thing, and the line says so: `no recording of 'X'; Y is synthesized`. The first
layer may say "falling" as an action ("books falling") without a bend: only a lone "falling" between commas bends.

## What each step does

**draft(words)** reads the request with earshot's reader (the thing, what it does, on what, where, how many, how
long) and picks a structure by the kind of sound its words name:

| structure | for | layers |
|---|---|---|
| one | a single event (a slam, a crack) | the nearest part, a second under it, a third lower and darker |
| row | a few events ("three knocks", footsteps, galloping) | the part as `events` |
| scatter | very many small events (rain, hail, applause, a swarm, a stampede) | a short part `scatter`ed, a bed under it |
| run | something that runs (a motor, a machine) | a long part with `spin`, a second, a little rumble |
| pass | something that passes (a car, a jet) | the part with `pass` (Doppler and distance) |
| voice | a creature's voice | the nearest voice moved in `pitch` (lower for big, higher for small) with a `bend` |
| long | something drawn out (a creak, a groan) | the part `stretch`ed and lower |
| magic | something made up (a portal, a spell, a force field) | a reversed part as the swell, a hit at its end, air |
| place | a place (a station, a forest at night) | two or three beds |

The parts are recorded designs only (never an instrument, a note or a synthesized design), the nearest to the whole
request and to each thing it names: with the listen extra by the listening model (half the atlas's ranking of a
design's sound and descriptions, half how well its sound alone fits the words), without it by the reader's ranking.
The head of the words leads ("a fire alarm" is an alarm: "fire" counts less). Many of one thing ("a stampede of
horses", "a flock of geese", "a shower of books", "applause") is a scatter of the single thing. A cookbook recipe of
the thing is the first draft, and the plain design `make` would hand over is always one of the drafts: when no build
beats it, the answer says so and hands it over as it is. `--without` leaves out designs (an id) or families (a name).

**hear(recipe, words)** renders and measures: clean or what the audit found; the length, where the hit is, how bright
(the median spectral centroid) and how dense (onsets a second), beside the design `make` would hand over; with the
listen extra also the fit (the same z as `earshot match`), what it is heard as, whether it is heard as recorded
(`earshot realness`, from the same embedding), a steady tone heard as a note, and per layer the fit with that layer
left out. When a design of the catalogue fits the words better than the build, it says which.

The verdict has a second ear that the tuner never sees: the AudioSet tagger of the listen extra (527 classes, hears
up to 8 kHz). Where it has a class of the thing (a label that shares a word with the request: Horse for horses), that
class must be among its ten top labels; where it has none, it must not put a clearly different sound first (music,
speech, a synth tone, at p 0.3 or more). Otherwise the verdict is "a sketch: the second ear does not hear it". Its top
four labels are printed in `heard as:`.

A remark that compares the build with the nearest design (brighter, duller, sparser, busier) is only made when that
design is of the thing the words name (it shares a word with them: rain/tin_roof for heavy rain), and only when
following it does not lower the fit: hear tries the change it proposes once; a remark whose change lowers the fit is
printed as `not advised: ...` instead.

**tune(recipe, words, rounds)** keeps the layers and their parts and moves only numbers: gains (layer 0 stays the
reference), pitch and glide, cutoffs, starts, lengths, stretch, the counts, gaps and spreads of `events`, `scatter`,
`pass` and `spin`, each within the range `earshot.recipe` checks (read from that module). Each round changes one or
two numbers, renders, hears once and keeps the change when the score rises. The score is the fit minus penalties:
not clean (1), heard as synthesized (up to 2), a steady tone (1.5), a length far from the one asked (2 for a
doubling past a quarter off). The same seed gives the same result; the result always passes `earshot recipe
--check`. Without the listen extra it can only keep the sound clean and near the length asked.

**build(words)** drafts, hears the drafts, tunes the best, hears it again and writes the sound and the recipe.

## Time

Loading the listening model takes 20 to 60 s on a laptop CPU (once per process; the MCP server keeps it warm). A
round is a render and one embedding: under a second for a short event, a few seconds for a long bed of several
layers. A default build (20 rounds) of a short sound takes about a minute after the model is loaded; a build of
three to five named parts with beds of 6 to 10 s takes one to two minutes, as long beds cost more per round.

## Without the listen extra

It drafts by the reader, renders, audits and measures (length, hit, brightness, density) and says in one line that
it did not hear: `pip install "earshot[listen] @ git+https://github.com/Leonxlnx/earshot"` gives it ears.

## Worked examples

**A thing the catalogue lacks.** The words name a creature's song, so the draft is a voice: the recorded calls nearest
the words, moved down an octave, the first bent and stretched. Twenty rounds moved the pitches and a gain.

```
$ earshot build "a whale singing" -o whale.wav
whale.wav: recipe/a_whale_singing (10.00 s, peak at 2.09 s, -14 LUFS, clean) [built]
built from: owl/hoot (-11.8 semitones, bent through +0, -2 semitones, stretched 2 times); wolf/howl (-14.2 semitones)
heard as: wolf/howl, downlifter/synth_fall, cat/meow; words: 'a cat meowing' (z +3.1), 'being underwater' (z +2.5), 'an explosion' (z +2.4); second ear: Howl 0.76, Animal 0.68, Canidae, dogs, wolves 0.45, Dog 0.40
fit: z +4.09 for 'a whale singing' (the nearest design wolf/howl: z +2.71): sounds like it; recorded: p_real 0.75
tuned in 20 rounds: layer 0 (owl/hoot) pitch -9 -> -11.8 semitones; layer 0 (owl/hoot) stretch 1.6 -> 2; layer 1 (wolf/howl) pitch -9 -> -14.2 semitones; layer 1 (wolf/howl) gain -9 -> -11.4 dB; fit z +3.07 -> +4.09
still off: nothing the measures find; a person should listen once
next: edit whale.recipe.json, then: earshot build "a whale singing" --recipe whale.recipe.json
```

The nearest design is a wolf, so no line compares the whale's brightness with it: a whale is darker than a wolf, and
raising the pitch to match the wolf lowered the fit from +4.09 to +3.71. The second ear's top labels are a wolf's:
the tagger has a class for a whale, and whether it is among its top ten decides the verdict at the margin. A person
is the last judge.

**Parts named by an agent.**

```
$ earshot build "a stampede of horses" -o stampede.wav --part "hundreds of hoofbeats on dirt, for 6 s, getting louder" --part "a low rumble under everything" --part "a horse neighing, twice, far"
part 1 'hundreds of hoofbeats on dirt, for 6 s, getting louder': horse/trot (scattered 16.5 a second over 6 s, its level drawn: -18, +0 dB); horse/trot names 'hoofbeats' in its own words (atlas and own words +3.5, a short take to repeat); next nearest: heart/beat, whisk/beat, step/concrete
part 2 'a low rumble under everything': thunder/roll; thunder/roll names 'low rumble' in its own words (atlas and own words +3.2, a long take for a bed); next nearest: roll/drum, explosion/far, cat/purr
part 3 'a horse neighing, twice, far': horse/neigh (2 times, under 2500 Hz); horse/neigh names 'horse neighing' in its own words (atlas and own words +6.6, a short take to repeat); next nearest: cow/moo, elephant/trumpet, bear/growl
fit: z +3.56 for 'a stampede of horses' (the nearest design horse/neigh: z +2.61): a sketch: not clearly it yet; recorded: p_real 0.96
still off: layer 1 (thunder/roll) covers the rest: the fit rises to z +3.88 without it; lower its `gain` by about 6 dB or leave it out
```

"hoofbeats" is not a word of the reader, but horse/trot's own words have hooves: the part finds it as `earshot sounds
--search hooves` does. The second ear hears Neigh, Horse and Clip-clop; the thunder under it is the part to turn down.
With the rumble said "soft" (6 dB lower) and the hoofbeats without "on dirt", the same build is "sounds like it" at
z +3.72, the second ear hearing Neigh, Horse and Clip-clop (the run is pasted in full in the README).

**Without the listen extra.** The words ask for many small events on a surface, so the draft is a scatter; the reader
found hail on a roof and rain on a tin roof. Nothing is heard, and it says so:

```
$ earshot build "hail on a tin roof" -o hail_plain.wav --no-listen --rounds 4
hail_plain.wav: recipe/hail_on_a_tin_roof (3.95 s, peak at 0.44 s, -19 LUFS, clean) [built]
built from: hail/roof; rain/tin_roof
not listened: tuned for a clean sound and the length only
measured: 3.95 s, hit at 0.43 s, 5.9 kHz bright, 0.8 onsets/s; the nearest real design hail/roof: 20.00 s, hit at 1.79 s, 6.3 kHz, 3.1 onsets/s
tuned in 4 rounds: no change scored better
still off: nothing the measures find; a person should listen once
next: edit hail_plain.recipe.json, then: earshot build "hail on a tin roof" --recipe hail_plain.recipe.json
```

## How well it does

Measured with the listening model as the judge, the same z as `earshot match`:

- 30 things the catalogue has a recording of (animals, weather, vehicles, household machines, tools, people, impacts,
  a night jungle), each built with its own design and family left out (`--without`). The build fits the words better
  than the design `make` would hand over without that family for 29 of 30, by 2.0 z on average; 15 of 30 pass the
  rule above; 20 of 30 fit better than the best design of any other family. Asked with the build's sound, the
  listening model puts the left-out recording among the 10 nearest of the 922 designs for 20 of 30 (median place 5).
- 30 things the catalogue lacks (a whale's song, a car alarm, a stampede, marbles on a wooden floor, an iceberg
  cracking, a squeaky shopping cart ...). The drafts alone fit better than what `make` hands over today for 28 of 30,
  the tuned builds for 29 of 30; 22 of 30 are "sounds like it", 8 are sketches.

- 12 more things it lacks (a stampede, a pack of wolves, a dragon's wings, a city thunderstorm, a forest fire, a steam
  train, a speedboat, a classroom, books falling, a spaceship hum, a robot arm, a whisper), each built three ways:
  what `make` hands over (mean z 2.9, 1 of 12 clean with z >= 3 and no objection from the second ear), the automatic
  build (mean z 3.9, 8 of 12), and the parts an agent named (mean z 3.6, 8 of 12; after one or two turns on the
  printed lines 3.7, 7 of 12). Named parts won where the agent knew the structure and the catalogue had the material
  (a stampede from scattered hooves, three wolves, a thunderstorm in five layers); they lost where a part was found
  by the wrong word or there was no material (a classroom of children, a whisper, a spaceship).
- The second ear, run on 57 of the 60 builds above and on 27 left-out real recordings: it takes back up to 12 of their
  36 "sounds like it" verdicts; it objects to up to 24 of the 57 builds and to 6 of the 27 real recordings (a kettle
  heard as a train, an owl as a whistle), so an objection is a reason to listen, not proof.

These numbers say how much the listening models hear the words in the sound. They do not say that a person hears the
thing: see below.

## What it cannot do

- **The judge is also the target.** The tuner raises the listening model's fit, and a sound tuned to it can please
  it more than a person. With every glass recording left out, a glass breaking built from a breaking plate and a
  bursting bulb fit the words at z 6.7, the real recording of a glass at 3.7. Over the 30 known things the builds fit
  as well as the real recordings do (mean z 4.2 against 4.1). The second ear (the AudioSet tagger) is not tuned to and
  takes some of these verdicts back, but it hears only up to 8 kHz, knows 527 classes, and has no class for most made-up
  things. Only listening tells a good build from a lucky one.
- **The words are read as they are written.** The head of a compound now leads ("a fire alarm" is an alarm), but a
  part is still found by its words: "a lion growling" can come back as a snore, "a wooden shelf crashing" as a tree
  falling, "a train chugging" as a train horn. Read the `part N` lines and swap with `sound=<id>` (the next nearest are
  printed) before handing a sound over.
- **No material, no sound.** A build is as good as the nearest recordings. With every wind recording left out, "a
  strong wind" came from a jet and thunder and stayed a sketch (z 1.9); "someone whispering" has no whisper or voice
  to start from (z 1.5); "a force field humming" found no hum of its kind (z 2.1).
- **Rain, hail and things going by.** Rebuilt from other recordings, textures of many similar events and pulsing
  came out near their real designs (a purr, a swarm of bees, a boiling pot, a campfire, a helicopter first among the
  922; applause and a brook among the nearest three; with the catalogue's applause in reach, `earshot build applause`
  prints "a sketch: not clearly it yet", z +4.69 against +4.50, where 1.0 more is asked), but scattered close drops
  never became the dense far wash of rain or hail on a roof or a window (the real design at places 17 to 124), and
  `pass` on another motor did not make a car, a motorcycle or a jet going by (places 17, 52 and 192).
- **A voice is a stand-in.** A whale is an owl and a wolf an octave down; a baby's cry from a laugh moved up stays a
  sketch. earshot makes no speech and no singing.
- **A tone is a recorded beeper.** A car alarm, a modem or a robot's beeps are built from a reverse alarm, a phone
  and a horn. The model hears them as the words; whether they are the thing is for a person to say.
- **The plain design is often the best answer.** The design `make` would hand over is always one of the drafts; when no
  draft beats it after tuning, the automatic build hands it over as it is and says so. A build from named parts is
  kept as built, and the answer says when the plain design fits the words better.
- **It keeps the structure.** The tuner moves numbers only; which parts, how many layers and which structure are the
  agent's edit of the recipe.
