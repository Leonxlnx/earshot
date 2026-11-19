# earshot build: a sound the catalogue lacks, built with earshot as the ears

`earshot make` answers from a catalogue of 922 designs. For a thing it has nothing of it says so (`earshot makes no
whale song; wolf/howl is only the nearest sound it has`). `earshot build` is the workbench for that case: it drafts a
recipe (`earshot.recipe`) from the words, renders it, listens to it with earshot's listening model, moves its numbers
until it fits the words better, and says what is still off in lines an agent can act on.

```
earshot build "hail on a tin roof" -o hail.wav                # draft, render, hear, tune: hail.wav, hail.recipe.json
earshot build "hail on a tin roof" --recipe hail.recipe.json   # your edited recipe: render it, hear it, say what is off
earshot build "a stampede" --rounds 40 --without step/gravel rain --json
```

```python
from earshot.build import build, draft, hear, tune
r = build('hail on a tin roof', out='hail.wav')
print(r['summary'])                      # the lines below; r['heard'] holds every number
drafts = draft('a whale singing')        # one to three recipes from the words alone
heard = hear('hail.recipe.json', 'hail on a tin roof')
better = tune(drafts[0], 'a whale singing', rounds=20)['recipe']
```

The MCP server has the same as the tool `build` (`want`, `out`, `recipe`, `rounds`, `without`, `seed`).

No model generates audio here. The material is the catalogue's designs, most of them learned from public-domain
recordings; the recipe gives the structure; the listening model (CLAP, the `listen` extra) only judges.

## The loop an agent runs

1. `earshot build "X" -o x.wav`. It writes `x.wav` and `x.recipe.json`.
2. Read `fit:`: the z of the sound for the words, beside the z of the design `make` would hand over. The verdict is
   "sounds like it" when the sound is clean, its z is at least 3.0 and at least 1.0 over that design; otherwise "a
   sketch: not clearly it yet". `heard as:` names the designs it sounds most like and the words it is nearest to.
3. Read `still off:`. Each line names the field of the recipe to change ("layer 1 (rain/tin_roof) covers the rest: the
   fit rises to z +4.10 without it; lower its `gain` by about 6 dB or leave it out").
4. Change one thing in `x.recipe.json` and hear it again: `earshot build "X" --recipe x.recipe.json` (add `--rounds
   20` to let the tuner move its numbers again from there).
5. Hand it over as what `fit` says. A sketch is a sketch. A person should listen once: the listening model is the
   tuner's target, and a sound tuned to it can please it more than an ear.

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
A cookbook recipe of the thing is the first draft. `--without` leaves out designs (an id) or families (a name).

**hear(recipe, words)** renders and measures: clean or what the audit found; the length, where the hit is, how bright
(the median spectral centroid) and how dense (onsets a second), beside the design `make` would hand over; with the
listen extra also the fit (the same z as `earshot match`), what it is heard as, whether it is heard as recorded
(`earshot realness`, from the same embedding), a steady tone heard as a note, and per layer the fit with that layer
left out. When a design of the catalogue fits the words better than the build, it says which.

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
layers. A default build (20 rounds) of a short sound takes about a minute after the model is loaded.

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
heard as: wolf/howl, downlifter/synth_fall, cat/meow; words: 'a cat meowing' (z +3.1), 'being underwater' (z +2.5), 'an explosion' (z +2.4)
fit: z +4.09 for 'a whale singing' (the nearest design wolf/howl: z +2.71): sounds like it; recorded: p_real 0.75
tuned in 20 rounds: layer 0 (owl/hoot) pitch -9 -> -11.8 semitones; layer 0 (owl/hoot) stretch 1.6 -> 2; layer 1 (wolf/howl) pitch -9 -> -14.2 semitones; layer 1 (wolf/howl) gain -9 -> -11.4 dB; fit z +3.07 -> +4.09
still off: duller than wolf/howl (0.3 against 0.7 kHz): layer 0 is moved 11.8 semitones down, which darkens it: raise its `pitch` by a few semitones
next: edit whale.recipe.json, then: earshot build "a whale singing" --recipe whale.recipe.json
```

**Following a `still off` line, and undoing it.** Raising layer 0's `pitch` from -11.8 to -8.8 in `whale.recipe.json`
and hearing it again:

```
$ earshot build "a whale singing" --recipe whale.recipe.json -o whale.wav
whale.wav: recipe/a_whale_singing (10.00 s, peak at 1.72 s, -14 LUFS, clean) [built]
built from: owl/hoot (-8.8 semitones, bent through +0, -2 semitones, stretched 2 times); wolf/howl (-14.2 semitones)
heard as: wolf/howl, downlifter/synth_fall, cat/meow; words: 'a cat meowing' (z +2.8), 'a sustained glass harp tone' (z +2.6), 'an explosion' (z +2.6)
fit: z +3.71 for 'a whale singing' (the nearest design wolf/howl: z +2.71): sounds like it; recorded: p_real 0.74
still off: duller than wolf/howl (0.3 against 0.7 kHz): layer 0 is moved 8.8 semitones down, which darkens it: raise its `pitch` by a few semitones
next: edit whale.recipe.json, then: earshot build "a whale singing" --recipe whale.recipe.json
```

The fit fell from +4.09 to +3.71: the brightness line compares the sound with the nearest design (a wolf), not with a
whale, and a whale is darker than a wolf. Put the pitch back. A `still off` line is a measured difference and a
suggestion, the fit is the judge, and a person is the last judge.

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

These numbers say how much the listening model hears the words in the sound. They do not say that a person hears the
thing: see below.

## What it cannot do

- **The judge is also the target.** The tuner raises the listening model's fit, and a sound tuned to it can please
  it more than a person. With every glass recording left out, a glass breaking built from a breaking plate and a
  bursting bulb fit the words at z 6.7, the real recording of a glass at 3.7. Over the 30 known things the builds fit
  as well as the real recordings do (mean z 4.2 against 4.1). Only listening tells a good build from a lucky one.
- **The words are read as they are written.** "A fire alarm" took a campfire as a part because of "fire"; a reverse
  beeper carried the sound. Read `built from` before handing a sound over.
- **No material, no sound.** A build is as good as the nearest recordings. With every wind recording left out, "a
  strong wind" came from a jet and thunder and stayed a sketch (z 1.9); "someone whispering" has no whisper or voice
  to start from (z 1.5); "a force field humming" found no hum of its kind (z 2.1).
- **A voice is a stand-in.** A whale is an owl and a wolf an octave down; a baby's cry from a laugh moved up stays a
  sketch. earshot makes no speech and no singing.
- **A tone is a recorded beeper.** A car alarm, a modem or a robot's beeps are built from a reverse alarm, a phone
  and a horn. The model hears them as the words; whether they are the thing is for a person to say.
- **It can end below the nearest design.** With the vacuum cleaner left out, the build from a blender, a shredder and
  a hard drive fit worse (z 3.5) than the blender alone (4.3); "a snake hissing" built from air ended level with the
  snake's rattle. The answer then names the better design.
- **It keeps the structure.** The tuner moves numbers only; which parts, how many layers and which structure are the
  agent's edit of the recipe.
