# Design format: the JSON that mix.py renders

The soundtrack is data. One top-level design (`audio/sounddesign.json`) includes one fragment per category
(`audio/design/<category>.json`); `mix.py` places real recordings on the picture's cue events, mixes three buses (sfx,
amb, music) and masters. Nothing is synthesized: the only processing is trim, fade, gain, pan, EQ, resampling (pitch),
a gain envelope, Rubber Band tempo stretch of the music, summing and mastering.

Templates: `templates/sounddesign.json` (top level), `templates/design_fragment.json` (every feature; its file names
are placeholders, so it renders with `templates/cues.json` once each one is replaced by a real recording from your
manifests), `templates/cues.json` (the cue file). `python` below = the interpreter `{{PY}}` described in SKILL.md.

```
python -I <skill>/scripts/mix.py --root . --design audio/sounddesign.json --cues out/cues.json --trace out/qa/adds.json
python -I <skill>/scripts/mix.py --only keyboard          # solo render of the fragments whose path contains "keyboard"
python -I <skill>/scripts/mix.py --only keyboard --dump out/audio-review/keyboard/dump   # every sample as a WAV
```

Every relative path (design, cues, outputs, each sample's `file`) resolves against `--root`. Keys starting with `_` are
comments (write the reason for every non-obvious number there: `_doc`, `_about`, `_buses_doc`); `mix.py` ignores
unknown keys everywhere except inside `buses` (each key there must be `sfx`, `amb` or `music`).

## Contents
1. Top level and includes
2. The cue file
3. samples
4. slices
5. pools
6. rules (cue-driven placements)
7. place (free placements)
8. ambience
9. music
10. buses and master
11. What mix.py writes
12. qa_ok: acknowledging a deliberate exception
13. Worked examples from a shipped film

## 1. Top level and includes

| key | meaning |
|---|---|
| `duration` | film length in seconds (default: cue `total` / `fps`) |
| `cues` | default cue file (the `--cues` flag wins; default `out/cues.json`) |
| `include` | fragment paths; their `samples`, `slices`, `pools` are merged (a name defined twice is an error), `rules`, `place`, `ambience` are concatenated, `music` is taken when the top level has none |
| `samples`, `slices`, `pools`, `rules`, `place`, `ambience`, `music` | as below (allowed at the top level too) |
| `buses`, `master` | read from the top-level file only |

Name everything with the fragment's prefix (`kb.`, `ui.`, `mo.`, `am.`, `mu.`): fragments are built by different agents
in parallel and must never collide. Variation (cycle order, jitter, random pitch) is keyed on rule names, so a solo
render (`--only`) plays exactly the same sounds as the full mix.

## 2. The cue file

`{fps, total, events: [{f, kind, id?, weight?, pan?, variant?, dur?, apexFrac?, pitch?}], acts?, cue?}` or a list whose
first element is that object. This is Cinetic's `out/cues.json` format (`scripts/export-cues.ts`), so a Cinetic film
plugs in directly (`docs/INTEGRATION.md`). `f` is the absolute frame the sound belongs on (fractional is fine).
Compute cue frames from the picture code (contact frames, 50 % pop frames, velocity peaks); typed-in sync points land
8-11 frames off in practice. A design with no rules needs no cue file.

## 3. samples

`name -> {file, start?, end?, gain_db?, norm_peak_db?, mono?, fade_in?, fade_out?, eq?, pitch?, glide?, env?}`:
a trimmed piece of a recording.

| key | default | meaning |
|---|---|---|
| `file` | required | the recording (any format ffmpeg reads; decoded to 48 kHz float stereo, cached) |
| `start`, `end` | 0, end of file | the trim in source seconds. Start 3-5 ms before the attack; end after the decay, before the next event |
| `mono` | false | fold to mono (point sources: keys, clicks, steps). Keep stereo for whooshes, beds, wide impacts |
| `eq` | none | `{hp, hp_order (2), lp, lp_order (2), peq: [[f0, q, gain_db], ...], lowshelf: [f0, gain_db], highshelf: [f0, gain_db]}` (Butterworth HP/LP, RBJ biquads) |
| `pitch` | 0 | semitones by resampling (tape-style: pitch AND length change together). +2..+6 makes a hand-sized sound miniature; keep tuned material within +-2 |
| `glide` | none | `[from, to]` semitones: a varispeed glide across the sample (band-limited, no aliasing) |
| `env` | none | `[[t_from_sample_start, dB], ...]` piecewise-linear gain inside the sample (shape a riser, duck a tail) |
| `norm_peak_db` | none | normalise the processed sample's peak to this level (makes a pool even before rule gains) |
| `fade_in`, `fade_out` | 0.002, 0.012 | raised-cosine fades in seconds, applied after normalisation |
| `gain_db` | 0 | static gain after everything |

Processing order: mono -> eq -> pitch -> glide -> env -> norm_peak_db -> fades -> gain_db. `reverse` is refused (a
reversed recording is still real, but the design keeps to what the recording is).

## 4. slices

`name -> {file, start?, end?, delta?, min_gap?, gate_db?, pre?, len?, stop_before_next?, pick?, max?, ...sample opts}`:
cuts a recording of many hits (typing, clicks, steps, dominoes) into single hits at its detected onsets. The slices are
named `name#0`, `name#1` ... in the order they are kept.

| key | default | meaning |
|---|---|---|
| `delta` | 0.07 | onset detector sensitivity (librosa, hop 128, backtracked); lower finds softer hits |
| `min_gap` | 0.06 | seconds between onsets |
| `gate_db` | -18 | drop onsets whose first 30 ms peak is this far below the loudest (handling noise, far keys) |
| `pre` | 0.004 | seconds kept before each onset |
| `len` | 0.16 | maximum slice length |
| `stop_before_next` | true | end each slice `pre` before the next onset (never two hits in one slice) |
| `pick` | all | list of onset indices to keep, in this order (after `pick`, `name#0` is the first picked) |
| `max` | all | keep at most this many |
| sample opts | | `mono`, `eq`, `pitch`, `glide`, `env`, `norm_peak_db`, `gain_db`, `fade_in` (0.002), `fade_out` (0.02) |

Always `--dump` a slice group and read `_samples.json` (duration, peak, attack, head/tail level per slice) before
using it; qa.py's edges section flags slices that start mid-sound or are cut mid-decay.

## 5. pools

`name -> [sample names]`; `"group#*"` expands to every slice of a slice group (sorted by index). An unknown name or an
empty pool is an error. A rule plays from a pool: one name = always that sound; several = variation.

## 6. rules (cue-driven placements)

`[{name, match, pool, mode?, gain_db?, weight_db?, weight_ref?, jitter_db?, pitch?, pitch_ramp?, gain_ramp?, pan?,
pan_scale?, align?, offset?, bus?, no_cue?, cycle_start?}]`. Every cue event that matches a rule gets a sound from the
rule's pool; several rules matching one event LAYER it (a key: thock + bottom-out + tiny motif note).

| key | default | meaning |
|---|---|---|
| `name` | rule index | stable name: the hash key for jitter / pitch / hash picks (renaming a rule changes its variation) |
| `match` | {} | `kind` (regex, full match), `id` (regex, search), `from` / `to` (film seconds, `[from, to)`), `variant` (exact) |
| `pool` | required | the pool to play |
| `mode` | cycle | `cycle`: round-robin, never the same sound twice in a row (`cycle_start` = first index); `hash`: picked per event by a hash of id + frame + rule name; `fixed`: always the first |
| `gain_db` | 0 | level |
| `weight_db` | 0 | dB per doubling of the event's `weight` vs `weight_ref` (default 1.0): bigger events play louder |
| `jitter_db` | 0 | +- random level per event (hashed: deterministic) |
| `pitch` | [0, 0] | `[lo, hi]` random semitones per event (hashed) |
| `pitch_ramp`, `gain_ramp` | none | `[a, b]` added across the matched events in time order: a climbing build, a decrescendo |
| `pan` | "cue" | `"cue"` = the event's `pan` x `pan_scale`, or a number -1..1 |
| `align` | attack | which point of the sound lands on the cue: `attack` (first ms within 20 dB of the peak; impacts, clicks, keys), `peak` (5 ms envelope maximum; whooshes), `end` (risers into a hit), `start` |
| `offset` | 0 | seconds added to the cue time (a release 85 ms after a press; a tail that trails the move) |
| `bus` | sfx | `sfx`, `amb` or `music` |
| `no_cue` | false | the placement does not mark the event as sounded (sweeteners; the A/V audit ignores it) |

Cue events no rule sounds are listed as `unsounded_events` in the report.

## 7. place (free placements)

`[{sample, at, gain_db?, pan?, align?, offset?, pitch?, bus?}]`: sounds with no cue event (a crowd bed, a lift before
a logo, a one-off sweetener). `at` is film seconds; `align` as in rules.

## 8. ambience

`[{name?, file, src_start?, src_end?, at, until, fade_in?, fade_out?, gain_db?, eq?, width?, auto?}]`: beds on the amb
bus. `src_start..src_end` is the clean window of the recording (no voices, dogs, bumps); if the bed needs more than the
window it is looped with a 1.5 s equal-power crossfade (the report says `looped`). `fade_in` / `fade_out` default 1 s.
`width` scales the side signal (0 = mono, 1 = as recorded, 1.15 = a little wider). `auto` = `[[film t, dB], ...]` rides
inside the bed (duck birds under a run, open the room on a cut).

## 9. music

`{file, src_bpm, target_bpm?, stretch?, gain_db?, eq?, segments: [{src_from, src_to, at, fade_in?, fade_out?,
fade_in_at_start?, tail_fade?, gain_db?, label?}]}`

| key | default | meaning |
|---|---|---|
| `src_bpm` | required | the track's measured tempo (arrange.py) |
| `target_bpm` | 120 | the picture's tempo |
| `stretch` | rubberband | `none` plays the source untouched (use it when the track is within ~0.2 % of the film tempo); otherwise the whole file is time-stretched by target/src with Rubber Band (keep the ratio within about +-6 %) |
| `segments[].src_from`, `src_to` | | ORIGINAL source seconds (before any stretch), on the track's downbeats |
| `segments[].at` | | film seconds where `src_from` lands; nudge by a few ms so the measured downbeat transient sits on the picture's downbeat |
| `fade_in`, `fade_out` | 0.008, 0.06 | the segment starts `fade_in` before `src_from` (a pre-roll, so the downbeat's own attack is kept) and fades out over `fade_out` after `src_to` |
| `fade_in_at_start` | true | false = no fade at the head (a hard cut on purpose) |
| `tail_fade` | none | seconds of a longer musical fade over the end of the segment (the last chord rings out) |
| `gain_db`, `label` | | per-segment level; a label that says which bars and why |

Edit at bar lines: a splice from bar n's end to bar m's start works when the harmony continues (V -> I, the same
chord) and the energy of the next bar is what the picture wants. `arrange.py` drafts the segment list; check every
splice (qa.py edges section, earshot timeline) and write the bars and the reason in each `label`.

## 10. buses and master

`buses: {sfx|amb|music: {gain_db?, eq?, auto?}}`: after everything is placed, each bus gets eq, then the `auto` ride
(`[[film t, dB], ...]`, piecewise linear), then `gain_db` (the music bus first gets `music.eq` and `music.gain_db`).
Rides shape the arc: the music dips under typing, lifts into the drop, settles under the logo. Put each step on a
musical splice or under a transient, never in the middle of a sustained note.

`master: {lufs?, ceiling?, no_glue?, no_comp?, mono_hz?}` (defaults -14, 0.77, true, false, 120): `master.py` high-passes
at 28 Hz, folds below `mono_hz` to mono, compresses gently (2:1 above -14 dBFS, unless `no_comp`), then runs loudness
passes with a 4x-oversampled true-peak limiter to `ceiling` (linear), a 3 ms fade-in at sample 0, a fade onto the last
sample and 480 zero samples at the end. The gate passes at |LUFS - target| <= 0.5 and true peak <= -1.5 dBTP.

## 11. What mix.py writes

- `--out` (default `out/audio/soundtrack.wav`): the master, 48 kHz 24-bit; the unmastered sum `premaster.wav` next to
  it. With `--no-master` or `--only`, `--out` is the unmastered sum itself (solo default
  `out/audio-review/solo/<name>.wav`).
- `--stems` (default `out/audio/stems`): `sfx.wav`, `amb.wav`, `music.wav` (pre-master, they sum to the premaster).
- `--report` (default `out/qa/mix_report.json`): `placements` (t, event, kind, rule, sample, gain_db, pitch, pan,
  align), `unsounded_events`, `buses` (peak / RMS), `ambience` (looped, body level), `music_segments`, `files_used`
  (every recording that is audible: the input of `credits.py` and `audit.py --report`), `premaster_peak_dbfs`;
  `master.json` (loudness gate) next to it.
- `--cues-out` (default `out/qa/cues_sounded.json`): the cue events that received an sfx-bus sound (for an A/V audit).
- `--trace FILE`: every `Mix.add()` (bus, start sample, length, gain, pan, sample / name / label, rule, kind): qa.py
  rebuilds every placement from it to attribute clicks, masking and harshness to a rule and a sample.
- `--dump DIR`: every sample and slice as WAV + `_samples.json` (duration, peak, RMS, attack, head/tail level), then exits.

## 12. qa_ok: acknowledging a deliberate exception

A sample, rule, free placement or music segment may carry `"qa_ok": "reason"` (mix.py ignores it). qa.py then lists
that item's edge / sync / audibility / harshness finding as acknowledged instead of a warning: e.g. a riser choked
25 ms before a pop on purpose, a chime that rings past its cue, a hard music cut into a quiet breakdown. Use it only for
decisions a critic has already confirmed, and say why.

## 13. Worked examples from a shipped film

From the Clawd film (60 s, 264 placements, real recordings only), trimmed for reading.

A key thock from a recording of single presses, shaped for a deep "thocky" board:
```json
"kb.thock.1": {"file": "audio/assets/keyboard/fs-789628_keychron-linear-individual-keys.ogg", "start": 1.168,
  "end": 1.598, "mono": true, "eq": {"hp": 115, "hp_order": 6, "peq": [[190, 0.8, 3.0], [380, 0.9, 3.0],
  [3300, 1.3, -1.5]], "highshelf": [5000, -4.5]}, "norm_peak_db": -10.0, "fade_in": 0.004, "fade_out": 0.16}
```

Typing on 16th notes: one key per typed pixel, climbing a semitone across the run, never the same key twice:
```json
{"name": "kb.type", "match": {"kind": "pop", "id": "^genesis:type"}, "pool": "kb.type", "mode": "cycle",
 "jitter_db": 1.5, "pitch": [1.0, 2.0], "pitch_ramp": [0.0, 1.0], "pan": 0.0, "align": "attack"}
```

Dominoes sliced into single clacks, 13 picked after measuring every slice (--dump):
```json
"ty.dom": {"file": "audio/assets/toy/fs-329098_domino-pieces-hit.ogg", "start": 0.2, "end": 16.4, "pre": 0.004,
  "len": 0.3, "min_gap": 0.2, "gate_db": -12, "pick": [2, 10, 11, 3, 17, 22, 23, 21, 24, 9, 16, 19, 4], "mono": true,
  "norm_peak_db": -3, "fade_out": 0.04, "eq": {"hp": 250, "hp_order": 4}}
```

A stacking flurry that ends ON each row's lock (aligned by its end):
```json
{"name": "ty.row.flurry", "match": {"kind": "pop", "id": "^swarm:row:"}, "pool": "ty.p.row.flurry", "mode": "cycle",
 "gain_db": -20.5, "gain_ramp": [-3, 2], "pitch_ramp": [0, 4], "jitter_db": 1.5, "pan": 0.12, "align": "end", "offset": 0.004}
```

Birdsong ducked under the hero's run (an `auto` ride inside the bed):
```json
{"name": "am.meadow_birds", "file": "audio/assets/ambience/fs-677535_woodlark-field-spring.ogg", "src_start": 15.5,
 "at": 28.15, "until": 52.2, "fade_in": 1.05, "fade_out": 0.8, "gain_db": 2.4, "width": 0.38,
 "eq": {"hp": 300, "lp": 13000, "lowshelf": [700, -5.0]},
 "auto": [[28.15, 0.0], [31.7, 0.0], [32.15, -2.5], [32.32, -10.0], [32.78, -10.0], [32.95, -2.5]]}
```

A music edit at bar lines (a 120 BPM track, no stretch), each splice explained:
```json
"segments": [
 {"label": "open bars 1-6 (E -> B), film = src - 0.5", "src_from": 0.5, "src_to": 12.5, "at": 0.0, "fade_in": 0.01, "fade_out": 0.05},
 {"label": "theme bars 17-24 (entrance on E: V -> I across the splice)", "src_from": 32.5, "src_to": 48.455, "at": 11.995, "fade_in": 0.01, "fade_out": 0.045},
 {"label": "drop bars 67-72 (E-major tutti) on the hero reveal", "src_from": 132.489, "src_to": 143.465, "at": 39.995, "fade_in": 0.008, "fade_out": 0.01},
 {"label": "coda bars 83-87 (final E on 59.0)", "src_from": 164.489, "src_to": 173.498, "at": 50.981, "fade_in": 0.01, "fade_out": 0.01, "tail_fade": 0.7}
]
```

Bus rides with their reasons (the drop hits, then a diminuendo lets the story read over the tutti):
```json
"_buses_doc": "the opening is lifted (+4.5 music/sfx) so it is soft but present; the DROP hits at -1.5 for its first beat, then -3.5 by 42.45 and -4.5 by 44.3 so the unfold/crowd/stacking story reads; a 2.5 dB duck 47.9-48.7 under the giant lock",
"music": {"auto": [[0, 0], [1.9, 0], [2.1, 4.5], [11.9, 4.5], [11.99, 0], [27.95, 0], [28.0, 5.5], [39.99, 5.5], [40.0, -1.5], [40.5, -1.5], [42.45, -3.5], [44.3, -4.5], [47.9, -4.5], [47.98, -7.0], [48.3, -7.0], [48.7, -4.5]]}
```
