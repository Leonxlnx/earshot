# Sound for films: the bank, `earshot sfx`, `earshot kit`

earshot 0.3.0 ships its forge sounds, three tuned voices and 15 sounds fitted from public-domain recordings as a bank
that renders by id, aims a take at the picture, audits it without models, and renders a whole film's kit into takes
with measured sync points. Everything here is pure DSP (numpy / scipy): no models, no network, no recorded audio (a
fitted sound is stored as its modes, partials and band envelopes, not as samples). It is the sound engine of a
code-made film: a film pipeline renders from the bank in seconds and only ever searches (`earshot forge`) or fits
(`earshot fit`) offline.

```bash
earshot sfx list --kind whoosh                                     # what is in the bank
earshot sfx render whoosh_fast/crisp -o w.wav --dur 0.8 --apex 0.32 # one take aimed at a move
earshot sfx render pluck/kalimba -o e5.wav --pitch E5               # a tuned note
earshot sfx check w.wav                                             # the model-free audit of any sound effect
earshot kit init --preset calm -o audio/kit.json                    # a film's kit from a preset
earshot kit render audio/kit.json --out audio/kit                   # takes + audio/kit/kit.lock.json (cached)
earshot kit check audio/kit.json --out audio/kit                    # ids, knobs, the audit of every take
earshot kit audition audio/kit.json -o audition.wav                 # every take, role by role, at its level
```

Contents: [the bank](#the-bank) · [knobs](#knobs-dur-apex-pitch-level) · [the audit](#the-audit-sfx-check) ·
[kits](#kits) · [presets](#the-presets) · [Python, MCP](#python-and-mcp) · [fit entries](#fit-entries-from-public-domain-recordings) ·
[cost](#cost) · [limits](#limits)

## The bank

`earshot/forge/bank/*.json` (package data; the forge file is 275 KB, 64 KB gzipped; the fit file 0.5 MB): 201 sounds
today. `earshot doctor` reports the version and the counts.

| file | version | entries | what |
|---|---|---|---|
| `forge_v2.json` | `forge_v2r6` | 183 | every delivered forge v2 final (34 categories, 4-6 named members each) |
| `fit.json` | `fit1` | 15 | recipes fitted from CC0 recordings (`fit/<role>/<name>`, [below](#fit-entries-from-public-domain-recordings)) |
| `pluck.json` | `pluck1` | 3 | the tuned kalimba voices (`pluck/kalimba`, `pluck/kalimba_soft`, `pluck/kalimba_ring`) |

`bank.VERSION` joins the file versions (`forge_v2r6+fit1+pluck1`); extra bank files load from `EARSHOT_BANK` (paths
joined by the OS path separator, `os.pathsep`: `;` on Windows, `:` elsewhere; a missing or unreadable file is an error
that names it). Ids are `<category>/<member>`: the category is the public name (a user never needs the
recipe module behind it: `typing_key` renders with `recipes/keypress.py`, `pro_pop` with `pop.render_v2`,
`pro_typing_run` with `typing.render_run`).

**Faithful.** An entry is the final's schema-2 render block (`earshot.forge.schema`): recipe module, render function,
member, the complete parameters at full precision, seed, peak and palette gain. The build re-renders every pick and
compares it with its delivered WAV; one pick off by more than one 24-bit step (1e-6) and nothing is written:

```bash
python -m earshot.forge.bank build out/forge_v2/*/final -o earshot/forge/bank/forge_v2.json   # 183 / 183 within 1.2e-7
python -m earshot.forge.bank build --pluck -o earshot/forge/bank/pluck.json
python -m earshot.forge.bank check [IDS ...]          # re-render against the stored fingerprints
```

The forge's seed banks (`earshot/forge/seeds/`) are search starting points, not this: rounded parameters without
their render function; re-rendered from them, 34 of 94 checked finals come out as different sounds.

**Meta.** Each entry stores, measured on its render at -1 dBFS: `dur_s`, `onset_s` (the 1 ms envelope's first point
within 30 dB of its maximum), `peak_s` (the loudest 5 ms, 1 ms hop), `end_s` (the last 5 ms within 50 dB of the
loudest), `attack_ms`, `centroid_hz`, `lufs_m_max`, `channels` and a fingerprint (50 ms RMS frames in dB re the loudest,
to where the sound stays 60 dB down). A re-render on another machine is checked against it: length exact, peak within
5 ms, envelope within 0.5 dB (`bank.check`, `tests/test_bank.py`). On the build machine every entry re-renders
bit-exactly.

**Categories** (`earshot sfx list` has every member; `kinds` = the cue kinds / kit roles a category serves):

| category | kinds | for |
|---|---|---|
| click, pro_click | click, tick.crisp | cursor presses; bright designed UI ticks |
| hover_tick, pro_hover_tick | tick, tick.crisp | hover, rows passing, cursor blinks, counters |
| tap, pro_tap | tock, land.light, click, tick | taps and touches on a surface; the soft thump of a touch |
| toggle | snap, click | switches and lock-ins (press + latch) |
| pop, pro_pop, pro_pop_in | pop, land, snap | bubble pops; tonal pops; an element popping in (round low thump) |
| notify, pro_notify | bell, hit, text, click.confirm | struck bell chimes A4-E6; short clean dings a semitone apart |
| shimmer, pro_shimmer | shimmer | chimes ringing out, granular sparkle |
| whoosh_fast, whoosh_big, pro_whoosh_big | whoosh.short, whoosh.long, suck, swell | fast passes; big air transitions; the designed low sub swell |
| swish, swipe | whoosh.short | thin flicks and text slides; cloth and paper swipes |
| riser, pro_riser, downlifter | riser, drop | builds that stop on their loudest sample; energy falling away |
| pro_zip | zip | tonal sweeps across a cut (up / up_hi / arc rise in pitch) |
| impact, pro_impact | hit, land | a cinematic hit (3-6 s tails); a designed gated sub impact (0.7-1.3 s) |
| sub_drop, pro_sub_drop | drop | weight under a hit; the designed drop that blooms after the cut |
| logo_sting, pro_logo_sting | sting | composite logo reveals; launch gestures onto a tone |
| glitch, pro_glitch | glitch | glitch transitions; a designed stutter ending on the cut |
| typing_key, pro_typing_key, pro_typing_run | key, key.space, key.return, key.run | mechanical keys; designed key ticks; a whole run |
| pluck | bell, text, hit | a kalimba tine: tuned notes (native C5) |
| fit | the role in the id, plus its `kinds` | recordings fitted as code: clicks, a tick, pops, light touches, keys, a space bar, a snap, a bell, a whoosh, a knock |

## Knobs: dur, apex, pitch, level

`bank.render(id, seed=None, dur=None, apex=None, pitch=None, peak_db=-1.0, sr=48000, weight=None)`;
`earshot sfx render ID -o out.wav [--seed N] [--dur S] [--apex S] [--pitch NOTE|HZ] [--peak DB] [--weight W] [--sr R]`.
Knobs edit the recipe's own parameters (`earshot/forge/knobs.py`); nothing is resampled or time-stretched.
`meta['edits']` reports what each knob did and how close it landed.

- **seed**: `None` renders the delivered final (its own seed). Any other seed is a new take of the same member: the
  noise, jitter and micro-timing change, the character stays (the finals were picked on robust re-renders).
- **dur** (seconds, the file length): the whoosh family (whoosh_fast, whoosh_big, pro_whoosh_big, swish, swipe) scales
  rise, hold and fall together, so the pass keeps its shape; recipes with a length key (risers, downlifters, sub drops,
  shimmers, zips, glitches, typing runs, pro_logo_sting, the UI sounds' end gate) scale that key, render, measure and
  correct along a secant through the renders so far (at most 2 corrections, keeping the closest); the rest (impact,
  toggle, typing_key, logo_sting, pluck) keep their sound as it is. Then every dur lands exactly: a longer file is faded
  out to the length (raised cosine over a quarter of it, 10-150 ms; with apex as well the fade starts at least 5 ms
  after the apex, shortening to as little as 2 ms), a shorter one padded with silence. A riser's
  length is exact by construction (`length` 1.0 -> a 1.000 s file that stops on its loudest 5 ms).
- **apex** (seconds to the loudest 5 ms, the whoosh family only): the pass engine's peak is analytic,
  `tp = rise * (|start_db| / 30) ** (1 / rise_k)`; rise is solved for the target, measured, corrected up to twice,
  then the closest render's peak is placed on the target by trimming the quiet approach (a head at least 20 dB under
  the peak, with a 3 ms fade-in) or padding silence before it, in whole 1 ms steps of the peak measure (a shift off
  that grid can re-read a fluttering pass's near-equal maxima, 0.2 dB and 15 ms apart on whoosh_big/swell). With dur
  as well, the candidates are ranked on apex error plus the error of the tail after it, and the apex is measured again
  on the file after its fit to dur, so `edits['apex']` describes the delivered file. Measured: the closed form lands within about 12 ms on
  the fast family, but a broad fluttering pass's 5 ms peak moves non-monotonically with rise (whoosh_big and
  pro_whoosh_big swell: up to 50 ms for a 10 % change), so the search alone cannot promise 5 ms; with the placement
  every tested target lands within 2.5 ms, also a whoosh peaking 0.03-0.1 s before its end (59 whoosh-family
  targets, tails 0.1-0.12 s; `edits['apex']`: `search_err_ms`, `fit`, `measured`, `err_ms`, `ok`). Combine with dur
  for a move: `--dur 0.8 --apex 0.32` = a 0.8 s file whose loudest moment is 0.32 s in.
- **pitch** (`'C5'`, `'Eb6'`, `'F#4'` or `'Fs4'`, `440`, `'440hz'`): the note goes into the recipe's pitch parameter
  and the parameters of the same body move by the same ratio:

  | category | parameter (co-scaled) | range |
  |---|---|---|
  | pluck | f0 | 130-2100 Hz |
  | notify / pro_notify | f0 (the head fundamental) | 220-3200 / 600-3200 Hz |
  | click, pro_click, toggle | pitch (the tink centre), landed on the heard partial (below) | the parameter's 1.2-9.5 kHz / 1.6-8 kHz, moved to the heard partial |
  | hover_tick, pro_hover_tick | ping_f (tick_f), only members whose ping is on | 3-15 kHz |
  | tap, pro_tap, pro_pop_in | body_f (knock_f, glass_f) | 70-220 Hz |
  | pop / pro_pop | f0 (f_hi, f_lo, f_blip / + n_f0, n_f1) | 400-3500 / 280-1400 Hz |
  | sub_drop, pro_sub_drop | f1, the landing pitch (f0 keeps the glide ratio) | 28-62 Hz |
  | pro_zip | f1 (f0, fpk) | 80 Hz-12 kHz |
  | riser, pro_riser, downlifter | tf0, only members whose tonal layer is on | 60-4000 / 150-2500 Hz |
  | logo_sting / pro_logo_sting | root / g_root (root) | 300-2200 / 50-1800 Hz |
  | impact, pro_impact | body_f1 (body_f0, sub_f) | 34-75 Hz |
  | pro_whoosh_big | sub_f, only sub-led members | 25-120 Hz |
  | shimmer, pro_shimmer | key: the pitch class only (the register stays) | - |

  Noise sounds are refused with the reason (whoosh_fast, whoosh_big, swish, swipe, glitch, pro_glitch, typing_key,
  pro_typing_key, pro_typing_run), as are targets outside the range and members whose tonal layer is off. The render is
  measured: `edits['pitch']['measured_hz']` / `cents_off` / `ok` (within 50 cents; pluck E5, notify G5,
  pro_sub_drop G1: 0 cents; a tap's body is a damped thump, within about 60 cents). A take whose note missed carries a
  `pitch_off` warning in its checks, so a kit re-rolls or reports it; risers and impacts are not measured (`ok` null).
  Clicks and toggles: their `pitch` parameter is the centre of the tink partials and noise band, not the partial you
  hear (click/soft: 2454 Hz, heard 2903 Hz, +291 cents). Their note is the heard partial, measured when the bank was
  built (`meta.pitch_hz`, shown by `sfx list`; null for click/thock and toggle/micro, which have no clear partial and
  refuse a pitch), and a target is landed by measurement: the parameter moves by target / heard, the render is
  measured and corrected at most twice. pro_click members land within a few cents; some click and toggle members do
  not reach every note, and then say so (`ok` false, `pitch_off`).
- **level**: `peak_db` sets the sample peak (-1 dBFS by default, 0 at most: a level over 0 dBFS is refused, it
  would clip); `weight` scales it by 6 dB per doubling (0.5 = 6 dB down), with the same limit. `sr` is 48000 or one
  of 44100 / 88200 / 96000 (the finished take resampled), as in a kit. Loudness inside a family was matched in the forge (`render_final()` keeps
  that palette gain); a kit's `level_db` carries the mix level per role.

## The audit: `sfx check`

`earshot sfx check FILE [--category C] [--role R]`, `sfx.check(y, category=, role=)`. Measurements only; every bank
render runs it (`meta['checks']`). Warnings make `ok` false; notes are information.

| code | what | from |
|---|---|---|
| clicks, edge | isolated discontinuities inside the sound; a first / last sample off zero | `polish.clicks` |
| clipping | flat tops at the peak, samples over full scale | `polish.clipping` |
| dc | an offset above -40 dB re peak | `polish.dc` |
| ring | a metallic comb in the tail (z > 15) | `polish.ring` |
| truncated | a tail cut while still decaying, above -45 dB; a note for the categories whose stop is the design (riser, pro_riser, pro_zip, glitch, pro_glitch, pro_impact) and for any sound that ends on its loudest moment | `polish.truncated` |
| silent | nothing to hear: a sample peak under -60 dBFS, or less than 5 ms within 40 dB of the peak (the shortest bank sound holds 9 ms): digital silence, or a take faded to nothing | `sfx.live_ms` |
| silence | digital silence inside the sound (a note for glitches) | `polish.silence` |
| tail | the tail rises again by more than 6 dB (a note: a roll, a second event, a double member) | `polish.tail` |
| rising_glide | a tonal sound whose pitch climbs more than 4 semitones over at least 50 ms: the rising "synth whip" | `sfx.rising_glide` |
| stacked, pre_swell | for a hit (role hit / land): two separate attacks, or more than 100 ms of build before the hit | `sfx.layers` |
| pitch_off | a render with a pitch knob whose measured note is more than 50 cents off (bank renders and kit takes) | `knobs.pitch_warning` |

The glide detector tracks the strongest partial in 21 ms frames every 5 ms where it holds at least half of the frame's
80 Hz-10 kHz energy within +-3 semitones (a tone, also a fast smeared sweep; a noise band does not), splits the track
where it jumps more than 3 semitones, and reports the largest climb of its 5-frame median. On the 183 finals it flags
pro_zip up / up_hi / arc (+20 to +64 semitones), the zip and sweep leads of logo_sting zip_ping and pro_logo_sting
sweep_hit, and two bubble pops (pop/2, pro_pop/bloop: +6 semitones in 55-75 ms, a physical bubble); falling sweeps,
bells and plucks read 0. The layer heuristic reads a 25 ms RMS envelope (peaks within 6 dB of the loudest, 10 dB of
dip between them, in the first 0.5 s): two noise hits 60 or 120 ms apart read 2, a gated grain texture reads 1
(pro_impact short / air / swell: 1; deep: 2; rumble: 3); impact swell builds 220 ms before its hit.

Audit of the whole bank (each sound at its own seed): 185 of 201 clean. Flagged: the three rising zips, the two
logo-sting leads, the two rising bubbles, typing_key light and laptop (comb ring), pro_hover_tick air and glass
(clicks), hover_tick detent and pro_typing_key soft (truncated), glitch dropout and logo_sting reverse_bloom, pro_shimmer
m3 (ring). The presets use none of them. All 15 fitted sounds are clean at their own seed; on some new seeds the
small bell and the lubed key show a comb ring in the tail, as their recordings do (the presets use neither).

## Kits

A kit is a film's sounds by role. Roles are the shared vocabulary of a film's cue kinds and their variants:
`tick`, `tick.crisp`, `tock`, `hit`, `land`, `land.light`, `pop`, `whoosh.short`, `whoosh.long`, `suck`, `riser`,
`drop`, `key`, `key.space`, `key.return`, `click`, `click.confirm`, `snap`, `swell`, `bell`, `text` (a soft sound for a
text fragment resolving). Other role names are allowed; a renderer must know them.

`audio/kit.json` (schema 1):

```json
{"kit": 1, "name": "calm", "sr": 48000,
 "roles": {
   "hit":  {"sound": "notify/m0", "takes": 2, "align": "onset", "level_db": -3.0, "note": "why"},
   "whoosh.short": {"sound": "swipe/velvet", "takes": 3, "edits": {"dur": 0.6, "apex": 0.2}, "align": "peak"},
   "bell": {"sound": "pluck/kalimba", "pitches": ["C5", "D5", "E5", "G5", "A5"]},
   "riser": null},
 "off": {"riser": "why this role is off"}}
```

| key | meaning |
|---|---|
| `sound` | a bank id |
| `takes` | takes per role (default 1, at most 32); take k renders on `seed + k - 1` |
| `seed` | default: the sound's own, so take 1 is the delivered final |
| `edits` | knobs for every take: `dur`, `apex`, `pitch`, `peak_db` (default -1, at most 0), `weight` |
| `pitches` | one set of takes per note (or Hz); each pitch once (`C#5`, `Db5`, `Cs5` and 554.37 are one pitch) |
| `align` | the sync point a mixer puts on the cue: `onset`, `peak` (whoosh apexes), `end` (risers, swells); default per role |
| `max_s` | a take longer than this is faded out to it (0.01 s or more) |
| `level_db` | the role's mix level relative to a take at its peak, a number (for the mixer; the audition applies it) |
| `note` | why this sound |
| `null` | the role is not sounded: a renderer skips its cues |

`earshot kit render audio/kit.json --out audio/kit` writes 24-bit takes (`hit_1.wav`, `tick-crisp_2.wav`,
`bell_C5.wav`, `text_Cs5_2.wav`; one canonical label per pitch, so `Db5` is `Cs5` and 523.25 Hz is `C5`) and
`kit.lock.json`:

```json
{"kit": 1, "earshot": "0.3.0", "bank": "forge_v2r6+fit1+pluck1", "sr": 48000, "name": "calm",
 "roles": {"hit": {"sound": "notify/m0", "align": "onset", "level_db": -3.0, "max_s": null,
   "takes": [{"file": "hit_1.wav", "seed": 1826701615, "pitch": null, "onset_s": 0.0, "peak_s": 0.0045,
              "end_s": 2.157, "dur_s": 2.3, "peak_db": -1.0, "lufs_m_max": -11.06, "sha1": "...", "key": "...",
              "checks": []}]},
   "riser": null}}
```

- **Deterministic.** The same kit renders the same takes. A take that fails the audit is rendered again on
  `seed + 1000 j` (up to 3 times) and the first clean one kept, as a sound designer discards a bad take; the lock then
  names `seed` and `planned_seed`. On the three presets 4 of 191 takes needed it (three key takes and a pitched ding,
  all forge members; no fitted sound); every preset take is clean.
- **Validated like a render.** `kit check` refuses what `kit render` cannot use, with the same checks: knob values
  and ranges (a whoosh-only apex, a pitch the sound cannot take, `dur` 0.01-60 s), `peak_db` / `weight` over 0 dBFS,
  `max_s` under 0.01 s, a `level_db` that is no number, an unknown sample rate, and two takes that would share a file
  (compared without regard to case, as on Windows and macOS file systems).
- **Cached.** A take whose key (bank entry, seed, pitch, edits, max_s, sr, bank version, earshot version) and file
  hash match the existing lock is kept; changed roles re-render; files the new kit no longer names are removed (only
  files the old lock listed); `--force` renders everything.
- **Checked.** `earshot kit check` validates the file (ids, keys, knob names; close ids suggested) and audits every
  take with its role as context (a `hit` warns on stacked attacks and pre-swells). It notes restraint (riser / swell /
  suck sounded) and sounds the presets never use (rising zips, shepard risers and downlifters, logo-sting composites,
  shimmers, glitches, tonal pops), so the choice is visible.
- **Audition.** `earshot kit audition` writes every take role by role at its `level_db`, with an index `.txt`.

## The presets

`earshot kit init --preset calm|crisp|minimal` (`earshot/forge/kits/*.json`). All three sound every role except
`riser`, `swell` and `suck` (calm also keeps `drop` off): a reveal resolves on one element, and one clean hit carries
the lockup with nothing layered on it. Each role's `note` says what the sound is and what was measured. How the members
were chosen:

- **Every take passes the audit** (`earshot kit check`, all takes of all three presets); members the audit flags
  (ringing keys, clicking airy ticks, rising zips) are out.
- **Measured against the role.** Per role, the candidates (forge members and the fitted sounds) were read the same
  way: `earshot match` of the role's description (a calibrated z, and a forced choice against three synth-blip
  descriptions: 'a cheap synthesizer laser zap', 'a video game blip', 'an 8-bit chiptune sound'), `earshot realness`
  (a model's reading of "a real recording": information, not a quality score) and the model-free measures (length,
  centroid, attack, crest). A fitted sound replaced a forge member where it measured clearly closer: a higher match
  and forced-choice score at an equal or higher realness. Where the two measured about the same, the forge member
  stayed (calm's light touch tap/soft: z +1.88, realness 0.96, the fitted fingertip +2.07, 0.95; calm's key
  typing_key/thock +5.74 against the fitted keys' +4.65 to +5.14).
- **Designed where the idiom is designed**: crisp's tink, gated hit, sub drop and dings; minimal's sub-led set and its
  high-passed key ticks.
- **Restraint**: no pitch-rising zip or whip, no shepard riser or downlifter, no logo-sting composite, no shimmer, no
  glitch, no tonal pop with the 8-bit risk; the hit is one sound with one attack and no build.
- **Levels**: a role whose sound changed kept its loudness: `level_db` moved by the measured loudness difference of
  the two sounds at the same peak (momentary-max LUFS at -1 dBFS), rounded to 0.5 dB.

| role | calm (text-led films, the default) | crisp (energetic product films) | minimal (designed, effects only) |
|---|---|---|---|
| tick / tick.crisp | fit/tick/domino / pro_tap/glass | fit/tick/domino / pro_click/tink | pro_hover_tick/soft / pro_hover_tick/ping |
| tock | tap/knuckle | tap/knuckle | pro_tap/soft |
| hit | notify/m0 (one struck C5 bell) | pro_impact/short (one gated hit) | pro_impact/air (one sub-led hit) |
| land / land.light | pro_pop_in/deep / tap/soft | pro_pop_in/deep / fit/land.light/fingertip_book | pro_pop_in/deep / fit/land.light/fingertip_book |
| pop | fit/pop/mouth_plop | pro_pop_in/round | pro_pop_in/round |
| whoosh.short / .long | fit/whoosh.short/bamboo_swish / whoosh_big/swell | whoosh_fast/tight / whoosh_big/swell | whoosh_fast/heavy / pro_whoosh_big/swell |
| drop | off | pro_sub_drop/quick | pro_sub_drop/ref |
| key / .space / .return | typing_key/thock / fit/key.space/thock_space / typing_key/deep | fit/key/medium / fit/key.space/thock_space / typing_key/clicky | pro_typing_key glass / tick / double |
| click / click.confirm | fit/click/floss_lid / pluck/kalimba_soft C6 | fit/click/floss_lid / pro_notify/ding_2045 | pro_click/soft / pro_notify/ding_1822 |
| snap | fit/snap/switch_flip | fit/snap/switch_flip | fit/snap/switch_flip |
| bell | pluck/kalimba, C pentatonic G4-E6 (10 notes) | pluck/kalimba, C pentatonic G4-E6 | pro_notify/ding_1822, F pentatonic A5-G7 (10 notes) |
| text | pluck/kalimba_soft C5 D5 E5 G5 A5 | fit/land.light/finger_phone (untuned) | fit/land.light/finger_phone (untuned) |
| riser / swell / suck | off | off | off |

The replacements, measured (match z / forced choice against the synth-blip descriptions / realness; `_compare` of
the fit run, the forge members at their own seed):

| role | forge member before | fitted sound now | where |
|---|---|---|---|
| tick ('a soft quiet UI tick') | tap/glass -0.08 / 0.01 / 0.45 | domino +1.02 / 0.67 / 0.83 | calm, crisp |
| click ('a soft UI button click') | tap/finger +1.37 / 0.61 / 0.89; pro_click/mouse +1.03 / 0.35 / 0.06 | floss_lid +1.70 / 0.73 / 0.97 | calm, crisp |
| pop ('a soft natural pop') | pro_pop_in/airy +1.88 / 0.98 / 0.65 | mouth_plop +2.98 / 1.00 / 0.97 | calm |
| land.light ('a soft finger tap on wood') | pro_tap/knock +0.13 / 0.00 / 0.28; pro_tap/finger +0.69 / 0.01 / 0.16 | fingertip_book +2.07 / 0.99 / 0.95 | crisp, minimal |
| whoosh.short ('a short fast whoosh swish through the air') | swipe/velvet +2.01 / 1.00 / 0.97 (tagged 'a heavy thud') | bamboo_swish +5.32 / 1.00 / 0.99 | calm |
| key ('a single mechanical keyboard key press') | typing_key/crisp +4.48 / 1.00 / 0.95 | medium +4.65 / 1.00 / 0.99 (2.2 kHz centroid against 0.6) | crisp |
| key.space ('a spacebar press') | typing_key/deep +2.14 / 0.97 / 0.84; typing_key/thock +2.59 / 0.98 / 0.96 | thock_space +3.22 / 1.00 / 0.99 | calm, crisp |
| snap ('a toggle switch click') | pro_pop_in/knock +0.61 / 0.27 / 0.08; toggle/latch +2.04 / 0.97 / 0.50 | switch_flip +2.24 / 1.00 / 0.97 | calm, crisp, minimal |
| text (a soft touch for a resolving fragment) | pro_tap/soft, pro_tap/finger (realness 0.16-0.25) | finger_phone (tagged 'a soft UI button tap', realness 0.89) | crisp, minimal |

Set `edits.pitch` on `hit` (calm) and `drop` (crisp, minimal) to the film's root. The bell spans two octaves of
one pentatonic so a climbing progress motif finds a rung for each step; a tonal text role holds the five pentatonic
degrees of one octave, so a chain of fragments can step through whichever of the chord's tones the takes hold. crisp
and minimal resolve text on an untuned soft tap (a new take per fragment), which never clashes with a note.

## Python and MCP

```python
from earshot.forge import bank, kit, sfx, knobs
bank.VERSION; bank.list(category=None, kind=None); bank.get(id); bank.ids()
y, meta = bank.render('pro_sub_drop/ref', seed=3, pitch='G1')   # float32 (n,) or (n, 2), meta (above)
sfx.check(y, role='hit'); sfx.rising_glide(y); sfx.layers(y)
kit.init('calm', 'audio/kit.json'); lock = kit.render('audio/kit.json', 'audio/kit'); kit.check(...); kit.audition(...)
from earshot import api   # api.sfx_list / sfx_render / sfx_check / kit_init / kit_render / kit_check / kit_audition
```

MCP tools (`earshot mcp`): `sfx_list`, `sfx_render` (writes a WAV, returns the measured onset / peak / end and the
audit) and `kit_render` (a kit.json into takes + lock). They load no models.

## Fit entries (from public-domain recordings)

`earshot fit` turns a recording into a model ([docs/FIT.md](FIT.md)); a fitted model in a bank file is a sound like any
other, `fit/<role>/<name>`, with its source. The bank ships 15, fitted from CC0 recordings on Freesound (credits:
[docs/CREDITS_FIT.md](CREDITS_FIT.md)):

| id | what | model | length (take) | CLAP to the recording | kinds |
|---|---|---|---|---|---|
| `fit/click/floss_lid` | a small plastic click | sines | 0.13 s | 0.96 | click, tick.crisp |
| `fit/tick/domino` | domino pieces knocking together | sines | 0.14 s | 0.94 | tick |
| `fit/pop/mouth_plop` | a soft mouth plop | hybrid | 0.28 s | 0.96 | pop |
| `fit/pop/lip_pop` | a short pop | modal | 0.22 s | 0.95 | pop |
| `fit/land.light/fingertip_book` | light finger steps on a book | modal | 0.38 s | 0.94 | land.light, tock |
| `fit/land.light/finger_wood` | a finger tap on a wooden counter | modal | 0.34 s | 0.93 | land.light, tock |
| `fit/land.light/finger_phone` | a finger tapping a phone | hybrid | 0.33 s | 0.93 | land.light, text |
| `fit/whoosh.short/bamboo_swish` | a short whoosh | sines | 0.48 s | 0.94 | whoosh.short |
| `fit/key/medium` | one mechanical key, press and release | hybrid | 0.43 s | 0.94 | key |
| `fit/key/thock` | one low mechanical key press | modal | 0.45 s | 0.94 | key |
| `fit/key/lubed` | one mechanical key press | modal | 0.42 s | 0.98 | key |
| `fit/key.space/thock_space` | a mechanical keyboard space bar | sines | 0.39 s | 0.95 | key.space |
| `fit/snap/switch_flip` | a switch flip | sines | 0.19 s | 0.94 | snap, click |
| `fit/bell/small_bell` | a small bell struck once | sines | 1.66 s | 0.94 | bell |
| `fit/hit/bamboo_tok` | a bamboo hit | hybrid | 0.17 s | 0.97 | hit, tock |

Every entry renders through `earshot.forge.fit.render(model, seed, vary=vary, take=True)`: a new seed is a new take of
the same object (modes, partials and noise jittered by `vary`, 0.3 for all 15), it rings out instead of stopping where
the trimmed recording stopped, and it carries no DC offset (FIT.md, "Takes"). A take is 0.06-0.2 s longer than its
recording; `max_s` in a kit trims it.

**How they were chosen.** 46 CC0 recordings were fitted (`earshot fit --clap`) for the film roles. A fit ships when:
- its grade is high: CLAP to the recording >= 0.92 and mel distance within 2.5 dB of the model's own floor (two
  renders of the model);
- five takes (seeds 0-4, rendered as the bank renders them, at the shipped `vary`) pass the take check: no hard
  artifact beyond the recording's own (`polish.PolishRef`), every audit warning also on the recording itself, CLAP >=
  0.90 to the fit's render and within 0.06 of the fit's CLAP to the recording (compared over the recording's length:
  CLAP repeat-pads a short clip, so another length reads as another rhythm), excess energy within 1.5 dB of the fit's,
  loudness spread <= 2 LU. The check tries vary 0.3, then 0.15, then fresh noise only; a fit clean only with fresh
  noise does not ship (its takes barely differ);
- the source is CC0 1.0 and no title, author or page names a commercial sound library (a re-upload is not the
  recordist's to release).

15 of 46 pass, all at vary 0.3. Of the other 31, 20 grade good or fair; of the 11 high-grade ones, 7 make takes that
drift from the recording (CLAP), add an artifact the recording lacks (a comb ring, a rising glide) or spread over
2 LU, 3 are clean only with fresh noise, and one is a re-upload. Two shipped sounds, the small bell and the lubed
key, show the comb-ring warning on some takes, as their recordings do: in a kit, such a take is re-rolled on a new
seed (or kept with its warning when no re-roll is clean).

### Add your own fit

```bash
# 1. a CC0 / public-domain recording of ONE sound, trimmed close (a short fade-out at the cut is fine)
earshot fit door.wav -o made/door_fit.py --clap --takes --report   # model made/door_fit.json + quality + floor
# 2. hear and audit takes the way the bank will render them
python -c "from earshot.forge.fit import auto; from earshot.forge import sfx; import soundfile as sf; m = auto.load('made/door_fit.json'); [print(s, sfx.check(auto.render(m, seed=s, vary=0.3, take=True))['summary']) or sf.write(f'made/take{s}.wav', auto.render(m, seed=s, vary=0.3, take=True), 48000) for s in range(5)]"
# 3. add it to a bank file of your own (or, in a checkout, the default earshot/forge/bank/fit.json)
python -m earshot.forge.bank add-fit made/door_fit.json --id fit/hit/door --source source.json --vary 0.3 --bank my_bank.json
#    source.json: {"id": "freesound:123", "title": "...", "author": "...", "license": "CC0 1.0", "url": "https://..."}
EARSHOT_BANK=my_bank.json earshot sfx render fit/hit/door -o door_take.wav --seed 3
```

- `--takes` scores the candidates on seeds 0-2 (a fit meant to make takes); `--clap` lets CLAP help choose (loads
  CLAP, about 50 s). The report's `quality` against `floor` says how close the fit is (FIT.md, "Quality metrics").
- Lower `--vary` (0.15, 0) when takes drift: listen to the takes, and run `earshot sfx check` on them.
- The licence must be CC0 or public domain; `--allow-license` keeps an attribution licence (credit the source), and NC
  or ND licences are always refused: a fit is a derivative of its recording and the bank ships in an MIT package.
- The entry keeps the model (its fit report dropped), seed, `vary`, the source, the measured note of a tonal fit
  (`pitch_hz`: the pitch knob tunes it through the fit's own pitch / transpose / size knob, within an octave), the
  kinds (`--kinds`, default the role in the id) and the usual meta and fingerprint (`bank check` re-renders it).

## Cost

Measured on the development laptop (AMD Ryzen 7 260, 2 threads, while other jobs kept the CPU near 100 %):

| step (one process) | time |
|---|---|
| `import earshot.forge.bank` + load the bank | 1.3 s (3 s on a cold disk) |
| first render in the process (recipe import, numba kernels from their cache) | 10 s (25-45 s the first time after an install, while numba compiles) |
| the same render again, warm | 54 ms |
| warm render / render + audit: UI sounds (click, tap, pop-in, key, ding) | 40-65 ms / 80-160 ms |
| warm render / render + audit: whooshes, swipes, bells, sub drops | 0.1-0.45 s / 0.25-0.8 s |
| warm render / render + audit: pluck (3 s), impact (4.8 s) | 1.0-1.1 s / 1.4-2.2 s |
| a targeted whoosh (dur + apex: up to 3 renders + audit) | 1.9 s |
| render + audit, median over the 186 forge sounds and plucks (a cold process per batch) | 0.5 s |
| `kit render`, whole preset, fresh process | calm 34 s (64 takes), crisp 22 s (64), minimal 9 s (63), with the fitted sounds (inside the process, numba kernels cached; the first run after an install adds the compile) |
| `kit render`, everything cached | 3-5 s (the process start; 0.1-0.6 s inside it) |

Render a film's sounds in one process and keep the takes: the lock and its cache are made for that.

## Limits

- The bank is the forge's sound plus 15 fitted recordings, chosen and checked by measurement; it is not a library of
  recordings. No blind listening result for the v2 sounds is published: the README's blind test covers the v1 recipes
  (one listener, the author, 15 trials), and `eval/forge/blind_v2.py` builds the kit for a v2 test. Measurements (realness, a match against a role's description, the audit) are models and heuristics,
  not ears: a film's own listening pass still decides.
- Bit-exact re-renders hold on the machine that built the bank; another platform (numpy, scipy, numba versions) can
  differ by tiny amounts, which is why entries carry fingerprints with tolerances instead of hashes.
- Knobs leave the searched point: a far dur or pitch changes a sound's character, and only the model-free audit checks
  the edit. Ranges keep pitch inside each parameter's designed range.
- The whoosh engine picks its pan direction from the seed; a mixer mirrors a take for travel direction.
- No music and no ambience: the bank is effects and tuned notes.
