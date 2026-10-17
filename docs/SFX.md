# Sound for films: the bank, `earshot sfx`, `earshot kit`

earshot 0.3.0 ships its forge sounds as a bank that renders by id, aims a take at the picture, audits it without
models, and renders a whole film's kit into takes with measured sync points. Everything here is pure DSP (numpy /
scipy): no models, no network, no recorded audio. It is the sound engine of a code-made film: a film pipeline renders
from the bank in seconds and only ever searches (`earshot forge`) offline.

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

`earshot/forge/bank/*.json` (package data; the forge file is 275 KB, 64 KB gzipped): 186 sounds today.

| file | version | entries | what |
|---|---|---|---|
| `forge_v2.json` | `forge_v2r6` | 183 | every delivered forge v2 final (34 categories, 4-6 named members each) |
| `pluck.json` | `pluck1` | 3 | the tuned kalimba voices (`pluck/kalimba`, `pluck/kalimba_soft`, `pluck/kalimba_ring`) |
| `fit.json` (later) | `fit1` | - | recipes fitted from public-domain recordings (`fit/<role>/<name>`) |

`bank.VERSION` joins the file versions (`forge_v2r6+pluck1`); extra bank files load from `EARSHOT_BANK` (paths joined
by the OS path separator). Ids are `<category>/<member>`: the category is the public name (a user never needs the
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

## Knobs: dur, apex, pitch, level

`bank.render(id, seed=None, dur=None, apex=None, pitch=None, peak_db=-1.0, sr=48000, weight=None)`;
`earshot sfx render ID -o out.wav [--seed N] [--dur S] [--apex S] [--pitch NOTE|HZ] [--peak DB] [--weight W]`.
Knobs edit the recipe's own parameters (`earshot/forge/knobs.py`); nothing is resampled or time-stretched.
`meta['edits']` reports what each knob did and how close it landed.

- **seed**: `None` renders the delivered final (its own seed). Any other seed is a new take of the same member: the
  noise, jitter and micro-timing change, the character stays (the finals were picked on robust re-renders).
- **dur** (seconds, the file length): the whoosh family (whoosh_fast, whoosh_big, pro_whoosh_big, swish, swipe) scales
  rise, hold and fall together, so the pass keeps its shape; recipes with a length key (risers, downlifters, sub drops,
  shimmers, zips, glitches, typing runs, pro_logo_sting, the UI sounds' end gate) scale that key, render, measure and
  correct along a secant through the renders so far (at most 2 corrections, keeping the closest); the rest (impact,
  toggle, typing_key, logo_sting, pluck) keep their sound as it is. Then every dur lands exactly: a longer file is faded
  out to the length (raised cosine over a quarter of it, 10-150 ms), a shorter one padded with silence. A riser's
  length is exact by construction (`length` 1.0 -> a 1.000 s file that stops on its loudest 5 ms).
- **apex** (seconds to the loudest 5 ms, the whoosh family only): the pass engine's peak is analytic,
  `tp = rise * (|start_db| / 30) ** (1 / rise_k)`; rise is solved for the target, measured, corrected up to twice,
  then the closest render's peak is placed on the target by trimming the quiet approach (a head at least 20 dB under
  the peak, with a 3 ms fade-in) or padding silence before it. Measured: the closed form lands within about 12 ms on
  the fast family, but a broad fluttering pass's 5 ms peak moves non-monotonically with rise (whoosh_big and
  pro_whoosh_big swell: up to 50 ms for a 10 % change), so the search alone cannot promise 5 ms; with the placement
  every tested target lands within 2.5 ms (`edits['apex']`: `search_err_ms`, `fit`, `err_ms`, `ok`). Combine with dur
  for a move: `--dur 0.8 --apex 0.32` = a 0.8 s file whose loudest moment is 0.32 s in.
- **pitch** (`'C5'`, `'Eb6'`, `'F#4'`, `440`, `'440hz'`): the note goes into the recipe's pitch parameter and the
  parameters of the same body move by the same ratio:

  | category | parameter (co-scaled) | range |
  |---|---|---|
  | pluck | f0 | 130-2100 Hz |
  | notify / pro_notify | f0 (the head fundamental) | 220-3200 / 600-3200 Hz |
  | click, pro_click, toggle | pitch (the tink partials) | 1.2-9.5 kHz / 1.6-8 kHz |
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
  measured: `edits['pitch']['measured_hz']` / `cents_off` (pluck E5, notify G5, pro_sub_drop G1: 0 cents; a tap's body
  is a damped thump, within about 60 cents).
- **level**: `peak_db` sets the sample peak (-1 dBFS by default); `weight` scales it by 6 dB per doubling (0.5 = 6 dB
  down) and refuses a level over 0 dBFS. Loudness inside a family was matched in the forge (`render_final()` keeps
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
| silence | digital silence inside the sound (a note for glitches) | `polish.silence` |
| tail | the tail rises again by more than 6 dB (a note: a roll, a second event, a double member) | `polish.tail` |
| rising_glide | a tonal sound whose pitch climbs more than 4 semitones over at least 50 ms: the rising "synth whip" | `sfx.rising_glide` |
| stacked, pre_swell | for a hit (role hit / land): two separate attacks, or more than 100 ms of build before the hit | `sfx.layers` |

The glide detector tracks the strongest partial in 21 ms frames every 5 ms where it holds at least half of the frame's
80 Hz-10 kHz energy within +-3 semitones (a tone, also a fast smeared sweep; a noise band does not), splits the track
where it jumps more than 3 semitones, and reports the largest climb of its 5-frame median. On the 183 finals it flags
pro_zip up / up_hi / arc (+20 to +64 semitones), the zip and sweep leads of logo_sting zip_ping and pro_logo_sting
sweep_hit, and two bubble pops (pop/2, pro_pop/bloop: +6 semitones in 55-75 ms, a physical bubble); falling sweeps,
bells and plucks read 0. The layer heuristic reads a 25 ms RMS envelope (peaks within 6 dB of the loudest, 10 dB of
dip between them, in the first 0.5 s): two noise hits 60 or 120 ms apart read 2, a gated grain texture reads 1
(pro_impact short / air / swell: 1; deep: 2; rumble: 3); impact swell builds 220 ms before its hit.

Audit of the whole bank (each final at its own seed): 170 of 186 clean. Flagged: the three rising zips, the two
logo-sting leads, the two rising bubbles, typing_key light and laptop (comb ring), pro_hover_tick air and glass
(clicks), hover_tick detent and pro_typing_key soft (truncated), glitch dropout and logo_sting reverse_bloom, pro_shimmer
m3 (ring). The presets use none of them.

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
| `edits` | knobs for every take: `dur`, `apex`, `pitch`, `peak_db` (default -1), `weight` |
| `pitches` | one set of takes per note (or Hz) |
| `align` | the sync point a mixer puts on the cue: `onset`, `peak` (whoosh apexes), `end` (risers, swells); default per role |
| `max_s` | a take longer than this is faded out to it |
| `level_db` | the role's mix level relative to a take at its peak (for the mixer; the audition applies it) |
| `note` | why this sound |
| `null` | the role is not sounded: a renderer skips its cues |

`earshot kit render audio/kit.json --out audio/kit` writes 24-bit takes (`hit_1.wav`, `tick-crisp_2.wav`,
`bell_C5.wav`, `text_Cs5_2.wav`) and `kit.lock.json`:

```json
{"kit": 1, "earshot": "0.3.0", "bank": "forge_v2r6+pluck1", "sr": 48000, "name": "calm",
 "roles": {"hit": {"sound": "notify/m0", "align": "onset", "level_db": -3.0, "max_s": null,
   "takes": [{"file": "hit_1.wav", "seed": 1826701615, "pitch": null, "onset_s": 0.0, "peak_s": 0.0045,
              "end_s": 2.157, "dur_s": 2.3, "peak_db": -1.0, "lufs_m_max": -11.06, "sha1": "...", "key": "...",
              "checks": []}]},
   "riser": null}}
```

- **Deterministic.** The same kit renders the same takes. A take that fails the audit is rendered again on
  `seed + 1000 j` (up to 3 times) and the first clean one kept, as a sound designer discards a bad take; the lock then
  names `seed` and `planned_seed`. On the three presets 6 of 174 takes needed it (four key takes and a pitched ding
  that rang, a tap that stopped early); every preset take is clean.
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
the lockup with nothing layered on it. Each role's `note` says why its member was chosen. The choices follow what
earshot measured and heard:

- In the author's blind test of the v2 finals (one listener, loudness-matched, randomised), the made clicks, pops,
  impacts, toggles, hover ticks, swipes, swishes and risers were picked out; taps, struck bells (notify) and shimmers
  were taken for recordings. So taps and bells carry the natural roles, and designed members appear where the idiom
  is designed (crisp UI ticks, the gated hit, minimal's sub-led set).
- Every member passes the audit; members the audit flags (ringing keys, clicking airy ticks, rising zips) are out.
- No pitch-rising zip or whip, no shepard riser or downlifter, no logo-sting composite, no shimmer, no glitch, no
  tonal pop with the 8-bit risk; the hit is one sound with one attack and no build.

| role | calm (text-led films, the default) | crisp (energetic product films) | minimal (designed, effects only) |
|---|---|---|---|
| tick / tick.crisp | tap/glass / pro_tap/glass | tap/glass / pro_click/tink | pro_hover_tick/soft / pro_hover_tick/ping |
| tock | tap/knuckle | tap/knuckle | pro_tap/soft |
| hit | notify/m0 (one struck C5 bell) | pro_impact/short (one gated hit) | pro_impact/air (one sub-led hit) |
| land / land.light | pro_pop_in/deep / tap/soft | pro_pop_in/deep / pro_tap/knock | pro_pop_in/deep / pro_tap/finger |
| pop | pro_pop_in/airy | pro_pop_in/round | pro_pop_in/round |
| whoosh.short / .long | swipe/velvet / whoosh_big/swell | whoosh_fast/tight / whoosh_big/swell | whoosh_fast/heavy / pro_whoosh_big/swell |
| drop | off | pro_sub_drop/quick | pro_sub_drop/ref |
| key / .space / .return | typing_key thock / deep / deep | typing_key crisp / thock / clicky | pro_typing_key glass / tick / double |
| click / click.confirm | tap/finger / pluck/kalimba_soft C6 | pro_click/mouse / pro_notify/ding_2045 | pro_click/soft / pro_notify/ding_1822 |
| snap | pro_pop_in/knock | toggle/latch | pro_pop_in/knock |
| bell | pluck/kalimba C D E G A (octave 5) | pluck/kalimba C D E G A | pro_notify/ding_1822 F G A C D (octaves 6-7) |
| text | pluck/kalimba_soft C5 E5 G5 | pro_tap/soft | pro_tap/finger |
| riser / swell / suck | off | off | off |

Set `edits.pitch` on `hit` (calm) and `drop` (crisp, minimal) to the film's root. A later bank version may swap in
fitted sounds (`fit/<role>/<name>`) where they beat a forge member; a kit references them like any other id.

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

`earshot fit` turns a recording into a recipe; `add-fit` puts the fitted model into a bank file as
`fit/<role>/<name>` with its source:

```bash
earshot fit door_cc0.wav -o made/door_fit.py          # writes made/door_fit.json (the model) next to it
python -m earshot.forge.bank add-fit made/door_fit.json --id fit/hit/door --source source.json [--vary 0.2]
#   source.json: {"id": "freesound:123", "title": "...", "author": "...", "license": "CC0 1.0", "url": "..."}
```

The licence must be CC0 or public domain (`--allow-license` keeps an attribution licence; NC and ND licences are
always refused: a fit is a derivative and the bank ships inside an MIT package). The entry keeps the model (its fit
report dropped), seed, `vary` (take-to-take jitter), the source, the measured note of a tonal fit (`pitch_hz`, so the
pitch knob can tune it through the fit's own pitch / transpose / size knob, within an octave) and the usual meta.
Fit entries render through `earshot.forge.fit.render`. Default file: `earshot/forge/bank/fit.json`; `--bank` writes
elsewhere (load it with `EARSHOT_BANK`).

## Cost

Measured on the development laptop (AMD Ryzen 7 260, 2 threads, while other jobs kept the CPU near 100 %):

| step | time |
|---|---|
| `import earshot.forge.bank` + load the bank | ~3 s cold |
| first render in a fresh process | 10-45 s (numba compiles the filter kernels once per install, imports) |
| warm render + audit, median over the 186 finals | 0.5 s (UI sounds 0.05-0.3 s, whooshes 0.5-1.5 s, impacts and plucks 2-6 s) |
| `kit render`, whole preset, fresh process | calm 73 s (57 takes), crisp 61 s (59), minimal 25 s (58) |
| `kit render`, everything cached | 3-5 s (the process start; 0.1-0.6 s inside it) |

Render a film's sounds in one process and keep the takes: the lock and its cache are made for that.

## Limits

- The bank is the forge's sound, chosen and checked by measurement and one listener; it is not a library of
  recordings. A blind listener picked out several v2 categories (above); the presets lean on the ones that passed.
- Bit-exact re-renders hold on the machine that built the bank; another platform (numpy, scipy, numba versions) can
  differ by tiny amounts, which is why entries carry fingerprints with tolerances instead of hashes.
- Knobs leave the searched point: a far dur or pitch changes a sound's character, and only the model-free audit checks
  the edit. Ranges keep pitch inside each parameter's designed range.
- The whoosh engine picks its pan direction from the seed; a mixer mirrors a take for travel direction.
- No music and no ambience: the bank is effects and tuned notes.
