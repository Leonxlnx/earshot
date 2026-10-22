# Integration: earshot inside a Cinetic film

Cinetic (`~/.claude/skills/cinetic`) directs and renders launch films from code. Its Step 5 builds the soundtrack
(`scripts/audio/score.py`) from a beat-locked timeline. Two routes give that soundtrack better sounds: an **earshot
sound kit** (the first section: forge sounds rendered from earshot's bank, the route cinetic 1.1 takes by default when
earshot is installed), and a **real-recordings soundtrack** from `skills/real-sound` (sections 1-4, which use Cinetic
exactly as it was before 1.1). Either way earshot gives Cinetic's review loop ears (section 3).

## cinetic uses earshot

The effects of a cinetic film come from an earshot kit; cinetic keeps the conductor (timing, typing plan, levels,
ducks, arc, master, report, stems) and earshot supplies the voices. Rendering a kit is pure DSP: no models, no torch
import, seconds per film once cached.

```
src/sync.ts --export-cues--> out/cues.json ---------------------------------+
audio/score.json  "sfx": {"engine": "auto", "kit": "audio/kit.json"} ------+--> score.py --> master.py --> soundtrack.wav
audio/kit.json --scripts/audio/kit.py--> earshot kit render --> audio/kit/*.wav + kit.lock.json --^
```

1. **The kit.** `audio/kit.json` starts from a preset (`earshot kit init --preset calm|crisp|minimal|launch`; calm
   is the default for text-led films, launch for films cut to music) and is edited per film: the lockup hit tuned to
   the film's root (`edits.pitch`), another member for a role, a fitted sound. Roles are cinetic's cue kinds and their
   variants (below); `null` roles are not sounded.
2. **The bridge.** cinetic's `scripts/audio/kit.py` finds earshot (importable in cinetic's Python, `sfx.python` in
   `score.json`, `EARSHOT_PYTHON`, or `earshot` on PATH), runs `earshot kit render audio/kit.json --out audio/kit` in
   one process (cached: an unchanged kit re-renders nothing) and keeps the takes in `audio/kit/`, so the film rebuilds
   on a machine without earshot.
3. **Placement.** score.py plays one take per event (a new take per event by hash; typing cycles through the takes),
   placed by the sync point the lock measured for the role's `align`: the take's onset on the cue frame, a whoosh's
   loudest 5 ms on its apex, a riser's end on the hit (or its climax, for a riser or swell aligned on its `peak` as
   the launch preset's are: their 0.4 members fall after the climax). `level_db` and the event weight set its level;
   a cue whose role is `null` is skipped with a report line. A role with the 0.4 mix keys (the launch preset) asks for more: its
   `offset_ms` moves the sync point off the frame (`kit.place`), its `bed_db` sets the take against the music under
   the cue (`kit.bed_level` + `kit.gain_db`, instead of `level_db`), its `min_gap_ms` spaces a run (`kit.space_run`),
   and the lock's `mix.stop_down` asks for a music dip before hits rather than a duck after them
   ([docs/SFX.md](SFX.md#mix-keys-04)). A renderer that ignores them still gets valid takes and `level_db`.
4. **Fallback.** earshot render -> the vendored `audio/kit/` takes -> cinetic's own synth voices with a loud warning.
   The build never fails only because earshot is missing.
5. **Listening.** `scripts/audio/listen.py` runs `earshot timeline --cues`, `sheet` and `critique` in one process and
   writes `out/qa/listen.json`.
6. **The lockup** is a stop-down and one clean element. `earshot kit check` warns when the hit role has two attacks or
   a build before the hit, and when any role's pitch climbs (the rising synth whip); score.py's stack guard warns on a
   second designed effect within 3 frames of the lockup or payoff.

| cinetic cue (kind, variant) | kit role | align |
|---|---|---|
| tick / tick crisp | `tick` / `tick.crisp` | onset |
| tock | `tock` | onset |
| hit | `hit` | onset |
| land / land light | `land` / `land.light` | onset |
| pop | `pop` | onset |
| whoosh (short / long move) | `whoosh.short` / `whoosh.long` | peak |
| suck, riser, swell | `suck`, `riser`, `swell` (off in calm, crisp, minimal; launch sounds riser and swell, both aligned on the peak) | end |
| drop | `drop` | onset |
| key / key space / key return | `key` / `key.space` / `key.return` | onset |
| click / click confirm | `click` / `click.confirm` | onset |
| snap | `snap` | onset |
| bell (motif notes, progress steps) | `bell` (one take per pitch) | onset |
| a text fragment resolving | `text` | onset |

What earshot guarantees the bridge: the role vocabulary, kit and lock schema 1 (docs/SFX.md#kits), the CLI flags
above, deterministic takes, and a lock that names the bank version (a new bank version re-renders the kit, because
it is part of every take's cache key). The bank, knobs, audit and presets are documented in
[docs/SFX.md](SFX.md).

## 1. What Cinetic produces that the soundtrack needs

| Cinetic output | content | used by real-sound / earshot as |
|---|---|---|
| `out/cues.json` (`npx tsx scripts/export-cues.ts`) | `{fps, bpm, total, acts, cue, events: [{f, kind, weight, pan, pitch?, apexFrac?, dur?, variant?, id?}]}`, sync frames computed from the picture code | **the cue file, unchanged**: `mix.py --cues out/cues.json` (mix.py's cue format is this format). Rules match `kind` (regex), `id` (regex), `variant`, a time window; `weight` drives `weight_db`; `pan` drives `pan: "cue"` |
| `src/timeline.ts` (`FPS`, `BPM`, `TOTAL`, `ACT`, `CUE`) | the grid | `TOTAL / FPS` = the design's `duration`; `BPM` = `arrange.py --film-bpm`; `ACT` (exported as `acts`) = qa.py's per-act loudness |
| `audio/score.json` | key, bpm, one chord per bar, sections (`intro / build / groove / drive / breakdown / resolve / hold`), drops + silences, `payoff`, signature tick/tock/motif, typing mode, mix, `master.lufs` | the musical intent: sections -> `arc.json` for `arrange.py`; `payoff` / drops -> `drop_film_bar`; key and chords -> which tuned recordings to source and whether a splice keeps the harmony; `master` -> the design's `master` |
| `TREATMENT.md` / `BRIEF.md` | the brand's three adjectives, the personality row (`references/sound.md` §5.5a), the beat sheet with a sound column | the SOUND_BRIEF's concept and picture map |
| `public/audio/soundtrack.wav` (written by `npm run audio`) | the master Cinetic's `render.sh --audio` muxes | **replaced by `mix.py`'s `out/audio/soundtrack.wav`**, passed to `render.sh --audio` (render.sh takes any WAV; keeping it out of `public/` keeps the premaster and stems out of the Remotion bundle) |
| `out/stems/` | stems for `av-audit.py --stems` | `mix.py --stems out/stems` writes `sfx.wav`, `amb.wav`, `music.wav`: av-audit picks `sfx*.wav` as the effects and `music.wav` as the music bus |

### score.json sections -> an arrange.py arc
One number per film bar (dB relative to the track's loudest sections): `intro` -14 (rising to -10), `build` -10 -> -4
across its bars, `groove` -6, `drive` -2..0, `breakdown` -10, `resolve` -8 -> -14 over its bars, `hold` / `tail` -20.
`payoff` (usually the drop's bar) becomes `drop_film_bar`; a drop's `silence` becomes a deliberate hole in the music
edit (the segment before the drop ends `silence` early; qa.py lists it as a note, not a defect).

### Synth voices -> real recordings

| Cinetic kind | synthesized today | real-sound replacement (category / source / rule) |
|---|---|---|
| music stems: drums, bass, pad, arp, bells, reverb | generated per chord and section | ONE licensed track edited at bar lines (`arrange.py` draft, music supervisor); its own drop lands on `payoff` |
| `tick` / `tock` (signature clock) | tuned wooden mallet | a real wood block / soft mechanical switch, two takes a fourth apart, resampled within +-2 st to the key |
| `key` (typing) | dark modal tock per key, thinned to ~12/s | slices of one real keyboard recording (`slices` + `mode: cycle`); `variant: "space"/"return"` -> a deeper spacebar / enter take; per-word style -> a rule on `variant: "word"` only |
| `click` | bright 2 kHz voice + release | a real mouse / trackpad press + release (two rules, `offset` ~0.085) |
| `pop` | pitched blip | a real cork / mouth pop / plastic tick, `pitch_ramp` up across a run |
| `snap` | click + body on the chord root | a real latch / Lego / magnet clack, layered with a low body |
| `land` | wooden knock + sub | real foley thumps by weight (`weight_db`), light taps for `variant: "light"` |
| `hit` | sub boom + clap + burst | a real drum / taiko / kick hit + a short top transient (two rules) |
| `drop` | long sub boom | the music's own drop (edit), optionally a real low impact under it |
| `whoosh` | noise band sweep | a real swish (bamboo, stick, cloth), `align: "peak"` |
| `riser` / `swell` | noise + saw glide / reverse chord | a real cymbal roll or orchestral swell, `align: "end"` (or the track's own build) |
| `suck` | reverse whoosh | a real air-suck recording, or nothing (mix.py does not reverse recordings) |
| `bell` (`variant: "motif"`) | FM bell / the motif | a real celesta / bell / glass note in the key (one sample per note; the Clawd film built its motif this way) |

Cinetic's sound rules stay valid (the sound map, levels, the drop on the hero reveal, the resolved logo, restraint);
`skills/real-sound/references/sound-design.md` restates them for recordings.

## 2. The bridge, step by step (Cinetic untouched)

Inside a Cinetic film project (`films/<name>/`), with `<rs>` = `skills/real-sound` of this repo:

1. Cinetic Steps 0-4 as usual. In `BRIEF.md` write the spec line with `audio: real recordings (real-sound)`.
2. Cues: `npx tsx scripts/export-cues.ts` -> `out/cues.json`.
3. Brief: copy `<rs>/templates/SOUND_BRIEF.md` to `audio/SOUND_BRIEF.md`; take the picture map from the treatment's beat
   sheet (its sound column) and `out/cues.json`; take the personality row and the arc from `audio/score.json`
   (write it even though score.py will not run: it is the musical intent).
4. Sourcing, licence audit, music edit, design fragments, critique and fix: the real-sound process
   (`<rs>/SKILL.md`, prompts in `<rs>/templates/prompts/`), with every path relative to the film project.
5. Render the soundtrack:
   ```
   python -I <rs>/scripts/mix.py --root . --design audio/sounddesign.json --cues out/cues.json \
       --out out/audio/soundtrack.wav --stems out/stems --report out/qa/mix_report.json --trace out/qa/adds.json
   ```
   Set the design's `master.lufs` to `audio/score.json`'s `master.lufs` (-14 for launch films). Do not run
   `npm run audio` for this film (it writes the synthesized score).
6. Cinetic Step 6 with the real soundtrack: `bash scripts/render.sh Film out/preview.mp4 --preview --audio
   out/audio/soundtrack.wav` (and `--blur` for the master; never `--build-audio`, which re-runs score.py). Its `check-sync.py` gates lag and true peak on the decoded
   MP4 as before: `python3 scripts/check-sync.py out/film.mp4 out/audio/soundtrack.wav`.
7. Cinetic Step 7 (review loop) with ears (section 3).
8. Credits: `audit.py --report out/qa/mix_report.json` and `credits.py`; put `out/CREDITS_description.txt` into the
   delivery's README / video description.

`real-sound`'s own `mux.mjs` is not needed in a Cinetic film: `render.sh` muxes. Use `mux.mjs` for films rendered
outside Cinetic.

## 3. Where earshot slots into Cinetic's self-critique loop

Cinetic's loop (§8): render -> measure (probe, forensics, av-audit) -> sheets -> critic lenses -> verify -> fix.
earshot adds hearing to the measure and lens steps:

| Cinetic step | add | what it gives the critics |
|---|---|---|
| Step 5 gate (after the soundtrack is written) | `python -I <rs>/scripts/qa.py --layers --stems out/stems` | clicks / cuts with the guilty sample, masked events, harsh 2-6 kHz peaks, chopped music splices, sync per event, partial coverage, loudness and LRA |
| Step 7 measure | `python -m earshot timeline out/preview.mp4 --cues out/qa/cues_sounded.json` | a second-by-second listening transcript of the decoded film: events with tags, sections, loudness curve, beat grid, sound-to-cut and sound-to-cue offsets |
| Step 7 measure | `python -m earshot sheet out/audio/soundtrack.wav -o out/qa/sound_sheet.png --video out/preview.mp4 --cues out/qa/cues_sounded.json` | one PNG (frames, sections, tagged events, spectrogram, loudness, cue sync) the director and sound lenses look at next to the contact sheets |
| Step 7 measure | `python -m earshot critique out/audio/soundtrack.wav --video out/preview.mp4` | the scorecard against the profile of professional launch films (metric, value, reference range, what to do) |
| av-audit | `python3 scripts/av-audit.py out/preview.mp4 --cues out/qa/cues_sounded.json --stems out/stems` | Cinetic's own onset/sync audit on the real soundtrack; `cues_sounded.json` (written by mix.py) lists only events that received a sound, so deliberate silences are not counted as misses |
| sound-sync lens (`assets/critics/sound-sync.md`) | hand the lens `qa.json`, the earshot transcript, sheet and scorecard with `av.json` | the lens can name the exact rule / sample to change; with real recordings its fixes are design operations (`<rs>/templates/prompts/fixer.md`), not score.py parameters |
| verifier (`assets/critics/verifier.md`) | refute a sound issue with earshot evidence (`earshot match` on the placed sample, the transcript at that time) | fewer false P1s from reading code alone |
| a synthesized score already exists | `python -m earshot realness public/audio/soundtrack.wav` (score.py's output) | shows (per 4 s window) where the synthesized score reads as code-made: useful when the brief asks for recorded sound |

## 4. A minimal change list for a future Cinetic

Each item is small and keeps the synthesized path as the default.

1. **One switch in `audio/score.json`:** `"engine": "synth" | "real"` and `"design": "audio/sounddesign.json"`. With
   `real`, `npm run audio` runs real-sound's `mix.py` with the same `--cues/--stems`, `--out` outside `public/`
   (render.sh's `--audio` default and `--build-audio` follow it) and `--json` mapped to `--report`; `score.py` stays
   for `synth`.
2. **`export-cues.ts --thin-typing`:** apply `plan_typing()`'s thinning (<= 12 keys/s, per-word above 25 chars/s) at
   export, marking skipped keys `variant: "silent"`, so a recordings design sounds exactly the keys score.py would.
3. **`av-audit.py`:** when `engine` is `real`, read `out/qa/cues_sounded.json` and skip the score.py typing replay.
4. **`references/sound.md`:** a §1b "Real recordings" pointing at real-sound, and a "real palette" column in the
   personality table (calm: felt piano, soft wood; fast: tight drum hits, metal latches; bright: marimba, claps, cork
   pops; precise: sparse piano, glass; raw: switches, latches; warm: plucked strings, soft taps).
5. **`assets/critics/sound-sync.md`:** with `engine: real`, replace "Never propose stock or sampled sounds" by
   "propose recordings from the manifests or a new sourcing need; never synthesized sounds", and read `qa.json` and
   the earshot transcript / scorecard.
6. **`references/review-loop.md` Measure step:** add `qa.py --layers` and `earshot timeline / sheet / critique` to the
   scripts that write `out/qa/`.
7. **`scripts/new-film.sh --audio real`:** scaffold `audio/SOUND_BRIEF.md`, `audio/sounddesign.json`, `audio/design/`
   and `audio/assets/manifests/` from real-sound's templates, and write `audio: real` in the BRIEF spec line.
8. **One `master.py`:** real-sound vendors Cinetic's `master.py` byte for byte (sha1
   2af9044a923cdbe01a81bb1d7922b97d41641479 in both today); keep a single source when the two ship together.
