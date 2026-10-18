---
name: real-sound
description: Build premium, realistic soundtracks for code-made videos and animations from real licensed recordings - launch films, product and feature videos, UI walkthroughs, logo stings, explainers, trailers and animated shorts made with Remotion, HyperFrames, three.js, GSAP or any renderer. Use this whenever a video or animation needs sound design, SFX, foley, ambience, music, a soundtrack, a score, UI click / whoosh / riser / impact / logo-sting sounds, or a music edit to picture; whenever a synthesized or "random" sounding soundtrack must be replaced with real recordings; or when the user asks to source, license, credit, mix, master, critique or mux audio for a film. It covers cue lists from the picture, sourcing from Freesound / Mixkit / composer libraries with licence checks, a music edit at bar lines, a JSON sound design rendered by a deterministic mixer, measured QA (clicks, masking, harshness, sync, loudness) plus earshot listening, a multi-agent critique, mastering, credits and the mux.
---

# real-sound

You are the sound supervisor of a film you cannot hear. The soundtrack is built ONLY from real, professionally
recorded or produced audio (one licensed music track, real foley and SFX, real ambiences), placed on the picture's
cue frames by a deterministic mixer, measured instead of listened to, critiqued through five lenses, fixed by one
agent, mastered, credited and muxed. Code selects, trims, places, levels, pans, EQs, pitches by resampling,
time-stretches the music and masters; code never generates a sound.

The bar is a professional launch film: every sound caused by something on screen, one material palette, the music's
drop on the hero reveal, restraint everywhere else, a resolved logo, a tail into silence. Rules:
`references/sound-design.md` (and the earshot repo's `docs/launch-video-sound.md` for the bundled reference profile:
7 professional launch films).

`<skill>` below is this folder. **Python:** `{{PY}}` below (and in every prompt template) is ONE interpreter that has
numpy, scipy, soundfile and the earshot package: the earshot repo's venv
(`<earshot repo>/.venv/Scripts/python.exe` on Windows, `<earshot repo>/.venv/bin/python` elsewhere), or
any Python where `<python> -c "import earshot"` works. Bare `python` on Windows is often the Microsoft Store stub and
fails. Substitute the full path for `{{PY}}` when you run a command or hand a prompt to a sub-agent (the other
placeholders are listed in `references/process.md`). Run the scripts with `-I` (isolated: downloaded files can never
shadow a module); every script except `scripts/licences.py` (the licence policy module that audit.py and credits.py
import) prints `--help`.

## The process

| step | who | output | gate |
|---|---|---|---|
| 0. cue list | lead | `out/cues.json` | sync frames computed from the picture code |
| 1. brief | lead | `audio/SOUND_BRIEF.md` | every cue has a sound need (or "nothing") |
| 2. sourcing (per category, parallel) | sourcing agents | `audio/assets/<cat>/`, manifests | each file measured + earshot `realness` "real or pro-produced" on the trim used (an "uncertain" only with its evidence and a trim-level re-check written in the manifest) |
| 3. licence audit | auditor | `audit.json` | `audit.py` exit 0 for what the film uses |
| 4. music edit | music supervisor | `audio/design/music.json` | grid verified, drop on the reveal, splices clean |
| 5. design fragments (per category, parallel) | designers | `audio/design/<cat>.json` | solo render + qa clean |
| 6. integrate | lead | soundtrack, stems, report, trace | master gate (+-0.5 LU, <= -1.5 dBTP) |
| 7. QA + listening | lead | `out/qa/qa.json`, earshot timeline / sheet / critique | no qa fail |
| 8. critique (5 lenses, parallel, read-only) | critics | ranked issues + tested operations | |
| 9. fix | ONE fixer | the design | qa: no regression |
| 10. master, credits, mux | lead | `CREDITS.md`, description text, the film | credits exit 0, lag 0 |

The detail of every step, the roles and what each critic measures: `references/process.md`. Every sub-agent prompt
is a ready-to-fill brief in `templates/prompts/`.

### 0. Cue list from the picture
Every sound sits on a frame the picture code computes: a contact frame, a pop-in's 50 % frame, a move's velocity
peak, a riser's hit, the frame the wordmark becomes readable. Cinetic films already have `out/cues.json`
(`export-cues.ts`); for anything else write the same format (`templates/cues.json`) from the timeline constants.
Stable ids (`act:thing:n`) let rules match classes of action with a regex.

### 1. Brief
Copy `templates/SOUND_BRIEF.md` to `audio/SOUND_BRIEF.md`: concept (palette, the music's arc, places), the picture map
(every cue: what is on screen, the sound need), ambience zones, allowed licences, quality bar, file layout. It is the
contract every sub-agent gets verbatim. Pick the brand's personality row (`references/sound-design.md` §3).

### 2. Sourcing, per category (`templates/prompts/sourcing.md`)
Typical launch-film categories: `ui` (clicks, taps, toggles, pops), `keyboard`, `motion` (whooshes, risers),
`impacts` (reveal hits, the logo), `ambience`, `music`; a character film adds `foley`, its worlds, its objects.
Freesound (CC0 / CC BY; HQ preview `cdn.freesound.org/previews/<id//1000>/<id>_<user>-hq.ogg`), Mixkit (WAV
`assets.mixkit.co/active_storage/sfx/<id>/<id>.wav`), composer libraries for music; licences read on each item's own
page (`references/licences.md`); polite sequential downloads; every file measured:
```
{{PY}} -I <skill>/scripts/analyze.py audio/assets/ui/* --png out/audio-review/ui --out out/audio-review/ui.json
{{PY}} -m earshot realness audio/assets/ui/                  # gate: "real or pro-produced" (re-check trims: --start --end)
{{PY}} -m earshot rank "a single crisp trackpad click" audio/assets/ui/ --top 5
{{PY}} -m earshot match <file> "a soft cork pop" --alts "a gunshot" "a balloon burst"
{{PY}} -m earshot find <long take> "a single mechanical key press" --top 8   # trims; skip any marked head cut / tail cut
```

### 3. Licence audit (`templates/prompts/licence_audit.md`)
The auditor records what each page says (`licence_seen`, `licence_confirmed`) in
`audio/assets/manifests/audit.json`; the script checks the rest (classification, ids vs URLs, files decode, durations,
credit lines):
```
{{PY}} -I <skill>/scripts/audit.py --root . --report out/qa/mix_report.json [--fetch]
```
Allowed: CC0, public domain, CC BY 2.0-4.0, Mixkit free licences, Pixabay. Forbidden: NC, ND, SA, Sampling+, personal,
custom, disputed. Unverified = blocked.

### 4. Music (`templates/prompts/music_supervisor.md`)
One licensed track whose structure fits the film. Write the arc per film bar (`templates/arc.json`), then:
```
{{PY}} -I <skill>/scripts/arrange.py audio/assets/music/<track> --film-bpm 120 --arc audio/arc.json --out out/audio-review/music/arrange.json
```
It fits the tempo, moves the grid onto the measured attacks (beat trackers lag 20-50 ms), proposes the downbeat
phase, anchors the drop film bar on the strongest drop candidate and drafts segments at bar lines (a ready-to-paste
`music` block). The downbeat is the weak spot: when it prints LOW confidence (on one of the two tracks it was tested on
it was half a bar off), decide between the two printed grids from the first note, the section entrances and the chord
changes in the bar table, and re-run with `--phase N`. Then edit by judgement: harmony continues across each splice,
the drop's downbeat on the hero frame, the final chord on the logo, `tail_fade` into the end.

### 5. Design fragments, per category (`templates/prompts/designer.md`)
`audio/design/<category>.json` with samples (trims), slices (cut a take of many hits into single hits), pools, rules
on cue events, free placements, ambience beds, the music block. Reference with every key and real examples:
`references/design-format.md`; a complete launch-video example: `templates/design_fragment.json` (its file names are
placeholders such as `mk-0000_mouse-click.wav`: it renders once each one is replaced by a real recording from your
manifests; mix.py stops and lists the missing files until then).
Verify alone:
```
{{PY}} -I <skill>/scripts/mix.py --only ui --dump out/audio-review/ui/dump        # every trim as WAV + numbers
{{PY}} -I <skill>/scripts/mix.py --only ui --trace out/audio-review/ui/adds.json --out out/audio-review/ui/solo.wav
{{PY}} -I <skill>/scripts/qa.py --only ui --report out/audio-review/ui/solo_report.json --stems out/audio-review/ui/solo_stems --trace out/audio-review/ui/adds.json --out out/audio-review/ui/qa.json
{{PY}} -m earshot timeline out/audio-review/ui/solo.wav --cues out/cues.json
```

### 6-7. Integrate, QA, listen
```
{{PY}} -I <skill>/scripts/mix.py --root . --design audio/sounddesign.json --cues out/cues.json --trace out/qa/adds.json
{{PY}} -I <skill>/scripts/qa.py --layers                      # clicks, edges, audibility, harshness, loudness, coverage, sync, density
{{PY}} -I <skill>/scripts/qa.py --listen --realness --video out/film.mp4   # + earshot timeline, sheet, critique, realness
{{PY}} -m earshot critique out/audio/soundtrack.wav --video out/film.mp4    # scorecard vs professional launch films
{{PY}} -m earshot sheet out/audio/soundtrack.wav -o out/qa/sheet.png --video out/film.mp4 --cues out/qa/cues_sounded.json
```
`qa.py` ties every finding to the rule and sample that caused it (the trace). It is usable without the models;
`--listen` adds earshot's transcript (`out/qa/qa_timeline.txt`), annotated sheet and scorecard.

### 8-9. Critique and fix
Five read-only critics in parallel, each with its prompt: `critic_sync.md`, `critic_edges.md`, `critic_masking.md`,
`critic_tonal.md`, `critic_director.md`. Each tests its fixes as a variant in its own folder:
```
{{PY}} -I <skill>/scripts/variant.py --ops out/audio-review/critic_tonal/ops_v1.json --out out/audio-review/critic_tonal/v1 --render --qa
```
and returns ranked issues with exact operations and before -> after numbers. Then ONE fixer (`fixer.md`) merges the
operations, resolves conflicts, checks the merged variant against the previous qa.json, and applies it
(`variant.py --apply`, originals backed up). Acknowledge confirmed deliberate exceptions with `"qa_ok": "<reason>"`.
At least one round; a second when levels or the music edit changed.

### 10. Master, credits, mux
`mix.py` masters with `master.py` (-14 LUFS, true-peak limiter, ceiling 0.77 linear; gate +-0.5 LU and <= -1.5 dBTP).
```
{{PY}} -I <skill>/scripts/audit.py --root . --report out/qa/mix_report.json            # exit 0
{{PY}} -I <skill>/scripts/credits.py --root . --design audio/sounddesign.json --audit audio/assets/manifests/audit.json
node <skill>/scripts/mux.mjs --video out/film.mp4 --audio out/audio/soundtrack.wav --out out/film_sound.mp4
{{PY}} -m earshot timeline out/film_sound.mp4 --cues out/qa/cues_sounded.json          # the finished file, heard
```
`CREDITS.md` + `out/CREDITS_description.txt` (paste into the video description: CC BY credits are required there).
`mux.mjs` stream-copies the picture and cuts the AAC audio to the picture's duration.

## Launch videos: the sound map

| picture | sound | sits on |
|---|---|---|
| cursor moves, hovers, scrolls, progress, streaming text | nothing (the music carries it) | |
| click / tap | ONE real click voice for the film, press + release ~85 ms later, same level every time | the press frame |
| typing | one real keyboard, sliced; <= ~12 keys/s, one per word above ~25 chars/s; music dips 6-10 dB | each sounded character |
| a card / chip / badge appears | a soft real pop or tick, climbing across a run | 50 % of its travel |
| toggle, lock-in, a counter's final value | a real latch / click / crisp tick | the contact / lock frame |
| camera-scale move, whip, zoom (~2 per film) | a real swish, apex on the velocity peak | velocity peak |
| hard cuts | nothing | |
| tension into the reveal | the music's build, or a real riser ending 30-250 ms before the hit | the hit |
| hero reveal | the music's DROP (edit the track so its downbeat lands here) + optionally one real impact | the frame |
| logo / lockup | music stops on a bar line or filters down, 0.2-0.7 s near-silence, ONE soft note in key, tail to silence | the frame the wordmark reads |

Levels: -14 LUFS, true peak <= -1 dBTP after AAC, LRA 5-9 LU, the loudest moment on the drop (30-60 % in), the logo
8-20 dB under it, about 5 designed effects per 10 s, effects dry and centred. Measured norms and the cheap tells:
`references/sound-design.md`.

## Tools

| need | tool |
|---|---|
| measure recordings (format, floor, bandwidth, onsets, spikes, tempo / key) + spectrogram PNGs | `scripts/analyze.py` |
| hear recordings: real vs synthesized, what it sounds like, best match, best take inside a long file | `{{PY}} -m earshot realness / describe / match / rank / find / compare / ab` |
| levels / brightness / onsets of one file or a stretch of it (`--start --end`) | `{{PY}} -m earshot measure` (Python: `from earshot import api; api.measure(...)`, likewise `api.find`, `api.realness` ...) |
| licence + file audit | `scripts/audit.py` (policy in `scripts/licences.py`) |
| music grid, downbeats, bar-line edit draft | `scripts/arrange.py` |
| render the design (+ solo `--only`, `--dump` trims, `--trace` for QA) | `scripts/mix.py` |
| master and loudness measurement | `scripts/master.py` (`--measure`) |
| objective QA tied to placements | `scripts/qa.py` (`--layers`, `--listen`, `--realness`) |
| critic variants, the fixer's apply | `scripts/variant.py` |
| credits for CREDITS.md and the video description | `scripts/credits.py` |
| put the soundtrack on the picture | `scripts/mux.mjs` |
| listen to a whole mix / film; annotated picture; launch-film scorecard | `{{PY}} -m earshot timeline / sheet / critique` |

## Files

```
audio/SOUND_BRIEF.md                 the contract              (templates/SOUND_BRIEF.md)
audio/assets/<category>/             downloaded recordings     (<source>-<id>_<slug>.<ext>; never commit or publish them)
audio/assets/manifests/<cat>.json    one entry per recording   (templates/manifest_entry.json)
audio/assets/manifests/audit.json    licence audit             (templates/audit_entry.json)
audio/sounddesign.json               top level: includes, buses, master   (templates/sounddesign.json)
audio/design/<category>.json         one fragment per designer             (templates/design_fragment.json)
out/cues.json                        the picture's cue events              (templates/cues.json)
out/audio/soundtrack.wav, stems/     the master and its stems
out/qa/                              mix_report.json, adds.json, cues_sounded.json, master.json, qa.json
out/audio-review/                    per-category and per-critic work, variants, dumps (the audit trail)
CREDITS.md, out/CREDITS_description.txt
```

## References
- `references/process.md`: roles, steps, what each critic measures, thresholds and their sources, variants, rounds.
- `references/design-format.md`: every design key, defaults, processing order, outputs, real examples.
- `references/sound-design.md`: launch-video and film sound rules, levels, sync, variation, tuning, selection, cheap tells.
- `references/licences.md`: allowed / forbidden, verification per site, attribution, Content ID.
- `references/troubleshooting.md`: errors and fixes.
- Cinetic films: `docs/INTEGRATION.md` in the earshot repo (how the cue file, timeline and review loop connect).
