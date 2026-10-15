# SOUND BRIEF: {{FILM}} soundtrack from professional recordings

<!-- Copy to audio/SOUND_BRIEF.md and fill every {{FIELD}}. Every sub-agent (sourcing, music, design, critics, fixer)
receives this file verbatim: it is the contract. Delete the example rows. -->

The picture is final: `{{VIDEO}}`, {{DURATION}} s, {{FPS}} fps, cut on a {{BPM}} BPM grid (1 beat = {{BEAT}} s,
1 bar = {{BAR}} s). Cue events: `{{CUES}}` (computed from the picture code; never re-time them by hand).

The soundtrack uses ONLY real, professionally recorded or professionally produced audio from licensed libraries:
- one licensed music track, edited to the picture at bar lines;
- real foley and SFX recordings placed on the picture's cues;
- real field-recorded ambiences where the picture has a place.

Code may only select, trim, place, level, pan, EQ, crossfade, time-stretch the music (small ratios), pitch by
resampling and master these recordings. Code never generates a tone, a noise or a "sound" (no oscillators, no
synthesized noise, no synthesized reverb tails). The bar: {{BAR_REFERENCE}} (e.g. "a premium product launch film:
crisp, tactile, confident, every sound caused by something on screen"), not a game jam.

## Sound concept
<!-- 4-8 bullets: the palette (materials), the music's character and arc, the places. Example for a launch film: -->
- **One tactile UI palette.** {{PRODUCT}} is precise and calm: every interface sound comes from one family of real
  close-miked sources (a laptop trackpad click, soft felt-tip taps, one mechanical keyboard), dry and centred.
- **One music track carries the film.** {{MOOD}}: a soft intro under the cold open (no drums), a groove for the feature
  beats, a breakdown at {{BREAKDOWN_T}} s before the turn, the DROP on the hero reveal at {{DROP_T}} s, a resolved ending
  under the logo ({{LOGO_T}} s), tail to {{DURATION}} s. Instrumental only, {{TEMPO_RANGE}} BPM.
- **Restraint.** About 5 designed effects per 10 s, clustered in the cold open and the logo build; the music carries
  the cuts.
- **Places.** {{ZONES_SUMMARY}} (e.g. "a quiet office room tone under the UI part only; no ambience in the abstract
  sections").

## Picture map (film seconds; cue ids from `{{CUES}}`)
| t (s) | what happens on screen | cue (kind / id) | sound need |
|---|---|---|---|
| 0.0 | the cursor glides onto the empty search field | (none) | nothing: the music enters soft |
| 1.0 | click into the field | click / demo:cursor:click | the film's one click voice (press + release) |
| 2.0-3.5 | the query types itself (12 characters) | key / demo:type:key:n, demo:type:space | real keyboard, one key per sounded character, space deeper; music dips 6 dB |
| 4.0 | whip pan into the results | whoosh / demo:transition | one real swish, apex on the velocity peak |
| 5.5-6.0 | three result cards pop in | pop / demo:card:n | soft real pops climbing 2 st across the three |
| 7.0-8.0 | everything pulls back; the product mark flies in | riser / demo:build | the music's build (or a real cymbal roll ending 30 ms before 8.0) |
| **8.0** | **hero reveal: the whole product UI lands** | hit / demo:reveal | **the music's DROP on the frame**; a low real impact under it |
| 10.0 | the wordmark settles | bell / demo:lockup | music stops on the bar line; one soft celesta note in key; tail to the end |

Ambience zones: {{ZONES}} (e.g. "office room tone 0-8.2 s; none after the reveal").

## Licences: what is allowed
Allowed (record each in the manifest, read on the item's own page): CC0 1.0 and public domain; CC BY 3.0 / 4.0
(credits in CREDITS.md and the video description, never required on screen); Mixkit Sound Effects / Stock Music Free
License (read the current text at https://mixkit.co/license/); Pixabay Content License (manual download only).
NOT allowed: NonCommercial (BY-NC), NoDerivatives (BY-ND: syncing to picture IS a derivative), ShareAlike (BY-SA),
Sampling+, "personal use only", custom or unclear licences, disputed licences, YouTube rips, anything behind a login
or paywall you would have to bypass, AI-generated or code-synthesized audio. Never fabricate a licence.
Details: `<skill>/references/licences.md`.

## Quality bar
- SFX / foley: real recordings, close and clean. Prefer lossless WAV/FLAC (Mixkit WAV
  `assets.mixkit.co/active_storage/sfx/<id>/<id>.wav`) or Freesound's HQ preview
  (`cdn.freesound.org/previews/<id//1000>/<id>_<userid>-hq.ogg`). No clipping, floor under -55 dBFS between events,
  95 % rolloff above 12 kHz, no untreated room on foley, no voices / handling noise / background events.
- earshot: `realness` verdict "real or pro-produced" for every file used; `match` z >= 2.5 for the need it fills.
- Ambiences: clean (no people / traffic / planes / dogs unless the story wants them), no loud single events, >= 40 s
  usable, real stereo.
- Music: professionally produced and mastered, full bandwidth, instrumental, steady tempo, 4/4, {{TEMPO_RANGE}} BPM
  (stretched to {{BPM}} only if needed). Avoid: vocals, corporate ukulele + whistling, EDM brostep, lo-fi hiss,
  bombastic trailer music, moods against the brand, famous overused stock tracks.
- Mix: -14 LUFS integrated, true peak <= -1 dBTP after AAC, LRA 5-9 LU, the loudest moment on the drop, the logo
  8-20 dB under it, a tail into silence.

## Files, manifests, tools
- Downloads: `audio/assets/<category>/` (audio files only; never a zip with programs; never run anything downloaded).
  Name: `<source>-<id>_<short-slug>.<ext>`, e.g. `fs-450282_mech-typing.ogg`, `mk-1386_keyboard-typing.wav`.
- Manifests: `audio/assets/manifests/<category>.json`, a list of entries shaped like
  `<skill>/templates/manifest_entry.json` (id, category, file, title, author, source, page_url, download_url, license,
  license_url, attribution, need, rank, best_segment, notes, analysis).
- Audit: `audio/assets/manifests/audit.json` (agent fields per `<skill>/templates/audit_entry.json`) + `audit.py`.
- Design: `audio/sounddesign.json` (lead) + `audio/design/<category>.json` (one designer each).
- Review output: `out/audio-review/<category or critic>/`. QA: `out/qa/`.
- Analyse: `python -I <skill>/scripts/analyze.py <files> [--music] [--png out/audio-review/<category>] [--out ....json]`;
  listen: `python -m earshot realness|describe|match|rank|find ...`.
- Be polite to the sites: sequential downloads, a pause between requests, at most ~40 files per category.
