# Troubleshooting

## mix.py

| symptom | cause | fix |
|---|---|---|
| `cue file out/cues.json not found (the design has cue rules)` | no cue file at the default path | pass `--cues path` or set `"cues"` in the design |
| `set "duration" (seconds) in the design or "total" (frames) in the cue file` | neither is known | add `"duration"` to the top-level design |
| `KeyError: 'pool X: unknown sample Y'` | a typo, or a slice group without `#*` | fix the name; `"kb.type#*"` expands every slice |
| `ValueError: pool X is empty` / `slices F: no onsets found` | the slice window has no onsets above the gate | widen `start`/`end`, lower `delta` (0.05), lower `gate_db` (-24), check the file with `analyze.py` |
| `KeyError: 'fragment.json: samples "x" is defined twice'` | two fragments use one name | prefix every name with its category (`kb.`, `mo.`) |
| `KeyError: '_doc'` in the bus loop | a comment key inside `buses` | move comments to `_buses_doc` at the top level |
| `ValueError: reverse is not used` | a sample asks for `reverse` | find a recording that already has the shape (a real swell, a reverse cymbal recorded as such) |
| `rubberband failed` | the ffmpeg build has no librubberband | use `"stretch": "none"` when the track is within ~0.2 % of the film tempo, else ask the user for an ffmpeg with rubberband (`ffmpeg -filters | grep rubberband`) |
| `master gate FAILED` | integrated loudness or true peak missed | read `master.json` next to the report: lower `master.lufs` by 1 for a dynamic track, or trim the loudest peaks (often one impact or the drop's first beat) with a bus ride; `ceiling` 0.77 leaves AAC headroom |
| the first run is slow | every recording is decoded once (and the music stretched once) | later runs read `out/audio/cache` (keyed on path, size, mtime); `--cache DIR` moves it |
| a solo render "has no file" | older habit: looking for premaster.wav | `--only NAME` writes the unmastered mix to `--out` (default `out/audio-review/solo/<NAME>.wav`) |
| the solo sounds different from the full mix | a rule without `name` (variation keyed on its index) | name every rule; then solo and full mix play identical placements |
| an event is in `unsounded_events` | no rule matches its kind / id / time | add a rule, or accept it on purpose (qa.py coverage lists partial classes, which read as mistakes) |

## qa.py

| symptom | cause | fix |
|---|---|---|
| sections "skipped: needs the trace" | the mix ran without `--trace` | re-run `mix.py ... --trace out/qa/adds.json` |
| `trace (N placements) and report (M) disagree` | the trace and the report come from different runs or a different `--only` | re-render with both from one run; pass the same `--only` to qa.py |
| `placement(s) could not be rebuilt` | the design changed after the mix | re-render before qa |
| a click "inside a recording" | the recording itself has a crisp transient (a tick, a pop) or a defect (a digital click, a bump) | `--dump` the sample and look at its spectrogram (`analyze.py --png`); if it is a defect, trim around it or use another take; if it is the sound itself, acknowledge with `qa_ok` |
| a music splice "chops the music" | the outgoing segment ends while its note still sounds and the next one is quieter | end the segment where the chord has decayed, splice into a louder bar, add `tail_fade`, or acknowledge a deliberate hard cut with `qa_ok` |
| many "bright" notes | a tiny-object palette (plastic, glass, bells) is bright by nature | fine when it is the film's palette; otherwise a `highshelf` cut or a darker take |
| `--listen` says earshot is not importable | earshot is not installed in this Python | `pip install -e <earshot repo>`, or run qa.py without `--listen` and call `python -m earshot ...` separately |

## audit.py / credits.py

| symptom | fix |
|---|---|
| every recording "licence not verified" | the auditor has not recorded `licence_seen` / `licence_confirmed` in `audio/assets/manifests/audit.json` yet: run `templates/prompts/licence_audit.md` |
| `licence_seen reads as X but the manifest says Y` | the page and the manifest disagree: re-read the page, fix the manifest's `license`, or reject the item |
| `used in the film but in no manifest` | a file was added to a fragment without a manifest entry: add the entry (and audit it) or remove the sound |
| `entry id ... but the Freesound page is sound ...` | the entry, the page and the download URL describe different sounds: re-download from the right page |
| credits list "PROBLEMS" | a used recording is forbidden, unknown, unconfirmed or lacks author / title / page_url: fix before publishing |

## Sources and downloads

| symptom | fix |
|---|---|
| Freesound preview 404 | the preview folder is `id // 1000` and the number after `_` is the uploader's user id (take it from the page's player) |
| Pixabay returns 403 / a challenge page | Pixabay blocks scripted downloads: give the user the link and ask them to download the file |
| a Mixkit item has no page | Mixkit SFX have none: record the category page and the listed title |
| a file decodes with errors | re-download; prefer the lossless (Mixkit WAV) or HQ (Freesound -hq.ogg) version |

## Mux and delivery

| symptom | fix |
|---|---|
| true peak over -1 dBTP after AAC | lower `master.ceiling` (0.77 -> 0.72) and re-render; measure the MP4's decoded audio with `python -m earshot measure film.mp4` (`master.py --measure` reads WAV / FLAC only) |
| the audio ends early | the soundtrack is shorter than the picture: set `duration` to the picture's length (mux.mjs reports the mismatch) |
| sync looks off | `mux.mjs` stream-copies the picture and cuts the audio to the picture's duration; cross-correlate the MP4's decoded audio with the WAV (lag must be 0) and check the cue frames; never re-time the audio to fix a picture-side offset |

## Platform

- Run the scripts with `python -I` (isolated: downloaded files can never shadow a module); the scripts add their own
  folder to the path themselves.
- Console output is ASCII (Windows cp1252 consoles); quote paths with spaces (`--root "/path/to/my film"`).
- `mix.py`'s audio maths are frozen: the Clawd soundtrack reproduces bit-identically
  (`sha1 b48c5e8872fe96765624f88db18a0b49a0a05d68`). Change the DSP only with a new version and a new reference.
