# Licences: what may go into a film, how to verify it, how to credit it

A launch video is commercial use, it is edited (trimmed, layered, synced to picture: legally a derivative), and it is
published on platforms that scan audio. Every recording in the film needs a licence that allows all three, read on the
item's own page, recorded in its manifest, confirmed by the licence audit and credited. When in doubt, leave it out:
there is always another recording.

## Allowed / forbidden

| licence | status | why | credit |
|---|---|---|---|
| CC0 1.0, Public Domain Mark | allowed | no conditions | courtesy only |
| CC BY 4.0 / 3.0 / 2.5 / 2.0 | allowed | commercial use and edits allowed with attribution | REQUIRED (title, author, source link, licence link, "edited") |
| Mixkit Sound Effects Free License / Stock Music Free License | allowed | free for commercial video, no attribution; read the current text at https://mixkit.co/license/ | courtesy only; never redistribute the raw files, never register the film with Content ID |
| Pixabay Content License | allowed | free for commercial use, no attribution (Pixabay blocks scripted access: the user downloads by hand) | courtesy only |
| CC BY-NC (any version) | forbidden | NonCommercial: a launch video is commercial | |
| CC BY-ND | forbidden | NoDerivatives: trimming and syncing to picture are derivatives | |
| CC BY-SA | forbidden | ShareAlike: the whole film would have to be released under CC BY-SA | |
| Sampling+, Sampling | forbidden | retired, restrictive | |
| "personal use", "editorial", "custom", "free with credit on social media", all rights reserved | forbidden | not a standard licence the audit can trust | |
| two sources disagree ("DISPUTED") | forbidden | the strictest reading wins: reject it | |
| "royalty free" with no licence text, YouTube rips, anything behind a login or paywall you would bypass, AI-generated or code-synthesized audio | not used | unknown terms, or not a real recording | |

`scripts/licences.py` encodes this table (`classify()`); audit.py and credits.py use it.

## Verify on the item's own page (never from a search result, an aggregator or memory)

**Freesound** (most foley, ambiences, whooshes):
- The page is `https://freesound.org/people/<user>/sounds/<id>/`; the licence is in the sidebar of THAT page ("Creative
  Commons 0", "Attribution 4.0", "Attribution NonCommercial 4.0" ...). Licences are per sound, not per user.
- Record: licence text as shown, page author, title, the date you read it (`licence_seen` in audit.json).
- Originals need a login; the public high-quality preview is `https://cdn.freesound.org/previews/<id // 1000>/<id>_<user_id>-hq.ogg`
  (Ogg Vorbis, about 192 kbps; `-hq.mp3` also exists). `<user_id>` is the uploader's number: it must belong to the same
  page. audit.py checks that the id in the entry, the page and the preview URL agree.
- Read the description: some authors ask for a specific credit form (e.g. "<title> by <name> (www.site) of
  Freesound.org"), or forbid putting raw files online. Copy the requested form into the manifest's `attribution` and
  any restriction into the audit's `agent_notes`.

**Mixkit** (lossless WAV effects, some music):
- No per-item page exists for sound effects: record the category page that lists the item and its title there.
- WAV download: `https://assets.mixkit.co/active_storage/sfx/<id>/<id>.wav`. Read the licence text itself
  (https://mixkit.co/license/) once per session and quote what it allows in `licence_seen`.
- Do not publish the raw assets folder; do not register the film's audio with Content ID or a rights-management service.

**Music sites** (one licensed track carries the film):
- Composer libraries that release CC BY music (e.g. Scott Buckley, scottbuckley.com.au): the licence line is on the
  track's page; many ask for a specific credit line: use it verbatim.
- Aggregators (chosic.com and similar) are not authoritative: confirm on the artist's own page. A real case: an
  aggregator listed a track as CC BY 3.0 while the artist's page said "free with credit on social media" (copyrighted):
  rejected as disputed.
- Free Music Archive, Internet Archive, SoundCloud: the licence is per track and often NC or ND: read it every time.
- YouTube Audio Library, platform libraries: terms tied to one platform: ask the user before using.

**Pixabay**: scripted downloads are blocked (bot protection). If a Pixabay item is the best choice, give the user the
page link and ask them to download it into the assets folder; then audit it like any other file.

## Attribution (CC BY)

A complete credit names the Title, the Author, the Source (link to the item page), the Licence (name + link) and says
it was modified. The manifest's `attribution` holds the line the author asked for (or this default):

```
"Rain on Windows, Interior, A.wav" by InspectorJ (www.jshaw.co.uk) of Freesound.org
(https://freesound.org/people/InspectorJ/sounds/346642/), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
```

`credits.py` writes `CREDITS.md` (music + every sound, grouped by licence, with the sentence saying how the recordings
were edited) and `out/CREDITS_description.txt` (paste it into the video description). It completes a requested line
with the source link and the licence link when the author's form leaves them out. Credits never need to be on screen.

## Content ID and CC BY music

Some composers register their CC BY tracks with YouTube Content ID through a distributor. A video using the track
correctly can still get an automatic claim. The licence still allows the use: put the credit in the description, and
if a claim arrives, dispute it citing the licence and the credit (many composers also clear claims on request). Choose
a track whose composer explains this on their site, and tell the user before publishing. Mixkit and Pixabay forbid the
reverse: never register a film that contains their items with Content ID.

## The audit (scripts/audit.py, prompts/licence_audit.md)

1. The licence auditor opens every in-scope item's page, reads the licence and writes `licence_seen` (quote + URL +
   date + page author/title) and `licence_confirmed: true` into `audio/assets/manifests/audit.json` (format:
   `templates/audit_entry.json`). Never confirm a page you did not read in this session.
2. `audit.py --report out/qa/mix_report.json` classifies every manifest entry, checks ids and URLs, checks every file
   (exists, codec, rate, channels, duration vs the manifest, decodes without errors), merges the auditor's fields and
   writes audit.json + a Markdown table. In-scope = audible in the film. Exit 1 while any in-scope recording is blocked.
3. `--fetch` compares the local bytes with `download_url` (sequential, 1.5 s apart, a clear User-Agent, in memory,
   nothing executed; never Pixabay). Re-encoded or edited files show up here.
4. `credits.py --audit audio/assets/manifests/audit.json` refuses to call the credits clean while a used recording is
   unconfirmed.

On the Clawd film this caught four used recordings (celesta notes added after the hand audit) with no audit record.
Their manifest notes do say the licence was read on the Freesound page, so it was a bookkeeping gap, not an unverified
licence: audit.py quotes such a note and asks the auditor to copy it into `licence_seen`.
