# The process: how a real-recordings soundtrack gets built by agents who cannot hear

This is the process that built the Clawd film's soundtrack (60 s, 83 real recordings, 264 placements, reproduced
bit-identically by `scripts/mix.py`), generalised. Agents cannot listen, so every decision is measured, every claim is
backed by a command, and the work is split so that parallel agents never edit the same file.

## Contents
1. Roles and files
2. The steps
3. What each critic measures (and the qa.py section that automates it)
4. Thresholds and where they come from
5. Variants: how critics test a fix without touching the design
6. Rounds, stopping, reporting

## 1. Roles and files

| role | prompt | owns (writes) | reads |
|---|---|---|---|
| lead | SKILL.md | `audio/SOUND_BRIEF.md`, `audio/sounddesign.json` (top level: include list, buses, master), final reports | everything |
| sourcing agent (one per category) | `templates/prompts/sourcing.md` | `audio/assets/<category>/`, `audio/assets/manifests/<category>.json`, `out/audio-review/<category>/` | brief |
| licence auditor | `templates/prompts/licence_audit.md` | agent fields of `audio/assets/manifests/audit.json` | manifests, item pages |
| music supervisor | `templates/prompts/music_supervisor.md` | `audio/assets/manifests/music_choice.json`, `audio/design/music.json` | music candidates, picture map |
| designer (one per category) | `templates/prompts/designer.md` | `audio/design/<category>.json` | its manifest, cues, solo renders |
| critic (one per lens) | `templates/prompts/critic_*.md` | only `out/audio-review/critic_<lens>/` (variants, measurements) | the whole render, design, trace |
| fixer (exactly one) | `templates/prompts/fixer.md` | every fragment + top-level buses/master during the fix round | all critic reports |

**Filling a prompt.** The lead replaces every `{{NAME}}` before handing a prompt to a sub-agent; an unfilled
placeholder must never reach one. The common ones: `{{PY}}` = the full path of the Python that has numpy, scipy,
soundfile and earshot (see SKILL.md, e.g. `<earshot repo>/.venv/Scripts/python.exe`); `{{SKILL}}` = this skill's folder;
`{{ROOT}}` = the film project root; `{{CATEGORY}}` = the sound category the agent owns; `{{FILM}}` = the film's name;
`{{ROUND}}` = the critique round number; `{{RENDER}}` = the folder of the render under review (soundtrack.wav,
stems); `{{REPORT}}` / `{{TRACE}}` / `{{QA_JSON}}` = its mix report, placement trace and qa.py report; `{{VIDEO}}` =
the rendered film; `{{CUES}}` = the cue file. The rest (`{{NEED_DESCRIPTION}}`, `{{KEY}}`, `{{BPM}}`, `{{DROP_T}}` ...)
are explained in the prompt that uses them.

## 2. The steps

1. **Cue list from the picture.** `out/cues.json` (format in `templates/cues.json`; Cinetic writes it with
   `export-cues.ts`). Sync frames are computed from the picture code (contact frames, 50 % pops, velocity peaks), never
   typed. Check the event kinds and ids are stable enough for rules to match with a regex (`act:thing:n`).
2. **Brief.** `audio/SOUND_BRIEF.md` from `templates/SOUND_BRIEF.md`: concept, picture map (every cue with what is on
   screen and the sound need), ambience zones, licences, quality bar, file layout, manifest format. The brief is the
   contract every sub-agent receives verbatim.
3. **Sourcing, per category, in parallel.** 15-40 candidates per category, licences read on each item's page,
   downloads polite (sequential, a pause between requests), each file measured (`analyze.py`, spectrogram PNG) and
   heard through earshot (`realness` gate, `describe` / `match` / `rank` against the need, `find` for the best take),
   ranked per need in the manifest with notes saying exactly what to use and what to avoid.
4. **Licence audit.** The auditor reads every candidate's page and records `licence_seen`; `audit.py` checks the rest.
   Nothing unconfirmed reaches the design.
5. **Music supervisor.** Measures every music candidate (`analyze.py --music`, `arrange.py`), verifies the beat grid
   and the downbeats against transients (the beat tracker lags attacks by 20-50 ms; arrange.py corrects the phase),
   picks the winner with reasons, and writes the edit: segments at bar lines, the drop's downbeat on the hero frame, each
   splice labelled with its bars and why it works.
6. **Design fragments, per category, in parallel.** Each designer trims samples (earshot `find` gives trims; `--dump`
   checks them), builds pools and rules on the cue events, and verifies the fragment alone: `mix.py --only <category>`
   (solo render, same variation as the full mix), `qa.py --only <category>`, `earshot timeline` of the solo with the
   cues. A designer never touches another fragment or the top level.
7. **Integrate.** The lead runs the full mix with `--trace`, then `qa.py` and earshot (`timeline`, `sheet`, `critique`
   against the launch-film profile). Fix blockers (missing files, failed gate) before critique.
8. **Critique, in parallel, read-only.** Five lenses (sync, edges, masking, tonal, director), each with its own prompt.
   Each critic measures, tests its proposed fixes as variants in its own folder (section 5) and returns ranked issues
   with exact JSON operations. Critics never edit the design.
9. **One fixer.** Merges the critics' operations, resolves conflicts (masking wants a hit louder, tonal wants its
   2-6 kHz softer: the fixer finds the change that satisfies both, e.g. louder low body + softer top), applies them,
   re-renders, re-runs qa.py and compares with the previous round (nothing that passed may now fail). One writer
   avoids lost edits and contradictory fixes.
10. **Master and gate.** `mix.py` masters (`master.py`): integrated LUFS within 0.5 of the target, true peak at most
    -1.5 dBTP on the WAV; then check again after the AAC encode (about -1 dBTP).
11. **Credits.** `audit.py --report out/qa/mix_report.json` (exit 0) then `credits.py --audit ...` (exit 0): CREDITS.md
    and the video-description text.
12. **Mux.** `mux.mjs` (stream copy of the picture, AAC 320k, cut to the picture's duration); verify the audio track of
    the MP4 against the WAV (lag 0 samples) and run `earshot timeline film.mp4 --cues out/qa/cues_sounded.json` on the
    finished file.

## 3. What each critic measures

| lens | question | measurements (Clawd critic scripts, now qa.py) | typical fixes |
|---|---|---|---|
| **sync** | does every sound land on its cause? | per event: attack, 5 ms peak and 50 ms apex of the rebuilt event vs its cue (qa.py sync); risers by their end; per-frame picture motion around the event (frame differences of the film, earshot timeline cut/cue sync); envelope shape of each dumped sample (attack, -3/-6 dB span); solo renders of one rule | `align` (attack/peak/end), `offset`, a trim that starts closer to the attack, a different take whose peak is earlier |
| **edges** (clicks) | is anything cut, clicked or truncated? | > 14 kHz bursts concentrated in 1 ms in the master and every stem, attributed to the layer carrying the burst (qa.py clicks); unfaded head / tail of each used sample vs its peak and the source just outside the trim; second hits inside a sample; slices; placements past the film ends; music splices (the source level just after a segment end vs what enters) (qa.py edges) | longer `fade_in`/`fade_out`, move `start` 3-5 ms before the attack, end after the decay, `stop_before_next`, a different splice bar, `tail_fade` |
| **masking** (audibility, density) | can every effect be heard, and is the effect layer dense or repetitive? | per event: 1/3-octave margin vs the bed and vs everything else over its loudest 60 ms and its first 20 ms (`read_db`), per layer (dead layers); onsets per second; sample and take reuse; per-rule regularity (inter-onset CV, gain and pitch spread, distinct samples) (qa.py audibility, density); A/B variants of crowd beds | gain, EQ that carves the effect's band (or the bed's), a brighter take, fewer layers, a bus ride under the moment, more takes in a pool, `jitter_db` |
| **tonal** | is it pleasant on phones and laptops? | per event 2-6 kHz 5 ms peak vs the music's loudest (qa.py harshness) and its centroid; band levels over time of master vs stems (2-6k, 6-10k, 10-16k, >16k); low end under 120 Hz (mono, the impact's low partials vs the music's at the drop: beating); stereo width per section; tuned samples vs the key; limiter gain reduction at candidate (lufs, ceiling) settings | `highshelf` / `peq` cuts, `norm_peak_db` down, `mono`, `lp`, a pitch to the key, `master.lufs` / `ceiling` |
| **director** | does the soundtrack tell the film's story? | momentary / short-term loudness of the master and each stem across the picture map's beats, LRA, section medians, the loudest moment's position, the drop vs the logo, every beat's sfx and music peaks, ambience sources per zone (qa.py loudness + earshot timeline + sheet + critique) | bus rides, the music edit, which events are sounded, the ending |

## 4. Thresholds and where they come from

| check | threshold | source |
|---|---|---|
| click | > 14 kHz sample > 9x its 20 ms RMS and > 6x its 4 ms RMS, >= 80 % of the burst's energy in 1 ms | Cinetic av-audit's detector, ported (the same 7 click times as av-audit on the Clawd master: a check of the port, not independent validation) |
| click at an edge = a cut | the carrying layer's start or end within 2 ms | qa.py (fail); inside a recording = warn (the recording's own crisp transient, or a defect in the source) |
| abrupt entry | source 20 ms before the trim within 20 dB of the trim's peak, fade-in < 5 ms, and no attack within 10 ms to cover it (or the source within 10 dB) | Clawd critic_clicks edges.py (-30 dB listing), tightened so attacks that follow at once are not flagged |
| chopped decay | unfaded tail > -35 dB re peak, the source after the trim > -35 dB, fade-out < 30 ms | Clawd critic_clicks edges.py |
| chopped music | the source keeps sounding after `src_to` (within 6 dB) and is 6 dB above what enters, fade-out < 100 ms, no `tail_fade` | qa.py (generalises the supervisor's splice listings) |
| audibility | read = max(60 ms margin vs everything, 20 ms margin vs everything) over bands within 30 dB of the effect's strongest: >= 6 clear, 3-6 ok, 0-3 barely, < 0 masked | Clawd critic_masking measure.py |
| harshness | an event's 2-6 kHz 5 ms RMS more than 6 dB over the music's loudest 2-6 kHz 5 ms RMS | qa.py (the critic listed both levels; the 6 dB margin is a judgement); centroid > 6.5 kHz noted (norm) |
| sync | impacts: attack >= -45 ms and peak <= +75 ms from the cue; whooshes and peak-aligned swells: apex within +-45 ms; risers and end- / attack-aligned swells: audible end (last sample above -20 dB re its peak) -250..+30 ms (checked at -300..+60); a deviation that one layer's rule offset removes is listed as deliberate, naming that rule | measured norms of pro launch films (Cinetic sound.md §13) |
| machine gun | a rule with >= 4 events, one sample, pitch sd < 0.1 st, gain sd < 0.5 dB | Clawd critic_density metrics.py (regularity table) |
| loudness | integrated within 0.5 LU of the target (fail), true peak <= -1.0 dBTP (fail), last 50 ms <= -45 dBFS RMS and last 10 ms <= -60 dBFS peak (fail), first 5 ms <= -40 dBFS (warn), LRA 4-10 LU (warn), loudest moment before 85 % of the runtime (warn) | master.py gate, Cinetic av-audit head/tail, norms |

What qa.py flagged on the finished Clawd mix (which had been through five critics and a supervisor): 7 clicks in the
master, all inside recordings (the detector is av-audit's, so these are the transients the critics had already heard
about and accepted as crisp), 5 of 153 events masked (busy stacking moments), 3 events hot at 2-6 kHz, 2 edge defects
(one a deliberate 25 ms choke, one a deliberate music cut into the meadow), no sync defect (4 events sit off their cue
by exactly a rule offset, listed as deliberate with the rule that explains each), 2 cue events unsounded. Treat findings as questions
for a critic, not as orders; acknowledge confirmed exceptions with `qa_ok` (design-format.md §12).

## 5. Variants: how critics test a fix without touching the design

A critic writes its proposed operations to a JSON list and lets `variant.py` copy the design (top level + fragments)
into its own folder, apply them, render with every output inside that folder and run qa.py on the result:

```
python -I <skill>/scripts/variant.py --ops out/audio-review/critic_tonal/ops_v1.json \
   --out out/audio-review/critic_tonal/v1 --render --qa [--cues out/cues.json]
```

It prints every change as old -> new (an operation that matches nothing writes nothing), then the mix's gate line and
the qa digest; `v1/qa.json` sits next to the base render's `out/qa/qa.json` for a before / after comparison. The
critic reports those numbers next to each proposed operation. Operations are exact and small:
`{"f": "keyboard", "sample": "kb.thock.1", "set": {"norm_peak_db": -10}}`,
`{"f": "motion", "rule": "mo.leap.air", "set": {"offset": 0.30}}`,
`{"sd": "buses.music.auto", "value": [[0, 0], ...]}` (all forms in `templates/prompts/fixer.md`). The fixer applies
the merged list to the real files with `variant.py --apply` (originals copied to `--backup` first).

Variants catch interactions nobody predicts by reading JSON. A real case on the Clawd design: a 4 dB high-shelf cut on
a plastic "toy open" sample made its event LOUDER at 2-6 kHz (-15.8 -> -15.2 dBFS), because `norm_peak_db`
renormalises the sample after EQ (design-format.md §3, processing order); the fix is a lower `norm_peak_db` with the cut.

## 6. Rounds, stopping, reporting

- One critique round + one fix round is the minimum; a second round when the first fix round changed levels or the
  music edit. Stop when qa.py shows no fail, every remaining warn is acknowledged with a reason, and the director lens
  has no P0.
- Every number in a report comes from a command that was run; the report lists the commands.
- Keep `out/audio-review/` (critic folders, variants, dumps): it is the audit trail of why each number is what it is.
