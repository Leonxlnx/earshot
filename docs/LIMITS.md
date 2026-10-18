# What earshot can and cannot judge

earshot gives an agent measured evidence about sound, not taste. Read this before trusting a verdict.

## Trustworthy
- **Measurements**: loudness (LUFS, true peak, LRA), timing (onsets, offsets to cuts and cue frames, to the
  millisecond), levels, spectrum, envelope, clipping, silence, tempo of steady music. These are meters, not opinions.
- **"Is this the kind of sound I asked for?"** on single, exposed sounds (a key press, a whoosh, rain, a glass ting):
  a "strong" match is very reliable on held-out recordings; "plausible" and "weak" are hints worth a second check.
- **Picking among candidates** (`rank`) and **finding the clean moment** (`find`) in real recordings.
- **Obvious code-synthesized sources** (noise risers, simple FM bells, steady test tones): flagged with ~92% precision
  when the verdict is firm (6.7% of real clips get called synthesized, 5.5% of synthesized clips real). Bare beeps
  are less reliable than that list suggests: with the beep family held out, beeps were among the most-missed types
  (accuracy 0.71).
- **Whether sounds land on the picture**: offsets of hard hits to cuts / cue frames (median 0 ms on Clawd, 96% within
  a frame).

## Hints only
- **Tags of events inside a busy mix** (what each hit "is"): right about half the time for exposed events, a quarter for
  events under the music. Times and levels are right; the label is a guess.
- **Music genre and mood**: often wrong (an upbeat track's top mood was a dark one 18% of the time); describe keeps
  them apart as unvalidated hints. Instruments, rain, birds and voice are usually right.
- **"Uncertain" from realness** on designed library sounds ("magic" chimes/celesta, "motion" whooshes) and on real
  electronic music: they sit between real and synthesized. Read it as "sounds designed", not "fake".
- **The launch scorecard ranges** come from 7 films the author picked. They describe that style (modern SaaS / AI
  launch films, kinetic type, UI demos), not every good film. A cinematic short or a quiet piano piece will break
  rules that do not apply to it. Build your own profile from films you admire (`earshot profile`).
- **The "cuts without a hit" list** (timeline, sheet, critique tips): soft build-ups (whooshes, risers, swells) can
  still be missed - on Clawd both cues it named were sounded whooshes. Look at the sheet before adding a sound.
- **critique's "onset on visual events" share** counts only new attacks within +-100 ms (the definition the reference
  ranges were measured with), so a whoosh swelling through a cut does not count; its fixes and tips name only the cuts
  with no sound at all (timeline's matcher). critique's stop-downs (short drop-outs before a hit, per minute) and
  timeline's gaps (silences of any kind) are different counts.
- **realness verdicts depend on length**: a sound up to 6 s is judged with the clip thresholds (lo 0.17), a longer file
  by the mean of its 4 s windows with the weakly calibrated file thresholds (lo 0.40), so a lower p_real can get the
  milder verdict. Judge the trim you will actually use (`--start / --end`), not a multi-take file.
- **AudioSet labels on one-shots under ~0.5 s**: often 'unsure' or a near class (Zipper for a swish, Finger snapping
  for a click). Treat those as neutral; only an outright different class (Speech, Music, Engine) is a red flag.

## Cannot judge
- **Taste and emotion**: whether a sound is beautiful, whether the music fits the brand, whether a moment feels
  satisfying. earshot can say "a strong match for a deep thocky key press, 3 ms attack, lands 4 ms after the cut"; it
  cannot say the film feels premium.
- **A code-made layer hidden under a louder real mix**: usually missed. Judge source files and stems, not only the
  final mix.
- **Provenance**: realness hears synthesis, not licences. A professionally made, licensed synthetic UI sound can read
  "likely code-synthesized"; code-made noise beds that imitate nature (rain, wind) or telephone tones can read real.
- **Anything above 14 kHz from the models** (CLAP's front end stops there; the AudioSet tagger at 8 kHz). Use
  `measure` (bandwidth, bands) for air, hiss and codec cut-offs.
- **Speech content**: no transcription; voice is detected, words are not.
- **Licences**: earshot never decides whether a file may be used. The real-sound skill's `audit.py` checks manifests;
  the licence itself must be read on the item's page.

## Known engineering limits
- Cold start: every CLI call is a fresh process that loads the models (~20-40 s on a busy laptop), even for a cached
  file; the MCP server or one Python script (`from earshot import api`) loads them once. `doctor` without `--load`
  loads nothing heavy.
- bf16 is used only on CPUs with native bf16 (AVX512-BF16 / AMX); elsewhere earshot runs fp32 (bf16 emulation was
  measured 16x slower). The bundled calibration was measured with bf16 embeddings (fp32 agrees to cosine >= 0.9993 on
  12 real clips, `eval/semantics/precision.json`).
- Caches are keyed on path, size, modification time and 16 sampled 4 KB blocks of the file: a re-render that differs
  only between those blocks and keeps the old size and timestamp would still read stale. Use `timeline --fresh` or
  touch the file when in doubt.
- The models were evaluated on one sound library (Freesound / Mixkit picks for one film), 7 launch films and three
  synthesis code bases. Other domains (orchestral recording sessions, field recordings in other climates, other synth
  engines) are untested.
- The optional Gemini critique has not been validated against real answers (it needs the user's key and consent to
  upload).
- Cut detection marks big visual changes (a blinking cursor counts); slow crossfades are missed.
