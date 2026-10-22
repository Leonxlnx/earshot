# earshot fit: translate a recording into code

**Can an audio file be translated into code? Yes, within limits.** The method is *analysis by synthesis*. You pick a
small synthesis model with readable parameters and fit those parameters until the model's render matches the
recording. The fitted model is then written out as a Python module whose `render()` makes the sound. Examples of the
parameters: the frequency, decay and level of each ringing mode, a noise band's loudness over time, a partial's
pitch glide.

The code does not store the recording's samples. It stores the structure of the sound: modes, partials, band
envelopes. From that structure it can make new takes (fresh noise each seed), variations (knobs: size, damping,
pitch, brightness, decay, width, ...), and morphs between two recordings.

On 12 real references, the renders of the generated code sit close to the originals:
- median CLAP cosine 0.945 (minimum 0.881);
- spectrograms a median 0.75 dB further apart than two renders of the same model are from each other;
- `earshot describe` gives the same top tag for both in 10 of 12.

They are close but not identical: measured differences remain in onsets, tone balance and noise texture, listed
below. In the mini blind test (below) the listener could not tell the recordings from the renders.

## How it works

Four fitters, each analysis by synthesis, each producing a plain-JSON model:

| fitter | for | the model |
|---|---|---|
| `fit/modal.py` | struck, ringing sounds (clicks, taps, keys, impacts) | per strike, damped sinusoids `[Hz, decay, amplitude, phase]` (ESPRIT on sub-bands + least squares); the contact as fast modes; a sub-20 Hz pulse; band noise envelopes; for a stereo recording the side over time per octave band (`side_t`, 0.4) |
| `fit/sines.py` | tonal sounds (dings, chimes, pops with a pitch) | tracked partials with frequency and level breakpoints (McAulay-Quatieri / SMS; glides kept); strike atoms; 1/3-octave residual noise including the recording's own floor; for a stereo recording the side over time per octave band (`side`, 0.4) |
| `fit/spectral.py` | noisy sounds (whooshes, swishes, cloth) | a time-varying 48-band noise gesture, stereo per band group, flutter / crackle texture statistics, tonal ridges; motion knobs (doppler, proximity, pan, motion) read from its trajectory (0.4) |
| `fit/gesture.py` (0.4) | designed gestures (whooshes, swells, risers, camera moves), and short sounds where tones plus a little air beat partials plus residual noise | tones (the deterministic part of the sines candidate) + air: a detail noise model of what the tones leave in the recording, split in the magnitude domain (an STFT mask), so a take is fresh air around the same tones, not a re-jittered bank of noise partials |

`fit/space.py` (0.4) adds a fitted diffuse tail to the chosen model of a stereo recording whose tail is decorrelated
(a seeded room in the side: the mono sum and every mono measure stay as they were). `fit/auto.py` is the engine on
top of them.

0. **Strip a bed.** A recording with a steady pre-roll 12-40 dB under the sound (a film's music under an effect) is
   stripped (`bed_of`, `strip_bed`) and the sound alone is fitted from its onset. The bed's spectrum is kept as
   `post['bed']`: `render()` puts a steady noise with it back under the sound, so the fit is scored against the
   recording as it is; `render(bed=False)` gives the sound alone. `earshot fit --keep-bed` fits the recording as it
   is. Analysis by synthesis decides whether it was a bed (0.4 round C2, `BED_CHECK`): after the choice (step 4) the
   winning candidate's kind is fitted on the recording as it is too, and the bed is dropped when that scores lower
   on the same seeds. A whoosh's own steady approach 13.6 dB under its peak had been stripped as music, and every
   candidate fitted on the rest fell (CLAP 0.88 -> 0.64); on the 12 dev clips with a bed the plain fit won on 8.
1. **Measure the sound.** `character()` measures attack, decay, crest, tonal share, partial persistence, glide (2
   semitones over 15 ms count; before 0.4 the gate never fired on the reference clips' 2-6 semitone pops and dings) and
   flatness, and names a class (glide, ringing, struck, gesture, mixed). The class orders the candidates.
2. **Fit every candidate.** Under 1 s that is all seven: modal, sines, gesture, spectral, and three hybrids. Longer
   sounds get the class's four (a long gesture, mixed, glide or ringing sound gets the gesture model second).
   - In a **hybrid**, each layer but the last keeps only its deterministic part, and the next layer is fitted on
     what is left. Example: sines partials, then modal modes on the rest, then spectral noise on what remains.
   - So a pop becomes chirp + body + air, and a key press becomes body + air.
   - A sines layer of a stereo recording keeps its fitted side (0.4), so what it leaves for the next layers is taken
     out in stereo: the air models only the side the tones do not hold (a panned tone's side, counted twice before,
     came out about 3 dB too wide).
3. **Measure each candidate's post stage** from two renders (analysis by synthesis again):
   - a smooth third-octave EQ that moves the render's average balance onto the original's;
   - the level;
   - the original's attack envelope. Before the onset a render may not stand above it: band noise and window smear
     leak energy before a strike, which softens the attack. Over the first 6-25 ms after a sharp onset the render
     follows the original's contour within ±9 dB. This fixed a ding whose onset the sines fit had blunted from 60
     to 4 dB/ms, and noise up to 12 dB above the original in the 4 ms before a key's strike. For a noise model
     (spectral, gesture) the clamp's correction is high-passed at 200 Hz and kept only when it lowers the score
     (its gain steps follow the noise's own flicker and put energy into empty low bands);
   - the gate (0.4): a cut-only gain curve where the renders stand over the original after the onset: the gaps of a
     run of pulses, a hard stop, a tail the model lets ring on (modes and partials decay exponentially and cannot
     stop; a gated pulse falls 30-60 dB in 5 ms). Kept only when it lowers the score and, with `--clap`, costs at
     most 0.005 of CLAP. Knobs that move time move it: a sines stretch and a gesture length about the first strike
     (the attack clamp too), a spectral length / speed about the start, and a modal spacing strike by strike (the
     cut after a strike moves with it and holds across a longer gap or ends where a shorter one does; the reopening
     in the last 2 ms before a strike moves with that strike). A morph warps each source's gate through its strike,
     onset or peak anchors before the two mix, so the cuts sit in the morph's gaps.
4. **Score each candidate's render against the original** and keep the lowest (`quality.score`, metrics below).
   `--clap` adds CLAP to the score; loading CLAP takes about 50 s. With seed 0 alone, the best three are scored on
   two more seeds and the choice is made on the mean (`CONFIRM`): on short noise-led sounds one seed's CLAP spread
   0.84-0.92, and the seed-0 choice took the lucky draws. Then the CLAP margin rule (0.4 round C2, `CLAP_MARGIN`): a
   candidate whose CLAP is at least 0.02 higher on seed 0 and on the mean over the confirm seeds, whose score is at
   most 1.5 higher (both) and whose envelope r is at most 0.05 lower, is chosen instead. The model-free terms (mel,
   onset, envelope, excess) had decided near-ties against a clearly closer sound: on the first held-out check five of
   six regressions were the gesture model chosen with 0.015-0.06 less CLAP than another candidate.
5. **Finish** (0.4). A stereo recording with a decorrelated tail gives the chosen model `fit/space.py`'s side layer.
   Every render ends with a fade-in (2 ms, shorter when the onset comes earlier) and a gain down to -1 dBFS when it
   peaks over that (18 of 464 takes of the reference clips' fits had peaked over 0 dBFS). A model fitted before 0.4
   (modal, sines or hybrid without the `coherent` flag: the 0.3 bank's fit entries; spectral without the `finish`
   flag every 0.4 spectral fit carries) renders as it did, without either (`auto.finished()`): the presets' takes
   stay byte-identical to 0.3.0, and a 0.3 spectral fit or its recipe module renders sample for sample.

`fit/codegen.py` then writes the model as a **forge recipe module**. It contains:
- `TARGET` / `ALTS`: from the refset category's recipe when the source is a refset file;
- `REFSET` / `REFS`;
- `SPACE`: the kind's knobs with search ranges around the identity;
- `DEFAULTS`: the identity, which re-renders the fit;
- `MODEL`: the fitted parameters as a readable literal, one mode, partial or keyframe per line;
- `render(p, r)` and `render_seed(seed, **knobs)`.

The module's render is **bitwise identical** to the engine's render of the model. This is tested, and true for all
12 validation fits.

Variation comes in three forms:
- **Knob perturbations:** the `SPACE` of the recipe, searched by the forge critic with `earshot fit --forge`.
- **`vary`:** seeded jitter of modes and partials, so each take sounds like a new take of the same object.
- **Morph:** two fits of the same kind.
  - Paired modes and partials interpolate: frequency, decay and level geometrically, phase circularly. Pairing is
    an optimal assignment on log frequency within one octave.
  - Unpaired ones fade with their side's weight.
  - Envelopes interpolate in dB on a time axis warped through the strikes or the loudness peak.
  - Hybrids morph layer by layer, and the post stages interpolate too.
  - A morph recipe has a `morph` knob (0.2-0.8 in `SPACE`), so a search explores sounds between two recordings,
    not copies of either.

## Commands

```
earshot fit click.wav -o made/click_fit.py            # recipe + made/click_fit.wav + .png (side by side) + .json
earshot fit click.wav -o made/click_fit.py --report   # + render-vs-render floor + CLAP cosine -> click_fit.report.json
earshot fit pop.wav --kind hybrid --layers sines,modal,spectral --render pop_fit.wav
earshot fit whoosh.wav --clap                         # let CLAP help choose the fit (loads CLAP, ~50 s)
earshot fit swell.wav --kind gesture                  # tones + air (0.4)
earshot fit film_clip.wav --keep-bed                  # fit a clip with music under it as it is (0.4 strips the bed)
earshot fit --morph a.wav b.wav --amount 0.5 -o made/ab_morph.py
earshot fit --forge made/click_fit.py --budget 96 --dir made/click_variations   # critic-searched variations
python made/click_fit.py 3 take3.wav                  # the module renders on its own (seed 3)
```

`--forge` loads the recipe from its path and runs the forge search (pre-screen, robust re-renders, report.md,
compare.png, audition.wav). Alternatively, copy the file into `earshot/forge/recipes/` and run `earshot forge <name>`.
Picks of a recipe loaded from a path cannot be re-rendered by `earshot forge --check`, which needs an importable
module name.

```python
from earshot.forge import fit
from earshot.forge.fit.quality import quality
m = fit.fit('click.wav')                  # kind='auto' | 'modal' | 'sines' | 'spectral' | 'gesture' | 'hybrid'
y = fit.render(m, seed=0, vary=0.3)       # knobs: earshot.forge.fit.auto.knobs_of(m)
q = quality('click.wav', fit.render(m))   # the metrics below; clap=True / describe=True add the model checks
src = fit.to_recipe(m, name='click_fit', source='click.wav')
mm = fit.morph(m_a, m_b, 0.5)
```

## Quality metrics (`fit/quality.py`)

Both sounds are mono-summed (except for the stereo measures). Both are cut at the original's onset, so a timing error
still counts. Each is levelled to its own loudest 50 ms; the level difference is reported separately.

| metric | what |
|---|---|
| `mel_db` | multi-resolution log-mel distance (64 bands, FFT 256 / 1024 / 4096), mean \|dB\| over cells within 60 dB of the top; 0 = the same spectrogram |
| `env_corr` | correlation of the 5 ms dB envelopes |
| `onset_err` | \|log2\| of the onset-sharpness ratio. Sharpness: 17 dB over the time the first onset takes from 20 to 3 dB under its local peak (0.125 ms frames, interpolated) |
| `tone_db` | RMS third-octave balance difference (`earshot.forge.tone`) |
| `excess_db` / `missing_db` | energy the resynthesis has where the recording is quiet, or the reverse. Mean dB beyond 6 dB over third-octave band x 10 ms cells, looking 100 dB deep. It catches hiss bands, noise floors and rings that outlive the original, where mel and envelope stop looking. It tracks CLAP's verdicts closely |
| `side_db`, `width_db` (0.4) | side / mid energy of each sound over the original's active span, and the per-octave-band width error weighted by the original's band level |
| `tail_corr`, `tail_corr_err` (0.4) | L/R correlation over the original's tail (from its loudest 10 ms + 30 ms to the end of its tail) of each sound, and their difference: a reverberant space decorrelates |
| `clap_cos`, `describe` | the model checks (optional) |

**floor**: `quality(render seed 11, render seed 12)`. A model with noise in it can never beat its own floor on
`mel_db`.

**score** = mel + 0.3 tone + 8 (1 - env r) + onset (capped at 2) + 1.5 excess + missing + 0.04 width (capped at
20 dB) + tail correlation error [+ 20 (1 - CLAP)]. The stereo terms (0.4) cost a mono re-make of a wide recording
about what 0.06 of CLAP costs: they decide between candidates that sound alike (before 0.4 a sines render of a wide
ding at side / mid -16 dB against -5 scored as well as a wide one).
- With excess and missing, the model-free score's choices had a mean CLAP of 0.927 on the 12 refs; without them,
  0.856 (the worst choice 0.52). Measured on the same refs, so in-sample.
- The CLAP-including choice agreed with the model-free one on 11 of 12.

## Measured quality

The **12 forge-split refs** (never bench) are 2 each of click, typing_key, pop, impact, whoosh_fast and notify: the
first file of each category and the first from another source.

For each ref: every candidate was fitted, the engine chose (with `--clap`), the recipe module was written and
imported, and **its** render (seed 0) was measured. Results are in `out/fit/validation/` on the dev machine
(`summary.md`, `all_metrics.json`, `<category>/<stem>/pair.png`, the recipe modules, the WAVs; local, because they
are derived from licensed recordings).

| category | ref | chosen | mel dB (floor) | env r | tone dB | onset err | excess dB | CLAP (floor) |
|---|---|---|---|---|---|---|---|---|
| click | fs-167829_1 | sines | 3.15 (2.35) | 0.997 | 1.14 | 0.27 | 0.16 | 0.959 (0.967) |
| click | kit-ui_click_01 | sines + modal + spectral | 3.26 (2.69) | 0.993 | 1.38 | 0.03 | 0.06 | 0.948 (0.962) |
| typing_key | fs-561697_1 | modal | 4.65 (2.49) | 0.975 | 1.93 | 0.60 | 1.06 | 0.881 (0.976) |
| typing_key | kit-kb_thock_01 | modal | 4.25 (3.28) | 0.946 | 4.40 | 0.87 | 0.49 | 0.944 (0.984) |
| pop | kit-ui_pop_01 | sines + modal + spectral | 3.89 (2.50) | 0.970 | 2.05 | 0.03 | 0.79 | 0.959 (0.988) |
| pop | mk-3000_1 | sines | 6.12 (3.24) | 0.989 | 3.45 | 1.36 | 1.15 | 0.923 (0.989) |
| impact | kit-impact_01 | sines | 4.41 (3.92) | 0.964 | 1.99 | 1.12 | 0.68 | 0.910 (0.978) |
| impact | kit-impact_02 | sines | 4.32 (3.96) | 0.919 | 0.61 | 0.68 | 0.59 | 0.953 (0.991) |
| whoosh_fast | fs-719636_1 | modal | 3.31 (5.56) | 0.972 | 2.22 | 0.78 | 0.84 | 0.898 (0.948) |
| whoosh_fast | kit-whoosh_fast_01 | sines | 3.54 (3.40) | 0.984 | 2.45 | 1.49 | 1.25 | 0.947 (0.973) |
| notify | fs-681787_1 | sines + spectral | 2.03 (1.81) | 0.999 | 1.48 | 0.41 | 0.32 | 0.964 (0.982) |
| notify | kit-ui_notify_01 | sines | 1.87 (0.72) | 0.998 | 1.18 | 0.48 | 0.89 | 0.937 (0.985) |

Per category, the mean CLAP of the two refs:

| category | mean CLAP |
|---|---|
| notify | 0.95 |
| click | 0.95 |
| pop | 0.94 |
| impact | 0.93 |
| whoosh_fast | 0.92 |
| typing_key | 0.91 |

All 12:
- median CLAP 0.945; 4 of 12 reach 0.95;
- median mel 3.71 dB against a median floor of 2.96;
- median envelope r 0.98;
- median tone 2.0 dB;
- `earshot describe` top tag the same in 10 of 12. The two that differ: 'an enter key press' was heard as 'a soft
  UI button tap', and 'a cinematic hit with a long tail' as 'a deep cinematic whoosh'.

**Each fitter alone versus the engine.** The builders reported each fitter alone at a CLAP median of about 0.90.
The engine's choice per sound plus the post stage reaches 0.945. Two things carry that gain:
- The three-layer hybrid won on clicks and pops.
- The sines fitter won on impacts. The spectral and noise-heavy candidates of the impacts scored CLAP 0.47-0.65,
  caught by `excess_db`: a hiss band and a noise floor through the tail.

**Size and time:**
- Recipe modules are 24-274 KB; the biggest is a whoosh fitted as about 600 modes in 10 strikes.
- `earshot fit` on a 0.2 s pop takes 12 s (six candidates, model-free choice).
- Fitting all six candidates with CLAP scoring took 8-98 s for short sounds. The 4.5 / 6.5 s impacts took
  250-340 s with four to six candidates.

## The mini blind test

`out/fit/blind_fit/` holds 12 trials. In each, one clip is the real recording and the other is the render of its
generated recipe module, in random order (the recording is A on exactly half).

The files:
- `blind_test.wav` (65 s), or `blind_test_trials/tNN_A.wav` + `tNN_B.wav` one trial at a time;
- `answer_sheet.txt`;
- `KEY_open_after.json`, the answers: do not open it first;
- `cleanliness.json`, safe to open: no answers;
- `score_answers.py`.

Trim, level and the cleanliness checks are the clean blind test's own: `out/fit/build_blind_fit.py` imports
`eval/forge/blind_v2.py`'s functions.
- Both clips are cut at the same onset and end levels above the recording's floor, so neither plays room noise.
- They are matched on the K-weighted loudest 100 / 400 ms, and true peaks stay at or under -1 dBTP.
- Cleanliness PASS: loudest moments 0.00 LU apart.
- 2 warnings: recording floors at -53 and -59 dBFS under the sound, 10 dB below where the clips start and end.

After listening: `.venv\Scripts\python.exe out\fit\blind_fit\score_answers.py A B ...` (12 letters). 10 or more
right is unlikely by luck (p = 0.019).

**Result:** one listener took the test and could not tell the recording from the render; both clips of a trial
sounded the same to them. One listener and 12 trials is a small test, not a proof. The kit stays local, because it
plays licensed recordings.

## Morph examples

`out/fit/morphs/` has three morphs:
- `click_morph` (hybrid);
- `key_morph` (modal, two key presses);
- `ding_morph` (sines, two dings).

Each comes with:
- the morph recipe;
- its render at 0.5;
- a side-by-side spectrogram (A | morph | B);
- `<name>_audition.wav`: A, morph 0.25, 0.5, 0.75, B, peak-matched.

The 0.5 morphs sit between their sources: CLAP 0.48-0.76 to each, so they are new sounds, not copies.

## Takes: a fit rendered for use (0.3)

A fit is measured against a trimmed recording: the identity render (seed 0, no knobs) keeps the recording's length
and every bit of its low end, and that is what the tables above measure. A take for a film is another job, and the
0.2 renderer made takes with three measured defects. Since 0.3, `fit.render(model, seed, vary=v)` with `vary > 0`, or
any render with `take=True`, renders a take (the bank renders every fit entry so):

| defect in 0.2 takes | measured on | fix (take mode) |
|---|---|---|
| tails cut where the trimmed recording stopped: fitted modes outlive it, vary lengthens decays further, a sines residual stops on the last sample | a cut tail in the takes of 13 of 46 CC0 fits (keys, clicks, a clip, a key release, a whoosh) | modal: modes ring on until all are 60 dB under the loudest strike's mode sum, at most 0.5 x the sound's span longer (at least 60 ms; 200 ms when the modes are still above -50 dB at the cut), the added stretch fading 60 dB linear in dB. sines: renders to the last partial's release, the residual noise fading out over 60 ms past the recording |
| a random DC offset per seed: a sines residual's lowest band is flat to DC, and its envelope spreads energy down there | takes of one whoosh at -31 to -34 dB re peak (the audit warns over -40) | takes are high-passed at 8 Hz (2nd order) |
| a spectral model's takes 2-7 LU apart: low noise bands hold few independent bins over a short gesture, and each take was levelled by its own loudest 50 ms | a cloth whoosh 7.3 LU, a key 3.8 LU, a cape 1.4 LU over 8 seeds | each take is levelled by its expected loudest 50 ms (from the gesture) and corrected to its expected K-weighted energy: 0.45, 2.6 and 0.15 LU |

Two more 0.3 changes:
- **The sines residual follows the recording's roll-off above 16 kHz.** The top residual band is flat to Nyquist, so a
  fit put noise above 20 kHz that a 16-20 kHz band-limited recording never had (a click: -19 dB of its energy above
  20 kHz against -34 dB in the recording). A sines fit now stores `residual.hf`, the residual's spectrum above
  16 kHz relative to 16 kHz, made monotone, and the noise follows it; `sines.hf_rolloff(x)` measures it for an older
  model. On the six sines validation refs (mean of 6 seeds) the energy above 20 kHz moved toward the recording's
  (click -19.5 -> -30.0 dB, recording -33.5; impact -44.8 -> -60.5, recording -62.4; whoosh -41.8 -> -60.1, recording
  -75.3) and mel dB changed by -0.10 to +0.00.
- **`earshot fit --takes`** scores every candidate on seeds 0, 1, 2 instead of seed 0 alone
  (`auto.fit(seeds=auto.TAKE_SEEDS)`). A candidate with noise in it can win on one lucky draw: on 10 of the 12
  validation refs the chosen fit's seed-0 score sat 0.1-2.4 under its mean over seeds 0-7.

**The fit itself did not move.** Every identity render of the 12 validation refs is the same as in 0.2 (a hybrid's
noise layer keeps its 0.2 levelling): seed-0 and seed-averaged quality (seeds 0-7: mel, envelope, tone, onset,
excess, missing, CLAP) are identical, so the tables above stand.

**0.4 takes.** Measured with the replication benchmark's take check (below), the bank's own check run on four takes
(seeds 1-4, vary 0.3) of each fit:
- **Modal vary keeps the fit's structure.** A modal fit's modes are tied to each other by the least squares (close
  modes, and a later strike's modes, shape and cancel the earlier ones' ring). Vary used to detune every mode on its
  own (0.6 % x vary) and jitter the strike times: even 0.1 % undid the cancellations, so a key's takes rang on 2.5x
  the recording's tail. In a model fitted by 0.4 (marked `coherent`) vary moves the modes by a smooth random
  function of log frequency (neighbouring modes and the same mode in every strike move together), strikes keep their
  times, and when a knob moves a strike or a mode's frequency (vary, size, spacing) every mode keeps its phase on the
  first strike's timeline. On the modal and hybrid dev fits the takes now score as vary-0 takes do (a key
  0.75 -> 0.89 CLAP to the recording, a pop 0.87 -> 0.94); 16 of 29 pass the take check, 11 before. A modal or
  hybrid fit also carries `vary_scale` 0.5 (`auto.TAKE_VARY`): the bank's vary 0.3 jitters it as 0.15 did. Models
  without the flag (the 0.3 bank's fit entries) render exactly as before.
- **Sines vary moves neighbouring partials together** in a coherent model (`sines.vary_field`): independently
  detuned, a harmonic series' upper partials went off the harmonic grid and takes read as aliasing or rang.
- A coherent modal ring-out continues the noise un-faded, like the modes (it used to fall at the recording's
  fade-out slope, leaving the modes alone in the added stretch).
- Every render starts at zero and peaks at -1 dBFS at most (above).
- All of it on the dev fits' saved models: takes passing 39 -> 44 of 78, median take CLAP 0.890 -> 0.896.
- **Clicks** (round C2), traced to three sources and fixed where they start: the post gate's breakpoints sit 0.5 ms
  apart, and a 30-80 dB step between two of them put a kink into a loud ring (its linear gain is now smoothed by a
  1 ms Hann window, `auto.GATE_SMOOTH_S`); a later modal strike's modes started at full amplitude inside the ring of
  the strikes before it, a step in the fit itself (they now rise over 0.15 ms, `modal.STRIKE_RAMP_S`, in a coherent
  model; the first strike keeps its step); a sines strike's atoms are fitted jointly and cancel each other's step at
  the onset, and vary jittered each on its own (a coherent model now jitters a strike's atoms together).
- **The gesture air's grains** (round C2): the air's grain texture is a Poisson click train with log-normal
  amplitudes, and one draw could put its loudest clicks where the sound is quiet: takes of short noise-led fits
  (keys, ticks, toggles fitted as air) spread 0.84-0.92 CLAP from seed to seed, and some clicked. The gesture air now
  levels the train to unit power over 5 ms windows (at least 20 clicks; `grains['norm_ms']`, `spectral.fit(
  grain_norm_ms=)`) and keeps grains only when their excess fast variance is in both halves of the active span
  (`grain_halves`: stationary texture, as the AM is judged). Levelling every noise band instead (the first idea)
  cost CLAP on 6 of 9 such fits: it removes fluctuation the recordings have. On 20 dev clips the takes' median gap
  under seed 0 fell 0.046 -> 0.031 CLAP, take fails for dropping more than 0.06 under seed 0 7 -> 5.

## The replication benchmark (0.4)

`eval/forge/replicate.py` asks whether `earshot fit` can re-make professional sound effects. The 0.4 study used a
private set of licensed reference clips: 116 clean one-gesture clips in 24 categories (they, the renders and the
per-clip results stay local). The 0.4.0 rows below were measured before the release review's hybrid side fix (step 2:
a sines layer's stereo side is left out of the residual), which can change hybrid candidates on stereo tonal clips;
they were not measured again. Each clip is fitted with `--clap`; its seed-0 render, the model's floor and four
takes (seeds 1-4, vary 0.3) are measured against it. Grades: **high** CLAP >= 0.92 and mel gap <= 2.5 dB (the bank's
ship bar), **good** >= 0.88 / <= 3.5, **fair** >= 0.80 / <= 5.0, **poor** the rest. "Takes pass" is the bank's take
check on all four takes. A fixed split by clip id holds 38 clips out; 0.4 was tuned on the other 78 only, and the
held-out clips were fitted once, at the end.

| split | engine | CLAP (median) | mel gap dB | high / good / fair / poor | takes pass |
|---|---|---|---|---|---|
| dev (78) | 0.3 | 0.884 | 1.43 | 19 / 20 / 31 / 8 | 29 |
| dev (78) | 0.4, first round | 0.907 | 0.90 | 30 / 26 / 20 / 2 | 46 |
| dev (78) | 0.4.0 (round C2) | 0.921 | 0.85 | 40 / 20 / 17 / 1 | 42 |
| held-out (38) | 0.3 | 0.878 | 1.73 | 7 / 8 / 19 / 4 | 21 |
| held-out (38) | 0.4, first round | 0.909 | 0.98 | 14 / 9 / 12 / 3 | 25 |
| held-out (38), opened: a reference | 0.4.0 (round C2) | 0.903 | 1.04 | 14 / 11 / 12 / 1 | 25 |
| fresh held-out (28) | 0.3 | 0.853 | 0.87 | 2 / 6 / 14 / 6 | 8 |
| fresh held-out (28) | 0.4, first round | 0.830 | 0.75 | 4 / 3 / 14 / 7 | 12 |
| fresh held-out (28) | 0.4.0 (round C2) | 0.851 | 0.58 | 5 / 6 / 13 / 4 | 7 |

(The first-round dev row is the committed engine's own run; an earlier run of the same changes before the confirm
stage went in read 0.907 / 0.86 / 32 / 24 / 18 / 4.) On the first held-out split the gain matched the dev gain (CLAP
+0.031 against +0.023; high 7 -> 14 against 19 -> 32): per held-out clip the grade went up on 14, stayed on 20 and
went down on 4. The held-out width misses fell 15 -> 0 of 38,
the tail correlation misses 9 -> 1 of 29. Of the six held-out regressions, five are the score choosing the gesture
model on its mel, onset and envelope terms over a candidate with 0.015-0.06 more CLAP; one is a bed false positive (a
whoosh's own steady approach 13.6 dB under its peak taken for music and stripped: CLAP 0.88 -> 0.64).

**Round C2 and the fresh held-out set.** The first held-out split was opened by that check, so before any further
tuning a second one was drawn from the same private set of licensed reference clips by a fixed rule (sha1 order
inside per-category caps, nothing within CLAP 0.97 of a benchmark target, no fitting or listening) and frozen
(`replicate.py --split-file`). The clean clips had all gone into the benchmark, so its isolation bar is lower (7 dB
over the local bed against 12): a harder, more bedded set (the bed detector fires on half of it), compared engine
against engine on the same 28 clips. Round C2 (bed check, CLAP margin rule, take clicks, gesture air grains: above)
was tuned on dev only, run once on dev, then opened the fresh set once:
- The first-round 0.4 engine had not beaten 0.3 on the fresh clips (median CLAP 0.830 against 0.853; grades up on 5,
  down on 6): its bed stripping cost the stripped fits up to 0.16 of CLAP against a fit of the recording as it is.
- Round C2 lifts it on the fresh clips by about what it gained on dev (median +0.021, mean 0.840 -> 0.860; grades up
  on 8, down on 2; take CLAP 0.824 -> 0.863), level with 0.3 on the median and ahead on the mean (0.844), the mel gap,
  the grades and the take CLAP.
- Takes passing fell 12 -> 7 there (dev 46 -> 42): a plain fit of a bedded clip carries the bed as part of its sound,
  so its takes vary it and end it at the clip's end (four of the six new failures). The onset error also grew (0.53 for
  0.3, 0.82 first round, 1.00 now).
- On the opened first held-out split: CLAP mean 0.894 -> 0.903 (median 0.909 -> 0.903), poor 3 -> 1, takes 25 -> 25;
  the whoosh whose approach was stripped as a bed is back at 0.88.

What moved the dev numbers, each measured on the dev split by the change's own run:
- **The gesture model** won 38 of the 78 dev clips. On the 9 dev clips of the gesture categories (camera air,
  glitch, reverse swell, riser, swish, fast whoosh, shimmer) it lifted the median CLAP 0.858 -> 0.885 and the takes
  passing 3 -> 7.
- **Stereo**: on 29 wide-category clips the width misses (over 3 dB) fell 16 -> 0 and the tail correlation misses
  (over 0.3) 13 -> 2.
- **The gate** and the **modal pre-onset cap**: on the 67 gated, multi-pulse and short-click dev clips the gated
  clips' gate error 21 -> 1.7 dB; modal onset error 0.39 -> 0.25 octave.
- **Beds**: 12 of 78 dev clips carry a music bed; fitting the stripped sound and rendering the bed back under it
  (rather than rendering it dry) kept clips like a camera move at CLAP 0.85 instead of 0.72.
- **Takes**: coherent vary (above) 39 -> 44 of 78.
- **Round C2** (from the first-round engine's own dev run, 0.907 / 30 / 26 / 20 / 2 / 46): the bed check on the 12 bed
  clips CLAP 0.907 -> 0.928 (the plain fit won on 8); the CLAP margin rule 5 choices, all to a higher CLAP; the click
  fixes cleared the traced clicks (four clips' takes); the grain levelling cut the takes' gap under seed 0 0.046 -> 0.031 on
  20 clips. Together: 0.921, 40 / 20 / 17 / 1, takes 42.

Still weak on dev: takes (42 of 78; a take 0.06 under seed 0 on 12 clips, loudness spread over 2 LU on 9, a comb ring
on 8); modal ring-outs of short clips read as a comb ring (modes whose decays the clip could not pin ring alone once
the noise fades); multi-pulse counts on long gesture wins (7 of the 28 clips with 3 or more pulses miss by more than
2); clips of 0.5 s or longer reach high on 3 of 13.

**Known gaps after 0.4.0**, for the next round:
- The bed check decides by the score alone: it needs a CLAP guard (a margin like the choice rule's) and should weigh
  the takes. A plain fit of a bedded clip keeps the bed in its sound, which a kit should not have (`render(bed=False)`
  separates only a stripped fit); one option is to use the plain fit's score only to decide whether the bed is real
  and keep the stripped model's sound.
- Onsets: 0.53 octave for 0.3 against 1.00 for 0.4.0 on the fresh held-out set (14 of 28 clips off by more than an
  octave). Measuring the onset after the bed and a sharper first grain of the gesture air are the leads.
- Take clicks left: a loud grain inside an air-only key's attack window (levelling over 5 ms cannot stop it) and one
  modal-layer click in a hybrid take; the comb ring of short modal ring-outs (capping the ring-out trades against the
  audit's 'truncated').
- Takes passing fell in round C2 (dev 46 -> 42, fresh held-out 12 -> 7) although the take CLAP rose; the loudness
  spread and ring borderlines need their own look.
- Both held-out sets are opened. Under the fresh set's rule only 25 eligible clips remain, fewer in most categories:
  a next held-out set needs more source material or a relaxed rule.

## Licence

A fit of a recording is a derivative of that recording: the parameters are measured from it. Keep the source's
attribution and licence with the generated recipe; its docstring names the source.
- Prefer CC0 sources for anything shipped as a seed.
- A morph across several sources is less derivative of any one of them. Keep every source's attribution anyway.
- The refsets' recordings are licensed and are not redistributed. The fitted models and recipes of them, under
  `out/`, stay local like the refs.

## Limits (honest)

**Close but not identical.**
- The CLAP target of about 0.95 is reached on 4 of 12.
- `mel_db` stays 0.1-2.9 dB above each model's own floor on 11 of 12; one whoosh sits under its own unusually high
  floor. The worst: the pop mk-3000_1 at 6.1 dB against a 3.2 dB floor, and the key fs-561697_1 at 4.7 against 2.5.

**Onsets.** Even with the attack match, the measured onset sharpness differs by 0.6-1.5 octaves on 6 of 12. The
metric itself is noisy on stochastic renders: different seeds of one model ranged 23-140 dB/ms on a key. On the 0.4
dev fits, 8 seeds of one model spread 0.63 octave (standard deviation) on a whoosh and 0.91 on a riser: on 0.125 ms
frames a noise's own flicker sets the crossings. Reading a slow original on 0.5 ms frames (and its render with it)
cut that to 0.35 and 0.48 and a swish's error from 2.7 to 0.03 octave, but it also re-read 31 of the 78 dev
originals, sharp two-part key onsets among them (58 -> 3 dB/ms), and worsened a glitch (0.9 -> 3.2 octaves); it
needs its own benchmark run and is not in 0.4.

**Noise models.** A noise band is spectrally flat within its band, which fills valleys: the thock key's tone sits at
4.4 dB even after the EQ. The modal key fits have high floors (2.5-3.3 dB): their noise varies take to take more
than the recording suggests.

**Whooshes fitted as modes or partials.** The engine chose modal and sines for the two fast whooshes because they
re-render them closest. Such a model is a sinusoidal decomposition of noise, about 600 modes and a 274 KB recipe.
Its knob variations (size, pitch) may sound metallic. The spectral fitter is the generative choice for whooshes, but
it lost on these two refs. Since 0.4 the gesture model (tones + air) is a candidate for every short sound and second
for long gestures; on the dev clips it won the whooshes, swells, risers and camera moves the sines and modal
fits had taken.

**Morphs.**
- Strikes pair by index, so a key with 2 strikes and one with 4 pair crudely.
- Partials more than an octave apart do not pair, so they fade instead of gliding (the ding morph keeps both pitches
  at -6 dB).
- Exact pole copies that stop a ring in a later strike stop cancelling once morphed.

**The post stage pins the attack** to the original's contour, so the `hardness` knob changes the attack's colour
but not its envelope. The EQ is static under pitch knobs. An envelope calibration (fix the tail's level over time)
is in the code but off: it lowered mel and raised envelope r, but CLAP fell from 0.925 to 0.918. A cut-only
spectro-temporal ceiling was tried and left out: CLAP +0.004, mel -0.3 dB.

**Speed.** Default `earshot fit` takes 10-100 s for short sounds and minutes for long impacts. `--clap` adds about
50 s to load CLAP.

**Choosing.** The weights of `quality.score` were set on the same 12 refs they were checked on. The 0.4 engine was
tuned on the replication benchmark's dev split and checked once on each held-out set (above). The CLAP margin rule
settles clear near-ties for CLAP, but a candidate 0.02-0.03 higher on seed 0 and not on the confirm seeds' mean still
loses. The bed check decides by the score alone: on the fresh held-out set it dropped a bed whose plain fit had 0.087
less CLAP, and a plain fit of a bedded clip keeps the bed in its sound (`render(bed=False)` separates only a stripped
fit), which costs its takes.

**Not tested here:** zips and long glides (the sines fitter tracks chirps, but no zip ref exists), swipes, and
anything from the bench split. The replication benchmark (above) covers risers, shimmers, swells, camera moves and
stereo designed sounds among its reference clips.
