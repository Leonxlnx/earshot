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

Three fitters, each analysis by synthesis, each producing a plain-JSON model:

| fitter | for | the model |
|---|---|---|
| `fit/modal.py` | struck, ringing sounds (clicks, taps, keys, impacts) | per strike, damped sinusoids `[Hz, decay, amplitude, phase]` (ESPRIT on sub-bands + least squares); the contact as fast modes; a sub-20 Hz pulse; band noise envelopes |
| `fit/sines.py` | tonal sounds (dings, chimes, pops with a pitch) | tracked partials with frequency and level breakpoints (McAulay-Quatieri / SMS; glides kept); strike atoms; 1/3-octave residual noise including the recording's own floor |
| `fit/spectral.py` | noisy sounds (whooshes, swishes, cloth) | a time-varying 48-band noise gesture, stereo per band group, flutter / crackle texture statistics, tonal ridges |

`fit/auto.py` is the engine on top of them.

1. **Measure the sound.** `character()` measures attack, decay, crest, tonal share, partial persistence, glide and
   flatness, and names a class (glide, ringing, struck, gesture, mixed). The class orders the candidates.
2. **Fit every candidate.** Under 1 s that is all six: modal, sines, spectral, and three hybrids. Longer sounds get
   the class's four.
   - In a **hybrid**, each layer but the last keeps only its deterministic part, and the next layer is fitted on
     what is left. Example: sines partials, then modal modes on the rest, then spectral noise on what remains.
   - So a pop becomes chirp + body + air, and a key press becomes body + air.
3. **Measure each candidate's post stage** from two renders (analysis by synthesis again):
   - a smooth third-octave EQ that moves the render's average balance onto the original's;
   - the level;
   - the original's attack envelope. Before the onset a render may not stand above it: band noise and window smear
     leak energy before a strike, which softens the attack. Over the first 6-25 ms after a sharp onset the render
     follows the original's contour within ±9 dB. This fixed a ding whose onset the sines fit had blunted from 60
     to 4 dB/ms, and noise up to 12 dB above the original in the 4 ms before a key's strike.
4. **Score each candidate's render against the original** and keep the lowest (`quality.score`, metrics below).
   `--clap` adds CLAP to the score; loading CLAP takes about 50 s.

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
m = fit.fit('click.wav')                  # kind='auto' | 'modal' | 'sines' | 'spectral' | 'hybrid'
y = fit.render(m, seed=0, vary=0.3)       # knobs: earshot.forge.fit.auto.knobs_of(m)
q = quality('click.wav', fit.render(m))   # the metrics below; clap=True / describe=True add the model checks
src = fit.to_recipe(m, name='click_fit', source='click.wav')
mm = fit.morph(m_a, m_b, 0.5)
```

## Quality metrics (`fit/quality.py`)

Both sounds are mono-summed. Both are cut at the original's onset, so a timing error still counts. Each is levelled
to its own loudest 50 ms; the level difference is reported separately.

| metric | what |
|---|---|
| `mel_db` | multi-resolution log-mel distance (64 bands, FFT 256 / 1024 / 4096), mean \|dB\| over cells within 60 dB of the top; 0 = the same spectrogram |
| `env_corr` | correlation of the 5 ms dB envelopes |
| `onset_err` | \|log2\| of the onset-sharpness ratio. Sharpness: 17 dB over the time the first onset takes from 20 to 3 dB under its local peak (0.125 ms frames, interpolated) |
| `tone_db` | RMS third-octave balance difference (`earshot.forge.tone`) |
| `excess_db` / `missing_db` | energy the resynthesis has where the recording is quiet, or the reverse. Mean dB beyond 6 dB over third-octave band x 10 ms cells, looking 100 dB deep. It catches hiss bands, noise floors and rings that outlive the original, where mel and envelope stop looking. It tracks CLAP's verdicts closely |
| `clap_cos`, `describe` | the model checks (optional) |

**floor**: `quality(render seed 11, render seed 12)`. A model with noise in it can never beat its own floor on
`mel_db`.

**score** = mel + 0.3 tone + 8 (1 - env r) + onset (capped at 2) + 1.5 excess + missing [+ 20 (1 - CLAP)].
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
metric itself is noisy on stochastic renders: different seeds of one model ranged 23-140 dB/ms on a key.

**Noise models.** A noise band is spectrally flat within its band, which fills valleys: the thock key's tone sits at
4.4 dB even after the EQ. The modal key fits have high floors (2.5-3.3 dB): their noise varies take to take more
than the recording suggests.

**Whooshes fitted as modes or partials.** The engine chose modal and sines for the two fast whooshes because they
re-render them closest. Such a model is a sinusoidal decomposition of noise, about 600 modes and a 274 KB recipe.
Its knob variations (size, pitch) may sound metallic. The spectral fitter is the generative choice for whooshes, but
it lost on these two refs.

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

**Choosing.** The weights of `quality.score` were set on the same 12 refs they were checked on. The selection has
no held-out test yet.

**Not tested here:** zips and long glides (the sines fitter tracks chirps, but no zip ref exists), risers, swipes,
shimmers, stereo-heavy designed sounds, and anything from the bench split.
