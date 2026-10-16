# Motion-design SFX: how the pros build them, as synthesis recipes

For anyone writing an earshot forge recipe (`earshot/forge/recipes/*.py`) who wants sound effects that hold up in a
motion-design launch video. Research notes, 2026-10-16, branch `forge-v2`.

How to read the numbers:

- **(src N)**: the number or claim comes from reference N at the bottom of this page.
- **[sv]**: a *starting value*. This is synthesis practice, not something a cited source states. Tune it by listening
  and measuring (earshot's tone, polish and envelope checks).
- **(earshot)**: measured by earshot itself, either in `docs/launch-video-sound.md` (7 pro launch films) or in a
  recipe's notes on its real reference recordings.

The web sources are mostly practitioner tutorials and vendor guides. They agree on structure (layers, movement,
processing). They give few hard numbers, and the numbers they do give vary. Where only a search summary was available
and the page itself was not read, the reference says so.

---

## 0. The difference between a "real" sound and a motion-design sound

The forge's v1 recipes are physical models fitted to raw recordings: a button click, a key press, a bubble pop, a
swung stick, a drum boom. Launch-video SFX are a different product. They are **designed**, which means:

1. **Several layers, each with one job.** Every source describes a pro SFX as 2-4 layers that peak together:
   transient / body / sub / tail for hits (src 9, 10), and air / texture / tonal / landing for whooshes (src 2, 3, 4).
   The usual fix for a flat hit is another layer, not another plugin (src 10). A single layer is the amateur
   signature (src 1, 42).
2. **Each layer owns a band.** Layers are EQ'd so they do not fight. Two body layers sharing ~200 Hz cancel each
   other and turn to mud (src 10). Keep 3-4 elements at most (src 7).
3. **Movement.** A whoosh or riser with a static spectrum sounds cheap. "Movement is the whole effect" (src 1). Filter,
   pitch, level and pan move together, along curves that accelerate into the peak (src 7).
4. **Processed and glued.** Light saturation for harmonics (so it survives small speakers), transient shaping, bus
   compression that glues without squashing, one shared space, then a limiter (src 9, 10, 41).
5. **Edited clean.** The file starts on the first sample of the transient with no leading silence (src 16) and has
   no noise floor or room tone unless that is the point. The tail is trimmed but never chopped. Pops in a pro kit last
   0.19-0.41 s, with almost nothing below 150 Hz or above 4 kHz (earshot, `pop.py` notes).
6. **Exaggerated, not literal.** Pitching a sound down makes it bigger and heavier; pitching it up makes it smaller and
   lighter (src 42). Bigger whooshes are lower and slower (src 3, search summary). Designers pitch bright recordings
   down 2-3 octaves to make sub-heavy material (src 9).
7. **A palette.** UI sounds come from one patch, with only pitch and envelope changed (src 16). Earcon families share
   timbre, register and rhythm (src 17). The tonal parts are tuned to the music's key (src 10).
8. **Synced to the frame and used sparingly.** A sound a frame or two off breaks the illusion (src 42). Most motion
   needs no sound, because the music carries it. The big accents are rare (src 18, src 4, earshot).

So for the forge, a physical model is a good **source layer**. A motion-design SFX recipe needs a **design stage** on
top of it: layers, per-layer EQ and saturation, glue, space and an edit. Its references should be designed library
SFX, not raw foley recordings.

### The universal design chain (per sound)

| step | what | start values |
|---|---|---|
| 1 | Band each layer (HP / LP, one static EQ move) | sub < 120 Hz, body 80-800 Hz, transient > 1-2 kHz, air > 2 kHz [sv] |
| 2 | Saturate body and transient lightly, never the clean sub | tanh/tape, drive 3-9 dB [sv]; distortion adds overtones, so EQ and compress after it (src 9) |
| 3 | Shape transients | keep the crack, shorten its tail (src 10) |
| 4 | Glue the sum | bus comp 2:1-4:1, attack 10-30 ms (lets the transient through), release 60-150 ms, 2-4 dB gain reduction [sv]; glue, not squash (src 10) |
| 5 | Space | reverb on body and tail only, sub dry (src 10); pre-delay 10-30 ms keeps the attack clear [sv]; darken the return (LP 3-6 kHz) [sv] |
| 6 | Stereo | transient and sub centred; width from reverb and decorrelated layers; delays of 2-50 ms fuse into one sound rather than an echo (src 36), but check the mono fold-down (src 4) |
| 7 | Finish | HP at 20-25 Hz (energy below 25 Hz wastes headroom, src 10); look-ahead limiter at -1 dBFS true peak; loudness-match the family (src 16) |

The round-1 `earshot/forge/fx.py` (in progress; names may change) already has most steps: `saturate`, `transient`,
`reverb`, `limit`, `width` / `decorrelate` / `pan`, `doppler`, `sweep_filter`, `tape_stop`, `crush`, `stutter`,
`glitch`, `granular` and `reverse_swell`. `earshot/forge/polish.py` detects the artifacts that make a sound cheap:
clicks, aliasing, DC, rings, chopped tails, leading silence and loudness spread.

### Variation (round robin)

Repeated sounds need variants: a few, chosen at random (src 22, search summary). Typing must never repeat the same
sample twice in a row (earshot). Per-render jitter: pitch +-0.3-1 semitone, level +-1-2 dB, timing and decay +-10 %,
with fresh noise seeds [sv]. Timbre stays fixed per *family*, so a set sounds like one product.

---

## 1. Whoosh / swish / swipe

**Anatomy** (src 1-5)

| layer | job | notes |
|---|---|---|
| air body | the motion itself | filtered noise band that moves; pink is smoother than white (src 1) |
| texture | material and distance | cloth, paper, brush (src 4); cymbal swells exaggerate the peak (src 2) |
| tonal / style | identity, "digital" or "designed" feel | optional, and it gets repetitive if overused (src 4) |
| low weight | size | sine or hum for big moves (src 2, 3) |
| landing | the stop | a separate impact, click or pop, only when the object visibly stops (src 4) |

All layers share one peak moment (src 2, 3). Doppler: the pitch rises on approach and falls after the pass (src 1).

**Durations, matched to the move.** For quick UI swipes or fast cuts, 0.25-0.5 s (src 1). A small card crossing ~10 %
of the screen takes about 180 ms (src 4). A title reveal is about 0.48 s (src 4). A general whoosh is ~0.8 s (src 1).
A product fly-by is 1.15 s with its peak at 620 ms, i.e. 54 % (src 4). A heavy close pass is about 1.2 s (src 4), and
big reveals run 1 s or more (src 1). Make separate short, medium and long variants rather than time-stretching one
file, because stretching smears transients (src 4).

**Envelope.** The sound starts quietly, builds quickly to its loudest point, then trails off (src 3). Mark three
points: prepare (motion starts), cross (fastest frame or cut), release (src 4). Put the peak at 50-70 % of the length.
The rise follows an accelerating curve (power 2-3, or exponential in dB); the fall is an exponential decay at about
twice the rate of the rise [sv]. Loudness of a moving object goes roughly as speed³ (dipole radiation ~U⁶), which
gives a crisp, short top (earshot `whoosh.py` notes). Shorter, steeper rises read as sudden; keep anticipation short
when the visual should feel sudden (src 4).

**Synthesis** [sv unless marked]

- Source: pink noise (src 1). Use 2-3 band-pass bands at fixed ratios, e.g. 1 : 1.6 : 2.5, moving together. One band
  is a noise wash.
- Air whoosh: band-pass Q 1-3. The centre sweeps from 300-600 Hz up to 1.5-5 kHz at the peak, then falls back to
  400-800 Hz. The sweep follows the speed curve: centre ∝ speed^0.5-1.
- Heavy whoosh: centre 150 Hz - 2 kHz, longer, plus a sine at 40-80 Hz swelling with the peak, about 12 dB under the
  body. For a cymbal-like peak, reverse a short noise-burst reverb tail into the peak (src 2).
- Swish (cane, whip, blade, thin fast object): 80-250 ms, brighter (2-10 kHz), steeper rise. Add a narrow aeolian
  whistle at f = 0.2·U/d (Strouhal; e.g. 30 m/s past a 1 cm rod ≈ 600 Hz), as band-passed noise at Q 15-40.
- UI swipe (digital): 100-250 ms. High-passed noise (HP 300-800 Hz) plus an optional tiny tonal chirp (a sine gliding
  up an octave across the swipe, ~10 dB under the noise). No low end.
- Doppler: the frequency ratio across a pass is (c+v)/(c−v). At 20 m/s that is ≈ 2 semitones in total; at 50 m/s,
  ≈ 5 semitones (c = 343 m/s, src 39). Apply it to the tonal layer and the filter centre. Designed whooshes often
  exaggerate it to 3-12 semitones on the tonal layer.
- Pan: equal-power L→R following tanh((t−t_peak)/w), with narrow-wide-narrow width across the pass (src 4). Keep the
  mono sum intact.
- Tail: a short reverb for size (src 1), or a pitched delay for a sci-fi feel (src 1). If an impact follows, leave
  ~80 ms of silence before it (src 4).

**Processing.** Carve each layer's band, compress lightly to tie the layers together (src 2), give the tail a short
room or plate, and add a little distortion only for aggressive whooshes (src 1).

**In the mix.** The loudest point lands on the cut or the fastest frame (src 1, 4). It starts 0.2-0.6 s before
(earshot). It sits under the picture; it does not shout (src 1). Reserve the longest, widest whoosh for the biggest
move (src 4).

**Cheap-sounding mistakes.** Static noise with a volume fade and no filter or pitch movement. Harsh unfiltered white
noise. Too slow (it drags) or too fast (it reads as a click). Mono with no movement. One layer (src 1). The same
envelope on every edit. A tail running into the next event. A stereo sweep on every sideways move. Making it louder to
make it "better". A whoosh on a scene change where nothing actually moves (src 4).

---

## 2. Riser (uplifter, build)

**Anatomy.** A noise riser (the filter opens), a tonal riser (the pitch rises), an optional texture or granular layer,
a rhythmic element (accelerating LFO, tremolo or roll), and a growing reverb (src 6, 9). Use 3-4 elements at most
(src 7). **Let one motion lead**: filter for noise sources, pitch for tonal ones; the others support it (src 7).

**Synthesis**

- Noise: low-pass or band-pass at 24 dB/oct, cutoff 150 Hz → 10 kHz over the length, resonance 15-20 % (src 6;
  Q ≈ 1.5-3 [sv]). The noise layer peaks slightly earlier than the tonal layer (src 6).
- Tonal: saw or triangle, 5-7 unison voices detuned 10-15 cents (src 6). Pitch rises +12 to +24 semitones, narrowing
  to 12 near the peak (src 6). Stack layers at 0 / +12 / −12 st that fade in at different times (src 6). Detune can
  widen as it rises (src 8).
- Accelerating modulation: LFO rate 1/2 note → 1/16 note (at 120 BPM, 1 Hz → 8 Hz), depth 10 % → 70 % (src 6). LFO
  speed and intensity rise together (src 9).
- A frequency shifter from 0 to +200 Hz (src 6) gives an inharmonic, sci-fi lift.
- **Curves are exponential, not linear**: most of the change lands in the last beat or two (src 7). For example
  p(x) = p0 + (p1 − p0)·x³, with x = t/T, or exponential in log-frequency [sv].
- Level: −20 dB → 0 dB across the riser (src 6), on the same accelerating curve.
- Endless rise (Shepard-Risset): octave-spaced sine (or saw) partials under a raised-cosine loudness envelope over
  log-frequency, all gliding up together (src 35). Each layer must cover a full octave, or the reset is audible
  (src 6). Fade the top and bottom layers (src 7).
- "Convergence" reveal riser (THX Deep Note): 30 voices start at random pitches between 200 and 400 Hz and glide to a
  chord spanning three octaves (src 34). That is a riser that resolves into the logo chord.
- Width grows into the peak (src 7). Reverb grows too: size 30 % → 70 %; 2-3 s decay at 20-30 % wet for short risers,
  up to 6 s at 40 % for long ones (src 6).
- Reverse swell: reverse a reverb tail so it rises into the hit (src 40). `fx.reverse_swell` does this.

**Length.** In a launch film, 1-4 s (earshot). In music, 4 bars for short transitions and 8-16 bars before a drop
(src 6).

**Ending.** The peak lands exactly on the downbeat or the reveal frame; a climax that arrives late steps on the new
section (src 7). Then cut it abruptly, duck it 3-6 dB just before the drop (src 6), or leave silence for 0.1-0.5 s
(earshot) or one beat (src 7), then the impact. A riser ends in a hit or a short silence (src 9).

**Cheap-sounding mistakes.** Energy piling up below 100 Hz (high-pass it, ~150 Hz). A climax that lands after the
downbeat. No release, so it feels relentless. Too many layers. The same riser on every transition (src 7). Too much
white noise or saturation (src 6). Linear curves (src 7).

---

## 3. Downlifter (downsweep, fall)

The mirror of a riser, placed on beat one of the new section or right after an impact. It is a push-and-release
after the build (src 7). Length 0.5-4 s [sv].

**Synthesis**

- Noise: the filter starts fully open and closes, low-pass from 8-12 kHz down to 200-400 Hz (src 49, forum answer via
  search summary; values [sv]). Band-pass with extra bandwidth and resonance also works (src 8).
- Tonal: pitch falls 12-24 st [sv]. For the pitch-down impact, a sine-square wave with an envelope on pitch *and* on
  the low-pass cutoff keeps the harmonics audible as the fundamental drops below hearing (src 8).
- **Decelerating curves**: fast at first, slowing at the end, i.e. exponential decay with a time constant ≈ length/3
  [sv].
- Level decays while the reverb gets wetter: the dry part dies first [sv]. An LFO whose rate is shaped by a reversed
  envelope, plus phase modulation and flanger, gives the classic synth downlifter (src 50).
- Tape-stop variant: playback rate glides to 0 over 0.3-1 s (`fx.tape_stop`) [sv]. How steep the fall is sets whether
  it feels like a lazy sag or a hard yank (src 51, search summary).

**Mistakes.** The same as risers, plus a downlifter that masks the drop it follows: keep it 6-10 dB under the impact
[sv].

---

## 4. Impact / hit (and braam)

**Anatomy** (src 9, 10, 11)

| layer | job | sources and numbers |
|---|---|---|
| transient | the timing, the "crack" the ear locks onto | short bright burst, mostly above 1 kHz: snare, clap, stick or a noise burst; shorten its tail (src 10) |
| body | weight and character | kick, tom, taiko, metal resonance, gong (src 9, 10) |
| sub | punch you feel | sine at 30-60 Hz with a pitch drop over 200-400 ms (src 10) |
| tail | size and space | long dark reverb on the body; sub stays dry (src 10); the tail is typically a long reverb that fades (src 9) |
| sheen / atmos | top and air | processed cymbal, metal, an atmospheric sound (src 9) |

An example stack: distorted kick + high percussive element + processed cymbal + sub + atmosphere (src 9). Before the
impact: pull back to near silence, because dynamic contrast makes the slam (src 11). Or use a reverse swell into it
(src 40).

**Synthesis** [sv unless marked]

- Transient: a noise burst of 2-10 ms with an envelope raised to the 2nd-4th power (Farnell's bell striker: 10 ms of
  noise, env⁴, src 15/06), high-passed or band-passed above 1-2 kHz, plus a 1-3 ms click. 0 to −6 dB against the body
  peak.
- Body (drum-like): a sine or triangle starting at 100-180 Hz and falling to 45-70 Hz with a pitch time constant of
  30-80 ms. Add inharmonic membrane overtones at ×1.59, 2.14, 2.30 that die faster than the fundamental (earshot
  `impact.py` notes). Metal body: a cloud of inharmonic modes from 300 Hz to 4 kHz with decays falling as frequency
  rises.
- Sub: a sine starting 12-24 st above its end note and falling exponentially over 200-400 ms (src 10). Its final note
  is tuned to the track's key (src 10). For trailer and logo hits, let it ring for 800 ms or more; game and UI hits
  get a few hundred ms (src 10).
- Big explosive boom (Farnell): a "fireball" of noise band-passed at 70 Hz then low-passed at 70 Hz twice, attack
  100 ms, decay 8 s. A "blast" of high-passed noise through a resonant low-pass near 7 kHz, swept in 30 ms, driven ×50
  into a hard clip (src 15/31).
- Size trick: render a bright layer (noise, metal) and play it back 2-3 octaves down (¼-⅛ speed) for sub-heavy
  material. This beats pitching down an already sub-heavy sound (src 9).
- Tail: a 2-6 s reverb with frequency-dependent decay (lows ring longest), darkened (LP 3-5 kHz), pre-delay 10-30 ms.
  A flanged tail (a comb whose delay drifts) gives the descending partial fan heard in trailer hits (earshot
  `impact.py`).

**Processing.** Distortion on body and transient, then EQ and compression (src 9, 10). Bus compression or a little
OTT for trailer and electronic styles, but over-compression flattens the transient (src 10). High-pass below about
25 Hz (src 10). Pan the layers (src 9), but keep transient and sub centred [sv]. Limit the finished sum (src 9).

**Braam.** A long, loud, low brass note at C1-C2 (≈ 33-65 Hz), with layers an octave apart (C0, C2 under C1). Vary
the intonation or playing technique per layer. Automate distortion on one or more layers, bend the pitch down
gradually, and put the group through a reverb (src 9). For synthesis [sv]: 3-6 detuned saws per octave layer through
a low-pass whose cutoff opens from 200 Hz to 1.5-3 kHz over 50-150 ms (brass "blat"), then settles to ~800 Hz.
Waveshaper drive rising over the note, a −1 to −3 st bend over 2-4 s, and a sine sub under it.

**In the mix.** On the frame. Rare: the riser → stop-down → impact move happens once or twice per film (earshot). The
biggest sounds are saved for the biggest moments (src 42).

**Cheap-sounding mistakes.** Two body layers fighting around 200 Hz. An untuned sub clashing with the bass. An
over-compressed bus that kills the transient. Sub-25 Hz energy that eats headroom. Trying to fix a flat hit with
another plugin instead of another layer (src 10). Reverb on the sub (mud) [sv]. Generic stock stingers (src 11).

---

## 5. Sub drop (and 808 boom)

**Anatomy.** A sine gliding down into the sub-bass region. It is usually a pure sine because it has no harmonics.
Designers add subtle distortion or harmonics so it survives phones and laptops. The glide lasts from a fraction of a
second to a few seconds: a quick punchy drop reinforces an impact, a long descent signals dread. It usually sits under
an impact or downbeat as its low foundation, with the surrounding elements high-passed to make room (src 12).

**Synthesis** [sv unless marked]

- Pitch: start 1-2 octaves above the end note; end at 30-45 Hz, tuned to the key. Glide exponentially in frequency
  (constant semitones per second, or faster at first) over 0.4-3 s.
- Amplitude: a 2-5 ms attack ramp, starting at a zero crossing so it does not click. Hold ~30 % of the length, then an
  exponential decay. End on a fade, never a cut.
- Small speakers: phones cannot reproduce much below ~300 Hz, yet the ear infers the missing fundamental from its
  harmonics (src 37). So add a parallel saturated copy (tanh, drive 6-12 dB) low-passed at 200-400 Hz under the
  clean sine (a clean sub plus a distorted duplicate, src 48), or synthesise the 2nd and 3rd harmonics directly at
  −12 to −20 dB.
- 808 tutorials put the base sine at ~40-60 Hz, the pitch starting 1-2 st high and falling within 10-30 ms for a
  click-like attack (a bigger, longer pitch fall gives a more obvious sweep), and the amplitude at ~5 ms attack,
  ~700 ms decay, no sustain, 400-500 ms release (src 14, search summary).
- 808-style boom: a sine at 32' (around C1-C2), fastest attack, decay fully open, sustain down. The pitch envelope
  decides kick vs boom: shorter decay gives a kick, longer gives a boom. Then saturation (harmonics), mono,
  compression and a low shelf cut. Avoid a sub-oscillator, because its extra harmonics muddy the sub (src 13).
- Mono, always. One sub at a time in the whole mix (earshot).

**Mistakes.** A pure sine that vanishes on phones (src 12, 37). Clicks at the start or end. Energy below 25 Hz
(src 10). Reverb on the sub. A sub that is not in key (src 10).

---

## 6. UI click / tap / toggle / hover

**Durations.** Taps and clicks last 30-150 ms; most interface cues are under 300 ms; notifications go up to ~500 ms
(src 16). A reference table puts clicks at 10-80 ms (src 46c, search summary). Game UI guidance: a click under ~100 ms,
a confirm 100-250 ms (src 22, search summary). Trim cues shorter than feels natural in isolation, because they play
often (src 16).

**Semantics** (shared expectations; inverting them feels wrong)

- Neutral (tap, toggle, selection): a single tone with no pitch movement (src 16).
- Positive (confirm, success, forward): rising pitch, a major third or perfect fifth up (src 16). Upward means yes or
  forward (src 22, search summary).
- Negative (error, cancel, back): lower, falling, slightly detuned or mildly dissonant (src 16). A flat, dull tone
  means "denied" (src 22, search summary).
- Toggle on vs off: on rises or sits higher, off falls or sits lower [sv, follows the rules above]. A physical toggle
  is two contacts 30-150 ms apart (src 15/19).
- Hover: near silence or nothing (src 22, search summary). If used: ≤ 30-50 ms, 12-20 dB under the click, nothing
  above ~4 kHz [sv].
- Frequent cues are higher, brighter and quieter; rare cues are lower and fuller (src 16, 18). Most interactions
  should make no sound at all. Silence is the negative space that lets important sounds stand out (src 18).

**Synthesis recipes**

1. *Mechanical / plastic click* (Farnell, src 15/19): white noise → band-pass 5 kHz, Q 12. Envelope: a 1 ms ramp up,
   then a 20 ms ramp down starting at 2 ms, **squared** before it multiplies the noise. A reusable click abstraction
   takes (attack ms, decay ms, centre Hz) with Q 8.
   - Push-button composite: clicks at 0 ms (1/20 ms, 4 kHz), 30 ms (1/50 ms, 3 kHz), 140 ms (4 kHz) and 150 ms
     (7 kHz). All are summed into a 50 ms delay line whose output goes through ×0.1 gain and a band-pass at 700 Hz
     Q 3, fed back for a small resonant body.
   - Slide switch: pink-noise "clunks" at 0 and 200 ms, plus a short ping of two sines at 10.5 and 9.45 kHz with a
     50 ms decay.
2. *FM tick* (SuperCollider example, src 23): carrier 1-10 kHz, modulator 100-1000 Hz, index 10-1000, attack 1-10 ms,
   release 1-10 ms, high-pass at 1 kHz. A fast percussive envelope on both pitch and filter or amp gives bells, pops,
   clicks, boinks and bleeps (src 24b, forum via search summary).
3. *Tonal blip* (the "digital" UI style): a sine or triangle for a smooth, clean tone (src 16) at 600 Hz-2 kHz. Attack
   0.5-2 ms, exponential decay 20-60 ms; optionally a 1-3 st pitch drop over 10-20 ms for a "plip". Layer a short
   tactile texture under it (src 16): a 1-3 ms noise click high-passed at 3 kHz, 6-10 dB down [sv].
4. *Physical button* (earshot `click.py`): one dominant, very damped shell mode at 2-7 kHz (gone in 0.2-2 ms) over
   weaker modes. Few mid housing modes, gone in a few ms; a long-ringing ~1 kHz housing reads as a wood block. The
   release comes 25-35 ms later at −25 to −30 dB, or not at all.

**Processing** (src 16). High-pass ~120 Hz. Tame harsh peaks at 2-5 kHz. At most a tiny, short room or plate. Limit
and normalise so every cue sits at one *perceived* loudness. Mono by default. Trim to the first transient; leading
silence makes the interface feel laggy.

**Palette.** Build the family from one patch and change only pitch and envelope, with a shared reverb and character
(src 16). Or make a confirm sound you like and derive the rest by pitching, shortening and filtering it (src 22,
search summary). Earcons: start from timbre, register and rhythm; musical timbres with several harmonics beat pure
tones (src 17). Rounder, softer, "less digital" sounds alert without stressing (Windows 11's redesign, src 21).

**In the mix.** One click per on-screen click, on the frame the cursor presses. Short, bright (2-6 kHz) and ~10-15 dB
under the music peaks, panned toward where it happens on screen (earshot).

**Cheap-sounding mistakes.** Too long. Too much low end, which turns to mud on phones and laptops. Inconsistent
loudness across the family. Auditioning only on good monitors. Leading silence (src 16). Melodic tails or personality
on cues heard hundreds of times (src 46, search summary). Hover sounds on everything.

---

## 7. Pop / bubble

**Physics.** A gas pocket rings at the Minnaert frequency, f ≈ 3.26 / r (Hz, r in metres): 0.5-3.5 kHz for
millimetre bubbles. Damping gives Q ≈ 5-40. The pitch rises as the pocket shrinks and its neck opens: +10-120 % within
3-80 ms. A film rupture tick comes first, with breathy noise that follows the pitch and a small low "puff" (earshot
`pop.py`, src 45).

**Farnell's bubble** (src 15/12): a cosine oscillator whose frequency follows an exponentially shaped ramp (`expcurve~
100`, ~100 ms; a variant uses 200 ms) times a base frequency (e.g. 600 Hz; 240-270 Hz in the bubble-stream example).
The amplitude envelope is `adenv~ 10 80` (≈ 10 ms attack, 80 ms decay), started 3 ms after the trigger. Then ×0.1 and
a 40 Hz high-pass. Streams come from a 15 ms clock, prime-numbered steps and a 50 % probability gate.

**The motion-design "pop-in"** (an element appearing: an icon, a bubble, a chat message) [sv]

- A tonal blip with an upward sweep: a sine from 300-500 Hz up to 800-1500 Hz in 15-40 ms. Attack 0.5-1 ms (not 0:
  a perfect step onset reads as a digital tick, earshot `pop.py`). Exponential decay 40-120 ms.
- A 1-2 ms band-passed noise tick (2-4 kHz) on the onset, 6-12 dB down.
- An optional second, smaller, higher ring 5-15 ms later (a daughter bubble, earshot `pop.py`) for a wetter, cuter pop.
- For several items popping in, step the base pitch up a scale tuned to the music's key (pentatonic or major).
- Edit like the kit: 0.2-0.4 s total, 40 dB down 35-145 ms after the peak, little below 150 Hz or above 4 kHz
  (earshot).

**Mistakes.** Long tails, a full-band tick and hiss, a big low puff, the same pitch for every element.

---

## 8. Notification / ding / success chime

**Rules**

- Success: two quick rising notes, a major third or fifth up (src 16). A notification is a short 2-3 note motif, up to
  ~500 ms (src 16).
- Error: lower, falling, slightly detuned (src 16). Success and error must be distinguishable without looking (src 16).
- Hierarchy (Material Design): brand sounds first, then hero sounds (rare celebrations), then alerts and
  notifications, then primary and secondary UX sounds. Notifications fire more often than hero sounds, so they are
  shorter and built to survive repetition (src 20, search summary).
- Earcon guidelines (src 17): pitch no higher than 5 kHz and no lower than 125-150 Hz. Notes no shorter than 0.0825 s,
  though 1-2-note earcons can use notes as short as 0.03 s. Up to 6 notes per second. A 0.1 s gap between serial
  earcons. Loudness 10-20 dB above the background. Register differences of 2-3 octaves if they must be told apart
  absolutely.
- Brand sound (Google "Super G"): two piano notes an octave apart, played slightly apart (flammed) for playfulness,
  done in under 1.5 s. Its key (G) anchors the whole palette (src 18).
- Calmer, rounder, softer alerts still notify without being overwhelming (src 21).

**Synthesis**

1. *FM bell / chime*: a non-integer carrier-to-modulator ratio gives inharmonic, metallic partials. Violet suggests
   trying 1.41 (src 25, search summary); Chowning's classic bell is about 1:1.4 [not verified from a fetched source].
   The index envelope decays fast, so the attack is bright and quickly mellows, like a real bell (src 24, 25). The
   amplitude decays as a long exponential (src 24). Start values [sv]: carrier = the note (e.g. C6 1047 Hz, E6, G6).
   Ratio 1.4 (bell), 3.5 (glassy) or 1 / 2 with ~0.2 % detune (e-piano or soft chime). Index from 3-8 down to ~0.5
   with τ = 30-80 ms; amplitude τ = 150-600 ms. A slightly mistuned integer ratio gives an electric piano (src 24).
2. *Modal bell* (Farnell's telephone bell, src 15/06). Partial ratios to the fundamental, by group, with relative
   strengths and decay multipliers (decay = 2000 ms × multiplier):

   | group | ratios | strengths | decay × |
   |---|---|---|---|
   | primary | 2.002, 3, 9.6 | .008, .02, .004 | 0.9 |
   | secondary | 2.49, 2.571, 11 | .02, .04, .02 | 0.25 |
   | sub / fundamental | 0.501, 1, 0.7 | .002, .001, (.02) | 1.2 |
   | circular modes | 3.05, 6.242, 12.49 | .005, .05, .05 | 0.14 |
   | striker group | 13, 16, 24 | .02, .03, .04 | 0.07 |

   Striker: noise with a 10 ms decay, env⁴. Two slightly detuned gongs (650 and 653 Hz) are struck alternately every
   60 ms (the phone's ring); for a single ding, use one. The casing is two short
   feedback delays (0.77 / 0.88 ms, gain 0.7) and band-passes at 287, 431 and 1243 Hz, Q 12. Higher partials die
   faster; that is what makes it read as struck metal.
3. *Glass / marimba "ding"* [sv]: a free bar's modes sit at 1 : 2.756 : 5.404 : 8.933 (free-free beam), with each
   partial's decay falling roughly as f^-0.9 (earshot `pluck.py` notes the same law for tines). Add a soft 2-5 ms
   mallet (low-passed noise pulse).

**Processing.** High-pass ~120 Hz, a tiny plate or room, a little width, limit (src 16). Tune the notes to the
music's key.

**Mistakes.** Alarm-like beeps (earshot); for contrast, Farnell's alarms are plain sines at 600-1012 Hz switched
roughly 2-4 times a second (src 15/04). Square waves. Tails over ~0.6 s on a frequent cue. Success and error too
similar. Pitches outside 125 Hz-5 kHz (src 17).

---

## 9. Shimmer / sparkle

This is decoration: use it only where something magical happens (an AI generation reveal, a "new" badge), short and
low in level. Pro categories: a wide, glassy wash of highs; small bell-like notes scattered at random ("twinkle");
fine granular sparkle that drifts rather than strikes; a single bright glitter burst with a fast tail (src 47, vendor
categories via search summary).

**Synthesis** [sv unless marked]

- Ping cloud: 15-60 short pings per second, as sine or FM-bell grains. Pitches come from a pentatonic or major scale in
  the music's key, between 2 and 8 kHz (C7 2093 Hz to C9 8372 Hz). Each ping: 0.5-1 ms attack, 30-200 ms exponential
  decay, random pan, level +-6 dB at random. Density follows an envelope that swells with the motion and thins out at
  the end. Random pitches that are not on a scale clash with the music.
- Wash: noise high-passed at 6-8 kHz through a slowly modulated comb or flanger, ~18 dB under the pings.
- Shimmer reverb: the reverb's wet signal is pitch-shifted an octave up and fed back into the reverb input, so each
  pass climbs again. The shifted signal is not mixed straight to the output; the send level sets the strength. Add a
  low-pass in the loop, because shifting up adds highs (src 27, forum via search summary). Some designs add filtered
  delay with heavy feedback (src 27).
- Farnell's "transporter" (src 15/33) is a sci-fi shimmer. Twelve band-passes (Q 1) at 620-1200 Hz with swept centres
  on a buzzy source. Noise through band-passes at 775, 1550 and 2325 Hz (Q 13), shaped by a 10 ms rise and 5 s fall.
  Three modulated taps on a 120 ms delay line with 0.7 feedback, then reverb.

**Processing.** High-pass at 1-2 kHz, a de-esser at 6-10 kHz, a long airy reverb, width.

**Mistakes.** Fatiguing 8-12 kHz energy. Off-key pitches. Constant density with no envelope. Too long or too loud.

---

## 10. Glitch / stutter / digital

**Techniques with numbers**

- Bitcrush (src 28): 8-12 bits for mild grit, 4-6 bits heavy (sweet spot 6-10), 1-bit at the extreme. Sample-rate
  reduction to ~8 kHz for warmth or ~2 kHz for harsh aliasing. Crush after an EQ that isolates 500 Hz-2 kHz, so the
  low end stays intact. A heavily crushed copy 20-30 % under the clean signal (src 28). Low-pass or a gentle high shelf
  after crushing (src 29).
- Buffer repeats (src 28): slices of 5-50 ms (start at 10-30 ms), repeated 4-8 times. Tighten the spacing for ramps or
  keep it even for stutters; tempo-wise 1/32 to 1/64 notes. Step the repeat rate from 1/16 toward 1/64 for an
  accelerating fill (src 29).
- Granular (src 28): grains of 1-100 ms. 5-20 ms gives distinct artifacts; start at 20-30 ms. Smaller grains sound more
  digital and stuttery, larger ones smoother.
- Time-stretch artifacts: 400-800 % stretches, using the first 100-200 ms of the warped audio (src 28). Reverse slices,
  random pitch jumps, odd-length cuts (src 30).
- Sources (src 29): sharp transients with clear pitch survive chopping; pads and noise smear into atmospheres.
- Keep a steady sub or bass under the glitch so the low end does not thin out (src 29).
- Computed digital sources: R2D2-style chirps are random-stepped tones with random glides. Farnell's version: event
  periods of 100-700 ms, a random 0-0.999 step value, FM with ratio 0-32 and index 0-2 (src 15/34). Data chirps [sv]:
  square or pulse bursts at 1-4 kHz, 5-30 ms long, a new pitch per burst. Electrical buzz [sv]: mains harmonics,
  gated.

**For motion design** [sv]: glitch hits are short (100-400 ms) and lock to the visual glitch. One buffer slice per
glitched video frame (33 ms at 30 fps, 42 ms at 24 fps) makes picture and sound stutter together. `fx.glitch` /
`fx.stutter` / `fx.crush` / `fx.granular` already implement the operations.

**Mistakes.** Overuse. Chops drifting off the grid. Stacked harshness from crushing plus aliasing. A thin low end.
Judging the glitch only in solo (src 29).

---

## 11. Typing

**Anatomy of a keystroke.** The press peak spans ~10 ms: a "touch" peak (finger on key) and a "hit" peak (key on the
plate) at its two ends. The release peak comes ~100 ms later and is much weaker; laptop keyboards may lack it (src 31,
via search summary of Zhuang et al. and Asonov & Agrawal). earshot's measured mechanical keys (`keypress.py`): a
broadband bottom-out click; keycap and plate modes near 1.25-1.6 kHz plus a cluster up to ~2.3x; a case body at
110-260 Hz; a 58-77 Hz "thock" that peaks 20-35 ms after the hit; a release 70-125 ms later (two small ticks, then the
top-out); and an 8-10 kHz spring ping.

**Timing.** One key sound per typed character, never the same sample twice in a row, slightly irregular timing
(earshot). Human typing at 40-80 wpm is ≈ 3-7 characters per second (≈ 150-300 ms apart). On-screen "typed" prompts in
launch films often run faster (10-20 chars/s, 50-100 ms apart), so releases overlap the next press [sv]. Jitter
intervals by +-15-25 %; put a slightly longer gap before spaces and after punctuation [sv].

**Synthesis** [sv]

- A fixed modal set per key (the same key sounds the same twice) plus per-press variation: force +-2 dB, timing
  +-10-20 ms, mode frequencies +-1-2 %. `keypress.py` already does this with its `key` parameter.
- The space bar and Enter are larger keys: lower cap modes (×0.6-0.8), a stabiliser rattle (2-4 small ticks within
  5-15 ms), a louder release.
- Laptop and scissor keys (the usual UI-demo sound): softer, shorter, no thock, brighter (2-6 kHz), a quieter release,
  high-passed at 120-200 Hz. Mechanical "thock" is a deliberate style choice, not the default.
- Stereo: a slight pan by key position (left-hand keys left), within +-20 %.

**Mistakes.** A machine gun (identical samples or perfectly even timing). A big low thock under a light UI demo.
Characters and sounds out of sync.

---

## 12. Logo sting / reveal

**Anatomy.** A build (riser or reverse swell, 0.5-2 s) → a stop-down (0.1-0.5 s of near silence) → a hit (an impact
with a sub) on the logo frame → a tonal tail or "bloom" (2-4 s) that resolves the music (earshot). Real examples:

- Netflix "ta-dum": Lon Bender's wedding ring knocking on a cabinet, mixed with a hammer on an anvil and other
  sweeteners, then a synth-like "blossom" from an effects-laden guitar placed directly after the knock. The brief was
  tension, release and quirkiness; 20-30 versions were tried (src 32).
- Google "Super G": two piano notes an octave apart, flammed, in under 1.5 s (src 18).
- THX Deep Note: voices converging from random pitches to a wide chord (src 34).
- Sonic logos are usually 2-5 s, with 3 s the most common figure. Lenovo specifies a 3 s end-bumper version (src 33,
  vendor sources via search summary).

**Synthesis** [sv]

- Hit: the impact recipe with the sub shortened to 0.8-1.5 s.
- Tonal bloom: a chord (root, fifth, octave, optionally a major third or ninth) in the music's key, built from FM or
  modal bells plus a pad. It enters 0-30 ms after the hit with a 50-200 ms attack and a low-pass opening from ~1 kHz to
  6-8 kHz. Shimmer reverb on the bloom only.
- Optional: a reverse swell (the hit's own reverb, reversed) for 0.3-1 s into the hit (`fx.reverse_swell`), or a
  convergence riser ending on the bloom chord.

**Timing.** The hit lands on the logo's landing frame. Finish within ~0.5 s of the last frame. The strongest hit of
the last 4 s rings out 2.0-3.6 s in 5 of 7 pro films (earshot).

**Mistakes.** A long fade-out or dead air at the end (earshot). A riser that is too long. A tonal tail out of key.
Everything at once with no stop-down. A sub tail that outlasts the picture.

---

## 13. Levels and mixing in launch videos

Measured on 7 pro launch films (earshot, `docs/launch-video-sound.md`) unless marked:

- Music leads; SFX are accents. There are 1.2-1.7 salient hits per second in total (music included). There is **no**
  measurable hit on every cut. Designed accents go to the reveal, the logo, the key UI action and section changes.
- Master at about −14 LUFS integrated, true peak ≤ −1 dBTP.
- Stop-downs: 4 of 7 films drop ≥ 10 dB for 0.15-1.5 s and come back on a hit.
- UI foley sits ~10-15 dB below the music peaks, at 2-6 kHz, panned to its place on screen [practice]. Impacts are
  the loudest SFX. Whooshes sit under the picture (src 1).
- One element owns the sub at a time. High-pass everything else. Check the mono fold-down; phones are mono.
- Sync: viewers detect sound 45 ms *early* or 125 ms *late* (ITU-R BT.1359). EBU R37 allows +40 / −60 ms end to end
  (src 38). A frame is 33 ms at 30 fps, so align the perceived onset or peak to the exact frame. If in doubt, let the
  sound lead by up to 1 frame; never let it lag by 2 or more [sv]. Align samples on their attack, not their file
  start (earshot).
- Leave room: silence before a big moment is part of the sound (src 11, 18).

---

## 14. The cheap-SFX checklist (all categories)

Each item is phrased so a recipe or critic can measure it:

1. **One layer**, or layers whose peaks are not aligned (src 1, 2, 10).
2. **No movement**: a static spectrum and static pan on anything that moves (src 1).
3. **Linear curves** on risers and sweeps, where the pros use accelerating or exponential ones (src 7).
4. **Mud**: excess energy at 200-400 Hz (src 41), or two layers fighting in one band (src 10).
5. **Weak or blurry attack**: an over-compressed bus, or a slow onset where a crack belongs (src 10, 41).
6. **Too long**: UI cues over ~150 ms, notifications over ~500 ms (src 16), tails that run into the next event (src 4).
7. **Bad low end**: below 25 Hz (src 10); a pure-sine sub with no harmonics (src 12, 37); an untuned sub (src 10);
   reverb on the sub.
8. **Harshness**: unfiltered white noise (src 1), 2-5 kHz peaks on UI sounds (src 16), stacked crush and aliasing
   (src 29). A naive oscillator folds over (`polish.py` checks it).
9. **Artifacts**: clicks at edges or splices, DC, chopped tails, leading silence (src 16, `polish.py`).
10. **Inconsistency**: a family of sounds at different loudnesses or timbres (src 16). Pitches not in the music's key.
11. **Overuse**: the same whoosh or riser on every cut (src 4, 7), sound on every micro-motion (src 18), hover sounds.
12. **Wrong size or texture** for the object on screen (src 49b, search summary). Big means lower, slower and wider;
    small means higher, faster and tighter (src 42).
13. **Off sync** by a frame or more (src 42, 38).
14. **Mono-incompatible width** that vanishes on phones (src 4).
15. **Judged in solo only**. Audition against music and on small speakers (src 16, 29).

---

## 15. What this means for earshot forge recipes

1. **Add a design stage to every motion recipe.** Keep the physical model as a source layer. Add the category's other
   layers (sections 1-12) and the universal chain (section 0), built from `fx.py`. A recipe's parameters should include
   per-layer levels, saturation drive, bus glue amount, reverb send and pre-delay, and the edit length.
2. **Expose sync parameters.** Duration and peak time (whoosh, riser, downlifter, sub drop), the hit frame (impact,
   sting) and note or key (anything tonal or with a sub), so a film can request "a 0.48 s whoosh peaking at 0.30 s in
   D minor".
3. **Expose family parameters.** A `family` seed that fixes timbre (modal set, FM ratio, filter character) across a
   set, while per-render seeds vary only force, timing and noise. That gives a consistent UI palette and safe round
   robins.
4. **Reference sets should be designed library SFX** of the target style (motion-design packs: digital, tonal, UI,
   glitch, impact, notification and whoosh categories, src 44), not raw foley. A model trained on "real vs synthetic"
   rewards a mic noise floor and room tone, which a designed motion SFX does not have.
5. **Measurable targets from this page** (for the critic or the polish checks):
   - peak position as a fraction of the length (whoosh 0.5-0.7);
   - curve shape (riser change concentrated in the last 25 %);
   - band shares (UI: little below 120-150 Hz; pops: little below 150 Hz or above 4 kHz; impact: sub, body and
     transient bands all present);
   - attack time (clicks < 2 ms to peak);
   - durations per category (sections 1-12);
   - tonal pitch inside 125 Hz-5 kHz for earcons;
   - loudness spread across a family;
   - mono-fold loss.

---

## References

Tutorials and articles (read with WebFetch unless marked "search summary", which means only a search engine's
summary of the page was available, so treat those numbers as softer):

1. Violet Recording, *How to Make Whoosh Sounds*. https://violetrecording.com/how-to-make-whoosh-sounds/
2. Krotos, *Whoosh Sound Effects - Tutorial*. https://sound.krotosaudio.com/?p=1555
3. Krotos Studio, *4 whoosh sound effects tips*. https://webflow.krotos.studio/guide/4-whoosh-sound-effects-tips
   ("bigger = lower and slower" is from a Krotos article seen only in a search summary:
   https://www.krotosaudio.com/?p=1006981)
4. Sonilo, *Whoosh sound effect guide* (motion and editing). https://sonilo.com/ai-music/whoosh-sound-effect-guide
5. Designing Sound, *Whoosh review* (Tonsturm Whoosh: source samplers, Doppler engine, impact layer, sidechain).
   https://designingsound.org/2014/02/whoosh-review
6. Unison Audio, *How to create risers*. https://unison.audio/how-to-create-risers/
7. Violet Recording, *How to Make Risers and Sweeps*. https://violetrecording.com/how-to-make-risers-and-sweeps/
8. Soundbridge, *Transition effects* (noise sweep, pitch up, pitch-down impact in Massive).
   https://www.soundbridge.io/transition-effects
9. Ableton, *Learn How to Make High-Impact Sounds for Movies and Trailers*.
   https://www.ableton.com/en/blog/learn-how-to-make-high-impact-sounds-for-movies-and-trailers/
10. Violet Recording, *How to Make Impact and Hit Sounds*. https://violetrecording.com/how-to-make-impact-sounds/
11. IRPR Sound Design, *Weak trailer impact*. https://sounddesign.irpr.agency/solutions/weak-trailer-impact/
12. IRPR Sound Design, *Sub drop* (glossary). https://sounddesign.irpr.agency/glossary/sub-drop/
13. Attack Magazine, *Synth Secrets: 808-style booms*.
    https://www.attackmagazine.com/technique/synth-secrets/808-style-booms/
14. 808 / sub tutorials (search summary across several pages; numbers are one author's each): Ableton forum,
    https://forum.ableton.com/viewtopic.php?p=873995; Unison, https://unison.audio/808-sound-design/; ADSR Sounds,
    https://www.adsrsounds.com/ni-massive-tutorials/making-an-ni-massive-808-kick-drum/
15. Andy Farnell, *Designing Sound* (MIT Press), Pure Data practicals (patch values read from the code listings):
    index https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/8375/designing_sound.zip/index.html
    - 04 Alarm and menu sounds: .../practical04.html
    - 06 Telephone bell: .../practical06.html
    - 12 Bubbles: .../practical12.html
    - 18 Wind: .../practical18.html
    - 19 Switches: .../practical19.html
    - 31 Explosions: .../practical31.html
    - 33 Transporter: .../practical33.html
    - 34 R2D2: .../practical34.html
16. Violet Recording, *How to Design UI and App Sounds*. https://violetrecording.com/how-to-design-ui-sounds/
17. Brewster, Wright & Edwards, *Guidelines for the creation of earcons* (University of Glasgow).
    https://www.dcs.gla.ac.uk/~stephen/earcon_guidelines.shtml
18. Google Design, *Good Vibrations* (sound and haptics, Super G, silence as negative space).
    https://design.google/library/ux-sound-haptic-material-design
19. Google Design, *Conor O'Sullivan on Pixel sound design*.
    https://design.google/library/google-pixel-sound-design-ux-conor-osullivan
20. Material Design 2, *Applying sound to UI* / *About sound* (sound hierarchy; the pages need JavaScript, so search
    summary). https://m2.material.io/design/sound/applying-sound-to-ui.html,
    https://m2.material.io/design/sound/about-sound.html
21. Windows Central, *Microsoft rounded sounds, not just corners, when designing Windows 11* (search summary).
    https://www.windowscentral.com/microsoft-rounded-sounds-not-just-corners-when-designing-windows-11
22. Game UI sound guidance (search summaries): GameDev.net, *What was that noise? Mastering creative UI sound design
    with intention*,
    https://gamedev.net/news/5816-what-was-that-noise-mastering-creative-ui-sound-design-with-intention/; Audiokinetic,
    *Approaching UI Audio from a UI Design Perspective* (parts 1-2; the page returned 403),
    https://www.audiokinetic.com/en/blog/approaching-ui-audio-ui-design-perspective-1
23. sccode.org, FM click / UI SynthDef (SuperCollider). https://sccode.org/1-5bj
24. University of Victoria, MU307 *Tutorial: Synthesis - Frequency modulation*.
    https://web.uvic.ca/~aschloss/course_mat/MU307/MU307%20Labs/Tutorial%20FM.htm
    24b. VI-Control forum, *Which tools do you use to create UI sound effects for games* (search summary).
    https://vi-control.net/community/threads/which-tools-do-you-use-to-create-ui-sound-effects-for-games.152737/
25. Violet Recording, *How to Use FM Synthesis for Sound Design* (search summary).
    https://violetrecording.com/how-to-use-fm-synthesis-for-sound-design/
26. Sound On Sound, *Synth Secrets: Synthesizing Bells* (bell partials 2:3:4 strike, FM cymbal from three pulse
    pairs; most numbers are only in its figures). https://www.soundonsound.com/techniques/synthesizing-bells
27. Shimmer reverb (search summaries): Sequential forum,
    https://forum.sequential.com/index.php/topic,2071.msg23030.html; Reason Studios, *Building shimmer reverb*,
    https://reasonstudios.com/news/post/building-shimmer-reverb
28. SampleFocus, *5 Essential Techniques for Creating Glitch Sound Effects*.
    https://samplefocus.com/blog/5-essential-techniques-for-creating-glitch-sound-effects/
29. Violet Recording, *How to Make Glitch Sounds*. https://violetrecording.com/how-to-make-glitch-sounds/
30. Native Instruments, *Glitch effects*. https://blog.native-instruments.com/glitch-effects/
31. Keyboard acoustic emanations (search summaries of Zhuang, Zhou & Tygar, *Keyboard Acoustic Emanations Revisited*,
    and Asonov & Agrawal, *Keyboard Acoustic Emanations*): survey https://arxiv.org/pdf/2309.11012; paper copies
    https://www.cs.toronto.edu/~scottl/research/keyboard_emanations.pdf,
    https://davidsalomon.name/CompSec/auxiliary/KybdEmanation.pdf (the PDFs could not be read as text)
32. Netflix "ta-dum" (search summaries): Twenty Thousand Hertz, https://www.20k.org/episodes/netflix; Boing Boing,
    https://boingboing.net/2020/08/07/ta-dum-how-the-netflix-sound.html/amp
33. Sonic logo length (vendor sources, search summary): https://www.heardis.com/services/sound-logo/,
    https://origin-brandworld.lenovo.com/sonic-identity/, https://www.artyfile.com/blog/audio-branding-audio-logos-guide
34. Wikipedia, *Deep Note*. https://en.wikipedia.org/wiki/Deep_Note
35. Wikipedia, *Shepard tone*. https://en.wikipedia.org/wiki/Shepard_tone
36. Wikipedia, *Precedence effect*. https://en.wikipedia.org/wiki/Precedence_effect
37. Wikipedia, *Missing fundamental*. https://en.wikipedia.org/wiki/Missing_fundamental
38. Wikipedia, *Audio-to-video synchronization* (ITU-R BT.1359, EBU R37, ATSC IS-191).
    https://en.wikipedia.org/wiki/Audio-to-video_synchronization
39. Wikipedia, *Doppler effect* (standard formula; not fetched). https://en.wikipedia.org/wiki/Doppler_effect
40. Amped Studio, *Reverse reverb* (search summary). https://ampedstudio.com/blog/reverse-reverb/
41. Pitch Innovations, *Why Some Sounds Feel Expensive (Even When They're Not)*.
    https://pitchinnovations.com/blog/why-some-sounds-feel-expensive-even-when-theyre-not/
42. Violet Recording, *Sound Design for Animation*. https://violetrecording.com/sound-design-for-animation/
43. IRPR Sound Design, *Motion graphics*. https://sounddesign.irpr.agency/applications/motion-graphics/
44. AEJuice, *Motion Design Sound Effects* (category structure of a motion-design SFX pack: digital, tonal, UI elements,
    UI notification, glitch, impact, whoosh, data, paper; stems plus pre-mixed designs).
    https://aejuice.com/product/motion-design-sound-effects/
45. Bubble acoustics background: van den Doel, *Physically-based models for liquid sounds* (record),
    https://repository.gatech.edu/entities/publication/dcc121b6-5fdb-42db-b758-971522f01c70; Cornell bubble sound
    project, https://research.cs.cornell.edu/Sound/bubbles/ (the Minnaert relation itself is from earshot `pop.py`)
46. UI cue durations and fatigue (search summaries; the "a tap is heard thousands of times, so a tail, melody or
    personality becomes irritating" point is from one of these): Morphic, *Interface sound effects*,
    https://morphic.com/resources/sounds/interface-sound-effects; IRPR, *UI sounds annoying*,
    https://sounddesign.irpr.agency/solutions/ui-sounds-annoying/
    46c. UI sound design agent skill (category table: clicks 10-80 ms; search summary).
    https://agentskills.so/skills/dannyjpwilliams-ui-sound-design-skill-ui-sound-design
47. Morphic, *Magic sound effects* (category descriptions only, search summary).
    https://morphic.com/resources/sounds/magic-sound-effects
48. Violet Recording, *How to Design an 808 From Scratch*. https://violetrecording.com/how-to-design-an-808/
49. Downlifter forum answer (search summary): Gearspace, https://gearspace.com/threads/total-newbie-needs-some-basic-help.891066/
    49b. IRPR on SFX that feel too small, too synthetic or the wrong texture for the action (search summary):
    https://sounddesign.irpr.agency/solutions/sfx-dont-match-picture/
50. ADSR Sounds, *FX with Massive part 2: downlifter* (search summary).
    https://www.adsrsounds.com/ni-massive-tutorials/fx-with-massive-part-2-downlifter/
51. Morphic, *Tape stop sound effects* (search summary). https://morphic.com/resources/sounds/tape-stop-sound-effects

earshot internal: `docs/launch-video-sound.md` (measurements of 7 pro launch films), and the measured notes in
`earshot/forge/recipes/{click,keypress,pop,whoosh,impact,pluck}.py`.
