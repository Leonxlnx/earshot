"""Curated sound-design events in the reference films (picked by looking at earshot sheets, overview pages and zoomed
spectrogram crops; times in seconds of the film's own audio). Film ids only (ref_1..ref_8): see videos.py.

EVENTS rows: (film, t0, t1, category, picture cut or None, note, bed window or None)
  note starting with 'noexp' = measured for the study but not exported as a reference (masked / contaminated)
  bed None = the quietest 150 ms in the 0.6 s before t0 (measure.quiet_bed)
Categories follow earshot.forge.refsets.CATEGORIES, plus three the films use a lot that the forge list lacks:
  pop_in   a designed micro-hit for an element appearing: a short low thump (70-200 Hz) fused with a short noise burst
           (60-150 ms), usually fired in rhythmic runs (one per element)
  zip      a fast tonal pitch sweep (a sine / saw glide, 0.1-0.4 s), up for 'in', down for 'out'
  typing_run  a run of designed typing ticks (whole run, 0.5-2 s)
"""

SPLIT = {   # film -> split for EVERY category (no film feeds both sides)
    'ref_1': 'bench', 'ref_2': 'forge', 'ref_3': 'forge', 'ref_4': 'bench',
    'ref_5': 'bench', 'ref_6': 'bench', 'ref_7': 'forge', 'ref_8': 'bench',
}

NOTES = {
    'whoosh_big': 'forge only (one SFX-only film); the bench films bury their whooshes under music + voice.',
    'impact': 'forge only (one SFX-only film); designed, sub-heavy, gated tails.',
    'shimmer': 'forge only; one granular sparkle + one melodic chime run.',
    'glitch': 'forge only, one example.', 'notify': 'forge only.', 'tap': 'forge only, one example.',
    'pop_in': 'not a forge category yet: element-in micro hits (thump + tick), forge only.',
    'typing_run': 'not a forge category yet: one whole designed typing run.',
    'zip': 'not a forge category yet: tonal pitch sweeps, bench only (two films over music).',
    'sub_drop': 'bench only, one example under music + voice.',
    'riser': 'forge: one designed stutter riser; bench: one noise riser at a film end.',
    'logo_sting': 'forge: falling chirp into a tonal hum; bench: a music end chord with a 4 s tail.',
}

EVENTS = [
    # ---- ref_2: a pure sound-design piece (no music bed), the cleanest examples
    ('ref_2', 2.30, 2.82, 'whoosh_big', 2.88, 'low swell into the cut (ends before the clicks that follow)'),
    ('ref_2', 8.28, 9.30, 'whoosh_big', 8.64, 'soft round whoosh, peak just after the cut'),
    ('ref_2', 11.55, 12.62, 'whoosh_big', 12.64, 'noexp soft swell under the typing run (clicks on top)'),
    ('ref_2', 4.715, 5.95, 'impact', None, 'hit + granular low rumble tail, gated end'),
    ('ref_2', 14.02, 14.62, 'impact', 14.28, 'hit with sustained air, hard gated end'),
    ('ref_2', 16.78, 17.62, 'riser', 17.6, 'stutter-gated rising tone over a drone, peaks on the cut'),
    ('ref_2', 6.15, 8.10, 'shimmer', None, 'granular sparkle texture, swells and fades'),
    ('ref_2', 24.25, 26.20, 'shimmer', None, 'melodic chime run, gated end'),
    ('ref_2', 27.90, 28.60, 'glitch', 28.6, 'crackle then dense stutter, ends on the cut'),
    ('ref_2', 28.62, 29.45, 'logo_sting', 28.6, 'falling chirp into a low tonal hum', (26.5, 27.8)),
    ('ref_2', 13.10, 13.25, 'pop', None, 'falling chirp pop (10 kHz -> low)'),
    ('ref_2', 13.295, 13.37, 'pop', None, 'short knock pop ~1.3 kHz'),
    ('ref_2', 15.275, 15.50, 'notify', None, 'ding'),
    ('ref_2', 15.553, 15.80, 'notify', None, 'ding'),
    ('ref_2', 15.840, 16.20, 'notify', None, 'ding'),
    ('ref_2', 16.275, 16.55, 'notify', None, 'ding / glass'),
    ('ref_2', 18.705, 18.80, 'tap', 18.8, 'low thump + click'),
    ('ref_2', 2.938, 3.02, 'click', None, 'noexp bright click pair inside a chime texture'),
    ('ref_2', 3.849, 3.90, 'click', None, 'broadband click'),
    ('ref_2', 3.766, 3.845, 'hover_tick', None, 'high tonal tick'),
    ('ref_2', 3.905, 3.96, 'hover_tick', None, 'high tonal tick'),
    ('ref_2', 4.028, 4.10, 'hover_tick', None, 'high tonal tick'),
    ('ref_2', 10.458, 10.53, 'typing_key', None, 'designed typing tick (high-passed)'),
    ('ref_2', 10.686, 10.78, 'typing_key', None, ''),
    ('ref_2', 10.797, 10.88, 'typing_key', None, ''),
    ('ref_2', 11.130, 11.20, 'typing_key', None, ''),
    ('ref_2', 11.464, 11.54, 'typing_key', None, ''),
    ('ref_2', 11.551, 11.61, 'typing_key', None, ''),
    ('ref_2', 10.40, 11.95, 'typing_run', 10.68, 'whole designed typing run'),
    ('ref_2', 20.109, 20.25, 'pop_in', 20.08, 'element-in micro hit'),
    ('ref_2', 20.632, 20.79, 'pop_in', None, ''),
    ('ref_2', 21.105, 21.30, 'pop_in', None, ''),
    ('ref_2', 21.313, 21.58, 'pop_in', None, ''),
    ('ref_2', 21.826, 22.03, 'pop_in', None, ''),
    ('ref_2', 22.617, 22.81, 'pop_in', None, ''),
    ('ref_2', 22.821, 23.04, 'pop_in', None, ''),
    ('ref_2', 23.044, 23.22, 'pop_in', None, ''),
    ('ref_2', 23.226, 23.41, 'pop_in', None, ''),
    ('ref_2', 23.421, 23.60, 'pop_in', 23.4, ''),
    # ---- ref_6: kinetic type cut to the beat (music bed); designed zips and opening UI ticks
    ('ref_6', 1.60, 1.86, 'zip', 1.64, 'upward tonal sweep over the bed'),
    ('ref_6', 3.78, 4.06, 'zip', 4.04, 'upward sweep 100 Hz -> 10 kHz, ends on the cut'),
    ('ref_6', 0.100, 0.150, 'hover_tick', None, 'high noise tick 3-12 kHz over a near-silent bed'),
    ('ref_6', 0.205, 0.240, 'click', 0.24, 'broadband click'),
    ('ref_6', 0.343, 0.385, 'hover_tick', None, 'high noise tick'),
    ('ref_6', 0.588, 0.625, 'hover_tick', None, 'mid-high tick (2 kHz centred)'),
    # ---- ref_5: explainer, voice-over + music + UI sounds
    ('ref_5', 2.825, 2.88, 'click', None, 'broadband click with a low body, over music'),
    ('ref_5', 16.835, 16.96, 'pop', None, 'tonal blip ~1.5 kHz + a click, over music'),
    ('ref_5', 22.80, 22.98, 'zip', 22.9, 'falling tonal sweep 8 kHz -> 200 Hz over the cut'),
    ('ref_5', 28.22, 28.48, 'zip', None, 'arc sweep up then down (1 -> 3 -> 1 kHz)'),
    # ---- ref_7: aggressive music wall to wall; a noise riser into a drop, the opening drop hit
    ('ref_7', 26.90, 27.48, 'riser', 27.7, 'noexp noise riser into the drop, only +5 dB over the music (part of it)', (26.40, 26.85)),
    ('ref_7', 1.625, 2.10, 'impact', 1.767, 'noexp drop hit, the music enters with it', (1.0, 1.55)),
    # ---- ref_8: long voice-over product film over music; few exposed effects
    ('ref_8', 85.58, 86.45, 'sub_drop', 85.627, 'sub tone under the cut with a bright transient on top'),
    ('ref_8', 88.76, 93.00, 'logo_sting', None, 'end hit + long ringing chord tail (music-like)'),
    ('ref_8', 32.570, 32.612, 'click', None, 'noexp click in a voice-over gap, only +3 dB over the bed'),
    ('ref_8', 32.680, 32.76, 'typing_key', None, 'key tick in a voice-over gap'),
    # ---- ref_1: voice-over + music; the exposed ending riser, whooshes masked
    ('ref_1', 19.30, 20.45, 'riser', None, 'noise riser to the end of the film', (18.6, 19.2)),
    ('ref_1', 10.90, 11.30, 'whoosh_big', 11.053, 'noexp masked by a music onset'),
    ('ref_1', 5.95, 6.60, 'whoosh_big', 6.381, 'noexp masked by music + voice'),
]
