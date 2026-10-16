"""A designed riser (uplifter / build) for motion-design edits: 0.8-4 s of rising tension that ends exactly on the hit.

Every sample is computed; no recorded audio is loaded, sampled or convolved. This is a DESIGNED sound, not a physical
model: it follows how pros build risers (docs/motion-sfx-techniques.md section 2) and what the pro films measurably do
(docs/motion-sfx-study.md, riser):
  * One motion leads, the others support it. Up to five layers, each owning a band:
      noise   white / pink noise under a resonant low-pass whose cutoff opens nf0 -> nf1 (the filter riser), high-passed
              so nothing piles up under 100-200 Hz (the cheap-riser mistake); decorrelated, its width grows to the peak
      tonal   a unison of 5 detuned voices (sine / tri / saw) gliding up `toct` octaves from tf0 on an exponential
              (equal octaves per second) curve, with an octave layer that fades in late; tmode 'blips' instead plays
              short percussive tone blips on the gate, each a small falling chirp, their pitch stepping up pulse by
              pulse: the stepped arpeggio of the pro stutter riser (ref_2: blips ~2 kHz -> 8-13 kHz)
      shepard octave-spaced sines gliding up under a fixed loudness bell (dsp.shepard): an endless rise
      drone   a steady low saw + sine (35-110 Hz, low-passed at ~450 Hz) under it all, swelling only a little
      verb    a stereo FDN reverb of the build, so the space grows with it (cut at the end, never after the hit)
  * Curves accelerate (concave in dB). Level, filter and pitch move on w = u^curve (u = t / length): with curve 2-3
    most of the change lands in the last third, as in all three pro risers. The level rises rise_db over the build.
  * The stutter gate. A gate whose period shrinks gate_p0 -> gate_p1 (pro: 24 -> 20 ms, 40-50 pulses/s; slower
    tremolo 1 -> 12 Hz for long builds), depth fading in from gate_on; gate_sharp blends a smooth tremolo into a hard
    gate with 1 ms ramps (no clicks). Its phase is laid out backwards from the end so the last pulse is open on the
    final sample: the riser stops on its peak.
  * The end. Hard: the loudest moment is the last ~50 ms, then an end_ms fade (2-20 ms). The file ends on the cut and
    hands over to the hit (study: risers end on the cut, 0-20 ms).
  * Glue: light tanh saturation on the sum, a 30 Hz high-pass, peak normalised by the forge.

CHARACTER parameters (what makes the variants of a family audibly different but one palette): length, curve, rise
depth, brightness (nf1, tbright), pitch range (tf0, toct), weight (drone, hp), texture (noise vs tonal vs shepard,
blips, gate). FAMILIES holds the hand-designed character presets the final runs start from; DEFAULTS is the 'air'
build.

REFSET 'riser' is thin (one designed trailer riser in the forge split; the bench is two real cymbal swells); the
'pro_riser' set (one stutter riser from the pro films) is used for the short stutter family. The search runs in
designed mode, and `target_check()` adds the study's model-free targets to the critic (concave rise, ends on its
peak, high band rising, little energy under 100 Hz).
"""
import numpy as np
from scipy import signal

from .. import dsp, fx

TARGET = 'a cinematic riser building tension into a hit'
ALTS = ['a jet engine', 'an air raid siren', 'heavy rain on a roof', 'a vacuum cleaner']
REFSET = 'riser'

SPACE = {
    'length': (0.8, 4.0),                 # s, the hit lands on the last sample
    'curve': (1.0, 4.0),                  # acceleration of level / filter / pitch (1 = straight in dB)
    'rise_db': (12.0, 45.0),              # level rise over the build
    'noise_db': (-36.0, 0.0),             # layer levels (each layer normalised at its loudest)
    'nf0': (150.0, 1500.0, 'log'),        # noise low-pass: start ...
    'nf1': (2500.0, 16000.0, 'log'),      # ... and end cutoff (brightness at the peak)
    'nq': (0.7, 4.0, 'log'),              # resonance of the noise filter
    'pink': (0.0, 1.0),                   # 0 white (hissy) .. 1 pink (rounder)
    'tone_db': (-36.0, 0.0),
    'tmode': ['glide', 'blips'],
    'wave': ['sine', 'tri', 'saw'],
    'tf0': (60.0, 4000.0, 'log'),         # tonal start pitch (Hz)
    'toct': (0.5, 3.0),                   # octaves of rise
    'tbright': (1.5, 12.0, 'log'),        # saw low-pass = pitch x tbright
    'detune': (0.0, 30.0),                # unison spread in cents (widens as it rises)
    'oct_db': (-36.0, -3.0),              # +12 layer, fades in late
    'shep_db': (-48.0, 0.0),
    'shep_rate': (0.3, 1.6),              # octaves per second of the shepard glide
    'drone_db': (-48.0, -3.0),
    'drone_f': (35.0, 110.0, 'log'),
    'metal_db': (-48.0, 0.0),             # inharmonic shimmer partials (a reverse-cymbal wash)
    'metal_f': (1500.0, 5000.0, 'log'),   # lowest metal partial
    'gate_depth': (0.0, 1.0),
    'gate_on': (0.0, 0.7),                # fraction of the length where the gate fades in
    'gate_p0': (0.02, 0.5, 'log'),        # gate period at the start (s)
    'gate_p1': (0.012, 0.08, 'log'),      # gate period at the end (s)
    'gate_duty': (0.25, 0.8),
    'gate_sharp': (0.0, 1.0),             # 0 smooth tremolo .. 1 hard gate (1 ms ramps)
    'gate_drone': (0.0, 1.0),             # how much the gate also chops the drone
    'blip_dec': (0.1, 0.5),               # blips: decay time constant as a fraction of the pulse period
    'verb': (0.0, 0.6),
    'rt60': (0.6, 3.5, 'log'),
    'width': (0.0, 1.0),
    'sat': (0.0, 12.0),                   # tanh drive on the sum (dB)
    'hp': (60.0, 500.0, 'log'),           # high-pass of everything but the drone
    'end_ms': (2.0, 20.0, 'log'),         # final fade (ms): the end is a cut, not a ring-out
}

# character presets (round 1 hand design, then refined by the forge; see out/forge_v2/riser/LOG.md)
FAMILIES = {
    # a bright noise swell: the filter leads, a faint glide and growing space under it
    'air': {'length': 2.6, 'curve': 2.4, 'rise_db': 36.0, 'noise_db': 0.0, 'nf0': 300.0, 'nf1': 12000.0, 'nq': 1.6,
            'pink': 0.55, 'tone_db': -20.0, 'tmode': 'glide', 'wave': 'tri', 'tf0': 400.0, 'toct': 1.5, 'tbright': 4.0,
            'detune': 12.0, 'oct_db': -30.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -30.0, 'drone_f': 55.0,
            'gate_depth': 0.0, 'gate_on': 0.5, 'gate_p0': 0.2, 'gate_p1': 0.04, 'gate_duty': 0.5, 'gate_sharp': 0.0,
            'gate_drone': 0.0, 'verb': 0.3, 'rt60': 2.0, 'width': 0.7, 'sat': 2.0, 'hp': 180.0, 'end_ms': 6.0},
    # a supersaw tension build: pitch leads (+2 oct), tremolo speeding up 4 -> 16 Hz, noise and drone support
    'tension': {'length': 3.2, 'curve': 2.6, 'rise_db': 32.0, 'noise_db': -10.0, 'nf0': 250.0, 'nf1': 9000.0,
                'nq': 1.2, 'pink': 0.7, 'tone_db': 0.0, 'tmode': 'glide', 'wave': 'saw', 'tf0': 110.0, 'toct': 2.0,
                'tbright': 5.0, 'detune': 18.0, 'oct_db': -10.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -14.0,
                'drone_f': 55.0, 'gate_depth': 0.55, 'gate_on': 0.25, 'gate_p0': 0.25, 'gate_p1': 0.06,
                'gate_duty': 0.5, 'gate_sharp': 0.2, 'gate_drone': 0.0, 'verb': 0.3, 'rt60': 2.4, 'width': 0.5,
                'sat': 5.0, 'hp': 120.0, 'end_ms': 5.0},
    # the pro stutter riser (ref_2): stepped tone blips 2 -> 8 kHz on an accelerating 45/s gate over a gated drone
    'stutter': {'length': 1.1, 'curve': 2.0, 'rise_db': 26.0, 'noise_db': -14.0, 'nf0': 600.0, 'nf1': 12000.0,
                'nq': 1.0, 'pink': 0.3, 'tone_db': 0.0, 'tmode': 'blips', 'wave': 'sine', 'tf0': 1800.0, 'toct': 2.1,
                'tbright': 4.0, 'detune': 6.0, 'oct_db': -24.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -16.0,
                'drone_f': 95.0, 'gate_depth': 1.0, 'gate_on': 0.0, 'gate_p0': 0.05, 'gate_p1': 0.022,
                'gate_duty': 0.55, 'gate_sharp': 1.0, 'gate_drone': 0.8, 'verb': 0.12, 'rt60': 1.0, 'width': 0.3,
                'sat': 4.0, 'hp': 250.0, 'end_ms': 4.0},
    # an endless shepard rise in a big space, dark to bright, slow and wide
    'shepard': {'length': 3.8, 'curve': 1.8, 'rise_db': 30.0, 'noise_db': -12.0, 'nf0': 200.0, 'nf1': 7000.0,
                'nq': 2.2, 'pink': 0.9, 'tone_db': -36.0, 'tmode': 'glide', 'wave': 'sine', 'tf0': 300.0,
                'toct': 1.0, 'tbright': 3.0, 'detune': 8.0, 'oct_db': -36.0, 'shep_db': 0.0, 'shep_rate': 0.9,
                'drone_db': -22.0, 'drone_f': 45.0, 'gate_depth': 0.0, 'gate_on': 0.5, 'gate_p0': 0.2,
                'gate_p1': 0.04, 'gate_duty': 0.5, 'gate_sharp': 0.0, 'gate_drone': 0.0, 'verb': 0.45, 'rt60': 3.0,
                'width': 0.9, 'sat': 1.5, 'hp': 140.0, 'end_ms': 8.0},
    # a short punchy pitch riser: a sine-tri glide +3 oct with a hard gate rolling 12 -> 50 Hz, little noise
    'rush': {'length': 1.6, 'curve': 3.0, 'rise_db': 30.0, 'noise_db': -8.0, 'nf0': 400.0, 'nf1': 14000.0,
             'nq': 2.8, 'pink': 0.2, 'tone_db': -2.0, 'tmode': 'glide', 'wave': 'tri', 'tf0': 220.0, 'toct': 3.0,
             'tbright': 6.0, 'detune': 10.0, 'oct_db': -14.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'drone_db': -20.0,
             'drone_f': 82.0, 'gate_depth': 0.8, 'gate_on': 0.15, 'gate_p0': 0.085, 'gate_p1': 0.02,
             'gate_duty': 0.5, 'gate_sharp': 0.85, 'gate_drone': 0.3, 'verb': 0.18, 'rt60': 1.4, 'width': 0.4,
             'sat': 6.0, 'hp': 200.0, 'end_ms': 4.0},
}
for _f in FAMILIES.values():
    _f.setdefault('metal_db', -48.0)
    _f.setdefault('metal_f', 2600.0)
    _f.setdefault('blip_dec', 0.35)
# round 3: 'rush' was a gated tri / saw glide that CLAP heard as 'a square wave chiptune blip' (z 1.7-2.6, 0/3 pass);
# now the noise leads (a resonant whistle opening 500 Hz -> 15 kHz) with a sine glide under it and a smooth flutter
# speeding up 10 -> 35 Hz
FAMILIES['rush'].update({'length': 1.5, 'curve': 3.2, 'noise_db': 0.0, 'nf0': 500.0, 'nf1': 15000.0, 'nq': 2.4,
                         'pink': 0.4, 'tone_db': -10.0, 'wave': 'sine', 'tf0': 260.0, 'toct': 2.8, 'oct_db': -20.0,
                         'gate_depth': 0.5, 'gate_on': 0.1, 'gate_p0': 0.1, 'gate_p1': 0.028, 'gate_sharp': 0.1,
                         'gate_drone': 0.0, 'drone_db': -22.0, 'sat': 2.0, 'verb': 0.22, 'width': 0.5})
# a reverse-cymbal swell: metallic inharmonic wash + bright noise, exponential, no tone (the bench riser kind)
FAMILIES['cymbal'] = {**FAMILIES['air'], 'length': 2.2, 'curve': 2.8, 'rise_db': 40.0, 'noise_db': -4.0, 'nf0': 900.0,
                      'nf1': 16000.0, 'nq': 0.8, 'pink': 0.25, 'tone_db': -36.0, 'oct_db': -36.0, 'drone_db': -48.0,
                      'metal_db': 0.0, 'metal_f': 2400.0, 'verb': 0.35, 'rt60': 1.6, 'width': 0.85, 'sat': 1.0,
                      'hp': 400.0, 'end_ms': 4.0}
# the pro stutter set (short builds, docs/motion-sfx-study.md riser: 0.8-1.5 s, gate 24 -> 20 ms, +2 oct tone, drone)
_ST = FAMILIES['stutter']
_ST.update({'length': 0.9, 'curve': 1.6, 'rise_db': 16.0, 'noise_db': -20.0, 'tf0': 1700.0, 'toct': 2.6,
            'gate_p0': 0.055, 'gate_p1': 0.036, 'gate_duty': 0.5, 'drone_db': -12.0, 'gate_drone': 1.0, 'verb': 0.08,
            'blip_dec': 0.2})
PRO = {
    'blips': dict(_ST),
    # gated noise: a filter sweep chopped by the same accelerating 45/s gate, a thin tone on top
    'chop': {**_ST, 'length': 1.0, 'tmode': 'glide', 'wave': 'tri', 'tone_db': -14.0, 'noise_db': 0.0, 'nf0': 700.0,
             'nf1': 14000.0, 'nq': 2.0, 'gate_p0': 0.04, 'gate_p1': 0.02, 'gate_duty': 0.45, 'drone_db': -20.0,
             'hp': 500.0, 'rise_db': 26.0, 'curve': 2.2, 'verb': 0.05},
    # a saw glide under a hard accelerating gate, darker, longer
    'gated_saw': {**_ST, 'length': 1.4, 'tmode': 'glide', 'wave': 'saw', 'tf0': 300.0, 'toct': 2.0, 'tbright': 3.0,
                  'detune': 14.0, 'noise_db': -12.0, 'gate_p0': 0.06, 'gate_p1': 0.022, 'gate_duty': 0.6,
                  'drone_db': -14.0, 'drone_f': 60.0, 'hp': 150.0, 'sat': 3.0, 'rise_db': 26.0, 'curve': 2.2,
                  'verb': 0.06},
    # a roll: noise bursts speeding up 12 -> 45 per second (a snare-roll build), no tone
    'roll': {**_ST, 'length': 1.3, 'curve': 2.4, 'tmode': 'glide', 'tone_db': -36.0, 'oct_db': -36.0, 'noise_db': 0.0,
             'nf0': 1200.0, 'nf1': 12000.0, 'nq': 0.9, 'pink': 0.6, 'gate_p0': 0.085, 'gate_p1': 0.022,
             'gate_duty': 0.4, 'gate_sharp': 1.0, 'drone_db': -24.0, 'gate_drone': 1.0, 'hp': 300.0, 'verb': 0.1,
             'rise_db': 28.0},
    # the shortest: 0.8 s, higher blips 3 -> 12 kHz, faster gate
    'blips_hi': {**_ST, 'length': 0.8, 'tf0': 2600.0, 'toct': 2.2, 'gate_p0': 0.045, 'gate_p1': 0.028,
                 'drone_db': -18.0, 'drone_f': 110.0, 'noise_db': -14.0, 'rise_db': 20.0},
}
FAMILIES_BY_CAT = {'riser': {k: FAMILIES[k] for k in ('air', 'tension', 'shepard', 'rush', 'cymbal')},
                   'pro_riser': PRO}
TARGETS = {'pro_riser': 'a short stuttering digital riser into a cut'}
DEFAULTS = dict(FAMILIES['air'])

UNISON = (-1.0, -0.45, 0.0, 0.45, 1.0)
FINAL_OPEN = 0.045    # s: the gate opens (and the blips sustain) over the last 45 ms, so the riser peaks on the cut     # voice offsets (x detune cents) and pans (x width)


# ------------------------------------------------------------------------------------------------ shared helpers
# (downlifter.py and zip.py import these; kept here because recipes may not change the core modules)
def lvl(x, frac=None):
    """peak of a 10 ms RMS envelope (frac: only the last `frac` of the sound): a layer's level for mixing"""
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    if frac:
        m = m[int(len(m) * (1 - frac)):]
    h = dsp.n_of(0.01)
    k = max(1, len(m) // h)
    return float(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)).max()) + 1e-12


def phase_from_end(rate, end_phase=0.0):
    """cycles (float) of a pulse train with a per-sample rate (Hz), laid out backwards so phase(end) = end_phase"""
    c = np.cumsum(rate[::-1])[::-1] / dsp.SR             # cycles from each sample to the end
    return end_phase - c


def gate_shape(phase, period, duty=0.5, sharp=1.0, ramp=0.0012, perc=0.0):
    """one value per sample in 0..1: a smooth tremolo (sharp 0) .. a hard gate with `ramp` s edges (sharp 1);
    perc > 0 makes each pulse a percussive blip decaying with time constant perc x period"""
    fr = np.mod(phase, 1.0)
    trem = 0.5 + 0.5 * np.cos(2 * np.pi * (fr - 0.5 * duty))
    on = np.clip(fr * period / ramp, 0, 1)
    off = np.clip((duty - fr) * period / (ramp * 1.5), 0, 1)
    sq = np.minimum(on, off)
    if perc > 0:
        sq = on * np.exp(-fr / max(perc, 1e-3)) * np.clip((0.98 - fr) * period / ramp, 0, 1)
    return (1 - sharp) * trem + sharp * sq


def unison(freq, wave, detune, width, phases):
    """5 detuned voices (cents offsets UNISON x detune, an array over time allowed), panned UNISON x width: stereo"""
    out = np.zeros((len(freq), 2))
    for k, off in enumerate(UNISON):
        f = freq * 2 ** (off * detune / 1200.0)
        v = dsp.osc(f, wave, phase=phases[k])
        th = (np.clip(off * width, -1, 1) + 1) * np.pi / 4
        out[:, 0] += v * np.cos(th)
        out[:, 1] += v * np.sin(th)
    return out * np.sqrt(2) / len(UNISON)


_IR = {}


def verb_ir(rt60, size=0.75, predelay=0.02, damp=0.5, lowcut=200.0):
    """a cached stereo FDN reverb IR (fx.reverb_ir on a fixed rng, rt60 rounded to 0.1 s): building the IR costs ~1 s,
    convolving it ~0.1 s, so a search re-uses one IR per decay time (the noise seeds still vary every render)"""
    key = (round(float(rt60), 1), round(size, 2), round(predelay, 3), round(damp, 2), round(lowcut))
    if key not in _IR:
        _IR[key] = fx.reverb_ir(dsp.rng(1234), key[0], size, predelay, damp, inputs=2, lowcut=lowcut)
    return _IR[key]


def wet_of(x, rt60, n_out=None, **kw):
    """the reverb return alone (stereo, unit-energy IR) of x, cut to n_out samples (default len(x) + the tail)"""
    ir = verb_ir(rt60, **kw)
    x2 = stereo_of(x)
    n_out = n_out or len(x2) + len(ir) - 1
    out = np.zeros((len(x2) + len(ir) - 1, 2))
    for c in range(2):
        out += signal.fftconvolve(x2[:, c:c + 1], ir[:, c, :], axes=0)
    return out[:n_out]


def stereo_of(x):
    x = np.asarray(x, np.float64)
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def env_db(x, hop_ms=10.0):
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    h = max(1, int(hop_ms * dsp.SR / 1000))
    k = max(1, len(m) // h)
    return 20 * np.log10(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-9)


def band_env_db(x, lo, hi, hop_ms=10.0):
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    y = dsp.butter(m, 'band', (lo, min(hi, 23000)), order=4) if lo > 20 else dsp.butter(m, 'low', hi, order=4)
    return env_db(y, hop_ms)


def side_db(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        return -99.0
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return float(10 * np.log10((side ** 2).sum() / ((mid ** 2).sum() + 1e-20) + 1e-12))


def low_share_db(x, f=100.0):
    """energy under f Hz re the total (dB)"""
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    lo = dsp.butter(m, 'low', f, order=4)
    return float(10 * np.log10((lo ** 2).sum() / ((m ** 2).sum() + 1e-20) + 1e-12))


def centroid_track(x, hop_ms=20.0):
    """spectral centroid (Hz) per frame of a mono fold"""
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    h = int(hop_ms * dsp.SR / 1000)
    k = max(1, len(m) // h)
    fr = m[:k * h].reshape(k, h) * np.hanning(h)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(h, 1 / dsp.SR)
    return (S * f).sum(1) / (S.sum(1) + 1e-20), S.sum(1)


# ------------------------------------------------------------------------------------------------ measurement
def measure(x):
    """the study's riser targets, model-free: length_s, last_third (share of the dB rise in the last third; pro risers
    are concave: >= 0.5), end_db (the last 50 ms re the peak: ~0 = it stops on its peak), hf_rise_db (> 4 kHz band,
    last 10 % vs first 25 %: pro +30-35 dB), cent_oct (centroid rise in octaves), side_db (pro: about -14..-18),
    low_db (energy < 100 Hz re total)"""
    e = env_db(x)
    k = len(e)
    pk = e.max()
    a = np.median(e[:max(1, k // 10)])
    b = e[-max(1, k // 10):].max()
    sm = np.convolve(e, np.ones(5) / 5, mode='same')
    two = sm[min(k - 1, int(k * 2 / 3))]
    rise = max(b - a, 1e-3)
    hf = band_env_db(x, 4000, 20000)
    c, pw = centroid_track(x)
    loud = pw > pw.max() * 1e-4
    ci = np.nonzero(loud)[0]
    cent_oct = float(np.log2(np.median(c[ci[-3:]]) / max(np.median(c[ci[:3]]), 1.0))) if len(ci) >= 6 else 0.0
    return {'length_s': round(len(x) / dsp.SR, 3), 'last_third': round(float((b - two) / rise), 2),
            'end_db': round(float(e[-5:].max() - pk), 1),
            'hf_rise_db': round(float(hf[-max(1, k // 10):].max() - np.median(hf[:max(1, k // 4)])), 1),
            'cent_oct': round(cent_oct, 2), 'side_db': round(side_db(x), 1), 'low_db': round(low_share_db(x), 1)}


def target_check(x, row=None):
    """critic `extra`: minus the distance (about 0..3) from the study's riser targets"""
    m = measure(x)
    pen = max(0.0, 0.45 - m['last_third']) * 4                  # concave: most of the rise late
    pen += max(0.0, -3.0 - m['end_db']) / 4                      # stops on its peak
    pen += max(0.0, 20.0 - m['hf_rise_db']) / 10                 # the top opens up
    pen += max(0.0, m['low_db'] + 10.0) / 6                      # no pile-up under 100 Hz
    pen += max(0.0, m['side_db'] + 6.0) / 6                      # mostly centred
    return -min(3.0, pen)


# ------------------------------------------------------------------------------------------------ render
def render(p, r):
    p = {**DEFAULTS, **p}
    L = float(p['length'])
    n = dsp.n_of(L)
    t = dsp.t_of(n)
    u = t / L
    w = u ** p['curve']                                          # accelerating progress
    lv = 10 ** (-p['rise_db'] * (1 - w) / 20)

    # gate (laid out from the end so the last pulse is open on the final sample)
    per = p['gate_p0'] * (p['gate_p1'] / p['gate_p0']) ** (u ** 1.4)
    ph = phase_from_end(1.0 / per, end_phase=0.5 * p['gate_duty'])
    blips = p['tmode'] == 'blips'
    g = gate_shape(ph, per, p['gate_duty'], p['gate_sharp'])
    on = np.clip((u - p['gate_on']) / 0.2, 0, 1)
    fin = np.clip((L - t) / FINAL_OPEN, 0, 1)                     # the last FINAL_OPEN s: gate open, the peak
    fin = fin * fin * (3 - 2 * fin)
    depth = p['gate_depth'] * (on * on * (3 - 2 * on)) * fin
    gate = 1 - depth * (1 - g)

    # noise: resonant low-pass opening on the curve, high-passed, decorrelated and widening
    wn = (1 - p['pink']) * r.standard_normal(n) + p['pink'] * dsp.noise(L, r, 'pink')[:n] * 1.6
    fc = p['nf0'] * (p['nf1'] / p['nf0']) ** w
    nz = dsp.svf(wn, fc, p['nq'], 'low')
    nz = dsp.butter(nz, 'high', p['hp'], order=4)
    nz_st = fx.width(fx.decorrelate(nz, 1.0), p['width'] * (0.35 + 0.65 * w))

    # tonal: unison glide (or stepped blips on the gate) + a late octave layer
    wt = u ** max(1.0, 0.8 * p['curve'])
    f = p['tf0'] * 2 ** (p['toct'] * wt)
    ph0 = r.uniform(0, 1, 2 * len(UNISON))
    if blips:
        idx = np.floor(ph).astype(np.int64)
        starts = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
        held = np.repeat(f[starts], np.diff(np.r_[starts, n]))   # pitch held per pulse: a stepped arpeggio
        fr = np.mod(ph, 1.0)
        fb = held * 2 ** (-0.35 * fr)                             # each blip a small falling chirp
        tone = unison(fb, p['wave'], p['detune'], 0.25 * p['width'], ph0[:5])
        tone *= (gate_shape(ph, per, p['gate_duty'], 1.0, perc=p['blip_dec']) * fin + 1 - fin)[:, None]
    else:
        tone = unison(f, p['wave'], p['detune'] * (0.6 + 0.8 * w), 0.5 * p['width'], ph0[:5])
    octv = unison(2 * (held * 2 ** (-0.35 * np.mod(ph, 1.0)) if blips else f), p['wave'], p['detune'] * 0.7,
                  0.6 * p['width'], ph0[5:])
    octv *= np.clip((u - 0.4) / 0.6, 0, 1)[:, None] ** 2
    if blips:
        octv *= (gate_shape(ph, per, p['gate_duty'], 1.0, perc=0.7 * p['blip_dec']) * fin + 1 - fin)[:, None]
    if p['wave'] == 'saw':
        cut = np.minimum(f * p['tbright'], 18000)
        tone = np.stack([dsp.svf(tone[:, c], cut, 0.8, 'low') for c in range(2)], axis=1)
        octv = np.stack([dsp.svf(octv[:, c], np.minimum(2 * cut, 18000), 0.8, 'low') for c in range(2)], axis=1)
    tone = dsp.butter(tone, 'high', p['hp'] * 0.8, order=2)
    octv = dsp.butter(octv, 'high', p['hp'], order=2)

    # shepard: an endless glide up
    if p['shep_db'] > -47:
        sh = dsp.shepard(L, r, up=True, rate=p['shep_rate'], f_lo=60.0, f_hi=9000.0, kind='tone')[:n]
        sh = dsp.butter(sh, 'high', p['hp'], order=2)
        sh_st = fx.width(fx.decorrelate(sh, 0.6), 0.6 * p['width'])
    else:
        sh_st = np.zeros((n, 2))

    # drone: steady low saw + sine, gently swelling
    df = p['drone_f'] * 2 ** (0.08 * w)
    dr = sum(dsp.osc(df * 2 ** (c / 1200), 'saw', phase=r.uniform()) for c in (-7, 0, 7)) / 3
    dr = dsp.butter(dr, 'low', 500, order=2) * 0.8 + 0.35 * dsp.osc(df, 'sine', phase=r.uniform())
    dr = dsp.butter(dr, 'high', 30, order=2)
    dr *= 10 ** (-0.3 * p['rise_db'] * (1 - w) / 20)
    g2 = gate_shape(2 * ph, per / 2, p['gate_duty'], p['gate_sharp']) if blips else g   # blips: drone at 2x rate
    dr *= 1 - p['gate_drone'] * depth * (1 - g2)

    # metal: inharmonic partials, each shimmering under its own slow random amplitude, brightening with the build
    if p['metal_db'] > -47:
        ratios = np.array([1.0, 1.37, 1.71, 2.13, 2.58, 3.11, 3.66, 4.29, 5.02])
        met = np.zeros(n)
        for k, q in enumerate(ratios):
            fk = p['metal_f'] * q * (1 + 0.004 * r.standard_normal())
            if fk > 19000:
                continue
            am = dsp.butter(r.standard_normal(n), 'low', 220, order=2)
            am = np.abs(am) / (np.abs(am).mean() + 1e-12)
            open_k = np.clip(w * len(ratios) * 0.9 - k * 0.6 + 1.0, 0, 1)
            met += np.sin(2 * np.pi * fk * t + r.uniform(0, 2 * np.pi)) * am * open_k / (1 + 0.15 * k)
        met = dsp.butter(met, 'high', 1200, order=2)
        met_st = fx.width(fx.decorrelate(met, 1.0), p['width'])
    else:
        met_st = None

    def at(db, x):
        return 10 ** (db / 20) * x / lvl(x, 0.25)

    lay_n, lay_t, lay_o = at(p['noise_db'], nz_st), at(p['tone_db'], tone), at(p['oct_db'], octv)
    lay_s = at(p['shep_db'], sh_st) if p['shep_db'] > -47 else 0.0
    if met_st is not None:
        lay_s = lay_s + at(p['metal_db'], met_st)
    if blips:                                                    # the blips carry their own percussive gate
        send = ((lay_n + lay_s) * gate[:, None] + lay_t + lay_o) * lv[:, None]
    else:
        send = (lay_n + lay_t + lay_o + lay_s) * (lv * gate)[:, None]
    mix = send + at(p['drone_db'], stereo_of(dr)) * (lv ** 0.3)[:, None]
    if p['verb'] > 0.005:
        wet = wet_of(send, p['rt60'], n)
        mix = mix + p['verb'] * wet / lvl(wet, 0.25) * lvl(send, 0.25)
    if p['sat'] > 0.1:
        mix = fx.saturate(mix, drive_db=p['sat'], mode='tanh')
    mix = dsp.butter(mix, 'high', 30, order=2)
    mix = dsp.butter(mix, 'low', 18000, order=4)                 # round 3: polish found -30..-34 dB above 20 kHz
    return dsp.fade(mix, 0.004, p['end_ms'] / 1000.0)
