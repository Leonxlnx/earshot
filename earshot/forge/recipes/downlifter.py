"""A designed downlifter (downsweep, fall) for motion-design edits: the push-and-release right after an impact.

Every sample is computed; no recorded audio is loaded, sampled or convolved. The mirror of the riser recipe
(riser.py), built the way docs/motion-sfx-techniques.md section 3 describes:
  * Decelerating curves: filter, pitch and level move on v = 1 - (1 - u)^curve (u = t / length): fast at first,
    settling at the end (curve 1 = straight in dB / octaves, 2-3 = a hard yank that sags out).
  * Layers, each owning a band:
      noise    white / pink noise under a resonant low-pass closing nf0 -> nf1 (8-16 kHz down to 150-500 Hz), wide
      tonal    a 5-voice unison (sine / tri / saw / square) falling `toct` octaves from tf0, its low-pass tracking the
               pitch (the harmonics stay audible as the fundamental sinks, the sine-square trick); tmode 'tape' instead
               holds a chord (root, fifth, octave) and tape-stops it over the length (speed (1 - u)^tape_curve)
      sub      a sine falling sub_f0 -> sub_f1 with its own decay (weight under the hit's tail), mono, dry
      shepard  octave-spaced sines gliding down under a fixed bell (an endless fall)
      flanger  a moving comb (0.4-9 ms delay, slowing down as it falls): the classic synth downlifter movement
  * The dry part dies first: a stereo reverb of the whole fall keeps ringing (verb, rt60) after the dry layers fade,
    so the space gets relatively wetter as it sinks.
  * The start swells in over att_ms (2-300 ms; round 2: a 2-8 ms push read as 'a cinematic hit / an explosion' to the
    critic, a 40-200 ms swell reads as a sweep) so it can sit right after an impact; the end is a natural fade to
    silence (no cut: the tail is trimmed only once it is 60 dB down, with a smooth fade).
CHARACTER parameters: length, curve, depth of the pitch fall, brightness (nf0, tbright), weight (sub, drone), texture
(noise / tonal / tape / shepard / flanger), space. FAMILIES are the hand-designed character presets.

REFSET 'downlifter' is empty (no licensed downlifter, real or designed), so the forge runs in designed mode: the
critic's description match, artifact and polish checks, plus `target_check()` (the model-free targets above: the
loudest point early, the centroid falling at least an octave, a decelerating fall, a clean faded tail, mostly centred
low end).
"""
import numpy as np

from .. import dsp, fx
from .riser import centroid_track, env_db, low_share_db, lvl, side_db, stereo_of, unison, wet_of

TARGET = 'a downlifter sound effect: a sinking filtered sweep'
ALTS = ['a jet engine', 'an air raid siren', 'a plane flying overhead', 'a gust of wind']
REFSET = 'downlifter'

SPACE = {
    'length': (1.0, 4.0),
    'curve': (1.0, 4.0),                  # deceleration (1 = straight, 3 = fast fall, long sag)
    'drop_db': (24.0, 50.0),              # dry level fall over the length
    'att_ms': (2.0, 300.0, 'log'),        # the swell-in at the start (round 2: a 2-8 ms push read as a hit)
    'noise_db': (-36.0, 0.0),
    'nf0': (5000.0, 16000.0, 'log'),      # noise low-pass start ...
    'nf1': (120.0, 800.0, 'log'),         # ... and end
    'nq': (0.7, 4.0, 'log'),
    'pink': (0.0, 1.0),
    'tone_db': (-36.0, 0.0),
    'tmode': ['fall', 'tape'],
    'wave': ['sine', 'tri', 'saw', 'square'],
    'tf0': (150.0, 2500.0, 'log'),        # tonal start pitch
    'toct': (0.7, 3.0),                   # octaves of fall
    'tbright': (1.5, 10.0, 'log'),        # low-pass = pitch x tbright (saw / square)
    'detune': (0.0, 30.0),
    'tape_curve': (0.7, 2.5),
    'sub_db': (-48.0, 0.0),
    'sub_f0': (60.0, 140.0, 'log'),
    'sub_f1': (28.0, 50.0, 'log'),
    'shep_db': (-48.0, 0.0),
    'shep_rate': (0.3, 1.6),
    'flange': (0.0, 0.8),                 # comb mix
    'fl_rate': (0.3, 4.0, 'log'),         # comb sweep rate at the start (Hz), slowing to a third
    'verb': (0.0, 0.7),
    'rt60': (0.8, 4.0, 'log'),
    'width': (0.0, 1.0),
    'sat': (0.0, 10.0),
    'hp': (40.0, 300.0, 'log'),
}

FAMILIES = {
    # white-noise downsweep: the filter closes 14 kHz -> 250 Hz, wide, a little space
    'air_fall': {'length': 2.2, 'curve': 2.2, 'drop_db': 40.0, 'att_ms': 90.0, 'noise_db': 0.0, 'nf0': 14000.0,
                 'nf1': 250.0, 'nq': 1.4, 'pink': 0.45, 'tone_db': -36.0, 'tmode': 'fall', 'wave': 'tri',
                 'tf0': 800.0, 'toct': 1.5, 'tbright': 4.0, 'detune': 10.0, 'tape_curve': 1.4, 'sub_db': -48.0,
                 'sub_f0': 90.0, 'sub_f1': 38.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'flange': 0.0, 'fl_rate': 1.0,
                 'verb': 0.3, 'rt60': 1.8, 'width': 0.8, 'sat': 1.0, 'hp': 80.0},
    # a supersaw falling two octaves with a slow flanger, noise underneath
    # (round 3: noise -30 dB, darker: with the noise at -10 dB polish's aliasing check read the noise under the saw
    #  partials as fold-back, 'fold' 0.42-0.48, a hard artifact; at -30 dB 0.02-0.06)
    'synth_fall': {'length': 2.8, 'curve': 2.0, 'drop_db': 36.0, 'att_ms': 120.0, 'noise_db': -30.0, 'nf0': 6000.0,
                   'nf1': 300.0, 'nq': 1.2, 'pink': 0.7, 'tone_db': 0.0, 'tmode': 'fall', 'wave': 'saw',
                   'tf0': 880.0, 'toct': 2.0, 'tbright': 4.0, 'detune': 16.0, 'tape_curve': 1.4, 'sub_db': -30.0,
                   'sub_f0': 90.0, 'sub_f1': 38.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'flange': 0.35,
                   'fl_rate': 1.2, 'verb': 0.35, 'rt60': 2.4, 'width': 0.6, 'sat': 3.0, 'hp': 60.0},
    # heavy and dark: a sine-square tone sinking 400 -> 70 Hz over a sub drop, short
    'sub_sink': {'length': 1.6, 'curve': 2.6, 'drop_db': 38.0, 'att_ms': 45.0, 'noise_db': -16.0, 'nf0': 6000.0,
                 'nf1': 150.0, 'nq': 1.0, 'pink': 0.9, 'tone_db': 0.0, 'tmode': 'fall', 'wave': 'square',
                 'tf0': 400.0, 'toct': 2.5, 'tbright': 2.5, 'detune': 6.0, 'tape_curve': 1.4, 'sub_db': -6.0,
                 'sub_f0': 85.0, 'sub_f1': 34.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'flange': 0.0, 'fl_rate': 1.0,
                 'verb': 0.2, 'rt60': 1.4, 'width': 0.3, 'sat': 4.0, 'hp': 40.0},
    # a tape stop: a held chord slowing to a halt
    'tape': {'length': 1.3, 'curve': 1.6, 'drop_db': 30.0, 'att_ms': 40.0, 'noise_db': -20.0, 'nf0': 9000.0,
             'nf1': 400.0, 'nq': 1.0, 'pink': 0.5, 'tone_db': 0.0, 'tmode': 'tape', 'wave': 'saw', 'tf0': 330.0,
             'toct': 1.0, 'tbright': 5.0, 'detune': 10.0, 'tape_curve': 1.5, 'sub_db': -24.0, 'sub_f0': 82.0,
             'sub_f1': 36.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'flange': 0.0, 'fl_rate': 1.0, 'verb': 0.15,
             'rt60': 1.2, 'width': 0.4, 'sat': 3.0, 'hp': 50.0},
    # an endless shepard fall in a big space
    'shepard_fall': {'length': 3.4, 'curve': 1.5, 'drop_db': 34.0, 'att_ms': 200.0, 'noise_db': -10.0,
                     'nf0': 9000.0, 'nf1': 300.0, 'nq': 2.0, 'pink': 0.8, 'tone_db': -36.0, 'tmode': 'fall',
                     'wave': 'sine', 'tf0': 800.0, 'toct': 1.0, 'tbright': 3.0, 'detune': 8.0, 'tape_curve': 1.4,
                     'sub_db': -36.0, 'sub_f0': 80.0, 'sub_f1': 40.0, 'shep_db': 0.0, 'shep_rate': 0.9,
                     'flange': 0.0, 'fl_rate': 1.0, 'verb': 0.45, 'rt60': 3.0, 'width': 0.85, 'sat': 1.0,
                     'hp': 70.0},
    # noise through a deep flanger that slows as it falls (a jet-like sweep, designed)
    'flange_air': {'length': 3.0, 'curve': 1.8, 'drop_db': 38.0, 'att_ms': 150.0, 'noise_db': 0.0, 'nf0': 12000.0,
                   'nf1': 400.0, 'nq': 0.9, 'pink': 0.6, 'tone_db': -24.0, 'tmode': 'fall', 'wave': 'tri',
                   'tf0': 1200.0, 'toct': 2.0, 'tbright': 4.0, 'detune': 10.0, 'tape_curve': 1.4, 'sub_db': -40.0,
                   'sub_f0': 90.0, 'sub_f1': 38.0, 'shep_db': -48.0, 'shep_rate': 0.8, 'flange': 0.7,
                   'fl_rate': 0.8, 'verb': 0.3, 'rt60': 2.2, 'width': 0.7, 'sat': 1.0, 'hp': 90.0},
}
DEFAULTS = dict(FAMILIES['air_fall'])


def comb(x, delay_s, mix):
    """a moving comb (flanger): x + mix * x(t - delay(t)), linear-interpolated fractional delay (per sample)"""
    n = len(x)
    pos = np.arange(n) - np.asarray(delay_s) * dsp.SR
    y = np.interp(pos, np.arange(n), x, left=0.0)
    return (x + mix * y) / (1 + mix)


# ------------------------------------------------------------------------------------------------ measurement
def measure(x):
    """the downlifter targets, model-free: length_s, peak_frac (where the loudest 10 ms sits, as a fraction of the
    length: early), sag (dB fall in the first third / the whole fall: decelerating > 0.5), cent_oct (centroid fall,
    octaves: < -1), tail_db (the last 10 ms re the peak: a faded tail is < -45), side_db, low_db (< 100 Hz re total)"""
    e = env_db(x)
    k = len(e)
    ip = int(np.argmax(e))
    pk = e[ip]
    act = np.nonzero(e > pk - 45)[0]
    end = act[-1] if len(act) else k - 1
    third = ip + max(1, (end - ip) // 3)
    sm = np.convolve(e, np.ones(5) / 5, mode='same')
    fall = max(pk - sm[end], 1e-3)
    c, pw = centroid_track(x)
    loud = np.nonzero(pw > pw.max() * 1e-3)[0]
    cent_oct = float(np.log2(np.median(c[loud[-3:]]) / max(np.median(c[loud[:3]]), 1.0))) if len(loud) >= 6 else 0.0
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    tail = 20 * np.log10(np.sqrt(np.mean(m[-480:] ** 2)) + 1e-12) - pk
    return {'length_s': round(len(m) / dsp.SR, 3), 'peak_frac': round(ip / max(k, 1), 3),
            'sag': round(float((pk - sm[min(third, k - 1)]) / fall), 2), 'cent_oct': round(cent_oct, 2),
            'tail_db': round(float(tail), 1), 'side_db': round(side_db(x), 1), 'low_db': round(low_share_db(x), 1)}


def target_check(x, row=None):
    """critic `extra`: minus the distance (about 0..3) from the downlifter targets"""
    m = measure(x)
    pen = max(0.0, m['peak_frac'] - 0.15) * 6                   # the push comes first
    pen += max(0.0, -1.0 - (-m['cent_oct'])) if m['cent_oct'] > -1.0 else 0.0   # it must fall
    pen += max(0.0, 0.3 - m['sag']) * 3                          # decelerating
    pen += max(0.0, m['tail_db'] + 45.0) / 8                     # faded, not cut
    pen += max(0.0, m['side_db'] + 4.0) / 6
    return -min(3.0, pen)


# ------------------------------------------------------------------------------------------------ render
def render(p, r):
    p = {**DEFAULTS, **p}
    L = float(p['length'])
    n = dsp.n_of(L)
    t = dsp.t_of(n)
    u = t / L
    v = 1 - (1 - u) ** p['curve']                                # decelerating progress
    att = np.clip(t / (p['att_ms'] / 1000.0), 0, 1)
    att = np.sin(0.5 * np.pi * att) ** 2
    lv = att * 10 ** (-p['drop_db'] * v / 20)
    lv *= np.clip((1 - u) / 0.12, 0, 1) ** 1.5                  # the dry part reaches silence at the end

    # noise: resonant low-pass closing
    wn = (1 - p['pink']) * r.standard_normal(n) + p['pink'] * dsp.noise(L, r, 'pink')[:n] * 1.6
    fc = p['nf0'] * (p['nf1'] / p['nf0']) ** v
    nz = dsp.butter(dsp.svf(wn, fc, p['nq'], 'low'), 'high', p['hp'], order=2)
    nz = dsp.butter(nz, 'low', 17000, order=4)                   # nothing above 20 kHz
    if p['flange'] > 0.01:
        rate = p['fl_rate'] * (1 - 0.66 * v)
        ph = np.cumsum(rate) / dsp.SR
        d = 0.0004 + 0.0086 * (0.5 - 0.5 * np.cos(2 * np.pi * ph))
        nz = comb(nz, d, p['flange'])
    nz_st = fx.width(fx.decorrelate(nz, 1.0), p['width'])

    # tonal
    ph0 = r.uniform(0, 1, 5)
    if p['tmode'] == 'tape':
        chord = np.zeros((n + dsp.n_of(0.05), 2))
        for ratio, g in ((1.0, 1.0), (1.5, 0.6), (2.0, 0.5)):
            chord[:n] += g * unison(np.full(n, p['tf0'] * ratio), p['wave'], p['detune'], 0.5 * p['width'], ph0)[:n]
        sp = np.maximum((1 - u) ** p['tape_curve'], 0.0)
        pos = np.concatenate([[0.0], np.cumsum(sp[:-1])])
        tone = np.stack([np.interp(pos, np.arange(len(chord)), chord[:, c]) for c in range(2)], axis=1)
        ftrack = p['tf0'] * np.maximum(sp, 0.02)
    else:
        ftrack = p['tf0'] * 2 ** (-p['toct'] * v)
        tone = unison(ftrack, p['wave'], p['detune'], 0.5 * p['width'], ph0)
    if p['wave'] in ('saw', 'square'):
        cut = np.clip(ftrack * p['tbright'], 200, 18000)
        tone = np.stack([dsp.svf(tone[:, c], cut, 0.9, 'low') for c in range(2)], axis=1)
    tone = dsp.butter(tone, 'high', p['hp'], order=2)

    # sub: a falling sine with its own decay (dry, mono)
    fs = p['sub_f0'] * (p['sub_f1'] / p['sub_f0']) ** (1 - (1 - u) ** 2.2)
    sub = dsp.osc(fs, 'sine') * att * np.exp(-u / 0.22) * np.clip((1 - u) / 0.2, 0, 1) ** 2
    sub += 0.15 * np.tanh(3 * sub)                               # a few harmonics so it reads on small speakers
    sub = dsp.butter(sub, 'high', 25, order=2)

    # shepard fall
    if p['shep_db'] > -47:
        sh = dsp.shepard(L, r, up=False, rate=p['shep_rate'], f_lo=60.0, f_hi=9000.0, kind='tone')[:n]
        sh = dsp.svf(dsp.butter(sh, 'high', p['hp'], order=2), np.minimum(fc * 1.3, 18000), 0.7, 'low')  # darkens
        sh_st = fx.width(fx.decorrelate(sh, 0.6), 0.6 * p['width'])
    else:
        sh_st = None

    def at(db, x):
        return 10 ** (db / 20) * x / lvl(x)

    mix = at(p['noise_db'], nz_st) + at(p['tone_db'], tone)
    if sh_st is not None:
        mix = mix + at(p['shep_db'], sh_st)
    dry = mix * lv[:, None]
    subl = at(p['sub_db'], stereo_of(sub)) if p['sub_db'] > -47 else 0.0
    tail = dsp.n_of(min(1.5, 0.6 * p['rt60']))
    out = np.zeros((n + tail, 2))
    out[:n] = dry + subl
    if p['verb'] > 0.005:
        wet = wet_of(dry, p['rt60'], n + tail, size=0.8, predelay=0.015, lowcut=150.0)
        out += p['verb'] * wet / lvl(wet) * lvl(dry)
    if p['sat'] > 0.1:
        out = fx.saturate(out, drive_db=p['sat'], mode='tanh')
    out = dsp.butter(out, 'high', 25, order=2)
    out = dsp.trim_tail(out, -62.0, 0.02)
    return dsp.fade(out, 0.0015, min(0.12, 0.25 * len(out) / dsp.SR))
