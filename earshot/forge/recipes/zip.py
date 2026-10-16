"""A designed tonal zip for motion-design edits: one fast exponential pitch sweep (up, down or an arc), 0.06-0.3 s.

Every sample is computed; no recorded audio is loaded, sampled or convolved. Built to the pro-film measurements in
docs/motion-sfx-study.md (zip, 4 events in two films):
  * The gesture is a glide that is a straight line on a log-frequency axis: f(t) = f0 * (f1 / f0)^(t / T), i.e.
    constant octaves per second (pro: about +20 oct/s for 100 Hz -> 1.5 kHz in 0.2 s, +30 oct/s for 120 Hz -> 10 kHz
    in 0.22-0.25 s, -60 oct/s for 7.2 kHz -> 420 Hz in about 60 ms, and an arc 1 -> 3 -> 1 kHz over 0.25 s). `bend`
    warps time a little (> 0 accelerates into the end, < 0 arrives early) without leaving the exponential family.
  * Timbre: a sine glide plus a few harmonics (h2, h3, h5 levels; each partial fades out before it would pass 18 kHz,
    so nothing aliases), an optional FM sheen (index fm, ratio 1 or 2) and a thin air layer (noise in a band that
    follows the pitch, Q 3-8) for the 'zip' texture. A 3-voice unison (detune cents) thickens it, mostly mono.
  * Envelope: a raised-cosine attack att_ms (pro: 20-35 ms), a level tilt across the sweep (tilt_db), and a short
    release rel_ms: the sweep ends on the cut (hard end, no ring-out) apart from an optional tiny room (verb).
  * Light saturation glues the partials; stereo stays near mono (side 15-25 dB under mid).
CHARACTER parameters: direction (shape), range (f0, f1 / the arc's peak), speed (length), brightness (harmonics, fm,
air), weight (h2 / sub octave), attack and end. FAMILIES are presets after the four measured pro zips plus one.

The pro_zip refset has no forge refs (the four pro zips are all bench, held out), so the forge runs in designed mode
and `target_check()` adds the study targets as the critic's `extra` (sweep rate 15-70 oct/s on the ridge, length,
attack, near mono).
"""
import numpy as np

from .. import dsp, fx
from .riser import env_db, lvl, side_db, stereo_of, wet_of

TARGET = 'a quick tonal zip, a fast synth pitch sweep'
ALTS = ['a bird chirp', 'a squeaky door hinge', 'a car horn', 'a phone ringing']
REFSET = 'pro_zip'

SPACE = {
    'shape': ['up', 'down', 'arc'],
    'length': (0.06, 0.3),
    'f0': (80.0, 8000.0, 'log'),          # start pitch (Hz)
    'f1': (80.0, 12000.0, 'log'),         # end pitch (up / down); the arc returns to f0 * arc_back
    'fpk': (500.0, 6000.0, 'log'),        # the arc's peak
    'pk_at': (0.3, 0.7),                  # where the arc peaks (fraction)
    'arc_back': (0.5, 1.0),               # the arc ends at f0 * arc_back
    'bend': (-0.5, 0.6),                  # time warp of the sweep (power 2^bend)
    'h2_db': (-40.0, -6.0),
    'h3_db': (-40.0, -6.0),
    'h5_db': (-48.0, -12.0),
    'sub_db': (-48.0, -6.0),              # an octave below (weight for low zips)
    'fm': (0.0, 1.5),                     # FM index (sheen)
    'fm_ratio': [1.0, 2.0],
    'saw_db': (-48.0, 0.0),               # round 2: a band-limited saw under a pitch-tracking low-pass (synth body)
    'saw_lp': (1.5, 6.0, 'log'),          #   its low-pass = pitch x saw_lp
    'air_db': (-40.0, -10.0),             # noise band following the pitch
    'air_q': (2.0, 8.0, 'log'),
    'detune': (0.0, 15.0),
    'att_ms': (3.0, 40.0, 'log'),
    'rel_ms': (4.0, 40.0, 'log'),
    'tilt_db': (-9.0, 9.0),               # level change across the sweep
    'verb': (0.0, 0.3),
    'width': (0.0, 0.35),
    'sat': (0.0, 9.0),
}

FAMILIES = {
    # ref_6 1.60: up, about 100 Hz -> 1.5 kHz in 0.2 s (+19.5 oct/s), starting just before a cut
    'up': {'shape': 'up', 'length': 0.19, 'f0': 100.0, 'f1': 1600.0, 'fpk': 2000.0, 'pk_at': 0.5, 'arc_back': 1.0,
           'bend': 0.05, 'h2_db': -14.0, 'h3_db': -18.0, 'h5_db': -30.0, 'sub_db': -48.0, 'fm': 0.3,
           'fm_ratio': 1.0, 'air_db': -28.0, 'air_q': 4.0, 'detune': 6.0, 'att_ms': 28.0, 'rel_ms': 10.0,
           'tilt_db': 3.0, 'verb': 0.06, 'width': 0.12, 'sat': 3.0},
    # ref_6 3.78: up, about 130 Hz -> 10 kHz in 0.24 s (+26 oct/s), ending on the cut
    'up_hi': {'shape': 'up', 'length': 0.24, 'f0': 130.0, 'f1': 10000.0, 'fpk': 2000.0, 'pk_at': 0.5,
              'arc_back': 1.0, 'bend': 0.25, 'h2_db': -16.0, 'h3_db': -24.0, 'h5_db': -36.0, 'sub_db': -48.0,
              'fm': 0.15, 'fm_ratio': 2.0, 'air_db': -24.0, 'air_q': 5.0, 'detune': 4.0, 'att_ms': 30.0,
              'rel_ms': 6.0, 'tilt_db': -2.0, 'verb': 0.05, 'width': 0.1, 'sat': 2.0},
    # ref_5 22.80: down, 7.2 kHz -> 420 Hz in about 70 ms (-58 oct/s), across a cut
    'down': {'shape': 'down', 'length': 0.075, 'f0': 7200.0, 'f1': 420.0, 'fpk': 2000.0, 'pk_at': 0.5,
             'arc_back': 1.0, 'bend': -0.2, 'h2_db': -18.0, 'h3_db': -26.0, 'h5_db': -40.0, 'sub_db': -48.0,
             'fm': 0.2, 'fm_ratio': 1.0, 'air_db': -26.0, 'air_q': 4.0, 'detune': 3.0, 'att_ms': 6.0,
             'rel_ms': 12.0, 'tilt_db': 2.0, 'verb': 0.08, 'width': 0.1, 'sat': 2.0},
    # ref_5 28.22: an arc about 1 -> 3 -> 1 kHz over 0.25 s
    'arc': {'shape': 'arc', 'length': 0.25, 'f0': 1000.0, 'f1': 1000.0, 'fpk': 3000.0, 'pk_at': 0.45,
            'arc_back': 0.9, 'bend': 0.0, 'h2_db': -16.0, 'h3_db': -22.0, 'h5_db': -36.0, 'sub_db': -48.0,
            'fm': 0.25, 'fm_ratio': 1.0, 'air_db': -30.0, 'air_q': 5.0, 'detune': 5.0, 'att_ms': 22.0,
            'rel_ms': 18.0, 'tilt_db': 0.0, 'verb': 0.07, 'width': 0.12, 'sat': 2.0},
    # a longer warm down-zip: 2.4 kHz -> 160 Hz in 0.2 s (-19.5 oct/s), heavier (sub octave, more h2)
    'down_warm': {'shape': 'down', 'length': 0.2, 'f0': 2400.0, 'f1': 160.0, 'fpk': 2000.0, 'pk_at': 0.5,
                  'arc_back': 1.0, 'bend': -0.3, 'h2_db': -10.0, 'h3_db': -16.0, 'h5_db': -28.0, 'sub_db': -20.0,
                  'fm': 0.5, 'fm_ratio': 1.0, 'air_db': -32.0, 'air_q': 3.0, 'detune': 8.0, 'att_ms': 10.0,
                  'rel_ms': 22.0, 'tilt_db': -4.0, 'verb': 0.1, 'width': 0.15, 'sat': 4.0},
}
# round 2: the critic heard the sine glides as 'an 8-bit sound effect / a cheap synthesized beep' (CLAP), and the
# flat box envelope is the synthetic part: each family gets a filtered saw body and a level contour that follows
# the gesture (up: swells into the cut, down: falls away)
for _k, _v in FAMILIES.items():
    _v.setdefault('saw_db', -48.0)
    _v.setdefault('saw_lp', 3.0)
FAMILIES['up'].update({'saw_db': -6.0, 'saw_lp': 3.0, 'tilt_db': 6.0, 'h3_db': -24.0})
FAMILIES['up_hi'].update({'saw_db': -10.0, 'saw_lp': 2.5, 'tilt_db': 4.0, 'detune': 7.0})
FAMILIES['down'].update({'saw_db': -8.0, 'saw_lp': 3.5, 'tilt_db': -6.0})
FAMILIES['arc'].update({'saw_db': -14.0, 'saw_lp': 2.0, 'tilt_db': -3.0})
FAMILIES['down_warm'].update({'saw_db': -2.0, 'saw_lp': 2.2, 'tilt_db': -8.0})
DEFAULTS = dict(FAMILIES['up'])


def trajectory(p, n):
    """per-sample pitch (Hz) of the sweep"""
    u = np.linspace(0, 1, n)
    uw = u ** (2.0 ** p['bend'])
    if p['shape'] == 'arc':
        a = p['pk_at']
        up = np.clip(u / a, 0, 1)
        dn = np.clip((u - a) / (1 - a), 0, 1)
        lf = np.where(u <= a, np.log2(p['f0']) + (np.log2(p['fpk']) - np.log2(p['f0'])) * np.sin(0.5 * np.pi * up),
                      np.log2(p['fpk']) + (np.log2(p['f0'] * p['arc_back']) - np.log2(p['fpk']))
                      * (1 - np.cos(0.5 * np.pi * dn)))
        return 2 ** lf
    f1 = p['f1']
    f0 = p['f0']
    if p['shape'] == 'up' and f1 < f0 * 1.5:
        f1 = f0 * 4
    if p['shape'] == 'down' and f1 > f0 / 1.5:
        f1 = f0 / 4
    return f0 * (f1 / f0) ** uw


# ------------------------------------------------------------------------------------------------ measurement
def ridge(x, hop_ms=2.5, win=1024):
    """dominant spectral peak (Hz) per hop and the frame level (dB)"""
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    h = int(hop_ms * dsp.SR / 1000)
    m = np.pad(m, (win // 2, win // 2))
    k = max(1, (len(m) - win) // h)
    idx = np.arange(win)[None, :] + h * np.arange(k)[:, None]
    fr = m[idx] * np.hanning(win)
    S = np.abs(np.fft.rfft(fr, axis=1))
    f = np.fft.rfftfreq(win, 1 / dsp.SR)
    j = np.argmax(S[:, 2:], axis=1) + 2
    lv = 20 * np.log10(S.max(axis=1) + 1e-12)
    return f[j], lv - lv.max(), hop_ms / 1000.0


def measure(x):
    """the zip targets, model-free: length_s, active_s (within 30 dB of the peak), attack_ms (start to the loudest 5 ms), oct_span (octaves the ridge
    covers within 20 dB of the peak), oct_per_s (least-squares slope of log2 ridge vs time over those frames; for an
    arc, the steepest half), end_db (the last 5 ms re the peak), side_db"""
    f, lv, hop = ridge(x)
    act = np.nonzero(lv > -20)[0]
    e = env_db(x, 5.0)
    act_e = np.nonzero(e > e.max() - 30)[0]
    out = {'length_s': round(len(x) / dsp.SR, 3), 'attack_ms': round(float(np.argmax(e) * 5.0), 1),
           'active_s': round(float((act_e[-1] - act_e[0] + 1) * 0.005), 3),
           'end_db': round(float(e[-1] - e.max()), 1), 'side_db': round(side_db(x), 1)}
    if len(act) >= 4:
        lf = np.log2(f[act])
        tt = act * hop
        out['oct_span'] = round(float(lf.max() - lf.min()), 2)
        s_all = np.polyfit(tt, lf, 1)[0]
        h = len(act) // 2
        if h >= 3:
            s1, s2 = np.polyfit(tt[:h], lf[:h], 1)[0], np.polyfit(tt[h:], lf[h:], 1)[0]
            if np.sign(s1) != np.sign(s2) and min(abs(s1), abs(s2)) > 3:
                s_all = s1 if abs(s1) > abs(s2) else s2
                out['arc'] = True
        out['oct_per_s'] = round(float(s_all), 1)
    else:
        out['oct_span'], out['oct_per_s'] = 0.0, 0.0
    out.setdefault('arc', False)
    return out


def target_check(x, row=None):
    """critic `extra`: minus the distance (about 0..3) from the study's zip targets"""
    m = measure(x)
    r = abs(m['oct_per_s'])
    lo, span = (8.0, 1.0) if m['arc'] else (12.0, 1.5)            # an arc (1 -> 3 -> 1 kHz) covers ~1.6 oct each way
    pen = max(0.0, lo - r) / 6 + max(0.0, r - 75.0) / 20          # 15-70 oct/s
    pen += max(0.0, span - m['oct_span']) * 1.5                   # a clear gesture
    pen += max(0.0, m['active_s'] - 0.33) * 10                    # short (within 30 dB of the peak; a room tail is fine)
    pen += max(0.0, m['side_db'] + 10.0) / 6                      # near mono
    return -min(3.0, pen)


# ------------------------------------------------------------------------------------------------ render
def render(p, r):
    p = {**DEFAULTS, **p}
    L = float(p['length'])
    n = dsp.n_of(L)
    f = trajectory(p, n)
    out = np.zeros((n, 2))
    for k, (cents, pan) in enumerate(((0.0, 0.0), (-p['detune'], -1.0), (p['detune'], 1.0))):
        fk = f * 2 ** (cents / 1200.0)
        ph = 2 * np.pi * np.cumsum(fk) / dsp.SR + r.uniform(0, 2 * np.pi)
        fmi = p['fm'] * np.clip((14000 - fk * (1 + 2 * p['fm_ratio'])) / 4000, 0, 1)   # sidebands stay < 20 kHz
        mod = fmi * np.sin(ph * p['fm_ratio']) if p['fm'] > 0.01 else 0.0
        v = np.sin(ph + mod)
        for h, db in ((2, p['h2_db']), (3, p['h3_db']), (5, p['h5_db'])):
            g = 10 ** (db / 20) * np.clip((18000 - h * fk) / 3000, 0, 1)   # fade before 18 kHz: no aliasing
            v += g * np.sin(h * ph + h * mod * 0.5)
        if p['sub_db'] > -47:
            v += 10 ** (p['sub_db'] / 20) * np.sin(0.5 * ph)
        gain = 1.0 if k == 0 else 0.5 * min(1.0, p['detune'] / 4.0)
        th = (np.clip(pan * p['width'], -1, 1) + 1) * np.pi / 4
        out[:, 0] += gain * v * np.cos(th) * np.sqrt(2)
        out[:, 1] += gain * v * np.sin(th) * np.sqrt(2)
    if p['saw_db'] > -47:                                        # a filtered saw body (the analog-synth zip)
        sw = sum(dsp.osc(f * 2 ** (c / 1200.0), 'saw', phase=r.uniform()) for c in (-p['detune'], 0.0, p['detune']))
        sw = dsp.svf(sw, np.clip(f * p['saw_lp'], 150, 16000), 1.1, 'low')
        sw = dsp.svf(sw, np.clip(f * p['saw_lp'] * 1.05, 150, 16000), 0.7, 'low')   # 24 dB/oct
        sw = dsp.butter(sw, 'low', 15000, order=6)                # the saw's upper partials never pass 20 kHz
        out = out / lvl(out) + 10 ** (p['saw_db'] / 20) * stereo_of(sw) / lvl(sw)
    tone = out / lvl(out)

    # air: noise in a band following the pitch
    nz = r.standard_normal(n)
    air = dsp.svf(nz, np.clip(f * 1.5, 200, 16000), p['air_q'], 'bandn')
    air = stereo_of(air) / lvl(air)
    y = tone + 10 ** (p['air_db'] / 20) * air

    # envelope: raised-cosine attack, level tilt, short release (the zip ends on the cut)
    t = dsp.t_of(n)
    a = np.clip(t / (p['att_ms'] / 1000.0), 0, 1)
    rl = np.clip((L - t) / (p['rel_ms'] / 1000.0), 0, 1)
    env = (0.5 - 0.5 * np.cos(np.pi * a)) * (0.5 - 0.5 * np.cos(np.pi * rl)) * 10 ** (p['tilt_db'] * (t / L) / 20)
    y = y * env[:, None]
    y = dsp.butter(y, 'high', 40, order=2)
    if p['sat'] > 0.1:
        y = fx.saturate(y, drive_db=p['sat'], mode='tanh')
        y = dsp.butter(y, 'low', 18000, order=4)                  # the drive's 3rd harmonic of a 7-10 kHz tone
    if p['verb'] > 0.005:
        tail = dsp.n_of(0.3)
        wet = wet_of(y, 0.3, n + tail, size=0.3, predelay=0.004, lowcut=300.0)
        y = np.vstack([y, np.zeros((tail, 2))]) + p['verb'] * wet / lvl(wet) * lvl(y)
        y = dsp.trim_tail(y, -60.0, 0.004)
        y = dsp.fade(y, 0.0, min(0.02, 0.1 * len(y) / dsp.SR))
    y = dsp.fade(y, 0.0005, 0.002)
    w = np.hanning(len(y))                                       # round 3: zero mean without touching the edges (a
    return y - (y.mean(axis=0) / w.mean())[None, :] * w[:, None]  # short low saw left DC -24..-29 dB re its peak)
