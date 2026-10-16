"""A designed UI click for motion-design launch videos (v2): one clean gesture, bright, 0-3 ms attack, a hard end.

What a pro UI click is (docs/motion-sfx-study.md, measured in the reference films; docs/motion-sfx-techniques.md 6):
  attack 0-3 ms, t20 6-16 ms, 30-60 ms in all; centroid 5-10 kHz, tilt +4..+12 dB/oct; tonal partials at 5-8 kHz are
  allowed (a "tink" rather than a noise snap); mono; the over-music "mouse" variant has a low body (centroid ~430 Hz,
  t40 ~37 ms). The real close-up clicks of the 'click' refset are broader (centroid 1-5 kHz, t40 33-107 ms).
So v2 is a DESIGNED sound, not the v1 physical model (v1 lives on as demo/forge-test history): four layers, each owning
a band, peaking together, then the universal chain (band, saturate lightly, shape, tiny space, finish):
  1. transient: a 0.15-1.5 ms noise burst, band-passed around `pitch` (Farnell: squared decay = a crisp snap)
  2. tink: 1-3 damped partials at `pitch` x (1, inharm, ...): the tonal "tink" of premium UI clicks (`tone` sets it
     against the noise snap)
  3. body: a short damped sine at `body_f` (150-1200 Hz) with a small pitch drop: weight without mud (`body`)
  4. grit: a short band-passed noise tail (the plastic texture), and an optional quiet release click 25-90 ms later
then a high-pass (`hp`), a de-harsh dip at 2.5-4 kHz (the touchy band), light tanh saturation, a tiny dark early-
reflection space (and for the real-click families a small dark room tail: `verb`, `rt`), an end gate at `length`
(hold, then a smooth release: the pro "hard end" without a chopped tail).

CHARACTER parameters (what a family varies): pitch, tone, body / body_f, tink_ms (length of the ring), length,
bright (transient band), rel (press + release). FAMILIES below are the palette: every category's finals come from one
patch with only these moved, so a set sounds like one product. Other recipes (toggle, hover, popin) import the layer
helpers from here. Every sample is computed; no recorded audio is used.
"""
import numpy as np

from .. import dsp, fx
from . import REF_ROOT

TARGET = 'a clean bright user interface click'
ALTS = ['a coin dropped on a table', 'a mechanical keyboard key press', 'a camera shutter', 'a woodblock hit']
TARGETS = {'click': 'a clean bright user interface click',
           'pro_click': 'a crisp high digital interface tick for a product video'}
REFSET = 'click'   # the CLI's default refs: this refsets category, 'forge' split (bench held out)
_K = REF_ROOT + '/ui/'
REFS = [_K + f for f in ('ui_click_01.wav', 'ui_click_02.wav', 'ui_click_03.wav', 'ui_click_04.wav')]

SPACE = {
    'pitch': (1200.0, 9500.0, 'log'),    # centre of the tink partials and the transient band (real clicks: 1.5-4.5k)
    'bright': (0.0, 1.0),                # transient band: 0 = an octave around pitch, 1 = up to 16 kHz
    'snap_ms': (0.15, 1.5, 'log'),       # transient burst length
    'tone': (0.0, 1.0),                  # tink partials vs noise snap
    'partials': (1, 3, 'int'),
    'inharm': (1.25, 2.8),               # second partial ratio (third = inharm ** 1.6)
    'tink_ms': (1.5, 25.0, 'log'),       # partial decay time constant
    'body': (0.005, 0.8, 'log'),         # body level re the transient
    'body_f': (150.0, 1200.0, 'log'),
    'body_ms': (1.5, 12.0, 'log'),       # <= 12 ms: a 20 ms body rang as a narrow 250-500 Hz line (click round 1)
    'body_drop': (0.0, 0.5),             # body pitch drop (fraction, over its decay)
    'grit': (0.0, 0.6),                  # plastic texture tail level
    'grit_ms': (2.0, 30.0, 'log'),
    'rel': (0.0, 0.9),                   # second transient level (0 = none): the pro double tick at 8-16 ms, or a
    'rel_dt': (0.008, 0.09),             # quiet release click 25-90 ms later (real buttons: -25..-30 dB)
    'room': (0.0, 0.3),                  # tiny early-reflection space, wet
    'verb': (0.0, 0.35),                 # a small dark room tail (the real close-up clicks fall to -20 dB in ~20 ms,
    'rt': (0.04, 0.35, 'log'),           # then hold a room tail to -40 dB at 60-75 ms); rt = its RT60 s
    'hp': (80.0, 2500.0, 'log'),         # > 800 Hz: a steeper 4th-order high-pass (pro clicks: empty below 2 kHz)
    'lp': (7000.0, 19000.0, 'log'),      # top roll-off (real clicks: > 12 kHz at -29 dB of total; pro: 18-20 kHz)
    'deharsh': (0.0, 8.0),               # dB dip at deharsh_f
    'deharsh_f': (2400.0, 4200.0, 'log'),
    'sat': (0.0, 9.0),                   # tanh drive dB
    'attack_ms': (0.05, 2.5, 'log'),     # onset ramp
    'length': (0.025, 0.14),             # end gate (s from the onset)
    'release_ms': (2.0, 30.0, 'log'),    # gate release
    'width': (0.0, 0.25),                # decorrelated space / grit only (objects stay mono)
}
DEFAULTS = {'pitch': 6000.0, 'bright': 0.5, 'snap_ms': 0.4, 'tone': 0.6, 'partials': 2, 'inharm': 1.62,
            'tink_ms': 5.0, 'body': 0.08, 'body_f': 600.0, 'body_ms': 5.0, 'body_drop': 0.15, 'grit': 0.15,
            'grit_ms': 6.0, 'rel': 0.0, 'rel_dt': 0.05, 'room': 0.06, 'verb': 0.0, 'rt': 0.12, 'hp': 200.0,
            'lp': 17000.0, 'deharsh': 3.0, 'deharsh_f': 3200.0, 'sat': 2.0, 'attack_ms': 0.3, 'length': 0.06,
            'release_ms': 10.0, 'width': 0.05}

# The palette: per category, named families (character presets). A value fixes the parameter for the family; a
# (lo, hi[, 'log']) tuple narrows its range; parameters not named keep SPACE (fine detail the search tunes).
FAMILIES = {
    'pro_click': {   # ref_2 (forge): two short bright bursts ~12 ms apart, 3-15 kHz, gated at ~25 ms. Round 3: the other
        # films' clicks carry a body (the study's ref_5 'mouse' click: centroid ~430 Hz, t20 11, t40 37 ms; the round-2
        # judge: 150 Hz-1 kHz -41 dB of total made vs -6 in the held-out film clicks), so every family keeps a body
        # (lower high-pass, body 400-1000 Hz) and 'mouse' is the study's body click; 'glass' stays the bright ref_2 idiom
        'tink': {'verb': 0.0, 'pitch': (5000, 8000, 'log'), 'tone': (0.4, 0.8), 'partials': (2, 3),
                 'tink_ms': (1.5, 4, 'log'), 'rel': (0.35, 0.8), 'rel_dt': (0.009, 0.015),
                 'body': (0.06, 0.18, 'log'), 'body_f': (600, 1000, 'log'), 'body_ms': (2, 5, 'log'),
                 'hp': (350, 800, 'log'), 'length': (0.025, 0.04), 'release_ms': (2, 8, 'log')},
        'snap': {'verb': 0.0, 'hp': (250, 600, 'log'), 'pitch': (4000, 7000, 'log'), 'tone': (0.0, 0.3),
                 'tink_ms': (1.5, 3, 'log'),
                 'bright': (0.5, 1.0), 'snap_ms': (0.3, 1.2, 'log'), 'rel': (0.3, 0.7), 'rel_dt': (0.009, 0.016),
                 'body': (0.12, 0.3, 'log'), 'body_f': (500, 900, 'log'), 'body_ms': (2, 5, 'log'),
                 'length': (0.025, 0.045), 'release_ms': (2, 8, 'log')},
        'mouse': {'verb': 0.0, 'pitch': (2500, 4500, 'log'), 'tone': (0.1, 0.4), 'tink_ms': (1.5, 3, 'log'),
                  'body': (0.35, 0.8, 'log'), 'body_f': (300, 600, 'log'), 'body_ms': (3, 7, 'log'),
                  'hp': (100, 200, 'log'), 'attack_ms': (0.3, 1.5, 'log'), 'rel': 0.0, 'deharsh': (0.0, 3.0),
                  'lp': (11000, 16000, 'log'), 'length': (0.04, 0.06), 'release_ms': (4, 12, 'log')},
        'glass': {'verb': 0.0, 'pitch': (7000, 9500, 'log'), 'tone': (0.6, 0.9), 'partials': 3,
                  'tink_ms': (2, 5, 'log'), 'rel': 0.0, 'body': (0.03, 0.1, 'log'), 'body_f': (700, 1200, 'log'),
                  'body_ms': (2, 4, 'log'), 'hp': (500, 1000, 'log'), 'length': (0.03, 0.06),
                  'release_ms': (4, 12, 'log')},
        'soft': {'verb': 0.0, 'hp': (150, 350, 'log'), 'pitch': (3500, 5500, 'log'), 'tone': (0.3, 0.7),
                 'tink_ms': (1.5, 4, 'log'), 'rel': 0.0, 'body': (0.2, 0.5, 'log'), 'body_f': (400, 900, 'log'),
                 'body_ms': (2, 6, 'log'), 'deharsh': (2, 6), 'length': (0.035, 0.06), 'release_ms': (4, 12, 'log')},
    },
    'click': {   # round 3: the real clicks' balance is 1-4 kHz -3.7 / 4-12 kHz -4.7 / > 12 kHz -29 dB of total (forge
        # refs, median): pitch 1.5-4.5 kHz, no de-harsh dip in the 1-4 kHz band, a top roll-off at 5-17 kHz, short tink
        # rings (<= 3 ms), little tink (a strong one made a narrow -2 dB 1/3-octave peak), short bodies (<= 6 ms) so no
        # narrow line rings through the click. Spread like the refs (1-4 kHz -2..-20, 150 Hz-1 kHz -5..-44 dB): 'thock'
        # is the dark, bodied one (kit_click_01), 'tink' the bright, thin one (kit_click_03); soft / snap / double sit between
        'soft': {'pitch': (1800, 3200, 'log'), 'tone': (0.05, 0.3), 'tink_ms': (1.5, 3, 'log'), 'bright': (0.1, 0.5),
                 'body': (0.1, 0.35, 'log'), 'body_f': (300, 700, 'log'), 'body_ms': (2, 5, 'log'), 'rel': 0.0,
                 'deharsh': (0.0, 2.0), 'grit': (0.15, 0.45), 'verb': (0.08, 0.3), 'rt': (0.08, 0.25, 'log'),
                 'lp': (8000, 12000, 'log'), 'length': (0.08, 0.12)},
        'thock': {'pitch': (1200, 2200, 'log'), 'tone': (0.0, 0.2), 'tink_ms': (1.5, 3, 'log'), 'bright': (0.0, 0.3),
                  'body': (0.5, 0.8, 'log'), 'body_f': (120, 260, 'log'), 'body_ms': (3, 6, 'log'), 'rel': (0.0, 0.08),
                  'rel_dt': (0.025, 0.04), 'hp': (50, 100, 'log'), 'deharsh': (0.0, 2.0), 'grit': (0.1, 0.35),
                  'verb': (0.05, 0.25), 'rt': (0.08, 0.25, 'log'), 'lp': (5000, 8000, 'log'), 'length': (0.09, 0.13)},
        'tink': {'pitch': (3800, 6000, 'log'), 'tone': (0.25, 0.5), 'partials': 2, 'tink_ms': (1.5, 3, 'log'),
                 'bright': (0.4, 0.8), 'body': (0.01, 0.05, 'log'), 'body_ms': (1.5, 3, 'log'), 'rel': 0.0,
                 'hp': (400, 900, 'log'), 'deharsh': (0.0, 2.0), 'grit': (0.1, 0.4), 'verb': (0.05, 0.25),
                 'lp': (13000, 17000, 'log'), 'length': (0.06, 0.1)},
        'snap': {'pitch': (2200, 4000, 'log'), 'tone': (0.0, 0.2), 'bright': (0.3, 0.7), 'grit': (0.2, 0.5),
                 'rel': (0.2, 0.6), 'rel_dt': (0.008, 0.016), 'body': (0.03, 0.2, 'log'), 'body_ms': (1.5, 5, 'log'),
                 'deharsh': (0.0, 2.0), 'verb': (0.05, 0.3), 'lp': (10000, 15000, 'log'), 'length': (0.06, 0.1)},
        'double': {'pitch': (2000, 3800, 'log'), 'tone': (0.05, 0.3), 'tink_ms': (1.5, 3, 'log'), 'rel': (0.05, 0.15),
                   'rel_dt': (0.03, 0.07), 'body': (0.05, 0.3, 'log'), 'body_ms': (2, 6, 'log'),
                   'deharsh': (0.0, 2.0), 'grit': (0.1, 0.4), 'verb': (0.05, 0.25), 'lp': (9000, 13000, 'log'),
                   'length': (0.09, 0.13)},
    },
}


# ------------------------------------------------------------------------------------------------ shared UI layers
def burst(seconds, r, f_lo, f_hi, att_s=0.0003, order=2):
    """a band-passed noise burst with a 'squared' decay (Farnell's click envelope: crisp, no long hiss), unit peak"""
    n = dsp.n_of(max(seconds * 6, 0.004))
    t = dsp.t_of(n)
    e = np.exp(-t / max(seconds, 5e-5)) ** 2
    a = max(2, dsp.n_of(att_s))
    e[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    lo, hi = max(40.0, f_lo), min(21500.0, f_hi)
    y = dsp.butter(r.standard_normal(n), 'band', (lo, max(lo * 1.05, hi)), order=order) * e
    return y / (np.abs(y).max() + 1e-12)


def partials(f0, ratios, amps, tau, seconds, r, glide=0.0, att_s=0.0002):
    """damped partials (a small hard body ringing), higher ones die faster; unit peak; soft onset ramp"""
    ratios = np.asarray(ratios, float)
    fs = f0 * ratios
    dec = tau * ratios ** -0.6
    y = dsp.modal(fs, dec, amps, seconds, r, jitter=0.004, glide=glide)
    a = max(2, dsp.n_of(att_s))
    y[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    return y / (np.abs(y).max() + 1e-12)


def thump(f, seconds, att_s, hold_s, rel_s, drop=0.1, harm=0.1, r=None):
    """a designed low thump: a sine at f whose pitch falls from f * (1 + drop), raised-cosine attack, a hold, then an
    exponential release (convex: holds, then stops); harm adds a little 2nd harmonic (small speakers); unit peak"""
    n = dsp.n_of(seconds)
    t = dsp.t_of(n)
    ph0 = r.uniform(0, 2 * np.pi) if r is not None else 0.0
    fi = f * (1 + drop * np.exp(-t / max(att_s + hold_s * 0.5, 2e-3)))
    ph = 2 * np.pi * np.cumsum(fi) / dsp.SR + ph0
    y = np.sin(ph) + harm * np.sin(2 * ph + 0.3)
    a = max(2, dsp.n_of(att_s))
    e = np.ones(n)
    e[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    h = a + dsp.n_of(hold_s)
    if h < n:
        e[h:] = np.exp(-(t[h:] - t[h]) / max(rel_s, 1e-3))
    y = y * e
    return y / (np.abs(y).max() + 1e-12)


def gate(x, length, release, start=0.0):
    """the pro 'hard end': unity until start + length - release, then a raised-cosine release to zero at length;
    the file ends there (no chopped tail: the release is smooth)"""
    n = len(x)
    end = min(n, dsp.n_of(start + length))
    rl = max(8, min(end - 1, dsp.n_of(release)))
    g = np.zeros(n)
    g[:end - rl] = 1.0
    g[end - rl:end] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, rl))
    y = x * (g if x.ndim == 1 else g[:, None])
    return y[:end]


def finish(x, hp=120.0, deharsh=0.0, deharsh_f=3200.0, sat=0.0, lp=None):
    """band + colour: high-pass, a de-harsh dip, light tanh saturation (rms-matched), optional low-pass"""
    y = dsp.butter(x, 'high', hp, order=4 if hp > 800 else 2)
    if deharsh > 0.05:
        y = dsp.peq(y, deharsh_f, -deharsh, q=1.2)
    if lp:
        y = dsp.butter(y, 'low', lp, order=2)
    if sat > 0.05:
        y = fx.saturate(y, drive_db=sat, mode='tanh', level='peak')
    return y


def space(x, r, wet, width=0.0, size=0.12, damp=0.7):
    """a tiny dark early-reflection space; the direct sound stays mono, only the wet part is (optionally) wide.
    Returns stereo (n, 2), same length as x"""
    n = len(x)
    if wet <= 0.002:
        return np.stack([x, x], 1)
    er = fx.early(x, r, size=size, taps=14, damp=damp, width=1.0)[:n]
    er = dsp.butter(er, 'low', 7000, order=1)
    m = er.mean(1, keepdims=True)
    er = m + width * 4 * (er - m)
    er = er / (np.sqrt(np.mean(er ** 2)) + 1e-12) * np.sqrt(np.mean(x ** 2)) * wet
    return np.stack([x, x], 1) + er


def click_layers(p, r, f_scale=1.0, level=1.0, seconds=0.16, send=False):
    """transient + tink + body + grit of one click (mono, unit-ish peak x level), onset at sample 0.
    send=True also returns the broadband part alone (transient + grit, same scale): the bus a room tail is fed from,
    so the tink partials and the body never ring on in the room (round 3: a room fed with everything drew narrow
    6 kHz / 250-500 Hz lines through the whole click)"""
    pitch = p['pitch'] * f_scale * np.exp(0.01 * r.standard_normal())
    n = dsp.n_of(seconds)
    x = np.zeros(n)
    s = np.zeros(n)
    hi = pitch * (2 ** (0.5 + 1.6 * p['bright']))
    lo = pitch / 2 ** 0.6
    tr = burst(p['snap_ms'] * 1e-3, r, lo, hi, att_s=p['attack_ms'] * 1e-3)
    dsp.place(x, (1.0 - 0.5 * p['tone']) * tr, 0.0)
    dsp.place(s, (1.0 - 0.5 * p['tone']) * tr, 0.0)
    k = int(p['partials'])
    ratios = [1.0, p['inharm'], p['inharm'] ** 1.6][:k]
    amps = [1.0, 0.55, 0.3][:k]
    tk = partials(pitch, ratios, amps, p['tink_ms'] * 1e-3, seconds, r, glide=0.004,
                  att_s=max(0.0001, p['attack_ms'] * 1e-3))
    dsp.place(x, 1.4 * p['tone'] * tk, 0.0)
    if p['body'] > 0.004:
        bd = thump(p['body_f'] * f_scale ** 0.5, seconds, max(0.0004, p['attack_ms'] * 1e-3), 0.0,
                   p['body_ms'] * 1e-3, drop=p['body_drop'], harm=0.15, r=r)
        dsp.place(x, p['body'] * 1.5 * bd, 0.0)
    if p['grit'] > 0.005:
        g = burst(p['grit_ms'] * 1e-3, r, pitch * 0.5, min(16000, pitch * 2.5), att_s=0.0008, order=1)
        g = g * np.sqrt(np.exp(-dsp.t_of(len(g)) / (p['grit_ms'] * 1e-3)))   # softer than squared: a short texture
        dsp.place(x, 0.35 * p['grit'] * g / (np.abs(g).max() + 1e-12), 0.0004)
        dsp.place(s, 0.35 * p['grit'] * g / (np.abs(g).max() + 1e-12), 0.0004)
    k = level / (np.abs(x).max() + 1e-12)
    return (k * x, k * s) if send else k * x


def room_tail(st, snd, r, p, size=0.12, lowcut=300.0):
    """the small dark room after a click, fed from the broadband bus `snd` (transient + grit) only, added to the dry
    stereo `st` at the level a full-signal send of wet `verb` would have had (round 3: no tonal lines in the room)"""
    if p['verb'] <= 0.003:
        return st
    snd = np.asarray(snd, np.float64)
    g = np.sqrt(np.mean(st ** 2) / (np.mean(snd ** 2) + 1e-20)) if np.any(snd) else 0.0
    wet = fx.reverb(np.stack([snd, snd], 1) * g, r, rt60=p['rt'], size=size, predelay=0.003, damp=0.75, wet=p['verb'],
                    dry=0.0, early_db=-12.0, width=min(1.0, 4 * p['width']), stereo=True, lowcut=lowcut)
    out = np.zeros((max(len(st), len(wet)), 2))
    out[:len(st)] += st
    out[:len(wet)] += wet
    return out


def render(p, r):
    lvl = 10 ** (0.04 * r.standard_normal())        # round-robin level / pitch jitter (natural, small)
    x, snd = click_layers(p, r, level=lvl, send=True)
    if p['rel'] > 0.004:
        y, ys = click_layers(dict(p, body=p['body'] * 0.5, tone=p['tone'] * 0.8), r, f_scale=1.18,
                             level=p['rel'] * lvl, send=True)
        dt = p['rel_dt'] * np.exp(0.05 * r.standard_normal())
        dsp.place(x, y, dt)
        dsp.place(snd, ys, dt)
    x = finish(x, p['hp'], p['deharsh'], p['deharsh_f'], p['sat'], lp=p['lp'])
    snd = finish(snd, max(p['hp'], 300.0), p['deharsh'], p['deharsh_f'], 0.0, lp=p['lp'])
    pre = dsp.n_of(0.002)                            # 2 ms lead-in: the file starts at (almost) the transient
    x = np.concatenate([np.zeros(pre), x])
    snd = np.concatenate([np.zeros(pre), snd])
    st = space(x, r, p['room'], p['width'])
    st = room_tail(st, snd, r, p)
    end = max(p['length'], (p['rel_dt'] + 0.015) if p['rel'] > 0.004 else 0.0)
    st = gate(st, end, p['release_ms'] * 1e-3, start=0.002)
    st = fx.limit(st, ceiling_db=-1.0)
    return dsp.fade(st, 0.0005, 0.002)


# ------------------------------------------------------------------------------------------------ families
def narrow(space, fam):
    """a SPACE narrowed to one family dict: fixed values become one-choice lists, tuples narrow ranges"""
    sp = dict(space)
    for k, v in fam.items():
        if isinstance(v, tuple):
            kind = v[2] if len(v) > 2 else ('int' if len(sp[k]) > 2 and sp[k][2] == 'int' else 'lin')
            sp[k] = (v[0], v[1], kind) if kind != 'lin' else (v[0], v[1])
        else:
            sp[k] = [v]
    return sp


def centre(defaults, fam):
    """DEFAULTS moved to the centre of one family dict (geometric centre for log ranges)"""
    p = dict(defaults)
    for k, v in fam.items():
        if isinstance(v, tuple):
            c = float(np.sqrt(v[0] * v[1])) if len(v) > 2 and v[2] == 'log' else (v[0] + v[1]) / 2
            p[k] = int(round(c)) if isinstance(p.get(k), int) else c
        else:
            p[k] = v
    return p


def family_space(cat, fam):
    return narrow(SPACE, FAMILIES[cat][fam])


def family_params(cat, fam):
    return centre(DEFAULTS, FAMILIES[cat][fam])
