"""A short cut hit: the sub hit with a kick that music-led films put on a cut, and its punchy mid sibling; every sample
is computed.

What the reference clips do (a private licensed set, 'sub_drop' / 'impact', medians
with the interquartile range; the film clips are local study material and are not shipped):
  * the sub drop of the voice-over and music films is SHORT: 0.26 s (0.11-0.59) long, its loudest point 61 ms (37-137)
    after the onset, -20 dB 116 ms (67-145) after that, the sub itself at 57-155 Hz, mono (L/R correlation 0.96). It is
    usually laid with a kick ('sub-bass drop + kick', 12 pairs within 60 ms of each other): the beater lifts the
    centroid to 435 Hz (149-598) and only 45 % (25-86 %) of the energy stays under 150 Hz. earshot 0.3's sub drops
    were 1-1.5 s blooms at 54-83 Hz: faded to 0.3 s they stayed 5-7x darker than the films';
  * the music films' impact is a short punchy hit: 0.28 s, centroid 306 Hz, 64 % under 150 Hz, peak 52 ms in;
  * the effects-led film's designed hits are longer and wide: a kick on a sub drop that ends in a granular rumble
    (L/R correlation about 0), or a fast sub hit under a sustained 5-10 kHz air band.
Layers (each normalised to its own peak, then mixed at its level re the sub):
  sub     a sine falling sub_f0 -> sub_f1 (time constant sub_glide), a raised-cosine bloom of sub_attack to its peak,
          then straight in dB to -20 dB in sub_t20 s; tanh saturation (sub_drive) adds the 2nd / 3rd harmonics small
          speakers need; sub_phase turns the right channel up to 90 degrees (a designed wide sub: correlation
          cos(phase))
  kick    a sine sweeping kick_f0 -> kick_f1 in kick_tau with its own decay kick_decay (the drum under the sub)
  click   the beater: click_ms of band noise around click_f with a (1 - t / len)^3 envelope, plus a 3 ms ping
  punch   a low-mid noise band at punch_f decaying in punch_decay: the smack that puts a punchy hit at ~300 Hz
  rumble  Hann grains (25-60 ms) of low brown noise, grain_rate per second slowing to half, falling to -20 dB over
          rumble_t20; each grain ticks softly through 0.8-5 kHz (grain_tick) and can carry a broadband click
          (crackle: the effects-led film's crackling hit tails); a faint low bed fills the gaps; independent per
          channel
  air     high-passed noise from air_f, a 5 ms rise and a decay of air_len, independent per channel
  space   width decorrelates everything above 150 Hz (the sub keeps its own phase), a short dark room on click + punch
The file ends at `length` with a raised-cosine fade of `fade` s: a sub must never stop audibly.

Character parameters (each family member is one corner): weight (sub_f0 / sub_f1, sub_t20, length), punch (kick,
click, punch, punch_f), bloom (sub_attack), texture (rumble, air) and width (sub_phase, width).
"""
import types

import numpy as np

from .. import dsp

SR = dsp.SR
TARGET = 'a short deep sub bass hit with a kick drum'
ALTS = ['an explosion', 'a door slam', 'thunder', 'a low hum']
REFSET = 'cut_hit'

SPACE = {
    'length': (0.12, 1.8),             # s, the file (it fades over its last `fade` s)
    'fade': (0.004, 0.25, 'log'),
    'sub_f0': (50.0, 260.0, 'log'),    # sub start pitch (Hz)
    'sub_f1': (36.0, 130.0, 'log'),    # sub landing pitch (Hz)
    'sub_glide': (0.008, 0.4, 'log'),  # time constant of the fall (s)
    'sub_attack': (0.002, 0.35, 'log'),  # bloom to the sub's peak (s)
    'sub_t20': (0.03, 0.8, 'log'),     # s from the sub's peak to -20 dB (straight in dB)
    'sub_drive': (0.0, 3.0),           # tanh harmonics
    'sub_phase': (0.0, 1.0),           # right channel phase, x 90 degrees
    'kick': (0.0, 1.5),                # kick level re the sub
    'kick_f0': (120.0, 700.0, 'log'),
    'kick_f1': (40.0, 130.0, 'log'),
    'kick_tau': (0.003, 0.06, 'log'),
    'kick_decay': (0.015, 0.25, 'log'),
    'click': (0.0, 1.0),               # beater level re the sub
    'click_f': (1200.0, 9000.0, 'log'),
    'click_ms': (1.0, 15.0, 'log'),
    'punch': (0.0, 1.2),               # low-mid smack level re the sub
    'punch_f': (140.0, 1000.0, 'log'),
    'punch_decay': (0.008, 0.15, 'log'),
    'rumble': (0.0, 1.2),              # granular low rumble level re the sub
    'rumble_t20': (0.1, 1.5, 'log'),
    'grain_rate': (12.0, 70.0, 'log'),
    'grain_tick': (0.0, 1.0),
    'crackle': (0.0, 1.5),             # a broadband click (0.5-20 kHz, 1-3 ms) on each rumble grain
    'air': (0.0, 1.0),               # high air band level re the sub
    'air_f': (3000.0, 12000.0, 'log'),
    'air_len': (0.03, 0.8, 'log'),
    'width': (0.0, 1.0),               # decorrelation above 150 Hz (0 = mono)
    'room': (0.0, 0.4),
}

FAMILIES = {
    # earshot 0.4 phase B: hand-designed corners, then tuned toward each style on the measured features (the films'
    # category profile and the study's dev-split clips: band balance, envelope, onset, decay, low share, width) and on
    # CLAP only up to the grade bar (cosine 0.70: no pull past it); every member stays an original (max CLAP cosine to
    # any study clip 0.79 or less). A slow-blooming member ('bloom', 0.25 s swell) was dropped: it read as a held
    # tone, not a hit, and it needed a rising sub to fit its one music-bed clip.
    'cut_hit': [
        # the music films' short sub hit: an 808-like sub falling 225 -> 46 Hz under a low punch, -20 dB in ~0.15 s
        ('sub', {'length': 0.2858, 'fade': 0.1949, 'sub_f0': 224.8, 'sub_f1': 45.69, 'sub_glide': 0.04827,
                 'sub_attack': 0.01613, 'sub_t20': 0.04364, 'sub_drive': 0.6188, 'sub_phase': 0.0, 'kick': 0.6151,
                 'kick_f0': 174.9, 'kick_f1': 92.57, 'kick_tau': 0.008307, 'kick_decay': 0.2089, 'click': 0.1667,
                 'click_f': 1226.0, 'click_ms': 2.506, 'punch': 1.069, 'punch_f': 374.4, 'punch_decay': 0.1464,
                 'rumble': 0.0, 'rumble_t20': 0.1565, 'grain_rate': 12.92, 'grain_tick': 0.002876, 'crackle': 0.0,
                 'air': 0.0, 'air_f': 9115.0, 'air_len': 0.03, 'width': 0.3875, 'room': 0.05584}),
        # the same gesture led by the kick: the beater and a 440 Hz smack lift the centroid (the films' median drop)
        ('kick', {'length': 1.022, 'fade': 0.0589, 'sub_f0': 176.6, 'sub_f1': 37.04, 'sub_glide': 0.1106,
                  'sub_attack': 0.007655, 'sub_t20': 0.05602, 'sub_drive': 0.3078, 'sub_phase': 0.0, 'kick': 1.025,
                  'kick_f0': 363.2, 'kick_f1': 108.2, 'kick_tau': 0.009914, 'kick_decay': 0.03626, 'click': 0.1688,
                  'click_f': 2440.0, 'click_ms': 5.914, 'punch': 1.077, 'punch_f': 437.1, 'punch_decay': 0.04711,
                  'rumble': 0.0, 'rumble_t20': 0.1437, 'grain_rate': 29.02, 'grain_tick': 0.0, 'crackle': 0.0,
                  'air': 0.0, 'air_f': 6018.0, 'air_len': 0.07952, 'width': 0.0, 'room': 0.01595}),
        # the music films' punchy impact: ~0.3 s, a driven sub and a bright smack, two thirds under 150 Hz
        ('punch', {'length': 0.2823, 'fade': 0.01307, 'sub_f0': 144.2, 'sub_f1': 47.8, 'sub_glide': 0.06248,
                   'sub_attack': 0.06125, 'sub_t20': 0.03285, 'sub_drive': 2.878, 'sub_phase': 0.09153,
                   'kick': 0.6973, 'kick_f0': 284.4, 'kick_f1': 84.94, 'kick_tau': 0.0141, 'kick_decay': 0.04286,
                   'click': 0.4873, 'click_f': 9000.0, 'click_ms': 5.467, 'punch': 0.9001, 'punch_f': 1000.0,
                   'punch_decay': 0.07649, 'rumble': 0.0, 'rumble_t20': 0.1443, 'grain_rate': 15.95,
                   'grain_tick': 0.04938, 'crackle': 0.0, 'air': 0.0, 'air_f': 5858.0, 'air_len': 0.1071,
                   'width': 0.346, 'room': 0.1467}),
        # the effects-led film's designed hit: a kick on a slow sub drop that ends in a wide granular rumble
        ('rumble', {'length': 1.4, 'fade': 0.04147, 'sub_f0': 125.0, 'sub_f1': 40.27, 'sub_glide': 0.15,
                    'sub_attack': 0.07542, 'sub_t20': 0.09859, 'sub_drive': 0.863, 'sub_phase': 1.0, 'kick': 0.3807,
                    'kick_f0': 197.6, 'kick_f1': 69.6, 'kick_tau': 0.004458, 'kick_decay': 0.0623, 'click': 0.0,
                    'click_f': 2568.0, 'click_ms': 2.467, 'punch': 0.03031, 'punch_f': 140.0,
                    'punch_decay': 0.01199, 'rumble': 1.2, 'rumble_t20': 1.5, 'grain_rate': 21.45,
                    'grain_tick': 0.6988, 'crackle': 0.0, 'air': 0.3205, 'air_f': 3809.0, 'air_len': 0.166,
                    'width': 0.6293, 'room': 0.3665}),
        # a fast sub hit under a crackling granular tail and a sustained bright air band (the effects-led film's
        # erase hit), wide
        ('air', {'length': 0.5493, 'fade': 0.2087, 'sub_f0': 203.3, 'sub_f1': 50.96, 'sub_glide': 0.02,
                 'sub_attack': 0.05396, 'sub_t20': 0.03, 'sub_drive': 3.0, 'sub_phase': 1.0, 'kick': 1.084,
                 'kick_f0': 193.6, 'kick_f1': 50.09, 'kick_tau': 0.006564, 'kick_decay': 0.03479, 'click': 0.4183,
                 'click_f': 3710.0, 'click_ms': 1.241, 'punch': 0.1212, 'punch_f': 190.7, 'punch_decay': 0.1313,
                 'rumble': 0.9522, 'rumble_t20': 0.4372, 'grain_rate': 14.47, 'grain_tick': 0.4375,
                 'crackle': 0.7462, 'air': 0.824, 'air_f': 10810.0, 'air_len': 0.6526, 'width': 1.0,
                 'room': 0.4}),
    ],
}
DEFAULTS = dict(FAMILIES['cut_hit'][0][1])
TARGETS = {'cut_hit': TARGET}


def _bloom(t, attack, t20):
    """raised-cosine rise to 1 over `attack`, then straight in dB: -20 dB every t20 s"""
    e = np.empty_like(t)
    s = t < attack
    e[s] = 0.5 - 0.5 * np.cos(np.pi * t[s] / attack)
    e[~s] = 10 ** (-(t[~s] - attack) / t20)
    return e


def _peak1(x):
    return x / (np.abs(x).max() + 1e-12)


def _sub(p, t, r):
    """(left, right) of the falling sub, saturated; the right channel turned by sub_phase x 90 degrees"""
    f = p['sub_f1'] + (p['sub_f0'] - p['sub_f1']) * np.exp(-t / p['sub_glide'])
    ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(0, 2 * np.pi)
    env = _bloom(t, p['sub_attack'], p['sub_t20'])
    g = 1.0 + 2.0 * p['sub_drive']
    out = []
    for off in (0.0, 0.5 * np.pi * p['sub_phase']):
        s = np.tanh(g * np.sin(ph + off)) / np.tanh(g)
        out.append(dsp.butter(s, 'low', 900.0, 2) * env)
    return out


def _kick(p, t, r):
    f = p['kick_f1'] + (p['kick_f0'] - p['kick_f1']) * np.exp(-t / p['kick_tau'])
    ph = 2 * np.pi * np.cumsum(f) / SR + r.uniform(-0.2, 0.2)
    return np.sin(ph) * np.exp(-t / p['kick_decay']) * (1 - np.exp(-t / 0.0004))


def _click(p, n, r):
    L = min(n, max(16, dsp.n_of(p['click_ms'] / 1000)))
    u = np.arange(L) / L
    w = dsp.butter(r.standard_normal(L) * (1 - u) ** 3, 'band', (p['click_f'] * 0.5, min(p['click_f'] * 2.2, 19000)),
                   2)
    k = min(n, dsp.n_of(0.012))
    tk = np.arange(k) / SR
    ping = np.sin(2 * np.pi * p['click_f'] * tk) * np.exp(-tk / 0.003) * (1 - np.exp(-tk / 0.0002))
    y = np.zeros(n)
    y[:L] += _peak1(w)
    y[:k] += 0.5 * ping
    return _peak1(y)


def _punch(p, t, r):
    f = p['punch_f']
    w = dsp.butter(r.standard_normal(len(t)), 'band', (f / 1.7, min(f * 1.7, 19000)), 2)
    return _peak1(w * np.exp(-t / p['punch_decay']) * (1 - np.exp(-t / 0.0008)))


def _air(p, t, r):
    out = []
    for _ in range(2):
        w = dsp.butter(r.standard_normal(len(t)), 'high', p['air_f'], 4)
        w = dsp.butter(w, 'low', min(p['air_f'] * 2.5, 19000), 2)
        out.append(w * (1 - np.exp(-t / 0.005)) * np.exp(-t / p['air_len']))
    return out


def _rumble(p, n, r):
    """(left, right): Hann grains of low brown noise, independent per channel, slowing, falling to -20 dB per
    rumble_t20; each grain carries a soft 0.8-5 kHz tick (grain_tick), a broadband click (crackle), and a faint low bed
    fills the gaps"""
    out = []
    span = n / SR
    for _ in range(2):
        y = np.zeros(n)
        tg = 0.01
        while tg < span:
            u = tg / span
            m = dsp.n_of(r.uniform(0.025, 0.06))
            g = np.cumsum(r.standard_normal(m))
            g = dsp.butter(g - g.mean(), 'band', (35.0, 220.0), 2) * np.hanning(m)
            # level: falling with rumble_t20, a log-normal spread capped at +2.6 dB (a grain standing over the hit reads
            # as a second attack: the audit's stacked hit)
            a = 10 ** (-tg / p['rumble_t20']) * np.exp(min(0.35 * r.standard_normal(), 0.3))
            i = dsp.n_of(tg)
            mm = min(m, n - i)
            y[i:i + mm] += a * _peak1(g)[:mm]
            if p['grain_tick'] > 0.01:              # a 2.5 ms rise, a 3 ms decay, 0.8-5 kHz: a grain's tick, not a
                k = min(dsp.n_of(0.014), n - i)     # click (shorter, brighter ticks trip the audit's click detector in
                if k > 8:                           # the quiet tail)
                    tk = np.arange(k) / SR
                    tick = dsp.butter(r.standard_normal(k), 'band', (800.0, 5000.0), 2)
                    tick *= np.sin(0.5 * np.pi * np.minimum(tk / 0.0025, 1.0)) ** 2 * np.exp(-tk / 0.003)
                    y[i:i + k] += 0.25 * p['grain_tick'] * a * _peak1(tick)
            if p.get('crackle', 0.0) > 0.01:            # drawn only when used: members without it render as before
                k = min(dsp.n_of(r.uniform(0.001, 0.003)), n - i)
                if k > 8:
                    ck = dsp.butter(r.standard_normal(k), 'high', 500.0, 2) * np.hanning(k)
                    y[i:i + k] += p['crackle'] * a * np.exp(0.5 * r.standard_normal()) * _peak1(ck)
            tg += float(r.exponential(1.0 / (p['grain_rate'] * (1.0 - 0.5 * u))))
        if p['rumble'] > 0.01:                     # a faint low bed under the grains: no digital silence in the gaps
            b = np.cumsum(r.standard_normal(n))
            b = dsp.butter(b - b.mean(), 'band', (35.0, 220.0), 2)
            y += 0.06 * _peak1(b) * 10 ** (-np.arange(n) / SR / p['rumble_t20'])
        out.append(y)
    return out


def render(p, r):
    p = {**DEFAULTS, **p}
    n = dsp.n_of(p['length'])
    t = np.arange(n) / SR
    # the layers draw their noise in this order whatever their level (crackle and the rumble bed only when they are on),
    # so a seed is one take
    subL, subR = _sub(p, t, r)
    kick = _kick(p, t, r)
    click = _click(p, n, r)
    punch = _punch(p, t, r)
    airL, airR = _air(p, t, r)
    rumL, rumR = _rumble(p, n, r)
    top = 0.6 * p['kick'] * kick + 0.5 * p['click'] * click + 0.6 * p['punch'] * punch
    if p['room'] > 0.01:
        top = dsp.room(top, r, rt60=0.3, size=0.35, damp=0.7, wet=p['room'])[:n]
    else:
        dsp.room_ir(r, 0.3, 0.35, 0.7)                     # keep the draw order
    w = p['width']
    rk = np.sqrt(max(0.0, 1 - w * w))                  # right = rk x left + w x independent: correlation ~ rk
    # width: a delayed copy above 150 Hz on the right (the low end of the top layers stays centred)
    hiL = dsp.butter(top, 'high', 150.0, 2)
    lo = top - hiL
    side = dsp.butter(np.concatenate([np.zeros(dsp.n_of(0.0007)), hiL])[:n], 'high', 150.0, 2)
    topL, topR = lo + hiL, lo + rk * hiL + w * side
    rumR = rk * rumL + w * rumR
    rl = np.abs(np.stack([rumL, rumR])).max() + 1e-12
    airR = rk * airL + w * airR
    al = np.abs(np.stack([airL, airR])).max() + 1e-12
    L = subL + topL + p['rumble'] * 0.7 * rumL / rl + p['air'] * 0.35 * airL / al
    R = subR + topR + p['rumble'] * 0.7 * rumR / rl + p['air'] * 0.35 * airR / al
    y = np.stack([L, R], axis=1)
    y = dsp.butter(y, 'high', 24.0, 2)
    y = _end(y, -60.0)                                  # no near-silent tail when `length` outlives the decay
    y = dsp.fade(y, 0.0004, min(p['fade'], 0.9 * len(y) / SR))
    return y / (np.abs(y).max() + 1e-12) * 0.89


def _end(y, end_db):
    """y cut 30 ms after its 5 ms envelope has fallen end_db under its peak for good (it never comes back up)"""
    h = dsp.n_of(0.005)
    m = y.mean(axis=1)
    k = len(m) // h
    if k < 4:
        return y
    e = 20 * np.log10(np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12)
    live = np.nonzero(e >= e.max() + end_db)[0]
    return y[:min(len(y), (int(live[-1]) + 1) * h + dsp.n_of(0.03))]


def space_for(cat='cut_hit'):
    return dict(SPACE)


def recipe_for(cat='cut_hit', preset=0):
    """a recipe object for one family member (index or name), for forge(): see subdrop.recipe_for"""
    fam = FAMILIES[cat]
    i = preset if isinstance(preset, int) else [nm for nm, _ in fam].index(preset)
    return types.SimpleNamespace(__name__=__name__, SPACE=space_for(cat), DEFAULTS=dict(fam[i][1]), render=render,
                                 TARGET=TARGETS[cat], ALTS=ALTS, REFSET=REFSET, PRESET=fam[i][0])
