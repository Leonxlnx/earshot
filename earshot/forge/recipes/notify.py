"""Notification / success dings for motion design, written in code (every sample computed, no recorded audio).

    from earshot.forge import dsp
    from earshot.forge.recipes import notify
    x = notify.render(notify.FAMILY['pro_notify'][0], dsp.rng(1))      # 48 kHz stereo, the pro two-stage ding
    xs = [notify.render(p, dsp.rng(i)) for i, p in enumerate(notify.FAMILY['pro_notify'])]   # a palette of 5

What a pro launch-film ding is (docs/motion-sfx-study.md, measured on the 4 dings of ref_2, looked at in spectrograms):
  * A struck tone at 1.8-2.2 kHz with ONE inharmonic partial near 3.4x (6.3-6.9 kHz); nothing under 500 Hz.
  * Two stages, not a natural ring-out: the bright HEAD holds almost flat for 30-50 ms, then a gate drops it about
    18-22 dB within a few ms; a quiet BODY (-20 dB) carries on to 0.22-0.36 s with a slow tremolo (~20-30 Hz) and
    lines an octave under the head (0.5x) and at 1.5x / 3.5x, then fades (convex: it holds, then stops).
  * Attack 1-15 ms (a -30 dB step, then a short rise); t20 27-42 ms, t40 30-180 ms; near mono (side 5-13 dB under
    mid: the body is a little wide, the head centred).
  * A sequence steps up one semitone per ding (1822, 1934, 2045, 2168 Hz).
Real notification bells (the 'notify' refset: struck bells, glass, kalimba; held-out: celesta, glass, triangle) ring
out naturally for 0.8-2 s; style 'bell' makes that kind (modal partials whose highs die first, a mallet, a room).

Styles (`style`):
  'ding'     the pro two-stage ding above (head + gated drop + tremolo body), 0.2-0.4 s
  'success'  two dings of the same patch, the second a third / fourth / fifth up (`interval` semitones) `gap` s later
  'bell'     a struck modal bell / glass / celesta / glock bar ringing out (`partials`), 0.6-1.8 s
CHARACTER parameters (what a family varies on purpose): f0 (the note), hold (head length), body_db / body_ratio /
trem_hz (the body), bright (the 3.4x partial), length, width; for 'bell': partials, decay, hardness.
FAMILY[refset] = 5 hand-designed palette members (pro_notify: the semitone ladder + a success + a glass variant;
notify: celesta / glass / glock / bowl / two-note bell). The forge tunes the fine parameters of each member with the
character ones fixed (family_space()). The delivered (round-2) tuned members are in out/forge_v2/<refset>/final/
report.json (picks[i]['family_params']) and in the seed bank; FAMILY['notify'] already holds the round-3 proposal
(shorter decays, harder mallets) that no forge run has scored yet.

Helpers shared with the other tonal recipes (pop, shimmer): measure(x) = the pro-study measurements (attack, t20 / t40,
band shares, tilt, stereo correlation, head pitch, ridge sweep) and pro_penalty(x, targets) = how far a render falls
outside a category's pro targets (used as the critic's `extra=` check).
"""
import numpy as np

from .. import dsp, fx

SR = dsp.SR
TARGET = 'a short bright notification ding'
ALTS = ['a phone ringtone', 'an alarm clock beep', 'a wine glass clink', 'a xylophone note', 'a typewriter bell']
# (deliberately not the benchmark's notify confusions: eval/forge/bench.py CONFUSIONS stay independent)
REFSET = 'pro_notify'

SEMI = 2 ** (1 / 12)

SPACE = {
    'style': ['ding', 'success', 'bell'],
    'f0': (1100.0, 2600.0, 'log'),    # the note (head fundamental); pro dings 1822-2168 Hz
    # head
    'att': (0.0008, 0.012, 'log'),    # rise after the -30 dB step
    'step_db': (-36.0, -18.0),        # the onset step before the rise
    'hold': (0.018, 0.07),            # head holds this long ...
    'hold_tau': (0.04, 0.4, 'log'),   # ... sagging with this time constant
    'drop_db': (-26.0, -12.0),        # then a gate drops the head this far
    'drop_ms': (2.0, 14.0),           # within this time
    'ring_tau': (0.02, 0.15, 'log'),  # what is left of the head rings out with this tau
    'bright': (0.05, 0.9),            # level of the inharmonic partial
    'p2': (3.2, 3.6),                 # its ratio (3.4x in the films)
    'oct': (0.0, 0.3),                # a faint octave (2x) in the head
    'detune': (0.0, 3.0),             # head doublet beating (Hz)
    'tick_db': (-40.0, -18.0),        # mallet tick (band-passed noise, 1-2 ms)
    'tick_f': (4000.0, 11000.0, 'log'),
    # body
    'body_db': (-26.0, -12.0),        # body level re head
    'body_ratio': [0.5, 1.0, 1.5],    # body fundamental / f0 (0.5 = the films' octave-under line)
    'body_in': (0.004, 0.03),         # body fade-in time
    'body_tau': (0.08, 0.6, 'log'),   # body decay
    'body_p3': (0.0, 0.8),            # body partial at 3x its fundamental
    'body_p7': (0.0, 0.6),            # and near 7x (the films' ~3.5x-of-head line)
    'trem_hz': (12.0, 34.0),          # body tremolo rate
    'trem': (0.0, 0.85),              # and depth
    # glue / space / edit
    'drive': (0.0, 9.0),              # soft saturation dB (presence on small speakers)
    'verb': (0.0, 0.35),              # small plate send
    'rt60': (0.15, 0.7, 'log'),
    'width': (0.0, 1.0),              # body / reverb decorrelation
    'head_width': (0.0, 0.35),        # a little decorrelation on the head too (the films: corr 0.67-0.92)
    'length': (0.2, 0.42),            # total length
    'fade': (0.02, 0.09),             # convex end fade
    'hp': (450.0, 900.0, 'log'),      # nothing under 500 Hz
    # success
    'interval': [3, 4, 5, 7],
    'gap': (0.05, 0.13),
    'second_db': (-6.0, 1.0),
    # bell
    'partials': ['celesta', 'glass', 'bar', 'bowl', 'tine'],
    'decay': (0.2, 1.6, 'log'),
    'slope': (0.4, 1.2),
    'hard': (0.0, 1.0),               # mallet hardness (strike brightness)
    'beat': (0.3, 4.0, 'log'),
    'room': (0.0, 0.3),
}

PARTIALS = {   # ratios, amplitudes
    'celesta': ([1.0, 2.0, 3.98, 6.9, 10.1], [1.0, 0.06, 0.22, 0.05, 0.03]),   # tuned steel bar over a resonator
    'glass': ([1.0, 2.32, 4.25, 6.63, 9.38], [1.0, 0.45, 0.25, 0.12, 0.06]),
    'bar': ([1.0, 2.756, 5.404, 8.933], [1.0, 0.35, 0.14, 0.06]),               # glockenspiel (free-free beam)
    'bowl': ([1.0, 2.71, 5.13, 8.27], [1.0, 0.6, 0.3, 0.12]),
    'tine': ([1.0, 2.0, 5.8, 6.27], [1.0, 0.12, 0.12, 0.25]),                    # clamped tine (kalimba-like)
}

DEFAULTS = {'style': 'ding', 'f0': 1822.0, 'att': 0.003, 'step_db': -30.0, 'hold': 0.032, 'hold_tau': 0.15,
            'drop_db': -24.0, 'drop_ms': 5.0, 'ring_tau': 0.05, 'bright': 0.35, 'p2': 3.42, 'oct': 0.06,
            'detune': 1.2, 'tick_db': -30.0, 'tick_f': 7000.0, 'body_db': -23.0, 'body_ratio': 0.5, 'body_in': 0.012,
            'body_tau': 0.3, 'body_p3': 0.45, 'body_p7': 0.6, 'trem_hz': 24.0, 'trem': 0.55, 'drive': 3.0,
            'verb': 0.12, 'rt60': 0.3, 'width': 0.8, 'head_width': 0.15, 'length': 0.26, 'fade': 0.07, 'hp': 550.0, 'interval': 7,
            'gap': 0.085, 'second_db': -2.0, 'partials': 'celesta', 'decay': 0.8, 'slope': 0.8, 'hard': 0.4,
            'beat': 1.0, 'room': 0.12}


def _fam(**kw):
    return {**DEFAULTS, **kw}


_LADDER = [1822.0 * SEMI ** k for k in range(5)]      # 1822, 1930, 2045, 2167, 2296 Hz
FAMILY = {
    # the pro palette: one patch, five members a listener can tell apart (pitch step, head length, body, motif)
    'pro_notify': [
        _fam(f0=_LADDER[0]),                                                            # the film ding
        _fam(f0=_LADDER[1], hold=0.022, body_ratio=1.0, body_db=-25.0, trem=0.2, bright=0.55, length=0.22),
        _fam(style='success', f0=_LADDER[0], interval=7, hold=0.026, length=0.38, body_db=-25.0),
        _fam(f0=_LADDER[2], hold=0.04, bright=0.15, body_db=-21.0, body_tau=0.4, width=1.0, length=0.33,
             trem_hz=18.0),
        _fam(f0=_LADDER[3], hold=0.018, bright=0.8, p2=3.5, body_ratio=1.5, trem=0.7, trem_hz=30.0, length=0.2),
    ],
    # real-style notification bells: struck, ringing out
    'notify': [   # round 3: a bright strike that dies fast over a shorter ring (round 2 rang flat, then faded)
        _fam(style='bell', f0=1568.0, partials='celesta', decay=0.45, slope=1.1, hard=0.75, length=1.2, room=0.12),
        _fam(style='bell', f0=1175.0, partials='glass', decay=0.55, slope=1.1, hard=0.8, length=1.4, room=0.15),
        _fam(style='bell', f0=2093.0, partials='bar', decay=0.35, slope=1.0, hard=0.85, length=0.9, room=0.1),
        _fam(style='bell', f0=659.0, partials='bowl', decay=0.8, slope=0.9, hard=0.6, length=1.7, room=0.15,
             hp=200.0),
        _fam(style='bell', f0=1319.0, partials='tine', decay=0.4, slope=1.1, hard=0.7, length=1.1, room=0.1,
             hp=300.0),
    ],
}
# what each style lets the forge tune (the character keys stay fixed per family member)
FINE = {
    'ding': ['att', 'step_db', 'head_width', 'hold_tau', 'drop_db', 'drop_ms', 'ring_tau', 'p2', 'oct', 'detune', 'tick_db',
             'tick_f', 'body_in', 'body_tau', 'body_p3', 'body_p7', 'drive', 'verb', 'rt60', 'fade', 'hp'],
    'bell': ['slope', 'hard', 'beat', 'room', 'tick_db', 'tick_f', 'drive', 'fade', 'att'],
}
FINE['success'] = FINE['ding'] + ['gap', 'second_db']


def family_space(params, style=None):
    """the search space for one family member: its FINE keys free (full SPACE ranges), every other key fixed to its
    value (a one-element choice list: the forge's way of pinning a parameter)"""
    st = style or params.get('style', 'ding')
    free = set(FINE.get(st, []))
    sp = {}
    for k, spec in SPACE.items():
        if k in free:
            sp[k] = spec
        else:
            sp[k] = [params.get(k, DEFAULTS.get(k))]
    return sp


# ------------------------------------------------------------------------------------------------ synthesis
def _partial(t, f, amp, r, detune=0.0):
    ph = r.uniform(0, 2 * np.pi)
    y = np.sin(2 * np.pi * f * t + ph)
    if detune > 0:                       # a doublet that starts in phase at the strike, then beats
        y = (y + 0.6 * np.sin(2 * np.pi * (f + detune * r.uniform(0.6, 1.4)) * t + ph)) / 1.6
    return amp * y


def _head_env(t, p):
    """the -30 dB step, a short rise, a sagging hold, a fast gated drop, then what is left rings out"""
    att, hold = p['att'], p['hold']
    step = 10 ** (p['step_db'] / 20)
    rise = np.clip(t / att, 0, 1)
    rise = step + (1 - step) * (0.5 - 0.5 * np.cos(np.pi * rise))
    sag = np.exp(-t / p['hold_tau'])
    drop_t = max(p['drop_ms'] / 1000, 0.001)
    floor = 10 ** (p['drop_db'] / 20)
    u = np.clip((t - hold) / drop_t, 0, 1)
    gate = 1 - (1 - floor) * (0.5 - 0.5 * np.cos(np.pi * u))        # 1 -> floor along a half cosine
    after = np.where(t > hold + drop_t, np.exp(-(t - hold - drop_t) / p['ring_tau']), 1.0)
    e = rise * sag * gate * after
    e[0] = 0.0
    return e


def _tick(n, p, r):
    k = min(n, dsp.n_of(0.004))
    w = r.standard_normal(k)
    lo = p['tick_f'] / 1.6
    w = dsp.butter(w, 'band', (lo, min(p['tick_f'] * 1.6, 20000)), order=2)
    w *= np.exp(-dsp.t_of(k) / 0.0007)
    w /= np.abs(w).max() + 1e-12
    y = np.zeros(n)
    y[:k] = w * 10 ** (p['tick_db'] / 20)
    return y


def _ding(p, r, n):
    """one two-stage ding: (head mono, body mono); both start at t = 0"""
    t = dsp.t_of(n)
    f0 = p['f0']
    head = _partial(t, f0, 1.0, r, p['detune'])
    hp2 = p['bright'] * np.exp(-t / 0.09)                                  # the high partial dies a little faster
    if f0 * p['p2'] < 0.45 * SR:
        head += hp2 * _partial(t, f0 * p['p2'], 1.0, r)
    if p['oct'] > 0:
        head += p['oct'] * _partial(t, 2 * f0, 1.0, r)
    head *= _head_env(t, p)
    head += _tick(n, p, r)
    # body: a quieter tone under / beside the head, swelling in during the hold, tremolo, slow decay
    F = f0 * p['body_ratio']
    body = _partial(t, F, 1.0, r, 0.8)
    norm = 1.0
    for mult, a in ((3.0, p['body_p3']), (7.0 * 0.995, p['body_p7'])):
        if a > 0 and F * mult < 0.45 * SR:
            body += a * _partial(t, F * mult, 1.0, r)
            norm += a
    body /= norm                                                           # body_db = the body's peak level
    t0 = p['hold'] * 0.6
    ein = np.clip((t - t0) / max(p['body_in'], 1e-3), 0, 1)
    env = (0.5 - 0.5 * np.cos(np.pi * ein)) * np.exp(-np.maximum(t - t0, 0) / p['body_tau'])
    trem = 1 - p['trem'] * (0.5 - 0.5 * np.cos(2 * np.pi * p['trem_hz'] * t + r.uniform(0, 1)))
    body *= env * trem * 10 ** (p['body_db'] / 20)
    return head, body


def _bell(p, r, n):
    """a struck modal bell: partials whose highs die first, doublet beating, a mallet whose hardness sets how much
    of the upper partials it excites"""
    t = dsp.t_of(n)
    ratios, amps = PARTIALS[p['partials']]
    y = np.zeros(n)
    for q, a in zip(ratios, amps):
        f = p['f0'] * q
        if f > 0.45 * SR:
            continue
        a = a * q ** (1.5 * (p['hard'] - 0.5))
        tau = max(p['decay'] * q ** -p['slope'], 0.004)
        b = p['beat'] * r.uniform(0.5, 1.5)                      # doublet beating as a slow AM, loudest at the strike
        am = 1 - 0.35 * (0.5 - 0.5 * np.cos(2 * np.pi * b * t))
        y += a * np.exp(-t / tau) * am * np.sin(2 * np.pi * f * t + r.uniform(0, 6.3))
    a = dsp.n_of(max(p['att'] * 0.5, 0.0005))
    y[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    # mallet: low-passed noise pulse, brighter when hard
    k = min(n, dsp.n_of(0.006))
    m = dsp.butter(r.standard_normal(k), 'low', 2000 + 14000 * p['hard'], order=2) * np.exp(-dsp.t_of(k) / 0.001)
    y[:k] += (0.05 + 0.25 * p['hard']) * m / (np.abs(m).max() + 1e-12) * np.abs(y).max()
    y += _tick(n, p, r) * np.abs(y).max()
    return y


def _finish(x, p, r, n_len):
    """glue, space, stereo, band limit, edit"""
    if p['drive'] > 0.2:
        x = fx.saturate(x, p['drive'], mix=0.6)
    x = dsp.butter(x, 'high', p['hp'], order=4)
    return x


def _end(y, p, n_len, fade=None):
    y = y[:n_len]
    if len(y) < n_len:
        pad = np.zeros((n_len - len(y),) + y.shape[1:])
        y = np.concatenate([y, pad])
    k = min(len(y), dsp.n_of(fade or p['fade']))
    w = np.cos(np.linspace(0, np.pi / 2, k)) ** 2            # convex: holds, then stops
    y[-k:] *= w[:, None] if y.ndim == 2 else w
    a = dsp.n_of(0.0003)                                     # round 2: the mallet / tick started off zero
    wi = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    y[:a] *= wi[:, None] if y.ndim == 2 else wi
    return y


def render(p, r):
    p = {**DEFAULTS, **p}
    st = p['style']
    n_len = dsp.n_of(p['length'] if st != 'bell' else max(p['length'], 0.5))
    n = n_len + dsp.n_of(0.05)
    if st == 'bell':
        y = _bell(p, r, n)
        y = _finish(y, p, r, n_len)
        y = fx.decorrelate(y, p['width'] * 0.5, cut=600.0)
        if p['room'] > 0:
            y = fx.reverb(y, r, rt60=0.35 + 1.5 * p['room'], size=0.35, predelay=0.008, damp=0.55, wet=p['room'],
                          stereo=True, width=0.8)
        return _end(y, p, n_len, fade=max(p['fade'], 0.4 * n_len / SR))
    head, body = _ding(p, r, n)
    if st == 'success':
        q = {**p, 'f0': p['f0'] * SEMI ** p['interval']}
        h2, b2 = _ding(q, r, n)
        g = 10 ** (p['second_db'] / 20)
        off = dsp.n_of(p['gap'])
        h2 = np.concatenate([np.zeros(off), h2])[:n] * g
        b2 = np.concatenate([np.zeros(off), b2])[:n] * g
        body = body * np.clip(1 - dsp.t_of(n) / (p['gap'] + 0.03), 0.25, 1) + b2   # the first body ducks under
        head = head + h2
    head = _finish(head, p, r, n_len)    # round 3 tried a clean head (no drive): every critic number got worse
    body = _finish(body, p, r, n_len)    # (totals -5.7..-7.2 -> -10.0..-12.9, tone 8-11 -> 16-18 dB), so it stays
    st_body = fx.decorrelate(body, p['width'], cut=600.0)
    y = fx.decorrelate(head, p['head_width'], cut=1500.0) + st_body
    if p['verb'] > 0.005:                    # the plate is fed mostly by the body: the head's gate stays a gate
        send = 0.3 * np.stack([head, head], 1) + st_body
        wet = fx.reverb(send, r, rt60=p['rt60'], size=0.25, predelay=0.006, damp=0.45, wet=1.0, dry=0.0, stereo=True,
                        width=0.4 + p['width'])
        y = np.concatenate([y, np.zeros((len(wet) - len(y), 2))]) + p['verb'] * wet
    return _end(y, p, n_len)


# ------------------------------------------------------------------------------------------------ measurements
REGIONS = ((0, 150), (150, 500), (500, 1000), (1000, 4000), (4000, 12000), (12000, 24000))


def _env_db(m, ms):
    h = max(1, int(SR * ms / 1000))
    k = len(m) // h
    if k < 1:
        return np.array([-120.0])
    e = np.sqrt((m[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12
    return 20 * np.log10(e)


def measure(x):
    """the pro-study measurements of one render (bed = silence): length_s, attack_ms, t20_ms, t40_ms, shares (dB of
    total in REGIONS), tilt (dB / octave, 100 Hz - 10 kHz), corr / side_mid_db (stereo), head_hz (strongest peak in
    the 40 ms after the onset), ridge (strongest peak per 5 ms frame while within 30 dB of the peak: start / end Hz,
    sweep octaves), peak_t (s)"""
    x = np.asarray(x, np.float64)
    st = x if x.ndim == 2 else np.stack([x, x], 1)
    m = st.mean(1)
    out = {'length_s': round(len(m) / SR, 3)}
    e1 = _env_db(m, 1.0)
    pk = int(np.argmax(e1))
    first = int(np.argmax(e1 >= e1[pk] - 20))
    out['attack_ms'] = float(pk - first)
    e10 = _env_db(m, 2.0)
    e10 = np.convolve(e10, np.ones(5) / 5, 'same')
    p10 = int(np.argmax(e10))
    for d, key in ((20, 't20_ms'), (40, 't40_ms')):
        below = np.nonzero(e10[p10:] < e10[p10] - d)[0]
        out[key] = float(below[0] * 2.0) if len(below) else None
    out['peak_t'] = round(pk / 1000, 4)
    F = np.fft.rfft(m * np.hanning(len(m)))
    P = np.abs(F) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    tot = P.sum() + 1e-20
    out['shares'] = [round(float(10 * np.log10(P[(f >= a) & (f < b)].sum() / tot + 1e-12)), 1) for a, b in REGIONS]
    cs = [1000 * 2 ** (k / 3) for k in range(-10, 11)]
    bl = []
    for c in cs:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        bl.append(max(-60.0, 10 * np.log10(P[sel].sum() / tot + 1e-12)))   # empty bands floored (as the study)
    out['tilt'] = round(float(np.polyfit(np.log2(cs), bl, 1)[0]), 2)
    L, R = st[:, 0], st[:, 1]
    out['corr'] = round(float(np.corrcoef(L, R)[0, 1]) if L.std() > 0 and R.std() > 0 else 1.0, 3)
    M, S = (L + R) / 2, (L - R) / 2
    out['side_mid_db'] = round(float(10 * np.log10((S ** 2).mean() / ((M ** 2).mean() + 1e-20) + 1e-12)), 1)
    a = int(first * SR / 1000)
    seg = m[a:a + int(0.04 * SR)]
    if len(seg) > 256:
        S1 = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8192))
        out['head_hz'] = round(float(np.fft.rfftfreq(8192, 1 / SR)[int(np.argmax(S1))]), 0)
    hop = int(0.005 * SR)
    win = np.hanning(512)
    tr = []
    for i in range(0, len(m) - 512, hop):
        fr = m[i:i + 512] * win
        lev = 10 * np.log10((fr ** 2).mean() + 1e-20)
        tr.append((lev, float(np.fft.rfftfreq(4096, 1 / SR)[int(np.argmax(np.abs(np.fft.rfft(fr, 4096))))])))
    if tr:
        top = max(l for l, _ in tr)
        act = [hz for l, hz in tr if l >= top - 30]
        out['ridge'] = {'start_hz': round(act[0]), 'end_hz': round(act[-1]), 'max_hz': round(max(act)),
                        'min_hz': round(min(act)),
                        'sweep_oct': round(float(np.log2(max(act[-1], 1.0) / max(act[0], 1.0))), 2)}
    return out


def _outside(v, lo, hi, scale):
    if v is None:
        return 1.0
    if v < lo:
        return (lo - v) / scale
    if v > hi:
        return (v - hi) / scale
    return 0.0


# per category: (measurement key, lo, hi, scale) -> penalty = sum of (distance outside / scale), capped per key
TARGETS = {
    'pro_notify': [('length_s', 0.2, 0.42, 0.1), ('attack_ms', 0.0, 15.0, 10.0), ('t20_ms', 20.0, 45.0, 20.0),
                   ('t40_ms', 150.0, 340.0, 80.0), ('head_hz', 1700.0, 2400.0, 400.0), ('low_db', -200.0, -38.0, 8.0),
                   ('corr', 0.6, 0.97, 0.2)],
    'notify': [('length_s', 0.5, 2.0, 0.4), ('attack_ms', 0.0, 10.0, 10.0), ('low_db', -200.0, -20.0, 8.0)],
}


def pro_penalty(x, cat):
    """sum over the category's TARGETS of how far the render falls outside each range (in the range's own scale,
    each capped at 2); 0 = every pro target met"""
    m = measure(x)
    m['low_db'] = 10 * np.log10(sum(10 ** (s / 10) for s in m['shares'][:2]) + 1e-12)
    if 'ridge' in m:
        m['sweep_oct'] = m['ridge']['sweep_oct']
    pen = 0.0
    for key, lo, hi, sc in TARGETS.get(cat, []):
        pen += min(2.0, _outside(m.get(key), lo, hi, sc))
    return pen


def extra_for(cat, weight=1.0):
    """an `extra=` check for the Critic: -weight * pro_penalty"""
    return lambda x, row: -weight * pro_penalty(x, cat)
