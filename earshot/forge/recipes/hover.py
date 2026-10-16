"""A designed hover / scroll tick for motion-design launch videos: tiny, high, nothing under 500 Hz.

Measured in the reference films (docs/motion-sfx-study.md, 'Hover tick'): 35-80 ms windows, attack 0-5 ms, t20 6-25 ms;
centroid 2.8-9.5 kHz; tilt +2.4 to +6.2 dB/oct; balance 1 kHz -23, 2 kHz -18, 4 kHz -13, 8 kHz -8 dB re total, almost
nothing under 500 Hz. Two kinds: tonal pings (a single partial at 6-15 kHz, ~10 ms) and filtered noise ticks (2-12 kHz
noise, 25-40 ms, the spectral peak falling as it decays). The real ticks of the 'hover_tick' refset come in two
registers too: ~4.5 kHz (2-4 kHz peak, t40 65-90 ms) and ~10-12 kHz (t40 25-40 ms). A hover sits 12-20 dB under the
click in a mix and must never get personality: it is heard hundreds of times (techniques 6).

Layers: ping (one damped partial at ping_f, optionally a faint upper partner), tick (band-passed noise whose band
centre falls by `fall` octaves over its decay: the falling peak of the pro noise ticks), an optional second micro tick
(a scroll detent: two contacts 4-15 ms apart), then a steep high-pass at `hp` (>= 500 Hz), a soft top, a tiny bright
room tail (verb, rt), an end gate.
The pro ticks of the SFX-only film are decorrelated (correlation ~0): `width` decorrelates the tick (the ping stays
centred). CHARACTER parameters: ping (ping vs noise), ping_f / tick_f (register), decays, fall, detent, width.
Every sample is computed; no recorded audio is used.
"""
import numpy as np

from .. import dsp, fx
from .click import centre, gate, narrow, partials

TARGET = 'a tiny high soft tick when the cursor hovers over a button'
ALTS = ['a clock ticking', 'a drop of water', 'a mouse click', 'a cricket chirp']
TARGETS = {'hover_tick': 'a tiny high soft tick when the cursor hovers over a button',
           'pro_hover_tick': 'a tiny airy digital tick for an interface animation'}
REFSET = 'hover_tick'

SPACE = {
    'ping': (0.0, 1.0),                  # share of the tonal ping vs the noise tick
    'ping_f': (3000.0, 15000.0, 'log'),
    'ping_ms': (1.0, 8.0, 'log'),        # <= 8 ms: longer pings drew a narrow ringing line through the tick (round 1)
    'ping_up': (0.0, 0.5),               # a faint upper partner partial (x 1.5-2.4)
    'ping_ratio': (1.5, 2.4),
    'tick_f': (2500.0, 11000.0, 'log'),  # noise band centre at the onset
    'tick_q': (0.8, 4.0, 'log'),
    'tick_ms': (1.5, 20.0, 'log'),
    'snap': (1.0, 3.0),                  # decay exponent: higher = a faster first fall (refs: -20 dB in ~8 ms)
    'fall': (0.0, 1.5),                  # octaves the noise band falls over its decay
    'detent': (0.0, 0.8),                # second micro tick level (0 = none)
    'detent_dt': (0.004, 0.016),
    'attack_ms': (0.05, 3.0, 'log'),
    'hp': (500.0, 2500.0, 'log'),
    'lp': (8000.0, 20000.0, 'log'),
    'sat': (0.0, 6.0),
    'length': (0.02, 0.11),
    'release_ms': (3.0, 25.0, 'log'),
    'width': (0.0, 1.0),
    'verb': (0.0, 0.35),                 # a tiny bright room after the tick: the refs fall fast to -30..-40 dB, then
    'rt': (0.04, 0.3, 'log'),            # hold a low tail (t40 25-87 ms)
    'contact': (0.0, 40.0),              # round 3: a 0.1-0.6 ms broadband contact impulse (2-20 kHz, decorrelated with
    'contact_ms': (0.08, 1.0, 'log'),    # the tick) re the tick peak: the pro ticks' peak IS that impulse (t20 1-3 ms,
    'grains': (0.0, 0.5),                # sparse micro-grains after the contact (4 impulses at 2-25 ms, each 0-20 dB
}                                        # under `grains` x the contact): the refs' scattered crackle after the spike;
                                         # contact = 0 = none of this (the round-2 sounds)
DEFAULTS = {'ping': 0.5, 'ping_f': 8000.0, 'ping_ms': 6.0, 'ping_up': 0.15, 'ping_ratio': 1.9, 'tick_f': 6000.0,
            'tick_q': 1.6, 'tick_ms': 6.0, 'snap': 1.5, 'fall': 0.6, 'detent': 0.0, 'detent_dt': 0.008,
            'attack_ms': 0.3, 'hp': 900.0, 'lp': 16000.0, 'sat': 1.0, 'length': 0.045, 'release_ms': 10.0,
            'width': 0.3, 'verb': 0.0, 'rt': 0.1, 'contact': 0.0, 'contact_ms': 0.2, 'grains': 0.0}

FAMILIES = {   # round 2: pings only where the family is a ping (<= 8 ms), faster first falls
    'pro_hover_tick': {   # the pro ticks fall to -40 dB within 6-14 ms and are bright at the onset: dry, fast, high.
        # Round 3: the forge refs' peak is a broadband contact impulse (t20 1-3 ms, > 12 kHz -5 dB of total, L/R
        # correlation ~0) and the round-2 ticks did not read as ticks (phrase z 0.85 vs real 1.8-3.0): every family now
        # has the decorrelated contact over its ping / noise tick, and more width; the contact weight differs per family
        # (ping 3-7x and soft 1.5-4x keep their ping / tick audible, air and detent 12-30x are the refs' spike + grains)
        'ping': {'detent': 0.0, 'ping': (0.6, 0.9), 'ping_f': (8000, 14000, 'log'), 'ping_ms': (1.5, 3, 'log'),
                 'snap': (2.0, 3.0), 'contact': (3.0, 7.0), 'grains': 0.0, 'contact_ms': (0.25, 0.6, 'log'), 'width': (0.6, 1.0),
                 'verb': (0.0, 0.06), 'lp': (13000, 17000, 'log'), 'length': (0.025, 0.045)},
        'air': {'detent': 0.0, 'ping': (0.0, 0.15), 'tick_f': (7000, 11000, 'log'), 'fall': (0.4, 1.2),
                'tick_ms': (2, 6, 'log'), 'snap': (2.0, 3.0), 'contact': (12.0, 30.0), 'grains': (0.05, 0.35), 'contact_ms': (0.3, 0.8, 'log'),
                'width': (0.8, 1.0), 'verb': (0.0, 0.06), 'lp': (13000, 17000, 'log'), 'length': (0.03, 0.055)},
        'glass': {'detent': 0.0, 'ping': (0.3, 0.6), 'ping_f': (7000, 11000, 'log'), 'ping_ms': (1.5, 3, 'log'),
                  'snap': (2.0, 3.0), 'tick_f': (7000, 11000, 'log'), 'ping_up': (0.15, 0.4), 'contact': (6.0, 12.0), 'grains': (0.0, 0.15), 'hp': (1000, 2200, 'log'),
                  'contact_ms': (0.25, 0.6, 'log'), 'width': (0.6, 1.0), 'verb': (0.0, 0.06),
                  'lp': (13000, 17000, 'log'), 'length': (0.03, 0.05)},
        'soft': {'detent': 0.0, 'ping': (0.0, 0.3), 'ping_ms': (1.5, 3, 'log'), 'tick_f': (5000, 8000, 'log'),
                 'tick_ms': (2, 6, 'log'), 'snap': (1.5, 2.5), 'attack_ms': (0.3, 1.2, 'log'), 'contact': (1.5, 4.0), 'grains': 0.0,
                 'contact_ms': (0.4, 1.0, 'log'), 'width': (0.6, 1.0), 'hp': (800, 1500, 'log'), 'verb': (0.0, 0.08),
                 'lp': (12000, 16000, 'log'), 'length': (0.03, 0.06)},
        'detent': {'ping': (0.0, 0.35), 'ping_ms': (1.0, 3, 'log'), 'tick_f': (7000, 11000, 'log'),
                   'snap': (2.0, 3.0), 'tick_ms': (1.5, 4, 'log'), 'detent': (0.35, 0.8), 'detent_dt': (0.006, 0.012),
                   'contact': (12.0, 30.0), 'grains': (0.05, 0.35), 'contact_ms': (0.2, 0.5, 'log'), 'width': (0.7, 1.0),
                   'verb': (0.0, 0.06), 'lp': (13000, 17000, 'log'), 'length': (0.025, 0.045)},
    },
    'hover_tick': {
        'high': {'detent': 0.0, 'ping': (0.0, 0.3), 'ping_ms': (1.0, 3, 'log'), 'tick_f': (8000, 11000, 'log'),
                 'tick_ms': (1.5, 5, 'log'), 'snap': (1.5, 3.0), 'hp': (1500, 2500, 'log'), 'verb': (0.05, 0.3),
                 'rt': (0.05, 0.25, 'log'), 'length': (0.04, 0.07)},
        'mid': {'detent': 0.0, 'ping': (0.0, 0.3), 'ping_ms': (1.0, 4, 'log'), 'tick_f': (3000, 5000, 'log'),
                'tick_ms': (4, 14, 'log'), 'snap': (1.0, 2.0), 'hp': (700, 1400, 'log'), 'verb': (0.05, 0.3),
                'rt': (0.05, 0.25, 'log'), 'length': (0.065, 0.11)},
        'ping': {'detent': 0.0, 'ping': (0.75, 1.0), 'ping_f': (5000, 9000, 'log'), 'ping_ms': (2, 7, 'log'), 'snap': (1.5, 3.0),
                 'verb': (0.05, 0.3), 'rt': (0.05, 0.25, 'log'), 'length': (0.045, 0.085)},
        'air': {'detent': 0.0, 'ping': (0.0, 0.12), 'tick_f': (5000, 9000, 'log'), 'fall': (0.6, 1.4), 'tick_ms': (3, 12, 'log'),
                'verb': (0.05, 0.3), 'rt': (0.05, 0.25, 'log'), 'length': (0.05, 0.095)},
        'detent': {'ping': (0.0, 0.3), 'ping_ms': (1.0, 3, 'log'), 'tick_f': (4500, 8000, 'log'), 'snap': (1.5, 3.0),
                   'detent': (0.35, 0.8), 'detent_dt': (0.006, 0.014), 'verb': (0.05, 0.3), 'rt': (0.05, 0.25, 'log'),
                   'length': (0.05, 0.095)},
    },
}


def _tick(p, r, f_scale=1.0):
    """noise through a band-pass whose centre falls `fall` octaves as the burst decays (time-varying SVF)"""
    tau = p['tick_ms'] * 1e-3
    n = dsp.n_of(max(0.012, tau * 6))
    t = dsp.t_of(n)
    fc = p['tick_f'] * f_scale * 2 ** (-p['fall'] * (1 - np.exp(-t / (tau * 1.5))))
    y = dsp.svf(r.standard_normal(n), np.minimum(fc, 20000), q=p['tick_q'], mode='band')
    e = np.exp(-t / tau) ** p['snap']
    a = max(2, dsp.n_of(p['attack_ms'] * 1e-3))
    e[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    y = y * e
    return y / (np.abs(y).max() + 1e-12)


def _ping(p, r, f_scale=1.0):
    k = 2 if p['ping_up'] > 0.01 else 1
    return partials(p['ping_f'] * f_scale, [1.0, p['ping_ratio']][:k], [1.0, p['ping_up']][:k], p['ping_ms'] * 1e-3,
                    0.1, r, att_s=max(0.0001, p['attack_ms'] * 1e-3))


def _contact(p, r):
    """the contact impulse: a 0.1-1 ms 2-15 kHz noise burst with a squared decay, unit peak (a 20 kHz band edge put
    energy above 20 kHz at -20..-34 dB where the refs have -70..-82: round 3, first run)"""
    from .click import burst
    return burst(p['contact_ms'] * 1e-3, r, 2000.0, 15000.0, att_s=0.00005, order=4)


def _one(p, r, f_scale=1.0):
    """(ping, tick, contact): mono ping and tick (the tick is decorrelated later), and the contact as a stereo
    (n, 2) pair of independent bursts (no all-pass decorrelator: it smeared the 0.3 ms spike over several ms)"""
    pg = _ping(p, r, f_scale)
    tk = _tick(p, r, f_scale)
    m = max(len(pg), len(tk))
    pg, tk = np.pad(pg, (0, m - len(pg))), np.pad(tk, (0, m - len(tk)))
    tk = (1 - p['ping']) * tk
    ct = np.zeros((m, 2))
    if p.get('contact', 0.0) > 0.01:                    # no rng draw without a contact: round-2 renders unchanged
        a = p['contact'] * max(np.abs(tk).max(), np.abs(p['ping'] * pg).max())
        th = 0.5 * np.pi * min(1.0, p['width'])         # width 1: independent L / R bursts (correlation ~0)
        for t_g, amp, ms in [(0.0, 1.0, p['contact_ms'])] + (
                [(tg, p['grains'] * 10 ** (db / 20) * (r.integers(0, 2) * 2 - 1), 0.1)
                 for tg, db in zip(r.uniform(0.002, 0.025, 4), r.uniform(-20.0, 0.0, 4))]
                if p.get('grains', 0.0) > 0.005 else []):
            c1, c2 = _contact(dict(p, contact_ms=ms), r), _contact(dict(p, contact_ms=ms), r)
            L, R = c1, np.cos(th) * c1 + np.sin(th) * c2
            k = dsp.n_of(t_g)
            if k < m:
                q = min(len(L), m - k)
                ct[k:k + q, 0] += a * amp * L[:q]
                ct[k:k + q, 1] += a * amp * R[:q]
    return p['ping'] * pg, tk, ct


def render(p, r):
    lvl = 10 ** (0.03 * r.standard_normal())
    fs = np.exp(0.01 * r.standard_normal())
    n = dsp.n_of(0.14)
    pre = 0.002
    pg_all, tk_all, ct_all = np.zeros(n), np.zeros(n), np.zeros((n, 2))
    pg, tk, ct = _one(p, r, fs)
    dsp.place(pg_all, pg * lvl, pre)
    dsp.place(tk_all, tk * lvl, pre)
    dsp.place(ct_all, ct * lvl, pre)
    if p['detent'] > 0.02:
        pg2, tk2, ct2 = _one(p, r, fs * 1.06)
        dsp.place(pg_all, pg2 * p['detent'] * lvl, pre + p['detent_dt'])
        dsp.place(tk_all, tk2 * p['detent'] * lvl, pre + p['detent_dt'])
        dsp.place(ct_all, ct2 * p['detent'] * lvl, pre + p['detent_dt'])
    st = np.stack([pg_all, pg_all], 1) + fx.decorrelate(tk_all, amount=p['width'], cut=1500.0)[:n] + ct_all
    st = dsp.butter(st, 'high', p['hp'], order=4)        # nothing under 500 Hz
    st = dsp.butter(st, 'low', p['lp'], order=2)
    if p.get('contact', 0.0) > 0.01:                    # the curve below is not level-normalised: a 20x contact
        st = st / (np.abs(st).max() + 1e-12)              # would be clipped flat into the tick (round 3)
    if p['sat'] > 0.05:
        st = fx.saturate(st, drive_db=p['sat'], mode='tanh', level='peak')
    if p['verb'] > 0.003:
        st = fx.reverb(st, r, rt60=p['rt'], size=0.1, predelay=0.002, damp=0.5, wet=p['verb'], early_db=-10.0,
                       width=1.0, stereo=True, lowcut=1500.0)
    last = p['detent_dt'] if p['detent'] > 0.02 else 0.0           # never gate a hit before it has decayed ~50 dB
    ring = 5.76e-3 * max(p['ping_ms'] if p['ping'] > 0.05 else 0.0, p['tick_ms'] / p['snap'] if p['ping'] < 0.95 else 0.0)
    st = gate(st, max(p['length'], last + ring + p['release_ms'] * 1e-3), p['release_ms'] * 1e-3, start=pre)
    st = fx.limit(st, ceiling_db=-1.0)
    return dsp.fade(st, 0.0005, 0.002)


def family_space(cat, fam):
    return narrow(SPACE, FAMILIES[cat][fam])


def family_params(cat, fam):
    return centre(DEFAULTS, FAMILIES[cat][fam])
