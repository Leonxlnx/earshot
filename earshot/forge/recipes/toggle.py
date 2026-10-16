"""A designed UI toggle / switch flip for motion-design launch videos: two contacts, one gesture.

A physical toggle is two contacts 30-150 ms apart (Farnell; docs/motion-sfx-techniques.md 6): the lever leaves its
detent (a lighter click) and lands in the other (the main click, with the body). The semantics a viewer expects:
ON rises or sits higher, OFF falls or sits lower (a neutral toggle keeps one pitch). The real switch flips of the
'toggle' refset last 0.19-0.45 s, centroid 1.7-5.8 kHz, t40 90-180 ms, and the main contact is sometimes the second
(fs-448393: 160 ms after a softer first). No pro film had an exposed toggle (motion-sfx-study.md), so the designed
idiom is borrowed from the pro click: bright, short, band-limited, mono object, hard end.

Layers: contact A (the click recipe's transient + tink + body + grit at a_lvl, pitch `pitch`), an optional friction
slide between the contacts (band-passed noise swelling into B), contact B (level 1, every layer's pitch x `dir`), an
optional faint spring ring after B; then click.finish (high-pass, de-harsh dip, light saturation), a tiny space, a
small dark room tail (verb, rt) and the end gate. CHARACTER parameters: gap, dir (on / off / neutral), a_lvl, pitch,
body, slide, length. FAMILIES are the palette (on, off, latch, micro, rocker). Every sample is computed; no recorded
audio is used.
"""
import numpy as np

from .. import dsp, fx
from .click import burst, centre, click_layers, finish, gate, narrow, partials, room_tail, space

TARGET = 'a small switch flipped on with two quick clicks'
ALTS = ['a gun being cocked', 'a pen cap snapping shut', 'a mechanical keyboard key press', 'a coin dropped']
TARGETS = {'toggle': 'a small switch flipped on with two quick clicks'}
REFSET = 'toggle'

SPACE = {
    'gap': (0.025, 0.16, 'log'),        # contact A -> contact B
    'dir': (0.7, 1.45, 'log'),          # B pitch / A pitch (> 1: on, < 1: off)
    'a_lvl': (0.12, 2.5, 'log'),        # contact A level re B (> 1: the first contact is the loudest, round 3)
    'pitch': (1600.0, 8000.0, 'log'),
    'bright': (0.0, 1.0),
    'snap_ms': (0.15, 1.2, 'log'),
    'tone': (0.0, 1.0),
    'partials': (1, 3, 'int'),
    'inharm': (1.25, 2.8),
    'tink_ms': (1.5, 8.0, 'log'),       # <= 8 ms: longer rings drew narrow 3-8 kHz lines after B (round 1)
    'body': (0.01, 0.8, 'log'),         # contact body (both contacts: one patch, so on / off reads as pitch)
    'body_f': (150.0, 1200.0, 'log'),
    'body_ms': (2.0, 12.0, 'log'),      # <= 12 ms: longer bodies rang as narrow 400-500 Hz lines (round 1)
    'body_drop': (0.0, 0.5),
    'grit': (0.0, 0.6),
    'grit_ms': (2.0, 30.0, 'log'),
    'slide': (0.0, 0.12),               # friction noise between the contacts (0.3 swelled into a zip in round 1)
    'slide_f': (1500.0, 7000.0, 'log'),
    'spring': (0.0, 0.08),              # faint spring rattle after B (round 4: a short noisy rattle, <= 0.08; the round-3
    'spring_f': (2000.0, 6000.0, 'log'),  # tonal spring rang as the narrow 2.5 / 3.7 kHz lines of toggle_01 / 03)
    'room': (0.0, 0.3),
    'verb': (0.0, 0.35),                # a small dark room tail after B (refs t40 70-180 ms; round 1 stopped at 26-58)
    'rt': (0.06, 0.4, 'log'),
    'hp_steep': [0, 1],                 # round 4: 1 = a 4th-order high-pass at hp (round 3's 2nd order left 50-80 Hz)
    'hp': (80.0, 500.0, 'log'),
    'lp': (7000.0, 19000.0, 'log'),
    'deharsh': (0.0, 8.0),
    'deharsh_f': (2400.0, 4200.0, 'log'),
    'sat': (0.0, 8.0),
    'attack_ms': (0.05, 2.0, 'log'),
    'tail': (0.03, 0.2),                # B -> end gate
    'release_ms': (5.0, 35.0, 'log'),
    'width': (0.0, 0.25),
}
DEFAULTS = {'gap': 0.06, 'dir': 1.15, 'a_lvl': 0.45, 'pitch': 4500.0, 'bright': 0.5, 'snap_ms': 0.4, 'tone': 0.5,
            'partials': 2, 'inharm': 1.62, 'tink_ms': 5.0, 'body': 0.15, 'body_f': 500.0, 'body_ms': 6.0,
            'body_drop': 0.15, 'grit': 0.15, 'grit_ms': 6.0, 'slide': 0.05, 'slide_f': 3500.0, 'spring': 0.0,
            'spring_f': 3500.0, 'room': 0.08, 'verb': 0.12, 'rt': 0.15, 'hp': 150.0, 'lp': 17000.0, 'deharsh': 3.0,
            'deharsh_f': 3200.0, 'sat': 2.0, 'attack_ms': 0.3, 'tail': 0.06, 'release_ms': 12.0, 'width': 0.05, 'hp_steep': 0}

FAMILIES = {
    # round 3 (the round-2 judge: 150 Hz-1 kHz 14 dB too heavy, and contact B was the peak in all five while the held-out
    # switches peak on their first contact): bodies 0.01-0.1, pitch 2-4.5 kHz and a top roll-off at 9-14 kHz (forge refs:
    # 150 Hz-1 kHz -12.6, 1-4 kHz -4.4, 4-12 kHz -7.2, > 12 kHz -22 dB of total), and the first contact is the loudest
    # (a_lvl > 1) in four of five families; rocker keeps the softer-first idiom (2 of the 4 forge refs).
    # Round 4 (the round-3 judge: narrow ringing lines in 01 / 03 = the 18 ms tonal spring at 2.5 / 3.7 kHz; tone 8.8 ->
    # 10.4 dB with 50-80 Hz and 250-400 Hz bodies standing out; tails 1.7x long): the spring is a short noisy rattle
    # (hp_steep 1), a 4th-order high-pass, bodies at 450-1000 Hz (latch 320-600) and <= 4 ms, a smaller darker room
    # (verb <= 0.22, rt <= 0.2 s, early room <= 0.12: 0.3 drew a 2.2 ms comb) and B -> end 70-130 ms, so t40 lands ~80-110
    # ms (forge refs 75-180); tink tone <= 0.45 (0.66 read as a ding; on / off keep >= 0.2: B's noise band is capped by the
    # low-pass, so the tink carries the on-rises / off-falls pitch move); latch: B ~3 dB under A, so its tail carries the t40
    'toggle': {
        'on': {'dir': (1.12, 1.35, 'log'), 'gap': (0.045, 0.08, 'log'), 'a_lvl': (1.7, 2.5, 'log'),
               'pitch': (2400, 4200, 'log'), 'body': (0.015, 0.07, 'log'), 'body_f': (450, 900, 'log'),
               'body_ms': (2, 4, 'log'), 'tink_ms': (1.5, 3, 'log'), 'spring': (0.0, 0.04), 'hp_steep': 1,
               'tone': (0.2, 0.45), 'room': (0.0, 0.12), 'slide': (0.0, 0.06),
               'hp': (180, 400, 'log'), 'verb': (0.06, 0.22), 'rt': (0.08, 0.2, 'log'),
               'deharsh': (0.0, 2.0), 'lp': (9000, 14000, 'log'), 'tail': (0.07, 0.12)},
        'off': {'dir': (0.72, 0.88, 'log'), 'gap': (0.045, 0.08, 'log'), 'a_lvl': (1.7, 2.5, 'log'),
                'pitch': (2800, 4500, 'log'), 'body': (0.015, 0.07, 'log'), 'body_f': (450, 900, 'log'),
                'body_ms': (2, 4, 'log'), 'tink_ms': (1.5, 3, 'log'), 'spring': (0.0, 0.04), 'hp_steep': 1,
                'tone': (0.2, 0.45), 'room': (0.0, 0.12), 'slide': (0.0, 0.06),
                'hp': (180, 400, 'log'), 'verb': (0.06, 0.22), 'rt': (0.08, 0.2, 'log'),
                'deharsh': (0.0, 2.0), 'lp': (9000, 14000, 'log'), 'tail': (0.07, 0.12)},
        'latch': {'gap': (0.08, 0.12, 'log'), 'a_lvl': (1.15, 1.6, 'log'), 'body': (0.03, 0.08, 'log'),
                  'slide': (0.03, 0.08), 'tone': (0.1, 0.4), 'room': (0.0, 0.12),
                  'body_f': (320, 600, 'log'), 'body_ms': (2, 4, 'log'), 'pitch': (2000, 3500, 'log'),
                  'tink_ms': (1.5, 3, 'log'), 'hp_steep': 1, 'verb': (0.08, 0.2), 'rt': (0.08, 0.18, 'log'),
                  'hp': (150, 300, 'log'), 'spring': (0.02, 0.06), 'deharsh': (0.0, 2.0), 'lp': (8000, 12000, 'log'),
                  'tail': (0.09, 0.13)},
        'micro': {'gap': (0.025, 0.045, 'log'), 'a_lvl': (1.5, 2.3, 'log'), 'pitch': (3200, 4800, 'log'),
                  'body': (0.01, 0.04, 'log'), 'body_f': (600, 1100, 'log'), 'body_ms': (2, 4, 'log'),
                  'hp': (250, 500, 'log'), 'room': (0.0, 0.12),
                  'tink_ms': (1.5, 3, 'log'), 'slide': 0.0, 'spring': 0.0, 'hp_steep': 1, 'verb': (0.06, 0.2),
                  'rt': (0.07, 0.15, 'log'), 'lp': (10000, 13000, 'log'), 'tail': (0.07, 0.11)},
        'rocker': {'gap': (0.06, 0.1, 'log'), 'a_lvl': (0.45, 0.8, 'log'), 'pitch': (2000, 3500, 'log'),
                   'tone': (0.1, 0.35), 'body': (0.03, 0.09, 'log'), 'body_f': (400, 800, 'log'),
                   'tink_ms': (1.5, 3, 'log'), 'spring': (0.0, 0.03), 'hp_steep': 1, 'hp': (150, 350, 'log'),
                   'room': (0.0, 0.12),
                   'verb': (0.05, 0.2), 'rt': (0.08, 0.2, 'log'),
                   'body_ms': (2, 4, 'log'), 'deharsh': (0.0, 2.0), 'slide': (0.02, 0.08),
                   'lp': (8000, 12000, 'log'), 'tail': (0.08, 0.13)},
    },
}

def render(p, r):
    lvl = 10 ** (0.03 * r.standard_normal())
    sec = 0.45
    n = dsp.n_of(sec)
    pre = 0.002
    gap = p['gap'] * np.exp(0.04 * r.standard_normal())
    x = np.zeros(n)
    snd = np.zeros(n)                                     # broadband bus (transients + grit + slide) for the room
    a, a_s = click_layers(p, r, f_scale=1.0, level=p['a_lvl'] * lvl, send=True)   # one patch for A and B
    dsp.place(x, a, pre)
    dsp.place(snd, a_s, pre)
    if p['slide'] > 0.004:                                # friction swelling from A into B
        m = dsp.n_of(gap)
        s = dsp.butter(r.standard_normal(m), 'band', (p['slide_f'] / 1.6, min(16000, p['slide_f'] * 1.6)), order=2)
        s *= np.linspace(0, 1, m) ** 2
        dsp.place(x, p['slide'] * lvl * s / (np.abs(s).max() + 1e-12), pre)
        dsp.place(snd, p['slide'] * lvl * s / (np.abs(s).max() + 1e-12), pre)
    b, b_s = click_layers(dict(p, body_f=p['body_f'] * p['dir'] ** 0.5), r, f_scale=p['dir'], level=lvl, send=True)
    dsp.place(x, b, pre + gap)                                                                        # all of B moves
    dsp.place(snd, b_s, pre + gap)
    if p['spring'] > 0.004:
        if p.get('hp_steep', 0):                          # round 4: a rattle (noise + 5 ms partials), not a 18 ms ring
            f_s = p['spring_f'] * p['dir']                # part of B: it moves with B (on rises, off falls)
            sp = partials(f_s, [1.0, 1.37, 2.11], [1.0, 0.6, 0.35], 0.005, 0.06, r, att_s=0.001)
            nz = burst(0.006, r, f_s / 1.8, min(16000.0, f_s * 1.8), att_s=0.001, order=1)
            sp = 0.4 * sp + np.pad(nz, (0, max(0, len(sp) - len(nz))))[:len(sp)]
            sp = sp / (np.abs(sp).max() + 1e-12)
        else:
            sp = partials(p['spring_f'], [1.0, 1.37, 2.11], [1.0, 0.6, 0.35], 0.018, 0.12, r, att_s=0.001)
        dsp.place(x, p['spring'] * lvl * sp, pre + gap + 0.002)
    if p.get('hp_steep', 0):                              # round 4: 2nd-order hp here + finish's = 4th order at hp
        x = dsp.butter(x, 'high', p['hp'], order=2)       # (an hp after the tanh, tried in r4b, also removed the
    x = finish(x, p['hp'], p['deharsh'], p['deharsh_f'], p['sat'], lp=p['lp'])   # -25 dB < 150 Hz the forge refs have)
    snd = finish(snd, max(p['hp'], 300.0), p['deharsh'], p['deharsh_f'], 0.0, lp=p['lp'])
    st = space(x, r, p['room'], p['width'])
    st = room_tail(st, snd, r, p, size=0.15)
    st = gate(st, gap + p['tail'], p['release_ms'] * 1e-3, start=pre)
    st = st * (10 ** (-1.5 / 20) / (np.abs(st).max() + 1e-12))   # round 3: peak-normalise first, so the limiter does
    st = fx.limit(st, ceiling_db=-1.0)                            # not squash a louder first contact into the second
    return dsp.fade(st, 0.0005, 0.002)


def family_space(cat, fam):
    return narrow(SPACE, FAMILIES[cat][fam])


def family_params(cat, fam):
    return centre(DEFAULTS, FAMILIES[cat][fam])
