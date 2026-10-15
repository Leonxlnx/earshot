"""A soft premium UI button click, built from physics (final: the round-14 version).

What a real button click is, physically:
  1. the fingertip lands on the cap: a soft, slow (2-10 ms) force pulse into a heavily damped low mass-spring (finger
     pad + cap + travel) -> a dull low "thump" you mostly feel, heard as a low pressure bump; the actuator touching
     the dome / latch often makes a tiny precursor tick 1-6 ms before the snap;
  2. the dome / latch snaps: the force changes almost instantly (tens to hundreds of microseconds) -> a broadband
     impulse. A small sound source radiates pressure proportional to its acceleration, so the snap itself reaches the
     mic as a sharp click, and it rings the light hollow plastic shell. Close-up recordings of real clicks show ONE
     dominant, very damped shell mode (2-7 kHz, a handful of cycles: gone in 0.2-2 ms) over weaker modes, and the
     case / housing adds a few mid modes that are also gone within a few ms (a long-ringing ~1 kHz housing reads
     as a wooden block, not a button). A small object radiates low frequencies poorly (6 dB/oct below a
     size-dependent corner), so the click is light in the low end unless a finger thump is part of it;
  3. the contact chatters: the latch / dome micro-bounces once to three times within a few milliseconds (each a
     smaller impulse on the same bodies), and a millisecond or so later the stem bottoms out: a softer contact;
     lossy plastic also has a dense crowd of high, very damped modes that read as a short grainy texture, not tones;
  4. optionally the release: 20-90 ms later the dome springs back, a thinner, quieter click;
  5. friction and spring rattle leave a short band of noise; a small room and a close microphone (low cut, soft top)
     frame it, and a quiet click recorded close is never far above the preamp hiss: a broadband floor (white to pink)
     only 40-80 dB under the peak, with some low room tone (rumble) under it.
Two ears disagree here: the critic's description match prefers a duller, thumpier, tonal sound with a precursor tick,
a loud release and more room, while a held-out model (AudioSet) and the measured spectra put the real clicks
elsewhere (dense damped shell modes, a strong direct pulse and bottom-out, little room, quiet release). The ranges
below keep the search inside the region where the real recordings live.
(Round 9 tried a statistical body - 60-220 overlapping modes with a smooth envelope - instead of the hand-placed shell
and housing modes: smoother spectra, but the held-out tagger heard it as "crunch" and the critic liked it less, so
round 10 returns to this structure.) The refs' envelopes fall fast to about -40 dB and then hold a slow, bright tail
(small reflective room / case ring-down) for 50-150 ms, so the room may be wetter than the tagger alone suggested.
Round 10 measured close to the refs in envelope and tail balance, but its shell modes still drew clean lines
lasting 30-40 ms (heard as "coin dropping" / "ice cubes in a glass") and its finger thump sat at 60-250 Hz, where the
refs are quiet, leaving a hole at 300-600 Hz where they are not: round 11 damps the shell harder and stiffens the
finger / cap contact. Round 12 also lets the small-source radiation corner act on the finger thump (the cap
radiates it), which takes the remaining excess out of 50-250 Hz.
Round 13 passed the critic, but a diagnostic re-render showed WHY: its release click (40-45 ms, -18 dB) is what the
description match rewards ("a soft key release click"), and the same release made the held-out tagger place the
sounds nearer real mechanical keyboard presses than real UI clicks. The refs' second events come at 25-35 ms and
-25 to -30 dB, or not at all, so round 14 keeps the release early and quiet. Round 14 then measured as close to the
refs as the refs are to each other on every held-out check except the tail (-46 to -51 dB at 30-80 ms where the refs
sit near -40 dB for 50-150 ms). Round 15 forced more room and longer clips: the tails barely rose and the critic
dropped, so these round-14 ranges are the final ones (the finals in demo/forge-test/click/final come from them).
Real close-up clicks are BRIGHT and broadband: their first 30 ms have a plateau from 2 to 10 kHz and little below
150 Hz; a dull top, a strong sub thump or a single peaky mid mode is what made earlier rounds read as a key press, a
wooden block or a finger snap.
Every sample is computed; no recorded audio is used.
"""
import numpy as np

from .. import dsp
from . import REF_ROOT

TARGET = 'a soft premium UI button click'
ALTS = ['a mechanical keyboard key press', 'a wooden knock', 'a camera shutter click', 'a finger snap']
_K = REF_ROOT + '/ui/'
REFS = [_K + f for f in ('ui_click_01.wav', 'ui_click_02.wav', 'ui_click_03.wav', 'ui_click_04.wav',
                         'ui_toggle_01.wav', 'ui_toggle_02.wav')]

SPACE = {
    'thump_amp': (0.003, 0.12, 'log'),    # fingertip landing level (relative to the snap)
    'thump_f': (250, 600, 'log'),        # finger pad on a small hard cap: k ~ 1e4 N/m, m ~ 2 g -> ~350 Hz
    'thump_w': (0.0015, 0.005, 'log'),   # fingertip force pulse width
    'pre': (0.001, 0.012, 'log'),        # fingertip landing -> snap
    'snap_w': (0.00002, 0.00012, 'log'), # snap force pulse width: a raised cosine of width w has its first
                                         # spectral null at 2/w, so > 0.12 ms leaves the 6-12 kHz plateau dull
    'snap_bright': (0.0, 0.5),           # contact grit in the snap
    'direct': (0.4, 3.0, 'log'),         # directly radiated pulse vs body ringing
    'main_f': (2500, 8000, 'log'),       # the dominant, very damped shell mode
    'main_decay': (0.00015, 0.0008, 'log'),
    'main_amp': (0.3, 4.0, 'log'),       # its level against the other shell modes
    'tick_lvl': (0.0, 0.08),             # precursor tick (actuator meets the dome)
    'tick_dt': (0.001, 0.006, 'log'),    # ... this long before the snap
    'radiate_hz': (60, 1500, 'log'),     # small-source radiation corner (6 dB/oct below)
    'shell_f': (2000, 8000, 'log'),      # lowest of the other shell modes
    'shell_n': (7, 14, 'int'),           # how many shell modes
    'shell_decay': (0.0004, 0.0025, 'log'),
    'grain': (0.05, 1.2),                # dense, very damped high mode crowd (lossy plastic texture)
    'body_f': (400, 2500, 'log'),        # housing modes
    'body_decay': (0.0005, 0.003, 'log'),
    'body_amp': (0.02, 0.5, 'log'),
    'chatter': (0, 2, 'int'),            # micro-bounces of the latch after the snap
    'chatter_dt': (0.0006, 0.003, 'log'),
    'bottom_dt': (0.0003, 0.0015, 'log'),
    'bottom_lvl': (0.2, 0.6),
    'rel_dt': (0.02, 0.036),
    'rel_lvl': (0.0, 0.05),
    'tail_lvl': (0.005, 0.2, 'log'),     # friction / spring rattle / case ring-down noise
    'tail_tau': (0.004, 0.08, 'log'),
    'tail_f': (2000, 10000, 'log'),
    'tail_bw': (2.0, 6.0, 'log'),        # tail band half-width (ratio)
    'room_wet': (0.0, 0.3),
    'room_damp': (0.1, 0.8),
    'rt60': (0.05, 0.35, 'log'),
    'floor_db': (-72.0, -38.0),         # preamp hiss under the peak
    'hiss_tilt': (0.0, 1.0),             # 0 white .. 1 pink
    'rumble_db': (-80.0, -45.0),         # room tone: low rumble (traffic, HVAC) under the peak
    'clip_len': (0.07, 0.16),            # the edited clip: room tone runs on to here, then a short fade
}
DEFAULTS = {'main_f': 4500, 'main_decay': 0.0006, 'main_amp': 2.5, 'tick_lvl': 0.08, 'tick_dt': 0.003, 'radiate_hz': 300,
            'tail_bw': 1.8, 'room_damp': 0.6, 'thump_amp': 0.15, 'thump_f': 140, 'thump_w': 0.003, 'pre': 0.0015, 'snap_w': 0.0002, 'snap_bright': 0.5,
            'direct': 0.8, 'shell_f': 4200, 'shell_n': 10, 'shell_decay': 0.0018, 'grain': 0.5, 'body_f': 1500,
            'body_decay': 0.002, 'body_amp': 0.3, 'chatter': 1, 'chatter_dt': 0.0008, 'bottom_dt': 0.0015,
            'bottom_lvl': 0.5, 'rel_dt': 0.035, 'rel_lvl': 0.12, 'tail_lvl': 0.03, 'tail_tau': 0.012, 'tail_f': 1500,
            'room_wet': 0.1, 'rt60': 0.15, 'floor_db': -66.0, 'hiss_tilt': 0.5, 'rumble_db': -60.0, 'clip_len': 0.12}

PRE_ROLL = 0.003


def _pulse(width, r, bright=0.0):
    """a force pulse: raised cosine of `width` s, with optional contact grit (micro-roughness of the surfaces)"""
    n = max(2, dsp.n_of(width))
    p = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, n))
    if bright > 0:
        p = p + bright * r.standard_normal(n) * p
    return p / (p.sum() + 1e-12) * n  # unit-ish peak, energy scales with width


def _bodies(p, r, f_scale=1.0, seconds=0.12):
    """impulse response of the shell (bright, few, damped modes) + housing (low-mid modes)"""
    n = int(p['shell_n'])
    # plastic shell: modes roughly following a plate-like spacing, higher ones die faster
    ratios = np.array([1.0, 1.52, 2.03, 2.61, 3.18, 3.9, 4.57, 5.3, 6.1, 6.9, 7.8, 8.7, 9.7, 10.8])[:n]
    ratios = ratios * np.exp(0.06 * r.standard_normal(n))
    fs = p['shell_f'] * f_scale * ratios
    dec = p['shell_decay'] * ratios ** -0.7
    amps = r.uniform(0.4, 1.0, n) * ratios ** -0.5
    shell = dsp.modal(fs, dec, amps, seconds, r, jitter=0.0)
    main = dsp.modal([p['main_f'] * f_scale], [p['main_decay']], [1.0], seconds, r, jitter=0.01)
    shell = shell / (np.abs(shell).max() + 1e-12) + p['main_amp'] * main
    hf = p['body_f'] * f_scale * np.array([1.0, 1.71, 2.47])
    body = dsp.modal(hf, p['body_decay'] * np.array([1.0, 0.6, 0.4]), np.array([1.0, 0.5, 0.3]) * p['body_amp'],
                     seconds, r)
    out = shell + body * (np.abs(shell).max() / (np.abs(body).max() + 1e-12)) * min(1.0, p['body_amp'])
    if p['grain'] > 0:
        cloud = dsp.modal_cloud(p['shell_f'] * f_scale * 0.8, 16000, 40, 0.0015, 0.0003, seconds, r, tilt=-2.0)
        out = out + p['grain'] * cloud / (np.abs(cloud).max() + 1e-12) * np.abs(shell).max()
    return dsp.butter(out, 'high', p['radiate_hz'], order=1)   # small source: poor low-frequency radiation


def _snap(p, r, level=1.0, f_scale=1.0, width_scale=1.0):
    """the snap + bottom-out on the bodies, plus the directly radiated pulse"""
    seconds = 0.12
    ir = _bodies(p, r, f_scale, seconds)
    exc = _pulse(p['snap_w'] * width_scale, r, p['snap_bright'])
    x = dsp.excite(ir, exc)
    x = x / (np.abs(x).max() + 1e-12)
    d = np.zeros_like(x)
    d[:len(exc)] += exc / (np.abs(exc).max() + 1e-12)
    d = dsp.butter(d, 'high', max(250, p['radiate_hz']), order=1)   # small source: pressure ~ acceleration
    out = x + p['direct'] * d
    t, lvl = 0.0, 1.0
    for _ in range(int(p['chatter'])):   # micro-bounces: each later and weaker than the last
        t += p['chatter_dt'] * r.uniform(0.6, 1.4)
        lvl *= r.uniform(0.25, 0.6)
        dsp.place(out, lvl * x, t)
        dsp.place(out, lvl * p['direct'] * d, t)
    if p['bottom_lvl'] > 0:
        exc2 = _pulse(p['snap_w'] * width_scale * 2.5, r, p['snap_bright'] * 0.5)
        y = dsp.excite(_bodies(p, r, f_scale * 0.92, seconds), exc2)
        y = y / (np.abs(y).max() + 1e-12)
        dsp.place(out, p['bottom_lvl'] * y, p['bottom_dt'])
    return level * out


def _thump(p, r):
    """fingertip landing: slow force pulse into a heavily damped low mass-spring, heard as a low pressure bump"""
    exc = _pulse(p['thump_w'], r, 0.0)
    ir = dsp.modal([p['thump_f']], [1.0 / (np.pi * p['thump_f'] * 1.2)], [1.0], 0.06, r, jitter=0.0)
    y = dsp.excite(ir, exc)
    return y / (np.abs(y).max() + 1e-12)


def _tail(p, r):
    """friction and spring rattle after the snap: band-passed noise with an exponential decay"""
    sec = min(0.25, p['tail_tau'] * 7)
    nz = dsp.noise(sec, r, 'white') * dsp.env_exp(sec, p['tail_tau'], 0.0003)
    lo, hi = p['tail_f'] / p['tail_bw'], min(21000, p['tail_f'] * p['tail_bw'])
    return dsp.butter(nz, 'band', (lo, hi), order=2)


def render(p, r):
    x = np.zeros(dsp.n_of(0.35))
    t0 = PRE_ROLL + p['pre']
    th = dsp.butter(_thump(p, r), 'high', p['radiate_hz'], order=1)   # the cap is a small radiator too
    dsp.place(x, p['thump_amp'] * th / (np.abs(th).max() + 1e-12), PRE_ROLL)
    if p['tick_lvl'] > 0.005:
        dsp.place(x, _snap(dict(p, chatter=0, bottom_lvl=0.0), r, p['tick_lvl'], f_scale=1.07, width_scale=1.5),
                  max(0.0, t0 - p['tick_dt']))
    s = _snap(p, r)
    dsp.place(x, s, t0)
    dsp.place(x, p['tail_lvl'] * _tail(p, r), t0)
    if p['rel_lvl'] > 0.01:
        dsp.place(x, _snap(p, r, p['rel_lvl'], f_scale=1.15, width_scale=0.7), t0 + p['rel_dt'])
    x = dsp.room(x, r, rt60=p['rt60'], size=0.2, damp=p['room_damp'], wet=p['room_wet'])
    x = dsp.trim_tail(x, -60, 0.02)   # the sound ends where it falls under the floor ...
    if len(x) < dsp.n_of(p['clip_len']):   # ... but the recording (room tone, hiss) runs on to the edit point
        x = np.pad(x, (0, dsp.n_of(p['clip_len']) - len(x)))
    x = dsp.mic(x, r, low_cut=40.0, air_db=-1.0, floor_db=-120.0)
    pk = np.abs(x).max()   # ... then the preamp adds its own hiss
    h = (1 - p['hiss_tilt']) * dsp.noise(len(x) / dsp.SR, r, 'white') + p['hiss_tilt'] * dsp.noise(len(x) / dsp.SR, r, 'pink')
    x = x + h / (np.sqrt(np.mean(h ** 2)) + 1e-12) * pk * 10 ** (p['floor_db'] / 20)
    rb = dsp.butter(dsp.noise(len(x) / dsp.SR, r, 'brown'), 'band', (30, 220), order=2)
    x = x + rb / (np.sqrt(np.mean(rb ** 2)) + 1e-12) * pk * 10 ** (p['rumble_db'] / 20)
    return dsp.fade(x, 0.0003, 0.02)
