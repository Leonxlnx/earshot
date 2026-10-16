"""A deep "thocky" mechanical keyboard key press, built from physics (the model after 15 forge rounds; demo/forge-test/
keypress/LOG.md has the round-by-round story and what earshot heard). Rounds 13-15 (after the judge) fixed the tonal
balance measured model-free against the real keys: the case body moved below 260 Hz (it was a +16 dB boxy 'tok' at
500-630 Hz), the 8-10 kHz spring ping came back, the high shell modes start above the refs' 4-6 kHz dip, the finger's
push is shorter and a session high-pass removes the 15-45 Hz shove: 8-13 dB from the real keys' median balance before,
5.4-5.6 dB after (the real keys sit 3.1-6.0 dB from each other).

What the real Keychron references show (measured in demo/forge-test/keypress/analyze_refs.py, lowwave.py, and seen on
their waveforms / spectrograms), and how each part is modelled here:

1. Bottom-out contact (t = 0). The stem / keycap slams into the bottom housing and plate in well under a millisecond.
   - an instantaneous full-band click (the refs' onset is one vertical line up to 20 kHz; 7-14 kHz is 20 dB down after
     ~2 ms): a broadband crack plus a dense high shell-mode cloud, driven by a half-sine Hertzian contact pulse whose
     length (click_tc) sets how far up the spectrum it reaches; a direct pressure snap (the first big asymmetric spike);
   - the keycap / plate modes: a main mode near 1.25-1.6 kHz plus a cluster up to ~2.3x, mixed with band noise (a
     cluster too dense to resolve). The finger keeps the stem pressed, so the cap rings freely for ~1 ms and is then
     clamped to a residual (cap_fast / cap_res: time-varying damping). Longer cap rings (tau > ~4.5 ms) read as a
     dropped coin to AudioSet, so cap_tau is capped there;
   - the case body (110-260 Hz since round 13; the refs dip at 250-630 Hz) rings a little longer.
   Part levels are energy ratios over the first 5 ms (as the refs' band energies are measured), not peak ratios.
2. The "thock": a 58-77 Hz low wave that swells to its maximum 20-35 ms after the hit, beats (dips near 45 and 85 ms)
   and dies with tau ~50-60 ms - the board + desk moving on its feet (thock_f) beating with the case's own low mode
   (case_f), driven by the finger's smooth push (thock_rise). The mic hears pressure, which follows acceleration, so
   the quasi-static push itself is high-passed away.
3. The release 70-125 ms later: two small high ticks (the stem leaving the bottom housing / the spring snapping off its
   seat, tick_dt before the top-out), then the top-out - a smaller, brighter contact with the cap modes shifted, and a
   small kick to the low wave. A probe showed the critic hears the ticks as essential (without them typicality -4).
4. Pre-travel friction (a faint swell before the hit), a spaced stereo pair in a small room (removing the room drops
   typicality to -10: the refs are close-mic'd but not dry; a sparse-echo or a diffuse room model), a noise floor (low
   rumble + hiss rolled off near 18-20 kHz) and the kit's edit: 0.2-0.4 s after the hit, faded.
The modal layout and the mic / room geometry belong to the board and the session, so they come from the `key`
parameter (fixed per key); each press only varies force, timing and noise - the same key sounds the same twice.

v2 (forge v2, overnight 2026-10-16; the thock model above is unchanged): the v1 finals were clones (CLAP cosine
0.965 between variants, real sets 0.67-0.72), their tails ran long (t40 1.67 x the held-out median) and their onset
was soft and their top dull (4-12 kHz -18 dB re total where the clicky forge refs sit at -3..-14 dB; the loudest moment
was the low thock, not the contact). New keys, only read when present (parameter sets without them render exactly the
v1 sound):
  top_lvl / top_lo / top_tau   the keycap's broadband rattle at the contact: a 2-12 kHz noise burst (the refs' onset is
                               a vertical line with most of its energy at 2-10 kHz), energy relative to the cap modes
  jacket_lvl / jacket_dt       a clicky switch's click jacket: a bright pre-click 3-12 ms before the bottom-out (the
                               clicky refs show two onsets ~10 ms apart); 0 = a linear / tactile switch
  pre_gap                      the lead before the key travel (v1: a random 3-20 ms): the file starts on the attack
  clip                         down to 0.12 s (the clicky refs are 0.11-0.28 s)
Character for families: thock_lvl + case_lvl (deep thock vs light clack), cap_f (pitch), top_lvl + click_lvl (crisp
top), jacket_lvl (clicky vs linear), up_lvl (release), clip + thock_tau (length), room_wet (dry vs roomy).

v2 round 3: `width` (read only when present) folds the spaced pair towards mono, mid + width x side: the r2 finals'
pair decorrelated the 2-12 kHz layer to corr 0.04-0.47 where the forge refs sit at 0.53-1.0 (median 0.76) and the
held-out keys are mono; DEFAULTS use 0.2 (corr ~0.9-0.98). The round-3 family also runs ~5 dB less thock (thock_lvl
x0.45-0.5: < 150 Hz share median -10.2 dB, refs -12.6, r2 -5.5).

No recorded audio is used: every sample is computed.
"""
import numpy as np
from scipy import signal

from .. import dsp
from . import REF_ROOT

SR = dsp.SR
TARGET = 'a single mechanical keyboard key press with a crisp click and a short thock'
ALTS = ['a wooden knock', 'a computer mouse click', 'a typewriter key', 'a cork pop']
_K = REF_ROOT + '/keyboard/'
REFSET = 'typing_key'   # the CLI's default refs: this refsets category, 'forge' split (bench held out)
REFS = [_K + f for f in ('kb_thock_01.wav', 'kb_thock_02.wav', 'kb_thock_03.wav', 'kb_thock_04.wav',
                         'kb_enter_01.wav', 'kb_space_01.wav')]

# ranges marked (refs) are limits measured on the references, set after the search exploited regions no real
# recording occupies (LOG.md rounds 5, 6, 12)
SPACE = {
    'click_tc': (0.00003, 0.0004, 'log'),   # contact time of the bottom-out (sets how high the click reaches)
    'click_tau': (0.00015, 0.001, 'log'),   # high shell modes' decay (refs: 7-14 kHz 20 dB down in ~2 ms)
    'click_lvl': (0.1, 2.0, 'log'),         # click energy relative to the cap modes (first 5 ms)
    'click_lo': (6500, 9500, 'log'),        # lowest of the high shell modes (round 13: the refs dip at 4-6.3 kHz)
    'click_tilt': (-3.0, 3.0),              # their spectral tilt, dB / octave
    'crack': (0.2, 2.0, 'log'),             # broadband crack relative to the high modes
    'snap': (0.0, 1.0),                     # direct pressure spike
    'cap_f': (900, 1800, 'log'),            # main keycap / plate mode
    'cap_tau': (0.0015, 0.0045, 'log'),     # (refs: 1.2-3 kHz 20 dB down in 6-10 ms)
    'cap_noise': (0.0, 1.0),                # dense-cluster share (band noise) vs discrete modes
    'cap_n': (6, 30, 'int'),
    'cap_cl': (0.1, 1.0, 'log'),            # the cluster's level under the main cap mode
    'cap_fast': (0.0005, 0.003, 'log'),     # free ring before the stem clamps the cap
    'cap_res': (0.05, 1.0, 'log'),          # residual ring after clamping (1 = no clamping)
    'mid_f': (110, 260, 'log'),             # case body mode (rounds 13-15: the refs dip at 250-630 Hz and are full
    'mid_tau': (0.003, 0.016, 'log'),       #  at 100-200 Hz; at 337-615 Hz it was a +16 dB boxy 'tok', at 200-340
    'mid_lvl': (0.01, 0.2, 'log'),          #  Hz still +10 dB at 250 Hz). Energy relative to the cap
    'thock_f': (55, 90, 'log'),             # the low board + desk mode (refs: 58-77 Hz)
    'thock_tau': (0.045, 0.1, 'log'),       # (refs: tau ~50-60 ms)
    'thock_lvl': (0.05, 0.8, 'log'),        # its amplitude relative to the click peak
    'thock_rise': (0.001, 0.025, 'log'),    # length of the finger's push (round 13: longer pushes put -9..-12 dB
                                            #  at 15-45 Hz, a sub-sonic shove the refs do not have)
    'case_f': (70, 220, 'log'),             # the case's own low mode (beats with the board mode)
    'case_lvl': (0.0, 0.8),
    'case_dec': (0.3, 1.0),                 # its decay relative to the board mode's
    'up_delay': (0.07, 0.125),              # (refs: top-out 75-120 ms after the hit)
    'up_lvl': (0.08, 0.35, 'log'),          # (refs: 10-22 dB under the bottom-out)
    'up_pitch': (0.9, 1.5),
    'tick_lvl': (0.0, 1.0),                 # release ticks before the top-out
    'tick_dt': (0.004, 0.025, 'log'),       # how long before the top-out they come
    'pre_len': (0.004, 0.025, 'log'),       # key travel before the bottom-out (friction swell)
    'pre_lvl': (0.001, 0.06, 'log'),
    'ping_f': (7500, 11000, 'log'),         # spring ping (round 13: the refs' 8-10 kHz line, 10-40 ms; it was
    'ping_tau': (0.004, 0.03, 'log'),       #  muted or at 6-7 kHz, -14 dB at 10 kHz)
    'ping_lvl': (0.04, 0.3),
    'room_wet': (0.0, 0.4),
    'rt60': (0.05, 0.4, 'log'),
    'room_size': (0.0, 1.0),
    'room_damp': (0.2, 0.95),
    'room_model': ['sparse', 'diffuse'],    # dsp.room (6 discrete early echoes + tail) or a diffuse field
    'floor_db': (-78.0, -55.0),             # (refs: tails at -53..-77 dB re peak)
    'rumble': (0.0, 0.7),                   # low-frequency share of the noise floor
    'clip': (0.12, 0.4),                    # clip length after the hit (the kit's edits: 0.2-0.39 s; v2: clicky refs
                                            #  0.11-0.28 s)
    'hpf': (36, 46, 'log'),                 # the session's high-pass (round 13: the refs are -38 / -47 dB at 39 /
    'key': (0, 999, 'int'),                 #  31 Hz; the slow push and the rumble put +16-39 dB there). key: see below
    # v2 (only read when present)
    'top_lvl': (0.0, 2.5),                  # keycap rattle at the contact, 2-12 kHz band noise (energy re the cap)
    'top_lo': (1800, 4500, 'log'),
    'top_tau': (0.0006, 0.004, 'log'),
    'jacket_lvl': (0.0, 0.9),               # click jacket pre-click (0 = linear / tactile switch)
    'jacket_dt': (0.003, 0.012, 'log'),
    'pre_gap': (0.0005, 0.006, 'log'),      # lead before the key travel (v1: 3-20 ms)
    # v2 round 3 (only read when present)
    'width': (0.0, 0.6),                    # side level of the spaced pair (1 = the v1 / r2 pair as recorded)
}
V1_DEFAULTS = None   # set below: the v1 final (round 15 pick 4), kept for reference / the seed bank
# the most reliable sound after the judge (round 15 pick 4 = final keypress_01, parameters rounded to 6 digits): 8 new
# seeds -> critic mean 6.74 with the model-free tone check in the total (Critic tone=1.5), 7/8 pass, 5.6 dB from the
# real keys' median balance
DEFAULTS = {'click_tc': 7.62977e-05, 'click_tau': 0.000400034, 'click_lvl': 0.195527, 'click_lo': 7323.18,
            'click_tilt': -1.25571, 'crack': 0.395643, 'snap': 0.0, 'cap_f': 1489.18, 'cap_tau': 0.00399449,
            'cap_noise': 1.0, 'cap_n': 7, 'cap_cl': 0.357156, 'cap_fast': 0.000995191, 'cap_res': 0.749554,
            'mid_f': 205.793, 'mid_tau': 0.00918634, 'mid_lvl': 0.0902506, 'thock_f': 86.8567,
            'thock_tau': 0.0549253, 'thock_lvl': 0.226067, 'thock_rise': 0.00834742, 'case_f': 70.0,
            'case_lvl': 0.581134, 'case_dec': 0.736866, 'up_delay': 0.124575, 'up_lvl': 0.0969741,
            'up_pitch': 1.04402, 'tick_lvl': 0.581597, 'tick_dt': 0.0111911, 'pre_len': 0.0128195,
            'pre_lvl': 0.0207414, 'ping_f': 9083.43, 'ping_tau': 0.00578368, 'ping_lvl': 0.122274,
            'room_wet': 0.261726, 'rt60': 0.113685, 'room_size': 0.212985, 'room_damp': 0.95, 'room_model': 'sparse',
            'floor_db': -55.0, 'rumble': 0.0177183, 'clip': 0.38353, 'hpf': 44.5337, 'key': 999}
V1_DEFAULTS = dict(DEFAULTS)
# v2 start: the v1 key with a crisp top, a short edit and the file starting on the attack (the search tunes from here)
DEFAULTS = {**V1_DEFAULTS, 'click_lvl': 0.6, 'thock_tau': 0.05, 'clip': 0.24, 'top_lvl': 0.8, 'top_lo': 2500.0,
            'top_tau': 0.0015, 'jacket_lvl': 0.0, 'jacket_dt': 0.008, 'pre_gap': 0.002, 'pre_len': 0.006,
            'width': 0.2}        # r3: near-mono (the r2 pair without `width` sat at corr 0.04-0.47; refs 0.53-1.0)


# ------------------------------------------------------------------------------------------------ helpers
def _t(n):
    return np.arange(n) / SR


def _key_rng(p, part):
    """the key's (and the session's) fixed physical layout: mode frequencies and mic reflections belong to this board,
    not to one press, so they come from the `key` parameter; the per-press generator r only adds natural variation"""
    return dsp.rng(int(p['key']) * 7919 + part * 104729 + 17)


def _ring(freqs, taus, amps, n, r):
    """damped sinusoids, each computed only over ~9 time constants (cheap for short plastic modes)"""
    out = np.zeros(n)
    t = _t(n)
    for f, d, a in zip(freqs, taus, amps):
        if a == 0 or f >= SR * 0.47:
            continue
        m = min(n, int(9 * d * SR) + 2)
        out[:m] += a * np.exp(-t[:m] / d) * np.sin(2 * np.pi * f * t[:m] + r.uniform(0, 2 * np.pi))
    return out


def _cloud(f_lo, f_hi, count, tau_lo, tau_hi, n, kr, r, tilt=-3.0, detune=0.002):
    """a dense set of modes, log-spread (layout from kr: the object's own modes); higher ones decay faster; tilt
    dB/oct; each press detunes them a hair (contact point, temperature)"""
    f = np.sort(np.exp(kr.uniform(np.log(f_lo), np.log(f_hi), int(count))))
    a = 10 ** (tilt * np.log2(f / f_lo) / 20) * kr.uniform(0.3, 1.0, len(f))
    f = f * (1 + detune * r.standard_normal(len(f)))
    u = np.log(f / f_lo) / max(1e-9, np.log(f_hi / f_lo))
    d = np.exp(np.log(tau_lo) + np.clip(u, 0, 1) * (np.log(tau_hi) - np.log(tau_lo)))
    return _ring(f, d, a, n, r)


def _hiss(lo, hi, tau, n, r, order=2):
    """band noise with an exponential decay (a mode cluster too dense to resolve)"""
    m = min(n, int(9 * tau * SR) + 2)
    w = dsp.butter(r.standard_normal(m), 'band', (lo, min(hi, SR * 0.45)), order)
    out = np.zeros(n)
    out[:m] = w * np.exp(-_t(m) / tau)
    return out


def _pulse(tc):
    """half-sine contact force of duration tc, unit area"""
    k = max(1, int(round(tc * SR)))
    q = np.sin(np.pi * (np.arange(k) + 0.5) / k)
    return q / q.sum()


def _drive(ir, tc):
    return np.convolve(ir, _pulse(tc))[: len(ir)]


def _unit(x):
    return x / (np.abs(x).max() + 1e-12)


def _en(x, ms=5.0):
    """scale to unit RMS over the first `ms` milliseconds (levels between the parts of a hit are energy ratios, as
    measured in the refs' bands, not peak ratios: a noise burst's peak says little about how loud it sounds)"""
    m = max(8, dsp.n_of(ms / 1000))
    return x / (np.sqrt(np.mean(x[:m] ** 2)) + 1e-12)


# ------------------------------------------------------------------------------------------------ the parts
def _crack(r, n, tau, tc):
    """the impact's own pressure transient: an extremely short broadband burst, flat up to 20 kHz (the refs' onset is
    one vertical line across the whole spectrum)"""
    m = min(n, max(dsp.n_of(9 * tau), 8))
    w = r.standard_normal(m) * np.exp(-_t(m) / tau)
    out = np.zeros(n)
    out[:m] = dsp.butter(w, 'high', 1200, 1)
    return _drive(out, tc)


def _contact(p, r, n, part, pitch=1.0, bright=1.0):
    """one plastic contact (part 0 bottom-out, 1 top-out): crack + high shell modes + snap + cap modes + mid body"""
    kr = _key_rng(p, part)
    tc = p['click_tc'] * float(np.exp(0.15 * r.standard_normal()))          # a press is never exactly the same
    hf = _cloud(p['click_lo'], 18000, 40, p['click_tau'] * 1.6, p['click_tau'] * 0.6, n, kr, r, tilt=p['click_tilt'])
    hf = _drive(_unit(hf), tc)
    crack = _unit(_crack(r, n, p['click_tau'] * 0.4, tc * 0.5))
    cf = p['cap_f'] * pitch
    modes = _ring([cf], [p['cap_tau'] * 1.2], [1.0], n, r) + p['cap_cl'] * \
        _unit(_cloud(cf * 1.08, cf * 2.3, p['cap_n'], p['cap_tau'] * 0.9, p['cap_tau'] * 0.45, n, kr, r))
    band = _hiss(cf * 0.75, cf * 2.4, p['cap_tau'] * 0.8, n, r)
    cap = (1 - p['cap_noise']) * _en(modes) + p['cap_noise'] * _en(band)
    # the finger keeps the stem pressed on the bottom housing: the cap rings freely for a millisecond or two, then it
    # is clamped and only a residual rings on (refs: the dense 1-3 kHz burst dies within ~3 ms, a softer ring follows;
    # measured attack 1 ms = the onset energy is front-loaded). Time-varying damping = a gain envelope on the cap.
    cap = cap * (p['cap_res'] + (1 - p['cap_res']) * np.exp(-_t(n) / p['cap_fast']))
    cap = _drive(cap, max(tc, 0.00012))
    mf = p['mid_f'] * pitch
    ratios = [1.0, kr.uniform(1.35, 1.6), kr.uniform(1.9, 2.4)]
    mid = _ring([mf * q for q in ratios], [p['mid_tau'], p['mid_tau'] * 0.7, p['mid_tau'] * 0.5], [1, .5, .3], n, r)
    k = max(3, int(kr.uniform(0.25, 0.6) * 1e-3 * SR))
    snap = np.zeros(n)
    snap[:k] = -np.hanning(k)
    x = bright * p['click_lvl'] * _en(_unit(hf) + p['crack'] * crack) + _en(cap) + p['mid_lvl'] * _en(mid)
    if p.get('top_lvl', 0.0) > 0:      # v2: the keycap's broadband rattle (2-12 kHz) at the contact
        top = _drive(_hiss(p['top_lo'], 12000, p['top_tau'], n, r, order=2), tc)
        x = x + bright * p['top_lvl'] * _en(top)
    return _unit(_unit(x) + p['snap'] * snap)


def _ping(p, r, n, f_mul=1.0):
    """the spring's ping / the plate's high ring: a few close high modes (8-10 kHz line in the refs, 10-40 ms long)"""
    f = p['ping_f'] * f_mul
    return _unit(_ring([f, f * 1.013, f * 1.62], [p['ping_tau'], p['ping_tau'] * 0.8, p['ping_tau'] * 0.4],
                       [1.0, 0.5, 0.1], n, r))


def _low(p, r, n):
    """the board + desk moving as a mass on its feet (thock_f) plus the case's own low mode (case_f), driven by the
    finger's push: a smooth force of thock_rise seconds, so the low wave swells after the click instead of starting at
    full size. The refs' low wave (lowwave.py) is 58-77 Hz, swells to its maximum 20-35 ms after the hit and then beats
    (dips near 45 and 85 ms): two close low modes, both dying with tau ~50 ms. The mic hears pressure, which follows
    the surface's acceleration, not its displacement: the quasi-static push itself is inaudible (high-pass)."""
    t = _t(n)
    f = p['thock_f'] * (1 + 0.01 * r.standard_normal())
    ir = np.exp(-t / p['thock_tau']) * np.sin(2 * np.pi * f * t)
    ir += p['case_lvl'] * np.exp(-t / (p['thock_tau'] * p['case_dec'])) * np.sin(2 * np.pi * p['case_f'] * t + 0.5)
    k = max(2, dsp.n_of(p['thock_rise']))
    force = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, k))
    y = np.convolve(ir, force / force.sum())[:n]
    return _unit(dsp.butter(y, 'high', 0.75 * min(f, p['case_f']), order=2))


def _pretravel(p, r, n):
    """the finger riding the cap down and the stem sliding in its housing: a faint friction noise that swells over
    the travel and stops at the bottom-out (refs sit at -45..-28 dB in the 5-25 ms before the hit)"""
    m = dsp.n_of(p['pre_len'])
    t = np.linspace(0, 1, m)
    w = dsp.butter(r.standard_normal(m), 'band', (700, 9000), 2)
    grit = (r.uniform(0, 1, m) < 0.004) * r.standard_normal(m) * 4.0      # stick-slip micro clicks
    w = (w + dsp.butter(grit, 'high', 2000, 1)) * t ** 2
    out = np.zeros(n)
    out[:m] = w / (np.abs(w).max() + 1e-12)
    return out


def _diffuse_room(x, kr, p):
    """a small furnished room as a diffuse field: no sparse discrete echoes (those comb-filter into a metallic, boxy
    ring: AudioSet heard a dropped coin in the sparse dsp.room version), but one desk reflection, then a smooth
    build-up of dense reflections (room_size sets how slowly it builds) and a tail whose highs die faster than its
    lows (room_damp: soft furnishings absorb highs)"""
    rt = p['rt60']
    m = dsp.n_of(min(0.6, rt * 1.1 + 0.03))
    t = _t(m)
    w = kr.standard_normal(m)
    ramp = 0.0015 + 0.008 * p['room_size']
    onset = (1 - np.exp(-t / ramp)) * (t > 0.0008)
    ir = np.zeros(m)
    for band, k in ((dsp.butter(w, 'low', 700, 2), 1.0), (dsp.butter(w, 'band', (700, 4000), 2), 0.75),
                    (dsp.butter(w, 'high', 4000, 2), 0.25)):
        rtb = max(0.02, rt * (1 - p['room_damp'] * (1 - k)))
        ir += band * np.exp(-6.91 * t / rtb)
    ir *= onset
    d = int(kr.uniform(0.8e-3, 2.5e-3) * SR)                   # the desk under the keyboard
    ir[d] += 0.35 * np.sqrt(np.sum(ir ** 2))
    ir = dsp.butter(ir, 'low', 9000, 1) * 0.6 + ir * 0.4
    y = signal.fftconvolve(x, ir)[: len(x)]
    return x + p['room_wet'] * y / (np.sqrt(np.sum(ir ** 2)) + 1e-9)


def _spaced_pair(x, p, r):
    """two close mics a few cm apart in a small room (all fixed for the session, so from the key rng): each sees the
    source with its own tiny delay and weak early reflections (the desk, the case), plus the room's own response.
    The probe (probe.py) showed the room is not optional: the same press without it lands at typicality z -10."""
    kr = _key_rng(p, 9)
    chans = []
    for c in range(2):
        ir = np.zeros(int(0.007 * SR))
        ir[int(kr.uniform(0, 0.25e-3) * SR)] = 1.0
        refl = np.zeros_like(ir)
        for _ in range(6):
            refl[int(kr.uniform(0.3e-3, 6.5e-3) * SR)] += kr.uniform(0.03, 0.12) * kr.choice([-1, 1])
        ir += dsp.butter(refl, 'low', 12000, order=1)
        y = np.convolve(x, ir)[: len(x)]
        if p['room_model'] == 'diffuse':
            y = _diffuse_room(y, _key_rng(p, 10 + c), p)
        else:
            y = dsp.room(y, _key_rng(p, 10 + c), rt60=p['rt60'], size=p['room_size'], damp=p['room_damp'],
                         wet=p['room_wet'])[: len(x)]
        chans.append(y)
    return np.stack(chans, axis=1)


def _floor(n, r, rumble, rum_lo=25):
    """the recording's own noise: low rumble + near-white hiss rolled off around 18-20 kHz (unit RMS)"""
    hiss = dsp.noise(n / SR, r, 'pink') + dsp.noise(n / SR, r, 'white')
    hiss = dsp.butter(hiss, 'low', r.uniform(17000, 20000), order=4)
    rum = dsp.butter(dsp.noise(n / SR, r, 'brown'), 'band', (rum_lo, 120), order=2)
    y = (1 - rumble) * hiss / (np.std(hiss) + 1e-12) + rumble * 2.0 * rum / (np.std(rum) + 1e-12)
    return y / (np.std(y) + 1e-12)


# ------------------------------------------------------------------------------------------------ render
def render(p, r):
    v2 = 'top_lvl' in p
    pre = p['pre_len'] + (p['pre_gap'] * float(np.exp(0.2 * r.standard_normal())) if v2 else r.uniform(0.003, 0.02))
    if v2 and p['jacket_lvl'] > 0.005:
        pre += p['jacket_dt']
    dur = pre + p['clip'] * (1 + 0.05 * r.standard_normal())      # the kit's clips are edited and faded
    n = dsp.n_of(dur + 0.05)
    nh = dsp.n_of(0.12)
    force = float(np.exp(0.1 * r.standard_normal()))        # this press's strength (brightness follows force)
    x = np.zeros(n)
    dsp.place(x, _pretravel(p, r, dsp.n_of(p['pre_len'])), pre - p['pre_len'], p['pre_lvl'])
    if v2 and p['jacket_lvl'] > 0.005:  # a clicky switch: the click jacket snaps a few ms before the bottom-out
        jk = _drive(_unit(_cloud(2500, 15000, 24, 0.0012, 0.0004, nh, _key_rng(p, 6), r, tilt=0.0)), 0.00004)
        dsp.place(x, jk, max(0.0, pre - p['jacket_dt'] * (1 + 0.06 * r.standard_normal())), p['jacket_lvl'])
    dsp.place(x, _contact(p, r, nh, 0, bright=force), pre)
    dsp.place(x, _ping(p, r, nh), pre, p['ping_lvl'])
    dsp.place(x, _low(p, r, n - dsp.n_of(pre)), pre, p['thock_lvl'] * force)
    tu = pre + p['up_delay'] * (1 + 0.04 * r.standard_normal())
    if p['tick_lvl'] > 0.005:          # the stem leaving the bottom housing / the spring snapping off its seat: two
        kr = _key_rng(p, 5)             # small high ticks 5-25 ms before the top-out (refs: ~80 + ~88 ms, up at ~97)
        t1 = tu - p['tick_dt'] * (1 + 0.08 * r.standard_normal())
        for j, (dt, g) in enumerate([(0.0, 1.0), (r.uniform(0.003, 0.009), r.uniform(0.4, 1.0))]):
            tick = _drive(_unit(_cloud(3000, 14000, 14, 0.0008, 0.0003, nh, kr, r)), 0.00005)
            dsp.place(x, tick, min(t1 + dt, tu - 0.001), g * p['tick_lvl'] * p['up_lvl'])
    dsp.place(x, _contact(p, r, nh, 1, pitch=p['up_pitch'], bright=1.2), tu, p['up_lvl'])
    dsp.place(x, _ping(p, r, nh, 1.0 + 0.01 * r.standard_normal()), tu, p['ping_lvl'] * p['up_lvl'] * 0.7)
    dsp.place(x, _low(p, r, n - dsp.n_of(tu)), tu, p['thock_lvl'] * p['up_lvl'] * 0.3)
    st = _spaced_pair(x, p, r)
    if 'width' in p:                   # v2 r3: the pair folded towards mono (mid + width x side); the r2 finals' pair
        mid = 0.5 * (st[:, 0] + st[:, 1])  # decorrelated the 2-12 kHz layer to corr 0.04-0.47 (refs 0.53-1.0, median
        side = 0.5 * (st[:, 0] - st[:, 1])  # 0.76; the held-out keys are mono)
        st = np.stack([mid + p['width'] * side, mid - p['width'] * side], axis=1)
    st = dsp.butter(st, 'high', 22, order=2)
    pk = np.abs(st).max()
    hpf = p.get('hpf')                 # absent in the forge test's round 1-12 params: their sound is unchanged
    for c in range(2):
        st[:, c] += _floor(n, r, p['rumble'], 25 if hpf is None else hpf) * pk * 10 ** (p['floor_db'] / 20)
    if hpf is not None:                # the session's high-pass, as on the kit's edits (order 8: steep under ~45 Hz)
        st = dsp.butter(st, 'high', hpf, order=8)
    st = st[: dsp.n_of(dur)]
    return dsp.fade(st, 0.002, r.uniform(0.025, 0.05))


# ------------------------------------------------------------------------------------------------ v2 checks
def key_stats(x):
    """model-free numbers for the v2 fixes: onset (ms from the first 1 ms frame within 20 dB of the 2-12 kHz band's
    peak to that peak: a crisp contact is 0-2 ms), top (2-12 kHz share, dB re total), t40 (ms, peak -> -40 dB, 5 ms
    frames)"""
    m = np.asarray(x, np.float64)
    m = m.mean(1) if m.ndim == 2 else m
    hi = dsp.butter(m, 'band', (2000, 12000), 4)
    k = SR // 1000
    n = len(hi) // k
    e = 10 * np.log10((hi[:n * k].reshape(n, k) ** 2).mean(1) + 1e-14)
    ip = int(np.argmax(e))
    i20 = int(np.nonzero(e[:ip + 1] >= e[ip] - 20)[0][0])
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    top = 10 * np.log10(P[(f >= 2000) & (f < 12000)].sum() / (P.sum() + 1e-14) + 1e-14)
    k5 = 5 * k
    n5 = len(m) // k5
    e5 = 10 * np.log10((m[:n5 * k5].reshape(n5, k5) ** 2).mean(1) + 1e-14)
    j = int(np.argmax(e5))
    below = np.nonzero(e5[j:] <= e5[j] - 40)[0]
    return {'onset_ms': float(ip - i20), 'top_db': round(float(top), 1),
            't40_ms': float(below[0] * 5) if len(below) else float((n5 - j) * 5)}


def key_extra(x, row=None):
    """Critic(extra=...) for typing_key v2: a top that is not dull (2-12 kHz share >= -5 dB re total; the forge refs
    sit at -11..-0.3, median -2.8, the v1 finals at -8..-4) and a tail that is not long (t40 <= 200 ms; v1 ran 1.67 x
    the held-out median). The onset number is reported only: the clicky refs have two onsets 8-35 ms apart (a click
    jacket / a touch, then the hit), so a 'crisp onset' rule would fail real keys. At most -1.5"""
    s = key_stats(x)
    pen = min(1.0, max(0.0, -5 - s['top_db']) / 6) + min(1.0, max(0.0, s['t40_ms'] - 200) / 100)
    return -0.75 * pen
