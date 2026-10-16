"""A small bubble pop, built from physics (no recorded audio anywhere: every sample is computed).

What happens when a small bubble pops, and how each part is modelled:
 0. Leak before the release. A lip / film seal does not open in zero time: for a few ms before the main release
    air already leaks through a growing gap - turbulence shaped by the same cavity, swelling up to the burst (real
    pops rise over several ms; a perfect step onset reads as a digital tick).
 1. Rupture. The thin liquid wall tears and the hole opens at the Taylor-Culick speed (~10 m/s for a micron film),
    so a millimetre cap is gone in 0.1-3 ms - not in one clean step: the film tears in a few jerks (a short crackle
    of micro-ruptures). Each jerk releases part of the trapped overpressure (4 sigma / R): a positive-first pressure
    pulse with a smaller negative recovery lobe (zero net pressure). Plus film-shred noise (droplets and fragments
    thrown off) that dies within a millisecond or two. A low-pass sets how bright the tear reads at the mic.
 2. Cavity ring (Minnaert / Helmholtz). The gas pocket left behind breathes at f = (1 / 2 pi R) sqrt(3 gamma p0 / rho)
    = 3.26 Hz*m / R: 0.5-3.5 kHz for millimetre bubbles. Damping (thermal + radiation + viscous) gives Q ~ 5-40, so it
    rings for a few ms. The pitch RISES as the pocket shrinks and its neck opens (+10-120 % within 3-80 ms). The
    amplitude builds over a fraction of a millisecond or longer (the wall must accelerate; radiation grows as the
    neck opens). Breathing is slightly nonlinear: a weak 2nd harmonic that follows the squared envelope.
 3. Second cavity formant. A real cavity (the pocket and its neck, like a mouth) has a higher resonance too; it is
    excited by the same release, has its own (lower) Q and, being a mode of the same shrinking / opening cavity, it
    follows most of the same pitch rise (a fixed-pitch higher mode reads as a glass 'ting', not as a cavity).
    Real cavities are never perfectly still: the wall shape fluctuates, so the pitch wobbles by a few per cent.
    Air rushing through the opening also drives the same cavity with turbulence for the first milliseconds: a
    breathy band of noise that follows the ring's pitch (this is what makes a mouth / lip pop sound wet and soft).
 4. Daughter bubble. A burst often pinches off a smaller bubble a few ms later: a second, higher, shorter ring.
 5. Air release. The escaping gas jet and the retracting film rim give a few ms of band-limited hiss.
 6. Puff. The net volume change is a monopole pulse: a soft low half-cycle under the tick.
 7. Room tone before the event, a small room (size, damping of highs) and a close microphone (tilt, noise floor).

FAMILIES narrows SPACE for the kinds of pop the forge test tried (demo/forge-test/pop/): the main mouth / bubble pop,
the slow bloop (an entrained bubble: no film to tear, rings up while its pitch doubles), the small high pop (a
physically small bubble; round 7, read as a finger snap) and, from round 8, the lip pop (a cavity that opens without
the bubble's big pitch glide: little rise, lower Q, breathier). Apply one with {**SPACE, **FAMILIES[name]}.

 8. The edit (round 8, after the judge's measurements). The kit's pops are mid-band events, edited clean: 0.19-0.41 s
    long, 40 dB down 35-145 ms after the peak, almost nothing under 150 Hz or above 4 kHz (energy shares median
    -26 / -37 / -43 dB below 150 Hz / 4-12 kHz / above 12 kHz). Rounds 1-7 ran 0.7-0.85 s over a -55 dB hiss floor
    (the realness model liked the floor) with a low puff and a full-band tick and hiss: +15-18 dB at 40-80 Hz and
    +12-19 dB at 2-16 kHz against the kit's median. So: a recording chain high-pass (hp) and gentle low-pass (lp), a
    quieter floor, and an edit cut `clip` s after the release with a short fade, never before the ring has fallen
    `cut_db` under its peak. These keys are absent in the round 1-7 parameters, which therefore still render exactly
    as before (and round 8's picks, which have no cut_db, render as round 8 wrote them).
"""
import numpy as np

from .. import dsp
from . import REF_ROOT

TARGET = 'a small bubble pop'
ALTS = ['a cork pop', 'a water drop', 'a finger snap', 'a mouse click']
REFSET = 'pop'   # the CLI's default refs: this refsets category, 'forge' split (bench held out)
REFS = [REF_ROOT + '/ui/ui_pop_0%d.wav' % i for i in (1, 2, 3, 4)]

SPACE = {
    'f0': (400, 3500, 'log'),          # cavity (Minnaert / Helmholtz) breathing frequency at the burst
    'q': (4, 40, 'log'),               # its quality factor (decay tau = q / (pi f0))
    'rise': (-0.1, 1.4),               # pitch rise as the cavity shrinks / opens (fraction of f0)
    'rise_tau': (0.002, 0.08, 'log'),  # how fast the pitch rises
    'att': (0.0001, 0.03, 'log'),      # build-up time of the ring
    'h2': (0.0, 0.3),                  # nonlinear 2nd harmonic
    'wobble': (0.0, 0.05),             # random pitch fluctuation of the cavity (fraction)
    'breath_amp': (0.0, 1.0),          # turbulent air driving the cavity
    'breath_tau': (0.001, 0.03, 'log'),
    'f2_ratio': (1.6, 4.5),            # second cavity formant / f0
    'f2_amp': (0.0, 0.8),
    'f2_q': (3, 25, 'log'),
    'f2_follow': (0.5, 1.0),           # share of the pitch rise the formant follows (same cavity)
    'sub_amp': (0.0, 0.6),             # daughter bubble
    'sub_ratio': (1.2, 2.6),
    'sub_delay': (0.002, 0.03, 'log'),
    'click_amp': (0.0, 1.5),           # rupture pressure pulses
    'click_w': (0.00005, 0.0006, 'log'),
    'tear': (0.0001, 0.003, 'log'),    # time over which the film tears (spread of the micro-ruptures)
    'jerks': (1, 6, 'int'),
    'click_lp': (2500, 20000, 'log'),
    'shred_amp': (0.0, 1.0),           # film-shred noise at the rupture
    'shred_tau': (0.0002, 0.003, 'log'),
    'shred_hp': (500, 8000, 'log'),
    'hiss_amp': (0.0, 0.6),            # air release
    'hiss_f': (1500, 10000, 'log'),
    'hiss_q': (0.5, 3.0),
    'hiss_tau': (0.001, 0.03, 'log'),
    'puff_amp': (0.0, 1.0),            # low monopole puff
    'puff_f': (60, 400, 'log'),
    'ring_delay': (0.0, 0.003),        # ring starts slightly after the first rupture
    'rt60': (0.08, 0.9, 'log'),
    'wet': (0.0, 0.6),
    'damp': (0.3, 0.95),               # how much faster the room's highs die
    'size': (0.05, 0.8),
    'air_db': (-12.0, 2.0),            # mic / distance tilt at 10 kHz
    'floor_db': (-80, -55),
    'swell_amp': (0.0, 0.6),           # leak before the release
    'swell_t': (0.0005, 0.012, 'log'),
    'pre': (0.004, 0.035),             # room tone before the event (a recording starts before the pop)
    'hp': (40, 160, 'log'),            # 8. recording chain high-pass (order 4)
    'lp': (4000, 14000, 'log'),        #    and a gentle low-pass (order 2): a close mic on a mid-band event
    'clip': (0.14, 0.38),              #    edit point after the release (the kit: 0.19-0.41 s files)
    'cut_fade': (0.004, 0.03, 'log'),  #    the edit's fade
    'cut_db': (-56, -42),              #    never cut before the ring has fallen this far under its peak (the kit's
}                                      #    pops are cut at -46..-62 dB; round 8's bloops were cut at -26 dB)
# round 8 limits (measured on the kit's pops; see 8. above): no -55 dB hiss bed, no big low puff, no full-band tick
SPACE.update({'floor_db': (-86, -64), 'puff_amp': (0.0, 0.35), 'puff_f': (120, 400, 'log'), 'click_lp': (2000, 9000, 'log'),
              'shred_amp': (0.0, 0.5), 'shred_hp': (500, 5000, 'log'), 'hiss_f': (1500, 6000, 'log'),
              'rt60': (0.05, 0.45, 'log'), 'wet': (0.0, 0.45)})
# the main pop after the judge (round 8 main pick 2 = final pop_01, parameters rounded to 6 digits): 8 new seeds ->
# critic mean 6.76 with the tone and region checks in the total, 7/8 pass; 0.27 s, cut clean, band-limited like the
# kit's pops
DEFAULTS = {'f0': 440.159, 'q': 22.1207, 'rise': 0.991892, 'rise_tau': 0.0159722, 'att': 0.000357098,
            'h2': 0.00847939, 'wobble': 0.0112906, 'breath_amp': 0.0, 'breath_tau': 0.00848874, 'f2_ratio': 3.82514,
            'f2_amp': 0.8, 'f2_q': 25.0, 'f2_follow': 0.948223, 'sub_amp': 0.00328062, 'sub_ratio': 2.08913,
            'sub_delay': 0.0111736, 'click_amp': 0.685048, 'click_w': 7.72837e-05, 'tear': 0.000171419, 'jerks': 1,
            'click_lp': 4881.31, 'shred_amp': 0.200779, 'shred_tau': 0.00153019, 'shred_hp': 5000.0,
            'hiss_amp': 0.0659931, 'hiss_f': 2418.88, 'hiss_q': 2.86992, 'hiss_tau': 0.00113073, 'puff_amp': 0.0,
            'puff_f': 121.063, 'ring_delay': 0.00105542, 'rt60': 0.249181, 'wet': 0.291252, 'damp': 0.490701,
            'size': 0.05, 'air_db': -0.476997, 'floor_db': -64.4367, 'swell_amp': 0.160452, 'swell_t': 0.00252584,
            'pre': 0.0269276, 'hp': 92.6866, 'lp': 10181.4, 'clip': 0.274425, 'cut_fade': 0.0153059}

_SOFT = {'swell_amp': (0.15, 0.6), 'swell_t': (0.002, 0.012, 'log')}
FAMILIES = {
    'main': {**_SOFT, 'click_amp': (0.0, 0.7)},
    'bloop': {'f0': (350, 900, 'log'), 'rise': (0.7, 1.4), 'rise_tau': (0.025, 0.08, 'log'), 'att': (0.006, 0.04, 'log'),
              'q': (18, 40, 'log'), 'click_amp': (0.0, 0.35), 'shred_amp': (0.0, 0.2), 'ring_delay': (0.0, 0.025)},
    'small': {**_SOFT, 'f0': (1400, 3500, 'log'), 'rise': (0.0, 1.0), 'click_amp': (0.0, 0.8), 'shred_amp': (0.0, 0.45)},
    'lip': {'f0': (450, 1300, 'log'), 'rise': (-0.1, 0.35), 'rise_tau': (0.005, 0.05, 'log'), 'q': (4, 14, 'log'),
            'breath_amp': (0.2, 1.0), 'att': (0.0003, 0.004, 'log'), 'click_amp': (0.0, 0.6), 'sub_amp': (0.0, 0.2)},
}

DUR = 0.35


def rupture_pulse(w, n):
    """positive raised-cosine pulse of width w followed by a negative lobe of 2.5 w at matching area (zero DC)"""
    a, b = max(2, dsp.n_of(w)), max(3, dsp.n_of(2.5 * w))
    pos = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, a))
    neg = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, b))
    neg *= -pos.sum() / neg.sum()
    x = np.zeros(n)
    m = min(n, a + b)
    x[:m] = np.concatenate([pos, neg])[:m]
    return x


def tear_crackle(w, tear, jerks, n, r):
    """the film tears in `jerks` steps spread over `tear` s; the first is strongest, later ones smaller"""
    x = np.zeros(n)
    one = rupture_pulse(w, n)
    times = np.sort(np.concatenate([[0.0], r.uniform(0, tear, max(0, jerks - 1))]))
    for k, tk in enumerate(times):
        amp = (1.0 if k == 0 else r.uniform(0.25, 0.8) * 0.8 ** k) * r.choice([1, 1, 1, -1]) ** (k > 0)
        dsp.place(x, one * amp, tk)
    return x


def wobble_curve(depth, n, r, rate=60.0):
    """slow random pitch fluctuation (low-passed noise, unit RMS scaled to depth)"""
    if depth <= 0:
        return np.zeros(n)
    w = dsp.butter(r.standard_normal(n), 'low', rate, order=2)
    return depth * w / (np.sqrt(np.mean(w ** 2)) + 1e-12)


def pitch_track(f0, rise, rise_tau, n, wob):
    t = dsp.t_of(n)
    return f0 * (1 + rise * (1 - np.exp(-t / rise_tau))) * (1 + wob)


def gliding_ring(f0, q, rise, rise_tau, att, h2, n, r, wob=None):
    """damped breathing mode whose pitch rises f0 -> f0 (1 + rise); 2nd harmonic follows the squared envelope"""
    t = dsp.t_of(n)
    f = pitch_track(f0, rise, rise_tau, n, 0.0 if wob is None else wob)
    ph = 2 * np.pi * np.cumsum(f) / dsp.SR + r.uniform(-0.3, 0.3)
    tau = q / (np.pi * f0)
    env = (1 - np.exp(-t / att)) * np.exp(-t / tau)
    return env * np.sin(ph) + h2 * env ** 2 * np.sin(2 * ph + 0.5)


def render(p, r):
    n = dsp.n_of(DUR)
    t = dsp.t_of(n)
    x = np.zeros(n)
    # 1. rupture: torn-film pressure pulses (band-limited at the mic) + film shred noise
    tick = tear_crackle(p['click_w'], p['tear'], p['jerks'], n, r)
    if p['click_lp'] < 19000:
        tick = dsp.butter(tick, 'low', p['click_lp'], order=2)
    x += p['click_amp'] * tick
    shred = dsp.butter(r.standard_normal(n), 'high', p['shred_hp']) * np.exp(-t / p['shred_tau'])
    x += p['shred_amp'] * 0.5 * shred
    # 2. cavity ring + 3. its second formant (start a little after the first rupture)
    wob = wobble_curve(p['wobble'], n, r)
    body = gliding_ring(p['f0'], p['q'], p['rise'], p['rise_tau'], p['att'], p['h2'], n, r, wob)
    f2 = p['f0'] * p['f2_ratio']
    body += p['f2_amp'] * gliding_ring(f2, p['f2_q'], p['rise'] * p['f2_follow'], p['rise_tau'], p['att'], 0.0, n, r,
                                       wob)
    #    turbulent air through the opening drives the same (gliding) cavity: a breathy band following the pitch
    if p['breath_amp'] > 0.01:
        fc = pitch_track(p['f0'], p['rise'], p['rise_tau'], n, wob)
        br = dsp.svf(r.standard_normal(n), fc, max(2.0, p['q'] * 0.5), 'band')
        br *= (1 - np.exp(-t / max(p['att'], 0.0003))) * np.exp(-t / p['breath_tau'])
        body += p['breath_amp'] * br / (np.abs(br).max() + 1e-9)
    # 4. daughter bubble: higher, shorter, less glide
    if p['sub_amp'] > 0.01:
        sub = gliding_ring(p['f0'] * p['sub_ratio'], p['q'] * 0.7, p['rise'] * 0.4, p['rise_tau'] * 0.6,
                           max(p['att'], 0.0002), p['h2'], n, r)
        dsp.place(body, p['sub_amp'] * sub, p['sub_delay'])
    dsp.place(x, body, p['ring_delay'])
    # 5. air release hiss
    hiss = dsp.svf(r.standard_normal(n), p['hiss_f'], p['hiss_q'], 'band')
    hiss *= (1 - np.exp(-t / 0.0003)) * np.exp(-t / p['hiss_tau'])
    x += p['hiss_amp'] * hiss / (np.abs(hiss).max() + 1e-9)
    # 6. low monopole puff: one damped low cycle
    puff = np.sin(2 * np.pi * p['puff_f'] * t) * np.exp(-t * p['puff_f'] * 2.5) * (1 - np.exp(-t / 0.0004))
    x += p['puff_amp'] * puff
    # 0. leak before the release: cavity-shaped turbulence swelling up to the burst (-43 dB -> 0 dB)
    if p['swell_amp'] > 0.01:
        m = dsp.n_of(p['swell_t'])
        w = r.standard_normal(m)
        leak = dsp.svf(w, p['f0'], 3.0, 'band') + 0.6 * dsp.svf(w, p['f0'] * p['f2_ratio'], 2.0, 'band')
        leak += 0.3 * dsp.butter(w, 'high', 3000) * 3.0
        leak *= np.exp(5.0 * (np.arange(m) / m - 1.0))
        x = np.concatenate([p['swell_amp'] * leak / (np.abs(leak).max() + 1e-9), x])
    # 7. room tone before the event, room, mic
    lead = len(x) - n                                   # the leak swell, if any, comes before the release
    x = np.concatenate([np.zeros(dsp.n_of(p['pre'])), x])
    x = dsp.room(x, r, rt60=p['rt60'], size=p['size'], damp=p['damp'], wet=p['wet'])
    if p.get('clip') is None:                           # rounds 1-7: unchanged
        x = dsp.mic(x, r, air_db=p['air_db'], floor_db=p['floor_db'])
        x = dsp.trim_tail(x, -62, 0.02)
        return dsp.fade(x, 0.001, 0.02)
    # 8. the recording chain and the edit
    x = dsp.butter(x, 'high', p['hp'], order=4)
    x = dsp.butter(x, 'low', p['lp'], order=2)
    x = dsp.mic(x, r, air_db=p['air_db'], floor_db=p['floor_db'])
    end = dsp.n_of(p['pre']) + lead + dsp.n_of(p['clip'])
    if p.get('cut_db') is not None:                     # an editor cuts once the ring has died away
        h = dsp.n_of(0.005)
        mono = x if x.ndim == 1 else x.mean(axis=1)
        k = len(mono) // h
        e = 20 * np.log10(np.sqrt((mono[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12)
        i = int(np.argmax(e))
        below = np.nonzero(e[i:] < e[i] + p['cut_db'])[0]
        died = (i + (below[0] if len(below) else k - i)) * h + dsp.n_of(p['cut_fade'])
        end = min(len(x), max(end, died), dsp.n_of(p['pre']) + lead + dsp.n_of(0.42))
    return dsp.fade(x[:end], 0.001, p['cut_fade'])
