"""A fast air whoosh: an object (stick, cane, arm, cloth edge) swung fast past a close microphone, from aeroacoustics.

Every sample is computed; no recorded audio is loaded, sampled or convolved. Physics modelled:
  * Speed profile. The swing accelerates, peaks at the bottom of the arc (near the mic) and decelerates: U(t) is an
    asymmetric generalised bell exp(-|dt/s|^beta / 2) around the pass time tc (rise width s_rise, fall width s_fall;
    beta > 2 is a flatter top with sharper edges, beta < 2 a pointed peak).
  * Vortex shedding (the body of a whoosh). A bluff object sheds vortices at f = St * U / d (Strouhal ~0.2); at high
    Reynolds numbers the Aeolian tone smears into a band of width f/Q. A swung object has a speed gradient along its
    length (the tip moves fastest), so each strip sheds at its own frequency: K strips from ROOT to the tip give K bands
    at f_peak * U(t)^f_track * s_k (f_track < 1 lets the ridge hold steadier, as a body / cavity resonance does). A
    dipole radiates intensity ~ U^6, so each strip's amplitude goes ~ (s_k U)^3 and the tip band dominates: energy
    piles up below a resonant ridge at the tip frequency, as in real recordings. Loudness ~ U^a_pow / r(t). Each band
    falls steeply above its peak (`steep`, a second low-pass stage), as a turbulent cylinder wake's spectrum does
    above its Strouhal peak: the real recordings show a -23..-31 dB/oct cliff above the ridge.
  * Wake. Behind the object the shed vortices keep turning over as a decaying turbulent wake: a narrow noise band
    (wake_q) that starts at the pass, stays where the object passed (no Doppler), decays with wake_tau and darkens as
    it decays (large eddies outlive small ones, so the highs die first and the centre sinks: f ~ (1 + dt /
    wake_tau)^-wake_dark). This is the flat ridge that outlasts the hiss in real whooshes. It is bounded below the
    shedding level (a wake cannot out-radiate the shedding at full speed).
  * Proximity and Doppler. Distance r(t) = sqrt(1 + ((t - tc) / prox_w)^2) (prox_w = closest distance / pass speed);
    amplitude ~ 1/r; frequencies scaled by 1 - doppler * radial speed (higher on approach, lower after the pass).
  * Hiss. Small-scale trailing-edge turbulence: white + pink noise above a steep edge (hiss_hp) under a low-pass that
    opens with U; level ~ U^hiss_pow (a steep power makes the crisp, short top of a fast pass). Real recordings keep
    a dark valley between the ridge and ~2 kHz, so the hiss starts above it.
  * Wake hiss tail. The small eddies of the wake die fastest but not instantly: a broadband tail above the same edge,
    decaying with 0.4 * wake_tau, at wake_hiss_db.
  * Push. A close pass shoves the air at the diaphragm: a short sub-bass pressure gust centred on the pass (low-passed
    turbulence under a window of width push_w plus the smooth bipolar pressure pulse of the displaced air).
  * Buffet. The passing pressure field and the wake gust hit the diaphragm: random low-frequency pressure
    (< buffet_f), arriving buffet_delay after the pass and dying with buffet_tau.
  * Turbulence flutter. Random log-normal amplitude modulation (bandwidth flut_f) on the shedding and hiss.
  * Whip tone. A thin, smooth object keeps a narrower Aeolian tone (Q whip_q) at whip_ratio * tip frequency.
  * Stereo. The source moves across the mic: a mono-compatible balance pan follows tanh((t - tc) / prox_w) plus a
    little decorrelated side signal (the mono sum is unchanged, so stereo cannot game a mono critic); ranges follow
    the real references (balance swings of a few dB, L/R correlation ~0.9).
  * A small room (independent reflections per channel) and a microphone (low cut, top roll-off air_db, noise floor,
    recorder low-pass at 19 kHz).

  * The edit (round 12, after the judge). The kit's fast whooshes are edited files of 0.32-0.5 s that fall 40 dB in
    90-180 ms after the pass; rounds 1-11 ran 0.68-0.9 s with t40 230-285 ms ('medium' whooshes: their critic also
    compared them with the three medium refs). `cut_db`: the file is cut once the envelope has fallen that far under
    its peak, with a `cut_fade` fade. Parameters without these keys (rounds 1-11) render exactly as before.

What the listening loop taught (demo/forge-test/whoosh/LOG.md): the critic's description match rewards longer,
fuller tails than the real fast whooshes have; when the wake was unbounded it grew as loud as the pass itself and
doubled the tail (found by muting components one at a time), hence the physical bounds on wake level and Q.
"""
import numpy as np

from .. import dsp
from . import REF_ROOT

TARGET = 'a fast crisp air whoosh'
ALTS = ['a wind gust', 'rain', 'a sword swing', 'a jet engine']
# round 12: the fast references only (the target is a FAST whoosh; rounds 1-11 also used whoosh_med_01-03, whose
# longer tails the critic then rewarded). REFS_ROUNDS_1_11 is the old set, kept for reproducing those reports.
REFSET = 'whoosh_fast'   # the CLI's default refs: this refsets category, 'forge' split (bench held out)
REFS = [f'{REF_ROOT}/whoosh/whoosh_fast_0{i}.wav' for i in range(1, 5)]
REFS_ROUNDS_1_11 = REFS + [f'{REF_ROOT}/whoosh/whoosh_med_0{i}.wav' for i in range(1, 4)]

SPACE = {
    'tc': (0.06, 0.2),                    # pass time (s)
    's_rise': (0.012, 0.08, 'log'),       # speed bell widths (s)
    's_fall': (0.015, 0.1, 'log'),
    'beta': (1.2, 3.5),                   # bell shape
    'prox_w': (0.004, 0.06, 'log'),       # closest distance / speed (s)
    'f_peak': (150, 1500, 'log'),         # tip shedding frequency at full speed (Hz)
    'f_track': (0.0, 1.0),                # ridge frequency ~ U^f_track (1: pure Strouhal, 0: fixed resonance)
    'q_shed': (1.5, 12.0, 'log'),
    'steep': (0.0, 1.0),                  # 4-pole skirt above the shedding peak (0: plain 2-pole band)
    'a_pow': (1.0, 4.0),                  # loudness ~ U^a_pow
    'doppler': (0.0, 0.08),               # pass speed / speed of sound
    'wake_db': (-24, -6),                 # round 9: a wake cannot out-radiate the shedding at full speed
    'wake_tau': (0.03, 0.15, 'log'),
    'wake_q': (4.0, 16.0, 'log'),         # the wake ridge is narrow and steady in real recordings (Q 5-10)
    'wake_dark': (0.0, 1.0),
    'wake_hiss_db': (-36, -16),           # broadband wake tail vs shedding
    'hiss_db': (-38, -14),
    'hiss_hp': (800, 3000, 'log'),        # lower edge of the small-scale hiss
    'hiss_f': (1500, 14000, 'log'),
    'hiss_pow': (2.0, 7.0),
    'push_db': (-24, 6),                  # close-pass pressure gust vs shedding
    'push_w': (0.006, 0.04, 'log'),
    'push_f': (40, 250, 'log'),
    'buffet_db': (-30, 3),
    'buffet_f': (60, 500, 'log'),
    'buffet_delay': (0.0, 0.04),
    'buffet_tau': (0.015, 0.08, 'log'),
    'flut_depth': (0.0, 0.4),
    'flut_f': (8, 60, 'log'),
    'whip_db': (-45, -15),
    'whip_q': (8, 60, 'log'),
    'whip_ratio': (1.0, 3.0),
    'pan_w': (0.05, 0.25),                # real refs: balance swings of 1-6 dB, L/R correlation mostly 0.9-1.0
    'width': (0.0, 0.12),
    'room_wet': (0.0, 0.35),
    'rt60': (0.1, 0.5, 'log'),
    'air_db': (-12, 0),                   # mic / recorder top roll-off at 10 kHz
    'floor_db': (-95, -70),               # mic / preamp noise floor relative to the peak
    'cut_db': (-52, -38),                 # round 12: the edit, once the envelope has fallen this far under its peak
    'cut_fade': (0.015, 0.06, 'log'),     #  (the kit's fast whooshes: 0.32-0.5 s files, t40 90-180 ms)
}
# = final/whoosh_01 after the judge (round 12 pick 1, parameters rounded to 6 digits): fast refs only, tone check in
# the objective; 8 new seeds -> critic mean 8.02, 8/8 pass; t40 125 ms, 0.31 s
DEFAULTS = {'tc': 0.0706346, 's_rise': 0.0380857, 's_fall': 0.0578346, 'beta': 1.49847, 'prox_w': 0.0243735,
            'f_peak': 398.391, 'f_track': 0.108498, 'q_shed': 7.05767, 'steep': 0.23086, 'a_pow': 2.90371,
            'doppler': 0.00143736, 'wake_db': -14.0966, 'wake_tau': 0.0532322, 'wake_q': 4.45435,
            'wake_dark': 0.613911, 'wake_hiss_db': -16.0, 'hiss_db': -19.345, 'hiss_hp': 1671.88, 'hiss_f': 9908.19,
            'hiss_pow': 2.57151, 'push_db': -4.56985, 'push_w': 0.0251241, 'push_f': 125.898, 'buffet_db': -9.18542,
            'buffet_f': 95.0096, 'buffet_delay': 0.0219106, 'buffet_tau': 0.0262978, 'flut_depth': 0.0178753,
            'flut_f': 30.2817, 'whip_db': -16.8109, 'whip_q': 22.6043, 'whip_ratio': 1.75668, 'pan_w': 0.102279,
            'width': 0.0730709, 'room_wet': 0.158579, 'rt60': 0.138233, 'air_db': -12.0, 'floor_db': -75.3278,
            'cut_db': -46.4213, 'cut_fade': 0.0281062}

LEN = 0.8
K = 6
ROOT = 0.45          # innermost shedding strip (fraction of tip speed); no effect on the score in round 2, fixed
WAKE_RATIO = 0.9     # wake centre / tip shedding frequency at the pass
HISS_PINK = 0.5


def _lvl(x):
    """peak of a 10 ms RMS envelope (component level for mixing)"""
    n = dsp.n_of(0.01)
    k = len(x) // n
    return float(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)).max()) + 1e-12


def _lfnoise(n, fc, r):
    """unit-std random control signal band-limited to fc (for flutter / jitter)"""
    w = dsp.butter(r.standard_normal(n), 'low', fc, order=2)
    return w / (w.std() + 1e-12)


def _band(n, fc, q, r, steep=0.0):
    """white noise through a moving band-pass, scaled to unity peak gain (power ~ bandwidth). steep 0..1 blends in a
    second, low-pass stage tuned just above fc: the skirt above the peak falls at 18-24 dB/oct instead of 6, as the
    spectrum of a turbulent cylinder wake does above its Strouhal peak"""
    y = dsp.svf(r.standard_normal(n), fc, q, 'band') / q
    if steep > 0:
        y = (1 - steep) * y + steep * dsp.svf(y, np.asarray(fc) * 1.08, 0.9, 'low')
    return y


def render(p, r):
    n = dsp.n_of(LEN)
    t = dsp.t_of(n)
    tc = p['tc']
    dt = t - tc
    s = np.where(dt < 0, p['s_rise'], p['s_fall'])
    U = np.exp(-0.5 * np.abs(dt / s) ** p['beta'])                # speed / max speed
    rr = np.sqrt(1 + (dt / p['prox_w']) ** 2)                      # distance / closest distance
    v_rad = (dt / p['prox_w']) / rr * U                            # radial speed fraction (+ receding)
    dop = 1.0 - p['doppler'] * v_rad
    amp = U ** p['a_pow'] / rr
    flut = np.exp(p['flut_depth'] * _lfnoise(n, p['flut_f'], r) - 0.5 * p['flut_depth'] ** 2)

    # vortex shedding along the span
    f_tip = np.maximum(p['f_peak'] * np.maximum(U, 1e-3) ** p['f_track'] * dop, 30.0)
    shed = np.zeros(n)
    for k in range(1, K + 1):
        sk = ROOT + (1 - ROOT) * k / K
        shed += sk ** 3 * _band(n, np.maximum(f_tip * sk, 25.0), p['q_shed'], r, p['steep'])
    shed *= amp * flut

    # decaying, darkening wake behind the object
    after = np.maximum(dt, 0.0)
    w_on = 0.5 + 0.5 * np.tanh(dt / max(0.25 * p['s_rise'], 0.003))
    w_env = w_on * np.exp(-after / p['wake_tau'])
    f_w = np.maximum(p['f_peak'] * WAKE_RATIO * (1 + after / p['wake_tau']) ** (-p['wake_dark']), 40.0)
    wake = _band(n, f_w, p['wake_q'], r, p['steep']) * w_env          # stationary in space: no Doppler

    # small-scale wake hiss
    w = (1 - HISS_PINK) * r.standard_normal(n) + HISS_PINK * dsp.noise(LEN, r, 'pink')[:n] * 3.0
    hiss = dsp.svf(w, np.maximum(p['hiss_f'] * U ** 0.7 * dop, 400.0), 0.6, 'low')
    hiss = dsp.butter(hiss, 'high', p['hiss_hp'], order=4) * U ** p['hiss_pow'] / rr * flut

    # wake hiss tail
    wh = dsp.butter(dsp.butter(dsp.noise(LEN, r, 'pink')[:n], 'high', p['hiss_hp'], order=4), 'low', 9000, order=2)
    wh = wh * w_on * np.exp(-after / (0.4 * p['wake_tau']))

    # push: the close pass shoves air at the diaphragm
    pw = p['push_w']
    win = np.exp(-0.5 * (dt / pw) ** 2)
    ricker = (1 - (dt / (0.35 * pw)) ** 2) * np.exp(-0.5 * (dt / (0.35 * pw)) ** 2)
    push = dsp.butter(dsp.noise(LEN, r, 'brown')[:n], 'low', p['push_f'], order=2) * win
    push = push / _lvl(push) + 0.7 * dsp.butter(ricker, 'low', p['push_f'] * 2, order=2) / _lvl(ricker)

    # buffet: wake gust on the diaphragm
    tb = t - (tc + p['buffet_delay'])
    benv = np.where(tb < 0, np.exp(-0.5 * (tb / max(p['s_rise'] * 0.5, 0.004)) ** 2), np.exp(-tb / p['buffet_tau']))
    buf = dsp.butter(dsp.noise(LEN, r, 'brown')[:n], 'low', p['buffet_f'], order=2) * benv

    # whip tone
    whip = _band(n, np.minimum(f_tip * p['whip_ratio'], 16000), p['whip_q'], r) * U ** 4 / rr

    x = shed / _lvl(shed)
    x += 10 ** (p['wake_db'] / 20) * wake / _lvl(wake)
    x += 10 ** (p['hiss_db'] / 20) * hiss / _lvl(hiss)
    x += 10 ** (p['wake_hiss_db'] / 20) * wh / _lvl(wh)
    x += 10 ** (p['push_db'] / 20) * push / _lvl(push)
    x += 10 ** (p['buffet_db'] / 20) * buf / _lvl(buf)
    x += 10 ** (p['whip_db'] / 20) * whip / _lvl(whip)

    # moving source -> stereo
    side_dir = 1.0 if r.uniform() < 0.5 else -1.0
    pan = side_dir * p['pan_w'] * np.tanh(dt / p['prox_w'])
    d1 = int(r.uniform(0.2, 0.9) * 1e-3 * dsp.SR)
    side = np.zeros(n)
    side[d1:] = x[:n - d1]
    side = dsp.butter(side - x, 'high', 300) * p['width']
    L = x * (1 + pan) + side                                       # L + R = 2x: mono sum unchanged
    R = x * (1 - pan) - side
    out = []
    for ch in (L, R):
        y = dsp.room(ch, r, rt60=p['rt60'], size=0.4, wet=p['room_wet'])
        y = dsp.mic(y, r, low_cut=25.0, air_db=p['air_db'], floor_db=p['floor_db'])
        y = dsp.butter(y, 'low', 19000, order=4)
        out.append(y[:n + dsp.n_of(0.1)])
    m = min(len(o) for o in out)
    y = np.stack([o[:m] for o in out], axis=1)
    if p.get('cut_db') is None:                                    # rounds 1-11: unchanged
        return dsp.fade(dsp.trim_tail(y, -62, 0.03), 0.002, 0.06)
    h = dsp.n_of(0.005)                                            # round 12: the edit
    mono = y.mean(axis=1)
    k = len(mono) // h
    e = 20 * np.log10(np.sqrt((mono[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-12)
    i = int(np.argmax(e))
    below = np.nonzero(e[i:] < e[i] + p['cut_db'])[0]
    end = min(len(y), (i + (below[0] if len(below) else k - i)) * h + dsp.n_of(p['cut_fade']))
    return dsp.fade(y[:end], 0.002, p['cut_fade'])
