"""A deep cinematic boom impact (trailer hit), built from physics and studio practice; every sample is computed.

What a trailer boom is, physically:
  * a big, slack drum-like body (a large membrane / a heavy panel) struck hard: its fundamental starts high and SLIDES
    DOWN as the amplitude dies (tension modulation: a big displacement stretches the skin and raises the pitch; as it
    rings down the pitch settles), with a few inharmonic membrane overtones (Bessel ratios 1, 1.59, 2.14, 2.30 ...)
    that die faster than the fundamental;
  * the contact: a broadband crack of a few ms to a few tens of ms (the striker, the air shock, splintering), much
    quieter than the body but carrying the bandwidth;
  * the rest of the struck structure (drum shell, stage, a big panel): a dense cloud of low-mid modes, 70 Hz to ~1.5 kHz,
    the higher ones dying faster;
  * a sub layer: the long low tail (a sine at 30-60 Hz with a slow decay) that you feel more than hear;
  * rumble: low turbulent energy (brown noise through a falling low-pass) from the structure and the air settling;
  * a whoosh-down: air displaced by the hit, heard as resonant noise whose band sweeps down after the contact;
  * a flanged tail: trailer hits are often processed with a comb whose delay drifts (a jet flanger on the reverb):
    the comb's peaks at k / delay slide down together as the delay grows, a fan of descending partials in the tail;
  * distant echoes: the boom comes back from far walls / buildings, later, quieter and darker each time;
  * a large hall: frequency-dependent decay (lows ring longest, highs die early), diffuse early reflections,
    pre-delay, a decorrelated stereo tail;
  * a reverse swell into the hit (the trailer "suck-in"): a broadband hall tail of the contact computed, reversed,
    filtered and faded in so it rises into the contact;
  * an air riser before it (air pulled in: band-passed noise rising in pitch and level into the contact);
  * console saturation that only bites on the loud part (tanh, small signals pass clean) followed by the console's
    high-frequency roll-off, then a microphone with a noise floor.
No recorded audio is loaded, sampled or convolved: the impulse responses are computed noise.

History (the forge agent's listening notes): round 1 had every layer but the highs hung on as a flat broadband
curtain for seconds (hard-clipped sub = square-wave harmonics), the swell was a dark low blob, bandwidth was ~2 kHz
against 5-16 kHz in the refs. Round 2: the low layers are rendered at 6 kHz and up-sampled (speed), saturation is
followed by a roll-off, the swell is broadband, echoes were added, the hall's early reflections are diffuse.
Round 3: the crack's single noise burst printed static stripes through the hall tail (its speckled spectrum) -> a
smooth contact pulse + a micro-impulse crackle; a flanged tail (time-varying feedback comb) for the descending
partial lines heard and seen in the refs' tails; low bounds extended where the search sat on them.
Round 4: the stripes were still there (the impulse cloud was only a few ms long) and a dark line sat at 6 kHz (the
hall's crossover notch): subtractive hall bands; contact = smooth pulse + damped striker mode + debris spread over
30-500 ms; the direct sound stays centred and the stereo comes from the hall and from diffuse layers rendered
partly independent per channel (the old mid/side widener combed each channel).
Round 5: the tails were a near-pure 38 Hz tone (the <50 Hz band held the energy, 200-500 Hz sat 10 dB below the refs)
and the striker mode (up to 9 ms) printed a thin line at ~2 kHz through the hall: a low-mid shell mode cloud, striker
damping limited to 3 ms.
Round 6: the swells stayed dark low blobs (the search sat on pre_bright's lower bound) while refs 2 and 3 rise into
the hit with a broadband air swell: an air riser (rising band-passed noise), pre_bright allowed lower.
Round 7: the search kept pushing the debris burst to its 30 ms floor with the striker mode at its 3 ms ceiling,
which brings faint static stripes back into the tail (the critic likes them; the refs have none): debris >= 60 ms.
Round 8 (launch film, after a human blind test): of 15 forged sounds the listener could pick out only the impact, and
the reason was its tail: it falls -22 dB by 2 s and dies, while real booms keep rolling at about -10..-14 dB for 2-3 s
before they fall (kit impact_03 holds within -6 dB for 2.5 s). A new layer, the ROLL: the low energy of a big space
settling after a hit (distant reflections arriving and overlapping, the structure and the air still moving), brown
noise under a low-pass with a slow, irregular swell-and-recede (a roll, as thunder rolls), building over ~0.15 s,
holding for roll_hold s with a slow decline, then falling (roll_fall). It is diffuse (partly independent per
channel). It draws its noise only when roll_level > 0 and after every other layer, so the round-1-7 finals re-render
sample-exact from their parameters.
"""
import numpy as np
from scipy import signal

from .. import dsp
from . import REF_ROOT

SR = dsp.SR
LO = 6000                     # sample rate of the low layers (body, sub, rumble), up-sampled x8
TARGET = 'a deep cinematic boom impact'
ALTS = ['a kick drum', 'an explosion', 'a door slam', 'thunder']
REFS = [f'{REF_ROOT}/impact/impact_0{i}.wav' for i in (1, 2, 3)]

MEMBRANE = [1.0, 1.594, 2.136, 2.296, 2.653, 2.918]

SPACE = {
    # reverse swell into the hit
    'pre_len': (0.08, 1.6, 'log'),
    'pre_level': (0.02, 1.0, 'log'),
    'pre_bright': (200, 16000, 'log'),
    'pre_low': (0.0, 1.0),
    'riser_level': (0.0, 0.6),
    'riser_f': (800, 12000, 'log'),
    # contact crack
    'crack_level': (0.01, 0.8, 'log'),
    'crack_ms': (0.2, 3.0, 'log'),
    'crack_f': (600, 9000, 'log'),
    'contact_ms': (0.15, 3.0, 'log'),
    'crackle': (0.0, 1.5),
    'debris_ms': (60, 500, 'log'),
    # the body: pitch-dropping membrane
    'thump_f0': (70, 400, 'log'),
    'thump_f1': (22, 70, 'log'),
    'drop_tau': (0.006, 0.3, 'log'),
    'thump_tau': (0.12, 2.0, 'log'),
    'modes_level': (0.0, 0.6),
    # the struck structure's low-mid mode cloud
    'shell_level': (0.0, 0.8),
    'shell_lo': (70, 250, 'log'),
    'shell_span': (2.0, 10.0, 'log'),
    'shell_tau': (0.05, 1.2, 'log'),
    # sub tail
    'sub_f': (24, 62, 'log'),
    'sub_tau': (0.15, 3.0, 'log'),
    'sub_level': (0.0, 1.0),
    # rumble
    'rumble_level': (0.02, 1.2, 'log'),
    'rumble_cut': (80, 1500, 'log'),
    'rumble_tau': (0.3, 3.5, 'log'),
    # whoosh-down
    'sweep_level': (0.005, 0.5, 'log'),
    'sweep_hi': (250, 8000, 'log'),
    'sweep_tau': (0.08, 1.2, 'log'),
    # distant echoes
    'echo_level': (0.0, 0.6),
    'echo_delay': (0.08, 0.6, 'log'),
    # hall + chain
    # flanged tail (time-varying feedback comb on the hall)
    'comb_mix': (0.0, 0.8),
    'comb_d0': (0.15, 2.0, 'log'),
    'comb_ratio': (1.2, 6.0, 'log'),
    'comb_fb': (0.2, 0.85),
    'hall_wet': (0.05, 1.2, 'log'),
    'hall_rt': (0.5, 8.0, 'log'),
    'hall_damp': (0.0, 1.0),
    'predelay': (0.0, 0.06),
    'drive': (0.0, 1.0),
    'sat_tone': (1500, 14000, 'log'),
    'width': (0.0, 1.0),
    'floor_db': (-95.0, -66.0),
    # round 8: the roll (the long low tail of a big space settling)
    'roll_level': (0.0, 1.5),
    'roll_cut': (60, 400, 'log'),
    'roll_hold': (0.8, 3.2),
    'roll_fall': (0.25, 1.5, 'log'),
    'roll_rate': (0.3, 3.0, 'log'),
    'roll_depth': (0.0, 0.8),
}
# round 8 pick 1 (launch film forge: the launch-film forge run (impact), tone=1.5): the forge-test final impact_01 + the roll
DEFAULTS = {'pre_len': 0.7875, 'pre_level': 0.133, 'pre_bright': 600.0, 'pre_low': 0.5775, 'riser_level': 0.0,
            'riser_f': 3063.0, 'crack_level': 0.72, 'crack_ms': 1.907, 'crack_f': 1133.0, 'contact_ms': 0.6308,
            'crackle': 1.035, 'debris_ms': 66.93, 'thump_f0': 141.9, 'thump_f1': 30.16, 'drop_tau': 0.03454,
            'thump_tau': 0.2087, 'modes_level': 0.2633, 'shell_level': 0.2035, 'shell_lo': 121.7, 'shell_span': 3.57,
            'shell_tau': 0.3507, 'sub_f': 60.13, 'sub_tau': 0.8391, 'sub_level': 0.0, 'rumble_level': 0.1937,
            'rumble_cut': 103.9, 'rumble_tau': 1.658, 'sweep_level': 0.1123, 'sweep_hi': 365.9, 'sweep_tau': 0.5692,
            'echo_level': 0.485, 'echo_delay': 0.1192, 'comb_mix': 0.1677, 'comb_d0': 0.2453, 'comb_ratio': 3.23,
            'comb_fb': 0.7521, 'hall_wet': 0.8017, 'hall_rt': 3.897, 'hall_damp': 0.2887, 'predelay': 0.0,
            'drive': 0.3529, 'sat_tone': 8003.0, 'width': 0.3363, 'floor_db': -88.67, 'roll_level': 0.4072,
            'roll_cut': 141.7, 'roll_hold': 2.26, 'roll_fall': 0.5904, 'roll_rate': 0.6466, 'roll_depth': 0.5458}

TAIL_S = 5.0
NEUTRAL = {'comb_mix': 0.0, 'riser_level': 0.0, 'roll_level': 0.0}     # values that switch a newer layer off (to seed from older rounds)


# ------------------------------------------------------------------------------------------------ helpers
def _noise(n, r, color='white'):
    """white / pink / brown noise of n samples, unit RMS (IIR colouring: cheap at any length)"""
    w = r.standard_normal(n)
    if color == 'pink':      # Paul Kellet's economy pink filter (-3 dB/oct within 0.5 dB above ~10 Hz at 48 kHz)
        w = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                           [1, -2.494956002, 2.017265875, -0.522189400], w)
    elif color == 'brown':   # leaky integrator (-6 dB/oct above a few Hz)
        w = signal.lfilter([1.0], [1.0, -0.997], w)
    return w / (np.sqrt(np.mean(w ** 2)) + 1e-12)


def _mic(y, r, floor_db, low_cut=20.0, air_db=-1.0):
    """a microphone + preamp on a stereo array: low cut, soft top, independent pink noise floor per channel"""
    y = dsp.butter(y, 'high', low_cut, order=1)
    y = dsp.peq(y, 10000, air_db, 0.6)
    pk = np.abs(y).max() + 1e-12
    for c in range(y.shape[1]):
        y[:, c] += _noise(len(y), r, 'pink') * pk * 10 ** (floor_db / 20)
    return y


def _t(n, sr=SR):
    return np.arange(n) / sr


def _glide(t, f_start, f_end, tau, r):
    """a sine whose frequency slides exponentially from f_start to f_end (time constant tau), closed-form phase"""
    ph = 2 * np.pi * (f_end * t + (f_start - f_end) * tau * (1 - np.exp(-t / max(tau, 1e-4))))
    return np.sin(ph + r.uniform(-0.3, 0.3))


def _env(t, attack, tau):
    e = np.exp(-t / max(tau, 1e-4))
    if attack > 0:
        a = t < attack
        e[a] *= 0.5 - 0.5 * np.cos(np.pi * t[a] / attack)
    return e


def _lp(x, f, order=2, sr=SR):
    return signal.sosfilt(signal.butter(order, float(np.clip(f, 10, sr * 0.45)), 'low', fs=sr, output='sos'), x, axis=0)


def _hp(x, f, order=2, sr=SR):
    return signal.sosfilt(signal.butter(order, float(np.clip(f, 10, sr * 0.45)), 'high', fs=sr, output='sos'), x, axis=0)


def _up(x, n):
    """6 kHz -> 48 kHz, cut / padded to n samples"""
    y = signal.resample_poly(x, SR // LO, 1)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


_XO = None


def hall_ir(r, rt60, damp, size=0.8, predelay=0.02, seconds=None):
    """a computed hall impulse response: ~40 diffuse early reflections over the first ~(20 + 80 size) ms, then a noise
    tail that builds up and decays with a frequency-dependent RT60 (lows longer, highs shorter: air + wall absorption).
    The 4 decay bands are split subtractively (low-pass, remainder, low-pass, remainder ...) so they sum back exactly:
    summed Butterworth band-passes left phase notches at the crossovers (a dark line at 6 kHz through every tail)."""
    global _XO
    if _XO is None:
        _XO = [signal.butter(2, f, 'low', fs=SR, output='sos') for f in (250, 1500, 6000)]
    seconds = seconds or min(5.0, rt60 * 1.1 + 0.1)
    n = dsp.n_of(seconds)
    t = _t(n)
    w = r.standard_normal(n)
    rts = [rt60 * 1.25, rt60, rt60 * (1 - 0.5 * damp), rt60 * (1 - 0.8 * damp)]
    tail = np.zeros(n)
    rest = w
    for k, rt in enumerate(rts):
        if k < 3:
            band = signal.sosfilt(_XO[k], rest)
            rest = rest - band
        else:
            band = rest
        tail += band * np.exp(-6.91 * t / max(rt, 0.05))
    tail *= 1 - np.exp(-t / (0.008 + 0.03 * size))           # diffuse build-up
    er = np.zeros(n)
    for _ in range(40):
        d = int(r.uniform(0.003, 0.02 + 0.08 * size) * SR)
        if d < n:
            er[d] += r.uniform(0.05, 0.2) * r.choice([-1, 1]) * np.exp(-d / SR / 0.06)
    ir = tail * 0.25 + _lp(er, 7000, 1)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    pd = dsp.n_of(predelay) if predelay > 0 else 0
    return np.concatenate([np.zeros(pd), ir])


def _sat(x, drive, tone):
    """soft clipping that only bites on the loud part (small signals pass at unity), then the console's top roll-off"""
    if drive <= 1e-3:
        return x
    g = 1 + 7 * drive
    pk = np.abs(x).max() + 1e-12
    y = pk * np.tanh(g * x / pk) / g * (1 + 0.15 * drive)
    return _lp(y, tone, 2)


try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _vcomb_core(x, d, fb):
    """feedback comb with a time-varying fractional delay d (samples): y[i] = x[i] + fb * y[i - d[i]]"""
    n = len(x)
    y = np.empty(n)
    for i in range(n):
        v = x[i]
        q = i - d[i]
        if q >= 1.0:
            j = int(q)
            fr = q - j
            v += fb * ((1.0 - fr) * y[j] + fr * y[j + 1])
        y[i] = v
    return y


def _flange(x, p):
    """the comb's peaks sit at k / delay: a delay that grows over the tail makes a fan of partials sliding down (the
    descending lines in the refs' tails). Mixed in parallel, so the plain hall stays underneath."""
    if p['comb_mix'] < 0.01:
        return x
    t = _t(len(x))
    d = p['comb_d0'] / 1000 * SR * (1 + (p['comb_ratio'] - 1) * (1 - np.exp(-t / 1.2)))
    y = _vcomb_core(np.ascontiguousarray(x), np.ascontiguousarray(d), p['comb_fb'])
    y *= (1 - p['comb_fb'])
    return x + p['comb_mix'] * (y - x)


# ------------------------------------------------------------------------------------------------ layers
def _body(p, t, r):
    """the struck membrane at LO rate: fundamental + inharmonic overtones, all sliding down together"""
    x = _glide(t, p['thump_f0'], p['thump_f1'], p['drop_tau'], r) * _env(t, 0.0015, p['thump_tau'])
    ml = p['modes_level']
    if ml > 0.005:
        for j, ratio in enumerate(MEMBRANE[1:]):
            f0, f1 = p['thump_f0'] * ratio, p['thump_f1'] * ratio
            if f0 > LO * 0.45:
                continue
            a = ml * (0.75 ** j) * r.uniform(0.6, 1.0)
            x += a * _glide(t, f0, f1, p['drop_tau'], r) * _env(t, 0.001, p['thump_tau'] / (1.5 + 0.8 * j))
    return x


def _shell(p, t, r):
    """the rest of the struck structure at LO rate: a dense cloud of low-mid modes (drum shell, the floor / stage, a
    big panel) between shell_lo and shell_lo * shell_span, higher modes decaying faster. Fills the 100-600 Hz region
    that a lone sliding fundamental leaves empty (the refs' tails carry it at -7..-10 dB, round 4 at -20 dB)."""
    lo = p['shell_lo']
    hi = min(lo * p['shell_span'], LO * 0.42)
    cnt = 28
    f = np.sort(np.exp(r.uniform(np.log(lo), np.log(hi), cnt)))
    u = np.log(f / lo) / max(1e-9, np.log(hi / lo))
    tau = p['shell_tau'] * np.exp(-1.6 * u)                  # top modes ring ~5x shorter
    a = 10 ** (-4.0 * np.log2(f / lo) / 20) * r.uniform(0.3, 1.0, cnt)
    x = np.zeros(len(t))
    for fi, ti, ai in zip(f, tau, a):
        x += ai * np.sin(2 * np.pi * fi * t + r.uniform(0, 2 * np.pi)) * np.exp(-t / ti)
    x *= 1 - np.exp(-t / 0.0015)
    return x / (np.abs(x).max() + 1e-12)


def _rumble(p, t, r):
    """low turbulent energy at LO rate: brown noise through a low-pass that falls as the energy drains, rolling a bit"""
    n = len(t)
    w = _noise(n, r, 'brown')
    c = min(p['rumble_cut'], LO * 0.42)
    a = _lp(w, c, 2, LO)
    b = _lp(w, c * 0.3, 2, LO)
    k = np.exp(-t / max(p['rumble_tau'] * 0.5, 0.05))
    y = k * a + (1 - k) * b * 1.4
    y = _hp(y, 22, 2, LO)
    mod = np.ones(n)
    for _ in range(3):
        mod += 0.12 * np.sin(2 * np.pi * r.uniform(1.5, 6.0) * t + r.uniform(0, 2 * np.pi))
    y = y * mod * _env(t, 0.012, p['rumble_tau'])
    return y / (np.abs(y).max() + 1e-12)


def _crack(p, n, r):
    """the contact: a smooth force pulse (contact_ms long: its spectrum rolls off smoothly above ~1/contact_ms), the
    striker's heavily damped high mode at crack_f ringing for crack_ms (a smooth spectral bump), and debris: a crackle
    of micro-impulses spread over debris_ms. History: a single few-ms noise burst (round 2) and a few-ms impulse cloud
    (round 3) both have a speckled spectrum (random deep notches every ~1/duration Hz) that the hall prints as static
    stripes through the whole tail; spread over >= 30 ms the speckle is finer than the ear's (and the spectrogram's)
    resolution."""
    m = min(n, dsp.n_of(0.6))
    out = np.zeros(n)
    k = max(2, dsp.n_of(p['contact_ms'] / 1000))
    out[:k] += 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, k))
    t = _t(m)
    tau = p['crack_ms'] / 1000
    out[:m] += 0.8 * np.sin(2 * np.pi * p['crack_f'] * t) * np.exp(-t / tau) * (1 - np.exp(-t / 0.0002))
    out[:m] = _hp(out[:m], 30, 1)
    if p['crackle'] > 0.01:
        td = p['debris_ms'] / 1000
        cnt = int(60 + 3000 * td)
        ts = r.exponential(td * 0.5, cnt)
        ts = ts[ts < 0.58]
        cl = np.zeros(m)
        np.add.at(cl, (ts * SR).astype(int), r.standard_normal(len(ts)) * np.exp(-ts / td))
        f = p['crack_f']
        cl = dsp.butter(cl, 'band', (f * 0.45, min(f * 4.0, SR * 0.45)), 2)
        cl /= np.abs(cl).max() + 1e-12
        out[:m] += p['crackle'] * cl
    return out / (np.abs(out).max() + 1e-12)


def _sweep(p, n, r):
    t = _t(n)
    w = _noise(n, r, 'pink')
    lo = 70.0
    fc = lo + (p['sweep_hi'] - lo) * np.exp(-t / p['sweep_tau'])
    y = dsp.svf(w, fc, 2.5, 'band')
    y = y * _env(t, 0.004, p['sweep_tau'] * 1.8)
    return y / (np.abs(y).max() + 1e-12)


def _riser(p, n, r):
    """air pulled in before the hit (the trailer suck-in as air, not reverb): noise through a band-pass whose centre
    rises exponentially to riser_f at the contact while the level grows (a reversed whoosh, steepest at the end)"""
    t = _t(n)
    u = t / max(t[-1], 1e-6)                              # 0 .. 1 over the swell
    fc = 150.0 * (p['riser_f'] / 150.0) ** (u ** 1.5)
    w = _noise(n, r, 'pink')
    y = dsp.svf(w, fc, 1.2, 'band')
    y *= (np.exp(3.5 * u) - 1) / (np.exp(3.5) - 1)
    return y / (np.abs(y).max() + 1e-12)


def _roll(p, n, r):
    """round 8: the long low roll of a big space settling (see the history above), 48 kHz, peak 1. The noise is
    rendered at LO rate and up-sampled; the swell-and-recede is a smooth random log-amplitude (roll_rate Hz)"""
    m = int(n * LO / SR) + 8
    t = _t(m, LO)
    w = _lp(_noise(m, r, 'brown'), p['roll_cut'], 2, LO)
    w += 0.5 * _lp(_noise(m, r, 'pink'), p['roll_cut'] * 1.8, 2, LO)          # a little body above the cut
    w = _hp(w, 24, 2, LO)
    k = max(2, int(LO / max(p['roll_rate'], 0.05)))
    lf = _lp(_noise(m + k, r), p['roll_rate'], 2, LO)[k:]
    lf /= np.sqrt(np.mean(lf ** 2)) + 1e-12
    hold, fall = p['roll_hold'], p['roll_fall']
    e = (1 - np.exp(-t / 0.15)) * np.exp(-t / (hold * 2.5))
    e *= np.where(t > hold, np.exp(-(t - hold) / fall), 1.0)
    y = w * e * np.exp(p['roll_depth'] * lf - 0.5 * p['roll_depth'] ** 2)
    y = _up(y, n)
    return y / (np.abs(y).max() + 1e-12)


def _echoes(x, p, r):
    """2-3 distant reflections: later, quieter, darker each time"""
    out = np.zeros_like(x)
    if p['echo_level'] < 0.01:
        return out
    d = 0.0
    for k in range(3):
        d += p['echo_delay'] * r.uniform(0.8, 1.3)
        i = dsp.n_of(d)
        if i >= len(x):
            break
        e = _lp(x[:len(x) - i], 4000 * 0.5 ** k, 2)
        out[i:] += p['echo_level'] * 0.6 ** k * e
    return out


def render(p, r):
    pre = p['pre_len']
    n_hit = dsp.n_of(TAIL_S)
    # centred direct sound: the body + sub (6 kHz, up-sampled, saturated) and the contact
    tl = _t(int(TAIL_S * LO), LO)
    low = (_body(p, tl, r) + p['sub_level'] * np.sin(2 * np.pi * p['sub_f'] * tl + r.uniform(0, 1)) * _env(tl, 0.02, p['sub_tau'])
           + p['shell_level'] * _shell(p, tl, r))
    low = _sat(_up(low, n_hit), p['drive'], p['sat_tone'])
    crack = _crack(p, n_hit, r)
    direct = low + p['crack_level'] * crack
    # diffuse layers (rumble, whoosh-down): partly independent per channel (width), like a diffuse sound field
    w = p['width']
    rum_c, swp_c = _up(_rumble(p, tl, r), n_hit), _sweep(p, n_hit, r)
    n_pre = dsp.n_of(pre)
    irs = [hall_ir(r, p['hall_rt'], p['hall_damp'], 0.8, p['predelay']),
           hall_ir(r, p['hall_rt'], p['hall_damp'], 0.8, p['predelay'] + r.uniform(0.001, 0.006))]
    n_tot = n_pre + n_hit + max(len(i) for i in irs)
    out = np.zeros((n_tot, 2))
    for c in range(2):
        rum = (1 - w) * rum_c + w * _up(_rumble(p, tl, r), n_hit) if w > 0.01 else rum_c
        swp = (1 - w) * swp_c + w * _sweep(p, n_hit, r) if w > 0.01 else swp_c
        dry = direct + p['rumble_level'] * rum + p['sweep_level'] * swp
        dry = dry + _echoes(dry, p, r)
        dry = dsp.fade(dry, 0.0, 0.4)
        out[n_pre:n_pre + n_hit, c] += dry
        wet = _flange(signal.fftconvolve(dry, irs[c]), p)
        out[n_pre:n_pre + len(wet), c] += p['hall_wet'] * wet
    # reverse swell: a hall tail of the contact (plus some of the low body), reversed into the hit
    m = dsp.n_of(0.2)
    head = crack[:m] / (np.abs(crack[:m]).max() + 1e-12) + p['pre_low'] * low[:m] / (np.abs(low[:m]).max() + 1e-12)
    pk = np.abs(out).max()
    for c in range(2):
        sw_ir = hall_ir(r, max(0.3, pre * 1.6), 0.4, 0.6, 0.0, seconds=pre * 1.2 + 0.25)
        sw = signal.fftconvolve(head, sw_ir)[:n_pre][::-1]
        sw = _lp(sw, p['pre_bright'], 2)
        sw = dsp.fade(sw, min(pre * 0.25, 0.1), 0.003)
        out[:n_pre, c] += sw * p['pre_level'] * pk / (np.abs(sw).max() + 1e-12)
        if p['riser_level'] > 0.005 and n_pre > 64:
            out[:n_pre, c] += p['riser_level'] * pk * dsp.fade(_riser(p, n_pre, r), 0.01, 0.002)
    if p.get('roll_level', 0.0) > 0.005:                  # round 8: drawn last, so older params re-render exactly
        n_roll = min(len(out) - n_pre, dsp.n_of(p['roll_hold'] + 6 * p['roll_fall'] + 0.5))
        rc = _roll(p, n_roll, r)
        w = p['width']
        for c in range(2):
            rl = (1 - w) * rc + w * _roll(p, n_roll, r) if w > 0.01 else rc
            out[n_pre:n_pre + n_roll, c] += p['roll_level'] * pk * dsp.fade(rl, 0.0, 0.3)
    y = _mic(out, r, p['floor_db'])
    y = dsp.trim_tail(y, -66, 0.05)
    return dsp.fade(y, 0.002, 0.25)


def swell(seconds, r, f_start=150.0, f_end=6000.0, q=1.2, curve=1.5, grow=3.5, width=0.6, room_wet=0.25,
          tube=0.5, tube_lo=90.0, tube_hi=600.0, flutter=0.25, rev=0.5):
    """the air riser on its own (launch film): a rising swell that ENDS on its loudest sample, for a stop-down cut
    into a hit. Layers, all computed:
      * air pulled in: pink noise through a band-pass whose centre rises from f_start to f_end
        (f = f_start * (f_end / f_start) ** u**curve, steepest at the end) while the level grows exponentially (grow);
      * a shortening air column (tube): the same air driving a pipe whose resonances k * f_tube (k = 1..6, Q ~ 9,
        amplitude 1 / k) rise from tube_lo to tube_hi: the tonal, harmonic sweep of a tension riser;
      * turbulence: a slow random log-normal flutter of the level (flutter) - moving air is never steady;
      * a reverse swell (rev): the computed hall tail of a broadband strike, reversed, so a diffuse wash grows into
        the end (the trailer reverse-cymbal);
    partly independent per channel (width), in a short hall, through a microphone. Stereo 48 kHz, peak 1, a 6 ms
    fade at the end. History: round 1 (band-passed noise alone) read 'filtered white noise', p_real 0.12."""
    n = dsp.n_of(seconds)
    t = _t(n)
    u = t / max(t[-1], 1e-6)
    fc = f_start * (f_end / f_start) ** (u ** curve)
    ft = tube_lo * (tube_hi / tube_lo) ** (u ** curve)
    lvl = (np.exp(grow * u) - 1) / (np.exp(grow) - 1)
    ir = hall_ir(r, 0.9, 0.5, 0.6, 0.004)

    def one():
        w = _noise(n, r, 'pink')
        y = dsp.svf(w, fc, q, 'band')
        y /= np.sqrt(np.mean(y ** 2)) + 1e-12
        if tube > 0:
            pipe = np.zeros(n)
            for k in range(1, 7):
                pipe += dsp.svf(w, np.minimum(ft * k, SR * 0.4), 9.0, 'bandn') / k
            y += tube * pipe / (np.sqrt(np.mean(pipe ** 2)) + 1e-12)
        if flutter > 0:
            k = int(SR / 12)
            lf = _lp(_noise(n + k, r), 12.0, 2)[k:]
            y *= np.exp(flutter * lf / (np.sqrt(np.mean(lf ** 2)) + 1e-12) - 0.5 * flutter ** 2)
        return y * lvl

    c = one()
    out = np.zeros((n, 2))
    if rev > 0:
        strike = np.zeros(dsp.n_of(0.02))
        strike[:dsp.n_of(0.004)] = _noise(dsp.n_of(0.004), r)
        rv_ir = hall_ir(r, seconds * 1.4, 0.45, 0.7, 0.0, seconds=seconds + 0.1)
    for ch in range(2):
        y = (1 - width) * c + width * one()
        y /= np.abs(y).max() + 1e-12
        if rev > 0:
            tail = signal.fftconvolve(strike, rv_ir)[:n][::-1]
            tail = _lp(tail, f_end * 0.8, 2)
            y += rev * tail / (np.abs(tail).max() + 1e-12) * (0.3 + 0.7 * u)
        wet = signal.fftconvolve(y, ir)[:n]
        out[:, ch] = y + room_wet * wet / (np.sqrt(np.sum(ir ** 2)) + 1e-9)
    out = _mic(out, r, -80.0)
    out = dsp.fade(out, 0.03, 0.006)
    return out / (np.abs(out).max() + 1e-12)
