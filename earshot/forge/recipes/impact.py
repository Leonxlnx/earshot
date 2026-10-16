"""A deep cinematic boom impact (trailer hit) and the designed motion-design hit; every sample is computed.

v3 (forge v2 round 2; out/forge_v2/impact/LOG.md and out/forge_v2/pro_impact/LOG.md have the numbers). One render, two
idioms, each a FAMILY of presets (FAMILIES) that are finished sounds and audibly different (length, weight, pitch,
brightness, texture), searched one by one around themselves (space_for / recipe_for):
  * `impact`, the cinematic boom (real refs). v1 / v2 were spotted by their short tails; real booms decay straight in
    dB at 12-18 dB/s over 3-5 s (forge refs: t40 1.4-2.0 s) and stay broadband for the first second. Layers: contact
    (smooth pulse + a band-passed noise crack with a (1 - t/len)^3 envelope + debris spread over 40-250 ms), the air
    shock (blast: a 2-octave noise band that dulls as it decays: it fills 300 Hz - 4 kHz, where v2 was 5-15 dB thin),
    the pitch-dropping membrane body (saturated), a dry sub falling onto its note, a low-mid shell cloud, the roll (the
    long low settling of a big space, a 4th-order low-pass), a hall that only hears what is above 110 Hz (a low tone in
    a long hall rang as a steady line through the tail), an optional reverse swell, a faint mic floor.
  * `pro_impact`, the designed hit (docs/motion-sfx-study.md "Impact"): the body and sub SWELL to their peak over
    50-290 ms, the energy sits under 150 Hz, the tail is a granular rumble (pulses every ~28 ms, slowing; the grains
    also tick through 1-10 kHz) and/or a sustained 5-10 kHz air band, decorrelated, and it ends on a HARD GATE.
measure() / study_penalty() / study_extra() measure a render the way the pro study measured the films and turn the
study's targets (STUDY) into a model-free critic term. Parameter sets with v2 keys render with render_v2 (below).
Round 3 (forge v2; the logs have the numbers): `impact` gets a 5-15 ms > 6 kHz `sheen` on the contact, a louder crack and
air shock (1-4 kHz) over less sub / body, and six members (classic, deep, punch, metal, swell, thunder; `pre_shape` 1 =
a swell that rises straight in dB from -34 dB); `pro_impact` gets a straight-in-dB fall into the gate (`gate_db` over
`gate_fall`) and sparser, sharper high grains (`air_div`, `grain_sharp`); measure() reports `pre_gate`. Every new key's
neutral value reproduces round 2, and the round-2 families are kept in FAMILIES_R2. Every key a family member sets
must be inside SPACE: forge's final scoring pass decodes through SPACE and fills keys outside it from the first member.
Round 4 (out/forge_v2/impact/LOG.md, pro_impact/LOG.md): the r3 contact sheen overshot, so `impact` gets a quieter, longer,
band-limited sheen (`sheen_top`), a darker crack, a 2nd-order roll-off of the whole hit (`top`) and members that are
different kinds of boom: a struck metal `plate` layer (metal) and falling debris (`scatter`, the blast member);
`pro_impact` gets random grain timing / levels (`grain_jit`) and complete parameter sets (the r3 finals carried a
snap + sheen click that forge's final pass filled in at mid-range). Neutral values render round 3 exactly; the r3
families are kept in FAMILIES_R3.
Round 6 (out/forge_v2/pro_impact/LOG.md): `pro_impact` gets a designed BOOM body (`boom*`: a sine sliding ~95-140 ->
36-52 Hz, no rng draw), a rumble that enters just after it (`roll_in`) and grains near the film's ~36 / s; boom 0 /
roll_in 0 render round 4 exactly, and the r4 family is kept in FAMILIES_R4.

v2 (rounds 1-8, the forge test and the launch film), what a trailer boom is, physically:
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
REFSET = 'impact'   # the CLI's default refs: this refsets category, 'forge' split (bench held out)
REFS = [f'{REF_ROOT}/impact/impact_0{i}.wav' for i in (1, 2, 3)]

MEMBRANE = [1.0, 1.594, 2.136, 2.296, 2.653, 2.918]

SPACE_V2 = {
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
DEFAULTS_V2 = {'pre_len': 0.7875, 'pre_level': 0.133, 'pre_bright': 600.0, 'pre_low': 0.5775, 'riser_level': 0.0,
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


def render_v2(p, r):
    """the v2 boom (rounds 1-8, kept so the v2 finals, seed entries and demo scripts re-render sample-exact)"""
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


# ================================================================================================ v3 (forge v2 round 2)
# One recipe, two idioms (see the module docstring, "v3"): the cinematic boom (`impact`: real refs, a long natural
# roll / hall) and the designed motion hit (`pro_impact`: swell into a sub body, a granular rumble, a hard gate).
# render(p) dispatches: parameter sets with v2 keys ('pre_len') render with render_v2.
SPACE = {
    # the swell into the hit (a computed reverse hall of the contact + body, before t = 0)
    'pre': (0.0, 1.2),                      # s of swell before the contact (0 = none)
    'pre_level': (0.02, 0.7, 'log'),        # its peak against the hit's
    # contact: smooth force pulse + band-passed crack + a spread crackle of debris (brightness)
    'crack': (0.03, 2.0, 'log'),            # crack level against the body peak (r3: up to 2)
    'crack_f': (900, 7000, 'log'),          # crack centre
    'crack_ms': (2.0, 18.0, 'log'),         # crack length
    'snap': (0.0, 1.5),                     # a 1.5 ms > 5 kHz snap on the contact (round 2: 'dull onset')
    'sheen': (0.0, 2.0),                    # round 3: a 5-15 ms > 6 kHz burst on the contact (the onset top)
    'sheen_ms': (4.0, 16.0, 'log'),
    'sheen_f': (4500, 9000, 'log'),
    'pre_shape': [0, 1],                    # round 3: 1 = the swell rises straight in dB from -34 dB (0 = r2's shape)
    # (every rendered key must be in SPACE: forge's final run fills keys outside it from the first member)
    'debris': (0.0, 1.0),                   # debris crackle level (spread over 40-250 ms)
    'blast': (0.0, 1.0),                    # the air shock: broadband noise burst (fills 300 Hz - 4 kHz)
    'blast_f': (250, 3000, 'log'),          # its centre
    'blast_tau': (0.03, 0.8, 'log'),        # its decay
    # body: the pitch-dropping membrane (weight, pitch)
    'body_f0': (70, 230, 'log'),
    'body_f1': (34, 75, 'log'),
    'drop_tau': (0.015, 0.14, 'log'),
    'body_tau': (0.08, 0.7, 'log'),
    'body_attack': (0.001, 0.3, 'log'),     # rise of body + sub to their peak (designed hits swell 50-290 ms)
    'modes': (0.0, 0.6),                    # membrane overtones
    'body': (0.1, 1.2, 'log'),              # body level (every other level is relative to a body peak of 1)
    'drive': (0.0, 1.0),                    # body saturation (harmonics that survive small speakers)
    # sub: dry sine falling sub_semis semitones onto sub_f
    'sub': (0.0, 1.2),
    'sub_f': (28, 60, 'log'),
    'sub_semis': (0.0, 24.0),
    'sub_tau': (0.2, 2.2, 'log'),
    # low-mid structure (fills 120-600 Hz)
    'shell': (0.0, 0.8),
    'shell_lo': (90, 260, 'log'),
    # the roll: low noise of a big space settling (length, texture)
    'roll': (0.0, 1.5),
    'roll_cut': (55, 320, 'log'),
    'roll_hold': (0.2, 3.0),
    'roll_fall': (0.12, 1.6, 'log'),
    'roll_rate': (0.3, 3.0, 'log'),
    'roll_depth': (0.0, 0.8),
    'grain_rate': (18, 48, 'log'),          # granular rumble pulses per second (slowing over the tail)
    'grain_depth': (0.0, 1.0),
    # air band over the body (designed hits: a sustained 5-10 kHz band)
    'air': (0.0, 1.0),
    'air_f': (2500, 11000, 'log'),
    'air_div': (0.25, 1.0),                 # round 3: the high grains run at grain_rate x air_div (0.5: distinct
                                            # stripes every ~55 ms with dark gaps, as in ref_2 4.71)
    'grain_sharp': (2.0, 6.0),              # round 3: exponent of the high grain envelope (3 = r2)
    'air_q': (0.5, 3.0),                    # round 3: Q of the air band (0.8 = r2)
    # space + edit
    'hall_wet': (0.03, 1.0, 'log'),
    'hall_rt': (0.4, 6.0, 'log'),
    'hall_damp': (0.0, 1.0),
    'hall_tone': (3000, 12000, 'log'),      # low-pass on what the hall hears (round 2: the tails were 5-10 dB thin
                                            # at 1-8 kHz against every ref)
    'width': (0.0, 1.0),                    # decorrelation of the diffuse layers (body + sub stay centred)
    'gate': (0.0, 1.6),                     # 0 = natural ring-out; > 0 = hard gate this many s after the contact
    'gate_ms': (6.0, 40.0),                 # gate fade
    'gate_fall': (0.08, 0.6, 'log'),        # round 3: s of straight-in-dB fall before the gate ...
    'gate_db': (0.0, 30.0),                 # ... by this many dB (0 = none: r2 parameter sets render as before)
    'floor_db': (-110.0, -70.0),            # microphone floor (<= -105: none, a designed file)
    # round 4 (every neutral value renders round 3 exactly; new layers draw only when used)
    'sheen_top': (7000, 20000, 'log'),      # low-pass on the sheen (>= 19 kHz: none = r3; the r3 sheen ran flat to 24 kHz:
                                            # > 12 kHz share of the onset -10..-14 dB vs the forge refs' -31..-48)
    'top': (6000, 20000, 'log'),            # 2nd-order low-pass on the whole hit (>= 19 kHz: none; real booms roll off)
    'plate': (0.0, 1.0),                    # a struck metal plate / panel: an inharmonic mode cloud at 48 kHz
    'plate_f': (150, 900, 'log'),           # its lowest mode (modes up to ~16x, <= 9 kHz)
    'plate_tau': (0.2, 2.0, 'log'),         # decay of the lowest mode (higher modes die faster)
    'scatter': (0.0, 1.0),                  # debris / rubble falling after the hit (diffuse crackle over scatter_s)
    'scatter_s': (0.3, 2.5),
    'grain_jit': (0.0, 1.0),                # random grain timing / level (0 = r3's even ~28 / 55 ms stripes)
    # round 6 (pro_impact): a designed BOOM body, the clear pitch-dropping sine kick a designed hit is built on (r4's
    # members were a gated rumble with a faint body; CLAP: 'a snare roll', phrase z down). 0 = none (r4 exact; no rng)
    'boom': (0.0, 2.5),                     # level against the body peak of 1
    'boom_f0': (80, 260, 'log'),            # start pitch ...
    'boom_f1': (34, 70, 'log'),             # ... sliding exponentially onto this
    'boom_drop': (0.015, 0.15, 'log'),      # pitch time constant
    'boom_tau': (0.08, 0.9, 'log'),         # amplitude decay after the rise
    'boom_att': (0.002, 0.12, 'log'),       # rise (raised cosine)
    'boom_drive': (0.0, 1.0),               # tanh drive: harmonics that carry the boom on small speakers
    'roll_in': (0.0, 0.4),                  # round 6: the rumble enters this many s after the contact (0 = r4), so the
                                            # boom is heard first and the grains carry the tail
}

# the cinematic boom (family 'impact', preset 'classic'); FAMILIES has the palettes
DEFAULTS = {'pre': 0.35, 'pre_level': 0.12, 'crack': 0.35, 'crack_f': 2200.0, 'crack_ms': 6.0, 'debris': 0.4,
            'blast': 0.5, 'blast_f': 900.0, 'blast_tau': 0.25, 'body_f0': 140.0, 'body_f1': 46.0, 'drop_tau': 0.05, 'body_tau': 0.32, 'body_attack': 0.004,
            'modes': 0.25, 'body': 1.0, 'drive': 0.35, 'sub': 0.7, 'sub_f': 40.0, 'sub_semis': 12.0, 'sub_tau': 1.1,
            'shell': 0.3, 'shell_lo': 130.0, 'roll': 0.6, 'roll_cut': 140.0, 'roll_hold': 1.6, 'roll_fall': 0.6,
            'roll_rate': 0.7, 'roll_depth': 0.45, 'grain_rate': 30.0, 'grain_depth': 0.0, 'air': 0.0,
            'air_f': 7000.0, 'hall_wet': 0.45, 'hall_rt': 3.2, 'hall_damp': 0.4, 'width': 0.55, 'gate': 0.0,
            'gate_ms': 20.0, 'floor_db': -90.0, 'snap': 0.0, 'hall_tone': 5000.0, 'sheen': 0.0, 'sheen_ms': 9.0,
            'sheen_f': 6000.0, 'pre_shape': 0, 'gate_fall': 0.25, 'gate_db': 0.0, 'air_div': 1.0, 'grain_sharp': 3.0,
            'air_q': 0.8, 'sheen_top': 20000.0, 'top': 20000.0, 'plate': 0.0, 'plate_f': 400.0, 'plate_tau': 0.8,
            'scatter': 0.0, 'scatter_s': 1.2, 'grain_jit': 0.0, 'boom': 0.0, 'boom_f0': 150.0, 'boom_f1': 48.0,
            'boom_drop': 0.045, 'boom_tau': 0.3, 'boom_att': 0.01, 'boom_drive': 0.3, 'roll_in': 0.0}


def _crack3(p, n, r):
    """contact: a smooth force pulse (~0.6 ms), a band-passed noise crack whose envelope is (1 - t / len)^3 (Farnell's
    striker shape), and debris: micro-impulses spread over 40-250 ms (spread wide so the hall prints no static
    stripes: v2 rounds 3-4)"""
    m = min(n, dsp.n_of(0.4))
    out = np.zeros(m)
    k = max(2, dsp.n_of(0.0006))
    out[:k] += 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, k))
    L = min(m, max(8, dsp.n_of(p['crack_ms'] / 1000)))
    u = np.arange(L) / L
    w = r.standard_normal(L) * (1 - u) ** 3 * (1 - np.exp(-np.arange(L) / (SR * 0.0003)))
    f = p['crack_f']
    w = dsp.butter(w, 'band', (f * 0.5, min(f * 3.0, SR * 0.45)), 2)
    out[:L] += 1.4 * w / (np.abs(w).max() + 1e-12)
    if p['debris'] > 0.01:
        td = 0.04 + 0.2 * p['debris']
        cnt = int(80 + 2500 * td)
        ts = r.exponential(td * 0.45, cnt)
        ts = ts[ts < 0.38]
        cl = np.zeros(m)
        np.add.at(cl, np.minimum((ts * SR).astype(int), m - 1), r.standard_normal(len(ts)) * np.exp(-ts / td))
        cl = dsp.butter(cl, 'band', (f * 0.35, min(f * 4.0, SR * 0.45)), 2)
        out += 0.5 * p['debris'] * cl / (np.abs(cl).max() + 1e-12)
    out = _hp(out, 40, 1)
    y = np.zeros(n)
    y[:m] = out / (np.abs(out).max() + 1e-12)
    return y


def _snap(n, r):
    """the first 1.5 ms of a hard contact: a > 5 kHz noise snap (envelope (1 - t/len)^4), peak 1"""
    L = dsp.n_of(0.0015)
    u = np.arange(L) / L
    w = dsp.butter(r.standard_normal(L) * (1 - u) ** 4, 'high', 5000.0, 2)
    y = np.zeros(n)
    y[:L] = w / (np.abs(w).max() + 1e-12)
    return y


def _sheen(p, n, r):
    """round 3: the bright top of a hard contact, 5-15 ms long (the judge: the made impacts' > 6 kHz onset share sat
    7 dB under the held-out hit; the 1.5 ms snap was too short to count): two partly independent > 6 kHz noise bursts
    (envelope (1 - u)^2, 0.4 ms rise) for L / R, so the sheen is a little wide; stereo, peak 1"""
    L = max(16, dsp.n_of(p['sheen_ms'] / 1000))
    u = np.arange(L) / L
    env = (1 - u) ** 2 * (1 - np.exp(-np.arange(L) / (SR * 0.0002)))
    c = r.standard_normal(L)
    y = np.zeros((n, 2))
    for ch in range(2):
        w = dsp.butter((0.7 * c + 0.3 * r.standard_normal(L)) * env, 'high', p['sheen_f'], 4)
        if p.get('sheen_top', 20000.0) < 19000:          # round 4: a band, not white to 24 kHz (no rng draw)
            w = _lp(w, p['sheen_top'], 2)
        y[:L, ch] = w
    return y / (np.abs(y).max() + 1e-12)


def _plate(p, n, r):
    """round 4 (the judge: the six booms are near clones to CLAP, 0.905): a struck metal plate / panel, the 'metal'
    member's ring. 40 inharmonic modes log-spread from plate_f up to ~16 x plate_f (<= 9 kHz), amplitude ~ f^-0.6,
    decay plate_tau x (f / plate_f)^-0.55 (the top dies first), a 0.8 ms strike rise and a slight downward pitch glide
    in the first 100 ms (a big plate's nonlinear 'bwang'); L / R share the modes with different phases (a little
    width). Stereo, peak 1, zero after its ring."""
    m = min(n, dsp.n_of(min(4.0, 5.0 * p['plate_tau'] + 0.1)))
    t = _t(m)
    cnt = 40
    f0 = p['plate_f']
    f = np.sort(f0 * np.exp(r.uniform(0, np.log(min(16.0, 9000.0 / f0)), cnt)))
    a = (f / f0) ** -0.6 * r.uniform(0.3, 1.0, cnt)
    tau = p['plate_tau'] * (f / f0) ** -0.55
    ph = r.uniform(0, 2 * np.pi, (cnt, 2))
    glide = 0.012 * 0.1 * (1 - np.exp(-t / 0.1))            # integral of 1.2 % * exp(-t / 0.1): phase of the glide
    y = np.zeros((n, 2))
    on = 1 - np.exp(-t / 0.0008)
    for k in range(cnt):
        arg = 2 * np.pi * f[k] * (t + glide)
        e = a[k] * np.exp(-t / tau[k]) * on
        y[:m, 0] += e * np.sin(arg + ph[k, 0])
        y[:m, 1] += e * np.sin(arg + ph[k, 1])
    return y / (np.abs(y).max() + 1e-12)


def _scatter(p, n, r):
    """round 4: debris / rubble falling after the hit (the 'blast' member): a diffuse crackle of small impacts (Poisson,
    thinning out) over scatter_s, each a click band-passed 700 Hz - 6 kHz with a few heavier, darker pieces; the first
    pieces land ~30 ms after the contact. Independent per channel, stereo, peak 1."""
    T = p['scatter_s']
    L = min(n, dsp.n_of(T + 0.25))
    y = np.zeros((n, 2))
    for c in range(2):
        cnt = int(60 + 260 * T)
        ts = 0.03 + r.exponential(T * 0.4, cnt)
        ts = ts[ts < T]
        cl = np.zeros(L)
        amp = r.standard_normal(len(ts)) * np.exp(-ts / (0.5 * T)) * r.uniform(0.2, 1.0, len(ts)) ** 2
        np.add.at(cl, np.minimum((ts * SR).astype(int), L - 1), amp)
        k = dsp.n_of(0.004)                              # each piece a 2-4 ms decaying noise grain, not one sample
        cl = signal.fftconvolve(cl, r.standard_normal(k) * np.exp(-np.arange(k) / (SR * 0.0012)))[:L]
        hi = dsp.butter(cl, 'band', (700.0, 6000.0), 2)
        lo = dsp.butter(cl, 'band', (250.0, 1400.0), 2)
        y[:L, c] = hi / (np.abs(hi).max() + 1e-12) + 0.6 * lo / (np.abs(lo).max() + 1e-12)
    return y / (np.abs(y).max() + 1e-12)


def _rise(t, a):
    """0 -> 1 over a seconds (raised cosine; a sharp onset when a is tiny)"""
    e = np.ones_like(t)
    if a > 0:
        s = t < a
        e[s] = 0.5 - 0.5 * np.cos(np.pi * t[s] / a)
    return e


def _body3(p, t, r):
    """pitch-dropping membrane + overtones (dying faster), swelling over body_attack, then saturated"""
    a = p['body_attack']
    late = np.maximum(t - a, 0)
    x = _glide(t, p['body_f0'], p['body_f1'], p['drop_tau'], r) * np.exp(-late / p['body_tau'])
    if p['modes'] > 0.005:
        for j, q in enumerate(MEMBRANE[1:]):
            x += (p['modes'] * 0.75 ** j * r.uniform(0.6, 1.0)
                  * _glide(t, p['body_f0'] * q, p['body_f1'] * q, p['drop_tau'], r)
                  * np.exp(-late / (p['body_tau'] / (1.5 + 0.8 * j))))
    x *= _rise(t, a)
    if p['drive'] > 0.01:
        g = 1 + 6 * p['drive']
        pk = np.abs(x).max() + 1e-12
        x = pk * np.tanh(g * x / pk) / np.tanh(g)
        x = _lp(x, min(2700, 900 + 2500 * (1 - p['drive'])), 2, LO)
    return x / (np.abs(x).max() + 1e-12)


def _sub3(p, t, r):
    """the dry sub: a sine falling sub_semis semitones onto sub_f (time constant 90 ms), slow decay"""
    f1 = p['sub_f']
    f0 = f1 * 2 ** (p['sub_semis'] / 12)
    a = max(p['body_attack'], 0.008)
    return _glide(t, f0, f1, 0.09, r) * _rise(t, a) * np.exp(-np.maximum(t - a, 0) / p['sub_tau'])


def _grains(t, rate, depth, r, slow=0.18, jit=0.0):
    """a granular amplitude pattern: pulses every 1 / rate s, the spacing growing by `slow` over 1.2 s (the rumble
    slows down), level and timing jittered; returns a gain curve around 1 (depth 0 = flat). Round 4: jit > 0 makes the
    spacing random (each step x (1 + jit * U(-0.6, 0.9))) and the levels more uneven (r3's even stripes read as a
    snare roll); jit 0 draws exactly as round 3."""
    if depth <= 0.01:
        return np.ones_like(t)
    g = np.zeros_like(t)
    T = t[-1]
    tk = 0.0
    while tk < T:
        sp = (1 + slow * min(tk / 1.2, 1.5)) / rate
        i0 = max(0, int((tk + r.uniform(-0.12, 0.12) * sp) * SR))
        L = min(len(t) - i0, int(sp * 1.6 * SR))
        if L > 8:
            u = np.arange(L) / SR
            g[i0:i0 + L] += r.uniform(0.55 - 0.4 * jit, 1.0) * (1 - np.exp(-u / 0.0025)) * np.exp(-u / (sp * 0.32))
        tk += sp * (1 + jit * r.uniform(-0.6, 0.9)) if jit > 0 else sp
    g /= np.percentile(g, 95) + 1e-9
    return (1 - depth) + depth * np.clip(g, 0, 1.6)


def _roll3(p, n, r):
    """the roll (v2 round 8): brown + pink noise under a low-pass, a slow irregular swell-and-recede, holding
    roll_hold s and falling with roll_fall; with grain_depth > 0 a granular pulse pattern (the designed hits' rumble)"""
    m = int(n * LO / SR) + 8
    tl = _t(m, LO)
    w = _lp(_noise(m, r, 'brown'), p['roll_cut'], 4, LO) + 0.5 * _lp(_noise(m, r, 'pink'), p['roll_cut'] * 1.4, 4, LO)
    w = _hp(w, 26, 2, LO)                                # band-limited on purpose (steep 4th-order low-pass)
    k = max(2, int(LO / max(p['roll_rate'], 0.05)))
    lf = _lp(_noise(m + k, r), p['roll_rate'], 2, LO)[k:]
    lf /= np.sqrt(np.mean(lf ** 2)) + 1e-12
    hold, fall = p['roll_hold'], p['roll_fall']
    e = (1 - np.exp(-tl / 0.08)) * np.exp(-tl / (hold * 2.5))
    e *= np.where(tl > hold, np.exp(-(tl - hold) / fall), 1.0)
    y = _up(w * e * np.exp(p['roll_depth'] * lf - 0.5 * p['roll_depth'] ** 2), n)
    y *= _grains(_t(n), p['grain_rate'], p['grain_depth'], r, jit=p.get('grain_jit', 0.0))   # r4: grain_jit
    return y / (np.abs(y).max() + 1e-12)


def _air3(p, n, r, a):
    """a sustained high air band over the body (5-10 kHz noise), swelling with the body, fading with the roll"""
    t = _t(n)
    y = dsp.svf(_noise(n, r, 'pink'), p['air_f'], p.get('air_q', 0.8), 'bandn')     # r3: air_q (0.8 = r2)
    e = _rise(t, max(a, 0.02)) * np.exp(-np.maximum(t - a, 0) / max(0.25, 0.5 * p['roll_hold'] + p['roll_fall']))
    if p['grain_depth'] > 0.01:                          # the grains reach up here too (ref_2 4.71: 1-10 kHz stripes)
        g = _grains(t, p['grain_rate'] * p.get('air_div', 1.0), min(1.0, 1.5 * p['grain_depth']), r,  # r3: air_div
                    jit=p.get('grain_jit', 0.0))
        e = e * g ** p.get('grain_sharp', 3.0)          # sharper than the low rumble's: distinct high grains (r3: grain_sharp)
    return y * e / (np.abs(y * e).max() + 1e-12)


def _boom(p, t):
    """round 6: the designed boom (pro_impact), at LO rate: a sine sliding exponentially from boom_f0 onto boom_f1
    (time constant boom_drop), a raised-cosine rise over boom_att and an exponential decay boom_tau, then tanh drive
    (boom_drive) and a low-pass that keeps its harmonics under ~1 kHz. Deterministic (no rng draw: every other layer
    renders exactly as without it). Peak 1."""
    f0, f1, td = p['boom_f0'], p['boom_f1'], max(p['boom_drop'], 1e-3)
    ph = 2 * np.pi * (f1 * t + (f0 - f1) * td * (1 - np.exp(-t / td)))
    a = p['boom_att']
    x = np.sin(ph) * _rise(t, a) * np.exp(-np.maximum(t - a, 0) / p['boom_tau'])
    if p['boom_drive'] > 0.01:
        g = 1 + 8 * p['boom_drive']
        x = np.tanh(g * x) / np.tanh(g)
        x = _lp(x, 1100, 2, LO)
    return x / (np.abs(x).max() + 1e-12)


def _blast3(p, n, r):
    """the air shock of a big hit (Farnell's 'blast' as a band of noise): pink noise band-passed around blast_f
    (about 2 octaves wide), 2 ms attack, decaying with blast_tau while its band sinks an octave (the shock dulls)"""
    t = _t(n)
    fc = p['blast_f'] * (0.5 + 0.5 * np.exp(-t / max(p['blast_tau'], 0.02)))
    y = dsp.svf(_noise(n, r, 'pink'), fc, 0.55, 'bandn')
    y *= (1 - np.exp(-t / 0.002)) * np.exp(-t / p['blast_tau'])
    return y / (np.abs(y).max() + 1e-12)


def _gate_curve(n, at, ms):
    g = np.ones(n)
    i = dsp.n_of(at)
    if i >= n:
        return g
    j = min(n, i + max(8, dsp.n_of(ms / 1000)))
    g[i:j] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, j - i))
    g[j:] = 0.0
    return g


def render(p, r):
    if 'pre_len' in p:                                   # a v2 parameter set
        return render_v2(p, r)
    p = {**DEFAULTS, **p}
    gate = p['gate']
    ring = p['roll_hold'] + 5 * p['roll_fall'] + 0.4
    hit_s = min(7.0, max(ring, 3 * p['sub_tau'] + 0.3, 1.2 * p['hall_rt'] + 0.3))
    if gate > 0:
        hit_s = min(hit_s, gate + p['gate_ms'] / 1000 + 0.01)
    n = dsp.n_of(hit_s)
    t = _t(n)
    # centred direct sound: body + shell + contact (+ the dry sub, added per channel below)
    a = p['body_attack']
    tl = _t(int(hit_s * LO) + 8, LO)                     # the low layers render at 6 kHz and are up-sampled
    body = _up(_body3(p, tl, r), n)
    sub = _up(_sub3(p, tl, r), n)
    shell = (_up(_shell({'shell_lo': p['shell_lo'], 'shell_span': 4.5, 'shell_tau': 0.3}, tl, r) * _rise(tl, a), n)
             if p['shell'] > 0.01 else np.zeros(n))
    crack = _crack3(p, n, r)
    direct = p['body'] * body + 0.6 * p['shell'] * shell + p['crack'] * crack
    if p.get('boom', 0.0) > 0.005:                       # round 6: the designed boom (no rng draw)
        direct = direct + p['boom'] * _up(_boom(p, tl), n)
    if p['snap'] > 0.005:                                # drawn only when used: older parameter sets stay exact
        direct = direct + p['snap'] * _snap(n, r)
    shn = _sheen(p, n, r) if p.get('sheen', 0.0) > 0.005 else None     # round 3 (drawn only when used, as snap)
    pl = _plate(p, n, r) if p.get('plate', 0.0) > 0.005 else None      # round 4 (drawn only when used)
    sc = _scatter(p, n, r) if p.get('scatter', 0.0) > 0.005 else None  # round 4 (drawn only when used)
    # diffuse layers: roll (+ grains) and air, partly independent per channel
    wd = p['width']
    rc = _roll3(p, n, r) if p['roll'] > 0.01 else None
    ac = _air3(p, n, r, a) if p['air'] > 0.005 else None
    bc = _blast3(p, n, r) if p['blast'] > 0.01 else None
    hall = p['hall_wet'] > 0.031
    irs = ([hall_ir(r, p['hall_rt'], p['hall_damp'], 0.8, 0.012),
            hall_ir(r, p['hall_rt'], p['hall_damp'], 0.8, 0.012 + r.uniform(0.001, 0.005))] if hall else [])
    n_pre = dsp.n_of(p['pre']) if p['pre'] > 0.02 else 0
    n_tot = n_pre + n + (max(len(i) for i in irs) if irs and gate <= 0 else 0)
    out = np.zeros((n_tot, 2))
    pk_d = 1.25                                          # levels are relative to a body peak of 1 (+ the usual contact)
    for c in range(2):
        dry = direct.copy()
        if shn is not None:
            dry += p['sheen'] * shn[:, c]
        if pl is not None:
            dry += p['plate'] * pl[:, c]
        if sc is not None:
            dry += p['scatter'] * pk_d * sc[:, c]
        if rc is not None:
            rl = (1 - wd) * rc + wd * _roll3(p, n, r) if wd > 0.01 else rc
            if p.get('roll_in', 0.0) > 0.005:            # round 6: the rumble enters after the boom
                rl = rl * _rise(t, p['roll_in'])
            dry += p['roll'] * pk_d * rl
        if ac is not None:
            al = (1 - wd) * ac + wd * _air3(p, n, r, a) if wd > 0.01 else ac
            dry += p['air'] * pk_d * al
        if bc is not None:
            bl = (1 - wd) * bc + wd * _blast3(p, n, r) if wd > 0.01 else bc
            dry += p['blast'] * pk_d * bl
        out[n_pre:n_pre + n, c] += dsp.fade(dry + p['sub'] * pk_d * sub, 0.0, hit_s * 0.3 if gate <= 0 else 0.0)
        if hall:
            # the hall hears body, contact and half the diffuse layers, high-passed: a low tone in a long hall
            # rings as a steady line through the tail (round 2 spectrogram); the low tail is the roll's job
            wet = signal.fftconvolve(_hp(_lp(direct + 0.5 * (dry - direct), p['hall_tone'], 1), 110, 2), irs[c])
            m = min(len(wet), n_tot - n_pre)
            out[n_pre:n_pre + m, c] += p['hall_wet'] * wet[:m]
    if n_pre > 64:                                       # the swell: the hit's own hall tail, reversed into it
        m = dsp.n_of(0.15)
        head = crack[:m] + 0.6 * body[:m]
        pk = np.abs(out).max()
        for c in range(2):
            sw_ir = hall_ir(r, max(0.4, p['pre'] * 1.8), 0.4, 0.6, 0.0, seconds=p['pre'] * 1.2 + 0.2)
            sw = signal.fftconvolve(head, sw_ir)[:n_pre][::-1]
            sw = _lp(sw, 2500, 2)
            if p.get('pre_shape', 0) > 0.5:              # round 3: the reversed hall's own rise flattened into a
                ev = np.sqrt(np.convolve(sw ** 2, np.ones(480) / 480, 'same')) + 1e-9   # straight-in-dB rise from
                sw *= 10 ** ((-34 + 34 * np.linspace(0, 1, n_pre) ** 1.3) / 20) / (ev / ev.max())  # -34 dB, so a swell
                sw = _lp(sw, 4000, 2)                    # starts audibly (r2 'swell': a bench 'late start' flag)
            else:
                sw *= np.linspace(0, 1, n_pre) ** 1.5
            sw = dsp.fade(sw, 0.01, 0.002)
            out[:n_pre, c] += sw * p['pre_level'] * pk / (np.abs(sw).max() + 1e-12)
    out = _hp(out, 24, 2)
    if p.get('top', 20000.0) < 19000:                   # round 4: the whole hit rolls off on top (r3: > 12 kHz share
        out = _lp(out, p['top'], 2)                      # -22..-31 dB of total, forge refs -39 / -62)
    if p['floor_db'] > -105:
        out = _mic(out, r, p['floor_db'], low_cut=22.0, air_db=-0.5)
    if gate > 0:
        # the gate comes last (round 2: gated before the high-pass, the filter's ringing left a -33..-47 dB tail that
        # earshot's polish read as a chopped tail), then 20 ms of silence so the file ends below the floor
        if p.get('gate_db', 0.0) > 0.1:                 # round 3: the tail falls gate_db (straight in dB) over the
            gf = max(0.02, min(p['gate_fall'], gate * 0.8))   # last gate_fall s before the cut (the pro refs fall
            tt = _t(len(out) - n_pre) - (gate - gf)            # ~20-28 dB, then the gate; r2 held at -14..-21 dB)
            out[n_pre:] *= 10 ** (-p['gate_db'] * np.clip(tt / gf, 0, 1) / 20)[:, None]
        out[n_pre:] *= _gate_curve(len(out) - n_pre, gate, p['gate_ms'])[:, None]
        out = np.concatenate([out, np.zeros((dsp.n_of(0.02), 2))])
    if gate <= 0:
        out = dsp.trim_tail(out, -70, 0.05)
        out = dsp.fade(out, 0.001, min(0.35, 0.1 * len(out) / SR))
    else:
        out = dsp.fade(out, 0.001, 0.0)
    return out / (np.abs(out).max() + 1e-12) * 0.89


# ------------------------------------------------------------------------------------------------ v3 families
# A family = presets that are each a finished sound and audibly different (length, weight, brightness, texture),
# consistent as a palette. The forge searches around each one (space_for) and keeps the family's spread.
_PRO = {'pre': 0.0, 'pre_level': 0.1, 'crack': 0.3, 'crack_f': 3000.0, 'crack_ms': 4.0, 'debris': 0.1,
        'blast': 0.05, 'blast_f': 300.0, 'blast_tau': 0.15, 'body_f0': 120.0, 'body_f1': 75.0, 'drop_tau': 0.05,
        'body_tau': 0.18, 'body_attack': 0.1, 'modes': 0.1, 'body': 0.35, 'drive': 0.15, 'sub': 0.15, 'sub_f': 48.0,
        'sub_semis': 7.0, 'sub_tau': 0.5, 'shell': 0.1, 'shell_lo': 110.0, 'roll': 1.5, 'roll_cut': 200.0,
        'roll_hold': 0.7, 'roll_fall': 0.5, 'roll_rate': 1.5, 'roll_depth': 0.4, 'grain_rate': 36.0,
        'grain_depth': 1.0, 'air': 0.2, 'air_f': 4500.0, 'hall_wet': 0.04, 'hall_rt': 0.6, 'hall_damp': 0.4,
        'width': 1.0, 'gate': 1.1, 'gate_ms': 15.0, 'floor_db': -110.0}
_CINE = {**DEFAULTS, 'sub': 0.35, 'sub_f': 42.0, 'sub_tau': 0.7, 'body_f0': 160.0, 'body_f1': 62.0, 'body_tau': 0.25,
         'shell': 0.8, 'shell_lo': 110.0, 'crack': 0.8, 'crack_f': 1500.0, 'crack_ms': 10.0, 'debris': 0.7,
         'blast': 0.8, 'blast_f': 600.0, 'blast_tau': 0.3, 'hall_wet': 0.8, 'roll': 0.45, 'roll_cut': 200.0,
         'pre': 0.0, 'snap': 0.6, 'hall_tone': 9000.0, 'roll_depth': 0.22, 'roll_rate': 0.5}
# round 3 (the judge: onset > 6 kHz 7 dB under the held-out hit, 1-4 kHz ~8 dB thin, variants near clones at CLAP cos
# 0.88): a 5-15 ms > 6 kHz sheen and a louder crack on every member, the air shock (blast) up and higher (fills
# 1-4 kHz), less sub / body (the share under 150 Hz was 90-94 % against the forge refs' 72-82 %), and six members
# that differ in size, material and texture (classic trailer boom, deep, punch, metal slam, swell, rolling thunder)
_CINE3 = {**_CINE, 'blast': 1.0, 'blast_f': 1200.0, 'sub': 0.2, 'body': 0.7, 'crack': 1.4, 'crack_f': 2500.0,
          'roll': 0.35, 'sheen': 1.4, 'sheen_ms': 7.0, 'sheen_f': 6000.0}
_PRO3 = {**_PRO, 'air': 0.7, 'air_f': 8500.0, 'air_q': 1.6, 'gate_db': 15.0, 'gate_fall': 0.25, 'gate_ms': 8.0,
         'air_div': 0.5, 'grain_sharp': 4.0}   # run 1: at 6.5 kHz / Q 0.8 the sparkle sat +10..+20 dB over both refs at
         # 4-6.3 kHz in the tail (ref_2 14.02's band is at 8 kHz, ref_2 4.71's stripes reach 10 kHz)
FAMILIES_R2 = {'impact': [     # the round-2 family (the r2 finals), kept so the showcase can re-render it
        ('classic', dict(_CINE)),
        ('deep', {**_CINE, 'body_f0': 110.0, 'body_f1': 44.0, 'sub': 0.6, 'sub_f': 34.0, 'sub_tau': 1.0,
                  'roll': 0.6, 'roll_cut': 140.0, 'roll_hold': 2.4, 'crack': 0.4, 'crack_f': 1100.0,
                  'blast_f': 400.0, 'blast_tau': 0.45, 'hall_rt': 4.5, 'hall_damp': 0.6}),
        ('punch', {**_CINE, 'body_f0': 210.0, 'body_f1': 58.0, 'drop_tau': 0.03, 'body_tau': 0.18, 'crack': 1.0,
                   'crack_f': 2600.0, 'crack_ms': 7.0, 'blast_tau': 0.15, 'roll_hold': 0.9, 'roll_fall': 0.4,
                   'hall_rt': 2.0, 'hall_wet': 0.6, 'sub_tau': 0.45}),
        ('metal', {**_CINE, 'shell': 0.8, 'shell_lo': 210.0, 'modes': 0.5, 'crack_f': 4000.0, 'debris': 1.0,
                   'air': 0.08, 'air_f': 6500.0, 'hall_damp': 0.1, 'hall_rt': 3.6, 'blast_f': 1200.0}),
        ('swell', {**_CINE, 'pre': 0.9, 'pre_level': 0.35, 'body_attack': 0.03, 'roll_hold': 2.0, 'roll': 0.55,
                   'hall_rt': 3.8}),
    ]}
FAMILIES = {
    'impact': [     # cinematic booms: a straight-in-dB decay of 12-18 dB/s over 3-5 s, like the real kit hits
        ('classic', dict(_CINE3)),
        ('deep', {**_CINE3, 'body_f0': 110.0, 'body_f1': 44.0, 'body': 0.8, 'sub': 0.45, 'sub_f': 34.0,
                  'sub_tau': 1.0, 'roll': 0.5, 'roll_cut': 140.0, 'roll_hold': 2.2, 'crack': 1.3, 'crack_f': 1600.0,
                  'blast_f': 700.0, 'blast_tau': 0.45, 'hall_rt': 4.5, 'hall_damp': 0.6, 'sheen': 1.5,
                  'sheen_ms': 10.0}),
        ('punch', {**_CINE3, 'body_f0': 210.0, 'body_f1': 58.0, 'drop_tau': 0.03, 'body_tau': 0.18, 'crack': 1.35,
                   'crack_f': 3000.0, 'crack_ms': 7.0, 'blast_tau': 0.12, 'roll_hold': 0.9, 'roll_fall': 0.4,
                   'hall_rt': 2.0, 'hall_wet': 0.6, 'sub': 0.15, 'sub_tau': 0.45, 'sheen': 1.6, 'sheen_ms': 5.0}),
        ('metal', {**_CINE3, 'shell': 0.8, 'shell_lo': 260.0, 'modes': 0.5, 'crack_f': 3500.0, 'debris': 0.6,
                   'air': 0.0, 'air_f': 6500.0, 'hall_damp': 0.35, 'hall_rt': 3.6, 'blast_f': 2000.0, 'sheen': 1.8,
                   'sheen_ms': 9.0, 'sheen_f': 7000.0, 'crack': 2.0, 'body_attack': 0.001, 'hall_tone': 5500.0}),
                   # r3 run 1: the shell's slower swell made the loudest 0.5 ms land 40 ms late ('attack 40 ms', refs
                   # 3-10) -> body_attack 0.001, crack 2; run 2: +30 dB over every ref at 12-16 kHz (bright debris, an
                   # undamped 9 kHz hall, air) -> debris 0.6, hall_tone 5.5 kHz, hall_damp 0.35, no air
        ('swell', {**_CINE3, 'pre': 0.5, 'pre_level': 0.35, 'pre_shape': 1, 'body_attack': 0.03, 'roll_hold': 2.0, 'sheen': 1.0,
                   'roll': 0.45, 'hall_rt': 3.8}),
        ('thunder', {**_CINE3, 'body': 0.5, 'body_f0': 130.0, 'body_f1': 48.0, 'debris': 1.0, 'crack_f': 1800.0,
                     'blast': 1.0, 'blast_f': 900.0, 'blast_tau': 0.5, 'roll': 0.7, 'roll_hold': 1.9,
                     'roll_depth': 0.35, 'roll_rate': 1.2, 'roll_cut': 260.0, 'sheen': 1.6, 'sheen_ms': 9.0,
                     'crack': 1.7, 'hall_rt': 3.4}),   # r3 run 1: roll_depth 0.6 re-rose +12 dB ('bumpy tail'), and
                     # a 14 ms sheen at 1.0 still read 'dull onset' (-28 dB, refs -23..-16)
    ],
    'pro_impact': [  # designed motion hits: swell into a sub body, granular rumble, a hard gate (0.6-1.3 s)
        # round 3 (the judge: 4-12 kHz grain sparkle ~15 dB under the refs, the tail held at -14..-21 dB into the gate
        # where the refs fall to -27..-30 dB first): the air / grain band up and higher, and a straight-in-dB fall of
        # gate_db over the last gate_fall s before the cut
        ('rumble', dict(_PRO3)),
        ('air', {**_PRO3, 'gate': 0.6, 'body_attack': 0.25, 'air': 0.9, 'air_f': 9000.0, 'grain_depth': 0.5,
                 'roll': 1.0, 'roll_hold': 0.5, 'sub': 0.35, 'gate_db': 18.0, 'gate_fall': 0.15, 'air_div': 1.0,
                 'grain_sharp': 3.0}),
        ('short', {**_PRO3, 'gate': 0.65, 'body_attack': 0.05, 'crack': 0.5, 'grain_rate': 42.0, 'roll_hold': 0.4,
                   'roll_fall': 0.3, 'body': 0.5, 'gate_db': 14.0, 'gate_fall': 0.2}),
        ('deep', {**_PRO3, 'gate': 1.3, 'sub': 0.6, 'sub_f': 36.0, 'body_f1': 62.0, 'grain_rate': 28.0,
                  'roll_cut': 150.0, 'roll_hold': 0.9, 'air': 0.6, 'gate_db': 16.0, 'gate_fall': 0.25}),
        ('swell', {**_PRO3, 'pre': 0.18, 'pre_level': 0.3, 'body_attack': 0.2, 'gate': 0.9, 'grain_rate': 32.0}),
    ],
}
FAMILIES_R2['pro_impact'] = [
        ('rumble', dict(_PRO)),
        ('air', {**_PRO, 'gate': 0.6, 'body_attack': 0.25, 'air': 0.5, 'air_f': 9000.0, 'grain_depth': 0.5,
                 'roll': 1.0, 'roll_hold': 0.5, 'sub': 0.35}),
        ('short', {**_PRO, 'gate': 0.65, 'body_attack': 0.05, 'crack': 0.5, 'grain_rate': 42.0, 'roll_hold': 0.4,
                   'roll_fall': 0.3, 'body': 0.5}),
        ('deep', {**_PRO, 'gate': 1.3, 'sub': 0.6, 'sub_f': 36.0, 'body_f1': 62.0, 'grain_rate': 28.0,
                  'roll_cut': 150.0, 'roll_hold': 0.9, 'air': 0.06}),
        ('swell', {**_PRO, 'pre': 0.18, 'pre_level': 0.3, 'body_attack': 0.2, 'gate': 0.9, 'grain_rate': 32.0}),
    ]
FAMILIES_R3 = {k: list(v) for k, v in FAMILIES.items()}   # the round-3 families (the r3 finals, in final_r3/)
# round 4 (the round-3 judge: the contact sheen overshot, onset > 6 kHz share -10.9 dB vs the held-out hit's -21.6 and
# the forge refs' -23 / -27 by the judge's measure, > 12 kHz of that window -10..-14 vs -31..-48; variety 0.905, the
# deep / classic / thunder trio at CLAP cos 0.94-0.97): the sheen ~9 dB down and band-limited (sheen_top), the whole
# hit rolled off on top (top), and members that are different KINDS of boom: a struck metal plate (metal), a blast
# with falling debris (blast, was thunder), a darker sub boom (deep), a tight drum hit (punch)
_CINE4 = {**_CINE3, 'sheen': 0.3, 'sheen_ms': 14.0, 'sheen_top': 11000.0, 'top': 13000.0, 'crack_f': 1600.0, 'crack': 0.95}
# (palette run 1: a 7 ms sheen put the top before the body's peak, so polish's onset window (from the 10 % point before
# the peak) read 'dull onset' -27 dB (forge refs -16 / -23) while the judge's window read -16..-19; a 14 ms sheen at
# the same level reads -19..-20 on both)
# (probe4, seeds 3 / 11: without any sheen the r3 crack alone gave onset > 6 kHz -15..-17 dB: crack 1.4 at 2.5 kHz
# reaches 7.5 kHz; at 1.8 kHz / 1.25 the 1-4 kHz body stays at -14..-15 dB and the onset falls to -19..-20)
FAMILIES['impact'] = [
    ('classic', dict(_CINE4)),
    ('deep', {**_CINE4, 'body_f0': 95.0, 'body_f1': 38.0, 'body': 0.8, 'sub': 0.55, 'sub_f': 34.0, 'sub_tau': 1.0,
              'roll': 0.55, 'roll_cut': 110.0, 'roll_hold': 2.1, 'crack': 1.1, 'crack_f': 1500.0, 'blast': 0.9,
              'blast_f': 600.0, 'blast_tau': 0.45, 'shell': 0.7, 'hall_rt': 4.5, 'hall_damp': 0.6, 'hall_tone': 4000.0,
              'sheen': 0.45, 'sheen_ms': 16.0, 'top': 11000.0}),   # (palette run 1: top 9 kHz / sheen 0.25: 'dull onset' -40 dB)
              # r4: a darker, bigger sub boom (sub 0.8 / top 7 kHz went 98 % < 150 Hz: past the forge refs' 72-82 %)
    ('punch', {**_CINE4, 'body_f0': 230.0, 'body_f1': 66.0, 'drop_tau': 0.03, 'body_tau': 0.22, 'crack': 0.95,
               'modes': 0.5, 'shell': 0.5, 'roll': 0.25,   # r4: a big struck drum (taiko-like), drier
               'crack_f': 1800.0, 'crack_ms': 7.0, 'blast_tau': 0.12, 'roll_hold': 0.9, 'roll_fall': 0.4,
               'hall_rt': 2.1, 'hall_wet': 0.45, 'sub': 0.15, 'sub_tau': 0.45, 'sheen': 0.35, 'sheen_ms': 10.0}),
    ('metal', {**_CINE4, 'shell': 0.8, 'shell_lo': 260.0, 'modes': 0.5, 'crack_f': 2200.0, 'debris': 0.4,
               'air': 0.0, 'air_f': 6500.0, 'hall_damp': 0.35, 'hall_rt': 3.6, 'blast_f': 2000.0, 'sheen': 0.35,
               'sheen_ms': 12.0, 'sheen_f': 7000.0, 'crack': 0.95, 'body_attack': 0.001, 'hall_tone': 5500.0,
               'plate': 0.7, 'plate_f': 380.0, 'plate_tau': 0.9, 'top': 11000.0}),
               # (palette run 1, plate 1.0 / tau 1.3: late 630-1250 Hz +20 dB beyond every ref, +13 dB at 12.7 kHz)
    ('swell', {**_CINE4, 'pre': 0.5, 'pre_level': 0.35, 'pre_shape': 1, 'body_attack': 0.03, 'roll_hold': 2.0,
               'sheen': 0.3, 'roll': 0.45, 'hall_rt': 3.8}),
    ('blast', {**_CINE4, 'body': 0.5, 'body_f0': 130.0, 'body_f1': 48.0, 'debris': 0.6, 'crack_f': 1400.0,
               'blast': 1.0, 'blast_f': 900.0, 'blast_tau': 0.5, 'roll': 0.7, 'roll_hold': 1.9,
               'roll_depth': 0.35, 'roll_rate': 1.2, 'roll_cut': 260.0, 'sheen': 0.3, 'sheen_ms': 12.0,
               'crack': 1.1, 'hall_rt': 3.4, 'scatter': 0.5, 'scatter_s': 2.0}),
               # (palette run 1: single-sample debris impulses at 0.8 read as '5 clicks' (art 2.0): now 2-4 ms grains)
]
# round 4 (the judge: r3's high grains are evenly spaced stripes, one every ~55 ms, and the bench margin fell 0.60 ->
# -0.07: CLAP hears 'a snare roll'): random grain timing and uneven grain levels (grain_jit) on every member
# The r3 pro presets did not set snap / sheen / sheen_ms / sheen_f / hall_tone / pre_shape (_PRO is not built on
# DEFAULTS), and the final scoring pass decodes every SPACE key: missing keys came back at mid-range (the r3 finals carry
# snap 0.75 + sheen 1.0: a bright click at the onset, > 6 kHz share of the first 15 ms -0.4..-1.7 dB vs the forge refs'
# -14 / -33). Round 4 members are complete parameter sets (DEFAULTS underneath: no snap, no sheen)
# and a darker contact (crack_f 3 -> 1.8 kHz: onset > 6 kHz -10..-17 dB on seeds 3 / 11, refs -14 / -33)
_PRO4 = {**DEFAULTS, **_PRO3, 'grain_jit': 0.7, 'crack_f': 1800.0, 'snap': 0.0, 'sheen': 0.0}
# The members start from the r3 FINALS' searched parameters (out/forge_v2/pro_impact/final_r3/report.json; the first
# r4 palette of the bare presets lost the search's gains), with the fill-bug keys put right (snap / sheen 0; pre_shape,
# hall_tone, sheen_ms / sheen_f as delivered, inert without a sheen), crack_f 1.8 kHz and grain_jit 0.7.
FAMILIES['pro_impact'] = [
    ('rumble', {**_PRO4, **dict(FAMILIES_R3['pro_impact'])['rumble'], 'crack_f': 1800.0, 'crack': 0.5122,
        'body_f1': 39.5286, 'body_attack': 0.1351, 'body': 0.1221, 'sub': 0.0082, 'roll': 1.264,
        'roll_cut': 101.7411, 'roll_hold': 0.6747, 'roll_fall': 0.3832, 'grain_rate': 22.7054, 'grain_depth': 0.9699,
        'air': 0.4949, 'gate': 0.955, 'gate_db': 20.3501, 'pre_shape': 1, 'hall_tone': 6000.0, 'sheen_ms': 8.0,
        'sheen_f': 6363.961}),
    ('air', {**_PRO4, **dict(FAMILIES_R3['pro_impact'])['air'], 'crack_f': 1800.0, 'crack': 0.3841,
        'body_f1': 42.8117, 'body_attack': 0.0444, 'body': 0.3126, 'sub': 0.6875, 'roll': 1.4018,
        'roll_cut': 255.8402, 'roll_hold': 0.7123, 'roll_fall': 0.4128, 'grain_rate': 22.3646, 'grain_depth': 0.8043,
        'air': 0.6265, 'gate': 0.8096, 'gate_db': 15.4599, 'pre_shape': 1, 'hall_tone': 6000.0, 'sheen_ms': 8.0,
        'sheen_f': 6363.961}),
    ('short', {**_PRO4, **dict(FAMILIES_R3['pro_impact'])['short'], 'crack_f': 1800.0, 'crack': 0.4808,
        'body_f1': 75.0, 'body_attack': 0.0464, 'body': 0.5303, 'sub': 0.1124, 'roll': 1.4651, 'roll_cut': 194.2516,
        'roll_hold': 0.4826, 'roll_fall': 0.2785, 'grain_rate': 39.6209, 'grain_depth': 1.0, 'air': 0.6444,
        'gate': 0.6288, 'gate_db': 14.1698, 'pre_shape': 1, 'hall_tone': 6000.0, 'sheen_ms': 8.0,
        'sheen_f': 6363.961}),
    ('deep', {**_PRO4, **dict(FAMILIES_R3['pro_impact'])['deep'], 'crack_f': 1800.0, 'crack': 0.1025,
        'body_f1': 50.8261, 'body_attack': 0.0534, 'body': 0.2825, 'sub': 0.4785, 'roll': 1.3857,
        'roll_cut': 109.0885, 'roll_hold': 0.3807, 'roll_fall': 0.601, 'grain_rate': 18.6677, 'grain_depth': 0.6976,
        'air': 0.7944, 'gate': 1.2768, 'gate_db': 16.8876, 'pre_shape': 1, 'hall_tone': 6000.0, 'sheen_ms': 8.0,
        'sheen_f': 6363.961}),
    ('swell', {**_PRO4, **dict(FAMILIES_R3['pro_impact'])['swell'], 'crack_f': 1800.0, 'crack': 0.3, 'body_f1': 75.0,
        'body_attack': 0.2, 'body': 0.35, 'sub': 0.15, 'roll': 1.5, 'roll_cut': 200.0, 'roll_hold': 0.7,
        'roll_fall': 0.5, 'grain_rate': 32.0, 'grain_depth': 1.0, 'air': 0.7, 'gate': 0.9, 'gate_db': 15.0,
        'pre_shape': 1, 'hall_tone': 6000.0, 'sheen_ms': 8.0, 'sheen_f': 6363.961}),
]
FAMILIES_R4 = {'pro_impact': list(FAMILIES['pro_impact'])}   # the round-4 family (the r4 finals, in final_r4/)
# round 6 (the round-5 judge: the held-out model checks prefer r3, phrase z 2.55 vs r4 2.17): keep r4's fixes (no onset
# tick, irregular grains) and give the hit back a clear designed BOOM body (`boom`: a sine sliding ~95-140 -> 36-52 Hz,
# rising over 60-120 ms so the loudest point still lands 50-290 ms in), let the rumble enter just after it (`roll_in`),
# and run the grains near the film's ~36 per s (grain_rate 34-42 with grain_jit 0.5: mean step x 1.075). The boom is
# centred and kept at 0.5-0.65 of the body scale so the whole stays decorrelated (corr < 0.7; the film's tails ~0-0.2).
_R6 = lambda rate: {'grain_rate': rate, 'grain_jit': 0.5, 'boom_drive': 0.1}   # noqa: E731
FAMILIES['pro_impact'] = [(n, {**p, **o}) for (n, p), o in zip(FAMILIES_R4['pro_impact'], [
    {'boom': 0.6, 'boom_f0': 120.0, 'boom_f1': 42.0, 'boom_drop': 0.07, 'boom_tau': 0.3, 'boom_att': 0.07,
     'roll_in': 0.03, 'crack': 0.7, **_R6(40.0)},                                                  # rumble
    {'boom': 0.55, 'boom_f0': 130.0, 'boom_f1': 46.0, 'boom_drop': 0.06, 'boom_tau': 0.25, 'boom_att': 0.08,
     'roll_in': 0.03, **_R6(38.0)},                                                                # air
    {'boom': 0.55, 'boom_f0': 140.0, 'boom_f1': 52.0, 'boom_drop': 0.05, 'boom_tau': 0.2, 'boom_att': 0.06,
     'roll_in': 0.03, **_R6(42.0)},                                                                # short
    {'boom': 0.65, 'boom_f0': 95.0, 'boom_f1': 36.0, 'boom_drop': 0.1, 'boom_tau': 0.3, 'boom_att': 0.08,
     'roll_in': 0.04, 'gate_db': 12.0, 'air': 0.45, **_R6(34.0)},                                  # deep
    {'boom': 0.5, 'boom_f0': 110.0, 'boom_f1': 44.0, 'boom_drop': 0.08, 'boom_tau': 0.35, 'boom_att': 0.12,
     'roll_in': 0.06, **_R6(36.0)},                                                                # swell
])]
TARGETS = {'impact': TARGET,
           'pro_impact': 'a deep designed sub impact with a granular rumble tail, gated'}
# per category: the parameters the forge may move (everything else stays at the family preset)
_SEARCH = {
    'impact': ['crack', 'crack_f', 'sheen', 'sheen_ms', 'debris', 'blast', 'blast_f', 'blast_tau', 'body_f0',
               'body_f1', 'body_tau', 'body', 'sub', 'shell', 'roll', 'roll_cut', 'roll_hold', 'roll_fall',
               'roll_depth', 'hall_wet', 'hall_rt', 'hall_tone', 'width'],
    # width stays at the preset (decorrelated tails are part of the idiom; round 2's search traded them for model
    # points: corr 0.85-0.99 against the refs' -0.01 / 0.22)
    # round 3: air_f stays at the preset (r2's search moved the 'air' member's 9 kHz band to 3.8 kHz), gate_db joins
    'pro_impact': ['crack', 'body', 'body_f1', 'body_attack', 'sub', 'roll', 'roll_cut', 'roll_hold', 'roll_fall',
                   'grain_rate', 'grain_depth', 'air', 'gate', 'gate_db'],
}
_LIMITS = {'pro_impact': {'gate': (0.55, 1.35), 'body_attack': (0.04, 0.3, 'log'), 'roll_cut': (100, 260, 'log'),
                          'roll_hold': (0.2, 1.2), 'roll_fall': (0.15, 0.8, 'log'), 'sub': (0.0, 0.7),
                          'air': (0.2, 1.0), 'gate_db': (12.0, 22.0)},   # run 1: the search took gate_db to
                          # 8-10 dB on two members (pre_gate -15 dB again)
           'impact': {'roll_hold': (0.8, 3.0), 'hall_rt': (1.6, 6.0, 'log'), 'roll_fall': (0.25, 1.6, 'log')}}


def space_for(cat):
    """the search space of one category: the parameters in _SEARCH with the category's limits"""
    lim = _LIMITS.get(cat, {})
    return {k: lim.get(k, SPACE[k]) for k in _SEARCH[cat]}


def recipe_for(cat, preset=0):
    """a recipe object for one category and family preset (index or name): SPACE = space_for(cat), DEFAULTS = the
    preset, so forge(recipe_for('pro_impact', 'air'), refset='pro_impact', mode='designed') searches around it"""
    import types
    fam = FAMILIES[cat]
    i = preset if isinstance(preset, int) else [n for n, _ in fam].index(preset)
    return types.SimpleNamespace(__name__=__name__, SPACE=space_for(cat), DEFAULTS=dict(fam[i][1]), render=render,
                                 TARGET=TARGETS[cat], ALTS=ALTS, REFSET=cat, PRESET=fam[i][0])


# ------------------------------------------------------------------------------------------------ measurements
def measure(x, sr=SR):
    """the pro study's model-free measurements of a render (docs/motion-sfx-study.md 'How to read the numbers', bed =
    silence): attack_ms (first ms within 20 dB of the peak -> peak), t20 / t40 (10 ms smoothing), slope_dBs (fit from
    -6 to -40 dB), len50, gate_db40ms (drop over the last 40 ms of the active part: a hard gate reads > 10 dB),
    sh150 (energy share under 150 Hz), tilt (dB / octave, 100 Hz - 10 kHz), octave balance, centroid, stereo
    correlation / side level and the dominant 25-3000 Hz ridge (first, median, last)"""
    x = np.asarray(x, np.float64)
    st = x if x.ndim == 2 else np.stack([x, x], 1)
    m = st.mean(1)
    n = len(m)
    # envelope 1 ms rms
    h = sr // 1000
    k = n // h
    e = np.sqrt(np.mean(m[:k * h].reshape(k, h) ** 2, 1) + 1e-20)
    edb = 20 * np.log10(e + 1e-12)
    pk = int(np.argmax(edb))
    top = edb[pk]
    first = int(np.argmax(edb >= top - 20))
    attack_ms = pk - first
    # 10 ms smoothing for t20/t40
    es = np.convolve(e ** 2, np.ones(10) / 10, 'same')
    sdb = 10 * np.log10(es + 1e-20)
    spk = int(np.argmax(sdb))
    st_top = sdb[spk]

    def tfall(d):
        w = np.nonzero(sdb[spk:] < st_top - d)[0]
        return float(w[0]) if len(w) else float(len(sdb) - spk)
    t20, t40 = tfall(20), tfall(40)
    # decay slope dB/s between -6 and -40 under peak (after peak)
    seg = sdb[spk:]
    i6 = np.nonzero(seg < st_top - 6)[0]
    i40 = np.nonzero(seg < st_top - 40)[0]
    a = i6[0] if len(i6) else 0
    b = i40[0] if len(i40) else len(seg)
    if b - a > 5:
        sl = np.polyfit(np.arange(a, b) / 1000, seg[a:b], 1)[0]
    else:
        sl = float('nan')
    # end: last frame above top-50
    act = np.nonzero(sdb > st_top - 50)[0]
    length = (act[-1] - act[0]) / 1000 if len(act) else 0
    # gate: drop in the last 40 ms of the active part
    end = act[-1] if len(act) else 0
    gate = float(sdb[max(0, end - 40)] - sdb[end]) if end > 40 else 0
    pre_gate = float(sdb[max(0, end - 60)] - st_top) if end > 60 else 0.0     # round 3
    # balance: 1/3 oct bands
    f, P = signal.welch(m, sr, nperseg=min(n, 16384))
    tot = P.sum() + 1e-30
    cents = 1000 * 2 ** (np.arange(-17, 14) / 3)
    bal = []
    for c in cents:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        bal.append(10 * np.log10(P[sel].sum() / tot + 1e-12) if sel.any() else -120)
    bal = np.array(bal)
    sh150 = P[f < 150].sum() / tot
    sel = (cents >= 100) & (cents <= 10000)
    tilt = np.polyfit(np.log2(cents[sel]), bal[sel], 1)[0]
    oct_ = {}
    for c in (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000):
        s = (f >= c / 2 ** 0.5) & (f < c * 2 ** 0.5)
        oct_[int(c)] = round(10 * np.log10(P[s].sum() / tot + 1e-12), 1)
    cen = float((f * P).sum() / tot)
    # stereo
    L, R = st[:, 0], st[:, 1]
    corr = float(np.sum(L * R) / (np.sqrt(np.sum(L * L) * np.sum(R * R)) + 1e-20))
    S = (L - R) / 2
    M = (L + R) / 2
    side_db = 10 * np.log10((S ** 2).sum() / ((M ** 2).sum() + 1e-20) + 1e-12)
    # ridge: dominant peak 25-3000 Hz in 40 ms frames (long window)
    nper = 8192
    ff, tt, Z = signal.stft(m, sr, nperseg=nper, noverlap=nper - 960)
    A = np.abs(Z)
    band = (ff >= 25) & (ff <= 3000)
    fr = ff[band][np.argmax(A[band], 0)]
    lvl = 20 * np.log10(A[band].max(0) + 1e-12)
    good = lvl > lvl.max() - 25
    rid = fr[good]
    ridge = (float(rid[0]), float(np.median(rid)), float(rid[-1])) if len(rid) else (0, 0, 0)
    return {'dur': round(n / sr, 3), 'attack_ms': attack_ms, 'peak_ms': pk, 't20': t20, 't40': t40,
            'slope_dBs': round(float(sl), 1), 'len50': round(float(length), 3), 'gate_db40ms': round(gate, 1),
            'sh150': round(float(sh150), 3), 'tilt': round(float(tilt), 2), 'cen': round(cen), 'oct': oct_,
            'corr': round(corr, 2), 'side_db': round(float(side_db), 1),
            'ridge': tuple(round(v) for v in ridge), 'pre_gate': round(pre_gate, 1)}


# ------------------------------------------------------------------------------------------------ study checks
def study_penalty(m, targets):
    """how far measurements m (measure()) fall outside target ranges {key: (lo, hi)}: the sum of the distances in
    octaves (log2) for positive quantities, in units of the range width for tilt / slope / corr; 0 inside"""
    pen = 0.0
    for k, (lo, hi) in targets.items():
        v = m[k] if k in m else None
        if k.startswith('ridge'):
            v = m['ridge'][{'ridge0': 0, 'ridge': 1, 'ridge1': 2}[k]]
        if v is None or not np.isfinite(v):
            continue
        if v < lo or v > hi:
            if lo > 0 and v > 0 and k not in ('tilt', 'slope_dBs', 'corr', 'gate_db40ms', 'sh150'):
                pen += abs(np.log2(v / (lo if v < lo else hi)))
            else:
                pen += abs(v - (lo if v < lo else hi)) / max(1e-6, hi - lo)
    return float(pen)


def study_extra(targets, weight=1.0):
    """an extra= term for the Critic (critic.Critic(..., extra=...)): -weight * study_penalty(measure(x), targets),
    so a search moves toward the pro study's measured targets (model-free) where a category has few or no refs"""
    def extra(x, row=None):
        return -weight * study_penalty(measure(x), targets)
    return extra


STUDY = {   # docs/motion-sfx-study.md "Targets for synthesis recipes" (+ the real kit hits for the cinematic boom)
    'impact': {'t40': (1300, 2600), 'slope_dBs': (-20, -10), 'tilt': (-6.5, -3.5), 'sh150': (0.65, 0.93)},
    'pro_impact': {'len50': (0.6, 1.3), 'attack_ms': (50, 290), 't40': (250, 1100), 'gate_db40ms': (8, 40),
                   'sh150': (0.75, 0.97), 'tilt': (-6.0, -2.0), 'corr': (-0.1, 0.5), 'pre_gate': (-34, -22)},
    # round 3: pre_gate = the refs' -30.3 / -27.5 dB 60 ms before the end; tilt up to -2 (ref_2_014020's 8 kHz air
    # band measures -4.2 with the band, the judge asked for more 4-12 kHz grain sparkle)
}
