"""Drums and percussion played from real recordings: a kit (kicks, snares, rims, toms, hi-hats, crashes, a ride),
hands (claps) and hand percussion (shaker, tambourine, congas, bongos, cajon, darbuka, frame drum, claves, cabasa,
slapstick), for music beds, motion design and games, and the drum-machine voices, which are synthetic by nature.

Why recordings. A drum is its detail: the stick's exact strike, the head's pitch settling after a hard hit, the
wires answering the bottom head, a cymbal's wash opening up. Invented from physics these scored 1-2 of 5 in listening
tests, while a take out of earshot.synth.resyn is heard as the recording itself. So a drum here is one THING
(`snare/tight`, `tom/floor`) and its recordings are its members: public-domain one-shots played by drummers in
velocity layers (Versilian Community Sample Library, Karoryfer Big Rusty Drums; CC0) and, for `clap/hands`, single
claps cut out of a recording of one person clapping (Freesound; CC0), kept as a few kilobytes of numbers each (never
audio) in earshot/data/drums/ and played as new takes.

How a take is chosen and played (take()):
    energy   how hard it is played: the members are stored softest first with their velocity within their
             instrument (0..1, from the recordings' levels); energy moves the velocity a design is played at from its
             own by up to half the range, the seed walks through the nearest half of the members (a drummer never
             hits twice the same), and the take is then levelled by its loudest 50 ms and turned up or down 6 dB
    size     a drum 0.7 .. 1.4 times as big: tuned lower and ringing longer (every frequency divided by it, every
             duration multiplied)
    pitch    tuned drums (kicks, toms, hand drums): the take moved to the note asked for, at most PITCH semitones
    bright   spectral tilt, +-TILT dB per octave around 1 kHz
    hard     the strikes up to 6 dB louder and brighter (or softer and duller)
    ring     less or more damping: the ringing bands decay 0.5 .. 2 times as long and the tail is drawn out with them
    space    the shared room (earshot.synth.space); width: mid / side (a mono close mic is spread by decorrelation)
Designs played as a performance of their members: `drum/roll` (snare strokes, soft to hard, into an accent) and
`shaker/roll` (shaker strokes on a grid). Invented, because no recording of the thing exists in the public-domain
corpus, or because the thing is a synthesizer: `kick/electro`, `snare/electro`, `hat/electro`, `clap/electro` (the
drum-machine voices of earshot.synth.drums), `snare/brush`, `snap/finger`, `sticks/count`.

    from earshot.sounds import registry
    y, meta = registry.render('snare/tight', seed=2, energy=0.6)       # meta['member']: which recording was played
    y, meta = registry.render('tom/floor', pitch='E2', ring=0.4)
    y, meta = registry.render('drum/roll', dur=2.5)
"""
import json
import os
import zlib
from functools import lru_cache

import numpy as np

from ..forge import dsp, fx
from ..synth import drums as D
from ..synth import modal, resyn, space
from .registry import design, member

SR = dsp.SR
AXES = ('energy', 'size', 'bright', 'hard', 'ring', 'space', 'width')
DOM = ('music', 'motion', 'game')
STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'drums')
SIZE = (0.7, 1.4)            # size -1 .. 1: a smaller .. bigger drum (log)
RING = (0.5, 2.0)            # ring -1 .. 1: decay factor of the ringing bands (log)
LENGTH = (0.7, 1.4)          # the tail drawn out with ring (its square root, within these): beyond 0.6 .. 1.8 a warp
#                              turns the grain of a band's envelope into a buzz or a flutter (earshot.sounds.learned)
TILT = 2.0                   # dB per octave at bright +-1
PITCH = 7.0                  # semitones a tuned drum is moved at most
VARY = 0.5                   # resyn's vary per unit of the request's vary (0.35 by default: 0.18)
SPREAD = 0.3                 # decorrelation per unit of width for a mono recording
REF = -14.0                  # dBFS the loudest 50 ms of a take are brought to before energy
STEP = 6.0                   # dB of level per unit of energy
CREST, CUT = 14.0, 8.0       # dB: present()
# resyn.finish on the designs the index marks `finish` (dry close-mic hits: kick, snare, toms, rims, claps, hand
# drums), without its limiter (lufs -40: master.finish levels afterwards) and with a third to a half of its EQ and
# drive. Measured on three seeds of each as the registry ships them, with and without: loudest 50 ms +0.1 (kicks) to
# +0.8 dB (toms), low end (under 150 Hz) +0.1 to +1.2 dB, centroid of the first 100 ms within 5 %. Hats, cymbals and
# shaken things go without it (it adds air where there is nothing to bring forward)
FINISH = dict(lufs=-40.0, weight=1.5, presence=1.0, air=1.0, drive=3.0)


# ------------------------------------------------------------------------------------------------------ the store
@lru_cache(maxsize=1)
def index():
    """the store's index: {id: entry} (entry: says, uses, tags, kind, dur_s, note, vel, size, room, amount, width,
    finish, prior, members [{file, source, credit, vel, hz, onset_s, dur_s, stereo, quality ...}])"""
    with open(os.path.join(STORE, 'index.json'), encoding='utf-8') as fh:
        return {e['id']: e for e in json.load(fh)}


@lru_cache(maxsize=None)
def model(file):
    """the model of one member (its file name in the store), read once"""
    return resyn.load(os.path.join(STORE, file))


def credit(sid, member=0):
    """the recording a member was made from: {title, author, page, licence, licence_url}"""
    return dict(index()[sid]['members'][member]['credit'])


@lru_cache(maxsize=None)
def order(sid):
    """the members of a design in the fixed shuffled order the seeds walk through"""
    n = len(index()[sid]['members'])
    return tuple(int(k) for k in np.random.default_rng(zlib.crc32(sid.encode())).permutation(n))


def pick(sid, seed, energy=0.0):
    """the member (its position in the index) played for a seed at an energy: the velocity wanted is the design's
    own moved by energy / 2; of the members nearest to it (half of them, at least two, and all that are as near as
    the last of those: recordings of one velocity are equals) the seed walks the fixed shuffled order"""
    e = index()[sid]
    ms = e['members']
    if len(ms) == 1:
        return 0
    want = float(np.clip(e['vel'] + 0.5 * energy, 0.0, 1.0))
    d = np.array([abs(m['vel'] - want) for m in ms])
    edge = np.sort(d)[max(2, (len(ms) + 1) // 2) - 1]
    walk = [k for k in order(sid) if d[k] <= edge + 1e-9]
    return walk[int(seed) % len(walk)]


def loudest(y, s=0.05):
    """RMS of the loudest s seconds over both channels, dBFS"""
    p = np.mean(np.atleast_2d(np.asarray(y).T).T ** 2, axis=1)
    k = max(2, dsp.n_of(s))
    if len(p) <= k:
        return float(10 * np.log10(p.mean() + 1e-30))
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(10 * np.log10(np.max(c[k:] - c[:-k]) / k + 1e-30))


def hit(e, c, r, k, energy):
    """a dry take of member k of design e under the controls c (module doc), its loudest 50 ms at REF + energy"""
    m = e['members'][k]
    size = e['size'] * c.lerp('size', *SIZE, 1.0, log=True)
    pitch = 0.0
    if e.get('note') and m.get('hz'):
        pitch = float(np.clip(12.0 * np.log2(c.hz(m['hz']) / m['hz']), -PITCH, PITCH))
    ring = c.lerp('ring', *RING, 1.0, log=True)
    y = resyn.render(model(m['file']), r, length=float(np.clip(ring ** 0.5, *LENGTH)), pitch=pitch, size=size,
                     bright=TILT * c.bright + 0.3 * energy, hard=float(np.clip(c.hard + 0.25 * energy, -1.0, 1.0)),
                     ring=ring, vary=VARY * c.vary)
    if e.get('finish'):
        y = resyn.finish(y, **FINISH)
    return y * 10 ** ((REF + STEP * energy - loudest(y)) / 20)


def present(y):
    """a take of 0.5 s and more whose few loudest peaks stand more than CREST dB over its loudest 50 ms has those
    peaks limited (at most CUT dB, 2 ms look-ahead): levelled by its peak, a ride whose stick tick stands 22 dB over
    its wash otherwise plays 10 dB under the other drums (registry.LIFT reaches only sounds under 1 s). Shorter takes
    are left to the registry's lift, which folds the spike (as earshot.sounds.learned does)"""
    if len(y) < dsp.n_of(0.5):
        return y
    pk = 20 * np.log10(np.abs(y).max() + 1e-12)
    over = min(pk - loudest(y) - CREST, CUT)
    if over < 0.5:
        return y
    return fx.limit(y, ceiling_db=pk - over, lookahead=0.002, release=0.03, true_peak=False)


def image(y, w, stereo):
    """the take's stereo image at width w (0 mono, 1 as recorded, 2 wider): side times w; a mono recording, or a
    stereo one asked to be wider, is spread by decorrelation (SPREAD per unit of width)"""
    y = np.asarray(y, np.float64)
    mid, side = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])
    st = fx.decorrelate(mid, min(1.0, SPREAD * (max(w - 1.0, 0.0) if stereo else w)))
    return st + w * np.stack([side, -side], axis=1)


def place(e, c, r, y, stereo):
    """a take into its image and the design's room"""
    y = image(y, e['width'] * c.lerp('width', 0.0, 2.0, 1.0), stereo)
    return space.room(y, c.room(e['room']), amount=float(np.clip(e['amount'] + c.space, -1.0, 1.0)), r=r)


def take(e, c, r):
    """one take of the design e (an index entry): the member the seed and the energy pick (module doc)"""
    energy = c.energy
    k = pick(e['id'], c.seed, energy)
    m = e['members'][k]
    c.member = member(k + 1, len(e['members']))
    return place(e, c, r, present(hit(e, c, r, k, energy)), m['stereo'])


def _register(e):
    def fn(c, r):
        return take(e, c, r)
    fn.__name__ = e['id'].replace('/', '_')
    fn.__doc__ = (f"{e['says'][0]} (from {len(e['members'])} public-domain recording"
                  f"{'s' if len(e['members']) > 1 else ''}, credits in earshot/data/drums/CREDITS.md)")
    design(e['id'], kind=e['kind'], says=tuple(e['says']), uses=tuple(e['uses']),
           tags=tuple(e['tags']) + ('drums', 'real'), domains=DOM, dur=e['dur_s'], pitch=e.get('note'), axes=AXES,
           allows=tuple(e.get('allows', ())), prior=e.get('prior', 1.0))(fn)


for _e in index().values():
    _register(_e)


# ------------------------------------------------------------------------------------------------------ performances
def _strokes(e, c, r, takes, times, gains, n):
    """takes [(member, energy)] rendered once each and struck at times (s) with gains, take i % len(takes) for
    stroke i (two hands): each take placed so that its recording's onset lands on the stroke"""
    ys = [hit(e, c, r, k, en) for k, en in takes]
    out = np.zeros((n, 2))
    for i, (t, g) in enumerate(zip(times, gains)):
        k, _ = takes[i % len(takes)]
        y = ys[i % len(takes)]
        at = dsp.n_of(max(t - e['members'][k]['onset_s'] * e['size'], 0.0))
        m = min(len(y), n - at)
        if m > 0:
            out[at:at + m] += g * y[:m]
    return out


@design('drum/roll', kind='hit',
        says=('a snare drum roll swelling into a hit', 'a fast press roll on a snare drum growing louder',
              'a drum roll building up to a crash of the snare'),
        uses=('drum roll', 'build', 'reveal', 'announcement', 'tension', 'suspense', 'drum', 'transition'),
        tags=('drums', 'kit', 'real'), domains=DOM, dur=2.0, axes=AXES, allows=('pre_swell', 'stacked', 'tail'))
def drum_roll(c, r):
    """a snare roll for dur s into an accent, played with the real snare's recordings: double strokes from both
    hands (about 26 strokes a second, the hands never quite even, the second of each pair weaker), a soft recording
    for the first half and a medium one for the second, louder and louder, then one full hit of a hard recording"""
    e = index()['snare/tight']
    T = c.seconds(2.0)
    energy = c.energy
    seed = c.seed
    soft, mid, full = (pick('snare/tight', seed + i, v) for i, v in enumerate((-1.0, -0.2, 0.8 + 0.2 * energy)))
    rate = 26.0 * r.uniform(0.95, 1.05)
    t = np.arange(0.0, T - 0.03, 1.0 / rate)
    k = np.arange(len(t))
    t = np.clip(t + 0.008 * (k % 4 >= 2) + r.normal(0, 0.0025, len(t)), 0.0, T)
    u = t / T
    a = (0.06 + 0.5 * u ** 1.6) * np.where(k % 2 == 0, 1.0, 0.7) * np.exp(0.12 * r.standard_normal(len(t)))
    n = dsp.n_of(T + 1.2)
    y = _strokes(e, c, r, [(soft, -1.0)], t[u < 0.5], a[u < 0.5] * 10 ** (STEP / 20), n)
    y += _strokes(e, c, r, [(mid, -0.2)], t[u >= 0.5], a[u >= 0.5] * 10 ** (0.2 * STEP / 20), n)
    y += _strokes(e, c, r, [(full, 0.8)], [T], [10 ** ((STEP * (energy - 0.8)) / 20)], n)
    return place(e, c, r, dsp.fade(y, 0.0, 0.3), e['members'][full]['stereo'])


@design('shaker/roll', kind='noise',
        says=('a shaker being shaken steadily', 'a continuous rhythmic rattle of an egg shaker',
              'beads rattling back and forth in a shell'),
        uses=('shaker', 'groove', 'rhythm', 'percussion', 'loop'), tags=('drums', 'hand percussion', 'real'),
        domains=DOM, dur=2.0, axes=AXES, allows=('tail',))
def shaker_roll(c, r):
    """a steady shake for dur s from the real shaker's strokes: about 8 strokes a second, two recordings
    alternating as the hand goes forth and back (the forward one stronger: the hand's accent), the hand drifting a
    little in time and force"""
    e = index()['shaker/tick']
    T = c.seconds(2.0)
    energy = c.energy
    takes = [(pick('shaker/tick', c.seed + i, energy), energy) for i in range(2)]
    if takes[0][0] == takes[1][0]:
        takes = takes[:1]
    rate = 8.0 * r.uniform(0.95, 1.05)
    t = np.arange(0.06, T, 1.0 / rate)
    t = t + r.normal(0, 0.004, len(t))
    g =np.where(np.arange(len(t)) % 2 == 0, 1.0, 0.65) * np.exp(0.15 * r.standard_normal(len(t)))
    y = _strokes(e, c, r, takes, t, g * np.clip(t / 0.15, 0.4, 1.0), dsp.n_of(T + 0.6))
    return place(e, c, r, dsp.fade(y, 0.002, 0.25), True)


# ------------------------------------------------------------------------------------------------------ invented
def _built(name):
    """the instrument: one generator per design, the same for every take"""
    return dsp.rng(zlib.crc32(('drums ' + name).encode()))


def _vel(c, r, base=0.7):
    """velocity 0..1 from energy (-1 .. 1 is about 0.4 .. 1.0 around `base`), varied a little per take"""
    v = base + (0.3 if c.energy >= 0 else 0.32) * c.energy
    return float(np.clip(v * np.exp(0.04 * r.standard_normal()), 0.12, 1.0))


def _size(c):
    """(frequency factor, length factor) of the size axis: a bigger drum is lower and rings longer"""
    return c.lerp('size', 1.3, 0.77, 1.0, log=True), c.lerp('size', 0.75, 1.35, 1.0, log=True)


def _ring(c):
    """factor on ring times (less damping)"""
    return c.lerp('ring', 0.5, 1.9, 1.0, log=True)


def _st(x):
    x = np.asarray(x, np.float64)
    return x if x.ndim == 2 else np.stack([x, x], 1)


def _mix(layers, n=None):
    """layers (x, gain dB[, t s]) each brought to peak 1 first, into one stereo buffer"""
    ls = [(_st(L[0]), L[1], L[2] if len(L) > 2 else 0.0) for L in layers]
    n = n or max(dsp.n_of(t) * (t > 0) + len(x) for x, _, t in ls)
    out = np.zeros((n, 2))
    for x, g, t in ls:
        if np.abs(x).max() > 0:
            dsp.place(out, x / (np.abs(x).max() + 1e-12), t, 10 ** (g / 20))
    return out


def _out(c, r, y, room='studio', amount=0.0, tilt=0.0):
    """the take finished: width, a high shelf for bright (a harder hit is a little brighter too), the shared room"""
    y = fx.width(_st(y), c.lerp('width', 0.2, 1.8, 1.0))
    db = 6.0 * c.bright + 1.5 * c.energy + tilt
    if db:
        y = y + (10 ** (db / 20) - 1) * dsp.butter(y, 'high', 3000.0, order=1)
    return space.room(y, c.room(room), amount=float(np.clip(c.space + amount, -1.0, 1.0)), r=r)


@design('kick/electro', kind='thud',
        says=('an electronic kick drum with a long booming sub tail', 'a deep synthesized bass drum boom',
              'a drum machine kick with a falling pitch'),
        uses=('drum', 'kick drum', 'drop', 'bass hit', 'logo', 'trap', 'hip hop', 'edm'),
        tags=('drums', 'electronic'), domains=DOM, dur=1.0, pitch='F#1', axes=AXES)
def kick_electro(c, r):
    """the synthetic kick (kick_synth): a sine falling 2.6 octaves onto its pitch in two stages, a long sub tail,
    tape saturation and a click; harder = a faster snap and more click, ring = the tail's length"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    n = dsp.n_of(1.3 * kl * g)
    y = D.kick_synth(n, r, c.hz('F#1') * kf, vel, sweep=2.6 * (1 + 0.15 * c.hard), drop=0.045 * kl,
                     snap_=0.006 * 2 ** (-0.5 * c.hard), decay=0.42 * kl * g,
                     click=0.6 * 2 ** c.hard * 10 ** (c.bright / 4), drive=6.0 + 3.0 * c.bright)
    return _out(c, r, fx.decorrelate(dsp.fade(y, 0.0, 0.3 * len(y) / SR), 0.15), 'booth', -0.6)


@design('snare/electro', kind='hit',
        says=('an electronic snare drum from a drum machine', 'a synthesized snare, a tone and a burst of noise',
              'a drum machine snare hit'),
        uses=('drum', 'snare', 'backbeat', 'edm', 'hip hop', 'hit'), tags=('drums', 'electronic'),
        domains=DOM, dur=0.35, pitch='F#3', axes=AXES)
def snare_electro(c, r):
    """the synthetic snare (snare_synth): two tuned tones and a burst of filtered noise with its own longer
    envelope; ring lengthens the noise, hard sharpens it, size lowers the tones"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    y = D.snare_synth(dsp.n_of(0.45 * kl * g), r, c.hz('F#3') * kf, vel, tone=0.55 * 10 ** (-c.hard / 8),
                      noise_tau=0.12 * kl * g, snappy=1.0, bright=c.bright)
    return _out(c, r, fx.decorrelate(dsp.fade(y, 0.0, 0.08), 0.3), 'room', -0.3)


@design('hat/electro', kind='tick',
        says=('an electronic hi-hat tick from a drum machine', 'a short synthesized metallic hi-hat',
              'a crisp drum machine hat'),
        uses=('drum', 'hi-hat', 'trap', 'edm', 'tick', 'rhythm'), tags=('drums', 'electronic'), domains=DOM,
        dur=0.12, axes=AXES)
def hat_electro(c, r):
    """the synthetic hi-hat (hat_synth): six inharmonic square waves through two band-passes and a high-pass, a
    sharp decay; ring = the decay, size = the oscillators' scale, hard = a sharper start"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    y = D.hat_synth(dsp.n_of(0.25 * kl * g), r, vel, decay=0.03 * kl * g, scale=kf)
    a = dsp.n_of(0.0004 * 2 ** (-c.hard))
    y[:a] *= np.linspace(0, 1, a)
    return _out(c, r, fx.decorrelate(dsp.fade(y, 0.0, 0.05), 0.4), 'room', -0.3)


@design('clap/electro', kind='hit',
        says=('an electronic hand clap from a drum machine', 'a synthesized clap, a few quick bursts of noise',
              'a drum machine clap with a short tail'),
        uses=('clap', 'backbeat', 'edm', 'hip hop', 'accent'), tags=('drums', 'electronic'), domains=DOM,
        dur=0.3, axes=AXES, allows=('stacked',))
def clap_electro(c, r):
    """the synthetic clap (clap_synth): four bursts of band-passed noise 10 ms apart and a tail; ring = the tail,
    size = a lower band, hard = sharper bursts"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    y = D.clap_synth(dsp.n_of(0.5 * kl * g), r, vel, gap=0.0105 * 2 ** (-0.2 * c.hard), centre=1150.0 * kf,
                     tail=0.13 * kl * g)
    return _out(c, r, fx.decorrelate(dsp.fade(y, 0.0, 0.1), 0.5), 'room', 0.0)


def _wires(r, motion, f_bottom, vel, gap=0.05, release=0.022, bright=0.0, body=0.3):
    """the snare wires under a head whose displacement is `motion`: the bottom head (tuned f_bottom Hz) drives them"""
    bands = ((1700.0 * 2 ** (0.3 * bright), 1.2, 1.0), (3400.0 * 2 ** (0.3 * bright), 1.5, 0.9),
             (6500.0 * 2 ** (0.3 * bright), 1.8, 0.45 + 0.3 * vel))
    return D.rattle(D.bottom_head(motion, f_bottom), r, gap, release, bands, body)


@design('snare/brush', kind='hit',
        says=('a soft brush tap on a snare drum', 'a jazz brush hitting a snare, a soft swish with a gentle buzz',
              'a wire brush on a drum'),
        uses=('drum', 'snare', 'brush', 'jazz', 'soft hit', 'tap'), tags=('drums', 'kit'), domains=DOM,
        dur=0.35, pitch='A3', axes=AXES)
def snare_brush(c, r):
    """a wire brush (no public-domain recording of one: invented): its bristles land over a few milliseconds (some
    50 small contacts, each a tiny stroke), so the head is driven by a spread-out train instead of one blow (no
    crack), the wires buzz softly, and the bristles themselves hiss as they splay on the coated head"""
    built = _built('snare/brush')
    vel = _vel(c, r, 0.45)
    kf, kl = _size(c)
    f = c.hz('A3') * kf
    g = _ring(c)
    n = dsp.n_of(0.6 * kl * g)
    m = D.membrane(f, n, r, built, vel=vel, beater='brush', hard=c.hard, strike=0.45, t60=0.3 * kl * g,
                   damp=0.8 / g, load=0.5, seal=0.6, count=26, fall=0.0, radius=0.18 / kf, click=0.0, glide=False)
    spread_s = 0.006 * 2 ** (-0.5 * c.hard)
    t = np.sort(np.abs(r.normal(0.0, spread_s / 2.5, 50)))
    a = np.exp(0.5 * r.standard_normal(50)) * np.exp(-t / spread_s)
    head = D.train(n, m['y'], t, a / a.sum())
    mot = D.train(n, m['motion'], t, a / a.sum())[:, 0]
    mot *= vel / (np.abs(mot).max() + 1e-12)
    w = _wires(r, mot, 1.45 * f, vel, 0.03, 0.03, c.bright)
    th = np.arange(dsp.n_of(0.06)) / SR
    hiss = dsp.butter(r.standard_normal((len(th), 2)), 'band', (2500.0, 11000.0), 2)
    hiss *= ((th / 0.004).clip(0, 1) * np.exp(-th / 0.018))[:, None]
    return _out(c, r, _mix([(head, 0.0), (w, -5.0), (hiss, -9.0 + 3 * c.bright)], n), 'studio', -0.2)


@design('snap/finger', kind='click',
        says=('a finger snap', 'someone snapping their fingers once', 'a sharp crisp finger click'),
        uses=('snap', 'click', 'select', 'attention', 'cool'), tags=('drums', 'body'), domains=DOM, dur=0.15,
        axes=AXES)
def snap_finger(c, r):
    """a snap (no public-domain recording of one: invented): the thumb slips off the middle finger (a faint skin
    hiss), the finger strikes the palm's base and the cavity between them pops: shorter, sharper and higher than a
    clap (2.2-3.5 kHz)"""
    vel = _vel(c, r)
    kf, _ = _size(c)
    y = D.snap(dsp.n_of(0.15), r, 0.025, 2800.0 * kf * 2 ** (0.3 * c.bright + 0.2 * r.standard_normal()),
               5.0 * 2 ** (0.5 * c.hard), vel)
    y = modal.early(fx.decorrelate(y, 0.2), _built('snap'), 0.15 * _ring(c))
    return _out(c, r, y, 'room', -0.2)


@design('sticks/count', kind='click',
        says=('two drumsticks clicked together', 'a dry wooden stick click', 'a drummer clicking sticks to count in'),
        uses=('count', 'countdown', 'count-in', 'click', 'tick', 'metronome', 'start'), tags=('drums', 'kit'),
        domains=DOM, dur=0.15, axes=AXES)
def sticks_count(c, r):
    """two hickory sticks crossed and struck near their shoulders (no public-domain recording: invented): each a
    round rod held at its fulcrum (its modes from about 400 Hz, quickly damped by the grip), wood on wood, so the
    click is the upper modes and the contact's tick, plus the dry crack of the grain at the contact"""
    built = _built('sticks')
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    n = dsp.n_of(0.15 * kl * g)
    out = []
    for f0, strike in ((400.0, 0.18), (430.0, 0.22)):
        s = modal.tube_modes(f0 * kf, 1, gyration=0.0036, c=4300.0, strike=strike * r.uniform(0.9, 1.1), count=7,
                             shear=3.9, rest=0.3)
        fr = f0 * kf * s['ratio']
        out.append(D.body(fr, 0.04 * kl * g * s['ratio'] ** -0.5,
                          0.3 * np.abs(s['gain']) * (0.15 + s['held']) ** -0.5, n, r, built, vel=vel, beater='wood',
                          hard=c.hard + 0.6, click=4.0))
    tk = np.arange(dsp.n_of(0.008)) / SR
    crack = dsp.butter(r.standard_normal((len(tk), 2)), 'band', (1200.0, 7000.0), 2) * \
        ((tk / 0.0003).clip(0, 1) * np.exp(-tk / 0.0012))[:, None]
    return _out(c, r, _mix([(out[0], 0.0), (out[1], -2.0), (crack, -6.0 + 3.0 * c.hard)], n), 'room', -0.3)
