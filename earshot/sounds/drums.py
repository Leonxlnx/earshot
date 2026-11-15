"""Drums and percussion: a kit (kicks, snares, rims, toms, hi-hats, cymbals), hands (claps, a snap), shaken and
hand-struck percussion (shaker, tambourine, conga, bongo, cajon, clave, sticks) and the classic synthetic drum
voices, for music beds, motion design and games. Every hit should be heard as a drum played by a person in a room,
never as a sine with a noise burst.

Each design is built from the engine in earshot.synth.drums, layer by layer:
    contact   the beater's Hertz contact (felt, plastic, a stick's shoulder or tip, a brush, a palm, fingertips):
              its length follows hardness and velocity, so a harder hit is brighter as well as louder
    body      the head's inharmonic modes (membrane: Bessel ratios moved by air loading and the second head; the
              pitch drops in the first tens of ms after a hard hit), a cymbal's hundreds of modes whose energy climbs
              upward after the strike, a wooden bar or box, a hoop
    rattle    the snare wires driven by the bottom head, the sizzle of two hi-hat plates, the cajon's wires, the
              jingles and beads of shaken things: collisions whose rate and size follow the motion
    space     one shared room per family (earshot.synth.space), after a close stereo pair with the stand's and the
              floor's earliest reflections

    from earshot.sounds import registry, drums       # drums registers its designs on import
    y, meta = registry.render('snare/tight', seed=2, energy=0.6)
    y, meta = registry.render('tom/floor', pitch='E2', ring=0.4)
    y, meta = registry.render('drum/roll', dur=2.5)

The axes: energy = how hard it is played (velocity: brighter, a bigger pitch drop, more wire and shimmer, then
louder), size = a bigger drum (lower, longer), bright = more top, hard = a harder beater or stick (a shorter
contact), ring = less damping (a longer ring), space = more room, width = a wider pair. `pitch` tunes the drums that
have one. Seeds are takes: the instrument stays the same (its modes are drawn from a generator fixed by the design),
the hit moves: velocity by a few per cent, the strike point, the contact noise, the timing of the rattles.
"""
import zlib

import numpy as np

from ..forge import dsp, fx
from ..synth import drums as D
from ..synth import modal, space
from .registry import design

SR = dsp.SR
AXES = ('energy', 'size', 'bright', 'hard', 'ring', 'space', 'width')
DOM = ('music', 'motion', 'game')


# ------------------------------------------------------------------------------------------------------ controls
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


def _norm(x):
    return x / (np.abs(x).max() + 1e-12)


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
            dsp.place(out, _norm(x), t, 10 ** (g / 20))
    return out


def _shelf(y, db, fc=3000.0):
    return y + (10 ** (db / 20) - 1) * dsp.butter(y, 'high', fc, order=1) if db else y


def _out(c, r, y, room='studio', amount=0.0, tilt=0.0):
    """the take finished: width, a high shelf for bright (a harder hit is a little brighter too), the shared room"""
    y = _st(y)
    y = fx.width(y, c.lerp('width', 0.2, 1.8, 1.0))
    y = _shelf(y, 6.0 * c.bright + 1.5 * c.energy + tilt)
    return space.room(y, c.room(room), amount=float(np.clip(c.space + amount, -1.0, 1.0)), r=r)


def _n(seconds):
    return dsp.n_of(seconds)


def _wires(r, motion, f_bottom, vel, gap=0.05, release=0.022, bright=0.0, body=0.3):
    """the snare wires under a head whose displacement is `motion`: the bottom head (tuned f_bottom Hz) drives them"""
    bands = ((1700.0 * 2 ** (0.3 * bright), 1.2, 1.0), (3400.0 * 2 ** (0.3 * bright), 1.5, 0.9),
             (6500.0 * 2 ** (0.3 * bright), 1.8, 0.45 + 0.3 * vel))
    return D.rattle(D.bottom_head(motion, f_bottom), r, gap, release, bands, body)


# ------------------------------------------------------------------------------------------------------ kicks
def _kick(c, r, name, f, beater, t60, damp, fall, strike, click_db, radius, room_amount=-0.3, dur=0.9):
    """a kick drum: the batter head (load 1, seal 1: the lowest modes near 1 : 1.5 : 2.3) struck near the centre,
    the shell and the air in it (a broad resonance an octave and a half over the head, short), the beater's click
    (the skin's slap, 2-5 kHz, its level by hardness and velocity), a little tape on the close mic as recorded"""
    built = _built(name)
    vel = _vel(c, r)
    kf, kl = _size(c)
    f = c.hz(f) * kf
    g = _ring(c)
    hard = c.hard
    m = D.membrane(f, _n(dur * kl * g), r, built, vel=vel, beater=beater, hard=hard,
                   strike=strike * r.uniform(0.8, 1.2), t60=t60 * kl * g, damp=damp / g, load=1.0, seal=1.0,
                   count=20, fall=fall, radius=radius / kf, mono=1.3, click=0.0, spread=0.15,
                   shell=((2.6 * f, 0.06, 0.08), (4.3 * f, 0.04, 0.05)))
    tau, p, ck, grit = D.contact(beater, vel, hard)
    skin = D._click(tau, r, p, 0.25 + ck, 0.4 + grit, _n(0.03), low=1200.0, ring_s=0.0025)
    skin = dsp.svf(skin, 3200.0 * 2 ** (0.4 * hard + 0.3 * c.bright), 1.2, 'bandn') + 0.4 * skin
    y = _mix([(m['y'], 0.0), (skin, click_db + 4.0 * hard + 4.0 * (vel - 0.7))])
    y = fx.saturate(y, 3.0 + 3.0 * vel, 'tape', level='peak')
    return _out(c, r, y, 'booth', room_amount)


@design('kick/soft', kind='thud',
        says=('a soft round kick drum, a felt beater on a damped bass drum', 'a gentle muffled bass drum thump',
              'a warm low drum thud'),
        uses=('drum', 'kick drum', 'beat', 'logo', 'soft impact', 'heartbeat', 'pulse'), tags=('drums', 'kit'),
        domains=DOM, dur=0.6, pitch='A1', axes=AXES)
def kick_soft(c, r):
    """a felt beater on a kick with a pillow in it: a long soft contact (no click to speak of), struck in the
    centre (the (0,n) thud), heavy damping on the upper modes, a short ring"""
    return _kick(c, r, 'kick/soft', 'A1', 'felt', 0.42, 1.4, 0.09, 0.18, -20.0, 0.27, dur=0.7)


@design('kick/punch', kind='thud',
        says=('a tight punchy kick drum with a clear click', 'a hard rock kick drum hit',
              'a short bass drum with a sharp beater attack'),
        uses=('drum', 'kick drum', 'beat', 'impact', 'hit', 'accent', 'downbeat'), tags=('drums', 'kit'),
        domains=DOM, dur=0.5, pitch='A#1', axes=AXES)
def kick_punch(c, r):
    """a plastic beater on a damped 22-inch kick: a short contact (the click of the batter head), a little off
    the centre so the (1,1) adds punch, a tight ring and a strong pitch drop on a hard hit"""
    return _kick(c, r, 'kick/punch', 'A#1', 'plastic', 0.31, 0.7, 0.15, 0.28, -8.0, 0.27, dur=0.6)


@design('kick/deep', kind='thud',
        says=('a big open bass drum booming', 'a large deep kick drum with a long low ring',
              'a deep resonant drum boom'),
        uses=('drum', 'kick drum', 'beat', 'impact', 'boom', 'logo', 'trailer hit'), tags=('drums', 'kit'),
        domains=DOM, dur=1.2, pitch='F1', axes=AXES)
def kick_deep(c, r):
    """a 24-inch kick with no pillow and a resonant front head: the low modes ring for most of a second, the
    felt beater a little harder, more room"""
    return _kick(c, r, 'kick/deep', 'F1', 'felt', 0.85, 0.25, 0.12, 0.15, -14.0, 0.31, room_amount=0.0, dur=1.5)


@design('kick/electro', kind='thud',
        says=('an electronic kick drum with a long booming sub tail', 'a deep synthesized bass drum boom',
              'a drum machine kick with a falling pitch'),
        uses=('drum', 'kick drum', 'beat', 'drop', 'bass hit', 'logo', 'trap', 'hip hop', 'edm'),
        tags=('drums', 'electronic'), domains=DOM, dur=1.0, pitch='F#1', axes=AXES)
def kick_electro(c, r):
    """the synthetic kick (kick_synth): a sine falling 2.6 octaves onto its pitch in two stages, a long sub tail,
    tape saturation and a click; harder = a faster snap and more click, ring = the tail's length"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    f = c.hz('F#1') * kf
    n = _n(1.3 * kl * g)
    y = D.kick_synth(n, r, f, vel, sweep=2.6 * (1 + 0.15 * c.hard), drop=0.045 * kl, snap_=0.006 * 2 ** (-0.5 * c.hard),
                     decay=0.42 * kl * g, click=0.6 * 2 ** c.hard * 10 ** (c.bright / 4), drive=6.0 + 3.0 * c.bright)
    y = dsp.fade(y, 0.0, 0.3 * len(y) / SR)
    return _out(c, r, fx.decorrelate(y, 0.15), 'booth', -0.6)


# ------------------------------------------------------------------------------------------------------ snares
def _snare(c, r, name, f, t60, damp, gap, release, wire_db, strike=0.32, beater='stick', shell=None, vel=None,
           fall=0.045, dur=0.7, rim=0.0):
    """a snare drum: the batter head off the centre, the shell (maple: a few broad modes), the bottom head an
    octave-and-a-half up driving the wires, the stick's tick; rim > 0 adds a stick across the hoop (a rimshot)"""
    built = _built(name)
    vel = _vel(c, r) if vel is None else vel
    kf, kl = _size(c)
    f = c.hz(f) * kf
    g = _ring(c)
    n = _n(dur * kl * max(g, 1.0))
    shell = shell or ((2.4 * f, 0.05, 0.03), (3.9 * f, 0.035, 0.02))
    m = D.membrane(f, n, r, built, vel=vel, beater=beater, hard=c.hard, strike=strike * r.uniform(0.85, 1.15),
                   t60=t60 * kl * g, damp=damp / g, load=0.5, seal=0.6, count=26, fall=fall, radius=0.18 / kf,
                   click=0.5, spread=0.3, shell=shell)
    w = _wires(r, m['motion'], 1.45 * f, vel, gap, release * g ** 0.5, c.bright)
    layers = [(m['y'], 0.0), (w, wire_db + 3.0 * (vel - 0.7))]
    hf, hg = D.hoop(820.0 * built.uniform(0.97, 1.03), n)
    if rim:
        hp = D.body(hf, 0.35 * hf[0] / hf, hg, n, r, built, vel=vel, beater='rim', hard=c.hard, click=2.0)
        layers.append((hp, -4.0 + rim))
    else:                    # the hoop and the lugs ring faintly with every stroke, through the head's edge
        hp = D.body(hf, 0.25 * hf[0] / hf, hg, n, r, built, vel=vel, beater='stick', click=0.0)
        layers.append((hp, -24.0))
    return _mix(layers, n)


@design('snare/tight', kind='hit',
        says=('a tight snare drum hit', 'a crisp snare drum crack with a short buzz of the wires',
              'a stick hitting a snare drum'),
        uses=('drum', 'snare', 'beat', 'hit', 'accent', 'backbeat', 'cut'), tags=('drums', 'kit'), domains=DOM,
        dur=0.4, pitch='B3', axes=AXES)
def snare_tight(c, r):
    """a 14-inch snare tuned high (the (0,1) near 250 Hz), lightly damped, wires tight to the bottom head: a
    crack, a short bright buzz that follows the head and a little shell"""
    return _out(c, r, _snare(c, r, 'snare/tight', 'B3', 0.42, 0.7, 0.015, 0.05, -3.0), 'studio', -0.2)


@design('snare/fat', kind='hit',
        says=('a fat low snare drum with a long buzz', 'a deep snare drum hit with loose rattling wires',
              'a big snare drum crack'),
        uses=('drum', 'snare', 'beat', 'hit', 'backbeat', 'accent'), tags=('drums', 'kit'), domains=DOM, dur=0.6,
        pitch='F#3', axes=AXES)
def snare_fat(c, r):
    """a deep snare tuned low with the wires loose: the head rings longer, the wires engage earlier and bounce
    longer between slaps (more body in the buzz), a deeper shell"""
    f = c.hz('F#3')
    return _out(c, r, _snare(c, r, 'snare/fat', 'F#3', 0.5, 0.35, 0.008, 0.05, -1.0,
                             shell=((2.1 * f, 0.07, 0.04), (3.4 * f, 0.05, 0.03)), dur=0.9), 'studio', 0.0)


@design('snare/brush', kind='hit',
        says=('a soft brush tap on a snare drum', 'a jazz brush hitting a snare, a soft swish with a gentle buzz',
              'a wire brush on a drum'),
        uses=('drum', 'snare', 'brush', 'jazz', 'soft hit', 'tap', 'beat'), tags=('drums', 'kit'), domains=DOM,
        dur=0.35, pitch='A3', axes=AXES)
def snare_brush(c, r):
    """a wire brush: its bristles land over a few milliseconds (some 50 small contacts, each a tiny stroke), so
    the head is driven by a spread-out train instead of one blow (no crack), the wires buzz softly, and the
    bristles themselves hiss as they splay on the coated head"""
    built = _built('snare/brush')
    vel = _vel(c, r, 0.45)
    kf, kl = _size(c)
    f = c.hz('A3') * kf
    g = _ring(c)
    n = _n(0.6 * kl * g)
    m = D.membrane(f, n, r, built, vel=vel, beater='brush', hard=c.hard, strike=0.45, t60=0.3 * kl * g,
                   damp=0.8 / g, load=0.5, seal=0.6, count=26, fall=0.0, radius=0.18 / kf, click=0.0, glide=False)
    spread_s = 0.006 * 2 ** (-0.5 * c.hard)
    k = 50
    t = np.sort(np.abs(r.normal(0.0, spread_s / 2.5, k)))
    a = np.exp(0.5 * r.standard_normal(k)) * np.exp(-t / spread_s)
    head = D.train(n, m['y'], t, a / a.sum())
    mot = D.train(n, m['motion'], t, a / a.sum())[:, 0]
    mot *= vel / (np.abs(mot).max() + 1e-12)
    w = _wires(r, mot, 1.45 * f, vel, 0.03, 0.03, c.bright)
    hiss_n = _n(0.06)
    th = np.arange(hiss_n) / SR
    hiss = dsp.butter(r.standard_normal((hiss_n, 2)), 'band', (2500.0, 11000.0), 2)
    hiss *= ((th / 0.004).clip(0, 1) * np.exp(-th / 0.018))[:, None]
    y = _mix([(head, 0.0), (w, -5.0), (hiss, -9.0 + 3 * c.bright)], n)
    return _out(c, r, y, 'studio', -0.2)


@design('snare/electro', kind='hit',
        says=('an electronic snare drum from a drum machine', 'a synthesized snare, a tone and a burst of noise',
              'a drum machine snare hit'),
        uses=('drum', 'snare', 'beat', 'backbeat', 'edm', 'hip hop', 'hit'), tags=('drums', 'electronic'),
        domains=DOM, dur=0.35, pitch='F#3', axes=AXES)
def snare_electro(c, r):
    """the synthetic snare (snare_synth): two tuned tones and a burst of filtered noise with its own longer
    envelope; ring lengthens the noise, hard sharpens it, size lowers the tones"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(0.45 * kl * g)
    y = D.snare_synth(n, r, c.hz('F#3') * kf, vel, tone=0.55 * 10 ** (-c.hard / 8), noise_tau=0.12 * kl * g,
                      snappy=1.0, bright=c.bright)
    y = dsp.fade(y, 0.0, 0.08)
    return _out(c, r, fx.decorrelate(y, 0.3), 'room', -0.3)


@design('rim/shot', kind='hit',
        says=('a snare drum rimshot, a loud ringing crack', 'a stick hitting the rim and the head of a snare at once',
              'a sharp metallic snare crack'),
        uses=('drum', 'snare', 'rimshot', 'accent', 'hit', 'impact', 'cut'), tags=('drums', 'kit'), domains=DOM,
        dur=0.5, pitch='B3', axes=AXES)
def rim_shot(c, r):
    """a rimshot: the stick lands on the head near the edge and across the metal hoop in the same instant: the
    hoop rings (its in-plane bending modes from 820 Hz, a few tenths of a second) with a hard metal crack, the
    head is driven at the edge (more ring, the upper modes), the wires buzz hard"""
    vel = max(0.85, _vel(c, r, 0.9))
    return _out(c, r, _snare(c, r, 'rim/shot', 'B3', 0.4, 0.5, 0.04, 0.02, -4.0, strike=0.85, vel=vel,
                             rim=2.0 + 2.0 * c.hard), 'studio', 0.0)


@design('rim/click', kind='click',
        says=('a cross-stick click on a snare drum', 'a dry woody rim click', 'a hollow wooden knock on a drum rim'),
        uses=('drum', 'rim', 'click', 'beat', 'bossa nova', 'ballad', 'tick', 'count'), tags=('drums', 'kit'),
        domains=DOM, dur=0.25, pitch='B3', axes=AXES)
def rim_click(c, r):
    """a cross-stick: the stick lies on the head with its butt across the hoop and is lifted and dropped onto
    it: a dry woody click (the stick's own bending modes, held in the hand and damped), the hoop's contact, the
    shell's resonance (the hollow tone) and only a little head and wire"""
    built = _built('rim/click')
    vel = _vel(c, r, 0.6)
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(0.3 * kl * g)
    s = modal.tube_modes(420.0 * kf, mode=1, gyration=0.0036, c=4300.0, strike=0.08, count=6, shear=3.9,
                         rest=0.3)
    fs = 420.0 * kf * s['ratio']
    stick = D.body(fs, 0.05 * g * (fs / fs[0]) ** -0.4, np.abs(s['gain']) + 0.1, n, r, built, vel=vel,
                   beater='wood', hard=c.hard + 0.5, click=1.5)
    f = c.hz('B3') * kf
    shell = D.body(np.array([2.3, 3.6, 5.1]) * f, np.array([0.09, 0.06, 0.04]) * kl * g, [1.0, 0.7, 0.4], n, r,
                   built, vel=vel, beater='wood', hard=c.hard, click=0.0)
    m = D.membrane(f, n, r, built, vel=0.35 * vel, beater='wood', strike=0.8, t60=0.18 * kl * g, damp=1.0,
                   load=0.5, seal=0.6, fall=0.0, radius=0.18, click=0.0)
    w = _wires(r, m['motion'], 1.45 * f, 0.35 * vel, 0.08, 0.015, c.bright)
    y = _mix([(stick, 0.0), (shell, -3.0), (m['y'], -10.0), (w, -22.0)], n)
    return _out(c, r, y, 'studio', -0.2)


# ------------------------------------------------------------------------------------------------------ toms
def _tom(c, r, name, note, t60, radius, dur):
    """a tom-tom: two heads and a shell; the pitch is heard on the (1,1) mode (the (0,1) couples to the second
    head and the air in the shell and dies fast); struck a third of the way in; a hard hit bends the pitch down
    by a semitone or more"""
    built = _built(name)
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    f = c.hz(note) * kf
    n = _n(dur * kl * g)
    m = D.membrane(f, n, r, built, vel=vel, beater='stick', hard=c.hard, strike=0.38 * r.uniform(0.85, 1.15),
                   t60=t60 * kl * g, damp=0.25 / g, load=0.7, seal=0.5, tune=1, count=24, fall=0.1, mono=2.5,
                   radius=radius / kf, spread=0.35, shell=((2.9 * f, 0.08, 0.03),))
    return _out(c, r, m['y'], 'studio', 0.0)


@design('tom/high', kind='hit',
        says=('a high tom drum hit', 'a small tom-tom with a short pitched ring', 'a stick on a high rack tom'),
        uses=('drum', 'tom', 'fill', 'beat', 'hit', 'accent'), tags=('drums', 'kit'), domains=DOM, dur=0.8,
        pitch='F#3', axes=AXES)
def tom_high(c, r):
    """a 10-inch rack tom"""
    return _tom(c, r, 'tom/high', 'F#3', 0.7, 0.13, 1.1)


@design('tom/mid', kind='hit',
        says=('a medium tom drum hit', 'a tom-tom with a round pitched ring', 'a stick on a rack tom'),
        uses=('drum', 'tom', 'fill', 'beat', 'hit', 'accent'), tags=('drums', 'kit'), domains=DOM, dur=1.0,
        pitch='C#3', axes=AXES)
def tom_mid(c, r):
    """a 12-inch rack tom"""
    return _tom(c, r, 'tom/mid', 'C#3', 0.9, 0.155, 1.3)


@design('tom/floor', kind='hit',
        says=('a deep floor tom hit', 'a big low tom drum with a long ring', 'a stick on a floor tom'),
        uses=('drum', 'tom', 'fill', 'beat', 'hit', 'impact', 'tribal'), tags=('drums', 'kit'), domains=DOM, dur=1.4,
        pitch='F#2', axes=AXES)
def tom_floor(c, r):
    """a 16-inch floor tom"""
    return _tom(c, r, 'tom/floor', 'F#2', 1.25, 0.2, 1.8)


# ------------------------------------------------------------------------------------------------------ hi-hats
def _hat(c, r, name, vel, t60, sizzle, gap, dur, beater='tip', shimmer=0.6, chick=0.0, tick=1.2):
    built = _built('hat')                    # the same pair of hi-hat cymbals for every hi-hat design
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(dur * kl * g)
    y = D.cymbal(n, r, built, vel=vel, beater=beater, hard=c.hard, f_lo=700.0 * kf, f_hi=17500.0, density=0.03,
                 t60=(t60[0] * kl * g, t60[1] * kl * g), tilt=0.6 + 0.2 * c.bright, shimmer=shimmer, build=0.03,
                 cross=3500.0, tick=tick, sizzle=sizzle, gap=gap, spread=0.5)['y']
    if chick:
        k = _n(0.012)
        tk = np.arange(k) / SR
        air = dsp.butter(r.standard_normal((k, 2)), 'band', (1500.0, 7000.0), 2) * \
            ((tk / 0.0015).clip(0, 1) * np.exp(-tk / 0.003))[:, None]
        y = _mix([(y, 0.0), (air, chick)], n)
    return y


@design('hat/closed', kind='tick',
        says=('a closed hi-hat, a short tight chick', 'a stick tapping closed hi-hat cymbals',
              'a crisp short metallic tick of a hi-hat'),
        uses=('drum', 'hi-hat', 'beat', 'tick', 'groove', 'rhythm'), tags=('drums', 'kit'), domains=DOM, dur=0.2,
        axes=AXES)
def hat_closed(c, r):
    """two 14-inch plates clamped together: their contact drains every mode in a tenth of a second or less
    (choke), and while they ring they buzz against each other (sizzle): a short tight chick after the tick"""
    vel = _vel(c, r, 0.65)
    return _out(c, r, _hat(c, r, 'hat/closed', vel, (0.26, 0.1), 0.9, 0.1, 0.4), 'studio', -0.3)


@design('hat/open', kind='hit',
        says=('an open hi-hat, a sizzling cymbal wash', 'a stick hitting a loose open hi-hat',
              'a bright metallic sizzle ringing out'),
        uses=('drum', 'hi-hat', 'beat', 'groove', 'accent', 'swish'), tags=('drums', 'kit'), domains=DOM, dur=1.0,
        axes=AXES)
def hat_open(c, r):
    """the two plates a few millimetres apart: they ring for a second, and the larger motion of a struck plate
    keeps meeting the other one (a long sizzle that fades with the ring)"""
    vel = _vel(c, r, 0.75)
    return _out(c, r, _hat(c, r, 'hat/open', vel, (1.6, 0.45), 0.9, 0.18, 1.6, shimmer=0.5), 'studio', 0.0)


@design('hat/pedal', kind='tick',
        says=('a hi-hat closed with the foot pedal, a soft chick', 'two cymbals clapped together softly',
              'a soft short metallic chick'),
        uses=('drum', 'hi-hat', 'beat', 'groove', 'tick', 'rhythm'), tags=('drums', 'kit'), domains=DOM, dur=0.15,
        axes=AXES)
def hat_pedal(c, r):
    """the pedal brings the top plate down onto the bottom one: metal on metal but no stick (a faint tick), the
    air squeezed out between them (a short puff), the plate landing two or three times within 10 ms as the
    pedal settles, then the clamped plates' brief ring and their sizzle"""
    vel = _vel(c, r, 0.5)
    y = _hat(c, r, 'hat/pedal', vel, (0.22, 0.09), 1.0, 0.08, 0.3, shimmer=0.2, chick=-6.0, tick=0.3)
    d = r.uniform(0.003, 0.006)                  # the top plate lands, lifts and lands again as the pedal settles,
    for t, g in ((d, 0.45), (1.9 * d, 0.2)):     # each time touching at other points (another take of the plates)
        b = _hat(c, r, 'hat/pedal', vel * g, (0.22, 0.09), 1.0, 0.08, 0.3, shimmer=0.0, tick=0.3)
        y = _mix([(y, 0.0), (b, 20 * np.log10(g), t)], len(y))
    return _out(c, r, y, 'studio', -0.3)


@design('hat/electro', kind='tick',
        says=('an electronic hi-hat tick from a drum machine', 'a short synthesized metallic hi-hat',
              'a crisp drum machine hat'),
        uses=('drum', 'hi-hat', 'beat', 'trap', 'edm', 'tick', 'rhythm'), tags=('drums', 'electronic'), domains=DOM,
        dur=0.12, axes=AXES)
def hat_electro(c, r):
    """the synthetic hi-hat (hat_synth): six inharmonic square waves through two band-passes and a high-pass, a
    sharp decay; ring = the decay, size = the oscillators' scale, hard = a sharper start"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(0.25 * kl * g)
    y = D.hat_synth(n, r, vel, decay=0.03 * kl * g, scale=kf)
    a = _n(0.0004 * 2 ** (-c.hard))
    y[:a] *= np.linspace(0, 1, a)
    return _out(c, r, fx.decorrelate(dsp.fade(y, 0.0, 0.05), 0.4), 'room', -0.3)


# ------------------------------------------------------------------------------------------------------ cymbals
def _cym(c, r, name, vel, f_lo, f_hi, t60, tilt, shimmer, cross, dur, bell=0.0, bell_hz=620.0, beater='tip',
         tick=1.0, density=0.035):
    built = _built(name)
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(dur * kl * g)
    y = D.cymbal(n, r, built, vel=vel, beater=beater, hard=c.hard, f_lo=f_lo * kf, f_hi=f_hi, density=density,
                 t60=(t60[0] * kl * g, t60[1] * kl * g), tilt=tilt + 0.2 * c.bright, shimmer=shimmer, cross=cross,
                 tick=tick, bell=bell, bell_hz=bell_hz * kf, bell_t60=2.2 * kl * g, spread=0.7)['y']
    return dsp.fade(y, 0.0, 0.4 * dur)


@design('crash/bright', kind='hit',
        says=('a bright crash cymbal', 'a loud shimmering cymbal crash', 'a splashy metallic cymbal hit'),
        uses=('drum', 'crash', 'cymbal', 'accent', 'impact', 'finale', 'reveal', 'transition'),
        tags=('drums', 'kit'), domains=DOM, dur=3.0, axes=AXES, allows=('ring',))
def crash_bright(c, r):
    """a thin 16-inch crash hit on the edge with the stick's shoulder: a fast bright wash whose energy climbs
    into the top within 50-100 ms of the hit, then a two- to three-second decay"""
    vel = _vel(c, r, 0.8)
    return _out(c, r, _cym(c, r, 'crash/bright', vel, 380.0, 17000.0, (3.2, 0.8), 0.0, 0.9, 1800.0, 3.6,
                           beater='shoulder'), 'studio', 0.0)


@design('crash/dark', kind='hit',
        says=('a dark crash cymbal', 'a deep washy cymbal crash', 'a large trashy cymbal hit'),
        uses=('drum', 'crash', 'cymbal', 'accent', 'impact', 'finale', 'transition'), tags=('drums', 'kit'),
        domains=DOM, dur=4.0, axes=AXES, allows=('ring',))
def crash_dark(c, r):
    """a heavier 19-inch crash: lower modes, longer, less top; the wash swells more slowly"""
    vel = _vel(c, r, 0.8)
    return _out(c, r, _cym(c, r, 'crash/dark', vel, 260.0, 13000.0, (4.6, 1.1), -0.35, 0.7, 1500.0, 4.6,
                           beater='shoulder', density=0.04), 'studio', 0.0)


@design('ride/ping', kind='hit',
        says=('a ride cymbal ping', 'a stick tapping a ride cymbal, a clear ping with a long shimmer',
              'a bright metallic ping that rings on'),
        uses=('drum', 'ride', 'cymbal', 'jazz', 'groove', 'beat', 'rhythm'), tags=('drums', 'kit'), domains=DOM,
        dur=3.0, axes=AXES, allows=('ring',))
def ride_ping(c, r):
    """the stick's tip on the bow of a heavy 21-inch ride: a clear tick and ping, little wash (a thick plate:
    the coupling is weak at this level), a long even decay"""
    vel = _vel(c, r, 0.6)
    return _out(c, r, _cym(c, r, 'ride/ping', vel, 300.0, 16000.0, (6.0, 1.3), -0.15, 0.5, 2500.0, 3.5,
                           tick=1.6), 'studio', 0.0)


@design('ride/bell', kind='chime',
        says=('a ride cymbal bell', 'a clear bright bell-like cymbal ping', 'a stick on the bell of a cymbal'),
        uses=('drum', 'ride', 'cymbal', 'bell', 'accent', 'groove', 'latin'), tags=('drums', 'kit'), domains=DOM,
        dur=2.5, axes=AXES, allows=('ring',))
def ride_bell(c, r):
    """the stick's shoulder on the dome of the ride: the dome's few clear partials ring like a bell over a
    quieter wash of the bow"""
    vel = _vel(c, r, 0.75)
    return _out(c, r, _cym(c, r, 'ride/ping', vel, 300.0, 16000.0, (5.0, 1.2), -0.25, 0.15, 3000.0, 3.2,
                           bell=2.8, bell_hz=640.0, beater='shoulder', tick=1.0), 'studio', 0.0)


@design('cymbal/swell', kind='rise',
        says=('a cymbal swell rolled with soft mallets', 'a rising shimmering cymbal roll',
              'a cymbal growing louder into a wash'),
        uses=('swell', 'riser', 'build', 'transition', 'reveal', 'intro', 'tension', 'cymbal'),
        tags=('drums', 'orchestral'), domains=DOM, dur=2.5, axes=AXES,
        allows=('pre_swell', 'truncated', 'ring'))
def cymbal_swell(c, r):
    """two soft yarn mallets rolling on a suspended cymbal for dur s (alternating, about 9 strokes a second each,
    the hands never quite even), louder and louder: each stroke a dark long contact; the plate is linear at first,
    but as it moves more its energy climbs into the top (the strokes' squared level drives the bright modes),
    so the swell opens up as it grows, then rings out"""
    built = _built('cymbal/swell')
    T = c.seconds(2.5)
    kf, kl = _size(c)
    g = _ring(c)
    vel = _vel(c, r, 0.7)
    tail = 2.3 * kl * g
    n = _n(T + tail)
    dark = D.cymbal(_n(tail + 0.5), r, built, vel=0.45, beater='mallet', hard=c.hard, f_lo=300.0 * kf,
                    f_hi=12000.0, density=0.03, t60=(4.0 * kl * g, 1.0 * kl * g), tilt=-0.6, shimmer=0.0, tick=0.0,
                    spread=0.7, early=0.0)['y']
    bright = D.cymbal(_n(tail + 0.5), r, built, vel=0.9, beater='tip', f_lo=2500.0, f_hi=16000.0, density=0.025,
                      t60=(2.0 * kl * g, 0.7 * kl * g), tilt=-0.3 + 0.3 * c.bright, shimmer=0.0, tick=0.0,
                      spread=0.8, early=0.0)['y']
    rate = 18.0 * r.uniform(0.92, 1.08)
    t = np.arange(0.0, T, 1.0 / rate)
    t = t + r.normal(0.0, 0.006, len(t)) + 0.012 * (np.arange(len(t)) % 2)
    t = np.clip(t, 0.0, T)
    u = np.clip(t / T, 0, 1)
    a = (0.04 + 0.96 * u ** 2.2) * np.exp(0.12 * r.standard_normal(len(t))) * (0.6 + 0.4 * vel)
    y = D.train(n, dark, t, a) + (0.25 + 0.35 * vel) * D.train(n, bright, t, a ** 2)
    y = modal.early(y, built, 0.1)
    return _out(c, r, dsp.fade(y, 0.0, 0.4 * tail), 'studio', 0.0)


# ------------------------------------------------------------------------------------------------------ hands
def _claps(c, r, people, spread_s, vel, flat=False):
    """`people` hand claps scattered over spread_s s (normal timing spread), each hand shape its own cavity
    (0.8-2.2 kHz, a broad peak), its own place across the pair. flat: hands clapping hard and flat, as a crowd
    does, the fingers meeting the other palm 1-2.5 ms before the palms do (a second, higher slap)"""
    n = _n(0.25 + spread_s * 3)
    out = np.zeros((n, 2))
    kf, _ = _size(c)
    t0 = np.sort(np.abs(r.normal(0.0, spread_s, people)))
    t0 -= t0.min()
    for i in range(people):
        cav = 1300.0 * kf * 2 ** r.uniform(-0.7, 0.7) * 2 ** (0.3 * c.bright)
        x = D.clap(n, r, 0.004 + t0[i], cav, r.uniform(1.6, 3.0), vel * r.uniform(0.75, 1.0),
                   slap=0.6 * 2 ** c.hard, length=0.0022 * 2 ** (-0.6 * c.hard), flesh=0.12)
        if flat:
            x += 0.5 * D.clap(n, r, 0.004 + t0[i] - r.uniform(0.001, 0.0025), 1.4 * cav, 2.0, 0.75 * vel, 0.8,
                              0.0008, 0.0)
        pan = 0.0 if people == 1 else r.uniform(-0.8, 0.8)
        out += dsp.pan_to_stereo(x, pan) * r.uniform(0.6, 1.0)
    return out


@design('clap/hands', kind='hit',
        says=('a single hand clap', 'two hands clapping once', 'a sharp clap of palms'),
        uses=('clap', 'applause', 'beat', 'accent', 'success', 'attention'), tags=('drums', 'body'), domains=DOM,
        dur=0.2, axes=AXES)
def clap_hands(c, r):
    """one clap: the palms slap together over a millisecond or two and the air pushed out of the cupped space
    between them rings at 1-2 kHz (a broad peak); a small room"""
    y = _claps(c, r, 1, 0.0, _vel(c, r))
    y = modal.early(fx.decorrelate(y.mean(1), 0.3), _built('clap/hands'), 0.25 * _ring(c))
    return _out(c, r, y, 'room', 0.35)


@design('clap/group', kind='hit',
        says=('a group of people clapping together once', 'several hand claps at once, a crowd clap',
              'a layered clap of many hands'),
        uses=('clap', 'applause', 'beat', 'accent', 'success', 'crowd'), tags=('drums', 'body'), domains=DOM,
        dur=0.35, axes=AXES, allows=('stacked',))
def clap_group(c, r):
    """ten to thirteen people clapping hard on the same beat: the claps scattered 5-25 ms apart (a timing spread
    of 15 ms), each hand shape its own cavity and place, in a hall"""
    people = int(r.integers(10, 14) * c.lerp('size', 0.5, 1.6, 1.0))
    y = _claps(c, r, max(3, people), 0.015 * _ring(c) ** 0.5, _vel(c, r), flat=True)
    return _out(c, r, y, 'hall', -0.4)


@design('clap/electro', kind='hit',
        says=('an electronic hand clap from a drum machine', 'a synthesized clap, a few quick bursts of noise',
              'a drum machine clap with a short tail'),
        uses=('clap', 'beat', 'backbeat', 'edm', 'hip hop', 'accent'), tags=('drums', 'electronic'), domains=DOM,
        dur=0.3, axes=AXES, allows=('stacked',))
def clap_electro(c, r):
    """the synthetic clap (clap_synth): four bursts of band-passed noise 10 ms apart and a tail; ring = the tail,
    size = a lower band, hard = sharper bursts"""
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(0.5 * kl * g)
    y = D.clap_synth(n, r, vel, gap=0.0105 * 2 ** (-0.2 * c.hard), centre=1150.0 * kf, tail=0.13 * kl * g)
    return _out(c, r, fx.decorrelate(dsp.fade(y, 0.0, 0.1), 0.5), 'room', 0.0)


@design('snap/finger', kind='click',
        says=('a finger snap', 'someone snapping their fingers once', 'a sharp crisp finger click'),
        uses=('snap', 'click', 'beat', 'select', 'attention', 'cool'), tags=('drums', 'body'), domains=DOM, dur=0.15,
        axes=AXES)
def snap_finger(c, r):
    """a snap: the thumb slips off the middle finger (a faint skin hiss), the finger strikes the palm's base and
    the cavity between them pops: shorter, sharper and higher than a clap (2.2-3.5 kHz)"""
    vel = _vel(c, r)
    kf, _ = _size(c)
    y = D.snap(_n(0.15), r, 0.025, 2800.0 * kf * 2 ** (0.3 * c.bright + 0.2 * r.standard_normal()),
               5.0 * 2 ** (0.5 * c.hard), vel)
    y = modal.early(fx.decorrelate(y, 0.2), _built('snap'), 0.15 * _ring(c))
    return _out(c, r, y, 'room', -0.2)


# ------------------------------------------------------------------------------------------------------ shaken
def _shell(c, r):
    kf, _ = _size(c)
    b = 2 ** (0.4 * c.bright)
    return ((4800.0 * kf * b, 2.5, 1.0), (7600.0 * kf * b, 2.2, 0.7), (2600.0 * kf, 3.0, 0.3 * (1 - 0.3 * c.bright)))


@design('shaker/tick', kind='tick',
        says=('a single shake of a shaker', 'one short stroke of an egg shaker, a quick chk',
              'a small rattle of beads in a shell'),
        uses=('shaker', 'beat', 'rhythm', 'tick', 'groove', 'percussion'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=0.2, axes=AXES)
def shaker_tick(c, r):
    """one stroke: the beads slide across the shell for 40-60 ms (a thin rustle) and then hit the far wall all
    at once (a dense burst), thinning out into single grains as they settle"""
    vel = _vel(c, r)
    g = _ring(c)
    n = _n(0.3 * g)
    y = D.shaker(n, r, [(0.055, vel)], beads=55.0 * c.lerp('size', 0.5, 2.0, 1.0, log=True),
                 settle=0.03 * g * 2 ** (-0.4 * c.hard), pre=0.15, travel=0.05, res=_shell(c, r),
                 grain=0.0004 * 2 ** (-0.6 * c.hard))
    return _out(c, r, dsp.fade(y, 0.0, 0.03), 'room', -0.3)


@design('shaker/roll', kind='noise',
        says=('a shaker being shaken steadily', 'a continuous rhythmic rattle of an egg shaker',
              'beads rattling back and forth in a shell'),
        uses=('shaker', 'groove', 'rhythm', 'percussion', 'beat', 'loop'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=2.0, axes=AXES, allows=('tail',))
def shaker_roll(c, r):
    """a steady shake for dur s: strokes at about 8 a second (the forward one stronger: the hand's accent),
    each a wall hit after a slide; the shaking hand drifts a little in time and force"""
    T = c.seconds(2.0)
    vel = _vel(c, r)
    g = _ring(c)
    n = _n(T + 0.2)
    rate = 8.0 * r.uniform(0.95, 1.05)
    t = np.arange(0.06, T, 1.0 / rate)
    t = t + r.normal(0, 0.004, len(t))
    e = vel * np.where(np.arange(len(t)) % 2 == 0, 1.0, 0.65) * np.exp(0.15 * r.standard_normal(len(t)))
    e *= np.clip(t / 0.15, 0.4, 1.0)
    y = D.shaker(n, r, list(zip(t, e)), beads=55.0 * c.lerp('size', 0.5, 2.0, 1.0, log=True),
                 settle=0.03 * g * 2 ** (-0.4 * c.hard), pre=0.25, travel=0.045, res=_shell(c, r),
                 grain=0.0004 * 2 ** (-0.6 * c.hard))
    return _out(c, r, dsp.fade(y, 0.002, 0.08), 'room', -0.2)


def _tamb(c, r, strokes, n, head=0.0, shake=0.0):
    built = _built('tambourine')
    kf, kl = _size(c)
    g = _ring(c)
    j = D.tambourine(n, r, built, strokes, jingles=8, f=(1150.0 * kf, 1650.0 * kf), t60=0.33 * kl * g,
                     restitution=0.55 * 2 ** (0.2 * c.hard), shake=shake)
    layers = [(j, 0.0)]
    if head:
        m = D.membrane(330.0 * kf, n, r, built, vel=strokes[0][1], beater='hand', hard=c.hard, strike=0.5,
                       t60=0.15 * kl * g, damp=1.0, load=0.2, count=16, fall=0.03, radius=0.12, click=0.5)
        slap = dsp.butter(r.standard_normal(_n(0.01)), 'band', (600.0, 5000.0), 2) * \
            np.exp(-np.arange(_n(0.01)) / (0.0015 * SR))
        layers += [(m['y'], head - 5.0), (slap, head - 8.0, strokes[0][0])]
    return _mix(layers, n)


@design('tambourine/hit', kind='hit',
        says=('a tambourine struck once', 'a hand hitting a tambourine, a bright jingle',
              'a burst of small metal jingles'),
        uses=('tambourine', 'beat', 'accent', 'backbeat', 'percussion', 'success'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=0.6, axes=AXES)
def tambourine_hit(c, r):
    """the palm strikes the head: a short thud of the small head and the hand's slap, and every pair of jingles
    on the frame is thrown apart and clashes again and again, faster and weaker (restitution), each pair a dished
    brass disc pair ringing in its own inharmonic modes"""
    vel = _vel(c, r)
    n = _n(0.7 * _ring(c))
    return _out(c, r, _tamb(c, r, [(0.002, vel)], n, head=-7.0), 'room', -0.1)


@design('tambourine/shake', kind='noise',
        says=('a tambourine being shaken', 'jingles of a tambourine rattling', 'a shimmering shake of metal jingles'),
        uses=('tambourine', 'groove', 'rhythm', 'percussion', 'shake', 'celebration'),
        tags=('drums', 'hand percussion'), domains=DOM, dur=1.0, axes=AXES, allows=('tail',))
def tambourine_shake(c, r):
    """the tambourine shaken for dur s, about 7 strokes a second: every stroke throws the jingles into a new run
    of clashes before the last one has died"""
    T = c.seconds(1.0)
    vel = _vel(c, r)
    n = _n(T + 0.5 * _ring(c))
    rate = 7.0 * r.uniform(0.94, 1.06)
    t = np.arange(0.01, T, 1.0 / rate)
    t = t + r.normal(0, 0.004, len(t))
    e = vel * np.where(np.arange(len(t)) % 2 == 0, 1.0, 0.7) * np.exp(0.15 * r.standard_normal(len(t)))
    return _out(c, r, _tamb(c, r, list(zip(np.maximum(t, 0.0), e)), n, shake=1.0), 'room', -0.1)


# ------------------------------------------------------------------------------------------------------ hand drums
def _hand(c, r, name, note, stroke, radius, t60, dur):
    """a hand drum (one head, an open shell): open = the fingers strike near the edge and bounce off (the head's
    note rings, (0,1) and its neighbours, over the shell's air resonance an octave down); slap = cupped fingers hit
    the edge and stay (a short contact, the low modes muted, a sharp pop of the upper modes and the skin's slap);
    bass = the palm in the centre (the (0,n) thud and the shell's air)"""
    built = _built(name)
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    f = c.hz(note) * kf
    n = _n(dur * kl * g)
    if stroke == 'open':
        kw = dict(beater='fingers', strike=0.82, t60=t60 * kl * g, damp=0.6 / g, click=0.3)
    elif stroke == 'slap':
        kw = dict(beater='fingers', strike=0.9, t60=0.09 * kl * g, damp=0.25 / g, click=1.0)
    else:
        kw = dict(beater='hand', strike=0.05, t60=0.5 * t60 * kl * g, damp=1.0 / g, click=0.2)
    m = D.membrane(f, n, r, built, vel=vel, hard=c.hard + (1.0 if stroke == 'slap' else 0.0), load=0.25, seal=0.0,
                   count=22, fall=0.05, radius=radius / kf, mono=1.2, spread=0.3,
                   shell=((0.5 * f, 0.22 * kl * g, 0.2 if stroke != 'slap' else 0.05),), **kw)
    layers = [(m['y'], 0.0)]
    if stroke == 'slap':
        k = _n(0.02)
        tk = np.arange(k) / SR
        skin = dsp.butter(r.standard_normal((k, 2)), 'band', (700.0, 6000.0), 2) * \
            ((tk / 0.0005).clip(0, 1) * np.exp(-tk / 0.004))[:, None]
        pop = np.sin(2 * np.pi * 3.1 * f * tk) * np.exp(-tk / 0.012)
        layers += [(skin, -3.0 + 3 * c.hard), (pop, -8.0)]
    return _mix(layers, n)


@design('conga/open', kind='hit',
        says=('an open conga tone', 'a hand striking a conga drum, a round ringing note',
              'a warm pitched hand drum hit'),
        uses=('conga', 'latin', 'groove', 'percussion', 'beat', 'rhythm'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=0.6, pitch='E3', axes=AXES)
def conga_open(c, r):
    """an open tone on an 11-inch conga"""
    return _out(c, r, _hand(c, r, 'conga/open', 'E3', 'open', 0.14, 0.42, 0.8), 'room', 0.0)


@design('conga/slap', kind='hit',
        says=('a conga slap, a sharp high crack of a hand on a drum', 'a slapped conga, short and popping',
              'a hand slapping a drum head'),
        uses=('conga', 'latin', 'accent', 'percussion', 'beat', 'slap'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=0.3, pitch='E3', axes=AXES)
def conga_slap(c, r):
    """a slap on the conga"""
    return _out(c, r, _hand(c, r, 'conga/slap', 'E3', 'slap', 0.14, 0.42, 0.4), 'room', 0.0)


@design('bongo/high', kind='hit',
        says=('a high bongo hit', 'a small hand drum with a bright pitched pop', 'fingers striking a bongo'),
        uses=('bongo', 'latin', 'groove', 'percussion', 'beat', 'rhythm', 'tap'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=0.35, pitch='G4', axes=AXES)
def bongo_high(c, r):
    """an open tone on the small (7-inch) bongo: high, short, bright"""
    return _out(c, r, _hand(c, r, 'bongo/high', 'G4', 'open', 0.09, 0.22, 0.45), 'room', 0.0)


def _cajon(c, r, stroke):
    """a cajon: a plywood box, the player sits on it and strikes the front plate (the tapa). The tapa rings in
    the modes of a thin plate clamped on its edges (1 : 2.09 : 3.43 : 3.91 : 5.0 : 6.21), the box's air
    through the sound hole is a Helmholtz resonance near 80-90 Hz (the bass), and wires behind the tapa buzz
    when it moves (rattle). bass = the palm in the middle: the hole booms, little wire; slap = fingers at the top
    edge: the upper plate modes, the crack of the plate's corner and much more wire"""
    built = _built('cajon')
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(0.7 * kl * g)
    ratio = np.array([1.0, 2.09, 3.43, 3.91, 5.0, 6.21, 7.5, 9.1])
    f0 = 118.0 * kf
    bass = stroke == 'bass'
    gain = np.array([1.0, 0.5, 0.3, 0.3, 0.2, 0.15, 0.1, 0.08]) if bass else \
        np.array([0.3, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4])
    plate = D.body(f0 * ratio, 0.16 * kl * g * ratio ** -0.5, gain, n, r, built, vel=vel,
                   beater='hand' if bass else 'fingers', hard=c.hard + (0.0 if bass else 1.0),
                   click=0.3 if bass else 1.5)
    t = np.arange(n) / SR
    hz = 84.0 * kf
    boom = np.sin(2 * np.pi * hz * t) * np.exp(-modal.T60 * t / (0.28 * kl * g)) * (1 - np.exp(-t / 0.003))
    motion = dsp.butter(plate.mean(1), 'low', 400.0, 2)
    motion = motion / (np.abs(motion).max() + 1e-12) * vel
    w = D.rattle(motion, r, 0.25 if bass else 0.08, 0.02 * g ** 0.5,
                 ((2200.0, 1.3, 1.0), (4200.0, 1.6, 0.8), (7500.0, 2.0, 0.4)), body=0.2)
    layers = [(plate, 0.0), (boom, 0.0 if bass else -14.0), (w, -14.0 if bass else -4.0)]
    if not bass:
        k = _n(0.015)
        tk = np.arange(k) / SR
        crack = dsp.butter(r.standard_normal((k, 2)), 'band', (900.0, 7000.0), 2) * \
            ((tk / 0.0004).clip(0, 1) * np.exp(-tk / 0.003))[:, None]
        layers.append((crack, -4.0 + 3.0 * c.hard))
    return _mix(layers, n)



@design('cajon/bass', kind='thud',
        says=('a cajon bass tone, a deep thump on a wooden box drum', 'a palm striking the middle of a cajon',
              'a low woody box drum thud'),
        uses=('cajon', 'acoustic', 'groove', 'beat', 'percussion', 'kick drum'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=0.5, axes=AXES)
def cajon_bass(c, r):
    """the palm in the middle of the tapa"""
    return _out(c, r, _cajon(c, r, 'bass'), 'room', 0.0)


@design('cajon/slap', kind='hit',
        says=('a cajon slap, a sharp snare-like crack on a wooden box', 'fingers slapping the top edge of a cajon',
              'a woody crack with a buzz of wires'),
        uses=('cajon', 'acoustic', 'groove', 'beat', 'percussion', 'snare'), tags=('drums', 'hand percussion'),
        domains=DOM, dur=0.4, axes=AXES)
def cajon_slap(c, r):
    """the fingers at the top edge of the tapa"""
    return _out(c, r, _cajon(c, r, 'slap'), 'room', 0.0)


@design('clave/click', kind='click',
        says=('a clave click, two hard wooden sticks struck together', 'a bright hollow wooden click',
              'a high woody tock of claves'),
        uses=('clave', 'latin', 'click', 'tick', 'rhythm', 'metronome', 'percussion'),
        tags=('drums', 'hand percussion'), domains=DOM, dur=0.25, pitch='A6', axes=AXES)
def clave_click(c, r):
    """two rosewood rods: the one in the cupped hand rests on the fingertips at the nodes of its fundamental and
    rings (a round rod's bending modes, 1 : 2.6 : 4.8 with shear), the hand's cavity under it lifts the note; the
    striking rod rings too, a little higher and damped by the grip; rosewood on rosewood is a very short contact"""
    built = _built('clave')
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    f = c.hz('A6') * kf
    n = _n(0.35 * kl * g)
    s = modal.tube_modes(f, 1, gyration=0.0055, c=4400.0, strike=0.5, count=4, shear=3.9)
    fr = f * s['ratio']
    held = D.body(fr, 0.22 * kl * g * s['ratio'] ** -1.2, [1.0, 0.25, 0.12, 0.05], n, r, built, vel=vel,
                  beater='clave', hard=c.hard, click=1.0)
    other = D.body(fr * 1.09, 0.06 * kl * g * s['ratio'] ** -1.2, [1.0, 0.3, 0.1, 0.05], n, r, built, vel=vel,
                   beater='clave', hard=c.hard, click=0.0)
    return _out(c, r, _mix([(held, 0.0), (other, -7.0)], n), 'room', -0.2)


@design('sticks/count', kind='click',
        says=('two drumsticks clicked together', 'a dry wooden stick click', 'a drummer clicking sticks to count in'),
        uses=('count', 'countdown', 'count-in', 'click', 'tick', 'metronome', 'start'), tags=('drums', 'kit'),
        domains=DOM, dur=0.15, axes=AXES)
def sticks_count(c, r):
    """two hickory sticks crossed and struck near their shoulders: each a round rod held at its fulcrum (its
    modes from about 400 Hz, quickly damped by the grip), wood on wood, so the click is the upper modes and the
    contact's tick, plus the dry crack of the grain at the contact (a millisecond of noise): dry, short and
    woody, not a ringing clave"""
    built = _built('sticks')
    vel = _vel(c, r)
    kf, kl = _size(c)
    g = _ring(c)
    n = _n(0.15 * kl * g)
    out = []
    for f0, strike in ((400.0, 0.18), (430.0, 0.22)):
        s = modal.tube_modes(f0 * kf, 1, gyration=0.0036, c=4300.0, strike=strike * r.uniform(0.9, 1.1), count=7,
                             shear=3.9, rest=0.3)
        fr = f0 * kf * s['ratio']
        out.append(D.body(fr, 0.04 * kl * g * s['ratio'] ** -0.5,
                          0.3 * np.abs(s['gain']) * (0.15 + s['held']) ** -0.5, n, r, built, vel=vel, beater='wood',
                          hard=c.hard + 0.6, click=4.0))
    k = _n(0.008)
    tk = np.arange(k) / SR
    crack = dsp.butter(r.standard_normal((k, 2)), 'band', (1200.0, 7000.0), 2) * \
        ((tk / 0.0003).clip(0, 1) * np.exp(-tk / 0.0012))[:, None]
    return _out(c, r, _mix([(out[0], 0.0), (out[1], -2.0), (crack, -6.0 + 3.0 * c.hard)], n), 'room', -0.3)


@design('drum/roll', kind='hit',
        says=('a snare drum roll swelling into a hit', 'a fast press roll on a snare drum growing louder',
              'a drum roll building up to a crash of the snare'),
        uses=('drum roll', 'build', 'reveal', 'announcement', 'tension', 'suspense', 'drum', 'transition'),
        tags=('drums', 'kit'), domains=DOM, dur=2.0, pitch='B3', axes=AXES,
        allows=('pre_swell', 'stacked', 'tail'))
def drum_roll(c, r):
    """a snare roll for dur s into an accent: double strokes from both hands (about 26 strokes a second, the two
    hands never quite even and the second of each pair weaker), louder and louder, then one full hit. The head
    and its motion are linear at roll levels (each stroke is one stroke's response, two sticks at two strike
    points); the wires are not: they rattle on the summed motion of the head, so they buzz continuously once the
    roll is loud enough"""
    built = _built('drum/roll')
    T = c.seconds(2.0)
    vel = _vel(c, r, 0.8)
    kf, kl = _size(c)
    g = _ring(c)
    f = c.hz('B3') * kf
    n = _n(T + 0.6 * kl * g)
    resp = [D.membrane(f, _n(0.5), r, built, vel=0.5, beater='stick', hard=c.hard, strike=s, t60=0.3 * kl * g,
                       damp=0.7 / g, load=0.5, seal=0.6, count=26, fall=0.0, radius=0.18 / kf, click=1.0,
                       glide=False) for s in (0.3, 0.42)]
    rate = 26.0 * r.uniform(0.95, 1.05)
    t = np.arange(0.0, T - 0.03, 1.0 / rate)
    k = np.arange(len(t))
    t = t + 0.008 * (k % 4 >= 2) + r.normal(0, 0.0025, len(t))
    t = np.clip(t, 0.0, T)
    u = t / T
    a = (0.06 + 0.5 * u ** 1.6) * np.where(k % 2 == 0, 1.0, 0.7) * np.exp(0.12 * r.standard_normal(len(t)))
    head = sum(D.train(n, m['y'], t[i::2], a[i::2]) for i, m in enumerate(resp))
    mot = sum(D.train(n, m['motion'], t[i::2], a[i::2])[:, 0] for i, m in enumerate(resp))
    hit = D.membrane(f, n - _n(T), r, built, vel=vel, beater='stick', hard=c.hard, strike=0.35,
                     t60=0.35 * kl * g, damp=0.6 / g, load=0.5, seal=0.6, count=26, fall=0.05, radius=0.18 / kf)
    scale = np.abs(resp[0]['y']).max() / 0.5
    head = head / scale
    hy = np.zeros((n, 2))
    hy[_n(T):] = hit['y'] / scale * (vel / 0.5) * 0.6
    hm = np.zeros(n)
    hm[_n(T):] = hit['motion']
    mot = mot / (np.abs(resp[0]['motion']).max() + 1e-12) * 0.5 + hm
    w = _wires(r, mot, 1.45 * f, vel, 0.05, 0.025 * g ** 0.5, c.bright)
    y = head + hy
    y = y + w * (np.abs(y).max() / (np.abs(w).max() + 1e-12)) * 10 ** (-4.0 / 20)
    return _out(c, r, dsp.fade(y, 0.003, 0.1), 'studio', 0.0)
