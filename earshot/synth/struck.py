"""Struck and plucked instruments, each built from its physics (earshot/synth/modal.py): mallet percussion, bells,
chimes, tines, bowls, glass, and small metal and wood objects.

    from earshot.synth import struck
    y = struck.note('glockenspiel', 'C6')                       # stereo float64 (n, 2), rings out naturally
    y = struck.note('marimba', 'C3', vel=0.4, dur=0.5)          # soft, damped by the player after 0.5 s
    y = struck.note('vibraphone', 'F4', motor=5.5, ring=0.3)    # motor on, a little longer
    y = struck.note('church_bell', 'G4', dur=3.0, r=dsp.rng(7), vary=0.5)
    y = struck.note('coin', bounce=0.6)                         # pitch None = INSTRUMENTS[name]['default_pitch']
    y = struck.chord('kalimba', ['C5', 'E5', 'G5'], spread_s=0.03)

Every instrument is a geometry (where its modes lie and how the strike position weights them), a material (how
fast each mode dies), a mallet (mass, stiffness: how long the contact lasts at a given speed) and what it sits on
(a resonator tube, a box, a cord). `vel` is the speed of the mallet: it sets the level and, through the contact
time, the brightness. Controls, all in -1 .. 1 with 0 = the instrument as built:
    bright    harder mallet and more radiation of the high modes
    hard      mallet hardness only (the stiffness changes 16-fold per unit)
    ring      damping: -1 dry and short (a quarter of the ring time), +1 long (four times)
    size      resonator or body size: more boost and bloom from a tube or box; a larger, darker body otherwise
    position  where it is struck: 0 = the usual place; INSTRUMENTS[name]['position'] says what -1 and +1 are
    width     stereo width of the close microphone pair: -1 = mono
    vary      0 .. 1: take-to-take variation of strike position, hardness and speed (drawn from r); pitch never moves
Some instruments have options of their own (INSTRUMENTS[name]['options']): vibraphone motor, kalimba buzz, coin
bounce, glass_clink interval. The level is not normalised: vel = 1 peaks near 0.9 and softer notes are lower.
"""
import functools
import zlib

import numpy as np

from ..forge import dsp
from ..forge.knobs import hz
from . import modal

SR = dsp.SR
CONTROLS = ('bright', 'hard', 'ring', 'size', 'position', 'width', 'vary')
INSTRUMENTS = {}
_SPEC = {}


def _over(f0, *points):
    """a property that changes along the range of an instrument: interpolated over log pitch through (note, value)"""
    notes, values = zip(*points)
    return float(np.interp(np.log2(f0), np.log2([hz(p) for p in notes]), values))


def _add(name, low, high, default, says, ring_s, position, options=None, **spec):
    """Enter an instrument: its public entry (playable range, plain descriptions of the sound, default pitch, the
    seconds a medium stroke at that pitch takes to fall 60 dB, what the position control does, its own options)
    and what it is made of:
      modes       function (pitch Hz, strike coordinate, vel) -> a mode set of earshot.synth.modal
      strike      the strike coordinate at position 0, -1 and +1 (along a bar from its end, up a bell from its
                  soundbow, out from the centre of a disc ...)
      material, loss, support, cord, radiation
                  the damping of a mode at f on a body tuned to f0:
                  loss * modal.decay_rate(f, material, radiation)
                  + (support[0] + cord * held of the mode) * (f0 / default pitch)^support[1]
      mallet      a name in _MALLET; speed: m/s of a full stroke (vel 1); track: the contact time goes as
                  pitch^-track (small bars are lighter and are played with harder mallets).
                  Or release: the seconds a plucked tine takes to slip off the nail or pin
      knee, order radiation efficiency (modal.radiation); knee_x: the knee as a multiple of the pitch
      tick        click, grit and ring time of modal.contact_noise (and, as a fourth, the Hz its grit starts at)
      resonator   function pitch -> the tube or box under the bar (q, boost_db, load, detune of modal.resonator)
      body, direct  soundboard modes that colour every partial (modal.body_response)
      knock       (level, modes): what the stroke makes of the thing the instrument sits on or is played through
      damper      seconds the note takes to die once the player damps it
      extent      size of the source in the stereo picture, 0 .. 1"""
    INSTRUMENTS[name] = {'range': (low, high), 'says': says, 'default_pitch': default, 'ring_s': ring_s,
                         'position': position, 'options': options or {}}
    _SPEC[name] = spec


# Mallets: (mass kg, stiffness N/m^p, exponent p of force = stiffness * depth^p). The contact time follows from
# these and the speed (modal.contact_time). Stiffness of the hard ones is the Hertz value 4/3 E* sqrt(radius) for
# the pair of materials (a 25 mm plastic ball on steel 6e8, on rosewood across the grain 2e8, glass on glass 7e9,
# an iron clapper on bell bronze 4e10); the wound and felted ones are set to the contact times reported for a medium
# stroke (cord on aluminium 0.6 ms, yarn on rosewood 1 - 3 ms, felt hammers about 0.7 ms).
# [cal] Two mallets are set from recordings (anechoic single notes of the University of
# Iowa MIS and far-miked ones of VSCO-2 CE). Yarn: wound over a core, it stiffens like felt (p 3.5), 1 ms on the bar at
# a loud stroke, 1.5 ms at half the speed: the second mode of a C4 marimba bar comes 17 dB closer to its fundamental
# from mf to ff. Hard rubber: 0.2 ms, so the third mode of a xylophone bar (5 - 6 kHz at C6) is still struck: it lies
# 17 - 21 dB under the fundamental in notes of both sources.
_MALLET = {
    'plastic': (0.012, 6.0e8, 1.5), 'brass': (0.012, 2.0e9, 1.5), 'hard rubber': (0.025, 9.0e8, 1.5),
    'cord': (0.030, 1.9e9, 2.0), 'yarn': (0.032, 1.3e13, 3.5), 'felt': (0.006, 1.2e10, 2.5),
    'chime hammer': (0.25, 3.0e9, 1.5), 'wood': (0.030, 1.0e9, 1.5), 'clapper': (0.050, 2.2e8, 1.5),
    'iron clapper': (15.0, 4.0e10, 1.5), 'suede': (0.060, 1.0e9, 2.0), 'nail': (0.002, 5.0e7, 1.5),
    'finger': (0.020, 1.5e8, 2.0), 'pan stick': (0.015, 8.0e7, 1.5), 'drumstick': (0.045, 4.0e8, 1.5),
    'steel': (0.020, 5.0e9, 1.5), 'stone': (0.008, 5.0e9, 1.5), 'glass': (0.010, 7.0e9, 1.5),
}
_BAR = '-1 = the dead centre of the bar (no second mode), +1 = on the node of the fundamental (thin, all attack)'
_BELL = '-1 = on the lip, +1 = up on the waist (more prime and tierce, less of the bright rim partials)'
_RIM = '-1 = on the very rim, +1 = well down the wall (fewer high modes)'
_FIELD = '-1 = the centre of the note field (mostly the note), +1 = its edge (more octave and twelfth)'


# ------------------------------------------------------------------------------------------------ mallet keyboards
# glockenspiel: steel bars 9.5 x 32 mm, not undercut, on felt rails; hard plastic mallets.
#   partials (Rossing: 1 : 2.71 : 5.15 measured mid-range; thin bar 1 : 2.756 : 5.404; the short bars read lower)
#            G5 784 Hz        C7 2093 Hz       C8 4186 Hz
#   mode 2   2.73             2.69             2.62
#   mode 3   5.29             5.11             4.86
#   T60 f1   8 s              4 s              2 s        (steel bars ring 5 - 15 s low, 1 - 3 s at the top)
#   torsion  4.1, 8.2 x f0    2.5, 5.0         1.8, 3.6   (weak, -25 dB)
# [cal] Five notes of one glockenspiel (VSCO-2 CE, C6 to C8): a bar this small hardly radiates its fundamental, so in
# the first 100 ms modes 2 and 3 stand as high as it (-2 and 0 dB at C6, +5 and 0 at G6; `order`, `strike`) and the
# first 20 ms have their centroid near 6 kHz; mode 2 falls 50 dB/s at C6, 90 at G6, 130 at C7 (`cord`); between the
# modes the stroke leaves a hiss at 4 - 8 kHz, 20 - 30 dB under the note, that is still there 60 ms later (`tick`).
# Its bars measure 2.94 : 5.54 at C6 falling to 2.68 at C8, above the 2.71 : 5.15 kept here from Rossing.
def _glockenspiel(f0, x, vel):
    return modal.bar_modes(f0, x, 0.0095, 0.032, 5100.0, 8)


_add('glockenspiel', 'C5', 'C8', 'C6',
     ['a glockenspiel note', 'a small steel bar struck with a hard mallet, bright and ringing',
      'orchestra bells playing one high clear note'], 4.6, _BAR,
     modes=_glockenspiel, strike=(0.42, 0.5, 0.224), material='steel', support=(0.66, 0.33), loss=0.64, cord=25.0,
     mallet='plastic', speed=4.0, track=0.15, knee=3400.0, order=1.5, tick=(0.25, 0.1, 0.04, 3000.0), damper=0.18,
     extent=0.5)


# celesta: steel plates like glockenspiel bars (6 x 30 mm), each over a tuned wooden box, struck by felt hammers
# through a piano action (whose thump is part of the sound).
#            C4 262 Hz        C6 1047 Hz       C8 4186 Hz
#   mode 2   2.75             2.73             2.66       (uniform bar; the felt leaves little of it)
#   T60 f1   4 s              2.5 s            1 s        (shorter than a glockenspiel: the box drains the plate)
_ACTION = ((140.0, 3.0, 1.0), (330.0, 4.0, 0.6), (700.0, 4.0, 0.3))      # key bed and hammer rail: Hz, q, gain


def _celesta(f0, x, vel):
    return modal.bar_modes(f0, x, 0.006, 0.030, 5100.0, 6)


def _celesta_box(f0):
    return {'q': 14.0, 'boost_db': _over(f0, ('C4', 11.0), ('C6', 7.0), ('C8', 2.0)),
            'load': _over(f0, ('C4', 1.2), ('C8', 0.4)), 'detune': 0.004}


_add('celesta', 'C4', 'C8', 'C5',
     ['a celesta note', 'a soft bell-like keyboard note, a steel plate struck by a felt hammer',
      'a music-box-like chime played on a celesta'], 3.0, _BAR,
     modes=_celesta, strike=(0.42, 0.5, 0.224), material='steel', support=(0.82, 0.29), loss=0.91, cord=25.0,
     order=0.5, mallet='felt', speed=3.0, track=0.25, knee=3400.0, tick=(0.05, 0.03, 0.004), resonator=_celesta_box,
     knock=(0.06, _ACTION), damper=0.12, extent=0.4)


# vibraphone: aluminium bars 12.7 mm thick, deep arch, tube resonators with motor-driven discs; cord mallets.
#   partials (measured bars: 1 : 3.98 : 10.67 : 18.0 : 23.7 : 33.6 and 1 : 4.00 : 9.47 : 15.6 : 20.9 : 29.4)
#            F3 175 Hz        C5 523 Hz        F6 1397 Hz
#   mode 2   3.99             3.99             3.90       (tuned two octaves up to about C6)
#   mode 3   10.2             9.6              8.9        (tuned near 10 only on the low bars)
#   T60 f1   22 s             16 s             11 s       (pedal down, motor off; 10 - 40 s reported low)
# [cal] Five anechoic notes of one vibraphone (University of Iowa MIS, C3 to C6, mf and ff at C4): the fundamental
# falls 2.5 - 2.7 dB/s at C3 and C4 and 4.2 - 4.3 dB/s at C5 and C6, so the top rings almost as long as the bottom
# (`support`, `loss`); its envelope peaks 63 ms after the stroke at C4, 41 ms at C5, 15 ms at C6: some 17 - 22
# periods (the tube's q); the bars are struck in the middle, on the node of mode 2, which lies 19 - 36 dB under the
# fundamental while mode 3 reaches -4 dB at ff (`strike`).
def _vibraphone(f0, x, vel):
    second = _over(f0, ('F3', 3.99), ('C6', 3.99), ('F6', 3.90))
    third = _over(f0, ('F3', 10.2), ('C4', 9.9), ('C5', 9.6), ('C6', 9.2), ('F6', 8.9))
    return modal.tuned_bar_modes(second, third, x, 7, twist=_over(f0, ('F3', 6.5), ('F6', 2.6)))


def _vibraphone_tube(f0):
    return {'q': _over(f0, ('F3', 18.0), ('C4', 10.0), ('F6', 9.0)),
            'boost_db': _over(f0, ('F3', 10.0), ('C5', 7.0), ('F6', 3.0)),
            'load': _over(f0, ('F3', 0.8), ('F6', 0.5)), 'detune': 0.003}


_add('vibraphone', 'F3', 'F6', 'F4',
     ['a vibraphone note', 'an aluminium bar struck with a soft mallet, a long mellow ringing tone',
      'a vibraphone with a slow shimmering sustain'], 19.0, _BAR,
     options={'motor': 'tremolo of the resonator discs in Hz, 3 - 8 is usual; 0 = motor off (default)'},
     modes=_vibraphone, strike=(0.485, 0.5, 0.224), material='aluminium', support=(0.14, 0.0), loss=0.375, cord=6.0,
     order=0.7, mallet='cord', speed=3.5, track=0.25, knee=2500.0, tick=(0.1, 0.03, 0.003), resonator=_vibraphone_tube,
     damper=0.22, extent=0.4)


# marimba: rosewood bars with a deep arch over tuned tubes; yarn mallets, softer in the bass.
#   partials (1 : 4 : 10 tuning; measured third 9.2 - 10; one measured bar 1 : 3.93 : 9.54 : 16.7 : 24.6 : 31.1)
#            C2 65 Hz         C4 262 Hz        C6 1047 Hz
#   mode 2   4.00             4.00             4.00       (falls to 3.6 by C7: the top bars are too short to tune)
#   mode 3   10.0             9.7              8.6
#   T60 f1   3.5 s            1.5 s            0.42 s     (with resonator; about twice that without, after Bork)
# [cal] Four anechoic notes with yarn mallets (University of Iowa MIS, C3 to C5) agree with these ring times (C4:
# 1.45 - 1.57 s; C5: 1.0 s) and with the tube's bloom (the envelope peaks after 160, 75, 35 ms at C3, C4, C5). The
# bars are struck near the middle, between the antinode of mode 3 and the node of mode 2: at ff mode 2 lies 8, 13 and
# 43 dB under the fundamental at C3, C4, C5 and mode 3 of the C3 bar as high as its fundamental (`strike`); one pair
# of mallets plays the whole instrument (`track`). Seven far-miked notes of another marimba (VSCO-2 CE) ring 2.5
# times as long and are brighter still: not followed.
def _marimba(f0, x, vel):
    second = _over(f0, ('C2', 4.0), ('C6', 4.0), ('C7', 3.6))
    third = _over(f0, ('C2', 10.0), ('C3', 9.9), ('C4', 9.7), ('C5', 9.3), ('C6', 8.6), ('C7', 7.6))
    return modal.tuned_bar_modes(second, third, x, 7, twist=_over(f0, ('C2', 7.0), ('C7', 2.4)))


def _marimba_tube(f0):
    return {'q': 30.0, 'boost_db': _over(f0, ('C2', 14.0), ('C4', 10.0), ('C6', 5.0), ('C7', 2.0)),
            'load': _over(f0, ('C2', 1.0), ('C5', 1.0), ('C7', 0.3)), 'detune': 0.003}


_add('marimba', 'C2', 'C7', 'C4',
     ['a marimba note', 'a wooden bar struck with a soft yarn mallet, warm and round',
      'a low wooden mallet percussion note with a hollow resonance'], 1.3, _BAR,
     modes=_marimba, strike=(0.46, 0.5, 0.224), material='rosewood', support=(0.15, -0.8), loss=0.6, cord=6.0,
     mallet='yarn', speed=3.0, track=0.0, knee=1800.0, tick=(0.1, 0.08, 0.005), resonator=_marimba_tube,
     damper=0.07, extent=0.4)


# xylophone: thicker, shorter rosewood bars with a shallow arch, tuned in twelfths; hard mallets. Its closed tubes
# resonate at odd multiples, so they also catch the second partial at 3 x f0.
#            F4 349 Hz        C6 1047 Hz       C8 4186 Hz
#   mode 2   3.00             3.00             2.80       (tuned a twelfth up to about C7)
#   mode 3   6.1              5.9              5.4        (not tuned; measured 6 - 7 low, the uniform bar at the top)
#   T60 f1   1.3 s            0.45 s           0.10 s     (0.5 - 1.5 s reported; partial 3 dies 2 - 4 times faster)
# [cal] Two anechoic notes (University of Iowa MIS, C5 and C7, hard rubber, ff) and six far-miked ones (VSCO-2 CE,
# G5 to C8): the fundamental of C5 falls 62 dB/s, as here; struck near the middle, mode 2 lies 21 dB (Iowa C5), 5 to
# 65 dB (VSCO) under the fundamental and mode 3 17 - 21 dB under it up to C6 (`strike`, the hard rubber mallet).
def _xylophone(f0, x, vel):
    second = _over(f0, ('F4', 3.0), ('C7', 3.0), ('C8', 2.8))
    third = _over(f0, ('F4', 6.1), ('C6', 5.9), ('C8', 5.4))
    return modal.tuned_bar_modes(second, third, x, 7, twist=_over(f0, ('F4', 4.2), ('C8', 1.9)))


def _xylophone_tube(f0):
    return {'q': 25.0, 'boost_db': _over(f0, ('F4', 7.0), ('C6', 4.0), ('C8', 1.0)),
            'load': _over(f0, ('F4', 0.7), ('C8', 0.3)), 'detune': 0.003, 'twelfth_db': 4.0}


_add('xylophone', 'F4', 'C8', 'C6',
     ['a xylophone note', 'a hard wooden bar struck with a hard mallet, short, dry and bright',
      'a sharp wooden mallet percussion note'], 0.4, _BAR,
     modes=_xylophone, strike=(0.46, 0.5, 0.224), material='rosewood', support=(0.2, 0.0), loss=0.6, cord=8.0,
     mallet='hard rubber', speed=4.0, track=0.2, knee=2200.0, tick=(0.3, 0.14, 0.003), resonator=_xylophone_tube,
     damper=0.06, extent=0.4)


# crotales: thick bronze discs bolted through the centre, brass mallets. The bolt kills the modes that move the
# centre, so the (m,0) series of a free disc is left: (2,0) 1 : (3,0) 2.33 : (4,0) 4.11 : (5,0) 6.30 (plate theory;
# the measured values of Rossing were not at hand).
#            C6 1047 Hz       C7 2093 Hz       C8 4186 Hz
#   T60 f1   11.5 s           6.3 s            3 s
def _crotales(f0, x, vel):
    return modal.plate_modes(x, centre_held=True, split=0.0008)


_add('crotales', 'C6', 'C8', 'C7',
     ['a crotale note', 'a small bronze disc struck with a brass mallet, a very long pure high ring',
      'an antique cymbal ringing'], 5.0, '-1 = on the very edge, +1 = halfway to the centre (fewer high modes)',
     modes=_crotales, strike=(0.95, 1.0, 0.55), material='bronze', support=(0.15, 0.0), loss=0.7,
     order=0.5, mallet='brass', speed=3.5, track=0.1, knee=4500.0, tick=(0.2, 0.03, 0.002), damper=0.25, extent=0.6)


# ------------------------------------------------------------------------------------------------------- tines
# kalimba: spring steel tines over a bridge on a small wooden box, plucked with the thumb nail.
#   partials (ideal clamp 1 : 6.27 : 17.55; tines on a bridge with a back length measure 5 - 7.5 for mode 2)
#            C4 262 Hz        C5 523 Hz        C6 1047 Hz
#   mode 2   5.6              5.6              5.6
#   T60 f1   3.1 s            1.5 s            0.7 s      (1 - 4 s reported; the bridge takes the energy, so the
#   T60 m2   0.35 s           0.17 s           0.07 s      loss factor of the mounted tine is about 2.5e-3)
# The tine presses on the bridge harder on the way down than on the way up, so the board is also driven at twice
# and three times the pitch, the more the harder the pluck: the weak harmonics of a kalimba tone.
_BOX = ((190.0, 6.0, 0.9), (410.0, 9.0, 1.0), (640.0, 10.0, 0.8), (980.0, 10.0, 0.7), (1500.0, 9.0, 0.6),
        (2300.0, 8.0, 0.5), (3600.0, 7.0, 0.4))            # air mode of the box, then soundboard modes: Hz, q, gain


def _kalimba(f0, x, vel):
    return modal.join(modal.tine_modes(5.6, x, 4), modal.modes([2.0, 3.0], [0.07 * (0.4 + vel), 0.03 * (0.4 + vel)]))


_add('kalimba', 'G3', 'E6', 'C5',
     ['a kalimba note', 'a thumb piano, a metal tine plucked on a small wooden box',
      'a soft plucked metallic note with a woody knock'], 1.4,
     '-1 = plucked a third of the way in (softer overtone), +1 = at the very tip',
     options={'buzz': '0 .. 1: a loose ring on the tine rattling while the note is loud (default 0)'},
     modes=_kalimba, strike=(0.95, 0.6, 1.0), material='steel', support=(0.0, 0.0), loss=18.0,
     order=0.0, release=0.0003, speed=1.0, knee=1000.0, tick=(0.15, 0.04, 0.003), body=_BOX, direct=0.15,
     knock=(0.25, _BOX), damper=0.05, extent=0.3)


# music box: a steel comb screwed to a metal bedplate in a small wooden case, teeth plucked by pins. The bass
# teeth carry lead under their tips, which lowers the fundamental more than the upper modes.
#            C5 523 Hz        C6 1047 Hz       C8 4186 Hz
#   mode 2   8.9              6.27             6.27       (6.27 unloaded; higher for the leaded teeth under C6)
#   T60 f1   3 s              2 s              0.6 s
_CASE = ((620.0, 10.0, 0.5), (1150.0, 12.0, 0.9), (1900.0, 12.0, 1.0), (2900.0, 10.0, 0.9), (4300.0, 9.0, 0.7),
         (6200.0, 8.0, 0.5))


def _music_box(f0, x, vel):
    return modal.tine_modes(6.27 * max(1.0, (1047.0 / f0) ** 0.5), x, 3)


_add('music_box', 'C4', 'C8', 'C6',
     ['a music box note', 'a tiny steel comb tooth plucked by a pin, a delicate tinkling chime',
      'a wind-up music box playing one note'], 2.0,
     '-1 = the pin lifts the tooth further in, +1 = at the very tip',
     modes=_music_box, strike=(0.97, 0.7, 1.0), material='steel', support=(2.24, 0.35), loss=2.09,
     order=0.5, release=0.00015, speed=1.0, knee=1000.0, tick=(0.3, 0.03, 0.002), body=_CASE, direct=0.25,
     knock=(0.06, _CASE), damper=0.04, extent=0.25)


# ------------------------------------------------------------------------------------------------ tubes and rods
# tubular bell: a brass tube 38 mm wide, 1.0 - 1.6 m long, struck on its top edge with a rawhide or plastic hammer.
# Pitch names the strike tone, the missing fundamental of modes 4, 5, 6.
#   modes 4 : 5 : 6 relative to the strike tone (thin beam 2 : 2.99 : 4.17; measured chimes about 2 : 2.94 : 4.0)
#            C4 262 Hz        C5 523 Hz        G5 784 Hz
#   mode 5   2.94             2.91             2.87
#   mode 6   4.02             3.92             3.82
#   T60 m4   17 s             11 s             8 s        (10 - 30 s reported)
def _tubular_bell(f0, x, vel):
    m = modal.tube_modes(2.0 * f0, 4, 0.0129, 3430.0, x, 12, 0.0012, rest=0.0)
    m['ratio'] = m['ratio'] * 2.0
    return m


_add('tubular_bell', 'C4', 'G5', 'C5',
     ['a tubular bell', 'an orchestral chime, a long brass tube struck with a hammer',
      'a church-like chime note from tubular bells'], 13.0,
     '-1 = on the very top edge, +1 = a fifth of the way down the tube',
     modes=_tubular_bell, strike=(0.03, 0.0, 0.2), material='brass', support=(0.2, 0.0), loss=0.6,
     mallet='chime hammer', speed=2.5, track=0.2, knee=2900.0, order=1.5, tick=(0.3, 0.07, 0.006), damper=0.3,
     extent=0.5)


# wind chime: one aluminium tube (16 mm) hung at a node and struck near its middle by a wooden clapper.
#            C5 523 Hz        C6 1047 Hz       C7 2093 Hz
#   mode 2   2.71             2.67             2.60       (free tube: 2.756 thin, lower as the tube gets short)
#   mode 3   5.19             5.00             4.75       (thin 5.404)
#   T60 f1   9 s              6 s              3.5 s
def _wind_chime(f0, x, vel):
    return modal.tube_modes(f0, 1, 0.00525, 5050.0, x, 8, 0.002)


_add('wind_chime', 'C5', 'C8', 'E6',
     ['a wind chime', 'a small metal tube chime ringing', 'one aluminium wind chime tube struck by its clapper'],
     4.4, '-1 = the dead centre of the tube, +1 = near its end (every mode)',
     modes=_wind_chime, strike=(0.44, 0.5, 0.05), material='aluminium', support=(0.43, 0.0), loss=1.3, cord=2.0,
     order=0.5, mallet='wood', speed=1.5, track=0.2, knee=4000.0, tick=(0.15, 0.04, 0.003), damper=0.15, extent=0.5)


# triangle: a steel rod (10 mm) bent twice, struck with a steel beater. Pitch names its lowest bending mode, which is
# barely radiated; what is heard is the dense series above 2 kHz, every mode split in two (in the plane of the
# triangle and across it, a few percent apart because of the bends).
#   modes: free rod 1 : 2.76 : 5.40 : 8.93 : 13.3 : 18.6 : 24.8 ... each +-1.5 %
#   T60 of the modes at 3 - 6 kHz: 7 - 3 s
def _triangle(f0, x, vel):
    return modal.tube_modes(f0, 1, 0.0025, 5100.0, x, 22, 0.03, shear=3.9)


_add('triangle', 'D3', 'D5', 'A3',
     ['a triangle', 'an orchestral triangle struck with a metal beater, a bright shimmering ring',
      'a high metallic ting that rings on'], 5.9, '-1 = the middle of a side, +1 = right at the open corner',
     modes=_triangle, strike=(0.12, 0.5, 0.02), material='steel', support=(0.3, 0.0), loss=0.4,
     mallet='steel', speed=2.0, track=0.0, knee=9000.0, order=1.5, tick=(0.15, 0.03, 0.002), damper=0.12,
     extent=0.8)


# ------------------------------------------------------------------------------------------------------- bells
# hand bell: a bronze English handbell, clapper with a firm felt or plastic face. (2,0) is the note, (3,0) a tuned
# twelfth; the rest is untuned and weak.
#            C4 262 Hz        C5 523 Hz        C7 2093 Hz
#   (3,0)    3.00             3.00             3.00
#   (4,0)    5.73             5.50             5.06
#   T60 f1   16 s             10 s             2.6 s      (10 - 30 s reported for the fundamental of low handbells)
def _hand_bell(f0, x, vel):
    return modal.bell_modes('hand', x, f0, 0.0012)


_add('hand_bell', 'C3', 'C8', 'C5',
     ['a handbell', 'a small bronze bell rung by hand, a clear bell tone with a long ring', 'a bell ding'], 8.9,
     _BELL,
     modes=_hand_bell, strike=(0.0, -1.0, 1.0), material='bronze', support=(0.086, -0.64), loss=1.53, radiation=0.3,
     mallet='clapper', speed=2.5, track=0.5, knee_x=2.5, tick=(0.15, 0.04, 0.003), damper=0.12, extent=0.8)


# church bell: a tuned bronze bell rung by an iron clapper on the soundbow. Pitch names the strike note (the prime).
# The same profile is cast in every size, so the ratios do not change with pitch. The clapper stays on a big bell
# longer (about 1 ms on the bass bells of a carillon, 0.2 ms in the treble: Fletcher, McGee & Tarnopolsky 2002), so
# a big bell is darker than a small one played an octave down.
#   hum 0.5, prime 1, tierce 1.183, quint 1.506, nominal 2, superquint 3.011, octave nominal 4.166 (cents to the
#   nominal: -2400, -1200, -909, -491, 0, +708, +1271)
#            C3 131 Hz        C4 262 Hz        C5 523 Hz
#   T60 hum  95 s             50 s             26 s       (Q about 3000: 30 - 60 s reported for a 0.5 - 1 t bell)
#   nominal  23 s             11.5 s           5.8 s      (8 - 15 s reported)
#   oct.nom. 7.8 s            3.9 s            2.0 s      (upper partials 1 - 5 s)
def _church_bell(f0, x, vel):
    return modal.bell_modes('church', x, f0, 0.0008)


_add('church_bell', 'G2', 'C6', 'G4',
     ['a church bell', 'a large bronze bell tolling, a deep bell strike with a long hum',
      'a tower bell ringing once'], 19.0, _BELL,
     modes=_church_bell, strike=(0.0, -1.0, 1.0), material='bronze', support=(0.041, 0.68), loss=1.25, radiation=0.3,
     mallet='iron clapper', speed=2.5, track=0.5, knee_x=1.0, tick=(0.3, 0.08, 0.012), damper=0.5, extent=0.9)


# singing bowl: a hammered bronze bowl on a cushion, struck on the rim with a suede-covered stick.
#   rim modes (2,0) 1 : 2.74 : 5.05 : 7.83 : 11.0 : 14.9 (means of three measured bowls: 2.66 - 2.78, 4.84 - 5.18)
#            A3 220 Hz        A4 440 Hz        A5 880 Hz
#   T60 f1   30 s             19 s             11 s       (damping ratios of 1e-4 measured: 50 s free at 220 Hz)
#   beat     about 1 Hz       2 Hz             4 Hz       (pairs split by about 0.5 %)
def _singing_bowl(f0, x, vel):
    return modal.rim_modes('bowl', x, 0.005)


_add('singing_bowl', 'C3', 'C6', 'A3',
     ['a singing bowl', 'a Tibetan singing bowl struck once, a long humming metallic ring that slowly beats',
      'a meditation bowl ringing'], 28.0, _RIM,
     modes=_singing_bowl, strike=(0.0, -1.0, 1.0), material='bronze', support=(0.113, 0.16), loss=0.81,
     radiation=0.15, mallet='suede', speed=1.5, track=0.5, knee_x=1.5, tick=(0.1, 0.04, 0.004), damper=0.3,
     extent=0.9)


# wine glass: a crystal glass held by the stem, flicked with a fingernail near the rim.
#   rim modes 1 : 2.32 : 4.25 : 6.63 : 9.38 (measured, Rossing)
#            C5 523 Hz        G5 784 Hz        C7 2093 Hz
#   T60 f1   6 s              4.3 s            1.65 s     (2 - 10 s reported)
def _wine_glass(f0, x, vel):
    return modal.rim_modes('glass', x, 0.0012)


_add('wine_glass', 'C5', 'C7', 'G5',
     ['a wine glass ringing', 'a crystal glass tapped with a fingernail, a clear thin ring', 'a glass ding'], 3.6,
     _RIM,
     modes=_wine_glass, strike=(0.0, -1.0, 1.0), material='crystal', support=(0.3, 0.0), loss=1.0, radiation=0.1,
     mallet='nail', speed=2.0, track=0.2, knee_x=2.0, tick=(0.15, 0.03, 0.002), damper=0.08, extent=0.6)


# ----------------------------------------------------------------------------------------------------- steel pans
# A note field of a steel pan is a shallow shell, thin enough to be non-linear. Its lowest modes are tuned to the
# note, the octave and the twelfth (or double octave), never exactly; the motion at the note also drives exact
# harmonics, the more the harder the stroke (levels rising faster than the note itself). The tuned mode near 2 f0
# and the driven harmonic at exactly 2 f0 beat a few times per second: the shimmer of a pan.
# handpan: nitrided steel, played with the fingers; the air in the shell sounds at about 85 Hz under every note.
#            D3 147 Hz        A3 220 Hz        D5 587 Hz
#   modes    1 : 2 : 3        1 : 2 : 3        1 : 2 : 3  (octave and twelfth within a few cents)
#   T60 f1   3.5 s            2.6 s            1.3 s
_SHELL = ((85.0, 6.0, 1.0), (172.0, 5.0, 0.4), (410.0, 3.0, 0.25))      # air mode, then the shell: Hz, q, gain


def _handpan(f0, x, vel):
    push = 0.4 + vel
    tuned = modal.modes([1.0, 2.004, 2.994, 4.9, 6.3, 7.7],
                        [1.0 - 0.5 * x, 0.3 + 0.8 * x, 0.2 + 0.4 * x, 0.08, 0.05, 0.03],
                        damp=[1.0, 1.3, 1.6, 3.0, 3.5, 4.0])
    return modal.join(tuned, modal.modes([2.0, 3.0, 4.0], [0.22 * push, 0.12 * push ** 1.5, 0.05 * push ** 2],
                                         damp=[2.0, 3.0, 4.0]))


_add('handpan', 'C3', 'C5', 'A3',
     ['a handpan note', 'a hang drum played with the fingers, a warm steel tone with a soft attack',
      'a mellow steel pan-like note'], 2.3, _FIELD,
     modes=_handpan, strike=(0.35, 0.0, 1.0), material='steel', support=(2.6, 0.7), loss=1.0,
     order=0.5, mallet='finger', speed=1.5, track=0.2, knee=1200.0, tick=(0.05, 0.05, 0.006), knock=(0.12, _SHELL),
     damper=0.1, extent=0.5)


# steel drum: a note of a tenor pan (thin sunk oil-drum steel) struck with a rubber-tipped stick; the notes next
# to it (a fifth and a fourth away) ring along 20 - 30 dB down.
#            C4 262 Hz        C5 523 Hz        C6 1047 Hz
#   modes    1 : 2.006 : 2.988 (tuned by hammer, a few cents off), driven harmonics at exactly 2, 3, 4, 5
#   T60 f1   1.3 s            0.9 s            0.5 s
def _steel_drum(f0, x, vel):
    push = 0.3 + vel
    tuned = modal.modes([1.0, 2.006, 2.988, 4.03, 5.3, 6.6], [1.0 - 0.4 * x, 0.35 + 0.5 * x, 0.25, 0.12, 0.08, 0.05],
                        damp=[1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    driven = modal.modes([2.0, 3.0, 4.0, 5.0], [0.45 * push, 0.3 * push ** 1.5, 0.18 * push ** 2, 0.1 * push ** 2.5],
                         damp=[2.0, 3.0, 4.0, 5.0])
    return modal.join(tuned, driven, modal.modes([1.498, 2.996, 1.335, 2.67], [0.05, 0.03, 0.035, 0.02], damp=1.3))


_add('steel_drum', 'C4', 'E6', 'C5',
     ['a steel drum note', 'a steel pan from Trinidad struck with a rubber mallet, bright and tropical',
      'a caribbean steel pan note'], 0.7, _FIELD,
     modes=_steel_drum, strike=(0.3, 0.0, 1.0), material='steel', support=(7.5, 0.7), loss=1.0,
     mallet='pan stick', speed=3.0, track=0.2, knee=1500.0, tick=(0.1, 0.05, 0.004), damper=0.08, extent=0.6)


# ------------------------------------------------------------------------------------------------- small objects
# wood block: a slotted maple block struck with a drumstick: the wall over the slot and the air in it give one
# main resonance, a few plate modes above it, all gone in a tenth of a second.
#   modes (free wood plates measure 1 : 1.47 : 2.09 : 2.56)   T60 f1: 0.13 s at C5, 0.09 s at C6, 0.06 s at C7
def _wood_block(f0, x, vel):
    return modal.modes([1.0, 1.47, 2.09, 2.56, 3.4, 4.6, 6.1], [1.0, 0.45, 0.4, 0.3, 0.25, 0.2, 0.15])


_add('wood_block', 'C5', 'C7', 'C6',
     ['a wood block', 'a hollow wooden block hit with a drumstick, a short dry knock', 'a woodblock tock'], 0.1,
     'no effect: the block sounds the same wherever the stick lands',
     modes=_wood_block, strike=(0.5, 0.5, 0.5), material='maple', support=(60.0, 0.5), loss=0.4,
     mallet='drumstick', speed=3.0, track=0.0, knee=1500.0, tick=(0.3, 0.2, 0.004), damper=0.03, extent=0.3)


# cowbell: a folded and welded steel bell struck on its mouth edge with a stick: two strong low modes about a
# fifth apart (1 : 1.48) and a dense untuned series above; hand-held, so it dies in under a second.
#   T60 f1: 0.9 s at C4, 0.6 s at C#5, 0.4 s at C6
def _cowbell(f0, x, vel):
    return modal.modes([1.0, 1.48, 2.24, 2.78, 3.13, 3.6, 4.2, 4.75, 5.4, 5.9, 6.3, 7.1, 7.5, 8.3, 8.9, 9.8, 10.9],
                       [1.0, 0.9, 0.35, 0.5, 0.3, 0.4, 0.35, 0.3, 0.3, 0.25, 0.25, 0.2, 0.2, 0.18, 0.15, 0.12, 0.1],
                       split=0.004, order=1)


_add('cowbell', 'C4', 'C6', 'C#5',
     ['a cowbell', 'a metal cowbell hit with a drumstick, a short clanky tone', 'a latin percussion cowbell'], 0.5,
     'no effect',
     modes=_cowbell, strike=(0.0, 0.0, 0.0), material='steel', support=(8.3, 0.38), loss=11.6,
     mallet='drumstick', speed=3.0, track=0.0, knee=1200.0, tick=(0.4, 0.15, 0.004), damper=0.05, extent=0.6)


# coin: a disc of coin metal knocked on a hard table. Free-plate modes 1 : 1.73 : 2.33 (a measured 2 euro coin:
# 10.4 - 11.1 kHz, ratios 1 : 1.70 : 2.30, ringing 0.4 s; 1 euro 14 - 15.5 kHz; a 1 oz bullion coin 3.8 kHz).
# bounce > 0 drops it instead: impacts at intervals shrinking by the restitution, each from a new angle.
#            C8 4186 Hz       E9 10548 Hz      C10 16744 Hz
#   T60 f1   1.1 s            0.45 s           0.2 s
_TABLE = ((300.0, 3.5, 1.0), (850.0, 4.5, 0.7), (2100.0, 5.0, 0.45))    # the knock of the table top: Hz, q, gain


def _coin(f0, x, vel):
    return modal.plate_modes(x, False, 0.0015)


_add('coin', 'C8', 'C10', 'E9',
     ['a coin ringing', 'a coin dropped on a hard table, a tiny bright metallic ring', 'a coin clink'], 0.4,
     '-1 = it lands flatter, struck well inside the edge, +1 = exactly on the edge',
     options={'bounce': '0 .. 1: drop the coin from higher and let it bounce to rest (default 0: one ring)'},
     modes=_coin, strike=(0.95, 0.3, 1.0), material='brass', support=(0.76, -1.0), loss=1.17,
     mallet='stone', speed=2.0, track=0.0, knee=9000.0, tick=(0.12, 0.06, 0.002), knock=(0.3, _TABLE),
     damper=0.03, extent=0.6)


# glass clink: two stemmed glasses touched together; both ring (rim modes 1 : 2.32 : 4.25 ...), a whole tone or so
# apart, after a glass-on-glass contact of a twentieth of a millisecond. Pitch names the lower glass.
#   T60 f1: 1.8 s at C5, 1.2 s at A5, 0.55 s at C7 (ordinary glass, held in the hand, a little wine in it)
def _glass(f0, x, vel):
    return modal.rim_modes('glass', x, 0.0015, fall=0.0)


_add('glass_clink', 'C5', 'C7', 'D6',
     ['two glasses clinking', 'a toast, wine glasses touching with a bright clink', 'a glass clink'], 0.7, _RIM,
     options={'interval': 'pitch of the second glass above the first, in semitones (default 1.7)'},
     modes=_glass, strike=(0.25, -1.0, 1.0), material='glass', support=(1.2, 0.0), loss=1.0, radiation=0.1,
     mallet='glass', speed=1.0, track=0.0, knee_x=2.0, tick=(0.15, 0.05, 0.002), damper=0.06, extent=0.7)


# metal ping: a small steel offcut (6 x 20 mm bar) hanging free, tapped with something hard.
#            C5 523 Hz        C7 2093 Hz       C8 4186 Hz
#   mode 2   2.75             2.73             2.70
#   T60 f1   7 s              3.6 s            1.6 s
def _metal_ping(f0, x, vel):
    return modal.bar_modes(f0, x, 0.006, 0.020, 5100.0, 8, off_axis=0.3)


_add('metal_ping', 'C5', 'C8', 'C7',
     ['a metal ping', 'a small piece of steel struck, a bright metallic ting', 'a metallic ding'], 3.0,
     '-1 = the middle of the piece, +1 = its end (every mode)',
     modes=_metal_ping, strike=(0.42, 0.5, 0.05), material='steel', support=(0.44, -0.35), loss=1.05, cord=1.0,
     order=0.5, mallet='steel', speed=2.0, track=0.1, knee=5000.0, tick=(0.2, 0.04, 0.002), damper=0.08, extent=0.5)


# ------------------------------------------------------------------------------------------------------ rendering
def _built(name, f0):
    """the generator of the instrument itself (how it was cast and set up): the same for every take of a note"""
    return dsp.rng(zlib.crc32(name.encode()) + int(round(1200.0 * np.log2(f0 / 440.0))))


def _strike(name, f0, vel, c, r, built):
    """The physics of one stroke, before any sound is made: per partial its frequency (Hz), decay rate (1/s),
    complex amplitude and place in the stereo picture (-1 .. 1); the contact time (s) and force exponent; the index
    of the fundamental; the tube under it (or None); the mallet speed relative to a full stroke; and the seconds
    the note takes to fall 60 dB"""
    S = _SPEC[name]
    f_ref = hz(INSTRUMENTS[name]['default_pitch'])
    jit = r.standard_normal(4) * c['vary']
    speed = S['speed'] * max(vel, 0.02) * float(np.exp(0.10 * jit[0]))
    hard = c['hard'] + c['bright'] + 0.3 * jit[1]
    pos = float(np.clip(c['position'] + 0.4 * jit[2], -1.0, 1.0))
    usual, low, high = S['strike']
    x = usual + (high - usual) * pos if pos >= 0 else usual + (usual - low) * pos
    m = S['modes'](f0, x, vel)
    keep = f0 * m['ratio'] < 0.45 * SR
    m = {k: v[keep] for k, v in m.items()}
    freq = f0 * m['ratio']

    support, slope = S['support']
    rate = modal.decay_rate(freq, S['material'], 0.0, S.get('radiation', 0.0), f0) * S['loss']
    rate = rate + (support + S.get('cord', 0.0) * m['held']) * (f0 / f_ref) ** slope
    rate = rate * m['damp'] * 4.0 ** -c['ring'] * 2.0 ** (-0.3 * c['size'])

    if 'release' in S:                              # a pluck: the nail slips off faster on a harder pluck
        tau, expo = S['release'] * (max(vel, 0.02) / 0.7) ** -0.3 * 10.0 ** (-0.4 * hard), 1.5
    else:
        mass, stiff, expo = _MALLET[S['mallet']]
        tau = modal.contact_time(mass, stiff * 10.0 ** (1.2 * hard), speed, expo) * (f0 / f_ref) ** -S['track']
    tau = float(np.clip(tau * 2.0 ** (0.3 * c['size']), 2e-5, 0.012))

    knee = (S['knee_x'] * f0 if 'knee_x' in S else S['knee']) * 2.0 ** -c['size']
    lift = modal.radiation(freq, knee, S.get('order', 1.0)) / modal.radiation(f0, knee, S.get('order', 1.0))
    amp = m['gain'] * lift * (freq / f0) ** (3.0 * c['bright'] / 6.02) * (speed / S['speed'])
    amp = amp * modal.air_absorption(freq, 0.6)
    if 'release' in S:
        amp = amp * 1j

    index, detune, weight, side = modal.doublets(m['split'], m['order'], 0.4 + 0.6 * jit[3], built)
    place = built.uniform(0.5, 1.0, len(freq)) * built.choice([-1.0, 1.0], len(freq))
    azimuth = np.where(side == 0, place[index], side * np.abs(place[index]))
    first = int(np.nonzero(index == 0)[0][0])                      # where the fundamental ended up
    freq, rate, amp = freq[index] * detune, rate[index] * (1.0 + 0.04 * side), amp[index] * weight
    if 'body' in S:
        body = [(fk * 2.0 ** (-0.4 * c['size']), q, g) for fk, q, g in S['body']]
        amp = amp * modal.body_response(freq, body, S['direct'])

    tube = S['resonator'](f0) if 'resonator' in S else None
    life = rate.copy()
    if tube:
        life[first] *= 1.0 + tube['load'] * (1.0 + 0.5 * c['size'])
    natural = modal.ring_time(life, np.abs(amp) * modal.force_gain(freq, tau, expo), -60.0, 30.0) + 0.02
    return {'freq': freq, 'rate': rate, 'amp': amp, 'azimuth': azimuth, 'tau': tau, 'expo': expo, 'first': first,
            'tube': tube, 'drive': speed / S['speed'], 'natural': natural}


def _voice(name, f0, vel, c, r, seconds=None, dur=None):
    """one strike of one body, stereo (n, 2), in model units (note() scales it). seconds: stop there, uncut and
    unfaded (for the level reference); dur: the player damps it then"""
    S = _SPEC[name]
    built = _built(name, f0)
    hit = _strike(name, f0, vel, c, r, built)
    freq, rate, amp, tau, expo, first, tube, drive = (hit[v] for v in ('freq', 'rate', 'amp', 'tau', 'expo', 'first',
                                                                       'tube', 'drive'))
    mute = S['damper'] * 4.0 ** (0.5 * c['ring'])                  # seconds the damped note takes to die
    if seconds is not None:
        n = dsp.n_of(seconds)
    elif dur is not None and dur + 1.4 * mute < hit['natural']:
        n = dsp.n_of(dur + 1.4 * mute)
    else:
        n, dur = dsp.n_of(hit['natural']), None

    wide = 1.0 + c['width']
    pair = modal.mic_pair(amp, hit['azimuth'], freq, 0.25 * S['extent'] * wide, 2.5e-4 * wide, 0.6 * wide ** 0.8)
    if tube:
        tubed = [first] + ([int(np.argmin(np.abs(freq - 3.0 * f0)))] if 'twelfth_db' in tube else [])
        rest = np.ones(len(freq), bool)
        rest[tubed] = False
        y = modal.ring(freq[rest], rate[rest], pair[:, rest], n, tau, expo).imag.T
        for k, i in enumerate(tubed):
            boost = (tube['boost_db'] if k == 0 else tube['twelfth_db']) + 5.0 * c['size']
            gain = modal.resonator(n, freq[i], rate[i], tube['q'] / (1.0 + 2.0 * k), max(boost, 0.0),
                                   tube['load'] * (1.0 + 0.5 * c['size']) / (1.0 + k), tube['detune'],
                                   c.get('motor', 0.0), phase=built.uniform(0.0, np.pi))
            y += (modal.ring(freq[i:i + 1], rate[i:i + 1], pair[:, i:i + 1], n, tau, expo) * gain).imag.T
    else:
        y = modal.ring(freq, rate, pair, n, tau, expo).imag.T
    if 'knock' in S:                                # what the strike does to the thing the instrument sits on
        level, thud = S['knock']
        fk, qk, gk = (np.array(v) for v in zip(*thud))
        fk = fk * 2.0 ** (-0.4 * c['size'])
        y += modal.ring(fk, np.pi * fk / qk, level * gk * drive, n, max(tau, 2e-4), expo).imag.T

    tick = modal.contact_noise(tau, r, expo, *S['tick']) * drive
    k = min(n, len(tick))
    y[:k] += tick[:k, None]
    if c.get('buzz', 0.0) > 0:
        y += _buzz(n, f0, rate[first], c['buzz'] * drive, r)[:, None]

    y = modal.early(y, built, 0.2, wide)
    if dur is not None:
        y *= modal.damper(n, dur, mute)[:, None]
    if seconds is not None:
        return y
    tail = 0.15 * n / SR if dur is None else min(0.15 * n / SR, 1.2 * mute)     # never fade before the damper lands
    return dsp.fade(y, 0.0, max(0.02, tail))


def _buzz(n, f0, rate, amount, r):
    """A loose ring or bottle cap on a kalimba tine: while the tine swings wider than the play of the ring, the
    ring is thrown and lands once per cycle: a train of tiny metallic ticks at the pitch of the note, each a few
    kHz wide, that thins out and stops as the note decays"""
    t = np.arange(0.5 / f0, n / SR, 1.0 / f0)
    t = t + r.normal(0.0, 0.03 / f0, len(t))
    level = np.maximum(np.exp(-rate * t) - 0.25, 0.0) * (1.0 + 0.3 * r.standard_normal(len(t)))
    hits = np.zeros(n)
    np.add.at(hits, np.clip((t * SR).astype(np.int64), 0, n - 1), np.maximum(level, 0.0))
    k = dsp.n_of(0.004)
    tick = dsp.butter(r.standard_normal(k), 'band', (2500.0, 9000.0), 2) * np.exp(-dsp.t_of(k) / 0.0008)
    tick[:8] *= np.linspace(0.0, 1.0, 8)
    return 0.35 * amount * np.convolve(hits, tick / (np.abs(tick).max() + 1e-12))[:n]


def _clink(name, f0, vel, c, r, seconds=None, dur=None):
    """two glasses: one contact, both ring; the second glass is a little higher and takes the knock on its side"""
    a = _voice(name, f0, vel, c, r, seconds, dur)
    other = dict(c, position=float(np.clip(c['position'] + 0.5, -1.0, 1.0)))
    b = _voice(name, f0 * 2.0 ** (c.get('interval', 1.7) / 12.0), 0.8 * vel, other, r, seconds, dur)
    y = np.zeros((max(len(a), len(b)), 2))
    y[:len(a)] += a * (1.0, 0.8)
    y[:len(b)] += b * (0.75, 1.0)
    return y


def _drop(name, f0, vel, c, r, seconds=None, dur=None):
    """a coin dropped on a table: it lands, rings, and bounces at intervals that shrink by the restitution (about
    0.6 for metal on wood or stone) until it lies still and the table damps it. Every landing is a new strike at
    another point of the edge, so the doublets of the disc mix differently each time"""
    if c.get('bounce', 0.0) <= 0 or seconds is not None:
        return _voice(name, f0, vel, c, r, seconds, dur)
    gap, e = 0.10 + 0.25 * c['bounce'], 0.55 + 0.1 * r.uniform()
    times, levels = [0.0], [1.0]
    while gap > 0.012 and len(times) < 14:
        times.append(times[-1] + gap * r.uniform(0.9, 1.1))
        levels.append(levels[-1] * e * r.uniform(0.8, 1.15))
        gap *= e
    rest = times[-1] + 0.03
    out = np.zeros((dsp.n_of(rest + 0.25), 2))
    for t, g in zip(times, levels):
        hit = dict(c, vary=max(c['vary'], 0.6), bounce=0.0)
        dsp.place(out, _voice(name, f0, vel * g, hit, r, None, max(rest - t, 0.01)), t)
    return dsp.fade(dsp.trim_tail(out, -66.0, 0.02), 0.0, 0.02)


_COMPOSE = {'glass_clink': _clink, 'coin': _drop}


def _pitch(name, pitch):
    """the pitch in Hz (a note name, a whole number 0 .. 127 as a MIDI note, any other number as Hz), moved by
    octaves into the playable range of the instrument"""
    I = INSTRUMENTS[name]
    if isinstance(pitch, (int, np.integer)) and not isinstance(pitch, bool) and 0 <= pitch <= 127:
        f = 440.0 * 2.0 ** ((int(pitch) - 69) / 12.0)
    else:
        f = hz(I['default_pitch'] if pitch is None else pitch)
    lo, hi = hz(I['range'][0]), hz(I['range'][1])
    while f < lo * 0.999:
        f *= 2.0
    while f > hi * 1.001:
        f *= 0.5
    return f


def _controls(name, controls):
    opts = INSTRUMENTS[name]['options']
    bad = sorted(set(controls) - set(CONTROLS) - set(opts))
    if bad:
        raise ValueError(f'{name} has no control {bad[0]!r}: valid are {", ".join(CONTROLS + tuple(opts))}')
    c = {k: float(np.clip(controls.get(k, 0.0), 0.0 if k == 'vary' else -1.0, 1.0)) for k in CONTROLS}
    c.update({k: float(controls[k]) for k in opts if k in controls})
    return c


@functools.lru_cache(maxsize=512)
def _peak(name, f0):
    """peak of the first half second of a full stroke (vel 1, the instrument as built), the highest of three takes
    (the contact noise differs from take to take): the level reference"""
    play, c = _COMPOSE.get(name, _voice), _controls(name, {})
    return max(float(np.abs(play(name, f0, 1.0, c, dsp.rng(seed), 0.5)).max()) for seed in (0, 1, 2))


def note(instrument, pitch=None, vel=0.7, dur=None, r=None, **controls):
    """One note of a struck or plucked instrument: stereo float64 (n, 2) at 48 kHz.

    instrument: a name in INSTRUMENTS. pitch: a note name ('C6'), a MIDI note (a whole number 0 .. 127: 60 is C4), Hz
    (any other number: 440.0, 1046.5), or None for the default pitch of the instrument; a pitch outside the
    playable range is moved into it by octaves. vel 0 .. 1: mallet speed, which sets the level and the brightness
    (a faster stroke is a shorter contact). Nothing is normalised: a full stroke on the instrument as built peaks
    near 0.9, softer strokes are lower in proportion, and controls that add brightness add level; only a note that
    would pass 0.98 is turned down to it. dur: None lets the note ring out; a number of seconds damps it there the
    way a player mutes it (the array is a little longer than dur: the damped tail). r: numpy Generator for the take
    (default dsp.rng(0)). controls: bright, hard, ring, size, position, width in -1 .. 1, vary in 0 .. 1, and the
    options of the instrument (see the module docstring and INSTRUMENTS[name]['options'])"""
    if instrument not in INSTRUMENTS:
        raise ValueError(f'no instrument {instrument!r}: valid are {", ".join(INSTRUMENTS)}')
    f0 = _pitch(instrument, pitch)
    c = _controls(instrument, controls)
    r = dsp.rng(0) if r is None else r
    y = _COMPOSE.get(instrument, _voice)(instrument, f0, float(np.clip(vel, 0.0, 1.0)), c, r, None, dur)
    y *= 0.9 / _peak(instrument, round(f0, 4))
    return y * min(1.0, 0.98 / (np.abs(y).max() + 1e-12))


def chord(instrument, pitches, vel=0.7, spread_s=0.0, dur=None, r=None, **controls):
    """Several notes of one instrument struck together, or rolled with spread_s seconds between successive notes
    (in the order given). Stereo (n, 2); when the notes add up over 0.9 the whole chord is scaled down to it"""
    if not len(pitches):
        raise ValueError('a chord needs at least one pitch')
    r = dsp.rng(0) if r is None else r
    notes = [note(instrument, p, vel, dur, r, **controls) for p in pitches]
    starts = [int(round(i * spread_s * SR)) for i in range(len(notes))]
    out = np.zeros((max(s + len(y) for s, y in zip(starts, notes)), 2))
    for s, y in zip(starts, notes):
        out[s:s + len(y)] += y
    return out * min(1.0, 0.9 / (np.abs(out).max() + 1e-12))
