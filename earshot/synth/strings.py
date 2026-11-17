"""Strings, pianos and electric pianos as physics: one note at a time, every sample computed.

A code model's piano is a sine with harmonics under one decay. A real string is stiff, so its partials run sharp
(f_k = f0 k sqrt(1 + B k^2)); each partial dies at its own rate (air, internal friction and the bridge); it swings in
two polarisations and, on a piano, sits in a unison of two or three strings on one bridge, so a note has a loud fast
"prompt sound", a quiet slow "aftersound" and slow beats; a felt hammer stays on the string for a shorter time the
harder it is thrown, so loud is brighter; and everything is heard through a soundboard or body whose own short-lived
modes are the knock at the start of the note. This module models those things and adds nothing else: no sample, no
reverb (a shared room is another module's job).

    from earshot.synth import strings
    y = strings.note('felt_piano', 'C4', vel=0.5)                  # stereo (n, 2), rings to its natural end
    y = strings.note('electric_piano', 'E3', vel=0.9, dur=0.8)     # the damper stops it at 0.8 s
    y = strings.note('nylon_guitar', 196.0, hard=-0.5, position=0.6)
    y = strings.chord('harp', ['C4', 'E4', 'G4', 'C5'], vel=0.6, strum_s=0.25)
    strings.INSTRUMENTS['harp']                                    # range, says, default_pitch, ring_s

How a note is made. The string group is a set of modes (partial x polarisation x string of the unison, coupled
through the bridge after Weinreich 1977). The body is a second set of modes plus a smooth radiation curve. The
excitation is a force pulse a few milliseconds long: a felt hammer simulated against the string it strikes, or the
release of a plucked string. The cascade pulse -> strings -> body is expanded in partial fractions, so the note is a
sum of exact damped sinusoids (string modes coloured by the body, body modes rung by the string's attack) rendered by
matrix products, and the first milliseconds are the pulse convolved with the modes' onset: a real attack, no fade-in.
The electric pianos are a struck tine or plucked reed in front of a pickup whose output is a nonlinear function of
where the tine is, which is why they bark when played hard.
"""
import numpy as np
from scipy import signal

from ..forge import dsp, fx
from ..forge.dsp import SR, njit
from ..forge.knobs import hz as _hz

CONTROLS = ('bright', 'hard', 'ring', 'size', 'position', 'width', 'vary')

INSTRUMENTS = {
    'felt_piano': {'range': ('A0', 'C8'), 'default_pitch': 'C4', 'ring_s': 12.0, 'says': [
        'a soft felt piano note', 'an upright piano with felted hammers played softly, recorded close',
        'a warm, muted, intimate piano key']},
    'grand_piano': {'range': ('A0', 'C8'), 'default_pitch': 'C4', 'ring_s': 14.0, 'says': [
        'a grand piano note', 'a single clear piano key ringing out', 'a concert piano note']},
    'electric_piano': {'range': ('E1', 'E7'), 'default_pitch': 'C4', 'ring_s': 8.0, 'says': [
        'an electric piano note', 'a mellow tine electric piano key with a bell-like attack',
        'a 1970s electric piano note']},
    'reed_piano': {'range': ('A1', 'C7'), 'default_pitch': 'C4', 'ring_s': 3.0, 'says': [
        'a reed electric piano note', 'a hollow, biting vintage electric piano key']},
    'harp': {'range': ('C1', 'G7'), 'default_pitch': 'G4', 'ring_s': 3.0, 'says': [
        'a harp note', 'a plucked concert harp string ringing', 'a soft round plucked string']},
    'nylon_guitar': {'range': ('E2', 'B5'), 'default_pitch': 'A3', 'ring_s': 3.0, 'says': [
        'a classical guitar note', 'a nylon-string acoustic guitar plucked with a finger',
        'a warm fingerpicked guitar string']},
    'steel_guitar': {'range': ('E2', 'E6'), 'default_pitch': 'A3', 'ring_s': 4.5, 'says': [
        'an acoustic guitar note', 'a steel-string acoustic guitar picked', 'a bright ringing guitar string']},
    'muted_guitar': {'range': ('E2', 'E5'), 'default_pitch': 'E3', 'ring_s': 0.35, 'says': [
        'a palm-muted guitar note', 'a short muffled guitar pluck', 'a dead, thumpy plucked guitar string']},
    'pizzicato': {'range': ('C2', 'E6'), 'default_pitch': 'D4', 'ring_s': 0.9, 'says': [
        'a pizzicato string note', 'a plucked violin string', 'a short, woody plucked string']},
    'koto': {'range': ('C3', 'C6'), 'default_pitch': 'D4', 'ring_s': 2.0, 'says': [
        'a koto note', 'a Japanese koto string plucked with a pick', 'a bright twangy zither pluck']},
    'upright_bass': {'range': ('E1', 'G4'), 'default_pitch': 'D2', 'ring_s': 2.0, 'says': [
        'an upright bass note', 'a plucked double bass string', 'a jazz bass pizzicato note']},
    'harpsichord': {'range': ('F1', 'F6'), 'default_pitch': 'C4', 'ring_s': 5.5, 'says': [
        'a harpsichord note', 'a harpsichord key, a string plucked by a quill', 'a bright baroque keyboard note']},
    'dulcimer': {'range': ('D3', 'D6'), 'default_pitch': 'A4', 'ring_s': 5.0, 'says': [
        'a hammered dulcimer note', 'metal strings struck with a small wooden hammer',
        'a shimmering struck zither note']},
    'banjo': {'range': ('C3', 'D6'), 'default_pitch': 'D4', 'ring_s': 1.3, 'says': [
        'a banjo note', 'a plucked banjo string', 'a bright twangy banjo pick']},
}

# Literature targets the builds are calibrated to (each build was measured against them). Sources (as recalled from them, rounded; attack times and centroids are class-level ranges, not data):
# [F] Fletcher 1964 and Conklin 1996, inharmonicity of piano strings; [C] Chaigne & Askenfelt 1994, string and
# hammer data of C2, C4 and C7 (B = pi^2 eps: 7e-5, 3.8e-4, 8.6e-3; felt F = K x^p with p 2.3, 2.5, 3.0);
# [A] Askenfelt & Jansson 1990-91, hammer contact about 4 ms in the bass, 2 ms at C4, under 1 ms at the top, shorter
# when louder; [M] Martin 1947 and [W] Weinreich 1977, decay rates and the two-stage decay of coupled strings (a
# mid-range note starts at about 8-10 dB/s and ends at 2-3 dB/s); [V] Valette 1995 and Woodhouse 2004, loss of nylon
# and steel guitar strings; [R] Fletcher & Rossing ch. 9-12, body modes; [G] Gabrielli et al. 2020, tine overtones at
# 7.11 f and 20.25 f.
#
#   note                  B          T60 partial 1         T60 near 3 kHz   attack     centroid mf / ff (Hz)
#   grand piano C2        7e-5       13 s, then 35 s       1.5 s            3-12 ms    500-900 / 800-1400
#   grand piano C4        3.8e-4     2.5 s, then 25 s      1.5 s            8-25 ms    350-550 / 450-700
#   grand piano C7        8.6e-3     1.5 s, then 3 s       -                1-5 ms     2000-2400 / 2100-2600
#   felt piano C2         1.3e-4     11 s, then 30 s       0.9 s            5-20 ms    250-450 / 350-700
#   felt piano C4         6.8e-4     3 s, then 20 s        0.9 s            8-30 ms    280-450 / 350-600
#   felt piano C6         4.7e-3     0.7 s, then 5 s       -                2-25 ms    1050-1300 / 1050-1400
#   electric piano F2     harmonic   14 s                  ("bell" 0.5 s)   3-12 ms    150-400 / 300-900
#   electric piano C4     harmonic   7 s                   ("bell" 0.3 s)   3-10 ms    350-600 / 600-1100
#   electric piano C6     harmonic   3 s                   ("bell" 0.15 s)  2-6 ms     1100-1500 / 1300-1900
#   harp C2 (wire)        7e-5       12 s                  1.0 s            3-15 ms    250-500 / 350-650
#   harp C4 (gut, nylon)  3e-5       5 s                   0.9 s            2-10 ms    450-700 / 550-900
#   harp C6               1.5e-4     1.2 s                 0.6 s            2-8 ms     1100-1500 / 1200-1700
#   nylon guitar E2       2e-5       5 s                   1.0 s            2-10 ms    350-650 / 450-800
#   nylon guitar G3       1e-4       3.5 s                 0.8 s            2-10 ms    450-750 / 550-900
#   nylon guitar E5       8e-5       1.6 s                 0.6 s            2-8 ms     800-1200 / 900-1500
# A partial's T60 is 6.91 / rate. Rates follow rate = a + b1 f + b2 f^2 per string (air, internal friction) plus what
# the bridge takes from the polarisation that moves it; the T60 near 3 kHz of nylon strings follows from the loss
# factors of [V]. mf is vel 0.5, ff vel 1.
#
# [cal] The pianos are also calibrated against recordings: eight single notes of a
# 2.1 m grand (University of Iowa MIS: C2 to C7, and pp / mf / ff at C4, microphones a hand's breadth over the
# strings) and seven of an upright (VSCO-2 CE: C2 to C6, three dynamics at C4). What the notes of both agree on:
#   prompt sound   the fundamental falls its first 12 dB at 12-20 dB/s at C3, 16-32 at C4, 43-48 at C5 and 60-100
#                  at C6 (the 8-10 dB/s of [M] is the slope of the whole first seconds); then the aftersound takes
#                  over 22-30 dB down and falls 2-3 dB/s, and the unison beats more the higher the note
#                  (bridge, bridge_h, pol, detune)
#   knock          between the partials from 150 to 1500 Hz the first 50 ms hold -21 to -27 dB of the note's
#                  power at C4, at mf and ff alike (it is board and case answering the blow, so it follows the
#                  note), -13 to -15 dB at C6 and -7 dB at C7; 150 ms later 22-26 dB less; little above 2 kHz
#                  (cloud, diffuse, knock, knock_hz, knock_vel, case)
#   spectrum       the partials above 3 kHz of a C4 at mf lie 47 dB under its fundamental (30 dB in the upright),
#                  with no hiss between them (rough); from C5 up the fundamental stands nearly alone, partial 2 of
#                  C6 at -18 to -27 dB and of C7 at -34 dB (h_stiff above C4); the fundamental of C2 lies 26 to
#                  39 dB under its strongest partial (hp); the tilt of the board (flat, lp, lp_order)
#   attack         the 3 ms envelope of a C4 takes 13-25 ms from 10 % to 90 % of its peak (cloud, diffuse)
# Both libraries kept their recording level, and mf lies 11-15 dB under ff and pp another 13-20 dB lower: mf is a
# hammer at about 1 m/s (vel 0.3 here), ff one near 5 m/s. The centroids and attack times of the two C4 rows and
# the C7 row above are those measured on these notes; the other rows are still literature values.

_KEYS = (21, 36, 60, 84, 96, 108)                    # A0 C2 C4 C6 C7 C8: where the piano tables are given


def _tab(*values):
    return tuple(zip(_KEYS, values))


_GRAND = dict(
    kind='hammer', seed=11, open=None, stop_loss=0.0,
    B=((21, 2.8e-4), (28, 1.2e-4), (36, 7.4e-5), (48, 1.6e-4), (60, 3.8e-4), (72, 9.5e-4), (84, 2.6e-3),
       (96, 8.6e-3), (108, 2.5e-2)),
    strings=((29, 1), (41, 2), (200, 3)),           # one string per note up to key 29, two up to 41, then three
    # spread of a tuned unison, Hz at the fundamental: close to one cent from C4 up [cal]
    detune=((21, 0.06), (48, 0.08), (60, 0.14), (72, 0.27), (84, 0.4), (108, 0.8)),
    uneven=0.15,                                    # how unequally the hammer meets the strings of a unison
    # b2: 2.5e-7 s is the wire alone [C]; the rest is what bridge and board take at high frequencies, set so that
    # the band near 3 kHz of a mid-range note falls 60 dB in about 1.5 s (partial 10 of C4: some 40 dB/s)
    loss=(_tab(0.14, 0.15, 0.2, 0.6, 1.2, 2.4), 0.0, _tab(4.5e-7, 4.5e-7, 4.5e-7, 1e-7, 4e-8, 2e-8)),
    # what one string alone loses to the bridge, 1/s: the prompt sound of the unison [cal]
    bridge=((21, 0.24), (36, 0.19), (48, 0.45), (60, 0.85), (72, 0.75), (84, 1.8), (96, 1.5), (108, 3.0)),
    bridge_h=0.05, bridge_fc=2500.0, pol=(0.3, 0.6),
    # Hammer and string, per string of the unison: masses, impedances and striking points of Chaigne & Askenfelt
    # 1994 at C2, C4, C7, extended to both ends. The felt is close to theirs at a force of 20 N up to C4 and softer
    # above it [cal], with an exponent a quarter higher (inside the measured 2.2-3.5) and stiffening with the speed of
    # compression (h_hyst, s/m: Stulov's hysteresis in its simplest form), so that it is as soft at pianissimo as
    # contact times and spectra say. rough is the grain of the felt: the share of the force that fluctuates during
    # the contact; it is what a soft note has above 3 kHz, and no more than that [cal].
    h_mass=_tab(10.5e-3, 4.9e-3, 2.97e-3, 2.4e-3, 2.2e-3, 2.0e-3),
    h_stiff=_tab(1.67e8, 7.0e8, 8.2e9, 7.0e10, 4.0e11, 2.0e12), h_exp=_tab(2.45, 2.55, 2.75, 3.05, 3.25, 3.45),
    h_hyst=0.15, rough=0.008, z=_tab(12.0, 3.8, 2.06, 2.0, 1.98, 1.9),
    strike=_tab(0.12, 0.12, 0.12, 0.085, 0.0625, 0.05),
    speed=(0.3, 5.0, 1.6),                          # hammer speed m/s at vel 0 and 1, and the curve between
    damp=((21, 0.45), (40, 0.25), (60, 0.12), (88, 0.06)), damp_top=88.5, damp_at=0.16,     # no dampers above E6
    # the board [cal]: level from 350 Hz up for the long bridge (the bass bridge keeps its rise to 1400 Hz), 12 dB
    # per octave down above 2.6 kHz, deaf below 100 Hz and, at the stiff treble end of the bridge, below some
    # hundred Hz; its modes carry twice the smooth path, ring with a loss factor near 2 % and reach the
    # microphones in any phase
    hp=((84, 100.0), (96, 500.0), (108, 1000.0)), flat=((48, 1400.0), (60, 350.0)), lp=2600.0, lp_order=2, modes=(),
    cloud=(60.0, 8000.0, 17.0, 0.028, 50.0, 90.0, 2.0), diffuse=1.0,
    bridge_k=0.11, pan=0.22, spread=0.75, sym=0.3,
    # the blow through action, keybed and case: as loud against the note at every dynamic [cal]
    knock=((21, -30.0), (36, -28.0), (48, -24.0), (60, -18.0), (72, -19.0), (84, -15.0), (96, -20.0), (108, -16.0)),
    knock_hz=600.0, knock_vel=0.0, case=(150.0, 2500.0, 60, 0.5, 0.2), felt=None, release=-38.0)

_FELT = dict(
    _GRAND, seed=12,
    B=tuple((k, 1.8 * b) for k, b in _GRAND['B']), uneven=0.3,
    detune=((21, 0.09), (48, 0.12), (60, 0.2), (72, 0.4), (84, 0.6), (108, 1.2)),
    loss=(_tab(0.17, 0.18, 0.25, 0.75, 1.5, 3.0), 0.0, _tab(7e-7, 7e-7, 7e-7, 2e-7, 8e-8, 5e-8)),
    bridge=((21, 0.27), (36, 0.2), (48, 0.2), (60, 0.42), (72, 0.85), (84, 2.2), (96, 1.7), (108, 3.4)),
    # an upright's lighter hammers striking through the felt strip of the practice rail: a softer, lossier spring
    h_mass=_tab(9.0e-3, 4.2e-3, 2.6e-3, 2.1e-3, 1.9e-3, 1.8e-3),
    h_stiff=_tab(9.0e7, 3.8e8, 4.5e9, 6.0e10, 3.8e11, 1.7e12), h_exp=_tab(2.65, 2.75, 2.95, 3.25, 3.45, 3.65),
    h_hyst=0.3,
    speed=(0.25, 4.2, 1.6),
    hp=((72, 130.0), (84, 300.0), (96, 600.0), (108, 1000.0)), flat=450.0, lp=2600.0,
    cloud=(85.0, 7000.0, 22.0, 0.03, 50.0, 90.0, 2.0), pan=0.15, spread=0.6, sym=0.2,
    knock=((21, -28.0), (36, -26.0), (48, -22.0), (60, -16.0), (72, -17.0), (84, -13.0), (96, -18.0), (108, -14.0)),
    case=(140.0, 2500.0, 60, 0.5, 0.2), felt=(-30.0, 120.0, 1400.0), release=-34.0)

_PLUCK = dict(
    kind='pluck', strings=((200, 1),), detune=0.0, bridge_h=0.25, bridge_fc=3000.0, pol=(0.0, 0.3), fc_vel=1.6,
    stop_loss=0.0, open=None, damp=0.07, damp_top=200.0, damp_at=None, bridge_k=0.03, pan=0.0,
    spread=0.7, sym=0.0, knock=None, knock_hz=300.0, knock_vel=8.0, case=None, felt=None, release=-44.0, modes=(),
    lp_order=1, diffuse=0.0)

_SPEC = {
    'grand_piano': _GRAND,
    'felt_piano': _FELT,
    'harp': dict(
        _PLUCK, seed=21, B=((24, 1.5e-4), (40, 6e-5), (60, 3e-5), (84, 1.5e-4), (103, 5e-4)),
        loss=(((24, 0.2), (36, 0.28), (60, 0.6), (84, 2.4), (103, 8.0)), 1.6e-3, 2.4e-7),
        bridge=((24, 0.15), (36, 0.2), (60, 0.55), (84, 2.0), (103, 5.0)),
        pluck=0.37, fc=((24, 450.0), (60, 1200.0), (103, 3000.0)), width=60.0, noise=0.07, angle=42.0,
        hp=110.0, flat=380.0, lp=4500.0,
        modes=((150, 22, 2.2), (190, 25, 1.8), (245, 28, 2.0), (320, 30, 1.5), (410, 30, 1.4), (520, 30, 1.2)),
        cloud=(600.0, 8000.0, 40.0, 0.04, 35.0, 20.0, 0.5), bridge_k=0.09, pan=0.12, damp=0.09, release=-42.0),
    'nylon_guitar': dict(
        _PLUCK, seed=22, open=(82.41, 110.0, 146.83, 196.0, 246.94, 329.63),
        B=((40, 2e-5), (50, 2.5e-5), (55, 1e-4), (59, 5e-5), (64, 2e-5), (83, 2e-5)),
        loss=(((40, 0.5), (52, 0.55), (55, 0.7), (83, 0.8)), ((40, 1.8e-3), (52, 1.8e-3), (55, 2.5e-3), (83, 2.5e-3)),
              ((40, 3.0e-7), (52, 3.0e-7), (55, 4.5e-7), (83, 4.5e-7))),      # wound basses, plain nylon trebles
        stop_loss=0.2, bridge=0.7, pluck=0.185, fc=1900.0, width=70.0, noise=0.1, angle=40.0,
        hp=80.0, flat=420.0, lp=3600.0,
        modes=((98, 28, 3.0), (198, 30, 3.5), (242, 35, 1.3), (285, 30, 1.5), (385, 35, 1.6), (430, 35, 1.2),
               (545, 35, 1.5), (640, 35, 1.1)),
        cloud=(700.0, 7000.0, 45.0, 0.045, 38.0, 22.0, 0.5)),
    'steel_guitar': dict(
        _PLUCK, seed=23, open=(82.41, 110.0, 146.83, 196.0, 246.94, 329.63),
        B=((40, 4e-5), (58, 4e-5), (59, 8e-5), (64, 3e-5), (88, 3e-5)),
        loss=(0.35, 1.3e-3, 7e-8), stop_loss=0.2, bridge=0.45, pluck=0.17, fc=5000.0, width=150.0, noise=0.09,
        angle=25.0, hp=85.0, flat=550.0, lp=5500.0,
        modes=((102, 30, 3.0), (185, 30, 3.2), (220, 35, 1.4), (300, 30, 1.6), (390, 35, 1.5), (480, 35, 1.3),
               (600, 35, 1.2)),
        cloud=(700.0, 9000.0, 45.0, 0.04, 40.0, 22.0, 0.5), damp=0.08),
    'muted_guitar': dict(
        _PLUCK, seed=23, open=(82.41, 110.0, 146.83, 196.0, 246.94, 329.63),
        B=((40, 4e-5), (58, 4e-5), (59, 8e-5), (64, 3e-5), (88, 3e-5)),
        loss=(14.0, 3.5e-2, 7e-8),                  # the palm on the strings at the bridge: every partial dies fast
        bridge=0.7, pluck=0.16, fc=2800.0, width=150.0, noise=0.08, angle=25.0, hp=85.0, flat=380.0, lp=5000.0,
        modes=((102, 30, 3.0), (185, 30, 3.2), (220, 35, 1.4), (300, 30, 1.6), (390, 35, 1.5), (480, 35, 1.3),
               (600, 35, 1.2)),
        cloud=(700.0, 9000.0, 45.0, 0.04, 40.0, 22.0, 0.5), knock=-18.0, knock_hz=260.0, damp=0.04),
    'pizzicato': dict(
        _PLUCK, seed=24, open=(65.41, 98.0, 146.83, 196.0, 293.66, 440.0, 659.26), B=6e-5,
        loss=(1.2, 1.6e-2, 0.0), stop_loss=1.0, bridge=0.8, pluck=0.22, fc=2400.0, width=35.0, noise=0.15,
        angle=27.0,                                 # plucked along the bowing direction, the one the bridge passes on
        hp=240.0, flat=1200.0, lp=4500.0, size_by=((45, 2.9), (57, 1.0)),
        modes=((275, 14, 2.5), (405, 30, 1.0), (460, 40, 3.0), (550, 40, 3.2), (700, 30, 1.6), (860, 30, 1.6),
               (1050, 28, 1.5), (1300, 25, 1.5), (2500, 2.5, 1.8)),      # the last: the bridge's own resonance
        cloud=(1500.0, 8000.0, 80.0, 0.05, 30.0, 18.0, 0.6), knock=-24.0, knock_hz=350.0, damp=0.05),
    'koto': dict(
        _PLUCK, seed=25, B=3e-5, loss=(((48, 0.7), (62, 1.0), (84, 2.2)), 2.5e-3, 2.5e-7), bridge=0.7,
        pluck=0.07, fc=4500.0, width=200.0, noise=0.14, angle=23.0, hp=100.0, flat=450.0, lp=5000.0,
        modes=((118, 18, 1.6), (175, 20, 1.5), (228, 22, 1.8), (310, 25, 1.6), (395, 25, 1.4), (520, 28, 1.3)),
        cloud=(600.0, 9000.0, 40.0, 0.04, 32.0, 20.0, 0.5), bridge_k=0.2, pan=0.1, knock=-30.0, knock_hz=500.0),
    'upright_bass': dict(
        _PLUCK, seed=26, open=(41.2, 55.0, 73.42, 98.0), B=1.2e-4, loss=(1.0, 1.1e-2, 0.0), stop_loss=0.5,
        bridge=0.7, pluck=0.17, fc=600.0, width=22.0, noise=0.1, angle=27.0, hp=55.0, flat=300.0, lp=2500.0,
        modes=((62, 12, 2.2), (104, 25, 2.8), (118, 25, 1.5), (160, 28, 2.0), (195, 28, 1.6), (250, 30, 1.6),
               (310, 30, 1.3), (390, 30, 1.3)),
        cloud=(450.0, 5000.0, 40.0, 0.045, 30.0, 18.0, 0.5), knock=-20.0, knock_hz=220.0, damp=0.09),
    'harpsichord': dict(
        _PLUCK, seed=27, B=((29, 3e-5), (60, 5e-5), (89, 2e-4)),
        loss=(((29, 0.25), (60, 0.5), (89, 1.4)), 8e-4, 5e-8), bridge=((29, 0.3), (60, 0.6), (89, 1.6)),
        pluck=((29, 0.08), (60, 0.14), (89, 0.3)), fc=6000.0, fc_vel=0.1, width=300.0, noise=0.05, angle=17.0,
        hp=60.0, flat=250.0, lp=6500.0, cloud=(65.0, 10000.0, 16.0, 0.028, 50.0, 22.0, 0.6),
        bridge_k=0.11, pan=0.2, knock=-24.0, knock_hz=900.0, case=(300.0, 4000.0, 30, 0.06, 0.015), damp=0.06,
        release=-30.0),
    'dulcimer': dict(
        _GRAND, seed=28, B=((50, 3e-5), (86, 1.5e-4)), strings=((200, 2),), detune=0.7,
        loss=(((50, 0.35), (62, 0.45), (86, 0.9)), 1.0e-3, 7e-8), bridge=((50, 0.12), (62, 0.18), (86, 0.4)),
        bridge_fc=4000.0, bridge_h=0.15, pol=(0.15, 0.5),
        # a light wooden hammer on thin steel: hard, in contact for a fraction of a millisecond
        h_mass=0.6e-3, h_stiff=2.0e9, h_exp=1.7, h_hyst=0.03, rough=0.02, z=0.5, strike=0.1, speed=(0.5, 4.0, 1.3),
        damp=0.08, damp_top=200.0, damp_at=None, hp=130.0, flat=400.0, lp=6000.0,
        modes=((165, 25, 2.0), (240, 28, 1.8), (330, 30, 1.6), (420, 30, 1.5), (540, 30, 1.3)),
        cloud=(620.0, 9000.0, 40.0, 0.035, 35.0, 20.0, 0.5), pan=0.12, sym=0.0, knock=-16.0, knock_hz=1500.0,
        case=(300.0, 3500.0, 24, 0.05, 0.012), release=-44.0, knock_vel=8.0, lp_order=1, diffuse=0.0),
    'banjo': dict(
        _PLUCK, seed=29, open=(146.83, 196.0, 246.94, 293.66), B=4e-5, loss=(0.8, 2.0e-3, 1.0e-7), stop_loss=0.4,
        bridge=4.5, bridge_fc=5000.0,               # the bridge stands on a membrane: the string empties fast
        pluck=0.1, fc=5500.0, width=250.0, noise=0.08, angle=24.0, hp=220.0, flat=600.0, lp=6000.0,
        modes=((290, 8, 2.5), (462, 10, 2.5), (620, 12, 2.2), (666, 12, 1.8), (770, 14, 2.0), (846, 14, 1.8),
               (915, 14, 1.8), (1015, 15, 1.6), (1045, 15, 1.6)),
        cloud=(1100.0, 9000.0, 40.0, 0.03, 16.0, 14.0, 0.8), knock=-22.0, knock_hz=700.0, damp=0.05),
    # a struck steel tine with its tonebar in front of a magnetic pickup; a plucked reed at a capacitive one
    'electric_piano': dict(
        kind='tine', seed=31, extra=('tremolo',), t60=(16.0, 0.7), over=(7.11, 20.25), over_t60=(0.5, 0.12),
        bar=(5.6, 0.1),
        contact=((28, 4.5e-3), (64, 1.8e-3), (100, 0.6e-3)),
        reach=((28, 1.1), (52, 1.0), (64, 0.9), (76, 0.55), (88, 0.3), (100, 0.2)),
        offset=0.3, pickup='magnetic', pickup_lp=5200.0, drive=0.5, cab=(70.0, 5500.0, (2400.0, 2.5)), damp=0.1,
        stereo=0.6, tremolo_hz=5.2),
    'reed_piano': dict(
        kind='tine', seed=32, extra=('tremolo',), t60=(5.0, 0.6), over=(5.8, 16.0), over_t60=(0.25, 0.06), bar=None,
        contact=((33, 5.0e-3), (64, 3.0e-3), (96, 1.5e-3)), reach=((33, 2.2), (64, 1.5), (96, 0.8)),
        offset=0.3, pickup='electrostatic', pickup_lp=4200.0, drive=0.8, cab=(90.0, 4800.0, (1800.0, 3.0)),
        damp=0.07, stereo=0.6, tremolo_hz=5.8),
}
_BLOCK = 256                                         # samples per block of the mode bank
_LONGEST = 24.0                                      # seconds: no note is rendered longer than this
_SPREAD = {1: (0.0,), 2: (-0.7, 0.7), 3: (-1.1, 0.1, 1.0)}        # unison mistuning pattern, in units of `detune`


# ------------------------------------------------------------------------------------------------------ the mode bank
def _ring(out, lam, amp, start=0, stop=None, block=_BLOCK, floor=3e-6):
    """Add sum_k Re(amp[k, c] exp(lam[k] t)), t = 0, 1, ..., into out[start:stop]: exact damped sinusoids (lam is
    the complex frequency per sample, its real part the decay). exp(lam (q B + m)) = exp(lam q B) exp(lam m), so
    the sum over all modes and a block of B samples is a pair of real matrix products; a mode is computed only until
    it is `floor` under the loudest one, so the many short-lived high partials of a string cost almost nothing."""
    stop = len(out) if stop is None else min(stop, len(out))
    n = stop - start
    if n <= 0 or not len(lam):
        return
    mag = np.abs(amp).max(axis=1)
    top = mag.max()
    keep = mag > top * floor
    if top <= 0 or not keep.any():
        return
    lam, amp, mag = lam[keep], amp[keep], mag[keep]
    life = np.minimum(np.log(mag / (top * floor)) / np.maximum(-lam.real, 1e-12), n)
    blocks = int(np.ceil(n / block))
    q = np.clip(np.ceil(life / block), 1, blocks)
    q = np.minimum(blocks, 2 ** np.ceil(np.log2(q))).astype(int)        # a few groups of equal length
    m = np.arange(block)
    for qg in np.unique(q):
        i = q == qg
        w = np.exp(np.outer(lam[i], m))                                 # (modes, block): inside a block
        e = np.exp(np.outer(np.arange(qg) * block, lam[i]))             # (blocks, modes): from block to block
        cw = (amp[i][:, :, None] * w[:, None, :]).reshape(len(w), -1)
        y = np.ascontiguousarray(e.real) @ np.ascontiguousarray(cw.real) \
            - np.ascontiguousarray(e.imag) @ np.ascontiguousarray(cw.imag)
        y = y.reshape(qg, amp.shape[1], block).transpose(0, 2, 1).reshape(qg * block, -1)
        k = min(n, len(y))
        out[start:start + k] += y[:k]


def _drive(lam, res, e):
    """What a short pulse e (P samples) does to modes whose impulse response is sum Re(res exp(lam t)). From sample
    P on the response is the same sum with every residue times E = sum_m e[m] exp(-lam m), the pulse's transform at
    that mode's complex frequency: a longer contact leaves less in the high modes. Before that it is the plain
    convolution of the pulse with the modes' first P samples, returned as head (P, channels): the attack is the
    pulse's own rise, with no fade-in and no click."""
    w = np.exp(np.outer(lam, np.arange(len(e))))
    head = np.stack([np.convolve(g, e)[:len(e)] for g in (res.T @ w).real], axis=1)
    return (1.0 / w) @ e, head


def _through(lam_s, a_s, lam_b, r_b, pan_b, const, pan_c, jitter=0.0):
    """A string group (modes lam_s, residues a_s) heard through a body (modes lam_b with residues r_b, plus a
    constant gain const): the partial fractions of the cascade. Each string mode takes the body's response at its
    own complex frequency; each body mode rings with what the strings deliver at its frequency (the knock of the
    attack). Stereo: every body mode radiates a little more to one microphone than to the other (pan_b, -1..1;
    pan_c for the constant). A string mode then differs between the channels as the body modes that carry it do
    (plus jitter, the as-built difference of the two microphone paths); the difference is held under 0.6 of the
    common part, so no partial can cancel in the mono sum. Returns residues (Ms, 2) and (Mb, 2)."""
    zs, zb = np.exp(lam_s)[:, None], np.exp(lam_b)[None, :]
    g1, g2 = 1.0 / (1.0 - zb / zs), 1.0 / (1.0 - np.conj(zb) / zs)
    each = 0.5 * (g1 * r_b + g2 * np.conj(r_b))                    # every body mode's response at every string mode
    mid = const + each.sum(axis=1)
    side = (const * pan_c + each @ pan_b) / mid + jitter               # the side signal over the mid, per string mode
    side = side * np.minimum(1.0, 0.6 / (np.abs(side) + 1e-12))
    at_b = 0.5 * ((1.0 - g1).T @ a_s + np.conj(1.0 - g2).T @ np.conj(a_s))
    return ((a_s * mid)[:, None] * np.stack([1.0 - side, 1.0 + side], axis=1),
            (r_b * at_b)[:, None] * np.stack([1.0 - pan_b, 1.0 + pan_b], axis=1))


def _level(lam, res, t):
    """the level of a set of modes at times t (s): the root of their summed power (phases taken as random)"""
    return np.sqrt(np.exp(2.0 * SR * np.outer(t, lam.real)) @ (np.abs(res) ** 2).sum(axis=1))


def _fall(lam, res, floor, longest):
    """seconds until the modes' level is under `floor`, at most `longest`"""
    t = np.geomspace(0.005, longest, 240)
    under = np.nonzero(_level(lam, res, t) < floor)[0]
    return float(t[under[0]]) if len(under) else float(longest)


def _reg(tab, key, log=True):
    """a build parameter over the keyboard: a number, or ((key, value), ...) interpolated over the keys"""
    if np.isscalar(tab):
        return float(tab)
    k, v = zip(*tab)
    return float(np.exp(np.interp(key, k, np.log(v)))) if log else float(np.interp(key, k, v))


def _seed(sp, key):
    """the instrument as built does not change from take to take: its tuning and wood are drawn from this"""
    return 1000 * sp['seed'] + int(round(key))


# ---------------------------------------------------------------------------------------------------------- strings
def _unison(detune, loss, couple, hit):
    """Normal modes of N strings on one bridge (Weinreich 1977), per partial. Each string alone would move as
    exp(i (w + detune) t - loss t); the bridge they share adds `couple` to every element of the frequency matrix
    (imaginary part: the decay rate one string alone gets from the bridge; real part: the bridge's springiness or
    mass). In phase the strings push the bridge together and die N times faster; out of phase they hardly move it
    and ring on; mistuning and an uneven hammer (hit, one amplitude per string) mix the two: the prompt sound, the
    aftersound and the beats of a piano note. detune (K, N) rad/s, loss and couple (K,). Returns the complex
    frequencies (K, N) in 1/s, to be added to i w, and each mode's share of the summed bridge force (K, N)."""
    K, N = detune.shape
    if N == 1:
        return 1j * (detune + couple[:, None]) - loss[:, None], np.full((K, 1), hit[0] + 0j)
    mat = couple[:, None, None] * np.ones((N, N)) + detune[:, :, None] * np.eye(N)
    mu, vec = np.linalg.eig(mat)
    hit = np.broadcast_to(hit, (K, N))[:, :, None].astype(complex)
    share = vec.sum(axis=1) * np.linalg.solve(vec, hit)[:, :, 0]
    return 1j * mu - loss[:, None], share / N


@njit(cache=True)
def _past(f, x):
    if x <= 0.0:
        return 0.0
    j = int(x)
    return f[j] + (f[j + 1] - f[j]) * (x - j)


@njit(cache=True)
def _hammer_force(v0, m, K, p, hyst, Z, t1, t2, refl, fs, n):
    """Force between a felt hammer and the string it strikes, sample by sample at fs. The hammer (mass m kg, speed
    v0 m/s) compresses its felt by c metres: F = K c^p (1 + hyst dc/dt), a hardening spring that gives back less
    than it took. The string is an ideal one of wave impedance Z: it yields with velocity F / 2Z, and the waves the
    force sent out come back from the near end after t1 samples and from the far end after t2, inverted, while the
    hammer is still there. The contact ends when the string throws the hammer off. Each step solves the felt's
    compression implicitly (Newton), because the treble felt is too stiff for an explicit step."""
    F = np.zeros(n)
    dt = 1.0 / fs
    t0 = t1 + t2
    vh = v0
    c = 0.0
    for i in range(1, n):
        back = 0.0
        g = 1.0
        base = float(i)
        while base > 0.0:
            if base < i:
                back += 2.0 * g * _past(F, base)
            back -= g * refl * (_past(F, base - t1) + _past(F, base - t2))
            g *= refl * refl
            base -= t0
        free = c + dt * (vh - back / (2.0 * Z))          # where the felt would be without its own force
        f = 0.0
        x = free
        if free > 0.0:
            for _ in range(20):
                h = max(1.0 + hyst * (x - c) / dt, 0.0)
                xp = K * x ** p
                f = xp * h
                df = p * xp / x * h
                if h > 0.0:
                    df += xp * hyst / dt
                step = (x - free + dt * f / (2.0 * Z)) / (1.0 + dt * df / (2.0 * Z))
                x = max(x - step, 1e-3 * x)
                if abs(step) < 1e-13:
                    break
            f = K * x ** p * max(1.0 + hyst * (x - c) / dt, 0.0)
            x = free - dt * f / (2.0 * Z)
        F[i] = f
        vh -= f / m * dt
        c = x
        if c < -3e-4 and vh < 0.0:
            break
    return F


def _hammer(sp, key, f0, speed, hard, pos, r=None):
    """the hammer's force on one string at SR (N), from the simulated contact; pos is the striking point as a
    fraction of the string. hard scales the felt's stiffness by up to ten either way; with r the felt has its
    grain (a new one in every take)"""
    fs = 4 * SR
    F = _hammer_force(float(speed), _reg(sp['h_mass'], key), _reg(sp['h_stiff'], key) * 10.0 ** hard,
                      _reg(sp['h_exp'], key), float(sp['h_hyst']), _reg(sp['z'], key), pos / f0 * fs,
                      (1.0 - pos) / f0 * fs, 0.97, float(fs), int(0.0102 * fs))
    if r is not None:                               # felt is fibre: the force has a grain, up to a few kHz
        grain = signal.sosfilt(signal.butter(2, 5000.0, fs=fs, output='sos'), r.standard_normal(len(F)))
        F = F * (1.0 + sp['rough'] * min(1.0, (500.0 / f0) ** 2) * grain / 0.23)      # thin hard treble felt: none
    F = signal.resample_poly(np.concatenate([np.zeros(48), F]), 1, 4)
    last = int(np.nonzero(np.abs(F) > 1e-4 * np.abs(F).max())[0][-1])
    return F[:min(len(F), last + 4)]


def contact_ms(instrument, pitch, vel=0.7):
    """how long the hammer of a struck instrument stays on the string (ms, between the points where the force
    passes 5 % of its peak): shorter for a faster hammer and towards the treble"""
    sp, f0 = _SPEC[instrument], _pitch(pitch)
    key = 69.0 + 12.0 * np.log2(f0 / 440.0)
    lo, hi, curve = sp['speed']
    F = _hammer(sp, key, f0, lo + (hi - lo) * vel ** curve, 0.0, _reg(sp['strike'], key))
    on = np.nonzero(F > 0.05 * F.max())[0]
    return float((on[-1] - on[0] + 1) / SR * 1e3)


def _release(fc, noise, r, n=_BLOCK):
    """The moment a plucked string leaves the finger, as the pulse the string and body are driven with (unit area).
    The string slips off over a fraction of a millisecond: a pulse (t / tau^2) exp(-t / tau), whose transform falls
    12 dB per octave above fc = 1 / (2 pi tau) with no notches; flesh is slow (about 1 kHz), a nail or plectrum fast
    (3-6 kHz). Before and while it slips, the string scrapes along the finger: a little high-passed noise (its energy
    `noise` relative to the pulse), new in every take."""
    t = np.arange(n) / SR - 0.002
    tau = 1.0 / (2.0 * np.pi * fc)
    u = np.maximum(t, 0.0)
    e = u / tau ** 2 * np.exp(-u / tau) / SR
    e[-24:] *= np.cos(np.linspace(0.0, np.pi / 2, 24)) ** 2
    if noise > 0:
        w = np.diff(r.standard_normal(n), prepend=0.0)
        w = w * np.sin(np.pi * np.minimum(np.arange(n) / (0.004 * SR), 1.0)) ** 2
        e += noise * w / np.sqrt(np.sum(w ** 2))
    e[0] = 0.0
    return e


def _radiation(sp, scale, key):
    """The smooth part of how a body turns bridge force into sound, as a filter (b, a). A plate driven at one
    point moves with a velocity that is, on average, flat over frequency, and sound pressure follows its
    acceleration while the body is smaller than the wavelength: pressure per force rises 6 dB per octave up to
    `flat` Hz and stays level above it; below the lowest body mode (`hp`) the body is a stiff spring and radiates
    little; above `lp` the bridge is too heavy to follow (6 dB per octave; 12 on a piano's ribbed board, lp_order).
    The filter is that curve divided by frequency, because the excitation it meets is the rate of change of the
    force: a second-order high-pass at hp, a first-order low-pass at flat and the low-pass at lp. hp and flat are
    numbers, or tables over the keys where the board answers differently along the bridge. scale stretches the
    body."""
    sos = np.vstack([signal.butter(2, _reg(sp['hp'], key) * scale, 'high', fs=SR, output='sos'),
                     signal.butter(1, _reg(sp['flat'], key) * scale, fs=SR, output='sos'),
                     signal.butter(sp['lp_order'], min(sp['lp'], 0.4 * SR), fs=SR, output='sos')])
    return signal.sos2tf(sos)


def _body(sp, key, scale, c, pan_note):
    """The body as built: modes (complex frequency per sample), their residues, their place between the two
    microphones (-1..1), a constant gain, the smooth radiation filter the modes stand on, and how many of the
    modes are the body's own (the rest are a piano's undamped strings). First come the poles of the smooth
    radiation filter (_radiation), so the smooth path is causal and has a rise time like everything else. Then a
    few named low modes (a guitar's air and top resonance, a violin's) and a cloud whose spacing is that of a plate
    (df Hz apart at the bottom, a fixed fraction of the frequency higher up), each with the Q of wood. A mode's
    residue puts its peak `prom` times the smooth curve, divided by the root of the modal overlap where modes
    crowd; a named mode adds in phase with the smooth path, so it is a peak of the response, the cloud's modes add
    with either sign (and, where `diffuse` is set, in any phase: the modes of a piano's board reach a close
    microphone from all over the board). Where on the bridge the string sits changes how much of each mode it
    drives (key). bright
    tilts the modes by 3 dB per octave around 700 Hz. The undamped strings above E6 of a piano are modes of the
    same kind: they ring on when a partial of the note passes close to them."""
    b = dsp.rng(_seed(sp, 0))
    num, den = _radiation(sp, scale, key)
    res, pole, const = signal.residuez(num, den)
    up = pole.imag >= 0
    lam = np.log(pole[up].astype(complex))
    res = np.where(pole[up].imag > 0, 2.0, 1.0) * res[up]
    pan = np.full(len(lam), pan_note)
    f_lo, f_hi, df, frac, q_lo, q_hi, prom_c = sp['cloud']
    f, q, prom = [m[0] for m in sp['modes']], [m[1] for m in sp['modes']], [m[2] for m in sp['modes']]
    x = f_lo
    while x < f_hi:
        f.append(x)
        q.append(np.exp(np.interp(np.log(x), [np.log(f_lo), np.log(f_hi)], [np.log(q_lo), np.log(q_hi)])))
        prom.append(prom_c)
        x += max(df, frac * x) * b.uniform(0.6, 1.4)
    f, q, prom = np.array(f) * scale, np.array(q), np.array(prom)
    rate = np.pi * f / q
    overlap = rate / np.pi / np.gradient(f)
    smooth = signal.freqz(num, den, worN=f, fs=SR)[1]                  # complex: a mode adds in phase with it
    amp = 2.0 * rate / SR * prom * smooth / np.sqrt(1.0 + overlap) * (f / 700.0) ** (0.5 * c['bright'])
    place = np.cos(b.uniform(0, 2 * np.pi, len(f)) + key * sp['bridge_k'] * np.sqrt(f / 200.0))
    named = np.arange(len(f)) < len(sp['modes'])                       # a named mode is a peak, the cloud either
    amp = amp * np.where(named, 1.0, np.sign(place)) * (0.35 + 0.65 * np.abs(place))
    if sp['diffuse'] > 0:                           # a large board heard from a hand's breadth: every mode of the
        turn = sp['diffuse'] * b.uniform(-0.5 * np.pi, 0.5 * np.pi, len(f))       # cloud arrives in its own phase,
        amp = amp * np.where(named, 1.0, np.exp(1j * turn))                        # so the attack is no single spike
    lam = np.concatenate([lam, (-rate + 2j * np.pi * f) / SR])
    res = np.concatenate([res, amp])
    pan = np.concatenate([pan, b.uniform(-1, 1, len(f)) * sp['spread']])
    if sp['sym'] > 0:                               # the undamped top strings of a piano answer through the bridge
        keys = np.arange(89, 109)
        fs = 440.0 * 2.0 ** ((keys - 69) / 12.0)
        rs = np.array([_reg(sp['loss'][0], k) + _reg(sp['bridge'], k) for k in keys])
        lam = np.concatenate([lam, (-rs + 2j * np.pi * fs) / SR])
        other = np.abs(keys - key) > 0.5                               # the note's own strings are not among them
        res = np.concatenate([res, 2.0 * rs / SR * sp['sym'] * other * signal.freqz(num, den, worN=fs, fs=SR)[1]])
        pan = np.concatenate([pan, b.uniform(-1, 1, len(keys)) * sp['spread']])
    pan = np.clip(pan * (1.0 + c['width']), -0.95, 0.95)
    return lam, res.astype(complex), pan, float(const[0].real), (num, den), len(lam) - (20 if sp['sym'] > 0 else 0)


def _power(lam, res):
    """the energy of a set of ringing modes (each |residue|^2 / (2 decay))"""
    return float(np.sum((np.abs(res) ** 2).sum(axis=1) / (-2.0 * lam.real)))


def _knock(sp, lam_b, r_b, pan_b, const, n, wide):
    """The thump that reaches the body without going through the string (the hammer's blow through the action and
    keybed, a finger landing): a soft pulse into the body's modes and into the short wooden modes of the case."""
    e = np.diff(_release(sp['knock_hz'], 0.0, None), prepend=0.0)
    lam, res = lam_b, r_b[:, None] * np.stack([1.0 - pan_b, 1.0 + pan_b], axis=1)
    if sp['case']:
        b = dsp.rng(_seed(sp, 1))
        f_lo, f_hi, count, t_lo, t_hi = sp['case']
        f = np.sort(np.exp(b.uniform(np.log(f_lo), np.log(f_hi), count)))
        rate = 6.91 / np.exp(np.interp(np.log(f), [np.log(f_lo), np.log(f_hi)], [np.log(t_lo), np.log(t_hi)]))
        lam_c = (-rate + 2j * np.pi * f) / SR
        pan = b.uniform(-0.6, 0.6, count) * min(wide, 1.5)
        r_c = ((b.choice([-1.0, 1.0], count) * b.uniform(0.4, 1.0, count))[:, None]
               * np.stack([1 - pan, 1 + pan], axis=1)).astype(complex)
        E_b, E_c = _drive(lam, res, e)[0], _drive(lam_c, r_c, e)[0]
        # as much energy in the wood of the case as in the board
        r_c = r_c * np.sqrt(_power(lam, res * E_b[:, None]) / _power(lam_c, r_c * E_c[:, None]))
        lam, res = np.concatenate([lam, lam_c]), np.concatenate([res, r_c])
    E, head = _drive(lam, res, e)
    out = np.zeros((n, 2))
    out[:len(e)] = (head + const * e[:, None])[:n]
    _ring(out, lam, res * (E * np.exp(lam * len(e)))[:, None], start=len(e))
    return out


def _burst(r, seconds, lo, hi, wide):
    """a soft noise of felt on wire or skin on string: band-passed, under a raised-cosine envelope"""
    n = dsp.n_of(seconds)
    w = dsp.butter(r.standard_normal(n + 64), 'band', (lo, hi), order=2)[64:]
    w = w * np.sin(np.pi * np.arange(n) / n) ** 2
    pan = r.uniform(-0.3, 0.3) * min(wide, 1.5)
    return np.stack([w * (1 - pan), w * (1 + pan)], axis=1)


def _mix(y, x, at, level):
    """add x into y at sample `at`, its peak at `level` (linear)"""
    k = min(len(x), len(y) - at)
    if k > 0:
        y[at:at + k] += x[:k] * (level / (np.abs(x).max() + 1e-12))


def _finish(y, vel, fade, vary, r):
    """the level law (a note peaks at 0.9 vel) and a faded end"""
    y = y * (0.9 * vel * 10.0 ** (0.05 * vary * r.uniform(-1, 0)) / (np.abs(y).max() + 1e-12))
    k = int(min(len(y), max(fade, 64)))
    y[-k:] *= (np.cos(np.linspace(0.0, np.pi / 2, k)) ** 2)[:, None]
    return y


def _string_note(sp, f0, vel, dur, r, c):
    """A note of a struck or plucked string instrument (sp: its build, c: the controls). In order: the modes of the
    string group and how the bridge couples and drains them; the body they are heard through; the excitation pulse
    and what it leaves in every mode; how long the note rings and what the damper does to it at dur; the bank of
    damped sinusoids; and the sounds that reach the microphones without passing through the string."""
    key = 69.0 + 12.0 * np.log2(f0 / 440.0)
    built = dsp.rng(_seed(sp, key))               # this key's strings as built: tuning, bridge, voicing
    vary, hammer = c['vary'], sp['kind'] == 'hammer'
    scale = 2.0 ** (-0.35 * c['size']) * (_reg(sp['size_by'], key) if sp.get('size_by') else 1.0)

    # the string(s): stretched partials, each string of the unison a few tenths of a cent off
    fret = 1.0
    if sp['open']:
        op = np.asarray(sp['open'])
        fret = max(1.0, f0 / op[max(0, int(np.searchsorted(op, f0 * 1.0005)) - 1)])
    B = _reg(sp['B'], key) * fret ** 2
    k = np.arange(1, 401)
    fk = f0 * k * np.sqrt((1.0 + B * k * k) / (1.0 + B))
    k, fk = k[fk < 0.45 * SR], fk[fk < 0.45 * SR]
    K, N = len(k), next(count for top, count in sp['strings'] if key <= top)
    # No unison is perfect and none of a tuned piano beats fast: a tuner works by beats, so the spread is in Hz (at
    # the fundamental; partial k is k times as far off), the same pattern dealt differently to every key.
    off = _reg(sp['detune'], key) / f0 * (built.permutation(_SPREAD[N]) * built.choice([-1.0, 1.0])
                                         + 0.3 * built.normal(size=N))
    stiff = built.normal(0, 0.012, N)
    detune = 2 * np.pi * fk[:, None] * (off - off.mean() + (0.5 * B * k * k / (1 + B * k * k))[:, None]
                                         * (stiff - stiff.mean()))
    a, b1, b2 = sp['loss']
    longer = 2.0 ** (-1.3 * c['ring'])
    loss = (_reg(a, key) + (sp['stop_loss'] if fret > 1.01 else 0.0) + _reg(b1, key) * fk
            + _reg(b2, key) * fk ** 2) * longer
    # the body: what a partial gives to the bridge is what the body radiates of it, so a partial on a body
    # resonance is louder and shorter (a guitar's wolf note), one in a gap quieter and longer
    pan_c = float(np.clip(sp['pan'] * (key - 60.0) / 48.0 * (1.0 + c['width']), -0.9, 0.9))
    lam_b, r_b, pan_b, const, smooth, own = _body(sp, key, scale, c, pan_c)
    zk = np.exp(2j * np.pi * fk / SR)[:, None]
    zb = np.exp(lam_b)[None, :]
    resp = const + 0.5 * ((1.0 / (1.0 - zb / zk)) @ r_b + (1.0 / (1.0 - np.conj(zb) / zk)) @ np.conj(r_b))
    carried = np.clip(np.abs(resp / signal.freqz(*smooth, worN=fk, fs=SR)[1]) ** 2, 0.6, 1.8)
    bridge = _reg(sp['bridge'], key) * carried / (1.0 + (fk / sp['bridge_fc']) ** 2) * longer
    couple_v = bridge * (1j + built.uniform(-0.5, 0.5, K))
    couple_h = sp['bridge_h'] * bridge * (1j + built.uniform(-0.5, 0.5, K))
    split = 2 * np.pi * 0.05 * np.sqrt(k) * built.normal(0, 1, K)       # the two polarisations never quite agree
    if hammer:
        hit_v = 1.0 + sp['uneven'] * built.normal(size=N) + 0.06 * vary * r.normal(size=N)
        hit_h = sp['pol'][0] * built.choice([-1.0, 1.0]) * (1.0 + 0.4 * built.normal(size=N)
                                                            + 0.4 * vary * r.normal(size=N))
        pos = _reg(sp['strike'], key)
    else:
        angle = np.radians(sp['angle'] + 12.0 * vary * r.normal())
        hit_v, hit_h = np.full(N, np.cos(angle)), np.full(N, np.sin(angle))
        pos = _reg(sp['pluck'], key) * fret
    pos = float(np.clip(pos * 2.0 ** (1.2 * c['position']) * (1.0 + 0.06 * vary * r.normal()), 0.02, 0.5))
    comb = np.sin(np.pi * k * pos)                                     # a mode is not driven at its own node,
    comb = np.where(np.abs(comb) < 0.03, 0.03, comb) * (-1.0) ** (k + 1)           # but no node is perfect
    if not hammer:
        comb = comb / (1.0 + (k / (sp['width'] * 2.0 ** c['hard'])) ** 2)          # the finger has a width
    # the bridge's reactance would shift the pitch of the mode that carries the note: tuned out, as a tuner does
    s_v, g_v = _unison(detune - N * couple_v.real[:, None], loss, couple_v, hit_v)
    s_h, g_h = _unison(detune + (split - N * couple_h.real)[:, None], loss, couple_h, hit_h)
    lam_s = ((np.concatenate([s_v, s_h], axis=1) + 2j * np.pi * fk[:, None]) / SR).ravel()
    a_s = -1j * comb[:, None] * np.concatenate([g_v, sp['pol'][1] * g_h], axis=1)
    a_s = (a_s * ((fk / 700.0) ** (0.5 * c['bright']))[:, None]).ravel()

    # the cascade strings -> body
    jitter = 0.4 * sp['spread'] * (1.0 + c['width']) * (built.uniform(-1, 1, K) + 1j * built.uniform(-1, 1, K))
    jitter = np.repeat(jitter, 2 * N)
    res_s, res_b = _through(lam_s, a_s, lam_b, r_b, pan_b, const, pan_c, jitter)

    # the excitation
    if hammer:
        lo, hi, curve = sp['speed']
        speed = (lo + (hi - lo) * vel ** curve) * (1.0 + 0.05 * vary * r.normal())
        e = np.diff(_hammer(sp, key, f0, speed, c['hard'], pos, r), prepend=0.0)
    else:
        fc = _reg(sp['fc'], key) * (0.3 + 1.4 * vel) ** sp['fc_vel'] * 2.0 ** (1.5 * c['hard'])
        e = _release(fc * (1.0 + 0.1 * vary * r.normal()), sp['noise'], r)
    P = len(e)
    block = _BLOCK if P <= _BLOCK else 2 * _BLOCK
    lam = np.concatenate([lam_s, lam_b])
    E, head = _drive(lam, np.concatenate([res_s, res_b]), e)
    E = E * np.exp(lam * P)                                            # residues as of sample P
    E_s, E_b = E[:len(lam_s)], E[len(lam_s):]
    res_s, res_b = res_s * E_s[:, None], res_b * E_b[:, None]

    # how long it rings, and the damper
    t20 = np.array([0.02])
    ref = float(_level(lam_s, res_s, t20)[0] + _level(lam_b, res_b, t20)[0])
    n = int(SR * _fall(lam, np.concatenate([res_s, res_b]), ref * 1e-3, _LONGEST))
    n_d, fade = None, int(0.2 * n)
    if dur is not None and key <= sp['damp_top'] and dur * SR < n - fade:
        n_d = max(P + 1, int(round(dur * SR)))
        state = a_s * E_s * np.exp(lam_s * (n_d - P))                  # each string mode when the damper lands
        heard = np.abs(res_s).max(axis=1) * np.exp(lam_s.real * (n_d - P))
        on = heard > 1e-6 * heard.max()
        kk = np.repeat(k, 2 * N)[on]
        grip = 1.0 if sp['damp_at'] is None else 0.15 + 0.85 * 0.5 * (
            1.0 - np.cos(2 * np.pi * kk * sp['damp_at']) * np.sinc(kk * 0.05))     # weak at its own nodes
        lam_d = lam_s[on] - 6.91 / _reg(sp['damp'], key) * grip / SR
        res_d, at_new = _through(lam_d, state[on], lam_b, r_b, pan_b, const, pan_c, jitter[on])
        _, at_old = _through(lam_s[on], state[on], lam_b, r_b, pan_b, const, pan_c, jitter[on])
        tail = _fall(np.concatenate([lam_d, lam_b]), np.concatenate([res_d, at_new - at_old]), ref * 1e-3, 6.0)
        n, fade = n_d + int(SR * tail), int(0.3 * SR * tail)

    y = np.zeros((n, 2))
    y[:P] = head[:n]
    _ring(y, lam_s, res_s, start=P, stop=n_d, block=block)
    _ring(y, lam_b, res_b, start=P, block=block)
    if n_d is not None:
        _ring(y, lam_d, res_d, start=n_d, block=block)
        _ring(y, lam_b, at_new - at_old, start=n_d, block=block)

    # what reaches the microphones besides the string: knock, felt, the damper landing
    peak = np.abs(y).max()
    if sp['knock'] is not None:
        level = _reg(sp['knock'], key, log=False) - sp['knock_vel'] * np.log10(vel / 0.5)
        knock = _knock(sp, lam_b[:own], r_b[:own], pan_b[:own], const, min(n, int(0.6 * SR)), 1.0 + c['width'])
        _mix(y, knock, 0, peak * 10.0 ** (level / 20.0))
    if sp['felt']:
        db, lo, hi = sp['felt']
        _mix(y, _burst(r, 0.012, lo, hi, 1.0 + c['width']), 0,
             peak * 10.0 ** ((db - sp['knock_vel'] * np.log10(vel / 0.5)) / 20.0))
    if n_d is not None:
        _mix(y, _burst(r, 0.03, 120.0, 1600.0, 1.0 + c['width']), n_d, peak * 10.0 ** (sp['release'] / 20.0))
    return _finish(y, vel, fade, vary, r)


# -------------------------------------------------------------------------------------------------- electric pianos
def _tine_note(sp, f0, vel, dur, r, c):
    """A tine electric piano note: a steel tine (a clamped rod, struck near its root by a rubber-tipped hammer) and
    the tonebar it is mounted on, read by a magnetic pickup; or a reed plucked by a pad at a capacitive pickup.
    The tine's overtones are those of a clamped bar (measured 7.11 and 20.25 times the fundamental), far from
    harmonic and gone in a fraction of a second: the bell of the attack. The pickup's flux depends on where the tine
    tip is, as 1 / (1 + u^2) with u the tip's distance from the pickup axis over the gap, and its voltage is the
    rate of change of that flux: a small swing is a near-sine, a large one sweeps the curve and barks. The tip's
    motion is computed at twice the sample rate, so the harmonics the pickup and the amplifier add do not alias."""
    key = 69.0 + 12.0 * np.log2(f0 / 440.0)
    built = dsp.rng(_seed(sp, key))
    vary = c['vary']
    os_ = 2
    fs = os_ * SR
    magnetic = sp['pickup'] == 'magnetic'
    longer = 2.0 ** (1.3 * c['ring'])
    t1 = min(22.0, sp['t60'][0] * (f0 / 100.0) ** -sp['t60'][1]) * longer
    t2 = sp['over_t60'][0] * (f0 / 100.0) ** -0.5 * longer
    # modes of the tip's displacement: frequency over f0, T60, weight
    modes = [(1.0, t1, 1.0), (1.0 + built.normal(0, 1.2e-3), 0.6 * t1, 0.12 * built.choice([-1, 1])),
             (sp['over'][0] * (1 + 0.04 * built.uniform(-1, 1)), t2, -4.0 / sp['over'][0]),
             (sp['over'][1] * (1 + 0.05 * built.uniform(-1, 1)), sp['over_t60'][1] / sp['over_t60'][0] * t2,
              4.0 / sp['over'][1]),
             (built.uniform(45.0, 65.0) / f0, 0.12, 0.05)]                         # the whole assembly on its mounts
    if sp['bar']:
        modes.append((sp['bar'][0] * (1 + 0.05 * built.uniform(-1, 1)), 1.6 * t2, sp['bar'][1]))
    ratio, t60, weight = (np.array(v, float) for v in zip(*modes))
    ok = ratio * f0 < 0.45 * SR
    ratio, t60, weight = ratio[ok], t60[ok], weight[ok]
    lam = (-6.91 / t60 + 2j * np.pi * ratio * f0) / fs
    speed = (0.08 + 0.92 * vel ** 1.3) * (1.0 + 0.05 * vary * r.normal())          # hammer speed over fortissimo
    tau = _reg(sp['contact'], key) * speed ** -0.25 * 2.0 ** (-0.8 * c['hard'])
    P = max(8, int(round(tau * fs)))
    e = np.sin(np.pi * np.arange(P) / P) ** 2
    if magnetic:                                                       # struck: a force pulse, velocity starts at 0
        res = (-1j * weight)[:, None]
        held = np.zeros(P)
    else:                                                              # plucked: let go from where the pad held it
        e /= e.sum()
        res = (weight / ratio ** 2).astype(complex)[:, None]
        held = res.real.sum() * (1.0 - np.cumsum(e))
    E, head = _drive(lam, res, e)
    res = res * (E * np.exp(lam * P))[:, None]
    gain = _reg(sp['reach'], key) * speed / (np.abs(res[0, 0]) + 1e-30)            # the swing of the tip over the gap
    res, head, held = res * gain, head[:, 0] * gain, held * gain

    ref = float(_level(lam, res, np.array([0.02 * os_]))[0])
    n = int(fs * _fall(lam, res, ref * 1e-3, _LONGEST * os_) / os_)
    n_d, fade = None, int(0.2 * n / os_)
    if dur is not None and dur * fs < 0.8 * n:
        n_d = max(P + 1, int(round(dur * fs)))
        state = res * np.exp(lam * (n_d - P))[:, None]
        lam_d = lam - 6.91 / (sp['damp'] * (f0 / 262.0) ** -0.3) / fs
        tail = _fall(lam_d, state, ref * 1e-3, 3.0 * os_) / os_
        n, fade = n_d + int(fs * tail), int(0.3 * SR * tail)
    x = np.zeros((n, 1))
    x[:P, 0] = (head + held)[:n]
    _ring(x, lam, res, start=P, stop=n_d, block=2 * _BLOCK)
    if n_d is not None:
        _ring(x, lam_d, state, start=n_d, block=2 * _BLOCK)

    offset = sp['offset'] * 2.0 ** (1.1 * c['position']) * (1.0 + 0.1 * built.uniform(-1, 1))
    u = offset + x[:, 0]
    if magnetic:
        y = np.gradient(1.0 / (1.0 + u * u)) * fs / (2 * np.pi * f0)
    else:       # the reed's overlap with the pickup plate saturates either way; the preamp forgets the held position
        y = np.tanh(1.5 * u) - np.tanh(1.5 * (offset + held[0] * np.exp(-np.arange(n) / (0.004 * fs))))
    lp = min(0.45 * SR, sp['pickup_lp'] * 2.0 ** (0.8 * c['bright']))
    y = signal.sosfilt(signal.butter(1, lp, fs=fs, output='sos'), y)
    drive = sp['drive'] * (0.4 + vel) / (np.abs(y).max() + 1e-12)                  # the amplifier's soft ceiling
    y = signal.resample_poly(np.tanh(drive * y) / drive, 1, os_)
    lo, hi, (pf, pg) = sp['cab']
    y = dsp.butter(dsp.butter(y, 'high', lo * 2.0 ** (-0.5 * c['size']), order=1), 'low',
                   hi * 2.0 ** (0.5 * c['bright']), order=2)
    y = dsp.peq(y, pf, pg, 0.8)
    y = fx.decorrelate(y, float(np.clip(sp['stereo'] * (1.0 + c['width']), 0.0, 0.9)))
    if c['tremolo'] > 0:                                               # the two speakers trade the note at 4-6 Hz
        lfo = 0.5 * c['tremolo'] * np.sin(2 * np.pi * sp['tremolo_hz'] * np.arange(len(y)) / SR)
        y = y * np.stack([1.0 - 0.5 * c['tremolo'] + lfo, 1.0 - 0.5 * c['tremolo'] - lfo], axis=1)
    return _finish(y, vel, fade, vary, r)


# ------------------------------------------------------------------------------------------------------------- API
def _pitch(pitch):
    """Hz of a note name ('C4'), a MIDI number (an int from 0 to 127) or a frequency (any other number, '440hz')"""
    if isinstance(pitch, (int, np.integer)) and not isinstance(pitch, bool) and 0 <= pitch <= 127:
        return 440.0 * 2.0 ** ((int(pitch) - 69) / 12.0)
    return _hz(pitch)


def _controls(instrument, controls):
    valid = CONTROLS + _SPEC[instrument].get('extra', ())
    bad = sorted(set(controls) - set(valid))
    if bad:
        raise ValueError(f'{instrument} has no control {", ".join(bad)}: the controls are {", ".join(valid)}')
    c = dict.fromkeys(valid, 0.0)
    for name, v in controls.items():
        if v is not None:
            c[name] = float(np.clip(v, 0.0 if name in ('vary', 'tremolo') else -1.0, 1.0))
    return c


def note(instrument, pitch=None, vel=0.7, dur=None, r=None, **controls):
    """One note: stereo float64 (n, 2) at 48 kHz, starting at its attack.

    instrument: a name in INSTRUMENTS. pitch: a note name ('C4'), a MIDI number (an int from 0 to 127) or Hz (a
    float, or '440hz'); None is the instrument's default pitch; it must lie in the instrument's range.
    vel 0..1: how hard it is played. It sets the hammer's speed or the pluck's, so a louder note is also brighter,
    and the level: the note peaks at 0.9 * vel (nothing is normalised).
    dur: None lets the note ring to its natural end (60 dB down, at most 24 s); a number of seconds is when the
    damper, the hand or the finger stops the string (piano notes above E6 have no damper and ring on).
    r: a numpy Generator (default dsp.rng(0)); the same arguments give the same samples.
    controls, each -1..1 with 0 the instrument as built: bright (tilt of the radiated sound), hard (hammer felt or
    finger: harder is a shorter contact, more high partials), ring (-1 damped .. 1 long), size (of the body: 1 is
    larger and lower), position (strike or pluck point: 1 towards the middle of the string, round and hollow; -1 at
    the bridge, thin and nasal; on electric pianos the tine's rest position against the pickup), width (stereo: -1
    mono), and vary 0..1 (take-to-take variation of touch; the pitch never moves). The electric pianos also take
    tremolo 0..1, the depth of the stereo tremolo."""
    if instrument not in _SPEC:
        raise ValueError(f'unknown instrument {instrument!r}: one of {", ".join(INSTRUMENTS)}')
    info, sp = INSTRUMENTS[instrument], _SPEC[instrument]
    c = _controls(instrument, controls)
    f0 = _pitch(info['default_pitch'] if pitch is None else pitch)
    lo, hi = (_hz(p) for p in info['range'])
    if not lo * 0.97 <= f0 <= hi * 1.03:
        raise ValueError(f'{instrument} plays {info["range"][0]} to {info["range"][1]} ({lo:.0f} to {hi:.0f} Hz), '
                         f'not {f0:.1f} Hz')
    vel = float(np.clip(vel, 0.02, 1.0))
    r = dsp.rng(0) if r is None else r
    make = _tine_note if sp['kind'] == 'tine' else _string_note
    return make(sp, f0, vel, None if dur is None else max(0.01, float(dur)), r, c)


def chord(instrument, pitches, vel=0.7, strum_s=0.0, dur=None, r=None, **controls):
    """Several notes as one gesture: stereo (n, 2), peaking at 0.9 * vel.

    pitches are played in the order given, spread over strum_s seconds (0.02-0.06 a strum, 0.15-0.5 a rolled chord;
    negative reverses the order). A hand does not cross strings evenly: the gaps shrink a little along the way and
    wander by a few percent, and with strum_s = 0 the notes still land within a few milliseconds of each other.
    Each note gets its own velocity (about 8 % spread, the last note of a roll a little stronger) and its own take.
    dur counts from the start of the gesture: all notes are stopped together."""
    r = dsp.rng(0) if r is None else r
    pitches = list(pitches)[::-1] if strum_s < 0 else list(pitches)
    if not pitches:
        raise ValueError('chord needs at least one pitch')
    count = len(pitches)
    at = abs(strum_s) * (np.arange(count) / max(1, count - 1)) ** 0.85
    at = np.maximum(0.0, at + r.normal(0, 0.0025 + 0.05 * abs(strum_s) / count, count))
    at -= at.min()
    notes = []
    for i, (p, t) in enumerate(zip(pitches, at)):
        v = vel * (1.0 + 0.08 * r.normal()) * (1.06 if strum_s and i == count - 1 else 1.0)
        take = dsp.rng(int(r.integers(2 ** 32)))
        notes.append(note(instrument, p, float(np.clip(v, 0.02, 1.0)),
                          None if dur is None else max(0.03, dur - t), take, **controls))
    out = np.zeros((max(int(t * SR) + len(y) for t, y in zip(at, notes)), 2))
    for t, y in zip(at, notes):
        out[int(t * SR):int(t * SR) + len(y)] += y
    return out * (0.9 * float(np.clip(vel, 0.02, 1.0)) / (np.abs(out).max() + 1e-12))
