"""Short music beds written as rules: a key, a tempo, a progression voiced by voice leading, patterns played with
a human feel, an arrangement over a few bars and a bed mix, as a seamless loop.

    from earshot.forge import dsp
    from earshot.synth import music
    plan = music.compose('warm', dsp.rng(3), key='D', bars=8, energy=0.2)   # the score: chords, voicings, notes
    y = music.render(plan, dsp.rng(3))                                      # (n, 2) float64: exactly 8 bars, a loop
    music.chart(plan)                                                       # 'D major, 84 bpm, 8 bars: Em9 | A11 ...'
    y = music.bed('playful', dsp.rng(1), bars=4, loop=False)                # once, ending on a chord that rings out

Three styles: `warm` (lo-fi electric piano over soft acoustic drums), `playful` (pizzicato, muted guitar, claps)
and `8bit` (chip channels). earshot makes sound effects; these are the beds its owner kept.

Generated music is recognised in two seconds. What gives it away, and the rule that answers it here:
  block chords in root position   chords come from a small grammar of common progressions in scale degrees and take
                                  a colour by mood (add9, sus2, 6/9, maj7, m7, m9, the eleventh as a soft dominant);
                                  every chord is voiced so the upper voices move least from the chord before (a
                                  shortest path through all voicings, closed over the loop point), nothing a minor
                                  ninth apart, no close interval under middle C; the bass has the register under
                                  125 Hz to itself and takes a first inversion where that makes its line move by step
  every note at one level         velocities follow the bar (downbeat, beats, eighths, sixteenths), the pattern's own
                                  accents and ghost notes, and a few per cent of touch; a velocity is timbre as well
                                  as level (each voice is played at the nearest of its velocity steps)
  mechanical timing               each role sits a few ms ahead of or behind the grid (hats ahead, the snare behind,
                                  keys late and rolled from the bottom up), drifts slowly and scatters a little; a
                                  style may swing its off-beats
  a melody that wanders           most beds have none; where there is a top line it is one motif of two to five
                                  notes of the pentatonic, drawn once, restated every two bars with another ending,
                                  its last note on a chord tone, and silent for the rest of the phrase
  no arrangement                  parts enter after a quarter or half of the loop and leave room; a fill ends each
                                  four-bar phrase; the last bar turns round on the dominant as a suspended chord
  instruments that fight          one register and one place per part, the low end mono, keys and bass ducked under
                                  the kick, each part set to its level by the RMS of its own loudest 400 ms, one
                                  shared room (earshot.synth.space), a dip around 2.5 kHz in a top line (where a
                                  voice-over sits), peaks far over the bed's level limited
  a cut at the loop point         the bed is rendered round a circle: notes, echoes, filters, the limiter and the
                                  room's tail run past the end onto the start, so the file repeats with nothing cut
                                  and nothing faded (a record crackle, which is noise, closes with a short crossfade)

Every pitch is a note of the key's scale (chords are built on the scale, motifs on its pentatonic). Nothing is
quoted: progressions are common practice and motifs are drawn from the seed.

Cost: every distinct (voice, pitch, velocity step, length, take) is rendered once and placed as often as it is
played, so a bar that comes back costs nothing. On one thread of a laptop (low priority, a machine shared
with other jobs: times vary by half) a style at its own length of 15-23 s takes 1.5 s (chip) to about 10 s: a
piano note 0.06-0.2 s, a plucked or synthesized note 0.02-0.1 s, the room and the mix 2-3 s. A bed played once
is rendered with RING s of room after its last bar and costs up to half as much again.
"""
import itertools
import zlib

import numpy as np
from scipy.fft import next_fast_len

from ..forge import dsp, fx, knobs
from . import chip, drums, space, texture, voices
from . import voice as synth

SR = dsp.SR
BEATS = 4                      # every bed is in four; a bar has sixteen steps
REF = -26.0                    # dBFS: the RMS of the loudest 400 ms of a part whose level is 0 dB, before the bed
#                                as a whole is levelled
RING = 6.0                     # s of room after the last bar for the final chord of a bed played once
CREST = 14.0                   # dB: peaks further than this over the RMS of the bed's loudest 400 ms are limited
#                                (a kick or a snap alone does not set the level of the whole bed)
MODES = {'major': (0, 2, 4, 5, 7, 9, 11), 'minor': (0, 2, 3, 5, 7, 8, 10), 'dorian': (0, 2, 3, 5, 7, 9, 10)}
MINORS = ('minor', 'dorian')
PENTATONIC = {False: (0, 1, 2, 4, 5), True: (0, 2, 3, 4, 6)}     # scale steps: 1 2 3 5 6 of a major mode, 1 3 4 5 7
#                                                                  of a minor one
NAMES = ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')
# a chord's colour: the scale steps over its root that sound (2 = the third, 4 = the fifth, 6 = the seventh,
# 8 = the ninth, 1 / 3 / 5 = second, fourth, sixth), the ones the upper voices must hold, and its name on a major
# and on a minor third. Steps are counted in the key's scale, so every colour stays in the key
COLOURS = {
    'triad': ((0, 2, 4), (2,), '', 'm'),
    'add9': ((0, 2, 4, 8), (2, 8), 'add9', 'm(add9)'),
    'sus2': ((0, 1, 4), (1, 4), 'sus2', 'sus2'),
    'sus4': ((0, 3, 4), (3,), 'sus4', 'sus4'),
    '6': ((0, 2, 4, 5), (2, 5), '6', 'm6'),
    '69': ((2, 4, 5, 8), (2, 5, 8), '6/9', 'm6/9'),
    '7': ((0, 2, 4, 6), (2, 6), 'maj7', 'm7'),
    '9': ((2, 4, 6, 8), (2, 6, 8), 'maj9', 'm9'),
    '11': ((3, 4, 6, 8), (3, 6), '11', '11'),
}
# the colours a mood draws from, with weights; a colour that does not fit a degree (a ninth a semitone over the
# root, an augmented fourth, a minor sixth, a dominant seventh's tritone) is left out there
PALETTES = {
    'jazz': (('9', 3), ('7', 3), ('69', 1), ('add9', 1)),
    'pop': (('add9', 3), ('triad', 2), ('sus2', 1), ('6', 1)),
    'plain': (('triad', 1),),
}
# where a role sits against the grid: ms late (+) or ahead (-), the spread of its timing in ms, the spread of its
# velocity (a share). A style scales the timing by its `feel` (0 = a machine)
FEEL = {'keys': (5.0, 5.0, 0.07), 'arp': (0.0, 3.5, 0.07), 'lead': (8.0, 6.0, 0.08), 'bass': (3.0, 3.5, 0.05),
        'kick': (0.0, 1.5, 0.04), 'snare': (9.0, 3.5, 0.07), 'hat': (-5.0, 2.5, 0.09)}
STRENGTH = {'X': 1.0, 'x': 0.78, 'o': 0.45, '?': 0.42}    # a rhythm grid: accent, stroke, ghost note, a ghost note
#                                                           that is played or not from bar to bar ('.' = a rest)


# ------------------------------------------------------------------------------------------------------- the styles
# A style: tempo, home key and mode, usual length in bars, how many bars a chord may last (`hold`), its colours,
# the number of upper voices and their centre (MIDI), the lowest bass note (the bass keeps to the octave above it),
# `feel` and `swing` (the grid unit in beats, the share of it the off-beat comes late), the room, `duck` (dB the
# keys and bass give way to the kick), `turn` (the last bar turns round on the dominant), `invert` (how often a
# bass line that can move by step does), `tilt` (dB of high shelf on the mix), `tape` and `top` (lo-fi: dB of tape
# saturation, a low-pass in Hz), progressions as (mode, scale degrees with 0 = I), and the parts.
# A part: its role (FEEL), a voice of earshot.synth.voices or a `piece` of the kit, a pattern and what the pattern
# reads, its level in dB (see REF), pan, `send` to the room, a low-pass `lp` in Hz, `echo` (beats, feedback, mix),
# and when it plays: `enter` (the share of the loop it sits out at first, `pickup`: led in by two soft strokes;
# from the start when energy >= `full`), `need` / `upto` (the lowest and highest energy it plays at all)
STYLES = {
    'warm': dict(
        bpm=84.0, key='F', mode='major', bars=8, hold=(1,), palette='jazz', voices=4, centre=62, bass_lo=36,
        feel=1.5, swing=(0.25, 0.18), room='room', duck=2.5, turn=True, invert=0.0, tape=5.0, top=8500.0,
        progs=(('major', (1, 4, 0, 5)), ('major', (0, 5, 1, 4)), ('major', (3, 2, 1, 0)), ('major', (3, 4, 2, 5)),
               ('dorian', (0, 3)), ('minor', (0, 5, 2, 6)), ('minor', (0, 3, 5, 4))),
        parts=(
            dict(name='keys', role='keys', voice='electric_piano', main=True, pattern='comp', level=-8.0, vel=0.62,
                 pan=-0.15, send=0.5, duck=True, strum=14.0, lp=6500.0,
                 cells=(((0.0, 3.8, 1.0, 'all'), (2.5, 1.3, 0.6, 'top2')),
                        ((0.0, 3.3, 1.0, 'all'), (1.75, 1.5, 0.55, 'mid'), (3.5, 4.3, 0.85, 'all', 'push')),
                        ((0.0, 3.8, 1.0, 'all'), (2.5, 1.3, 0.65, 'up')),
                        ((0.0, 3.8, 1.0, 'all'), (3.0, 0.9, 0.5, 'top')))),
            dict(name='bass', role='bass', voice='round_bass', pattern='bass', level=-13.0, vel=0.6, duck=True,
                 cells=(((0.0, 1.4, 1.0, 'r'), (2.5, 0.9, 0.75, 'r')),
                        ((0.0, 2.0, 1.0, 'r'), (2.5, 0.7, 0.7, '5'), (3.5, 0.4, 0.6, 'a')))),
            dict(name='kick', role='kick', piece='kick', pattern='drum', level=-13.0, vel=0.8,
                 grid=('X......x..x.....', 'X.....?...x....?')),
            dict(name='snare', role='snare', piece='snare', pattern='drum', level=-12.0, vel=0.7, send=0.5,
                 need=-0.3, grid=('....X.......X...', '....X..?....X...'), fill='....X.......X.oo'),
            dict(name='rim', role='snare', piece='rim', pattern='drum', level=-15.0, vel=0.7, send=0.5, upto=-0.3,
                 grid=('....X.......X...',)),
            dict(name='shaker', role='hat', piece='shaker', pattern='drum', level=-23.0, vel=0.7, pan=0.25,
                 send=0.3, enter=0.25, full=0.5, need=-0.6, grid=('x.x.x.x.x.x.x.x.', 'x.x?x.x.x.x?x.xo')),
            dict(name='crackle', role='texture', scene='crackle', level=-38.0),
            dict(name='lead', role='lead', voice='electric_piano', pattern='motif', level=-13.0, vel=0.6, pan=0.2,
                 send=0.8, enter=0.5, need=-0.2, centre=74, count=3, span=2, echo=(0.75, 0.3, 0.25), lp=6000.0),
        )),
    'playful': dict(
        bpm=120.0, key='C', mode='major', bars=8, hold=(1, 2), palette='pop', voices=3, centre=64, bass_lo=36,
        feel=0.9, swing=(0.5, 0.2), room='room', turn=True, invert=0.3, tilt=1.5,
        progs=(('major', (0, 3, 4, 0)), ('major', (0, 5, 3, 4)), ('major', (0, 4, 5, 3)), ('major', (0, 3, 0, 4)),
               ('major', (0, 5, 1, 4)), ('minor', (0, 6, 5, 6)), ('minor', (0, 3, 5, 4))),
        parts=(
            dict(name='pizzicato', role='keys', voice='pizzicato', pattern='comp', level=-10.0, vel=0.65, pan=-0.2,
                 send=0.5, strum=6.0,
                 cells=(((1.0, 0.4, 1.0, 'all'), (3.0, 0.4, 0.9, 'all')),
                        ((1.0, 0.4, 1.0, 'all'), (2.5, 0.3, 0.6, 'top2'), (3.0, 0.4, 0.9, 'all')))),
            dict(name='tune', role='lead', voice='muted_guitar', main=True, pattern='motif', level=-9.0, vel=0.7,
                 pan=0.15, send=0.5, centre=67, count=4, span=2, ring=True),
            dict(name='bass', role='bass', voice='upright_bass', pattern='bass', level=-11.0, vel=0.7,
                 cells=(((0.0, 0.8, 1.0, 'r'), (2.0, 0.8, 0.9, '5')),
                        ((0.0, 0.8, 1.0, 'r'), (2.0, 0.8, 0.9, '5'), (3.5, 0.4, 0.65, 'a')))),
            dict(name='kick', role='kick', piece='kick', pattern='drum', level=-13.0, vel=0.7, need=-0.3,
                 grid=('X.......x.......',)),
            dict(name='clap', role='snare', piece='clap', pattern='drum', level=-17.0, vel=0.7, send=0.5,
                 enter=0.5, full=0.6, need=-0.4, pickup=True, grid=('....X.......X...',),
                 fill='....X.......X.X.'),
            dict(name='shaker', role='hat', piece='shaker', pattern='drum', level=-23.0, vel=0.7, pan=0.3, send=0.3,
                 enter=0.25, full=0.5, need=-0.6, grid=('x.x.x.x.x.x.x.x.', 'x.x.x.xox.x.x.xo')),
        )),
    '8bit': dict(
        bpm=126.0, key='C', mode='major', bars=8, hold=(1, 2), palette='plain', voices=3, centre=64, bass_lo=36,
        feel=0.0, room='dry', turn=True, invert=0.0,
        progs=(('major', (0, 4, 5, 3)), ('major', (0, 3, 4, 0)), ('major', (0, 5, 3, 4)), ('major', (0, 3, 4, 3)),
               ('minor', (0, 6, 5, 6)), ('minor', (0, 5, 2, 6))),
        parts=(
            dict(name='lead', role='lead', voice='chip:lead', pattern='motif', level=-9.0, vel=0.8, pan=0.1,
                 centre=76, count=5, span=2, last=1.5),
            dict(name='arp', role='arp', voice='chip:arp', pattern='arp', level=-14.0, vel=0.7, pan=-0.2, gate=0.8,
                 grid=('xxxxxxxxxxxxxxxx',), seq=((0, 1, 2, 1),)),
            dict(name='bass', role='bass', voice='chip:bass', pattern='bass', level=-12.0, vel=0.9,
                 cells=(((0.0, 0.45, 1.0, 'r'), (0.5, 0.45, 0.9, 'r'), (1.0, 0.45, 1.0, '8'), (1.5, 0.45, 0.9, 'r'),
                         (2.0, 0.45, 1.0, 'r'), (2.5, 0.45, 0.9, 'r'), (3.0, 0.45, 1.0, '8'),
                         (3.5, 0.45, 0.9, '5')),)),
            dict(name='kick', role='kick', piece='chip_kick', pattern='drum', level=-12.0, vel=0.8,
                 grid=('X.......X.......', 'X.......X.X.....')),
            dict(name='snare', role='snare', piece='chip_snare', pattern='drum', level=-14.0, vel=0.8,
                 grid=('....X.......X...',), fill='....X.......X.XX'),
            dict(name='hat', role='hat', piece='chip_hat', pattern='drum', level=-22.0, vel=0.7, need=-0.5,
                 grid=('x.x.x.x.x.x.x.x.', 'x.x.x.x.x.x.x.xx')),
        )),
}


# ---------------------------------------------------------------------------------------------------------- harmony
def _pc(key):
    """the pitch class 0-11 of a key given as a note name with or without an octave ('D', 'F#', 'Bb3'), as Hz or as
    a pitch class"""
    if isinstance(key, str):
        m = knobs.midi(key if key[-1].isdigit() else key + '4')
        if m is None:
            raise ValueError(f'unknown key {key!r}: a note name such as D, F# or Bb')
        return m % 12
    if isinstance(key, (int, np.integer)) and 0 <= key < 12:
        return int(key)
    return int(round(69 + 12 * np.log2(float(key) / 440.0))) % 12


def _semi(mode, step):
    """semitones over the tonic of scale step `step` (7 = the octave)"""
    return MODES[mode][step % 7] + 12 * (step // 7)


def _fits(mode, deg, colour):
    """whether a colour sounds on a degree: a perfect fifth, a major ninth, a perfect fourth, a major sixth over a
    major third where it uses them, and no dominant seventh over a major third (the tritone), which the eleventh
    replaces"""
    steps = COLOURS[colour][0]
    iv = {s: (_semi(mode, deg + s) - _semi(mode, deg)) % 12 for s in (1, 2, 3, 4, 5, 6)}
    if iv[4] != 7 or ((1 in steps or 8 in steps) and iv[1] != 2) or (3 in steps and iv[3] != 5):
        return False
    if (5 in steps and (iv[5] != 9 or iv[2] != 4)) or (colour == '11' and iv[6] != 10):
        return False
    return not (2 in steps and 6 in steps and iv[2] == 4 and iv[6] == 10)


def _progression(st, r, bars, mode):
    """(mode, degrees, bars per chord) of a bed: one of the style's progressions in the mode asked for (or its own),
    a whole number of times in the bars where one fits"""
    want = mode or st['mode']
    pool = [g for g in st['progs'] if g[0] == want] if mode else []
    pool = pool or [g for g in st['progs'] if (g[0] in MINORS) == (want in MINORS)]
    fit = [(m, p, h) for m, p in pool for h in st['hold'] if bars % (len(p) * h) == 0]
    fit = fit or [(m, p, st['hold'][0]) for m, p in pool]
    return fit[int(r.integers(len(fit)))]


def _spans(st, mode, prog, hold, bars, loop):
    """the chords in time, [{'beat', 'len', 'deg', 'turn', 'final'}]: the progression bar by bar; the second half of
    the last bar turns round on the fifth degree unless the bar already leads home; played once, the bed ends with a
    bar of the tonic instead of repeating"""
    body = bars if loop else bars - 1
    cells = [[b * 4.0, 4.0, prog[(b // hold) % len(prog)], False, False] for b in range(body)]
    leads_home = cells[-1][2] == 4 or (mode in MINORS and cells[-1][2] == 6)
    if st['turn'] and not leads_home:
        cells[-1][1] = 2.0
        cells.append([cells[-1][0] + 2.0, 2.0, 4, True, False])
    if not loop:
        cells.append([body * 4.0, 4.0, 0, False, True])
    spans = []
    for beat, length, deg, turn, final in cells:
        if spans and (spans[-1]['deg'], spans[-1]['turn'], spans[-1]['final']) == (deg, turn, final):
            spans[-1]['len'] += length
        else:
            spans.append(dict(beat=beat, len=length, deg=deg, turn=turn, final=final))
    return spans


def _colour(st, spans, mode, tonic, r, loop):
    """give every chord its colour, its notes and its bass: one colour per degree for the whole bed (a chord that
    comes back is the same chord), the turnaround a suspended fourth or an eleventh, and a first inversion where the
    third lies a step from the roots on both sides"""
    chosen = {}
    for s in spans:
        want = (s['deg'], s['turn'])
        if want not in chosen:
            pal = [(c, w) for c, w in PALETTES[st['palette']] if _fits(mode, s['deg'], c)] or [('triad', 1)]
            if s['turn']:
                pal = [(c, 1) for c in (('11', 'sus4') if st['palette'] == 'jazz' else ('sus4',))
                       if _fits(mode, s['deg'], c)][:1] or pal
            w = np.array([w for _, w in pal], float)
            chosen[want] = pal[int(r.choice(len(pal), p=w / w.sum()))][0]
        steps, need, major, minor = COLOURS[chosen[want]]
        root = (tonic + _semi(mode, s['deg'])) % 12
        at = {k: (root + _semi(mode, s['deg'] + k) - _semi(mode, s['deg'])) % 12 for k in set(steps) | {2}}
        name = minor if (at[2] - root) % 12 == 3 else major
        s.update(colour=chosen[want], root=root, pcs=tuple(at[k] for k in steps), need={at[k] for k in need},
                 third=at[2] if 2 in steps else None, bass_pc=root, name=NAMES[root] + name)
    for i, s in enumerate(spans):
        prev = spans[i - 1] if (i or loop) else None
        nxt = spans[(i + 1) % len(spans)] if (i + 1 < len(spans) or loop) else None
        if s['third'] is None or s['turn'] or s['final'] or prev is None or nxt is None:
            continue
        near = [min((s['third'] - o['root']) % 12, (o['root'] - s['third']) % 12) for o in (prev, nxt)]
        if all(1 <= d <= 2 for d in near) and r.uniform() < st['invert']:
            s.update(bass_pc=s['third'], need=(s['need'] - {s['third']}) | {s['root']},
                     pcs=tuple(p for p in s['pcs'] if p != s['third']), name=s['name'] + '/' + NAMES[s['third']])
    for s in spans:
        s['bass'] = st['bass_lo'] + (s['bass_pc'] - st['bass_lo']) % 12


def _voicings(pcs, need, k, lo, hi):
    """every way to set k voices between lo and hi (MIDI) on the pitch classes pcs, as an array (m, k) low to high:
    the `need` tones all present, no tone doubled while another is missing, neighbours a major second to a major
    sixth apart, and no two voices a semitone or a minor ninth apart. A chord that has no such voicing in the
    window (a minor ninth chord whose third must lie under its ninth) gets an octave between neighbours and four
    semitones more room at the top, then gives up the tones it must hold, then the rule against minor ninths"""
    for gap, room, must, clean in ((9, 0, need, True), (12, 4, need, True), (12, 4, set(), True),
                                   (12, 4, set(), False)):
        cand = [m for m in range(lo, hi + room + 1) if m % 12 in pcs]
        out = []
        for v in itertools.combinations(cand, k):
            gaps = [b - a for a, b in zip(v, v[1:])]
            cls = {m % 12 for m in v}
            if min(gaps) < 2 or max(gaps) > gap or not must <= cls or len(cls) < min(k, len(pcs)):
                continue
            if clean and any((b - a) % 12 == 1 for a, b in itertools.combinations(v, 2)):
                continue
            out.append(v)
        if out:
            return np.array(out, dtype=np.int64)
    raise ValueError(f'no voicing of {k} voices on the pitch classes {sorted(pcs)} between {lo} and {hi}')


def _move(a, b):
    """the cost of going from each voicing in a (m, k) to each in b (n, k): the semitones the voices move, the top
    voice counted half again (it is the one heard), and a penalty when every voice jumps the same way"""
    d = b[None, :, :] - a[:, None, :]
    block = (np.all(d >= 2, axis=-1) | np.all(d <= -2, axis=-1)) * 4.0
    return np.abs(d).sum(axis=-1) + 0.5 * np.abs(d[..., -1]) + block


def _voice_lead(st, spans, r, loop):
    """voice every chord: the path through all voicings that moves the voices least in total (Viterbi; closed over
    the loop point for a loop), near the style's centre, without a close interval under middle C. Returns the semitones
    each voice moves at each chord change"""
    k = st['voices']
    centre = st['centre'] + int(r.integers(-2, 3))
    lo, hi = max(centre - 10, 53), centre + 10
    cands, stat = [], []
    for s in spans:
        v = _voicings(s['pcs'], s['need'], k, lo, hi)
        mud =((v[:, 1] - v[:, 0] < 5) & (v[:, 0] < 60)) * 3.0
        dbl = np.array([sum(max(0, list(row % 12).count(p) - 1) for p in s['need']) for row in v]) * 1.5
        cands.append(v)
        stat.append(0.3 * np.abs(v.mean(axis=1) - centre) + mud + dbl + r.uniform(0.0, 0.5, len(v)))
    n = len(spans)
    steps = [_move(cands[i], cands[i + 1]) for i in range(n - 1)]
    close = _move(cands[-1], cands[0]) if loop and n > 1 else None
    best, pick = np.inf, [int(np.argmin(stat[0]))]
    for s in (np.argsort(stat[0])[:12] if close is not None else [None]):
        cost = stat[0].copy()
        if s is not None:
            cost = np.full(len(stat[0]), np.inf)
            cost[s] = stat[0][s]
        back = []
        for i in range(n - 1):
            m = cost[:, None] + steps[i]
            back.append(m.argmin(axis=0))
            cost = m.min(axis=0) + stat[i + 1]
        if s is not None:
            cost = cost + close[:, s]
        j = int(cost.argmin())
        if cost[j] < best:
            best, path = cost[j], [j]
            for b in reversed(back):
                path.append(int(b[path[-1]]))
            pick = path[::-1]
    for s, v, j in zip(spans, cands, pick):
        s['voicing'] = tuple(int(m) for m in v[j])
    pairs = list(zip(spans, spans[1:] + (spans[:1] if loop else [])))
    return [float(np.mean(np.abs(np.subtract(b['voicing'], a['voicing'])))) for a, b in pairs if a is not b]


# --------------------------------------------------------------------------------------------------------- patterns
def _at(ctx, beat):
    """the chord sounding at a beat (a loop wraps; a bed played once holds its last chord)"""
    b = beat % ctx['total'] if ctx['loop'] else min(max(beat, 0.0), ctx['total'] - 1e-6)
    for s in reversed(ctx['spans']):
        if b >= s['beat'] - 1e-9:
            return s
    return ctx['spans'][0]


def _pick(v, what):
    """the notes of a voicing (low to high) a hit plays: all, the top one or two, the middle, or all but the
    lowest"""
    return {'all': v, 'top': v[-1:], 'top2': v[-2:], 'mid': v[1:-1] or v[:1], 'up': v[1:]}[what]


def _bars_of(ctx, p):
    """the bars a part plays: none outside its range of energy; from the start when the bed is energetic, else
    after it has sat out its share of the loop (to an even bar)"""
    e, body = ctx['energy'], ctx['body']
    if not p.get('need', -2.0) <= e <= p.get('upto', 2.0):
        return []
    start = 0 if e >= p.get('full', 2.0) else min(body - 1, 2 * int(round(p.get('enter', 0.0) * body / 2.0)))
    return list(range(start, body))


def _grid(ctx, p, bars):
    """the strokes of a rhythm grid over the bars: (bar, beat, strength, the stroke's number in its bar, the step in
    beats). A part may give one grid per bar in turn and a `fill` for the bar that ends a four-bar phrase; '?'
    strokes come and go with the seed and the energy"""
    r, e, body = ctx['r'], ctx['energy'], ctx['body']
    out = []
    for b in bars:
        g = p['grid'][b % len(p['grid'])]
        if p.get('fill') and body >= 4 and ((b + 1) % 4 == 0 or b == body - 1):
            g = p['fill']
        step, k = 4.0 / len(g), -1
        for i, ch in enumerate(g):
            if ch == '.':
                continue
            k += 1
            if (ch == '?' and r.uniform() > 0.4 + 0.35 * e) or (ch == 'o' and e < -0.6):
                continue
            out.append((b, b * 4.0 + i * step, STRENGTH[ch], k, step))
    return out


def _drum(ctx, p, bars):
    """a kit piece on its grid; a part with `pickup` that enters late leads into its first bar with two soft
    strokes on the last two sixteenths before it"""
    out = [[beat, p['piece'], s * p['vel'], None, 0.0] for _, beat, s, _, _ in _grid(ctx, p, bars)]
    if p.get('pickup') and bars[0] > 0:
        out += [[bars[0] * 4.0 - back, p['piece'], s * p['vel'], None, 0.0] for back, s in ((0.5, 0.45), (0.25, 0.62))]
    return out


def _arp(ctx, p, bars):
    """an arpeggio or ostinato: the grid's strokes walk `seq` through the chord's voicing from the bottom (an index
    past the top voice is the same chord an octave up), each note held for `gate` of its step"""
    out = []
    for b, beat, s, k, step in _grid(ctx, p, bars):
        v = sorted(_at(ctx, beat)['voicing'])
        seq = p['seq'][b % len(p['seq'])]
        i = seq[k % len(seq)]
        out.append([beat, v[i % len(v)] + 12 * (i // len(v)), s * p['vel'], p['gate'] * step, 0.0])
    return out


def _comp(ctx, p, bars):
    """chords in a rhythm: each bar plays one of the part's cells in turn, a cell a list of hits (beat, length in
    beats, strength, which notes[, 'push']). A chord is rolled from the bottom over `strum` ms with the top note
    strongest; soft hits come and go with the energy; a 'push' hit plays the next bar's chord early and holds it
    over the bar line, so that bar does not strike its downbeat again"""
    r, e = ctx['r'], ctx['energy']
    ms = 1e-3 * ctx['bpm'] / 60.0
    out, pushed = [], False
    for b in bars:
        was, pushed = pushed, False
        for hit in p['cells'][b % len(p['cells'])]:
            pos, dur, strength, what = hit[:4]
            push = len(hit) > 4
            if (was and pos == 0.0) or (push and b == bars[-1]):
                continue
            if strength < 0.7 and r.uniform() > 0.75 + 0.5 * e:
                continue
            beat = b * 4.0 + pos
            v = sorted(_at(ctx, beat + (1.0 if push else 0.0))['voicing'])
            for j, m in enumerate(_pick(v, what)):
                bal = 1.0 if m == v[-1] else 0.86 if m == v[0] else 0.72
                roll = j * p.get('strum', 0.0) * ms * r.uniform(0.7, 1.3)
                out.append([beat, m, strength * bal * p['vel'], dur, roll])
            pushed = push
    return out


def _approach(ctx, frm, to):
    """the scale note next to `to` on the side `frm` comes from: the step that leads into the next root"""
    d = 1 if frm > to else -1
    m = to + d
    while m % 12 not in ctx['scale']:
        m += d
    return m


def _bass(ctx, p, bars):
    """a bass line: each bar plays one of the part's cells in turn, a cell a list of (beat, length, strength, tone)
    with tone 'r' the chord's bass note, '5' its fifth (under it when that stays in the register), '8' the octave
    and 'a' a step that leads into the next chord's bass note. A note ends before the bar does"""
    out = []
    for b in bars:
        for pos, dur, strength, tone in p['cells'][b % len(p['cells'])]:
            beat = b * 4.0 + pos
            c = _at(ctx, beat)
            root, own = c['bass'], c['bass_pc'] == c['root']
            if tone == '5' and own:
                m = root - 5 if root - 5 >= ctx['st']['bass_lo'] - 2 else root + 7
            elif tone == '8' and own:
                m = root + 12
            elif tone == 'a':
                m = _approach(ctx, root, _at(ctx, (b + 1) * 4.0)['bass'])
            else:
                m = root
            out.append([beat, m, strength * p['vel'], dur, 0.0])
    return out


def _motif(ctx, p, bars):
    """a top line: one motif of `count` notes (more with energy) on the key's pentatonic, drawn once: onsets a
    half or a whole beat apart from the start of a `span`-bar phrase, steps of one or two scale notes around
    `centre`. It is restated every phrase; every second statement ends on another note; the last note of each lands
    on a tone of the chord under it. The rest of the phrase is silence"""
    r, e = ctx['r'], ctx['energy']
    span, res = p.get('span', 2), p.get('res', 0.5)
    count = int(np.clip(round(p.get('count', 3) + 1.5 * e * r.uniform(0.5, 1.0)), 2, 6))
    on = [float(r.choice([0.0, res, 2 * res]))]
    while len(on) < count and on[-1] + 3 * res <= span * 4.0 - 2.0:
        on.append(on[-1] + res * int(r.choice([1, 2, 3], p=[0.4, 0.4, 0.2])))
    penta = {(ctx['tonic'] + _semi(ctx['mode'], s)) % 12 for s in PENTATONIC[ctx['mode'] in MINORS]}
    ladder = [m for m in range(p['centre'] - 8, p['centre'] + 9) if m % 12 in penta]
    home = set(_at(ctx, bars[0] * 4.0)['pcs'])
    walk = [min(range(len(ladder)), key=lambda i: (ladder[i] % 12 not in home, abs(ladder[i] - p['centre'])))]
    for _ in on[1:]:
        step = int(r.choice([-2, -1, 1, 2], p=[0.15, 0.35, 0.35, 0.15]))
        j = walk[-1] + step
        walk.append(j if 0 <= j < len(ladder) else walk[-1] - step)
    answer = int(r.choice([-1, 1]))
    out = []
    for k, b0 in enumerate(b for b in bars if (b - bars[0]) % span == 0):
        idx = list(walk)
        if k % 2:
            idx[-1] = int(np.clip(idx[-1] + answer, 0, len(ladder) - 1))
        for q, (t, i) in enumerate(zip(on, idx)):
            beat, last = b0 * 4.0 + t, q == len(on) - 1
            if last:
                tones = set(_at(ctx, beat)['pcs'])
                near = [j for j in range(max(0, i - 2), min(len(ladder), i + 3)) if ladder[j] % 12 in tones]
                i = min(near, key=lambda j: abs(j - i)) if near else i
            dur = None if p.get('ring') else p.get('last', 2.0) if last else (on[q + 1] - t) * 0.9
            out.append([beat, ladder[i], p['vel'] * (1.0 if last else 0.9 if q == 0 else 0.8), dur, 0.0])
    return out


PATTERNS = {'comp': _comp, 'arp': _arp, 'bass': _bass, 'motif': _motif, 'drum': _drum}


def _final(ctx, p):
    """what a part plays on the last chord of a bed played once: the chord once, rolled, left to ring; the bass its
    root; the kick one soft stroke; the other drums and the top line nothing"""
    c = ctx['spans'][-1]
    role, beat, ms = p['role'], c['beat'], 1e-3 * ctx['bpm'] / 60.0
    if role == 'kick':
        return [[beat, p['piece'], 0.7 * p['vel'], None, 0.0]]
    if role == 'bass':
        return [[beat, c['bass'], 0.9 * p['vel'], 4.0, 0.0]]
    if role in ('keys', 'arp'):
        return [[beat, m, 0.9 * p['vel'], 4.0, j * 35.0 * ms] for j, m in enumerate(sorted(c['voicing']))]
    return []


def _metric(beat):
    """how strong a position in the bar is: the downbeat, the third beat, the other beats, eighths, sixteenths"""
    b = beat % BEATS
    if abs(b - round(b)) < 1e-6:
        return (1.0, 0.86, 0.93, 0.86)[int(round(b)) % BEATS]
    return 0.76 if abs(2 * b - round(2 * b)) < 1e-6 else 0.64


def _humanise(ctx, p, notes):
    """notes [grid beat, pitch, velocity, length, offset] as played: (beat, pitch, velocity, length, grid beat).
    Timing: the swing of the style, the role's place against the grid, a slow drift and a little scatter (the notes
    of one chord move together). Velocity: the strength of the position in the bar (to a depth of 30 %), the
    energy, and a few per cent of touch"""
    st, r = ctx['st'], ctx['r']
    late, spread, touch = FEEL[p['role']]
    k = st['feel'] * 1e-3 * ctx['bpm'] / 60.0
    unit, swing = st.get('swing', (0.5, 0.0))
    depth, drift, dev, seen, out = 0.3, 0.0, 0.0, None, []
    for g, m, vel, dur, off in sorted(notes, key=lambda x: (x[0], x[4])):
        if g != seen:
            seen = g
            drift = 0.7 * drift + 0.5 * spread * r.standard_normal()
            dev = late + drift + 0.6 * spread * r.standard_normal()
        odd = abs(g / unit - round(g / unit)) < 1e-6 and int(round(g / unit)) % 2 == 1
        t = g + off + dev * k + (swing * unit if odd else 0.0)
        v = vel * (1.0 - depth * (1.0 - _metric(g))) * (1.0 + 0.12 * ctx['energy'])
        v *= float(np.exp(touch * min(1.0, st['feel']) * r.standard_normal()))
        out.append((float(t), m, float(np.clip(v, 0.05, 1.0)), dur, float(g)))
    return out


def compose(style, r, key=None, bpm=None, bars=None, mode=None, energy=0.0, loop=True, voice=None):
    """the score of a bed as a dict. style: a key of STYLES; r: a numpy Generator (the same arguments give the same
    score); key: the tonic as a note name, Hz or pitch class (default: the style's); bpm, bars: tempo and length
    (default: the style's); mode: 'major', 'minor' or 'dorian' (default: the style's; a
    mode the style has no progression in falls back to major or minor); energy -1..1: how much plays (parts, ghost
    notes, motif notes) and how hard; loop=False: played once, the last bar a held tonic chord; voice: another
    voice for the style's main instrument.
    The result: 'style', 'tonic' (pitch class), 'mode', 'bpm', 'bars', 'loop', 'energy', 'length_s' (the bars),
    'chords' [{'beat', 'len', 'name', 'voicing', 'bass', ...}], 'lead_st' (the semitones a voice moves per chord
    change, mean) and 'lead_steps' (per change), 'parts' [the style's part with 'notes': (beat, pitch or kit piece,
    velocity, length in beats or None = it rings, grid beat)]"""
    if style not in STYLES:
        raise ValueError(f'unknown style {style!r}: one of {", ".join(STYLES)}')
    if mode is not None and mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}: one of {", ".join(MODES)}')
    st = STYLES[style]
    bpm, bars = float(bpm or st['bpm']), int(bars or st['bars'])
    if bars < 2 or not 30.0 <= bpm <= 240.0:
        raise ValueError('a bed has at least two bars and a tempo of 30 to 240 bpm')
    tonic = _pc(st['key'] if key is None else key)
    mode, prog, hold = _progression(st, r, bars, mode)
    spans = _spans(st, mode, prog, hold, bars, loop)
    _colour(st, spans, mode, tonic, r, loop)
    steps = _voice_lead(st, spans, r, loop)
    ctx = dict(st=st, r=r, spans=spans, body=bars if loop else bars - 1, total=bars * 4.0, bpm=bpm,
               energy=float(np.clip(energy, -1.0, 1.0)), loop=loop, tonic=tonic, mode=mode,
               scale={(tonic + s) % 12 for s in MODES[mode]})
    parts = []
    for spec in st['parts']:
        p = dict(spec)
        if voice and p.get('main'):
            p['voice'] = voice
        on = _bars_of(ctx, p)
        if not on:
            continue
        notes = PATTERNS[p['pattern']](ctx, p, on) if p.get('pattern') else []
        p['notes'] = _humanise(ctx, p, notes + ([] if loop else _final(ctx, p))) if p.get('pattern') else []
        parts.append(p)
    return dict(style=style, tonic=tonic, mode=mode, bpm=bpm, bars=bars, loop=loop, energy=ctx['energy'],
                length_s=bars * BEATS * 60.0 / bpm, chords=spans, lead_steps=steps,
                lead_st=float(np.mean(steps)) if steps else 0.0, parts=parts)


def chart(plan):
    """a bed's key, tempo and chords in a line: 'D major, 84 bpm, 8 bars: Em9 | A11 | Dmaj9 | Bm7 ...'"""
    return (f"{NAMES[plan['tonic']]} {plan['mode']}, {plan['bpm']:.4g} bpm, {plan['bars']} bars: "
            + ' | '.join(c['name'] for c in plan['chords']))


# -------------------------------------------------------------------------------------------------------------- kit
def _chip_drum(kind, vel):
    """the noise and triangle channels of earshot.synth.chip as drums"""
    p = {'chip_hat': dict(wave='noise', freq=chip.NOISE_CLOCKS[2], sustain=0.008, decay=0.03),
         'chip_snare': dict(wave='noise', freq=chip.NOISE_CLOCKS[7], sustain=0.02, decay=0.1, punch=0.4),
         'chip_kick': dict(wave='triangle', freq=150.0, slide=-260.0, sustain=0.05, decay=0.07, steps=False)}[kind]
    return chip.voice(p)


# the built-in kit: the synthetic drum voices and the shaker and clap of earshot.synth.drums, a dry rim and the chip
# channels; each takes (velocity 0..1, generator) and returns one stroke. A caller may hand `render` other pieces
# under the same names (earshot.sounds.beds plays the catalogue's acoustic drums)
KIT = {
    'kick': lambda v, r: dsp.fade(drums.kick_synth(dsp.n_of(0.45), r, 60.0, v, sweep=1.8, drop=0.04, decay=0.13,
                                                   click=0.25, drive=1.5), 0.0, 0.15),
    'snare': lambda v, r: dsp.fade(drums.snare_synth(dsp.n_of(0.3), r, 190.0, v, tone=0.45, noise_tau=0.07), 0.0,
                                   0.08),
    'clap': lambda v, r: dsp.fade(drums.clap_synth(dsp.n_of(0.35), r, v, tail=0.09), 0.0, 0.1),
    'shaker': lambda v, r: dsp.fade(drums.shaker(dsp.n_of(0.22), r, [(0.055, v)]), 0.0, 0.03),
    'rim': lambda v, r: synth.tick(r, 0.07, 500.0, 3500.0, 0.007),
    'chip_hat': lambda v, r: _chip_drum('chip_hat', v),
    'chip_snare': lambda v, r: _chip_drum('chip_snare', v),
    'chip_kick': lambda v, r: _chip_drum('chip_kick', v),
}
CHIP = {'chip:lead': dict(wave='pulse', duty=0.25, punch=0.4), 'chip:arp': dict(wave='pulse', duty=0.125, punch=0.3),
        'chip:bass': dict(wave='triangle', steps=False)}


# ----------------------------------------------------------------------------------------------------------- render
def _stereo(x):
    x = np.asarray(x, np.float64)
    return x if x.ndim == 2 else np.stack([x, x], axis=1)


def _wrap_add(buf, y, i, loop):
    """add y into buf from sample i on; in a loop, what starts before the start or runs past the end goes round"""
    n = len(buf)
    if not loop:
        if i < 0:
            y, i = y[-i:], 0
        m = min(len(y), n - i)
        if m > 0:
            buf[i:i + m] += y[:m]
        return
    i %= n
    while len(y):
        m = min(len(y), n - i)
        buf[i:i + m] += y[:m]
        y, i = y[m:], 0


def _around(fn, y, loop, pad=SR // 2):
    """fn(y) for a process with a short memory (a filter, a limiter). In a loop it sees the end of y before the
    start and the start after the end, so it meets the loop point in the state the loop leaves it in and the
    result still closes"""
    if not loop:
        return fn(y)
    k = min(pad, len(y))
    return fn(np.concatenate([y[-k:], y, y[:k]]))[k:k + len(y)]


def _loudest(y, window=0.4):
    """the RMS of the loudest `window` s of y in dBFS: what a part's level is set by. Unweighted, so a bass or a
    kick is not turned up for the lows a loudness meter discounts"""
    hop = SR // 20                                             # measured in steps of 50 ms
    k = max(1, min(len(y) // hop, int(round(window * SR / hop))))
    p = np.concatenate([[0.0], np.cumsum(np.add.reduceat((y ** 2).sum(axis=1), np.arange(0, len(y), hop)))])
    return float(10 * np.log10((p[k:] - p[:-k]).max() / (2 * k * hop) + 1e-30))


def _texture(scene, n, r, loop):
    """n samples of a scene of earshot.synth.texture (record crackle): a seamless loop of the scene of up to eight
    seconds, made at a length its transforms are fast at, and repeated. The bed's own loop point is closed with a
    100 ms equal-power crossfade, which noise does not show; a bed played once fades the scene in"""
    k = dsp.n_of(0.1)
    unit = next_fast_len(min(n + k, 8 * SR), real=True)
    y = np.tile(texture.loop(scene, unit / SR, r), (-(-(n + k) // unit), 1))
    u = np.linspace(0.0, np.pi / 2, k)[:, None]
    out = y[:n].copy()
    out[:k] = out[:k] * np.sin(u) + (y[n:n + k] * np.cos(u) if loop else 0.0)
    return out


def _note(p, m, vel, dur, kit, cache, seed, count, bright):
    """one note or stroke: (stereo samples, the sample of its attack, the gain that gives it its velocity). A take
    is rendered once per (voice, pitch, velocity step, length); the step sets the timbre (five steps for an
    instrument and a drum, three for a synthesizer patch or a chip channel) and the gain the exact level. Drums
    and the instruments of an arpeggio or a top line alternate two takes: the same sample twice in a row is a
    machine, which a patch or a chip channel is and a chord struck once a bar does not show"""
    drum = isinstance(m, str)
    synthetic = not drum and (p['voice'] in CHIP or voices.info(p['voice'])['engine'] == 'patches')
    steps = 3 if synthetic else 5
    vq = float(np.clip(round(vel / 0.15) * 0.15, 0.3, 0.9)) if drum else max(1, int(np.ceil(vel * steps - 1e-9))) / steps
    key = (p.get('voice'), m, vq, None if dur is None else round(dur, 3))
    count[key] = count.get(key, -1) + 1
    key += (count[key] % (2 if drum or (p['role'] in ('arp', 'lead') and not synthetic) else 1),)
    if key not in cache:
        rr = dsp.rng(zlib.crc32(repr(key).encode()) ^ seed)
        hz = None if drum else 440.0 * 2.0 ** ((m - 69) / 12.0)
        if drum:
            x = _stereo(kit[m](vq, rr))
            x = x / (np.abs(x).max() + 1e-12)
            anchor = int(np.argmax(np.abs(x).max(axis=1) >= 0.3))
        elif p['voice'] in CHIP:
            d = dur or 0.3
            env = dict(sustain=0.9 * d, decay=0.02) if p['voice'] == 'chip:bass' else dict(sustain=0.55 * d,
                                                                                         decay=0.4 * d + 0.02)
            x, anchor = _stereo(chip.voice(dict(CHIP[p['voice']], freq=hz, **env))) * vq, 0
        else:
            x, anchor = voices.play(p['voice'], hz, vel=vq, dur=dur, r=rr, vary=0.3,
                                    bright=float(np.clip(0.5 * bright, -1.0, 1.0))), 0
        cache[key] = (x, anchor)
    x, anchor = cache[key]
    return x, anchor, vel ** 1.4 if drum else vel / vq


def _duck(plan, n, pre, spb, depth):
    """the gain that moves keys and bass out of the kick's way: down `depth` dB within 4 ms of each kick and back
    with a time constant of a third of a beat"""
    kicks = [t for p in plan['parts'] if p['role'] == 'kick' for t, *_ in p['notes']]
    if not depth or not kicks:
        return None
    k = int(2.0 * spb * SR)
    t = np.arange(k) / SR
    dip = np.minimum(t / 0.004, 1.0) * np.exp(-t / (0.33 * spb))
    g = np.zeros(n)
    for beat in kicks:
        i = pre + int(round(beat * spb * SR)) + np.arange(k)
        ok = np.ones(k, bool) if plan['loop'] else (i >= 0) & (i < n)
        g[i[ok] % n] = np.maximum(g[i[ok] % n], dip[ok])
    return 10.0 ** (-depth * g / 20.0)


def _treat(plan, p, y, spb):
    """a part's own processing: the low parts mono, its low-pass, a dip where a voice would sit, an echo in time,
    its place"""
    loop = plan['loop']
    if p['role'] in ('bass', 'kick'):
        y = np.repeat(y.mean(axis=1, keepdims=True), 2, axis=1)
    if p.get('lp'):
        y = _around(lambda z: dsp.butter(z, 'low', p['lp']), y, loop)
    if p['role'] == 'lead':
        y = _around(lambda z: dsp.peq(z, 2500.0, -2.5, 0.8), y, loop)
    if p.get('echo'):
        beats, feedback, mix = p['echo']
        z = synth.echo(y, time=beats * spb, feedback=feedback, mix=mix, damp=3500.0)
        y = np.zeros_like(y)
        _wrap_add(y, z, 0, loop)
    return dsp.pan_to_stereo(y, p['pan']) if p.get('pan') else y


def render(plan, r, kit=None, room=None, amount=0.0, bright=0.0, width=0.0, stems=False):
    """a composed bed as audio: stereo float64 (n, 2), peak 0.5. A loop is exactly its bars long
    (round(plan['length_s'] * 48000) samples) and repeats without a seam; a bed played once has 40 ms before its
    first beat and ends when its last chord has rung out (at most RING s after the last bar, faded).
    r: a numpy Generator (the takes); kit: {name: fn(velocity, generator)} pieces that replace those of KIT;
    room: a space of earshot.synth.space (default: the style's) and amount -1..1 more or less of it; bright -1..1:
    darker or brighter voices and mix; width -1..1: narrower or wider. stems=True returns (y, {part: its samples in
    the mix before the room}).
    The mix: each part is brought to its level by the RMS of its own loudest 400 ms (REF + level), keys and bass
    duck under the kick, the parts send to one room, everything under 120 Hz is mono"""
    st = STYLES[plan['style']]
    loop, spb = plan['loop'], 60.0 / plan['bpm']
    n = int(round(plan['length_s'] * SR))
    pre = 0 if loop else dsp.n_of(0.04)
    total = n if loop else pre + n + dsp.n_of(RING)
    seed = int(r.integers(2 ** 31))
    kit = {**KIT, **(kit or {})}
    duck = _duck(plan, total, pre, spb, st.get('duck', 0.0))
    mix, send, parts, cache = np.zeros((total, 2)), np.zeros((total, 2)), {}, {}
    for p in plan['parts']:
        y = np.zeros((total, 2))
        if p['role'] == 'texture':
            y = _texture(p['scene'], total, dsp.rng(seed + 7), loop)
        count = {}
        for t, m, vel, dur, _ in p['notes']:
            x, anchor, gain = _note(p, m, vel, None if dur is None else dur * spb, kit, cache, seed, count, bright)
            _wrap_add(y, gain * x, pre + int(round(t * spb * SR)) - anchor, loop)
        if np.abs(y).max() <= 0.0:
            continue
        y = _treat(plan, p, y, spb)
        y *= 10.0 ** ((REF + p['level'] - _loudest(y)) / 20.0)
        if p.get('duck') and duck is not None:
            y *= duck[:, None]
        mix += y
        send += p.get('send', 0.0) * y
        parts[p['name']] = y
    if width:
        w = 1.0 + (0.6 if width > 0 else 1.0) * width
        mix, send = fx.width(mix, w), fx.width(send, w)
    wet = space.room(send, room or st['room'], amount=float(np.clip(amount, -1.0, 1.0)), r=dsp.rng(seed + 1))
    k = min(len(wet), total)
    wet[:k] -= send[:k]                      # the room's tail alone: it goes round the loop point onto the start
    _wrap_add(mix, wet, 0, loop)
    mix = _around(lambda z: synth.mono_below(z, 120.0), mix, loop)
    tilt = 5.0 * bright + st.get('tilt', 0.0)
    if tilt:
        mix = _around(lambda z: z + (10.0 ** (tilt / 20.0) - 1.0) * dsp.butter(z, 'high', 3000.0, order=1), mix, loop)
    if st.get('tape'):
        mix = _around(lambda z: fx.saturate(z, st['tape'], 'tape', os=2), mix, loop, SR // 4)
    if st.get('top'):
        mix = _around(lambda z: dsp.butter(z, 'low', st['top']), mix, loop)
    mix = _around(lambda z: dsp.butter(z, 'high', 28.0), mix, loop, SR)
    ceiling = _loudest(mix) + CREST
    if 20 * np.log10(np.abs(mix).max() + 1e-12) > ceiling:
        mix = _around(lambda z: fx.limit(z, ceiling_db=ceiling, true_peak=False), mix, loop, SR // 4)
    if not loop:
        mix = dsp.fade(mix, 0.0, 2.0)
    gain = 0.5 / (np.abs(mix).max() + 1e-12)
    return (mix * gain, {k: v * gain for k, v in parts.items()}) if stems else mix * gain


def bed(style, r, kit=None, room=None, amount=0.0, bright=0.0, width=0.0, **score):
    """compose and render in one call: `score` goes to compose (key, bpm, bars, mode, energy, loop, voice), the
    rest to render. The generator is used for the score first, then for the takes"""
    return render(compose(style, r, **score), r, kit=kit, room=room, amount=amount, bright=bright, width=width)
