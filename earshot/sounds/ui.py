"""Interface sounds: the figures of product UI (confirmations, errors, notifications, toggles, navigation), each
playable by any voice, and the small material ticks that go with them.

The gestures follow what interface sound design has settled on (Material and Windows guidance, earcon research):
rising and consonant reads as positive, falling or repeated low notes as negative, a leap as a call for attention;
frequent sounds are short, soft and low in the mix, rare ones may ring; fundamentals stay between about 500 Hz and
2 kHz, where small speakers still reproduce them and nothing is piercing. Notes of a figure are at least 35 ms
apart (two notes closer than that fuse into one), motifs keep to two or three notes.

Each figure names its interval in semitones above its root. A theme (earshot.sounds.theme) moves every root into one
key and gives every figure one voice and one room, so a set reads as one product.
"""
from ..synth import space, voices
from . import figure
from .registry import design

FIGURES = {
    # ---- positive
    # three notes climbing from the fifth onto the root a fourth above: it rises and resolves
    'success/rise': dict(kind='chime', says=('three quick rising notes, a short soft confirmation',
                                             'a small three-note lift going up'),
                         uses=('success', 'confirm', 'saved', 'complete', 'check mark'), prior=1.15,
                         fig=dict(steps=(4, 7, 12), gap=0.09, root='G4', vel=0.6, vels=(0.8, 0.9, 1.0), last=0.55)),
    'success/fifth': dict(kind='chime', says=('two notes a fifth apart rising, an open clear signal',
                                              'a clean two-note signal going up'),
                          uses=('success', 'confirm', 'done', 'unlock', 'verified'),
                          fig=dict(steps=(0, 7), gap=0.095, root='E5', vel=0.6, vels=(0.8, 1.0), last=0.6)),
    'success/triad': dict(kind='figure', says=('a quick rising major arpeggio landing on the octave with the fifth '
                                               'under it, a small happy flourish',
                                               'a fast ascending run of notes that ends on a ringing chord'),
                          uses=('success', 'complete', 'achievement', 'purchase', 'level up', 'win'),
                          fig=dict(steps=(0, 4, 7, 12, 7), gap=(0.07, 0.07, 0.08, 0.012), root='C5', vel=0.64,
                                   vels=(0.72, 0.8, 0.88, 1.0, 0.55), last=0.9, accent=3, room='studio')),
    'success/done': dict(kind='chime', says=('one clear note with its octave above', 'a single soft round note'),
                         uses=('success', 'done', 'task done', 'check mark', 'timer done'),
                         fig=dict(steps=(0, 12), gap=0.012, root='A5', vel=0.55, vels=(1.0, 0.45), last=0.7, accent=0,
                                  spread=0.0)),
    'purchase/chime': dict(kind='figure',
                           says=('four rising notes ending an octave up and ringing, a celebratory arpeggio',
                                 'a quick ascending run of four notes'),
                           uses=('purchase', 'payment', 'order placed', 'achievement', 'reward', 'celebration'),
                           fig=dict(steps=(0, 4, 7, 12), gap=0.08, root='C5', vel=0.65, vels=(0.7, 0.8, 0.9, 1.0),
                                    last=1.0, room='studio')),
    'like/twinkle': dict(kind='figure', says=('three very quick high notes, a tiny rising flourish',
                                              'a fast grace-note run upward'),
                         uses=('like', 'favorite', 'star', 'reaction', 'sparkle'),
                         fig=dict(steps=(0, 7, 12), gap=0.04, root='C6', vel=0.45, vels=(0.7, 0.85, 1.0), last=0.35)),
    # ---- negative and cautionary
    'error/fall': dict(kind='chime', says=('two low notes falling by a minor third, a muted negative tone',
                                           'a short descending two-note signal'),
                       uses=('error', 'failed', 'invalid', 'denied', 'wrong'), prior=1.15,
                       # low and short: the first note damped as the second is struck, the whole figure gone by
                       # about half a second (a negative signal that lingers reads as a lament)
                       fig=dict(steps=(3, 0), gap=0.12, root='A3', vel=0.55, vels=(1.0, 0.9), gate=0.14, last=0.32,
                                accent=0, longest=0.46)),
    'error/double': dict(kind='chime', says=('the same low note struck twice quickly, a dull double knock of a tone',
                                             'two short low repeated notes'),
                         uses=('error', 'blocked', 'forbidden', 'not allowed', 'invalid'),
                         fig=dict(steps=(0, 0), gap=0.12, root='F4', vel=0.55, vels=(1.0, 0.85), gate=0.07, last=0.16,
                                  accent=0,
                                  voice='marimba', avoid=('metal', 'glass', 'synth'))),
    'warning/pulse': dict(kind='chime', says=('the same mid note three times, an even three-pulse signal',
                                              'three repeated short notes calling for attention'),
                          uses=('warning', 'caution', 'attention', 'alert'),
                          fig=dict(steps=(0, 0, 0), gap=0.17, root='F5', vel=0.6, vels=(1.0, 0.92, 0.96), gate=0.13,
                                   last=0.28, accent=None)),
    'delete/drop': dict(kind='chime', says=('two quick notes falling by a fourth, a small dark descending blip',
                                            'a short low two-note drop'),
                        uses=('delete', 'remove', 'trash', 'discard', 'clear'),
                        fig=dict(steps=(5, 0), gap=0.07, root='E4', vel=0.45, vels=(0.9, 1.0), last=0.2, accent=0,
                                 longest=0.75)),
    # ---- attention
    'notify/pair': dict(kind='chime', says=('two notes falling by a major third, a gentle two-note call',
                                            'a soft two-note notification'),
                        uses=('notify', 'notification', 'message received', 'reminder', 'doorbell'), prior=1.15,
                        fig=dict(steps=(4, 0), gap=0.17, root='C5', vel=0.6, vels=(1.0, 0.85), last=0.75, accent=0,
                                 room='studio')),
    'notify/lift': dict(kind='chime', says=('two notes rising by a fourth, a light upward call',
                                            'a short friendly two-note signal'),
                        uses=('notify', 'message', 'mention', 'incoming', 'ping'),
                        fig=dict(steps=(0, 5), gap=0.11, root='G5', vel=0.58, vels=(0.85, 1.0), last=0.65,
                                 room='studio')),
    'notify/motif': dict(kind='figure',
                         says=('three notes: up a fifth, then settling on the third, a small melodic call',
                               'a three-note notification melody'),
                         uses=('notify', 'notification', 'reminder', 'incoming call', 'alert'),
                         fig=dict(steps=(0, 7, 4), gap=(0.12, 0.14), root='G5', vel=0.6, vels=(0.9, 1.0, 0.85),
                                  last=0.8, accent=1, room='studio')),
    'notify/single': dict(kind='note', says=('one clear ringing note', 'a single soft note that rings briefly'),
                          uses=('notify', 'ping', 'bell', 'ding', 'reminder'),
                          fig=dict(steps=(0,), root='A5', vel=0.6, last=0.9, room='studio')),
    'timer/done': dict(kind='figure', says=('the same note three times, evenly, a soft alarm',
                                            'three repeated short notes'),
                       uses=('timer', 'alarm', 'time up', 'countdown end', 'reminder'),
                       fig=dict(steps=(0, 0, 0), gap=0.19, root='A5', vel=0.62, gate=0.12, last=0.6, accent=None)),
    # ---- messages and navigation
    # send runs on past where open stops (three notes up to the octave), so it leaves rather than opens
    'send/lift': dict(kind='chime', says=('three very quick notes running up an octave, a light upward blip',
                                          'a fast rising three-note chirp'),
                      uses=('send', 'message sent', 'submit', 'post', 'share', 'upload'),
                      fig=dict(steps=(0, 5, 12), gap=0.06, root='G4', vel=0.45, vels=(0.75, 0.85, 1.0), last=0.25)),
    'receive/drop': dict(kind='chime', says=('two very quick notes dropping a fifth, a light downward blip',
                                             'a fast falling two-note chirp'),
                         uses=('receive', 'message received', 'incoming', 'download'),
                         fig=dict(steps=(7, 0), gap=0.07, root='C5', vel=0.45, vels=(0.75, 1.0), last=0.3)),
    # a switch is a click-clack: two tactile parts 45-60 ms apart (the lever leaving its rest, the detent catching),
    # the second a little higher when it goes on and lower when it goes off; the notes under them only colour it
    'toggle/up': dict(kind='click', says=('a small two-part switch click, click-clack, with a soft rising tone in it',
                                          'a short tactile switch flipped on'),
                      uses=('switch on', 'toggle', 'enable', 'check'),
                      fig=dict(steps=(0, 5), gap=0.055, root='C6', vel=0.18, vels=(0.7, 1.0), gate=0.04, last=0.1,
                               longest=0.3, tap=2.4, tap_pitch=(0.85, 1.12))),
    'toggle/down': dict(kind='click', says=('a small two-part switch click, clack-click, a soft falling tone in it',
                                            'a short tactile switch flipped off'),
                        uses=('switch off', 'toggle', 'disable', 'deselect', 'uncheck'),
                        fig=dict(steps=(5, 0), gap=0.055, root='C6', vel=0.17, vels=(1.0, 0.7), gate=0.04, last=0.09,
                                 accent=0, longest=0.3, tap=2.4, tap_pitch=(1.0, 0.8))),
    'open/lift': dict(kind='chime', says=('two soft notes rising a fifth, a gentle opening gesture',
                                          'a quiet upward two-note gesture'),
                      uses=('open', 'expand', 'show', 'modal', 'menu open', 'next'),
                      fig=dict(steps=(0, 7), gap=0.10, root='G4', vel=0.42, vels=(0.85, 1.0), last=0.4)),
    'close/drop': dict(kind='chime', says=('two soft notes falling a fifth, a gentle closing gesture',
                                           'a quiet downward two-note gesture'),
                       uses=('close', 'collapse', 'dismiss', 'hide', 'back', 'cancel'),
                       fig=dict(steps=(7, 0), gap=0.10, root='G4', vel=0.4, vels=(1.0, 0.85), last=0.32, accent=0)),
    'next/step': dict(kind='chime', says=('two tiny notes a whole step up, a small forward tick',
                                          'a minimal rising two-note blip'),
                      uses=('next', 'forward', 'continue', 'tab', 'navigate'),
                      fig=dict(steps=(0, 2), gap=0.055, root='A5', vel=0.36, vels=(0.85, 1.0), gate=0.05, last=0.15,
                               longest=0.32, tap=0.45)),
    'back/step': dict(kind='chime', says=('two tiny notes a whole step down, a small backward tick',
                                          'a minimal falling two-note blip'),
                      uses=('undo', 'back', 'previous', 'return'),
                      fig=dict(steps=(2, 0), gap=0.055, root='A5', vel=0.36, vels=(1.0, 0.85), gate=0.05, last=0.15,
                               accent=0, longest=0.32, tap=0.45)),
    # ---- the smallest ones: one short note of a sustaining metal bar or a synthesizer is a bare tone burst, heard as
    # an electronic beep, so these keep to struck wood unless the theme's voice is itself short (wood, tines,
    # strings) or the theme is a synthesizer's, where a soft blip is the point
    'tap/tone': dict(kind='tap', says=('a soft rounded tactile tap, a key-like thock with a faint note in it',
                                       'a short muted press of a button'),
                     uses=('tap', 'button press', 'click', 'select', 'key'),
                     fig=dict(steps=(0,), root='C6', vel=0.22, last=0.07, room='booth', voice='marimba',
                              avoid=('metal', 'glass', 'electric'), longest=0.22, tap=1.2)),
    'hover/tone': dict(kind='tick', says=('a tiny soft tactile tick with a faint high note',
                                          'a very light short touch sound'),
                       uses=('hover', 'focus', 'highlight', 'scroll tick'),
                       fig=dict(steps=(0,), root='G6', vel=0.13, last=0.04, room='booth', voice='marimba',
                                avoid=('metal', 'glass', 'electric'), longest=0.14, tap=0.9, tap_pitch=1.5)),
}
figure.register(design, FIGURES, domains=('ui', 'game'))

# material ticks: one short strike of a small object, for taps and detents that should not be a note.
# (voice, note, damped after s or None = its own short ring, says). Heard by earshot's ears, a glass damped early and
# every small metal bar or plate read as an electronic blip; a glass left to ring and a coin read as what they are.
TICKS = {'wood': ('wood_block', 'C6', 0.05, 'a small dry wooden tick', 'a short knock on a little wooden block'),
         'glass': ('glass_clink', 'D6', None, 'a light tap on a glass', 'a small bright glass clink'),
         'metal': ('coin', 'E9', None, 'a small metallic tick, a coin clink', 'a coin tapped on a hard surface')}


def _tick(voice, note, damp):
    def fn(c, r):
        dur = None if damp is None and c.ring >= 0 else (damp or 0.25) * c.lerp('ring', 0.4, 3.0, 1.0, log=True)
        y = voices.play(voice, c.hz(note), vel=0.4 * c.db('energy', 7.0), dur=dur, r=r, vary=c.vary,
                        **c.axes('bright', 'hard', 'size', 'width'))
        return space.room(y, 'booth', amount=c.space, r=r)
    return fn


for _m, (_v, _n, _d, _a, _b) in TICKS.items():
    design(f'tick/{_m}', kind='tick', says=(_a, _b), uses=('tap', 'tick', 'click', 'detent', 'select', 'timer'),
           tags=(_m,), domains=('ui', 'game', 'motion'), dur=0.08 if _d else 0.3, pitch=_n, material=_m,
           axes=('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy'))(_tick(_v, _n, _d))

# the interface roles made of recorded things (a key, a switch, a page, knocks): what a plain interface request gets
from . import interface  # noqa: E402,F401
