"""Interface sounds made of real things: what a hand does with an object (a key, a switch, a page, a knuckle on wood,
a pop of the lips, a stick through the air), taken from the learned recordings (earshot.sounds.learned) and shaped
into the gestures an interface needs. No note, no instrument, no synthesised tone: a website that clicks, knocks and
rustles sounds like a well-made object, where a row of pitched notes sounds like a toy or a ringtone.

A pair is one thing done two ways, so a listener tells them apart without seeing the name: the switch on is smaller
and brighter, off bigger and duller; the page flicks forward and drops back darker and shorter; open lets air out
after a pen cap is pulled, close draws air in and ends on the same cap snapped on. A signal of two strikes keeps them
90 to 160 ms apart (two distinct touches, close enough to read as one gesture); the second of a positive one is
brighter (it goes up), of a negative one lower and duller.

Each design plays fixed members of learned designs (`_take(c, r, design, member)`), so a set always sounds like the
same objects; the seed gives a new take of them (fresh noise and a small variation, as every learned take). Nothing
here is modelled: a part that is not a recording of the store does not belong in this module.

    from earshot.sounds import registry
    y, meta = registry.render('notify/knock', seed=0)
    y, meta = registry.render('toggle/rocker_off', seed=1, size=0.3)
"""

import numpy as np

from ..synth import design as D, space
from . import learned
from .registry import AXES, design


def _take(c, r, sid, k, size=0.0, bright=0.0, energy=0.0):
    """a dry take (stereo) of member k of the learned design sid under the request's controls, moved by `size` and
    `energy` (axis units, -1 .. 1) and tilted by `bright` (dB per octave): one object of a gesture made a little
    bigger, softer or duller than the others. The take is scaled to a peak of 0.5 (the recordings of the store sit
    up to 20 dB apart), then by the nudged energy (6 dB per unit), so the gains of a gesture mean what they say"""
    e = learned.index()[sid]
    y = learned.take(e, c, r, k=k, room=False, nudge={'size': size, 'bright': bright, 'energy': energy})
    return y * (0.5 / (np.abs(y).max() + 1e-12)) * 10 ** (6.0 * (c.energy + energy) / 20)


def _cut(y, before=0.02, after=0.3, fout=0.06):
    """the part of a take from `before` s ahead of its strike to `after` s past it, faded at the cuts: a long
    recording (a page turn, a crumpled sheet) cut down to the one motion an interface needs"""
    on = D.onset(y)
    return D.span(y, max(0.0, on - before), on + after, fin=min(before, 0.02) or 0.004, fout=fout)


def _two(a, b, gap, gain=-2.0):
    """two strikes as one gesture: b `gap` s after a, `gain` dB against it"""
    return D.layer([(a, 0.0, 0.0), (b, gain, gap)])[0]


def _room(c, r, y):
    return space.room(y, c.room('room'), amount=c.space, r=r)


def _gap(c, base):
    """the time between two strikes: shorter for a harder, more urgent request"""
    return base * c.lerp('hard', 1.25, 0.8, 1.0)


# ---- touches
def hover_fingertip(c, r):
    """one fingertip on a wooden table, taken small and soft: the lightest touch of the set, a body without a
    click, so a pointer passing over many buttons never ticks"""
    return _room(c, r, _cut(_take(c, r, 'fingers/drumming', 1, size=-0.2, energy=-0.6), 0.01, 0.12, 0.05))


def hover_pad(c, r):
    """a trackpad pressed lightly, taken bigger and darker: a soft muted press"""
    return _room(c, r, _cut(_take(c, r, 'click/trackpad', 0, size=0.5, bright=-1.0, energy=-0.6), 0.01, 0.12, 0.05))


def type_key(c, r):
    """a full key of a desktop keyboard: its strike and the plastic of its case"""
    return _room(c, r, _cut(_take(c, r, 'keyboard/desktop', 0), 0.01, 0.22, 0.08))


# ---- switches: one rocker, on smaller and brighter, off bigger, duller and softer
def rocker_on(c, r):
    """a rocker switch flipped on: the switch taken smaller and brighter, a crisp upward click"""
    return _room(c, r, _cut(_take(c, r, 'switch/power_strip', 1, size=-0.5, bright=1.5, energy=0.2), 0.01, 0.16))


def rocker_off(c, r):
    """the same rocker flipped off: bigger, darker and softer, a duller click that settles"""
    return _room(c, r, _cut(_take(c, r, 'switch/power_strip', 1, size=0.7, bright=-1.2, energy=-0.3), 0.01, 0.2))


# ---- pages: forward a full flick of the sheet, back the same sheet darker and shorter
def page_next(c, r):
    """a page of a paperback flicked forward: the rustle of the sheet lifting and its snap as it lands"""
    return _room(c, r, _cut(_take(c, r, 'page/turn', 0, size=-0.3), 0.18, 0.28, 0.1))


def page_back(c, r):
    """the same page flicked back: bigger, darker and shorter, the snap without the long lift"""
    return _room(c, r, _cut(_take(c, r, 'page/turn', 0, size=0.5, bright=-1.5, energy=-0.2), 0.08, 0.2, 0.08))


def swipe_next(c, r):
    """a hand swept quickly through the air: a soft swipe"""
    return _room(c, r, _cut(_take(c, r, 'swipe/hand', 0, size=-0.3, bright=0.5), 0.03, 0.25, 0.1))


def swipe_back(c, r):
    """the same swipe played backwards and darker: the air drawn back in"""
    y = D.reverse(_cut(_take(c, r, 'swipe/hand', 0, size=0.3, bright=-0.8, energy=-0.2), 0.03, 0.2, 0.08))
    return _room(c, r, D.fade(y, 0.02, 0.0))


# ---- open and close: a small click and air going out; air drawn in and the cap snapped on
def air_open(c, r):
    """a small click with a soft breath of air leaving after it: a lid lifted"""
    click = _cut(_take(c, r, 'pen/cap', 0, size=0.4, bright=-1.0, energy=-0.3), 0.005, 0.08, 0.04)
    air = _cut(_take(c, r, 'whoosh/airy', 2, size=-0.2, energy=-0.2), 0.03, 0.3, 0.12)
    return _room(c, r, D.layer([(click, 0.0, 0.0), (air, -8.0, 0.025)])[0])


def air_close(c, r):
    """a soft breath of air drawn in that ends on a small snap: a pen cap snapped on, a lid set down"""
    air = D.reverse(_cut(_take(c, r, 'whoosh/airy', 2, size=0.2, bright=-0.8, energy=-0.2), 0.03, 0.28, 0.12))
    air = D.fade(air, 0.04, 0.0)
    snap = _cut(_take(c, r, 'pen/cap', 0, size=0.6, bright=-1.2, energy=-0.3), 0.005, 0.1, 0.05)
    return _room(c, r, D.layer([(air, -8.0, 0.0, D.end(air)), (snap, 0.0, -0.012)])[0])


def drawer_open(c, r):
    """a wooden drawer slid open: the runners and the soft knock at the end of the travel"""
    return _room(c, r, _cut(_take(c, r, 'drawer/wood', 1, size=-0.2), 0.3, 0.25, 0.1))


# ---- messages
def stick_send(c, r):
    """a thin stick swung fast through the air, small: a short swoosh going away"""
    return _room(c, r, _cut(_take(c, r, 'whoosh/stick', 4, size=-0.4, bright=0.4), 0.06, 0.26, 0.1))


def lips_receive(c, r):
    """a soft pop of the lips, round and short: something arriving"""
    return _room(c, r, _cut(_take(c, r, 'pop/mouth', 1, size=-0.2), 0.008, 0.26, 0.1))


def cork_receive(c, r):
    """a small cork popped out of a bottle, short: a hollow plop"""
    return _room(c, r, _cut(_take(c, r, 'cork/bottle', 1, size=-0.3), 0.006, 0.3, 0.12))


# ---- signals of two strikes
def knock_notify(c, r):
    """two light knuckle knocks on hollow wood: someone at the door, politely"""
    a = _take(c, r, 'knock/hollow', 0, size=-0.4, bright=0.6, energy=-0.2)
    b = _take(c, r, 'knock/hollow', 0, size=-0.4, bright=0.8, energy=-0.3)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.13), -1.5), 0.01, 0.4, 0.12))


def pop_notify(c, r):
    """two soft pops of the lips, the second a little higher: a light call"""
    a = _take(c, r, 'pop/mouth', 1, size=0.1)
    b = _take(c, r, 'pop/mouth', 1, size=-0.4, bright=0.5)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.14), -1.0), 0.01, 0.42, 0.12))


def knock_warning(c, r):
    """two firm knocks on a door, dull and close: a call to stop and look"""
    a = _take(c, r, 'door/knock', 0, size=0.2, bright=-0.5, energy=0.4)
    b = _take(c, r, 'door/knock', 0, size=0.2, bright=-0.5, energy=0.3)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.16), -0.5), 0.01, 0.45, 0.12))


def thud_error(c, r):
    """two low dull thuds, the second bigger and lower: a soft bag dropped twice, a firm no without a buzzer"""
    a = _cut(_take(c, r, 'bag/drop', 2, size=0.3, bright=-0.6, energy=0.2), 0.01, 0.12, 0.05)
    b = _cut(_take(c, r, 'bag/drop', 2, size=1.0, bright=-2.5), 0.01, 0.3, 0.12)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.15), -2.0), 0.01, 0.42, 0.12))


def click_success(c, r):
    """a small click and a brighter one right after it: a pen clicked, going up"""
    a = _take(c, r, 'pen/ballpoint', 1, size=0.6, bright=-0.8, energy=-0.2)
    b = _take(c, r, 'pen/ballpoint', 1, size=-0.5, bright=0.5)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.09), 0.0), 0.01, 0.3, 0.1))


def shutter_success(c, r):
    """a camera shutter fired: a quick two-part mechanical click"""
    return _room(c, r, _cut(_take(c, r, 'camera/shutter', 2), 0.06, 0.24, 0.08))


def stamp_complete(c, r):
    """an office stapler pressed down hard on a desk with a book set down under it: a full, round stamp"""
    a = _cut(_take(c, r, 'stapler/desk', 0, size=0.2), 0.01, 0.22, 0.08)
    b = _cut(_take(c, r, 'book/place', 0, size=0.4, bright=-1.0, energy=-0.3), 0.01, 0.25, 0.1)
    return _room(c, r, D.layer([(a, 0.0, 0.0), (b, -5.0, 0.004)])[0])


def latch_complete(c, r):
    """a latch snapped shut and a soft pop after it: something closed and done"""
    a = _cut(_take(c, r, 'lock/latch', 0, size=0.3, bright=-0.6), 0.005, 0.12, 0.05)
    b = _cut(_take(c, r, 'pop/mouth', 1, size=-0.2), 0.008, 0.26, 0.1)
    return _room(c, r, _two(a, b, _gap(c, 0.11), -1.0))


def pop_like(c, r):
    """a small bright pop of the lips over a fingertip on wood: a light, quick pop with a soft body under it"""
    a = _take(c, r, 'pop/mouth', 1, size=-0.6, bright=1.5)
    b = _take(c, r, 'fingers/drumming', 1, size=0.3, bright=-1.0, energy=-0.4)
    return _room(c, r, _cut(D.layer([(a, 0.0, 0.0), (b, -6.0, 0.002)])[0], 0.008, 0.15, 0.06))


def crumple_delete(c, r):
    """a sheet of paper crumpled quickly in the hand"""
    return _room(c, r, _cut(_take(c, r, 'paper/ball', 0, size=-0.2), 0.04, 0.45, 0.15))


def swish_delete(c, r):
    """a quick swish into a soft thud: something thrown into a bin"""
    a = _cut(_take(c, r, 'swish/fast', 0, size=-0.2, energy=-0.2), 0.04, 0.16, 0.06)
    b = _cut(_take(c, r, 'bag/drop', 2, size=-0.3, bright=-0.5, energy=-0.2), 0.01, 0.25, 0.1)
    return _room(c, r, D.layer([(a, -7.0, 0.0, D.end(a)), (b, 0.0, -0.01)])[0])


# id: (function, kind, says, uses, dur). DEFAULTS are what a plain request for a job gets (prior 1.25); the others
# (prior 0.9) answer when their own words are asked for ("a camera shutter for success", "two pops")
ROLES = {
    'hover/fingertip': (hover_fingertip, 'tap', ('a single soft fingertip tap on a wooden table',
                                                 'a very light muted touch'),
                        ('hover', 'focus', 'highlight', 'mouse over'), 0.15),
    'hover/pad': (hover_pad, 'tap', ('a trackpad pressed lightly, soft and muted', 'a very soft short press'),
                  ('hover', 'focus', 'highlight'), 0.15),
    'type/key': (type_key, 'click', ('a key of a desktop keyboard pressed, a full plastic clack',
                                     'one keystroke on a computer keyboard'),
                 ('type', 'typing', 'keystroke', 'key press', 'input'), 0.25),
    'toggle/rocker_on': (rocker_on, 'click', ('a small rocker switch flipped on, a crisp bright click',
                                              'a light switch clicked on'),
                         ('switch on', 'toggle on', 'toggle', 'enable', 'check'), 0.2),
    'toggle/rocker_off': (rocker_off, 'click', ('a rocker switch flipped off, a lower duller click',
                                                'a switch clicked off'),
                          ('switch off', 'toggle off', 'toggle', 'disable', 'uncheck'), 0.22),
    'next/page': (page_next, 'whoosh', ('a page of a book flicked forward, a short paper rustle and snap',
                                        'a sheet of paper turned quickly'),
                  ('next', 'forward', 'next page', 'continue', 'swipe', 'page turn'), 0.48),
    'back/page': (page_back, 'whoosh', ('a page flicked back, a short darker paper flick',
                                        'a sheet of paper turned back'),
                  ('back', 'previous', 'go back', 'return', 'undo'), 0.3),
    'next/swipe': (swipe_next, 'whoosh', ('a hand swept quickly through the air, a soft swipe',
                                          'a short soft swish of air'),
                   ('next', 'swipe', 'forward', 'slide'), 0.3),
    'back/swipe': (swipe_back, 'whoosh', ('a soft swish of air drawn back in, darker', 'a short reversed swish'),
                   ('back', 'previous', 'swipe back'), 0.25),
    'open/air': (air_open, 'whoosh', ('a small click and a soft breath of air after it, a lid lifted',
                                      'a light click that opens into air'),
                 ('open', 'expand', 'show', 'menu open', 'modal'), 0.35),
    'close/air': (air_close, 'whoosh', ('a soft breath of air drawn in that ends on a small plastic snap',
                                        'air sucked in and a lid clicked shut'),
                  ('close', 'collapse', 'dismiss', 'hide'), 0.32),
    'open/drawer': (drawer_open, 'noise', ('a wooden drawer slid open, its runners and a soft knock',
                                           'a drawer pulled out'),
                    ('open', 'expand', 'drawer open'), 0.55),
    'send/swoosh': (stick_send, 'whoosh', ('a thin stick swung fast through the air, a short swoosh',
                                           'a quick swoosh going away'),
                    ('send', 'message sent', 'submit', 'post', 'share', 'upload'), 0.32),
    'receive/pop': (lips_receive, 'pop', ('a soft round pop of the lips', 'a short soft plop'),
                    ('receive', 'message received', 'incoming', 'new message'), 0.27),
    'receive/cork': (cork_receive, 'pop', ('a small cork popped out of a bottle, a short hollow plop',
                                           'a soft hollow pop'),
                     ('receive', 'message received', 'incoming'), 0.31),
    'notify/knock': (knock_notify, 'hit', ('two light knuckle knocks on hollow wood', 'a soft double knock'),
                     ('notify', 'notification', 'reminder', 'message received', 'ping'), 0.55),
    'notify/pop': (pop_notify, 'pop', ('two soft pops of the lips, the second a little higher',
                                       'a light double pop'),
                   ('notify', 'notification', 'ping', 'mention'), 0.55),
    'warning/knock': (knock_warning, 'hit', ('two firm dull knocks on a door, close', 'a firm double knock'),
                      ('warning', 'caution', 'attention', 'alert'), 0.6),
    'error/thud': (thud_error, 'thud', ('two low dull thuds, the second lower', 'a soft bag dropped twice'),
                   ('error', 'failed', 'invalid', 'denied', 'wrong'), 0.57),
    'success/click': (click_success, 'click', ('a small click and a brighter click right after it, a pen clicked',
                                               'a quick two-part click going up'),
                      ('success', 'confirm', 'saved', 'done', 'check mark'), 0.4),
    'success/shutter': (shutter_success, 'click', ('a camera shutter fired, a quick two-part mechanical click',
                                                   'a crisp mechanical snap'),
                        ('success', 'confirm', 'saved'), 0.3),
    'complete/stamp': (stamp_complete, 'hit', ('a stapler pressed down hard on a desk, a full round stamp',
                                               'a firm mechanical press with a soft thump under it'),
                       ('complete', 'success', 'done', 'finished', 'approved', 'task done'), 0.3),
    'complete/latch': (latch_complete, 'click', ('a latch snapped shut and a soft pop after it',
                                                 'a metal click followed by a round pop'),
                       ('complete', 'success', 'done', 'closed'), 0.45),
    'like/pop': (pop_like, 'pop', ('a small bright pop of the lips with a soft tap under it',
                                   'a light quick pop with a soft body'),
                 ('like', 'favorite', 'heart icon', 'upvote'), 0.24),
    'delete/crumple': (crumple_delete, 'noise', ('a sheet of paper crumpled quickly in the hand',
                                                 'a short paper crunch'),
                       ('delete', 'remove', 'trash', 'discard', 'clear'), 0.5),
    'delete/swish': (swish_delete, 'thud', ('a quick swish ending in a soft thud, something thrown in a bin',
                                            'a short swish and a dull drop'),
                     ('delete', 'remove', 'trash', 'discard'), 0.42),
}
# the default design of each job (the one a plain request and the ui set get)
DEFAULTS = ('hover/fingertip', 'type/key', 'toggle/rocker_on', 'toggle/rocker_off', 'next/page', 'back/page',
            'open/air', 'close/air', 'send/swoosh', 'receive/pop', 'notify/knock', 'warning/knock', 'error/thud',
            'success/click', 'complete/stamp', 'delete/crumple', 'like/pop')

# audit findings that are the design: two strikes are two attacks (and two copies of one strike a short comb in the
# tail), a swish or breath before a click or thud is its lead-in
ALLOWS = {'notify/knock': ('stacked', 'ring'), 'notify/pop': ('stacked', 'ring'), 'warning/knock': ('stacked', 'ring'),
          'error/thud': ('stacked', 'ring'), 'success/click': ('stacked', 'ring'), 'complete/latch': ('stacked',),
          'complete/stamp': ('stacked',), 'open/air': ('stacked',), 'close/air': ('pre_swell', 'stacked'),
          'delete/swish': ('pre_swell', 'stacked'), 'back/swipe': ('pre_swell',), 'next/page': ('pre_swell',)}

# the learned designs each one is made of: its axes are theirs (a take reads what its recording answers to), the room's
# and, for two strikes, hard (a harder request brings them closer together)
PARTS = {'hover/fingertip': ('fingers/drumming',), 'hover/pad': ('click/trackpad',), 'type/key': ('keyboard/desktop',),
         'toggle/rocker_on': ('switch/power_strip',), 'toggle/rocker_off': ('switch/power_strip',),
         'next/page': ('page/turn',), 'back/page': ('page/turn',), 'next/swipe': ('swipe/hand',),
         'back/swipe': ('swipe/hand',), 'open/air': ('pen/cap', 'whoosh/airy'),
         'close/air': ('whoosh/airy', 'pen/cap'),
         'open/drawer': ('drawer/wood',), 'send/swoosh': ('whoosh/stick',), 'receive/pop': ('pop/mouth',),
         'receive/cork': ('cork/bottle',), 'notify/knock': ('knock/hollow',), 'notify/pop': ('pop/mouth',),
         'warning/knock': ('door/knock',), 'error/thud': ('bag/drop',), 'success/click': ('pen/ballpoint',),
         'success/shutter': ('camera/shutter',), 'complete/stamp': ('stapler/desk', 'book/place'),
         'complete/latch': ('lock/latch', 'pop/mouth'), 'like/pop': ('pop/mouth', 'fingers/drumming'),
         'delete/crumple': ('paper/ball',),
         'delete/swish': ('swish/fast', 'bag/drop')}
TWO = ('notify/knock', 'notify/pop', 'warning/knock', 'error/thud', 'success/click', 'complete/latch')


def _axes(sid):
    own = set().union(*(learned.index()[p]['axes'] for p in PARTS[sid])) | {'space', 'energy'}
    return tuple(a for a in AXES if a in own or (a == 'hard' and sid in TWO))


for _id, (_fn, _kind, _says, _uses, _dur) in ROLES.items():
    design(_id, kind=_kind, says=_says, uses=_uses, tags=('real', 'interface'), domains=('ui',), dur=_dur,
           axes=_axes(_id), allows=ALLOWS.get(_id, ()), prior=1.25 if _id in DEFAULTS else 0.9)(_fn)
