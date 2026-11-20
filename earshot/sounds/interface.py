"""Interface sounds made of real things: what a hand does with an object (a key, a camera's shutter, a page, a drawer,
a cork, a knuckle on wood, scissors, a stick through the air), taken from the learned recordings
(earshot.sounds.learned) and shaped into the gestures an interface needs. No note, no instrument, no synthesised
tone: a website that clicks, knocks and rustles sounds like a well-made object, where a row of pitched notes sounds
like a toy or a ringtone.

A pair is two actions of one thing, so a listener tells them apart without seeing the name: a key goes down (the
first half of a keystroke, bigger and darker) and comes up (the second half, smaller and brighter); the page flicks
forward and is turned back darker, played the other way; a drawer is pulled open and pushed shut. A signal of two
strikes keeps them 90 to 180 ms apart (two distinct touches, close enough to read as one gesture); the second of a
positive one is brighter (it goes up), of a negative one lower and duller. The designs the `ui` set plays are
DEFAULTS; the others take a role by name.

Each design plays fixed members of learned designs (`_take(c, r, design, member)`), so a set always sounds like the
same objects; the seed gives a new take of them (fresh noise and a small variation, as every learned take). Nothing
here is modelled: a part that is not a recording of the store does not belong in this module.

    from earshot.sounds import registry
    y, meta = registry.render('notify/knock', seed=0)
    y, meta = registry.render('toggle/key_up', seed=1, size=0.3)
"""

import numpy as np

from ..forge import dsp
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


def _halves(y, apart=0.03):
    """a take of two strikes (a shutter opening and closing, a key going down and coming up) cut at its quietest
    moment between the two loudest strikes, which lie at least `apart` s from each other: (the first, the second),
    each faded at the cut"""
    k = dsp.n_of(0.004)
    e = np.convolve(np.mean(y ** 2, axis=1), np.ones(k) / k, 'same')
    a, g = int(np.argmax(e)), dsp.n_of(apart)
    rest = e.copy()
    rest[max(0, a - g):a + g] = 0.0
    a, b = sorted((a, int(np.argmax(rest))))
    cut = int(np.clip(a + int(np.argmin(e[a:b + 1])), g, max(g, len(y) - g)))
    return D.fade(y[:cut], 0.0, 0.01), D.fade(y[cut:], 0.004, 0.0)


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


# ---- switches: one rocker, on smaller and brighter, off bigger, duller and softer
def rocker_on(c, r):
    """a rocker switch flipped on: the switch taken smaller and brighter, a crisp upward click"""
    return _room(c, r, _cut(_take(c, r, 'switch/power_strip', 1, size=-0.5, bright=1.5, energy=0.2), 0.01, 0.16))


def rocker_off(c, r):
    """the same rocker flipped off: bigger, darker and softer, a duller click that settles"""
    return _room(c, r, _cut(_take(c, r, 'switch/power_strip', 1, size=0.7, bright=-1.2, energy=-0.3), 0.01, 0.2))


# ---- pages and a drawer
def page_next(c, r):
    """a page of a paperback flicked forward: the rustle of the sheet lifting and its snap as it lands"""
    return _room(c, r, _cut(_take(c, r, 'page/turn', 0, size=-0.3), 0.18, 0.28, 0.1))


def drawer_open(c, r):
    """a wooden drawer slid open: the runners and the soft knock at the end of the travel"""
    return _room(c, r, _cut(_take(c, r, 'drawer/wood', 1, size=-0.2), 0.3, 0.25, 0.1))


# ---- messages
def stick_send(c, r):
    """a thin stick swung fast through the air, small: a short swoosh going away"""
    return _room(c, r, _cut(_take(c, r, 'whoosh/stick', 4, size=-0.4, bright=0.4), 0.06, 0.26, 0.1))


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


def click_success(c, r):
    """a small click and a brighter one right after it: a pen clicked, going up"""
    a = _take(c, r, 'pen/ballpoint', 1, size=0.6, bright=-0.8, energy=-0.2)
    b = _take(c, r, 'pen/ballpoint', 1, size=-0.5, bright=0.5)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.09), 0.0), 0.01, 0.3, 0.1))


def shutter_success(c, r):
    """a camera shutter fired: a quick two-part mechanical click"""
    return _room(c, r, _cut(_take(c, r, 'camera/shutter', 2), 0.06, 0.24, 0.08))


def pop_like(c, r):
    """a small bright pop of the lips over a fingertip on wood: a light, quick pop with a soft body under it"""
    a = _take(c, r, 'pop/mouth', 1, size=-0.6, bright=1.5)
    b = _take(c, r, 'fingers/drumming', 1, size=0.3, bright=-1.0, energy=-0.4)
    return _room(c, r, _cut(D.layer([(a, 0.0, 0.0), (b, -6.0, 0.002)])[0], 0.008, 0.15, 0.06))


# ---- the same jobs from a key, a shutter, a page, a drawer, a door, knocks, a knife on a board, a broom, scissors
def key_type(c, r):
    """the mechanical key of a tap, pressed more softly: one keystroke among many"""
    return _room(c, r, _cut(_take(c, r, 'keyboard/mechanical', 1, size=-0.15, bright=-0.4, energy=-0.5), 0.012, 0.22,
                            0.08))


def key_light(c, r):
    """the same key taken smaller, brighter and lighter still: a quick light keystroke"""
    return _room(c, r, _cut(_take(c, r, 'keyboard/mechanical', 1, size=-0.5, bright=0.3, energy=-0.7), 0.012, 0.2,
                            0.08))


def key_down(c, r):
    """a mechanical key going down: the first half of a keystroke, the key hitting its bed"""
    return _room(c, r, _halves(_take(c, r, 'keyboard/mechanical', 1, size=0.4, bright=-0.6))[0])


def key_up(c, r):
    """the same key coming back up: the second half of the keystroke, lighter and brighter"""
    return _room(c, r, _halves(_take(c, r, 'keyboard/mechanical', 1, size=-0.4, bright=0.8))[1])


def shutter_open(c, r):
    """the shutter of a film camera opening at a slow speed: the first of its two clicks"""
    return _room(c, r, _halves(_take(c, r, 'camera/shutter', 0, size=-0.2, bright=0.3))[0])


def shutter_shut(c, r):
    """the same shutter closing: the second of its two clicks, bigger and darker"""
    return _room(c, r, _halves(_take(c, r, 'camera/shutter', 0, size=0.4, bright=-0.8))[1])


def turn_back(c, r):
    """a page turned back: the turn of the page played the other way and darker, its snap first"""
    y = D.reverse(_cut(_take(c, r, 'page/turn', 0, size=0.5, bright=-1.5), 0.18, 0.26, 0.1))
    return _room(c, r, D.fade(y, 0.02, 0.0))


def leaf_back(c, r):
    """another page of the same book turned, bigger and much darker: a heavier sheet falling back"""
    return _room(c, r, _cut(_take(c, r, 'page/turn', 2, size=0.8, bright=-2.0), 0.05, 0.42, 0.15))


def drawer_close(c, r):
    """a wooden drawer pushed shut, short: the end of its slide and the knock as it closes"""
    return _room(c, r, _cut(_take(c, r, 'drawer/wood', 0, size=-0.1, bright=0.5, energy=0.2), 0.3, 0.3, 0.12))


def door_close(c, r):
    """an apartment door closing, short: the door meeting its frame and the latch"""
    return _room(c, r, _cut(_take(c, r, 'door/shut', 1, size=-0.3), 0.08, 0.45, 0.15))


def shutter_complete(c, r):
    """a bigger camera's shutter fired, full and slow: a longer mechanical clack than the click of a success"""
    return _room(c, r, _cut(_take(c, r, 'camera/shutter', 3, size=0.3, bright=-0.4), 0.04, 0.5, 0.14))


def chop_complete(c, r):
    """one firm chop of a knife on a wooden board: cut and done"""
    return _room(c, r, _cut(_take(c, r, 'knife/chop', 0, size=0.2, energy=0.3), 0.02, 0.42, 0.14))


def knock_error(c, r):
    """two heavy knocks on a door that fall: lower than a warning's, the second lower than the first"""
    a = _take(c, r, 'door/knock', 3, size=0.5, bright=-0.3, energy=0.4)
    b = _take(c, r, 'door/knock', 3, size=1.0, bright=-0.6, energy=0.4)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.18), 0.0), 0.01, 0.52, 0.14))


def chop_error(c, r):
    """two firm chops of a knife on a board, the second bigger and lower: cut off, twice"""
    a = _cut(_take(c, r, 'knife/chop', 0, size=0.1, energy=0.3), 0.01, 0.2, 0.06)
    b = _take(c, r, 'knife/chop', 0, size=0.8, bright=-0.6, energy=0.3)
    return _room(c, r, _cut(_two(a, b, _gap(c, 0.17), 0.0), 0.01, 0.58, 0.15))


def sweep_delete(c, r):
    """one short stroke of a straw broom: swept away"""
    return _room(c, r, _cut(_take(c, r, 'broom/sweep', 0, size=-0.2), 0.03, 0.36, 0.15))


def snip_delete(c, r):
    """one cut of a pair of scissors: the blades sliding shut and their snip"""
    return _room(c, r, _cut(_take(c, r, 'scissors/cut', 0, size=-0.1), 0.26, 0.24, 0.1))


# id: (function, kind, says, uses, dur). DEFAULTS are what the ui set plays and PLAIN what a plain request for a job
# gets first (below)
ROLES = {
    'hover/fingertip': (hover_fingertip, 'tap', ('a single soft fingertip tap on a wooden table',
                                                 'a very light muted touch'),
                        ('hover', 'focus', 'highlight', 'mouse over'), 0.15),
    'toggle/rocker_on': (rocker_on, 'click', ('a small rocker switch flipped on, a crisp bright click',
                                              'a light switch clicked on'),
                         ('switch on', 'toggle on', 'toggle', 'enable', 'check'), 0.2),
    'toggle/rocker_off': (rocker_off, 'click', ('a rocker switch flipped off, a lower duller click',
                                                'a switch clicked off'),
                          ('switch off', 'toggle off', 'toggle', 'disable', 'uncheck'), 0.22),
    'next/page': (page_next, 'whoosh', ('a page of a book flicked forward, a short paper rustle and snap',
                                        'a sheet of paper turned quickly'),
                  ('next', 'forward', 'next page', 'continue', 'swipe', 'page turn'), 0.48),
    'open/drawer': (drawer_open, 'noise', ('a wooden drawer slid open, its runners and a soft knock',
                                           'a drawer pulled out'),
                    ('open', 'expand', 'show', 'menu open', 'modal', 'drawer open'), 0.55),
    'send/swoosh': (stick_send, 'whoosh', ('a thin stick swung fast through the air, a short swoosh',
                                           'a quick swoosh going away'),
                    ('send', 'message sent', 'submit', 'post', 'share', 'upload'), 0.32),
    'receive/cork': (cork_receive, 'pop', ('a small cork popped out of a bottle, a short hollow plop',
                                           'a soft hollow pop'),
                     ('receive', 'message received', 'incoming', 'new message'), 0.31),
    'notify/knock': (knock_notify, 'hit', ('two light knuckle knocks on hollow wood', 'a soft double knock'),
                     ('notify', 'notification', 'reminder', 'message received', 'ping'), 0.55),
    'notify/pop': (pop_notify, 'pop', ('two soft pops of the lips, the second a little higher',
                                       'a light double pop'),
                   ('notify', 'notification', 'ping', 'mention'), 0.55),
    'warning/knock': (knock_warning, 'hit', ('two firm dull knocks on a door, close', 'a firm double knock'),
                      ('warning', 'caution', 'attention', 'alert'), 0.6),
    'success/click': (click_success, 'click', ('a small click and a brighter click right after it, a pen clicked',
                                               'a quick two-part click going up'),
                      ('success', 'confirm', 'saved', 'done', 'check mark'), 0.4),
    'success/shutter': (shutter_success, 'click', ('a camera shutter fired, a quick two-part mechanical click',
                                                   'a crisp mechanical snap'),
                        ('success', 'confirm', 'saved', 'done', 'check mark'), 0.3),
    'like/pop': (pop_like, 'pop', ('a small bright pop of the lips with a soft tap under it',
                                   'a light quick pop with a soft body'),
                 ('like', 'favorite', 'heart icon', 'upvote'), 0.24),
    'type/mechanical': (key_type, 'click', ('a key of a mechanical keyboard pressed softly, a full plastic clack',
                                            'one soft keystroke on a mechanical keyboard'),
                        ('type', 'typing', 'keystroke', 'key press', 'input'), 0.25),
    'type/light': (key_light, 'click', ('a key of a mechanical keyboard touched lightly, a small bright clack',
                                        'one quick light keystroke'),
                   ('type', 'typing', 'keystroke'), 0.25),
    'toggle/key_down': (key_down, 'click', ('a mechanical key going down and hitting its bed, a full low clack',
                                            'a key pressed down and held'),
                        ('switch on', 'toggle on', 'toggle', 'enable', 'check'), 0.2),
    'toggle/key_up': (key_up, 'click', ('a mechanical key let go and coming back up, a lighter brighter click',
                                        'a held key released'),
                      ('switch off', 'toggle off', 'toggle', 'disable', 'uncheck'), 0.2),
    'toggle/shutter_open': (shutter_open, 'click', ('the shutter of a film camera opening, the first of two clicks',
                                                    'a mechanical click that opens'),
                            ('switch on', 'toggle on', 'toggle', 'enable', 'check'), 0.2),
    'toggle/shutter_shut': (shutter_shut, 'click', ('the shutter of a film camera closing, a bigger darker clack',
                                                    'a mechanical clack that shuts'),
                            ('switch off', 'toggle off', 'toggle', 'disable', 'uncheck'), 0.2),
    'back/turn': (turn_back, 'whoosh', ('a page turned back, the turn played the other way and darker',
                                        'a paper snap and a rustle settling after it'),
                  ('back', 'previous', 'go back', 'return', 'undo'), 0.45),
    'back/leaf': (leaf_back, 'whoosh', ('a heavier page of a book falling back, darker',
                                        'a sheet of paper turned back slowly'),
                  ('back', 'previous', 'go back'), 0.47),
    'close/drawer': (drawer_close, 'noise', ('a wooden drawer pushed shut, the end of its slide and a knock',
                                             'a drawer closed'),
                     ('close', 'collapse', 'dismiss', 'hide', 'drawer close'), 0.5),
    'close/door': (door_close, 'hit', ('an apartment door closing, short, the door in its frame and the latch',
                                       'a door shut'),
                   ('close', 'collapse', 'dismiss'), 0.53),
    'complete/shutter': (shutter_complete, 'click', ('a big camera shutter fired, a full slow mechanical clack',
                                                     'a long two-part mechanical clack'),
                         ('complete', 'success', 'done', 'finished', 'approved', 'task done'), 0.54),
    'complete/chop': (chop_complete, 'hit', ('one firm chop of a knife on a wooden board', 'a single firm chop'),
                      ('complete', 'done', 'finished'), 0.44),
    'error/knock': (knock_error, 'hit', ('two heavy low knocks on a door, the second lower than the first',
                                         'a heavy double knock that falls'),
                    ('error', 'failed', 'invalid', 'denied', 'wrong'), 0.53),
    'error/chop': (chop_error, 'hit', ('two firm chops of a knife on a board, the second lower',
                                       'a hard double chop that falls'),
                   ('error', 'failed', 'invalid', 'denied'), 0.59),
    'delete/sweep': (sweep_delete, 'noise', ('one short stroke of a straw broom on a floor', 'a short dry sweep'),
                     ('delete', 'remove', 'trash', 'discard', 'clear'), 0.39),
    'delete/snip': (snip_delete, 'click', ('one cut of a pair of scissors, the blades sliding shut and a snip',
                                           'a short metal snip'),
                    ('delete', 'remove', 'trash', 'discard'), 0.5),
}
# the design of each job the ui set plays (earshot.sounds.sets; its tap is the learned key itself)
DEFAULTS = ('hover/fingertip', 'type/mechanical', 'toggle/key_down', 'toggle/key_up', 'next/page',
            'back/turn', 'open/drawer', 'close/drawer', 'send/swoosh', 'receive/cork', 'notify/knock',
            'warning/knock', 'error/knock', 'success/shutter', 'complete/shutter', 'delete/snip')
# the design a plain request for a job is answered with first (prior 1.25; the others, at 0.9, answer when their own
# words are asked for: "a camera shutter for success", "two pops")
PLAIN = DEFAULTS + ('like/pop',)

# audit findings that are the design: two strikes are two attacks (and two copies of one strike a short comb in the
# tail), a swish or breath before a click or thud is its lead-in
ALLOWS = {'notify/knock': ('stacked', 'ring'), 'notify/pop': ('stacked', 'ring'), 'warning/knock': ('stacked', 'ring'),
          'success/click': ('stacked', 'ring'), 'next/page': ('pre_swell',),
          'error/knock': ('stacked', 'ring'), 'error/chop': ('stacked', 'ring'), 'close/drawer': ('pre_swell',),
          'close/door': ('pre_swell', 'stacked'), 'back/turn': ('pre_swell',), 'delete/snip': ('pre_swell',)}

# the learned designs each one is made of: its axes are theirs (a take reads what its recording answers to), the room's
# and, for two strikes, hard (a harder request brings them closer together)
PARTS = {'hover/fingertip': ('fingers/drumming',),
         'toggle/rocker_on': ('switch/power_strip',), 'toggle/rocker_off': ('switch/power_strip',),
         'next/page': ('page/turn',),
         'open/drawer': ('drawer/wood',), 'send/swoosh': ('whoosh/stick',),
         'receive/cork': ('cork/bottle',), 'notify/knock': ('knock/hollow',), 'notify/pop': ('pop/mouth',),
         'warning/knock': ('door/knock',), 'success/click': ('pen/ballpoint',),
         'success/shutter': ('camera/shutter',), 'like/pop': ('pop/mouth', 'fingers/drumming'),
         'type/mechanical': ('keyboard/mechanical',), 'type/light': ('keyboard/mechanical',),
         'toggle/key_down': ('keyboard/mechanical',), 'toggle/key_up': ('keyboard/mechanical',),
         'toggle/shutter_open': ('camera/shutter',), 'toggle/shutter_shut': ('camera/shutter',),
         'back/turn': ('page/turn',), 'back/leaf': ('page/turn',), 'close/drawer': ('drawer/wood',),
         'close/door': ('door/shut',), 'complete/shutter': ('camera/shutter',),
         'complete/chop': ('knife/chop',), 'error/knock': ('door/knock',), 'error/chop': ('knife/chop',),
         'delete/sweep': ('broom/sweep',), 'delete/snip': ('scissors/cut',)}
TWO = ('notify/knock', 'notify/pop', 'warning/knock', 'success/click', 'error/knock', 'error/chop')


def _axes(sid):
    own = set().union(*(learned.index()[p]['axes'] for p in PARTS[sid])) | {'space', 'energy'}
    return tuple(a for a in AXES if a in own or (a == 'hard' and sid in TWO))


for _id, (_fn, _kind, _says, _uses, _dur) in ROLES.items():
    design(_id, kind=_kind, says=_says, uses=_uses, tags=('real', 'interface'), domains=('ui',), dur=_dur,
           axes=_axes(_id), allows=ALLOWS.get(_id, ()), prior=1.25 if _id in PLAIN else 0.9)(_fn)
