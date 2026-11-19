"""Ready-made sets: the sounds a kind of product usually needs, by name, so one call makes all of them as one family.

    earshot make --set ui --theme glass -o public/sounds
    earshot make --set arcade -o assets/sfx --format ogg

A set is a list of (file name, design id). The theme decides the voice, key and room of the tonal ones; a spec may
add sounds, replace one by name, or change any of them (earshot.make.make_set). `seeds` pins the take of a name (for
a learned design: the member, by its place in the design's walk, earshot.sounds.learned); a name without one gets
the next take of its design (the first name of a design its best recording). `quiet` names sit at the set's lowest
level (make_set balances a set's one-shots by their loudest 50 ms). The lists follow the inventories interface and
game audio teams work from; a product rarely needs more than a dozen distinct sounds, and fewer is better than every
event making noise.
"""
SETS = {
    # every role is a real thing a hand does (earshot.sounds.interface, made of learned recordings): a key, a page,
    # a drawer, a cork, knocks, a camera's shutter, scissors, a stick through the air. A row of pitched notes on a
    # marimba or a bell reads as a toy or a ringtone on a website, so the figures (earshot.sounds.ui) play only when a
    # theme is named: `themed` is the set then. tap is the fullest mechanical key (keyboard/mechanical, first in its
    # walk); the pairs are two actions of one thing (the key going down and coming up for the toggles; the page
    # forward, and back darker and the other way; a drawer pulled open and pushed shut)
    'ui': dict(says='a product interface: app, website, dashboard', theme=None, seeds={'tap': 0},
               quiet=('hover',), sounds=(
        ('tap', 'keyboard/mechanical'), ('hover', 'hover/fingertip'), ('toggle_on', 'toggle/key_down'),
        ('toggle_off', 'toggle/key_up'), ('type', 'type/mechanical'), ('back', 'back/turn'),
        ('next', 'next/page'), ('open', 'open/drawer'), ('close', 'close/drawer'), ('send', 'send/swoosh'),
        ('receive', 'receive/cork'), ('notify', 'notify/knock'), ('success', 'success/shutter'),
        ('complete', 'complete/shutter'), ('warning', 'warning/knock'), ('error', 'error/knock'),
        ('delete', 'delete/snip')),
               themed=(
        ('tap', 'tap/tone'), ('hover', 'hover/tone'), ('toggle_on', 'toggle/up'), ('toggle_off', 'toggle/down'),
        ('type', 'tap/tone'), ('back', 'back/step'), ('next', 'next/step'), ('open', 'open/lift'),
        ('close', 'close/drop'), ('send', 'send/lift'), ('receive', 'receive/drop'), ('notify', 'notify/pair'),
        ('success', 'success/rise'), ('complete', 'success/triad'), ('warning', 'warning/pulse'),
        ('error', 'error/fall'), ('delete', 'delete/drop'))),
    'game': dict(says='a modern game: menus, rewards, jingles and the common actions', theme='tine', sounds=(
        ('menu_move', 'menu/move'), ('menu_select', 'menu/select'), ('menu_back', 'menu/back'),
        ('coin', 'coin/bright'), ('pickup', 'pickup/blip'), ('gem', 'gem/sparkle'), ('heal', 'heal/chime'),
        ('power_up', 'power_up/rise'), ('power_down', 'power_down/fall'), ('level_up', 'level_up/fanfare'),
        ('checkpoint', 'checkpoint/chime'), ('secret', 'secret/reveal'), ('win', 'win/fanfare'),
        ('lose', 'lose/descend'), ('jump', 'jump/soft'), ('land', 'land/soft'), ('step', 'step/grass'),
        ('hit', 'hit/punch'), ('hurt', 'hurt/thud'), ('explosion', 'explosion/small'), ('splash', 'splash/small'),
        ('door', 'door/open'))),
    'arcade': dict(says='a retro game in chip sound', theme='retro', sounds=(
        ('menu_move', 'menu/chip'), ('menu_select', 'select/chip'), ('menu_back', 'back/chip'),
        ('coin', 'coin/chip'), ('pickup', 'pickup/chip'), ('jump', 'jump/chip'), ('laser', 'laser/chip'),
        ('explosion', 'explosion/chip'), ('hurt', 'hurt/chip'), ('power_up', 'power_up/chip'),
        ('power_down', 'power_down/chip'), ('one_up', 'one_up/chip'), ('level_up', 'level_up/chip'),
        ('win', 'win/chip'), ('lose', 'lose/chip'), ('text', 'text/chip'))),
    # the kinds of effect professional motion films use (docs/motion-sfx-study.md): passes of air, short low hits,
    # thumps, ticks and clicks, each a small real action cut to the role (earshot.sounds.motion, "named recordings":
    # a door falling shut, a mechanical key, a page sliding, a stick and a broom through the air, a heavy bag dropped,
    # a car door, a breath of air, a sword drawn); no note, bell or synthesised tone (the film's music has those)
    'film': dict(says='a product film or motion-design piece: moves, arrivals, hits and a logo', theme=None, sounds=(
        ('whoosh', 'whoosh/fast'), ('whoosh_soft', 'whoosh/sweep'), ('slide', 'slide/paper'),
        ('pop', 'thump/door'), ('tick', 'tick/key'), ('click', 'click/cursor'), ('type', 'type/mechanical'),
        ('shutter', 'success/shutter'), ('hit', 'title/bag'), ('impact', 'impact/car'), ('drop', 'drop/sack'),
        ('riser', 'rise/breath'), ('swell', 'rise/hit'), ('glint', 'shine/blade'), ('success', 'success/click'),
        ('sting', 'sting/reveal'), ('logo', 'logo/bag'))),
}


def get(name, theme=None):
    """the set by name; with a theme named, a set that has a `themed` list plays that one (the tonal figures,
    which the theme puts in one key, voice and room)"""
    if name not in SETS:
        raise ValueError(f'unknown set {name!r}: one of {", ".join(SETS)}')
    base = SETS[name]
    if theme and base.get('themed'):
        return dict(base, sounds=base['themed'], seeds={}, quiet=())
    return base
