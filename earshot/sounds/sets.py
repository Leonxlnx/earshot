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
    # every role is a real thing a hand does (earshot.sounds.interface, made of learned recordings): a key, a switch,
    # a page, knocks, a pop, a stick through the air, a stapler, paper. A row of pitched notes on a marimba or a bell
    # reads as a toy or a ringtone on a website, so the figures (earshot.sounds.ui) play only when a theme is named:
    # `themed` is the set then. tap is the fullest mechanical key (keyboard/mechanical, first in its walk); the pairs
    # are one thing done two ways (the switch on smaller and brighter, off bigger and duller; the page forward, back
    # darker and shorter; open a click and air leaving, close air drawn in and a latch)
    'ui': dict(says='a product interface: app, website, dashboard', theme=None, seeds={'tap': 0},
               quiet=('hover',), sounds=(
        ('tap', 'keyboard/mechanical'), ('hover', 'hover/fingertip'), ('toggle_on', 'toggle/rocker_on'),
        ('toggle_off', 'toggle/rocker_off'), ('type', 'type/key'), ('back', 'back/page'), ('next', 'next/page'),
        ('open', 'open/air'), ('close', 'close/air'), ('send', 'send/swoosh'), ('receive', 'receive/pop'),
        ('notify', 'notify/knock'), ('success', 'success/click'), ('complete', 'complete/stamp'),
        ('warning', 'warning/knock'), ('error', 'error/thud'), ('delete', 'delete/crumple')),
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
    # thumps, ticks and clicks, made of real things; no note, bell or synthesised tone (the film's music has those)
    'film': dict(says='a product film or motion-design piece: moves, arrivals, hits and a logo', theme=None, sounds=(
        ('whoosh', 'whoosh/fast'), ('whoosh_soft', 'whoosh/soft'), ('slide', 'slide/panel'),
        ('pop', 'thump/appear'), ('tick', 'tick/soft'), ('click', 'click/mouse'), ('type', 'typing_key/light'),
        ('shutter', 'shutter/dslr'), ('hit', 'slam/title'), ('impact', 'boom/deep'), ('drop', 'fall/sub'),
        ('riser', 'rise/air'), ('swell', 'rise/reverse'), ('glint', 'shine/air'), ('success', 'success/click'),
        ('sting', 'sting/reveal'), ('logo', 'boom/short'))),
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
