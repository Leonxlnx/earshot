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
    # the touches are recorded keys and switches (earshot.sounds.learned): tonal touches are hard to recognise as
    # touches, a real key reads at once. tap is the fullest mechanical key (keyboard/mechanical, first in its walk),
    # type the other mechanical key; next and back the two laptop keys nearest that key (next member 4, back
    # member 1: the darker and softer); toggle_on and toggle_off two recordings of switch/lamp, the off the darker one
    # (member 1), the on member 2; hover the softest, lowest short touch that is not thin (leather/grab member 1).
    # No theme: the figures play in their own key and voice (moved to C, darker and softer, they sound dull)
    'ui': dict(says='a product interface: app, website, dashboard', theme=None,
               seeds={'tap': 0, 'type': 1, 'hover': 1, 'toggle_on': 1, 'toggle_off': 0, 'back': 4, 'next': 3},
               quiet=('hover',), sounds=(
        ('tap', 'keyboard/mechanical'), ('hover', 'leather/grab'), ('toggle_on', 'switch/lamp'),
        ('toggle_off', 'switch/lamp'), ('type', 'keyboard/mechanical'), ('back', 'keyboard/laptop'),
        ('next', 'keyboard/laptop'), ('open', 'open/lift'), ('close', 'close/drop'),
        ('send', 'send/lift'), ('receive', 'receive/drop'), ('notify', 'notify/pair'), ('success', 'success/rise'),
        ('complete', 'success/triad'), ('warning', 'warning/pulse'), ('error', 'error/fall'),
        ('delete', 'delete/drop'))),
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
    'film': dict(says='a product film or motion-design piece: moves, arrivals, hits and a logo', theme=None, sounds=(
        ('whoosh', 'whoosh/fast'), ('whoosh_soft', 'whoosh/soft'), ('slide', 'slide/panel'), ('pop', 'pop/soft'),
        ('tick', 'tock/wood'), ('click', 'click/soft'), ('type', 'typing_key/light'), ('shutter', 'shutter/dslr'),
        ('hit', 'slam/title'), ('impact', 'boom/deep'), ('drop', 'fall/sub'), ('riser', 'rise/air'),
        ('swell', 'rise/reverse'), ('glint', 'shine/glint'), ('chime', 'success/rise'), ('sting', 'sting/reveal'),
        ('logo', 'logo/bloom'))),
}


def get(name):
    if name not in SETS:
        raise ValueError(f'unknown set {name!r}: one of {", ".join(SETS)}')
    return SETS[name]
