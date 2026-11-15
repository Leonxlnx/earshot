"""Ready-made sets: the sounds a kind of product usually needs, by name, so one call makes all of them as one family.

    earshot make --set ui --theme glass -o public/sounds
    earshot make --set arcade -o assets/sfx --format ogg

A set is a list of (file name, design id). The theme decides the voice, key and room of the tonal ones; a spec may
add sounds, replace one by name, or change any of them (earshot.make.make_set). The lists follow the inventories
interface and game audio teams work from; a product rarely needs more than a dozen distinct sounds, and fewer is
better than every event making noise.
"""
SETS = {
    'ui': dict(says='a product interface: app, website, dashboard', theme='soft', sounds=(
        ('tap', 'tap/tone'), ('hover', 'hover/tone'), ('toggle_on', 'toggle/up'), ('toggle_off', 'toggle/down'),
        ('open', 'open/lift'), ('close', 'close/drop'), ('next', 'next/step'), ('back', 'back/step'),
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
