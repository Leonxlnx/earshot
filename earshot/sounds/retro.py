"""Chip sound for games: the generators of earshot.synth.chip as designs (`coin/chip`, `jump/chip`, `win/chip` ...).

They are tagged `retro` and only answer requests that ask for it ("8-bit", "retro", "chiptune", "arcade", a theme of
that material), so a product without a pixel look never gets a chip sound by accident. `pitch` moves the tonal
ones into a key, `dur` stretches a sound within its range, `bright` opens or closes the output filter, and a high
`energy` plays it through a 6-bit output stage.
"""
from ..synth import chip
from .registry import design

# chip generator -> (design id, kind, the jobs it plays, audit findings that are its design)
SOUNDS = {
    'coin': ('coin/chip', 'chime', ('coin', 'pickup', 'collect', 'score', 'ding'), ('rising_glide',)),
    'pickup': ('pickup/chip', 'chime', ('pickup', 'item', 'collect', 'heal'), ('rising_glide',)),
    'jump': ('jump/chip', 'zap', ('jump', 'hop'), ('rising_glide',)),
    'double_jump': ('jump/chip_double', 'zap', ('double jump', 'jump', 'dash'), ('rising_glide', 'tail')),
    'laser': ('laser/chip', 'zap', ('laser', 'shoot', 'zap'), ()),
    'explosion': ('explosion/chip', 'boom', ('explosion', 'blast', 'death'), ('stacked', 'pre_swell')),
    'power_up': ('power_up/chip', 'figure', ('power up', 'boost', 'upgrade'), ('rising_glide',)),
    'power_down': ('power_down/chip', 'figure', ('power down', 'shrink', 'drain'), ()),
    'hurt': ('hurt/chip', 'zap', ('hurt', 'damage', 'hit'), ()),
    'blip': ('menu/chip', 'tick', ('menu', 'cursor', 'tap', 'hover'), ()),
    'select': ('select/chip', 'chime', ('menu select', 'confirm', 'success', 'click'), ('rising_glide',)),
    'back': ('back/chip', 'chime', ('back', 'cancel', 'close', 'undo'), ()),
    'one_up': ('one_up/chip', 'figure', ('extra life', 'one up', 'bonus'), ('rising_glide',)),
    'level_up': ('level_up/chip', 'figure', ('level up', 'upgrade', 'achievement'), ('rising_glide',)),
    'win': ('win/chip', 'stinger', ('win', 'victory', 'level complete', 'fanfare'), ('rising_glide', 'tail')),
    'lose': ('lose/chip', 'stinger', ('lose', 'game over', 'death', 'defeat'), ('tail',)),
    'text_blip': ('text/chip', 'tick', ('text', 'dialogue', 'type', 'letter'), ()),
    'door': ('door/chip', 'noise', ('door', 'gate', 'open'), ('rising_glide',)),
    'step': ('footstep/chip', 'tick', ('footstep', 'walk', 'step'), ()),
    'splash': ('splash/chip', 'noise', ('splash', 'water', 'dive'), ()),
    'alarm': ('alarm/chip', 'figure', ('alarm', 'warning', 'alert', 'siren'), ('tail', 'rising_glide')),
    'engine': ('engine/chip', 'bed', ('engine', 'motor', 'vehicle'), ()),
    'teleport': ('teleport/chip', 'zap', ('teleport', 'warp', 'portal', 'magic'), ('rising_glide', 'tail')),
    'charge': ('charge/chip', 'rise', ('charge up', 'power up', 'loading'), ('rising_glide', 'truncated')),
}


def _render(name, pitched):
    def fn(c, r):
        return chip.render(name, r, pitch=c.hz() if pitched and c._pitch is not None else None,
                           dur=c.seconds() if c._dur is not None else None, bits=6 if c.energy >= 0.5 else None,
                           polish=c.lerp('bright', 0.9, 0.2, 0.5))
    return fn


for _name, (_id, _kind, _uses, _allows) in SOUNDS.items():
    _s = chip.SOUNDS[_name]
    # a low or thin pulse reads as isolated steps to the click detector and its harmonics as a comb to the ring
    # detector: that is the timbre, not a fault
    design(_id, kind=_kind, says=tuple(_s['says'][:3]), uses=_uses, tags=('retro', 'chip', '8-bit', 'arcade'),
           domains=('game',), dur=_s['dur_s'], pitch='C6' if _s['pitched'] else None, material='retro',
           axes=('bright', 'energy'), loop=bool(_s.get('loop')), allows=_allows + ('clicks', 'ring'))(_render(_name, _s['pitched']))
