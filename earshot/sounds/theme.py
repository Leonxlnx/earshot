"""Themes: what makes a set of sounds one family.

A product's sounds read as one voice when they share a timbre, a key, a space and a temperament (Material's and
Windows' guidance; a set where every sound comes from another world reads as stock). A theme is those shared
choices, applied to every design that honours them:

    voice      the instrument or patch that plays tonal designs (earshot.synth.voices)
    material   what taps, clicks and knocks are made of
    key, scale the key every tonal design is moved into, and the scale of its steps
    space      the room (earshot.synth.space)
    axes       offsets on the shared axes (bright, hard, ring, size, space, width, energy, motion)

    from earshot.sounds import theme
    t = theme.get('glass')                       # a preset
    t = theme.get({'base': 'wood', 'key': 'D'})  # a preset with changes
"""
KEYS = ('C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B')
FIELDS = ('voice', 'material', 'key', 'scale', 'space', 'axes', 'says')

THEMES = {
    'soft': dict(voice='marimba', material='soft', key='C', scale='major', space='room',
                 axes=dict(bright=-0.2, hard=-0.4),
                 says='a real marimba played softly: warm, rounded, mid-register'),
    'glass': dict(voice='glockenspiel', material='glass', key='D', scale='major', space='studio',
                  axes=dict(bright=0.2, ring=0.1), says='clear, bright, bell-like'),
    'wood': dict(voice='marimba', material='wood', key='G', scale='pentatonic', space='room',
                 axes=dict(bright=-0.1, ring=-0.2), says='a real marimba: warm, dry, organic'),
    'felt': dict(voice='felt_piano', material='soft', key='F', scale='major', space='room',
                 axes=dict(bright=-0.3, hard=-0.3), says='an intimate soft piano (a real upright, played softly)'),
    'tine': dict(voice='kalimba', material='wood', key='A', scale='pentatonic', space='room',
                 axes=dict(ring=-0.1), says='plucked tines: friendly, playful'),
    'metal': dict(voice='vibraphone', material='metal', key='E', scale='major', space='studio',
                  axes=dict(ring=0.2), says='smooth sustained metal bars'),
    'electric': dict(voice='electric_piano', material='soft', key='Eb', scale='major', space='room',
                     axes=dict(bright=-0.1), says='a mellow electric piano'),
    'digital': dict(voice='glass_pluck', material='digital', key='B', scale='major', space='dry',
                    axes=dict(bright=0.2, hard=0.2), says='clean synthetic tones: precise, modern'),
    'retro': dict(voice='square_lead', material='digital', key='C', scale='major', space='dry',
                  axes=dict(hard=0.3), says='chip-style: 8-bit games'),
}
DEFAULT = 'soft'


def get(theme=None):
    """a theme dict from a preset name, a dict (optionally with 'base': a preset it changes), or None (no theme)"""
    if theme is None:
        return None
    if isinstance(theme, str):
        if theme not in THEMES:
            raise ValueError(f'unknown theme {theme!r}: one of {", ".join(THEMES)}')
        return dict(THEMES[theme], name=theme)
    t = dict(theme)
    base = t.pop('base', None)
    if base is not None and base not in THEMES:
        raise ValueError(f'unknown theme {base!r}: one of {", ".join(THEMES)}')
    out = dict(THEMES[base], name=base) if base else {}
    bad = [k for k in t if k not in FIELDS + ('name',)]
    if bad:
        raise ValueError(f'unknown theme fields {bad}: {", ".join(FIELDS)}')
    if t.get('key') is not None and t['key'] not in KEYS:
        raise ValueError(f'unknown key {t["key"]!r}: one of {", ".join(KEYS)}')
    out.update({k: v for k, v in t.items() if k != 'axes'})
    out['axes'] = {**(out.get('axes') or {}), **(t.get('axes') or {})}
    return out
