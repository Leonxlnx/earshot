"""Synthesizer designs: the pads, drones, braams, risers and zaps of earshot.synth.patches under the names a request
uses (`pad/warm`, `braam/hit`, `riser/tone`, `zap/laser` ...).

These are finished sounds, not instruments: a pad is a held chord with its own chorus and space, a braam is the
whole blast. `pitch` is the root, `dur` how long it is held before the release, `motion` the movement inside it.
The patches that are played like instruments (plucks, keys, basses, leads) are notes: `note/soft_pluck` ...
"""
from ..synth import patches
from .registry import design

AXES = ('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy', 'motion')
# id -> (patch, kind, hold s, chord in semitones over the root, fixed controls, says, uses, domains, allows)
SOUNDS = {
    'pad/warm': ('warm_pad', 'pad', 2.5, (0, 7, 16), {}, ('a warm analog synthesizer pad chord',
                                                          'a slow lush soft pad'),
                 ('pad', 'background chord', 'bed', 'calm', 'intro'), ('motion', 'music', 'ambience'), ()),
    'pad/air': ('air_pad', 'pad', 2.5, (0, 7, 14), {}, ('a breathy airy synthesizer pad',
                                                        'an ethereal atmospheric pad chord'),
                ('pad', 'atmosphere', 'bed', 'dream', 'ethereal'), ('motion', 'music', 'ambience'), ()),
    'pad/glass': ('glass_pad', 'pad', 2.5, (0, 7, 16), {}, ('a glassy shimmering synthesizer pad',
                                                            'a bright crystalline pad chord'),
                  ('pad', 'shimmer', 'bed', 'magic', 'reveal'), ('motion', 'music', 'ambience'), ()),
    'pad/strings': ('string_pad', 'pad', 2.5, (0, 7, 16), {}, ('a string ensemble synthesizer chord',
                                                               'lush synthetic strings'),
                    ('pad', 'strings', 'bed', 'emotional'), ('motion', 'music'), ()),
    'drone/dark': ('drone', 'pad', 5.0, (0,), {}, ('a dark low synthesizer drone', 'a deep ominous evolving low bed'),
                   ('drone', 'tension bed', 'ominous', 'dark ambience', 'suspense'),
                   ('motion', 'game', 'ambience'), ()),
    'sub/swell': ('sub_swell', 'rise', 2.0, (0,), {}, ('a low sub bass swell rising in volume',
                                                       'a deep rumbling bass build-up'),
                  ('sub swell', 'riser', 'build', 'tension'), ('motion',), ('truncated',)),
    'braam/hit': ('braam', 'hit', 1.6, (0,), {}, ('a huge low brass-like blast, a cinematic braam',
                                                  'a deep distorted trailer horn hit'),
                  ('braam', 'trailer hit', 'impact', 'reveal', 'title'), ('motion', 'game'), ('stacked', 'pre_swell')),
    'braam/swell': ('braam', 'rise', 2.2, (0,), {'hard': -1.0}, ('a low brass-like swell growing into a blast',
                                                                 'a deep horn-like cinematic swell'),
                    ('braam', 'swell', 'build', 'tension', 'reveal'), ('motion', 'game'),
                    ('stacked', 'pre_swell', 'truncated')),
    'riser/tone': ('riser_tone', 'rise', 2.5, (0,), {}, ('a tonal synthesizer riser, swelling and opening',
                                                         'a rising tension build-up before a drop'),
                   ('riser', 'build', 'tension', 'anticipation'), ('motion', 'game'), ('rising_glide', 'truncated')),
    'sweep/noise': ('noise_sweep', 'rise', 1.5, (0,), {}, ('a filtered noise sweep rising',
                                                           'a rising noise swoosh transition'),
                    ('riser', 'sweep', 'transition', 'whoosh', 'build'), ('motion',), ('truncated',)),
    'stab/chord': ('stab', 'hit', 0.18, (0,), {}, ('a short punchy synthesizer chord stab', 'a minor chord synth hit'),
                   ('stab', 'accent', 'hit', 'rhythm'), ('motion', 'music'), ()),
    'zap/laser': ('zap', 'zap', 0.25, (0,), {}, ('a science fiction laser shot, a descending synth zap with a body',
                                                 'a designed laser zap'),
                  ('laser', 'zap', 'shoot', 'blaster'), ('game', 'motion'), ()),
    'power_up/synth': ('power_up', 'figure', 0.6, (0,), {}, ('a rising sparkling synthesizer arpeggio',
                                                             'a magical ascending synth run with shimmer'),
                       ('power up', 'boost', 'upgrade', 'magic'), ('game',), ('rising_glide', 'tail')),
    'power_down/synth': ('power_down', 'figure', 0.7, (0,), {}, ('a descending synthesizer arpeggio winding down',
                                                                 'a machine shutting down tone'),
                         ('power down', 'shutdown', 'drain', 'fail'), ('game', 'ui'), ('tail',)),
}
PATCHES_USED = tuple(sorted({v[0] for v in SOUNDS.values()}))


def _render(patch, hold, chord, fixed):
    def fn(c, r):
        f = c.hz()
        pitch = [f * 2 ** (s / 12) for s in chord] if len(chord) > 1 else f
        kw = {**c.axes('bright', 'hard', 'ring', 'size', 'motion', 'width', 'space'), **fixed}
        return patches.play(patch, pitch, vel=min(1.0, 0.8 * c.db('energy', 6.0)), dur=c.seconds(hold), r=r,
                            vary=c.vary, **kw)
    return fn


for _id, (_p, _kind, _hold, _chord, _fixed, _says, _uses, _dom, _allows) in SOUNDS.items():
    design(_id, kind=_kind, says=_says, uses=_uses, tags=('synth',), domains=_dom, dur=_hold,
           pitch=patches.PATCHES[_p]['default_pitch'], axes=AXES, material='digital', allows=_allows,
           prior=1.05)(_render(_p, _hold, _chord, _fixed))
