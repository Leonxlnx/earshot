"""Single notes: every voice of earshot.synth.voices as a design (`note/kalimba`, `note/felt_piano`,
`note/glockenspiel` ...), for when a request names the instrument itself ("a kalimba note in E", "one soft piano
note") or a product builds its own motifs note by note.

A note rings as long as its instrument does, at most 6 s, unless `dur` damps it earlier or lets it ring longer;
`energy` is how hard it is played.
"""
from ..synth import space, voices
from .registry import design

AXES = ('bright', 'hard', 'ring', 'size', 'space', 'width', 'energy', 'motion')
# a voice's material -> the room a single note of it sits in
ROOM = {'metal': 'studio', 'tine': 'room', 'wood': 'room', 'glass': 'studio', 'felt': 'room', 'piano': 'studio',
        'electric': 'room', 'string': 'studio', 'synth': 'dry'}


def _note(voice, room):
    def fn(c, r):
        dur = c.seconds()                    # the design's own length caps a bell's or a bowl's half-minute ring
        own = {'space': c.space} if room == 'dry' else {}        # a patch alone keeps its own delay and reverb
        y = voices.play(voice, c.hz(), vel=min(1.0, 0.65 * c.db('energy', 7.0)), dur=dur, r=r, vary=c.vary,
                        **c.axes('bright', 'hard', 'ring', 'size', 'width', 'motion'), **own)
        return y if own else space.room(y, c.room(room), amount=c.space, r=r)
    return fn


for _v, _i in voices.table().items():
    if _i['engine'] == 'patches' and _i.get('kind') in ('pad', 'hit', 'fx'):
        continue                         # finished sounds, not instruments: earshot.sounds.synth names them
    _words = _v.replace('_', ' ')
    design(f'note/{_v}', kind='note', says=tuple(_i['says'][:3]) if len(_i['says']) > 1 else (_i['says'][0], f'a {_words} note'),
           uses=(_words, 'note', 'tone'), tags=(_i.get('material') or 'synth', _i.get('kind') or 'instrument', 'tonal'),
           domains=('music', 'ui', 'game', 'motion'), dur=min(float(_i.get('ring_s') or 1.0), 6.0),
           pitch=_i['default_pitch'], axes=AXES, voice=_v, material=_i.get('material'),
           prior=0.95)(_note(_v, ROOM.get(_i.get('material') or 'synth', 'room')))
