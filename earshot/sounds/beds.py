"""Music beds: short loops in a key and a tempo that sit under a voice-over or under sound effects (`bed/warm`,
`bed/playful`, `bed/8bit`), written by the rules of earshot.synth.music. earshot makes sound effects; these three
beds are the ones kept (nine other styles were removed: long sustained synthesizer chords under a scene sound
cheap and pull attention from the picture).

    from earshot.sounds import registry
    y, meta = registry.render('bed/warm', seed=3, dur=20, pitch='D3', energy=0.2)    # a seamless loop, 7 bars
    y, meta = registry.render('bed/playful', loop=False, dur=16)                      # once, ending on a held chord

The controls:
    pitch    the key's tonic ('D3', 'F#3', Hz): the octave does not matter; a theme's key moves it as everywhere
    dur      the length; a bed is a whole number of bars, the nearest to dur at its tempo (meta['dur_s'] is the
             real length: bars x 4 x 60 / bpm)
    motion   the tempo: the style's own bpm x 1.15 ^ motion (-1 .. 1 = 15 % slower .. faster)
    energy   how much plays and how hard: parts enter earlier or stay out, ghost notes, motif notes
    voice    another instrument for the bed's main part (the electric piano of bed/warm, the muted guitar of
             bed/playful: 'felt_piano', 'marimba', 'harp' ...)
    bright, space, width   as everywhere
    loop     False: played once; the last bar is the tonic chord, held, and it rings out
A minor or major key in the request or the theme ('in D minor') changes the mode where the style has both.
A seed is another take of the style: another progression, voicing and motif, the same instruments.
`bed/8bit` is chip sound (earshot.synth.chip): it answers only when that is asked for.

The drums of bed/warm and bed/playful are the catalogue's own (earshot.sounds.drums: kick/soft, snare/brush,
rim/click, shaker/tick, clap/hands), played dry into the bed's one room; bed/8bit plays the chip channels.
"""
import numpy as np

from ..synth import music
from . import drums as _drums       # noqa: F401  (registers the drum designs the kits play)
from . import registry
from .registry import design

AXES = ('energy', 'motion', 'bright', 'space', 'width')
DOMAINS = ('music', 'motion', 'ui', 'game')
# what the audit of a single sound finds in music and is the music: drum strokes and plucked attacks (clicks), a
# top line that climbs (rising_glide); a chip bed has no room, so its rests are digital silence
ALLOWS = ('clicks', 'rising_glide')
# the scale a request or theme names -> the mode the bed is written in
MODE = {'major': 'major', 'pentatonic': 'major', 'lydian': 'major', 'minor': 'minor', 'minor_pentatonic': 'minor',
        'dorian': 'dorian'}
# kit piece -> the drum design that plays it and its tuning (None = as designed). The soft kick is tuned up a
# fourth to D2: a bed's low drums have to be heard on small speakers
ACOUSTIC = {'kick': ('kick/soft', 'D2'), 'snare': ('snare/brush', None), 'rim': ('rim/click', None),
            'clap': ('clap/hands', None), 'shaker': ('shaker/tick', None)}
# style -> the pieces of its kit that are acoustic (the rest are the engine's own)
KITS = {'warm': ('kick', 'snare', 'rim', 'shaker'), 'playful': ('kick', 'clap', 'shaker')}
# id -> (style, says, uses, tags)
BEDS = {
    'bed/warm': ('warm', ('slow warm electric piano chords over a soft kick, a brushed snare and a shaker, with '
                          'vinyl crackle', 'laid-back lo-fi hip hop background music',
                          'a mellow lo-fi beat with jazzy electric piano chords'),
                 ('lo-fi', 'lofi', 'background music', 'music', 'chill', 'study', 'landing page', 'vlog',
                  'beat'), ('lo-fi', 'hip hop')),
    'bed/playful': ('playful', ('a bouncy tune of pizzicato strings and a muted guitar with hand claps and a shaker',
                                'cheerful playful casual game menu music', 'a light funny cartoon music loop'),
                    ('background music', 'music', 'menu music', 'casual game', 'playful', 'kids', 'cartoon',
                     'fun', 'puzzle game', 'tutorial'), ('game', 'comedy')),
    'bed/8bit': ('8bit', ('a chiptune loop: a square wave melody over a fast arpeggio, a triangle bass and noise '
                          'drums', '8-bit video game music', 'retro console game background music'),
                 ('background music', 'music', 'game music', 'menu music', 'level music', 'chiptune', 'arcade'),
                 ('retro', 'chip', '8-bit', 'arcade')),
}


def _piece(sid, pitch=None):
    """a drum design as a kit piece: one dry stroke at a velocity (its energy axis is its velocity)"""
    def play(vel, r):
        d = registry.get(sid)
        c = registry.Controls(d, pitch=pitch, space=-1.0, energy=float(np.clip((vel - 0.7) / 0.3, -1.0, 1.0)))
        return d.fn(c, r)
    return play


def kit(style):
    """the acoustic pieces of a style's kit, for earshot.synth.music.render"""
    return {name: _piece(*ACOUSTIC[name]) for name in KITS.get(style, ())}


def _bed(style):
    st = music.STYLES[style]

    def fn(c, r):
        bpm = st['bpm'] * 1.15 ** c.motion
        bars = max(2, int(round(c.seconds() * bpm / (60.0 * music.BEATS))))
        plan = music.compose(style, r, key=c.hz(), bpm=bpm, bars=bars, mode=MODE.get(c._theme.get('scale')),
                             energy=c.energy, loop=c.loop, voice=None if style == '8bit' else c.voice)
        return music.render(plan, r, kit=kit(style), room=c.room(st['room']), amount=c.space, bright=c.bright,
                            width=c.width)
    return fn


for _id, (_style, _says, _uses, _tags) in BEDS.items():
    _st = music.STYLES[_style]
    design(_id, kind='music', says=_says, uses=_uses + ('loop',), tags=('music', 'bed') + _tags, domains=DOMAINS,
           dur=round(_st['bars'] * music.BEATS * 60.0 / _st['bpm'], 1), pitch=_st['key'] + '3', axes=AXES,
           voice=next((p['voice'] for p in _st['parts'] if p.get('main')), None),
           material='retro' if _style == '8bit' else None, loop=True,
           allows=ALLOWS + (('silence',) if _style == '8bit' else ()),
           prior=1.15 if _style == 'warm' else 1.0)(_bed(_style))       # warm: the bed for plain "background music"
