"""Ambience: real textures learned from public-domain recordings (`fire/camp`, `waves/sand`, `night/crickets`,
`stream/brook`, `cafe/busy`, `city/street` ...) and the scenes of earshot.synth.texture that are invented from
physics and kept where the owner rated them well or nothing real stands in for them (`rain/window`,
`forest/morning`, `wind/trees`, `thunder/clap` ...).

A learned bed is a texture stored as numbers (earshot/data/textures: resyn models of 12-16 s of one recording, a
few hundred kB, credits in CREDITS.md there) played endlessly by earshot.synth.quilt: fresh takes of the models,
cut into pieces of 1-3 s (whole waves for the sea) and stitched where the texture is in a similar state. The
owner hears no difference between a recording and its resyn take; the invented fire, sea and night he rated 2-3.

A bed is a seamless loop by default: the file can repeat for as long as the scene lasts (a game level, a landing
page, a meditation timer) without a seam. `dur` is the loop's length (longer loops repeat less often),
`loop=False` (`--no-loop`) gives the same scene once with a fade in and out, for a film. Beds sit at the
registry's level for beds: they lie under everything else.

The axes of a learned bed: `energy` picks the calmer or the busier moments of the recording (the lulls of a fire
or its crackling spells); `size` is the source a little smaller or bigger (resyn's size: 0.8-1.25, higher and
quicker or lower and slower, the sea's waves slower); `space` the distance (farther: the air takes the top, a
diffuse room joins; nearer: a little more presence); `bright` a tilt of +-1.5 dB per octave; `width` the stereo
image. Of an invented scene: `energy` how much of it there is, `size` how big the source is, `space` how far.

    from earshot.sounds import registry
    y, meta = registry.render('fire/camp', dur=30, seed=2, energy=0.4)       # 30 s loop, the busier moments
"""
import json
import os
from functools import lru_cache

import numpy as np

from ..forge import dsp, fx
from ..synth import quilt, resyn, space as rooms, texture
from .registry import design

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'textures')

# id -> (scene, fixed controls, says, uses)
SCENES = {
    'rain/window': ('rain', dict(surface='window'), ('steady rain on a window pane', 'rain tapping on glass',
                                                     'rain heard from indoors'),
                    ('rain', 'rainy weather', 'cozy', 'storm outside')),
    'rain/leaves': ('rain', dict(surface='leaves'), ('soft rain falling on leaves', 'gentle rain in a forest',
                                                     'rain pattering on foliage'),
                    ('rain', 'forest rain', 'drizzle', 'garden')),
    'rain/roof': ('rain', dict(surface='roof'), ('rain drumming on a roof', 'heavy rain on a tin roof',
                                                 'rain on a rooftop'),
                  ('rain', 'downpour', 'storm', 'shelter')),
    'rain/tent': ('rain', dict(surface='tent'), ('rain on a tent', 'rain pattering on fabric',
                                                 'rain on a canvas shelter'),
                  ('rain', 'camping', 'tent')),
    'rain/water': ('rain', dict(surface='water'), ('rain falling on a lake', 'rain drops hitting water',
                                                   'rain on a pond'),
                   ('rain', 'lake', 'pond', 'water')),
    'wind/open': ('wind', dict(place='open'), ('wind blowing over open land', 'steady gusts of wind',
                                               'a strong breeze'),
                  ('wind', 'breeze', 'gust', 'outdoors', 'desert')),
    'wind/window': ('wind', dict(place='window'), ('wind howling through a gap in a window', 'a whistling draught',
                                                   'wind moaning outside a house'),
                    ('wind', 'howling', 'storm', 'haunted', 'winter')),
    'wind/mountain': ('wind', dict(place='mountain'), ('high wind on a mountain', 'a cold gale over rock',
                                                       'wind roaring at altitude'),
                      ('wind', 'mountain', 'gale', 'summit', 'blizzard')),
    # invented, as the rain is: broadband air shaped by slow gusts and a leaf rustle that swells with them, no
    # Aeolian tones. The one public-domain recording of trees in wind holds a thin rustle around 5 kHz that does
    # not follow the gusts and little air under it: faithfully resynthesised it reads as trickling water
    'wind/trees': ('wind', dict(place='trees', howl=0.0), ('wind in the trees', 'leaves rustling in gusts of wind',
                                                           'a breeze through foliage'),
                   ('wind', 'trees', 'leaves', 'forest wind', 'breeze')),
    'fire/small': ('fire', dict(size=-0.7), ('a small fire crackling quietly', 'a little fireplace',
                                             'gentle fire crackle'),
                   ('fire', 'fireplace', 'candle', 'hearth')),
    'fire/large': ('fire', dict(size=0.8), ('a large roaring fire', 'a bonfire burning fiercely',
                                            'big flames roaring and crackling'),
                   ('fire', 'bonfire', 'burning building', 'inferno')),
    'stream/trickle': ('stream', dict(size=-0.8), ('a thin trickle of water', 'water dripping and trickling',
                                                   'a tiny rivulet'),
                       ('stream', 'trickle', 'fountain', 'drip')),
    'stream/river': ('stream', dict(size=0.8), ('a river flowing fast', 'a wide rushing river', 'white water rushing'),
                     ('stream', 'river', 'rapids', 'waterfall')),
    'forest/morning': ('forest', {}, ('birds singing in a forest', 'a quiet forest in the morning with birdsong',
                                      'birdsong and a light breeze'),
                       ('forest', 'birds', 'birdsong', 'nature', 'morning', 'garden', 'meadow')),
    'space_hum/ship': ('space_hum', {}, ('the low hum of a spaceship engine room', 'a deep machine drone',
                                         'a science fiction ship interior'),
                       ('space hum', 'spaceship', 'engine room', 'server room', 'reactor', 'sci-fi ambience')),
    'crackle/vinyl': ('crackle', {}, ('vinyl record crackle', 'crackling static from an old record',
                                      'soft pops and crackle'),
                      ('vinyl', 'record crackle', 'vinyl crackle', 'old radio', 'static')),
    'sparkle/magic': ('sparkle', {}, ('a magical shimmering sparkle', 'twinkling chimes in the air',
                                      'fairy dust glitter'),
                      ('sparkle', 'magic', 'shimmer', 'fairy', 'enchanted', 'glitter')),
}
ONCE = {
    'thunder/clap': ('thunder', {}, 7.0, ('a clap of thunder and a rolling rumble', 'thunder cracking overhead',
                                          'a thunderclap'),
                     ('thunder', 'lightning', 'storm', 'thunderclap', 'boom', 'crack')),
    'thunder/distant': ('thunder', dict(distance=0.8), 8.0, ('distant thunder rumbling', 'a far-off storm',
                                                             'low rolling thunder'),
                        ('thunder', 'distant storm', 'rumble', 'boom')),
}
# id -> (stored texture, says, uses): beds learned from real recordings (earshot/data/textures/index.json)
LEARNED = {
    'fire/camp': ('campfire', ('a crackling campfire of branches', 'wood burning with sharp cracks and a soft roar',
                               'a real fire crackling outdoors'),
                  ('fire', 'campfire', 'fireplace', 'cozy', 'camping', 'bonfire')),
    'waves/sand': ('beach', ('small waves breaking on a sandy beach', 'surf washing up the shore and drawing back',
                             'the sea at a beach'),
                   ('waves', 'ocean', 'beach', 'sea', 'surf', 'water', 'lake shore', 'lapping water', 'calm water')),
    'waves/pebbles': ('pebbles', ('waves breaking on a pebble beach', 'surf dragging over shingle',
                                  'the sea rolling stones'),
                      ('waves', 'pebble beach', 'shingle', 'shore', 'coast', 'water')),
    'waves/rocks': ('rocks', ('waves washing over rocks and a small beach', 'the sea breaking on a rocky shore',
                              'surf against stones'),
                    ('waves', 'rocky shore', 'rocks', 'cliffs', 'coast', 'sea', 'water')),
    'night/crickets': ('crickets', ('crickets chirping at night in the countryside', 'a field of crickets at night',
                                    'a summer night with crickets'),
                       ('night', 'crickets', 'insects', 'evening', 'summer night', 'camping night')),
    'stream/brook': ('brook', ('a small stream running over stones', 'water babbling in a brook',
                               'a gentle flowing creek'),
                     ('stream', 'brook', 'creek', 'running water', 'water')),
    'room_tone/quiet': ('room', ('the quiet sound of an apartment with the windows closed', 'room tone',
                                 'a faint hum of a building and far traffic'),
                        ('room tone', 'office', 'indoor ambience', 'silence', 'apartment', 'indoors')),
    'city/street': ('street', ('a city street with passing cars and footsteps', 'traffic and people in a town',
                               'a busy street'),
                    ('city', 'street', 'traffic', 'urban', 'downtown', 'town')),
    'city/distant': ('street', ('distant city traffic', 'the far hum of a town', 'traffic heard from a distance'),
                     ('city', 'traffic', 'urban', 'distant traffic', 'downtown')),
    'cafe/busy': ('restaurant', ('a busy restaurant with murmuring voices and clinking dishes',
                                 'people talking in a cafe', 'the chatter of a dining room'),
                  ('cafe', 'restaurant', 'coffee shop', 'crowd', 'chatter', 'murmur', 'diner', 'bar')),
    'underwater/deep': ('underwater', ('the rush of water heard under water', 'muffled sound under water',
                                       'a deep underwater roar'),
                        ('underwater', 'submerged', 'diving', 'ocean depths', 'water')),
    'rain/street': ('terrace', ('summer rain falling on a terrace', 'rain splashing on paving stones',
                                'steady rain outdoors'),
                    ('rain', 'city rain', 'street', 'urban', 'terrace')),
}
# what the audit finds in any scene of many small events and is the scene: single loud drops and crackles (clicks),
# events after a lull (tail), a bird's or a bubble's rising chirp (rising_glide), the comb of a steady insect or chime
ALLOWS = ('clicks', 'tail', 'silence', 'rising_glide', 'ring')
AXIS = {'intensity': 'energy', 'density': 'energy', 'size': 'size', 'bright': 'bright', 'width': 'width',
        'distance': 'space'}
LEARNED_AXES = ('energy', 'size', 'space', 'bright', 'width')
# a learned bed heard from farther than its recording (space added): the distant city is the street, far off
AWAY = {'city/distant': 0.7}


def _controls(c, scene, fixed):
    """the shared axes as the scene's own controls (AXIS), the design's fixed ones on top"""
    kw = {}
    for name in texture.SCENES[scene]['controls']:
        if name in AXIS and name not in fixed:
            kw[name] = getattr(c, AXIS[name])
        elif name in AXIS:
            kw[name] = max(-1.0, min(1.0, fixed[name] + getattr(c, AXIS[name])))
    return {**fixed, **kw}


def _bed(scene, fixed, shortest):
    def fn(c, r):
        dur = max(c.seconds(), 2.0)
        if c.loop:
            return texture.loop(scene, max(dur, shortest), r, **_controls(c, scene, fixed))
        return texture.render(scene, dur, r, **_controls(c, scene, fixed))
    return fn


def _axes(scene):
    return tuple(dict.fromkeys(AXIS[n] for n in texture.SCENES[scene]['controls'] if n in AXIS))


# ------------------------------------------------------------------------------------------------ learned beds
@lru_cache(maxsize=1)
def store():
    """the index of the stored textures: {name: {'ids', 'title', 'author', 'page', 'licence', 'chunks', ...}}"""
    with open(os.path.join(STORE, 'index.json'), encoding='utf-8') as fh:
        return json.load(fh)['textures']


@lru_cache(maxsize=None)
def models(name):
    """the chunk models of a stored texture (numbers, loaded once per process; their renders are not kept)"""
    return tuple(resyn.load(os.path.join(STORE, c['file'])) for c in store()[name]['chunks'])


def _circular(fn, y, loop):
    """fn(y) for a process with memory (a filter, a room): for a loop, run over two rounds and keep the second, so
    what rings past the end lies on the start and the loop still closes"""
    if not loop:
        return fn(y)[:len(y)]
    return fn(np.concatenate([y, y]))[len(y):2 * len(y)]


def place(y, c, mono, loop, away=0.0):
    """the image and the distance of a learned bed. width: a stereo recording's side scaled 0.4-1.6 times; a mono
    one opened by decorrelation (fx.decorrelate, 0.25-0.75). space > 0: farther, the air taking the top (a low-pass
    from 18 kHz to 3.5 kHz) and a diffuse hall joining; space < 0: nearer, up to 3 dB more presence at 3 kHz"""
    w, far = c.width, float(np.clip(c.space + away, -1.0, 1.0))
    if mono:
        y = _circular(lambda z: fx.decorrelate(z.mean(axis=1), 0.5 + 0.25 * w), y, loop)
    elif w:
        y = fx.width(y, 1.0 + 0.6 * w)
    if far > 0:
        y = _circular(lambda z: dsp.butter(z, 'low', 18000.0 * (3500.0 / 18000.0) ** far), y, loop)
        y = _circular(lambda z: rooms.room(z, c.room('hall'), amount=2.0 * far - 1.0, r=dsp.rng(7)), y, loop)
    elif far < 0:
        y = _circular(lambda z: dsp.peq(z, 3000.0, -3.0 * far, 0.7), y, loop)
    return y


def _learned(name, away=0.0):
    def fn(c, r):
        tex = store()[name]
        periods = [ch['period'] for ch in tex['chunks']] if tex['periodic'] else None
        dur = max(c.seconds(), 2.0 * max(periods) if periods else 4.0)      # the sea: at least two whole waves
        tk = quilt.takes(models(name), dur, r, periods=periods, energy=c.energy,
                         size=c.lerp('size', 0.8, 1.25, 1.0, log=True), bright=1.5 * c.bright)
        y, _ = quilt.stitch(tk, dur, r, loop=c.loop, periodic=tex['periodic'], energy=c.energy)
        return place(y, c, tex['channels'] == 1, c.loop, away)
    return fn


for _id, (_scene, _fixed, _says, _uses) in SCENES.items():
    _short = float(texture.SCENES[_scene]['min_loop'])
    design(_id, kind='bed', says=_says, uses=_uses + ('ambience', 'background', 'loop'), tags=('nature', 'ambience'),
           domains=('ambience', 'game', 'motion'), dur=max(_short, 12.0), axes=_axes(_scene), loop=True,
           allows=ALLOWS)(_bed(_scene, _fixed, _short))

for _id, (_name, _says, _uses) in LEARNED.items():
    design(_id, kind='bed', says=_says, uses=_uses + ('ambience', 'background', 'loop'),
           tags=('nature', 'ambience', 'learned', 'real'), domains=('ambience', 'game', 'motion'), dur=20.0,
           axes=LEARNED_AXES, loop=True, allows=ALLOWS)(_learned(_name, AWAY.get(_id, 0.0)))

for _id, (_scene, _fixed, _dur, _says, _uses) in ONCE.items():
    def _fn(c, r, scene=_scene, fixed=_fixed):
        return texture.render(scene, c.seconds(), r, **_controls(c, scene, fixed))
    design(_id, kind='noise', says=_says, uses=_uses, tags=('nature', 'weather'),
           domains=('ambience', 'game', 'motion'), dur=_dur, axes=_axes(_scene),
           allows=ALLOWS + ('stacked', 'pre_swell'))(_fn)
