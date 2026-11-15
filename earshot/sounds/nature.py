"""Ambience: the scenes of earshot.synth.texture as designs (`rain/window`, `wind/trees`, `fire/camp`,
`waves/sand`, `forest/morning`, `night/crickets` ...).

A bed is a seamless loop by default: the file can repeat for as long as the scene lasts (a game level, a landing
page, a meditation timer) without a seam or an audible period. `dur` is the loop's length (longer loops repeat less
obviously; each scene has a shortest sensible one), `loop=False` (`--no-loop`) gives the same scene once with a fade
in and out, for a film. Beds are quiet by design: they sit under everything else.

The axes: `energy` is how much of it there is (drizzle to downpour, breeze to gale), `size` how big the source is
(a trickle or a river, a candle or a bonfire), `space` how far away, `bright` and `width` as everywhere.
"""
from ..synth import texture
from .registry import design

# id -> (scene, fixed controls, shortest loop s, says, uses)
SCENES = {
    'rain/window': ('rain', dict(surface='window'), ('steady rain on a window pane', 'rain tapping on glass', 'rain heard from indoors'),
                    ('rain', 'rainy day', 'cozy', 'storm outside')),
    'rain/leaves': ('rain', dict(surface='leaves'), ('soft rain falling on leaves', 'gentle rain in a forest', 'rain pattering on foliage'),
                    ('rain', 'forest rain', 'drizzle', 'garden')),
    'rain/roof': ('rain', dict(surface='roof'), ('rain drumming on a roof', 'heavy rain on a tin roof', 'rain on a rooftop'),
                  ('rain', 'downpour', 'storm', 'shelter')),
    'rain/street': ('rain', dict(surface='street'), ('rain falling on a street', 'rain splashing on pavement', 'a rainy city street'),
                    ('rain', 'city rain', 'street', 'urban')),
    'rain/tent': ('rain', dict(surface='tent'), ('rain on a tent', 'rain pattering on fabric', 'rain on a canvas shelter'),
                  ('rain', 'camping', 'tent')),
    'rain/water': ('rain', dict(surface='water'), ('rain falling on a lake', 'rain drops hitting water', 'rain on a pond'),
                   ('rain', 'lake', 'pond', 'water')),
    'wind/open': ('wind', dict(place='open'), ('wind blowing over open land', 'steady gusts of wind', 'a strong breeze'),
                  ('wind', 'breeze', 'gust', 'outdoors', 'desert')),
    'wind/trees': ('wind', dict(place='trees'), ('wind in the trees', 'leaves rustling in gusts of wind', 'a breeze through foliage'),
                   ('wind', 'trees', 'leaves', 'forest wind')),
    'wind/window': ('wind', dict(place='window'), ('wind howling through a gap in a window', 'a whistling draught', 'wind moaning outside a house'),
                    ('wind', 'howling', 'storm', 'haunted', 'winter')),
    'wind/mountain': ('wind', dict(place='mountain'), ('high wind on a mountain', 'a cold gale over rock', 'wind roaring at altitude'),
                      ('wind', 'mountain', 'gale', 'summit', 'blizzard')),
    'fire/camp': ('fire', {}, ('a crackling campfire', 'a wood fire burning and crackling', 'fire crackle with a soft roar'),
                  ('fire', 'campfire', 'fireplace', 'cozy')),
    'fire/small': ('fire', dict(size=-0.7), ('a small fire crackling quietly', 'a little fireplace', 'gentle fire crackle'),
                   ('fire', 'fireplace', 'candle', 'hearth')),
    'fire/large': ('fire', dict(size=0.8), ('a large roaring fire', 'a bonfire burning fiercely', 'big flames roaring and crackling'),
                   ('fire', 'bonfire', 'burning building', 'inferno')),
    'stream/brook': ('stream', {}, ('a babbling brook', 'water flowing in a stream', 'a small stream over stones'),
                     ('stream', 'brook', 'creek', 'running water')),
    'stream/trickle': ('stream', dict(size=-0.8), ('a thin trickle of water', 'water dripping and trickling', 'a tiny rivulet'),
                       ('stream', 'trickle', 'fountain', 'drip')),
    'stream/river': ('stream', dict(size=0.8), ('a river flowing fast', 'a wide rushing river', 'white water rushing'),
                     ('stream', 'river', 'rapids', 'waterfall')),
    'waves/sand': ('waves', dict(surface='sand'), ('ocean waves breaking on a sandy beach', 'surf washing up the shore', 'sea waves'),
                   ('waves', 'ocean', 'beach', 'sea', 'surf')),
    'waves/pebbles': ('waves', dict(surface='pebbles'), ('waves breaking on a pebble beach', 'surf dragging over shingle', 'sea on stones'),
                      ('waves', 'pebble beach', 'shore', 'coast')),
    'underwater/deep': ('underwater', {}, ('underwater ambience with bubbles', 'muffled sound under water', 'bubbles rising in deep water'),
                        ('underwater', 'submerged', 'diving', 'ocean depths')),
    'forest/morning': ('forest', {}, ('birds singing in a forest', 'a quiet forest in the morning with birdsong', 'birdsong and a light breeze'),
                       ('forest', 'birds', 'birdsong', 'nature', 'morning', 'garden', 'meadow')),
    'night/crickets': ('night', {}, ('crickets chirping at night', 'night insects in a field', 'a summer night with crickets'),
                       ('night', 'crickets', 'insects', 'evening', 'camping night')),
    'room_tone/quiet': ('room_tone', {}, ('the quiet hum of an empty room', 'room tone', 'faint ventilation in a quiet room'),
                        ('room tone', 'office', 'indoor ambience', 'silence', 'air conditioning')),
    'city/distant': ('city', {}, ('distant city traffic', 'the far hum of a city', 'a distant highway'),
                     ('city', 'traffic', 'urban', 'street', 'downtown')),
    'space_hum/ship': ('space_hum', {}, ('the low hum of a spaceship engine room', 'a deep machine drone', 'a science fiction ship interior'),
                       ('space hum', 'spaceship', 'engine room', 'server room', 'reactor', 'sci-fi ambience')),
    'crackle/vinyl': ('crackle', {}, ('vinyl record crackle', 'crackling static from an old record', 'soft pops and crackle'),
                      ('vinyl', 'record crackle', 'lo-fi', 'old radio', 'static')),
    'sparkle/magic': ('sparkle', {}, ('a magical shimmering sparkle', 'twinkling chimes in the air', 'fairy dust glitter'),
                      ('sparkle', 'magic', 'shimmer', 'fairy', 'enchanted', 'glitter')),
}
ONCE = {
    'thunder/clap': ('thunder', {}, 7.0, ('a clap of thunder and a rolling rumble', 'thunder cracking overhead', 'a thunderclap'),
                     ('thunder', 'lightning', 'storm', 'thunderclap')),
    'thunder/distant': ('thunder', dict(distance=0.8), 8.0, ('distant thunder rumbling', 'a far-off storm', 'low rolling thunder'),
                        ('thunder', 'distant storm', 'rumble')),
}
# what the audit finds in any scene of many small events and is the scene: single loud drops and crackles (clicks),
# events after a lull (tail), a bird's or a bubble's rising chirp (rising_glide), the comb of a steady insect or chime
ALLOWS = ('clicks', 'tail', 'silence', 'rising_glide', 'ring')
AXIS = {'intensity': 'energy', 'density': 'energy', 'size': 'size', 'bright': 'bright', 'width': 'width',
        'distance': 'space'}


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


for _id, (_scene, _fixed, _says, _uses) in SCENES.items():
    _short = float(texture.SCENES[_scene]['min_loop'])
    design(_id, kind='bed', says=_says, uses=_uses + ('ambience', 'background', 'loop'), tags=('nature', 'ambience'),
           domains=('ambience', 'game', 'motion'), dur=max(_short, 12.0), axes=_axes(_scene), loop=True,
           allows=ALLOWS)(_bed(_scene, _fixed, _short))

for _id, (_scene, _fixed, _dur, _says, _uses) in ONCE.items():
    def _fn(c, r, scene=_scene, fixed=_fixed):
        return texture.render(scene, c.seconds(), r, **_controls(c, scene, fixed))
    design(_id, kind='noise', says=_says, uses=_uses, tags=('nature', 'weather'), domains=('ambience', 'game', 'motion'),
           dur=_dur, axes=_axes(_scene), allows=ALLOWS + ('stacked', 'pre_swell'))(_fn)
