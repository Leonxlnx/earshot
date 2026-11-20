"""Game figures: the tonal rewards, signals and jingles of a game (coins, pickups, power-ups, level-ups, win and
lose phrases, menu blips), each playable by any voice.

Game feedback is musical shorthand that players have learned: a reward rises and is consonant, a quick high
two-note ding is a coin, a rising arpeggio is power gained and a falling one power lost, a fanfare ends on a long
high note, a loss falls in a minor chord. The figures here are those shapes, not anyone's melody. They take the
theme's voice and key like the interface figures (earshot.sounds.ui), so a game's menu, its pickups and its
jingles come from one instrument family. The chip versions live in earshot.sounds.retro; the physical action
sounds (jumps, hits, explosions) in earshot.sounds.action.
"""
from . import figure
from .registry import design

FIGURES = {
    # ---- rewards
    'coin/bright': dict(
        kind='chime', says=('two quick high bell notes, the second a fourth higher and ringing',
                            'a bright metallic two-note ding'),
        uses=('coin', 'pickup', 'collect', 'score', 'money'), prior=1.15,
        fig=dict(steps=(0, 5), gap=0.07, root='B5', vel=0.6, vels=(0.8, 1.0), last=0.45, voice='glockenspiel')),
    'coin/soft': dict(
        kind='chime', says=('one small high note with a faint octave above', 'a single soft high note'),
        uses=('coin', 'pickup', 'collect', 'point'),
        fig=dict(steps=(0, 12), gap=0.012, root='E6', vel=0.5, vels=(1.0, 0.4), last=0.35, accent=0, spread=0.0)),
    'pickup/blip': dict(
        kind='chime', says=('three very fast rising plucked notes', 'a tiny quick upward run of notes'),
        uses=('pickup', 'item', 'collect', 'grab', 'ammo'), prior=1.1,
        fig=dict(steps=(0, 4, 7), gap=0.036, root='C6', vel=0.5, vels=(0.75, 0.85, 1.0), last=0.22,
                 voice='kalimba')),
    'gem/sparkle': dict(
        kind='figure', says=('a fast sparkling run of four high bell notes', 'a quick glittering upward flourish'),
        uses=('gem', 'jewel', 'treasure', 'crystal', 'rare item', 'star', 'sparkle'),
        fig=dict(steps=(0, 7, 12, 19), gap=0.045, root='E6', vel=0.5, vels=(0.7, 0.8, 0.9, 1.0), last=0.7,
                 room='studio', voice='glockenspiel', spread=0.5)),
    'heal/chime': dict(
        kind='figure', says=('a soft slow rising arpeggio of a major ninth chord', 'five gentle rising notes'),
        uses=('heal', 'restore', 'health', 'potion', 'revive', 'recover'),
        fig=dict(steps=(0, 4, 7, 11, 14), gap=0.09, root='G5', vel=0.42, vels=(0.8, 0.85, 0.9, 0.95, 1.0),
                 last=1.0, room='hall', spread=0.5)),
    # ---- power
    'power_up/rise': dict(
        kind='figure', says=('a quick rising arpeggio over two octaves', 'seven fast notes climbing a major chord'),
        uses=('power up', 'boost', 'upgrade', 'grow', 'buff', 'charge up'), prior=1.1,
        allows=('rising_glide',),            # seven struck notes 52 ms apart read as one glide to the audit
        fig=dict(steps=(0, 4, 7, 12, 16, 19, 24), gap=0.052, root='C5', vel=0.55,
                 vels=(0.7, 0.74, 0.78, 0.83, 0.88, 0.94, 1.0), last=0.55, room='studio', spread=0.4)),
    'power_down/fall': dict(
        kind='figure', says=('a falling arpeggio of a minor chord', 'four notes dropping quickly, darker each time'),
        uses=('power down', 'shrink', 'lose power', 'drain', 'debuff'),
        fig=dict(steps=(12, 7, 3, 0), gap=0.07, root='C5', vel=0.5, vels=(1.0, 0.9, 0.8, 0.75), last=0.4, accent=0)),
    'level_up/fanfare': dict(
        kind='stinger', says=('four rising notes ending on a held high note, a short bright fanfare',
                              'a quick ascending major arpeggio with a long last note'),
        uses=('level up', 'upgrade', 'achievement', 'rank up', 'new skill', 'extra life'), prior=1.1,
        fig=dict(steps=(0, 4, 7, 12, 16), gap=(0.085, 0.085, 0.12, 0.0), root='G5', vel=0.62,
                 vels=(0.75, 0.82, 0.9, 1.0, 0.7), last=1.0, room='studio', accent=3, spread=0.4)),
    # ---- endings
    'win/fanfare': dict(
        kind='stinger', says=('a rising six-note major fanfare ending on a long high chord',
                              'a triumphant ascending phrase'),
        uses=('win', 'victory', 'level complete', 'stage clear', 'fanfare', 'triumph', 'quest complete'),
        prior=1.1, allows=('tail',),
        fig=dict(steps=(0, 4, 7, 12, 16, 19, 24), gap=(0.095, 0.095, 0.095, 0.21, 0.15, 0.0), root='C5', vel=0.66,
                 vels=(0.75, 0.8, 0.85, 0.95, 0.95, 1.0, 0.75), last=1.4, room='hall', accent=5, spread=0.5)),
    'lose/descend': dict(
        kind='stinger', says=('three falling notes of a minor chord landing on a low minor chord that rings out',
                              'a descending sad phrase that ends on a dark held chord'),
        uses=('lose', 'game over', 'death', 'defeat', 'fail', 'time up'), prior=1.1, allows=('tail',),
        # the line falls through the minor triad and lands an octave down on the full chord (root, minor third,
        # fifth rolled 12 ms apart), held longest and loudest: the fall reads at once, the ending is weighty
        fig=dict(steps=(7, 3, 0, -12, -9, -5), gap=(0.16, 0.16, 0.22, 0.012, 0.012), root='C5', vel=0.58,
                 vels=(0.9, 0.85, 0.8, 1.0, 0.75, 0.7), last=1.4, room='hall', accent=3, spread=0.3)),
    'checkpoint/chime': dict(
        kind='figure', says=('three rising notes: root, fifth and octave, the last ringing',
                             'a calm open three-note figure'),
        uses=('checkpoint', 'save point', 'progress saved', 'flag', 'milestone'),
        fig=dict(steps=(0, 7, 12), gap=0.1, root='D5', vel=0.55, vels=(0.8, 0.9, 1.0), last=0.85, room='studio')),
    'secret/reveal': dict(
        kind='figure', says=('a quick rising whole-tone run, a mysterious flourish',
                             'five fast notes climbing in whole steps'),
        allows=('rising_glide',),            # five discrete notes a whole step apart read as one glide to the audit
        uses=('secret', 'discover', 'mystery', 'hidden', 'reveal', 'unlock'),
        fig=dict(steps=(0, 2, 4, 6, 8), gap=0.062, root='A5', vel=0.48, vels=(0.7, 0.78, 0.85, 0.92, 1.0), last=0.8,
                 room='studio', spread=0.5)),
    # ---- menus
    'menu/move': dict(
        kind='tick', says=('one very short soft note, a small tonal blip', 'a tiny muted high note'),
        uses=('menu', 'cursor', 'menu move', 'option', 'hover'),
        fig=dict(steps=(0,), root='A5', vel=0.32, last=0.06, room='booth', voice='marimba',
                 avoid=('metal', 'glass', 'electric'), longest=0.2)),
    'menu/select': dict(
        kind='chime', says=('two very quick notes leaping up a fifth', 'a short bright upward two-note blip'),
        uses=('menu select', 'confirm', 'start', 'accept', 'ok'),
        fig=dict(steps=(0, 7), gap=0.05, root='A5', vel=0.5, vels=(0.8, 1.0), last=0.3)),
    'menu/back': dict(
        kind='chime', says=('two very quick notes dropping a fifth', 'a short soft downward two-note blip'),
        uses=('menu back', 'cancel', 'back', 'close', 'pause'),
        fig=dict(steps=(7, 0), gap=0.05, root='A5', vel=0.42, vels=(1.0, 0.85), last=0.22, accent=0)),
}
figure.register(design, FIGURES, domains=('game',))
