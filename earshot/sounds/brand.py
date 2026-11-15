"""Brand figures: sonic logos, openings and endings, each playable by any voice.

An audio logo is short (one to three seconds), has few notes (two to five), moves upward or opens outward, and ends
on a note or chord that is left to ring in a larger room than the product's interface sounds use. The figures here
are shapes of that kind (a rolled open chord, two notes a fifth apart, a rise that lands high), meant as a starting
point in the product's own key and voice: `earshot make "logo reveal" --voice felt_piano --pitch D4`.
"""
from . import figure
from .registry import design

FIGURES = {
    'logo/bloom': dict(
        kind='stinger', says=('a slowly rolled open major chord ringing out in a hall',
                              'four notes spread upward into a long warm chord'),
        uses=('logo', 'logo reveal', 'brand', 'sonic logo', 'ident', 'intro', 'startup'), prior=1.15, allows=('tail',),
        fig=dict(steps=(0, 7, 16, 14), gap=(0.13, 0.13, 0.16), root='D4', vel=0.6, vels=(0.9, 0.85, 1.0, 0.8),
                 gate=3.0, last=3.0, room='hall', accent=2, spread=0.5, voice='felt_piano')),
    'logo/duo': dict(
        kind='stinger', says=('two notes a fifth apart, the second ringing long',
                              'a simple two-note signature in a hall'),
        uses=('logo', 'brand', 'sonic logo', 'ident', 'signature'), allows=('tail',),
        fig=dict(steps=(0, 7), gap=0.24, root='G4', vel=0.6, vels=(0.9, 1.0), gate=2.4, last=2.4, room='hall')),
    'logo/motif': dict(
        kind='stinger', says=('a four-note melodic signature that ends high and rings',
                              'up a fifth, down to the third, up to the octave'),
        uses=('logo', 'brand', 'mnemonic', 'signature', 'sting'), allows=('tail',),
        fig=dict(steps=(0, 7, 4, 12), gap=(0.17, 0.17, 0.2), root='C5', vel=0.6, vels=(0.9, 1.0, 0.85, 1.0), last=2.0,
                 room='hall', spread=0.4)),
    'startup/rise': dict(
        kind='stinger', says=('four slow rising notes of a major chord, a calm opening',
                              'a gentle ascending arpeggio that rings out'),
        uses=('startup', 'welcome', 'app open', 'intro', 'power on', 'hello', 'open'), allows=('tail',),
        fig=dict(steps=(0, 7, 12, 16), gap=0.16, root='F4', vel=0.52, vels=(0.85, 0.9, 0.95, 1.0), gate=1.6, last=2.0,
                 room='hall', spread=0.5)),
    'shutdown/fall': dict(
        kind='stinger', says=('four slow falling notes of a major chord, a calm ending',
                              'a gentle descending arpeggio that settles low'),
        uses=('shutdown', 'goodbye', 'outro', 'power off', 'close', 'end'), allows=('tail',),
        fig=dict(steps=(16, 12, 7, 0), gap=0.16, root='F4', vel=0.48, vels=(1.0, 0.92, 0.88, 0.95), gate=1.2, last=1.6,
                 room='hall', accent=0, spread=0.5)),
}
figure.register(design, FIGURES, domains=('motion', 'ui', 'music'))
