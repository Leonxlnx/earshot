"""The sound bank (earshot.forge.bank: the forge's delivered sounds for motion design and film) as catalogue designs,
so `earshot make` finds a whoosh, a riser or a key press the same way it finds everything else.

A bank entry keeps its id (`whoosh_fast/crisp`). It answers to `pitch` and `dur` where its recipe has those knobs
(earshot.forge.knobs: the recipe's own parameters move, nothing is resampled); the axes are not wired to bank
recipes. Members that only carry a number in the bank (`pop/3`, `shimmer/m2`) are left out of the catalogue: they
stay reachable through `earshot sfx`. Bank sounds rank under designs written for the catalogue (prior 0.8; the
stylised `pro_` families 0.7), so a request that fits both gets the newer design.
"""
import re

from .registry import design

# bank category -> (kind, jobs it plays, domains, audit findings that are its design)
CATEGORIES = {
    'click': ('click', ('click', 'tap', 'select'), ('ui', 'motion'), ()),
    'pro_click': ('click', ('click', 'tick', 'select'), ('ui', 'motion'), ()),
    'tap': ('tap', ('tap', 'touch', 'land'), ('ui', 'motion'), ()),
    'pro_tap': ('tap', ('tap', 'touch'), ('ui', 'motion'), ()),
    'hover_tick': ('tick', ('hover', 'tick', 'scroll'), ('ui', 'motion'), ()),
    'pro_hover_tick': ('tick', ('hover', 'tick', 'counter'), ('ui', 'motion'), ()),
    'toggle': ('click', ('toggle', 'switch', 'lock'), ('ui', 'motion'), ()),
    'typing_key': ('click', ('type', 'keyboard', 'key press'), ('ui', 'motion'), ()),
    'pro_typing_key': ('tick', ('type', 'text reveal', 'kinetic type'), ('motion',), ()),
    'pro_typing_run': ('texture', ('typing run', 'text reveal'), ('motion',), ('tail', 'silence')),
    'knock': ('thud', ('knock', 'land', 'drop on desk'), ('ui', 'motion'), ()),
    'ui_pop': ('pop', ('pop', 'appear', 'open', 'add'), ('ui', 'motion'), ()),
    'pro_pop': ('pop', ('pop', 'appear', 'badge'), ('ui', 'motion'), ('rising_glide',)),
    'pro_pop_in': ('pop', ('pop in', 'appear', 'icon lands'), ('ui', 'motion'), ()),
    'drip': ('pop', ('water drop', 'drip', 'bubble'), ('ui', 'motion', 'game'), ('rising_glide',)),
    'shutter': ('click', ('camera shutter', 'screenshot', 'record', 'snap zoom'), ('ui', 'motion'), ()),
    'mallet': ('note', ('bell', 'chime', 'text', 'accent'), ('ui', 'motion', 'music'), ()),
    'pluck': ('note', ('pluck', 'kalimba', 'accent'), ('ui', 'motion', 'music'), ()),
    'notify': ('chime', ('notify', 'bell', 'success'), ('ui', 'motion'), ()),
    'swish': ('whoosh', ('swish', 'flick', 'text slide'), ('motion', 'ui'), ()),
    'swipe': ('whoosh', ('swipe', 'page turn', 'card slide'), ('motion', 'ui'), ()),
    'whoosh_fast': ('whoosh', ('whoosh', 'whip pan', 'fly in', 'transition'), ('motion', 'game'), ()),
    'whoosh_big': ('whoosh', ('whoosh', 'big transition', 'pass by'), ('motion',), ()),
    'pro_whoosh_big': ('whoosh', ('whoosh', 'swell', 'low swell into a cut'), ('motion',), ()),
    'riser': ('rise', ('riser', 'build', 'tension'), ('motion',), ('truncated', 'rising_glide')),
    'pro_riser': ('rise', ('riser', 'stutter build'), ('motion',), ('truncated', 'rising_glide')),
    'reverse_swell': ('rise', ('reverse', 'swell', 'reverse cymbal'), ('motion',), ('truncated',)),
    'downlifter': ('fall', ('downlifter', 'fall', 'exit'), ('motion',), ()),
    'pro_zip': ('zap', ('zip', 'fast sweep'), ('motion',), ('rising_glide', 'truncated')),
    'impact': ('hit', ('impact', 'reveal', 'title slam'), ('motion', 'game'), ('stacked', 'pre_swell')),
    'pro_impact': ('hit', ('impact', 'sub impact'), ('motion',), ('truncated', 'stacked', 'pre_swell')),
    'cut_hit': ('hit', ('hit on a cut', 'sub hit', 'punch'), ('motion',), ()),
    'sub_drop': ('boom', ('sub drop', 'bass drop'), ('motion',), ()),
    'pro_sub_drop': ('boom', ('sub drop', 'low drop after a cut'), ('motion',), ()),
    'logo_sting': ('stinger', ('logo', 'logo reveal', 'sting'), ('motion',), ('stacked', 'pre_swell', 'rising_glide')),
    'pro_logo_sting': ('stinger', ('logo', 'sting'), ('motion',), ('stacked', 'pre_swell', 'rising_glide')),
    'glitch': ('noise', ('glitch', 'error', 'stutter'), ('motion', 'game'), ('clicks', 'silence', 'truncated')),
    'pro_glitch': ('noise', ('glitch', 'stutter into a cut'), ('motion',), ('clicks', 'silence', 'truncated')),
    'shimmer': ('texture', ('shimmer', 'chimes', 'sparkle'), ('motion', 'game'), ('tail',)),
    'pro_shimmer': ('texture', ('shimmer', 'sparkle', 'magic'), ('motion', 'game'), ('tail',)),
}
NUMBERED = re.compile(r'^(m?\d+|ding_\d+)$')
# members that are the first answer to a bare request whatever the newer designs: a plain "tap" or "click" in an
# interface is a soft tactile touch, not a note (the newer tap/tone is tonal)
FIRST = {'tap/soft': 1.05, 'click/soft': 1.0}


def _render(sound_id, takes):
    def fn(c, r):
        from ..forge import bank
        kw = {}
        if takes.get('pitch') and c._pitch is not None:
            kw['pitch'] = c.hz()
        if takes.get('dur') and c._dur is not None:
            kw['dur'] = c.seconds()
        y, _ = bank.render(sound_id, seed=int(r.integers(1, 2 ** 31 - 1)) if c.vary > 0 else None, checks=False, **kw)
        return y.astype('float64')
    return fn


# audit findings that are one member's design: a knock-knock is two taps 55 ms apart, which the audit measures as two
# separate attacks (at 0.01 and 0.07 s on every take): the thing itself, not layers stacked on one hit
MEMBER_ALLOWS = {'knock/double': ('stacked',)}


def register():
    """add every named bank member of the CATEGORIES to the catalogue"""
    from ..forge import bank, knobs
    for row in bank.list_sounds():
        cat, member = row['category'], str(row['member'])
        if cat not in CATEGORIES or NUMBERED.match(member):
            continue
        kind, uses, domains, allows = CATEGORIES[cat]
        allows = allows + MEMBER_ALLOWS.get(row['id'], ())
        info = bank.category_info(cat)
        k = knobs.describe(bank.get(row['id']))
        takes = {'pitch': bool(k.get('pitch')), 'dur': bool(k.get('dur'))}
        words = member.replace('_', ' ')
        says = (info['target'], f"{words}: {row.get('notes') or info.get('use') or info['target']}")
        pk = k.get('pitch') or {}
        design(row['id'], kind=kind, says=says, uses=uses, tags=(cat.replace('pro_', ''), words, 'bank'),
               domains=domains, dur=round(float(row['end_s'] or row['dur_s']), 3),
               pitch=(pk.get('note') or '').split(' ')[0] or None, allows=allows,
               prior=FIRST.get(row['id'], 0.7 if cat.startswith('pro_') else 0.8))(_render(row['id'], takes))


register()
