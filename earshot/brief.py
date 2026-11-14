"""Read a request for a sound, and find the designs that answer it.

    from earshot import brief
    b = brief.read('a soft wooden success chime for a finance app, 400 ms, in D')
    # {'concepts': ['success', 'bell'], 'axes': {'hard': -0.5, 'energy': -0.3}, 'material': 'wood', 'domain': 'ui',
    #  'dur': 0.4, 'key': 'D', ...}
    brief.rank(b)[:3]          # [{'id': ..., 'score': ..., 'why': [...]}, ...]

A request names a job ("success", "coin pickup", "logo reveal") or a sound ("chime", "whoosh", "rain"), often both,
plus how it should feel ("soft", "bigger", "not harsh"), what it is made of or played by ("wooden", "on a kalimba",
"8-bit"), where it plays ("for a platformer") and hard numbers (length, pitch, key, how many). `read` separates those
with the tables in earshot.sounds.lexicon, without any model: jobs and sounds become canonical words, feelings become
offsets on the shared axes (negations flip them, intensifiers scale them), the rest are parameters. `rank` scores every
design by the words it shares with the request, counting a design's jobs (`uses`) most, then its name, then how it is
described (`says`) and tagged; rare words count more than common ones. Words nothing recognises are returned as
`unknown`, so the caller can see what was not understood.
"""
import math
import re

from .sounds import lexicon as L

NOTE = re.compile(r'(?<![A-Za-z0-9#])([A-Ga-g])([#b]?)(-?\d)(?![A-Za-z0-9])')
HZ = re.compile(r'(\d+(?:\.\d+)?)\s*hz\b', re.I)
DUR = re.compile(r'(?<![\w.-])(\d+(?:\.\d+)?)\s*(ms|msec|milliseconds?|s|sec|secs|seconds?)\b', re.I)
COUNT = re.compile(r'(\d+)\s*(?:variations?|variants?|versions?|takes?|options?|alternatives?)\b', re.I)
KEY = re.compile(r'\b(?:in|key of|in the key of)\s+([A-G][#b]?)(?:[\s-]*(major|minor|maj|min))?(?![A-Za-z0-9#])')
WORD = re.compile(r"[a-z0-9#]+(?:[-'][a-z0-9]+)*")
VOICE_WORDS = {'piano': 'felt_piano', 'grand': 'grand_piano', 'kalimba': 'kalimba', 'mbira': 'kalimba',
               'marimba': 'marimba', 'glockenspiel': 'glockenspiel', 'glock': 'glockenspiel',
               'vibraphone': 'vibraphone', 'vibes': 'vibraphone', 'celesta': 'celesta', 'xylophone': 'xylophone',
               'harp': 'harp', 'guitar': 'nylon_guitar', 'rhodes': 'electric_piano', 'epiano': 'electric_piano',
               'e-piano': 'electric_piano', 'handpan': 'handpan', 'pizzicato': 'pizzicato', 'koto': 'koto',
               'harpsichord': 'harpsichord', 'bowl': 'singing_bowl', 'triangle': 'triangle', 'crotales': 'crotales'}
VOICE_PHRASES = {'music box': 'music_box', 'electric piano': 'electric_piano', 'felt piano': 'felt_piano',
                 'grand piano': 'grand_piano', 'steel drum': 'steel_drum', 'steel pan': 'steel_drum',
                 'singing bowl': 'singing_bowl', 'wine glass': 'wine_glass', 'hand bell': 'hand_bell',
                 'church bell': 'church_bell', 'tubular bell': 'tubular_bell', 'wind chime': 'wind_chime',
                 'wind chimes': 'wind_chime', 'thumb piano': 'kalimba', 'upright bass': 'upright_bass',
                 'double bass': 'upright_bass'}
LOOP_WORDS = ('loop', 'looping', 'loopable', 'seamless', 'looped')
CONTEXT_WORDS = ('for', 'when', 'while', 'as', 'after', 'before', 'during', 'whenever', 'once')
CONTEXT_WEIGHT, WORD_WEIGHT, HINT_WEIGHT = 0.5, 0.7, 0.35
FIELD_WEIGHT = (('uses', 3.0), ('name', 2.2), ('says', 1.6), ('tags', 1.2))
_rev = None


def stem(w):
    """a word without its plural or verb ending (a crude stemmer: enough to match 'chimes', 'clicking', 'popped')"""
    for end, keep in (('ies', 'y'), ('sses', 'ss'), ('ing', ''), ('ed', ''), ('es', ''), ('s', '')):
        if w.endswith(end) and len(w) - len(end) >= 3 and not w.endswith('ss'):
            return w[:len(w) - len(end)] + keep
    return w


def _reverse():
    """{word or phrase: canonical word} over lexicon.CONCEPTS"""
    global _rev
    if _rev is None:
        r = {c.replace('_', ' '): c for c in L.CONCEPTS}        # a canonical word always means itself
        for canon, others in L.CONCEPTS.items():
            for w in others:
                r.setdefault(w, canon)
                r.setdefault(' '.join(stem(p) for p in w.split()), canon)
        _rev = r
    return _rev


def words(text):
    return WORD.findall(text.lower())


def modifier(w):
    """the lexicon adjective a word is (also from its comparative: 'brighter', 'bigger', 'heavier'), or None"""
    if w in L.MODIFIERS:
        return w
    for end, keep in (('ier', 'y'), ('er', ''), ('r', '')):
        if w.endswith(end):
            base = w[:len(w) - len(end)] + keep
            for cand in (base, base[:-1] if len(base) > 2 and base[-1] == base[-2] else None):
                if cand in L.MODIFIERS:
                    return cand
    return None


def concepts_of(text):
    """(canonical words, leftover words) of a phrase: the longest lexicon phrases first, stop words dropped"""
    rev, toks = _reverse(), words(text)
    out, rest, i = [], [], 0
    while i < len(toks):
        for n in (3, 2, 1):
            ph = ' '.join(toks[i:i + n])
            c = rev.get(ph) or rev.get(' '.join(stem(p) for p in toks[i:i + n]))
            if c and len(toks[i:i + n]) == n:
                out.append(c)
                i += n
                break
        else:
            if toks[i] not in L.STOP:
                rest.append(stem(toks[i]))
            i += 1
    return out, rest


def read(text):
    """a request as a brief (module doc): {'text', 'concepts', 'words', 'axes', 'material', 'voice', 'domain', 'dur',
    'pitch', 'key', 'scale', 'n', 'loop', 'unknown'}; absent things are None / empty. `concepts` are the sound or job
    asked for, `context` the canonical words after "for" / "when" (where it plays: they count half), `hints` the
    adjectives (they break ties between members of a family: "crisp" finds whoosh_fast/crisp)"""
    b = {'text': text, 'concepts': [], 'context': [], 'words': [], 'hints': [], 'axes': {}, 'material': None,
         'voice': None, 'domain': None, 'dur': None, 'pitch': None, 'key': None, 'scale': None, 'n': None,
         'loop': False, 'unknown': []}
    m = KEY.search(text)
    if m:
        b['key'] = m.group(1)[0].upper() + m.group(1)[1:]
        b['scale'] = {'maj': 'major', 'min': 'minor'}.get((m.group(2) or '')[:3].lower()) if m.group(2) else None
        text = text[:m.start()] + ' ' + text[m.end():]
    m = COUNT.search(text)
    if m:
        b['n'] = max(1, min(12, int(m.group(1))))
        text = text[:m.start()] + ' ' + text[m.end():]
    m = HZ.search(text)
    if m:
        b['pitch'] = float(m.group(1))
        text = text[:m.start()] + ' ' + text[m.end():]
    m = DUR.search(text)
    if m:
        v = float(m.group(1))
        b['dur'] = v / 1000.0 if m.group(2).lower().startswith('m') else v
        text = text[:m.start()] + ' ' + text[m.end():]
    m = NOTE.search(text)
    if m:
        b['pitch'] = b['pitch'] or m.group(1).upper() + m.group(2) + m.group(3)
        text = text[:m.start()] + ' ' + text[m.end():]
    low = text.lower()
    for ph, v in VOICE_PHRASES.items():
        if ph in low:
            b['voice'] = v
            low = low.replace(ph, ' ')
    rev, toks = _reverse(), WORD.findall(low)
    i, scale, ctx = 0, 1.0, False
    while i < len(toks):
        t = toks[i]
        if t in CONTEXT_WORDS and b['concepts']:
            ctx = True                 # "an impact FOR a logo reveal": what follows is where it plays, not what it is
        if t in L.NEGATIONS:
            scale *= L.NEGATIONS[t]
        elif t in L.INTENSIFIERS and i + 1 < len(toks) and (modifier(toks[i + 1]) or toks[i + 1] in L.INTENSIFIERS):
            scale *= L.INTENSIFIERS[t]
        else:
            hit = False
            for n in (3, 2, 1):
                seg = toks[i:i + n]
                if len(seg) < n:
                    continue
                c = rev.get(' '.join(seg)) or rev.get(' '.join(stem(p) for p in seg))
                if c and (n > 1 or not modifier(t)):
                    if c not in b['concepts'] + b['context']:
                        b['context' if ctx else 'concepts'].append(c)
                    i += n - 1
                    hit = True
                    break
            if t in LOOP_WORDS:
                b['loop'] = hit = True
            if t in L.DOMAIN_WORDS:
                b['domain'] = b['domain'] or L.DOMAIN_WORDS[t]
                hit = hit or t not in rev
            if t in L.MATERIALS:
                b['material'] = b['material'] or L.MATERIALS[t]
                hit = True
            if t in VOICE_WORDS and b['voice'] is None:
                b['voice'] = VOICE_WORDS[t]
                hit = True
            if not hit and modifier(t):
                for ax, d in L.MODIFIERS[modifier(t)].items():
                    b['axes'][ax] = b['axes'].get(ax, 0.0) + d * scale
                if scale > 0:
                    b['hints'].append(modifier(t))
                hit = True
            if not hit and t not in L.STOP and not t.isdigit():
                b['words'].append(stem(t))
                b['unknown'].append(t)
            scale = 1.0
        i += 1
    b['axes'] = {a: round(max(-1.0, min(1.0, v)), 2) for a, v in b['axes'].items() if abs(v) >= 0.05}
    return b


def _terms(row):
    """a design's words by field: {'uses': set, 'name': set, 'says': set, 'tags': set}"""
    uses, name, says, tags = set(), set(), set(), set()
    for u in row['uses']:
        c, rest = concepts_of(u)
        uses.update(c)
        uses.update(rest)
    fam, _, nm = row['id'].partition('/')
    for part in (fam, nm):
        c, rest = concepts_of(part.replace('_', ' '))
        name.update(c)
        name.update(rest)
    for s in row['says']:
        c, rest = concepts_of(s)
        says.update(c)
        says.update(rest)
    for t in list(row['tags']) + [row['kind'], row.get('material') or '', row.get('voice') or '']:
        c, rest = concepts_of(str(t).replace('_', ' '))
        tags.update(c)
        tags.update(rest)
    return {'uses': uses, 'name': name, 'says': says, 'tags': tags}


def index(rows):
    """the search index of a catalogue: [(row, terms)], and each word's weight (rarer = heavier)"""
    docs = [(r, _terms(r)) for r in rows]
    df = {}
    for _, t in docs:
        for w in set().union(*t.values()):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(docs))
    return docs, {w: 1.0 + math.log(n / k) for w, k in df.items()}


def rank(b, rows=None, top=8, idx=None):
    """the designs that answer a brief, best first: [{'id', 'score', 'why'}]. rows: catalogue rows (default: every
    design); a design must share at least one word with the request to be listed"""
    if idx is None:
        if rows is None:
            from .sounds import registry
            rows = registry.rows()
        idx = index(rows)
    docs, idf = idx
    q = {**{w: HINT_WEIGHT for w in b.get('hints', ())}, **{w: WORD_WEIGHT for w in b['words']},
         **{w: CONTEXT_WEIGHT for w in b.get('context', ())}, **{w: 1.0 for w in b['concepts']}}
    out = []
    for row, t in docs:
        score, why = 0.0, []
        for w, qw in q.items():
            for field, fw in FIELD_WEIGHT:
                if w in t[field]:
                    score += fw * idf.get(w, 1.0) * qw
                    if qw > HINT_WEIGHT:
                        why.append(w)
                    break
        if not why:
            continue
        score *= row.get('prior', 1.0)
        if b['domain'] and b['domain'] in row['domains']:
            score *= 1.2
        mat = b['material']
        if mat:
            if mat == row.get('material') or mat in row['tags']:
                score *= 1.5
            elif mat == 'retro' or 'retro' in row['tags']:
                score *= 0.5                                   # chip sound only when asked for, and then only chip
        elif 'retro' in row['tags']:
            score *= 0.6
        if b['loop'] != bool(row['loop']):
            score *= 0.5 if b['loop'] else 0.8
        out.append({'id': row['id'], 'score': round(score, 3), 'why': why})
    out.sort(key=lambda r: (-r['score'], r['id']))
    return out[:top]
