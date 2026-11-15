"""Read a request for a sound, and find the designs that answer it.

    from earshot import brief
    b = brief.read('a soft wooden success chime for a finance app, 400 ms, in D')
    # {'concepts': ['success', 'bell'], 'axes': {'hard': -0.5, 'energy': -0.3}, 'material': 'wood', 'domain': 'ui',
    #  'dur': 0.4, 'key': 'D', ...}
    brief.rank(b)[:3]          # [{'id': ..., 'score': ..., 'why': [...]}, ...]

A request names a job ("success", "coin pickup", "logo reveal") or a sound ("chime", "whoosh", "rain"), often both,
plus how it should feel ("soft", "bigger", "not harsh"), what it is made of or played by ("wooden", "on a kalimba",
"8-bit"), where it plays ("for a platformer") and hard numbers (length, pitch, key, how many). `read` separates those
with the tables in earshot.sounds.lexicon, without any model:
- jobs and sounds become canonical words, the longest phrase first ("level complete" before "complete"), articles
  inside a phrase skipped ("beat the level" = "beat level");
- the first one named is the head of the request; what follows "with" describes a part of it ("a bell with an airy
  whoosh") and what follows "for", "when", "during" ... once the head is named is its context (where it plays): both
  count less;
- "like a ..." is a comparison, not the job "like"; what follows "than" is a reference and is skipped; "the
  opposite of X" and "X but the other way" turn X into its opposite;
- negations flip adjectives ("not harsh") and turn a named job into one to avoid ("not an error");
- adjectives become offsets on the shared axes (intensifiers scale them) and hints that break ties in a family;
- an instrument named as the thing itself ("a kalimba note", "marimba hit") is asked for as a note of that voice,
  one named as the player ("a chime on a kalimba", "a bass boom") only sets the voice;
- "rising", "falling" ... give the direction of the gesture; "8-bit", "retro" ... make chip sound a hard filter;
- words nothing knows are returned as `unknown` and count only a little.
`rank` scores every design by what it shares with the request: a design's jobs (`uses`) count most, then its name,
then how it is described (`says`) and tagged; rare words count more than common ones (idf), the design's own family
job gets a bonus, and the result is scaled by the design's prior, the domain, material, loop, length and direction.
`why` names the words that decided.
"""
import math
import re

from .sounds import lexicon as L

NOTE = re.compile(r'(?<![A-Za-z0-9#])([A-Ga-g])([#b]?)(-?\d)(?![A-Za-z0-9])')
HZ = re.compile(r'(\d+(?:\.\d+)?)\s*(k?hz)\b', re.I)
DUR = re.compile(r'(?<![\w.-])(\d+(?:\.\d+)?)\s*(ms|msec|milliseconds?|s|sec|secs|seconds?|min|minutes?)\b', re.I)
LIMIT = re.compile(r'\b(under|below|less than|at most|max|maximum|no longer than|within|up to|shorter than|tops)\s*$',
                   re.I)
NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                'ten': 10, 'eleven': 11, 'twelve': 12, 'a couple of': 2, 'a few': 3, 'several': 4}
COUNT = re.compile(r'\b(\d+|' + '|'.join(NUMBER_WORDS) + r')\s+(?:different\s+|unique\s+)?'
                   r'(?:variations?|variants?|versions?|takes?|options?|alternatives?)\b', re.I)
FIGURE = re.compile(r'\b(one|two|three|four|five|\d)-(note|tone|step|pulse|beat)s?\b', re.I)
KEY = re.compile(r'\b(?:in|key of|in the key of)\s+([A-G][#b]?)(?:[\s-]*(major|minor|maj|min))?(?![A-Za-z0-9#])')
WORD = re.compile(r"[a-z0-9#]+(?:[-'][a-z0-9#]+)*")
TOKEN = re.compile(r"[a-z0-9#]+(?:[-'][a-z0-9#]+)*|[,.;:!?()]")
PUNCT = frozenset(',.;:!?()')
LOOP_WORDS = ('loop', 'looping', 'loopable', 'seamless', 'looped', 'endless')
ONCE_WORDS = ('one-shot', 'oneshot')
CONTEXT_WORDS = ('for', 'when', 'while', 'as', 'after', 'before', 'during', 'whenever', 'once', 'where', 'until',
                 'whilst', 'if')
PLACE_WORDS = ('in', 'under', 'over', 'across', 'behind', 'beneath', 'inside')
# words whose ending is not an ending ('evening' is not 'even', 'string' is not 'str')
STEM_KEEP = frozenset(('evening', 'morning', 'ceiling', 'string', 'strings', 'thing', 'things', 'ring', 'rings',
                       'spring', 'springs', 'swing', 'sling', 'wing', 'wings', 'king', 'bring', 'during', 'nothing',
                       'something', 'anything', 'everything', 'bus', 'gas', 'news', 'lens', 'chaos', 'canvas'))
# a figure's shape in any words: "two-tone", "two-pulse", "two quick notes", "twice" -> '2-note'
FIGURE_TEXT = re.compile(r'\b(one|two|three|four|five|six|seven|\d)(?:-(?:note|tone|step|pulse|beat)s?\b'
                         r'|(?:\s+[a-z]+){0,2}\s+notes\b)', re.I)
ARTICLES = frozenset(('a', 'an', 'the', 'my', 'our', 'your', 'their', 'some', 'this', 'that', 'its', 'his', 'her'))
LIKE_JOB_NEXT = ('button', 'buttons', 'sound', 'tap', 'icon', 'animation', 'reaction', 'count', 'counter',
                 'feedback', 'click', 'action', 'and')
VOICE_PARAM_PREV = ('on', 'by', 'played', 'using', 'with', 'via', 'through')
# the sounds an instrument itself makes: "a piano key", "a bass note", "a bell ding" ask for a note of the voice
VOICE_OK = frozenset(('note', 'bass', 'piano', 'pluck', 'bell', 'beep', 'knock', 'hit', 'tap', 'cue'))
HYPHEN_SUFFIX = ('like', 'ish', 'style', 'styled', 'esque', 'type', 'themed', 'y')
OTHER_WAY = ('other way', 'the other way', 'opposite way', 'reverse direction', 'other direction')
TOGGLE_ON_NEXT = ('in', 'for', 'at', 'when', 'with', 'and', 'sound', 'click', 'snap', 'state', 'mode')
SCOPE_END = ('sound', 'sounds', 'noise', 'noises', 'thing', 'effect', 'sfx', 'but')
# weights of the query's parts and of the fields of a design
HEAD_WEIGHT, CONCEPT_WEIGHT, PART_WEIGHT, CONTEXT_WEIGHT, BROADER_WEIGHT, REF_WEIGHT = 1.0, 0.8, 0.6, 0.5, 0.4, 0.25
WORD_WEIGHT, HINT_WEIGHT, AXIS_WEIGHT = 0.6, 0.35, 0.12
FIELD_WEIGHT = {'uses': 3.0, 'name': 2.2, 'says': 1.6, 'tags': 1.2}
# the design's own family named in the request (by its very name: + EXACT_BONUS); its first job
FAMILY_BONUS, EXACT_BONUS, FIRST_USE_BONUS = 1.2, 0.3, 0.6
CONCEPT_IDF = 0.6                  # how much rarity counts for canonical words (1 = as for plain words)
STRETCH_KINDS = ('bed', 'pad', 'texture')   # rendered at any length asked for
LENGTH_SLOPE = 0.15                # how much a length mismatch costs per octave beyond the free half octave
_rev = None
_cache = {}


def stem(w):
    """a word without its plural or verb ending and a final e or doubled consonant (a crude stemmer: enough to
    match 'chimes', 'chiming', 'popped', 'rising' / 'rise')"""
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if w in STEM_KEEP:
        return w
    for end, keep in (('sses', 'ss'), ('ies', 'y'), ('ied', 'y'), ('ing', ''), ('ed', ''), ('es', ''), ('s', '')):
        if (w.endswith(end) and len(w) - len(end) >= 3 and not (end == 's' and w.endswith(('ss', 'us', 'is')))
                and not (end == 'ed' and w.endswith('eed'))):
            w = w[:len(w) - len(end)] + keep
            break
    if len(w) > 3 and w.endswith('e') and not w.endswith(('ee', 'ie')):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in 'slz' and w[-1].isalpha():
        w = w[:-1]
    return w


def _key(words_):
    return ' '.join(stem(p) for p in words_ if p not in ARTICLES)


def _reverse():
    """{kind: {stemmed phrase: value}} over the lexicon tables, and {kind + '=': {exact phrase: value}} (an exact
    form wins over a stemmed one: 'firing' is a shot, though it stems like 'fire'; where two stem alike, the shorter
    form wins: 'blocking' is 'block', a shield, not 'blocked', an error). Kinds: 'c' concept, 'v' voice,
    'd' direction, 'm' an adjective phrase ({axis: delta}), 'p' a material phrase, 'o' a domain phrase"""
    global _rev
    if _rev is None:
        r = {k + s: {} for k in 'cvdmpo' for s in ('', '=')}
        size = {}

        def put(kind, phrase, val):
            for form in {phrase, phrase.replace('-', ' ')}:
                ws = form.split()
                k = _key(ws)
                if k not in r[kind] or len(form) < size[kind, k]:
                    r[kind][k], size[kind, k] = val, len(form)
                r[kind + '='].setdefault(' '.join(w for w in ws if w not in ARTICLES), val)
        for canon in L.CONCEPTS:
            put('c', canon.replace('_', ' '), canon)            # a canonical word always means itself
        for canon, others in L.CONCEPTS.items():
            for w in others:
                put('c', w, canon)
        for table, kind in ((L.VOICES, 'v'), (L.DIRECTION, 'd'), (L.PHRASE_MODIFIERS, 'm'), (L.MATERIAL_PHRASES, 'p'),
                            (L.DOMAIN_PHRASES, 'o')):
            for w, v in table.items():
                put(kind, w, v)
        _rev = r
    return _rev


def _match(toks, i):
    """the lexicon entries of the longest phrase starting at token i (articles inside it skipped):
    (index after it, {kind: value}); kinds that match a shorter phrase are kept only where they belong together
    (a voice inside a concept phrase of an instrument: 'piano key', 'bass note')"""
    if toks[i] in PUNCT or toks[i] in ARTICLES:
        return i, {}
    idx, j = [], i
    while j < len(toks) and len(idx) < 4 and toks[j] not in PUNCT:
        if toks[j] not in ARTICLES:
            idx.append(j)
        j += 1
    rev, found = _reverse(), {}
    for n in range(len(idx), 0, -1):
        seg = [toks[k] for k in idx[:n]]
        key, exact = _key(seg), ' '.join(seg)
        for kind in 'cvdmpo':
            if kind in found:
                continue
            val = rev[kind + '='].get(exact, rev[kind].get(key))
            if val is None:
                continue
            if kind == 'c' and n == 1 and (modifier(seg[0]) or seg[0] in L.NOT_CONCEPT):
                continue                        # 'light rain': a single adjective is an adjective
            found[kind] = (val, n)
    if not found:
        return i, {}
    n = max(v[1] for v in found.values())
    keep = {k: v for k, (v, m) in found.items() if m == n}
    if 'v' in found and 'v' not in keep and keep.get('c') in VOICE_OK:
        keep['v'] = found['v'][0]
    return idx[n - 1] + 1, keep


def words(text):
    return WORD.findall(text.lower())


def modifier(w):
    """the lexicon adjective a word is (also from its comparative or adverb: 'brighter', 'bigger', 'heavier',
    'slowly', 'gently'), or None"""
    if w in L.MODIFIERS:
        return w
    for end, keep in (('ier', 'y'), ('er', ''), ('r', ''), ('ily', 'y'), ('ly', ''), ('ly', 'le')):
        if w.endswith(end) and len(w) > len(end) + 2:
            base = w[:len(w) - len(end)] + keep
            for cand in (base, base[:-1] if len(base) > 2 and base[-1] == base[-2] else None):
                if cand in L.MODIFIERS:
                    return cand
    return None


def _known(t):
    k = _key([t])
    return (any(k in d for d in _reverse().values()) or t in L.MATERIALS or t in L.DOMAIN_WORDS or modifier(t)
            or t in L.STOP or t in L.NEGATIONS or t in L.INTENSIFIERS or t in LOOP_WORDS or t in ONCE_WORDS
            or FIGURE.fullmatch(t))


def tokens(text):
    """the words and clause marks of a text, lower case; possessives dropped; a hyphenated word nothing knows is
    split into its parts ('xylophone-ish' -> 'xylophone', 'womp-womp' -> 'womp womp')"""
    out = []
    for t in TOKEN.findall(text.lower().replace('’', "'")):
        if t.endswith("'s") and t not in L.STOP:
            t = t[:-2]
        if '-' in t and not _known(t):
            parts = [p for p in t.split('-') if p]
            while len(parts) > 1 and parts[-1] in HYPHEN_SUFFIX:
                parts.pop()
            out.extend(parts)
        else:
            out.append(t)
    return out


def _feel(m):
    """the hint words of an adjective: itself and the axis it mostly moves, with the direction ('ax:size+')"""
    ax = max(L.MODIFIERS[m].items(), key=lambda kv: abs(kv[1]))
    return [m, f'ax:{ax[0]}{"+" if ax[1] > 0 else "-"}']


def concepts_of(text):
    """(canonical words, leftover words) of a phrase: the longest lexicon phrases first (voices as 'voice:<name>'),
    stop words dropped; adjectives stay as words"""
    c, rest, _, _ = _parse_plain(text)
    return c, rest


def _parse_plain(text):
    """(concepts, words, feel words, direction) of a design's own text, without the request's grammar"""
    toks = [t for t in tokens(text) if t not in PUNCT]
    out, rest, feel, d, i = [], [], [], 0, 0
    while i < len(toks):
        j, hit = _match(toks, i)
        if 'c' in hit:
            out.append(hit['c'])
        if 'v' in hit:
            out.append('voice:' + hit['v'])
        if 'd' in hit:
            d += hit['d']
            rest.extend(stem(t) for t in toks[i:j] if t not in L.STOP)
        if hit:
            i = j
            continue
        t = toks[i]
        m = modifier(t)
        if m:
            feel.extend(_feel(m))
        if t not in L.STOP:
            rest.append(stem(t))
        i += 1
    return out, rest, feel, d


def _numbers(b, text):
    """take the hard numbers out of a request into the brief: key, count, pitch, length; the text without them"""
    def cut(m, s):
        return s[:m.start()] + ' , ' + s[m.end():]
    m = KEY.search(text)
    if m:
        b['key'] = m.group(1)[0].upper() + m.group(1)[1:]
        b['scale'] = {'maj': 'major', 'min': 'minor'}.get((m.group(2) or '')[:3].lower()) if m.group(2) else None
        text = cut(m, text)
    m = COUNT.search(text)
    if m:
        v = m.group(1).lower()
        b['n'] = max(1, min(12, int(v) if v.isdigit() else NUMBER_WORDS[v]))
        text = cut(m, text)
    m = HZ.search(text)
    if m:
        b['pitch'] = float(m.group(1)) * (1000.0 if m.group(2).lower() == 'khz' else 1.0)
        text = cut(m, text)
    m = re.search(r'\bmiddle c\b', text, re.I)
    if m:
        b['pitch'] = b['pitch'] or 'C4'
        text = cut(m, text)
    m = re.search(r'\bhalf a second\b', text, re.I)
    if m:
        b['dur'] = 0.5
        text = cut(m, text)
    m = DUR.search(text)
    if m:
        v, u = float(m.group(1)), m.group(2).lower()
        v = v / 1000.0 if u.startswith('m') and not u.startswith('min') else v * 60.0 if u.startswith('min') else v
        lim = LIMIT.search(text[:m.start()])
        if lim:
            b['dur_max'] = v              # "under 80 ms": a ceiling, not a length to stretch to
            text = text[:lim.start()] + ' , ' + text[m.end():]
        else:
            b['dur'] = v
            text = cut(m, text)
    m = NOTE.search(text)
    if m:
        b['pitch'] = b['pitch'] or m.group(1).upper() + m.group(2) + m.group(3)
        text = cut(m, text)
    m = FIGURE.search(text)
    if m:
        v = m.group(1).lower()
        b['notes'] = int(v) if v.isdigit() else NUMBER_WORDS[v]
    return text


def _figure_words(text):
    """{'<n>-note'} for the figures a text describes ("a two-tone beep", "three rising notes", "twice")"""
    out = set()
    for m in FIGURE_TEXT.finditer(text):
        v = m.group(1).lower()
        out.add(f'{int(v) if v.isdigit() else NUMBER_WORDS[v]}-note')
    if re.search(r'\btwice\b', text, re.I):
        out.add('2-note')
    return out


def _status_keys():
    """the stemmed outcome words (lexicon.STATUS_WORDS)"""
    if 'status' not in _cache:
        _cache['status'] = {_key(w.split()) for w in L.STATUS_WORDS}
    return _cache['status']


def _domain(t):
    """the domain a word names (lexicon.DOMAIN_WORDS, also from its plural or verb form), or None"""
    if 'domain' not in _cache:
        _cache['domain'] = {stem(w): d for w, d in L.DOMAIN_WORDS.items()}
    return L.DOMAIN_WORDS.get(t) or _cache['domain'].get(stem(t))


def _sense():
    """{(domain, stemmed phrase): concept} of lexicon.DOMAIN_SENSE"""
    if 'sense' not in _cache:
        _cache['sense'] = {(d, _key(w.split())): c for (d, w), c in L.DOMAIN_SENSE.items()}
    return _cache['sense']


def _old_device(toks):
    """True when an old-sounding word comes within three words before a device word ("old Nintendo games")"""
    for i, t in enumerate(toks):
        if t in L.OLD_WORDS:
            for u in toks[i + 1:i + 4]:
                if u in PUNCT:
                    break
                if u in L.OLD_DEVICES:
                    return True
    return False


def _phrasal(toks):
    """{verb index: (concept, particle index)} of separable verbs with words between verb and particle
    ("picking something up", "putting a window away", "the mouse floats over"); adjacent ones are phrases"""
    if 'phrasal' not in _cache:
        _cache['phrasal'] = {stem(v): parts for v, parts in L.PHRASAL.items()}
    table, out = _cache['phrasal'], {}
    for i, t in enumerate(toks):
        parts = L.PHRASAL.get(t) or table.get(stem(t))
        if not parts:
            continue
        for k in range(i + 2, min(i + 5, len(toks))):
            if toks[k] in PUNCT:
                break
            if toks[k] in parts:
                out[i] = (parts[toks[k]], k)
                break
    return out


def _compose(b):
    """two jobs that together mean a third (lexicon.COMPOSE): the first is replaced, the second dropped"""
    for (a, c2), r in L.COMPOSE.items():
        lists = [k for k in ('concepts', 'parts', 'context') if a in b[k]]
        if lists and any(c2 in b[k] for k in ('concepts', 'parts', 'context')):
            for k in ('concepts', 'parts', 'context'):
                b[k] = [x for x in b[k] if x != c2 or c2 == r]
            k = lists[0]
            b[k] = list(dict.fromkeys(r if x == a else x for x in b[k]))


def _next_concept(toks, i):
    """the concept of the next content word from token i on (adjectives, materials, domains and fillers skipped),
    or None"""
    while i < len(toks) and toks[i] not in PUNCT:
        j, hit = _match(toks, i)
        if 'c' in hit:
            return hit['c']
        if hit:
            i = j
            continue
        t = toks[i]
        if t in CONTEXT_WORDS or t in PLACE_WORDS or t in ('with', 'to', 'but', 'than') or not (
                modifier(t) or t in L.MATERIALS or t in L.STOP or t in L.INTENSIFIERS or _domain(t)):
            return None
        i += 1
    return None


def read(text):
    """a request as a brief (module doc): {'text', 'concepts', 'parts', 'context', 'refs', 'not', 'words', 'hints',
    'axes', 'dir', 'material', 'voice', 'domain', 'dur', 'dur_max', 'pitch', 'key', 'scale', 'n', 'notes', 'loop',
    'unknown', 'context_words'}; absent things are None / empty. `concepts` are the sound or job asked for (the head
    first), `parts` what the request says it has ("with a soft shimmer"), `context` the canonical words after "for" /
    "when" (where it plays; `context_words` the plain words there), `refs` what it is compared with ("duller than an
    on-switch"), `not` the jobs the request rules out ("not an error"), `hints` the adjectives (they break ties
    between members of a family: "crisp" finds whoosh_fast/crisp), `dir` +1 for a rising gesture and -1 for a falling
    one, `dur_max` a ceiling ("under 80 ms"), `notes` the number of notes of a figure ("a two-note chime")"""
    b = {'text': text, 'concepts': [], 'parts': [], 'context': [], 'refs': [], 'not': [], 'words': [], 'hints': [],
         'axes': {},
         'dir': 0, 'material': None, 'voice': None, 'domain': None, 'dur': None, 'dur_max': None, 'pitch': None,
         'key': None, 'scale': None, 'n': None, 'notes': None, 'loop': None, 'unknown': []}
    toks = tokens(_numbers(b, text))
    n_tok = len(toks)
    st = {'scale': 1.0, 'neg': False, 'where': 'concepts', 'than': False, 'opposite': False, 'head': False,
          'any': False, 'object': False, 'simile': False, 'skipped': 0}
    domains, particles, src, ctx_words, outcomes = {}, [], {}, set(), []
    if _old_device(toks):
        b['material'] = 'retro'            # "old Nintendo games", "an ancient home computer"
    phrasal, skip = _phrasal(toks), set()

    def add(c, tok, where=None):
        if st['object'] and st['where'] == 'concepts':
            where, st['object'] = 'parts', False    # "parrying an attack": the attack is what is parried
        if st['simile'] and st['where'] == 'concepts':
            where = 'parts'                         # "a thump, like a door slamming": a comparison describes it
        where = where if where and st['where'] == 'concepts' else st['where']
        if st['opposite']:
            c, st['opposite'] = L.OPPOSITE.get(c, c), False
        if st['neg']:
            if c not in b['not']:
                b['not'].append(c)
            st['neg'], st['scale'] = False, 1.0
            return
        if all(c not in b[k] for k in ('concepts', 'parts', 'context')):
            b[where].append(c)
            src[c] = tok
        if where == 'concepts' and c not in L.GENERIC:
            st['head'] = True
        if where == 'concepts':
            st['any'] = True
            st['object'] = c in L.OBJECT_VERBS

    def axes(mods, scale):
        for ax, d in mods.items():
            b['axes'][ax] = b['axes'].get(ax, 0.0) + d * scale

    i = 0
    while i < n_tok:
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < n_tok else None
        prev = toks[i - 1] if i else None
        if i in skip:                       # the particle of a separable verb already read
            i += 1
            continue
        if t in PUNCT:
            st.update(than=False, neg=False, scale=1.0, simile=False)
            if st['where'] == 'parts':
                st['where'] = 'concepts'
            i += 1
            continue
        if t == 'but':
            st.update(than=False, neg=False, scale=1.0)
            i += 1
            continue
        if t == 'than':
            st['than'] = True                # "duller than an on-switch": the reference, not the request
            i += 1
            continue
        if st['than']:                       # the reference is the same kind of thing: it counts a little
            j, hit = _match(toks, i)
            if 'c' in hit and hit['c'] not in b['refs']:
                b['refs'].append(hit['c'])
            i = max(j, i + 1)
            continue
        if t == 'opposite' and nxt == 'of':
            st['opposite'] = True
            i += 2
            continue
        seg3 = ' '.join(toks[i:i + 3])
        way = next((w for w in OTHER_WAY if seg3 == w or seg3.startswith(w + ' ')), None)
        if way:                            # "like the next-step sound but going the other way"
            for k in ('concepts', 'parts'):
                b[k] = [L.OPPOSITE.get(c, c) for c in b[k]]
            b['dir'] = -b['dir']
            i += len(way.split())
            continue
        if t == 'like' and not (nxt in LIKE_JOB_NEXT or nxt is None or nxt in PUNCT
                                or (prev in ARTICLES and nxt not in ARTICLES)):
            st['simile'] = True            # "like a whisper of a click": a comparison, not the job "like"
            i += 1
            continue
        if st['head'] and (t in CONTEXT_WORDS or t == 'to' or (t in PLACE_WORDS and nxt in ARTICLES)):
            st['where'] = 'context'        # "an impact FOR a logo reveal": where it plays, not what it is
            i += 1
            continue
        if st['any'] and t == 'with' and st['where'] == 'concepts':
            st['where'] = 'parts'          # "a bell WITH an airy whoosh": a part of it
            i += 1
            continue
        if t in L.NEGATIONS:
            st['scale'] *= L.NEGATIONS[t]
            st['neg'] = True
            i += 1
            continue
        if t in L.INTENSIFIERS and nxt and (modifier(nxt) or nxt in L.INTENSIFIERS or nxt in L.NEGATIONS) \
                and not (t == 'little' and nxt in L.STOP):
            st['scale'] *= L.INTENSIFIERS[t]
            i += 1
            continue
        if t in ('on', 'off'):
            particles.append((t, nxt))
        if t in LOOP_WORDS or t in ONCE_WORDS:
            b['loop'] = st['scale'] > 0 and t in LOOP_WORDS
            st.update(scale=1.0, neg=False)
            i += 1
            continue
        if FIGURE.fullmatch(t):            # "a two-note ding": a figure's shape, said in the words of `says`
            b['words'].extend(_figure_words(t))
            st.update(scale=1.0, neg=False)
            i += 1
            continue
        if i in phrasal:                   # "picking something up", "putting a window away"
            c, k = phrasal[i]
            add(c, c)
            skip.add(k)
            st.update(scale=1.0, neg=False)
            i += 1
            continue
        j, hit = _match(toks, i)
        if hit:
            key = _key(toks[i:j])
            if 'm' in hit:
                axes(hit['m'], st['scale'])
            if 'p' in hit and not st['neg']:
                b['material'] = b['material'] or hit['p']
            if 'o' in hit:
                domains[hit['o']] = domains.get(hit['o'], 0) + 2
            if 'd' in hit:
                b['dir'] += hit['d'] * (1 if st['scale'] > 0 else -1)
            if 'v' in hit:
                v = hit['v']
                b['voice'] = b['voice'] or v
                player = prev in VOICE_PARAM_PREV or (prev in ARTICLES and i > 1 and toks[i - 2] in VOICE_PARAM_PREV)
                follow = None if hit.get('c') in VOICE_OK and j - i > 1 else _next_concept(toks, j)
                if j < n_tok and toks[j] in L.SOUND_NOUNS:
                    follow = None              # "a kalimba ping": a note of the instrument
                if player or (follow is not None and follow not in VOICE_OK):
                    c = hit.pop('c', None)     # "a bass boom": the instrument word only colours the boom
                    if c and not player:
                        add(c, key, 'parts')
                else:
                    add('voice:' + v, key)
            if 'c' in hit and not ('v' in hit and hit['c'] not in VOICE_OK):
                c, where = hit['c'], None
                if j < n_tok and toks[j] in L.ENVELOPE_WORDS:
                    where = 'parts'            # "a short shimmering decay": how it ends, not what it is
                elif c in L.VERB_JOBS and _next_concept(toks, j):
                    where = 'parts'            # "grabs a speed boost": the thing grabbed is the sound
                elif j - i == 1 and t in L.COLOUR_WORDS:
                    where = 'parts'            # "a windy zip": the weather colours the sound
                add(c, key, where)
                nx = toks[j] if j < n_tok else None
                if (key in _status_keys() and not st['simile'] and st['where'] == 'concepts'
                        and (nx is None or nx in PUNCT or nx in L.STOP or nx in CONTEXT_WORDS
                             or _key([nx]) in _status_keys())):
                    outcomes.append(c)         # "upload finished.": an outcome said of its subject
            if st['where'] == 'context':
                ctx_words.update(toks[i:j])
            if j - i == 1 and _domain(t) and st['scale'] > 0:
                dm = _domain(t)
                domains[dm] = domains.get(dm, 0) + (2 if t in L.DOMAIN_STRONG else 1)
            if j - i == 1 and t in L.MATERIALS and not st['neg']:
                b['material'] = b['material'] or L.MATERIALS[t]
            if 'c' in hit or 'v' in hit:
                st.update(scale=1.0, neg=False)
            i = j
            continue
        hit = False
        if _domain(t) and st['scale'] > 0:
            dm = _domain(t)
            domains[dm] = domains.get(dm, 0) + (2 if t in L.DOMAIN_STRONG else 1)
            hit = True
        if t in L.MATERIALS and not st['neg']:
            b['material'] = b['material'] or L.MATERIALS[t]
            hit = True
        m = modifier(t)
        if not hit and m:
            axes(L.MODIFIERS[m], st['scale'])
            if st['scale'] > 0:
                b['hints'].extend(h for h in _feel(m) if h not in b['hints'])
            hit = True
        if not hit and t in L.STOP and t not in SCOPE_END:
            if t in CONTEXT_WORDS or t in PLACE_WORDS:
                st.update(neg=False, skipped=0)   # a negation does not reach past "in a ...", "for ..."
            i += 1
            continue                       # articles and fillers keep a pending negation or intensifier
        if not hit and not t.isdigit() and t not in L.STOP:
            if not st['neg']:
                b['words'].append(stem(t))
                if st['where'] == 'context':
                    ctx_words.add(stem(t))
            b['unknown'].append(t)
            if st['neg'] and st['skipped'] < 2:
                st['skipped'] += 1             # "won't make people jump": the negation reaches the verb
                i += 1
                continue
        st.update(scale=1.0, neg=False, skipped=0)
        i += 1
    named = set(b['concepts']) | set(b['parts']) | set(b['context'])
    if named & {'toggle', 'light_switch'} and not named & {'switch_on', 'switch_off'}:
        for p, nx in particles:            # "toggles dark mode on", "flip it off"
            if p == 'off' or nx is None or nx in PUNCT or nx in TOGGLE_ON_NEXT:
                b['concepts'].append('switch_off' if p == 'off' else 'switch_on')
                break
    if domains:
        b['domain'] = max(domains, key=lambda k: domains[k])
    for k in ('concepts', 'parts', 'context'):         # "fails a stage" in a game is a loss, not an error
        b[k] = [x for x in (_sense().get((b['domain'], src.get(c)), c) for c in b[k]) if x]
    _compose(b)
    if 'ambience' in b['concepts'] and 'drone' not in b['concepts'] and set(b['hints']) & L.DARK_MOODS:
        b['concepts'].insert(0, 'drone')               # "an ominous low bed" is a drone
    status = [c for c in outcomes if c in b['concepts'][1:]]
    if status:                                         # "upload finished": the outcome is the job
        b['concepts'] = status[:1] + [c for c in b['concepts'] if c != status[0]]
    b['axes'] = {a: round(max(-1.0, min(1.0, v)), 2) for a, v in b['axes'].items() if abs(v) >= 0.05}
    b['dir'] = (b['dir'] > 0) - (b['dir'] < 0)
    b['context_words'] = sorted(ctx_words)
    return b


def _terms(row):
    """a design's words by field ({'uses', 'name', 'says', 'tags'}: sets), its own jobs ('family': its family's
    concept; 'first': its first use), what it feels like ('feel': hint words) and which way it moves ('dir')"""
    t = {f: set() for f in FIELD_WEIGHT}
    feel, d = set(), 0
    fam, _, nm = row['id'].partition('/')
    for field, texts in (('uses', row['uses']), ('name', (fam, nm)), ('says', row['says']),
                         ('tags', list(row['tags']) + [row['kind'], row.get('material') or ''])):
        for s in texts:
            c, rest, f, dd = _parse_plain(str(s).replace('_', ' '))
            t[field].update(c)
            t[field].update(rest)
            feel.update(f)
            if field in ('says', 'name'):
                d += dd
            if field == 'says':
                t[field].update(_figure_words(str(s)))
    voice = row.get('voice')
    if fam == 'note' and voice:
        t['uses'].add('voice:' + voice)            # a note design is that instrument
    for part in nm.split('_'):
        v = L.VOICE_FAMILIES.get(part)
        if v and fam in ('mallet', 'pluck'):
            t['uses'].add('voice:' + v)
    t['family'] = set(_parse_plain(fam.replace('_', ' '))[0])
    for c in list(t['uses'] | t['family']):        # a swish is also a whoosh, a sub drop also an impact
        if c in L.BROADER and L.BROADER[c] not in t['uses']:
            t['tags'].add(L.BROADER[c])
    t['exact'] = fam if fam in L.CONCEPTS else None             # 'power_up/chip' is a power-up by name
    t['mats'] = {L.MATERIALS[w] for s in list(row['says']) + [nm] + list(row['tags'])
                 for w in tokens(str(s).replace('_', ' ')) if w in L.MATERIALS}
    if row.get('material'):
        t['mats'].add(row['material'])
    t['first'] = set(_parse_plain(row['uses'][0])[0][:1]) if row['uses'] else set()
    d += sum(L.CONCEPT_DIR.get(c, 0) for c in t['family'] or t['first'])
    t['feel'] = feel
    t['dir'] = (d > 0) - (d < 0)
    return t


def index(rows):
    """the search index of a catalogue: (rows, terms, {word: [(row index, field score)]}, {word: weight}); a word's
    weight is 1 + log(N / designs that have it): rarer = heavier"""
    rows = list(rows)
    terms = [_terms(r) for r in rows]
    inv, df = {}, {}
    for k, t in enumerate(terms):
        for w in set().union(*(t[f] for f in FIELD_WEIGHT)):
            df[w] = df.get(w, 0) + 1
            fs = sorted((FIELD_WEIGHT[f] for f in FIELD_WEIGHT if w in t[f]), reverse=True)
            bonus = FAMILY_BONUS if w in t['family'] else FIRST_USE_BONUS if w in t['first'] else 0.0
            bonus += EXACT_BONUS if w == t['exact'] else 0.0
            inv.setdefault(w, []).append((k, fs[0] + 0.25 * sum(fs[1:]) + bonus))
        for w in t['feel']:
            inv.setdefault('feel:' + w, []).append((k, FIELD_WEIGHT['says']))
            df['feel:' + w] = df.get('feel:' + w, 0) + 1
    n = max(1, len(rows))
    # a job the request names is not less telling for being common in the catalogue: canonical words get a flatter
    # weight than plain words
    return rows, terms, inv, {w: 1.0 + (CONCEPT_IDF if w in L.CONCEPTS or w.startswith('voice:') else 1.0)
                              * math.log(n / k) for w, k in df.items()}


def _default_index():
    from .sounds import registry
    key = len(registry.load())
    if _cache.get('key') != key:
        _cache['key'], _cache['idx'] = key, index(registry.rows())
    return _cache['idx']


def _query(b):
    """{term: weight} of a brief"""
    q = {}
    for w in b.get('hints', ()):
        q['feel:' + w] = AXIS_WEIGHT if w.startswith('ax:') else HINT_WEIGHT
    ctx = set(b.get('context_words', ()))
    for w in b['words']:
        q[w] = max(q.get(w, 0.0), WORD_WEIGHT * (0.5 if w in ctx else 1.0))
    for key, wt in (('refs', REF_WEIGHT), ('context', CONTEXT_WEIGHT), ('parts', PART_WEIGHT)):
        for c in b.get(key, ()):
            q[c] = max(q.get(c, 0.0), wt)
    head = next((c for c in b['concepts'] if c not in L.GENERIC), None)
    for c in b['concepts']:
        q[c] = HEAD_WEIGHT if c == head else max(q.get(c, 0.0), CONCEPT_WEIGHT)
    for c, w in list(q.items()):
        up = L.BROADER.get(c)
        if up and up not in q and w >= CONTEXT_WEIGHT:
            q[up] = BROADER_WEIGHT * w
    if b.get('dir') and not (b['concepts'] or b.get('parts') or b.get('context')):
        q['riser' if b['dir'] > 0 else 'downlifter'] = CONTEXT_WEIGHT   # "something that keeps climbing"
    return q


def length_fit(b, row):
    """how well a design's usual length fits the length asked for (1 = fits): a soft factor, down to 0.5 for a
    design several times longer (or shorter) than asked, or much longer than a ceiling ("under 300 ms"). A usual
    length is approximate, so twice or half the length asked for is free; a ceiling is firmer (half an octave over
    it is free). Beds, pads, textures and loops are exempt: they are rendered at any length"""
    d = row.get('dur_s')
    if not d or row.get('loop') or row.get('kind') in STRETCH_KINDS:
        return 1.0
    f = 1.0
    if b.get('dur'):
        r = abs(math.log2(max(b['dur'], 0.01) / d))
        f *= max(0.5, 1.0 - LENGTH_SLOPE * max(0.0, r - 1.0))
    if b.get('dur_max') and d > b['dur_max']:
        f *= max(0.5, 1.0 - 1.6 * LENGTH_SLOPE * max(0.0, math.log2(d / b['dur_max']) - 0.5))
    return f


def rank(b, rows=None, top=8, idx=None):
    """the designs that answer a brief, best first: [{'id', 'score', 'why'}]. rows: catalogue rows (default: every
    design; the index of the whole catalogue is built once and kept); a design must share at least one word with
    the request to be listed; when none does, the designs closest in feel are listed"""
    if idx is None:
        idx = index(rows) if rows is not None else _default_index()
    rows, terms, inv, idf = idx
    q = _query(b)
    score, why = {}, {}
    for w, qw in q.items():
        for k, fw in inv.get(w, ()):
            score[k] = score.get(k, 0.0) + qw * idf.get(w, 1.0) * fw
            if not w.startswith('feel:'):
                why.setdefault(k, []).append(w.replace('voice:', ''))
    mat, dom, loop = b.get('material'), b.get('domain'), b.get('loop')
    pool = ([k for k in score if k in why] or [k for k in score if dom in rows[k].get('domains', ())]
            or list(score))                            # nothing named: the closest in feel, in the request's domain

    def chip(k):
        return 'retro' in rows[k]['tags'] or rows[k].get('material') == 'retro'
    if mat == 'retro' and not any(chip(k) for k in pool):
        # chip sound asked for and no chip design shares a word: every chip design, by feel and direction
        pool = [k for k in range(len(rows)) if chip(k)]
        for k in pool:
            score[k] = 1.0 + score.get(k, 0.0)
            why.setdefault(k, ['retro'])
    is_retro = [chip(k) for k in pool]
    only_retro = mat == 'retro' and any(is_retro)
    out = []
    for k, retro in zip(pool, is_retro):
        row, t, s = rows[k], terms[k], score[k]
        for c in b.get('not', ()):
            if c in t['uses'] or c in t['name']:
                s *= 0.5                                       # "not an error", "smooth, not clicky"
        s *= row.get('prior', 1.0)
        if dom and row.get('domains'):
            s *= 1.15 if dom in row['domains'] else 0.85
        if mat == 'retro':
            s *= 1.0 if retro else 0.05 if only_retro else 0.25   # chip sound asked for: only chip
        elif retro:
            s *= 0.35                                          # chip sound only when asked for
        elif mat and mat in t['mats']:
            s *= 1.3
        if loop and not row.get('loop'):
            s *= 0.6
        s *= length_fit(b, row)
        if b.get('dir'):
            s *= 0.92 if not t['dir'] else 1.35 if b['dir'] == t['dir'] else 0.7
        out.append({'id': row['id'], 'score': round(s, 3),
                    'why': why.get(k, [w[5:] for w in q if w.startswith('feel:') and w[5:] in t['feel']])})
    out.sort(key=lambda r: (-r['score'], r['id']))
    return out[:top]
