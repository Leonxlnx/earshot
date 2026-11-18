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
- the app, site, store or video it plays in ("in my running app, a success sound", "piano lessons website hover
  sound") names no sound once the request names one outside it; with nothing else named its words count as ever;
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
`why` names the words that decided. `rank(b, semantic=atlas.zmap(text))` blends in how much each design sounds like
the request (earshot.sounds.atlas, CLAP; `blend`): a named job keeps its lexical winner, a request that only
describes a sound is decided by the sound, and the hard filters (chip sound, loop, a job ruled out) still hold.
"""
import json
import math
import os
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
# a word is letters (also accented ones: "für" is one word, not "f" and "r"), digits and '#'
WORD = re.compile(r"[a-z0-9#ßà-öø-ÿ]+(?:[-'][a-z0-9#ßà-öø-ÿ]+)*")
TOKEN = re.compile(r"[a-z0-9#ßà-öø-ÿ]+(?:[-'][a-z0-9#ßà-öø-ÿ]+)*|[,.;:!?()]")
PUNCT = frozenset(',.;:!?()')
LOOP_WORDS = ('loop', 'looping', 'loopable', 'seamless', 'looped', 'endless')
ONCE_WORDS = ('one-shot', 'oneshot')
CONTEXT_WORDS = ('for', 'when', 'while', 'as', 'after', 'before', 'during', 'whenever', 'once', 'where', 'until',
                 'whilst', 'if', 'into')   # "a whoosh into the logo": where it lands
PLACE_WORDS = ('in', 'under', 'over', 'across', 'behind', 'beneath', 'inside')
# the place or product a sound plays in ("my running app", "a coffee shop checkout", "a video about dogs"): the
# words of that phrase say where it plays, not what it is, once the request names its sound elsewhere (`_settings`).
# A game is not one of them: its world shapes the sound ("an impact in a boxing game" is a punch)
SETTING_WORDS = frozenset(('app', 'apps', 'webapp', 'site', 'website', 'webpage', 'homepage', 'page', 'dashboard',
                           'panel', 'screen', 'store', 'shop', 'checkout', 'platform', 'tool', 'brand', 'channel',
                           'podcast', 'video', 'reel', 'trailer', 'presentation', 'menu', 'widget', 'tab', 'tracker',
                           'inbox'))
# places that also name a part of a thing ("a shop door", "screen time", "a panel slide"): they end the phrase only
# where nothing but a sound word or the end of a clause follows
SETTING_LOOSE = frozenset(('screen', 'store', 'shop', 'panel', 'menu'))
SETTING_TOPIC = ('video', 'reel', 'trailer', 'podcast', 'channel', 'presentation')   # "a video about dogs"
# the phrase begins the request or a clause, or follows one of these ("for my app", "the menu of a website")
SETTING_OPEN = ('for', 'in', 'on', 'inside', 'within', 'at', 'of', 'from')
SETTING_BREAK = ('with', 'to', 'about', 'by', 'than', 'like', 'but')   # these end a phrase that is no place
# words whose ending is not an ending ('evening' is not 'even', 'string' is not 'str')
STEM_KEEP = frozenset(('evening', 'morning', 'ceiling', 'string', 'strings', 'thing', 'things', 'ring', 'rings',
                       'spring', 'springs', 'swing', 'sling', 'wing', 'wings', 'king', 'bring', 'during', 'nothing',
                       'something', 'anything', 'everything', 'bus', 'gas', 'news', 'lens', 'chaos', 'canvas',
                       'tape', 'tapes', 'setting', 'settings', 'paste', 'pasted', 'cute'))
# (a tape is no tap, a setting no set, paste no past, cute no cut)
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
WORD_WEIGHT, HINT_WEIGHT, AXIS_WEIGHT, SAID_WEIGHT = 0.6, 0.35, 0.12, 0.005
FIELD_WEIGHT = {'uses': 3.0, 'name': 2.2, 'says': 1.6, 'tags': 1.2}
# the design's own family named in the request (by its very name: + EXACT_BONUS); its first job
FAMILY_BONUS, EXACT_BONUS, FIRST_USE_BONUS = 1.2, 0.3, 0.6
CONCEPT_IDF = 0.6                  # how much rarity counts for canonical words (1 = as for plain words)
STRETCH_KINDS = ('bed', 'pad', 'texture')   # rendered at any length asked for
LENGTH_SLOPE = 0.15                # how much a length mismatch costs per octave beyond the free half octave
# the blend of the lexical score with the semantic z (`blend`; chosen on eval/requests.json + requests_b.json)
LEX_FLOOR, LEX_POWER, SEM_WEIGHT, Z_FLOOR, Z_SAYS = 1.0, 0.25, 0.4, -1.0, 2.5
MEMBER_SHARE = 0.8                 # how close a member named in the request must score to the family's best
# the same for a named design of another family found by the very same words, of the same kind ("tape stop":
# downlifter/tape at 0.77 of fall/air, the downlifter concept's default)
KIN_SHARE = 0.7
UNNAMED_LEX = 0.25                # the most a lexical score counts in the blend when no job or sound word matched
# the tonal interface figures a plain request no longer gets (lexicon.BY_NAME_ONLY): their score is multiplied by
# this unless the request asks for a tone ("a tonal tap", "a tap with a note in it")
BY_NAME = 0.5
# a learned design whose interface job (its `roles`) is the request's head: a plain "tap", "hover", "toggle on",
# "next", "type" gets the recorded tap, tick, switch or key chosen for that job (earshot.sounds.sets 'ui')
ROLE = 2.0
# the sound of a real thing chosen for an interface job (lexicon.INTERFACE_JOBS), for a request of that job without
# musical words: the pitched figures step back by BY_NAME, so this only needs to win among its own alternatives and
# things that merely share a word; outside an interface job these designs step back by BY_NAME ("a knock cue when a
# guest arrives" is a door knocked on, "a capture button" a camera's shutter)
REAL = 1.4
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
            if end in ('ing', 'ed') and len(w) >= 6 and w.endswith('ell'):
                w = w[:-1]                # British doubling: 'levelling', 'cancelled' (not 'spelling')
            break
    if len(w) > 3 and w.endswith('e') and not w.endswith(('ee', 'ie')):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in 'slz' and w[-1].isalpha():
        w = w[:-1]
    return w


# a request that says one of these asks for a tone: the tonal interface figures answer it (BY_NAME)
TONE_STEMS = frozenset(stem(w) for w in ('tone', 'tonal', 'note', 'musical', 'melodic', 'pitched', 'chime', 'marimba'))
# a request that says one of these (or names an instrument, a voice, or music as its domain) asks for music: only then
# does an interface job get a pitched design (lexicon.INTERFACE_JOBS); otherwise it gets the sound of a real thing
MUSIC_STEMS = TONE_STEMS | frozenset(stem(w) for w in (
    'notes', 'melody', 'jingle', 'arpeggio', 'arpeggiated', 'tune', 'tuneful', 'instrument', 'music', 'musically',
    'harmony', 'chord', 'bell', 'bells', 'ding', 'chiming', 'xylophone', 'glockenspiel', 'vibraphone', 'harp',
    'piano', 'pizzicato', 'kalimba', 'celesta', 'synth', 'synthesizer', 'ringtone'))
# a motion request that says one of these asks for notes (a sonic logo of a few notes, a motif, a pad, a drone): only
# then does a design with a note in it answer a motion job (lexicon.MOTION_JOBS); otherwise it steps back by FILM_TONE
FILM_MUSIC = MUSIC_STEMS | frozenset(stem(w) for w in (
    'motif', 'sonic', 'mnemonic', 'pad', 'drone', 'catchy', 'memorable', 'signature', 'musical', 'marimba'))
FILM_TONE = 0.4


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


# "three knocks", "a few steps", "two quick clinks": a number of events (not of takes, notes, seconds: COUNT, FIGURE,
# DUR take those)
EVENTS = re.compile(r'\b(\d+|' + '|'.join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r')\s+(?:[a-z]+\s+){0,2}?'
                    r'([a-z]+)\b', re.I)
NOT_EVENTS = frozenset(stem(w) for w in (
    'variation', 'variant', 'version', 'take', 'option', 'alternative', 'note', 'tone', 'beat', 'pulse', 'second', 's',
    'sec', 'ms', 'msec', 'millisecond', 'minute', 'min', 'hz', 'khz', 'time', 'octave', 'semitone', 'bar', 'of',
    'different', 'unique', 'shot', 'player', 'people', 'bit', 'sound', 'file', 'level', 'more', 'another', 'after'))
TIMES = re.compile(r'\b(?:(twice|thrice)|(\d+|' + '|'.join(sorted(NUMBER_WORDS, key=len, reverse=True))
                   + r')\s+times)\b')
COUNTDOWN = re.compile(r'\b(?:10|\d)(?:[\s,.\-]+(?:10|\d))+\b')    # "3 2 1", "5, 4, 3, 2, 1" (checked: down to 1)
PLURAL_NOT = frozenset(('glass', 'bass', 'grass', 'brass', 'chaos', 'canvas', 'gas', 'news', 'lens', 'bus', 'its',
                        'this', 'is', 'was', 'has', 'does', 'goes', 'plus', 'less', 'across', 'always', 'perhaps',
                        'sounds', 'sfx', 'fx', 'effects', 'vibes', 'its', 'us', 'yes', 'as', 'thus', 'kiss', 'boss',
                        'mess', 'press', 'dress', 'loss', 'moss', 'toss'))


def _action(b, text):
    """the action words of a request into the brief (`read`: events, perform, plural)"""
    low = ' ' + (text or '').lower() + ' '
    b['events'], b['perform'], b['plural'], b['counted'], b['row'] = None, False, [], [], None
    b['act'] = next((w for w in WORD.findall(low) if w in L.ACTION_WORDS), None)   # "clapping": `acted`
    rows = {stem(w): p for w, p in L.ROW_WORDS.items()}
    for m in EVENTS.finditer(low):
        v, noun = m.group(1).lower(), stem(m.group(2))
        if noun in NOT_EVENTS or re.match(r'\s*-', low[m.end(1):]) or low[max(0, m.start(1) - 1)] in '.,:/':
            continue
        b['events'] = int(v) if v.isdigit() else NUMBER_WORDS[v]
        b['counted'] = [stem(w) for w in WORD.findall(low[m.end(1):m.end()])]   # "five coin pickups": coin, pickup
        # the counted thing within three words ("three mouse clicks"): a row when a person performs it in a row
        b['row'] = next((rows[stem(w)] for w in WORD.findall(low[m.end(1):])[:3] if stem(w) in rows), None)
        if b['row'] and 'countdown' in low:                 # "a countdown of three beeps": one a second
            b['row'] = 'count'
        break
    for c in (COUNTDOWN.finditer(low) if 'countdown' in low and b['events'] is None else ()):
        ds = [int(x) for x in re.findall(r'\d+', c.group(0))]
        k = next((i for i in range(1, len(ds)) if ds[i - 1] - ds[i] != 1), len(ds))
        ds = ds[:k]                                         # "3 2 1, 1 second": the countdown is 3 2 1
        if len(ds) > 1 and ds[-1] in (0, 1):
            b['events'], b['row'] = len(ds), 'count'   # "a countdown 3 2 1": three events a second apart
            break
    m = TIMES.search(low)
    if m and b['events'] is None:                           # "a dog barking twice", "knock three times"
        v = m.group(1) or m.group(2)
        b['events'] = {'twice': 2, 'thrice': 3}.get(v) or (int(v) if v.isdigit() else NUMBER_WORDS[v])
        b['row'] = next((rows[stem(w)] for w in WORD.findall(low[:m.start()]) if stem(w) in rows), None)
        b['counted'] = [stem(w) for w in WORD.findall(low[:m.start()])]        # what is done that many times
    if any(re.search(r'(?<![\w-])' + re.escape(w) + r'(?![\w-])', low) for w in L.SINGLE_WORDS):
        if not (b['events'] and b['events'] > 1):
            b['events'] = 1
    words = WORD.findall(low)
    b['perform'] = (b['events'] or 0) > 1 or any(w in L.PERFORM_WORDS for w in words) or any(
        f' {p} ' in low for p in L.PERFORM_PHRASES)
    if b['events'] == 1:
        b['perform'] = False
    b['plural'] = sorted({stem(w) for w in words if w.endswith('s') and not w.endswith(("'s", 'ss', 'us', 'is'))
                          and w not in PLURAL_NOT and len(w) > 3 and stem(w) != w})


def performance(b, sid, pattern=None):
    """(perform, events) for the design sid, which can be performed (earshot.sounds.learned.performs; pattern: its
    index `pattern`): (False, 1) for a single event ("a single knock", "a footstep", "a key press"); (True, n) for n
    events ("three knocks"); (True, None) for the action at its natural number or filling the length asked for
    ("footsteps on gravel", "knocking on a door", "5 seconds of typing"): an action word, a plural of the thing
    itself (its family word, or one of the pattern's `nouns`: "ice cubes"), or a singular that still means the
    action (the pattern's `singular` words: "a knock on the door")"""
    if b.get('events') == 1:
        return False, 1
    if b.get('events'):
        return True, int(b['events'])
    if b.get('perform') or acted(b):
        return True, None
    fam = sid.partition('/')[0]                    # the thing, not what it stands on ("a footstep on tiles": one)
    thing = {stem(fam), stem(L.FAMILY_SENSE.get(fam, fam))} | {stem(w) for w in (pattern or {}).get('nouns', ())}
    if set(b.get('plural', ())) & thing:
        return True, None
    said = set(WORD.findall((b.get('text') or '').lower()))
    if pattern and said & set(pattern.get('singular', ())):
        return True, None
    return False, None


def acted(b):
    """the pattern of earshot.synth.perform (lexicon.ROW_WORDS) of the action a request without a count names as its
    sound with the action form of a row word ("clapping", "knocking, 4 seconds": lexicon.ACTION_WORDS), else None.
    The word must be how the request names its head concept, not a comparison or the event the sound plays for
    ("like tapping a little bell", "a cue for tapping Continue", "the ready-set-go beeping" is a countdown)"""
    w = b.get('act')
    if not w or b.get('events') or not b.get('concepts'):
        return None
    head = b['concepts'][0]
    if head in b.get('framed', ()) or w not in (b.get('written', {}).get(head) or '').split():
        return None
    return L.ROW_WORDS[L.ACTION_WORDS[w]]


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


def _job_feel():
    """{stemmed phrase: adjective} of lexicon.JOB_FEEL"""
    if 'job_feel' not in _cache:
        _cache['job_feel'] = {_key(w.split()): m for w, m in L.JOB_FEEL.items()}
    return _cache['job_feel']


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
            said = [b['written'][x] for x in (a, c2) if x in b.get('written', {})]
            if said:
                b['written'].setdefault(r, ' '.join(said))


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


def _settings(toks):
    """the token indices of the phrases that say where a sound plays: a place or product word (SETTING_WORDS) with up
    to three words before it, from the start of a clause or a "for", "in", "on", "of" ... ("in my running club app",
    "a coffee shop checkout page", "the Thunder app's inbox"), and what a video or podcast is about ("a video about
    dogs"). A word that begins a job of its own ("menu select", "page turn") is no place; a phrase that opens with
    an event, a verb of an interface job or a sound ("when a picture slides off the screen", "swipe to dismiss") is
    none either"""
    span = set()
    for k, t in enumerate(toks):
        if t not in SETTING_WORDS or _match(toks, k)[0] > k + 1:
            continue
        nx = toks[k + 1] if k + 1 < len(toks) else None
        if t in SETTING_LOOSE and not (nx is None or nx in PUNCT or nx in L.STOP or nx in SETTING_WORDS
                                       or nx in L.SOUND_NOUNS):
            continue
        i, n = k - 1, 0                  # back to where the phrase opens; at most three words of its own
        while i >= 0 and toks[i] not in PUNCT and toks[i] not in SETTING_OPEN:
            p = toks[i]
            if (p in CONTEXT_WORDS or p in PLACE_WORDS or p in SETTING_BREAK or p in L.SOUND_NOUNS or p in SCOPE_END
                    or _match(toks, i)[1].get('c') in L.INTERFACE_JOBS):
                break
            if p not in ARTICLES and p not in SETTING_WORDS and p not in L.STOP:
                if n == 3:
                    break
                n += 1
            i -= 1
        else:
            j = k + 1
            if t in SETTING_TOPIC and j < len(toks) and toks[j] == 'about':
                j += 1
                while j < len(toks) and toks[j] not in PUNCT and toks[j] not in CONTEXT_WORDS:
                    j += 1
            span.update(range(i + 1, j))
    return span


def read(text):
    """a request as a brief (module doc): {'text', 'concepts', 'parts', 'context', 'refs', 'not', 'words', 'hints',
    'axes', 'dir', 'material', 'voice', 'domain', 'dur', 'dur_max', 'pitch', 'key', 'scale', 'n', 'notes', 'loop',
    'unknown', 'context_words', 'written', 'framed', 'phrases'}; absent things are None / empty. `concepts` are the
    sound or job asked for (the head first), `parts` what the request says it has ("with a soft shimmer"), `context`
    the canonical words after "for" / "when" (where it plays; `context_words` the plain words there), `refs` what it
    is compared with ("duller than an on-switch"), `not` the jobs the request rules out ("not an error"), `hints`
    the adjectives (they break ties between members of a family: "crisp" finds whoosh_fast/crisp), `dir` +1 for a
    rising gesture and -1 for a falling one, `dur_max` a ceiling ("under 80 ms"), `notes` the number of notes of a
    figure ("a two-note chime"), `written` the words as the request wrote them ({canonical word or stemmed word:
    'water drop'}; `shown`), `framed` the jobs named only in a comparison or in the event before the sound is named
    ("sound when the player grabs a boost"), `phrases` the runs of content words a design's multi-word job is
    matched against (`_phrases`); `events` the number of events of an action ("three knocks": 3, "a single step":
    1), `row` the pattern of earshot.synth.perform when the counted thing is one a person performs in a row
    (lexicon.ROW_WORDS: "four knocks": knock, "a countdown 3 2 1": count), `perform` True when the request asks
    for the action a person performs ("walking", "knocking", "keys
    jingling"), `act` the action form of a row word it says ("clapping": `acted`) and `plural` the stems of the
    plural nouns it says ("footsteps": footstep): `performance` decides
    with them for a design that can be performed"""
    b = {'text': text, 'concepts': [], 'parts': [], 'context': [], 'refs': [], 'not': [], 'words': [], 'hints': [],
         'axes': {},
         'dir': 0, 'material': None, 'voice': None, 'domain': None, 'dur': None, 'dur_max': None, 'pitch': None,
         'key': None, 'scale': None, 'n': None, 'notes': None, 'loop': None, 'unknown': [], 'written': {},
         'framed': []}
    _action(b, text)
    toks = tokens(_numbers(b, text))
    n_tok = len(toks)
    st = {'scale': 1.0, 'neg': False, 'where': 'concepts', 'than': False, 'opposite': False, 'head': False,
          'any': False, 'object': False, 'simile': False, 'skipped': 0, 'event': False, 'i': 0}
    domains, particles, src, ctx_words, outcomes = {}, [], {}, set(), []
    if _old_device(toks):
        b['material'] = 'retro'            # "old Nintendo games", "an ancient home computer"
    phrasal, skip = _phrasal(toks), set()
    # the words of where it plays ("for my running app") and what they named, against what the rest named
    settings, inside, outside, voiced = _settings(toks), set(), set(), []

    def add(c, tok, where=None, said=None):
        if st['object'] and st['where'] == 'concepts':
            where, st['object'] = 'parts', False    # "parrying an attack": the attack is what is parried
        if st['simile'] and st['where'] == 'concepts':
            where = 'parts'                         # "a thump, like a door slamming": a comparison describes it
        where = where if where and st['where'] == 'concepts' else st['where']
        if st['opposite']:
            c, st['opposite'], said = L.OPPOSITE.get(c, c), False, None
        if st['neg']:
            if c not in b['not']:
                b['not'].append(c)
            st['neg'], st['scale'] = False, 1.0
            return
        if st['i'] in settings:
            inside.add(c)
        elif where != 'context':
            outside.add(c)
        if all(c not in b[k] for k in ('concepts', 'parts', 'context')):
            b[where].append(c)
            src[c] = tok
            if said:
                b['written'].setdefault(c, said)
            if st['simile'] or st['event']:
                b['framed'].append(c)               # named in a comparison or the event: not the sound itself
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
        st['i'] = i
        if i in skip:                       # the particle of a separable verb already read
            i += 1
            continue
        if t in PUNCT:
            st.update(than=False, neg=False, scale=1.0, simile=False, event=False)
            if st['where'] == 'parts':
                st['where'] = 'concepts'
            i += 1
            continue
        timing = next((p for p in L.TIMING_PHRASES if toks[i:i + len(p.split())] == p.split()), None)
        if timing:                         # "a hit on the beat": when it lands, not music asked for
            i += len(timing.split())
            continue
        if (t == 'close' and (nxt is None or nxt in PUNCT) and st['head'] and b['concepts']
                and b['concepts'][0] not in L.OPPOSITE and (prev in PUNCT or prev in L.INTENSIFIERS)):
            axes(L.PHRASE_MODIFIERS['up close'], st['scale'])   # "footsteps on gravel, close": near, not a closing
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
        if t in ('each', 'every') and nxt == 'time':
            if st['head']:                 # "a tick each time a step number changes": whenever, where it plays
                st['where'] = 'context'
            else:
                st['event'] = True
            i += 2
            continue
        if st['head'] and (t in CONTEXT_WORDS or t == 'to' or (t in PLACE_WORDS and nxt in ARTICLES)):
            st['where'] = 'context'        # "an impact FOR a logo reveal": where it plays, not what it is
            i += 1
            continue
        if t in CONTEXT_WORDS:
            st['event'] = True             # "sound when the player grabs ...": the event, said before the sound
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
            add(c, c, said=f'{t} {toks[k]}')
            skip.add(k)
            st.update(scale=1.0, neg=False)
            i += 1
            continue
        j, hit = _match(toks, i)
        if hit:
            key, said = _key(toks[i:j]), ' '.join(toks[i:j])
            if ('c' in hit and j - i == 1 and t in L.MANNER_WORDS
                    and _next_concept(toks, j) not in (None, hit['c'])):
                del hit['c']               # "a careful step": how the step is taken, not a warning
                axes(L.MANNER_WORDS[t], st['scale'])
            feel = _job_feel().get(key) if 'c' in hit and st['scale'] > 0 else None
            if feel:                       # "the card settles": it lands, and softly
                axes(L.MODIFIERS[feel], st['scale'])
                b['hints'].extend(h for h in _feel(feel) if h not in b['hints'])
            if 'm' in hit:
                axes(hit['m'], st['scale'])
            elif j - i > 1:                # "a warm pad": the adjective inside a name still moves its axis
                for w in toks[i:j]:
                    if modifier(w):
                        axes(L.MODIFIERS[modifier(w)], st['scale'])
            if 'p' in hit and not st['neg']:
                b['material'] = b['material'] or hit['p']
            if 'o' in hit:
                domains[hit['o']] = domains.get(hit['o'], 0) + 2
            if 'd' in hit and st['where'] != 'context':     # "a riser before the drop" still rises
                b['dir'] += hit['d'] * (1 if st['scale'] > 0 else -1)
            if 'v' in hit:
                v = hit['v']
                if not b['voice']:
                    b['voice'] = v
                    voiced.append(i in settings)     # "a piano lessons website": where, not the instrument
                k = i - 1                  # "on a soft piano": articles and adjectives stand between
                while k >= 0 and (toks[k] in ARTICLES or modifier(toks[k])):
                    k -= 1
                player = k >= 0 and toks[k] in VOICE_PARAM_PREV
                follow = None if hit.get('c') in VOICE_OK and j - i > 1 else _next_concept(toks, j)
                if j < n_tok and toks[j] in L.SOUND_NOUNS:
                    follow = None              # "a kalimba ping": a note of the instrument
                if player or (follow is not None and follow not in VOICE_OK):
                    c = hit.pop('c', None)     # "a bass boom": the instrument word only colours the boom
                    if c and not player:
                        add(c, key, 'parts', said)
                else:
                    add('voice:' + v, key, said=said)
            if 'c' in hit and not ('v' in hit and hit['c'] not in VOICE_OK):
                c, where = hit['c'], None
                if j < n_tok and toks[j] in L.ENVELOPE_WORDS:
                    where = 'parts'            # "a short shimmering decay": how it ends, not what it is
                elif c in L.VERB_JOBS and _next_concept(toks, j) not in (None, c):
                    where = 'parts'            # "grabs a speed boost": the thing grabbed is the sound
                elif j - i == 1 and t in L.COLOUR_WORDS:
                    where = 'parts'            # "a windy zip": the weather colours the sound
                add(c, key, where, said)
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
                b['written'].setdefault(stem(t), t)
                if st['where'] == 'context':
                    ctx_words.add(stem(t))
            b['unknown'].append(t)
            if st['neg'] and st['skipped'] < 2:
                st['skipped'] += 1             # "won't make people jump": the negation reaches the verb
                i += 1
                continue
        st.update(scale=1.0, neg=False, skipped=0)
        i += 1
    if outside - L.GENERIC:                # "in my running app, a success sound": the running says where, not what
        gone = inside - outside
        for k in ('concepts', 'parts', 'context', 'framed'):
            b[k] = [c for c in b[k] if c not in gone]
        if voiced and voiced[0]:
            b['voice'] = None
        ctx_words.update(w for k in settings for w in (toks[k], stem(toks[k])))
    named = set(b['concepts']) | set(b['parts']) | set(b['context'])
    if named & {'toggle', 'light_switch'} and not named & {'switch_on', 'switch_off'}:
        for p, nx in particles:            # "toggles dark mode on", "flip it off"
            if p == 'off' or nx is None or nx in PUNCT or nx in TOGGLE_ON_NEXT:
                b['concepts'].append('switch_off' if p == 'off' else 'switch_on')
                break
    if domains:
        b['domain'] = max(domains, key=lambda k: domains[k])
    for c, tok in src.items():                        # the words of a job carry over to what it means here
        x = _sense().get((b['domain'], tok))
        if x and c in b['written']:
            b['written'].setdefault(x, b['written'][c])
    for k in ('concepts', 'parts', 'context'):         # "fails a stage" in a game is a loss, not an error
        b[k] = [x for x in (_sense().get((b['domain'], src.get(c)), c) for c in b[k]) if x]
    _compose(b)
    for c, tok in list(src.items()):              # "a toast at a wedding": glasses, not a notification
        if tok in L.WORD_WITH and set(toks) & L.WORD_WITH[tok][0]:
            to = L.WORD_WITH[tok][1]
            for k in ('concepts', 'parts', 'context'):
                b[k] = list(dict.fromkeys(to if x == c else x for x in b[k]))
            b['written'].setdefault(to, b['written'].get(c, tok))
        elif tok in L.WORD_WITHOUT and set(toks) & L.WORD_WITHOUT[tok]:
            for k in ('concepts', 'parts', 'context', 'framed'):    # "a crunchy bite into toast": bread, no job
                b[k] = [x for x in b[k] if x != c]
            b['words'].append(tok)
            b['unknown'].append(b['written'].pop(c, tok))
    group = {t for t in toks if t not in ctx_words}
    for one, many in L.CROWD_OF.items():       # "people clapping": an audience, not a row of claps
        if one in b['concepts'] and group & L.GROUP_WORDS and not group & L.UNISON_WORDS:
            b['concepts'] = [many if c == one else c for c in b['concepts']]
            if one in b['written']:
                b['written'].setdefault(many, b['written'][one])
    if 'ambience' in b['concepts'] and 'drone' not in b['concepts'] and set(b['hints']) & L.DARK_MOODS:
        b['concepts'].insert(0, 'drone')               # "an ominous low bed" is a drone
    status = [c for c in outcomes if c in b['concepts'][1:]]
    if status:                                         # "upload finished": the outcome is the job
        b['concepts'] = status[:1] + [c for c in b['concepts'] if c != status[0]]
    b['axes'] = {a: round(max(-1.0, min(1.0, v)), 2) for a, v in b['axes'].items() if abs(v) >= 0.05}
    b['dir'] = (b['dir'] > 0) - (b['dir'] < 0)
    # in a film or video "the drop" is the edit: everything cut for a moment, then a hit (lexicon.MOTION_DROP); a
    # falling sound only when the words say what falls (a sub drop, a bass drop, a downlifter). A drop said with no
    # domain and nothing else as its sound ("a drop", "the drop after the build") is that hit too
    # In an app a thing dropped with nothing else named is set down ("item dropped into a list": a soft landing)
    own = [c for c in b['concepts'] if c not in b['framed']]
    drop = set(tokens(text)) & (L.MOTION_DROP | ({'dropped', 'dropping'} if b['domain'] == 'ui' else set()))
    if drop and any(t in drop and i and toks[i - 1] in _dropped_things() for i, t in enumerate(toks)):
        drop = set()                   # "a heavy drop", "a bag drop": the thing a design drops, named
    if (drop and b['dir'] < 0 and not set(own) & {'sub_drop', 'downlifter', 'riser'}
            and (b['domain'] == 'motion' or (b['domain'] in (None, 'ui') and not own))):
        job = 'land' if b['domain'] == 'ui' else 'hit'
        if own:
            b['concepts'] = b['concepts'] + [job]
        else:                          # what else was named is when it plays ("the drop after the build")
            b['context'] += [c for c in b['concepts'] if c not in b['context']]
            b['concepts'] = [job]
        b['dir'] = 0
    b['context_words'] = sorted(ctx_words)
    b['phrases'] = _phrases(toks, b['written'])
    return b


def _dropped_things():
    """the family words of the designs of a thing dropped ('heavy_drop/floor', 'bag/drop', 'apple/drop'; not the
    film's own hit/drop)"""
    if 'dropped' not in _cache:
        ids = [r['id'] for r in _default_index()[0]]
        _cache['dropped'] = frozenset(w for i in ids if 'drop' in i.replace('/', '_').split('_')
                                      for w in i.split('/')[0].split('_') if w not in ('drop', 'hit'))
    return _cache['dropped']


def phrase_key(words_):
    """the key of a phrase of several words: the stems of its content words in alphabetical order, so that an
    inflected form or another word order finds it ('page turn', "turning a page", "pages turned" -> 'pag turn')"""
    return ' '.join(sorted(stem(w) for w in words_ if w not in L.STOP and w not in PUNCT))


def _phrases(toks, written):
    """the keys ('ph:' + phrase_key) of every run of 2 to 4 content words inside one clause of a request: what a
    design's multi-word job ('order ready', 'stutter into a cut') is matched against as a whole; `written` gets
    each run as the request wrote it"""
    out, run = [], []
    for t in toks + [',']:
        if t in PUNCT:
            for n in range(2, 5):
                for k in range(len(run) - n + 1):
                    key = 'ph:' + phrase_key(run[k:k + n])
                    out.append(key)
                    written.setdefault(key, ' '.join(run[k:k + n]))
            run = []
        elif t not in L.STOP:
            run.append(t)
    return list(dict.fromkeys(out))


def _terms(row):
    """a design's words by field ({'uses', 'name', 'says', 'tags'}: sets), its own jobs ('family': its family's
    concept; 'first': its first use), what it feels like ('feel': hint words) and which way it moves ('dir')"""
    t = {f: set() for f in FIELD_WEIGHT}
    feel, d, alias = set(), 0, set()
    fam, _, nm = row['id'].partition('/')
    sense = L.FAMILY_SENSE.get(fam)              # 'rise/air' is a riser, 'crash/dark' a cymbal
    for field, texts in (('uses', row['uses']), ('name', (sense or fam, nm)), ('says', row['says']),
                         ('tags', list(row['tags']) + [row['kind'], row.get('material') or ''])):
        for s in texts:
            c, rest, f, dd = _parse_plain(str(s).replace('_', ' '))
            t[field].update(c)
            content = [w for w in tokens(str(s).replace('_', ' ')) if w not in L.STOP and w not in PUNCT]
            if field == 'uses' and len(content) > 1 and rest:
                t[field].add('ph:' + phrase_key(content))   # a job of several words counts as a whole: 'rainy
                t['tags'].update(rest)                      # day'; "day" alone only as much as a tag
            else:
                t[field].update(rest)
            if field == 'uses' and len(c) == 1 and not rest and s != (row.get('voice') or '').replace('_', ' '):
                own = {stem(w) for w in c[0].split('_')}    # a job by another word for it ('beat', 'menu music' are
                #                                             music) is that job said so; a note's voice is a voice
                alias.update(f'fam:{stem(w)}:{c[0]}' for w in content if stem(w) not in own)
            if field == 'uses' and len(content) == 1:
                t[field].add('sw:' + stem(content[0]))     # the word itself, also when it names a job
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
    t['family'] = {sense} if sense else set(_parse_plain(fam.replace('_', ' '))[0])
    for c in list(t['uses'] | t['family']):        # a swish is also a whoosh, a sub drop also an impact
        if c in L.BROADER and L.BROADER[c] not in t['uses']:
            t['tags'].add(L.BROADER[c])
    t['exact'] = sense or (fam if fam in L.CONCEPTS else None)  # 'power_up/chip' is a power-up by name
    # a family named by another word for its job: 'tock/wood' is a knock that "a tock" asks for by name
    t['alias'] = alias | {f'fam:{stem(fam)}:{c}' for c in t['family'] if '_' not in fam
                          and stem(fam) not in {stem(w) for w in c.split('_')}}   # not 'coins/', the coin by its plural
    t['mats'] = {L.MATERIALS[w] for s in list(row['says']) + [nm] + list(row['tags'])
                 for w in tokens(str(s).replace('_', ' ')) if w in L.MATERIALS}
    if row.get('material'):
        t['mats'].add(row['material'])
    t['first'] = set(_parse_plain(row['uses'][0])[0][:1]) if row['uses'] else set()
    d += sum(L.CONCEPT_DIR.get(c, 0) for c in t['family'] or t['first'])
    t['feel'] = feel
    t['dir'] = (d > 0) - (d < 0)
    t['learned'] = row['id'] in _roles()     # a design of earshot.sounds.learned (recordings of one thing):
    #                                          `learned_ok`
    return t


def index(rows):
    """the search index of a catalogue: (rows, terms, {word: [(row index, field score)]}, {word: weight}); a word's
    weight is 1 + log(N / designs that have it): rarer = heavier. N and the counts leave out the designs of
    earshot.sounds.learned (a word only they have counts among them): a hundred recordings of footsteps and switches
    do not make "step" or "switch" a common word, and the other designs score as they did before any were learned"""
    rows = list(rows)
    terms = [_terms(r) for r in rows]
    inv, df, dfr = {}, {}, {}
    for k, t in enumerate(terms):
        count = dfr if t['learned'] else df
        for w in set().union(*(t[f] for f in FIELD_WEIGHT)):
            count[w] = count.get(w, 0) + 1
            fs = sorted((FIELD_WEIGHT[f] for f in FIELD_WEIGHT if w in t[f]), reverse=True)
            bonus = FAMILY_BONUS if w in t['family'] else FIRST_USE_BONUS if w in t['first'] else 0.0
            bonus += EXACT_BONUS if w == t['exact'] else 0.0
            inv.setdefault(w, []).append((k, fs[0] + 0.25 * sum(fs[1:]) + bonus))
        for w in t['feel']:
            inv.setdefault('feel:' + w, []).append((k, FIELD_WEIGHT['says']))
            count['feel:' + w] = count.get('feel:' + w, 0) + 1
        for w in t['alias']:
            inv.setdefault(w, []).append((k, FIRST_USE_BONUS))     # the job called by the request's word
            count[w] = count.get(w, 0) + 1
    for w, c in dfr.items():
        df.setdefault(w, c)
    n = max(1, sum(not t['learned'] for t in terms))
    # a job the request names is not less telling for being common in the catalogue: canonical words get a flatter
    # weight than plain words
    return rows, terms, inv, {w: 1.0 + (CONCEPT_IDF if w in L.CONCEPTS or w.startswith(('voice:', 'fam:')) else 1.0)
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
    said = b.get('written') or {}
    for k, w in _said_words(b.get('text')).items():     # the very words said break ties between the designs of
        if w not in L.STOP:                             # a job: "ocean waves on a beach" is the one whose job is
            q['sw:' + k] = SAID_WEIGHT                  # 'beach' itself, not the one with 'pebble beach'
    for p in b.get('phrases', ()):
        q[p] = WORD_WEIGHT * (0.5 if set(said.get(p, '').split()) & ctx else 1.0)
    for key, wt in (('refs', REF_WEIGHT), ('context', CONTEXT_WEIGHT), ('parts', PART_WEIGHT)):
        for c in b.get(key, ()):
            q[c] = max(q.get(c, 0.0), wt)
    head = next((c for c in b['concepts'] if c not in L.GENERIC), None)
    for c in b['concepts']:
        q[c] = HEAD_WEIGHT if c == head else max(q.get(c, 0.0), CONCEPT_WEIGHT)
        for w in said.get(c, '').split():           # "a tock": the family called so ('tock/', a knock) by name
            if w not in ARTICLES:
                q[f'fam:{stem(w)}:{c}'] = q[c]
    for c, w in list(q.items()):
        up = L.BROADER.get(c)
        if up and up not in q and w >= CONTEXT_WEIGHT:
            q[up] = BROADER_WEIGHT * w
    if not (b['concepts'] or b.get('parts') or b.get('context')):
        if b.get('dir'):
            q['riser' if b['dir'] > 0 else 'downlifter'] = CONTEXT_WEIGHT   # "something that keeps climbing"
        elif b.get('loop'):
            q['music'] = PART_WEIGHT                                       # "a loop for a video": music
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
    # "three knocks, 2 seconds", "clapping, 4 seconds", "footsteps on concrete, 4 seconds": the length is the
    # action's, not one event's
    if b.get('dur') and not (b.get('events') or 0) > 1 and not acted(b) and not b.get('perform'):
        r = abs(math.log2(max(b['dur'], 0.01) / d))
        f *= max(0.5, 1.0 - LENGTH_SLOPE * max(0.0, r - 1.0))
    if b.get('dur_max') and d > b['dur_max']:
        f *= max(0.5, 1.0 - 1.6 * LENGTH_SLOPE * max(0.0, math.log2(d / b['dur_max']) - 0.5))
    return f


def chip(row):
    """True for a chip-sound design (8-bit, retro)"""
    return 'retro' in row['tags'] or row.get('material') == 'retro'


def kit_out(b, row):
    """True for a drum of the kit or a hand drum (kick, tom, snare, cajon, conga ...) asked for by a film or motion
    request that names no part of the kit nor the drum itself: "a thud as the logo settles" is the film's low hit,
    "a kick drum on the beat of the trailer" the kit's, "a cajon hit for the video" the cajon's"""
    fam = row['id'].split('/')[0]
    return (b.get('domain') == 'motion' and fam in L.KIT_DRUMS
            and not set(tokens(b.get('text') or '')) & (L.KIT_WORDS | set(fam.split('_'))))


def excluded(b, row, t):
    """True when the request's hard filters rule a design out whatever it sounds like: chip sound asked for and the
    design is not one (or the other way round), a loop asked for and the design is a one-shot, or the design plays a
    job the request rules out ("not an error"), or it is a drum of the kit a film request did not ask for (kit_out)"""
    if (b.get('material') == 'retro') != chip(row):
        return True
    if b.get('loop') and not row.get('loop'):
        return True
    if kit_out(b, row):
        return True
    return any(c in t['uses'] or c in t['name'] for c in b.get('not', ()))


def named(why):
    """True when the words that decided a design (rank's `why`) include a job or a sound (a canonical word or a
    voice, or a job of several words matched as a whole), not only adjectives and plain words"""
    if 'voices' not in _cache:
        _cache['voices'] = frozenset(L.VOICES.values())
    return any(w in L.CONCEPTS or w in _cache['voices'] or w.startswith('ph:') for w in why)


def shown(b, why):
    """rank's `why` as a reader sees it: every word as the request wrote it ('coins', not coin_object; 'lake', not
    the stem 'lak'; 'level up' for a phrase), a broader job by the request's word that led to it, anything else
    (a voice, an adjective) without its prefix or underscore; each once"""
    said = dict(b.get('written') or {})
    for c in [c for k in ('concepts', 'parts', 'context') for c in b.get(k, ())]:
        if L.BROADER.get(c) and c in said:
            said.setdefault(L.BROADER[c], said[c])      # 'bell' found through "glasses" shows as "glasses"
    plain = _said_words(b.get('text'))
    out = [said.get(w) or plain.get(w.replace('sw:', '')) or w.replace('voice:', '').replace('ph:', '')
           .replace('_', ' ') for w in why]
    out = list(dict.fromkeys(out))
    return [w for w in out if not any(w != o and f' {w} ' in f' {o} ' for o in out)]   # a word once


def _families():
    """the canonical words that name a family of the catalogue: the sounds and jobs a design exists for"""
    idx = _default_index()
    if _cache.get('fam_key') != id(idx):
        _cache['fam_key'] = id(idx)
        _cache['families'] = frozenset().union(*(t['family'] for t in idx[1]))
    return _cache['families']


def uncovered(b, why):
    """the sounds and jobs the request names (its head and parts, as written) that the words a design was picked
    for (rank's `why`) do not include: "a water drop into a warm chime" answered by a bell leaves out the water drop.
    Only a word that names a family of the catalogue counts (a sound some design exists for: water, waves, whoosh,
    sparkle; not 'cue', 'item' or a verb such as 'grabs'), and not a generic one (lexicon.GENERIC) nor the shape of
    a short sound ('ping', 'pluck': lexicon.SOUND_NOUNS, which any struck design is); a job counts as
    covered when its broader word decided ('swish' by a whoosh). Context ("for a logo"), the event and comparisons
    (`framed`: "when the player grabs a boost", "like a door slamming"), voices, adjectives, materials and numbers
    never count, nor a job the reader inferred without the request naming it"""
    seen, said, out = set(why), b.get('written') or {}, []
    for c in list(b['concepts']) + list(b.get('parts', ())):
        if (c not in said or c in seen or L.BROADER.get(c) in seen or c in L.GENERIC or said[c] in L.SOUND_NOUNS
                or c in b.get('framed', ()) or c not in _families()):
            continue
        if said[c] not in out:
            out.append(said[c])
    return out


def not_made(b):
    """the first sound the request asks for that the catalogue has nothing of (lexicon.NOT_MADE: a dog's bark, a
    scream, a violin, a car horn), as written, or None. Only what the request says of its own sound counts: not a
    negation or comparison ("not a voice", "like a dog's bark"), nor where it plays ("for a dog-walking app"); a
    single word only when the reader did not understand it otherwise ("saying yes" is a success)"""
    ctx = set(b.get('context_words', ()))
    toks = [t for t in tokens(b.get('text') or '') if t not in PUNCT]
    said = set(_said_words(b.get('text')).values())
    text = ' '.join(toks)
    for p in L.NOT_MADE_PHRASES:
        if f' {p} ' in f' {text} ' and not set(p.split()) & ctx and set(p.split()) <= said:
            return p
    for t in toks:
        if (t in L.NOT_MADE and t in b['unknown'] and stem(t) in b['words'] and t in said and t not in ctx
                and stem(t) not in ctx):
            return t
    for c, whose in L.ONE_VOICE.items():      # "a woman crying": the only crying is a baby's (lexicon.ONE_VOICE)
        person = next((t for t in toks if t in L.PERSON_WORDS and t in said and t not in ctx), None)
        if c in b.get('concepts', ()) and person and not set(toks) & whose and c in b.get('written', {}):
            return f"{person} {b['written'][c]}"
    return None


def blend(b, lexical, semantic, idx):
    """lexical ranking and semantic z (earshot.sounds.atlas.zmap: {id: z}) as one ranking (module doc of
    earshot.sounds.atlas; constants chosen on the development request sets, eval/requests*.json):
        score = lexical / max(best lexical, LEX_FLOOR) ** LEX_POWER + SEM_WEIGHT * max(z, Z_FLOOR)
    A request that names a job or a sound has a high best lexical score: dividing by its power < 1 keeps the lexical
    gaps of a named job larger than the spread of z, so the job's winner stays. When no word of the request matched
    any design and the lexical ranking is by feel alone ("a thin airy streak fading away"), the lexical scores are
    scaled to at most UNNAMED_LEX instead, under the spread of z: z decides, adjectives break ties. A design the
    hard filters exclude (`excluded`) cannot gain from z (only lose); a design that matched no word gets the semantic
    part scaled by its length fit"""
    rows, terms = idx[0], idx[1]
    lex = {r['id']: r for r in lexical}
    s1 = max([r['score'] for r in lexical] or [0.0])
    feel = set(b.get('hints', ()))
    worded = any(w not in feel for r in lexical[:1] for w in r['why'])
    norm = max(s1, LEX_FLOOR) ** LEX_POWER if worded else max(s1, LEX_FLOOR) / UNNAMED_LEX
    out = []
    for k, row in enumerate(rows):
        z = semantic.get(row['id'])
        r = lex.get(row['id'])
        s = r['score'] if r else 0.0
        if z is None:
            sem = 0.0
        else:
            sem = max(z, Z_FLOOR)
            if excluded(b, row, terms[k]):
                sem = min(sem, 0.0)
            elif r is None and sem > 0:
                sem *= length_fit(b, row)
        if r is None and sem <= 0:
            continue
        why = list(r['why']) if r else []
        if z is not None and z >= Z_SAYS:
            why.append(f'sounds like it (z {z:+.1f})')
        out.append({'id': row['id'], 'score': round(s / norm + SEM_WEIGHT * sem, 3), 'why': why,
                    'lex': s, 'z': None if z is None else round(z, 2)})
    out.sort(key=lambda x: (-x['score'], x['id']))
    return out


def rank(b, rows=None, top=8, idx=None, semantic=None):
    """the designs that answer a brief, best first: [{'id', 'score', 'why'}]. rows: catalogue rows (default: every
    design; the index of the whole catalogue is built once and kept); a design must share at least one word with
    the request to be listed; when none does, the designs closest in feel are listed. semantic: {id: z} of how
    much each design sounds like the request (earshot.sounds.atlas.zmap), blended in (`blend`; the rows then also
    carry 'lex' and 'z'); None = the lexical ranking alone"""
    if idx is None:
        idx = index(rows) if rows is not None else _default_index()
    if semantic is not None:
        return _named_member(b, blend(b, rank(b, top=None, idx=idx), semantic, idx), idx)[:top]
    rows, terms, inv, idf = idx
    q = _query(b)
    score, why = {}, {}
    for w, qw in sorted(q.items(), key=lambda kv: kv[0].startswith('sw:')):
        for k, fw in inv.get(w, ()):
            if w.startswith('sw:') and k not in why:
                continue                  # a word said only breaks ties between designs found otherwise
            score[k] = score.get(k, 0.0) + qw * idf.get(w, 1.0) * fw
            if not w.startswith(('feel:', 'sw:', 'fam:')):
                why.setdefault(k, []).append(w.replace('voice:', ''))
    mat, dom, loop = b.get('material'), b.get('domain'), b.get('loop')
    said = set(_said_words(b.get('text')))
    head = next((c for c in b['concepts'] if c not in L.GENERIC), None)
    named_ = [k for k in score if k in why
              and (not terms[k]['learned'] or learned_ok(b, rows[k]['id'], terms[k], said))]
    invented = [k for k in score if not terms[k]['learned']]
    # the interface job alone ("tap", "hover", "toggle on", "type"), no thing named ("next page": a page): ROLE
    own = set().union(*(_family_words(i) for i, r in _roles().items() if head in r))
    alone = _interface_only(b)                                           # as learned_ok lets a role answer
    thing = any(terms[k]['learned'] and (_family_words(rows[k]['id']) & said) - own for k in named_)
    plain = alone and not said & TONE_STEMS and not thing
    # an interface job without musical words: no pitched design (any interface job named: `ui_job`), and the real
    # thing of the job (INTERFACE_JOBS) raised when the job is the head, or the first job of a request whose head is
    # a word too common to lead ("a modal slides in": open), the kind of sound ("a swoosh that flies away": send)
    # or the place of the job ("open a menu": open)
    music = bool(said & MUSIC_STEMS) or bool(b.get('voice')) or dom == 'music' or 'music' in b['concepts']
    # a sound word as the head and the moment after it ("a quick swoosh as the message goes out"): the job of the
    # moment when its real thing is that kind of sound (a send is a swoosh; a modal opening is no thump)
    kinds = {r['id']: r.get('kind') for r in rows}
    moment = next((c for c in b['context'] if c in L.INTERFACE_JOBS and kinds.get(L.INTERFACE_JOBS[c]) == head),
                  None) if head in L.SOUND_FAMILIES and not any(c in L.INTERFACE_JOBS for c in b['concepts']) else None
    ui_job = dom not in ('game', 'motion') and mat != 'retro' and (
        any(c in L.INTERFACE_JOBS for c in b['concepts']) or bool(moment))
    lead = None if head in L.SOUND_FAMILIES or head in L.INTERFACE_PLACES else head
    job = L.INTERFACE_JOBS.get(lead) or (next((L.INTERFACE_JOBS[c] for c in b['concepts'] + [moment]
                                               if c in L.INTERFACE_JOBS), None) if lead is None else None)
    job = job if ui_job else None
    # a page or an app (its domain said: a button, a list, a screen) asks for the sound of a real thing too: a
    # pitched design answers it only when the words ask for music, for jobs that all have a design without a note
    # ("a pop for each icon"; a startup, whose designs all have notes, keeps them)
    ui_dom = dom == 'ui' and mat != 'retro' and bool(b['concepts']) and set(b['concepts']) <= _real_jobs(idx)
    # a motion job (a logo, a hit, a transition ...; an intro, an appearance, a shine, or any job a design without a
    # note plays, when a film or video is the domain), not a game, and no word that asks for notes: a design with a
    # note in it steps back (FILM_TONE; a countdown, all beeps, keeps them)
    film = (dom != 'game' and not (said & FILM_MUSIC) and not b.get('voice') and 'music' not in b['concepts']
            and any(c in L.MOTION_JOBS or (dom == 'motion' and (c in L.MOTION_DOMAIN_JOBS or c in _real_jobs(idx)))
                    for c in b['concepts']))
    pool = (named_ or [k for k in invented if dom in rows[k].get('domains', ())]
            or invented or list(score))                # nothing named: the closest in feel, in the request's domain
    # another family of the job's real things (earshot.sounds.interface) named by its own word ("task complete":
    # complete/, a success by another word) is what was asked for: the job's default is not raised over it
    job_c = next((c for c in b['concepts'] if L.INTERFACE_JOBS.get(c) == job), None) if job else None
    rival = job_c and any(job_c in terms[k]['family'] and 'interface' in rows[k].get('tags', ())
                          and '_' not in rows[k]['id'].split('/')[0]
                          and rows[k]['id'].split('/')[0] != job.split('/')[0]
                          and f"fam:{stem(rows[k]['id'].split('/')[0])}:{job_c}" in q for k in pool)
    if mat == 'retro' and not any(chip(rows[k]) for k in pool):
        # chip sound asked for and no chip design shares a word: every chip design, by feel and direction
        pool = [k for k in range(len(rows)) if chip(rows[k])]
        for k in pool:
            score[k] = 1.0 + score.get(k, 0.0)
            why.setdefault(k, ['retro'])
    is_retro = [chip(rows[k]) for k in pool]
    only_retro = mat == 'retro' and any(is_retro)
    out = []
    for k, retro in zip(pool, is_retro):
        row, t, s = rows[k], terms[k], score[k]
        if kit_out(b, row):
            continue
        for c in b.get('not', ()):
            if c in t['uses'] or c in t['name']:
                s *= 0.5                                       # "not an error", "smooth, not clicky"
        s *= row.get('prior', 1.0)
        if row['id'] in L.BY_NAME_ONLY and alone and not said & TONE_STEMS:
            s *= BY_NAME
        elif (ui_job or ui_dom) and not music and _pitched(row, t):
            s *= BY_NAME      # "a success sound": no marimba figure; "a success chime", "on a harp": the figure
        elif ui_job and music and 'interface' in row.get('tags', ()):
            s *= BY_NAME
        elif job == row['id'] and not thing and not rival:
            s *= REAL         # "a success sound", "toggle on", "message received": the real thing of the job
        elif 'interface' in row.get('tags', ()) and (not ui_job or (lead and lead not in L.INTERFACE_JOBS)):
            s *= BY_NAME      # the request is about something else ("when the brand pops up": a logo)
        elif (plain and t['learned'] and head in _roles().get(row['id'], ()) and not job
              and not (mat and mat not in t['mats'])):
            s *= ROLE         # "a tap", "hover", "toggle on": the recorded one; not when it is of another material
            #                   than the request names ("a wooden tap" is no plastic key)
        if film and _pitched(row, t) and 'real' not in row.get('tags', ()):
            s *= FILM_TONE    # "a logo reveal hit": a low hit, not a felt piano; "a logo motif": the notes
        if dom and row.get('domains'):
            s *= 1.15 if dom in row['domains'] else 0.85
        if mat == 'retro':
            s *= 1.0 if retro else 0.05 if only_retro else 0.25   # chip sound asked for: only chip
        elif retro:
            s *= 0.35                                          # chip sound only when asked for
        if row.get('kind') == 'music' and q.get('music', 0.0) < PART_WEIGHT:
            s *= 0.35         # music only when asked for (not only where it plays): its other jobs say where it plays
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
    return _named_member(b, out, idx)[:top]


def _real_jobs(idx):
    """the jobs (concepts) a design without a note in it plays as its family: the sound of a real thing exists"""
    if _cache.get('real_key') != id(idx):
        _cache['real_key'] = id(idx)
        _cache['real_jobs'] = frozenset().union(*(t['family'] for r, t in zip(idx[0], idx[1]) if not _pitched(r, t)))
    return _cache['real_jobs']


def _pitched(row, t):
    """True for an invented design with a note in it (a figure on an instrument, a tick of a wood block, a game's
    menu blip): on a website it reads as a toy or a ringtone"""
    return not t['learned'] and bool(row.get('pitch'))


def _roles():
    """{id: the interface jobs a learned design plays} (earshot.sounds.learned index `roles`), read once"""
    if 'roles' not in _cache:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'resyn', 'index.json')
        with open(path, encoding='utf-8') as fh:
            _cache['roles'] = {e['id']: frozenset(e.get('roles', ())) for e in json.load(fh)}
    return _cache['roles']


def thing_named(sid, why):
    """True when sid is a design of earshot.sounds.learned (recordings of one thing) and the words that decided it
    (rank's `why`) name its thing: "a dumbbell dropped on the floor" names dumbbell/place, which is no guess"""
    return sid in _roles() and bool(_family_words(sid) & {stem(w) for w in why})


def _interface_only(b):
    """True when a request names interface jobs a learned design plays (its roles) and nothing else: "a tap",
    "button click sound", "toggle on"; not "a click like an old SLR" (a shutter is a part of it)"""
    if 'any_role' not in _cache:
        _cache['any_role'] = frozenset().union(*_roles().values())
    return bool(b['concepts']) and not b.get('parts') and set(b['concepts']) <= _cache['any_role']


def learned_ok(b, sid, t, said):
    """True when a learned design (earshot.sounds.learned: recordings of one thing) may answer the request: the
    request names its thing, as a job or a word it says (`said`: _said_words stems): the job its family is by name
    ("a footstep on gravel": step/, a footstep by lexicon.FAMILY_SENSE; "toggle dark mode": switch/), the
    interface job it plays (its `roles`: "hover" is a tick of a dial), its family's own word ("a wooden drawer",
    "a pot clanging") or the whole name of the invented design it is the recording of (lexicon.TWINS: "a knock on the
    desk" names knock/desk, "a tap on the table" tap/table). A recording never answers by feel alone nor by a job
    its thing merely has: "sword on armour" is a clang, but no recorded pot or anvil; "a heavy impact" reaches no
    recorded dumbbell; "a cue when the car
    passes a checkpoint" names a cue, and the passing car is the event, not a recording asked for (`framed`, when
    the request names something else as its sound)"""
    asked = set(b['concepts'])                           # "... muffled and bubbly": a part, not a recorded splash
    if _roles().get(sid, frozenset()) & asked and _interface_only(b):
        return True                     # the interface job alone ("a click like an old SLR" is a shutter's click)
    if (b.get('events') or 0) > 1 and any(v == sid and _family_words(k) & set(b.get('counted', ()))
                                          for k, v in L.TWINS.items()):
        return True                     # "four knocks": a number of events only the recorded twin can perform
    own = said - set(b.get('context_words', ()))         # "passes a save point flag": the flag is where, not what
    if sid.split('/')[0] in L.SOUND_FAMILIES and any(v == sid and _family_words(k) & own and _name_words(k) <= own
                                                     and _twin_said(b, k) for k, v in L.TWINS.items()):
        return True                     # "a knock on the desk" names knock/desk, whose recorded thing this is
    framed = set(b.get('framed', ()))
    if framed and asked - framed:
        asked -= framed
        own -= {stem(w) for c in framed for w in WORD.findall((b.get('written') or {}).get(c, '').lower())}
    fam = _family_words(sid)                             # "a drum hit" names no brake drum: every word of it
    by_family = (t['exact'] in asked or (any(L.SAME_THING.get(c, 0) == t['exact'] for c in asked)
                                         and bool(_name_words(sid) & own))    # "unlocking a door with a key"
                 or (fam <= own if len(fam) > 1 else bool(fam & own)))
    if sid.split('/')[0] in L.SOUND_FAMILIES:            # "a whoosh" names no recorded rope: "a rope whoosh" does
        return by_family and bool(_name_words(sid) & said)
    return by_family


def _name_words(sid):
    """the stems of the words of a design's own name ('pad/warm' -> {'warm'}, 'logo_sting/soft_bloom' -> {'soft',
    'bloom'})"""
    return {stem(p) for p in sid.split('/', 1)[1].split('_') if p.isalpha()}


def _family_words(sid):
    """the stems of the words of a design's family ('cut_hit/kick' -> {'cut', 'hit'})"""
    return {stem(p) for p in sid.split('/', 1)[0].split('_') if p.isalpha()}


def _said_words(text):
    """{stem: word} of the words a request says of its own sound (no articles): not what a negation reaches ("not
    soft"), nor a comparison ("like someone tapped it", "than a click"); none at all when it asks for the opposite
    of what it names ("the next-step sound but the other way")"""
    said = {}
    for s, t in _said_seq(text):
        if s:
            said.setdefault(s, t)
    return said


def _said_seq(text):
    """the words of `_said_words` in order, as (stem, word); (None, None) where a word was left out, so that only
    words said next to each other are next to each other"""
    toks = tokens(text or '')
    if 'opposite' in toks or any(w in ' '.join(toks) for w in OTHER_WAY):
        return []
    seq, neg, aside = [], 0, False
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t in ARTICLES and not aside and not neg:
            continue
        if t in PUNCT:
            neg, aside = 0, False
        elif t in L.NEGATIONS:
            neg = 2
        elif t == 'than' or (t == 'like' and nxt not in LIKE_JOB_NEXT and nxt is not None and nxt not in PUNCT):
            aside = True
        elif not aside and not neg:   # stop words too: toggle/off, notify/single are named by them
            seq.append((stem(t), t))
            continue
        elif neg and t not in ARTICLES:
            neg -= 1
        seq.append((None, ',' if t in PUNCT else None))
    return seq


def _compound(sid, seq):
    """True when a request says a design's name followed by its family, word for word: "a warm pad", "tiny drip",
    "punch cut hit" (the head noun last, as English compounds go, and a noun: "deep booming" is no boom/deep), and
    not after "with", "and" or "of" in its clause ("a swell of airy chimes and a warm pad": a part, not the sound)"""
    fam, name = sid.split('/')
    want = [stem(p) for p in name.split('_')] + [stem(p) for p in fam.split('_')]
    stems = [s for s, _ in seq]
    for k in range(len(stems) - len(want) + 1):
        last = seq[k + len(want) - 1][1] if stems[k:k + len(want)] == want else None
        if last and (last == fam.split('_')[-1] or not modifier(last)):
            before = [w for _, w in seq[:k]]
            clause = before[len(before) - before[::-1].index(','):] if ',' in before else before
            if not set(clause) & {'with', 'and', 'of'}:
                return True
    return False


def _twin(sid, out, idx):
    """the member of the same name in a family that answers the same job and ranks above sid's ("a deep impact":
    the bank's impact/deep is named word for word, but boom/deep, the film family's deep member, ranks above it; a
    family that is that job, not one that merely lists it: kick/deep is a drum, not a deep impact), or None"""
    if not idx:
        return None
    if _cache.get('pos_key') != id(idx):
        _cache['pos_key'], _cache['pos'] = id(idx), {r['id']: k for k, r in enumerate(idx[0])}
    pos, terms = _cache['pos'], idx[1]
    fam, name, jobs = sid.split('/')[0], _name_words(sid), terms[pos[sid]]['family']
    for r in out:
        if r['id'] == sid:
            return None
        t = terms[pos[r['id']]]
        if (r['id'].split('/')[0] != fam and _name_words(r['id']) == name and jobs & t['family']
                and not _family_words(r['id']) < _family_words(sid)):   # "punch cut hit": no plain hit/punch
            return r['id']
    return None


def _preferred(b, sid, out, idx, twins=True):
    """a learned member of sid's family (a recording played as a new take: earshot.sounds.learned, tagged 'real')
    with a prior at least as high that scores at least MEMBER_SHARE of sid's on the request's own words, or None: "a
    cork pop" names the invented cork/pop word for word, but the family's learned member is the same thing (its
    descriptions say every word of the name: "a cork ... with a hollow pop") and preferred. A member named by its
    character ("a deep impact": boom/deep) is what was asked for and stays, unless its name and family together are the
    name of a thing ("light switch": a switch for the light, not a light one; lexicon.CONCEPTS)"""
    if not idx:
        return None
    twin = _learned_twin(b, sid, out, idx) if twins else None
    if twin:
        return twin
    fam_, name_ = sid.split('/')
    if (any(modifier(w) for w in name_.split('_'))
            and _key(name_.split('_') + fam_.split('_')) not in _reverse()['c']):
        return None
    rows, terms = idx[0], idx[1]
    if _cache.get('pos_key') != id(idx):
        _cache['pos_key'], _cache['pos'] = id(idx), {r['id']: k for k, r in enumerate(rows)}
    pos = _cache['pos']
    fam, prior = sid.split('/')[0], rows[pos[sid]].get('prior', 1.0)
    if 'real' in rows[pos[sid]].get('tags', ()):
        return None
    own = next((r['score'] for r in out if r['id'] == sid), None)
    name = _name_words(sid)
    said = _cache.setdefault('said_stems', {})
    for r in out:
        k = pos[r['id']]
        if r['id'].split('/')[0] != fam or 'real' not in rows[k].get('tags', ()):
            continue
        if r['id'] not in said:
            said[r['id']] = {stem(w) for w in WORD.findall(' '.join(list(rows[k].get('says', ()))
                                                                     + list(rows[k].get('tags', ()))).lower())}
        if (rows[k].get('prior', 1.0) >= prior and name <= said[r['id']] | _name_words(r['id']) and own is not None
                and r['score'] >= MEMBER_SHARE * own and not excluded(b, rows[k], terms[k])
                and not (b.get('dir') and terms[k]['dir'] == -b['dir'])):
            return r['id']
    return None


def _twin_said(b, sid):
    """True unless the invented sid's recorded twin is the same thing only with words the request does not say
    (lexicon.TWIN_WORDS: "a coin dropped" is one coin, not the recorded handful)"""
    return sid not in L.TWIN_WORDS or bool(set(tokens(b.get('text') or '')) & set(L.TWIN_WORDS[sid]))


def _learned_twin(b, sid, out, idx):
    """the learned design of the same thing as the invented sid (lexicon.TWINS: "knocking on a wooden door" is
    door/knock, not knock/door; and for a film request without a word of the drum kit, the film's cymbal for the
    kit's: lexicon.FILM_CYMBAL) when it answers the request too (it is ranked: the request names its thing) and
    nothing rules it out (the hard filters, a direction against the request's), or None"""
    twin = L.TWINS.get(sid) if _twin_said(b, sid) else None
    fam = sid.split('/')[0]
    if (not twin and fam in L.FILM_CYMBAL and b.get('domain') == 'motion'
            and not set(tokens(b.get('text') or '')) & L.KIT_WORDS):
        twin = L.FILM_CYMBAL[fam]                    # "a cymbal crash for the trailer": the film's cymbal
    if not twin or not _family_words(sid) & set(_said_words(b.get('text'))):
        return None                     # only where the request says the invented one's family word ("punch impact"
        #                                 reaches hit/punch through impact/punch, but says no hit)
    if not any(r['id'] == twin for r in out) and (fam not in L.SOUND_FAMILIES or set(b['concepts']) & (
            set(L.INTERFACE_JOBS) | {'cue', 'notify'})):
        return None                     # an unranked recording only for a kind of sound named as the thing itself
        #                                 ("a knock": the door's knock, recorded; "a knock cue" keeps the cue)
    rows, terms = idx[0], idx[1]
    if _cache.get('pos_key') != id(idx):
        _cache['pos_key'], _cache['pos'] = id(idx), {r['id']: k for k, r in enumerate(rows)}
    k = _cache['pos'][twin]
    if excluded(b, rows[k], terms[k]) or (b.get('dir') and terms[k]['dir'] == -b['dir']):
        return None
    return twin


def _learned_first(b, out, idx):
    """a ranking whose first design gives way to a learned member of its family that is the same thing
    (_preferred: "a cork pop", "a door closing")"""
    pref = _preferred(b, out[0]['id'], out, idx) if out else None
    if not pref or pref == out[0]['id']:
        return out
    k = next((i for i, r in enumerate(out) if r['id'] == pref), None)
    if k is None:                       # a learned twin the request reached through its invented one only
        return [dict(out[0], id=pref)] + out
    best = dict(out[k], score=out[0]['score'], why=list(dict.fromkeys(out[k]['why'] + out[0]['why'])))
    return [best] + out[:k] + out[k + 1:]


def _named_member(b, out, idx=None):
    """a ranking where a design the request names by its own words comes first: "tiny drip", "a metal impact" (its
    name and then its family said word for word, as the request's own sound: `_said_seq`, `_compound`) is that
    design unless something rules it out (the hard filters, a direction against the request's); else the family
    that won is answered by the member the request names: "a warm pad chord" is pad/warm, though "warm" is an
    adjective (it still moves its axis) and the words of another member may score a little higher (MEMBER_SHARE).
    Among the ranked members of the first design's family, the one whose name shares the most words with the
    request goes first (then by score), and then a learned design of the same thing (`_learned_first`); the order
    of everything else stays"""
    seq = _said_seq(b.get('text'))
    said = {s: t for s, t in reversed(seq) if s}
    rows, terms = (idx[0], idx[1]) if idx else ((), ())
    score = {r['id']: r['score'] for r in out}
    full = []
    framed = set(b.get('framed', ()))
    sound = set(b['concepts']) - framed if framed else set()   # what the request names as its sound, when it does
    for k, row in enumerate(rows):
        i = row['id']
        if not (_name_words(i) and _compound(i, seq)) or excluded(b, row, terms[k]):
            continue
        if b.get('dir') and terms[k]['dir'] == -b['dir']:
            continue
        if terms[k]['family'] & set(b.get('parts', ())) and not terms[k]['family'] & set(b['concepts']):
            continue                  # "... with airy chimes and a warm pad": a part of the sound, not the sound
        if sound and terms[k]['family'] & framed and not terms[k]['family'] & sound:
            continue                  # "a cue when the car passes ...": the event, not the sound
        full.append((len(_name_words(i)) + len(_family_words(i)), score.get(i, 0.0), -k, i))
    if full:
        first = max(full)[3]
        sid = _twin(first, out, idx) or first
        sid = _preferred(b, sid, out, idx, twins=sid == first) or sid    # "punch cut hit" is no fist on a body
        if out and out[0]['id'] == sid:
            return out
        old = next((r for r in out if r['id'] == sid), {'id': sid, 'why': []})
        words_ = [said[w] for w in sorted((_name_words(sid) | _family_words(sid)) & said.keys())]
        best = dict(old, score=out[0]['score'] if out else 1.0, why=list(dict.fromkeys(old['why'] + words_)))
        return [best] + [r for r in out if r['id'] != sid]
    if not out:
        return out
    said = {s: t for s, t in said.items() if t not in L.STOP}     # "in a platformer" does not name teleport/in
    fam = out[0]['id'].split('/')[0]
    hits = [(len(_name_words(r['id']) & said.keys()), r['score'], -i) for i, r in enumerate(out)
            if r['id'].split('/')[0] == fam]
    if max(hits)[0] == 0:
        # no member of the winning family is named: a design of another family found by the same words, of the same
        # kind, whose own name the request says ("tape stop": downlifter/tape over fall/air, both a downlifter)
        # a name that is an adjective ("soft", "bright", "airy"), the very word that found the concept ("swell") or a
        # word that names a sound by itself ("hit" of darbuka/hit in "a drum hit") names no design: those stay with
        # the family that won
        top, kind = set(out[0]['why']), {r['id']: r['kind'] for r in rows}
        terms_ = {(b.get('written') or {}).get(c, c) for c in top}

        def kin(i):
            return {w for w in _name_words(i) & said.keys() if not modifier(said[w]) and said[w] not in terms_
                    and not read(said[w])['concepts']} - top
        hits += [(len(kin(r['id'])), r['score'], -i) for i, r in enumerate(out)
                 if i and set(r['why']) == top and kind.get(r['id']) and kind.get(r['id']) == kind.get(out[0]['id'])]
    n, _, neg = max(hits)
    pos = -neg
    share = MEMBER_SHARE if out[pos]['id'].split('/')[0] == fam else KIN_SHARE
    if (n == 0 or pos == 0 or len(_name_words(out[0]['id']) & said.keys()) >= n
            or out[pos]['score'] < share * out[0]['score']):
        return _learned_first(b, out, idx)
    best = dict(out[pos], score=out[0]['score'])
    named_ = [said[w] for w in sorted(_name_words(best['id']) & said.keys())]
    best['why'] = list(dict.fromkeys(list(best['why']) + named_))
    rest = out[:pos] + out[pos + 1:]
    if share == KIN_SHARE:              # the other family's default takes the named design's score: not a near tie
        rest[0] = dict(rest[0], score=out[pos]['score'])
    return _learned_first(b, [best] + rest, idx)