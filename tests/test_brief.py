"""earshot.brief: reading a request (numbers, key, pitch, counts, negation, comparatives, materials, voices, domain,
loop, context) and ranking the catalogue (anchor requests in plain words and the design family they must find
first). Model-free and fast: the catalogue index is built once for the module."""
import time

import pytest

from earshot import brief
from earshot.sounds import lexicon as L


@pytest.fixture(scope='module')
def idx():
    from earshot.sounds import registry
    return brief.index(registry.rows())


# ---- reading: numbers and units

def test_duration_in_ms_and_seconds():
    assert brief.read('a whoosh, 400 ms')['dur'] == pytest.approx(0.4)
    assert brief.read('a riser of 2.5 seconds')['dur'] == pytest.approx(2.5)
    assert brief.read('rain bed, 1 min')['dur'] == pytest.approx(60.0)


def test_duration_ceiling():
    b = brief.read('a tap for a pay button, under 80 ms')
    assert b['dur_max'] == pytest.approx(0.08) and b['dur'] is None
    assert 'tap' in b['concepts']


def test_counts():
    assert brief.read('coin pickup, 3 variations')['n'] == 3
    assert brief.read('three takes of a footstep on wood')['n'] == 3
    assert brief.read('a two-note chime')['n'] is None
    assert brief.read('a two-note chime')['notes'] == 2


def test_events_and_the_action():
    """a number of events, a single one, an action word or a plural (brief.performance decides with them)"""
    assert brief.read('three knocks on a door')['events'] == 3
    assert brief.read('a few footsteps on gravel')['events'] == 3
    assert brief.read('a single knock')['events'] == 1 and not brief.read('a single knock')['perform']
    assert brief.read('one footstep on wood')['events'] == 1
    assert brief.read('three takes of a footstep on wood')['events'] is None
    assert brief.read('a two-note chime')['events'] is None and brief.read('an 8-bit coin')['events'] is None
    assert brief.read('2 seconds of typing')['events'] is None and brief.read('2 seconds of typing')['perform']
    assert brief.read('knocking on a wooden door')['perform'] and brief.read('keys jingling')['perform']
    assert not brief.read('a footstep on gravel')['perform'] and not brief.read('a knock on the door')['perform']
    assert 'footstep' in brief.read('footsteps on gravel')['plural']
    assert 'glass' in brief.read('glasses clinking')['plural'] and brief.read('a glass')['plural'] == []
    # a singular that still means the action: the pattern's words ("a knock on the door": nobody knocks once)
    knock = {'name': 'knock', 'singular': ['door']}
    assert brief.performance(brief.read('a knock on the door'), 'door/knock', knock) == (True, None)
    assert brief.performance(brief.read('a knock'), 'door/knock', knock) == (False, None)
    assert brief.performance(brief.read('a single knock on the door'), 'door/knock', knock) == (False, 1)


@pytest.mark.parametrize('text,voice', [
    ('a success sound on a marimba', 'marimba'),
    ('a notification with a wooden mallet', 'marimba'),
    ('a notification played on a harp', 'harp'),
    ('a pizzicato confirmation', 'pizzicato'),
    ('plucked strings', 'pizzicato'),
    ('a soft piano notification', 'felt_piano'),
    ('upright piano', 'felt_piano'),
])
def test_the_real_instruments_are_reached(text, voice):
    assert brief.read(text)['voice'] == voice


@pytest.mark.parametrize('text', ['a warm notification', 'a wooden success sound', 'a playful tap',
                                  'a dreamy chime', 'an elegant confirmation'])
def test_general_words_name_no_instrument(text):
    assert brief.read(text)['voice'] is None


@pytest.mark.parametrize('text,family,voice', [
    ('a notification on a harp', 'notify/', 'harp'),
    ('a success sound with pizzicato strings', 'success/', 'pizzicato'),
    ('an error sound on a soft piano', 'error/', 'felt_piano'),
    ('a notification on a soft piano', 'notify/', 'felt_piano'),
    ('a success sound on a soft piano', 'success/', 'felt_piano'),
    ('a success sound on an upright piano', 'success/', 'felt_piano'),
])
def test_an_instrument_after_a_job_plays_the_figure(idx, text, family, voice):
    """an instrument named as the player of a job (also behind an adjective: "on a soft piano") sets the voice of
    the job's figure, not a single note of that instrument; "pizzicato strings" is one name"""
    b = brief.read(text)
    r = brief.rank(b, idx=idx, top=3)
    assert r[0]['id'].startswith(family) and b['voice'] == voice, (text, r, b['voice'])
    assert not b['unknown'], b['unknown']


def test_a_single_knock_is_a_plain_answer(idx):
    """the invented twin of the pick (knock/door for door/knock) and its family are the same thing, not a doubt"""
    from earshot.make import doubt
    b = brief.read('a single knock on a door')
    assert doubt(b, brief.rank(b, idx=idx, top=8)) is None


def test_pitch_hz_note_and_middle_c():
    assert brief.read('a beep at 880 Hz')['pitch'] == pytest.approx(880.0)
    assert brief.read('a marimba note, C5')['pitch'] == 'C5'
    assert brief.read('one grand piano key, middle C')['pitch'] == 'C4'


def test_key_and_scale():
    b = brief.read('a logo sting in D minor')
    assert (b['key'], b['scale']) == ('D', 'minor')
    b = brief.read('success chime in F#')
    assert b['key'] == 'F#' and b['scale'] is None
    assert brief.read('a whoosh in a trailer')['key'] is None


# ---- reading: feelings, negation, comparatives

def test_negation_flips_adjectives():
    b = brief.read('a beep, not harsh')
    assert b['axes']['bright'] < 0 and b['axes']['hard'] < 0
    assert brief.read('an impact with no reverb')['axes']['space'] < 0
    assert brief.read('a click without a tail')['axes']['ring'] < 0


def test_negated_job_is_avoided():
    b = brief.read('a gentle chime, more serious than a notification but not an error')
    assert 'error' in b['not'] and 'error' not in b['concepts']
    assert 'notify' not in b['concepts']                  # after "than": a reference, not the request
    assert 'notify' in b['refs']


def test_comparatives_and_intensifiers():
    assert brief.read('brighter')['axes']['bright'] > 0
    assert brief.read('a bigger boom')['axes']['size'] > 0
    soft, very = brief.read('soft tap')['axes']['hard'], brief.read('very soft tap')['axes']['hard']
    assert very < soft < 0
    assert -0.5 < brief.read('slightly darker pad')['axes']['bright'] < 0


def test_envelope_phrases_are_adjectives():
    b = brief.read('a dash with a sharp attack')
    assert b['axes']['hard'] > 0 and 'hit' not in b['concepts'] + b['parts']


def test_direction():
    assert brief.read('a rising chime')['dir'] == 1
    assert brief.read('a descending blip')['dir'] == -1
    assert brief.read('a tick')['dir'] == 0


# ---- reading: materials, voices, domain, loop, context

def test_materials():
    assert brief.read('a wooden tap')['material'] == 'wood'
    assert brief.read('8-bit jump')['material'] == 'retro'
    assert brief.read('a square wave blip')['material'] == 'retro'
    assert brief.read('a blip like an old console game')['material'] == 'retro'
    assert brief.read('a bottle smashing')['material'] == 'glass'


def test_voice_as_the_thing_or_as_the_player():
    b = brief.read('a kalimba note')
    assert b['voice'] == 'kalimba' and 'voice:kalimba' in b['concepts']
    b = brief.read('a success chime played on a kalimba')
    assert b['voice'] == 'kalimba' and 'voice:kalimba' not in b['concepts'] and b['concepts'][0] == 'success'
    b = brief.read('a deep bass boom')
    assert 'impact' in b['concepts'] and not any(c.startswith('voice:') for c in b['concepts'])
    assert brief.read('a music box melody')['voice'] == 'music_box'


def test_domain():
    assert brief.read('a jump for a platformer')['domain'] == 'game'
    assert brief.read('tap feedback in a banking app')['domain'] == 'ui'
    assert brief.read('an impact for a movie trailer')['domain'] == 'motion'
    assert brief.read('a pickup for a browser game')['domain'] == 'game'


def test_loop_and_no_loop():
    assert brief.read('rain loop')['loop'] is True
    assert brief.read('a seamless forest bed')['loop'] is True
    assert brief.read('thunder, no loop')['loop'] is False
    assert brief.read('a one-shot wind gust')['loop'] is False
    assert brief.read('rain on a window')['loop'] is None


def test_context_after_for():
    b = brief.read('a deep impact for a logo reveal')
    assert b['concepts'][0] == 'impact' and 'logo' in b['context'] and 'logo' not in b['concepts']
    b = brief.read('sound for when a dropdown opens')            # no head yet: what follows is the request
    assert 'open' in b['concepts']


def test_like_as_comparison_and_as_job():
    assert 'like' not in brief.read('something quiet, like a whisper of a click')['concepts']
    assert 'like' in brief.read('a like button tap')['concepts']
    assert 'like' in brief.read('sound for a like')['concepts']


def test_opposites():
    assert brief.read('the opposite of a riser')['concepts'] == ['downlifter']
    b = brief.read('like the next-step sound but going the other way')
    assert 'back' in b['concepts'] + b['parts'] and 'next' not in b['concepts'] + b['parts']


def test_toggle_particles():
    assert 'switch_on' in brief.read('when the user toggles dark mode on in settings')['concepts']
    assert 'switch_off' in brief.read('toggle off click')['concepts']


def test_domain_sense():
    assert 'lose' in brief.read('when the player fails a level in a puzzle game')['concepts']
    assert 'error' in brief.read('a failed payment in a banking app')['concepts']


def test_unknown_words_are_reported():
    b = brief.read('a zorblax whoosh')
    assert 'zorblax' in b['unknown'] and 'whoosh' in b['concepts']


# ---- the lexicon

def test_no_synonym_under_two_concepts():
    seen = {}
    for canon, others in L.CONCEPTS.items():
        for w in (canon.replace('_', ' '),) + tuple(others):
            assert w not in seen or seen[w] == canon, f'{w!r} is listed under {seen.get(w)} and {canon}'
            seen[w] = canon


def test_opposites_are_symmetric_and_known():
    for a, b in L.OPPOSITE.items():
        assert L.OPPOSITE[b] == a and a in L.CONCEPTS and b in L.CONCEPTS


# ---- ranking

def test_rank_shape_and_speed(idx):
    b = brief.read('a soft wooden success chime for a finance app, 400 ms, in D')
    r = brief.rank(b, idx=idx, top=5)
    assert 1 <= len(r) <= 5 and set(r[0]) == {'id', 'score', 'why'}
    assert r[0]['id'].startswith('success/') and 'success' in r[0]['why']
    assert all(a['score'] >= b_['score'] for a, b_ in zip(r, r[1:]))
    t0 = time.perf_counter()
    for text in ('coin pickup', 'a big slow whoosh for a trailer', 'rain on a tent, loop', 'kalimba note in E'):
        brief.rank(brief.read(text), idx=idx)
    assert (time.perf_counter() - t0) / 4 < 0.03


def test_rank_default_index_is_cached():
    brief.rank(brief.read('coin'))
    t0 = time.perf_counter()
    brief.rank(brief.read('a soft coin pickup'))
    assert time.perf_counter() - t0 < 0.03


def test_plain_interface_jobs_get_recorded_keys_and_switches(idx):
    """a plain tap, click, hover, toggle, next or key press is the recorded design chosen for that job (the learned
    index's roles, the ui set's sounds); the tonal figures answer when named or asked for as a tone; a recording
    answers only a request that names its thing"""
    from earshot.sounds import learned, registry
    roles = {r: sid for sid, e in learned.index().items() for r in e.get('roles', ())}
    for text, job in (('tap', 'tap'), ('a button press', 'tap'), ('click', 'click'), ('hover', 'hover'),
                      ('toggle on', 'switch_on'), ('toggle off', 'switch_off'), ('switch', 'toggle'),
                      ('next', 'next'), ('type', 'type'), ('key press', 'type'), ('select', 'menu_select')):
        got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
        assert got == roles[job] and 'real' in registry.get(got).tags, (text, got)
    assert brief.rank(brief.read('a tonal tap'), idx=idx, top=1)[0]['id'] == 'tap/tone'
    assert brief.rank(brief.read('a two-note next step chime'), idx=idx, top=1)[0]['id'] == 'next/step'
    for text in ('sword on armour', 'a heavy impact', 'a soft wooden success chime', 'thin swish', 'water droplet',
                 'a crisp mechanical click-clack like an old SLR'):
        got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
        assert registry.get(got).fn.__module__ != 'earshot.sounds.learned', (text, got)   # no single recording
    for text, want in (('footsteps on gravel', 'step/'), ('a wooden drawer', 'drawer/wood'),
                       ('camera shutter', 'camera/shutter'), ('a mechanical keyboard', 'keyboard/mechanical')):
        assert brief.rank(brief.read(text), idx=idx, top=1)[0]['id'].startswith(want), text


def test_unknown_word_does_not_decide(idx):
    r = brief.rank(brief.read('a zorblax whoosh'), idx=idx, top=3)
    assert 'whoosh' in r[0]['why']


def test_retro_is_a_hard_filter(idx):
    r = brief.rank(brief.read('8-bit coin'), idx=idx, top=8)
    assert r[0]['id'] == 'coin/chip'
    chips = [x['id'].endswith('/chip') for x in r]
    assert chips == sorted(chips, reverse=True)           # no other design above a chip design
    assert not brief.rank(brief.read('coin pickup'), idx=idx, top=1)[0]['id'].endswith('/chip')


def test_fallback_by_feel(idx):
    assert brief.rank(brief.read('something soft and warm'), idx=idx, top=3)


def test_fallback_by_direction(idx):
    b = brief.read('something that rises and rises')
    assert not b['concepts'] and b['dir'] == 1
    assert brief.rank(b, idx=idx, top=1)[0]['id'].startswith(('rise/', 'riser/', 'pro_riser/', 'sweep/', 'sub/',
                                                                    'braam/'))


def test_length_biases_the_ranking(idx):
    """a ceiling or a length pushes down designs much longer (or shorter) than asked; beds are exempt"""
    rows = {r['id']: r for r in idx[0]}
    short = brief.read('whoosh for a card sliding in, under 300 ms')
    assert short['dur_max'] == pytest.approx(0.3)
    first = brief.rank(short, idx=idx, top=1)[0]['id']
    assert rows[first]['dur_s'] <= 0.9, first
    assert brief.length_fit(short, rows['whoosh/pass']) < 0.8
    assert brief.length_fit(brief.read('rain, 20 seconds'), rows['rain/window']) == 1.0
    assert brief.length_fit(brief.read('a riser, 2 seconds'), {'dur_s': 2.5, 'kind': 'rise'}) == 1.0
    assert brief.length_fit(brief.read('a tap, 2 seconds'), {'dur_s': 0.06, 'kind': 'tap'}) < 0.7


# ---- reading: round-two classes

def test_separable_verbs():
    assert 'pickup' in brief.read('picking something up in a game')['concepts']
    assert 'close' in brief.read('putting a window away, soft')['concepts']
    assert 'hover' in brief.read('when the mouse floats over a link')['concepts']


def test_old_device_means_chip_sound():
    assert brief.read('the jingle from old Nintendo games')['material'] == 'retro'
    assert brief.read('a beep like an ancient home computer')['material'] == 'retro'
    assert brief.read('an old wooden door')['material'] == 'wood'


def test_outcome_words():
    assert brief.read('upload finished successfully')['concepts'][0] == 'success'
    assert brief.read('erasing a saved recipe')['concepts'][0] == 'delete'
    assert brief.read('a massive kaboom, like fireworks gone wrong')['concepts'][0] == 'explosion'


def test_composed_jobs():
    assert brief.read('losing the speed buff')['concepts'][0] == 'power_down'
    assert 'success' not in brief.read('a ring that says the egg timer is done')['concepts']


def test_more_negations_and_weather():
    assert 'jump' in brief.read("a gentle ping that won't make people jump")['not']
    b = brief.read('a fast windy zip')
    assert 'wind' in b['parts'] and 'wind' not in b['concepts']


def test_similes_and_domain_phrases():
    b = brief.read('a huge thump, like a door slamming in a movie')
    assert 'door' in b['parts'] and 'door' not in b['concepts']
    assert 'paper' not in brief.read('an impact under the title card')['context']
    assert 'paper' not in brief.read('a pop when a card appears in the app')['concepts']
    assert 'drone' == brief.read('an ominous low bed for a documentary')['concepts'][0]


def test_instrument_before_a_sound_noun():
    assert 'voice:kalimba' in brief.read('a kalimba ping for a meditation app')['concepts']
    assert 'voice:marimba' in brief.read('one marimba hit')['concepts']


def _sayable(word, d):
    """a member name a person would say: a lexicon word or a word of the design's own description"""
    tables = (L.MODIFIERS, L.MATERIALS, L.VOICES, L.CONCEPTS)
    said = {w for s in d.says for w in brief.words(s)}
    syn = {w for v in L.CONCEPTS.values() for p in v for w in p.split()}
    return word.isalpha() and len(word) > 2 and (any(word in t for t in tables) or word in said or word in syn)


def test_a_member_named_in_the_request_is_the_one_chosen():
    """"a warm pad chord" is pad/warm, though "warm" is an adjective: for every family of several members, the
    request "<member name> <family word>" ranks that member first"""
    from earshot.sounds import registry
    fams = {}
    for sid in registry.ids():
        fams.setdefault(sid.split('/')[0], []).append(sid)
    cases = [(f"{sid.split('/')[1]} {fam.replace('_', ' ')}", sid) for fam, ids in fams.items() if len(ids) > 1
             and all(_sayable(p, registry.get(ids[0])) or p in L.CONCEPTS for p in fam.split('_'))
             for sid in ids if _sayable(sid.split('/')[1], registry.get(sid))]
    assert len(cases) > 60
    terms = {r['id']: t for r, t in zip(*brief._default_index()[:2])}

    def twin(got, sid):
        """the member of the same name in a family that answers the same job ("deep impact": boom/deep, the film
        family's, ranks above the bank's impact/deep)"""
        return (got.split('/')[1] == sid.split('/')[1] and got.split('/')[0] != sid.split('/')[0]
                and terms[sid]['family'] & (terms[got]['family'] | terms[got]['uses']))
    wrong = [(text, sid, [r['id'] for r in brief.rank(brief.read(text), top=1)]) for text, sid in cases]
    def learned(got, sid):
        """a learned member of the same family that says what the name says ("grass step": a recorded footstep on
        grass, earshot.sounds.learned, preferred to the invented step/grass by brief._preferred), or the learned twin
        of the named design ("door knock": door/knock for knock/door, lexicon.TWINS)"""
        name, got_name = set(sid.split('/')[1].split('_')), set(got.split('/')[1].split('_'))
        same = got.split('/')[0] == sid.split('/')[0] or name <= got_name          # "drop drip": a water drop
        return (same or L.TWINS.get(sid) == got) and 'real' in registry.get(got).tags
    wrong = [w for w in wrong if w[2] != [w[1]] and not (w[2] and (twin(w[2][0], w[1]) or learned(w[2][0], w[1])))]
    assert not wrong, wrong
    assert brief.rank(brief.read('a deep impact'), top=1)[0]['id'] == 'boom/deep'
    assert brief.rank(brief.read('a metal impact'), top=1)[0]['id'] == 'impact/metal'      # no film member 'metal'
    assert brief.rank(brief.read('a warm pad chord in F'), top=1)[0]['id'] == 'pad/warm'
    b = brief.read('a warm pad chord in F')
    assert b['axes'].get('bright', 0) < 0                                         # warm still darkens


def test_why_shows_only_the_requests_own_words():
    """`[table, coin_object, coin]`, `[glass_object, bell]`, `[level_up, win, power_up]` showed internal names"""
    for text in ('coins on a table', 'glasses clinking', 'level up', 'a big win, level up!', 'a warm pad chord'):
        b = brief.read(text)
        for r in brief.rank(b, top=3):
            shown = brief.shown(b, r['why'])
            assert shown and not any('_' in w or ':' in w for w in shown), (text, shown)
            words_ = ' '.join(shown).split()
            assert len(words_) == len(set(words_)), (text, shown)                 # a word once
            assert all(w in text.lower() for w in shown if not w.startswith('sounds like')), (text, shown)
    assert brief.shown(brief.read('level up'), ['level_up', 'power_up']) == ['level up']


def test_job_phrases_match_inflected_and_reordered():
    for text in ('a page turning', 'turning a page', 'pages turn', 'turned the page'):     # the recorded page
        got = brief.rank(brief.read(text), top=1)[0]['id']                                 # turn (learned) or
        assert got in ('paper/page', 'page/turn'), text                                    # the invented one
    assert brief.stem('levelling') == brief.stem('leveled') == brief.stem('level')
    assert brief.stem('spelling') == 'spell'
    for text, fam in (('level up', 'level_up'), ('levelling up', 'level_up'), ('leveled up', 'level_up'),
                      ('power up', 'power_up'), ('powering up', 'power_up'), ('game over', 'lose'),
                      ('pick up', 'pickup'), ('picking it up', 'pickup')):
        got = brief.rank(brief.read(text), top=1)[0]['id']
        assert got.split('/')[0] in (fam, 'coin') if fam == 'pickup' else got.startswith(fam + '/'), (text, got)


def test_a_job_of_several_words_counts_only_as_a_whole():
    """a multi-word `uses` phrase is found by the whole phrase; one of its words alone counts only as a tag"""
    rows = [dict(id='a/x', kind='chime', says=['a plain chime', 'a small chime'], uses=['order ready'], tags=[],
                 domains=['ui'], dur_s=0.5, loop=False),
            dict(id='b/y', kind='chime', says=['a plain chime', 'a small chime'], uses=['ready'], tags=[],
                 domains=['ui'], dur_s=0.5, loop=False)]
    ix = brief.index(rows)
    terms = ix[1][0]
    assert 'ph:order ready' in terms['uses'] and 'ready' not in terms['uses'] and 'ready' in terms['tags']
    assert brief.rank(brief.read('ready'), idx=ix)[0]['id'] == 'b/y'            # "ready" alone: the one-word job
    top = brief.rank(brief.read('order ready'), idx=ix)[0]
    assert top['id'] == 'a/x' and 'ph:order ready' in top['why']
    b = brief.read('your order is ready, a ding')
    assert 'ph:order ready' in b['phrases'] and brief.shown(b, ['ph:order ready']) == ['order ready']
    assert 'ph:order ding' not in b['phrases']                                   # never across a clause
    # the request that found it: "day" no longer reaches a rain loop through a phrase
    assert brief.rank(brief.read('forest ambience by day with birds'))[0]['id'].startswith('forest/')


# ---- what a user sees: words as written, and what a pick leaves out

def test_why_shows_words_as_written_never_stems():
    """usability log: a stemmed 'lake' showed as `[lak, water]` next to `not understood: lake`"""
    b = brief.read('gentle water, small waves on a lake shore, calm')
    assert b['written']['lak'] == 'lake' and b['written']['water'] == 'water'
    assert brief.shown(b, ['lak', 'water', 'waves', 'voice:kalimba', 'lak']) == ['lake', 'water', 'waves', 'kalimba']
    b = brief.read('calm water at evening, gentle lapping waves')
    assert brief.shown(b, ['lap', 'water']) == ['lapping', 'water']
    assert brief.read('a short calm logo sound, a water drop into a warm chime')['written']['water'] == 'water drop'


def test_uncovered_names_the_sounds_a_pick_leaves_out():
    b = brief.read('a short calm logo sound, a water drop into a warm chime')
    assert brief.uncovered(b, ['logo', 'bell']) == ['water drop']
    assert brief.uncovered(b, ['logo', 'water', 'bell']) == []
    b = brief.read('calm water at evening, gentle lapping waves, quiet ambience')
    assert brief.uncovered(b, ['lap', 'water', 'waves', 'ambience']) == ['evening']   # evening is the night scene
    # axes, materials, numbers and context never count; nor the event, a comparison or the shape of a short sound
    assert brief.uncovered(brief.read('a soft wooden tap, 80 ms, for a logo reveal'), ['tap']) == []
    assert brief.uncovered(brief.read('sound when the player grabs a speed boost'), ['power_up']) == []
    assert brief.uncovered(brief.read('a huge thump, like a door slamming'), ['impact']) == []
    assert brief.uncovered(brief.read("a calm 'you're saved here' ping"), ['success']) == []
    # a job covered by its broader word: a swish by a whoosh
    b = brief.read('a short airy swish')
    assert 'swish' in b['concepts'] and brief.uncovered(b, [L.BROADER['swish']]) == []


# anchor requests (own wording, not from any eval set) -> accepted id prefixes of the first result
ANCHORS = [
    # interface
    ('a soft confirmation ding after saving a draft', ('success/',)),
    ('error tone for a rejected credit card', ('error/', 'ui/error')),
    ('incoming chat message', ('receive/', 'notify/')),
    ('checkbox switched on', ('toggle/', 'switch/')),
    ('turning a setting off', ('toggle/', 'switch/')),
    ('page swipe in a photo gallery', ('swipe/', 'swish/', 'paper/', 'slide/', 'flick/')),
    ('keyboard typing clicks for a chat box', ('typing_key/', 'pro_typing_key/', 'text/', 'keyboard/')),
    ('moving a file to the trash', ('delete/', 'paper/')),
    ('hovering over a navigation link', ('hover/', 'hover_tick/', 'pro_hover_tick/')),
    ('message sent', ('send/',)),
    ('a warning chime when the battery is low', ('warning/',)),
    ('welcome sound when the app boots', ('startup/', 'logo/')),
    ('camera click when taking a screenshot', ('shutter/', 'camera/shutter')),
    ('tapping the heart icon on a post', ('like/',)),
    # game
    ('8-bit coin collect', ('coin/chip',)),
    ('a character bouncing on a trampoline', ('jump/',)),
    ('footsteps on gravel', ('step/gravel',)),
    ('a big explosion with flying debris', ('explosion/',)),
    ('blaster shot for a space shooter', ('laser/', 'zap/')),
    ('sword slash', ('swing/', 'swish/')),
    ('a creaky wooden door opening in a haunted house', ('door/',)),
    ('a window shattering', ('break/glass',)),
    ('drinking a health potion', ('heal/',)),
    ('game over jingle', ('lose/',)),
    ('level complete fanfare', ('win/', 'level_up/')),
    ('opening a loot chest', ('chest/',)),
    ('a wizard casting a spell', ('magic/',)),
    ('the hero teleports away', ('teleport/',)),
    ('retro power up', ('power_up/chip',)),
    ('player takes damage from a spike', ('hurt/', 'hit/')),
    ('diving into a lake', ('splash/',)),
    # motion
    ('a fast whoosh for a title transition', ('whoosh', 'swish/')),
    ('tension riser before the chorus hits', ('rise/', 'riser/', 'pro_riser/', 'sub/swell', 'braam/swell')),
    ('a deep trailer impact under the title card', ('boom/', 'impact/', 'pro_impact/', 'braam/', 'cut_hit/')),
    ('logo reveal sting', ('sting/', 'logo/', 'logo_sting/', 'pro_logo_sting/')),
    ('digital glitch for a cyberpunk edit', ('glitch/', 'pro_glitch/')),
    ('a heavy bass drop', ('fall/sub', 'sub_drop/', 'pro_sub_drop/')),
    ('reverse cymbal into the cut', ('rise/reverse', 'reverse_swell/', 'cymbal/reverse')),
    ('fairy dust sparkle over a product shot', ('shimmer/', 'pro_shimmer/', 'sparkle/', 'gem/')),
    # ambience
    ('gentle rain on a tent, looping', ('rain/tent',)),
    ('forest ambience by day with birds', ('forest/',)),        # game log: 'rainy day' once won it by "day"
    ('ocean waves on a beach', ('waves/sand',)),                # a beach without another word is sand
    ('a warm pad chord in F', ('pad/warm',)),
    ('a page turning', ('paper/page', 'page/turn')),
    ('crickets on a summer night', ('night/',)),
    ('waves rolling over pebbles', ('waves/pebbles',)),
    ('busy city traffic in the distance', ('city/',)),
    ('spaceship engine room hum', ('space_hum/',)),
    # instruments
    ('a single marimba note in G', ('note/marimba', 'mallet/marimba')),
    ('music box melody', ('note/music_box',)),
    ('harp pluck, one note', ('note/harp',)),
    ('a deep sub bass note', ('note/sub_bass', 'note/round_bass', 'note/upright_bass', 'note/pluck_bass')),
    ('soft felt piano key', ('note/felt_piano',)),
    # objects
    ('a coin dropped on a wooden table', ('coin/drop', 'coin/flip', 'coins/')),
    ('two glasses clinking for a toast', ('glass/',)),
    ('knock knock on the door', ('knock/',)),
    ('popping a champagne cork', ('cork/',)),
    ('ringing the service bell at a hotel desk', ('bell/desk',)),
    ('a ticking wall clock', ('clock/',)),
    ('flipping a light switch', ('switch/', 'toggle/')),
]


def _learned_twin(sid, prefixes):
    """a learned design (earshot.sounds.learned: a recording played as new takes) of the family an anchor names:
    where it is the same thing, brief prefers it to the invented one (brief._preferred), and the learned twin of an
    invented design an anchor names (lexicon.TWINS: knock/door is door/knock)"""
    from earshot.sounds import registry
    from earshot.sounds.lexicon import TWINS
    return 'real' in registry.get(sid).tags and (any(sid.split('/')[0] == p.split('/')[0] for p in prefixes)
                                                 or any(i.startswith(prefixes) and t == sid for i, t in TWINS.items()))


@pytest.mark.parametrize('text,prefixes', ANCHORS)
def test_anchor(idx, text, prefixes):
    r = brief.rank(brief.read(text), idx=idx, top=3)
    assert r and (r[0]['id'].startswith(prefixes) or _learned_twin(r[0]['id'], prefixes)), (text, r)


# anchors for the families added after round two (motion, drums, music beds): own wording, plain and casual
NEW_ANCHORS = [
    # motion
    ('a whoosh as the logo flies in', ('whoosh/', 'swoosh/')),
    ('swoosh out when the card leaves the screen', ('swoosh/out',)),
    ('quick flick for a text change', ('flick/',)),
    ('a riser before the drop', ('rise/',)),
    ('a fast whoosh', ('whoosh/fast',)),
    ('a cinematic boom', ('boom/',)),
    ('a title slam', ('slam/title',)),
    ('noise riser into the reveal', ('rise/air',)),
    ('downlifter after the big hit', ('fall/',)),
    ('a cinematic boom for the title', ('boom/',)),
    ('a deep impact', ('boom/deep',)),
    ('title slam', ('slam/',)),
    ('a soft thump as the card lands', ('thump/', 'knock/soft')),
    ('sub drop right after the cut', ('fall/sub', 'sub_drop/')),
    ('reverse cymbal', ('rise/reverse', 'reverse_swell/', 'cymbal/reverse')),
    ('a cymbal played backwards into the cut', ('rise/reverse', 'reverse_swell/', 'cymbal/reverse')),
    ('a quick suck in before the cut', ('suck/',)),
    ('a soft pop when the icon appears', ('pop/',)),
    ('bubble pop', ('pop/bubble',)),
    ('a wooden tock when the tile snaps into place', ('tock/wood', 'knock/soft')),
    ('a glint on the logo', ('shine/glint',)),
    ('shimmer sweeping across the screen', ('shine/sweep',)),
    ('sparkle burst', ('sparkle/burst',)),
    ('a short glitch', ('glitch/',)),
    ('soft typing under kinetic text', ('type/',)),
    ('logo sting', ('sting/', 'logo/')),
    ('a tech sting for an app reveal', ('sting/', 'logo/', 'logo_sting/')),
    ('a panel sliding in', ('slide/panel',)),
    # film sounds from real material
    ('braam', ('braam/brass',)),
    ('trailer braam', ('braam/brass',)),
    ('impact with a suck-in', ('impact/suck',)),
    ('fly-by', ('pass/',)),
    ('a car passing by', ('pass/car',)),
    ('thunder', ('thunder/',)),
    ('drum roll build up', ('roll/', 'drum/roll', 'rise/tension')),
    ('fire whoosh', ('fire/whoosh',)),
    ('long whoosh', ('whoosh/long',)),
    ('timpani hit', ('timpani/hit',)),
    # a size word inside the name of a thing
    ('light switch', ('switch/wall',)),
    ('flip the light switch', ('switch/wall',)),
    ('small change', ('coin/', 'coins/')),
    # a learned design that is the thing asked for wins over its invented twin (lexicon.TWINS)
    ('knocking on a wooden door', ('door/knock',)),
    ('coins dropped on a table', ('coins/floor',)),
    ('a handful of coins', ('coins/floor',)),
    ('a coin dropped on the table', ('coin/drop',)),
    ('scissors cutting paper', ('scissors/cut',)),
    ('paper tearing', ('paper/rip',)),
    ('a distant explosion', ('explosion/far',)),
    # drums
    ('a punchy kick drum', ('kick/',)),
    ('kick for a hip hop track', ('kick/',)),
    ('a crisp snare for the backbeat', ('snare/',)),
    ('rimshot', ('rim/shot',)),
    ('cross stick on the snare', ('rim/click',)),
    ('closed hi-hat', ('hat/closed',)),
    ('open hihat', ('hat/open',)),
    ('a crash cymbal at the end of the fill', ('crash/',)),
    ('crash on the drums', ('crash/',)),
    ('a cymbal crash for a film trailer', ('cymbal/orchestral',)),
    ('ride cymbal ping', ('ride/ping',)),
    ('floor tom', ('tom/floor',)),
    ('a drum fill on the toms', ('tom/',)),
    ('hand clap', ('clap/',)),
    ('people clapping together', ('clap/group',)),
    ('snapping fingers', ('snap/finger',)),
    ('egg shaker', ('shaker/',)),
    ('tambourine', ('tambourine/',)),
    ('conga slap', ('conga/',)),
    ('a darbuka', ('darbuka/',)),
    ('doumbek', ('darbuka/',)),
    ('a bodhran', ('frame_drum/',)),
    ('afuche', ('cabasa/',)),
    ('slap stick', ('slapstick/',)),
    ('bongos', ('bongo/',)),
    ('cajon', ('cajon/',)),
    ('claves', ('clave/',)),
    ('drumroll before the winner is announced', ('drum/roll',)),
    ('the drummer counting in with the sticks', ('sticks/count',)),
    ('an 808 kick', ('kick/electro',)),
    ('drum machine clap', ('clap/electro',)),
    ('a drum hit', ('tom/', 'kick/', 'snare/')),
    # music beds
    ('some background music for my app demo', ('bed/',)),
    ('a music bed under a product video', ('bed/',)),
    ('chill lo-fi beat to study to', ('bed/warm',)),
    ('a seamless loop for my explainer video', ('bed/',)),
    ('menu music for a casual mobile game', ('bed/playful',)),
    ('chiptune level music', ('bed/8bit',)),
    ('soundtrack for a cozy farming game', ('bed/',)),
    ('upbeat corporate music', ('bed/',)),
    ('calm background music for a meditation app', ('bed/',)),
    ('a jingle', ('bed/',)),
    ('a beat', ('bed/',)),
    ('music', ('bed/',)),
    # and what music words must not take
    ('a single piano note', ('note/felt_piano', 'note/grand_piano')),
    ('a warm synth pad', ('pad/',)),
    ('rain in the background', ('rain/',)),
    ('warm sustained chords under a voice-over, not music', ('pad/',)),
]


@pytest.mark.parametrize('text,prefixes', NEW_ANCHORS)
def test_new_family_anchor(idx, text, prefixes):
    r = brief.rank(brief.read(text), idx=idx, top=3)
    assert r and (r[0]['id'].startswith(prefixes) or _learned_twin(r[0]['id'], prefixes)), (text, r)


# the thing a person does several times in a row: the design it finds, and whether the request asks for the action
# (True), a number of events (an int) or one event (False: a bare singular, "a single ...")
PERFORM_ANCHORS = [
    ('footsteps on gravel', 'step/pebbles', True),
    ('someone walking on a wooden floor', 'step/floorboard', True),
    ('a footstep on gravel', 'step/pebbles', False),
    ('a single footstep on concrete', 'step/concrete', False),
    ('two steps on a wooden floor', 'step/floorboard', 2),
    ('knocking on a wooden door', 'door/knock', True),
    ('a knock on the door', 'door/knock', True),
    ('a single knock on a door', 'door/knock', False),
    ('typing on a keyboard', 'keyboard/mechanical', True),
    ('three knocks on a door', 'door/knock', 3),
    ('typing on a mechanical keyboard', 'keyboard/mechanical', True),
    ('a bunch of keys jingling', 'keys/bunch', True),
    ('coins dropped on a table', 'coins/floor', True),
    ('glasses clinking', 'glass/cheers', True),
    ('ice cubes dropped into a glass', 'ice/cubes', True),
]


@pytest.mark.parametrize('text,sid,want', PERFORM_ANCHORS)
def test_perform_anchor(idx, text, sid, want):
    from earshot.make import performance
    b = brief.read(text)
    r = brief.rank(b, idx=idx, top=3)
    assert r and r[0]['id'] == sid, (text, r)
    perform, events = performance(b, sid)
    assert perform == bool(want) and (events == want if want not in (True, False) else True), (text, perform, events)
