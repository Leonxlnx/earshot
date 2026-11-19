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


def test_where_it_plays_names_no_sound_once_one_is_named():
    """the words of a place or product it plays in ("my running app", "a piano lessons website") say where, not
    what, when the request names its sound elsewhere; with nothing else named they count as before"""
    b = brief.read('in my running app, a success sound')
    assert b['concepts'] == ['success'] and 'footstep' not in b['context']
    b = brief.read('piano lessons website hover sound')
    assert b['concepts'] == ['hover'] and b['voice'] is None
    assert 'footstep' in brief.read('a sound for my running app')['concepts']


@pytest.mark.parametrize('text,want', [
    ('in my running app, a success sound', 'success/'),
    ("the dog walking app's error sound", 'error/'),
    ('piano lessons website hover sound', ('hover/', 'hover_tick/')),
    ('the Thunder app needs a notification sound', 'notify/'),
    ('weather dashboard rain widget click', ('switch/', 'click/', 'pen/click', 'tap/', 'keyboard/')),
    ('thunder sound for my weather app', 'thunder/'),
    ('dog bark for my pet store website', 'dog/'),
    ('a rain ambience for my sleep app', 'rain/'),
    ('a bell for a shop door', 'bell/shop'),
    ('footsteps for a horror game', 'step/'),
])
def test_the_thing_asked_for_wins_over_where_it_plays(idx, text, want):
    """an asked job or sound wins over a sounding word in the app, site or video it is for; a request whose place
    is the thing itself or shapes it ("for my weather app", "for a horror game") keeps it"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


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
    """a plain tap, click, hover, toggle, next or key press is the real sound chosen for that job (the sound of a
    real thing, lexicon.INTERFACE_JOBS, else the learned index's roles); the tonal figures answer when named or asked
    for as a tone; a recording answers only a request that names its thing"""
    from earshot.sounds import learned, registry
    roles = {r: sid for sid, e in learned.index().items() for r in e.get('roles', ())}
    for text, job in (('tap', 'tap'), ('a button press', 'tap'), ('click', 'click'), ('hover', 'hover'),
                      ('toggle on', 'switch_on'), ('toggle off', 'switch_off'), ('switch', 'toggle'),
                      ('next', 'next'), ('type', 'type'), ('key press', 'type'), ('select', 'menu_select')):
        got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
        want = L.INTERFACE_JOBS.get(job) or roles[job]
        assert got == want and 'real' in registry.get(got).tags, (text, got)
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
             for sid in ids if _sayable(sid.split('/')[1], registry.get(sid))
             and 'interface' not in registry.get(sid).tags]      # "door close" is a door: door/shut (brief._world)
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
    ('a soft pop when the icon appears', ('pop/', 'thump/appear')),
    ('bubble pop', ('pop/bubble',)),
    ('a wooden tock when the tile snaps into place', ('tock/wood', 'knock/soft')),
    ('a glint on the logo', ('shine/',)),
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
    ('four knocks on the table', ('knock/table',)),
    ('a knock on the desk', ('knock/table',)),
    ('a tap on the table', ('tap/fingertip',)),
    ('two taps', ('tap/fingertip',)),
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


# motion jobs get the sounds of real things motion films use (short low hits, thumps, ticks, passes of air); a design
# with a note in it answers only when the words ask for notes (brief.FILM_MUSIC) or the request is for a game
MOTION_REAL = [
    ('a logo reveal hit', ('logo/bag',)),
    ('a logo hit', ('logo/bag',)),
    ('a logo reveal', ('logo/bag',)),
    ('a title hit', ('title/bag',)),
    ('a cinematic boom', ('boom/deep',)),
    ('a pop when an element appears', ('thump/door',)),
    ('a soft ui tick', ('tick/key',)),
    ('a glitch transition', ('glitch/short',)),
    ('lens flare shimmer', ('shine/air',)),
    ('a logo reveal with a soft piano chord', ('logo/',)),
    ('a marimba logo jingle', ('logo/',)),
    ('a signature little tune for when the company name appears', ('logo/',)),
]


@pytest.mark.parametrize('text,prefixes', MOTION_REAL)
def test_motion_jobs_get_real_sounds_unless_notes_are_asked_for(idx, text, prefixes):
    r = brief.rank(brief.read(text), idx=idx, top=3)
    assert r and r[0]['id'].startswith(prefixes), (text, r)


def test_the_film_rule_steps_back_only_pitched_invented_designs():
    b = brief.read('a logo reveal hit')
    ids = [r['id'] for r in brief.rank(b, top=None)]
    assert ids.index('boom/short') < ids.index('logo/bloom')
    b = brief.read('a logo reveal hit for a platformer game')            # a game keeps its tonal stingers
    assert b['domain'] == 'game'


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


# ---- requests as agents and editors write them: one anchor per cause the reader got wrong

def test_accented_letters_stay_inside_a_word():
    """a German or French word is one word ("für", not "f" and "r"); a few common German words are synonyms"""
    assert brief.words('Klick für den Button') == ['klick', 'für', 'den', 'button']
    assert 'click' in brief.read('ein leiser Klick')['concepts']
    assert 'footstep' in brief.read('Schritte im Schnee')['concepts']
    assert 'coin' in brief.read('eine Münze fällt')['concepts']


def test_a_tape_is_not_a_tap(idx):
    assert brief.stem('tape') == 'tape' and brief.stem('tap') == 'tap'
    assert 'downlifter' in brief.read('tape stop')['concepts']
    assert 'downlifter' in brief.read('a tape-stop at the end of the song')['concepts']
    r = brief.rank(brief.read('packing tape ripped off a box'), idx=idx, top=3)
    assert r[0]['id'] in ('tape/peel', 'tape/roll'), r          # the invented tape or its recording (lexicon.TWINS)


def test_a_material_named_takes_the_recorded_role_away(idx):
    """"a tap" is the recorded key; "a wooden tap" is no plastic key"""
    assert brief.rank(brief.read('a tap'), idx=idx, top=1)[0]['id'].startswith('keyboard/')
    r = brief.rank(brief.read('a wooden tap, warm, no metal'), idx=idx, top=3)
    assert not r[0]['id'].startswith('keyboard/'), r


def test_a_number_of_events_reaches_the_twin_that_can_perform_them(idx):
    from earshot.make import performance, takes_asked
    b = brief.read('three knocks, slow')
    r = brief.rank(b, idx=idx, top=3)
    assert r[0]['id'] == 'door/knock' and performance(b, r[0]['id']) == (True, 3), r
    b = brief.read('five coin pickups in a row')       # a coin is not performed: five takes
    r = brief.rank(b, idx=idx, top=3)
    assert r[0]['id'].startswith('coin/') and takes_asked(b, r[0]['why'], performance(b, r[0]['id'])) == 5
    b = brief.read('level 2 complete jingle')           # the number counts no thing the design was picked for
    r = brief.rank(b, idx=idx, top=3)
    assert takes_asked(b, r[0]['why'], performance(b, r[0]['id'])) == 1


def test_a_countdown_with_a_length_keeps_its_count_and_a_rows_length_does_not_steer_the_pick(idx):
    """"3 2 1, 1 second": the countdown is 3 2 1; "a countdown of three beeps" is on the clock; the length of a
    counted action is the row's, so it ranks the designs as the count alone does"""
    b = brief.read('a countdown 3 2 1, 1 second')
    assert (b['events'], b['row'], b['dur']) == (3, 'count', 1.0)
    assert brief.read('a countdown of three beeps over 3 seconds')['row'] == 'count'
    a = brief.rank(brief.read('three claps'), idx=idx, top=3)
    b = brief.rank(brief.read('three claps, 3 seconds'), idx=idx, top=3)
    assert [x['id'] for x in a] == [x['id'] for x in b], (a, b)


@pytest.mark.parametrize('text,want', [
    ('clapping, 4 seconds', 'row'), ('someone clapping slowly', 'row'), ('knocking', 'knock'), ('beeping', 'row'),
    ('stepping', 'walk'), ('three claps', None), ('applause', None), ('a cue for tapping Continue', None),
    ('a single clear ding like tapping a little bell', None), ('the ready-set-go beeping before a race begins', None)])
def test_the_action_form_of_a_row_word_names_the_action_only_as_the_sound_itself(text, want, idx):
    assert brief.acted(brief.read(text)) == want
    if want == 'row':                       # the length is the row's: it ranks as the action word alone does
        a = brief.rank(brief.read(text.split(',')[0]), idx=idx, top=3)
        b = brief.rank(brief.read(text), idx=idx, top=3)
        assert [x['id'] for x in a] == [x['id'] for x in b]


@pytest.mark.parametrize('text,word', [
    ('a violin melody for the intro', 'violin'), ('someone saying welcome back', 'saying'),
    ('an electric guitar solo', 'electric guitar'), ('a woman shouting in a horror scene', 'shouting'),
])
def test_a_sound_the_catalogue_has_nothing_of_is_said_so(idx, text, word):
    """the pick of such a request is a guess, and the doubt names what is missing (or nothing is ranked at all)"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=8)
    if word is None:
        assert not cands or doubt(b, cands)
        return
    assert brief.not_made(b) == word
    assert not cands or doubt(b, cands).startswith(f'earshot makes no {word};'), doubt(b, cands)


@pytest.mark.parametrize('text', ['saying yes', 'a whoosh for a dog walking app', 'not a voice, a soft click',
                                  'glass clink, cheers', 'accordion expand sound'])
def test_nothing_missing_when_the_catalogue_has_it(text):
    assert brief.not_made(brief.read(text)) is None


@pytest.mark.parametrize('text,sid', [
    ('a dog barking', 'dog/bark'), ('a small dog yapping', 'dog/small'), ('a cat meowing', 'cat/meow'),
    ('a car horn honk', 'car_horn/honk'), ('a police siren', 'siren/pass'), ('applause', 'applause/audience'),
    ('a crowd cheering in a stadium', 'crowd/cheer'), ('a baby crying', 'baby/cry'),
    ('my phone vibrating on the desk', 'phone/vibrate'), ('soda can opening', 'can/open'),
    ('a toilet flush', 'toilet/flush'), ('a referee whistle', 'whistle/referee'),
])
def test_a_thing_recorded_is_answered_without_doubt(idx, text, sid):
    """animals, a crowd, a horn, a siren and household things are recordings (earshot.sounds.learned): the request
    is answered with the thing itself and nothing says it is missing"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=8)
    assert brief.not_made(b) is None and cands[0]['id'] == sid, [c['id'] for c in cands[:3]]
    assert not (doubt(b, cands) or '').startswith('earshot makes no'), doubt(b, cands)


def test_into_names_where_the_sound_lands(idx):
    r = brief.rank(brief.read('suck back reverse whoosh into the logo'), idx=idx, top=3)
    assert r[0]['id'].startswith(('suck/', 'reverse_swell/', 'whoosh', 'rise/reverse')), r


# ---- what people call things and jobs: one anchor per repaired cause, each with a second wording

@pytest.mark.parametrize('text,want', [
    ('punch cut hit', 'cut_hit/punch'), ('a cymbal reverse swell', 'reverse_swell/cymbal'),
])
def test_a_named_member_is_not_traded_for_a_plainer_namesake(idx, text, want):
    """a design named word for word keeps its family when the namesake's family is only a part of its own ("cut hit"
    names more than "hit": no fist on a body; a reverse swell is no forward one)"""
    assert brief.rank(brief.read(text), idx=idx, top=1)[0]['id'] == want


@pytest.mark.parametrize('text,want', [
    ('a bird chirping', 'chirp/'), ('a little bird singing on a branch', 'chirp/'),
    ('birds singing at dawn', 'forest/'),
    ('a telephone ringing', 'phone/ring'), ('the telephone rings twice', 'phone/ring'),
    ('a horn honking', 'car_horn/'), ('someone honks at a crossing', 'car_horn/'),
    ('a horn for the trailer title', 'braam/'), ('a gong', 'gong/'), ('an owl', 'owl/'),
    ('a newborn crying in the next room', 'baby/cry'), ('a light flicker on the neon sign', 'glitch/'),
    ('a flickering screen', 'glitch/'), ('a toast message slides in', 'notify/'),
    ('glasses clinking for a toast', 'glass/'), ('slider moving step by step', 'tick/'),
    ('a volume slider tick', 'tick/'), ('bass hit for the cut to black', ('boom/', 'sub_drop/', 'fall/sub')),
    ('a bass impact on the title', ('boom/', 'sub_drop/', 'fall/sub', 'pro_sub_drop/')),
])
def test_everyday_names_of_things_and_jobs(idx, text, want):
    """one bird is its call and birds are the forest; a telephone is a phone; a horn honks unless a film says it (a
    braam); a gong and an owl are their recordings; crying is a job; a flicker is a glitch; a toast in an app is a
    notification; a slider ticks; a film's bass hit is a sub hit"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text,want', [
    ('people clapping', 'applause/'), ('the whole room claps', 'applause/'),
    ('everyone clapping in unison', 'clap/'), ('people clapping together', 'clap/group'),
])
def test_many_people_clapping_is_an_audience(idx, text, want):
    assert brief.rank(brief.read(text), idx=idx, top=1)[0]['id'].startswith(want)


@pytest.mark.parametrize('text', ['a woman crying', 'a man weeping'])
def test_a_voice_of_another_person_is_not_made(idx, text):
    """the only crying is a baby's (lexicon.ONE_VOICE): a grown person's is the nearest sound, and said so"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=8)
    assert cands[0]['id'] == 'baby/cry' and doubt(b, cands).startswith('earshot makes no'), doubt(b, cands)


@pytest.mark.parametrize('text,want', [
    ('task complete', 'complete/'), ('checkout complete', 'complete/'), ('a success sound', 'success/shutter'),
    ('switch', 'toggle/key_down'),
])
def test_a_family_of_the_job_named_by_its_word_wins(idx, text, want):
    """the job's default (lexicon.INTERFACE_JOBS) is not raised over another interface family of the same job that
    the request names by its own word"""
    assert brief.rank(brief.read(text), idx=idx, top=1)[0]['id'].startswith(want)


@pytest.mark.parametrize('text,want', [
    ('a drop', 'hit/drop'), ('the drop after the build', 'hit/drop'), ('a big drop', 'hit/drop'),
    ('a heavy drop', 'heavy_drop/'), ('a bag drop', 'bag/drop'),
    ('item dropped into a list', 'land/'), ('a hit on the beat when the logo lands', 'hit/'),
    ('hit on the beat cut', 'hit/'),
])
def test_a_drop_and_a_beat_are_when_a_hit_lands(idx, text, want):
    """a drop with nothing else named is the film's hit (in an app: a soft landing; a thing a design drops named:
    that thing); "on the beat" asks for no music"""
    assert brief.rank(brief.read(text), idx=idx, top=1)[0]['id'].startswith(want)


def test_close_after_a_comma_is_near_and_times_count_events(idx):
    for text in ('footsteps on gravel, close', 'a knock on the door, close'):
        b = brief.read(text)
        assert b['axes'].get('space', 0) < 0 and 'close' not in b['concepts'], b
    assert brief.read('a dog barking twice')['events'] == 2
    b = brief.read('knock three times')
    assert b['events'] == 3 and brief.rank(b, idx=idx, top=1)[0]['id'] == 'door/knock'


@pytest.mark.parametrize('text,bad', [
    ('copied to clipboard', 'whoosh/'), ('a plate set down on the table', 'toggle/'), ('pull to refresh', None),
    ('pop for each icon as it comes on screen', 'pop/pluck'), ('a soft pop for each list item', 'pop/pluck'),
])
def test_stems_and_jobs_without_a_design(idx, text, bad):
    """'paste' is no 'past' and 'setting' no 'set'; a job no design plays reaches its gesture (a pull is a swipe, a
    copy a click); an app's pop is no note"""
    got = brief.rank(brief.read(text), idx=idx, top=1)
    assert got and (bad is None or not got[0]['id'].startswith(bad)), (text, got[:1])


@pytest.mark.parametrize('text,want', [('a knock', 'door/knock'), ('a punch', 'punch/body')])
def test_the_recording_answers_its_invented_twin_by_plain_name(idx, text, want):
    """lexicon.TWINS: the recorded thing gives way to nothing invented of the same thing, ranked or not"""
    assert brief.rank(brief.read(text), idx=idx, top=1)[0]['id'] == want


@pytest.mark.parametrize('text,want', [
    ('countdown before the video starts', 'countdown/'), ('a countdown for the intro of the film', 'countdown/'),
    ('unlocking a door with a key', 'lock/'), ('a key turning in a lock to unlock it', 'lock/'),
    ('toast sound for a wedding website', 'glass/'), ('the sound of a toast, cups going clink', 'glass/'),
    ('that photo-snap noise phones make when you take a picture', ('shutter/', 'camera/')),
])
def test_a_job_keeps_its_only_sounds_and_a_word_its_company(idx, text, want):
    """a film keeps the beeps of a countdown (no design without a note plays it); a recorded lock answers an
    unlocking with a key; a toast with glasses or a wedding is glasses; a phone that takes a picture is a shutter"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text,want', [
    ('when you pick up a file to drag it', ('tick/', 'tock/')),
    ('the big title lands on screen with real weight', ('boom/', 'thump/', 'slam/', 'hit/drop')),
    ('the logo settles into the middle at the end of the film', ('boom/short', 'sting/reveal', 'thump/')),
    ('a short papery crunch when an email is thrown away', ('delete/', 'paper/ball')),
    ('the camera zooms fast into the laptop screen', 'whoosh/'),
    ('a quick light swoosh as the message goes out', 'send/swoosh'),
    ('the bars of the chart grow up one after another', 'rise/air'),
    ('switching between tabs in the dashboard', 'next/'),
    ('the old screen flies off to the left', ('whoosh/', 'swoosh/')),
    ('a quick bright wipe to the next shot', ('whoosh/', 'swoosh/', 'swish/')),
    ('a pop when someone likes a photo in my cat app', ('like/pop', 'pop/')),
    ('thump when the card appears', 'thump/'),
    ('success chime for the checkout of a coffee shop website', 'success/rise'),
])
def test_a_described_moment_reaches_its_job_and_keeps_its_sound(idx, text, want):
    """a request that describes the moment on the screen without the job's name: a file picked up to drag it is a
    light tick, a title that lands or a logo that settles a low hit, bars that grow a riser, a camera zoom and a wipe
    passes of air; a sound word with a moment after it gets the job's real thing when that is the same kind of sound
    (a send is a swoosh, not a thump for a card appearing); a chime asked for keeps its notes"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text,want', [
    ('grabbing a card on the kanban board', ('tick/', 'tock/')),
    ('the photo clicks into place in the collage app', ('tick/', 'tock/')),
    ('the logo locks up at the end of the brand film', ('boom/short', 'sting/reveal', 'logo/bag')),
    ('the logo comes to rest in the centre of the frame in the ad', ('boom/short', 'sting/reveal', 'logo/bag')),
    ('the camera pushes in on the product in the commercial', 'whoosh/'),
    ('the camera pans across the mountains in the documentary', 'whoosh/'),
    ('a line draws on across the map in the video', ('flick/', 'swish/')),
    ('the card flips over in the animation', ('flick/', 'swish/')),
    ('the numbers count up to a million in the explainer video', ('tick/', 'clock/tick')),
    ('the price ticks up as the user adds extras in the app', ('tick/', 'clock/tick')),
])
def test_a_moment_word_reaches_its_kind_of_sound(idx, text, want):
    """the words people use for a moment on a screen: a thing grabbed or clicked into place is a light tick, a logo
    that locks up or comes to rest the logo's low hit, a camera move a pass of air, a line drawn on or a card flipped a
    short swish, numbers counting up ticks"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text', ['a thud as the logo appears at the end of the film',
                                  'a dull thud on the cut in the movie', 'thud for the title card in my video'])
def test_a_film_thud_is_no_drum_of_the_kit(idx, text):
    assert not any(r['id'].startswith(('kick/', 'tom/', 'snare/'))
                   for r in brief.rank(brief.read(text), idx=idx, top=8))


def test_a_film_request_that_names_the_kit_keeps_its_drum(idx):
    got = brief.rank(brief.read('a kick drum thud for the beat of the trailer music'), idx=idx, top=1)[0]['id']
    assert got.startswith('kick/'), got


@pytest.mark.parametrize('text,want', [
    ('a soft low thump when the pricing card settles', 'thump/'),
    ('a short low thump', 'thump/'),
    ('a card slides in from the right and settles, soft and short', ('slide/panel', 'thump/')),
    ('the panel settles into place', ('thump/', 'land/soft', 'slide/panel')),
    ('a modal settles into place', 'thump/'),
    ('the title settles in the middle of the frame in the video', 'thump/'),
    ('one careful step on creaky wooden planks', 'step/'),
    ('a cautious knock on the door', ('door/knock', 'knock/')),
    ('an enemy defeated sound', ('hit/', 'punch/')),
    ('an enemy is killed in a platformer', ('hit/', 'punch/')),
    ('a springboard throwing the player up into the air', 'jump/'),
    ('a small metallic clink when a key is picked up', ('keys/', 'key/')),
    ('a tiny bright tick each time a step number changes in the corner', 'tick/'),
])
def test_what_a_thing_does_on_a_page_in_a_game_or_a_film_is_its_own_sound(idx, text, want):
    """requests written for a landing page, a platformer and a product video: a low thump is a thump and no sub
    drop, a card or a panel that settles has landed softly (no falling sweep), "careful" and "cautious" before a
    thing say how it is done (no warning), one enemy beaten is a blow (no fanfare, no tune of losing), a springboard
    is a jump, the clink of a key is the keys', and what follows "each time" is where the sound plays (the step
    number is no footstep)"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text,want', [
    ("kind of a 'hey, careful' sound, more serious than a notification but not an error", 'warning/'),
    ('boss defeated', 'win/'), ('the player dies', 'lose/'),
    ('the dust settles after the explosion', ('fall/', 'downlifter/')),
    ('sound after the drop in a promo video as the energy settles, 3 seconds', ('fall/', 'downlifter/')),
    ('a toast slides up from the bottom', ('notify/', 'pop/', 'open/')), ('a heart beating', 'heart/'),
    ('two glasses clinking', 'glass/'),
])
def test_the_words_of_a_moment_keep_their_other_meanings(idx, text, want):
    """the same words where they are the job: "careful" alone is a warning, a boss defeated a win, the player's
    death a loss, energy or dust that settles winds down, a toast on a screen is a notification, a heart beats and
    glasses clink"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text,never', [
    ('a crunchy bite into toast', ('notify/', 'receive/')),
    ('a whisk beating eggs in a bowl', ('bed/',)),
])
def test_food_is_no_notification_and_no_music(idx, text, never):
    """toast that is bitten is bread and beating eggs is no beat: neither a notification nor a music loop answers"""
    b = brief.read(text)
    assert 'notify' not in b['concepts'] and 'music' not in b['concepts']
    assert not any(r['id'].startswith(never) for r in brief.rank(b, idx=idx, top=3))


def test_a_whisk_is_made():
    assert brief.not_made(brief.read('a whisk beating eggs in a bowl')) is None


@pytest.mark.parametrize('text,want', [('footsteps on concrete, 4 seconds', 'step/concrete'),
                                       ('footsteps on concrete', 'step/concrete'),
                                       ('footsteps on gravel, 5 seconds', ('step/pebbles', 'step/gravel'))])
def test_a_length_said_with_an_action_does_not_choose_the_surface(idx, text, want):
    """the seconds asked of a walk are the walk's: they do not pick the design whose single step is longest"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text', ['achievement unlocked sound', 'trophy unlocked in a racing game'])
def test_an_unlocked_achievement_is_no_lock(idx, text):
    assert not brief.rank(brief.read(text), idx=idx, top=1)[0]['id'].startswith(('lock/', 'keys/', 'bottle/'))


def test_stemmed_words_name_no_domain():
    """'cute' is no 'cut' (a film's edit): a cute sound asks for no film"""
    assert brief.read('a cute happy sound when you double-tap to love a post')['domain'] != 'motion'
    assert brief.read('a cute little pop')['domain'] != 'motion'


@pytest.mark.parametrize('text,want', [
    ('a copy to clipboard confirmation', 'success/shutter'),
    ('a subtle acknowledgement sound for a form', 'success/shutter'),
    ('a positive feedback sound for my app', 'success/shutter'),
    ('a soft approval sound', 'success/shutter'),
    ('a cue when the message is sent', 'send/swoosh'),
    ('a success chime', 'success/rise'),
    ('a success sound on a marimba', 'success/rise'),
    ('a level complete jingle for my game', 'win/fanfare'),
])
def test_an_interface_moment_gets_no_figure_unless_the_words_ask_for_one(idx, text, want):
    """a request whose own words name nothing a design exists for is the interface job of its moment and gets that
    job's real thing; a chime, an instrument by name and a game's jingle keep their notes"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got == want, (text, got)


@pytest.mark.parametrize('text,want', [
    ('a tap running', 'faucet/run'), ('running water from a tap', 'faucet/run'),
    ('a frog croaking', 'frog/croak'), ('glass breaking', 'glass/break'),
    ('a reload sound for my shooter game', 'gun/reload'),
    ('footsteps on stone', 'step/stone'), ('room tone at night', 'room_tone/'),
    ('steps through puddles', 'step/puddle'), ('chopping an onion', 'knife/chop'),
    ('water boiling in a pot', 'pot/boil'),
    ('turning on the gas burner', 'gas_stove/ignite'),
    # the same words where they mean what they did
    ('a tap on the screen', 'keyboard/'), ('pull to refresh', 'slide/'), ('crickets at night', 'night/crickets'),
    ('footsteps on gravel', 'step/pebbles'), ('a camera shutter', 'camera/shutter'),
])
def test_a_recorded_thing_is_reached_by_its_plain_request(idx, text, want):
    """a tap that runs is a faucet and no press, one frog is no night of crickets, glass that breaks is the recorded
    glass, a shooter's reload a gun's, a stone floor no gravel, "at night" says when, steps through puddles walk in
    them"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


def test_a_bee_is_the_recorded_bee(idx):
    from earshot.make import doubt
    b = brief.read('a bee buzzing')
    cands = brief.rank(b, idx=idx, top=4)
    assert cands[0]['id'] == 'bee/buzz' and doubt(b, cands) is None, cands[:2]


@pytest.mark.parametrize('text,want', [
    ('a tiger roar', 'lion/roar'), ('a dragon roars', 'lion/roar'), ('booing and jeering', 'crowd/boo'),
    ('a vinyl scratch', 'record/scratch'), ('a rubber duck squeak', 'toy/squeak'),
    ('the lift arrives', 'elevator/ding'),
    ('a bumblebee', 'bee/buzz'), ('a food processor', 'blender/run'), ('beating eggs', 'whisk/beat'),
    ('an elephant trumpet', 'elephant/trumpet'), ('footsteps through puddles', 'step/puddle'),
    ('a drawer opening', 'drawer/'),
])
def test_a_whole_job_of_a_recording_reaches_it_without_doubt(idx, text, want):
    """a recording is reached by a job it is named for said whole ("the lift arrives") or by a word of it that names
    nothing else ("a tiger roar"); a word of NOT_MADE inside such a job is that recording ("an elephant trumpet")"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=4)
    assert cands[0]['id'].startswith(want) and doubt(b, cands) is None, (text, cands[:2], doubt(b, cands))


@pytest.mark.parametrize('text,nearest', [
    ('a trumpet', 'win/fanfare'), ('a fire alarm', 'timer/done'), ('a toy gun', 'cap_gun/shot'),
    ('an airlock hissing open', 'door/slide'), ('a pan set on the stove', 'pot_lid/set'),
    ('machine gun burst, two seconds', 'gunshot/rifle'),
])
def test_a_thing_no_design_is_gets_its_nearest_and_says_so(idx, text, nearest):
    """a trumpet is no elephant, a fire alarm no fire, a toy gun no heavy shot and a mouse's squeak no mouse click:
    each is answered by its nearest design and the answer says earshot makes none"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=4)
    assert cands[0]['id'] == nearest and (doubt(b, cands) or '').startswith('earshot makes no '), (text, cands[:2])


def test_panning_words_are_read_and_not_unknown():
    b = brief.read('a car passing from left to right')
    assert b['pan'] == 'from left to right' and not {'left', 'right'} & set(b['unknown'])


@pytest.mark.parametrize('text,never', [
    ('chopping', 'pro_riser/'), ('the camera pushes into the product', ('camera/', 'shutter/')),
    ('a wake-up-ish ring that says your eggs are done boiling', 'pot/'),      # what ended is not the sound
    ('a wake-up-ish ring that says your eggs are done boiling', 'egg/'),      # what a ring says is no egg cracked
])
def test_a_word_of_an_id_or_of_another_trade_picks_no_design(idx, text, never):
    """chopping is the knife's, not the riser whose member is called chop; the camera of a film that moves takes no
    photo"""
    assert not any(r['id'].startswith(never) for r in brief.rank(brief.read(text), idx=idx, top=1))


@pytest.mark.parametrize('text,want', [
    ('a feedback sound for my app', 'notify/knock'),
    ('an alert when a delivery driver is nearby in a food ordering app', 'notify/knock'),
    ('a cue when the payment method is verified', 'success/shutter'),
    ('a purchase sound', 'success/shutter'), ('bike shop checkout purchase sound', 'success/shutter'),
    # the till by its own words, notes by theirs, and a game's shop keep their sounds
    ('a cash register', 'cash_register/open'), ('a purchase chime', 'purchase/chime'),
])
def test_a_cue_and_a_purchase_get_the_plain_sound_of_their_job(idx, text, want):
    """a request that only says there should be a cue is a notification's knocks; a purchase that went through in an
    app is confirmed by the success click, never by a chime or a bell the words did not ask for"""
    got = brief.rank(brief.read(text), idx=idx, top=1)[0]['id']
    assert got == want, (text, got)


def test_times_a_gap_and_hours():
    """"twice" is two events of one row, "200 ms apart" the gap between events and no length, hours are a length"""
    b = brief.read('a dog barking twice')
    assert (b['events'], b['row'], b['n']) == (2, 'row', None)
    assert brief.read('a knock two times')['row'] == 'knock'
    b = brief.read('two knocks 200 ms apart')
    assert b['events'] == 2 and b['gap'] == pytest.approx(0.2) and b['dur'] is None
    assert brief.read('three beeps 0.2 seconds apart, 2 seconds')['dur'] == pytest.approx(2.0)
    b = brief.read('rain, 10 hours')
    assert b['dur'] == pytest.approx(36000.0) and b['events'] is None
    b = brief.read('three knocks, 2 seconds')
    assert b['dur'] == pytest.approx(2.0) and b['gap'] is None
    b = brief.read('three soft thumps, 200 ms apart')
    assert (b['events'], b['row']) == (3, 'row') and b['gap'] == pytest.approx(0.2) and b['dur'] is None
    b = brief.read('three soft thumps over 0.6 seconds')
    assert (b['events'], b['row']) == (3, 'row') and b['dur'] == pytest.approx(0.6)


def test_a_request_in_another_language_is_no_chance_match(idx):
    """German "die" is no death: the request is marked as not English and nothing is picked for a look-alike; words
    the lexicon knows are still read, and English keeps its words"""
    b = brief.read('Erfolgston für die App')
    assert b['foreign'] and 'lose' not in b['concepts']
    assert brief.rank(b, idx=idx, top=1)[0]['id'] == 'success/shutter'
    b = brief.read('le son pour le bouton')         # no word found anything: a nearest sound, said to be a guess
    cands = brief.rank(b, idx=idx, top=1)
    assert [c['id'] for c in cands] == [L.NEAREST_ANY[0]] and cands[0]['why'] == []
    from earshot.make import doubt
    assert doubt(b, cands) == 'no job or sound word matched; the request does not read as English, write it in English'
    assert brief.read('Türklopfen, dreimal')['concepts'] == ['door', 'knock']
    b = brief.read('a sound when the player dies')
    assert not b['foreign'] and 'lose' in b['concepts']
    assert 'lose' in brief.read('die sound for a platformer')['concepts']


@pytest.mark.parametrize('text,absent,nearest', [
    ('a lightsaber', 'lightsaber', 'laser/beam'), ('a zorp squealing', 'zorp squealing', 'tyres/screech'),
    ('a gizmo whirring', 'gizmo whirring', 'drill/electric'), ('a blorb fizzing', 'blorb fizzing', 'spray_can/spray'),
    ('a cursed whisper', 'cursed whisper', 'sigh/person'), ('a zorp moaning', 'zorp moaning', 'sigh/person'),
    ('a zorp fluttering', 'zorp fluttering', 'wings/flap'),
    ('an old modem dialing', 'old modem dialing', 'dial/click'),
    ('a frobnicator', 'frobnicator', L.NEAREST_ANY[0]),
])
def test_an_unknown_thing_or_action_gets_the_nearest_and_says_so(idx, text, absent, nearest):
    """the thing that makes the sound, or what it does, is a word neither the lexicon nor a design's name or job
    knows: the request is answered with the nearest kind of sound (lexicon.SOUND_KINDS), else by its known words,
    else with the first of lexicon.NEAREST_ANY, and the doubt says it is only the nearest"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=4)
    assert b['absent'] == absent and cands[0]['id'] == nearest, (b['absent'], cands[:1])
    assert doubt(b, cands) == f'earshot makes no {absent}; {nearest} is only the nearest sound it has'


@pytest.mark.parametrize('text,want', [
    ('a ball bouncing on wood', 'ball/bounce'), ('a duck quacking on a lake', 'duck/quack'),
    ('snow crunching under boots', 'step/snow'), ('a man snoring loudly', 'snore/person'),
    ('a heartbeat getting faster', 'heart/beat'), ('a sword being drawn', 'sword/draw'),
    ('a camera shutter, three times', 'camera/shutter'), ('not a voice, a soft click', None),
    ("a little 'never mind' sound", None),
    ('a loud hollow cork pop followed by fizz', 'cork/'), ('when the user saves, a soft thwip', None),
    ('a jet flying overhead', 'pass/jet'), ('a plane passing overhead', 'pass/jet'),
])
def test_a_known_thing_is_not_said_to_be_missing(idx, text, want):
    """a thing a design is named for, a word that only describes or says whose it is, a caption in quotes, a
    negation and the event before the sound do not make a request one for a thing earshot lacks"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=4)
    assert want is None or cands[0]['id'].startswith(want), cands[:1]
    assert not (doubt(b, cands) or '').startswith('earshot makes no'), doubt(b, cands)


@pytest.mark.parametrize('text,want', [
    ('toggle dark mode', 'toggle/'), ('the heavy wooden front door falls shut', 'door/shut'),
    ('five coin pickups', 'coin/'), ('a riser before the drop', 'rise/'), ('a kitchen timer ringing', 'timer/done'),
])
def test_a_job_and_what_it_acts_on_which_one_a_count_and_a_name_of_the_job_are_no_guess(idx, text, want):
    """an interface job said with the thing it acts on, a word that says which one ("front", "kitchen"), a number
    read as the count, the riser's two families and a name the lexicon gives a job leave nothing missing"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=8)
    assert cands[0]['id'].startswith(want) and doubt(b, cands) is None, (cands[:2], doubt(b, cands))
    assert not {'five', 'front', 'kitchen'} & set(b['unknown']), b['unknown']


@pytest.mark.parametrize('text,said', [
    ('toggle a zorp machine', 'earshot makes no zorp machine'), ('a zorp door rolling up', 'earshot makes no zorp'),
    ('a front loader dumping gravel', 'earshot makes no front loader'), ('five zorp machines', 'earshot makes no zorp'),
    ('a kitchen mixer whirring', 'earshot makes no kitchen mixer'),
    ('the back door slams', 'another kind of sound fits almost as well'),
])
def test_the_neighbours_of_those_stay_honest(idx, text, said):
    """an unknown thing after a job, an owner that names another door, an unknown thing after "front" or "kitchen"
    or a count, and two families of different sounds that tie still say so"""
    from earshot.make import doubt
    b = brief.read(text)
    assert (doubt(b, brief.rank(b, idx=idx, top=8)) or '').startswith(said)


@pytest.mark.parametrize('text,want,material', [
    ('rain on wood', 'rain/', 'wood'), ('a ball bouncing on wood', 'ball/bounce', 'wood'),
    ('water dripping on metal', 'water/drop', 'metal'), ('footsteps in mud', 'step/mud', None),
    ('rain on a tarp', 'rain/tent', None), ('footsteps on snow', 'step/', None),
])
def test_what_follows_on_or_in_is_the_surface_not_another_thing(idx, text, want, material):
    """wood is no forest and metal no sizzling: the surface picks among the designs of the sound named before it,
    and a surface none of them has picks the nearest one (lexicon.SURFACE_KIN)"""
    b = brief.read(text)
    assert b['surface'] and b['material'] == material and 'forest' not in b['concepts'] + b['context']
    got = brief.rank(b, idx=idx, top=1)[0]['id']
    assert got.startswith(want), (text, got)


@pytest.mark.parametrize('text', ['a click on the submit button', 'a pop on hover', 'a chime on a piano',
                                  'sound for landing a jab in a 2D fighting game', 'toggle the lights on'])
def test_a_job_an_instrument_and_a_place_of_a_product_are_no_surface(text):
    assert brief.read(text)['surface'] is None


@pytest.mark.parametrize('text,english,want', [
    ('Regen auf Holz', 'rain on wood', 'rain/'), ('Schritte auf Kies', 'footsteps on gravel', 'step/'),
    ('ein Hund bellt', 'a dog barking', 'dog/bark'),
    ('Münzen fallen auf den Boden', 'coins falling on the floor', 'coins/floor'),
    ('ein leiser Klick', 'a quiet click', ''), ('schwere Schritte', 'heavy footsteps', 'step/'),
    ('Auto hupt', 'car honking', 'car_horn/'), ('Motor startet', 'engine starting', 'car_engine/'),
    ('Donner in der Ferne', 'thunder in the distance', 'thunder/'), ('Karten mischen', 'cards shuffling', 'cards/'),
    ('Tacker', 'stapler', 'stapler/'), ('jemand niest', 'someone sneeze', 'sneeze/'),
    ('eine knarrende Tür', 'a creak door', 'door/'), ('ein lauter Donner', 'a loud thunder', 'thunder/'),
])
def test_a_german_request_is_read_through_the_table(idx, text, english, want):
    """lexicon.GERMAN: the request is read as its English words, said to be foreign, and answered without a doubt"""
    from earshot.make import doubt
    b = brief.read(text)
    assert b['foreign'] and b['read_as'] == english == b['text'] and b['said'] == text
    cands = brief.rank(b, idx=idx, top=3)
    assert cands[0]['id'].startswith(want) and doubt(b, cands) is None, (cands[:1], doubt(b, cands))


def test_english_is_not_read_as_german():
    for text in ('a fallen tree hitting the ground', 'wind in the trees', 'a hammer on an anvil', 'a warm pad',
                 'die sound for a platformer', 'a laser blast'):
        assert brief.german(text) is None and not brief.read(text)['foreign'], text
    assert brief.german('Regen auf einem Blechdach, 10 Sekunden') == 'rain on a tin roof, 10 seconds'


@pytest.mark.parametrize('text', ['a blorf screeching in the jungle', 'a flock of zorps flying over',
                                  'a zeppelin arriving at the station', 'a zorp factory running',
                                  'a gizmo springing shut'])
def test_an_unknown_thing_doing_a_known_thing_is_said_to_be_missing(idx, text):
    """the thing that makes the sound is a word the reader does not know (a zeppelin; a word in -y such as
    "factory" is a thing, not a description): what it does is found, the thing is not, and the answer says so"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=8)
    assert (doubt(b, cands) or '').startswith('earshot makes no '), (text, cands[:1], doubt(b, cands))


@pytest.mark.parametrize('text,want', [('plates clattering', 'plate/'), ('twigs snapping underfoot', 'twigs/'),
                                       ('police siren getting closer', 'siren/')])
def test_a_known_thing_doing_an_unknown_thing_is_not_said_to_be_missing(idx, text, want):
    """the thing was found; only what it is doing is a word the reader does not know"""
    from earshot.make import doubt
    b = brief.read(text)
    cands = brief.rank(b, idx=idx, top=8)
    assert cands[0]['id'].startswith(want) and not (doubt(b, cands) or '').startswith('earshot makes no '), (
        text, cands[:1], doubt(b, cands))


def test_a_thing_the_cookbook_sketches_is_answered_by_a_part_of_its_sketch(idx):
    """a thing no design is and the cookbook only sketches (no kind of sound names it, it is in no list of things
    earshot lacks) is answered by a part its sketch is built on, of the family the request names first"""
    from earshot import recipe
    from earshot.make import doubt
    for name, rc in recipe.cookbook().items():
        if not rc.get('sketch'):
            continue
        b = brief.read(rc['things'][0])
        cands = brief.rank(b, idx=idx, top=4)
        listed = brief.not_made(b, cands[0]['why'], cands[0]['id']) in L.NOT_MADE | set(L.NOT_MADE_PHRASES)
        if b.get('kind') or listed or not (doubt(b, cands) or '').startswith('earshot makes no '):
            continue
        fams = {s.split('/')[0] for s in brief.sketched(b)}
        assert cands[0]['id'].split('/')[0] in fams, (name, cands[:1], brief.sketched(b))


@pytest.mark.parametrize('text,want,not_', [
    ('a car on gravel', 'step/gravel', 'hail/'),                       # 'car' names where the hail was heard
    ('a warm floaty background sound, just a calm hum of notes', 'pad/', 'fridge/'),
    ('hail on a car roof', 'hail/car', None), ('a refrigerator hum', 'fridge/hum', None),
    ('a zippy speed-burst noise for darting forward', 'dash/', 'dart/'),   # 'darting' is a dash, 'darts' the thing
    ('throwing darts', 'dart/hit', None),
    ('air blowing through a field, not stormy, just breezy', 'wind/', 'blow/'),
    ('blowing out birthday candles', 'blow/puff', None),
    ('that scary movie-trailer BRAAM, like a giant ship horn', 'braam/', 'ship_horn/'),
    ('a foghorn', 'ship_horn/', None),
    ('when you shut a popup window, sort of a gentle folding-away noise', None, 'window/'),
    ('putting a window away, kind of a soft tuck-in, nothing dramatic', None, 'window/'),
    ('opening a window', 'window/open', None), ('mice squeaking', 'mouse/squeak', None),
    ('watering the garden', 'hose/spray', None),
])
def test_a_thing_is_answered_by_its_own_words_not_by_a_word_it_shares(idx, text, want, not_):
    """a recording answers the words of its own thing: not a member name that is where it was heard ('hail/car'),
    a job only its name holds ('fridge/hum'), its family's word said as what something else does ("darting", "air
    blowing"), a comparison ("like a ship horn") or what a product's job is done to ("putting a window away")"""
    top = brief.rank(brief.read(text), idx=idx, top=3)[0]['id']
    assert (want is None or top.startswith(want)) and (not_ is None or not top.startswith(not_)), (text, top)
