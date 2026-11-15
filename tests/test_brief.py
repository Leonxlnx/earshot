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
    assert brief.rank(b, idx=idx, top=1)[0]['id'].startswith(('riser/', 'pro_riser/', 'sweep/', 'sub/', 'braam/'))


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
    ('page swipe in a photo gallery', ('swipe/', 'swish/', 'paper/')),
    ('keyboard typing clicks for a chat box', ('typing_key/', 'pro_typing_key/', 'text/')),
    ('moving a file to the trash', ('delete/', 'paper/')),
    ('hovering over a navigation link', ('hover/', 'hover_tick/', 'pro_hover_tick/')),
    ('message sent', ('send/',)),
    ('a warning chime when the battery is low', ('warning/',)),
    ('welcome sound when the app boots', ('startup/', 'logo/')),
    ('camera click when taking a screenshot', ('shutter/',)),
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
    ('tension riser before the chorus hits', ('riser/', 'pro_riser/', 'sub/swell', 'braam/swell')),
    ('a deep trailer impact under the title card', ('impact/', 'pro_impact/', 'braam/', 'cut_hit/')),
    ('logo reveal sting', ('logo/', 'logo_sting/', 'pro_logo_sting/')),
    ('digital glitch for a cyberpunk edit', ('glitch/', 'pro_glitch/')),
    ('a heavy bass drop', ('sub_drop/', 'pro_sub_drop/')),
    ('reverse cymbal into the cut', ('reverse_swell/',)),
    ('fairy dust sparkle over a product shot', ('shimmer/', 'pro_shimmer/', 'sparkle/', 'gem/')),
    # ambience
    ('gentle rain on a tent, looping', ('rain/tent',)),
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


@pytest.mark.parametrize('text,prefixes', ANCHORS)
def test_anchor(idx, text, prefixes):
    r = brief.rank(brief.read(text), idx=idx, top=3)
    assert r and r[0]['id'].startswith(prefixes), (text, r)
