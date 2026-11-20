"""Recipes: a sound built from the catalogue's parts by a small JSON form. Every mistake in a recipe is one plain
sentence that names the field; each layer control changes the sound as it says; a seed gives the same samples; the
result is checked like every sound and says it is a recipe; the ways in (Python, a set's entry, the command line,
the MCP tool) all render the same thing; a request that names a cookbook thing is answered by its recipe; every
cookbook recipe renders clean (slow tier)."""
import io
import json
import sys

import numpy as np
import pytest
import soundfile as sf

from earshot import cli, mcp_server, recipe as R
from earshot.forge import dsp
from earshot.make import make, make_set

TAP = {'sound': 'switch/power_strip'}
AIR = {'sound': 'swipe/hand'}


def rec(*layers, **top):
    return {'name': 'thing', 'layers': list(layers) or [dict(TAP)], **top}


def one(lay, seed=0):
    """a layer alone, before its gain and its place: stereo samples"""
    return R._one(R.check(rec(lay))['layers'][0], seed, 0, [])


def centroid(y):
    s = np.abs(np.fft.rfft(y.mean(axis=1)))
    return float((s * np.fft.rfftfreq(len(y), 1 / dsp.SR)).sum() / s.sum())


# ---------------------------------------------------------------------------------------------------- validation
@pytest.mark.parametrize('bad, says', [
    ([], 'a recipe is an object'),
    ({'layers': [TAP]}, 'name is the sound\'s name'),
    ({'name': 'two words', 'layers': [TAP]}, 'name is the sound\'s name'),
    ({'name': 'x'}, 'layers is a list of 1 to 12 layers'),
    ({'name': 'x', 'layers': []}, 'layers is a list of 1 to 12 layers'),
    ({'name': 'x', 'layers': [TAP] * 13}, 'layers is a list of 1 to 12 layers'),
    (rec(TAP, colour='red'), "unknown field 'colour' in the recipe: a recipe takes name, says, layers"),
    (rec(space='cave'), "space is one of dry, booth, room, studio, hall, plate, cathedral, not 'cave'"),
    (rec(kind='loop'), "kind is one of event, ambience, not 'loop'"),
    (rec(dur=0), 'dur is the length of the sound in seconds from 0.02 to 120, not 0'),
    (rec(says=''), 'says is what the sound is, in words'),
    (rec('door/slam'), 'layers[0] is an object such as'),
    (rec({}), 'layers[0] names either a catalogue design ("sound": "door/slam"), a plain source'),
    (rec({'sound': 'switch/power_strip', 'recipe': 'thunder_far'}), 'layers[0] names either a catalogue design'),
    (rec({'recipe': 'thunder_fat'}), "layers[0].recipe 'thunder_fat' is not in the cookbook; the nearest: "
                                     "thunder_far"),
    (rec({'recipe': 'thunder_far', 'size': 1}), 'layers[0].size does not belong to a recipe layer'),
    (rec({**TAP, 'scatter': 40}), 'layers[0].scatter is an object such as {"rate": 40, "dur": 6'),
    (rec({**TAP, 'scatter': {'rate': 40}}), 'layers[0].scatter.dur is missing'),
    (rec({**TAP, 'scatter': {'rate': 900, 'dur': 2}}), 'layers[0].scatter.rate is a number of events a second '
                                                        'from 0.5 to 400, not 900'),
    (rec({**TAP, 'scatter': {'rate': 9, 'dur': 2, 'grain': 0}}), 'layers[0].scatter.grain is a length in seconds '
                                                                  'from 0.005 to 2, not 0'),
    (rec({**TAP, 'scatter': {'rate': 9, 'dur': 2, 'spread': 1}}), "unknown field 'spread' in layers[0].scatter: "
                                                                   "scatter takes rate, rate_to, dur"),
    (rec({**TAP, 'scatter': {'rate': 9, 'dur': 2}, 'events': {'count': 2, 'gap': 1}}), 'layers[0] takes events '
                                                                                        '(a row of up to 64) or '
                                                                                        'scatter'),
    (rec({**TAP, 'level': -6}), 'layers[0].level is a list of two or more points [time s, dB] such as'),
    (rec({**TAP, 'level': [[0, 0]]}), 'layers[0].level is a list of two or more points'),
    (rec({**TAP, 'level': [[1, 0], [0.5, -6]]}), 'layers[0].level[1][0] is 0.5 s, before the point ahead of it'),
    (rec({**TAP, 'level': [[0, 0], [1, -90]]}), 'layers[0].level[1][1] is a level in dB from -60 to 12, not -90'),
    (rec({**TAP, 'bend': [[0, 0], [1, 30]]}), 'layers[0].bend[1][1] is a number of semitones from -24 to 12'),
    (rec({**TAP, 'pan': [-1, 0, 1]}), 'layers[0].pan is a position from -1 to 1 or two of them [from, to]'),
    (rec({**TAP, 'pan': [-1, 3]}), 'layers[0].pan[1] is a position from -1 to 1, not 3'),
    (rec({**TAP, 'pass': 2}), 'layers[0].pass is an object such as {"at": 2.0, "speed": 25}'),
    (rec({**TAP, 'pass': {'speed': 20}}), 'layers[0].pass.at is missing'),
    (rec({**TAP, 'pass': {'at': 1, 'speed': 400}}), 'layers[0].pass.speed is a speed in metres a second from 1 to '
                                                    '100, not 400'),
    (rec({**TAP, 'spin': {'depth': 1}}), 'layers[0].spin.rate is missing'),
    (rec({**TAP, 'spin': {'rate': 200}}), 'layers[0].spin.rate is a rate in Hz from 0.5 to 60, not 200'),
    (rec({**TAP, 'spin': {'rate': 9, 'shape': 'saw'}}), "layers[0].spin.shape is one of smooth, chop, not 'saw'"),
    (rec({**TAP, 'stretch': 10}), 'layers[0].stretch is a factor from 0.5 to 2, not 10: further than that a real '
                                  'take rings like a comb'),
    (rec({'sound': 'switch/power_strip', 'source': 'noise'}), 'layers[0] names either a catalogue design'),
    (rec({'sound': 'switch/power_stripe'}), "layers[0].sound 'switch/power_stripe' is not in the catalogue; the "
                                            "nearest ids: switch/power_strip"),
    (rec({**TAP, 'pitch': 40}), 'layers[0].pitch is a number of semitones from -24 to 12, not 40'),
    (rec(TAP, {**TAP, 'gain': 'loud'}), "layers[1].gain is a level in dB from -40 to 12, not 'loud'"),
    (rec({**TAP, 'size': 2}), 'layers[0].size is an amount from -1 to 1, not 2'),
    (rec({**TAP, 'reverse': 'yes'}), "layers[0].reverse is true or false, not 'yes'"),
    (rec({**TAP, 'cut': [0.2, 0.2]}), 'layers[0].cut is [from, to] with to at least 0.01 s after from'),
    (rec({**TAP, 'cut': 0.2}), 'layers[0].cut is two times in seconds [from, to], not 0.2'),
    (rec({**TAP, 'lowpass': 5}), 'layers[0].lowpass is a frequency in Hz from 20 to 20000, not 5'),
    (rec({**TAP, 'highpass': [100, 200, 300]}), 'layers[0].highpass is a frequency in Hz or two of them'),
    (rec({**TAP, 'loud': 1}), "unknown field 'loud' in layers[0]: a layer takes sound, source, recipe, seed"),
    (rec({**TAP, 'hz': 60}), 'layers[0].hz belongs to a source'),
    (rec({**TAP, 'events': 3}), 'layers[0].events is an object such as {"count": 4, "gap": 0.5}'),
    (rec({**TAP, 'events': {'count': 3}}), 'layers[0].events.gap is missing'),
    (rec({**TAP, 'events': {'count': 1, 'gap': 0.2}}), 'layers[0].events.count is a whole number from 2 to 64'),
    (rec({**TAP, 'events': {'count': 3, 'gap': 0.2, 'swing': 1}}), "unknown field 'swing' in layers[0].events"),
    (rec({'source': 'saw', 'dur': 1}), "layers[0].source is one of noise, sine, not 'saw'"),
    (rec({'source': 'noise'}), 'layers[0].dur is missing: a source needs its length in seconds'),
    (rec({'source': 'sine', 'dur': 1}), 'layers[0].hz is missing: a sine needs its frequency, from 20 to 150 Hz'),
    (rec({'source': 'sine', 'dur': 1, 'hz': 440}), 'layers[0].hz is a frequency in Hz from 20 to 150, not 440'),
    (rec({'source': 'noise', 'dur': 1, 'color': 'green'}), "layers[0].color is one of white, pink, brown"),
    (rec({'source': 'noise', 'dur': 1, 'size': 0.5}), 'layers[0].size does not belong to the source noise'),
    (rec({'source': 'noise', 'dur': 1, 'hz': 50}), 'layers[0].hz does not belong to the source noise'),
    (rec(TAP, {**TAP, 'follow': 'loud'}), 'layers[1].follow is an object such as {"layer": 0, "amount": 0.8}'),
    (rec(TAP, {**TAP, 'follow': True}), 'layers[1].follow is an object such as {"layer": 0, "amount": 0.8}'),
    (rec(TAP, {**TAP, 'follow': {'amount': 1}}), 'layers[1].follow.layer is missing'),
    (rec(TAP, {**TAP, 'follow': 1}), 'layers[1].follow.layer is 1, the layer itself'),
    (rec(TAP, {**TAP, 'follow': 5}), 'layers[1].follow.layer is 5, but the recipe has 2 layers (0 to 1)'),
    (rec(TAP, {**TAP, 'follow': 0.5}), 'layers[1].follow is an object such as'),
    (rec(TAP, {**TAP, 'follow': {'layer': 0, 'amount': 2}}), 'layers[1].follow.amount is an amount from -1 to 1'),
    (rec(TAP, {**TAP, 'follow': {'layer': 0, 'smooth': 0}}), 'layers[1].follow.smooth is a time in seconds from '
                                                             '0.005 to 2, not 0'),
    (rec(TAP, {**TAP, 'follow': {'layer': 0, 'speed': 2}}), "unknown field 'speed' in layers[1].follow: follow "
                                                            "takes layer, amount, smooth"),
])
def test_every_mistake_is_one_sentence_that_names_the_field(bad, says):
    with pytest.raises(R.RecipeError) as e:
        R.check(bad)
    assert says in str(e.value) and '\n' not in str(e.value)


def test_a_cut_past_the_take_and_a_layer_filtered_to_nothing_say_so():
    with pytest.raises(R.RecipeError, match=r'layers\[0\]\.cut starts at 30 s, after the end of the take'):
        R.render(rec({**TAP, 'cut': [30, 31]}))
    with pytest.raises(R.RecipeError, match='no file and no cookbook recipe is called'):
        R.load('no_such_recipe')
    with pytest.raises(R.RecipeError, match='the recipe is not valid JSON'):
        R.load('{"name": "x", ')


# ------------------------------------------------------------------------------------------------ layer controls
def test_pitch_moves_a_take_as_a_tape_does():
    y, low, high = one(TAP), one({**TAP, 'pitch': -12}), one({**TAP, 'pitch': 12})
    assert abs(len(low) / len(y) - 2.0) < 0.02 and abs(len(high) / len(y) - 0.5) < 0.02
    assert centroid(low) < 0.75 * centroid(y) < centroid(high)


def test_glide_moves_the_pitch_over_the_take_as_a_tape_that_slows():
    tone = {'source': 'sine', 'dur': 1.0, 'hz': 100}

    def hz(y):
        s = np.abs(np.fft.rfft(y[:, 0] * np.hanning(len(y)), 8 * len(y)))
        return float(np.fft.rfftfreq(8 * len(y), 1 / dsp.SR)[int(np.argmax(s))])
    y = one({**tone, 'glide': -12})
    q = len(y) // 8
    assert 90 < hz(y[:q]) < 101 and 50 < hz(y[-q:]) < 58                     # 100 Hz sinking to 50 Hz
    assert 1.42 < len(y) / dsp.SR < 1.46                      # slower as it sinks: 1 s of take lasts 1 / ln 2 s
    up = one({**tone, 'pitch': -12, 'glide': 0})
    assert 48 < hz(up[:len(up) // 8]) < 58 and 90 < hz(up[-len(up) // 8:]) < 101
    flat = one({**tone, 'pitch': -5, 'glide': -5})
    assert abs(len(flat) / len(one({**tone, 'pitch': -5})) - 1.0) < 0.01
    assert R.render(rec({**TAP, 'glide': -7}))[1]['parts'] == ['switch/power_strip (+0 to -7 semitones)']
    with pytest.raises(R.RecipeError, match=r'layers\[0\]\.glide is a number of semitones from -24 to 12, not 30'):
        R.check(rec({**TAP, 'glide': 30}))


def test_reverse_cut_filters_pan_and_fades():
    y = one(AIR)
    back = one({**AIR, 'reverse': True})
    assert len(back) == len(y)
    e, eb = np.abs(y).mean(axis=1), np.abs(back).mean(axis=1)
    half = len(y) // 2
    assert (e[:half].sum() > e[half:].sum()) != (eb[:half].sum() > eb[half:].sum())
    cut = one({**AIR, 'cut': [0.05, 0.15]})
    assert abs(len(cut) / dsp.SR - 0.10) < 0.002
    assert centroid(one({**AIR, 'lowpass': 400})) < 0.6 * centroid(y) < centroid(one({**AIR, 'highpass': 3000}))
    left, right = one({**AIR, 'pan': -1}), one({**AIR, 'pan': 1})
    assert np.abs(left[:, 1]).max() < 1e-6 < np.abs(left[:, 0]).max()
    assert np.abs(right[:, 0]).max() < 1e-6 < np.abs(right[:, 1]).max()
    for lay in (AIR, {**AIR, 'pitch': -5}, {**AIR, 'lowpass': 400}):
        assert abs(np.abs(one(lay)).max() - R.PEAK) < 1e-6           # the same peak before the gain
    slow = one({**AIR, 'fade_in': 0.1})
    k = dsp.n_of(0.02)
    assert np.abs(slow[:k]).max() < 0.5 * np.abs(y[:k]).max() + 1e-9
    swell, decay = one({**AIR, 'swell': 0.15}), one({**AIR, 'decay': 0.15})
    assert np.abs(swell[:k]).max() < 0.2 * np.abs(y[:k]).max() + 1e-9
    assert np.abs(decay[-k:]).max() < 0.2 * np.abs(y[-k:]).max() + 1e-9


def test_a_design_axis_reaches_the_design_and_one_it_lacks_is_a_note():
    small, big = one({**TAP, 'size': -1}), one({**TAP, 'size': 1})
    assert centroid(big) < centroid(small)
    noted = {a for a in R.AXES if R.render(rec({**TAP, a: 1.0}))[1].get('notes')}
    assert noted and 'size' not in noted
    a = sorted(noted)[0]
    said = R.render(rec(TAP, {**TAP, a: 1.0, 'at': 0.4}))[1]['notes']
    assert len(said) == 1 and said[0].startswith(f'layers[1]: switch/power_strip has no control of its {a} and '
                                                 'plays as designed')


def test_gain_and_at_place_a_layer_against_the_others():
    alone, _ = R.render(rec(TAP))
    late, m = R.render(rec(TAP, {**TAP, 'at': 1.2, 'gain': -12}))
    assert 1.15 < m['dur_s'] - len(alone) / dsp.SR < 1.26
    e = np.abs(late).max(axis=1)
    first, second = e[:dsp.n_of(0.3)].max(), e[dsp.n_of(1.2):].max()
    assert 9.0 < 20 * np.log10(first / second) < 15.0


def test_a_moving_cutoff_opens_noise_and_a_sine_stays_low():
    y = one({'source': 'noise', 'dur': 2.0, 'lowpass': [150, 6000]})
    assert abs(len(y) / dsp.SR - 2.0) < 0.01
    q = len(y) // 4
    assert centroid(y[:q]) < 0.4 * centroid(y[-q:])
    s = one({'source': 'sine', 'dur': 1.0, 'hz': 50, 'to': 40})
    spec = np.abs(np.fft.rfft(s[:, 0]))
    assert 38 <= np.fft.rfftfreq(len(s), 1 / dsp.SR)[int(np.argmax(spec))] <= 52
    colours = {c: centroid(one({'source': 'noise', 'dur': 1.0, 'color': c})) for c in R.COLORS}
    assert colours['brown'] < colours['pink'] < colours['white']


def test_events_repeat_a_layer_faster_and_softer():
    ev = {'count': 4, 'gap': 0.4, 'faster': 0.5, 'softer': -3}
    lay = R.check(rec({**TAP, 'at': 0.1, 'gain': -2, 'events': ev}))['layers'][0]
    rows = R._placed(lay, 0, 0, [])
    assert [round(t, 3) for _, t, _ in rows] == [0.1, 0.5, 0.7, 0.8]
    assert np.allclose([20 * np.log10(g) for _, _, g in rows], [-2, -5, -8, -11])
    assert not np.array_equal(rows[0][0], rows[1][0])                       # each event is a new take
    loose = [t for _, t, _ in R._placed({**lay, 'events': {**ev, 'loose': 1.0}}, 0, 0, [])]
    assert loose[0] == 0.1 and loose != [0.1, 0.5, 0.7, 0.8] and loose == sorted(loose)
    assert all(abs(a - b) <= 0.5 * g + 1e-9 for a, b, g in zip(loose, [0.1, 0.5, 0.7, 0.8], [0, 0.2, 0.1, 0.05]))
    again = [t for _, t, _ in R._placed({**lay, 'gain': -9, 'events': {**ev, 'loose': 1.0}}, 0, 3, [])]
    assert again == loose                                 # one hand for every layer: the same events stay together
    y, meta = R.render(rec({**TAP, 'events': ev}))
    assert 0.75 < meta['dur_s'] < 1.6 and meta['source'] == 'recipe'


def test_follow_gives_a_layer_the_loudness_of_another_or_makes_it_give_way():
    lead = np.zeros((dsp.n_of(1.0), 2))
    lead[dsp.n_of(0.2):dsp.n_of(0.4)] = 0.5
    full = R._following(lead, {'amount': 1.0, 'smooth': 0.05})
    half = R._following(lead, {'amount': 0.5, 'smooth': 0.05})
    duck = R._following(lead, {'amount': -1.0, 'smooth': 0.05})
    on, off = dsp.n_of(0.3), dsp.n_of(0.7)
    assert full[on] > 0.99 and full[off] < 1e-6
    assert half[on] > 0.99 and abs(half[off] - 0.5) < 1e-6
    assert duck[on] < 0.01 and duck[off] > 0.99
    # in a recipe: noise that follows four taps is loud at each tap and quiet between them; the parts say so
    taps = {**TAP, 'events': {'count': 4, 'gap': 0.5}}
    noise = {'source': 'noise', 'dur': 2.0, 'color': 'white', 'highpass': 2000, 'gain': 0}

    def between(y):
        e = np.abs(y).max(axis=1)
        return e[dsp.n_of(0.25):dsp.n_of(0.45)].max() / e.max()
    plain, _ = R.render(rec(taps, noise))
    followed, m = R.render(rec(taps, {**noise, 'follow': 0}))
    assert between(followed) < 0.1 * between(plain)
    assert m['parts'][1] == 'white noise (following layer 0, over 2000 Hz)'
    gave, m = R.render(rec(taps, {**noise, 'follow': {'layer': 0, 'amount': -1}}))
    assert m['parts'][1].startswith('white noise (giving way to layer 0') and between(gave) > 5 * between(followed)


def env(y, step=0.05):
    """the level of y in steps of `step` s (RMS)"""
    k = dsp.n_of(step)
    m = len(y) // k
    return np.sqrt((y[:m * k] ** 2).mean(axis=1).reshape(m, k).mean(axis=1))


def test_scatter_lays_a_take_many_times_at_random_moments():
    lay = R.check(rec({**TAP, 'scatter': {'rate': 40, 'dur': 3.0, 'pitch': 3, 'gain': 9, 'pan': 0.8}}))['layers'][0]
    y = R._scatter(lay, 0, 0, [])
    assert abs(np.abs(y).max() - R.PEAK) < 1e-9 and 3.0 <= len(y) / dsp.SR < 4.0
    e = env(y)[:58]
    assert (e > 0.02 * e.max()).mean() > 0.9                         # 40 a second: something sounds in every 50 ms
    assert abs(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) < 0.9              # spread over the sides
    assert np.array_equal(y, R._scatter(lay, 0, 0, [])) and not np.array_equal(y, R._scatter(lay, 1, 0, []))
    sparse = R._scatter({**lay, 'scatter': {'rate': 2, 'dur': 3.0}}, 0, 0, [])
    assert (env(sparse) > 0.02 * env(sparse).max()).mean() < 0.6
    # a rate that moves: few events at the start, many at the end
    grow = R._scatter({**lay, 'scatter': {'rate': 1, 'rate_to': 200, 'dur': 4.0}}, 0, 0, [])
    q = env(grow)
    assert q[:20].mean() < 0.3 * q[60:80].mean()
    # pieces of a texture: grains of 20 ms from a crackle
    fine = R._scatter(R.check(rec({'sound': 'fire/small', 'scatter': {'rate': 300, 'dur': 2.0, 'grain': 0.02}}))[
        'layers'][0], 0, 0, [])
    assert abs(len(fine) / dsp.SR - 2.02) < 0.01
    y, m = R.render(rec({**TAP, 'scatter': {'rate': 40, 'dur': 3.0}, 'pan': 0.5}))
    assert m['checks']['ok'] and m['parts'] == ['switch/power_strip (scattered 40 a second over 3 s)']


def test_scatter_stays_fast_with_thousands_of_events():
    import time
    r = rec({'sound': 'water/drop', 'scatter': {'rate': 200, 'dur': 10.0, 'pitch': 3, 'gain': 10, 'pan': 0.7}},
            dur=10)
    R.render(r)
    t = time.time()
    y, m = R.render(r, seed=1)
    assert time.time() - t < 3.0 and m['checks']['ok'] and abs(m['dur_s'] - 10.0) < 0.05


def test_level_draws_the_loudness_over_time_without_a_click():
    noise = {'source': 'noise', 'dur': 2.0, 'color': 'white'}
    plain = env(one(noise))
    y, _ = R.render(rec({**noise, 'level': [[0, 0], [1, 0], [1, -20], [2, -20]]}))
    e = env(y)
    assert 15 < 20 * np.log10(e[5:18].mean() / e[22:38].mean()) < 25
    jump = np.abs(np.diff(y[:, 0]))
    assert jump[dsp.n_of(0.99):dsp.n_of(1.01)].max() < 4 * np.percentile(jump, 99.9)
    assert plain.std() / plain.mean() < 0.2


def test_bend_moves_the_pitch_over_time_as_a_tape_does():
    tone = {'source': 'sine', 'dur': 1.0, 'hz': 100}

    def hz(y):
        s = np.abs(np.fft.rfft(y[:, 0] * np.hanning(len(y)), 8 * len(y)))
        return float(np.fft.rfftfreq(8 * len(y), 1 / dsp.SR)[int(np.argmax(s))])
    y, m = R.render(rec({**tone, 'bend': [[0, 0], [0.5, 0], [0.6, -12], [3, -12]]}))
    k = dsp.n_of(0.1)
    assert 95 < hz(y[k:4 * k]) < 105 and 47 < hz(y[-5 * k:-2 * k]) < 53
    assert 1.4 < m['dur_s'] < 1.5                            # 0.5 s at speed 1, the rest at half speed
    assert m['parts'] == ['a 100 Hz sine (bent through +0, +0, -12, -12 semitones)']


def test_pan_moves_across_and_pass_goes_by_with_its_pitch_falling():
    noise = {'source': 'noise', 'dur': 2.0, 'color': 'pink'}
    y, _ = R.render(rec({**noise, 'pan': [-1, 1]}))
    q = len(y) // 5
    assert np.abs(y[:q, 0]).mean() > 3 * np.abs(y[:q, 1]).mean()
    assert np.abs(y[-q:, 1]).mean() > 3 * np.abs(y[-q:, 0]).mean()
    tone = {'source': 'sine', 'dur': 4.0, 'hz': 120}
    lay = R.check(rec({**tone, 'pass': {'at': 2.0, 'speed': 40}}))['layers'][0]
    z = R._moved(one(tone), lay)

    def hz(y):
        s = np.abs(np.fft.rfft(y.mean(axis=1) * np.hanning(len(y)), 8 * len(y)))
        return float(np.fft.rfftfreq(8 * len(y), 1 / dsp.SR)[int(np.argmax(s))])
    k = dsp.n_of(0.5)
    before, after = hz(z[:k]), hz(z[-2 * k:-k])
    assert 2.0 < 12 * np.log2(before / after) < 5.0                 # 40 m/s: about 3.5 semitones from near to far
    e = env(z)
    assert np.argmax(e) * 0.05 == pytest.approx(2.0, abs=0.15) and e.max() > 8 * e[:5].mean()
    assert np.abs(z[:k, 0]).mean() > np.abs(z[:k, 1]).mean() and np.abs(z[-k:, 1]).mean() > np.abs(z[-k:, 0]).mean()


def test_spin_pulses_the_loudness_smooth_or_chopped():
    noise = {'source': 'noise', 'dur': 2.0, 'color': 'white'}
    for shape in R.SHAPES:
        y, _ = R.render(rec({**noise, 'spin': {'rate': 10, 'depth': 1.0, 'shape': shape}}))
        e = env(y, 0.005)[40:360]
        spec = np.abs(np.fft.rfft(e - e.mean()))
        assert abs(np.fft.rfftfreq(len(e), 0.005)[int(np.argmax(spec))] - 10) < 1.0
        assert e.min() < 0.1 * e.max()
    half, _ = R.render(rec({**noise, 'spin': {'rate': 10, 'depth': 0.5}}))
    e = env(half, 0.005)[40:360]
    assert 0.3 < e.min() / e.max() < 0.7


def test_stretch_makes_a_take_longer_at_the_same_pitch_and_keeps_its_hit():
    tone = {'source': 'sine', 'dur': 1.0, 'hz': 110}

    def hz(y):
        s = np.abs(np.fft.rfft(y[:, 0] * np.hanning(len(y)), 8 * len(y)))
        return float(np.fft.rfftfreq(8 * len(y), 1 / dsp.SR)[int(np.argmax(s))])
    long = one({**tone, 'stretch': 2.0})
    assert abs(len(long) / dsp.SR - 2.0) < 0.1 and abs(hz(long) - 110) < 3     # (its first 55 ms kept as they are)
    short = one({**tone, 'stretch': 0.5})
    assert abs(len(short) / dsp.SR - 0.5) < 0.06 and abs(hz(short) - 110) < 5
    hit = one(TAP)
    held = one({**TAP, 'stretch': 2.0})
    k = dsp.n_of(0.005)
    a, b = np.argmax(np.abs(hit).max(axis=1)), np.argmax(np.abs(held).max(axis=1))
    assert a == b and np.allclose(hit[:a + k], held[:a + k], atol=1e-6)        # the hit itself is kept as it was
    assert R.render(rec({**TAP, 'stretch': 2}))[1]['parts'] == ['switch/power_strip (stretched 2 times)']


def test_a_layer_may_be_a_cookbook_recipe():
    y, m = R.render(rec({'recipe': 'thunder_far', 'gain': 0}, {**TAP, 'at': 0.2}))
    alone, _ = R.render(R.cookbook()['thunder_far'])
    assert m['checks']['ok'] and m['parts'][0] == 'the recipe thunder_far' and len(y) >= len(alone) - dsp.n_of(0.1)
    short, _ = R.render(rec({'recipe': 'thunder_far', 'dur': 1.0}))
    assert abs(len(short) / dsp.SR - 1.0) < 0.05


# ------------------------------------------------------------------------------------------------------ the whole
def test_a_seed_gives_the_same_samples_and_another_seed_another_take():
    hiss = {'source': 'noise', 'dur': 0.6, 'lowpass': [200, 2000], 'gain': -10, 'follow': {'layer': 0, 'amount': 0.5}}
    r = rec(TAP, hiss, space='room')
    a, ma = R.render(r, seed=3)
    b, mb = R.render(r, seed=3)
    c, _ = R.render(r, seed=4)
    assert a.shape == b.shape and np.array_equal(a, b) and ma == mb
    assert a.shape != c.shape or not np.allclose(a, c)


def test_the_result_is_checked_and_levelled_like_any_sound():
    y, m = R.render(rec(TAP, {**AIR, 'at': 0.1, 'gain': -6}, says='a switch and a swipe', space='room'))
    assert m['id'] == 'recipe/thing' and m['source'] == 'recipe' and m['kind'] == 'object'
    assert set(m) >= {'seed', 'dur_s', 'peak_dbfs', 'lufs_max', 'checks', 'peak_s', 'onset_s', 'parts', 'says'}
    assert m['checks']['ok'], m['checks']
    assert m['peak_dbfs'] <= -0.9 and m['lufs_max'] <= -13.9 and (m['peak_dbfs'] > -1.3 or m['lufs_max'] > -14.3)
    assert m['parts'] == ['switch/power_strip', 'swipe/hand'] and y.ndim == 2 and y.shape[1] == 2
    assert 0.0 <= m['peak_s'] <= m['dur_s']
    bed, mb = R.render(rec({'source': 'noise', 'dur': 3.0, 'lowpass': 1500}, kind='ambience'))
    assert mb['kind'] == 'bed' and mb['lufs_max'] <= -21.9 and abs(mb['dur_s'] - 3.0) < 0.1
    cut, mc = R.render(rec({'source': 'noise', 'dur': 3.0}, dur=1.0))
    assert abs(mc['dur_s'] - 1.0) < 0.02 and mc['checks']['ok'], mc['checks']
    short, ms = R.render(rec(TAP, dur=5.0))
    assert any('not the 5 s asked' in n for n in ms['notes'])
    dry, md = R.render(rec(TAP))
    wet, mw = R.render(rec(TAP, space='hall'))
    assert mw['dur_s'] > md['dur_s'] + 0.3


# ----------------------------------------------------------------------------------------------------- the ways in
def test_make_writes_a_recipe_and_says_what_it_is_built_from(tmp_path):
    out = tmp_path / 'thing.wav'
    r = make(recipe=rec(TAP, {**AIR, 'pitch': -5, 'reverse': True, 'gain': -6}, says='a tap and air'), out=str(out))
    assert r['source'] == 'recipe' and r['sound'] == 'recipe/thing' and r['files'] == [str(out)]
    lines = r['summary'].splitlines()
    assert lines[0].startswith('thing.wav: recipe/thing (') and lines[0].endswith('clean) [recipe]')
    assert lines[1] == 'built from: switch/power_strip; swipe/hand (-5 semitones, reversed)'
    assert r['takes'][0]['checks']['ok'] and r['takes'][0]['file'] == str(out)
    y, sr = sf.read(str(out))
    assert sr == 48000 and abs(len(y) / sr - r['takes'][0]['dur_s']) < 0.01
    three = make(recipe=rec(TAP), out=str(tmp_path / 't.wav'), n=3, seed=5)
    assert [t['seed'] for t in three['takes']] == [5, 6, 7] and len(three['files']) == 3
    text = make(recipe=json.dumps(rec(TAP)), out=str(tmp_path / 'j.wav'))
    assert text['sound'] == 'recipe/thing'
    with pytest.raises(ValueError, match='a recipe is rendered as it is written'):
        make(recipe=rec(TAP), out=str(out), pitch='C4')
    with pytest.raises(R.RecipeError, match=r'layers\[0\]\.pitch'):
        make(recipe=rec({**TAP, 'pitch': 99}), out=str(out))


def test_a_set_takes_a_recipe_as_one_of_its_sounds(tmp_path):
    r = make_set({'sounds': [{'name': 'tap', 'want': 'a soft tap'}, {'name': 'built', 'recipe': rec(TAP, AIR)}]},
                 out=str(tmp_path))
    rows = {s['name']: s for s in r['sounds']}
    assert rows['built']['sound'] == 'recipe/thing' and rows['built']['ok']
    assert (tmp_path / 'built.wav').exists() and (tmp_path / 'tap.wav').exists()
    assert any(ln.startswith(str(tmp_path / 'built.wav')[-9:]) or 'built.wav: recipe/thing' in ln
               for ln in r['summary'].splitlines())
    man = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    assert [s['sound'] for s in man['sounds']] == [rows['tap']['sound'], 'recipe/thing']
    with pytest.raises(ValueError, match='built: want: a recipe of a set takes name, recipe, seed, lufs and dur'):
        make_set({'sounds': [{'name': 'built', 'recipe': rec(TAP), 'want': 'a tap'}]}, out=str(tmp_path))


def test_the_command_line_renders_a_file_standard_input_and_says_a_mistake(tmp_path, capsys, monkeypatch):
    f = tmp_path / 'r.json'
    f.write_text(json.dumps(rec(TAP, AIR)), encoding='utf-8')
    out = tmp_path / 'a.wav'
    assert cli.main(['make', '--recipe', str(f), '-o', str(out)]) == 0
    said = capsys.readouterr().out
    assert 'a.wav: recipe/thing' in said and 'built from: switch/power_strip; swipe/hand' in said and out.exists()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(rec(TAP))))
    assert cli.main(['make', '--recipe', '-', '-o', str(tmp_path / 'b.wav'), '--json']) == 0
    res = json.loads(capsys.readouterr().out)
    assert res['source'] == 'recipe' and res['takes'][0]['checks']['ok']
    f.write_text(json.dumps(rec({**TAP, 'gain': 99})), encoding='utf-8')
    assert cli.main(['make', '--recipe', str(f), '-o', str(out)]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and 'layers[0].gain is a level in dB from -40 to 12, not 99' in err[0]
    assert cli.main(['recipe', '--check', str(f)]) == 1
    capsys.readouterr()
    f.write_text(json.dumps(rec(TAP, AIR)), encoding='utf-8')
    assert cli.main(['recipe', '--check', str(f)]) == 0
    assert 'thing: a recipe of 2 layers: switch/power_strip, swipe/hand' in capsys.readouterr().out


def test_the_recipe_command_prints_the_form_three_examples_and_the_cookbook(capsys):
    assert cli.main(['recipe', '--example']) == 0
    said = capsys.readouterr().out
    assert 'earshot make --recipe file.json -o out.wav' in said
    shown = [json.loads(part) for part in said.split('\n\n')[1:]]
    assert len(shown) == 3
    assert any('scatter' in lay for r in shown for lay in r['layers'])          # many of one, and motion
    assert any('pass' in lay or 'bend' in lay for r in shown for lay in r['layers'])
    assert '"scatter"' not in said.split('\n\n')[0] and 'scatter {' in said.split('\n\n')[0]
    for r in shown:
        assert R.check(r)['name'] in R.cookbook()
    assert cli.main(['recipe', '--list']) == 0
    listed = capsys.readouterr().out.splitlines()
    assert listed[0] == f'{len(R.cookbook())} recipes in the cookbook' and len(listed) == len(R.cookbook()) + 1
    name = next(iter(R.cookbook()))
    assert cli.main(['recipe', name]) == 0
    assert json.loads(capsys.readouterr().out)['name'] == name


def test_the_mcp_make_tool_takes_a_recipe(tmp_path):
    out = tmp_path / 'm.wav'
    r, _ = mcp_server._result('make', dict(want='a tap and air', out=str(out), recipe=rec(TAP, AIR)),
                              mcp_server._summary_only(make))
    assert r['source'] == 'recipe' and out.exists() and r['text'].splitlines()[0].startswith('m.wav: recipe/thing')
    with pytest.raises(mcp_server.ToolFailure, match=r'layers\[0\]\.sound .* is not in the catalogue'):
        mcp_server._result('make', dict(want='x', out=str(out), recipe=rec({'sound': 'no/such'})),
                           mcp_server._summary_only(make))
    import inspect
    assert 'recipe' in inspect.signature(mcp_server.tool_make).parameters


def test_recipes_load_no_package_of_an_extra(tmp_path):
    import os
    import pathlib
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[1]
    heavy = ('torch', 'transformers', 'huggingface_hub', 'librosa', 'matplotlib', 'sklearn', 'PIL', 'mcp', 'requests')
    code = ('import sys\n'
            'from earshot import cli\n'
            'from earshot.make import make\n'
            f'make(recipe={rec(TAP, {"source": "noise", "dur": 0.5, "lowpass": [200, 900], "gain": -9})!r}, '
            f'out={str(tmp_path / "a.wav")!r})\n'
            f'make("a dragon roar", {str(tmp_path / "b.wav")!r})\n'
            'cli.main(["recipe", "--example"])\n'
            f'print(sorted(m for m in {heavy!r} if m in sys.modules))\n')
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp_path, timeout=300,
                       env={**os.environ, 'PYTHONPATH': str(root)})
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip().splitlines()[-1] == '[]'


# ---------------------------------------------------------------------------------------------------- the cookbook
def test_the_cookbook_is_valid_and_names_what_it_answers():
    book = R.cookbook()
    assert len(book) >= 25
    for name, r in book.items():
        rc = R.check(r)
        assert rc['name'] == name and rc.get('says') and rc.get('things') and rc.get('built'), name
    things = R.things()
    assert things['rocket launch'] == 'rocket_launch' and things['dragon roar'] == 'dragon_roar'
    assert all(n in book for n in R.EXAMPLES)


def test_a_request_that_names_a_cookbook_thing_is_answered_by_its_recipe(tmp_path):
    from earshot import brief
    assert R.answer(brief.read('a car crash'))['name'] == 'car_crash'
    assert R.answer(brief.read('a soft click for a button')) is None
    assert R.answer(brief.read('a car crash'), why=('car', 'crash')) is None        # the catalogue has it itself
    assert R.answer(brief.read('a click for my car crash app')) is None             # where it plays, not the sound
    r = make('a car crash', str(tmp_path / 'd.wav'))
    assert r['source'] == 'recipe' and r['sound'] == 'recipe/car_crash' and r['takes'][0]['checks']['ok']
    assert 'built from parts by the cookbook recipe car_crash' in r['summary']
    assert make('a car crash', str(tmp_path / 'n.wav'), sound=r['also'][0])['sound'] == r['also'][0]
    bright = make('a car crash', str(tmp_path / 'e.wav'), bright=0.5)
    assert 'not followed: bright; a recipe plays as it is written' in bright['summary'].splitlines()


def test_a_sketch_is_named_as_a_start_and_not_given_as_the_answer(tmp_path):
    """a cookbook recipe the listening model does not hear as its thing yet is marked `sketch`: the catalogue
    answers the request as before, and one line names the recipe to start from"""
    from earshot import brief
    sketches = [n for n, r in R.cookbook().items() if r.get('sketch')]
    assert 'geiger_counter' in sketches and 'car_crash' not in sketches
    b = brief.read('a geiger counter')
    assert R.answer(b) is None and R.answer(b, sketch=True)['name'] == 'geiger_counter'
    assert R.answer(brief.read('a car crash'), sketch=True) is None
    r = make('a geiger counter', str(tmp_path / 'g.wav'))
    assert r.get('source') != 'recipe' and not r['sound'].startswith('recipe/')
    assert any(ln.startswith('a start from parts: earshot make --recipe geiger_counter (a sketch of a geiger '
                             'counter clicking;') for ln in r['summary'].splitlines())
    built = make(recipe='geiger_counter', out=str(tmp_path / 'h.wav'))
    assert built['sound'] == 'recipe/geiger_counter' and built['takes'][0]['checks']['ok']
    with pytest.raises(R.RecipeError, match="sketch is true or false, not 'yes'"):
        R.check(rec(sketch='yes'))
    assert any(ln.startswith('geiger_counter') and '(a sketch)' in ln for ln in R.listing()['summary'].splitlines())


def test_what_no_design_is_gets_the_line_that_says_it_can_be_built(tmp_path):
    r = make('a frobnicating zorp', str(tmp_path / 'z.wav'))
    lines = r['summary'].splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith('also (the pick is a guess: earshot makes no '))
    assert lines[i + 1] == 'build it from parts: earshot recipe --example'
    assert 'build it from parts' not in make('a soft click', str(tmp_path / 'c.wav'))['summary']


@pytest.mark.slow
@pytest.mark.parametrize('name', sorted(R.cookbook()))
def test_every_cookbook_recipe_renders_clean(name):
    y, m = R.render(R.cookbook()[name])
    assert m['checks']['ok'], (name, [w['msg'] for w in m['checks']['warnings']])
    assert not m.get('notes'), (name, m['notes'])
    assert 0.3 <= m['dur_s'] <= 30.0 and np.isfinite(y).all()
