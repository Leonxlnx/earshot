"""earshot.synth.voices and earshot.synth.phrase: the voice table carries what designs rely on, every voice plays
clean at its default pitch and in tune, out-of-range pitches move by octaves, a patch comes dry unless asked, and a
phrase puts its notes where it was told (onsets, accent, stereo spread, a shortened last note). Model-free."""
import numpy as np
import pytest

from earshot.forge import dsp, knobs
from earshot.synth import patches, phrase, played, strings, struck, voices

SR = dsp.SR
TABLE = voices.table()
# voices whose strongest low partial is not the note asked for: inharmonic bells, plates and drums heard by their
# strike tone or a virtual pitch, noise, and the effects that glide or play a figure of their own
UNPITCHED = {'tubular_bell', 'church_bell', 'triangle', 'cowbell', 'stab', 'zap', 'power_up', 'power_down',
             'riser_tone', 'noise_sweep'}


def _strongest(y, lo, hi, start=0.0, sec=0.5):
    """the frequency of the strongest spectral peak between lo and hi Hz in sec s of the mono sum from start"""
    m = y.mean(1) if y.ndim == 2 else y
    m = m[dsp.n_of(start):dsp.n_of(start) + dsp.n_of(sec)]
    n = len(m)
    nfft = 1 << int(np.ceil(np.log2(n * 4)))
    S = np.abs(np.fft.rfft(m * np.hanning(n), nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    b = np.nonzero((f >= lo) & (f <= hi))[0]
    return float(f[b[np.argmax(S[b])]])


def _cents(f, ref):
    return 1200 * np.log2(f / ref)


# ---- the table

def test_table_lists_every_instrument_and_patch():
    want = set(struck.INSTRUMENTS) | set(strings.INSTRUMENTS) | set(patches.PATCHES) | set(played.TABLE)
    assert set(TABLE) == want
    assert sorted(voices.names()) == sorted(want)
    assert set(voices.names('wood')) == {v for v, i in TABLE.items() if i['material'] == 'wood'}
    for v in ('marimba', 'harp', 'pizzicato', 'felt_piano'):          # the real instruments replace their models
        assert TABLE[v]['engine'] == 'played' and TABLE[v]['kind'] == 'real'
    assert TABLE['marimba']['material'] == 'wood' and TABLE['felt_piano']['material'] == 'felt'


@pytest.mark.parametrize('voice', sorted(TABLE))
def test_table_fields(voice):
    i = TABLE[voice]
    assert i['engine'] in voices.ENGINES
    assert i['says'] and all(isinstance(s, str) and s for s in i['says'])
    lo, hi = (knobs.hz(p) for p in i['range'])
    assert lo < hi and lo * 0.999 <= knobs.hz(i['default_pitch']) <= hi * 1.001
    assert 'material' in i
    if i['engine'] == 'patches':
        assert i['material'] == 'synth' and i['kind'] in voices.HOLD
    else:
        assert float(i['ring_s']) > 0                    # figure.play reads it to find sustaining voices
    assert voices.info(voice) is i


def test_unknown_voice():
    with pytest.raises(ValueError, match='kalimba'):
        voices.play('theremin', 'C5')
    with pytest.raises(ValueError):
        voices.info('theremin')


# ---- playing

@pytest.mark.parametrize('voice', sorted(TABLE))
def test_every_voice_plays_clean_and_in_tune(voice):
    y = voices.play(voice, r=dsp.rng(1))
    assert y.ndim == 2 and y.shape[1] == 2 and len(y) > dsp.n_of(0.03)
    assert np.isfinite(y).all() and 0 < np.abs(y).max() < 1.0
    if voice not in UNPITCHED:
        f0 = knobs.hz(TABLE[voice]['default_pitch'])
        f = _strongest(y, 0.6 * f0, 1.6 * f0, sec=min(3.0, len(y) / SR))
        assert abs(_cents(f, f0)) <= 30, (voice, f, f0)


def test_pitch_outside_the_range_moves_by_octaves():
    assert voices.clamp('kalimba', knobs.hz('C2')) == pytest.approx(knobs.hz('C4'))
    assert voices.clamp('glockenspiel', knobs.hz('C4')) == pytest.approx(knobs.hz('C5'))
    assert voices.clamp('marimba', knobs.hz('C9')) == pytest.approx(knobs.hz('C6'))     # the real marimba: A2..D6
    assert voices.clamp('marimba', knobs.hz('E4')) == pytest.approx(knobs.hz('E4'))
    for voice, asked, heard in (('kalimba', 'C2', 'C4'), ('marimba', 'A8', 'A5')):
        y = voices.play(voice, asked, r=dsp.rng(2))
        f0 = knobs.hz(heard)
        assert abs(_cents(_strongest(y, 0.75 * f0, 1.3 * f0), f0)) <= 30


def test_a_patch_comes_dry_unless_space_is_given():
    a = voices.play('soft_pluck', 'C5', r=dsp.rng(3))
    assert np.array_equal(a, voices.play('soft_pluck', 'C5', r=dsp.rng(3), space=-1.0))
    assert np.array_equal(a, patches.play('soft_pluck', knobs.hz('C5'), vel=0.7, dur=voices.HOLD['pluck'],
                                          r=dsp.rng(3), space=-1.0))
    wet = voices.play('soft_pluck', 'C5', r=dsp.rng(3), space=0.5)
    assert len(wet) > len(a)


def test_voices_are_deterministic():
    for v in ('marimba', 'harp', 'warm_keys'):
        assert np.array_equal(voices.play(v, r=dsp.rng(4)), voices.play(v, r=dsp.rng(4)))


# ---- phrase

def _onsets(y, k, min_gap=0.08):
    """the times of the k steepest rises of the 1 ms envelope, at least min_gap apart (a struck note's attack is its
    steepest rise)"""
    m = np.abs(y).max(1)
    hop = dsp.n_of(0.001)
    env = m[:len(m) // hop * hop].reshape(-1, hop).max(1)
    rise = np.maximum(np.diff(env, prepend=0.0), 0.0)
    out = []
    for i in np.argsort(rise)[::-1]:
        if all(abs(i - j) * 0.001 >= min_gap for j in out):
            out.append(int(i))
        if len(out) == k:
            break
    return [i * 0.001 for i in sorted(out)]


@pytest.mark.parametrize('vary', [0.0, 0.5])
def test_phrase_onsets_land_on_time(vary):
    at = [0.0, 0.21, 0.5, 0.62]
    # a modelled bar: the real marimba's yarn mallet reaches its steepest rise 6-12 ms after the note starts
    y = phrase.play([(t, 'C6') for t in at], 'xylophone', dsp.rng(5), vel=0.6, vary=vary)
    got = _onsets(y, len(at))
    jitter = 4 * 0.004 * vary                         # phrase.play moves a note by 4 ms x vary x N(0, 1)
    for t, g in zip(at, got):
        assert abs(g - t) <= 0.010 + jitter, (at, got)


def test_phrase_accent_is_louder():
    notes = [(0.0, 'G5'), (0.3, 'G5')]
    plain = phrase.play(notes, 'marimba', dsp.rng(6), vary=0.0)
    lead = phrase.play(notes, 'marimba', dsp.rng(6), vary=0.0, accent=1)
    k = dsp.n_of(0.3)
    p0, p1 = np.abs(plain[k:]).max(), np.abs(lead[k:]).max()
    assert 20 * np.log10(p1 / p0) == pytest.approx(2.5, abs=0.3)
    assert np.allclose(plain[:k - 10], lead[:k - 10])          # the other note is untouched
    last = phrase.play(notes, 'marimba', dsp.rng(6), vary=0.0, accent=-1)
    assert np.array_equal(last, lead)


def test_phrase_spread_moves_alternate_notes_left_and_right():
    notes = [(0.0, 'C6'), (0.3, 'C6'), (0.6, 'C6'), (0.9, 'C6')]
    y = phrase.play(notes, 'marimba', dsp.rng(7), vary=0.0, spread=0.8)
    c = phrase.play(notes, 'marimba', dsp.rng(7), vary=0.0, spread=0.0)
    w = dsp.n_of(0.25)

    def balance(x, i):
        """left over right RMS of note i"""
        L, R = np.sqrt((x[dsp.n_of(0.3 * i):dsp.n_of(0.3 * i) + w] ** 2).mean(0))
        return L / R
    for i in range(4):
        moved = balance(y, i) / balance(c, i)          # against the same note unpanned: an instrument has its own image
        assert (moved > 1.2) if i % 2 == 0 else (moved < 1 / 1.2), (i, moved)
    assert abs(np.log(balance(y, 3) / balance(c, 3))) > abs(np.log(balance(y, 1) / balance(c, 1)))  # wider later


def test_phrase_tail_shortens_the_last_note():
    notes = [(0.0, 'C5', 1.0, 0.08), (0.5, 'E5')]                   # the first note damped early (it lasts 0.39 s)
    full = phrase.play(notes, 'vibraphone', dsp.rng(8), vary=0.0)
    short = phrase.play(notes, 'vibraphone', dsp.rng(8), vary=0.0, tail=0.15)
    assert len(full) > dsp.n_of(1.5)
    assert len(short) == pytest.approx(dsp.n_of(0.65), abs=dsp.n_of(0.002))
    assert np.abs(short[-1]).max() < 1e-6
    k = dsp.n_of(0.5)
    assert np.array_equal(short[:k], full[:k])                         # the notes before it are untouched


def test_phrase_needs_a_note():
    with pytest.raises(ValueError):
        phrase.play([], 'marimba', dsp.rng(1))


def test_phrase_steps():
    s = phrase.steps('C5', [0, 4, 7], 0.1)
    assert [t for t, _ in s] == pytest.approx([0.0, 0.1, 0.2])
    assert [f for _, f in s] == pytest.approx([knobs.hz('C5') * 2 ** (k / 12) for k in (0, 4, 7)])
