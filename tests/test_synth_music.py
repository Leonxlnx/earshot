"""Music beds (earshot.synth.music, earshot.sounds.beds): the score keeps the rules it is written by (key, whole
bars, voice leading, feel, a small motif, an arrangement that follows energy) and the audio is what it says: a
loop at its tempo and in its key that closes without a seam, or a bed played once that ends on a ringing chord."""
import time

import numpy as np
import pytest
from scipy import signal

from earshot.forge import dsp
from earshot.sounds import beds, registry
from earshot.synth import music, texture

SR = dsp.SR
STYLES = list(music.STYLES)


def _scale(plan):
    return {(plan['tonic'] + s) % 12 for s in music.MODES[plan['mode']]}


def _pitches(plan):
    return [m for p in plan['parts'] for _, m, *_ in p['notes'] if not isinstance(m, str)]


def _count(plan):
    return sum(len(p['notes']) for p in plan['parts'])


def _part(plan, name):
    return next(p for p in plan['parts'] if p['name'] == name)


def _flux(y):
    f, t, z = signal.stft(y.mean(axis=1), SR, nperseg=1024, noverlap=1024 - 96, boundary=None)
    x = np.maximum(np.diff(np.log1p(30 * np.abs(z)), axis=1), 0).sum(axis=0)
    return x - x.mean()


def _tempo(y, bpm):
    """the beat by autocorrelation of the onset strength, searched within 8 % of bpm"""
    x = _flux(y)
    n = len(x)
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n] / (n - np.arange(n))
    lag = 60.0 / bpm * SR / 96
    lo, hi = int(lag * 0.92), int(lag * 1.08) + 1
    return 60.0 * SR / 96 / (lo + int(np.argmax(ac[lo:hi])))


def _in_scale(y, plan, shift=0):
    """the share of the energy on notes (55-660 Hz, within 35 cents) that lies on the plan's scale moved up by
    `shift` semitones"""
    f, t, z = signal.stft(y.mean(axis=1), SR, nperseg=16384, noverlap=12288)
    p = (np.abs(z) ** 2).sum(axis=1)
    ok = (f >= 55) & (f <= 660)
    midi = 69 + 12 * np.log2(f[ok] / 440.0)
    near = np.abs(midi - np.round(midi)) < 0.35
    hist = np.bincount(np.round(midi[near]).astype(int) % 12, p[ok][near], 12)
    return float(hist[[(k + shift) % 12 for k in _scale(plan)]].sum() / hist.sum())


@pytest.mark.parametrize('style', STYLES)
def test_every_note_is_in_the_key(style):
    for seed, key, tonic in ((1, None, None), (2, 'F#', 6), (3, 233.08, 10)):
        plan = music.compose(style, dsp.rng(seed), key=key)
        assert tonic is None or plan['tonic'] == tonic
        assert all(m % 12 in _scale(plan) for m in _pitches(plan))
        assert plan['length_s'] == pytest.approx(plan['bars'] * 240.0 / plan['bpm'])
        assert all(0.0 <= g < plan['bars'] * 4.0 for p in plan['parts'] for *_, g in p['notes'])


@pytest.mark.parametrize('style', STYLES)
def test_chords_are_voiced_by_voice_leading(style):
    moved = []
    for seed in (1, 2, 3, 4):
        plan = music.compose(style, dsp.rng(seed))
        for c in plan['chords']:
            v = c['voicing']
            assert list(v) == sorted(set(v)) and min(v) >= 53            # upper voices from F3 up
            assert {m % 12 for m in v} <= set(c['pcs'])
            assert not any((b - a) % 12 == 1 for i, a in enumerate(v) for b in v[i + 1:])     # no minor ninths
            assert c['bass'] < 48 and c['bass'] % 12 == c['bass_pc']      # the bass under 124 Hz, alone
        assert max(plan['lead_steps']) <= 4.0
        moved.append(plan['lead_st'])
    assert np.mean(moved) <= 2.2                                          # semitones a voice moves per change


def test_a_loop_turns_round_and_a_bed_played_once_ends_on_the_tonic():
    for style in STYLES:
        st = music.STYLES[style]
        loop = music.compose(style, dsp.rng(4))
        last = loop['chords'][-1]
        if st['turn']:
            assert last['deg'] == 4 or (loop['mode'] in music.MINORS and last['deg'] == 6)
        once = music.compose(style, dsp.rng(4), loop=False)
        end = once['chords'][-1]
        assert end['final'] and end['root'] == once['tonic'] and end['beat'] == (once['bars'] - 1) * 4.0
        assert not any(c['final'] for c in loop['chords'])


def test_a_seed_is_a_take_of_the_style():
    for style in STYLES:
        a, b = (music.compose(style, dsp.rng(7)) for _ in range(2))
        assert music.chart(a) == music.chart(b)
        assert [p['notes'] for p in a['parts']] == [p['notes'] for p in b['parts']]
        takes = {(music.chart(p), tuple(c['voicing'] for c in p['chords']))
                 for p in (music.compose(style, dsp.rng(s)) for s in range(1, 7))}
        assert len(takes) >= 3                                            # other progressions or voicings
        assert len({tuple(q['name'] for q in music.compose(style, dsp.rng(s))['parts']) for s in (1, 2)}) == 1


@pytest.mark.parametrize('style', STYLES)
def test_energy_is_density(style):
    quiet = np.mean([_count(music.compose(style, dsp.rng(s), energy=-0.9)) for s in (1, 2, 3)])
    busy = np.mean([_count(music.compose(style, dsp.rng(s), energy=0.9)) for s in (1, 2, 3)])
    assert busy > quiet


def test_the_feel_is_by_role_and_no_two_bars_are_the_same():
    plan = music.compose('warm', dsp.rng(3), energy=0.3)
    spb, (unit, swing) = 60.0 / plan['bpm'], music.STYLES['warm']['swing']

    def late(name):
        out = []
        for t, _, _, _, g in _part(plan, name)['notes']:
            odd = abs(g / unit - round(g / unit)) < 1e-6 and int(round(g / unit)) % 2 == 1
            out.append((t - g - (swing * unit if odd else 0.0)) * spb * 1e3)
        return np.array(out)
    assert late('shaker').mean() < -3.0 and late('snare').mean() > 5.0         # hats ahead, the snare laid back
    assert abs(late('kick').mean()) < 2.0 and 0.5 < late('kick').std() < 6.0   # the kick on the beat, not a machine
    vel = np.array([n[2] for n in _part(plan, 'shaker')['notes']])
    assert vel.std() > 0.05 and vel.max() > 1.5 * vel.min()                    # accents and ghost notes
    kick = _part(plan, 'kick')['notes']
    bars = [tuple((round(t - 4 * b, 4), round(v, 3)) for t, _, v, _, g in kick if b * 4 <= g < b * 4 + 4)
            for b in range(plan['bars'])]
    assert len(set(bars)) == len(bars)                                         # no two bars identical
    assert all(any(g == b * 4.0 for *_, g in kick) for b in range(plan['bars']))   # and each has its downbeat
    chip = music.compose('8bit', dsp.rng(3))
    assert all(t == g for p in chip['parts'] for t, _, _, _, g in p['notes'])      # a chip is a machine


def test_a_top_line_is_a_short_motif_that_leaves_room():
    for style in STYLES:
        for seed in (1, 2, 3):
            plan = music.compose(style, dsp.rng(seed), energy=0.3)
            minor = plan['mode'] in music.MINORS
            penta = {(plan['tonic'] + music._semi(plan['mode'], s)) % 12 for s in music.PENTATONIC[minor]}
            for p in plan['parts']:
                if p.get('pattern') != 'motif' or not p['notes']:
                    continue
                grid = sorted(g for *_, g in p['notes'])
                assert len({m for _, m, *_ in p['notes']}) <= 7
                assert all(m % 12 in penta or m % 12 in _scale(plan) for _, m, *_ in p['notes'])
                for b in range(int(grid[0] // 8) * 8, plan['bars'] * 4, 8):          # each two-bar phrase
                    inside = [g - b for g in grid if b <= g < b + 8]
                    assert len(inside) <= 6 and (not inside or max(inside) <= 6.0)


def test_a_loop_is_whole_bars_at_its_tempo_in_its_key_and_closes():
    plan = music.compose('8bit', dsp.rng(1), bars=3)
    t0 = time.perf_counter()
    y = music.render(plan, dsp.rng(5))
    took = time.perf_counter() - t0
    assert y.shape == (int(round(plan['length_s'] * SR)), 2) and np.isfinite(y).all()
    assert np.abs(y).max() == pytest.approx(0.5, abs=1e-6)
    assert np.array_equal(y, music.render(plan, dsp.rng(5)))                   # the same takes, the same samples
    assert took < 2.0 * plan['length_s'] + 5.0
    assert abs(_tempo(y, plan['bpm']) / plan['bpm'] - 1.0) < 0.02
    assert _in_scale(y, plan) > 0.8 and _in_scale(y, plan) > _in_scale(y, plan, 6) + 0.2
    assert texture.seam_db(y)['click'] <= 0.0                                  # no step at the loop point
    f = np.fft.rfftfreq(len(y), 1.0 / SR) < 100.0
    side, mid = (np.sum(np.abs(np.fft.rfft(y[:, 0] + s * y[:, 1])[f]) ** 2) for s in (-1.0, 1.0))
    assert 10 * np.log10(side / mid + 1e-30) < -40.0                           # the low end is mono
    assert 10 * np.log10(np.mean(y.mean(axis=1) ** 2) / np.mean(y ** 2)) > -1.5     # and survives a mono sum


def test_the_room_goes_round_the_loop_point_and_a_bed_played_once_rings_out():
    loop = music.render(music.compose('warm', dsp.rng(2), bars=2), dsp.rng(2))
    plan = music.compose('warm', dsp.rng(2), bars=2, loop=False)
    once = music.render(plan, dsp.rng(2))
    head = dsp.n_of(0.02)
    assert np.sqrt(np.mean(loop[:head] ** 2)) > 30 * np.sqrt(np.mean(once[:head] ** 2))    # tails on the start
    assert texture.seam_db(loop)['click'] <= 0.0
    bar = 240.0 / plan['bpm']
    end = dsp.n_of(0.04 + 2 * bar)
    assert len(once) > end and np.abs(once[-1]).max() < 1e-4                       # room after its bars, ends at 0
    chord = once[dsp.n_of(0.04 + bar + 0.3):end]                                   # the final bar: one chord, held
    first, second = chord[:len(chord) // 2], chord[len(chord) // 2:]
    assert 0.02 < np.sqrt(np.mean(second ** 2)) / np.sqrt(np.mean(first ** 2)) < 1.0   # still ringing, and dying
    assert np.abs(_flux(chord)).max() < 0.5 * np.abs(_flux(once[:end])).max()      # nothing is struck in it again


def test_the_beds_of_the_catalogue():
    rows = {r['id']: r for r in registry.rows(kind='music')}
    assert set(rows) == set(beds.BEDS) == {'bed/warm', 'bed/playful', 'bed/8bit'}
    for sid, r in rows.items():
        assert r['loop'] and 'music' in r['domains'] and len(r['says']) >= 2 and r['axes'] == list(beds.AXES)
        assert ('retro' in r['tags']) == (sid == 'bed/8bit') == (r['material'] == 'retro')
    bpm = 126.0 * 1.15 ** 0.5
    bars = round(4.0 * bpm / 240.0)
    y, meta = registry.render('bed/8bit', seed=1, dur=4.0, motion=0.5, pitch='E3')
    assert len(y) == int(round(bars * 240.0 / bpm * SR)) and meta['loop'] and meta['pitch'].startswith('E')
    assert 'ignored' not in meta and meta['lufs_max'] <= registry.LOUDEST['music'] + 0.1
    assert meta['checks']['ok'], meta['checks']['warnings']
    once, meta = registry.render('bed/8bit', seed=1, dur=4.0, loop=False, check=False)
    assert meta['loop'] is False and len(once) > len(y)
