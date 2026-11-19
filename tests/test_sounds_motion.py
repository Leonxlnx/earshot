"""earshot.sounds.motion: the film vocabulary renders clean, repeats per seed, varies between seeds, keeps the length
and the shape it is asked for and answers to its controls (model-free). The film designs are built from learned real
recordings (REAL); the film set's designs name their recordings by file, so they do not follow the store's order.
The fast tier renders the invented designs and the cheap film designs, --slow every film design (a boom of three
layers of real drums and a tam-tam takes several seconds to render)."""
import functools

import numpy as np
import pytest

from earshot.forge import dsp, sfx
from earshot.sounds import motion, registry
from earshot.synth import master

registry.load()
IDS = sorted(d for d, v in registry._designs.items() if v.fn.__module__ == motion.__name__)
REAL = sorted(d for d in IDS if 'real' in registry.get(d).tags)
INVENTED = sorted(set(IDS) - set(REAL))
FAST = ['whoosh/fast', 'flick/thin', 'swoosh/out', 'thump/appear', 'tick/soft', 'thump/door', 'tick/key',
        'click/cursor', 'slide/paper', 'whoosh/sweep', 'logo/bag']
AIR = ['whoosh/soft', 'whoosh/fast', 'whoosh/deep', 'whoosh/long', 'swoosh/in', 'swoosh/out', 'flick/paper',
       'flick/thin', 'slide/panel', 'pass/air', 'fall/air']
BUILDS = ['rise/air', 'rise/tension', 'rise/reverse', 'suck/in', 'rise/breath', 'rise/hit']


@functools.lru_cache(maxsize=None)
def _take(sid, seed=1, **req):
    return registry.render(sid, seed=seed, check=False, **req)


@functools.lru_cache(maxsize=None)
def _raw(sid, seed=1, **ax):
    """the design function's own output (before levelling), dry so the comparisons see the source"""
    d = registry.get(sid)
    c = registry.Controls(d, **{'space': -1.0, **ax})
    c.seed = seed
    return np.asarray(d.fn(c, dsp.rng(registry.seed_of(sid, seed))), np.float64)


def _mono(y):
    return y.mean(1) if y.ndim == 2 else y


def _spectrum(y):
    m = dsp.butter(_mono(y), 'high', 25.0)                   # as a finished take: nothing sub-audible
    return np.abs(np.fft.rfft(m)) ** 2, np.fft.rfftfreq(len(m), 1 / dsp.SR)


def _centroid(y):
    P, f = _spectrum(y)
    return float((P * f).sum() / P.sum())


def _share(y, lo, hi):
    """the share of the energy between lo and hi Hz"""
    P, f = _spectrum(y)
    return float(P[(f >= lo) & (f < hi)].sum() / P.sum())


def _env(y, hop=480):
    m = _mono(y)
    k = len(m) // hop
    return np.sqrt((m[:k * hop].reshape(k, hop) ** 2).mean(1))


def _peak_share(y, window=0.15):
    """the centre of the loudest stretch of a take (`window` of its length), as a share of the length"""
    e = _env(y, 240) ** 2
    w = max(1, int(window * len(e)))
    c = np.concatenate([[0.0], np.cumsum(e)])
    return float((np.argmax(c[w:] - c[:-w]) + w / 2) / len(e))


def test_catalogue_entries():
    assert len(IDS) == 69 and len(REAL) == 55
    assert len({i.split('/')[0] for i in IDS}) == 30
    for sid in IDS:
        d = registry.get(sid)
        assert d.domains[0] == 'motion', sid
        assert 2 <= len(d.says) <= 3 and len(d.uses) >= 4 and d.dur, sid
        hard = set() if d.axes == motion.REAL_SOFT else {'hard'}      # a low body whose strike hard does not move
        assert {'energy', 'size', 'bright', 'space', 'width'} | hard <= set(d.axes), sid


def _clean(sid, seed=1):
    d = registry.get(sid)
    c = registry.Controls(d)
    c.seed = seed
    y = master.finish(d.fn(c, dsp.rng(registry.seed_of(sid, seed))))            # as registry.render, keeping c
    assert np.isfinite(y).all() and y.ndim == 2
    assert np.abs(y).max() <= 1.0
    assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-3          # starts at zero, its tail is faded
    checks = registry.audit(y, d)
    assert checks['ok'], checks['warnings']
    assert 0.6 * d.dur <= len(y) / dsp.SR <= 1.6 * d.dur + 0.4, (len(y) / dsp.SR, d.dur)
    assert c.used & set(registry.AXES) == set(d.axes), sid                  # the axes it lists are the ones it reads
    if sid in REAL:
        assert c.member and c.member['recording'], sid                        # it says which recording it played


@pytest.mark.parametrize('sid', INVENTED + FAST)
def test_renders_clean(sid):
    _clean(sid)


@pytest.mark.slow
@pytest.mark.parametrize('sid', sorted(set(REAL) - set(FAST)))
def test_film_designs_render_clean(sid):
    _clean(sid)
    _clean(sid, 2)


@pytest.mark.parametrize('sid', ['whoosh/fast', 'knock/soft', 'pop/bubble', 'shine/sweep', 'glitch/short',
                                 'type/run'])
def test_deterministic_per_seed(sid):
    d = registry.get(sid)
    a, b, c = (np.asarray(d.fn(registry.Controls(d, space=-1.0), dsp.rng(registry.seed_of(sid, s))))
               for s in (4, 4, 5))
    assert np.array_equal(a, b)
    assert len(a) != len(c) or not np.allclose(a, c)


@pytest.mark.parametrize('sid', ['whoosh/fast', 'flick/thin'])
def test_dur_is_kept(sid):
    d = registry.get(sid)
    dur = round(1.3 * d.dur, 3)
    _, meta = _take(sid, 2, dur=dur)
    assert 'dur' not in meta.get('ignored', ())
    assert 0.75 * dur <= meta['dur_s'] <= 1.5 * dur + 0.4, (dur, meta['dur_s'])


@pytest.mark.slow
@pytest.mark.parametrize('sid', ['type/run', 'slide/panel', 'rise/reverse', 'fall/sub', 'pass/air'])
def test_other_lengths_are_kept(sid):
    dur = round(1.4 * registry.get(sid).dur, 3)
    _, meta = _take(sid, 2, dur=dur)
    assert 0.75 * dur <= meta['dur_s'] <= 1.5 * dur + 0.5, (dur, meta['dur_s'])


@pytest.mark.parametrize('sid', ['flick/thin', 'knock/soft', 'pop/soft', 'tock/glass', 'blip/soft',
                                 'shine/glint', 'sparkle/burst', 'glitch/short', 'type/soft', 'slide/panel'])
def test_size_lowers_centroid(sid):
    assert _centroid(_raw(sid, 2, size=0.8)) < _centroid(_raw(sid, 2, size=-0.8))


@pytest.mark.slow
@pytest.mark.parametrize('sid', ['whoosh/soft', 'boom/short', 'slam/title', 'pass/air', 'braam/brass', 'rise/tension',
                                 'title/bag', 'whoosh/sweep'])
def test_size_lowers_centroid_of_the_film_designs(sid):
    assert _centroid(_raw(sid, 2, size=0.8)) < _centroid(_raw(sid, 2, size=-0.8))


@pytest.mark.parametrize('sid', ['flick/thin', 'pop/soft', 'tock/wood', 'blip/soft', 'shine/glint',
                                 'glitch/short', 'type/soft'])
def test_energy_and_bright_act(sid):
    def top(**ax):
        y = _raw(sid, 2, **ax)
        return _centroid(y), _share(y, 2000.0, 24000.0)
    assert top(bright=0.8) > top() > top(bright=-0.8)
    assert top(energy=0.8) > top(energy=-0.8)
    assert top(bright=0.8)[1] > top(bright=-0.8)[1] and top(energy=0.8)[1] > top(energy=-0.8)[1]


@pytest.mark.slow
@pytest.mark.parametrize('sid', ['whoosh/soft', 'boom/short', 'slam/title', 'suck/in', 'rise/reverse', 'title/bag',
                                 'whoosh/sweep'])
def test_energy_and_bright_act_on_the_film_designs(sid):
    test_energy_and_bright_act(sid)


@pytest.mark.parametrize('sid', ['knock/soft', 'tock/wood', 'thump/soft'])
def test_hard_is_a_faster_onset(sid):
    def early(y):
        m = _mono(y)
        i = int(np.argmax(np.abs(m) > 0.1 * np.abs(m).max()))
        return float(np.sum(m[i:i + dsp.n_of(0.004)] ** 2) / np.sum(m ** 2))   # how much arrives within 4 ms
    assert early(_raw(sid, 2, hard=0.9)) > early(_raw(sid, 2, hard=-0.9))
    assert _centroid(_raw(sid, 2, hard=0.9)) > _centroid(_raw(sid, 2, hard=-0.9))


@pytest.mark.slow
@pytest.mark.parametrize('sid', ['slam/door'])
def test_hard_is_a_harder_strike_in_the_film_designs(sid):
    def strike(y):                      # dB of the strike (1 ms before to 4 ms after it) over the 12-60 ms after it
        p = _mono(y) ** 2
        t0 = float(np.argmax(np.convolve(p, np.ones(96), 'same'))) / dsp.SR      # the loudest 2 ms: the strike
        a, b = p[dsp.n_of(t0 - 0.001):dsp.n_of(t0 + 0.004)], p[dsp.n_of(t0 + 0.012):dsp.n_of(t0 + 0.06)]
        return 10 * np.log10(a.mean() / b.mean())
    assert strike(_raw(sid, 2, hard=0.9)) > strike(_raw(sid, 2, hard=-0.9)) + 0.5    # a slammed door: about 1 dB


@pytest.mark.parametrize('sid', ['whoosh/fast', 'flick/thin', 'thump/soft', 'pop/soft', 'tock/wood', 'type/soft',
                                 'sparkle/burst'])
def test_seeds_vary_but_measure_alike(sid):
    (a, ma), (b, mb) = _take(sid, 1), _take(sid, 2)
    k = min(len(a), len(b))
    assert abs(np.corrcoef(_mono(a)[:k], _mono(b)[:k])[0, 1]) < 0.9           # another take, not a copy
    assert abs(ma['dur_s'] - mb['dur_s']) <= 0.35 * max(ma['dur_s'], mb['dur_s'])
    assert abs(ma['lufs_max'] - mb['lufs_max']) <= 3.0
    assert abs(np.log2(_centroid(a) / _centroid(b))) < 1.2


def test_film_seeds_are_other_recordings():
    a, b = _take('whoosh/fast', 1)[1], _take('whoosh/fast', 2)[1]
    assert a['member']['recording'] != b['member']['recording']


def test_keys_differ_from_take_to_take():
    cen = [_centroid(_raw('type/soft', s)) for s in range(1, 7)]
    assert max(cen) / min(cen) > 1.05                               # another key under another finger
    assert max(cen) / min(cen) < 3.0                                # of the same keyboard


@pytest.mark.slow
@pytest.mark.parametrize('sid', AIR)
def test_no_rising_glide_in_air(sid):
    if 'rising_glide' in registry.get(sid).allows:
        pytest.skip('a real whoosh: the pitch detector reads its noise as gliding (motion.GLIDE)')
    assert sfx.rising_glide(_raw(sid, 3))['rise_st'] <= 4.0


@pytest.mark.slow
def test_builds_end_on_the_cut_and_die_after_it():
    for sid in BUILDS:
        raw = _raw(sid, 1)
        e = _env(raw, 240)
        assert _peak_share(raw) > 0.75, sid                         # its loudest stretch is its last
        assert _env(raw, 48)[-2:].max() < 0.3 * _env(raw, 48).max(), sid   # and it closes, not cut on its peak
        _, meta = _take(sid)
        assert meta['dur_s'] < len(raw) / dsp.SR + 1.2, sid         # the room (a hall at most) dies away


def _span20(y):
    """seconds from the first 5 ms frame within 20 dB of the loudest to the last one after it (20 ms smoothing)"""
    e = np.convolve(_env(y, 240) ** 2, np.ones(4) / 4, 'same')
    i = int(np.argmax(e))
    s, t = i, i
    while s > 0 and e[s - 1] > e[i] / 100:
        s -= 1
    while t < len(e) - 1 and e[t + 1] > e[i] / 100:
        t += 1
    return (t - s + 1) * 240 / dsp.SR, (i - s) * 240 / dsp.SR


def _peaky(y):
    """the share of the 300 Hz-12 kHz energy in its 15 strongest 23 Hz bins: about 0.2-0.5 for air, above 0.8 for a
    tone or a whistle"""
    P, f = _spectrum(y)
    k = (f >= 300) & (f < 12000)
    step = max(1, int(round(23.4 / (f[1] - f[0]))))
    b = P[k][:k.sum() // step * step].reshape(-1, step).sum(1)
    return float(np.sort(b)[::-1][:15].sum() / b.sum())


@pytest.mark.parametrize('seed', [1, 2])
def test_the_thump_of_something_appearing_is_short_low_and_dry(seed):
    y, meta = _take('thump/appear', seed)
    length, _ = _span20(y)
    assert 0.02 <= length <= 0.09 and meta['dur_s'] < 0.15             # about 50 ms, nothing rings after it
    assert _share(y, 0, 300) > 0.75                                     # a low body, not a pop or a bubble
    assert np.corrcoef(y[:, 0], y[:, 1])[0, 1] < 0.995                  # a little wide


def test_the_ui_tick_is_a_bright_dry_click():
    y, meta = _take('tick/soft', 1)
    assert meta['dur_s'] < 0.08 and _share(y, 0, 1000) < 0.05 and _centroid(y) > 3000.0


@pytest.mark.slow
@pytest.mark.parametrize('seed', [1, 2])
def test_title_and_logo_hits_are_short_low_and_swell_in(seed):
    title, logo, deep = _take('slam/title', seed)[0], _take('boom/short', seed)[0], _take('boom/deep', seed)[0]
    for y in (title, logo, deep):
        length, attack = _span20(y)
        assert 0.12 <= length <= 0.45 and attack >= 0.04, (length, attack)   # a quarter to a third of a second, with
        assert len(y) / dsp.SR < 0.6                                           # a short swell into it, then gone
    assert 0.3 < _share(title, 0, 100) < 0.85                           # half of the title hit under 100 Hz
    assert _share(logo, 0, 200) > 0.6 and _share(deep, 0, 200) > 0.7    # the logo and deep hits nearly all low
    assert _peaky(title) < 0.9                                          # a hit, not a tone


@pytest.mark.slow
@pytest.mark.parametrize('sid', ['slam/title', 'boom/short', 'boom/deep', 'hit/drop'])
def test_the_hits_have_no_skin_tone(sid):
    """the hits are heavy things slammed, not drums: nothing between 60 and 250 Hz still sounds 150 ms after the
    loudest moment (a drum's head rings there for a quarter second and longer)"""
    for seed in (1, 2):
        y = _take(sid, seed)[0]
        e = _env(dsp.butter(_mono(y), 'band', (60.0, 250.0)), 240)
        i = int(np.argmax(e))
        assert e[i + 30:].max(initial=0.0) < 0.1 * e[i], (sid, seed)


@pytest.mark.slow
def test_the_long_boom_only_falls():
    y, meta = _take('fall/sub', 1)
    assert 1.4 <= meta['dur_s'] <= 2.8
    e = _env(y, 2400)
    i = int(np.argmax(e))
    assert i < 0.2 * len(e) and np.all(np.diff(e[i + 2:]) < 0.1 * e.max())    # one fall (a rumble wavers on it)
    assert _share(y, 0, 160) > 0.8


@pytest.mark.slow
@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_the_transition_whoosh_is_air_and_keeps_its_length(seed):
    y, meta = _take('whoosh/soft', seed, dur=0.3)
    assert _peaky(y) < 0.7                                              # air, not a whistle or a hum
    assert 0.2 <= meta['dur_s'] <= 0.45, meta['dur_s']
    assert _peak_share(y, 0.1) > 0.45                                   # it swells, then is gone after its peak


@pytest.mark.slow
@pytest.mark.parametrize('sid', ['rise/air', 'rise/reverse'])
def test_risers_of_air_grow_louder_and_brighter_to_the_cut(sid):
    for seed in (1, 2):
        y = _raw(sid, seed)
        n = len(y) // 4
        q = [y[k * n:(k + 1) * n] for k in range(4)]
        lv = [float(np.sqrt(np.mean(p ** 2))) for p in q]
        assert lv[0] < lv[1] < lv[2] < lv[3], lv
        assert _centroid(q[3]) > 0.9 * _centroid(q[0])                   # never darker at the top
        assert _peaky(y) < 0.75


@pytest.mark.slow
def test_the_tape_stop_runs_down():
    y = _raw('stop/tape', 1)
    n = len(y) // 3
    assert _centroid(y[2 * n:]) < _centroid(y[:n])                       # pitch sinks as it slows
    assert _take('stop/tape', 1)[1]['dur_s'] > 1.0


@pytest.mark.slow
def test_booms_keep_the_low_end_of_their_drums():
    for sid in ('boom/deep', 'fall/sub', 'boom/short'):
        assert _share(_take(sid)[0], 0, 200) > 0.6, sid
    boom = _take('fall/sub')[0]
    assert _share(boom, 150, 24000) < 0.03 and _share(boom, 35, 80) > 0.25     # felt: no skin tone, a low weight
    assert _share(boom, 80, 160) > 0.03                              # and something a small speaker can play
    wide, short = _take('boom/wide')[0], _take('boom/short')[0]
    assert np.corrcoef(wide[:, 0], wide[:, 1])[0, 1] < np.corrcoef(short[:, 0], short[:, 1])[0, 1] - 0.1
    assert _take('boom/short')[1]['dur_s'] < 0.6 < 1.0 < _take('fall/sub')[1]['dur_s']


@pytest.mark.slow
def test_swooshes_arrive_and_leave():
    for seed in (1, 2):
        assert _peak_share(_raw('swoosh/in', seed)) > 0.6
        assert _peak_share(_raw('swoosh/out', seed)) < 0.3


@pytest.mark.slow
def test_a_pass_crosses_from_one_side_to_the_other():
    y = _raw('pass/air', 2)
    i = int(_peak_share(y, 0.1) * len(y))
    k = len(y) // 4
    lean = [float(np.log(np.sqrt(np.mean(p[:, 1] ** 2)) / np.sqrt(np.mean(p[:, 0] ** 2))))
            for p in (y[max(0, i - k):max(1, i - k // 4)], y[i + k // 4:i + k])]
    assert lean[0] * lean[1] < 0 and abs(lean[0] - lean[1]) > 0.4         # one side before it passes, the other after


@pytest.mark.slow
def test_the_reveal_swells_into_its_hit_and_the_braam_follows_its_root():
    y = _raw('sting/reveal', 1)
    e = _env(y, 240)
    hit = int(np.argmax(e))
    assert 0.5 < hit / len(e) < 0.9                                  # the swell leads, the hit stops soon after
    assert e[:hit // 3].max() < 0.3 * e.max()

    def chroma(y):                      # the energy of the 12 pitch classes between 40 and 500 Hz, held part
        m = _mono(y)[dsp.n_of(1.3):dsp.n_of(2.3)]
        P = np.abs(np.fft.rfft(m * np.hanning(len(m)), 1 << 18)) ** 2
        f = np.fft.rfftfreq(1 << 18, 1 / dsp.SR)
        k = (f > 40) & (f < 500)
        pc = np.round(12 * np.log2(f[k] / 440.0)).astype(int) % 12
        return np.bincount(pc, weights=P[k], minlength=12)
    a, d = chroma(_raw('braam/brass', 1, pitch='A1')), chroma(_raw('braam/brass', 1, pitch='D2'))
    shift = int(np.argmax([np.dot(np.roll(a, k), d) for k in range(12)]))
    assert shift == 5, shift                                       # the chord moved up a fourth with its root


def test_typing_run_follows_motion_and_dur():
    def keys(**req):
        e = _env(_raw('type/run', 2, **req), 480)
        return int(((e[1:] > 0.4 * e.max()) & (e[:-1] <= 0.4 * e.max())).sum())
    slow, fast = keys(dur=2.0, motion=-1.0), keys(dur=2.0, motion=1.0)
    assert 8 <= slow < fast <= 40
    assert keys(dur=1.0) < keys(dur=3.0)


def test_shimmer_sweeps_from_one_side_to_the_other():
    y = dsp.butter(_raw('shine/sweep', 2, motion=0.5), 'high', 3000.0)       # the grains, without the held tone
    i = int(_peak_share(y, 0.3) * len(y))
    k = len(y) // 4
    lean = [float(np.log(np.sqrt(np.mean(p[:, 1] ** 2)) / np.sqrt(np.mean(p[:, 0] ** 2))))
            for p in (y[max(0, i - k):i], y[i:i + k])]
    assert lean[0] * lean[1] < 0 and abs(lean[0] - lean[1]) > 0.5         # one side before its middle, the other after


def test_the_soft_sting_takes_a_voice_and_a_pitch():
    _, meta = _take('sting/soft', 1, pitch='A3')
    assert meta['voice'] == 'felt_piano' and meta['pitch'] == 'A3'


# ---- the film set: named recordings shaped to the roles motion films use
def test_the_named_recordings_are_in_the_store():
    """every recording a film design names by its file is one the store has (otherwise it plays the thing's first)"""
    import inspect
    import re
    from earshot.sounds import learned
    files = set(re.findall(r"'([a-z0-9_-]+-[0-9]+\.npz)'", inspect.getsource(motion)))
    have = {m['file'] for e in learned.index().values() for m in e['members']}
    assert len(files) >= 25 and files <= have, sorted(files - have)


def test_a_named_take_does_not_follow_the_order_of_the_store(monkeypatch):
    """the film set sounds the same when the store's order of a thing's recordings changes"""
    from earshot.sounds import learned

    def take(sid):
        d = registry.get(sid)
        c = registry.Controls(d)
        return np.asarray(d.fn(c, dsp.rng(registry.seed_of(sid, 0))))
    before = {sid: take(sid) for sid in ('whoosh/fast', 'thump/door', 'title/bag')}

    def backwards(sid, whole=False):
        return tuple(range(len(learned.index()[sid]['members'])))[::-1]
    monkeypatch.setattr(learned, 'order', backwards)
    monkeypatch.setattr(learned, 'pick', lambda sid, seed, bright=0.0: backwards(sid)[int(seed) % len(backwards(sid))])
    for sid, y in before.items():
        assert np.array_equal(take(sid), y), sid


@pytest.mark.parametrize('seed', [1, 2])
def test_the_film_set_thump_tick_and_click_are_short(seed):
    y, meta = _take('thump/door', seed)
    assert _span20(y)[0] <= 0.1 and meta['dur_s'] < 0.25 and _share(y, 0, 300) > 0.7      # a low body of about 60 ms
    for sid in ('tick/key', 'click/cursor'):
        y, meta = _take(sid, seed)
        assert _span20(y)[0] <= 0.05 and _share(y, 0, 1000) < 0.25 and _centroid(y) > 3000.0, sid   # bright and dry


@pytest.mark.parametrize('seed', [1, 2])
def test_the_film_set_slides_and_soft_whooshes_are_air(seed):
    for sid in ('whoosh/sweep', 'slide/paper'):
        y, _ = _take(sid, seed)
        length, _ = _span20(y)
        assert _peaky(y) < 0.5 and 0.2 <= length <= 0.4 and _peak_share(y, 0.1) > 0.45, sid   # swells, then gone


@pytest.mark.slow
@pytest.mark.parametrize('seed', [1, 2])
def test_the_film_set_hits_are_short_low_and_swell_in(seed):
    title, logo, impact = (_take(sid, seed)[0] for sid in ('title/bag', 'logo/bag', 'impact/car'))
    for y in (title, logo):
        length, attack = _span20(y)
        assert 0.12 <= length <= 0.45 and attack >= 0.08 and len(y) / dsp.SR < 0.6, (length, attack)
    assert 0.2 < _share(title, 0, 100) < 0.7 and _peaky(title) < 0.9           # half low, a hit and not a tone
    assert _share(logo, 0, 300) > 0.7 and _share(impact, 0, 300) > 0.7          # the logo and the impact nearly all low
    assert _span20(impact)[0] <= 0.3
