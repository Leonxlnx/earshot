"""Tests for the semantics modules (vocab, describe, match, search, compare). Fast tests use synthetic signals and fake
embeddings; tests marked slow load CLAP / AST (run with --slow)."""
import json

import numpy as np
import pytest

from earshot import compare, describe, features, match, search, vocab
from earshot.report import to_json

SR = 48000


def _fake_vocab(T=40, seed=0):
    rng = np.random.RandomState(seed)
    v = vocab.Vocab.__new__(vocab.Vocab)
    v.names = [f'term {i}' for i in range(T)]
    v.cats = {'a': v.names[:T // 2], 'b': v.names[T // 2:]}
    v.cat_of = {t: ('a' if i < T // 2 else 'b') for i, t in enumerate(v.names)}
    v.E = vocab._norm(rng.randn(T, 512).astype(np.float32))
    v.m = vocab._norm(rng.randn(512).astype(np.float32)) * 0.3
    v.mu_E = v.bias(v.E)
    return v


def _clicks(times, dur=4.0, amp=0.5, seed=1):
    rng = np.random.RandomState(seed)
    x = rng.randn(int(dur * SR)).astype(np.float32) * 1e-4
    n = int(0.08 * SR)
    burst = rng.randn(n) * np.exp(-np.arange(n) / SR / 0.008) * amp
    for t in times:
        a = int(t * SR)
        x[a:a + n] += burst.astype(np.float32)
    return x


# ---------------------------------------------------------------------------------------------------- vocab
def test_vocab_json_schema():
    d = vocab.load_vocab()
    assert set(d) >= {'templates', 'categories'}
    assert all('{}' in t for t in d['templates'])
    terms = [t for ts in d['categories'].values() for t in ts]
    assert len(terms) == len(set(terms)) and len(terms) > 200
    assert set(vocab.UNVALIDATED) <= set(d['categories'])


def test_stats_file_matches_vocab_and_model():
    s = vocab.load_stats()
    assert s is not None and len(s['mean']) == 512 and s['pad'] == vocab.PAD
    assert set(vocab.load_vocab()['templates']) == set(s['templates'])
    assert s['n_ref'] >= 100


def test_spans_for():
    assert vocab.spans_for(0.0, 3.0) == [(0.0, 3.0)]
    sp = vocab.spans_for(0.0, 25.0, max_win=None)
    assert [round(a, 3) for a, _ in sp] == [0.0, 5.0, 10.0, 15.0] and sp[-1][1] == pytest.approx(25.0)
    sp = vocab.spans_for(2.0, 12.5, max_win=None)        # last window flush with the end
    assert sp[0] == (2.0, 12.0) and sp[-1][1] == pytest.approx(12.5)
    sp = vocab.spans_for(0.0, 170.0, max_win=6)
    assert len(sp) == 6 and sp[0][0] == 0.0 and sp[-1][1] == pytest.approx(170.0)
    assert all(b - a == pytest.approx(10.0) for a, b in sp)


def test_group_weights_balance_folders(tmp_path):
    paths = [str(tmp_path / 'a' / f'{i}.wav') for i in range(4)] + [str(tmp_path / 'b' / 'x.wav')]
    w = vocab.group_weights(paths)
    assert w[:4].sum() == pytest.approx(w[4])


def test_calibrate_math():
    v = _fake_vocab()
    rng = np.random.RandomState(3)
    A = vocab._norm(rng.randn(5, 512).astype(np.float32))
    r = v.calibrate(A, v.E[:3], 'both')
    assert np.allclose(r['z_t'].mean(1), 0, atol=1e-5) and np.allclose(r['z_t'].std(1), 1, atol=1e-4)
    assert np.allclose(r['z_q'], r['z_t'][:, :3], atol=1e-5)      # a query equal to a term scores like the term
    assert np.allclose(r['cos_q'], A @ v.E[:3].T)
    raw = v.calibrate(A, v.E[:3], 'raw')
    assert np.allclose(raw['z_q'], raw['cos_q'])
    c = v.calibrate(A, None, 'center')
    assert np.allclose(c['u_t'], A @ v.E.T - v.E @ v.m, atol=1e-5)
    assert ((r['pct_q'] >= 0) & (r['pct_q'] <= 1)).all()


def test_top_terms_filters_categories():
    v = _fake_vocab()
    a = v.E[25] * 0.9 + v.E[3] * 0.1
    a /= np.linalg.norm(a)
    top = v.top_terms(a, k=3)
    assert top[0][0] == 'term 25' and top[0][2] > 2
    assert all(c == 'a' for _, c, _, _ in v.top_terms(a, k=5, cats=('a',)))
    assert all(c != 'b' for _, c, _, _ in v.top_terms(a, k=5, exclude=('b',)))


# ---------------------------------------------------------------------------------------------------- match
def test_verdict_thresholds_ordered():
    zs = [t for t, _ in match.VERDICTS]
    assert zs == sorted(zs, reverse=True)
    assert match.verdict(zs[0] + 0.01) == 'strong' and match.verdict(-5) == 'no match'
    assert set(match.VERDICT_EVIDENCE) == {name for _, name in match.VERDICTS} | {'no match'}
    lrs = [match.VERDICT_EVIDENCE[n]['lr'] for _, n in match.VERDICTS] + [match.VERDICT_EVIDENCE['no match']['lr']]
    assert lrs == sorted(lrs, reverse=True) and lrs[-1] < 1      # bands, each stronger evidence than the next


def test_z_text_never_contradicts_verdict():
    assert vocab.zf(2.96) == '+2.9' and vocab.zf(3.0) == '+3.0' and vocab.zf(-0.04) == '-0.1'
    for z in np.arange(1.5, 3.5, 0.01):
        assert match.verdict(float(vocab.zf(z))) == match.verdict(z) or float(vocab.zf(z)) < z


def test_check_rejects_bad_ranges(wav, click):
    p = wav(click)
    assert vocab.check(p) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        vocab.check(p, start=1.0)
    with pytest.raises(ValueError):
        vocab.check(p, start=0.1, end=0.05)
    with pytest.raises(Exception):
        vocab.check(p + '.missing.wav')
    with pytest.raises(ValueError):
        match.match(p, [])


def test_readable_skips_bad_files(wav, click, tmp_path):
    bad = tmp_path / 'not_audio.wav'
    bad.write_bytes(b'this is not audio')
    ok, skipped = vocab.readable([wav(click), str(bad)])
    assert len(ok) == 1 and skipped[0]['path'] == str(bad) and skipped[0]['error']


def test_softmax_is_a_distribution():
    p = match.softmax([0.3, 0.25, 0.1], 38.7)
    assert p.sum() == pytest.approx(1.0) and p[0] > p[1] > p[2]


# ---------------------------------------------------------------------------------------------------- search
def test_event_candidates_land_on_clicks():
    times = [0.5, 1.5, 2.6]
    x = _clicks(times)
    c = search.event_candidates(x)
    strokes = sorted(s for s in c if s[2] == 'stroke')
    assert len(strokes) == 3
    for (s, e, _, _), t in zip(strokes, times):
        assert t - 0.015 <= s <= t + 0.002          # trim starts just before the attack, never after it
        assert 0.02 < e - s < 0.5                    # ends after the decay, long before the next click


def test_down_up_strokes_are_paired():
    x = _clicks([0.5, 0.62, 1.8, 1.92], amp=0.5)
    c = search.event_candidates(x)
    pairs = sorted((round(s, 2), round(e, 2)) for s, e, k, _ in c if k == 'event')
    assert len(pairs) == 2 and pairs[0][0] == pytest.approx(0.49, abs=0.02) and pairs[0][1] > 0.62


def test_select_round_robin_covers_time():
    cands = [(t, t + 0.1, 'stroke', 50.0) for t in np.arange(0, 4, 0.05)] + [(10.0, 10.1, 'stroke', 1.0)]
    sel = search._select(cands, n=6, bucket=2.0)
    assert len(sel) == 6 and any(s[0] >= 10 for s in sel)


def test_window_candidates_cover_everything():
    w = search.window_candidates(18.0)
    assert w[0][:2] == (0.0, 3.0) and w[-1][1] == pytest.approx(18.0)
    assert search.window_candidates(0.5) == [(0.0, 0.5, 'window', 0.0)]
    for dur in (44.0, 300.0, 600.0, 1800.0):              # no gaps anywhere, whatever the length
        w = search.window_candidates(dur)
        assert w[0][0] == 0.0 and w[-1][1] == pytest.approx(dur)
        assert all(b[0] <= a[1] + 1e-6 for a, b in zip(w, w[1:]))
        assert all(b - a <= search.MAX_WIN_S + 1e-6 for a, b, _, _ in w)
    assert len(search.window_candidates(300.0)) <= search.MAX_WINDOWS
    w = search.window_candidates(12.0, t0=5.1)               # a --start/--end range in local time
    assert w[0][0] == 5.1 and w[-1][1] == pytest.approx(12.0)


def test_event_candidates_span_a_long_file():
    x = _clicks(list(np.arange(0.5, 299, 1.0)), dur=300.0, amp=0.3)
    allc = search.event_candidates(x, n=None)
    assert max(c[0] for c in allc) > 290                    # every onset is a candidate before any selection
    assert len(search.event_candidates(x)) <= search.MAX_EVENTS


def test_dedupe():
    c = [(0.0, 1.0, 'stroke', 1), (0.01, 1.0, 'event', 1), (2.0, 3.0, 'stroke', 1)]
    assert len(search._dedupe(c)) == 2


def test_tail_db():
    x = _clicks([0.5])
    env = search._env_db(x)
    assert search._tail_db(env, 0.495, 0.7) < -30
    assert search._tail_db(env, 0.495, 0.505) > -15


# ---------------------------------------------------------------------------------------------------- compare / describe
def _f(x):
    return compare.feats(features.measure(x, light=True), x)


def test_differences_in_words(click, sine):
    fa, fb = _f(sine * 0.1), _f(click)
    keys, txt = compare.differences(fa, fb)
    assert 'brightness_hz' in keys and 'brighter (brightness' in txt and len(keys) <= 4
    keys, txt = compare.differences(fa, dict(fa, loudness_lufs=fa['loudness_lufs'] + 6))
    assert keys == ['loudness_lufs'] and txt.startswith('6 dB louder')
    assert compare.differences(fa, fa)[0] == []


def test_trims_of_the_same_sound_read_the_same(click):
    short = click[: int(0.05 * SR)]
    for pad_s in (0.3, 0.5, 2.0):                           # extra silence after (and before) the same click
        padded = np.concatenate([np.zeros(int(0.1 * SR), np.float32), short, np.zeros(int(pad_s * SR), np.float32)])
        keys, txt = compare.differences(_f(short), _f(padded))
        assert keys == [], (pad_s, txt)
    st = np.stack([short, short], axis=1)                    # dual mono = mono: no width difference
    assert compare.differences(_f(short), _f(st))[0] == []
    assert _f(short)['width_db'] == compare.MONO_DB and 'noise_floor_dbfs' not in _f(short)


def test_closer_is_none_when_too_close():
    assert compare._closer(2.9, 2.91) == (None, 'Neither is closer')
    assert compare._closer(1.0, 2.5)[0] == 'b' and compare._closer(2.0, 1.6)[0] == 'a'


def test_typicality_flags_outlier():
    rng = np.random.RandomState(0)
    c = vocab._norm(rng.randn(512))
    R = vocab._norm(c + 0.3 * vocab._norm(rng.randn(12, 512)))
    inlier = vocab._norm(c + 0.3 * vocab._norm(rng.randn(512)))
    outlier = vocab._norm(rng.randn(512))
    assert compare._typicality(inlier, R, 3)[1] > -1
    assert compare._typicality(outlier, R, 3)[1] < -2


def test_shape_words(click, sine):
    assert 'short transient' in describe.shape(features.measure(click))
    s = describe.shape(features.measure(sine))
    assert 'tonal' in s and 'A5' in s


def test_distinct_windows_do_not_overlap_or_repeat():
    rng = np.random.RandomState(0)
    W = vocab._norm(np.repeat(rng.randn(1, 512), 6, axis=0) + 0.01 * rng.randn(6, 512))
    W[4] = vocab._norm(rng.randn(512))
    spans = [(5.0 * i, 5.0 * i + 10) for i in range(6)]
    tags = ['x', 'x', 'y', 'x', 'z', 'x']
    idx = describe.distinct_windows(spans, W, tags, n=3)
    assert 4 in idx and len(idx) >= 2
    assert all(spans[a][1] <= spans[b][0] for a, b in zip(idx, idx[1:]))
    assert len({tags[i] for i in idx}) == len(idx)
    assert describe.distinct_windows(spans, W, ['x'] * 6) == []      # uniform: nothing reads differently


def test_measurements_hide_what_is_not_measurable(click):
    m = describe.measurements(features.measure(click))
    assert m['width'] == 'mono' and 'noise_floor_dbfs' not in m and 'centroid_hz' not in m


# ---------------------------------------------------------------------------------------------------- with models
@pytest.mark.slow
def test_match_and_describe_end_to_end(wav, click):
    p = wav(np.concatenate([click, np.zeros(SR // 4, np.float32)]))
    r = match.match(p, ['a short click', 'a choir singing'], alts=['music'])
    json.loads(to_json(r))
    assert {'query', 'cosine', 'z', 'percentile', 'rank_in_vocab', 'p_vs_alts', 'verdict', 'summary'} <= set(r)
    assert r['results'][0]['z'] >= r['results'][1]['z']
    d = describe.describe(p)
    json.loads(to_json(d))
    assert d['top_tags'] and d['audioset'] and 'Sounds most like' in d['summary']
    assert all(t['category'] not in vocab.UNVALIDATED for t in d['top_tags'])
    tiny = describe.describe(wav(click[:100], 'tiny.wav'))
    assert 'too short' in tiny['summary'] and tiny['top_tags'] == []
    silent = describe.describe(wav(np.zeros(SR, np.float32), 'silence.wav'))
    assert 'near-silence' in silent['summary']


@pytest.mark.slow
def test_rank_skips_unreadable_and_flags_silence(wav, click, tmp_path):
    wav(click, 'click.wav')
    wav(np.zeros(SR, np.float32), 'silence.wav')
    (tmp_path / 'broken.wav').write_bytes(b'not audio at all')
    r = search.rank([str(tmp_path)], 'a short click')
    json.loads(to_json(r))
    assert r['n_files'] == 2 and len(r['skipped']) == 1
    assert r['results'][-1]['name'] == 'silence.wav' and r['results'][-1]['near_silent']
    assert 'unreadable' in r['summary'] and 'near-silent' in r['summary']


@pytest.mark.slow
def test_find_segments_hold_the_clicks(wav):
    times = [1.0, 3.0, 5.0]
    r = search.find(wav(_clicks(times, dur=6.0)), 'a single click', top=3)
    json.loads(to_json(r))
    assert r['query_kind'] == 'event' and len(r['segments']) == 3
    held = sorted(next(t for t in times if s['start'] <= t < s['end']) for s in r['segments'])
    assert held == times                                  # three non-overlapping segments, one per click
    r = search.find(wav(_clicks(times, dur=6.0), 'r.wav'), 'a single click', top=3, start=2.9, end=6.0)
    assert r['segments'] and all(s['start'] >= 2.9 - search.PRE_ROLL - 1e-3 for s in r['segments'])
