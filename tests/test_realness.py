"""realness: synthetic signals only; model-free unless marked slow (the CLAP model is stubbed)."""
import json

import numpy as np
import pytest

from earshot import realness as rl

SR = 48000


def tone(f=880.0, dur=1.0, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.4)).astype(np.float32)


def noisy_hit(dur=1.0, seed=0):
    """decaying noise transients over a -70 dBFS-ish noise floor (microphone-like)"""
    rng = np.random.RandomState(seed)
    n = int(dur * SR)
    x = rng.randn(n) * 3e-4
    for t0 in (0.1, 0.45):
        a, k = int(t0 * SR), int(0.2 * SR)
        x[a:a + k] += rng.randn(k) * np.exp(-np.arange(k) / SR / 0.02) * 0.4
    return x.astype(np.float32)


class FakeClap:
    """an embedding that drives the CLAP head to score +4 for rough (noise-like) clips and -4 for smooth (tonal) ones,
    and fixed unit text vectors"""
    def __init__(self, m):
        h = m['heads']['clap']
        self.mean, self.scale, self.coef = (np.asarray(h[k], np.float64) for k in ('mean', 'scale', 'coef'))
        self.b = h['intercept']

    def embed_audio(self, xs, **kw):
        out = []
        for x in xs:
            k = (4.0 if np.std(np.diff(x)) > 0.5 * np.std(x) else -4.0) - self.b
            out.append(self.mean + self.scale * self.coef * k / (self.coef @ self.coef))
        return np.stack(out).astype(np.float32)

    def embed_text(self, texts):
        rng = np.random.RandomState(1)
        v = rng.randn(len(texts), len(self.mean))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def fake_clap(monkeypatch):
    from earshot.models import clap
    monkeypatch.setattr(clap, 'get', lambda: FakeClap(rl.load_model()))


def test_model_is_plain_json_and_complete():
    m = json.load(open(rl.MODEL_PATH, encoding='utf-8'))
    assert set(m['heads']) == {'clap', 'signal'} and m['fusion']['heads'] == ['clap', 'signal']
    assert len(m['heads']['clap']['coef']) == 512
    assert m['heads']['signal']['inputs'] == list(m['heads']['signal']['inputs'])
    assert len(m['heads']['signal']['coef']) == len(m['heads']['signal']['inputs'])
    assert 0 < m['thresholds']['lo'] < 0.5 <= m['thresholds']['hi'] < 1
    ft = m['file_thresholds']
    assert m['thresholds']['lo'] <= ft['lo'] < 0.5 <= m['thresholds']['hi'] <= ft['hi'] < 1 and ft['synth_files'] > 0
    assert m['long_s'] == rl.LONG_S and m['win_s'] == rl.WIN_S


def test_signal_features_separate_tone_from_recording():
    ft, fr = rl.signal_features(tone()), rl.signal_features(noisy_hit())
    assert set(ft) == set(rl.load_model()['heads']['signal']['inputs'])
    assert ft['tonal_frames'] > 0.8 > fr['tonal_frames']
    assert ft['line_share'] > fr['line_share']
    assert ft['spec_floor_db'] < fr['spec_floor_db']
    assert ft['pitch_wobble'] < -1.0


def test_signal_features_ignore_level():
    a, b = rl.signal_features(noisy_hit()), rl.signal_features(noisy_hit() * 0.1)
    for k in a:
        if k != 'silence_frac' and np.isfinite(a[k]):
            assert a[k] == pytest.approx(b[k], abs=1e-3), k


def test_silent_and_tiny_clips_are_not_judged():
    assert rl.signal_features(np.zeros(SR, np.float32)) is None
    assert rl.signal_features(np.ones(100, np.float32) * 0.1) is None
    assert not rl.audible(np.full(SR, 1e-4, np.float32)) and rl.audible(tone())


def test_clips_windows():
    assert len(rl._clips(np.zeros(5 * SR))) == 1
    spans = rl._clips(np.zeros(int(13.55 * SR)))
    assert [round(s[0], 2) for s in spans] == [0.0, 4.0, 8.0, 9.55]
    spans = rl._clips(np.zeros(int(60.01 * SR)))
    assert len(spans) == 15 and spans[-1][0] == 56.0


def test_head_logit_and_fuse():
    m = rl.load_model()
    h = m['heads']['signal']
    logit, z = rl.head_logit(h, h['mean'])
    assert np.allclose(z, 0) and logit == pytest.approx(h['intercept'])
    p, s = rl.fuse(m['fusion'], {'clap': 2.0, 'signal': -1.0})
    assert 0 < p < 1 and p == pytest.approx(1 / (1 + np.exp(-s)))
    t = m['thresholds']
    assert rl.verdict_of(t['hi'], m) == 'real or pro-produced'
    assert rl.verdict_of(t['lo'], m) == 'likely code-synthesized'
    assert rl.verdict_of((t['lo'] + t['hi']) / 2, m) == 'uncertain'
    ft = m['file_thresholds']
    assert rl.verdict_of(ft['lo'], m, file=True) == 'likely code-synthesized'
    assert rl.verdict_of(ft['lo'] + 0.01, m, file=True) == 'uncertain'


def test_digital_silence_is_never_a_verdict_reason():
    m = rl.load_model()
    assert all(q['feature'] != 'silence_frac' for q in m['facts'])
    assert all(q['precision'] >= 0.9 for q in m['facts'])
    assert rl._silence_line(m, {'silence_frac': 0.01}) is None
    line = rl._silence_line(m, {'silence_frac': 0.38})
    assert 'weak evidence' in line and 'real/pro' in line and 'no room tone' not in line
    facts = rl._facts(m, {'silence_frac': 0.99, 'crest_db': 3.0})
    assert facts and all('silence' not in t for _, _, t in facts)


def test_realness_silent_file_needs_no_model(wav):
    r = rl.realness(wav(np.zeros(SR, np.float32)), cache=False)
    assert r['verdict'] == 'silent' and r['p_real'] is None and 'silent' in r['summary']


def noise(seconds, amp=0.1):
    return (np.random.RandomState(3).randn(int(seconds * SR)) * amp).astype(np.float32)


def test_realness_structure_with_stub(wav, fake_clap):
    r = rl.realness(wav(tone(dur=1.0)), cache=False)
    assert r['p_real'] < 0.5 and r['verdict'] in ('uncertain', 'likely code-synthesized')
    assert r['evidence'][0].startswith('CLAP head') and 'segments' not in r
    assert 'closest reference prompt' in r['evidence'][0] and 'weak context' in r['evidence'][0]
    assert 'nearest description' not in r['summary'] and 'calibrated on held-out clips' in r['note']
    json.dumps(r)
    x = np.concatenate([noisy_hit(dur=4.0), np.zeros(int(0.5 * SR), np.float32), tone(dur=4.0)])
    r = rl.realness(wav(x, 'long.wav'), cache=False)
    assert len(r['segments']) == 3 and r['segments'][-1]['end'] == pytest.approx(8.5, abs=1e-3)
    assert r['evidence'][0].startswith('evidence below is from the most')
    assert 'windows' in r['summary'] and r['summary'].isascii() and 'file-level thresholds' in r['note']
    text = rl._text(r)
    assert text.isascii() and 'p_real' in text


def test_long_file_reports_lean_windows_in_file_time(wav, fake_clap):
    x = np.concatenate([np.zeros(2 * SR, np.float32), tone(dur=4.0), noise(4.0), tone(dur=4.0, f=660.0)])
    path = wav(x, 'tnt.wav')
    r = rl.realness(path, cache=False)
    segs = r['segments']
    assert [s['start'] for s in segs] == [0.0, 4.0, 8.0, 10.0]
    assert r['lean_synth_windows'] == sum(s['p_real'] is not None and s['p_real'] < 0.5 for s in segs) >= 2
    assert 'lean synthesized' in r['summary'] and 'Leaning synthesized:' in r['summary']
    r = rl.realness(path, start=2.0, end=14.0, cache=False)
    assert r['range'] == [2.0, 14.0] and [s['start'] for s in r['segments']] == [2.0, 6.0, 10.0]
    assert r['segments'][1]['p_real'] > 0.5 > r['segments'][0]['p_real']
    assert '[0:02.00-0:14.00]' in r['summary'] and '0:02.00-0:06.00' in r['summary']
    clean = rl.realness(wav(noise(9.0), 'n.wav'), cache=False)
    assert clean['lean_synth_windows'] == 0 and 'does not rule out a code-made layer' in clean['summary']


def test_bad_ranges_raise_clear_errors(wav):
    path = wav(tone(dur=1.5), 'short.wav')
    with pytest.raises(ValueError, match='beyond the end'):
        rl.realness(path, start=5, end=6, cache=False)
    with pytest.raises(ValueError, match='greater than start'):
        rl.realness(path, start=1, end=0.5, cache=False)


def test_many_counts_silent_error_and_missing_separately(wav, tmp_path, fake_clap):
    bad = tmp_path / 'corrupt.wav'
    bad.write_text('not audio')
    many = rl.realness_many([wav(noise(1.0), 'a.wav'), wav(np.zeros(SR, np.float32), 'b.wav'), str(bad),
                             str(tmp_path / 'nope.wav')])
    v = [f['verdict'] for f in many['files']]
    assert v.count('silent') == 1 and v.count('error') == 2 and many['counts']['real or pro-produced'] <= 1
    assert '1 silent (not judged)' in many['summary'] and '2 could not be read' in many['summary']
    assert many['summary'].startswith('4 files:')
    table = rl._table(many)
    assert table.isascii() and len(table.splitlines()) == 7   # summary, thresholds line, header, 4 rows
    assert 'b.wav: b.wav' not in table and 'no such file or folder' in table
    assert 'Verdict thresholds' in table.splitlines()[1]


def test_many_applies_the_range_to_every_file(wav, fake_clap):
    many = rl.realness_many([wav(noise(2.0), 'a.wav'), wav(tone(dur=2.0), 'b.wav')], start=0.5, end=1.0)
    assert all(f['range'] == [0.5, 1.0] for f in many['files']) and 'Range 0:00.50-0:01.00' in many['summary']


def test_cli_missing_path_is_an_error(tmp_path, capsys):
    from earshot import cli
    assert cli.main(['realness', str(tmp_path / 'nope.wav')]) == 1
    assert 'no such file or folder' in capsys.readouterr().err


@pytest.mark.slow
def test_realness_flags_a_code_beep(wav):
    t = np.arange(int(0.6 * SR)) / SR
    x = np.r_[np.zeros(int(0.1 * SR)), 0.5 * np.sin(2 * np.pi * 880 * t) * np.exp(-t / 0.15), np.zeros(int(0.3 * SR))]
    r = rl.realness(wav(x.astype(np.float32), 'beep.wav'), cache=False)
    assert r['p_real'] < 0.5 and r['verdict'] != 'real or pro-produced'
    assert any('CLAP head' in e for e in r['evidence'])


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def test_quiet_sources_are_lifted_smoothly_not_called_silent():
    x = noise(1.0)

    def at(db):
        return (x * 10 ** ((db - rms_db(x)) / 20)).astype(np.float32)
    y, g, rdb, pdb = rl.lift_quiet(at(-75))
    assert rdb < rl.SILENT_DB and rl.audible(y) and rms_db(y) == pytest.approx(rl.LIFT_BELOW_DB, abs=0.01)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    assert rl.lift_quiet(at(rl.LIFT_BELOW_DB - 0.5))[1] == pytest.approx(0.5, abs=0.01)   # no cliff
    assert rl.lift_quiet(at(rl.LIFT_BELOW_DB + 0.5))[1] == 0.0
    assert rl.lift_quiet(np.zeros(SR, np.float32))[1] == 0.0
