"""Launch profile + critique: detectors on synthetic signals, scorecard rules, end-to-end on generated files."""
import json
import os
import subprocess

import numpy as np
import pytest

from earshot import config, video
from earshot import profile as P
from earshot.critique import SPEC, critique, real_thresholds, scorecard

SR = 48000


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    monkeypatch.setattr(config, 'DEC_CACHE', str(tmp_path / 'dec'))


def _bed(sec, level=0.05, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.randn(int(sec * SR)) * level).astype(np.float32)


def _hit(x, t, amp=0.8, seed=1):
    rng = np.random.RandomState(seed)
    n = int(0.15 * SR)
    i = int(t * SR)
    x[i:i + n] += (rng.randn(n) * amp * np.exp(-np.arange(n) / SR / 0.03)).astype(np.float32)
    return x


def _soundtrack(sec=12.0, hits=(2.0, 5.0, 8.0), lead=0.0, hole=None):
    x = _bed(sec)
    for t in hits:
        _hit(x, t)
    x = np.stack([x, 0.9 * x + 0.1 * _bed(sec, 0.05, 3)], axis=1)
    if lead:
        x[: int(lead * SR)] = 0
    if hole:
        x[int(hole[0] * SR): int(hole[1] * SR)] = 0
    return x


def test_salient_hits_and_stopdown(wav):
    x = _soundtrack(hits=(2.0, 5.0, 8.0))
    x[int(4.5 * SR): int(5.0 * SR)] *= 0.01  # 0.5 s stop-down right before the hit at 5.0
    a = P._audio_pass(wav(x))
    hits = P.salient_hits(a['tf'], a['Lf'], a['meas']['lufs'], a['env'], a['e2'])
    assert len(hits) == 3
    for t in (2.0, 5.0, 8.0):
        assert np.min(np.abs(hits - t)) < 0.006
        assert np.min(np.abs(a['onsets'] - t)) < 0.006
    sd = P.stopdowns(a['tf'], a['Lf'])
    assert len(sd) == 1 and abs(sd[0][0] - 4.5) < 0.1 and 0.35 < sd[0][1] < 0.6


def test_visual_events_relative_threshold():
    fps = 30.0
    d = np.full(300, 0.003, np.float32)
    d[100:140] = 0.02 + 0.005 * np.sin(np.arange(40))  # steady motion: no event
    d[60] = 0.4  # hard cut
    d[200] = 0.05  # small change on a still frame: an event
    ev = video.diff_events(d, fps)
    assert list(ev) == [60, 200]


def test_near_and_chance():
    ev = [1.0, 2.0, 3.0]
    assert P.near_frac(ev, [1.02, 2.5, 3.09], 0.1) == pytest.approx(2 / 3)
    assert P.near_frac(ev, [], 0.1) == 0.0 and P.near_frac([], [1.0], 0.1) is None
    dense = np.arange(0, 10, 0.05)
    assert P.chance_frac(ev, dense, 0.1, 10.0) == pytest.approx(1.0)


def test_film_metrics_arc_gap_and_ending(wav):
    x = _soundtrack(sec=12.0, hits=(2.0, 6.0, 11.0), lead=1.0, hole=(4.0, 5.0))
    x[int(11.4 * SR):] = 0
    f = P.film_metrics(wav(x), use_ast=False)
    m = f['metrics']
    assert m['first_sound_s'] == pytest.approx(1.0, abs=0.06)
    assert m['max_gap_s'] == pytest.approx(1.0, abs=0.08)
    assert m['end_silence_s'] == pytest.approx(0.6, abs=0.08)
    assert m['music_frac'] is None and m['cuts_with_sound'] is None
    assert m['hit_rate'] == pytest.approx(5 / 12, abs=0.01)  # 3 hits + sound starting at 1.0 and after the hole
    json.dumps(f)


def _prof(**stats):
    return {'n': 7, 'metrics': {k: {'n': 7, 'min': lo, 'max': hi, 'median': (lo + hi) / 2, 'p10': lo, 'p90': hi}
                                for k, (lo, hi) in stats.items()}}


def test_scorecard_rules():
    prof = _prof(onset_rate=(2.5, 8.0), music_frac=(0.67, 1.0), first_sound_s=(0.0, 0.5))
    vals = {'lufs_i': -14.0, 'true_peak': -0.5, 'onset_rate': 1.0, 'music_frac': 0.3, 'first_sound_s': 0.0}
    rows = {r['metric']: r for r in scorecard(vals, prof)['metrics']}
    assert rows['lufs_i']['status'] == 'ok' and rows['lufs_i']['basis'] == 'web'
    assert rows['true_peak']['status'] == 'high'
    assert {r['metric']: r for r in scorecard(vals, prof, lossy=True)['metrics']}['true_peak']['status'] == 'ok'
    assert rows['onset_rate']['status'] == 'low' and rows['onset_rate']['range'] == [pytest.approx(2.2), pytest.approx(8.3)]
    assert rows['music_frac']['status'] == 'low' and rows['music_frac']['weight'] == 'critical'
    assert rows['first_sound_s']['range'][0] is None and rows['first_sound_s']['status'] == 'ok'
    assert rows['band_sub']['status'] == 'n.a.' and rows['tempo']['weight'] == 'info'
    near = {r['metric']: r for r in scorecard({'lufs_i': -17.5}, prof)['metrics']}['lufs_i']
    far = {r['metric']: r for r in scorecard({'lufs_i': -25.0}, prof)['metrics']}['lufs_i']
    assert near['weight'] == 'major' and far['weight'] == 'critical'
    sc = scorecard({'lufs_i': -14.0, 'onset_rate': 5.0, 'music_frac': 0.9}, prof)
    assert sc['verdict'] == 'launch-grade' and sc['score'] == 1.0
    assert set(SPEC) - {'p_real'} <= set(P.METRICS)


def test_scorecard_realness_and_density_floor():
    prof = _prof(hit_rate=(0.1, 1.66), music_frac=(0.67, 1.0))
    t = real_thresholds()
    row = lambda vals, k: {r['metric']: r for r in scorecard(vals, prof)['metrics']}[k]
    synth = row({'p_real': 0.3}, 'p_real')
    assert synth['status'] == 'low' and synth['weight'] == 'critical' and synth['range'][0] == t['hi']
    assert scorecard({'p_real': 0.3, 'lufs_i': -14.0, 'music_frac': 1.0}, prof)['verdict'] == 'needs work'
    if t['hi'] > 0.55:
        assert row({'p_real': 0.52}, 'p_real')['weight'] == 'minor'
    assert row({'p_real': 0.99}, 'p_real')['status'] == 'ok'
    floor = row({'hit_rate': 0.0}, 'hit_rate')
    assert floor['status'] == 'low' and floor['range'][0] == pytest.approx(0.05)


def test_bundled_profile_is_anonymous():
    p = P.load_profile()
    assert p['n'] == 7 and [f['id'] for f in p['films']] == [f'ref_{i}' for i in range(1, 8)]
    s = json.dumps(p)
    assert '.mp4' not in s and 'Downloads' not in s and 'path' not in s
    for k, spec in SPEC.items():
        if spec[0] and k != 'p_real':
            assert p['metrics'][k]['n'] >= 6, k


def test_critique_end_to_end_audio_only(wav):
    path = wav(_soundtrack(sec=12.0, hits=(2.0, 6.0, 9.0), lead=1.5))
    r = critique(path, use_ast=False, use_realness=False)
    json.dumps(r)
    assert r['verdict'] in ('launch-grade', 'close', 'needs work', 'not launch-grade')
    rows = {m['metric']: m for m in r['metrics']}
    assert rows['first_sound_s']['status'] == 'high'
    assert any(a['metric'] == 'first_sound_s' and '0:01.5' in a['action'] for a in r['actions'])
    assert rows['cuts_with_sound']['status'] == 'n.a.' and 'cuts_with_sound' in r['summary']
    r['text'].encode('ascii')


def test_critique_refuses_silent_and_tiny(wav):
    for x in (np.zeros(8 * SR, np.float32), _bed(1.0)):
        path = wav(x)
        r = critique(path, use_realness=False)
        assert r['verdict'] == 'n.a.' and not r['actions'] and 'nothing to score' in r['summary']
        assert not os.path.exists(P.cache_paths(path)['audio'])  # refused before the librosa pass


def test_critique_mono_and_short_clip(wav):
    r = critique(wav(_bed(20.0)[:, None].repeat(2, axis=1)), use_ast=False, use_realness=False)
    rows = {m['metric']: m for m in r['metrics']}
    assert rows['side_to_mid_db']['value'] == P.MONO_DB and rows['side_to_mid_db']['status'] == 'low'
    assert any(a['metric'] == 'side_to_mid_db' and 'mono' in a['action'] for a in r['actions'])
    r = critique(wav(_soundtrack(sec=6.0, hits=(2.0, 4.0))), use_ast=False, use_realness=False)
    rows = {m['metric']: m for m in r['metrics']}
    assert all(rows[k]['status'] == 'n.a.' for k in P.SHAPE) and 'clip under 10 s' in r['summary']
    assert r['partial'] == ['realness', 'music presence'] and 'partial' in r['text'].splitlines()[1]


def test_input_errors(wav, tmp_path, capsys):
    from earshot.cli import main
    a = wav(_soundtrack(sec=8.0, hits=(2.0, 5.0)))
    with pytest.raises(P.InputError, match='file not found'):
        critique(str(tmp_path / 'nope.wav'), use_ast=False, use_realness=False)
    with pytest.raises(P.InputError, match='folder'):
        critique(str(tmp_path), use_ast=False, use_realness=False)
    with pytest.raises(P.InputError, match='has no picture'):
        critique(a, video=wav(_bed(4.0), 'other.wav'), use_ast=False, use_realness=False)
    assert critique(a, video=a, use_ast=False, use_realness=False)['video'] is None  # "its own picture": none
    with pytest.raises(P.InputError, match='profile not found'):
        critique(a, profile=str(tmp_path / 'p.json'), use_ast=False, use_realness=False)
    bad = tmp_path / 'bad.json'
    bad.write_text('{"n": 1}')
    with pytest.raises(P.InputError, match='not a profile'):
        critique(a, profile=str(bad), use_ast=False, use_realness=False)
    assert main(['critique', str(tmp_path / 'nope.wav'), '--fast']) == 1
    assert capsys.readouterr().err.strip() == f"earshot critique: error: file not found: {tmp_path / 'nope.wav'}"


def test_profile_from_two_files(wav, tmp_path):
    a = wav(_soundtrack(sec=10.0, hits=(2.0, 4.0)), 'a.wav')
    b = wav(_soundtrack(sec=10.0, hits=(3.0, 7.0, 8.0)), 'b.wav')
    out = str(tmp_path / 'p.json')
    p = P.profile([a, b], styles=['one', 'two'], use_ast=False, out=out)
    q = json.load(open(out))
    assert q['n'] == 2 and q['films'][1]['style'] == 'two' and q['metrics']['hit_rate']['n'] == 2
    assert 'a.wav' not in json.dumps(q) and p['summary'].startswith('profile of 2')
    r = critique(a, profile=out, use_ast=False, use_realness=False)
    rows = {m['metric']: m for m in r['metrics']}
    assert r['profile_n'] == 2 and rows['music_frac']['status'] == 'n.a.' and rows['onset_rate']['status'] != 'n.a.'
    no_ref = {m['metric']: m for m in scorecard({'cuts_with_sound': 0.1}, q)['metrics']}['cuts_with_sound']
    assert no_ref['status'] == 'n.a.' and no_ref['basis'] == 'no reference data'
    assert any('only 2 reference' in w for w in q['warnings'])


def test_profile_skips_unusable_files(wav, tmp_path):
    good = wav(_soundtrack(sec=10.0, hits=(2.0, 4.0)), 'good.wav')
    silent = wav(np.zeros(10 * SR, np.float32), 'silent.wav')
    tiny = wav(_bed(0.5), 'tiny.wav')
    p = P.profile([good, silent, tiny, str(tmp_path / 'missing.wav')], use_ast=False)
    assert p['n'] == 1 and p['metrics']['lufs_i']['min'] > -40 and '3 file(s) skipped' in p['warnings'][0]
    with pytest.raises(P.InputError, match='no usable reference films'):
        P.profile([silent, tiny], use_ast=False)


def test_picture_sync_on_generated_film(tmp_path):
    film = str(tmp_path / 'f.mp4')
    src = ['-f', 'lavfi', '-i', 'color=c=red:s=160x90:r=25:d=2', '-f', 'lavfi', '-i', 'color=c=blue:s=160x90:r=25:d=2',
           '-f', 'lavfi', '-i', 'color=c=white:s=160x90:r=25:d=2']
    aud = ['-f', 'lavfi', '-i', "aevalsrc='0.02*sin(2*PI*220*t)+0.8*exp(-60*mod(t,2))*sin(2*PI*900*t)':s=48000:d=6"]
    flt = '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]'
    p = subprocess.run(['ffmpeg', '-v', 'error', '-y', *src, *aud, '-filter_complex', flt, '-map', '[v]', '-map', '3:a',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', film], capture_output=True)
    if p.returncode != 0:
        pytest.skip('ffmpeg cannot write a test mp4 here')
    f = P.film_metrics(film, use_ast=False)
    assert f['lossy'] and f['fps'] == 25.0
    assert [round(t, 1) for t in f['events']['visual']] == [2.0, 4.0]
    assert f['metrics']['cuts_with_sound'] is None  # fewer than 3 events: no sync shares
    assert f['metrics']['cut_rate'] == pytest.approx(2 / 6, abs=0.02)
    assert os.path.exists(film)


def test_cli_critique_json(wav, capsys):
    from earshot.cli import main
    path = wav(_soundtrack(sec=8.0, hits=(2.0, 5.0)))
    main(['critique', path, '--fast', '--json'])
    r = json.loads(capsys.readouterr().out)
    assert r['scored'] > 10 and r['summary'] and r['partial'] == ['realness', 'music presence']


@pytest.mark.slow
def test_music_presence_with_ast(wav):
    t = np.arange(int(10 * SR)) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0)) * 0.1
    env = 0.6 + 0.4 * (np.sin(2 * np.pi * 2 * t) > 0)
    f = P.film_metrics(wav((chord * env).astype(np.float32)), use_ast=True)
    assert f['metrics']['music_frac'] is not None and len(f['events']['music_windows']) == 2
