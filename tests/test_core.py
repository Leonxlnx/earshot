"""Fast tests for core behaviour fixed in the final review: envelope times, decode errors, cache fingerprints, CLI
output on any code page, find's trims, critique's reference match and window thresholds. Synthetic signals only."""
import io
import json
import os
import subprocess

import numpy as np
import pytest

from earshot import audio, cli, critique, features, profile, report, search, sheet

SR = 48000


def _decaying(f, tau=0.5, dur=4.0, lead=0.1):
    t = np.arange(int(dur * SR)) / SR
    x = (0.8 * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)).astype(np.float32)
    return np.concatenate([np.zeros(int(lead * SR), np.float32), x])


# ---------------------------------------------------------------------------------------------- envelope
@pytest.mark.parametrize('f', [40, 80, 200, 1000])
def test_decay_is_right_at_any_pitch(f):
    true = 0.5 * np.log(10 ** 1.5) * 1000          # -30 dB of exp(-t / 0.5): 1727 ms
    r = features.measure(_decaying(f), light=True)
    assert abs(r['decay_ms'] - true) <= 0.02 * true
    assert r['attack_ms'] <= 6                       # an instant onset: ~4 ms on the 5 ms envelope


def test_decay_of_a_noise_burst_and_attack_of_a_steady_tone():
    rng = np.random.RandomState(0)
    n = 2 * SR
    burst = (rng.randn(n) * np.exp(-np.arange(n) / SR / 0.1) * 0.5).astype(np.float32)
    true = 0.1 * np.log(10 ** 1.5) * 1000           # 345 ms
    assert abs(features.measure(burst, light=True)['decay_ms'] - true) <= 0.05 * true
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR)).astype(np.float32)
    r = features.measure(tone, light=True)
    assert r['attack_ms'] <= 2 and r['decay_ms'] is None   # starts at full level, never decays


def test_measure_rejects_impossible_ranges(wav, sine):
    p = wav(sine)
    with pytest.raises(ValueError, match='beyond the end'):
        features.measure(p, start=5.0)
    with pytest.raises(ValueError, match='greater than start'):
        features.measure(p, start=0.5, end=0.2)


# ---------------------------------------------------------------------------------------------- audio
def test_file_id_sees_a_same_size_swap_with_the_old_timestamp(wav, sine):
    p = wav(sine, 'swap.wav')
    st = os.stat(p)
    a = audio.file_id(p)
    import soundfile as sf
    sf.write(p, (sine[::-1] * 0.5).astype(np.float32), SR, subtype='FLOAT')   # same length, same size
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(p).st_size == st.st_size and os.stat(p).st_mtime_ns == st.st_mtime_ns
    assert audio.file_id(p) != a


def test_decode_errors_are_one_readable_line(tmp_path):
    bad = tmp_path / 'corrupt.wav'
    bad.write_text('not audio')
    with pytest.raises(audio.AudioError) as e:
        audio.load(str(bad), cache=False)
    assert '\n' not in str(e.value) and 'cannot decode' in str(e.value)
    vid = tmp_path / 'silent render.mp4'
    r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=1', '-an', str(vid)],
                       capture_output=True)
    if r.returncode != 0:
        pytest.skip('ffmpeg cannot make a test video here')
    with pytest.raises(audio.AudioError, match='has no audio track'):
        audio.load(str(vid), cache=False)


def test_missing_paths_are_named(tmp_path, wav, sine):
    ok = wav(sine)
    audio.check_paths([ok])
    with pytest.raises(audio.AudioError, match='nope.wav'):
        audio.check_paths([ok, str(tmp_path / 'nope.wav')])


def test_replace_quietly_never_raises(tmp_path):
    tmp = tmp_path / 'x.tmp'
    tmp.write_text('1')
    assert audio.replace_quietly(str(tmp), str(tmp_path / 'x'))
    tmp.write_text('2')
    assert not audio.replace_quietly(str(tmp), str(tmp_path / 'no' / 'such' / 'dir' / 'x'), tries=1)
    assert not tmp.exists()


# ---------------------------------------------------------------------------------------------- CLI output
def test_json_on_a_cp1252_stdout_is_valid(monkeypatch):
    buf = io.BytesIO()
    out = io.TextIOWrapper(buf, encoding='cp1252')
    monkeypatch.setattr('sys.stdout', out)
    report.emit({'summary': 'ok', 'file': 'キー音.wav'}, type('A', (), {'json': True, 'out': None})())
    out.flush()
    assert json.loads(buf.getvalue().decode('cp1252'))['file'] == 'キー音.wav'


def test_threads_after_the_command_and_errors_on_stderr(wav, sine, capsys, monkeypatch):
    monkeypatch.setenv('EARSHOT_THREADS', '3')   # restored after the test
    p = wav(sine)
    assert cli.main(['measure', p, '--light', '--threads', '2']) == 0
    assert os.environ.get('EARSHOT_THREADS') == '2'
    capsys.readouterr()
    assert cli.main(['measure', p + '.nope']) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith('earshot measure: error:') and '\n' not in err


def test_profile_summary_survives_unmeasured_metrics():
    p = {'n': 2, 'metrics': {k: {'n': 0} for k in profile.METRICS}}
    assert 'LRA n.a.' in profile.profile_summary(p)


def test_sheet_ticks_at_quarter_seconds_are_exact():
    assert sheet._tick_label(0.25, 0.25) == '0:00.25' and sheet._tick_label(0.75, 0.25) == '0:00.75'
    assert sheet._tick_label(0.5, 0.5) == '0:00.5' and sheet._tick_label(61.0, 1) == '1:01'


# ---------------------------------------------------------------------------------------------- find trims
def _swish(n, rng, peak_at=0.1):
    t = np.arange(n) / SR
    env = np.exp(-((t - peak_at) / 0.05) ** 2)
    return (rng.randn(n) * env * 0.4).astype(np.float32)


def test_events_never_merge_across_an_edit_gap():
    rng = np.random.RandomState(3)
    a, b = _swish(int(0.3 * SR), rng), _swish(int(0.3 * SR), rng)
    x = np.concatenate([np.zeros(SR // 10, np.float32), a, np.zeros(int(0.05 * SR), np.float32), b,
                        np.zeros(SR // 2, np.float32)])
    env = search._env_db(x)
    assert search._edit_gaps(env, 0.35, 0.5)
    for s, e, k, _ in search.event_candidates(x, n=None):
        assert not (s < 0.42 and e > 0.47), (s, e, k)   # nothing spans the 50 ms of digital silence


def test_windows_snap_to_the_sound_and_multi_sound_windows_are_detected():
    rng = np.random.RandomState(4)
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    rise = (rng.randn(n) * np.minimum(1.0, t / 0.8) ** 3 * np.exp(-np.maximum(0, t - 0.8) / 0.05) * 0.4)
    x = np.concatenate([np.zeros(SR // 5, np.float32), rise.astype(np.float32), np.zeros(SR // 2, np.float32)])
    env = search._env_db(x)
    floor = float(np.percentile(env, 10))
    a, b = search.snap_window(env, floor, 0.7, 1.2, 0.0, len(x) / SR)   # starts mid-rise, ends mid-decay
    assert a < 0.5 and search._head_db(env, a, b) <= search.CUT_DB and search._tail_db(env, a, b) <= -30
    two = np.concatenate([_swish(int(0.3 * SR), rng), np.zeros(int(0.4 * SR), np.float32), _swish(int(0.3 * SR), rng)])
    e2 = search._env_db(two)
    f2 = float(np.percentile(e2, 10))
    assert search.multi_sound(e2, f2, 0.0, len(two) / SR) and not search.multi_sound(e2, f2, 0.0, 0.3)


# ---------------------------------------------------------------------------------------------- critique
def test_a_reference_film_is_scored_leave_one_out():
    prof = profile.load_profile()
    g = prof['films'][4]
    f = {'duration': g['duration'], 'metrics': dict(g['metrics'])}
    assert critique.reference_match(f, prof) == g['id']
    loo = critique.leave_one_out(prof, g['id'])
    assert loo['n'] == prof['n'] - 1 and g['id'] not in [x['id'] for x in loo['films']]
    f['duration'] += 1.0
    assert critique.reference_match(f, prof) is None


def test_critique_flags_windows_at_the_window_threshold():
    t = critique.real_thresholds(file=False)
    segs = [{'start': 4.0 * i, 'end': 4.0 * i + 4, 'p_real': p} for i, p in enumerate([0.3, 0.18, t['lo'] - 0.01])]
    f = {'realness': {'segments': segs}, 'events': {}, 'duration': 12.0, 'metrics': {}}
    _, flagged, tw = critique._realness_windows(f)
    assert tw == t and [q['p_real'] for q in flagged] == [t['lo'] - 0.01]
    assert critique._p(0.766, {'lo': 0.4, 'hi': 0.77}) == '0.766' and critique._p(0.81, {'lo': 0.4, 'hi': 0.77}) == '0.81'


# ------------------------------------------------------------------------------------------------- version
def test_one_version_everywhere():
    """the package, pyproject and the newest changelog entry name one version"""
    import re

    import earshot
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'pyproject.toml'), encoding='utf-8') as fh:
        pyproject = re.search(r'^version = "([^"]+)"', fh.read(), re.M).group(1)
    with open(os.path.join(root, 'CHANGELOG.md'), encoding='utf-8') as fh:
        top = re.search(r'^## (\S+)', fh.read(), re.M).group(1)
    assert earshot.__version__ == pyproject == top == '0.4.0'
