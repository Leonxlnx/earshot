"""Fast tests for earshot.forge.refsets (registry read / scan / write / split checks) and the model-free parts of the
forge benchmark (eval/forge/bench.py: envelope, artifact and diversity checks). Synthetic data only, no models."""
import importlib.util
import json
import os

import numpy as np
import pytest
import soundfile as sf

from earshot.forge import refsets

SR = 48000
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval', 'forge', 'bench.py')


def _wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, SR, subtype='PCM_24')
    return path


def _ping(f=1000.0, tau=0.05, n=0.4, pre=0.01, seed=0):
    t = np.arange(int(n * SR)) / SR
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / tau) + 1e-4 * np.random.default_rng(seed).standard_normal(len(t))
    x = np.concatenate([np.zeros(int(pre * SR)), x])
    k = int(0.002 * SR)
    x[int(pre * SR):int(pre * SR) + k] *= np.linspace(0, 1, k)
    x[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return 0.8 * x / np.abs(x).max()


@pytest.fixture
def layout(tmp_path):
    root = tmp_path / 'refs'
    for i in range(3):
        _wav(str(root / 'pop' / 'forge' / f'f{i}.wav'), _ping(800 + 100 * i))
        _wav(str(root / 'pop' / 'bench' / f'b{i}.wav'), _ping(900 + 100 * i, seed=i + 5))
    _wav(str(root / 'click' / 'bench' / 'c0.wav'), _ping(3000, 0.01))
    (root / '_work').mkdir()
    return root


def test_missing_registry_is_clear(tmp_path, monkeypatch):
    monkeypatch.setenv('EARSHOT_REFSETS', str(tmp_path / 'none.json'))
    with pytest.raises(refsets.RefsetsMissing, match='bring'):
        refsets.refset('pop')
    with pytest.raises(FileNotFoundError):            # RefsetsMissing is a FileNotFoundError
        refsets.categories()


def test_scan_write_and_read(layout, tmp_path, monkeypatch):
    reg = refsets.scan(str(layout))
    assert sorted(reg['categories']) == ['click', 'pop']          # _work is skipped
    assert reg['categories']['click']['forge'] == []
    path = refsets.write(reg, str(tmp_path / 'reg.json'))
    monkeypatch.setenv('EARSHOT_REFSETS', path)
    assert refsets.registry_path() == os.path.abspath(path)
    assert refsets.categories() == ['click', 'pop']
    assert [os.path.basename(p) for p in refsets.refset('pop')] == ['f0.wav', 'f1.wav', 'f2.wav']
    assert [os.path.basename(p) for p in refsets.refset('pop', split='bench')] == ['b0.wav', 'b1.wav', 'b2.wav']
    assert refsets.refset('click') == [] and refsets.note('click') == ''
    with pytest.raises(KeyError, match='known: click, pop'):
        refsets.refset('whoosh_fast')
    with pytest.raises(ValueError):
        refsets.refset('pop', split='test')


def test_relative_paths_and_missing_files(tmp_path, monkeypatch):
    _wav(str(tmp_path / 'a' / 'x.wav'), _ping())
    reg = {'version': 1, 'categories': {'pop': {'forge': ['a/x.wav'], 'bench': ['a/gone.wav'], 'note': 'thin'}}}
    path = tmp_path / 'reg.json'
    path.write_text(json.dumps(reg))
    monkeypatch.setenv('EARSHOT_REFSETS', str(path))
    assert refsets.refset('pop')[0].endswith('a/x.wav') and os.path.isabs(refsets.refset('pop')[0])
    assert refsets.note('pop') == 'thin'
    with pytest.raises(refsets.RefsetsMissing, match='missing'):
        refsets.refset('pop', split='bench')


def test_split_checks(tmp_path):
    reg = {'categories': {'pop': {'forge': ['a.wav'], 'bench': ['a.wav', 'b.wav']},
                          'tap': {'forge': ['c.wav'], 'bench': ['d.wav']}}}
    assert refsets.check_split(reg) == ['pop: a.wav in both splits']
    with pytest.raises(ValueError, match='leaks'):
        refsets.write(reg, str(tmp_path / 'r.json'))
    reg['categories']['pop']['bench'] = ['b.wav']
    assert refsets.check_split(reg) == []
    fam = {'a.wav': 'author1', 'b.wav': 'author2', 'c.wav': 'author3', 'd.wav': 'author1'}   # a sister take of a.wav
    bad = refsets.check_split(reg, fam)
    assert len(bad) == 1 and 'author1' in bad[0] and "['pop']" in bad[0] and "['tap']" in bad[0]


def test_categories_cover_the_launch_set():
    for c in ('whoosh_fast', 'whoosh_big', 'swish', 'swipe', 'riser', 'downlifter', 'impact', 'sub_drop', 'click', 'tap',
              'toggle', 'hover_tick', 'pop', 'notify', 'shimmer', 'typing_key', 'glitch', 'logo_sting'):
        assert c in refsets.CATEGORIES


# ------------------------------------------------------------------------------- bench.py model-free checks
@pytest.fixture(scope='module')
def bench():
    spec = importlib.util.spec_from_file_location('forge_bench', BENCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_envelope_features(bench):
    x = _ping(1000, 0.05, pre=0.1)
    f = bench.env_feats(x)
    assert abs(f['onset_s'] - 0.1) < 0.005
    assert f['attack_ms'] <= 4
    assert 200 < f['t40_ms'] < 260                   # 40 dB of an exp(-t / 50 ms) decay = 50 ms * ln(100) = 230 ms
    slow = bench.env_feats(_ping(1000, 0.2, n=1.2))
    assert slow['t40_ms'] > 3 * f['t40_ms'] and slow['active_s'] > f['active_s']


def test_env_distance_sees_a_short_tail(bench):
    refs = [_ping(1000, tau, seed=i) for i, tau in enumerate((0.08, 0.1, 0.12))]
    er = bench.EnvRef(refs)
    same = er.check(_ping(1000, 0.1, seed=9))
    short = er.check(_ping(1000, 0.02, seed=9))
    assert same['env_oct'] < 0.3 and short['env_oct'] > 1.0
    assert short['curve_db'] > same['curve_db'] + 5
    assert short['t40_ratio'] < 0.35
    loo = er.check(refs[1], exclude=1)
    assert loo['env_oct'] < 0.5


def test_artifacts(bench):
    clean = _ping()
    assert bench.artifacts(clean)['flags'] == []
    hard = np.sign(np.cos(2 * np.pi * 500 * np.arange(len(clean)) / SR))   # clipped square from a full-scale sample
    fl = bench.artifacts(hard)['flags']
    assert 'clipped' in fl and 'start click' in fl
    cut = _ping(1000, 0.5)[:int(0.15 * SR)]                                # ends while still ringing
    assert 'tail cut' in bench.artifacts(cut)['flags'] and 'end click' in bench.artifacts(cut)['flags']
    dc = clean + 0.2
    assert 'dc offset' in bench.artifacts(dc)['flags']
    late = np.concatenate([np.zeros(int(0.3 * SR)), clean])
    assert 'late start' in bench.artifacts(late)['flags']
    t = np.arange(int(1.0 * SR)) / SR                                      # a swell from the first sample: not late
    swell = 0.8 * np.sin(2 * np.pi * 300 * t) * np.minimum(1, t / 0.8) ** 1.5
    swell[-480:] *= np.linspace(1, 0, 480)
    assert 'late start' not in bench.artifacts(swell)['flags']
    assert 'tail cut' in bench.artifacts(cut)['flags'] and 'tail cut' not in bench.artifacts(cut, 'riser')['flags']


def test_diversity(bench):
    xs = [_ping(1000), _ping(1000, seed=1), _ping(3000, 0.01)]
    E = np.eye(3)
    E[1] = E[0]                                          # two identical embeddings
    d = bench.diversity(xs, E)
    assert d['n'] == 3 and d['near_duplicates'] == [[0, 1]]
    assert d['cos_max'] == pytest.approx(1.0)
    assert bench.diversity(xs[:1], E[:1])['n'] == 1


def test_summary_inside_real(bench):
    real = [{'tone_db': 4.0, 'typ_z': -0.5}, {'tone_db': 6.0, 'typ_z': -1.5}]
    made = [{'tone_db': 5.0, 'typ_z': -1.0}, {'tone_db': 12.0, 'typ_z': -4.0}]
    s = bench.inside_real(made, real, metrics=(('tone_db', 'low'), ('typ_z', 'high')))
    assert s['per_sound'] == [2, 0] and s['n_checks'] == 2


def test_inside_real_quantile(bench):
    real = [{'tone_db': v} for v in (2.0, 3.0, 4.0, 5.0, 20.0)]           # one far outlier among the real rows
    made = [{'tone_db': 10.0}]
    assert bench.inside_real(made, real, metrics=(('tone_db', 'low'),), q=1.0)['per_sound'] == [1]
    assert bench.inside_real(made, real, metrics=(('tone_db', 'low'),), q=0.8)['per_sound'] == [0]


def test_texture(bench):
    rng = np.random.default_rng(3)
    dry = _ping(2000, 0.1, n=0.8)                                          # digital silence before the attack
    tx = bench.texture(dry)
    assert tx['stereo_corr'] == 1.0 and tx['floor_db'] < -60
    floor = dry + 10 ** (-50 / 20) * rng.standard_normal(len(dry))
    assert -60 < bench.texture(floor)['floor_db'] < -40
    st = np.stack([floor, floor + 0.05 * rng.standard_normal(len(floor))], 1)
    assert bench.texture(st)['stereo_corr'] < 0.99
    t = np.arange(int(0.8 * SR)) / SR                                      # bright partial dies first: darkens
    dark = np.sin(2 * np.pi * 500 * t) * np.exp(-t / 0.15) + np.sin(2 * np.pi * 6000 * t) * np.exp(-t / 0.02)
    assert bench.texture(dark)['darken'] < -1
