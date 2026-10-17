"""model files (issue #2): pinned revisions, local-only loads, offline flags respected, no download behind the user's
back. A fake cache tree and fake transformers classes: no network, no model load (the real load is a slow test)."""
import glob
import json
import os
import re
import subprocess
import sys

import pytest

from earshot import config, doctor
from earshot.models import clap, hub

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAP_ID, AST_ID = 'laion/larger_clap_general', 'MIT/ast-finetuned-audioset-10-10-0.4593'
CLAP_PIN, AST_PIN = config.PINS[CLAP_ID], config.PINS[AST_ID]
LEGACY = CLAP_PIN['legacy']


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """an empty earshot model cache, default model ids, no revision override and no offline flag"""
    monkeypatch.setattr(config, 'HF_CACHE', str(tmp_path / 'hf'))
    monkeypatch.setattr(config, 'HOME', str(tmp_path))
    monkeypatch.setattr(config, 'EMB_CACHE', str(tmp_path / 'emb'))
    monkeypatch.setattr(config, 'DEC_CACHE', str(tmp_path / 'decoded'))
    for k, v in (('CLAP_MODEL', CLAP_ID), ('AST_MODEL', AST_ID), ('CLAP_REVISION', None), ('AST_REVISION', None)):
        monkeypatch.setattr(config, k, v)
    for k in config.OFFLINE_FLAGS:
        monkeypatch.delenv(k, raising=False)
    return tmp_path / 'hf'


@pytest.fixture
def downloads(monkeypatch):
    """replace huggingface_hub.snapshot_download: record each call, and write the requested files (minus `skip`)
    into the cache like the real one does"""
    import huggingface_hub
    calls = []

    def fake(repo_id, revision=None, cache_dir=None, allow_patterns=None, **kw):
        calls.append(dict(repo_id=repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=allow_patterns, **kw))
        if fake.fail:
            raise ConnectionError('no route to the Hub')
        files = [f for f in allow_patterns if '*' not in f and f not in fake.skip]
        return snap(repo_id, revision, files)
    fake.fail, fake.skip = False, ()
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fake)
    fake.calls = calls
    return fake


def snap(model_id, sha, files, ref=None):
    """write a snapshot folder into the (patched) cache, plus refs/<ref> -> sha"""
    d = os.path.join(hub.model_dir(model_id), 'snapshots', sha)
    os.makedirs(d, exist_ok=True)
    for f in files:
        with open(os.path.join(d, f), 'wb') as fh:
            fh.write(b'x')
    if ref:
        os.makedirs(os.path.join(hub.model_dir(model_id), 'refs'), exist_ok=True)
        with open(os.path.join(hub.model_dir(model_id), 'refs', ref), 'w') as fh:
            fh.write(sha)
    return d


def pinned(model_id):
    p = config.PINS[model_id]
    return snap(model_id, p['revision'], [*p['files'], p['weights']])


def legacy_clap():
    return snap(CLAP_ID, LEGACY['revision'], [*CLAP_PIN['files'], LEGACY['weights']], ref='main')


class Fake:
    """stands in for a transformers class: records from_pretrained calls, or raises `error`"""
    calls, error = [], None

    @classmethod
    def from_pretrained(cls, path, **kw):
        cls.calls.append((path, kw))
        if cls.error:
            raise cls.error
        return cls()


@pytest.fixture
def fake():
    Fake.calls, Fake.error = [], None
    return Fake


# ------------------------------------------------------------------------------------------------------- pins
def test_pins_are_full_shas_with_safetensors_weights():
    assert set(config.PINS) == {CLAP_ID, AST_ID}
    assert CLAP_PIN['revision'] == '16c8cc3159a3c8a31e8ff5ef1f66b0d9ab3667db'
    assert AST_PIN['revision'] == 'f826b80d28226b62986cc218e5cec390b1096902'
    assert LEGACY == {'revision': 'ada0c23a36c4e8582805bb38fec3905903f18b41', 'weights': 'pytorch_model.bin'}
    for p in config.PINS.values():
        assert re.fullmatch(r'[0-9a-f]{40}', p['revision']) and p['weights'] == 'model.safetensors'
        assert 'config.json' in p['files'] and not any(f.endswith('.bin') for f in p['files'])


@pytest.mark.parametrize('flag', config.OFFLINE_FLAGS)
@pytest.mark.parametrize('value, on', [('1', True), ('true', True), ('YES', True), ('on', True), ('0', False),
                                       ('', False), ('false', False)])
def test_offline_reads_every_flag_at_call_time(monkeypatch, flag, value, on):
    for k in config.OFFLINE_FLAGS:
        monkeypatch.delenv(k, raising=False)
    assert config.offline() is None
    monkeypatch.setenv(flag, value)
    assert config.offline() == (flag if on else None)


def test_no_code_sets_or_clears_the_hf_offline_flags():
    """earshot once set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE when a cache folder existed and removed them to
    download after any failed load; both are the user's settings"""
    writes = re.compile(r"environ(?:\.pop|\.setdefault|\.update|\[)\W*(?:HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)")
    hits = []
    for path in glob.glob(os.path.join(REPO, 'earshot', '**', '*.py'), recursive=True):
        with open(path, encoding='utf-8') as f:
            hits += [f'{os.path.relpath(path, REPO)}: {m.group(0)}' for m in writes.finditer(f.read())]
    assert not hits, hits
    assert not hasattr(clap, 'offline_if_cached')


# ----------------------------------------------------------------------------------------------------- resolve
def test_resolve_pinned_snapshot(cache):
    d = pinned(CLAP_ID)
    s = hub.resolve(CLAP_ID)
    assert (s.state, s.snapshot, s.weights, s.revision, s.sha) == ('cached', d, 'model.safetensors',
                                                                   CLAP_PIN['revision'], CLAP_PIN['revision'])
    assert s.local and s.pinned and s.missing == () and s.cache_tag == ''
    assert s.label == f'{CLAP_ID}@16c8cc3'


def test_resolve_empty_cache(cache):
    s = hub.resolve(AST_ID)
    assert (s.state, s.snapshot, s.local, s.sha) == ('missing', None, False, None)
    assert set(s.missing) == {*AST_PIN['files'], 'model.safetensors'}


def test_resolve_incomplete_download(cache):
    """an interrupted download leaves the folder without its weights: not loadable, reported as incomplete"""
    snap(AST_ID, AST_PIN['revision'], AST_PIN['files'])
    s = hub.resolve(AST_ID)
    assert (s.state, s.snapshot, s.missing) == ('incomplete', None, ('model.safetensors',))


def test_legacy_snapshot_stands_in_for_the_pin(cache):
    """a cache made before 0.2.1 (main ada0c23, pickle weights only) still loads, without a download"""
    d = legacy_clap()
    s = hub.resolve(CLAP_ID)
    assert (s.state, s.snapshot, s.weights, s.revision) == ('legacy', d, 'pytorch_model.bin', CLAP_PIN['revision'])
    assert s.sha == LEGACY['revision'] and s.pinned and s.cache_tag == ''   # same tensors: same embedding cache
    assert 'model.safetensors' in s.missing
    pinned(CLAP_ID)                                     # once the pinned files are there, they win
    assert hub.resolve(CLAP_ID).state == 'cached'


def test_legacy_needs_a_complete_snapshot_and_no_explicit_revision(cache):
    d = legacy_clap()
    assert hub.resolve(CLAP_ID, CLAP_PIN['revision']).state == 'missing'   # asked for the pin itself
    assert hub.resolve(CLAP_ID, LEGACY['revision']).state == 'incomplete'  # an override must be safetensors
    os.remove(os.path.join(d, 'tokenizer.json'))
    assert hub.resolve(CLAP_ID).state == 'missing'


def test_custom_model_resolves_through_refs(cache):
    sha = 'a' * 40
    d = snap('someone/clap-ft', sha, ['config.json', 'model.safetensors.index.json'], ref='main')
    s = hub.resolve('someone/clap-ft')
    assert (s.state, s.snapshot, s.revision, s.weights) == ('cached', d, 'main', 'model.safetensors.index.json')
    assert not s.pinned and s.cache_tag == '--aaaaaaaa'   # another checkpoint: its own embedding cache
    snap('someone/clap-ft', 'b' * 40, ['config.json', 'pytorch_model.bin'], ref='v2')
    assert hub.resolve('someone/clap-ft', 'v2').state == 'incomplete'   # pickle weights only: not loadable
    assert hub.resolve('someone/clap-ft', 'v3').state == 'missing'


# -------------------------------------------------------------------------------------------------------- load
def test_cached_load_reads_the_snapshot_folder_only(cache, fake):
    d = pinned(CLAP_ID)
    spec = hub.ensure(CLAP_ID)
    clap._load_hf(fake, spec, weights=True)
    clap._load_hf(fake, spec)
    assert fake.calls == [(d, {'local_files_only': True, 'use_safetensors': True}),
                          (d, {'local_files_only': True})]
    assert not any(k in os.environ for k in config.OFFLINE_FLAGS)   # nothing set behind the user's back


def test_legacy_load_uses_the_pickle_from_the_cache(cache, fake, downloads):
    d = legacy_clap()
    clap._load_hf(fake, hub.ensure(CLAP_ID), weights=True)
    assert fake.calls == [(d, {'local_files_only': True, 'use_safetensors': False})] and downloads.calls == []


def test_failed_cached_load_is_an_error_not_a_download(cache, fake, downloads):
    d = pinned(AST_ID)
    fake.error = OSError('unexpected end of file')
    with pytest.raises(RuntimeError, match='could not load') as e:
        clap._load_hf(fake, hub.ensure(AST_ID), weights=True)
    assert d in str(e.value) and 'unexpected end of file' in str(e.value)
    assert len(fake.calls) == 1 and downloads.calls == []


@pytest.mark.parametrize('flag', config.OFFLINE_FLAGS)
def test_offline_flag_refuses_a_download(cache, downloads, monkeypatch, flag):
    monkeypatch.setenv(flag, '1')
    with pytest.raises(RuntimeError, match=f'{flag} is set, so earshot will not download'):
        hub.ensure(CLAP_ID)
    snap(AST_ID, AST_PIN['revision'], AST_PIN['files'])
    with pytest.raises(RuntimeError, match='incomplete in the earshot cache \\(missing: model.safetensors\\)'):
        hub.ensure(AST_ID)
    assert downloads.calls == [] and os.environ[flag] == '1'   # refused, and the user's flag left alone


def test_offline_flag_still_loads_what_is_cached(cache, downloads, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    d = pinned(AST_ID)
    assert hub.ensure(AST_ID).snapshot == d and downloads.calls == []


def test_missing_model_downloads_the_pinned_files_once(cache, downloads):
    spec = hub.ensure(CLAP_ID)
    assert spec.state == 'cached' and spec.sha == CLAP_PIN['revision']
    [call] = downloads.calls
    assert call['repo_id'] == CLAP_ID and call['revision'] == CLAP_PIN['revision']
    assert call['cache_dir'] == config.HF_CACHE
    assert sorted(call['allow_patterns']) == sorted([*CLAP_PIN['files'], 'model.safetensors'])
    hub.ensure(CLAP_ID)
    assert len(downloads.calls) == 1   # cached now: no second download


def test_custom_model_download_never_asks_for_pickles(cache, downloads):
    with pytest.raises(RuntimeError, match='safetensors weights only'):
        hub.ensure('someone/clap-ft')   # the fake writes no files for wildcard patterns
    [call] = downloads.calls
    assert call['revision'] == 'main' and not any('.bin' in p or p == '*' for p in call['allow_patterns'])


def test_failed_download_is_one_readable_error(cache, downloads):
    downloads.fail = True
    with pytest.raises(RuntimeError, match='could not download .*ConnectionError: no route to the Hub'):
        hub.ensure(AST_ID)
    downloads.fail, downloads.skip = False, ('model.safetensors',)
    with pytest.raises(RuntimeError, match='still lacks model.safetensors'):
        hub.ensure(AST_ID)


# ------------------------------------------------------------------------------------------------------ doctor
def test_doctor_reports_revisions_and_cache_state(cache):
    legacy_clap()
    pinned(AST_ID)
    r = doctor.check()
    c, a = r['models'][CLAP_ID], r['models'][AST_ID]
    assert c == {'revision': CLAP_PIN['revision'], 'cached': 'legacy', 'loads': LEGACY['revision'],
                 'weights': 'pytorch_model.bin', 'snapshot': hub.resolve(CLAP_ID).snapshot,
                 'missing': [*CLAP_PIN['files'], 'model.safetensors']}
    assert (a['cached'], a['loads'], a['weights']) == ('cached', AST_PIN['revision'], 'model.safetensors')
    assert r['models_cached'] == {CLAP_ID: True, AST_ID: True} and r['offline'] is None
    assert 'earshot doctor --fetch' in r['summary'] and 'pickle weights' in r['summary']


def test_doctor_names_the_offline_flag_when_models_are_missing(cache, monkeypatch):
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    r = doctor.check()
    assert r['offline'] == 'TRANSFORMERS_OFFLINE' and r['models_cached'] == {CLAP_ID: False, AST_ID: False}
    assert any('TRANSFORMERS_OFFLINE is set' in p for p in r['problems'])


def test_doctor_load_reports_instead_of_failing(cache, monkeypatch):
    monkeypatch.setenv('EARSHOT_OFFLINE', '1')
    monkeypatch.setattr(doctor, '_load', lambda r: pytest.fail('loaded a model that is not cached while offline'))
    r = doctor.check(load=True)
    assert 'EARSHOT_OFFLINE is set' in r['summary'] and 'load failed' not in r['summary']
    pinned(CLAP_ID)
    pinned(AST_ID)

    def broken(r):
        raise RuntimeError('could not load the model (damaged).')
    monkeypatch.setattr(doctor, '_load', broken)
    r = doctor.check(load=True)
    assert r['problems'][-1] == 'model load failed: could not load the model (damaged).'
    assert r['summary'].endswith('(damaged).')   # one full stop, not two


def test_doctor_fetch_downloads_only_what_is_missing(cache, downloads):
    legacy_clap()
    pinned(AST_ID)
    r = doctor.check(fetch_missing=True)
    assert r['fetched'] == {CLAP_ID: 'downloaded 16c8cc3', AST_ID: 'already cached'}
    assert [c['repo_id'] for c in downloads.calls] == [CLAP_ID]
    assert r['models'][CLAP_ID]['cached'] == 'cached' and 'pickle' not in r['summary']


def test_doctor_cli_takes_fetch(cache, downloads, capsys, monkeypatch):
    monkeypatch.setenv('EARSHOT_OFFLINE', '1')
    from earshot import cli
    assert cli.main(['doctor', '--fetch', '--json']) == 0
    r = json.loads(capsys.readouterr().out)
    assert r['fetched'][CLAP_ID].startswith('failed: ') and 'EARSHOT_OFFLINE is set' in r['fetched'][CLAP_ID]
    assert downloads.calls == []


# --------------------------------------------------------------------------------------------------- real load
@pytest.mark.slow
def test_cached_models_load_without_network(tmp_path):
    """the real models from the real cache, with the Hub unreachable and no offline flag: a cached load must not try
    the network (earshot used to call the Hub for update checks and safetensors conversion)"""
    s = [hub.resolve(m) for m in (CLAP_ID, AST_ID)]
    if not all(x.local for x in s):
        pytest.skip('the models are not in the earshot cache')
    env = {k: v for k, v in os.environ.items() if k not in config.OFFLINE_FLAGS}
    env.update(HF_ENDPOINT='http://127.0.0.1:9', EARSHOT_THREADS='2', EARSHOT_HOME=config.HOME,
               PYTHONIOENCODING='utf-8')
    p = subprocess.run([sys.executable, '-m', 'earshot', 'doctor', '--load', '--json'], cwd=REPO, env=env,
                       capture_output=True, text=True, encoding='utf-8', timeout=900)
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout)
    assert not any('load failed' in x for x in r['problems']), r['problems']
    assert (r['clap_revision'], r['ast_revision']) == (s[0].sha, s[1].sha)
    assert (r['clap_weights'], r['ast_weights']) == (s[0].weights, s[1].weights)
