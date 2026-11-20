"""What the base install is: importing earshot and making sounds load none of the packages the extras bring, every
package earshot imports is declared (in the base install or an extra), and a call that needs a missing extra says
so in one plain sentence with the install command, on the command line, in Python and as an MCP tool."""
import ast
import os
import pathlib
import subprocess
import sys

import pytest

from earshot import cli, extras, mcp_server

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = {'numpy', 'scipy', 'soundfile', 'soxr', 'numba'}          # import names of the base install
HEAVY = ('torch', 'transformers', 'huggingface_hub', 'librosa', 'matplotlib', 'sklearn', 'PIL', 'mcp', 'pydantic',
         'requests', 'pyloudnorm')
OPTIONAL = {'threadpoolctl'}      # used when present, with a fallback (earshot.forge.fit.modal)


def test_import_and_making_sounds_load_no_package_of_an_extra(tmp_path):
    code = (
        'import sys, earshot\n'
        'from earshot.make import make, make_set\n'
        'from earshot import cli\n'
        f'make("a success sound", {str(tmp_path / "a.wav")!r})\n'
        f'make("footsteps on a wooden floor, 2 seconds", {str(tmp_path / "b.wav")!r})\n'
        f'make_set({{"sounds": [{{"name": "tap", "want": "a soft tap"}}]}}, out={str(tmp_path / "s")!r})\n'
        f'cli.main(["make", "a coin", "-o", {str(tmp_path / "c.wav")!r}])\n'
        f'print(sorted(m for m in {HEAVY!r} if m in sys.modules))\n')
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=tmp_path,
                       timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip().splitlines()[-1] == '[]'


def _imports():
    """the top-level names of every module imported anywhere in the package (also inside functions)"""
    out = {}
    for f in (ROOT / 'earshot').rglob('*.py'):
        for node in ast.walk(ast.parse(f.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module]
            else:
                continue
            for n in names:
                out.setdefault(n.split('.')[0], f.relative_to(ROOT).as_posix())
    return out


def test_every_imported_package_is_declared():
    std = set(sys.stdlib_module_names)
    third = {n: f for n, f in _imports().items() if n not in std and n != 'earshot'}
    undeclared = {n: f for n, f in third.items() if n not in BASE | set(extras.EXTRA) | OPTIONAL}
    assert not undeclared


def test_pyproject_declares_the_base_install_and_the_extras_the_messages_name():
    tomllib = pytest.importorskip('tomllib')
    proj = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    names = {d.split('>')[0].split('=')[0].split('[')[0].strip() for d in proj['dependencies']}
    assert names == BASE
    opt = proj['optional-dependencies']
    for extra, pkgs in extras.PACKAGES.items():
        assert tuple(opt[extra]) == pkgs
    assert {'listen', 'measure', 'images', 'mcp', 'llm', 'all', 'dev'} <= set(opt)
    for mod, (extra, pip) in extras.EXTRA.items():
        assert extra in opt
    assert not {'pyloudnorm', 'scikit-learn'} & {d.split('>')[0] for v in opt.values() for d in v}


def _without(monkeypatch, *gone):
    real = extras.missing
    monkeypatch.setattr(extras, 'missing', lambda *m: [x for x in m if x in gone] + real(*[x for x in m
                                                                                          if x not in gone]))


def test_a_missing_extra_is_one_plain_sentence(monkeypatch, tmp_path, capsys):
    _without(monkeypatch, 'torch', 'transformers')
    monkeypatch.delenv('EARSHOT_DEBUG', raising=False)
    from earshot.models import ast as ast_mod, clap
    monkeypatch.setattr(clap, '_inst', None)
    with pytest.raises(extras.MissingExtra, match=r'needs torch, which the base install of earshot leaves out: '
                                                  r'pip install "torch>=2\.6" "transformers>=4\.40" "librosa>=0\.10" '
                                                  r'\(the listen extra; or: pip install '):
        clap.Clap()
    with pytest.raises(extras.MissingExtra):
        ast_mod.AST()
    out = tmp_path / 'x.wav'
    assert cli.main(['make', 'a click', '-o', str(out), '--listen']) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and 'make with listen needs torch' in err[0] and 'pip install' in err[0]
    assert 'Traceback' not in err[0] and 'MissingExtra' not in err[0] and not out.exists()
    with pytest.raises(mcp_server.ToolFailure, match='make: make with listen needs torch'):
        mcp_server._result('make', dict(want='a click', out=str(out), listen=True), mcp_server._summary_only(
            __import__('earshot.make', fromlist=['make']).make))
    wav = tmp_path / 'a.wav'
    __import__('earshot.make', fromlist=['make']).make('a click', str(wav))
    assert cli.main(['describe', str(wav)]) == 1          # the whole extra, before any work
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ['earshot describe: error: earshot describe needs torch, transformers, which the base install of '
                   'earshot leaves out: pip install "torch>=2.6" "transformers>=4.40" "librosa>=0.10" (the listen '
                   'extra; or: pip install "earshot[listen] @ git+https://github.com/Leonxlnx/earshot")']
    with pytest.raises(mcp_server.ToolFailure, match='^earshot describe needs torch'):
        mcp_server._result('describe', dict(path=str(wav)))
    import earshot
    with pytest.raises(extras.MissingExtra, match='earshot describe needs torch'):
        earshot.__getattr__('describe')
    assert make_ok(tmp_path)


def make_ok(tmp_path):
    """making sounds is untouched by a missing extra"""
    from earshot.make import make
    return os.path.exists(make('a coin', str(tmp_path / 'coin.wav'))['files'][0])


def test_plain_names_the_extra_of_a_module_found_missing_deep_inside():
    try:
        raise RuntimeError('wrapped') from ModuleNotFoundError("No module named 'librosa'", name='librosa')
    except RuntimeError as e:
        said = extras.plain(e)
    assert said == ('this needs librosa, which the base install of earshot leaves out: pip install "librosa>=0.10" '
                    '(the measure extra)')
    assert extras.plain(ValueError('x')) is None
    assert extras.plain(ModuleNotFoundError("No module named 'foo'", name='foo')) is None


def test_doctor_says_making_works_without_the_listening_extra(monkeypatch):
    _without(monkeypatch, 'torch', 'transformers', 'librosa')
    from earshot import doctor
    r = doctor.check()
    assert r['extras']['listen'] is False
    assert not [p for p in r['problems'] if 'torch' in p or 'librosa' in p]
    assert any('Making sounds works' in n for n in r['notes'])


def test_every_hint_of_the_listen_extra_is_one_line(monkeypatch):
    _without(monkeypatch, 'torch', 'transformers')
    line = extras.command('listen')
    assert line.startswith('pip install "torch>=2.6" "transformers>=4.40" "librosa>=0.10" (the listen extra; or: ')
    assert line.endswith('pip install "earshot[listen] @ git+https://github.com/Leonxlnx/earshot")')
    from earshot import build
    ears, why = build.ears_for('rain on a tin roof')
    assert ears is None and why.startswith('no ears: ') and why.endswith(line)
    with pytest.raises(extras.MissingExtra) as e:
        extras.check('describe')
    assert str(e.value).endswith(line)
