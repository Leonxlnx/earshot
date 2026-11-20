# Fast tests without models

The fast-test workflow runs the existing default pytest suite on Linux with
Python 3.12, Node 24, and FFmpeg/ffprobe. Tests use synthetic media. It does not
pass `--slow`, download model weights, or run licensed-reference benchmarks.
Node is needed by the real-sound mux test; omitting it silently skips that test.

For the same reduced environment locally, make those executables available on
PATH, then run from a clone:

```sh
python3.12 -m venv .venv-fast
.venv-fast/bin/python -m pip --isolated install --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple -r tests/requirements-fast.txt
.venv-fast/bin/python -m pip --isolated install --no-deps --no-build-isolation --no-cache-dir -e .
EARSHOT_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-fast/bin/python -m pytest -q -p no:cacheprovider -rs tests
```

This intentionally leaves out Torch and Transformers, which remain required by
the normal package metadata. It is **not** a full installation or a packaging
dependency check. Follow the README's installation instructions when running
model-dependent features. Do not run `doctor --load` in this reduced environment.

`tests/requirements-fast.txt` mirrors the non-model runtime requirements and adds
the optional MCP/HTTP test dependencies, pytest, the real Hugging Face hub package
used by fast loader tests, and the existing setuptools build backend. Keep it in
sync when those requirements or tests change. Direct and transitive dependencies
are not locked; package versions, Python patch releases, and the runner's FFmpeg
can change between runs. The action implementations are pinned separately by
commit SHA.

Offline flags prevent normal model fetching; they are not a general network
firewall. Installation needs network access. The workflow does not supply service
credentials, persist checkout credentials, upload artifacts, or run with write
permissions. External PR jobs may still require a maintainer to approve execution.

Keep the repository's normal skip rules: slow/model tests and the optional
delivered-final rerender fixture are outside this job's coverage. A passing fast
job does not establish model quality, cross-platform behavior, full package
installation correctness, or compatibility with every Python version advertised
by the package.
