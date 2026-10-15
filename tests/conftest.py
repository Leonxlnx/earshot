"""Shared pytest fixtures: synthetic test signals only (never committed media)."""
import numpy as np
import pytest
import soundfile as sf

SR = 48000


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: loads the models or runs longer than ~10 s (run with --slow)')


def pytest_addoption(parser):
    parser.addoption('--slow', action='store_true', help='also run tests that load the models')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--slow'):
        return
    skip = pytest.mark.skip(reason='needs --slow (loads models or runs long)')
    for it in items:
        if 'slow' in it.keywords:
            it.add_marker(skip)


@pytest.fixture
def wav(tmp_path):
    """write a float array (n,) or (n, 2) at 48 kHz to a temp WAV and return its path"""
    def make(x, name='x.wav'):
        p = tmp_path / name
        sf.write(str(p), np.asarray(x, np.float32), SR, subtype='FLOAT')
        return str(p)
    return make


@pytest.fixture
def click():
    """a short decaying noise burst (a click-like transient) with a faint noise floor"""
    rng = np.random.RandomState(0)
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    x = rng.randn(n) * np.exp(-t / 0.01) * 0.5 + rng.randn(n) * 1e-4
    return x.astype(np.float32)


@pytest.fixture
def sine():
    t = np.arange(SR) / SR
    return (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
