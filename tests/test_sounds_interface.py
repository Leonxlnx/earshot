"""The interface roles made of real things (earshot.sounds.interface): made only of learned recordings, clean,
short, told apart in pairs, without a steady pitched partial, and what a plain interface request gets."""
import inspect
import re

import numpy as np
import pytest
from scipy import signal

from earshot import brief
from earshot.forge import dsp
from earshot.sounds import interface, learned, registry, sets

SR = dsp.SR
UI = dict(sets.SETS['ui']['sounds'])


def _mono(y):
    return np.mean(np.atleast_2d(y.T).T, axis=1)


def _centroid(y):
    m = _mono(y)
    p = np.abs(np.fft.rfft(m)) ** 2
    return float((p * np.fft.rfftfreq(len(m), 1 / SR)).sum() / p.sum())


def _tonal(y):
    """dB the strongest steady partial stands over the spectrum a third of an octave around it, on the spectrum
    averaged over the sound's active frames (noise peaks average out, a held partial does not)"""
    f, _, z = signal.stft(_mono(y), SR, nperseg=4096, noverlap=4096 - 512)
    p = np.abs(z) ** 2
    e = p.sum(0)
    a = 10 * np.log10(p[:, e > e.max() * 1e-3].mean(1) + 1e-20)
    out = 0.0
    for i in np.where((f > 150) & (f < 8000))[0]:
        lo, hi = np.searchsorted(f, f[i] * 2 ** (-1 / 6)), np.searchsorted(f, f[i] * 2 ** (1 / 6))
        out = max(out, a[i] - np.median(a[lo:hi + 1]))
    return out


def test_every_part_is_a_recording_of_the_store():
    """the module reads its material only from learned designs: every (design, member) it takes is in the store"""
    src = inspect.getsource(interface)
    parts = re.findall(r"_take\(c, r, '([a-z_]+/[a-z_]+)', (\d+)", src)
    assert len(parts) >= 20
    idx = learned.index()
    for sid, k in parts:
        assert sid in idx and int(k) < len(idx[sid]['members']), (sid, k)
    assert {sid for sid, _ in parts} == {p for v in interface.PARTS.values() for p in v}
    assert set(interface.PARTS) == set(interface.ROLES)


@pytest.mark.parametrize('sid', sorted(interface.ROLES))
def test_renders_clean_short_and_varied(sid):
    y, meta = registry.render(sid, seed=0)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all() and np.abs(y).max() < 1.0
    assert 0.15 <= meta['dur_s'] <= 1.2 and meta['checks']['ok'], meta['checks']
    assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-3 and abs(float(y.mean())) < 1e-3
    y2, _ = registry.render(sid, seed=0)
    assert np.array_equal(y, y2)
    y3, _ = registry.render(sid, seed=1)
    assert len(y3) != len(y) or not np.allclose(y3, y)


def test_the_ui_set_plays_the_defaults_and_a_theme_the_figures():
    assert {UI[n] for n in UI if n != 'tap'} == set(interface.DEFAULTS)
    assert (UI['open'], UI['receive'], UI['success']) == ('open/drawer', 'receive/cork', 'success/shutter')
    themed = dict(sets.get('ui', theme='glass')['sounds'])
    assert set(themed) == set(UI) and themed['success'] == 'success/rise'
    assert sets.get('ui') is sets.SETS['ui']


def test_the_ui_set_is_struck_pressed_turned_or_cut():
    """no role of the set is a breath of air, a soft bag dropped, a crumpled sheet or a pop of the lips; the one
    pass of air is the stick swung for a send. Each lasts 0.15 to 0.9 s"""
    parts = {p for sid in UI.values() for p in interface.PARTS.get(sid, (sid,))}
    assert not parts & {'whoosh/airy', 'bag/drop', 'paper/ball', 'swish/fast', 'swipe/hand', 'pop/mouth', 'pen/cap'}
    for name, sid in UI.items():
        assert 0.15 <= registry.render(sid, seed=0)[1]['dur_s'] <= 0.9, name


def test_pairs_are_told_apart():
    """a pair is two actions: the key coming up is brighter than the key going down; the page forward brighter
    than the page turned back; error's second strike lower than its first (it falls), the pen click's second
    brighter (it rises)"""
    take = {n: registry.render(UI[n], seed=0)[0] for n in ('toggle_on', 'toggle_off', 'next', 'back')}
    assert _centroid(take['toggle_off']) > 1.2 * _centroid(take['toggle_on'])
    assert _centroid(take['next']) > 1.2 * _centroid(take['back'])
    opens, shuts = (registry.render(f'toggle/shutter_{k}', seed=0)[0] for k in ('open', 'shut'))
    assert _centroid(opens) > 1.2 * _centroid(shuts) and len(shuts) > len(opens)
    for sid, up in (('error/knock', False), ('success/click', True)):
        y, _ = registry.render(sid, seed=0)
        on = int(0.06 * SR)
        e = np.convolve(_mono(y) ** 2, np.ones(on) / on, 'same')
        a = int(np.argmax(e[:int(0.08 * SR)]))
        b = int(0.08 * SR) + int(np.argmax(e[int(0.08 * SR):int(0.25 * SR)]))
        first, second = y[max(0, a - on // 2):a + on // 2], y[b - on // 2:b + on // 2]
        assert (_centroid(second) > _centroid(first)) == up, sid


@pytest.mark.parametrize('name', ['notify', 'success', 'error', 'toggle_on'])
def test_no_steady_pitched_partial(name):
    """the real thing has no partial standing out like the marimba figure of the same job does"""
    old = dict(sets.SETS['ui']['themed'])[name]
    real = _tonal(registry.render(UI[name], seed=0)[0])
    assert real < 15.0 and real < _tonal(registry.render(old, seed=0)[0]) - 8.0


@pytest.mark.parametrize('want, sid', [
    ('a success sound', 'success/shutter'), ('notification sound for my app', 'notify/knock'),
    ('error sound for a form', 'error/knock'), ('a sound for sending a message', 'send/swoosh'),
    ('toggle on', 'toggle/key_down'), ('toggle off', 'toggle/key_up'), ('hover', 'hover/fingertip'),
    ('message received', 'receive/cork'), ('back button', 'back/turn'), ('close the modal', 'close/drawer'),
    ('sound when a modal slides in from the right', 'open/drawer'), ('open a menu', 'open/drawer'),
    ('delete', 'delete/snip'), ('a quick bright upward swoosh that flies away', 'send/swoosh')])
def test_a_plain_interface_request_gets_the_real_thing(want, sid):
    assert brief.rank(brief.read(want), top=1)[0]['id'] == sid


@pytest.mark.parametrize('want, sid', [
    ('a success chime', 'success/'), ('a notification on a harp', 'notify/'), ('an error sound on a piano', 'error/'),
    ('a musical notification', 'notify/'), ('two-note success melody', 'success/')])
def test_musical_words_get_the_figure(want, sid):
    top = brief.rank(brief.read(want), top=1)[0]['id']
    assert top.startswith(sid) and 'interface' not in registry.get(top).tags, top
