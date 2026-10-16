"""forge v2 integration: refsets -> forge refs (held-out guard), critic v2 components in the reports, the CLI wiring,
and the new dsp / fx tools inside a recipe. Synthetic data, no models (the one model test is marked slow)."""
import json
import math
import os
import types

import numpy as np
import pytest
import soundfile as sf

from earshot.forge import dsp, forge, fx, pick_refs, refsets, report_md, resolve_refs
from earshot.forge.search import envelope_stats, prescreen_for, render

SR = 48000


def _ping(f=1000.0, tau=0.03, dur=0.3, seed=0):
    t = np.arange(int(dur * SR)) / SR
    r = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * f * t) * np.exp(-t / tau) + 1e-3 * r.standard_normal(len(t))) * 0.8


def _registry(tmp_path, monkeypatch, cats=('whoosh_fast', 'impact')):
    """<tmp>/refs/<cat>/{forge,bench}/*.wav -> a registry file EARSHOT_REFSETS points at; returns the registry dict"""
    root = tmp_path / 'refs'
    for c in cats:
        for s, n in (('forge', 3), ('bench', 2)):
            d = root / c / s
            d.mkdir(parents=True)
            for i in range(n):
                sf.write(str(d / f'{c}_{s}_{i}.wav'), _ping(500 + 100 * i, seed=i), SR)
    path = str(tmp_path / 'refsets.json')
    monkeypatch.setenv('EARSHOT_REFSETS', path)
    refsets.write(refsets.scan(str(root)), path)
    return refsets.load(path)


def test_held_out_flags_registry_bench_files_and_bench_source_kit(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    fo, be = reg['categories']['impact']['forge'][0], reg['categories']['impact']['bench'][0]
    kit = str(tmp_path / 'impact_03.wav')
    assert refsets.held_out([fo, be, kit]) == [be, kit]
    assert refsets.held_out([fo, be, kit], kit=False) == [be]
    monkeypatch.setenv('EARSHOT_REFSETS', str(tmp_path / 'none.json'))           # no registry: only the kit rule
    assert refsets.held_out([fo, be, kit]) == [kit]


def test_pick_refs_prefers_refset_then_refs_then_recipe_refset_then_recipe_refs(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    forge_w = reg['categories']['whoosh_fast']['forge']
    bench_w = reg['categories']['whoosh_fast']['bench']
    assert pick_refs(refset='whoosh_fast') == (forge_w, 'whoosh_fast')
    said = []
    got, cat = pick_refs(refs=forge_w[:1] + bench_w[:1], log=said.append)           # explicit refs: bench dropped
    assert got == forge_w[:1] and cat is None and 'held-out' in said[0]
    kit_ok, kit_bad = str(tmp_path / 'impact_01.wav'), str(tmp_path / 'impact_03.wav')
    for p in (kit_ok, kit_bad):
        sf.write(p, _ping(), SR)
    rec = types.SimpleNamespace(REFSET='impact', REFS=[kit_ok, kit_bad])
    assert pick_refs(recipe=rec) == (reg['categories']['impact']['forge'], 'impact')
    rec.REFSET = 'no_such_category'                                                 # unknown category: the kit REFS,
    assert pick_refs(recipe=rec) == ([kit_ok], None)                                # minus the bench-source kit file
    monkeypatch.setenv('EARSHOT_REFSETS', str(tmp_path / 'none.json'))
    rec.REFSET = 'impact'
    assert resolve_refs(recipe=rec) == [kit_ok]
    assert resolve_refs(recipe=types.SimpleNamespace()) is None


class V2Critic:
    """a model-free stand-in that returns rows shaped like critic v2's (v1 terms + env / art / polish components)"""
    target, alts, tone, tone_target, version, mode = 'a 1 kHz ping', ['a hum'], 0.0, None, 2, 'real'
    W = {'tone': 1.0, 'env': 1.0, 'art': 1.0, 'polish': 0.75}

    def __init__(self, refs=()):
        self.refs = list(refs)
        self.polish_ref = None

    def score(self, xs, exclude=None):
        rows = []
        for i, x in enumerate(xs):
            m = np.asarray(x).mean(1) if np.ndim(x) == 2 else np.asarray(x)
            f = np.fft.rfftfreq(len(m), 1 / SR)[int(np.argmax(np.abs(np.fft.rfft(m))))]
            t20 = envelope_stats(x)['t20_ms']
            env = round(abs(math.log(max(t20, 1.0) / 70.0)), 3)
            total = 5.0 - 4 * abs(math.log2(max(f, 1.0) / 1000)) - env
            rows.append({'i': i, 'total': round(total, 3), 'total_v1': round(total + env, 3), 'passes': total > 3.5,
                         'match_z': 4.0, 'alt_margin': 1.0, 'p_real': 0.9, 'ref_typ_z': None, 'feat_pen': 0.0,
                         'sounds_like': ['a ping'], 'nearest_ref': None, 'mode': 'real', 'tone_pen': 0.1,
                         'env_db': 5.0, 'env_ratio': 1.0, 'env_out': 0.0, 'env_pen': env, 'art_pen': 0.0,
                         'polish_pen': 0.25, 'hard': [], 'polish_words': 'clean', 'summary': f'total {total:.2f}'})
        return rows

    def score_refs(self, loo=True):
        return self.score([sf.read(p)[0] for p in self.refs])

    def holdout(self, xs):
        return [{'audioset_top': [['Ping', 0.5]], 'audioset_synth_max': 0.0} for _ in xs]


def _toy():
    def rend(p, r):
        t = np.arange(int(0.3 * SR)) / SR
        return np.sin(2 * np.pi * p['f'] * t) * np.exp(-t / p['tau']) + 1e-4 * r.standard_normal(len(t))
    return types.SimpleNamespace(__name__='earshot.forge.recipes.toy', SPACE={'f': (300.0, 3000.0, 'log'),
                                 'tau': (0.005, 0.1, 'log')}, DEFAULTS={'f': 600.0, 'tau': 0.05}, render=rend)


def test_forge_report_carries_the_v2_components(tmp_path):
    refs = []
    for i in range(4):
        p = str(tmp_path / f'ref_{i}.wav')
        sf.write(p, _ping(990 + 10 * i, 0.03 + 0.002 * i, seed=i), SR)
        refs.append(p)
    c = V2Critic(refs)
    out = tmp_path / 'made'
    rep = forge(_toy(), budget=16, pop=8, keep=2, seed=2, out=str(out), log=None, critic=c, prescreen=16)
    assert rep['critic'] == {'version': 2, 'mode': 'real', 'weights': V2Critic.W} and rep['tone_weight'] == 1.0
    disk = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    for k in ('total_v1', 'env_pen', 'art_pen', 'polish_pen', 'hard', 'mode'):
        assert k in disk['history'][0]['score'] and k in disk['ref_scores'][0], k
    md = (out / 'report.md').read_text(encoding='utf-8')
    assert 'critic v2, mode real' in md and '| env pen | art pen | polish pen | hard |' in md
    assert md.count('| real ref_') == 4


def test_cli_passes_refset_mode_and_report_only_tone(monkeypatch, tmp_path):
    import earshot.forge as F
    from earshot.cli import main
    _registry(tmp_path, monkeypatch, cats=('pop',))
    seen = {}

    def fake(rec, *a, **kw):
        seen.update(kw, refs=a[2])
        return {'summary': 'ok', 'picks': []}
    monkeypatch.setattr(F, 'forge', fake)
    assert main(['forge', 'pop', '--budget', '8', '--tone', '0', '--mode', 'designed', '--no-bank']) == 0
    assert seen['refset'] == 'pop' and len(seen['refs']) == 3                      # the recipe's REFSET, forge split
    assert seen['weights'] == {'tone': 0.0} and seen['mode'] == 'designed'
    assert main(['forge', 'pop', '--budget', '8', '--no-bank']) == 0
    assert seen['weights'] is None and seen['tone'] == 0.0                          # the critic's own v2 weight


def test_prescreen_reuses_the_critics_polish_ref():
    c = types.SimpleNamespace(refs=[], tone_target=None, polish_ref='sentinel')
    assert prescreen_for(c).pol == 'sentinel'


def test_a_recipe_built_on_dsp_and_fx_renders_through_the_search_path():
    """the round-2 recipes stack dsp sources and fx: a sub drop + air burst through reverb and the limiter"""
    def rend(p, r):
        sub = dsp.sub_drop(1.2, r, f0=p['f0'], f1=35.0)
        hit = dsp.stack([{'x': sub}, {'x': dsp.air(0.4, r), 'gain': -10.0, 't': 0.01}])
        return fx.limit(fx.reverb(hit, r, rt60=p['rt60'], lowcut=150.0), ceiling_db=-1.0)
    rec = types.SimpleNamespace(__name__='toy_hit', SPACE={'f0': (60.0, 140.0), 'rt60': (0.5, 2.5)},
                                DEFAULTS={'f0': 90.0, 'rt60': 1.2}, render=rend)
    x = render(rec, rec.DEFAULTS, 3)
    assert x is not None and np.all(np.isfinite(x)) and np.abs(x).max() <= 1.0
    assert len(x) >= int(1.2 * SR)


@pytest.mark.slow
def test_critic_typicality_needs_three_refs(tmp_path):
    from earshot.forge.critic import Critic
    refs = []
    for i in range(2):
        p = str(tmp_path / f'r{i}.wav')
        sf.write(p, _ping(800 + 50 * i, 0.04, seed=i), SR)
        refs.append(p)
    row = Critic('a short ping', refs=refs).score([_ping(820, 0.04, seed=9)])[0]
    assert row['ref_typ_z'] is None                                                 # 2 refs = one pair, no spread


def test_forge_without_refs_still_writes_every_output(tmp_path):
    """empty refsets (downlifter, glitch) give a critic without refs: designed mode, artifact-only pre-screen"""
    out = tmp_path / 'made'
    rep = forge(_toy(), budget=16, pop=8, keep=2, seed=1, out=str(out), log=None, critic=V2Critic(), prescreen=16)
    for f in ('report.md', 'report.json', 'compare.png', 'audition.wav'):
        assert (out / f).exists(), f
    assert rep['ref_scores'] == [] and 'compare_error' not in rep['files']
