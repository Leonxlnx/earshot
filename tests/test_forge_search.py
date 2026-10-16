"""Fast tests for forge search v2: CMA-ES with restarts, the model-free pre-screen, threaded rendering, envelope and
artifact measurements, diverse picks by sound, the seed bank, forge() outputs (report.md, compare.png, audition.wav),
render without search and the CLI. Toy recipes and fake critics only: no models are loaded."""
import json
import math
import os
import types

import numpy as np
import soundfile as sf

from earshot.forge import audition, forge, render_params, report_md
from earshot.forge import search
from earshot.forge.search import (Prescreen, artifacts, describe, diverse, envelope_stats, load_seeds, optimize,
                                  render, render_many, save_seeds, sound_dist)

SR = 48000


def _ping(f, tau, n=None, fade=True, r=None):
    n = n or int(0.25 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / tau) * np.minimum(1.0, t / 0.001)
    if r is not None:
        x = x + 0.0005 * r.standard_normal(n)
    if fade:
        x[-480:] *= np.linspace(1, 0, 480)
    return x


def _toy_render(p, r):
    return _ping(p['f'], p['tau'], r=r) * {'soft': 0.2, 'hard': 1.0}[p['shape']]


def toy(**over):
    m = types.SimpleNamespace(__name__='earshot.forge.recipes.toy',
                              SPACE={'f': (200.0, 4000.0, 'log'), 'tau': (0.005, 0.2, 'log'), 'shape': ['soft', 'hard'],
                                     'n': (1, 9, 'int')},
                              DEFAULTS={'f': 300.0, 'tau': 0.15, 'shape': 'soft', 'n': 3}, render=_toy_render)
    for k, v in over.items():
        setattr(m, k, v)
    return m


def _peak_hz(x):
    m = np.asarray(x).mean(axis=1) if np.ndim(x) == 2 else np.asarray(x)
    return np.fft.rfftfreq(len(m), 1 / SR)[int(np.argmax(np.abs(np.fft.rfft(m))))]


class FakeCritic:
    """scores by closeness to 1 kHz and a 30 ms decay (no models); counts how many sounds it heard"""
    target, alts, refs, tone, tone_target = 'a 1 kHz ping', ['a low hum'], [], 0.0, None

    def __init__(self):
        self.heard = 0

    def score(self, xs, exclude=None):
        rows = []
        for i, x in enumerate(xs):
            self.heard += 1
            tau = max(1e-3, envelope_stats(x)['t20_ms'] / 1000 / 2.303)
            total = 5.0 - 4 * abs(math.log2(max(_peak_hz(x), 1.0) / 1000)) - abs(math.log(tau / 0.03))
            rows.append({'i': i, 'total': round(total, 3), 'passes': total > 3.5, 'match_z': total, 'alt_margin': 1.0,
                         'p_real': 0.9, 'ref_typ_z': None, 'feat_pen': 0.0, 'sounds_like': ['a ping'],
                         'nearest_ref': None, 'summary': f'total {total:.2f}'})
        return rows

    def score_refs(self, loo=True):
        return self.score([sf.read(p)[0] for p in self.refs])

    def holdout(self, xs):
        return [{'audioset_top': [['Ping', 0.5]], 'audioset_synth_max': 0.0} for _ in xs]


def _refs(tmp_path, n=4):
    """four real-like 1 kHz pings (slightly different decays and pitches) written as WAVs"""
    out = []
    for i in range(n):
        p = str(tmp_path / f'ref_{i}.wav')
        sf.write(p, _ping(980 + 15 * i, 0.028 + 0.003 * i, r=np.random.default_rng(i)) * 0.8, SR)
        out.append(p)
    return out


# ---------------------------------------------------------------------------------------------- CMA-ES
def test_cma_improves_on_the_default_and_is_deterministic():
    rec = toy()
    base = FakeCritic().score([render(rec, rec.DEFAULTS, 0)])[0]['total']
    c = FakeCritic()
    a = optimize(rec, c, budget=96, pop=16, seed=3, strategy='cma')
    b = optimize(rec, FakeCritic(), budget=96, pop=16, seed=3, strategy='cma')
    assert len(a['history']) == 96 and a['generations'] == 6 and c.heard == 96
    assert a['best'][0]['score']['total'] > base + 2
    assert [h['params'] for h in a['history']] == [h['params'] for h in b['history']]
    assert a['history'][0]['params'] == {k: rec.DEFAULTS[k] for k in rec.SPACE}      # the default is always scored
    assert all('sigma' in g for g in a['gens']) and a['strategy'] == 'cma'


def test_cma_restarts_when_stuck():
    res = optimize(toy(), FakeCritic(), budget=128, pop=8, seed=1, strategy='cma', patience=1)
    assert res['restarts'] >= 1 and any(g.get('restart') for g in res['gens'])
    for h in res['history']:
        assert 200.0 <= h['params']['f'] <= 4000.0 and h['params']['shape'] in ('soft', 'hard')


def test_mirror_keeps_points_in_the_unit_cube():
    x = np.array([-0.3, 1.4, 0.5, 2.6, -1.2])
    u = search._mirror(x)
    assert np.allclose(u, [0.3, 0.6, 0.5, 0.6, 0.8]) and (u >= 0).all() and (u <= 1).all()


# ---------------------------------------------------------------------------------------------- pre-screen
def test_prescreen_scores_only_the_best_pop_and_keeps_the_starts(tmp_path):
    c = FakeCritic()
    pre = Prescreen(_refs(tmp_path))
    res = optimize(toy(), c, budget=48, pop=8, seed=2, strategy='cma', prescreen=32, pre=pre)
    assert c.heard == 48 and len(res['history']) == 48
    assert res['timing']['rendered'] == 6 * 32 and res['prescreen'] == 32
    assert res['history'][0]['params']['f'] == 300.0                                 # DEFAULTS scored in gen 0
    assert all('pre' in h and 'desc' in h for h in res['history'])
    late = [h for h in res['history'] if h['gen'] >= 3]
    assert np.median([h['pre']['tone_db'] for h in late]) < np.median([h['pre']['tone_db'] for h in res['history'][:8]])


def test_prescreen_ranks_a_ref_like_sound_first(tmp_path):
    pre = Prescreen(_refs(tmp_path))
    like, other, broken = _ping(1000, 0.03), _ping(250, 0.15), _ping(1000, 0.03, fade=False)
    broken[:20] = 0.9
    rows = pre.score([like, other, broken, None])
    assert rows[0]['pre'] > rows[1]['pre'] and rows[0]['pre'] > rows[2]['pre'] and rows[3]['pre'] == -np.inf
    assert rows[0]['tone_db'] < rows[1]['tone_db']
    assert 'start_jump' in rows[2]['art']['flags']
    assert pre.nearest(rows[0]['desc']) is not None
    assert Prescreen([], polish=False).score([like])[0]['pre'] == 0.0                 # no refs: artifacts only
    nr = Prescreen([]).score([like, broken])                                          # polish's absolute limits
    assert nr[0]['pre'] <= 0 and 'tone_db' not in nr[0]


def test_prescreen_uses_the_polish_verdict_when_there_is_one(tmp_path):
    class Pol:
        def check(self, x):
            return {'art_pen': 0.5, 'polish_pen': 1.0, 'hard': ['clicks']}
    pre = Prescreen(_refs(tmp_path), polish_ref=Pol())
    r = pre.row(_ping(1000, 0.03))
    assert r['polish']['hard'] == ['clicks']
    base = Prescreen(_refs(tmp_path), polish=False).row(_ping(1000, 0.03))
    assert abs((base['pre'] - r['pre']) - (0.5 + 0.75 + 1.0)) < 1e-6


def test_render_many_matches_serial_rendering():
    rec = toy()
    jobs = [({'f': 300.0 + 100 * i, 'tau': 0.03, 'shape': 'hard', 'n': 1}, 11 + i) for i in range(6)]
    a = render_many(rec, jobs, workers=1)
    b = render_many(rec, jobs, workers=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_es_with_threads_is_unchanged():
    a = optimize(toy(), FakeCritic(), budget=32, pop=16, seed=5, workers=1)
    b = optimize(toy(), FakeCritic(), budget=32, pop=16, seed=5, workers=4)
    assert [h['params'] for h in a['history']] == [h['params'] for h in b['history']]


# ---------------------------------------------------------------------------------------------- measurements
def test_envelope_stats_of_an_exponential_decay():
    tau = 0.04
    e = envelope_stats(_ping(1000, tau, n=SR))
    assert abs(e['t20_ms'] - 2.303 * tau * 1000) < 8 and abs(e['t40_ms'] - 4.605 * tau * 1000) < 12
    assert e['attack_ms'] <= 4 and e['t10_ms'] < e['t20_ms'] < e['t40_ms']


def test_artifacts_flag_clicks_cuts_gaps_and_clipping():
    clean = _ping(1000, 0.03)
    assert artifacts(clean)['flags'] == {} and artifacts(clean)['penalty'] == 0
    cut = _ping(1000, 0.3, n=SR // 10, fade=False)
    assert 'cut_end' in artifacts(cut)['flags']
    gap = clean.copy()
    gap[2400:3000] = 0.0
    assert 'gap' in artifacts(gap)['flags']
    clip = np.clip(3 * clean, -1, 1)
    assert 'flat_top' in artifacts(clip)['flags']
    dc = clean + 0.2
    assert 'dc' in artifacts(dc)['flags']


def test_sound_dist_separates_different_sounds():
    a, b, c = describe(_ping(1000, 0.03)), describe(_ping(1010, 0.031)), describe(_ping(300, 0.15))
    assert sound_dist(a, b) < 2.0 < sound_dist(a, c)


def test_diverse_prefers_different_sounds_then_fills_by_params():
    def h(u, total, x):
        return {'u': np.array(u), 'score': {'total': total, 'passes': True}, 'desc': describe(x)}
    same = _ping(1000, 0.03)
    res = [h([0.1, 0.1], 5.0, same), h([0.9, 0.9], 4.9, same), h([0.5, 0.2], 4.0, _ping(300, 0.15))]
    picks = diverse(res, toy(), k=2)
    assert [p['score']['total'] for p in picks] == [5.0, 4.0]                        # the sound twin is skipped
    picks3 = diverse(res, toy(), k=3)
    assert len(picks3) == 3                                                            # filled by the v1 rule


# ---------------------------------------------------------------------------------------------- seed bank
def test_seed_bank_round_trip_and_stays_small(tmp_path):
    folder = str(tmp_path)
    picks = [{'params': {'f': 1000.0 + i, 'tau': 0.03}, 'score': {'total': 3.0 + i, 'passes': True}}
             for i in range(8)]
    save_seeds('toy', picks, 'a ping', ['r1.wav'], folder=folder, per_context=3)
    bank = json.loads((tmp_path / 'toy.json').read_text(encoding='utf-8'))
    assert len(bank['entries']) == 3 and bank['entries'][0]['params']['f'] == 1007.0
    save_seeds('toy', [{'params': {'f': 500.0}, 'score': {'total': 9.0}, 'robust': {'mean': 1.0, 'pass_rate': 0}}],
               'another target', [], folder=folder, per_context=3)
    got = load_seeds('toy', 'a ping', ['r1.wav'], folder=folder)
    assert got[0]['f'] == 1007.0 and {'f': 500.0} in got                              # same context first
    got2 = load_seeds('toy', 'another target', [], folder=folder)
    assert got2[0] == {'f': 500.0}
    assert load_seeds('nothing_here', folder=folder) == []


# ---------------------------------------------------------------------------------------------- forge outputs
def test_forge_writes_report_md_compare_png_and_audition(tmp_path):
    c = FakeCritic()
    c.refs = _refs(tmp_path)
    out = tmp_path / 'made'
    rep = forge(toy(), budget=32, pop=8, keep=2, seed=4, out=str(out), log=None, critic=c, robust=2, shortlist=2,
                strategy='cma', prescreen=24)
    assert len(rep['picks']) == 2 and rep['strategy'] == 'cma' and rep['prescreen'] == 24
    for f in ('report.md', 'report.json', 'compare.png', 'audition.wav', 'audition.txt'):
        assert (out / f).exists(), f
    md = (out / 'report.md').read_text(encoding='utf-8')
    assert '## Search' in md and '## Picks' in md and '## Gap to the real references' in md
    assert md.count('| real ref_') == 4 and 'measured vs the refs' in md
    cues = (out / 'audition.txt').read_text(encoding='utf-8').splitlines()[1:]
    kinds = [ln.split()[2] for ln in cues]
    assert kinds == ['made', 'ref', 'made', 'ref']
    x, sr = sf.read(str(out / 'audition.wav'))
    assert sr == SR and x.shape[1] == 2 and abs(20 * np.log10(np.abs(x).max()) + 3) < 0.1
    for p in rep['picks']:
        assert p['robust']['n'] == 2 and p['measured']['nearest_ref'].startswith('ref_')
    assert rep['gap'].startswith('Measured gap')
    disk = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    assert len(disk['history']) == 32 and disk['timing']['scored'] == 32 and len(disk['gens']) == 4


def test_report_md_without_refs():
    rep = forge(toy(), budget=16, pop=8, keep=1, seed=1, log=None, critic=FakeCritic())
    md = report_md(rep)
    assert md.startswith("# forge toy: 'a 1 kHz ping'") and '## Gap' not in md


def test_audition_alternates_and_peak_matches(tmp_path):
    made = [('m1', 0.1 * _ping(1000, 0.03)), ('m2', _ping(800, 0.02))]
    refs = [('r1', 0.5 * _ping(990, 0.03))]
    a = audition(str(tmp_path / 'a.wav'), made, refs, [0, None])
    assert [c['kind'] for c in a['cues']] == ['made', 'ref', 'made']
    assert a['cues'][1]['t'] > a['cues'][0]['t']


def test_render_params_from_a_report_and_from_a_dict(tmp_path):
    rep = {'picks': [{'params': {'f': 1000.0, 'tau': 0.03, 'shape': 'hard', 'n': 1}, 'seed': 42}]}
    r = render_params(toy(), rep, out=str(tmp_path))
    assert r['files'][0]['seed'] == 42 and os.path.exists(r['files'][0]['file'])
    r2 = render_params(toy(), {'f': 500.0}, seeds=[1, 2], out=str(tmp_path), refs=_refs(tmp_path))
    assert [f['seed'] for f in r2['files']] == [1, 2] and 'tone_db' in r2['files'][0]['measured']
    a = sf.read(r2['files'][0]['file'])[0]
    assert abs(_peak_hz(a) - 500) < 10                                                # DEFAULTS filled the rest


def test_cli_list_and_render(tmp_path, capsys):
    from earshot.cli import main
    assert main(['forge', '--list']) == 0
    out = capsys.readouterr().out
    assert 'keypress: ' in out and 'recipes (name: TARGET)' in out
    pj = tmp_path / 'p.json'
    pj.write_text(json.dumps({}), encoding='utf-8')
    assert main(['forge', '--render', 'pop', '--params', str(pj), '--seeds', '3', '--dir', str(tmp_path)]) == 0
    assert (tmp_path / 'pop_r01_s3.wav').exists()
