"""The fit engine's choices and the takes' clicks (earshot 0.4, round C2), on synthetic sounds only:
- a bed is kept only when analysis by synthesis says so: the winner's kind is also fitted on the recording as it is,
  and the bed is dropped when that scores lower (a whoosh's own steady approach is not a bed);
- the CLAP margin rule picks a candidate CLAP clearly prefers (on seed 0 and on the confirm seeds' mean) only within
  the score and envelope tolerances;
- no clicks from the post stage's gate edges, from a later modal strike's onset, or from a coherent sines strike's
  atoms jittered apart; models fitted before 0.4 render as before."""
import numpy as np

from earshot.forge import polish
from earshot.forge.fit import auto, modal, sines

SR = 48000


def _db(v):
    return 10 * np.log10(np.mean(np.asarray(v, float) ** 2) + 1e-20)


def _strike(n, t0, f=3000.0, tau=0.02, amp=0.5):
    t = np.arange(n) / SR - t0
    return (t >= 0) * amp * np.exp(-np.clip(t, 0, None) / tau) * np.sin(2 * np.pi * f * np.clip(t, 0, None))


def _bed(n, db=-20.0):
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * 410 * t) + 0.6 * np.sin(2 * np.pi * 655 * t + 1.0) + 0.3 * np.random.default_rng(1).standard_normal(n)
    return y / np.sqrt(np.mean(y ** 2)) * 0.5 / np.sqrt(2) * 10 ** (db / 20)


# ------------------------------------------------------------------------------------------------------ bed check
def test_bed_check_records_both_scores_and_keeps_the_better():
    n = int(0.2 * SR)
    x = _strike(n, 0.06) + _bed(n)
    m = auto.fit(x, candidates=['modal'], fast=True)
    b = m['fit_engine']['bed']
    assert b['score_stripped'] is not None and b['score_plain'] is not None
    assert b['kept'] is (b['score_stripped'] <= b['score_plain'])
    assert bool((m.get('post') or {}).get('bed')) is b['kept']      # the model carries the bed only when kept
    assert any(r['candidate'] == 'plain:modal' for r in m['fit_engine']['candidates'])


def test_bed_check_drops_a_bed_whose_stripped_fit_loses(monkeypatch):
    n = int(0.2 * SR)
    x = _strike(n, 0.06) + _bed(n)
    ruin = lambda X, bd: 0.05 * np.asarray(X)[::-1]                 # a stripped "sound" that is not the sound
    monkeypatch.setattr(auto, 'strip_bed', ruin)
    m = auto.fit(x, candidates=['modal'], fast=True)
    b = m['fit_engine']['bed']
    assert b['kept'] is False and b['score_plain'] < b['score_stripped']
    assert 'bed' not in m['post']
    monkeypatch.setattr(auto, 'BED_CHECK', False)                   # without the check the stripped fit stands
    m2 = auto.fit(x, candidates=['modal'], fast=True)
    assert 'bed' in m2['post'] and 'kept' not in m2['fit_engine']['bed']


# ------------------------------------------------------------------------------------------------------ CLAP rule
def _row(clap, score, env=0.97, clap2=None, score2=None):
    return {'quality': {'clap_cos': clap, 'env_corr': env}, 'score': score, 'clap2': clap2, 'score2': score2}


def test_clap_rule_needs_the_margin_on_both_and_a_close_score():
    rows = {'gesture': _row(0.86, 10.0, clap2=0.85, score2=10.2),
            'sines': _row(0.90, 10.9, clap2=0.89, score2=11.1),     # clearly closer to CLAP, a near-tie on the score
            'modal': _row(0.95, 14.0, clap2=0.94, score2=14.5)}     # closer still, but far worse on the score
    alts = auto._clap_alts(rows, 'gesture')
    assert alts == ['sines']
    assert auto._clap_pick(rows, 'gesture', alts) == 'sines'
    rows['sines']['clap2'] = 0.86                                   # seed 0 was a lucky draw: the mean decides
    assert auto._clap_pick(rows, 'gesture', alts) is None
    rows['sines'].update(clap2=0.89, quality={'clap_cos': 0.90, 'env_corr': 0.80})   # a far worse loudness contour
    assert auto._clap_pick(rows, 'gesture', alts) is None
    assert auto._clap_alts({'a': _row(None, 9.0), 'b': _row(0.99, 9.1)}, 'a') == []


# ------------------------------------------------------------------------------------------------------ clicks
def test_gate_edges_do_not_click():
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    y = 0.5 * np.sin(2 * np.pi * 300 * t) * np.exp(-t / 0.3)        # a low ring under the cut
    gate = {'pts': [[0.05, 0.0], [0.0505, -60.0], [0.08, -60.0], [0.0805, 0.0], [0.12, 0.0], [0.1205, -40.0]]}
    g = auto._gate_gain(n, gate)
    assert np.max(np.abs(np.diff(g, 2))) < 0.01                     # no kinks: the corners are rounded
    assert polish.clicks(y * g)['clicks'] == 0
    assert _db((y * g)[int(0.055 * SR):int(0.075 * SR)]) < _db(y[int(0.055 * SR):int(0.075 * SR)]) - 55   # still shut
    assert 20 * np.log10(g[-1]) < -39                               # the final stop holds


def _two_strikes():
    n = int(0.25 * SR)
    return _strike(n, 0.01, f=900.0, tau=0.15) + _strike(n, 0.12, f=1300.0, tau=0.05, amp=0.35)


def test_later_modal_strike_starts_without_a_step():
    import copy
    m = modal.fit(_two_strikes(), passes=1)
    assert len(m['strikes']) >= 2 and m.get('coherent')
    st = m['strikes'][1]
    rows = np.asarray(st['modes'], float).reshape(-1, 4)
    rows[:, 3] = 0.0                                                # cosine phase: the strike starts on a step
    st['modes'] = rows.tolist()
    old = copy.deepcopy(m)
    old.pop('coherent')                                             # a model fitted before 0.4: rendered as it was
    y, y0 = modal.render(m, 0), modal.render(old, 0)
    o = int(round(st['t'] * SR))
    jump = lambda v: abs(v[o + 1] - v[o - 1])
    assert jump(y0) > 3 * jump(y)                                   # the 0.4 fit's later strike rises over 0.15 ms
    assert np.array_equal(y[:o], y0[:o])                            # nothing else moves
    assert np.allclose(y[o + int(modal.STRIKE_RAMP_S * SR) + 1:], y0[o + int(modal.STRIKE_RAMP_S * SR) + 1:])


def test_coherent_sines_strike_keeps_its_atoms_together():
    n = int(0.2 * SR)
    m = sines.fit(_strike(n, 0.02, f=700.0, tau=0.05), windows=(1024,))
    m['partials'] = []
    m['residual'] = {'bands': [1000.0], 'times': [0.0, m['duration']], 'db': [[-150.0, -150.0]]}
    m['transients'] = [{'t': 0.05, 'atoms': [[3000.0, 0.002, 1.0, 0.0], [3150.0, 0.002, -1.0, 0.0]]}]   # cancel at 0
    i0 = int(round(0.05 * SR))
    for seed in (1, 2, 3):
        y = sines.render(m, seed, knobs={'vary': 0.3, 'ring': False})
        assert abs(y[i0]) < 1e-6                                    # one level for both: their step still cancels
    m['coherent'] = False                                           # an older model: each atom jittered on its own
    assert max(abs(sines.render(m, s, knobs={'vary': 0.3, 'ring': False})[i0]) for s in (1, 2, 3)) > 1e-3


def test_levelled_grain_train_follows_its_expectation():
    from earshot.forge.fit import spectral
    n = SR // 2
    for rate in (10000.0, 2000.0):
        w = int(max(0.005 * SR, spectral.GRAIN_NORM_CLICKS * SR / rate)) | 1
        loud = lambda c: float(spectral._movavg(c ** 2, w).max())   # the loudest window re the train's mean power
        raw = [spectral._clicks(n, rate, np.random.default_rng(s)) for s in range(6)]
        lev = [spectral._clicks(n, rate, np.random.default_rng(s), norm_ms=5.0) for s in range(6)]
        assert all(abs(np.mean(c ** 2) - 1.0) < 1e-6 for c in lev)  # still unit power
        assert max(map(loud, lev)) < 2.2 < max(map(loud, raw))      # no window holds a draw's pile of loud clicks
    g = spectral._grains(1000.0)
    assert 'norm_ms' not in g and spectral._grains(1000.0, 5.0)['norm_ms'] == 5.0   # older textures render as before
    from earshot.forge.fit import gesture
    assert gesture.AIR_FIT['grain_norm_ms'] > 0 and gesture.AIR_FIT['grain_halves']
