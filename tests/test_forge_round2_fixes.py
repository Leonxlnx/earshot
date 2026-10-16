"""Round-2 integration fixes (forge v2): regression tests for the core requests the makers reported."""
import json
import os
import sys

import numpy as np

from earshot.forge import dsp, fx, search, tone

SR = 48000
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_reverb_ir_with_8_lines_is_finite_when_the_sign_columns_coincide():
    # seeds 19 and 135 draw equal / opposite L / R sign columns with 8 lines: Gram-Schmidt zeroed column 1 -> NaN IR
    for s in (19, 135, 0):
        ir = fx.reverb_ir(dsp.rng(s), rt60=0.3, lines=8, tail=0.05, air=0)
        assert np.isfinite(ir).all()
        assert np.abs(ir[:, 0, 1]).max() > 0


def test_tone_ref_with_one_ref_scored_leave_one_out_returns_nothing():
    t = np.arange(SR // 4) / SR
    x = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.05)
    ref = tone.ToneRef([x])
    assert ref.check(x, exclude=0) == {}
    assert ref.check(x)['tone_db'] < 0.01


def test_flat_top_needs_exact_repeats_not_a_slow_sine_crest():
    t = np.arange(SR) / SR
    sub = 0.8 * np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.4)        # dwells ~50 samples within 2 % of its crest
    sub[:240] *= np.linspace(0, 1, 240)
    sub[-2400:] *= np.linspace(1, 0, 2400) ** 2
    assert 'flat_top' not in search.artifacts(sub)['flags']
    assert 'flat_top' in search.artifacts(np.clip(3 * sub, -1, 1))['flags']


def test_seed_bank_default_cap_holds_several_categories(tmp_path):
    for c in range(4):
        picks = [{'params': {'f': 100.0 * c + i}, 'score': {'total': float(i)}} for i in range(6)]
        search.save_seeds('toy', picks, f'target {c}', [], folder=str(tmp_path))
    bank = json.loads((tmp_path / 'toy.json').read_text(encoding='utf-8'))
    assert len(bank['entries']) == 24
    assert {e['context'] for e in bank['entries']}.__len__() == 4


def test_optimize_keeps_defaults_outside_a_narrow_recipe_space():
    seen = []

    class Rec:
        SPACE = {'f': (200.0, 2000.0, 'log')}
        DEFAULTS = {'f': 500.0, 'tau': 0.02}

        @staticmethod
        def render(p, seed=0):
            seen.append(dict(p))
            t = np.arange(SR // 10) / SR
            return np.sin(2 * np.pi * p['f'] * t) * np.exp(-t / p.get('tau', 0.2))

    class Crit:
        refs = []

        def score(self, xs, E=None, exclude=None):
            return [{'total': float(-abs(np.log(np.abs(np.fft.rfft(x)).argmax() + 1.0))), 'passes': False}
                    for x in xs]

    search.optimize(Rec, Crit(), budget=8, pop=4, seed=1, log=None)
    assert seen and all(p.get('tau') == 0.02 for p in seen)


def test_bench_gather_reads_final_and_skips_audition(tmp_path):
    sys.path.insert(0, os.path.join(REPO, 'eval', 'forge'))
    import bench
    import soundfile as sf
    d = tmp_path / 'click' / 'final'
    d.mkdir(parents=True)
    for name in ('click_01.wav', 'click_02.wav', 'audition.wav', '_scratch.wav'):
        sf.write(str(d / name), np.zeros(480, np.float32), SR)
    sf.write(str(tmp_path / 'click' / 'old_round.wav'), np.zeros(480, np.float32), SR)
    got = bench.gather(str(tmp_path), [f'click={tmp_path}/click/final/*.wav'])
    names = [os.path.basename(p) for p in got['click']]
    assert names == ['click_01.wav', 'click_02.wav', 'click_01.wav', 'click_02.wav']
    assert all(c in bench.PHRASES and c in bench.CONFUSIONS and c in bench.AS_LABELS
               for c in ('pro_click', 'pro_zip', 'pro_typing_run', 'pro_logo_sting'))
