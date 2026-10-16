"""earshot gallery: the static HTML page of forge results (tiny synthetic WAVs only, no models)."""
import json
import os
import re

import numpy as np
import pytest
import soundfile as sf

from earshot import gallery as G

SR = 48000


def _tone(f=1000.0, dur=0.12, amp=0.3, seed=0, noise=0.02):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.RandomState(seed)
    x = amp * np.sin(2 * np.pi * f * t) * np.exp(-t / (dur / 4)) + noise * rng.randn(len(t)) * np.exp(-t / 0.01)
    return np.stack([x, x], axis=1).astype(np.float32)


def _write(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, SR, subtype='PCM_24')
    return path.replace(os.sep, '/')


def _report(d, picks, **meta):
    with open(os.path.join(d, 'report.json'), 'w', encoding='utf-8') as fh:
        json.dump({'recipe': meta.get('recipe', 'click'), 'target': meta.get('target', 'a clean bright click'),
                   'mode': 'designed', 'method': 'test family', 'picks': picks}, fh)


@pytest.fixture
def tree(tmp_path):
    """root/<category>/final/*.wav (+ report.json, an audition reel and a folder the gallery must skip)"""
    root = tmp_path / 'forge'
    c = root / 'click' / 'final'
    f1 = _write(str(c / 'click_01.wav'), _tone(3000, 0.08, seed=1))
    f2 = _write(str(c / 'click_02.wav'), _tone(1800, 0.10, seed=2))
    _write(str(c / 'audition.wav'), _tone(500, 0.5))
    _report(str(c), [
        {'name': 'click_01', 'file': f1, 'family': 'soft', 'seed': 11, 'params': {'pitch': 3000.5, 'bright': 0.25},
         'score': {'total': 2.5, 'passes': True, 'tone_db': 7.1, 'nearest_ref': 'r_b.wav'},
         'measured': {'t40_ms': 42.0}},
        {'name': 'click_02', 'file': f2, 'family': 'deep', 'seed': 12, 'params': {'pitch': 1800.0, 'body': 0.4},
         'score': {'total': -0.5, 'passes': False, 'polish_words': 'polish: tail dies fast'}}])
    (root / 'click' / 'LOG.md').write_text('# log\n', encoding='utf-8')
    w = root / 'whoosh_fast' / 'final'
    _write(str(w / 'whoosh_fast_01.wav'), _tone(400, 0.4, noise=0.2, seed=3))
    _write(str(root / '_judge' / 'final' / 'x_01.wav'), _tone())          # skipped: starts with _
    return root


def _html(res):
    with open(res['html'], encoding='utf-8') as fh:
        return fh.read()


def test_split_label_and_category():
    assert G._split_label('r2:out/forge_v2/*/final_r2') == ('r2', 'out/forge_v2/*/final_r2')
    assert G._split_label('E:/earshot/out') == (None, 'E:/earshot/out')
    assert G._split_label('eval/forge/results.json') == (None, 'eval/forge/results.json')
    assert G.category_of('out/forge_v2/click/final_r2') == 'click'
    assert G.category_of('out/forge_v2/click') == 'click'
    assert G.family_of('whoosh_fast') == 'whoosh' and G.family_of('pro_zip') == 'sweep'
    assert G.family_of('pro_new_thing_click') == 'ui' and G.family_of('mystery') == 'other'


def test_lufs_and_spectrogram():
    t = np.arange(SR) / SR
    x = (0.1 * np.sin(2 * np.pi * 997 * t)).astype(np.float32)[:, None]   # -20 dBFS sine, one channel
    assert abs(G.lufs_max(x) - (-23.0)) < 0.3
    m = G.measure(np.repeat(x, 2, axis=1), SR)
    assert abs(m['peak_dbfs'] + 20) < 0.1 and m['dur_s'] == 1.0
    rgb = G.spectrogram_rgb(x, SR, 1.0)
    assert rgb.shape == (G.SPEC_H, G.SPEC_W, 3) and rgb.dtype == np.uint8
    row = int(np.argmin(rgb.astype(int).sum(axis=(1, 2))))               # darkest row = the tone
    fy = np.geomspace(G.F_LO, G.F_HI, G.SPEC_H + 1)
    f_row = np.sqrt(fy[:-1] * fy[1:])[::-1][row]
    assert 800 < f_row < 1250


def test_gallery_page(tree, tmp_path):
    out = tmp_path / 'site' / 'g.html'
    res = G.gallery(str(tree), str(out))
    html = _html(res)
    assert res['categories'] == 2 and res['sounds'] == 3 and res['refs'] == 0
    assert html.startswith('<!doctype html>') and '<script src' not in html and 'http' not in html.split('<body>')[1]
    srcs = re.findall(r'<audio controls preload="none"[^>]*? src="([^"]+)"', html)
    assert len(srcs) == 3 and all(s.startswith('../forge/') for s in srcs)
    assert len(re.findall(r'<audio [^>]*data-lufs="-?[0-9.]+"', html)) == 3 and 'id="lm"' in html   # loudness match
    assert 'audition' not in ''.join(srcs)                       # the reel is not a sound (and has refs inside)
    for s in srcs:
        assert os.path.exists(os.path.join(os.path.dirname(res['html']), s))
    pngs = re.findall(r'<img class="spec" loading="lazy" src="([^"]+)"', html)
    assert len(pngs) == 3 and all(os.path.exists(os.path.join(os.path.dirname(res['html']), p)) for p in pngs)
    assert html.index('id="whoosh_fast"') < html.index('id="click"')    # family order: whoosh before ui
    assert 'soft' in html and 'deep' in html and 'critic pass <b>1 / 2</b>' in html
    assert '>pitch<' in html and '>3000<' in html and '>bright<' in html  # parameters table
    assert 'polish: tail dies fast' in html and 'LOG.md' in html and '_judge' not in html
    # a second build reuses the thumbnails; a removed sound's thumbnail is pruned
    os.remove(str(tree / 'whoosh_fast' / 'final' / 'whoosh_fast_01.wav'))
    res2 = G.gallery(str(tree), str(out))
    assert res2['images']['made'] == 0 and res2['images']['reused'] == 2 and res2['images']['pruned'] == 1


def test_results_numbers_and_staleness(tree, tmp_path):
    made = [{'file': 'out/forge/click/final/click_01.wav', 'length_s': 0.08, 'tone_db': 9.86, 't40_ratio': 1.22,
             'artifacts': [], 'artifact_detail': {'peak_dbfs': -10.5}},
            {'file': 'out/forge/click/final/click_02.wav', 'length_s': 0.5, 'tone_db': 4.0, 't40_ratio': 3.0}]
    x01, _ = sf.read(str(tree / 'click' / 'final' / 'click_01.wav'))
    made[0]['artifact_detail']['peak_dbfs'] = round(20 * np.log10(np.abs(x01).max()), 2)
    res_json = tmp_path / 'results.json'
    res_json.write_text(json.dumps({'tag': 'tst', 'created': '2000-01-01 00:00', 'categories': {'click': {
        'made': made, 'inside': {'per_sound': [6, 2], 'n_checks': 7, 'real_per_sound': [5, 6]},
        'bench_refs': ['E:/secret/bench/held_out_take.wav'],
        'summary': {'made': {'n': 2, 'var_cos': 0.5, 't40_ratio': 1.5, 'tone_db': 7.0}}}}}), encoding='utf-8')
    res = G.gallery(str(tree), str(tmp_path / 'g.html'), results=str(res_json))
    html = _html(res)
    assert 'held_out_take' not in html                                   # bench file names never reach a page
    # created in 2000: every file is newer than the run, so no benchmark number may be shown
    assert '9.9 dB' not in html and any('newer than the benchmark' in w for w in res['warnings'])
    data = json.loads(res_json.read_text(encoding='utf-8'))
    data['created'] = '2999-01-01 00:00'
    res_json.write_text(json.dumps(data), encoding='utf-8')
    res = G.gallery(str(tree), str(tmp_path / 'g.html'), results=str(res_json))
    html = _html(res)
    assert '9.9 dB' in html and '1.22 x' in html and '6 / 7' in html   # click_01 matches (length and peak)
    assert '3.00 x' not in html                                         # click_02: length differs, withheld
    assert any('click_02.wav (length' in w for w in res['warnings'])
    assert 'benchmark numbers withheld' in html                         # no category summary from a partial match


def test_compare_versions(tree, tmp_path):
    old = tmp_path / 'old'
    _write(str(old / 'click' / 'final_r2' / 'click_01.wav'), _tone(3000, 0.08, seed=1))   # same bytes as r3
    _write(str(old / 'click' / 'final_r2' / 'click_02.wav'), _tone(900, 0.2, seed=9))
    specs = [f'r2:{old}/*/final_r2', f'r3:{tree}/*/final']
    res = G.gallery(compare=specs, out=str(tmp_path / 'cmp.html'))
    html = _html(res)
    assert res['versions'] == {'r2': 1, 'r3': 2}
    assert '>= r2</span>' in html                                        # click_01 unchanged
    assert html.count('class="rlabel">r2') == 2 and 'no r2 sounds for this category' in html
    with pytest.raises(ValueError, match='label each results'):
        G.gallery(compare=specs, out=str(tmp_path / 'cmp.html'), results='x.json')
    with pytest.raises(ValueError, match='LABEL:GLOB'):
        G.compare_versions([str(old)])


def test_embed(tree, tmp_path):
    res = G.gallery(str(tree), str(tmp_path / 'e.html'), embed=True)
    html = _html(res)
    assert html.count('src="data:audio/flac;base64,') == 3 and html.count('src="data:image/png;base64,') == 3
    assert '.wav"' not in html and res['embedded']
    res = G.gallery(str(tree), str(tmp_path / 'w.html'), embed=True, codec='wav')
    assert _html(res).count('src="data:audio/wav;base64,') == 3


def test_refs_never_bench(tree, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='held out'):
        G.gallery(str(tree), str(tmp_path / 'g.html'), refs='bench')
    refdir = tmp_path / 'refs'
    a = _write(str(refdir / 'click' / 'r_a.wav'), _tone(2500, 0.07, seed=5))
    b = _write(str(refdir / 'click' / 'r_b.wav'), _tone(2600, 0.07, seed=6))
    _write(str(refdir / 'click' / 'whoosh_med_01.wav'), _tone(2600, 0.07, seed=7))   # a bench-source kit stem
    reg = tmp_path / 'refsets.json'
    reg.write_text(json.dumps({'version': 1, 'categories': {'click': {'forge': [b], 'bench': [a], 'note': ''}}}),
                   encoding='utf-8')
    monkeypatch.setenv('EARSHOT_REFSETS', str(reg))
    res = G.gallery(str(tree), str(tmp_path / 'g.html'), refs=str(refdir))
    html = _html(res)
    assert 'r_a.wav' not in html and 'whoosh_med_01' not in html         # registry bench + kit bench dropped
    assert 'r_b.wav' in html and res['refs'] == 1                        # the one forge ref for both: shown once
    assert any('held-out reference' in w for w in res['warnings'])
    assert 'class="audition"' not in html                               # no index file: its refs cannot be checked
    idx = tree / 'click' / 'final' / 'audition.txt'
    reel = 'audition.wav: ...\n    0.00 s  made  click_01 soft\n    0.50 s  ref   {}\n'
    idx.write_text(reel.format('r_b.wav'), encoding='utf-8')
    assert 'class="audition"' in _html(G.gallery(str(tree), str(tmp_path / 'g.html'), refs=str(refdir)))
    idx.write_text(reel.format('r_a.wav'), encoding='utf-8')
    res = G.gallery(str(tree), str(tmp_path / 'g.html'), refs=str(refdir))
    assert 'class="audition"' not in _html(res) and any('plays held-out refs' in w for w in res['warnings'])


def test_cli(tree, tmp_path, capsys):
    from earshot.cli import main
    out = tmp_path / 'c.html'
    assert main(['gallery', str(tree), '-o', str(out), '--json']) == 0
    j = json.loads(capsys.readouterr().out)
    assert j['sounds'] == 3 and os.path.exists(out)
    assert main(['gallery']) == 1                                         # neither folder nor --compare


def test_align_variants():
    snd = lambda n, v: {'name': n, 'variant': v}
    r2 = {'sounds': [snd('a1', 'crisp'), snd('a2', 'tight'), snd('a3', 'heavy')]}
    r3 = {'sounds': [snd('b1', 'crisp'), snd('b2', 'sleeve'), snd('b3', 'heavy'), snd('b4', 'tight')]}
    G.align_variants([r2, r3])
    assert [q['name'] for q in r2['order']] == ['a1', 'a2', 'a3']
    assert [q['name'] for q in r3['order']] == ['b1', 'b4', 'b3', 'b2']        # crisp, tight, heavy under r2's
    r4 = {'sounds': [snd('c1', 'x'), snd('c2', 'crisp')]}                      # one shared name: file order kept
    G.align_variants([r2, r4])
    assert [q['name'] for q in r4['order']] == ['c1', 'c2']
