"""earshot gallery: the static HTML page of forge results (tiny synthetic WAVs only, no models)."""
import base64
import glob
import json
import os
import re
import shutil

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
    assert G._split_label('D:/data/out') == (None, 'D:/data/out')
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
        'bench_refs': ['D:/secret/bench/held_out_take.wav'],
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
    # embedded: the unchanged click_01 goes into the page once, its second player copies the first one's source
    res = G.gallery(compare=specs, out=str(tmp_path / 'cmp_e.html'), embed=True)
    html = _html(res)
    assert html.count('src="data:audio/flac;base64,') == 4 and html.count('data-src-of="a') == 1
    first = re.search(r'data-src-of="(a\d+)"', html).group(1)
    assert f'id="{first}" src="data:audio/flac' in html and 'data-src-of' in G.JS
    with pytest.raises(ValueError, match='label each results'):
        G.gallery(compare=specs, out=str(tmp_path / 'cmp.html'), results='x.json')
    with pytest.raises(ValueError, match='LABEL:GLOB'):
        G.compare_versions([str(old)])
    # an older version's folder under another name lands on the newer category's card
    _write(str(tmp_path / 'v1' / 'keypress' / 'final' / 'keypress_01.wav'), _tone(2200, 0.09, seed=4))
    res = G.gallery(compare=[f'v1:{tmp_path}/v1/*/final', f'v2:{tree}/*/final'], out=str(tmp_path / 'a.html'),
                    alias=['keypress=click'])
    assert res['categories'] == 2 and res['versions'] == {'v1': 1, 'v2': 2}
    assert 'keypress_01' in _html(res) and 'id="keypress"' not in _html(res)
    with pytest.raises(ValueError, match='OLD=NEW'):
        G.parse_alias(['keypress'])


def test_embed(tree, tmp_path):
    res = G.gallery(str(tree), str(tmp_path / 'e.html'), embed=True)
    html = _html(res)
    assert html.count('src="data:audio/flac;base64,') == 3 and html.count('src="data:image/png;base64,') == 3
    assert '.wav"' not in html and res['embedded']
    res = G.gallery(str(tree), str(tmp_path / 'w.html'), embed=True, codec='wav')
    assert _html(res).count('src="data:audio/wav;base64,') == 3


needs_ffmpeg = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='needs ffmpeg on PATH')


@needs_ffmpeg
def test_embed_compressed(tree, tmp_path):
    res_json = tmp_path / 'results.json'
    res_json.write_text(json.dumps({'tag': 'tst', 'created': '2999-01-01 00:00', 'categories': {'click': {
        'made': [], 'bench_refs': ['D:/secret/bench/held_out_take.wav']}}}), encoding='utf-8')
    out = tmp_path / 'share' / 'one.html'
    res = G.gallery(str(tree), str(out), results=str(res_json), compressed=True)
    html = _html(res)
    assert res['embedded'] and res['codec'] == 'mp3 192 kbps' and 'previews 3 encoded' in res['text']
    mp3 = re.findall(r'<audio [^>]*src="data:audio/mpeg;base64,([^"]+)"', html)
    assert len(mp3) == 3 and len(re.findall(r'<audio [^>]*data-lufs=', html)) == 3       # loudness match still on
    raw = base64.b64decode(mp3[0])
    assert raw[:3] == b'ID3' or (raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0)           # an MP3 stream
    assert html.count('src="data:image/jpeg;base64,') == 3                             # thumbnails as JPEG
    # one self-contained file: nothing it would have to reach next to it, no machine paths, no bench names
    assert '.wav"' not in html and 'file://' not in html and 'LOG.md' not in html and 'report.md' not in html
    assert 'held_out_take' not in html and str(tmp_path).replace(os.sep, '/') not in html
    assert '<script src' not in html and 'http' not in html.split('<body>')[1]
    assert 'MP3 192 kbps' in html                                                      # says what plays
    # a second build takes the previews from the cache; a removed sound's preview is pruned
    res2 = G.gallery(str(tree), str(out), compressed=True)
    assert 'previews 0 encoded' in res2['text'] and _html(res2).count('data:audio/mpeg') == 3
    os.remove(str(tree / 'whoosh_fast' / 'final' / 'whoosh_fast_01.wav'))
    G.gallery(str(tree), str(out), compressed=True)
    assert len(os.listdir(str(tmp_path / 'share' / 'one_files' / 'enc'))) == 2
    # Ogg Opus at a set rate; a size limit only adds a note
    res = G.gallery(str(tree), str(tmp_path / 'o.html'), compressed=True, codec='opus', kbps=64, max_mb=1e-4)
    html = _html(res)
    assert res['codec'] == 'opus 64 kbps' and html.count('src="data:audio/ogg;base64,') == 2
    assert base64.b64decode(re.findall(r'data:audio/ogg;base64,([^"]+)"', html)[0])[:4] == b'OggS'
    assert any('over 0.0001 MB' in w for w in res['warnings'])


def test_embed_compressed_args(tree, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='kbps applies'):
        G.gallery(str(tree), str(tmp_path / 'x.html'), kbps=96)                       # not a lossy embed
    with pytest.raises(ValueError, match='between 16 and 320'):
        G.gallery(str(tree), str(tmp_path / 'x.html'), compressed=True, kbps=4)
    with pytest.raises(ValueError, match='codec must be'):
        G.gallery(str(tree), str(tmp_path / 'x.html'), embed=True, codec='aiff')
    with pytest.raises(ValueError, match='codec must be'):
        G.encode_lossy(str(tree / 'click' / 'final' / 'click_01.wav'), 'flac')
    monkeypatch.setattr(shutil, 'which', lambda name: None)   # no ffmpeg: a clear error, not a page that cannot play
    with pytest.raises(RuntimeError, match='needs ffmpeg'):
        G.gallery(str(tree), str(tmp_path / 'x.html'), compressed=True)


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
    assert not any('keep it private' in w for w in res['warnings'])
    res = G.gallery(str(tree), str(tmp_path / 'e.html'), refs=str(refdir), embed=True)   # licensed audio inside
    assert 'keep it private' in res['warnings'][0] and 'keep it private' in _html(res)
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
    if shutil.which('ffmpeg'):
        assert main(['gallery', str(tree), '-o', str(tmp_path / 'm.html'), '--embed-compressed', '--kbps', '96',
                     '--json']) == 0
        j = json.loads(capsys.readouterr().out)
        assert j['codec'] == 'mp3 96 kbps' and j['embedded_mb']['audio'] > 0


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


def _scrub_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval', 'forge', 'scrub_paths.py')
    spec = importlib.util.spec_from_file_location('scrub_paths_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_results_have_no_machine_paths():
    """the committed benchmark results (which --results reads) carry no machine paths: run eval/forge/scrub_paths.py
    after every eval/forge/bench.py run"""
    S = _scrub_module()
    files = sorted(glob.glob(os.path.join(S.HERE, 'results_*.json')) + glob.glob(os.path.join(S.HERE, 'results_*.md')))
    assert files
    for f in files:
        changed, left = S.scrub_file(f, write=False)
        assert not changed and not left, f'{os.path.basename(f)} has machine paths: run eval/forge/scrub_paths.py'


def test_scrub_rules():
    S = _scrub_module()
    S._COMPILED = [(S._pattern(r), ph) for r, ph in (('D:/lab/out/refs', '<refs>'), ('D:/lab/.cache', '<EARSHOT_HOME>'),
                                                      ('D:/lab', ''))]
    assert S.scrub_text(r'D:\lab\out\refs\click\bench\a.wav') == '<refs>/click/bench/a.wav'
    assert S.scrub_text(r'refs: d:\LAB\.cache\refsets.json; 2026') == 'refs: <EARSHOT_HOME>/refsets.json; 2026'
    got = S.scrub({'made': 'D:/lab/out/forge_v2/click/final/click_01.wav', 'other': ['D:/labx/a.wav', 3]})
    assert got == {'made': 'out/forge_v2/click/final/click_01.wav', 'other': ['D:/labx/a.wav', 3]}   # labx: no root
    assert S.leftovers('see D:/labx/a.wav and http://x.org/a') == ['D:/labx/a.wav']
