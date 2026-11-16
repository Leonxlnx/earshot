"""earshot narration: scene cuts, cues and subtitles timed to a synthetic voiceover with known pauses."""
import json

import numpy as np
import pytest

from earshot import narration

SR = 48000


def _speech(plan, seed=0):
    """'syllables' (150 ms voiced bursts, 40 ms gaps) for each ('talk', seconds) and silence for ('gap', seconds)"""
    rng = np.random.RandomState(seed)
    out = []
    for kind, s in plan:
        n = int(s * SR)
        if kind == 'gap':
            out.append(rng.randn(n) * 1e-4)
            continue
        y = np.zeros(n)
        t = np.arange(int(0.15 * SR)) / SR
        k = int((n - len(t)) / (0.19 * SR)) + 1       # syllables spread so the last one ends on the segment end
        for a in np.linspace(0, n - len(t), k).astype(int):
            f = rng.uniform(140, 220)
            y[a:a + len(t)] = 0.3 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))
        out.append(y + rng.randn(n) * 1e-4)
    return np.concatenate(out).astype(np.float32)


# 3 scenes of 4 / 6 / 3 s speech; the inner sentence breaks are short (0.3 s), the scene breaks long (0.7 s)
PLAN = [('gap', 0.8), ('talk', 2.0), ('gap', 0.3), ('talk', 2.0), ('gap', 0.7), ('talk', 3.0), ('gap', 0.3),
        ('talk', 3.0), ('gap', 0.7), ('talk', 3.0), ('gap', 1.0)]
SCRIPT = {'scenes': [
    {'id': 'intro', 'text': 'aaaa aaaaa aaaa. bbbb bbbbb bbbb'},
    {'id': 'body', 'text': 'cccc ccccc cccc ccccc cccc. {pop}dddd ddddd dddd ddddd dddd',
     'cues': [{'id': 'hit', 'at': 'dddd ddddd', 'kind': 'hit'}]},
    {'id': 'end', 'text': 'eeee eeeee eeee eeeee eeee'}]}


def _edges():
    t, out = 0.0, []
    for kind, s in PLAN:
        out.append((kind, t, t + s))
        t += s
    return out


def test_weights_and_lines():
    w = narration.weights('ab, c.\nd  e')
    assert list(w) == [1, 1, 3, 1, 1, 6, 6, 1, 3, 0, 1]   # a space run weighs once
    assert narration.weights('กิ')[1] == 0           # a Thai vowel mark takes no time of its own
    text = 'one, two. ' + 'word ' * 20
    spans = narration.lines(text, max_chars=30)
    assert [text[a:b].strip() for a, b in spans][:2] == ['one,', 'two.']
    assert all(len(text[a:b].strip()) <= 30 for a, b in spans)


def test_analyse_finds_speech_and_pauses():
    an = narration.analyse(_speech(PLAN))
    assert abs(an['start'] - 0.8) < 0.05 and abs(an['end'] - 15.8) < 0.05
    long = [p for p in an['pauses'] if p['len'] > 0.5]
    assert len(long) == 2 and all(0.6 < p['len'] < 0.8 for p in long)
    assert len(an['pauses']) == 4   # syllable gaps (40 ms) are no pauses


def test_scene_cuts_snap_to_the_scene_pauses(wav, tmp_path):
    vo = wav(_speech(PLAN))
    r = narration.sync(vo, SCRIPT, lead_in=0.4, tail=2.5, envelope_fps=30)
    gaps = [(a, b) for k, a, b in _edges() if k == 'gap' and 0.6 < b - a < 0.8]
    off = r['vo_offset_s']
    assert abs(off - (0.4 - 0.8)) < 0.05
    for sc, (a, b) in zip(r['scenes'][1:], gaps):
        assert sc['cut'] == 'pause' and abs(sc['start'] - ((a + b) / 2 + off)) < 0.05
    assert r['scenes'][0]['start'] == 0 and r['scenes'][-1]['end'] == r['duration']
    assert abs(r['duration'] - (0.4 + 15.0 + 2.5)) < 0.05
    cues = {c['id']: c for c in r['cues']}
    d_start = _edges()[7][1] + off     # 'dddd' is the second sentence of 'body'
    assert cues['pop']['snapped'] and abs(cues['pop']['t'] - d_start) < 0.05
    assert cues['hit']['found'] and abs(cues['hit']['t'] - d_start) < 0.05 and cues['hit']['kind'] == 'hit'
    body = [s for s in r['subtitles'] if s['scene'] == 'body']
    assert [s['text'][:4] for s in body] == ['cccc', 'dddd'] and abs(body[1]['start'] - d_start) < 0.25
    assert len(r['envelope']) == int(np.ceil(r['duration'] * 30)) and max(r['envelope']) == 1.0
    assert r['envelope'][0] == 0.0                      # silence before the speech
    assert '2/2 scene cuts on a pause' in r['summary']


def test_shift_nudge_and_missing_cue(wav):
    vo = wav(_speech(PLAN))
    base = narration.sync(vo, SCRIPT)
    s = json.loads(json.dumps(SCRIPT))
    s['scenes'][2]['nudge'] = 0.2
    s['scenes'][2]['cues'] = [{'id': 'lost', 'at': 'not in the text'}]
    r = narration.sync(vo, s, shift=0.1)
    assert abs(r['scenes'][1]['start'] - base['scenes'][1]['start'] - 0.1) < 1e-6
    assert abs(r['scenes'][2]['start'] - base['scenes'][2]['start'] - 0.3) < 1e-6
    lost = next(c for c in r['cues'] if c['id'] == 'lost')
    assert not lost['found'] and 'not found' in r['summary']
    s['scenes'][0]['nudge'] = 0.1
    with pytest.raises(narration.NarrationError):
        narration.sync(vo, s)


def test_text_script_and_cli(wav, tmp_path, capsys):
    from earshot import cli
    vo = wav(_speech(PLAN))
    txt = tmp_path / 'script.txt'
    txt.write_text('aaaa aaaaa aaaa. bbbb bbbbb bbbb\n\ncccc ccccc cccc ccccc cccc. dddd ddddd dddd ddddd dddd\n\n'
                   '{logo}eeee eeeee eeee eeeee eeee\n', encoding='utf-8')
    srt = tmp_path / 'subs.srt'
    out = tmp_path / 'sync.json'
    assert cli.main(['narration', vo, str(txt), '--srt', str(srt), '--out', str(out)]) == 0
    assert 'scene cuts on a pause' in capsys.readouterr().out
    r = json.loads(out.read_text(encoding='utf-8'))
    assert [s['id'] for s in r['scenes']] == ['s1', 's2', 's3'] and r['cues'][0]['id'] == 'logo'
    assert srt.read_text(encoding='utf-8').startswith('1\n00:00:00,')


def test_short_script_cut_falls_back_to_text(wav):
    """no pause near a cut: it stays where the text puts it, and the summary says so"""
    vo = wav(_speech([('gap', 0.5), ('talk', 6.0), ('gap', 0.5)]))
    r = narration.sync(vo, ['aaaa aaaa aaaa', 'bbbb bbbb bbbb'])
    assert r['scenes'][1]['cut'] == 'text' and 'placed by text length' in r['summary']


def test_errors(wav, tmp_path):
    with pytest.raises(narration.NarrationError, match='no speech'):
        narration.sync(wav(np.zeros(SR, np.float32)), ['hello'])
    with pytest.raises(narration.NarrationError, match='text'):
        narration.load_script({'scenes': [{'id': 'x'}]})
    bad = tmp_path / 'bad.json'
    bad.write_text('{"scenes": [', encoding='utf-8')
    with pytest.raises(narration.NarrationError, match='not valid JSON'):
        narration.load_script(str(bad))
