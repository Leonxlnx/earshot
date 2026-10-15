"""timeline + sheet: synthetic signals only; model-free unless marked slow (the CLAP / AST steps are stubbed)."""
import json
import os

import numpy as np
import pytest

from earshot import sheet as sheet_mod
from earshot import timeline as tl

SR = 48000


def click_track(times, dur, amp=0.5, bed=0.01, seed=0, sr=SR):
    """decaying noise bursts at `times` over a quiet noise bed"""
    rng = np.random.RandomState(seed)
    x = rng.randn(int(dur * sr)).astype(np.float32) * bed
    n = int(0.05 * sr)
    env = np.exp(-np.arange(n) / sr / 0.008)
    for t in times:
        a = int(round(t * sr))
        x[a:a + n] += (rng.randn(n) * env * amp)[: len(x) - a]
    return x


@pytest.fixture
def no_models(monkeypatch):
    """replace the model steps (CLAP tags, sections) so the pipeline runs in milliseconds"""
    def fake_tags(x, fid, t0, events, n_tag, n_ast, near=(), reuse=True, win=None):
        for e in sorted(events, key=lambda e: -e['prominence_db'])[:n_tag]:
            e.update(kind='HIT', label='punchy impact hit', tags=[{'tag': 'a punchy impact hit', 'kind': 'HIT', 'z': 3.0}])
        return {'tagged': min(n_tag, len(events)), 'ast': 0, 'clap_windows': 0}

    def fake_sections(x, fid, t0, win, hop, ast_budget=4, reuse=True, **kw):
        d = len(x) / SR
        return [{'start': 0.0, 'end': round(d, 2), 'label': 'test bed', 'lufs': -20.0, 'windows': 1}], \
            {'windows': 0, 'ast': 0}
    monkeypatch.setattr(tl, 'tag_events', fake_tags)
    monkeypatch.setattr(tl, 'sections', fake_sections)


def test_detects_clicks_at_their_attack():
    times = [0.5, 1.25, 2.0, 2.6, 3.33]
    ev = tl.detect_events(click_track(times, 4.0))
    hits = [e['t'] for e in ev if e['shape'] == 'hit']
    for t in times:
        assert min(abs(np.array(hits) - t)) <= 0.003, (t, hits)
    assert len(hits) <= len(times) + 1
    e = min(ev, key=lambda e: abs(e['t'] - 1.25))
    assert e['prominence_db'] > 20 and -12 < e['peak_dbfs'] < 6 and e['length_ms'] < 200


def test_quiet_click_is_less_prominent(wav):
    x = click_track([1.0], 2.0, amp=0.5) + click_track([1.5], 2.0, amp=0.05, bed=0.0, seed=1)
    ev = {round(e['t'], 1): e for e in tl.detect_events(x)}
    assert ev[1.0]['prominence_db'] > ev[1.5]['prominence_db'] + 10


def test_loudness_gaps_and_clipping(sine):
    x = np.concatenate([np.tile(sine, 3), np.zeros(SR), np.tile(sine, 2)])
    lo = tl.loudness(x)
    gaps = lo['silences']
    assert len(gaps) == 1 and gaps[0]['kind'] == 'digital silence'
    assert abs(gaps[0]['start'] - 3.0) < 0.15 and abs(gaps[0]['end'] - 4.0) < 0.15
    assert lo['clipping']['runs'] == 0
    clipped = np.clip(np.tile(sine, 6) * 5, -1, 1)
    assert tl.loudness(clipped)['clipping']['runs'] > 10


def test_beat_grid_on_a_click_track():
    beats = np.arange(0.25, 12.0, 0.5)
    x = click_track(beats, 12.5, amp=0.3)
    t = np.arange(int(0.12 * SR)) / SR
    thump = (np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.04) * 0.6).astype(np.float32)
    for b in beats[::4]:                      # accented low thump on every 4th beat = downbeat
        a = int(b * SR)
        x[a:a + len(thump)] += thump[: len(x) - a]
    g = tl.music_grid(x)
    assert abs(g['tempo'] - 120) < 2 and g['confidence'] == 'clear'
    d = [min(abs(beats - b)) for b in g['beats'] if 1.0 < b < 11.5]
    assert np.median(d) < 0.015
    assert min(abs(beats[::4] - g['downbeats'][1])) < 0.03


def test_no_beat_grid_in_noise():
    rng = np.random.RandomState(3)
    assert tl.music_grid((rng.randn(8 * SR) * 0.1).astype(np.float32))['confidence'] == 'none'


def test_read_cues_formats(tmp_path):
    f = tmp_path / 'cues.json'
    f.write_text(json.dumps({'fps': 60, 'events': [{'f': 120, 'kind': 'hit', 'id': 'a'}, {'f': 30, 'kind': 'tick'}]}))
    cues, fps = tl.read_cues(str(f))
    assert fps == 60 and [c['t'] for c in cues] == [0.5, 2.0] and cues[1]['id'] == 'a'
    assert [c['t'] for c in tl.read_cues('1.5, 3')[0]] == [1.5, 3.0]
    assert [c['t'] for c in tl.read_cues('[{"t": 2.25}]')[0]] == [2.25]
    assert tl.read_cues([4, 1])[0][0]['t'] == 1
    assert tl.read_cues(None) == ([], None)


def test_sync_offsets_status_and_chance():
    ev = [{'t': t, 'prominence_db': p, 'shape': 'hit'} for t, p in ((1.000, 12), (2.030, 12), (3.300, 2), (5.0, 12))]
    s = tl.sync([1.004, 2.0, 3.25, 4.0], ev, beats=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], frame_s=1 / 60)
    it = s['items']
    assert it[0]['status'] == 'OK' and it[0]['offset_ms'] == pytest.approx(-4.0)
    assert it[1]['status'] == 'near' and it[1]['offset_ms'] == pytest.approx(30.0)
    assert it[2]['status'] == 'near'
    assert it[3]['status'] == 'no sound' and it[3]['offset_ms'] is None
    st = s['stats']
    assert st['hit_within_1_frame'] == 0.25 and st['no_sound'] == [4.0] and 0 < st['chance_within_1_frame'] < 0.1
    assert st['on_beat'] == 0.75 and st['on_beat_chance'] == 0.2
    assert st['accent_min_db'] == 12.0 and [i['accent_offset_ms'] for i in it] == [-4.0, 30.0, None, None]
    assert st['accent_within_1_frame'] == 0.25 and st['accent_within_100ms'] == 0.5
    assert st['accent_chance_within_1_frame'] < st['chance_within_1_frame']


def test_video_cuts_and_frames(wav, no_models, tmp_path, monkeypatch):
    """a tiny film: black -> white picture cut at 1.5 s with a click on it; its own audio and cuts are used"""
    import subprocess
    monkeypatch.setattr(tl.config, 'EMB_CACHE', str(tmp_path / 'emb'))
    a = wav(click_track([0.6, 1.5, 2.4], 3.0), 'a.wav')
    mp4 = str(tmp_path / 'film.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=96x54:r=30:d=1.5', '-f', 'lavfi',
                    '-i', 'color=c=white:s=96x54:r=30:d=1.5', '-i', a, '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]',
                    '-map', '[v]', '-map', '2:a', '-c:v', 'mpeg4', '-c:a', 'aac', mp4], check=True)
    r = tl.timeline(mp4)
    assert r['video'] == mp4 and r['fps'] == 30 and [round(c, 1) for c in r['cuts']] == [1.5]
    it = r['sync']['cuts']['items'][0]
    assert it['status'] in ('OK', 'near') and abs(it['offset_ms']) < 50
    s = sheet_mod.sheet(mp4, str(tmp_path / 'film.png'), tl=r, width=1200)
    assert s['thumbnails'] >= 3 and s['cuts'] == 1


def test_timeline_end_to_end_without_models(wav, no_models, tmp_path, monkeypatch):
    monkeypatch.setattr(tl.config, 'EMB_CACHE', str(tmp_path / 'emb'))
    p = wav(click_track([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 6.0), 'clicks.wav')
    r = tl.timeline(p, cues=[1.5, 2.52, 5.0])
    assert r['summary'] and r['text'].startswith(r['summary'])
    r['text'].encode('ascii')
    assert len([e for e in r['events'] if e['shape'] == 'hit' and e['prominence_db'] > 10]) == 6
    st = r['sync']['cues']['stats']
    assert st['n'] == 3 and st['no_sound'] == [5.0]
    assert 'NO HIT' in r['text'] and 'cue -20 ms OK' in r['text']
    json.dumps(r)
    assert tl.timeline(p, cues=[1.5, 2.52, 5.0])['cached'] is True
    z = tl.timeline(p, start=2.0, end=4.0, cache=False)
    assert z['start'] == 2.0 and z['end'] == 4.0
    assert sorted(round(e['t'], 1) for e in z['events'] if e['prominence_db'] > 10) == [2.5, 3.5]


def test_sheet_png(wav, no_models, tmp_path, monkeypatch):
    monkeypatch.setattr(tl.config, 'EMB_CACHE', str(tmp_path / 'emb'))
    p = wav(click_track(np.arange(0.5, 8, 0.5), 8.0), 'beat.wav')
    out = tmp_path / 'sheet.png'
    r = sheet_mod.sheet(p, str(out), cues='1.0,2.0', width=1400)
    from PIL import Image
    im = Image.open(out)
    assert im.size == (r['width'], r['height']) and r['width'] == 1400
    assert r['labelled_events'] > 3 and 'Wrote' in r['summary']
    z = sheet_mod.sheet(p, str(tmp_path / 'zoom.png'), start=2.0, end=4.0)
    assert z['start'] == 2.0 and os.path.exists(z['png'])


def test_label_staggering_never_overlaps():
    items = [((1, -k), 1.0 + 0.01 * k, f'HIT label number {k}') for k in range(30)]
    placed = tl_rows = sheet_mod._place_labels(items, 0.0, 10.0, 1800, 10.0, 4)
    assert 0 < len(placed) <= 30
    rows = {}
    for row, t, a, text in tl_rows:
        b = a + (len(text) * 10.0 + 16) / 180.0
        for a2, b2 in rows.get(row, []):
            assert b <= a2 + 1e-9 or a >= b2 - 1e-9
        rows.setdefault(row, []).append((a, b))


def test_cli_registers():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command')
    tl.add_cli(sub)
    sheet_mod.add_cli(sub)
    a = ap.parse_args(['timeline', 'x.wav', '--cues', '1,2', '--from', '1', '--to', '3', '--detail', 'low'])
    assert (a.start, a.end, a.detail, a.cues) == (1.0, 3.0, 'low', '1,2')
    b = ap.parse_args(['sheet', 'x.mp4', '-o', 'o.png', '--width', '1600'])
    assert b.png == 'o.png' and b.width == 1600


@pytest.mark.slow
def test_timeline_with_models(wav, tmp_path, monkeypatch):
    monkeypatch.setattr(tl.config, 'EMB_CACHE', str(tmp_path / 'emb'))
    p = wav(click_track([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 6.0), 'clicks.wav')
    r = tl.timeline(p, detail='low')
    tagged = [e for e in r['events'] if e.get('label')]
    assert tagged and all(e['kind'] in tl.EVENT_VOCAB for e in tagged)
    assert r['sections'] and r['sections'][0]['label']
    assert r['model_calls']['clap_event_windows'] >= len(tagged)


def test_tag_background_covers_the_vocabulary():
    phrases = [t for ts in tl.EVENT_VOCAB.values() for t in ts]
    assert sorted(phrases) == sorted(tl.TAG_BG) and len(set(phrases)) == len(phrases)


def test_digital_silence_and_edge_gaps(sine):
    lo = tl.loudness(np.zeros((5 * SR, 2), np.float32))
    assert lo['digital_silence'] and lo['integrated_lufs'] is None and lo['true_peak_dbtp'] is None
    assert lo['silences'][0]['where'] == 'whole file' and min(lo['momentary']['lufs']) >= -70
    x = np.concatenate([np.zeros(SR), np.tile(sine, 3), np.zeros(SR), np.tile(sine, 2), np.zeros(SR)])
    g = tl.loudness(x)['silences']
    assert [q.get('where') for q in g] == ['lead-in', None, 'tail'] and g[0]['start'] == 0.0
    assert [q.get('where') for q in tl.loudness(x, at_start=False, at_end=False)['silences']] == [None] * 3


def test_level_peak_answers_a_soft_swell():
    rng = np.random.RandomState(5)
    t = np.arange(6 * SR) / SR
    x = (rng.randn(len(t)) * (0.01 + 0.2 * np.exp(-0.5 * ((t - 3.0) / 0.25) ** 2))).astype(np.float32)
    lv = tl.level_envs(x)
    pk = tl.level_peak(lv, 3.05)
    assert pk and abs(pk['t'] - 3.0) < 0.06 and pk['rise_db'] > 10
    assert tl.level_peak(lv, 1.0) is None
    it = tl.sync([3.05, 1.0], [], levels=lv)['items']
    assert it[0]['status'] == 'peak' and it[1]['status'] == 'no sound'


def test_cue_errors_and_aliases():
    assert tl.read_cues('{"fps": 50, "events": [{"frame": 100}]}')[0][0]['t'] == 2.0
    assert tl.read_cues([{'seconds': 1.25}])[0][0]['t'] == 1.25
    with pytest.raises(ValueError, match='no fps'):
        tl.read_cues('{"events": [{"f": 120, "kind": "hit"}]}')
    with pytest.raises(ValueError, match='without a time'):
        tl.read_cues([{'kind': 'hit'}])


def test_cover_art_is_not_a_video():
    art = {'has_video': True, 'video': {'fps': 90000.0}}
    assert not tl._is_video('song.mp3', art) and not tl._is_video('song.mp4', art)
    assert tl._is_video('film.mp4', {'has_video': True, 'video': {'fps': 30.0}})
    assert not tl._is_video('song.m4a', {'has_video': True, 'video': {'fps': 30.0}})


def test_burst_takes_one_tag_slot():
    ev = [{'t': 1.0 + 0.06 * k, 'band': 'high', 'prominence_db': 30.0 - k * 0.1, 'contrast_db': 20.0} for k in range(10)]
    ev += [{'t': t, 'band': 'full', 'prominence_db': 10.0, 'contrast_db': 8.0} for t in (3.0, 5.0, 7.0, 9.0)]
    pick, _ = tl._pick(ev, 6, (), 0.0, 10.0)
    assert sum(ev[i]['t'] < 2 for i in pick) <= 2 and {3.0, 5.0, 7.0, 9.0} <= {ev[i]['t'] for i in pick}


def test_zoom_keeps_full_file_verdicts(wav, no_models, tmp_path, monkeypatch):
    monkeypatch.setattr(tl.config, 'EMB_CACHE', str(tmp_path / 'emb'))
    p = wav(click_track(np.arange(0.5, 12, 0.5), 12.0), 'grid.wav')
    cues = [2.0, 4.02, 6.5, 7.97]
    full = {i['t']: (i['status'], i['offset_ms']) for i in tl.timeline(p, cues=cues)['sync']['cues']['items']}
    z = tl.timeline(p, cues=cues, start=4.0, end=8.0)
    assert z['context'] == [0.0, 12.0] and z['start'] == 4.0
    for i in z['sync']['cues']['items']:
        assert full[i['t']] == (i['status'], i['offset_ms'])
    assert all(4.0 <= e['t'] < 8.0 for e in z['events']) and all(4.0 <= b <= 8.0 for b in z['music']['beats'])


def test_picture_only_video(no_models, tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(tl.config, 'EMB_CACHE', str(tmp_path / 'emb'))
    mp4 = str(tmp_path / 'silent.mp4')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=96x54:r=30:d=1.5', '-f', 'lavfi',
                    '-i', 'color=c=white:s=96x54:r=30:d=1.5', '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]',
                    '-map', '[v]', '-c:v', 'mpeg4', mp4], check=True)
    r = tl.timeline(mp4)
    assert r['has_audio'] is False and [round(c, 1) for c in r['cuts']] == [1.5]
    assert r['sync']['cuts']['items'][0]['status'] == 'no sound' and 'NO AUDIO STREAM' in r['summary']


def test_ascii_keeps_non_ascii_names_recoverable():
    assert tl._ascii('t\xf6n.wav') == r't\xf6n.wav'
