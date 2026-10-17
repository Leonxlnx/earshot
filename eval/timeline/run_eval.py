"""Evaluate earshot.timeline / earshot.sheet against the Clawd film's ground truth (never used for tuning) and on the
7 reference launch films of the launch profile.

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/run_eval.py --parts detect,tags,sync,film,refs,timing60,timing93

Parts (each fits in one ~10 minute job; results merge into eval/timeline/results.json):
  detect  event detection on soundtrack.wav vs every sfx placement of out/qa/mix_report.json (30 ms), split by how
          loud the placement is against the music + ambience stems at that moment (SNR); also on the sfx stem alone
  tags    tag plausibility at the true times of keyboard / whoosh / grass-step placements, in the mix and in the stem;
          false KEY / UI tags on the ambience and music stems (no keyboard, no UI sounds in them); each for the
          shipped scoring (hub phrases damped, TAG_CENTER 0.5) and the first build's (TAG_CENTER 0)
  sync    timeline(soundtrack, cues=cues_all.json): measured cue->sound offsets vs the true placement offsets; the
          'without a hit' list vs the mixer's unsounded cues (all 153 cues); beat grid vs the 120 BPM score grid;
          section boundaries vs the music / ambience edits
  film    timeline + sheet on Clawd_1080p60.mp4 (picture cuts vs act changes)
  refs    timeline + sheet on 3 reference launch films
  timing60 / timing93  wall-clock of the CLI (fresh: no cached results or model outputs) for the 60 s soundtrack /
          the 93 s ref_1 film
Tuning data (out/timeline/mixes, made by mixes.py from recordings the Clawd mix does not use) is separate from this.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from earshot import audio, timeline as tl  # noqa: E402
from tune import match  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

CLAWD = f'{EP.FILM}/out'
SOUNDTRACK = f'{CLAWD}/audio/soundtrack.wav'
STEMS = {k: f'{CLAWD}/audio/stems_pro/{k}.wav' for k in ('sfx', 'amb', 'music')}
REPORT = f'{CLAWD}/qa/mix_report.json'
CUES = f'{CLAWD}/cues_all.json'
FILM = f'{CLAWD}/Clawd_1080p60.mp4'
REFS = {rid: EP.ref(rid) for rid in ('ref_1', 'ref_2', 'ref_4')}   # 3 of the 7 reference launch films
OUT = os.path.join(ROOT, 'out', 'timeline', 'eval')
RESULTS = os.path.join(HERE, 'results.json')
SR = audio.SR


def placements():
    return json.load(open(REPORT))['placements']


def onsets_truth(P, aligns=('attack', 'start')):
    """unique onset times (placement layers within 30 ms merged) for placements with the given alignment"""
    ts = sorted({round(p['t'], 4) for p in P if p.get('align', 'attack') in aligns})
    out = []
    for t in ts:
        if out and t - out[-1][-1] < 0.03:
            out[-1].append(t)
        else:
            out.append([t])
    return [g[0] for g in out]


def snr_db(t, sfx, bed):
    """placement level vs the bed at that moment: max 10 ms RMS of the sfx stem in [t, t+60 ms] over the RMS of
    music + ambience in [t-30, t+60 ms] (dB)"""
    a, b = int(t * SR), int((t + 0.06) * SR)
    s = sfx[a:b]
    if len(s) < 480:
        return None
    r = np.sqrt(np.convolve(s.astype(np.float64) ** 2, np.ones(480) / 480, 'valid').max())
    q = bed[max(0, a - int(0.03 * SR)): b]
    return float(20 * np.log10(r + 1e-9) - 20 * np.log10(np.sqrt(np.mean(q.astype(np.float64) ** 2)) + 1e-9))


def part_detect():
    P = placements()
    x = audio.load(SOUNDTRACK)
    sfx = audio.load(STEMS['sfx'], mono=True)
    bed = audio.load(STEMS['music'], mono=True) + audio.load(STEMS['amb'], mono=True)
    gt = onsets_truth(P)
    snr = np.array([snr_db(t, sfx, bed) for t in gt], float)
    t = time.time()
    ev = tl.detect_events(x)
    secs = time.time() - t
    det = [e['t'] for e in ev if e['shape'] == 'hit']
    m = match(gt, det, 0.03)
    hit = np.zeros(len(gt), bool)
    hit[[i for i, _, _ in m]] = True
    dts = np.array([dt for _, _, dt in m]) * 1000
    used = {j for _, j, _ in m}
    rest = [d for j, d in enumerate(det) if j not in used]
    bed_on = [e['t'] for e in tl.detect_events(bed, zthr=2.0, min_contrast_db=1.0)]
    mb = match(bed_on, rest, 0.03)
    expl = len(mb)
    rest2 = [d for j, d in enumerate(rest) if j not in {j for _, j, _ in mb}]
    sfx_loose = [e['t'] for e in tl.detect_events(sfx, zthr=2.0, min_contrast_db=1.0)]
    inner = len(match(sfx_loose, rest2, 0.03))
    bins = [('snr>=+6', snr >= 6), ('0..+6', (snr >= 0) & (snr < 6)), ('-6..0', (snr >= -6) & (snr < 0)),
            ('<-6', snr < -6)]
    sfx_ev = [e['t'] for e in tl.detect_events(audio.load(STEMS['sfx'])) if e['shape'] == 'hit']
    ms = match(gt, sfx_ev, 0.03)
    hs = np.zeros(len(gt), bool)
    hs[[i for i, _, _ in ms]] = True
    pk = onsets_truth(P, ('peak',))
    sw = [e['t'] for e in ev]
    mp = match(pk, sw, 0.12)
    res = {
        'n_truth_onsets': len(gt), 'n_detections': len(det), 'detect_seconds': round(secs, 2),
        'recall_all': round(float(hit.mean()), 3),
        'recall_loud_snr_ge0': round(float(hit[snr >= 0].mean()), 3), 'n_loud': int((snr >= 0).sum()),
        'recall_quiet_snr_lt0': round(float(hit[snr < 0].mean()), 3), 'n_quiet': int((snr < 0).sum()),
        'recall_by_snr': {k: [round(float(hit[v].mean()), 3) if v.any() else None, int(v.sum())] for k, v in bins},
        'recall_sfx_stem_alone': round(float(hs.mean()), 3),
        'precision_vs_placements': round(len(m) / max(1, len(det)), 3),
        'precision_incl_music_amb_onsets': round((len(m) + expl) / max(1, len(det)), 3),
        'extra_onsets_inside_sfx_samples': inner,
        'precision_incl_all_real_onsets': round((len(m) + expl + inner) / max(1, len(det)), 3),
        'unexplained_detections': len(det) - len(m) - expl - inner,
        'timing_error_ms': {'median': round(float(np.median(dts)), 1), 'median_abs': round(float(np.median(np.abs(dts))), 1),
                            'p90_abs': round(float(np.percentile(np.abs(dts), 90)), 1)},
        'peak_aligned_whoosh_swell': {'n': len(pk), 'found_within_120ms': len(mp)},
        'missed_loud_examples': [round(gt[i], 3) for i in np.where(~hit & (snr >= 6))[0][:10]],
    }
    return res


GROUPS = {'keyboard': (lambda p: str(p.get('sample', '')).startswith('kb.'), {'KEY'}, {'KEY', 'UI'}),
          'whoosh': (lambda p: p.get('kind') == 'whoosh', {'WHOOSH', 'RISER'}, {'WHOOSH', 'RISER'}),
          'grass_steps': (lambda p: str(p.get('sample', '')).startswith('gd.step'), {'STEP'}, {'STEP', 'FOLEY'})}


def _tag(x, sid, evs, n, center):
    keep = tl.TAG_CENTER
    tl.TAG_CENTER = center
    try:
        tl.tag_events(x, sid, 0.0, evs, n, 0)
    finally:
        tl.TAG_CENTER = keep


def part_tags():
    P = placements()
    out = {}
    srcs = {'mix': SOUNDTRACK, 'sfx_stem': STEMS['sfx']}
    music_layer = [p['t'] for p in P if str(p.get('sample', '')).startswith('mg.')]
    groups = dict(GROUPS, keyboard_without_celesta_layer=(
        lambda p: GROUPS['keyboard'][0](p) and not any(abs(p['t'] - t) < 0.03 for t in music_layer), {'KEY'}, {'KEY', 'UI'}))
    for g, (sel, strict, lenient) in groups.items():
        ts = sorted({round(p['t'], 3) for p in P if sel(p)})
        ts = [t for k, t in enumerate(ts) if k == 0 or t - ts[k - 1] >= 0.03]
        shape = 'swell' if g == 'whoosh' else 'hit'
        out[g] = {'n': len(ts)}
        for name, path in srcs.items():
            x = audio.load(path)
            for rule, center in (('shipped', tl.TAG_CENTER), ('first_build', 0.0)):
                evs = [{'t': t, 'shape': shape, 'band': 'full', 'rise_ms': 400, 'length_ms': 300, 'prominence_db': 0.0,
                        'contrast_db': 0.0} for t in ts]
                _tag(x, tl.span_id(path), evs, len(evs), center)
                k1 = [e['kind'] for e in evs]
                k3 = [{t['kind'] for t in e['tags']} for e in evs]
                out[g][f'{name}_{rule}'] = {
                    'top1_strict': round(float(np.mean([k in strict for k in k1])), 3),
                    'top1_lenient': round(float(np.mean([k in lenient for k in k1])), 3),
                    'top3_strict': round(float(np.mean([bool(k & strict) for k in k3])), 3),
                    'labels': dict(Counter(e['label'] for e in evs).most_common(5))}
    for stem in ('amb', 'music'):
        x = audio.load(STEMS[stem])
        for rule, center in (('shipped', tl.TAG_CENTER), ('first_build', 0.0)):
            ev = [e for e in tl.detect_events(x) if e['shape'] == 'hit']
            _tag(x, tl.span_id(STEMS[stem]), ev, 20, center)
            tg = [e for e in ev if e.get('kind')]
            out[f'{stem}_stem_{rule}'] = {
                'tagged': len(tg), 'key_or_ui': round(float(np.mean([e['kind'] in ('KEY', 'UI') for e in tg])), 3),
                'kinds': dict(Counter(e['kind'] for e in tg).most_common()),
                'labels': dict(Counter(e['label'] for e in tg).most_common(4))}
    return out


def part_sync():
    P = placements()
    t = time.time()
    r = tl.timeline(SOUNDTRACK, cues=CUES, detail='normal')
    wall = time.time() - t
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'soundtrack_timeline.txt'), 'w', encoding='utf-8').write(r['text'])
    by_id = {}
    for p in P:
        if p.get('event'):
            by_id.setdefault(p['event'], []).append(p)
    errs, peak_only, flagged, ok = [], [], [], []
    unsounded = {(round(u['t'], 2), u['id']) for u in json.load(open(REPORT))['unsounded_events']}
    nohit = {'no sound', 'off'}
    for it in r['sync']['cues']['items']:
        ps = [p for p in by_id.get(it.get('id'), []) if abs(p['t'] - it['t']) < 0.5]
        att = [p['t'] for p in ps if p.get('align', 'attack') in ('attack', 'start')]
        if (round(it['t'], 2), it.get('id')) in unsounded:
            flagged.append({'t': it['t'], 'id': it.get('id'), 'truth': 'unsounded', 'status': it['status']})
            continue
        if not att:
            peak_only.append({'t': it['t'], 'kind': it.get('kind'), 'status': it['status']})
            continue
        true_off = (min(att, key=lambda v: abs(v - it['t'])) - it['t']) * 1000
        if it['offset_ms'] is None:
            ok.append(None)
            continue
        errs.append(it['offset_ms'] - true_off)
    e = np.abs(np.array(errs))
    grid = r['music']
    beats = np.array([b for b in grid['beats'] if b < 52.0])
    on_grid = np.abs(beats / 0.5 - np.round(beats / 0.5)) * 0.5 if len(beats) else np.zeros(0)
    db = np.array([b for b in grid['downbeats'] if b < 50.5])
    true_bounds = [10.5, 12.0, 28.0, 40.0, 51.0, 52.2]
    det_bounds = [s['start'] for s in r['sections'][1:]]
    mb = match(true_bounds, det_bounds, 1.5)
    res = {
        'cues': r['sync']['cues']['stats'], 'n_cues_scored': len(errs) + len(ok),
        'peak_or_end_aligned_cues': peak_only,
        'without_a_hit_vs_truth_all_cues': {
            'reported_without_a_hit': len(r['sync']['cues']['stats']['no_sound']),
            'of_them_truly_unsounded': sum(f['status'] in nohit for f in flagged),
            'truly_unsounded': len(flagged),
            'sounded_reported_without_a_hit': len(r['sync']['cues']['stats']['no_sound'])
            - sum(f['status'] in nohit for f in flagged),
            'reported_level_peak_only': sum(it['status'] == 'peak' for it in r['sync']['cues']['items']),
            'level_peak_only_truly_unsounded': sum(f['status'] == 'peak' for f in flagged)},
        'offset_error_ms': {'within_5ms': round(float(np.mean(e <= 5)), 3), 'within_1_frame_16.7ms': round(float(np.mean(e <= 16.7)), 3),
                            'within_40ms': round(float(np.mean(e <= 40)), 3), 'median_abs': round(float(np.median(e)), 1),
                            'p90_abs': round(float(np.percentile(e, 90)), 1)},
        'sounded_cues_reported_without_sound': len(ok),
        'truly_unsounded_cues': flagged,
        'beat_grid': {'tempo': grid.get('tempo'), 'confidence': grid.get('confidence'),
                      'pulse_clarity': grid.get('pulse_clarity'), 'n_beats_0_52s': int(len(beats)),
                      'beats_within_40ms_of_0.5s_grid': round(float(np.mean(on_grid <= 0.04)), 3) if len(beats) else None,
                      'downbeats_on_even_seconds_60ms': round(float(np.mean(np.abs(db / 2 - np.round(db / 2)) * 2 <= 0.06)), 3)
                      if len(db) else None},
        'sections': {'true_boundaries': true_bounds, 'detected': det_bounds, 'matched_within_1.5s': len(mb),
                     'labels': [f"{s['start']}-{s['end']}: {s['label']}" for s in r['sections']]},
        'summary': r['summary'], 'timing_s': r['timing_s'], 'model_calls': r['model_calls'], 'wall_s': round(wall, 1),
    }
    return res


def part_film():
    from earshot.sheet import sheet
    t = time.time()
    r = tl.timeline(FILM, cues=CUES, detail='normal')
    wall = time.time() - t
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'film_timeline.txt'), 'w', encoding='utf-8').write(r['text'])
    acts = [8.0, 14.0, 28.0, 40.0, 52.0]
    mc = match(acts, r['cuts'], 0.1)
    s = sheet(FILM, os.path.join(OUT, 'clawd_film_sheet.png'), tl=r)
    z = sheet(FILM, os.path.join(OUT, 'clawd_zoom_14-24.png'), cues=CUES, start=14.0, end=24.0)
    return {'cuts': r['cuts'], 'act_changes': acts, 'act_changes_found_100ms': len(mc),
            'cut_sync': (r['sync'].get('cuts') or {}).get('stats'), 'cue_sync': r['sync']['cues']['stats'],
            'summary': r['summary'], 'timing_s': r['timing_s'], 'wall_s': round(wall, 1),
            'sheet': s['summary'], 'zoom_sheet': z['summary']}


def part_refs():
    from earshot.sheet import sheet
    out = {}
    for name, path in REFS.items():
        t = time.time()
        r = tl.timeline(path, detail='normal')
        wall = time.time() - t
        open(os.path.join(OUT, f'{name}_timeline.txt'), 'w', encoding='utf-8').write(r['text'])
        s = sheet(path, os.path.join(OUT, f'{name}_sheet.png'), tl=r)
        tg = [e for e in r['events'] if e.get('label')]
        top = Counter(e['label'] for e in tg).most_common(1)
        out[name] = {'duration': r['duration'], 'summary': r['summary'], 'n_events': len(r['events']),
                     'tagged': len(tg), 'key_or_ui': round(float(np.mean([e['kind'] in ('KEY', 'UI') for e in tg])), 3)
                     if tg else None, 'most_common_tag': [top[0][0], top[0][1]] if top else None,
                     'kinds': dict(Counter(e['kind'] for e in tg).most_common()),
                     'cuts': len(r['cuts']), 'sections': [s_['label'] for s_ in r['sections']],
                     'cut_sync': (r['sync'].get('cuts') or {}).get('stats'), 'timing_s': r['timing_s'],
                     'wall_s': round(wall, 1), 'sheet': s['summary']}
    return out


def part_timing60():
    return _timing('soundtrack_60s', SOUNDTRACK)


def part_timing93():
    return _timing('ref_1_film_93s', REFS['ref_1'])


def part_timing93low():
    return _timing('ref_1_film_93s_detail_low', REFS['ref_1'], 'low')


def _timing(name, path, detail='normal'):
    """wall-clock of a fresh CLI run (new process: imports, model load, no cached results or model outputs)"""
    env = dict(os.environ, EARSHOT_THREADS=os.environ.get('EARSHOT_THREADS', '2'))
    n_py = _n_python()
    t = time.time()
    p = subprocess.run([sys.executable, '-m', 'earshot', 'timeline', path, '--fresh', '--json', '--detail', detail],
                       cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    wall = time.time() - t
    try:
        j = json.loads(p.stdout)
    except ValueError:
        return {name: {'error': p.stderr[-800:]}}
    return {name: {'wall_s_cli_incl_imports_and_model_load': round(wall, 1), 'internal_timing_s': j['timing_s'],
                   'model_calls': j['model_calls'], 'threads': env['EARSHOT_THREADS'], 'duration_s': j['duration'],
                   'other_python_processes_at_start': n_py}}


def part_probe():
    """model load and per-window cost right now (the laptop is shared, so this varies with load)"""
    from earshot.models import ast, clap
    x = audio.load(SOUNDTRACK, mono=True)
    segs = [x[int(i * SR): int((i + 0.5) * SR)] for i in range(10, 18)]
    out = {'python_processes_running': _n_python()}
    t = time.time()
    c = clap.get()
    out['clap_load_s'] = round(time.time() - t, 1)
    t = time.time()
    c.embed_audio(segs, pad='zero')
    out['clap_s_per_window'] = round((time.time() - t) / len(segs), 2)
    t = time.time()
    a = ast.get()
    out['ast_load_s'] = round(time.time() - t, 1)
    t = time.time()
    a.probs(segs[:2])
    out['ast_s_per_window'] = round((time.time() - t) / 2, 2)
    return out


def _n_python():
    p = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/NH'], capture_output=True, text=True)
    return sum(1 for ln in p.stdout.splitlines() if ln.lower().startswith('python'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='detect,tags,sync')
    a = ap.parse_args()
    res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    os.makedirs(OUT, exist_ok=True)
    for part in a.parts.split(','):
        t = time.time()
        res[part] = globals()[f'part_{part}']()
        res[part]['_run_s'] = round(time.time() - t, 1)
        print(part, json.dumps(res[part], indent=1)[:4000], flush=True)
        with open(RESULTS, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
