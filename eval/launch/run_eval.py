"""Evaluate the launch profile and the critique scorecard.

    .venv/Scripts/python.exe eval/launch/run_eval.py --only ref_1,ref_2 --fresh   # measure films (cold), in chunks
    .venv/Scripts/python.exe eval/launch/run_eval.py --neg quiet,hot,mono          # known-bad mixes, in chunks
    .venv/Scripts/python.exe eval/launch/run_eval.py --build                      # + write earshot/data/launch_profile.json
    .venv/Scripts/python.exe eval/launch/run_eval.py                              # LOO + Clawd + RESULTS.md (cached films)

1. per film: earshot.profile.film_metrics for the 7 references and the two Clawd films; --fresh first deletes this
   module's audio-pass cache for the film (--fresh-ast also the AudioSet pass) so that pass is timed cold. Times and
   metrics go to eval/launch/films.json (numbers only).
2. --build: the shipped profile (earshot/data/launch_profile.json) from the 7 references, ids and styles from refs.json.
3. leave-one-out: each reference scored against a profile of the other six (do the ranges generalise?), plus the
   in-sample check against all seven.
4. the Clawd film with its real soundtrack vs the same film with the retired synthesized soundtrack: full critique
   against the 7-film profile (with earshot.realness when its model is installed).
5. sensitivity: known-bad mixes (the Clawd real soundtrack degraded one way at a time, its SFX / ambience stems alone,
   the retired synthesized bed, a sine, pink noise; scratch audio in out/launch/neg/, never committed): is the metric
   the defect should move flagged, and what is the verdict? Results go to eval/launch/negatives.json.
6. the visual-event detector (earshot.video frame difference) vs ffmpeg scene cuts (video.cuts(method='scene')), and
   a check that the core detector reproduces this module's first-build picture pass exactly.
Writes eval/launch/results.json and eval/launch/RESULTS.md.
"""
import hashlib
import argparse
import json
import os
import sys
import time

import subprocess  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

from earshot import audio, config, video  # noqa: E402
from earshot import profile as P  # noqa: E402
from earshot.critique import SPEC, critique, real_thresholds, scorecard  # noqa: E402

REFS = json.load(open(os.path.join(HERE, 'refs.json'), encoding='utf-8'))
for _r in REFS['refs']:
    _r['path'] = EP.ref(_r['id'])          # local files, see eval/README.md
for _r in REFS['tests']:
    _r['path'] = f"{EP.FILM}/{_r['file']}"
FILMS = os.path.join(HERE, 'films.json')
NEGS = os.path.join(HERE, 'negatives.json')
RESULTS = os.path.join(HERE, 'results.json')
NEG_DIR = os.path.join(ROOT, 'out', 'launch', 'neg')
ST = f'{EP.FILM}/out/audio/soundtrack.wav'
STEMS = f'{EP.FILM}/out/audio/stems_pro/'
FILM = f'{EP.FILM}/out/Clawd_1080p60.mp4'
# id: (source or lavfi graph, ffmpeg -af filter, models (AudioSet + realness), picture, metrics the defect should flag,
# what it is). Expectations were written down before the first run.
NEGATIVES = {
    'base': (ST, None, False, None, (), 'control: the real Clawd soundtrack untouched'),
    'quiet': (ST, 'volume=-12dB', False, None, ('lufs_i',), 'unmastered: 12 dB quieter'),
    'hot': (ST, 'volume=12dB', False, None, ('lufs_i', 'true_peak'), '12 dB hotter, clipped by the 16-bit file'),
    'mono': (ST, 'pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1', False, None, ('side_to_mid_db',), 'folded to mono'),
    'muffled': (ST, 'lowpass=f=3000,lowpass=f=3000', False, None, ('band_presence', 'band_air'),
                'low-passed at 3 kHz (12 dB/oct)'),
    'late': (ST, 'adelay=2500:all=1,atrim=0:60', False, None, ('first_sound_s',), '2.5 s of silence before the start'),
    'deadair': (ST, 'apad=pad_dur=4', False, None, ('end_silence_s',), '4 s of silence after the end'),
    'fadeout': (ST, 'afade=t=out:st=48:d=12', False, None, ('fade_s', 'final_lu', 'end_lu'), '12 s fade-out'),
    'hole': (ST, "volume=enable='between(t,25,27.5)':volume=0", False, None, ('max_gap_s',), '2.5 s muted mid-film'),
    'squashed': (ST, 'acompressor=threshold=-36dB:ratio=20:attack=1:release=60:makeup=12,alimiter=limit=0.7', False,
                 None, ('lra', 'crest_lu'), 'over-compressed: 20:1 from -36 dB + limiter'),
    'sfx_only': (STEMS + 'sfx.wav', None, True, FILM, ('music_frac',), 'the SFX stem alone, with the picture'),
    'amb_only': (STEMS + 'amb.wav', None, True, None, ('music_frac', 'onset_rate', 'hit_rate'), 'the ambience stem alone'),
    'synth_bed': (f'{EP.FILM}/audio/_retired_synth/bed.wav', None, True, None, ('p_real',),
                  'the retired code-synthesized music bed'),
    'sine': ('lavfi:sine=f=440:d=30:sample_rate=48000', 'volume=0.3', True, None, ('p_real', 'music_frac', 'onset_rate'),
             '30 s 440 Hz sine'),
    'pink': ('lavfi:anoisesrc=color=pink:d=30:a=0.3:r=48000', None, True, None, ('p_real', 'music_frac', 'onset_rate'),
             '30 s pink noise'),
}


def _all():
    return REFS['refs'] + REFS['tests']


def _clear(path, ast=False):
    for k, f in P.cache_paths(path).items():
        if os.path.exists(f) and (ast or k != 'ast'):
            os.remove(f)


def measure(only, fresh, fresh_ast=False):
    films = json.load(open(FILMS)) if os.path.exists(FILMS) else {}
    for r in _all():
        if only and r['id'] not in only:
            continue
        if fresh:
            _clear(r['path'], fresh_ast)
        t0 = time.time()
        f = P.film_metrics(r['path'])
        cold = round(time.time() - t0, 1)
        t0 = time.time()
        P.film_metrics(r['path'])
        warm = round(time.time() - t0, 1)
        old = films.get(r['id'], {})
        films[r['id']] = {'id': r['id'], 'style': r['style'], 'duration': f['duration'], 'fps': f['fps'],
                          'lossy': f['lossy'],
                          'metrics': f['metrics'], 'chance': f['chance'], 'arc': f['arc'],
                          'video': bool(f['video']),
                          'n_visual': len(f['events']['visual']), 'n_hits': len(f['events']['hits']),
                          'seconds_cold': cold if fresh else old.get('seconds_cold'), 'seconds_warm': warm,
                          'timing_cold': f['timing'] if fresh else old.get('timing_cold'),
                          'audioset_cold_s': (f['timing'].get('audioset') if fresh_ast else
                                              old.get('audioset_cold_s', (old.get('timing_cold') or {}).get('audioset')))}
        print(f"{r['id']}: {f['duration']:.1f} s film, {'cold ' + str(cold) + ' s, ' if fresh else ''}warm {warm} s | "
              f"{f['summary']}", flush=True)
        json.dump(films, open(FILMS, 'w'), indent=1)
    return films


def loo(films):
    refs = [films[r['id']] for r in REFS['refs']]
    full = P.aggregate(refs)
    out = []
    for i, f in enumerate(refs):
        others = refs[:i] + refs[i + 1:]
        sc = scorecard(f['metrics'], P.aggregate(others), f['chance'], f['lossy'])
        ins = scorecard(f['metrics'], full, f['chance'], f['lossy'])
        rows = [r for r in sc['metrics'] if r['weight'] != 'info' and r['status'] != 'n.a.']
        out.append({'id': f['id'], 'verdict': sc['verdict'], 'score': sc['score'], 'in_range': sc['in_range'],
                    'scored': sc['scored'], 'critical': sc['critical_issues'], 'major': sc['major_issues'],
                    'out': [f"{r['metric']} {r['status']}" for r in rows if r['status'] != 'ok'],
                    'refs_basis_in_range': sum(1 for r in rows if r['basis'] == 'references' and r['status'] == 'ok'),
                    'refs_basis_scored': sum(1 for r in rows if r['basis'] == 'references'),
                    'in_sample_out': [f"{r['metric']} {r['status']}" for r in ins['metrics']
                                      if r['weight'] != 'info' and r['status'] in ('low', 'high')]})
    per_metric = {}
    for k in SPEC:
        if SPEC[k][0] == 0 or k == 'p_real':
            continue
        st = []
        for i, f in enumerate(refs):
            sc = scorecard(f['metrics'], P.aggregate(refs[:i] + refs[i + 1:]), lossy=f['lossy'])
            st.append(next(r['status'] for r in sc['metrics'] if r['metric'] == k))
        per_metric[k] = {'ok': st.count('ok'), 'low': st.count('low'), 'high': st.count('high'), 'na': st.count('n.a.')}
    return out, per_metric


def clawd():
    res = {}
    for r in REFS['tests']:
        t0 = time.time()
        c = critique(r['path'])
        res[r['id']] = {'style': r['style'], 'verdict': c['verdict'], 'score': c['score'], 'in_range': c['in_range'],
                        'scored': c['scored'], 'critical': c['critical_issues'], 'major': c['major_issues'],
                        'actions': c['actions'], 'tips': c['tips'],
                        'status': {m['metric']: [m['value'], m['status']] for m in c['metrics']},
                        'realness': (c.get('realness') or {}).get('summary') or (c.get('realness') or {}).get('error'),
                        'seconds_critique': round(time.time() - t0, 1)}
        print(f"{r['id']}: {c['summary']}", flush=True)
    return res


def detector_check():
    out = []
    for r in _all():
        fps, d = video.frame_diff(r['path'])
        ev = video.diff_events(d, fps) / fps
        hc = video.cuts(r['path'], method='scene')
        row = {'id': r['id'], 'scene_cuts': len(hc), 'visual_events': int(len(ev))}
        if len(hc) >= 5:
            row['recall_of_scene_cuts'] = round(P.near_frac(hc, ev, 1.0 / fps + 1e-3), 2)
        st = os.stat(r['path'])  # the first build's cache key used the file id of that time (path, size, mtime only)
        fid = hashlib.sha1(f"{os.path.abspath(r['path'])}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
        k = hashlib.sha1('|'.join(map(str, (fid, 96, 54, 1))).encode()).hexdigest()[:20]
        old = os.path.join(config.EMB_CACHE, 'launch', f'frames_{k}.npz')
        if os.path.exists(old):
            with np.load(old) as z:
                row['same_as_first_build'] = bool(np.array_equal(z['d'], d))
        out.append(row)
    return out


def _make_neg(nid):
    src, af = NEGATIVES[nid][:2]
    if not af and not src.startswith('lavfi:'):
        return src
    os.makedirs(NEG_DIR, exist_ok=True)
    out = os.path.join(NEG_DIR, f'{nid}.wav')
    if not os.path.exists(out):
        inp = ['-f', 'lavfi', '-i', src[6:]] if src.startswith('lavfi:') else ['-i', src]
        subprocess.run(['ffmpeg', '-v', 'error', '-y', *inp, *(['-af', af] if af else []), '-ac', '2', '-ar', '48000',
                        '-c:a', 'pcm_s16le', out], check=True)
    return out


def negatives(ids):
    res = json.load(open(NEGS)) if os.path.exists(NEGS) else {}
    for nid in ids:
        src, af, models, vid, expect, what = NEGATIVES[nid]
        path = _make_neg(nid)
        t0 = time.time()
        c = critique(path, video=vid, use_ast=models, use_realness=models)
        st = {m['metric']: m['status'] for m in c['metrics']}
        flagged = [f"{m['metric']} {m['status']} ({m['weight']})" for m in c['metrics']
                   if m['status'] in ('low', 'high')]
        res[nid] = {'what': what, 'models': models, 'picture': bool(vid), 'expect': list(expect),
                    'caught': [k for k in expect if st.get(k) in ('low', 'high')], 'verdict': c['verdict'],
                    'partial': c.get('partial'), 'score': c['score'], 'critical': c['critical_issues'],
                    'major': c['major_issues'], 'flagged': flagged,
                    'first_action': c['actions'][0]['action'] if c['actions'] else None,
                    'p_real': next((m['value'] for m in c['metrics'] if m['metric'] == 'p_real'), None),
                    'seconds': round(time.time() - t0, 1)}
        print(f"{nid}: {c['summary'][:300]}", flush=True)
        json.dump(res, open(NEGS, 'w'), indent=1)
    return res


def hit_timing_check():
    """tuning data: detected hits / onsets of the Clawd real soundtrack vs the sample placements in its mix report"""
    mr = f'{EP.FILM}/out/qa/mix_report.json'
    film = next(r['path'] for r in REFS['tests'] if r['id'] == 'clawd_real')
    if not os.path.exists(mr):
        return None
    pl = sorted(p['t'] for p in json.load(open(mr))['placements'])
    f = P.film_metrics(film)
    on = P._audio_pass(film)['onsets']
    hits = f['events']['hits']
    off = [1000 * min(abs(p - t) for p in pl) for t in hits]
    row = {'placements': len(pl), 'hits': len(hits), 'onsets': len(on),
           'hit_offset_median_ms': round(float(sorted(off)[len(off) // 2]), 1) if off else None,
           'hits_within_2ms': sum(1 for o in off if o <= 2.0)}
    for tol in (0.02, 0.05, 0.1):
        row[f'hit_precision_{int(tol * 1000)}ms'] = round(P.near_frac(hits, pl, tol), 2)
        row[f'hit_chance_{int(tol * 1000)}ms'] = round(P.chance_frac(hits, pl, tol, f['duration']), 2)
        row[f'onset_precision_{int(tol * 1000)}ms'] = round(P.near_frac(on, pl, tol), 2)
        row[f'placement_recall_{int(tol * 1000)}ms'] = round(P.near_frac(pl, on, tol), 2)
    return row


def _fmt(k, v):
    return 'n.a.' if v is None else P.fmt_value(k, v) if k in P.METRICS else str(v)


def report(films, loo_rows, per_metric, cl, det, ht, neg=None):
    L = ['# Launch profile + critique: evaluation', '',
         'Generated by `eval/launch/run_eval.py` (numbers only; the films are local reference files of the author, '
         'never copied). Profile = `earshot/data/launch_profile.json`, built from the 7 references below.', '',
         '## Commands', '',
         '```',
         'EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py --only ref_1,ref_2,ref_3 --fresh --measure-only',
         'EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py --only ref_4,ref_5,ref_6,ref_7,clawd_real,'
         'clawd_synth --fresh --measure-only',
         'EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py --build   # profile, LOO, Clawd, this file',
         'EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py --neg base,quiet,hot,mono,muffled,late,deadair,'
         'fadeout,hole,squashed',
         'EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py --neg sfx_only,amb_only,synth_bed',
         'EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py --neg sine,pink',
         'EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/launch/run_eval.py           # rewrite this file from the caches',
         '```', '',
         '## What was tuned on what', '',
         '- **Detector thresholds** (salient hit: >= 6 dB jump of the 100 ms level within 80 ms, landing within 8 LU '
         'of integrated, one hit per run, timed at the attack; visual event: frame-difference peak >= 4x the median '
         'of the surrounding second or >= 0.15, now earshot.video.diff_events; stop-down: >= 10 dB dip for 0.15-1.5 s) '
         'were chosen by looking at plots of all 9 films (7 refs + 2 Clawd), at the Clawd mix-report placements (hits) '
         'and at ffmpeg scene cuts (visual events). The Clawd real soundtrack is therefore partly tuning data for the '
         'hit detector.',
         '- **Music threshold** (AudioSet Music >= 0.3 per 5 s window) was set after looking at the per-window '
         'probabilities of all 9 films (music under voice or SFX scored 0.4-0.7, one sound-design stretch 0.1-0.15).',
         '- **Changed after the first leave-one-out run** (first build; not a blind test of these choices): true peak '
         'went from critical to major and is judged against 0 dBTP for decoded lossy files (the -1 dBTP master rule '
         'failed all 7 AAC-decoded references); a critical rule missed by <= 2 tolerances counts as major; the metric '
         'max_gap_s was added; the music_frac tolerance went from 0.1 to 0.05 after an SFX-only stem (music in 42 % of '
         'windows) came out as only a near miss.',
         '- **Changed after the independent review, before the known-bad set below was first run:** p_real uses the '
         "realness model's own thresholds (ok at its 'real or pro-produced' threshold, critical below 0.5, minor in "
         'between; no near-miss downgrade); a density range that would reach 0 keeps a floor at half the sparsest '
         'reference (salient hits >= 0.05 /s); a mono mix (identical channels) counts as -40 dB stereo width instead '
         'of n.a.; files under 10 s get no loudness-shape rows; brightness_hz replaces the power-weighted centroid '
         '(info only). The known-bad expectations were written into run_eval.py before its first run; nothing was '
         'changed after seeing its results.',
         '- **Not adjusted after seeing results:** the other tolerances, the weights of the other metrics and the '
         'verdict thresholds (90 / 75 / 50 %).',
         "- **Ranges** are data: min-max of the reference films widened by each metric's tolerance. Loudness and true "
         'peak use web-delivery practice (-16..-9 LUFS; <= -1 dBTP for a master, <= 0 dBTP for decoded AAC / MP3) '
         'instead, so references can fail them.', '',
         '## Per-film measurements', '',
         '| film | dur s | LUFS | dBTP | LRA | music | speech | onsets/s | hits/s | vis events/s | events w/ sound '
         '(chance) | audio pass s (cold) | picture pass s | AudioSet pass s (cold, first build) | warm s |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in _all():
        f = films.get(r['id'])
        if not f:
            continue
        m, c, tc = f['metrics'], f['chance'], f.get('timing_cold') or {}
        L.append(f"| {r['id']} | {f['duration']:.1f} | {m['lufs_i']} | {m['true_peak']} | {m['lra']} | "
                 f"{_fmt('music_frac', m['music_frac'])} | {_fmt('speech_frac', m['speech_frac'])} | {m['onset_rate']} | "
                 f"{m['hit_rate']} | {m['cut_rate']} | {_fmt('cuts_with_sound', m['cuts_with_sound'])} "
                 f"({_fmt('cuts_with_sound', c.get('cuts_with_sound'))}) | {tc.get('audio')} | {tc.get('picture')} | "
                 f"{f.get('audioset_cold_s')} | {f.get('seconds_warm')} |")
    L += ['', 'Audio pass = decode + features.measure(light) + librosa onsets / beats + loudness curves, timed cold in '
          "this round (this module's cache deleted; the core decode cache was cold too after its v2 bump). Picture "
          'pass = earshot.video.frame_diff (core cache, shared with timeline / sheet): 0.0 means another module had '
          'already decoded that film; ref_6 (4K) and clawd_synth were cold. AudioSet = 5 s windows through AST, '
          'measured cold in the first build (same code and cache key, so not re-run): ~4-9 s per 5 s window on this '
          'laptop under load from 3-4 other model jobs; `earshot critique --fast` skips it and realness. Warm = '
          'everything cached. EARSHOT_THREADS=2.', '',
          '## Leave-one-out: each reference vs a profile of the other six', '',
          '| film | verdict | weighted score | scored metrics in range | reference-range metrics in range | out of range |',
          '|---|---|---|---|---|---|']
    n_ok = n_all = r_ok = r_all = 0
    for x in loo_rows:
        n_ok += x['in_range']; n_all += x['scored']; r_ok += x['refs_basis_in_range']; r_all += x['refs_basis_scored']
        L.append(f"| {x['id']} | {x['verdict']} | {100 * x['score']:.0f}% | {x['in_range']}/{x['scored']} | "
                 f"{x['refs_basis_in_range']}/{x['refs_basis_scored']} | {', '.join(x['out']) or '-'} |")
    L += ['', f'Total: {n_ok}/{n_all} scored metric checks in range ({100 * n_ok / max(1, n_all):.0f} %); '
          f'reference-range metrics only: {r_ok}/{r_all} ({100 * r_ok / max(1, r_all):.0f} %). For exchangeable data '
          'a held-out film falls outside the min-max of six others with probability 2/7 (29 %) per continuous metric '
          'before tolerance, so ~71 % is the no-overfit baseline; the tolerance adds a little. p_real is not part of '
          'this check (the realness model was trained on these films).', '',
          'In-sample (each reference vs the full 7-film profile): every reference-range metric is in range by '
          'construction; the only failures are web-practice rules: ' +
          '; '.join(f"{x['id']}: {', '.join(x['in_sample_out'])}" for x in loo_rows if x['in_sample_out']) + '.', '',
          '### Per metric (7 leave-one-out checks each)', '', '| metric | weight | ok | low | high | n.a. |',
          '|---|---|---|---|---|---|']
    for k, st in per_metric.items():
        L.append(f"| {k} | {['info', 'minor', 'major', 'critical'][SPEC[k][0]]} | {st['ok']} | {st['low']} | "
                 f"{st['high']} | {st['na']} |")
    if neg:
        bad = {k: v for k, v in neg.items() if v['expect']}
        caught = sum(1 for v in bad.values() if v['caught'])
        L += ['', '## Sensitivity: known-bad mixes', '',
              'Each row is one defect made on purpose (ffmpeg, scratch files in out/launch/neg/), critiqued against the '
              "7-film profile. 'Should flag' was written down before the first run; 'caught' = at least one of those "
              'metrics came out low / high. Rows without models skip AudioSet and realness (partial verdicts); the '
              'control is the untouched real soundtrack.', '',
              '| mix | what | should flag | caught | verdict | weighted score | all flagged rows | p_real | s (critique, cold) |',
              '|---|---|---|---|---|---|---|---|---|']
        for k, v in neg.items():
            verdict = v['verdict'] + (' (partial)' if v.get('partial') else '')
            L.append(f"| {k} | {v['what']} | {', '.join(v['expect']) or '-'} | {', '.join(v['caught']) or ('no' if v['expect'] else '-')} | "
                     f"{verdict} | {100 * (v['score'] or 0):.0f}% | {'; '.join(v['flagged']) or '-'} | "
                     f"{'-' if v['p_real'] is None else v['p_real']} | {v['seconds']} |")
        miss = [k for k, v in bad.items() if not v['caught']]
        L += ['', f'Caught: {caught} of {len(bad)} known-bad mixes have the expected metric flagged' +
              (f" (missed: {', '.join(miss)})" if miss else '') + '. Verdicts: ' +
              ', '.join(f"{n} {sum(1 for v in bad.values() if v['verdict'] == n)}" for n in
                        ('launch-grade', 'close', 'needs work', 'not launch-grade')) +
              f" (control: {neg.get('base', {}).get('verdict', 'n.a.')}). A single structural defect on a mix that "
              'is already close usually stays close: the verdict is coarse, the flagged rows and FIX FIRST actions '
              'are the signal. Over-compression is not a defect by these references: two of them (LRA 1.7 / 3.3 LU, '
              'loudest 3 s only 1.1-1.2 LU above integrated) are as dense as the squashed mix.']
    L += ['', '## The Clawd film: real soundtrack vs the retired synthesized soundtrack', '']
    r, s_ = cl.get('clawd_real'), cl.get('clawd_synth')
    if r and s_:
        pr, ps = r['status']['p_real'][0], s_['status']['p_real'][0]
        L += [f"Reading: the scorecard rates the real soundtrack {r['verdict']} ({100 * r['score']:.0f}%) and the "
              f"synthesized one {s_['verdict']} ({100 * s_['score']:.0f}%). The structural metrics separate them only "
              'a little: both are loud enough, continuous and on the picture; the synth version fades away at the end, '
              'is sparser and has a long fade, while the real one starts late (a 1 s near-silent cold open) and has '
              'one loudness peak that sticks out. The difference a listener hears most - synthesized timbre - is not '
              'structural; it is what p_real measures' +
              (f": real {pr:.2f} vs synth {ps:.2f} (earshot.realness, verdict thresholds lo {real_thresholds()['lo']} "
               f"/ hi {real_thresholds()['hi']} at this run); below 0.5 the p_real row is critical, which is what "
               "moves the synth version's verdict." if pr is not None and ps is not None else
               '; the realness model was not installed when this ran.') +
              ' Caveat: the realness model (another module) was trained partly on these 7 launch films, on the Clawd '
              "project's real asset library and on the retired synthesized stems (eval/realness/), so p_real on the "
              'Clawd films is not an independent test.', '']
    for k, c in cl.items():
        L += [f"### {k}: {c['style']}", '',
              f"Verdict **{c['verdict']}**, weighted score {100 * c['score']:.0f}%, {c['in_range']}/{c['scored']} in range, "
              f"{c['critical']} critical / {c['major']} major issues; critique call {c['seconds_critique']} s with the "
              f"measurement passes cached (realness runs inside it).", '']
        for a in c['actions']:
            L.append(f"- [{a['weight']}] {a['action']}")
        for t in c['tips']:
            L.append(f'- tip: {t}')
        L.append(f"- realness: {c['realness']}")
        L.append('')
    if ht:
        L += ['## Hit / onset timing vs ground truth (tuning data: the Clawd real soundtrack)', '',
              f"The mix report lists {ht['placements']} sample placements (every key tick, pop, whoosh, hit, music "
              f"motif note ...). earshot found {ht['hits']} salient hits and {ht['onsets']} onsets in the mixed film "
              f"audio; {ht['hits_within_2ms']} of the {ht['hits']} hits sit within 2 ms of a placement (median distance "
              f"{ht['hit_offset_median_ms']} ms).", '',
              '| tolerance | hits on a placement (chance) | onsets on a placement | placements with an onset |',
              '|---|---|---|---|']
        for ms in (20, 50, 100):
            L.append(f"| +-{ms} ms | {100 * ht[f'hit_precision_{ms}ms']:.0f}% ({100 * ht[f'hit_chance_{ms}ms']:.0f}%) | "
                     f"{100 * ht[f'onset_precision_{ms}ms']:.0f}% | {100 * ht[f'placement_recall_{ms}ms']:.0f}% |")
        L += ['', 'Chance = the same hits shifted off the placements by fixed offsets. Hit and onset times are moved to '
              'the steepest rise of the 2 ms level (attack), which is how a mixer aligns a sample.', '']
    L += ['## Visual-event detector (earshot.video frame difference) vs ffmpeg scene cuts', '',
          "Scene cuts = `video.cuts(path, method='scene')` (ffmpeg scene score 0.3). 'Same as first build' = the core "
          "video.frame_diff signal is bit-identical to this module's first-build picture pass, so moving the detector "
          'into core changed no number.', '',
          '| film | scene cuts | visual events | share of scene cuts also found (+-1 frame) | same as first build |',
          '|---|---|---|---|---|']
    for d in det:
        rec = d.get('recall_of_scene_cuts', 'n.a. (< 5 scene cuts)')
        L.append(f"| {d['id']} | {d['scene_cuts']} | {d['visual_events']} | {rec} | "
                 f"{d.get('same_as_first_build', 'n.a.')} |")
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='comma-separated ids to (re)measure')
    ap.add_argument('--fresh', action='store_true', help='delete the audio-pass cache first (cold timings)')
    ap.add_argument('--fresh-ast', action='store_true', help='with --fresh: also delete the AudioSet cache')
    ap.add_argument('--neg', help='comma-separated known-bad mixes to critique (or "all"): ' + ','.join(NEGATIVES))
    ap.add_argument('--build', action='store_true', help='write earshot/data/launch_profile.json')
    ap.add_argument('--measure-only', action='store_true')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    if a.neg:
        negatives(list(NEGATIVES) if a.neg == 'all' else a.neg.split(','))
        return
    films = (measure(only, a.fresh, a.fresh_ast) if (only or a.fresh or not os.path.exists(FILMS)) else
             json.load(open(FILMS)))
    if a.measure_only:
        return
    if a.build:
        P.profile([r['path'] for r in REFS['refs']], styles=[r['style'] for r in REFS['refs']], out=P.PROFILE_PATH)
    loo_rows, per_metric = loo(films)
    cl = clawd()
    det = detector_check()
    ht = hit_timing_check()
    neg = json.load(open(NEGS)) if os.path.exists(NEGS) else {}
    json.dump({'loo': loo_rows, 'per_metric': per_metric, 'clawd': cl, 'detector': det, 'hit_timing': ht},
              open(RESULTS, 'w'), indent=1)
    open(os.path.join(HERE, 'RESULTS.md'), 'w', encoding='utf-8').write(
        report(films, loo_rows, per_metric, cl, det, ht, neg))
    print('wrote', RESULTS, 'and RESULTS.md')


if __name__ == '__main__':
    main()
