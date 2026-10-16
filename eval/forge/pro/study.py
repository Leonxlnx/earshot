"""Measure, export and summarise the curated pro motion-SFX events (eval/forge/pro/events.py).

    python eval/forge/pro/study.py                 # measure all, export trims, manifest, registry, stats, sheets
    python eval/forge/pro/study.py --no-export     # measurements + stats only
    python eval/forge/pro/study.py --sheets-only   # only the per-category spectrogram crop sheets

Outputs (local only: the films are third-party):
  out/refs_pro/<category>/{forge,bench}/<ref>_<ms>.wav   trims (trim + fade only, original gain, stereo 48 kHz 24-bit)
  out/refs_pro/manifest.json    every event: film id, times, category, cut, bed level, measurements, earshot evidence
  out/refs_pro/registry.json    earshot.forge.refsets format: {'version', 'categories': {cat: {forge, bench, note}}}
  out/refs_pro/stats.json       per category: count, medians / quartiles of the measured numbers
  out/refs_pro/_work/cat/<category>.png   spectrogram crops of the category's events (look at these)
The split is by film: a film is 'forge' or 'bench' for every category (SPLIT below), so a forge tuned on the forge
side is benchmarked on films it never heard.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from videos import OUT, VIDEOS, WORK, wav  # noqa: E402
from measure import measure  # noqa: E402
from events import EVENTS, SPLIT, NOTES  # noqa: E402

SR = 48000


def load(ref, _c={}):
    if ref not in _c:
        x, sr = sf.read(wav(ref), dtype='float64', always_2d=True)
        assert sr == SR
        _c[ref] = x
    return _c[ref]


def tl(ref, _c={}):
    if ref not in _c:
        p = f'{WORK}/tl/{ref}.json'
        _c[ref] = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None
    return _c[ref]


def evidence(ref, t0, t1):
    r = tl(ref)
    if not r:
        return {}
    evs = []
    for e in r['events']:
        if t0 - 0.02 <= e['t'] <= t1:
            evs.append({'t': e['t'], 'kind': e.get('kind'), 'label': e.get('label'),
                        'tag': (e.get('tags') or [{}])[0].get('tag'), 'prominence_db': e.get('prominence_db'),
                        'bed_db': e.get('bed_db'), 'contrast_db': e.get('contrast_db'), 'band': e.get('band'),
                        'exposed': e.get('exposed'), 'length_ms': e.get('length_ms')})
    cuts = [c for c in (r.get('cuts') or []) if t0 - 0.5 <= c <= t1 + 0.5]
    return {'timeline_events': evs[:12], 'cuts_near': cuts}


def norm(e):
    d = {'ref': e[0], 't0': e[1], 't1': e[2], 'cat': e[3], 'cut': e[4] if len(e) > 4 else None,
         'note': e[5] if len(e) > 5 else '', 'bed': e[6] if len(e) > 6 else None}
    d['export'] = not d['note'].startswith('noexp')
    return d


def export(d, x, split):
    pre = 0.005
    a, b = max(0, int((d['t0'] - pre) * SR)), min(len(x), int(d['t1'] * SR))
    y = x[a:b].copy()
    fi = int(0.002 * SR)
    fo = int(min(0.03 + 0.1 * (d['t1'] - d['t0']), 0.25) * SR)
    fo = min(fo, len(y) // 3)
    y[:fi] *= np.linspace(0, 1, fi)[:, None]
    y[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
    folder = f"{OUT}/{d['cat']}/{split}"
    os.makedirs(folder, exist_ok=True)
    f = f"{folder}/{d['ref']}_{int(round(d['t0'] * 1000)):06d}.wav"
    sf.write(f, y.astype(np.float32), SR, subtype='PCM_24')
    return f, round(a / SR, 4), round(b / SR, 4), round(fo / SR, 3)


NUMS = ['dur_s', 'over_bed_db', 'snr_db', 'peak_over_bed_db', 'bed_lufs', 'ev_lufs', 'attack_ms', 'rise_ms',
        'peak_frac', 't20_ms', 't40_ms', 'decay_db_s', 'tilt_db_oct', 'centroid_hz', 'peak_vs_cut_ms',
        'attack_vs_cut_ms', 'end_vs_cut_ms']


def stats(rows):
    out = {}
    cats = sorted({r['cat'] for r in rows})
    for c in cats:
        rr = [r for r in rows if r['cat'] == c]
        s = {'n': len(rr), 'films': sorted({r['ref'] for r in rr})}
        for k in NUMS:
            v = [r['m'].get(k) for r in rr if r['m'].get(k) is not None]
            if v:
                s[k] = {'med': round(float(np.median(v)), 2), 'q1': round(float(np.percentile(v, 25)), 2),
                        'q3': round(float(np.percentile(v, 75)), 2), 'min': round(float(min(v)), 2),
                        'max': round(float(max(v)), 2), 'n': len(v)}
        for k in ('sweep_oct', 'to_peak_oct', 'from_peak_oct', 'start', 'peak', 'end'):
            v = [r['m'].get('centroid', {}).get(k) for r in rr if r['m'].get('centroid', {}).get(k) is not None]
            if v:
                s['centroid_' + k] = {'med': round(float(np.median(v)), 2), 'q1': round(float(np.percentile(v, 25)), 2),
                                      'q3': round(float(np.percentile(v, 75)), 2), 'n': len(v)}
        for k in ('side_mid_db', 'side_mid_raw_db', 'corr'):
            v = [r['m']['stereo'].get(k) for r in rr]
            s['stereo_' + k] = {'med': round(float(np.median(v)), 2), 'q1': round(float(np.percentile(v, 25)), 2),
                                'q3': round(float(np.percentile(v, 75)), 2)}
        v = [r['m']['stereo'].get('pan_db', {}).get('travel') for r in rr if r['m']['stereo'].get('pan_db')]
        if v:
            s['pan_travel_db'] = {'med': round(float(np.median(v)), 2), 'max': round(float(max(v)), 2)}
        for k in ('start_hz', 'end_hz', 'min_hz', 'max_hz', 'peak_hz', 'sweep_oct'):
            v = [r['m']['ridge'][k] for r in rr if r['m'].get('ridge')]
            if v:
                s['ridge_' + k] = {'med': round(float(np.median(v)), 2), 'min': round(float(min(v)), 2),
                                   'max': round(float(max(v)), 2), 'n': len(v)}
        v = [r['m'].get('am_rate_hz') for r in rr if r['m'].get('am_rate_hz')]
        if v:
            s['am_rate_hz'] = {'med': round(float(np.median(v)), 1), 'min': min(v), 'max': max(v)}
        f0 = [r['m']['f0_track'] for r in rr if r['m'].get('f0_track')]
        if f0:
            s['f0_start'] = round(float(np.median([q['start'] for q in f0])), 1)
            s['f0_end'] = round(float(np.median([q['end'] for q in f0])), 1)
            s['f0_oct'] = round(float(np.median([q['oct'] for q in f0])), 2)
        # bed-subtracted 1/3-octave balance: median over events of each band (dB re total), unmasked bands only
        ev = np.array([r['m']['third_oct']['ev_db'] for r in rr])
        mk = np.array([r['m']['third_oct']['masked'] for r in rr])
        med = []
        for j in range(ev.shape[1]):
            vv = ev[~mk[:, j], j]
            med.append(round(float(np.median(vv)), 1) if len(vv) else None)
        s['third_oct_med_db'] = med
        s['third_oct_centres'] = rr[0]['m']['third_oct']['centres']
        sh = np.array([r['m']['shares'] for r in rr])
        s['shares_med_db'] = [round(float(v), 1) for v in np.median(sh, 0)]
        shapes = [r['m'].get('decay_shape') for r in rr if r['m'].get('decay_shape')]
        s['decay_shapes'] = {k: shapes.count(k) for k in sorted(set(shapes))}
        rshapes = [r['m'].get('rise_shape') for r in rr if r['m'].get('rise_shape')]
        s['rise_shapes'] = {k: rshapes.count(k) for k in sorted(set(rshapes))}
        out[c] = s
    return out


def cat_sheets(rows, max_per=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import viewer
    os.makedirs(f'{WORK}/cat', exist_ok=True)
    outs = []
    for c in sorted({r['cat'] for r in rows}):
        rr = [r for r in rows if r['cat'] == c][:max_per]
        cols = 3
        nrow = int(np.ceil(len(rr) / cols))
        fig = plt.figure(figsize=(5.6 * cols, 3.3 * nrow), dpi=90)
        gs = fig.add_gridspec(2 * nrow, cols, height_ratios=[3, 1.2] * nrow, hspace=0.3, wspace=0.18)
        for i, r in enumerate(rr):
            x = load(r['ref']).astype(np.float32)
            cuts = (tl(r['ref']) or {}).get('cuts') or []
            evs = (tl(r['ref']) or {}).get('events') or []
            pad = max(0.15, 0.25 * (r['t1'] - r['t0']))
            q, k = divmod(i, cols)
            m = r['m']
            lab = (f"{c} {r['t0']:.2f}-{r['t1']:.2f} +{m['over_bed_db']}dB atk {m['attack_ms']:.0f}ms "
                   f"sw {m.get('centroid', {}).get('sweep_oct', '-')}oct")
            ax, ax2 = viewer.lane(fig, gs[2 * q, k], gs[2 * q + 1, k], x, r['t0'] - pad, r['t1'] + pad, cuts, evs,
                                  title=f"{r['ref']} {lab}")
            ax.axvline(r['t0'], color='w', lw=0.6, ls=':')
            ax.axvline(r['t1'], color='w', lw=0.6, ls=':')
            ax2.tick_params(labelsize=6)
        out = f'{WORK}/cat/{c}.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-export', action='store_true')
    ap.add_argument('--sheets-only', action='store_true')
    ap.add_argument('--no-sheets', action='store_true')
    ap.add_argument('--cats', nargs='*')
    a = ap.parse_args()
    rows = []
    for e in EVENTS:
        d = norm(e)
        if a.cats and d['cat'] not in a.cats:
            continue
        x = load(d['ref'])
        d['m'] = measure(x, d['t0'], d['t1'], bed=d['bed'], cut=d['cut'])
        rows.append(d)
    if a.sheets_only:
        print('\n'.join(cat_sheets(rows)))
        return
    man = []
    reg = {'version': 1, 'source': 'pro motion-design launch films (third-party, local only); see '
                                   'docs/motion-sfx-study.md', 'categories': {}}
    if not a.no_export:
        import shutil
        for c in sorted({r['cat'] for r in rows}):
            if os.path.isdir(f'{OUT}/{c}'):
                shutil.rmtree(f'{OUT}/{c}')
    for d in rows:
        split = SPLIT[d['ref']]
        ent = {'film': d['ref'], 'source_file': VIDEOS[d['ref']], 'category': d['cat'], 't0': d['t0'], 't1': d['t1'],
               'cut': d['cut'], 'split': split, 'note': d['note'], 'bed_lufs': d['m']['bed_lufs'],
               'over_bed_db': d['m']['over_bed_db'], 'snr_db': d['m']['snr_db'], 'measured': d['m'],
               'earshot': evidence(d['ref'], d['t0'], d['t1'])}
        if not a.no_export and d['export']:
            f, ta, tb, fo = export(d, load(d['ref']), split)
            ent.update({'file': f, 'trim_s': [ta, tb], 'fade_in_s': 0.002, 'fade_out_s': fo, 'gain_db': 0.0})
            c = reg['categories'].setdefault(d['cat'], {'forge': [], 'bench': [], 'note': NOTES.get(d['cat'], '')})
            c[split].append(os.path.relpath(f, OUT).replace(chr(92), '/'))
        man.append(ent)
    st = stats(rows)
    with open(f'{OUT}/stats.json', 'w', encoding='utf-8') as fh:
        json.dump(st, fh, indent=1)
    if not a.no_export:
        with open(f'{OUT}/manifest.json', 'w', encoding='utf-8') as fh:
            json.dump(man, fh, indent=1)
        with open(f'{OUT}/registry.json', 'w', encoding='utf-8') as fh:
            json.dump(reg, fh, indent=1)
        # sanity: the split never leaks a film
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
        from earshot.forge import refsets
        srcs = {f"{OUT}/{p}": p.split('/')[-1].split('_')[0] + '_' + p.split('/')[-1].split('_')[1]
                for cc in reg['categories'].values() for s in ('forge', 'bench') for p in cc[s]}
        reg_abs = refsets.load(f'{OUT}/registry.json')
        srcs = {}
        for cc in reg_abs['categories'].values():
            for s in ('forge', 'bench'):
                for p in cc[s]:
                    srcs[p] = os.path.basename(p).rsplit('_', 1)[0]
        bad = refsets.check_split(reg_abs, srcs)
        print('split check:', bad or 'ok')
    for c, s in st.items():
        def g(k, f='med'):
            return s.get(k, {}).get(f, '-') if isinstance(s.get(k), dict) else '-'
        print(f"{c:12s} n={s['n']:2d} films={','.join(f[4:] for f in s['films'])} dur {g('dur_s')} "
              f"over_bed {g('over_bed_db')} atk {g('attack_ms')} t20 {g('t20_ms')} tilt {g('tilt_db_oct')} "
              f"cen {g('centroid_hz')} sweep {g('centroid_sweep_oct')} S/M {g('stereo_side_mid_db')} "
              f"pk-cut {g('peak_vs_cut_ms')}")
    if not a.no_sheets:
        print('\n'.join(cat_sheets(rows)))


if __name__ == '__main__':
    main()
