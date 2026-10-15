#!/usr/bin/env python
"""Fit a music track to the film's bar structure by editing it at bar lines (no generation).

  python -I <skill>/scripts/arrange.py TRACK [--bpm 123.97] [--phase auto|0..3]
            [--film-bpm 120] [--beats-per-bar 4] [--bars 30 | --arc arc.json | --arc "-14,-12,...,0,0,-7"]
            [--drop-film-bar 21] [--drop auto|none|SRC_BAR] [--start-bar SRC_BAR] [--end-bar SRC_BAR] [--phrase 2]
            [--film-offset 0] [--file audio/assets/music/track.mp3] [--out out/audio-review/arrange.json] [--json]

The film's arc is a target loudness per film bar in dB relative to the track's loud bars (0 = the loudest sections,
-14 = a soft intro). Give it as a comma list, or a JSON file (templates/arc.json: {"arc": [...], "film_bpm": 120,
"beats_per_bar": 4, "drop_film_bar": 21}), or let the script build a generic one from --bars and --drop-film-bar (soft
open, groove, breakdown, build, big section from the drop, resolve). Write the arc from the picture map in SOUND_BRIEF.md.

1. Beat grid: librosa beats, a robust linear fit beats ~ t0 + k*T over the whole track (precise BPM), then the phase is
   moved onto the measured attacks (a 23 ms-window spectral flux at 2.9 ms resolution, folded at the beat period): the
   beat tracker's onset envelope lags real attacks by 20-50 ms, and a splice cut on a lagging grid chops the downbeat's
   own attack.
2. Downbeat: which beat phase starts bars (low-band onsets + harmonic change), with a confidence (z-score gap) and a
   local check in 8-bar windows (sections where another phase clearly wins are listed). Measured on the two
   hand-labelled tracks of the Clawd project (both seen while building this): Born Of The Sky correct (confidence 0.93);
   Bring Me The Sky half a bar off (confidence 0.15, flagged LOW: its bass plays on beats 1 and 3 and its accents move
   by half a bar later in the track). Under LOW confidence the downbeat is a guess between the two printed grids:
   decide from the music (the first note, section entrances, chord changes in the bar table) and pass --phase N.
3. Bar features: chroma, MFCC, loudness, low-band energy, onset density; novelty = change from the previous bar.
4. Arrangement by dynamic programming over the film's bars (film bar k spans bar_len*(k-1)..bar_len*k s):
   cost = energy-curve mismatch (the film's arc) + jump costs (a jump must land on a bar that sounds like the one that
   would naturally follow: chroma/MFCC/loudness distance) + phrase penalties (jumps prefer phrase starts) +
   hard anchors (--drop-film-bar = --drop source bar, the last film bar = --end-bar, film bar 1 = --start-bar). By
   default (--drop auto) the drop film bar is anchored on the strongest drop candidate (a bar within 3 dB of the loud
   sections that stays loud, scored by its lift over the 4 bars before + novelty); --drop none lets the arc decide.
   On the two Clawd tracks the top candidate is a section start the music supervisor marked by hand (Born Of The Sky
   bar 50 = the crash into big section 1; Bring Me The Sky the re-entry at 127.8 s); developed on those same tracks.
Prints the summary, the bar table, the chosen map and a ready-to-paste "music" block for the design (source seconds,
film seconds); --json prints the full result as JSON instead, --out writes it to a file.
It is a first draft: check every splice (qa.py edges, earshot timeline) and the bar table, then fine-tune "at" by a few
ms so the measured downbeat transients land on the picture's downbeats.
"""
import argparse, json, os, subprocess, sys
import numpy as np

SR = 22050


def default_arc(n, drop):
    """a generic short-film arc over n film bars with the big section starting at film bar `drop` (1-based):
    soft open -> groove -> breakdown -> build -> big section -> resolve"""
    arc = []
    big_until = drop + max(2, int(round((n - drop) * 0.6)))
    for k in range(1, n + 1):
        if k < drop:
            x = (k - 1) / max(1, drop - 2)            # 0..1 up to the bar before the drop
            v = -14 + 10 * min(1.0, x / 0.45)         # opening rises to the groove (-4)
            if 0.55 <= x < 0.8:
                v = -9.5                              # breakdown before the build
            elif x >= 0.8:
                v = -9 + 4 * (x - 0.8) / 0.2          # build
        elif k < big_until:
            v = 0.0                                   # the big section
        else:
            v = -13 + 2 * min(3, n - k)               # resolve to the end
        arc.append(round(v, 1))
    return arc


def decode(path):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                       capture_output=True)
    if p.returncode != 0 or not p.stdout:
        sys.exit(f'cannot decode {path}: {p.stderr.decode(errors="replace")[:300]}')
    return np.frombuffer(p.stdout, np.float32).copy()


def fit_grid(beats, T0):
    """robust fit of beat times to t0 + k*T, starting from period T0"""
    k = np.r_[0, np.cumsum(np.maximum(1, np.round(np.diff(beats) / T0)))]
    A = np.vstack([k, np.ones_like(k)]).T
    inl = np.ones(len(beats), bool)
    T, t0 = T0, float(beats[0])
    for _ in range(5):
        T, t0 = np.linalg.lstsq(A[inl], beats[inl], rcond=None)[0]
        res = beats - (t0 + k * T)
        inl = np.abs(res) < 0.035
        if inl.sum() < 8:
            inl[:] = True
    res = beats - (t0 + k * T)
    return float(T), float(t0), float(np.median(np.abs(res))), float(np.mean(np.abs(res) < 0.035))


def flux(y, n_fft=512, hop=64, block=4096):
    """log-compressed spectral flux with a short (23 ms) window at 2.9 ms resolution, computed in blocks of frames
    (bounded memory); returns (flux, frame centre times)"""
    win = np.hanning(n_fft).astype(np.float32)
    nfr = max(0, (len(y) - n_fft) // hop + 1)
    out, prev = np.zeros(nfr, np.float32), None
    for f0 in range(0, nfr, block):
        f = np.arange(f0, min(nfr, f0 + block))
        L = np.log1p(100 * np.abs(np.fft.rfft(y[f[:, None] * hop + np.arange(n_fft)] * win, axis=1)))
        d = np.diff(L, axis=0, prepend=L[:1] if prev is None else prev[None])
        out[f] = np.maximum(0, d).sum(axis=1)
        prev = L[-1]
    return out, (np.arange(nfr) * hop + n_fft / 2) / SR


def transient_phase(y, T, t0):
    """shift (s) that puts the grid on the measured attacks: the short-window flux folded at the beat period over the
    whole track, searched within +-min(100 ms, 0.3 beat); returns (shift, score ratio best / median)"""
    fl, tt = flux(y)
    hop_s = tt[1] - tt[0]
    search = min(0.1, 0.3 * T)
    offs = np.arange(-search, search + 1e-9, 0.001)
    k = np.arange(int(np.ceil((tt[0] - t0 + search) / T)), int((tt[-1] - t0 - search) / T))
    score = np.array([fl[np.round((t0 + o + k * T - tt[0]) / hop_s).astype(int)].mean() for o in offs])
    b = int(np.argmax(score))
    return float(offs[b]), float(score[b] / (np.median(score) + 1e-9))


def local_phases(score, grid, bpb, phase, win=8, hop=4, min_gap=0.5):
    """stretches where another downbeat phase clearly wins (by >= min_gap, the global LOW threshold) in windows of
    `win` bars: [{'from', 'to', 'phase', 'gap'}]"""
    out, W = [], win * bpb
    for s in range(0, len(grid) - W + 1, hop * bpb):
        seg = score[s:s + W]
        sc = [float(np.mean(seg[(p - s) % bpb::bpb])) for p in range(bpb)]
        o = np.argsort(sc)[::-1]
        gap = sc[o[0]] - sc[o[1]]
        if o[0] == phase or gap < min_gap:
            continue
        t0, t1 = round(float(grid[s]), 2), round(float(grid[min(len(grid) - 1, s + W)]), 2)
        if out and out[-1]['phase'] == int(o[0]) and t0 <= out[-1]['to']:
            out[-1].update({'to': t1, 'gap': max(out[-1]['gap'], round(gap, 2))})
        else:
            out.append({'from': t0, 'to': t1, 'phase': int(o[0]), 'gap': round(gap, 2)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('track')
    ap.add_argument('--bpm', type=float, help='the track\'s tempo if you know it (the phase is still fitted)')
    ap.add_argument('--phase', default='auto', help='downbeat phase 0..beats-per-bar-1, or auto')
    ap.add_argument('--film-bpm', type=float, default=120.0, help='tempo the picture is cut to (after any stretch)')
    ap.add_argument('--beats-per-bar', type=int, default=4)
    ap.add_argument('--bars', type=int, default=30, help='film bars (ignored when --arc is given)')
    ap.add_argument('--arc', help='target dB per film bar: comma list or JSON file')
    ap.add_argument('--drop-film-bar', type=int, help='1-based film bar of the drop / big section')
    ap.add_argument('--drop', default='auto', help='source bar for the drop film bar: auto (the top drop candidate), '
                                                   'none, or a 0-based bar index')
    ap.add_argument('--end-bar', type=int, help='source bar index for the film\'s last bar')
    ap.add_argument('--start-bar', type=int, help='source bar index for film bar 1')
    ap.add_argument('--phrase', type=int, default=2, help='jumps prefer film bars that start a phrase of this length')
    ap.add_argument('--film-offset', type=float, default=0.0, help='film time of film bar 1 (s)')
    ap.add_argument('--jump', type=float, default=3.0, help='base cost per jump')
    ap.add_argument('--no-snap', action='store_true', help='keep the beat tracker\'s phase (skip the transient fit)')
    ap.add_argument('--file', help='the track path as the design should reference it (default: TRACK as given)')
    ap.add_argument('--out', help='write the full result as JSON here')
    ap.add_argument('--json', action='store_true', help='print the full result as JSON instead of the table')
    a = ap.parse_args(argv)
    if a.drop not in ('auto', 'none') and not a.drop.lstrip('-').isdigit():
        sys.exit(f'--drop must be auto, none or a source bar index, not {a.drop!r}')
    if not os.path.isfile(a.track):
        sys.exit(f'track not found: {a.track}')
    cfg = {}
    if a.arc and a.arc.strip().endswith('.json'):
        cfg = json.load(open(a.arc, encoding='utf-8'))
        arc_list = [float(v) for v in cfg['arc']]
    elif a.arc:
        arc_list = [float(v) for v in a.arc.split(',') if v.strip()]
    else:
        arc_list = None
    film_bpm = float(cfg.get('film_bpm', a.film_bpm))
    bpb = int(cfg.get('beats_per_bar', a.beats_per_bar))
    FILM_BARS = len(arc_list) if arc_list else a.bars
    drop_k = int(a.drop_film_bar or cfg.get('drop_film_bar') or round(FILM_BARS * 0.7))
    ARC = arc_list or default_arc(FILM_BARS, drop_k)
    bar_len = bpb * 60.0 / film_bpm
    import librosa
    y = decode(a.track)
    dur = len(y) / SR
    hop = 256
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=hop, start_bpm=a.bpm or 120,
                                           tightness=400, units='time')
    beats = np.asarray(beats, float)
    if len(beats) < 8:
        sys.exit(f'only {len(beats)} beats tracked in {a.track}: not a steady-tempo track, or too short')
    T0 = 60.0 / a.bpm if a.bpm else float(np.median(np.diff(beats)))
    T, t0, med_res, inlier = fit_grid(beats, T0)
    if a.bpm:  # trust a given tempo but keep the fitted phase
        T = 60.0 / a.bpm
    shift, snap_ratio = (0.0, None) if a.no_snap else transient_phase(y, T, t0)
    t0 += shift
    bpm = 60.0 / T
    first = t0 - np.floor(t0 / T) * T
    grid = np.arange(first, dur, T)
    # downbeat phase: low-band onset strength + chroma change at each beat
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = S[freqs < 150].sum(axis=0)
    low_on = np.maximum(0, np.diff(low, prepend=low[0]))
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=SR, hop_length=hop)
    fr = lambda t: min(int(t * SR / hop), S.shape[1] - 1)
    beat_low = np.array([low_on[max(0, fr(t) - 2): fr(t) + 4].max() for t in grid])
    beat_chg = []
    for t in grid:
        a0, a1, b1 = fr(max(0, t - T)), fr(t), fr(min(dur, t + T))
        pa = chroma[:, a0:a1].mean(axis=1) if a1 > a0 else chroma[:, a1]
        pb = chroma[:, a1:b1].mean(axis=1) if b1 > a1 else chroma[:, a1]
        beat_chg.append(1 - float(np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-9)))
    beat_chg = np.array(beat_chg)
    zl = (beat_low - beat_low.mean()) / (beat_low.std() + 1e-9)
    zc = (beat_chg - beat_chg.mean()) / (beat_chg.std() + 1e-9)
    ph_score = [float(np.mean(zl[p::bpb]) + np.mean(zc[p::bpb])) for p in range(bpb)]
    order = np.argsort(ph_score)[::-1]
    phase = int(order[0]) if a.phase == 'auto' else int(a.phase)
    conf = ph_score[order[0]] - ph_score[order[1]]
    alt = int(order[1]) if order[0] == phase else int(order[0])
    local = local_phases(zl + zc, grid, bpb, phase)
    bars = grid[phase::bpb]
    bars = bars[bars + bpb * T <= dur + 1e-3]
    nb = len(bars)
    if nb < 2:
        sys.exit('the track is shorter than two bars')
    # bar features
    mf = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), sr=SR, n_mfcc=13)
    feats, loud, dens = [], [], []
    for b in bars:
        s0, s1 = int(b * SR), int((b + bpb * T) * SR)
        f0, f1 = fr(b), fr(b + bpb * T)
        loud.append(20 * np.log10(np.sqrt(np.mean(y[s0:s1] ** 2)) + 1e-9))
        dens.append(float((oenv[f0:f1] > np.median(oenv) * 1.5).mean()))
        c = chroma[:, f0:f1].mean(axis=1); c = c / (np.linalg.norm(c) + 1e-9)
        m = mf[1:, f0:f1].mean(axis=1); m = m / (np.linalg.norm(m) + 1e-9)
        feats.append(np.concatenate([c, m]))
    feats, loud = np.array(feats), np.array(loud)
    rel = loud - np.percentile(loud, 90)
    nov = np.r_[0, np.linalg.norm(np.diff(feats, axis=0), axis=1)]
    rise = np.r_[0, np.diff(loud)]
    # a drop: a loud bar (within 3 dB of the loud sections) that stays loud for 4 bars, lifted over the 4 bars before it
    lift = np.array([rel[i] - np.median(rel[max(0, i - 4):i]) if i else 0.0 for i in range(nb)])
    loud_now = (rel > -3) & (np.array([rel[i:i + 4].mean() for i in range(nb)]) > -4)
    drop_score = np.where(loud_now if loud_now.any() else rel > np.percentile(rel, 75), lift / 3 + nov * 2, -np.inf)
    drop_cands = [int(i) for i in np.argsort(drop_score)[::-1][:6] if np.isfinite(drop_score[i])]
    drop_src = drop_cands[0] if a.drop == 'auto' else None if a.drop == 'none' else int(a.drop)
    for what, v in (('--drop', drop_src), ('--start-bar', a.start_bar), ('--end-bar', a.end_bar)):
        if v is not None and not 0 <= v < nb:
            sys.exit(f'{what} {v} is outside the source bars 0..{nb - 1}')
    if not 1 <= drop_k <= FILM_BARS:
        sys.exit(f'--drop-film-bar {drop_k} is outside the film bars 1..{FILM_BARS}')
    # how different bar j sounds from bar i
    D = np.linalg.norm(feats[:, None] - feats[None], axis=2) + np.abs(loud[:, None] - loud[None]) / 6
    nxt = np.minimum(np.arange(nb) + 1, nb - 1)
    J = a.jump + 2.5 * D[nxt]                       # J[i, j]: jump from source bar i to j
    J[np.arange(nb - 1), np.arange(1, nb)] = 0.0    # the natural continuation is free
    J[np.arange(nb), np.arange(nb)] += 1.5          # repeating the same bar
    cont = np.zeros((nb, nb), bool)
    cont[np.arange(nb - 1), np.arange(1, nb)] = True
    INF = 1e18
    arc = np.array(ARC, float)
    cost = np.full((FILM_BARS, nb), INF)
    back = np.zeros((FILM_BARS, nb), int)
    cost[0] = np.abs(rel - arc[0]) / 3.0 + 0.4 * np.arange(nb)  # prefer starting near the top
    if a.start_bar is not None:
        keep = cost[0, a.start_bar] - 0.4 * a.start_bar
        cost[0] = INF
        cost[0, a.start_bar] = keep
    for k in range(1, FILM_BARS):
        Ck = J + np.where(cont, 0.0, 1.0 if k % a.phrase else 0.0)  # film bar k+1 not on a phrase start: jumps cost more
        tot = cost[k - 1][:, None] + Ck
        back[k] = np.argmin(tot, axis=0)
        cost[k] = tot[back[k], np.arange(nb)] + np.abs(rel - arc[k]) / 3.0
        if k == drop_k - 1 and drop_src is not None:
            cost[k, np.arange(nb) != drop_src] = INF
        if k == FILM_BARS - 1 and a.end_bar is not None:
            cost[k, np.arange(nb) != a.end_bar] = INF
    jend = int(np.argmin(cost[-1]))
    path = [jend]
    for k in range(FILM_BARS - 1, 0, -1):
        path.append(int(back[k, path[-1]]))
    path = path[::-1]
    segs = []
    for k, j in enumerate(path):
        if segs and j == segs[-1]['last'] + 1:
            segs[-1]['last'] = j
            segs[-1]['bars'] += 1
        else:
            segs.append({'first': j, 'last': j, 'film_bar': k + 1, 'bars': 1})
    out_segs = []
    for s in segs:
        src_from = float(bars[s['first']])
        out_segs.append({'src_from': round(src_from, 4), 'src_to': round(src_from + bpb * T * s['bars'], 4),
                         'at': round(a.film_offset + bar_len * (s['film_bar'] - 1), 4),
                         'src_bars': [s['first'], s['last']], 'film_bars': [s['film_bar'], s['film_bar'] + s['bars'] - 1]})
    ratio = film_bpm / bpm
    music = {'file': (a.file or a.track).replace('\\', '/'), 'src_bpm': round(bpm, 3), 'target_bpm': film_bpm,
             'stretch': 'none' if abs(ratio - 1) < 0.002 else 'rubberband', 'gain_db': 0.0,
             'segments': [{'label': f"src bars {s['src_bars'][0]}-{s['src_bars'][1]} -> film bars "
                                    f"{s['film_bars'][0]}-{s['film_bars'][1]}",
                           'src_from': s['src_from'], 'src_to': s['src_to'], 'at': s['at'],
                           'fade_in': 0.008, 'fade_out': 0.06} for s in out_segs]}
    if music['stretch'] == 'none':
        music['src_bpm'] = film_bpm  # within 0.2 %: play it unstretched on the film grid
    low = conf < 0.5 and a.phase == 'auto'
    summary = (f'{bpm:.3f} BPM (grid residual {med_res * 1000:.1f} ms, {inlier:.0%} of beats on it), downbeats at '
               f'{bars[0]:.3f} + {bpb * T:.4f}*n s (phase {phase}' + (
                   f', confidence {conf:.2f}' + (f': LOW, the other candidate is {grid[alt]:.3f} + {bpb * T:.4f}*n s '
                                                 f'(--phase {alt})' if low else '') if a.phase == 'auto' else
                   f' as given; the automatic choice is phase {int(order[0])}, confidence {conf:.2f}')
               + f'); {nb} source bars -> {FILM_BARS} film bars in {len(out_segs)} segment(s), drop film bar {drop_k} = '
               + (f'source bar {drop_src}' + (' (top drop candidate)' if a.drop == 'auto' else '') if drop_src is not None
                  else 'chosen by the arc') + f'; stretch ratio {ratio:.4f}')
    if local:
        summary += ('; another downbeat phase wins locally at ' + ', '.join(
            f"{x['from']:.1f}-{x['to']:.1f} s (phase {x['phase']})" for x in local[:4]) + ': check the bar lines there')
    res = {'summary': summary, 'track': a.track, 'duration': round(dur, 3), 'bpm': round(bpm, 4), 'beat_period': round(T, 6),
           'film_bpm': film_bpm, 'beats_per_bar': bpb, 'film_bars': FILM_BARS, 'drop_film_bar': drop_k, 'arc': ARC,
           'stretch_ratio_to_film': round(ratio, 6), 't0': round(t0, 4), 'transient_shift_ms': round(shift * 1000, 1),
           'transient_fit_ratio': round(snap_ratio, 2) if snap_ratio else None,
           'grid_median_residual_ms': round(med_res * 1000, 1), 'grid_inlier_frac': round(inlier, 3),
           'downbeat_phase': phase, 'phase_scores': [round(s, 3) for s in ph_score], 'phase_confidence': round(conf, 3),
           'phase_low_confidence': low, 'phase_alternative': {'phase': alt, 'first_downbeat': round(float(grid[alt]), 3)},
           'local_phase_disagreements': local, 'drop_source_bar': drop_src,
           'bars': [{'i': i, 't': round(float(bars[i]), 3), 'rel_db': round(float(rel[i]), 1), 'novelty': round(float(nov[i]), 2),
                     'rise_db': round(float(rise[i]), 1), 'density': round(dens[i], 2)} for i in range(nb)],
           'drop_candidates': drop_cands, 'map': path, 'segments': out_segs, 'music': music,
           'total_cost': round(float(cost[-1, jend]), 2)}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)
    if a.json:
        print(json.dumps(res, indent=1))
        return res
    print(summary)
    print(f'transient fit moved the grid {shift * 1000:+.0f} ms (peak/median {snap_ratio or 0:.2f}); '
          f'phase scores {[round(s, 2) for s in ph_score]}; drop candidates (source bars) {drop_cands}')
    if low:
        print(f'LOW downbeat confidence: bars may start at {grid[alt]:.3f} + {bpb * T:.4f}*n s instead. Decide from the '
              'music (first note, section entrances, chord changes) and re-run with --phase N.')
    print('bar  t      rel_dB  nov   rise  dens')
    for b in res['bars']:
        mark = ' <drop?' if b['i'] in drop_cands[:3] else ''
        print(f"{b['i']:3d} {b['t']:7.2f} {b['rel_db']:6.1f} {b['novelty']:5.2f} {b['rise_db']:5.1f} {b['density']:5.2f}{mark}")
    print('map (film bar -> source bar):', ' '.join(f'{k + 1}:{j}' for k, j in enumerate(path)))
    print('design "music" block (check every splice, then fine-tune "at" by a few ms):')
    print(json.dumps(music, indent=1))
    if a.out:
        print(f'wrote {a.out}')
    return res


if __name__ == '__main__':
    main()
