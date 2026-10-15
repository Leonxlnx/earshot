"""An annotated picture of a soundtrack or film for agents that can look at images.

    earshot sheet soundtrack.wav -o sheet.png [--video film.mp4] [--cues cues.json] [--from 10 --to 25]
    earshot sheet film.mp4 -o film_sheet.png            # a video: frames along the top, its cuts as lines

    from earshot.sheet import sheet
    sheet('film.mp4', 'film_sheet.png')                 # {'png': ..., 'summary': ...}

Top to bottom, on one shared time axis (m:ss): video frames (one per slot, taken just after a cut when the slot has
one; thin ticks show each frame's exact time), sections with their tags, event labels (kind + best CLAP tag,
staggered so they never overlap; pale with '?' = the event is under 12 dB over its bed, so the tag is a weak hint;
untagged onsets are short ticks), a log-frequency spectrogram (30 Hz - 16 kHz; digital silence draws black), the
waveform with momentary / short-term loudness (LUFS), the integrated level and gaps (yellow), then beats (tall =
downbeat guess) and cue / cut markers coloured by sync (green = an onset within a frame, amber = within 100 ms, blue =
a whoosh / riser peak, violet = only a level peak, red = none). Picture cuts are dashed lines through every lane. The
header repeats the key numbers (loudness, tempo, one sync line each for cuts and cues with their chance levels).
Built from earshot.timeline (its cached result is reused); --from/--to zooms in. An audio file's cover art is not a
video; a picture-only video draws its frames and cuts over silence.
"""
import os

import numpy as np

from . import audio
from .report import emit, fmt_t
from .timeline import DETAIL, timeline

SR = audio.SR
KIND_COLOR = {'WHOOSH': '#1f77b4', 'RISER': '#17becf', 'HIT': '#d62728', 'KEY': '#2ca02c', 'UI': '#8c564b',
              'STEP': '#bcbd22', 'FOLEY': '#e377c2', 'NOTE': '#9467bd', 'DRUM': '#ff7f0e', 'VOICE': '#0b6e4f',
              '-': '#7f7f7f'}
SYNC_COLOR = {'OK': '#2ca02c', 'near': '#ff9f1c', 'swell': '#1f77b4', 'peak': '#9467bd', 'off': '#d62728',
              'no sound': '#d62728'}
CUT_COLOR = '#00a6d6'
FONT = 'DejaVu Sans'
MONO = 'DejaVu Sans Mono'


def _ticks(t0, t1, max_n=24):
    span = max(t1 - t0, 1e-3)
    step = next((s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if span / s <= max_n), 600)
    first = np.ceil(t0 / step) * step
    return np.arange(first, t1 + 1e-9, step), step


def _tick_label(t, step):
    m = int(t // 60)
    s = t - 60 * m
    if step >= 1:
        return f'{m}:{int(round(s)):02d}'
    nd = 1 if abs(step * 10 - round(step * 10)) < 1e-9 else 2   # 0.25 s ticks need two decimals
    return f'{m}:{s:0{3 + nd}.{nd}f}'


def _logspec(m, t0, n_cols, n_rows=240, fmin=30.0, fmax=16000.0):
    """log-frequency magnitude spectrogram (dB) with about n_cols time columns"""
    n_fft = 4096
    hop = max(64, int(len(m) / n_cols))
    y = np.pad(np.asarray(m, np.float32), (n_fft // 2, n_fft // 2))
    n = max(1, 1 + (len(y) - n_fft) // hop)
    fr = np.lib.stride_tricks.as_strided(y, (n, n_fft), (y.strides[0] * hop, y.strides[0]))
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    edges = np.geomspace(fmin, fmax, n_rows + 2)
    lo, c, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    fb = np.maximum(0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))
    fb[fb.sum(axis=1) == 0, np.argmin(np.abs(f[None, :] - c[fb.sum(axis=1) == 0]), axis=1)] = 1.0
    fb /= fb.sum(axis=1, keepdims=True)
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n_rows, n), np.float32)
    for a in range(0, n, 1024):
        P = np.abs(np.fft.rfft(fr[a:a + 1024] * win, axis=1)) ** 2
        out[:, a:a + 1024] = 10 * np.log10(fb.astype(np.float32) @ P.T + 1e-14)
    return out, edges[1:-1]


def _thumb_times(t0, t1, cuts, n_slots):
    """one frame per slot: just after the first cut inside the slot, else the slot centre"""
    w = (t1 - t0) / n_slots
    out = []
    for k in range(n_slots):
        a, b = t0 + k * w, t0 + (k + 1) * w
        inside = [c for c in cuts if a <= c < b - 0.2]
        out.append(min(inside[0] + 0.15, b - 0.05) if inside else (a + b) / 2)
    return out


def _place_labels(items, t0, t1, plot_px, char_px, n_rows):
    """greedy staggering: items [(priority, t, text)] -> [(row, t, text)] with no horizontal overlap in a row"""
    px_s = plot_px / max(t1 - t0, 1e-6)
    rows = [[] for _ in range(n_rows)]
    out = []
    for _, t, text in sorted(items, key=lambda q: q[0]):
        w = (len(text) * char_px + 16) / px_s
        a, b = t - 0.15 * w, t + 0.85 * w
        if b > t1:
            a, b = t1 - w, t1
        a = max(a, t0)
        for r, used in enumerate(rows):
            if all(b <= u0 or a >= u1 for u0, u1 in used):
                used.append((a, b))
                out.append((r, t, a, text))
                break
    return out


def sheet(path, out_png, video=None, cues=None, start=None, end=None, detail='normal', width=2000, tl=None):
    """draw the annotated sheet PNG; returns {'png', 'summary', ...}. tl: a timeline() result to reuse"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patches
    r = tl or timeline(path, video=video, cues=cues, start=start, end=end, detail=detail)
    video = r.get('video')
    t0, t1 = r['start'], r['end']
    has_audio = r.get('has_audio', True)
    m = audio.to_mono(audio.load(path, start=t0, end=t1)) if has_audio else np.zeros(int((t1 - t0) * SR), np.float32)
    width = int(min(max(width, 1000), 2400))
    dpi = 100
    fs = max(11.0, width / 125)                  # base font size (pt)
    left_px, right_px = max(92, int(6.4 * fs) + 8), 64
    plot_px = width - left_px - right_px
    lf = fs * 0.85                               # event label font size (pt, monospace)
    char_px = 0.602 * lf * dpi / 72
    lanes = []
    if video:
        lanes.append(('frames', 1.55))
    lanes += [('sections', 0.6), ('labels', 1.55), ('spec', 3.6), ('wave', 0.95), ('loud', 1.75), ('beats', 0.75)]
    lo, mu = r['loudness'], r['music']
    sub = []
    if lo.get('integrated_lufs') is not None:
        sub.append(f"{lo['integrated_lufs']:.1f} LUFS   TP {lo['true_peak_dbtp']:.1f} dBTP" +
                   (f"   LRA {lo['lra_lu']:.1f} LU" if lo.get('lra_lu') is not None else ''))
    elif lo.get('digital_silence'):
        sub.append('digital silence' if r.get('has_audio', True) else 'no audio stream')
    if mu.get('tempo') and mu['confidence'] != 'none':
        sub.append(f"~{mu['tempo']:.0f}" + (f" (or {mu['tempo_alt']:.0f})" if mu.get('tempo_alt') else '') +
                   f" BPM ({mu['confidence']})")
    sub.append(f"{len(r['events'])} events")
    lines = [(os.path.basename(path) + f"   {fmt_t(t0)}-{fmt_t(t1)}", fs * 1.25, 'bold', '#000000'),
             ('    '.join(sub), fs * 0.95, 'normal', '#333333')]
    for kind in ('cuts', 'cues'):
        st = (r['sync'].get(kind) or {}).get('stats') or {}
        if st.get('n'):
            k1 = 'hit_within_1_frame' if 'hit_within_1_frame' in st else 'hit_within_40ms'
            ka = k1.replace('hit', 'accent')
            acc = (f", {100 * st[ka]:.0f}% on a strong accent (chance "
                   f"{100 * st[ka.replace('accent', 'accent_chance')]:.0f}%)" if ka in st else '')
            pk = f", {st['answered_by_level_peak']} only a level peak" if st.get('answered_by_level_peak') else ''
            lines.append((f"sync {st['n']} {kind}: {100 * st[k1]:.0f}% onset within "
                          f"{'1 frame' if 'frame' in k1 else '40 ms'} (chance "
                          f"{100 * st.get(k1.replace('hit', 'chance'), 0):.0f}%){acc}{pk}, "
                          f"{len(st['no_sound'])} without a hit", fs * 0.95, 'normal', '#333333'))
    if r['sync']:
        shapes = ', '.join(n for k, n in (('cues', 'v cue'), ('cuts', 'diamond cut')) if r['sync'].get(k))
        lines.append((f'markers ({shapes}): green = an onset within 1 frame, amber = within 100 ms, blue = a whoosh / '
                      'riser peak, violet = only a level peak (soft whoosh?), red = no hit.' +
                      ('  Dashed lines = picture cuts.' if r.get('cuts') else ''), fs * 0.72, 'normal', '#555555'))
    if any(e.get('label') for e in r['events']):
        lines.append(("event labels: kind + CLAP tag, hints only; pale + '?' = under 12 dB over the bed (mostly bed in "
                      'the excerpt): a weak hint.', fs * 0.72, 'normal', '#555555'))
    head = 0.22 + sum(sz * 1.45 / 72 for _, sz, _, _ in lines)
    fig_h = round((head + sum(h for _, h in lanes) + 0.55) * dpi) / dpi
    fig = plt.figure(figsize=(width / dpi, fig_h), dpi=dpi)
    fig.patch.set_facecolor('white')
    axes, y = {}, fig_h - head
    for name, h in lanes:
        y -= h
        axes[name] = fig.add_axes([left_px / width, y / fig_h, plot_px / width, (h - 0.06) / fig_h])
        axes[name].set_xlim(t0, t1)
    ticks, step = _ticks(t0, t1)
    for name, ax in axes.items():
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=fs * 0.85, length=3)
        ax.set_xticklabels([] if name != 'beats' else [_tick_label(t, step) for t in ticks], fontfamily=FONT)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color('#999999')

    def lane_title(ax, text, inside=False):
        if inside:
            ax.text(0.004, 0.94, text, transform=ax.transAxes, ha='left', va='top', fontsize=fs * 0.78, zorder=6,
                    fontfamily=FONT, color='#222222', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
        else:
            ax.text(-0.006, 0.5, text, transform=ax.transAxes, ha='right', va='center', fontsize=fs * 0.82,
                    fontfamily=FONT, color='#333333')

    yy = 0.12
    for text, size, weight, color in lines:
        yy += size * 0.725 / 72
        fig.text(left_px / width, 1 - yy / fig_h, text, fontsize=size, fontfamily=FONT, weight=weight, va='center',
                 color=color)
        yy += size * 0.725 / 72
    n_thumbs = 0
    if video:
        from . import video as vid
        ax = axes['frames']
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        info = vid.info(video)
        aspect = (info.get('width') or 16) / float(info.get('height') or 9)
        th_px = 1.55 * dpi - 18
        tw_px = th_px * aspect
        n_slots = max(1, int(plot_px // (tw_px + 6)))
        times = _thumb_times(t0, t1, r.get('cuts') or [], n_slots)
        slot = (t1 - t0) / n_slots
        for k, t in enumerate(times):
            im = vid.frame(video, t, width=int(tw_px * 1.5))
            if im is None:
                continue
            a = t0 + k * slot + (slot - tw_px / plot_px * (t1 - t0)) / 2
            ax.imshow(np.asarray(im), extent=(a, a + tw_px / plot_px * (t1 - t0), 0.12, 1.0), aspect='auto',
                      zorder=1)
            ax.plot([t, t], [0, 0.12], color='#333333', lw=1.2, zorder=2)
            n_thumbs += 1
        ax.set_xlim(t0, t1)
        lane_title(ax, 'frames')

    ax = axes['sections']
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    for k, s in enumerate(r['sections']):
        a, b = max(s['start'], t0), min(s['end'], t1)
        ax.add_patch(patches.Rectangle((a, 0.05), b - a, 0.9, fc=('#e8eef7', '#f4ead9')[k % 2], ec='#8899aa', lw=0.8))
        n_chars = int((b - a) / (t1 - t0) * plot_px / (0.55 * fs * 0.9 * dpi / 72)) - 1
        lab = s['label'] if len(s['label']) <= n_chars else s['label'][:max(0, n_chars - 2)] + '..'
        if n_chars >= 4:
            ax.text(a + (t1 - t0) * 4 / plot_px, 0.5, lab, va='center', ha='left', fontsize=fs * 0.9, fontfamily=FONT,
                    clip_on=True)
    lane_title(ax, 'sections')

    ax = axes['labels']
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.axis('off')
    n_rows = 4
    near = {it.get('event_t') for k in ('cuts', 'cues') for it in (r['sync'].get(k) or {}).get('items', [])}
    items = []
    for e in r['events']:
        if e.get('label'):
            pri = (0 if e['t'] in near else 1, -e['prominence_db'])
            items.append((pri, e['t'], f"{e['kind']}{'' if e.get('exposed') else '?'} {e['label']}"))
    placed = _place_labels(items, t0, t1, plot_px, char_px, n_rows)
    for row, t, a, text in placed:
        yy = 1 - (row + 0.5) / n_rows
        kind = text.split(' ', 1)[0]
        col, alpha = KIND_COLOR.get(kind.rstrip('?'), '#777777'), 0.55 if kind.endswith('?') else 1.0
        ax.text(a, yy, text, fontsize=lf, fontfamily=MONO, va='center', ha='left', color='white',
                bbox=dict(boxstyle='square,pad=0.15', fc=col, ec='none', alpha=alpha), clip_on=False, zorder=3)
        ax.plot([t, t], [0, yy - 0.5 / n_rows + 0.02], color=col, lw=1.0, alpha=alpha, zorder=2)
    lane_title(axes['labels'], 'events')
    xk = 1 - right_px / width
    for k in sorted({text.split(' ', 1)[0].rstrip('?') for _, _, _, text in placed}, reverse=True):
        fig.text(xk, 1 - 0.32 / fig_h, k, fontsize=fs * 0.85, fontfamily=MONO, ha='right', va='center',
                 color='white', bbox=dict(boxstyle='round,pad=0.25', fc=KIND_COLOR.get(k, '#777777'), ec='none'))
        xk -= (len(k) * char_px * 0.9 + 22) / width

    ax = axes['spec']
    S, fc = _logspec(m, t0, plot_px)
    top = max(float(np.percentile(S, 99.7)), -40.0)        # -40 dB here ~ a -100 dBFS tone: silence draws black
    ax.imshow(S, origin='lower', aspect='auto', cmap='magma', vmin=top - 75, vmax=top,
              extent=(t0, t1, 0, len(fc)), interpolation='bilinear')
    yt = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ax.set_yticks([float(np.interp(np.log(f), np.log(fc), np.arange(len(fc)))) for f in yt])
    ax.set_yticklabels(['50', '100', '200', '500', '1k', '2k', '5k', '10k'], fontsize=fs * 0.8, fontfamily=FONT)
    for e in r['events']:
        h = len(fc) * (0.07 if e.get('label') else 0.035)
        ax.plot([e['t'], e['t']], [len(fc) - h, len(fc)], color='white', lw=1.0 if e.get('label') else 0.6,
                alpha=0.9 if e.get('label') else 0.55)
    ax.set_ylim(0, len(fc))
    lane_title(ax, 'spectrogram (Hz, log)', inside=True)

    ax = axes['wave']
    n = len(m)
    edges = np.linspace(0, n, min(plot_px, max(1, n)) + 1).astype(int)
    seg_max = np.maximum.reduceat(m, edges[:-1]) if n else np.zeros(1)
    seg_min = np.minimum.reduceat(m, edges[:-1]) if n else np.zeros(1)
    pk = float(np.abs(m).max()) if n else 0.0
    sc = max(float(np.percentile(np.abs(m), 99.9)) if n else 0.0, 1e-9)
    ax.fill_between(t0 + edges[:-1] / SR, np.clip(seg_min / sc, -1, 1), np.clip(seg_max / sc, -1, 1), color='#6b7a90',
                    lw=0, step='post')
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([])
    lane_title(ax, f'waveform (full height = {audio.db(sc):.0f} dBFS, peak {audio.db(pk):.1f} dBFS)' if pk > 0 else
               'waveform (digital silence)', inside=True)

    ax = axes['loud']
    lo_y = -60.0
    if lo.get('momentary'):
        mm = lo['momentary']
        tt = mm['t0'] + mm['dt'] * np.arange(len(mm['lufs']))
        ax.plot(tt, np.maximum(mm['lufs'], lo_y), color='#3d6fb6', lw=1.0, label='momentary')
    if lo.get('short_term'):
        ss = lo['short_term']
        tt = ss['t0'] + ss['dt'] * np.arange(len(ss['lufs']))
        ax.plot(tt, np.maximum(ss['lufs'], lo_y), color='#0b2e66', lw=2.4, label='short-term')
    if lo.get('integrated_lufs') is not None and lo['integrated_lufs'] > lo_y:
        ax.axhline(lo['integrated_lufs'], color='#d1495b', lw=1.2, ls='--')
        ax.text(t1, lo['integrated_lufs'], f" {lo['integrated_lufs']:.1f}", color='#d1495b', fontsize=fs * 0.8,
                va='center', ha='left', fontfamily=FONT, clip_on=False)
    for g in lo.get('silences', []):
        ax.axvspan(g['start'], g['end'], color='#ffdd57', alpha=0.45, lw=0)
    for c in lo['clipping']['times']:
        ax.plot([c, c], [-3, 0], color='red', lw=2)
    ax.set_ylim(lo_y, 0)
    ax.set_yticks([-50, -40, -30, -20, -10])
    ax.tick_params(axis='y', labelsize=fs * 0.8)
    ax.legend(loc='lower left', frameon=False, ncol=2, prop={'family': FONT, 'size': fs * 0.78})
    lane_title(ax, 'loudness (LUFS)', inside=True)

    ax = axes['beats']
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    beats = mu.get('beats') or []
    if beats and mu.get('confidence') != 'none':
        db = set(mu.get('downbeats') or [])
        for b in beats:
            if t0 <= b <= t1:
                ax.plot([b, b], [0.0, 0.45 if b in db else 0.28], color='#222222' if b in db else '#666666', lw=1.0)
    for it in (r['sync'].get('cues') or {}).get('items', []):
        ax.plot([it['t']], [0.84], marker='v', ms=fs * 0.55, color=SYNC_COLOR.get(it['status'], '#777777'), mew=0)
    for it in (r['sync'].get('cuts') or {}).get('items', []):
        ax.plot([it['t']], [0.6], marker='D', ms=fs * 0.45, color=SYNC_COLOR.get(it['status'], '#777777'), mew=0)
    lane_title(ax, 'sync /\nbeats', inside=False)
    ax.set_xlabel('time (m:ss)', fontsize=fs * 0.85, fontfamily=FONT)

    for c in r.get('cuts') or []:
        for name, a in axes.items():
            if name != 'labels' and t0 <= c <= t1:
                a.axvline(c, color=CUT_COLOR, lw=1.1, ls=(0, (4, 3)), alpha=0.9, zorder=0.5 if name == 'frames' else 4)
    d = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(d, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor='white')
    plt.close(fig)
    w_px, h_px = int(width), int(round(fig_h * dpi))
    res = {'png': os.path.abspath(out_png), 'width': w_px, 'height': h_px, 'start': t0, 'end': t1,
           'thumbnails': n_thumbs, 'labelled_events': len(placed), 'tagged_events': len(items),
           'events': len(r['events']), 'cuts': len(r.get('cuts') or []), 'timeline_summary': r['summary']}
    res['summary'] = (f"Wrote {res['png']} ({w_px}x{h_px}): {fmt_t(t0)}-{fmt_t(t1)}, {n_thumbs} frames, "
                      f"{len(placed)} of {len(items)} tagged events labelled, {len(r['events'])} event ticks, "
                      f"{res['cuts']} cut lines. Open it with an image viewer / Read tool."
                      ).encode('ascii', 'backslashreplace').decode()
    res['text'] = res['summary'] + '\n\nTimeline: ' + r['summary']
    return res


def add_cli(sub):
    from .cli import common
    p = sub.add_parser('sheet', help='annotated PNG of a soundtrack / film: frames, spectrogram, events, loudness, sync')
    p.add_argument('file', help='audio file or video')
    p.add_argument('-o', dest='png', help='output PNG (default: <file stem>_sheet.png in the current folder)')
    p.add_argument('--video', help='video for frames and cuts (default: FILE itself when it is a video)')
    p.add_argument('--cues', help='cue events: JSON file {fps, events:[{f, kind, id}]}, a JSON list, or "1.5,3.2,..." s')
    p.add_argument('--from', '--start', dest='start', type=float, help='zoom: start (s)')
    p.add_argument('--to', '--end', dest='end', type=float, help='zoom: end (s)')
    p.add_argument('--detail', choices=list(DETAIL), default='normal', help='timeline model budget')
    p.add_argument('--width', type=int, default=2000, help='image width in px (1000-2400)')
    common(p)

    def run(a):
        png = a.png or os.path.splitext(os.path.basename(a.file))[0] + '_sheet.png'
        return emit(sheet(a.file, png, a.video, a.cues, a.start, a.end, a.detail, a.width), a) and 0
    p.set_defaults(func=run)


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
