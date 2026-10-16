"""Pictures for the pro motion-SFX study: paged overviews (8 s rows) and zoomed crops of single events.

    python eval/forge/pro/viewer.py overview ref_1            # out/refs_pro/_work/view/ref_1_p1.png ...
    python eval/forge/pro/viewer.py crops ref_1 6.07 10.93    # out/refs_pro/_work/view/ref_1_crops.png

Each lane: log-frequency spectrogram (mid = (L+R)/2, 30 Hz - 20 kHz, 2048-pt FFT, 5.3 ms hop), picture cuts as cyan
dashed lines, earshot timeline events as ticks (red = >= 12 dB over a running bed), then band levels (sub < 120 Hz,
mid 120 Hz - 2 kHz, high > 4 kHz, dB) and the side/mid ratio (dB - 40, stereo width) underneath.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from videos import WORK, wav  # noqa: E402

SR = 48000
NFFT, HOP = 2048, 256
FMIN, FMAX, NROWS = 30.0, 20000.0, 200


def load(ref):
    x, sr = sf.read(wav(ref), dtype='float32', always_2d=True)
    assert sr == SR
    return x


def tl(ref):
    p = f'{WORK}/tl/{ref}.json'
    if os.path.exists(p):
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    return None


def cuts_events(ref):
    r = tl(ref)
    if r is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
        from earshot import video
        from videos import path
        return video.cuts(path(ref)), []
    return r.get('cuts') or [], r.get('events') or []


def logspec(m, t0, t1):
    a, b = max(0, int(t0 * SR) - NFFT // 2), min(len(m), int(t1 * SR) + NFFT // 2)
    seg = m[a:b]
    f, t, Z = signal.stft(seg, SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    edges = np.geomspace(FMIN, FMAX, NROWS + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (f >= lo) & (f < hi)
        if not k.any():
            rows.append(P[np.argmin(np.abs(f - np.sqrt(lo * hi)))])
        else:
            rows.append(P[k].mean(0))
    S = 10 * np.log10(np.array(rows) + 1e-12)
    tt = (a + NFFT // 2 + t * SR - NFFT // 2) / SR
    return tt, edges, S


def band_db(m, t0, t1, lo, hi, win=0.02):
    a, b = max(0, int(t0 * SR)), min(len(m), int(t1 * SR))
    seg = m[a:b]
    if lo <= 0:
        sos = signal.butter(4, hi, 'lowpass', fs=SR, output='sos')
    elif hi >= SR / 2:
        sos = signal.butter(4, lo, 'highpass', fs=SR, output='sos')
    else:
        sos = signal.butter(4, [lo, hi], 'bandpass', fs=SR, output='sos')
    y = signal.sosfiltfilt(sos, seg)
    n = int(win * SR)
    k = len(y) // n
    e = 10 * np.log10((y[:k * n].reshape(k, n) ** 2).mean(1) + 1e-12)
    return a / SR + (np.arange(k) + 0.5) * win, e


def side_mid(x, t0, t1, win=0.02):
    a, b = max(0, int(t0 * SR)), min(len(x), int(t1 * SR))
    L, R = x[a:b, 0], x[a:b, 1]
    M, S = (L + R) / 2, (L - R) / 2
    n = int(win * SR)
    k = len(M) // n
    em = (M[:k * n].reshape(k, n) ** 2).mean(1)
    es = (S[:k * n].reshape(k, n) ** 2).mean(1)
    return a / SR + (np.arange(k) + 0.5) * win, 10 * np.log10((es + 1e-12) / (em + 1e-12))


def lane(fig, gs_spec, gs_env, x, t0, t1, cuts, events, title=None, vmax=None):
    m = x.mean(1)
    tt, edges, S = logspec(m, t0, t1)
    ax = fig.add_subplot(gs_spec)
    vmax = vmax if vmax is not None else np.percentile(S, 99.7)
    ax.pcolormesh(tt, np.sqrt(edges[:-1] * edges[1:]), S, shading='nearest', vmin=vmax - 75, vmax=vmax, cmap='magma', rasterized=True)
    ax.set_yscale('log')
    ax.set_ylim(FMIN, FMAX)
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_yticklabels(['50', '100', '200', '500', '1k', '2k', '5k', '10k'], fontsize=7)
    ax.minorticks_off()
    ax.set_xlim(t0, t1)
    ax.tick_params(labelbottom=False, labelsize=7)
    for c in cuts:
        if t0 <= c <= t1:
            ax.axvline(c, color='#00e5ff', ls='--', lw=1.0)
    for e in events:
        if t0 <= e['t'] <= t1:
            strong = e.get('prominence_db', 0) >= 12 and e.get('bed_db', -80) > -60
            ax.plot([e['t']], [FMAX * 0.8], marker='v', ms=4, color='#ff3030' if strong else '#bbbbbb')
    if title:
        ax.set_title(title, fontsize=8, loc='left')
    ax2 = fig.add_subplot(gs_env, sharex=ax)
    for lo, hi, col, nm in ((0, 120, '#1f77b4', 'sub'), (120, 2000, '#2ca02c', 'mid'), (4000, 24000, '#d62728', 'hi')):
        tb, eb = band_db(m, t0, t1, lo, hi)
        ax2.plot(tb, eb, color=col, lw=0.8, label=nm)
    ts, es = side_mid(x, t0, t1)
    ax2.plot(ts, es - 40, color='#9467bd', lw=0.6, label='S/M-40')
    ax2.set_ylim(-95, 0)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=7)
    for c in cuts:
        if t0 <= c <= t1:
            ax2.axvline(c, color='#00a6d6', ls='--', lw=0.8)
    return ax, ax2


def overview(ref, row_s=8.0, rows_per_page=6):
    x = load(ref)
    dur = len(x) / SR
    cuts, events = cuts_events(ref)
    vmax = np.percentile(logspec(x.mean(1), 0, dur)[2], 99.7)
    n_rows = int(np.ceil(dur / row_s))
    outs = []
    os.makedirs(f'{WORK}/view', exist_ok=True)
    for p0 in range(0, n_rows, rows_per_page):
        rr = range(p0, min(n_rows, p0 + rows_per_page))
        fig = plt.figure(figsize=(16, 3.0 * len(rr)), dpi=100)
        gs = fig.add_gridspec(3 * len(rr), 1, height_ratios=[3, 1.1, 0.35] * len(rr), hspace=0.08)
        for i, r in enumerate(rr):
            t0 = r * row_s
            ax, ax2 = lane(fig, gs[3 * i], gs[3 * i + 1], x, t0, t0 + row_s, cuts, events, vmax=vmax,
                           title=f'{ref}  {t0:.0f}-{t0 + row_s:.0f} s')
            ticks = np.arange(t0, t0 + row_s + 1e-6, 0.25)
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([f'{v:.0f}' if abs(v - round(v)) < 1e-6 else ('.5' if abs(v % 1 - 0.5) < 1e-6 else '')
                                 for v in ticks], fontsize=7)
            if i == 0:
                ax2.legend(fontsize=6, ncol=4, loc='lower right')
        out = f'{WORK}/view/{ref}_p{p0 // rows_per_page + 1}.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        outs.append(out)
    return outs


def crops(ref, specs, cols=3, out=None, half=1.0):
    """specs: [(t_center, half_width or None, label)]"""
    x = load(ref)
    cuts, events = cuts_events(ref)
    n = len(specs)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(5.6 * cols, 3.3 * rows), dpi=100)
    gs = fig.add_gridspec(2 * rows, cols, height_ratios=[3, 1.2] * rows, hspace=0.25, wspace=0.18)
    vmax = np.percentile(logspec(x.mean(1), 0, len(x) / SR)[2], 99.7)
    for i, (tc, hw, lab) in enumerate(specs):
        hw = hw or half
        r, c = divmod(i, cols)
        ax, ax2 = lane(fig, gs[2 * r, c], gs[2 * r + 1, c], x, tc - hw, tc + hw, cuts, events, vmax=vmax,
                       title=f'{ref} {tc:.2f}s {lab}')
        ax2.tick_params(labelsize=6)
    out = out or f'{WORK}/view/{ref}_crops.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


if __name__ == '__main__':
    cmd, ref = sys.argv[1], sys.argv[2]
    if cmd == 'overview':
        for r in sys.argv[2:]:
            print(overview(r), flush=True)
    else:
        print(crops(ref, [(float(a), None, '') for a in sys.argv[3:]]))
