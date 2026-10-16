"""Eyes for the forge: made sounds next to their nearest real references, as one PNG.

    from earshot.forge.viz import compare_png
    compare_png('made/keypress/compare.png', [('keypress_01', x1), ('keypress_02', x2)],
                [('kb_thock_02.wav', r2), ('kb_enter_01.wav', r5)], pairs=[0, 1])   # pairs: a ref index per made sound

One row per made sound: its log-frequency spectrogram | its nearest reference's spectrogram (same time span, same
80 dB range, both onset-aligned) | the two dB envelopes on one time axis (a tail that dies too fast shows here) |
the 1/3-octave tonal balance of the made sound against every reference's (min-max band and median). `pairs=None`
pairs each made sound with its nearest reference by earshot.forge.search.sound_dist (tonal balance + envelope).
Returns {'file', 'rows', 'pairs'}. Needs matplotlib (imported lazily, Agg backend).
"""
import numpy as np

SR = 48000


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def onset_align(x, db=-30.0, pre_ms=5.0):
    """the sound from `pre_ms` before its first sample within `db` of the peak"""
    m = _mono(x)
    on = int(np.argmax(np.abs(m) > np.abs(m).max() * 10 ** (db / 20)))
    return m[max(0, on - int(SR * pre_ms / 1000)):]


def spec_db(x, span_s):
    """log-magnitude STFT, 0 dB = the loudest bin, floored at -80: (t_ms, f_hz, S_db)"""
    from scipy import signal
    m = np.pad(x, (0, max(0, int(span_s * SR) - len(x))))[:max(int(span_s * SR), 256)]
    nfft = 512 if span_s < 0.6 else 1024 if span_s < 2.5 else 2048
    f, t, S = signal.stft(m, SR, nperseg=nfft, noverlap=nfft - nfft // 8, boundary=None)
    S = 20 * np.log10(np.abs(S) + 1e-12)
    return t * 1000, f, np.maximum(S - S.max(), -80.0)


def envelope_db(x, span_s, hop_ms=2.0):
    """`hop_ms` RMS envelope in dB re its peak, floored at -80: (t_ms, e_db)"""
    n = int(SR * hop_ms / 1000)
    m = np.pad(x, (0, max(0, int(span_s * SR) - len(x))))[:int(span_s * SR)]
    k = max(1, len(m) // n)
    e = 10 * np.log10((m[:k * n].reshape(k, n) ** 2).mean(1) + 1e-18)
    return np.arange(k) * hop_ms, np.maximum(e - e.max(), -80.0)


def compare_png(path, made, refs=(), pairs=None, title=None, max_s=8.0):
    """write the comparison sheet; made / refs = [(name, 48 kHz array), ...]; pairs = a ref index (or None) per made"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .search import describe, sound_dist
    from .tone import CEN, profile
    made, refs = list(made), list(refs)
    if pairs is None:
        rd = [describe(x) for _, x in refs]
        pairs = [int(np.argmin([sound_dist(describe(x), r) for r in rd])) if rd else None for _, x in made]
    RP = np.stack([profile(x) for _, x in refs]) if refs else None
    rows = max(1, len(made))
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8})
    fig, axes = plt.subplots(rows, 4, figsize=(15, 2.25 * rows + (0.5 if title else 0.2)), squeeze=False,
                             gridspec_kw={'width_ratios': [1.25, 1.25, 1.0, 1.0]})
    c_made, c_ref = '#1f77b4', '#d62728'
    for i, (name, x) in enumerate(made):
        a = onset_align(x)
        j = pairs[i] if i < len(pairs) else None
        b = onset_align(refs[j][1]) if j is not None else None
        span = min(max_s, max(len(a), len(b) if b is not None else 0) / SR)
        for col, (nm, y) in enumerate([(name, a), (refs[j][0] if j is not None else 'no reference', b)]):
            ax = axes[i, col]
            if y is None:
                ax.axis('off')
                continue
            t, f, S = spec_db(y, span)
            ax.pcolormesh(t, np.maximum(f, 20), S, shading='auto', cmap='magma', vmin=-80, vmax=0, rasterized=True)
            ax.set_yscale('log')
            ax.set_ylim(40, 24000)
            ax.set_xlim(0, span * 1000)
            ax.set_title(('made: ' if col == 0 else 'nearest ref: ') + nm, color=c_made if col == 0 else c_ref)
            ax.set_ylabel('Hz')
            if i == rows - 1:
                ax.set_xlabel('ms from onset')
        ax = axes[i, 2]
        hop = max(2.0, span * 1000 / 300)              # ~300 points per panel: long tails read as a line, not a band
        t, e = envelope_db(a, span, hop)
        ax.plot(t, e, color=c_made, lw=1.1, label='made')
        if b is not None:
            t2, e2 = envelope_db(b, span, hop)
            ax.plot(t2, e2, color=c_ref, lw=1.1, label='ref')
        ax.set_ylim(-70, 3)
        ax.set_xlim(0, span * 1000)
        ax.grid(alpha=0.25)
        ax.set_title('envelope (dB re peak)')
        ax.legend(loc='upper right', frameon=False)
        ax = axes[i, 3]
        if RP is not None:
            ax.fill_between(CEN, RP.min(0), RP.max(0), color=c_ref, alpha=0.18, lw=0, label='refs min-max')
            ax.plot(CEN, np.median(RP, 0), color=c_ref, lw=1.0, ls='--', label='refs median')
        p = profile(x)
        ax.plot(CEN, p, color=c_made, lw=1.3, label='made')
        if RP is not None:
            ax.set_title(f'1/3-octave balance: {np.sqrt(np.mean((p - np.median(RP, 0)) ** 2)):.1f} dB from the refs')
        else:
            ax.set_title('1/3-octave balance')
        ax.set_xscale('log')
        ax.set_xlim(25, 20000)
        ax.set_ylim(-60, 0)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc='lower left', frameon=False)
    for i in range(len(made), rows):
        for ax in axes[i]:
            ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return {'file': str(path).replace(chr(92), '/'), 'rows': len(made),
            'pairs': [refs[j][0] if j is not None else None for j in pairs]}
