"""How close is a resynthesis to its recording? One set of fit-quality metrics for every fitter.

    from earshot.forge.fit.quality import quality, floor, score, pair_png
    q = quality(original, resynth)                  # paths or 48 kHz arrays (mono or (n, 2))
    q['mel_db'], q['env_corr'], q['tone_db'], q['onset_err']
    q = quality(original, resynth, clap=True)       # + q['clap_cos'] (loads the CLAP model once)
    fl = floor(lambda s: render(model, seed=s))     # quality(render seed 11, render seed 12): the noise floor
    pair_png('pair.png', original, resynth)         # side-by-side spectrograms + envelopes + 1/3-octave balance

Metrics (both sounds mono-summed, except the stereo ones; the ORIGINAL's onset trims both by the same amount, so a
timing error still counts; each is levelled to its own loudest 50 ms RMS, so a level offset is reported apart as
`level_db`):
  mel_db      multi-resolution log-mel distance: 64 mel bands 30 Hz - 20 kHz, FFT frames 256 / 1024 / 4096 (hop 1/4),
              mean |dB| over the cells within 60 dB of the louder sound's top (floored 80 dB down), averaged over
              the three resolutions (mel_256 / mel_1024 / mel_4096 are reported too). 0 = identical spectrograms.
  env_corr    correlation of the 5 ms dB envelopes over the active span (1 = the same loudness contour)
  attack_ms   (original, resynth): the first onset (first 0.125 ms frame within 20 dB of the loudest) to its local
              peak (loudest frame in the next 10 ms)
  sharp_db_ms (original, resynth): onset sharpness, 17 dB / the time the first onset takes to rise from 20 dB to 3 dB
              under its local peak (independent of a floor or fade-in under the sound)
  onset_err   |log2| of the sharpness ratio (octaves; 0 = the same onset, 1 = twice as sharp or blunt)
  tone_db     earshot.forge.tone: RMS over 28 third-octave bands of the balance difference (dB)
  excess_db   energy the resynthesis has where the recording is quiet (hiss, a noise floor, a ring that outlives it):
              mean dB above the original beyond 6 dB over third-octave band x 10 ms cells, looking 100 dB deep
  missing_db  the same the other way (the original has it, the resynthesis does not)
  t40_ms      (original, resynth): peak until the 5 ms envelope first falls 40 dB
  level_db    resynth minus original, loudest 50 ms RMS (dB)
  side_db     (original, resynth): side / mid energy (dB, floored at -40) over the original's active span (its onset
              to the end of its tail, fit.space.active_span); M = (L + R) / 2, S = (L - R) / 2, a mono sound has none
  width_db    per-band width error: mean |difference| of the two sounds' side / mid (dB, each floored at -40) in the
              octave bands of fit.space.side_powers over that span, weighted by the original's band level (bands
              within 40 dB of its loudest, linear in dB); 0 = the same width and spread of width over frequency
  tail_corr   (original, resynth): L/R correlation over the original's tail (fit.space.tail_span: from its loudest
              10 ms + 30 ms to the end of its tail; None when shorter than 20 ms); a reverberant space decorrelates
  tail_corr_err  |difference| of the two (0 when either is None)
  clap_cos    (clap=True) cosine of the two CLAP embeddings (1 = the same to CLAP; same-category real refs sit at
              0.4-0.95 to each other)
  describe    (describe=True) the top tag of `earshot describe` of each sound
`score(q)` folds them into one number for choosing between fits (lower is better). `floor(render_fn)` measures two
renders of one model with different seeds: a stochastic model can never beat that floor on mel_db. Before 0.4 the
score was mono only: a wide recording re-made as a near-mono one (sines renders of the films' wide dings: side / mid
-16 dB against -5, tail correlation 0.92 against 0.39) scored as well as a wide one.
Pure numpy (matplotlib only for pair_png; the CLAP / describe models only when asked).
"""
import os

import numpy as np

SR = 48000
FRAMES = (256, 1024, 4096)
N_MEL = 64


def load(x):
    """a path -> (n, 2) float64 at 48 kHz (earshot.audio.load); an array is returned as float64 unchanged"""
    if isinstance(x, (str, os.PathLike)):
        from ... import audio
        return np.asarray(audio.load(str(x)), np.float64)
    return np.asarray(x, np.float64)


def mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _db(p, floor=1e-20):
    return 10 * np.log10(np.maximum(p, floor))


def peak_rms_db(x, win_s=0.05):
    """loudest `win_s` RMS (channel-mean power) in dB"""
    x = np.asarray(x, np.float64)
    p = (x ** 2).mean(1) if x.ndim == 2 else x ** 2
    k = max(1, int(win_s * SR))
    if len(p) < k:
        return float(_db(p.mean() if len(p) else 0.0))
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(_db(((c[k:] - c[:-k]) / k).max()))


def onset_index(m, db=-30.0, hop=48):
    """first 1 ms frame within |db| of the loudest frame (sample index)"""
    nb = len(m) // hop
    if nb < 2:
        return 0
    e = _db((m[:nb * hop].reshape(nb, hop) ** 2).mean(1))
    return int(np.argmax(e > e.max() + db)) * hop


_FB = {}


def _mel_fb(n_fft, n_mel=N_MEL, f_lo=30.0, f_hi=20000.0):
    key = (n_fft, n_mel)
    if key not in _FB:
        mel = lambda f: 2595 * np.log10(1 + f / 700)
        imel = lambda m: 700 * (10 ** (m / 2595) - 1)
        pts = imel(np.linspace(mel(f_lo), mel(f_hi), n_mel + 2))
        f = np.fft.rfftfreq(n_fft, 1 / SR)
        fb = np.zeros((n_mel, len(f)))
        for i in range(n_mel):
            a, c, b = pts[i], pts[i + 1], pts[i + 2]
            fb[i] = np.clip(np.minimum((f - a) / (c - a), (b - f) / (b - c)), 0, None)
            if fb[i].sum() == 0:
                fb[i, int(np.argmin(np.abs(f - c)))] = 1.0
        _FB[key] = fb
    return _FB[key]


def melspec(m, n_fft):
    """log-mel power (dB), frames x 64, hop n_fft / 4, Hann, centred"""
    hop = n_fft // 4
    T = max(1, (len(m) + hop - 1) // hop)
    win = np.hanning(n_fft + 2)[1:-1]
    xp = np.zeros(T * hop + n_fft)
    xp[n_fft // 2:n_fft // 2 + len(m)] = m
    fr = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:T]
    P = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
    return _db(P @ _mel_fb(n_fft).T + 1e-20)


def env_db(m, ms=5.0):
    k = max(1, int(SR * ms / 1000))
    nb = max(1, len(m) // k)
    seg = m[:nb * k] if len(m) >= k else np.pad(m, (0, k - len(m)))
    return _db((seg.reshape(nb, k) ** 2).mean(1) + 1e-20)


def attack(m):
    """(attack ms, sharpness dB/ms) of the FIRST onset on a 0.125 ms envelope: the onset = the first frame within 20 dB
    of the loudest; its local peak = the loudest frame in the 10 ms after; attack = onset -> local peak; sharpness =
    17 dB / the time the rise takes from 20 dB to 3 dB under the local peak (a recording floor or a fade-in under the
    sound does not change it)"""
    fm = 0.125                                     # frame ms (6 samples); crossings interpolated inside frames
    e = env_db(m, fm)
    if len(e) < 4:
        return 0.0, 1.0
    on = int(np.argmax(e > e.max() - 20))
    p = on + int(np.argmax(e[on:on + int(10 / fm) + 1]))
    j = p
    while j > 0 and e[j] > e[p] - 20:
        j -= 1

    def cross(level):                              # first time (frames) after j the envelope reaches `level`
        k = j + int(np.argmax(e[j:p + 1] >= level))
        if k == j or e[k] == e[k - 1]:
            return float(k)
        return k - 1 + (level - e[k - 1]) / (e[k] - e[k - 1])
    rise_ms = max((cross(e[p] - 3) - cross(e[p] - 20)) * fm, 0.02)
    return float(fm * (p - on)), float(17.0 / rise_ms)


def t40(m, db=40.0):
    e = env_db(m, 5.0)
    pk = int(np.argmax(e))
    below = np.nonzero(e[pk:] < e[pk] - db)[0]
    return float(5.0 * (below[0] if len(below) else len(e) - pk))


def tone_db(a, b):
    from ..tone import profile
    return float(np.sqrt(np.mean((profile(a) - profile(b)) ** 2)))


def _align(a, b):
    """mono a, b trimmed at a's onset (5 ms lead), each levelled to its loudest 50 ms RMS, zero-padded to one length"""
    A, B = mono(a), mono(b)
    on = max(0, onset_index(A) - int(0.005 * SR))
    A, B = A[on:], B[on:]
    A = A * 10 ** (-peak_rms_db(A) / 20)
    B = B * 10 ** (-peak_rms_db(B) / 20)
    n = max(len(A), len(B))
    return np.pad(A, (0, n - len(A))), np.pad(B, (0, n - len(B)))


def mel_distance(a, b, frames=FRAMES, detail=False):
    A, B = _align(a, b)
    ds = {}
    for nf in frames:
        MA, MB = melspec(A, nf), melspec(B, nf)
        top = max(MA.max(), MB.max())
        MA, MB = np.maximum(MA, top - 80), np.maximum(MB, top - 80)
        act = np.maximum(MA, MB) > top - 60
        ds[nf] = float(np.abs(MA - MB)[act].mean()) if act.any() else 0.0
    d = float(np.mean(list(ds.values())))
    return (d, ds) if detail else d


def env_corr(a, b):
    A, B = _align(a, b)
    ea, eb = env_db(A), env_db(B)
    top = max(ea.max(), eb.max())
    ea, eb = np.maximum(ea, top - 60), np.maximum(eb, top - 60)
    act = np.maximum(ea, eb) > top - 60
    if act.sum() <= 3 or np.std(ea[act]) < 1e-9 or np.std(eb[act]) < 1e-9:
        return 0.0
    return float(np.corrcoef(ea[act], eb[act])[0, 1])


_BM = {}


def bandgram(m, nfft=1024, hop=512):
    """third-octave band power (dB) per frame: the tone.CEN bands, the three lowest merged into 22-112 Hz"""
    from ..tone import CEN
    if nfft not in _BM:
        f = np.fft.rfftfreq(nfft, 1 / SR)
        M = np.stack([(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6)) for c in CEN], 1).astype(float)
        M[:, :3] = 0.0
        M[(f >= 22) & (f < 112), 2] = 1.0
        _BM[nfft] = M
    T = max(1, (len(m) + hop - 1) // hop)
    xp = np.zeros(T * hop + nfft)
    xp[nfft // 2:nfft // 2 + len(m)] = m
    fr = np.lib.stride_tricks.sliding_window_view(xp, nfft)[::hop][:T]
    P = np.abs(np.fft.rfft(fr * np.hanning(nfft + 2)[1:-1], axis=1)) ** 2
    return _db(P @ _BM[nfft] + 1e-20)


def excess(a, b, tol=6.0, depth=100.0):
    """(excess dB, missing dB): mean over the band x frame cells within `depth` - 10 dB of the top of how far b stands
    above a (excess) or below it (missing) beyond `tol` dB, both floored `depth` dB down. Excess is energy where the
    recording is quiet: a hiss band, a noise floor, a ring that outlives the original, 60-100 dB down where the mel
    and envelope measures stop looking; it tracks CLAP's verdicts on fits closely"""
    A, B = _align(a, b)
    EA, EB = bandgram(A), bandgram(B)
    top = max(EA.max(), EB.max())
    EA, EB = np.maximum(EA, top - depth), np.maximum(EB, top - depth)
    act = np.maximum(EA, EB) > top - depth + 10
    if not act.any():
        return 0.0, 0.0
    return (float(np.maximum(0.0, EB - EA - tol)[act].mean()), float(np.maximum(0.0, EA - EB - tol)[act].mean()))


def _align2(a, b):
    """a, b as (n, 2) trimmed at a's onset (5 ms lead) and zero-padded to one length (not levelled: the stereo
    measures are ratios)"""
    from .space import _stereo
    A, B = _stereo(a), _stereo(b)
    on = max(0, onset_index(A.mean(1)) - int(0.005 * SR))
    A, B = A[on:], B[on:]
    n = max(len(A), len(B))
    return np.pad(A, ((0, n - len(A)), (0, 0))), np.pad(B, ((0, n - len(B)), (0, 0)))


def stereo(a, b, floor_db=-40.0, depth_db=40.0):
    """the stereo measures of b against its original a (module doc): {'side_db': (a, b), 'width_db', 'tail_corr':
    (a, b), 'tail_corr_err'}"""
    from .space import active_span, corr, side_powers, tail_span
    A, B = _align2(a, b)
    if not (np.ndim(a) == 2 or np.ndim(b) == 2) or len(A) < 64:
        return {'side_db': (floor_db, floor_db), 'width_db': 0.0, 'tail_corr': (None, None), 'tail_corr_err': 0.0}
    _, _, end = active_span(A)
    end = max(end, min(len(A), int(0.02 * SR)))
    side, ratio, level = [], [], []
    for X in (A, B):
        P = side_powers(X[:end])
        pm, ps = P['PM'].sum(1) + 1e-30, P['PS'].sum(1) + 1e-30
        side.append(round(float(max(floor_db, _db(ps.sum() / pm.sum()))), 2))
        ratio.append(np.clip(_db(ps / pm), floor_db, 10.0))
        level.append(_db(pm))
    w = np.clip((level[0] - level[0].max() + depth_db) / depth_db, 0.0, 1.0)
    out = {'side_db': tuple(side),
           'width_db': round(float((w * np.minimum(np.abs(ratio[0] - ratio[1]), 30.0)).sum() / max(w.sum(), 1e-9)), 3)}
    span = tail_span(A)
    ca = cb = None
    if span is not None:
        i, j = span
        ca, cb = corr(A[i:j, 0], A[i:j, 1]), corr(B[i:j, 0], B[i:j, 1])
    out['tail_corr'] = (None if ca is None else round(ca, 4), None if cb is None else round(cb, 4))
    out['tail_corr_err'] = round(abs(ca - cb), 4) if ca is not None and cb is not None else 0.0
    return out


def quality(original, resynth, clap=False, describe=False):
    """the fit-quality metrics dict of a resynthesis against its original (module doc)"""
    xa, xb = load(original), load(resynth)
    a, b = mono(xa), mono(xb)
    d, ds = mel_distance(a, b, detail=True)
    q = {'mel_db': round(d, 3)}
    q.update({f'mel_{k}': round(v, 2) for k, v in ds.items()})
    q['env_corr'] = round(env_corr(a, b), 4)
    (aa, sa), (ab, sb) = attack(a), attack(b)
    q['attack_ms'] = (round(aa, 2), round(ab, 2))
    q['sharp_db_ms'] = (round(sa, 2), round(sb, 2))
    q['onset_err'] = round(abs(float(np.log2(sb / sa))), 3)
    q['tone_db'] = round(tone_db(a, b), 3)
    ex, mi = excess(a, b)
    q['excess_db'], q['missing_db'] = round(ex, 3), round(mi, 3)
    q['t40_ms'] = (round(t40(a), 1), round(t40(b), 1))
    q['level_db'] = round(peak_rms_db(xb) - peak_rms_db(xa), 2)
    q['len_s'] = (round(len(a) / SR, 4), round(len(b) / SR, 4))
    q.update(stereo(xa, xb))
    if clap:
        q['clap_cos'] = round(clap_cos(xa, xb), 4)
    if describe:
        q['describe'] = describe_pair(xa, xb)
    return q


WIDTH_W = 0.04            # score per dB of per-band width error (capped at WIDTH_CAP_DB)
WIDTH_CAP_DB = 20.0
TAIL_CORR_W = 1.0         # score per unit of tail L/R correlation error


def score(q):
    """one number to choose between fits of one recording (lower is better): mel dB + 0.3 x tone dB
    + 8 x (1 - envelope r) + 1 x onset error (octaves, capped at 2) + 1.5 x excess dB + 1 x missing dB
    + 0.04 x width dB (capped at 20) + 1 x tail correlation error [+ 20 x (1 - CLAP cos) when measured].
    On the 12 validation refs (in-sample) the model-free part alone chose fits with a mean CLAP cos of 0.927 (the best
    candidates: 0.944); without excess / missing it chose 0.856. The stereo terms (0.4) cost a mono re-make of a wide
    recording (width 15 dB, tail correlation off by 0.6) about 1.2, what 0.06 of CLAP costs: they decide between
    candidates that sound alike, not over a clearly closer one (on 29 wide-category dev clips of the replication
    benchmark they changed no winner). Mono recordings: both are 0"""
    s = (q['mel_db'] + 0.3 * q['tone_db'] + 8.0 * (1.0 - q['env_corr']) + 1.0 * min(q['onset_err'], 2.0)
         + 1.5 * q.get('excess_db', 0.0) + 1.0 * q.get('missing_db', 0.0)
         + WIDTH_W * min(q.get('width_db', 0.0), WIDTH_CAP_DB) + TAIL_CORR_W * q.get('tail_corr_err', 0.0))
    if q.get('clap_cos') is not None:
        s += 20.0 * (1.0 - q['clap_cos'])
    return float(s)


def floor(render_fn, seeds=(11, 12), clap=False):
    """quality(render_fn(seeds[0]), render_fn(seeds[1])): what two takes of one stochastic model differ by"""
    return quality(render_fn(seeds[0]), render_fn(seeds[1]), clap=clap)


# ----------------------------------------------------------------------------------------- model-based checks
def clap_embed(arrays):
    from ...models import clap as _clap
    return _clap.get().embed_audio([mono(load(x)).astype(np.float32) for x in arrays])


def clap_cos(a, b):
    E = clap_embed([a, b])
    return float(E[0] @ E[1])


def describe_pair(a, b):
    """the top `earshot describe` tag (term, z) and AudioSet label of each sound (writes temp WAVs)"""
    import tempfile

    import soundfile as sf

    from ...describe import describe as _describe
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for key, x in (('original', a), ('resynth', b)):
            p = os.path.join(td, f'{key}.wav')
            sf.write(p, np.asarray(load(x), np.float32), SR, subtype='FLOAT')
            r = _describe(p)
            top = (r.get('top_tags') or [{}])[0]
            ast = (r.get('audioset') or [{}])[0]
            out[key] = {'tag': top.get('term'), 'z': None if top.get('z') is None else round(float(top['z']), 2),
                        'audioset': ast.get('label'), 'p': None if ast.get('p') is None else round(float(ast['p']), 2),
                        'tags3': [t.get('term') for t in (r.get('top_tags') or [])[:3]]}
    return out


def text(q, fl=None):
    """one readable line of a quality dict (and its floor)"""
    s = (f"mel {q['mel_db']:.2f} dB" + (f" (floor {fl['mel_db']:.2f})" if fl else '')
         + f", env r {q['env_corr']:.3f}, tone {q['tone_db']:.2f} dB, onset err {q['onset_err']:.2f} oct "
         f"(sharpness {q['sharp_db_ms'][0]:.1f} vs {q['sharp_db_ms'][1]:.1f} dB/ms), "
         f"excess {q.get('excess_db', 0):.2f} / missing {q.get('missing_db', 0):.2f} dB, "
         f"t40 {q['t40_ms'][0]:.0f} vs {q['t40_ms'][1]:.0f} ms, level {q['level_db']:+.2f} dB")
    sd = q.get('side_db')
    if sd and max(sd) > -40:
        s += f", side {sd[0]:.0f} vs {sd[1]:.0f} dB (width err {q.get('width_db', 0):.1f} dB)"
        tc = q.get('tail_corr') or (None, None)
        if tc[0] is not None and tc[1] is not None:
            s += f", tail corr {tc[0]:.2f} vs {tc[1]:.2f}"
    if q.get('clap_cos') is not None:
        s += f", CLAP cos {q['clap_cos']:.3f}"
    if q.get('describe'):
        d = q['describe']
        s += f"; describe: '{d['original']['tag']}' vs '{d['resynth']['tag']}'"
    return s


# -------------------------------------------------------------------------------------------------- picture
def _spec_db(m, n_fft=1024, hop=128):
    T = max(1, (len(m) + hop - 1) // hop)
    win = np.hanning(n_fft + 2)[1:-1]
    xp = np.zeros(T * hop + n_fft)
    xp[n_fft // 2:n_fft // 2 + len(m)] = m
    fr = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:T]
    return _db(np.abs(np.fft.rfft(fr * win, axis=1)) ** 2).T


def pair_png(path, original, resynth, title=None, q=None, extra=None):
    """side-by-side picture: spectrogram of the original | the resynthesis (| `extra` = (label, array)), same dB
    range, log frequency; below, both 5 ms envelopes and both third-octave balances"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..tone import CEN, profile
    a, b = mono(load(original)), mono(load(resynth))
    cols = [('original', a), ('resynthesis', b)] + ([(extra[0], mono(load(extra[1])))] if extra else [])
    n = max(len(c[1]) for c in cols)
    fig = plt.figure(figsize=(4.4 * len(cols), 6.2))
    gs = fig.add_gridspec(2, len(cols), height_ratios=[2.2, 1.0])
    S = [_spec_db(np.pad(c[1], (0, n - len(c[1])))) for c in cols]
    top = max(s.max() for s in S)
    f = np.fft.rfftfreq(1024, 1 / SR)
    t = np.arange(S[0].shape[1]) * 128 / SR
    for i, ((lab, _), s) in enumerate(zip(cols, S)):
        ax = fig.add_subplot(gs[0, i])
        ax.pcolormesh(t, np.maximum(f, 20), s, vmin=top - 90, vmax=top, cmap='magma', shading='auto')
        ax.set_yscale('log')
        ax.set_ylim(40, 22000)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel('s', fontsize=8)
        ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[1, :max(1, len(cols) - 1)])
    for lab, c in cols:
        e = env_db(np.pad(c, (0, n - len(c))))
        ax.plot(np.arange(len(e)) * 0.005, e, lw=0.9, label=lab)
    ax.set_ylim(max(-120, ax.get_ylim()[1] - 80), ax.get_ylim()[1] + 3)
    ax.set_xlabel('s', fontsize=8)
    ax.set_ylabel('dB (5 ms)', fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax2 = fig.add_subplot(gs[1, -1])
    for lab, c in cols:
        ax2.semilogx(CEN, profile(c), lw=0.9, label=lab)
    ax2.set_xlabel('Hz (1/3 octave)', fontsize=8)
    ax2.set_ylabel('dB of total', fontsize=8)
    ax2.tick_params(labelsize=7)
    if q is not None:
        title = (title + '\n' if title else '') + text(q)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
