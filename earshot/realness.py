"""Is this a real recording / professionally produced audio, or a cheap code-synthesized sound?

    earshot realness hit.wav                       # p_real, verdict, the evidence behind it
    earshot realness soundtrack.wav                # files over 6 s also get per-4 s-window verdicts (segments)
    earshot realness film.mp4 --start 20 --end 32  # one part of a file (segment times stay file times)
    earshot realness assets/ more.ogg              # several files / folders: one table row per file
    from earshot.realness import realness
    realness('pop.ogg')  # {'p_real': 0.96, 'verdict': 'real or pro-produced', 'why': ..., 'evidence': [...], 'summary': ...}

What it judges: NOT acoustic vs electronic. Real recordings AND professionally produced audio (electronic music, pro
sound design such as risers and sub drops in launch films) count as real; what it flags is amateur procedural audio
written in code: oscillators, filtered random noise, simple envelopes, algorithmic reverb, beeps.

p_real is a probability from a small transparent model (earshot/data/realness_model.json, plain JSON, no pickle):
standardised logistic heads over two evidence families, fused by a logistic layer fitted on held-out head outputs:
  clap    the CLAP embedding of the clip (what it sounds like)
  signal  measurements a synthesizer leaves behind: no noise floor, energy in a few pure spectral lines, dead-steady
          pitch, mathematically smooth envelopes, Gaussian (non-crackly) noise ... and exact digital silence, which is
          weak evidence only (edited library recordings have it too) and is never reported as a verdict reason
The fitted fusion gives the signal head weight 0 (it added no held-out accuracy), so the CLAP head decides and the
measurements are evidence in words (each with the share of training clips past that value that were of that side).
One rule overrides the model: a pure, dead-steady tone (one spectral line in every frame, crest <= 4.5 dB, no pitch
drift: a test tone or bare oscillator, outside the training data) gets p_real <= 0.03, and its evidence says so.
ast=True (--ast, slower: one AudioSet pass per clip) adds the AudioSet synth classes (Sine wave, Beep, bleep ...).
Verdict: 'real or pro-produced' (p_real >= hi), 'likely code-synthesized' (p_real <= lo), else 'uncertain'; hi / lo hold
~92% precision on held-out clips. A long file's p_real is the mean of its windows and its verdict uses file-level
thresholds read from held-out long files (few synthesized ones: weak calibration), so the same p_real can get a
different verdict as a short clip (lo 0.17) and as a long file (lo 0.40): every summary names the thresholds it used.
For a multi-take file, judge the trim you will use (--start / --end): the mean over takes and silence is not it. The windows find stretches made
entirely of code synthesis; a code-made layer under a louder real mix is usually missed, so judge stems and source files.
It hears how a sound sounds, not where it came from: clean professionally synthesized tones (UI sounds, chimes) also
read as code-synthesized. Sources quieter than -30 dBFS RMS are raised to -30 dBFS RMS first (no level cliff).
Training data, held-out accuracy and what fools it: eval/realness/RESULTS.md.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

from . import audio, config
from .cli import common
from .report import emit, fmt_t

MODEL_PATH = os.path.join(config.DATA_DIR, 'realness_model.json')
CACHE_V = 6
LONG_S, WIN_S = 6.0, 4.0
SILENT_DB = -60.0
LIFT_BELOW_DB, QUIET_MIN_PEAK_DB = -30.0, -80.0
VERDICTS = ('real or pro-produced', 'uncertain', 'likely code-synthesized', 'silent', 'error')
SR32 = 32000
REAL_PROMPTS = ['a real recording made with a microphone', 'a field recording of a real place',
                'a real foley recording of an object', 'a professionally produced music track',
                'a professional cinematic sound effect from a sound library', 'a natural acoustic sound']
SYNTH_PROMPTS = ['a cheap synthesized beep', 'a pure sine wave test tone',
                 'a simple electronic synthesizer tone made by code', 'an 8-bit video game sound effect',
                 'filtered white noise', 'a basic drum machine loop']
AST_SYNTH = ['Synthesizer', 'Sine wave', 'Beep, bleep', 'Electronic tuner', 'Chirp tone', 'Dial tone', 'Pink noise',
             'White noise', 'Static', 'Buzz', 'Hum', 'Theremin', 'Video game music', 'Ringtone', 'Sound effect']
_model = None


# ---------------------------------------------------------------------------------------------- signal measurements
def _db(v):
    return 10 * math.log10(max(float(v), 1e-30))


def _frames(y, n, hop=None):
    hop = hop or n
    k = 1 + (len(y) - n) // hop if len(y) >= n else 0
    if k <= 0:
        return np.pad(y, (0, n - len(y)))[None, :]
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return y[idx]


def audible(x):
    """RMS above SILENT_DB: quieter clips are not judged (and were not used in training)"""
    return len(x) > 0 and audio.db(math.sqrt(float(np.mean(np.square(x, dtype=np.float64))))) > SILENT_DB


def lift_quiet(x):
    """a whole sound quieter than LIFT_BELOW_DB RMS that still has content (peak over -80 dBFS, e.g. a raw orchestral
    sample) is raised to LIFT_BELOW_DB RMS (peak at most -1 dBFS): very quiet input bends the CLAP features, and a
    gain that grows smoothly as the level drops has no cliff. Returns (x, gain_db, rms_db, peak_db)."""
    if not len(x):
        return x, 0.0, -200.0, -200.0
    rdb = audio.db(math.sqrt(float(np.mean(np.square(x, dtype=np.float64)))))
    pdb = audio.db(float(np.abs(x).max()))
    g = min(LIFT_BELOW_DB - rdb, -1.0 - pdb)
    if pdb < QUIET_MIN_PEAK_DB or g <= 0:
        return x, 0.0, rdb, pdb
    return (x * 10 ** (g / 20)).astype(np.float32), g, rdb, pdb


def signal_features(x):
    """measurements of a mono 48 kHz clip that separate synthesized from recorded audio (None when silent).
    Everything but silence_frac runs on a 32 kHz copy normalised to peak 1, so the source sample rate, codec cut-off
    above 16 kHz and level do not matter."""
    from scipy import signal as sg
    m = np.asarray(audio.to_mono(np.asarray(x, np.float32)), np.float64)
    if len(m) < int(0.02 * audio.SR) or np.abs(m).max() < 1e-7:
        return None
    f = {'silence_frac': float(np.mean(np.abs(m) < 1e-6))}
    y = audio.resample(m.astype(np.float32), audio.SR, SR32).astype(np.float64)
    y /= np.abs(y).max() + 1e-12
    r = np.sqrt(np.mean(_frames(y, 640) ** 2, axis=1) + 1e-30)
    rdb = 20 * np.log10(r)
    f['floor_db'] = float(max(-140.0, np.percentile(rdb, 10) - rdb.max()))
    fr, P = sg.welch(y, SR32, nperseg=min(2048, len(y)))
    band = (fr >= 40) & (fr <= 15000)
    Pb = P[band] + 1e-30
    f['log_flatness'] = float(np.log10(np.exp(np.mean(np.log(Pb))) / np.mean(Pb)))
    f['peakiness_db'] = _db(Pb.max() / np.median(Pb))
    k = max(1, int(0.01 * len(Pb)))
    f['line_share'] = float(np.sort(Pb)[-k:].sum() / Pb.sum())
    lv = 10 * np.log10(Pb)
    f['bandwidth_khz'] = float(fr[band][np.where(lv >= lv.max() - 60)[0][-1]] / 1000)
    # short-time spectra of the active frames (within 50 dB of the loudest)
    win = np.hanning(1024)
    S = np.abs(np.fft.rfft(_frames(y, 1024, 256) * win, axis=1)) ** 2 + 1e-30
    fb = np.fft.rfftfreq(1024, 1 / SR32)
    sb = (fb >= 40) & (fb <= 15000)
    S = S[:, sb]
    e = 10 * np.log10(S.sum(axis=1))
    loud = e > e.max() - 30
    L = 10 * np.log10(S[loud])
    f['spec_floor_db'] = float(np.median(np.percentile(L, 10, axis=1) - L.max(axis=1)))
    act = e > e.max() - 50
    quiet = act & (e <= np.percentile(e[act], 25)) if act.sum() >= 8 else act
    qs = S[quiet].mean(axis=0) if quiet.any() else S.mean(axis=0)
    f['quiet_flat'] = float(np.log10(np.exp(np.mean(np.log(qs))) / np.mean(qs)))
    # pure-tone frames: the strongest peak (+-2 bins, the Hann main lobe) holds over half of the frame's energy
    i = np.clip(S.argmax(axis=1), 2, S.shape[1] - 3)
    rows = np.arange(len(S))
    share = sum(S[rows, i + o] for o in (-2, -1, 0, 1, 2)) / S.sum(axis=1)
    tonal = act & (share > 0.5)
    f['tonal_frames'] = float(tonal.sum() / max(1, act.sum()))
    f['pitch_wobble'] = float('nan')
    trip = tonal[:-2] & tonal[1:-1] & tonal[2:]
    if trip.sum() >= 4:
        a, b, c = (np.log(S[rows, i + o]) for o in (-1, 0, 1))
        cents = 1200 * np.log2(np.maximum(fb[sb][i] + 0.5 * (a - c) / (a - 2 * b + c - 1e-12) * (fb[1] - fb[0]), 20))
        d2 = np.abs(cents[2:] - 2 * cents[1:-1] + cents[:-2])[trip]
        f['pitch_wobble'] = float(np.log10(np.median(d2) + 0.01))
    # envelope smoothness: 5 ms log-energy residual around a 35 ms median trend, on active frames
    e5 = 10 * np.log10(np.mean(_frames(y, 160) ** 2, axis=1) + 1e-30)
    ok = e5 > e5.max() - 40
    if ok.sum() >= 9:
        tr = sg.medfilt(e5, 7)
        f['env_resid_db'] = float(np.log10(np.median(np.abs(e5 - tr)[ok]) + 0.01))
    else:
        f['env_resid_db'] = float('nan')
    # texture: kurtosis of the 2-8 kHz band after removing its 10 ms envelope (3 = Gaussian noise, 1.5 = a sine)
    z = sg.sosfilt(sg.butter(4, [2000, 8000], 'band', fs=SR32, output='sos'), y)
    env = np.sqrt(np.convolve(z ** 2, np.ones(320) / 320, 'same')) + 1e-9
    zz = (z / env)[np.abs(z) > 0][env[np.abs(z) > 0] > env.max() * 10 ** (-40 / 20)] if np.any(z) else z
    f['texture_kurt'] = (float(np.log(np.mean(zz ** 4) / (np.mean(zz ** 2) ** 2 + 1e-30) + 1e-9)) if len(zz) > 100
                         else float('nan'))
    ya = y[np.abs(y) > 0]
    f['crest_db'] = float(20 * np.log10(1.0 / (np.sqrt(np.mean(ya ** 2)) + 1e-12))) if len(ya) else 0.0
    return f


# ---------------------------------------------------------------------------------------------- model
def load_model(path=None):
    global _model
    if path:
        return json.load(open(path, encoding='utf-8'))
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'{MODEL_PATH} missing: train it with eval/realness/train.py')
        _model = json.load(open(MODEL_PATH, encoding='utf-8'))
    return _model


def head_logit(h, v):
    v = np.asarray(v, np.float64)
    v = np.where(np.isfinite(v), v, np.asarray(h['mean']))
    if 'clip' in h:
        v = np.clip(v, h['clip'][0], h['clip'][1])
    z = (v - np.asarray(h['mean'])) / np.asarray(h['scale'])
    return float(z @ np.asarray(h['coef']) + h['intercept']), z


def fuse(fusion, logits):
    s = fusion['intercept'] + sum(w * logits[k] for k, w in zip(fusion['heads'], fusion['coef']))
    return 1 / (1 + math.exp(-s)), s


def thresholds_of(m, file=False):
    """verdict thresholds: per clip, or for the mean of a long file's windows (file-level held-out thresholds)"""
    return m.get('file_thresholds', m['thresholds']) if file else m['thresholds']


def verdict_of(p, m, file=False):
    t = thresholds_of(m, file)
    return 'real or pro-produced' if p >= t['hi'] else 'likely code-synthesized' if p <= t['lo'] else 'uncertain'


def _sig_vector(h, f):
    return [f.get(k, float('nan')) for k in h['inputs']]


# ---------------------------------------------------------------------------------------------- evidence in words
def _pct(v):
    return f'{100 * v:.1f}%' if 0 < v < 0.095 else f'{100 * v:.0f}%'


def _phrase(name, v, toward_real):
    """one measured reason in words; toward_real = the model reads this value as evidence of a real recording"""
    if name == 'floor_db':
        return (f'quietest parts {v:.0f} dB under the peak: cleaner than a microphone usually records' if v < -95 else
                f'noise floor {v:.0f} dB under the peak, like a microphone or room')
    if name == 'spec_floor_db':
        return (f'between its partials the spectrum falls {-v:.0f} dB under the peak (no measurable mic or room noise)'
                if not toward_real else f'a broadband noise floor {-v:.0f} dB under the spectral peak, as microphones leave')
    if name == 'line_share':
        return (f'{_pct(v)} of the energy sits in a few pure spectral lines (oscillator-like)' if not toward_real else
                f'energy spread over the spectrum (top 1% of bins hold {_pct(v)})')
    if name == 'pitch_wobble':
        return ('pitch is dead steady like an oscillator (no natural drift)' if not toward_real else
                'pitch drifts naturally from frame to frame')
    if name == 'env_resid_db':
        return ('envelope is mathematically smooth (exponential / linear ramps)' if not toward_real else
                'envelope has the micro-fluctuations of a recorded source')
    if name == 'texture_kurt':
        return ('its noise is Gaussian like filtered random numbers, not the crackle of a real texture' if not toward_real
                else 'high-frequency texture is crackly / impulsive like a real source')
    if name == 'quiet_flat':
        return ('the quiet parts are tonal tails, not broadband room noise' if not toward_real else
                'the quiet parts are broadband room / mic noise')
    if name == 'tonal_frames':
        return (f'{_pct(v)} of frames are dominated by one pure tone' if not toward_real else
                f'few pure-tone frames ({_pct(v)})')
    if name == 'log_flatness':
        return ('spectrum is a few pure lines over near-silence' if not toward_real else
                'spectrum is dense and noise-rich')
    if name == 'peakiness_db':
        return (f'one spectral line stands {v:.0f} dB above the median bin' if not toward_real else
                f'no dominant spectral line ({v:.0f} dB peak/median)')
    if name == 'bandwidth_khz':
        return f'content reaches {v:.1f} kHz'
    if name == 'crest_db':
        return (f'crest factor only {v:.0f} dB: a steady, oscillator-like level' if not toward_real else
                f'crest factor {v:.0f} dB: sharp natural transients')
    return f'{name} = {v:.3g}'


def pure_tone(f):
    """a pure, dead-steady tone (test tone / bare oscillator): nearly all energy in one spectral line in every frame,
    a flat level (crest <= 4.5 dB) and no pitch drift. Out of the models' training distribution, so it is decided by
    rule; it fired on none of 4741 audible clips of the 345 real licensed recordings (eval/realness/tone_guard.json;
    closest: a wine-glass tap, tonal but crest 20 dB)"""
    w = f.get('pitch_wobble', float('nan'))
    return (f.get('tonal_frames', 0) >= 0.95 and f.get('line_share', 0) >= 0.9 and f.get('crest_db', 99) <= 4.5
            and (not np.isfinite(w) or w <= -1.0))


def _facts(m, f):
    """signal facts that fire for these measurements: (side, precision, text). Each fact is a threshold measured on the
    training clips (train.py): past it, `precision` of the clips were of that side."""
    out = []
    for q in m.get('facts', []):
        v = f.get(q['feature'], float('nan'))
        if np.isfinite(v) and (v <= q['value'] if q['op'] == '<=' else v >= q['value']):
            who = 'real/pro' if q['side'] == 'real' else 'synthesized'
            out.append((q['side'], q['precision'], f"{_phrase(q['feature'], v, q['side'] == 'real')} "
                                                   f"({100 * q['precision']:.0f}% of training clips past this were {who})"))
    return sorted(out, key=lambda t: -t[1])


def _silence_line(m, f):
    """exact digital silence in words, with how common it was in each class of the training clips: edited and gated
    library recordings have it too (56% of the real training clips contain some), so it is context, never a verdict"""
    v = f.get('silence_frac', 0.0)
    rows = [r for r in m.get('silence_base', []) if v >= r['at_least']]
    if v < 0.02 or not rows:
        return None
    r = rows[-1]
    return (f"{_pct(v)} of samples are exact digital silence (edited, gated or synthesized; weak evidence: "
            f"{_pct(r['real'])} of real/pro and {_pct(r['synth'])} of synthesized training clips have "
            f">= {_pct(r['at_least'])})")


def _clap_evidence(emb, prompt_emb, m, score):
    """the CLAP head's score in words, plus the closest of the 12 reference prompts (weak context: zero-shot prompts
    alone rank clips worse than the trained head, and their cosines are small)"""
    cos = prompt_emb @ emb
    n = len(m['prompts']['real'])
    names = m['prompts']['real'] + m['prompts']['synth']
    o = [int(k) for k in np.argsort(-cos)]
    i = o[0]
    j = next(k for k in o if (k < n) != (i < n))
    side = 'real' if i < n else 'synth'
    line = (f"CLAP head (decides p_real): score {score:+.1f} ({'toward real' if score >= 0 else 'toward synthesized'}); "
            f"closest reference prompt '{names[i]}' ({side} side, cos {cos[i]:.2f}; best on the other side "
            f"'{names[j]}', cos {cos[j]:.2f}; prompt cosines are weak context, not the decision)")
    return names[i], line


# ---------------------------------------------------------------------------------------------- scoring
def _clips(x):
    """[(t0, t1, clip)] : the whole sound up to LONG_S, else 4 s windows (the last one aligned to the end, unless that
    would add under 0.25 s of new audio)"""
    n = len(x)
    if n <= LONG_S * audio.SR:
        return [(0.0, n / audio.SR, x)]
    w = audio.windows(x, WIN_S, WIN_S, pad_last=True)
    if len(w) > 2 and w[-1][0] - w[-2][0] < 0.25:
        w = w[:-1]
    return [(t0, t0 + WIN_S, c) for t0, c in w]


def score_clips(clips, m=None, use_ast=False):
    """p_real + head logits + evidence parts for a list of mono 48 kHz arrays (silent ones -> None)"""
    from .models import clap as clap_mod
    m = m or load_model()
    rows = [None] * len(clips)
    live = []
    for i, x in enumerate(clips):
        f = signal_features(x) if audible(x) else None
        if f is not None:
            live.append(i)
            rows[i] = {'features': f}
    if not live:
        return rows
    c = clap_mod.get()
    E = c.embed_audio([clips[i] for i in live])
    P = c.embed_text(m['prompts']['real'] + m['prompts']['synth'])
    A = None
    if use_ast:
        from .models import ast as ast_mod
        t = ast_mod.get()
        A = (t, t.probs([clips[i] for i in live]))
    hs = m['heads']
    for j, i in enumerate(live):
        r = rows[i]
        lc, _ = head_logit(hs['clap'], E[j])
        ls, _ = head_logit(hs['signal'], _sig_vector(hs['signal'], r['features']))
        p, _ = fuse(m['fusion'], {'clap': lc, 'signal': ls})
        near, line = _clap_evidence(E[j], P, m, lc)
        facts = _facts(m, r['features'])
        tone = pure_tone(r['features'])
        if tone:
            p = min(p, 0.03)
            facts.insert(0, ('synth', 0.99, 'a pure, dead-steady tone: one spectral line, flat level, no pitch drift '
                                            '(a test tone or bare oscillator; real sources drift and breathe)'))
            line = (f"decided by the pure-tone rule, not the CLAP head (the CLAP head alone scores {lc:+.1f}, "
                    f"{'toward real' if lc >= 0 else 'toward synthesized'}: a bare tone is outside its training data)")
        r.update(p_real=p, logits={'clap': lc, 'signal': ls}, near=near, clap_line=line, facts=facts, tone=tone)
        if A is not None:
            t, pr = A
            top = sorted(((lab, t.prob_of(pr[j], lab)) for lab in AST_SYNTH), key=lambda v: -v[1])[:3]
            r['ast_line'] = 'AST: ' + ', '.join(f'{lab} {v:.2f}' for lab, v in top) + \
                            ' | top: ' + ', '.join(f'{lab} {v:.2f}' for lab, v in t.top(pr[j], 3))
    return rows


def _cache_path(path, start, end, use_ast, m):
    k = hashlib.sha1(f'{audio.file_id(path)}|{start}|{end}|{use_ast}|{m["version"]}|{m.get("trained", "")}|{CACHE_V}'
                     .encode()).hexdigest()[:24]
    d = os.path.join(config.EMB_CACHE, 'realness')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, k + '.json')


def _check_range(path, start, end):
    """clear errors for a reversed range or one that starts after the end of the file"""
    if start is not None and start < 0:
        raise ValueError(f'start must be >= 0 s (got {start:g})')
    s0 = start or 0.0
    if end is not None and end <= s0:
        raise ValueError(f'end ({end:g} s) must be greater than start ({s0:g} s)')
    total = audio.probe(path).get('duration') or 0
    if total and s0 >= total:
        raise ValueError(f'start {s0:g} s is at or beyond the end of {os.path.basename(path)} ({total:g} s long)')


def _spans(segs):
    """merge touching windows into 'm:ss.ss-m:ss.ss' spans"""
    out = []
    for s in segs:
        if out and s['start'] <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], s['end'])
        else:
            out.append([s['start'], s['end']])
    return ', '.join(f'{fmt_t(a)}-{fmt_t(b)}' for a, b in out[:6]) + (' ...' if len(out) > 6 else '')


def realness(path, start=None, end=None, ast=False, cache=True):
    """p_real (probability the sound is a real recording or professionally produced, not code-synthesized), verdict,
    evidence in words and, for sounds over 6 s, per-4 s-window segments (times in seconds from the file start)"""
    m = load_model()
    if start is not None or end is not None:
        _check_range(path, start, end)
    cp = _cache_path(path, start, end, ast, m) if cache else None
    if cp and os.path.exists(cp):
        try:
            with open(cp, encoding='utf-8') as fh:
                return dict(json.load(fh), path=path)  # the path as given now, not as the first caller spelled it
        except Exception:  # noqa: BLE001  (a broken cache entry is recomputed)
            pass
    off = float(start or 0.0)
    x, gain, rdb, pdb = lift_quiet(audio.load(path, mono=True, start=start, end=end))
    dur = len(x) / audio.SR
    spans = _clips(x)
    rows = score_clips([w for _, _, w in spans], m, ast)
    res = {'path': path, 'duration': round(dur, 3)}
    name = os.path.basename(path)
    if start is not None or end is not None:
        res['range'] = [round(off, 3), round(off + dur, 3)]
        name += f' [{fmt_t(off)}-{fmt_t(off + dur)}]'
    scored = [(s, r) for s, r in zip(spans, rows) if r]
    if not scored:
        peak = 'exact digital silence' if pdb <= -199 else f'peak {pdb:.0f} dBFS'
        why = f'silent (no part above {SILENT_DB:.0f} dBFS RMS; {peak}), not judged'
        res.update(p_real=None, verdict='silent', why=why, evidence=['the sound is (near) silent: nothing to judge'],
                   summary=f'{name}: {why}.')
        return res
    long_ = len(spans) > 1
    ps = np.array([r['p_real'] for _, r in scored])
    p = float(ps.mean())
    v = verdict_of(p, m, long_)
    real_side = p >= 0.5
    s0, r0 = (max if real_side else min)(scored, key=lambda sr: sr[1]['p_real'])
    side = 'real' if real_side else 'synth'
    pro = [t for sd, _, t in r0['facts'] if sd == side][:3]
    con = [t for sd, _, t in r0['facts'] if sd != side][:1]
    fu = m['fusion']
    sil = _silence_line(m, r0['features'])
    ev = ([r0['clap_line']] + pro + ['but: ' + t for t in con] + ([sil] if sil else []) +
          ([r0['ast_line']] if 'ast_line' in r0 else []))
    if gain:
        res['gain_db'] = round(gain, 1)
    if gain >= 3:
        ev.append(f'quiet source (RMS {rdb:.0f} dBFS, peak {pdb:.0f} dBFS): judged after +{gain:.0f} dB of gain (very '
                  'quiet input bends the CLAP features; level otherwise barely moves p_real, see RESULTS.md)')
    if long_:
        ev.insert(0, f"evidence below is from the most {'real' if real_side else 'synthesized'}-sounding window, "
                     f"{fmt_t(off + s0[0])}-{fmt_t(off + min(s0[1], dur))} (p_real {r0['p_real']:.2f})")
    if fu['coef'][1] != 0:
        ev.append('fusion weights: CLAP head {:.2f}, signal head {:.2f} (signal head score {:+.1f})'.format(
            *fu['coef'], r0['logits']['signal']))
    elif pro or con or sil:
        ev.append('the signal measurements above are context, not votes: the signal head added no held-out accuracy, '
                  'so its fusion weight is 0')
    t = thresholds_of(m)
    if not long_:
        lc = r0['logits']['clap']
        why = (f"pure-tone rule (the CLAP head alone: {lc:+.1f})" if r0.get('tone') else
               f"CLAP head score {lc:+.1f} ({'toward real' if lc >= 0 else 'toward synthesized'})") + \
              (f'; {pro[0]}' if pro and not r0.get('tone') else '')
        res.update(p_real=round(p, 3), verdict=v, why=why, evidence=ev,
                   summary=f"{name}: {v} (p_real {_pf(p, t)}; clip thresholds lo {t['lo']} / hi {t['hi']}): {why}.",
                   note=f"p_real is calibrated on held-out clips (lo {t['lo']} / hi {t['hi']} hold ~92% precision "
                        'there). It hears how a sound sounds, not where it came from: clean professionally synthesized '
                        'tones (UI sounds, chimes, synth stabs) also read as code-synthesized. Trained on Freesound/'
                        'Mixkit recordings, pro launch films and music vs numpy-synthesized sounds: see '
                        'eval/realness/RESULTS.md for accuracy and what fools it')
    else:
        segs = []
        for (a, b, _), r in zip(spans, rows):
            seg = {'start': round(off + a, 3), 'end': round(off + min(b, dur), 3)}
            if not r:
                seg.update(p_real=None, verdict='silent')
            else:
                sd = 'real' if r['p_real'] >= 0.5 else 'synth'
                seg.update(p_real=round(r['p_real'], 3), verdict=verdict_of(r['p_real'], m))
                fx = [q for s_, _, q in r['facts'] if s_ == sd]
                if fx:
                    seg['why'] = fx[0]
            segs.append(seg)
        live = [s for s in segs if s['p_real'] is not None]
        lean = [s for s in live if s['p_real'] < 0.5]
        flagged = [s for s in live if s['p_real'] <= t['lo']]
        ft = thresholds_of(m, True)
        mlc = float(np.mean([r['logits']['clap'] for _, r in scored]))
        tones = sum(1 for _, r in scored if r.get('tone'))
        why = (f'{len(lean)} of {len(live)} windows lean synthesized (p_real < 0.5); mean CLAP head score {mlc:+.1f}' +
               (f'; {tones} window(s) decided by the pure-tone rule' if tones else ''))
        if lean:
            tail = f' Leaning synthesized: {_spans(lean)}' + (
                f"; {len(flagged)} at or below the 4 s-window threshold {t['lo']} (code-synthesized): {_spans(flagged)}."
                if flagged else f"; none at or below the 4 s-window threshold {t['lo']}.")
        else:
            tail = (f' No window leans synthesized (lowest p_real {ps.min():.2f}); that does not rule out a code-made '
                    'layer under a louder real mix (most are missed): judge the source files and stems.')
        res.update(p_real=round(p, 3), verdict=v, why=why, evidence=ev, segments=segs, synth_windows=len(flagged),
                   lean_synth_windows=len(lean), p_real_min=round(float(ps.min()), 3),
                   summary=(f"{name}: {v} (p_real {_pf(p, ft)}, mean of {len(live)} 4 s windows; file thresholds lo "
                            f"{ft['lo']} / hi {ft['hi']}, weak calibration): {why}." + tail),
                   note=f"a long file's p_real is the mean of its window probabilities; its verdict uses file-level "
                        f"thresholds lo {ft['lo']} / hi {ft['hi']} read from held-out long files "
                        f"({ft.get('files', '?')} files, only {ft.get('synth_files', '?')} synthesized: weak "
                        f"calibration). Each window is calibrated on held-out 4 s clips (lo {t['lo']} / hi {t['hi']}). "
                        'Windows find stretches made entirely of code synthesis; a code-made layer under a louder real '
                        'mix is usually missed, so judge the source files and stems too. It hears how a sound sounds, '
                        'not where it came from (clean pro-synthesized UI tones read as code-synthesized): see '
                        'eval/realness/RESULTS.md')
    if cp:
        tmp = f'{cp}.{os.getpid()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(res, fh)
        audio.replace_quietly(tmp, cp)
    return res


def _pf(p, t):
    """p_real for text: 2 decimals, 3 when 2 would round onto a threshold it is actually below / above"""
    s = f'{p:.2f}'
    return f'{p:.3f}' if any(float(s) == v and abs(p - v) > 1e-9 for v in (t['lo'], t['hi'])) else s


def _missing(paths):
    """[(path, reason)] for given paths that name no media file"""
    return [(q, 'no such file or folder' if not os.path.exists(q) else 'no audio or video files in it')
            for q in paths if not audio.list_media([q])]


def _error_row(path, why):
    return {'path': path, 'p_real': None, 'verdict': 'error', 'why': why,
            'summary': f'{os.path.basename(path) or path}: error: {why}'}


def realness_many(paths, ast=False, start=None, end=None):
    """realness of every media file in paths (files, folders, globs); the range applies to each file. Paths that
    name nothing and files that fail to decode are 'error' rows, silent files 'silent' rows: neither counts as real."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    rows = []
    for f in audio.list_media(paths):
        try:
            rows.append(realness(f, start, end, ast=ast))
        except Exception as e:  # noqa: BLE001  (one bad file must not stop the batch)
            msg = (str(e).strip().splitlines() or [''])[0]
            rows.append(_error_row(f, f'{type(e).__name__}: {msg}'[:200]))
    rows += [_error_row(q, why) for q, why in _missing(paths)]
    n = {k: sum(r.get('verdict') == k for r in rows) for k in VERDICTS}
    parts = [f"{n['real or pro-produced']} real or pro-produced", f"{n['uncertain']} uncertain",
             f"{n['likely code-synthesized']} likely code-synthesized"]
    if n['silent']:
        parts.append(f"{n['silent']} silent (not judged)")
    if n['error']:
        parts.append(f"{n['error']} could not be read (not judged: check them)")
    rng = f" Range {fmt_t(start or 0)}-{fmt_t(end) if end is not None else 'end'} of each file." \
        if start is not None or end is not None else ''
    return {'files': rows, 'counts': n, 'summary': f'{len(rows)} files: ' + ', '.join(parts) + '.' + rng}


def _text(r):
    lines = [r['summary']]
    for e in r.get('evidence', []):
        lines.append('  - ' + e)
    for s in r.get('segments', []):
        pr = '  -  ' if s['p_real'] is None else f"{s['p_real']:.2f}"
        lean = ', leans synthesized' if s['verdict'] == 'uncertain' and s['p_real'] < 0.5 else ''
        lines.append(f"  {fmt_t(s['start'])}-{fmt_t(s['end'])}  p_real {pr}  {s['verdict']}{lean}" +
                     (f"  ({s['why']})" if s.get('why') and s['verdict'] != 'real or pro-produced' else ''))
    if r.get('note'):
        lines.append('  note: ' + r['note'])
    return '\n'.join(lines)


def _table(res):
    try:
        m = load_model()
        t, ft = thresholds_of(m), thresholds_of(m, True)
        rule = (f"Verdict thresholds: sounds up to {LONG_S:.0f} s lo {t['lo']} / hi {t['hi']}; longer files (mean of 4 s "
                f"windows) lo {ft['lo']} / hi {ft['hi']}, weakly calibrated. [n low windows] = 4 s windows at or below "
                f"{t['lo']}. For a multi-take file judge the trim you will use (--start / --end).")
    except Exception:  # noqa: BLE001
        rule = ''
    lines = [res['summary']] + ([rule] if rule else []) + [f"{'p_real':>6}  {'verdict':<24} file: main reason"]
    for r in res['files']:
        p = '  -  ' if r.get('p_real') is None else f"{r['p_real']:.2f}"
        extra = f" [{r['synth_windows']} low window(s)]" if r.get('synth_windows') else ''
        long_ = ' (file)' if r.get('segments') else ''
        lines.append(f"{p:>6}  {r.get('verdict', '') + long_:<24} {os.path.basename(r['path']) or r['path']}{extra}: "
                     f"{r.get('why') or r.get('summary', '')}")
    return '\n'.join(lines)


def add_cli(sub):
    p = sub.add_parser('realness', help='real recording / pro-produced audio vs code-synthesized sound')
    p.add_argument('paths', nargs='+', help='files or folders (audio or video)')
    p.add_argument('--start', type=float, help='judge only from this time (seconds from the file start; with several '
                                               'files it applies to each)')
    p.add_argument('--end', type=float, help='judge only up to this time (seconds from the file start)')
    p.add_argument('--ast', action='store_true', help='also run the AudioSet tagger for evidence (slower)')
    common(p)

    def run(a):
        bad = _missing(a.paths)
        if bad:
            for q, why in bad:
                print(f'earshot realness: error: {q}: {why}', file=sys.stderr)
            return 1
        files = audio.list_media(a.paths)
        if len(files) == 1:
            r = realness(files[0], a.start, a.end, a.ast)
            emit(r, a, _text(r))
            return 0
        r = realness_many(files, a.ast, a.start, a.end)
        emit(r, a, _table(r))
        return 1 if r['counts']['error'] else 0
    p.set_defaults(func=run)


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
