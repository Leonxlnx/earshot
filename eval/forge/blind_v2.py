"""Clean blind test (forge v2 finals vs held-out REAL recordings) + the v1 -> v2 progress listening file.

    python eval/forge/blind_v2.py --out out/showcase/blind_v2 --seed 2026
    python eval/forge/blind_v2.py --dry-run                         (cleanliness only, writes nothing)
    python out/showcase/blind_v2/score_answers.py A B B A ...      (after listening)

One command rebuilds both from whatever is in out/forge_v2/<category>/final/ now (rerun it after a polish round).
Nothing is synthesised here: every sound is a file that already exists (forge finals, v1 finals, licensed bench refs),
only trimmed, faded, gain-changed and placed on silence. No markers, beeps or voice: trials are separated by silence.

BLIND TEST (--out, default out/showcase/blind_v2)
  The rule: nothing but the sound itself may give the answer away. A real recording carries a room-noise floor and
  a made sound sits on digital silence, so neither may play any floor before or after its sound; and the two must
  play equally loud at their loudest moment.
  Trials: N (default 20) over the licensed real categories that have bench refs (PLAN: two trials for the strongest
  categories of the round-3 verdicts, one for the others, including the weak ones so the test is honest). Each trial
  pairs one forge v2 final (<category>_NN.wav in out/forge_v2/<category>/final/) with one REAL recording of the same
  category from the held-out BENCH split (earshot.forge.refsets: source recordings no forge search ever used).
  Picks: the made final is drawn at random (seeded) among the category's length-compatible finals (audible extent
  within 1.3x of an available bench ref; if too few, the closest in length), never by a score; the real one is the
  bench ref whose audible extent is closest to it, with no real file and, where possible, no source recording used
  twice in the test (whoosh_fast / swish and notify / logo_sting share bench files). Both picks prefer files that trim
  cleanly, so the common trim levels below stay low: a cut file starts 30 dB under its loudest frame or with a strike
  (not inside a sustained sound), the sound decays 30 dB before the file's own end fade (or builds to its end: a
  riser), a measured recording floor at most -50 dB, no anti-phase stereo; the category with the fewest clean refs per
  trial picks first. Duration and these edge facts are all a pick looks at, never a score.
  Trim, the same way for both sounds of a pair. Levels are dB re each sound's own loudest envelope frame
  (channel-summed power; 2.5 ms frames smoothed over 7.5 ms in short pairs, 5 ms over 30 ms in long ones).
    end level   L_end = the highest of -50 dB, where each file's sound stands in the 15 ms before the file's own end
                fade + 3 dB, and each measured recording floor + 10 dB (at most -6 dB): a level both decays reach
                naturally, above any floor;
    onset level L_on  = the highest of -50 dB, each bench ref's level 2-5 ms after its start (just after its own 2 ms
                fade-in: a cut into the sound shows here; made finals are rendered from digital silence) + 1 dB, and
                each floor + 10 dB (at most -6 dB).
    Each clip starts 4 ms before its first 1 ms frame at L_on, faded in over those 4 ms (raised cosine; only sound
    below L_on is under the fade; zero-padded when the file has no 4 ms lead), and ends where its envelope last stands
    at L_end (before the file's own end fade), with a matched fade-out over the last 6 dB of its decay above L_end (at
    least 5 ms). So neither clip carries floor before or after its sound, and both stop at the same point of their
    decay. A pair that both build to their end (risers: loudest frame in the last 15 %) keep their files' own
    endings. The floor of a real ref is measured in its SOURCE recording (out/refs/manifest.json: 5th percentile of
    50 ms windows outside digital silence, within 20 s of the trim, at the trim's gain); made sounds have none.
  Level: a pair is short (loudest 100 ms) or long (momentary, 400 ms) by its longer audible extent (1 s); both clips
    are measured the same way. The made clip is matched to the real one on that pair measure (K-weighted, BS.1770),
    then one common gain puts it at -15 LUFS (short) / -16 LUFS (long) unless a true peak (4x oversampled) would pass
    -1 dBTP: then both go down together.
  Order: trials shuffled (no category twice in a row), real-as-A on exactly half the trials, both from --seed.
  Layout: 1 s lead, A, 1.2 s silence, B, 2.5 s silence, next trial ... 1 s tail; 48 kHz, 24-bit stereo. Gaps are exact
    digital silence.
  Cleanliness: every residual difference a listener could use besides quality is measured on the clips as they play:
    loudest-moment difference (the pair measure, plus the other window and earshot's own momentary meter), true-peak
    difference, pre-roll length and level, the recording floor as played, its margin under the onset / end levels and
    how long the sound dips to within 6 dB of it inside a clip, playback lengths, the level of what each fade-in ramps
    and of what each trim dropped after the end (onset / ending sharpness from trimming), fade lengths, stereo
    correlation, silent gaps, and the loudest moment and true peak again from the written 24-bit trial files. Hard
    limits (FAIL): pair measure > 0.5 LU, true peak > -1 dBTP, pre-roll > 5 ms or holding sound at the onset level,
    floor margin < 6 dB, floor heard > 100 ms inside a clip above -70 dBFS, trim edges that differ by > 10 dB with the
    louder above -20 dB, non-silent gaps. FAIL stops the build before anything is written (the folder keeps its
    previous build) unless --allow-fail. Warnings (listed, not fatal): length ratio > 1.6, true-peak difference
    > 6 dB, other window > 2 LU, a floor above -60 dBFS as played or within 6 dB for > 30 ms, one clip mono and the
    other wide, anti-phase stereo.
  Writes: blind_test.wav, answer_sheet.txt (trial, category, A start, B start, blank; one cleanliness line),
  cleanliness.json (per trial only differences and maxima, never which side: safe to open before listening),
  KEY_open_after.json (the answers, picks, trims, gains, per-side levels), score_answers.py (per trial right / wrong,
  per category, per tier, overall, one-sided binomial p), blind_test_trials/tNN_A.wav + tNN_B.wav (each trial on its
  own, same levels, 0.25 s of silence around each clip). The key stores the md5 of every file used. If a final or
  bench ref changes while this runs (a polish round writing finals), it stops before writing anything.

PROGRESS (--progress, default <out>/../progress)
  v1_vs_v2.wav: the five sounds v1 had (key press, UI click, pop, fast whoosh, impact): v1 final, v2 final, v1 final,
  v2 final (v1 = demo/forge-test/<sound>/final/<sound>_01 and _02; v2 = the v2 final closest to that v1 final in
  1/3-octave balance and audible length, so like is played against like). The four sounds of a group play equally
  loud at their loudest moment (the blind test's measure: K-weighted loudest 100 ms, or momentary 400 ms for a group
  with a sound of 1 s or more) at -15 / -16 LUFS, and one common gain per group keeps every true peak at or under
  -1 dBTP. progress.txt lists every start time and level. progress.md: the measured before / after numbers from
  eval/forge/results_v1.json against the newest eval/forge/results_v2r<N>.json (both from the same frozen held-out
  benchmark, eval/forge/bench.py).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE)).replace(chr(92), '/')
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import frozen_tone  # noqa: E402  (1/3-octave profile, the benchmark's own frozen copy)
from earshot import features  # noqa: E402  (BS.1770 K-weighting, momentary meter, true peak)
from earshot.forge import refsets  # noqa: E402

SR = 48000
MADE = f'{REPO}/out/forge_v2'
MANIFEST = f'{REPO}/out/refs/manifest.json'
LEAD, GAP, BETWEEN, TAIL = 1.0, 1.2, 2.5, 1.0
COMPAT = 1.3                 # made finals whose audible extent is within this factor of an available bench ref
# trim (see the module doc)
ON_DB, END_DB = -50.0, -50.0
REACH_MARGIN, FLOOR_MARGIN, HEAD_MARGIN = 3.0, 10.0, 1.0
LEVEL_CAP = -6.0
FADE_IN_S = 0.004
FADE_DB, FADE_MIN_S = 6.0, 0.005
REF_FADE_DEFAULT, MADE_FADE_GUARD = 0.03, 0.01
ENV = {'short': (0.0025, 3), 'long': (0.005, 6)}          # envelope hop (s), smoothing (frames)
CLEAN_HEAD, CLEAN_REACH, CLEAN_FLOOR = -30.0, -30.0, -50.0  # pick preference: quiet start, natural decay, low floor
# level
SHORT_S = 1.0
WIN = {'short': 0.1, 'long': 0.4}
TARGET = {'short': -15.0, 'long': -16.0}
TP_MAX = -1.0
# cleanliness limits
FAIL_AT = {'loud_lu': 0.5, 'true_peak_dbtp': TP_MAX + 0.05, 'floor_margin_db': 6.0, 'preroll_ms': 5.0,
           'edge_diff_db': 10.0, 'edge_loud_db': -20.0, 'floor_exposed_ms': 100.0, 'floor_exposed_dbfs': -70.0}
WARN_AT = {'length_ratio': 1.6, 'true_peak_diff_db': 6.0, 'other_window_lu': 2.0, 'mono_corr': 0.995,
           'wide_corr': 0.9, 'floor_dbfs': -60.0, 'floor_exposed_ms': 30.0}

# (category, answer-sheet name, trials at N = 20, tier from the round-3 verdicts in docs/history/FORGE_V2_LOG.md / results_v2r3.md;
#  round 4 moved toggle from weak to middle: tone 10.4 -> 7.8 dB, curve 14.6 -> 10.1 dB, ring lines gone, v2r4.md)
PLAN = [
    ('whoosh_big', 'big whoosh', 2, 'strong'),
    ('whoosh_fast', 'fast whoosh', 2, 'strong'),
    ('click', 'UI click', 2, 'strong'),
    ('typing_key', 'key press', 2, 'strong'),
    ('shimmer', 'shimmer', 2, 'strong'),
    ('pop', 'pop', 2, 'strong'),
    ('swipe', 'cloth swipe', 1, 'strong'),
    ('swish', 'swish', 1, 'strong'),
    ('impact', 'impact', 1, 'strong'),
    ('hover_tick', 'hover tick', 1, 'middle'),
    ('tap', 'tap', 1, 'weak'),
    ('toggle', 'toggle', 1, 'middle'),
    ('notify', 'notification ding', 1, 'weak'),
    ('riser', 'riser', 1, 'weak'),
]
EXCLUDED = {
    'logo_sting': 'its bench refs are a triangle hit and a struck metal ring (the very files of the notify bench): the '
                  'library has no real logo sting, so a pair would test the form (a composed reveal vs one ring), not '
                  'realism',
    'downlifter': 'no real or designed downlifter in the library (no bench refs)',
    'glitch': 'no glitch material in the library (no bench refs)',
    'sub_drop': 'one bench ref, a designed tom pitched down 24 semitones: not a recording of a real event',
    'pro_*': 'the pro refs are trims of launch films, most under music / voice beds: no clean A/B',
}
NOTES = {
    'impact': 'the only impact bench ref is a library cinematic hit (designed from recordings by a person)',
    'riser': 'the riser bench refs are real cymbal swells; the made risers are designed (synth / noise) risers',
    'swish': 'the swish bench refs are the same files as three whoosh_fast bench refs (used once each here)',
}
V1 = [   # (answer name, v2 category, v1 final folder, v1 file stem)
    ('key press', 'typing_key', 'demo/forge-test/keypress/final', 'keypress'),
    ('UI click', 'click', 'demo/forge-test/click/final', 'click'),
    ('pop', 'pop', 'demo/forge-test/pop/final', 'pop'),
    ('fast whoosh', 'whoosh_fast', 'demo/forge-test/whoosh/final', 'whoosh'),
    ('impact', 'impact', 'demo/forge-test/impact/final', 'impact'),
]


# ----------------------------------------------------------------------------------------------------------- audio
def rel(p):
    p = os.path.abspath(p).replace(chr(92), '/')
    return p[len(REPO) + 1:] if p.lower().startswith(REPO.lower() + '/') else p


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def load(p, start=None, stop=None):
    if start is None:
        x, sr = sf.read(p, always_2d=True, dtype='float64')
    else:
        sr = sf.info(p).samplerate
        x, sr = sf.read(p, start=int(start * sr), stop=int(stop * sr), always_2d=True, dtype='float64')
    if sr != SR:
        from math import gcd
        from scipy import signal
        g = gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, int(sr) // g, axis=0)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2]


def frames_db(x, ms):
    p = (x ** 2).mean(axis=1)
    n = max(1, int(SR * ms / 1000))
    k = len(p) // n
    if k < 1:
        return np.array([10 * np.log10(p.mean() + 1e-20)]), n
    return 10 * np.log10(p[:k * n].reshape(k, n).mean(axis=1) + 1e-20), n


def onset(x, below=40.0):
    e, n = frames_db(x, 1.0)
    return int(np.nonzero(e >= e.max() - below)[0][0] * n)


def end_of(x, below=50.0):
    e, n = frames_db(x, 5.0)
    return int((np.nonzero(e >= e.max() - below)[0][-1] + 1) * n)


def extent_s(x):
    """audible extent: first 1 ms frame within 40 dB of the loudest to the last 5 ms frame within 50 dB (picks only)"""
    return (end_of(x) - onset(x)) / SR


def raised(n):
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(1, n))


def power_frames(x, n, smooth=1):
    """channel-summed mean power per n-sample frame (the last partial frame zero-padded), optionally smoothed"""
    p = (x ** 2).sum(axis=1)
    k = -(-len(p) // n)
    p = np.concatenate([p, np.zeros(k * n - len(p))]).reshape(k, n).mean(axis=1)
    if smooth > 1 and k > 1:
        p = np.convolve(p, np.ones(smooth) / smooth, mode='same')
    return p


def db10(p):
    return 10 * np.log10(np.asarray(p, dtype='float64') + 1e-20)


def loud_max(x, win):
    """the loudest `win`-second window, K-weighted (BS.1770), LUFS scale (-0.691 + channel-summed mean square);
    every sample position; a sound shorter than the window counts over the whole window (silence after it)"""
    pad = np.zeros((int(0.05 * SR), 2))
    y = features._kw(np.concatenate([pad, x, pad]))
    p = (y ** 2).sum(axis=1)
    w = int(round(win * SR))
    if len(p) < w:
        p = np.concatenate([p, np.zeros(w - len(p))])
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(-0.691 + 10 * np.log10(((c[w:] - c[:-w]) / w).max() + 1e-20))


def meter_m400(x):
    """earshot's own momentary meter (features.loudness_curve, 400 ms, 10 ms hop), padded to 400 ms: a cross-check"""
    n = int(0.4 * SR)
    xx = x if len(x) >= n else np.concatenate([x, np.zeros((n - len(x), 2))])
    return float(np.max(features.loudness_curve(xx, 0.4, 0.01)[1]))


def true_peak(x):
    return float(features.true_peak_dbtp(x))


def rms_dbfs(x):
    return float(10 * np.log10(np.mean(x ** 2) + 1e-20)) if len(x) else -200.0


def stereo_corr(x):
    a, b = x[:, 0], x[:, 1]
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


# ------------------------------------------------------------------------------------------------ file facts, trim
_MANIFEST = None


def manifest_entry(p):
    """the refset manifest's record of a bench ref (source path, trim, fade, gain), or None"""
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = {}
        try:
            m = json.load(open(MANIFEST, encoding='utf-8'))
            for c in m.values():
                for split in ('forge', 'bench'):
                    for e in (c.get(split) or []) if isinstance(c, dict) else []:
                        if isinstance(e, dict) and e.get('file'):
                            _MANIFEST[norm(e['file'])] = e
        except (OSError, ValueError):
            pass
    return _MANIFEST.get(norm(p))


_FLOOR = {}


def source_floor(p):
    """recording floor of a bench ref (channel-summed mean power at the ref's own scale), measured in its source
    recording: 5th percentile of 50 ms windows above digital silence (-100 dB), within 20 s of the trim, at the trim's
    gain. None when the source is not on this machine (or p is not a bench ref)."""
    if p in _FLOOR:
        return _FLOOR[p]
    e, out = manifest_entry(p), None
    if e and e.get('source_path') and os.path.exists(e['source_path']):
        try:
            a, b = e.get('trim_s') or [0.0, 0.0]
            dur = sf.info(e['source_path']).duration
            x = load(e['source_path'], max(0.0, a - 20.0), min(dur, b + 20.0))
            w = power_frames(x, int(0.05 * SR))
            w = w[w > 2e-10]
            if len(w):
                out = float(np.percentile(w, 5)) * 10 ** (float(e.get('gain_db') or 0.0) / 10)
        except (RuntimeError, ValueError, OSError):
            out = None
    _FLOOR[p] = out
    return out


def end_fade_s(p):
    """length of the file's own end fade (bench refs: from the manifest; made finals: a short guard)"""
    e = manifest_entry(p)
    if e is not None:
        return float(e.get('fade_out_s') or REF_FADE_DEFAULT)
    return MADE_FADE_GUARD


def analyse(x, cls, fade_s, floor_pow=None, cut_file=True):
    """the facts a trim needs, levels in dB re the loudest envelope frame (`cls` envelope). cut_file: the file was cut
    from a longer recording (a bench ref, after its own 2 ms fade-in): its level 2-5 ms in says whether the cut went
    into the sound. Made finals are rendered from digital silence (head None)."""
    hop, sm = ENV[cls]
    n = int(round(hop * SR))
    p = power_frames(x, n, sm)
    top = float(db10(p.max()))
    e = db10(p) - top
    e1 = db10(power_frames(x, SR // 1000)) - top           # 1 ms frames, same reference
    ip = int(np.argmax(e))
    guard = int(np.ceil((fade_s + 0.005) * SR / n))
    stop = max(ip + 1, len(e) - guard)
    last = int(np.ceil(0.015 * SR / n))
    return {'e': e, 'n': n, 'e1': e1, 'top': top, 'ip': ip, 'stop': stop,
            'reach': float(e[max(ip, stop - last):stop].min()),     # where the sound stands as the file's fade begins
            'head': (float(e1[2:5].max()) if len(e1) > 5 else float(e1.max())) if cut_file else None,
            'peak_late': ip * n >= 0.85 * len(x),
            'strike': float(e1[:15].max()) >= -6.0,          # loud within its first 15 ms: a struck sound
            'floor': None if floor_pow is None else float(db10(floor_pow)) - top}


def pair_levels(ia, ib):
    """(L_on, L_end): the common onset and end levels of a pair, dB re each sound's loudest frame"""
    floors = [i['floor'] + FLOOR_MARGIN for i in (ia, ib) if i['floor'] is not None]
    l_end = max([END_DB] + [i['reach'] + REACH_MARGIN for i in (ia, ib)] + floors)
    l_on = max([ON_DB] + [i['head'] + HEAD_MARGIN for i in (ia, ib) if i['head'] is not None] + floors)
    return min(l_on, LEVEL_CAP), min(l_end, LEVEL_CAP)


def cut(x, info, l_on, l_end, keep_end=False):
    """one clip: 4 ms fade-in up to the onset (first 1 ms frame at l_on), to where the envelope last stands at l_end
    (before the file's end fade), fade-out over the last FADE_DB of the decay. keep_end (a pair that both build to
    their end, risers): the clip keeps the file's own ending (2 ms safety fade). Returns (clip, facts)."""
    e, n, e1 = info['e'], info['n'], info['e1']
    hit = np.nonzero(e1 >= l_on)[0]
    on = int(hit[0]) * (SR // 1000) if len(hit) else 0
    fi = int(round(FADE_IN_S * SR))
    start = on - fi
    above = np.nonzero(e[:info['stop']] >= l_end)[0]
    j = int(above[-1]) if len(above) else info['ip']
    end = int(min(len(x), (j + 1) * n))
    hi = np.nonzero(e[:j + 1] >= l_end + FADE_DB)[0]
    f0 = int((hi[-1] + 1) * n) if len(hi) else int((info['ip'] + 1) * n)
    f0 = max(on, min(f0, end - int(FADE_MIN_S * SR)))
    keep_end = bool(keep_end and e[info['stop'] - 1] >= l_end)
    if keep_end:        # still loud where the file's own end fade begins: the file's own ending stays
        end = len(x)
        j = len(e) - 1
        f0 = end - int(0.002 * SR)
    y = x[max(0, start):end].copy()
    if start < 0:
        y = np.concatenate([np.zeros((-start, 2)), y])
    y[:fi] *= raised(fi)[:, None]
    nf = end - f0
    y[len(y) - nf:] *= raised(nf)[::-1][:, None]
    # trim edges (dB re the loudest frame): the loudest 1 ms in the first half of the fade-in (what it audibly ramps:
    # the file before the onset, silence where the clip starts before the file) and the loudest of what the trim
    # dropped in the 5 ms after the end (at the file's end: its last frame)
    k1 = SR // 1000
    pre = e1[max(0, start) // k1:max(0, start + fi // 2) // k1]
    cut_in = float(pre.max()) if len(pre) else -200.0
    m = int(np.ceil(0.005 * SR / n))
    cut_out = float(e[j + 1:j + 1 + m].max()) if j + 1 < len(e) else float(e[-1])
    # floor heard inside the clip: unfaded envelope frames within 6 dB of the recording floor (dips between events)
    exposed = None
    if info['floor'] is not None:
        exposed = round(float((e[on // n:max(on // n, f0 // n)] < info['floor'] + 6.0).sum()) * n / SR * 1000, 1)
    return y, {'onset_s': round(on / SR, 4), 'start_s': round(start / SR, 4), 'end_s': round(end / SR, 4),
               'fade_out_ms': round(nf / SR * 1000, 1), 'cut_in_db': round(cut_in, 1),
               'cut_out_db': round(cut_out, 1), 'reach_db': round(info['reach'], 1),
               'head_db': None if info['head'] is None else round(info['head'], 1),
               'floor_db': None if info['floor'] is None else round(info['floor'], 1),
               'floor_exposed_ms': exposed, 'kept_file_end': keep_end, 'end_fade_s': None}


def pair_class(xa, xb):
    return 'long' if max(extent_s(xa), extent_s(xb)) >= SHORT_S else 'short'


def trim_pair(real0, made0, real_path, made_path):
    """both clips of a pair (see the module doc); returns (real, made, cls, l_on, l_end, facts real, facts made)"""
    cls = pair_class(real0, made0)
    ir = analyse(real0, cls, end_fade_s(real_path), source_floor(real_path))
    im = analyse(made0, cls, end_fade_s(made_path), None, cut_file=False)
    l_on, l_end = pair_levels(ir, im)
    keep = ir['peak_late'] and im['peak_late']          # both build to their end (risers): both keep their endings
    real, fr = cut(real0, ir, l_on, l_end, keep)
    made, fm = cut(made0, im, l_on, l_end, keep)
    fr['end_fade_s'], fm['end_fade_s'] = end_fade_s(real_path), end_fade_s(made_path)
    fr['top_db'], fm['top_db'] = round(ir['top'], 3), round(im['top'], 3)
    return real, made, cls, l_on, l_end, fr, fm


def match_pair(real, made, cls):
    """(made gain dB, common gain dB, cap hit dB): the made clip to the real one on the pair measure, then both to the
    target, lowered together if a true peak would pass TP_MAX"""
    g = loud_max(real, WIN[cls]) - loud_max(made, WIN[cls])
    gc = TARGET[cls] - loud_max(real, WIN[cls])
    tp = max(true_peak(real), true_peak(made) + g) + gc
    over = max(0.0, tp - TP_MAX)
    return g, gc - over, over


def clip_levels(y, cls, pre_n):
    other = 'long' if cls == 'short' else 'short'
    return {'loud_pair_measure': round(loud_max(y, WIN[cls]), 3),
            'loud_other_window': round(loud_max(y, WIN[other]), 3),
            'meter_m400': round(meter_m400(y), 3), 'true_peak_dbtp': round(true_peak(y), 3),
            'sample_peak_dbfs': round(float(20 * np.log10(np.abs(y).max() + 1e-12)), 3),
            'preroll_dbfs': round(rms_dbfs(y[:pre_n]), 1), 'length_s': round(len(y) / SR, 4),
            'stereo_corr': round(stereo_corr(y), 3)}


def cleanliness(lr, lm, fr, fm, cls, l_on, l_end, gain_r_db):
    """(per-side facts, symmetric row, fails, warnings) for one pair as played. gain_r_db: the gain the real clip got
    (its floor as played = floor re its loudest frame + its loudest frame + this gain)"""
    fails, warns = [], []
    d_pair = abs(lr['loud_pair_measure'] - lm['loud_pair_measure'])
    d_other = abs(lr['loud_other_window'] - lm['loud_other_window'])
    d_meter = abs(lr['meter_m400'] - lm['meter_m400'])
    d_tp = abs(lr['true_peak_dbtp'] - lm['true_peak_dbtp'])
    tp_max = max(lr['true_peak_dbtp'], lm['true_peak_dbtp'])
    pre_max = max(lr['preroll_dbfs'], lm['preroll_dbfs'])
    floor_played = margin = None
    if fr['floor_db'] is not None:
        floor_played = round(fr['floor_db'] + fr['top_db'] + gain_r_db - 10 * np.log10(2), 1)   # dBFS, per channel
        margin = round(min(l_end, l_on) - fr['floor_db'], 1)          # under the levels where clips start / end
    pre_ms = max(fr['onset_s'] - fr['start_s'], fm['onset_s'] - fm['start_s']) * 1000
    lens = sorted([lr['length_s'], lm['length_s']])
    ratio = lens[1] / max(lens[0], 1e-6)
    d_in = abs(fr['cut_in_db'] - fm['cut_in_db'])
    d_out = abs(fr['cut_out_db'] - fm['cut_out_db'])
    if d_pair > FAIL_AT['loud_lu']:
        fails.append(f'loudest-moment difference {d_pair:.2f} LU > {FAIL_AT["loud_lu"]}')
    if tp_max > FAIL_AT['true_peak_dbtp']:
        fails.append(f'true peak {tp_max:.2f} dBTP > {TP_MAX}')
    if pre_ms > FAIL_AT['preroll_ms']:
        fails.append(f'pre-roll {pre_ms:.1f} ms > {FAIL_AT["preroll_ms"]} ms')
    if max(fr['cut_in_db'], fm['cut_in_db']) >= l_on:
        fails.append('the pre-roll holds sound at or above the onset level')
    if margin is not None and margin < FAIL_AT['floor_margin_db']:
        fails.append(f'recording floor only {margin:.1f} dB under the onset / end level')
    exp = fr['floor_exposed_ms'] or 0.0
    if exp > FAIL_AT['floor_exposed_ms'] and floor_played > FAIL_AT['floor_exposed_dbfs']:
        fails.append(f'recording floor heard for {exp:.0f} ms inside the clip at {floor_played:.0f} dBFS')
    elif exp > WARN_AT['floor_exposed_ms']:
        warns.append(f'the sound dips to within 6 dB of a recording floor for {exp:.0f} ms inside a clip '
                     f'(floor {floor_played:.0f} dBFS as played)')
    if d_in > FAIL_AT['edge_diff_db'] and max(fr['cut_in_db'], fm['cut_in_db']) > FAIL_AT['edge_loud_db']:
        fails.append(f'onset trims differ by {d_in:.1f} dB (louder edge '
                     f'{max(fr["cut_in_db"], fm["cut_in_db"]):.1f} dB)')
    if d_out > FAIL_AT['edge_diff_db'] and max(fr['cut_out_db'], fm['cut_out_db']) > FAIL_AT['edge_loud_db']:
        fails.append(f'end trims differ by {d_out:.1f} dB (louder edge '
                     f'{max(fr["cut_out_db"], fm["cut_out_db"]):.1f} dB)')
    if ratio > WARN_AT['length_ratio']:
        warns.append(f'lengths {lens[0]:.2f} / {lens[1]:.2f} s (x{ratio:.2f})')
    if d_tp > WARN_AT['true_peak_diff_db']:
        warns.append(f'true peaks differ by {d_tp:.1f} dB (crest of the sounds themselves)')
    if d_other > WARN_AT['other_window_lu']:
        warns.append(f'the other loudness window differs by {d_other:.1f} LU')
    if floor_played is not None and floor_played > WARN_AT['floor_dbfs']:
        warns.append(f'a recording floor at {floor_played:.0f} dBFS as played (under the sound, {margin:.0f} dB below '
                     'where the clips start / end)')
    cs = sorted([lr['stereo_corr'], lm['stereo_corr']])
    if cs[1] >= WARN_AT['mono_corr'] and cs[0] < WARN_AT['wide_corr']:
        warns.append(f'one clip mono, the other wide (stereo corr {cs[0]:.2f} / {cs[1]:.2f})')
    if cs[0] < 0:
        warns.append(f'a clip has anti-phase stereo (corr {cs[0]:.2f})')
    row = {'class': cls, 'matched_on': f'K-weighted loudest {int(WIN[cls] * 1000)} ms',
           'loudest_moment_diff_lu': round(d_pair, 3), 'other_window_diff_lu': round(d_other, 2),
           'meter_m400_diff_lu': round(d_meter, 2), 'true_peak_diff_db': round(d_tp, 2),
           'true_peak_max_dbtp': round(tp_max, 2), 'preroll_max_dbfs': round(pre_max, 1),
           'preroll_ms': round(pre_ms, 1),
           'floor_as_played_dbfs': floor_played, 'floor_margin_db': margin, 'floor_exposed_ms': fr['floor_exposed_ms'],
           'onset_level_db': round(l_on, 1), 'end_level_db': round(l_end, 1),
           'length_s': lens, 'length_ratio': round(ratio, 3),
           'onset_trim_levels_db': sorted([fr['cut_in_db'], fm['cut_in_db']]),
           'end_trim_levels_db': sorted([fr['cut_out_db'], fm['cut_out_db']]),
           'fade_out_ms': sorted([fr['fade_out_ms'], fm['fade_out_ms']]), 'stereo_corr': cs}
    return row, fails, warns


def write(path, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert np.isfinite(y).all(), path
    sf.write(path, y, SR, subtype='PCM_24')


def place(parts, total_s):
    y = np.zeros((int(round(total_s * SR)) + 1, 2))
    for t0, x in parts:
        i = int(round(t0 * SR))
        y[i:i + len(x)] += x
    return y


# ------------------------------------------------------------------------------------------------------ the picks
def finals(cat):
    return sorted(glob.glob(f'{MADE}/{cat}/final/{cat}_[0-9][0-9].wav'))


def source_of(p):
    """the source recording of a bench trim: 'fs-463763_2.wav' -> 'fs-463763' (sister trims share a source)"""
    return re.sub(r'_\d+$', '', os.path.splitext(os.path.basename(p))[0])


def allocate(n):
    alloc = {c: 0 for c, *_ in PLAN}
    left = n
    for want in (1, 2):
        for c, _, k, _ in PLAN:
            if left and alloc[c] < min(want, k):
                alloc[c] += 1
                left -= 1
    while left:                                   # more than the plan: round-robin over the plan again
        grew = False
        for c, *_ in PLAN:
            if left and alloc[c] < min(len(finals(c)), len(refsets.refset(c, 'bench'))):
                alloc[c] += 1
                left -= 1
                grew = True
        if not grew:
            sys.exit(f'cannot build {n} trials: not enough finals / bench refs')
    return {c: k for c, k in alloc.items() if k}


_FACTS = {}


def facts(p):
    """(audible extent s, trims cleanly): a cut file starts quietly (2-5 ms in, <= CLEAN_HEAD) or with a strike (within
    6 dB of its loudest frame in its first 15 ms), not inside a sustained sound; the sound decays naturally before the
    file's own end fade (<= CLEAN_REACH) or builds to its end (loudest frame in the last 15 %, a riser); its recording
    floor, where measurable, is at most CLEAN_FLOOR; no anti-phase stereo (corr > -0.5). On the envelope of the file's
    own length class."""
    if p not in _FACTS:
        x = load(p)
        ext = extent_s(x)
        ref = manifest_entry(p) is not None
        i = analyse(x, 'long' if ext >= SHORT_S else 'short', end_fade_s(p), source_floor(p) if ref else None,
                    cut_file=ref)
        ok = ((i['head'] is None or i['head'] <= CLEAN_HEAD or i['strike'])
              and (i['reach'] <= CLEAN_REACH or i['peak_late'])
              and (i['floor'] is None or i['floor'] <= CLEAN_FLOOR) and stereo_corr(x) > -0.5)
        _FACTS[p] = (ext, bool(ok))
    return _FACTS[p]


def pick_trials(n, seed):
    import hashlib
    alloc = allocate(n)
    rng_made = np.random.default_rng([seed, 1])
    info = {c: (name, tier) for c, name, _, tier in PLAN}
    bench = {c: refsets.refset(c, 'bench') for c in alloc}
    digest = {}
    for c in alloc:
        for p in bench[c]:
            digest[p] = hashlib.md5(load(p).astype(np.float32).tobytes()).hexdigest()
    ext = lambda p: facts(p)[0]
    clean = lambda p: facts(p)[1]
    # most constrained category first: fewest distinct clean-trimming sources per trial, then fewest sources
    order = sorted(alloc, key=lambda c: (len({digest[p] for p in bench[c] if clean(p)}) / alloc[c],
                                         len({source_of(p) for p in bench[c]}) / alloc[c], c))
    used_hash, used_src, trials = set(), set(), []

    def available(c):
        """unused bench refs; clean trimmers first, then (where possible) a source recording not used yet"""
        cands = [p for p in bench[c] if digest[p] not in used_hash]
        fresh = [p for p in cands if source_of(p) not in used_src]
        for pool in ([p for p in fresh if clean(p)], [p for p in cands if clean(p)], fresh, cands):
            if pool:
                return pool, any(source_of(p) in used_src for p in pool)
        return [], False

    for c in order:
        fin = finals(c)
        if len(fin) < alloc[c]:
            sys.exit(f'{c}: {len(fin)} finals, {alloc[c]} trials wanted')
        # random among the length-compatible finals (audible extent within COMPAT of an available bench ref), clean
        # trimmers first; if fewer than needed, the finals closest in length. Duration and the edge levels are all
        # this pick looks at.
        cands0, _ = available(c)
        gap = {m: min(abs(np.log(ext(p) / ext(m))) for p in cands0) for m in fin}
        pool = [m for m in fin if gap[m] <= np.log(COMPAT) and clean(m)]
        if len(pool) < alloc[c]:
            pool = [m for m in fin if gap[m] <= np.log(COMPAT)]
        if len(pool) < alloc[c]:
            pool = sorted(fin, key=lambda m: (gap[m], m))[:alloc[c]]
        made = [pool[i] for i in sorted(rng_made.choice(len(pool), alloc[c], replace=False))]
        for m in made:
            cands, relaxed = available(c)
            if not cands:
                sys.exit(f'{c}: no unused bench ref left')
            em = ext(m)
            real = min(cands, key=lambda p: (abs(np.log(ext(p) / em)), p))
            relaxed = source_of(real) in used_src
            used_hash.add(digest[real])
            used_src.add(source_of(real))
            trials.append({'category': c, 'sound': info[c][0], 'tier': info[c][1], 'made': m, 'real': real,
                           'source_reused': relaxed, 'pool': [os.path.basename(q) for q in pool],
                           'clean_trim': {'real': clean(real), 'made': clean(m)}})
    # order: shuffled, no category twice in a row (deterministic from the seed)
    rng_order = np.random.default_rng([seed, 2])
    for _ in range(5000):
        perm = rng_order.permutation(len(trials))
        if all(trials[perm[i]]['category'] != trials[perm[i + 1]]['category'] for i in range(len(perm) - 1)):
            break
    trials = [trials[i] for i in perm]
    rng_ab = np.random.default_rng([seed, 3])
    ab = np.array(['A'] * (len(trials) // 2) + ['B'] * (len(trials) - len(trials) // 2))
    if len(trials) % 2 and rng_ab.integers(0, 2):
        ab[-1] = 'A'
    ab = ab[rng_ab.permutation(len(ab))]
    for t, r in zip(trials, ab):
        t['real_is'] = str(r)
    return trials


# --------------------------------------------------------------------------------------------------- blind build
SCORE_PY = r'''"""Score the blind test: python score_answers.py A B B A ...   (one letter per trial, in trial order: which one
was the REAL recording). Use ? for a trial you could not decide (it is left out of the count).

Prints per trial right / wrong, per sound, per tier (the round-3 / round-4 verdict of the category: strong / middle / weak) and
overall, with the chance of doing at least that well by pure guessing (one-sided binomial p). Reads
KEY_open_after.json next to this file.
"""
import json
import os
import sys
from math import comb


def p_at_least(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def p_at_most(k, n):
    return sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n if n else 1.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    key = json.load(open(os.path.join(here, 'KEY_open_after.json'), encoding='utf-8'))['trials']
    ans = [a.strip().upper() for a in ' '.join(sys.argv[1:]).replace(',', ' ').split()]
    if len(ans) != len(key) or any(a not in ('A', 'B', '?') for a in ans):
        sys.exit(f'give {len(key)} answers, each A or B (which one was the REAL recording) or ? (undecided), '
                 'in trial order')
    per, tier = {}, {}
    for k, a in zip(key, ans):
        ok = None if a == '?' else a == k['real']
        mark = 'skipped' if ok is None else ('right' if ok else 'wrong')
        print(f"trial {k['trial']:2d}  {k['sound']:<18} you: {a}  real: {k['real']}  {mark:<7}  "
              f"(real {k['real_file'].rsplit('/', 1)[-1]}, made {k['made_file'].rsplit('/', 1)[-1]})")
        if ok is not None:
            per.setdefault(k['sound'], []).append(ok)
            tier.setdefault(k['tier'], []).append(ok)
    print()
    for s, v in per.items():
        print(f'{s:<18} {sum(v)} of {len(v)} right')
    print()
    for s in ('strong', 'middle', 'weak'):
        if s in tier:
            v = tier[s]
            print(f'{s + " categories":<18} {sum(v)} of {len(v)} right  (p = {p_at_least(sum(v), len(v)):.3f})')
    n = sum(len(v) for v in per.values())
    right = sum(sum(v) for v in per.values())
    print()
    print(f'overall {right} of {n} right; chance of doing at least this well by guessing: '
          f'p = {p_at_least(right, n):.3f}')
    print('p < 0.05: you can tell made from real (at least for some sounds). Otherwise this test could not show it.')
    if n and p_at_most(right, n) < 0.05:
        print(f'note: this few right is itself unlikely by guessing (p = {p_at_most(right, n):.3f}): you heard the made '
              'sounds as the real ones more often than not.')


if __name__ == '__main__':
    main()
'''


def fingerprint(paths):
    """{path: (size, mtime_ns)}: to notice a final being rewritten (a polish round) while this builds"""
    out = {}
    for p in paths:
        try:
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime_ns)
        except OSError:
            out[p] = None
    return out


def md5(p):
    import hashlib
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def build_blind(out, n, seed, quiet=False, allow_fail=False, dry_run=False):
    inputs = sorted({q for c, *_ in PLAN for q in finals(c) + refsets.refset(c, 'bench')})
    before = fingerprint(inputs)
    trials = pick_trials(n, seed)
    tdir = os.path.join(out, 'blind_test_trials')
    pre_n = int(round(FADE_IN_S * SR))
    parts, key, files, rows, spans, t = [], [], [], [], [], LEAD
    all_fails, all_warns = [], []
    for i, tr in enumerate(trials, 1):
        real0, made0 = load(tr['real']), load(tr['made'])
        real, made, cls, l_on, l_end, fr, fm = trim_pair(real0, made0, tr['real'], tr['made'])
        g, gc, over = match_pair(real, made, cls)
        made = made * 10 ** ((g + gc) / 20)
        real = real * 10 ** (gc / 20)
        lr, lm = clip_levels(real, cls, pre_n), clip_levels(made, cls, pre_n)
        row, fails, warns = cleanliness(lr, lm, fr, fm, cls, l_on, l_end, gc)
        A, B = (real, made) if tr['real_is'] == 'A' else (made, real)
        ta = round(t, 2)                                  # on the 10 ms grid: the sheet times are exact
        tb = round(ta + len(A) / SR + GAP, 2)
        parts += [(ta, A), (tb, B)]
        spans += [(int(round(ta * SR)), len(A)), (int(round(tb * SR)), len(B))]
        pad = np.zeros((int(0.25 * SR), 2))
        files += [(os.path.join(tdir, f't{i:02d}_A.wav'), np.concatenate([pad, A, pad])),
                  (os.path.join(tdir, f't{i:02d}_B.wav'), np.concatenate([pad, B, pad]))]
        rows.append({'trial': i, 'sound': tr['sound'], **row, 'fails': fails, 'warnings': warns})
        all_fails += [f'trial {i} ({tr["sound"]}): {m}' for m in fails]
        all_warns += [f'trial {i} ({tr["sound"]}): {m}' for m in warns]
        key.append({
            'trial': i, 'category': tr['category'], 'sound': tr['sound'], 'tier': tr['tier'],
            'A_start': round(ta, 2), 'B_start': round(tb, 2),
            'length_s': {'A': round(len(A) / SR, 3), 'B': round(len(B) / SR, 3)}, 'real': tr['real_is'],
            'real_file': rel(tr['real']), 'made_file': rel(tr['made']),
            'real_source_reused_elsewhere': tr['source_reused'], 'clean_trim_pick': tr['clean_trim'],
            'file_length_s': {'real': round(len(real0) / SR, 3), 'made': round(len(made0) / SR, 3)},
            'audible_extent_s': {'real': round(extent_s(real0), 3), 'made': round(extent_s(made0), 3)},
            'class': cls, 'onset_level_db': round(l_on, 1), 'end_level_db': round(l_end, 1),
            'trim': {'real': fr, 'made': fm},
            'made_drawn_from': tr['pool'],
            'match': f'K-weighted loudest {int(WIN[cls] * 1000)} ms, made to real, then both to {TARGET[cls]} LUFS'
                     f' (true peak <= {TP_MAX} dBTP)',
            'made_gain_db': round(g, 2), 'pair_gain_db': round(gc, 2), 'peak_cap_lowered_db': round(over, 2),
            'levels_as_played': {'real': lr, 'made': lm},
            'cleanliness': {'fails': fails, 'warnings': warns},
            'note': NOTES.get(tr['category'], ''),
            'md5': {'real': md5(tr['real']), 'made': md5(tr['made'])},
        })
        t = round(tb + len(B) / SR + BETWEEN, 2)
    now = fingerprint(inputs)
    if now != before:          # nothing is written: the folder keeps the previous complete build
        changed = [rel(q) for q in inputs if now.get(q) != before.get(q)]
        sys.exit(f'input files changed while building ({", ".join(changed[:6])}): a polish round is writing finals; '
                 'run this again when it is done (nothing written)')
    total = t - BETWEEN + TAIL
    y = place(parts, total)
    # silence-only gaps: every sample outside the clips is exactly zero
    mask = np.ones(len(y), bool)
    for s, ln in spans:
        mask[s:s + ln] = False
    gaps_silent = bool(np.all(y[mask] == 0.0))
    if not gaps_silent:
        all_fails.append('the gaps between clips are not digital silence')
    if np.abs(y).max() > 10 ** (TP_MAX / 20) + 1e-6:
        all_fails.append(f'a sample above {TP_MAX} dBFS in blind_test.wav')
    worst = lambda k: max((r[k] for r in rows if r[k] is not None), default=None)
    summary = {
        'status': 'FAIL' if all_fails else 'PASS', 'trials': len(rows),
        'pairs_with_fails': sum(bool(r['fails']) for r in rows),
        'pairs_with_warnings': sum(bool(r['warnings']) for r in rows),
        'max_loudest_moment_diff_lu': worst('loudest_moment_diff_lu'),
        'max_meter_m400_diff_lu': worst('meter_m400_diff_lu'),
        'max_true_peak_dbtp': worst('true_peak_max_dbtp'), 'max_true_peak_diff_db': worst('true_peak_diff_db'),
        'max_preroll_dbfs': worst('preroll_max_dbfs'), 'max_floor_as_played_dbfs': worst('floor_as_played_dbfs'),
        'min_floor_margin_db': min((r['floor_margin_db'] for r in rows if r['floor_margin_db'] is not None),
                                   default=None),
        'max_length_ratio': worst('length_ratio'), 'gaps_digital_silence': gaps_silent,
        'fails': all_fails, 'warnings': all_warns}
    clean = {
        'what': 'residual differences between the two clips of each trial that a listener could use besides sound '
                'quality, measured on the clips as they play. No answers here: every value is a difference, a '
                'maximum or a sorted pair, never "real" or "made", so this file is safe to open before listening.',
        'limits_fail': FAIL_AT, 'limits_warn': WARN_AT,
        'method': {'trim': 'both clips start 4 ms (raised-cosine fade-in) before their first 1 ms frame at the '
                           'common onset level and end where their decay last stands at the common end level, with a '
                           'fade-out over the last 6 dB above it (see eval/forge/blind_v2.py)',
                   'level': 'made matched to real on the K-weighted loudest 100 ms (short pairs) or 400 ms (long '
                            'pairs), then both to -15 / -16 LUFS, lowered together if a true peak would pass -1 dBTP',
                   'floor': "a real ref's recording floor is measured in its source recording (5th percentile of 50 "
                            'ms windows), as it plays after the gains'},
        'summary': summary, 'trials': rows}
    if dry_run:
        print(f"cleanliness {summary['status']} (dry run, nothing written): {len(rows)} trials, max loudest-moment "
              f"difference {summary['max_loudest_moment_diff_lu']} LU, max true peak {summary['max_true_peak_dbtp']} "
              f"dBTP, min floor margin {summary['min_floor_margin_db']} dB")
        for m in all_fails:
            print('  FAIL ' + m)
        for m in all_warns:
            print('  warning: ' + m)
        return key, clean
    if all_fails and not allow_fail:
        print('CLEANLINESS FAIL (nothing written; the folder keeps its previous build):')
        for m in all_fails:
            print('  ' + m)
        sys.exit(2)
    json.dumps(clean), json.dumps(key)       # serialisable before anything is replaced
    os.makedirs(tdir, exist_ok=True)
    for old in glob.glob(os.path.join(tdir, 't*_[AB].wav')):
        os.remove(old)
    for f, x in files:
        write(f, x)
    write(os.path.join(out, 'blind_test.wav'), y)
    # read the written trial files back (24-bit) and measure them again: what the listener gets
    pad_n = int(0.25 * SR)
    ver = []
    for i, k in enumerate(key, 1):
        cls = k['class']
        ab = {s_: sf.read(os.path.join(tdir, f't{i:02d}_{s_}.wav'), always_2d=True, dtype='float64')[0][pad_n:-pad_n]
              for s_ in 'AB'}
        ver.append((abs(loud_max(ab['A'], WIN[cls]) - loud_max(ab['B'], WIN[cls])),
                    max(true_peak(ab['A']), true_peak(ab['B']))))
    summary['verified_from_written_files'] = {'max_loudest_moment_diff_lu': round(max(v[0] for v in ver), 3),
                                              'max_true_peak_dbtp': round(max(v[1] for v in ver), 2)}
    if max(v[0] for v in ver) > FAIL_AT['loud_lu'] or max(v[1] for v in ver) > FAIL_AT['true_peak_dbtp']:
        summary['status'] = 'FAIL'
        summary['fails'].append('the written trial files do not hold the matched levels')
    json.dump(clean, open(os.path.join(out, 'cleanliness.json'), 'w', encoding='utf-8'), indent=1)
    sign = lambda f, eps: {'made_more': int(sum(f(k) > eps for k in key)),
                           'made_less': int(sum(f(k) < -eps for k in key)),
                           'same': int(sum(abs(f(k)) <= eps for k in key))}
    lv = lambda k, q: k['levels_as_played']['made'][q] - k['levels_as_played']['real'][q]
    signs = {'other_loudness_window_0.1lu': sign(lambda k: lv(k, 'loud_other_window'), 0.1),
             'meter_m400_0.1lu': sign(lambda k: lv(k, 'meter_m400'), 0.1),
             'true_peak_0.5db': sign(lambda k: lv(k, 'true_peak_dbtp'), 0.5),
             'length_10pct': sign(lambda k: np.log(k['levels_as_played']['made']['length_s']
                                                    / k['levels_as_played']['real']['length_s']), np.log(1.1))}
    json.dump({'seed': seed, 'built': time.strftime('%Y-%m-%d %H:%M'), 'trials': key,
               'excluded_categories': EXCLUDED, 'cleanliness': summary,
               'residual_signs_made_vs_real': signs,
               'how': 'built by eval/forge/blind_v2.py (see its docstring): forge v2 finals (random pick among the '
                      'length-compatible, cleanly trimmable ones) vs held-out bench refs (closest audible length, no '
                      'file / source twice), both trimmed to their audible extent at common levels re their own '
                      'loudest frame (matched fades), made matched to real on the loudest moment, random order and '
                      'A/B (real = A on half the trials)'},
              open(os.path.join(out, 'KEY_open_after.json'), 'w', encoding='utf-8'), indent=1)
    open(os.path.join(out, 'score_answers.py'), 'w', encoding='utf-8').write(SCORE_PY)
    from math import comb
    nt = len(key)
    pk = lambda k: sum(comb(nt, j) for j in range(k, nt + 1)) / 2 ** nt
    k05 = next(k for k in range(nt + 1) if pk(k) < 0.05)
    cats = sorted({k['category'] for k in key})
    fl, fm_ = summary['max_floor_as_played_dbfs'], summary['min_floor_margin_db']
    fl_s = (f'recording floors at most {fl:.0f} dBFS, at least {fm_:.0f} dB under where the clips start and end'
            if fl is not None
            else 'no recording floor measurable')
    cl = (f"Cleanliness: {summary['status']}. Loudest moments within {summary['max_loudest_moment_diff_lu']:.2f} LU "
          f"in every pair; true peaks at most {summary['max_true_peak_dbtp']:.1f} dBTP; no clip plays room noise "
          f"before or after its sound (each starts with a {FADE_IN_S * 1000:.0f} ms fade-in just before its onset and "
          f"ends with a matched fade at the same decay level as its partner; {fl_s}); gaps are digital silence. "
          f"{summary['pairs_with_warnings']} pair(s) with warnings, 0 with fails: cleanliness.json (safe to open now: "
          'it never says which clip is real).')
    if all_fails:
        cl = f'Cleanliness: FAIL ({len(all_fails)} problem(s), see cleanliness.json); written with --allow-fail.'
    import textwrap
    cl = chr(10).join(textwrap.wrap(cl, 112))
    L = ['Blind test: blind_test.wav (or blind_test_trials/tNN_A.wav + tNN_B.wav, one trial at a time).',
         'In every trial you hear A, then B. One is a REAL recording, the other was made entirely in code by earshot',
         'forge. Write down which one you think is REAL (A or B; ? if you cannot decide). Do not open KEY_open_after.json',
         'first. Both sounds of a trial are trimmed the same way to where they are audible (no room noise before or',
         'after either), play equally loud at their loudest moment, and the real recordings come from a held-out set',
         'the forge never tuned against. Silence between trials, no markers: each trial starts at the times below.',
         f'Then:  .venv\\Scripts\\python.exe {rel(out).replace("/", chr(92))}\\score_answers.py A B ...  '
         f'({nt} letters, in trial order)',
         f'By chance you would get about {nt // 2} of {nt} right; {k05} or more is unlikely by luck '
         f'(p = {pk(k05):.3f}).', '', cl, '',
         f"{'trial':>5}  {'sound':<18} {'A starts':>8}  {'B starts':>8}   real is (A/B)"]
    for k in key:
        L.append(f"{k['trial']:>5}  {k['sound']:<18} {k['A_start']:8.2f}  {k['B_start']:8.2f}   ____")
    L += ['', f'total length {len(y) / SR:.1f} s ({len(y) / SR / 60:.1f} min), 48 kHz, 24-bit stereo; '
              f'{nt} trials over {len(cats)} sounds']
    open(os.path.join(out, 'answer_sheet.txt'), 'w', encoding='utf-8').write(chr(10).join(L) + chr(10))
    if not quiet:
        print(chr(10).join(L))
        for m in all_warns:
            print('  warning: ' + m)
    return key, clean


# ------------------------------------------------------------------------------------------------ progress build
def results_v2(path=None):
    if path:
        return path
    found = []
    for p in glob.glob(f'{HERE}/results_v2r*.json'):
        m = re.match(r'results_v2r(\d+)\.json$', os.path.basename(p))
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        sys.exit('no eval/forge/results_v2r<N>.json')
    return max(found)[1]


def nearest_v2(v1_file, cands):
    a = load(v1_file)
    pa, ea = frozen_tone.profile(a), extent_s(a)
    best = None
    for c in cands:
        b = load(c)
        d = float(np.sqrt(np.mean((frozen_tone.profile(b) - pa) ** 2))) + 6.0 * abs(np.log2(extent_s(b) / ea))
        if best is None or d < best[0]:
            best = (d, c)
    return best


def fmt(v, nd=2):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


# (key in summary, label, rule, eps): what "closer to real" means per check. The real rows are the bench refs against
# each other (real loo) and the forge refs against the bench (real x); a check only gets better / worse when it moves
# relative to what those real recordings score ('same' within eps):
#   'low'   lower is better, but anything at or below the best real row counts as real (the excess over it is compared)
#   'one'   a ratio: log distance to the span of 1 and the real rows
#   'span'  distance to the span of the real rows' values (variety, texture)
#   'model' a model check: higher is better per bench.py, but saturated; reported as up / down only
METRICS = [
    ('var_cos', "variety: CLAP cos between variants (the refs' own in the real columns)", 'span', 0.01),
    ('tone_db', 'tone: 1/3-octave distance to the bench median, dB', 'low', 0.3),
    ('env_oct', 'envelope: attack / t40 / length off, octaves', 'low', 0.03),
    ('curve_db', 'envelope curve distance, dB', 'low', 0.5),
    ('t40_ratio', 'tail t40 / bench median (1 = same)', 'one', 0.03),
    ('artifacts', 'artifact flags (count)', 'low', 0.5),
    ('stereo_corr', 'stereo correlation (1 = mono)', 'span', 0.02),
    ('floor_db', 'floor, dB re the loudest 5 ms', 'span', 1.0),
    ('darken', 'tail darkening (log2 centroid half-tail / peak)', 'span', 0.05),
    ('typ_z', 'CLAP typicality z (model)', 'model', 0.05),
    ('phrase_z', 'CLAP held-out phrasings z (model)', 'model', 0.05),
    ('margin', 'phrasing margin over confusions (model)', 'model', 0.05),
    ('as_names', 'AudioSet names the category, share (model)', 'model', 0.05),
]


def verdict(rule, a, b, reals, eps):
    """'better' / 'worse' / 'same' (+ why) for a v1 -> v2 move of one check, given the real rows' values"""
    if a is None or b is None:
        return ''
    reals = [r for r in reals if r is not None]
    if rule == 'model':
        v = 'up' if b > a + eps else ('down' if b < a - eps else 'same')
        return v + (' (v1 sat above the real rows)' if reals and a > max(reals) else '')
    if rule == 'low':
        ref = min(reals) if reals else 0.0
        dist = lambda v: max(0.0, v - ref)
    elif rule == 'one':
        lo, hi = min([1.0] + reals), max([1.0] + reals)
        dist = lambda v: max(0.0, np.log(lo / max(v, 1e-3)), np.log(max(v, 1e-3) / hi))
    else:
        if not reals:
            return ''
        lo, hi = min(reals), max(reals)
        dist = lambda v: max(0.0, lo - v, v - hi)
    da, db = dist(a), dist(b)
    if da == 0 and db == 0:
        return 'same (both within the real rows)'
    return 'better' if db < da - eps else ('worse' if db > da + eps else 'same')


def build_progress(pdir, results_path=None, quiet=False):
    os.makedirs(pdir, exist_ok=True)
    r1 = json.load(open(f'{HERE}/results_v1.json', encoding='utf-8'))
    rp = results_v2(results_path)
    r2 = json.load(open(rp, encoding='utf-8'))
    # listening file
    parts, lines, t = [], [], LEAD
    picks = []
    for name, cat, v1dir, stem in V1:
        v1 = [f'{REPO}/{v1dir}/{stem}_01.wav', f'{REPO}/{v1dir}/{stem}_02.wav']
        cands = finals(cat)
        seq = []
        for f in v1:
            d, c = nearest_v2(f, cands)
            cands = [q for q in cands if q != c]
            seq += [('v1', f, None), ('v2', c, d)]
        xs = [load(f) for _, f, _ in seq]
        # the four sounds equally loud at their loudest moment (the blind test's measure), one common gain per group
        # keeps every true peak at or under TP_MAX
        cls = 'long' if max(extent_s(x) for x in xs) >= SHORT_S else 'short'
        gains = [TARGET[cls] - loud_max(x, WIN[cls]) for x in xs]
        over = max(0.0, max(true_peak(x) + g for x, g in zip(xs, gains)) - TP_MAX)
        gains = [g - over for g in gains]
        lines.append(f'{name}  (v2 category {cat}; matched on the K-weighted loudest {int(WIN[cls] * 1000)} ms'
                     + (f', all {over:.1f} dB under the target so no true peak passes {TP_MAX} dBTP)' if over else ')'))
        for (ver, f, d), x, g in zip(seq, xs, gains):
            x = x * 10 ** (g / 20)
            parts.append((t, x))
            lm, tp = loud_max(x, WIN[cls]), true_peak(x)
            extra = f'   (v2 final closest to the v1 final before it: distance {d:.1f})' if d is not None else ''
            lines.append(f'  {t:7.2f} s  {ver}  loudest {lm:6.2f} LUFS  true peak {tp:6.2f} dBTP  {rel(f)}{extra}')
            picks.append({'sound': name, 'version': ver, 'start': round(t, 2), 'file': rel(f),
                          'gain_db': round(g, 2), 'loudest_lufs': round(lm, 3), 'true_peak_dbtp': round(tp, 2),
                          'measure': f'K-weighted loudest {int(WIN[cls] * 1000)} ms'})
            t += len(x) / SR + 1.0
        t += 2.0
    total = t - 3.0 + TAIL
    y = place(parts, total)
    write(os.path.join(pdir, 'v1_vs_v2.wav'), y)
    head = ['v1_vs_v2.wav: the five sounds forge v1 had, before (v1) and after (v2), alternating v1, v2, v1, v2.',
            'The four sounds of a group play equally loud at their loudest moment (K-weighted loudest 100 ms; 400 ms',
            f'when a sound is 1 s or longer) at {TARGET["short"]:g} / {TARGET["long"]:g} LUFS; one common gain per '
            f'group keeps every true peak at or under {TP_MAX:g} dBTP',
            '(so neither version wins by level; the peaks left differ by the crest of the sounds). 1 s between sounds,',
            '3 s between sound groups. Each v2 final is the one closest to the v1 final before it in 1/3-octave',
            'balance + audible length (distance = RMS dB over the bands + 6 dB per octave of length), so each pair is',
            'like for like.',
            f'total {len(y) / SR:.1f} s, 48 kHz, 24-bit stereo', '']
    open(os.path.join(pdir, 'progress.txt'), 'w', encoding='utf-8').write(chr(10).join(head + lines) + chr(10))
    # numbers
    md = ['# forge v1 -> v2: measured before / after', '',
          f'Same frozen, held-out benchmark (`eval/forge/bench.py`): v1 = `eval/forge/results_v1.json` '
          f'({r1.get("created", "?")}), v2 = `{rel(rp)}` ({r2.get("created", "?")}). Made sounds are measured against '
          'bench recordings no forge search ever used. Columns: v1 and v2 made medians; "bench" = each bench ref against '
          'the other bench refs (real loo); "forge refs" = real recordings the bench never saw, scored the same way '
          '(real x). Listening: `v1_vs_v2.wav` (`progress.txt` has the times).', '']
    # scope
    cats_v2 = sorted(d for d in os.listdir(MADE) if finals(d))
    n_fin = sum(len(finals(d)) for d in cats_v2)
    md += [f'Scope: v1 had 5 sounds, 3 finals each (15). v2 delivers {len(cats_v2)} categories, {n_fin} finals '
           f'(`out/forge_v2/<category>/final/`).', '']
    stale = []
    try:              # when the benchmark ran (its own stamp: the file may be rewritten later, e.g. paths anonymised)
        made_at = time.mktime(time.strptime(r2['created'], '%Y-%m-%d %H:%M'))
    except (KeyError, ValueError):
        made_at = os.path.getmtime(rp)
    for name, cat, *_ in V1:
        listed = {os.path.normcase(os.path.abspath(os.path.join(REPO, m['file']))) for m in
                  r2['categories'].get(cat, {}).get('made', [])}
        now = {os.path.normcase(os.path.abspath(p)) for p in finals(cat)}
        newer = [p for p in finals(cat) if os.path.getmtime(p) > made_at + 60]
        if listed != now or newer:
            stale.append(cat)
    if stale:
        md += [f'**Stale:** the finals of {", ".join(stale)} changed after `{rel(rp)}` was written: these rows measure '
               'the earlier files. Re-run eval/forge/bench.py and rebuild.', '']
    summary_rows = []
    for name, cat, *_ in V1:
        c1, c2 = r1['categories'][cat], r2['categories'][cat]
        s1, s2 = c1['summary'], c2['summary']
        loo, rx = s1.get('real_loo') or {}, s1.get('real_cross') or {}
        m1, m2 = s1['made'], s2['made']
        i1, i2 = c1.get('inside') or {}, c2.get('inside') or {}
        marker = ''.join(f' ({os.path.splitext(os.path.basename(p))[0]})' for p in
                         glob.glob(f'{MADE}/{cat}/final/KEPT_*.txt'))
        md += [f'## {name} (`{cat}`){marker}', '',
               f'n: v1 {m1.get("n")}, v2 {m2.get("n")} finals. Bench refs: {c1.get("n_bench")}.', '',
               '| check | v1 | v2 | bench (real loo) | forge refs (real x) | v1 -> v2 |', '|---|---|---|---|---|---|']
        nb = nw = 0
        for k, label, rule, eps in METRICS:
            a, b = m1.get(k), m2.get(k)
            if k == 'var_cos':
                realv = (c1.get('diversity_bench') or {}).get('cos_mean')
                rxv = (c1.get('diversity_forge') or {}).get('cos_mean')
            else:
                realv, rxv = loo.get(k), rx.get(k)
            v = verdict(rule, a, b, [realv, rxv], eps)
            if rule != 'model':
                nb += v == 'better'
                nw += v == 'worse'
            md.append(f'| {label} | {fmt(a)} | {fmt(b)} | {fmt(realv)} | {fmt(rxv)} | {v} |')
        ins = lambda d: (f'{np.median(d["per_sound"]):g} / {d["n_checks"]}' if d.get('per_sound') else '-')
        real_ins = (f'{np.median(i1["real_per_sound"]):g} / {i1["n_checks"]}' if i1.get('real_per_sound') else '-')
        iv = ''
        if i1.get('per_sound') and i2.get('per_sound'):
            a, b = float(np.median(i1['per_sound'])), float(np.median(i2['per_sound']))
            iv = 'more' if b > a + 0.4 else ('fewer' if b < a - 0.4 else 'same')
        md.append(f'| inside the real 80 % limits (median checks) | {ins(i1)} | {ins(i2)} | {real_ins} (real rows) | | '
                  f'{iv} |')
        md.append('')
        summary_rows.append((name, cat, nb, nw))
        if c1.get('note'):
            md += [f'Bench note: {c1["note"]}', '']
    md += ['## In one line per sound (model-free checks: tone, envelope, curve, tail, artifacts, variety, texture)', '',
           '| sound | better | worse |', '|---|---|---|']
    md += [f'| {n} (`{c}`) | {b} | {w} |' for n, c, b, w in summary_rows]
    md += ['', '## How to read this honestly', '',
           '- "better" / "worse" for the model-free checks means closer to / further from what real recordings score '
           'on the same check (the bench refs against each other, and the forge refs against the bench): a tone, '
           'envelope or curve distance only counts above the best real row, a tail ratio or a variety / texture value '
           'only outside the span of the real rows ("same (both within the real rows)" otherwise). The counts weight '
           'every check the same.',
           '- The model checks (CLAP typicality and phrasings, AudioSet) saturate: v1 sat above the real recordings on '
           'them after its searches tuned against CLAP, so a v2 drop toward the real rows is not by itself a loss. They '
           'are listed as up / down only. The "inside" row counts 7 checks of which 4 are model checks (CLAP '
           'typicality, phrasings, margin, AudioSet p) and 3 model-free (tone, envelope, curve): a fall there while '
           'the model-free rows improve may be the saturation unwinding rather than the sound '
           'getting worse; the numbers cannot settle which, the blind test can.',
           '- v1 impact numbers are optimistic: v1 impact was tuned against kit impact_01-03, and impact_03 is the '
           'only impact bench ref today. (v1 fast whoosh used kit whoosh_fast_01-04, none cut from a bench source.)',
           '- Impact has a single bench ref (no spread, no variety baseline); typing_key has 12, click 7.',
           '- Numbers are not ears: the clean blind test (`../blind_v2/`) is the deciding test.', '']
    open(os.path.join(pdir, 'progress.md'), 'w', encoding='utf-8').write(chr(10).join(md) + chr(10))
    json.dump({'results_v1': 'eval/forge/results_v1.json', 'results_v2': rel(rp), 'picks': picks,
               'built': time.strftime('%Y-%m-%d %H:%M')},
              open(os.path.join(pdir, 'progress.json'), 'w', encoding='utf-8'), indent=1)
    if not quiet:
        print(chr(10).join(head + lines))
        print(f'progress.md from {rel(rp)}' + (f'; STALE: {stale}' if stale else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument('--out', default=f'{REPO}/out/showcase/blind_v2', help='blind test folder')
    ap.add_argument('--progress', default=None, help='progress folder (default: <out>/../progress)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--n', type=int, default=20, help='trials (default 20)')
    ap.add_argument('--results', default=None, help='v2 benchmark JSON (default: the newest results_v2r<N>.json)')
    ap.add_argument('--only', choices=('blind', 'progress'), default=None)
    ap.add_argument('--allow-fail', action='store_true',
                    help='write the blind test even when a cleanliness limit fails (marked FAIL in the files)')
    ap.add_argument('--dry-run', action='store_true', help='build the blind test in memory, print its cleanliness, '
                    'write nothing (the progress file is skipped too)')
    a = ap.parse_args()
    out = os.path.abspath(a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out))
    pdir = a.progress or os.path.join(os.path.dirname(out), 'progress')
    pdir = os.path.abspath(pdir if os.path.isabs(pdir) else os.path.join(REPO, pdir))
    if a.only != 'progress':
        build_blind(out, a.n, a.seed, allow_fail=a.allow_fail, dry_run=a.dry_run)
    if a.only != 'blind' and not a.dry_run:
        build_progress(pdir, a.results)


if __name__ == '__main__':
    main()
