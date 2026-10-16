"""Clean blind test (forge v2 finals vs held-out REAL recordings) + the v1 -> v2 progress listening file.

    E:/earshot/.venv/Scripts/python.exe eval/forge/blind_v2.py --out out/showcase/blind_v2 --seed 2026
    E:/earshot/.venv/Scripts/python.exe out/showcase/blind_v2/score_answers.py A B B A ...      (after listening)

One command rebuilds both from whatever is in out/forge_v2/<category>/final/ now (rerun it after a polish round).
Nothing is synthesised here: every sound is a file that already exists (forge finals, v1 finals, licensed bench refs),
only trimmed, faded, gain-changed and placed on silence. No markers, beeps or voice: trials are separated by silence.

BLIND TEST (--out, default out/showcase/blind_v2)
  Trials: N (default 20) over the licensed real categories that have bench refs (PLAN: two trials for the strongest
  categories of the round-3 verdicts, one for the others, including the weak ones so the test is honest). Each trial
  pairs one forge v2 final (<category>_NN.wav in out/forge_v2/<category>/final/) with one REAL recording of the same
  category from the held-out BENCH split (earshot.forge.refsets: source recordings no forge search ever used).
  Picks: the made final is drawn at random (seeded) among the category's length-compatible finals (audible extent
  within 1.3x of an available bench ref; if too few, the closest in length), never by a score; the real one is the
  bench ref whose audible extent is closest to it, with no real file and, where possible, no source recording used
  twice in the test (whoosh_fast / swish and notify / logo_sting share bench files). Duration is the only property
  either pick looks at, so the trims below cut little.
  Trim: A and B get the SAME length. Usually both start 5 ms before their onset (first 1 ms frame within 40 dB of the
  loudest; zero-padded if the file has no 5 ms lead) and run for the shorter of the two files, at most the longer
  audible extent (last 5 ms frame within 50 dB of the loudest) + 30 ms; a clip cut before its file ends gets a
  raised-cosine fade-out (15 % of the length, 10-300 ms), a clip that ends with its file keeps its own ending, both
  get a 1 ms fade-in. Pairs that both build to their end (risers: loudest frame in the last 30 %) are end-aligned
  instead: both end with their files and the longer one loses its quiet start under a fade-in of the same size.
  Level: the made sound is matched to the real one. Short pairs (< 1 s): half the sample-peak difference plus half
  the loudest-100-ms RMS difference, the RMS K-weighted (BS.1770 pre-filter: unweighted, a real whoosh's low rumble
  counted as loud and its made partner played 5-6 LU louder). Long pairs: half the integrated-LUFS difference
  (BS.1770, gated, of the equal-length clips) plus half the momentary-max-LUFS difference (integrated alone let a made
  hit with a shorter tail play 3-4 LU louder at its head). What is left is half the two sounds' crest difference
  (about +-2 LU momentary). Then one common gain for the pair: the louder momentary-max LUFS at -16 LUFS, never a peak
  above -1 dBFS.
  Order: trials shuffled (no category twice in a row), real-as-A on exactly half the trials, both from --seed.
  Layout: 1 s lead, A, 1.2 s silence, B, 2.5 s silence, next trial ... 1 s tail; 48 kHz, 24-bit stereo.
  Writes: blind_test.wav, answer_sheet.txt (trial, category, A start, B start, blank), KEY_open_after.json (the
  answers, picks, trims, gains and levels), score_answers.py (per trial right / wrong, per category, per tier,
  overall, one-sided binomial p), blind_test_trials/tNN_A.wav + tNN_B.wav (each trial on its own, same levels).
  The key stores the md5 of every file used. If a final or bench ref changes while this runs (a polish round writing
  finals), it stops before writing anything, so the folder always holds one complete, consistent build.

PROGRESS (--progress, default <out>/../progress)
  v1_vs_v2.wav: the five sounds v1 had (key press, UI click, pop, fast whoosh, impact): v1 final, v2 final, v1 final,
  v2 final, each peak-matched to -1 dBFS (v1 = demo/forge-test/<sound>/final/<sound>_01 and _02; v2 = the v2 final
  closest to that v1 final in 1/3-octave balance and audible length, so like is played against like). progress.txt
  lists every start time. progress.md: the measured before / after numbers from eval/forge/results_v1.json against the
  newest eval/forge/results_v2r<N>.json (both from the same frozen held-out benchmark, eval/forge/bench.py).
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
from earshot import features  # noqa: E402  (BS.1770 integrated LUFS)
from earshot.forge import refsets  # noqa: E402

SR = 48000
MADE = f'{REPO}/out/forge_v2'
PEAK_MAX_DB = -1.0
PAIR_LUFS_M = -16.0          # the louder sound of a pair at this momentary-max loudness (unless the peak cap is lower)
PRE, FADE_IN = 0.005, 0.001
LEAD, GAP, BETWEEN, TAIL = 1.0, 1.2, 2.5, 1.0
SHORT_S = 1.0
COMPAT = 1.3                 # made finals whose audible extent is within this factor of an available bench ref

# (category, answer-sheet name, trials at N = 20, tier from the round-3 verdicts in OVERNIGHT.md / results_v2r3.md)
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
    ('toggle', 'toggle', 1, 'weak'),
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


def load(p):
    x, sr = sf.read(p, always_2d=True, dtype='float64')
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
    return (end_of(x) - onset(x)) / SR


def peak_db(x):
    return float(20 * np.log10(np.abs(x).max() + 1e-12))


def rms100_db(x, kw=False):
    """RMS of the loudest 100 ms (the whole sound over 100 ms when shorter); kw=True: K-weighted first (BS.1770
    pre-filter), so low rumble does not count as loud"""
    if kw:
        x = features._kw(np.concatenate([np.zeros((int(0.05 * SR), 2)), x]))[int(0.05 * SR):]
    m = (x ** 2).mean(axis=1)
    w = int(0.1 * SR)
    c = np.convolve(m, np.ones(w) / w, mode='valid') if len(m) > w else np.array([m.sum() / w])
    return float(10 * np.log10(c.max() + 1e-20))


def lufs_int(x):
    n = int(0.4 * SR)
    if len(x) < n:
        x = np.concatenate([x, np.zeros((n - len(x), 2))])
    return float(features.integrated_lufs(x))


def lufs_max(x):
    """momentary-max loudness (K-weighted 400 ms windows, zero-padded when shorter), as a meter shows it"""
    t, L = features.loudness_curve(x if len(x) >= int(0.4 * SR) else np.concatenate(
        [x, np.zeros((int(0.4 * SR) - len(x), 2))]), 0.4, 0.01)
    return float(np.max(L))


def raised(n):
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(1, n))


def peak_late(x):
    """riser-like: the loudest 5 ms frame lies in the last 30 % of the audible extent (it builds to its end)"""
    e, n = frames_db(x, 5.0)
    o, end = onset(x), end_of(x)
    return int(np.argmax(e)) * n >= o + 0.7 * (end - o)


def level_at(x, i):
    """level (dB re the loudest 5 ms frame) of the 5 ms frame at sample i"""
    e, n = frames_db(x, 5.0)
    return round(float(e[max(0, min(len(e) - 1, i // n))] - e.max()), 1)


def trim_pair(a, b):
    """one common length for both clips (see the module doc); returns (a', b', length, align, cut levels).
    'start' alignment: both begin PRE before their onset; a clip that has to stop before its file ends gets a fade-out
    (15 % of the length, 10-300 ms), a clip that ends with its file keeps its own ending. 'end' alignment (both build
    to their end, risers): both end where their files end; the longer one loses its quiet start under a fade-in."""
    pre = int(PRE * SR)
    fo = lambda L: int(np.clip(0.15 * L, 0.010 * SR, 0.300 * SR))
    if peak_late(a) and peak_late(b):
        starts = [max(0, onset(x) - pre) for x in (a, b)]
        L = int(min(len(a) - starts[0], len(b) - starts[1]))
        out, cut = [], []
        for x in (a, b):
            s = len(x) - L
            y = x[s:].copy()
            n_in = fo(L) if s > max(0, onset(x) - pre) + int(0.001 * SR) else int(FADE_IN * SR)
            cut.append(level_at(x, s) if n_in > int(FADE_IN * SR) else None)
            y[:n_in] *= raised(n_in)[:, None]
            out.append(y)
        return out[0], out[1], L, 'end', cut
    starts, avail, ext = [], [], []
    for x in (a, b):
        s = onset(x) - pre
        starts.append(s)
        avail.append(len(x) - s)
        ext.append(end_of(x) - s)
    L = int(min(min(avail), max(ext) + 0.03 * SR))
    out, cut = [], []
    for x, s, av in zip((a, b), starts, avail):
        y = np.concatenate([np.zeros((-s, 2)), x[:s + L]]) if s < 0 else x[s:s + L].copy()
        y = y[:L]
        fi = int(FADE_IN * SR)
        y[:fi] *= raised(fi)[:, None]
        if L < av - int(0.001 * SR):                      # truncated: fade out (and say how loud it was there)
            f = fo(L)
            cut.append(level_at(x, max(0, s) + L - f))
            y[L - f:] *= raised(f)[::-1][:, None]
        else:
            cut.append(None)
        out.append(y)
    return out[0], out[1], L, 'start', cut


def match_pair(real, made):
    """gain (dB) on the made clip so it plays as loud as the real one, the method name, and the two differences"""
    if len(real) / SR < SHORT_S:
        dp, dr = peak_db(real) - peak_db(made), rms100_db(real, kw=True) - rms100_db(made, kw=True)
        return 0.5 * (dp + dr), 'short: mean of sample peak and K-weighted loudest-100-ms RMS', {
            'peak_diff_db': dp, 'rms100_kw_diff_db': dr}
    di, dm = lufs_int(real) - lufs_int(made), lufs_max(real) - lufs_max(made)
    return 0.5 * (di + dm), 'long: mean of integrated and momentary-max LUFS', {'lufs_i_diff_db': di,
                                                                                  'lufs_m_max_diff_db': dm}


def levels(x):
    return {'peak_dbfs': round(peak_db(x), 2), 'rms100_dbfs': round(rms100_db(x), 2),
            'rms100_kw_db': round(rms100_db(x, kw=True), 2), 'lufs_i': round(lufs_int(x), 2),
            'lufs_m_max': round(lufs_max(x), 2)}


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
    # most constrained category first: fewest distinct sources per trial
    order = sorted(alloc, key=lambda c: (len({source_of(p) for p in bench[c]}) / alloc[c], c))
    used_hash, used_src, trials = set(), set(), []
    ext_cache = {}

    def ext(p):
        if p not in ext_cache:
            ext_cache[p] = extent_s(load(p))
        return ext_cache[p]

    def available(c):
        cands = [p for p in bench[c] if digest[p] not in used_hash and source_of(p) not in used_src]
        if cands:
            return cands, False
        return [p for p in bench[c] if digest[p] not in used_hash], True

    for c in order:
        fin = finals(c)
        if len(fin) < alloc[c]:
            sys.exit(f'{c}: {len(fin)} finals, {alloc[c]} trials wanted')
        # random among the length-compatible finals (audible extent within COMPAT of an available bench ref);
        # if fewer than needed, the finals closest in length. Duration is the only property this pick looks at.
        cands0, _ = available(c)
        gap = {m: min(abs(np.log(ext(p) / ext(m))) for p in cands0) for m in fin}
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
            used_hash.add(digest[real])
            used_src.add(source_of(real))
            trials.append({'category': c, 'sound': info[c][0], 'tier': info[c][1], 'made': m, 'real': real,
                           'source_reused': relaxed, 'pool': [os.path.basename(q) for q in pool]})
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

Prints per trial right / wrong, per sound, per tier (the round-3 verdict of the category: strong / middle / weak) and
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


def build_blind(out, n, seed, quiet=False):
    inputs = sorted({q for c, *_ in PLAN for q in finals(c) + refsets.refset(c, 'bench')})
    before = fingerprint(inputs)
    trials = pick_trials(n, seed)
    tdir = os.path.join(out, 'blind_test_trials')
    parts, key, files, t = [], [], [], LEAD
    for i, tr in enumerate(trials, 1):
        real0, made0 = load(tr['real']), load(tr['made'])
        real, made, L, align, cut = trim_pair(real0, made0)
        g, how, diff = match_pair(real, made)
        made = made * 10 ** (g / 20)
        loud = max(lufs_max(real), lufs_max(made))
        gc = min(PEAK_MAX_DB - max(peak_db(real), peak_db(made)), PAIR_LUFS_M - loud)
        real, made = real * 10 ** (gc / 20), made * 10 ** (gc / 20)
        A, B = (real, made) if tr['real_is'] == 'A' else (made, real)
        slot = L / SR
        ta = round(t, 2)                                  # on the 10 ms grid: the sheet times are exact
        tb = round(ta + slot + GAP, 2)
        parts += [(ta, A), (tb, B)]
        pad = np.zeros((int(0.25 * SR), 2))
        files += [(os.path.join(tdir, f't{i:02d}_A.wav'), np.concatenate([pad, A, pad])),
                  (os.path.join(tdir, f't{i:02d}_B.wav'), np.concatenate([pad, B, pad]))]
        lr, lm = levels(real), levels(made)
        key.append({
            'trial': i, 'category': tr['category'], 'sound': tr['sound'], 'tier': tr['tier'],
            'A_start': round(ta, 2), 'B_start': round(tb, 2), 'length_s': round(slot, 3), 'real': tr['real_is'],
            'real_file': rel(tr['real']), 'made_file': rel(tr['made']),
            'real_source_reused_elsewhere': tr['source_reused'],
            'file_length_s': {'real': round(len(real0) / SR, 3), 'made': round(len(made0) / SR, 3)},
            'audible_extent_s': {'real': round(extent_s(real0), 3), 'made': round(extent_s(made0), 3)},
            'align': align,
            'cut_at_db': {'real': cut[0], 'made': cut[1]},     # level where a truncated clip's fade begins (None: kept)
            'made_drawn_from': tr['pool'],
            'match': how, 'made_gain_db': round(g, 2), 'pair_gain_db': round(gc, 2),
            'differences_before_match_db': {k: round(v, 2) for k, v in diff.items()},
            'levels_as_played': {'real': lr, 'made': lm},
            'note': NOTES.get(tr['category'], ''),
            'md5': {'real': md5(tr['real']), 'made': md5(tr['made'])},
        })
        t = round(tb + slot + BETWEEN, 2)
    now = fingerprint(inputs)
    if now != before:          # nothing is written: the folder keeps the previous complete build
        changed = [rel(q) for q in inputs if now.get(q) != before.get(q)]
        sys.exit(f'input files changed while building ({", ".join(changed[:6])}): a polish round is writing finals; '
                 'run this again when it is done (nothing written)')
    total = t - BETWEEN + TAIL
    y = place(parts, total)
    assert np.abs(y).max() <= 10 ** (PEAK_MAX_DB / 20) + 1e-6
    os.makedirs(tdir, exist_ok=True)
    for old in glob.glob(os.path.join(tdir, 't*_[AB].wav')):
        os.remove(old)
    for f, x in files:
        write(f, x)
    write(os.path.join(out, 'blind_test.wav'), y)
    json.dump({'seed': seed, 'built': time.strftime('%Y-%m-%d %H:%M'), 'trials': key,
               'excluded_categories': EXCLUDED,
               'how': 'built by eval/forge/blind_v2.py (see its docstring): forge v2 finals (random pick among the '
                      'length-compatible ones) vs held-out bench refs (closest audible length, no file / source '
                      'twice), equal-length trims (fade only where a clip is cut), made matched to real in level, '
                      'random order and A/B (real = A on half the trials)'},
              open(os.path.join(out, 'KEY_open_after.json'), 'w', encoding='utf-8'), indent=1)
    open(os.path.join(out, 'score_answers.py'), 'w', encoding='utf-8').write(SCORE_PY)
    from math import comb
    nt = len(key)
    pk = lambda k: sum(comb(nt, j) for j in range(k, nt + 1)) / 2 ** nt
    k05 = next(k for k in range(nt + 1) if pk(k) < 0.05)
    cats = sorted({k['category'] for k in key})
    L = ['Blind test: blind_test.wav (or blind_test_trials/tNN_A.wav + tNN_B.wav, one trial at a time).',
         'In every trial you hear A, then B. One is a REAL recording, the other was made entirely in code by earshot',
         'forge. Write down which one you think is REAL (A or B; ? if you cannot decide). Do not open KEY_open_after.json',
         'first. The two sounds of a trial are matched in loudness and trimmed to the same length (short fades),',
         'and the real recordings come from a held-out set the forge never tuned against. Silence between trials, no',
         'markers: each trial starts at the times below.',
         f'Then:  .venv\\Scripts\\python.exe {rel(out).replace("/", chr(92))}\\score_answers.py A B ...  '
         f'({nt} letters, in trial order)',
         f'By chance you would get about {nt // 2} of {nt} right; {k05} or more is unlikely by luck '
         f'(p = {pk(k05):.3f}).', '',
         f"{'trial':>5}  {'sound':<18} {'A starts':>8}  {'B starts':>8}   real is (A/B)"]
    for k in key:
        L.append(f"{k['trial']:>5}  {k['sound']:<18} {k['A_start']:8.2f}  {k['B_start']:8.2f}   ____")
    L += ['', f'total length {len(y) / SR:.1f} s ({len(y) / SR / 60:.1f} min), 48 kHz, 24-bit stereo; '
              f'{nt} trials over {len(cats)} sounds']
    open(os.path.join(out, 'answer_sheet.txt'), 'w', encoding='utf-8').write(chr(10).join(L) + chr(10))
    if not quiet:
        print(chr(10).join(L))
    return key


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
    gain = 10 ** (PEAK_MAX_DB / 20)
    picks = []
    for name, cat, v1dir, stem in V1:
        v1 = [f'{REPO}/{v1dir}/{stem}_01.wav', f'{REPO}/{v1dir}/{stem}_02.wav']
        cands = finals(cat)
        seq = []
        for f in v1:
            d, c = nearest_v2(f, cands)
            cands = [q for q in cands if q != c]
            seq += [('v1', f, None), ('v2', c, d)]
        lines.append(f'{name}  (v2 category {cat})')
        for ver, f, d in seq:
            x = load(f)
            x = x * (gain / (np.abs(x).max() + 1e-12))
            parts.append((t, x))
            extra = f'   (v2 final closest to the v1 final before it: distance {d:.1f})' if d is not None else ''
            lines.append(f'  {t:7.2f} s  {ver}  {rel(f)}{extra}')
            picks.append({'sound': name, 'version': ver, 'start': round(t, 2), 'file': rel(f)})
            t += len(x) / SR + 1.0
        t += 2.0
    total = t - 3.0 + TAIL
    y = place(parts, total)
    write(os.path.join(pdir, 'v1_vs_v2.wav'), y)
    head = ['v1_vs_v2.wav: the five sounds forge v1 had, before (v1) and after (v2), alternating v1, v2, v1, v2.',
            'Every sound peak-matched to -1 dBFS (so the after is not just louder); 1 s between sounds, 3 s between',
            'sound groups. Each v2 final is the one closest to the v1 final before it in 1/3-octave balance + audible',
            'length (distance = RMS dB over the bands + 6 dB per octave of length), so each pair is like for like.',
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
    a = ap.parse_args()
    out = os.path.abspath(a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out))
    pdir = a.progress or os.path.join(os.path.dirname(out), 'progress')
    pdir = os.path.abspath(pdir if os.path.isabs(pdir) else os.path.join(REPO, pdir))
    if a.only != 'progress':
        build_blind(out, a.n, a.seed)
    if a.only != 'blind':
        build_progress(pdir, a.results)


if __name__ == '__main__':
    main()
