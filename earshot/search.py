"""Find the best sound for a description: rank many files, or find the best moments inside one long recording.

    earshot rank "a deep thocky key press" sfx/keyboard/ --top 5
    earshot find typing.ogg "a single mechanical key press" --top 5
    earshot find meadow.ogg "a skylark singing" --kind texture --win 4
    from earshot.search import rank, find
    find('keys.ogg', 'a single key press')['segments'][0]     # {'start': 1.119, 'end': 1.734, 'z': 3.1, ...}

rank: every readable file is embedded once (cached; long files = mean of up to 6 evenly spaced 10 s windows) and sorted
by the calibrated z of the description (earshot/vocab.py), with the verdict of earshot.match. Unreadable files are
skipped (listed under 'skipped'), near-silent files are flagged and ranked last. vocab_percentile = where the query
ranks among the vocabulary descriptions for that file (not a rank among the files).
find: candidate segments are scored with CLAP and the best non-overlapping ones returned. Candidate kinds:
  stroke  - one onset until it has decayed by 35 dB (never into the next onset)
  event   - strokes that belong together: pairs (gap < 0.25 s: a key's down- and up-stroke; never led by a stroke 15 dB
            weaker than the rest) and longer chains, at most 2 s; never across an edit gap (>= 10 ms of exact digital
            silence: separate takes in a sample pack). Up to 60 s: at most 48 strokes / events, picked round-robin over
            2 s stretches (isolated, prominent ones first). Longer files: two stages - the windows are scored first,
            then the strokes / events inside the 6 best windows, so a strong moment anywhere in the file is found.
  window  - sliding windows covering the whole range with no gaps (default win = duration / 6 clamped to 1..5 s,
            hop = win / 2; long files grow the window up to 10 s, then the hop up to the window, to stay near 40).
            For one-shot queries each window's edges are snapped to the sound before scoring: a start
            inside a rising sound moves back to where it is quiet, a start inside the tail of an earlier sound moves
            forward past it, an end inside a decay moves on until it has decayed (each by at most 1 s), an end inside
            a new attack moves back before it. For in-between and continuous queries windows are plain stretches
            (crossfade them).
kind='auto' reads the query (CLAP text similarity to generic one-shot vs continuous phrases; thresholds from labelled
query texts, eval/semantics/kind.json): one-shot queries score all three kinds, never return a window that holds
two or more separate sounds (a quiet gap >= 0.25 s or an edit gap between them) and replace a winning window by the
best stroke / event inside it (+-50 ms) when that one scores within 0.5 z (a tighter trim of the same moment); clearly
continuous queries (rain, room tone, music) use windows only; anything in between scores all kinds and lets the
scores decide. kind='event' returns strokes / events only (windows only when the recording has no onsets).
start/end are sample trims in seconds: strokes / events start 5 ms before the measured attack, snapped windows 5 ms
before their sound and end 20 ms after it has decayed; head_db / tail_db say how loud the first 5 ms / last 10 ms
are against the segment's peak (tail -30 dB or lower = decayed cleanly; head or tail
above -15 dB = cut into a sound, flagged 'head cut' / 'tail cut' and ranked 0.5 z lower; plain windows of in-between
and continuous queries are not judged on their edges, they are cut anywhere by design). With --start/--end only segments whose attack lies inside the
range are returned (the audio is read from 0.1 s earlier so an attack at the boundary is seen whole).
Speed: every candidate costs one CLAP window (~0.3-1 s on CPU); candidates are query-independent and cached, so
more queries on the same file are fast.
"""
import os

import numpy as np

from . import audio, features, vocab
from .cli import common
from .match import SILENT_DBFS, VERDICT_EVIDENCE, VERDICTS, verdict
from .report import fmt_t

FRAME = 0.005
PRE_ROLL = 0.005
MARGIN = 0.1
CAP_S = 2.0
MERGE_GAP = 0.25
DECAY_DB = 35.0
MAX_EVENTS = 48
MAX_WINDOWS = 40
MAX_WIN_S = 10.0
SINGLE_STAGE_S = 60.0
TOP_WINDOWS = 6
BUCKET_S = 2.0
REFINE_DZ = 0.5
WEAK_DB = 15.0
CUT_DB = -15.0
TAIL_PENALTY = 0.5
EDIT_DB, EDIT_S = -100.0, 0.01   # an edit gap: >= 10 ms at or below -100 dBFS (exact digital silence between takes)
SNAP_S = 1.0                     # how far a window edge may move to reach the sound's start / decay
QUIET_DB = 40.0                  # 'quiet' = 40 dB under the window's peak (or 6 dB over the floor, if higher)
EVENT_ANCHORS = ['a single short sound effect', 'one short hit', 'a click', 'a single impact', 'one short pop',
                 'a quick one-shot sound']
TEXTURE_ANCHORS = ['a continuous sound', 'a long background atmosphere', 'a steady drone', 'a piece of music',
                   'ambient room noise', 'a soundscape']
# from the labelled tuning phrasings (eval/semantics/kind_queries.json "tune"): event = the lowest event score
# (rounded down), texture = the median texture score (only clearly continuous descriptions skip the onset candidates)
KIND_EVENT, KIND_TEXTURE = -0.08, -0.27


KIND_WORDS = {'event': 'read as a one-shot query', 'texture': 'read as a continuous-sound query (windows only)',
              'either': 'read as either a one-shot or a continuous sound'}


def _weakly(v):
    return 'not a match' if v == 'no match' else f'only a {v} match'


# ---------------------------------------------------------------------------------------------------- rank
def rank(paths, query, top=None):
    """files (or folders / globs) sorted by how well they match `query` (calibrated z); a given path that yields no
    media file raises audio.AudioError"""
    audio.check_paths(paths)
    files, bad = vocab.readable(audio.list_media(paths))
    skipped = f" Skipped {len(bad)} unreadable file(s): {', '.join(os.path.basename(b['path']) for b in bad[:3])}." \
        if bad else ''
    if not files:
        return {'summary': f'no readable media files found in {paths}.' + skipped, 'query': query, 'n_files': 0,
                'results': [], 'skipped': bad}
    v = vocab.get()
    A = vocab.embed_files(files)
    r = v.calibrate(A, v.embed_text([query]))
    z, cos, pct = np.round(r['z_q'][:, 0], 2), r['cos_q'][:, 0], r['pct_q'][:, 0]
    silent = np.array([vocab.peak_db(p) < SILENT_DBFS for p in files])
    order = sorted(range(len(files)), key=lambda i: (silent[i], -z[i]))
    res = []
    for i in order[:top] if top else order:
        bt = v.top_terms(A[i], k=1, exclude=('quality', 'production'))[0]
        res.append({'path': files[i], 'name': os.path.basename(files[i]), 'z': round(float(z[i]), 2),
                    'cosine': round(float(cos[i]), 3), 'vocab_percentile': round(100 * float(pct[i]), 1),
                    'verdict': 'no match' if silent[i] else verdict(float(z[i])), 'near_silent': bool(silent[i]),
                    'sounds_most_like': f'{bt[0]} (z {vocab.zf(bt[2])})'})
    counts = {n: sum(1 for i in order if not silent[i] and verdict(float(z[i])) == n) for _, n in VERDICTS}
    head = ', '.join(f"{x['name']} (z {vocab.zf(x['z'])}, {x['verdict']})" for x in res[:3] if not x['near_silent'])
    s = (f"Ranked {len(files)} files for '{query}'. Best: {head or 'none (all near-silent)'}. "
         f"{counts['strong']} strong, {counts['plausible']} plausible, {counts['weak']} weak match(es).")
    if res and not res[0]['near_silent'] and res[0]['verdict'] in ('weak', 'no match'):
        s += f" Even the best is {_weakly(res[0]['verdict'])}: it sounds more like {res[0]['sounds_most_like']}."
    if counts['strong'] > 1:
        ev = VERDICT_EVIDENCE['strong']
        s += (f" In a held-out test 'strong' was right {100 * ev['precision_library']:.0f}% of the time, "
              f"{100 * ev['precision_library_strict']:.0f}% when a close variant (a mouth pop for a cork pop) counts as wrong.")
    if silent.any():
        s += f' {int(silent.sum())} near-silent file(s) (peak < {SILENT_DBFS:.0f} dBFS) ranked last: their scores mean nothing.'
    return {'summary': s + skipped, 'query': query, 'n_files': len(files), 'results': res, 'skipped': bad}


# ---------------------------------------------------------------------------------------------------- find
def _env_db(x, frame=FRAME):
    w = max(1, int(round(frame * audio.SR)))
    return 20 * np.log10(np.maximum(features.frame_rms(x, w), 1e-12))


def _refine_start(env1, t, floor):
    """first 1 ms frame of the attack near onset t (s), so the trim does not clip it"""
    i0, i1 = max(0, int(t * 1000) - 30), min(len(env1), int(t * 1000) + 30)
    seg = env1[i0:i1]
    if not len(seg):
        return t
    k = int(np.argmax(seg >= max(floor + 10, seg.max() - 30)))
    return (i0 + k) / 1000


def query_kind(v, query):
    """('event' | 'texture' | 'either', score): CLAP text similarity to one-shot vs continuous anchor phrases"""
    q = v.embed_text([query])[0]
    d = float(q @ v.embed_text(EVENT_ANCHORS).mean(0) - q @ v.embed_text(TEXTURE_ANCHORS).mean(0))
    return ('event' if d >= KIND_EVENT else 'texture' if d <= KIND_TEXTURE else 'either'), round(d, 3)


def _tail_db(env, a, b):
    """level of the last 10 ms of [a, b) relative to the segment's peak (5 ms frames)"""
    seg = env[int(a / FRAME):max(int(a / FRAME) + 2, int(b / FRAME))]
    return float(seg[-2:].max() - seg.max()) if len(seg) else 0.0


def _head_db(env, a, b):
    """level of the first 5 ms frame of [a, b) (the pre-roll of a stroke / event) relative to the segment's peak"""
    seg = env[int(a / FRAME):max(int(a / FRAME) + 2, int(b / FRAME))]
    return float(seg[0] - seg.max()) if len(seg) else 0.0


def _edit_gaps(env, a, b):
    """True when [a, b) holds an edit gap: >= 10 ms of exact digital silence (5 ms frames)"""
    seg = (env[int(a / FRAME):int(b / FRAME)] <= EDIT_DB).astype(float)
    n = int(round(EDIT_S / FRAME))
    return bool(len(seg) >= n and np.convolve(seg, np.ones(n), 'valid').max() >= n)


def _quiet_level(env, floor, i0, i1):
    seg = env[i0:i1]
    return max(floor + 6.0, float(seg.max()) - QUIET_DB) if len(seg) else floor + 6.0


def snap_window(env, floor, a, b, lo, hi):
    """(a, b) of a window moved onto the sound's own edges (see the module docstring); lo / hi bound the search"""
    i0, i1 = int(round(a / FRAME)), max(int(round(a / FRAME)) + 1, int(round(b / FRAME)))
    i1 = min(i1, len(env))
    if i0 >= i1 - 1:
        return a, b
    q = _quiet_level(env, floor, i0, i1)
    q_end = max(floor + 4.0, float(env[i0:i1].max()) - DECAY_DB)
    w, n_lo, n_hi = 6, max(0, int(round(lo / FRAME))), min(len(env), int(round(hi / FRAME)))
    step = int(SNAP_S / FRAME)
    if env[i0] > q:
        falling = i0 - w >= n_lo and env[i0 - w:i0].mean() > env[i0:i0 + w].mean() + 1.0
        if falling:  # the window starts inside the tail of an earlier sound: start after that tail
            k = next((k for k in range(i0, min(i1 - 1, i0 + step)) if env[k] <= q), None)
            if k is not None and (i1 - k) * FRAME >= 0.1:
                i0 = k
        else:        # it starts inside a rising / sounding stretch: go back to where it is quiet
            k = next((k for k in range(i0, max(n_lo, i0 - step) - 1, -1) if env[k] <= q), None)
            if k is not None:
                i0 = k
    last = i1 - 1
    if env[last] > q_end:
        rising = last + 1 + w <= n_hi and env[last + 1:last + 1 + w].mean() > env[max(i0, last - w):last + 1].mean() + 1.0
        if rising:   # a new sound starts at the end: stop before its attack
            k = next((k for k in range(last, i0, -1) if env[k] <= q), None)
            if k is not None and (k - i0) * FRAME >= 0.1:
                i1 = k + 1
        else:        # still decaying (or sounding): follow it until it has decayed, never into a new attack
            k, low = last, env[last]
            while k + 1 < min(n_hi, last + step) and env[k] > q_end and env[k + 1] <= low + 6.0:
                k += 1
                low = min(low, env[k])
            i1 = k + 1
    seg = env[i0:i1]  # tighten: from one frame (5 ms) before the sound to its decay, no leading / trailing silence
    loud = np.nonzero(seg > q)[0]
    if len(loud):
        i0 = i0 + max(0, int(loud[0]) - 1)
        dec = np.nonzero(env[i0:i1] > q_end)[0]
        if len(dec):
            i1 = i0 + int(dec[-1]) + 1
    a2, b2 = i0 * FRAME, i1 * FRAME
    return max(lo, a2), min(hi, b2 + (0.02 if b2 < hi else 0.0))


def multi_sound(env, floor, a, b, gap=MERGE_GAP):
    """True when [a, b) holds two or more separate sounds: a quiet stretch >= gap, or an edit gap, with sound on both
    sides"""
    i0, i1 = int(round(a / FRAME)), int(round(b / FRAME))
    seg = env[i0:i1]
    if len(seg) < 3:
        return False
    q = _quiet_level(env, floor, i0, i1)
    loud = np.nonzero(seg > q)[0]
    if len(loud) < 2:
        return False
    if _edit_gaps(env, a + loud[0] * FRAME, a + (loud[-1] + 1) * FRAME):
        return True
    quiet = (seg[loud[0]:loud[-1] + 1] <= q).astype(float)
    n = int(round(gap / FRAME))
    return bool(len(quiet) >= n and np.convolve(quiet, np.ones(n), 'valid').max() >= n)


def _select(cands, n=MAX_EVENTS, bucket=BUCKET_S):
    """at most n candidates, round-robin over time buckets, best priority first within each bucket"""
    by = {}
    for c in sorted(cands, key=lambda c: -c[3]):
        by.setdefault(int(c[0] // bucket), []).append(c)
    out = []
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop(0))
    return out


def event_candidates(x, cap=CAP_S, n=MAX_EVENTS):
    """[(start, end, kind, priority)] onset-based segments of a mono 48 kHz array (priority = prominence over the
    noise floor in dB, +10 when the event is isolated: quiet before it and decayed by 25 dB at its end); at most n
    (round-robin over time), all of them when n is None"""
    dur = len(x) / audio.SR
    on = features.onsets(x)
    if not on:
        return []
    env = _env_db(x)
    env1 = _env_db(x, 0.001)
    floor = float(np.percentile(env, 10))
    strokes = []
    for i, o in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else dur
        f0, f1 = int(o / FRAME), max(int(o / FRAME) + 1, int(nxt / FRAME))
        seg = env[f0:f1]
        if not len(seg):
            continue
        pk = f0 + int(np.argmax(seg[:int(0.06 / FRAME) + 1]))
        below = np.nonzero(env[pk:f1] < max(env[pk] - DECAY_DB, floor + 4))[0]
        end = (pk + below[0]) * FRAME + 0.02 if len(below) else nxt
        st = max(0.0, _refine_start(env1, o, floor) - PRE_ROLL)
        if strokes:
            st = max(st, strokes[-1][1])
        end = min(end, nxt - 0.002 if i + 1 < len(on) else dur, st + cap)
        if end - st >= 0.02:
            strokes.append((st, end, float(env[pk] - floor)))
    out = [(s, e, 'stroke', p) for s, e, p in strokes]
    joins = [b[0] - a[1] < MERGE_GAP and not _edit_gaps(env, a[0], b[0]) for a, b in zip(strokes, strokes[1:])]
    for k, (a, b) in enumerate(zip(strokes, strokes[1:])):
        if joins[k] and b[1] - a[0] <= cap and a[2] >= b[2] - WEAK_DB:
            out.append((a[0], b[1], 'event', max(a[2], b[2])))
    i = 0
    while i < len(strokes):
        j = i
        while j + 1 < len(strokes) and joins[j] and strokes[j + 1][1] - strokes[i][0] <= cap:
            j += 1
        ch = strokes[i:j + 1]
        top = max(s[2] for s in ch)
        while ch and ch[0][2] < top - WEAK_DB:
            ch = ch[1:]
        if len(ch) > 2:
            out.append((ch[0][0], ch[-1][1], 'event', top))
        i = j + 1
    res = []
    for s, e, k, p in out:
        pre = env[max(0, int(s / FRAME) - 6):int(s / FRAME)]
        iso = (not len(pre) or pre.mean() <= floor + 10) and _tail_db(env, s, e) <= -25
        res.append((s, e, k, p + 10 * iso))
    return res if n is None else _select(res, n)


def window_candidates(dur, win=None, hop=None, t0=0.0):
    """sliding windows covering [t0, dur) without gaps; with defaults about MAX_WINDOWS of them at most (the window
    grows up to 10 s, then the hop up to the window length, before the count grows)"""
    span = dur - t0
    if not win:
        win = float(np.clip(span / 6, 1.0, 5.0))
        while win < MAX_WIN_S and (span - win) / (win / 2) + 1 > MAX_WINDOWS:
            win = min(MAX_WIN_S, win * 1.25)
    if span <= win:
        return [(t0, dur, 'window', 0.0)]
    if not hop:
        hop = win / 2
        if (span - win) / hop + 1 > MAX_WINDOWS:
            hop = min(win, (span - win) / (MAX_WINDOWS - 1))
    st = list(np.arange(t0, dur - win + 1e-6, hop))
    if dur - win - st[-1] > 1e-3:
        st.append(dur - win)
    return [(float(s), float(s + win), 'window', 0.0) for s in st]


def _dedupe(cands, iou=0.9):
    out = []
    for c in cands:
        if all(min(c[1], d[1]) - max(c[0], d[0]) < iou * (max(c[1], d[1]) - min(c[0], d[0])) for d in out):
            out.append(c)
    return out


def _score(v, path, m0, cands, Qe):
    """(z, cosine) of every candidate (local times + m0 = file times)"""
    if not cands:
        return np.zeros(0), np.zeros(0)
    r = v.calibrate(vocab.embed_spans([(path, m0 + a, m0 + b) for a, b, _, _ in cands]), Qe)
    return r['z_q'][:, 0], r['cos_q'][:, 0]


def find(path, query, win=None, hop=None, top=5, start=None, end=None, kind='auto'):
    """best non-overlapping segments of a recording for `query`; start/end are sample trims in seconds"""
    if not query or not query.strip():
        raise ValueError('no query given')
    full = vocab.check(path, start, end)
    s0, s1 = start or 0.0, min(end, full) if end is not None else full
    m0 = max(0.0, s0 - MARGIN)
    x = audio.load(path, mono=True, start=m0, end=s1)
    off, dur = s0 - m0, len(x) / audio.SR
    v = vocab.get()
    Qe = v.embed_text([query])
    qkind, qscore = query_kind(v, query)
    use_ev = kind == 'event' or (kind == 'auto' and qkind != 'texture')
    prefer_ev = kind == 'event' or (kind == 'auto' and qkind == 'event')
    env = _env_db(x)
    floor = float(np.percentile(env, 10))
    wins = window_candidates(dur, win, hop, off)
    n_win, win_s = len(wins), wins[0][1] - wins[0][0]
    if prefer_ev:  # one-shot queries: window edges on the sound, not on a fixed grid
        wins = _dedupe([snap_window(env, floor, a, b, off, dur) + (k, p) for a, b, k, p in wins])
    zw, cw = _score(v, path, m0, wins, Qe)
    evs, stage = [], 'one stage'
    if use_ev:
        allev = [c for c in event_candidates(x, n=None) if c[0] + PRE_ROLL >= off - 1e-3]
        if s1 - s0 <= SINGLE_STAGE_S:
            evs = _select(allev)
        else:
            best = [wins[i] for i in np.argsort(-zw)[:TOP_WINDOWS]]
            evs = _select([c for c in allev if any(c[0] < b and c[1] > a for a, b, _, _ in best)])
            stage = f'two stages: events inside the {len(best)} best windows'
    evs = _dedupe(evs)
    ze, ce = _score(v, path, m0, evs, Qe)
    if kind == 'event' and evs:  # onset-based segments only
        wins, zw, cw = [], zw[:0], cw[:0]
    cands = evs + wins
    z, cos = np.round(np.concatenate([ze, zw]), 2), np.concatenate([ce, cw])
    tails = np.array([_tail_db(env, a, b) for a, b, _, _ in cands])
    heads = np.array([_head_db(env, a, b) for a, b, _, _ in cands])
    judged = np.array([use_ev and (prefer_ev or c[2] != 'window') for c in cands], bool)  # whose edges must be clean
    cut = ((tails > CUT_DB) | (heads > CUT_DB)) & judged
    score = z - TAIL_PENALTY * cut
    rank_z = score.copy()
    if prefer_ev:
        for i, (a, b, k, _) in enumerate(cands):
            if k != 'window':
                continue
            inside = [z[j] for j, c in enumerate(cands) if c[2] != 'window' and c[0] >= a - 0.05 and c[1] <= b + 0.05]
            if (inside and max(inside) >= z[i] - REFINE_DZ) or (evs and multi_sound(env, floor, a, b)):
                rank_z[i] = -np.inf
    xs = audio.load(path, start=m0, end=s1)
    loud_file = audio.db(np.abs(xs).max() if len(xs) else 0.0) >= SILENT_DBFS
    picked = []
    for i in np.argsort(-rank_z):
        if not np.isfinite(rank_z[i]):
            break
        a, b = float(cands[i][0]), float(cands[i][1])
        if all(b <= c['_a'] + 1e-3 or a >= c['_b'] - 1e-3 for c in picked):
            seg = xs[int(a * audio.SR):int(b * audio.SR)]
            if loud_file and audio.db(np.abs(seg).max() if len(seg) else 0) < SILENT_DBFS:
                continue  # a near-silent stretch of a sounding file is never an answer
            picked.append({'_a': a, '_b': b, 'start': round(m0 + a, 3), 'end': round(m0 + b, 3),
                           'duration': round(b - a, 3), 'score': round(float(score[i]), 2), 'z': round(float(z[i]), 2),
                           'cosine': round(float(cos[i]), 3), 'verdict': verdict(float(z[i])), 'kind': cands[i][2],
                           'peak_dbfs': round(audio.db(np.abs(seg).max() if len(seg) else 0), 1),
                           'head_db': round(float(heads[i]), 1), 'tail_db': round(float(tails[i]), 1),
                           'cut': ', '.join(w for w, c in (('head cut', heads[i] > CUT_DB), ('tail cut', tails[i] > CUT_DB))
                                            if c and judged[i])})
        if len(picked) >= top:
            break
    for p in picked:
        del p['_a'], p['_b']
    rng = f'{fmt_t(s0)}-{fmt_t(s1)} ({s1 - s0:.1f} s)' if start is not None or end is not None else f'all {s1 - s0:.1f} s'
    nw = len(wins) if wins else n_win
    how = (f"{KIND_WORDS[qkind] if kind == 'auto' else 'kind=' + kind}; searched {rng}: {nw} windows of ~{win_s:.1f} s"
           + (' (edges snapped to the sound)' if prefer_ev else '') + (' to locate events, none returned' if not wins else '')
           + (f" + {len(evs)} onset-based candidates ({stage})" if use_ev else ''))
    if picked:
        segs = '; '.join(f"{fmt_t(p['start'])}-{fmt_t(p['end'])} ({p['duration']:.2f} s {p['kind']}, z {vocab.zf(p['z'])} "
                         f"{p['verdict']}" + (f", {p['cut']}" if p['cut'] else '') + ')' for p in picked)
        s = (f"Best moments for '{query}' in {os.path.basename(path)} ({how}): {segs}. start/end are sample trims in "
             'seconds (strokes / events start 5 ms before the attack' +
             ('; windows are snapped to the sound' if prefer_ev else '; windows are plain stretches: crossfade them') +
             "; a 'head cut' / 'tail cut' trim cuts into a sound).")
        if picked[0]['verdict'] in ('weak', 'no match'):
            s += f" Even the best segment is {_weakly(picked[0]['verdict'])}: the recording may not contain this sound."
    else:
        s = f'No candidate segments in {os.path.basename(path)} ({how}).'
    pk = audio.db(np.abs(xs).max() if len(xs) else 0.0)
    if pk < SILENT_DBFS:
        s = f'WARNING: near-silent input (peak {pk:.0f} dBFS), scores are meaningless. ' + s
    return {'summary': s, 'path': path, 'query': query, 'query_kind': qkind, 'query_kind_score': qscore,
            'duration': round(s1 - s0, 3), 'n_candidates': len(cands), 'n_windows': len(wins), 'n_events': len(evs),
            'segments': picked}


# ---------------------------------------------------------------------------------------------------- cli
def _label(x, dup):
    """file name, with its parent folder when another result has the same name"""
    return os.path.join(os.path.basename(os.path.dirname(x['path'])), x['name']) if x['name'] in dup else x['name']


def _rank_text(r):
    lines = [r['summary'], '']
    names = [x['name'] for x in r['results']]
    dup = {n for n in names if names.count(n) > 1}
    for i, x in enumerate(r['results'], 1):
        lines.append(f"{i:3d}. z {vocab.zf(x['z']):>5}  {x['verdict']:<9} cos {x['cosine']:.3f}  {_label(x, dup)}  "
                     f"(sounds most like {x['sounds_most_like']})" + ('  NEAR-SILENT' if x['near_silent'] else ''))
    return '\n'.join(lines)


def _find_text(r):
    lines = [r['summary'], '']
    for i, p in enumerate(r['segments'], 1):
        lines.append(f"{i}. {p['start']:.3f}-{p['end']:.3f} s ({fmt_t(p['start'])}, {p['duration']:.3f} s)  z {vocab.zf(p['z'])} "
                     f"{p['verdict']:<9} {p['kind']:<6} peak {p['peak_dbfs']:.1f} dBFS  head {p['head_db']:.0f} dB  "
                     f"tail {p['tail_db']:.0f} dB" + (f"  (ranked {vocab.zf(p['score'])}: {p['cut']})" if p.get('cut') else ''))
    return '\n'.join(lines)


def add_cli(sub):
    p = sub.add_parser('rank', help='rank sound files by how well they match a description')
    p.add_argument('query')
    p.add_argument('paths', nargs='+', help='files, folders or globs')
    p.add_argument('--top', type=int, help='list only the best N files (default: all)')
    common(p)
    p.set_defaults(func=lambda a: vocab.run_cli(lambda: rank(a.paths, a.query, a.top), a, _rank_text))
    p = sub.add_parser('find', help='find the best moments for a description inside a long recording (sample trims)')
    p.add_argument('path')
    p.add_argument('query')
    p.add_argument('--top', type=int, default=5, help='number of non-overlapping segments (default 5)')
    p.add_argument('--win', type=float, help='window length in s (default: duration/6 clamped to 1..5)')
    p.add_argument('--hop', type=float, help='hop between windows in s (default: win/2)')
    p.add_argument('--kind', choices=['auto', 'event', 'texture'], default='auto',
                   help='event = onset-based strokes / events only (hits, clicks, steps); texture = sliding windows '
                        'only (beds, ambiences); auto = read the query and score both')
    p.add_argument('--start', type=float, help='search from this time (s)')
    p.add_argument('--end', type=float, help='search up to this time (s)')
    common(p)
    p.set_defaults(func=lambda a: vocab.run_cli(
        lambda: find(a.path, a.query, a.win, a.hop, a.top, a.start, a.end, a.kind), a, _find_text))
