"""Launch-film sound profile: what professional launch films do with sound, measured the same way for every film.

    earshot profile refs/ more_ref.mp4 -o my_profile.json [--styles styles.json] [--no-ast]

    from earshot.profile import film_metrics, profile, load_profile
    f = film_metrics('film.mp4')              # one film: metrics + event lists (cached under EARSHOT_HOME/emb/launch)
    f = film_metrics('mix.wav', video='film.mp4')
    p = profile(['a.mp4', 'b.mp4'], styles=['UI walkthrough', 'kinetic type'])   # median, p10-p90, min-max per metric
    load_profile()                            # the bundled profile of 7 professional launch films (data/launch_profile.json)

Per-film metrics (keys of METRICS; None when a film is too short or has no picture):
  loudness  lufs_i (BS.1770 integrated), true_peak (dBTP, 4x oversampled), lra (LU), crest_lu (loudest 3 s minus
            integrated)
  arc       arc (11 points: loudness at 0 %, 10 % .. 100 % of the film in LU vs integrated; shown, not scored),
            intro_lu / end_lu (energy mean of the first / last 10 % of the film relative to integrated), peak_pos
            (where the loudest 3 s sit, 0 = start, 1 = end), first_sound_s (first 100 ms block within 20 LU of the
            integrated level), max_gap_s (longest stretch between first and last sound where the 100 ms level stays
            30 LU below integrated)
  ending    end_silence_s (dead air after the last block within 30 LU of integrated), fade_s (last moment at full
            level -> end of sound; < 0.5 s = hard out), final_lu (loudest 0.4 s of the last 3 s of sound relative to
            integrated: is there a final hit?), tail_s (strongest onset of the last 4 s -> end of sound, when it is a
            real hit: how long the final sting rings out; None = no final hit)
  music     music_frac (share of 5 s windows with AudioSet 'Music' >= 0.3: music under voice or SFX scores 0.4-0.7,
            a bare music track ~0.9, no music ~0.1), speech_frac (share with 'Speech' >= 0.5), tempo (BPM)
  density   onset_rate (onsets / s, librosa spectral flux), hit_rate (salient hits / s: the 100 ms loudness jumps
            >= 6 dB within 80 ms to within 8 LU of the integrated level), stopdowns_min (dips >= 10 dB below the
            preceding 2 s for 0.15-1.5 s that snap back to full level within 0.3 s, per minute)
  sync      cut_rate (visual events / s: hard cuts and big picture changes, see below),
            cuts_with_sound (share of visual events with an onset within +-100 ms), cut_hit_frac (salient hit
            within +-100 ms), cut_frame_sync (onset within +-1 frame), cuts_on_beat (within +-70 ms of the tracked
            beat). Each sync share also has a 'chance' value: what the same events shifted off the picture score.
  stereo    side_to_mid_db (side vs mid energy: higher = wider; identical channels = a mono mix count as -40 dB),
            correlation (L/R)
  spectrum  band_sub .. band_air (energy share per band in dB over the whole file: sub 20-60, low 60-250, lowmid 250-1k,
            mid 1-4k, presence 4-8k, air 8-16k Hz; > 16 kHz is left out: AAC web encodes cut it), brightness_hz
            (features.brightness_hz: median magnitude-weighted spectral centroid)
Files shorter than 10 s get no loudness-shape metrics (lra, crest_lu, intro_lu, end_lu, final_lu): on a short clip
they measure the clip, not a film. profile() skips files that are missing, have no audio, are shorter than 3 s or
silent (with a warning) and refuses an empty result.

Visual events are earshot.video.cuts' frame-difference detector (video.frame_diff + video.diff_events: 96x54 grey
frames, a peak >= 4x the median of the surrounding second or a change of >= 15 % mean grey level), which also catches
cuts on white-background motion graphics where ffmpeg's scene score finds almost nothing (eval/launch/RESULTS.md).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

from . import audio, config, features
from .cli import common
from .report import emit, error_line
from .video import diff_events, frame_diff

VERSION = 2
AUDIO_V = 'a3'
PROFILE_PATH = os.path.join(config.DATA_DIR, 'launch_profile.json')
AST_WIN = 5.0
MUSIC_P = 0.3
HR = 22050
HOP = 256
E2 = 0.002
MIN_S, SHORT_S = 3.0, 10.0
MONO_DB = -40.0
SHAPE = ('lra', 'crest_lu', 'intro_lu', 'end_lu', 'final_lu')
SHIFTS = (0.23, 0.31, 0.43, 0.57, 0.71, 0.87, 1.13, 1.37, -0.29, -0.41, -0.53, -0.67, -0.79, -0.97, -1.19, -1.43)

# key: (group, label, unit, decimals)
METRICS = {
    'lufs_i': ('loudness', 'integrated loudness', 'LUFS', 1),
    'true_peak': ('loudness', 'true peak', 'dBTP', 1),
    'lra': ('loudness', 'loudness range', 'LU', 1),
    'crest_lu': ('loudness', 'loudest 3 s above integrated', 'LU', 1),
    'intro_lu': ('arc', 'opening level (first 10%)', 'LU', 1),
    'end_lu': ('arc', 'closing level (last 10%)', 'LU', 1),
    'peak_pos': ('arc', 'position of the loudest 3 s', '0-1', 2),
    'first_sound_s': ('arc', 'time to first sound', 's', 2),
    'max_gap_s': ('arc', 'longest silent gap inside', 's', 2),
    'end_silence_s': ('ending', 'dead air at the end', 's', 2),
    'fade_s': ('ending', 'final fade / ring-out length', 's', 2),
    'final_lu': ('ending', 'final moment vs integrated', 'LU', 1),
    'tail_s': ('ending', 'final hit ring-out', 's', 2),
    'music_frac': ('music', 'music presence', 'share', 2),
    'speech_frac': ('music', 'voice / speech presence', 'share', 2),
    'tempo': ('music', 'tempo', 'BPM', 0),
    'onset_rate': ('density', 'onset density', '/s', 2),
    'hit_rate': ('density', 'salient hit density', '/s', 2),
    'stopdowns_min': ('density', 'stop-downs', '/min', 1),
    'cut_rate': ('sync', 'visual events (cuts, big changes)', '/s', 2),
    'cuts_with_sound': ('sync', 'onset on visual events +-100ms', 'share', 2),
    'cut_hit_frac': ('sync', 'visual events with a salient hit +-100 ms', 'share', 2),
    'cut_frame_sync': ('sync', 'visual events with an onset +-1 frame', 'share', 2),
    'cuts_on_beat': ('sync', 'visual events on the beat grid +-70 ms', 'share', 2),
    'band_sub': ('spectrum', 'sub 20-60 Hz share', 'dB', 1),
    'band_low': ('spectrum', 'low 60-250 Hz share', 'dB', 1),
    'band_lowmid': ('spectrum', 'low-mid 250-1k share', 'dB', 1),
    'band_mid': ('spectrum', 'mid 1-4 kHz share', 'dB', 1),
    'band_presence': ('spectrum', 'presence 4-8 kHz share', 'dB', 1),
    'band_air': ('spectrum', 'air 8-16 kHz share', 'dB', 1),
    'brightness_hz': ('spectrum', 'brightness (median spectral centroid)', 'Hz', 0),
    'side_to_mid_db': ('stereo', 'stereo width (side vs mid)', 'dB', 1),
    'correlation': ('stereo', 'L/R correlation', 'r', 2),
}


class InputError(ValueError):
    """a file that cannot be measured: missing, a folder, no audio / picture stream"""


def media_problem(path, need='audio'):
    """None when `path` is a media file with an audio (need='audio') or picture (need='video') stream, else why not"""
    if not os.path.exists(path):
        return f'file not found: {path}'
    if os.path.isdir(path):
        return f'{path} is a folder, not a media file'
    pr = audio.probe(path)
    if pr.get('probe_error') or not (pr.get('codec') or pr.get('has_video')):
        return f'{path} is not a readable audio or video file'
    if need == 'audio' and not pr.get('codec'):
        return f'{path} has no audio track'
    if need == 'video' and not pr.get('has_video'):
        return f'{path} has no picture (no video stream)'
    return None


def unscorable(path):
    """(duration, reason) - reason is None unless the sound is shorter than MIN_S or silent (cheap: no librosa)"""
    if os.path.exists(cache_paths(path)['audio']):
        m = _audio_pass(path)['meas']
        dur, lufs = float(m['duration']), m.get('lufs', -70.0)
    else:
        x = audio.load(path)
        dur = len(x) / audio.SR
        lufs = features.integrated_lufs(x) if dur >= MIN_S else -70.0
    if dur < MIN_S:
        return dur, f'is shorter than {MIN_S:.0f} s ({dur:.2f} s)'
    if lufs is None or lufs <= -69.5:
        return dur, 'is silent (nothing above -70 LUFS)'
    return dur, None


def _cache_file(kind, *ids):
    d = os.path.join(config.EMB_CACHE, 'launch')
    os.makedirs(d, exist_ok=True)
    k = hashlib.sha1('|'.join(map(str, ids)).encode()).hexdigest()[:20]
    return os.path.join(d, f'{kind}_{k}')


def cache_paths(path):
    """this module's cache files for one media file (audio pass, AudioSet pass); the audio key follows the core decode
    version, so a decoder change re-measures; the picture pass is cached by earshot.video.frame_diff"""
    return {'audio': _cache_file('audio', audio._key(path, audio.SR, 2), AUDIO_V) + '.npz',
            'ast': _cache_file('ast', audio.file_id(path), AST_WIN, 1) + '.npz'}


def _energy_mean(L):
    L = np.asarray(L, np.float64)
    return float(10 * np.log10(np.mean(10 ** (L / 10)))) if len(L) else None


def _audio_pass(path):
    """light measurements + onset envelope, onsets, beats and loudness curves (cached)"""
    f = cache_paths(path)['audio']
    if os.path.exists(f):
        try:
            z = np.load(f, allow_pickle=False)
            r = {k: z[k] for k in z.files if k != 'meas'}
            r['meas'] = json.loads(str(z['meas']))
            return r
        except Exception:
            pass
    import librosa
    x = audio.load(path)
    meas = features.measure(x, light=True)
    y = audio.resample(x.mean(axis=1), audio.SR, HR)
    env = librosa.onset.onset_strength(y=y, sr=HR, hop_length=HOP)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=HR, hop_length=HOP, units='time')
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=HR, hop_length=HOP, units='time')
    tf, Lf = features.loudness_curve(x, 0.1, 0.02)
    tm, Lm = features.loudness_curve(x, 0.4, 0.1)
    ts, Ls = features.loudness_curve(x, 3.0, 0.1)
    e2 = 20 * np.log10(features.frame_rms(x.mean(axis=1), int(E2 * audio.SR)) + 1e-9).astype(np.float32)
    r = dict(env=env.astype(np.float32), e2=e2, onsets=attack_times(on, e2), beats=np.asarray(beats, np.float64),
             tempo=np.atleast_1d(np.asarray(tempo, np.float64))[:1], tf=tf, Lf=Lf, tm=tm, Lm=Lm, ts=ts, Ls=Ls)
    tmp = f'{f}.{os.getpid()}.tmp.npz'
    np.savez(tmp, meas=json.dumps(meas), **r)
    audio.replace_quietly(tmp, f)
    r['meas'] = meas
    return r


def attack_times(times, e2, h=E2):
    """move onset / hit times to the attack: the steepest rise of the 2 ms level within 60 ms before .. 20 ms after
    (spectral-flux peaks lag the attack by 10-30 ms; an editor aligns on the attack)"""
    out = []
    for t in np.asarray(times, np.float64):
        a, b = max(1, int((t - 0.06) / h)), min(len(e2) - 1, int((t + 0.02) / h) + 1)
        if b - a < 2:
            out.append(float(t))
            continue
        d = e2[a:b + 1] - e2[a - 1:b]
        out.append(round((a + int(np.argmax(d))) * h, 4))
    return np.array(out, np.float64)


def _ast_pass(path):
    """AudioSet probabilities per 5 s window (cached): (window starts, (n, 527) float16)"""
    f = cache_paths(path)['ast']
    if os.path.exists(f):
        try:
            z = np.load(f, allow_pickle=False)
            return z['t'], z['p'].astype(np.float32), [str(v) for v in z['labels']]
        except Exception:
            pass
    from .models import ast
    A = ast.get()
    x = audio.load(path, mono=True)
    w = int(AST_WIN * audio.SR)
    starts = [s for s in range(0, len(x), w) if len(x) - s >= audio.SR or s == 0]
    p = A.probs([x[s:s + w] for s in starts])
    t = np.array(starts, np.float64) / audio.SR
    tmp = f'{f}.{os.getpid()}.tmp.npz'
    np.savez(tmp, t=t, p=p.astype(np.float16), labels=np.array(A.labels))
    audio.replace_quietly(tmp, f)
    return t, p, A.labels


def _windows_min_max(L, a, b):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    n = len(L)
    pre = sw(np.concatenate([np.full(a, L[0]), L]), a).min(axis=1)[:n]
    post = sw(np.concatenate([L, np.full(b - 1, L[-1])]), b).max(axis=1)[:n]
    return pre, post


def salient_hits(tf, Lf, lufs, env, e2=None, jump=6.0, rel=-8.0):
    """times of salient hits: the 100 ms loudness jumps >= jump dB within 80 ms (vs the quietest block of the 200 ms
    before) and lands within `rel` LU of the integrated level; one hit per run of such blocks, timed at the
    strongest onset-envelope frame of the run, moved to its attack when the 2 ms level `e2` is given"""
    if len(Lf) < 20:
        return np.zeros(0)
    h = float(tf[1] - tf[0])
    pre, post = _windows_min_max(Lf, int(round(0.2 / h)), int(round(0.08 / h)))
    idx = np.where((post - pre >= jump) & (post >= lufs + rel))[0]
    if not len(idx):
        return np.zeros(0)
    te = np.arange(len(env)) * HOP / HR
    out = []
    for run in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1):
        sel = (te >= tf[run[0]] - 0.05) & (te <= tf[run[-1]] + 0.1)
        t = float(te[sel][int(np.argmax(env[sel]))]) if sel.any() else float(tf[run[0]]) + 0.03
        if not out or t - out[-1] > 0.1:
            out.append(t)
    return attack_times(out, e2) if e2 is not None else np.array(out)


def stopdowns(tf, Lf, drop=10.0, back=3.0, min_s=0.15, max_s=1.5):
    """[(start, length, depth_db)] of short dips before a hit: >= drop dB below the median of the preceding 2 s for
    min_s..max_s, back to within `back` dB of that median within 0.3 s"""
    if len(Lf) < 3:
        return []
    h = float(tf[1] - tf[0])
    w = int(round(2.0 / h))
    if len(Lf) <= w:
        return []
    from numpy.lib.stride_tricks import sliding_window_view as sw
    ref = np.concatenate([np.full(w, np.nan), np.median(sw(Lf, w), axis=1)[:len(Lf) - w]])
    out, i, n = [], w, len(Lf)
    while i < n:
        r = ref[i]
        if Lf[i] < r - drop:
            j = i
            while j < n and Lf[j] < r - drop:
                j += 1
            ln = (j - i) * h
            if min_s <= ln <= max_s and j < n and Lf[j: j + int(round(0.3 / h)) + 1].max() >= r - back:
                out.append((round(float(tf[i]) - 0.05, 3), round(ln, 3), round(float(r - Lf[i:j].min()), 1)))
            i = j + int(round(0.5 / h))
        else:
            i += 1
    return out


def near_frac(events, targets, tol):
    """share of events with a target within +-tol seconds"""
    events, targets = np.asarray(events, float), np.sort(np.asarray(targets, float))
    if not len(events):
        return None
    if not len(targets):
        return 0.0
    i = np.searchsorted(targets, events)
    d = np.minimum(np.abs(targets[np.clip(i, 0, len(targets) - 1)] - events),
                   np.abs(targets[np.clip(i - 1, 0, len(targets) - 1)] - events))
    return float(np.mean(d <= tol))


def chance_frac(events, targets, tol, dur):
    """the same share for the events shifted off the picture (circularly, by fixed offsets): what chance scores"""
    events = np.asarray(events, float)
    if not len(events) or dur <= 0:
        return None
    return float(np.mean([near_frac((events + s) % dur, targets, tol) for s in SHIFTS]))


def _edge(e2, t, first=True, h=E2):
    """refine the start (first=True) or end of sound found with a 100 ms block centred at t: the first / last 2 ms
    frame of that block within 20 dB of the loudest 2 ms of the next / previous 150 ms"""
    a, b = max(0, int((t - 0.05) / h)), min(len(e2), int((t + 0.05) / h) + 1)
    if b <= a:
        return max(0.0, t)
    ref = e2[a:min(len(e2), a + int(0.15 / h))].max() if first else e2[max(0, b - int(0.15 / h)):b].max()
    ok = np.where(e2[a:b] >= ref - 20)[0]
    if not len(ok):
        return max(0.0, t)
    return round((a + (ok[0] if first else ok[-1] + 1)) * h, 3)


def _arc(tm, Lm, I, dur):
    """loudness arc: energy-mean momentary loudness around 0 %, 10 % .. 100 % of the duration, in LU vs integrated"""
    w = max(1.5, 0.05 * dur)
    out = []
    for k in range(11):
        sel = np.abs(tm - k / 10 * dur) <= w
        out.append(round(_energy_mean(Lm[sel]) - I, 1) if sel.any() else None)
    return out


def _longest_run(mask):
    best = cur = 0
    for b in mask:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def _final_hit_tail(env, onsets, t_end, look=4.0):
    """seconds from the strongest onset of the last `look` s of sound to the end of sound, if that onset is a real
    hit (onset strength in the film's top quarter); None when the film just stops or fades without one"""
    if not len(onsets):
        return None
    st = env[np.clip(np.round(onsets * HR / HOP).astype(int), 0, len(env) - 1)]
    sel = np.where((onsets >= t_end - look) & (onsets < t_end))[0]
    if not len(sel):
        return None
    j = sel[int(np.argmax(st[sel]))]
    return t_end - float(onsets[j]) if st[j] >= np.percentile(st, 75) else None


def _r(v, nd):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), nd)


def film_metrics(path, video=None, use_ast=True):
    """launch metrics + event lists for one soundtrack or film (picture from `video`, else from `path` if it has one)"""
    t0 = time.time()
    a = _audio_pass(path)
    timing = {'audio': round(time.time() - t0, 1)}
    m = a['meas']
    dur = float(m['duration'])
    I = m.get('lufs', -70.0)
    tf, Lf, tm, Lm, ts, Ls = a['tf'], a['Lf'], a['tm'], a['Lm'], a['ts'], a['Ls']
    onsets, beats = a['onsets'], a['beats']
    hits = salient_hits(tf, Lf, I, a['env'], a['e2'])
    sd = stopdowns(tf, Lf)
    v = {k: None for k in METRICS}
    mono = m.get('side_to_mid_db') is None and I > -69.5
    v.update(lufs_i=m.get('lufs'), true_peak=m.get('true_peak_dbtp'), lra=m.get('lra_lu'), correlation=m.get('correlation'),
             side_to_mid_db=MONO_DB if mono else m.get('side_to_mid_db'), brightness_hz=m.get('brightness_hz'))
    for b in ('sub', 'low', 'lowmid', 'mid', 'presence', 'air'):
        v['band_' + b] = (m.get('bands_db') or {}).get(b)
    if m.get('short_term_max_lufs') is not None and m.get('lufs') is not None:
        v['crest_lu'] = m['short_term_max_lufs'] - m['lufs']
    if len(Lm) > 2:
        v['intro_lu'] = _energy_mean(Lm[tm <= max(0.1 * dur, 0.5)]) - I
        v['end_lu'] = _energy_mean(Lm[tm >= dur - max(0.1 * dur, 0.5)]) - I
    if len(Ls) > 1 and dur >= 6:
        v['peak_pos'] = float(ts[int(np.argmax(Ls))]) / dur
    arc = _arc(tm, Lm, I, dur) if len(Lm) > 2 else None
    loud = np.where(Lf > I - 20)[0]
    audible = np.where(Lf > I - 30)[0]
    t_end = dur
    if len(loud):
        v['first_sound_s'] = _edge(a['e2'], float(tf[loud[0]]), first=True)
    if len(audible):
        t_end = min(dur, _edge(a['e2'], float(tf[audible[-1]]), first=False))
        v['end_silence_s'] = max(0.0, dur - t_end)
        full = np.where((Lm >= I - 6) & (tm <= t_end))[0]
        if len(full):
            v['fade_s'] = max(0.0, t_end - float(tm[full[-1]]) - 0.2)
        inside = (tf >= (v['first_sound_s'] or 0.0)) & (tf <= t_end)
        v['max_gap_s'] = _longest_run(Lf[inside] < I - 30) * float(tf[1] - tf[0]) if inside.sum() > 1 else 0.0
        last = (tm >= t_end - 3.0) & (tm <= t_end)
        if last.any():
            v['final_lu'] = float(Lm[last].max()) - I
        v['tail_s'] = _final_hit_tail(a['env'], onsets, t_end)
    v['onset_rate'] = len(onsets) / dur if dur > 0 else None
    v['hit_rate'] = len(hits) / dur if dur > 0 else None
    v['stopdowns_min'] = len(sd) / dur * 60 if dur >= 6 else None
    v['tempo'] = float(a['tempo'][0]) if len(beats) > 4 else None
    if dur < SHORT_S:
        v.update({k: None for k in SHAPE})
    music = None
    if use_ast:
        t1 = time.time()
        tw, P, labels = _ast_pass(path)
        timing['audioset'] = round(time.time() - t1, 1)
        ix = {l: i for i, l in enumerate(labels)}
        music = [(round(float(t), 2), round(float(P[i, ix['Music']]), 2), round(float(P[i, ix['Speech']]), 2))
                 for i, t in enumerate(tw)]
        v['music_frac'] = float(np.mean(P[:, ix['Music']] >= MUSIC_P))
        v['speech_frac'] = float(np.mean(P[:, ix['Speech']] >= 0.5))
    pr = audio.probe(path)
    vid = video or (path if pr.get('has_video') else None)
    ev_t, ev_d, chance, fps = [], [], {}, None
    if vid:
        t1 = time.time()
        fps, d = frame_diff(vid)
        timing['picture'] = round(time.time() - t1, 1)
        ev = diff_events(d, fps)
        ev_t = [round(float(i / fps), 3) for i in ev if i / fps < dur]
        ev_d = [round(float(d[i]), 3) for i in ev if i / fps < dur]
        v['cut_rate'] = len(ev_t) / dur
        if len(ev_t) >= 3:
            tol_f = 1.0 / fps + 0.006
            for key, targets, tol in (('cuts_with_sound', onsets, 0.1), ('cut_hit_frac', hits, 0.1),
                                      ('cut_frame_sync', onsets, tol_f), ('cuts_on_beat', beats, 0.07)):
                v[key] = near_frac(ev_t, targets, tol)
                chance[key] = _r(chance_frac(ev_t, targets, tol, dur), 2)
    vals = {k: _r(x, METRICS[k][3] if METRICS[k][3] else 0) for k, x in v.items()}
    for k in ('tempo', 'brightness_hz'):
        if vals[k] is not None:
            vals[k] = int(vals[k])
    res = {'path': path, 'video': vid, 'duration': round(dur, 3), 'fps': fps, 'codec': pr.get('codec'),
           'lossy': bool(pr.get('lossy')), 'mono': mono, 'short': dur < SHORT_S, 'metrics': vals, 'chance': chance, 'arc': arc,
           'events': {'hits': [round(float(t), 3) for t in hits], 'visual': ev_t, 'visual_strength': ev_d,
                      'stopdowns': sd, 'beats': [round(float(b), 3) for b in beats], 'music_windows': music,
                      'onsets_n': len(onsets)},
           'seconds': round(time.time() - t0, 1), 'timing': timing}
    res['summary'] = _film_summary(res)
    return res


def _film_summary(r):
    m = r['metrics']
    s = f"{os.path.basename(r['path'])}: {r['duration']:.1f} s, {m['lufs_i']} LUFS, true peak {m['true_peak']} dBTP"
    if m['lra'] is not None:
        s += f", LRA {m['lra']} LU"
    if m['music_frac'] is not None:
        s += f", music in {100 * m['music_frac']:.0f}% of 5 s windows"
    s += f", {m['onset_rate']} onsets/s, {m['hit_rate']} salient hits/s"
    if m['cuts_with_sound'] is not None:
        s += f", {len(r['events']['visual'])} visual events ({100 * m['cuts_with_sound']:.0f}% with an onset +-100 ms)"
    return s + '.'


def _stats(vals):
    x = np.array([v for v in vals if v is not None], np.float64)
    if not len(x):
        return {'n': 0}
    q = lambda p: round(float(np.percentile(x, p)), 3)
    return {'n': int(len(x)), 'median': q(50), 'p10': q(10), 'p90': q(90), 'min': round(float(x.min()), 3),
            'max': round(float(x.max()), 3), 'values': [None if v is None else round(float(v), 3) for v in vals]}


def aggregate(films):
    """profile dict from a list of film_metrics() results (ids/styles already set on each)"""
    stats = {k: _stats([f['metrics'].get(k) for f in films]) for k in METRICS}
    chance = {k: _stats([f.get('chance', {}).get(k) for f in films])
              for k in ('cuts_with_sound', 'cut_hit_frac', 'cut_frame_sync', 'cuts_on_beat')}
    arcs = [f['arc'] for f in films if f.get('arc')]
    arc = {q: [None if all(a[i] is None for a in arcs) else round(float(np.percentile(
        [a[i] for a in arcs if a[i] is not None], pq)), 1) for i in range(11)] for q, pq in
        (('median', 50), ('p10', 10), ('p90', 90))} if arcs else None
    return {'version': VERSION, 'n': len(films), 'metrics': stats, 'chance': chance, 'arc': arc,
            'films': [{'id': f['id'], 'style': f.get('style', ''), 'duration': f['duration'], 'has_video': bool(f['video']),
                       'metrics': f['metrics'], 'chance': f.get('chance', {}), 'arc': f.get('arc')} for f in films]}


def _warn(msg):
    print(f'  warning: {msg}', file=sys.stderr, flush=True)


def profile(paths, styles=None, use_ast=True, out=None):
    """build a reference profile from reference films / soundtracks (numbers only: no media, no file names); files
    that are missing, have no audio, are shorter than 3 s or silent are skipped with a warning"""
    paths = [paths] if isinstance(paths, str) else list(paths)
    skipped = [f'{p}: not found (or no media files in it)' for p in paths if not audio.list_media([p])]
    for w in skipped:
        _warn(f'skipped {w}')
    files = audio.list_media(paths)
    films = []
    for i, p in enumerate(files):
        try:
            why = media_problem(p)
            if not why:
                r = unscorable(p)[1]
                why = r and f'{p} {r}'
            if not why:
                f = film_metrics(p, use_ast=use_ast)
        except audio.AudioError as e:
            why = f'{p}: cannot decode ({str(e)[:160]})'
        if why:
            skipped.append(why)
            _warn(f'skipped {why}')
            continue
        f['id'] = f'ref_{len(films) + 1}'
        f['style'] = (styles.get(os.path.basename(p), '') if isinstance(styles, dict) else
                      styles[i] if styles and i < len(styles) else '')
        films.append(f)
        print(f"  {f['id']}: {f['duration']:.1f} s analysed in {f['seconds']:.1f} s", file=sys.stderr, flush=True)
    if not films:
        raise InputError('no usable reference films: ' + ('; '.join(skipped) if skipped else 'no media files given'))
    prof = aggregate(films)
    prof['warnings'] = [f'{len(skipped)} file(s) skipped (missing, no audio, under {MIN_S:.0f} s or silent)'] if skipped else []
    if len(films) < 3:
        prof['warnings'].append(f'only {len(films)} reference film(s): the ranges are the min-max of {len(films)} '
                                'value(s) plus a small tolerance, far too narrow to judge by; use 5 or more references')
        _warn(prof['warnings'][-1])
    prof['summary'] = profile_summary(prof) + ''.join(f' Warning: {w}.' for w in prof['warnings'])
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(prof, fh, indent=1)
    return prof


def profile_summary(p):
    m = p['metrics']
    med = lambda k: fmt_value(k, (m.get(k) or {}).get('median'))  # n.a. when no film measured it (LRA needs >= 10 s)
    li = m.get('lufs_i') or {}
    s = (f"profile of {p['n']} reference films (medians): integrated {med('lufs_i')} (p10-p90 "
         f"{fmt_value('lufs_i', li.get('p10'))} .. {fmt_value('lufs_i', li.get('p90'))}), LRA {med('lra')}, "
         f"onsets {med('onset_rate')}, salient hits {med('hit_rate')}")
    if m['music_frac'].get('n'):
        s += f", music in {med('music_frac')} of 5 s windows"
    if m['cut_rate'].get('n'):
        s += f", visual events {med('cut_rate')}"
    return s + '.'


def load_profile(path=None):
    with open(path or PROFILE_PATH, encoding='utf-8') as fh:
        return json.load(fh)


def fmt_value(k, v):
    if v is None:
        return 'n.a.'
    unit, nd = METRICS[k][2], METRICS[k][3]
    if unit == 'share':
        return f'{100 * v:.0f}%'
    return f'{v:.{nd}f} {unit}' if unit not in ('0-1', 'r') else f'{v:.{nd}f}'


def profile_text(p):
    lines = [p['summary'], '']
    for f in p['films']:
        lines.append(f"  {f['id']} ({f['duration']:.0f} s): {f['style'] or '-'}")
    if p.get('arc'):
        lines += ['', 'loudness arc, median (LU vs integrated at 0%, 10% .. 100%): ' +
                  ' '.join('-' if x is None else f'{x:+.1f}' for x in p['arc']['median'])]
    lines += ['', f"{'metric':40s} {'key':15s} {'median':>11s}   {'p10 .. p90':24s} min .. max"]
    for k, st in p['metrics'].items():
        if st.get('n'):
            lines.append(f"{METRICS[k][1][:40]:40s} {k:15s} {fmt_value(k, st['median']):>11s}   "
                         f"{fmt_value(k, st['p10']) + ' .. ' + fmt_value(k, st['p90']):24s} "
                         f"{fmt_value(k, st['min'])} .. {fmt_value(k, st['max'])}")
    return '\n'.join(lines)


def _cmd(args):
    styles = None
    try:
        if args.styles:
            if not os.path.isfile(args.styles):
                raise InputError(f'styles file not found: {args.styles}')
            with open(args.styles, encoding='utf-8') as fh:
                styles = json.load(fh)
        p = profile(args.refs, styles=styles, use_ast=not args.no_ast)
    except (InputError, audio.AudioError, json.JSONDecodeError) as e:
        error_line('profile', e)
        return 1
    emit(p, args, text=profile_text(p))


def add_cli(sub):
    p = sub.add_parser('profile', help='build a launch-sound reference profile from your own reference films',
                       description='Measure reference launch films / soundtracks (loudness, arc, ending, music and '
                       'voice presence, onset / hit density, picture sync, spectrum, stereo) and write their median, '
                       'p10-p90 and min-max as a profile for `earshot critique --profile`.')
    p.add_argument('refs', nargs='+', help='reference videos / soundtracks or folders')
    p.add_argument('-o', dest='out', help='write the profile JSON here (same as --out)')
    p.add_argument('--styles', help='JSON: list of style descriptions in file order, or {basename: description}')
    p.add_argument('--no-ast', action='store_true', help='skip the AudioSet pass (no music / speech presence; faster)')
    common(p)
    p.set_defaults(func=_cmd)


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
