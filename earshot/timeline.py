"""Listen through a whole soundtrack or film like a sound supervisor: events, sections, loudness, beat grid, sync.

    earshot timeline soundtrack.wav [--video film.mp4] [--cues cues.json] [--from 10 --to 40] [--detail low|normal|high]
    earshot timeline film.mp4                  # a video: its audio is analysed and its own picture cuts are used

    from earshot.timeline import timeline
    r = timeline('film.mp4', cues='cues.json')
    print(r['text'])                           # the listening transcript (ASCII, chronological)
    r['events'], r['sections'], r['loudness'], r['music'], r['sync']

What the numbers mean (and how far to trust them; measured in eval/timeline/RESULTS.md):
  events     onsets from a spectral-flux detector (5 ms frames), refined to the attack on a 1 ms envelope (the first
             millisecond within 20 dB of the event's peak above the bed, like an editor aligning a sample). On a
             finished film mix (Clawd) 93% of placed sound effects are found within 30 ms (65% of those more than
             6 dB under the music + ambience), median |timing error| 0.5 ms. Of the detections 51% are placed
             effects, 81% placed effects or music / ambience onsets, 96% counting the extra hits inside multi-hit
             effect samples. A file that starts with sound has an event at its first millisecond.
             prominence_db = rise of the K-weighted level over the bed just before (broadband, perceptual; bed_db =
             that bed, ~LUFS, floored at -80, so a hit after silence shows a huge prominence: the transcript prints
             'silent' in its 'over bed' column when the bed is below -60, and the summary's strongest accents only
             count events over a running bed, each also listed in the transcript);
             contrast_db = the same in the band where it stands out most (band: full / low < 200 Hz / high > 2 kHz);
             z = onset novelty in robust SDs vs the surrounding 2 s. 'swell' events are build-ups timed at their peak.
  tags       CLAP guesses for the event (compact event vocabulary, repeat-padded excerpt; phrases that score high on
             any short sound are damped by their mean over 536 real background events). They are hints: on the
             tuning mixes the right kind came first for 1 in 4 events masked by their bed and 1 in 2 events at least
             12 dB over it ('exposed'; the transcript marks the others with '?'), and no score we tried predicts a
             right tag better. On Clawd: key presses 58%, whooshes 56%, grass footsteps 0%; false KEY / UI tags on
             its ambience stem 35%, music stem 5%. Where AST runs (the most important events), AST speech >= 0.5
             overrides the kind with VOICE.
  sections   CLAP windows merged while they sound alike, split at strong timbre / level changes (Foote novelty);
             genre / mood / instrument tags from CLAP ('?' = weak genre; a mood only when it stands out, z >= 2.5:
             busy sound effects over music push weaker moods toward 'dark / aggressive'), extra sources and Music /
             Speech from AST; neighbours with the same genre and sources are merged.
  loudness   BS.1770: momentary (0.4 s) / short-term (3 s) LUFS (floored at the -70 LUFS gate), integrated LUFS (None
             when all is gated out), LRA, true peak (4x oversampled; None for digital silence); gaps touching the
             start / end of the file are a lead-in / tail, not a stop-down.
  music      tempo + beats (own autocorrelation / dynamic-programming tracker), downbeat guess, pulse clarity;
             tempo_alt = the double tempo when it pulses nearly as clearly (the felt tempo may be either).
  sync       each cut / cue -> the onset that answers it: offset_ms = sound - picture (negative = sound early);
             OK = within a frame (or 40 ms), near = within 100 ms, swell = a detected build-up peaks within 300 ms,
             peak = no onset, only a level peak within 150 ms (a soft whoosh / swell, or just the bed: check it),
             off / no sound = 'without a hit'. Build-ups are hard: on Clawd 2 sounded whooshes are still listed
             without a hit and 1 unsounded cue is answered by a level peak. 'chance' = what random edit times would
             score with this event density, so a 60% hit rate means little when chance is 50%. accent_* = the same
             for strong accents only (top 20% by prominence and >= 6 dB over the bed): the number to read for a
             dense mix, where some onset is always near.
--from/--to (or --start/--end) zooms in: the zoom is analysed inside +-30 s of context, so its verdicts match the full
analysis. True peak on decoded lossy audio (AAC / MP3 / Opus) includes the codec's overshoot: up to 0 dBTP is normal for
a delivered file (a master should leave the mixer at <= -1 dBTP before encoding); above 0 dBTP it clips on playback.
Model time is budgeted by --detail (CLAP ~0.3-1 s and AST several s per window on CPU): events near a cut / cue
first, spread over the timeline, one per burst, then the most prominent. Results are cached (EARSHOT_HOME/emb/timeline,
keyed on the file, the options, this module and the core decode / features / cut code); --fresh recomputes
everything and stores the new result.
"""
import hashlib
import json
import os
import time

import numpy as np
from scipy import ndimage, signal

from . import audio, config, features
from .report import emit, fmt_t, to_json

SR = audio.SR
SPAN_VERSION = 1               # bump when an excerpt / embedding definition changes (per-span model cache)
CODE_ID = hashlib.sha1(open(__file__, 'rb').read()).hexdigest()[:12]   # cached results never outlive a code change
HOP = 240                      # onset-detector frame hop: 5 ms
DETAIL = {                     # events CLAP-tagged / AST-labelled; section window, hop (s, grows to keep <= max_win
                               # windows on long files); AST section crops
    'low': {'tag': 12, 'ast': 0, 'win': 4.0, 'hop': 4.0, 'max_win': 30, 'ast_sec': 0},
    'normal': {'tag': 20, 'ast': 1, 'win': 3.0, 'hop': 2.0, 'max_win': 32, 'ast_sec': 3},
    'high': {'tag': 96, 'ast': 16, 'win': 3.0, 'hop': 1.5, 'max_win': 200, 'ast_sec': 12},
}
EVENT_VOCAB = {                # compact event vocabulary: kind -> caption-like phrases (tuned on synthetic mixes)
    'WHOOSH': ['a fast air whoosh', 'a swoosh passing by', 'a short swish', 'a whip pan whoosh transition'],
    'RISER': ['a rising riser building tension', 'a reverse cymbal swell', 'an uplifter sweep'],
    'HIT': ['a deep cinematic boom', 'a punchy impact hit', 'a heavy thud', 'a sub bass drop', 'a soft thump'],
    'KEY': ['a mechanical keyboard key press', 'typing on a computer keyboard', 'a laptop key click',
            'a spacebar press'],
    'UI': ['a mouse click', 'a soft UI button tap', 'a notification ding', 'a chat message pop',
           'a toggle switch click', 'a camera shutter click'],
    'STEP': ['footsteps on grass', 'footsteps on wood', 'a small soft footstep'],
    'FOLEY': ['a plastic click', 'lego bricks clicking', 'a wooden block hit', 'glass clinking', 'paper rustling',
              'leaves rustling', 'a cork pop', 'a bubble pop', 'a water drop', 'a small object dropping on a table'],
    'NOTE': ['a piano note', 'a bell chime', 'a plucked string note', 'an orchestral stab', 'a celesta note',
             'a glockenspiel note'],
    'DRUM': ['a kick drum', 'a snare drum hit', 'a hi-hat', 'a drum hit'],
    'VOICE': ['a spoken word', 'a person talking', 'a breath', 'a short vocal sound'],
}
TAG_BG = {                     # mean cosine of each event phrase over 536 real background events (tag_bg.py)
    'a fast air whoosh': -0.0137, 'a swoosh passing by': -0.0271, 'a short swish': 0.0538,
    'a whip pan whoosh transition': 0.0173, 'a rising riser building tension': -0.0403,
    'a reverse cymbal swell': -0.0354, 'an uplifter sweep': -0.0522, 'a deep cinematic boom': -0.0294,
    'a punchy impact hit': 0.0361, 'a heavy thud': 0.0383, 'a sub bass drop': -0.0194, 'a soft thump': -0.0344,
    'a mechanical keyboard key press': 0.1109, 'typing on a computer keyboard': -0.0065,
    'a laptop key click': 0.1327, 'a spacebar press': 0.1456, 'a mouse click': 0.0743,
    'a soft UI button tap': 0.0578, 'a notification ding': 0.0487, 'a chat message pop': 0.1634,
    'a toggle switch click': 0.1030, 'a camera shutter click': 0.0964, 'footsteps on grass': -0.0284,
    'footsteps on wood': -0.0025, 'a small soft footstep': 0.0282, 'a plastic click': 0.0729,
    'lego bricks clicking': -0.0168, 'a wooden block hit': 0.0335, 'glass clinking': -0.0419,
    'paper rustling': -0.0469, 'leaves rustling': -0.0228, 'a cork pop': 0.0633, 'a bubble pop': 0.0977,
    'a water drop': 0.1012, 'a small object dropping on a table': 0.0316, 'a piano note': 0.0143,
    'a bell chime': -0.0081, 'a plucked string note': 0.0687, 'an orchestral stab': 0.0640,
    'a celesta note': 0.0378, 'a glockenspiel note': 0.0350, 'a kick drum': 0.0659, 'a snare drum hit': 0.0322,
    'a hi-hat': 0.0719, 'a drum hit': 0.0758, 'a spoken word': -0.0140, 'a person talking': -0.1026,
    'a breath': 0.0520, 'a short vocal sound': 0.0318}
TAG_TEMPLATES = ('{}', 'the sound of {}', 'a recording of {}', '{} sound effect', 'an audio clip of {}')
TAG_CENTER = 0.5                # share of each phrase's background mean removed before ranking (hub phrases)
EXPOSED_DB = 12.0               # an event this far over its bed dominates its excerpt: its tag is right ~2x as often
BURST_S = 0.5                   # one CLAP tag per burst: same-band events this close share the budget
CONTEXT_S = 30.0                # a zoom is analysed inside +-30 s of context (sync verdicts, accents, beat grid)
SECTION_CATS = ('music_genre', 'music_mood', 'instruments', 'ambience', 'voice')
Z_OK, Z_WEAK = 2.5, 1.5
NEAR_S = 0.10                  # an event within 100 ms of a cut / cue is "near" it and always tagged
SYNC_S = 0.25                  # search radius for the event that answers a cut / cue
ON_BEAT_S = 0.05
ACCENT_DB = 6.0                 # an accent (deliberate hit) is >= 6 dB over its bed and in the file's top 20%
SWELL_RISE_DB = 8.0             # whoosh / riser build-up: minimum rise (dB)
PEAK_S, PEAK_RISE_DB = 0.15, 3.0  # a cut / cue without an onset: level peak within 150 ms rising >= 3 dB (swell_tune)
NOVELTY_Z = 3.0                 # section change: Foote novelty robust z (tuned on synthetic bed switches)
CLEAR_PULSE, WEAK_PULSE = 0.30, 0.15  # pulse clarity thresholds (calibrated on music vs ambience beds)
BED_FLOOR = 10 ** (-80 / 20)    # beds quieter than -80 dBFS count as -80 (keeps prominence finite)
SILENT_BED_DB = -60.0           # a bed below this is a gap: the rise over it ('over bed') is not shown as a number


# ------------------------------------------------------------------ events (no models)
def _band_envs(x):
    """1 ms max-abs envelopes of the full signal, its lows (< 200 Hz) and highs (> 2 kHz); the band filters run on a
    reflected 100 ms lead-in so they start without a transient"""
    m = audio.to_mono(x).astype(np.float64)
    w = SR // 1000
    n = len(m) // w
    pad = min(len(m) - 1, SR // 10)
    mp = np.concatenate([m[pad:0:-1], m])

    def env(y):
        return np.abs(y[: n * w]).reshape(n, w).max(axis=1) + 1e-7
    bands = {'full': np.abs(x[: n * w]).reshape(n, w, -1).max(axis=(1, 2)) + 1e-7 if x.ndim == 2 else env(m)}
    for kind, f in (('low', 200), ('high', 2000)):
        bands[kind] = env(signal.sosfilt(signal.butter(4, f, kind, fs=SR, output='sos'), mp)[pad:])
    return bands


def _mel_fb(n_fft, n_mels, fmin, fmax):
    """triangular mel filterbank (HTK mel scale), (n_mels, n_fft // 2 + 1)"""
    def mel(f):
        return 2595 * np.log10(1 + f / 700.0)
    edges = 700 * (10 ** (np.linspace(mel(fmin), mel(fmax), n_mels + 2) / 2595) - 1)
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    lo, c, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))


def log_mel(m, n_fft=1024, hop=HOP, n_mels=96, fmin=30, fmax=16000):
    """log-mel power spectrogram (dB), frame i centred at i * hop / SR"""
    y = np.pad(np.asarray(m, np.float32), (n_fft // 2, n_fft // 2), mode='reflect' if len(m) > n_fft else 'constant')
    n = 1 + (len(y) - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(y, (n, n_fft), (y.strides[0] * hop, y.strides[0]))
    out = np.empty((n_mels, n), np.float32)
    fb, win = _mel_fb(n_fft, n_mels, fmin, fmax).astype(np.float32), np.hanning(n_fft).astype(np.float32)
    for a in range(0, n, 4096):
        P = np.abs(np.fft.rfft(fr[a:a + 4096] * win, axis=1)) ** 2
        out[:, a:a + 4096] = 10 * np.log10(fb @ P.T + 1e-12)
    return out


def onset_curve(m):
    """superflux onset strength (mean positive log-mel increase, dB per frame) at 5 ms frames"""
    L = log_mel(m)
    L = np.maximum(L, L.max() - 80)
    d = L[:, 2:] - ndimage.maximum_filter1d(L, 3, axis=0)[:, :-2]
    return np.concatenate([np.zeros(2), np.maximum(d, 0).mean(axis=0)])


def _robust_z(v, size):
    med = ndimage.median_filter(v, size, mode='nearest')
    mad = ndimage.median_filter(np.abs(v - med), size, mode='nearest') * 1.4826
    pos = v[v > 0]
    floor = 0.02 * float(np.percentile(pos, 99)) if len(pos) else 1e-6
    return (v - med) / (mad + floor + 1e-9)


def _attack(envs, tc):
    """refine a detector frame time to the attack: find the band where the event stands out most above its bed,
    then walk back from the event's peak to the first millisecond within 20 dB of it (above the bed); the timing
    comes from the full-band envelope whenever the event rises 12 dB there too (filtered bands smear time)"""
    ic = int(round(tc * 1000))
    res = {}
    for band, e in envs.items():
        a, b = max(0, ic - 15), min(len(e), ic + 50)
        if b - a < 5:
            continue
        ip = a + int(np.argmax(e[a:b]))
        pre = e[max(0, ic - 160): max(0, ic - 40)]
        bed = max(float(np.median(pre if len(pre) >= 20 else e[: ic + 250])), BED_FLOOR)
        thr = bed + 0.1 * (e[ip] - bed)
        j, lo = ip, max(0, ic - 40)
        while j > lo and e[j - 1] >= thr:
            j -= 1
        res[band] = (e[ip] / bed, band, j, ip, bed)
    if not res:
        return None
    best = max(res.values())
    timing = res['full'] if 'full' in res and res['full'][0] >= 4.0 else best
    return best[0], best[1], timing


def _length_ms(e, i0, ip, bed):
    """from the attack until the band envelope falls 20 dB below the event peak or back to 3 dB over its bed
    (and stays there for 30 ms); capped at 2 s (a sustained note or a bed change)"""
    floor = max(e[ip] * 0.1, bed * 1.41)
    seg = e[ip: ip + 2000] < floor
    run = np.convolve(seg, np.ones(30, int), 'valid') == 30
    k = int(np.argmax(run)) if run.any() else len(seg)
    return int(min(ip + k - i0, 2000))


def _excess_centroid(m, t, length_s):
    """spectral centroid of what the event adds over the bed just before it (Hz)"""
    a, n = int(t * SR), int(min(max(length_s, 0.03), 0.12) * SR)
    seg = m[a: a + n]
    pre = m[max(0, a - int(0.15 * SR)): max(0, a - int(0.01 * SR))]
    if len(seg) < 256:
        return None
    nf = 4096
    f = np.fft.rfftfreq(nf, 1 / SR)
    P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nf)) ** 2 / len(seg)
    if len(pre) >= 256:
        Q = np.abs(np.fft.rfft(pre * np.hanning(len(pre)), nf)) ** 2 / len(pre)
        X = np.maximum(P - Q, 0)
        if X.sum() > 0.2 * P.sum():
            P = X
    k = (f >= 20) & (f <= 20000)
    return int((f[k] * P[k]).sum() / (P[k].sum() + 1e-30))


def _kw_env_db(x):
    """K-weighted 10 ms power (dB) at 1 ms steps: index i = window starting at i ms"""
    y = features._kw(x if x.ndim == 2 else x[:, None]).astype(np.float64)
    p = np.concatenate([[0.0], np.cumsum((y ** 2).sum(axis=1))])
    st = np.arange(0, max(1, len(y) - 480), 48)
    return 10 * np.log10((p[np.minimum(st + 480, len(y))] - p[st]) / 480 + 1e-12)


def detect_events(x, zthr=3.0, min_contrast_db=3.0, min_gap=0.03, lead=True):
    """transient onsets + loudness swells of a 48 kHz array -> list of event dicts (times relative to x); lead: x
    starts where the file does, so sound from the first millisecond is an event too (nothing precedes it)"""
    m = audio.to_mono(x).astype(np.float32)
    if len(m) < int(0.1 * SR):
        return []
    odf = onset_curve(m)
    z = _robust_z(odf, 401)
    loc = (odf == ndimage.maximum_filter1d(odf, 11)) & (z >= zthr) & (odf > 0)
    envs = _band_envs(x)
    lm_t, lm = features.loudness_curve(x, 0.4, 0.05) if len(m) > 0.4 * SR else (np.array([0.0]), np.array([-70.0]))
    out = []
    for i in np.where(loc)[0]:
        att = _attack(envs, i * HOP / SR)
        if att is None or 20 * np.log10(att[0]) < min_contrast_db:
            continue
        ratio, band, (_, tband, j, ip, bed) = att
        out.append({'t': j / 1000, 'shape': 'hit', 'band': band, 'contrast_db': min(20 * np.log10(ratio), 60.0),
                    'z': min(float(z[i]), 99.0), 'length_ms': _length_ms(envs[tband], j, ip, bed)})
    out.sort(key=lambda e: e['t'])
    merged = []
    for e in out:
        if merged and e['t'] - merged[-1]['t'] < min_gap:
            if e['contrast_db'] > merged[-1]['contrast_db']:
                merged[-1] = e
            continue
        merged.append(e)
    full = envs['full']
    if lead and len(full) > 5 and full[:50].max() >= 1e-3 and not any(e['t'] < 0.05 for e in merged):
        ip = int(np.argmax(full[:50]))
        j = int(np.argmax(full[:ip + 1] >= 0.1 * full[ip]))
        merged.insert(0, {'t': j / 1000, 'shape': 'hit', 'band': 'full', 'contrast_db': 60.0, 'z': 0.0,
                          'length_ms': _length_ms(full, j, ip, BED_FLOOR), 'starts_file': True})
    kw = _kw_env_db(x)
    for e in merged:
        j = int(round(e['t'] * 1000))
        pre = kw[max(0, j - 300): max(0, j - 20)]
        bed = -80.0 if e.get('starts_file') else max(float(np.median(pre)) if len(pre) >= 20 else
                                                     float(np.percentile(kw[: j + 2000], 10)), -80.0)
        e['prominence_db'] = max(0.0, float(kw[j: j + 80].max()) - bed) if j < len(kw) else 0.0
        e['bed_db'] = round(bed, 1)
    merged += _swells(x, merged)
    merged.sort(key=lambda e: e['t'])
    xs = x if x.ndim == 2 else x[:, None]
    for e in merged:
        a = int(e['t'] * SR)
        b = a + int(max(0.05, min(e['length_ms'] / 1000, 0.3)) * SR)
        e['peak_dbfs'] = round(audio.db(np.abs(xs[a:max(b, a + 1)]).max()), 1)
        e['local_lufs'] = round(float(np.interp(e['t'], lm_t, lm)), 1)
        e['centroid_hz'] = _excess_centroid(m, e['t'], e['length_ms'] / 1000)
        e['t'] = round(float(e['t']), 3)
        e['prominence_db'] = round(float(e['prominence_db']), 1)
        e['contrast_db'] = round(float(e['contrast_db']), 1)
        e['z'] = round(float(e['z']), 1)
    return merged


def level_envs(x):
    """10 ms RMS (dB, floored at -80) of the full signal and of its highs (> 2 kHz, where a whoosh under music
    shows), smoothed over 90 ms: {'full': s, 'high': s}, frame i = i * 10 ms"""
    m = audio.to_mono(x).astype(np.float64)
    n = len(m) // 480
    if n < 3:
        return {}
    pad = min(len(m) - 1, SR // 10)
    hi = signal.sosfilt(signal.butter(4, 2000, 'high', fs=SR, output='sos'), np.concatenate([m[pad:0:-1], m]))[pad:]
    out = {}
    for band, y in (('full', m), ('high', hi)):
        r = np.maximum(20 * np.log10(np.sqrt(np.mean(y[: n * 480].reshape(n, 480) ** 2, axis=1)) + 1e-9), -80.0)
        out[band] = ndimage.uniform_filter1d(r, 9)
    return out


def level_peak(envs, t, radius=PEAK_S, min_rise_db=PEAK_RISE_DB, min_fall_db=2.0, local=0.3):
    """a level peak near t that a transient detector misses (a whoosh / riser / swell peaking on a cut): the
    loudest smoothed 10 ms frame within +-radius, a local maximum over +-local s, rising >= min_rise_db over the
    0.6 s before it and falling >= min_fall_db within 0.3 s after -> {'t', 'band', 'rise_db'} or None"""
    best = None
    for band, s in envs.items():
        c = int(round(t * 100))
        a, b = max(0, c - int(radius * 100)), min(len(s), c + int(radius * 100) + 1)
        if b - a < 3:
            continue
        i = a + int(np.argmax(s[a:b]))
        w = int(local * 100)
        if s[i] < s[max(0, i - w): i + w + 1].max() or i < 5:
            continue
        rise = float(s[i] - s[max(0, i - 60): i].min())
        after = s[i: i + 31]
        if rise < min_rise_db or len(after) < 5 or s[i] - after.min() < min_fall_db:
            continue
        if best is None or rise > best['rise_db']:
            best = {'t': round(i / 100, 3), 'band': band, 'rise_db': round(rise, 1)}
    return best


def _swells(x, hits, min_rise_db=SWELL_RISE_DB, mono_frac=0.6, min_dur=0.25, envs=None):
    """build-ups (whooshes, risers, swells) timed at their peak: the level_envs rising >= min_rise_db over 0.25-2 s,
    mostly monotonically, then falling; a peak that is just a hit's attack is skipped"""
    envs = level_envs(x) if envs is None else envs
    if len(envs.get('full', ())) < 60:
        return []
    hit_t = np.array([h['t'] for h in hits]) if hits else np.zeros(0)
    out = []
    for band, s in envs.items():
        for i in np.where((s == ndimage.maximum_filter1d(s, 31)) & (s > -70))[0]:
            a = max(0, i - 200)
            if i - a < 25:
                continue
            j = a + int(np.argmin(s[a: i - 15]))
            rise, dur, after = s[i] - s[j], (i - j) / 100, s[i: i + 100]
            if rise < min_rise_db or dur < min_dur or len(after) < 10 or s[i] - after.min() < 4:
                continue
            if np.mean(np.diff(s[j: i + 1]) > 0) < mono_frac:
                continue
            t = i / 100
            if len(hit_t) and np.any(np.abs(hit_t - (t - 0.03)) < 0.05):
                continue
            if any(abs(o['t'] - t) < 0.15 for o in out):
                continue
            fall = np.where(after < s[i] - 10)[0]
            out.append({'t': t, 'shape': 'swell', 'band': band, 'contrast_db': float(rise), 'prominence_db': float(rise),
                        'bed_db': round(float(s[j]), 1), 'z': 0.0, 'rise_ms': int(dur * 1000),
                        'length_ms': int(fall[0] * 10) if len(fall) else 1000})
    return out


# ------------------------------------------------------------------ loudness
def loudness(x, at_start=True, at_end=True):
    """BS.1770 loudness picture: curves (floored at the -70 LUFS absolute gate), integrated (None when everything is
    gated out), LRA, true peak (None for digital silence), loudest moments, stop-downs, clipping. at_start / at_end:
    x begins / ends where the file does (gaps touching it are a lead-in / tail, not a stop-down)"""
    dur = len(x) / SR
    pk = float(np.abs(x).max()) if len(x) else 0.0
    r = {'integrated_lufs': None, 'true_peak_dbtp': round(features.true_peak_dbtp(x), 1) if pk > 0 else None,
         'digital_silence': pk == 0.0}
    if dur >= 0.4:
        tm, M = features.loudness_curve(x, 0.4, 0.1)
        M = np.maximum(M, -70.0)
        if M.max() > -70.0:
            r['integrated_lufs'] = round(features.integrated_lufs(x), 1)
            r['max_momentary'] = {'t': round(float(tm[np.argmax(M)]), 2), 'lufs': round(float(M.max()), 1)}
        r['momentary'] = {'t0': round(float(tm[0]), 3), 'dt': 0.1, 'lufs': [round(float(v), 1) for v in M]}
    if dur >= 3.0:
        ts, S = features.loudness_curve(x, 3.0, 0.5)
        S = np.maximum(S, -70.0)
        k = int(np.argmax(S))
        if S[k] > -70.0:
            r['max_short_term'] = {'start': round(float(ts[k]) - 1.5, 2), 'end': round(float(ts[k]) + 1.5, 2),
                                   'lufs': round(float(S[k]), 1)}
        r['short_term'] = {'t0': round(float(ts[0]), 3), 'dt': 0.5, 'lufs': [round(float(v), 1) for v in S]}
    v = features.lra(x) if dur >= 6 else None
    r['lra_lu'] = round(v, 1) if v is not None else None
    r['silences'] = _silences(x, r['integrated_lufs'], at_start, at_end)
    a = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    clip = a >= 0.999
    runs = _runs(clip, 3)
    r['clipping'] = {'samples': int(clip.sum()), 'runs': len(runs),
                     'times': [round(s / SR, 3) for s, _ in runs[:20]]}
    return r


def _runs(mask, min_len):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(st, en) if e - s >= min_len]


def _silences(x, integ, at_start=True, at_end=True):
    """gaps >= 0.25 s where the fast (0.1 s) loudness drops 20 LU under the integrated level (or below -60 LUFS);
    'where' = lead-in / tail when the gap touches the start / end of the file"""
    if len(x) < SR // 2:
        return []
    t, F = features.loudness_curve(x, 0.1, 0.05)
    F = np.maximum(F, -70.0)
    thr = -70.0 if integ is None or integ <= -60 else max(-60.0, integ - 20.0)
    low = (F <= thr) if integ is None else (F < thr)
    low = ndimage.binary_closing(np.pad(low, 3, mode='edge'), np.ones(7, bool))[3:-3] & (F < thr + 6)  # edge pad
    out, dur = [], len(x) / SR
    for s, e in _runs(low, 5):
        a, b = max(0.0, t[s] - 0.05), min(dur, t[e - 1] + 0.05)
        a, b = (0.0 if s == 0 else a), (dur if e == len(F) else b)
        seg = x[int(a * SR): int(b * SR)]
        pk = float(np.abs(seg).max()) if len(seg) else 0.0
        kind = 'digital silence' if pk < 1e-5 else ('near silence' if float(F[s:e].max()) < -60 else 'drop')
        g = {'start': round(a, 2), 'end': round(b, 2), 'dur': round(b - a, 2), 'kind': kind,
             'level_lufs': round(float(F[s:e].mean()), 1)}
        head, tail = a <= 0.05 and at_start, b >= dur - 0.05 and at_end
        if head or tail:
            g['where'] = 'whole file' if head and tail else ('lead-in' if head else 'tail')
        out.append(g)
    return out


# ------------------------------------------------------------------ music grid
def _pulse_envs(m):
    """beat-tracking novelty at 10 ms frames, broadband and low band (30-200 Hz) log-mel flux, local mean removed"""
    L = log_mel(m, n_fft=2048, hop=480, n_mels=64, fmin=30, fmax=8000)
    L = np.maximum(L, L.max() - 80)
    d = np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0)
    out = []
    for env in (d.mean(axis=0), d[:6].mean(axis=0)):
        env = np.maximum(env - ndimage.uniform_filter1d(env, 101), 0)
        out.append(env / (env.std() + 1e-9))
    return out


def _period(env, min_bpm, max_bpm):
    """(beat period in 10 ms frames, pulse clarity, score, autocorrelation) with a soft 120 BPM prior"""
    e0 = env - env.mean()
    f = np.fft.rfft(e0, 2 * len(e0))
    ac = np.fft.irfft(np.abs(f) ** 2)[: len(e0)]
    ac /= ac[0] + 1e-12
    lags = np.arange(int(6000 / max_bpm), min(int(6000 / min_bpm), len(ac) // 2))
    if not len(lags):
        return None
    prior = np.exp(-0.5 * (np.log2(6000 / lags / 120.0) / 0.9) ** 2)
    sc = (ac[lags] + 0.5 * ac[np.minimum(2 * lags, len(ac) - 1)]) * prior
    k = int(np.argmax(sc))
    P = float(lags[k])
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = sc[k - 1], sc[k], sc[k + 1]
        P += 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return P, float(ac[int(round(P))]), float(sc[k]), ac


def music_grid(m, min_bpm=60, max_bpm=200):
    """tempo (autocorrelation with a soft 120 BPM prior, on the broadband or the low-band flux, whichever pulses
    more clearly), beats (dynamic programming, Ellis 2007), a downbeat guess and how clear the pulse is
    (pulse_clarity = normalised autocorrelation at the beat period); tempo_alt = the double tempo when it pulses at
    least half as clearly (the felt tempo may be either: eval/timeline/tempo_tune.py)"""
    none = {'tempo': None, 'beats': [], 'downbeats': [], 'confidence': 'none'}
    if len(m) < 5 * SR:
        return dict(none, note='too short for a beat grid')
    cands = [(p, env) for env in _pulse_envs(m) for p in [_period(env, min_bpm, max_bpm)] if p]
    if not cands:
        return dict(none, note='too short for a beat grid')
    (P, clarity, _, ac), env = max(cands, key=lambda c: c[0][2])
    win = np.arange(max(1, int(P / 2)), int(2 * P) + 1)
    pen = 100.0 * np.log(win / P) ** 2
    score, back = env.copy(), np.full(len(env), -1)
    for t in range(int(P / 2), len(env)):
        prev = t - win
        ok = prev >= 0
        if not ok.any():
            continue
        c = score[prev[ok]] - pen[ok]
        j = int(np.argmax(c))
        if c[j] > 0:
            score[t] += c[j]
            back[t] = prev[ok][j]
    t = len(env) - int(P) + int(np.argmax(score[-int(P):]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = np.array(beats[::-1])
    if len(beats) < 4:
        return dict(none, note='no steady pulse found')
    ibi = np.diff(beats)
    reg = float(np.std(ibi) / np.mean(ibi))
    conf = 'clear' if clarity >= CLEAR_PULSE and reg < 0.1 else ('weak' if clarity >= WEAK_PULSE else 'none')
    ph = _downbeat_phase(m, beats)
    bt = beats * 0.01
    h = int(round(P / 2))
    alt = round(12000.0 / P, 1) if 12000.0 / P <= max_bpm and ac[h] >= 0.5 * clarity else None
    return {'tempo': round(6000.0 / P, 1), 'tempo_alt': alt, 'beats': [round(float(b), 3) for b in bt],
            'downbeats': [round(float(b), 3) for b in bt[ph::4]], 'pulse_clarity': round(clarity, 3),
            'beat_regularity': round(reg, 3), 'confidence': conf,
            'note': 'beats at 10 ms resolution; downbeats are a guess (4/4 assumed); tempo may be half/double the felt '
                    'tempo'}


def _downbeat_phase(m, beats):
    """which of 4 beat phases starts the bar: harmony changes (pitch-class profile between beats) plus low-band
    onsets (kick) are strongest on downbeats"""
    n_fft, hop = 4096, 480
    y = np.pad(np.asarray(m, np.float32), (n_fft // 2, n_fft // 2))
    n = 1 + (len(y) - n_fft) // hop
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    sel = (f >= 55) & (f <= 2000)
    pc = (np.round(12 * np.log2(f[sel] / 440.0)) % 12).astype(int)
    fr = np.lib.stride_tricks.as_strided(y, (n, n_fft), (y.strides[0] * hop, y.strides[0]))
    C = np.zeros((n, 12), np.float32)
    low = np.zeros(n, np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    for a in range(0, n, 2048):
        M = np.abs(np.fft.rfft(fr[a:a + 2048] * win, axis=1))
        for k in range(12):
            C[a:a + 2048, k] = M[:, sel][:, pc == k].sum(axis=1)
        low[a:a + 2048] = np.log1p(M[:, f < 150].sum(axis=1))
    b = np.minimum(beats, n - 1)
    seg = np.array([C[p:q].mean(axis=0) if q > p else C[p] for p, q in zip(b[:-1], b[1:])] + [C[b[-1]]])
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9
    harm = np.concatenate([[0.0], 1 - np.sum(seg[1:] * seg[:-1], axis=1)])
    kick = np.maximum(np.diff(low, prepend=low[0]), 0)
    kick = np.array([kick[max(0, i - 3): i + 4].max() for i in b])
    acc = harm / (harm.std() + 1e-9) + kick / (kick.std() + 1e-9)
    return int(np.argmax([acc[p::4].mean() for p in range(4)]))


# ------------------------------------------------------------------ cues and sync
def read_cues(cues):
    """cue events from a JSON path / JSON text / 'a,b,c' seconds / list / {fps, events:[{t|f, kind, id}]}
    (also accepted: time / sec / seconds for t, frame for f) -> (cues, fps)"""
    if cues is None:
        return [], None
    if isinstance(cues, str):
        s = cues.strip()
        if os.path.exists(s):
            cues = json.load(open(s, encoding='utf-8'))
        elif s[:1] in '[{':
            cues = json.loads(s)
        else:
            try:
                cues = [float(v) for v in s.replace(';', ',').split(',') if v.strip()]
            except ValueError:
                raise ValueError('cues: expected a JSON file, JSON text or comma-separated seconds, '
                                 f'got "{s[:60]}"') from None
    fps = None
    if isinstance(cues, dict):
        fps = cues.get('fps')
        cues = cues.get('events') or cues.get('cues') or []
    out = []
    for c in cues:
        if isinstance(c, (int, float)):
            out.append({'t': float(c), 'kind': 'cue', 'id': None})
            continue
        tk = next((k for k in ('t', 'time', 'sec', 'seconds') if k in c), None)
        fk = next((k for k in ('f', 'frame') if k in c), None)
        if tk:
            t = float(c[tk])
        elif fk and fps:
            t = float(c[fk]) / float(fps)
        elif fk:
            raise ValueError('cues have frame numbers but no fps: pass {"fps": 60, "events": [{"f": 120, ...}]}')
        else:
            raise ValueError(f'cue without a time: {json.dumps(c)[:80]} (give "t" in seconds or "f" with "fps")')
        out.append({'t': round(t, 4), 'kind': c.get('kind', 'cue'), 'id': c.get('id')})
    out.sort(key=lambda c: c['t'])
    return out, fps


def _strong(events):
    if not events:
        return []
    p = np.array([e['prominence_db'] for e in events])
    thr = min(6.0, float(np.percentile(p, 40)))
    return [e for e in events if e['prominence_db'] >= thr]


def _nearest(t, evs, ts, radius):
    if not len(ts):
        return None
    k = int(np.searchsorted(ts, t))
    best = None
    for j in (k - 1, k, k + 1):
        if 0 <= j < len(ts) and abs(ts[j] - t) <= radius and (best is None or abs(ts[j] - t) < abs(ts[best] - t)):
            best = j
    return None if best is None else evs[best]


def _answer(t, events, ts, strong, st, swells, sw, tol1, levels=None):
    """the event that answers a cut / cue at t: any onset within tol1 (OK), else the nearest strong one within
    100 ms, else any within 100 ms (near), else a detected build-up (whoosh / riser) peaking within 300 ms (swell),
    else a level peak within 150 ms that the onset detector misses (peak: a soft whoosh / swell?), else a strong
    onset within 250 ms (off); None = no sound"""
    for evs, arr, rad, status in ((events, ts, tol1, 'OK'), (strong, st, 0.1, 'near'), (events, ts, 0.1, 'near'),
                                  (swells, sw, 0.3, 'swell')):
        e = _nearest(t, evs, arr, rad)
        if e is not None:
            return e, status
    pk = level_peak(levels, t) if levels else None
    if pk is not None:
        return {'t': pk['t'], 'prominence_db': pk['rise_db'], 'shape': 'peak', 'band': pk['band']}, 'peak'
    e = _nearest(t, strong, st, SYNC_S)
    return (e, 'off') if e is not None else (None, 'no sound')


def _accents(events):
    """events that read as a deliberate hit: the top 20% by prominence and >= 6 dB over the bed -> (events, dB)"""
    if not events:
        return [], None
    thr = max(ACCENT_DB, float(np.percentile([e['prominence_db'] for e in events], 80)))
    return [e for e in events if e['prominence_db'] >= thr], round(thr, 1)


def _chance(ts, lo, hi, tol):
    """fraction of random times in [lo, hi] that fall within tol of one of ts (sorted)"""
    if not len(ts):
        return 0.0
    grid = np.arange(lo, hi, 0.005)
    k = np.clip(np.searchsorted(ts, grid), 1, max(1, len(ts) - 1))
    d = np.minimum(np.abs(grid - ts[k - 1]), np.abs(grid - ts[np.minimum(k, len(ts) - 1)]))
    return round(float(np.mean(d <= tol)), 3)


def sync(times, events, beats=None, frame_s=None, levels=None):
    """for each cut / cue time: the audio event that answers it (offset ms, negative = sound early) and stats;
    levels = level_envs(x) also looks for level peaks (soft whooshes / swells) where no onset answers"""
    strong = _strong(events)
    swells = [e for e in events if e['shape'] == 'swell']
    st, sw = np.array([e['t'] for e in strong]), np.array([e['t'] for e in swells])
    ts = np.array([e['t'] for e in events])
    acc, acc_db = _accents(events)
    at = np.array([e['t'] for e in acc])
    tol1 = frame_s or 0.04
    bt = np.asarray(beats) if beats is not None and len(beats) else None
    items = []
    for c in times:
        t = c['t'] if isinstance(c, dict) else float(c)
        e, status = _answer(t, events, ts, strong, st, swells, sw, tol1, levels)
        it = {'t': round(t, 3)}
        if isinstance(c, dict):
            it.update({k: c[k] for k in ('kind', 'id') if c.get(k) is not None})
        it.update({'offset_ms': None if e is None else round((e['t'] - t) * 1000, 1), 'status': status})
        if e is not None:
            it.update({'event_t': e['t'], 'event_prominence_db': e['prominence_db']})
            if e.get('label'):
                it['event'] = e['label']
        a = _nearest(t, acc, at, 0.1)
        it['accent_offset_ms'] = None if a is None else round((a['t'] - t) * 1000, 1)
        if bt is not None:
            it['beat_offset_ms'] = round((t - float(bt[int(np.argmin(np.abs(bt - t)))])) * 1000, 1)
        items.append(it)
    n = len(items)
    stats = {'n': n}
    if not n:
        return {'items': items, 'stats': stats}
    offs = np.array([abs(i['offset_ms']) for i in items if i['offset_ms'] is not None])
    k1 = 'within_1_frame' if frame_s else 'within_40ms'
    stats.update({f'hit_{k1}': round(float(np.mean([i['status'] == 'OK' for i in items])), 3),
                  'hit_within_100ms': round(float(np.mean([i['status'] in ('OK', 'near') for i in items])), 3),
                  'median_abs_offset_ms': round(float(np.median(offs)), 1) if len(offs) else None,
                  'answered_by_swell': sum(i['status'] == 'swell' for i in items),
                  'answered_by_level_peak': sum(i['status'] == 'peak' for i in items),
                  'no_sound': [i['t'] for i in items if i['status'] in ('no sound', 'off')],
                  ('frame_ms' if frame_s else 'tolerance_ms'): round(tol1 * 1000, 1)})
    lo, hi = min(i['t'] for i in items) - 1, max(i['t'] for i in items) + 1
    if len(ts):
        stats[f'chance_{k1}'] = _chance(ts, lo, hi, tol1)
        stats['chance_within_100ms'] = _chance(ts, lo, hi, 0.1)
    if len(at):
        ao = np.array([abs(i['accent_offset_ms']) if i['accent_offset_ms'] is not None else 1e9 for i in items])
        stats.update({'accent_min_db': acc_db, f'accent_{k1}': round(float(np.mean(ao <= tol1 * 1000 + 1e-6)), 3),
                      f'accent_chance_{k1}': _chance(at, lo, hi, tol1),
                      'accent_within_100ms': round(float(np.mean(ao <= 100)), 3),
                      'accent_chance_within_100ms': _chance(at, lo, hi, 0.1)})
    if bt is not None and len(bt) > 1:
        bo = np.array([abs(i['beat_offset_ms']) for i in items])
        stats['on_beat'] = round(float(np.mean(bo <= ON_BEAT_S * 1000)), 3)
        stats['on_beat_chance'] = round(min(1.0, 2 * ON_BEAT_S / float(np.median(np.diff(bt)))), 3)
    return {'items': items, 'stats': stats}


# ------------------------------------------------------------------ models (CLAP tags, AST labels)
def _embed_text(c, texts):
    """c.embed_text, surviving a failed save of the shared text cache (Windows: os.replace over an open file); the
    embeddings are then kept in memory for this process"""
    try:
        return c.embed_text(texts)
    except PermissionError:
        c._text_dirty = False
        return c.embed_text(texts)


def _phrase_matrix(c, terms, tpl=None):
    """[(kind, phrase)] embedded through prompt templates (default: earshot's vocab.json ones; mean, unit rows)"""
    if tpl is None:
        v = json.load(open(os.path.join(config.DATA_DIR, 'vocab.json'), encoding='utf-8'))
        tpl = v.get('templates') or ['{}']
    E = _embed_text(c, [p.format(t) for _, t in terms for p in tpl]).reshape(len(terms), len(tpl), -1).mean(axis=1)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def _vocab_matrix(c):
    cats = json.load(open(os.path.join(config.DATA_DIR, 'vocab.json'), encoding='utf-8'))['categories']
    terms = [(k, t) for k, ts in cats.items() for t in ts]
    return [t for _, t in terms], np.array([k for k, _ in terms]), _phrase_matrix(c, terms)


def _event_matrix(c):
    """compact event vocabulary: (terms, kinds, E), embedded through timeline's own templates (TAG_BG is
    measured with them)"""
    terms = [(k, t) for k, ts in EVENT_VOCAB.items() for t in ts]
    return [t for _, t in terms], np.array([k for k, _ in terms]), _phrase_matrix(c, terms, TAG_TEMPLATES)


def tag_scores(A, E, terms):
    """CLAP audio embeddings A (n, 512) vs the event phrases E -> (cos, z): z within each event after removing
    TAG_CENTER of each phrase's background mean, so phrases that score high on any short sound ('a chat
    message pop', 'a spacebar press') no longer win everywhere (eval/timeline/tag_bg.py)"""
    cos = A @ E.T
    d = cos - TAG_CENTER * np.array([TAG_BG[t] for t in terms])
    return cos, (d - d.mean(axis=1, keepdims=True)) / (d.std(axis=1, keepdims=True) + 1e-9)


def _zscores(A, E):
    cos = A @ E.T
    return cos, (cos - cos.mean(axis=1, keepdims=True)) / (cos.std(axis=1, keepdims=True) + 1e-9)


def _top(terms, cats, cos, z, allowed, k=3):
    idx = [i for i in np.argsort(-z) if cats[i] in allowed][:k]
    return [{'tag': terms[i], 'cat': str(cats[i]), 'z': round(float(z[i]), 2), 'cos': round(float(cos[i]), 3)}
            for i in idx]


def _short(term):
    for a in ('a ', 'an ', 'the '):
        if term.startswith(a):
            return term[len(a):]
    return term


def _cached(model, fid, spans, kind, compute, reuse=True):
    """per-span model outputs cached on disk next to the model's own cache (reuse=False recomputes and rewrites)"""
    keys = [hashlib.sha1(f'{fid}|{a:.3f}|{b:.3f}|{kind}|tl{SPAN_VERSION}'.encode()).hexdigest()[:24] for a, b in spans]
    files = [os.path.join(model.dir, f'tl_{k}.npy') for k in keys]
    out, todo = [None] * len(spans), []
    for i, f in enumerate(files):
        if reuse and os.path.exists(f):
            try:
                out[i] = np.load(f)
                continue
            except Exception:
                pass
        todo.append(i)
    if todo:
        res = compute([spans[i] for i in todo])
        for i, v in zip(todo, res):
            out[i] = v
            tmp = f'{files[i]}.{os.getpid()}.tmp.npy'
            np.save(tmp, v)
            audio.replace_quietly(tmp, files[i])
    return np.stack(out) if out else None


def span_id(path):
    """file id for the per-span model cache; mono sources decode at unity gain since core decode v2"""
    return audio.file_id(path) + ('|mono2' if audio.probe(path).get('channels') == 1 else '')


def _seg(x, a, b):
    return audio.to_mono(x[max(0, int(a * SR)): max(int(a * SR) + 480, int(b * SR))])


def _event_span(e, dur):
    if e['shape'] == 'swell':
        a, b = e['t'] - e.get('rise_ms', 500) / 1000, e['t'] + 0.3
    else:
        a, b = e['t'] - 0.02, e['t'] + min(max(e['length_ms'] / 1000, 0.3), 1.5)
    return round(max(0.0, a), 3), round(min(dur, b), 3)


def _spread(order, times, hi, n, lo=0.0):
    """choose n indices from a priority order so the whole range [lo, hi] is covered: the best event of each of
    n // 2 equal time bins first, then the rest by priority"""
    nb = max(1, n // 2)
    first, seen = [], set()
    for i in order:
        b = min(nb - 1, max(0, int((times[i] - lo) / max(hi - lo, 1e-6) * nb)))
        if b not in seen:
            seen.add(b)
            first.append(i)
    rest = [i for i in order if i not in set(first)]
    return sorted((first + rest)[:n])


def _pick(events, n_tag, near, lo, hi):
    """events to tag: those near a cut / cue first, then by prominence, spread over the timeline; later events of
    a burst (same band within BURST_S of a picked one) go last, so one typing run does not take the whole budget"""
    near = np.asarray(sorted(near))
    is_near = [bool(len(near) and np.min(np.abs(near - e['t'])) <= NEAR_S) for e in events]
    order = sorted(range(len(events)),
                   key=lambda i: (-is_near[i], -events[i]['prominence_db'], -events[i]['contrast_db']))
    kept, dup = [], []
    for i in order:
        e = events[i]
        same = any(abs(events[j]['t'] - e['t']) < BURST_S and events[j]['band'] == e['band'] for j in kept)
        (dup if same else kept).append(i)
    return _spread(kept + dup, [e['t'] for e in events], hi, n_tag, lo), kept + dup


def tag_events(x, fid, t0, events, n_tag, n_ast, near=(), reuse=True, win=None):
    """CLAP character tags for up to n_tag events (see _pick): the event excerpt, repeat-padded to one 10 s window,
    scored against the compact event vocabulary (tag_scores); 'exposed' = the event is >= EXPOSED_DB over its bed,
    so it dominates the excerpt. AST labels for the n_ast most important tagged events; AST speech >= 0.5 overrides
    the CLAP kind with VOICE. Repeat padding beat zero padding and bed subtraction on the tuning mixes
    (eval/timeline/tag_tune.py) at half the cost of the latter. Times are relative to x, which starts at file time
    t0 (span cache keys); win = the analysed window inside x (default: all of it)"""
    from .models import clap
    if not events or n_tag <= 0:
        return {'tagged': 0, 'ast': 0, 'clap_windows': 0}
    dur = len(x) / SR
    pick, order = _pick(events, n_tag, near, *(win or (0.0, dur)))
    c = clap.get()
    terms, kinds, E = _event_matrix(c)
    spans = [_event_span(events[i], dur) for i in pick]
    A = _cached(c, fid, [(a + t0, b + t0) for a, b in spans], 'clap-repeat',
                lambda q: c.embed_audio([_seg(x, a - t0, b - t0) for a, b in q], pad='repeatpad'), reuse)
    _, z = tag_scores(A, E, terms)
    for j, i in enumerate(pick):
        idx = np.argsort(-z[j])[:3]
        e = events[i]
        e['tags'] = [{'tag': terms[k], 'kind': str(kinds[k]), 'z': round(float(z[j][k]), 2)} for k in idx]
        e['kind'] = e['tags'][0]['kind']
        e['label'] = _short(terms[idx[0]])
        e['exposed'] = bool(e['prominence_db'] >= EXPOSED_DB)
    picked = set(pick)
    ranked = [i for i in order if i in picked][:n_ast]
    if ranked:
        from .models import ast
        a = ast.get()
        sp = [_event_span(events[i], dur) for i in ranked]
        P = _cached(a, fid, [(s + t0, b + t0) for s, b in sp], 'ast',
                    lambda q: a.probs([_seg(x, s - t0, b - t0) for s, b in q]), reuse)
        for i, p in zip(ranked, P):
            e = events[i]
            e['ast'] = a.top(p, 4, min_p=0.05)
            sp_p = a.prob_of(p, 'Speech')
            if sp_p >= 0.5 and e['kind'] != 'VOICE':
                e.update(kind='VOICE', label=f'speech (AST {sp_p:.2f})', exposed=True)
    return {'tagged': len(pick), 'ast': len(ranked), 'clap_windows': len(pick)}


# ------------------------------------------------------------------ sections
def _novelty_refine(m, t, radius):
    """move a coarse boundary to the strongest spectral change within +-radius (1 s log-mel halves)"""
    a, b = max(0.0, t - radius - 1.0), min(len(m) / SR, t + radius + 1.0)
    seg = m[int(a * SR): int(b * SR)]
    if len(seg) < 2 * SR:
        return t
    L = log_mel(seg, n_fft=2048, hop=2400, n_mels=40)
    best, bt = -1.0, t
    for k in range(10, L.shape[1] - 10):
        tt = a + k * 0.05
        if abs(tt - t) > radius:
            continue
        d = float(np.linalg.norm(L[:, k - 10:k].mean(axis=1) - L[:, k:k + 10].mean(axis=1)))
        if d > best:
            best, bt = d, tt
    return round(bt, 2)


def novelty_bounds(m, kernel_s=4.0, zthr=NOVELTY_Z):
    """strong timbre / level changes: Foote novelty (checkerboard kernel, +-kernel_s) on 0.5 s log-mel frames;
    returns [(time_s, z)] of peaks at least kernel_s apart with robust z >= zthr"""
    L = log_mel(m, n_fft=2048, hop=2400, n_mels=40, fmin=30, fmax=16000)
    n, w = L.shape[1] // 10, int(kernel_s / 0.5)
    if n < 4 * w:
        return []
    F = L[:, : n * 10].reshape(L.shape[0], n, 10).mean(axis=2)
    F = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + 1e-6)
    F /= np.linalg.norm(F, axis=0, keepdims=True) + 1e-9
    S = F.T @ F
    g = np.exp(-0.5 * (np.arange(-w, w) + 0.5) ** 2 / (0.5 * w) ** 2)
    K = np.outer(np.r_[-np.ones(w), np.ones(w)], np.r_[-np.ones(w), np.ones(w)]) * np.outer(g, g)
    nov = np.zeros(n)
    for i in range(w, n - w):
        nov[i] = float(np.sum(K * S[i - w:i + w, i - w:i + w]))
    z = _robust_z(nov, n)
    pk = np.where((nov == ndimage.maximum_filter1d(nov, 2 * w + 1)) & (z >= zthr))[0]
    return [(round(i * 0.5, 2), round(float(z[i]), 1)) for i in pk if w <= i < n - w]


def sections(x, fid, t0, win, hop, ast_budget=4, merge_cos=0.84, min_len=4.0, reuse=True, nov_z=None):
    """CLAP windows merged into contiguous sections; each section tagged (genre, mood, instruments, ambience,
    voice) plus AST Music / Speech / Silence"""
    from .models import clap
    dur = len(x) / SR
    if dur < 1.0:
        return [], {'windows': 0, 'ast': 0}
    c = clap.get()
    terms, cats, E = _vocab_matrix(c)
    starts = list(np.arange(0.0, max(dur - win, 0.0) + 1e-6, hop))
    if not starts or starts[-1] + win < dur - 0.5:
        starts.append(max(0.0, dur - win))
    spans = [(round(s, 3), round(min(dur, s + win), 3)) for s in starts]
    A = _cached(c, fid, [(a + t0, b + t0) for a, b in spans], 'clap-repeat',
                lambda sp: c.embed_audio([_seg(x, a - t0, b - t0) for a, b in sp], pad='repeatpad'), reuse)
    segs = [[i] for i in range(len(spans))]

    def mean(seg):
        v = A[seg].mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    def length(seg):
        return spans[seg[-1]][1] - spans[seg[0]][0]
    m = audio.to_mono(x).astype(np.float32)
    nov = [t for t, _ in novelty_bounds(m, zthr=NOVELTY_Z if nov_z is None else nov_z)]

    def blocked(a, b):
        j = spans[b[0]][0] + (win if spans[b[0]][0] >= spans[a[-1]][1] else (spans[a[-1]][1] - spans[b[0]][0])) / 2
        return any(abs(t - j) <= hop for t in nov)
    max_n = max(2, int(np.ceil(dur / 8.0)))
    while len(segs) > 1:
        sims = np.array([float(mean(a) @ mean(b)) - blocked(a, b) for a, b in zip(segs[:-1], segs[1:])])
        short = [j for j in range(len(sims)) if length(segs[j]) < min_len or length(segs[j + 1]) < min_len]
        j = int(np.argmax(sims))
        if sims[j] < merge_cos and len(segs) <= max_n:
            if not short:
                break
            j = max(short, key=lambda k: sims[k])
        segs[j] = segs[j] + segs.pop(j + 1)
    bounds = [0.0]
    for a, b in zip(segs[:-1], segs[1:]):
        coarse = (spans[a[-1]][0] + spans[b[0]][0] + win) / 2 if spans[b[0]][0] < spans[a[-1]][1] else spans[b[0]][0]
        bounds.append(_novelty_refine(m, coarse, hop))
    bounds.append(round(dur, 2))
    k = 1
    while k < len(bounds) - 1:                       # a refined boundary may leave a sliver: merge it away
        if bounds[k] - bounds[k - 1] < min_len / 2 or bounds[k + 1] - bounds[k] < min_len / 2:
            segs[k - 1] = segs[k - 1] + segs.pop(k)
            bounds.pop(k)
        else:
            k += 1
    out = []
    for k, seg in enumerate(segs):
        v = mean(seg)
        cos, z = _zscores(v[None], E)
        s = {'start': round(bounds[k], 2), 'end': round(bounds[k + 1], 2), 'windows': len(seg),
             'coherence': round(float(np.mean(A[seg] @ v)), 3)}
        for cat in SECTION_CATS:
            s[cat] = _top(terms, cats, cos[0], z[0], (cat,), k=2)
        _section_level(x, s)
        out.append(s)
    n_ast = _section_ast(x, fid, t0, out, ast_budget, reuse)
    for s in out:
        s['label'] = _section_label(s)
    def key(s):                                      # genre + other sources (instruments / mood vary in a track)
        parts = s['label'].split(' | ')
        return parts[0].split(',')[0], parts[1:]
    k = 1
    while k < len(out):                              # neighbours that read alike, no strong change between: merge
        a, b = out[k - 1], out[k]
        if key(a) != key(b) or any(abs(t - b['start']) <= 1.0 for t in nov):
            k += 1
            continue
        keep = a if a['end'] - a['start'] >= b['end'] - b['start'] else b
        out[k - 1] = dict(keep, start=a['start'], end=b['end'], windows=a['windows'] + b['windows'])
        _section_level(x, out[k - 1])
        out.pop(k)
    return out, {'windows': len(spans), 'ast': n_ast}


def _section_level(x, s):
    xs = x[int(s['start'] * SR): int(s['end'] * SR)]
    v = features.integrated_lufs(xs) if len(xs) > 0.4 * SR else None
    s['lufs'] = round(v, 1) if v is not None and v > -70 else None
    s['digital_silence'] = bool(len(xs) and float(np.abs(xs).max()) == 0.0)


def _section_ast(x, fid, t0, secs, budget, reuse=True):
    """AST Music / Speech / Silence and top labels per section from centred 10.24 s crops: one per section (longest
    first) while the budget lasts, a second one for sections over 15 s if crops are left"""
    if budget <= 0 or not secs:
        return 0
    from .models import ast
    a = ast.get()
    order = sorted(range(len(secs)), key=lambda k: -(secs[k]['end'] - secs[k]['start']))
    plan = {k: 1 for k in order[:budget]}
    for k in order:
        if sum(plan.values()) < budget and k in plan and secs[k]['end'] - secs[k]['start'] > 15:
            plan[k] = 2
    spans, owner = [], []
    for k, n in sorted(plan.items()):
        s0, d = secs[k]['start'], secs[k]['end'] - secs[k]['start']
        w = min(d, 10.24)
        for j in range(n):
            c0 = s0 + (d - w) * (j + 0.5) / n if n > 1 else s0 + (d - w) / 2
            spans.append((round(c0, 3), round(c0 + w, 3)))
            owner.append(k)
    P = _cached(a, fid, [(p + t0, q + t0) for p, q in spans], 'ast',
                lambda sp: a.probs([_seg(x, p - t0, q - t0) for p, q in sp]), reuse)
    owner = np.array(owner)
    for k in plan:
        p = P[owner == k].mean(axis=0)
        secs[k]['ast'] = {lab: round(a.prob_of(p, lab), 3) for lab in ('Music', 'Speech', 'Silence')}
        secs[k]['ast_top'] = a.top(p, 5, min_p=0.05)
    return len(spans)


AST_GENERIC = {'Music', 'Musical instrument', 'Speech', 'Silence', 'Sound effect', 'Inside, small room',
               'Outside, rural or natural', 'Outside, urban or manmade', 'Inside, large room or hall', 'Animal'}


def _section_label(s):
    """'cinematic orchestral music, strings ensemble, uplifting | also: rain, typing (AST) | voice: ...' -- genre
    tags under z 2.5 carry a '?'"""
    if s.get('digital_silence'):
        return 'digital silence'
    if s.get('lufs') is None or s['lufs'] < -55:
        return 'near silence'
    ap = s.get('ast') or {}
    parts = []
    music = ap['Music'] >= 0.3 if ap else s['music_genre'][0]['z'] >= s['ambience'][0]['z']
    if music:
        g, md, ins = s['music_genre'][0], s['music_mood'][0], s['instruments'][0]
        bits = [_short(g['tag']) + ('' if g['z'] >= Z_OK else '?')]
        if ins['z'] >= Z_WEAK:
            bits.append(_short(ins['tag']))
        if md['z'] >= Z_OK:                        # weaker moods were wrong ~45% under sfx (label_tune.py)
            bits.append(_short(md['tag']).replace(' music', ''))
        parts.append(', '.join(bits))
    extra = [lab.lower() for lab, p in s.get('ast_top') or [] if lab not in AST_GENERIC and p >= 0.08][:3]
    if extra:
        parts.append('also: ' + ', '.join(extra) + ' (AST)')
    elif s['ambience'][0]['z'] >= (Z_OK if music else Z_WEAK):
        parts.append('amb: ' + _short(s['ambience'][0]['tag']))
    if ap.get('Speech', 0) >= 0.3 or (not ap and s['voice'][0]['z'] >= Z_OK):
        parts.append('voice: ' + _short(s['voice'][0]['tag']))
    return ' | '.join(parts) or 'unclear (no confident music / ambience / voice tag)'


# ------------------------------------------------------------------ transcript
def _ascii(s):
    return str(s).encode('ascii', 'backslashreplace').decode()


def _hz(f):
    return '-' if not f else (f'{f / 1000:.1f} kHz' if f >= 1000 else f'{f} Hz')


BUILDUP_WORDS = ('whoosh', 'swoosh', 'swish', 'riser', 'rise', 'swell', 'sweep', 'wave', 'uplift')


def _event_line(e, marks):
    """one transcript line: kind ('?' = the event is masked by its bed, so the tag is a weak hint), best tag /
    runner-up of another kind, level, prominence, length, centroid, AST, sync marks"""
    if e.get('label'):
        alt = next((t['tag'] for t in e['tags'] if t['kind'] != e['kind']), None)
        what = e['label'] + (f' / {_short(alt)}' if alt else '')
        k = e['kind'] + ('' if e.get('exposed') else '?')
    else:
        what, k = f"{e['band']} {'swell' if e['shape'] == 'swell' else 'onset'}", '-'
    over = f"{e['prominence_db']:+5.1f} dB" if e.get('bed_db', 0.0) > SILENT_BED_DB else f"{'silent':>8}"
    ln = (f"{fmt_t(e['t']):>8}  {k:<6} {what[:46]:<46} {e['peak_dbfs']:6.1f} dBFS {over} "
          f"{e['length_ms']:>5} ms {_hz(e.get('centroid_hz')):>8}")
    lab = next((f'{l} {p:.2f}' for l, p in e.get('ast') or [] if l != 'Music' or len(e['ast']) == 1), None)
    if lab:
        ln += f'  [AST {lab}]'
    if marks:
        ln += '  | ' + '; '.join(marks)
    return _ascii(ln)


def top_accents(ev, n=3):
    """the n most prominent events over a running bed (bed above -60), at least 0.5 s apart"""
    acc = []
    for e in sorted([e for e in ev if e.get('bed_db', -80) > SILENT_BED_DB], key=lambda e: -e['prominence_db']):
        if all(abs(e['t'] - a['t']) >= 0.5 for a in acc):
            acc.append(e)
        if len(acc) == n:
            break
    return acc


def _unanswered_line(kind, it):
    who = ' '.join(v for v in (it.get('kind') if it.get('kind') != 'cue' else '', it.get('id') or '') if v)
    who = f' {who}' if who and kind == 'cues' else ''
    if it['status'] == 'peak':
        why = (f"no onset; level peak {it['offset_ms']:+.0f} ms, rising {it['event_prominence_db']:.0f} dB "
               '(a soft whoosh / swell, or just the bed?)  <- CHECK')
    else:
        why = ('no onset or level peak within 250 ms' if it['status'] == 'no sound' else
               f"no onset within 100 ms (strong sound at {it['offset_ms']:+.0f} ms)")
        build = any(w in f"{it.get('kind') or ''} {it.get('id') or ''}".lower() for w in BUILDUP_WORDS)
        why += '  <- NO HIT' + (' (a soft build-up can still be missed: look at the sheet)' if build else '')
    return _ascii(f"{fmt_t(it['t']):>8}  {kind[:3].upper():<6}{who}  {why}")


def transcript(r):
    """the listening transcript: summary, sections, then one chronological line per tagged event, unanswered cut /
    cue, gap and section change"""
    L = [r['summary'], '']
    secs = r.get('sections') or []
    if secs:
        L.append('SECTIONS')
        for s in secs:
            ap = s.get('ast') or {}
            extra = f"  [AST music {ap['Music']:.2f}, speech {ap['Speech']:.2f}]" if ap else ''
            lu = f"  {s['lufs']:.0f} LUFS" if s.get('lufs') is not None else ''
            L.append(_ascii(f"  {fmt_t(s['start'])}-{fmt_t(s['end'])}  {s['label']}{lu}{extra}"))
        L.append('')
    ev, sy = r.get('events') or [], r.get('sync') or {}
    marks = {}
    for kind in ('cuts', 'cues'):
        for it in (sy.get(kind) or {}).get('items', []):
            if it.get('event_t') is not None and it['status'] not in ('off', 'peak'):
                tag = 'cut' if kind == 'cuts' else (it.get('kind') or 'cue')
                marks.setdefault(it['event_t'], []).append(f"{tag} {it['offset_ms']:+.0f} ms {it['status']}")
    acc = {e['t'] for e in top_accents(ev)}
    for t in acc:
        marks.setdefault(t, []).append('strong accent (summary)')
    rows = [(e['t'], 1, _event_line(e, marks.get(e['t'], []))) for e in ev if e.get('label') or e['t'] in marks]
    for kind in ('cuts', 'cues'):
        for it in (sy.get(kind) or {}).get('items', []):
            if it['status'] in ('no sound', 'off', 'peak'):
                rows.append((it['t'], 2, _unanswered_line(kind, it)))
    bounds = [s['start'] for s in secs] + [r['end']]
    for k, s in enumerate(secs):
        rest = [e for e in ev if s['start'] <= e['t'] < bounds[k + 1] and not e.get('label') and e['t'] not in marks]
        more = f'  (+{len(rest)} untagged onsets, {len(rest) / max(s["end"] - s["start"], 1e-6):.1f}/s)' if rest else ''
        rows.append((s['start'], 0, _ascii(f"== {fmt_t(s['start'])}-{fmt_t(s['end'])}  {s['label']}{more}")))
    for g in (r.get('loudness') or {}).get('silences', []):
        lvl = '< -70' if g['level_lufs'] <= -70 else f"{g['level_lufs']:.0f}"
        rows.append((g['start'], 1, f"{fmt_t(g['start']):>8}  {'GAP':<6} {g['kind']} {g['dur']:.2f} s ({lvl} LUFS)"
                                    + (f" = {g['where']}" if g.get('where') else '')))
    rows.sort(key=lambda q: (q[0], q[1]))
    L.append('TRANSCRIPT   time  kind   what (CLAP tag / runner-up)                     peak  over bed  length  '
             'centroid  | sync')
    L += [q[2] for q in rows]
    L += ['', 'NOTES  times, levels, loudness and sync offsets are measurements. Event kinds / tags are CLAP guesses: '
          f"'?' = the event is less than {EXPOSED_DB:.0f} dB over its bed, so its excerpt is mostly bed and the tag a "
          'weak hint (right kind about 1 in 4 on the tuning mixes, 1 in 2 for exposed events; often wrong on music '
          "and voice). Where AST ran, its speech overrides the kind with VOICE. Section genres marked ? are weak. "
          "'over bed' = how far the event rises over the level just before it; 'silent' = that bed was below -60 "
          '(after a gap the rise means nothing). Full data: --json.']
    return '\n'.join(L)


# ------------------------------------------------------------------ main entry
def _src_hash():
    h = hashlib.sha1()
    for f in ('audio.py', 'features.py', 'video.py', 'config.py'):
        h.update(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), 'rb').read())
    return h.hexdigest()[:12]


CORE_ID = _src_hash()           # cached results never outlive a change of decode, features or cut detection


def _cue_key(cues):
    if cues is None:
        return 'none'
    if isinstance(cues, str) and os.path.exists(cues):
        return audio.file_id(cues)
    return hashlib.sha1(json.dumps(cues, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _is_video(path, pr):
    """a real picture stream: not the cover art of an audio file (one attached picture at 90000 'fps')"""
    fps = (pr.get('video') or {}).get('fps') or 0
    return bool(pr.get('has_video')) and 0 < fps <= 240 and os.path.splitext(path)[1].lower() not in audio.AUDIO_EXTS


def timeline(path, video=None, cues=None, start=None, end=None, detail='normal', cache=True, fresh=False):
    """listen through a soundtrack / film: events with tags, sections, loudness, beat grid, cut / cue sync,
    transcript. cache: read / write cached results and model outputs; fresh: recompute everything and store it"""
    if detail not in DETAIL:
        raise ValueError(f'detail must be one of {list(DETAIL)}')
    for f in (path, video):
        if f is not None and not os.path.isfile(f):
            raise FileNotFoundError(f'no such file: {f}')
    pr = audio.probe(path)
    if video is None and _is_video(path, pr):
        video = path
    elif video is not None and not _is_video(video, pr if video == path else audio.probe(video)):
        raise ValueError(f'--video {os.path.basename(video)} has no picture stream (an audio file?)')
    if not pr.get('codec') and not video:
        raise ValueError(f'{os.path.basename(path)} has no audio stream')
    cue_list, cue_fps = read_cues(cues)
    key = hashlib.sha1(f'{audio.file_id(path)}|{audio.file_id(video) if video else None}|{_cue_key(cues)}|{start}|'
                       f'{end}|{detail}|{CODE_ID}|{CORE_ID}'.encode()).hexdigest()[:24]
    cfile = os.path.join(config.EMB_CACHE, 'timeline', f'{key}.json')
    if cache and not fresh and os.path.exists(cfile):
        try:
            with open(cfile, encoding='utf-8') as fh:
                return dict(json.load(fh), cached=True, file=path)  # the path as given now
        except Exception:
            pass
    r = _analyse(path, pr, video, cue_list, cue_fps, start, end, DETAIL[detail], cache and not fresh)
    r['detail'] = detail
    r['summary'] = _summary(r)
    r['text'] = transcript(r)
    if cache or fresh:
        os.makedirs(os.path.dirname(cfile), exist_ok=True)
        tmp = f'{cfile}.{os.getpid()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(to_json(r))
        audio.replace_quietly(tmp, cfile)
    return dict(r, cached=False)


def _load_range(path, pr, start, end):
    """(x_context, c0, t0, t1, file_duration): a zoom [start, end] is analysed inside +-CONTEXT_S of context, so
    verdicts near its edges and the accent threshold match the full analysis"""
    name = os.path.basename(path)
    if not pr.get('codec'):                                    # picture only: analyse silence of the same length
        fdur = float(pr.get('duration') or 0)
        t0, t1 = max(0.0, float(start or 0)), min(fdur, float(end)) if end is not None else fdur
        return np.zeros((int(round(fdur * SR)), 2), np.float32), 0.0, t0, t1, fdur
    if start is None and end is None:
        x = audio.load(path)
        fdur = len(x) / SR
        if fdur < 0.1:
            raise ValueError(f'{name} is {fdur:.2f} s long: too short for a timeline (>= 0.1 s); for a one-shot use '
                             'earshot measure / describe')
        return x, 0.0, 0.0, fdur, fdur
    fdur = float(pr.get('duration') or 0) or audio.duration(path)
    t0, t1 = max(0.0, float(start or 0)), min(fdur, float(end)) if end is not None else fdur
    if t1 - t0 < 0.1:
        raise ValueError(f"nothing to analyse between {t0:g} s and {'the end' if end is None else f'{end:g} s'}: "
                         f'{name} is {fdur:.2f} s long')
    c0, c1 = max(0.0, t0 - CONTEXT_S), min(fdur, t1 + CONTEXT_S)
    return audio.load(path, start=c0, end=c1), c0, t0, t1, fdur


def _analyse(path, pr, video, cue_list, cue_fps, start, end, cfg, reuse):
    clock, t_all = {}, time.time()

    def lap(name, t):
        clock[name] = round(time.time() - t, 2)
        return time.time()
    t = time.time()
    xc, c0, t0, t1, fdur = _load_range(path, pr, start, end)
    wa, wb = t0 - c0, min(t1 - c0, len(xc) / SR)               # the window inside the context (context times)
    x = xc[int(round(wa * SR)): int(round(wb * SR))]
    dur = len(x) / SR
    t = lap('decode_s', t)
    loud = loudness(x, at_start=t0 <= 0.01, at_end=t0 + dur >= fdur - 0.01)
    t = lap('loudness_s', t)
    silent = bool(loud['digital_silence'])
    ev_c = [] if silent else detect_events(xc, lead=c0 == 0)
    lv = level_envs(xc)
    t = lap('events_s', t)
    grid = music_grid(audio.to_mono(xc))
    t = lap('beats_s', t)
    cuts, fps = [], None
    if video:
        from . import video as vid
        fps = vid.info(video).get('fps')
        cuts = [round(c - c0, 3) for c in vid.cuts(video) if t0 + 0.05 < c < t0 + dur - 0.05]
    cues_w = [dict(c, t=round(c['t'] - c0, 4)) for c in cue_list if t0 - 1e-6 <= c['t'] <= t0 + dur + 1e-6]
    t = lap('cuts_s', t)
    events = [e for e in ev_c if wa - 1e-6 <= e['t'] < wb]
    sid = span_id(path)
    tg = tag_events(xc, sid, c0, events, cfg['tag'], cfg['ast'], cuts + [c['t'] for c in cues_w], reuse, (wa, wb))
    t = lap('tag_events_s', t)
    hop = max(cfg['hop'], round((dur - cfg['win']) / max(1, cfg['max_win'] - 1), 2))
    if silent:
        secs, sn = ([{'start': 0.0, 'end': round(dur, 2), 'windows': 0, 'label': 'digital silence', 'lufs': None,
                      'digital_silence': True}] if dur >= 1.0 else []), {'windows': 0, 'ast': 0}
    else:
        secs, sn = sections(x, sid, t0, cfg['win'], hop, cfg['ast_sec'], reuse=reuse)
    t = lap('sections_s', t)
    beats = grid['beats'] if grid.get('confidence') in ('clear', 'weak') else None
    if beats:
        bt = np.asarray(beats)
        for e in events:
            e['beat_offset_ms'] = round((e['t'] - float(bt[int(np.argmin(np.abs(bt - e['t'])))])) * 1000, 1)
    sy = {}
    if cuts:
        sy['cuts'] = sync(cuts, ev_c, beats, 1.0 / fps if fps else None, lv)
    if cues_w:
        sy['cues'] = sync(cues_w, ev_c, beats, 1.0 / float(cue_fps) if cue_fps else None, lv)
    lap('total_s', t_all)
    _shift(loud, secs, t0)
    for e in events:
        e['t'] = round(e['t'] + c0, 3)
    for k in ('beats', 'downbeats'):
        grid[k] = [round(b + c0, 3) for b in grid[k] if wa - 1e-6 <= b <= wb + 1e-6]
    for v in sy.values():
        for it in v['items']:
            it['t'] = round(it['t'] + c0, 3)
            if it.get('event_t') is not None:
                it['event_t'] = round(it['event_t'] + c0, 3)
        v['stats']['no_sound'] = [round(s + c0, 3) for s in v['stats'].get('no_sound', [])]
    return {'file': path, 'video': video, 'start': round(t0, 3), 'end': round(t0 + dur, 3), 'duration': round(dur, 3),
            'file_duration': round(fdur, 3), 'context': [round(c0, 3), round(c0 + len(xc) / SR, 3)],
            'has_audio': bool(pr.get('codec')), 'lossy': bool(pr.get('lossy')), 'loudness': loud, 'music': grid,
            'events': events, 'sections': secs,
            'sync': sy, 'cuts': [round(c + c0, 3) for c in cuts],
            'cut_method': 'frame difference (earshot.video.cuts)' if video else None, 'fps': fps,
            'model_calls': {'clap_event_windows': tg['clap_windows'], 'ast_event_windows': tg['ast'],
                            'clap_section_windows': sn['windows'], 'ast_section_windows': sn['ast']},
            'timing_s': clock}


def _shift(loud, secs, t0):
    """window-relative loudness and section times -> file times"""
    def sh(v, nd=3):
        return round(v + t0, nd)
    for g in loud['silences']:
        g['start'], g['end'] = sh(g['start'], 2), sh(g['end'], 2)
    if 'max_momentary' in loud:
        loud['max_momentary']['t'] = sh(loud['max_momentary']['t'], 2)
    if 'max_short_term' in loud:
        ms = loud['max_short_term']
        ms['start'], ms['end'] = sh(ms['start'], 2), sh(ms['end'], 2)
    for k in ('momentary', 'short_term'):
        if k in loud:
            loud[k]['t0'] = sh(loud[k]['t0'])
    loud['clipping']['times'] = [sh(v) for v in loud['clipping']['times']]
    for s in secs:
        s['start'], s['end'] = sh(s['start'], 2), sh(s['end'], 2)


def _sync_sentence(kind, st, zoomed):
    k1 = 'hit_within_1_frame' if 'hit_within_1_frame' in st else 'hit_within_40ms'
    c1 = st.get(k1.replace('hit', 'chance'))
    lab = '1 frame' if 'frame' in k1 else '40 ms'
    s = (f"Sync to {st['n']} {kind}: {100 * st[k1]:.0f}% have an onset within {lab}"
         + (f" (chance {100 * c1:.0f}%)" if c1 is not None else '')
         + f", {100 * st['hit_within_100ms']:.0f}% within 100 ms")
    if st.get('median_abs_offset_ms') is not None:
        s += f", median |offset| {st['median_abs_offset_ms']:.0f} ms"
    ka = k1.replace('hit', 'accent')
    if ka in st:
        s += (f"; {100 * st[ka]:.0f}% land on a strong accent (>= {st['accent_min_db']:.0f} dB over the bed"
              + (', ranked over the +-30 s context' if zoomed else '') + f") within {lab} "
              f"(chance {100 * st[ka.replace('accent', 'accent_chance')]:.0f}%)")
    if 'on_beat' in st:
        s += f"; {100 * st['on_beat']:.0f}% on the beat grid (chance {100 * st['on_beat_chance']:.0f}%)"
    if st.get('answered_by_swell'):
        s += f"; {st['answered_by_swell']} met by a whoosh / riser peak"
    if st.get('answered_by_level_peak'):
        s += f"; {st['answered_by_level_peak']} only by a level peak (soft whoosh / swell?)"
    if st.get('no_sound'):
        s += f"; {len(st['no_sound'])} without a hit: " + ', '.join(fmt_t(v) for v in st['no_sound'][:6])
    return s + '.'


def _summary(r):
    lo, mu, ev = r['loudness'], r['music'], r['events']
    p = [f"{os.path.basename(r['file'])}: {fmt_t(r['start'])}-{fmt_t(r['end'])} ({r['duration']:.1f} s)."]
    if not r.get('has_audio', True):
        p.append(f"NO AUDIO STREAM: picture only ({len(r['cuts'])} picture cuts; every cut / cue is unsounded).")
    elif lo.get('digital_silence'):
        p.append('DIGITAL SILENCE: every sample is zero.')
    elif lo.get('integrated_lufs') is not None:
        s = f"Loudness {lo['integrated_lufs']:.1f} LUFS integrated, true peak {lo['true_peak_dbtp']:.1f} dBTP"
        if lo.get('lra_lu') is not None:
            s += f", LRA {lo['lra_lu']:.1f} LU"
        if lo.get('max_short_term'):
            ms = lo['max_short_term']
            s += f"; loudest {fmt_t(ms['start'])}-{fmt_t(ms['end'])} ({ms['lufs']:.1f} LUFS short-term)"
        p.append(s + '.')
    else:
        p.append(f"True peak {lo['true_peak_dbtp']:.1f} dBTP"
                 + (' (below the -70 LUFS gate: near silence).' if r['duration'] >= 0.4 else
                    ' (too short for a loudness measurement).'))
    if lo['clipping']['runs']:
        p.append(f"CLIPPING: {lo['clipping']['runs']} runs of clipped samples "
                 f"(first at {fmt_t(lo['clipping']['times'][0])}).")
    elif lo.get('true_peak_dbtp') is not None and r.get('lossy') and lo['true_peak_dbtp'] > 0.0:
        p.append(f"True peak {lo['true_peak_dbtp']:+.1f} dBTP on decoded lossy audio: above 0 dBTP it clips on playback "
                 '(the master should leave the mixer at <= -1 dBTP before encoding).')
    elif lo.get('true_peak_dbtp') is not None and r.get('lossy') and lo['true_peak_dbtp'] > -1.0:
        p.append(f"True peak {lo['true_peak_dbtp']:+.1f} dBTP on decoded lossy audio (codec overshoot included: fine for "
                 'a delivered file; a master should be <= -1 dBTP before encoding).')
    elif lo.get('true_peak_dbtp') is not None and lo['true_peak_dbtp'] > -1.0:
        p.append(f"True peak {lo['true_peak_dbtp']:.1f} dBTP is above -1 dBTP (risk of overs after encoding).")
    gaps = [g for g in lo['silences'] if not g.get('where')]
    edges = [g for g in lo['silences'] if g.get('where') and g['where'] != 'whole file']
    if gaps:
        p.append(f"{len(gaps)} stop-down(s)/gap(s): " + ', '.join(f"{fmt_t(g['start'])} {g['dur']:.1f} s {g['kind']}"
                                                              for g in gaps[:5]) + '.')
    if edges:
        p.append('Quiet ' + ' and '.join(f"{g['where']} {g['dur']:.1f} s ({g['kind']})" for g in edges) + '.')
    if mu.get('tempo') and mu['confidence'] != 'none':
        alt = f" or {mu['tempo_alt']:.0f}" if mu.get('tempo_alt') else ''
        p.append(f"Beat grid: ~{mu['tempo']:.0f}{alt} BPM ({mu['confidence']} pulse, clarity "
                 f"{mu['pulse_clarity']:.2f}, {len(mu['beats'])} beats).")
    elif not lo.get('digital_silence'):
        p.append('No steady beat grid' + (f" (pulse clarity {max(0.0, mu['pulse_clarity']):.2f} < {WEAK_PULSE})"
                                          if mu.get('pulse_clarity') is not None else '') + '.')
    if r['sections'] and not lo.get('digital_silence'):
        p.append(f"{len(r['sections'])} section(s): " + '; '.join(
            f"{fmt_t(s['start'])}-{fmt_t(s['end'])} {s['label']}" for s in r['sections'][:6]) + '.')
    tagged = [e for e in ev if e.get('kind')]
    if ev:
        kinds = {}
        for e in tagged:
            kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
        ks = ', '.join(f'{v} {k.lower()}' for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        p.append(f"{len(ev)} events ({len(ev) / max(r['duration'], 1e-6):.1f}/s); {len(tagged)} tagged"
                 + (f" ({ks}; tags are hints)" if tagged else '') + '.')
        acc = top_accents(ev)
        if acc:
            p.append('Strongest accents over a running bed: ' + ', '.join(
                f"{fmt_t(e['t'])} +{e['prominence_db']:.0f} dB" + (f" ({e['label']})" if e.get('label') else '')
                for e in acc) + '.')
    zoomed = r['context'] != [r['start'], r['end']]
    for kind in ('cuts', 'cues'):
        st = (r['sync'].get(kind) or {}).get('stats')
        if st and st.get('n'):
            p.append(_sync_sentence(kind, st, zoomed))
    return _ascii(' '.join(p))


def add_cli(sub):
    from .cli import common
    p = sub.add_parser('timeline', help='listen through a soundtrack / film: events, sections, loudness, beats, sync')
    p.add_argument('file', help='audio file or video (its audio is analysed, its picture cuts are used)')
    p.add_argument('--video', help='video for picture cuts (default: FILE itself when it is a video)')
    p.add_argument('--cues', help='cue events: JSON file {fps, events:[{f, kind, id}]}, a JSON list, or "1.5,3.2,..." s')
    p.add_argument('--from', '--start', dest='start', type=float, help='zoom: start (s)')
    p.add_argument('--to', '--end', dest='end', type=float, help='zoom: end (s)')
    p.add_argument('--detail', choices=list(DETAIL), default='normal',
                   help='model budget: low = 12 CLAP-tagged events, no AST (fastest); normal = 20 events + AST on the '
                        'top event and 3 section crops; high = up to 96 events (most important first), 16 AST event '
                        'labels, finer sections')
    p.add_argument('--fresh', action='store_true', help='recompute everything (no cached result or model outputs) and '
                                                       'store the new result')
    common(p)
    p.set_defaults(func=lambda a: emit(timeline(a.file, a.video, a.cues, a.start, a.end, a.detail, fresh=a.fresh), a)
                   and 0)


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
