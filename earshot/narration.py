"""earshot narration: time a film's picture to its voiceover (no models, no word recognition).

    earshot narration vo.wav script.json [--lead-in 0.4] [--tail 2.5] [--shift 0] [--envelope 30] [--srt subs.srt]

    from earshot import narration
    plan = narration.sync('vo.wav', 'script.json')     # scenes, cues, subtitles in film time

The voiceover is read as loudness only: 20 ms RMS frames, voiced where louder than 8 % of the file's 95th percentile
frame (about -22 dB under its speech), trimmed of the silence before and after the speech; every silent stretch of
PAUSE_MIN_S or longer inside the speech is a pause candidate. The script is turned into time weights (a letter or
digit 1, a space 1, two or more spaces / , ; : 3, . ! ? or a line break 6, combining marks such as Thai vowel and tone
marks 0) spread over the speech. Each scene cut is snapped to a pause within SCENE_SNAP_S of where the text puts it,
chosen jointly for all cuts (in order, scenes at least MIN_SCENE_S long, closer and longer pauses preferred); a cut
with no pause in reach stays where the text puts it. Cues and subtitle lines are placed the same way inside their
scene and snapped to a speech onset (the end of a pause) or a pause within CUE_SNAP_S. This is an estimate from
pauses and text length, not recognition: check the cuts it reports as 'text' (and any cue far from a pause).

script: a .json file
    {"lead_in": 0.4, "tail": 2.5,
     "scenes": [{"id": "s1", "text": "...", "nudge": 0.0,
                 "cues": [{"id": "logo_hit", "at": "the words it lands on", "kind": "hit"}]}, ...]}
  (or a bare list of scenes), or a .txt / .md file with scenes separated by blank lines. In either, a marker {name} in
  the text is a cue at that spot (removed from the text). nudge moves the scene's first cut by seconds after snapping.
Film time: the speech starts at lead_in; the film ends tail seconds after the speech ends; the voiceover file's
sample 0 plays at vo_offset_s (negative: start the file that far in). --shift moves every picture event (cuts, cues,
subtitles) against the voiceover: + = the picture later.
"""
import os
import re
import unicodedata

import numpy as np

from . import audio

SR = audio.SR
FRAME_S = 0.02
THRESH_FRAC = 0.08       # voiced: frame RMS over this fraction of the 95th percentile frame
FLOOR_DBFS = -60.0       # never call quieter than this voiced
BLIP_S = 0.04            # a voiced run this short inside silence (a click, a breath) does not break a pause
PAUSE_MIN_S = 0.22
SCENE_SNAP_S = 1.2
MIN_SCENE_S = 1.2
CUE_SNAP_S = 0.4
LINE_MAX_CHARS = 42      # a subtitle line longer than this is split at the space nearest its middle
LEAD_IN_S = 0.4
TAIL_S = 2.5
LONG_PAUSE_S = 0.8       # a pause this long earns the full preference bonus
PAUSE_BONUS_S = 0.3

_MARK = re.compile(r'\{([A-Za-z0-9_.:-]+)\}')
_STOP = set('.!?…。！？ฯ๚๛')   # incl. Thai paiyannoi / angkhankhu / khomut
_COMMA = set(',;:、，；：')
_BREAK = _STOP | _COMMA | {'\n'}


class NarrationError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------- the script
def _scene(raw, i):
    if isinstance(raw, str):
        raw = {'text': raw}
    if not isinstance(raw, dict) or not isinstance(raw.get('text'), str) or not raw['text'].strip():
        raise NarrationError(f'scene {i + 1}: needs a non-empty "text"')
    text, marks, pos = '', [], 0
    for m in _MARK.finditer(raw['text']):          # {name} markers: cues at that spot, removed from the text
        text += raw['text'][pos:m.start()]
        marks.append({'id': m.group(1), 'char': len(text)})
        pos = m.end()
    text += raw['text'][pos:]
    cues = []
    for j, c in enumerate(raw.get('cues') or []):
        if not isinstance(c, dict) or not isinstance(c.get('at'), str) or not c['at'].strip():
            raise NarrationError(f'scene {i + 1}, cue {j + 1}: needs "at", the words the cue lands on')
        cues.append({'id': str(c.get('id') or f'cue{j + 1}'), 'at': c['at'], 'kind': c.get('kind')})
    nudge = raw.get('nudge', 0.0)
    if not isinstance(nudge, (int, float)) or isinstance(nudge, bool):
        raise NarrationError(f'scene {i + 1}: "nudge" must be a number of seconds')
    return {'id': str(raw.get('id') or f's{i + 1}'), 'text': text, 'marks': marks, 'cues': cues,
            'nudge': float(nudge)}


def load_script(script):
    """a script path (.json, or .txt / .md with blank-line-separated scenes) or an already loaded dict / list"""
    opts = {}
    if isinstance(script, str):
        if not os.path.exists(script):
            raise NarrationError(f'no such script: {script}')
        with open(script, encoding='utf-8-sig') as fh:
            body = fh.read()
        if script.lower().endswith('.json'):
            import json
            try:
                script = json.loads(body)
            except json.JSONDecodeError as e:
                raise NarrationError(f'{script}: not valid JSON ({e})') from None
        else:
            script = [s for s in re.split(r'\n\s*\n', body.replace('\r\n', '\n')) if s.strip()]
    if isinstance(script, dict):
        opts = {k: script[k] for k in ('lead_in', 'tail') if k in script}
        script = script.get('scenes')
    if not isinstance(script, list) or not script:
        raise NarrationError('the script needs at least one scene ({"scenes": [{"text": ...}]})')
    scenes = [_scene(s, i) for i, s in enumerate(script)]
    ids = [s['id'] for s in scenes]
    if len(set(ids)) != len(ids):
        raise NarrationError('scene ids must be unique')
    return scenes, opts


def weights(text):
    """time weight of every character of `text` (same length): speech letters ~1, pauses written as punctuation more"""
    w = np.zeros(len(text))
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            j = i
            while j < len(text) and text[j].isspace():
                j += 1
            run = text[i:j]
            w[i] = 6.0 if '\n' in run else (3.0 if j - i >= 2 else 1.0)
            i = j
            continue
        cat = unicodedata.category(ch)
        if ch in _STOP:
            w[i] = 6.0
        elif ch in _COMMA:
            w[i] = 3.0
        elif cat in ('Mn', 'Me', 'Cf'):
            w[i] = 0.0
        else:
            w[i] = 1.0
        i += 1
    return w


def lines(text, max_chars=LINE_MAX_CHARS):
    """subtitle lines: (start_char, end_char) spans split after punctuation / line breaks, long ones at a space"""
    spans, a = [], 0
    for i, ch in enumerate(text):
        if ch in _BREAK:
            spans.append((a, i + 1))
            a = i + 1
    spans.append((a, len(text)))
    out = []

    def split(a, b):
        seg = text[a:b]
        if len(seg.strip()) <= max_chars:
            out.append((a, b))
            return
        mid = (a + b) / 2
        sp = [a + k for k, c in enumerate(seg) if c == ' ' and a + k not in (a, b - 1)]
        if not sp:
            out.append((a, b))
            return
        k = min(sp, key=lambda s: abs(s - mid))
        split(a, k)
        split(k, b)
    for a, b in spans:
        if text[a:b].strip():
            split(a, b)
    return out


# ---------------------------------------------------------------------------------------------- the voiceover
def analyse(x, sr=SR):
    """frame RMS, voiced mask, speech start / end and the pauses inside the speech of a mono signal"""
    hop = int(round(FRAME_S * sr))
    n = len(x) // hop
    if n < 3:
        raise NarrationError('the voiceover is shorter than 60 ms')
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1))
    thr = max(THRESH_FRAC * float(np.percentile(rms, 95)), 10 ** (FLOOR_DBFS / 20))
    voiced = rms > thr
    blip = max(1, int(round(BLIP_S / FRAME_S)))
    runs = _runs(voiced)
    for a, b, v in runs:                    # drop isolated clicks / breaths between silences
        if v and b - a <= blip and 0 < a and b < n:
            voiced[a:b] = False
    if not voiced.any():
        raise NarrationError('no speech found in the voiceover (it is silent or too quiet)')
    first = int(np.argmax(voiced))
    last = n - int(np.argmax(voiced[::-1]))
    pmin = int(round(PAUSE_MIN_S / FRAME_S))
    pauses = [{'start': a * FRAME_S, 'end': b * FRAME_S, 'mid': (a + b) / 2 * FRAME_S, 'len': (b - a) * FRAME_S}
              for a, b, v in _runs(voiced[first:last], first) if not v and b - a >= pmin]
    return {'rms': rms, 'voiced': voiced, 'thr': thr, 'start': first * FRAME_S, 'end': last * FRAME_S,
            'pauses': pauses, 'dur': len(x) / sr}


def _runs(mask, off=0):
    out, a = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[a]:
            out.append((a + off, i + off, bool(mask[a])))
            a = i
    return out


def snap(expected, cands, window, min_gap, lo, hi):
    """move each expected boundary onto a candidate (t, len) within `window`, jointly: in order, at least `min_gap`
    apart and from lo / hi; closer and longer candidates win; a boundary with none in reach keeps its time. Returns
    [(time, candidate index or None)]; the expected times as they are if no arrangement keeps the gaps."""
    if not expected:
        return []
    opts = []
    for e in expected:
        o = [(e, None, window)]
        for j, (t, ln) in enumerate(cands):
            if abs(t - e) <= window:
                o.append((t, j, abs(t - e) - PAUSE_BONUS_S * min(ln / LONG_PAUSE_S, 1.0)))
        opts.append(o)
    inf = float('inf')
    cost = [[(c if t - lo >= min_gap - 1e-9 else inf) for t, _, c in opts[0]]]
    back = [[None] * len(opts[0])]
    for k in range(1, len(opts)):
        row, brow = [], []
        for t, _, c in opts[k]:
            best, arg = inf, None
            for p, (tp, _, _) in enumerate(opts[k - 1]):
                if cost[-1][p] < best and t - tp >= min_gap - 1e-9:
                    best, arg = cost[-1][p], p
            row.append(best + c)
            brow.append(arg)
        cost.append(row)
        back.append(brow)
    last = [(c if hi - opts[-1][i][0] >= min_gap - 1e-9 else inf) for i, c in enumerate(cost[-1])]
    i = int(np.argmin(last))
    if last[i] == inf:
        return [(e, None) for e in expected]
    pick = []
    for k in range(len(opts) - 1, -1, -1):
        pick.append(opts[k][i][:2])
        i = back[k][i]
    return pick[::-1]


# --------------------------------------------------------------------------------------------------- the sync
def sync(vo, script, lead_in=None, tail=None, shift=0.0, envelope_fps=None):
    """time the scenes, cues and subtitle lines of `script` to the voiceover `vo` (a path, or a mono float array at
    48 kHz). Returns a dict: summary, film duration, voiceover placement, scenes, cues, subtitles, pauses and an
    optional loudness envelope (envelope_fps frames per second of film, 0..1, for mouths and music ducking)."""
    scenes, opts = load_script(script)
    lead_in = float(opts.get('lead_in', LEAD_IN_S) if lead_in is None else lead_in)
    tail = float(opts.get('tail', TAIL_S) if tail is None else tail)
    if lead_in < 0 or tail < 0:
        raise NarrationError('lead_in and tail must be 0 or more seconds')
    x = audio.load(vo, mono=True) if isinstance(vo, str) else np.asarray(vo, np.float32)
    an = analyse(x)
    s0, s1, pauses = an['start'], an['end'], an['pauses']

    # where the text puts every cut: the script's weights spread over the speech (a scene break weighs like a '.')
    ws = [weights(s['text']) for s in scenes]
    tot = [float(w.sum()) for w in ws]
    sep = 6.0
    total = sum(tot) + sep * (len(scenes) - 1)
    if total <= 0:
        raise NarrationError('the script has no words to time')
    per = (s1 - s0) / total
    expected, acc = [], 0.0
    for k in range(len(scenes) - 1):
        acc += tot[k]
        expected.append(s0 + (acc + sep / 2) * per)
        acc += sep
    cand = [(p['mid'], p['len']) for p in pauses]
    cuts = snap(expected, cand, SCENE_SNAP_S, MIN_SCENE_S, s0, s1)

    vo_off = lead_in - s0          # film time = vo time + vo_off (before --shift)
    dur = lead_in + (s1 - s0) + tail
    onsets = [s0] + [p['end'] for p in pauses]
    out_scenes, cues, subs = [], [], []
    for k, s in enumerate(scenes):
        # the scene's speech: from the end of the pause before its cut to the start of the pause after
        if k == 0:
            a = s0
        else:
            t, j = cuts[k - 1]
            a = pauses[j]['end'] if j is not None else t
        if k == len(scenes) - 1:
            b = s1
        else:
            t, j = cuts[k]
            b = pauses[j]['start'] if j is not None else t
        b = max(b, a + 1e-3)
        cw = np.concatenate([[0.0], np.cumsum(ws[k])])

        def at_char(c, a=a, b=b, cw=cw, tw=tot[k]):
            return a + (cw[min(c, len(cw) - 1)] / tw if tw > 0 else 0.0) * (b - a)

        def to_onset(t):
            near = [o for o in onsets if abs(o - t) <= CUE_SNAP_S and a - 1e-6 <= o <= b + 1e-6]
            return (min(near, key=lambda o: abs(o - t)), True) if near else (t, False)

        found = [{'id': m['id'], 'at': None, 'kind': None, 'char': m['char'], 'found': True} for m in s['marks']]
        low = s['text'].casefold()
        for c in s['cues']:
            i = low.find(c['at'].casefold().strip())
            found.append({'id': c['id'], 'at': c['at'], 'kind': c['kind'], 'char': max(i, 0), 'found': i >= 0})
        for c in found:
            t, snapped = to_onset(at_char(c['char'])) if c['found'] else (a, False)
            cues.append({'id': c['id'], 'scene': s['id'], 'at': c['at'], 'kind': c['kind'], 'found': c['found'],
                         't': round(t + vo_off + shift, 3), 'vo_t': round(t, 3), 'snapped': snapped})

        spans = lines(s['text'])
        inner = [(at_char(sp[0])) for sp in spans[1:]]
        inside = [(p['mid'], p['len']) for p in pauses if a < p['mid'] < b]
        ts = [a] + [t for t, _ in snap(inner, inside, CUE_SNAP_S, 0.3, a, b)] + [b]
        for i, (ca, cb) in enumerate(spans):
            subs.append({'scene': s['id'], 'text': s['text'][ca:cb].strip(),
                         'start': round(ts[i] + vo_off + shift, 3), 'end': round(ts[i + 1] + vo_off + shift, 3)})
        info = {'id': s['id']}
        if k > 0:
            t, j = cuts[k - 1]
            info.update(cut='pause' if j is not None else 'text', cut_shift_s=round(t - expected[k - 1], 3),
                        pause_s=round(pauses[j]['len'], 3) if j is not None else None)
        out_scenes.append(info)

    # film-time edges: the first scene from 0, the last to the end; cuts at their pause middles, plus shift / nudge
    edges = [0.0] + [t + vo_off + shift + scenes[k + 1]['nudge'] for k, (t, _) in enumerate(cuts)] + [dur]
    if scenes[0]['nudge']:
        raise NarrationError(f'scene {scenes[0]["id"]}: the first scene starts at 0; nudge its cut on the next scene')
    for k, sc in enumerate(out_scenes):
        sc['start'], sc['end'] = round(edges[k], 3), round(edges[k + 1], 3)
        if sc['end'] - sc['start'] < 0.1:
            raise NarrationError(f'scene {sc["id"]} is {sc["end"] - sc["start"]:.2f} s long after shift / nudge: '
                                 'move it less')
    for sub in subs:   # subtitles never outlive their scene's edges (a nudge moves a cut, not the words)
        sc = next(x for x in out_scenes if x['id'] == sub['scene'])
        sub['start'] = max(sub['start'], sc['start'])
        sub['end'] = max(sub['start'], min(sub['end'], sc['end']))
    subs = [x for x in subs if x['text']]
    res = {
        'duration': round(dur, 3), 'lead_in': lead_in, 'tail': tail, 'shift': shift,
        'vo_offset_s': round(vo_off, 3), 'speech_start_vo': round(s0, 3), 'speech_end_vo': round(s1, 3),
        'scenes': out_scenes, 'cues': cues, 'subtitles': subs,
        'pauses': [{k: round(v, 3) for k, v in p.items()} for p in pauses],
    }
    if envelope_fps:
        res['envelope_fps'] = envelope_fps
        res['envelope'] = envelope(an, vo_off, dur, envelope_fps)
    res['summary'] = _summary(res, len(scenes))
    return res


def envelope(an, vo_off, dur, fps):
    """the voiceover's loudness per film frame, 0..1 (1 = its 95th percentile frame): mouths, music ducking"""
    rms, n = an['rms'], int(np.ceil(dur * fps))
    ref = float(np.percentile(rms, 95)) or 1.0
    t = np.arange(n) / fps - vo_off
    idx = np.floor(t / FRAME_S).astype(int)
    ok = (idx >= 0) & (idx < len(rms))
    v = np.zeros(n)
    v[ok] = rms[idx[ok]] / ref
    v[ok & (rms[np.clip(idx, 0, len(rms) - 1)] <= an['thr'])] = 0.0
    return [round(float(min(x, 1.0)), 3) for x in v]


def _summary(r, n):
    from .report import fmt_t
    cuts = [s for s in r['scenes'][1:]]
    on_pause = sum(1 for s in cuts if s['cut'] == 'pause')
    lost = [c['id'] for c in r['cues'] if not c['found']]
    head = (f"film {r['duration']:.2f} s: speech {r['speech_end_vo'] - r['speech_start_vo']:.2f} s from "
            f"{r['lead_in']:.2f} s, {len(r['pauses'])} pauses >= {PAUSE_MIN_S * 1000:.0f} ms; "
            f"{on_pause}/{len(cuts)} scene cuts on a pause" + (f", {len(cuts) - on_pause} placed by text length"
                                                              if len(cuts) - on_pause else '')
            + f"; {len(r['cues'])} cues" + (f" ({len(lost)} not found in their scene: {', '.join(lost)})" if lost else '')
            + f"; {len(r['subtitles'])} subtitle lines. Voiceover file starts at {r['vo_offset_s']:+.3f} s. "
            "Estimated from pauses and text length, not word recognition: watch the cuts marked 'text'.")
    rows = [head]
    for s in r['scenes']:
        how = '' if 'cut' not in s else (f"  cut on a {s['pause_s'] * 1000:.0f} ms pause ({s['cut_shift_s']:+.2f} s "
                                         "from the text estimate)" if s['cut'] == 'pause' else
                                         '  cut by text length (no pause within '
                                         f'{SCENE_SNAP_S} s)')
        rows.append(f"  {s['id']}: {fmt_t(s['start'])} - {fmt_t(s['end'])}{how}")
    for c in r['cues']:
        rows.append(f"  cue {c['id']} ({c['scene']}): {fmt_t(c['t'])}" + (' on a speech onset' if c['snapped'] else '')
                    + ('' if c['found'] else f' (text "{c["at"]}" not found: at the scene start)'))
    return '\n'.join(rows)


def srt(subs):
    def ts(t):
        ms = int(round(t * 1000))
        return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'
    return '\n'.join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n" for i, s in enumerate(subs, 1))


def add_cli(sub):
    from .cli import common
    from .report import emit
    p = common(sub.add_parser('narration', help="time a film's scenes, cues and subtitles to its voiceover (no "
                                                 'models)'))
    p.add_argument('voiceover', help='the voiceover recording (any format ffmpeg reads)')
    p.add_argument('script', help='script.json ({"scenes": [{"id", "text", "cues": [{"id", "at"}]}]}) or a .txt '
                                  'with scenes separated by blank lines; {name} in the text marks a cue')
    p.add_argument('--lead-in', type=float, help=f'seconds of film before the speech (default {LEAD_IN_S})')
    p.add_argument('--tail', type=float, help=f'seconds of film after the speech, the end card (default {TAIL_S})')
    p.add_argument('--shift', type=float, default=0.0, help='move every picture event by seconds (+ = later)')
    p.add_argument('--envelope', type=int, metavar='FPS', help='also the voiceover loudness per film frame (0..1)')
    p.add_argument('--srt', help='also write the subtitle lines as an SRT file')

    def run(a):
        audio.check_paths([a.voiceover], 'voiceover')
        r = sync(a.voiceover, a.script, lead_in=a.lead_in, tail=a.tail, shift=a.shift, envelope_fps=a.envelope)
        if a.srt:
            with open(a.srt, 'w', encoding='utf-8') as fh:
                fh.write(srt(r['subtitles']))
        emit(r, a, text=r['summary'])
        return 0
    p.set_defaults(func=run)
