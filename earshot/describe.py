"""What does this sound sound like? Calibrated vocabulary tags, AudioSet labels and key measurements in one paragraph.

    earshot describe key.ogg
    earshot describe ambience.ogg --start 30 --end 60 --json
    from earshot.describe import describe
    describe('enter.ogg')['summary']
    # '0.56 s short transient (attack 3 ms, decays in 190 ms). Sounds most like: a single mechanical keyboard key press
    #  (z +5.4), a soft key release click (z +3.6), a clicky mechanical keyboard (z +3.2) [clear reading]. ...'

Tags: CLAP against earshot/data/vocab.json, calibrated per sound and per term (earshot/vocab.py). Reading of the best
tag: z >= 3 clear, 2.5-3 plausible, 2-2.5 weak, below 2 tentative (match.VERDICTS; held-out evidence in
match.VERDICT_EVIDENCE and eval/semantics/RESULTS.md). 'Other readings' = the best tags of other categories: the
closest alternative interpretations. Music genre / mood, instrument and voice tags were never validated: they are
listed separately as hints (on 45 upbeat library tracks the top mood tag was a dark / sad / tense one for 18%).
AudioSet labels come from the AST tagger (independent per-class probabilities, hears up to 8 kHz only). Measurements
come from earshot.features (no model; brightness = median magnitude-weighted spectral centroid). Files longer than
20 s are described as a whole (mean of up to 6 evenly spaced 10 s windows, the same ones rank and match use) plus up
to 3 non-overlapping windows that read differently from each other. Near-silent input (peak below -70 dBFS) and
input shorter than 5 ms are reported as such, untagged.
"""
import warnings

import numpy as np

from . import audio, features, vocab
from .cli import common
from .match import SKIP_CATS, VERDICTS, verdict
from .report import fmt_t

LONG_S = 20.0
SILENT_DBFS = -70.0
MONO_DB = -40.0
MUSIC_P = 0.5
NOTE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _hz(f):
    return f'{f / 1000:.1f} kHz' if f >= 1000 else f'{f:.0f} Hz'


def _note(f):
    m = int(round(69 + 12 * np.log2(f / 440.0)))
    return f'{NOTE[m % 12]}{m // 12 - 1}'


def _ms(v):
    return f'{v / 1000:.1f} s' if v >= 2000 else f'{v:.0f} ms'


def shape(m):
    """a few words on the envelope / texture from the measurements"""
    d, att, dec, ons = m['duration'], m.get('attack_ms'), m.get('decay_ms'), m.get('onset_count')
    rate = m.get('onsets_per_s', 0)
    if d > LONG_S:
        words = ['long recording' if ons is None else f'busy, {rate:.1f} onsets/s' if rate >= 1.5 else
                 'steady, few onsets' if ons <= d / 10 else f'{ons} separate events']
        if m.get('lra_lu') is not None:
            words.append(f'loudness range {m["lra_lu"]:.0f} LU')
        return ', '.join(words)
    if d < 2.5 and (ons is None or ons <= 3):
        kind = 'transient' if att is not None and att <= 15 else 'swell' if att is not None and att >= 150 else 'one-shot'
        words = [f'short {kind}']
    elif rate >= 1.5:
        words = [f'busy, {rate:.1f} onsets/s']
    elif (ons or 0) >= 2:
        words = [f'{ons} separate events']
    else:
        words = ['sustained']
    if m.get('f0_hz'):
        words.append(f'tonal, pitch ~{m["f0_hz"]:.0f} Hz ({_note(m["f0_hz"])})')
    elif m.get('flatness', 0) >= 0.3:
        words.append('noise-like')
    env = ([f'attack {att:.0f} ms'] if att is not None else []) + ([f'decays in {_ms(dec)}'] if dec is not None else [])
    return ', '.join(words) + (f' ({", ".join(env)})' if env else '')


MEAS_KEYS = ['duration', 'peak_dbfs', 'true_peak_dbtp', 'rms_dbfs', 'lufs', 'lra_lu', 'attack_ms', 'decay_ms',
             'brightness_hz', 'bandwidth_hz', 'noise_floor_dbfs', 'flatness', 'harmonic_ratio', 'f0_hz',
             'onset_count', 'onsets_per_s', 'width', 'digital_silence_frac']


def measurements(m):
    """the measurements worth showing: no noise floor under 1 s (it would be the padding), width 'mono' below -40 dB"""
    out = dict(m)
    sm = m.get('side_to_mid_db')
    out['width'] = 'mono' if sm is None or sm < MONO_DB else f'side/mid {sm:.0f} dB'
    if m['duration'] < 1.0:
        out.pop('noise_floor_dbfs', None)
    return {k: out[k] for k in MEAS_KEYS if out.get(k) is not None}


def _meas_text(m):
    s = [f'peak {m["peak_dbfs"]:.1f} dBFS']
    if m.get('lufs') is not None:
        s.append(f'{m["lufs"]:.1f} LUFS')
    if m.get('brightness_hz'):
        s.append(f'brightness {_hz(m["brightness_hz"])}')
    if m.get('bandwidth_hz'):
        s.append(f'bandwidth {_hz(m["bandwidth_hz"])}')
    nf = m.get('noise_floor_dbfs')
    if nf is not None and m['duration'] <= LONG_S:
        s.append('digital silence between sounds' if nf < -120 else f'noise floor {nf:.0f} dBFS')
    s.append(m['width'])
    return ', '.join(s)


def _ast_top(p, t, k=5, min_p=0.05):
    top = t.top(p, k)
    return [{'label': l, 'p': pr} for i, (l, pr) in enumerate(top) if pr >= min_p or i == 0]


def _ast_text(top):
    labels = ', '.join(f"{d['label']} {d['p']:.2f}" for d in top[:3])
    return f'AudioSet: {labels}.' if top[0]['p'] >= 0.2 else f'AudioSet tagger unsure ({labels}).'


def reading(z):
    v = verdict(z)
    return {'strong': 'clear reading', 'plausible': 'plausible reading', 'weak': 'weak reading'}.get(
        v, 'tentative: nothing in the vocabulary stands out')


def _tags(v, a, top):
    """(validated top tags, unvalidated hints, best 3 per category)"""
    rows = v.top_terms(a, k=len(v.names), exclude=SKIP_CATS)
    best = [dict(term=t, category=c, z=z, cosine=cos, verdict=verdict(z))
            for t, c, z, cos in rows if c not in vocab.UNVALIDATED][:top]
    hints = [dict(term=t, category=c, z=z, cosine=cos) for t, c, z, cos in rows if c in vocab.UNVALIDATED][:top]
    per = {cat: [dict(term=t, z=z, cosine=cos) for t, _, z, cos in v.top_terms(a, k=3, cats=(cat,))] for cat in v.cats}
    return best, hints, per


def distinct_windows(spans, W, tags, n=3):
    """up to n non-overlapping windows, farthest from the file's mean embedding first, each with a different top tag"""
    if len(spans) <= 1:
        return []
    mean = W.mean(0)
    d0 = 1 - W @ (mean / (np.linalg.norm(mean) + 1e-12))
    picked = []
    for i in np.argsort(-d0):
        if len(picked) >= n:
            break
        if all(spans[i][1] <= spans[j][0] + 1e-6 or spans[i][0] >= spans[j][1] - 1e-6 for j in picked) and \
                tags[i] not in {tags[j] for j in picked}:
            picked.append(int(i))
    return sorted(picked) if len(picked) > 1 else []


def _ast():
    from .models import ast as ast_mod
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ast_mod.get()


def _untagged(res, why):
    res.update(shape=why, top_tags=[], music_hints=[], tags={}, audioset=[], character=[], summary=why)
    return res


def describe(path, start=None, end=None, top=8):
    """tags (calibrated), AudioSet labels and measurements of a sound or a [start, end) segment of it"""
    vocab.check(path, start, end)
    x = audio.load(path, start=start, end=end)
    dur = len(x) / audio.SR
    s0 = start or 0.0
    m = features.measure(x, light=dur > 60)
    res = {'path': path, 'start': start, 'end': end, 'duration': round(dur, 3)}
    if m.get('error'):
        res['measurements'] = {'duration': round(dur, 3)}
        return _untagged(res, f'{dur * 1000:.1f} ms of audio: too short to describe or measure (needs at least 5 ms).')
    meas = measurements(m)
    res['measurements'] = meas
    if m['peak_dbfs'] < SILENT_DBFS:
        return _untagged(res, f'{dur:.2f} s of near-silence (peak {m["peak_dbfs"]:.0f} dBFS): nothing to describe; '
                              f'model tags of silence are meaningless.')
    v = vocab.get()
    t = _ast()
    xm = audio.to_mono(x)
    if dur > LONG_S:
        spans = vocab.spans_for(s0, s0 + dur)
        W = vocab.embed_spans([(path, a, b) for a, b in spans])
        a = vocab._norm(W.mean(0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pw = t.probs([xm[int((a_ - s0) * audio.SR):int((b_ - s0) * audio.SR)] for a_, b_ in spans])
        p = pw.max(0)
    else:
        a = vocab.embed_file(path, start, end)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = t.probs_file(path, start, end)
    best, hints, per = _tags(v, a, top)
    ast_top = _ast_top(p, t)
    p_music = t.prob_of(p, 'Music')
    other, seen = [], {b['category'] for b in best[:3]}
    for b in best[3:]:
        if b['category'] not in seen and len(other) < 2:
            other.append(b)
            seen.add(b['category'])
    char = [d['term'].split(' ', 1)[1].removesuffix(' sound') for d in per.get('quality') or []
            if d['z'] >= VERDICTS[1][0]][:2]
    res.update(shape=shape(m), top_tags=best, music_hints=hints, tags=per, audioset=ast_top, character=char)
    s = f'{dur:.2f} s {res["shape"]}.'
    if p_music >= MUSIC_P:
        gen = [h for h in hints if h['category'] == 'music_genre'][:2]
        mood = [h for h in hints if h['category'] == 'music_mood'][:2]
        s += (f' Music (AudioSet Music {p_music:.2f}). Unvalidated hints - genre: '
              + ', '.join(f"{h['term']} (z {vocab.zf(h['z'])})" for h in gen)
              + '; mood (wrong for ~1 in 5 tracks in a test): ' + ', '.join(f"{h['term']} (z {vocab.zf(h['z'])})"
                                                                             for h in mood) + '.')
    if best[0]['z'] >= VERDICTS[-1][0]:
        s += (' Sounds most like: ' + ', '.join(f"{b['term']} (z {vocab.zf(b['z'])})" for b in best[:3])
              + f' [{reading(best[0]["z"])}].')
        if other:
            s += ' Other readings: ' + ', '.join(f"{b['term']} ({b['category']}, z {vocab.zf(b['z'])})"
                                                 for b in other) + '.'
    else:
        s += (' No sound-effect description stands out (closest: ' + ', '.join(
            f"{b['term']} z {vocab.zf(b['z'])}" for b in best[:2]) + '; tentative).')
    if p_music < MUSIC_P and hints and hints[0]['z'] >= VERDICTS[0][0]:
        s += f" Unvalidated hint: {hints[0]['term']} (z {vocab.zf(hints[0]['z'])})."
    if char:
        s += ' Character: ' + ', '.join(char) + '.'
    s += ' ' + _ast_text(ast_top) + ' ' + _meas_text(meas) + '.'
    if dur > LONG_S:
        wt = [_tags(v, W[i], 1)[0][0] for i in range(len(spans))]
        wins = []
        for i in distinct_windows(spans, W, [w['term'] for w in wt]):
            wins.append({'start': round(spans[i][0], 3), 'end': round(spans[i][1], 3), 'top_tag': wt[i],
                         'audioset': _ast_top(pw[i], t, k=3, min_p=0.1)})
        res['distinct_windows'] = wins
        if wins:
            s += ' Windows that read differently: ' + '; '.join(
                f"{fmt_t(w['start'])}-{fmt_t(w['end'])} {w['top_tag']['term']} (z {vocab.zf(w['top_tag']['z'])}; "
                f"AudioSet {w['audioset'][0]['label']} {w['audioset'][0]['p']:.2f})" for w in wins) + '.'
        else:
            s += f" Uniform: all {len(spans)} analysed 10 s windows read as {wt[0]['term']}."
    res['summary'] = s
    return res


def _text(r):
    lines = [r['summary']]
    if r['top_tags']:
        lines += ['', 'top tags (z calibrated: >= 3 clear, 2.5-3 plausible, 2-2.5 weak, < 2 tentative):']
        lines += [f"  {vocab.zf(b['z']):>5}  {b['term']}  [{b['category']}]  cos {b['cosine']:.3f}" for b in r['top_tags']]
        if r['music_hints']:
            lines.append('unvalidated hints (music genre / mood, instruments, voice): ' + ', '.join(
                f"{h['term']} ({vocab.zf(h['z'])})" for h in r['music_hints'][:5]))
        lines.append('best per category: ' + '; '.join(f"{c}: {d[0]['term']} ({vocab.zf(d[0]['z'])})"
                                                        for c, d in r['tags'].items() if d))
        lines.append('audioset: ' + ', '.join(f"{d['label']} {d['p']:.2f}" for d in r['audioset']))
    lines.append('measurements: ' + ', '.join(f'{k} {v}' for k, v in r['measurements'].items()))
    return '\n'.join(lines)


def add_cli(sub):
    p = sub.add_parser('describe', help='what does a sound sound like: calibrated tags, AudioSet labels, measurements')
    p.add_argument('path', help='audio file or video')
    p.add_argument('--start', type=float, help='describe only from this time (s), e.g. a trim from find')
    p.add_argument('--end', type=float, help='describe only up to this time (s)')
    p.add_argument('--top', type=int, default=8, help='tags per category (default 8)')
    common(p)
    p.set_defaults(func=lambda a: vocab.run_cli(lambda: describe(a.path, a.start, a.end, a.top), a, _text))


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
