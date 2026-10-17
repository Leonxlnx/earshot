"""Launch-video sound scorecard: compare a soundtrack or film with professional launch films and say what to fix.

    earshot critique film.mp4                          # audio + its own picture (cuts and big visual changes)
    earshot critique mix.wav --video film.mp4          # a soundtrack checked against the picture it is for
    earshot critique film.mp4 --profile my_refs.json   # your own references (earshot profile ...)
    earshot critique film.mp4 --fast                   # skip the AudioSet and realness passes (seconds, not minutes)

    from earshot.critique import critique
    r = critique('film.mp4')
    r['verdict'], r['score'], r['actions'], r['metrics']

Each metric row: value, reference range, typical band (p10-p90), median, status ok / low / high / n.a., weight
(critical / major / minor; info rows are style choices that are shown but not scored), why it matters and concrete
actions with timestamps. Ranges come from the profile: the min-max of the reference films widened by a per-metric
tolerance (measurement precision; a density whose range would reach 0 keeps a floor at half the sparsest reference),
except loudness and true peak, which follow web-delivery practice (-16..-9 LUFS integrated; true peak <= -1 dBTP for a
master, <= 0 dBTP for audio decoded from AAC / MP3, which already contains the codec overshoot) and show the references
for context. p_real comes from earshot.realness (when installed) and uses its verdict thresholds: ok at 'real or
pro-produced', critical below 0.5 (synthesized is the more likely reading), minor in between; a minor realness miss with no 4 s
window leaning synthesized is a tip, not a fix (nothing to replace was found). The 4 s windows it lists use the
window (clip) thresholds, as `earshot realness` does. A mono mix (identical channels) scores as stereo width -40 dB. The verdict weighs critical 3, major 2, minor 1 (a critical reference / web
rule missed by <= 2 tolerances counts as major): launch-grade (>= 90 % and no critical issue), close (>= 75 %, no
critical issue), needs work (>= 50 %), else not launch-grade; --fast (no realness, no music presence) marks the verdict
partial. Files under 3 s or silent are not scored; under 10 s the loudness-shape rows are n.a.
Sound on cuts: the scored share counts visual events with an onset (a new attack) within +-100 ms, the definition the
reference ranges were measured with; a whoosh swelling through a cut or music already playing is not an onset. The
fixes and tips instead name the events that get no sound at all by earshot.timeline's matcher (no onset within
100 ms, no whoosh / riser peak within 300 ms, no level peak within 150 ms), so critique and timeline list the same
cuts. A file that is one of the profile's reference films (same duration and measurements) is scored leave-one-out,
against the other references, and says so. r['events'] lists the visual events, those without an onset within
+-100 ms, those with no sound at all (timeline matcher), those without a salient hit, the hits and the stop-downs
(seconds). Missing files, files without audio, a --video without picture and a bad --profile raise
earshot.profile.InputError with a plain message. How the references fare against leave-one-out ranges and which
known-bad mixes get caught: eval/launch/RESULTS.md; the playbook behind the metrics: docs/launch-video-sound.md.
"""
import os
import sys
import time
from functools import lru_cache

import numpy as np

from . import audio, profile as P
from .cli import common
from .report import emit, error_line, fmt_t

WEIGHT = {3: 'critical', 2: 'major', 1: 'minor', 0: 'info'}
W = {v: k for k, v in WEIGHT.items()}
# key: (weight, tolerance, sides checked, practice window or None, why it matters)
SPEC = {
    'lufs_i': (3, 1.0, 'both', (-16.0, -9.0),
               'how loud the film plays next to other launch videos: X, LinkedIn and site autoplay do not normalise, '
               'YouTube turns masters above -14 LUFS down'),
    'true_peak': (2, 0.5, 'upper', (None, -1.0),
                  'a master above -1 dBTP clips once encoded to AAC / MP3 (a decoded lossy file above 0 dBTP already '
                  'clips on playback): crackle on the biggest hits'),
    'p_real': (3, 0.0, 'lower', None,
               'the reference launch films use real recordings and produced music; next to them, code-synthesized '
               'sound reads as cheap and generic'),
    'music_frac': (3, 0.05, 'lower', None,
                   'the references are music-led: one continuous track carries the energy and hides edit seams'),
    'cuts_with_sound': (2, 0.05, 'lower', None,
                        'a cut or big visual change with no sound under it feels dead; viewers expect sound on motion'),
    'lra': (2, 1.0, 'both', None, 'how much the loudness swings: launch films keep the body consistently strong'),
    'crest_lu': (2, 0.5, 'both', None, 'how far the loudest 3 s stick out above the average level'),
    'first_sound_s': (2, 0.1, 'upper', None, 'silence at the start loses autoplay viewers: the references sound at once'),
    'end_silence_s': (2, 0.25, 'upper', None, 'dead air after the sound stops makes the ending feel unfinished'),
    'max_gap_s': (2, 0.25, 'upper', None, 'a silent hole mid-film reads as a mistake unless it is a deliberate stop-down'),
    'final_lu': (2, 1.5, 'both', None, 'the logo / end-card moment should land at body level or above, not fade away'),
    'onset_rate': (2, 0.3, 'both', None, 'rhythmic activity: too sparse feels empty, too dense feels cluttered'),
    'hit_rate': (2, 0.2, 'both', None, 'punch: salient hits mark the key moments; without them nothing lands'),
    'intro_lu': (1, 1.5, 'both', None, 'how the film opens: the references open close to body level (one builds up)'),
    'end_lu': (1, 1.5, 'both', None, 'how the last 10 % sit against the body'),
    'fade_s': (1, 0.5, 'both', None, 'launch films end on a sting or a hard out; long fade-outs lose the button'),
    'band_sub': (1, 1.5, 'both', None, 'sub (20-60 Hz) weight under hits and music'),
    'band_low': (1, 1.5, 'both', None, 'low end body (60-250 Hz)'),
    'band_lowmid': (1, 1.5, 'both', None, 'low-mids (250 Hz-1 kHz): too much = muddy, too little = hollow'),
    'band_mid': (1, 1.5, 'both', None, 'mids (1-4 kHz): where clicks and voice cut through phone speakers'),
    'band_presence': (1, 1.5, 'both', None, 'presence (4-8 kHz): bite of transients, clicks and typing'),
    'band_air': (1, 1.5, 'both', None, 'air (8-16 kHz): openness; missing air sounds muffled or low-bitrate'),
    'side_to_mid_db': (1, 1.0, 'both', None, 'stereo width: near-mono sounds small, too wide collapses on phones'),
    'speech_frac': (0, 0, 'both', None, 'voice share (4 of 7 references have a voice-over, 3 are music-only)'),
    'tempo': (0, 0, 'both', None, 'music tempo'),
    'peak_pos': (0, 0, 'both', None, 'where the loudest moment sits'),
    'tail_s': (0, 0, 'both', None, 'how long the final hit rings out'),
    'stopdowns_min': (0, 0, 'both', None, 'short drop-outs before hits (4 of 7 references use them)'),
    'cut_rate': (0, 0, 'both', None, 'edit pace (visual events per second)'),
    'cut_hit_frac': (0, 0, 'both', None, 'visual events with a salient hit (references: near chance)'),
    'cut_frame_sync': (0, 0, 'both', None, 'visual events with an onset within one frame (references: near chance)'),
    'cuts_on_beat': (0, 0, 'both', None, 'visual events on the beat grid (references: near chance)'),
    'brightness_hz': (0, 0, 'both', None, 'perceived brightness of the mix'),
    'correlation': (0, 0, 'both', None, 'L/R correlation (1 = mono)'),
}
BAND_FIX = {
    'band_sub': ('thin bottom: add a sub layer (sub drop or kick) under reveals, or pick music with real low end',
                 'too much 20-60 Hz rumble eats headroom and distorts small speakers: high-pass SFX and ambience at '
                 '30-40 Hz and cut the sub of the music by 2-4 dB'),
    'band_low': ('weak low end: the music has no body; choose a fuller track or add a low layer (bass, low hits)',
                 'boomy: cut 100-200 Hz on the music and on heavy SFX by 2-3 dB'),
    'band_lowmid': ('hollow: the 250 Hz-1 kHz body is scooped; lift it on the music or use fuller recordings',
                    'muddy / boxy: cut 300-500 Hz on the music bed and on roomy SFX by 2-3 dB'),
    'band_mid': ('recessed mids: clicks and voice will not cut through phone speakers; lift 1-3 kHz on SFX / VO',
                 'honky or harsh mids: cut 1-3 kHz on the loudest layer by 2 dB'),
    'band_presence': ('dull: transients, clicks and typing lack bite; brighten SFX or use brighter recordings',
                      'harsh / sibilant: cut 5-7 kHz or de-ess the brightest layer'),
    'band_air': ('no air: it sounds muffled or low-bitrate; use full-band recordings (44.1 / 48 kHz, no low-pass)',
                 'hissy / fizzy: low-pass or de-noise the noisiest layer'),
}


def _times(ts, k=8):
    ts = list(ts)
    s = ', '.join(fmt_t(t) for t in ts[:k])
    return s + (f' (+{len(ts) - k} more)' if len(ts) > k else '')


@lru_cache(maxsize=2)
def real_thresholds(file=True):
    """earshot.realness verdict thresholds {'lo', 'hi'}: the file-level ones for a long file's mean p_real (what
    critique scores), the clip-level ones for a single short clip; 0.5 / 0.5 when the model is not installed"""
    try:
        from .realness import load_model, thresholds_of
        t = thresholds_of(load_model(), file=file)
        return {'lo': float(t['lo']), 'hi': float(t['hi'])}
    except Exception:  # noqa: BLE001  (realness is optional)
        return {'lo': 0.5, 'hi': 0.5}


def _window(key, stats, lossy=False):
    """(lo, hi, basis) for a metric: practice window, or the reference hull widened by the tolerance"""
    w, tol, sides, practice, _ = SPEC[key]
    if key == 'p_real':
        return real_thresholds()['hi'], None, 'realness model'
    if key == 'true_peak' and lossy:
        return None, 0.0, 'decoded lossy'
    if practice:
        return practice[0], practice[1], 'web'
    if not stats or not stats.get('n'):
        return None, None, 'no reference data'
    lo, hi = stats['min'] - tol, stats['max'] + tol
    unit = P.METRICS[key][2]
    if lo <= 0 and unit in ('s', '/s', '/min', 'share'):
        lo = round(stats['min'] / 2, 3) if unit == '/s' and stats['min'] > 0 else None
    return (lo if sides in ('both', 'lower') else None), (hi if sides in ('both', 'upper') else None), 'references'


def _status(v, lo, hi):
    if v is None:
        return 'n.a.'
    if lo is not None and v < lo:
        return 'low'
    if hi is not None and v > hi:
        return 'high'
    return 'ok'


def _missing(events, targets, tol=0.1):
    targets = np.sort(np.asarray(targets, float))
    out = []
    for t in events:
        if not len(targets) or np.min(np.abs(targets - t)) > tol:
            out.append(t)
    return out


def _ranges(spans):
    """merge [(t0, t1)] that touch"""
    out = []
    for a, b in spans:
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _p(v, t):
    """p_real for text: 2 decimals, 3 when 2 would round onto a threshold it is actually below / above"""
    s = f'{v:.2f}'
    return f'{v:.3f}' if any(float(s) == x and abs(v - x) > 1e-9 for x in (t['lo'], t['hi'])) else s


def _realness_windows(f):
    """(4 s windows with a p_real, those at or below the window threshold sorted lowest first, window thresholds)"""
    rl = f.get('realness') or {}
    t = real_thresholds(file=False)  # each 4 s window is judged like a clip, as `earshot realness` does
    segs = [q for q in rl.get('segments') or [] if q.get('p_real') is not None]
    return segs, sorted((q for q in segs if q['p_real'] <= t['lo']), key=lambda q: q['p_real']), t


def _quiet_realness_miss(f, v):
    """a minor realness miss with nothing to replace: p_real between 0.5 and the 'real' line and no 4 s window
    leaning synthesized"""
    segs, _, _ = _realness_windows(f)
    return v is not None and v >= 0.5 and not any(q['p_real'] < 0.5 for q in segs)


def _unanswered(f):
    """(no sound at all, only a level peak) visual-event times, by earshot.timeline's matcher (None: not computed)"""
    sy = f.get('cut_sync')
    if not sy:
        return None, None
    items = sy['items']
    return ([i['t'] for i in items if i['status'] in ('no sound', 'off')], [i['t'] for i in items if i['status'] == 'peak'])


def _actions(key, st, v, f, a, refs):
    """concrete fixes for one out-of-range metric"""
    ev, dur, m = f['events'], f['duration'], f['metrics']
    rng = f"refs {P.fmt_value(key, refs.get('min'))} .. {P.fmt_value(key, refs.get('max'))}" if refs.get('n') else ''
    if key == 'lufs_i':
        if st == 'low':
            return [f'raise the master by {-14 - v:+.1f} dB to about -14 LUFS integrated (web) with a true-peak limiter '
                    f'(ceiling -1 dBTP); if the limiter works harder than ~3 dB, compress the music bus first ({rng})']
        return [f'turn the master down by {v + 12:.1f} dB to about -12 LUFS: above -9 LUFS normalising platforms turn it '
                f'down anyway and the limiter flattens every hit ({rng})']
    if key == 'true_peak':
        clip = ' (this decoded lossy file already clips on playback)' if f['lossy'] else ''
        return [f'true peak {v:+.1f} dBTP{clip}: end the master chain with a true-peak limiter, ceiling -1.0 dBTP, '
                'then re-encode']
    if key == 'p_real':
        rl = f.get('realness') or {}
        t = real_thresholds(file=bool(rl.get('segments')))  # the mean of a long file's windows: file-level thresholds
        segs, flagged, tw = _realness_windows(f)
        if _quiet_realness_miss(f, v):
            return []  # nothing synthetic found to replace: a tip (see _tips), not a fix
        if v <= t['lo']:
            head = f"realness: likely code-synthesized (p_real {_p(v, t)}, at or below the model's {t['lo']:.2f} threshold)"
        elif v < 0.5:
            head = (f"realness leans code-synthesized: p_real {_p(v, t)} (below 0.5 the model rates synthesized as more "
                    f"likely; realness calls {t['lo']:.2f}-{t['hi']:.2f} 'uncertain')")
        else:
            head = f"realness is uncertain (p_real {_p(v, t)}; real or pro-produced sound scores >= {t['hi']:.2f})"
        pick = []
        for q in flagged or sorted(segs, key=lambda q: q['p_real']):  # lowest first, so a cut list keeps the worst
            if all(abs(q['start'] - x['start']) >= 2.0 for x in pick):
                pick.append(q)
        pick = sorted(pick[:6 if flagged else 3], key=lambda q: q['start'])
        spans = ', '.join(f"{fmt_t(q['start'])}-{fmt_t(q['end'])} ({q['p_real']:.2f})" for q in pick)
        more = f' (+{len(flagged) - len(pick)} more)' if flagged and len(flagged) > len(pick) else ''
        where = (f"; 4 s windows at or below the window threshold {tw['lo']:.2f} (code-synthesized): {spans}{more}"
                 if flagged else f'; lowest-scoring 4 s windows: {spans}' if spans else '')
        fix = ('if the film should sound recorded or produced, replace oscillator / noise layers with real recordings '
               'and produced music (skills/real-sound has the sourcing and licence workflow)' if v < 0.5 else
               'if the film should sound recorded or produced, find and replace the layer that sounds synthetic')
        return [f'{head}{where}: {fix}; judge each stem with `earshot realness stem.wav` (a finished mix is harder to '
                'judge than its sources)']
    if key == 'music_frac':
        gaps = _ranges([(t, min(dur, t + P.AST_WIN)) for t, mu, _ in ev['music_windows'] or [] if mu < P.MUSIC_P])
        where = ', '.join(f'{fmt_t(x)}-{fmt_t(y)}' for x, y in gaps[:6])
        return [f'music is heard in only {100 * v:.0f}% of 5 s windows ({rng}); no music at {where}: run one '
                'continuous track under the whole film and duck it under VO instead of stopping it']
    if key == 'cuts_with_sound':
        miss = _missing(ev['visual'], a['onsets'])
        none, peak = _unanswered(f)
        if none is None:
            return [f'{len(miss)} of {len(ev["visual"])} visual events have no onset (new attack) within +-100 ms ({rng}): '
                    f'add a whoosh into the transition (peak on the cut), a hit or a UI click at {_times(miss)}']
        where = (f'no sound at all at {_times(none)}: add a whoosh into the transition (peak on the cut), a hit or a UI '
                 'click there' if none else 'every one has some sound under it')
        thru = len(miss) - len(none)
        return [f'{len(miss)} of {len(ev["visual"])} visual events have no onset (new attack) within +-100 ms ({rng}); '
                f'{where}' + (f'; {thru} have sound playing through them (a swell, music or a level peak) but no new '
                              'attack: add an accent on the ones that matter' if thru > 0 else '')]
    if key in ('lra', 'crest_lu'):
        ts, Ls = a['ts'], a['Ls']
        body = (ts > 0.1 * dur) & (ts < 0.9 * dur)
        if st == 'high' and body.any():
            iq = np.where(body)[0][int(np.argmin(Ls[body]))]
            il = int(np.argmax(Ls))
            quiet = f'{fmt_t(max(0, ts[iq] - 1.5))}-{fmt_t(ts[iq] + 1.5)} at {Ls[iq]:.1f} LUFS short-term'
            if key == 'lra':
                return [f'the loudness swings {v:.1f} LU (LRA; {rng}): the quietest body stretch is {quiet}, the loudest 3 s '
                        f'sit around {fmt_t(ts[il])} ({Ls[il]:.1f} LUFS); lift the quiet parts or compress the music bus '
                        f'2-4 dB so the body holds']
            return [f'the loudest 3 s (around {fmt_t(ts[il])}, {Ls[il]:.1f} LUFS) stand {v:.1f} LU above the integrated '
                    f'level ({rng}): the rest of the film plays quiet next to it; lift the body (quietest stretch {quiet}) '
                    f'or tame that peak']
        if key == 'lra':
            return [f'loudness range (LRA) only {v:.1f} LU ({rng}): the level never moves; give the film sections (a '
                    'calmer VO or UI passage, a lift into the reveal) or ease the limiter']
        return [f'the loudest 3 s are only {v:.1f} LU above integrated ({rng}): no moment stands out; let the main reveal '
                '/ logo hit rise 2-3 LU above the body or ease the limiter']
    if key == 'first_sound_s':
        pre = a['Lf'][a['tf'] < v]
        lvl = P._energy_mean(pre) if len(pre) else None
        if lvl is not None and lvl > -60:
            return [f'nothing reaches body level until {fmt_t(v)} (before that the sound sits around {lvl:.0f} LUFS, '
                    f"{m['lufs_i'] - lvl:.0f} LU under the body; {rng}): raise the opening or start the music at 0:00"]
        return [f'silence until {fmt_t(v)} ({rng}): start the music, ambience or a riser at 0:00']
    if key == 'end_silence_s':
        return [f'{v:.1f} s of dead air after the sound ends at {fmt_t(dur - v)}: trim the picture or let the final sting '
                f'ring out over the end card ({rng})']
    if key == 'max_gap_s':
        Lf, tf, I = a['Lf'], a['tf'], m['lufs_i']
        h = float(tf[1] - tf[0])
        gaps, run = [], []
        for t, q in zip(tf, Lf < I - 30):
            if q:
                run.append(t)
            elif run:
                if len(run) * h >= 0.5 and run[0] > (m['first_sound_s'] or 0) and t < dur - (m['end_silence_s'] or 0):
                    gaps.append((run[0] - 0.05, run[-1] + 0.05))
                run = []
        where = ', '.join(f'{fmt_t(x)}-{fmt_t(y)}' for x, y in gaps[:6])
        return [f'silent holes inside the film (longest {v:.1f} s; {rng}): {where}; keep a music bed or room tone under '
                'them, or make each one a deliberate stop-down that ends on a hit']
    if key == 'final_lu':
        last = ev['visual'][-1] if ev['visual'] else None
        if st == 'low':
            at = f' at the last visual event ({fmt_t(last)})' if last is not None else ''
            return [f'the final moment sits {-v:.1f} LU below the body: end on a logo sting / hit{at} instead of fading '
                    f'away ({rng})']
        return [f'the final hit is {v:.1f} LU above the body: pull it down 2-3 dB so it does not jump out ({rng})']
    if key == 'onset_rate':
        if st == 'low':
            return [f'sparse: {v:.1f} onsets/s ({rng}): use music with more rhythmic drive, or add UI foley (clicks, '
                    f'typing, ticks) on the on-screen actions']
        return [f'busy: {v:.1f} onsets/s ({rng}): thin out the SFX layer and keep accents for the important moments']
    if key == 'hit_rate':
        if st == 'low':
            big = sorted(zip(ev['visual_strength'], ev['visual']), reverse=True)
            cand = _missing([t for _, t in big], ev['hits'])[:6]
            at = f' - the biggest visual events without one: {_times(sorted(cand))}' if cand else ''
            return [f"{'no' if v == 0 else 'few'} salient hits ({v:.2f}/s, {rng}): add impacts / accents (a hit, a sub "
                    f'drop, a snappy UI click) on the key moments{at}']
        return [f'too many salient hits ({v:.2f}/s, {rng}): keep the biggest for reveals, soften the rest by 3-6 dB']
    if key in ('intro_lu', 'end_lu'):
        span = f'0:00-{fmt_t(0.1 * dur)}' if key == 'intro_lu' else f'{fmt_t(0.9 * dur)}-{fmt_t(dur)}'
        part = 'opening' if key == 'intro_lu' else 'ending'
        if st == 'low':
            return [f'the {part} ({span}) sits {-v:.1f} LU below the body ({rng}): ' +
                    ('start closer to full level or shorten the build-up' if key == 'intro_lu' else
                     'keep the energy up into the end card and finish on a sting')]
        return [f'the {part} ({span}) is {v:.1f} LU louder than the body ({rng}): bring it down or lift the body']
    if key == 'fade_s':
        if st == 'high':
            return [f'{v:.1f} s fade-out ({rng}): end on a sting with a short ring-out, or cut hard on the last picture cut']
        return [f'the end is cut hard; let the last hit ring out briefly ({rng})']
    if key in BAND_FIX:
        return [BAND_FIX[key][0 if st == 'low' else 1] + f' ({P.fmt_value(key, v)} share, {rng})']
    if key == 'side_to_mid_db':
        if st == 'low':
            what = ('the mix is mono (identical left and right channels' if f.get('mono') else
                    f'nearly mono (side {v:.1f} dB vs mid')
            return [f'{what}; {rng}): use stereo music and ambience, pan SFX with the motion']
        return [f'very wide (side {v:.1f} dB vs mid; {rng}): check the mono fold-down, narrow wide ambience / reverbs']
    return []


def _tips(f, a, flagged, p_real=None):
    """not scored: craft suggestions drawn from what the references do"""
    ev, m, out = f['events'], f['metrics'], []
    none, peak = _unanswered(f)
    if none and m.get('cuts_with_sound') is not None and 'cuts_with_sound' not in flagged:
        out.append(f'{len(none)} of {len(ev["visual"])} visual events get no sound at all (no onset within 100 ms, no '
                   f'whoosh / riser peak within 300 ms, no level peak within 150 ms; the matcher earshot timeline uses): '
                   f'{_times(none)}; give the ones that matter a whoosh into the cut, a hit or a UI click, or keep one as '
                   'a deliberate stop-down' + (f'; {len(peak)} more are met only by a level peak (a soft swell, or just '
                                                f'the bed: check {_times(peak, 4)} on the sheet)' if peak else ''))
    if p_real is not None and 'p_real' not in flagged and p_real < real_thresholds()['hi'] and \
            _quiet_realness_miss(f, p_real):
        segs, _, _ = _realness_windows(f)
        low = min((q['p_real'] for q in segs), default=None)
        t = real_thresholds()
        out.append(f"realness not confirmed (p_real {_p(p_real, t)}, under the {t['hi']:.2f} 'real or pro-produced' line) "
                   'but nothing synthetic was found' + (f' (no 4 s window leans synthesized; lowest {low:.2f})' if segs
                                                        else '') +
                   ': no layer to replace on this evidence; a finished mix is harder to judge than its sources, so '
                   'check the stems with `earshot realness stem.wav` if you have them')
    if m.get('stopdowns_min') == 0 and ev['visual']:
        big = max(zip(ev['visual_strength'], ev['visual']))[1]
        out.append(f'no stop-downs: 4 of 7 references drop the sound out for 0.2-1 s right before a big hit; try one '
                   f'before the biggest visual change at {fmt_t(big)}')
    if m.get('speech_frac') is not None and m['speech_frac'] >= 0.5:
        out.append(f'voice detected in {100 * m["speech_frac"]:.0f}% of windows: duck music 3-6 dB under the voice and '
                   'keep SFX out of 1-4 kHz while it speaks')
    return out


def _load(profile):
    if isinstance(profile, dict):
        return profile
    if profile and not os.path.isfile(profile):
        raise P.InputError(f'profile not found: {profile}')
    try:
        prof = P.load_profile(profile)
    except (ValueError, UnicodeDecodeError) as e:
        raise P.InputError(f'{profile} is not a profile JSON (make one with `earshot profile`): {e}') from None
    if not isinstance(prof, dict) or not isinstance(prof.get('metrics'), dict):
        raise P.InputError(f'{profile} is not a profile JSON (no "metrics"; make one with `earshot profile`)')
    return prof


def _na_reason(key, f, use_ast, use_realness):
    if key == 'p_real':
        return 'skipped by --fast' if not use_realness else 'realness unavailable'
    if key == 'music_frac':
        return 'skipped by --fast' if not use_ast else 'no AudioSet windows'
    if key == 'cuts_with_sound':
        return 'no picture: pass --video film.mp4' if not f['video'] else 'fewer than 3 visual events'
    if key in P.SHAPE and f['short']:
        return f'clip under {P.SHORT_S:.0f} s; the references are 20-93 s films'
    return 'not measurable on this file'


def critique(path, profile=None, video=None, use_ast=True, use_realness=True):
    """scorecard of a soundtrack / film against a reference profile (default: the bundled 7 launch films)"""
    t0 = time.time()
    if video and os.path.abspath(video) == os.path.abspath(path) and not audio.probe(path).get('has_video'):
        video = None
    for p_, need in ((path, 'audio'), (video, 'video')):
        if p_ and (why := P.media_problem(p_, need)):
            raise P.InputError(why)
    prof = _load(profile)
    dur, why = P.unscorable(path)
    if why:
        msg = (f'{os.path.basename(path)} {why}: nothing to score against launch films. For a single sound use '
               '`earshot describe` / `earshot realness`.')
        return {'path': path, 'video': video, 'duration': round(dur, 3), 'verdict': 'n.a.', 'score': None,
                'in_range': 0, 'scored': 0, 'critical_issues': 0, 'major_issues': 0, 'partial': [], 'actions': [],
                'tips': [], 'metrics': [], 'summary': msg, 'text': msg}
    f = P.film_metrics(path, video=video, use_ast=use_ast)
    a = P._audio_pass(path)
    ref = reference_match(f, prof)
    if ref:  # one of the profile's own films: score it against the others (its values must not set the ranges)
        prof = leave_one_out(prof, ref)
    f['cut_sync'] = cut_sync(path, f)
    vals = dict(f['metrics'])
    vals['p_real'] = None
    if use_realness:
        try:
            from .realness import realness
            rl = realness(path)
            f['realness'] = {k: rl.get(k) for k in ('p_real', 'verdict', 'synth_windows', 'segments', 'summary')}
            vals['p_real'] = rl.get('p_real')
        except Exception as e:  # noqa: BLE001  (realness is optional)
            f['realness'] = {'error': f'{type(e).__name__}: {e}'}
    sc = scorecard(vals, prof, f['chance'], f['lossy'])
    actions = []
    for row in sc['metrics']:
        if row['status'] in ('low', 'high'):
            w, tol = W[row['weight']], SPEC[row['metric']][1]
            lo, hi = row['range']
            edge = lo if row['status'] == 'low' else hi
            row['actions'] = _actions(row['metric'], row['status'], row['value'], f, a,
                                      prof['metrics'].get(row['metric'], {}))
            for act in row['actions']:
                actions.append({'metric': row['metric'], 'weight': WEIGHT[w], 'w': w,
                                'dist': abs(row['value'] - edge) / max(tol, 1e-3), 'action': act})
    actions.sort(key=lambda x: (-x['w'], -x['dist']))
    ev = f['events']
    partial = [n for n, k in (('realness', 'p_real'), ('music presence', 'music_frac')) if vals.get(k) is None]
    res = {'path': path, 'video': f['video'], 'duration': f['duration'], 'profile_n': prof.get('n'),
           'lossy': f['lossy'],
           'profile': 'bundled launch films' if profile is None else
           (profile if isinstance(profile, str) else 'custom')}
    res.update({k: v for k, v in sc.items() if k != 'metrics'})
    res.update(partial=partial,
               actions=[{'weight': x['weight'], 'metric': x['metric'], 'action': x['action']} for x in actions],
               tips=_tips(f, a, {x['metric'] for x in actions}, vals.get('p_real')), metrics=sc['metrics'],
               reference_match=ref and {'id': ref, 'scored_against': prof.get('n')},
               events={'visual': ev['visual'], 'visual_without_onset': _missing(ev['visual'], a['onsets']),
                       'visual_without_any_sound': _unanswered(f)[0],
                       'visual_without_hit': _missing(ev['visual'], ev['hits']), 'hits': ev['hits'],
                       'stopdowns': ev['stopdowns']},
               arc={'film': f.get('arc'), **{f'refs_{q}': v for q, v in (prof.get('arc') or {}).items()}},
               realness=f.get('realness'), seconds=round(time.time() - t0, 1))
    na = {}
    for r in sc['metrics']:
        if r['status'] == 'n.a.' and r['weight'] != 'info':
            na.setdefault(_na_reason(r['metric'], f, use_ast, use_realness), []).append(r['metric'])
    res['not_measured'] = na
    first = actions[0]['action'] if actions else None
    res['summary'] = (f"{os.path.basename(path)} vs {prof.get('n')} reference films ({res['profile']}): "
                      + (f'{_ref_note(res)} ' if ref else '') + f"{_verdict_line(res)}" +
                      (f" Fix first: {first}{'' if first.endswith('.') else '.'}" if first else
                       ' Nothing to fix against the references.') +
                      (' Not measured: ' + '; '.join(f"{', '.join(k)} ({why})" for why, k in na.items()) + '.'
                       if na else ''))
    res['text'] = critique_text(res)
    return res


def reference_match(f, prof):
    """id of the profile film this file is (same duration within 50 ms and the same loudness, true peak, brightness
    and mid-band share), else None"""
    m = f['metrics']
    tol = {'lufs_i': 0.15, 'true_peak': 0.15, 'brightness_hz': 15.0, 'band_mid': 0.15}
    for g in prof.get('films') or []:
        if abs(float(g.get('duration') or -1) - float(f['duration'])) > 0.05:
            continue
        both = [k for k in tol if m.get(k) is not None and (g.get('metrics') or {}).get(k) is not None]
        if len(both) >= 3 and all(abs(m[k] - g['metrics'][k]) <= tol[k] for k in both):
            return g.get('id') or 'a reference'
    return None


def leave_one_out(prof, ref):
    """the profile rebuilt from every film except `ref` (needs at least 2 others; else the profile unchanged)"""
    others = [dict(g, video=g.get('has_video', True), style=g.get('style', '')) for g in prof.get('films') or []
              if g.get('id') != ref]
    if len(others) < 2:
        return prof
    return dict(P.aggregate(others), loo_of=ref, n_full=prof.get('n'))


def cut_sync(path, f):
    """each visual event matched to the sound that answers it with earshot.timeline's matcher (onset, whoosh / riser
    peak, level peak), so critique and timeline name the same unsounded cuts; None without picture events"""
    vis = f['events'].get('visual') or []
    if not vis:
        return None
    try:
        from . import timeline as T
        x = audio.load(path)
        fps = f.get('fps')
        return T.sync(vis, T.detect_events(x), None, 1.0 / fps if fps else None, T.level_envs(x))
    except Exception:  # noqa: BLE001  (the onset-based list is the fallback)
        return None


def _ref_note(res):
    rm = res.get('reference_match') or {}
    return (f"NOTE: this file is {rm.get('id')} of the profile (same duration and measurements), so it is scored "
            f"against the other {rm.get('scored_against')} references (leave-one-out): its own values do not set the "
            'ranges.')


def _verdict_line(res):
    part = f" (partial: {' and '.join(res['partial'])} not checked)" if res.get('partial') else ''
    return (f"{res['verdict']}{part} - {res['in_range']} of {res['scored']} scored metrics in range (weighted "
            f"{100 * (res['score'] or 0):.0f}%), {res['critical_issues']} critical and {res['major_issues']} major "
            'issues.')


def scorecard(vals, prof, chance=None, lossy=False):
    """status of every metric in `vals` against a profile + weighted score and verdict (no actions; used by
    critique() and by the leave-one-out evaluation). lossy=True: the audio was decoded from AAC / MP3 / Opus, so true
    peak is judged against 0 dBTP (codec overshoot is already in it) instead of the -1 dBTP master rule."""
    rows = []
    for key, (w, tol, sides, practice, why) in SPEC.items():
        stats = prof['metrics'].get(key, {})
        lo, hi, basis = _window(key, stats, lossy)
        v = vals.get(key)
        st = 'n.a.' if basis == 'no reference data' else _status(v, lo, hi) if w else ('n.a.' if v is None else 'info')
        if key == 'p_real' and st == 'low':
            w = 3 if v < 0.5 else 1
        elif w == 3 and st in ('low', 'high') and abs(v - (lo if st == 'low' else hi)) <= 2 * tol:
            w = 2  # a near miss of a critical rule counts as major
        real = key == 'p_real'
        row = {'metric': key, 'label': 'realness (p_real)' if real else P.METRICS[key][1],
               'group': 'realness' if real else P.METRICS[key][0], 'value': v, 'unit': 'p' if real else P.METRICS[key][2],
               'range': [lo, hi], 'basis': basis, 'median': stats.get('median'),
               'typical': [stats.get('p10'), stats.get('p90')] if stats.get('n') else None,
               'status': st, 'weight': WEIGHT[w], 'why': why}
        if chance and key in chance:
            row['chance'] = chance[key]
        rows.append(row)
    scored = [(W[r['weight']], r) for r in rows if r['weight'] != 'info' and r['status'] != 'n.a.']
    wsum = sum(w for w, _ in scored)
    score = sum(w for w, r in scored if r['status'] == 'ok') / wsum if wsum else None
    n_crit = sum(1 for w, r in scored if w == 3 and r['status'] != 'ok')
    n_major = sum(1 for w, r in scored if w == 2 and r['status'] != 'ok')
    verdict = ('n.a.' if score is None else 'launch-grade' if score >= 0.9 and not n_crit else
               'close' if score >= 0.75 and not n_crit else 'needs work' if score >= 0.5 else 'not launch-grade')
    return {'verdict': verdict, 'score': None if score is None else round(score, 3),
            'in_range': sum(1 for _, r in scored if r['status'] == 'ok'), 'scored': len(scored),
            'critical_issues': n_crit, 'major_issues': n_major, 'metrics': rows}


def _range_txt(r):
    lo, hi = r['range']
    k = r['metric']
    fv = (lambda x: f'{x:.2f}') if k == 'p_real' else (lambda x: P.fmt_value(k, x))
    if r['weight'] == 'info':
        t = r['typical']
        return f"refs p10-p90 {fv(t[0])} .. {fv(t[1])}" if t and t[0] is not None else ''
    if lo is None and hi is None:
        return r['basis']
    s = f'>= {fv(lo)}' if hi is None else f'<= {fv(hi)}' if lo is None else f'{fv(lo)} .. {fv(hi)}'
    return s + (f" ({r['basis']})" if r['basis'] != 'references' else '')


def critique_text(res):
    lines = [f"earshot critique: {os.path.basename(res['path'])} ({res['duration']:.1f} s) vs {res['profile_n']} "
             f"reference films ({res['profile']})"] + ([_ref_note(res)] if res.get('reference_match') else []) + \
        [f'VERDICT: {_verdict_line(res)}']
    if res.get('partial'):
        lines.append('  (run without --fast before shipping: realness and music presence are critical checks)'
                     if len(res['partial']) == 2 else f"  ({' and '.join(res['partial'])} could not be measured)")
    if res['actions']:
        lines.append('FIX FIRST (most important first):')
        for i, x in enumerate(res['actions'], 1):
            lines.append(f"  {i}. [{x['weight']}] {x['action']}")
    for tip in res['tips']:
        lines.append(f'  tip: {tip}')
    if res.get('not_measured'):
        lines.append('NOT MEASURED: ' + '; '.join(f"{', '.join(k)} ({why})" for why, k in res['not_measured'].items()))
    lines.append(f"SCORECARD {'':30s} {'value':>11s}   {'ok range':30s} {'ref median':>11s}  status")
    for r in res['metrics']:
        if r['weight'] == 'info':
            continue
        v = ('n.a.' if r['value'] is None else _p(r['value'], real_thresholds()) if r['metric'] == 'p_real' else
             P.fmt_value(r['metric'], r['value']))
        med = '' if r['median'] is None else P.fmt_value(r['metric'], r['median'])
        lines.append(f"  {r['group'][:8]:8s} {r['label'][:30]:30s} {v:>11s}   {_range_txt(r)[:30]:30s} {med:>11s}  "
                     f"{r['status']}{'' if r['status'] in ('ok', 'n.a.') else ' (' + r['weight'] + ')'}")
    arc = res.get('arc') or {}
    if arc.get('film') and arc.get('refs_median'):
        fa = lambda xs: ' '.join('   -  ' if x is None else f'{x:+6.1f}' for x in xs)
        lines.append('LOUDNESS ARC (LU vs integrated at 0%, 10% .. 100% of the film; not scored):')
        lines.append(f"  this film    {fa(arc['film'])}")
        lines.append(f"  refs median  {fa(arc['refs_median'])}")
    lines.append('NOT SCORED (style choices, shown for context):')
    for r in res['metrics']:
        if r['weight'] != 'info' or r['value'] is None:
            continue
        ch = f" (chance {P.fmt_value(r['metric'], r['chance'])})" if r.get('chance') is not None else ''
        lines.append(f"  {r['label'][:44]:44s} {P.fmt_value(r['metric'], r['value']) + ch:>24s}   {_range_txt(r)}")
    rl = res.get('realness') or {}
    if rl.get('summary'):
        lines.append('realness: ' + rl['summary'])
    elif rl.get('error'):
        lines.append('realness: not available (' + rl['error'][:120] + ')')
    return '\n'.join(lines)


def _cmd(args):
    try:
        r = critique(args.file, profile=args.profile, video=args.video, use_ast=not args.fast,
                     use_realness=not args.fast)
    except (P.InputError, audio.AudioError) as e:
        error_line('critique', e)
        return 1
    emit(r, args)


def add_cli(sub):
    p = sub.add_parser('critique', help='launch-video sound scorecard against professional reference films',
                       description='Measure a soundtrack or film (loudness, true peak, arc, ending, music presence, '
                       'onset / hit density, sound on cuts, spectrum, stereo, realness) and compare it with a profile '
                       'of professional launch films: status per metric and concrete fixes with timestamps.')
    p.add_argument('file', help='soundtrack or film')
    p.add_argument('--video', help='the picture the soundtrack is for (default: the file itself if it is a video)')
    p.add_argument('--profile', help='profile JSON from `earshot profile` (default: the bundled 7 launch films)')
    p.add_argument('--fast', action='store_true', help='skip the AudioSet (music / speech) and realness passes')
    common(p)
    p.set_defaults(func=_cmd)


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
