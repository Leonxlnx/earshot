"""Knobs: aim one bank sound at a film's numbers through its own recipe parameters (never by resampling).

    from earshot.forge import bank, knobs
    knobs.describe(bank.get('whoosh_fast/crisp'))      # {'dur': 'shape', 'apex': True, 'pitch': None, 'why': ...}
    y, meta = bank.render('whoosh_fast/crisp', dur=0.8, apex=0.32)    # bank.render calls knobs.apply
    y, meta = bank.render('pro_sub_drop/ref', pitch='G1')            # f1 = 49 Hz, f0 keeps the glide ratio
    knobs.hz('Eb5'), knobs.hz(622.25), knobs.hz('622.25hz'), knobs.note(622.25)

dur (s, the length of the file):
  shape   the whoosh family (whoosh_fast, whoosh_big, pro_whoosh_big, swish, swipe): rise, hold and fall scale
          together, so the pass keeps its shape; with apex as well, rise comes from apex and hold / fall from the rest
  param   the recipe's own length key (riser, downlifter, sub drops, shimmer, zip: `length`; glitch and typing runs:
          `dur`; pro_logo_sting: `g_length`; the UI sounds' end gate: `length`; a spectral fit: its `length` knob):
          scaled, rendered, measured, then corrected along the secant through the renders so far, at most MAX_FIX
          times (a key that turns out not to move the length falls back to trim)
  trim    no length key (impact, toggle, typing_key, logo_sting, pluck: its file always holds 3 s of ring and room,
          struck fits): the sound as it is
  Every dur lands exactly: after the renders, a longer file is faded out to the length (a raised-cosine fade over a
  quarter of the length, 10-150 ms; with apex as well the fade starts at least APEX_KEEP after the apex) and a
  shorter one is padded with silence; edits['dur'] says which and how much.
apex (s, the loudest 5 ms): the whoosh family only. The pass engine's peak is analytic, tp = rise * (|start_db| / 30)
  ** (1 / rise_k) (recipes/whoosh.py envelope()): rise is solved for the target, the render measured and rise corrected
  by target / measured, at most MAX_FIX times, keeping the closest render; place_apex() then moves that render's peak
  onto the target exactly (it trims the quiet approach, a head at least 20 dB under the peak, or pads silence before
  it). Measured: the closed form lands within ~12 ms on the fast family, but the 5 ms peak of a broad fluttering pass
  moves non-monotonically with rise (whoosh_big / pro_whoosh_big swell: +-50 ms for a 10 % change), so the search
  alone cannot promise APEX_TOL (5 ms); the placement does, in whole 1 ms steps of the peak measure. With dur as
  well, the candidate renders are ranked on apex error plus tail-length error, and the apex is measured again on the
  file after its fit to dur (edits['apex']: search_err_ms, fit, measured, err_ms, ok: what the delivered file does).
pitch ('C5', 'Eb6', 'F#4' or 'Fs4', 440, '440hz'): the note goes into the recipe's own pitch parameter (PITCH: the
  pluck's f0, a bell's head fundamental, a sub drop's landing pitch f1 with f0 keeping the glide ratio, a tap's body
  resonance with its knock and glass partials, a riser's tonal layer ...), the parameters of the same body scaled by
  the same ratio. The note must lie inside the parameter's designed range (the forge's search range, widened where the
  model allows it); noise sounds (whooshes, swishes, swipes, glitches, keys) and members whose tonal layer is off are
  refused with the reason. edits['pitch']['measured_hz'] = the strongest spectral peak near the target in the render
  (None when there is no clear one), cents_off, and ok (within PITCH_TOL_CENTS; None where the sound is not measured:
  risers, impacts); a take whose pitch is off carries a 'pitch_off' warning in its checks (pitch_warning()).
  Clicks and toggles (click, pro_click, toggle): `pitch` is the centre of their tink partials and noise band, not the
  partial you hear (click/soft: 2454 Hz, heard 2903 Hz). Their note is the measured partial (meta pitch_hz, measured
  when the bank was built; null = no clear partial, so no pitch knob); a target is landed by measurement: the
  parameter is scaled by target / heard, rendered, measured and corrected (at most MAX_FIX times), keeping the closest.
Level (peak_db, weight) is bank.render's, after the knobs.
"""
import re

import numpy as np

SR = 48000
APEX_TOL = 0.005          # s: the apex is met within this
APEX_KEEP = 0.005         # s: the fade that fits a whoosh to dur starts at least this long after its apex
PITCH_TOL_CENTS = 50      # a measured pitch within this of the target is ok
PARTIAL_SPAN_ST = 6.0     # the heard partial of a click / toggle: the strongest peak within this of its `pitch`
DUR_REL = 0.01            # param / shape renders stop within 1 % of dur (or DUR_ABS); the fit to the sample is exact
DUR_ABS = 0.005
MAX_FIX = 2               # corrections after the first targeted render
SCALE = (0.15, 6.0)       # a length key moves at most this far from the member's own value
WHOOSH = ('whoosh_fast', 'whoosh_big', 'pro_whoosh_big', 'swish', 'swipe')
WHOOSH_MAX_S = 4.4        # the pass engine renders at most 4.5 s

# category -> the parameter dur scales (absent: trim)
LENGTH = {'riser': 'length', 'pro_riser': 'length', 'downlifter': 'length', 'sub_drop': 'length',
          'pro_sub_drop': 'length', 'shimmer': 'length', 'pro_shimmer': 'length', 'pro_zip': 'length',
          'glitch': 'dur', 'pro_glitch': 'dur', 'pro_typing_run': 'dur', 'pro_logo_sting': 'g_length',
          'click': 'length', 'pro_click': 'length', 'hover_tick': 'length', 'pro_hover_tick': 'length',
          'tap': 'length', 'pro_tap': 'length', 'pro_pop_in': 'length', 'pop': 'length', 'pro_pop': 'length',
          'notify': 'length', 'pro_notify': 'length', 'pro_typing_key': 'length'}

# category -> pitch spec: param (the note), co (scaled with it), range (Hz), needs (param, minimum: the tonal layer
# must be on), measure ('start': the first 0.6 s, 'end': the last half, None: not measured), mode 'pc' = a pitch class,
# partial: the parameter is not the heard partial (module doc: clicks and toggles)
PITCH = {
    'pluck': {'param': 'f0', 'range': (130.0, 2100.0)},
    'notify': {'param': 'f0', 'range': (220.0, 3200.0)},
    'pro_notify': {'param': 'f0', 'range': (600.0, 3200.0)},
    'click': {'param': 'pitch', 'range': (1200.0, 9500.0), 'partial': True},
    'pro_click': {'param': 'pitch', 'range': (1200.0, 9500.0), 'partial': True},
    'toggle': {'param': 'pitch', 'range': (1600.0, 8000.0), 'partial': True},
    'hover_tick': {'param': 'ping_f', 'co': ('tick_f',), 'range': (3000.0, 15000.0), 'needs': ('ping', 0.1)},
    'pro_hover_tick': {'param': 'ping_f', 'co': ('tick_f',), 'range': (3000.0, 15000.0), 'needs': ('ping', 0.1)},
    'tap': {'param': 'body_f', 'co': ('knock_f', 'glass_f'), 'range': (70.0, 220.0)},
    'pro_tap': {'param': 'body_f', 'co': ('knock_f', 'glass_f'), 'range': (70.0, 220.0)},
    'pro_pop_in': {'param': 'body_f', 'co': ('knock_f', 'glass_f'), 'range': (70.0, 220.0)},
    'pop': {'param': 'f0', 'co': ('f_hi', 'f_lo', 'f_blip'), 'range': (400.0, 3500.0)},
    'pro_pop': {'param': 'f0', 'co': ('f_hi', 'f_lo', 'f_blip', 'n_f0', 'n_f1'), 'range': (280.0, 1400.0)},
    'sub_drop': {'param': 'f1', 'co': ('f0',), 'range': (28.0, 62.0), 'measure': 'end'},
    'pro_sub_drop': {'param': 'f1', 'co': ('f0',), 'range': (28.0, 62.0), 'measure': 'end'},
    'pro_zip': {'param': 'f1', 'co': ('f0', 'fpk'), 'range': (80.0, 12000.0), 'measure': 'end'},
    'riser': {'param': 'tf0', 'range': (60.0, 4000.0), 'needs': ('tone_db', -40.0), 'measure': None},
    'pro_riser': {'param': 'tf0', 'range': (60.0, 4000.0), 'needs': ('tone_db', -40.0), 'measure': None},
    'downlifter': {'param': 'tf0', 'range': (150.0, 2500.0), 'needs': ('tone_db', -40.0), 'measure': None},
    'logo_sting': {'param': 'root', 'range': (300.0, 2200.0)},
    'pro_logo_sting': {'param': 'g_root', 'co': ('root',), 'range': (50.0, 1800.0)},
    'impact': {'param': 'body_f1', 'co': ('body_f0', 'sub_f'), 'range': (34.0, 75.0), 'measure': None},
    'pro_impact': {'param': 'body_f1', 'co': ('body_f0', 'sub_f'), 'range': (34.0, 75.0), 'measure': None},
    'pro_whoosh_big': {'param': 'sub_f', 'range': (25.0, 120.0), 'needs': ('sub_db', -30.0), 'measure': 'end'},
    'shimmer': {'param': 'key', 'mode': 'pc'},
    'pro_shimmer': {'param': 'key', 'mode': 'pc'},
}
UNTUNED = {
    'whoosh_fast': 'a noise pass-by: its band moves with the motion, it has no note',
    'whoosh_big': 'a noise pass-by: its band moves with the motion, it has no note',
    'swish': 'a noise flick: its whistle follows the speed, not a note',
    'swipe': 'a friction noise: it has no note',
    'glitch': 'a digital stutter: no note to tune',
    'pro_glitch': 'a digital stutter: no note to tune',
    'typing_key': 'keys are not tuned (takes and seeds give each press its own sound)',
    'pro_typing_key': 'designed key ticks are not tuned (takes give each press its own sound)',
    'pro_typing_run': 'a typed run is not tuned',
}
FIT_PITCH = {'hybrid': 'pitch', 'sines': 'transpose', 'modal': 'size'}   # fit kind -> knob (semitones; size ratio)
FIT_LENGTH = {'spectral': 'length', 'sines': 'stretch'}
FIT_PITCH_ST = 12.0       # a fit moves at most an octave from its own note


class KnobError(ValueError):
    """a knob this sound cannot take, or a target outside its range"""


# ------------------------------------------------------------------------------------------------------- notes
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_NAMES = ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')


def midi(pitch):
    """a note name ('C5', 'Eb6', 'F#4', 'Bb-1'; 's' is a sharp too, as in take file names: 'Cs5' = 'C#5') as a MIDI
    number, or None when it is no note name"""
    m = re.fullmatch(r'([A-Ga-g])([#sb]{0,2})(-?\d+)', str(pitch).strip())
    if not m:
        return None
    acc = m.group(2)
    return 12 * (int(m.group(3)) + 1) + _PC[m.group(1).upper()] + acc.count('#') + acc.count('s') - acc.count('b')


def hz(pitch):
    """a pitch as Hz: a note name (A4 = 440), a number, or '440hz'"""
    if isinstance(pitch, bool):
        raise KnobError(f'pitch must be a note name or Hz, got {pitch!r}')
    if isinstance(pitch, (int, float, np.integer, np.floating)):
        f = float(pitch)
    else:
        n = midi(pitch)
        if n is not None:
            f = 440.0 * 2 ** ((n - 69) / 12)
        else:
            m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:hz)?', str(pitch).strip(), re.I)
            if not m:
                raise KnobError(f"pitch must be a note name ('C5', 'Eb6', 'F#4', 'Fs4') or Hz (440, '440hz'), got "
                                f"{pitch!r}")
            f = float(m.group(1))
    if not 8.0 <= f <= 20000.0:
        raise KnobError(f'pitch {pitch!r} ({f:g} Hz) is outside 8 Hz - 20 kHz')
    return f


def note(f):
    """the nearest note name of a frequency, with cents when off by 5 or more ('E5', 'A4 +12c')"""
    n = 69 + 12 * np.log2(float(f) / 440.0)
    k = int(round(n))
    c = int(round(100 * (n - k)))
    return f'{_NAMES[k % 12]}{k // 12 - 1}' + (f' {c:+d}c' if abs(c) >= 5 else '')


def label(pitch):
    """one canonical file-name label per pitch: the note name ('C#5' -> 'Cs5', 'Db5' -> 'Cs5', 523.25 -> 'C5') when
    the pitch is within 5 cents of a note, else Hz ('445hz'). Raises KnobError for no pitch"""
    f = hz(pitch)
    n = 69 + 12 * np.log2(f / 440.0)
    k = int(round(n))
    if abs(n - k) < 0.05:
        return f'{_NAMES[k % 12]}{k // 12 - 1}'.replace('#', 's')
    return f'{round(f, 2):g}hz'


# ---------------------------------------------------------------------------------------------------- describe
def describe(entry):
    """what the knobs can do on one bank entry: {'dur': 'shape' | 'param' | 'trim', 'apex': bool, 'pitch': {param,
    native_hz, note, range_hz} | {param, mode: 'pc'} | None, 'why': the reason when pitch is None}"""
    out = {'dur': _dur_method(entry)[0], 'apex': _cat(entry) in WHOOSH, 'pitch': None}
    try:
        out['pitch'] = _pitch_spec(entry)
    except KnobError as e:
        out['why'] = str(e).split(': ', 1)[-1]
    return out


def _cat(entry):
    return entry.get('category')


def _dur_method(entry):
    if entry.get('type') == 'fit':
        k = FIT_LENGTH.get(_fit_kind(entry))
        return ('param', k) if k else ('trim', None)
    cat = _cat(entry)
    if cat in WHOOSH:
        return 'shape', None
    key = LENGTH.get(cat)
    return ('param', key) if key and key in entry.get('params', {}) else ('trim', None)


def _fit_kind(entry):
    from .fit import auto
    return auto.kind_of(entry['model'])


def _pitch_spec(entry):
    sid = entry.get('id') or _cat(entry)
    if entry.get('type') == 'fit':
        k = _fit_kind(entry)
        if not entry.get('pitch_hz'):
            raise KnobError(f'{sid}: this fit has no measured note (not a tonal recording)')
        if k not in FIT_PITCH:
            raise KnobError(f'{sid}: a {k} fit has no pitch knob')
        f = float(entry['pitch_hz'])
        return {'param': FIT_PITCH[k], 'native_hz': round(f, 2), 'note': note(f),
                'range_hz': [round(f / 2 ** (FIT_PITCH_ST / 12), 1), round(f * 2 ** (FIT_PITCH_ST / 12), 1)]}
    cat = _cat(entry)
    if cat in UNTUNED:
        raise KnobError(f'{sid}: {UNTUNED[cat]}')
    s = PITCH.get(cat)
    if s is None or s['param'] not in entry['params']:
        raise KnobError(f'{sid}: its recipe has no pitch parameter')
    p = entry['params']
    if s.get('mode') == 'pc':
        return {'param': s['param'], 'mode': 'pc', 'key': int(p[s['param']]), 'note': _NAMES[int(p[s['param']]) % 12]}
    if s.get('partial'):
        return _partial_spec(entry, s, sid)
    need = s.get('needs')
    if need and float(p.get(need[0], -1e9)) < need[1]:
        raise KnobError(f'{sid}: its tonal layer is off ({need[0]} {float(p.get(need[0], 0)):g}), it has no note to '
                        'tune')
    f = float(p[s['param']])
    return {'param': s['param'], 'native_hz': round(f, 2), 'note': note(f), 'range_hz': list(s['range'])}


def _partial_spec(entry, s, sid):
    """a click / toggle: the heard partial is the note (module doc)"""
    p0 = float(entry['params'][s['param']])
    meta = entry.get('meta') or {}
    lo, hi = s['range']
    out = {'param': s['param'], 'mode': 'partial', 'param_hz': round(p0, 2)}
    if 'pitch_hz' not in meta:     # an entry from a bank file built before 0.3.0: measured on the first render
        return out | {'native_hz': None, 'note': None, 'range_hz': [lo, hi], 'heard': 'measured when rendered'}
    fh = meta['pitch_hz']
    if not fh:
        raise KnobError(f'{sid}: no clear tonal partial near its tink ({p0:.0f} Hz): it has no note to tune')
    fh = float(fh)
    return out | {'native_hz': round(fh, 2), 'note': note(fh),
                  'range_hz': [round(lo * fh / p0, 1), round(hi * fh / p0, 1)]}


def partial_hz(entry, x):
    """the heard partial of a click / toggle render: the strongest clear peak within PARTIAL_SPAN_ST of its pitch
    parameter (Hz), or None"""
    from .bank import estimate_pitch
    return estimate_pitch(x, float(entry['params']['pitch']), span_st=PARTIAL_SPAN_ST)


def pitch_warning(edits):
    """the audit warning of a take whose pitch knob missed its note ({'code': 'pitch_off', 'msg'}), else None"""
    p = (edits or {}).get('pitch') or {}
    if p.get('ok') is not False:
        return None
    got = p.get('measured_hz')
    heard = (f"measured {note(got)} ({got:.1f} Hz, {p.get('cents_off', 0):+d} cents)" if got else
             'no clear partial near it in the render')
    return {'code': 'pitch_off', 'msg': f"pitch {p.get('note')}: {heard}; the knob did not land the note "
                                        f'(tolerance {PITCH_TOL_CENTS} cents)'}


# ------------------------------------------------------------------------------------------------------- apply
def _number(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(float(v)):
        raise KnobError(f'{name} must be a number, got {v!r}')
    return float(v)


def validate(entry, dur=None, apex=None, pitch=None):
    """the knobs an entry can take, checked without rendering (KnobError says why): what apply() refuses, so a kit
    check refuses the same (earshot.forge.kit.problems)"""
    sid = entry.get('id') or _cat(entry)
    if dur is not None:
        dur = _number(dur, 'dur')
        if not 0.01 <= dur <= 60:
            raise KnobError(f'dur must be 0.01-60 s, got {dur:g}')
        if (_cat(entry) in WHOOSH) and dur > WHOOSH_MAX_S:
            raise KnobError(f'{sid}: the pass engine renders at most {WHOOSH_MAX_S} s (dur {dur:g})')
    if apex is not None:
        apex = _number(apex, 'apex')
        if _cat(entry) not in WHOOSH:
            raise KnobError(f'{sid}: apex applies to the whoosh family ({", ".join(WHOOSH)}); a riser stops on its '
                            'loudest sample, so aim it with dur and align its end')
        if not 0.01 <= apex <= WHOOSH_MAX_S - 0.1 or (dur is not None and apex >= dur - 0.02):
            raise KnobError(f'apex {apex:g} s must lie inside the sound (0.01 s .. dur - 0.02 s)')
    if pitch is not None:
        _pitch(entry, pitch)


def apply(entry, run, dur=None, apex=None, pitch=None):
    """(raw render, edits) of an entry with its knobs set: run(params_edit, fit_knobs) renders (bank.renderer)"""
    from . import bank
    validate(entry, dur, apex, pitch)
    edits, pe, fk = {}, {}, {}
    if pitch is not None:
        p_upd, f_upd, info = _pitch(entry, pitch)
        pe.update(p_upd)
        fk.update(f_upd)
        edits['pitch'] = info
    meta = entry.get('meta') or {}
    if pitch is not None and edits['pitch'].get('mode') == 'partial':
        pe.update(_land_partial(entry, run, pe, fk, edits['pitch']))
    if apex is not None or (dur is not None and _cat(entry) in WHOOSH):
        x, info = _whoosh(entry, run, pe, dur, apex, meta)
        edits.update(info)
    elif dur is not None:
        x, edits['dur'] = _param_dur(entry, run, pe, fk, float(dur), meta)
    else:
        x = run(pe, fk)
    if dur is not None:
        x, how = fit_length(x, float(dur), keep_s=None if apex is None else float(apex) + APEX_KEEP)
        edits['dur']['fit'] = how
        if apex is not None:   # the fit to dur comes after the apex was placed: measure the file that ships
            got = float(_peak_s(x))
            edits['apex'].update(measured=round(got, 4), err_ms=round(1000 * (got - float(apex)), 1),
                                 ok=bool(abs(got - float(apex)) <= APEX_TOL))
    if pitch is not None and edits['pitch'].get('target_hz'):
        where = (PITCH.get(_cat(entry)) or {}).get('measure', 'start') if entry.get('type') != 'fit' else 'start'
        got = bank.estimate_pitch(_tail(x) if where == 'end' else x, edits['pitch']['target_hz']) if where else None
        edits['pitch']['measured_hz'] = None if got is None else round(got, 2)
        if got:
            edits['pitch']['cents_off'] = int(round(1200 * np.log2(got / edits['pitch']['target_hz'])))
        ok = bool(got) and abs(edits['pitch'].get('cents_off', 0)) <= PITCH_TOL_CENTS
        edits['pitch']['ok'] = ok if where else None   # None: a sound whose pitch is not measured (riser, impact)
    return x, edits


def _land_partial(entry, run, pe, fk, info):
    """a click / toggle aimed at a heard note: the parameter scaled by target / heard (heard: meta pitch_hz, else
    measured on a first render), then rendered, measured and corrected along the ratio at most MAX_FIX times; the
    closest render's parameter is kept (its pe update). A render with no clear partial stops the search"""
    k = info['param']
    f = float(info['target_hz'])
    lo, hi = PITCH[_cat(entry)]['range']
    p0 = float(entry['params'][k])
    v = (info.get('params') or {}).get(k)
    if v is None:                                   # heard partial unknown: measure the member as it is
        fh = partial_hz(entry, run(pe, fk))
        if fh is None:
            raise KnobError(f"{entry.get('id')}: no clear tonal partial near its tink ({p0:.0f} Hz): it has no "
                            'note to tune')
        info['native_hz'] = round(fh, 2)
        v = p0 * f / fh
    best, n = None, 0
    for _ in range(MAX_FIX + 1):
        v = float(np.clip(v, lo, hi))
        x = run({**pe, k: v}, fk)
        n += 1
        from .bank import estimate_pitch
        got = estimate_pitch(x, f)
        if got is None:
            best = best or (v, None)
            break
        c = 1200 * np.log2(got / f)
        if best is None or best[1] is None or abs(c) < abs(best[1]):
            best = (v, c)
        if abs(c) <= PITCH_TOL_CENTS / 2:
            break
        v *= f / got
    info['params'] = {k: round(best[0], 3)}
    info['renders'] = n
    return {k: best[0]}


def _tail(x):
    """the second half of a render (a glide's landing), as a sound of its own"""
    return x[len(x) // 2:]


def _pitch(entry, pitch):
    s = _pitch_spec(entry)
    sid = entry.get('id') or _cat(entry)
    if s.get('mode') == 'pc':
        n = midi(pitch)
        if n is None:
            n = int(round(69 + 12 * np.log2(hz(pitch) / 440.0)))
        return {s['param']: n % 12}, {}, {'param': s['param'], 'key': n % 12, 'note': _NAMES[n % 12],
                                          'native_key': s['key'], 'mode': 'pitch class only (the register stays)'}
    f = hz(pitch)
    lo, hi = s['range_hz']
    if s.get('mode') == 'partial' and s['native_hz'] is None:   # heard partial unknown until a render: range check
        if not lo <= f <= hi:                                   # on the parameter, landed in apply()
            raise KnobError(f'{sid}: {note(f)} ({f:.1f} Hz) is outside its {s["param"]} range {lo:g}-{hi:g} Hz')
        return {}, {}, {'target_hz': round(f, 3), 'note': note(f), 'native_hz': None, 'param': s['param'],
                        'mode': 'partial'}
    if not lo <= f <= hi:
        raise KnobError(f'{sid}: {note(f)} ({f:.1f} Hz) is outside its {s["param"]} range {lo:g}-{hi:g} Hz '
                        f'({note(lo)} - {note(hi)}); its own note is {s["note"]} ({s["native_hz"]:g} Hz)')
    st = 12 * np.log2(f / s['native_hz'])
    info = {'target_hz': round(f, 3), 'note': note(f), 'native_hz': s['native_hz'], 'semitones': round(float(st), 2),
            'param': s['param']}
    if entry.get('type') == 'fit':
        k = s['param']
        v = float(2 ** (-st / 12)) if k == 'size' else float(st)
        info['knobs'] = {k: round(v, 5)}
        return {}, {k: v}, info
    p = entry['params']
    ratio = f / s['native_hz']
    if s.get('mode') == 'partial':     # the parameter moves by the heard ratio; apply() measures and corrects
        info['mode'] = 'partial'
        upd = {s['param']: float(p[s['param']]) * ratio}
        info['params'] = {k: round(v, 3) for k, v in upd.items()}
        return upd, {}, info
    upd = {s['param']: f}
    for k in (PITCH[_cat(entry)].get('co') or ()):
        if k in p and isinstance(p[k], (int, float)) and not isinstance(p[k], bool):
            upd[k] = float(p[k]) * ratio
    info['params'] = {k: round(v, 3) for k, v in upd.items()}
    return upd, {}, info


def _len_s(x):
    return len(x) / SR


def _peak_s(x):
    from .bank import times
    return times(x)['peak_s']


def _secant(make, target, s0, L0, measure=_len_s, tol=None, lo=SCALE[0], hi=SCALE[1]):
    """find s with measure(make(s)) ~ target, starting from the known point (s0, L0) and the ratio guess, at most
    MAX_FIX corrections; returns the closest render (x, s, measured, renders). The closest, not the last: a noise
    render's length (the edit at its end_db) is not monotonic in the parameter, so a correction can land further off"""
    tol = max(DUR_ABS, DUR_REL * target) if tol is None else tol
    s = float(np.clip(s0 * target / max(L0, 1e-6), lo, hi))
    x = make(s)
    L = measure(x)
    best, n = (x, s, L), 1
    sp, Lp = s0, L0
    for _ in range(MAX_FIX):
        if abs(L - target) <= tol or abs(L - Lp) < 1e-4 or s == sp:
            break
        s_new = float(np.clip(s + (target - L) * (s - sp) / (L - Lp), lo, hi))
        sp, Lp = s, L
        s = s_new
        x = make(s)
        L = measure(x)
        n += 1
        if abs(L - target) < abs(best[2] - target):
            best = (x, s, L)
    return (*best, n)


def _param_dur(entry, run, pe, fk, dur, meta):
    method, key = _dur_method(entry)
    if method == 'trim':
        return run(pe, fk), {'target': dur, 'method': 'trim', 'renders': 1}
    L0 = float(meta.get('dur_s') or _len_s(run(pe, fk)))
    if entry.get('type') == 'fit':
        x, s, L, n = _secant(lambda s: run(pe, {**fk, key: s}), dur, 1.0, L0)
        return x, {'target': dur, 'method': 'param', 'knob': key, 'value': round(s, 4), 'renders': n}
    v0 = float(entry['params'][key])
    x, s, L, n = _secant(lambda s: run({**pe, key: v0 * s}, fk), dur, 1.0, L0)
    return x, {'target': dur, 'method': 'param', 'param': key, 'value': round(v0 * s, 5), 'renders': n}


def _whoosh(entry, run, pe, dur, apex, meta):
    """the pass engine aimed at dur and / or apex (module doc)"""
    p = entry['params']
    r0, h0, f0 = float(p['rise']), float(p.get('hold', 0.0)), float(p['fall'])
    if dur is not None and float(dur) > WHOOSH_MAX_S:
        raise KnobError(f"{entry.get('id')}: the pass engine renders at most {WHOOSH_MAX_S} s (dur {dur})")
    L0 = float(meta.get('dur_s') or 0) or None
    pk0 = float(meta.get('peak_s') or 0) or None
    if L0 is None or pk0 is None:
        x0 = run(pe)
        L0, pk0 = _len_s(x0), _peak_s(x0)
    info = {}
    if apex is None:                                    # dur alone: the whole pass scales
        x, s, L, n = _secant(lambda s: run({**pe, 'rise': r0 * s, 'hold': h0 * s, 'fall': f0 * s}), float(dur), 1.0,
                             L0)
        info['dur'] = {'target': float(dur), 'method': 'shape', 'scale': round(s, 4), 'renders': n}
        return x, info
    apex = float(apex)
    k = (max(-float(p['start_db']), 1.0) / 30.0) ** (1.0 / float(p['rise_k']))
    rise = r1 = apex / k                               # the closed form
    post_t = None if dur is None else float(dur) - apex
    q = 1.0 if post_t is None else float(np.clip(post_t / max(L0 - pk0, 1e-6), *SCALE))
    best, n = None, 0
    while True:
        x = run({**pe, 'rise': rise, 'hold': h0 * q, 'fall': f0 * q})
        n += 1
        pk, L = _peak_s(x), _len_s(x)
        err = abs(pk - apex) + (0.0 if post_t is None else abs((L - pk) - post_t))
        if best is None or err < best[4]:
            best = (x, pk, rise, q, err)
        ok_d = post_t is None or abs(L - float(dur)) <= max(DUR_ABS, DUR_REL * float(dur))
        if (abs(pk - apex) <= APEX_TOL and ok_d) or n > MAX_FIX:
            break
        rise = float(np.clip(rise * apex / max(pk, 1e-3), 0.25 * r1, 4.0 * r1))
        if post_t is not None:
            q = float(np.clip(q * post_t / max(L - pk, 1e-3), *SCALE))
    x, pk, rise, q, _ = best
    x, head = place_apex(x, pk, apex)
    got = float(_peak_s(x))
    info['apex'] = {'target': apex, 'measured': round(got, 4), 'err_ms': round(1000 * (got - apex), 1),
                    'search_err_ms': round(float(1000 * (pk - apex)), 1), 'rise': round(rise, 5), 'renders': n,
                    'fit': head, 'ok': bool(abs(got - apex) <= APEX_TOL)}
    if dur is not None:
        info['dur'] = {'target': float(dur), 'method': 'shape', 'post_scale': round(q, 4), 'renders': n}
    return x, info


HEAD_MAX_DB = -20.0       # place_apex trims only a head that stays this far under the peak (the quiet approach)


def place_apex(x, pk, apex):
    """move the loudest moment from pk to apex exactly: trim the quiet approach (a 3 ms fade-in after the cut) when
    the peak is late, pad silence before the start when it is early. The parameter search gets close (a fluttering
    pass's 5 ms peak moves non-monotonically with rise: +-50 ms for a 10 % change on the big families); this lands it.
    Returns (x, 'exact' | 'head trim N ms' | 'head pad N ms' | 'head kept: ...'). The shift is a whole number of the
    peak measure's 1 ms hops (bank.times): a shift off that grid re-reads a fluttering pass's near-equal maxima (0.2 dB
    apart, 15 ms apart on whoosh_big/swell) and can move the measured peak by more than the shift"""
    hop = SR // 1000
    n = int(round((pk - apex) * SR / hop)) * hop
    if n == 0:
        return x, 'exact'
    if n < 0:
        return np.pad(x, [(-n, 0)] + [(0, 0)] * (x.ndim - 1)), f'head pad {-1000 * n / SR:.1f} ms'
    m = np.abs(x if x.ndim == 1 else x.mean(1))
    lvl = 20 * np.log10(m[:n].max() / (m.max() + 1e-12) + 1e-12)
    if lvl > HEAD_MAX_DB:
        return x, f'head kept: trimming {1000 * n / SR:.0f} ms would cut into the pass ({lvl:.0f} dB re peak)'
    y = np.array(x[n:], np.float64)
    k = min(len(y), int(0.003 * SR))
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    y[:k] *= w if y.ndim == 1 else w[:, None]
    return y, f'head trim {1000 * n / SR:.1f} ms'


FADE_MIN_S = 0.002        # fit_length's shortest fade-out (when keep_s leaves less room than the usual quarter)


def fit_length(x, dur, keep_s=None):
    """x cut or padded to exactly dur seconds: (x, 'exact' | 'trim N ms' | 'pad N ms'). A cut fades out over a quarter
    of the length (10-150 ms); keep_s (s) keeps the fade from starting before that point (an apex just before the end:
    the fade then shortens, to at least FADE_MIN_S)"""
    n = int(round(dur * SR))
    if len(x) == n:
        return x, 'exact'
    if len(x) < n:
        pad = [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1)
        return np.pad(x, pad), f'pad {1000 * (n - len(x)) / SR:.0f} ms'
    cut = len(x) - n
    y = np.array(x[:n], np.float64)
    k = min(n, max(int(0.01 * SR), min(int(0.15 * SR), int(0.25 * n))))
    if keep_s is not None:
        k = min(n, max(int(FADE_MIN_S * SR), min(k, n - int(round(float(keep_s) * SR)))))
    w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, k))
    y[n - k:] *= w if y.ndim == 1 else w[:, None]
    return y, f'trim {1000 * cut / SR:.0f} ms' + (f' (fade {1000 * k / SR:.0f} ms)' if keep_s is not None else '')
