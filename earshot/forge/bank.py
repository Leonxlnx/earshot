"""The shipped sound bank: every delivered forge v2 final, tuned plucks and fitted recipes, rendered by id.

    from earshot.forge import bank
    bank.VERSION                                  # 'forge_v2r6+fit1+pluck1' (one tag per bank file)
    bank.list(kind='whoosh')                      # [{id, category, member, type, dur_s, peak_s, onset_s, ...}, ...]
    y, meta = bank.render('whoosh_fast/crisp')    # the delivered final (its own seed), peak -1 dBFS, float32
    y, meta = bank.render('whoosh_fast/crisp', seed=7, dur=0.8, apex=0.32)   # a new take aimed at the picture
    y, meta = bank.render('pluck/kalimba', pitch='E5')                       # tuned through the recipe's own f0

    python -m earshot.forge.bank build out/forge_v2/*/final -o earshot/forge/bank/forge_v2.json
    python -m earshot.forge.bank build --pluck -o earshot/forge/bank/pluck.json
    python -m earshot.forge.bank add-fit model.json --id fit/hit/door --source source.json
    python -m earshot.forge.bank check [IDS ...]   # re-render entries against their stored fingerprints

Sound ids are `<category>/<member>` (`pro_impact/short`, `typing_key/crisp`), `pluck/<name>` for the tuned kalimba
voices and `fit/<role>/<name>` for recipes fitted from public-domain recordings. Categories are the public names: a
user never needs the recipe module behind one (typing_key renders with recipes/keypress.py, pro_pop with
pop.render_v2 ...). Rendering needs numpy / scipy only: no models, no network, no files outside the package.

Why a bank file and not the seed banks: the forge's seed banks (earshot/forge/seeds) keep good parameter sets for
searching, rounded and without their render function; re-rendered from them, a third of the delivered finals come out
as different sounds. A bank entry is the final's schema-2 render block (earshot.forge.schema) at full precision, and
`build` refuses any final whose block does not re-render its delivered WAV (within one 24-bit step).

Bank file (earshot/forge/bank/*.json, package data; extra files: EARSHOT_BANK, paths joined by os.pathsep: ';' on
Windows, ':' elsewhere; a missing or unreadable one is a BankError naming it):
    {"bank": 1, "name": "forge_v2", "version": "forge_v2r6", "earshot": "0.3.0", "note": "...",
     "categories": {"click": {"recipe": "click", "target": "...", "kinds": ["click"], "use": "..."}, ...},
     "entries": {"click/soft": {<entry>}, ...}}
Entry types:
    forge  category, member, recipe (module of earshot.forge.recipes), fn, params (complete), seed, peak_db,
           gain_db (+ gain): the render block; the delivered WAV = normalise(fn(params, rng(seed)), peak_db) * gain
    fit    category 'fit', member, model (an `earshot fit` model, its fit_engine report dropped), seed, vary, source
           {id, title, author, license, url}, pitch_hz (the measured note of a tonal fit, else null): rendered by
           earshot.forge.fit.render(model, seed, vary=vary, take=True) (a take: it rings out to 60 dB down instead
           of stopping where the trimmed recording stopped, no DC offset)
Every entry carries `meta`, measured on its render at peak -1 dBFS: dur_s, onset_s, peak_s, end_s, attack_ms,
centroid_hz, lufs_m_max, channels and a short fingerprint `fp` (50 ms RMS frames in dB re the loudest frame, to the
point where the sound stays 60 dB down) so a later re-render (another machine, numpy, scipy) can be checked against
it: length exact, peak time within 5 ms, envelope within 0.5 dB (`check`, tests/test_bank.py).

render() returns the take at `peak_db` (sample peak) and meta {id, seed, sr, dur_s, onset_s, peak_s, end_s, peak_db,
lufs_m_max, edits, checks}: onset = the 1 ms envelope's first point within 30 dB of its maximum, peak = the loudest
5 ms (1 ms hop), end = the last 5 ms within 50 dB of the loudest; `edits` = what the knobs did (earshot.forge.knobs:
dur, apex, pitch, weight), `checks` = the model-free audit (earshot.forge.sfx.check: clicks, truncation, clipping, DC,
ring, tail, rising pitch glide, stacked layers). A take with seed=None is the delivered final itself; other seeds are
new takes of the same member (the noise, jitter and micro-timing change; the character stays).
"""
import copy
import difflib
import glob
import json
import os
import re
import sys

import numpy as np

SR = 48000
BANK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bank')
FORMAT = 1
FORGE_VERSION = 'forge_v2r6'      # the delivery the shipped forge_v2.json was built from (forge v2, round 6)
PEAK_DB = -1.0
SRS = (44100, 48000, 88200, 96000)   # the sample rates a take is written at (render, sfx render, kit sr)
ENV_DB = -30.0                    # onset: the 1 ms envelope's first point within this of its maximum
END_DB = -50.0                    # end: the last 5 ms frame within this of the loudest
FP_HOP = 0.05                     # fingerprint frame (s)
FP_FLOOR = -60.0                  # the fingerprint stops where the sound stays this far down

# category -> what it is for in a film: the cue kinds it serves (cinetic's kinds and kit roles; a few categories
# serve none of them yet) and one line of use. The recipe module behind a category is in the bank file.
CATEGORIES = {
    'click': (['click'], 'cursor presses and buttons: a short modal tick with a small body'),
    'pro_click': (['click', 'tick.crisp'], 'bright designed UI ticks'),
    'hover_tick': (['tick'], 'hover, list rows passing, tiny ticks'),
    'pro_hover_tick': (['tick', 'tick.crisp'], 'airy digital ticks: cursor blinks, counters'),
    'tap': (['tock', 'land.light', 'click', 'tick'], 'screen taps and light touches on a surface'),
    'pro_tap': (['land.light', 'tock'], 'the soft low thump of a touch'),
    'toggle': (['snap', 'click'], 'switches, settings toggles, lock-ins (press + latch)'),
    'pop': (['pop'], 'bubble and cork pops (the bubble rises in pitch)'),
    'pro_pop': (['pop'], 'tonal pops for a word or badge appearing'),
    'pro_pop_in': (['pop', 'land', 'land.light', 'snap'], 'an icon or element popping in: a round low thump'),
    'notify': (['bell', 'hit', 'text', 'click.confirm'], 'struck bell chimes (A4-E6): success, a message, a lockup'),
    'pro_notify': (['bell', 'click.confirm'], 'short clean dings a semitone apart'),
    'shimmer': (['shimmer'], 'chimes ringing out (end-card sparkle)'),
    'pro_shimmer': (['shimmer'], 'granular sparkle'),
    'whoosh_fast': (['whoosh.short', 'suck'], 'whip pans, fast fly-ins, element moves'),
    'whoosh_big': (['whoosh.long', 'swell'], 'big scene transitions, large objects passing'),
    'pro_whoosh_big': (['whoosh.long', 'swell'], 'a designed low sub swell into a cut'),
    'swish': (['whoosh.short'], 'thin quick flicks, text slides, small UI moves'),
    'swipe': (['whoosh.short'], 'card and panel swipes, page turns (cloth, paper)'),
    'riser': (['riser'], 'a build into a reveal; it stops on its loudest sample'),
    'pro_riser': (['riser'], 'a short stutter riser into a cut'),
    'downlifter': (['drop'], 'energy falling away after a hit, scene exits'),
    'pro_zip': (['zip'], 'tonal sweeps across a cut (up / arc members rise in pitch)'),
    'impact': (['hit', 'land'], 'a cinematic hit: reveal, title slam (long tails: trim with dur or max_s)'),
    'pro_impact': (['hit', 'land'], 'a designed gated sub impact'),
    'sub_drop': (['drop'], 'weight under a hit, a music drop'),
    'pro_sub_drop': (['drop'], 'the designed low drop that blooms after the cut'),
    'logo_sting': (['sting'], 'a composite logo reveal (lead-in + hit + chime)'),
    'pro_logo_sting': (['sting'], 'designed launch gestures onto a tone'),
    'glitch': (['glitch'], 'glitch transitions, error states'),
    'pro_glitch': (['glitch'], 'a designed stutter ending on the cut'),
    'typing_key': (['key', 'key.space', 'key.return'], 'realistic mechanical keys, one take per character'),
    'pro_typing_key': (['key'], 'high-passed designed ticks for kinetic type'),
    'pro_typing_run': (['key.run'], 'a whole typed run as one file'),
    'pluck': (['bell', 'text', 'hit'], 'a kalimba tine: tuned melodic notes (bells, motifs, a tonal lockup)'),
    'fit': ([], 'a recipe fitted from a public-domain recording'),
}

# the tuned voices: pluck/<name> = edits of the launch film's forge pick (recipes/pluck.py DEFAULTS = that pick),
# rendered at C5 with the film's seeds (8001 ...); every take re-tunes through f0 (knobs.py)
PLUCK_F0 = 523.2511306011972          # C5
PLUCKS = {
    'kalimba': ({}, 8001, 'the launch film kalimba: a bright steel tine over a small wooden box'),
    'kalimba_soft': ({'thumb_ms': 1.4, 'skin_lvl': 0.4, 'knock_lvl': 0.5, 'zing_lvl': 0.5, 'p2_lvl': 0.75}, 8009,
                     'a fleshy-thumb pluck: darker and softer at the attack (the film\'s soft notes)'),
    'kalimba_ring': ({'tau1': 1.5}, 8006, 'a longer ring (the film\'s chord tones): end cards, a tonal lockup'),
}
# PLUCKS edits are factors on the base value (1.4 for thumb_ms is a value: see _pluck_params)
_PLUCK_VALUES = {'thumb_ms'}


class BankError(ValueError):
    """an unknown id, a refused build or knob, a bad bank file"""


# ------------------------------------------------------------------------------------------------------- loading
_STATE = {}


def _files():
    """the bank files: the shipped ones (forge_v2 first, then by name), then EARSHOT_BANK's"""
    own = sorted(glob.glob(os.path.join(BANK_DIR, '*.json')),
                 key=lambda p: (not os.path.basename(p).startswith('forge'), os.path.basename(p)))
    extra = [p for p in (os.environ.get('EARSHOT_BANK') or '').split(os.pathsep) if p.strip()]
    return own + [os.path.abspath(p) for p in extra]


def _read(path):
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
    except FileNotFoundError:
        raise BankError(f'EARSHOT_BANK file {path} not found (paths are joined by {os.pathsep!r})') from None
    except (OSError, ValueError) as e:
        raise BankError(f'{path}: not readable as a bank file ({type(e).__name__}: {e})') from None
    if not isinstance(d, dict) or d.get('bank') != FORMAT or not isinstance(d.get('entries'), dict):
        raise BankError(f'{path}: not an earshot bank file (bank: {FORMAT} with entries)')
    return d


def _load():
    key = tuple(_files())
    if _STATE.get('key') == key:
        return _STATE
    entries, cats, versions, where = {}, {}, [], {}
    for p in key:
        d = _read(p)
        versions.append(str(d.get('version') or os.path.splitext(os.path.basename(p))[0]))
        for c, info in (d.get('categories') or {}).items():
            cats.setdefault(c, dict(info))
        for i, e in d['entries'].items():
            entries[i] = e
            where[i] = p
    _STATE.clear()
    _STATE.update(key=key, entries=entries, categories=cats, version='+'.join(versions), where=where)
    return _STATE


def reload():
    """forget the loaded bank files (after a build / add-fit in this process)"""
    _STATE.clear()


def __getattr__(name):
    if name == 'VERSION':
        return _load()['version']
    raise AttributeError(name)


def ids():
    return sorted(_load()['entries'])


def get(sound_id):
    """the bank entry (a copy) for an id; BankError with the closest ids when unknown"""
    e = _load()['entries'].get(sound_id)
    if e is None:
        near = difflib.get_close_matches(str(sound_id), ids(), n=4, cutoff=0.5)
        cat = str(sound_id).split('/')[0]
        same = [i for i in ids() if i.split('/')[0] == cat][:8]
        hint = f' Did you mean {", ".join(near)}?' if near else (f' {cat} has: {", ".join(same)}.' if same else '')
        raise BankError(f'unknown sound id {sound_id!r}.{hint} (`earshot sfx list` shows all {len(ids())})')
    return copy.deepcopy(e) | {'id': sound_id}


def category_info(cat):
    kinds, use = CATEGORIES.get(cat, ([], ''))
    info = dict(_load()['categories'].get(cat) or {})
    info.setdefault('kinds', kinds)
    info.setdefault('use', use)
    return info


def list_sounds(category=None, kind=None):
    """the bank as rows {id, category, member, type, dur_s, peak_s, onset_s, end_s, centroid_hz, attack_ms,
    lufs_m_max, kinds, knobs, source (fit entries), notes}, sorted by id. kind matches a cue kind or role:
    'whoosh' finds whoosh.short and whoosh.long, 'whoosh.long' only those."""
    from . import knobs
    out = []
    for i in ids():
        e = get(i)
        cat = e['category']
        if category and cat != category:
            continue
        info = category_info(cat)
        kinds = e.get('kinds') or info.get('kinds') or []
        if kind and not any(k == kind or k.split('.')[0] == kind for k in kinds):
            continue
        m = e.get('meta') or {}
        row = {'id': i, 'category': cat, 'member': e.get('member'), 'type': e.get('type', 'forge'),
               **{k: m.get(k) for k in ('dur_s', 'peak_s', 'onset_s', 'end_s', 'centroid_hz', 'attack_ms',
                                        'lufs_m_max')},
               'kinds': kinds, 'knobs': knobs.describe(e),
               'notes': e.get('notes') or info.get('use') or ''}
        if e.get('source'):
            row['source'] = e['source']
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------------ rendering
def renderer(entry, seed=None):
    """f(params_edit=None, fit_knobs=None) -> the raw render (float64, (n,) or (n, ch)) of an entry with its own
    params updated by params_edit (forge) or with fit knobs (fit); seed None = the entry's own"""
    seed = int(entry['seed'] if seed is None else seed)
    kind = entry.get('type', 'forge')
    if kind == 'forge':
        from . import dsp, schema
        fn = schema.resolve(entry)

        def run(edit=None, fit_knobs=None):
            p = dict(entry['params'])
            if edit:
                p.update(edit)
            x = np.asarray(fn(p, dsp.rng(seed)), np.float64)
            return _valid(x, entry)
        return run
    if kind == 'fit':
        from . import fit as _fit

        def run(edit=None, fit_knobs=None):
            kn = {'vary': float(entry.get('vary') or 0.0), 'take': True, **(fit_knobs or {})}
            if not kn['vary']:
                kn.pop('vary')
            return _valid(np.asarray(_fit.render(entry['model'], seed=seed, **kn), np.float64), entry)
        return run
    raise BankError(f"{entry.get('id')}: unknown entry type {kind!r}")


def _valid(x, entry):
    if not len(x) or not np.isfinite(x).all() or np.abs(x).max() < 1e-7:
        raise BankError(f"{entry.get('id')}: the render failed or is silent")
    return x


def normalize(x, peak_db=PEAK_DB):
    return np.asarray(x, np.float64) * (10 ** (peak_db / 20) / (np.abs(x).max() + 1e-12))


def level(peak_db=PEAK_DB, weight=None):
    """the sample peak (dBFS) a take is written at: peak_db, moved 6 dB per doubling of weight. BankError for a
    level over 0 dBFS (a clipped take), a weight outside (0, 4] or a value that is no number"""
    def num(v, name):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
            raise BankError(f'{name} must be a number, got {v!r}')
        return float(v)
    lv = num(peak_db, 'peak_db')
    if lv > 0:
        raise BankError(f'peak_db {lv:g} is over 0 dBFS: the take would clip (use 0 or less)')
    if weight is not None:
        w = num(weight, 'weight')
        if not 0 < w <= 4:
            raise BankError(f'weight must be in (0, 4], got {weight}')
        lv += 20 * np.log10(w)
        if lv > 0:
            raise BankError(f'weight {w:g} at peak_db {peak_db:g} puts the peak at {lv:+.1f} dBFS: lower peak_db')
    return float(lv)


def check_sr(sr):
    """sr as an int when it is one of SRS (an integral float such as 48000.0 too), else BankError"""
    ok = not isinstance(sr, bool) and isinstance(sr, (int, float, np.integer, np.floating)) and float(sr) in SRS
    if not ok:
        raise BankError(f'sr must be one of {", ".join(map(str, SRS))}, got {sr!r}')
    return int(sr)


def render(sound_id, seed=None, dur=None, apex=None, pitch=None, peak_db=PEAK_DB, sr=SR, weight=None, checks=True):
    """one take of a bank sound: (y float32 (n,) or (n, 2), meta). seed None = the entry's own seed (the delivered
    final); dur / apex (s), pitch ('E5' or Hz) aim it through the recipe's own parameters (earshot.forge.knobs);
    peak_db = its sample peak (0 dBFS or less); weight scales that by 6 dB per doubling (1 = peak_db, 0.5 = 6 dB
    down; a level over 0 dBFS is refused: level()). sr (one of SRS) != 48000 resamples the finished take (soxr).
    checks=False skips the model-free audit (meta['checks'] None)."""
    lv = level(peak_db, weight)
    sr = check_sr(sr)
    e = sound_id if isinstance(sound_id, dict) else get(sound_id)
    sid = e.get('id')
    run = renderer(e, seed)
    from . import knobs
    if any(v is not None for v in (dur, apex, pitch)):
        x, edits = knobs.apply(e, run, dur=dur, apex=apex, pitch=pitch)
    else:
        x, edits = run(), {}
    level_db = lv
    if weight is not None:
        edits['weight'] = {'weight': float(weight), 'peak_db': round(level_db, 2)}
    y = normalize(x, level_db).astype(np.float32)
    meta = {'id': sid, 'seed': int(e['seed'] if seed is None else seed), 'sr': int(sr), **measure(y),
            'edits': edits, 'checks': None}
    if checks:
        from . import sfx
        meta['checks'] = sfx.check(y, category=e.get('category'), sound=sid)
        add_warning(meta['checks'], knobs.pitch_warning(edits))
    if int(sr) != SR:
        import soxr
        y = np.asarray(soxr.resample(y, SR, int(sr)), np.float32)
        meta['dur_s'] = round(len(y) / int(sr), 4)
    return y, meta


def add_warning(c, w):
    """add one warning ({code, msg} or None) to an audit result (sfx.check): ok becomes False, the summary says it"""
    if not w:
        return c
    c['warnings'].append(w)
    c['ok'] = False
    name = c['summary'].split(':', 1)[0]
    c['summary'] = (f"{name}: {len(c['warnings'])} warning{'s' if len(c['warnings']) != 1 else ''}: "
                    + '; '.join(x['msg'] for x in c['warnings']))
    return c


def render_final(sound_id):
    """a forge entry at its delivered level (schema.rerender: the palette gain included): the WAV the forge wrote"""
    from . import schema
    e = get(sound_id) if isinstance(sound_id, str) else sound_id
    if e.get('type', 'forge') != 'forge':
        raise BankError(f"{e.get('id')}: render_final applies to forge entries")
    return schema.rerender(_block(e))


def _block(e):
    return {k: e[k] for k in ('recipe', 'fn', 'member', 'params', 'seed', 'peak_db', 'gain_db', 'gain', 'mode')
            if k in e}


# ------------------------------------------------------------------------------------------------------ measuring
def _mono(y):
    y = np.asarray(y, np.float64)
    return y.mean(1) if y.ndim == 2 else y


def _env(m, win, hop):
    """(RMS of win-sample windows every hop samples, window starts)"""
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    st = np.arange(0, max(1, len(m) - win + 1), hop)
    e = (c[np.minimum(st + win, len(m))] - c[st]) / win
    return np.sqrt(np.maximum(e, 0.0)), st


def times(y, sr=SR):
    """onset_s, peak_s, end_s, dur_s of a take (module doc)"""
    m = _mono(y)
    e1, s1 = _env(m, max(1, sr // 1000), max(1, sr // 4000))
    on = s1[int(np.argmax(e1 >= e1.max() * 10 ** (ENV_DB / 20)))] if len(e1) else 0
    w5, h1 = max(1, sr // 200), max(1, sr // 1000)
    e5, s5 = _env(m, w5, h1)
    ip = int(np.argmax(e5))
    live = np.nonzero(e5 >= e5[ip] * 10 ** (END_DB / 20))[0]
    end = min(len(m), s5[live[-1]] + w5) if len(live) else len(m)
    return {'onset_s': round(on / sr, 4), 'peak_s': round((s5[ip] + w5 / 2) / sr, 4), 'end_s': round(end / sr, 4),
            'dur_s': round(len(m) / sr, 4)}


def fingerprint(y, sr=SR):
    """50 ms RMS frames in dB re the loudest, cut where the sound stays FP_FLOOR down"""
    m = _mono(y)
    n = int(FP_HOP * sr)
    k = max(1, -(-len(m) // n))
    p = np.pad(m, (0, k * n - len(m)))
    e = 10 * np.log10((p.reshape(k, n) ** 2).mean(1) + 1e-24)
    e -= e.max()
    live = np.nonzero(e > FP_FLOOR)[0]
    return [round(float(v), 1) for v in e[:live[-1] + 1]]


def measure(y, sr=SR):
    """times() + peak_db and momentary-max loudness of a finished take"""
    from .polish import lufs_max
    out = times(y, sr)
    out['peak_db'] = round(float(20 * np.log10(np.abs(y).max() + 1e-12)), 2)
    out['lufs_m_max'] = round(float(lufs_max(np.asarray(y, np.float64))), 2) if sr == SR else None
    return out


def meta_of(y, sr=SR):
    """the stored per-entry meta (module doc)"""
    from .polish import attack
    m = _mono(y)
    F = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / sr)
    out = measure(y, sr)
    out.pop('peak_db')
    out['attack_ms'] = attack(np.asarray(y, np.float64))['attack_ms']
    out['centroid_hz'] = round(float((F * f).sum() / (F.sum() + 1e-30)), 1)
    out['channels'] = int(np.asarray(y).shape[1]) if np.asarray(y).ndim == 2 else 1
    out['fp'] = {'hop_s': FP_HOP, 'env_db': fingerprint(y, sr)}
    return out


def compare_fp(meta, y, sr=SR, floor=-40.0):
    """a re-render against an entry's stored meta: {'len_diff', 'peak_diff_ms', 'env_db'} (samples, ms, the largest
    envelope difference over the stored frames above `floor`)"""
    t = times(y, sr)
    fp = fingerprint(y, sr)
    ref = (meta.get('fp') or {}).get('env_db') or []
    k = min(len(ref), len(fp))
    d = [abs(a - b) for a, b in zip(ref[:k], fp[:k]) if a > floor]
    return {'len_diff': int(round((t['dur_s'] - meta['dur_s']) * sr)),
            'peak_diff_ms': round(float(1000 * (t['peak_s'] - meta['peak_s'])), 2),
            'env_db': round(max(d), 2) if d else 0.0}


def estimate_pitch(y, near_hz, span_st=4.0, sr=SR):
    """the strongest spectral peak within span_st semitones of near_hz (Hz), from the onset to 0.6 s after it, or
    None when no clear peak stands there (a noise sound)"""
    m = _mono(y)
    on = int(times(y, sr)['onset_s'] * sr)
    seg = m[on:on + int(0.6 * sr)]
    if len(seg) < 512:
        return None
    n = 1 << int(np.ceil(np.log2(max(len(seg), 1 << 14)) + 2))
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    lo, hi = np.searchsorted(f, [near_hz * 2 ** (-span_st / 12), near_hz * 2 ** (span_st / 12)])
    if hi - lo < 3:
        return None
    j = lo + int(np.argmax(F[lo:hi]))
    wide = F[np.searchsorted(f, near_hz / 2):np.searchsorted(f, near_hz * 2)]
    if F[j] < 10 * np.median(wide):                         # 10 dB over the octave around it, or no pitch
        return None
    a, b, c = np.log(F[j - 1:j + 2] + 1e-30)
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    return float((j + d) * sr / n)


# -------------------------------------------------------------------------------------------------------- building
def _write(path, doc):
    """one entry per line: small diffs, still one JSON document"""
    head = {k: v for k, v in doc.items() if k not in ('categories', 'entries')}
    lines = ['{' + ', '.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in head.items()) + ',',
             ' "categories": {']
    cats = [*(doc.get('categories') or {}).items()]
    lines += [f'  {json.dumps(c)}: {json.dumps(v)}' + (',' if i < len(cats) - 1 else '') for i, (c, v) in
              enumerate(cats)]
    lines += [' },', ' "entries": {']
    ents = [*doc['entries'].items()]
    lines += [f'  {json.dumps(i)}: {json.dumps(e)}' + (',' if j < len(ents) - 1 else '') for j, (i, e) in
              enumerate(ents)]
    lines += [' }', '}']
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    json.loads(open(tmp, encoding='utf-8').read())          # never replace a bank with a broken file
    os.replace(tmp, path)
    reload()
    return path


def _version():
    from .. import __version__
    return __version__


def build(dirs, out, version=FORGE_VERSION, tol=None, log=print):
    """the forge bank file from final folders (out/forge_v2/<category>/final): every pick's own schema-2 render block
    re-rendered and compared with its delivered WAV (within schema.TOL); one pick that does not match, or that has no
    block of its own, and nothing is written (BankError). Returns the document."""
    import soundfile as sf
    from . import schema
    tol = schema.TOL if tol is None else tol
    dirs = sorted({os.path.abspath(d) for p in dirs for d in (glob.glob(p) or [p])
                   if os.path.isfile(os.path.join(d, 'report.json'))})
    if not dirs:
        raise BankError('no final folders with a report.json')
    entries, cats, bad = {}, {}, []
    for d in dirs:
        with open(os.path.join(d, 'report.json'), encoding='utf-8') as fh:
            rep = json.load(fh)
        cat = os.path.basename(os.path.dirname(d)) if os.path.basename(d) == 'final' else os.path.basename(d)
        for pk in rep.get('picks') or []:
            name = f"{cat}/{pk.get('name')}"
            if not isinstance(pk.get('render'), dict):
                bad.append(f'{name}: no render block of its own (run python -m earshot.forge.schema --upgrade)')
                continue
            b = schema.block(rep, pk)
            sid = f"{cat}/{b.get('member') or pk.get('name')}"
            try:
                x = schema.raw(b)
                if x is None:
                    raise schema.NotReproducible('failed or silent')
                y, _ = sf.read(schema._wav(d, pk), always_2d=True)
                err = schema.compare(schema.level(x, b), y)
            except (schema.NotReproducible, OSError, RuntimeError) as e:
                bad.append(f'{sid}: {e}')
                continue
            if err > tol:
                bad.append(f'{sid}: re-render differs from its WAV by {err:.2e} (> {tol:g})')
                continue
            if sid in entries:
                bad.append(f'{sid}: two picks with one member name')
                continue
            e = {'type': 'forge', 'category': cat, 'member': b.get('member') or pk.get('name'),
                 **{k: b[k] for k in ('recipe', 'fn', 'params', 'seed', 'peak_db', 'gain_db', 'gain', 'mode')
                    if k in b}}
            e['meta'] = meta_of(normalize(x).astype(np.float32))
            add_partial(e, x)
            entries[sid] = e
            if log:
                log(f"  {sid:30s} {e['meta']['dur_s']:6.3f} s  err {err:.1e}")
        kinds, use = CATEGORIES.get(cat, ([], ''))
        cats[cat] = {'recipe': (rep.get('picks') or [{}])[0].get('render', {}).get('recipe'),
                     'target': rep.get('target'), 'kinds': kinds, 'use': use}
    if bad:
        raise BankError(f'{len(bad)} pick(s) refused, nothing written:\n  ' + '\n  '.join(bad))
    doc = {'bank': FORMAT, 'name': 'forge_v2', 'version': version, 'earshot': _version(),
           'note': 'the delivered forge v2 finals as render blocks (earshot.forge.schema); every entry re-rendered '
                   'its WAV within one 24-bit step when this file was built',
           'categories': cats, 'entries': entries}
    _write(out, doc)
    return doc


def add_partial(e, x=None):
    """meta['pitch_hz'] of a click / toggle entry: its heard partial (knobs.partial_hz on its own render), None when it
    has no clear one; other entries are left as they are. Returns the entry"""
    from . import knobs
    if (knobs.PITCH.get(e.get('category')) or {}).get('partial') and e.get('type', 'forge') == 'forge':
        x = renderer(e)() if x is None else x
        f = knobs.partial_hz(e, x)
        e.setdefault('meta', {})['pitch_hz'] = None if f is None else round(float(f), 2)
    return e


def _pluck_params(edit):
    from .recipes import pluck
    p = {**pluck.DEFAULTS, 'f0': PLUCK_F0}
    for k, v in edit.items():
        p[k] = float(v) if k in _PLUCK_VALUES else p[k] * float(v)
    return p


def build_pluck(out, version='pluck1', log=print):
    """the tuned-voice bank file (pluck/<name>, PLUCKS)"""
    entries = {}
    for name, (edit, seed, note) in PLUCKS.items():
        e = {'type': 'forge', 'category': 'pluck', 'member': name, 'recipe': 'pluck', 'fn': 'render',
             'params': _pluck_params(edit), 'seed': seed, 'peak_db': PEAK_DB, 'gain_db': 0.0, 'notes': note}
        x = renderer(e)()
        e['meta'] = meta_of(normalize(x).astype(np.float32))
        entries[f'pluck/{name}'] = e
        if log:
            log(f"  pluck/{name:24s} {e['meta']['dur_s']:6.3f} s")
    kinds, use = CATEGORIES['pluck']
    doc = {'bank': FORMAT, 'name': 'pluck', 'version': version, 'earshot': _version(),
           'note': 'tuned kalimba voices from recipes/pluck.py (the launch film pick and its soft / long-ring edits), '
                   'native pitch C5; takes re-tune through f0',
           'categories': {'pluck': {'recipe': 'pluck', 'target': 'a kalimba note, a thumb piano pluck',
                                    'kinds': kinds, 'use': use}},
           'entries': entries}
    _write(out, doc)
    return doc


FIT_ID = re.compile(r'^fit/[a-z0-9_.]+/[a-z0-9_]+$')
SOURCE_KEYS = ('id', 'title', 'author', 'license', 'url')
FREE = ('cc0', 'public domain', 'publicdomain', 'cc-pdm', 'pdm')


def add_fit(model, sound_id, source, bank_path=None, seed=0, vary=0.0, notes=None, kinds=None,
            allow_license=False, force=False, log=print):
    """add a fitted recipe to a bank file (default earshot/forge/bank/fit.json): model = an `earshot fit` model
    (dict or model.json path), sound_id 'fit/<role>/<name>', source = {id, title, author, license, url} (dict or JSON
    path). The licence must be CC0 / public domain unless allow_license (never NC or ND: a fit is a derivative that
    ships inside an MIT package). Returns the entry."""
    from .fit import auto
    if not FIT_ID.match(str(sound_id)):
        raise BankError(f'fit ids look like fit/<role>/<name> (lowercase, digits, _), got {sound_id!r}')
    m = auto.load(model) if isinstance(model, str) else copy.deepcopy(model)
    if isinstance(source, str):
        with open(source, encoding='utf-8') as fh:
            source = json.load(fh)
    miss = [k for k in SOURCE_KEYS if not str((source or {}).get(k) or '').strip()]
    if miss:
        raise BankError(f'source needs {", ".join(SOURCE_KEYS)}; missing {", ".join(miss)}')
    lic = str(source['license']).lower()
    if re.search(r'\b(nc|nd)\b|noncommercial|non-commercial|noderiv', lic):
        raise BankError(f"licence {source['license']!r} forbids commercial use or derivatives: a fit cannot ship")
    if not any(f in lic for f in FREE) and not allow_license:
        raise BankError(f"licence {source['license']!r} is not CC0 / public domain (pass allow_license to keep an "
                        'attribution licence, and credit the source)')
    fe = m.pop('fit_engine', None) or {}
    role = sound_id.split('/')[1]
    e = {'type': 'fit', 'category': 'fit', 'member': sound_id.split('/', 1)[1], 'model': m, 'seed': int(seed),
         'vary': float(vary), 'source': {k: source[k] for k in SOURCE_KEYS},
         'fit': {'kind': m.get('kind'), 'chosen': fe.get('chosen'),
                 'class': (fe.get('character') or {}).get('class')},
         'kinds': kinds or [role], 'notes': notes or f"fitted from {source['title']} ({source['license']})"}
    x = renderer(e)()
    y = normalize(x).astype(np.float32)
    e['meta'] = meta_of(y)
    e['pitch_hz'] = _fit_pitch(y, fe)
    path = bank_path or os.path.join(BANK_DIR, 'fit.json')
    doc = _read(path) if os.path.isfile(path) else {
        'bank': FORMAT, 'name': 'fit', 'version': 'fit1', 'earshot': _version(),
        'note': 'recipes fitted from public-domain recordings (earshot fit); each entry names its source',
        'categories': {'fit': {'recipe': None, 'target': None, 'kinds': [], 'use': CATEGORIES['fit'][1]}},
        'entries': {}}
    if sound_id in doc['entries'] and not force:
        raise BankError(f'{sound_id} is already in {path} (force=True / --force replaces it)')
    doc['entries'][sound_id] = e
    doc['earshot'] = _version()
    _write(path, doc)
    if log:
        log(f"added {sound_id} to {path}: {e['meta']['dur_s']:.3f} s, {e['fit']['kind']}"
            + (f", pitch {e['pitch_hz']:.1f} Hz" if e['pitch_hz'] else '') + f", {source['license']}")
    return e | {'id': sound_id}


def _fit_pitch(y, fe):
    """the note of a tonal fit (its strongest partial over the first 0.6 s), else None"""
    ch = fe.get('character') or {}
    if ch and not ch.get('tonal'):
        return None
    m = _mono(y)
    on = int(times(y)['onset_s'] * SR)
    seg = m[on:on + int(0.6 * SR)]
    if len(seg) < 1024:
        return None
    n = 1 << 17
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f >= 60) & (f <= 5000)
    j = int(np.argmax(np.where(band, F, 0)))
    est = estimate_pitch(y, float(f[j]), span_st=1.0)
    return round(est, 2) if est else None


def check(sound_ids=None, log=print):
    """re-render entries (all by default) and compare with their stored meta: rows {id, len_diff, peak_diff_ms,
    env_db, ok} (ok: length exact, peak within 5 ms, envelope within 0.5 dB)"""
    rows = []
    for i in sound_ids or ids():
        e = get(i)
        y, _ = render(e, checks=False)
        r = {'id': i, **compare_fp(e['meta'], y)}
        r['ok'] = r['len_diff'] == 0 and abs(r['peak_diff_ms']) <= 5 and r['env_db'] <= 0.5
        rows.append(r)
        if log:
            log(f"  {i:32s} len {r['len_diff']:+d}  peak {r['peak_diff_ms']:+.1f} ms  env {r['env_db']:.2f} dB  "
                + ('ok' if r['ok'] else 'DIFFERS'))
    return rows


# ------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m earshot.forge.bank', description='build and check the sound bank')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='write a bank file from forge final folders (or the plucks)')
    b.add_argument('dirs', nargs='*', help='final folders (globs work), e.g. out/forge_v2/*/final')
    b.add_argument('-o', '--out', required=True)
    b.add_argument('--pluck', action='store_true', help='build the tuned pluck voices instead (no folders)')
    b.add_argument('--version', help=f'bank version tag (default {FORGE_VERSION}, or pluck1 with --pluck)')
    f = sub.add_parser('add-fit', help='add a fitted recipe (earshot fit model.json) as fit/<role>/<name>')
    f.add_argument('model')
    f.add_argument('--id', required=True, dest='sound_id')
    f.add_argument('--source', required=True, help='JSON {id, title, author, license, url}')
    f.add_argument('--bank', help='bank file to write (default earshot/forge/bank/fit.json)')
    f.add_argument('--seed', type=int, default=0)
    f.add_argument('--vary', type=float, default=0.0, help='take-to-take jitter of the fit (0-0.6)')
    f.add_argument('--kinds', nargs='*', help='cue kinds / roles it serves (default: the role in the id)')
    f.add_argument('--notes')
    f.add_argument('--allow-license', action='store_true', help='keep a non-CC0 attribution licence (never NC / ND)')
    f.add_argument('--force', action='store_true', help='replace an entry with this id')
    c = sub.add_parser('check', help='re-render entries against their stored fingerprints')
    c.add_argument('ids', nargs='*')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'build':
            if a.pluck:
                doc = build_pluck(a.out, version=a.version or 'pluck1')
            else:
                doc = build(a.dirs, a.out, version=a.version or FORGE_VERSION)
            print(f"wrote {a.out}: {len(doc['entries'])} entries ({doc['version']}), "
                  f"{os.path.getsize(a.out) / 1024:.0f} KB")
        elif a.cmd == 'add-fit':
            add_fit(a.model, a.sound_id, a.source, a.bank, seed=a.seed, vary=a.vary, notes=a.notes, kinds=a.kinds,
                    allow_license=a.allow_license, force=a.force)
        else:
            rows = check(a.ids or None)
            bad = [r['id'] for r in rows if not r['ok']]
            print(f"{len(rows) - len(bad)} / {len(rows)} entries re-render as stored" + (
                f"; differ: {', '.join(bad)}" if bad else ''))
            return 1 if bad else 0
    except BankError as e:
        print(f'bank: {e}', file=sys.stderr)
        return 1
    return 0


list = list_sounds   # noqa: A001  (the public name, bank.list(); kept last: the module uses builtins below it)

if __name__ == '__main__':
    sys.exit(main())
