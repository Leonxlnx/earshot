"""earshot kit: one film's sounds by role, rendered from the bank into takes with measured sync points.

    earshot kit init --preset calm -o audio/kit.json               # calm | crisp | minimal (earshot/forge/kits)
    earshot kit render audio/kit.json --out audio/kit [--force]    # cached; writes audio/kit/kit.lock.json
    earshot kit check audio/kit.json [--out audio/kit] [--json]    # ids, knobs, the model-free audit of every take
    earshot kit audition audio/kit.json -o audition.wav [--out audio/kit]   # every take, role by role, at its level

    from earshot.forge import kit
    lock = kit.render('audio/kit.json', 'audio/kit')             # the lock dict (also written as kit.lock.json)

A kit names a bank sound per role (the shared vocabulary of a film's cue kinds and their variants, ROLES) and how to
render it; a renderer (cinetic's score.py, a mixer) places the takes by the measured onset / peak / end in the lock.

kit.json (schema 1):
    {"kit": 1, "name": "calm", "sr": 48000,
     "roles": {"hit":  {"sound": "notify/m0", "takes": 3, "seed": 1, "edits": {"dur": 1.6}, "align": "onset",
                        "max_s": 2.0, "level_db": -2.0, "note": "why this sound"},
               "bell": {"sound": "pluck/kalimba", "pitches": ["C5", "D5", "E5", "G5", "A5"], "align": "onset"},
               "riser": null},
     "off": {"riser": "why this role is off"}}
  sound     a bank id (`earshot sfx list`), forge, pluck/* or fit/*
  takes     how many takes (default 1, at most 32); take k renders with seed + k - 1 (seed default: the sound's own,
            so take 1 is the delivered final). A take that fails the audit is rendered again on seed + 1000 j (up to
            REROLL times) and the first clean one kept (the lock names it, with planned_seed). Same kit, same takes:
            rendering is deterministic.
  edits     knobs for every take: dur, apex, pitch, peak_db (default -1, at most 0), weight (earshot.forge.knobs,
            bank.render); checked like a render checks them (a whoosh-only apex, a pitch the sound cannot take)
  pitches   one set of takes per pitch (note names, 'Cs5' sharps included, or Hz), through the recipe's own pitch
            parameter; two spellings of one pitch ('C#5', 'Db5', 554.37) are one take, so a list may not repeat one
  align     the sync point a mixer puts on the cue: onset (the take's measured onset), peak (its loudest 5 ms: whoosh
            apexes), end (its end: risers, swells); default ALIGN[role]
  max_s     a take longer than this is faded out to it (a quarter of the length, at most 150 ms); 0.01 s or more
  level_db  the mix level of the role relative to a take at its peak: information for the mixer (audition uses it); a
            number (leave it out for 0)
  note      why this sound (the presets say why for every role; "off" says why a role is null)
  null      the role is not sounded: a renderer skips such cues (the presets keep riser, swell and suck off)

kit.lock.json (written by render into --out with the takes):
    {"kit": 1, "earshot": "0.3.0", "bank": "<bank.VERSION>", "sr": 48000, "name": "calm",
     "roles": {"hit": {"sound": "notify/m0", "align": "onset", "level_db": -2.0, "max_s": 2.0,
                       "takes": [{"file": "hit_1.wav", "seed": 1, "pitch": null, "onset_s": 0.0, "peak_s": 0.004,
                                  "end_s": 1.6, "dur_s": 1.6, "peak_db": -1.0, "lufs_m_max": -11.1, "sha1": "...",
                                  "key": "...", "checks": []}]},
               "riser": null}}
Takes are 24-bit WAVs named <role>_<k>.wav (dots in a role become '-': tick-crisp_1.wav) or <role>_<pitch>[_k].wav
with one canonical label per pitch (knobs.label: C#5 / Db5 / 554.37 -> Cs5, 445 -> 445hz). Names must differ also
without regard to case (Windows and macOS file systems): a kit whose takes would share a file is refused. The
cache: a take whose key (bank entry, seed, pitch, edits, max_s, sr, bank version, earshot version) and file hash
match the existing lock is kept as it is; --force renders everything; files of the old lock
that the new kit no longer names are removed (only those). `checks` lists the warning codes of the take's audit
(earshot.forge.sfx.check with the role as context: a hit warns on stacked attacks and pre-swells; a pitch knob that
missed its note by more than 50 cents adds pitch_off).
"""
import hashlib
import json
import math
import os
import re
import time

import numpy as np

from ..report import emit

SR = 48000
SCHEMA = 1
KITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kits')
LOCK = 'kit.lock.json'
ROLES = ('tick', 'tick.crisp', 'tock', 'hit', 'land', 'land.light', 'pop', 'whoosh.short', 'whoosh.long', 'suck',
         'riser', 'drop', 'key', 'key.space', 'key.return', 'click', 'click.confirm', 'snap', 'swell', 'bell', 'text')
ALIGN = {'whoosh.short': 'peak', 'whoosh.long': 'peak', 'suck': 'end', 'riser': 'end', 'swell': 'end'}
ALIGNS = ('onset', 'peak', 'end')
EDITS = ('dur', 'apex', 'pitch', 'peak_db', 'weight')
ROLE_KEYS = ('sound', 'takes', 'seed', 'edits', 'pitches', 'align', 'max_s', 'level_db', 'note')
MAX_TAKES = 32
REROLL = 3                # extra seeds a take that fails the audit may try (render_take)
RESTRAINED = ('riser', 'swell', 'suck')
# sounds the presets never use, with the reason (kit check notes them; the choice stays the kit author's)
NEVER = {'pro_zip/up': 'rises in pitch (a synth whip)', 'pro_zip/up_hi': 'rises in pitch (a synth whip)',
         'pro_zip/arc': 'rises in pitch (a synth whip)', 'riser/shepard': 'a shepard riser',
         'downlifter/shepard_fall': 'a shepard downlifter', 'downlifter/synth_fall': 'a synth downlifter'}
NEVER_CATEGORIES = {'logo_sting': 'a composite sting (lead-in + hit + chime stacked)',
                    'pro_logo_sting': 'a composite launch gesture', 'shimmer': 'a shimmer wash',
                    'pro_shimmer': 'a granular shimmer', 'glitch': 'a glitch', 'pro_glitch': 'a glitch',
                    'pro_pop': 'tonal pops (the 8-bit risk the forge measured)'}


class KitError(ValueError):
    """an invalid kit file"""


# ----------------------------------------------------------------------------------------------- files and checks
def presets():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(KITS_DIR) if f.endswith('.json'))


def preset(name):
    p = os.path.join(KITS_DIR, f'{name}.json')
    if not os.path.isfile(p):
        raise KitError(f'no preset {name!r} (presets: {", ".join(presets())})')
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def load(kit):
    """a kit dict from a path (or the dict itself), validated: KitError lists every problem"""
    if isinstance(kit, str):
        with open(kit, encoding='utf-8') as fh:
            try:
                kit = json.load(fh)
            except json.JSONDecodeError as e:
                raise KitError(f'{kit}: not JSON ({e})') from None
    bad = problems(kit)
    if bad:
        raise KitError('invalid kit: ' + '; '.join(bad))
    return kit


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def problems(kit):
    """what is wrong with a kit dict: schema, roles, ids, keys, and every value a render would refuse (knob ranges,
    pitches the sound cannot take, levels over 0 dBFS, take files that would collide); [] when it is valid"""
    from . import bank
    out = []
    if not isinstance(kit, dict) or kit.get('kit') != SCHEMA:
        return [f'not a kit file (needs "kit": {SCHEMA})']
    try:
        bank.check_sr(kit.get('sr', SR))
    except bank.BankError:
        out.append(f"sr {kit.get('sr')!r}: use 48000 (or 44100 / 88200 / 96000)")
    roles = kit.get('roles')
    if not isinstance(roles, dict) or not roles:
        return out + ['roles: a non-empty object {role: spec | null}']
    known = set(bank.ids())
    for role, spec in roles.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*(\.[a-z0-9_]+)*', str(role)):
            out.append(f'role {role!r}: lowercase words joined by dots (hit, whoosh.short)')
        if spec is None:
            continue
        if not isinstance(spec, dict):
            out.append(f'{role}: a spec object or null')
            continue
        extra = set(spec) - set(ROLE_KEYS)
        if extra:
            out.append(f'{role}: unknown keys {sorted(extra)} (known: {", ".join(ROLE_KEYS)})')
        sid = spec.get('sound')
        if sid not in known:
            from difflib import get_close_matches
            near = get_close_matches(str(sid), sorted(known), n=3, cutoff=0.5)
            out.append(f'{role}: unknown sound {sid!r}' + (f' (did you mean {", ".join(near)}?)' if near else ''))
        t = spec.get('takes', 1)
        if not isinstance(t, int) or isinstance(t, bool) or not 1 <= t <= MAX_TAKES:
            out.append(f'{role}: takes must be an integer 1-{MAX_TAKES}')
        if spec.get('seed') is not None and (not isinstance(spec['seed'], int) or isinstance(spec['seed'], bool)):
            out.append(f'{role}: seed must be an integer')
        ed = spec.get('edits') if spec.get('edits') is not None else {}
        if not isinstance(ed, dict) or set(ed) - set(EDITS):
            out.append(f'{role}: edits takes {", ".join(EDITS)}')
            ed = {}
        ps = spec.get('pitches')
        if ps is not None and (not isinstance(ps, list) or not ps):
            out.append(f'{role}: pitches must be a non-empty list of notes / Hz')
            ps = None
        if ps and ed.get('pitch') is not None:
            out.append(f'{role}: give edits.pitch or pitches, not both')
        if spec.get('align', ALIGN.get(role, 'onset')) not in ALIGNS:
            out.append(f'{role}: align must be one of {", ".join(ALIGNS)}')
        if 'level_db' in spec and not _is_num(spec['level_db']):
            out.append(f"{role}: level_db must be a number (leave it out for 0 dB), got {spec['level_db']!r}")
        if spec.get('max_s') is not None and (not _is_num(spec['max_s']) or spec['max_s'] < 0.01):
            out.append(f"{role}: max_s must be a number of seconds, 0.01 or more (a shorter take is silence), got "
                       f"{spec['max_s']!r}")
        out += _edit_problems(role, sid if sid in known else None, ed, ps)
    out += _name_problems(kit)
    return out


def _edit_problems(role, sid, ed, ps):
    """the knob values of one role that a render would refuse (the same checks: bank.level, knobs.validate)"""
    from . import bank, knobs
    out = []
    try:
        bank.level(ed.get('peak_db', -1.0), ed.get('weight'))
    except bank.BankError as e:
        out.append(f'{role}: {e}')
    for k in ('dur', 'apex'):
        if ed.get(k) is not None and not _is_num(ed[k]):
            out.append(f'{role}: edits.{k} must be a number of seconds, got {ed[k]!r}')
            return out
    if sid is None:
        return out
    e = bank.get(sid)
    try:
        knobs.validate(e, dur=ed.get('dur'), apex=ed.get('apex'))
    except knobs.KnobError as x:
        out.append(f'{role}: {x}')
    for p in (ps or [ed.get('pitch')]):
        if p is None:
            continue
        try:
            knobs.validate(e, pitch=p)
        except knobs.KnobError as x:
            out.append(f'{role}: pitch {p!r}: {x}')
    return out


def _take_names(kit):
    """[(role, file name, pitch)] of every take a kit would write, for the specs whose pitches all parse (plan's
    naming)"""
    from . import knobs
    out = []
    for role, spec in (kit.get('roles') or {}).items():
        if not isinstance(spec, dict) or not isinstance(role, str):
            continue
        n = spec.get('takes', 1)
        n = n if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= MAX_TAKES else 1
        ps = spec.get('pitches') if isinstance(spec.get('pitches'), list) and spec.get('pitches') else None
        try:
            labels = [knobs.label(p) for p in ps] if ps else [None]
        except knobs.KnobError:
            continue
        stem = role.replace('.', '-')
        for p, lb in zip(ps or [None], labels):
            for k in range(1, n + 1):
                f = stem + (f'_{lb}' if lb else '') + (f'_{k}' if (n > 1 or not ps) else '') + '.wav'
                out.append((role, f, p))
    return out


def _name_problems(kit):
    """take files that would share a name, compared without regard to case (Windows / macOS file systems)"""
    seen, out = {}, []
    for role, f, p in _take_names(kit):
        key = f.lower()
        if key not in seen:
            seen[key] = (role, f, p)
            continue
        r0, f0, p0 = seen[key]
        if r0 == role and p is not None:
            out.append(f'{role}: pitches {p0!r} and {p!r} are one pitch ({f}): list each pitch once')
        else:
            out.append(f'{role}: take file {f} is also {f0} of role {r0} on a case-insensitive file system: rename '
                       'the role')
    return sorted(set(out), key=out.index)


def init(preset_name='calm', out='audio/kit.json', force=False):
    """write a preset as a film's kit file: {'summary', 'file', 'kit'}"""
    k = preset(preset_name)
    if os.path.exists(out) and not force:
        raise KitError(f'{out} exists (force=True / --force overwrites it)')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(k, indent=1, ensure_ascii=False) + '\n')
    on = [r for r, s in k['roles'].items() if s]
    return {'summary': f"wrote {out}: preset {preset_name}, {len(on)} roles sounded, "
                       f"{len(k['roles']) - len(on)} off ({', '.join(r for r, s in k['roles'].items() if not s)})",
            'file': out.replace(chr(92), '/'), 'kit': k}


# ------------------------------------------------------------------------------------------------------- planning
def _label(p):
    from .knobs import label
    return label(p)


def plan(kit):
    """the takes a kit renders: [{role, file, sound, seed, pitch, edits, max_s, k}]. KitError when two takes would
    share a file (case-insensitively)"""
    from . import bank
    dup = _name_problems(kit)
    if dup:
        raise KitError('invalid kit: ' + '; '.join(dup))
    out = []
    for role, spec in kit['roles'].items():
        if not spec:
            continue
        e = bank.get(spec['sound'])
        base = int(spec['seed']) if spec.get('seed') is not None else int(e['seed'])
        n = int(spec.get('takes', 1))
        stem = role.replace('.', '-')
        ed = dict(spec.get('edits') or {})
        pitches = spec.get('pitches') or [ed.pop('pitch', None)]
        ed.pop('pitch', None)
        for p in pitches:
            for k in range(1, n + 1):
                name = stem + (f'_{_label(p)}' if spec.get('pitches') else '') + (
                    f'_{k}' if (n > 1 or not spec.get('pitches')) else '')
                out.append({'role': role, 'file': name + '.wav', 'sound': spec['sound'], 'seed': base + k - 1,
                            'pitch': p, 'edits': ed, 'max_s': spec.get('max_s'), 'k': k})
    return out


def _digest(o):
    return hashlib.sha1(json.dumps(o, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def take_key(t, sr=SR):
    """the cache key of one planned take (module doc)"""
    from .. import __version__
    from . import bank
    return _digest({'sound': t['sound'], 'entry': _digest(bank.get(t['sound'])), 'seed': t['seed'],
                    'pitch': t['pitch'], 'edits': t['edits'], 'max_s': t['max_s'], 'sr': int(sr),
                    'bank': bank.VERSION, 'earshot': __version__})


def _sha1(path):
    with open(path, 'rb') as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def _render_one(t, seed, role):
    from . import bank, knobs, sfx
    ed = t['edits']
    y, meta = bank.render(t['sound'], seed=seed, dur=ed.get('dur'), apex=ed.get('apex'), pitch=t['pitch'],
                          peak_db=ed.get('peak_db', -1.0), weight=ed.get('weight'), checks=False)
    if t['max_s'] and len(y) > t['max_s'] * SR:
        y = knobs.fit_length(y, float(t['max_s']))[0].astype(np.float32)
    c = sfx.check(y, category=bank.get(t['sound'])['category'], role=role or t['role'], sound=t['sound'])
    bank.add_warning(c, knobs.pitch_warning(meta.get('edits')))
    return y, c, meta


def render_take(t, role=None):
    """one planned take at 48 kHz: (y float32, check dict, meta, the seed used). max_s applied, audited with the
    role as context; a take that fails the audit is rendered again on seed + 1000 j (j = 1 .. REROLL) and the first
    clean one is kept, like a sound designer discarding a bad take (measured on the presets: about 1 take in 20 of
    a new seed rings or stops early); when none is clean the planned seed's take is kept with its warnings."""
    first = None
    for j in range(REROLL + 1):
        s = int(t['seed']) + 1000 * j
        y, c, meta = _render_one(t, s, role)
        if c['ok']:
            return y, c, meta, s
        first = first or (y, c, meta, s)
    return first


# ------------------------------------------------------------------------------------------------------ rendering
def _read_lock(out):
    p = os.path.join(out, LOCK)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def render(kit, out, force=False, log=None):
    """render every take of a kit into `out` (cached, module doc) and write out/kit.lock.json; returns the lock dict
    plus 'summary' and 'stats' {takes, rendered, cached, removed, seconds}"""
    from .. import __version__
    from . import bank, sfx
    t0 = time.perf_counter()
    k = load(kit)
    sr = int(k.get('sr', SR))
    os.makedirs(out, exist_ok=True)
    old = _read_lock(out) or {}
    cache = {}
    for r in (old.get('roles') or {}).values():
        for tk in (r or {}).get('takes') or []:
            cache[(tk.get('key'), tk.get('file'))] = tk
    lock = {'kit': SCHEMA, 'earshot': __version__, 'bank': bank.VERSION, 'sr': sr, 'name': k.get('name'),
            'roles': {}}
    for role, spec in k['roles'].items():
        lock['roles'][role] = None if not spec else {
            'sound': spec['sound'], 'align': spec.get('align', ALIGN.get(role, 'onset')),
            'level_db': float(spec.get('level_db') or 0.0), 'max_s': spec.get('max_s'), 'takes': []}
    rendered = cached = 0
    for t in plan(k):
        key = take_key(t, sr)
        path = os.path.join(out, t['file'])
        hit = cache.get((key, t['file']))
        if not force and hit and os.path.isfile(path) and _sha1(path) == hit.get('sha1'):
            lock['roles'][t['role']]['takes'].append(hit)
            cached += 1
            continue
        y, c, _, used = render_take(t)
        m = bank.measure(y)
        if sr != SR:
            import soxr
            y = np.asarray(soxr.resample(y, SR, sr), np.float32)
        sfx.write(path, y, sr)
        rec = {'file': t['file'], 'seed': used, 'pitch': t['pitch'], **{x: m[x] for x in (
               'onset_s', 'peak_s', 'end_s', 'dur_s', 'peak_db', 'lufs_m_max')}, 'sha1': _sha1(path), 'key': key,
               'checks': [w['code'] for w in c['warnings']]}
        if used != t['seed']:
            rec['planned_seed'] = t['seed']
        lock['roles'][t['role']]['takes'].append(rec)
        rendered += 1
        if log:
            log(f"  {t['file']:24s} {t['sound']:26s} seed {t['seed']:<11d} {m['dur_s']:6.3f} s"
                + (f"  WARN {', '.join(rec['checks'])}" if rec['checks'] else ''))
    keep = {tk['file'] for r in lock['roles'].values() if r for tk in r['takes']}
    removed = 0
    for r in (old.get('roles') or {}).values():
        for tk in (r or {}).get('takes') or []:
            f = tk.get('file')
            if f and f not in keep and os.path.basename(f) == f and os.path.isfile(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
                removed += 1
    path = os.path.join(out, LOCK)
    with open(path + '.tmp', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(lock, indent=1) + '\n')
    os.replace(path + '.tmp', path)
    n = rendered + cached
    warn = sorted({f"{r}:{c}" for r, v in lock['roles'].items() if v for tk in v['takes'] for c in tk['checks']})
    dt = time.perf_counter() - t0
    lock_out = dict(lock)
    lock_out['stats'] = {'takes': n, 'rendered': rendered, 'cached': cached, 'removed': removed,
                         'seconds': round(dt, 2)}
    lock_out['summary'] = (f"kit {k.get('name') or ''}: {n} takes in {out} ({rendered} rendered, {cached} cached"
                           + (f', {removed} stale removed' if removed else '') + f') in {dt:.1f} s; '
                           + ('checks clean' if not warn else 'warnings: ' + ', '.join(warn)))
    lock_out['file'] = path.replace(chr(92), '/')
    return lock_out


def check(kit, out=None, log=None):
    """validate a kit and audit every take (the rendered files in `out` when the lock there is current for them,
    else renders in memory): {'ok', 'summary', 'problems', 'roles': {role: [take rows]}, 'notes'}"""
    from . import bank, sfx
    try:
        if isinstance(kit, dict):
            k = kit
        else:
            with open(kit, encoding='utf-8') as fh:
                k = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        return {'ok': False, 'summary': f'kit unreadable: {e}', 'problems': [str(e)], 'roles': {}, 'notes': []}
    probs = problems(k)
    if probs:
        return {'ok': False, 'summary': f'{len(probs)} problem(s): ' + '; '.join(probs), 'problems': probs,
                'roles': {}, 'notes': []}
    notes = []
    for role, spec in k['roles'].items():
        if role.split('.')[0] in RESTRAINED and spec:
            notes.append(f'{role} is sounded: the presets keep {", ".join(RESTRAINED)} off (one clean element on a '
                         'reveal reads better than a build stacked onto it)')
        if spec:
            cat = spec['sound'].split('/')[0]
            why = NEVER.get(spec['sound']) or NEVER_CATEGORIES.get(cat)
            if why:
                notes.append(f"{role}: {spec['sound']} is {why}; the presets never use it")
        if role.split('.')[0] not in {r.split('.')[0] for r in ROLES}:
            notes.append(f'{role}: not one of the shared roles ({", ".join(ROLES)}); a renderer must know it')
    lock = _read_lock(out) if out else None
    have = {}
    for r in ((lock or {}).get('roles') or {}).values():
        for tk in (r or {}).get('takes') or []:
            have[tk.get('key')] = tk
    sr = int(k.get('sr', SR))
    rows, bad = {}, 0
    for t in plan(k):
        row = {'file': t['file'], 'sound': t['sound'], 'seed': t['seed'], 'pitch': t['pitch']}
        try:
            tk = have.get(take_key(t, sr))
            if tk and os.path.isfile(os.path.join(out, tk['file'])):
                c = sfx.check(os.path.join(out, tk['file']), category=bank.get(t['sound'])['category'],
                              role=t['role'], sound=t['sound'])
                if 'pitch_off' in (tk.get('checks') or []):   # measured at render (the file does not say)
                    bank.add_warning(c, {'code': 'pitch_off', 'msg': f"pitch {t['pitch']}: the knob did not land "
                                                                     'the note (kit.lock.json)'})
                row['source'] = 'rendered file'
            else:
                _, c, _, used = render_take(t)
                row['source'] = 'rendered in memory'
                if used != t['seed']:
                    row['seed'], row['planned_seed'] = used, t['seed']
            row.update(ok=c['ok'], warnings=c['warnings'], notes=c['notes'])
        except Exception as e:  # a knob a sound cannot take: say which, keep checking the rest
            row.update(ok=False, warnings=[{'code': 'render', 'msg': f'{type(e).__name__}: {e}'}], notes=[])
        bad += not row['ok']
        rows.setdefault(t['role'], []).append(row)
        if log:
            log(f"  {t['file']:24s} {t['sound']:26s} " + ('ok' if row['ok'] else '; '.join(
                w['msg'] for w in row['warnings'])))
    n = sum(len(v) for v in rows.values())
    off = [r for r, s in k['roles'].items() if not s]
    warned = [f"{r['file']}: {', '.join(w['code'] for w in r['warnings'])}" for v in rows.values() for r in v
              if not r['ok']]
    summary = (f"kit {k.get('name') or ''}: {n - bad} / {n} takes clean"
               + (f", {bad} with warnings: " + '; '.join(warned) if warned else '')
               + (f"; off: {', '.join(off)}" if off else '') + (f'; {len(notes)} note(s)' if notes else ''))
    return {'ok': not bad, 'summary': summary, 'problems': [], 'roles': rows, 'notes': notes}


def audition(kit, out_wav, takes=None, gap_s=0.3, role_gap_s=0.8):
    """every take of a kit in role order, each at its role's level_db, into one WAV (renders the takes first, cached
    in `takes` or in EARSHOT_HOME/kits/<name>_<hash>): {'summary', 'file', 'index': [{t, role, file, sound}]}"""
    from .. import config
    from . import sfx
    k = load(kit)
    if takes is None:
        takes = os.path.join(config.HOME, 'kits', f"{k.get('name') or 'kit'}_{_digest(k)[:10]}")
    lock = render(k, takes)
    import soundfile as sf
    parts, index, t = [], [], 0.0
    sr = lock['sr']
    for role, r in lock['roles'].items():
        if not r:
            continue
        g = 10 ** (float(r.get('level_db') or 0.0) / 20)
        for tk in r['takes']:
            y, fs = sf.read(os.path.join(takes, tk['file']), always_2d=True, dtype='float32')
            y = np.repeat(y, 2, 1) if y.shape[1] == 1 else y[:, :2]
            index.append({'t': round(t, 3), 'role': role, 'file': tk['file'], 'sound': r['sound'],
                          'pitch': tk.get('pitch')})
            parts.append(y * g)
            parts.append(np.zeros((int(gap_s * sr), 2), np.float32))
            t += len(y) / sr + gap_s
        parts.append(np.zeros((int(role_gap_s * sr), 2), np.float32))
        t += role_gap_s
    if not parts:
        raise KitError('the kit sounds no role')
    y = np.concatenate(parts)
    pk = np.abs(y).max()
    if pk > 10 ** (-1 / 20):                    # level_db above 0 on a -1 dBFS take: keep the audition unclipped
        y *= 10 ** (-1 / 20) / pk
    f = sfx.write(out_wav, y, sr)
    with open(os.path.splitext(out_wav)[0] + '.txt', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(f"{i['t']:8.3f}  {i['role']:14s} {i['sound']:26s} {i['file']}" for i in index) + '\n')
    return {'summary': f"wrote {f}: {len(index)} takes of {len({i['role'] for i in index})} roles, "
                       f"{len(y) / sr:.1f} s (index in {os.path.splitext(f)[0]}.txt)",
            'file': f, 'index': index, 'takes_dir': takes.replace(chr(92), '/')}


# --------------------------------------------------------------------------------------------------- the API
def kit_init(preset='calm', out='audio/kit.json', force=False):
    return init(preset, out, force)


def kit_render(kit, out, force=False):
    return render(kit, out, force)


def kit_check(kit, out=None):
    return check(kit, out)


def kit_audition(kit, out, takes=None):
    return audition(kit, out, takes)


# ------------------------------------------------------------------------------------------------------------ CLI
def _flags(p):
    """--json and --threads: the kit commands that take a folder use --out for it, not for a JSON copy"""
    import argparse
    p.add_argument('--json', action='store_true', help='print the full result as JSON')
    p.add_argument('--threads', type=int, default=argparse.SUPPRESS, help='CPU threads (default EARSHOT_THREADS)')
    return p


def add_cli(sub):
    from ..cli import common
    p = sub.add_parser('kit', help="a film's sound kit: one bank sound per role, rendered to takes with measured "
                                   'sync points (no models)')
    s = p.add_subparsers(dest='kit_cmd', metavar='<init|render|check|audition>')
    pi = common(s.add_parser('init', help='write a preset kit file'))
    pi.add_argument('--preset', default='calm', help='calm (text-led films, default), crisp, minimal')
    pi.add_argument('-o', '--output', default='audio/kit.json')
    pi.add_argument('--force', action='store_true', help='overwrite an existing file')
    pr = _flags(s.add_parser('render', help='render every take (cached) and write kit.lock.json'))
    pr.add_argument('kit')
    pr.add_argument('--out', dest='takes', required=True, help='folder for the takes and kit.lock.json')
    pr.add_argument('--force', action='store_true', help='render every take again')
    pc = _flags(s.add_parser('check', help='validate the kit and audit every take (model-free)'))
    pc.add_argument('kit')
    pc.add_argument('--out', dest='takes', help='the rendered takes, when current (else rendered in memory)')
    pa = _flags(s.add_parser('audition', help='every take in role order into one WAV'))
    pa.add_argument('kit')
    pa.add_argument('-o', '--output', required=True, help='audition WAV (an index .txt is written next to it)')
    pa.add_argument('--out', dest='takes', help='folder of the takes (default: a cache folder in EARSHOT_HOME)')

    def run(a):
        if a.kit_cmd == 'init':
            r = init(a.preset, a.output, a.force)
            emit(r, a, text=r['summary'])
            return 0
        if a.kit_cmd == 'render':
            r = render(a.kit, a.takes, a.force, log=None if a.json else print)
            emit(r, a, text=r['summary'])
            return 0
        if a.kit_cmd == 'check':
            r = check(a.kit, a.takes, log=None if a.json else print)
            emit(r, a, text='\n'.join([r['summary']] + [f'note: {n}' for n in r['notes']]))
            return 0 if r['ok'] else 1
        if a.kit_cmd == 'audition':
            r = audition(a.kit, a.output, a.takes)
            emit(r, a, text=r['summary'])
            return 0
        p.print_help()
        return 2
    p.set_defaults(func=run)
    return p
