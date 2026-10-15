#!/usr/bin/env python
"""Apply design operations: build a variant to test (critics) or apply the agreed fix (the fixer).

  python -I <skill>/scripts/variant.py --ops ops.json --out out/audio-review/critic_tonal/v1 [--render] [--qa]
                                       [--cues out/cues.json] [--cache DIR]
  python -I <skill>/scripts/variant.py --ops ops.json --apply [--backup out/audio-review/backup_round1]
                                       [--root .] [--design audio/sounddesign.json]

A variant is a copy of the top-level design and its fragments in --out (design/<fragment>.json, include paths
rewritten), with the operations applied; --render mixes it with every output inside --out (soundtrack.wav, stems/,
report.json, cues.json, adds.json) and --qa runs qa.py on it (qa.json). --apply edits the real files instead, after
copying them to --backup (never overwritten). Operations (a JSON list, applied in order; "f" = fragment file name
without .json, "sd" = a dotted path in the top-level design):
  {"f": "keyboard", "sample": "kb.thock.1", "set": {"norm_peak_db": -10}}      also "slice": name
  {"f": "keyboard", "sample": "kb.thock.1", "eq": {"highshelf": [5000, -6]}}   merge into its eq
  {"f": "motion", "rule": "mo.leap.air", "set": {"offset": 0.3}}              {"..., "rule": name, "delete": true}
  {"f": "motion", "rule_add": {...}}   {"f": "garden", "place": "gd.crowd_a", "set": {...}}   {"f": ..., "place_add": {...}}
  {"f": "ambience", "amb": "am.room", "set": {"gain_db": -3}}                 {"f": "music", "segment": 2, "set": {...}}
  {"f": "toy", "pool": "ty.p.row", "value": ["ty.row.a", "ty.row.b"]}
  {"sd": "buses.music.auto", "value": [[0, 0], [12, -3]]}                     {"sd": "master.lufs", "value": -15}
Prints every change as old -> new (or unchanged); an operation that matches nothing is an error (nothing is
written). Only files whose content changes are written, and only the changed values are rewritten in place (hand
alignment, key order, comments and number spelling elsewhere stay as they were), so the fix reads as a small diff; the
files written are listed.
"""
import argparse, copy, difflib, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = re.compile(r'[ \t\r\n]*')
DEC = json.JSONDecoder()


# ------------------------------------------------------------------ format-preserving JSON edits
def scan(t, i, path, sp, kids):
    """parse the JSON value at t[i:], recording sp[path] = (item_start, value_start, value_end) for every value and
    kids[path] = child keys / indices in order; returns the end index"""
    i = WS.match(t, i).end()
    item = sp.get(path, (i,))[0]
    if t[i] in '{[':
        close, kids[path] = '}' if t[i] == '{' else ']', []
        j = WS.match(t, i + 1).end()
        n = 0
        while t[j] != close:
            if close == '}':
                key, k = json.decoder.scanstring(t, j + 1)
                k = WS.match(t, k).end() + 1  # past the colon
                sub = path + (key,)
            else:
                k, sub = j, path + (n,)
            sp[sub] = (j,)
            kids[path].append(sub[-1])
            j = WS.match(t, scan(t, k, sub, sp, kids)).end()
            if t[j] == ',':
                j = WS.match(t, j + 1).end()
            n += 1
        sp[path] = (item, i, j + 1)
        return j + 1
    _, j = DEC.raw_decode(t, i)
    sp[path] = (item, i, j)
    return j


def dump1(v):
    return json.dumps(v, ensure_ascii=False)


def edits_for(t, old, new, path, sp, kids, out):
    """text edits (start, end, replacement) that turn the value at path from old into new, touching only what changed"""
    if old == new and type(old) is type(new):
        return
    if isinstance(old, dict) and isinstance(new, dict) and path in kids:
        ks = kids[path]
        m = 0
        while m < len(ks):
            if ks[m] in new:
                edits_for(t, old[ks[m]], new[ks[m]], path + (ks[m],), sp, kids, out)
                m += 1
                continue
            m1 = m  # a run of deleted entries m..m1, removed with one separator
            while m1 + 1 < len(ks) and ks[m1 + 1] not in new:
                m1 += 1
            at = lambda n, w: sp[path + (ks[n],)][w]
            out.append((at(m, 0), at(m1 + 1, 0), '') if m1 + 1 < len(ks) else
                       (at(m - 1, 2), at(m1, 2), '') if m else (at(m, 0), at(m1, 2), ''))
            m = m1 + 1
        add = [k for k in new if k not in old]
        if add:
            insert(t, path, sp, kids, [f'{dump1(k)}: {dump1(new[k])}' for k in add], out)
        return
    if isinstance(old, list) and isinstance(new, list) and path in kids:
        sm = difflib.SequenceMatcher(None, list(map(ident, old)), list(map(ident, new)), autojunk=False)
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op == 'equal' or (op == 'replace' and a1 - a0 == b1 - b0):  # the same items (or one for one): patch inside
                for d in range(a1 - a0):
                    edits_for(t, old[a0 + d], new[b0 + d], path + (a0 + d,), sp, kids, out)
            elif op == 'insert':
                insert(t, path, sp, kids, [dump1(x) for x in new[b0:b1]], out, after=a0 - 1)
            else:  # delete / uneven replace: the old items' whole stretch (with one separator) -> the new items
                s0 = sp[path + (a0,)][0]
                e0 = sp[path + (a1 - 1,)][2]
                txt = sep(t, path, sp, kids).join(dump1(x) for x in new[b0:b1])
                if not txt:
                    if a1 < len(old):
                        e0 = sp[path + (a1,)][0]
                    elif a0:
                        s0 = sp[path + (a0 - 1,)][2]
                out.append((s0, e0, txt))
        return
    _, s0, e0 = sp[path]
    out.append((s0, e0, dump1(new)))


def ident(x):
    """what makes a list item the same item after an edit: a dict's name / sample / label, else its whole value"""
    k = [(key, dump1(x[key])) for key in ('name', 'sample', 'label') if isinstance(x, dict) and key in x]
    return json.dumps(k) if k else dump1(x)


def sep(t, path, sp, kids):
    """the separator used between the container's items (',' + its line break and indent), from its last item"""
    ks = kids[path]
    if len(ks) < 2:
        return ', '
    a, b = sp[path + (ks[-2],)][2], sp[path + (ks[-1],)][0]
    return t[a:b] if t[a:b].strip() == ',' else ', '


def insert(t, path, sp, kids, items, out, after=None):
    ks = kids[path]
    if not ks:
        _, s0, e0 = sp[path]
        out.append((s0 + 1, s0 + 1, ', '.join(items)))
        return
    s = sep(t, path, sp, kids)
    if after is None:
        after = len(ks) - 1
    if after < 0:
        out.append((sp[path + (ks[0],)][0],) * 2 + (s.join(items) + s,))
    else:
        e = sp[path + (ks[after],)][2]
        out.append((e, e, s + s.join(items)))


def patch_text(text, new):
    """text of a JSON file with only the values that differ from `new` rewritten; None when that is not possible"""
    try:
        old = json.loads(text)
        sp, kids, out = {}, {}, []
        scan(text, 0, (), sp, kids)
        edits_for(text, old, new, (), sp, kids, out)
        out.sort()
        if any(out[n][1] > out[n + 1][0] for n in range(len(out) - 1)):
            return None
        for s0, e0, rep in reversed(out):
            text = text[:s0] + rep + text[e0:]
        return text if json.loads(text) == new else None
    except (ValueError, KeyError, IndexError):
        return None


def write_json(path, new, base_text=None):
    """write `new` to path keeping base_text's formatting where possible; returns 'patched' or 'reformatted'"""
    txt = patch_text(base_text, new) if base_text is not None else None
    how = 'patched' if txt is not None else 'reformatted'
    if txt is None:
        txt = json.dumps(new, indent=1, ensure_ascii=False) + '\n'
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(txt)
    return how


def find(items, key, name):
    for i, it in enumerate(items):
        if str(it.get(key)) == str(name):
            return i
    raise KeyError(f'no item with {key} = {name!r}')


def change(what, old, new):
    if old == new and type(old) is type(new):
        return f'{what}: unchanged ({json.dumps(old)[:80]})'
    return f'{what}: {json.dumps(old)[:80]} -> {json.dumps(new)[:80]}'


def apply_ops(top, frags, ops):
    """apply operations to in-memory copies; returns a list of change lines"""
    log = []
    for op in ops:
        if 'sd' in op:
            node, keys = top, op['sd'].split('.')
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            log.append(change(f"sounddesign.{op['sd']}", node.get(keys[-1]), op['value']))
            node[keys[-1]] = op['value']
            continue
        f = op['f']
        if f not in frags:
            raise KeyError(f'no fragment "{f}" (have {", ".join(sorted(frags))})')
        d = frags[f]
        if 'sample' in op or 'slice' in op:
            kind = 'samples' if 'sample' in op else 'slices'
            name = op.get('sample') or op.get('slice')
            s = d.get(kind, {}).get(name)
            if s is None:
                raise KeyError(f'{f}: no {kind[:-1]} "{name}"')
            for k, v in (op.get('set') or {}).items():
                log.append(change(f'{f}.{name}.{k}', s.get(k), v))
                s[k] = v
            if 'eq' in op:
                old = copy.deepcopy(s.get('eq'))
                s.setdefault('eq', {}).update(op['eq'])
                log.append(change(f'{f}.{name}.eq', old, s['eq']))
        elif 'rule' in op or 'place' in op or 'amb' in op:
            kind, key, name = (('rules', 'name', op['rule']) if 'rule' in op else ('place', 'sample', op['place'])
                               if 'place' in op else ('ambience', 'name', op['amb']))
            items = d.get(kind, [])
            i = find(items, key, name)
            if op.get('delete'):
                log.append(f'{f}.{kind}[{name}]: deleted')
                items.pop(i)
                continue
            for k, v in (op.get('set') or {}).items():
                log.append(change(f'{f}.{kind}[{name}].{k}', items[i].get(k), v))
                items[i][k] = v
        elif 'rule_add' in op or 'place_add' in op:
            kind = 'rules' if 'rule_add' in op else 'place'
            d.setdefault(kind, []).append(op.get('rule_add') or op.get('place_add'))
            log.append(f'{f}.{kind}: added {json.dumps(d[kind][-1])[:100]}')
        elif 'segment' in op:
            sg = d['music']['segments'][int(op['segment'])]
            for k, v in (op.get('set') or {}).items():
                log.append(change(f"{f}.music.segments[{op['segment']}].{k}", sg.get(k), v))
                sg[k] = v
        elif 'pool' in op:
            log.append(change(f"{f}.pools.{op['pool']}", d.get('pools', {}).get(op['pool']), op['value']))
            d.setdefault('pools', {})[op['pool']] = op['value']
        else:
            raise ValueError(f'unknown operation {op}')
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default='.')
    ap.add_argument('--design', default='audio/sounddesign.json')
    ap.add_argument('--ops', required=True, help='JSON list of operations (or [] for an unchanged copy)')
    ap.add_argument('--out', help='variant folder (critics)')
    ap.add_argument('--apply', action='store_true', help='edit the real design files (the fixer)')
    ap.add_argument('--backup', default='out/audio-review/backup_before_fix', help='with --apply: copy the originals here first')
    ap.add_argument('--render', action='store_true', help='mix the variant (outputs inside --out)')
    ap.add_argument('--qa', action='store_true', help='also run qa.py on the rendered variant')
    ap.add_argument('--cache', help='decode cache for --render / --qa (default: <root>/out/audio/cache, as mix.py)')
    ap.add_argument('--cues', help='cue file for the variant (written into its design; default: the design\'s "cues", '
                                   'else out/cues.json)')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    R = lambda p: os.path.normpath(p if os.path.isabs(p) else os.path.join(root, p))
    if bool(a.out) == bool(a.apply):
        sys.exit('give exactly one of --out (a variant) or --apply (edit the design)')
    if not os.path.exists(R(a.design)):
        sys.exit(f'design file not found: {R(a.design)} (pass --design)')
    text = {p: open(R(p), encoding='utf-8', newline='').read() for p in [a.design] + json.load(
        open(R(a.design), encoding='utf-8')).get('include', [])}
    top = json.loads(text[a.design])
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in top.get('include', [])}
    frags = {k: json.loads(text[p]) for k, p in paths.items()}
    ops = json.load(open(R(a.ops), encoding='utf-8')) if os.path.exists(R(a.ops)) else json.loads(a.ops)
    new_top, new_frags = copy.deepcopy(top), copy.deepcopy(frags)
    try:
        log = apply_ops(new_top, new_frags, ops)
    except (KeyError, ValueError, IndexError) as e:
        sys.exit(f'nothing written: {e}')
    if a.apply:
        bk = R(a.backup)
        if os.path.exists(bk):
            sys.exit(f'backup {bk} exists: pass a new --backup')
        os.makedirs(os.path.join(bk, 'design'))
        shutil.copy(R(a.design), os.path.join(bk, os.path.basename(a.design)))
        for k, p in paths.items():
            shutil.copy(R(p), os.path.join(bk, 'design', os.path.basename(p)))
        todo = [(a.design, new_top, top)] + [(p, new_frags[k], frags[k]) for k, p in paths.items()]
        written = [(p, write_json(R(p), new, text[p])) for p, new, old in todo if new != old]
        dest = f'the design (originals in {bk})'
    else:
        out = R(a.out)
        os.makedirs(os.path.join(out, 'design'), exist_ok=True)
        new_top['include'] = [os.path.join(out, 'design', os.path.basename(p)).replace('\\', '/') for p in top['include']]
        if a.cues or top.get('cues'):
            new_top['cues'] = R(a.cues or top['cues']).replace('\\', '/')
        dpath = os.path.join(out, 'sounddesign.json').replace('\\', '/')
        written = [(os.path.join(out, 'design', os.path.basename(p)), write_json(
            os.path.join(out, 'design', os.path.basename(p)), new_frags[k], text[p])) for k, p in paths.items()]
        written.append((dpath, write_json(dpath, new_top, text[a.design])))
        dest = dpath
    n = sum(1 for line in log if ': unchanged (' not in line)
    print(f'{n} change(s), {len(log) - n} already as asked -> {dest}' if n < len(log) else f'{n} change(s) -> {dest}')
    for line in log:
        print('  ' + line.encode('ascii', 'replace').decode())
    for p, how in written:
        print(f'  wrote {p} ({how}' + (', only the changed values rewritten)' if how == 'patched' else
                                       ': could not patch in place, rewritten with indent 1)'))
    if a.out and (a.render or a.qa):
        o = lambda n: os.path.join(R(a.out), n)
        cmd = [sys.executable, '-I', os.path.join(HERE, 'mix.py'), '--root', root, '--design', dest, '--out', o('soundtrack.wav'),
               '--stems', o('stems'), '--report', o('report.json'), '--cues-out', o('cues.json'), '--trace', o('adds.json')]
        cache = ['--cache', R(a.cache)] if a.cache else []
        r = subprocess.run(cmd + cache, capture_output=True, text=True)
        print(r.stdout[-1200:], r.stderr[-800:])
        if r.returncode:
            return r.returncode
        if a.qa:
            r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'qa.py'), '--root', root, '--design', dest,
                                '--report', o('report.json'), '--trace', o('adds.json'), '--stems', o('stems'),
                                '--master', o('soundtrack.wav'), '--out', o('qa.json')] + cache, capture_output=True, text=True)
            print(r.stdout, r.stderr[-800:])
    return 0


if __name__ == '__main__':
    sys.exit(main())
