"""Build the forge's per-category reference sets from the licensed library (dev machine only; nothing is committed).

    python eval/forge/build_refsets.py                      # every category (re-uses cached earshot finds)
    python eval/forge/build_refsets.py --cats click tap     # only these (merged into the existing manifest)
    python eval/forge/build_refsets.py --plan               # print the plan (sources per split), no audio work

Sources: demo/launch-sfx-kit/sources.json (licences, paths into the local licensed library) and the kit WAVs'
own trims (kit.json: the source file + trim of every kit sound; re-cut here from the source with trim + fade only, no
high-pass). Each source entry is cut one of four ways:
  kit    the kit's trim of the source (its fades)
  find   earshot find(path, query) one-shot segments: clean edges (no head / tail cut, except at the file's own
         start / end), length inside the category's range, z >= 2 (weak or better; toggle 1.5: CLAP hears switches
         only weakly), peak within 18 dB of the file's loudest candidate; best k by score
  whole  the one sound in the file: from 5 ms before the first frame within 30 dB of the peak until the last frame
         above peak - 45 dB (+20 ms), capped at the category's max length (a capped tail gets a 0.25 s fade)
  rise   a crescendo: back from the loudness peak while within 35 dB of it (at most max length), until peak + 0.3 s
Every cut gets fades (2 ms in; 30 ms out, 0.25 s when capped) and peak gain to -1 dBFS, nothing else. Evidence per
WAV: earshot rank z of the category query (one CLAP read of the exported file), the find z / edges when found, the
kit's own earshot evidence, the measured t40 tail and length.

The split: every category lists its 'forge' and 'bench' sources explicitly. A source FAMILY (Freesound: the author,
which keeps sister takes and sample packs together; VSCO: the library; Mixkit: the file) may be forge or bench, never both, in
any category; the build refuses to write a registry that breaks this. Sources the v1 forge recipes used as refs (the
kit WAVs named in earshot/forge/recipes/*.py) are on the forge side, so the bench is held out from v1 too, except
where a category has nothing else (noted).

Outputs: <repo>/out/refs/<category>/{forge,bench}/*.wav, out/refs/manifest.json (source, trim, licence, earshot
evidence, rejected candidates and why) and the registry <EARSHOT_HOME>/refsets.json (earshot.forge.refsets).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE)).replace(chr(92), '/')
sys.path.insert(0, REPO)
from earshot import audio  # noqa: E402
from earshot.forge import refsets  # noqa: E402
sys.path.insert(0, HERE)
from frozen_tone import t40_ms  # noqa: E402

KIT = REPO + '/demo/launch-sfx-kit'
OUT = REPO + '/out/refs'
SR = 48000
CACHE = OUT + '/_work/find_cache.json'

# category: query (earshot find / rank, used only to locate and evidence the trims; the benchmark's phrasings are
# different), dur (min, max s), forge / bench entries: 'kit:<name>' | (source id, k, mode[, {start, end}]), note
CATS = {
    'whoosh_fast': dict(query='a fast crisp whoosh of a stick swung through the air', dur=(0.2, 0.9),
                        forge=['kit:whoosh_fast_01', 'kit:whoosh_fast_02', 'kit:whoosh_fast_03', 'kit:whoosh_fast_04',
                               ('fs-719636', 1, 'whole')],
                        bench=[('fs-352719', 1, 'whole'), ('fs-216675', 1, 'whole'), ('fs-463763', 3, 'find')]),
    'whoosh_big': dict(query='a big deep heavy whoosh passing by', dur=(0.7, 3.2),
                       forge=['kit:whoosh_big_01', ('mk-1167', 1, 'whole'), ('mk-1168', 1, 'whole', {'start': 0.9})],
                       bench=['kit:whoosh_big_02', ('fs-711122', 3, 'find', {'start': 13.0, 'end': 19.5})],
                       note='forge = three designed Mixkit whooshes; bench = real air / cape swoops.'),
    'swish': dict(query='a quick light swish of something thrown', dur=(0.12, 0.55),
                  forge=[('fs-427979', 4, 'find'), ('fs-719637', 1, 'whole'), ('fs-425856', 2, 'find')],
                  bench=[('fs-463763', 4, 'find'), ('fs-216675', 1, 'whole'), ('fs-352719', 1, 'whole')]),
    'swipe': dict(query='a soft cloth swipe, fabric swished quickly', dur=(0.25, 1.0),
                  forge=['kit:swipe_02', 'kit:whoosh_med_03', ('fs-425856', 2, 'find')],
                  bench=['kit:swipe_01', ('fs-495390', 3, 'find')],
                  note='cloth only: the library has no paper slides.'),
    'riser': dict(query='a rising swell building up to a loud peak', dur=(1.5, 6.0),
                  forge=['kit:riser_01'],
                  bench=['kit:riser_02', ('fs-425434', 1, 'rise')],
                  note='thin: one designed trailer riser (forge); the bench is two real cymbal swells, no synth risers.'),
    'downlifter': dict(query='a falling downlifter sweep sinking away', dur=(1.0, 4.0), forge=[], bench=[],
                       note='empty: the licensed library has no downlifter, real or designed (no reversing or pitching '
                            'of other sounds here: trim + fade only). Benchmark made downlifters on phrasings, '
                            'AudioSet and artifacts only.'),
    'impact': dict(query='a deep cinematic boom impact hit', dur=(1.0, 8.0),
                   forge=['kit:impact_01', 'kit:impact_02'],
                   bench=['kit:impact_03'],
                   note='thin: the library has three cinematic hits (all three were the v1 impact forge refs); one '
                        'is held out as bench, so v1 impact bench numbers are optimistic.'),
    'sub_drop': dict(query='a deep sub bass drop', dur=(1.0, 4.0),
                     forge=[], bench=[('fs-428073', 1, 'whole')],
                     note='one designed source (a tom pitched down 24 semitones by its author), held out as bench; '
                          'the forge has no sub-drop refs.'),
    'click': dict(query='a small plastic button click', dur=(0.03, 0.35),
                  forge=['kit:ui_click_01', 'kit:ui_click_02', 'kit:ui_click_03', 'kit:ui_click_04',
                         ('fs-167829', 2, 'find'), ('fs-383863', 2, 'find')],
                  bench=[('mk-1124', 1, 'whole'), ('mk-2364', 1, 'whole'), ('fs-610524', 2, 'find'),
                         ('fs-461799', 2, 'find'), ('fs-431419', 2, 'find')]),
    'tap': dict(query='a soft finger tap on a hard surface', dur=(0.03, 0.3),
                forge=['kit:ui_click_01', ('fs-704862', 3, 'find'), ('fs-452621', 3, 'find')],
                bench=[('fs-631822', 3, 'find'), ('fs-707827', 3, 'find')]),
    'toggle': dict(query='a toggle switch flipped with a crisp click', dur=(0.05, 0.45), zmin=1.5,
                   forge=['kit:ui_toggle_01', 'kit:ui_toggle_02', ('fs-364200', 1, 'whole'), ('fs-448393', 1, 'whole')],
                   bench=[('fs-278205', 1, 'whole'), ('mk-2585', 1, 'whole'),
                          ('fs-401803', 2, 'find', {'end': 5.0})]),
    'hover_tick': dict(query='a tiny soft tick', dur=(0.012, 0.15),
                       forge=['kit:ui_tick_01', 'kit:ui_tick_02', ('fs-376195', 2, 'find'), ('fs-329098', 2, 'find')],
                       bench=[('fs-401803', 3, 'find', {'start': 72.0}), ('fs-873306', 2, 'find', {'end': 5.2}),
                              ('fs-326470', 2, 'find')]),
    'pop': dict(query='a short round pop', dur=(0.04, 0.5),
                forge=['kit:ui_pop_01', 'kit:ui_pop_02', 'kit:ui_pop_03', 'kit:ui_pop_04', ('mk-3000', 1, 'whole'),
                       ('fs-543200', 2, 'find')],
                bench=[('fs-202230', 1, 'find'), ('fs-33369', 1, 'find'), ('fs-545201', 1, 'find'),
                       ('fs-573152', 2, 'find'), ('mk-3192', 1, 'whole'), ('mk-3070', 1, 'whole')]),
    'notify': dict(query='a short bright notification ding', dur=(0.4, 2.0),
                   forge=['kit:ui_notify_01', ('fs-681787', 1, 'whole'), ('fs-681789', 1, 'whole'),
                          ('fs-681792', 1, 'whole'), ('fs-549899', 1, 'whole')],
                   bench=[('fs-410636', 1, 'whole'), ('fs-410649', 1, 'whole'), ('fs-410664', 1, 'whole'),
                          ('fs-170237', 1, 'whole', {'end': 1.3}), ('vsco-Triangle6-Hit_v1_rr1', 1, 'whole')],
                   note='struck bells / glass / kalimba (forge) and celesta / glass / triangle (bench), rings capped '
                        'at 2 s with a 0.25 s fade; no designed UI dings in the library.'),
    'shimmer': dict(query='a magical shimmering chime sweep', dur=(1.0, 6.0),
                    forge=['kit:shimmer_02', ('fs-271355', 1, 'find'), ('mk-2014', 1, 'whole'), ('fs-495327', 2, 'find')],
                    bench=['kit:shimmer_01', ('fs-257558', 1, 'find'), ('vsco-BellTree_Stroke1', 1, 'whole'),
                           ('vsco-glock_fx_up_chromatic_slow_01', 1, 'whole')],
                    note='kit shimmer_01 (bar chimes sweep) is bench: its author also recorded the celesta notify bench.'),
    'typing_key': dict(query='a single mechanical keyboard key press', dur=(0.06, 0.45),
                       forge=['kit:kb_thock_01', 'kit:kb_thock_02', 'kit:kb_thock_03', 'kit:kb_thock_04',
                              'kit:kb_enter_01', 'kit:kb_space_01', 'kit:kb_click_01', 'kit:kb_click_02',
                              'kit:kb_enter_02', ('fs-561697', 1, 'find'), ('fs-561699', 1, 'find'),
                              ('fs-709460', 2, 'find'), ('fs-665075', 2, 'find')],
                       bench=[('fs-378083', 1, 'find'), ('fs-378084', 1, 'find'), ('fs-378085', 1, 'find'),
                              ('fs-570754', 1, 'find'), ('fs-766639', 1, 'find'), ('fs-766640', 1, 'find'),
                              ('fs-609534', 3, 'find'), ('fs-515810', 1, 'find'), ('fs-815614', 2, 'find')]),
    'glitch': dict(query='a digital glitch stutter', dur=(0.1, 1.0), forge=[], bench=[],
                   note='empty: no glitch material, real or designed, in the licensed library. Benchmark made '
                        'glitches on phrasings, AudioSet and artifacts only.'),
    'logo_sting': dict(query='a struck chime ringing out, a logo sting', dur=(1.0, 6.0),
                       forge=['kit:sting_tail_01', ('mk-3108', 1, 'whole')],
                       bench=[('vsco-Triangle6-Hit_v1_rr1', 1, 'whole'), ('fs-170237', 1, 'whole', {'end': 1.3})],
                       note='no real logo stings exist in the library (they are composed); the refs are the struck '
                            'metal / glass rings a sting ends on: a tail-shape and tone reference only.'),
}


def sources():
    with open(KIT + '/sources.json', encoding='utf-8') as fh:
        return {s['id']: s for s in json.load(fh)}


def kit_sounds():
    with open(KIT + '/kit.json', encoding='utf-8') as fh:
        k = json.load(fh)
    return {s['name']: s for a in k['areas'] for s in a['sounds']}


def family(src):
    """sister takes stay together: Freesound by author, VSCO as one library, Mixkit by file"""
    if src['id'].startswith('vsco-'):
        return 'vsco'
    return f"fs-author:{src['author']}" if src['id'].startswith('fs-') else src['id']


def env_db(m, ms=5.0):
    n = int(SR * ms / 1000)
    k = max(1, len(m) // n)
    return 20 * np.log10(np.sqrt((m[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-12), n


def cut_whole(m, dmax, rise=False):
    """(a, b, capped) sample trims of the one sound in mono m (see the module docstring: whole / rise)"""
    e, n = env_db(m)
    p = int(np.argmax(e))
    if rise:
        j = p
        while j > 0 and e[j - 1] > e[p] - 35 and (p - j + 1) * n < (dmax - 0.3) * SR:
            j -= 1
        return j * n, min(len(m), p * n + int(0.3 * SR)), False
    on = int(np.nonzero(e >= e[p] - 30)[0][0])
    a = max(0, on * n - int(0.005 * SR))
    last = int(np.nonzero(e >= e[p] - 45)[0][-1])
    b = min(len(m), (last + 1) * n + int(0.02 * SR))
    if b - a > dmax * SR:
        return a, a + int(dmax * SR), True
    return a, b, False


def load_cache():
    try:
        with open(CACHE, encoding='utf-8') as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def find_cached(cache, path, query, top, start=None, end=None):
    key = f'{path}|{query}|{top}|{start}|{end}'
    if key not in cache:
        from earshot import api
        r = api.find(path, query, top=top, start=start, end=end)
        cache[key] = r['segments']
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh)
    return cache[key]


def real_cut(s, t0, t1):
    """find's 'head cut' / 'tail cut' flags, except at the edges of the searched range (a one-shot file that starts
    on the attack cannot be trimmed earlier: its head is not cut)"""
    return [c for c in (s['cut'] or '').split(', ') if c and not (c == 'head cut' and s['start'] <= t0 + 0.03)
            and not (c == 'tail cut' and s['end'] >= t1 - 0.03)]


def pick_found(segs, k, dur, taken, t0=0.0, t1=1e9, zmin=2.0):
    """the best k clean segments inside the length range (module docstring: find); returns (picks, rejected)"""
    loud = max((s['peak_dbfs'] for s in segs), default=-99)
    picks, rej = [], []
    for s in sorted(segs, key=lambda s: -s['score']):
        why = ('edge cut' if real_cut(s, t0, t1) else f"length {s['duration']:.3f} s" if not dur[0] <= s['duration'] <= dur[1]
               else f"z {s['z']:+.2f} < {zmin}" if s['z'] < zmin else f"{loud - s['peak_dbfs']:.0f} dB under the loudest"
               if s['peak_dbfs'] < loud - 18 else
               'overlaps a kept trim' if any(s['start'] < b and s['end'] > a for a, b in taken) else '')
        if why or len(picks) >= k:
            rej.append({'start': s['start'], 'end': s['end'], 'z': s['z'], 'why': why or 'enough kept'})
            continue
        picks.append(s)
        taken.append((s['start'], s['end']))
    return picks, rej


def export(x, a, b, path, fade_out):
    """trim [a, b) samples of stereo x, 2 ms fade in, fade_out s out, peak to -1 dBFS; returns gain dB"""
    import soundfile as sf
    y = np.array(x[a:b], np.float64)
    nf = min(len(y) // 4, int(0.002 * SR))
    y[:nf] *= np.linspace(0, 1, nf)[:, None]
    nf2 = max(1, min(len(y) // 3, int(fade_out * SR)))
    y[-nf2:] *= np.linspace(1, 0, nf2)[:, None]
    pk = float(np.abs(y).max()) + 1e-12
    g = 10 ** (-1 / 20) / pk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y * g, SR, subtype='PCM_24')
    return round(20 * np.log10(g), 2)


def build_cat(name, spec, S, K, cache, log=print):
    out = {'query': spec['query'], 'dur': list(spec['dur']), 'note': spec.get('note', ''), 'forge': [], 'bench': [],
           'rejected': []}
    for split in refsets.SPLITS:
        taken = {}
        for ent in spec[split]:
            if isinstance(ent, str):
                ks = K[ent[4:]]
                src = next(s for s in S.values() if os.path.normpath(s['path']) == os.path.normpath(ks['source']['path']))
                t0, t1 = ks['source']['trim_s']
                x = audio.load(src['path'])
                a, b = int(t0 * SR), min(len(x), int(t1 * SR))
                capped = b - a > spec['dur'][1] * SR
                b = a + int(spec['dur'][1] * SR) if capped else b
                fo = 0.25 if capped else ks['processing'].get('fade_out_s') or 0.03
                fname = f"kit-{ent[4:]}.wav"
                how = {'how': 'kit trim' + (' (capped)' if capped else ''), 'kit_sound': ent[4:],
                       'kit_match_query': ks['evidence'].get('match_query'), 'kit_match_z': ks['evidence'].get('match_z'),
                       'kit_p_real': ks['evidence'].get('realness_p')}
                cuts = [(a, b, fo, fname, how)]
                taken.setdefault(src['id'], []).append((a / SR, b / SR))
            else:
                sid, k, mode = ent[:3]
                opt = ent[3] if len(ent) > 3 else {}
                src = S[sid]
                x = audio.load(src['path'])
                cuts = []
                if mode == 'find':
                    segs = find_cached(cache, src['path'], spec['query'], max(8, 3 * k), opt.get('start'), opt.get('end'))
                    picks, rej = pick_found(segs, k, spec['dur'], taken.setdefault(sid, []), opt.get('start', 0.0),
                                            opt.get('end', len(x) / SR), spec.get('zmin', 2.0))
                    out['rejected'] += [{'source_id': sid, 'split': split, **r} for r in rej]
                    for q, s in enumerate(picks):
                        cuts.append((int(s['start'] * SR), min(len(x), int(s['end'] * SR)),
                                     min(0.03, s['duration'] / 4), f'{sid}_{q + 1}.wav',
                                     {'how': 'find', 'find_z': s['z'], 'find_verdict': s['verdict'], 'find_kind': s['kind'],
                                      'head_db': s['head_db'], 'tail_db': s['tail_db'],
                                      **({'edge': s['cut'] + ' at the file edge (kept)'} if s['cut'] else {})}))
                    if not picks:
                        log(f'  {name}/{split}: nothing usable in {sid} ({len(segs)} candidates)')
                else:
                    o0 = int(opt.get('start', 0) * SR)
                    o1 = int(opt['end'] * SR) if 'end' in opt else len(x)
                    a, b, capped = cut_whole(audio.to_mono(x[o0:o1]), spec['dur'][1], rise=mode == 'rise')
                    a, b = a + o0, b + o0
                    if (b - a) / SR < spec['dur'][0]:
                        out['rejected'].append({'source_id': sid, 'split': split, 'start': a / SR, 'end': b / SR,
                                                'why': f'{mode} cut {(b - a) / SR:.3f} s shorter than the range'})
                    else:
                        cuts.append((a, b, 0.25 if capped or mode == 'rise' else 0.03, f'{sid}_1.wav',
                                     {'how': mode + (' (capped)' if capped else '')}))
            for a, b, fo, fname, how in cuts:
                path = f'{OUT}/{name}/{split}/{fname}'
                g = export(x, a, b, path, fo)
                y = audio.load(path, cache=False)
                out[split].append({'file': path, 'source_id': src['id'], 'family': family(src), 'title': src['title'],
                                   'author': src['author'], 'licence': src['license'], 'page_url': src['page_url'],
                                   'source_path': src['path'], 'trim_s': [round(a / SR, 3), round(b / SR, 3)],
                                   'fade_in_s': 0.002, 'fade_out_s': round(fo, 3), 'gain_db': g,
                                   'evidence': {**how, 't40_ms': t40_ms(y), 'length_s': round(len(y) / SR, 3)}})
    files = [r['file'] for s in refsets.SPLITS for r in out[s]]
    if files:
        from earshot import api
        rk = {r['path'].replace(chr(92), '/'): r for r in api.rank(files, spec['query'])['results']}
        for s in refsets.SPLITS:
            for r in out[s]:
                h = rk.get(r['file'])
                if h:
                    r['evidence'].update(rank_z=h['z'], rank_verdict=h['verdict'], sounds_most_like=h['sounds_most_like'])
    return out


def registry(man):
    reg = {'version': 1, 'built_by': 'eval/forge/build_refsets.py', 'root': OUT,
           'categories': {c: {s: [r['file'] for r in m[s]] for s in refsets.SPLITS} for c, m in man.items()}}
    for c, m in man.items():
        reg['categories'][c]['note'] = m['note']
    fam = {r['file']: r['family'] for m in man.values() for s in refsets.SPLITS for r in m[s]}
    return reg, refsets.check_split(reg, fam)


def plan(S, K):
    seen = {s: {} for s in refsets.SPLITS}
    for c, spec in CATS.items():
        for split in refsets.SPLITS:
            ids = []
            for ent in spec[split]:
                if isinstance(ent, str):
                    p = K[ent[4:]]['source']['path']
                    sid = next(s['id'] for s in S.values() if os.path.normpath(s['path']) == os.path.normpath(p))
                else:
                    sid = ent[0]
                ids.append(sid)
                seen[split].setdefault(family(S[sid]), set()).add(c)
            print(f'{c:12s} {split:5s} {len(set(ids)):2d} sources: {" ".join(sorted(set(ids)))}')
    bad = sorted(set(seen['forge']) & set(seen['bench']))
    print('family leaks:', bad or 'none')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cats', nargs='*', help='only these categories (merged into the existing manifest)')
    ap.add_argument('--plan', action='store_true', help='print the source plan and the leak check, no audio work')
    ap.add_argument('--registry', help='registry path (default earshot.forge.refsets.registry_path())')
    a = ap.parse_args(argv)
    S, K = sources(), kit_sounds()
    if plan(S, K) or a.plan:
        return
    mpath = OUT + '/manifest.json'
    man = {}
    if os.path.exists(mpath):
        with open(mpath, encoding='utf-8') as fh:
            man = json.load(fh)
    cache = load_cache()
    for c in a.cats or list(CATS):
        print(f'{c} ...', flush=True)
        for s in refsets.SPLITS:
            d = f'{OUT}/{c}/{s}'
            if os.path.isdir(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
        man[c] = build_cat(c, CATS[c], S, K, cache)
        print(f"  forge {len(man[c]['forge'])}, bench {len(man[c]['bench'])}, rejected {len(man[c]['rejected'])}")
        with open(mpath, 'w', encoding='utf-8') as fh:
            json.dump(man, fh, indent=1)
    man = {c: man[c] for c in CATS if c in man}
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump(man, fh, indent=1)
    reg, bad = registry(man)
    if bad:
        raise SystemExit('split leak, registry not written: ' + '; '.join(bad))
    print('registry:', refsets.write(reg, a.registry))


if __name__ == '__main__':
    main()
