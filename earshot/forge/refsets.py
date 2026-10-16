"""Reference sets for the forge: per category, real / pro recordings split by source file into 'forge' and 'bench'.

    from earshot.forge.refsets import refset, categories, note
    refs = refset('whoosh_fast')                  # the forge may optimise against these (Critic refs=, ToneRef)
    held = refset('whoosh_fast', split='bench')   # held out: only eval/forge/bench.py reads these, never the forge
    categories()                                  # ['click', 'downlifter', 'glitch', 'hover_tick', 'impact', ...]
    note('sub_drop')                              # why a set is thin or empty ('' when nothing to say)
    held_out(paths)                               # which of these must never be forge refs (bench / bench-source kit)

The recordings are licensed third-party sounds and are not redistributed, so the sets live in a local registry, a JSON
file: EARSHOT_REFSETS, else <EARSHOT_HOME>/refsets.json. On the dev machine eval/forge/build_refsets.py writes it from
the licensed library (trims located with earshot find / rank, trim + fade + peak gain only). Public users bring their
own refs: lay them out as <root>/<category>/{forge,bench}/*.wav and run `write(scan(root))`, or write the JSON by hand:
    {"version": 1, "categories": {"whoosh_fast": {"forge": ["a.wav", ...], "bench": ["b.wav", ...], "note": ""}}}
Relative paths are read relative to the registry file.

The split rule that makes the benchmark mean something: no source recording (nor a sister take of it) is in both
splits of any category, so a forge tuned against 'forge' refs is measured against recordings it never heard.
CATEGORIES lists the motion-design SFX categories the benchmark knows, with what each one means.
"""
import json
import os

SPLITS = ('forge', 'bench')
CATEGORIES = {
    'whoosh_fast': 'a short crisp pass-by whoosh, a stick or rod swung through air (0.2-0.8 s)',
    'whoosh_big': 'a big, deep, slower whoosh / swoop (0.8-3 s)',
    'swish': 'a light thin quick swish, a thrown object (0.15-0.5 s)',
    'swipe': 'a soft cloth / paper slide or swipe (0.25-1 s)',
    'riser': 'a swell building up to a peak (1.5-6 s)',
    'downlifter': 'a falling sweep that sinks away after a hit (1-4 s)',
    'impact': 'a deep cinematic boom hit with a long tail (1-8 s)',
    'sub_drop': 'a sub-bass pitch drop felt under a hit (1-4 s)',
    'click': 'a small plastic UI button click (30-350 ms)',
    'tap': 'a soft finger tap on glass / a hard surface (30-300 ms)',
    'toggle': 'a switch flipped: a crisp click, often two-part (50-450 ms)',
    'hover_tick': 'a tiny soft tick for hovers and scroll detents (15-150 ms)',
    'pop': 'a short round pop: bubble, mouth, cork (50-500 ms)',
    'notify': 'a short bright notification ding / chime (0.4-2 s)',
    'shimmer': 'a sparkling chime sweep / bell-tree shimmer (1-6 s)',
    'typing_key': 'a single mechanical keyboard key press (60-450 ms)',
    'glitch': 'a digital glitch / stutter burst (designed; 0.1-1 s)',
    'logo_sting': 'a short logo sting: a struck hit with a ringing tail (1-6 s)',
}


# kit files (the v1 recipes' REFS, refs/<group>/<name>.wav) cut from sources that are 'bench' in some category: a forge
# run must never use them (they would leak the held-out side into the search). held_out() flags them.
KIT_BENCH = ('ui_notify_02', 'whoosh_med_01', 'whoosh_med_02', 'whoosh_big_02', 'swipe_01', 'riser_02', 'impact_03',
             'shimmer_01')


class RefsetsMissing(FileNotFoundError):
    """no reference-set registry on this machine"""


def registry_path():
    """where the registry is read from: EARSHOT_REFSETS, else <EARSHOT_HOME>/refsets.json"""
    if os.environ.get('EARSHOT_REFSETS'):
        return os.path.abspath(os.environ['EARSHOT_REFSETS'])
    from .. import config
    return os.path.join(config.HOME, 'refsets.json')


def load(path=None):
    """the registry dict ({'version', 'categories': {cat: {'forge', 'bench', 'note'}}}) with absolute paths"""
    path = path or registry_path()
    if not os.path.exists(path):
        raise RefsetsMissing(
            f'no reference-set registry at {path}. The forge refs are licensed recordings and are not shipped: bring '
            'your own (lay them out as <root>/<category>/{forge,bench}/*.wav and run '
            'earshot.forge.refsets.write(scan(root))), point EARSHOT_REFSETS at a registry JSON, or on the dev '
            'machine run eval/forge/build_refsets.py.')
    with open(path, encoding='utf-8') as fh:
        reg = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    for c in reg.get('categories', {}).values():
        for s in SPLITS:
            c[s] = [p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p)).replace(chr(92), '/')
                    for p in c.get(s, [])]
    return reg


def categories(path=None):
    """category names present in the registry (sorted)"""
    return sorted(load(path).get('categories', {}))


def _cat(category, path):
    cats = load(path).get('categories', {})
    if category not in cats:
        raise KeyError(f"no reference set '{category}' in {path or registry_path()}; known: {', '.join(sorted(cats))}")
    return cats[category]


def refset(category, split='forge', path=None):
    """paths of the `split` ('forge' | 'bench') refs of `category` (may be empty: see note()); missing files raise"""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    paths = list(_cat(category, path).get(split, []))
    gone = [p for p in paths if not os.path.exists(p)]
    if gone:
        raise RefsetsMissing(f"{len(gone)} {category}/{split} ref(s) missing, e.g. {gone[0]}: rebuild the refs")
    return paths


def note(category, path=None):
    """the registry's note on a category ('' when there is none)"""
    return _cat(category, path).get('note', '')


def held_out(paths, kit=True, path=None):
    """the paths among `paths` that must not be forge refs: a file in any category's 'bench' split of the registry
    (when there is one) or, with kit=True, a kit file cut from a bench source (KIT_BENCH, matched by file stem)"""
    norm = lambda q: os.path.normcase(os.path.abspath(q))
    try:
        bench = {norm(q) for c in load(path).get('categories', {}).values() for q in c.get('bench', [])}
    except RefsetsMissing:
        bench = set()
    return [q for q in paths if norm(q) in bench
            or (kit and os.path.splitext(os.path.basename(q))[0] in KIT_BENCH)]


def scan(root, exts=('.wav', '.flac', '.ogg', '.mp3', '.aif', '.aiff')):
    """a registry dict from a folder layout <root>/<category>/{forge,bench}/<files> (absolute paths)"""
    cats = {}
    for c in sorted(os.listdir(root)):
        d = os.path.join(root, c)
        if not os.path.isdir(d) or c.startswith(('_', '.')):
            continue
        entry = {s: sorted(os.path.join(d, s, f).replace(chr(92), '/') for f in os.listdir(os.path.join(d, s))
                           if f.lower().endswith(exts)) if os.path.isdir(os.path.join(d, s)) else [] for s in SPLITS}
        if entry['forge'] or entry['bench']:
            cats[c] = {**entry, 'note': ''}
    return {'version': 1, 'categories': cats}


def check_split(reg, sources=None):
    """problems with a registry as a list of strings (empty = fine): a file in both splits of a category, or (with
    sources = {path: source_id}) a source id in forge anywhere and in bench anywhere"""
    bad = []
    seen = {s: {} for s in SPLITS}
    for c, e in reg.get('categories', {}).items():
        both = set(e.get('forge', [])) & set(e.get('bench', []))
        bad += [f'{c}: {os.path.basename(p)} in both splits' for p in sorted(both)]
        for s in SPLITS:
            for p in e.get(s, []):
                if sources and p in sources:
                    seen[s].setdefault(sources[p], set()).add(c)
    for src in sorted(set(seen['forge']) & set(seen['bench'])):
        bad.append(f"source {src} is forge in {sorted(seen['forge'][src])} and bench in {sorted(seen['bench'][src])}")
    return bad


def write(reg, path=None):
    """write a registry dict (from scan() or by hand) to `path` (default registry_path()); returns the path"""
    path = path or registry_path()
    bad = check_split(reg)
    if bad:
        raise ValueError('refuse to write a registry that leaks across splits: ' + '; '.join(bad[:3]))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(reg, fh, indent=1)
    return path
