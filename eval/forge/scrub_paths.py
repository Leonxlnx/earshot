"""Remove machine-specific absolute paths from the committed forge benchmark results (and any other text / JSON file).

    python eval/forge/scrub_paths.py                    # eval/forge/results_*.json and results_*.md, in place
    python eval/forge/scrub_paths.py --check            # exit 1 if a machine path is left (nothing is written)
    python eval/forge/scrub_paths.py out/x/bench.json   # these files instead
    from scrub_paths import scrub; json.dump(scrub(res), fh, indent=1)    # inside a script that writes results

eval/forge/bench.py records where every reference came from: the registry file and each bench / forge ref, as absolute
paths of the machine it ran on. Those are not needed to read the results and do not belong in a public repo, so they
are rewritten relative to named roots (longest root first):
  <refs_pro>/...       the pro-study trims (EARSHOT_REFS_PRO, default <repo>/out/refs_pro; eval/forge/pro/)
  <refs>/...           the licensed reference library the registry points into (<repo>/out/refs, written by
                       eval/forge/build_refsets.py; plus the common folder of the registry's own paths)
  <EARSHOT_HOME>/...   earshot's cache root (the registry lives at <EARSHOT_HOME>/refsets.json)
  repo-relative        anything else inside this checkout (out/forge_v2/click/final/click_01.wav)
Ref file names stay (they are the evidence of which recording a row measured); only the machine part goes. The JSON is
written back with bench.py's own layout (indent 1), so a scrubbed file differs from the original only on those lines.
--check also flags any other absolute drive / home path (X:/..., /Users/..., /home/...) left in the files.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE)).replace(chr(92), '/')
_REST = r'((?:[\\/]+[^\s"\'`;,)\]<>|*?]*)?)'      # the rest of a path after its root (no spaces: roots have none)
LEFT = re.compile(r'(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]{1,2}[^\s"\'`;,)\]]*|/(?:Users|home)/[^\s"\'`;,)\]]+)')


def _home():
    if os.environ.get('EARSHOT_HOME'):
        return os.path.abspath(os.environ['EARSHOT_HOME'])
    try:
        sys.path.insert(0, REPO)
        from earshot import config
        return config.HOME
    except Exception:
        return os.path.join(REPO, '.cache')


def _registry_root():
    """the common folder of the registry's ref paths (None without a registry)"""
    try:
        sys.path.insert(0, REPO)
        from earshot.forge import refsets
        paths = [p for c in refsets.load().get('categories', {}).values() for s in refsets.SPLITS for p in c.get(s, [])]
        if paths:
            root = os.path.commonpath([os.path.dirname(os.path.dirname(os.path.dirname(p))) for p in paths])
            return root if len(root) > 3 else None
    except Exception:
        return None


def roots():
    """[(absolute root, placeholder or '' for repo-relative)], longest root first"""
    pro = os.environ.get('EARSHOT_REFS_PRO') or os.path.join(REPO, 'out', 'refs_pro')
    cand = [(pro, '<refs_pro>'), (os.path.join(REPO, 'out', 'refs'), '<refs>'), (_home(), '<EARSHOT_HOME>'),
            (REPO, '')]
    reg = _registry_root()
    inside = reg and os.path.abspath(reg).replace(chr(92), '/').lower().startswith(REPO.lower() + '/')
    if reg and not inside:          # a library elsewhere (inside the checkout, out/refs and repo-relative cover it)
        cand.append((reg, '<refs>'))
    out, seen = [], set()
    for r, ph in cand:
        r = os.path.abspath(r).replace(chr(92), '/').rstrip('/')
        if r.lower() not in seen:
            seen.add(r.lower())
            out.append((r, ph))
    return sorted(out, key=lambda q: -len(q[0]))


def _pattern(root):
    parts = [re.escape(p) for p in root.split('/') if p]
    sep = r'[\\/]+'
    lead = sep if root.startswith('/') else ''
    return re.compile(r'(?<![A-Za-z0-9])' + lead + sep.join(parts) + r'(?![A-Za-z0-9_.-])' + _REST, re.I)


_COMPILED = None


def _compiled():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = [(_pattern(r), ph) for r, ph in roots()]
    return _COMPILED


def scrub_text(s):
    """a string with every known machine root replaced (forward slashes in what follows it)"""
    if not isinstance(s, str) or not s:
        return s
    for pat, ph in _compiled():
        def sub(m, ph=ph):
            rest = re.sub(r'[\\/]+', '/', m.group(1) or '')
            if ph:
                return ph + (rest or '')
            return rest.lstrip('/') or '.'
        s = pat.sub(sub, s)
    return s


def scrub(obj):
    """a JSON-like object with scrub_text() applied to every string (keys too)"""
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        return {scrub_text(k) if isinstance(k, str) else k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def leftovers(text):
    """absolute machine paths still in a text ([] = clean)"""
    return sorted(set(LEFT.findall(text)))


def scrub_file(path, write=True):
    """(changed, leftovers) for one .json (rewritten with indent 1, as bench.py writes it) or text file"""
    with open(path, encoding='utf-8') as fh:
        old = fh.read()
    if path.lower().endswith('.json'):
        new = json.dumps(scrub(json.loads(old)), indent=1)
        if old.endswith('\n'):
            new += '\n'
    else:
        new = scrub_text(old)
    if write and new != old:
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            fh.write(new)
    return new != old, leftovers(new)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('files', nargs='*', help='files to scrub (default eval/forge/results_*.json / .md)')
    ap.add_argument('--check', action='store_true', help='only report; exit 1 when a machine path would change or '
                    'is left')
    a = ap.parse_args(argv)
    files = a.files or sorted(glob.glob(os.path.join(HERE, 'results_*.json')) +
                              glob.glob(os.path.join(HERE, 'results_*.md')))
    bad = 0
    for f in files:
        changed, left = scrub_file(f, write=not a.check)
        rel = os.path.relpath(f, REPO).replace(chr(92), '/') if os.path.abspath(f).startswith(os.path.abspath(REPO)) \
            else os.path.basename(f)
        state = ('would change' if a.check else 'scrubbed') if changed else 'clean'
        print(f'{rel}: {state}' + (f'; left: {", ".join(left[:3])}' if left else ''))
        bad += bool(left) or (a.check and changed)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
