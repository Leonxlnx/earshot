#!/usr/bin/env python
"""Credits for every recording audible in the film, built from the mix report and the licence manifests.

  python -I <skill>/scripts/credits.py [--root .] [--report out/qa/mix_report.json] [--manifests audio/assets/manifests]
                                       [--design audio/sounddesign.json] [--audit audio/assets/manifests/audit.json]
                                       [--out CREDITS.md] [--desc out/CREDITS_description.txt] [--title "Film name"]

Reads report.files_used (written by mix.py: every recording that is audible), matches each file to its manifest entry
and writes:
  CREDITS.md        music + every sound with title, author, link and licence (grouped by licence); a CC BY entry uses
                    the manifest's "attribution" line (the form the author asked for) when there is one
  CREDITS_description.txt   plain text to paste into the video description (CC BY credits are REQUIRED there; CC0 /
                    Mixkit / Pixabay items are listed as a courtesy)
The processing sentence (trimmed, equalised, pitch-shifted ...) and the music edit sentence come from the design when
--design is given; the source names (Freesound.org, Mixkit ...) come from the manifests. With --audit, recordings the
licence audit did not confirm are listed as problems. Both texts say that every work was modified (CC BY 4.0
s.3(a)(1)(B)). Exit code 1 when anything needs fixing before publishing, 2 when the report or manifests are missing.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import licences as L  # noqa: E402

ORDER = ['by4', 'by3', 'by25', 'by2', 'cc0', 'pd', 'mixkit-sfx', 'mixkit-music', 'pixabay']
SHORT = {'cc0': 'CC0', 'pd': 'public domain', 'mixkit-sfx': 'Mixkit free licence', 'mixkit-music': 'Mixkit free licence',
         'pixabay': 'Pixabay licence'}


def say(s):
    print(str(s).encode('ascii', 'replace').decode())


def and_join(xs):
    xs = list(xs)
    return ' and '.join([', '.join(xs[:-1]), xs[-1]]) if len(xs) > 1 else ''.join(xs)


def processing(design_path, root):
    """(sentence for the effects, sentence for the music edit, music stretched?) derived from the design"""
    if not design_path:
        return ('trimmed, layered, level-adjusted, equalised and panned for the film',
                'Edited for the film: cut to fit the picture and levels adjusted.', False)
    import mix  # read-only use of the design loader
    mix.ROOT = root
    d = mix.load_design(design_path)
    opts = list((d.get('samples') or {}).values()) + list((d.get('slices') or {}).values())
    rules, place = d.get('rules') or [], d.get('place') or []
    did = ['trimmed']
    if d.get('slices'):
        did.append('cut into single hits')
    did.append('layered' if len(rules) + len(place) > len(opts) * 0.5 or len(rules) > 1 else 'placed')
    did.append('level-adjusted')
    if any(s.get('eq') for s in opts) or any((b or {}).get('eq') for b in (d.get('buses') or {}).values()):
        did.append('equalised')
    if any(r.get('pan') not in (None, 0, 0.0) for r in rules) or any(p.get('pan') for p in place):
        did.append('panned')
    if any(s.get('pitch') or s.get('glide') for s in opts) or any(r.get('pitch') or r.get('pitch_ramp') for r in rules):
        did.append('pitch-shifted by resampling')
    fx = and_join(did)
    mu = d.get('music')
    if not mu:
        return fx + ' for the film', None, False
    segs = mu.get('segments') or []
    stretched = mu.get('stretch', 'rubberband') != 'none' and abs(mu.get('target_bpm', 120) / mu['src_bpm'] - 1) > 1e-6
    parts = ['cut and rearranged at bar lines to fit the picture' if len(segs) > 1 else 'trimmed to the picture']
    if stretched:
        parts.append(f"time-stretched from {mu['src_bpm']:g} to {mu.get('target_bpm', 120):g} BPM")
    parts.append('levels adjusted')
    if any(s.get('tail_fade') for s in segs):
        parts.append('a fade at the end')
    return fx + ' for the film', 'Edited for the film: ' + ', '.join(parts) + '.', stretched


def credits(root='.', report='out/qa/mix_report.json', manifests='audio/assets/manifests', design=None, audit=None,
            title=None):
    """build the CREDITS.md and description texts; returns {'summary', 'markdown', 'description', 'rows', 'problems'}"""
    R = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
    if not os.path.exists(R(report)):
        raise FileNotFoundError(f'mix report not found: {R(report)} (run mix.py first, or pass --report)')
    if not os.path.isdir(R(manifests)):
        raise FileNotFoundError(f'manifest folder not found: {R(manifests)} (pass --manifests)')
    rep = json.load(open(R(report), encoding='utf-8'))
    used = [L.rel(f, root) for f in rep.get('files_used', [])]
    entries, notes = L.load_manifests(R(manifests))
    by_file = {}
    for e in entries:
        by_file.setdefault(L.rel(e['file'], root).lower(), e)
    confirmed = None
    if audit and os.path.exists(R(audit)):
        confirmed = {L.rel((r.get('file_check') or {}).get('path') or '', root).lower(): r
                     for r in json.load(open(R(audit), encoding='utf-8'))}
    rows, problems = [], []
    for f in used:
        e = by_file.get(f.lower())
        if not e:
            problems.append(f'no manifest entry for {f}')
            continue
        lic = L.classify(e.get('license'))
        if lic['status'] != 'allowed':
            problems.append(f'{f}: licence "{e.get("license")}" is {lic["status"]} ({lic["reason"]})')
        if lic['attribution'] and not (e.get('author') and e.get('title') and e.get('page_url')):
            problems.append(f'{f}: CC BY needs author, title and page_url for the credit')
        if confirmed is not None:
            a = confirmed.get(f.lower())
            if not a:
                problems.append(f'{f}: the licence audit has no record of this file (run audit.py, record licence_seen)')
            elif not a.get('licence_confirmed'):
                problems.append(f'{f}: audit.json does not confirm this licence (licence_seen / licence_confirmed missing)')
        rows.append({'file': f, 'id': e.get('id'), 'title': e.get('title'), 'author': e.get('author'),
                     'url': e.get('page_url'), 'lic': lic, 'attribution': e.get('attribution'),
                     'category': e.get('category'), 'source': L.source_name(e)})
    music = [r for r in rows if r['category'] == 'music' or '/music/' in '/' + r['file'].lower()]
    sfx = [r for r in rows if r not in music]
    rank = lambda k: ORDER.index(k) if k in ORDER else len(ORDER)
    sfx.sort(key=lambda r: (rank(r['lic']['key']), (r['author'] or '').lower(), r['title'] or ''))
    fx_did, mu_did, stretched = processing(design and R(design), root)

    def line(r):
        """the author's requested credit (manifest "attribution") or a built one, completed with the source link and
        the licence link when the requested form leaves them out (CC BY asks for both)"""
        name, url = r['lic']['name'] or r['lic']['key'], r['lic']['url']
        if not (r['lic']['attribution'] and r['attribution']):
            return f'"{r["title"]}" by {r["author"]} ({r["url"]}), licensed under {name}' + (f' ({url})' if url else '')
        base = r['attribution'].strip().rstrip('.')
        if r['url'] and 'http' not in base and 'www.' not in base:
            base += f' ({r["url"]})'
        if url and 'creativecommons.org' not in base:
            base += f' ({url})'
        return base

    srcs = {}
    for r in sfx:
        srcs[r['source']] = srcs.get(r['source'], 0) + 1
    src_txt = and_join(s for s, _ in sorted(srcs.items(), key=lambda kv: -kv[1]))
    md = [f'# Credits{": " + title if title else ""}', '',
          'All credits live here and in the video description (CC BY credits are required there).', '']
    if music:
        md += ['## Music', '']
        for r in music:
            md.append(f'- {line(r)}.')
            if mu_did:
                md.append(f'  {mu_did}')
        md.append('')
    if sfx:
        md += ['## Sound effects and ambience', '',
               f'Real recordings from {src_txt or "the sources below"}, {fx_did}. Nothing in the soundtrack is '
               'synthesized.', '']
        cur = None
        for r in sfx:
            if r['lic']['key'] != cur:
                cur = r['lic']['key']
                md += ['', f'### {r["lic"]["name"] or cur}', '']
            md.append(f'- {line(r)}')
    tools = 'ffmpeg, NumPy/SciPy, soxr, librosa and pyloudnorm' + (', Rubber Band (music tempo)' if stretched else '')
    md += ['', '## Tools', '', f'Edited, mixed and mastered with the real-sound skill (mix.py, master.py) using {tools}.', '']
    if problems:
        md += ['', '## PROBLEMS (fix before publishing)', ''] + [f'- {p}' for p in problems]
    d = []
    if music:  # CC BY 4.0 s.3(a)(1)(B): say that the work was modified
        d.append('Music:')
        d += [line(r) for r in music] + ([mu_did] if mu_did else []) + ['']
    by = [r for r in sfx if r['lic']['attribution']]
    if by:
        d.append('Sound effects (CC BY):')
        d += [line(r) for r in by]
        d += [f'All sound effects and ambiences were modified: {fx_did}.', '']
    rest = sorted({r['author'] for r in sfx if not r['lic']['attribution'] and r['author']}, key=str.lower)
    if rest:
        kinds = sorted({SHORT.get(r['lic']['key'], r['lic']['key']) for r in sfx if not r['lic']['attribution']})
        d.append(f'More sounds ({", ".join(kinds)}) by: ' + ', '.join(rest))
    if sfx and not by:
        d.append(f'All sound effects and ambiences were {fx_did}.')
    n_by = sum(1 for r in rows if r['lic']['attribution'])
    summary = (f'{len(rows)} recordings credited ({len(music)} music, {len(sfx)} sound effects / ambience; {n_by} CC BY '
               f'need the credit in the video description)' + (f'; {len(problems)} PROBLEM(S) to fix before publishing'
                                                                if problems else '; no problems'))
    if any(r['lic']['attribution'] for r in music):
        summary += ('. CC BY music: put the credit in the description; if a Content ID claim appears, dispute it with '
                    'the licence link (references/licences.md)')
    return {'summary': summary, 'markdown': '\n'.join(md) + '\n', 'description': '\n'.join(d) + '\n', 'rows': [
        {k: (v['key'] if k == 'lic' else v) for k, v in r.items()} for r in rows], 'problems': problems, 'notes': notes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default='.', help='project root: relative paths resolve against it')
    ap.add_argument('--report', default='out/qa/mix_report.json')
    ap.add_argument('--manifests', default='audio/assets/manifests')
    ap.add_argument('--design', help='the design file (derives the processing and music-edit sentences)')
    ap.add_argument('--audit', help='audit.json from audit.py: unconfirmed licences become problems')
    ap.add_argument('--out', default='CREDITS.md')
    ap.add_argument('--desc', default='out/CREDITS_description.txt')
    ap.add_argument('--title', help='film name for the heading')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    try:
        res = credits(root, a.report, a.manifests, a.design, a.audit, a.title)
    except FileNotFoundError as e:
        say(f'credits.py: {e}')
        return 2
    for p, text in ((a.out, res['markdown']), (a.desc, res['description'])):
        p = p if os.path.isabs(p) else os.path.join(root, p)
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        open(p, 'w', encoding='utf-8').write(text)
    say(res['summary'])
    say(f'wrote {a.out} and {a.desc}')
    for p in res['problems']:
        say(f'PROBLEM: {p}')
    return 1 if res['problems'] else 0


if __name__ == '__main__':
    sys.exit(main())
