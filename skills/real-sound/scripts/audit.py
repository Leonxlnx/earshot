#!/usr/bin/env python
"""Licence + file audit of every recording in the manifests: is each one allowed, verified, credited and intact?

  python -I <skill>/scripts/audit.py [--root .] [--manifests audio/assets/manifests] [--report out/qa/mix_report.json]
                                     [--out audio/assets/manifests/audit.json] [--table out/qa/audit.md]
                                     [--fetch] [--fetch-all] [--decode scope|all|none] [--json]

Per manifest entry (references/licences.md has the policy; templates/manifest_entry.json the fields):
  licence      classified from the manifest's "license": allowed (CC0, public domain, CC BY 2.0-4.0, Mixkit free
               licences, Pixabay) / forbidden (NC, ND, SA, Sampling+, personal, custom, editorial, disputed) / unknown
  verified     the agent's own reading of the licence on the item's page: "licence_seen" (+ "licence_confirmed") merged
               from an existing audit.json (the --out file); the script never verifies a licence itself, it only checks
               that what the agent saw classifies the same as the manifest
  attribution  CC BY items need a ready-to-paste "attribution" credit line (and author, title, page_url)
  urls         Freesound / Mixkit ids agree between the entry id, the page and the download URL
  file         exists, ffprobe codec / rate / channels / duration, decodes without errors (ffmpeg -f null; in-scope
               recordings unless --decode all), duration matches the manifest's analysis; with --fetch the bytes are
               compared with download_url (sequential, 1.5 s apart, in memory, never executed; only for recordings
               used in the film unless --fetch-all)
in_scope = used in the film (report.files_used) when --report is given, else the existing audit's value, else true.
Severity: blocker (in scope and forbidden / unknown / unverified licence, missing or broken file, CC BY without a
credit line), warning, ok. Exit code 1 when an in-scope recording has a blocker.
Writes audit.json (a list, one record per entry; agent notes kept under "agent_notes") and a Markdown table.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import licences as L  # noqa: E402

TOOL = 'real-sound audit.py v2'
NEEDED = ('id', 'file', 'title', 'author', 'page_url', 'license')
NO_SCRIPTS = ('pixabay.com',)  # sites that block scripted downloads: never fetch
UA = 'real-sound-licence-audit/1.0 (polite single-file check; contact: project owner)'


def say(s):
    print(str(s).encode('ascii', 'replace').decode())


COMPUTED = ('path', 'exists', 'codec', 'sample_rate', 'channels', 'kbps', 'duration_s', 'video', 'decodes_clean',
            'max_volume_db', 'bytes_match_download_url', 'fetch_note')
AUDIT_KEYS = ('id', 'manifest', 'need', 'rank', 'in_scope', 'severity', 'licence_in_manifest', 'licence', 'licence_status',
              'licence_seen', 'licence_confirmed', 'attribution_required', 'attribution_ok', 'attribution_in_manifest',
              'file_ok', 'file_check', 'earlier_file_check', 'problems', 'warnings', 'agent_notes', 'audit_tool', 'problem',
              'fix', 'duplicate_of')
SENT = r'(?:[^.]|\.(?=\d))'  # inside one sentence ("4.0" does not end it)
READ_NOTE = re.compile(SENT + r'*\blicen[cs]e' + SENT + r'{0,40}\b(read|verified|checked|seen|confirmed)\b' + SENT + '*',
                       re.I)


def ffprobe(path):
    """first audio stream's codec / rate / channels / bit rate, the duration, and any real video stream (not cover art)"""
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,codec_name,sample_rate,channels,'
                        'bit_rate:stream_disposition=attached_pic:format=duration,bit_rate', '-of', 'json', path],
                       capture_output=True, text=True, timeout=120)
    j = json.loads(p.stdout or '{}')
    st = j.get('streams') or []
    s = next((x for x in st if x.get('codec_type') == 'audio'), {})
    f = j.get('format') or {}
    brs = [int(b) for b in (s.get('bit_rate'), f.get('bit_rate')) if str(b).isdigit() and 0 < int(b) < 10 ** 7]
    video = [x.get('codec_name') for x in st if x.get('codec_type') == 'video'
             and not (x.get('disposition') or {}).get('attached_pic')]
    return {'codec': s.get('codec_name'), 'sample_rate': int(s.get('sample_rate') or 0), 'channels': int(s.get('channels') or 0),
            'kbps': round(brs[0] / 1000) if brs else None, 'duration_s': round(float(f.get('duration') or 0), 3),
            **({'video': video[0]} if video else {})}


def decodes_clean(path):
    """(decodes without errors, first error line, max volume dBFS) from one full ffmpeg decode of the audio stream"""
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'level+info', '-i', path, '-map', '0:a:0',
                        '-af', 'volumedetect', '-f', 'null', '-'], capture_output=True, timeout=600)
    lines = p.stderr.decode(errors='replace').splitlines()
    err = [ln for ln in lines if '[error]' in ln or '[fatal]' in ln]
    vol = [float(m.group(1)) for ln in lines for m in [re.search(r'max_volume: (-?[\d.]+|-inf) dB', ln)] if m
           and m.group(1) != '-inf'] or [None]
    return p.returncode == 0 and not err, err[0][:200] if err else None, vol[-1]


def url_checks(e):
    """id consistency for the two sites whose URLs encode it"""
    out = []
    page, dl, eid = e.get('page_url') or '', e.get('download_url') or '', str(e.get('id') or '')
    m = re.search(r'freesound\.org/people/([^/]+)/sounds/(\d+)', page)
    if m:
        sid = m.group(2)
        if re.match(r'fs-\d+', eid) and eid[3:] != sid:
            out.append(f'entry id {eid} but the Freesound page is sound {sid}')
        d = re.search(r'previews/(\d+)/(\d+)_(\d+)-(hq|lq)\.(ogg|mp3)', dl)
        if d:
            if d.group(2) != sid:
                out.append(f'download_url is sound {d.group(2)}, the page is sound {sid}')
            if d.group(1) != str(int(sid) // 1000):
                out.append(f'preview folder {d.group(1)} should be {int(sid) // 1000}')
            if d.group(4) == 'lq':
                out.append('low-quality Freesound preview (-lq): use the -hq preview')
        if e.get('author') and m.group(1).lower() != str(e['author']).lower():
            out.append(f'author "{e["author"]}" differs from the page owner "{m.group(1)}"')
    d = re.search(r'assets\.mixkit\.co/active_storage/sfx/(\d+)/(\d+)\.(wav|mp3)', dl)
    if d:
        if d.group(1) != d.group(2):
            out.append('Mixkit download_url ids disagree')
        if re.match(r'mk-\d+', eid) and eid[3:] != d.group(1):
            out.append(f'entry id {eid} but the Mixkit download is item {d.group(1)}')
    return out


def fetch_match(url, path):
    """True / False when the bytes at url equal the local file; None + reason when not checked"""
    host = re.sub(r'^[a-z]+://', '', url.lower()).split('/')[0]
    if any(host.endswith(h) for h in NO_SCRIPTS):
        return None, f'{host} blocks scripted downloads: compare by hand'
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(200 * 1024 * 1024 + 1)
    except Exception as ex:  # noqa: BLE001
        return None, f'fetch failed: {str(ex)[:160]}'
    local = open(path, 'rb').read()
    return hashlib.sha1(data).hexdigest() == hashlib.sha1(local).hexdigest(), None


def agent_fields(old):
    """what the agent wrote into an earlier audit record (licence reading, notes, hand-recorded file checks such as a
    byte comparison); script findings are recomputed, earlier evidence is kept under earlier_file_check"""
    if not old:
        return {}
    keep = {k: old[k] for k in ('licence_seen', 'licence_confirmed') if k in old}
    fc = old.get('file_check') or {}
    earlier = dict(old.get('earlier_file_check') or {})
    if old.get('audit_tool'):
        notes = old.get('agent_notes') or []
        earlier.update({k: fc[k] for k in ('bytes_match_download_url', 'fetch_note') if k in fc})
    else:  # a hand-written audit: its warnings / problem / fix are the agent's notes, its file checks earlier evidence
        notes = list(old.get('warnings') or []) + [old[k] for k in ('problem', 'fix') if old.get(k)]
        earlier.update({k: v for k, v in fc.items() if k not in COMPUTED or k in ('bytes_match_download_url', 'fetch_note')})
        keep.update({k: v for k, v in old.items() if k not in AUDIT_KEYS})
    if earlier:
        keep['earlier_file_check'] = earlier
    if notes:
        keep['agent_notes'] = notes
    if 'in_scope' in old:
        keep['_in_scope'] = old['in_scope']
    return keep


def audit_entry(e, root, used, old, decode='scope'):
    path = os.path.join(root, e['file']) if not os.path.isabs(e['file']) else e['file']
    f = L.rel(e['file'], root)
    lic = L.classify(e.get('license'))
    ag = agent_fields(old)
    in_scope = (f.lower() in used) if used is not None else bool(ag.get('_in_scope', True))
    problems, warnings = [], []
    for k in NEEDED:
        if not e.get(k):
            (problems if k in ('file', 'license') else warnings).append(f'missing "{k}"')
    if lic['status'] == 'forbidden':
        problems.append(f'licence forbidden: {lic["reason"]}')
    elif lic['status'] == 'unknown':
        problems.append(f'licence unknown: {lic["reason"]}')
    seen = ag.get('licence_seen')
    confirmed = bool(ag.get('licence_confirmed')) and bool(seen)
    if not seen:
        m = READ_NOTE.search(str(e.get('notes') or ''))
        problems.append('no licence record in audit.json' + (
            f' (the manifest notes say: "{m.group(0).strip()[:90]}"): copy that reading into "licence_seen" and set '
            '"licence_confirmed"' if m else ': read the licence on the item page and record "licence_seen"'))
    else:
        sk = L.classify(seen)['key']
        if sk not in ('unknown', 'none', lic['key']):
            warnings.append(f'licence_seen reads as {sk} but the manifest says {lic["key"]}')
        if not ag.get('licence_confirmed'):
            problems.append('licence_seen recorded but licence_confirmed is not true')
    attr_ok = True
    if lic['attribution']:
        attr_ok = bool(e.get('attribution')) and bool(e.get('author')) and bool(e.get('page_url'))
        if not attr_ok:
            problems.append('CC BY needs "attribution" (a ready-to-paste credit line), "author" and "page_url"')
        else:
            at = e['attribution']
            if str(e['author']).split(' (')[0].lower() not in at.lower():
                warnings.append('the attribution line does not name the author')
            if not re.search(r'cc[ -]?by|attribution|creativecommons\.org', at, re.I):
                warnings.append('the attribution line does not name the licence (credits.py adds the licence link)')
    warnings += url_checks(e)
    fc = {'path': f, 'exists': os.path.isfile(path)}
    if not fc['exists']:
        problems.append('file missing')
    else:
        try:
            fc.update(ffprobe(path))
        except Exception as ex:  # noqa: BLE001
            problems.append(f'ffprobe failed: {str(ex)[:120]}')
        if fc.get('video'):
            warnings.append(f"the file carries a {fc['video']} video stream: extract the audio to its own file and make "
                            'sure the licence covers the audio')
        if decode == 'all' or (decode == 'scope' and in_scope):
            ok, err, vol = decodes_clean(path)
            fc['decodes_clean'], fc['max_volume_db'] = ok, vol
            if not ok:
                problems.append(f'decode errors: {err}')
            if vol is not None and vol <= -90:
                problems.append(f'the recording is digital silence (max {vol} dB)')
            elif vol is not None and vol < -60:
                warnings.append(f'the recording is near-silent (max {vol} dB): the wrong file?')
        an = (e.get('analysis') or {}).get('duration')
        if an and fc.get('duration_s') and abs(float(an) - fc['duration_s']) > 0.05:
            warnings.append(f'duration {fc["duration_s"]} s differs from the manifest analysis ({an} s): file replaced?')
        if fc.get('sample_rate') and fc['sample_rate'] < 44100:
            warnings.append(f'sample rate {fc["sample_rate"]} Hz (dull above {fc["sample_rate"] // 2000} kHz)')
        if fc.get('codec') == 'mp3' and fc.get('kbps') and fc['kbps'] < 192:
            warnings.append(f'MP3 at {fc["kbps"]} kbps: look for a lossless or higher-rate source')
    r = {'id': e.get('id'), 'manifest': e['_manifest'], 'need': e.get('need'), 'rank': e.get('rank'),
         'in_scope': in_scope, 'severity': None,
         'licence_in_manifest': e.get('license'), 'licence': lic['key'], 'licence_status': lic['status'],
         'licence_seen': seen, 'licence_confirmed': confirmed, 'attribution_required': lic['attribution'],
         'attribution_ok': attr_ok, 'attribution_in_manifest': e.get('attribution'),
         'file_ok': fc['exists'] and fc.get('decodes_clean', True) is not False, 'file_check': fc,
         **({'earlier_file_check': ag['earlier_file_check']} if ag.get('earlier_file_check') else {}),
         'problems': problems, 'warnings': warnings, 'agent_notes': ag.get('agent_notes', []), 'audit_tool': TOOL,
         **{k: v for k, v in ag.items() if k not in AUDIT_KEYS and not k.startswith('_')}}
    r['severity'] = severity(r)
    return r


def severity(r):
    sev = 'blocker' if r['problems'] else 'warning' if r['warnings'] else 'ok'
    return sev if r['in_scope'] else 'info' if sev != 'ok' else 'ok'


def audit(root='.', manifests='audio/assets/manifests', report=None, existing=None, fetch=False, fetch_all=False,
          decode='scope'):
    """audit every manifest entry; returns {'summary', 'counts', 'entries', 'notes'}"""
    mdir = manifests if os.path.isabs(manifests) else os.path.join(root, manifests)
    if not os.path.isdir(mdir):
        raise FileNotFoundError(f'manifest folder not found: {mdir} (pass --manifests)')
    entries, notes = L.load_manifests(mdir)
    if not entries:
        raise FileNotFoundError(f'no manifest entries in {mdir}' + (f' ({"; ".join(notes)})' if notes else ''))
    used = None
    if report:
        rp = report if os.path.isabs(report) else os.path.join(root, report)
        used = {L.rel(u, root).lower() for u in json.load(open(rp, encoding='utf-8')).get('files_used', [])}
    old = {}
    existing = existing and (existing if os.path.isabs(existing) else os.path.join(root, existing))
    if existing and os.path.exists(existing):
        for r in json.load(open(existing, encoding='utf-8')):
            p = L.rel(((r.get('file_check') or {}).get('path') or ''), root).lower()
            old[(str(r.get('id')), p)] = old[(str(r.get('id')), '')] = r
    rows, first = [], {}
    for e in entries:
        f = L.rel(e['file'], root).lower()
        o = old.get((str(e.get('id')), f)) or old.get((str(e.get('id')), ''))
        if f in first:  # the same recording in another manifest: audited once, under its first entry
            r0, e0 = first[f]
            note = f'duplicate of {e0.get("id")} in {os.path.basename(e0["_manifest"])}'
            lic = L.classify(e.get('license'))
            if lic['status'] != 'allowed':
                r0['problems'].append(f'{os.path.basename(e["_manifest"])} ({e.get("id")}) lists this file under a '
                                      f'{lic["status"]} licence "{e.get("license")}" ({lic["reason"]}): one recording has '
                                      'one licence; fix the manifest or reject the file')
            elif lic['key'] != r0['licence']:
                r0['warnings'].append(f'{os.path.basename(e["_manifest"])} lists this file as {lic["key"]}, not '
                                      f'{r0["licence"]}')
            r0['severity'] = severity(r0)
            rows.append({**r0, 'id': e.get('id'), 'manifest': e['_manifest'], 'need': e.get('need'), 'rank': e.get('rank'),
                         'in_scope': False, 'severity': 'info', 'duplicate_of': e0.get('id'), 'problems': [],
                         'warnings': [note]})
            continue
        r = audit_entry(e, root, used, o, decode)
        first[f] = (r, e)
        rows.append(r)
    if fetch or fetch_all:
        n = 0
        for r, e in zip(rows, entries):
            if not (fetch_all or r['in_scope']) or not e.get('download_url') or not r['file_check']['exists']:
                continue
            if n:
                time.sleep(1.5)
            n += 1
            ok, why = fetch_match(e['download_url'], os.path.join(root, e['file']))
            r['file_check']['bytes_match_download_url'] = ok
            if why:
                r['file_check']['fetch_note'] = why
            if ok is False:
                r['warnings'].append('local bytes differ from download_url (re-encoded or edited file?)')
                r['severity'] = severity(r)
    if used is not None:
        listed = {L.rel(e['file'], root).lower() for e in entries}
        for u in sorted(used - listed):
            rows.append({'id': None, 'manifest': None, 'in_scope': True, 'severity': 'blocker', 'licence': 'none',
                         'licence_status': 'unknown', 'file_check': {'path': u}, 'problems': ['used in the film but in no manifest'],
                         'warnings': [], 'agent_notes': [], 'audit_tool': TOOL})
    sc = [r for r in rows if r['in_scope']]
    cnt = {k: sum(1 for r in sc if r['severity'] == k) for k in ('ok', 'warning', 'blocker')}
    by_lic = {}
    for r in sc:
        by_lic[r['licence']] = by_lic.get(r['licence'], 0) + 1
    scope = 'used in the film' if used is not None else 'in scope'
    summary = (f'{len(rows)} manifest entries, {len(sc)} {scope}: {cnt["ok"]} ok, {cnt["warning"]} with warnings, '
               f'{cnt["blocker"]} blocked. Licences {scope}: ' + ', '.join(f'{k} {v}' for k, v in sorted(by_lic.items())) + '.')
    n_notes = sum(1 for r in sc if r.get('agent_notes'))
    if n_notes:
        summary += f' {n_notes} of them carry auditor notes (credit forms, restrictions): read agent_notes.'
    now = sum(1 for r in sc if (r.get('file_check') or {}).get('bytes_match_download_url'))
    before = sum(1 for r in sc if not (r.get('file_check') or {}).get('bytes_match_download_url')
                 and (r.get('earlier_file_check') or {}).get('bytes_match_download_url'))
    if now:
        summary += (f' Bytes match download_url: {now} checked in this run' + (f', {before} more recorded by an earlier '
                    'audit (earlier_file_check)' if before else '') + '.')
    elif before:
        summary += f' Bytes match download_url for {before} according to an earlier audit (not re-checked: --fetch).'
    if cnt['blocker']:
        summary += ' Fix every blocker before publishing (see the table).'
    return {'summary': summary, 'counts': {**cnt, 'entries': len(rows), 'in_scope': len(sc)}, 'licences_in_scope': by_lic,
            'notes': notes, 'entries': rows}


def table(rows):
    order = {'blocker': 0, 'warning': 1, 'info': 2, 'ok': 3}
    out = ['| id | in film | severity | licence | verified | credit | file | issues |', '|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (not r['in_scope'], order.get(r['severity'], 9), str(r.get('id')))):
        fc = r.get('file_check') or {}
        fdesc = (f"{fc.get('codec')} {fc.get('sample_rate', 0) // 1000 if fc.get('sample_rate') else '?'}k "
                 f"{fc.get('channels')}ch {fc.get('duration_s')}s") if fc.get('exists') else 'MISSING'
        issues = '; '.join(r.get('problems', []) + r.get('warnings', []))[:300].replace('|', '/')
        out.append(f"| {r.get('id')} | {'yes' if r['in_scope'] else 'no'} | {r['severity']} | {r.get('licence')} | "
                   f"{'yes' if r.get('licence_confirmed') else 'NO'} | "
                   f"{'n/a' if not r.get('attribution_required') else 'ok' if r.get('attribution_ok') else 'MISSING'} | "
                   f"{fdesc} | {issues} |")
    return '\n'.join(out) + '\n'


def backup(out):
    """before overwriting an audit: keep the previous one as <name>.prev.json, and a hand-written one (no audit_tool
    field) once as <name>.handwritten.json; returns the paths written"""
    if not os.path.exists(out):
        return []
    stem = os.path.splitext(out)[0]
    done = [stem + '.prev.json']
    shutil.copy(out, done[0])
    try:
        rows = json.load(open(out, encoding='utf-8'))
        hand = isinstance(rows, list) and rows and not any(isinstance(r, dict) and r.get('audit_tool') for r in rows)
    except ValueError:
        hand = True
    if hand and not os.path.exists(stem + '.handwritten.json'):
        shutil.copy(out, stem + '.handwritten.json')
        done.append(stem + '.handwritten.json')
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default='.', help='project root: relative paths resolve against it')
    ap.add_argument('--manifests', default='audio/assets/manifests')
    ap.add_argument('--report', help='mix_report.json: in_scope = the files the film uses')
    ap.add_argument('--out', help='audit JSON to write (default <manifests>/audit.json); agent fields are merged from it')
    ap.add_argument('--existing', help='read agent fields from this audit instead of --out')
    ap.add_argument('--table', help='Markdown table (default: --out with .md)')
    ap.add_argument('--fetch', action='store_true', help='compare bytes with download_url for recordings in the film')
    ap.add_argument('--fetch-all', action='store_true', help='... for every entry')
    ap.add_argument('--decode', choices=('scope', 'all', 'none'), default='scope',
                    help='full decode check for in-scope recordings (default), all, or none')
    ap.add_argument('--json', action='store_true', help='print the full result as JSON')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    mdir = a.manifests if os.path.isabs(a.manifests) else os.path.join(root, a.manifests)
    out = a.out or os.path.join(mdir, 'audit.json')
    out = os.path.normpath(out if os.path.isabs(out) else os.path.join(root, out))
    try:
        res = audit(root, a.manifests, a.report, a.existing or out, a.fetch, a.fetch_all, a.decode)
    except FileNotFoundError as e:
        say(f'audit.py: {e}')
        return 2
    os.makedirs(os.path.dirname(out), exist_ok=True)
    backups = backup(out)
    json.dump(res['entries'], open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    tab = a.table or os.path.splitext(out)[0] + '.md'
    tab = os.path.normpath(tab if os.path.isabs(tab) else os.path.join(root, tab))
    open(tab, 'w', encoding='utf-8').write('# Licence and file audit\n\n' + res['summary'] + '\n\n' + table(res['entries']))
    if a.json:
        print(json.dumps(res, indent=1, ensure_ascii=True))
    else:
        say(res['summary'])
        for n in res['notes']:
            say(f'  note: {n}')
        groups = {}
        for r in res['entries']:
            if r['in_scope'] and r['severity'] != 'ok':
                for m in r['problems'] + r['warnings']:
                    groups.setdefault((m in r['problems'], m), []).append(str(r.get('id')))
        for (blk, m), ids in sorted(groups.items(), key=lambda kv: (not kv[0][0], -len(kv[1]))):
            say(f"  {'BLOCKER' if blk else 'warning'} x{len(ids)}: {m}  [{', '.join(ids[:8])}{' ...' if len(ids) > 8 else ''}]")
        say(f'wrote {out} and {tab}' + (f' (previous audit kept as {", ".join(backups)})' if backups else ''))
    return 1 if res['counts']['blocker'] else 0


if __name__ == '__main__':
    sys.exit(main())
