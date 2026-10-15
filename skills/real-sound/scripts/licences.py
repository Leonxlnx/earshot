"""Licence classification and manifest loading shared by audit.py and credits.py (real-sound skill).

    import licences as L
    L.classify('CC BY 4.0')        # {'key': 'by4', 'status': 'allowed', 'name': 'CC BY 4.0', 'url': ..., 'attribution': True}
    L.classify('CC BY-NC 4.0')     # status 'forbidden', reason 'NonCommercial: ...'
    entries, notes = L.load_manifests('audio/assets/manifests')

status is 'allowed' (CC0, public domain, CC BY 2.0-4.0, Mixkit free licences, Pixabay Content License), 'forbidden'
(NonCommercial, NoDerivatives, ShareAlike, Sampling+, personal-use, custom, editorial or disputed licences) or 'unknown'
(anything else: read the item's own page and fix the manifest). The policy and how to verify a licence on each site:
references/licences.md.
"""
import glob
import json
import os
import re

CC = 'https://creativecommons.org/'
FIXED = {
    'cc0': ('CC0 1.0 (public domain dedication)', CC + 'publicdomain/zero/1.0/', False),
    'pd': ('Public Domain Mark 1.0', CC + 'publicdomain/mark/1.0/', False),
    'mixkit-sfx': ('Mixkit Sound Effects Free License', 'https://mixkit.co/license/#sfxFree', False),
    'mixkit-music': ('Mixkit Stock Music Free License', 'https://mixkit.co/license/#musicFree', False),
    'pixabay': ('Pixabay Content License', 'https://pixabay.com/service/license-summary/', False),
}
FORBID = [  # (pattern on the lower-cased licence text, reason)
    (r'disputed', 'disputed: the sources disagree about the licence'),
    (r'\bnc\b|non-?commercial', 'NonCommercial: a launch video is commercial use'),
    (r'\bnd\b|no-?deriv', 'NoDerivatives: syncing to picture, trimming and editing are derivatives'),
    (r'\bsa\b|share-?alike', 'ShareAlike: the whole film would have to be released under CC BY-SA'),
    (r'sampling', 'Sampling+ / Sampling licences are retired and restrict use'),
    (r'personal', 'personal use only'),
    (r'custom|free with credit|social media', 'a custom licence: not a standard licence the audit can trust'),
    (r'editorial', 'editorial use only'),
]
RESERVED = r'all rights reserved|\bcopyrighted\b'  # forbidden alone; next to a free licence it is ambiguous


def classify(text):
    """licence text (as written in a manifest or seen on a page) -> {key, status, name, url, attribution, reason}"""
    s = (text or '').strip().lower()
    if not s:
        return {'key': 'none', 'status': 'unknown', 'name': None, 'url': None, 'attribution': False,
                'reason': 'no licence recorded'}
    for pat, why in FORBID:
        if re.search(pat, s.replace('_', ' ')):
            return {'key': 'forbidden', 'status': 'forbidden', 'name': text, 'url': None, 'attribution': False, 'reason': why}
    if 'restricted' in s:
        return {'key': 'restricted', 'status': 'unknown', 'name': text, 'url': None, 'attribution': False,
                'reason': 'a restricted licence (e.g. Mixkit Restricted): read its terms for this use'}
    key = None
    if re.search(r'\bcc0\b|cc-0|creative commons 0|publicdomain/zero|\bzero\b|public domain dedication', s):
        key = 'cc0'
    elif 'public domain' in s or 'publicdomain/mark' in s:
        key = 'pd'
    elif 'mixkit' in s:
        key = 'mixkit-music' if 'music' in s else 'mixkit-sfx'
    elif 'pixabay' in s:
        key = 'pixabay'
    res = None
    if key:
        name, url, attr = FIXED[key]
        res = {'key': key, 'status': 'allowed', 'name': name, 'url': url, 'attribution': attr, 'reason': None}
    m = re.search(r'(?:\bby\b|attribution)[^0-9]{0,24}([1-4](?:\.[05])?)', s)
    if m and not res:
        ver = {'1': '1.0', '2': '2.0', '3': '3.0', '4': '4.0'}.get(m.group(1), m.group(1))
        if ver.startswith('1'):
            return {'key': 'by1', 'status': 'unknown', 'name': f'CC BY {ver}', 'url': None, 'attribution': True,
                    'reason': 'CC BY 1.0 is rare: confirm it on the item page'}
        res = {'key': 'by' + ver.replace('.0', '').replace('.', ''), 'status': 'allowed', 'name': f'CC BY {ver}',
               'url': f'{CC}licenses/by/{ver}/', 'attribution': True, 'reason': None}
    reserved = re.search(RESERVED, s)
    if reserved and not res:
        return {'key': 'forbidden', 'status': 'forbidden', 'name': text, 'url': None, 'attribution': False,
                'reason': 'all rights reserved'}
    if reserved and 'all rights reserved' in s:  # a free licence and "all rights reserved" contradict each other
        return {'key': 'unknown', 'status': 'unknown', 'name': text, 'url': None, 'attribution': res['attribution'],
                'reason': f'names {res["name"]} and "all rights reserved": read the item page'}
    return res or {'key': 'unknown', 'status': 'unknown', 'name': text, 'url': None, 'attribution': False,
                   'reason': 'not a licence this skill recognises: read the item page and record the exact licence'}


def rel(path, root):
    """path relative to the project root with forward slashes (case-insensitive prefix match, for Windows)"""
    p = (path or '').replace('\\', '/')
    r = os.path.abspath(root).replace('\\', '/').rstrip('/') + '/'
    return p[len(r):] if p.lower().startswith(r.lower()) else p


def load_manifests(folder, skip=('audit.json',)):
    """every manifest entry (dicts with a "file") from folder/*.json, each with "_manifest"; plus notes on skipped files.
    A manifest is a JSON list of entries, or an object holding one under "entries" / "items"."""
    entries, notes = [], []
    for m in sorted(glob.glob(os.path.join(folder, '*.json'))):
        if os.path.basename(m) in skip or os.path.basename(m).startswith('audit.'):  # audits and their backups
            continue
        try:
            d = json.load(open(m, encoding='utf-8'))
        except Exception as e:  # noqa: BLE001
            notes.append(f'{os.path.basename(m)}: not valid JSON ({e})')
            continue
        if isinstance(d, dict):
            d = d.get('entries') or d.get('items')
        if not isinstance(d, list):
            notes.append(f'{os.path.basename(m)}: not a list of entries (a decision record?), skipped')
            continue
        for e in d:
            if isinstance(e, dict) and e.get('file'):
                entries.append({**e, '_manifest': m.replace('\\', '/')})
    return entries, notes


def source_name(entry):
    """a readable source name from the entry's page_url host (Freesound.org, Mixkit, ...) or its "source" field"""
    url = entry.get('page_url') or entry.get('download_url') or ''
    host = re.sub(r'^www\.', '', re.sub(r'^[a-z]+://', '', url.lower()).split('/')[0])
    known = {'freesound.org': 'Freesound.org', 'mixkit.co': 'Mixkit', 'pixabay.com': 'Pixabay',
             'archive.org': 'the Internet Archive', 'commons.wikimedia.org': 'Wikimedia Commons', 'github.com': 'GitHub'}
    for k, v in known.items():
        if host.endswith(k):
            return v
    return host or (entry.get('source') or 'unknown source').split(' (')[0]
