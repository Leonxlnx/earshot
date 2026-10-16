"""The reference films of the pro motion-SFX study (third-party, local only: never committed or redistributed).

ref_1..ref_8 are anonymous ids. The mapping to the local video files lives only in the local data folder
(out/refs_pro/films_map.json: {"src": folder, "films": {"ref_1": "file.mp4", ...}}), so nothing committed names a film.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.environ.get('EARSHOT_REFS_PRO', os.path.join(REPO, 'out', 'refs_pro')).replace(chr(92), '/')   # local data
WORK = OUT + '/_work'
_MAP = OUT + '/films_map.json'


def _load():
    if not os.path.exists(_MAP):
        return {'src': '', 'films': {}}
    with open(_MAP, encoding='utf-8') as fh:
        return json.load(fh)


_M = _load()
SRC = _M.get('src', '')
VIDEOS = _M.get('films', {})


def path(ref):
    return os.path.join(SRC, VIDEOS[ref]).replace(chr(92), '/')


def wav(ref):
    return f'{WORK}/audio/{ref}.wav'
