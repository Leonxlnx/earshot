"""Where the evaluation data lives on your machine.

The evaluation data (licensed recordings, a film project with its stems and mix report, and 7 third-party launch
films) is not redistributed. Each script reads the locations from environment variables or from a gitignored
`eval/local_paths.json`; see eval/README.md. Environment variables win over the file, the file wins over the defaults.

    EARSHOT_EVAL_FILM    film project root: audio/assets/, audio/_retired_synth/, out/audio/, out/qa/mix_report.json,
                         out/cues_all.json, out/Clawd_1080p60.mp4 ...           (default eval/data/film)
    EARSHOT_EVAL_ASSETS  library of real recordings: <category>/, _rejected/<category>/, manifests/
                                                                               (default <film>/audio/assets)
    EARSHOT_EVAL_REFS    folder with the reference launch films ref_1.mp4 .. ref_7.mp4   (default eval/data/refs)
    EARSHOT_EVAL_SYNTH   folder of the synth engine scripts (synth.py, score.py)         (default eval/data/synth)

local_paths.json may also map single reference films to any file: {"refs": {"ref_1": "/films/a.mp4", ...}}.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, 'local_paths.json')
try:
    with open(LOCAL, encoding='utf-8') as fh:
        _CFG = json.load(fh)
except FileNotFoundError:
    _CFG = {}


def _get(key, env, default):
    return (os.environ.get(env) or _CFG.get(key) or default).replace('\\', '/').rstrip('/')


FILM = _get('film_project', 'EARSHOT_EVAL_FILM', os.path.join(HERE, 'data', 'film'))
ASSETS = _get('assets', 'EARSHOT_EVAL_ASSETS', FILM + '/audio/assets')
REFS_DIR = _get('refs_dir', 'EARSHOT_EVAL_REFS', os.path.join(HERE, 'data', 'refs'))
SYNTH = _get('synth_scripts', 'EARSHOT_EVAL_SYNTH', os.path.join(HERE, 'data', 'synth'))
REF_IDS = tuple(f'ref_{i}' for i in range(1, 8))


def ref(rid):
    """path of reference launch film `rid` ('ref_1' .. 'ref_7')"""
    if not os.environ.get('EARSHOT_EVAL_REFS') and rid in (_CFG.get('refs') or {}):
        return _CFG['refs'][rid].replace('\\', '/')
    return f'{REFS_DIR}/{rid}.mp4'


def anon(path):
    """a reference film's local path -> 'refs/ref_N.mp4' (for anything written into committed results)"""
    p = str(path).replace('\\', '/')
    for rid in REF_IDS:
        if p == ref(rid):
            return f'refs/{rid}.mp4'
    return path
