"""skills/real-sound: the scripts run end to end on a synthetic project (test signals only, never committed media).

The project is the design template (templates/design_fragment.json) + the cue template, with synthetic stand-ins at
every file path, so these tests also prove the templates render. A second tiny project carries deliberate defects
(a cut into a playing sine, a masked tick, a partly sounded class of action, a trim on digital silence, a riser that
ends long before its hit) that qa.py must catch. The end-to-end tests start the scripts as subprocesses (mix, qa,
arrange: tens of seconds each on this laptop) and are marked slow; the fast ones run in-process.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

SR = 48000
SKILL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'skills', 'real-sound')
SCRIPTS = os.path.join(SKILL, 'scripts')
sys.path.insert(0, SCRIPTS)
import licences  # noqa: E402
import variant  # noqa: E402

RNG = np.random.RandomState(7)


def run(script, *args, cwd=None):
    p = subprocess.run([sys.executable, '-I', os.path.join(SCRIPTS, script), *args], capture_output=True, text=True,
                       cwd=cwd, timeout=600)
    return p.returncode, p.stdout + p.stderr


def hits(dur, times, sr=SR, tau=0.012, amp=0.5, floor=1e-4):
    n = int(dur * sr)
    x = RNG.randn(n) * floor
    for t in times:
        i = int(t * sr)
        k = np.arange(min(n - i, int(0.25 * sr)))
        x[i:i + len(k)] += RNG.randn(len(k)) * np.exp(-k / sr / tau) * amp
    return x.astype(np.float32)


def write(root, rel, x, sr=SR):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    sf.write(p, x, sr, format='WAV', subtype='PCM_16')  # WAV bytes; ffmpeg probes content, not the extension
    return p


def music(dur, sr=22050, start=0.5, bpm=120):
    """a steady test track: a low thump on every downbeat, a tick on every beat, a different chord per bar"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    beat = 60 / bpm
    for k in range(int((dur - start) / beat)):
        i = int((start + k * beat) * sr)
        m = np.arange(min(n - i, int(0.3 * sr)))
        if k % 4 == 0:
            x[i:i + len(m)] += 0.6 * np.sin(2 * np.pi * 55 * m / sr) * np.exp(-m / sr / 0.12)
        x[i:i + len(m)] += 0.15 * RNG.randn(len(m)) * np.exp(-m / sr / 0.01)
    for b in range(int((dur - start) / (4 * beat))):
        a0, a1 = int((start + 4 * beat * b) * sr), int((start + 4 * beat * (b + 1)) * sr)
        for f in ([220, 277, 330], [196, 247, 294], [175, 220, 262], [165, 208, 247])[b % 4]:
            x[a0:a1] += 0.05 * np.sin(2 * np.pi * f * t[a0:a1])
    return x.astype(np.float32), sr


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    """the design template + cue template with synthetic recordings at every path, mixed once"""
    root = str(tmp_path_factory.mktemp('rs_project'))
    write(root, 'audio/assets/ui/mk-0000_mouse-click.wav', hits(0.3, [0.1, 0.185]))
    write(root, 'audio/assets/keyboard/fs-000001_mech-spacebar.ogg', hits(1.0, [0.5], tau=0.03))
    write(root, 'audio/assets/motion/fs-000002_stick-swish.ogg',
          (RNG.randn(int(0.6 * SR)) * np.hanning(int(0.6 * SR)) * 0.3).astype(np.float32))
    write(root, 'audio/assets/ui/fs-000003_cork-pop.ogg', hits(0.3, [0.03], tau=0.02))
    write(root, 'audio/assets/motion/fs-000004_cymbal-roll.ogg',
          (RNG.randn(int(2.5 * SR)) * np.linspace(0.02, 0.3, int(2.5 * SR))).astype(np.float32))
    write(root, 'audio/assets/impacts/mk-0005_deep-drum-impact.wav', hits(2.0, [0.0], tau=0.25, amp=0.8))
    write(root, 'audio/assets/impacts/fs-000006_wood-slam.ogg', hits(0.5, [0.02], tau=0.04))
    tt = np.arange(3 * SR) / SR
    write(root, 'audio/assets/magic/fs-000007_celesta-E5.ogg',
          (0.4 * np.sin(2 * np.pi * 659.3 * tt) * np.exp(-tt / 0.5)).astype(np.float32))
    write(root, 'audio/assets/keyboard/fs-000008_mech-typing.ogg', hits(10.0, np.arange(0.6, 9.5, 0.17), tau=0.01))
    write(root, 'audio/assets/ambience/fs-000009_quiet-office-roomtone.ogg',
          (RNG.randn(46 * 8000) * 0.01).astype(np.float32), sr=8000)
    m, msr = music(40.0)
    write(root, 'audio/assets/music/sb-0000_bright-launch-track.mp3', m, sr=msr)
    os.makedirs(os.path.join(root, 'out'), exist_ok=True)
    shutil.copy(os.path.join(SKILL, 'templates', 'cues.json'), os.path.join(root, 'out', 'cues.json'))
    os.makedirs(os.path.join(root, 'audio'), exist_ok=True)
    shutil.copy(os.path.join(SKILL, 'templates', 'design_fragment.json'), os.path.join(root, 'audio', 'design.json'))
    code, out = run('mix.py', '--root', root, '--design', 'audio/design.json', '--trace', 'out/qa/adds.json')
    assert code == 0, out
    return root


@pytest.mark.slow
def test_template_mix_renders(project):
    rep = json.load(open(os.path.join(project, 'out/qa/mix_report.json')))
    cues = json.load(open(os.path.join(project, 'out/cues.json')))
    assert rep['unsounded_events'] == []
    assert len(rep['placements']) == 24  # 23 rule placements on 19 events (layered) + 1 free placement
    x, sr = sf.read(os.path.join(project, 'out/audio/soundtrack.wav'))
    assert sr == SR and abs(len(x) / SR - cues['total'] / cues['fps']) < 1e-3
    master = json.load(open(os.path.join(project, 'out/qa/master.json')))
    assert abs(master['lufs'] - (-14)) <= 0.5
    trace = json.load(open(os.path.join(project, 'out/qa/adds.json')))
    assert len([a for a in trace if 'sample' in a]) == len(rep['placements'])
    assert 'audio/assets/music/sb-0000_bright-launch-track.mp3' in rep['files_used']


@pytest.mark.slow
def test_solo_and_dump(project, tmp_path):
    out = str(tmp_path / 'solo' / 'design.wav')
    code, txt = run('mix.py', '--root', project, '--design', 'audio/design.json', '--only', 'design', '--out', out)
    assert code == 0, txt
    assert os.path.exists(out)  # the unmastered solo mix is written to --out itself
    code, txt = run('mix.py', '--root', project, '--design', 'audio/design.json', '--dump', str(tmp_path / 'dump'))
    assert code == 0, txt
    rows = json.load(open(tmp_path / 'dump' / '_samples.json'))
    names = {r.get('name') for r in rows}
    assert 'ui.press' in names and any(n and n.startswith('kb.type#') for n in names)


@pytest.mark.slow
def test_qa_on_template(project):
    code, txt = run('qa.py', '--root', project, '--design', 'audio/design.json', '--out', 'out/qa/qa.json')
    res = json.load(open(os.path.join(project, 'out/qa/qa.json')))
    for k in ('clicks', 'loudness', 'coverage', 'edges', 'audibility', 'harshness', 'sync', 'density'):
        assert k in res['sections'] and res['sections'][k]['summary']
    assert res['sections']['coverage']['status'] == 'ok'
    assert res['sections']['loudness']['lufs'] == pytest.approx(-14, abs=0.5)
    assert txt.startswith('QA ')


@pytest.mark.slow
def test_qa_catches_defects(tmp_path):
    root = str(tmp_path)
    t = np.arange(3 * SR) / SR
    write(root, 'a/sine.wav', (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32))
    write(root, 'a/tick.wav', hits(0.3, [0.05], amp=0.5))
    write(root, 'a/silence.wav', np.zeros(SR, np.float32))
    ramp = RNG.randn(int(2.6 * SR)) * np.r_[np.linspace(0.01, 0.4, 2 * SR), np.full(int(0.6 * SR), 0.004)]
    write(root, 'a/riser.wav', ramp.astype(np.float32))  # its body ends 0.6 s before the file does
    bed = signal.sosfilt(signal.butter(2, 2000, 'low', fs=SR, output='sos'), RNG.randn(4 * SR)) * 0.3
    write(root, 'a/bed.wav', bed.astype(np.float32))  # a dark bed: masks the quiet tick, leaves the click audible
    cues = {'fps': 60, 'total': 240, 'events': [{'f': 60, 'kind': 'pop', 'id': 't:cut'},
                                               {'f': 120, 'kind': 'tick', 'id': 't:quiet'},
                                               {'f': 150, 'kind': 'land', 'id': 't:step:0'},
                                               {'f': 180, 'kind': 'land', 'id': 't:step:1'},
                                               {'f': 200, 'kind': 'boop', 'id': 't:silent'},
                                               {'f': 228, 'kind': 'riser', 'id': 't:riser'}]}
    design = {'duration': 4.0, 'samples': {
        'cut': {'file': 'a/sine.wav', 'start': 1.0003, 'end': 1.3, 'fade_in': 0, 'fade_out': 0.05},
        'quiet': {'file': 'a/tick.wav', 'norm_peak_db': -60},
        'step': {'file': 'a/tick.wav', 'norm_peak_db': -6},
        'sil': {'file': 'a/silence.wav', 'start': 0.2, 'end': 0.5},
        'rise': {'file': 'a/riser.wav', 'fade_out': 0.01}},
        'pools': {'cut': ['cut'], 'quiet': ['quiet'], 'step': ['step'], 'sil': ['sil'], 'rise': ['rise']},
        'rules': [{'name': 'cut', 'match': {'kind': 'pop'}, 'pool': 'cut', 'mode': 'fixed'},
                  {'name': 'quiet', 'match': {'kind': 'tick'}, 'pool': 'quiet', 'mode': 'fixed'},
                  {'name': 'step', 'match': {'kind': 'land', 'to': 2.9}, 'pool': 'step', 'mode': 'fixed'},
                  {'name': 'sil', 'match': {'kind': 'boop'}, 'pool': 'sil', 'mode': 'fixed'},
                  {'name': 'rise', 'match': {'kind': 'riser'}, 'pool': 'rise', 'mode': 'fixed', 'align': 'end',
                   'gain_db': -12}],
        'ambience': [{'file': 'a/bed.wav', 'at': 0, 'until': 4.0, 'fade_in': 0.2, 'fade_out': 0.2}],
        'master': {'lufs': -14}}
    os.makedirs(os.path.join(root, 'out'), exist_ok=True)
    json.dump(cues, open(os.path.join(root, 'out/cues.json'), 'w'))
    json.dump(design, open(os.path.join(root, 'd.json'), 'w'))
    code, txt = run('mix.py', '--root', root, '--design', 'd.json', '--trace', 'out/qa/adds.json')
    assert code == 0, txt
    code, txt = run('qa.py', '--root', root, '--design', 'd.json', '--out', 'out/qa/qa.json')
    s = json.load(open(os.path.join(root, 'out/qa/qa.json')))['sections']
    cut = [c for c in s['clicks']['items'] if c['owners'] and c['owners'][0].get('what') == 'cut']
    assert cut and cut[0]['owners'][0]['where'] == 'start'  # the cut into a playing sine clicks, at its start
    assert s['clicks']['status'] == 'fail' and code == 1
    edge = [it for it in s['edges']['items'] if it['name'] == 'cut']
    assert edge and any('already playing' in m for m in edge[0]['issues'])
    quiet = [r for r in s['audibility']['events'] if r['label'] == 't:quiet']
    assert quiet and quiet[0]['verdict'] == 'masked'
    assert s['coverage']['status'] == 'warn' and s['coverage']['partial'][0]['id'] == 't:step'
    silent = [r for r in s['audibility']['events'] if r['label'] == 't:silent']
    assert silent and silent[0]['verdict'] == 'silent' and s['audibility']['status'] == 'fail'
    assert [q['id'] for q in s['coverage']['placed_but_silent']] == ['t:silent']
    riser = [r for r in s['sync']['items'] if r['label'] == 't:riser']
    assert riser and riser[0]['test'] == 'end' and riser[0]['end_ms'] < -500  # the riser placed by its end is checked
    code, txt = run('qa.py', '--root', root, '--design', 'd.json', '--report', 'out/nope.json')
    assert code == 2 and 'mix report not found' in txt and 'Traceback' not in txt


def test_licence_classes():
    c = licences.classify
    assert c('CC0 1.0')['status'] == 'allowed' and c('Creative Commons 0')['key'] == 'cc0'
    assert c('CC BY 4.0')['key'] == 'by4' and c('Attribution 4.0')['attribution']
    assert c('Mixkit Sound Effects Free License')['key'] == 'mixkit-sfx'
    assert c('Mixkit Stock Music Free License')['key'] == 'mixkit-music'
    assert c('Pixabay Content License')['status'] == 'allowed'
    for bad in ('CC BY-NC 4.0', 'CC BY-ND 4.0', 'CC BY-SA 4.0', 'Sampling+ 1.0', 'personal use only',
                'DISPUTED: chosic says CC BY 3.0; artist says custom'):
        assert c(bad)['status'] == 'forbidden', bad
    assert c('Royalty free')['status'] == 'unknown' and c('')['status'] == 'unknown'
    assert c('Creative Commons Attribution 4.0 International (copyrighted by the author)')['key'] == 'by4'
    assert c('All rights reserved')['status'] == 'forbidden' and c('CC BY 4.0, all rights reserved')['status'] == 'unknown'


def manifests(root):
    m = os.path.join(root, 'audio/assets/manifests')
    os.makedirs(m, exist_ok=True)
    ents = [
        {'id': 'fs-111', 'category': 'ui', 'file': 'a/tick.wav', 'title': 'Tick', 'author': 'alice', 'license': 'CC BY 4.0',
         'page_url': 'https://freesound.org/people/alice/sounds/111/',
         'download_url': 'https://cdn.freesound.org/previews/0/111_42-hq.ogg',
         'attribution': '"Tick" by alice (https://freesound.org/s/111/), licensed under CC BY 4.0'},
        {'id': 'fs-222', 'category': 'ui', 'file': 'a/sine.wav', 'title': 'Tone', 'author': 'bob', 'license': 'CC BY-NC 4.0',
         'page_url': 'https://freesound.org/people/bob/sounds/223/'},
        {'id': 'mk-5', 'category': 'amb', 'file': 'a/bed.wav', 'title': 'Bed', 'author': 'Mixkit',
         'license': 'Mixkit Sound Effects Free License', 'page_url': 'https://mixkit.co/free-sound-effects/'},
    ]
    json.dump(ents, open(os.path.join(m, 'ui.json'), 'w'))
    json.dump([dict(ents[2], category='dup')], open(os.path.join(m, 'zz.json'), 'w'))
    return m


def test_audit_and_credits(tmp_path):
    root = str(tmp_path)
    write(root, 'a/tick.wav', hits(0.3, [0.05]))
    write(root, 'a/sine.wav', hits(0.3, [0.05]))
    write(root, 'a/bed.wav', hits(1.0, [0.1]))
    m = manifests(root)
    os.makedirs(os.path.join(root, 'out/qa'), exist_ok=True)
    json.dump({'files_used': ['a/tick.wav', 'a/bed.wav', 'a/sine.wav']}, open(os.path.join(root, 'out/qa/mix_report.json'), 'w'))
    code, txt = run('audit.py', '--root', root, '--report', 'out/qa/mix_report.json')
    rows = {r['id']: r for r in json.load(open(os.path.join(m, 'audit.json'))) if not r.get('duplicate_of')}
    assert code == 1
    assert rows['fs-222']['severity'] == 'blocker' and any('forbidden' in p for p in rows['fs-222']['problems'])
    assert any('page is sound 223' in w for w in rows['fs-222']['warnings'])  # the id does not match the page
    assert any('no licence record' in p for p in rows['fs-111']['problems'])
    assert rows['mk-5']['in_scope'] and rows['mk-5']['file_check']['decodes_clean']
    assert sum(1 for r in json.load(open(os.path.join(m, 'audit.json'))) if r.get('duplicate_of')) == 1
    agent = json.load(open(os.path.join(m, 'audit.json')))
    for r in agent:  # the auditor records what it read on each page; audit.py keeps it on the next run
        if r['id'] in ('fs-111', 'mk-5'):
            r['licence_seen'] = 'page shows Attribution 4.0' if r['id'] == 'fs-111' else 'Mixkit Sound Effects Free License (read)'
            r['licence_confirmed'] = True
    json.dump(agent, open(os.path.join(m, 'audit.json'), 'w'))
    code, txt = run('audit.py', '--root', root, '--report', 'out/qa/mix_report.json')
    rows = {r['id']: r for r in json.load(open(os.path.join(m, 'audit.json'))) if not r.get('duplicate_of')}
    assert rows['fs-111']['severity'] == 'ok' and rows['mk-5']['severity'] == 'ok' and rows['fs-222']['severity'] == 'blocker'
    code, txt = run('credits.py', '--root', root, '--audit', 'audio/assets/manifests/audit.json')
    assert code == 1 and 'PROBLEM' in txt  # the NC recording is used
    md = open(os.path.join(root, 'CREDITS.md'), encoding='utf-8').read()
    assert 'by alice' in md and 'creativecommons.org/licenses/by/4.0/' in md  # the licence link is completed
    desc = open(os.path.join(root, 'out/CREDITS_description.txt'), encoding='utf-8').read()
    assert 'Sound effects (CC BY):' in desc and 'Mixkit' in desc
    assert 'were modified' in desc  # CC BY 4.0 s.3(a)(1)(B): the description says the works were changed
    assert os.path.exists(os.path.join(m, 'audit.prev.json'))  # the earlier audit is kept before it is rewritten


def test_audit_inputs(tmp_path):
    code, txt = run('audit.py', '--root', str(tmp_path), '--manifests', 'nope', '--out', 'a.json')
    assert code == 2 and 'manifest folder not found' in txt
    m = tmp_path / 'm'
    m.mkdir()
    write(str(tmp_path), 'a/x.wav', hits(0.3, [0.05]))
    json.dump([{'id': 'fs-1', 'file': 'a/x.wav', 'title': 'x', 'author': 'a', 'license': 'CC0 1.0',
                'page_url': 'https://freesound.org/people/a/sounds/1/'},
               {'id': 'fs-2', 'file': 'a/x.wav', 'title': 'x', 'author': 'a', 'license': 'CC BY-SA 4.0'}], open(m / 's.json', 'w'))
    code, txt = run('audit.py', '--root', str(tmp_path), '--manifests', 'm', '--out', 'a.json', '--decode', 'none')
    rows = json.load(open(tmp_path / 'a.json'))
    assert code == 1 and any('forbidden licence' in p for p in rows[0]['problems'])  # the same file, a second licence


@pytest.mark.slow
def test_arrange_grid(tmp_path):
    x, sr = music(24.0)
    p = str(tmp_path / 'track.wav')
    sf.write(p, x, sr)
    code, txt = run('arrange.py', p, '--bars', '6', '--drop-film-bar', '4', '--out', str(tmp_path / 'a.json'))
    assert code == 0, txt
    r = json.load(open(tmp_path / 'a.json'))
    assert r['drop_source_bar'] == r['drop_candidates'][0] and r['map'][3] == r['drop_source_bar']
    assert abs(r['bpm'] - 120) < 0.5
    ph = (r['bars'][0]['t'] - 0.5) % 2.0
    assert min(ph, 2.0 - ph) < 0.02, r['bars'][0]  # downbeats on the thumps (0.5 + 2k s), within 20 ms
    assert len(r['map']) == 6 and r['music']['segments'][0]['at'] == 0.0


@pytest.mark.skipif(shutil.which('node') is None or shutil.which('ffmpeg') is None, reason='needs node and ffmpeg')
def test_mux(tmp_path):
    v, a, o = str(tmp_path / 'v.mp4'), str(tmp_path / 'a.wav'), str(tmp_path / 'o.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=160x90:rate=30:duration=2', '-c:v',
                    'mpeg4', v], check=True)
    sf.write(a, hits(2.5, [0.5, 1.0]), SR)
    p = subprocess.run(['node', os.path.join(SCRIPTS, 'mux.mjs'), '--video', v, '--audio', a, '--out', o],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert 'longer than the picture' in p.stdout
    j = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,codec_name:format=duration',
                                   '-of', 'json', o], capture_output=True, text=True).stdout)
    kinds = {s['codec_type']: s['codec_name'] for s in j['streams']}
    assert kinds == {'video': 'mpeg4', 'audio': 'aac'} and abs(float(j['format']['duration']) - 2.0) < 0.1


def test_templates_parse():
    tdir = os.path.join(SKILL, 'templates')
    for f in os.listdir(tdir):
        if f.endswith('.json'):
            json.load(open(os.path.join(tdir, f), encoding='utf-8'))
    arc = json.load(open(os.path.join(tdir, 'arc.json')))
    assert len(arc['arc']) >= arc['drop_film_bar']
    top = json.load(open(os.path.join(tdir, 'sounddesign.json')))
    assert top['master']['lufs'] == -14 and all(i.startswith('audio/design/') for i in top['include'])


def test_variant_patch_keeps_format():
    text = ('{\n  "rules": [\n    {"name": "a",   "gain_db": -3.0, "pitch": [1, 1]},\n'
            '    {"name": "b",   "gain_db": -6},\n    {"name": "c",   "gain_db": -9}\n  ],\n'
            '  "samples": {"s1": {"file": "x.ogg", "end": 0.250}}\n}\n')
    new = json.loads(text)
    del new['rules'][1]
    new['rules'].append({'name': 'd'})  # a delete plus an add keeps the length: must not patch pairwise
    new['rules'][0]['gain_db'] = -4
    new['samples']['s1']['eq'] = {'lp': 9000}
    out = variant.patch_text(text, new)
    assert json.loads(out) == new
    assert '{"name": "a",   "gain_db": -4, "pitch": [1, 1]}' in out and '"end": 0.250' in out  # spacing and 0.250 kept
    assert '{"name": "c",   "gain_db": -9}' in out and '"b"' not in out
    assert variant.change('x', 0.3, 0.3).endswith('unchanged (0.3)')


@pytest.mark.slow
def test_variant_ops(project, tmp_path):
    top = {'duration': 12.0, 'cues': 'out/cues.json', 'include': ['audio/design.json'], 'master': {'lufs': -14}}
    json.dump(top, open(os.path.join(project, 'audio/top.json'), 'w'))
    ops = [{'f': 'design', 'rule': 'fx.hit', 'set': {'gain_db': -5}}, {'f': 'design', 'sample': 'fx.pop', 'eq': {'lp': 9000}},
           {'sd': 'master.lufs', 'value': -15}]
    json.dump(ops, open(tmp_path / 'ops.json', 'w'))
    out = str(tmp_path / 'v1')
    code, txt = run('variant.py', '--root', project, '--design', 'audio/top.json', '--ops', str(tmp_path / 'ops.json'), '--out', out)
    assert code == 0 and '3 change(s)' in txt, txt
    v = json.load(open(os.path.join(out, 'design', 'design.json')))
    assert [r for r in v['rules'] if r['name'] == 'fx.hit'][0]['gain_db'] == -5
    assert v['samples']['fx.pop']['eq'] == {'hp': 250, 'peq': [[3200, 1.2, -2]], 'lp': 9000}
    assert json.load(open(os.path.join(out, 'sounddesign.json')))['master']['lufs'] == -15
    orig = json.load(open(os.path.join(project, 'audio/design.json')))
    assert [r for r in orig['rules'] if r['name'] == 'fx.hit'][0]['gain_db'] == -2  # the design itself is untouched
    json.dump([{'f': 'design', 'rule': 'no.such.rule', 'set': {'gain_db': 0}}], open(tmp_path / 'bad.json', 'w'))
    code, txt = run('variant.py', '--root', project, '--design', 'audio/top.json', '--ops', str(tmp_path / 'bad.json'),
                    '--out', str(tmp_path / 'v2'))
    assert code != 0 and 'nothing written' in txt and not os.path.exists(tmp_path / 'v2')
