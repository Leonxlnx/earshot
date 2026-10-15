"""Build the realness dataset: clips of real / professionally produced audio and of code-synthesized audio.

    .venv/Scripts/python.exe eval/realness/build_data.py [--only real,rejected,films,retired,cinetic,generic,mixes,robust,augment]
    .venv/Scripts/python.exe eval/realness/build_data.py --only layers     # real beds + one code-synthesized layer

Writes mono float32 48 kHz clips to out/realness/clips/<family>/ (scratch, gitignored, never committed) and the
manifest out/realness/manifest.json: one entry per clip with its label (1 = real or pro-produced, 0 = synthesized),
family, CV group, type and provenance (source file + seconds). Deterministic: same inputs -> same clips.

Clip policy (the same for both classes, and the same as earshot.realness at inference): a source up to 6 s is one
clip; a longer source is cut into 4 s windows (evenly spaced picks for long real recordings, every window for the
retired score and the films); one-shot events are cut from long one-shot recordings with the same onset cutter for
real and synthesized sources. Near-silent windows (RMS < -60 dBFS) are skipped. Full synthesized mixes (part 'mixes'):
6 random cinetic soundtracks rendered by its score.py and 6 generic code-made mixes (generic_synth voices, Schroeder
reverb, limiter), cut into 4 s windows. The current robust / augment parts were built before the mixes (no perturbed
copies of mixes); rebuilding them would add some and change the robustness set. Part 'layers' (build_layers): real
4 s beds with one code-synthesized layer mixed in at -12 / -6 / 0 / +6 dB, label 0 (the layer-in-a-mix test).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402
from earshot import audio  # noqa: E402

SR = audio.SR
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'out', 'realness')
CLIPS = os.path.join(OUT, 'clips')
ASSETS = EP.ASSETS          # local data, see eval/README.md
CINETIC = EP.SYNTH
FILMS = {rid: EP.ref(rid) for rid in EP.REF_IDS}   # the 7 reference launch films
RETIRED = {'bed': f'{EP.FILM}/audio/_retired_synth/bed.wav',
           'melody': f'{EP.FILM}/out/audio/melody.wav', 'ambience': f'{EP.FILM}/out/audio/ambience.wav',
           **{f'stem_{s}': f'{EP.FILM}/out/audio/stems/{s}.wav'
              for s in ('arp', 'bass', 'bed', 'bells', 'drums', 'pad', 'reverb', 'sfx')}}
CATS = ('ambience', 'foley', 'garden', 'keyboard', 'magic', 'motion', 'music', 'toy')
LONG_S, WIN_S = 6.0, 4.0


def rms_db(x):
    return audio.db(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-30))


def cached(family, name):
    p = os.path.join(CLIPS, family, name + '.wav')
    return (os.path.relpath(p, OUT).replace('\\', '/'), sf.info(p).frames / SR) if os.path.exists(p) else None


def write(x, family, name):
    d = os.path.join(CLIPS, family)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + '.wav')
    sf.write(p, np.asarray(x, np.float32), SR, subtype='FLOAT')
    return os.path.relpath(p, OUT).replace('\\', '/')


def pick_windows(n, k, margin=0.0):
    """k evenly spaced 4 s window starts (seconds) inside [margin, dur - margin]"""
    dur = n / SR
    lo, hi = min(margin, max(0.0, dur - WIN_S) / 2), max(0.0, dur - WIN_S - min(margin, max(0.0, dur - WIN_S) / 2))
    if k <= 1:
        return [round((lo + hi) / 2, 3)]
    return [round(lo + (hi - lo) * i / (k - 1), 3) for i in range(k)]


def events(x, k, max_len=1.0, min_gap=0.25):
    """up to k one-shot events: onset -> next onset (<= max_len), loudest first, chronological output"""
    from earshot.features import onsets
    ons = [o for o in onsets(x) if o * SR < len(x) - 0.03 * SR]
    cand = []
    for i, o in enumerate(ons):
        if i and o - ons[i - 1] < min_gap:
            continue
        a = max(0, int((o - 0.005) * SR))
        nxt = next((v for v in ons[i + 1:] if v - o >= min_gap), None)
        b = min(len(x), int((min(o + max_len, nxt - 0.01) if nxt else o + max_len) * SR))
        if b - a > 0.03 * SR and rms_db(x[a:b]) > -50:
            cand.append((float(np.abs(x[a:b]).max()), a, b))
    cand.sort(reverse=True)
    return sorted((a, b) for _, a, b in cand[:k])


def clip_entry(cid, rel, label, family, group, typ, src, start, end):
    return {'id': cid, 'wav': rel, 'label': label, 'family': family, 'group': group, 'type': typ, 'src': src,
            'start': round(start, 3), 'end': round(end, 3)}


def build_real(kept=True):
    out = []
    for cat in CATS:
        files = audio.list_media(os.path.join(ASSETS, cat) if kept else os.path.join(ASSETS, '_rejected', cat))
        fam = ('pro-music' if cat == 'music' else 'real-asset') if kept else ('pro-music' if cat == 'music' else 'real-rejected')
        for f in files:
            x = audio.load(f, mono=True, cache=False)
            base = os.path.splitext(os.path.basename(f))[0][:40]
            tag = ('k_' if kept else 'r_') + base
            if len(x) <= LONG_S * SR:
                out.append(clip_entry(tag, write(x, fam, tag), 1, fam, cat, cat, f, 0, len(x) / SR))
                continue
            k = (4 if kept else 2) if cat == 'music' else 2 if cat == 'ambience' else 1
            for s in pick_windows(len(x), k, margin=5.0 if cat == 'music' else 0.5):
                seg = x[int(s * SR): int((s + WIN_S) * SR)]
                if rms_db(seg) > -60:
                    cid = f'{tag}@{s:.1f}'
                    out.append(clip_entry(cid, write(seg, fam, cid.replace('@', '_')), 1, fam, cat, cat, f, s, s + WIN_S))
            if cat not in ('music', 'ambience'):
                for a, b in events(x, 2):
                    cid = f'{tag}#ev{a / SR:.2f}'
                    out.append(clip_entry(cid, write(x[a:b], fam, cid.replace('#', '_')), 1, fam, cat, cat + '-event',
                                          f, a / SR, b / SR))
    return out


def build_films():
    out = []
    for name, f in FILMS.items():
        x = audio.load(f, mono=True, cache=False)
        for i, s in enumerate(np.arange(0, len(x) / SR - WIN_S + 1e-6, WIN_S)):
            seg = x[int(s * SR): int((s + WIN_S) * SR)]
            if rms_db(seg) > -60:
                cid = f'film_{name}@{s:.0f}'
                out.append(clip_entry(cid, write(seg, 'pro-film', cid.replace('@', '_')), 1, 'pro-film', 'film', name, f,
                                      float(s), float(s + WIN_S)))
    return out


def build_retired():
    out = []
    for name, f in RETIRED.items():
        x = audio.load(f, mono=True, cache=False)
        for s in np.arange(0, len(x) / SR - WIN_S + 1e-6, WIN_S):
            seg = x[int(s * SR): int((s + WIN_S) * SR)]
            if rms_db(seg) > -60:
                cid = f'retired_{name}@{s:.0f}'
                out.append(clip_entry(cid, write(seg, 'retired', cid.replace('@', '_')), 0, 'retired', 'retired', name, f,
                                      float(s), float(s + WIN_S)))
        if name in ('stem_sfx', 'stem_drums', 'melody', 'stem_bells'):
            for a, b in events(x, 25):
                cid = f'retired_{name}#ev{a / SR:.2f}'
                out.append(clip_entry(cid, write(x[a:b], 'retired', cid.replace('#', '_')), 0, 'retired', 'retired',
                                      name + '-event', f, a / SR, b / SR))
    return out


def cinetic_voice(S, kind, r, seed):
    note = lambda lo, hi: S.mtof(int(r.integers(lo, hi)))  # noqa: E731
    u = r.uniform
    if kind == 'kick':
        return S.kick(note(28, 40), u(0.4, 1.2), u(0.6, 1.4), seed)
    if kind == 'sub_boom':
        f1 = note(26, 40)
        return S.sub_boom(f1 * u(1.5, 3), f1, u(0.05, 0.3), u(1.0, 3.0), seed, u(0.4, 1.2))
    if kind == 'clap':
        return S.clap(u(0.25, 0.6), seed)
    if kind == 'hat':
        return S.hat(u(0.05, 0.3), u(0.008, 0.08), u(5000, 9000), seed)
    if kind == 'tick':
        return S.tick(note(72, 90), u(0.05, 0.1), u(0.006, 0.015), u(0.6, 1.0), seed, r.random() < 0.4)
    if kind == 'tock':
        return S.tock(u(600, 1600), u(0.08, 0.2), u(0.015, 0.05), u(0.2, 0.6), seed)
    if kind == 'key_click':
        return S.key_click(int(r.integers(0, 60)), u(0.98, 1.02), None, str(r.choice(S.KEY_STYLES)),
                           str(r.choice(['letter', 'letter', 'back', 'space', 'return'])), seed, u(0.6, 1.3),
                           0.09 if r.random() < 0.2 else None, second=0.03 if r.random() < 0.5 else None)
    if kind == 'thock':
        return S.thock(u(800, 1100), int(r.integers(0, 60)), str(r.choice(S.KEY_STYLES)), str(r.choice(['space', 'return'])), seed)
    if kind == 'ui_click':
        return S.ui_click(note(88, 100), 0.14, int(r.integers(0, 40)), 0.085 if r.random() < 0.6 else None,
                          note(76, 86) if r.random() < 0.3 else None, seed)
    if kind == 'crisp_tick':
        return S.crisp_tick(seed, u(0.03, 0.08), note(90, 100) if r.random() < 0.3 else None)
    if kind == 'snap':
        return S.snap(note(28, 40), u(0.3, 0.8), u(0.5, 1.2), seed)
    if kind == 'blip':
        return S.blip(note(70, 92), r.random() < 0.5, u(0.08, 0.2))
    if kind == 'word_blip':
        return S.word_blip(note(64, 86), u(0.036, 0.06), u(1.0, 4.0), seed)
    if kind == 'fm_bell':
        return S.fm_bell(note(72, 92), u(1.5, 3.0), u(1.0, 3.5), float(r.choice([3.5, 2.0, 1.4, 2.76])), u(0.4, 1.2))
    if kind == 'marker':
        return S.marker(u(0.15, 0.5), seed)
    if kind == 'whoosh':
        return S.whoosh(u(0.4, 1.4), u(80, 250), u(500, 1500), u(150, 400), u(0.3, 0.7), u(-1, 1), u(0.01, 0.05), seed)
    if kind == 'riser':
        return S.riser(u(1.0, 3.0), u(150, 400), u(4000, 9000), note(48, 72) if r.random() < 0.6 else None, seed)
    if kind == 'suck':
        return S.suck(u(0.2, 0.5), seed)
    if kind == 'swell':
        root = int(r.integers(55, 67))
        return S.swell([S.mtof(root + i) for i in (0, 4, 7)], u(0.6, 2.0), seed)
    if kind == 'supersaw':
        return S.supersaw(note(48, 66), u(1.0, 3.0), int(r.integers(3, 8)), u(0.05, 0.2), u(900, 3500), u(0.05, 0.4), u(0.3, 1.0), seed)
    if kind == 'pluck':
        return S.pluck(note(55, 80), u(0.3, 1.0), u(2500, 6000), u(300, 800), u(0.04, 0.15), u(0.1, 0.4))
    if kind == 'bass':
        return S.bass(note(28, 45), u(0.4, 1.5), u(1.0, 2.0), u(0.1, 0.35))
    if kind == 'drone':
        f = note(33, 48)
        return S.drone([f, f * 1.5], u(2.0, 4.0), u(300, 900), u(0.5, 2.0))
    if kind == 'keys':
        return S.keys(note(55, 76), u(1.0, 2.5), u(0.8, 2.2), u(0.4, 1.2))
    if kind == 'typing':
        bus = S.Bus(seconds=4.0)
        t, key = 0.1, int(r.integers(0, 30))
        style = str(r.choice(S.KEY_STYLES))
        while t < 3.6:
            bus.place(S.key_click(key + int(r.integers(0, 26)), u(0.985, 1.015), None, style,
                                  'space' if r.random() < 0.15 else 'letter', int(t * 1000), u(0.6, 1.2)), t)
            t += u(0.06, 0.2)
        return bus.x
    if kind == 'motif':
        voice = str(r.choice(['pluck', 'keys', 'fm_bell', 'blip', 'tick']))
        root = int(r.integers(60, 76))
        bus = S.Bus(seconds=4.0)
        for k in range(int(r.integers(3, 7))):
            f = S.mtof(root + int(r.choice([0, 2, 4, 7, 9, 12])))
            v = {'pluck': lambda: S.pluck(f), 'keys': lambda: S.keys(f, 1.2), 'fm_bell': lambda: S.fm_bell(f, 2.0),
                 'blip': lambda: S.blip(f), 'tick': lambda: S.tick(f)}[voice]()
            bus.place(v[: S.ns(3.0)] * 0.5, 0.1 + 0.45 * k)
        return S.pingpong(bus.x, 0.3, 0.35, 0.3) if r.random() < 0.5 else bus.x
    if kind == 'mix':
        return cinetic_mix(S, r, seed)
    raise ValueError(kind)


def cinetic_mix(S, r, seed):
    """four bars of a cinetic-style arrangement (drums, bass, pad, arp, rooms): like score.py output, not the film's"""
    bpm = r.uniform(96, 132)
    beat = 60 / bpm
    bus = S.Bus(seconds=4.0)
    root = int(r.integers(36, 48))
    third = 3 if r.random() < 0.5 else 4
    layers = {k: r.random() < p for k, p in (('drums', 0.8), ('bass', 0.8), ('pad', 0.7), ('arp', 0.7), ('bells', 0.3))}
    if not any(layers.values()):
        layers['pad'] = True
    for b in range(int(4.0 / beat) + 1):
        t = b * beat
        if layers['drums']:
            bus.place(S.kick(S.mtof(root - 12)) * 0.8, t)
            bus.place(S.hat(0.08, 0.012, seed=seed + b) * 0.3, t + beat / 2)
            if b % 2:
                bus.place(S.clap(seed=seed + b) * 0.4, t)
        if layers['bass'] and b % 2 == 0:
            bus.place(S.bass(S.mtof(root), beat * 1.8) * 0.5, t)
        if layers['arp']:
            for k in range(2):
                bus.place(S.pluck(S.mtof(root + 24 + int(r.choice([0, third, 7, 12])))) * 0.25, t + k * beat / 2)
    if layers['pad']:
        for i in (0, third, 7):
            bus.place(S.supersaw(S.mtof(root + 12 + i), 4.0, seed=seed + i) * 0.2, 0.0)
    if layers['bells']:
        bus.place(S.fm_bell(S.mtof(root + 36), 2.0) * 0.3, r.uniform(0, 2))
    return bus.x + S.reverb(bus.x, 'hall') * r.uniform(0.15, 0.4)


CINETIC_TYPES = ('kick', 'sub_boom', 'clap', 'hat', 'tick', 'tock', 'key_click', 'thock', 'ui_click', 'crisp_tick',
                 'snap', 'blip', 'word_blip', 'fm_bell', 'marker', 'whoosh', 'riser', 'suck', 'swell', 'supersaw',
                 'pluck', 'bass', 'drone', 'keys', 'typing', 'motif', 'mix')


def build_cinetic(per_type=10):
    sys.path.insert(0, CINETIC)
    import synth as S
    out = []
    for kind in CINETIC_TYPES:
        for k in range(per_type + (6 if kind == 'mix' else 0)):
            seed = 100 + k
            cid = f'cinetic_{kind}_{k}'
            hit = cached('cinetic', cid)
            if hit:
                out.append(clip_entry(cid, hit[0], 0, 'cinetic', 'cinetic', kind, 'synth.py', 0, hit[1]))
                continue
            r = np.random.default_rng(10007 * (CINETIC_TYPES.index(kind) + 1) + k)
            x = np.asarray(cinetic_voice(S, kind, r, seed), np.float64)
            if kind not in ('mix', 'typing', 'motif') and r.random() < 0.35:
                x = S.stereo(x) + S.reverb(S.filt(S.stereo(x), S.hp(250)), str(r.choice(['hall', 'room', 'small']))) * r.uniform(0.15, 0.5)
            x = audio.to_mono(x) if x.ndim == 2 else x
            x = x[: int(LONG_S * SR)]
            x = x / (np.abs(x).max() + 1e-12) * 10 ** (r.uniform(-20, -1) / 20)
            out.append(clip_entry(cid, write(x, 'cinetic', cid), 0, 'cinetic', 'cinetic', kind, 'synth.py', 0, len(x) / SR))
    return out


def build_generic(per_type=14):
    import generic_synth as G
    out = []
    for kind in G.TYPES:
        for k in range(per_type):
            cid = f'generic_{kind}_{k}'
            hit = cached('generic', cid)
            if hit:
                out.append(clip_entry(cid, hit[0], 0, 'generic', 'generic', kind, 'generic_synth.py', 0, hit[1]))
                continue
            x = G.render(kind, k)
            r = np.random.default_rng(5003 * k + 17)
            x = audio.to_mono(x)[: int(LONG_S * SR)]
            x = x / (np.abs(x).max() + 1e-12) * 10 ** (r.uniform(-20, -1) / 20)
            out.append(clip_entry(cid, write(x, 'generic', cid), 0, 'generic', 'generic', kind, 'generic_synth.py', 0,
                                  len(x) / SR))
    return out


CHORDS = ['', 'm', 'maj7', 'm7', '7', 'sus2', 'sus4', 'add9', 'm9', 'maj9', '6']
NOTES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
EVENT_KINDS = ['tick', 'tock', 'hit', 'land', 'pop', 'whoosh', 'suck', 'riser', 'drop', 'key', 'click', 'snap', 'swell', 'bell']


def cinetic_score(k, seconds=28.0):
    """render a random cinetic soundtrack (score.py: music from a random score.json, effects from random cue events,
    mastered) - a full synthesized film mix like the retired score, but with other cues, keys, tempi and arrangements"""
    r = np.random.default_rng(777 + k)
    d = os.path.join(OUT, 'mixes', f'cinetic_score_{k}')
    os.makedirs(d, exist_ok=True)
    wav = os.path.join(d, 'soundtrack.wav')
    if os.path.exists(wav):
        return wav
    bpm, fps = float(r.choice([96, 100, 108, 110, 116, 120, 124, 128])), 60
    total = int(seconds * fps)
    nbars = int(np.ceil(seconds / (240.0 / bpm)))
    events, f = [], 30.0
    while f < total - 60:
        kind = str(r.choice(EVENT_KINDS, p=np.r_[[0.08, 0.05, 0.08, 0.08, 0.08, 0.1, 0.03, 0.04, 0.03, 0.25, 0.08, 0.05, 0.03, 0.02]]))
        n = int(r.integers(4, 30)) if kind == 'key' else 1
        for j in range(n):
            events.append({'f': round(f, 2), 'kind': kind, 'weight': round(float(r.uniform(0.2, 1.0)), 2), 'id': f'e{len(events)}',
                           'pan': round(float(r.uniform(-0.5, 0.5)), 2)})
            f += float(r.uniform(4, 9)) if kind == 'key' else 0
        f += float(r.uniform(15, 120))
    cues = {'fps': fps, 'bpm': bpm, 'total': total, 'cue': {'mid': total // 2}, 'events': events}
    tonic = str(r.choice(NOTES))
    mode = str(r.choice(['major', 'minor', 'dorian', 'mixolydian', 'lydian']))
    styles = ['intro', 'build', 'groove', 'drive', 'breakdown', 'resolve']
    secs, b = [], 1
    while b <= nbars:
        ln = int(r.integers(1, 4))
        secs.append({'bars': [b, min(nbars, b + ln - 1)], 'style': str(r.choice(styles)), 'level': round(float(r.uniform(0.7, 1.0)), 2)})
        b += ln
    score = {'key': f'{tonic} {mode}', 'bpm': bpm, 'sections': secs,
             'chords': [NOTES[(NOTES.index(tonic) + int(r.choice([0, 2, 5, 7, 9]))) % 12] + str(r.choice(CHORDS)) for _ in range(nbars)],
             'typing': str(r.choice(['auto', 'soft', 'mechanical', 'pitched', 'muted', 'word'])),
             'drops': [{'at': '@mid', 'silence': '1/8', 'suck': 0.2}] if r.random() < 0.5 else [],
             'master': {'lufs': float(r.choice([-16, -14, -12]))}}
    json.dump(cues, open(os.path.join(d, 'cues.json'), 'w'))
    json.dump(score, open(os.path.join(d, 'score.json'), 'w'))
    subprocess.run([sys.executable, os.path.join(CINETIC, 'score.py'), '--cues', os.path.join(d, 'cues.json'), '--score',
                    os.path.join(d, 'score.json'), '--out', wav, '--quiet'], capture_output=True, cwd=d)
    return wav if os.path.exists(wav) else None


def generic_mix(k, seconds=24.0):
    """a generic code-made soundtrack: generic_synth voices layered on a timeline, a Schroeder reverb send, a tanh
    limiter and loudness normalisation (no cinetic code)"""
    import generic_synth as G
    r = np.random.default_rng(999 + k)
    n = int(seconds * SR)
    bus = np.zeros(n + 6 * SR)
    beds = [t for t in ('pad', 'chord', 'wind', 'rain', 'drums', 'chiptune') if r.random() < 0.5] or ['pad']
    for t in beds:
        pos = 0
        while pos < n:
            x = audio.to_mono(G.render(t, int(r.integers(0, 10 ** 6)))).astype(np.float64)
            bus[pos: pos + len(x)] += x * r.uniform(0.2, 0.5)
            pos += max(len(x) - int(0.1 * SR), int(0.5 * SR))
    for _ in range(int(r.integers(6, 25))):
        t = str(r.choice(['beep', 'fm', 'pluck', 'whoosh', 'riser', 'impact', 'click', 'siren', 'lfsr', 'dtmf']))
        x = audio.to_mono(G.render(t, int(r.integers(0, 10 ** 6)))).astype(np.float64)
        i = int(r.integers(0, n))
        bus[i: i + len(x)] += x[: len(bus) - i] * r.uniform(0.2, 0.7)
    bus = bus[:n]
    if r.random() < 0.7:
        bus = G.schroeder(bus, r, mix=r.uniform(0.15, 0.35))[:n]
    bus = np.tanh(bus / (np.percentile(np.abs(bus), 99.5) + 1e-9) * r.uniform(0.8, 1.6))
    return bus / (np.abs(bus).max() + 1e-12) * 10 ** (-1 / 20)


def build_mixes(n=6):
    out = []
    for fam, maker in (('cinetic', cinetic_score), ('generic', generic_mix)):
        for k in range(n):
            if fam == 'cinetic':
                path = maker(k)
                if not path:
                    print(f'cinetic score {k}: render failed')
                    continue
                x = audio.load(path, mono=True, cache=False)
                src = path
            else:
                cached_wav = os.path.join(OUT, 'mixes', f'generic_mix_{k}.wav')
                if not os.path.exists(cached_wav):
                    os.makedirs(os.path.dirname(cached_wav), exist_ok=True)
                    sf.write(cached_wav, maker(k).astype(np.float32), SR, subtype='FLOAT')
                x, _ = sf.read(cached_wav, dtype='float32')
                src = cached_wav
            for s in np.arange(0, len(x) / SR - WIN_S + 1e-6, WIN_S):
                seg = x[int(s * SR): int((s + WIN_S) * SR)]
                if rms_db(seg) > -60:
                    cid = f'{fam}_mix{k}@{s:.0f}'
                    out.append(clip_entry(cid, write(seg, fam, cid.replace('@', '_')), 0, fam, fam,
                                          'score' if fam == 'cinetic' else 'mix', src, float(s), float(s + WIN_S)))
    return out


def _mp3(x):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                        '-c:a', 'libmp3lame', '-b:a', '128k', '-f', 'mp3', '-'], input=np.asarray(x, np.float32).tobytes(),
                       capture_output=True, check=True)
    q = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'mp3', '-i', '-', '-f', 'f32le', '-ac', '1', '-ar',
                        str(SR), '-'], input=p.stdout, capture_output=True, check=True)
    return np.frombuffer(q.stdout, np.float32)


def _pinkish(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X[1:] /= np.sqrt(f[1:] / 100.0)
    X[0] = 0
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def perturb(kind, x, rng, room=None):
    if kind == 'noise70':
        return x + _pinkish(len(x), rng) * 10 ** (-70 / 20)
    if kind == 'room70':
        s = int(rng.integers(0, len(room) - len(x))) if len(room) > len(x) else 0
        r = np.resize(room[s:], len(x))
        return x + r / (np.sqrt(np.mean(r ** 2)) + 1e-12) * 10 ** (-70 / 20)
    if kind == 'gated':
        k = int(0.01 * SR)
        e = np.repeat(np.sqrt(np.convolve(x ** 2, np.ones(k) / k, 'same')[::k] + 1e-20), k)[: len(x)]
        y = x.copy()
        y[: len(e)][e < np.abs(x).max() * 10 ** (-45 / 20)] = 0.0
        return y
    if kind == 'short':
        i = int(np.argmax(np.abs(x)))
        a = max(0, min(i - int(0.02 * SR), len(x) - int(0.15 * SR)))
        return x[a: a + int(0.15 * SR)]
    if kind == 'mp3':
        return _mp3(x)
    raise ValueError(kind)


ROBUST = {'noise70': 0, 'room70': 0, 'gated': 1, 'short': None, 'mp3': None}


def build_robust(clips, per_group=12):
    """perturbed copies of clips spread over every CV group (they keep the group, so held-out predictions stay
    held out): synthesized + pink noise floor / real room tone at -70 dBFS RMS, real gated to digital silence below
    -45 dB re peak, 0.15 s excerpts around the peak (both classes), MP3 128 kbps round trip (both classes)"""
    room = audio.load(os.path.join(ASSETS, 'ambience', 'fs-346642_rain-on-windows-interior.ogg'), mono=True, cache=False)
    room = room[np.argsort([rms_db(room[i:i + SR]) for i in range(0, len(room) - SR, SR)])[0] * SR:][: 20 * SR]
    out = []
    for kind, lab in ROBUST.items():
        rng = np.random.default_rng(len(kind))
        by = {}
        for c in clips:
            if lab is None or c['label'] == lab:
                key = (c['group'], c['type']) if c['label'] == 0 else c['group']
                by.setdefault(key, []).append(c)
        for key in sorted(by, key=str):
            pool = by[key]
            n = max(1, per_group // (6 if isinstance(key, tuple) else 1))
            for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False)):
                c = pool[i]
                x, _ = sf.read(os.path.join(OUT, c['wav']), dtype='float32')
                y = perturb(kind, x, rng, room)
                cid = f'{kind}:{c["id"]}'
                e = dict(c, id=cid, wav=write(y, 'robust', cid.replace(':', '_').replace('@', '_').replace('#', '_')),
                         perturb=kind, base=c['id'])
                out.append(e)
    return out


def _codec(x, rng):
    c = [('libmp3lame', 'mp3', ['-b:a', f'{int(rng.choice([96, 128, 192]))}k'], 'mp3'),
         ('libvorbis', 'ogg', ['-q:a', str(int(rng.integers(3, 7)))], 'ogg'),
         ('aac', 'adts', ['-b:a', '128k'], 'aac')][int(rng.integers(0, 3))]
    p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                        '-c:a', c[0], *c[2], '-f', c[1], '-'], input=np.asarray(x, np.float32).tobytes(),
                       capture_output=True, check=True)
    q = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', c[3], '-i', '-', '-f', 'f32le', '-ac', '1', '-ar',
                        str(SR), '-'], input=p.stdout, capture_output=True, check=True)
    y = np.frombuffer(q.stdout, np.float32)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def augment(kind, x, rng):
    """training-only variants with randomised parameters (different from the fixed robustness set): a synthetic
    coloured noise floor (-85..-50 dBFS RMS; never a real room tone), gating to digital silence (-55..-30 dB re peak),
    a 0.08-0.4 s excerpt, a lossy codec round trip (mp3 96-192k / vorbis q3-6 / aac 128k)"""
    if kind == 'floor':
        X = np.fft.rfft(rng.standard_normal(len(x)))
        f = np.fft.rfftfreq(len(x), 1 / SR)
        X[1:] /= (f[1:] / 100.0) ** (float(rng.choice([0.0, 1.0, 2.0])) / 2)
        X[0] = 0
        nz = np.fft.irfft(X, len(x))
        return x + nz / (np.std(nz) + 1e-12) * 10 ** (rng.uniform(-85, -50) / 20)
    if kind == 'gate':
        k = int(rng.uniform(0.005, 0.02) * SR)
        e = np.repeat(np.sqrt(np.convolve(x ** 2, np.ones(k) / k, 'same')[::k] + 1e-20), k)[: len(x)]
        y = x.copy()
        y[: len(e)][e < np.abs(x).max() * 10 ** (rng.uniform(-55, -30) / 20)] = 0.0
        return y
    if kind == 'short':
        n = int(rng.uniform(0.08, 0.4) * SR)
        i = int(np.argmax(np.abs(x))) - int(rng.uniform(0.0, 0.05) * SR)
        a = max(0, min(i, len(x) - n))
        return x[a: a + n]
    return _codec(x, rng)


def build_augment(clips):
    """one randomised training variant for about half of the clean clips (chosen by id hash)"""
    import hashlib
    out = []
    for c in clips:
        h = int(hashlib.sha1(c['id'].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        if rng.random() >= 0.5:
            continue
        kinds, probs = (['floor', 'short', 'codec'], [0.45, 0.25, 0.3]) if c['label'] == 0 else             (['gate', 'short', 'codec'], [0.35, 0.35, 0.3])
        kind = str(rng.choice(kinds, p=probs))
        x, _ = sf.read(os.path.join(OUT, c['wav']), dtype='float32')
        y = augment(kind, x.astype(np.float64), rng)
        cid = f'aug-{kind}:{c["id"]}'
        out.append(dict(c, id=cid, wav=write(y, 'augment', cid.replace(':', '_').replace('@', '_').replace('#', '_')),
                        perturb='aug-' + kind, base=c['id'], augment=True))
    return out


LAYER_LEVELS = (-12, -6, 0, 6)


def build_layers(clips, per_group=24):
    """real 4 s beds (windows of ambience, music, films ... up to per_group per CV group) with one code-synthesized
    layer mixed in, label 0: does a code-made layer inside a real mix get found? Each bed gets two different layers at
    two of LAYER_LEVELS (layer RMS over the span it plays vs the bed RMS, dB). A one-shot layer plays 1-4 times at
    random offsets, a 4 s synth window plays throughout. The clip keeps the layer's family / group / type and records
    the bed's (bed_group, bed_src): run_eval keeps both groups out of a model before it scores the clip."""
    import hashlib
    beds = {}
    for c in clips:
        if c['label'] == 1 and c['end'] - c['start'] >= 3.5:
            beds.setdefault(c['group'], []).append(c)
    syn = sorted((c for c in clips if c['label'] == 0), key=lambda c: c['id'])
    out = []
    for g in sorted(beds):
        pool = sorted(beds[g], key=lambda c: hashlib.sha1(c['id'].encode()).hexdigest())[:per_group]
        for j, b in enumerate(pool):
            rng = np.random.default_rng(int(hashlib.sha1(b['id'].encode()).hexdigest()[:8], 16))
            x, _ = sf.read(os.path.join(OUT, b['wav']), dtype='float64')
            x = x[: int(WIN_S * SR)]
            for k in range(2):
                lc = syn[int(rng.integers(0, len(syn)))]
                lev = LAYER_LEVELS[(j + 2 * k) % len(LAYER_LEVELS)]
                y, _ = sf.read(os.path.join(OUT, lc['wav']), dtype='float64')
                lay, act = np.zeros(len(x)), np.zeros(len(x), bool)
                if len(y) >= 0.75 * len(x):
                    lay[: min(len(x), len(y))] = y[: len(x)]
                    act[:] = True
                else:
                    for _ in range(int(rng.integers(1, 5))):
                        a = int(rng.integers(0, len(x) - len(y)))
                        lay[a: a + len(y)] += y
                        act[a: a + len(y)] = True
                gain = 10 ** ((lev + rms_db(x) - rms_db(lay[act])) / 20)
                mix = x + gain * lay
                mix *= min(1.0, 0.98 / (np.abs(mix).max() + 1e-12))
                cid = f'layer{lev:+d}:{lc["id"]}+{b["id"]}'
                out.append(dict(clip_entry(cid, write(mix, 'layers', f'L{len(out):04d}'), 0, lc['family'], lc['group'],
                                           lc['type'], lc['src'], b['start'], b['end']),
                                perturb='layer', level_db=lev, layer=lc['id'], bed=b['id'], bed_group=b['group'],
                                bed_src=b['src'], bed_family=b['family']))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', default='real,rejected,films,retired,cinetic,generic,mixes,robust,augment',
                    help='parts to (re)build; also: layers (real beds + a code-synthesized layer, label 0)')
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, 'manifest.json')
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    builders = {'real': build_real, 'rejected': lambda: build_real(False), 'films': build_films, 'retired': build_retired,
                'cinetic': build_cinetic, 'generic': build_generic, 'mixes': build_mixes}
    for part in a.only.split(','):
        if part in builders:
            man[part] = builders[part]()
            print(f'{part}: {len(man[part])} clips')
            json.dump(man, open(mpath, 'w'), indent=1)
    base = [c for k in builders for c in man.get(k, [])]
    if 'robust' in a.only.split(','):
        man['robust'] = build_robust(base)
        print(f'robust: {len(man["robust"])} clips')
    if 'augment' in a.only.split(','):
        man['augment'] = build_augment(base)
        print(f'augment: {len(man["augment"])} clips')
    if 'layers' in a.only.split(','):
        man['layers'] = build_layers(base)
        print(f'layers: {len(man["layers"])} clips')
    json.dump(man, open(mpath, 'w'), indent=1)
    n = sum(len(v) for v in man.values())
    print(f'manifest: {n} clips -> {mpath}')


if __name__ == '__main__':
    main()
