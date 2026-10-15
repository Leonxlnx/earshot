"""Choose how timeline scores section tags (genre / mood / instruments / ambience). Never uses the Clawd soundtrack.

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/timeline/label_tune.py

Data: (1) music recordings of the Clawd library that the Clawd mix does not use (assets/music minus the film's track,
plus _rejected/music): bright / playful / uplifting tracks picked for a cute launch film; each scored like a timeline
section (mean of three 3 s repeat-padded CLAP windows at 25 / 50 / 75 % of the file). (2) the music and ambience beds
of the synthetic tuning mixes (out/timeline/mixes: real beds with real one-shots on top), section windows inside each
bed averaged. Variants: raw = per-sound z of the raw cosine over all vocab terms (first build); centred = the same
after removing each term's bias over earshot's reference set (cos - m.q, data/vocab_stats.json, as earshot.vocab does).
Metrics: share of tracks whose top mood is a dark / aggressive one (wrong for this library), the most common top mood
and genre (hub terms), genre / instrument keyword agreement with the file names, ambience keyword agreement for beds.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from earshot import audio, config, timeline as tl  # noqa: E402
from earshot.models import clap  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

ASSETS = EP.ASSETS
SR = audio.SR
DARK = {'aggressive music', 'dark and tense music', 'suspenseful build-up music', 'mysterious music',
        'sad and melancholic music'}
GENRE_KW = {'orch': {'cinematic orchestral music', 'uplifting orchestral film score', 'epic trailer music',
                     'classical string quartet'},
            'piano': {'solo piano music', "playful children's piano music"},
            'pop': {'upbeat corporate pop music', 'indie pop music', 'inspirational corporate background music'}}
INSTR_KW = {'piano': {'a grand piano', 'an electric piano'}, 'celesta': {'a celesta', 'a glockenspiel', 'bells',
            'a music box'}, 'marimba': {'a marimba'}, 'ukulele': {'a ukulele'}, 'cello': {'strings ensemble'},
            'bells': {'bells', 'a glockenspiel', 'a celesta', 'a music box'}}
AMB_KW = {'rain': {'rain on a window', 'heavy rain', 'rain on a roof', 'a thunderstorm'},
          'wind': {'wind blowing', 'a gentle breeze in tall grass', 'wind in pine trees'},
          'bird': {'birds singing in the morning', 'skylarks over a meadow', 'a forest ambience',
                   'a quiet countryside morning'},
          'chorus': {'birds singing in the morning', 'skylarks over a meadow', 'a forest ambience'},
          'meadow': {'skylarks over a meadow', 'a summer meadow with insects', 'a quiet countryside morning',
                     'a gentle breeze in tall grass', 'birds singing in the morning'},
          'forest': {'a forest ambience', 'birds singing in the morning', 'wind in pine trees'},
          'night': {'crickets at night', 'a quiet room at night'}, 'cricket': {'crickets at night'},
          'room': {'a quiet room tone', 'a quiet room at night', 'an office ambience', 'a computer fan hum'},
          'hum': {'a computer fan hum', 'an electrical hum'}, 'cafe': {'a busy cafe'},
          'city': {'a city street with traffic'}, 'traffic': {'a city street with traffic'},
          'ocean': {'ocean waves'}, 'wave': {'ocean waves'}, 'fire': {'a crackling fire'}}


def music_files():
    fs = sorted(glob.glob(f'{ASSETS}/music/*.mp3')) + sorted(glob.glob(f'{ASSETS}/_rejected/music/*.mp3'))
    film = {os.path.basename(f) for f in json.load(open(f'{EP.FILM}/out/qa/mix_report.json'))['files_used']}
    return [f for f in fs if os.path.basename(f) not in film]


def embed_music(c, files):
    out = []
    for f in files:
        d = audio.duration(f)
        A = tl._cached(c, audio.file_id(f), [(d * q, d * q + 3.0) for q in (0.25, 0.5, 0.75)], 'clap-repeat',
                       lambda sp: c.embed_audio([audio.load(f, mono=True, start=a, end=b) for a, b in sp]))
        v = A.mean(axis=0)
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def embed_beds(c):
    out = []
    for jf in sorted(glob.glob(os.path.join(ROOT, 'out/timeline/mixes/mix_*.json'))):
        meta = json.load(open(jf))
        wav = jf[:-5] + '.wav'
        x = audio.load(wav)
        fid = audio.file_id(wav)
        for b in meta['beds']:
            st = np.arange(b['start'] + 0.25, b['end'] - 3.25 + 1e-6, 2.0)
            if not len(st):
                continue
            sp = [(round(s, 3), round(s + 3.0, 3)) for s in st]
            A = tl._cached(c, fid, sp, 'clap-repeat', lambda q: c.embed_audio([tl._seg(x, a, e) for a, e in q]))
            v = A.mean(axis=0)
            out.append((b['kind'], b['file'], v / np.linalg.norm(v)))
    return out


def scorer(E, terms, cats, mvec, variant):
    bias = E @ mvec if variant == 'centred' else np.zeros(len(terms))

    def top(v, cat, k=1):
        u = E @ v - bias
        z = (u - u.mean()) / (u.std() + 1e-9)
        idx = [i for i in np.argsort(-z) if cats[i] == cat][:k]
        return [terms[i] for i in idx], [float(z[i]) for i in idx]
    return top


def main():
    c = clap.get()
    terms, cats, E = tl._vocab_matrix(c)
    mvec = np.array(json.load(open(os.path.join(config.DATA_DIR, 'vocab_stats.json')))['mean'], np.float32)
    files = music_files()
    M = embed_music(c, files)
    beds = embed_beds(c)
    names = [os.path.basename(f).lower() for f in files]
    res = {'n_music_files': len(files), 'n_beds': len(beds)}
    for variant in ('raw', 'centred'):
        top = scorer(E, terms, cats, mvec, variant)
        moods = [top(v, 'music_mood')[0][0] for v in M]
        genres = [top(v, 'music_genre')[0][0] for v in M]
        g_ok = [genres[i] in s for i, n in enumerate(names) for kw, s in GENRE_KW.items() if kw in n]
        ins3 = [set(top(v, 'instruments', 3)[0]) for v in M]
        i_ok = [bool(ins3[i] & s) for i, n in enumerate(names) for kw, s in INSTR_KW.items() if kw in n]
        a_ok, mus_ok, bed_dark, bed_g = [], [], [], []
        for kind, f, v in beds:
            if kind == 'music':
                bed_dark.append(top(v, 'music_mood')[0][0] in DARK)
                bed_g += [top(v, 'music_genre')[0][0] in s for kw, s in GENRE_KW.items() if kw in f.lower()]
            gz, az = top(v, 'music_genre')[1][0], top(v, 'ambience')[1][0]
            mus_ok.append((gz >= az) == (kind == 'music'))
            if kind == 'ambience':
                hits = [s for kw, s in AMB_KW.items() if kw in f.lower()]
                if hits:
                    a_ok.append(any(set(top(v, 'ambience', 2)[0]) & s for s in hits))
        hub_m = max(set(moods), key=moods.count)
        hub_g = max(set(genres), key=genres.count)
        res[variant] = {
            'dark_or_aggressive_top_mood': round(float(np.mean([m in DARK for m in moods])), 3),
            'aggressive_top_mood': round(float(np.mean([m == 'aggressive music' for m in moods])), 3),
            'hub_mood': [hub_m, round(moods.count(hub_m) / len(moods), 3)], 'distinct_moods': len(set(moods)),
            'hub_genre': [hub_g, round(genres.count(hub_g) / len(genres), 3)], 'distinct_genres': len(set(genres)),
            'genre_keyword_agree': [round(float(np.mean(g_ok)), 3), len(g_ok)],
            'instrument_keyword_in_top3': [round(float(np.mean(i_ok)), 3), len(i_ok)],
            'bed_music_vs_ambience_correct': [round(float(np.mean(mus_ok)), 3), len(mus_ok)],
            'ambience_keyword_in_top2': [round(float(np.mean(a_ok)), 3), len(a_ok)],
            'bed_music_with_hits_dark_top_mood': [round(float(np.mean(bed_dark)), 3), len(bed_dark)],
            'bed_music_with_hits_genre_keyword_agree': [round(float(np.mean(bed_g)), 3) if bed_g else None, len(bed_g)],
        }
        print(variant, json.dumps(res[variant]))
    json.dump(res, open(os.path.join(HERE, 'label_tune.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
