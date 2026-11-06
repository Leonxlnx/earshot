"""Real sound effects judged by what they are and by the job they play: does earshot hear real recordings?

Data (read-only, never copied, only aggregates are written): a library of real licensed one-shot effects (Freesound
CC0 / Mixkit), each with its title as published (a person's label of what it is) and the film jobs it was filed under
(appear, toggle, logo-lock ...) with the description an agent judged that job by (`ears.<job>.query`). Location:
EARSHOT_EVAL_SFX_LIBRARY or local_paths.json "sfx_library" (eval/README.md).

Per padding of the short sounds (repeatpad, shipped; spaced; zero) and per way of asking:
  title       each sound's own title as the query (z, verdict; rank of its title among all titles)
  job         the job's description as filed (often job words: "a soft rounded pop, something appearing")
  sound_only  the best of two phrasings of what the job's sound is (SOUND_ONLY, written before any score)
  union       the best of the job's description and the two phrasings
it reports the verdicts for each sound's own job, the AUC of own against other jobs, the share of other jobs a sound
passes (z >= 2), and how often the forced choice against four cheap-synth descriptions goes to a cheap synth.

    EARSHOT_THREADS=2 .venv/Scripts/python.exe eval/semantics/real_effects.py   # -> eval/semantics/real_effects.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from earshot import match, vocab  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/: local data locations
import evalpaths as EP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PADS = ('repeatpad', 'spaced', 'zero')
CHEAP = ['a cheap synthesizer laser zap', 'a video game blip', 'an 8-bit chiptune sound', 'a rising synth whip sweep']
# written 2026-11-06 before any of them was scored: what each job's sound is, never the job
SOUND_ONLY = {
    'logo-lock': ['a single low impact', 'a struck note ringing out'],
    'error': ['a short error buzz', 'a short glitchy digital buzz'],
    'letter': ['a tiny soft tap', 'a tiny click'],
    'detent': ['a small mechanical tick', 'a ratchet click'],
    'appear': ['a soft pop', 'a mouth pop'],
    'jump-cut': ['a dry quick click', 'a short sharp snap'],
    'touch': ['a light finger tap', 'a soft tap on glass'],
    'lock': ['a short wooden knock', 'a plastic latch clicking shut'],
    'press': ['a soft mouse click', 'a button click'],
    'toggle': ['a light switch click', 'a small switch being flipped'],
    'shutter': ['a camera shutter', 'a camera shutter click'],
    'slide': ['a thin quick swish', 'a soft sliding swoosh'],
    'type': ['a mechanical keyboard key press', 'a keyboard key click'],
    'swipe': ['a paper swipe', 'a quick soft swoosh'],
    'whip': ['a fast air whoosh', 'a whip swoosh'],
    'card-land': ['paper dropped on a table', 'a soft slap of a card on a table'],
    'land': ['a soft low thump', 'a soft thud'],
    'success': ['a soft ding', 'a soft chime notification'],
    'chime': ['a struck bell note', 'a chime'],
    'text': ['a soft plucked or mallet note', 'a kalimba or marimba note'],
    'reveal': ['a single clean low impact', 'a deep cinematic hit'],
    'thud': ['a deep short sub thud', 'a low boom'],
    'pass': ['a big low whoosh', 'a deep air whoosh'],
    'shape': ['a soft air whoosh', 'a soft swoosh'],
    'swell': ['a reverse swell', 'a reversed cymbal'],
    'build': ['a short dark riser', 'a rising whoosh'],
}
BANDS = ('strong', 'plausible', 'weak', 'no match')


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def bands(zs):
    return {b: int(sum(match.verdict(float(z)) == b for z in zs)) for b in BANDS}


def best_of(v, A, groups):
    """groups: {key: [queries]} -> {key: (z (n,), cosine (n,))} of the best-z query per sound"""
    flat = [(k, q) for k, qs in groups.items() for q in qs]
    sc = v.scores(A, [q for _, q in flat])
    Z = np.array([[r['z'] for r in row] for row in sc])
    C = np.array([[r['cosine'] for r in row] for row in sc])
    out = {}
    for k in groups:
        cols = [i for i, (kk, _) in enumerate(flat) if kk == k]
        pick = np.array(cols)[np.argmax(Z[:, cols], axis=1)]
        out[k] = (Z[np.arange(len(A)), pick], C[np.arange(len(A)), pick])
    return out


def run(lib_path):
    lib = json.load(open(lib_path, encoding='utf-8'))
    root = os.path.join(os.path.dirname(lib_path), lib.get('root', ''))
    sounds = [s for s in lib['sounds'] if os.path.isfile(os.path.join(root, s['file']))]
    job_q = {}
    for s in sounds:
        for a, e in (s.get('ears') or {}).items():
            job_q.setdefault(a, e['query'])
    jobs = sorted(a for a in job_q if a in SOUND_ONLY)
    ways = {'job': {a: [job_q[a]] for a in jobs}, 'sound_only': {a: SOUND_ONLY[a] for a in jobs},
            'union': {a: [job_q[a]] + SOUND_ONLY[a] for a in jobs}}
    v = vocab.get()
    scale = vocab.logit_scale()
    titles = [s['title'] for s in sounds]
    same = np.array([[t.strip().lower() == u.strip().lower() for u in titles] for t in titles])
    mine = [[jobs.index(a) for a in s['actions'] if a in jobs] for s in sounds]
    res = dict(n_sounds=len(sounds), n_jobs=len(jobs), n_short=sum((s.get('dur_s') or 9) < 1.0 for s in sounds),
               pads={})
    for pad in PADS:
        A = vocab.embed_files([os.path.join(root, s['file']) for s in sounds], pad=pad)
        cheap = np.array([[r['cosine'] for r in row] for row in v.scores(A, CHEAP)])
        zt = np.array([[r['z'] for r in row] for row in v.scores(A, titles)])
        own_t = np.diag(zt)
        r = dict(title=dict(verdicts=bands(own_t), median_z=round(float(np.median(own_t)), 2),
                            auc=round(auc(zt[same], zt[~same]), 4),
                            top1=int(sum((zt[i] > zt[i, i]).sum() == 0 for i in range(len(sounds)))),
                            top5=int(sum((zt[i] > zt[i, i]).sum() < 5 for i in range(len(sounds))))))
        for way, groups in ways.items():
            b = best_of(v, A, groups)
            Z = np.stack([b[a][0] for a in jobs], axis=1)
            C = np.stack([b[a][1] for a in jobs], axis=1)
            own, pos, neg, other_pass, cheap_won, per_job = [], [], [], [], 0, {}
            for i, m in enumerate(mine):
                if not m:
                    continue
                j = max(m, key=lambda k: Z[i, k])
                own.append(Z[i, j])
                p = match.softmax([C[i, j]] + list(cheap[i]), scale)
                cheap_won += int(p[0] < max(p[1:]))
                others = [k for k in range(len(jobs)) if k not in m]
                pos += [Z[i, k] for k in m]
                neg += [Z[i, k] for k in others]
                other_pass.append(float(np.mean([Z[i, k] >= 2.0 for k in others])))
                pj = per_job.setdefault(jobs[j], [0, 0])
                pj[0] += int(Z[i, j] >= 2.0)
                pj[1] += 1
            r[way] = dict(verdicts=bands(own), median_z=round(float(np.median(own)), 2), auc=round(auc(pos, neg), 4),
                          other_jobs_passed=round(float(np.mean(other_pass)), 4), cheap_synth_won=cheap_won,
                          passed_by_job={a: f'{x[0]}/{x[1]}' for a, x in sorted(per_job.items())})
        res['pads'][pad] = r
        print(pad, json.dumps({k: {kk: vv for kk, vv in x.items() if kk != 'passed_by_job'} for k, x in r.items()}))
        sys.stdout.flush()
    return res


def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else EP.SFX_LIBRARY
    res = dict(what=__doc__.split('\n')[0], queries=dict(sound_only=SOUND_ONLY, cheap_synth=CHEAP), **run(lib))
    with open(os.path.join(HERE, 'real_effects.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(res, fh, indent=1)
        fh.write('\n')


if __name__ == '__main__':
    main()
