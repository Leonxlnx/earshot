"""The catalogue as earshot hears it: every design rendered once and embedded with CLAP, so a request that only
describes how a sound sounds ("a thin rising air streak that zips off into the distance") finds the designs that
sound like that.

    from earshot.sounds import atlas
    atlas.nearest('a plump, warm double plink, mellow and rounded', top=3)
    # [{'id': 'note/...', 'cosine': 0.41, 'z': 3.2}, ...]
    python -m earshot.sounds.atlas build --work DIR      # (re)build data/sounds_atlas.npz, a few minutes per call
    python -m earshot.sounds.atlas status                # is the atlas complete and current?

The atlas (data/sounds_atlas.npz) holds numbers only: the design ids; per design two float16 unit vectors, `vecs`
(CLAP's audio tower on the design rendered at its defaults with seed 1, beds at 8 s, a take under a second padded
with silence to one second) and `says` (CLAP's text tower on the design's own descriptions of its sound, averaged);
the model and revision they were made with, a key per design and the digest of the catalogue those keys make. A
design's key covers what its default render is made from (id, kind, length, pitch, voice, material, loop) and its
`says`; a change inside a design's code is not seen, so rebuild after re-voicing designs.

A request is scored by its CLAP text embedding (the vocabulary's prompt templates, earshot.vocab) against every
design. Raw cosines of different texts are not comparable (some phrases score high for every sound), so each is
turned into a z over the atlas for this very text, and the two are mixed (SAYS_WEIGHT): CLAP matching the request
to the rendered sound alone finds the right family first for a quarter of the development requests, to the
descriptions for half, the mix for a little more. CLAP hears sounds, not jobs ("something appearing" matches
nothing), so earshot.brief.rank blends z with the lexical score instead of replacing it. The model loads on the
first text that is not in the embedding cache; nothing is downloaded.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

from .. import config
from ..forge import dsp

ATLAS_PATH = os.path.join(config.DATA_DIR, 'sounds_atlas.npz')
SEED = 1
BED_S = 8.0              # a bed is rendered at 8 s: one CLAP window of its texture
PAD = 'spaced'         # a take under SPACED_S gets silence up to SPACED_S, then repeats (one hit per second)
SPACED_S = 1.0
# how much the request's likeness to a design's own descriptions (`says`, CLAP text against text) counts against its
# likeness to the design's sound (CLAP text against audio). Development sets, first pick over all designs: sound alone
# 26 %, descriptions alone 50 %, 0.7 of descriptions + 0.3 of sound 53 % (eval/requests.json, 440 requests)
SAYS_WEIGHT = 0.7
_state = {}


class AtlasError(RuntimeError):
    """the atlas or the model it needs is missing or does not fit the catalogue"""


def _model_tag():
    """(model id, revision, embedding-cache name) of the CLAP model earshot loads; filesystem only"""
    from ..models import hub
    spec = hub.resolve(config.CLAP_MODEL, config.CLAP_REVISION)
    return config.CLAP_MODEL, spec.revision, spec.cache_name


def _h(x):
    return hashlib.sha1(json.dumps(x).encode()).hexdigest()[:12]


def design_key(d):
    """'<render>.<says>': what a design's default render is made of and what its `says` are, hashed (module doc)"""
    return _h([d.id, d.kind, d.dur, d.pitch, d.voice, d.material, bool(d.loop)]) + '.' + _h(list(d.says))


def catalogue(designs=None):
    """({id: key} of the catalogue, its digest): the digest changes when a design is added, removed or its key
    changes"""
    if designs is None:
        from . import registry
        designs = registry.load()
    keys = {i: design_key(d) for i, d in sorted(designs.items())}
    return keys, hashlib.sha1(json.dumps(sorted(keys.items())).encode()).hexdigest()[:16]


def render_default(id):
    """the take the atlas hears: the design at its defaults, seed 1, beds at BED_S; mono float32"""
    from . import registry
    d = registry.get(id)
    y, _ = registry.render(id, seed=SEED, check=False, **({'dur': BED_S} if d.kind == 'bed' else {}))
    y = np.asarray(y, np.float64)
    return (y.mean(axis=1) if y.ndim == 2 else y).astype(np.float32)


def spaced(x):
    """a short take padded with silence to SPACED_S, so CLAP's repeat padding plays it once a second: a 50 ms tick
    repeated back to back would be a 20 Hz buzz"""
    n = int(SPACED_S * dsp.SR)
    return np.pad(x, (0, n - len(x))) if len(x) < n else x


# ---------------------------------------------------------------------------------------------------- building
def build(work, budget=150.0, path=ATLAS_PATH, batch=8, log=print):
    """one resumable step of building the atlas: renders the designs not yet rendered into `work`, then (when every
    design is rendered and at least a minute of the budget is left) embeds the ones not yet embedded, then writes
    `path` when every design is embedded. Stops after about `budget` seconds; call again until it says done.
    Renders and embeddings are kept in `work` under the design's key, so a later build only redoes what changed.
    Returns {'rendered', 'embedded', 'total', 'done'}"""
    t0 = time.time()
    os.makedirs(work, exist_ok=True)
    keys, digest = catalogue()
    model, rev, tag = _model_tag()

    def name(i):
        return i.replace('/', '__')

    def wav(i):
        return os.path.join(work, f'r_{name(i)}.{keys[i].split(".")[0]}.npy')

    def emb(i):
        return os.path.join(work, f'e_{name(i)}.{keys[i].split(".")[0]}.{tag}.{PAD}.npy')

    def semb(i):
        return os.path.join(work, f's_{name(i)}.{keys[i].split(".")[1]}.{tag}.npy')

    todo = [i for i in keys if not os.path.exists(wav(i))]
    for i in todo:
        if time.time() - t0 > budget:
            break
        np.save(wav(i), render_default(i))
    left = [i for i in keys if not (os.path.exists(emb(i)) and os.path.exists(semb(i)))]
    rendered = all(os.path.exists(wav(i)) for i in keys)
    if left and rendered and budget - (time.time() - t0) > 60:
        from ..models import clap
        from . import registry
        c = clap.get()
        for k in range(0, len(left), batch):
            if time.time() - t0 > budget:
                break
            chunk = left[k:k + batch]
            heard = [i for i in chunk if not os.path.exists(emb(i))]
            if heard:
                vecs = c.embed_audio([spaced(np.load(wav(i))) for i in heard], pad='repeatpad', batch=batch)
                for i, v in zip(heard, vecs):
                    np.save(emb(i), v.astype(np.float32))
            for i in chunk:
                s = c.embed_text(list(registry.get(i).says)).mean(axis=0)
                np.save(semb(i), (s / np.linalg.norm(s)).astype(np.float32))
    have = [i for i in keys if os.path.exists(emb(i)) and os.path.exists(semb(i))]
    res = {'rendered': sum(os.path.exists(wav(i)) for i in keys), 'embedded': len(have), 'total': len(keys),
           'done': len(have) == len(keys), 'seconds': round(time.time() - t0, 1)}
    if res['done']:
        ids = list(keys)
        V, W = (np.stack([np.load(f(i)) for i in ids]).astype(np.float64) for f in (emb, semb))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        tmp = path + '.tmp.npz'
        np.savez_compressed(tmp, ids=np.array(ids), keys=np.array([keys[i] for i in ids]),
                            vecs=V.astype(np.float16), says=W.astype(np.float16), model=np.array(model),
                            revision=np.array(rev),
                            cache=np.array(tag), digest=np.array(digest), seed=np.array(SEED), pad=np.array(PAD),
                            bed_s=np.array(BED_S))
        os.replace(tmp, path)
        res['path'] = path
    log(json.dumps(res))
    return res


# ---------------------------------------------------------------------------------------------------- reading
def load(path=ATLAS_PATH):
    """the atlas as {'ids', 'keys', 'vecs' (float32), 'model', 'revision', 'cache', 'digest'}; AtlasError when the
    file is missing"""
    if not os.path.exists(path):
        raise AtlasError(f'no sound atlas at {path}: build it with `python -m earshot.sounds.atlas build --work DIR`')
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    out = {k: str(a[k]) for k in ('model', 'revision', 'cache', 'digest')}
    if 'says' not in a:
        raise AtlasError(f'the sound atlas at {path} is from an older format: rebuild it')
    out.update(ids=[str(i) for i in a['ids']], keys=[str(k) for k in a['keys']], vecs=a['vecs'].astype(np.float32),
               says=a['says'].astype(np.float32))
    return out


def check(a=None):
    """problems that make the atlas unfit for this catalogue and model, as plain sentences ([] = fit)"""
    a = a or load()
    keys, digest = catalogue()
    out = []
    model, _, tag = _model_tag()
    if a['model'] != model or a['cache'] != tag:
        out.append(f'the atlas was made with {a["model"]} ({a["cache"]}), earshot now loads {model} ({tag})')
    if a['digest'] != digest:
        have = dict(zip(a['ids'], a['keys']))
        missing = sorted(i for i in keys if i not in have)
        changed = sorted(i for i in keys if i in have and have[i] != keys[i])
        gone = sorted(i for i in have if i not in keys)
        bits = [f'{len(x)} {w} ({", ".join(x[:5])}{" ..." if len(x) > 5 else ""})'
                for x, w in ((missing, 'designs missing'), (changed, 'designs changed'), (gone, 'designs removed'))
                if x]
        out.append('the atlas does not match the catalogue: ' + '; '.join(bits))
    return out


def _atlas():
    """the loaded, checked atlas (kept for the process); AtlasError when missing or stale"""
    if 'atlas' not in _state:
        a = load()
        bad = check(a)
        if bad:
            raise AtlasError('the sound atlas is stale: ' + ' and '.join(bad) +
                             '. Rebuild it: python -m earshot.sounds.atlas build --work DIR')
        _state['atlas'] = a
    return _state['atlas']


def warm():
    """True when this process has the CLAP model loaded already (the MCP server keeps it warm): then a semantic
    ranking costs milliseconds"""
    m = sys.modules.get('earshot.models.clap')
    return bool(m is not None and getattr(m, '_inst', None) is not None)


def embed_text(texts):
    """(n, 512) text embeddings through the vocabulary's prompt templates, averaged and renormalised (as
    earshot.vocab.Vocab.embed_text, sharing its per-phrase disk cache: a text seen before needs no model). A new text
    goes through CLAP's text tower directly: the model's own cache of every templated phrase is one file rewritten
    on each new text (seconds once it is large). AtlasError when the model is not in the cache"""
    from .. import audio, vocab
    texts = [texts] if isinstance(texts, str) else list(texts)
    templates = vocab.load_vocab()['templates']
    try:
        d = vocab._cache_dir()
        out = []
        for t in texts:
            f = os.path.join(d, 't_' + vocab._key(json.dumps([templates, [t]])) + '.npy')
            v = np.load(f) if os.path.exists(f) else None
            if v is None or v.shape != (512,):
                v = _forward([tp.format(t) for tp in templates]).mean(axis=0)
                v = (v / np.linalg.norm(v)).astype(np.float32)
                tmp = f'{f}.{os.getpid()}.tmp.npy'
                np.save(tmp, v)
                audio.replace_quietly(tmp, f)
            out.append(v)
        return np.stack(out)
    except (RuntimeError, OSError, ImportError) as e:
        raise AtlasError(f'cannot embed the request with CLAP: {e}') from e


def _forward(phrases):
    """(n, 512) unit text embeddings of CLAP's text tower, uncached"""
    from ..models import clap
    c = clap.get()
    inp = c.proc(text=list(phrases), return_tensors='pt', padding=True)
    with c.torch.inference_mode():
        e = c.model.get_text_features(**inp)
        e = getattr(e, 'pooler_output', e)
    e = e.float().numpy()
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def scores(text, q=None):
    """(ids, cosines with the sound, z) of every design for a request; q: its text embedding when already known.
    z = the z over the atlas of SAYS_WEIGHT * z(cosine with the design's `says`) + the rest * z(cosine with its
    sound) (module doc)"""
    a = _atlas()
    q = embed_text(text)[0] if q is None else np.asarray(q, np.float32)
    cos = a['vecs'] @ q
    return a['ids'], cos, _z(SAYS_WEIGHT * _z(a['says'] @ q) + (1.0 - SAYS_WEIGHT) * _z(cos))


def nearest(text, top=10, q=None):
    """the designs that sound most like a description: [{'id', 'cosine', 'z'}], best first (module doc)"""
    ids, cos, z = scores(text, q)
    order = np.argsort(-z, kind='stable')[:top]
    return [{'id': ids[k], 'cosine': round(float(cos[k]), 3), 'z': round(float(z[k]), 2)} for k in order]


def zmap(text, q=None):
    """{id: z} of every design for a request (what earshot.brief.rank blends in)"""
    ids, _, z = scores(text, q)
    return dict(zip(ids, (float(x) for x in z)))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m earshot.sounds.atlas', description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='one resumable step of (re)building the atlas; repeat until it says done')
    b.add_argument('--work', default=os.path.join(config.HOME, 'atlas-build'),
                   help='folder for the renders and embeddings kept between steps')
    b.add_argument('--budget', type=float, default=150.0, help='seconds one step may take (about)')
    b.add_argument('--out', default=ATLAS_PATH)
    sub.add_parser('status', help='is the atlas complete and current?')
    n = sub.add_parser('nearest', help='the designs that sound most like a description')
    n.add_argument('text')
    n.add_argument('--top', type=int, default=10)
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        return 0 if build(a.work, a.budget, a.out)['done'] else 3
    if a.cmd == 'status':
        try:
            at = load()
            bad = check(at)
        except AtlasError as e:
            print(e)
            return 1
        print(f'{len(at["ids"])} designs, {at["model"]} ({at["cache"]}), digest {at["digest"]}: '
              + ('; '.join(bad) if bad else 'current'))
        return 1 if bad else 0
    for r in nearest(a.text, a.top):
        print(f'{r["id"]:<28} z {r["z"]:+.2f}  cos {r["cosine"]:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
