"""The model files earshot loads: pinned revisions in earshot's own Hugging Face cache, never fetched behind your back.

    from earshot.models import hub
    s = hub.resolve('laion/larger_clap_general')   # filesystem only; s.state: cached / legacy / incomplete / missing
    s = hub.ensure('laion/larger_clap_general')    # the same, plus one download of the pinned files if none can load

The models load only from a complete snapshot folder (EARSHOT_HOME/hf/models--org--name/snapshots/<sha>): transformers
gets a folder, never a repo id, so a cached load never asks the Hub, and a damaged cache is an error that names the
folder instead of a silent new download. A model that is not cached is downloaded once at its pinned revision
(config.PINS), with safetensors weights, unless an offline flag is set (config.offline()). earshot never sets or
clears HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE. Imports nothing heavy: earshot doctor uses it without torch.
"""
import os
import re
from dataclasses import dataclass

from .. import config

SAFETENSORS = ('model.safetensors', 'model.safetensors.index.json')   # one file, or the index of a sharded checkpoint
CUSTOM_PATTERNS = ('*.json', '*.txt', '*.model', '*.safetensors')     # an unpinned model: configs, tokenizer, weights
_SHA = re.compile(r'[0-9a-f]{40}')


@dataclass(frozen=True)
class Spec:
    """one model as earshot will load it"""
    model_id: str
    revision: str            # the revision earshot wants: its pin, an EARSHOT_*_REVISION override, or 'main'
    state: str               # cached | legacy (a pre-0.2.1 snapshot of the pin stands in) | incomplete | missing
    snapshot: str | None     # the complete local folder to load from (state cached or legacy), else None
    weights: str             # the weights file loaded or to download (pytorch_model.bin only for a legacy snapshot)
    missing: tuple = ()      # files of the wanted revision that are not in the cache
    pinned: bool = False     # the wanted revision is earshot's pin

    @property
    def local(self):
        return self.snapshot is not None

    @property
    def sha(self):
        """the commit of the snapshot that loads (None when nothing local can)"""
        return os.path.basename(self.snapshot) if self.snapshot else None

    @property
    def label(self):
        return f'{self.model_id}@{self.revision[:7] if _SHA.fullmatch(self.revision) else self.revision}'

    @property
    def cache_tag(self):
        """suffix for embedding caches: none for the pin and its legacy snapshot (the same tensors), else the commit,
        so embeddings of another checkpoint never mix with the pinned model's"""
        if self.pinned:
            return ''
        return '--' + (self.sha or re.sub(r'[^0-9A-Za-z._-]', '_', self.revision))[:8]


def model_dir(model_id):
    return os.path.join(config.HF_CACHE, 'models--' + model_id.replace('/', '--'))


def snapshot_dir(model_id, revision):
    """the cache folder of a revision: a 40-hex sha directly, a branch / tag / PR ref through the cache's refs file;
    None when the cache has never resolved that name"""
    if _SHA.fullmatch(revision):
        return os.path.join(model_dir(model_id), 'snapshots', revision)
    try:
        with open(os.path.join(model_dir(model_id), 'refs', *revision.split('/')), encoding='utf-8') as f:
            sha = f.read().strip()
    except OSError:
        return None
    return os.path.join(model_dir(model_id), 'snapshots', sha) if _SHA.fullmatch(sha) else None


def _missing(folder, names):
    """the names not present in folder; a tuple entry lists alternatives (any one will do). os.path.exists follows
    the cache's symlinks, so a dangling link (a download that never finished) counts as missing"""
    out = []
    for n in names:
        alts = n if isinstance(n, tuple) else (n,)
        if not folder or not any(os.path.exists(os.path.join(folder, a)) for a in alts):
            out.append(alts[0])
    return tuple(out)


def _found(folder, alts):
    return next((a for a in alts if folder and os.path.exists(os.path.join(folder, a))), alts[0])


def resolve(model_id, revision=None):
    """what the cache holds for model_id at revision. revision None = earshot's pin (config.PINS), or 'main' for a
    model without one. Only a pinned model with no explicit revision may fall back to its legacy snapshot, and only
    when that snapshot is complete in the cache. Filesystem only, no network"""
    pin = config.PINS.get(model_id)
    if pin and revision in (None, pin['revision']):
        rev, files, weights = pin['revision'], tuple(pin['files']), (pin['weights'],)
        legacy = pin.get('legacy') if revision is None else None
    else:
        rev, files, weights, legacy, pin = revision or 'main', ('config.json',), SAFETENSORS, None, None
    folder = snapshot_dir(model_id, rev)
    missing = _missing(folder, files + (weights,))
    if not missing:
        return Spec(model_id, rev, 'cached', folder, _found(folder, weights), pinned=bool(pin))
    if legacy:
        old = snapshot_dir(model_id, legacy['revision'])
        if not _missing(old, files + (legacy['weights'],)):
            return Spec(model_id, rev, 'legacy', old, legacy['weights'], missing, pinned=True)
    state = 'incomplete' if folder and os.path.isdir(folder) else 'missing'
    return Spec(model_id, rev, state, None, weights[0], missing, pinned=bool(pin))


def fetch(spec):
    """download the wanted revision of spec.model_id into the earshot cache and return the new, complete Spec. For a
    pinned model exactly its files (safetensors weights, never the pickle); for another model its configs, tokenizer
    and safetensors weights. Refused while an offline flag is set"""
    flag = config.offline()
    where = (f'incomplete in the earshot cache (missing: {", ".join(spec.missing)})' if spec.state == 'incomplete' else
             f'not in the earshot cache ({config.HF_CACHE})')
    if flag:
        raise RuntimeError(f'{spec.label} is {where} and {flag} is set, so earshot will not download it. Run once '
                           f'without {flag} (`earshot doctor --fetch` downloads the models, ~1.1 GB), copy a cache '
                           'there, or point EARSHOT_HOME at one.')
    from huggingface_hub import snapshot_download   # a dependency of transformers
    pin = config.PINS.get(spec.model_id) if spec.pinned else None
    patterns = [*pin['files'], pin['weights']] if pin else list(CUSTOM_PATTERNS)
    config.ensure_dirs()
    try:
        snapshot_download(spec.model_id, revision=spec.revision, cache_dir=config.HF_CACHE, allow_patterns=patterns)
    except Exception as e:  # noqa: BLE001  (network, auth, unknown repo or revision: one readable error)
        raise RuntimeError(f'could not download {spec.label} into {config.HF_CACHE}: {type(e).__name__}: {e}') from e
    new = resolve(spec.model_id, spec.revision)
    if new.state != 'cached':
        hint = '' if pin else ': earshot loads safetensors weights only'
        raise RuntimeError(f'downloaded {spec.label}, but the cache still lacks {", ".join(new.missing)} '
                           f'({model_dir(spec.model_id)}){hint}')
    return new


def ensure(model_id, revision=None):
    """the Spec to load from: what the cache holds, or the wanted revision downloaded now when nothing local can
    load (one download, never while an offline flag is set)"""
    spec = resolve(model_id, revision)
    return spec if spec.local else fetch(spec)
