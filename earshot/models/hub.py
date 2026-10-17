"""The model files earshot loads: pinned revisions in earshot's own Hugging Face cache, never fetched behind your back.

    from earshot.models import hub
    s = hub.resolve('laion/larger_clap_general')   # filesystem only; s.state: cached / legacy / incomplete / missing
    s = hub.ensure('laion/larger_clap_general')    # the same, plus one download of the pinned files if none can load

The models load only from a complete snapshot folder (EARSHOT_HOME/hf/models--org--name/snapshots/<sha>): transformers
gets a folder, never a repo id, so a cached load never asks the Hub, and a damaged cache is an error that names the
folder instead of a silent new download. A model that is not cached is downloaded once at its pinned revision
(config.PINS), with safetensors weights, unless an offline flag is set (config.offline()). earshot never sets or
clears HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE. Imports nothing heavy: earshot doctor uses it without torch.

EARSHOT_CLAP / EARSHOT_AST may also name a local model folder (config.json plus model.safetensors, or the
pytorch_model.bin of an older checkpoint, as transformers reads it): it loads from that folder as it is, with no
network and no revision, and earshot never downloads into it (an incomplete folder is an error naming what is missing).
"""
import hashlib
import os
import re
from dataclasses import dataclass

from .. import config

SAFETENSORS = ('model.safetensors', 'model.safetensors.index.json')   # one file, or the index of a sharded checkpoint
PICKLE = ('pytorch_model.bin', 'pytorch_model.bin.index.json')          # loaded only from a folder the user names
# An unpinned Hub model: its top-level configs, tokenizer files and safetensors weights, by name. snapshot_download
# matches allow_patterns with fnmatch, where '*' also matches '/', so bare '*.json' / '*.safetensors' globs would pull
# every subfolder and weight variant of a repo (onnx/, fp16 copies, other pipelines' components): possibly several GB.
CUSTOM_PATTERNS = ('config.json', 'preprocessor_config.json', 'feature_extractor_config.json', 'processor_config.json',
                   'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                   'vocab.json', 'vocab.txt', 'merges.txt', 'tokenizer.model', 'spiece.model',
                   'sentencepiece.bpe.model', 'model.safetensors', 'model.safetensors.index.json',
                   'model-?????-of-?????.safetensors')
LOCAL = 'local'   # Spec.revision of a local model folder
_SHA = re.compile(r'[0-9a-f]{40}')


@dataclass(frozen=True)
class Spec:
    """one model as earshot will load it"""
    model_id: str
    revision: str            # the revision earshot wants: its pin, an EARSHOT_*_REVISION override, 'main' or 'local'
    state: str               # cached | legacy (a pre-0.2.1 snapshot of the pin stands in) | incomplete | missing
    snapshot: str | None     # the complete local folder to load from (state cached or legacy), else None
    weights: str             # the weights file loaded or to download (pytorch_model.bin only for a legacy snapshot)
    missing: tuple = ()      # files of the wanted revision that are not in the cache
    pinned: bool = False     # the wanted revision is earshot's pin
    folder: bool = False     # model_id names a local model folder (revision 'local'), not a Hub repo id

    @property
    def local(self):
        return self.snapshot is not None

    @property
    def sha(self):
        """the commit of the snapshot that loads (None when nothing local can, and for a local model folder)"""
        return os.path.basename(self.snapshot) if self.snapshot and not self.folder else None

    @property
    def label(self):
        if self.folder:
            return f'the local model folder {self.model_id}'
        return f'{self.model_id}@{self.revision[:7] if _SHA.fullmatch(self.revision) else self.revision}'

    @property
    def cache_tag(self):
        """suffix for embedding caches: none for the pin and its legacy snapshot (the same tensors), else the commit,
        so embeddings of another checkpoint never mix with the pinned model's. A local folder: a hash of its path and
        of its weights file's size and time, so new weights in the same folder get a fresh cache"""
        if self.pinned:
            return ''
        if self.folder:
            p = _abs(self.model_id)
            try:
                st = os.stat(os.path.join(p, self.weights))
                key = f'{os.path.normcase(p)}|{st.st_size}|{st.st_mtime_ns}'
            except OSError:
                key = os.path.normcase(p)
            return '--local' + hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]
        return '--' + (self.sha or re.sub(r'[^0-9A-Za-z._-]', '_', self.revision))[:8]

    @property
    def cache_name(self):
        """one path component that names this model's embedding caches: slug + cache_tag. A Hub id keeps its name
        (org--name, unchanged for the default models); a local folder becomes local--<folder name>, never a path (a
        drive letter in it would leave the cache root on Windows)"""
        if self.folder:
            base = re.sub(r'[^0-9A-Za-z._-]', '_', os.path.basename(os.path.normpath(_abs(self.model_id))))[:40]
            return 'local--' + (base or 'model') + self.cache_tag
        return slug(self.model_id) + self.cache_tag


def _abs(path):
    return os.path.abspath(os.path.expanduser(path))


def slug(model_id):
    """a Hub repo id as one path component: org/name -> org--name"""
    return model_id.replace('/', '--')


def is_folder(model_id):
    """True when model_id names an existing local folder rather than a Hub repo id (transformers reads an existing
    folder the same way). earshot's pinned ids always mean the Hub model: a folder of that name in the working
    directory does not replace the pin"""
    return bool(model_id) and model_id not in config.PINS and os.path.isdir(os.path.expanduser(model_id))


def cache_name(model_id, revision=None):
    """resolve(model_id, revision).cache_name: the embedding-cache component of a model, filesystem only"""
    return resolve(model_id, revision).cache_name


def model_dir(model_id):
    return os.path.join(config.HF_CACHE, 'models--' + slug(model_id))


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


def _resolve_folder(model_id):
    """a local model folder as it is: complete with config.json and weights (safetensors preferred; the
    pytorch_model.bin of an older checkpoint loads as transformers would load it, torch weights_only), else
    incomplete. No revision, never downloaded into"""
    p = _abs(model_id)
    alts = SAFETENSORS + PICKLE
    missing = _missing(p, ('config.json', alts))
    if missing:
        return Spec(model_id, LOCAL, 'incomplete', None, _found(p, alts), missing, folder=True)
    return Spec(model_id, LOCAL, 'cached', p, _found(p, alts), folder=True)


def resolve(model_id, revision=None):
    """what the cache holds for model_id at revision. revision None = earshot's pin (config.PINS), or 'main' for a
    model without one. Only a pinned model with no explicit revision may fall back to its legacy snapshot, and only
    when that snapshot is complete in the cache. A model_id that names a local folder resolves to that folder
    (revision 'local'; a revision means nothing there and is ignored). Filesystem only, no network"""
    if is_folder(model_id):
        return _resolve_folder(model_id)
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
    pinned model exactly its files (safetensors weights, never the pickle); for another model its top-level configs,
    tokenizer and safetensors weights, by name. Refused while an offline flag is set, and always for a local folder"""
    if spec.folder:
        raise RuntimeError(f'{spec.label} lacks {" and ".join(spec.missing)}: earshot loads a local model folder as '
                           'it is and never downloads into it. Put config.json and model.safetensors (or the '
                           'pytorch_model.bin of an older checkpoint) there, or set EARSHOT_CLAP / EARSHOT_AST to a '
                           'Hub model id.')
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
    """the Spec to load from: what the cache (or a local model folder) holds, or the wanted revision downloaded now
    when nothing local can load (one download, never while an offline flag is set, never into a local folder)"""
    spec = resolve(model_id, revision)
    return spec if spec.local else fetch(spec)
