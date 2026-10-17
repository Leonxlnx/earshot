"""Paths, model ids and CPU limits. Everything can be overridden with environment variables.

EARSHOT_HOME     cache root (models, embeddings, decoded audio). Default: <repo>/.cache in a source checkout,
                 else ~/.cache/earshot.
EARSHOT_THREADS  torch CPU threads per process (default: min(4, cores // 2)); earshot never uses a GPU.
EARSHOT_CLAP     CLAP model: a Hub id (default laion/larger_clap_general) or a local model folder (config.json plus
                 model.safetensors), loaded from disk as it is, never downloaded into
EARSHOT_AST      AudioSet tagger: a Hub id (default MIT/ast-finetuned-audioset-10-10-0.4593) or a local model folder
EARSHOT_CLAP_REVISION, EARSHOT_AST_REVISION
                 Hub revision (commit sha, branch or tag) to use instead of the pinned one (PINS below); a custom
                 model id without a revision uses 'main'
EARSHOT_OFFLINE  1 = never touch the network (models must already be in the cache). HF_HUB_OFFLINE=1 and
                 TRANSFORMERS_OFFLINE=1 mean the same to earshot, which never sets or clears any of the three
EARSHOT_PRECISION  bf16 (default; used only on CPUs with native bf16, AVX512-BF16 / AMX, else fp32: on the dev laptop
                 1.4x faster, embeddings within cosine 0.9993 of fp32, eval/semantics/precision.json) | fp32
"""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(PKG_DIR, 'data')


def _default_home():
    if os.environ.get('EARSHOT_HOME'):
        return os.path.abspath(os.environ['EARSHOT_HOME'])
    if os.path.exists(os.path.join(REPO_DIR, 'pyproject.toml')) and os.access(REPO_DIR, os.W_OK):
        return os.path.join(REPO_DIR, '.cache')
    return os.path.join(os.path.expanduser('~'), '.cache', 'earshot')


HOME = _default_home()
HF_CACHE = os.path.join(HOME, 'hf')
EMB_CACHE = os.path.join(HOME, 'emb')
DEC_CACHE = os.path.join(HOME, 'decoded')

CLAP_MODEL = os.environ.get('EARSHOT_CLAP', 'laion/larger_clap_general')
AST_MODEL = os.environ.get('EARSHOT_AST', 'MIT/ast-finetuned-audioset-10-10-0.4593')
CLAP_REVISION = os.environ.get('EARSHOT_CLAP_REVISION') or None
AST_REVISION = os.environ.get('EARSHOT_AST_REVISION') or None
PRECISION = os.environ.get('EARSHOT_PRECISION', 'bf16').lower()

# Pinned model files (earshot.models.hub downloads exactly these, at exactly this commit, weights as safetensors).
# - laion/larger_clap_general @ 16c8cc3: the Hub's refs/pr/2, the safetensors conversion bot's commit (2024-05-06) on
#   top of main ada0c23 (2023-10-31). Same files as main plus model.safetensors; main itself has only the pickle
#   pytorch_model.bin. The Hub keeps PR refs, also after a close, and the sha stays valid if the PR is merged
#   (EARSHOT_CLAP_REVISION overrides it). Caches made before earshot 0.2.1 hold ada0c23: that 'legacy' snapshot is
#   still loaded, from the local cache only and never downloaded (the same tensors, so cached embeddings stay valid),
#   until the pinned one is there (earshot doctor --fetch).
# - MIT/ast-finetuned-audioset-10-10-0.4593 @ f826b80: main (2023-09-06), which already has model.safetensors.
# 'files' are the config / tokenizer files next to the weights (the legacy snapshot has the same ones).
PINS = {
    'laion/larger_clap_general': {
        'revision': '16c8cc3159a3c8a31e8ff5ef1f66b0d9ab3667db', 'weights': 'model.safetensors',
        'files': ('config.json', 'preprocessor_config.json', 'merges.txt', 'special_tokens_map.json',
                  'tokenizer.json', 'tokenizer_config.json', 'vocab.json'),
        'legacy': {'revision': 'ada0c23a36c4e8582805bb38fec3905903f18b41', 'weights': 'pytorch_model.bin'}},
    'MIT/ast-finetuned-audioset-10-10-0.4593': {
        'revision': 'f826b80d28226b62986cc218e5cec390b1096902', 'weights': 'model.safetensors',
        'files': ('config.json', 'preprocessor_config.json')},
}
OFFLINE_FLAGS = ('EARSHOT_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')


def offline():
    """the name of the first offline flag that is set (1 / true / yes / on, any case, as huggingface_hub reads
    them), else None. Read at call time: earshot never downloads while one is set"""
    for k in OFFLINE_FLAGS:
        if os.environ.get(k, '').strip().upper() in ('1', 'TRUE', 'YES', 'ON'):
            return k
    return None


OFFLINE = offline() is not None   # the state at import, kept for old callers; earshot itself calls offline()


def threads():
    if os.environ.get('EARSHOT_THREADS'):
        return max(1, int(os.environ['EARSHOT_THREADS']))
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def ensure_dirs():
    for d in (HOME, HF_CACHE, EMB_CACHE, DEC_CACHE):
        os.makedirs(d, exist_ok=True)
