"""Paths, model ids and CPU limits. Everything can be overridden with environment variables.

EARSHOT_HOME     cache root (models, embeddings, decoded audio). Default: <repo>/.cache in a source checkout,
                 else ~/.cache/earshot.
EARSHOT_THREADS  torch CPU threads per process (default: min(4, cores // 2)); earshot never uses a GPU.
EARSHOT_CLAP     CLAP model id (default laion/larger_clap_general)
EARSHOT_AST      AudioSet tagger id (default MIT/ast-finetuned-audioset-10-10-0.4593)
EARSHOT_OFFLINE  1 = never touch the network (models must already be in the cache)
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
OFFLINE = os.environ.get('EARSHOT_OFFLINE', '0') == '1'
PRECISION = os.environ.get('EARSHOT_PRECISION', 'bf16').lower()


def threads():
    if os.environ.get('EARSHOT_THREADS'):
        return max(1, int(os.environ['EARSHOT_THREADS']))
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def ensure_dirs():
    for d in (HOME, HF_CACHE, EMB_CACHE, DEC_CACHE):
        os.makedirs(d, exist_ok=True)
