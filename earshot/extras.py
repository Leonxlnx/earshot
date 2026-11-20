"""What the base install leaves out, and the plain sentence a call says when it needs it.

The base install makes sounds: make, make_set, sounds, the sound bank and film kits, pack, on the command line, in
Python and as MCP tools, with numpy, scipy, soundfile, soxr and numba. The other tools hear or measure sound files or
draw pictures, and need an extra (pyproject.toml):

    listen    torch, transformers, librosa: the listening models (CLAP, the AudioSet tagger): describe, match, rank,
              find, compare, ab, realness, timeline, critique, profile, make with listen
    measure   librosa: measure, find, profile's onsets, without the models
    images    matplotlib, pillow: sheet, gallery, picture frames of a video, plots
    mcp       the MCP server
    llm       requests: the optional Gemini critique
    all       all of them

A call that needs a missing one raises MissingExtra (an ImportError) whose message is one sentence with the command;
the command line and the MCP server print that sentence as their error.
"""
import importlib.util

# the import name of a package an extra brings -> (the extra, the package as pip names it)
EXTRA = {'torch': ('listen', 'torch'), 'transformers': ('listen', 'transformers'),
         'huggingface_hub': ('listen', 'huggingface_hub'), 'librosa': ('measure', 'librosa'),
         'matplotlib': ('images', 'matplotlib'), 'PIL': ('images', 'pillow'), 'mcp': ('mcp', 'mcp'),
         'pydantic': ('mcp', 'pydantic'), 'requests': ('llm', 'requests')}
# what each extra installs, as pyproject.toml declares it (a test keeps the two the same). The install command names
# the packages, not "earshot[extra]": it is true however earshot was installed (from a clone, from git, as a uv tool)
PACKAGES = {'listen': ('torch>=2.6', 'transformers>=4.40', 'librosa>=0.10'), 'measure': ('librosa>=0.10',),
            'images': ('matplotlib>=3.8', 'pillow>=10'), 'mcp': ('mcp>=2',), 'llm': ('requests>=2.31',)}
LISTEN = ('torch', 'transformers', 'librosa')      # the import names the listening tools need
# a command, tool or function of earshot -> the extra it needs (one that measures first asks for librosa, but
# without the models it would fail next: the message names the whole extra)
NEEDS = {**{c: 'listen' for c in ('describe', 'match', 'rank', 'find', 'compare', 'ab', 'realness', 'realness_many',
                                  'timeline', 'critique', 'profile')},
         'measure': 'measure', 'sheet': 'images', 'gallery': 'images', 'llm': 'llm', 'llm_critique': 'llm'}
IMPORTS = {'listen': LISTEN, 'measure': ('librosa',), 'images': ('matplotlib', 'PIL'), 'mcp': ('mcp', 'pydantic'),
           'llm': ('requests',)}


class MissingExtra(ImportError):
    """a call needs a package the base install leaves out; str(e) is one plain sentence with the install command"""


SOURCE = 'git+https://github.com/Leonxlnx/earshot'   # where earshot installs from


def command(extra):
    """the pip command that adds an extra's packages to the environment earshot runs in; for the listen extra also
    the install of earshot with it (the one line every hint of the listen extra prints)"""
    line = 'pip install ' + ' '.join(f'"{p}"' for p in PACKAGES[extra])
    if extra == 'listen':
        return line + f' (the listen extra; or: pip install "earshot[listen] @ {SOURCE}")'
    return line + f' (the {extra} extra)'


def sentence(module, what=None):
    """the plain sentence for a missing package (its import name)"""
    top = str(module).split('.')[0]
    extra, pkg = EXTRA.get(top, (None, top))
    head = f'{what} needs {pkg}' if what else f'this needs {pkg}'
    if extra is None:
        return f'{head}, which is not installed: pip install {pkg}'
    return f'{head}, which the base install of earshot leaves out: {command(extra)}'


def missing(*modules):
    """the import names among `modules` that are not installed"""
    out = []
    for m in modules:
        try:
            if importlib.util.find_spec(m) is None:
                out.append(m)
        except (ImportError, ValueError):
            out.append(m)
    return out


def need(*modules, what=None):
    """raise MissingExtra for the first of `modules` (import names) that is not installed; checks the file system
    only, imports nothing"""
    gone = missing(*modules)
    if gone:
        raise MissingExtra(sentence(gone[0], what), name=gone[0])


def check(name):
    """raise MissingExtra when the command, tool or function `name` (NEEDS) needs an extra that is not all there;
    imports nothing"""
    extra = NEEDS.get(name)
    gone = missing(*IMPORTS[extra]) if extra else []
    if gone:
        pkgs = ', '.join(EXTRA[m][1] for m in gone)
        raise MissingExtra(f'earshot {name} needs {pkgs}, which the base install of earshot leaves out: '
                           f'{command(extra)}', name=gone[0])


def plain(e, name=None):
    """the plain sentence for an exception that a missing extra caused (MissingExtra, or a ModuleNotFoundError of a
    package an extra brings, also as the cause of another error), else None. name: the command or tool that failed,
    whose whole extra the sentence then names"""
    if name and _caused(e):
        try:
            check(name)
        except MissingExtra as m:
            return str(m)
    return _caused(e)


def _caused(e):
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, MissingExtra):
            return str(e)
        if isinstance(e, ModuleNotFoundError) and (e.name or '').split('.')[0] in EXTRA:
            return sentence(e.name)
        e = e.__cause__ or e.__context__
    return None
