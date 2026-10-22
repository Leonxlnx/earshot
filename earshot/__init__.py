"""earshot: ears for coding agents.

Python API (models load lazily on first use and stay warm):

    import earshot
    earshot.describe('whoosh.wav')
    earshot.match('key.ogg', 'a deep thocky mechanical keyboard press', alts=['a clicky blue switch'])
    earshot.find('typing.ogg', 'a single isolated key press')
    earshot.timeline('soundtrack.wav', video='film.mp4')

or explicitly: `from earshot import api` / `from earshot.describe import describe`.
"""
import importlib

__version__ = '0.4.0'

# modules that are callable as their main function (see _callable.py)
_CALLABLE = {'describe', 'match', 'compare', 'realness', 'timeline', 'sheet', 'critique', 'profile'}
# functions whose name is not a module name
_FUNCS = {'rank': ('search', 'rank'), 'find': ('search', 'find'), 'ab': ('compare', 'ab'),
          'measure': ('features', 'measure'), 'llm_critique': ('llm', 'critique')}


def __getattr__(name):
    if name in _CALLABLE:
        return importlib.import_module(f'earshot.{name}')
    if name in _FUNCS:
        mod, fn = _FUNCS[name]
        return getattr(importlib.import_module(f'earshot.{mod}'), fn)
    raise AttributeError(name)  # anything else: normal submodule import
