"""earshot: ears for coding agents.

Python API (models load lazily on first use and stay warm):

    import earshot
    earshot.describe('whoosh.wav')
    earshot.match('key.ogg', 'a deep thocky mechanical keyboard press', alts=['a clicky blue switch'])
    earshot.find('typing.ogg', 'a single isolated key press')
    earshot.timeline('soundtrack.wav', video='film.mp4')

or explicitly: `from earshot import api` / `from earshot.describe import describe`.
Sounds made from a request in words (no models): `from earshot.make import make; make('a soft success chime', 'a.wav')`.
Making sounds needs only the base install; the listening and measuring functions need an extra (earshot.extras), and
without it they raise earshot.extras.MissingExtra, whose message is the install command.
"""
import importlib

__version__ = '1.0.0'

# modules that are callable as their main function (see _callable.py)
_CALLABLE = {'describe', 'match', 'compare', 'realness', 'timeline', 'sheet', 'critique', 'profile'}
# functions whose name is not a module name
_FUNCS = {'rank': ('search', 'rank'), 'find': ('search', 'find'), 'ab': ('compare', 'ab'),
          'measure': ('features', 'measure'), 'llm_critique': ('llm', 'critique')}


def __getattr__(name):
    if name in _CALLABLE or name in _FUNCS:
        from . import extras
        extras.check(name)      # earshot.describe without the listen extra: one sentence with the install command
    if name in _CALLABLE:
        return importlib.import_module(f'earshot.{name}')
    if name in _FUNCS:
        mod, fn = _FUNCS[name]
        return getattr(importlib.import_module(f'earshot.{mod}'), fn)
    raise AttributeError(name)  # anything else: normal submodule import
