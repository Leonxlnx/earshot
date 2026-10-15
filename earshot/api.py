"""The Python API in one namespace (models load lazily on first use and stay warm in this process).

    from earshot import api
    api.describe('whoosh.wav')
    api.match('key.ogg', 'a deep thocky mechanical keyboard press', alts=['a clicky blue switch'])
    api.find('typing.ogg', 'a single isolated key press')
    api.timeline('soundtrack.wav', video='film.mp4')
"""
import importlib

_FUNCS = {
    'describe': ('describe', 'describe'), 'match': ('match', 'match'), 'rank': ('search', 'rank'),
    'find': ('search', 'find'), 'compare': ('compare', 'compare'), 'ab': ('compare', 'ab'),
    'realness': ('realness', 'realness'), 'realness_many': ('realness', 'realness_many'), 'timeline': ('timeline', 'timeline'), 'sheet': ('sheet', 'sheet'),
    'critique': ('critique', 'critique'), 'profile': ('profile', 'profile'), 'measure': ('features', 'measure'),
    'llm_critique': ('llm', 'critique'), 'doctor': ('doctor', 'check'),
}


def _fn(name):
    mod, fn = _FUNCS[name]
    return getattr(importlib.import_module(f'earshot.{mod}'), fn)


def __getattr__(name):
    if name in _FUNCS:
        return _fn(name)
    raise AttributeError(name)


__all__ = sorted(_FUNCS)
