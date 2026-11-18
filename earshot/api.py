"""The Python API in one namespace (models load lazily on first use and stay warm in this process).

    from earshot import api
    api.describe('whoosh.wav')
    api.match('key.ogg', 'a deep thocky mechanical keyboard press', alts=['a clicky blue switch'])
    api.find('typing.ogg', 'a single isolated key press')
    api.timeline('soundtrack.wav', video='film.mp4')
    api.sfx_render('whoosh_fast/crisp', 'w.wav', dur=0.8, apex=0.32)   # the sound bank (earshot.forge.bank)
    api.kit_render('audio/kit.json', 'audio/kit')                       # a film's kit: takes + kit.lock.json
    api.make('a soft success chime', 'success.ogg')                     # a sound made from words (earshot.make)
"""
import functools
import importlib

_FUNCS = {
    'describe': ('describe', 'describe'), 'match': ('match', 'match'), 'rank': ('search', 'rank'),
    'find': ('search', 'find'), 'compare': ('compare', 'compare'), 'ab': ('compare', 'ab'),
    'realness': ('realness', 'realness'), 'realness_many': ('realness', 'realness_many'), 'timeline': ('timeline', 'timeline'), 'sheet': ('sheet', 'sheet'),
    'critique': ('critique', 'critique'), 'profile': ('profile', 'profile'), 'measure': ('features', 'measure'),
    'llm_critique': ('llm', 'critique'), 'doctor': ('doctor', 'check'),
    # the sound bank and film kits (no models): earshot.forge.sfx / earshot.forge.kit
    'sfx_list': ('forge.sfx', 'sfx_list'), 'sfx_render': ('forge.sfx', 'sfx_render'),
    'sfx_check': ('forge.sfx', 'sfx_check'), 'kit_init': ('forge.kit', 'kit_init'),
    'kit_render': ('forge.kit', 'kit_render'), 'kit_check': ('forge.kit', 'kit_check'),
    'kit_audition': ('forge.kit', 'kit_audition'),
    # sounds made from a request in words (no models): earshot.make
    'make': ('make', 'make'), 'make_set': ('make', 'make_set'), 'sounds': ('make', 'sounds'),
    'pack': ('pack', 'pack'),
}


def _fn(name):
    """the function `name`; a package it needs that the base install leaves out raises MissingExtra (an ImportError)
    with the plain sentence of earshot.extras, on import or on the call, never a bare ModuleNotFoundError"""
    from . import extras
    mod, fn = _FUNCS[name]
    try:
        f = getattr(importlib.import_module(f'earshot.{mod}'), fn)
    except ImportError as e:
        msg = extras.plain(e, name)
        if msg is None:
            raise
        raise extras.MissingExtra(msg, name=getattr(e, 'name', None)) from e

    @functools.wraps(f)
    def call(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ImportError as e:
            msg = None if isinstance(e, extras.MissingExtra) else extras.plain(e, name)
            if msg is None:
                raise
            raise extras.MissingExtra(msg, name=getattr(e, 'name', None)) from e
    return call


def __getattr__(name):
    if name in _FUNCS:
        return _fn(name)
    raise AttributeError(name)


__all__ = sorted(_FUNCS)
