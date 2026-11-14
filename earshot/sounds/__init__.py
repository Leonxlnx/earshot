"""The sounds earshot can make: the catalogue of designs (registry), the themes that make a set of them one family
(theme) and the modules that define them.

    from earshot import sounds
    y, meta = sounds.render('coin/bright', pitch='B5')
    sounds.rows(domain='ui')
"""
from .registry import AXES, DOMAINS, KINDS, Controls, Design, design, get, ids, load, render, rows  # noqa: F401
