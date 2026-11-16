"""Theme edits must not change presets or other rendering requests."""
from copy import deepcopy

from earshot.sounds import theme


def test_named_theme_axes_do_not_alias_presets_or_other_requests():
    for name in theme.THEMES:
        saved = deepcopy(theme.THEMES[name])
        try:
            edited = theme.get(name)
            other = theme.get(name)
            edited['axes']['bright'] = 0.99
            assert other['axes'] == saved['axes'], name
            assert theme.get(name)['axes'] == saved['axes'], name
            assert theme.THEMES[name] == saved, name
            assert edited['axes']['bright'] == 0.99
        finally:
            # Keep the suite isolated even when run against the broken baseline.
            theme.THEMES[name] = saved


def test_named_theme_axes_can_be_cleared_without_resetting_the_preset():
    for name in theme.THEMES:
        saved = deepcopy(theme.THEMES[name])
        try:
            edited = theme.get(name)
            edited['axes'].clear()
            assert theme.get(name)['axes'] == saved['axes'], name
            assert theme.THEMES[name] == saved, name
        finally:
            theme.THEMES[name] = saved


def test_dict_theme_overrides_are_independent_and_preserve_values():
    override = {'base': 'wood', 'key': 'D', 'axes': {'hard': 0.2}}
    saved = deepcopy(override)
    result = theme.get(override)
    assert result['axes'] == {'bright': -0.1, 'ring': -0.2, 'hard': 0.2}
    assert result['key'] == 'D'
    assert result['voice'] == 'marimba'
    result['axes']['hard'] = -0.8
    assert override == saved
    assert theme.get(None) is None
    assert theme.get({'key': 'E'}) == {'key': 'E', 'axes': {}}
