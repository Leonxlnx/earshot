"""Walking distance uses the same composed space axis as the rendered room."""
import unittest

import numpy as np

from earshot.sounds import learned, theme


class TestWalkEffectiveSpace(unittest.TestCase):
    def assert_same_take(self, left, right, **extra):
        a, ma = learned.perform('step/pebbles', seed=0, **extra, **left)
        b, mb = learned.perform('step/pebbles', seed=0, **extra, **right)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ma, mb)

    def test_nonwalk_theme_offset(self):
        a, ma = learned.perform('door/knock', seed=0, space=-.5)
        b, mb = learned.perform('door/knock', seed=0, theme={'axes': {'space': -.5}})
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ma, mb)

    def test_whole_negative_theme_offset(self):
        self.assert_same_take({'space': -.5}, {'theme': {'axes': {'space': -.5}}})

    def test_sequence_negative_theme_offset(self):
        self.assert_same_take({'space': -.5}, {'theme': {'axes': {'space': -.5}}}, events=3)

    def test_cancelled_negative_at_zero(self):
        self.assert_same_take({'space': 0}, {'space': -.5, 'theme': {'axes': {'space': .5}}}, events=3)

    def test_positive_composition_with_room(self):
        self.assert_same_take({'space': .5, 'theme': {'space': 'hall'}},
                              {'space': -.5, 'theme': {'space': 'hall', 'axes': {'space': 1}}}, events=3)

    def test_clipped_negative_named_base(self):
        self.assert_same_take({'space': -1, 'theme': theme.get('wood')},
                              {'space': -.75, 'theme': theme.get({'base': 'wood', 'axes': {'space': -.75}})})


if __name__ == '__main__':
    unittest.main()
