import unittest
import numpy as np
from delivery_depth.cloud_core import filter_xyz


class CloudTests(unittest.TestCase):
    def test_invalid_and_out_of_range(self):
        points = [[1, 2, 1], [0, 0, 0], [0, 0, 5], [float('nan'), 0, 1], [0, 0, float('inf')]]
        np.testing.assert_array_equal(filter_xyz(points, stride=1), [[1, 2, 1]])

    def test_boundaries_and_stride(self):
        points = [[0, 0, .2], [1, 1, 1], [0, 0, 4]]
        self.assertEqual(len(filter_xyz(points, stride=2)), 2)

    def test_empty(self):
        self.assertEqual(filter_xyz([]).shape, (0, 3))

    def test_invalid_parameters(self):
        for args in [(2, 1, 1), (0, 4, 0), (0, float('nan'), 1)]:
            with self.assertRaises(ValueError):
                filter_xyz([], *args)
