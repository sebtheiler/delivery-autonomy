import unittest
import numpy as np
from delivery_depth.core import measure


class DepthTests(unittest.TestCase):
    def test_millimeters(self):
        a = np.array([[1000, 2000], [0, 3000]], dtype='<u2')
        r = measure(a.tobytes(), 2, 2, 4, '16UC1')
        self.assertEqual(r['median_depth_m'], 2.0)
        self.assertEqual(r['valid_fraction'], 0.75)

    def test_big_endian_padding(self):
        data = np.array([1000, 2000, 9999, 3000, 4000, 9999], dtype='>u2').tobytes()
        self.assertEqual(measure(data, 2, 2, 6, '16UC1', True)['median_depth_m'], 2.5)

    def test_float_invalid(self):
        a = np.array([1, float('nan'), float('inf'), -1], dtype='<f4')
        r = measure(a.tobytes(), 2, 2, 8, '32FC1')
        self.assertEqual(r['status'], 'insufficient_depth')
        self.assertIsNone(r['median_depth_m'])

    def test_center_roi(self):
        a = np.full((4, 4), 9000, dtype='<u2')
        a[1:3, 1:3] = 500
        self.assertEqual(measure(a.tobytes(), 4, 4, 8, '16UC1', roi_size=2)['median_depth_m'], 0.5)

    def test_empty_and_bad_inputs(self):
        self.assertIsNone(measure(bytes(8), 2, 2, 4, '16UC1')['median_depth_m'])
        with self.assertRaises(ValueError):
            measure(bytes(2), 2, 2, 4, '16UC1')
        with self.assertRaises(ValueError):
            measure(bytes(8), 2, 2, 4, 'rgb8')


if __name__ == '__main__':
    unittest.main()
