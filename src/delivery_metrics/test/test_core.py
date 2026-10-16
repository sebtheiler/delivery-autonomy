import json
import tempfile
import unittest
from pathlib import Path

from delivery_metrics.core import Measurement, Store


class MeasurementTests(unittest.TestCase):
    def test_motion_and_stop_include_reverse_distance(self):
        measurement = Measurement(10, max_gap=1)
        for stamp, speed in [(10, -2), (11, -2), (12, 0), (13, 0)]:
            measurement.sample(stamp, speed)
        result = measurement.result(13)
        self.assertEqual(result['encoder_distance_m'], 3)
        self.assertEqual(result['moving_seconds'], 2)
        self.assertEqual(result['stopped_seconds'], 1)
        self.assertEqual(result['odom_coverage_fraction'], 1)

    def test_gaps_and_boundaries_are_unknown_not_stopped(self):
        measurement = Measurement(0, max_gap=1)
        for stamp in (1, 2, 8, 9):
            measurement.sample(stamp, 1)
        result = measurement.result(10)
        self.assertEqual(result['encoder_distance_m'], 2)
        self.assertEqual(result['unobserved_seconds'], 8)
        self.assertEqual(result['stopped_seconds'], 0)

    def test_invalid_and_out_of_order_samples_do_not_add_distance(self):
        measurement = Measurement(10, max_gap=1)
        for stamp, speed in [(9, 1), (10, 1), (10, 1), (11, float('nan')), (11, 1)]:
            measurement.sample(stamp, speed)
        self.assertEqual(measurement.rejected, 3)
        self.assertEqual(measurement.result(11)['encoder_distance_m'], 1)

    def test_no_samples_has_zero_coverage(self):
        result = Measurement(10).result(15)
        self.assertEqual(result['unobserved_seconds'], 5)
        self.assertEqual(result['odom_coverage_fraction'], 0)

    def test_durable_record_and_interrupted_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.sqlite3'
            store = Store(path)
            store.save({'record_id': 'finished', 'status': 'recording'})
            store.save({'record_id': 'finished', 'status': 'completed'})
            store.save({'record_id': 'unfinished', 'status': 'recording'})
            store.close()
            store = Store(path)
            records = {key: json.loads(value) for key, value in
                       store.connection.execute('SELECT record_id, payload FROM records')}
            self.assertEqual(len(records), 2)
            self.assertEqual(records['finished']['status'], 'completed')
            self.assertEqual(records['unfinished']['status'], 'interrupted')
            store.close()


if __name__ == '__main__':
    unittest.main()
