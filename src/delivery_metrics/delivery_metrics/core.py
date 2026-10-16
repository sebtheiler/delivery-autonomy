"""ROS-independent measurement and durable storage helpers."""
import json
import math
import sqlite3
from pathlib import Path


class Measurement:
    def __init__(self, start, speed_threshold=0.05, max_gap=0.5):
        self.start = start
        self.threshold = speed_threshold
        self.max_gap = max_gap
        self.previous = None
        self.distance = 0.0
        self.moving = 0.0
        self.stopped = 0.0
        self.samples = 0
        self.rejected = 0

    def sample(self, stamp, speed):
        if not math.isfinite(stamp) or not math.isfinite(speed) or stamp < self.start:
            self.rejected += 1
            return
        if self.previous is not None:
            old_stamp, old_speed = self.previous
            dt = stamp - old_stamp
            if dt <= 0:
                self.rejected += 1
                return
            if dt <= self.max_gap:
                self.distance += (abs(speed) + abs(old_speed)) * 0.5 * dt
                if max(abs(speed), abs(old_speed)) >= self.threshold:
                    self.moving += dt
                else:
                    self.stopped += dt
        self.previous = stamp, speed
        self.samples += 1

    def result(self, end):
        duration = max(0.0, end - self.start)
        covered = self.moving + self.stopped
        return {
            'elapsed_seconds': duration,
            'encoder_distance_m': self.distance,
            'moving_seconds': self.moving,
            'stopped_seconds': self.stopped,
            'unobserved_seconds': max(0.0, duration - covered),
            'odom_coverage_fraction': min(1.0, covered / duration) if duration else 0.0,
            'odom_samples': self.samples, 'rejected_samples': self.rejected,
        }


class Store:
    def __init__(self, path):
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(Path(path).expanduser()))
        self.connection.execute('''CREATE TABLE IF NOT EXISTS records (
            record_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
            uploaded INTEGER NOT NULL DEFAULT 0)''')
        # A previous process may have crashed during a traversal. Keep the
        # incomplete record, but make sure analysis cannot mistake it for success.
        for record_id, payload in self.connection.execute('SELECT record_id, payload FROM records').fetchall():
            row = json.loads(payload)
            if row.get('status') == 'recording':
                row['status'] = 'interrupted'
                self.save(row)

    def save(self, record):
        payload = json.dumps(record, allow_nan=False)
        with self.connection:
            self.connection.execute('''INSERT INTO records (record_id, payload, uploaded)
                VALUES (?, ?, 0) ON CONFLICT(record_id) DO UPDATE SET
                payload=excluded.payload, uploaded=0''', (record['record_id'], payload))
        return payload

    def close(self):
        self.connection.close()
