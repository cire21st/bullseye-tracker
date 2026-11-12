import unittest
from bullseye_tracker.geometry import Detection, pick_primary


class CoordinatesTest(unittest.TestCase):
    def test_centered_object(self):
        d = Detection((300, 220, 340, 260), 0.9, 0, "bullseye")
        self.assertEqual(d.coordinates(640, 480), dict(cx=320, cy=240, error_x=0, error_y=0))

    def test_top_left_and_subpixel(self):
        d = Detection((0, 0, 21, 31), 0.9, 0, "bullseye")
        self.assertEqual(d.coordinates(640, 480),
                         dict(cx=10.5, cy=15.5, error_x=-309.5, error_y=-224.5))

    def test_bottom_right_actual_resolution(self):
        d = Detection((1000, 600, 1200, 700), 0.9, 0, "bullseye", 7)
        self.assertEqual(d.coordinates(1280, 720),
                         dict(cx=1100, cy=650, error_x=460, error_y=290))


class PickPrimaryTest(unittest.TestCase):
    a = Detection((0, 0, 10, 10), 0.5, 0, "bullseye", 1)
    b = Detection((20, 20, 30, 30), 0.9, 0, "bullseye", 2)

    def test_empty(self):
        self.assertIsNone(pick_primary([], 1))

    def test_highest_confidence_without_history(self):
        self.assertIs(pick_primary([self.a, self.b]), self.b)

    def test_keeps_previous_track(self):
        self.assertIs(pick_primary([self.a, self.b], previous_track_id=1), self.a)

    def test_previous_track_lost(self):
        self.assertIs(pick_primary([self.a, self.b], previous_track_id=9), self.b)


if __name__ == "__main__":
    unittest.main()
