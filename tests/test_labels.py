import unittest
from bullseye_tracker.labels import convert_label_text, convert_line, target_class_ids


class TargetClassTest(unittest.TestCase):
    def test_shooting_dataset(self):
        self.assertEqual(target_class_ids(["bullet", "target", "black_contour"]), {1})

    def test_archery_drops_arrow(self):
        self.assertEqual(target_class_ids({0: "arrow", 1: "Target"}), {1})

    def test_bullseye_names(self):
        self.assertEqual(target_class_ids(["Bullseye", "target-hole", "person"]), {0})

    def test_explicit_names(self):
        self.assertEqual(target_class_ids(["a", "b", "c"], {"b", "c"}), {1, 2})


class ConvertLineTest(unittest.TestCase):
    def test_bbox_remapped_to_zero(self):
        self.assertEqual(convert_line("1 0.5 0.5 0.2 0.4", {1}), "0 0.500000 0.500000 0.200000 0.400000")

    def test_other_class_dropped(self):
        self.assertIsNone(convert_line("0 0.5 0.5 0.2 0.4", {1}))

    def test_polygon_to_bbox(self):
        self.assertEqual(convert_line("2 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6", {2}),
                         "0 0.200000 0.400000 0.200000 0.400000")

    def test_clipped_to_image(self):
        self.assertEqual(convert_line("0 0.95 0.5 0.2 0.2", {0}), "0 0.925000 0.500000 0.150000 0.200000")

    def test_malformed(self):
        self.assertIsNone(convert_line("1 0.5 0.5", {1}))
        self.assertIsNone(convert_line("x 0.5 0.5 0.1 0.1", {1}))
        self.assertIsNone(convert_line("1 0.5 0.5 0 0.1", {1}))

    def test_text(self):
        text = "1 0.5 0.36 0.77 0.57\n2 0.5 0.36 0.3 0.22\n0 0.87 0.26 0.04 0.03\n"
        self.assertEqual(convert_label_text(text, {1}), ["0 0.500000 0.360000 0.770000 0.570000"])


if __name__ == "__main__":
    unittest.main()
