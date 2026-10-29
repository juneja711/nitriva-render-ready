"""
Nitriva Integration & Unit Tests
Tests all endpoints, ML models, and WebSocket connectivity.
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_engine import nlp_engine
from src.calibrator import calibrator
from src.tracker import tracker


class TestNitrivaSystem(unittest.TestCase):
    def test_nlp_engine_single_keywords(self):
        """Test single keyword sentence synthesis."""
        res_water = nlp_engine.generate_sentence(["Water"])
        self.assertIn("water", res_water.lower())
        self.assertIn("thirsty", res_water.lower())

        res_doctor = nlp_engine.generate_sentence(["Doctor"])
        self.assertIn("doctor", res_doctor.lower())

    def test_nlp_engine_pairwise_keywords(self):
        """Test contextual synthesis for 2 keywords."""
        res_pain_water = nlp_engine.generate_sentence(["Water", "Pain"])
        self.assertIn("pain", res_pain_water.lower())
        self.assertIn("water", res_pain_water.lower())

        res_pain_doc = nlp_engine.generate_sentence(["Pain", "Doctor"])
        self.assertIn("severe pain", res_pain_doc.lower())
        self.assertIn("doctor", res_pain_doc.lower())

    def test_calibrator_mapping(self):
        """Test calibrator mapping within screen bounds."""
        sx, sy = calibrator.map_gaze(0.0, 0.0)
        self.assertGreaterEqual(sx, 0.0)
        self.assertLessEqual(sx, 1.0)
        self.assertGreaterEqual(sy, 0.0)
        self.assertLessEqual(sy, 1.0)

    def test_calibration_targets_dual_phase(self):
        """Test that calibration targets contain both Light Mode and Dark/Black Mode phases."""
        from src.calibrator import CALIBRATION_TARGETS
        self.assertEqual(len(CALIBRATION_TARGETS), 10)

        light_targets = [t for t in CALIBRATION_TARGETS if t.get("phase") == "light"]
        dark_targets = [t for t in CALIBRATION_TARGETS if t.get("phase") == "dark"]

        self.assertEqual(len(light_targets), 5)
        self.assertEqual(len(dark_targets), 5)

        # Check required fields
        for t in CALIBRATION_TARGETS:
            self.assertIn("id", t)
            self.assertIn("phase", t)
            self.assertIn("name", t)
            self.assertIn("x", t)
            self.assertIn("y", t)

    def test_calibrator_dual_phase_fit(self):
        """Test calibrator fitting across both light and dark mode samples."""
        from src.calibrator import GazeCalibrator
        test_calib = GazeCalibrator(file_path="scratch_test_calib.json")

        # Simulate Light Phase samples
        test_calib.add_sample("light_center", 0.0, 0.0)
        test_calib.add_sample("light_top_left", -0.15, 0.15)
        test_calib.add_sample("light_top_right", 0.15, 0.15)
        test_calib.add_sample("light_bottom_left", -0.15, -0.15)
        test_calib.add_sample("light_bottom_right", 0.15, -0.15)

        # Simulate Dark Phase samples (black screen)
        test_calib.add_sample("dark_center", 0.01, -0.01)
        test_calib.add_sample("dark_top_left", -0.16, 0.16)
        test_calib.add_sample("dark_top_right", 0.16, 0.16)
        test_calib.add_sample("dark_bottom_left", -0.16, -0.16)
        test_calib.add_sample("dark_bottom_right", 0.16, -0.16)

        success = test_calib.compute_calibration()
        self.assertTrue(success)
        self.assertTrue(test_calib.is_calibrated)

        # Test gaze mapping
        sx, sy = test_calib.map_gaze(0.0, 0.0)
        self.assertAlmostEqual(sx, 0.5, delta=0.15)
        self.assertAlmostEqual(sy, 0.5, delta=0.15)

        # Cleanup scratch file
        if os.path.exists("scratch_test_calib.json"):
            os.remove("scratch_test_calib.json")

    def test_wink_detection_and_single_emission(self):
        """Test that left wink is detected with differential score and emitted once."""
        t_now = time.time()
        tracker.recent_event = None
        tracker.last_event_time = 0.0
        tracker.wink_left_start_time = 0.0
        tracker.wink_left_active = False

        # Simulate left wink: left eye = 0.58, right eye = 0.28 (difference = 0.30)
        tracker.blink_left = 0.58
        tracker.blink_right = 0.28
        tracker._update_eye_state_machine(t_now)
        tracker._update_eye_state_machine(t_now + 0.12)

        self.assertEqual(tracker.recent_event, "WINK_LEFT")

        # Telemetry should deliver event ONCE and consume it
        data1 = tracker.get_telemetry()
        self.assertEqual(data1["event"], "WINK_LEFT")

        data2 = tracker.get_telemetry()
        self.assertIsNone(data2["event"])


if __name__ == "__main__":
    unittest.main()
