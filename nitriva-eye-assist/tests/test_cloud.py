"""Cloud startup and real MediaPipe request-path regression checks."""
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient

from src.server import app
from src.tracker import tracker


class TestCloudDeployment(unittest.TestCase):
    def test_browser_startup_does_not_open_camera_or_run_synthetic_loop(self):
        with patch.dict(os.environ, {"NITRIVA_CAMERA_MODE": "browser"}):
            with patch("src.tracker.cv2.VideoCapture") as camera:
                with patch.object(tracker, "_generate_synthetic_frame") as synthetic:
                    with TestClient(app) as client:
                        self.assertEqual(client.get("/healthz").status_code, 200)
                        self.assertIsNone(tracker.capture_thread)
                        self.assertTrue(tracker.simulation_mode)
                    camera.assert_not_called()
                    synthetic.assert_not_called()
        self.assertFalse(tracker.is_running)

    def test_model_failure_is_unhealthy(self):
        with TestClient(app) as client:
            with patch.object(tracker, "detector", None):
                self.assertEqual(client.get("/healthz").status_code, 503)

    def test_static_api_and_real_websocket_frame_inference(self):
        # A blank JPEG exercises decoding and the actual model without a webcam.
        ok, jpeg = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
        self.assertTrue(ok)
        with patch.dict(os.environ, {"NITRIVA_CAMERA_MODE": "browser"}):
            with TestClient(app) as client:
                self.assertIn("Nitriva", client.get("/").text)
                self.assertEqual(client.get("/static/app.js").status_code, 200)
                self.assertEqual(client.post("/api/login", json={"key": "1234"}).status_code, 200)
                self.assertIn("water", client.post("/api/speak", json={"keywords": ["Water"]}).json()["sentence"])
                with client.websocket_connect("/ws/telemetry") as ws:
                    ws.send_bytes(jpeg.tobytes())
                    data = ws.receive_json()
                    self.assertTrue(data["frame_processed"])
                    self.assertFalse(data["face_detected"])
                    self.assertIn("gaze_x", data)
                    # A second frame verifies the connection remains usable.
                    import base64
                    ws.send_json({"type": "frame", "data": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()})
                    self.assertTrue(ws.receive_json()["frame_processed"])
                self.assertEqual(client.post("/api/process_frame", content=jpeg.tobytes()).status_code, 200)

    def test_blueprint_uses_docker_and_real_health_endpoint(self):
        with open("render.yaml", encoding="utf-8") as source:
            service = yaml.safe_load(source)["services"][0]
        self.assertEqual(service["runtime"], "docker")
        self.assertEqual(service["healthCheckPath"], "/healthz")
        self.assertEqual(service["plan"], "free")


if __name__ == "__main__":
    os.environ["NITRIVA_CAMERA_MODE"] = "browser"
    unittest.main()
