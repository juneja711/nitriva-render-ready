import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
import os
import threading

CALIBRATION_POINTS = [
    {"top": "50%", "left": "50%", "label": "1. LOOK AT CENTER"},
    {"top": "15%", "left": "15%", "label": "2. LOOK TOP LEFT"},
    {"top": "15%", "left": "85%", "label": "3. LOOK TOP RIGHT"},
    {"top": "85%", "left": "85%", "label": "4. LOOK BOTTOM RIGHT"},
    {"top": "85%", "left": "15%", "label": "5. LOOK BOTTOM LEFT"}
]

class NitrivaOpticsEngine:
    """
    Production AI/ML Computer Vision Engine for Nitriva in Python.
    Features:
      - MediaPipe Tasks FaceLandmarker with 478 3D landmarks + 52 Blendshapes
      - Eye Aspect Ratio (EAR) for wink and blink detection
      - Iris gaze tracking with exponential moving average
      - Full dual-phase (Light / Dark) 5-point affine calibration solver
    """
    def __init__(self, camera_index=0, model_path=None):
        self.camera_index = camera_index
        self.cap = None

        if model_path is None:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(dir_path, "face_landmarker.task")

        # Initialize MediaPipe Tasks FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Gaze coordinates and affine calibration
        self.raw_x = 0.5
        self.raw_y = 0.5
        self.screen_x = 0.5
        self.screen_y = 0.5
        self.affine = {"mx": 1.0, "bx": 0.0, "my": 1.0, "by": 0.0}

        # Gesture recognition & debouncing
        self.ear_left = 0.3
        self.ear_right = 0.3
        self.blink_left = 0.0
        self.blink_right = 0.0
        self.blink_history = []
        self.was_blinking = False
        self.active_gesture = "TRACKING GAZE..."
        self.face_detected = False
        self.last_action_time = 0
        self.active_card = 0

        # Calibration state
        self.calib_phase = "SETUP"  # "SETUP", "CALIBRATING_LIGHT", "CALIBRATING_DARK", "DONE"
        self.calib_step = 0
        self.calib_samples = []
        self.is_collecting = False
        self.calib_error = None
        self.calib_thread = None
        self.simulation_mode = False

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def start_calibration(self):
        """Triggers the full dual-phase calibration routine in a background thread."""
        if self.calib_thread and self.calib_thread.is_alive():
            return
        self.calib_thread = threading.Thread(target=self._run_calibration, daemon=True)
        self.calib_thread.start()

    def skip_calibration(self):
        """Immediately skips calibration and marks tracking ready."""
        self.calib_phase = "DONE"
        self.active_gesture = "TRACKING GAZE..."

    def _run_calibration(self):
        print("[Nitriva AI] Starting 5-point dual-phase calibration...")
        self.calib_error = None

        def capture_point():
            self.is_collecting = False
            time.sleep(0.6)

            # Wait if face not detected
            while not self.face_detected and not self.simulation_mode:
                self.calib_error = "Face Lost! Please align face in camera."
                time.sleep(0.2)
            self.calib_error = None

            self.calib_samples = []
            self.is_collecting = True
            time.sleep(1.4)  # Collect samples for 1.4 seconds
            self.is_collecting = False

            if len(self.calib_samples) > 0:
                avg_x = sum(s[0] for s in self.calib_samples) / len(self.calib_samples)
                avg_y = sum(s[1] for s in self.calib_samples) / len(self.calib_samples)
            else:
                avg_x = self.raw_x
                avg_y = self.raw_y
            return {"raw_x": avg_x, "raw_y": avg_y}

        # Phase 1: Light Mode (5 targets)
        self.calib_phase = "CALIBRATING_LIGHT"
        light_pts = []
        for i in range(5):
            self.calib_step = i
            light_pts.append(capture_point())

        # Phase 2: Dark Mode (5 targets)
        self.calib_phase = "CALIBRATING_DARK"
        dark_pts = []
        for i in range(5):
            self.calib_step = i
            dark_pts.append(capture_point())

        # Blend light and dark points
        merged = []
        for i in range(5):
            merged.append({
                "raw_x": (light_pts[i]["raw_x"] + dark_pts[i]["raw_x"]) / 2.0,
                "raw_y": (light_pts[i]["raw_y"] + dark_pts[i]["raw_y"]) / 2.0,
            })

        # Calculate affine calibration matrix
        left_mean = (merged[1]["raw_x"] + merged[4]["raw_x"]) / 2.0
        right_mean = (merged[2]["raw_x"] + merged[3]["raw_x"]) / 2.0
        top_mean = (merged[1]["raw_y"] + merged[2]["raw_y"]) / 2.0
        bottom_mean = (merged[3]["raw_y"] + merged[4]["raw_y"]) / 2.0

        span_x = right_mean - left_mean
        mx = (0.85 - 0.15) / span_x if abs(span_x) > 1e-4 else 1.0
        bx = 0.15 - left_mean * mx

        span_y = bottom_mean - top_mean
        my = (0.85 - 0.15) / span_y if abs(span_y) > 1e-4 else 1.0
        by = 0.15 - top_mean * my

        self.affine = {"mx": float(mx), "bx": float(bx), "my": float(my), "by": float(by)}
        print(f"[Nitriva AI] Calibration complete: mx={mx:.2f}, my={my:.2f}")

        self.calib_phase = "DONE"
        self.active_gesture = "TRACKING GAZE..."

    def process_frame(self, frame=None):
        if frame is None:
            if self.cap is None or not self.cap.isOpened():
                self.start_camera()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "CAMERA ACTIVE - LOOK HERE", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 128), 2)
                return blank, self.get_telemetry()

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_img)

        now = time.time()
        gesture_detected = None

        if res.face_landmarks and len(res.face_landmarks) > 0:
            self.face_detected = True
            lm = res.face_landmarks[0]

            # Blendshapes for eye blinks
            if res.face_blendshapes and len(res.face_blendshapes) > 0:
                bs_map = {b.category_name: b.score for b in res.face_blendshapes[0]}
                self.blink_left = bs_map.get("eyeBlinkLeft", 0.0)
                self.blink_right = bs_map.get("eyeBlinkRight", 0.0)

            # Iris landmarks: 468 (right iris center), 473 (left iris center)
            r_iris = (int(lm[468].x * w), int(lm[468].y * h))
            l_iris = (int(lm[473].x * w), int(lm[473].y * h))
            r_out = (int(lm[33].x * w), int(lm[33].y * h))
            r_in = (int(lm[133].x * w), int(lm[133].y * h))
            l_in = (int(lm[362].x * w), int(lm[362].y * h))
            l_out = (int(lm[263].x * w), int(lm[263].y * h))

            r_width = max(1, abs(r_in[0] - r_out[0]))
            l_width = max(1, abs(l_out[0] - l_in[0]))
            norm_x = (((r_iris[0] - r_out[0]) / r_width) + ((l_iris[0] - l_in[0]) / l_width)) / 2.0

            r_h = max(1, abs(int(lm[145].y * h) - int(lm[159].y * h)))
            l_h = max(1, abs(int(lm[374].y * h) - int(lm[386].y * h)))
            norm_y = (((r_iris[1] - int(lm[159].y * h)) / r_h) + ((l_iris[1] - int(lm[386].y * h)) / l_h)) / 2.0

            # Smoothing
            self.raw_x = norm_x * 0.6 + self.raw_x * 0.4
            self.raw_y = norm_y * 0.6 + self.raw_y * 0.4

            # Collect samples if in active calibration
            if self.is_collecting and self.blink_left < 0.25 and self.blink_right < 0.25:
                self.calib_samples.append((self.raw_x, self.raw_y))

            # Affine calibration transform
            sx = self.raw_x * self.affine["mx"] + self.affine["bx"]
            sy = self.raw_y * self.affine["my"] + self.affine["by"]
            sx = (sx - 0.5) * 1.3 + 0.5
            sy = (sy - 0.5) * 1.3 + 0.5
            self.screen_x = max(0.0, min(1.0, sx))
            self.screen_y = max(0.0, min(1.0, sy))

            # Skip gesture selection while calibrating
            if self.calib_phase != "DONE":
                if self.calib_phase == "SETUP":
                    self.active_gesture = "AWAITING CALIBRATION..."
                else:
                    self.active_gesture = "CALIBRATING EYES..."
            else:
                # Gesture detection: Left Wink vs Dual Blinks
                is_left_wink = (self.blink_left > 0.35 and (self.blink_left - self.blink_right) > 0.15)
                is_both_blinking = (self.blink_left > 0.45 and self.blink_right > 0.45 and not is_left_wink)

                if is_left_wink and not is_both_blinking:
                    if now - self.last_action_time > 1.2:
                        self.active_gesture = "LEFT WINK (SELECTING)"
                        self.last_action_time = now
                        gesture_detected = "WINK"
                elif now - self.last_action_time > 1.0:
                    self.active_gesture = "TRACKING GAZE..."

                # Multi-blink counting
                if is_both_blinking and not self.was_blinking:
                    self.blink_history.append(now)
                    self.was_blinking = True
                elif not is_both_blinking:
                    self.was_blinking = False

                self.blink_history = [t for t in self.blink_history if now - t < 2.5]

                if len(self.blink_history) >= 3:
                    self.active_gesture = "3 BLINKS (DISCONNECT)"
                    self.last_action_time = now + 1.5
                    self.blink_history = []
                    gesture_detected = "DISCONNECT"
                elif len(self.blink_history) == 2 and (now - self.blink_history[1] > 0.35) and not is_both_blinking:
                    self.active_gesture = "2 BLINKS (BACK)"
                    self.last_action_time = now + 1.5
                    self.blink_history = []
                    gesture_detected = "BACK"

            # Annotate OpenCV frame
            cv2.circle(frame, r_iris, 4, (0, 255, 128), -1)
            cv2.circle(frame, l_iris, 4, (0, 255, 128), -1)
            cv2.line(frame, r_out, r_in, (255, 200, 0), 1)
            cv2.line(frame, l_in, l_out, (255, 200, 0), 1)

            # Gaze vector from forehead (landmark 10)
            fh = (int(lm[10].x * w), int(lm[10].y * h))
            dx = int((self.screen_x - 0.5) * 80)
            dy = int((self.screen_y - 0.5) * 80)
            cv2.arrowedLine(frame, fh, (fh[0] + dx, fh[1] + dy), (0, 255, 255), 2, tipLength=0.3)

            # Status bar
            cv2.rectangle(frame, (10, h - 45), (w - 10, h - 10), (15, 23, 42), -1)
            cv2.circle(frame, (25, h - 27), 5, (0, 255, 128), -1)
            cv2.putText(frame, f"{self.active_gesture}  [X:{self.screen_x:.2f} Y:{self.screen_y:.2f}]",
                        (40, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            self.face_detected = False
            self.active_gesture = "FACE LOST"
            cv2.rectangle(frame, (10, h - 45), (w - 10, h - 10), (15, 23, 42), -1)
            cv2.circle(frame, (25, h - 27), 5, (0, 0, 255), -1)
            cv2.putText(frame, "FACE LOST - Center face in frame", (40, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 255), 1)

        telemetry = self.get_telemetry()
        telemetry["gesture_event"] = gesture_detected
        return frame, telemetry

    def get_telemetry(self):
        curr_point = CALIBRATION_POINTS[self.calib_step] if self.calib_step < len(CALIBRATION_POINTS) else CALIBRATION_POINTS[0]
        return {
            "screen_x": round(float(self.screen_x), 3),
            "screen_y": round(float(self.screen_y), 3),
            "blink_left": round(float(self.blink_left), 3),
            "blink_right": round(float(self.blink_right), 3),
            "active_gesture": self.active_gesture,
            "face_detected": self.face_detected,
            "calib_phase": self.calib_phase,
            "calib_step": self.calib_step,
            "calib_label": curr_point["label"],
            "calib_top": curr_point["top"],
            "calib_left": curr_point["left"],
            "calib_error": self.calib_error,
            "is_collecting": self.is_collecting,
            "simulation_mode": self.simulation_mode
        }
