"""
Nitriva Eye-Tracking & Blink/Wink ML Engine.
Uses MediaPipe FaceLandmarker with Geometric Iris Displacement,
Blendshape Fusion, and Adaptive Deadband Smoothing for rock-solid stability.
"""

import os
import time
import math
import threading
import base64
from typing import Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.calibrator import calibrator


class EyeTracker:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.capture_thread = None

        # Model setup
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.model_path = os.path.join(base_dir, "models", "face_landmarker.task")
        self.detector = None
        self._init_detector()

        # Telemetry state
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.raw_gx = 0.0
        self.raw_gy = 0.0

        self.blink_left = 0.0
        self.blink_right = 0.0
        self.is_blinking = False
        self.blink_count = 0
        self.blink_state_text = "TRACKING ACTIVE"
        self.recent_event: Optional[str] = None
        self.event_timestamp: float = 0.0

        self.wink_left_active = False
        self.wink_right_active = False
        self.wink_left_start_time = 0.0
        self.wink_right_start_time = 0.0
        self.last_event_time = 0.0

        # Multi-blink tracking
        self.blink_in_progress = False
        self.blink_history = []
        self.last_blink_time = 0.0

        # Adaptive Deadband Smoothing State
        self.target_gx = 0.5
        self.target_gy = 0.5
        self.fps = 0.0
        self.prev_frame_time = time.time()
        self.face_detected = False

        # Annotated frame for Optics Sensor
        self.current_frame = None
        self.simulation_mode = False
        self.simulated_gaze: Optional[Tuple[float, float]] = None

    def _init_detector(self):
        """Initialize MediaPipe FaceLandmarker with blendshapes."""
        if not os.path.exists(self.model_path):
            print(f"[Tracker] Model not found at {self.model_path}")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("[Tracker] MediaPipe FaceLandmarker initialized successfully.")
        except Exception as e:
            print(f"[Tracker] Error initializing FaceLandmarker: {e}")

    def start_camera(self):
        """Start the webcam capture thread (supports local camera and cloud headless mode)."""
        with self.lock:
            if self.is_running:
                return

            mode = os.environ.get("NITRIVA_CAMERA_MODE", "auto").lower()
            if mode not in {"auto", "browser", "local"}:
                raise ValueError("NITRIVA_CAMERA_MODE must be auto, browser, or local")
            if mode == "browser":
                self.simulation_mode = True
                self.is_running = True
                print("[Tracker] Browser webcam mode; no server camera or synthetic loop.")
                return

            self.simulation_mode = False
            try:
                if os.name == "nt":
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index)

                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_index)

                if not self.cap or not self.cap.isOpened():
                    print(f"[Tracker] Local hardware camera not found (normal in Cloud Deployment). Client browser webcam mode active.")
                    self.simulation_mode = True
                else:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception as e:
                print(f"[Tracker] Camera initialization note: {e}")
                self.simulation_mode = True

            self.is_running = True
            # Synthetic frames would overwrite browser gaze and waste idle CPU.
            if self.simulation_mode:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

    def process_client_frame(self, frame_bytes_or_b64: Union[bytes, str]) -> Dict[str, Any]:
        # Serialize access to MediaPipe and the gaze smoothing state.
        with self.inference_lock:
            return self._process_client_frame(frame_bytes_or_b64)

    def _process_client_frame(self, frame_bytes_or_b64: Union[bytes, str]) -> Dict[str, Any]:
        """
        Process a video frame received from a client browser (used in Cloud Deployment).
        Supports raw JPEG/PNG bytes or base64 data URLs.
        """
        try:
            if isinstance(frame_bytes_or_b64, str):
                if "," in frame_bytes_or_b64:
                    frame_bytes_or_b64 = frame_bytes_or_b64.split(",", 1)[1]
                frame_bytes = base64.b64decode(frame_bytes_or_b64)
            else:
                frame_bytes = frame_bytes_or_b64

            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return self.get_telemetry()

            # Flip horizontally to match mirror preview
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - self.prev_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.prev_frame_time = now

            if self.detector is not None:
                self._process_frame_ml(frame, now)

            # Update Optics Sensor current frame with reticle overlay
            self._render_optics_overlay(frame)
            with self.lock:
                self.current_frame = frame

            return self.get_telemetry()
        except Exception:
            return self.get_telemetry()

    def stop_camera(self):
        """Stop camera capture."""
        with self.lock:
            self.is_running = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
            self.capture_thread = None

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Fallback synthetic camera stream showing face silhouette with animated pupils."""
        frame = np.full((480, 640, 3), 28, dtype=np.uint8)
        t = time.time()

        center_x = 320
        center_y = 240
        cv2.ellipse(frame, (center_x, center_y), (110, 150), 0, 0, 360, (50, 50, 60), -1)
        cv2.ellipse(frame, (center_x, center_y), (110, 150), 0, 0, 360, (80, 100, 120), 2)

        eye_y = center_y - 20
        left_eye_x = center_x - 45
        right_eye_x = center_x + 45

        sim_blink = (math.sin(t * 1.5) > 0.95) or self.is_blinking
        left_h = 2 if sim_blink else 10
        right_h = 2 if sim_blink else 10

        pupil_dx = int(math.sin(t * 0.8) * 12)
        pupil_dy = int(math.cos(t * 0.6) * 6)

        cv2.ellipse(frame, (left_eye_x, eye_y), (18, left_h), 0, 0, 360, (220, 220, 230), -1)
        cv2.ellipse(frame, (right_eye_x, eye_y), (18, right_h), 0, 0, 360, (220, 220, 230), -1)

        if not sim_blink:
            cv2.circle(frame, (left_eye_x + pupil_dx, eye_y + pupil_dy), 6, (34, 197, 94), -1)
            cv2.circle(frame, (right_eye_x + pupil_dx, eye_y + pupil_dy), 6, (34, 197, 94), -1)

        if self.simulated_gaze:
            self.gaze_x, self.gaze_y = self.simulated_gaze
        else:
            self.gaze_x = 0.5 + 0.35 * math.sin(t * 0.8)
            self.gaze_y = 0.5 + 0.25 * math.cos(t * 0.6)

        self.face_detected = True
        return frame

    def _capture_loop(self):
        """Continuous frame processing loop."""
        while self.is_running:
            raw_frame = None
            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if not ret:
                    raw_frame = None

            if raw_frame is None:
                raw_frame = self._generate_synthetic_frame()
                use_synthetic = True
            else:
                raw_frame = cv2.flip(raw_frame, 1)
                use_synthetic = False

            now = time.time()
            dt = now - self.prev_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.prev_frame_time = now

            if not use_synthetic and self.detector is not None:
                self._process_frame_ml(raw_frame, now)

            # Draw Optics Sensor overlays
            self._render_optics_overlay(raw_frame)

            with self.lock:
                self.current_frame = raw_frame.copy()

            time.sleep(0.016)

    def _process_frame_ml(self, frame: np.ndarray, now: Optional[float] = None):
        """Run MediaPipe inference and extract eye metrics with anti-jitter deadband."""
        if now is None:
            now = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            result = self.detector.detect(mp_image)
        except Exception:
            return

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            self.face_detected = False
            return

        self.face_detected = True
        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        # Extract blendshapes
        shapes = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for b in result.face_blendshapes[0]:
                shapes[b.category_name] = b.score

        self.blink_left = shapes.get("eyeBlinkLeft", 0.0)
        self.blink_right = shapes.get("eyeBlinkRight", 0.0)

        look_in_l = shapes.get("eyeLookInLeft", 0.0)
        look_out_l = shapes.get("eyeLookOutLeft", 0.0)
        look_in_r = shapes.get("eyeLookInRight", 0.0)
        look_out_r = shapes.get("eyeLookOutRight", 0.0)
        look_up_l = shapes.get("eyeLookUpLeft", 0.0)
        look_up_r = shapes.get("eyeLookUpRight", 0.0)
        look_down_l = shapes.get("eyeLookDownLeft", 0.0)
        look_down_r = shapes.get("eyeLookDownRight", 0.0)

        # Blendshapes horizontal & vertical vectors
        blend_gx = (look_out_l - look_in_l + look_in_r - look_out_r) / 2.0
        blend_gy = ((look_up_l + look_up_r) - (look_down_l + look_down_r)) / 2.0

        # Geometric Iris landmark extraction
        if len(landmarks) > 473:
            left_iris = landmarks[468]
            right_iris = landmarks[473]

            # Eye bounding landmarks
            l_outer = landmarks[33]
            l_inner = landmarks[133]
            r_inner = landmarks[362]
            r_outer = landmarks[263]

            l_width = abs(l_inner.x - l_outer.x) + 1e-6
            r_width = abs(r_outer.x - r_inner.x) + 1e-6

            # Safe horizontal displacement from eye center
            l_cx = (l_outer.x + l_inner.x) / 2.0
            r_cx = (r_inner.x + r_outer.x) / 2.0
            iris_gx_l = (left_iris.x - l_cx) / l_width
            iris_gx_r = (right_iris.x - r_cx) / r_width

            # Vertical displacement from eye center with safe denominator clamp (prevents explode on wink/blink)
            l_top = landmarks[159]
            l_bot = landmarks[145]
            r_top = landmarks[386]
            r_bot = landmarks[374]

            l_h = max(abs(l_bot.y - l_top.y), 0.012)
            r_h = max(abs(r_bot.y - r_top.y), 0.012)
            l_cy = (l_top.y + l_bot.y) / 2.0
            r_cy = (r_top.y + r_bot.y) / 2.0
            iris_gy_l = (left_iris.y - l_cy) / l_h
            iris_gy_r = (right_iris.y - r_cy) / r_h

            # Wink-robust eye fusion:
            # If an eye is closed (winking), use ONLY the open eye!
            is_l_closed = self.blink_left > 0.35
            is_r_closed = self.blink_right > 0.35

            if is_l_closed and not is_r_closed:
                # Left eye is winking: track exclusively with open right eye!
                iris_gx = iris_gx_r
                iris_gy = iris_gy_r
                eff_blend_gx = look_in_r - look_out_r
                eff_blend_gy = look_up_r - look_down_r
            elif is_r_closed and not is_l_closed:
                # Right eye is winking: track exclusively with open left eye!
                iris_gx = iris_gx_l
                iris_gy = iris_gy_l
                eff_blend_gx = look_out_l - look_in_l
                eff_blend_gy = look_up_l - look_down_l
            elif is_l_closed and is_r_closed:
                # Both eyes closed (blink): FREEZE gaze coordinates completely!
                iris_gx = None
                iris_gy = None
                eff_blend_gx = blend_gx
                eff_blend_gy = blend_gy
            else:
                # Both eyes open: average both
                iris_gx = (iris_gx_l + iris_gx_r) / 2.0
                iris_gy = (iris_gy_l + iris_gy_r) / 2.0
                eff_blend_gx = blend_gx
                eff_blend_gy = blend_gy

            if iris_gx is not None and iris_gy is not None:
                # Compensate resting screen vertical gaze (~ -0.46)
                self.raw_gx = float(0.65 * iris_gx + 0.35 * eff_blend_gx)
                self.raw_gy = float(0.65 * (iris_gy + 0.46) + 0.35 * (eff_blend_gy + 0.60))

        # Map to calibrated screen space
        target_sx, target_sy = calibrator.map_gaze(self.raw_gx, self.raw_gy)

        # Do NOT drift or jerk gaze coordinates while user is winking or blinking
        is_winking_or_blinking = (self.blink_left > 0.35 or self.blink_right > 0.35)
        if not is_winking_or_blinking:
            # Adaptive Deadband & Fixation Filter (Prevents jitter)
            dx = abs(target_sx - self.gaze_x)
            dy = abs(target_sy - self.gaze_y)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.045:
                # Fixation zone: clamp micro-jitter
                smooth_factor = 0.06
            elif dist < 0.12:
                # Moderate drift
                smooth_factor = 0.20
            else:
                # Saccade: rapid responsive movement
                smooth_factor = 0.48

            self.gaze_x = float(smooth_factor * target_sx + (1.0 - smooth_factor) * self.gaze_x)
            self.gaze_y = float(smooth_factor * target_sy + (1.0 - smooth_factor) * self.gaze_y)

        # Process Blinks & Winks state machine
        self._update_eye_state_machine(now)

    def _update_eye_state_machine(self, now: float):
        """Multi-blink and Left/Right wink detection with differential thresholding."""
        diff_lr = self.blink_left - self.blink_right
        diff_rl = self.blink_right - self.blink_left

        both_closed = (self.blink_left >= 0.40 and self.blink_right >= 0.40 and abs(diff_lr) < 0.20)
        left_wink = (self.blink_left >= 0.36 and diff_lr >= 0.15 and self.blink_right < 0.48)
        right_wink = (self.blink_right >= 0.36 and diff_rl >= 0.15 and self.blink_left < 0.48)

        # Multi-blink counting
        if both_closed:
            if not self.blink_in_progress:
                self.blink_in_progress = True
        else:
            if self.blink_in_progress:
                self.blink_in_progress = False
                self.blink_history.append(now)
                self.last_blink_time = now

        # Prune blinks outside 800ms window
        self.blink_history = [t for t in self.blink_history if now - t < 0.80]
        count = len(self.blink_history)

        if count == 1:
            self.blink_state_text = "1 BLINK..."
        elif count == 2:
            self.blink_state_text = "2 BLINKS (BACK)"
            if now - self.last_event_time > 0.40:
                self._trigger_event("DOUBLE_BLINK", "2 BLINKS (BACK)")
                self.last_event_time = now
            self.blink_history.clear()
        elif count >= 3:
            self.blink_state_text = "3 BLINKS = LOGOUT"
            if now - self.last_event_time > 0.40:
                self._trigger_event("TRIPLE_BLINK", "3 BLINKS = LOGOUT")
                self.last_event_time = now
            self.blink_history.clear()
        else:
            if now - self.event_timestamp > 1.2:
                self.blink_state_text = "TRACKING ACTIVE"

        # Left Wink Detection (Select)
        if left_wink and (now - self.last_event_time > 0.38):
            if self.wink_left_start_time == 0.0:
                self.wink_left_start_time = now
            elif (now - self.wink_left_start_time >= 0.08) and not self.wink_left_active:
                self.wink_left_active = True
                self._trigger_event("WINK_LEFT", "WINK LEFT (SELECT)")
                self.last_event_time = now
        elif self.blink_left < 0.28:
            self.wink_left_start_time = 0.0
            self.wink_left_active = False

        # Right Wink Detection (Clear / Alternate Select)
        if right_wink and (now - self.last_event_time > 0.38):
            if self.wink_right_start_time == 0.0:
                self.wink_right_start_time = now
            elif (now - self.wink_right_start_time >= 0.08) and not self.wink_right_active:
                self.wink_right_active = True
                self._trigger_event("WINK_RIGHT", "WINK RIGHT (CLEAR)")
                self.last_event_time = now
        elif self.blink_right < 0.28:
            self.wink_right_start_time = 0.0
            self.wink_right_active = False

    def _trigger_event(self, event_code: str, label: str):
        """Register a user event for telemetry broadcast."""
        self.recent_event = event_code
        self.blink_state_text = label
        self.event_timestamp = time.time()
        print(f"[Tracker] Event Triggered: {event_code} -> {label}")

    def trigger_simulated_action(self, action: str):
        """Simulate an action from keyboard shortcut or test harness."""
        if action == "WINK_LEFT":
            self._trigger_event("WINK_LEFT", "WINK LEFT (SELECT)")
        elif action == "WINK_RIGHT":
            self._trigger_event("WINK_RIGHT", "WINK RIGHT (CLEAR)")
        elif action == "DOUBLE_BLINK":
            self._trigger_event("DOUBLE_BLINK", "2 BLINKS (BACK)")
        elif action == "TRIPLE_BLINK":
            self._trigger_event("TRIPLE_BLINK", "3 BLINKS = LOGOUT")
        elif action == "BLINK":
            self._trigger_event("BLINK", "1 BLINK...")

    def _render_optics_overlay(self, frame: np.ndarray):
        """Render stylish modern Optics Sensor HUD matching Nayan Vaani."""
        h, w, _ = frame.shape

        box_w = int(w * 0.55)
        box_h = int(h * 0.70)
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        box_color = (34, 197, 94) if self.face_detected else (70, 70, 80)
        corner_len = 24
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 2)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 2)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 2)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 2)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 2)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 2)

        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """Fetch current telemetry payload for WebSocket transmission."""
        now = time.time()
        event_to_send = None
        if self.recent_event and (now - self.event_timestamp < 0.35):
            event_to_send = self.recent_event
            self.recent_event = None  # Consume event immediately so each physical wink is sent once!

        return {
            "gaze_x": round(self.gaze_x, 4),
            "gaze_y": round(self.gaze_y, 4),
            "blink_left": round(self.blink_left, 3),
            "blink_right": round(self.blink_right, 3),
            "state_text": self.blink_state_text,
            "event": event_to_send,
            "face_detected": self.face_detected,
            "fps": round(self.fps, 1),
            "is_calibrated": calibrator.is_calibrated,
        }

    def generate_mjpeg(self):
        """Generator yielding MJPEG video frames for HTTP streaming."""
        while True:
            with self.lock:
                if self.current_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_bytes = buffer.tobytes() if ret else None
                else:
                    frame_bytes = None

            if frame_bytes:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            time.sleep(0.033)


tracker = EyeTracker(camera_index=0)
