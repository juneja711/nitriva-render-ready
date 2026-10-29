"""
Nitriva Gaze Calibration Engine.
Calibrates eye vector displacements to 2D normalized screen coordinates (0.0 to 1.0)
using a dual-phase (Light Mode + Dark/Black Mode) 5-point calibration protocol.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration_data.json")

# 10 Dual-Phase Calibration Targets: Phase 1 (Light Mode) + Phase 2 (Dark / Black Mode)
CALIBRATION_TARGETS = [
    # Phase 1: Light Mode (Screen White)
    {"id": "light_center", "phase": "light", "name": "1. LOOK CENTER (Phase: Light Mode)", "x": 0.50, "y": 0.50},
    {"id": "light_top_left", "phase": "light", "name": "2. LOOK TOP LEFT (Phase: Light Mode)", "x": 0.12, "y": 0.18},
    {"id": "light_top_right", "phase": "light", "name": "3. LOOK TOP RIGHT (Phase: Light Mode)", "x": 0.88, "y": 0.18},
    {"id": "light_bottom_left", "phase": "light", "name": "4. LOOK BOTTOM LEFT (Phase: Light Mode)", "x": 0.12, "y": 0.82},
    {"id": "light_bottom_right", "phase": "light", "name": "5. LOOK BOTTOM RIGHT (Phase: Light Mode)", "x": 0.88, "y": 0.82},

    # Phase 2: Dark / Black Mode (Screen Black)
    {"id": "dark_center", "phase": "dark", "name": "1. LOOK CENTER (Phase: Dark Mode)", "x": 0.50, "y": 0.50},
    {"id": "dark_top_left", "phase": "dark", "name": "2. LOOK TOP LEFT (Phase: Dark Mode)", "x": 0.12, "y": 0.18},
    {"id": "dark_top_right", "phase": "dark", "name": "3. LOOK TOP RIGHT (Phase: Dark Mode)", "x": 0.88, "y": 0.18},
    {"id": "dark_bottom_left", "phase": "dark", "name": "4. LOOK BOTTOM LEFT (Phase: Dark Mode)", "x": 0.12, "y": 0.82},
    {"id": "dark_bottom_right", "phase": "dark", "name": "5. LOOK BOTTOM RIGHT (Phase: Dark Mode)", "x": 0.88, "y": 0.82},
]


class GazeCalibrator:
    def __init__(self, file_path: str = CALIBRATION_FILE):
        self.file_path = file_path
        self.is_calibrated = False

        # Smart default bounds centered around 0.0
        self.min_gx = -0.22
        self.max_gx = 0.22
        self.min_gy = -0.25
        self.max_gy = 0.25

        self.samples: Dict[str, List[Tuple[float, float]]] = {}
        self.load_calibration()

    def add_sample(self, target_id: str, raw_gx: float, raw_gy: float):
        """Record a raw gaze vector observation at a specific calibration point."""
        if target_id not in self.samples:
            self.samples[target_id] = []
        self.samples[target_id].append((raw_gx, raw_gy))

    def compute_calibration(self) -> bool:
        """
        Fit calibration parameters from recorded samples across Light & Dark phases.
        Maps raw gaze displacements (gx, gy) to screen space [0, 1].
        """
        if not self.samples:
            return False

        # Calculate median gaze values for each calibration point
        point_medians = {}
        for pid, vals in self.samples.items():
            if len(vals) > 0:
                arr = np.array(vals)
                point_medians[pid] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

        def get_pos(name):
            vals = []
            for k in [name, f"light_{name}", f"dark_{name}"]:
                if k in point_medians:
                    vals.append(point_medians[k])
            if vals:
                return (
                    float(np.mean([v[0] for v in vals])),
                    float(np.mean([v[1] for v in vals]))
                )
            return None

        tl = get_pos("top_left")
        tr = get_pos("top_right")
        bl = get_pos("bottom_left")
        br = get_pos("bottom_right")

        left_gx = []
        right_gx = []
        top_gy = []
        bottom_gy = []

        if tl:
            left_gx.append(tl[0])
            top_gy.append(tl[1])
        if bl:
            left_gx.append(bl[0])
            bottom_gy.append(bl[1])
        if tr:
            right_gx.append(tr[0])
            top_gy.append(tr[1])
        if br:
            right_gx.append(br[0])
            bottom_gy.append(br[1])

        if left_gx and right_gx:
            self.min_gx = min(float(np.mean(left_gx)), float(np.mean(right_gx)))
            self.max_gx = max(float(np.mean(left_gx)), float(np.mean(right_gx)))
            if abs(self.max_gx - self.min_gx) < 0.08:
                self.min_gx -= 0.12
                self.max_gx += 0.12

        if top_gy and bottom_gy:
            self.min_gy = min(float(np.mean(top_gy)), float(np.mean(bottom_gy)))
            self.max_gy = max(float(np.mean(top_gy)), float(np.mean(bottom_gy)))
            if abs(self.max_gy - self.min_gy) < 0.08:
                self.min_gy -= 0.12
                self.max_gy += 0.12

        # Center reference anchor (blends light_center and dark_center)
        center = get_pos("center")
        if center:
            cx, cy = center
            half_w = max(abs(cx - self.min_gx), abs(self.max_gx - cx), 0.12)
            self.min_gx = min(self.min_gx, cx - half_w)
            self.max_gx = max(self.max_gx, cx + half_w)
            half_h = max(abs(cy - self.min_gy), abs(self.max_gy - cy), 0.12)
            self.min_gy = min(self.min_gy, cy - half_h)
            self.max_gy = max(self.max_gy, cy + half_h)

        self.is_calibrated = True
        self.save_calibration()
        return True

    def map_gaze(self, raw_gx: float, raw_gy: float) -> Tuple[float, float]:
        """
        Convert raw gaze vector into normalized screen coordinates [0.0, 1.0].
        0,0 is Top-Left, 1,1 is Bottom-Right.
        """
        range_x = self.max_gx - self.min_gx
        range_y = self.max_gy - self.min_gy

        if abs(range_x) < 1e-4:
            range_x = 0.44
        if abs(range_y) < 1e-4:
            range_y = 0.50

        # Invert horizontal axis if needed (camera mirroring)
        screen_x = (raw_gx - self.min_gx) / range_x
        # Invert vertical axis: up in eyes = smaller y (top of screen)
        screen_y = 1.0 - ((raw_gy - self.min_gy) / range_y)

        # Clamp smoothly to [0.05, 0.95]
        screen_x = float(np.clip(screen_x, 0.05, 0.95))
        screen_y = float(np.clip(screen_y, 0.05, 0.95))

        return screen_x, screen_y

    def save_calibration(self):
        """Save calibration matrix to JSON file."""
        data = {
            "is_calibrated": self.is_calibrated,
            "min_gx": self.min_gx,
            "max_gx": self.max_gx,
            "min_gy": self.min_gy,
            "max_gy": self.max_gy,
        }
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Calibrator] Error saving calibration: {e}")

    def load_calibration(self):
        """Load calibration if existing file is present."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.is_calibrated = data.get("is_calibrated", False)
                    self.min_gx = data.get("min_gx", -0.22)
                    self.max_gx = data.get("max_gx", 0.22)
                    self.min_gy = data.get("min_gy", -0.25)
                    self.max_gy = data.get("max_gy", 0.25)
            except Exception as e:
                print(f"[Calibrator] Error loading calibration: {e}")


calibrator = GazeCalibrator()
