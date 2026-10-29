import cv2
import numpy as np
import time
from optics_engine import NitrivaOpticsEngine
from sentence_generator import NitrivaSentenceSynthesizer
from tts_engine import NitrivaVoiceEngine

def run_desktop_app():
    """
    Runs Nitriva as a pure Python native OpenCV application with an interactive HUD.
    Features:
      - Live Face Mesh and Iris Landmarks display
      - 4 Interactive Quadrants (Speech Generator, Device Control, SOS, Preferences)
      - Left Wink selects quadrant
      - 2 Blinks navigates back
      - Press 'q' to quit
    """
    print("Starting Nitriva Native Python Desktop HUD...")
    engine = NitrivaOpticsEngine(camera_index=0)
    synthesizer = NitrivaSentenceSynthesizer()
    voice = NitrivaVoiceEngine()

    engine.start_camera()
    voice.speak("Nitriva Python Desktop AI HUD initialized.")

    current_view = "DASHBOARD"  # DASHBOARD, COMM, CONTROL, SOS
    selected_comm_words = []
    app_devices = {"Light": False, "AC": False, "TV": False}

    cv2.namedWindow("Nitriva - Python AI/ML Optics HUD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Nitriva - Python AI/ML Optics HUD", 1024, 768)

    while True:
        frame, telemetry = engine.process_frame()
        h, w, _ = frame.shape
        hud = frame.copy()

        sx = telemetry["screen_x"]
        sy = telemetry["screen_y"]
        gesture = telemetry.get("gesture_event")

        # Determine gaze quadrant (0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right)
        quadrant = 0
        if sx >= 0.5 and sy < 0.5:
            quadrant = 1
        elif sx < 0.5 and sy >= 0.5:
            quadrant = 2
        elif sx >= 0.5 and sy >= 0.5:
            quadrant = 3

        # Overlay UI based on current view
        if current_view == "DASHBOARD":
            # 4 Quadrants
            cards = [
                ("1. Speech Generator", (0, 0, w//2, h//2), (255, 140, 0)),
                ("2. Device Control", (w//2, 0, w, h//2), (255, 200, 0)),
                ("3. Emergency SOS", (0, h//2, w//2, h), (0, 0, 220)),
                ("4. System Prefs", (w//2, h//2, w, h), (180, 180, 180))
            ]

            for i, (title, (x1, y1, x2, y2), color) in enumerate(cards):
                is_active = (quadrant == i)
                alpha = 0.4 if is_active else 0.15
                overlay = hud.copy()
                cv2.rectangle(overlay, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), color, -1)
                cv2.addWeighted(overlay, alpha, hud, 1 - alpha, 0, hud)
                thickness = 3 if is_active else 1
                cv2.rectangle(hud, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), color, thickness)
                cv2.putText(hud, title, (x1 + 25, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            if gesture == "WINK":
                if quadrant == 0:
                    current_view = "COMM"
                    voice.speak("Communication Board")
                elif quadrant == 1:
                    current_view = "CONTROL"
                    voice.speak("Device Control")
                elif quadrant == 2:
                    current_view = "SOS"
                    voice.speak("Emergency SOS Triggered.")
                elif quadrant == 3:
                    voice.speak("System Preferences")

        elif current_view == "COMM":
            cv2.putText(hud, "COMMUNICATION BOARD - Wink to Speak", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(hud, "2 Blinks: Back", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            words = ["Water", "Pain", "Washroom", "Family"]
            for i, w_text in enumerate(words):
                y_pos = 100 + i * 80
                is_sel = (quadrant == i)
                cv2.rectangle(hud, (50, y_pos), (w - 50, y_pos + 60), (0, 255, 128) if is_sel else (100, 100, 100), 2)
                cv2.putText(hud, w_text, (70, y_pos + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if gesture == "WINK":
                word = words[quadrant]
                sentence = synthesizer.generate([word])
                voice.speak(sentence)

        elif current_view == "CONTROL":
            cv2.putText(hud, "DEVICE CONTROL - Wink to Toggle", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(hud, "2 Blinks: Back", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            devices = list(app_devices.keys())
            for i, d in enumerate(devices):
                y_pos = 100 + i * 90
                is_sel = (quadrant == i)
                is_on = app_devices[d]
                color = (0, 255, 0) if is_on else (80, 80, 80)
                cv2.rectangle(hud, (50, y_pos), (w - 50, y_pos + 70), (0, 255, 255) if is_sel else color, 2)
                cv2.putText(hud, f"{d}: {'ON' if is_on else 'OFF'}", (70, y_pos + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if gesture == "WINK" and quadrant < len(devices):
                target = devices[quadrant]
                app_devices[target] = not app_devices[target]
                voice.speak(f"{target} {'turned on' if app_devices[target] else 'turned off'}")

        elif current_view == "SOS":
            hud[:] = (0, 0, 180)  # Flashing red background
            cv2.putText(hud, "EMERGENCY SOS TRIGGERED", (w//2 - 260, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(hud, "Caregiver alert dispatched.", (w//2 - 180, h//2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

        if gesture == "BACK" and current_view != "DASHBOARD":
            current_view = "DASHBOARD"
            voice.speak("Returned to Dashboard.")

        # Gaze Cursor circle
        cursor_x = int(sx * w)
        cursor_y = int(sy * h)
        cv2.circle(hud, (cursor_x, cursor_y), 12, (0, 255, 255), -1)
        cv2.circle(hud, (cursor_x, cursor_y), 16, (0, 200, 200), 2)

        cv2.imshow("Nitriva - Python AI/ML Optics HUD", hud)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    engine.stop_camera()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_desktop_app()
